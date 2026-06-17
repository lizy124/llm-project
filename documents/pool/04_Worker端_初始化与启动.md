# 第 4 章：Worker 端 - 初始化与启动

Worker 端的核心是 [`KVPoolWorker`](../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py)，它负责实际的 KV Cache 存取操作。

## 4.1 KVPoolWorker 初始化

```python
class KVPoolWorker:
    def __init__(self, vllm_config, use_layerwise, kv_cache_config=None):
        # ====== 读取并行配置 ======
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size
        self.pcp_size = get_pcp_group().world_size
        self.dcp_size = get_decode_context_model_parallel_world_size()
        
        # ====== KV 角色 ======
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        # "kv_producer" / "kv_consumer" / "kv_both"
        
        # ====== 后端选择 ======
        self.backend = ...  # "mooncake" / "memcache" / "yuanrong"
        self.use_hybrid = self._uses_hybrid_kv_cache(...)
        self.use_mamba = self._uses_mamba_kv_cache(...)
        
        # ====== Block 大小计算 ======
        self.original_block_size = self._infer_group_block_sizes(...)
        # 例如: [128] (单组) 或 [128, 8, 32] (DeepSeek V4 的多组)
        
        cp_scale = self.pcp_size * self.dcp_size
        self.grouped_block_size = [bs * cp_scale for bs in self.original_block_size]
        self.hash_block_size = ...
        self.block_size = self.grouped_block_size[0]
        self.lcm_block_size = math.lcm(*self.grouped_block_size)
        
        # ====== KV Head 配置 ======
        self.num_kv_head = model_config.get_total_num_kv_heads()
        # MLA 模式下 num_kv_head = 1
        
        # head_or_tp_rank 决定 TP 分片
        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1
        
        # ====== 创建 KeyMetadata ======
        for group_id in range(self.num_kv_cache_groups):
            self.metadata.append(KeyMetadata(
                model_name=...,
                head_or_tp_rank=group_tp_rank,
                pcp_rank=self.pcp_rank,
                dcp_rank=self.dcp_rank,
                pp_rank=self.pp_rank,
                kv_cache_group_id=group_id,
            ))
        
        # ====== 创建 ChunkedTokenDatabase ======
        self.token_database = ChunkedTokenDatabase(
            self.metadata, self.grouped_block_size, partitions, self.use_hybrid, self.hash_block_size
        )
        
        # ====== ★ 创建 MooncakeBackend ======
        backend_module = importlib.import_module(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend"
        )
        self.m_store = MooncakeBackend(parallel_config, lazy_init=...)
        
        # ====== 后台传输线程（初始为 None，在 register_kv_caches 中创建）======
        self.kv_send_thread = None
        self.kv_recv_thread = None
```

### 4.1.1 后端选择映射

```python
backend_map = {
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "...mooncake_backend",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "...memcache_backend",
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "...yuanrong_backend",
    },
}
```

## 4.2 MooncakeBackend 初始化

[`MooncakeBackend`](../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py) 是 Mooncake 分布式存储的后端实现。

### 4.2.1 构造函数

```python
class MooncakeBackend(Backend):
    def __init__(self, parallel_config, lazy_init=False):
        # 1. 从环境变量 MOONCAKE_CONFIG_PATH 指定的 JSON 文件加载配置
        self.config = MooncakeStoreConfig.load_from_env()
        
        # 2. 检查协议（必须是 "ascend"）
        if self.config.protocol != "ascend":
            raise NotImplementedError(...)
        
        # 3. 是否使用 Fabric Memory（统一内存地址直传，仅 800I/T A3 系列）
        self._use_fabric_mem = os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
        
        # 4. 延迟初始化（仅 fabric_mem + 压缩模型场景）
        self._lazy_init = lazy_init and self._use_fabric_mem
        
        # 5. 非延迟初始化时，立即创建 store
        if not self._lazy_init:
            self.store = self._setup_store()
            self._store_initialized = True
```

### 4.2.2 _setup_store：创建 Mooncake Distributed Store

```python
def _setup_store(self):
    from mooncake.store import MooncakeDistributedStore
    
    store = MooncakeDistributedStore()
    local_hostname = get_ip()
    
    if not self._use_fabric_mem:
        # 标准路径：获取 TransferEngine 并注册
        transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
        self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
        
        ret = store.setup(
            local_hostname=self.local_seg,
            metadata_server=self.config.metadata_server,  # 例如 redis://127.0.0.1:6379
            global_segment_size=self.config.global_segment_size,  # 例如 1GB
            local_buffer_size=self.config.local_buffer_size,
            protocol=self.config.protocol,  # "ascend"
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
            engine=transfer_engine.get_engine(),
            **ssd_kwargs,  # 可选的 SSD offload
        )
    else:
        # Fabric Memory 路径：不需要 TransferEngine
        self.local_seg = local_hostname
        ret = store.setup(
            local_hostname=self.local_seg,
            metadata_server=self.config.metadata_server,
            global_segment_size=self.config.global_segment_size,
            local_buffer_size=0,  # FM 不需要 local buffer
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
            **ssd_kwargs,
        )
    
    if ret != 0:
        raise RuntimeError("Initialize mooncake failed.")
    
    return store
```

### 4.2.3 MooncakeStoreConfig 配置

从 JSON 文件（由 `MOONCAKE_CONFIG_PATH` 环境变量指定）读取：

```json
{
    "metadata_server": "redis://127.0.0.1:6379",
    "global_segment_size": "2GB",
    "local_buffer_size": "1GB",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "",
    "preferred_segment": false,
    "prefer_alloc_in_same_node": true,
    "enable_ssd_offload": false,
    "ssd_offload_path": ""
}
```

`global_segment_size` 和 `local_buffer_size` 在 JSON 中可以写成 `"2GB"` / `"1GB"` 这样的字符串，加载后会通过 `_parse_global_segment_size()` 转换为字节数整数。

## 4.3 KV Cache 注册：register_kv_caches

这是 Worker 端初始化过程中最关键的一步，它连接了 NPU 显存中的 KV Cache 张量和外部池。

```python
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    # 1. 获取 num_blocks
    self.num_blocks = self.kv_cache_config.num_blocks
    self.kv_caches = kv_caches  # {layer_name: (k_cache, v_cache), ...}
    
    # 2. 为每个 KV Cache 组计算元数据
    for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
        self._infer_cache_group_metadata(group_id, group_spec.layer_names)
    # 结果存储在：
    #   self.group_kv_caches_base_addr[group_id] = [base_addr1, base_addr2, ...]
    #   self.group_block_len[group_id] = [block_len1, block_len2, ...]
    #   self.group_block_stride[group_id] = [block_stride1, block_stride2, ...]
    #   self.group_num_layers[group_id] = len(layer_names)
    
    # 3. 注册 buffer 到 Backend
    registered_regions = {}  # storage_key → (start, end)
    for cache_or_caches in kv_caches.values():
        for cache in self._as_cache_tuple(cache_or_caches):
            base_addr = cache.data_ptr()
            region_len = ...
            storage_key = self._get_storage_key(cache)
            # 合并同一 storage 的连续区域
            registered_regions[storage_key] = (min(start, old_start), max(end, old_end))
    
    ptrs = [start for start, _ in registered_regions.values()]
    lengths = [end - start for start, end in registered_regions.values()]
    
    # ★ 向 Mooncake Backend 注册内存区域
    self.m_store.register_buffer(ptrs, lengths)
    
    # 4. 设置 TokenDatabase 的 buffer 信息
    self.token_database.set_group_buffers(
        self.group_kv_caches_base_addr,
        self.group_block_len,
        self.group_block_stride,
        cache_role="kv",
        group_cache_families=self.group_kv_cache_families,
        group_num_layers=self.group_num_layers,
    )
    
    # 5. ★ 启动后台传输线程
    if self.use_layerwise:
        self._start_layerwise_threads()
    else:
        self._start_standard_threads()
```

### 4.3.1 _infer_cache_group_metadata 详解

```python
def _infer_cache_group_metadata(self, group_id, layer_names):
    group_addrs = []
    group_block_lens = []
    group_block_strides = []
    
    for layer_name in layer_names:
        cache_or_caches = self.kv_caches[layer_name]  # (k_cache, v_cache)
        for cache in self._as_cache_tuple(cache_or_caches):
            base_addr = cache.data_ptr()
            # 计算 block_len 和 block_stride
            block_len, block_stride, _, _ = self._get_cache_block_metadata(cache)
            group_addrs.append(base_addr)
            group_block_lens.append(block_len)
            group_block_strides.append(block_stride)
    
    self.group_kv_caches_base_addr[group_id] = group_addrs
    self.group_block_len[group_id] = group_block_lens
    self.group_block_stride[group_id] = group_block_strides
    self.group_num_layers[group_id] = len(layer_names)
```

### 4.3.2 MooncakeBackend.register_buffer

```python
def register_buffer(self, ptrs: list[int], lengths: list[int]):
    if not self._use_fabric_mem:
        local_hostname = get_ip()
        # 确保 TransferEngine 已初始化
        global_te.get_transfer_engine(local_hostname, device_name=None)
        # ★ 向 TransferEngine 注册内存
        global_te.register_buffer(ptrs, lengths)
```

### 4.3.3 GlobalTE.register_buffer

```python
class GlobalTE:
    def register_buffer(self, ptrs, sizes):
        with self.register_buffer_lock:
            if self.is_register_buffer:
                return  # 只注册一次
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.is_register_buffer = True
```

## 4.4 启动后台传输线程

### 4.4.1 标准模式（非 Layerwise）

```python
# Sending 线程（kv_producer / kv_both / consumer_is_to_put）
if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put:
    self.kv_send_thread = KVCacheStoreSendingThread(
        self.m_store,       # MooncakeBackend
        self.token_database, # ChunkedTokenDatabase
        self.grouped_block_size,
        self.tp_rank,
        self.dcp_size,
        self.put_step,
        self.kv_role,
        ready_event_sending,
        self.group_uses_align_state,
        self.enable_kv_events,
    )
    self.kv_send_thread.start()
    # 当前实现创建了 ready_event_sending，但不会等待 ready_event_sending.wait()

# Recving 线程（load_async 模式）
if self.load_async:
    self.kv_recv_thread = KVCacheStoreRecvingThread(
        self.m_store,
        self.token_database,
        self.grouped_block_size,
        self.tp_rank,
        self.dcp_size,
        ready_event,
        self._invalid_block_ids,
        self._invalid_block_ids_lock,
    )
    self.kv_recv_thread.start()
    ready_event.wait()  # Recving 线程会等待线程就绪
```

### 4.4.2 Layerwise 模式

```python
if self.kv_role in ["kv_producer", "kv_both"]:
    self.kv_send_thread = KVCacheStoreLayerSendingThread(
        self.m_store, self.token_database, self.grouped_block_size,
        self.tp_rank, self.dcp_size, self.put_step,
        ready_event_sending, self.num_layers, self.enable_kv_events,
    )
    self.kv_send_thread.start()

self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
    self.m_store, self.token_database, self.grouped_block_size,
    self.tp_rank, self.dcp_size,
    ready_event, self.get_event,
    self._invalid_block_ids, self._invalid_block_ids_lock,
)
self.kv_recv_thread.start()
ready_event.wait()
```

## 4.5 LookupKeyServer 启动

仅在 rank 0 上启动，作为 Scheduler 端 `LookupKeyClient` 的 ZMQ 服务端。

```python
class LookupKeyServer:
    def __init__(self, pool_worker, vllm_config, use_layerwise):
        # 创建 ZMQ REP socket
        self.socket = make_zmq_socket(ctx, socket_path, zmq.REP, bind=True)
        
        # 启动后台线程处理请求
        def process_request():
            while self.running:
                all_frames = self.socket.recv_multipart(copy=False)
                token_len = int.from_bytes(all_frames[0], byteorder="big")
                kv_group_ids = self.decoder.decode([all_frames[1]])
                hash_frames = all_frames[2:]
                hashes_str = self.decoder.decode(hash_frames)
                
                # ★ 调用 pool_worker.lookup_scheduler()
                result = self.pool_worker.lookup_scheduler(
                    token_len, hashes_str, kv_group_ids, self.use_layerwise,
                )
                
                # 返回 hit 的 token 数量
                response = result.to_bytes(4, "big")
                self.socket.send(response)
        
        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()
```

## 4.6 Worker 端初始化时序图

```
KVPoolWorker.__init__()
    │
    ├── 读取并行配置 (TP/PP/PCP/DCP)
    ├── 计算 block 大小
    ├── 创建 KeyMetadata
    ├── 创建 ChunkedTokenDatabase
    └── 创建 MooncakeBackend
            │
            ├── 加载 mooncake.json 配置
            ├── 初始化 TransferEngine（全局单例）
            └── 创建 MooncakeDistributedStore.setup()
    
register_kv_caches(kv_caches)
    │
    ├── 遍历所有 KV Cache 张量，收集 base_addr/block_len/block_stride
    ├── MooncakeBackend.register_buffer(ptrs, lengths)
    │       └── TransferEngine.register_memory(ptr, size)
    ├── ChunkedTokenDatabase.set_group_buffers(...)
    └── 启动后台线程
            ├── KVCacheStoreSendingThread.start()  (kv_producer)
            └── KVCacheStoreRecvingThread.start()  (load_async)

LookupKeyServer.__init__()  (仅 rank 0)
    └── 启动 ZMQ REP 服务线程
```