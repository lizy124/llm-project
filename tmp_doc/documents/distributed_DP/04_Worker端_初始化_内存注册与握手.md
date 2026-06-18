# 第 4 章：Worker 端 - 初始化、内存注册与握手

Worker 端核心类是 `MooncakeConnectorWorker`，负责：

- 初始化 Mooncake `TransferEngine`
- 计算 P/D 并行关系
- 注册 KV Cache 内存到 Mooncake
- 生成可被远端读取的 `MooncakeAgentMetadata`
- 启动发送或接收后台线程

代码位置：`code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`。

## 4.1 初始化

```python
class MooncakeConnectorWorker:
    def __init__(self, vllm_config, engine_id, kv_cache_config):
        self._get_prefill_decode_size(vllm_config)
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())

        if self._prefill_tp_size < self._decode_tp_size:
            raise ValueError("prefill_tp_size must be >= decode_tp_size")

        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.side_channel_host = get_ip()
        self.side_channel_port = kv_port + dp_rank * tp_size * pp_size * pcp_size
        self.handshake_port = self.side_channel_port + (pp_rank + pcp_rank) * tp_size + tp_rank

        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()
```

限制和断言：

- `prefill_tp_size >= decode_tp_size`
- `decode.pp_size == 1`
- `pp_size > 1` 和 `pcp_size > 1` 当前不能同时开启
- Mamba/hybrid 场景还有额外 TP 约束，后续章节说明

## 4.2 并行配置解析

```python
def _get_prefill_decode_size(self, vllm_config):
    prefill_parallel_config = kv_transfer_config.get_from_extra_config("prefill", {})
    self._prefill_tp_size = prefill_parallel_config["tp_size"]
    self._prefill_dp_size = prefill_parallel_config["dp_size"]
    self._prefill_pp_size = prefill_parallel_config.get("pp_size", 1)

    decode_parallel_config = kv_transfer_config.get_from_extra_config("decode", {})
    self._decode_tp_size = decode_parallel_config["tp_size"]
    self._decode_dp_size = decode_parallel_config["dp_size"]
    self._decode_pp_size = decode_parallel_config.get("pp_size", 1)
    assert self._decode_pp_size == 1
    self._prefill_pp_layer_partition = prefill_parallel_config.get("pp_layer_partition")
```

这些配置不会直接创建通信组，但用于计算：

- D rank 需要从哪些 P rank 拉 KV
- 每个 P rank 监听哪个 handshake port
- P 侧 PP rank 对应哪些 layer
- 非对称 TP 时每个 D rank 需要拉几个 P shard

## 4.3 TransferEngine 全局单例

`global_te` 定义在 `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py`：

```python
class GlobalTE:
    def get_transfer_engine(self, hostname, device_name):
        if self.transfer_engine is None:
            self.transfer_engine = TransferEngine()
            ret = self.transfer_engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                "ascend",
                device_name or "",
            )
            if ret != 0:
                raise RuntimeError(...)
        return self.transfer_engine

    def register_buffer(self, ptrs, sizes):
        with self.register_buffer_lock:
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret = self.transfer_engine.register_memory(ptr, size)
                if ret != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.is_register_buffer = True
```

注意：`register_buffer()` 全局只注册一次。如果同进程里需要注册额外 buffer（例如 layerwise 中的 resharding buffer），会直接调用 `engine.register_memory()`。

## 4.4 KV Cache metadata 构建

`register_kv_caches()` 会遍历 `kv_caches`，构建每层每个 cache tensor 的元信息。

```python
self.kv_group2layeridx = self._build_kv_group2layeridx()
self.kv_caches_base_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
self.block_size_scale: list[list[int]] = [[] for _ in range(metadata_layers)]
self.block_len_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
self.block_stride_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]

for layer_name, kv_cache_tuple in kv_caches.items():
    layer_idx = layer_name_to_idx[layer_name]
    for single_kv_cache in self._as_kv_cache_tuple(kv_cache_tuple):
        tensor_num_blocks = single_kv_cache.shape[0]
        block_size_scale = tensor_num_blocks // self.num_blocks
        block_shape = single_kv_cache.shape[1:]
        self.block_len_per_addr[layer_idx].append(
            single_kv_cache.element_size() * math.prod(block_shape)
        )
        self.block_stride_per_addr[layer_idx].append(
            single_kv_cache.stride(0) * single_kv_cache.element_size()
        )
        self.block_size_scale[layer_idx].append(block_size_scale)
        self.kv_caches_base_addr[layer_idx].append(single_kv_cache.data_ptr())
```

字段含义：

| 字段 | 含义 |
|------|------|
| `kv_caches_base_addr[layer_idx][cache_idx]` | 某层某个 cache tensor 的 NPU 起始地址 |
| `block_len_per_addr[layer_idx][cache_idx]` | 一个逻辑 tensor block 的字节数 |
| `block_stride_per_addr[layer_idx][cache_idx]` | 相邻 block 的字节跨度 |
| `block_size_scale[layer_idx][cache_idx]` | tensor 物理 block 数 / 逻辑 block 数 |
| `kv_group2layeridx` | KV group 到实际 layer indices 的映射 |

## 4.5 注册内存区域

标准 Attention / sparse-c8 路径会合并底层 storage 区间，避免超过 HCCL/Mooncake 注册区域数量限制：

```python
register_regions = collect_storage_merged_register_regions(kv_caches)
validate_register_region_count(register_regions)
global_te.register_buffer(register_regions.ptrs, register_regions.lengths)
```

Hybrid/Mamba 路径会按 `kv_cache_tensors` 注册连续大块：

```python
if has_mamba_group:
    ptrs, lengths = self._get_registered_kv_tensor_buffers(kv_caches)
elif self.use_hybrid:
    ptrs, lengths = self._get_registered_kv_tensor_buffers_hybrid(kv_caches)
```

MTP / Mamba 场景中，注册起始地址可能需要包含 conv padding：

```python
if has_mtp:
    base_addr -= conv_padding
assert base_addr % (2 * 1024 * 1024) == 0
```

## 4.6 MooncakeAgentMetadata

内存注册后，Worker 构建 metadata：

```python
metadata = MooncakeAgentMetadata(
    engine_id=self.engine_id,
    te_rpc_port=self.te_rpc_port,
    kv_group2layeridx=self.kv_group2layeridx,
    block_size=self.block_size,
    kv_caches_base_addr=self.kv_caches_base_addr,
    block_size_scale=self.block_size_scale,
    num_blocks=self.num_blocks,
    block_lens=self.block_len_per_addr,
    block_strides=self.block_stride_per_addr,
    local_ip=get_ip(),
)
self.xfer_handshake_metadata = metadata
```

该 metadata 有两种用途：

1. 本地 Worker 通过 `get_handshake_metadata()` 返回给 Scheduler，Scheduler 保存跨节点 rank 的 host/engine_id。
2. P 侧 `KVCacheSendingThread` 通过 ZMQ `GET_META_MSG` 返回给 D 侧 Worker，D 侧据此知道 P 侧 KV 内存地址和 TransferEngine rpc port。

## 4.7 P 侧发送线程

P 侧 `kv_role == "kv_producer"` 时启动：

```python
self.kv_send_thread = KVCacheSendingThread(
    vllm_config,
    self.tp_rank,
    self._prefill_tp_size,
    self.engine_id,
    self.side_channel_host,
    self.side_channel_port,
    metadata,
    ready_event,
    self.kv_caches,
    self.pcp_rank,
)
self.kv_send_thread.start()
```

线程监听端口：

```python
device_index = pp_rank * tp_size + tp_rank + pcp_rank * prefill_tp_size
handshake_port = side_channel_port + device_index
path = make_zmq_path("tcp", side_channel_host, handshake_port)
```

处理消息：

| 消息 | 发送方 | 作用 |
|------|--------|------|
| `GET_META_MSG` | D 侧 RecvingThread | 获取 P 侧 `MooncakeAgentMetadata` |
| `DONE_RECVING_MSG` | D 侧 RecvingThread | 通知 P 侧某请求 KV 已拉完，可以释放延迟 blocks |

## 4.8 D 侧接收线程

D 侧 `kv_role == "kv_consumer"` 时启动：

```python
self.kv_recv_thread = KVCacheRecvingThread(
    self.tp_rank,
    self.tp_size,
    self._prefill_pp_size,
    self.engine,
    self.engine_id,
    self.handshake_port,
    self.side_channel_port,
    self.kv_caches_base_addr,
    self.block_len_per_addr,
    self.block_stride_per_addr,
    self._is_hma_required,
    ready_event,
    self.vllm_config,
    self.kv_caches,
    self._prefill_pp_layer_partition,
    self.kv_group2layeridx,
    self.block_size_scale,
)
self.kv_recv_thread.start()
```

D 侧线程不监听远端 metadata 请求；它主要负责：

- 接收 Scheduler/Worker metadata 中的拉取任务
- 通过 ZMQ 向 P 侧请求 metadata
- 调用 `TransferEngine.batch_transfer_sync_read()` 从 P 侧读 KV 到 D 本地 KV Cache
- 发送 `DONE_RECVING_MSG`
- 记录失败请求和 invalid block ids

## 4.9 启动等待

标准 `MooncakeConnectorWorker.register_kv_caches()` 会等待后台线程 ready：

```python
start_wait_time = time.time()
thread = self.kv_send_thread if self.kv_role == "kv_producer" else self.kv_recv_thread
while not ready_event.is_set():
    if not thread.is_alive():
        raise RuntimeError("KV Cache sending/receiving thread failed to start.")
    if time.time() - start_wait_time > 5 * 60:
        raise RuntimeError("Timeout waiting for KV Cache thread to be ready.")
    time.sleep(3)
```

这保证 engine 开始处理请求前，side-channel 或接收队列已经可用。
