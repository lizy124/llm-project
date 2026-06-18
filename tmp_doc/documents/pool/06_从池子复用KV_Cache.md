# 第 6 章：从池子复用 KV Cache（取）

本章详细讲解如何从外部池（Mooncake Store）中查询并加载已缓存的 KV Cache。

## 6.1 触发入口：start_load_kv

在每次推理步骤开始时，vLLM 调用 `start_load_kv()` 开始加载外部 KV Cache。

```python
# ascend_store_connector.py
def start_load_kv(self, forward_context, **kwargs):
    metadata = self._get_connector_metadata()
    self.connector_worker.start_load_kv(metadata)
```

### 6.1.1 KVPoolWorker.start_load_kv

```python
def start_load_kv(self, metadata: AscendConnectorMetadata):
    self.current_layer = 0
    self.layerwise_retrievers = []
    
    for request in metadata.requests:
        load_spec = request.load_spec
        
        # 1. 检查是否需要加载
        if load_spec is None or not load_spec.can_load:
            continue  # 跳过没有外部命中的请求
        
        # 2. 确定加载的 token 数量
        token_len = request.token_len_chunk
        if (load_spec.kvpool_cached_tokens % self.cache_transfer_granularity != 0) and (
            load_spec.kvpool_cached_tokens == token_len - 1
        ):
            token_len = request.load_spec.kvpool_cached_tokens + 1
        else:
            token_len = request.load_spec.kvpool_cached_tokens
        
        request.load_spec.token_len = token_len
        
        # 3. 根据模式选择加载方式
        if self.use_layerwise:
            # Layerwise 模式：逐层加载
            layerwise_retriever = self.retrieve_layer(request)
            next(layerwise_retriever)  # 触发第一层加载
            self.layerwise_retrievers.append(layerwise_retriever)
        else:
            if self.load_async:
                # 异步模式：将请求放入 RecvingThread 队列
                self.kv_recv_thread.add_request(request)
            else:
                # ★ 同步模式：直接调用 Backend.get()
                self._sync_load_kv(request)
```

### 6.1.2 同步加载模式（_sync_load_kv）

```python
# 在 start_load_kv 中内联的同步加载逻辑
for group_id in load_group_ids:
    block_ids = request.block_ids_by_group[group_id]
    group_block_size = self.grouped_block_size[group_id]
    mask_num = request.load_spec.vllm_cached_tokens // group_block_size * group_block_size
    
    # 1. 生成 key 和地址
    for start, end, key, _ in self.token_database.process_tokens_with_block_ids(
        token_len, request.block_hashes, block_ids, mask_num,
        kv_cache_group_id=group_id,
        skip_null_blocks=skip_null,
    ):
        addr, size, block_id = self.token_database.prepare_value(
            start, end, block_ids, kv_cache_group_id=group_id,
        )
        key_list.append(key.to_string())
        addr_list.append(addr)
        size_list.append(size)
        block_id_list.append(block_id)
    
    if not key_list:
        continue
    
    # 2. TP 轮转（避免所有 TP rank 同时访问同一 key）
    key_list_c = key_list[self.tp_rank % len(key_list):] + key_list[:self.tp_rank % len(key_list)]
    addr_list_c = addr_list[self.tp_rank % len(addr_list):] + addr_list[:self.tp_rank % len(addr_list)]
    size_list_c = size_list[self.tp_rank % len(size_list):] + size_list[:self.tp_rank % len(size_list)]
    block_id_list_c = block_id_list[self.tp_rank % len(block_id_list):] + block_id_list[:self.tp_rank % len(block_id_list)]
    
    # 3. ★★★ 调用 Backend.get() 从外部池读取数据
    ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
    
    # 4. 记录失败的 block
    if ret is not None and any(r != 0 for r in ret):
        missing_block_ids = record_failed_blocks(block_id_list_c, ret)
        self._invalid_block_ids.update(missing_block_ids)
    elif ret is None:
        missing_block_ids = record_failed_blocks(block_id_list_c, [1] * len(block_id_list_c))
        self._invalid_block_ids.update(missing_block_ids)
```

同步加载路径在 Worker 主执行流中直接更新 `_invalid_block_ids`；异步加载路径运行在 RecvingThread 中，因此会通过 `_invalid_block_ids_lock` 加锁更新。

## 6.2 查询命中：lookup_scheduler 详解

在 Scheduler 端决定是否加载之前，需要通过 `lookup_scheduler` 查询外部池中是否存在匹配的 KV Cache。

```python
def lookup_scheduler(self, token_len, block_hashes, kv_cache_group_ids=None, use_layerwise=False):
    hits = []
    kv_cache_group_ids = self._get_lookup_gate_group_ids(kv_cache_group_ids)
    
    for group_id in kv_cache_group_ids:
        keys = []
        starts = []
        ends = []
        
        # 1. 生成 key
        for start, end, key in self.token_database.process_tokens(
            token_len, block_hashes, kv_cache_group_id=group_id,
        ):
            if use_layerwise:
                keys_multi_layer = key.split_layers(self.num_layers)
                for item in keys_multi_layer:
                    keys.append(item.to_string())
            else:
                keys.append(key.to_string())
            starts.append(start)
            ends.append(end)
        
        if not keys:
            return 0
        
        # 2. ★ 查询当前 rank / 当前 group 生成的 key 是否存在
        # 当前实现不再在单个 Worker 内手动展开所有 TP/PP rank 的 key。
        # 每个 rank 的 KeyMetadata 已包含自身 head_or_tp_rank / pp_rank。
        res = self.m_store.exists(keys)
        
        # 3. Layerwise 模式需要先把每个 chunk 的所有层合并成一个存在性结果
        if use_layerwise:
            res = self.check_all_layers_exists(res, self.num_layers)
        
        # 4. ★ 找到最大连续命中位置，并对齐到 cache_transfer_granularity
        if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
            hit_end = 0
            for index in range(len(ends) - 1, -1, -1):
                if res[index] == 1 and ends[index] % self.cache_transfer_granularity == 0:
                    hit_end = ends[index]
                    break
        else:
            hit_end = ends[-1]
            for index, value in enumerate(res):
                if value != 1:
                    hit_end = 0
                    for hit_index in range(index, 0, -1):
                        if starts[hit_index] % self.cache_transfer_granularity == 0:
                            hit_end = starts[hit_index]
                            break
                    break
        hits.append(hit_end)
    
    return min(hits) if hits else 0
```

### 6.2.1 lookup_scheduler 的关键逻辑

**为什么当前实现不手动扩展 TP/PP key？**

`PoolKey` 的 `KeyMetadata` 已包含当前 rank 的 `head_or_tp_rank` 和 `pp_rank`。因此 `lookup_scheduler()` 在 Worker rank 0 的服务线程中，按当前 rank / 当前 group 生成 key 并查询 `Backend.exists()`。文档旧版本中的“在单个 Worker 内把 `@head_or_tp_rank:0` / `@pp_rank:0` 替换成所有 rank 变体”的逻辑，已经不是当前 `vllm-ascend` 代码路径。

**Gateway 组过滤**：`_get_lookup_gate_group_ids` 会过滤掉不适合作为查询 gate 的组。当前规则包括：
- Mamba align / align-state group 不作为 gate
- `cache_family == "c128"` 不作为 gate
- 非 `c1` 的压缩组不作为 gate
- group block size 必须等于主 block size

这样可以避免 DeepSeek V4 等 hybrid 布局中，稀疏或辅助 group 把本可加载的主 KV group 误判成 0 命中。

## 6.3 KVCacheStoreRecvingThread._handle_request：异步加载

当 `load_async=True` 时，使用后台线程异步加载。

```python
class KVCacheStoreRecvingThread(KVTransferThread):
    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len
        req_id = req_meta.req_id
        
        addr_list = []
        size_list = []
        key_list = []
        block_id_list = []
        
        # 1. 遍历所有 KV Cache 组，生成 key 和地址
        for group_id in req_meta.kv_cache_group_ids or [0]:
            block_ids = req_meta.block_ids_by_group[group_id]
            group_block_size = self._get_block_size(group_id)
            mask_num = (req_meta.load_spec.vllm_cached_tokens // group_block_size * group_block_size)
            
            for start, end, key, _ in self._process_tokens_with_block_ids(
                token_len, req_meta.block_hashes, block_ids, mask_num,
                kv_cache_group_id=group_id,
                skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            ):
                addr, size, block_id = self._prepare_value(
                    start, end, block_ids, kv_cache_group_id=group_id,
                )
                key_list.append(key.to_string())
                addr_list.append(addr)
                size_list.append(size)
                block_id_list.append(block_id)
        
        if not key_list:
            self.set_finished_request(req_id)
            return
        
        # 2. TP 轮转
        key_list_c = key_list[self.tp_rank % len(key_list):] + key_list[:self.tp_rank % len(key_list)]
        addr_list_c = addr_list[self.tp_rank % len(addr_list):] + addr_list[:self.tp_rank % len(addr_list)]
        size_list_c = size_list[self.tp_rank % len(size_list):] + size_list[:self.tp_rank % len(size_list)]
        block_id_list_c = block_id_list[self.tp_rank % len(block_id_list):] + block_id_list[:self.tp_rank % len(block_id_list)]
        
        # 3. ★★★ 调用 Backend.get()
        ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
        
        # 4. 记录失败的 block
        if ret is not None and any(r != 0 for r in ret):
            missing_block_ids = record_failed_blocks(block_id_list_c, ret)
            self._invalid_block_ids.update(missing_block_ids)
        elif ret is None:
            self._invalid_block_ids.update(block_id_list_c)
        
        self.set_finished_request(req_id)
        self.request_queue.task_done()
```

## 6.4 MooncakeBackend.get：从外部存储读取

```python
class MooncakeBackend(Backend):
    def get(self, keys, addrs, sizes):
        if self._lazy_init and not self._store_initialized:
            logger.error("get() called before store init.")
            return
        
        # ★ 调用 Mooncake 的批量读取 API
        res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
        
        res_list = list(res)
        for i, value in enumerate(res_list):
            if value < 0:
                # 读取失败
                logger.error("Failed to get key. keys=%s", keys)
            elif value > 0:
                res_list[i] = 0  # 成功，归一化
        
        return res_list
```

`batch_get_into_multi_buffers` 是 Mooncake Distributed Store 的原生 API，它会：
1. 根据 key 查询 metadata server 获取存储位置
2. 通过 TransferEngine 将数据从存储节点通过 RDMA 传输到目标 NPU 内存地址
3. 返回每个 key 的读取状态（0 = 成功，<0 = 失败）

## 6.5 Layerwise 加载模式

### 6.5.1 retrieve_layer 生成器

```python
def retrieve_layer(self, request):
    token_len = request.token_len_chunk
    mask_num = request.load_spec.vllm_cached_tokens // self.block_size * self.block_size
    
    ret_mask = torch.zeros(token_len, dtype=torch.bool, device="cpu")
    
    starts = []
    ends = []
    keys = []
    
    # 1. 生成所有层的 key
    for start, end, key in self.token_database.process_tokens(token_len, request.block_hashes, mask_num):
        keys_multi_layer = key.split_layers(self.num_layers)
        starts.append(start)
        ends.append(end)
        keys.append(keys_multi_layer)
        ret_mask[start:end] = True
    
    # 2. 转置为 [num_layer, block_num]
    keys = [list(row) for row in zip(*keys)]
    
    # 3. 逐层加载
    for layer_id, keys_multi_chunk in enumerate(keys):
        if not first_flag:
            # 等待上一层的 get 完成
            is_finish = self.get_event.wait(timeout=3)
        self.get_event.clear()
        
        req_meta = LayerMultiBlockReqMeta(
            request.req_id, keys_multi_chunk, starts, ends,
            request.block_ids_by_group, layer_id,
        )
        self.kv_recv_thread.add_request(req_meta)
        first_flag = False
        yield None  # 等待下一层
    
    yield ret_mask  # 最后返回 mask
```

### 6.5.2 KVCacheStoreLayerRecvingThread._handle_request

```python
def _handle_request(self, req_meta: LayerMultiBlockReqMeta):
    addr_list = []
    size_list = []
    key_list = []
    block_id_list = []
    
    for index, key in enumerate(req_meta.keys):
        addr, size, block_id = self.token_database.prepare_value_layer(
            req_meta.starts[index], req_meta.ends[index],
            req_meta.block_ids_by_group[0], req_meta.layer_id,
        )
        key_list.append(key.to_string())
        addr_list.append(addr)
        size_list.append(size)
        block_id_list.append(block_id)
    
    # TP 轮转 + 调用 Backend.get()
    offset = self.tp_rank % len(key_list)
    key_list_c = key_list[offset:] + key_list[:offset]
    addr_list_c = addr_list[offset:] + addr_list[:offset]
    size_list_c = size_list[offset:] + size_list[:offset]
    
    ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
    
    # 记录失败 block
    # ...
    
    # ★ 通知主线程本层加载完成
    self.get_event.set()
```

## 6.6 "取" 的完整时序图

### 6.6.1 同步模式

```
start_load_kv()
    │
    ├── 遍历 metadata.requests
    │       │
    │       ├── 检查 load_spec.can_load
    │       ├── 确定 token_len
    │       │
    │       ├── process_tokens_with_block_ids()
    │       │       └── 生成 (start, end, key, block_id) 迭代器
    │       │
    │       ├── prepare_value()  → 计算目标 NPU 内存地址
    │       │
    │       ├── TP 轮转 key/addr/size
    │       │
    │       └── m_store.get(keys, addrs, sizes)
    │               └── MooncakeDistributedStore.batch_get_into_multi_buffers()
    │                       ├── Metadata Server 查询 key 位置
    │                       ├── TransferEngine 传输数据到 NPU 内存
    │                       └── 返回每个 key 的读取状态
    │
    └── 记录失败的 block_ids → _invalid_block_ids
```

### 6.6.2 异步模式

```
start_load_kv()
    │
    └── kv_recv_thread.add_request(request)
            │
            ▼
    KVCacheStoreRecvingThread._handle_request()
            │
            ├── 生成 key 和地址
            ├── TP 轮转
            ├── m_store.get(keys, addrs, sizes)
            └── set_finished_request(req_id)
```

## 6.7 加载失败处理

当某些 block 加载失败时，系统通过 `_invalid_block_ids` 集合记录失败的 block ID。这些 block 会被标记为无效，后续推理时不会使用这些 block 中的 KV Cache 数据。

```python
def get_block_ids_with_load_errors(self) -> set[int]:
    with self._invalid_block_ids_lock:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
    return invalid_blocks
```

## 6.8 "取"的流程关键设计点

1. **TP 轮转**：不同 TP rank 通过不同的起始偏移访问 key 列表，避免所有 rank 同时竞争同一 key。
2. **rank 感知的 key 查询**：`PoolKey` 中包含 `head_or_tp_rank` 和 `pp_rank`，当前实现按当前 rank 生成并查询 key；hybrid 场景下会先筛选适合作为 lookup gate 的 group。
3. **对齐到 cache_transfer_granularity**：加载的 token 数量必须对齐到传输粒度（如 16384），避免部分块不一致。
4. **Layerwise 模式**：逐层加载，每层完成后通过 `get_event` 通知主线程，实现流水线化。
5. **失败容错**：单个 block 加载失败不会导致整个请求失败，系统会记录失败的 block 并继续处理其他 block。