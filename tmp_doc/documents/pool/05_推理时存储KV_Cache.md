# 第 5 章：推理时存储 KV Cache（存）

本章详细讲解推理过程中如何将计算好的 KV Cache 存入外部池（Mooncake Store）。

## 5.1 触发入口：wait_for_save

在每次推理步骤结束时，vLLM 调用 `wait_for_save()` 触发 KV Cache 的保存。

```python
# ascend_store_connector.py
def wait_for_save(self):
    if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
        return  # Consumer 不做保存
    
    if self.use_layerwise:
        return  # Layerwise 模式在 save_kv_layer 中逐个保存
    
    self.connector_worker.wait_for_save(self._get_connector_metadata())
```

### 5.1.1 KVPoolWorker.wait_for_save

```python
def wait_for_save(self, connector_metadata):
    # 1. 只有存在 can_save=True 的请求时，才创建 NPU Event
    current_event = None
    has_save_request = False
    for request in connector_metadata.requests:
        can_save = request.can_save
        if can_save is None or not can_save:
            continue
        current_event = torch.npu.Event()
        current_event.record()
        break
    
    # 2. 遍历所有需要保存的请求；同一批请求共享这个 Event
    for request in connector_metadata.requests:
        can_save = request.can_save
        if can_save is None or not can_save:
            continue  # 不需要保存的请求跳过
        
        request.skip_null_blocks_by_group = self.group_uses_align_state
        request.current_event = current_event
        # 注册到 stored_requests 计数器
        self.kv_send_thread.add_stored_request(request.req_id)
        # ★ 将请求放入发送队列
        self.kv_send_thread.add_request(request)
        has_save_request = True
    
    # 3. ★ 等待所有 put 操作完成
    if has_save_request:
        self.kv_send_thread.request_queue.join()
```

**关键点**：`request_queue.join()` 会阻塞直到队列中所有请求都被处理完毕。这确保了在请求被标记为 "finished" 之前，KV Cache 已经被写入外部池。

## 5.2 KVCacheStoreSendingThread._handle_request：存的核心逻辑

[`KVCacheStoreSendingThread`](../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py) 继承自 `KVTransferThread`，是后台发送线程。

```python
class KVCacheStoreSendingThread(KVTransferThread):
    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        
        # 1. 检查请求是否还在 stored_requests 中
        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return
        
        # 2. 遍历每个 KV Cache 组
        for group_id in req_meta.kv_cache_group_ids or [0]:
            starts = []
            ends = []
            keys = []
            block_hashes = []
            block_ids = req_meta.block_ids_by_group[group_id]
            group_block_size = self._get_block_size(group_id)
            
            # 3. ★ 用 process_tokens_with_block_ids 生成 (start, end, key, block_id)
            for start, end, key, _ in self._process_tokens_with_block_ids(
                token_len, req_meta.block_hashes, block_ids,
                kv_cache_group_id=group_id,
                skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            ):
                starts.append(start)
                ends.append(end)
                keys.append(key.to_string())
                block_hashes.append(group_block_hashes[start // group_block_size])
            
            # 4. TP 分片：只处理属于当前 TP rank 的 block
            if not self.dcp_size > 1 and not req_meta.disable_tp_key_sharding:
                starts = starts[self.tp_rank % self.put_step :: self.put_step]
                ends = ends[self.tp_rank % self.put_step :: self.put_step]
                keys = keys[self.tp_rank % self.put_step :: self.put_step]
            
            if not keys:
                continue
            
            # 5. ★ 去重：先检查哪些 key 已经存在（避免重复存储）
            exists_states = self.lookup(keys)
            missing_indices = [i for i, exists in enumerate(exists_states) if not exists]
            
            if not missing_indices:
                continue  # 全部已存在，无需存储
            
            starts = [starts[i] for i in missing_indices]
            ends = [ends[i] for i in missing_indices]
            keys = [keys[i] for i in missing_indices]
            
            # 6. ★ 计算每个 block 的 NPU 内存地址和大小
            addrs = []
            sizes = []
            for index, start in enumerate(starts):
                addr, size, _ = self._prepare_value(
                    start, ends[index], block_ids, kv_cache_group_id=group_id,
                )
                addrs.append(addr)
                sizes.append(size)
            
            # 7. consumer 模式下适配 prefill PP 分区
            if self.kv_role == "kv_consumer":
                keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                    keys, addrs, sizes, kv_cache_group_id=group_id,
                )
            
            # 8. ★ 同步 NPU 计算（确保 KV Cache 数据已写入内存）
            if current_event is not None:
                current_event.synchronize()
            
            # 9. ★★★ 调用 MooncakeBackend.put() 写入外部池
            self.m_store.put(keys, addrs, sizes)
        
        # 10. 清理
        self.dec_stored_request(req_id)
        self.request_queue.task_done()
```

### 5.2.1 流程关键步骤详解

**步骤 3：生成 key 和地址范围**

`_process_tokens_with_block_ids` 调用 `ChunkedTokenDatabase.process_tokens_with_block_ids`，将 token 序列按 block_size 分块，为每个块生成一个唯一的 `PoolKey`（格式如 `model_name@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:c1@<hash>`）。

**步骤 5：去重检查**

```python
# KVTransferThread.lookup → Backend.exists
def lookup(self, keys):
    res = self.m_store.exists(keys)  # → MooncakeBackend.exists()
    return [value == 1 for value in res]
```

```python
# MooncakeBackend.exists()
def exists(self, keys):
    if self._lazy_init and not self._store_initialized:
        return [0] * len(keys)  # 延迟初始化时返回全部不存在
    return self.store.batch_is_exist(keys)  # → MooncakeDistributedStore.batch_is_exist()
```

**步骤 6：地址计算**

`_prepare_value` → `ChunkedTokenDatabase.prepare_value`：

```python
def prepare_value(self, start, end, block_ids, kv_cache_group_id=0, cache_role="kv"):
    group_block_size = self.get_block_size(kv_cache_group_id)
    block_id = block_ids[start // group_block_size]
    group_addrs, group_block_len, group_block_stride = self._get_group_buffers(kv_cache_group_id)
    
    for index, base_addr in enumerate(group_addrs):
        block_len = group_block_len[index % length]
        block_stride = group_block_stride[index % length] if group_block_stride else block_len
        addr = base_addr + block_id * block_stride  # ★ 计算实际 NPU 内存地址
        size = int(block_len / group_block_size * (end - start))
        addr_list.append(addr)
        size_list.append(size)
    
    return addr_list, size_list, block_id
```

**步骤 8：NPU 同步**

`current_event.synchronize()` 会阻塞直到 NPU 上所有之前的操作完成，确保 KV Cache 数据已经从计算单元写回显存。这是数据正确性的关键保证。

## 5.3 MooncakeBackend.put：写入外部存储

```python
class MooncakeBackend(Backend):
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        self._ensure_initialized()  # 延迟初始化时会在此触发
        config = ReplicateConfig()
        if self.config.preferred_segment:
            config.preferred_segment = self.local_seg
        config.prefer_alloc_in_same_node = self.config.prefer_alloc_in_same_node
        
        # ★ 调用 Mooncake 的批量写入 API
        res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
        
        for value in res:
            if value < 0:
                logger.error("Failed to put key. keys=%s, result=%s", keys, res)
```

`batch_put_from_multi_buffers` 是 Mooncake Distributed Store 的原生 API，它会：
1. 根据 key 的 hash 确定存储位置
2. 通过 TransferEngine 将 NPU 内存中的数据通过 RDMA 传输到目标机器的 CPU 内存
3. 如果配置了 SSD offload，还会将数据写入 SSD
4. 在 metadata server（如 Redis）中记录 key → 存储位置的映射

## 5.4 Layerwise 存储模式（KVCacheStoreLayerSendingThread）

当 `use_layerwise=True` 时，使用逐层存储模式。与标准模式的区别：

### 5.4.1 触发方式

```python
def save_kv_layer(self, connector_metadata):
    if self.current_layer == 0:
        # 第一层：初始化 layerwise_storers 生成器列表
        for request in connector_metadata.requests:
            self.kv_send_thread.add_stored_request(request.req_id)
            layerwise_storer = self.store_layer(request, current_event)
            self.layerwise_storers.append(layerwise_storer)
    
    # 逐层调用
    for layerwise_storer in self.layerwise_storers:
        next(layerwise_storer)
    
    self.current_layer += 1
```

### 5.4.2 store_layer 生成器

```python
def store_layer(self, request, current_event):
    # 1. 生成所有层的 key
    for start, end, key in self.token_database.process_tokens(...):
        keys_multi_layer = key.split_layers(self.num_layers)
        starts.append(start)
        ends.append(end)
        keys.append(keys_multi_layer)  # [block_num, layer_num]
    
    # 2. 转置为 [layer_num, block_num]
    keys = [list(row) for row in zip(*keys)]
    
    # 3. 逐层 yield
    for layer_id, keys_multi_chunk in enumerate(keys):
        req_meta = LayerMultiBlockReqMeta(
            req_id, keys_multi_chunk, starts, ends, block_ids_by_group,
            layer_id, is_last_chunk, current_event, token_ids=...,
            original_block_size=..., block_hashes=..., kv_cache_group_id=...,
        )
        self.kv_send_thread.add_request(req_meta)
        yield  # 等待下一层
```

### 5.4.3 KVCacheStoreLayerSendingThread._handle_request

```python
def _handle_request(self, req_meta: LayerMultiBlockReqMeta):
    starts = req_meta.starts
    keys = req_meta.keys
    layer_id = req_meta.layer_id
    current_event = req_meta.current_event
    
    # TP 分片
    starts = starts[self.tp_rank % self.put_step :: self.put_step]
    keys = keys[self.tp_rank % self.put_step :: self.put_step]
    
    # 去重
    key_list = [key.to_string() for key in keys]
    exists_states = self.lookup(key_list)
    missing_indices = [i for i, exists in enumerate(exists_states) if not exists]
    
    # 地址计算（使用 prepare_value_layer）
    for index, key in enumerate(key_list):
        addr, size, _ = self.token_database.prepare_value_layer(
            starts[index], ends[index], req_meta.block_ids_by_group[0], layer_id
        )
    
    # 同步 & 写入
    if current_event is not None:
        current_event.synchronize()
    self.m_store.put(key_list, addr_list, size_list)
    
    # 最后一层时标记完成
    if layer_id == self.final_layer_id and is_last_chunk:
        self.dec_stored_request(req_meta.req_id)
        self.set_finished_request(req_meta.req_id)
```

## 5.5 "存" 的完整时序图

```
推理循环
    │
    ▼
wait_for_save()
    │
    ├── 创建 NPU Event，调用 record()
    ├── 遍历 connector_metadata.requests
    │       │
    │       ├── add_stored_request(req_id)  # 计数器 +1
    │       └── add_request(req_meta)       # 放入队列
    │
    ├── request_queue.join()  # 阻塞等待
    │       │
    │       ▼
    │   KVCacheStoreSendingThread._handle_request()
    │       │
    │       ├── process_tokens_with_block_ids()
    │       │       └── 生成 (start, end, key, block_id) 迭代器
    │       │
    │       ├── TP 分片 (只处理当前 rank 的 block)
    │       │
    │       ├── lookup(keys) → Backend.exists()
    │       │       └── 去重：跳过已存在的 key
    │       │
    │       ├── prepare_value()
    │       │       └── 计算 NPU 内存地址
    │       │
    │       ├── current_event.synchronize()  # ★ 确保 NPU 计算完成
    │       │
    │       └── m_store.put(keys, addrs, sizes)
    │               └── MooncakeDistributedStore.batch_put_from_multi_buffers()
    │                       ├── TransferEngine 传输数据
    │                       └── Metadata Server 记录 key 映射
    │
    └── 返回（所有 put 操作已完成）
```

## 5.6 关键设计点总结

1. **去重存储**：在 put 之前先 `exists` 检查，避免重复写入相同内容的 KV Cache。
2. **NPU 同步**：`current_event.synchronize()` 确保 KV Cache 数据已从计算单元写回显存。
3. **TP 分片**：每个 TP rank 只存储自己负责的 KV head 部分，避免数据冗余。
4. **队列阻塞**：`request_queue.join()` 确保在请求完成前所有存储操作已提交。
5. **延迟初始化**：压缩模型（如 DeepSeek V4）使用 fabric memory 时，store 可以延迟到第一次 put 时初始化。