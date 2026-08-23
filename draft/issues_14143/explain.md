# Issue #14143 问题解答记录

本文用于记录对 Issue #14143 的后续问题和代码核查结果。

## 编号规则

- 第一个问题编号为 `Q001`。
- 后续问题依次使用 `Q002`、`Q003`，不因问题属于同一主题而复用编号。
- 每个问题包含：问题、结论、代码证据、调用链和限定条件。

## Q001：如何确认“AscendStore 的 KV 传输线程包含 key 构造、token/block 遍历、列表构造、地址计算以及外部 store 调用”？

### 问题

文档中有如下表述：

> AscendStore 的 KV 传输工作当前放在 Python 线程中，线程内包含 key 构造、token/block 遍历、列表构造、地址计算以及外部 store 调用。

需要定位具体代码，确认这句话是否准确。

### 结论

这句话的核心判断可以由源码确认，但应加上路径限定：

> 在异步保存、异步加载以及 layerwise 传输路径中，后台 Python 传输线程会执行 token/block 遍历、key 生成、地址和 size 列表构造，以及 `m_store.get()`/`m_store.put()` 调用。

初始 KV buffer 注册并不发生在传输线程中；此外，当普通 load 使用同步模式时，部分 key/address 准备和 `m_store.get()` 会由 `KVPoolWorker.start_load_kv()` 的调用线程执行。因此不能笼统地说“所有 KV 传输工作都在线程中”。

### 1. 先确认：这些对象确实是 Python 线程

传输线程的创建入口是：

`vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` 的 `KVPoolWorker._start_kv_transfer_threads()`。

普通模式下，该方法创建并启动：

- `KVCacheStoreSendingThread`（约第 554 行）；
- 异步加载启用时的 `KVCacheStoreRecvingThread`（约第 571 行）。

layerwise 模式下，它创建：

- `KVCacheStoreKeyLayerSendingThread` 或 `KVCacheStoreLayerSendingThread`；
- `KVCacheStoreKeyLayerRecvingThread` 或 `KVCacheStoreLayerRecvingThread`。

这些类都继承 `kv_transfer.py` 中约第 305 行的：

```python
class KVTransferThread(threading.Thread):
```

基类的 `run()`（约第 487 行）展示了实际执行模型：

```python
request_data = self.request_queue.get()
self._handle_request(request_data)
```

因此，后续 `_handle_request()` 中的 Python 代码确实由后台 `threading.Thread` 执行。

### 2. 普通异步保存：token/block 遍历和 key 构造

普通保存线程的入口是：

`kv_transfer.py` 中 `KVCacheStoreSendingThread._handle_request()`，约第 654 行。

它进一步调用 `_handle_stored_request()`，约第 692 行。

在这个方法中，代码按 KV cache group 处理请求：

```python
for group_id in req_meta.kv_cache_group_ids or [0]:
```

随后在约第 762 行调用：

```python
iterator = self.token_database.process_token_key_strings_with_block_ids(
    token_len,
    req_meta.block_hashes,
    block_ids,
    ...,
)
```

然后在约第 772 行遍历生成结果，并构造 Python 列表：

```python
for start, end, key, block_hash, block_id in iterator:
    starts.append(start)
    ends.append(end)
    keys.append(key)
    key_block_ids.append(block_id)
```

生成器的具体遍历逻辑位于 `metadata.py` 的 `ChunkedTokenDatabase._iter_token_chunks()`，约第 490 行。它会：

- 根据 `token_len` 和 block size 计算逻辑 block 数量；
- 遍历 `chunk_id`；
- 计算每个 chunk 的 `start_idx` 和 `end_idx`；
- 根据 `block_ids` 解析对应的 block id；
- 应用 mask、chunk filter 和 TP/DCP shard 过滤。

`process_token_key_strings_with_block_ids()` 位于 `metadata.py` 约第 590 行，最终通过：

```python
prefix + block_hash_to_str(hash_val)
```

生成 key 字符串。因此，token/block 遍历和 key 生成都在保存线程调用栈内发生。

### 3. 普通异步保存：列表过滤和地址计算

保存线程先查询哪些 key 已存在：

`kv_transfer.py` 约第 782 行：

```python
exists_states = self.lookup(keys)
missing_indices = [
    index for index, exists in enumerate(exists_states) if not exists
]
```

然后重新构造只包含 missing block 的列表，约第 786-791 行：

```python
starts = [starts[index] for index in missing_indices]
ends = [ends[index] for index in missing_indices]
keys = [keys[index] for index in missing_indices]
key_block_ids = [key_block_ids[index] for index in missing_indices]
```

地址和 size 列表的构造在约第 819 行开始：

```python
addr, size, _ = self._prepare_value(...)
addrs.append(addr)
sizes.append(size)
```

`KVTransferThread._prepare_value()` 只是转调 `ChunkedTokenDatabase.prepare_value()`。后者在 `metadata.py` 约第 437 行实际计算地址：

```python
addr = base_addr + block_id * block_stride
size = int(block_len / group_block_size * (end - start))
addr_list.append(addr)
size_list.append(size)
```

这说明这里不是简单地把现成参数转交给后端，而是在传输线程中进行 Python 对象、列表和整数地址的准备。

### 4. 普通异步保存：外部 store 调用

保存线程在完成 key、地址和 size 准备后，在 `kv_transfer.py` 约第 863 行调用：

```python
self.m_store.put(keys, addrs, sizes)
```

在调用前还可能执行：

```python
current_event.synchronize()
```

也就是说，同一个传输线程负责 Python 侧准备、必要的事件等待以及后端写入调用。

`self.m_store` 的具体实现取决于 backend 配置。例如 `backend/memcache_backend.py` 的 `put()` 最终调用底层 store 的批量写接口；`backend/mooncake_backend.py` 也有对应的 `put()` 实现。

### 5. 普通异步加载：相同结构

普通异步接收线程是 `KVCacheStoreRecvingThread._handle_request()`，位于 `kv_transfer.py` 约第 896 行。

关键步骤如下：

- 约第 923 行：调用 `token_database.load_mask()`；
- 约第 932 行：调用 `process_token_key_strings_with_block_ids()`；
- 约第 941 行：遍历 token/block 结果；
- 约第 942 行：调用 `_prepare_value()` 计算地址和 size；
- 约第 949-951 行：追加 `key_list`、`addr_list`、`size_list`；
- 约第 970 行：调用 `self.m_store.get(key_list_c, addr_list_c, size_list_c)`。

因此，异步 load 线程也执行了 token/block 遍历、key/address 列表构造和外部 store 读取。

### 6. key-layerwise：`PoolKey` 和每层 key 的证据

key-layerwise 保存线程是 `KVCacheStoreKeyLayerSendingThread._handle_request()`，约第 1076 行。

当没有缓存的 token 处理结果时，它在约第 1116 行执行：

```python
for start, end, key in self.token_database.process_tokens(...):
```

随后：

- `key.split_layers(...)` 将一个 block key 拆成各层 key；
- `key.to_string()`（约第 1134 行）生成远端字符串 key；
- `prepare_value_layer()`（约第 1135 行）计算当前 layer 的地址和 size；
- `key_list`、`addr_list`、`size_list` 在循环中不断追加；
- `lookup()` 检查远端 key 是否存在；
- `self.m_store.put(...)`（约第 1155 行）执行写入。

key-layerwise 接收线程是 `KVCacheStoreKeyLayerRecvingThread._handle_request()`，约第 1205 行。它会：

- 等待上一层保存完成事件和 attention gate；
- 遍历 `block_range`；
- 根据 block hash 调用 `_make_key_by_hash()`；
- 调用 `split_layers()` 和 `to_string()`；
- 调用 `prepare_value_layer()`；
- 最后在约第 1256 行调用 `self.m_store.get(...)`。

### 7. GVA-layerwise：数组构造和 batch copy 也在线程中

GVA-layerwise 路径使用 `KVCacheStoreLayerSendingThread` 和 `KVCacheStoreLayerRecvingThread`。

保存线程的 `_handle_request()` 约从第 1329 行开始。它在传输线程中：

- 调用 `builder.build_addrs(...)`；
- 构造 `all_gvas`、`all_addrs`、`all_sizes`；
- 用 `np.concatenate(...)` 合并数组；
- 调用 `_batch_copy_with_limits(...)`；
- 调用 `self.m_store.batch_write_finish(...)`。

接收线程约从第 1481 行开始，在约第 1517 行调用 `builder.build_addrs()` 或 `builder.build()`，之后在约第 1555-1557 行合并 GVA、地址和 size 数组，最后在约第 1560 行调用 `_batch_copy_with_limits(...)`。

这条路径的后端传输调用是 `batch_copy`，而不是普通模式的 `get/put`，但它同样包含 Python 侧的任务解析、列表/NumPy 数组构造和后端调用。

### 8. 哪些内容不应归到传输线程

需要区分以下两个阶段。

#### 8.1 初始 buffer 注册

`KVPoolWorker.register_kv_caches()` 在 `pool_worker.py` 约第 708 行遍历 NPU tensor，并在约第 748 行读取：

```python
base_addr = cache.data_ptr()
```

随后在约第 818 行执行：

```python
self.m_store.register_buffer(ptrs, lengths)
```

这发生在线程启动之前，不是 `KVTransferThread._handle_request()` 的工作。它可以证明当前 buffer 注册使用裸地址，但不能作为“传输线程执行了 `data_ptr()` 注册”的证据。

#### 8.2 同步 load

当 `load_async=False` 时，`KVPoolWorker.start_load_kv()` 会在 worker 调用线程中构造 `key_list`、`addr_list`、`size_list`，并调用 `self.m_store.get()`。因此普通同步 load 不应被描述成“由接收线程完成”。

### 9. 最准确的表述

建议在分析文档中使用下面这句话：

> AscendStore 的后台异步保存、异步加载和 layerwise 传输由继承 `threading.Thread` 的 Python 线程执行。这些线程不仅调用后端 I/O，还会在 `_handle_request()` 路径中遍历 token/block、生成 key、筛选并构造 key/address/size 列表，以及计算 block 对应的 NPU 地址。初始 `data_ptr()`/`register_buffer()` 和同步 load 的部分准备工作由 worker 调用线程执行。

### 10. 这对 GIL 结论意味着什么

源码可以确认：传输线程不是纯粹的“等待 I/O”包装，它包含明显的 Python 计算和容器操作。

但源码不能单独证明：

- 这些 Python 计算占总传输时间的比例；
- `m_store.get/put` 是否释放 GIL；
- GIL 是否已经是目标 workload 的主要瓶颈；
- 改成进程后端到端性能是否一定提升。

因此，源码证据支持的是“存在 GIL 竞争的结构性可能”，性能结论仍需要分别 profiling：token/key 生成、地址准备、列表构造、backend 调用、NPU 等待、模型 forward 和端到端 TTFT。
