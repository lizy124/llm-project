# 02_4. 传输线程与 Backend 如何执行 KV I/O？

源码位置：

- `ascend_store/kv_transfer.py`
- `ascend_store/backend/base.py`
- `ascend_store/backend/memcache_backend.py`
- `ascend_store/backend/mooncake_backend.py`
- `ascend_store/backend/yuanrong_backend.py`
- `ascend_store/attention_fence.py`

本文关注 `ascend_store` 的执行层：Worker 已经决定 key、block range 和 layer 后，线程怎样批量组织地址，三种 backend 分别提供什么能力，以及完成、异常和设备同步如何返回上层。

---

## 1. 执行层的边界

线程和 backend 的分工可以压缩为：

```text
KVPoolWorker
  决定本轮有哪些 request/group/layer 任务

LayerBatchBuilder
  把 block range 展开为批量 key/GVA/address/size

KVTransferThread 派生类
  排队、同步、拆批、调用 backend、记录完成或失败

Backend
  实现具体外部存储的 exists/alloc/put/get/copy 协议
```

执行层不决定外部命中应贡献多少 token，也不直接修改 vLLM request status。

---

## 2. LayerBatchBuilder 为什么单独存在？

GVA layerwise 的任务有明显的“跨 layer 不变部分”和“逐 layer 变化部分”：

```text
跨 layer 不变
  req ids、block ranges、keys、GVA、每个请求的 offset/count

逐 layer 变化
  本地 HBM address、该层 cache entry、实际 size
```

`LayerBatchBuilder.build_shared()` 先完成去重、key/GVA 检查和请求边界计算，生成 `SharedBlockData`；`build_addrs()` 再为指定 group 内 layer 展开本地地址。`build()` 是没有预计算 shared data 时的一步式入口。

Builder 内部复用 NumPy buffer，减少 layerwise 热路径上反复分配数组。扩容由 `_ensure_buf()` 统一完成；生成结果前必须确认 request 的 block/GVA NumPy 快照覆盖任务范围。

### 2.1 去重的原因

同一物理 block 可能由于多个 request range 或共享 prefix 重复出现。`_dedupe_transfer_blocks()` 保留第一次出现的传输项，避免一次 batch 对同一 GVA/local address 重复 copy，也避免 stored request 引用计数被重复扣减。

### 2.2 shared data 不能跨 step 复用

它可以跨 layer 复用，但只属于当前 step 的 task 集合。下一轮 allocation、GVA offset 或 partial block 可能变化，不能把上一轮 `SharedBlockData` 当成长期 cache。

---

## 3. KVTransferThread 提供什么公共协议？

`KVTransferThread` 继承 `threading.Thread`，为所有方向提供共同状态：

```text
request_queue
finished_requests + done_task_lock
stored_requests 引用计数
failure 异常槽
KV events
backend、token database、并行与 block 配置
```

### 3.1 run loop

线程启动后先设置设备和 OS thread name，再持续从 queue 取任务并调用派生类 `_handle_request()`。若处理异常，则保存失败、清理当前 request 的必要状态，并让后续同步点通过 `raise_if_failed()` 抛出。

后台异常不能被解释成普通 cache miss。cache miss 是 lookup/get 返回的业务结果；线程异常可能意味着 backend、地址或同步协议已经失效。

### 3.2 request-level 完成

`set_finished_request()` 写入完成集合，`get_and_clear_finished_requests()` 原子取走匹配请求。派生线程必须在其真正的 request 完成边界调用它，而不是每处理一个 layer/block 就报告完成。

### 3.3 stored request 引用计数

Layerwise save 会把一个 request 拆成多个 layer/group task。`add_stored_request()`、`dec_stored_request()` 和 `try_finish_and_delete_stored_request()` 维护剩余任务数；只有最后一项完成，request 才进入 `finished_requests`。

### 3.4 拆批限制

`_split_transfer_packets()` 与 `_batch_copy_with_limits()` 根据最大 block 数或最大字节数拆分大批量传输：

```text
完整 array
  -> packet 1: [offset, end)
  -> backend.store.batch_copy(...)
  -> packet 2
  -> ...
```

拆批不改变 request 完成语义。只有所有 packet 成功后才能发布 write finish 或设置 layer event。

---

## 4. 六类派生线程分别做什么？

### 4.1 KVCacheStoreSendingThread

这是非 layerwise 保存线程，输入为 `ReqMeta`。主要过程是：

```text
等待 request.current_event
  -> 按 group 和 store mask 遍历增量 token range
  -> 生成 PoolKey/address/size
  -> backend.put()
  -> 更新 KV event（可选）
  -> request finished
```

它还处理重复存储请求、null block、并行 key 轮转和最后 chunk 完成条件。

### 4.2 KVCacheStoreRecvingThread

这是非 layerwise 异步加载线程，输入同样为 `ReqMeta`：

```text
读取 LoadSpec
  -> 按 group/load mask 生成 key/address/size/block id
  -> backend.get()
  -> 将失败结果映射成本地 invalid block ids
  -> request finished
```

同步非 layerwise load 不使用此线程，而是在 `KVPoolWorker.start_load_kv()` 中执行相同方向的核心逻辑。

### 4.3 KVCacheStoreKeyLayerSendingThread

这是普通 object key 的逐层保存线程。输入为当前 layer 的 `LayerTransferTask` 列表：

```text
复用 cached process_tokens
  -> 为当前 layer 生成 LayerPoolKey 和本地地址
  -> 过滤 backend 已存在的 key
  -> 等待 sync_save_event[layer]
  -> backend.put()
  -> final layer + last chunk 时完成 request
  -> 设置 layer_save_finished_event
```

逐层 key 直接对应 backend object，不使用 GVA alloc/write-finish 协议。

### 4.4 KVCacheStoreKeyLayerRecvingThread

这是普通 object key 的逐层加载线程。它先等待复用 slot 的前一 layer save，必要时等待 attention gate，然后：

```text
LayerPoolKey
  -> prepare_value_layer()
  -> backend.get()
  -> final layer + last chunk 时完成 request
  -> 设置 layer_load_finished_event 和 get_event
```

当前实现期望一次 `LayerLoadTask` 至多携带一个 transfer task；这与 GVA multi-group batching 的组织方式不同。

### 4.5 KVCacheStoreLayerSendingThread

这是 MemCache GVA layerwise 保存线程。Worker 已经把外部 GVA 放入 `SharedBlockData`，线程只需为当前 layer 构造本地地址：

```text
build_addrs(shared, layer_idx_in_group)
  -> 等待 sync_save_event[physical_layer]
  -> batch_copy(local address -> GVA)
  -> 最后相关 task 调用 batch_write_finish(keys, results)
  -> stored request 引用归零后 request finished
  -> 设置 layer save event
```

`write_results` 暂存跨 layer 的 copy 结果；只有所有 layer 数据写完后，key 才应被 backend 视为完整可读对象。

### 4.6 KVCacheStoreLayerRecvingThread

这是 MemCache GVA layerwise 加载线程，依赖最多：

```text
等待上一 owner layer 的 save event
  -> 同步其 NPU save event
  -> 检查 save thread failure
  -> 构造所有 group 的 GVA/local address batch
  -> 等待 attention_start_gate
  -> 可选按 TP/layer 错开 H2D 提交
  -> 等待 external slot release（适用时）
  -> batch_copy(GVA -> local address)
  -> 设置 layer load event
```

若当前 layer 没有实际 load task，也必须正确处理 slot release 并设置 layer event，否则 forward 会永远等待一个“空任务”。

---

## 5. Backend 抽象提供哪些能力？

`backend/base.py` 的最小统一接口是：

```text
set_device()
register_buffer(ptrs, lengths)
exists(keys)
put(keys, addrs, sizes)
get(keys, addrs, sizes)
```

GVA/lease 能力作为可选协议存在：

```text
batch_get_key_info(keys)
batch_alloc(keys, sizes)
batch_add_lease(keys, ttl)
batch_remove_lease(keys)
batch_write_finish(keys, results)
```

基类默认对这些扩展方法抛出 `NotImplementedError`。因此调用方不能仅根据 `Backend` 类型就假设所有实现都支持 GVA；当前运行路径通过 backend 名称和 `use_gva_layerwise` 限制能力组合。

Scheduler client 由 `create_scheduler_client()` 创建。默认可复用 backend 类本身，但具体实现可以构造只具备 lookup 能力的轻量 client。

---

## 6. 三种 backend 的语义差异

### 6.1 MooncakeBackend

Mooncake 路径提供完整对象式 `exists/put/get`：

- 初始化 Mooncake store 与本地 segment；
- 注册 NPU buffer；
- 把多个地址片段作为同一 key 的 value；
- `put/get` 返回或检查 Mooncake transfer 状态。

它适合普通 key 和 key-based layerwise，不提供本文 GVA layerwise 所需的 alloc/lease/write-finish 接口。

### 6.2 MemcacheBackend

MemCache 同时支持普通 `put/get` 与 GVA 扩展：

- `batch_get_key_info()` 查询 key 对应 GVA/size；
- `batch_alloc()` 为新 key 分配外部 slot；
- lease 控制 slot 生命周期；
- `batch_write_finish()` 在所有数据写完后发布对象；
- 底层 `store.batch_copy` 执行 GVA 与本地 NPU 地址间 copy。

`ensure_initialized()` 是惰性初始化边界。Scheduler lookup client 与 Worker 数据 client 可以处于不同初始化阶段，因此代码不能假设构造 backend 后立即拥有所有设备资源。

### 6.3 YuanrongBackend

YuanRong 提供对象式 `exists/get/put` 和 buffer 注册，协议表面与 Mooncake 类似，但客户端配置、返回值和底层存储 API 不同。

三种实现共享 connector 语义，不意味着错误码与完成细节完全一致。调用方依赖的是统一返回约定，而不是直接访问某个 SDK 的内部对象。

---

## 7. put/get 中的地址为什么是二维列表？

backend 接口形态是：

```text
keys:  [key0, key1, ...]
addrs: [[addr00, addr01, ...], [addr10, ...], ...]
sizes: [[size00, size01, ...], [size10, ...], ...]
```

一个外部 key 可以由多个不连续内存片段组成，例如多个 layer、K/V 两个 tensor 或 hybrid state entries。二维结构保持：

```text
keys[i] <-> addrs[i] <-> sizes[i]
```

每个 `addrs[i][j]` 又与 `sizes[i][j]` 一一对应。循环移位或过滤 key 时，三组数组必须同步重排，否则会把一个对象写入另一对象的地址。

GVA batch copy 则将所有片段压平成同长 NumPy array，由 Builder 保存 request/offset 边界。

---

## 8. 同步边界有几层？

### 8.1 Python queue/event

queue 决定任务何时被线程消费；`threading.Event` 表示某 layer 的 CPU 侧处理阶段完成。它不能证明 NPU stream 上的工作已经结束。

### 8.2 torch.npu.Event

`sync_save_events[layer]` 记录 forward 写 KV 的设备顺序。发送线程在读取本地 HBM 前必须 synchronize。复用 buffer 的接收线程也可能在清除前一 layer save event 后再次同步 NPU event。

### 8.3 AttentionComputeStartGate

gate 让后台线程等待 compute stream 真正到达 attention 提交边界，而不是仅依赖 Python 调用时序。它不是 load event 的替代品：前者允许在该边界启动相应预取，后者表示新内容已经加载完成；gate event 本身也不表示 attention op 已执行完。

### 8.4 backend publish

GVA copy 成功不等于 key 已发布。保存方向还要 `batch_write_finish()`，否则其他进程可能看到已经分配但内容不完整的 slot。

因此正确顺序可能是：

```text
NPU producer event 完成
  -> D2H/remote batch copy 完成
  -> backend write finish
  -> request send finished
  -> Scheduler 允许释放 block
```

---

## 9. 错误如何传播？

### 9.1 业务失败

`get()` 返回逐项非零状态时，接收路径把失败项映射为对应本地 block id。上层可以回退已计算 token 或触发重算。

### 9.2 线程异常

地址不一致、backend 抛异常、batch copy 非零、write-finish 失败等会被线程捕获到 failure 槽。Worker 在 `wait_for_layer_load()`、`wait_for_save()` 或 `get_finished()` 的明确同步点调用 `raise_if_failed()`。

### 9.3 等待超时

layer wait 使用周期性超时是为了打印状态并检查另一线程是否失败，不是总超时后自动成功。只要依赖 event 未设置，线程就继续等待或抛出已记录异常。

### 9.4 空任务

空 task 不是错误。派生线程必须调用 `task_done()` 并设置对应 layer event，否则 queue join 或 forward wait 会死锁。

---

## 10. 完成状态如何闭环？

线程完成集合先被 `KVPoolWorker.get_finished()` 汇总为：

```text
finished_sending
finished_recving
invalid block ids
completed events
```

connector 再包装为 `KVConnectorOutput`，Scheduler 侧分别执行：

```text
update_finished_recving()
  -> 清除 loading request 状态

update_finished_sending()
  -> 清除 delayed-free 状态

update_connector_output()
  -> 聚合 Worker completed event
  -> 达到 world size 后释放 sending blocks
```

Layer event 不直接出现在 Scheduler 输出中。它只在 Worker 内保证 forward 可以逐层推进；request-level finished 才跨过 connector 边界。

---

## 11. 调试执行层的最短路径

```text
任务没有生成
  -> 先看 02_3 的 Worker 分流和 ReqMeta 条件

key 数与地址数不一致
  -> LayerBatchBuilder / ChunkedTokenDatabase

线程卡住
  -> queue、layer event、sync_save_event、attention gate、slot waiter

MemCache key 存在但读到旧数据
  -> batch_copy 结果与 batch_write_finish 顺序

load 返回失败
  -> backend 返回数组、循环移位后的 block id 配对

请求结束但 block 未释放
  -> stored request count、finished_sending、completed event 聚合
```

跨线程的完整资源安全与配置组合继续见 [05](05_data_flow_concurrency_and_config.md)；一次请求如何穿过全部文件见 [06](06_code_reading_guide.md)。
