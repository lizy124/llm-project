# 05. kv_pool 的异步传输如何保证数据可用和资源安全？

源码基线：

- vLLM Ascend：`d85e6714a09bef4d9de6b8c05e9425183d46ba23`
- vLLM：`58d3918e3ea0a544ffedadad2ba84559e9c51d8f`

源码位置：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py`

本文的问题是：`kv_pool` 中 lookup、load、save 都可能异步执行，layerwise 模式还会把一次 forward 拆成多个 layer 阶段。系统如何保证传输顺序、NPU 数据可见性、请求结束后的资源释放，以及不同配置组合下的行为一致？

本文只讨论跨组件的不变量和竞态。线程派生类、批量地址构造和 backend 方法的逐项说明见 [02_4](02_4_ascend_store_transfer_and_backend.md)，Worker 如何生成任务见 [02_3](02_3_ascend_store_worker_pipeline.md)。

---

## 0. 本文要回答的问题

```text
1. 线程、Worker、设备 event 和 Scheduler 各自保证哪一层完成？
2. layerwise load/save 和非 layerwise 路径的同步点有什么不同？
3. AttentionComputeStartGate 解决的是哪一种竞态？
4. cache_transfer_granularity 和 layerwise layout 如何影响可传输范围？
5. finished_recving、finished_sending、delayed_free 分别意味着什么？
6. 配置 backend、use_layerwise、consumer_is_to_put 时，运行路径如何变化？
```

---

## 1. 一句话回答

`kv_pool` 的异步安全依赖三层同步：传输线程自己的队列和完成状态、Worker 的 layer/event 等待、以及 NPU stream/event 对 attention 的设备级 fence；Scheduler 只有在 connector 回报合适的完成状态后，才推进请求或释放 block。

```text
任务队列完成
  != NPU tensor 已可读
  != 请求资源已经可以复用
```

这三个条件必须分别满足。

---

## 2. 线程完成为什么不足以证明数据安全？

### 2.1 线程只保证执行层完成

`KVTransferThread` 提供 queue、finished set、stored request 引用计数和异常槽。派生类完成 backend I/O 后设置 layer 或 request 状态，但该状态只说明线程职责已经完成。

```text
- request queue；
- add_request / discard_finished_requests；
- finished request 集合；
- stored request 引用计数；
- get_and_clear_finished_requests()；
- raise_if_failed()；
- 线程 run loop 和异常传播。
```

六类派生线程及各自的完成边界见 `02_4`。本文只保留一个判断原则：先确认观察到的是 queue item、layer、request 还是 distributed sending event 的完成。

### 2.2 线程完成不等于 Scheduler 状态完成

线程只知道待处理的 `ReqMeta`、`LayerTransferTask` 或 shared block data。它不决定请求是否应当运行，也不修改 vLLM 的 request status。线程完成后，把请求 id 和 block/layer 结果暴露给 `KVPoolWorker.get_finished()`，再由 connector 回传 Scheduler。

### 2.3 异常必须在同步点重新抛出

Worker 在等待 load/save 或轮询完成时调用 `raise_if_failed()`。这样后台线程的 backend 异常不会静默变成一个普通 cache miss，而会在明确的同步点被上层发现。周期性 wait timeout 的用途之一就是检查这类失败，并不表示等待已成功。

---

## 3. 非 layerwise 路径的同步边界

非 layerwise 通常把一组 layer/block 作为整体任务提交：

```text
start_load_kv()
  -> 生成整组 load task
  -> kv_recv_thread.add_request()
  -> forward 使用本地 KV 前等待必要完成状态

save_kv_layer()/wait_for_save()
  -> 生成整组 save task
  -> kv_send_thread.add_request()
  -> 等待 request queue 或 NPU event
```

`KVPoolWorker.get_finished()` 从发送和接收线程分别收集完成请求。Scheduler 看到 `finished_recving` 后，才把异步 load 对应的请求视为可继续推进；看到 `finished_sending` 后，才可以解除保存任务对资源的占用。

---

## 4. layerwise 路径为什么需要更多同步？

### 4.1 按 layer 交错

layerwise 模式允许模型执行到某一层时，传输该层所需的 KV，而不是等待整组 KV 完成。`AscendStoreConnector` 在 `use_layerwise` 时：

```text
wait_for_layer_load(layer_name)
  -> Worker 等待当前 layer 的完成 event

save_kv_layer(layer_name, ...)
  -> Worker 提交当前 layer 的 save task
```

`KVPoolWorker.process_layer_data()`、`_submit_ready_layer_loads()` 和 `layer_load_finished_events`/`layer_save_finished_events` 共同维护当前 layer 的流水。

### 4.2 layer event 的含义

layer event 表示某一层的传输阶段完成，不一定表示整个 request 的所有 layer 都完成。因此不能拿单层 event 直接当作 request-level `finished_recving`。

Worker 在等待 layer 时会循环等待 event，并周期性调用接收线程的 `raise_if_failed()`；超时不是成功，后台线程异常也不能被 event 等待吞掉。

### 4.3 layerwise 与 CUDA graph

`AscendStoreConnector.requires_piecewise_for_cudagraph()` 在启用 `use_layerwise` 时要求 piecewise graph mode。这反映了一个运行时事实：layerwise transfer 要在 forward 的 layer 边界插入 wait/save 操作，不能假设整个 forward 是一个不可切分的图。

---

## 5. AttentionComputeStartGate 解决什么问题？

### 5.1 竞态来源

某些 backend worker 线程可能与 attention 所在的 NPU compute stream 并发运行。仅仅执行到 Python 调用点，不代表 compute stream 已经到达允许预取传输提交的 attention 边界，因此需要一个设备事件把两条时间线对齐。

### 5.2 gate 的工作方式

`attention_fence.py` 中的 `AttentionComputeStartGate` 使用 NPU event 和 condition：

```text
attention worker 在提交 attention op 前 record(stream)
  -> gate 保存 event
  -> transfer worker wait()
  -> event synchronize 后确认 compute stream 已到达该边界
```

`record_attention_compute_start()` 和 `get/reset_attention_compute_start_gate()` 提供进程内的 gate 管理。它的职责是建立设备提交顺序，不是判断外部 backend key 是否存在，也不表示 attention op 已执行完成。

### 5.3 与 load finished 的区别

```text
load finished：外部数据已经传到目标地址；
attention gate：compute stream 已到达记录 event 的 attention 提交边界。
```

两者回答的是不同问题，不能用其中一个状态替代另一个。共享 buffer 的完整复用顺序还要结合上一 owner 的 layer save event 和 NPU save event 判断。

---

## 6. delayed_free 和 finished_sending 为什么要分开？

请求结束时，Scheduler 可能已经不再需要 request 的 bookkeeping，但外部 save 仍在后台进行。此时立即把 block 交回 free queue，可能导致：

```text
request A 的 block
  -> save 尚未完成
  -> 被 request B 重新分配
  -> A 的发送线程仍读取同一地址
  -> 产生覆盖或脏数据
```

因此 `KVPoolScheduler` 会把需要延迟释放的 request id 放入 metadata；Worker 的 `get_finished()` 根据 `delayed_free_req_ids` 过滤和收集发送完成集合。只有收到 `finished_sending` 后，Scheduler 才能安全推进真正的 block 释放。

抢占请求、layerwise 请求和多 batch 并发会让这个窗口更长，所以不能把 request finished 事件直接等同于 buffer free。

---

## 7. cache_transfer_granularity 和 layout 如何限制传输？

### 7.1 transfer granularity

`KVPoolScheduler._floor_to_cache_transfer_granularity()` 和 Worker 的 lookup/filter 逻辑会把命中范围向下裁剪到允许的传输粒度。原因是外部对象通常以完整 chunk/block 存储，半个 chunk 不能被当作可恢复 KV。

因此命中 token 数可能小于请求实际命中的 hash 数，最终 Scheduler 只会把对齐后的范围纳入 load 计划。

### 7.2 layerwise cache layout

`layerwise_cache_layout.py` 和 metadata 中的 layer/group 描述负责把逻辑 KV group、物理 layer、block range 映射到传输任务。需要同时考虑：

```text
- KV cache group；
- layer id/physical layer index；
- K/V 子 tensor 和 stride；
- partial block；
- Mamba/SWA 等特殊状态。
```

layout 错误通常表现为“key 命中但目标 tensor 内容不完整”，而不是简单的 backend miss。

---

## 8. 配置如何改变运行路径？

### 8.1 `backend`

Worker 根据 extra config 选择 `memcache`、`mooncake` 或 `yuanrong` 等 backend。backend 改变的是外部存储协议、地址注册和完成语义，不改变 connector 的 Scheduler/Worker 接口。

### 8.2 `use_layerwise`

启用后会同时改变：

```text
- connector 是否要求 piecewise graph；
- lookup key 和命中粒度；
- Worker 是否按 layer 提交 load/save；
- wait_for_layer_load()/save_kv_layer() 是否生效；
- layer event 和 GVA buffer 的管理方式。
```

它不是一个只影响性能的开关，而是改变了数据流拓扑。

### 8.3 `consumer_is_to_put`

在 kv_consumer 角色下，如果该开关关闭，consumer 只执行 load，不向外部 store 发布 KV。`AscendStoreConnector.save_kv_layer()` 和 `wait_for_save()` 会据此跳过保存路径。

### 8.4 `use_gva_layerwise`

该模式通常与 layerwise 和 memcache backend 组合出现，会引入外部 slot release waiter、GVA key 和 partial key。阅读时要把“layerwise 传输”与“GVA 资源释放”分开追踪。

---

## 9. 一次异步 load 的状态图

```text
Scheduler 命中并分配本地 blocks
  -> build_connector_meta()
  -> Worker start_load_kv()
  -> 生成 layer/block load tasks
  -> recv thread 入队
  -> backend.get / 设备写入
  -> layer event 或 request finished
  -> attention fence 确认可读
  -> Worker get_finished()
  -> Scheduler update_connector_output()
  -> 请求继续运行
```

任一阶段失败，都不能只通过“请求已结束”清理：必须同时处理 load error、线程异常、block 状态和可能的延迟释放。

---

## 10. 本文结论

```text
1. KVTransferThread 管理异步队列和请求完成，Worker 管理 layer/request 同步，attention fence 管理 NPU stream 顺序。
2. layerwise event 只代表单层传输完成，request-level finished 仍由 Worker 汇总。
3. finished_sending 与 delayed_free 分离，防止后台 save 读取已被复用的 block。
4. cache_transfer_granularity 和 layout 决定命中范围能否安全地变成传输任务。
5. backend、use_layerwise、consumer_is_to_put 等配置会改变数据流路径，不能只当作性能参数。
```

下一篇建议阅读 `06_code_reading_guide.md`，按文件和方法入口回顾完整阅读顺序，并把本文的异步边界对应到具体调用链。
