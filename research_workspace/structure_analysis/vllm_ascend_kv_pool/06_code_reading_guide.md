# 06. 一个请求如何贯穿整个 kv_pool？

源码基线：

- vLLM Ascend：`d85e6714a09bef4d9de6b8c05e9425183d46ba23`
- vLLM：`58d3918e3ea0a544ffedadad2ba84559e9c51d8f`

源码位置：

- vLLM Scheduler：`code/vllm/vllm/v1/core/sched/scheduler.py`
- vLLM KV block 管理：`code/vllm/vllm/v1/core/kv_cache_manager.py`、`block_pool.py`
- vLLM connector 接口：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/`
- Ascend connector：`code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
- Ascend Scheduler 侧：`.../kv_pool/ascend_store/pool_scheduler.py`
- Ascend Worker 侧：`.../kv_pool/ascend_store/pool_worker.py`
- Ascend transfer data plane：`.../kv_pool/ascend_store/kv_transfer.py`

本文的问题是：不按文件孤立阅读时，一个请求如何从 vLLM 的 waiting 队列进入 `kv_pool`，经过命中判断、本地 block 分配、异步 load、模型 forward、KV save 和完成回传？阅读时应该沿哪些方法入口走，才能建立完整的运行图？

本文只串联一次请求，不重复解释每个组件内部实现。控制面、metadata、Worker 和传输线程的细节分别见 `02_1` 至 `02_4`。

---

## 0. 本文要回答的问题

```text
1. 请求第一次进入 Scheduler 时，kv_pool 何时参与？
2. 本地 prefix cache 命中和外部 pool 命中如何合并？
3. SchedulerOutput 中的 metadata 如何触发 Worker load？
4. forward 期间 layerwise/non-layerwise load 分别在哪里等待？
5. 请求产生的新 KV 何时 save 到外部 pool？
6. Worker 的 finished/error 如何影响 Scheduler 的下一轮状态？
7. 如何用最少的方法入口读完这条主链？
```

---

## 1. 一句话回答

一个请求贯穿 `kv_pool` 的主链可以压缩成：

```text
waiting request
  -> Scheduler 查询 local/external hit
  -> KVCacheManager 分配本地 blocks
  -> AscendStoreConnector 构造 metadata
  -> Worker 注册的 NPU cache 接收 load
  -> forward 使用本地 KV
  -> Worker 异步 save 新 KV
  -> finished/error 回到 Scheduler
  -> 继续运行、重算、释放或结束
```

关键是区分两个时间点：

```text
Scheduler 阶段决定“需要哪些 KV”；
Worker 阶段才执行“KV 实际如何搬运”。
```

---

## 2. 第一次调度：命中查询发生在哪里？

### 2.1 Scheduler 先查什么

请求进入 waiting 队列后，vLLM Scheduler 的调度逻辑会先通过 `KVCacheManager.get_computed_blocks()` 查询本地 prefix cache。命中的 block 会减少本轮需要真正计算的 token 数。

### 2.2 外部 pool 何时参与

如果配置了 Ascend connector，Scheduler 再调用 connector 的 `get_num_new_matched_tokens()`。在 `AscendStoreConnector` 中，该调用进入 `KVPoolScheduler`，由其生成外部 key、查询命中并按 transfer granularity 对齐。

最小路径是：

```text
Scheduler.schedule()
  -> KVCacheManager.get_computed_blocks()
  -> connector.get_num_new_matched_tokens()
  -> KVPoolScheduler 查询外部 pool
  -> 合并 local hit + external hit
```

这里的 external hit 仍然只是“可恢复 token 范围”，并不代表 NPU block 已经装载完成。

### 2.3 为什么还要 allocate_slots()

Scheduler 必须先为命中的外部 KV 分配本地 block，Worker 才知道外部数据应该写到哪里。因此命中查询后仍会回到：

```text
KVCacheManager.allocate_slots()
  -> 生成本地 KVCacheBlocks
  -> connector.update_state_after_alloc()
```

`update_state_after_alloc()` 是把外部 token 范围和本地 block ids 连接起来的关键入口。

---

## 3. SchedulerOutput 如何形成 Worker 的 load 计划？

### 3.1 构造 metadata

本轮 block 分配完成后，vLLM 调用 connector 的 `build_connector_meta()`。`KVPoolScheduler` 根据 request tracker、loading request、preempted request 和本轮 block ranges 构造 `AscendConnectorMetadata`。

metadata 通常携带：

```text
- request id；
- 要 load/save 的范围；
- 本地 block ids；
- 外部 key/hash 或 layer/group 信息；
- loading、preempted、delayed-free 请求集合。
```

### 3.2 写入 SchedulerOutput

vLLM 把 connector metadata 放入 `SchedulerOutput.kv_connector_metadata`，随本轮调度结果交给 Executor 和 Worker。Scheduler 不传输 KV 数据本身，只传输目标和计划。

```text
KVPoolScheduler
  -> AscendConnectorMetadata
  -> SchedulerOutput
  -> Worker execute_model()
```

---

## 4. Worker 如何开始 load？

### 4.1 cache registration 是前置条件

模型初始化后，Worker 侧 connector 会先执行 `register_kv_caches()`。在 `KVPoolWorker` 中，这一步建立：

```text
layer name
  -> KV cache tensor
  -> physical layer/group
  -> block address/stride
```

没有这张映射，metadata 里的 block id 和外部 key 无法转换成实际 NPU 地址。

### 4.2 start_load_kv()

Worker 收到 SchedulerOutput 后，`AscendStoreConnector.start_load_kv()` 获取 metadata 并调用 `KVPoolWorker.start_load_kv()`。Worker 随后：

```text
读取 request metadata
  -> 准备 load task
  -> 生成 key/address/size
  -> 提交 kv_recv_thread
```

非 layerwise 路径通常以 request/批次组织；layerwise 路径会进一步按当前 layer 组织任务。

### 4.3 forward 前等待

在 layerwise 模式下，模型 runner 在每个相关 layer 的 attention 前调用 `wait_for_layer_load()`；Worker 等待当前 layer event，并检查接收线程异常。

非 layerwise 模式可能在更粗的阶段等待 load 完成，但仍需要满足 NPU stream/event 的可见性条件。这里不要把“线程任务已入队”当成“attention 已可读”。

---

## 5. forward 期间如何 save 新 KV？

### 5.1 layerwise save

启用 layerwise 时，模型 runner 在 layer 边界调用 `save_kv_layer()`。`KVPoolWorker` 根据当前 layer 的 task 把新生成的 KV 地址交给发送线程。

```text
当前 layer forward 完成
  -> save_kv_layer()
  -> 构造 layer save task
  -> kv_send_thread.add_request()
  -> 下一 layer 可以继续
```

这样 load/save 可以和后续 layer forward 重叠，但每层都需要自己的 event 和失败检查。

### 5.2 非 layerwise save

非 layerwise 通常由 `wait_for_save()` 统一处理一组请求或整段 cache。Worker 会记录 NPU event，确保 source KV 不再被当前 forward 使用，再让发送线程读取地址并调用 backend.put。

### 5.3 consumer 角色

如果 `kv_role` 是 `kv_consumer` 且 `consumer_is_to_put` 关闭，connector 会跳过 save。此时 consumer 只负责从外部 pool 获取数据，不向外部 store 发布新 KV。

---

## 6. 完成状态如何回到 Scheduler？

### 6.1 Worker 汇总

`KVPoolWorker.get_finished()` 从发送/接收线程清理并汇总：

```text
done_recving
done_sending
load error block ids
KV cache events
```

它还会依据 `preempted_req_ids`、`delayed_free_req_ids` 和 loading request 集合过滤结果，避免把过期或仍被引用的任务误报告。

### 6.2 connector 输出

`AscendStoreConnector.get_finished()` 将 Worker 结果包装成 vLLM `KVConnectorOutput`。之后 Scheduler 调用 `update_connector_output()`，由 `KVPoolScheduler.update_connector_output()` 更新 request tracker。

### 6.3 Scheduler 的后续动作

```text
finished_recving
  -> 外部 KV 已经可在本地 block 中使用
  -> 请求可以离开 WAITING_FOR_REMOTE_KVS 或继续运行

finished_sending
  -> 外部保存完成
  -> 延迟释放的 block 才可能真正回收

invalid block ids
  -> 回退 num_computed_tokens
  -> 触发 recompute 或请求失败
```

最终仍由 vLLM Scheduler 决定请求是继续 forward、重新计算、被抢占还是结束。

---

## 7. 请求结束时发生什么？

请求结束并不等于所有 KV transfer 都结束。Scheduler 通过 connector 的 `request_finished()` 或 `request_finished_all_groups()` 通知 pool：

```text
request bookkeeping 可以清理
  -> 但 pending send/load 可能仍存在
  -> delayed_free 保留相关 block 约束
  -> finished_sending 后再解除外部写入占用
  -> 最终由 KVCacheManager/BlockPool 回收本地 block
```

如果请求被抢占，metadata 中的 preempted request 集合还会影响 Worker 清理哪些完成结果，防止旧请求的异步回调污染新一轮调度。

---

## 8. 推荐的最短源码阅读路线

如果只想跟读一条完整主链，建议按下面顺序：

```text
1. vllm/v1/core/sched/scheduler.py
   找 waiting 调度、allocate_slots、update_from_output

2. vllm/v1/core/kv_cache_manager.py + block_pool.py
   理解 local block 与 prefix cache 账本

3. vllm/distributed/kv_transfer/kv_connector/v1/base.py
   确认 Scheduler/Worker connector 契约

4. ascend_store/ascend_store_connector.py
   看接口如何转发到 scheduler/worker 对象

5. ascend_store/pool_scheduler.py
   跟 get_num_new_matched_tokens、update_state_after_alloc、build_connector_meta

6. ascend_store/coordinator.py + metadata.py
   理解 hybrid 命中 mask、RequestTracker 和 ReqMeta

7. ascend_store/pool_worker.py
   跟 register_kv_caches、start_load_kv、save_kv_layer、get_finished

8. ascend_store/kv_transfer.py + backend/
   最后深入线程、批量地址、backend put/get 和完成状态
```

阅读每个方法时，建议固定问四个问题：

```text
输入是 request、block、metadata 还是 tensor？
输出是 token 数、任务、地址还是完成状态？
执行发生在 Scheduler、Worker 还是后台线程？
失败后由谁决定重算、释放或报错？
```

---

## 9. 本文结论

```text
1. external hit 先参与 Scheduler 的 token/block 决策，之后才进入 Worker 的真实 load。
2. SchedulerOutput 传递的是 connector metadata，Worker 注册的 cache 映射负责把它落到 NPU 地址。
3. layerwise 模式把 load/save 拆到 layer 边界，非 layerwise 模式按更粗粒度批次推进。
4. Worker 只报告完成和失败，是否继续、重算、释放由 Scheduler 和 KVCacheManager 决定。
5. request finished、send finished 和 block free 是三个不同事件，必须沿异步链分别追踪。
```

本目录的阅读主线是：先看 vLLM/Ascend 的层次边界，再通过 `02_1` 至 `02_4` 深入 `ascend_store`，然后对比 `kv_offload` 本地搬运和 `recompute_cpu_offload` 恢复语义，最后用本文把 Scheduler、Worker、transfer thread 和 backend 串回一次请求。
