# Recompute CPU Offload 如何工作？

源码基线：

- vLLM Ascend：`0a97c475ab120ab2e182a358f5b1306eeddc7a8f`
- vLLM：`ba07e4a48fc951300d97eb506217dd530583dea3`

源码位置：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/recompute_cpu_offload_connector.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/worker.py`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/metadata.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`

本文的问题是：当系统不保存完整 KV tensor 时，`recompute_cpu_offload` 如何记录可恢复信息、如何在 Scheduler 和 Worker 之间传递状态，以及它如何把一次“KV 命中”转换为后续重计算？

---

## 0. 本文要回答的问题

```text
1. recompute_cpu_offload 和完整 KV offload 的恢复语义有什么不同？
2. RecomputeCPUOffloadConnectorV1 为什么同时持有 scheduler_manager 和 worker_handler？
3. Scheduler 侧如何判断请求可以使用重计算缓存？
4. RecomputeCPUOffloadMetadata 和 WorkerMetadata 分别传递什么？
5. Worker 侧是否真的把完整 KV 写入 CPU？
6. 请求抢占、恢复、结束和 reset 时，manager 的状态如何变化？
```

---

## 1. 一句话回答

`recompute_cpu_offload` 不把完整 KV 当作可直接加载的对象，而是把重计算所需的 request/token 状态保存在 CPU 侧；命中后，Scheduler 减少需要重新处理的范围，Worker 根据 metadata 维护相应状态，最终由模型重新计算缺失 KV。

可以用下面的对比记忆：

```text
完整 offload：CPU KV buffer -> NPU KV tensor -> 直接使用
重计算 offload：CPU metadata/state -> Scheduler/Worker 恢复状态 -> 重新 forward
```

---

## 2. 为什么 connector 要分 manager 和 worker？

`RecomputeCPUOffloadConnectorV1.__init__()` 根据 connector role 初始化两类对象：

```text
SCHEDULER
  -> RecomputeCPUOffloadScheduler

WORKER
  -> RecomputeCPUOffloadWorker
```

它们的资源边界和 `ascend_store` 类似，但职责更窄：

```text
Scheduler manager
  - 记录哪些 request/token 可以作为重计算缓存
  - 参与匹配、分配和抢占状态
  - 构造 scheduler metadata

Worker handler
  - 注册实际 KV cache
  - 接收 metadata 并维护 worker 侧状态
  - 执行必要的同步或辅助 transfer
  - 报告完成、pending 和失败状态
```

这里的 worker 并不等价于 `ascend_store` 的 backend worker；重计算的核心结果是“后续重新算”，不是从 CPU buffer 读回完整 KV。

---

## 3. Scheduler 侧如何判断重计算命中？

### 3.1 connector 入口

Scheduler 侧仍然通过 vLLM connector 的通用接口进入：

```text
get_num_new_matched_tokens()
update_state_after_alloc()
build_connector_meta()
update_state_before_preempt()
request_finished()
```

`RecomputeCPUOffloadConnectorV1` 将这些调用转给 `scheduler_manager`。因此 Scheduler 不需要知道重计算缓存的内部结构，只看到“可以匹配的 token 数”和 connector metadata。

### 3.2 命中含义

当 manager 报告一段 token 可匹配时，含义不是“这些 token 对应的 KV tensor 已经在本地 GPU/NPU block 中”，而是：

```text
这些 token 的必要状态仍可用于恢复，后续可以减少重复处理或按重计算路径推进。
```

具体是否减少 `num_new_tokens`、是否需要保留最后一个 token 的 forward，以及和本地 prefix cache 如何合并，仍由 vLLM Scheduler/KVCacheManager 的通用调度逻辑决定。

### 3.3 分配后的状态

`update_state_after_alloc()` 接收 request、分配得到的 `KVCacheBlocks` 和外部匹配 token 数，manager 在这里记录 request 与重计算状态之间的对应关系。

这一步的关键不是发起设备 copy，而是把：

```text
request
  + matched token count
  + 本轮本地 block 分配
  + 可能的 preempt/recompute 状态
```

合并成后续 Worker 能理解的 metadata。

---

## 4. Metadata 如何从 Scheduler 传到 Worker？

### 4.1 Scheduler metadata

`recompute_cpu_offload/metadata.py` 中的 `RecomputeCPUOffloadMetadata` 实现 vLLM 的 `KVConnectorMetadata`。它承载 scheduler 到 worker 的本轮计划，例如：

```text
- 当前请求及其重计算范围；
- 已匹配 token 或可恢复状态；
- preemption/recompute 标记；
- 需要 worker 侧处理的 request 集合。
```

`build_connector_meta()` 返回该对象，并由 vLLM 放入 SchedulerOutput。

### 4.2 Worker metadata

`RecomputeCPUOffloadWorkerMetadata` 实现 `KVConnectorWorkerMetadata`，用于 worker 之间或 worker 输出聚合。其 `aggregate()` 逻辑负责把多个 worker 的结果合并成 connector 可以上报的状态。

因此需要分清两种 metadata：

```text
RecomputeCPUOffloadMetadata
  Scheduler -> Worker：本轮要处理什么

RecomputeCPUOffloadWorkerMetadata
  Worker -> Scheduler/connector：处理到了什么状态
```

---

## 5. Worker 侧做了什么？

### 5.1 注册 cache

`RecomputeCPUOffloadWorker.register_kv_caches()` 接收 vLLM 注册的 KV cache tensor，并据此建立 worker 侧可引用的 cache/block 描述。它可能分配 CPU 侧辅助空间，但这不等于把所有 KV 内容长期复制到 CPU。

### 5.2 绑定和清理 metadata

`bind_connector_metadata()` 保存当前 step 的 `RecomputeCPUOffloadMetadata`；`clear_connector_metadata()` 在 step 完成后清理引用，避免旧请求状态泄漏到下一轮。

### 5.3 load/save 接口的含义

Worker 仍实现 vLLM connector 所需的：

```text
start_load_kv()
wait_for_layer_load()
save_kv_layer()
wait_for_save()
get_finished()
```

但这些方法的语义与完整 KV offload 不同：

```text
start_load_kv()
  -> 准备重计算相关状态，而不是必然把完整 KV 从 CPU copy 回 NPU

save_kv_layer()
  -> 保存后续重计算所需的信息或状态

get_finished()
  -> 汇总 pending/finished 状态，通知上层可以推进调度
```

具体的 transfer event、CPU block 管理和提交顺序集中在 `worker.py` 的 transfer helper 和 polling 方法中。

---

## 6. 抢占和请求结束时如何处理？

### 6.1 抢占

`handle_preemptions()` 和 scheduler 侧的 `update_state_before_preempt()` 用于告诉 connector：请求暂时离开运行队列，但它的重计算状态是否保留、转移或清理，需要由 manager 决定。

典型状态变化是：

```text
running request
  -> preempted
  -> 保留可重计算 metadata
  -> 重新进入 waiting
  -> Scheduler 再次匹配并构造 metadata
```

这和直接释放 GPU KV block 不同；重计算缓存的核心是 metadata 生命周期。

### 6.2 请求结束

`request_finished()` 和 `request_finished_all_groups()` 通知 scheduler manager 清理 request 记录。若仍有 pending transfer 或 worker-side 状态，connector 不能只依赖 request 对象销毁，而要等相应状态完成。

### 6.3 reset

`reset_cache()` 用于清空重计算缓存的全局状态。它应与 vLLM 的 prefix cache reset 区分：前者清理重计算 manager/worker 的记录，后者清理本地 KV block hash 和 block pool。

---

## 7. 重计算路径最容易误读的地方

### 7.1 “matched” 不等于 “loaded”

manager 报告匹配，只说明状态可用于恢复或减少重复工作；Worker 不一定会执行完整 KV tensor load。

### 7.2 CPU offload 不等于 CPU KV mirror

该路径的设计重点是以较低内存成本保留重计算信息，而不是在 CPU 建立一份完整、可随机读取的 KV mirror。

### 7.3 block 账本仍由 vLLM 管理

重计算 connector 可以参与 matched token 和 request 状态，但 Scheduler 侧的 KV block allocation/free 仍由 `KVCacheManager`、`BlockPool` 和通用 connector 生命周期共同完成。

---

## 8. 本文结论

```text
1. recompute_cpu_offload 保存的是恢复/重计算所需状态，不是完整 KV tensor。
2. RecomputeCPUOffloadScheduler 管理匹配和 request 生命周期，RecomputeCPUOffloadWorker 管理 worker 侧执行状态。
3. Connector 通过 matched token、metadata 和 finished 状态接入 vLLM 调度链。
4. load/save 方法仍存在，但不能按完整 KV offload 的“CPU buffer <-> NPU tensor”语义理解。
5. 抢占、结束和 reset 的关键是 metadata/manager 状态回收，而不是单纯释放一块 CPU KV buffer。
```

`ascend_store` 主线的异步数据流和配置维度见 [../ascend_store/06_concurrency_and_config.md](../ascend_store/06_concurrency_and_config.md)，可与本文的重计算路径对照阅读。
