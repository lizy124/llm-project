# vLLM V1 KV Cache / KV Transfer / KVPool 问题目录

源码位置：

- `vllm/vllm/v1/core/kv_cache_manager.py`
- `vllm/vllm/v1/core/block_pool.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/sched/output.py`
- `vllm/vllm/distributed/kv_transfer/`
- `vllm/vllm/distributed/ec_transfer/`
- `vllm/vllm/v1/worker/`

这个目录按问题拆解 vLLM V1 中 KV Cache、Prefix Cache、KV Connector、外部 KV Cache / KVPool、KV load / save、invalid blocks、deferred free 等链路。重点回答：KV block 如何分配和复用，Scheduler 如何查询本地 / 外部 KV 命中，KV Connector metadata 如何从 Scheduler 传到 Worker，Worker 如何执行 KV load / save，以及请求结束后 KV 资源如何释放或延迟释放。

---

## 1. 总览文档

- [vLLM V1 KV Cache / KV Transfer / KVPool 逻辑梳理](kv_cache_transfer_overview.md)

适合第一次建立全局印象。

总览主链路：

```text
Request
  → Scheduler.schedule()
  → KVCacheManager.get_computed_blocks()
  → KV Connector get_num_new_matched_tokens()
  → KVCacheManager.allocate_slots()
  → SchedulerOutput.kv_connector_metadata
  → Executor / Worker / ModelRunner
  → KV Connector load / save
  → ModelRunnerOutput.kv_connector_output
  → Scheduler.update_from_output()
  → finished_recving / finished_sending / invalid blocks
  → free / deferred free / external KV save
```

---

## 2. 主线专题阅读顺序

### 01. KV Cache Manager 的定位

- [KVCacheManager 在 vLLM V1 里负责什么？](01_kv_cache_manager_role.md)

回答：

```text
KVCacheManager 是什么层？
它和 Scheduler / BlockPool / Worker 什么关系？
它负责 prefix cache、block 分配、block 释放中的哪一部分？
```

### 02. BlockPool 与 block 生命周期

- [BlockPool 和 KV block 生命周期如何工作？](02_block_pool_and_block_lifecycle.md)

回答：

```text
KV block 是什么？
BlockPool 如何分配、引用、释放、缓存 block？
prefix cache block、new block、external block 有什么区别？
```

### 03. Prefix Cache 命中

- [本地 prefix cache 如何命中？](03_prefix_cache_lookup.md)

回答：

```text
Scheduler 什么时候查询本地 prefix cache？
get_computed_blocks() 返回什么？
为什么 full hit 仍可能需要重算最后一个 token？
```

### 04. Scheduler 侧 KV Connector 链路

- [Scheduler 如何接入 KV Connector？](04_scheduler_kv_connector_flow.md)

回答：

```text
Scheduler 什么时候创建 connector？
get_num_new_matched_tokens() 如何影响 num_computed_tokens？
update_state_after_alloc() 做什么？
build_connector_meta() 如何进入 SchedulerOutput？
```

### 05. 外部 KV load 链路

- [外部 KV Cache 如何 load 回本地？](05_external_kv_load_flow.md)

回答：

```text
外部 KV 命中如何转成 load metadata？
load_kv_async=True 时为什么本轮不 forward？
WAITING_FOR_REMOTE_KVS 如何恢复？
finished_recving 如何回到 Scheduler？
```

### 06. 外部 KV save 链路

- [请求结束后 KV 如何保存到外部 KVPool？](06_external_kv_save_flow.md)

回答：

```text
请求结束时 Scheduler 如何通知 connector？
request_finished() / request_finished_all_groups() 做什么？
finished_sending 如何影响 block 释放？
```

### 07. Worker / ModelRunner 侧 KV Connector 链路

- [Worker / ModelRunner 如何消费 KV Connector metadata？](07_worker_kv_connector_flow.md)

回答：

```text
SchedulerOutput.kv_connector_metadata 如何进入 Worker？
Worker 侧 connector 如何 load / save KV？
ModelRunnerOutput.kv_connector_output 包含什么？
```

### 08. invalid blocks 与 recompute

- [外部 KV load 失败后如何回退重算？](08_invalid_blocks_and_recompute.md)

回答：

```text
invalid_block_ids 从哪里来？
Scheduler 如何定位失败 block？
为什么可以把 num_computed_tokens 回退到失败 block 前？
failure policy 如何决定 fail 或 recompute？
```

### 09. deferred free 与异步安全

- [deferred free 如何避免异步 KV 竞态？](09_deferred_free_and_async_safety.md)

回答：

```text
为什么不能立即释放某些 block？
deferred_frees 保存什么？
什么时候真正归还 block pool？
和 async scheduling / pipeline parallel / KV consumer 有什么关系？
```

### 10. KVPool 端到端链路

- [KVPool 端到端如何贯穿 Scheduler 和 Worker？](10_kvpool_end_to_end.md)

回答：

```text
一个请求如何查询 KVPool 命中？
命中后如何分配本地 block 并 load KV？
请求结束后如何 save KV 到 KVPool？
失败和异步完成如何闭环？
```

---

## 3. 推荐阅读路线

### 3.1 快速建立全局印象

```text
kv_cache_transfer_overview.md
  → 01_kv_cache_manager_role.md
  → 02_block_pool_and_block_lifecycle.md
  → 04_scheduler_kv_connector_flow.md
```

### 3.2 按 KV 命中和调度链路完整阅读

```text
kv_cache_transfer_overview.md
  → 03_prefix_cache_lookup.md
  → 04_scheduler_kv_connector_flow.md
  → 05_external_kv_load_flow.md
  → 07_worker_kv_connector_flow.md
  → 08_invalid_blocks_and_recompute.md
```

### 3.3 按请求结束和资源释放阅读

```text
06_external_kv_save_flow.md
  → 09_deferred_free_and_async_safety.md
  → 02_block_pool_and_block_lifecycle.md
```

### 3.4 和 Scheduler 文档联动阅读

```text
../scheduler/05_prefix_and_external_kv_hits.md
  → ../scheduler/06_kv_block_allocation_and_preemption.md
  → 04_scheduler_kv_connector_flow.md
  → 05_external_kv_load_flow.md
  → ../scheduler/08_update_after_worker_output.md
```

### 3.5 和执行层文档联动阅读

```text
../executor_worker_model_runner/04_execute_model_flow.md
  → ../executor_worker_model_runner/09_worker_kv_cache_interaction.md
  → 07_worker_kv_connector_flow.md
```

---

## 4. 文档定位

```text
README.md：
  当前目录索引和阅读路线。

kv_cache_transfer_overview.md：
  总览主文档，适合快速建立 KV Cache / KV Transfer 全局图。

01-10：
  按问题拆开的专题文档，适合逐段精读 KV cache、connector、worker 相关源码。
```

---

## 5. 最小心智模型

如果只记一条主线，可以记：

```text
KV Cache / KV Transfer = Scheduler 侧决定 KV 命中和 block 分配，Worker 侧执行 KV load / save，Scheduler 再根据 Worker 回报完成状态闭环。
```
