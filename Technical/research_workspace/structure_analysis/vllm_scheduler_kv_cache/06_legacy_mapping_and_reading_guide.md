# 06. 旧架构到 V1 映射与阅读顺序

## 1. 当前代码库的总体判断

当前 `D:/lzy/project/kv_pool/code/vllm` 中，调度与 KV Cache 主线实现已经集中在 `vllm/v1`。

旧资料中常见的这些概念：

- `Sequence`
- `SequenceGroup`
- `BlockManager`
- `CacheEngine`
- `block table` 旧实现
- `LLMEngine` 旧主实现

在当前代码库中不再作为核心主线出现，或者已经被 V1 新结构替代。

## 2. 老概念与 V1 对应表

| 老架构概念 | V1 中主要对应结构 |
|---|---|
| `Sequence` | `Request` |
| `SequenceGroup` | `Scheduler` 内的 `running` / `waiting` 请求集合 |
| `BlockManager` | `KVCacheManager` + `KVCacheCoordinator` + `SingleTypeKVCacheManager` |
| `CacheEngine` | `KVCacheManager` + `BlockPool` + worker 侧 KV cache tensor 初始化 |
| logical block | `KVCacheBlock` 元数据 + scheduler block size |
| physical block | `BlockPool.blocks` 中的 `KVCacheBlock` |
| block table | worker 侧 `BlockTables` |
| sequence status | `RequestStatus` |
| prefill/decode 阶段 | `Request.num_computed_tokens` 与 `num_tokens` 的差值 |
| prefix cache | `Request.block_hashes` + `BlockHashToBlockMap` |
| swap/offload | KV Connector / distributed kv_transfer |

## 3. 为什么说 V1 已经是主线

### 3.1 Engine 入口

`D:/lzy/project/kv_pool/code/vllm/vllm/engine/llm_engine.py` 当前只是把 `LLMEngine` 指向 V1：

```text
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
LLMEngine = V1LLMEngine
```

这说明传统路径下的 `LLMEngine` 已经不是独立主实现。

### 3.2 sequence.py

`D:/lzy/project/kv_pool/code/vllm/vllm/sequence.py` 不再承担传统 `Sequence` / `SequenceGroup` 主体职责。当前调度请求对象是：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:59`：`Request`

## 4. V1 的核心对象关系

```text
EngineCore
  owns Scheduler
  owns model_executor

Scheduler
  owns RequestQueue(waiting/skipped_waiting)
  owns running list
  owns KVCacheManager
  owns EncoderCacheManager
  optionally owns KVConnector / ECConnector

KVCacheManager
  owns KVCacheCoordinator

KVCacheCoordinator
  owns BlockPool
  owns SingleTypeKVCacheManager per KV cache group

SingleTypeKVCacheManager
  owns req_to_blocks
  owns num_cached_block

BlockPool
  owns all KVCacheBlock metadata
  owns FreeKVCacheBlockQueue
  owns BlockHashToBlockMap

GPUModelRunner
  owns worker-side RequestState
  owns BlockTables
  owns actual KV cache tensors
  owns attention metadata builders / sampler / model state
```

## 5. 推荐阅读顺序

如果目标是完全看懂调度与 KV Cache，建议按下面顺序阅读。

### 第 1 阶段：先理解主链路

1. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py`
   - 重点：`_initialize_kv_caches()`、`add_request()`、`step()`、`step_with_batch_queue()`。
2. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py`
   - 重点：`Request.__init__()`、`from_engine_core_request()`、`append_output_token_ids()`、`update_block_hashes()`。
3. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py`
   - 重点：`__init__()`、`schedule()`、`update_from_output()`、`add_request()`、`finish_requests()`。
4. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py`
   - 重点：`NewRequestData`、`CachedRequestData`、`SchedulerOutput`。

读完这一阶段，应能回答：请求如何进来、如何调度、如何输出、如何结束。

### 第 2 阶段：深入 KV block 分配

5. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py`
   - 重点：`get_computed_blocks()`、`allocate_slots()`、`free()`。
6. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py`
   - 重点：coordinator 如何把多 group 分发到 single type manager。
7. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py`
   - 重点：`get_num_blocks_to_allocate()`、`add_local_computed_blocks()`、`allocate_new_blocks()`、`cache_blocks()`、`remove_skipped_blocks()`。
8. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py`
   - 重点：`BlockPool`、`BlockHashToBlockMap`、`get_cached_block()`、`cache_full_blocks()`。
9. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_utils.py`
   - 重点：`KVCacheBlock`、`FreeKVCacheBlockQueue`、block hash helpers。

读完这一阶段，应能回答：一个请求为什么能/不能分配 blocks，prefix cache 如何命中，block 如何释放和驱逐。

### 第 3 阶段：worker 与 attention

10. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py`
    - 重点：`add_requests()`、`update_requests()`、`prepare_inputs()`、`prepare_attn()`、`execute_model()`。
11. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/block_table.py`
    - 重点：`BlockTables`、`append_block_ids()`、`gather_block_tables()`、`compute_slot_mappings()`。
12. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/attn_utils.py`
    - 重点：`get_kv_cache_spec()`、`init_attn_backend()`、`_allocate_kv_cache()`、`_reshape_kv_cache()`。

读完这一阶段，应能回答：scheduler 给的 block ids 如何变成 attention kernel 的输入。

### 第 4 阶段：高级能力

13. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py`
14. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`
15. `D:/lzy/project/kv_pool/code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/*`
16. `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/encoder_cache_manager.py`

读完这一阶段，应能回答：remote KV、P/D disaggregation、多模态 encoder cache、异步 load/save 如何接入主调度链路。

## 6. 从问题出发的定位指南

### 6.1 “为什么某个请求没被调度？”

优先看：

- `Scheduler.schedule()` waiting 部分：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:624`
- `RequestQueue`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/request_queue.py`
- `KVCacheManager.allocate_slots()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:244`

常见原因：

- token budget 不足。
- running 请求数达到上限。
- KV blocks 不足。
- watermark/reserved blocks 限制。
- LoRA 数量限制。
- encoder budget/cache 不足。
- remote KV 尚未完成。
- structured output grammar 尚未准备好。

### 6.2 “为什么发生 preemption？”

优先看：

- running 调度路径：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:521`
- `_preempt_request()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1105`
- `KVCacheManager.allocate_slots()` 返回 None 的条件：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:414`

常见原因：

- free blocks 不足。
- prefix hit blocks 被 touch 后减少可用 free capacity。
- waiting/preempted admission 触发 watermark。
- external KV load 或 in-flight prefill 预留 blocks。

### 6.3 “prefix cache 为什么没命中？”

优先看：

- `Request.update_block_hashes()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:237`
- `KVCacheManager.get_computed_blocks()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:202`
- `KVCacheCoordinator.find_longest_cache_hit()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:355`
- `BlockPool.get_cached_block()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:184`

常见原因：

- prefix caching 被禁用。
- request 设置 `skip_reading_prefix_cache`。
- block 未满，不能作为 prefix hit。
- hash block size / cache salt / group id 不一致。
- sliding window 或 hybrid group 需要所有相关 group 同时满足命中。
- block 已被 eviction。

### 6.4 “block id 如何进入 attention？”

优先看：

- `SchedulerOutput.scheduled_new_reqs` / `scheduled_cached_reqs`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py:180`
- `GPUModelRunner.add_requests()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:771`
- `GPUModelRunner.update_requests()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:818`
- `BlockTables.append_block_ids()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/block_table.py:107`
- `GPUModelRunner.prepare_attn()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:1011`

路径：

```text
KVCacheBlocks -> block_ids -> SchedulerOutput -> BlockTables -> block_tables tensor + slot_mappings -> attention backend
```

### 6.5 “请求结束后资源什么时候释放？”

优先看：

- `Scheduler.update_from_output()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1463`
- `_free_request()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2046`
- `_free_request_blocks()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2077`
- `_drain_deferred_frees()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2092`
- worker `finish_requests()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:750`

普通情况立即释放。async/PP/KV connector 场景可能 deferred free。

## 7. 关键设计思想总结

1. V1 不再显式区分大阶段 prefill/decode，而是统一为“让 `num_computed_tokens` 追上当前请求长度”。
2. KV 管理不再是单一 BlockManager，而是 manager/coordinator/single-type/block-pool 四层结构。
3. Prefix cache 以 block hash 为核心，但必须带 group id。
4. Scheduler 只做决策，worker 才做 tensor 布局和执行。
5. Sliding window、Mamba、cross attention 等差异被下沉到 `SingleTypeKVCacheManager` 派生类。
6. Remote KV / offload 通过 connector 扩展 prefix cache 语义，但仍要本地 block 接收和一致性保护。
7. Deferred free 是并发批处理下保证 KV block 不被过早复用的关键机制。

## 8. 最短掌握路径

如果只想快速抓住主干，按这个最短路径阅读：

1. `Scheduler.schedule()`
2. `KVCacheManager.allocate_slots()`
3. `SingleTypeKVCacheManager.get_num_blocks_to_allocate()`
4. `BlockPool`
5. `SchedulerOutput`
6. `GPUModelRunner.add_requests()` / `update_requests()`
7. `BlockTables.append_block_ids()` / `compute_slot_mappings()`
8. `Scheduler.update_from_output()`

这 8 个点串起来，就是 vLLM V1 调度与 KV Cache 管理的主链路。
