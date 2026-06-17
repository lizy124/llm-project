# vLLM Scheduler 与 KV Cache 技术点问答

本文基于本目录已有文档，整理调度层与 KV Cache 管理层在技术考察、面试、源码走查中可能被问到的问题，并给出参考答案。

覆盖范围：

- V1 Scheduler 总体设计
- Request 生命周期
- waiting/running/skipped_waiting 队列
- FCFS / Priority 调度策略
- token budget、running request budget、encoder budget、LoRA 限制
- prefix cache 与 block hash
- KVCacheManager / Coordinator / SingleTypeKVCacheManager / BlockPool
- KV block 分配、释放、驱逐、watermark、deferred free
- SchedulerOutput 与 Worker 边界
- GPUModelRunner、BlockTables、slot mapping、attention metadata
- KV Connector、Remote KV、Offload、P/D disaggregation
- 旧架构到 V1 的映射
- 常见调试定位

---

## 一、整体架构与职责边界

### Q1：vLLM V1 调度与 KV Cache 管理层的核心职责是什么？

答：调度层负责“谁算、算多少、占哪些 KV block、何时抢占/释放”；KV Cache 管理层负责“block 的分配、复用、prefix cache 命中、回收和驱逐”；worker/model runner 负责“把调度结果转成真实 GPU tensor、block table、slot mapping 并执行模型”。

整体链路：

```text
EngineCore / LLMEngine
  -> Request
  -> Scheduler.add_request()
  -> Scheduler.schedule()
  -> SchedulerOutput
  -> model_executor.execute_model()
  -> GPUModelRunner.execute_model()
  -> ModelRunnerOutput
  -> Scheduler.update_from_output()
  -> EngineCoreOutputs
```

---

### Q2：Scheduler 位于 vLLM 哪一层？

答：Scheduler 位于 EngineCore 与 Worker/ModelRunner 之间。

它接收 EngineCore 中的 Request，决定每一轮应该调度哪些请求、每个请求计算多少 token、需要哪些 KV blocks，然后输出 SchedulerOutput 给 worker 执行。

它不直接执行 tensor 计算，也不关心 CUDA tensor 的具体 layout。

---

### Q3：Scheduler 具体负责哪些事情？

答：Scheduler 负责：

1. 管理请求集合：`requests`、`waiting`、`skipped_waiting`、`running`；
2. 根据策略选择请求：FCFS 或 priority；
3. 控制预算：最大 running 请求数、最大 scheduled token 数、encoder budget、LoRA 数量限制、KV block 容量、水位线等；
4. 查询 prefix cache 命中；
5. 请求 KV block 分配；
6. 处理 preemption、remote KV load、deferred free、finish/abort；
7. 构建 SchedulerOutput；
8. 接收 ModelRunnerOutput，更新请求状态、输出 token、释放资源。

---

### Q4：Scheduler 不负责什么？

答：Scheduler 只做调度决策，不执行 tensor 计算。

它不直接负责：

- CUDA tensor 具体布局；
- attention kernel 细节；
- logits 计算；
- sampler 内部实现；
- KV cache tensor 的真实 GPU 内存写入；
- block table tensor / slot mapping tensor 的计算。

这些由 worker、GPUModelRunner、AttentionBackend 和底层 kernel 负责。

---

### Q5：V1 的调度思想和传统 prefill/decode 两阶段有什么区别？

答：V1 不再把调度器简单分成 prefill scheduler 和 decode scheduler，而是用 `Request.num_computed_tokens` 统一表达进度。

核心思想：每一轮调度都尝试让请求的 `num_computed_tokens` 追上当前需要计算的位置。

这套统一模型覆盖：

- chunked prefill；
- decode；
- prefix cache；
- speculative decoding；
- remote KV load；
- 未来 jump decoding 等。

如果 `num_computed_tokens < num_prompt_tokens`，就是 prefill 相关计算；如果 prompt 已经计算完，新增 token 通常来自上一轮 sampled token 或 draft tokens，就是 decode。

---

### Q6：当前 V1 中旧架构概念如何映射？

答：常见映射如下：

| 老架构概念 | V1 中主要对应结构 |
|---|---|
| `Sequence` | `Request` |
| `SequenceGroup` | Scheduler 内的 `running` / `waiting` 请求集合 |
| `BlockManager` | `KVCacheManager` + `KVCacheCoordinator` + `SingleTypeKVCacheManager` |
| `CacheEngine` | `KVCacheManager` + `BlockPool` + worker 侧 KV cache tensor 初始化 |
| logical block | `KVCacheBlock` 元数据 + scheduler block size |
| physical block | `BlockPool.blocks` 中的 `KVCacheBlock` |
| block table | worker 侧 `BlockTables` |
| sequence status | `RequestStatus` |
| prefill/decode 阶段 | `Request.num_computed_tokens` 与 `num_tokens` 的差值 |
| prefix cache | `Request.block_hashes` + `BlockHashToBlockMap` |
| swap/offload | KV Connector / distributed kv_transfer |

---

### Q7：为什么说当前代码库中 V1 已经是主线？

答：因为当前 `vllm/engine/llm_engine.py` 中的传统 `LLMEngine` 已经指向 V1 实现：

```text
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
LLMEngine = V1LLMEngine
```

同时传统 `Sequence` / `SequenceGroup` 不再是调度主对象，V1 的 `Request`、`Scheduler`、`KVCacheManager` 成为主线。

---

## 二、Scheduler 初始化与核心字段

### Q8：Scheduler 初始化时保存哪些配置？

答：主要保存：

- `self.vllm_config`
- `self.scheduler_config`
- `self.cache_config`
- `self.lora_config`
- `self.kv_cache_config`
- `self.parallel_config`

这些配置决定调度预算、并发、prefix cache、KV connector、pipeline parallel、data parallel、LoRA 限制等行为。

---

### Q9：Scheduler 有哪些核心调度约束？

答：关键约束包括：

- `max_num_running_reqs`
  - 来源于 `scheduler_config.max_num_seqs`。
  - 控制最多有多少请求处于 running。

- `max_num_scheduled_tokens`
  - 优先使用 `scheduler_config.max_num_scheduled_tokens`，否则使用 `max_num_batched_tokens`。
  - 控制每轮最多调度多少 token。

- `max_model_len`
  - 模型最大长度限制。

- `num_sampled_tokens_per_step`
  - 普通生成模型通常为 1，diffusion 模型可能为 0。

---

### Q10：Scheduler 中有哪些请求集合？

答：主要有：

- `requests: dict[str, Request]`
  - 全局请求表，request_id 到 Request。

- `waiting`
  - 普通待调度队列。

- `skipped_waiting`
  - 因结构化输出 grammar、remote KV、streaming input 或临时约束而暂时跳过的 waiting 请求队列。

- `running`
  - 已经进入运行态、持有 KV/cache 状态的请求。

- `finished_req_ids`
  - 上一步与当前步之间完成的请求，用来通知 worker 清理状态。

---

### Q11：Scheduler 中有哪些 KV 相关字段？

答：主要包括：

- `kv_cache_manager`
  - Scheduler 与 KV cache 管理层之间的主接口。

- `connector`
  - scheduler 侧 KV connector，处理 remote KV、offload、P/D disaggregation 等。

- `defer_block_free`
  - 在 overlapping batches、async scheduling、pipeline parallel 或 consumer KV connector 场景中，避免 block 过早释放复用。

- `finished_recving_kv_req_ids`
  - 已完成 remote KV receive 的请求集合。

- `failed_recving_kv_req_ids`
  - remote KV receive 失败的请求集合。

---

### Q12：Scheduler 中 encoder/multimodal 相关字段有什么作用？

答：主要字段包括：

- `encoder_cache_manager`
- `max_num_encoder_input_tokens`
- `ec_connector`

它们用于处理 encoder-decoder、多模态 encoder cache、EC connector、encoder compute budget 等。

---

## 三、请求队列与调度策略

### Q13：vLLM V1 支持哪些请求调度策略？

答：主要支持两种：

```text
FCFS = "fcfs"
PRIORITY = "priority"
```

FCFS 是先到先服务；Priority 使用优先级排序，priority 值越小越先处理，priority 相同时 arrival_time 更早的请求先处理。

---

### Q14：FCFSRequestQueue 如何工作？

答：FCFSRequestQueue 基于 `deque`：

- `add_request()`：append 到尾部；
- `pop_request()`：popleft；
- `peek_request()`：查看队首；
- `prepend_request()`：appendleft。

适合普通先到先服务。

---

### Q15：PriorityRequestQueue 如何工作？

答：PriorityRequestQueue 基于 `heapq`。

排序规则：

1. priority 值越小越先处理；
2. priority 相同，则 arrival_time 更早的请求先处理。

---

### Q16：waiting 和 skipped_waiting 有什么区别？

答：

- `waiting` 是普通待调度队列。
- `skipped_waiting` 是暂时不能继续调度的请求队列。

`skipped_waiting` 不是失败队列，而是等待某些条件满足，例如：

- `WAITING_FOR_REMOTE_KVS`：等待 remote KV load；
- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`：等待结构化输出 grammar；
- `WAITING_FOR_STREAMING_REQ`：等待 streaming request 新输入。

---

### Q17：Scheduler 如何在 waiting 和 skipped_waiting 之间选择？

答：通过 `_select_waiting_queue_for_scheduling()`。

规则：

- FCFS 模式：优先尝试 `skipped_waiting`，再尝试 `waiting`。
- PRIORITY 模式：比较两个队列的队首，选择优先级更高的请求所在队列。

---

### Q18：blocked waiting 状态有哪些？

答：常见 blocked waiting 状态包括：

- `WAITING_FOR_REMOTE_KVS`
- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`
- `WAITING_FOR_STREAMING_REQ`

这些状态会由 `_is_blocked_waiting_status()` 判断。如果条件未满足，请求继续留在 skipped_waiting。

---

## 四、Scheduler.schedule 主流程

### Q19：`Scheduler.schedule()` 每轮大致做什么？

答：每轮流程：

```text
1. current_step += 1
2. 初始化本轮临时容器
3. token_budget = max_num_scheduled_tokens
4. kv_cache_manager.new_step_starts()
5. 先调度 running 请求
6. 再调度 waiting / skipped_waiting 请求
7. 计算 common prefix blocks
8. 构建 SchedulerOutput
9. 构建 KV connector metadata / EC connector metadata
10. _update_after_schedule()
11. 返回 SchedulerOutput
```

---

### Q20：schedule() 中有哪些本轮临时容器？

答：包括：

- `scheduled_new_reqs`
- `scheduled_resumed_reqs`
- `scheduled_running_reqs`
- `preempted_reqs`
- `req_to_new_blocks`
- `num_scheduled_tokens`
- `scheduled_spec_decode_tokens`
- `scheduled_encoder_inputs`

这些用于记录本轮调度结果，并最终构造 SchedulerOutput。

---

### Q21：为什么 Scheduler 先调度 running 请求，再调度 waiting 请求？

答：running 请求已经持有 KV blocks 和 worker 侧状态，继续推进 running 请求可以：

1. 减少重复 prefill；
2. 减少缓存抖动；
3. 避免已占资源的请求长期停滞；
4. 更好维持 continuous batching 的稳定性。

waiting 请求只有在没有本轮 preempted 请求、running 数未达上限、系统未暂停等条件满足时才继续调度。

---

### Q22：running 请求调度的核心逻辑是什么？

答：running 路径主要做：

1. 遍历 `self.running`；
2. 跳过暂时不能继续 decode 的请求；
3. 计算本轮需要新增计算的 token 数；
4. 应用 long prefill threshold；
5. 应用 token budget；
6. 不超过 max model len；
7. 调度 encoder inputs；
8. 处理 Mamba block-aligned split；
9. 调 `kv_cache_manager.allocate_slots()` 分配新增 KV blocks；
10. 分配失败时触发 preemption。

---

### Q23：running 请求的 `num_new_tokens` 如何计算？

答：核心公式是：

```text
num_new_tokens = request.num_tokens_with_spec
               + request.num_output_placeholders
               - request.num_computed_tokens
```

含义：请求当前需要计算到的位置，减去已经计算完成的位置。

---

### Q24：waiting 请求调度的核心流程是什么？

答：waiting 路径主要做：

1. 确认本轮没有 preempted 请求；
2. 检查 waiting / skipped_waiting 是否有请求；
3. 检查 running 数量是否达到 `max_num_running_reqs`；
4. 从合适队列取队首请求；
5. 如果请求处于 blocked waiting 状态，尝试 promote；
6. 检查 LoRA 限制；
7. 如果是新请求，查询本地 prefix cache；
8. 如果有 KV connector，查询外部 KV 命中；
9. 计算本轮需要计算的 token 数；
10. 分配 KV slots；
11. 如果是 async remote KV load，放入 `WAITING_FOR_REMOTE_KVS`；
12. 否则进入 running，并构造 new/resumed request 输出数据。

---

### Q25：waiting 请求首次调度时为什么要查 prefix cache？

答：如果 request 的前缀已经被之前请求计算过，就可以复用已有 KV blocks，避免重复 prefill。

首次调度时，如果 `request.num_computed_tokens == 0`，Scheduler 会调用：

```text
kv_cache_manager.get_computed_blocks(request)
```

得到本地 prefix cache 命中的 blocks 和 token 数。

如果启用 KV connector，还会查询外部 KV 命中。

---

### Q26：SchedulerOutput 是什么？

答：SchedulerOutput 是 Scheduler 和 Worker/ModelRunner 之间的主通信对象。

它描述本轮执行计划，包括：

- 哪些新请求需要完整加入 worker；
- 哪些已有请求只需发送差量；
- 每个请求本轮执行多少 token；
- 新增 block ids；
- finished / preempted 请求；
- spec decode tokens；
- encoder inputs；
- KV connector metadata；
- 需要清零的新 block ids。

---

### Q27：SchedulerOutput 中 `scheduled_new_reqs` 和 `scheduled_cached_reqs` 有什么区别？

答：

- `scheduled_new_reqs`
  - 第一次被调度的请求。
  - worker 需要完整信息：prompt tokens、采样参数、block ids、LoRA request、多模态特征等。

- `scheduled_cached_reqs`
  - 已经在 worker 侧有状态的请求。
  - 只需要发送差量：req ids、新 token ids、新 block ids、num computed tokens、num output tokens 等。

---

### Q28：SchedulerOutput 中 `finished_req_ids` 和 `preempted_req_ids` 有什么作用？

答：

- `finished_req_ids`
  - 通知 worker 删除已完成请求状态。

- `preempted_req_ids`
  - 通知 worker 删除或重置被抢占请求状态。
  - 对 v2 runner 很重要，因为 preempted 请求释放了 KV blocks，后续需要重算。

---

### Q29：SchedulerOutput 中 `new_block_ids_to_zero` 的作用是什么？

答：它告诉 worker 哪些新分配的 KV blocks 需要清零。

目的：避免新 block 中残留的 NaN 或旧数据污染 attention / SSM 计算。

worker 侧在 `update_requests()` 中调用 GPU block zeroer 清零这些 blocks。

---

### Q30：`_update_after_schedule()` 做什么？

答：它在构建完 SchedulerOutput 后执行，主要做：

1. 将每个被调度请求的 `request.num_computed_tokens` 增加本轮 `num_scheduled_tokens`；
2. 如果启用 deferred free，记录 `request.last_sched_seq`；
3. 更新 `request.is_prefill_chunk`；
4. 判断是否存在 structured output 请求；
5. 从 `_inflight_prefills` 移除不再 prefilling 的请求；
6. 清空 `finished_req_ids`。

关键点：SchedulerOutput 携带的是本轮执行前 worker 需要的信息，而 scheduler 内部会提前推进 `num_computed_tokens`，便于下一轮继续调度 chunked prefill。

---

## 五、Request 生命周期

### Q31：Request 在 V1 中的定位是什么？

答：Request 是 V1 中调度和执行的基础单位，替代旧架构中的 Sequence / SequenceGroup 主体地位。

它保存请求的输入 token、输出 token、状态、优先级、prefix cache hash、KV 调度进度、LoRA、多模态、streaming 等信息。

---

### Q32：Request 从哪里创建？

答：通过：

```text
Request.from_engine_core_request()
```

它从 EngineCoreRequest 拷贝：

- request_id；
- client_index；
- prompt_token_ids；
- prompt_embeds；
- mm_features；
- sampling_params / pooling_params；
- arrival_time；
- lora_request；
- cache_salt；
- priority；
- trace_headers；
- resumable；
- abort_immediately 等。

---

### Q33：Request 中哪些字段和调度强相关？

答：关键字段包括：

- `status`：请求状态；
- `priority`：priority scheduling 使用；
- `arrival_time`：FCFS 或同 priority 排序依据；
- `num_computed_tokens`：核心进度指针；
- `spec_token_ids`：spec decoding draft tokens；
- `block_hashes`：prefix cache hash；
- `skip_reading_prefix_cache`：是否跳过 prefix cache；
- `is_prefill_chunk`：是否处于 chunked prefill；
- `num_preemptions`：抢占次数；
- `last_sched_seq`：deferred free fencing 使用。

---

### Q34：`num_computed_tokens` 为什么重要？

答：它是 V1 Scheduler 的核心进度指针。

Scheduler 每轮通过它判断请求还有多少 token 需要计算：

```text
需要计算的位置 - num_computed_tokens
```

它统一表达 prefill、decode、chunked prefill、prefix cache 命中、spec decode 回退等场景。

---

### Q35：Request 的 token 字段有哪些？

答：主要包括：

- `prompt_token_ids`
- `prompt_embeds`
- `prompt_is_token_ids`
- `num_prompt_tokens`
- `_output_token_ids`
- `_all_token_ids`
- `output_token_ids`
- `all_token_ids`

`_all_token_ids = prompt tokens + output tokens`。

每次 `append_output_token_ids()` 追加生成 token 后，会调用 `update_block_hashes()` 更新 prefix cache hash。

---

### Q36：Request 加入 Scheduler 时发生什么？

答：`Scheduler.add_request()` 分两种情况：

1. 新请求：
   - 如果 resumable，初始化 streaming queue；
   - 调 `_enqueue_waiting_request(request)`；
   - 加入 `self.requests`；
   - 如果有 connector，调用 `connector.on_new_request(request)`；
   - 记录 queued event。

2. 重复 request_id / streaming session：
   - 对 resumable request，把新输入片段转成 `StreamingUpdate`；
   - 如果请求正在等待 streaming input，则立即更新 session；
   - 否则把 update 放入已有 session queue。

---

### Q37：新请求如何从 waiting 进入 running？

答：waiting 调度阶段：

1. 选择 waiting 或 skipped_waiting 队列；
2. 检查 blocked 状态；
3. 查询本地 prefix cache；
4. 查询外部 KV cache；
5. 计算本轮 num_new_tokens；
6. 分配 KV slots；
7. 如果需要 async remote KV load，进入 `WAITING_FOR_REMOTE_KVS`；
8. 否则加入 `running`；
9. 记录 scheduled_new_reqs 或 scheduled_resumed_reqs；
10. 记录 block ids 和 num_scheduled_tokens；
11. request.status 变成 RUNNING。

---

### Q38：running 请求继续调度时 worker 收到的是完整请求还是差量？

答：通常是差量。

running 请求已经在 worker 侧有 request state 和 block table，因此 SchedulerOutput 中 `scheduled_cached_reqs` 只发送：

- req id；
- 新 token ids；
- 新 block ids；
- num_computed_tokens；
- num_output_tokens 等。

---

### Q39：请求完成时 Scheduler 做什么？

答：当请求 stopped：

1. 记录 finish reason；
2. 调 `_handle_stopped_request(request)`；
3. 如果真正 finished，调用 `_free_request(request)`；
4. 从 running 或 waiting 队列移除；
5. 构造 EngineCoreOutput 返回前端。

对于 resumable request，如果 streaming queue 有下一段输入，会更新 session 并重新入队；否则进入 `WAITING_FOR_STREAMING_REQ`。

---

### Q40：Abort 生命周期如何处理？

答：外部 abort 由 EngineCore 调用 Scheduler 的 `finish_requests()`。

Scheduler 会：

1. 收集 running 和 waiting 中要移除的请求；
2. 批量从队列移除；
3. 设置 finished status；
4. 调 `_free_request()` 释放资源。

如果请求正在 `WAITING_FOR_REMOTE_KVS`，可能需要 delay free，避免 connector 资源状态不同步。

---

### Q41：请求有哪些常见状态？

答：常见状态包括：

- `WAITING`
- `RUNNING`
- `PREEMPTED`
- `WAITING_FOR_REMOTE_KVS`
- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`
- `WAITING_FOR_STREAMING_REQ`
- `FINISHED_STOPPED`
- `FINISHED_ABORTED`
- `FINISHED_ERROR`

其中 finished 状态最终都会走 `_free_request()`。

---

## 六、Prefill、Decode、Chunked Prefill 与 Spec Decode

### Q42：V1 如何统一 prefill 和 decode？

答：通过 `num_computed_tokens` 与当前请求 token 数的差值统一表达。

- 如果 `num_computed_tokens < num_prompt_tokens`，说明还在 prefill。
- 如果 prompt 已经计算完，输入通常来自 sampled token 或 draft tokens，就是 decode。

Scheduler 不需要维护两个完全分离的大阶段。

---

### Q43：worker 侧如何判断某请求是否在 prefill？

答：GPUModelRunner 在 `prepare_inputs()` 中通过类似：

```text
num_computed_prefill_tokens_np < prefill_len_np
```

得到 `is_prefilling_np`。

如果存在 prefill 请求，会调用 `prepare_prefill_inputs()`。

---

### Q44：decode 输入来自哪里？

答：当 prompt 已经计算完，decode 输入通常来自：

- 上一轮 sampled token；
- speculative decoding draft tokens。

worker 侧会调用类似 `combine_sampled_and_draft_tokens()` 的逻辑把 sampled token 和 draft token 合并成当前输入。

---

### Q45：chunked prefill 是如何表达的？

答：如果 prompt 很长，Scheduler 可以只调度一部分 prompt tokens。

`_update_after_schedule()` 会设置：

```text
request.is_prefill_chunk = request.num_computed_tokens < (request.num_tokens + request.num_output_placeholders)
```

表示当前请求还处于 chunked prefill 的中间段。

---

### Q46：spec decode 对调度有什么影响？

答：spec decode 会引入 draft tokens 和 lookahead tokens。

影响包括：

- SchedulerOutput 中携带 `scheduled_spec_decode_tokens`；
- KVCacheManager.allocate_slots 需要 `num_lookahead_tokens` 预留 KV slots；
- update_from_output 中如果 draft tokens 被 reject，需要回退 `request.num_computed_tokens`；
- worker 侧需要处理 sampled token 与 draft token 合并。

---

### Q47：spec decode rejected tokens 为什么要回退 `num_computed_tokens`？

答：因为被 reject 的 draft tokens 不应被视为已经有效计算完成的请求进度。

Scheduler 在 `update_from_output()` 中根据 ModelRunnerOutput 判断哪些 draft token 被 reject，并相应回退 `request.num_computed_tokens`，确保后续重新计算正确位置。

---

## 七、KV Cache 管理分层

### Q48：V1 KV Cache 管理分成哪几层？

答：分成四层：

```text
Scheduler
  -> KVCacheManager
      -> KVCacheCoordinator
          -> SingleTypeKVCacheManager per KV cache group/type
              -> BlockPool
                  -> KVCacheBlock / FreeKVCacheBlockQueue / BlockHashToBlockMap
```

---

### Q49：Scheduler 层和 KVCacheManager 层的职责区别是什么？

答：

Scheduler 关心：

- 请求能否调度；
- 请求已有多少 computed tokens；
- 新增多少 tokens 要执行；
- 是否命中 prefix cache；
- 是否需要外部 KV connector；
- 分配出来的 block ids 是什么。

KVCacheManager 负责隐藏 coordinator、single type manager、block pool 的细节，并提供：

- prefix cache 查询；
- KV slots 分配；
- 请求 blocks 释放；
- block ids 获取；
- cache blocks 提交；
- prefix cache reset；
- KV cache events。

---

### Q50：KVCacheCoordinator 的作用是什么？

答：KVCacheCoordinator 负责协调多个 KV cache group。

一个模型可能同时有：

- full attention group；
- sliding window group；
- chunked local attention group；
- MLA group；
- Mamba state group；
- cross attention group。

这些 group 的 block size、缓存行为、窗口规则、是否保留全部历史可能不同。Coordinator 统一包装这些 group，对上提供一致接口。

---

### Q51：SingleTypeKVCacheManager 的作用是什么？

答：每个 SingleTypeKVCacheManager 管理一种 cache spec。

它负责：

- 每个 request 的 block 列表：`req_to_blocks`；
- 已缓存 block 数：`num_cached_block`；
- 新分配 block ids：`new_block_ids`；
- 不同 attention 类型的 prefix hit、跳过窗口、回收逻辑。

不同派生类处理不同 cache 语义。

---

### Q52：BlockPool 的作用是什么？

答：BlockPool 是所有实际 block 元数据的池。

它负责：

- 创建 KVCacheBlock 列表；
- 维护 free block 双向链表；
- 维护 prefix cache hash -> block 映射；
- 分配、释放、touch、evict；
- 维护 null block；
- 发出 KV cache events。

---

### Q53：不同 SingleTypeKVCacheManager 派生类分别处理什么？

答：

| Manager | 语义 |
|---|---|
| `FullAttentionManager` | 完整保留上下文 |
| `SlidingWindowManager` | 只保留滑动窗口内 blocks，窗口外释放或用 null block 占位 |
| `ChunkedLocalAttentionManager` | 局部块注意力 |
| `MambaManager` | 状态空间模型 cache，语义不同于 KV block |
| `CrossAttentionManager` | encoder-decoder cross attention 的静态分配 |

---

### Q54：为什么需要 HybridKVCacheCoordinator？

答：因为一个模型可能混合多种 cache group，例如 full attention + Mamba + sliding window。

Hybrid 模式下，不同 group 的 block size、命中规则、释放规则不同。命中可能要按 group 分别计算，并取满足所有 group 的共同可用位置。

---

## 八、KVCacheBlocks 与 KVCacheManager

### Q55：KVCacheBlocks 是什么？

答：KVCacheBlocks 是 Scheduler 和 KV manager 之间的结果容器。

字段：

```text
blocks: tuple[Sequence[KVCacheBlock], ...]
```

外层 tuple 对应 KV cache group，内层 sequence 是该 group 下的 block 列表。

它可以转成 worker 使用的 block ids。

---

### Q56：为什么 KVCacheBlocks 的外层是 group，而不是 token block？

答：因为未来不同 KV cache group 可能有不同 block_size。当前大多数情况下 group 数和 block 数对齐，但代码为 hybrid/multi-group 场景保留扩展。

---

### Q57：KVCacheBlocks 有哪些关键方法？

答：包括：

- `get_block_ids()`：转成 worker 能使用的 block id 列表；
- `get_unhashed_block_ids()`：找还没有 hash 的 block；
- `get_unhashed_block_ids_all_groups()`：多 group 版本；
- `new_empty()`：创建同 group 数的空 blocks；
- `__add__()`：合并两组 blocks。

---

### Q58：KVCacheManager 暴露哪些核心方法给 Scheduler？

答：主要包括：

- `get_computed_blocks(request)`：查询 prefix cache；
- `allocate_slots(...)`：为请求分配 slots/blocks；
- `free(request)`：释放请求 blocks；
- `get_blocks(request_id)` / `get_block_ids(request_id)`：获取请求当前 blocks；
- `cache_blocks(request, num_computed_tokens)`：提交可缓存 block；
- `reset_prefix_cache()`：重置 prefix cache；
- `take_events()`：取 KV cache events。

---

### Q59：KVCacheManager 的 `usage` 表示什么？

答：它直接返回 `block_pool.get_usage()`，即 BlockPool 视角下的 block 使用率。

---

## 九、Prefix Cache 与 Block Hash

### Q60：prefix cache 的基本原理是什么？

答：prefix cache 基于 block hash。

请求的 token 被切成 block，并对完整 block 计算 hash。如果后续请求拥有相同前缀 block hash，就可以复用已经计算好的 KV blocks，避免重复 prefill。

生命周期：

```text
Request 创建 / token 追加
  -> update_block_hashes()
  -> Scheduler 调度 waiting 请求
  -> KVCacheManager.get_computed_blocks()
  -> Coordinator.find_longest_cache_hit()
  -> BlockPool.get_cached_block()
  -> 命中 blocks 被 touch 并加入 req_to_blocks
  -> 新计算完成的 full blocks 通过 cache_blocks() 插入 prefix cache
```

---

### Q61：`get_computed_blocks()` 做什么？

答：它查询本地 prefix cache 命中。

流程：

1. 如果 prefix caching 禁用，或请求要求跳过读取 prefix cache，则返回空；
2. 设置 `max_cache_hit_length = request.num_tokens - 1`；
3. 调 coordinator 的 `find_longest_cache_hit(request.block_hashes, max_cache_hit_length)`；
4. 记录 prefix cache stats；
5. 返回命中的 KVCacheBlocks 与命中 token 数。

---

### Q62：为什么 prefix cache 即使全部 prompt 命中也要重新计算最后一个 token？

答：因为即使 prompt 的 KV 已经命中，也通常需要重新计算最后一个 token 以获得 logits。

所以 `get_computed_blocks()` 设置：

```text
max_cache_hit_length = request.num_tokens - 1
```

---

### Q63：prefix cache 命中为什么必须是完整 block？

答：因为 prefix cache 以 block hash 为单位。只有完整 block 才有稳定 hash 并能保证 KV 内容完整一致。

尾部不完整 block 不能直接作为 computed block 复用，否则可能产生不完整或错误的 KV 状态。

---

### Q64：block hash 为什么要带 group id？

答：同样 token 内容在不同 KV cache group 中可能对应不同 layout、attention 语义、sliding window 规则或 cache 类型，因此不能跨 group 复用。

所以 hash 会组合：

```text
block_hash + group_id
```

形成 `BlockHashWithGroupId`。

---

### Q65：Request 的 block_hashes 如何更新？

答：Request 初始化时会调用 `update_block_hashes()`。

每次生成新 token 后，`append_output_token_ids()` 也会调用 `update_block_hashes()`，保证输出 token 形成的新完整 block 也可以进入 prefix cache。

---

### Q66：BlockHashToBlockMap 为什么 value 可能是单 block 或 dict？

答：为了减少内存和 GC 开销。

- 大多数 hash 只对应一个 block，用单对象存储；
- 如果出现多个相同 hash 的 block，再升级成 dict。

它不去重已缓存 blocks，因为 block table 需要保持 append-only 语义，不能因为发现相同内容就替换已分配 block id。

---

### Q67：prefix cache 为什么可能没有命中？

答：常见原因：

- prefix caching 被禁用；
- request 设置了 `skip_reading_prefix_cache`；
- block 未满，不能作为 prefix hit；
- hash block size 不一致；
- cache salt 不一致；
- group id 不一致；
- sliding window 或 hybrid group 需要所有相关 group 同时命中；
- cached block 已被 eviction。

---

## 十、KV block 分配与 allocate_slots

### Q68：`KVCacheManager.allocate_slots()` 是什么？

答：它是 KV block 分配的核心方法。

Scheduler 调用它为某个请求在本轮新增计算 token、prefix hit token、external KV token、lookahead token 等分配或绑定本地 KV blocks。

---

### Q69：allocate_slots 的核心输入参数有哪些？

答：主要参数包括：

- `request`：目标请求；
- `num_new_tokens`：本轮新增计算 token 数；
- `num_new_computed_tokens`：新命中的本地 prefix cache token 数；
- `new_computed_blocks`：对应本地命中的 blocks；
- `num_lookahead_tokens`：spec decode 额外预留 tokens；
- `num_external_computed_tokens`：connector 外部命中的 tokens；
- `delay_cache_blocks`：是否延迟 cache blocks；
- `num_encoder_tokens`：cross-attention encoder tokens；
- `full_sequence_must_fit`：admission gate；
- `reserved_blocks`：为其他 in-flight prefills 预留的 free blocks；
- `has_scheduled_reqs`：是否已有调度请求，用于决定 watermark 是否生效。

---

### Q70：allocate_slots 中 token 区间如何理解？

答：源码中的核心区间是：

```text
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
```

含义：

- `comp`：请求之前已经 computed 的 tokens；
- `new_comp`：本轮新命中的本地 prefix cache tokens；
- `ext_comp`：外部 connector 已经缓存的 tokens；
- `new`：本轮要实际计算的 tokens；
- `lookahead`：spec decode 预留 tokens。

---

### Q71：allocate_slots 的三阶段是什么？

答：三阶段：

1. 释放 `comp` 中不再需要的 blocks，并检查 free blocks 是否足够；
2. 处理 prefix tokens：local computed + external computed；
3. 为新计算 tokens 和 lookahead tokens 分配新 blocks。

---

### Q72：allocate_slots 如何计算 total computed tokens？

答：

```text
num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens

total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
```

其中 external computed tokens 来自 KV connector 的远端命中。

---

### Q73：allocate_slots 如何判断需要多少 token 拥有 slot？

答：它计算：

```text
num_tokens_need_slot = min(
    total_computed_tokens + num_new_tokens + num_lookahead_tokens,
    max_model_len,
)
```

即本地需要为已 computed、本轮新算、spec lookahead 等 token 准备 KV slot，但不能超过模型最大长度。

---

### Q74：allocate_slots 的容量检查是什么？

答：核心检查是：

```text
required_blocks = num_blocks_to_allocate + watermark_blocks
required_blocks <= block_pool.get_num_free_blocks() - reserved_blocks
```

如果不满足，则返回 None，表示分配失败。

running 路径分配失败通常会触发 preemption；waiting 路径分配失败则停止或跳过 waiting 调度。

---

### Q75：`reserved_blocks` 的作用是什么？

答：reserved_blocks 用于为其他 in-flight prefills 保留 free blocks，尤其在 async remote KV load 场景中避免死锁。

因为 async KV load 会占用 blocks 但没有 forward progress，如果它占光 free blocks，其他正在 prefill 的请求可能无法完成，导致系统卡住或频繁抢占。

---

### Q76：`delay_cache_blocks` 的作用是什么？

答：当 remote KV load 或 P/D disaggregation 场景中，分配出来的 blocks 还没有真实 KV 内容，不能立即写入 prefix cache。

如果 `delay_cache_blocks=True`，allocate_slots 会分配 blocks，但不会调用 `coordinator.cache_blocks(...)`。

---

### Q77：`full_sequence_must_fit` 是什么？

答：它是 admission gate。启用时，要求整段序列的 KV blocks 能容纳，否则请求不被接纳。

常用于某些必须确保完整序列能进入 cache 的场景，避免中途无法继续导致复杂回退。

---

### Q78：watermark 的作用是什么？

答：watermark_blocks 用于 waiting/preempted 请求入场时保留一些 free blocks，避免系统刚接纳新请求就频繁抢占。

watermark 只在满足以下条件时应用：

- `has_scheduled_reqs` 为 True；
- request status 是 `WAITING` 或 `PREEMPTED`。

running 请求不套 waiting admission watermark。

---

### Q79：为什么 prefix hit blocks 被 touch 后也可能影响容量检查？

答：如果 prefix hit blocks 当前在 free queue 中，它们本来可以被 eviction。命中后调用 `touch()` 会把它们从 free queue 移除并增加引用，导致可用 free capacity 下降。

因此 get_num_blocks_to_allocate 在容量检查中要把这些被 touch 的 blocks 也考虑进去。

---

## 十一、SingleTypeKVCacheManager 细节

### Q80：`req_to_blocks` 是什么？

答：`req_to_blocks` 是每个请求持有哪些 KV blocks 的核心索引：

```text
req_to_blocks: defaultdict[str, list[KVCacheBlock]]
```

block list 中可能包含真实 block，也可能包含 null block 占位。请求完成或抢占时，需要根据该列表释放或弹出 blocks。

---

### Q81：`num_cached_block` 是什么？

答：它记录某个 request 已经被缓存的 block 数。

用途：

- 避免重复缓存同一 block；
- 区分 running 请求和首次调度请求；
- prefix hit 后，将 computed blocks 计入 cached 区间。

---

### Q82：`get_num_blocks_to_allocate()` 做什么？

答：它计算某个请求为了覆盖目标 token 数，还需要新增多少 block。

核心逻辑：

1. `num_required_blocks = cdiv(num_tokens, block_size)`；
2. 如果是 recycling-aware specs 且 admission cap 开启，限制每请求最大占用；
3. 获取当前请求已有 blocks；
4. running 请求且无新 prefix hit 时，直接计算差值；
5. 计算 skipped tokens 与 skipped blocks；
6. 计算还需要新增多少 block；
7. 如果 prefix hit blocks 在 free queue 中可被驱逐，需要计入 capacity check。

---

### Q83：`add_local_computed_blocks()` 做什么？

答：它把本地 prefix cache 命中的 blocks 加到请求上。

关键步骤：

1. 计算 skipped tokens；
2. 对滑窗跳过的 blocks 做截断；
3. 如果开启 caching，`block_pool.touch(new_computed_blocks)`，避免命中 block 被 eviction；
4. 在 req blocks 前补 null blocks 表示 skipped 区域；
5. 把命中的 computed blocks 加入 req blocks；
6. 设置 `num_cached_block[request_id]`。

---

### Q84：`allocate_external_computed_blocks()` 做什么？

答：它用于 KV connector 场景：外部已经 computed，但本地仍需要 block 来接收这些 KV。

它必须在所有 group 的 local computed blocks 已经 touch 后运行，避免一个 group 的 external allocation 驱逐另一个 group 尚未 touch 的 local cache hit blocks。

---

### Q85：`allocate_new_blocks()` 做什么？

答：它为请求实际新增 block。

新增 block ids 会进入 `new_block_ids`，scheduler 后续通过 `take_new_block_ids()` 传给 worker 清零，避免新 block 中残留旧数据。

---

### Q86：`cache_blocks()` 做什么？

答：它把已经完整计算的 blocks 变成 prefix cache 可复用 blocks。

它会基于 request 的 `block_hashes` 给 full block 写入 hash，并插入 `BlockPool.cached_block_hash_to_block`。

---

### Q87：`remove_skipped_blocks()` 做什么？

答：用于 sliding window、local attention 等场景。

当 computed tokens 前面的一些 blocks 不再会被 attention 访问时，可以释放这些 blocks，并用 null block 占位，保持位置语义。

---

## 十二、BlockPool、KVCacheBlock、Free Queue

### Q88：BlockPool 初始化时有哪些核心字段？

答：主要字段包括：

- `num_gpu_blocks`
- `enable_caching`
- `hash_block_size`
- `blocks`
- `free_block_queue`
- `cached_block_hash_to_block`
- `null_block`
- `enable_kv_cache_events`
- `kv_event_queue`
- `metrics_collector`

---

### Q89：null block 是什么？

答：null block 是特殊占位 block，常用于 sliding window 或跳过 block 场景。

特点：

- `block_id` 来自实际 pool，但被标记为 `is_null=True`；
- 不参与正常缓存/释放；
- ref count 不按普通 block 维护；
- 用于保持 block 位置语义。

---

### Q90：KVCacheBlock 有哪些字段和语义？

答：字段包括：

- `block_id`
- `ref_cnt`
- `_block_hash`
- `prev_free_block`
- `next_free_block`
- `is_null`

语义：

- `ref_cnt > 0` 表示正在被请求引用；
- `ref_cnt == 0` 且有 hash 的 block 可以作为 prefix cache eviction candidate；
- `_block_hash` 只能在 block full 且 cached 后设置；
- eviction 时调用 `reset_hash()` 清除 hash。

---

### Q91：为什么 FreeKVCacheBlockQueue 不用 Python deque？

答：因为需要 O(1) 删除链表中间的 block。

在 prefix cache 命中或 touch 时，可能需要把某个 cached block 从 free queue 中移除。为了减少 GC 和提升效率，直接操作 block 上的 `prev_free_block` / `next_free_block`。

---

### Q92：FreeKVCacheBlockQueue 的顺序语义是什么？

答：队列顺序语义：

1. 初始按 block id 排序；
2. block 被释放后按 eviction 顺序放回；
3. LRU block 在队头；
4. 同一时间释放的 block 中，tail block 更靠前，优先驱逐。

tail block 通常代表更长 hash chain 的后部，优先驱逐对 prefix 命中影响较小。

---

### Q93：`BlockPool.get_cached_block()` 如何判断命中？

答：输入 block_hash 和 kv_cache_group_ids。

它会对每个 group id 生成 `BlockHashWithGroupId`，分别查找。如果任意 group miss，则整体 miss。

这保证 hybrid/multi-group 场景下所有相关 group 都可复用时才算 prefix hit。

---

### Q94：`cache_full_blocks()` 做什么？

答：它把 request 中已经完整计算的 blocks 写入 prefix cache。

流程：

1. 读取 request 预先计算好的 block hashes；
2. 给 block 设置 hash metadata；
3. 插入 `cached_block_hash_to_block`；
4. 可能产生 `BlockStored` KV cache event。

---

## 十三、抢占、释放、驱逐、Deferred Free

### Q95：什么情况下会发生 preemption？

答：主要发生在 running 请求需要新增 KV blocks，但 `kv_cache_manager.allocate_slots()` 返回 None 时。

常见原因：

- free blocks 不足；
- watermark/reserved blocks 限制；
- prefix hit blocks 被 touch 后减少可用 free capacity；
- external KV load 或 in-flight prefill 预留 blocks；
- max model len 或其他 admission 约束。

---

### Q96：preemption 选择哪个请求被抢占？

答：

- priority 策略下：选择 `(priority, arrival_time)` 最大的 running 请求作为被抢占对象。
- FCFS/默认路径下：从 running 列表尾部 pop。

---

### Q97：`_preempt_request()` 做什么？

答：行为：

1. 要求请求必须处于 RUNNING；
2. `_free_request_blocks(request)` 释放 KV blocks；
3. 释放 encoder cache；
4. 从 `_inflight_prefills` 移除；
5. 状态设为 PREEMPTED；
6. `num_computed_tokens = 0`；
7. 清空 spec tokens；
8. `num_preemptions += 1`；
9. 放回 waiting 队列头部。

V1 当前 preemption 的主要恢复策略是 recompute。

---

### Q98：V1 preemption 为什么主要是 recompute？

答：因为被抢占时会释放本地 KV blocks，并把请求放回 waiting。后续恢复时需要重新从 prefix cache 命中处或从头开始计算。

这简化了资源回收和一致性管理，但代价是被抢占请求可能需要重新计算部分上下文。

---

### Q99：`_free_request()` 做什么？

答：请求真正完成时，`_free_request()` 会：

1. 确认请求 finished；
2. 通知 connector request finished；
3. 释放 encoder cache；
4. 加入 `finished_req_ids`；
5. 根据 connector/deferred 状态决定是否立即 `_free_blocks()`；
6. 返回可能需要回传给前端的 `kv_xfer_params`。

---

### Q100：普通 KV blocks 释放路径是什么？

答：普通情况：

```text
_free_request()
  -> _free_blocks()
      -> _free_request_blocks()
          -> kv_cache_manager.free(request)
      -> del self.requests[request_id]
```

worker 侧则通过 SchedulerOutput 中的 `finished_req_ids` 清理 request state。

---

### Q101：deferred free 是什么？

答：deferred free 是延迟释放 block 的机制。

在 async scheduling、overlapping batches、pipeline parallel、KV connector consumer 等场景，GPU 可能仍在写某些 KV blocks。如果 scheduler 立即释放并复用这些 blocks，可能产生数据竞争。

因此会先从 manager bookkeeping 中弹出 blocks，放入 `deferred_frees`，等 `update_from_output()` 确认 GPU 写完成后，再真正归还给 block pool。

---

### Q102：deferred free 的具体流程是什么？

答：

```text
_free_request_blocks(request)
  -> blocks = kv_cache_manager.pop_blocks_for_free(request)
  -> deferred_frees.append((sched_step_seq, blocks))

update_from_output()
  -> _drain_deferred_frees()
  -> 真正把 blocks 归还 block_pool
```

它是并发批处理下避免 KV block 过早复用的关键机制。

---

### Q103：eviction 和 free 有什么区别？

答：

- free：请求完成或抢占后释放它引用的 blocks，使 ref_cnt 降为 0，block 进入 free queue，但如果有 hash，仍可作为 prefix cache 候选。
- eviction：从 prefix cache 中移除某些 cached blocks，清除 hash，使其成为普通可复用 block。

free 不一定立即删除 prefix cache hash；eviction 会清除 cached 状态。

---

### Q104：为什么释放 blocks 时通常按反向顺序？

答：按反向顺序释放可以让 tail blocks 更早被 eviction。

tail blocks 对应更长 prefix chain 的后部，优先驱逐对 prefix cache 命中影响较小。

---

## 十四、EngineCore 初始化 KV Cache

### Q105：EngineCore 初始化 KV cache 的流程是什么？

答：核心流程：

1. `register_all_kvcache_specs(vllm_config)`；
2. `model_executor.get_kv_cache_specs()` 获取模型各层 KV cache spec；
3. 如果存在 non-causal attention，禁用 chunked prefill 和 prefix caching；
4. `model_executor.determine_available_memory()` profile 可用显存；
5. `get_kv_cache_configs(...)` 计算 worker 侧 KV cache config；
6. `generate_scheduler_kv_cache_config(...)` 生成 scheduler 侧统一配置；
7. 写回 `cache_config.num_gpu_blocks`、`block_size`、`kv_cache_size_tokens`、`kv_cache_max_concurrency`；
8. `model_executor.initialize_from_config(kv_cache_configs)` 初始化 worker 侧 KV cache tensor 并 warmup。

---

### Q106：为什么 non-causal attention 会禁用 chunked prefill 和 prefix caching？

答：non-causal attention 可能允许 token 双向可见，prefix cache 和 chunked prefill 的因果增量假设不一定成立。

为了避免错误复用或分块计算破坏 attention 语义，初始化时检测到 non-causal attention 会禁用这些优化。

---

### Q107：scheduler block size、hash block size、实际 KV block size 有什么区别？

答：

- `scheduler_block_size`
  - 调度粒度，通常是各 group block size 的 LCM。

- `hash_block_size`
  - prefix cache hash 的粒度。

- `kv_cache_spec.block_size`
  - 某个 group 的实际 KV cache block size。

- `storage_block_size`
  - 某些压缩/特殊 KV spec 中的存储 block size。

- `kernel_block_size`
  - worker attention backend 支持的 kernel block size。

Coordinator 初始化时要求：

```text
scheduler_block_size % hash_block_size == 0
scheduler_block_size % each_group.block_size == 0
```

---

## 十五、Scheduler 与 Worker / Attention 边界

### Q108：SchedulerOutput 到 GPUModelRunner 的总体链路是什么？

答：

```text
SchedulerOutput
  -> GPUModelRunner.execute_model()
      -> finish_requests()
      -> add_requests()
      -> update_requests()
      -> block_tables.apply_staged_writes()
      -> prepare_inputs()
      -> prepare_attn()
      -> model forward
      -> sample / pool
      -> ModelRunnerOutput
```

---

### Q109：Scheduler 和 Worker 的职责如何划分？

答：Scheduler 负责：

- 请求是否进入 running；
- 每个请求本轮执行多少 token；
- 使用哪些 KV block ids；
- 哪些请求完成或被抢占；
- 哪些 block 是新分配的，需要清零；
- KV connector 本轮 load/save metadata；
- encoder input 是否需要计算。

Worker/GPUModelRunner 负责：

- 缓存 request state；
- 维护 worker 侧 req index；
- 维护 block table；
- 将 block ids 转成 block table tensor 和 slot mapping；
- 准备 input ids、positions、seq lens、query start loc；
- 执行模型 forward；
- 执行 sampler 或 pooling；
- 返回 ModelRunnerOutput。

---

### Q110：Attention backend 负责什么？

答：Attention backend 使用 KV cache tensors、block tables、slot mappings、query start loc、sequence lengths、positions、backend-specific metadata 做高性能 attention。

它不关心：

- 请求队列；
- priority；
- preemption；
- block 分配策略；
- prefix cache 决策。

---

### Q111：GPUModelRunner 接收 SchedulerOutput 后的操作顺序为什么重要？

答：顺序通常是：

```text
1. update_pp_decode_requests()
2. finish_requests(scheduler_output)
3. free_states(scheduler_output)
4. add_requests(scheduler_output)
5. update_requests(scheduler_output)
6. block_tables.apply_staged_writes()
```

重要原因：

- 先删除 finished/preempted 请求，释放 req index；
- 再添加新请求，避免 req index 冲突；
- 再更新已有请求新增 block；
- 最后一次性 apply staged writes，减少 GPU 写入开销。

---

### Q112：新请求在 worker 侧如何处理？

答：`add_requests()` 对每个 scheduled_new_req：

1. 如果是 streaming input update，先清理旧 state；
2. 计算 prompt_len；
3. 创建 worker 侧 request state；
4. 拿到 req_index；
5. encoder cache 记录多模态 feature；
6. model_state 添加请求；
7. `block_tables.append_block_ids(..., overwrite=True)` 写入 block ids；
8. 添加 LoRA state；
9. 最后一个 pipeline rank 初始化 sampler 和 prompt logprobs worker。

---

### Q113：已有请求在 worker 侧如何更新？

答：`update_requests()` 对 scheduled_cached_reqs：

1. 根据 req_id 找到 req_index；
2. 更新 worker 侧 `num_computed_tokens`；
3. 如有新增 block ids，append 到 block table；
4. 更新 prefill token 统计；
5. 对 `new_block_ids_to_zero` 执行 GPU block 清零。

---

### Q114：BlockTables 是什么？

答：BlockTables 是 worker 侧把 scheduler 分配的 block ids 转成 attention backend 输入的核心结构。

它维护：

- 每个 request 的 block table；
- 当前 batch 的 input_block_tables；
- 当前 batch token 的 slot_mappings。

---

### Q115：BlockTables 中 `block_tables` 和 `input_block_tables` 有什么区别？

答：

- `block_tables`
  - 全局/staged write tensor。
  - 形状大致是：`num_kv_cache_groups x [max_num_reqs, max_num_blocks]`。
  - 每行对应一个 worker req index。

- `input_block_tables`
  - 本次 model forward 实际使用的 block table tensor。
  - 通过 `gather_block_tables()` 根据当前 batch 的 idx_mapping 从全局 block table 中收集需要的行。

---

### Q116：`append_block_ids()` 的 overwrite 参数是什么意思？

答：

- `overwrite=True`
  - 新请求或恢复请求，重新写入 block ids。

- `overwrite=False`
  - 已有请求，追加新增 block ids。

如果 `blocks_per_kv_block > 1`，说明 scheduler KV block 和 kernel block 粒度不同，需要展开 block ids。

---

### Q117：`prepare_inputs()` 做什么？

答：它把 SchedulerOutput 转成 InputBatch。

主要步骤：

1. 读取 total_num_scheduled_tokens；
2. 获取本轮 request 集合；
3. 按每个请求 scheduled token 数排序，通常 decode 在 prefill 前；
4. 构建 idx_mapping：batch index -> worker req index；
5. 处理 spec decode draft token 数量；
6. 构建 query_start_loc；
7. 判断哪些请求在 prefill；
8. 准备 prompt input ids；
9. 准备 positions 和 seq_lens；
10. 处理 context parallel local seq lens；
11. 合并 last sampled tokens 和 draft tokens；
12. 构建 InputBatch。

---

### Q118：为什么 worker 侧通常 decode 排在 prefill 前？

答：源码中采用 “Decode first, then prefill”。

通常 decode 每个请求 token 数较少，prefill chunk token 数较大。将 decode 放前面有利于降低 decode 请求延迟，并符合连续批处理中 decode 优先的常见优化思路。

---

### Q119：`prepare_attn()` 做什么？

答：它做两件事：

1. `block_tables.gather_block_tables(...)`
   - 收集当前 batch 需要的 block table rows。

2. `block_tables.compute_slot_mappings(...)`
   - 计算当前 batch tokens 的 KV cache 写入 slot。

返回：

```text
(block_tables, slot_mappings)
```

---

### Q120：slot_mappings 如何计算？

答：`compute_slot_mappings()` 调用 Triton kernel，根据：

- request index mapping；
- query start loc；
- positions；
- block table；
- block size；
- context parallel rank/size；

计算每个 token 对应的 slot id。

slot_mappings 形状大致为：

```text
[num_kv_cache_groups, max_num_batched_tokens]
```

---

### Q121：block table 和 slot mapping 分别给 attention 什么信息？

答：

- block table：告诉 attention 每个请求有哪些历史 KV blocks，用于读取历史 K/V。
- slot mapping：告诉 attention 本轮输入 token 的 KV 要写到哪个 slot，用于写入当前 K/V。

简短表达：

```text
block table 负责读历史
slot mapping 负责写当前
```

---

### Q122：KV cache tensor 在 worker 侧如何初始化？

答：worker 侧通过 attn_utils：

1. `get_kv_cache_spec()` 扫描模型 attention layers，收集 layer -> KVCacheSpec；
2. `init_attn_backend()` 按 KV cache group 初始化 backend、kernel block size 和 metadata builders；
3. `_allocate_kv_cache()` 创建 raw tensor；
4. `_reshape_kv_cache()` 按 backend 需要 reshape 成 KV cache shape；
5. 处理 shared KV cache layers、Mamba state tensor 等。

---

### Q123：scheduler 给出的 block ids 如何进入 attention kernel？

答：路径：

```text
KVCacheBlocks
  -> block_ids
  -> SchedulerOutput.scheduled_new_reqs / scheduled_cached_reqs
  -> GPUModelRunner.add_requests() / update_requests()
  -> BlockTables.append_block_ids()
  -> block_tables.apply_staged_writes()
  -> prepare_attn()
  -> input_block_tables + slot_mappings
  -> attention backend
```

---

## 十六、ModelRunnerOutput 与 update_from_output

### Q124：ModelRunnerOutput 包含什么？

答：通常包含：

- sampled token ids；
- logprobs；
- prompt logprobs；
- pooler output；
- req id 到 batch index 映射；
- KV connector output；
- cudagraph stats；
- routed experts 信息。

Scheduler 用它在 `update_from_output()` 中推进请求状态。

---

### Q125：`Scheduler.update_from_output()` 做什么？

答：它接收 ModelRunnerOutput 后：

1. 处理 deferred free fence；
2. 处理 KV connector invalid blocks；
3. 遍历本轮 scheduled 请求；
4. 处理 spec decode rejected tokens，回退 num_computed_tokens；
5. 释放已消费的 encoder inputs；
6. 追加生成 token；
7. 检查 stop 条件；
8. 若请求完成，调用 `_free_request()`；
9. 从 running/waiting 队列移除已停止请求；
10. 更新 KV connector transfer 完成状态；
11. 收集并发布 KV cache events；
12. 构建 EngineCoreOutputs。

---

### Q126：`_update_request_with_output()` 做什么？

答：它逐个追加 sampled token，并调用 `check_stop(request, max_model_len)` 检查是否应该停止。

停止条件可能来自 stop token、stop string、最大生成长度、最大模型长度、abort/error 等。

---

### Q127：Scheduler 为什么要根据 `req_id_to_index` 找输出？

答：ModelRunnerOutput 中的输出是按当前 batch 顺序排列的，而 Scheduler 需要按 request_id 更新对应 Request。

因此需要 `req_id_to_index` 从 request_id 映射到输出数组中的 index。

---

## 十七、KV Connector、Offload 与 Remote KV

### Q128：KV Connector 的定位是什么？

答：KV Connector 是连接 Scheduler、Worker、远端 KV 存储或 P/D disaggregation 的机制。

它有两侧：

- Scheduler 侧 connector
  - 负责请求级状态、prefix match、load/save 计划、preemption metadata。

- Worker 侧 connector
  - 负责绑定真实 KV cache tensor、执行 load/save、报告完成/失败/events/stats。

---

### Q129：Scheduler 侧 connector 在什么时候创建？

答：在 Scheduler 初始化时，如果 `vllm_config.kv_transfer_config` 不为空，会创建 scheduler 侧 connector：

```text
KVConnectorFactory.create_connector(
    config=vllm_config,
    role=KVConnectorRole.SCHEDULER,
    kv_cache_config=kv_cache_config,
)
```

---

### Q130：Request 中的 KV transfer 参数从哪里来？

答：Request 有字段：

```text
kv_transfer_params: dict[str, Any] | None
```

如果 `sampling_params.extra_args` 中带有 `kv_transfer_params`，会写入 Request。

如果请求带 KV transfer params，但 Scheduler 没有 connector，EngineCore 会发 warning。

---

### Q131：新请求如何进入 connector？

答：Scheduler.add_request() 中，新请求加入 `self.requests` 后，如果 connector 存在，会调用：

```text
connector.on_new_request(request)
```

让 connector 在请求调度前初始化 transfer 相关状态。

---

### Q132：本地 prefix cache 与 external KV 命中如何组合？

答：waiting 请求首次调度时：

1. 先查本地 prefix cache：

```text
new_computed_blocks, num_new_local_computed_tokens = kv_cache_manager.get_computed_blocks(request)
```

2. 再查外部 KV 命中：

```text
ext_tokens, load_kv_async = connector.get_num_new_matched_tokens(
    request,
    num_new_local_computed_tokens,
)
```

3. 组合：

```text
num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
```

---

### Q133：`load_kv_async=True` 时 Scheduler 怎么处理？

答：如果 connector 返回 `load_kv_async=True`：

1. `num_new_tokens = 0`；
2. 调 `kv_cache_manager.allocate_slots(...)` 为 external computed tokens 分配本地接收 blocks；
3. 设置 `delay_cache_blocks=True`；
4. 调 `connector.update_state_after_alloc(...)` 通知 connector 本地 blocks；
5. 请求状态设为 `WAITING_FOR_REMOTE_KVS`；
6. 请求放入 skipped_waiting；
7. 本轮不进入 running，不执行本地 forward。

---

### Q134：为什么 remote KV load 仍需要本地 blocks？

答：外部 KV 命中只表示“内容存在于远端”，但 worker 执行 attention 时仍需要本地 KV cache tensor 中有可访问的 blocks。

因此即使命中 remote KV，也要为这些 external computed tokens 分配本地接收 blocks，用于加载远端 KV 内容。

---

### Q135：async remote KV load 为什么需要 reserved_blocks？

答：async remote KV load 会占用 blocks 一段时间，但没有 forward progress，也不可通过普通 preemption 释放。

如果它占光 free blocks，其他正在 prefill 的请求可能无法完成，造成死锁或频繁抢占。

reserved_blocks 为 in-flight prefills 保留容量，避免这种情况。

---

### Q136：`connector.update_state_after_alloc()` 做什么？

答：waiting 请求成功分配 slots 后，Scheduler 调用它告诉 scheduler 侧 connector：

- 请求对应的本地 blocks 是哪些；
- 有多少 tokens 是 external computed；
- 是否需要安排 load/save。

这些信息后续会被构造成 connector metadata 发给 worker。

---

### Q137：SchedulerOutput 中的 connector metadata 是什么？

答：如果 connector 存在，调度尾部会构建：

```text
scheduler_output.kv_connector_metadata = meta
```

它是 opaque object，SchedulerOutput 只负责携带，worker connector 负责解读，并据此执行 load/save、preemption handling 等。

---

### Q138：Worker 侧 KVConnector 有哪些接口？

答：Worker 侧 `KVConnector` 接口包括：

- `pre_forward(scheduler_output)`；
- `post_forward(finished_req_ids, wait_for_save=True)`；
- `no_forward(scheduler_output)`；
- `set_disabled(disabled)`。

默认实现是 no-op；ActiveKVConnector 会执行真实 KV transfer。

---

### Q139：ActiveKVConnector 初始化做什么？

答：ActiveKVConnector 初始化时：

1. 获取全局 KV transfer group；
2. 注册 KV cache tensors：`register_kv_caches(kv_caches_dict)`；
3. 设置 host transfer copy 操作：`set_host_xfer_buffer_ops(copy_kv_blocks)`。

---

### Q140：Worker 侧 `pre_forward()` 做什么？

答：它在模型 forward 前执行：

1. 读取 `scheduler_output.kv_connector_metadata`；
2. `handle_preemptions(kv_connector_metadata)`；
3. `bind_connector_metadata(kv_connector_metadata)`；
4. `start_load_kv(get_forward_context())`。

这会在 forward 前开始加载 remote KV。

---

### Q141：Worker 侧 `post_forward()` 做什么？

答：它在模型 forward 后执行：

1. 等待保存完成：`wait_for_save()`；
2. 获取 finished sending / receiving；
3. 获取 load error block ids；
4. 获取 connector stats；
5. 获取 KV cache events；
6. 构建 worker meta；
7. 清理 connector metadata；
8. 返回 KVConnectorOutput。

---

### Q142：Worker 侧 `no_forward()` 用于什么场景？

答：当本轮没有 token 需要执行，但需要处理 KV transfer 时使用。

它会：

```text
pre_forward()
post_forward(wait_for_save=False)
返回只包含 KV connector output 的 ModelRunnerOutput
```

典型场景是 remote KV async load，本轮无需本地模型 forward。

---

### Q143：KV load 失败如何处理？

答：Scheduler.update_from_output() 开始处会检查：

```text
kv_connector_output.invalid_block_ids
```

如果存在 invalid blocks，会调用 `_handle_invalid_blocks(...)`。

后续：

- 如果 `recompute_kv_load_failures=False`，请求会被标记为 error finished；
- 如果启用 recompute 策略，Scheduler 会调整请求 computed token 状态，后续重新计算失败部分。

---

### Q144：KV transfer 完成后 Scheduler 如何更新状态？

答：Scheduler.update_from_output() 末尾调用：

```text
_update_from_kv_xfer_finished(kv_connector_output)
```

它负责：

- 记录 finished sending；
- 记录 finished receiving；
- 将 `WAITING_FOR_REMOTE_KVS` 请求恢复为可调度；
- 对成功加载的 tokens 调整缓存状态；
- 对失败加载根据策略重算或报错。

---

### Q145：Preemption 和 connector 如何协同？

答：当请求被抢占时，Scheduler 调 `_preempt_request()` 释放本地 KV blocks，并将请求放回 waiting。

同时调度输出构建 connector metadata 时，connector 会把 preemption 信息传给 worker。worker 侧 `ActiveKVConnector.pre_forward()` 调 `handle_preemptions()`，清理或同步 transfer 状态。

---

### Q146：Request finished 与 connector 有什么关系？

答：请求完成释放时，`_free_request()` 会先调用 `_connector_finished(request)`。

connector 可以：

- 判断是否需要延迟释放 blocks；
- 返回需要回传给前端的 `kv_xfer_params`；
- 清理 connector 侧请求状态。

如果 connector 要求 delay free，则 `_free_request()` 不会立即释放 blocks。

---

### Q147：KV Connector / Offload 的核心不变量有哪些？

答：核心不变量：

1. 外部 KV 命中只表示内容存在，本地仍需要 block 接收或引用；
2. async load 的 blocks 在 load 完成前不能进入 prefix cache；
3. remote KV load 占用 block 但可能没有 forward progress，因此需要 reserved blocks 避免死锁；
4. overlapping batches 下 block 释放必须 fenced，避免 GPU 写未完成就复用；
5. load 失败不能让请求继续使用坏 KV，要么 recompute，要么 error finish。

---

## 十八、Encoder Cache 与多模态相关

### Q148：Scheduler 为什么要管理 encoder budget？

答：多模态或 encoder-decoder 请求可能需要额外 encoder compute 和 encoder cache。

Scheduler 需要控制每轮 encoder input token 数，避免 encoder 侧计算或缓存占用超过预算，同时确保 decoder KV 调度和 encoder cache 协同。

---

### Q149：scheduled_encoder_inputs 在 SchedulerOutput 中有什么作用？

答：它告诉 worker 本轮需要处理哪些 encoder/multimodal inputs。

worker 侧会根据它执行 encoder 计算或使用 encoder cache，并将结果与 decoder forward 对齐。

---

### Q150：请求完成或抢占时 encoder cache 如何释放？

答：Scheduler 在 `_preempt_request()` 或 `_free_request()` 中会释放 encoder cache。

worker 侧在 `finish_requests()` / `_remove_request()` 中也会释放对应 worker-side encoder cache request state。

---

## 十九、常见问题定位

### Q151：为什么某个请求没有被调度？

答：优先检查：

- Scheduler.schedule() waiting 部分；
- RequestQueue；
- KVCacheManager.allocate_slots()。

常见原因：

- token budget 不足；
- running 请求数达到上限；
- KV blocks 不足；
- watermark/reserved blocks 限制；
- LoRA 数量限制；
- encoder budget/cache 不足；
- remote KV 尚未完成；
- structured output grammar 尚未准备好；
- streaming request 还在等新输入。

---

### Q152：为什么发生 preemption？

答：通常是 running 请求调度时需要新增 KV blocks，但 allocate_slots 返回 None。

常见原因：

- free blocks 不足；
- prefix hit blocks 被 touch 后减少可用 capacity；
- waiting/preempted admission 触发 watermark；
- external KV load 或 in-flight prefill 预留 blocks；
- block pool 使用率过高。

---

### Q153：prefix cache 为什么没命中？

答：常见原因：

- prefix caching 被禁用；
- request 设置 `skip_reading_prefix_cache`；
- block 未满；
- hash block size 不一致；
- cache salt 不一致；
- group id 不一致；
- sliding window/hybrid group 要求所有 group 同时命中；
- block 已被 eviction；
- 请求 token 序列实际不一致。

---

### Q154：block id 如何进入 attention？

答：路径：

```text
KVCacheBlocks
  -> block_ids
  -> SchedulerOutput
  -> GPUModelRunner.add_requests()/update_requests()
  -> BlockTables.append_block_ids()
  -> gather_block_tables()
  -> compute_slot_mappings()
  -> attention backend
```

---

### Q155：请求结束后资源什么时候释放？

答：普通情况：Scheduler 在 `update_from_output()` 检测到请求 finished 后调用 `_free_request()`，立即释放 KV blocks，并通过 SchedulerOutput 的 `finished_req_ids` 通知 worker 清理状态。

async / PP / KV connector / overlapping batches 场景可能启用 deferred free，需要等 GPU 写完成或 connector 状态安全后再真正归还 blocks。

---

### Q156：为什么新分配 block 需要清零？

答：因为 block pool 复用的物理 block 中可能残留旧数据、NaN 或 SSM 状态。

如果不清零，attention/SSM 可能读取到无效内容，造成输出错误或 NaN 扩散。

Scheduler 通过 `new_block_ids_to_zero` 通知 worker 清零。

---

### Q157：为什么 remote KV load 成功但请求仍可能没进入 running？

答：可能原因：

- connector finished 状态还未被 scheduler update_from_output 处理；
- 请求仍在 `WAITING_FOR_REMOTE_KVS`；
- KV load 有 invalid blocks，需要 recompute 或 error；
- 本轮 token budget / running 数 / KV blocks / encoder budget 不足；
- blocked waiting promote 失败。

---

### Q158：为什么 KV block 释放后没有立刻变成可用？

答：可能启用了 deferred free。

在 async scheduling、PP、KV connector consumer、overlapping batches 场景，GPU 可能仍在写这些 blocks。Scheduler 会延迟归还 block_pool，直到确认安全。

---

### Q159：为什么 block_pool 看起来有空闲 block，但 allocate_slots 仍失败？

答：可能原因：

- watermark_blocks 占用 admission 余量；
- reserved_blocks 为 in-flight prefill 保留；
- prefix hit blocks touch 后减少 free capacity；
- full_sequence_must_fit 要求整段序列能容纳；
- 某些 group/hybrid manager 需要更多 blocks；
- sliding window/Mamba/cross attention 特殊 manager 规则导致不足。

---

### Q160：为什么 worker 侧 block table 出错会影响 attention？

答：attention backend 根据 block table 读取历史 KV。如果 block table 中 block ids 错误、缺失、未 apply staged writes 或粒度展开错误，attention kernel 会读取错误 KV block，导致输出错误或非法访问。

---

### Q161：slot mapping 错误会导致什么？

答：slot mapping 决定当前 token 的 K/V 写入位置。

错误可能导致：

- K/V 写到错误 block；
- 写越界；
- 覆盖其他请求的 KV；
- 后续 attention 读取错历史；
- CUDA illegal memory access；
- 输出异常或 NaN。

---

### Q162：如何定位 KV connector load 失败？

答：优先看：

- SchedulerOutput 中 kv_connector_metadata；
- Worker connector pre_forward/post_forward；
- `kv_connector_output.invalid_block_ids`；
- `_handle_invalid_blocks()`；
- recompute_kv_load_failures 策略；
- connector events/stats；
- 请求是否停留在 `WAITING_FOR_REMOTE_KVS`。

---

## 二十、高频综合题

### Q163：请完整描述一个请求从进入到释放的生命周期。

答：

```text
外部请求
  -> EngineCoreRequest
  -> Request.from_engine_core_request()
  -> Scheduler.add_request()
  -> waiting / skipped_waiting
  -> Scheduler.schedule()
      -> prefix cache 查询
      -> KV slots 分配
      -> running
      -> SchedulerOutput
  -> model_executor.execute_model()
  -> GPUModelRunner.execute_model()
      -> add/update request state
      -> 更新 block table
      -> 构建 input batch
      -> attention + model forward
      -> sample / pool
      -> ModelRunnerOutput
  -> Scheduler.update_from_output()
      -> 追加 output token
      -> stop 检查
      -> 释放 encoder/KV cache
      -> EngineCoreOutputs
```

核心驱动力是 `Request.num_computed_tokens`。

---

### Q164：请完整描述 KV block 分配链路。

答：

```text
Scheduler.schedule()
  -> 计算 num_new_tokens / prefix hit / external KV hit / lookahead
  -> kv_cache_manager.allocate_slots(...)
  -> KVCacheCoordinator 汇总多 group 需求
  -> SingleTypeKVCacheManager.get_num_blocks_to_allocate()
  -> BlockPool 分配或 touch blocks
  -> SingleTypeKVCacheManager 更新 req_to_blocks
  -> KVCacheBlocks 返回 Scheduler
  -> SchedulerOutput 携带 block ids
  -> Worker 写入 BlockTables
```

---

### Q165：请解释 prefix cache 的完整生命周期。

答：

```text
Request 创建或输出 token 追加
  -> update_block_hashes()
  -> 等完整 block 形成 hash
  -> 请求首次调度时 get_computed_blocks()
  -> Coordinator.find_longest_cache_hit()
  -> BlockPool.get_cached_block()
  -> 命中 blocks 被 touch，加入 req_to_blocks
  -> 新计算完成的 full blocks 调 cache_blocks()
  -> BlockPool.cache_full_blocks()
  -> hash -> block 插入 BlockHashToBlockMap
  -> 后续请求可复用
  -> block ref_cnt 为 0 后可作为 eviction candidate
```

---

### Q166：请解释 SchedulerOutput 到 AttentionBackend 的完整数据转换。

答：

```text
SchedulerOutput
  -> scheduled_new_reqs / scheduled_cached_reqs 携带 block ids 和 token 数
  -> GPUModelRunner.finish_requests/add_requests/update_requests
  -> BlockTables.append_block_ids 写入 worker 侧 block table
  -> block_tables.apply_staged_writes
  -> prepare_inputs 构建 InputBatch、positions、seq_lens、query_start_loc
  -> prepare_attn gather block tables + compute slot mappings
  -> attention metadata builder 构造 backend-specific metadata
  -> model forward
  -> Attention backend 使用 KV cache tensor、block table、slot mapping 执行
```

---

### Q167：请解释 preemption、recompute、prefix cache 三者关系。

答：preemption 会释放请求当前 KV blocks，并把请求状态设为 PREEMPTED，`num_computed_tokens` 重置为 0，后续放回 waiting 队列。

恢复时，Scheduler 会像新请求一样再次查询 prefix cache。如果之前完整计算的 blocks 还在 prefix cache 中，可以复用一部分；如果已被 eviction，则需要 recompute。

因此 V1 preemption 的恢复策略是 recompute，但 prefix cache 可以减少 recompute 成本。

---

### Q168：请解释 deferred free 为什么必要。

答：在 overlapping batches、async scheduling、pipeline parallel 或 KV connector consumer 场景中，Scheduler 可能在某个请求逻辑完成后立即想复用它的 blocks，但 GPU 上可能仍有未完成写操作。

如果立刻把 block 分给其他请求或 connector load，会出现未排序写冲突和数据污染。

Deferred free 通过 fencing，让 block 在确认安全后再回到 block pool。

---

### Q169：请解释 remote KV load 如何扩展 prefix cache 语义。

答：普通 prefix cache 只查询本地已缓存 blocks。KV connector 把它扩展为：

```text
本地 prefix cache 命中 tokens
+ 外部 remote KV 命中 tokens
= num_computed_tokens
```

但 external computed tokens 仍需要本地 blocks 接收 KV。Scheduler 分配这些 blocks，worker connector 执行 load，完成后请求才能继续本地 forward。

---

### Q170：请解释 Scheduler、KVCacheManager、BlockPool、Worker BlockTables 的边界。

答：

- Scheduler：决定请求、token 数、调度顺序、是否抢占。
- KVCacheManager：给 Scheduler 提供 prefix hit、allocate/free 的统一接口。
- BlockPool：维护真实 KVCacheBlock 元数据、free queue、hash map、events。
- Worker BlockTables：把 Scheduler 分配的 block ids 转成 GPU attention backend 使用的 block table tensor 和 slot mapping。

简短说：Scheduler 做决策，KV manager 管元数据，BlockPool 管物理 block 元数据，Worker 把 block ids 转成 tensor 输入。

---

## 二十一、简短背诵版

### Q171：一句话解释 Scheduler。

答：Scheduler 决定每轮哪些请求执行、每个请求执行多少 token、占用哪些 KV blocks，以及何时抢占和释放。

### Q172：一句话解释 Request。

答：Request 是 V1 调度基础单位，用 `num_computed_tokens` 表达执行进度，并保存 token、状态、priority、block hashes、LoRA、多模态等信息。

### Q173：一句话解释 KVCacheManager。

答：KVCacheManager 是 Scheduler 操作 KV cache 的统一入口，负责 prefix hit 查询、slot/block 分配、释放和事件收集。

### Q174：一句话解释 KVCacheCoordinator。

答：KVCacheCoordinator 协调多个 KV cache group，把 full attention、sliding window、Mamba、cross attention 等不同 cache 类型统一起来。

### Q175：一句话解释 SingleTypeKVCacheManager。

答：SingleTypeKVCacheManager 管理某一种 cache spec 下每个请求的 block 列表、缓存状态、分配和释放规则。

### Q176：一句话解释 BlockPool。

答：BlockPool 维护所有 KVCacheBlock 元数据、free list、prefix hash map、null block 和 KV cache events。

### Q177：一句话解释 prefix cache。

答：prefix cache 通过完整 block 的 hash 复用已经计算好的 KV blocks，减少重复 prefill。

### Q178：一句话解释 block table。

答：block table 是 worker 侧请求到物理 KV block id 的 tensor 映射，attention kernel 用它读取历史 KV。

### Q179：一句话解释 slot mapping。

答：slot mapping 把当前 batch 中每个 token 映射到具体 KV cache slot，用于写入当前 K/V。

### Q180：一句话解释 preemption。

答：preemption 在 KV blocks 不足时释放某个 running 请求的 blocks，将其放回 waiting，后续通过 recompute/prefix cache 恢复。

### Q181：一句话解释 deferred free。

答：deferred free 延迟归还 KV blocks，避免 GPU 或 connector 仍在写时 block 被过早复用。

### Q182：一句话解释 KV Connector。

答：KV Connector 把本地 prefix cache 扩展到远端 KV/offload/P-D disaggregation，Scheduler 规划 external computed tokens，Worker 执行真实 KV load/save。

### Q183：一句话解释 SchedulerOutput。

答：SchedulerOutput 是 Scheduler 给 Worker 的执行计划，包含新请求、已有请求差量、token 数、block ids、finished/preempted 请求和 connector metadata。

### Q184：一句话解释 update_from_output。

答：update_from_output 根据 worker 返回的 ModelRunnerOutput 更新请求 token、处理 spec decode 回退、检查 stop、释放资源并生成 EngineCoreOutputs。

### Q185：一句话解释 V1 调度核心设计。

答：V1 用 `num_computed_tokens` 统一 prefill、decode、chunked prefill、prefix cache、spec decode 和 remote KV 场景，让调度器每轮只需决定“还差多少 token 需要推进”。
