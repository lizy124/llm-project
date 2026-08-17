# 05 scheduler 背诵文档

## 1. 专题定位

`scheduler` 讲的是 vLLM V1 的请求调度中心。

它不执行模型 forward。

它决定每一轮：哪些请求可以跑、每个请求跑多少 token、需要哪些 KV block、哪些请求等待、哪些请求抢占、Worker 返回后如何更新状态。

一句话：

```text
Scheduler 是 EngineCore 内部的调度和请求状态账本中心。
```

## 2. 最小心智模型

最小链路是：

```text
Request
  → Scheduler.add_request()
  → waiting 队列
  → Scheduler.schedule()
  → running / waiting 两阶段调度
  → KVCacheManager 分配 block
  → SchedulerOutput
  → Worker / ModelRunner 执行
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → Request 状态更新 / block 释放 / EngineCoreOutputs
```

要背住：

```text
schedule() 发任务，update_from_output() 收结果并对账。
```

## 3. Scheduler 负责什么

Scheduler 负责：

```text
1. 管理所有未释放请求。
2. 管理 waiting / skipped_waiting / running 队列。
3. 控制每轮 token budget。
4. 先调度 running 请求，再调度 waiting 请求。
5. 查询本地 prefix cache。
6. 查询外部 KV cache / KV Connector。
7. 调用 KVCacheManager 分配 KV block。
8. KV 不够时抢占请求。
9. 调度多模态 encoder input。
10. 调度 speculative decoding draft tokens。
11. 生成 SchedulerOutput。
12. Worker 返回后更新 request 状态。
13. 处理 stop、logprobs、pooling、grammar、KV connector、资源释放。
```

## 4. Scheduler 不负责什么

Scheduler 不负责：

```text
用户输入预处理
detokenize
模型 forward
attention metadata 构造
sampler 具体执行
GPU KV tensor 读写
最终 RequestOutput 构造
```

这些属于：

```text
InputProcessor
OutputProcessor
ModelRunner
Attention backend
Sampler
Worker
```

## 5. 核心队列和状态

Scheduler 内部最重要的是四类状态。

### self.requests

```text
request_id → Request
```

保存所有还没彻底释放的请求。

即使请求已经不在 running / waiting，也可能因为 KV connector 还有异步 transfer，暂时保留。

### self.waiting

正常等待调度的请求队列。

新请求通常进入这里。

### self.skipped_waiting

被临时跳过的 waiting 请求队列。

常见原因：

```text
等 structured output grammar 初始化
等远端 KV load 完成
等 streaming input 下一段输入
LoRA 数量限制
KV connector 暂时无法确定命中数
encoder cache / encoder budget 不足
```

### self.running

已经进入模型执行流、后续可继续调度的请求列表。

## 6. 调度策略

Scheduler 支持不同 policy：

```text
FCFS：先到先服务。
PRIORITY：按优先级排序。
```

策略会影响：

```text
waiting 队列顺序
skipped_waiting 和 waiting 谁先调度
KV block 不足时抢占哪个请求
```

## 7. Scheduler 初始化

初始化时读取配置：

```text
scheduler_config
cache_config
lora_config
parallel_config
kv_cache_config
```

核心限制：

```text
max_num_running_reqs：最多同时 running 请求数。
max_num_scheduled_tokens：单步最多调度 token 数。
max_model_len：模型最大上下文长度。
```

还会创建：

```text
KVCacheManager
EncoderCacheManager
KV Connector
EC Connector
Spec Decode 状态
StructuredOutputManager 引用
```

一句话：

```text
Scheduler 初始化时准备调度规则、KV 账本、encoder cache 账本和外部 connector。
```

## 8. schedule() 总览

`schedule()` 每调用一次，生成一个 `SchedulerOutput`。

压缩流程：

```text
schedule()
  → 初始化 token_budget
  → 调度 running 请求
  → 调度 waiting 请求
  → 构造 SchedulerOutput
  → 构建 KVConnector / ECConnector metadata
  → _update_after_schedule()
  → 返回 SchedulerOutput
```

更详细：

```text
1. token_budget = max_num_scheduled_tokens。
2. 先遍历 running 请求，计算本轮还要跑多少 token。
3. 为 running 请求分配 KV block。
4. block 不够则抢占其他 running 请求。
5. 如果没有抢占，再从 waiting / skipped_waiting 拉新请求。
6. waiting 请求先查 prefix cache，再查外部 KV。
7. 计算需要 prefill 的 token 数。
8. 分配 KV block。
9. 新请求进入 running，或进入 WAITING_FOR_REMOTE_KVS。
10. 输出 SchedulerOutput。
```

## 9. 本轮 token budget

每轮开头：

```text
token_budget = max_num_scheduled_tokens
```

如果 Scheduler 被 pause：

```text
token_budget = 0
```

还会有 encoder budget：

```text
encoder_compute_budget = max_num_encoder_input_tokens
```

要背住：

```text
token_budget 控制 decoder token；encoder budget 控制多模态 / encoder input。
```

## 10. 第一阶段：调度 running 请求

Scheduler 优先处理 running 请求。

原因：

```text
running 请求已经占用 KV block。
继续推进它们可以降低 decode 延迟，并尽快释放已占资源。
```

running 请求本轮 token 数公式：

```text
num_new_tokens = request.num_tokens_with_spec
               + request.num_output_placeholders
               - request.num_computed_tokens
```

含义：

```text
num_tokens_with_spec：prompt + output + pending draft tokens。
num_output_placeholders：异步 / PP 场景预留输出。
num_computed_tokens：Scheduler 认为已经计算到的位置。
```

差值就是本轮还要补算的 token。

## 11. running 请求的限制

`num_new_tokens` 会被限制：

```text
long_prefill_token_threshold
本轮 token_budget
max_model_len
encoder input budget
Mamba block 对齐
KV block capacity
```

如果有 encoder input，会先尝试调度 encoder。

如果 KV block 分配失败，进入抢占逻辑。

## 12. KV block 不够时抢占

当 `allocate_slots()` 返回失败，说明 KV cache 空间不够。

抢占规则：

```text
PRIORITY policy：抢占优先级最低的 running 请求。
FCFS / 默认：通常抢占 running 队尾请求。
```

被抢占请求会：

```text
释放 KV blocks
释放 encoder cache
状态改为 PREEMPTED
num_computed_tokens 重置为 0
放回 waiting 队列头部
```

一句话：

```text
抢占是为了腾出 KV block，让更应该运行的请求继续推进。
```

## 13. 第二阶段：调度 waiting 请求

只有在本轮没有发生抢占，并且 Scheduler 没有暂停新请求时，才调度 waiting。

waiting 请求来源：

```text
waiting：正常等待。
skipped_waiting：之前被临时跳过。
```

FCFS 下通常优先处理 skipped_waiting。

因为它们已经等过一轮。

## 14. waiting 请求的阻塞状态

waiting 请求可能处于：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
WAITING_FOR_REMOTE_KVS
WAITING_FOR_STREAMING_REQ
```

调度时会尝试提升阻塞状态。

例如：

```text
远端 KV load 完成
  → WAITING_FOR_REMOTE_KVS 恢复为 WAITING / PREEMPTED
```

如果还没满足条件，请求继续留在 skipped_waiting。

## 15. LoRA 限制

如果启用了 LoRA，一轮调度中 active LoRA 数量不能超过 `max_loras`。

当某个 waiting 请求使用新的 LoRA，而当前 batch 已达到上限：

```text
暂时跳过该请求
放入 skipped_waiting
```

这说明：

```text
Scheduler 不执行 LoRA，但它要保证本轮 active LoRA 数量满足执行层限制。
```

## 16. waiting 请求的 prefix cache 查询

waiting 请求第一次调度时，`num_computed_tokens == 0`。

Scheduler 先查本地 prefix cache：

```text
KVCacheManager.get_computed_blocks(request)
  → new_computed_blocks
  → num_new_local_computed_tokens
```

含义：

```text
new_computed_blocks：本地已命中的 KV blocks。
num_new_local_computed_tokens：本地命中的 token 数。
```

要记住：

```text
prefix cache 命中可以减少本轮 prefill token。
```

## 17. 外部 KV cache 查询

如果配置了 KV Connector，还会查外部 KV：

```text
connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)
  → ext_tokens, load_kv_async
```

`ext_tokens` 表示：

```text
除本地 prefix cache 已命中部分外，外部 KV cache 还能新增命中的 token 数。
```

如果 `ext_tokens is None`：

```text
connector 暂时无法确定命中数
Scheduler 本轮跳过请求
放入 skipped_waiting
```

## 18. waiting 请求的 token 计算

如果不是异步 KV load：

```text
num_new_tokens = request.num_tokens - num_computed_tokens
```

如果 `load_kv_async=True`：

```text
num_new_tokens = 0
```

因为本轮只发起远端 KV load，不做模型 forward。

`num_new_tokens` 还会被限制：

```text
long_prefill_token_threshold
enable_chunked_prefill
token_budget
encoder input budget
Mamba block 对齐
```

## 19. chunked prefill

如果没有启用 chunked prefill：

```text
一个 waiting 请求剩余 prefill token 超过本轮预算，就不调度它。
```

如果启用了 chunked prefill：

```text
可以只调度 prompt 的一部分。
```

一句话：

```text
chunked prefill 允许长 prompt 被拆成多轮计算。
```

## 20. waiting 请求分配 block

waiting 请求分配 block 时，信息比 running 更复杂：

```text
num_new_tokens：本轮要计算的 token。
num_new_computed_tokens：本地 prefix hit token。
new_computed_blocks：本地命中的 blocks。
num_external_computed_tokens：外部 KV 命中 token。
delay_cache_blocks：异步 KV load 时先分配但延迟 cache。
num_encoder_tokens：encoder / cross attention blocks。
num_lookahead_tokens：spec decode 预留。
```

关键点：

```text
外部 KV 命中的 token 不需要 forward，但仍需要分配本地 block 作为 load 目标。
```

## 21. waiting 请求进入 running

分配成功后：

```text
request_queue.pop_request()
self.running.append(request)
request.status = RUNNING
request.num_computed_tokens = prefix/local/external 命中数
```

如果是新请求：

```text
记录到 scheduled_new_reqs
```

如果是抢占恢复请求：

```text
记录到 scheduled_resumed_reqs
```

之后 `_update_after_schedule()` 会再把本轮实际调度 token 加到 `num_computed_tokens`。

## 22. WAITING_FOR_REMOTE_KVS

如果 `load_kv_async=True`：

```text
request.status = WAITING_FOR_REMOTE_KVS
request 放入 skipped_waiting
本轮不进入 running
本轮不 forward
SchedulerOutput 仍携带 kv_connector_metadata
Worker 负责异步 load KV
```

等 Worker 返回 `finished_recving` 后，后续 schedule 再把请求恢复。

## 23. SchedulerOutput 是什么

`SchedulerOutput` 是本轮执行计划。

核心字段：

```text
scheduled_new_reqs
scheduled_cached_reqs
num_scheduled_tokens
total_num_scheduled_tokens
scheduled_spec_decode_tokens
scheduled_encoder_inputs
num_common_prefix_blocks
preempted_req_ids
finished_req_ids
kv_connector_metadata
ec_connector_metadata
new_block_ids_to_zero
has_structured_output_requests
```

一句话：

```text
SchedulerOutput 是 Scheduler 发给 Worker 的本轮执行说明书，也是回收结果时的对账凭证。
```

## 24. _update_after_schedule

构造 SchedulerOutput 后，Scheduler 会立即更新内部状态。

核心动作：

```text
request.num_computed_tokens += num_scheduled_token
```

这叫乐观推进。

为什么是乐观：

```text
spec decode 中 draft token 可能被拒绝。
如果被拒绝，update_from_output() 会回退 num_computed_tokens。
```

## 25. update_from_output 总览

`update_from_output()` 负责收结果。

输入：

```text
SchedulerOutput
ModelRunnerOutput
```

输出：

```text
dict[client_index, EngineCoreOutputs]
```

主流程：

```text
1. 处理 deferred free。
2. 处理 KV load 失败 block。
3. 遍历本轮调度的 request。
4. 根据 req_id_to_index 找到输出。
5. 取 sampled token。
6. 处理 spec decode 接受 / 拒绝。
7. append output tokens。
8. 检查 stop。
9. 推进 structured output grammar。
10. 处理 logprobs / prompt logprobs / pooling output。
11. 结束请求释放资源。
12. 构造 EngineCoreOutput。
13. 按 client_index 分组返回 EngineCoreOutputs。
```

## 26. sampled token 处理

每个请求根据：

```text
model_runner_output.req_id_to_index[req_id]
```

找到对应输出。

普通 decode：

```text
通常每个请求 1 个 sampled token。
```

chunked prefill：

```text
可能没有 sampled token。
```

spec decode：

```text
每个请求一轮可能输出多个 token。
```

append 后检查：

```text
EOS
stop token
max tokens
max model len
repetition detection
```

## 27. spec decode 回账

如果本轮调度了 draft tokens：

```text
scheduled_spec_token_ids
  + generated_token_ids
  → num_accepted
  → num_rejected
```

被拒绝的 token 要回滚：

```text
request.num_computed_tokens -= num_rejected
request.num_output_placeholders -= num_rejected
```

一句话：

```text
Scheduler 会把被 target model 拒绝的 speculative token 从计算进度账本里扣掉。
```

## 28. structured output

Scheduler 会生成 grammar bitmask：

```text
get_grammar_bitmask(scheduler_output)
```

只对：

```text
使用 structured output
且不是 prefill chunk 的请求
```

采样后 Scheduler 推进 grammar 状态：

```text
grammar.accept_tokens(req_id, new_token_ids)
```

边界：

```text
Scheduler 管 grammar 状态。
ModelRunner 用 grammar bitmask 屏蔽 logits。
```

## 29. 多模态 encoder 调度

Scheduler 通过 `_try_schedule_encoder_inputs()` 判断本轮是否需要 encoder output。

判断依据：

```text
本轮 decoder token window 是否覆盖某个 placeholder。
encoder output 是否已缓存。
encoder compute budget 是否足够。
encoder cache 是否有空间。
是否允许切分多模态输入。
远端 EC connector 是否命中。
```

输出到：

```text
SchedulerOutput.scheduled_encoder_inputs
SchedulerOutput.free_encoder_mm_hashes
```

## 30. 请求结束与资源释放

请求结束可能来自：

```text
正常 stop
用户 abort
错误
pooling 输出完成
structured output error
```

结束时：

```text
从 running / waiting / skipped_waiting 移除
设置 finished status
释放 encoder cache
释放 KV blocks
通知 KV connector request_finished
添加 finished_req_ids
```

如果 KV connector 要保存 KV：

```text
可能 delay free，等 finished_sending 后再释放 blocks。
```

## 31. 常见易混点

### running 优先于 waiting

```text
因为 running 已占 KV block，继续推进更有利于释放资源和降低延迟。
```

### num_computed_tokens 不等于真实输出 token 数

```text
它是 Scheduler 的计算进度账本，spec decode / async scheduling 下可能乐观推进或回退。
```

### prefix cache 命中不等于不跑 forward

```text
即使 prompt 基本命中，通常也要重算最后 token / block 来获得 next-token logits。
```

### SchedulerOutput 不是 Worker 输出

```text
SchedulerOutput 是计划。
ModelRunnerOutput 是执行结果。
EngineCoreOutputs 是 Scheduler 消化结果后的内部输出。
```

## 32. 与其他专题的关系

```text
engine_core：EngineCore 调用 Scheduler.schedule 和 update_from_output。
executor_worker_model_runner：Worker 消费 SchedulerOutput。
attention：Scheduler 分配 block，ModelRunner 构造 attention metadata。
sampling_and_output：ModelRunnerOutput 由 Scheduler 消化后再进入 OutputProcessor。
spec_decode：Scheduler 负责 draft tokens 调度和 rejected token 回滚。
multimodal：Scheduler 负责 encoder input budget 和 encoder cache。
kv_cache_transfer：Scheduler 接入 KV Connector 和外部 KV load/save。
```

## 33. 背诵总结

背这一段：

```text
Scheduler 是 vLLM V1 的调度和请求状态账本中心。它维护 requests、waiting、skipped_waiting 和 running，每轮 schedule 先调度 running 请求，再在没有抢占时调度 waiting 请求。它用 token_budget 控制每轮 token 数，用 KVCacheManager 查询 prefix cache 和分配 KV block，用 connector 查询外部 KV，用 encoder budget 调度多模态 encoder，用 spec token 状态调度 speculative decoding。SchedulerOutput 是本轮计划；Worker 返回 ModelRunnerOutput 后，Scheduler.update_from_output 把 sampled tokens、logprobs、pooling、grammar、KV connector 和资源释放全部对账回 Request 状态，并生成 EngineCoreOutputs。
```
