# 14 spec_decode 背诵文档

## 1. 专题定位

`spec_decode` 讲的是 vLLM V1 speculative decoding 如何用便宜路径先猜 token，再用 target model 验证，从而提高生成吞吐。

它不是单纯 sampler 分支。

一句话：

```text
Spec decode 是一条跨 Scheduler、ModelRunner、Sampler、KV cache、grammar 和输出回收的多 token 验证协议。
```

## 2. 最小心智模型

核心思想：

```text
用 drafter 先猜多个 draft tokens，
再用 target model 一次 forward 验证这些 tokens，
通过 rejection sampling 接受一段前缀，
拒绝时采 recovered token，
全部接受时补 bonus token，
最后把 Request、KV cache、grammar、logprobs 和输出状态修正一致。
```

最小链路：

```text
SpeculativeConfig
  → Scheduler.num_spec_tokens / num_lookahead_tokens
  → Request.spec_token_ids
  → SchedulerOutput.scheduled_spec_decode_tokens
  → InputBatch.spec_token_ids
  → SpecDecodeMetadata
  → target model forward / logits
  → RejectionSampler
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler.update_from_output()
  → Request.output_token_ids / num_computed_tokens 修正
  → proposer 产生下一轮 DraftTokenIds
  → Scheduler.update_draft_token_ids()
```

## 3. 它解决什么问题

普通 decode：

```text
当前上下文
  → target model forward
  → logits
  → Sampler
  → 1 个 token
```

spec decode：

```text
当前上下文
  → drafter 先猜 [d1, d2, d3, ...]
  → target model 一次 forward 验证这些 draft tokens
  → 接受尽可能长的 draft prefix
  → 如果拒绝，采 recovered token
  → 如果全接受，额外采 bonus token
```

收益：

```text
一次 target model forward 可能推进多个 token。
```

复杂度：

```text
draft tokens 被猜过、调度过、forward 过，不代表它们一定是最终输出。
```

## 4. 关键术语

```text
candidate tokens：drafter 猜出来的候选。
verified tokens：target model 验证过的位置。
accepted tokens：真正被接受的 draft prefix。
recovered token：draft 被拒绝后从 target / residual 分布采出的替代 token。
bonus token：所有 draft 都接受后额外采样的 token。
finalized tokens：写入 Request.output_token_ids 的真实输出。
```

一句话：

```text
spec decode 的收益来自多 token 验证，复杂度来自只提交 accepted / recovered / bonus 后的真实 token。
```

## 5. 和普通 decode 的差异

```text
请求状态：
  普通 decode 只看 prompt + output tokens。
  spec decode 还要暂存 Request.spec_token_ids。

Scheduler token 模型：
  普通 decode 追 num_tokens。
  spec decode 追 num_tokens_with_spec。

KV allocation：
  普通 decode 为本轮 token 分配 slot。
  spec decode 还要传 num_lookahead_tokens。

Worker batch：
  普通 decode 每请求通常追加 1 token。
  spec decode 在真实 token 后追加 draft tokens。

logits 布局：
  普通 decode 通常每请求 1 行 logits。
  spec decode 有 target verification rows + bonus rows。

sampler：
  普通 decode 用 Sampler。
  spec decode 用 RejectionSampler。

状态回收：
  普通 decode append sampled token。
  spec decode 统计 accepted / rejected，回滚 rejected，append真实 generated tokens。
```

## 6. 两个闭环

Spec decode 有两个相连闭环。

### 验证闭环

```text
Request.spec_token_ids
  → Scheduler.schedule()
  → SchedulerOutput.scheduled_spec_decode_tokens
  → ModelRunner target verification forward
  → RejectionSampler
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler.update_from_output()
  → Request.output_token_ids / num_computed_tokens
```

回答：

```text
上一轮猜的 tokens，这一轮接受几个？
```

### 生成闭环

```text
本轮 sampled_token_ids / hidden_states / metadata
  → drafter.propose(...)
  → GPUModelRunner._draft_token_ids
  → take_draft_token_ids()
  → DraftTokenIds
  → Scheduler.update_draft_token_ids()
  → Request.spec_token_ids
```

回答：

```text
下一轮要让 target model 验哪些 tokens？
```

## 7. SpeculativeConfig

`SpeculativeConfig` 是控制面入口。

它描述：

```text
是否开启 speculative decoding
使用哪种 method
每轮最多 draft 多少 token
是否使用 draft model
draft model 的并行 / 量化 / attention backend
rejection sampling 方法
dynamic speculative decoding 策略
```

关键字段：

```text
num_speculative_tokens
model
method
draft_tensor_parallel_size
quantization
moe_backend
attention_backend
max_model_len
prompt_lookup_max / prompt_lookup_min
parallel_drafting
num_speculative_tokens_per_batch_size
rejection_sample_method
draft_sample_method
```

## 8. method 决定 drafter 类型

常见 drafter 方法：

```text
ngram
ngram_gpu
draft_model
dflash
suffix
eagle / eagle3
medusa
custom_class
extract_hidden_states
模型特化 MTP 路径
```

一句话：

```text
SpeculativeConfig 决定用什么 speculative 方法；GPUModelRunner 把它落成运行时 drafter 和 rejection sampler。
```

## 9. num_speculative_tokens 的影响

它不只是性能参数。

它影响：

```text
Scheduler 默认让 proposer 生成几个 draft tokens
Scheduler.num_lookahead_tokens
RejectionSampler 输出矩阵宽度 max_spec_len + 1
InputBatch 预留 spec token 空间
async spec decode placeholders
CUDA graph / padding / attention metadata 最大 shape
```

一句话：

```text
num_speculative_tokens 既是吞吐参数，也是 shape 和状态边界参数。
```

## 10. Scheduler 侧状态

Request 上有：

```text
spec_token_ids：等待下一轮验证的 draft tokens。
num_tokens：prompt + accepted output。
num_tokens_with_spec：num_tokens + len(spec_token_ids)。
num_computed_tokens：Scheduler 的计算进度账本。
num_output_placeholders：async / PP 场景的输出占位。
```

关键区别：

```text
num_tokens 是真实 token 边界。
num_tokens_with_spec 是真实 token + pending draft 的候选边界。
num_computed_tokens schedule 后可能乐观包含 draft，后续可回滚。
```

## 11. Scheduler 的统一 token 模型

Scheduler 没有单独的 spec phase。

它只是让：

```text
request.num_computed_tokens 追上 request.num_tokens_with_spec
```

running 请求公式：

```text
num_new_tokens = request.num_tokens_with_spec
               + request.num_output_placeholders
               - request.num_computed_tokens
```

如果 `spec_token_ids` 非空，`num_new_tokens` 可能覆盖多个 draft verification tokens。

## 12. scheduled_spec_decode_tokens

当请求带 draft tokens 时，Scheduler 会找出本轮实际调度到的 spec 区间。

然后写入：

```text
SchedulerOutput.scheduled_spec_decode_tokens[request_id] = spec_token_ids
```

并清空：

```text
request.spec_token_ids = []
```

含义：

```text
Request.spec_token_ids 是跨 step 暂存的 draft tokens。
SchedulerOutput.scheduled_spec_decode_tokens 是本轮真实要验证的 draft tokens。
调度后清空，避免重复验证。
```

## 13. num_lookahead_tokens

某些 speculative 方法需要 KV lookahead。

Scheduler 分配 KV slots 时传：

```text
num_lookahead_tokens
```

例如：

```text
EAGLE / draft model：通常等于 num_spec_tokens。
DFlash：可能是 num_spec_tokens + 1。
```

重要点：

```text
spec decode 会影响 KV block admission，不只是采样层优化。
```

## 14. SchedulerOutput 中 spec 字段

直接相关字段：

```text
scheduled_spec_decode_tokens：本轮要验证的 draft tokens。
num_invalid_spec_tokens：非法 draft tokens 统计 / 修正。
num_spec_tokens_to_schedule：下一轮 proposer 应生成的 draft token 数。
```

间接受影响字段：

```text
num_scheduled_tokens
total_num_scheduled_tokens
scheduled_cached_reqs
new_block_ids_to_zero
has_structured_output_requests
```

一句话：

```text
SchedulerOutput 是 spec decode 从 Scheduler 进入 Worker / ModelRunner 的出站协议。
```

## 15. EngineCore 的作用

EngineCore 基础闭环仍是：

```text
schedule → execute_model → sample_tokens → update_from_output
```

spec decode 增加 post-step draft 回写：

```text
model_executor.take_draft_token_ids()
  → Scheduler.update_draft_token_ids()
  → Request.spec_token_ids
```

要记住：

```text
EngineCore 把 target 验证闭环和下一轮 draft 生成闭环串起来。
```

## 16. ModelRunner 的 spec 职责

GPUModelRunner 负责：

```text
创建 drafter
创建 RejectionSampler
接收 SchedulerOutput.scheduled_spec_decode_tokens
把 draft tokens 写入 InputBatch
准备 input_ids / positions / slot mapping
构造 SpecDecodeMetadata
执行 target model forward
计算 target / bonus logits
在 sample_tokens() 中调用 RejectionSampler
调用 proposer 生成下一轮 DraftTokenIds
```

为什么只在 last PP rank 做：

```text
只有 last PP rank 拿到最终 hidden states / logits，才能采样、rejection sampling 和 draft proposal。
```

## 17. InputBatch 中 draft tokens 如何放

`_update_states()` 读取：

```text
scheduler_output.scheduled_spec_decode_tokens
```

然后：

```text
InputBatch.update_req_spec_token_ids(req_state, spec_tokens)
```

Worker 侧把 spec tokens 写到真实 token 后面：

```text
真实 token ids 后追加 draft token ids
```

一句话：

```text
spec tokens 在 Worker 侧不是另起 batch，而是追加在同一 request row 的真实 token 后面。
```

## 18. SpecDecodeMetadata

`SpecDecodeMetadata` 是本轮 logits 行号说明书。

字段：

```text
draft_token_ids：flatten 后 draft token ids。
num_draft_tokens：每个请求有几个 draft tokens。
cu_num_draft_tokens：draft tokens 前缀和。
cu_num_sampled_tokens：每个请求最多输出 draft_len + 1 的前缀和。
target_logits_indices：用于验证 draft tokens 的 logits rows。
bonus_logits_indices：用于 bonus sampling 的 logits rows。
logits_indices：从 full hidden states 中取哪些 rows 计算 logits。
```

一句话：

```text
SpecDecodeMetadata 不保存长期请求状态，它保存本轮 spec decode 的 logits 行号布局。
```

## 19. execute_model：target verification forward

spec decode 的 target forward 仍然是普通 ModelRunner execute_model。

主线：

```text
_update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
  → _model_forward()
  → hidden_states[logits_indices]
  → model.compute_logits()
  → ExecuteModelState(..., spec_decode_metadata, logits)
  → return None
```

为什么返回 None：

```text
execute_model 只做 forward / logits，sample_tokens 再做 grammar、RejectionSampler、bookkeeping 和 draft proposal。
```

## 20. sample_tokens：分流 Sampler / RejectionSampler

`sample_tokens()` 先应用 grammar bitmask。

然后：

```text
如果 spec_decode_metadata is None：
  走普通 Sampler。

如果 spec_decode_metadata 存在：
  走 RejectionSampler。
```

RejectionSampler 输入：

```text
SpecDecodeMetadata
draft_probs
target logits
sampling metadata
```

输出：

```text
accepted / recovered / bonus 后的 sampled_token_ids
```

## 21. RejectionSampler 规则

从左到右验证 draft tokens：

```text
接受：输出该 draft token，继续验证下一个。
拒绝：输出 recovered token，后续 draft 不再输出。
全部接受：输出所有 draft tokens，再追加 bonus token。
```

输出 tensor 通常形状：

```text
[batch_size, max_spec_len + 1]
```

其中：

```text
真实 token id：accepted / recovered / bonus。
-1：padding 或 rejected 后无效位置。
```

## 22. _bookkeeping_sync

RejectionSampler 输出通常仍是 GPU tensor。

`_bookkeeping_sync()` 负责：

```text
过滤 -1 padding
生成 valid_sampled_token_ids
生成 logprobs_lists
构造 ModelRunnerOutput.sampled_token_ids
处理 prompt_logprobs / logprobs
```

Scheduler 看到的 sampled_token_ids 已经是真实输出 token list。

## 23. Scheduler.update_from_output 回账

Scheduler 消费结果：

```text
generated_token_ids = sampled_token_ids[req_index]
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens[req_id]
```

计算：

```text
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = len(scheduled_spec_token_ids) - num_accepted
```

回滚：

```text
request.num_computed_tokens -= num_rejected
request.num_output_placeholders -= num_rejected
```

提交真实输出：

```text
request.append_output_token_ids(generated_token_ids)
```

一句话：

```text
Worker 返回真实输出 token；Scheduler 用原计划 draft tokens 计算接受 / 拒绝，并修正账本。
```

## 24. KV cache 和 num_computed_tokens

spec decode 中，Scheduler schedule 后会乐观推进 `num_computed_tokens`。

如果 draft token 被拒绝：

```text
这些 rejected token 对应的计算进度不能算作真实可复用上下文。
```

所以必须：

```text
回退 num_computed_tokens
修正 output placeholders
后续 attention / KV cache 只能基于 accepted tokens 继续
```

关键：

```text
num_computed_tokens 是调度账本，不是最终 output token 数。
```

## 25. structured output 交互

结构化输出会影响 spec decode。

Scheduler：

```text
生成 grammar bitmask。
validate / filter draft tokens。
采样后推进 grammar.accept_tokens。
```

ModelRunner：

```text
在 logits 上应用 grammar bitmask。
RejectionSampler 在合法空间内采样。
```

如果 grammar 拒绝某些 token：

```text
需要记录 invalid spec tokens，避免状态不一致。
```

## 26. dynamic speculative decoding

如果配置按 batch size 动态决定 draft 数：

```text
num_speculative_tokens_per_batch_size
```

Scheduler 会设置：

```text
SchedulerOutput.num_spec_tokens_to_schedule
```

它含义是：

```text
本轮执行后，proposer 下一轮应该最多生成几个 draft tokens。
```

不是本轮验证了几个 draft tokens。

## 27. 常见边界

```text
chunked prefill 通常不直接输出 sampled token，spec decode 主要作用于 decode。
PP 下只有 last rank 做 rejection sampler 和 proposer。
KV connector / async scheduling 可能让 num_computed_tokens 和真实输出存在延迟。
structured output 会限制 draft tokens 合法性。
logprobs 在 spec decode 下长度不固定。
CUDA graph shape 受 num_speculative_tokens 影响。
```

## 28. 常见易混点

### draft tokens 不是最终输出

```text
draft 被验证后，只有 accepted 部分可能输出。
```

### sampled_token_ids 不是原始 draft

```text
它是 accepted / recovered / bonus 后真实要提交的 token。
```

### Request.spec_token_ids 会被清空

```text
调度到 SchedulerOutput 后清空，避免重复验证。
```

### num_spec_tokens_to_schedule 不是本轮验证数

```text
它是下一轮 proposer 的建议 draft 数。
```

### spec decode 影响 KV allocation

```text
num_lookahead_tokens 会进入 KVCacheManager.allocate_slots。
```

## 29. 与其他专题的关系

```text
scheduler：draft tokens 调度、num_computed_tokens 回滚、update_from_output。
executor_worker_model_runner：ModelRunner 如何构造 spec forward 输入。
sampling_and_output：RejectionSampler 如何产生 sampled_token_ids。
attention：spec decode 改变 query length、positions、slot mapping 和 metadata。
kv_cache_transfer：spec decode 与 KV lookahead / connector 可能交互。
compilation_and_cuda_graph：num_speculative_tokens 影响 shape 和 graph key。
structured output：grammar bitmask 和 draft token 合法性。
```

## 30. 背诵总结

背这一段：

```text
Spec decode 的核心是 drafter 猜 token，target model 批量验证，RejectionSampler 决定接受、恢复或补 bonus，Scheduler 再把真实输出和状态账本修正一致。Scheduler 用 Request.spec_token_ids 保存下一轮候选，用 num_tokens_with_spec 把 draft 纳入调度，用 SchedulerOutput.scheduled_spec_decode_tokens 下发本轮验证 tokens，并为 KV allocation 传 num_lookahead_tokens。ModelRunner 把 draft tokens 追加到 InputBatch 中，构造 SpecDecodeMetadata 指明 target logits 和 bonus logits 的行号，target forward 后 sample_tokens 调 RejectionSampler 产出真实 sampled_token_ids。Scheduler.update_from_output 根据原 scheduled draft 和真实输出计算 accepted / rejected，回退 rejected 对应的 num_computed_tokens，append accepted / recovered / bonus tokens，并回写下一轮 DraftTokenIds。
```
