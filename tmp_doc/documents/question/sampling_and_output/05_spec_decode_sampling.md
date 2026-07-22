# 05. Spec decode 如何影响采样和输出？

源码位置：

- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/sched/output.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/outputs.py`
- `vllm/vllm/v1/request.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/spec_decode/metadata.py`
- `vllm/vllm/v1/spec_decode/metrics.py`
- `vllm/vllm/v1/spec_decode/utils.py`
- `vllm/vllm/v1/spec_decode/draft_model.py`
- `vllm/vllm/v1/spec_decode/eagle.py`
- `vllm/vllm/v1/spec_decode/ngram_proposer.py`
- `vllm/vllm/v1/spec_decode/ngram_proposer_gpu.py`
- `vllm/vllm/v1/spec_decode/llm_base_proposer.py`
- `vllm/vllm/v1/sample/rejection_sampler.py`
- `vllm/vllm/v1/sample/sampler.py`
- `vllm/vllm/v1/sample/metadata.py`
- `vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- `vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`

本问题关注：speculative decoding 下，draft token、target logits、bonus token、rejection sampler、grammar / logits processors、logprobs、ModelRunnerOutput、Scheduler 状态回收之间的关系。

---

## 1. 一句话回答

Spec decode 让一轮采样从“每个 request 采一个 token”变成：

```text
先用 drafter 预测一串 draft tokens；
再让 target model 一次性验证这些 draft tokens；
rejection sampler 决定接受多少 draft tokens；
如果中途拒绝，就从修正分布采 recovered token；
如果全部接受，就额外输出 target model 采样的 bonus token；
最后 Scheduler 根据实际输出 token 数回滚 rejected token 的计算状态。
```

最核心的区别是：

```text
普通 decode：logits → sampler → 1 个 token
spec decode：target logits + draft probs + draft token ids → rejection sampler → 0 到 num_draft+1 个 token
```

一句话记忆：

```text
Spec decode 的采样不是“直接相信 draft”，而是“target model 批量验证 draft，并把 accepted / recovered / bonus tokens 作为本轮真实输出”。
```

---

## 2. 本文要回答的问题

```text
Scheduler 如何把 draft tokens 调度进本轮执行？
SpecDecodeMetadata 里的 logits_indices / target_logits_indices / bonus_logits_indices 分别是什么？
target model 为什么要为 num_draft + 1 个位置算 logits？
rejection sampler 如何接受 / 拒绝 draft token？
recovered token 和 bonus token 分别来自哪里？
grammar / structured output 如何约束 spec decode？
logprobs 如何对应 accepted / recovered / bonus token？
ModelRunnerOutput.sampled_token_ids 在 spec decode 下为什么是 list[list[int]]？
Scheduler.update_from_output() 如何根据 rejected tokens 回滚 num_computed_tokens？
下一轮 draft tokens 如何从 worker 回到 Scheduler？
```

---

## 3. 最小主链路

Spec decode 与采样 / 输出的主链路可以压缩为：

```text
上一轮 sample_tokens()
  → drafter 生成 draft_token_ids
  → EngineCore.post_step()
  → Scheduler.update_draft_token_ids()
  → request.spec_token_ids = draft_token_ids

本轮 Scheduler.schedule()
  → request.num_tokens_with_spec 包含 spec_token_ids
  → scheduled_spec_decode_tokens[req_id] = 本轮要验证的 draft tokens
  → SchedulerOutput.scheduled_spec_decode_tokens

ModelRunner._prepare_inputs()
  → 把 draft tokens 放进 input_ids / positions
  → 计算 SpecDecodeMetadata
  → logits_indices = target model 需要取 logits 的位置

Target model forward
  → logits[logits_indices]

sample_tokens()
  → apply_grammar_bitmask()
  → _sample(logits, spec_decode_metadata)
  → RejectionSampler
      → accepted tokens
      → recovered token（如果发生拒绝）
      → bonus token（如果全部接受）
  → RejectionSampler.parse_output()
  → ModelRunnerOutput.sampled_token_ids

Scheduler.update_from_output()
  → 计算 accepted / rejected 数
  → 回滚 rejected draft 对 num_computed_tokens 的影响
  → append 真实输出 tokens
  → 生成 EngineCoreOutput
```

---

## 4. Scheduler 侧：draft tokens 如何进入调度

### 4.1 Request 中保存 spec_token_ids

每个 `Request` 有：

```python
self.spec_token_ids: list[int] = []
```

位置：`vllm/vllm/v1/request.py:167`

同时 `num_tokens_with_spec` 定义为：

```python
return len(self._all_token_ids) + len(self.spec_token_ids)
```

位置：`vllm/vllm/v1/request.py:266`

这表示：

```text
对 Scheduler 来说，draft tokens 会临时被看成 request 需要追赶计算的 token。
```

### 4.2 Scheduler 没有单独的 decode / prefill 状态

`Scheduler.schedule()` 注释里明确说明：

```text
Scheduler 不区分 decoding phase / prefill phase；
每个 request 只有 num_computed_tokens 和 num_tokens_with_spec；
调度目标是让 num_computed_tokens 追上 num_tokens_with_spec。
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:433`

这套抽象能同时覆盖：

```text
chunked prefill
prefix caching
speculative decoding
future jump decoding
```

### 4.3 schedule() 计算本轮要验证哪些 spec tokens

当 request 有 `spec_token_ids` 时：

```python
num_scheduled_spec_tokens = (
    num_new_tokens
    + request.num_computed_tokens
    - request.num_tokens
    - request.num_output_placeholders
)
```

如果这个值大于 0，就把对应 draft tokens 放入：

```python
scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:631`

随后清空 request 上旧的 spec tokens：

```python
request.spec_token_ids = []
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:646`

含义是：

```text
本轮已经决定要验证这些 draft tokens；
新的 draft tokens 会在下一轮 sample 后重新写回 request。
```

### 4.4 SchedulerOutput 显式携带 scheduled_spec_decode_tokens

`SchedulerOutput` 里有字段：

```python
scheduled_spec_decode_tokens: dict[str, list[int]]
```

位置：`vllm/vllm/v1/core/sched/output.py:183`

注释说明：

```text
如果某个 request 没有 spec decode tokens，它不会出现在这个 dict 里。
```

这就是 Worker / ModelRunner 识别本轮是否走 spec decode 的入口。

---

## 5. ModelRunner 输入准备阶段的变化

### 5.1 是否启用 spec decode 看 scheduled_spec_decode_tokens

`GPUModelRunner._prepare_inputs()` 中：

```python
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2202`

如果没有 spec decode：

```text
logits_indices = 每个 request 的最后一个 token 位置
num_sampled_tokens = 1
spec_decode_metadata = None
```

如果有 spec decode：

```text
num_draft_tokens[req_idx] = len(draft_token_ids)
spec_decode_metadata = _calc_spec_decode_metadata(...)
logits_indices = spec_decode_metadata.logits_indices
num_sampled_tokens = num_draft_tokens + 1
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2229`

### 5.2 为什么 num_sampled_tokens = num_draft_tokens + 1

对每个 spec decode request，本轮 target model 需要：

```text
1. 为每个 draft token 生成一个 target logits，用于接受 / 拒绝判断；
2. 再为最后一个位置生成一个 bonus logits，用于“全部接受 draft 后再前进一个 token”。
```

所以：

```text
num_sampled_tokens = num_draft_tokens + 1
```

这也是为什么 spec decode 下一次 sampling 可能输出多个 token。

### 5.3 draft tokens 会被放进 input_ids

准备输入时，ModelRunner 会把上一步生成并由 Scheduler 调度的 draft tokens 放到本轮 input_ids 对应位置。

在 async / batch 重排场景下，代码还会把 sampled tokens 和 draft tokens scatter 回 GPU 输入缓冲：

```text
prev_sampled_token_ids → input_ids
draft_token_ids → input_ids
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:1771`

这说明：

```text
spec decode 的 target model forward 真正会看到 draft token 序列，
并为这些位置计算 target logits。
```

---

## 6. SpecDecodeMetadata 是什么

`SpecDecodeMetadata` 定义在：`vllm/vllm/v1/spec_decode/metadata.py:10`

字段包括：

```text
draft_token_ids
num_draft_tokens
cu_num_draft_tokens
cu_num_sampled_tokens
target_logits_indices
bonus_logits_indices
logits_indices
max_spec_len
```

### 6.1 draft_token_ids

```text
shape: [num_draft_tokens_total]
```

表示本轮被验证的 draft token ids，跨 request flatten 后存储。

它来自 `input_ids.gpu[logits_indices]` 再按 target logits 位置取对应 draft token。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2874`

### 6.2 num_draft_tokens

```text
shape: [batch_size]
```

每个 request 本轮有多少 draft tokens。

例如：

```text
[3, 0, 2, 0, 1]
```

表示第 0 / 2 / 4 个 request 走 spec decode，其余 request 普通采样。

### 6.3 cu_num_draft_tokens

```text
shape: [batch_size]
```

是 draft tokens 的 cumulative sum，用于把每个 request 的 draft token span 映射到 flatten tensor。

例如：

```text
num_draft_tokens:    [3, 0, 2, 0, 1]
cu_num_draft_tokens: [3, 3, 5, 5, 6]
```

### 6.4 cu_num_sampled_tokens

每个 request 最多会产生：

```text
num_draft_tokens + 1
```

所以：

```text
num_sampled_tokens:    [4, 1, 3, 1, 2]
cu_num_sampled_tokens: [4, 5, 8, 9, 11]
```

这个数组用于把每个 request 的输出 token span 映射到 flatten logits / output。

### 6.5 logits_indices

`logits_indices` 是 target model 需要真正取 logits 的所有位置：

```text
每个 request 取 num_draft_tokens + 1 个 logits 位置。
```

源码示例注释：

```text
logits_indices: [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2814`

它最终会成为：

```text
spec_decode_metadata.logits_indices
```

并传给 logits 计算 / sampling。

### 6.6 target_logits_indices

`target_logits_indices` 指向 draft tokens 对应的 target logits：

```text
每个 draft token 都有一个 target logits 来判断它能否被接受。
```

源码示例：

```text
target_logits_indices: [0, 1, 2, 5, 6, 9]
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2855`

### 6.7 bonus_logits_indices

`bonus_logits_indices` 指向每个 request 的最后一个 logits：

```text
如果该 request 的所有 draft tokens 都被接受，
就从 bonus logits 采一个 bonus token。
```

源码中：

```python
bonus_logits_indices = cu_num_sampled_tokens - 1
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2846`

### 6.8 max_spec_len

`SpecDecodeMetadata.__post_init__()` 会记录：

```python
self.max_spec_len = max(self.num_draft_tokens)
```

位置：`vllm/vllm/v1/spec_decode/metadata.py:26`

rejection sampler 用它来确定输出矩阵形状：

```text
[batch_size, max_spec_len + 1]
```

---

## 7. 用例子理解 logits_indices / target / bonus

假设某个 request 有 3 个 draft tokens：

```text
当前真实序列：A
Draft proposes：d1 d2 d3
```

target model 本轮会看：

```text
A d1 d2 d3
```

并取出 4 个 logits：

```text
logits_0：用于判断 d1 是否可接受
logits_1：用于判断 d2 是否可接受
logits_2：用于判断 d3 是否可接受
logits_3：如果 d1 d2 d3 全部接受，用于采 bonus token
```

因此：

```text
target_logits_indices = [logits_0, logits_1, logits_2]
bonus_logits_indices  = [logits_3]
logits_indices        = [logits_0, logits_1, logits_2, logits_3]
```

如果 d2 被拒绝：

```text
输出 = [d1, recovered_token]
后面的 d2/d3 不会成为真实输出。
```

如果全部接受：

```text
输出 = [d1, d2, d3, bonus_token]
```

---

## 8. _sample() 如何分流普通采样和 spec decode

`GPUModelRunner._sample()` 的入口：

```python
def _sample(
    self,
    logits: torch.Tensor | None,
    spec_decode_metadata: SpecDecodeMetadata | None,
) -> SamplerOutput:
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3623`

如果没有 spec decode：

```python
return self.sampler(logits=logits, sampling_metadata=sampling_metadata)
```

如果有 spec decode：

```python
draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
sampler_output = self.rejection_sampler(
    spec_decode_metadata,
    draft_probs,
    logits,
    sampling_metadata,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3645`

因此：

```text
普通采样走 Sampler；
spec decode 走 RejectionSampler；
```

但 RejectionSampler 内部仍会调用 Sampler 来采 bonus token，并复用 sampling metadata。

---

## 9. RejectionSampler 的核心职责

`RejectionSampler` 定义在：`vllm/vllm/v1/sample/rejection_sampler.py:37`

文件注释把几个术语分得很清楚：

```text
accepted tokens：
  根据 raw draft probability 和 target probability 接受的 draft tokens。

recovered tokens：
  某个 draft token 被拒绝后，从 adjusted probability distribution 重新采样的 token。

bonus tokens：
  如果所有 draft tokens 都被接受，则追加一个只来自 target probabilities 的 token。

output tokens：
  accepted tokens + recovered token + bonus token。
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:37`

### 9.1 RejectionSampler.forward 的输入

```text
metadata：
  SpecDecodeMetadata。

draft_probs：
  draft model 对 draft tokens 的 probability distribution。
  ngram spec decode 场景下可能为 None。

logits：
  target model 的 logits，shape 是 [num_draft_tokens_total + batch_size, vocab_size]。

sampling_metadata：
  temperature / top-k / top-p / penalties / logprobs / generator 等采样参数。
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:88`

### 9.2 bonus token 先用普通 sampler 采出来

代码先取：

```python
bonus_logits = logits[bonus_logits_indices]
bonus_sampler_output = self.sampler(..., predict_bonus_token=True)
bonus_token_ids = bonus_sampler_output.sampled_token_ids
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:121`

注意：

```text
bonus token 是 target model 的普通采样结果；
它可以使用 top-p / top-k 等采样策略；
而 draft token 的接受 / 拒绝本身不等同于普通采样。
```

源码注释也说明：

```text
bonus token 传入 rejection sampler，而不是在 rejection sampler 内部直接采，
是为了让 bonus token 能使用更灵活的采样流程。
```

### 9.3 target logits 会应用 logits processors 和 sampling constraints

对 draft token 对应的 target logits：

```python
target_logits = self.apply_logits_processors(...)
target_logits = apply_sampling_constraints(...)
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:157`

这里会处理：

```text
penalties
allowed token ids
bad words
non-argmax-invariant logits processors
MinTokensLogitsProcessor
thinking budget state
temperature
top-k / top-p
```

### 9.4 调用 rejection_sample 得到输出矩阵

最后调用：

```python
output_token_ids = rejection_sample(
    metadata.draft_token_ids,
    metadata.num_draft_tokens,
    metadata.max_spec_len,
    metadata.cu_num_draft_tokens,
    draft_probs,
    target_logits,
    bonus_token_ids,
    sampling_metadata,
    ...
)
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:169`

返回形状是：

```text
[batch_size, max_spec_len + 1]
```

被拒绝后不再有效的位置会填：

```text
PLACEHOLDER_TOKEN_ID = -1
```

---

## 10. 接受 / 拒绝算法怎么理解

### 10.1 Greedy 情况

如果请求是 greedy sampling：

```text
draft token 被接受，当且仅当 draft_token_id == target_argmax_id。
```

对应 kernel 中：

```text
rejected = draft_token_id != target_argmax_id
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:743`

如果发生拒绝：

```text
输出 target_argmax_id，后续 draft tokens 失效。
```

如果全部接受：

```text
追加 bonus token。
```

### 10.2 Random sampling 情况

random sampling 下，标准 speculative decoding 接受概率是：

```text
accept draft token x with probability min(1, p_target(x) / p_draft(x))
```

源码中对应：

```text
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:816`

如果被拒绝，就从 recovered distribution 采一个 token。

### 10.3 recovered token 从哪里来

`sample_recovered_tokens()` 会用 target_probs 和 draft_probs 构造修正分布，并采样 recovered token。

位置：`vllm/vllm/v1/sample/rejection_sampler.py:663`

直观理解：

```text
如果 draft 分布某个 token 过度乐观导致拒绝，
recovered token 需要从 target - draft 修正后的剩余概率中采样，
从而保持整体分布与 target model 一致。
```

### 10.4 ngram spec decode 没有 draft_probs 怎么办

`draft_probs` 可以为 None。

源码注释说明：

```text
draft_probs can be None if probabilities are not provided,
which is the case for ngram spec decode.
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:98`

在 random kernel 中，如果没有 draft_probs：

```text
draft_prob = 1
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:816`

这是一种特殊 fallback / 近似路径，适配没有完整 draft probability distribution 的 proposer。

---

## 11. output_token_ids 为什么有 -1 placeholder

rejection sampler 输出矩阵形状固定：

```text
[batch_size, max_spec_len + 1]
```

但每个 request 实际输出长度不同：

```text
拒绝第 0 个 draft：输出 1 个 recovered token；
接受 2 个后拒绝第 3 个：输出 3 个 token；
全部接受：输出 num_draft + 1 个 token；
没有 draft：输出 1 个普通 / bonus token。
```

为了保持 GPU tensor 形状固定，rejection sampler 会把无效位置填成：

```text
PLACEHOLDER_TOKEN_ID = -1
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:30`

随后 `parse_output()` 会过滤这些placeholder。

位置：`vllm/vllm/v1/sample/rejection_sampler.py:248`

---

## 12. logprobs 如何处理

### 12.1 spec decode 下 logprobs 需要重新对齐

普通采样中，一个 request 一般只有一个 sampled token。

spec decode 中，一个 request 可能输出：

```text
多个 accepted draft tokens
一个 recovered token
一个 bonus token
```

因此 logprobs 不能简单使用原始 logits 行顺序。

### 12.2 RejectionSampler 会构造 final_logits

`_get_logprobs_tensors()` 中：

```python
final_logits[target_logits_indices] = target_logits
final_logits[bonus_logits_indices] = bonus_logits
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:199`

然后根据输出 token 位置 gather logprobs。

### 12.3 rejected token 的 logprobs 会先算后过滤

源码注释说明：

```text
为了避免 CPU-GPU 同步，现在会为所有 draft tokens 计算 indices，
包括 rejected ones；rejected tokens 会在 parse_output 中过滤掉。
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:218`

### 12.4 parse_output 同时过滤 tokens 和 logprobs

`parse_output()` 会：

```text
1. 把 output_token_ids 拷到 CPU；
2. 构造 valid_mask：token != -1 且 token < vocab_size；
3. 用 valid_mask 过滤 logprobs_tensors；
4. 转成 LogprobsLists；
5. 返回 list[list[int]] 和对应 logprobs。
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:248`

这就是为什么最终 `ModelRunnerOutput.sampled_token_ids` 是可变长度的 `list[list[int]]`。

---

## 13. grammar / structured output 如何影响 spec decode

### 13.1 grammar bitmask 仍在采样前应用

`sample_tokens()` 中先执行：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4512`

这意味着：

```text
结构化输出约束会先限制 target logits，
再进入普通 sampler 或 rejection sampler。
```

### 13.2 draft tokens 也需要 grammar validation

当 worker 生成下一轮 draft tokens 后，EngineCore 会调用：

```python
draft_token_ids = self.model_executor.take_draft_token_ids()
self.scheduler.update_draft_token_ids(draft_token_ids)
```

位置：`vllm/vllm/v1/engine/core.py:519`

Scheduler 里：

```python
if self.structured_output_manager.should_advance(request):
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
request.spec_token_ids = spec_token_ids
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:2005`

也就是说：

```text
grammar 不仅约束 target sampling，
也会提前过滤下一轮 draft tokens，避免调度明显非法的 spec tokens。
```

### 13.3 deferred grammar bitmask 场景

在 batch queue / deferred sampling 场景中，如果当前需要先处理上一轮输出，EngineCore 会在计算 grammar bitmask 之前调用：

```python
update_draft_token_ids_in_output(draft_token_ids, deferred_scheduler_output)
```

位置：`vllm/vllm/v1/engine/core.py:613`

这个函数会：

```text
1. 把 draft tokens 截断到本轮 scheduled 数量；
2. 用 grammar validate；
3. 对非法 spec tokens 用 -1 padding；
4. 记录 num_invalid_spec_tokens。
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:2027`

这些 `-1` 会让 grammar bitmask 和后续统计知道哪些 draft tokens 无效。

---

## 14. sample_tokens() 中 spec decode 的完整位置

`sample_tokens()` 的 spec decode 相关步骤可以按顺序看：

```text
1. 解包 execute_model_state
   包括 logits、spec_decode_metadata、spec_decode_common_attn_metadata。

2. apply_grammar_bitmask()
   如果有 structured output。

3. _sample(logits, spec_decode_metadata)
   spec decode 下进入 RejectionSampler。

4. _update_states_after_model_execute()
   用 sampled_token_ids 更新 Worker 侧临时状态。

5. 清理旧 draft 缓存
   _draft_token_ids / _draft_probs / _draft_prob_req_ids / _draft_token_req_ids。

6. 根据 speculative_config 决定是否立即 propose_draft_token_ids()
   EAGLE / draft model / extract hidden states 等可以使用 GPU sampled tokens。

7. _bookkeeping_sync()
   把 sampler_output 变成 valid_sampled_token_ids / logprobs_lists。

8. 如果某些 proposer 需要 CPU tokens，则 bookkeeping 后再 propose_draft_token_ids()。

9. finalize_kv_connector()
   spec decode 下 draft model 可能也要使用 KV connector。

10. 构造 ModelRunnerOutput。
```

对应源码：`vllm/vllm/v1/worker/gpu_model_runner.py:4482`

---

## 15. draft proposer 在 sample 阶段的位置

### 15.1 为什么 sample 后还要生成 draft tokens

本轮 rejection sampler 得到真实输出后，系统希望下一轮继续 spec decode。

所以 sample 阶段末尾会调用 drafter：

```text
当前真实输出 tokens
  → drafter proposes next draft tokens
  → DraftTokenIds
  → Scheduler.update_draft_token_ids()
  → 下一轮 scheduled_spec_decode_tokens
```

### 15.2 propose_draft_token_ids 的调用

`sample_tokens()` 内部定义：

```python
def propose_draft_token_ids(sampled_token_ids):
    self._draft_token_ids = self.propose_draft_token_ids(...)
    self._copy_draft_token_ids_to_cpu(scheduler_output)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4541`

它会把新的 draft tokens 缓存在 worker 里，等待 executor / EngineCore 调用：

```text
take_draft_token_ids()
```

### 15.3 哪些 proposer 可以直接用 GPU tokens

代码中 `use_gpu_toks` 包括：

```text
EAGLE
DraftModel
ExtractHiddenStates
```

以及对应的 proposer 类型：

```text
EagleProposer
DFlashProposer
DraftModelProposer
ExtractHiddenStatesProposer
Gemma4Proposer
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4571`

这些路径可以在 bookkeeping 前生成 draft，因为它们能直接使用 GPU 上的 sampled tokens。

### 15.4 ngram / 其他 CPU token 路径

对于 ngram 或其他需要 CPU sampled tokens 的方法：

```python
propose_drafts_after_bookkeeping = input_fits_in_drafter
```

bookkeeping 完成后再：

```python
propose_draft_token_ids(valid_sampled_token_ids)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4663`

### 15.5 如果输入不适合 drafter，要清零 draft tokens

如果超过 drafter max len 或其他条件导致不能继续 draft：

```python
self._draft_token_ids = torch.zeros(...).expand(...)
self._draft_probs = None
self._copy_draft_token_ids_to_cpu(..., zeros_only=True)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4633`

原因是：

```text
不能让 Scheduler 下一轮调度上一步遗留的 stale draft tokens。
```

源码注释还提到 Nemotron-H / Mamba recurrent state 场景下 stale drafts 会污染状态和 logprobs。

---

## 16. DraftTokenIds 如何回到 Scheduler

### 16.1 DraftTokenIds 结构

`DraftTokenIds` 定义在：`vllm/vllm/v1/outputs.py:311`

字段：

```text
req_ids: list[str]
draft_token_ids: list[list[int]]
```

### 16.2 EngineCore.post_step 回收 draft tokens

非 async scheduling 下：

```python
if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
    draft_token_ids = self.model_executor.take_draft_token_ids()
    if draft_token_ids is not None:
        self.scheduler.update_draft_token_ids(draft_token_ids)
```

位置：`vllm/vllm/v1/engine/core.py:519`

### 16.3 Scheduler 写回 request.spec_token_ids

`Scheduler.update_draft_token_ids()` 会遍历：

```text
DraftTokenIds.req_ids
DraftTokenIds.draft_token_ids
```

然后写入：

```python
request.spec_token_ids = spec_token_ids
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:2005`

这就是下一轮 spec decode 的来源。

### 16.4 prefill chunk 不使用 draft tokens

如果 request 仍是 prefill chunk：

```python
if request.is_prefill_chunk:
    request.spec_token_ids = []
    continue
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:2015`

原因是：

```text
chunked prefill 不是稳定的 decode 推进点，不能直接把 draft tokens 当下一轮输出后缀。
```

---

## 17. Scheduler.update_from_output 如何回收 spec decode 结果

### 17.1 ModelRunnerOutput 中 sampled_token_ids 已经过滤

`sample_tokens()` 构造的 `ModelRunnerOutput` 中：

```python
sampled_token_ids=valid_sampled_token_ids
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4696`

spec decode 下 `valid_sampled_token_ids` 已经经过：

```text
RejectionSampler.parse_output()
```

过滤掉 `-1` placeholder 和无效 token。

### 17.2 update_from_output 取出本 request 的生成 tokens

Scheduler 中：

```python
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1632`

这里的 `generated_token_ids` 对 spec decode 来说是：

```text
accepted draft tokens + recovered token
或
accepted draft tokens + bonus token
```

### 17.3 计算 accepted / rejected 数

如果本轮有 scheduled spec tokens：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1647`

这里 `num_sampled_tokens_per_step` 通常代表非 draft 的真实采样 token 数，常规生成是 1。

所以：

```text
len(generated_token_ids) = accepted_draft_count + 1
num_accepted = len(generated_token_ids) - 1
num_rejected = num_draft_tokens - num_accepted
```

如果全部 draft 接受：

```text
generated_token_ids = draft_1 ... draft_N bonus
num_accepted = N
num_rejected = 0
```

如果第 k 个 draft 被拒绝：

```text
generated_token_ids = draft_1 ... draft_{k-1} recovered
num_accepted = k - 1
num_rejected = N - (k - 1)
```

### 17.4 回滚 num_computed_tokens

如果有 rejected tokens：

```python
request.num_computed_tokens -= num_rejected
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1656`

原因是：

```text
target model forward 已经计算了所有 scheduled draft tokens 的 KV / hidden 状态；
但被拒绝的 draft tokens 不应该成为 request 的真实历史；
因此 Scheduler 需要把 computed token count 回退。
```

这一步是 spec decode 状态正确性的关键。

### 17.5 async placeholders 也要回滚

async scheduling 下还有：

```python
request.num_output_placeholders -= num_rejected
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1660`

原因是：

```text
async scheduling 中 output placeholders 也包含 scheduled spec tokens，
rejected tokens 不能继续placeholder。
```

### 17.6 append 真实输出 tokens

随后 Scheduler 调用：

```python
new_token_ids, stopped = self._update_request_with_output(
    request, new_token_ids
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1683`

只有 `new_token_ids` 中的 tokens 会真正进入 request 输出状态。

这就是为什么：

```text
draft token 只有被 RejectionSampler 输出后，才真正成为 request.output_token_ids。
```

---

## 18. spec decode stats 如何计算

Scheduler 在回收输出时会调用：

```python
self.make_spec_decoding_stats(
    spec_decoding_stats,
    num_draft_tokens=num_draft_tokens,
    num_accepted_tokens=num_accepted,
    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
    request_id=req_id,
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1662`

`SpecDecodingStats` 记录：

```text
num_drafts
num_draft_tokens
num_accepted_tokens
num_accepted_tokens_per_pos
num_draft_tokens_per_pos
```

位置：`vllm/vllm/v1/spec_decode/metrics.py:17`

其中：

```text
num_drafts：有 draft 的请求次数；
num_draft_tokens：总 draft token 数；
num_accepted_tokens：被接受的 draft token 数；
num_accepted_tokens_per_pos：每个 draft 位置的接受次数。
```

日志里常见的 mean acceptance length 计算为：

```text
1 + num_accepted_tokens / num_drafts
```

位置：`vllm/vllm/v1/spec_decode/metrics.py:113`

这里的 `+1` 是 conventionally including bonus token。

---

## 19. structured output / grammar 与 rejected tokens 的关系

### 19.1 grammar 可能让部分 draft token 无效

如果 draft token 序列违反 grammar，`validate_tokens()` 会截断或过滤。

在 `update_draft_token_ids_in_output()` 中，无效 tokens 会被补成：

```text
-1
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:2056`

并记录：

```python
scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens
```

### 19.2 invalid spec tokens 会影响统计

`make_spec_decoding_stats()` 中：

```python
if num_invalid_spec_tokens:
    num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:2393`

这意味着：

```text
被 grammar 提前判定无效的 draft tokens 不应该计入正常 acceptance rate 的分母。
```

### 19.3 target sampling 仍由 grammar bitmask 约束

即便 draft tokens 已经被 validate，target logits 采样仍会应用 grammar bitmask。

所以结构化输出对 spec decode 有双重约束：

```text
1. 提前过滤下一轮 draft tokens；
2. 本轮 target sampling 前 mask logits。
```

---

## 20. 和普通 sampler 的关系

### 20.1 普通 sampler 仍负责 bonus token

RejectionSampler 内部会调用普通 `Sampler` 来采 bonus token。

因此：

```text
temperature / top-k / top-p / allowed token ids / logits processors
```

仍会影响 bonus token。

### 20.2 target logits 也会应用部分 sampling 约束

draft token 的接受 / recovered token 的采样需要处理：

```text
penalties
bad words
allowed token ids
MinTokensLogitsProcessor
thinking budget
temperature
top-k / top-p
```

对应入口：

```text
RejectionSampler.apply_logits_processors()
apply_sampling_constraints()
```

位置：`vllm/vllm/v1/sample/rejection_sampler.py:285`、`vllm/vllm/v1/sample/rejection_sampler.py:510`

### 20.3 为什么 spec decode 不直接复用 Sampler.sample()

普通 Sampler 的输入输出模型是：

```text
每个 request 一个 logits row → 一个 sampled token
```

而 spec decode 需要：

```text
多个 target logits rows；
draft token ids；
draft probabilities；
accept / reject 过程；
recovered distribution；
bonus token；
可变长度输出。
```

所以必须有 RejectionSampler 这一层。

---

## 21. PP / async scheduling 下的特殊点

### 21.1 PP 下 sample 通常在 last rank

Pipeline parallel 下非 last rank 可能只传递 intermediate tensors。

如果是 async scheduling + PP，last rank 采样后需要把 sampled token ids 广播给其他 PP ranks：

```python
self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4529`

非 last rank 在没有 execute_model_state 时会接收：

```python
self._pp_receive_prev_sampled_token_ids_to_input_batch()
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4486`

### 21.2 async scheduling 下 token 可以先留在 GPU

`_bookkeeping_sync()` 中，如果 `use_async_scheduling` 为真，可能不会立刻把 sampled tokens 转成 CPU list，而是：

```text
缓存 sampled_token_ids GPU tensor；
后续通过 AsyncGPUModelRunnerOutput 做 D2H copy；
下一步 prepare_inputs 直接使用 GPU sampled tokens。
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3728`

### 21.3 spec decode 下 async 的 draft token 也要同步状态

如果 async scheduling 且需要 penalties / bad words 等依赖 output token ids，`_sample()` 会先把真实 draft tokens 更新到 `InputBatch`：

```python
draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3639`

这样 logits processors 才能看到正确的 spec token 历史。

---

## 22. 新旧 GPU ModelRunner 路径的关系

当前源码中能看到两套相关实现：

```text
旧 / MRV1 路径：
  vllm/vllm/v1/worker/gpu_model_runner.py
  vllm/vllm/v1/sample/rejection_sampler.py

新 GPU package 路径：
  vllm/vllm/v1/worker/gpu/model_runner.py
  vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
  vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
```

两者核心语义一致：

```text
draft tokens + target logits
  → rejection sampling
  → sampled / num_sampled / num_rejected
  → output / scheduler state update
```

差异主要在：

```text
- 新路径把 input batch / sampler / rejection sampler 拆成更细的 GPU package；
- 新路径的 rejection sampler 返回 num_sampled / num_rejected 等 GPU 侧辅助信息；
- 旧路径文档化更完整，当前 sampling_and_output 文档主要围绕 gpu_model_runner.py 的执行链路展开。
```

---

## 23. 一个完整例子

假设某个 request 当前输出为：

```text
真实历史：A
```

上一轮 drafter 生成：

```text
spec_token_ids = [d1, d2, d3]
```

### 23.1 Scheduler 本轮调度

```text
num_tokens_with_spec = len(A) + 3
num_new_tokens = 4  # d1 d2 d3 + bonus 位置
scheduled_spec_decode_tokens[req] = [d1, d2, d3]
```

### 23.2 ModelRunner 准备 metadata

```text
num_draft_tokens = 3
num_sampled_tokens = 4
logits_indices = [l0, l1, l2, l3]
target_logits_indices = [l0, l1, l2]
bonus_logits_indices = [l3]
```

### 23.3 RejectionSampler 情况 A：全部接受

```text
d1 accepted
d2 accepted
d3 accepted
bonus = b

output = [d1, d2, d3, b]
num_accepted = 4 - 1 = 3
num_rejected = 0
```

Scheduler 不回滚 computed tokens。

### 23.4 RejectionSampler 情况 B：d2 被拒绝

```text
d1 accepted
d2 rejected
recovered = r

output = [d1, r]
num_accepted = 2 - 1 = 1
num_rejected = 3 - 1 = 2
```

Scheduler 会：

```text
request.num_computed_tokens -= 2
append [d1, r]
```

也就是说：

```text
d2 / d3 对应的 speculative 计算不会成为真实历史。
```

---

## 24. 容易混淆的点

### 24.1 draft token 是不是最终一定输出？

不是。

```text
draft token 只有被 rejection sampler 接受后，才会进入 ModelRunnerOutput.sampled_token_ids。
```

### 24.2 target model 是否只算一个 logits？

不是。

spec decode 下每个 request 通常要取：

```text
num_draft_tokens + 1
```

个 logits：draft 验证 logits + bonus logits。

### 24.3 bonus token 是 draft model 产生的吗？

不是。

```text
bonus token 来自 target model 的 bonus logits，并由普通 Sampler 采样。
```

### 24.4 recovered token 和 bonus token 是一回事吗？

不是。

```text
recovered token：某个 draft 被拒绝后，从修正分布采样，用来替代被拒绝位置。
bonus token：所有 draft 都接受时，从 target logits 多采一个 token。
```

### 24.5 rejected token 的 KV 状态会保留吗？

逻辑上不会。

target forward 可能已经计算过这些 draft token，但 Scheduler 会通过：

```text
num_computed_tokens -= num_rejected
```

让后续调度从真实接受的位置继续。

### 24.6 sampled_token_ids 为什么是 list[list[int]]？

因为 spec decode 下每个 request 一轮可能输出不同数量的 tokens。

```text
普通 decode：通常每行 1 个 token。
spec decode：每行 1 到 num_draft+1 个 token。
```

### 24.7 grammar 是否只影响 bonus token？

不是。

```text
grammar 会过滤 draft tokens，也会在 target sampling 前 mask logits。
```

### 24.8 ngram spec decode 没有 draft_probs 是否还能走 rejection sampler？

可以。

`draft_probs` 可以为 None，代码会走专门分支适配 ngram proposer。

### 24.9 spec decode 是否一定比普通 decode 多输出 token？

不一定。

如果第一个 draft 就被拒绝，本轮只输出一个 recovered token，和普通 decode 输出数量相同，但多做了 draft 验证计算。

### 24.10 accepted token 数是不是等于输出 token 数？

不是。

常规情况下：

```text
输出 token 数 = accepted draft 数 + 1
```

最后的 `+1` 是 recovered token 或 bonus token。

---

## 25. 最终可以记成一张表

| 阶段 | 关键函数 / 类 | 核心产物 | 作用 |
|---|---|---|---|
| 保存 draft | `Request.spec_token_ids` | draft token ids | 暂存下一轮要验证的 tokens |
| 调度 draft | `Scheduler.schedule()` | `scheduled_spec_decode_tokens` | 把 request 上的 draft tokens 放进本轮 SchedulerOutput |
| 准备 metadata | `_calc_spec_decode_metadata()` | `SpecDecodeMetadata` | 计算 target / bonus / logits indices |
| target forward | `GPUModelRunner.execute_model()` | target logits | 为 draft 验证和 bonus 采样提供 logits |
| 采样分流 | `_sample()` | `SamplerOutput` | 普通采样走 Sampler，spec decode 走 RejectionSampler |
| bonus 采样 | `RejectionSampler.forward()` | `bonus_token_ids` | 全部 draft 接受时追加 target token |
| 接受 / 拒绝 | `rejection_sample()` | `[batch, max_spec_len + 1]` token matrix | 输出 accepted / recovered / bonus tokens，拒绝后填 -1 |
| 输出过滤 | `RejectionSampler.parse_output()` | `list[list[int]]`、`LogprobsLists` | 过滤 placeholder，转成 ModelRunnerOutput 可用结构 |
| 状态回收 | `Scheduler.update_from_output()` | accepted / rejected count | 回滚 rejected tokens，append 真实输出 |
| 新 draft 回写 | `take_draft_token_ids()` + `update_draft_token_ids()` | 下一轮 `request.spec_token_ids` | 让下一轮继续 spec decode |
| 指标 | `SpecDecodingStats` | acceptance stats | 统计 draft / accepted / per-position acceptance |

---

## 26. 总结

Spec decode 影响采样和输出的完整链路是：

```text
上一轮真实输出
  → drafter 生成 draft tokens
  → Scheduler 保存到 request.spec_token_ids
  → 下一轮 schedule 把 draft tokens 放进 scheduled_spec_decode_tokens
  → ModelRunner 构造 SpecDecodeMetadata
  → target model 一次 forward 产生 draft 验证 logits + bonus logits
  → RejectionSampler 接受 / 拒绝 draft
  → 产出 accepted + recovered 或 accepted + bonus
  → parse_output 过滤placeholder并整理 logprobs
  → ModelRunnerOutput 返回可变长度 sampled_token_ids
  → Scheduler 回滚 rejected tokens 的 computed state
  → append 真实输出并统计 acceptance
```

如果只记住三句话：

```text
1. draft token 只是候选，只有被 target model 的 rejection sampler 接受后才是输出。
2. spec decode 下每个 request 会取 num_draft+1 个 logits：前面验证 draft，最后一个采 bonus。
3. Scheduler 必须根据 rejected token 数回滚 num_computed_tokens，否则被拒绝的 draft KV / 状态会污染后续解码。
```

最终一句话：

```text
Spec decode 把采样从“直接从 target logits 选 token”变成“用 target logits 批量验证 draft，并把 accepted / recovered / bonus 结果回收到 Scheduler 状态机”的过程。
```
