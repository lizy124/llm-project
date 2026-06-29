# 06. RejectionSampler 如何接受 / 拒绝 draft tokens？

源码位置：

- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/sample/sampler.py`
- `code/vllm/vllm/v1/sample/metadata.py`
- `code/vllm/vllm/v1/spec_decode/metadata.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`

本问题关注：target model 已经计算出 spec decode 所需 logits 后，`GPUModelRunner._sample()` 如何进入 `RejectionSampler`；`RejectionSampler` 如何拆分 target logits / bonus logits，如何用 target 分布验证 draft tokens，如何在拒绝时采样 recovered token，如何在全接受时追加 bonus token，以及最终如何把 `[batch_size, max_spec_len + 1]` 的 padded 输出变成 Scheduler 能回收的 token 列表。

---

## 1. 一句话回答

`RejectionSampler` 是 spec decode 的判决器。

它做的事情是：

```text
给定：
  draft token ids
  draft probabilities（可选）
  target model logits
  bonus logits
  sampling params

输出：
  每个请求最终真正生成的 token 序列：
    accepted draft tokens
    + 拒绝位置的 recovered token
    + 全部接受时的 bonus token
```

核心规则：

```text
从左到右验证 draft tokens：
  - 接受：继续看下一个 draft token；
  - 拒绝：停止后续 draft，采样一个 recovered token；
  - 全部接受：追加一个 bonus token。
```

---

## 2. RejectionSampler 在整条链路中的位置

完整链路是：

```text
GPUModelRunner.execute_model()
  → _prepare_inputs()
      → SpecDecodeMetadata
      → logits_indices
  → hidden_states[logits_indices]
  → model.compute_logits(sample_hidden_states)
  → ExecuteModelState(logits, spec_decode_metadata, ...)

GPUModelRunner.sample_tokens()
  → _sample(logits, spec_decode_metadata)
      → RejectionSampler.forward(...)
      → SamplerOutput(sampled_token_ids, logprobs_tensors)
  → _bookkeeping_sync()
      → RejectionSampler.parse_output()
      → valid_sampled_token_ids
  → ModelRunnerOutput

Scheduler.update_from_output()
  → 统计 accepted / rejected
  → 修正 num_computed_tokens
  → append output tokens
```

入口在：`gpu_model_runner.py:3570`

```python
def _sample(
    self,
    logits: torch.Tensor | None,
    spec_decode_metadata: SpecDecodeMetadata | None,
) -> SamplerOutput:
```

---

## 3. 普通 Sampler 和 RejectionSampler 的分流

`_sample()` 先取 `SamplingMetadata`：

```python
sampling_metadata = self.input_batch.sampling_metadata
self.input_batch.update_async_output_token_ids()
```

位置：`gpu_model_runner.py:3575` 到 `gpu_model_runner.py:3579`

如果没有 spec decode metadata，直接走普通 sampler：

```python
if spec_decode_metadata is None:
    return self.sampler(
        logits=logits,
        sampling_metadata=sampling_metadata,
    )
```

位置：`gpu_model_runner.py:3580` 到 `gpu_model_runner.py:3584`

如果有 spec decode metadata：

```python
draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
sampler_output = self.rejection_sampler(
    spec_decode_metadata,
    draft_probs,
    logits,
    sampling_metadata,
)
```

位置：`gpu_model_runner.py:3592` 到 `gpu_model_runner.py:3598`

所以分流可以记成：

```text
spec_decode_metadata is None
  → Sampler

spec_decode_metadata exists
  → RejectionSampler
```

---

## 4. RejectionSampler 初始化在哪里

`GPUModelRunner` 初始化 speculative decoding 时会创建：

```python
self.rejection_sampler = RejectionSampler(
    self.sampler, self.speculative_config, self.device
)
```

位置：`gpu_model_runner.py:618` 到 `gpu_model_runner.py:620`

`RejectionSampler` 持有普通 `Sampler`：

```python
self.sampler = sampler
```

位置：`rejection_sampler.py:60` 到 `rejection_sampler.py:67`

它复用普通 sampler 来做：

```text
bonus token sampling
logprobs 计算
top-k / top-p / temperature 等采样约束
```

---

## 5. RejectionSampler 的术语定义

源码 docstring 对术语说明很明确：`rejection_sampler.py:37`

```text
accepted tokens：
  基于 draft 和 target 概率关系被接受的 draft tokens。

recovered tokens：
  draft 被拒绝后，从修正分布采样出来的替代 token。

bonus tokens：
  如果所有 draft tokens 都接受，在末尾额外追加的 target token。

output tokens：
  最终输出 token = accepted tokens + recovered tokens + bonus tokens。
```

位置：`rejection_sampler.py:37` 到 `rejection_sampler.py:58`

---

## 6. RejectionSampler.forward() 的输入

入口：`rejection_sampler.py:88`

```python
def forward(
    self,
    metadata: SpecDecodeMetadata,
    draft_probs: torch.Tensor | None,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> SamplerOutput:
```

位置：`rejection_sampler.py:88` 到 `rejection_sampler.py:96`

输入含义：

| 参数 | 含义 |
|---|---|
| `metadata` | `SpecDecodeMetadata`，描述 draft / target / bonus logits 行布局 |
| `draft_probs` | draft model 对 draft tokens 的概率分布，可为 `None` |
| `logits` | target model 的 logits，形状是 `[num_tokens + batch_size, vocab_size]` |
| `sampling_metadata` | temperature / top-k / top-p / penalties / bad words / logprobs 等采样配置 |

其中：

```text
num_tokens = total draft token 数
batch_size = 当前 batch 请求数
```

---

## 7. RejectionSampler.forward() 的主流程

源码主流程：

```text
1. 检查 max_spec_len；
2. 从 logits 中取 bonus_logits；
3. 用普通 Sampler 先采样 bonus_token_ids；
4. 从 logits 中取 target_logits；
5. 对 target_logits 应用 logits processors / penalties / allowed tokens / bad words；
6. 对 target_logits 应用 temperature / top-k / top-p；
7. 调用 rejection_sample() 进行接受 / 拒绝判定；
8. 如需 logprobs，调用 _get_logprobs_tensors()；
9. 返回 SamplerOutput。
```

对应源码位置：`rejection_sampler.py:119` 到 `rejection_sampler.py:197`

---

## 8. 第一步：检查 max_spec_len

```python
assert metadata.max_spec_len <= MAX_SPEC_LEN
```

位置：`rejection_sampler.py:119`

常量：

```python
MAX_SPEC_LEN = 128
```

位置：`rejection_sampler.py:32` 到 `rejection_sampler.py:34`

含义：

```text
单请求单步最多支持 128 个 speculative draft tokens。
```

这是 kernel 侧为了固定循环 / 避免过度动态化设置的上限。

---

## 9. 第二步：取 bonus logits 并采样 bonus token

`metadata.bonus_logits_indices` 指向 logits 中每个请求的 bonus row：

```python
bonus_logits_indices = metadata.bonus_logits_indices
bonus_logits = logits[bonus_logits_indices]
```

位置：`rejection_sampler.py:121` 到 `rejection_sampler.py:129`

然后调用普通 sampler：

```python
bonus_sampler_output = self.sampler(
    logits=bonus_logits,
    sampling_metadata=replace(
        sampling_metadata,
        max_num_logprobs=-1,
    ),
    predict_bonus_token=True,
    logprobs_mode_override="processed_logits"
    if self.is_processed_logprobs_mode
    else "raw_logits",
)
bonus_token_ids = bonus_sampler_output.sampled_token_ids
```

位置：`rejection_sampler.py:130` 到 `rejection_sampler.py:143`

为什么先采样 bonus？

```text
如果某个请求所有 draft tokens 都被接受，
RejectionSampler 直接把这个 bonus token 放到输出末尾。
```

为什么 bonus 用普通 Sampler？

```text
bonus token 只来自 target model，
不需要和 draft_probs 做接受率判断；
而且普通 sampler 已经实现了 top-p / top-k / temperature / logits processors 等逻辑。
```

---

## 10. 第三步：取 target logits

`metadata.target_logits_indices` 指向 logits 中用于验证 draft tokens 的 rows：

```python
target_logits_indices = metadata.target_logits_indices
raw_target_logits = logits[target_logits_indices]
raw_target_logits = raw_target_logits.to(torch.float32)
```

位置：`rejection_sampler.py:121` 到 `rejection_sampler.py:150`

这里的对应关系是：

```text
raw_target_logits[j]
  用来验证
metadata.draft_token_ids[j]
```

也就是说：

```text
target logits 行数 == draft token 总数
```

`rejection_sample()` 后面会 assert：

```python
num_tokens = draft_token_ids.shape[0]
vocab_size = target_logits.shape[-1]
assert target_logits.shape == (num_tokens, vocab_size)
```

位置：`rejection_sampler.py:418` 到 `rejection_sampler.py:425`

---

## 11. 为什么要 clone target logits

源码：

```python
target_logits = raw_target_logits
if not self.is_processed_logprobs_mode:
    target_logits = target_logits.clone()
```

位置：`rejection_sampler.py:151` 到 `rejection_sampler.py:156`

原因：

```text
apply_logits_processors() 会原地修改 logits；
如果 logprobs 模式需要 raw logits / raw logprobs，
就不能把 raw_target_logits 覆盖掉。
```

所以在非 processed logprobs 模式下会 clone 一份。

---

## 12. 第四步：应用 logits processors

```python
target_logits = self.apply_logits_processors(
    target_logits, sampling_metadata, metadata
)
```

位置：`rejection_sampler.py:157` 到 `rejection_sampler.py:159`

这一步处理：

```text
allowed_token_ids
bad_words
presence / frequency / repetition penalties
MinTokensLogitsProcessor
thinking budget
```

注意：这里只对 target verification logits 做处理。

bonus token 的 logits processors 已经由普通 `Sampler` 在采样 bonus 时处理。

---

## 13. penalties 为什么要展开到 draft-token 级别

在普通 decode 中，采样参数是 request 级的。

但 spec decode 中，target logits 是 draft-token 级 flatten 后的：

```text
req0 draft0
req0 draft1
req0 draft2
req2 draft0
req2 draft1
...
```

所以 `apply_logits_processors()` 会构造 `repeat_indices`：

```python
num_requests = len(metadata.num_draft_tokens)
num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
original_indices = torch.arange(num_requests, device="cpu")
repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
repeat_indices = repeat_indices_cpu.to(device=logits.device, non_blocking=True)
```

位置：`rejection_sampler.py:305` 到 `rejection_sampler.py:319`

例子：

```text
num_draft_tokens = [3, 0, 2]
repeat_indices = [0, 0, 0, 2, 2]
```

这样 penalties / allowed_token_ids 才能按正确请求扩展到每个 draft verification row。

---

## 14. 为什么 bad_words / penalties 要合并 spec tokens

如果需要 penalties、bad words 或 thinking budget：

```python
output_token_ids = self._combine_outputs_with_spec_tokens(
    output_token_ids,
    sampling_metadata.spec_token_ids,
)
```

位置：`rejection_sampler.py:298` 到 `rejection_sampler.py:303`

`_combine_outputs_with_spec_tokens()`：

```python
for out, spec in zip(output_token_ids, spec_token_ids):
    if len(spec) == 0:
        continue
    result.append(out)
    for i in range(len(spec) - 1):
        result.append([*result[-1], spec[i]])
```

位置：`rejection_sampler.py:376` 到 `rejection_sampler.py:391`

含义：

```text
验证第 1 个 draft：上下文是原 output；
验证第 2 个 draft：上下文应包含第 1 个 draft；
验证第 3 个 draft：上下文应包含前 2 个 draft。
```

因此，penalties / bad words 的上下文不是固定的，而是随 draft 位置逐步增加。

---

## 15. 第五步：应用 sampling constraints

```python
target_logits = apply_sampling_constraints(
    target_logits,
    metadata.cu_num_draft_tokens,
    sampling_metadata,
)
```

位置：`rejection_sampler.py:163` 到 `rejection_sampler.py:167`

`apply_sampling_constraints()` 做：

```text
temperature
top_k
top_p
```

入口：`rejection_sampler.py:510`

如果所有请求都是 greedy：

```python
if sampling_metadata.all_greedy:
    return logits
```

位置：`rejection_sampler.py:531` 到 `rejection_sampler.py:534`

非 greedy 时，会把 request 级参数扩展到 draft-token 级：

```python
temperature = expand_batch_to_tokens(
    sampling_metadata.temperature,
    cu_num_draft_tokens,
    num_tokens,
    replace_from=GREEDY_TEMPERATURE,
    replace_to=1,
)
```

位置：`rejection_sampler.py:536` 到 `rejection_sampler.py:545`

`top_k` / `top_p` 也类似展开：`rejection_sampler.py:547` 到 `rejection_sampler.py:565`

---

## 16. expand_batch_to_tokens() 是什么

入口：`rejection_sampler.py:568`

它把 batch 级 tensor 展开成 token 级 tensor。

注释例子：

```text
x = [a, b, c]
cu_num_tokens = [2, 5, 6]
num_tokens = 6
expanded_x = [a, a, b, b, b, c]
```

位置：`rejection_sampler.py:575` 到 `rejection_sampler.py:580`

在 spec decode 里，这用于：

```text
每个 draft verification row 都拿到所属 request 的 temperature / top_k / top_p。
```

---

## 17. 第六步：调用 rejection_sample()

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
    synthetic_mode=self.synthetic_mode,
    synthetic_conditional_rates=self.synthetic_conditional_rates,
    use_fp64_gumbel=self.use_fp64_gumbel,
)
```

位置：`rejection_sampler.py:169` 到 `rejection_sampler.py:181`

`rejection_sample()` 是实际接受 / 拒绝判定入口。

位置：`rejection_sampler.py:394`

---

## 18. rejection_sample() 的输入形状

```python
draft_token_ids: torch.Tensor        # [num_tokens]
num_draft_tokens: list[int]          # [batch_size]
cu_num_draft_tokens: torch.Tensor    # [batch_size]
draft_probs: torch.Tensor | None     # [num_tokens, vocab_size]
target_logits: torch.Tensor          # [num_tokens, vocab_size]
bonus_token_ids: torch.Tensor        # [batch_size, 1]
```

位置：`rejection_sampler.py:394` 到 `rejection_sampler.py:412`

其中：

```text
num_tokens = sum(num_draft_tokens)
```

它会检查：

```python
assert draft_token_ids.ndim == 1
assert draft_probs is None or draft_probs.ndim == 2
assert cu_num_draft_tokens.ndim == 1
assert target_logits.ndim == 2
assert target_logits.shape == (num_tokens, vocab_size)
```

位置：`rejection_sampler.py:413` 到 `rejection_sampler.py:425`

---

## 19. 输出 buffer 的形状

`rejection_sample()` 先创建输出矩阵：

```python
output_token_ids = torch.full(
    (batch_size, max_spec_len + 1),
    PLACEHOLDER_TOKEN_ID,
    dtype=torch.int32,
    device=device,
)
```

位置：`rejection_sampler.py:427` 到 `rejection_sampler.py:433`

`PLACEHOLDER_TOKEN_ID` 是：

```python
PLACEHOLDER_TOKEN_ID = -1
```

位置：`rejection_sampler.py:30`

含义：

```text
每个请求最多输出 max_spec_len + 1 个 token。

真实 token id：
  accepted / recovered / bonus。

-1：
  padding 或 rejected 后不再使用的位置。
```

---

## 20. greedy 和 random 请求如何分开处理

`rejection_sample()` 会根据 sampling metadata 判断请求类型：

```python
if sampling_metadata.all_greedy:
    is_greedy = None
else:
    is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
```

位置：`rejection_sampler.py:435` 到 `rejection_sampler.py:438`

然后：

```text
非 all_random：
  先跑 greedy kernel。

如果 all_greedy：
  greedy kernel 结束后直接返回。

否则：
  继续计算 target_probs / recovered_token_ids，
  再跑 random kernel 处理非 greedy 请求。
```

对应源码：`rejection_sampler.py:453` 到 `rejection_sampler.py:507`

这样 mixed batch 可以同时包含 greedy 和 random 请求。

---

## 21. greedy acceptance 规则

`rejection_greedy_sample_kernel` 入口：`rejection_sampler.py:715`

对 greedy 请求，每个 draft token 的判定是：

```python
target_argmax = target_logits.argmax(dim=-1)
```

位置：`rejection_sampler.py:453` 到 `rejection_sampler.py:456`

kernel 内：

```python
draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
token_id = target_argmax_id
rejected = draft_token_id != target_argmax_id
```

位置：`rejection_sampler.py:740` 到 `rejection_sampler.py:756`

含义：

```text
如果 draft token == target argmax：接受。
如果 draft token != target argmax：拒绝，并输出 target argmax 作为 recovered / replacement token。
```

greedy 场景不需要概率比，也不需要从 residual distribution 随机采样。

---

## 22. greedy 全接受时如何追加 bonus

kernel 末尾：

```python
if not rejected:
    bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
    tl.store(
        output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
        bonus_token_id,
    )
```

位置：`rejection_sampler.py:758` 到 `rejection_sampler.py:764`

含义：

```text
只有所有 draft tokens 都 accepted，才追加 bonus token。
```

如果中间某个 draft 被拒绝，后续 draft 不再处理，bonus 也不会追加。

---

## 23. random acceptance 需要 uniform_probs

非 greedy 请求需要随机数：

```python
uniform_probs = generate_uniform_probs(
    num_tokens,
    num_draft_tokens,
    sampling_metadata.generators,
    device,
)
```

位置：`rejection_sampler.py:440` 到 `rejection_sampler.py:451`

`generate_uniform_probs()` 定义在：`rejection_sampler.py:608`

它会为每个 draft token 生成 `[0, 1)` 的随机数。

如果请求有自己的 `torch.Generator`，会用它保证可复现：

```python
for i, generator in sampling_metadata.generators.items():
    if num_draft_tokens[i] > 0:
        uniform_probs[start_idx:end_idx].uniform_(generator=generator)
```

位置：`rejection_sampler.py:649` 到 `rejection_sampler.py:659`

---

## 24. random acceptance 规则

`rejection_random_sample_kernel` 入口：`rejection_sampler.py:769`

对每个 draft token：

```python
draft_prob = q(x)
target_prob = p(x)
uniform_prob = u
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```

源码：

```python
if NO_DRAFT_PROBS:
    draft_prob = 1
else:
    draft_prob = tl.load(... draft_probs ...)
target_prob = tl.load(... target_probs ...)
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```

位置：`rejection_sampler.py:796` 到 `rejection_sampler.py:817`

数学上就是：

```text
accept with probability min(1, p(x) / q(x))
```

其中：

```text
p(x)：target model 对 draft token x 的概率
q(x)：draft model 对 draft token x 的概率
```

---

## 25. draft_probs 为 None 时是什么意思

`draft_probs` 可以为 `None`。

注释说明：

```text
Can be None if probabilities are not provided,
which is the case for ngram spec decode.
```

位置：`rejection_sampler.py:101` 到 `rejection_sampler.py:104`

如果 `NO_DRAFT_PROBS=True`：

```python
draft_prob = 1
```

位置：`rejection_sampler.py:803` 到 `rejection_sampler.py:806`

这等价于 one-hot draft proposal：

```text
draft proposer 只给 token，不给完整概率分布；
接受率按 target_prob / 1 判断；
拒绝后的 recovered distribution 会把 rejected draft token 排除。
```

这通常用于 ngram / prompt lookup 类 proposer。

---

## 26. random 拒绝后输出 recovered token

random kernel 中：

```python
if accepted:
    token_id = draft_token_id
else:
    rejected = True
    token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
```

位置：`rejection_sampler.py:818` 到 `rejection_sampler.py:825`

含义：

```text
第一个 rejected 位置输出 recovered token。
后续 draft token 位置保持 -1。
bonus token 不追加。
```

---

## 27. recovered token 从哪里来

在 random kernel 前，会先计算 `target_probs`：

```python
target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
```

位置：`rejection_sampler.py:471` 到 `rejection_sampler.py:473`

然后调用：

```python
recovered_token_ids = sample_recovered_tokens(
    max_spec_len,
    num_draft_tokens,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    sampling_metadata,
    device,
    use_fp64_gumbel,
)
```

位置：`rejection_sampler.py:475` 到 `rejection_sampler.py:487`

`sample_recovered_tokens()` 定义在：`rejection_sampler.py:663`

它为每个 draft 位置预先采样一个 recovered token。

---

## 28. recovered distribution 是什么

`sample_recovered_tokens_kernel` 定义在：`rejection_sampler.py:860`

如果没有 draft_probs：

```python
prob = target_probs[v]
mask=(vocab_mask & (vocab_offset != draft_token_id))
```

位置：`rejection_sampler.py:885` 到 `rejection_sampler.py:902`

含义：

```text
从 target distribution 中采样，但排除 rejected draft token。
```

如果有 draft_probs：

```python
prob = tl.maximum(target_prob - draft_prob, 0.0)
```

位置：`rejection_sampler.py:903` 到 `rejection_sampler.py:916`

也就是从 residual distribution 采样：

```text
max(p(x) - q(x), 0)
```

这是 speculative decoding 论文中的修正分布，用来保证最终分布仍等价于 target model。

---

## 29. recovered sampling 如何实现

kernel 里会构造 Gumbel-max 风格采样：

```python
q = torch.empty((batch_size, vocab_size), ...)
q.exponential_()
inv_q = q.reciprocal()
```

位置：`rejection_sampler.py:678` 到 `rejection_sampler.py:696`

然后在 kernel 中：

```python
score = prob * inv_q
local_max, local_id = tl.max(score, axis=0, return_indices=True)
```

位置：`rejection_sampler.py:918` 到 `rejection_sampler.py:935`

最终写入：

```python
tl.store(output_token_ids_ptr + token_idx, recovered_id)
```

位置：`rejection_sampler.py:936` 到 `rejection_sampler.py:937`

直观理解：

```text
对 residual distribution 做一次随机采样，
为每个可能 rejected 的位置提前准备 recovered token。
```

真正是否使用 recovered token，由 `rejection_random_sample_kernel` 的 accepted 判断决定。

---

## 30. random 全接受时如何追加 bonus

和 greedy kernel 一样，random kernel 末尾也会：

```python
if not rejected:
    bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
    tl.store(
        output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
        bonus_token_id,
    )
```

位置：`rejection_sampler.py:827` 到 `rejection_sampler.py:833`

所以：

```text
全部 draft accepted
  → 输出 draft tokens + bonus token。

任一 draft rejected
  → 输出 accepted prefix + recovered token。
```

---

## 31. synthetic mode 是什么

`RejectionSampler.__init__()` 中：

```python
if spec_config.rejection_sample_method == "synthetic":
    self.synthetic_conditional_rates = torch.tensor(
        unconditional_to_conditional_rates(
            spec_config.synthetic_acceptance_rates
        ),
        dtype=torch.float32,
        device=device,
    )
self.synthetic_mode = self.synthetic_conditional_rates is not None
```

位置：`rejection_sampler.py:73` 到 `rejection_sampler.py:86`

在 synthetic mode 下，接受判断不使用真实 `p/q`，而使用配置的条件接受率：

```python
rate = tl.load(synthetic_conditional_rates_ptr + pos)
accepted = uniform_prob < rate
```

greedy kernel 位置：`rejection_sampler.py:744` 到 `rejection_sampler.py:749`

random kernel 位置：`rejection_sampler.py:800` 到 `rejection_sampler.py:803`

用途可以理解为：

```text
模拟指定 acceptance rate，用于实验 / profile / synthetic benchmarking。
```

---

## 32. RejectionSampler 输出 SamplerOutput

`forward()` 最终返回：

```python
return SamplerOutput(
    sampled_token_ids=output_token_ids,
    logprobs_tensors=logprobs_tensors,
)
```

位置：`rejection_sampler.py:194` 到 `rejection_sampler.py:197`

其中：

```text
sampled_token_ids shape:
  [batch_size, max_spec_len + 1]

logprobs_tensors:
  如果请求需要 logprobs，则包含对应 logprobs；否则为 None。
```

`SamplerOutput` 定义：`outputs.py:185`

```python
@dataclass
class SamplerOutput:
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
```

位置：`outputs.py:185` 到 `outputs.py:192`

---

## 33. logprobs 如何处理

如果需要 logprobs：

```python
if sampling_metadata.max_num_logprobs is not None:
    logprobs_tensors = self._get_logprobs_tensors(...)
```

位置：`rejection_sampler.py:183` 到 `rejection_sampler.py:192`

`_get_logprobs_tensors()` 会先构造 `final_logits`：

```python
final_logits = torch.zeros_like(logits, dtype=torch.float32)
final_logits[target_logits_indices] = target_logits.to(torch.float32)
final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)
```

位置：`rejection_sampler.py:211` 到 `rejection_sampler.py:216`

然后根据 sampled token 的位置取对应 logits：

```python
accepted_logit_indices = (
    logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
).flatten()
accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
accepted_tokens = sampled_token_ids.clone().flatten()
accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0
```

位置：`rejection_sampler.py:221` 到 `rejection_sampler.py:233`

注意：

```text
rejected / padding 位置是 -1，
为了避免 gather_logprobs 出错，会先替换成 0；
真正输出时会在 parse_output 中过滤掉。
```

最后：

```python
return self.sampler.gather_logprobs(
    accepted_logprobs,
    max_num_logprobs,
    accepted_tokens.to(torch.int64),
)
```

位置：`rejection_sampler.py:235` 到 `rejection_sampler.py:246`

---

## 34. parse_output() 如何过滤 -1

`RejectionSampler.parse_output()` 定义在：`rejection_sampler.py:248`

输入：

```text
output_token_ids: [batch_size, max_spec_len + 1]
vocab_size
discard_req_indices
logprobs_tensors
```

它先把 GPU tensor 转成 numpy：

```python
output_token_ids_np = output_token_ids.cpu().numpy()
```

位置：`rejection_sampler.py:267`

然后构造有效 token mask：

```python
valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
    output_token_ids_np < vocab_size
)
```

位置：`rejection_sampler.py:268` 到 `rejection_sampler.py:271`

如果有 logprobs，也同步过滤：

```python
cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
output_logprobs = filtered_tensors.tolists(cu_num_tokens)
```

位置：`rejection_sampler.py:272` 到 `rejection_sampler.py:276`

最后返回每个请求的有效 token list：

```python
outputs = [
    row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
]
```

位置：`rejection_sampler.py:278` 到 `rejection_sampler.py:283`

---

## 35. _bookkeeping_sync() 如何调用 parse_output()

在 `GPUModelRunner._bookkeeping_sync()` 中：

```python
max_gen_len = sampled_token_ids.shape[-1]
if max_gen_len == 1:
    valid_sampled_token_ids = self._to_list(sampled_token_ids)
else:
    valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
        sampled_token_ids,
        self.input_batch.vocab_size,
        discard_sampled_tokens_req_indices,
        logprobs_tensors=logprobs_tensors,
    )
```

位置：`gpu_model_runner.py:3656` 到 `gpu_model_runner.py:3674`

判断逻辑：

```text
max_gen_len == 1：
  普通 decode 输出，每个请求 1 token。

max_gen_len > 1：
  spec decode 输出，需要过滤 -1 padding / rejected 后空位。
```

---

## 36. _bookkeeping_sync() 如何更新 Worker 侧 token 状态

解析出 `valid_sampled_token_ids` 后，Worker 会把它们写回 `InputBatch` 和 `CachedRequestState`：

```python
start_idx = self.input_batch.num_tokens_no_spec[req_idx]
end_idx = start_idx + num_sampled_ids
self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
self.input_batch.num_tokens_no_spec[req_idx] = end_idx

req_id = req_ids[req_idx]
req_state = self.requests[req_id]
req_state.output_token_ids.extend(sampled_ids)
```

位置：`gpu_model_runner.py:3693` 到 `gpu_model_runner.py:3725`

注意：

```text
这是 Worker / ModelRunner 侧的 cached state，
Scheduler 侧真正的 Request.output_token_ids 仍在 Scheduler.update_from_output() 中更新。
```

---

## 37. Scheduler 如何统计 accepted / rejected

`Scheduler.update_from_output()` 中会拿到：

```python
generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
```

位置：`scheduler.py:1542` 到 `scheduler.py:1549`

如果本轮有 scheduled draft：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

位置：`scheduler.py:1550` 到 `scheduler.py:1556`

通常 `num_sampled_tokens_per_step = 1`，所以：

```text
len(generated_token_ids) = accepted_draft_count + 1
```

这里的 `+1` 可能是：

```text
recovered token
或 bonus token
或普通 sampled token
```

因此：

```text
accepted draft 数 = max(输出 token 数 - 1, 0)
rejected draft 数 = scheduled draft 数 - accepted draft 数
```

---

## 38. Scheduler 如何修正 num_computed_tokens

如果有 rejected tokens：

```python
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

位置：`scheduler.py:1557` 到 `scheduler.py:1567`

原因：

```text
Scheduler 在 schedule 后已经乐观推进了 num_computed_tokens，
认为 scheduled draft tokens 都被计算并可能接受。

RejectionSampler 输出后，
rejected draft tokens 不能成为真实上下文，
所以要把 rejected 数量回退。
```

---

## 39. Scheduler 如何把输出 token 写入 request

在 `Scheduler.update_from_output()` 后续：

```python
new_token_ids = generated_token_ids
if new_token_ids:
    new_token_ids, stopped = self._update_request_with_output(
        request, new_token_ids
    )
```

位置：`scheduler.py:1580` 到 `scheduler.py:1592`

`_update_request_with_output()` 会逐个 append：

```python
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)
    stopped = check_stop(request, self.max_model_len)
    if stopped:
        del new_token_ids[num_new:]
        break
```

位置：`scheduler.py:1848` 到 `scheduler.py:1864`

也就是说：

```text
RejectionSampler 的有效输出 token
  → Scheduler.update_from_output()
  → Request.output_token_ids
```

只有这一步之后，accepted draft / recovered / bonus token 才变成 Scheduler 侧正式输出。

---

## 40. 示例：全部接受

假设：

```text
draft tokens = [A, B, C]
target 对 A/B/C 都接受
bonus token = D
```

RejectionSampler 输出矩阵一行：

```text
[A, B, C, D]
```

`parse_output()` 后：

```text
generated_token_ids = [A, B, C, D]
```

Scheduler 计算：

```text
num_draft_tokens = 3
num_sampled = 1
num_accepted = len([A,B,C,D]) - 1 = 3
num_rejected = 0
```

结果：

```text
A/B/C 作为 accepted draft 进入 output；
D 作为 bonus token 进入 output；
num_computed_tokens 不回退。
```

---

## 41. 示例：部分接受后拒绝

假设：

```text
draft tokens = [A, B, C]
A/B accepted
C rejected
recovered token = X
```

RejectionSampler 输出矩阵一行：

```text
[A, B, X, -1]
```

`parse_output()` 后：

```text
generated_token_ids = [A, B, X]
```

Scheduler 计算：

```text
num_draft_tokens = 3
num_sampled = 1
num_accepted = len([A,B,X]) - 1 = 2
num_rejected = 1
```

结果：

```text
A/B 作为 accepted draft 进入 output；
X 作为 recovered token 进入 output；
C 被拒绝，不进入 output；
num_computed_tokens 回退 1。
```

---

## 42. 示例：第一个 token 就拒绝

假设：

```text
draft tokens = [A, B, C]
A rejected
recovered token = X
```

RejectionSampler 输出矩阵一行：

```text
[X, -1, -1, -1]
```

`parse_output()` 后：

```text
generated_token_ids = [X]
```

Scheduler 计算：

```text
num_accepted = len([X]) - 1 = 0
num_rejected = 3
```

结果：

```text
没有 draft token 被接受；
只输出 recovered token X；
num_computed_tokens 回退 3。
```

---

## 43. 无 draft 请求在 mixed batch 中如何处理

如果 batch 中有 spec decode，但某个请求 `num_draft_tokens=0`：

```text
它没有 target verification rows；
但有一个 bonus / normal sampling row。
```

RejectionSampler 对它的输出就是一个普通 sampled token：

```text
[bonus_or_normal_token]
```

这让 mixed batch 可以统一使用：

```text
[batch_size, max_spec_len + 1]
```

的输出格式。

---

## 44. async scheduling 下的区别

在 async scheduling 下，`_bookkeeping_sync()` 不会立即把 sampled tokens 转成 CPU list：

```python
else:
    valid_sampled_token_ids = []
    invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
    ...
    if self.input_batch.prev_sampled_token_ids is None:
        assert sampled_token_ids.shape[-1] == 1
        self.input_batch.prev_sampled_token_ids = sampled_token_ids
```

位置：`gpu_model_runner.py:3675` 到 `gpu_model_runner.py:3691`

但 spec decode 的 padded-batch drafter 会在后续 proposal 中处理 GPU 上的 sampled tokens：

```text
sampled_token_ids shape:
  [num_reqs, num_spec_tokens + 1]

rejected tokens:
  value = -1
```

相关注释位置：`gpu_model_runner.py:5020` 到 `gpu_model_runner.py:5027`

这就是 async spec decode 中很多 `prev_num_draft_len`、`valid_sampled_token_count`、`prev_sampled_token_ids` 逻辑存在的原因。

---

## 45. v1/sample 与 worker/gpu/spec_decode 两套实现的区别

本文件主链路使用的是：

```text
code/vllm/vllm/v1/sample/rejection_sampler.py
```

也就是 `GPUModelRunner._sample()` 当前直接调用的 `RejectionSampler`。

源码中还存在：

```text
code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
```

这是一套更靠近新 GPU worker 分层的实现，接口围绕：

```text
InputBatch
cu_num_logits
idx_mapping
expanded_idx_mapping
positions
draft_logits
num_sampled / num_rejected
```

例如 `worker/gpu/spec_decode/rejection_sampler.py` 中：

```python
sampled, num_sampled = rejection_sample(
    processed_logits,
    draft_logits,
    draft_sampled,
    input_batch.cu_num_logits,
    pos,
    input_batch.idx_mapping,
    input_batch.expanded_idx_mapping,
    input_batch.expanded_local_pos,
    ...
)
```

位置：`worker/gpu/spec_decode/rejection_sampler.py:98` 到 `worker/gpu/spec_decode/rejection_sampler.py:132`

它的核心思想相同：

```text
验证 draft；
拒绝后从 residual distribution resample；
全接受时输出 bonus；
返回 sampled / num_sampled / num_rejected。
```

但本文档梳理的是当前 `vllm/v1/sample/rejection_sampler.py` 的主路径。

---

## 46. 容易混淆的点

### 46.1 accepted token 和 generated token 是一回事吗？

不是。

```text
accepted token：
  被接受的 draft token。

generated_token_ids：
  Scheduler 看到的最终输出 token，包含 accepted draft + recovered 或 bonus。
```

### 46.2 rejected token 会出现在输出里吗？

不会。

rejected 后的位置会是 `-1` padding，`parse_output()` 会过滤掉。

### 46.3 recovered token 是 draft token 吗？

不是。

它是拒绝 draft 后，从 target / residual distribution 采样出来的替代 token。

### 46.4 bonus token 什么时候出现？

只有所有 draft tokens 都被接受时才出现。

### 46.5 draft_probs 为 None 是错误吗？

不是。

ngram / prompt lookup 类 proposer 可能只提供 token，不提供概率。此时按 one-hot draft 处理。

### 46.6 RejectionSampler 会直接修改 Scheduler Request 吗？

不会。

它只返回 `SamplerOutput`。Scheduler Request 的更新发生在 `Scheduler.update_from_output()`。

---

## 47. 总结

`RejectionSampler` 的核心流程是：

```text
SpecDecodeMetadata
  → 拆出 target_logits 和 bonus_logits
  → bonus_logits 先用普通 Sampler 采样 bonus token
  → target_logits 应用 processors / penalties / top-k / top-p
  → greedy 请求：draft == target argmax 才接受
  → random 请求：按 min(1, target_prob / draft_prob) 接受
  → 拒绝时输出 recovered token
  → 全接受时输出 bonus token
  → 输出 [batch_size, max_spec_len + 1]，用 -1 padding
  → parse_output() 过滤 -1
  → Scheduler 统计 accepted / rejected 并写回 request
```

如果只记一句话：

```text
RejectionSampler 不负责产生 draft，而负责把 draft 转换成“accepted prefix + recovered 或 bonus”的真实 target-equivalent 输出。
```
