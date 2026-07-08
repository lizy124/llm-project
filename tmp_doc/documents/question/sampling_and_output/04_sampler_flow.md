# 04. Sampler 如何采样下一个 token？

源码位置：

- `vllm/vllm/v1/worker/gpu/model_runner.py`
- `vllm/vllm/v1/worker/gpu/sample/sampler.py`
- `vllm/vllm/v1/worker/gpu/sample/states.py`
- `vllm/vllm/v1/worker/gpu/sample/logit_bias.py`
- `vllm/vllm/v1/worker/gpu/sample/penalties.py`
- `vllm/vllm/v1/worker/gpu/sample/bad_words.py`
- `vllm/vllm/v1/worker/gpu/sample/min_p.py`
- `vllm/vllm/v1/worker/gpu/sample/gumbel.py`
- `vllm/vllm/v1/worker/gpu/sample/logprob.py`
- `vllm/vllm/v1/sample/sampler.py`
- `vllm/vllm/v1/sample/ops/topk_topp_sampler.py`
- `vllm/vllm/v1/worker/gpu/sample/output.py`
- `vllm/vllm/sampling_params.py`

本问题关注：Sampler 如何把 logits 变成 sampled token ids，以及 `temperature`、`top_k`、`top_p`、`min_p`、penalty、seed、grammar、logprobs 等参数如何在采样链路中发挥作用。

---

## 1. 一句话回答

Sampler 的职责是：

```text
每个 request 的 logits
  → 结构化输出 / allowed tokens / bad words / min_tokens / logit_bias
  → repetition / frequency / presence penalties
  → temperature / min_p / top_k / top_p
  → greedy 或随机采样
  → sampled_token_ids + logprobs_tensors
```

在新的 GPU ModelRunner 路径中，采样入口是：

```python
sampler_output = self.sampler(logits, input_batch)
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1055` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1058`

所以可以记成：

```text
ModelRunner 负责算 logits；
Sampler 负责把 logits 按每个请求的 SamplingParams 变成 token。
```

---

## 2. Sampler 在整条执行链路中的位置

generation 模型的一步输出大致是：

```text
SchedulerOutput
  → ModelRunner.prepare_inputs()
  → model forward
  → hidden_states[input_batch.logits_indices]
  → model.compute_logits()
  → grammar bitmask（可选）
  → Sampler
  → AsyncOutput / ModelRunnerOutput
  → Scheduler.update_from_output()
```

在新的 `v1/worker/gpu/model_runner.py` 中，采样前先取需要 logits 的 hidden states：

```python
sample_hidden_states = hidden_states[input_batch.logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1043` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1044`

如果有 structured output，则先修改 logits，再采样：

```python
self.structured_outputs_worker.apply_grammar_bitmask(...)
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1045` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1053`

然后才进入：

```python
sampler_output = self.sampler(logits, input_batch)
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1055` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1058`

---

## 3. 两套 Sampler 的分工

源码里容易混淆的是有两套采样实现：

```text
vllm/v1/worker/gpu/sample/sampler.py
  新 GPU ModelRunner 使用的 stateful GPU sampler。
  每个 request 的 sampling 参数被维护在 GPU/CPU backing state 中。

vllm/v1/sample/sampler.py
  较通用的 nn.Module sampler。
  输入是 SamplingMetadata，旧的 gpu_model_runner.py 路径会使用它。
```

### 3.1 新 GPU sampler

定义在：`vllm/vllm/v1/worker/gpu/sample/sampler.py:30`

它在初始化时持有这些状态：

```text
SamplingStates：temperature / top_k / top_p / min_p / seed / logprobs 数量；
PenaltiesState：repetition / frequency / presence penalty；
LogitBiasState：allowed_token_ids / logit_bias / min_tokens stop token mask；
BadWordsState：bad words 的 token 序列状态；
LogprobTokenIdsState：指定 token ids 的 logprobs 请求。
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:47` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:53`

新请求加入 batch 时，ModelRunner 会把 `SamplingParams` 写入 sampler：

```python
self.sampler.add_request(req_index, prompt_len, new_req_data.sampling_params)
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:801` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:805`

然后统一执行 staged writes：

```python
self.sampler.apply_staged_writes()
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:811` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:815`

### 3.2 通用 SamplingMetadata sampler

定义在：`vllm/vllm/v1/sample/sampler.py:20`

它的注释把采样顺序写得很完整：

```text
1. 如需 logprobs，先保存 raw logprobs / raw logits；
2. logits 转 float32；
3. allowed token ids；
4. bad words；
5. 非 argmax-invariant logits processors；
6. penalties；
7. greedy / random sample；
8. gather logprobs；
9. 返回 SamplerOutput。
```

位置：`vllm/vllm/v1/sample/sampler.py:20` 到 `vllm/vllm/v1/sample/sampler.py:59`

这套 sampler 的核心逻辑和新 GPU sampler 一致，只是参数组织方式不同：

```text
新 GPU sampler：参数长期存在 SamplingStates / PenaltiesState 等状态里；
通用 sampler：参数随 SamplingMetadata 一次性传入 forward()。
```

---

## 4. Sampler 的输入是什么

新 GPU sampler 的入口是：

```python
def __call__(
    self,
    logits: torch.Tensor,
    input_batch: InputBatch,
) -> SamplerOutput:
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:72` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:76`

它从 `InputBatch` 里取出：

```text
expanded_idx_mapping：logits row 到 request state idx 的映射；
idx_mapping_np：当前 batch 的 request state idx；
cu_num_logits_np：expanded logits 的累积边界；
expanded_local_pos：expanded logits 内每个 token 的本地位置；
positions[input_batch.logits_indices]：当前 logits 对应的位置；
input_ids[input_batch.logits_indices]：当前 logits 对应的输入 token。
```

对应代码：

```python
expanded_idx_mapping = input_batch.expanded_idx_mapping
idx_mapping_np = input_batch.idx_mapping_np
cu_num_logits_np = input_batch.cu_num_logits_np
expanded_local_pos = input_batch.expanded_local_pos
pos = input_batch.positions[input_batch.logits_indices]
input_ids = input_batch.input_ids[input_batch.logits_indices]
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:77` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:82`

这些信息的作用是：

```text
logits 可能是 [num_logits, vocab_size]，
但每一行 logits 属于哪个 request、处在该 request 的哪个位置，
需要通过 input_batch 的 mapping 才能找到对应 sampling 参数和历史 token 状态。
```

---

## 5. SamplingParams 如何进入 GPU sampler

用户传入的采样配置定义在 `SamplingParams`：

```text
temperature
top_p
top_k
min_p
seed
presence_penalty
frequency_penalty
repetition_penalty
stop_token_ids
min_tokens
logprobs
prompt_logprobs
logprob_token_ids
allowed_token_ids
logit_bias
bad_words
structured_outputs
```

位置：`vllm/vllm/sampling_params.py:224` 到 `vllm/vllm/sampling_params.py:341`

在新 GPU sampler 中，这些参数被拆到多个 state：

```text
SamplingStates.add_request()
  → temperature / top_p / top_k / min_p / seed / num_logprobs

PenaltiesState.add_request()
  → repetition_penalty / frequency_penalty / presence_penalty

LogitBiasState.add_request()
  → allowed_token_ids / logit_bias / min_tokens + stop_token_ids

BadWordsState.add_request()
  → bad_words_token_ids

LogprobTokenIdsState.add_request()
  → logprob_token_ids
```

入口：`vllm/vllm/v1/worker/gpu/sample/sampler.py:56` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:64`

### 5.1 SamplingParams 的基础校验

`SamplingParams.__post_init__()` 会先规范化参数：

```text
seed == -1 → None；
stop / stop_token_ids / bad_words 的 None 归一化为空列表；
logprobs=True → 1；
prompt_logprobs=True → 1；
temperature 接近 0 时转为 greedy，并重置 top_p/top_k/min_p。
```

位置：`vllm/vllm/sampling_params.py:429` 到 `vllm/vllm/sampling_params.py:476`

基础范围校验包括：

```text
temperature ∈ [0, 2]
top_p ∈ (0, 1]
top_k = 0/-1 或 >= 1
min_p ∈ [0, 1]
presence_penalty / frequency_penalty ∈ [-2, 2]
repetition_penalty > 0
```

位置：`vllm/vllm/sampling_params.py:499` 到 `vllm/vllm/sampling_params.py:551`

### 5.2 greedy 的特殊规则

当 `temperature < _SAMPLING_EPS` 时：

```python
self.top_p = 1.0
self.top_k = 0
self.min_p = 0.0
self._verify_greedy_sampling()
```

位置：`vllm/vllm/sampling_params.py:471` 到 `vllm/vllm/sampling_params.py:476`

含义是：

```text
greedy 不需要 top_p / top_k / min_p；
并且 greedy sampling 下 n 必须为 1。
```

---

## 6. 新 GPU sampler 的主流程

`Sampler.__call__()` 的核心流程是：

```text
1. 从 InputBatch 取 request 映射和 logits 位置；
2. 可选统计 logits 中 NaN 数量；
3. 判断是否需要返回 logprobs；
4. 调用 sample() 生成 token；
5. 如果需要 logprobs，计算 top-k / 指定 token logprobs；
6. 计算每个 request 本轮有效 sampled token 数；
7. 组装 worker/gpu/sample/output.py::SamplerOutput。
```

对应代码：

```python
sampled, processed_logits = self.sample(...)
...
logprobs_tensors = compute_topk_logprobs(...)
...
sampler_output = SamplerOutput(
    sampled_token_ids=sampled.view(-1, 1),
    logprobs_tensors=logprobs_tensors,
    num_nans=num_nans,
    num_sampled=num_sampled,
    num_rejected=num_rejected,
)
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:94` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:144`

---

## 7. apply_sampling_params() 的处理顺序

真正修改 logits 的主函数是：

```python
def apply_sampling_params(...):
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:146`

处理顺序是：

```text
raw logits
  → copy 成 FP32 processed logits
  → logit bias / allowed token ids / min_tokens
  → repetition / frequency / presence penalties
  → bad words mask
  → temperature
  → min_p
  → top_k / top_p（可选跳过）
```

对应代码：

```python
logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
self.logit_bias_state.apply_logit_bias(...)
self.penalties_state.apply_penalties(...)
self.bad_words_state.apply_bad_words(...)
self.sampling_states.apply_temperature(...)
self.sampling_states.apply_min_p(...)
return self.sampling_states.apply_top_k_top_p(...)
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:156` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:196`

这个顺序很关键：

```text
约束类 mask / bias 先改可选空间；
penalty 再根据历史 token 调整分数；
temperature / min_p / top_k / top_p 最后塑造随机采样分布。
```

---

## 8. grammar / structured output 如何影响采样

structured output 不是采样后修正，而是采样前修改 logits。

新 GPU ModelRunner 中：

```python
if grammar_output is not None:
    self.structured_outputs_worker.apply_grammar_bitmask(
        logits,
        input_batch,
        grammar_output.structured_output_request_ids,
        grammar_output.grammar_bitmask,
    )
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1045` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1053`

旧 `gpu_model_runner.py` 路径中也一样：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4455` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4459`

因此顺序是：

```text
compute_logits()
  → grammar bitmask 把不合法 token logits 置为不可选
  → sampler 内部再做 allowed_token_ids / bad_words / penalties / top-k / top-p
```

这保证最终 sampled token 从概率空间上就是合法 token，而不是采样后再“纠错”。

---

## 9. allowed_token_ids / logit_bias / min_tokens

这三类逻辑集中在：

```text
vllm/vllm/v1/worker/gpu/sample/logit_bias.py
```

`LogitBiasState.add_request()` 会记录：

```text
allowed_token_ids：只允许这些 token；
logit_bias：给指定 token 加 bias；
min_tokens：没达到最小生成长度前，屏蔽 stop tokens。
```

位置：`vllm/vllm/v1/worker/gpu/sample/logit_bias.py:52` 到 `vllm/vllm/v1/worker/gpu/sample/logit_bias.py:107`

真正 kernel 中的顺序是：

```text
1. 如果有 allowed_token_ids：
   先把整行 logits 写成 -inf，
   再恢复 allowed token 的原 logits。

2. 如果有 logit_bias：
   对指定 token 的 logits 加 bias。

3. 如果有 min_tokens 且当前位置还没达到 min_len：
   把 stop_token_ids 写成 -inf。
```

位置：`vllm/vllm/v1/worker/gpu/sample/logit_bias.py:177` 到 `vllm/vllm/v1/worker/gpu/sample/logit_bias.py:235`

可以理解为：

```text
allowed_token_ids 决定“候选集合”；
logit_bias 调整“候选分数”；
min_tokens 临时禁止“过早停止”。
```

---

## 10. bad_words 如何屏蔽

bad words 的逻辑在：

```text
vllm/vllm/v1/worker/gpu/sample/bad_words.py
```

每个 bad word 被编码成 token 序列。采样时不是永远屏蔽整个序列，而是：

```text
如果当前已经生成的尾部 token 匹配某个 bad word 的前缀，
就把该 bad word 的最后一个 token logits 置为 -inf，
从而阻止完整 bad word 被生成出来。
```

对应 kernel 中的关键逻辑：

```python
prefix_len = bad_word_len - 1
...
if match:
    tl.store(logits_ptr + token_idx * logits_stride + last_token, -float("inf"))
```

位置：`vllm/vllm/v1/worker/gpu/sample/bad_words.py:140` 到 `vllm/vllm/v1/worker/gpu/sample/bad_words.py:162`

它会同时考虑：

```text
历史输出 token；
当前 step / speculative 输入中的 token；
当前 token 在 expanded logits 中的位置。
```

所以 bad words 是一个依赖“上下文尾部匹配”的动态 mask。

---

## 11. penalties 如何生效

penalty 逻辑在：

```text
vllm/vllm/v1/worker/gpu/sample/penalties.py
```

`PenaltiesState` 保存：

```text
repetition_penalty
frequency_penalty
presence_penalty
prompt_bin_mask
output_bin_counts
```

位置：`vllm/vllm/v1/worker/gpu/sample/penalties.py:22` 到 `vllm/vllm/v1/worker/gpu/sample/penalties.py:42`

### 11.1 prompt_bin_mask 和 output_bin_counts

新请求加入时，如果启用了 penalty，会先统计：

```text
prompt_bin_mask：prompt 中出现过哪些 token；
output_bin_counts：已经生成的输出 token 出现次数。
```

位置：`vllm/vllm/v1/worker/gpu/sample/penalties.py:56` 到 `vllm/vllm/v1/worker/gpu/sample/penalties.py:75`

### 11.2 repetition penalty

kernel 中：

```text
如果 token 出现在 prompt 或 output 中：
  logits > 0：logits /= repetition_penalty
  logits <= 0：logits *= repetition_penalty
```

对应代码：

```python
scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
logits *= tl.where(logits > 0, 1.0 / scale, scale)
```

位置：`vllm/vllm/v1/worker/gpu/sample/penalties.py:173` 到 `vllm/vllm/v1/worker/gpu/sample/penalties.py:176`

### 11.3 frequency / presence penalty

kernel 中：

```python
logits -= freq_penalty * output_bin_counts
logits -= pres_penalty * output_bin_mask
```

位置：`vllm/vllm/v1/worker/gpu/sample/penalties.py:178` 到 `vllm/vllm/v1/worker/gpu/sample/penalties.py:181`

含义是：

```text
frequency_penalty：出现次数越多，扣分越多；
presence_penalty：只要出现过，就扣一次分。
```

---

## 12. temperature 如何生效

`temperature` 存在 `SamplingStates.temperature` 中：

```python
self.temperature = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
```

位置：`vllm/vllm/v1/worker/gpu/sample/states.py:22`

加入请求时写入：

```python
self.temperature.np[req_idx] = sampling_params.temperature
```

位置：`vllm/vllm/v1/worker/gpu/sample/states.py:42`

采样时：

```python
self.sampling_states.apply_temperature(logits, expanded_idx_mapping, idx_mapping_np)
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:182` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:185`

kernel 的逻辑是：

```text
如果 temperature == 0 或 1：跳过；
否则 logits = logits / temperature。
```

位置：`vllm/vllm/v1/worker/gpu/sample/gumbel.py:26` 到 `vllm/vllm/v1/worker/gpu/sample/gumbel.py:40`

效果是：

```text
temperature 越小，分布越尖锐；
temperature 越大，分布越平；
temperature == 0，后续 gumbel_sample 退化为 argmax。
```

---

## 13. min_p 如何生效

`min_p` 表示：

```text
只保留概率至少达到 max_prob * min_p 的 token。
```

在 logits 空间中等价于：

```text
logit >= max_logit + log(min_p)
```

kernel 里先找每行最大 logits：

```python
max_val = tl.max(...)
```

再计算阈值：

```python
threshold = max_val + tl.log(min_p)
logits = tl.where(logits < threshold, float("-inf"), logits)
```

位置：`vllm/vllm/v1/worker/gpu/sample/min_p.py:23` 到 `vllm/vllm/v1/worker/gpu/sample/min_p.py:45`

采样主链路中，`min_p` 在 temperature 之后、top-k/top-p 之前：

```python
self.sampling_states.apply_temperature(...)
self.sampling_states.apply_min_p(...)
...
self.sampling_states.apply_top_k_top_p(...)
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:182` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:195`

---

## 14. top_k / top_p 如何过滤

top-k/top-p 的通用实现位于：

```text
vllm/vllm/v1/sample/ops/topk_topp_sampler.py
```

入口：

```python
def apply_top_k_top_p(logits, k, p)
```

位置：`vllm/vllm/v1/sample/ops/topk_topp_sampler.py:345`

### 14.1 top_k

PyTorch 实现中，top-k 的做法是：

```text
1. 对 logits 排序；
2. 找到每行第 k 大的阈值；
3. 小于该阈值的 logits 写成 -inf。
```

对应代码：

```python
top_k_mask = logits_sort.size(1) - k.to(torch.long)
top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
top_k_mask = logits_sort < top_k_mask
logits_sort.masked_fill_(top_k_mask, -float("inf"))
```

位置：`vllm/vllm/v1/sample/ops/topk_topp_sampler.py:386` 到 `vllm/vllm/v1/sample/ops/topk_topp_sampler.py:392`

### 14.2 top_p

top-p 的做法是：

```text
1. 按 logits 从低到高排序；
2. softmax 得到排序后的概率；
3. 从低概率端累计；
4. 累计概率 <= 1 - top_p 的低概率 token 被屏蔽；
5. 最高概率 token 至少保留一个。
```

对应代码：

```python
probs_sort = logits_sort.softmax(dim=-1)
probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
top_p_mask[:, -1] = False
logits_sort.masked_fill_(top_p_mask, -float("inf"))
```

位置：`vllm/vllm/v1/sample/ops/topk_topp_sampler.py:394` 到 `vllm/vllm/v1/sample/ops/topk_topp_sampler.py:401`

最后再 scatter 回原 vocab 顺序：

```python
return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)
```

位置：`vllm/vllm/v1/sample/ops/topk_topp_sampler.py:403` 到 `vllm/vllm/v1/sample/ops/topk_topp_sampler.py:404`

### 14.3 top_k/top_p 的实现选择

`apply_top_k_top_p()` 会根据平台和 batch size 选择实现：

```text
可用 Triton 且 batch 较大：Triton 实现；
不适合 Triton 或小 batch：PyTorch sort 实现；
不同平台会按当前能力和 batch size 选择实现。
```

位置：`vllm/vllm/v1/sample/ops/topk_topp_sampler.py:345` 到 `vllm/vllm/v1/sample/ops/topk_topp_sampler.py:360`

---

## 15. random sampling 为什么不用 torch.multinomial

通用 sampler 的随机采样实现是：

```python
q = empty_exponential_noise_like(probs, use_fp64_gumbel)
q.exponential_()
return sample_with_exponential_noise(probs, q)
```

位置：`vllm/vllm/v1/sample/ops/topk_topp_sampler.py:446` 到 `vllm/vllm/v1/sample/ops/topk_topp_sampler.py:468`

注释说明：

```text
不用 torch.multinomial，
因为 torch.multinomial 会导致 CPU-GPU synchronization。
```

位置：`vllm/vllm/v1/sample/ops/topk_topp_sampler.py:451` 到 `vllm/vllm/v1/sample/ops/topk_topp_sampler.py:454`

其数学形式可以理解为：

```text
sample token = argmax(probs / exponential_noise)
```

而新 GPU sampler 直接使用 Gumbel-max 风格的 Triton kernel：

```python
sampled = gumbel_sample(...)
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:234` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:243`

---

## 16. gumbel_sample 如何同时支持 greedy 和 random

新 GPU sampler 的最终采样函数是：

```python
def gumbel_sample(...):
```

位置：`vllm/vllm/v1/worker/gpu/sample/gumbel.py:205`

kernel 中：

```text
如果 temp != 0：
  根据 seed 和 pos 生成 gumbel noise；
  logits += gumbel_noise；

然后对 logits 做 argmax。
```

对应代码：

```python
if temp != 0.0:
    seed = tl.load(seeds_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx)
    gumbel_seed = tl.randint(seed, pos)
    ...
    logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

value, idx = tl.max(logits, axis=0, return_indices=True)
```

位置：`vllm/vllm/v1/worker/gpu/sample/gumbel.py:124` 到 `vllm/vllm/v1/worker/gpu/sample/gumbel.py:147`

所以：

```text
temperature == 0：不加随机噪声，argmax 就是 greedy；
temperature > 0：加 gumbel noise，再 argmax，相当于按概率分布采样。
```

这也是为什么新 GPU sampler 不需要像通用 sampler 那样显式分出 greedy_sample 和 random_sample 两条完整路径。

---

## 17. seed 如何保证请求级随机性

`SamplingStates` 为每个 request 保存一个 seed：

```python
self.seeds = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
self.seeds_set = np.zeros(max_num_reqs, dtype=bool)
```

位置：`vllm/vllm/v1/worker/gpu/sample/states.py:26` 到 `vllm/vllm/v1/worker/gpu/sample/states.py:29`

加入请求时：

```python
seed = sampling_params.seed
self.seeds_set[req_idx] = seed is not None
if seed is None:
    seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
self.seeds.np[req_idx] = seed
```

位置：`vllm/vllm/v1/worker/gpu/sample/states.py:50` 到 `vllm/vllm/v1/worker/gpu/sample/states.py:54`

采样 kernel 中再结合位置生成当前 token 的随机种子：

```python
seed = tl.load(seeds_ptr + req_state_idx)
pos = tl.load(pos_ptr + token_idx)
gumbel_seed = tl.randint(seed, pos)
```

位置：`vllm/vllm/v1/worker/gpu/sample/gumbel.py:124` 到 `vllm/vllm/v1/worker/gpu/sample/gumbel.py:128`

含义是：

```text
同一个 request 的随机序列由 request seed 控制；
不同生成位置通过 pos 派生不同随机数；
未显式设置 seed 的请求会分配随机 seed。
```

FlashInfer 路径有一个限制：如果当前 batch 中有显式 seed，则新 GPU sampler 不走 FlashInfer，而回退到能处理 per-request seed 的路径。

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:220` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:228`

---

## 18. sample() 如何选择 FlashInfer / 原生路径

新 GPU sampler 的 `sample()` 会先处理除 top-k/top-p 以外的采样参数：

```python
processed_logits = self.apply_sampling_params(..., skip_top_k_top_p=True)
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:208` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:216`

然后取出当前 batch 是否真的需要 top-k/top-p：

```python
top_k, top_p = self.sampling_states.get_top_k_top_p(...)
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:217` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:219`

FlashInfer 只有在这些条件都满足时才使用：

```text
有 top_k 或 top_p；
不需要返回 processed_logprobs；
没有 greedy 请求；
没有显式 per-request seed。
```

对应代码：

```python
use_flashinfer = self.use_flashinfer and not (
    (top_k is None and top_p is None)
    or (return_logprobs and self.logprobs_mode == "processed_logprobs")
    or self.sampling_states.any_greedy(idx_mapping_np)
    or self.sampling_states.any_explicit_seed(idx_mapping_np)
)
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:220` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:228`

之后二选一：

```text
FlashInfer：flashinfer_sample(processed_logits, top_k, top_p)
原生路径：apply_top_k_top_p() → gumbel_sample()
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:230` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:244`

---

## 19. logprobs 如何计算

Sampler 输出的 logprobs 不是直接把整张 `[batch, vocab]` 概率表全部返回，而是按请求需要计算。

### 19.1 判断是否需要 logprobs

新 GPU sampler 中：

```python
max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
max_per_req_token_ids = self.logprob_token_ids_state.max_num_token_ids(idx_mapping_np)
return_logprobs = max_num_logprobs != NO_LOGPROBS or max_per_req_token_ids > 0
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:88` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:92`

### 19.2 计算哪些 logprobs

如果需要 logprobs，会调用：

```python
logprobs_tensors = compute_topk_logprobs(...)
```

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:104` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:118`

`compute_topk_logprobs()` 支持两类需求：

```text
logprobs=N：返回 sampled token + top-N token 的 logprobs；
logprob_token_ids=[...]：返回 sampled token + 指定 token ids 的 logprobs。
```

位置：`vllm/vllm/v1/worker/gpu/sample/logprob.py:101` 到 `vllm/vllm/v1/worker/gpu/sample/logprob.py:170`

### 19.3 不物化完整 logprobs 的优化

`compute_token_logprobs()` 的注释说明：

```text
为了节省 GPU 内存，不 materialize 完整 [batch_size, vocab_size] logprobs；
kernel 只计算每行的 max + logsumexp，
并只输出指定 token ids 的 logprobs。
```

位置：`vllm/vllm/v1/worker/gpu/sample/logprob.py:78` 到 `vllm/vllm/v1/worker/gpu/sample/logprob.py:98`

这意味着：

```text
采样用 logits；
返回 logprobs 时，只为 sampled/top-k/指定 token 计算需要的概率。
```

---

## 20. SamplerOutput 包含什么

新 GPU sampler 的输出结构是：

```python
@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor | None
    num_rejected: torch.Tensor | None = None
```

位置：`vllm/vllm/v1/worker/gpu/sample/output.py:10` 到 `vllm/vllm/v1/worker/gpu/sample/output.py:16`

字段含义：

```text
sampled_token_ids：GPU 上的 token ids，通常 shape 为 [num_reqs, 1]；
logprobs_tensors：生成 token 的 logprobs 张量；
num_nans：可选的 logits NaN 统计；
num_sampled：每个 request 本轮有效 sampled token 数；
num_rejected：spec decode 中每个 request 被拒绝的 token 数。
```

普通一步 decode 通常是：

```text
每个 request 采 1 个 token；
num_rejected = 0。
```

chunked prefill / speculative decoding 下，某些请求可能没有输出 token，或者一轮产生多个 token。

位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:122` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:131`

---

## 21. speculative decoding 下 Sampler 如何变化

在新 GPU ModelRunner 中，如果当前 batch 没有 draft tokens，或者没有 rejection sampler，就走普通 sampler：

```python
if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
    sampler_output = self.sampler(logits, input_batch)
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1055` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1058`

如果有 draft tokens，则走 rejection sampling：

```python
sampler_output = self.rejection_sampler(
    logits,
    input_batch,
    self.speculator.draft_logits,
)
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1059` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1067`

区别是：

```text
普通采样：target logits 直接选下一个 token；
spec decode：target logits 用于验证 draft tokens，并可能补采 bonus token。
```

但是最终仍然回到同一种输出抽象：

```text
SamplerOutput.sampled_token_ids
SamplerOutput.num_sampled
SamplerOutput.num_rejected
```

---

## 22. 采样后如何进入 ModelRunnerOutput

新 GPU ModelRunner 的 `sample_tokens()` 中，最后 PP rank 会：

```text
1. 调用 self.sample(hidden_states, input_batch, grammar_output)；
2. 必要时把 sampled tokens 广播给非最后 PP rank；
3. 计算 prompt_logprobs；
4. 构造 ModelRunnerOutput；
5. 创建 AsyncOutput，把 GPU sampled/logprobs 异步拷到 CPU；
6. postprocess_sampled() 更新 request states；
7. speculator 生成下一轮 draft tokens；
8. kv_connector.post_forward()；
9. 返回 AsyncOutput。
```

对应代码范围：`vllm/vllm/v1/worker/gpu/model_runner.py:1359` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1467`

关键对象：

```python
async_output = AsyncOutput(
    model_runner_output=model_runner_output,
    sampler_output=sampler_output,
    num_sampled_tokens=num_sampled,
    main_stream=self.main_stream,
    copy_stream=self.output_copy_stream,
)
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1392` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1399`

也就是说：

```text
SamplerOutput 是 GPU 侧采样结果；
AsyncOutput / ModelRunnerOutput 是交给 Engine/Scheduler 后续消费的结构化输出。
```

---

## 23. prompt_logprobs 和 sampled logprobs 的区别

sampled logprobs：

```text
针对本轮新采样出来的 token；
由 Sampler 根据当前 logits 计算；
存在 sampler_output.logprobs_tensors。
```

prompt logprobs：

```text
针对 prompt token；
在 ModelRunner.sample_tokens() 中由 PromptLogprobsWorker 计算；
最后放入 ModelRunnerOutput.prompt_logprobs_dict。
```

新 GPU ModelRunner 中：

```python
prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
    self.model.compute_logits,
    hidden_states,
    input_batch,
    self.req_states.all_token_ids.gpu,
    self.req_states.num_computed_tokens.gpu,
    self.req_states.prompt_len.np,
)
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1373` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1381`

因此要区分：

```text
Sampler 负责 sampled token logprobs；
PromptLogprobsWorker 负责 prompt token logprobs。
```

---

## 24. 通用 sampler 的 greedy / random 混合逻辑

旧 `vllm/v1/sample/sampler.py` 路径更显式地区分 greedy 和 random。

入口：

```python
def sample(self, logits, sampling_metadata, ...)
```

位置：`vllm/vllm/v1/sample/sampler.py:243`

逻辑是：

```text
1. 如果不是 all_random，先计算 greedy_sampled = argmax(logits)。
2. 如果 all_greedy，直接返回 greedy_sampled。
3. 否则应用 temperature。
4. 应用 argmax-invariant logits processors（默认包括 min_p）。
5. 应用 top_k/top_p 并随机采样。
6. 混合 batch 中 greedy 行和 random 行：
   temperature < eps 的行取 greedy_sampled，其他行取 random_sampled。
```

对应代码：

```python
greedy_sampled = self.greedy_sample(logits)
...
logits = self.apply_temperature(...)
...
random_sampled, processed_logprobs = self.topk_topp_sampler(...)
...
sampled = torch.where(
    sampling_metadata.temperature < _SAMPLING_EPS,
    greedy_sampled,
    random_sampled,
)
```

位置：`vllm/vllm/v1/sample/sampler.py:255` 到 `vllm/vllm/v1/sample/sampler.py:302`

这套逻辑和新 GPU sampler 的 `temperature == 0 → 不加 gumbel noise → argmax` 在语义上等价。

---

## 25. 常见参数的最终效果

### 25.1 temperature

```text
0：greedy；
(0, 1)：降低随机性；
1：不缩放 logits；
>1：提高随机性。
```

源码点：`vllm/vllm/v1/worker/gpu/sample/gumbel.py:26` 到 `vllm/vllm/v1/worker/gpu/sample/gumbel.py:40`

### 25.2 top_k

```text
只保留 logits 最高的 k 个 token；
0 或 -1 表示不限制，内部会按 vocab_size 处理。
```

源码点：`vllm/vllm/v1/worker/gpu/sample/states.py:44` 到 `vllm/vllm/v1/worker/gpu/sample/states.py:47`

### 25.3 top_p

```text
保留累计概率质量达到 top_p 的高概率 token 集合；
1.0 表示不限制。
```

源码点：`vllm/vllm/v1/sample/ops/topk_topp_sampler.py:394` 到 `vllm/vllm/v1/sample/ops/topk_topp_sampler.py:401`

### 25.4 min_p

```text
只保留相对最高概率足够大的 token；
0 表示不限制。
```

源码点：`vllm/vllm/v1/worker/gpu/sample/min_p.py:35` 到 `vllm/vllm/v1/worker/gpu/sample/min_p.py:45`

### 25.5 repetition_penalty

```text
对 prompt 或 output 中出现过的 token 改 logits；
>1 通常抑制重复；
<1 可能鼓励重复。
```

源码点：`vllm/vllm/v1/worker/gpu/sample/penalties.py:173` 到 `vllm/vllm/v1/worker/gpu/sample/penalties.py:176`

### 25.6 frequency_penalty / presence_penalty

```text
frequency_penalty：按出现次数扣分；
presence_penalty：只要出现过就扣分。
```

源码点：`vllm/vllm/v1/worker/gpu/sample/penalties.py:178` 到 `vllm/vllm/v1/worker/gpu/sample/penalties.py:181`

### 25.7 seed

```text
显式 seed 控制单个 request 的随机序列；
未设置 seed 时，系统为 request 随机分配 seed。
```

源码点：`vllm/vllm/v1/worker/gpu/sample/states.py:50` 到 `vllm/vllm/v1/worker/gpu/sample/states.py:54`

---

## 26. 一条完整采样时间线

可以把一次普通 decode 采样理解成：

```text
1. Scheduler 把 SamplingParams 随请求交给 ModelRunner；
2. ModelRunner.add_requests() 调用 sampler.add_request()；
3. sampler 把参数写入 SamplingStates / PenaltiesState / LogitBiasState 等；
4. execute_model() 完成 forward，得到 hidden_states；
5. sample_tokens() 取 hidden_states[input_batch.logits_indices]；
6. model.compute_logits() 得到 logits；
7. structured output grammar bitmask 修改 logits；
8. sampler.__call__(logits, input_batch)；
9. apply_sampling_params() 处理 bias / mask / penalties / temperature / min_p；
10. top_k/top_p 过滤；
11. gumbel_sample 或 FlashInfer 采样；
12. 可选 compute_topk_logprobs()；
13. 返回 SamplerOutput；
14. ModelRunner 创建 AsyncOutput / ModelRunnerOutput；
15. postprocess_sampled() 更新 request state；
16. Scheduler 后续消费输出并推进请求状态。
```

---

## 27. 容易疑惑的点

### 27.1 grammar bitmask 和 top-k/top-p 谁先执行？

grammar bitmask 先执行。

```text
compute_logits()
  → grammar bitmask
  → sampler 内部 top_k/top_p
```

这样 top-k/top-p 看到的候选空间已经排除了 grammar 不允许的 token。

### 27.2 allowed_token_ids 和 grammar 是一回事吗？

不是。

```text
grammar：来自 structured output，每一步根据语法状态动态约束；
allowed_token_ids：来自 SamplingParams，是请求级固定白名单。
```

它们都会通过 logits mask 影响采样空间。

### 27.3 temperature=0 是否还会走 top-k/top-p？

参数规范化时，temperature 接近 0 会把：

```text
top_p = 1.0
top_k = 0
min_p = 0.0
```

位置：`vllm/vllm/sampling_params.py:471` 到 `vllm/vllm/sampling_params.py:476`

因此 greedy 请求不需要 top-k/top-p/min-p。

### 27.4 penalties 使用 prompt tokens 还是 output tokens？

两者都可能使用：

```text
repetition_penalty：prompt 或 output 中出现过的 token 都会影响；
frequency / presence penalty：主要基于 output 出现次数 / 是否出现。
```

源码点：`vllm/vllm/v1/worker/gpu/sample/penalties.py:173` 到 `vllm/vllm/v1/worker/gpu/sample/penalties.py:181`

### 27.5 为什么 sampled_token_ids 是二维的？

`SamplerOutput.sampled_token_ids` 通常是 `[num_reqs, 1]`。

原因是输出层需要兼容：

```text
普通 decode：每个 request 1 个 token；
chunked prefill：某些 request 本轮没有 token；
spec decode：一轮可能接受多个 token。
```

### 27.6 为什么有时返回 AsyncOutput？

新 GPU ModelRunner 中采样结果还在 GPU 上，`AsyncOutput` 会让 GPU→CPU 拷贝和后续工作重叠。

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1392` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1399`

---

## 28. 总结

Sampler 的完整心智模型是：

```text
SamplingParams
  → per-request sampler states
  → logits processors / masks / penalties
  → temperature / min_p / top_k / top_p
  → greedy or gumbel / FlashInfer sampling
  → sampled_token_ids
  → optional logprobs_tensors
  → ModelRunnerOutput / AsyncOutput
```

如果只记住一句话：

```text
Sampler 不是简单 argmax；它是把每个 request 的 SamplingParams、结构化约束、历史 token 状态和随机数状态一起应用到 logits 上，最终在合法且过滤后的概率空间中选出下一个 token。
```
