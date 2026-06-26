# 05. Spec decode 如何影响采样和输出？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/outputs.py`

本问题关注：speculative decoding 下，draft token、target logits、bonus token、rejection sampler、Scheduler 回收之间的关系。

---

## 1. 一句话回答

Spec decode 让一轮采样不再只是“一个 request 采一个 token”，而是变成：

```text
先验证 draft tokens，
再决定接受多少，
必要时采样 bonus token，
最后把接受 / 拒绝结果回收到 Scheduler 状态。
```

---

## 2. 主链路占位

```text
Scheduler 选择 spec decode 请求
  → SchedulerOutput 携带 scheduled_spec_decode_tokens
  → ModelRunner._prepare_inputs()
      → draft token positions
      → target_logits_indices
      → bonus_logits_indices
      → SpecDecodeMetadata
  → target model forward
  → target logits
  → rejection sampler
      → accepted tokens
      → rejected token 后的 sampled token
      → bonus token
  → ModelRunnerOutput
  → Scheduler.update_from_output()
      → append accepted tokens
      → 回退 rejected 部分
      → 更新 num_computed_tokens / draft state
```

---

## 3. spec decode metadata 占位

后续补充：

```text
- draft_token_ids
- num_draft_tokens
- cu_num_draft_tokens
- target_logits_indices
- bonus_logits_indices
- logits_indices
- accepted token mask
- sample positions
```

---

## 4. rejection sampler 占位

需要回答：

```text
1. draft token 如何和 target probability 比较？
2. 接受多个 draft token 后，bonus token 如何产生？
3. 拒绝某个 draft token 后，替代 token 如何采样？
4. logprobs 如何对应 accepted / rejected / bonus token？
5. grammar / structured output 是否参与 spec decode 采样？
```

---

## 5. 容易混淆点占位

```text
1. draft token 不是最终一定输出的 token。
2. target model logits 是 spec decode 接受 / 拒绝的依据。
3. bonus token 是为了在全部 draft 接受时继续推进一个 token。
4. Scheduler.update_from_output() 必须修正 request 的 token 和 KV 状态。
5. spec decode 的 logits_indices 和普通 decode 不一样。
```

---

## 6. 一句话总结

```text
Spec decode 把采样变成“验证 draft + 必要时重新采样 + 状态回滚/推进”的组合过程。
```
