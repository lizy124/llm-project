# 02. SamplingParams 和 SamplingMetadata 如何进入 sampler？

源码位置：

- `code/vllm/vllm/sampling_params.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/sample/sampler.py`
- `code/vllm/vllm/v1/worker/gpu/sample/output.py`

本问题关注：用户传入的 sampling 参数如何从请求层进入 worker 侧 batch，并最终被 sampler 消费。

---

## 1. 一句话回答

`SamplingParams` 是用户级采样配置，`SamplingMetadata` 是 worker / sampler 侧按当前 batch 整理后的执行态。

```text
SamplingParams
  → request state
  → InputBatch.sampling_metadata / per-request sampling state
  → sampler input
  → sampled token / logprobs
```

---

## 2. SamplingParams 包含什么

占位：后续展开源码字段。

```text
- temperature
- top_p
- top_k
- min_p
- seed
- max_tokens
- stop / stop_token_ids
- ignore_eos
- logprobs
- prompt_logprobs
- presence_penalty
- frequency_penalty
- repetition_penalty
- n / best_of / output kind
- detokenize / skip_special_tokens
- guided decoding / structured output 相关字段
```

---

## 3. worker 侧为什么还需要 metadata

占位：解释为什么不能直接把 `SamplingParams` 丢给 kernel。

```text
因为 sampler 面对的是一个 batch：

- 每个 request 的参数可能不同；
- 本轮每个 request 采样位置不同；
- 有些请求需要 logprobs，有些不需要；
- 有些请求是 greedy，有些是 random；
- structured output / grammar 可能只作用于部分请求；
- spec decode 会改变 logits_indices 和采样位置。
```

---

## 4. 数据流占位

```text
Engine/InputProcessor
  → Request.sampling_params
  → Scheduler 持有 request 状态
  → SchedulerOutput 选中本轮请求
  → ModelRunner._update_states()
  → InputBatch 更新 sampling metadata
  → ModelRunner.sample_tokens()
  → sampler 读取 batch 化参数
```

---

## 5. 后续需要补充的源码点

```text
1. SamplingParams 的字段校验和默认值。
2. InputBatch 中 sampling metadata 的组织方式。
3. request index 如何映射到 sampling 参数。
4. batch 内 greedy / random / beam-like 或 parallel sampling 的差异。
5. logprobs / prompt_logprobs 参数如何影响后续计算。
6. stop 参数哪些在 Scheduler 检查，哪些在 OutputProcessor / detokenizer 层处理。
```

---

## 6. 一句话总结

```text
SamplingParams 是“用户想怎么采”，SamplingMetadata 是“这一轮 batch 实际该怎么采”。
```
