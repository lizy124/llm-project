# 01. Spec decode 在 vLLM 中负责什么？

源码位置：

- `code/vllm/vllm/v1/spec_decode/`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/outputs.py`

本问题关注：speculative decoding 在 vLLM 推理链路中的定位、目标、收益和边界。

---

## 1. 一句话回答

Spec decode 通过“先猜多个 token，再用 target model 一次验证”的方式，减少 target model 的逐 token 调用次数，提高 decode 吞吐。

```text
普通 decode：
  target model 每轮生成 1 个 token。

spec decode：
  draft path 先给出多个候选 token，
  target model 一次验证多个位置，
  接受一段合法前缀，
  拒绝后重新采样或生成 bonus token。
```

---

## 2. 它不只是 sampler 优化

Spec decode 会同时改变：

```text
- Scheduler 本轮调度多少 token；
- ModelRunner 如何准备 input_ids / positions / logits_indices；
- attention metadata 的 token 布局；
- sampler 如何接受 / 拒绝 draft tokens；
- Scheduler 如何回收输出并修正 request 状态；
- KV cache / num_computed_tokens 如何保持一致；
- structured output / grammar 如何处理提前猜出的 tokens。
```

---

## 3. 和普通 decode 的边界占位

后续补充：

```text
普通 decode：
  logits row 通常对应每个请求的最后一个 token。

spec decode：
  logits rows 可能对应多个 draft token 的 target 验证位置，
  还可能包含 bonus token 位置。
```

---

## 4. 需要串起来的对象占位

```text
Request draft state
SchedulerOutput.scheduled_spec_decode_tokens
SpecDecodeMetadata
ExecuteModelState.spec_decode_metadata
RejectionSampler
SamplerOutput
ModelRunnerOutput
Scheduler.update_from_output()
```

---

## 5. 一句话总结

```text
Spec decode 的核心不是“多采几个 token”，而是“猜测、验证、接受、拒绝、修正状态”的完整闭环。
```
