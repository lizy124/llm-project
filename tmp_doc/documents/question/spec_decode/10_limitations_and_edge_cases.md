# 10. Spec decode 有哪些限制和边界场景？

源码位置：

- `code/vllm/vllm/sampling_params.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/structured_output/utils.py`

本问题关注：spec decode 与采样参数、structured output、KV cache、chunked prefill、logprobs、backend 等能力之间的限制。

---

## 1. 一句话回答

Spec decode 提升 decode 吞吐，但它会放大状态复杂度，因此不是所有场景都能无条件启用。

---

## 2. sampling 参数限制占位

后续补充：

```text
- min_p 是否支持；
- logit_bias 是否支持；
- allowed_token_ids / bad_words 如何处理；
- penalties 如何按 draft token 展开；
- logprobs 如何对齐 accepted / rejected tokens。
```

---

## 3. structured output 限制占位

```text
- draft tokens 可能被 grammar 提前 trim；
- grammar bitmask 需要按 spec logits rows 对齐；
- grammar 拒绝 accepted token 时如何报错；
- spec decode 和 strict schema 的性能 / 可用性边界。
```

---

## 4. KV cache / prefix cache 边界占位

```text
- rejected draft tokens 是否导致 recompute；
- num_computed_tokens 如何修正；
- prefix cache 命中和 spec decode 是否兼容；
- external KV connector 是否影响 spec decode；
- chunked prefill 阶段是否允许 spec decode。
```

---

## 5. 输出边界占位

```text
- accepted tokens 数量可能为 0；
- 全部 draft 接受时可能有 bonus token；
- rejected 后 replacement token 如何返回；
- stop / EOS 可能出现在 accepted token 序列中间；
- logprobs 需要和最终输出 token 对齐。
```

---

## 6. 一句话总结

```text
Spec decode 的限制通常来自状态一致性：只要某个功能会改变 token 合法性、采样分布或 KV 进度，就必须和接受 / 拒绝逻辑对齐。
```
