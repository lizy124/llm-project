# 04. Sampler 如何采样下一个 token？

源码位置：

- `code/vllm/vllm/v1/worker/gpu/sample/sampler.py`
- `code/vllm/vllm/v1/sample/sampler.py`
- `code/vllm/vllm/v1/sample/ops/topk_topp_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/sample/output.py`
- `code/vllm/vllm/sampling_params.py`

本问题关注：sampler 如何把 logits 变成 sampled token ids，以及 temperature、top-k、top-p、min-p、penalty、seed 等参数如何发挥作用。

---

## 1. 一句话回答

sampler 的职责是把每个 request 对应的 logits，根据该 request 的 sampling 参数和约束，转换成一个或多个 sampled token。

```text
logits
  → penalties / bias / mask
  → temperature
  → top-k / top-p / min-p
  → softmax / sampling
  → sampled_token_ids
```

---

## 2. sampler 输入占位

```text
- logits
- sampling metadata
- selected_token_indices / logits_indices
- grammar bitmask
- request 到 batch row 的映射
- random seed / generator state
- previous output tokens / prompt tokens，用于 penalties
```

---

## 3. 采样模式占位

```text
greedy：
  temperature = 0 或等价配置，直接 argmax。

random sampling：
  应用 temperature / top-k / top-p / min-p 后采样。

parallel sampling：
  一个请求可能生成多个分支或多个样本。

spec decode target sampling：
  采样还要和 draft tokens 的接受 / 拒绝逻辑配合。
```

---

## 4. 典型处理顺序占位

```text
1. 根据请求参数准备 per-row sampling config。
2. 应用 repetition / presence / frequency penalties。
3. 应用 structured output / grammar mask。
4. 应用 temperature。
5. 应用 top-k / top-p / min-p 过滤。
6. 执行 multinomial 或 greedy。
7. 计算需要返回的 logprobs。
8. 组装 sampler output。
```

---

## 5. 后续需要补充的源码点

```text
1. GPU sampler 和通用 sampler 的分工。
2. `topk_topp_sampler.py` 的过滤逻辑。
3. seed / generator 如何保证请求级随机性。
4. penalties 的输入 token 范围。
5. grammar bitmask 和 top-k/top-p 的先后关系。
6. sampler output 如何转成 ModelRunnerOutput 字段。
```

---

## 6. 一句话总结

```text
sampler 是 logits 到 token 的转换器，SamplingParams 决定“怎么选”，grammar / spec decode 决定“哪些选择是合法的”。
```
