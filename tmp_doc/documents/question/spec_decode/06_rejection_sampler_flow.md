# 06. RejectionSampler 如何接受 / 拒绝 draft tokens？

源码位置：

- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`
- `code/vllm/vllm/v1/sample/sampler.py`
- `code/vllm/vllm/v1/sample/metadata.py`

本问题关注：target logits 产生后，RejectionSampler 如何验证 draft tokens、决定接受数量、拒绝后采样 replacement，以及全接受时采样 bonus token。

---

## 1. 一句话回答

RejectionSampler 用 target model 的概率分布验证 draft token：

```text
如果 draft token 和 target 分布一致性足够高，就接受；
一旦某个 draft token 被拒绝，就从修正后的 target 分布采样替代 token；
如果所有 draft tokens 都接受，则可以继续采样一个 bonus token。
```

---

## 2. 输入占位

```text
target logits
draft token ids
draft probabilities / draft scores
SpecDecodeMetadata
SamplingMetadata
bonus token 标记
structured output mask 后的 logits
```

---

## 3. 输出占位

```text
accepted token ids
rejected position
replacement token
bonus token
logprobs tensors
accepted count / mask
```

---

## 4. 和普通 Sampler 的关系

```text
普通 decode：
  Sampler 直接从 target logits 采样。

spec decode：
  RejectionSampler 先验证 draft tokens，
  必要时内部调用普通 Sampler 采样 replacement / bonus token。
```

---

## 5. SamplingMetadata 如何复用

RejectionSampler 仍然需要 `SamplingMetadata`，因为：

```text
- temperature / top-k / top-p 仍要生效；
- allowed_token_ids / bad_words 仍要生效；
- repetition / presence / frequency penalties 仍要生效；
- seed / generator 仍要生效；
- logprobs 仍可能需要返回。
```

---

## 6. 一句话总结

```text
RejectionSampler 是 spec decode 的核心判决器：它决定哪些 draft tokens 真正成为输出。
```
