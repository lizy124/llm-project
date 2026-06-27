# 09. Pooling / Embedding / Rerank 模型如何接入？

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/layers/pooler.py`
- `code/vllm/vllm/v1/pool/metadata.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/pooling_params.py`

本问题关注：非生成模型如何在模型架构层接入 pooler，并与生成式模型区分。

---

## 1. 一句话回答

Pooling / embedding / rerank 模型共享模型 forward 前半段，但不接 lm_head + sampler，而是在 hidden states 后接 pooler。

---

## 2. 模型结构占位

```text
input ids
  → backbone
  → hidden states
  → pooler
      → embedding vector
      → classification logits / probs
      → score scalar
      → token-level output
```

---

## 3. 需要梳理的问题

```text
- pooler 如何定义；
- pooling method 是 last / mean / cls / all tokens 还是模型自定义；
- PoolingMetadata 如何提供 prompt lens / cursor；
- task=embed/classify/token_embed/token_classify 如何影响 pooler；
- rerank query-doc pair 如何变成 classify/score 输出。
```

---

## 4. 和 generation model 的区别

```text
generation model：
  hidden states → compute_logits → sampler。

pooling model：
  hidden states → pooler → PoolingRequestOutput。
```

---

## 5. 一句话总结

```text
pooling 模型架构的关键，是把 hidden states 转成向量、类别或分数，而不是 token logits。
```
