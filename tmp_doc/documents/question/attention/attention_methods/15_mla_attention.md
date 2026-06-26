# 15. MLA Attention：Multi-Head Latent Attention 如何接入 vLLM？

源码位置：

- `code/vllm/vllm/model_executor/layers/mla.py`
- `code/vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/model_executor/models/`

本文用于梳理 MLA 的模型结构特点、KV cache 表示方式、FlashMLA / MLA backend 接入，以及它和普通 MHA / GQA attention 的差异。

---

## 1. 本文要回答的问题

```text
MLA 和普通 MHA / GQA 的结构差异是什么？
MLA 的 KV cache 保存什么？
MLACommonMetadata 是什么？
FlashMLA backend 如何接入？
DeepSeek 类模型在 vLLM 中如何走 MLA 路径？
MLA 对 KV connector / slot mapping / paged cache 有什么特殊要求？
```

---

## 2. 核心结论占位

占位：后续补充 MLA attention 的结构与执行路径。

```text
MLA 是模型结构和 KV 表示方式的变化：它通过 latent 表示压缩或重组 K/V 信息，vLLM 需要用专门的 metadata、KV cache layout 和 backend 路径来执行。
```

---

## 3. 待梳理源码点

```text
MultiHeadLatentAttentionWrapper
MLACommonMetadata
MLA attention layer
FlashMLA backend
model-specific MLA layer
KV cache shape for MLA
slot mapping for MLA
```

---

## 4. 主链路占位

```text
DeepSeek / MLA model layer
  → MLA-specific Q / latent KV projection
  → MLA metadata
  → MLA KV cache layout
  → FlashMLA / MLA backend forward
  → attention output
```

---

## 5. 后续重点占位

```text
- MLA cache 和普通 K/V cache 的 shape 差异；
- MLACommonMetadata 与普通 AttentionMetadata 的差异；
- FlashMLA backend 的选择条件；
- MLA 对 prefix cache / KV transfer 的影响。
```
