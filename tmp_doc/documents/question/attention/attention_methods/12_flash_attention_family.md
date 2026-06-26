# 12. FlashAttention family：v1 / v2 / v3 / v4 与 vLLM 的关系

源码位置：

- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/vllm_flash_attn/`
- `code/vllm/vllm/platforms/`
- `code/vllm/vllm/v1/attention/selector.py`

本文用于梳理 FlashAttention 家族的概念、版本差异和 vLLM 中的 backend 接入方式。这里的 FlashAttention 是 kernel / backend 家族，不等同于 PagedAttention，也不等同于 MHA / GQA 这种模型结构。

---

## 1. 本文要回答的问题

```text
FlashAttention v1 / v2 / v3 / v4 分别解决什么问题？
vLLM 中哪些 backend 使用 FlashAttention 类 kernel？
FlashAttention 和 PagedAttention 如何配合？
FlashAttention 和 FlashInfer / FlashMLA 是什么关系？
不同硬件平台如何影响 FlashAttention backend 选择？
FlashAttention family 对 prefill / decode / paged KV cache 的支持边界是什么？
```

---

## 2. 核心结论占位

占位：后续补充 FlashAttention family 的版本脉络和 vLLM 接入方式。

```text
FlashAttention family 主要解决 attention kernel 的 IO / memory / parallelism 效率问题；vLLM 会把它作为具体 AttentionBackend 或 backend 内部 kernel，用于执行由 ModelRunner metadata 描述的 paged attention 工作。
```

---

## 3. 待梳理源码点

```text
FlashAttention backend
vllm_flash_attn
backend selector
platform capability
attention kernel call path
prefill kernel
decode kernel
varlen / paged attention support
```

---

## 4. 对比维度占位

```text
版本演进：
  v1 / v2 / v3 / v4 的算法和硬件适配差异。

vLLM 接入：
  backend class、selector 条件、kernel 调用路径。

能力边界：
  causal / sliding window / paged KV / GQA / MLA / FP8 KV cache。

和其他概念关系：
  FlashAttention ≠ PagedAttention；FlashAttention ≠ FlashInfer；FlashMLA 是 MLA 专用路径。
```
