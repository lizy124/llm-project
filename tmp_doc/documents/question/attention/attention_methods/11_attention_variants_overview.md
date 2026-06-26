# 11. Attention variants 总览：各种 attention 名词如何分类？

源码位置：

- `code/vllm/vllm/v1/attention/`
- `code/vllm/vllm/model_executor/layers/attention/`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/distributed/kv_transfer/`

本文用于给 FlashAttention、PagedAttention、MHA / MQA / GQA、MLA、Sliding Window、FlashInfer、FlashMLA、Triton、HMA 等名词建立分类框架，避免把模型结构、KV cache 管理、kernel backend 和调度优化混在一起。

---

## 0. 梳理规划

本篇是 `attention_methods` 的总览文档，后续按“先分类，再逐类讲清代表概念”的方式组织。

要回答的问题分成 8 组：

```text
1. 各种 attention 名词分别属于哪一层概念？
2. FlashAttention 和 PagedAttention 是一类东西吗？
3. MHA / MQA / GQA 和 backend 有什么关系？
4. MLA 是模型结构、KV layout，还是 backend？
5. Sliding window / local attention 属于算法语义还是 backend？
6. FlashInfer / FlashMLA / Triton 和 FlashAttention family 如何区分？
7. HMA 算 attention 类型吗？它如何影响 attention？
8. vLLM 中这些概念如何落到源码路径？
```

阅读顺序建议：

```text
11_attention_variants_overview.md
  → 12_flash_attention_family.md
  → 13_paged_attention.md
  → 14_mha_mqa_gqa.md
  → 15_mla_attention.md
  → 16_sliding_window_and_local_attention.md
  → 17_flashinfer_flashmla_triton_backends.md
  → 18_hma_and_kv_cache_layout.md
```

---

## 1. 分类框架占位

```text
模型结构类：
  MHA / MQA / GQA / MLA

KV cache 管理类：
  PagedAttention / block table / slot mapping / HMA / KV cache groups

kernel backend 类：
  FlashAttention / FlashInfer / FlashMLA / Triton / torch SDPA

调度优化类：
  cascade attention / chunked prefill / prefix cache / spec decode attention metadata

mask 语义类：
  causal attention / sliding window / local attention

分布式和内存协作类：
  KV connector / disaggregated prefill-decode / cross-layer KV cache / HMA
```

---

## 2. 核心结论占位

占位：后续补充一句话结论。

```text
“attention 类型”不是单一维度：有的是模型结构，有的是 KV cache 访问方式，有的是底层 kernel，有的是调度优化；阅读 vLLM attention 代码时必须先判断它属于哪一层。
```

---

## 3. 后续专题占位

```text
12_flash_attention_family.md：
  梳理 FlashAttention v1-v4 和 vLLM backend 的关系。

13_paged_attention.md：
  梳理 vLLM paged KV cache 与 attention 访问方式。

14_mha_mqa_gqa.md：
  梳理 head 结构如何影响 KV cache shape 和 backend 参数。

15_mla_attention.md：
  梳理 DeepSeek 类 MLA 的 latent KV 表示和 FlashMLA 路径。

16_sliding_window_and_local_attention.md：
  梳理局部可见窗口和 mask 语义。

17_flashinfer_flashmla_triton_backends.md：
  对比 vLLM 常见 attention backend。

18_hma_and_kv_cache_layout.md：
  梳理 HMA / hybrid KV cache groups 与 attention layout 的关系。
```
