# 09. Attention backend 如何感知并行？

源码位置：

- `vllm/vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/vllm/v1/attention/selector.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/`
- `vllm/vllm/v1/attention/ops/`

本问题关注：attention backend 如何受到 TP / CP / DCP / PP / DP 的影响，num_heads / num_kv_heads 如何按 TP 分布，GQA/MQA/MLA 在并行下如何执行，DCP 为什么要求 backend 返回 LSE，以及 FlashAttention / FlashInfer / Triton / FlashMLA 等 backend 对并行特性的支持边界。

---

## 1. 一句话回答

Attention 是并行体系中最敏感的执行点之一：

```text
TP 改变本 rank 的 head 数；
KV cache layout 决定 backend 如何读写 cache；
CP / DCP 改变 attention 的上下文范围和 merge 方式；
MLA / GQA / MQA 又会进一步改变 Q/K/V 组织。
```

---

## 2. 本文要回答的问题

```text
TP 下 num_heads / num_kv_heads 如何变化？
GQA / MQA 的 KV heads 如何分配到 TP rank？
attention metadata 中哪些字段和并行有关？
DCP 为什么需要 LSE？
哪些 backend 支持 DCP / PCP / batch invariance？
MLA backend 如何处理 prefill / decode 并行？
PP 下 attention layer 的 KV cache 归属如何处理？
DP 下 attention state 是否 replica-local？
```

---

## 3. 最小主链路占位

```text
ParallelConfig / model config
  → Attention 初始化时计算本 rank num_heads / num_kv_heads
  → get_attn_backend 选择 backend
  → ModelRunner 构造 CommonAttentionMetadata
  → backend MetadataBuilder 结合并行信息生成 metadata
  → Impl.do_kv_cache_update 写本 rank KV cache
  → Impl.forward 执行本 rank attention
  → 必要时跨 CP / DCP group merge partial states
```

---

## 4. 重点并行影响占位

```text
TP：
  影响 head 分片、QKV projection、本地 KV head 数。

PP：
  影响哪些 rank 有 attention layers 和 KV cache。

DP：
  每个 replica 有独立 attention batch / metadata。

CP / DCP：
  影响 seq_lens、local context length、partial attention merge。

EP：
  不直接影响 dense attention，但 MoE 模型中会和 attention 层交替出现。
```

---

## 5. 待梳理源码点

```text
Attention.__init__ 中 get_attn_backend 参数
model_config.get_num_attention_heads
model_config.get_num_kv_heads
AttentionMetadataBuilder 中 dcp_world_size / pcp_world_size
CommonAttentionMetadata 并行相关字段
FlashAttentionImpl._forward_with_dcp
FlashInfer DCP / TRTLLM 路径
MLAAttention forward_mha / forward_mqa
backend supports_batch_invariance / can_return_lse_for_decode
```

---

## 6. 和 attention_methods 的关系

```text
../attention/attention_methods/12_flash_attention_family.md：
  FlashAttention family 与 DCP / LSE / paged KV 的关系。

../attention/attention_methods/15_mla_attention.md：
  MLA prefill/decode 与并行 backend 的关系。

../attention/attention_methods/14_mha_mqa_gqa.md：
  MHA / MQA / GQA head 组织和 TP 的关系。
```
