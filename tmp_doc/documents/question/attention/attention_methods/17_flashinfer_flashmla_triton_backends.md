# 17. FlashInfer / FlashMLA / Triton backend 如何区分？

源码位置：

- `vllm/vllm/v1/attention/selector.py`
- `vllm/vllm/platforms/cuda.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/registry.py`
- `vllm/vllm/v1/attention/backends/flashinfer.py`
- `vllm/vllm/v1/attention/backends/triton_attn.py`
- `vllm/vllm/v1/attention/backends/triton_attn_diffkv.py`
- `vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py`
- `vllm/vllm/v1/attention/backends/mla/flashmla.py`
- `vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py`
- `vllm/vllm/v1/attention/backends/mla/triton_mla.py`
- `vllm/vllm/v1/attention/ops/flashmla.py`
- `vllm/vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/vllm/v1/attention/ops/triton_decode_attention.py`
- `vllm/vllm/v1/attention/ops/triton_prefill_attention.py`

本文关注：vLLM V1 attention backend 里 `FlashInfer`、`FlashMLA`、`Triton` 三类名字相近但定位不同的后端。它们都不是新的 attention 语义，而是把同一批 `CommonAttentionMetadata`、KV cache、Q/K/V 张量翻译给不同 kernel 执行的实现路径。

---

## 1. 本文要回答的问题

```text
1. FlashInfer backend 适合什么场景？
2. FlashMLA backend 和 MLA / DeepSeek 类模型是什么关系？
3. Triton backend 为什么经常是 fallback，同时又有专门优化路径？
4. selector 如何在 FlashInfer / FlashAttention / FlashMLA / Triton 之间排序？
5. 这些 backend 的 metadata、KV cache layout、kernel call path 有什么区别？
6. 普通 attention backend 和 MLA backend 为什么不能混用？
7. FlashInfer MLA、FlashMLA、Triton MLA 分别是什么？
```

---

## 2. 一句话回答

`FlashInfer / FlashMLA / Triton` 是三组 attention kernel 后端：

```text
FlashInfer：
  面向普通 paged attention 的高性能第三方 backend，内部又可走 FlashInfer native wrapper
  或 TRT-LLM / trtllm-gen kernel 路径。

FlashMLA：
  面向 MLA（Multi-head Latent Attention）的专用 backend，典型服务 DeepSeek 类 MLA 模型，
  处理的 KV cache 不是普通 K/V cache，而是 MLA 的 latent cache。

Triton：
  vLLM 自带的 Triton kernel 后端，覆盖面广，常作为 fallback；同时也提供普通 attention、
  DiffKV、MLA 等专门路径。
```

最重要的边界是：

```text
use_mla = False：
  在普通 attention backend 里选 FlashAttention / FlashInfer / Triton 等。

use_mla = True：
  在 MLA backend 里选 FlashInfer_MLA / FlashMLA / Triton_MLA / FlashMLA_Sparse 等。
```

也就是说，`FlashInferBackend` 和 `FlashMLABackend` 不只是“名字不同的加速库”，它们面对的 attention layer 形态就不同。

---

## 3. 整体心智模型

可以先把 vLLM attention 执行拆成三层：

```text
ModelRunner
  → 构造 CommonAttentionMetadata
  → 按 layer / KV cache group 取具体 AttentionMetadata
  → 模型层里的 Attention 调用 backend impl
  → backend impl 更新 KV cache 并执行 kernel
```

backend 的职责不是调度请求，而是回答：

```text
给定 query / key / value / kv_cache / metadata，
这一层 attention 用哪个 kernel 跑？
KV cache 应该是什么形状？
prefill 和 decode 怎么分派？
能不能支持 FP8 / NVFP4 / sliding window / sink / non-causal / cudagraph？
```

基础抽象在 `AttentionBackend`、`AttentionMetadataBuilder`、`AttentionImpl` / `MLAAttentionImpl` 中定义。

源码位置：`vllm/vllm/v1/attention/backend.py:55`

核心接口包括：

```text
AttentionBackend：
  - get_name()
  - get_impl_cls()
  - get_builder_cls()
  - get_kv_cache_shape()
  - get_kv_cache_stride_order()
  - supports_dtype()
  - supports_kv_cache_dtype()
  - supports_block_size()
  - supports_attn_type()
  - supports_compute_capability()
  - validate_configuration()

AttentionMetadataBuilder：
  - build(common_prefix_len, common_attn_metadata)
  - build_for_cudagraph_capture(...)
  - build_for_drafting(...)
  - update_block_table(...)
  - use_cascade_attention(...)

AttentionImpl / MLAAttentionImpl：
  - forward(...)         普通 attention
  - forward_mqa(...)     MLA decode / MQA-style path
  - forward_mha(...)     MLA prefill / MHA-style path
  - do_kv_cache_update(...)
```

---

## 4. backend selector 先看哪些信息

`get_attn_backend()` 是模型层选择 backend 的入口。

源码位置：`vllm/vllm/v1/attention/selector.py:54`

它会把当前 attention layer 的关键信息打包成 `AttentionSelectorConfig`：

```text
head_size
model dtype
kv_cache_dtype
block_size
use_mla
has_sink
use_sparse
use_mm_prefix
use_per_head_quant_scales
attn_type
use_non_causal
use_batch_invariant
use_kv_connector
```

然后交给当前平台：

```text
current_platform.get_attn_backend_cls(
  backend=用户显式指定的 backend 或 None,
  attn_selector_config=上述配置,
  num_heads=可选 head 数
)
```

源码位置：`vllm/vllm/v1/attention/selector.py:121`

这里有两个关键点：

### 4.1 用户显式指定 backend 时，只校验这个 backend

如果用户通过 attention config 指定了 backend，CUDA 平台会先只校验它。

源码位置：`vllm/vllm/platforms/cuda.py:381`

```text
指定 backend 合法：
  直接使用。

指定 backend 不合法：
  抛 ValueError，并说明 invalid_reasons。
```

它不会在“用户强指定但不合法”时默默换成别的 backend。

### 4.2 未显式指定时，按平台优先级自动选择第一个合法 backend

CUDA 平台会先取优先级列表，再对每个候选调用 `validate_configuration()`。

源码位置：`vllm/vllm/platforms/cuda.py:337`

校验失败原因可能包括：

```text
head_size not supported
dtype not supported
kv_cache_dtype not supported
block_size not supported
MLA not supported / non-MLA not supported
attention sinks not supported
sparse not supported
partial multimodal token full attention not supported
compute capability not supported
attention type xxx not supported
non-causal attention not supported
batch invariance not supported
KV connector not supported
```

最后选择优先级最高且合法的 backend。

源码位置：`vllm/vllm/platforms/cuda.py:426`

---

## 5. CUDA 上的默认优先级

优先级定义在 `_get_backend_priorities()`。

源码位置：`vllm/vllm/platforms/cuda.py:82`

### 5.1 普通 attention：`use_mla = False`

Blackwell / SM100 系列：

```text
FLASHINFER
  → FLASH_ATTN
  → TRITON_ATTN
  → FLEX_ATTENTION
  → TURBOQUANT
```

源码位置：`vllm/vllm/platforms/cuda.py:138`

非 SM100 CUDA：

```text
FLASH_ATTN
  → FLASHINFER
  → TRITON_ATTN
  → FLEX_ATTENTION
  → TURBOQUANT
```

源码位置：`vllm/vllm/platforms/cuda.py:146`

这说明：

```text
FlashInfer 不是所有 CUDA 上都第一优先级。
在 SM100 上它优先于 FlashAttention；在其他 CUDA 上通常排在 FlashAttention 后面、Triton 前面。
```

### 5.2 MLA attention：`use_mla = True`

SM100 上：

```text
FLASHINFER_MLA
  → TOKENSPEED_MLA
  → CUTLASS_MLA
  → FLASH_ATTN_MLA
  → FLASHMLA
  → TRITON_MLA
  → sparse MLA backends
```

源码位置：`vllm/vllm/platforms/cuda.py:92`

非 SM100 CUDA：

```text
FLASH_ATTN_MLA
  → FLASHMLA
  → FLASHINFER_MLA
  → TRITON_MLA
  → FLASHMLA_SPARSE
```

源码位置：`vllm/vllm/platforms/cuda.py:129`

这说明：

```text
FlashMLA 是 MLA 专用候选，不会参与普通 attention 的选择；
Triton_MLA 是 MLA fallback / 通用路径之一，也不会参与普通 attention 的选择。
```

### 5.3 sparse MLA 的额外排序

SM100 上 sparse MLA 还会根据 KV cache dtype 和 head 数微调：

```text
FP8 / quantized KV cache：
  FLASHINFER_MLA_SPARSE
    → FLASHMLA_SPARSE

BF16 KV cache 且 num_heads <= 16：
  FLASHINFER_MLA_SPARSE
    → FLASHMLA_SPARSE

BF16 KV cache 且 num_heads > 16：
  FLASHMLA_SPARSE
    → FLASHINFER_MLA_SPARSE
```

源码位置：`vllm/vllm/platforms/cuda.py:97`

这类路径主要服务 DeepSeek V3.2 / V4 这类 sparse MLA 模型，不是普通 Transformer attention 的 fallback。

---

## 6. registry 里这些名字分别指向什么

backend 枚举在 `AttentionBackendEnum` 中注册。

源码位置：`vllm/vllm/v1/attention/backends/registry.py:34`

和本文相关的条目可以分成三组：

```text
普通 attention：
  FLASHINFER
  TRITON_ATTN
  TRITON_ATTN_DIFFKV
  FLASH_ATTN
  FLASH_ATTN_DIFFKV

MLA attention：
  FLASHINFER_MLA
  FLASHMLA
  TRITON_MLA
  FLASH_ATTN_MLA
  CUTLASS_MLA
  TOKENSPEED_MLA

Sparse / DeepSeek 特化 MLA：
  FLASHINFER_MLA_SPARSE
  FLASHMLA_SPARSE
  FLASHMLA_SPARSE_DSV4
  FLASHINFER_MLA_SPARSE_DSV4
```

所以看到日志里的 backend 名字时，先判断它属于哪一组：

```text
FLASHINFER：普通 paged attention。
FLASHINFER_MLA：FlashInfer 提供的 MLA decode kernel 路径。
FLASHMLA：FlashMLA 项目的 dense MLA backend。
FLASHMLA_SPARSE：FlashMLA sparse MLA backend。
TRITON_ATTN：普通 Triton paged attention。
TRITON_MLA：Triton MLA backend。
TRITON_ATTN_DIFFKV：普通 attention 的 DiffKV 变体。
```

---

## 7. FlashInferBackend：普通 paged attention 高性能后端

源码位置：`vllm/vllm/v1/attention/backends/flashinfer.py:325`

### 7.1 它适合什么

`FlashInferBackend` 面向普通 decoder paged attention。它处理的是常规 Q/K/V + paged KV cache，而不是 MLA latent cache。

它的典型职责是：

```text
- 支持 prefill / decode 混合 batch；
- 把 vLLM block table 转成 FlashInfer paged_kv_indptr / paged_kv_indices / last_page_len；
- 根据场景选择 FlashInfer native wrapper 或 TRT-LLM / trtllm-gen path；
- 支持 FP8 / NVFP4 KV cache 等量化路径；
- 在部分平台上要求或偏好特定 KV cache layout。
```

### 7.2 支持范围

`FlashInferBackend` 的核心支持条件包括：

```text
dtype：
  float16 / bfloat16

KV cache dtype：
  auto / float16 / bfloat16 / fp8 / fp8_e4m3 / fp8_e5m2 / nvfp4

head_size：
  64 / 128 / 256 / 512

compute capability：
  SM75 到 SM121

block size：
  常规 16 / 32 / 64；
  在 trtllm-gen GQA/MQA 可用时可扩展到 128 / 256 / 512 / 1024。
```

源码位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:325`
- `vllm/vllm/v1/attention/backends/flashinfer.py:337`
- `vllm/vllm/v1/attention/backends/flashinfer.py:420`
- `vllm/vllm/v1/attention/backends/flashinfer.py:425`

### 7.3 KV cache shape 和 layout

普通 FlashInfer 的 KV cache 逻辑 shape 通常是：

```text
[num_blocks, 2, block_size, num_kv_heads, head_size]
```

如果是 NVFP4 KV cache，最后一维会变成 packed layout：

```text
[num_blocks, 2, block_size, num_kv_heads, nvfp4_full_dim]
```

源码位置：`vllm/vllm/v1/attention/backends/flashinfer.py:374`

FlashInfer 可以根据全局 KV cache layout 使用 NHD 或 HND stride order。SM100 上还会声明需要 `HND` layout。

源码位置：`vllm/vllm/v1/attention/backends/flashinfer.py:388`

```text
SM100：
  get_required_kv_cache_layout() → HND

其他平台：
  不强制，跟随当前 KV cache layout。
```

源码位置：`vllm/vllm/v1/attention/backends/flashinfer.py:447`

### 7.4 metadata builder 做了什么

`FlashInferMetadataBuilder` 把 `CommonAttentionMetadata` 翻译成 `FlashInferMetadata`。

源码位置：`vllm/vllm/v1/attention/backends/flashinfer.py:558`

`FlashInferMetadata` 的核心字段：

```text
num_actual_tokens
slot_mapping
q_data_type
num_decodes / num_decode_tokens
num_prefills / num_prefill_tokens
causal
prefill
decode
use_cascade
cascade_wrapper
```

源码位置：`vllm/vllm/v1/attention/backends/flashinfer.py:519`

其中 `prefill` 和 `decode` 不是简单 tensor，而是再分成两类：

```text
FlashInfer native path：
  FIPrefill(wrapper=BatchPrefillWithPagedKVCacheWrapper / BatchDCPPrefillWrapper)
  FIDecode(wrapper=BatchDecodeWithPagedKVCacheWrapper)

TRT-LLM / trtllm-gen path：
  TRTLLMPrefill(block_tables, seq_lens, cum_seq_lens_q, cum_seq_lens_kv, ...)
  TRTLLMDecode(block_tables, seq_lens, max_seq_len)
```

源码位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:457`
- `vllm/vllm/v1/attention/backends/flashinfer.py:471`
- `vllm/vllm/v1/attention/backends/flashinfer.py:499`

### 7.5 prefill / decode 如何分派

builder 会先用 `split_decodes_and_prefills()` 把 batch 切成 decode 段和 prefill 段。

源码位置：`vllm/vllm/v1/attention/backends/flashinfer.py:946`

然后决定：

```text
cascade attention？
  走 MultiLevelCascadeAttentionWrapper。

prefill 部分：
  如果满足 TRTLLM 条件，构造 TRTLLMPrefill；
  否则 plan FlashInfer native prefill wrapper。

decode 部分：
  如果满足 TRTLLM decode 条件，构造 TRTLLMDecode；
  否则 plan FlashInfer native decode wrapper。
```

源码位置：`vllm/vllm/v1/attention/backends/flashinfer.py:969`

### 7.6 forward 里真正调用什么

`FlashInferImpl.forward()` 会按 metadata 决定具体 kernel：

```text
prefill native：
  BatchPrefillWithPagedKVCacheWrapper.run(...)

prefill TRTLLM：
  trtllm_batch_context_with_kv_cache(...)

decode native：
  BatchDecodeWithPagedKVCacheWrapper.run(...)

decode TRTLLM：
  trtllm_batch_decode_with_kv_cache(...)

cascade：
  MultiLevelCascadeAttentionWrapper.run(...)
```

源码位置：`vllm/vllm/v1/attention/backends/flashinfer.py:1436`

这里还有两个容易忽略的点：

```text
1. forward 前会按 num_actual_tokens 去掉 CUDA graph padding。
2. KV cache update 不在 FlashInfer forward 内完成，而是由 do_kv_cache_update() 单独调用 reshape_and_cache_flash。
```

源码位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:454`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1903`

---

## 8. FlashInfer_MLA：FlashInfer 的 MLA 专用路径

源码位置：`vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:38`

`FLASHINFER_MLA` 不是 `FLASHINFER` 的普通 attention 变体，而是 MLA backend。

它的特点是：

```text
- use_mla = True 时才参与选择；
- 只支持 decoder attention；
- 主要走 FlashInfer / TRTLLM 的 MLA decode kernel；
- SM100 / Blackwell 上优先级很高；
- 要求 HND KV cache layout；
- qk_nope_head_dim 需要在 [64, 128, 192]。
```

源码位置：

- `vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:72`
- `vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:77`
- `vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:90`
- `vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:103`

`FlashInferMLAImpl.forward_mqa()` 会把 MLA 的 `q_nope / q_pe` 拼成 kernel 需要的 query，然后调用：

```text
trtllm_batch_decode_with_kv_cache_mla(...)
```

源码位置：`vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:164`

它不返回 LSE：

```text
return o, None
```

源码位置：`vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:216`

---

## 9. FlashMLABackend：FlashMLA dense MLA 后端

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla.py:47`

### 9.1 它和 MLA 模型的关系

`FlashMLABackend` 继承 `MLACommonBackend`，是 MLA attention 的实现之一。

它处理的不是普通 K/V cache，而是 MLA 的：

```text
kv_c_and_k_pe_cache
```

也就是把 latent KV 表示和 RoPE 相关 K_PE 表示合在一起的缓存。

所以它只会在 `use_mla=True` 的模型层里被 selector 考虑。

### 9.2 支持范围

```text
dtype：
  float16 / bfloat16

KV cache dtype：
  auto / float16 / bfloat16 / fp8 / fp8_e4m3

kernel block size：
  64

compute capability：
  SM90 / SM100

attention type：
  decoder only
```

源码位置：

- `vllm/vllm/v1/attention/backends/mla/flashmla.py:48`
- `vllm/vllm/v1/attention/backends/mla/flashmla.py:58`
- `vllm/vllm/v1/attention/backends/mla/flashmla.py:81`
- `vllm/vllm/v1/attention/backends/mla/flashmla.py:245`

它还会调用 FlashMLA ops 的支持检查：

```text
is_flashmla_dense_supported()
is_flashmla_sparse_supported()
```

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla.py:86`

### 9.3 metadata builder 做了什么

`FlashMLAMetadataBuilder` 基于 `MLACommonMetadataBuilder` 构造 MLA metadata。

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla.py:118`

它设置了两个重要约束：

```text
cudagraph support：
  UNIFORM_BATCH

query_len_support：
  UNIFORM
```

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla.py:119`

对 decode，它会调用 `get_mla_metadata()` 生成 FlashMLA kernel scheduler metadata：

```text
scheduler_metadata = get_mla_metadata(
  seq_lens_device,
  num_q_tokens_per_head_k,
  topk=1,
  is_fp8_kvcache=...
)
```

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla.py:161`

如果是 FP8 KV cache，还会生成额外的 tile scheduler metadata 和 num_splits，并在 full cudagraph 下复制到持久 buffer。

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla.py:181`

### 9.4 forward_mqa 调用什么 kernel

`FlashMLAImpl.forward_mqa()` 处理 decode / MQA-style 路径。

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla.py:263`

核心调用：

```text
非 FP8 KV cache：
  flash_mla_with_kvcache(...)

FP8 KV cache：
  flash_mla_with_kvcache_fp8(...)
```

源码位置：

- `vllm/vllm/v1/attention/backends/mla/flashmla.py:314`
- `vllm/vllm/v1/attention/backends/mla/flashmla.py:328`

它可以返回 decode LSE：

```text
can_return_lse_for_decode = True
```

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla.py:213`

这对 decode context parallelism 等需要 LSE 归并的场景很重要。

---

## 10. FlashMLA_Sparse：DeepSeek sparse MLA 特化路径

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:90`

`FLASHMLA_SPARSE` 是 sparse MLA backend，不是 dense FlashMLA 的普通 fallback。

它的定位：

```text
- 只在 use_mla=True 且 use_sparse=True 时合法；
- 典型服务 DeepSeek V3.2 / V4 这类 sparse MLA；
- 支持 BF16 和 DeepSeek MLA FP8 cache layout；
- 需要 topk_indices / indexer 参与；
- prefill 和 decode 的 FP8 cache 路径会根据 head 数走 mixed batch 或 separate prefill/decode。
```

支持范围：

```text
dtype：
  bfloat16

KV cache dtype：
  auto / bfloat16 / fp8_ds_mla / fp8(alias)

head_size：
  576

block size：
  64

compute capability：
  SM90 / SM100
```

源码位置：

- `vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:90`
- `vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:99`
- `vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:115`
- `vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:129`

FP8 sparse cache 的存储格式在源码注释里写得很明确：

```text
DeepSeek V3.2：
  512B quantized NoPE + 16B scales + 128B RoPE

DeepSeek V4：
  448B quantized NoPE + 128B RoPE + 8B scales/pad
```

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:65`

---

## 11. TritonAttentionBackend：覆盖面广的普通 attention 后端

源码位置：`vllm/vllm/v1/attention/backends/triton_attn.py:248`

### 11.1 它为什么经常是 fallback

`TritonAttentionBackend` 是 vLLM 自带 Triton kernel 实现，特点是：

```text
- 不依赖 FlashAttention / FlashInfer 外部 attention kernel 包；
- 支持的 attention type 更广；
- 支持普通 decoder、encoder、encoder_only、encoder_decoder；
- 支持 non-causal；
- 支持 multimodal prefix range；
- 支持 attention sink；
- 支持 batch invariant；
- 支持 FP32 dtype；
- compute capability 返回 True，平台约束相对少。
```

源码位置：

- `vllm/vllm/v1/attention/backends/triton_attn.py:248`
- `vllm/vllm/v1/attention/backends/triton_attn.py:277`
- `vllm/vllm/v1/attention/backends/triton_attn.py:285`
- `vllm/vllm/v1/attention/backends/triton_attn.py:346`
- `vllm/vllm/v1/attention/backends/triton_attn.py:351`
- `vllm/vllm/v1/attention/backends/triton_attn.py:354`
- `vllm/vllm/v1/attention/backends/triton_attn.py:359`
- `vllm/vllm/v1/attention/backends/triton_attn.py:372`

这就是它经常作为 fallback 的原因：当更快但限制更多的 backend 不合法时，Triton 往往还能覆盖。

### 11.2 支持范围

```text
dtype：
  float16 / bfloat16 / float32

KV cache dtype：
  auto / float16 / bfloat16 / fp8 / fp8_e4m3 / fp8_e5m2 /
  int8_per_token_head / fp8_per_token_head

block size：
  multiple of 16

head_size：
  >= 32

attention type：
  decoder / encoder / encoder_only / encoder_decoder
```

源码位置：

- `vllm/vllm/v1/attention/backends/triton_attn.py:249`
- `vllm/vllm/v1/attention/backends/triton_attn.py:254`
- `vllm/vllm/v1/attention/backends/triton_attn.py:265`
- `vllm/vllm/v1/attention/backends/triton_attn.py:346`
- `vllm/vllm/v1/attention/backends/triton_attn.py:359`

### 11.3 metadata 里有什么

`TritonAttentionMetadata` 相比 FlashInfer metadata 更接近 kernel 参数集合。

源码位置：`vllm/vllm/v1/attention/backends/triton_attn.py:58`

核心字段包括：

```text
num_actual_tokens
max_query_len
query_start_loc
max_seq_len
seq_lens
block_table
slot_mapping
seq_threshold_3D
num_par_softmax_segments
softmax_segm_output
softmax_segm_max
softmax_segm_expsum
causal
use_cascade
common_prefix_len
cu_prefix_query_lens
prefix_kv_lens
suffix_kv_lens
scheduler_metadata
prefix_scheduler_metadata
mm_prefix_range
mm_prefix_range_tensor
```

其中 `softmax_segm_*` 是 Triton unified attention 为分段 softmax / 3D kernel 路径准备的工作区。

### 11.4 普通 decoder forward 调用什么

普通 decoder 路径里，`TritonAttentionImpl.forward()` 会：

```text
1. 处理 FP8 / per-token-head quant KV cache 的 view 和 scale；
2. 取 query_start_loc / seq_lens / max_query_len / max_seq_len / block_table；
3. 调用 unified_attention(...)
```

源码位置：`vllm/vllm/v1/attention/backends/triton_attn.py:530`

核心 kernel 调用：

```text
unified_attention(
  q=query[:num_actual_tokens],
  k=key_cache,
  v=value_cache,
  out=output[:num_actual_tokens],
  cu_seqlens_q=query_start_loc,
  seqused_k=seq_lens,
  max_seqlen_q=max_query_len,
  max_seqlen_k=max_seq_len,
  block_table=block_table,
  ...
)
```

源码位置：`vllm/vllm/v1/attention/backends/triton_attn.py:642`

### 11.5 encoder attention 走另一条 Triton prefill kernel

如果是 encoder / encoder_only attention，它不使用 decoder KV cache，而是直接用 Q/K/V 调：

```text
context_attention_fwd(..., is_causal=False)
```

源码位置：`vllm/vllm/v1/attention/backends/triton_attn.py:678`

这也是 Triton backend 覆盖面比 FlashInferBackend 更广的原因之一。

### 11.6 KV cache update

Triton 普通 attention 的 `forward_includes_kv_cache_update = False`，KV cache 更新也单独做。

源码位置：`vllm/vllm/v1/attention/backends/triton_attn.py:275`

普通路径调用：

```text
triton_reshape_and_cache_flash(...)
```

per-token-head quant 路径调用：

```text
triton_reshape_and_cache_flash_per_token_head_quant(...)
```

源码位置：`vllm/vllm/v1/attention/backends/triton_attn.py:724`

---

## 12. TritonAttentionDiffKV：普通 attention 的 DiffKV 变体

源码位置：`vllm/vllm/v1/attention/backends/triton_attn_diffkv.py:73`

`TRITON_ATTN_DIFFKV` 解决的是：

```text
K 的 head dim 和 V 的 head dim 不同。
```

它的 KV cache layout 是把 K 和 V 沿最后一维 packed 到一起：

```text
[num_blocks, block_size, num_kv_heads, head_size_qk + head_size_v]
```

源码位置：`vllm/vllm/v1/attention/backends/triton_attn_diffkv.py:100`

和普通 Triton attention 的区别：

```text
- 只支持 decoder self-attention；
- 不支持 quantized KV cache；
- 不支持 per-token-head quant；
- 不支持 chunked attention lookback；
- fused rope + KV cache update 不支持；
- forward 调 unified_attention_diffkv(...)
```

源码位置：

- `vllm/vllm/v1/attention/backends/triton_attn_diffkv.py:141`
- `vllm/vllm/v1/attention/backends/triton_attn_diffkv.py:149`
- `vllm/vllm/v1/attention/backends/triton_attn_diffkv.py:195`

---

## 13. TritonMLABackend：MLA 的通用 Triton 路径

源码位置：`vllm/vllm/v1/attention/backends/mla/triton_mla.py:36`

`TRITON_MLA` 是 MLA 后端，不是普通 `TRITON_ATTN`。

它的定位更像：

```text
MLA 场景里覆盖面较广、平台约束少的 Triton fallback。
```

支持范围：

```text
dtype：
  float16 / bfloat16

KV cache dtype：
  auto / float16 / bfloat16 / fp8 / fp8_e4m3

block size：
  multiple of 16

compute capability：
  True，不额外限制

batch invariance：
  支持

attention type：
  decoder only
```

源码位置：

- `vllm/vllm/v1/attention/backends/mla/triton_mla.py:37`
- `vllm/vllm/v1/attention/backends/mla/triton_mla.py:51`
- `vllm/vllm/v1/attention/backends/mla/triton_mla.py:72`
- `vllm/vllm/v1/attention/backends/mla/triton_mla.py:84`
- `vllm/vllm/v1/attention/backends/mla/triton_mla.py:121`

`TritonMLAImpl.forward_mqa()` 会把 MLA query 拼接后调用通用 decode attention Triton kernel：

```text
decode_attention_fwd(..., is_mla=True)
```

源码位置：`vllm/vllm/v1/attention/backends/mla/triton_mla.py:144`

它也可以返回 LSE：

```text
can_return_lse_for_decode = True
```

源码位置：`vllm/vllm/v1/attention/backends/mla/triton_mla.py:89`

---

## 14. 三类 backend 的核心差异表

| 维度 | FlashInfer | FlashMLA | Triton |
|---|---|---|---|
| 主要对象 | 普通 paged attention | MLA / DeepSeek 类 attention | 普通 attention + MLA fallback / 特化 |
| selector 条件 | `use_mla=False` 的候选 | `use_mla=True` 的候选 | 普通是 `TRITON_ATTN`，MLA 是 `TRITON_MLA` |
| 典型优先级 | SM100 普通 attention 第一；其他 CUDA 在 FlashAttention 后 | 非 SM100 MLA 中靠前；SM100 中在 FlashInfer_MLA / CUTLASS / FlashAttn_MLA 后 | 通常靠后，但覆盖面广 |
| 主要 metadata | `FlashInferMetadata` | `MLACommonMetadata` / `FlashMLAMetadata` | `TritonAttentionMetadata` / `MLACommonMetadata` |
| prefill/decode 分派 | 显式拆 prefill / decode，可走 FI native 或 TRTLLM | MLA common + FlashMLA scheduler metadata | 普通走 unified attention，MLA 走 decode_attention_fwd |
| KV cache | 普通 K/V paged cache | MLA latent cache：`kv_c_and_k_pe_cache` | 普通 K/V cache 或 MLA latent cache，取决于 backend |
| block size | 常规 16/32/64，TRTLLM 大页可更多 | dense FlashMLA 固定 64 | 普通和 MLA 多为 multiple of 16 |
| dtype | fp16 / bf16 | fp16 / bf16；sparse 常 bf16 | fp16 / bf16 / fp32（普通 Triton） |
| quant KV | fp8 / nvfp4 等较丰富 | fp8 / fp8_ds_mla 等 MLA 特化 | fp8 / per-token-head quant 等普通 Triton 支持较多 |
| attention type | FlashInferImpl 只实现 decoder | decoder only | 普通 Triton 支持 decoder / encoder / encoder_only / encoder_decoder |
| non-causal | backend 声明支持，但路径有限制 | 不支持 | 普通 Triton 支持 |
| cudagraph | 取决于 TRTLLM decode 支持，可能是 uniform batch 或 single-token decode | dense MLA 是 uniform batch | 普通 Triton 是 always；MLA 是 uniform batch |
| 外部依赖 | flashinfer / TRTLLM kernels | FlashMLA ops / flashinfer MLA variants | vLLM Triton kernels |

---

## 15. metadata 差异：为什么不是一个统一对象

`CommonAttentionMetadata` 是所有 backend 的公共输入，但不是最终 kernel 参数。

源码位置：`vllm/vllm/v1/attention/backend.py:393`

公共字段包括：

```text
query_start_loc
query_start_loc_cpu
seq_lens
num_reqs
num_actual_tokens
max_query_len
max_seq_len
block_table_tensor
slot_mapping
causal
encoder_seq_lens
dcp_local_seq_lens
positions
is_prefilling
seq_lens_cpu_upper_bound
mm_req_doc_ranges
```

不同 backend 需要把它翻译成不同形态：

```text
FlashInfer：
  需要 paged_kv_indptr / paged_kv_indices / last_page_len，
  并决定 native wrapper 还是 TRTLLM path。

FlashMLA：
  需要 MLA scheduler metadata，尤其是 tile_scheduler_metadata / num_splits，
  并围绕 kv_c_and_k_pe_cache 组织 decode。

Triton：
  需要 query_start_loc / seq_lens / block_table / softmax segment buffers，
  并把这些参数直接传给 unified_attention / decode_attention_fwd。
```

所以 `CommonAttentionMetadata` 是“公共 batch 描述”，`FlashInferMetadata / FlashMLAMetadata / TritonAttentionMetadata` 才是“kernel 可消费描述”。

---

## 16. KV cache layout 差异

### 16.1 普通 FlashInfer / Triton

普通 paged attention 的常见逻辑 shape 是：

```text
[num_blocks, 2, block_size, num_kv_heads, head_size]
```

其中 `2` 表示 K / V。

FlashInfer 和 Triton 都支持 NHD / HND layout 的 stride order，但具体要求不同：

```text
FlashInfer：
  SM100 上要求 HND；其他平台通常跟随全局 layout。

Triton：
  根据 get_kv_cache_layout() 返回 NHD 或 HND 的 stride order。
```

源码位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:388`
- `vllm/vllm/v1/attention/backends/triton_attn.py:318`

### 16.2 MLA dense

MLA dense backend 的 cache shape 通常少一层 K/V split，因为缓存的是：

```text
kv_c_and_k_pe_cache
```

FlashMLA / FlashInfer_MLA / Triton_MLA 都通过 MLA common 抽象处理这类 cache。

它们的 stride order 都是类似：

```text
without layer dimension：
  (0, 1, 2)

with layer dimension：
  (1, 0, 2, 3)
```

源码位置：

- `vllm/vllm/v1/attention/backends/mla/flashmla.py:62`
- `vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:53`
- `vllm/vllm/v1/attention/backends/mla/triton_mla.py:60`

### 16.3 Sparse MLA

Sparse MLA 的 cache 可能是普通 head_size，也可能是 DeepSeek 自定义 FP8 packed layout。

例如 FlashMLA sparse 对 `fp8_ds_mla` 返回：

```text
[num_blocks, block_size, 656]
```

源码位置：`vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:132`

这已经不是普通 `[num_blocks, 2, block_size, num_heads, head_size]` 的 K/V cache 结构。

---

## 17. prefill / decode 的处理差异

### 17.1 FlashInfer：显式拆分并分别 plan

FlashInfer 的 builder 会把 batch 拆成：

```text
decodes at front
prefills at back
```

然后分别构造 prefill 和 decode metadata。

好处是：

```text
- decode 可以走更快的 decode wrapper / TRTLLM decode；
- prefill 可以走 paged prefill wrapper / TRTLLM context；
- 混合 batch 可以在一个 backend 内部分派不同 kernel。
```

### 17.2 Triton：统一 kernel 接口更明显

普通 Triton attention 主要把参数传给 `unified_attention()`，prefill / decode 的差异更多体现在：

```text
query_start_loc
seq_lens
max_query_len
max_seq_len
block_table
causal / non-causal
```

也就是 kernel 内部根据这些 metadata 处理不同形态。

### 17.3 FlashMLA：MLA common 先规范化，再调用 dense / sparse MLA kernel

FlashMLA 关注的是 MLA decode / MQA-style 路径：

```text
q_nope / q_pe
kv_c_and_k_pe_cache
seq_lens
block_table
scheduler_metadata
```

dense FlashMLA 调 `flash_mla_with_kvcache`；sparse FlashMLA 还要引入 topk indices，把每个 token 的稀疏访问映射成全局 cache slot 或 prefill workspace offset。

---

## 18. CUDA graph 支持差异

`AttentionCGSupport` 定义了 backend 对 CUDA graph 的支持等级。

源码位置：`vllm/vllm/v1/attention/backend.py:548`

```text
ALWAYS：
  支持混合 prefill/decode 等更通用形状。

UNIFORM_BATCH：
  batch 内 query length 需要统一，适合 decode / spec-decode 等。

UNIFORM_SINGLE_TOKEN_DECODE：
  只支持 query_len == 1 的 decode。

NEVER：
  不支持。
```

相关 backend：

```text
TritonAttentionMetadataBuilder：
  ALWAYS

FlashInferMetadataBuilder：
  如果所有 KV spec 都支持 TRTLLM attention，返回 UNIFORM_BATCH；
  否则返回 UNIFORM_SINGLE_TOKEN_DECODE。

FlashMLAMetadataBuilder：
  UNIFORM_BATCH

TritonMLAMetadataBuilder：
  UNIFORM_BATCH

FlashMLASparseMetadataBuilder：
  UNIFORM_BATCH
```

源码位置：

- `vllm/vllm/v1/attention/backends/triton_attn.py:98`
- `vllm/vllm/v1/attention/backends/flashinfer.py:738`
- `vllm/vllm/v1/attention/backends/mla/flashmla.py:118`
- `vllm/vllm/v1/attention/backends/mla/triton_mla.py:32`
- `vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:231`

---

## 19. 常见选择场景

### 19.1 普通 Llama / Qwen 类 decoder-only 模型

通常是普通 attention：

```text
use_mla = False
use_sparse = False
```

CUDA 上候选大致是：

```text
FlashAttention / FlashInfer / TritonAttention
```

SM100 上更可能优先 FlashInfer；其他 CUDA 上更可能先 FlashAttention，FlashInfer 次之，Triton 兜底。

### 19.2 DeepSeek MLA dense 模型

通常是：

```text
use_mla = True
use_sparse = False
```

候选大致是：

```text
FlashInfer_MLA
FlashAttention_MLA
FlashMLA
Triton_MLA
Cutlass_MLA / Tokenspeed_MLA
```

能不能选上取决于：

```text
- GPU compute capability；
- qk_nope_head_dim；
- block size；
- KV cache dtype；
- 是否有 unsupported feature：alibi / sliding_window / logits_soft_cap；
- backend 包或 kernel 是否可 import。
```

### 19.3 DeepSeek sparse MLA 模型

通常是：

```text
use_mla = True
use_sparse = True
```

候选会进入：

```text
FlashInfer_MLA_Sparse
FlashMLA_Sparse
平台 / 模型特化 sparse backend
```

这类模型还会额外依赖 top-k indexer、稀疏索引映射、DeepSeek FP8 cache layout。

### 19.4 encoder / encoder-only / encoder-decoder attention

普通 `FlashInferImpl` 和 MLA backend 都只实现 decoder self-attention。

如果是 encoder attention，Triton 普通 backend 更有机会覆盖，因为它声明支持：

```text
DECODER
ENCODER
ENCODER_ONLY
ENCODER_DECODER
```

并且 encoder path 直接走 `context_attention_fwd(..., is_causal=False)`。

源码位置：`vllm/vllm/v1/attention/backends/triton_attn.py:359`

---

## 20. 最容易混淆的几个点

### 20.1 FlashInfer 和 FlashInfer_MLA 不是同一个 backend

```text
FLASHINFER：
  普通 paged attention backend。

FLASHINFER_MLA：
  MLA backend，调用 FlashInfer / TRTLLM MLA decode kernel。
```

它们在 registry、selector 分组、KV cache 形态上都不同。

### 20.2 FlashMLA 不是 FlashAttention 的 MLA 版本

FlashMLA 是 MLA 专用 kernel 路径；FlashAttention_MLA 是另一类 MLA backend。

在 CUDA priority 里它们是不同候选：

```text
FLASH_ATTN_MLA
FLASHMLA
FLASHINFER_MLA
TRITON_MLA
```

### 20.3 Triton 不只是“慢 fallback”

Triton 经常是 fallback，但它也有很多特化能力：

```text
- 普通 paged attention unified kernel；
- encoder attention path；
- DiffKV path；
- MLA decode path；
- per-token-head quant KV cache；
- fused RoPE + KV cache update 在部分 ROCm aiter 场景中可用。
```

所以它的定位更准确地说是：

```text
覆盖面广、可维护性强、可作为 fallback，同时也承载特化 kernel 的 vLLM 内建 backend。
```

### 20.4 block size 不只是 cache_config 参数

selector 会用 backend 的 `supports_block_size()` 校验 block size。

例如：

```text
FlashMLA dense：
  kernel block size 64。

Triton普通 / Triton_MLA：
  block size 必须是 16 的倍数。

FlashInfer：
  常规 16 / 32 / 64；特定 trtllm-gen GQA/MQA 场景可更大。
```

如果用户强制 `--block-size`，可能会排除更高优先级 backend。CUDA 平台会在这种情况下打印 warning。

源码位置：`vllm/vllm/platforms/cuda.py:436`

### 20.5 `forward_includes_kv_cache_update` 会影响 KV cache 更新位置

FlashInfer 和 Triton 普通 backend 都声明：

```text
forward_includes_kv_cache_update = False
```

因此 attention layer 会在 forward 之外调用 backend 的 KV cache update 方法。

源码位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:454`
- `vllm/vllm/v1/attention/backends/triton_attn.py:275`

这会影响你追踪 kernel 调用时看到的顺序：

```text
先 reshape/cache 当前 token 的 K/V，
再执行 attention kernel，
或者由 layer 逻辑按 backend 能力拆开执行。
```

---

## 21. 调试 backend 选择时看哪里

如果想判断为什么选了某个 backend，可以按这个顺序看：

```text
1. 模型 attention layer 调 get_attn_backend() 时传了什么：
   - head_size
   - dtype
   - kv_cache_dtype
   - use_mla
   - use_sparse
   - attn_type
   - has_sink

2. 当前平台 priority list：
   - CUDA：platforms/cuda.py::_get_backend_priorities
   - ROCm / XPU / CPU：各自 platforms 文件

3. 每个候选 backend 的 validate_configuration() 为什么失败：
   - supports_dtype
   - supports_kv_cache_dtype
   - supports_block_size
   - supports_compute_capability
   - supports_attn_type
   - supports_combination

4. 是否用户显式指定 backend：
   - 显式指定不合法会直接报错，不会自动 fallback。

5. 是否 backend 要求特定 KV cache layout：
   - get_required_kv_cache_layout()
```

对应源码：

```text
selector.py:get_attn_backend()
  → current_platform.get_attn_backend_cls()
  → cuda.py:get_valid_backends()
  → AttentionBackend.validate_configuration()
  → backend.get_required_kv_cache_layout()
```

---

## 22. 最终可以记成一张表

| backend 名 | 是否 MLA | 是否 sparse | 主要文件 | 一句话定位 |
|---|---:|---:|---|---|
| `FLASHINFER` | 否 | 否 | `backends/flashinfer.py` | 普通 paged attention 的 FlashInfer / TRTLLM 高性能路径 |
| `FLASHINFER_MLA` | 是 | 否 | `backends/mla/flashinfer_mla.py` | FlashInfer / TRTLLM MLA decode 路径，SM100 优先 |
| `FLASHINFER_MLA_SPARSE` | 是 | 是 | `backends/mla/flashinfer_mla_sparse.py` | FlashInfer sparse MLA 路径 |
| `FLASHMLA` | 是 | 否 | `backends/mla/flashmla.py` | FlashMLA dense MLA 路径，SM90/SM100 |
| `FLASHMLA_SPARSE` | 是 | 是 | `backends/mla/flashmla_sparse.py` | DeepSeek sparse MLA / FP8 cache 特化路径 |
| `TRITON_ATTN` | 否 | 否 | `backends/triton_attn.py` | vLLM 内建普通 Triton paged attention，覆盖面广 |
| `TRITON_ATTN_DIFFKV` | 否 | 否 | `backends/triton_attn_diffkv.py` | K/V head dim 不同的普通 attention 特化 |
| `TRITON_MLA` | 是 | 否 | `backends/mla/triton_mla.py` | MLA 的 Triton fallback / 通用 decode 路径 |

---

## 23. 一句话总结

`FlashInfer`、`FlashMLA`、`Triton` 的区别不是“哪个名字更快”，而是它们各自承接不同的 attention 形态和 kernel 约束：

```text
普通 paged attention：
  FlashAttention / FlashInfer / TritonAttention 之间选择。

MLA dense attention：
  FlashInfer_MLA / FlashAttention_MLA / FlashMLA / Triton_MLA 等之间选择。

Sparse MLA attention：
  FlashInfer_MLA_Sparse / FlashMLA_Sparse / 模型特化 backend 之间选择。
```

最终选择由：

```text
use_mla / use_sparse
  → CUDA 平台优先级
  → backend.validate_configuration()
  → required KV cache layout
  → metadata builder + impl kernel path
```

共同决定。

如果只记一句话：

```text
FlashInfer 偏普通 paged attention 和 SM100 TRTLLM 路径，
FlashMLA 偏 DeepSeek/MLA 专用 latent attention，
Triton 偏覆盖面广的内建实现和 fallback / 特化补位。
```
