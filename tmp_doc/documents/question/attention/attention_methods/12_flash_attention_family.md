# 12. FlashAttention family：v1 / v2 / v3 / v4 与 vLLM 的关系

源码位置：

- `vllm/vllm/v1/attention/selector.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/registry.py`
- `vllm/vllm/v1/attention/backends/fa_utils.py`
- `vllm/vllm/v1/attention/backends/flash_attn.py`
- `vllm/vllm/v1/attention/backends/flash_attn_diffkv.py`
- `vllm/vllm/v1/attention/backends/flashinfer.py`
- `vllm/vllm/v1/attention/backends/triton_attn.py`
- `vllm/vllm/v1/attention/backends/mla/flashattn_mla.py`
- `vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py`
- `vllm/vllm/v1/attention/backends/mla/flashmla.py`
- `vllm/vllm/platforms/cuda.py`
- `vllm/vllm/platforms/rocm.py`
- `vllm/vllm/vllm_flash_attn/flash_attn_interface.py`

本文用于梳理 FlashAttention family 的版本脉络、vLLM 中的 backend 接入方式、与 PagedAttention / FlashInfer / FlashMLA / Triton / ROCm AITER 的关系。这里的 FlashAttention 是 attention kernel / backend 家族，不等同于 PagedAttention，也不等同于 MHA / GQA / MLA 这类模型结构。

---

## 1. 一句话回答

FlashAttention family 解决的是 attention 计算中的 IO、显存访问、并行调度和硬件利用率问题；vLLM 把它接成一种 `AttentionBackend`，由 `ModelRunner` 准备 `block_table / slot_mapping / seq_lens / query_start_loc` 等 metadata，再由具体 backend 在 paged KV cache 上执行 prefill / decode attention。

主链路可以概括为：

```text
ModelRunner
  → 构造 CommonAttentionMetadata
  → AttentionMetadataBuilder.build()
  → Attention.forward()
  → unified_kv_cache_update / unified_attention_with_output
  → backend Impl.do_kv_cache_update()
  → backend Impl.forward()
  → flash_attn_varlen_func / FlashInfer wrapper / Triton op / FlashMLA kernel
```

所以：

```text
FlashAttention 是 kernel 算法家族；
PagedAttention 是 vLLM 的 KV cache 分页组织方式；
AttentionBackend 是 vLLM 把不同 kernel 接入同一 forward 链路的抽象；
FlashInfer / FlashMLA / Triton attention 是和 FlashAttention 并列或互补的 backend / kernel 路径。
```

---

## 2. 本文要回答的问题

```text
FlashAttention v1 / v2 / v3 / v4 分别解决什么问题？
vLLM 中哪些 backend 使用 FlashAttention 类 kernel？
FlashAttention 和 PagedAttention 如何配合？
FlashAttention 和 FlashInfer / FlashMLA / Triton attention 是什么关系？
不同硬件平台如何影响 FlashAttention backend 选择？
FlashAttention family 对 prefill / decode / paged KV cache / MLA / FP8 的支持边界是什么？
```

---

## 3. 先区分几个容易混淆的概念

### 3.1 FlashAttention 不是 PagedAttention

FlashAttention 关注的是：

```text
给定 Q / K / V，如何更高效地算 softmax(QK^T)V。
```

PagedAttention 关注的是：

```text
KV cache 如何按 block/page 存储、复用、映射和回收。
```

在 vLLM 中二者会一起出现：

```text
block_table / slot_mapping 描述 KV cache 的分页位置；
FlashAttention kernel 根据 block_table 到 paged KV cache 中读 K/V；
reshape_and_cache_flash 根据 slot_mapping 把新 K/V 写入 paged KV cache。
```

也就是说，PagedAttention 提供 KV cache 的地址组织，FlashAttention 提供具体 attention 计算 kernel。

### 3.2 FlashAttention 不是 MHA / GQA / MQA / MLA

MHA / GQA / MQA / MLA 是 attention 结构或模型参数组织方式。

FlashAttention 是计算这些结构时可用的 kernel 路径。

例如：

```text
MHA/GQA/MQA:
  FlashAttentionBackend 可以支持。

MLA:
  普通 FlashAttentionBackend 不直接等同于 MLA backend；
  vLLM 有专门的 FLASH_ATTN_MLA / FLASHINFER_MLA / FLASHMLA / TRITON_MLA。
```

### 3.3 FlashAttention backend 不是唯一的高性能 attention backend

vLLM V1 里 attention backend 由 registry 管理。常见 backend 包括：

```text
普通 attention:
  FLASH_ATTN
  FLASH_ATTN_DIFFKV
  FLASHINFER
  TRITON_ATTN
  TRITON_ATTN_DIFFKV
  ROCM_ATTN
  ROCM_AITER_FA
  ROCM_AITER_UNIFIED_ATTN
  FLEX_ATTENTION
  TURBOQUANT

MLA attention:
  FLASH_ATTN_MLA
  FLASHINFER_MLA
  FLASHMLA
  FLASHMLA_SPARSE
  FLASHINFER_MLA_SPARSE
  TRITON_MLA
  CUTLASS_MLA
  TOKENSPEED_MLA
  ROCM_AITER_MLA
```

对应枚举在 `vllm/v1/attention/backends/registry.py:34` 开始。

---

## 4. FlashAttention v1 / v2 / v3 / v4 的版本脉络

### 4.1 v1：IO-aware attention 的起点

FlashAttention v1 的核心思想是 IO-aware tiling：

```text
不要显式 materialize 完整 attention matrix；
按 tile 读取 Q/K/V；
在 SRAM/register 层面做 online softmax；
边算边归约，减少 HBM 读写。
```

它主要解决：

```text
标准 attention 的 O(seq_len^2) attention score 矩阵带来的巨大显存读写；
softmax 中间结果需要落显存的问题；
长序列 prefill 时 memory bandwidth 成为瓶颈的问题。
```

对 vLLM 来说，v1 更多是算法背景。当前 vLLM 的实际 backend 不会选择“FA1 backend”，而是在 FA2/FA3/FA4 或其他 backend 中选择。

### 4.2 v2：更好的 work partition 和通用 CUDA 路径

FlashAttention v2 继续沿用 IO-aware 思路，但改进了并行切分和 GPU 利用率。

它主要解决：

```text
v1 在不同 head size、batch、sequence shape 下并行度不够稳定的问题；
前向/反向 kernel 的线程块划分和 warp 级 work partition；
更好支持 MQA / GQA / varlen attention。
```

在 vLLM 中，FA2 是通用 fallback 路径之一：

```text
非 Hopper / Blackwell，或者 FA3/FA4 不适配时，通常回退到 FA2；
ALiBi 会让 FA3/FA4 回退到 FA2；
batch invariant 模式下 FA4 会回退到 FA2。
```

选择逻辑见 `vllm/v1/attention/backends/fa_utils.py:56`。

### 4.3 v3：面向 Hopper 的 FA 路径

FlashAttention v3 面向 Hopper SM90 做了更深的硬件适配。

它在 vLLM 中的关键特征是：

```text
只支持 device capability 9.x；
支持 AOT scheduler metadata；
支持 paged KV cache + varlen func；
支持部分 FP8 KV cache 场景；
FLASH_ATTN_MLA 依赖 FA3 + SM90。
```

vLLM 对 FA3 的支持判断在 `vllm/vllm_flash_attn/flash_attn_interface.py:62`：

```text
FA3 extension 可导入；
当前设备是 compute capability 9.x。
```

FA3 的 scheduler metadata 入口是 `get_scheduler_metadata()`，在 `vllm/vllm_flash_attn/flash_attn_interface.py:121`。`FlashAttentionMetadataBuilder` 在 full cuda graph + FA3 时会预分配 scheduler metadata buffer，见 `vllm/v1/attention/backends/flash_attn.py:372`。

### 4.4 v4：CuTe / Blackwell 相关的新路径

vLLM 中 FA4 通过 `vllm.vllm_flash_attn.cute.interface` 判断是否可用，支持的设备族包括 9.x、10.x、11.x，见 `vllm/vllm_flash_attn/flash_attn_interface.py:72`。

在 vLLM 当前实现里，FA4 主要带来这些能力或约束：

```text
Blackwell SM100 优先尝试 FA4；
mm_prefix / PrefixLM bidirectional multimodal ranges 需要 FA4；
per-sequence causal / dynamic_causal 需要 FA4；
SM90 上 head_size > 256 的部分场景可从 FA3 升级到 FA4；
SM100 上 FA4 受 TMEM 容量限制，head_size > 128 且不是 192 时会回退到 FA2；
batch invariant 模式下 FA4 会回退到 FA2。
```

这些逻辑集中在 `vllm/v1/attention/backends/fa_utils.py:118` 到 `vllm/v1/attention/backends/fa_utils.py:174`。

---

## 5. vLLM 如何选择 attention backend

### 5.1 Attention 层初始化时选择 backend

普通 attention 层在初始化时会调用 `get_attn_backend()`：

```text
Attention.__init__
  → get_attn_backend(
      head_size,
      dtype,
      kv_cache_dtype,
      use_mla=False,
      has_sink=...,
      use_mm_prefix=...,
      attn_type=...
    )
  → backend.get_impl_cls()
  → self.impl = backend impl
```

对应位置：`vllm/model_executor/layers/attention/attention.py:349` 到 `vllm/model_executor/layers/attention/attention.py:431`。

### 5.2 selector 负责把模型条件整理成 AttentionSelectorConfig

`get_attn_backend()` 会把选择条件整理成：

```text
head_size
dtype
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

然后调用平台侧的：

```text
current_platform.get_attn_backend_cls(...)
```

对应位置：`vllm/v1/attention/selector.py:54` 到 `vllm/v1/attention/selector.py:144`。

### 5.3 backend 的通用合法性检查

所有 backend 都要经过 `AttentionBackend.validate_configuration()`，通用检查包括：

```text
head_size 是否支持；
dtype 是否支持；
kv_cache_dtype 是否支持；
block_size 是否支持；
是否支持 MLA / sparse / sink / mm_prefix；
是否支持 per-head quant scales；
是否支持当前 compute capability；
是否支持 decoder / encoder / cross attention；
是否支持 non-causal；
是否支持 batch invariance；
是否支持 KV connector。
```

对应位置：`vllm/v1/attention/backend.py:308` 到 `vllm/v1/attention/backend.py:375`。

### 5.4 CUDA 平台的默认优先级

CUDA 平台根据 `use_mla` 和 `device_capability` 给出默认 backend 优先级。

普通非 MLA attention：

```text
Blackwell / SM100:
  FLASHINFER
  FLASH_ATTN
  TRITON_ATTN
  FLEX_ATTENTION
  TURBOQUANT

其他 CUDA:
  FLASH_ATTN
  FLASHINFER
  TRITON_ATTN
  FLEX_ATTENTION
  TURBOQUANT
```

对应位置：`vllm/platforms/cuda.py:137` 到 `vllm/platforms/cuda.py:153`。

MLA attention：

```text
Blackwell / SM100:
  FLASHINFER_MLA
  TOKENSPEED_MLA
  CUTLASS_MLA
  FLASH_ATTN_MLA
  FLASHMLA
  TRITON_MLA
  FLASHINFER_MLA_SPARSE / FLASHMLA_SPARSE

其他 CUDA:
  FLASH_ATTN_MLA
  FLASHMLA
  FLASHINFER_MLA
  TRITON_MLA
  FLASHMLA_SPARSE
```

SM100 sparse MLA 的 `FLASHINFER_MLA_SPARSE` / `FLASHMLA_SPARSE` 顺序不是固定的：量化 KV cache 或 BF16 且本地 head 数较少时优先 FlashInfer sparse；BF16 且 head 数较多时优先 FlashMLA sparse。

对应位置：`vllm/platforms/cuda.py:92` 到 `vllm/platforms/cuda.py:136`。

### 5.5 显式配置可以覆盖自动选择

配置项在 `vllm/config/attention.py`：

```text
backend:
  显式指定 attention backend；auto / None 表示自动选择。

flash_attn_version:
  强制使用 FA2 / FA3 / FA4，仅对 flash-attention backend 有意义。

use_trtllm_attention:
  控制 FlashInfer 内部是否使用 TRTLLM attention。

mla_prefill_backend:
  控制 MLA prefill 使用 FLASH_ATTN / FLASHINFER / TRTLLM_RAGGED 等路径。
```

---

## 6. vLLM 内置的 FlashAttention wrapper

vLLM 没有直接把上游 flash-attn 的全部 Python API 暴露出来，而是在 `vllm/vllm_flash_attn/flash_attn_interface.py` 里维护自己需要的 wrapper。

关键点：

```text
FA2 extension: _vllm_fa2_C
FA3 extension: _vllm_fa3_C
FA4 availability: vllm.vllm_flash_attn.cute.interface
主要 wrapper: flash_attn_varlen_func()
FA3 scheduler: get_scheduler_metadata()
```

代码注释说明 vLLM 主要关心 `flash_attn_varlen_func` 和 KV cache 相关路径，见 `vllm/vllm_flash_attn/flash_attn_interface.py:111`。当前 V1 ordinary FlashAttention 主链路实际调用的是 `flash_attn_varlen_func()`；注释中提到的 `flash_attn_with_kvcache` 更像历史/扩展说明，不应作为本文主链路。

`flash_attn_varlen_func()` 接收的核心参数包括：

```text
q / k / v
cu_seqlens_q / cu_seqlens_k
max_seqlen_q / max_seqlen_k
seqused_k
block_table
causal
window_size
softcap
alibi_slopes
scheduler_metadata
q_descale / k_descale / v_descale
fa_version
num_splits
dynamic_causal
mask_mod / aux_tensors
```

这说明 vLLM 的 FlashAttention 路径不是只服务“连续 KV”，也可以通过 `block_table` 读取 paged KV cache。

---

## 7. FLASH_ATTN backend 如何工作

### 7.1 Backend 能力声明

`FlashAttentionBackend` 位于 `vllm/v1/attention/backends/flash_attn.py:71`。

它声明的基础能力是：

```text
supported_dtypes:
  fp16 / bf16

supported_kv_cache_dtypes:
  auto / float16 / bfloat16
  fp8 / fp8_e4m3 在 CUDA 上需要特定平台和 FA3 + SM90；XPU 路径另行支持

block_size:
  MultipleOf(16)

head_size:
  必须 8 对齐；
  <= 256 默认支持；
  FA4 可支持到 <= 512。

compute capability:
  >= 8.0

attn_type:
  decoder / encoder / encoder_only / encoder_decoder
```

KV cache shape 是：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

对应位置：`vllm/v1/attention/backends/flash_attn.py:127` 到 `vllm/v1/attention/backends/flash_attn.py:137`。

### 7.2 Metadata 的作用

`FlashAttentionMetadata` 位于 `vllm/v1/attention/backends/flash_attn.py:228`。

它包含：

```text
num_actual_tokens
max_query_len
query_start_loc
max_seq_len
seq_lens
block_table
slot_mapping
use_cascade
common_prefix_len
scheduler_metadata
causal
mm_prefix_range_tensor
```

其中：

```text
query_start_loc:
  每个请求在 query tensor 中的起始偏移。

seq_lens:
  每个请求当前总序列长度。

block_table:
  每个请求的 KV block 映射。

slot_mapping:
  新 token 写入 KV cache 的扁平 slot 位置。

scheduler_metadata:
  FA3 AOT scheduler 使用。

mm_prefix_range_tensor:
  FA4 mm_prefix / PrefixLM multimodal bidirectional attention 使用。
```

### 7.3 MetadataBuilder 从 CommonAttentionMetadata 翻译 backend metadata

`FlashAttentionMetadataBuilder.build()` 位于 `vllm/v1/attention/backends/flash_attn.py:443`。

它做的事情可以拆成：

```text
1. 从 CommonAttentionMetadata 取出公共字段：
   query_start_loc / seq_lens / block_table / slot_mapping / causal 等。

2. 判断是否启用 FA3 AOT scheduler：
   full cuda graph + FA3 + 非 fast_build + 非 batch invariant。

3. 处理 DCP：
   计算每个 DCP rank 看到的 context KV 长度。

4. 处理 cascade attention：
   公共 prefix 单独作为 prefix attention；
   剩余部分作为 suffix attention。

5. 处理 mm_prefix：
   把多模态 bidirectional ranges 转成 mm_prefix_range_tensor。
```

构造出的 metadata 会在 attention forward 阶段被每层 backend impl 使用。

### 7.4 KV cache update

FlashAttention backend 不在 `forward()` 内部更新 KV cache：

```text
forward_includes_kv_cache_update = False
```

真正写 KV cache 的方法是 `do_kv_cache_update()`，见 `vllm/v1/attention/backends/flash_attn.py:927`。

逻辑是：

```text
key_cache, value_cache = kv_cache.unbind(1)
reshape_and_cache_flash(
  key,
  value,
  key_cache,
  value_cache,
  slot_mapping,
  kv_cache_dtype,
  k_scale,
  v_scale,
)
```

其中 `slot_mapping` 决定每个新 token 的 K/V 写到哪个 KV cache slot。

### 7.5 Forward 主路径

普通 decoder 路径的核心调用在 `vllm/v1/attention/backends/flash_attn.py:870`：

```text
flash_attn_varlen_func(
  q=query,
  k=key_cache,
  v=value_cache,
  out=output,
  cu_seqlens_q=query_start_loc,
  seqused_k=seq_lens,
  max_seqlen_q=max_query_len,
  max_seqlen_k=max_seq_len,
  block_table=block_table,
  causal=causal,
  window_size=sliding_window,
  alibi_slopes=alibi_slopes,
  softcap=logits_soft_cap,
  scheduler_metadata=scheduler_metadata,
  fa_version=vllm_flash_attn_version,
  q_descale/k_descale/v_descale=...,
)
```

注意这里的 `k/v` 是 paged KV cache，而不是当前 step 的连续 K/V tensor。当前 step 的 K/V 先通过 `reshape_and_cache_flash()` 写入 cache，再由 attention kernel 按 `block_table` 读取。

### 7.6 Encoder attention 路径

encoder attention 不使用 KV cache，直接用当前 batch 的 Q/K/V：

```text
query / key / value
  → flash_attn_varlen_func(..., cu_seqlens_q, cu_seqlens_k, causal=False)
```

对应位置：`vllm/v1/attention/backends/flash_attn.py:1061` 到 `vllm/v1/attention/backends/flash_attn.py:1128`。

### 7.7 DCP 路径

Decode Context Parallelism 下，FlashAttention 会拆成两段：

```text
context attention:
  query_across_dcp attend to paged KV cache 的 context 部分；

query attention:
  当前 query attend to 本轮新 token 部分；

最后：
  merge_attn_states(context_out, context_lse, query_out, query_lse)
```

对应位置：`vllm/v1/attention/backends/flash_attn.py:962` 到 `vllm/v1/attention/backends/flash_attn.py:1059`。

---

## 8. Cascade attention：公共 prefix 的特殊优化

当一个 batch 中多个请求共享很长 prefix 时，vLLM 可以把 attention 拆成：

```text
prefix attention:
  所有 query 一起 attend 到 shared prefix。

suffix attention:
  每个请求 attend 到自己的 suffix KV。

merge:
  用 LSE 合并 prefix/suffix 两部分 attention state。
```

是否启用由 `use_cascade_attention()` 决定，见 `vllm/v1/attention/backends/flash_attn.py:1172`。

启用条件大致是：

```text
common_prefix_len >= 256；
请求数 >= 8；
不使用 ALiBi；
不使用 sliding window / local attention；
DCP world size == 1；
启发式判断 cascade 比普通 FlashDecoding 更划算。
```

真正执行在 `cascade_attention()`，见 `vllm/v1/attention/backends/flash_attn.py:1250`。

限制：

```text
不支持 ALiBi；
不支持 sliding window；
common_prefix_len 必须按 block_size 对齐。
```

---

## 9. FLASH_ATTN_DIFFKV：QK head size 和 V head size 不同的路径

`FlashAttentionDiffKVBackend` 用于：

```text
head_size_qk != head_size_v
```

普通 FlashAttention backend 默认 K/V head dim 相同；DiffKV backend 把 K/V pack 到同一个 cache 的最后一维：

```text
(num_blocks, block_size, num_kv_heads, head_size_qk + head_size_v)
```

核心逻辑：

```text
KV cache update:
  triton_reshape_and_cache_flash_diffkv

Forward:
  从 packed cache 切出 k_cache 和 v_cache；
  调 flash_attn_varlen_func；
  要求 FA3 / FA4。
```

适用场景主要是一些 K/Q 和 V 维度不同的模型或变体。

---

## 10. FlashAttention 和 FlashInfer 的关系

### 10.1 二者是不同 backend

在 vLLM 中：

```text
FLASH_ATTN:
  使用 vllm.vllm_flash_attn 的 flash_attn_varlen_func。

FLASHINFER:
  使用 flashinfer 的 BatchPrefillWithPagedKVCacheWrapper、
  BatchDecodeWithPagedKVCacheWrapper，或 TRTLLM attention kernel。
```

它们都可以服务 paged KV cache，但实现和调度不同。

### 10.2 FlashInferBackend 的能力

`FlashInferBackend` 位于 `vllm/v1/attention/backends/flashinfer.py:325`。

它支持：

```text
dtype:
  fp16 / bf16

kv_cache_dtype:
  auto / float16 / bfloat16 / fp8 / fp8_e4m3 / fp8_e5m2 / nvfp4

head_size:
  64 / 128 / 256 / 512

block_size:
  16 / 32 / 64；
  TRTLLM GQA/MQA 动态 kernel 可支持 128 / 256 / 512 / 1024。

compute capability:
  7.5 到 12.1
```

Blackwell 上 FlashInfer 会要求 HND KV cache layout，见 `vllm/v1/attention/backends/flashinfer.py:447`。

### 10.3 FlashInfer 内部再分 native 和 TRTLLM 路径

FlashInfer backend 不是单一 kernel。

它会根据配置和硬件拆成：

```text
native FlashInfer prefill:
  BatchPrefillWithPagedKVCacheWrapper

native FlashInfer decode:
  BatchDecodeWithPagedKVCacheWrapper

TRTLLM prefill:
  trtllm_batch_context_with_kv_cache

TRTLLM decode:
  trtllm_batch_decode_with_kv_cache
```

所以 FlashInfer 和 FlashAttention 的关系是：

```text
它们都是 attention backend 候选；
FlashInfer 在 Blackwell / FP8 / NVFP4 / TRTLLM 场景中更重要；
FlashAttention 在常规 CUDA，尤其非 SM100 场景中仍是默认高优先级 backend。
```

---

## 11. FlashAttention 和 Triton attention 的关系

Triton attention 是 vLLM 自己维护的 Triton kernel backend，不属于 flash-attn 包。

`TritonAttentionBackend` 位于 `vllm/v1/attention/backends/triton_attn.py`。

它的定位是：

```text
作为 FLASH_ATTN / FLASHINFER 之外的通用 fallback 或特性补充；
支持更多 KV cache dtype 变体；
支持 unified attention Triton op；
支持 non-causal、mm_prefix、sink、ALiBi sqrt 等特性。
```

普通 forward 调用：

```text
unified_attention(...)
```

KV cache update 调用：

```text
triton_reshape_and_cache_flash(...)
```

DiffKV 也有 Triton 版本：

```text
TRITON_ATTN_DIFFKV
  → unified_attention_diffkv
```

因此，Triton attention 和 FlashAttention 的关系是：

```text
不是版本关系；
是并列 backend；
当 FlashAttention/FlashInfer 因 dtype、head_size、feature、平台限制不可用时，Triton 可能成为候选。
```

---

## 12. FlashAttention 和 MLA / FlashMLA 的关系

### 12.1 MLA 在 vLLM 中有单独 attention 层

MLA attention 不是普通 `Attention` 的一个小分支，而是有专门的 `MLAAttention` 和 `MLACommonBackend`。

主入口在：

```text
vllm/model_executor/layers/attention/mla_attention.py
```

MLA 的执行思路是：

```text
prefill:
  更偏 compute-friendly，通常走 MHA-style forward_mha。

decode:
  更偏 data-movement-friendly，走 MQA-style forward_mqa。
```

### 12.2 FLASH_ATTN_MLA

`FlashAttnMLABackend` 位于 `vllm/v1/attention/backends/mla/flashattn_mla.py:43`。

限制和特点：

```text
只支持 SM90；
依赖 flash_attn_supports_mla()，也就是 FA3 + SM90；
KV dtype 只支持 auto / float16 / bfloat16；
不支持 FP8 KV cache；
不支持 ALiBi / sliding_window / logits_soft_cap；
只支持 decoder attention。
```

`forward_mqa()` 中会把 MLA 的 q 拆成：

```text
q_nope
q_pe
```

然后调用：

```text
flash_attn_varlen_func(
  q=q_pe,
  k=k_pe_cache.unsqueeze(-2),
  v=kv_c_cache.unsqueeze(-2),
  q_v=q_nope,
  fa_version=3,
  ...
)
```

对应位置：`vllm/v1/attention/backends/mla/flashattn_mla.py:318` 到 `vllm/v1/attention/backends/mla/flashattn_mla.py:365`。

### 12.3 FlashMLA

`FlashMLA` 是 MLA 专用 kernel backend，不等同于 FlashAttention 普通 backend。

特点：

```text
block_size 固定 64；
支持 SM90 / SM100；
在 vLLM registry 中，dense backend `FLASHMLA` 和 sparse backend `FLASHMLA_SPARSE` 是不同 backend；不要把 sparse 能力理解为 dense `FLASHMLA` 的普通运行分支；
BF16/FP16 调 flash_mla_with_kvcache；
FP8 调 flash_mla_with_kvcache_fp8；
不支持 ALiBi / sliding_window / logits_soft_cap；
只支持 decoder。
```

所以：

```text
FLASH_ATTN_MLA:
  用 FA3 varlen func 来服务 MLA decode。

FLASHMLA:
  用 MLA 专用 kernel 服务 MLA decode。

FLASHINFER_MLA:
  用 FlashInfer / TRTLLM MLA kernel，Blackwell 上优先级更高。

TRITON_MLA:
  用 Triton decode attention 作为 MLA fallback / 补充。
```

---

## 13. ROCm 上的 FlashAttention family

ROCm 平台不完全复用 CUDA 的 vLLM FlashAttention wrapper。

`fa_utils.py` 中的逻辑是：

```text
CUDA:
  from vllm.vllm_flash_attn import flash_attn_varlen_func

XPU:
  from vllm._xpu_ops import xpu_ops.flash_attn_varlen_func

ROCm:
  尝试导入 upstream flash_attn.flash_attn_varlen_func；
  reshape_and_cache_flash 使用 vLLM custom ops；
  scheduler metadata 返回 None。
```

对应位置：`vllm/v1/attention/backends/fa_utils.py:18` 到 `vllm/v1/attention/backends/fa_utils.py:54`。

ROCm 平台的默认 backend 选择更偏向 ROCm 原生和 AITER：

```text
非 MLA:
  ROCM_ATTN
  ROCM_AITER_FA
  ROCM_AITER_UNIFIED_ATTN
  TRITON_ATTN
  TURBOQUANT

MLA:
  ROCM_AITER_MLA
  TRITON_MLA
  ROCM_AITER_TRITON_MLA
```

AITER MLA / FA / Unified 是否进入候选取决于 AITER 是否可用或启用；如果 AITER MLA 未启用，ROCm MLA 通常只返回 `TRITON_MLA`。另外启用 KV connector 时，ROCm 普通 attention 会跳过 layout 不兼容的 `ROCM_ATTN`。

其中：

```text
ROCM_ATTN:
  ROCm native paged attention。

ROCM_AITER_FA:
  AITER FlashAttention 风格 backend，prefill/extend 可走 flash_attn_varlen_func。

ROCM_AITER_UNIFIED_ATTN:
  AITER unified attention。
```

因此 ROCm 上的“FlashAttention family”更像一组 ROCm/AITER/上游 flash-attn 组合路径，而不是 CUDA `vllm_flash_attn` 的简单复刻。

---

## 14. XPU 和 TPU 简述

### 14.1 XPU

`fa_utils.py` 中 XPU 会把：

```text
flash_attn_varlen_func
get_scheduler_metadata
```

映射到 `_xpu_ops.xpu_ops`，并且 `get_flash_attn_version()` 对 XPU 返回 2。

也就是说，XPU 有自己的 FA-like op 接口，但不走 CUDA FA3/FA4 选择逻辑。

### 14.2 TPU

vLLM 仓库内的 TPU 平台文件主要是外部包接入口：

```text
vllm/platforms/tpu.py
  → from tpu_inference.platforms import TpuPlatform
```

当前源码目录中没有看到 TPU 专属 FlashAttention backend 的实现，TPU attention backend 大概率由外部 `tpu_inference` 包提供。

---

## 15. prefill / decode / paged KV cache 的支持边界

### 15.1 prefill

prefill 的典型形态是：

```text
query_len > 1；
需要处理 prompt 的一段连续 token；
attention 计算量大，FlashAttention 的 IO-aware tiling 很重要。
```

FlashAttention backend 在 prefill 中使用同一个 `flash_attn_varlen_func()`，通过：

```text
cu_seqlens_q
max_seqlen_q
seqused_k / cu_seqlens_k
block_table
causal / non-causal
```

描述 varlen batch 和 KV 读取范围。

### 15.2 decode

decode 的典型形态是：

```text
每个请求通常 query_len = 1；
KV cache 很长；
主要瓶颈是从 KV cache 读历史 K/V。
```

FlashAttention backend 仍然可以走 `flash_attn_varlen_func()` + paged KV cache。对于 GQA/MQA decode，底层 flash-attn 可能走 FlashDecoding 类路径。

FlashInfer / TRTLLM 在 decode 场景也很重要，尤其 Blackwell 和 FP8/NVFP4 场景。

### 15.3 paged KV cache

vLLM 的 paged KV cache 对 FlashAttention backend 的关键输入是：

```text
block_table:
  请求 → KV block id 列表。

slot_mapping:
  token → KV cache slot。

seq_lens:
  请求当前有效 KV 长度。

block_size:
  KV cache page/block 大小。
```

FlashAttention backend 要求 block size 是 16 的倍数；FlashInfer 默认支持 16/32/64，并在 TRTLLM GQA/MQA 动态 kernel 下支持更大 page。

---

## 16. 能力边界汇总

### 16.1 FLASH_ATTN

```text
适合：
  CUDA 常规 fp16/bf16 attention；
  paged KV cache；
  varlen prefill/decode；
  encoder / decoder / cross attention；
  sliding window / ALiBi / softcap；
  FA3 AOT scheduler；
  FA4 mm_prefix / dynamic causal。

限制：
  compute capability >= 8.0；
  head_size 需要 8 对齐；
  head_size > 256 依赖 FA4；
  block_size 需要 16 的倍数；
  sink 需要 FA3/FA4 且高算力设备；
  mm_prefix 需要 FA4；
  fused output quantization 暂不支持；
  cascade 不支持 ALiBi / sliding window。
```

### 16.2 FLASHINFER

```text
适合：
  CUDA paged prefill/decode；
  FP8 / FP8_E5M2 / NVFP4 KV cache；
  Blackwell / TRTLLM attention 路径；
  大 page GQA/MQA decode。

限制：
  head_size 主要是 64 / 128 / 256 / 512；
  native FlashInfer 对全局超参一致性要求更高；
  encoder/cross attention 不如 FLASH_ATTN 通用；
  Blackwell 上强制 HND KV layout。
```

### 16.3 TRITON_ATTN

```text
适合：
  FlashAttention / FlashInfer 不满足特性组合时的 fallback；
  更多 KV dtype 变体；
  vLLM 自维护 unified attention；
  一些 non-causal / mm_prefix / sink / ALiBi sqrt 场景。

限制：
  性能特征取决于具体模型、batch shape、GPU 和 Triton kernel；
  不是 flash-attn 包的一部分。
```

### 16.4 FLASH_ATTN_MLA / FLASHMLA / FLASHINFER_MLA

```text
FLASH_ATTN_MLA:
  FA3 + SM90；不支持 FP8 KV；适合部分 MLA decode。

FLASHMLA:
  MLA 专用 dense kernel；SM90/SM100；block_size 固定 64。Sparse MLA 在 vLLM 中是独立 backend `FLASHMLA_SPARSE`。

FLASHINFER_MLA:
  Blackwell 上优先级高；调用 TRTLLM MLA decode kernel；强制 HND layout。
```

---

## 17. 最小调用链总结

把普通 decoder attention 放到一条线上：

```text
SchedulerOutput
  → GPUModelRunner._update_states()
  → GPUModelRunner._prepare_inputs()
  → GPUModelRunner._get_slot_mappings()
  → GPUModelRunner._build_attention_metadata()
  → CommonAttentionMetadata
  → FlashAttentionMetadataBuilder.build()
  → FlashAttentionMetadata
  → set_forward_context(...)
  → model forward
  → Attention.forward()
  → backend Impl.do_kv_cache_update()
      → reshape_and_cache_flash(..., slot_mapping, ...)
  → backend Impl.forward()
      → flash_attn_varlen_func(..., block_table, seq_lens, query_start_loc, ...)
  → attention output
```

这条链路说明：

```text
ModelRunner 不直接调用 FlashAttention；
Attention 层也不自己决定请求级 batch 结构；
backend metadata 是二者之间的桥梁；
Paged KV cache 的 block_table / slot_mapping 是 FlashAttention 在 vLLM 中运行的关键。
```

---

## 18. 阅读源码时的抓手

如果只想快速定位，可以按下面顺序读：

```text
1. vllm/v1/attention/selector.py
   看 backend 如何被选出来。

2. vllm/platforms/cuda.py
   看 CUDA 上 FLASH_ATTN / FLASHINFER / TRITON_ATTN / MLA backend 的优先级。

3. vllm/v1/attention/backends/fa_utils.py
   看 FA2/FA3/FA4 如何选择和回退。

4. vllm/vllm_flash_attn/flash_attn_interface.py
   看 vLLM 对 flash-attn extension 的 wrapper。

5. vllm/v1/attention/backends/flash_attn.py
   看 FLASH_ATTN 的 metadata、KV cache update、forward 调用。

6. vllm/v1/attention/backends/flashinfer.py
   看 FlashInfer native / TRTLLM prefill/decode 分派。

7. vllm/model_executor/layers/attention/mla_attention.py
   看 MLA 为什么有单独 forward_mha / forward_mqa。

8. vllm/v1/attention/backends/mla/flashattn_mla.py
   看 FlashAttention 如何服务 MLA decode。

9. vllm/v1/attention/backends/mla/flashmla.py
   看 FlashMLA 专用 kernel 路径。
```

---

## 19. 核心结论

```text
1. FlashAttention family 是 attention kernel 演进线：
   v1 解决 IO-aware tiling；
   v2 改进并行切分和通用 CUDA 性能；
   v3 面向 Hopper，加入 AOT scheduler 等能力；
   v4 面向 CuTe / Hopper / Blackwell 新路径，支持 mm_prefix、dynamic causal 等 vLLM 特性，但也有 TMEM / batch invariant 限制。

2. vLLM 中的 FLASH_ATTN backend 通过 flash_attn_varlen_func 接入 paged KV cache：
   block_table 负责读 cache；
   slot_mapping 负责写 cache；
   query_start_loc / seq_lens / max_seq_len 描述 varlen batch。

3. FlashAttention 和 PagedAttention 是互补关系：
   PagedAttention 管 KV cache 分页；
   FlashAttention 算 attention。

4. FlashInfer、Triton attention、FlashMLA 不是 FlashAttention 的简单别名：
   它们是 vLLM attention backend 生态中的并列或专用路径。

5. CUDA 默认选择随硬件变化：
   非 Blackwell 常规 attention 通常 FLASH_ATTN 优先；
   Blackwell 更偏 FLASHINFER；
   MLA 在 Blackwell 更偏 FLASHINFER_MLA / FlashMLA / CUTLASS 等专用路径。

6. 判断一个模型最终走哪条路径，不能只看“是不是 FlashAttention”：
   要同时看 head_size、dtype、kv_cache_dtype、block_size、use_mla、sink、mm_prefix、attn_type、compute capability、batch invariant、KV connector 和用户显式配置。
```
