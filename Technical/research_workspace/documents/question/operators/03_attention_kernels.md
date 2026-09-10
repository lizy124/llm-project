# 03. Attention 算子如何支撑 prefill / decode？

源码位置：

- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `code/vllm/vllm/v1/attention/backend.py`
- `code/vllm/vllm/v1/attention/selector.py`
- `code/vllm/vllm/v1/attention/backends/registry.py`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py`
- `code/vllm/vllm/v1/attention/backends/flashinfer.py`
- `code/vllm/vllm/v1/attention/backends/mla/`
- `code/vllm/vllm/v1/attention/ops/`
- `code/vllm/vllm/_custom_ops.py`
- `code/vllm/csrc/libtorch_stable/attention/`

这个问题关注：PagedAttention、FlashAttention、FlashInfer、FlashMLA、Triton attention 等 attention kernel 如何被选择、如何消费 metadata、如何支撑 prefill / decode / mixed batch / cascade / sliding window / MLA 等路径。

---

## 1. 一句话回答

Attention kernel 是 vLLM 推理性能最核心的算子族，负责在不同 batch 状态下高效读取 KV cache 并完成 attention 计算。

最小链路是：

```text
ModelRunner 构造 attention metadata
  → Attention layer forward
  → selected attention backend
  → KV cache update
  → prefill / decode / mixed kernel
  → attention output
```

更完整一点：

```text
SchedulerOutput
  → ModelRunner._prepare_inputs()
  → ModelRunner._build_attention_metadata()
  → model attention layer
  → Attention.forward()
  → torch.ops.vllm.unified_attention_with_output()
  → AttentionImpl.forward()
  → FlashAttention / FlashInfer / Triton / ROCm / MLA kernel
```

---

## 2. Attention layer 是统一入口

标准 attention 层入口是 `Attention`。

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:178`

它的类注释说明了三件事：

```text
1. 把输入 key / value 存入 KV cache；
2. 执行 multi-head / multi-query / grouped-query attention；
3. 返回 output tensor。
```

关键初始化逻辑在：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:190`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:224`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:303`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:373`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:390`

初始化阶段会确定：

```text
- num_heads / num_kv_heads / head_size / scale；
- sliding_window；
- kv_cache_dtype；
- 是否需要 KV cache quantization；
- 是否需要 per-head quant scales；
- 是否使用 mm_prefix；
- attention backend；
- backend impl；
- 是否把 attention 包成 opaque torch custom op；
- 当前 layer 在 static_forward_context 中的注册。
```

一句话：

```text
Attention layer 是模型结构层入口，backend impl 才是真正决定 kernel 路径的执行对象。
```

---

## 3. Attention backend 如何选择

选择入口：`get_attn_backend()`。

源码位置：`code/vllm/vllm/v1/attention/selector.py:54`

主链路：

```text
Attention.__init__()
  → get_attn_backend(head_size, dtype, kv_cache_dtype, ...)
  → AttentionSelectorConfig(...)
  → _cached_get_attn_backend(...)
  → current_platform.get_attn_backend_cls(...)
  → backend.validate_configuration(...)
  → resolve_obj_by_qualname(attention_cls)
  → backend class
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:303`
- `code/vllm/vllm/v1/attention/selector.py:54`
- `code/vllm/vllm/v1/attention/selector.py:90`
- `code/vllm/vllm/v1/attention/selector.py:113`
- `code/vllm/vllm/v1/attention/selector.py:121`
- `code/vllm/vllm/v1/attention/selector.py:132`

`AttentionSelectorConfig` 包含：

```text
- head_size；
- dtype；
- kv_cache_dtype；
- block_size；
- use_mla；
- has_sink；
- use_sparse；
- use_mm_prefix；
- use_per_head_quant_scales；
- attn_type；
- use_non_causal；
- use_batch_invariant；
- use_kv_connector。
```

源码位置：`code/vllm/vllm/v1/attention/selector.py:21`

---

## 4. backend registry 管理哪些实现

Attention backend 枚举在 `registry.py`。

源码位置：`code/vllm/vllm/v1/attention/backends/registry.py:34`

主要 backend 包括：

```text
- FLASH_ATTN
- FLASH_ATTN_DIFFKV
- TRITON_ATTN
- TRITON_ATTN_DIFFKV
- ROCM_ATTN
- ROCM_AITER_FA
- ROCM_AITER_UNIFIED_ATTN
- FLASHINFER
- FLASHINFER_MLA
- FLASHINFER_MLA_SPARSE
- TRITON_MLA
- CUTLASS_MLA
- FLASHMLA
- FLASHMLA_SPARSE
- FLASH_ATTN_MLA
- FLEX_ATTENTION
- CPU_ATTN
- TURBOQUANT
- CUSTOM
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/registry.py:44`
- `code/vllm/vllm/v1/attention/backends/registry.py:48`
- `code/vllm/vllm/v1/attention/backends/registry.py:65`
- `code/vllm/vllm/v1/attention/backends/registry.py:76`
- `code/vllm/vllm/v1/attention/backends/registry.py:78`
- `code/vllm/vllm/v1/attention/backends/registry.py:98`
- `code/vllm/vllm/v1/attention/backends/registry.py:103`
- `code/vllm/vllm/v1/attention/backends/registry.py:104`
- `code/vllm/vllm/v1/attention/backends/registry.py:220`

registry 还支持覆盖和自定义：

```text
register_backend(AttentionBackendEnum.FLASH_ATTN)
register_backend(AttentionBackendEnum.CUSTOM, "my.module.MyBackend")
```

源码位置：`code/vllm/vllm/v1/attention/backends/registry.py:220`

---

## 5. backend 抽象定义了什么

`AttentionBackend` 是 backend 抽象基类。

源码位置：`code/vllm/vllm/v1/attention/backend.py:55`

它要求每个 backend 提供：

```text
get_name()
get_impl_cls()
get_builder_cls()
get_kv_cache_shape()
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:68`
- `code/vllm/vllm/v1/attention/backend.py:73`
- `code/vllm/vllm/v1/attention/backend.py:78`
- `code/vllm/vllm/v1/attention/backend.py:83`
- `code/vllm/vllm/v1/attention/backend.py:88`

同时它提供一组能力检查：

```text
supports_head_size()
supports_dtype()
supports_kv_cache_dtype()
supports_block_size()
is_mla()
supports_sink()
supports_mm_prefix()
is_sparse()
supports_per_head_quant_scales()
supports_non_causal()
supports_batch_invariance()
supports_kv_connector()
supports_attn_type()
supports_compute_capability()
validate_configuration()
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:153`
- `code/vllm/vllm/v1/attention/backend.py:162`
- `code/vllm/vllm/v1/attention/backend.py:166`
- `code/vllm/vllm/v1/attention/backend.py:174`
- `code/vllm/vllm/v1/attention/backend.py:204`
- `code/vllm/vllm/v1/attention/backend.py:208`
- `code/vllm/vllm/v1/attention/backend.py:216`
- `code/vllm/vllm/v1/attention/backend.py:220`
- `code/vllm/vllm/v1/attention/backend.py:224`
- `code/vllm/vllm/v1/attention/backend.py:228`
- `code/vllm/vllm/v1/attention/backend.py:240`
- `code/vllm/vllm/v1/attention/backend.py:244`
- `code/vllm/vllm/v1/attention/backend.py:248`
- `code/vllm/vllm/v1/attention/backend.py:256`
- `code/vllm/vllm/v1/attention/backend.py:276`

一句话：

```text
AttentionBackend 不是 kernel 本身，而是“这个 backend 能不能用于当前配置”的能力描述和 impl / metadata builder 工厂。
```

---

## 6. CUDA 平台如何选择 backend

CUDA 平台选择逻辑在 `CudaPlatform.get_attn_backend_cls()`。

源码位置：`code/vllm/vllm/platforms/cuda.py:351`

主流程：

```text
如果用户显式指定 selected_backend：
  → 只校验这个 backend；
  → 不合法则报错；
  → 合法则使用。

否则：
  → 根据优先级枚举候选 backend；
  → 对每个 backend 执行 validate_configuration()；
  → 收集 invalid_reasons；
  → 选择优先级最高的 valid backend；
  → 如果 block_size 排除了高优先级 backend，打印 warning；
  → 返回 backend class path。
```

关键源码：

- `code/vllm/vllm/platforms/cuda.py:316`
- `code/vllm/vllm/platforms/cuda.py:328`
- `code/vllm/vllm/platforms/cuda.py:351`
- `code/vllm/vllm/platforms/cuda.py:360`
- `code/vllm/vllm/platforms/cuda.py:381`
- `code/vllm/vllm/platforms/cuda.py:399`
- `code/vllm/vllm/platforms/cuda.py:405`
- `code/vllm/vllm/platforms/cuda.py:415`
- `code/vllm/vllm/platforms/cuda.py:436`

需要特别注意：

```text
用户显式设置 --attention-backend 时，vLLM 不会静默 fallback；
如果这个 backend 对当前配置无效，会直接报错。
```

而自动选择时，vLLM 会从候选里选最高优先级可用 backend。

---

## 7. metadata 如何进入 attention kernel

Attention kernel 不只吃 Q/K/V，还需要 batch / cache metadata。

公共 metadata 结构：`CommonAttentionMetadata`。

源码位置：`code/vllm/vllm/v1/attention/backend.py:361`

核心字段：

```text
query_start_loc：
  每个 request 在 query tensor 中的起始位置。

seq_lens：
  每个 request 当前总序列长度。

num_reqs：
  batch 中 request 数。

num_actual_tokens：
  当前 batch 实际 token 数。

max_query_len：
  batch 中最大 query 长度。

max_seq_len：
  batch 中最大上下文长度。

block_table_tensor：
  request 到 KV cache block 的映射。

slot_mapping：
  当前 token 写入 KV cache 的物理 slot 映射。

causal：
  是否因果 attention，可以是全局 bool 或 per-seq tensor。

is_prefilling：
  request 是否仍在 prefill 阶段。

mm_req_doc_ranges：
  multimodal PrefixLM 双向 attention 范围。
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:370`
- `code/vllm/vllm/v1/attention/backend.py:374`
- `code/vllm/vllm/v1/attention/backend.py:377`
- `code/vllm/vllm/v1/attention/backend.py:380`
- `code/vllm/vllm/v1/attention/backend.py:382`
- `code/vllm/vllm/v1/attention/backend.py:384`
- `code/vllm/vllm/v1/attention/backend.py:387`
- `code/vllm/vllm/v1/attention/backend.py:388`
- `code/vllm/vllm/v1/attention/backend.py:409`
- `code/vllm/vllm/v1/attention/backend.py:420`

每个 backend 还有自己的 metadata builder。

抽象类：`AttentionMetadataBuilder`。

源码位置：`code/vllm/vllm/v1/attention/backend.py:533`

它负责：

```text
- 接收 CommonAttentionMetadata；
- 转换成 backend-specific metadata；
- 判断 CUDA Graph 支持级别；
- 需要时 reorder batch；
- 需要时 update block table；
- 为 capture / drafting 构造特殊 metadata。
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:558`
- `code/vllm/vllm/v1/attention/backend.py:599`
- `code/vllm/vllm/v1/attention/backend.py:619`
- `code/vllm/vllm/v1/attention/backend.py:634`
- `code/vllm/vllm/v1/attention/backend.py:646`
- `code/vllm/vllm/v1/attention/backend.py:668`

---

## 8. Attention.forward 如何连接 opaque op 和 backend impl

`Attention.forward()` 是 runtime 执行入口。

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:438`

关键步骤：

```text
1. 如果需要动态计算 KV scale，调用 maybe_calc_kv_scales；
2. 如果 backend 支持 query quant input，可能先量化 query；
3. 分配 output tensor；
4. 把 query / key / value view 成 [tokens, heads, head_dim]；
5. 如果 backend 的 forward 不包含 KV cache update，则先调用 unified_kv_cache_update；
6. 调用 unified_attention_with_output；
7. 返回 output.view(-1, hidden_size)。
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:457`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:462`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:479`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:481`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:491`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:499`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:502`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:519`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:522`

注意这里有两种调用模式：

```text
use_direct_call == True：
  直接调用 Python 函数 unified_attention_with_output()。

use_direct_call == False：
  调用 torch.ops.vllm.unified_attention_with_output()。
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:490`

`use_direct_call` 的决定：

```text
self.use_direct_call = not current_platform.opaque_attention_op()
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:394`

---

## 9. unified_attention_with_output 做了什么

`unified_attention_with_output()` 是 attention opaque op 的 Python 实现。

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:736`

它做的事情很少但很关键：

```text
1. 从 forward context 中按 layer_name 取出 attention metadata；
2. 取出对应 Attention layer；
3. 取出绑定到该 layer 的 KV cache；
4. 调用 self.impl.forward(..., output=output)。
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:649`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:671`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:682`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:736`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:751`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:753`

也就是说：

```text
torch.ops.vllm.unified_attention_with_output 不是最终 attention kernel；
它是 compile / graph 友好的统一入口，真正执行由 backend impl.forward() 完成。
```

---

## 10. KV cache update 和 attention forward 为什么可能分开

`AttentionBackend.forward_includes_kv_cache_update` 表示 backend 的 forward 是否自己包含 KV cache 更新。

源码位置：`code/vllm/vllm/v1/attention/backend.py:65`

例如 FlashAttention backend：

```text
forward_includes_kv_cache_update = False
```

源码位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:96`

Triton backend 也是：

```text
forward_includes_kv_cache_update = False
```

源码位置：`code/vllm/vllm/v1/attention/backends/triton_attn.py:275`

当 backend 不在 forward 内更新 KV cache 时，Attention.forward 会先调用：

```text
unified_kv_cache_update(key, value, layer_name)
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:499`

`unified_kv_cache_update()` 内部会调用 backend impl 的：

```text
do_kv_cache_update(...)
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:692`

这一步返回一个 dummy tensor，用作数据依赖，保证 torch.compile 保留 KV cache update 和 attention forward 的顺序。

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:697`

---

## 11. KV cache 写入路径

### 11.1 PagedAttention cache writer

PagedAttention 的写入入口：

```text
PagedAttention.write_to_paged_cache()
  → ops.reshape_and_cache(...)
```

源码位置：`code/vllm/vllm/v1/attention/ops/paged_attn.py:31`

这个路径把当前 step 的 key / value 根据 `slot_mapping` 写入物理 KV cache。

关键输入：

```text
- key
- value
- key_cache
- value_cache
- slot_mapping
- kv_cache_dtype
- k_scale
- v_scale
```

源码位置：`code/vllm/vllm/v1/attention/ops/paged_attn.py:31`

### 11.2 TritonAttention KV cache update

Triton backend 的 cache update：

```text
TritonAttentionImpl.do_kv_cache_update()
  → triton_reshape_and_cache_flash_per_token_head_quant()
  或 triton_reshape_and_cache_flash()
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:724`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:737`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:743`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:758`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:33`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:334`

它支持：

```text
- 普通 KV cache；
- FP8 KV cache；
- per-token-head quant scales；
- NHD / HND cache layout；
- stride-aware cache writer。
```

### 11.3 MLA KV cache update

MLA impl 的 KV cache update 不写 K/V 两个 cache，而是 concat MLA 的 compressed KV 表示。

源码位置：`code/vllm/vllm/v1/attention/backend.py:931`

调用：

```text
ops.concat_and_cache_mla(
  kv_c_normed,
  k_pe.squeeze(1),
  kv_cache,
  slot_mapping.flatten(),
  kv_cache_dtype,
  scale,
)
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:931`
- `code/vllm/vllm/v1/attention/backend.py:944`
- `code/vllm/vllm/_custom_ops.py:2690`

---

## 12. FlashAttention backend

FlashAttention backend 源码：`code/vllm/vllm/v1/attention/backends/flash_attn.py:68`

它声明：

```text
- supported_dtypes: fp16 / bf16；
- supported_kv_cache_dtypes: auto / fp16 / bf16；
- block_size 通常要求 multiple of 16；
- 支持 batch invariance；
- 支持 non-causal；
- 支持 decoder / encoder / encoder_only / encoder_decoder；
- FA3 支持 per-head quant scales；
- KV cache shape: (num_blocks, 2, block_size, num_kv_heads, head_size)。
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/flash_attn.py:68`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:76`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:96`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:104`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:109`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:117`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:126`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:140`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:172`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:182`

FlashAttention metadata：

源码位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:236`

核心字段：

```text
- num_actual_tokens
- max_query_len
- query_start_loc
- max_seq_len
- seq_lens
- block_table
- slot_mapping
- use_cascade
- common_prefix_len
- prefix / suffix kv lens
- DCP context kv lens
- scheduler_metadata
- causal
- mm_prefix_range_tensor
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/flash_attn.py:246`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:248`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:250`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:252`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:254`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:256`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:263`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:265`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:270`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:274`

FlashAttention 的 FA 版本选择在 `fa_utils.py`：

```text
CUDA：根据 compute capability 优先选择 FA3 / FA4 / FA2；
XPU：返回 FA2；
ROCm：返回 None，使用 upstream flash_attn 或 AITER 路径；
attention_config.flash_attn_version 可以 override；
ALiBi / sinks / head_size / batch invariance 会触发降级。
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/fa_utils.py:56`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:78`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:89`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:98`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:106`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:118`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:150`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:231`

---

## 13. TritonAttention backend

Triton backend 源码：`code/vllm/vllm/v1/attention/backends/triton_attn.py:248`

它声明：

```text
- 支持 fp16 / bf16 / fp32；
- 支持 auto / fp16 / bf16 / fp8 / fp8_e4m3 / fp8_e5m2 / per-token-head KV；
- block_size 要是 16 的倍数；
- 支持 non-causal；
- 支持 batch invariance；
- 支持 mm_prefix；
- 支持 sinks；
- 支持所有 attention type；
- CUDA Graph support 为 ALWAYS。
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:98`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:248`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:254`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:265`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:270`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:275`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:277`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:285`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:346`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:350`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:354`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:359`

Triton metadata：

源码位置：`code/vllm/vllm/v1/attention/backends/triton_attn.py:58`

核心字段包括：

```text
- num_actual_tokens
- max_query_len
- query_start_loc
- max_seq_len
- seq_lens
- block_table
- slot_mapping
- seq_threshold_3D
- softmax segment workspace
- causal
- cascade fields
- scheduler metadata
- mm_prefix_range_tensor
```

TritonAttentionImpl.forward 主链路：

```text
TritonAttentionImpl.forward()
  → 处理 profiling run
  → encoder attention 特殊路径
  → 根据 kv_cache_dtype 拆 key_cache / value_cache / scale cache
  → 取 query_start_loc / seq_lens / block_table
  → 调用 unified_attention(...)
  → kernel_unified_attention[grid](...)
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:530`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:560`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:577`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:590`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:628`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:642`
- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:179`
- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:780`
- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:1019`

Triton attention 的 `kernel_unified_attention` 支持：

```text
- paged KV cache；
- query_start_loc；
- block_tables；
- seq_lens；
- ALiBi；
- softcap；
- sinks；
- sliding window；
- causal / per-seq causal；
- multimodal PrefixLM ranges；
- FP8 / per-token-head KV quantization；
- 2D / 3D kernel 模式；
- tensor descriptor 路径。
```

关键源码：`code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:179`

---

## 14. FlashInfer backend

FlashInfer backend 源码：`code/vllm/vllm/v1/attention/backends/flashinfer.py:325`

它直接使用 FlashInfer wrapper：

```text
BatchDecodeWithPagedKVCacheWrapper
BatchPrefillWithPagedKVCacheWrapper
BatchPrefillWithRaggedKVCacheWrapper
MultiLevelCascadeAttentionWrapper
```

源码位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:11`

支持的 KV cache dtype 包括：

```text
- auto
- float16
- bfloat16
- fp8
- fp8_e4m3
- fp8_e5m2
- nvfp4
```

源码位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:325`

block size 支持逻辑更动态：

```text
默认支持 16 / 32 / 64；
如果满足 TRT-LLM attention 大 page 条件，可以支持 128 / 256 / 512 / 1024。
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/flashinfer.py:337`
- `code/vllm/vllm/v1/attention/backends/flashinfer.py:354`

FlashInfer 还包含一些辅助 Triton kernel，例如 prefill 时对 FP8 KV cache 进行 dequant 并构造 mock cache：

- `code/vllm/vllm/v1/attention/backends/flashinfer.py:98`
- `code/vllm/vllm/v1/attention/backends/flashinfer.py:161`

---

## 15. PagedAttention 是什么

PagedAttention 的核心不是单个函数，而是一套“用 block table 访问 paged KV cache”的 attention 组织方式。

在 Python 侧的简单包装：

- `code/vllm/vllm/v1/attention/ops/paged_attn.py:15`

它提供两个基础动作：

```text
split_kv_cache()
  → 把 KV cache tensor 拆成 key_cache / value_cache 视图。

write_to_paged_cache()
  → 调用 ops.reshape_and_cache() 按 slot_mapping 写入 cache。
```

关键源码：

- `code/vllm/vllm/v1/attention/ops/paged_attn.py:17`
- `code/vllm/vllm/v1/attention/ops/paged_attn.py:31`
- `code/vllm/vllm/v1/attention/ops/paged_attn.py:42`

native paged attention wrapper：

- `code/vllm/vllm/_custom_ops.py:114`
- `code/vllm/vllm/_custom_ops.py:158`

CUDA kernel 侧：

- `code/vllm/csrc/libtorch_stable/attention/attention_kernels.cuh:494`
- `code/vllm/csrc/libtorch_stable/attention/paged_attention_v1.cu:47`
- `code/vllm/csrc/libtorch_stable/attention/paged_attention_v1.cu:164`

最小输入关系：

```text
query
key_cache / value_cache
block_tables
seq_lens
block_size
max_seq_len
scale
k_scale / v_scale
```

PagedAttention 的关键价值是：

```text
request 的逻辑上下文不需要连续存储，kernel 通过 block_table 找到物理 KV cache block。
```

---

## 16. prefill / decode / mixed batch 的差异

### 16.1 prefill

prefill 的特点：

```text
- query_len 通常大于 1；
- 可能是第一次处理 prompt，也可能是 chunked prefill 的一段；
- K/V 新 token 多；
- attention 需要处理 query 内部的新 token 以及已有 context；
- 更适合 varlen flash attention 或 prefill kernel。
```

相关 metadata：

```text
query_start_loc
max_query_len
seq_lens
block_table
slot_mapping
is_prefilling
```

### 16.2 decode

decode 的特点：

```text
- query_len 通常等于 1；
- 每个 request 只追加一个或少量 token；
- 大量读取已有 KV cache；
- 性能瓶颈通常是 memory bandwidth / paged KV access；
- 更适合 paged decode kernel。
```

### 16.3 mixed batch

vLLM V1 中经常出现 decode + prefill 混合 batch。

backend utils 提供 split 逻辑：

- `code/vllm/vllm/v1/attention/backends/utils.py:538`
- `code/vllm/vllm/v1/attention/backends/utils.py:637`

`split_decodes_and_prefills()` 假设 batch 被排成：

```text
decode → short_extend → long_extend → prefill
```

源码位置：`code/vllm/vllm/v1/attention/backends/utils.py:548`

返回：

```text
num_decodes
num_prefills
num_decode_tokens
num_prefill_tokens
```

源码位置：`code/vllm/vllm/v1/attention/backends/utils.py:538`

`reorder_batch_to_split_decodes_and_prefills()` 会把 batch 重排成：

```text
1. decode
2. short_extend
3. long_extend
4. prefill
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/utils.py:637`
- `code/vllm/vllm/v1/attention/backends/utils.py:646`
- `code/vllm/vllm/v1/attention/backends/utils.py:677`

---

## 17. cascade attention 和 merge_attn_states

Cascade attention 会把 attention 拆成 prefix 和 suffix 两部分，再合并结果。

合并入口：

- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:9`
- `code/vllm/vllm/_custom_ops.py:266`
- `code/vllm/vllm/v1/attention/ops/triton_merge_attn_states.py:14`

`merge_attn_states()` 做的是：

```text
用 log-sum-exp rescaling 把 prefix_output 和 suffix_output 合并成最终 output。
```

源码位置：`code/vllm/vllm/v1/attention/ops/merge_attn_states.py:19`

fallback 逻辑：

```text
CUDA + supported dtype + supported head_dim：
  → custom CUDA merge_attn_states
否则：
  → Triton merge_attn_states
```

关键源码：

- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:49`
- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:62`
- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:72`
- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:90`

FlashAttention metadata 中有 cascade 字段：

- `code/vllm/vllm/v1/attention/backends/flash_attn.py:256`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:257`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:258`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:259`

Triton metadata 中也有 cascade 字段：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:84`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:85`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:86`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:89`

---

## 18. sliding window / local attention 如何影响 kernel

`Attention.__init__()` 会从 per-layer 或 cache config 中得到 sliding window：

```text
per_layer_sliding_window 优先；
否则使用 cache_config.sliding_window；
否则 None。
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:214`

Triton backend 会把 sliding window 转成 `(left, right)`：

```text
None：(-1, -1)
encoder / encoder_only：(sliding_window - 1, sliding_window - 1)
decoder：(sliding_window - 1, 0)
```

源码位置：`code/vllm/vllm/v1/attention/backends/triton_attn.py:460`

FlashAttention 和 Triton 都会在 metadata / kernel 参数中带入 sliding window 或 window_size。

Triton forward 传参：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:655`

Triton kernel 参数：

- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:217`

---

## 19. KV cache dtype / quantization 如何影响 attention

KV cache dtype 从 `CacheConfig.cache_dtype` 来，进入 Attention 初始化后会影响：

```text
- backend selection；
- kv_cache_torch_dtype；
- KV cache shape；
- 是否创建 q/k/v/prob scale；
- 是否启用 per-head quant scales；
- 是否预量化 query；
- cache writer 如何写入；
- attention kernel 如何 dequant。
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:224`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:238`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:247`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:277`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:415`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:420`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:462`

Triton backend 中，per-token-head KV quant 会从 padded head dimension 中切出 scale cache：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:303`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:382`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:590`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:737`

FlashInfer 支持更多 KV cache dtype，包括 FP8 和 NVFP4：

- `code/vllm/vllm/v1/attention/backends/flashinfer.py:325`

---

## 20. MLA / sparse MLA 路径

MLA 路径和标准 Attention 不完全一样，因为它的 KV cache、prefill、decode 形态不同。

相关抽象：

- `code/vllm/vllm/v1/attention/backend.py:863`
- `code/vllm/vllm/v1/attention/backend.py:954`

`MLAAttentionImpl` 定义：

```text
forward_mha()
  MHA-style prefill forward pass。

forward_mqa()
  MQA-style decode forward pass。

do_kv_cache_update()
  concat compressed KV 表示到 MLA KV cache。
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:892`
- `code/vllm/vllm/v1/attention/backend.py:907`
- `code/vllm/vllm/v1/attention/backend.py:931`

MLA backend 分布：

```text
vllm/v1/attention/backends/mla/flashmla.py
vllm/v1/attention/backends/mla/flashinfer_mla.py
vllm/v1/attention/backends/mla/triton_mla.py
vllm/v1/attention/backends/mla/cutlass_mla.py
vllm/v1/attention/backends/mla/flashattn_mla.py
vllm/v1/attention/backends/mla/rocm_aiter_mla.py
vllm/v1/attention/backends/mla/*_sparse.py
```

一句话：

```text
标准 Attention 通常是 Q/K/V + paged KV cache；MLA 是 Q + compressed KV + rope 部分的专用 cache 和专用 prefill/decode backend。
```

---

## 21. CUDA Graph 对 attention kernel 的约束

attention backend 会声明 CUDA Graph 支持级别。

枚举：`AttentionCGSupport`。

源码位置：`code/vllm/vllm/v1/attention/backend.py:516`

级别包括：

```text
ALWAYS：
  总是支持，包括 mixed prefill-decode。

UNIFORM_BATCH：
  batch 内 query length 一致时支持。

UNIFORM_SINGLE_TOKEN_DECODE：
  只支持 query_len == 1 decode。

NEVER：
  不支持。
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:521`
- `code/vllm/vllm/v1/attention/backend.py:523`
- `code/vllm/vllm/v1/attention/backend.py:527`
- `code/vllm/vllm/v1/attention/backend.py:529`

FlashAttention builder 对 FA2 / FA3 有不同支持：

- `code/vllm/vllm/v1/attention/backends/flash_attn.py:294`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:313`

TritonAttention builder 声明为 ALWAYS：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:98`

Triton builder 还会为了 CUDA Graph 调整 `seq_threshold_3D` 和 workspace：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:119`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:138`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:150`

---

## 22. Attention backend 的最小对照

```text
FlashAttention：
  适合高性能 varlen attention；依赖 vllm_flash_attn / flash_attn；
  对 compute capability、FA version、head_size、block_size 有明确限制；
  常用于 prefill / mixed 场景。

TritonAttention：
  vLLM 自有 Triton attention；支持更多 fallback 和特殊 dtype；
  通过 unified_attention kernel 支撑 paged KV；
  对 CUDA Graph 支持强，便于作为稳定通用路径。

FlashInfer：
  使用 FlashInfer paged decode / prefill wrapper；
  支持 FP8 / NVFP4 等 KV cache dtype；
  page size 和 TRT-LLM attention 条件会影响路径。

ROCm / AITER：
  AMD 平台专用优化路径；
  可能使用 AITER flash attention、unified attention 或 ROCm paged attention。

MLA / FlashMLA / CUTLASS_MLA / Triton_MLA：
  服务 DeepSeek 类 MLA 结构；
  prefill / decode、compressed KV、sparse MLA 都有专门路径。

CPU_ATTN：
  CPU fallback / CPU 平台路径。
```

---

## 23. 如何读 attention kernel 源码

推荐阅读顺序：

```text
1. Attention.__init__()
   看 backend 如何选择，KV cache dtype 如何确定。

2. Attention.forward()
   看 Q/K/V reshape、KV cache update、opaque op 调用。

3. v1/attention/backend.py
   看 backend / metadata / impl 抽象。

4. v1/attention/selector.py + platforms/cuda.py
   看 backend selection 和 invalid reasons。

5. 具体 backend：flash_attn.py / triton_attn.py / flashinfer.py / mla/*
   看 metadata builder 和 impl.forward。

6. v1/attention/ops/*
   看 Triton wrapper、merge、paged cache writer。

7. csrc/libtorch_stable/attention/*
   看 native paged attention kernel。
```

---

## 24. 常见问题定位

### 24.1 为什么没有走预期 backend？

先看：

```text
- 是否显式设置 attention_config.backend / --attention-backend；
- get_attn_backend() 的 AttentionSelectorConfig；
- platform.get_attn_backend_cls() 的 invalid_reasons；
- head_size / dtype / kv_cache_dtype / block_size；
- compute capability；
- use_mla / use_sparse / has_sink / use_mm_prefix；
- batch invariance / non-causal / KV connector。
```

关键源码：

- `code/vllm/vllm/v1/attention/selector.py:90`
- `code/vllm/vllm/platforms/cuda.py:337`
- `code/vllm/vllm/platforms/cuda.py:370`
- `code/vllm/vllm/platforms/cuda.py:395`

### 24.2 为什么手动 block_size 影响性能？

CUDA selection 中有专门 warning：如果用户指定 `--block-size` 导致高优先级 backend 因 `block_size not supported` 被排除，会警告使用较低优先级 backend。

源码位置：`code/vllm/vllm/platforms/cuda.py:415`

### 24.3 为什么 attention 和 KV cache update 分开？

看：

```text
backend.forward_includes_kv_cache_update
Attention.forward()
unified_kv_cache_update()
impl.do_kv_cache_update()
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:65`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:499`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:692`

### 24.4 为什么 merge_attn_states 走 Triton？

看 output / prefix dtype 和 head_dim 是否满足 native CUDA kernel 条件。

关键源码：

- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:49`
- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:62`
- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:72`

### 24.5 为什么 CUDA Graph capture 失败或慢？

看：

```text
- backend 的 AttentionCGSupport；
- builder.build_for_cudagraph_capture()；
- batch 是否 mixed prefill/decode；
- query_len 是否 uniform；
- workspace 是否提前分配；
- attention 是否被 opaque custom op 包住。
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:516`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:390`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:294`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:119`

---

## 25. 一句话总结

Attention 算子的核心不是单个 kernel，而是一条从 `Attention.forward()` 到 backend selection、metadata builder、KV cache update、prefill/decode kernel、cascade merge、CUDA Graph 支持的完整执行链。

最核心主线是：

```text
ModelRunner attention metadata
  → Attention.forward()
  → get_attn_backend()
  → backend impl.forward()
  → KV cache writer + prefill/decode kernel
  → output tensor
```
