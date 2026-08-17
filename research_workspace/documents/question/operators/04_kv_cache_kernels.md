# 04. KV Cache 相关算子负责什么？

源码位置：

- `code/vllm/vllm/_custom_ops.py`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/attention/ops/paged_attn.py`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py`
- `code/vllm/vllm/v1/attention/backends/flashinfer.py`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu`
- `code/vllm/csrc/libtorch_stable/cache_kernels_fused.cu`
- `code/vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp`

这个问题关注：KV cache 写入、复制、reshape、swap、gather、block 操作、slot mapping、KV cache quantization 等底层算子如何把逻辑 token 位置映射到物理 cache 张量。

---

## 1. 一句话回答

KV cache 算子负责把模型产生的 K/V 状态写入、搬移或读取到正确的 cache block 位置。

最小链路是：

```text
Scheduler 分配 block
  → SchedulerOutput 携带 block / cache 相关信息
  → ModelRunner 准备 slot_mapping / block_table
  → model forward 产生 K/V
  → KV cache kernel 按 slot_mapping 写入 cache
  → attention kernel 按 block_table 读取 cache
```

如果只记一个边界：

```text
slot_mapping 管“当前 token 写到哪里”，block_table 管“attention 读历史上下文时从哪些 block 读”。
```

---

## 2. KV cache 算子层处于哪一层

KV cache 算子不负责分配 cache block，也不负责调度请求。

职责边界是：

```text
Scheduler / KVCacheManager：
  决定哪些 request 拿到哪些 logical / physical blocks。

ModelRunner：
  根据 SchedulerOutput 和当前 batch 生成 slot_mapping、block_table、seq_lens、query_start_loc。

Attention layer / backend：
  使用这些 metadata 调用 cache 写入和 attention 读取 kernel。

KV cache kernels：
  真正执行 reshape、cache write、copy、swap、gather、dequant。
```

这和执行层文档的关系是：

```text
../executor_worker_model_runner/06_prepare_inputs_and_attention_metadata.md
  → 准备 slot_mapping / block table

../executor_worker_model_runner/09_worker_kv_cache_interaction.md
  → 解释 Worker / ModelRunner 如何使用 KV cache

operators/04_kv_cache_kernels.md
  → 解释这些状态最后如何进入底层 cache op
```

---

## 3. KV cache 的核心数据结构

### 3.1 KVCacheSpec

KV cache 规格定义在 `kv_cache_interface.py`。

源码位置：`code/vllm/vllm/v1/kv_cache_interface.py:95`

`KVCacheSpec` 是所有 cache spec 的基类，至少包含：

```text
block_size：
  一个 cache block / page 包含多少 token。

page_size_bytes：
  一个 page 实际需要多少字节。

storage_block_size：
  存储层的 block size，默认等于 block_size。
```

关键源码：

- `code/vllm/vllm/v1/kv_cache_interface.py:95`
- `code/vllm/vllm/v1/kv_cache_interface.py:101`
- `code/vllm/vllm/v1/kv_cache_interface.py:104`
- `code/vllm/vllm/v1/kv_cache_interface.py:115`

### 3.2 AttentionSpec

标准 attention 的 KV cache 规格是 `AttentionSpec`。

源码位置：`code/vllm/vllm/v1/kv_cache_interface.py:159`

核心字段：

```text
- block_size
- num_kv_heads
- head_size
- dtype
- kv_quant_mode
- page_size_padded
```

`page_size_bytes` 会根据 KV quantization 模式调整内存估算。

关键源码：

- `code/vllm/vllm/v1/kv_cache_interface.py:159`
- `code/vllm/vllm/v1/kv_cache_interface.py:167`
- `code/vllm/vllm/v1/kv_cache_interface.py:182`

### 3.3 KVQuantMode

KV cache quantization mode 用 enum 表示，避免 kernel 内反复 string matching。

源码位置：`code/vllm/vllm/v1/kv_cache_interface.py:33`

支持模式：

```text
NONE = 0
FP8_PER_TENSOR = 1
INT8_PER_TOKEN_HEAD = 2
FP8_PER_TOKEN_HEAD = 3
NVFP4 = 4
```

映射函数：

- `code/vllm/vllm/v1/kv_cache_interface.py:60`
- `code/vllm/vllm/v1/kv_cache_interface.py:73`
- `code/vllm/vllm/v1/kv_cache_interface.py:77`

一句话：

```text
kv_cache_dtype 是用户/配置层字符串，KVQuantMode 是 attention backend 和 kernel 更容易消费的内部模式。
```

---

## 4. KV cache tensor layout 谁决定

不同 attention backend 会声明自己的 KV cache shape 和 stride order。

抽象接口：

- `code/vllm/vllm/v1/attention/backend.py:88`
- `code/vllm/vllm/v1/attention/backend.py:118`

以 FlashAttention 为例：

```text
get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
  → (num_blocks, 2, block_size, num_kv_heads, head_size)
```

源码位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:140`

FlashAttention 还会根据全局 cache layout 返回 stride order：

- `code/vllm/vllm/v1/attention/backends/flash_attn.py:151`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:157`

TritonAttention 也定义了类似 shape，并在 per-token-head quant 时给 head dimension 追加 scale padding：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:294`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:303`

因此：

```text
KV cache 算子不能脱离 backend layout 理解。
同样是 “KV cache”，FlashAttention / Triton / MLA / NVFP4 可能有不同物理布局。
```

---

## 5. Python wrapper：_custom_ops.py 的 cache op

KV cache native op 的 Python 入口集中在 `_custom_ops.py`。

关键 wrapper：

```text
reshape_and_cache()
reshape_and_cache_flash()
concat_and_cache_mla()
concat_and_cache_mla_rope_fused()
swap_blocks()
swap_blocks_batch()
convert_fp8()
gather_and_maybe_dequant_cache()
cp_gather_cache()
cp_gather_and_upconvert_fp8_kv_cache()
```

源码位置：

- `code/vllm/vllm/_custom_ops.py:2579`
- `code/vllm/vllm/_custom_ops.py:2601`
- `code/vllm/vllm/_custom_ops.py:2690`
- `code/vllm/vllm/_custom_ops.py:2703`
- `code/vllm/vllm/_custom_ops.py:2729`
- `code/vllm/vllm/_custom_ops.py:2758`
- `code/vllm/vllm/_custom_ops.py:2788`
- `code/vllm/vllm/_custom_ops.py:2794`
- `code/vllm/vllm/_custom_ops.py:2818`
- `code/vllm/vllm/_custom_ops.py:2831`

这些 wrapper 主要转发到：

```text
torch.ops._C_cache_ops.*
```

例如：

```text
reshape_and_cache()
  → torch.ops._C_cache_ops.reshape_and_cache(...)

reshape_and_cache_flash()
  → torch.ops._C_cache_ops.reshape_and_cache_flash(...)

swap_blocks()
  → torch.ops._C_cache_ops.swap_blocks(...)
```

---

## 6. C++ / CUDA 注册入口

`_C_cache_ops` 的 schema 和实现注册在 stable torch bindings。

源码位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:781`

schema 包括：

```text
swap_blocks
swap_blocks_batch
reshape_and_cache
reshape_and_cache_flash
concat_and_cache_mla
concat_and_cache_mla_rope_fused
convert_fp8
gather_and_maybe_dequant_cache
cp_gather_cache
cp_gather_and_upconvert_fp8_kv_cache
indexer_k_quant_and_cache
concat_mla_q
cp_gather_indexer_k_quant_cache
```

关键源码：

- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:781`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:783`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:793`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:801`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:810`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:818`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:832`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:837`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:847`

CUDA 实现绑定：

- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:911`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:912`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:913`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:914`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:915`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:918`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:919`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:921`

CPU 侧目前注册了 `swap_blocks_batch`：

- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:907`

---

## 7. native cache kernel 文件

核心 native 实现在：

```text
code/vllm/csrc/libtorch_stable/cache_kernels.cu
code/vllm/csrc/libtorch_stable/cache_kernels_fused.cu
code/vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu
```

关键 kernel / launcher：

```text
swap_blocks()
swap_blocks_batch()
copy_blocks_kernel()
reshape_and_cache_kernel()
reshape_and_cache_flash_kernel()
concat_and_cache_mla_kernel()
concat_and_cache_ds_mla_kernel()
indexer_k_quant_and_cache_kernel()
cp_gather_indexer_k_quant_cache_kernel()
reshape_and_cache()
reshape_and_cache_flash()
concat_and_cache_mla()
convert_fp8_kernel()
convert_fp8()
gather_and_maybe_dequant_cache()
cp_gather_and_upconvert_fp8_kv_cache()
cp_gather_cache()
reshape_and_cache_nvfp4_kernel()
reshape_and_cache_nvfp4_dispatch()
```

源码位置：

- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:33`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:79`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:186`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:250`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:310`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:398`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:442`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:545`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:609`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:696`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:742`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:838`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:904`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:924`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:988`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:1095`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:1165`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:1233`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:1311`
- `code/vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:49`
- `code/vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:184`

---

## 8. slot_mapping：写入 cache 的关键

`slot_mapping` 表示当前 step 产生的每个 token 应该写入 KV cache 的哪个物理 slot。

在 cache 写入 op 中，它是必需输入：

```text
reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, ...)
reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, ...)
concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, ...)
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:2584`
- `code/vllm/vllm/_custom_ops.py:2606`
- `code/vllm/vllm/_custom_ops.py:2694`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:797`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:806`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:814`

在 `Attention.forward()` 中，slot_mapping 不是显式参数，而是从 forward context 中按 layer 取出：

```text
get_attention_context(layer_name)
  → forward_context.slot_mapping
  → layer_slot_mapping = slot_mapping.get(layer_name)
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:649`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:671`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:684`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:688`

然后：

```text
unified_kv_cache_update()
  → attn_layer.impl.do_kv_cache_update(..., layer_slot_mapping)
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:692`

---

## 9. block_table：attention 读取 cache 的关键

`block_table` / `block_table_tensor` 描述每个 request 的逻辑 block 到物理 KV cache block 的映射。

它在 `CommonAttentionMetadata` 中保存：

源码位置：`code/vllm/vllm/v1/attention/backend.py:387`

attention backend 会把它放进 backend-specific metadata。

FlashAttention metadata：

- `code/vllm/vllm/v1/attention/backends/flash_attn.py:252`

Triton metadata：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:73`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:196`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:222`

Triton attention forward 把它传给 kernel wrapper：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:632`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:656`

Paged attention native wrapper 也直接需要 block table：

- `code/vllm/vllm/_custom_ops.py:121`
- `code/vllm/vllm/_custom_ops.py:168`

一句话：

```text
slot_mapping 是写路径，block_table 是读路径。
```

---

## 10. 标准 Attention 的 cache 写入主线

标准 Attention 的执行入口：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:438`

如果 backend 的 forward 不包含 KV cache update，`Attention.forward()` 会先调用：

```text
unified_kv_cache_update(key, value, layer_name)
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:491`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:499`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:519`

`unified_kv_cache_update()` 的逻辑：

```text
get_attention_context(layer_name)
  → 取 attn_layer / kv_cache / layer_slot_mapping
  → 如果 layer_slot_mapping 不为空
  → 调用 attn_layer.impl.do_kv_cache_update(...)
  → 返回 dummy tensor 建立数据依赖
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:692`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:701`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:707`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:715`

dummy tensor 的意义是：

```text
确保 torch.compile 保留 KV cache update 和 attention forward 的执行顺序。
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:697`

---

## 11. PagedAttention 写入路径

PagedAttention Python wrapper：

源码位置：`code/vllm/vllm/v1/attention/ops/paged_attn.py:15`

`write_to_paged_cache()`：

```text
PagedAttention.write_to_paged_cache()
  → ops.reshape_and_cache(
       key,
       value,
       key_cache,
       value_cache,
       slot_mapping.flatten(),
       kv_cache_dtype,
       k_scale,
       v_scale,
    )
```

关键源码：

- `code/vllm/vllm/v1/attention/ops/paged_attn.py:31`
- `code/vllm/vllm/v1/attention/ops/paged_attn.py:42`

它适合标准 paged KV cache 写入。

---

## 12. Flash / Triton cache 写入路径

TritonAttention 的 cache update：

```text
TritonAttentionImpl.do_kv_cache_update()
  → 如果是 encoder attention，直接 return
  → 如果是 per-token-head quant，调用 triton_reshape_and_cache_flash_per_token_head_quant()
  → 否则调用 triton_reshape_and_cache_flash()
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:724`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:732`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:737`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:743`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:753`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:758`

Triton cache writer：

- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:33`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:262`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:334`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:405`

FlashAttention 相关工具在 `fa_utils.py` 中根据平台提供 `reshape_and_cache_flash`：

```text
CUDA：from vllm._custom_ops import reshape_and_cache_flash
XPU：ops.reshape_and_cache_flash
ROCm：ops.reshape_and_cache_flash
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/fa_utils.py:18`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:25`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:32`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:50`

---

## 13. MLA cache 写入路径

MLA 的 cache 不是标准 K/V 两个 tensor，而是 compressed KV 表示。

抽象方法在 `MLAAttentionImpl.do_kv_cache_update()`：

```text
ops.concat_and_cache_mla(
  kv_c_normed,
  k_pe.squeeze(1),
  kv_cache,
  slot_mapping.flatten(),
  kv_cache_dtype,
  k_scale,
)
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:931`
- `code/vllm/vllm/v1/attention/backend.py:944`
- `code/vllm/vllm/_custom_ops.py:2690`

另外还有融合 RoPE 的 MLA cache 写入：

```text
concat_and_cache_mla_rope_fused()
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:2703`
- `code/vllm/csrc/libtorch_stable/cache_kernels_fused.cu:24`
- `code/vllm/csrc/libtorch_stable/cache_kernels_fused.cu:217`

这条路径用于把 RoPE 和 MLA cache insert 合并，减少单独 kernel / memory pass。

---

## 14. swap_blocks / swap_blocks_batch

`swap_blocks()` 用于按 block 粒度在两个 cache tensor 之间复制。

源码位置：`code/vllm/vllm/_custom_ops.py:2729`

语义：

```text
src / dst 被视为连续 block 序列；
block_mapping shape = (num_blocks_to_copy, 2)；
block_mapping[i][0] 是 source block；
block_mapping[i][1] 是 destination block。
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:2735`
- `code/vllm/vllm/_custom_ops.py:2743`
- `code/vllm/vllm/_custom_ops.py:2751`

`swap_blocks_batch()` 是批量版本，用一次 driver call 提交多段 raw pointer copy。

源码位置：`code/vllm/vllm/_custom_ops.py:2758`

特点：

```text
- src_ptrs / dst_ptrs / sizes 都是 CPU tensor；
- CUDA 12.8+ 可用 cuMemcpyBatchAsync；
- 旧 CUDA fallback 到 cudaMemcpyAsync 循环；
- XPU 有不同调用签名。
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:2764`
- `code/vllm/vllm/_custom_ops.py:2771`
- `code/vllm/vllm/_custom_ops.py:2780`

native 实现入口：

- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:33`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:79`

---

## 15. convert_fp8 和 KV cache quantization

`convert_fp8()` 用于把 cache 转成 FP8 存储。

Python wrapper：`code/vllm/vllm/_custom_ops.py:2788`

native schema：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:832`

native kernel：

- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:904`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:924`

KV cache quantization 对 page size 和 kernel 路径都有影响。

`AttentionSpec.page_size_bytes`：

```text
- per-token-head scales 会额外预算 2 * block_size * num_kv_heads * float32_size；
- NVFP4 会使用 packed fp4 data + fp8 block scales 的 full_dim。
```

关键源码：

- `code/vllm/vllm/v1/kv_cache_interface.py:167`
- `code/vllm/vllm/v1/kv_cache_interface.py:173`
- `code/vllm/vllm/v1/kv_cache_interface.py:184`

NVFP4 专用 cache writer：

- `code/vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:49`
- `code/vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:184`

---

## 16. gather cache / dequant cache

`gather_and_maybe_dequant_cache()` 用于从 paged cache 中 gather token，并在需要时 dequant 到目标 dtype。

Python wrapper：`code/vllm/vllm/_custom_ops.py:2794`

输入包括：

```text
- src_cache
- dst
- block_table
- cu_seq_lens
- token_to_seq
- num_tokens
- kv_cache_dtype
- scale
- seq_starts
```

schema：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:837`

native kernel / launcher：

- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:988`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:1095`

CP / DCP 相关 gather：

```text
cp_gather_cache()
cp_gather_and_upconvert_fp8_kv_cache()
cp_gather_indexer_k_quant_cache()
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:2818`
- `code/vllm/vllm/_custom_ops.py:2831`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:1165`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:1233`
- `code/vllm/csrc/libtorch_stable/cache_kernels.cu:1311`

这些路径通常服务于 context parallel、MLA、sparse attention 或需要把局部 cache gather 到 workspace 的场景。

---

## 17. cache op 和 attention kernel 的关系

KV cache op 和 attention op 是配套关系，不是替代关系。

典型 decode / prefill 执行中：

```text
1. 当前 token 的 K/V 先通过 cache writer 写入 KV cache；
2. attention kernel 通过 block_table + seq_lens 读取历史 KV cache；
3. 对 prefill / mixed batch，可能还要合并 prefix / suffix attention states；
4. 对 quantized KV cache，读取时可能需要 dequant 或使用 kernel 内量化路径。
```

Attention.forward 中这两个阶段被显式串起来：

```text
unified_kv_cache_update()
  → dummy dependency
  → unified_attention_with_output()
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:499`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:502`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:519`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:522`

---

## 18. CUDA Graph 下的 cache 地址稳定性

CUDA Graph / torch.compile 下，KV cache op 需要满足：

```text
- cache tensor 地址稳定；
- output / workspace 预分配；
- kernel 顺序不能被编译器错误重排；
- shape 尽量稳定；
- fallback 分支不能随 batch 随意变化。
```

`unified_kv_cache_update()` 返回 dummy tensor 就是为顺序依赖服务：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:697`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:715`

Attention 层把 attention 包成 opaque custom op 也和 graph / compile 稳定性有关：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:390`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:779`

---

## 19. 常见问题排查

### 19.1 shape mismatch

重点检查：

```text
- backend.get_kv_cache_shape()；
- block_size 是否满足 backend 要求；
- num_kv_heads / head_size 是否和模型一致；
- kv_cache_dtype 是否改变 head dim 或 page size；
- per-token-head / NVFP4 是否需要额外 scale storage。
```

关键源码：

- `code/vllm/vllm/v1/attention/backend.py:88`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:140`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:294`
- `code/vllm/vllm/v1/kv_cache_interface.py:167`

### 19.2 illegal memory access

重点检查：

```text
- slot_mapping 是否越界；
- block_table 是否包含无效 block id；
- block_size_in_bytes 是否和 cache block 实际大小一致；
- KV cache tensor stride / layout 是否符合 backend；
- quantized cache 是否用正确 view / dtype 读取。
```

相关入口：

- `code/vllm/vllm/_custom_ops.py:2579`
- `code/vllm/vllm/_custom_ops.py:2729`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:590`

### 19.3 prefix cache 命中但结果异常

重点检查：

```text
- block_table 是否正确指向 prefix blocks；
- common_prefix_len / cascade metadata 是否正确；
- merge_attn_states 是否使用正确 dtype / head_dim 路径；
- slot_mapping 是否只写当前新 token。
```

相关入口：

- `code/vllm/vllm/v1/attention/backend.py:387`
- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:9`

### 19.4 quantized KV cache 输出异常

重点检查：

```text
- kv_cache_dtype → KVQuantMode 映射；
- k_scale / v_scale 是否加载或初始化；
- calculate_kv_scales 是否只执行一次；
- gather_and_maybe_dequant_cache 是否用正确 scale；
- FP8 / NVFP4 是否使用匹配 kernel。
```

相关入口：

- `code/vllm/vllm/v1/kv_cache_interface.py:60`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:95`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:122`
- `code/vllm/vllm/_custom_ops.py:2788`
- `code/vllm/vllm/_custom_ops.py:2794`

---

## 20. 推荐阅读路线

### 20.1 从系统侧理解

```text
../executor_worker_model_runner/09_worker_kv_cache_interaction.md
  → 04_kv_cache_kernels.md
  → 03_attention_kernels.md
```

### 20.2 从写入路径理解

```text
model_executor/layers/attention/attention.py
  → unified_kv_cache_update()
  → backend impl do_kv_cache_update()
  → _custom_ops.reshape_and_cache*()
  → csrc/libtorch_stable/cache_kernels.cu
```

### 20.3 从读取路径理解

```text
CommonAttentionMetadata.block_table_tensor
  → backend-specific metadata.block_table
  → attention impl.forward()
  → paged / flash / triton / flashinfer attention kernel
```

---

## 21. 一句话总结

KV cache 算子的核心职责是把 ModelRunner 准备好的 `slot_mapping`、`block_table`、`kv_cache_dtype` 和 backend-specific layout，落实成真实的 cache 写入、搬移、gather、dequant 和读取路径；它是 Scheduler 的 block 分配和 Attention kernel 的 paged 读取之间的底层桥梁。

最核心主线是：

```text
block allocation
  → slot_mapping / block_table
  → reshape_and_cache / concat_and_cache / swap / gather
  → attention reads paged KV cache
```
