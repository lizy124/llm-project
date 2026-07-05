# 02. Python 层如何调用自定义算子？

源码位置：

- `code/vllm/vllm/_custom_ops.py`
- `code/vllm/vllm/utils/torch_utils.py`
- `code/vllm/vllm/model_executor/layers/`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/v1/attention/ops/`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/platforms/`
- `code/vllm/csrc/torch_bindings.cpp`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp`
- `code/vllm/csrc/`

这个问题关注：Python 层如何通过 wrapper、torch extension、torch.ops、Triton 或 fallback 路径调用底层 native op，以及这些入口如何屏蔽不同平台、backend、dtype、shape 的差异。

---

## 1. 一句话回答

Python 层通常不直接操作 CUDA kernel，而是通过 **薄 wrapper + backend dispatch** 调用 native op。

可以先记成：

```text
Python model layer
  → vLLM op wrapper
  → torch.ops / extension binding / Triton function / third-party wrapper / torch fallback
  → concrete kernel
```

在源码里最常见的 4 条路径是：

```text
1. vllm._custom_ops.*
   → torch.ops._C / torch.ops._rocm_C / torch.ops._moe_C / torch.ops._C_cache_ops

2. torch.ops.vllm.*
   → direct_register_custom_op() 注册的 Python custom op

3. Triton wrapper
   → @triton.jit kernel launch

4. Third-party wrapper
   → FlashAttention / FlashInfer / AITER / zentorch / oneDNN
```

---

## 2. 总览主链路

一个普通模型层进入 native op 的链路通常是：

```text
model_executor/layers/*
  → Python wrapper
  → platform / dtype / shape 判断
  → torch.ops namespace 或 Triton wrapper
  → csrc / Triton / third-party kernel
  → output tensor
```

以 attention 为例：

```text
Attention.forward()
  → torch.ops.vllm.unified_attention_with_output()
  → get_attention_context()
  → self.impl.forward()
  → FlashAttentionImpl / TritonAttentionImpl / FlashInferImpl / MLA impl
  → concrete attention kernel
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:438`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:522`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:736`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:753`

---

## 3. `_custom_ops.py` 的职责

`_custom_ops.py` 是 vLLM Python 层最集中的 native op wrapper 文件。

源码位置：`code/vllm/vllm/_custom_ops.py:1`

它的特点是：

```text
- 大多数函数很薄，只转发到 torch.ops；
- 一部分函数会分配输出 tensor；
- 一部分函数会按平台 / dtype / env flag 选择不同实现；
- 一部分函数注册 fake impl，支持 torch.compile / fake tensor / shape 推断；
- 它屏蔽了 _C / _rocm_C / _moe_C / CPU op namespace 的差异。
```

### 3.1 Attention / KV cache 相关 wrapper

代表函数：

```text
paged_attention_v1()
paged_attention_v2()
paged_attention_rocm()
merge_attn_states()
reshape_and_cache()
reshape_and_cache_flash()
concat_and_cache_mla()
swap_blocks()
gather_and_maybe_dequant_cache()
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:114`
- `code/vllm/vllm/_custom_ops.py:158`
- `code/vllm/vllm/_custom_ops.py:208`
- `code/vllm/vllm/_custom_ops.py:266`
- `code/vllm/vllm/_custom_ops.py:2579`
- `code/vllm/vllm/_custom_ops.py:2601`
- `code/vllm/vllm/_custom_ops.py:2690`
- `code/vllm/vllm/_custom_ops.py:2729`
- `code/vllm/vllm/_custom_ops.py:2794`

典型形式：

```text
_custom_ops.paged_attention_v1(...)
  → torch.ops._C.paged_attention_v1(...)

_custom_ops.paged_attention_rocm(...)
  → torch.ops._rocm_C.paged_attention(...)

_custom_ops.reshape_and_cache(...)
  → torch.ops._C_cache_ops 或 torch.ops._C 里的 cache op
```

### 3.2 Norm / RoPE / activation 相关 wrapper

代表函数：

```text
rotary_embedding()
rms_norm()
fused_add_rms_norm()
fused_qk_norm_rope()
rms_norm_dynamic_per_token_quant()
rms_norm_per_block_quant()
silu_and_mul_per_block_quant()
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:289`
- `code/vllm/vllm/_custom_ops.py:317`
- `code/vllm/vllm/_custom_ops.py:326`
- `code/vllm/vllm/_custom_ops.py:336`
- `code/vllm/vllm/_custom_ops.py:418`
- `code/vllm/vllm/_custom_ops.py:438`
- `code/vllm/vllm/_custom_ops.py:497`

调用方示例：

- `code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:238`
- `code/vllm/vllm/model_executor/layers/activation.py:97`
- `code/vllm/vllm/model_executor/layers/activation.py:133`
- `code/vllm/vllm/model_executor/layers/layernorm.py`

### 3.3 Quantization 相关 wrapper

代表函数：

```text
awq_dequantize()
awq_gemm()
gptq_gemm()
cutlass_scaled_mm()
cutlass_scaled_mm_azp()
marlin_gemm()
machete_mm()
cutlass_w4a8_mm()
scaled_fp4_quant()
scaled_fp8_quant()
scaled_int8_quant()
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:548`
- `code/vllm/vllm/_custom_ops.py:582`
- `code/vllm/vllm/_custom_ops.py:615`
- `code/vllm/vllm/_custom_ops.py:813`
- `code/vllm/vllm/_custom_ops.py:864`
- `code/vllm/vllm/_custom_ops.py:1327`
- `code/vllm/vllm/_custom_ops.py:1422`
- `code/vllm/vllm/_custom_ops.py:1494`
- `code/vllm/vllm/_custom_ops.py:1650`
- `code/vllm/vllm/_custom_ops.py:1955`
- `code/vllm/vllm/_custom_ops.py:2109`

有些 wrapper 会带 fallback，例如 `cutlass_scaled_mm()`：

```text
如果是 ROCm 或 B 矩阵形状不满足 CUTLASS 兼容条件：
  → triton_scaled_mm()
否则：
  → torch.ops._C.cutlass_scaled_mm()
```

源码位置：`code/vllm/vllm/_custom_ops.py:813`

### 3.4 Fused MoE 相关 wrapper

代表函数：

```text
get_cutlass_moe_mm_data()
cutlass_moe_mm()
cutlass_fp4_moe_mm()
cutlass_mxfp4_moe_mm()
moe_align_block_size()
batched_moe_align_block_size()
moe_wna16_gemm()
topk_softmax()
grouped_topk()
moe_sum()
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:904`
- `code/vllm/vllm/_custom_ops.py:1026`
- `code/vllm/vllm/_custom_ops.py:1067`
- `code/vllm/vllm/_custom_ops.py:1107`
- `code/vllm/vllm/_custom_ops.py:2228`
- `code/vllm/vllm/_custom_ops.py:2232`
- `code/vllm/vllm/_custom_ops.py:2252`
- `code/vllm/vllm/_custom_ops.py:2302`
- `code/vllm/vllm/_custom_ops.py:2380`
- `code/vllm/vllm/_custom_ops.py:2440`

调用方示例：

- `code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:596`
- `code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1490`
- `code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1735`

### 3.5 Sampling / logits 相关 wrapper

代表函数：

```text
apply_repetition_penalties()
apply_repetition_penalties_cuda()
apply_repetition_penalties_torch()
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:366`
- `code/vllm/vllm/_custom_ops.py:382`
- `code/vllm/vllm/_custom_ops.py:393`

它展示了最直观的 fallback 模式：

```text
if logits.is_cuda and logits.is_contiguous():
  → CUDA custom op
else:
  → torch 实现
```

源码位置：`code/vllm/vllm/_custom_ops.py:407`

---

## 4. torch.ops namespace 是怎么来的

vLLM 的 C++ / CUDA 扩展通过 Torch Library 注册到不同 namespace。

关键源码：

- `code/vllm/csrc/core/registration.h:20`
- `code/vllm/csrc/torch_bindings.cpp:21`
- `code/vllm/csrc/torch_bindings.cpp:75`
- `code/vllm/csrc/torch_bindings.cpp:93`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:10`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:622`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:781`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:868`

常见 namespace：

```text
torch.ops._C：
  CUDA / composite / stable libtorch 主 op namespace。

torch.ops._rocm_C：
  ROCm 专用 op namespace。

torch.ops._moe_C：
  MoE 相关 op namespace。

torch.ops._C_cache_ops：
  cache 操作相关 namespace。

torch.ops._C_custom_ar：
  custom all-reduce 相关 namespace。

torch.ops.vllm：
  Python 侧 direct_register_custom_op 注册的 vLLM custom op。
```

---

## 5. C++ / CUDA op 注册链路

一个 native op 从 csrc 暴露到 Python，通常经过：

```text
CUDA / C++ kernel implementation
  → C++ launcher / wrapper
  → TORCH_LIBRARY / TORCH_LIBRARY_IMPL 注册 schema 和实现
  → 编译成 torch extension
  → current_platform.import_kernels()
  → Python 使用 torch.ops.<namespace>.<op>() 调用
```

例如 paged attention：

```text
csrc/libtorch_stable/attention/attention_kernels.cuh
  → paged_attention_v1_kernel
  → csrc/libtorch_stable/attention/paged_attention_v1.cu
  → paged_attention_v1 launcher
  → csrc/libtorch_stable/torch_bindings.cpp 注册 paged_attention_v1
  → Python _custom_ops.paged_attention_v1()
  → torch.ops._C.paged_attention_v1()
```

关键源码：

- `code/vllm/csrc/libtorch_stable/attention/attention_kernels.cuh:494`
- `code/vllm/csrc/libtorch_stable/attention/paged_attention_v1.cu:47`
- `code/vllm/csrc/libtorch_stable/attention/paged_attention_v1.cu:164`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:598`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:737`
- `code/vllm/vllm/_custom_ops.py:114`

例如 RMSNorm：

```text
csrc/libtorch_stable/layernorm_kernels.cu
  → rms_norm_kernel
  → rms_norm launcher
  → csrc/libtorch_stable/torch_bindings.cpp 注册 rms_norm
  → Python _custom_ops.rms_norm()
  → torch.ops._C.rms_norm()
```

关键源码：

- `code/vllm/csrc/libtorch_stable/layernorm_kernels.cu:15`
- `code/vllm/csrc/libtorch_stable/layernorm_kernels.cu:211`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:372`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:668`
- `code/vllm/vllm/_custom_ops.py:317`

---

## 6. `current_platform.import_kernels()` 的作用

`_custom_ops.py` 在导入时会调用：

```text
current_platform.import_kernels()
```

源码位置：`code/vllm/vllm/_custom_ops.py:20`

它的作用是让当前平台加载对应 extension，使 `torch.ops._C` / `_rocm_C` / CPU ops 等 namespace 可用。

这也是为什么很多 wrapper 可以直接调用 `torch.ops._C.*`：

```text
先由 platform 完成 extension import；
然后 torch.ops namespace 中才会出现对应 op。
```

不同平台会有不同导入逻辑，例如：

```text
CUDA：导入 CUDA extension；
ROCm：导入 ROCm extension / AITER；
CPU：导入 CPU extension；
XPU：导入 XPU ops；
TPU：可能走不同路径或 fallback。
```

---

## 7. `direct_register_custom_op()`：Python 函数注册成 torch op

除了 C++ / CUDA extension，vLLM 还会把 Python 函数注册成 `torch.ops.vllm.*`。

注册工具源码：

- `code/vllm/vllm/utils/torch_utils.py:927`
- `code/vllm/vllm/utils/torch_utils.py:931`

关键逻辑：

```text
vllm_lib = Library("vllm", "FRAGMENT")

direct_register_custom_op(...)
  → infer_schema(op_func, mutates_args)
  → my_lib.define(op_name + schema)
  → my_lib.impl(op_name, op_func, dispatch_key=current_platform.dispatch_key)
  → my_lib._register_fake(op_name, fake_impl)
```

它的设计目标是：

```text
- 比 torch.library.custom_op 更低调度开销；
- 允许指定 mutates_args；
- 按当前平台 dispatch_key 注册实现；
- 支持 fake_impl；
- 给 torch.compile / CUDA Graph 提供稳定 op 边界。
```

### 7.1 Attention opaque op

Attention 层会注册三个关键 custom op：

```text
maybe_calc_kv_scales
unified_kv_cache_update
unified_attention_with_output
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:614`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:641`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:692`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:726`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:736`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:779`

调用链是：

```text
Attention.forward()
  → torch.ops.vllm.maybe_calc_kv_scales()
  → torch.ops.vllm.unified_kv_cache_update()        # 某些 backend 需要拆出 KV update
  → torch.ops.vllm.unified_attention_with_output()
  → get_attention_context()
  → self.impl.forward(...)
```

关键点：

```text
unified_attention_with_output 自身不是最终 kernel；
它是一个 opaque op 边界，内部再调用具体 attention impl。
```

### 7.2 为什么要注册成 opaque op

Attention 层注释解释了原因：

```text
对于 CUDA-like 和 CPU 平台，vLLM 控制 torch.compile 的方式是把 attention 注册成一个大的 opaque custom op；
其他平台则直接调用，让 torch.compile 自己处理。
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:390`

这可以减少 hot path 上的 Python / dispatcher 开销，也能让 compile / CUDA Graph 看到稳定边界。

---

## 8. fake impl / register_fake 的作用

很多 custom op 旁边会有 fake implementation。

典型源码：

- `code/vllm/vllm/_custom_ops.py:89`
- `code/vllm/vllm/_custom_ops.py:91`
- `code/vllm/vllm/_custom_ops.py:565`
- `code/vllm/vllm/_custom_ops.py:596`
- `code/vllm/vllm/_custom_ops.py:637`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:632`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:718`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:766`

它的作用是：

```text
- 在 torch.compile / fake tensor / symbolic shape 阶段返回正确 shape / dtype；
- 避免编译阶段真的执行 CUDA kernel；
- 告诉 PyTorch 哪些参数会被 mutation；
- 支持 out variant 的 buffer 管理。
```

例如 `scaled_fp4_quant` 会注册 fake impl，根据 input shape 创建 fake output 和 scale tensor：

- `code/vllm/vllm/_custom_ops.py:89`
- `code/vllm/vllm/_custom_ops.py:91`

Attention 的 `unified_attention_with_output_fake()` 则直接返回 `None`，因为真正输出 tensor 是传入的 `output`，op 原地写入。

- `code/vllm/vllm/model_executor/layers/attention/attention.py:766`

---

## 9. wrapper 如何做 fallback

vLLM 的 Python wrapper 常见 fallback 条件包括：

```text
- 平台不是 CUDA；
- dtype 不支持；
- head_size / block_size 不满足 kernel 限制；
- tensor 非 contiguous；
- output dtype 不支持 native kernel；
- extension import 失败；
- 用户通过 env flag 强制使用 Triton 或禁用某 backend；
- ROCm / XPU / CPU 需要不同实现。
```

### 9.1 repetition penalties：CUDA / torch fallback

源码位置：`code/vllm/vllm/_custom_ops.py:393`

逻辑：

```text
if logits.is_cuda and logits.is_contiguous():
  → torch.ops._C.apply_repetition_penalties_()
else:
  → torch.where 实现的 torch fallback
```

相关源码：

- `code/vllm/vllm/_custom_ops.py:366`
- `code/vllm/vllm/_custom_ops.py:382`
- `code/vllm/vllm/_custom_ops.py:407`

### 9.2 AWQ：env flag 切换 Triton / CUDA

源码位置：`code/vllm/vllm/_custom_ops.py:548`

逻辑：

```text
if envs.VLLM_USE_TRITON_AWQ:
  → awq_dequantize_triton / awq_gemm_triton
else:
  → torch.ops._C.awq_dequantize / awq_gemm
```

相关源码：

- `code/vllm/vllm/_custom_ops.py:548`
- `code/vllm/vllm/_custom_ops.py:582`

### 9.3 CUTLASS scaled mm：ROCm / shape fallback 到 Triton

源码位置：`code/vllm/vllm/_custom_ops.py:813`

逻辑：

```text
如果是 ROCm 或 b 矩阵不满足 CUTLASS 16 对齐：
  → triton_scaled_mm()
否则：
  → torch.ops._C.cutlass_scaled_mm()
```

相关源码：

- `code/vllm/vllm/_custom_ops.py:850`
- `code/vllm/vllm/_custom_ops.py:852`
- `code/vllm/vllm/_custom_ops.py:858`

### 9.4 merge_attn_states：dtype / headdim fallback 到 Triton

源码位置：`code/vllm/vllm/v1/attention/ops/merge_attn_states.py:9`

逻辑：

```text
如果是 CUDA，且 dtype 是 fp32/fp16/bf16，且 head_dim 满足 native kernel pack size：
  → _custom_ops.merge_attn_states()
否则：
  → triton_merge_attn_states.merge_attn_states()
```

相关源码：

- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:49`
- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:72`
- `code/vllm/vllm/v1/attention/ops/merge_attn_states.py:90`

---

## 10. Triton op 的 Python 调用方式

Triton kernel 通常不是通过 `torch.ops._C` 调用，而是 Python wrapper 直接 launch。

以 Triton attention 为例：

```text
TritonAttentionImpl.forward()
  → unified_attention(...)
  → kernel_unified_attention[grid](...)
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:530`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:642`
- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:179`
- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:780`
- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:1019`

KV cache 写入的 Triton 路径：

```text
TritonAttentionImpl.do_kv_cache_update()
  → triton_reshape_and_cache_flash()
  → reshape_and_cache_kernel_flash[grid](...)
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:724`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:758`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:33`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:334`
- `code/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:405`

---

## 11. 第三方 kernel wrapper 的调用方式

### 11.1 FlashAttention

FlashAttention 的平台封装在 `fa_utils.py`。

关键源码：

- `code/vllm/vllm/v1/attention/backends/fa_utils.py:18`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:20`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:56`
- `code/vllm/vllm/v1/attention/backends/fa_utils.py:231`

来源按平台变化：

```text
CUDA：
  vllm.vllm_flash_attn.flash_attn_varlen_func

XPU：
  xpu_ops.flash_attn_varlen_func

ROCm：
  upstream flash_attn.flash_attn_varlen_func，如果安装可用
```

### 11.2 FlashInfer

FlashInfer 直接 import wrapper 类：

```text
BatchDecodeWithPagedKVCacheWrapper
BatchPrefillWithPagedKVCacheWrapper
BatchPrefillWithRaggedKVCacheWrapper
MultiLevelCascadeAttentionWrapper
```

源码位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:11`

FlashInfer backend 会根据 page size、KV dtype、TRT-LLM attention 是否可用等条件选择内部路径。

关键源码：

- `code/vllm/vllm/v1/attention/backends/flashinfer.py:325`
- `code/vllm/vllm/v1/attention/backends/flashinfer.py:337`
- `code/vllm/vllm/v1/attention/backends/flashinfer.py:354`

### 11.3 AITER / ROCm

ROCm 相关路径分布在：

```text
vllm/_aiter_ops.py
vllm/v1/attention/backends/rocm_aiter_fa.py
vllm/v1/attention/backends/rocm_aiter_unified_attn.py
vllm/v1/attention/backends/mla/rocm_aiter_mla.py
```

这些路径通常由 platform / backend selection 选择。

---

## 12. Python wrapper 和 platform 的关系

平台影响算子路径的方式主要有两种。

### 12.1 import kernels

`_custom_ops.py` 导入时执行：

```text
current_platform.import_kernels()
```

源码位置：`code/vllm/vllm/_custom_ops.py:20`

这决定对应 torch extension 是否被加载。

### 12.2 backend selection

Attention backend 选择通过 platform 实现：

```text
get_attn_backend()
  → current_platform.get_attn_backend_cls(...)
  → validate_configuration(...)
  → 返回 backend class path
```

关键源码：

- `code/vllm/vllm/v1/attention/selector.py:54`
- `code/vllm/vllm/v1/attention/selector.py:121`
- `code/vllm/vllm/platforms/cuda.py:351`

平台会影响：

```text
- 可用 backend 列表和优先级；
- compute capability；
- dispatch_key；
- 是否 opaque attention op；
- FP8 dtype；
- 是否支持 pinned memory / CUDA Graph / custom all-reduce；
- 是否使用 ROCm / XPU / CPU 专用 op。
```

---

## 13. Python 到 native op 的典型源码路径

### 13.1 RMSNorm

```text
RMSNorm layer
  → ops.rms_norm(out, input, weight, epsilon)
  → torch.ops._C.rms_norm(...)
  → csrc/libtorch_stable/layernorm_kernels.cu
```

关键源码：

- `code/vllm/vllm/model_executor/layers/layernorm.py`
- `code/vllm/vllm/_custom_ops.py:317`
- `code/vllm/csrc/libtorch_stable/layernorm_kernels.cu:15`
- `code/vllm/csrc/libtorch_stable/layernorm_kernels.cu:211`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:372`
- `code/vllm/csrc/libtorch_stable/torch_bindings.cpp:668`

### 13.2 RoPE

```text
RotaryEmbedding.forward()
  → ops.rotary_embedding(...)
  → torch.ops._C.rotary_embedding(...)
```

关键源码：

- `code/vllm/vllm/model_executor/layers/rotary_embedding/base.py:238`
- `code/vllm/vllm/_custom_ops.py:289`

### 13.3 Attention opaque op

```text
Attention.forward()
  → torch.ops.vllm.unified_attention_with_output(...)
  → unified_attention_with_output()
  → self.impl.forward(...)
  → backend-specific kernel
```

关键源码：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:438`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:522`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:736`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:753`

### 13.4 Triton attention

```text
TritonAttentionImpl.forward()
  → unified_attention()
  → kernel_unified_attention[grid](...)
```

关键源码：

- `code/vllm/vllm/v1/attention/backends/triton_attn.py:530`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py:642`
- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:179`
- `code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:780`

### 13.5 Quantized GEMM

```text
quantized linear method
  → ops.cutlass_scaled_mm() / ops.gptq_gemm() / ops.awq_gemm()
  → torch.ops._C.* 或 Triton fallback
```

关键源码：

- `code/vllm/vllm/_custom_ops.py:582`
- `code/vllm/vllm/_custom_ops.py:615`
- `code/vllm/vllm/_custom_ops.py:813`
- `code/vllm/vllm/model_executor/layers/quantization/`

### 13.6 MoE grouped GEMM

```text
fused_experts()
  → torch.ops.vllm.fused_experts() 或 ops.cutlass_moe_mm()
  → _moe_C / _C grouped GEMM kernels
```

关键源码：

- `code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1418`
- `code/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1490`
- `code/vllm/vllm/_custom_ops.py:904`
- `code/vllm/vllm/_custom_ops.py:1026`

---

## 14. 如何确认实际调用路径

定位某个 op 实际走哪条路径时，按这个顺序查：

```text
1. 从 layer.forward() 找调用入口；
2. 如果调用 ops.xxx，去 _custom_ops.py 看 wrapper；
3. 如果调用 torch.ops.vllm.xxx，找 direct_register_custom_op 注册点；
4. 如果是 attention，先看 get_attn_backend() 选了哪个 backend；
5. 看 wrapper 是否有 platform / dtype / shape / env fallback；
6. 再看 csrc torch_bindings 或 Triton wrapper；
7. 最后看 CUDA / Triton kernel。
```

常见判断点：

```text
- current_platform.is_cuda() / is_rocm() / is_xpu() / is_cpu()
- envs.VLLM_USE_TRITON_AWQ
- attention_config.backend
- kv_cache_dtype
- head_size / block_size
- tensor.is_cuda / is_contiguous
- output dtype
- compute capability
```

---

## 15. 一句话总结

Python 层调用 native op 的核心不是“直接调 CUDA kernel”，而是通过 `_custom_ops.py`、`torch.ops.vllm`、backend impl、Triton wrapper 和第三方 wrapper 组成一层分发面；这层负责整理参数、选择实现、处理 fallback、支持 compile / CUDA Graph，并最终把模型层计算落到具体 kernel 上。

最核心链路是：

```text
model layer
  → Python wrapper
  → torch.ops / Triton / third-party wrapper
  → native kernel
  → output tensor
```
