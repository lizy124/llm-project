# 05 量化、GEMM、CUTLASS、Marlin、Machete

## 1. 量化/GEMM 为什么是 vLLM 底层重点

LLM 推理的主要计算量来自矩阵乘：

- attention QKV/O projection
- MLP up/gate/down projection
- MoE experts GEMM
- router GEMM

为了提升吞吐、降低显存占用，vLLM 支持大量量化格式和 GEMM kernel：

- INT8 / W8A8
- FP8 / block FP8
- FP4 / NVFP4 / MXFP4
- AWQ
- GPTQ
- W4A8
- WNA16
- Marlin
- Machete
- CUTLASS scaled_mm / grouped GEMM
- AllSpark W8A16
- DeepSeek/MiniMax 特化 kernel

## 2. 主要源码目录

| 路径 | 内容 |
|---|---|
| `csrc/libtorch_stable/quantization/w8a8/` | INT8/FP8 W8A8、CUTLASS scaled_mm/grouped_mm |
| `csrc/libtorch_stable/quantization/fp4/` | FP4/NVFP4/MXFP4 quant/mm/MoE |
| `csrc/libtorch_stable/quantization/awq/` | AWQ dequant/gemm |
| `csrc/libtorch_stable/quantization/gptq/` | GPTQ q_gemm/shuffle |
| `csrc/libtorch_stable/quantization/gptq_allspark/` | AllSpark W8A16 |
| `csrc/libtorch_stable/quantization/marlin/` | Marlin generated kernels / repack |
| `csrc/libtorch_stable/quantization/machete/` | Machete Hopper mixed precision GEMM |
| `csrc/libtorch_stable/quantization/cutlass_w4a8/` | CUTLASS W4A8 |
| `csrc/libtorch_stable/quantization/fused_kernels/` | fused layernorm/SiLU + quant |
| `csrc/cutlass_extensions/` | CUTLASS epilogue/type/numeric conversion/builder 工具 |

## 3. 注册入口

### 3.1 C++ schema

大部分量化/GEMM op 在 `csrc/libtorch_stable/torch_bindings.cpp` 中注册。

核心段落：

- per-token group quant：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:12-30`
- Machete / Marlin schema：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:37-97`
- CUTLASS scaled_mm/grouped GEMM：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:100-171`
- FP4/NVFP4/MXFP4：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:173-244`
- W4A8：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:246-286`
- AWQ：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:301-309`
- mxfp8 / DeepSeek / AllSpark：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:311-349`
- INT8/FP8/GPTQ impl：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:720-732`

### 3.2 头文件声明

`csrc/libtorch_stable/ops.h` 中声明了量化相关函数：

- per-token group quant：`code/vllm/csrc/libtorch_stable/ops.h:9-28`
- CUTLASS scaled_mm / MoE mm / azp：`code/vllm/csrc/libtorch_stable/ops.h:33-90`
- FP4/NVFP4/MXFP4：`code/vllm/csrc/libtorch_stable/ops.h:91-147`
- AWQ：`code/vllm/csrc/libtorch_stable/ops.h:149-160`
- quant fused norm：`code/vllm/csrc/libtorch_stable/ops.h:198-232`
- INT8/FP8/GPTQ：`code/vllm/csrc/libtorch_stable/ops.h:386-423`

## 4. Python wrapper

`vllm/_custom_ops.py` 对量化/GEMM 提供 Python wrapper。

典型函数：

- `awq_dequantize()`：`code/vllm/vllm/_custom_ops.py:548-562`
- `awq_gemm()`：`code/vllm/vllm/_custom_ops.py:582-593`
- `gptq_gemm()`：`code/vllm/vllm/_custom_ops.py:614-634`
- `gptq_shuffle()`：`code/vllm/vllm/_custom_ops.py:655-656`
- `cutlass_scaled_fp4_mm()`：`code/vllm/vllm/_custom_ops.py:790-802`
- `cutlass_scaled_mm()`：`code/vllm/vllm/_custom_ops.py:813-861`
- `cutlass_scaled_mm_azp()`：`code/vllm/vllm/_custom_ops.py:864-891`

wrapper 负责：

- 创建输出 tensor。
- 判断平台和 shape 是否兼容。
- 在 ROCm 或不兼容场景走 Triton fallback。
- 注册 fake function 以支持 torch.compile。

例如 `cutlass_scaled_mm()` 在 ROCm 或 B 矩阵不满足 CUTLASS 对齐时会转到 Triton：`code/vllm/vllm/_custom_ops.py:850-861`。

## 5. INT8 / FP8 W8A8

### 5.1 dynamic/static quant

注册接口：

- `static_scaled_int8_quant`
- `dynamic_scaled_int8_quant`
- `static_scaled_fp8_quant`
- `dynamic_scaled_fp8_quant`
- `dynamic_per_token_scaled_fp8_quant`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:534-562`。

impl 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:720-728`。

源码文件包括：

- `csrc/libtorch_stable/quantization/w8a8/int8/scaled_quant.cu`
- `csrc/libtorch_stable/quantization/w8a8/fp8/common.cu`
- `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`
- `csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu`

CMake 基础列表位置：`code/vllm/CMakeLists.txt:432-438`。

### 5.2 CUTLASS scaled_mm

接口：

- `cutlass_scaled_mm`
- `cutlass_scaled_mm_azp`
- `cutlass_scaled_mm_supports_fp8`
- `cutlass_scaled_mm_supports_block_fp8`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:100-122`、`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:168-171`。

CUTLASS scaled_mm 的构建按架构分支：

- SM90：`code/vllm/CMakeLists.txt:733-765`
- SM120：`code/vllm/CMakeLists.txt:768-799`
- SM100：`code/vllm/CMakeLists.txt:802-833`
- CUTLASS 2.x fallback：`code/vllm/CMakeLists.txt:835-859`

## 6. FP4 / NVFP4 / MXFP4

### 6.1 接口

注册接口包括：

- `cutlass_scaled_fp4_mm`
- `cutlass_fp4_group_mm`
- `cutlass_mxfp4_group_mm`
- `scaled_fp4_quant`
- `scaled_fp4_quant.out`
- `scaled_fp4_experts_quant`
- `silu_and_mul_scaled_fp4_experts_quant`
- `mxfp4_experts_quant`
- `silu_and_mul_mxfp4_experts_quant`
- `silu_and_mul_nvfp4_quant`
- `cutlass_scaled_mm_supports_fp4`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:173-244`。

Python wrapper 中也包含 FP4 输出 tensor 创建逻辑：`code/vllm/vllm/_custom_ops.py:36-87`。

fake registration：`code/vllm/vllm/_custom_ops.py:89-110`。

### 6.2 构建分支

FP4/NVFP4 构建按 Blackwell family 划分：

- SM12x NVFP4：`code/vllm/CMakeLists.txt:940-967`
- SM10x/11x NVFP4/MXFP4：`code/vllm/CMakeLists.txt:969-1000`

源码文件包括：

- `nvfp4_quant_kernels.cu`
- `activation_nvfp4_quant_fusion_kernels.cu`
- `nvfp4_experts_quant.cu`
- `nvfp4_scaled_mm_kernels.cu`
- `nvfp4_blockwise_moe_kernel.cu`
- `mxfp4_experts_quant.cu`
- `mxfp4_blockwise_moe_kernel.cu`
- `nvfp4_kv_cache_kernels.cu`

## 7. AWQ

AWQ 接口：

- `awq_gemm`
- `awq_dequantize`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:301-309`。

impl 注册：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:654-656`。

Python wrapper：

- `awq_dequantize()`：`code/vllm/vllm/_custom_ops.py:548-562`
- `awq_gemm()`：`code/vllm/vllm/_custom_ops.py:582-593`

wrapper 支持 `VLLM_USE_TRITON_AWQ` 环境变量切换到 Triton AWQ：`code/vllm/vllm/_custom_ops.py:556-562`、`code/vllm/vllm/_custom_ops.py:589-593`。

源码文件：

- `csrc/libtorch_stable/quantization/awq/gemm_kernels.cu`
- `csrc/libtorch_stable/quantization/awq/dequantize.cuh`

## 8. GPTQ

GPTQ 接口：

- `gptq_gemm`
- `gptq_shuffle`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:564-574`。

impl 注册：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:730-732`。

Python wrapper：`code/vllm/vllm/_custom_ops.py:614-656`。

源码文件：

- `csrc/libtorch_stable/quantization/gptq/q_gemm.cu`
- `compat.cuh`
- `matrix_view.cuh`
- `qdq_*.cuh`

ROCm RDNA3 GPTQ 走 `_rocm_C`：

- `gptq_gemm_rdna3`
- `gptq_gemm_rdna3_wmma`

C++ 注册：`code/vllm/csrc/rocm/torch_bindings.cpp:42-52`。

Python wrapper：`code/vllm/vllm/_custom_ops.py:659-702`。

## 9. Marlin

Marlin 是用于量化 GEMM 的高性能 kernel family。

CMake 逻辑：

- 根据架构选择 `MARLIN_ARCHS`、`MARLIN_BF16_ARCHS`、`MARLIN_FP8_ARCHS`、`MARLIN_OTHER_ARCHS`
- 运行 `csrc/libtorch_stable/quantization/marlin/generate_kernels.py`
- glob 生成的 `sm80_kernel_*`、`sm75_kernel_*`、`sm89_kernel_*` 等源码

源码位置：`code/vllm/CMakeLists.txt:544-681`。

注册接口：

- `marlin_gemm`
- `gptq_marlin_repack`
- `awq_marlin_repack`
- `marlin_int4_fp8_preprocess`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:70-97`。

## 10. Machete

Machete 是 Hopper 上的 mixed precision GEMM。

构建条件：

- CUDA >= 12.0
- 目标架构兼容 sm90a

CMake 会运行：

```text
csrc/libtorch_stable/quantization/machete/generate.py
```

源码位置：`code/vllm/CMakeLists.txt:471-538`。

注册接口：

- `machete_supported_schedules`
- `machete_mm`
- `machete_prepack_B`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:37-68`。

## 11. W4A8

CUTLASS W4A8 接口：

- `cutlass_w4a8_mm`
- `cutlass_pack_scale_fp8`
- `cutlass_encode_and_reorder_int4b`
- `cutlass_w4a8_moe_mm`
- `cutlass_encode_and_reorder_int4b_grouped`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:246-286`。

构建条件：

- SM90a
- CUDA >= 12.0

源码位置：`code/vllm/CMakeLists.txt:1002-1033`。

## 12. AllSpark W8A16

接口：

- `rearrange_kn_weight_as_n32k16_order`
- `allspark_w8a16_gemm`

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:336-349`。

构建条件：Ampere arch 8.0/8.6/8.7/8.9。

源码位置：`code/vllm/CMakeLists.txt:717-731`。

## 13. fused quant kernels

包括：

- RMSNorm + dynamic per-token quant
- RMSNorm + per-block quant
- SiLU+Mul + per-block quant
- SiLU+Mul + FP4 expert quant

schema 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:381-415`。

Python wrapper：

- `rms_norm_dynamic_per_token_quant()`：`code/vllm/vllm/_custom_ops.py:417-434`
- `rms_norm_per_block_quant()`：`code/vllm/vllm/_custom_ops.py:437-493`
- `silu_and_mul_per_block_quant()`：`code/vllm/vllm/_custom_ops.py:496-543`

这些 fused kernel 的目的：把 norm/activation 和 quant 合并，减少一次读写和一次 kernel launch。

## 14. 关键结论

量化/GEMM 层是 vLLM native 代码中最复杂的部分，复杂性主要来自：

1. 量化格式多：INT8、FP8、FP4、W4A8、AWQ、GPTQ、NVFP4/MXFP4。
2. 硬件差异大：Ampere、Hopper、Blackwell、ROCm RDNA3。
3. kernel family 多：CUTLASS、Marlin、Machete、AllSpark、Triton fallback。
4. 需要和 torch.compile 兼容：fake/meta function、stable ABI、out variant。

整体调用链仍然很清晰：Python quantization layer 选择某种 quant method，最终进入 `vllm._custom_ops`，再调用 `torch.ops._C` 或 `_rocm_C` 的 native GEMM/quant kernel。