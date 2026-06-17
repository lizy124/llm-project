# 07 CPU 后端

## 1. CPU 后端定位

vLLM 不只支持 GPU 推理，也提供 CPU native extension。CPU 后端的目标是：

- 在无 GPU 或 CPU-only 环境中运行推理。
- 使用 AVX/AMX/ARM/Power/RISC-V 等 ISA 加速关键算子。
- 使用 oneDNN/SGL kernels 加速 GEMM、量化 GEMM、MoE、attention。
- 尽量复用 `_C` custom op namespace，让 Python 层调用方式保持一致。

CPU 构建入口：

```text
cmake/cpu_extension.cmake
```

从主 CMake 进入 CPU 路径的位置：`code/vllm/CMakeLists.txt:143-154`。

## 2. CPU 源码目录

| 路径 | 作用 |
|---|---|
| `csrc/cpu/torch_bindings.cpp` | CPU custom ops 注册入口 |
| `csrc/cpu/activation.cpp` | CPU activation ops |
| `csrc/cpu/layernorm.cpp` | CPU RMSNorm / fused add RMSNorm |
| `csrc/cpu/pos_encoding.cpp` | CPU RoPE |
| `csrc/cpu/cpu_attn.cpp` | CPU attention 主实现 |
| `csrc/cpu/cpu_attn_*.hpp` | 不同 ISA attention 实现 |
| `csrc/cpu/dnnl_helper.cpp` / `.h` | oneDNN helper |
| `csrc/cpu/dnnl_kernels.cpp` | oneDNN matmul/scaled_mm |
| `csrc/cpu/sgl-kernels/` | SGL CPU GEMM/MoE/conv/FLA kernels |
| `csrc/cpu/cpu_fused_moe.cpp` | CPU fused MoE |
| `csrc/cpu/cpu_wna16.cpp` | CPU WNA16 GEMM |
| `csrc/cpu/spec_decode_utils.cpp` | speculative decoding CPU utilities |
| `csrc/cpu/shm.cpp` | CPU shared-memory collectives |
| `csrc/moe/dynamic_4bit_int_moe_cpu.cpp` | CPU dynamic 4-bit int MoE |

## 3. CPU ISA 检测与编译 flags

CPU CMake 会检测：

- x86_64 / amd64
- ARM NEON / BF16
- Power9/10/11
- S390
- RISC-V RVV FP16/BF16
- Apple Silicon

源码位置：`code/vllm/cmake/cpu_extension.cmake:89-221`。

### 3.1 x86

x86 要求 gcc/g++ >= 12.3：`code/vllm/cmake/cpu_extension.cmake:112-117`。

编译 flags 包括：

- `-mf16c`
- `-mavx512f`
- `-mavx512vl`
- `-mavx512bw`
- `-mavx512dq`
- `-mamx-bf16`
- `-mamx-tile`
- `-mavx512bf16`
- `-mavx512vnni`
- `-mavx2`

源码位置：`code/vllm/cmake/cpu_extension.cmake:118-133`。

### 3.2 ARM

ARMv8/NEON 下根据 BF16 支持选择：

- `-march=armv8.2-a+bf16+dotprod+fp16`
- 或 `-march=armv8.2-a+dotprod+fp16`

源码位置：`code/vllm/cmake/cpu_extension.cmake:148-158`。

### 3.3 RISC-V

RISC-V 支持 `VLLM_RVV_VLEN` 显式指定，或者从 `/proc/cpuinfo` auto-detect `zvl<N>b`。

源码位置：`code/vllm/cmake/cpu_extension.cmake:167-218`。

## 4. oneDNN / ACL 集成

CPU 后端会在支持的 ISA 下构建 oneDNN：

```text
oneDNN v3.10
```

ARM AArch64 下可能使用 pinned oneDNN commit，并可构建 Arm Compute Library ACL。

源码位置：`code/vllm/cmake/cpu_extension.cmake:224-357`。

oneDNN 配置：

- static library
- inference workload
- primitives: MATMUL / REORDER
- JIT profiling on
- max CPU ISA hints on

源码位置：`code/vllm/cmake/cpu_extension.cmake:320-331`。

CPU binding 中 oneDNN 相关接口：

- `create_onednn_scaled_mm_handler`
- `onednn_scaled_mm`
- `create_onednn_mm_handler`
- `onednn_mm`
- `is_onednn_acl_supported`

声明位置：`code/vllm/csrc/cpu/torch_bindings.cpp:13-33`。

注册位置：`code/vllm/csrc/cpu/torch_bindings.cpp:337-366`。

## 5. CPU extension targets

### 5.1 x86 多 target

x86 下会构建：

- `_C`：AMX + AVX512F + BF16 + VNNI
- `_C_AVX512`
- `_C_AVX2`

源码位置：`code/vllm/cmake/cpu_extension.cmake:498-535`。

这样可以按机器能力加载合适实现。

### 5.2 非 x86 target

非 x86 下构建一个 `_C`：`code/vllm/cmake/cpu_extension.cmake:536-554`。

## 6. CPU custom ops 注册

CPU binding 入口：

```text
csrc/cpu/torch_bindings.cpp
```

它强制：

```cpp
#define TORCH_EXTENSION_NAME _C
```

源码位置：`code/vllm/csrc/cpu/torch_bindings.cpp:7-10`。

这意味着 CPU 后端同样注册到 `torch.ops._C`，使 Python wrapper 可以复用 `_C` namespace。

## 7. CPU activation / norm / RoPE

注册接口：

- `silu_and_mul`
- `gelu_and_mul`
- `gelu_tanh_and_mul`
- `gelu_new`
- `gelu_fast`
- `gelu_quick`
- `activation_lut_bf16`
- `rms_norm`
- `fused_add_rms_norm`
- `rotary_embedding`

源码位置：

- activation：`code/vllm/csrc/cpu/torch_bindings.cpp:275-308`
- layernorm：`code/vllm/csrc/cpu/torch_bindings.cpp:310-321`
- RoPE：`code/vllm/csrc/cpu/torch_bindings.cpp:323-330`

## 8. CPU quantization / GEMM

### 8.1 INT8 quant

注册：

- `static_scaled_int8_quant`
- `dynamic_scaled_int8_quant`

源码位置：`code/vllm/csrc/cpu/torch_bindings.cpp:368-379`。

### 8.2 SGL kernels

SGL CPU kernels 包括：

- packed linear
- INT8 scaled mm
- FP8 W8A16
- INT4 W4A8
- causal conv1d
- MoE

注册位置：`code/vllm/csrc/cpu/torch_bindings.cpp:408-473`。

相关源码：

```text
csrc/cpu/sgl-kernels/gemm.cpp
csrc/cpu/sgl-kernels/gemm_int8.cpp
csrc/cpu/sgl-kernels/gemm_fp8.cpp
csrc/cpu/sgl-kernels/gemm_int4.cpp
csrc/cpu/sgl-kernels/moe.cpp
csrc/cpu/sgl-kernels/moe_int8.cpp
csrc/cpu/sgl-kernels/moe_int4.cpp
csrc/cpu/sgl-kernels/moe_fp8.cpp
```

### 8.3 WNA16

接口：

```text
cpu_gemm_wna16
```

源码位置：`code/vllm/csrc/cpu/torch_bindings.cpp:528-535`。

## 9. CPU attention

CPU attention 相关接口：

- `get_scheduler_metadata`
- `cpu_attn_reshape_and_cache`
- `cpu_attention_with_kv_cache`
- `mla_decode_kvcache`
- `compute_slot_mapping_kernel_impl`

源码位置：

- scheduler metadata / attention：`code/vllm/csrc/cpu/torch_bindings.cpp:499-521`
- MLA decode：`code/vllm/csrc/cpu/torch_bindings.cpp:550-554`
- slot mapping：`code/vllm/csrc/cpu/torch_bindings.cpp:556-560`

CPU attention 头文件按 ISA 拆分：

- `cpu_attn_amx.hpp`
- `cpu_attn_fp8.hpp`
- `cpu_attn_impl.hpp`
- `cpu_attn_neon.hpp`
- `cpu_attn_neon_bfmmla.hpp`
- `cpu_attn_rvv.hpp`
- `cpu_attn_vec.hpp`
- `cpu_attn_vec16.hpp`
- `cpu_attn_vsx.hpp`
- `cpu_attn_vxe.hpp`

CMake 会生成 CPU attention dispatch header：`code/vllm/cmake/cpu_extension.cmake:387-398`。

## 10. CPU shared memory collectives

CPU 后端支持 shared memory collectives：

- `init_shm_manager`
- `join_shm_manager`
- `shm_allreduce`
- `shm_gather`
- `shm_all_gather`
- `shm_send_tensor_list`
- `shm_recv_tensor_list`

声明位置：`code/vllm/csrc/cpu/torch_bindings.cpp:39-57`。

注册位置：`code/vllm/csrc/cpu/torch_bindings.cpp:382-405`。

这些用于 CPU 多进程/多 rank 间通信。

## 11. CPU speculative decoding utilities

注册接口包括：

- `eagle_prepare_inputs_padded_kernel_impl`
- `eagle_prepare_next_token_padded_kernel_impl`
- `eagle_step_slot_mapping_metadata_kernel_impl`
- `copy_and_expand_eagle_inputs_kernel_impl`
- `rejection_greedy_sample_kernel_impl`
- `rejection_random_sample_kernel_impl`
- `expand_kernel_impl`
- `sample_recovered_tokens_kernel_impl`

源码位置：`code/vllm/csrc/cpu/torch_bindings.cpp:564-626`。

这些服务于 EAGLE/speculative decoding 在 CPU 后端的辅助数据处理。

## 12. CPU 后端调用链

```text
Python model/attention/quant layer
  -> vllm._custom_ops wrapper
  -> torch.ops._C.xxx
  -> CPU _C extension
  -> csrc/cpu/torch_bindings.cpp
  -> csrc/cpu/*.cpp / sgl-kernels / oneDNN
```

由于 CPU `_C` 与 GPU `_C` 使用同一个 namespace，Python 层通常不需要大改，只需 current_platform 加载不同 extension。

## 13. 关键结论

CPU 后端不是简单 fallback，而是完整 native 后端：它按 ISA 生成不同 extension，集成 oneDNN、SGL kernels、CPU attention、CPU MoE、SHM collectives 和 speculative decoding 工具。其核心设计是“保持 Python custom op namespace 一致，同时在 CMake 和 extension 加载阶段选择不同实现”。