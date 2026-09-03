# 02 构建系统与扩展库

## 1. 总体构建入口

vLLM 的 native 构建主入口是：

```text
CMakeLists.txt
```

它定义 C++/CUDA/HIP 标准、选择 target device、寻找 Python/PyTorch、配置 CUDA/HIP 架构、构建多个 extension target。

CMake 最小版本和语言标准：

- CMake >= 3.26
- C++20
- CUDA 20
- HIP 20

源码位置：`code/vllm/CMakeLists.txt:1-21`。

CMake 直接构建示例也写在文件头部：`code/vllm/CMakeLists.txt:3-13`。

## 2. Target device 分支

默认 target device 是 CUDA：

```cmake
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")
```

源码位置：`code/vllm/CMakeLists.txt:32-35`。

如果不是 `cuda` 或 `rocm`：

- `cpu`：include `cmake/cpu_extension.cmake`
- 其他设备：直接 return

源码位置：`code/vllm/CMakeLists.txt:143-154`。

这意味着构建有三条主路径：

```text
VLLM_TARGET_DEVICE=cuda
  -> CUDA extension path

VLLM_TARGET_DEVICE=rocm
  -> HIP/ROCm extension path

VLLM_TARGET_DEVICE=cpu
  -> cmake/cpu_extension.cmake
```

## 3. Python / PyTorch / 编译器约束

### 3.1 Python 版本

支持 Python：

```text
3.10, 3.11, 3.12, 3.13, 3.14
```

源码位置：`code/vllm/CMakeLists.txt:45-49`。

必须通过 `VLLM_PYTHON_EXECUTABLE` 指定 Python，否则 CMake fatal：`code/vllm/CMakeLists.txt:74-84`。

### 3.2 GCC 版本

PyTorch C++20 headers 要求 GCC >= 11.3：`code/vllm/CMakeLists.txt:23-29`。

### 3.3 PyTorch 版本

CUDA / ROCm 预期 PyTorch 版本：

```text
TORCH_SUPPORTED_VERSION_CUDA = 2.11.0
TORCH_SUPPORTED_VERSION_ROCM = 2.11.0
```

源码位置：`code/vllm/CMakeLists.txt:62-72`。

版本不匹配时是 warning，不是 fatal：`code/vllm/CMakeLists.txt:160-180`。

## 4. CUDA/HIP 架构处理

### 4.1 CUDA 架构

根据 CUDA 编译器版本选择支持架构：

- CUDA >= 13.0：`7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0`
- CUDA >= 12.8：额外包含 Blackwell 相关 arch
- 否则：`7.0;7.5;8.0;8.6;8.7;8.9;9.0`

源码位置：`code/vllm/CMakeLists.txt:105-118`。

CMake 会从 `TORCH_CUDA_ARCH_LIST` 提取实际目标架构，并与支持列表求交集：`code/vllm/CMakeLists.txt:186-205`。

### 4.2 HIP 架构

支持 AMD GPU arch：

```text
gfx906, gfx908, gfx90a, gfx942, gfx950, gfx1030, gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1152, gfx1153, gfx1200, gfx1201
```

源码位置：`code/vllm/CMakeLists.txt:51-52`。

HIP 路径会显式 `enable_language(HIP)`：`code/vllm/CMakeLists.txt:167-173`。

## 5. extension target 总览

### 5.1 `spinloop`

- 纯 C++ extension
- Python >= 3.11 才构建
- 源码：`csrc/spinloop.cpp`
- x86_64/amd64 下添加 `-mmwaitx`

源码位置：`code/vllm/CMakeLists.txt:121-140`。

### 5.2 `cumem_allocator`

- 源码：`csrc/cumem_allocator.cpp`
- CUDA 链接 `CUDA::cuda_driver`
- HIP 链接 `amdhip64`
- 输出到 `vllm` 包

源码位置：`code/vllm/CMakeLists.txt:280-319`。

### 5.3 `_C`

GPU 路径下 `_C` 当前较小，主要包括：

- `csrc/quantization/activation_kernels.cu`
- `csrc/torch_bindings.cpp`

源码位置：`code/vllm/CMakeLists.txt:321-328`。

CUDA 下会加载 CUTLASS，并设置 `CUTLASS_REVISION = v4.4.2`：`code/vllm/CMakeLists.txt:329-360`。

定义 target：`code/vllm/CMakeLists.txt:409-420`。

`csrc/torch_bindings.cpp` 注释也说明：基础 ops 已迁移到 `_C_stable_libtorch`，这里主要保留量化激活和 ROCm 兼容相关：`code/vllm/csrc/torch_bindings.cpp:1-8`、`code/vllm/csrc/torch_bindings.cpp:44-52`。

### 5.4 `_C_stable_libtorch`

这是当前 GPU 主要 native ops extension。

特点：

- 使用 PyTorch stable ABI。
- C++ 注册入口：`csrc/libtorch_stable/torch_bindings.cpp`
- 虽然 extension target 叫 `_C_stable_libtorch`，但注册 namespace 仍是 `_C`，兼容现有 Python `torch.ops._C.xxx`。

源码注释位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:7-10`。

基础源码列表包括：

- activation
- W8A8 quant
- permute cols
- GPTQ
- RoPE / fused qk norm rope
- layernorm
- fused quant kernels
- attention merge
- sampler / topk
- Mamba selective scan
- paged attention v1/v2
- cache kernels
- custom all-reduce
- DeepSeek V4 fused kernel

源码位置：`code/vllm/CMakeLists.txt:430-458`。

CUDA 下还追加：

- cuda_view / cuda_utils
- CUTLASS common
- scaled_mm
- FP4/NVFP4
- AWQ
- minimax reduce RMS

源码位置：`code/vllm/CMakeLists.txt:460-469`。

定义 target：`code/vllm/CMakeLists.txt:1075-1086`。

stable ABI 版本宏：

- CUDA：`TORCH_TARGET_VERSION=0x020B...`，即 PyTorch 2.11
- HIP：`TORCH_TARGET_VERSION=0x020A...`，即 PyTorch 2.10

源码位置：`code/vllm/CMakeLists.txt:1088-1108`。

### 5.5 `_moe_C_stable_libtorch`

MoE 专用 extension。

基础源码：

- `csrc/libtorch_stable/moe/torch_bindings.cpp`
- `moe_align_sum_kernels.cu`
- `topk_softmax_kernels.cu`
- `topk_softplus_sqrt_kernels.cu`

源码位置：`code/vllm/CMakeLists.txt:1131-1138`。

CUDA 下追加：

- `moe_wna16.cu`
- `grouped_topk_kernels.cu`
- `moe_permute_unpermute_kernel.cu`
- `moe_permute_unpermute_op.cu`
- Marlin MoE generated kernels
- DeepSeek V3 router GEMM kernels

源码位置：`code/vllm/CMakeLists.txt:1140-1299`。

定义 target：`code/vllm/CMakeLists.txt:1301-1312`。

注册 namespace 是 `_moe_C`：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:6-7`。

### 5.6 `_rocm_C`

只有 HIP/ROCm 构建时启用。

基础源码：

- `csrc/rocm/torch_bindings.cpp`
- `csrc/rocm/skinny_gemms.cu`
- `csrc/rocm/attention.cu`

如果目标架构包含 `gfx1100`，追加：

- `q_gemm_rdna3.cu`
- `q_gemm_rdna3_wmma.cu`
- `moe_q_gemm_rdna3.cu`

源码位置：`code/vllm/CMakeLists.txt:1355-1386`。

ROCm binding 注册 namespace 是 `_rocm_C`：`code/vllm/csrc/rocm/torch_bindings.cpp:14-15`。

## 6. CUDA 条件编译的大型 kernel families

`_C_stable_libtorch` 会根据 CUDA 版本和目标 SM 架构有条件编译很多 kernel family。

### 6.1 Machete

- Hopper / sm90a
- CUDA >= 12.0
- 自动运行 `quantization/machete/generate.py` 生成源码

源码位置：`code/vllm/CMakeLists.txt:471-538`。

### 6.2 Marlin

- 自动运行 `quantization/marlin/generate_kernels.py`
- 分 fp16 output、bf16 output、fp8 input、sm75 等架构集合

源码位置：`code/vllm/CMakeLists.txt:544-681`。

### 6.3 DeepSeek V3 fused A GEMM

- SM90+ / Blackwell 系列
- CUDA >= 12.0

源码位置：`code/vllm/CMakeLists.txt:683-699`。

### 6.4 FP32 router GEMM

- SM90+ / CUDA >= 12.0
- 针对 H=3072、E=256、M<=32 的 router GEMM

源码位置：`code/vllm/CMakeLists.txt:701-715`。

### 6.5 AllSpark

- Ampere 系列：8.0/8.6/8.7/8.9
- W8A16 fused GEMM

源码位置：`code/vllm/CMakeLists.txt:717-731`。

### 6.6 CUTLASS scaled_mm

分多代实现：

- SM90 CUTLASS 3.x
- SM120 CUTLASS 3.x
- SM100 CUTLASS 3.x
- CUTLASS 2.x fallback for older arch

源码位置：`code/vllm/CMakeLists.txt:733-859`。

### 6.7 CUTLASS MoE grouped MM

- SM90：CUDA >= 12.3
- SM100：CUDA >= 12.8
- `moe_data.cu` 支撑 problem sizes / expert offsets 等数据

源码位置：`code/vllm/CMakeLists.txt:861-934`。

### 6.8 FP4 / NVFP4 / MXFP4

- SM12x NVFP4
- SM10x/11x NVFP4/MXFP4
- CUDA >= 12.8

源码位置：`code/vllm/CMakeLists.txt:936-1000`。

### 6.9 W4A8

- SM90a / CUDA >= 12.0
- 使用 `cutlass_w4a8` 相关源码

源码位置：`code/vllm/CMakeLists.txt:1002-1033`。

### 6.10 CUTLASS MLA

- SM100/SM11x/SM10x
- CUDA >= 12.8
- 运行时由 Python attention backend gate

源码注释位置：`code/vllm/CMakeLists.txt:1035-1059`。

## 7. CPU 构建路径

CPU 构建路径在：

```text
cmake/cpu_extension.cmake
```

特点：

- C++20
- OpenMP
- 根据 CPU ISA 选择编译 flags
- 可能构建 oneDNN / ACL
- 生成 CPU attention dispatch header
- 构建 `_C`、`_C_AVX512`、`_C_AVX2` 等 CPU extension

入口位置：`code/vllm/CMakeLists.txt:143-154`。

详细见 [07_cpu_backend.md](07_cpu_backend.md)。

## 8. Rust 构建系统

Rust workspace：`code/vllm/rust/Cargo.toml:1-16`。

主要 workspace dependencies 包括：

- axum / tower / tower-http
- tokio / tokio-stream / tokio-util
- tonic / prost
- serde / rmp-serde / rmpv
- minijinja
- tokenizers
- pyo3 / pythonize
- zeromq
- vllm-chat / vllm-llm / vllm-server / vllm-engine-core-client 等本地 crate

源码位置：`code/vllm/rust/Cargo.toml:24-129`。

Python setuptools-rust 构建入口：`code/vllm/tools/build_rust.py:18-36`。

输出：

1. `vllm.vllm-rs`：Rust executable frontend。
2. `vllm._rust_tool_parser`：PyO3 parser 模块。

## 9. 外部项目

CUDA/HIP 构建还可能包含外部项目：

- `triton_kernels`
- `deepgemm`
- `fmha_sm100`
- `flashmla`
- `qutlass`
- `vllm_flash_attn`

包含位置：`code/vllm/CMakeLists.txt:1394-1407`。

这些通常用于补充高性能 attention/GEMM/MLA/Triton kernel 包。

## 10. 总结

vLLM 的 native 构建不是单一扩展，而是一组按职责拆开的 extension：

```text
_C                      legacy / CPU main / small GPU legacy ops
_C_stable_libtorch      主力 GPU ops，stable ABI
_moe_C_stable_libtorch  MoE 专用 GPU ops
_rocm_C                 ROCm 专用 ops
cumem_allocator         GPU memory allocator
spinloop                轻量 C++ runtime 辅助
vllm-rs                 Rust frontend binary
_rust_tool_parser       Rust parser Python module
```

这种拆分的好处是：

- 减少 ABI 兼容风险。
- MoE 大量 kernel 单独管理。
- ROCm/CPU 分支更清晰。
- Rust frontend 和 Python engine 可以独立演进。