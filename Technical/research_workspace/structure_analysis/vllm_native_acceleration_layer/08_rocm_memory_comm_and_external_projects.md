# 08 ROCm、内存、通信与外部项目

## 1. 本文范围

除了 CUDA 主路径和 CPU 后端，vLLM native 层还有几类重要支撑模块：

- ROCm/HIP 专用扩展 `_rocm_C`
- GPU memory allocator：`cumem_allocator`
- 自定义 all-reduce / quick reduce
- CUDA/HIP 工具函数
- external projects：Triton kernels、DeepGEMM、FlashMLA、FMHA SM100、Qutlass、vLLM FlashAttention

这些模块不一定直接对应一个模型 layer，但对性能、显存管理和分布式执行很关键。

## 2. ROCm/HIP 构建路径

CMake 判断 GPU language：

- CUDA：`VLLM_GPU_LANG = CUDA`
- HIP/ROCm：`VLLM_GPU_LANG = HIP`

ROCm 分支位置：`code/vllm/CMakeLists.txt:160-183`。

HIP 会显式 `enable_language(HIP)`：`code/vllm/CMakeLists.txt:167-173`。

支持 AMD GPU arch 列表：`code/vllm/CMakeLists.txt:51-52`。

## 3. `_rocm_C` extension

ROCm 专用 extension target：

```text
_rocm_C
```

基础源码：

- `csrc/rocm/torch_bindings.cpp`
- `csrc/rocm/skinny_gemms.cu`
- `csrc/rocm/attention.cu`

如果目标 arch 包含 `gfx1100`，追加：

- `csrc/rocm/q_gemm_rdna3.cu`
- `csrc/rocm/q_gemm_rdna3_wmma.cu`
- `csrc/rocm/moe_q_gemm_rdna3.cu`

CMake 位置：`code/vllm/CMakeLists.txt:1355-1386`。

## 4. ROCm ops 注册

ROCm binding 文件：

```text
csrc/rocm/torch_bindings.cpp
```

使用：

```cpp
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops)
```

源码位置：`code/vllm/csrc/rocm/torch_bindings.cpp:14-15`。

注册到 Python 后通常通过：

```python
torch.ops._rocm_C.xxx
```

调用。

## 5. ROCm GEMM ops

ROCm 注册了 skinny GEMM / matrix-vector 相关 ops：

| op | 作用 | 源码位置 |
|---|---|---|
| `LLMM1` | matrix-vector multiplication | `code/vllm/csrc/rocm/torch_bindings.cpp:17-21` |
| `wvSplitK` | skinny matrix-matrix multiplication | `code/vllm/csrc/rocm/torch_bindings.cpp:23-27` |
| `wvSplitKrc` | skinny matrix-matrix 变体 | `code/vllm/csrc/rocm/torch_bindings.cpp:29-33` |
| `wvSplitKQ` | FP8 quant 相关 splitK | `code/vllm/csrc/rocm/torch_bindings.cpp:35-40` |

源码文件：

```text
csrc/rocm/skinny_gemms.cu
```

## 6. ROCm RDNA3 GPTQ

当构建 `gfx1100` 时启用：

- `gptq_gemm_rdna3`
- `gptq_gemm_rdna3_wmma`
- `moe_gptq_gemm_rdna3`

C++ 注册：`code/vllm/csrc/rocm/torch_bindings.cpp:42-62`。

Python wrapper：

- `gptq_gemm_rdna3()`：`code/vllm/vllm/_custom_ops.py:659-669`
- fake registration：`code/vllm/vllm/_custom_ops.py:672-702`
- `moe_gptq_gemm_rdna3()`：`code/vllm/vllm/_custom_ops.py:704-733`

## 7. ROCm PagedAttention

ROCm 专用 attention op：

```text
paged_attention
```

C++ schema：`code/vllm/csrc/rocm/torch_bindings.cpp:64-82`。

Python wrapper：`code/vllm/vllm/_custom_ops.py:208-251`。

参数比 CUDA PagedAttention 额外包含：

- `query_start_loc`
- `fp8_out_scale`
- `mfma_type`

`mfma_type` 根据环境变量可选 `fp8` 或 `f16`：`code/vllm/vllm/_custom_ops.py:208-229`。

## 8. ROCm 与 stable ABI extensions

ROCm 也会构建 `_C_stable_libtorch` 和 `_moe_C_stable_libtorch`，但定义：

```cmake
TORCH_TARGET_VERSION=0x020A000000000000ULL
USE_ROCM
```

源码位置：

- `_C_stable_libtorch`：`code/vllm/CMakeLists.txt:1100-1108`
- `_moe_C_stable_libtorch`：`code/vllm/CMakeLists.txt:1326-1334`

ROCm 下还特别处理 PyTorch bundled `libamdhip64`，避免 raw HIP APIs 链到系统 ROCm 副本导致双 runtime：

- `_C_stable_libtorch`：`code/vllm/CMakeLists.txt:1110-1127`
- `_moe_C_stable_libtorch`：`code/vllm/CMakeLists.txt:1336-1353`

## 9. cumem_allocator

`cumem_allocator` 是 GPU 内存分配器 extension。

源码：

```text
csrc/cumem_allocator.cpp
```

CMake 定义：`code/vllm/CMakeLists.txt:280-319`。

CUDA 下链接：

```text
CUDA::cuda_driver
```

HIP 下优先查找 `${ROCM_PATH}/lib/libamdhip64.so`，否则 fallback 到 `amdhip64`：`code/vllm/CMakeLists.txt:290-310`。

作用：

- 支撑底层 GPU memory allocation。
- 与 CUDA/HIP driver API 交互。
- 为 vLLM 的显存管理、共享 buffer、allocator 策略提供 native 支撑。

## 10. custom all-reduce

### 10.1 stable CUDA custom_ar

`_C_custom_ar` fragment 注册在 `csrc/libtorch_stable/torch_bindings.cpp`：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:868-900`。

包含：

- `init_custom_ar`
- `all_reduce`
- `dispose`
- `meta_size`
- `register_buffer`
- `get_graph_buffer_ipc_meta`
- `register_graph_buffers`
- `allocate_shared_buffer_and_handle`
- `open_mem_handle`
- `free_shared_buffer`

声明位置：`code/vllm/csrc/libtorch_stable/ops.h:349-367`。

实现源码：

```text
csrc/libtorch_stable/custom_all_reduce.cu
csrc/custom_all_reduce.cuh
```

作用：

- 自定义 GPU all-reduce。
- 支持 IPC buffer 注册。
- 支持 CUDA graph buffer metadata。
- 支持共享 buffer handle。

### 10.2 ROCm quick reduce

ROCm legacy `_C` 中注册 quick reduce：

- `qr_all_reduce`
- `init_custom_qr`
- `qr_destroy`
- `qr_get_handle`
- `qr_open_handles`
- `qr_max_size`

源码位置：`code/vllm/csrc/torch_bindings.cpp:74-90`。

CMake 中 ROCm 会把 `csrc/custom_quickreduce.cu` 加入 `_C`：`code/vllm/CMakeLists.txt:398-405`。

## 11. CUDA/HIP utils

### 11.1 CUDA utils kernels

相关文件：

- `csrc/cuda_utils.h`
- `csrc/cuda_compat.h`
- `csrc/libtorch_stable/cuda_utils_kernels.cu`
- `csrc/libtorch_stable/cuda_view.cu`

注册接口：

- `get_device_attribute`
- `get_max_shared_memory_per_block_device_attribute`

stable binding 位置：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:748-759`。

legacy ROCm fallback 位置：`code/vllm/csrc/torch_bindings.cpp:92-104`。

### 11.2 CPU tensor -> CUDA view

`get_cuda_view_from_cpu_tensor` 用于 CPU tensor 的 CUDA UVA view。

注册位置：

- schema：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:34-35`
- CPU impl：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:741-746`
- declaration：`code/vllm/csrc/libtorch_stable/ops.h:168-171`

## 12. external projects

CMake 最后包含外部项目。

### 12.1 CUDA/HIP 都包含

```cmake
include(cmake/external_projects/triton_kernels.cmake)
```

源码位置：`code/vllm/CMakeLists.txt:1394-1397`。

### 12.2 CUDA 专用

```cmake
include(cmake/external_projects/deepgemm.cmake)
include(cmake/external_projects/fmha_sm100.cmake)
include(cmake/external_projects/flashmla.cmake)
include(cmake/external_projects/qutlass.cmake)
include(cmake/external_projects/vllm_flash_attn.cmake)
```

源码位置：`code/vllm/CMakeLists.txt:1399-1407`。

这些外部项目用于补充：

- DeepGEMM
- Blackwell/SM100 FMHA
- FlashMLA
- Qutlass
- vLLM FlashAttention
- Triton kernels package

## 13. HIPIFY

ROCm 构建结束后会调用：

```cmake
vllm_finalize_hipify_target()
```

源码位置：`code/vllm/CMakeLists.txt:1388-1392`。

相关脚本：

```text
cmake/hipify.py
```

用于把 CUDA 风格源码转换/适配到 HIP 构建。

## 14. 总结

ROCm、内存、通信和外部项目是 vLLM native 层的“支撑系统”：

- ROCm 提供 AMD GPU 专用 attention/GEMM/GPTQ ops。
- `cumem_allocator` 支撑底层 GPU 显存管理。
- custom all-reduce/quick reduce 支撑多 GPU 通信优化。
- external projects 补充高性能 attention/GEMM/Triton kernel 包。
- CUDA/HIP utils 提供设备属性、UVA view 等底层工具。

这些模块和 attention/quant/MoE 一起构成 vLLM 高性能推理的 native 基础。