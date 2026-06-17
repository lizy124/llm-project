# 01 底层加速层总览

## 1. 底层加速层在 vLLM 中的位置

vLLM 的上层 Python 负责：

- API server
- scheduler
- worker / executor
- model runner
- attention backend 选择
- quantization layer 选择
- KV cache 管理策略

但真正耗时的张量计算、cache 搬运、attention 解码、量化矩阵乘、MoE token 重排与 grouped GEMM 等，都落在 native 层。

可以把底层加速层理解成：

```text
Python model_runner / attention backend / quantization layer
  -> Python wrapper: vllm/_custom_ops.py
  -> torch.ops._C / torch.ops._moe_C / torch.ops._rocm_C
  -> C++ binding: csrc/**/torch_bindings.cpp
  -> CUDA/HIP/CPU kernel implementation
```

Rust 则是另一条路线：

```text
Rust vllm-rs frontend
  -> axum HTTP / tonic gRPC
  -> vllm-chat / vllm-text / vllm-llm
  -> vllm-engine-core-client
  -> ZMQ + MessagePack
  -> Python EngineCore
```

Rust 不主要承担 GPU kernel，而是承担更高性能的 frontend/protocol/chat/template/parser/tokenizer 等北向服务层。

## 2. 关键目录结构

### 2.1 `csrc/`

`csrc/` 是 C++/CUDA/HIP/CPU 源码根目录，包含：

| 路径 | 作用 |
|---|---|
| `csrc/torch_bindings.cpp` | legacy `_C` 扩展注册入口，当前只保留少量 legacy/ROCm 相关 ops |
| `csrc/libtorch_stable/` | 主力 GPU stable ABI 扩展，attention、cache、activation、quant、sampler、Mamba 等 |
| `csrc/libtorch_stable/torch_bindings.cpp` | `_C_stable_libtorch` 的核心注册入口，namespace 仍注册到 `_C` |
| `csrc/libtorch_stable/moe/` | MoE 专用 stable ABI 扩展 `_moe_C_stable_libtorch` |
| `csrc/cpu/` | CPU 后端：activation、layernorm、attention、oneDNN、SGL kernels、MoE、SHM |
| `csrc/rocm/` | ROCm 专用 `_rocm_C` 扩展 |
| `csrc/attention/` | attention 通用模板、dtype 特化头文件 |
| `csrc/cutlass_extensions/` | CUTLASS 类型、epilogue、builder、numeric conversion 等扩展工具 |
| `csrc/quantization/` | legacy/特定量化源码 |
| `csrc/core/` | registration、异常、scalar type、batch invariant 等基础设施 |
| `csrc/cumem_allocator.cpp` | CUDA/HIP memory allocator 扩展 |
| `csrc/custom_all_reduce.cuh` / `custom_quickreduce.cu` | 自定义通信/规约相关 |

### 2.2 `rust/`

`rust/` 是 Rust workspace。`rust/README.md` 说明它是 vLLM 的 Rust drop-in alternative frontend，目标是用 Rust 重建北向 serving layer，同时通过已有 engine boundary 用 ZMQ 与 Python vLLM engine 通信：`code/vllm/rust/README.md:1-7`。

workspace 成员在 `rust/Cargo.toml` 中声明：

- `src/chat`
- `src/cmd`
- `src/engine-core-client`
- `src/llm`
- `src/managed-engine`
- `src/metrics`
- `src/mock-engine`
- `src/reasoning-parser`
- `src/server`
- `src/text`
- `src/tokenizer`
- `src/tool-parser`
- `src/tool-parser/python`

源码位置：`code/vllm/rust/Cargo.toml:1-16`。

## 3. Native extension 的主要输出

CMake 会构建多个 Python extension / binary：

| Target | 来源 | 作用 |
|---|---|---|
| `spinloop` | `csrc/spinloop.cpp` | 轻量 C++ 扩展，Python 3.11+，用于低层循环/等待相关能力 |
| `cumem_allocator` | `csrc/cumem_allocator.cpp` | CUDA/HIP 内存分配器扩展 |
| `_C` | `csrc/torch_bindings.cpp` 等 | legacy custom ops namespace，GPU/CPU 下含义不同 |
| `_C_stable_libtorch` | `csrc/libtorch_stable/**` | 主力 stable ABI GPU ops，仍注册到 `torch.ops._C` |
| `_moe_C_stable_libtorch` | `csrc/libtorch_stable/moe/**` | MoE 专用 stable ABI ops，注册到 `torch.ops._moe_C` |
| `_rocm_C` | `csrc/rocm/**` | ROCm/HIP 专用 ops |
| `vllm-rs` | `rust/src/cmd` | Rust frontend 可执行文件 |
| `vllm._rust_tool_parser` | `rust/src/tool-parser/python` | Rust tool parser PyO3 模块 |

相关 CMake target 定义：

- `spinloop`：`code/vllm/CMakeLists.txt:121-140`
- `cumem_allocator`：`code/vllm/CMakeLists.txt:280-319`
- `_C`：`code/vllm/CMakeLists.txt:321-420`
- `_C_stable_libtorch`：`code/vllm/CMakeLists.txt:430-1128`
- `_moe_C_stable_libtorch`：`code/vllm/CMakeLists.txt:1131-1353`
- `_rocm_C`：`code/vllm/CMakeLists.txt:1355-1386`

Rust 构建入口定义在 `tools/build_rust.py`，其中 `vllm.vllm-rs` 是 executable binding，`vllm._rust_tool_parser` 是 PyO3 binding：`code/vllm/tools/build_rust.py:18-36`。

## 4. csrc 的核心能力分类

### 4.1 Attention 与 KV Cache

相关文件：

- `csrc/attention/attention_generic.cuh`
- `csrc/attention/attention_dtypes.h`
- `csrc/libtorch_stable/attention/paged_attention_v1.cu`
- `csrc/libtorch_stable/attention/paged_attention_v2.cu`
- `csrc/libtorch_stable/attention/merge_attn_states.cu`
- `csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu`
- `csrc/libtorch_stable/cache_kernels.cu`
- `csrc/libtorch_stable/cache_kernels_fused.cu`

能力：

- PagedAttention v1/v2 decode
- split-KV attention state merge
- MLA decode / CUTLASS MLA
- KV cache reshape/cache/flash cache
- block swap / batch swap
- gather and dequant cache
- FP8/NVFP4 KV cache 相关处理

### 4.2 Activation / Norm / Positional Encoding / Sampling

相关文件：

- `csrc/libtorch_stable/activation_kernels.cu`
- `csrc/libtorch_stable/layernorm_kernels.cu`
- `csrc/libtorch_stable/layernorm_quant_kernels.cu`
- `csrc/libtorch_stable/pos_encoding_kernels.cu`
- `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu`
- `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`
- `csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`
- `csrc/libtorch_stable/sampler.cu`
- `csrc/libtorch_stable/topk.cu`

能力：

- SiLU/GELU/SwiGLU/FATReLU 等激活
- RMSNorm / fused add RMSNorm
- RMSNorm + quant fused
- RoPE / QK norm + RoPE fused
- DeepSeek/MiniMax 特定 fused kernel
- repetition penalty、top-k、persistent top-k

### 4.3 Quantization / GEMM

相关文件：

- `csrc/libtorch_stable/quantization/w8a8/**`
- `csrc/libtorch_stable/quantization/fp4/**`
- `csrc/libtorch_stable/quantization/awq/**`
- `csrc/libtorch_stable/quantization/gptq/**`
- `csrc/libtorch_stable/quantization/gptq_allspark/**`
- `csrc/libtorch_stable/quantization/marlin/**`
- `csrc/libtorch_stable/quantization/machete/**`
- `csrc/libtorch_stable/quantization/cutlass_w4a8/**`
- `csrc/libtorch_stable/quantization/fused_kernels/**`

能力：

- INT8 / FP8 / FP4 / NVFP4 / MXFP4 quant
- CUTLASS scaled MM
- CUTLASS grouped GEMM
- AWQ GEMM / dequant
- GPTQ GEMM / shuffle
- Marlin / Machete kernels
- AllSpark W8A16 kernels
- DeepSeek V3 fused A GEMM、router GEMM

### 4.4 MoE

相关文件：

- `csrc/libtorch_stable/moe/torch_bindings.cpp`
- `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`
- `csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu`
- `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu`
- `csrc/libtorch_stable/moe/moe_permute_unpermute_op.cu`
- `csrc/libtorch_stable/moe/permute_unpermute_kernels/**`
- `csrc/libtorch_stable/moe/marlin_moe_wna16/**`
- `csrc/libtorch_stable/moe/mxfp8_moe/**`

能力：

- top-k routing
- token/expert 对齐 padding
- permute / unpermute
- grouped topk
- moe_sum
- quantized MoE GEMM
- Marlin MoE
- DeepSeek V3 router GEMM

### 4.5 CPU 后端

相关文件：

- `csrc/cpu/torch_bindings.cpp`
- `csrc/cpu/cpu_attn.cpp`
- `csrc/cpu/cpu_attn_*.hpp`
- `csrc/cpu/dnnl_kernels.cpp`
- `csrc/cpu/sgl-kernels/**`
- `csrc/cpu/cpu_fused_moe.cpp`
- `csrc/cpu/cpu_wna16.cpp`
- `csrc/cpu/spec_decode_utils.cpp`
- `cmake/cpu_extension.cmake`

能力：

- CPU attention + KV cache
- activation / layernorm / RoPE
- oneDNN matmul / scaled_mm
- SGL CPU GEMM / INT4 / INT8 / FP8 / MoE
- shared memory collectives
- speculative decoding utilities
- 多 ISA 编译：AVX2、AVX512、AMX、ARM NEON/BF16、Power、S390、RISC-V RVV

## 5. Python 到 native 的入口

`vllm/_custom_ops.py` 是 Python 层的主要 wrapper。它启动时调用：

```python
current_platform.import_kernels()
```

源码位置：`code/vllm/vllm/_custom_ops.py:20`。

这一步会根据平台加载 `_C`、`_moe_C`、`_rocm_C` 等 extension，然后 wrapper 函数调用 `torch.ops._C.xxx` 或 `torch.ops._rocm_C.xxx`。

典型 wrapper：

- `paged_attention_v1()` -> `torch.ops._C.paged_attention_v1`：`code/vllm/vllm/_custom_ops.py:113-155`
- `paged_attention_v2()` -> `torch.ops._C.paged_attention_v2`：`code/vllm/vllm/_custom_ops.py:158-205`
- `paged_attention_rocm()` -> `torch.ops._rocm_C.paged_attention`：`code/vllm/vllm/_custom_ops.py:208-251`
- `rotary_embedding()` -> `torch.ops._C.rotary_embedding`：`code/vllm/vllm/_custom_ops.py:289-313`
- `rms_norm()` -> `torch.ops._C.rms_norm`：`code/vllm/vllm/_custom_ops.py:316-323`
- `cutlass_scaled_mm()` -> `torch.ops._C.cutlass_scaled_mm`：`code/vllm/vllm/_custom_ops.py:813-861`

## 6. 底层加速层和上层 engine 的边界

上层模型执行过程通常会进入：

```text
model_runner
  -> model_executor layer
  -> attention backend / quantization method / MoE layer
  -> vllm._custom_ops wrapper
  -> torch.ops._C / _moe_C / _rocm_C
  -> native kernel
```

底层 native 层不关心 HTTP 请求、scheduler 策略、tokenizer 或 OpenAI 协议。它只接受已经准备好的 tensors 和标量参数，完成高性能计算或数据搬运。

## 7. 总结

vLLM 的底层加速层可以分成三块：

1. GPU native ops：CUDA/HIP kernels，覆盖 attention、KV cache、quant GEMM、MoE、采样、norm、activation。
2. CPU native ops：oneDNN/SGL/ISA-specialized CPU kernels，覆盖 CPU inference 所需的 attention、GEMM、MoE、norm、activation。
3. Rust frontend/native protocol layer：用 Rust 实现 HTTP/gRPC/chat/text/tokenizer/frontend，并通过 ZMQ 连接 Python engine。