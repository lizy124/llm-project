# vLLM C++ / CUDA / Rust 底层加速层梳理

本目录系统梳理 `D:/lzy/project/kv_pool/code/vllm` 中的底层加速层，重点覆盖：

- `csrc/`：C++ / CUDA / HIP / CPU 原生算子实现
- `csrc/libtorch_stable/`：基于 PyTorch stable ABI 的主要 GPU 自定义算子库
- `csrc/cpu/`：CPU 后端、oneDNN、SGL CPU kernels、CPU attention、CPU MoE
- `csrc/rocm/`：ROCm/HIP 专用扩展
- `csrc/cutlass_extensions/`：CUTLASS 扩展与 epilogue/type 工具
- `rust/`：Rust frontend、chat/template/parser/tokenizer/server/engine-core-client
- Python 调用桥：`vllm/_custom_ops.py`、`torch.ops._C`、`torch.ops._moe_C`、`torch.ops._rocm_C`
- 构建系统：`CMakeLists.txt`、`cmake/cpu_extension.cmake`、`tools/build_rust.py`、Cargo workspace

## 文档索引

1. [01_native_acceleration_layer_overview.md](01_native_acceleration_layer_overview.md)
   - 总体结构、模块分层、C++/CUDA/Rust 在 vLLM 中的位置。
2. [02_build_system_and_extension_libs.md](02_build_system_and_extension_libs.md)
   - CMake、CUDA/HIP/CPU 分支、extension target、Rust 构建。
3. [03_python_to_native_operator_bridge.md](03_python_to_native_operator_bridge.md)
   - `current_platform.import_kernels()`、`torch.ops._C`、fake/meta 注册、Python wrapper。
4. [04_cuda_core_ops_attention_and_kv_cache.md](04_cuda_core_ops_attention_and_kv_cache.md)
   - PagedAttention、MLA、KV cache reshape/cache/gather/swap、RoPE/fused kernels。
5. [05_quant_gemm_cutlass_marlin_machete.md](05_quant_gemm_cutlass_marlin_machete.md)
   - INT8/FP8/FP4/NVFP4/AWQ/GPTQ/CUTLASS/Marlin/Machete/AllSpark。
6. [06_moe_low_level_ops.md](06_moe_low_level_ops.md)
   - topk、token/expert 对齐、permute/unpermute、grouped GEMM、MoE quant。
7. [07_cpu_backend.md](07_cpu_backend.md)
   - CPU attention、oneDNN、AVX/AMX/ARM/Power/RISC-V、SGL kernels、CPU MoE。
8. [08_rocm_memory_comm_and_external_projects.md](08_rocm_memory_comm_and_external_projects.md)
   - ROCm 扩展、custom all-reduce、cumem allocator、external projects。
9. [09_rust_low_level_and_frontend.md](09_rust_low_level_and_frontend.md)
   - Rust workspace、server、chat、text、LLM、engine-core-client、ZMQ 协议边界。
10. [10_end_to_end_call_chain.md](10_end_to_end_call_chain.md)
   - 从模型层 Python 到 native kernel 的调用路径，按 generate/attention/MoE/quant/Rust frontend 串联。

## 一句话总结

vLLM 的底层加速层由两条主线组成：一条是 `csrc/` 编译出的 PyTorch native extension，负责 attention、KV cache、量化 GEMM、MoE、采样、CPU/ROCm 等高性能算子；另一条是 `rust/` 的 Rust frontend/协议层，负责更高性能的北向 HTTP/gRPC/chat/template/parser/tokenizer，并通过 ZMQ/MessagePack 连接 Python engine core。