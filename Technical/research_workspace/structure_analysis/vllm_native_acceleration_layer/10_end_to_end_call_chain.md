# 10 端到端调用链

## 1. 本文目标

前面文档分别拆解了 C++/CUDA/CPU/ROCm/Rust 模块。本文把它们串起来，看一次 vLLM 请求如何进入 native 层。

重点链路：

1. Python online serving 到 CUDA attention。
2. Python model layer 到 KV cache kernel。
3. quantized linear 到 CUTLASS/Marlin/Machete/GPTQ/AWQ。
4. MoE layer 到 `_moe_C`。
5. CPU 后端调用 native ops。
6. Rust frontend 到 Python EngineCore。

## 2. Python online serving 到 CUDA kernel

在线服务请求路径：

```text
HTTP /v1/chat/completions
  -> OpenAIServingChat.create_chat_completion
  -> engine_client.generate
  -> AsyncLLM.add_request
  -> EngineCoreClient.add_request_async
  -> EngineCore / Scheduler
  -> Worker / ModelRunner
  -> model_executor layers
  -> attention backend / quantization / MoE
  -> vllm._custom_ops
  -> torch.ops._C / _moe_C / _rocm_C
  -> native kernel
```

入口层和 engine 边界已经在 `vllm_entrypoints_api_layer` 文档中梳理。底层加速层从 `model_executor layers -> _custom_ops` 开始接管。

## 3. Attention decode 链路

典型 decode attention：

```text
Python attention backend
  -> vllm._custom_ops.paged_attention_v1/v2
  -> torch.ops._C.paged_attention_v1/v2
  -> csrc/libtorch_stable/torch_bindings.cpp
  -> csrc/libtorch_stable/attention/paged_attention_v1.cu/v2.cu
  -> csrc/attention/attention_generic.cuh
  -> dtype-specific implementation
```

关键源码：

- Python wrapper：`code/vllm/vllm/_custom_ops.py:113-205`
- schema：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:594-619`
- impl 注册：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:737-738`
- 声明：`code/vllm/csrc/libtorch_stable/ops.h:424-448`
- 实现文件：`csrc/libtorch_stable/attention/paged_attention_v1.cu`、`csrc/libtorch_stable/attention/paged_attention_v2.cu`

ROCm 对应链路：

```text
vllm._custom_ops.paged_attention_rocm
  -> torch.ops._rocm_C.paged_attention
  -> csrc/rocm/torch_bindings.cpp
  -> csrc/rocm/attention.cu
```

关键源码：`code/vllm/vllm/_custom_ops.py:208-251`、`code/vllm/csrc/rocm/torch_bindings.cpp:64-82`。

## 4. KV cache 写入链路

KV cache 写入通常发生在 prefill/decode 的 attention 前后。

```text
model_runner / attention layer
  -> key/value tensor + slot_mapping
  -> vllm._custom_ops.reshape_and_cache 或 concat_and_cache_mla
  -> torch.ops._C_cache_ops.reshape_and_cache / concat_and_cache_mla
  -> csrc/libtorch_stable/torch_bindings.cpp _C_cache_ops fragment
  -> csrc/libtorch_stable/cache_kernels.cu
  -> paged KV cache memory
```

关键源码：

- cache schema：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:780-866`
- cache declaration：`code/vllm/csrc/cache.h:9-94`

对于 fused DeepSeek/MiniMax kernel，链路会更短：

```text
Q/K norm + RoPE + KV insert
  -> single fused op
  -> fused_*_qnorm_rope_kv_insert_kernel.cu
```

接口：

- DeepSeek V4：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:433-451`
- MiniMax M3：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:465-476`

## 5. Quantized linear / GEMM 链路

### 5.1 CUTLASS scaled_mm

```text
Python quantization layer
  -> vllm._custom_ops.cutlass_scaled_mm
  -> shape/platform 判断
  -> torch.ops._C.cutlass_scaled_mm
  -> csrc/libtorch_stable/torch_bindings.cpp
  -> csrc/libtorch_stable/quantization/w8a8/cutlass/**
  -> CUTLASS GEMM kernel
```

关键源码：

- Python wrapper：`code/vllm/vllm/_custom_ops.py:813-861`
- schema：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:100-106`
- impl：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:633-635`
- CMake arch 分支：`code/vllm/CMakeLists.txt:733-859`

### 5.2 AWQ

```text
AWQ layer
  -> vllm._custom_ops.awq_gemm / awq_dequantize
  -> torch.ops._C.awq_gemm / awq_dequantize
  -> csrc/libtorch_stable/quantization/awq/gemm_kernels.cu
```

关键源码：`code/vllm/vllm/_custom_ops.py:548-593`、`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:301-309`。

如果 `VLLM_USE_TRITON_AWQ` 开启，则 Python wrapper 直接走 Triton fallback：`code/vllm/vllm/_custom_ops.py:556-562`、`code/vllm/vllm/_custom_ops.py:589-593`。

### 5.3 GPTQ

```text
GPTQ layer
  -> vllm._custom_ops.gptq_gemm
  -> torch.ops._C.gptq_gemm
  -> csrc/libtorch_stable/quantization/gptq/q_gemm.cu
```

关键源码：`code/vllm/vllm/_custom_ops.py:614-656`、`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:564-574`。

ROCm RDNA3：

```text
vllm._custom_ops.gptq_gemm_rdna3
  -> torch.ops._rocm_C.gptq_gemm_rdna3
  -> csrc/rocm/q_gemm_rdna3.cu
```

关键源码：`code/vllm/vllm/_custom_ops.py:659-702`、`code/vllm/csrc/rocm/torch_bindings.cpp:42-52`。

### 5.4 Marlin / Machete

Marlin/Machete 通常由 quantization layer 根据模型格式、硬件能力、schedule 选择。

Marlin 注册：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:70-97`。

Machete 注册：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:37-68`。

CMake 会自动生成 kernel source：

- Machete：`code/vllm/CMakeLists.txt:471-538`
- Marlin：`code/vllm/CMakeLists.txt:544-681`

## 6. MoE 链路

典型 MoE 执行：

```text
Python MoE layer
  -> router logits
  -> torch.ops._moe_C.topk_softmax / grouped_topk
  -> torch.ops._moe_C.moe_align_block_size
  -> torch.ops._moe_C.moe_permute
  -> torch.ops._C.cutlass_moe_mm 或 torch.ops._moe_C.moe_wna16_marlin_gemm
  -> torch.ops._moe_C.moe_unpermute
  -> torch.ops._moe_C.moe_sum
```

关键源码：

- `_moe_C` schema：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:6-128`
- `_moe_C` impl：`code/vllm/csrc/libtorch_stable/moe/torch_bindings.cpp:131-155`
- `_moe_C` target：`code/vllm/CMakeLists.txt:1131-1353`
- CUTLASS MoE ops：`code/vllm/csrc/libtorch_stable/torch_bindings.cpp:124-166`

## 7. CPU 后端链路

CPU-only 构建下，Python 仍可能调用：

```python
torch.ops._C.xxx
```

但 `_C` 是 CPU extension。

```text
Python wrapper
  -> torch.ops._C.rms_norm / rotary_embedding / cpu_attention_with_kv_cache / onednn_mm / cpu_fused_moe
  -> csrc/cpu/torch_bindings.cpp
  -> csrc/cpu/*.cpp / sgl-kernels / oneDNN
```

关键源码：

- CPU binding：`code/vllm/csrc/cpu/torch_bindings.cpp:263-630`
- CPU CMake target：`code/vllm/cmake/cpu_extension.cmake:401-557`

CPU 后端与 GPU 后端的共性：Python namespace 尽量保持 `_C`，差异在 platform import 和 extension target。

## 8. Rust frontend 链路

Rust frontend 路径绕开 Python FastAPI，但不绕开 Python EngineCore。

```text
HTTP request
  -> vllm-server axum route
  -> AppState
  -> ChatLlm / TextLlm
  -> Llm.generate
  -> EngineCoreClient.call
  -> ZMQ + MessagePack
  -> Python EngineCore
  -> scheduler / worker / model_runner
  -> native C++/CUDA ops
```

关键源码：

- Rust server build state：`code/vllm/rust/src/server/src/lib.rs:47-106`
- Rust server serve：`code/vllm/rust/src/server/src/lib.rs:121-268`
- ChatLlm.chat：`code/vllm/rust/src/chat/src/lib.rs:170-209`
- Llm.generate：`code/vllm/rust/src/llm/src/lib.rs:76-107`
- EngineCoreClient exports：`code/vllm/rust/src/engine-core-client/src/lib.rs:1-16`

## 9. Rust managed Python engine 链路

`vllm-rs serve` 可以自己 spawn Python headless engine：

```text
vllm-rs serve
  -> ManagedEngineHandle::spawn
  -> Python headless engine
  -> Rust frontend connects via handshake address
```

关键源码：`code/vllm/rust/src/cmd/src/main.rs:100-190`。

如果 `data_parallel_size_local == 0`，Rust frontend-only 模式不启动本地 engine，只连接外部 engine：`code/vllm/rust/src/cmd/src/main.rs:103-116`。

## 10. 统一分层图

```text
外部请求层
  ├─ Python FastAPI / CLI
  └─ Rust axum/tonic frontend

协议与渲染层
  ├─ Python OpenAIServingRender / Tokenization
  └─ Rust vllm-chat / vllm-text

Engine client 层
  ├─ Python AsyncLLM / LLMEngine
  └─ Rust EngineCoreClient + Llm

EngineCore / Scheduler / Worker
  └─ Python V1 engine

模型执行层
  └─ model_executor / attention backend / quantization / MoE

Native ops 层
  ├─ torch.ops._C
  ├─ torch.ops._moe_C
  ├─ torch.ops._rocm_C
  └─ CPU _C

Kernel 层
  ├─ CUDA / CUTLASS / Marlin / Machete / Triton external
  ├─ HIP / ROCm
  └─ CPU oneDNN / SGL / ISA kernels
```

## 11. 哪些层是“底层加速层”

严格来说：

- `csrc/` 是 tensor compute 和 memory movement 的底层加速层。
- `rust/` 是 serving/protocol/frontend 的底层 runtime 加速层。
- `cmake/`、`tools/build_rust.py` 是构建支撑层。
- `vllm/_custom_ops.py` 是 Python 到 native 的桥，不是 kernel，但必须一起看。

## 12. 最重要的判断方法

如果你看到 Python 代码调用：

```python
torch.ops._C.xxx
torch.ops._moe_C.xxx
torch.ops._rocm_C.xxx
```

就说明已经进入 native op 边界。

如果你看到 Rust 代码调用：

```rust
EngineCoreClient::connect(...)
client.call(...)
```

说明 Rust frontend 已经越过 HTTP/chat/text 层，进入 Python engine core 通信边界。

## 13. 关键结论

vLLM 的底层加速不是一个单点，而是两条端到端路径：

1. Python model execution -> native tensor kernels。
2. Rust frontend -> Python EngineCore -> native tensor kernels。

两条路径最终都会进入同一套 engine/model/native ops，只是 frontend 和协议处理的位置不同。理解这些边界后，就能定位某个性能问题到底属于：API overhead、engine scheduling、attention/KV cache、quant GEMM、MoE routing/GEMM、CPU/ROCm 后端，还是 Rust frontend transport。