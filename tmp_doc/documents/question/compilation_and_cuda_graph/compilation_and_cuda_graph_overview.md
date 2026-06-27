# vLLM V1 Compilation / CUDA Graph 逻辑梳理

源码位置：

- `code/vllm/vllm/config.py`
- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/compilation/`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `code/vllm/vllm/v1/attention/backend.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/layers/`
- `code/vllm/vllm/utils.py`

本文按“先定边界，再走初始化 capture，再走 runtime dispatch，最后拆 padding、attention metadata、fallback 和调试”的方式，梳理 vLLM V1 中 compilation 与 CUDA graph 的关系。

它和 `executor_worker_model_runner` 的执行链路关系很密切：

```text
SchedulerOutput
  → ModelRunner 准备输入
  → 判断本轮是否可用 cudagraph
  → 必要时 padding 到固定 batch shape
  → 构造 capture / replay 兼容的 attention metadata
  → set_forward_context(...)
  → model forward
      → eager / compiled / cudagraph replay
  → logits / sampler / output
```

---

## 0. 梳理规划

本目录要回答的问题分成 11 组：

```text
1. Compilation 和 CUDA graph 在 vLLM 中分别解决什么问题？
2. CompileConfig、cudagraph 配置和 runtime mode 如何决定执行路径？
3. Worker warmup / compile / CUDA graph capture 生命周期是什么？
4. 动态 batch 如何通过 padding / shape bucket 变成可 replay 形态？
5. ModelRunner 每轮如何选择 eager、compiled、cudagraph replay？
6. Attention metadata 在 capture 和 replay 时有什么特殊路径？
7. model forward 如何被 compile wrapper / graph runner 包装？
8. sampler / output 是否也被 capture，和 forward graph 的边界在哪里？
9. TP / PP / DP 并行如何影响 compilation / cudagraph？
10. 哪些功能会导致 fallback 或禁用 cudagraph？
11. 如何调试 cudagraph miss、compile overhead、shape mismatch 和 graph replay 问题？
```

阅读顺序建议：

```text
compilation_and_cuda_graph_overview.md
  → 01_compilation_cuda_graph_role.md
  → 02_compile_config_and_runtime_modes.md
  → 03_warmup_and_capture_lifecycle.md
  → 04_batch_padding_and_shape_stability.md
  → 05_cudagraph_dispatch_flow.md
  → 06_attention_metadata_capture.md
  → 07_model_forward_compile_wrapper.md
  → 08_sampler_and_output_interaction.md
  → 09_parallelism_and_cudagraph.md
  → 10_limitations_and_fallbacks.md
  → 11_debugging_and_metrics.md
```

---

## 1. 一句话回答

Compilation / CUDA graph 的目标是减少 Python 调度、kernel launch 和动态图开销。

```text
torch.compile：
  尽量把模型 forward 编译成更高效的图或 kernel 组合。

CUDA graph：
  对固定 shape 的 GPU 执行序列做 capture，运行时直接 replay，减少 launch overhead。
```

在 vLLM 中，它们共同面对一个核心矛盾：

```text
LLM serving 的 batch 是动态的，
但 CUDA graph replay 需要固定 shape 和稳定内存地址。
```

因此 vLLM 需要：

```text
动态请求
  → batch descriptor / padding / shape bucket
  → capture-compatible metadata
  → graph replay
  → 不满足条件时 fallback eager
```

---

## 2. 总体流程图

初始化阶段：

```text
Worker.initialize_from_config()
  → allocate KV cache
  → warmup model
  → compile model if enabled
  → capture CUDA graphs for selected batch sizes / shapes
  → 保存 graph runners / static buffers
```

运行阶段：

```text
SchedulerOutput
  → GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
  → _determine_batch_execution_and_padding()
      → 判断 cudagraph / eager / padding
  → _get_slot_mappings()
  → _build_attention_metadata()
      → capture / replay 兼容路径
  → _preprocess()
  → set_forward_context(...)
  → _model_forward()
      → eager / compiled / cudagraph replay
  → logits / pooling / sampler
```

---

## 3. 核心概念先占位

```text
CompileConfig：
  控制 torch.compile、cudagraph、capture sizes、dynamic shapes、splitting 等配置。

Cudagraph runtime mode：
  描述本轮是否 capture、replay、eager 或 fallback。

Batch descriptor：
  描述本轮 batch 的执行形态，用于决定 padding、ubatch、graph path。

Static input buffers：
  CUDA graph replay 要求地址稳定，因此需要固定 buffer。

Padding：
  把动态 token / request 数补到 capture 过的形状。

Capture attention metadata：
  metadata 也要满足固定 shape 或可 update 的约束。

Graph replay：
  重用 capture 过的 GPU 执行图。
```

---

## 4. 和其他专题的关系

```text
executor_worker_model_runner：
  解释 ModelRunner 主执行链路，本专题细化其中 compile / cudagraph 分支。

attention：
  attention metadata builder 需要支持 cudagraph capture / replay。

parallelism：
  TP / PP / DP 下 graph capture 要考虑 rank、collective 和 intermediate tensors。

sampling_and_output：
  sampler 通常在 forward graph 边界之外，但 logits shape 和 async output 会受影响。

quantization / lora：
  某些量化 kernel、动态 LoRA mapping 可能限制 cudagraph 或 compile。
```

---

## 5. 后续专题占位

```text
01_compilation_cuda_graph_role.md：
  定义 torch.compile 和 CUDA graph 在 vLLM 中的职责、区别和边界。

02_compile_config_and_runtime_modes.md：
  梳理 compile config、cudagraph 配置、runtime mode 和执行路径选择。

03_warmup_and_capture_lifecycle.md：
  梳理 Worker 初始化、warmup、compile、capture 和 graph 缓存生命周期。

04_batch_padding_and_shape_stability.md：
  梳理动态 batch 如何 padding 到固定 shape，以及 shape bucket 如何选择。

05_cudagraph_dispatch_flow.md：
  梳理 ModelRunner 每轮如何判断使用 eager、compiled 还是 cudagraph replay。

06_attention_metadata_capture.md：
  梳理 attention metadata 在 cudagraph capture / replay 下的特殊构造和 update 机制。

07_model_forward_compile_wrapper.md：
  梳理模型 forward 如何被 compile wrapper、graph runner 或 static buffer 包装。

08_sampler_and_output_interaction.md：
  梳理 logits、sampler、pooling、output 与 cudagraph forward 的边界。

09_parallelism_and_cudagraph.md：
  梳理 TP / PP / DP / collective 对 compile 和 graph replay 的影响。

10_limitations_and_fallbacks.md：
  梳理哪些功能会禁用 cudagraph、导致 graph miss 或 fallback。

11_debugging_and_metrics.md：
  梳理 cudagraph stats、compile metrics、profile 和常见问题定位。
```

---

## 6. 一句话总结

Compilation / CUDA graph 是 vLLM 在执行层降低运行时开销的机制：

```text
它通过编译、固定 shape、padding、static buffers 和 graph replay，
把动态 serving 尽量转成可复用的高效 GPU 执行路径；
如果条件不满足，则必须安全回退到 eager。
```
