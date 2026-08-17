# 01. Compilation / CUDA graph 在 vLLM 中负责什么？

源码位置：

- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/compilation/wrapper.py`
- `code/vllm/vllm/compilation/cuda_graph.py`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/v1/cudagraph_dispatcher.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：torch.compile 和 CUDA graph 在 vLLM V1 中各自负责什么、边界在哪里，以及它们如何嵌入 `GPUModelRunner.execute_model()` 主链路。

---

## 1. 一句话回答

Compilation / CUDA graph 都是执行优化层，但优化对象不同：

```text
torch.compile / vLLM compile：
  优化模型 forward 的 Python / Dynamo / Inductor / custom pass 路径，
  尽量把 forward 转成更稳定、更高效的 compiled graph 或 kernel 组合。

CUDA graph：
  对固定 shape、固定内存地址、固定 kernel launch 序列做 capture，
  运行时 replay，减少每步 decode / small batch 的 kernel launch overhead。
```

它们共同面对的核心矛盾是：

```text
LLM serving 的请求、token 数、attention metadata 每轮都动态变化；
CUDA graph replay 却要求 shape 和地址稳定。
```

所以 vLLM 的做法是：

```text
动态 SchedulerOutput
  → ModelRunner 准备 persistent buffers
  → padding 到 capture bucket
  → 构造 capture/replay 兼容 metadata
  → set_forward_context(cudagraph_runtime_mode, batch_descriptor)
  → compiled forward / cudagraph replay
  → 不满足条件时回退 NONE/eager 路径
```

---

## 2. 它们分别解决什么问题

### 2.1 torch.compile / vLLM compile

`CompilationMode` 定义在 `config/compilation.py`：

```text
NONE                 ：不使用 torch.compile。
STOCK_TORCH_COMPILE  ：使用标准 torch.compile。
DYNAMO_TRACE_ONCE    ：只做一次 Dynamo trace，避免后续 recompile。
VLLM_COMPILE         ：vLLM 自定义 Inductor backend、piecewise compilation、custom passes。
```

源码位置：`code/vllm/vllm/config/compilation.py:37`

它主要优化：

```text
- Python eager op 调度开销；
- Dynamo / Inductor 编译后的 graph 执行效率；
- vLLM 自定义 Inductor pass 和 fusion；
- piecewise compilation 中对 attention / KV update 等 unsafe op 的切分；
- compiled graph 缓存和复用。
```

### 2.2 CUDA graph

`CUDAGraphMode` 也定义在 `config/compilation.py`：

```text
NONE               ：不 capture / replay CUDA graph。
PIECEWISE          ：只 capture cudagraph-safe 的 piecewise 子图。
FULL               ：capture 整个 forward 图。
FULL_DECODE_ONLY   ：decode batch 走 FULL，mixed/prefill 不走 FULL。
FULL_AND_PIECEWISE ：decode 走 FULL，prefill/mixed 走 PIECEWISE。
```

源码位置：`code/vllm/vllm/config/compilation.py:53`

它主要优化：

```text
- 小 batch / decode 阶段大量 kernel launch 的 CPU overhead；
- Python 调度和 CUDA launch 序列重复构造；
- 固定 shape 下反复执行同一批 GPU kernel 的开销。
```

### 2.3 二者不是同一层

源码注释明确说明：

```text
CUDA graph 逻辑总体上和 compilation 逻辑正交；
piecewise cudagraph 需要 piecewise compilation，
但 full cudagraph 可以在有无 compilation 的情况下工作。
```

源码位置：`code/vllm/vllm/config/compilation.py:615`

所以可以记成：

```text
compile：让 forward 代码本身更适合高效执行。
cudagraph：把某次固定形态的 GPU 执行序列记录下来，下次直接 replay。
```

---

## 3. 在执行链路中的位置

一次正常 `execute_model()` 中，相关位置是：

```text
GPUModelRunner.execute_model()
  → _prepare_inputs()
  → _determine_batch_execution_and_padding()
      → CudagraphDispatcher.dispatch(...)
      → 得到 cudagraph_mode 和 BatchDescriptor
  → _get_slot_mappings(... padded ...)
  → _build_attention_metadata(... padded ...)
  → _preprocess()
  → set_forward_context(...)
      → cudagraph_runtime_mode
      → batch_descriptor
      → attn_metadata
      → slot_mapping
  → _model_forward()
      → self.model(...)
      → CUDAGraphWrapper / compiled wrapper 决定实际执行方式
```

关键入口：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`
- `code/vllm/vllm/forward_context.py:249`

---

## 4. `ForwardContext` 是运行时控制总线

`ForwardContext` 保存当前 forward 所需的动态运行时信息：

```text
attn_metadata
slot_mapping
dp_metadata
cudagraph_runtime_mode
batch_descriptor
ubatch_slices
skip_compiled
```

源码位置：`code/vllm/vllm/forward_context.py:128`

`CUDAGraphWrapper.__call__()` 不自己判断 batch 是否该 graph，而是读取 forward context：

```text
batch_descriptor = forward_context.batch_descriptor
cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
```

源码位置：`code/vllm/vllm/compilation/cuda_graph.py:240`

这形成一个重要边界：

```text
ModelRunner / CudagraphDispatcher：
  决定本轮 runtime mode 和 padded batch descriptor。

CUDAGraphWrapper：
  信任 forward context，根据 key capture 或 replay。
```

---

## 5. 它们不负责什么

Compilation / CUDA graph 不改变模型语义，也不替代调度层。

它们不负责：

```text
- 不决定哪些 request 被调度；
- 不分配 KV cache blocks；
- 不改变 attention 数学语义；
- 不改变 sampler / structured output 规则；
- 不构造最终 RequestOutput；
- 不保证所有动态功能都能被 capture。
```

这些仍然分别属于 Scheduler、KVCacheManager、attention backend、sampler、OutputProcessor。

---

## 6. CUDA graph 的边界在哪里

`CUDAGraphWrapper` 注释说明它不管理 persistent buffers，也不把输入复制到内部 static buffer：

```text
CUDAGraphWrapper does not store persistent buffers or copy runtime inputs.
We assume implementing them is done outside of the wrapper.
```

源码位置：`code/vllm/vllm/compilation/cuda_graph.py:161`

因此：

```text
ModelRunner 负责：
  input_ids / positions / slot_mapping / block_table / attention metadata 的稳定 buffer 和 padding。

CUDAGraphWrapper 负责：
  对 batch_descriptor 对应的 runnable 做 capture/replay。
```

---

## 7. 为什么 vLLM 需要这些优化

decode 阶段每步通常只追加少量 token：

```text
batch size 可能较小；
每 token 的 GPU kernel 数很多；
Python 调度和 launch overhead 占比变大；
请求持续生成时这个开销会重复很多次。
```

CUDA graph replay 可以把固定 shape 的 kernel launch 序列复用；torch.compile / vLLM compile 可以减少 eager op 组织、融合可融合算子、减少 Python 和 Inductor runtime overhead。

---

## 8. 一句话总结

```text
Compilation / CUDA graph 是 vLLM 执行层的性能优化机制：
compile 优化 forward 图本身，CUDA graph 复用固定 shape 的 GPU launch 序列；
它们不改变调度和模型语义，只要求 ModelRunner 把动态 serving batch 整理成可编译、可 capture、可 replay 的形态。
```
