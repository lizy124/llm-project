# 02. CompileConfig 和 runtime mode 如何决定执行路径？

源码位置：

- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/config/observability.py`
- `code/vllm/vllm/compilation/wrapper.py`
- `code/vllm/vllm/compilation/backends.py`
- `code/vllm/vllm/compilation/cuda_graph.py`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/v1/cudagraph_dispatcher.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/attention/backend.py`

本问题关注：`CompilationConfig` 如何决定 vLLM 能否使用 torch.compile / vLLM compile / CUDA graph；`CUDAGraphMode` 如何表达配置层和运行时的不同模式；`GPUModelRunner` 每轮如何把真实 batch 归一成 `BatchDescriptor`，并通过 `ForwardContext` 让 wrapper 选择 eager、compiled、capture 或 replay。

---

## 0. 梳理规划

本篇按“配置对象 → 模式语义 → 初始化修正 → 运行时 dispatch → wrapper 执行”的顺序梳理。

要回答的问题分成 10 组：

```text
1. CompilationConfig 管哪些配置？
2. CompilationMode 和 CUDAGraphMode 有什么区别？
3. NONE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE / VLLM_COMPILE 分别意味着什么？
4. NONE / PIECEWISE / FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE 分别意味着什么？
5. cudagraph_capture_sizes / max_cudagraph_capture_size / compile_sizes 如何协作？
6. 配置阶段会在哪些情况下自动降级 cudagraph_mode？
7. attention backend 如何进一步修正最终 runtime mode？
8. GPUModelRunner 每轮如何得到 CUDAGraphMode 和 BatchDescriptor？
9. ForwardContext 和 CUDAGraphWrapper 如何消费 runtime mode？
10. `cudagraph_runtime_mode=NONE` 和“没有 compile”为什么不是一回事？
```

---

## 1. 一句话回答

`CompilationConfig` 决定系统“最多能使用哪些优化路径”，包括 torch.compile、vLLM compile、CUDA graph mode、capture sizes、compile sizes 和 warmup 次数；runtime mode 决定“这一轮实际走哪条路径”，可能是 `FULL` CUDA graph replay、`PIECEWISE` 子图 replay、或者 `NONE` pass-through。

最小主线是：

```text
CompilationConfig
  → 配置归一和自动降级
  → attention backend 能力修正
  → CudagraphDispatcher.initialize_cudagraph_keys()
  → capture_model() 捕获合法 BatchDescriptor
  → 每轮 _determine_batch_execution_and_padding()
  → CudagraphDispatcher.dispatch()
  → set_forward_context(cudagraph_runtime_mode, batch_descriptor)
  → CUDAGraphWrapper / compile wrapper 执行 replay 或 pass-through
```

一句话记忆：

```text
配置决定“能不能 graph / compile 以及 capture 哪些形状”，runtime mode 决定“这一轮实际 replay 哪个 graph，或回退到普通执行”。
```

---

## 2. CompilationConfig 是什么

`CompilationConfig` 定义在：`code/vllm/vllm/config/compilation.py:379`

它是 vLLM compilation / CUDA graph 的配置中心。

可以把字段分成 6 类。

### 2.1 compile 模式相关

核心字段：

```text
mode
backend
custom_ops
splitting_ops
use_inductor_graph_partition
compile_sizes
compile_ranges
compile_ranges_endpoints
debug_dump_path
```

这些字段决定：

```text
- 是否启用 torch.compile；
- 使用 stock torch.compile 还是 vLLM 自定义 backend；
- 是否进行 piecewise graph split；
- 哪些 op 作为 splitting ops；
- 哪些 shape 要提前编译；
- 是否 dump compile 调试信息。
```

其中 `compile_sizes` 定义在：`code/vllm/vllm/config/compilation.py:556`

它可以显式列出 static compile shape，也可以包含特殊值：

```text
"cudagraph_capture_sizes"
```

表示把 cudagraph capture sizes 同步作为 compile sizes。

### 2.2 CUDA graph 模式相关

核心字段：

```text
cudagraph_mode
cudagraph_num_of_warmups
cudagraph_capture_sizes
max_cudagraph_capture_size
cudagraph_copy_inputs
cudagraph_specialize_lora
```

位置：

- `code/vllm/vllm/config/compilation.py:588`
- `code/vllm/vllm/config/compilation.py:624`
- `code/vllm/vllm/config/compilation.py:629`
- `code/vllm/vllm/config/compilation.py:673`

它们决定：

```text
- 是否 capture / replay CUDA graph；
- capture 哪些 batch token size；
- 最大 capture size；
- capture 前 warmup 几次；
- piecewise cudagraph 是否使用 static input copy；
- LoRA active adapter 数是否进入 graph key。
```

### 2.3 multimodal encoder 相关

多模态 encoder 有单独配置：

```text
compile_mm_encoder
cudagraph_mm_encoder
encoder_cudagraph_token_budgets
encoder_cudagraph_max_vision_items_per_batch
encoder_cudagraph_max_frames_per_batch
```

相关位置：`code/vllm/vllm/config/compilation.py:516`

原因是：

```text
multimodal encoder 的动态维度是图像数量、token budget、frames 等，
和 decoder forward 的 num_tokens / num_reqs 不是同一组 shape key。
```

### 2.4 pass config / custom pass 相关

`CompilationConfig` 还包含 pass config，例如：

```text
enable_sp
fuse_gemm_comms
fuse_attn_quant
```

这些字段会影响 graph split、sequence parallelism、GEMM + communication fusion、quantization attention 等路径。

它们看起来不是 cudagraph 字段，但会影响：

```text
- 是否能使用 piecewise cudagraph；
- capture sizes 是否要满足 TP / SP 对齐；
- 某些 fused path 是否只能 full graph。
```

### 2.5 observability / metrics 相关

运行时 graph 统计不在 `CompilationConfig`，而在 `ObservabilityConfig` 中打开：

```text
cudagraph_metrics
```

位置：`code/vllm/vllm/config/observability.py:56`

打开后，`GPUModelRunner` 会为每轮生成：

```text
num_unpadded_tokens
num_padded_tokens
num_paddings
runtime_mode
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907`

### 2.6 compile / capture 时间记录

`CompilationConfig` 中还记录：

```text
compilation_time
encoder_compilation_time
```

位置：`code/vllm/vllm/config/compilation.py:729`

这些不是用户直接配置的 runtime mode，但对定位 compile overhead 很重要。

---

## 3. CompilationMode：底层 runnable 怎么来

`CompilationMode` 定义在：`code/vllm/vllm/config/compilation.py:37`

它控制底层 model forward 是 eager 还是 compiled。

### 3.1 NONE

```text
CompilationMode.NONE
```

含义：

```text
不启用 torch.compile / vLLM compile。
```

如果同时：

```text
cudagraph_mode = NONE
```

那就是普通 eager forward。

但要注意：

```text
CompilationMode.NONE 不一定等于没有 CUDA graph。
```

full CUDA graph 可以包住 eager forward，因为 full cudagraph 和 compilation 基本正交。

### 3.2 STOCK_TORCH_COMPILE

```text
CompilationMode.STOCK_TORCH_COMPILE
```

含义：

```text
使用 PyTorch 原生 nn.Module.compile / torch.compile。
```

`GPUModelRunner.load_model()` 会走：

```text
self.model.compile(fullgraph=True, backend=...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5273`

这条路径更接近 PyTorch 标准 compile，不使用 vLLM 的 piecewise graph split 主流程。

### 3.3 DYNAMO_TRACE_ONCE

```text
CompilationMode.DYNAMO_TRACE_ONCE
```

含义：

```text
通过 Dynamo 做一次 trace，避免后续服务请求反复 recompile。
```

它仍然使用 vLLM 的 compile wrapper 机制，但不走完整的 vLLM piecewise compilation 设计。

### 3.4 VLLM_COMPILE

```text
CompilationMode.VLLM_COMPILE
```

这是 vLLM 自定义 compile 主路径。

典型流程：

```text
@support_torch_compile
  → TorchCompileWithNoGuardsWrapper
  → torch.compile(..., backend=VllmBackend)
  → Dynamo 捕获 FX graph
  → VllmBackend split_graph(splitting_ops)
  → PiecewiseBackend 编译子图
  → 可选 CUDAGraphWrapper(PIECEWISE)
```

关键源码：

- `code/vllm/vllm/compilation/decorators.py`
- `code/vllm/vllm/compilation/wrapper.py`
- `code/vllm/vllm/compilation/backends.py`
- `code/vllm/vllm/compilation/piecewise_backend.py`

---

## 4. CUDAGraphMode：这一轮是否 graph replay

`CUDAGraphMode` 定义在：`code/vllm/vllm/config/compilation.py:53`

它有两层含义：

```text
配置层：用户希望启用哪种 CUDA graph 策略；
运行时：这一轮实际是 FULL、PIECEWISE 还是 NONE。
```

### 4.1 NONE

```text
CUDAGraphMode.NONE
```

含义：

```text
不 capture / replay CUDA graph。
```

但底层仍可能是 compiled callable。

所以：

```text
cudagraph_runtime_mode=NONE ≠ 完全 eager。
```

它只表示这一轮 `CUDAGraphWrapper` pass-through。

### 4.2 PIECEWISE

```text
CUDAGraphMode.PIECEWISE
```

含义：

```text
只对 vLLM compile 切出的 piecewise 子图 capture / replay。
```

特点：

```text
- attention / KV update / dynamic Python control flow 通常在 graph 外；
- 对 metadata 动态性更宽容；
- 收益通常小于 full graph；
- 依赖 vLLM compile / piecewise graph boundary。
```

### 4.3 FULL

```text
CUDAGraphMode.FULL
```

含义：

```text
capture / replay 整个 model forward。
```

特点：

```text
- 覆盖 embedding、layers、attention、KV update、MLP、norm 等 forward 主体；
- 要求 input buffers、attention metadata、BatchDescriptor、kernel routine 都稳定；
- 对 backend 支持要求最高；
- 通常 decode 场景最容易命中。
```

### 4.4 FULL_DECODE_ONLY

```text
CUDAGraphMode.FULL_DECODE_ONLY
```

含义：

```text
只有 decode / uniform decode 类 batch 尝试 FULL；
prefill / mixed batch 不走 FULL。
```

它适合 attention backend 只支持 decode full graph 的情况。

### 4.5 FULL_AND_PIECEWISE

```text
CUDAGraphMode.FULL_AND_PIECEWISE
```

含义：

```text
能 FULL 时用 FULL；
FULL 不适合时，如果 PIECEWISE 可用就用 PIECEWISE；
否则 NONE。
```

典型运行时分流：

```text
普通 decode / spec decode uniform batch
  → FULL

mixed prefill-decode
  → PIECEWISE

超出 capture size / key miss / feature 不兼容
  → NONE
```

这是最能体现 vLLM 混合执行模式的配置。

---

## 5. compile_sizes、capture_sizes、max capture size 的关系

### 5.1 `cudagraph_capture_sizes`

`cudagraph_capture_sizes` 是 CUDA graph 预先捕获的 token size 档位。

位置：`code/vllm/vllm/config/compilation.py:629`

例如：

```text
[1, 2, 4, 8, 16, 32]
```

表示初始化时会为这些 shape 准备 graph key。

运行时如果真实 token 数落在两个档位之间，会 pad 到下一个档位。

### 5.2 `max_cudagraph_capture_size`

位置：`code/vllm/vllm/config/compilation.py:673`

它是最大的 capture size。

如果运行时：

```text
num_tokens > max_cudagraph_capture_size
```

`CudagraphDispatcher.dispatch()` 会直接返回 `NONE`。

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:274`

### 5.3 `compile_sizes`

`compile_sizes` 是 Inductor / piecewise compile 使用的 static shape 集合。

位置：`code/vllm/vllm/config/compilation.py:556`

它和 cudagraph capture sizes 关系密切，因为：

```text
如果某个 compiled subgraph 是按 size=A 编译的，
runtime 却被 cudagraph padding 到 size=B，
那么 compiled shape 和 graph replay shape 就不一致。
```

因此 `CudagraphDispatcher._compute_bs_to_padded_graph_size()` 会检查：

```text
compile_sizes 不能落在会被 padding 改写的位置。
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:72`

如果不满足，会报错，而不是运行时悄悄 fallback。

### 5.4 `post_init_cudagraph_sizes()` 做什么

入口：`code/vllm/vllm/config/compilation.py:1070`

它会：

```text
1. 展开 compile_sizes 中的 "cudagraph_capture_sizes"；
2. 去重 compile_sizes；
3. 排序 cudagraph_capture_sizes；
4. 保证 cudagraph_capture_sizes 最大值等于 max_cudagraph_capture_size。
```

这一步让 compile sizes 和 capture sizes 在配置归一阶段就对齐。

---

## 6. 配置阶段的自动降级

用户配置的 mode 不是最终结果。`VllmConfig` 和 `CompilationConfig` 会根据模型、环境变量、并行配置、backend 能力自动修正。

### 6.1 enforce_eager

如果：

```text
model_config.enforce_eager=True
```

则会禁用 compile 和 cudagraph。

相关位置：`code/vllm/vllm/config/vllm.py:1071`

语义：

```text
用户显式要求 eager，就不能再偷偷 replay graph。
```

### 6.2 TORCH_COMPILE_DISABLE

如果环境变量禁用 torch compile，vLLM 会禁用相关 compile 路径。

相关位置：`code/vllm/vllm/config/vllm.py:1079`

这会影响 PIECEWISE，因为 piecewise graph 通常依赖 vLLM compile pipeline。

### 6.3 mode 不支持 piecewise

如果 cudagraph mode 依赖 piecewise，但：

```text
compilation_config.mode != VLLM_COMPILE
```

可能会降级为 `NONE`。

相关位置：`code/vllm/vllm/config/vllm.py:1182`

原因：

```text
没有 vLLM compile split，就没有 piecewise graph boundary。
```

### 6.4 SP / async TP 影响 piecewise

在某些 pass config 下，如果启用 sequence parallelism 或 GEMM + communication fusion，且不使用 inductor graph partition，vLLM 会清空 splitting ops，并把依赖 piecewise 的 mode 改成 FULL。

相关位置：`code/vllm/vllm/config/compilation.py:1167`

### 6.5 DeepEP high-throughput + DP

如果：

```text
all2all_backend = deepep_high_throughput
data_parallel_size > 1
cudagraph_mode != NONE
```

会直接禁用 CUDA graph。

相关位置：`code/vllm/vllm/config/compilation.py:1186`

原因是该 All2All backend 与 CUDA Graph 兼容性有限。

### 6.6 pooling model

pooling model 不支持 FULL cudagraph，会被降级到 PIECEWISE。

相关位置：`code/vllm/vllm/config/vllm.py:1243`

原因是 pooling 输出形态和请求级 pooling metadata 更动态。

### 6.7 encoder-decoder

encoder-decoder 模型只允许更保守的 cudagraph 模式，例如 `NONE` 或 `FULL_DECODE_ONLY`。

相关位置：`code/vllm/vllm/config/vllm.py:1255`

原因：

```text
encoder input / encoder output / cross attention metadata 更动态。
```

### 6.8 KV connector

某些 KV connector 与 FULL graph 不兼容，会把 FULL 降为 PIECEWISE。

相关位置：`code/vllm/vllm/config/vllm.py:1269`

---

## 7. attention backend 如何修正 cudagraph mode

FULL graph 是否可用，很大程度取决于 attention backend。

attention backend 通过 `AttentionCGSupport` 声明能力。

定义位置：`code/vllm/vllm/v1/attention/backend.py:516`

```text
ALWAYS：
  支持 mixed prefill-decode 和 uniform decode 的 full graph。

UNIFORM_BATCH：
  只支持所有 request query length 一致的 batch。

UNIFORM_SINGLE_TOKEN_DECODE：
  只支持普通单 token decode。

NEVER：
  不支持 full cudagraph。
```

`GPUModelRunner.initialize_attn_backend()` 会在 attention backend 初始化后调用 `_check_and_update_cudagraph_mode()`。

相关位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6831`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6877`

然后 `CompilationConfig.resolve_cudagraph_mode_and_sizes()` 会根据 backend 能力修正 mode。

入口：`code/vllm/vllm/config/compilation.py:1316`

典型降级包括：

```text
FULL 不支持 mixed batch：
  → FULL_AND_PIECEWISE 或 FULL_DECODE_ONLY

decode FULL 也不支持：
  → PIECEWISE 或 NONE

spec decode query length > 1 但 backend 不支持 uniform batch：
  → PIECEWISE 或 NONE
```

所以调试时要注意：

```text
命令行里配置的 cudagraph_mode 不是最终 runtime mode。
```

最终要看 attention backend 初始化后的 `compilation_config.cudagraph_mode`。

---

## 8. CudagraphDispatcher：合法 key 和运行时 dispatch

`CudagraphDispatcher` 定义在：`code/vllm/vllm/v1/cudagraph_dispatcher.py:15`

它维护两组 key：

```text
cudagraph_keys[PIECEWISE]
cudagraph_keys[FULL]
```

这些 key 是 runtime 能 dispatch 到 graph 的合法集合。

### 8.1 初始化合法 key

入口：`code/vllm/vllm/v1/cudagraph_dispatcher.py:166`

```text
initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)
```

它会：

```text
1. 如果 cudagraph_mode == NONE，不创建 key；
2. 预计算真实 token size → padded capture size 映射；
3. 根据 LoRA 配置生成 capture cases；
4. 为 mixed mode 创建 PIECEWISE / FULL key；
5. 为 uniform decode 创建 FULL key；
6. 标记 keys_initialized=True。
```

### 8.2 运行时 dispatch

入口：`code/vllm/vllm/v1/cudagraph_dispatcher.py:235`

输入包括：

```text
num_tokens
num_reqs
uniform_decode
has_lora
num_active_loras
valid_modes
invalid_modes
```

返回：

```text
cudagraph_mode: CUDAGraphMode
batch_descriptor: BatchDescriptor
```

逻辑可以压缩为：

```text
if keys not initialized or mode NONE or num_tokens too large:
  return NONE, BatchDescriptor(num_tokens)

batch_desc = _create_padded_batch_descriptor(...)

if FULL allowed and FULL key exists:
  return FULL, batch_desc

if PIECEWISE allowed and relaxed PIECEWISE key exists:
  return PIECEWISE, relaxed_desc

return NONE, BatchDescriptor(num_tokens)
```

### 8.3 valid_modes / invalid_modes

调用方可以传：

```text
valid_modes={NONE}
  强制 eager，例如 profile / warmup。

invalid_modes={FULL}
  禁用 full，但允许 piecewise / none，例如 cascade attention 或 encoder output。

valid_modes={synced_mode}
  DP 协调后强制所有 rank 使用同一种 mode。
```

这让 dispatch 不是纯配置查表，而是配置、backend 和本轮 runtime feature 的交集。

---

## 9. GPUModelRunner 每轮如何决定 runtime mode

主入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`

```text
_determine_batch_execution_and_padding()
```

它接收本轮执行信息：

```text
num_tokens
num_reqs
max_num_scheduled_tokens
num_encoder_reqs
use_cascade_attn
force_eager
force_uniform_decode
force_has_lora
```

核心步骤：

```text
1. 判断 uniform_decode。
2. 统计 active LoRA。
3. 判断 has_encoder_output。
4. num_tokens 先经过 _pad_for_sequence_parallelism()。
5. 调 CudagraphDispatcher.dispatch()。
6. 如果 use_cascade_attn 或 has_encoder_output，禁用 FULL。
7. 如果 force_eager，valid_modes={NONE}。
8. 如果 data_parallel_size > 1，调用 coordinate_batch_across_dp()。
9. DP 协调后可能用 synced mode 再次 dispatch。
10. 构造 cudagraph_stats。
```

返回：

```text
cudagraph_mode
batch_descriptor
should_ubatch
num_tokens_across_dp
cudagraph_stats
```

这些值随后影响：

```text
- num_tokens_padded；
- num_reqs_padded；
- slot_mapping padding；
- attention metadata padding；
- ubatch slices；
- set_forward_context()；
- CUDAGraphWrapper replay key。
```

---

## 10. ForwardContext 如何传递 runtime mode

`ForwardContext` 定义在：`code/vllm/vllm/forward_context.py:129`

`set_forward_context()` 定义在：`code/vllm/vllm/forward_context.py:250`

`GPUModelRunner.execute_model()` 在 forward 前调用：

`code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`

```text
set_forward_context(
  attn_metadata,
  vllm_config,
  num_tokens=num_tokens_padded,
  num_tokens_across_dp=num_tokens_across_dp,
  cudagraph_runtime_mode=cudagraph_mode,
  batch_descriptor=batch_desc,
  ubatch_slices=ubatch_slices_padded,
  slot_mapping=slot_mappings,
  skip_compiled=has_encoder_input,
)
```

这一步的意义是：

```text
ModelRunner 不把 cudagraph_mode 显式传给每层模型；
模型内部的 wrapper、attention layer、custom op 都从 ForwardContext 读取同一个运行态。
```

主要消费方：

```text
CUDAGraphWrapper：
  读取 cudagraph_runtime_mode / batch_descriptor。

@support_torch_compile 注入的 __call__：
  读取 skip_compiled。

Attention backend：
  读取 attn_metadata / slot_mapping。

MoE / DP：
  读取 dp_metadata。

UBatchWrapper：
  读取 ubatch_slices。
```

---

## 11. CUDAGraphWrapper 如何根据 runtime mode 执行

`CUDAGraphWrapper` 定义在：`code/vllm/vllm/compilation/cuda_graph.py:145`

运行入口：`code/vllm/vllm/compilation/cuda_graph.py:233`

核心逻辑：

```text
ctx = get_forward_context()
mode = ctx.cudagraph_runtime_mode
batch_descriptor = ctx.batch_descriptor

if mode == NONE:
  return runnable(...)

if mode != self.runtime_mode:
  return runnable(...)

entry = concrete_cudagraph_entries[batch_descriptor]

if entry.cudagraph is None:
  validate_cudagraph_capturing_enabled()
  with torch.cuda.graph(...):
      output = runnable(...)
  save output and cudagraph
  return output

entry.cudagraph.replay()
return entry.output
```

这说明：

```text
runtime mode 的最终执行不是在 _model_forward() 里 if/else；
而是在 wrapper.__call__() 里根据 ForwardContext 自动生效。
```

`_model_forward()` 本身只是：

```text
return self.model(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3757`

---

## 12. runtime mode 和 eager / compiled 的组合关系

### 12.1 cudagraph NONE + compiled callable

```text
CompilationMode = VLLM_COMPILE
cudagraph_runtime_mode = NONE
```

含义：

```text
不 replay CUDA graph，
但底层 model.__call__ 仍可能走 compiled callable。
```

这通常发生在：

```text
- batch 超过 capture size；
- BatchDescriptor key miss；
- runtime feature 禁用 graph；
- mixed batch 没有 piecewise key。
```

### 12.2 cudagraph FULL + eager runnable

```text
CompilationMode = NONE
cudagraph_runtime_mode = FULL
```

含义：

```text
不使用 torch.compile，
但整个 eager model forward 被 full CUDAGraphWrapper capture/replay。
```

这说明 full CUDA graph 和 compilation 可以正交。

### 12.3 PIECEWISE + vLLM compile

```text
CompilationMode = VLLM_COMPILE
cudagraph_runtime_mode = PIECEWISE
```

含义：

```text
只 replay vLLM compile split 出来的子图，
attention / dynamic ops 可能仍在 graph 外。
```

### 12.4 FULL_AND_PIECEWISE

配置：

```text
CompilationMode = VLLM_COMPILE
CUDAGraphMode = FULL_AND_PIECEWISE
```

runtime：

```text
uniform decode 命中 FULL key：
  外层 FULL wrapper replay；内层 PIECEWISE pass-through。

mixed prefill-decode 命中 PIECEWISE key：
  外层 FULL wrapper pass-through；内层 PIECEWISE replay。

都不命中：
  两层 wrapper 都 pass-through。
```

---

## 13. 几个典型 runtime 场景

### 13.1 普通 decode

```text
num_tokens = num_reqs
max_num_scheduled_tokens = 1
uniform_decode = True
```

如果 FULL key 存在：

```text
runtime_mode = FULL
```

否则可能：

```text
PIECEWISE 或 NONE
```

这是 CUDA graph 最容易发挥收益的场景。

### 13.2 speculative decode 验证步

```text
uniform_decode_query_len = 1 + num_speculative_tokens
num_tokens = num_reqs * uniform_decode_query_len
uniform_decode = True
```

如果 attention backend 支持 uniform batch full graph，可能走 FULL。

否则退到：

```text
PIECEWISE / NONE
```

capture sizes 还需要满足 spec query length 倍数约束。

### 13.3 mixed prefill-decode

```text
部分请求 prefill 多 token
部分请求 decode 1 token
uniform_decode = False
```

通常难以命中 decode FULL。

在 `FULL_AND_PIECEWISE` 下常见结果：

```text
runtime_mode = PIECEWISE
```

如果 piecewise 不可用，则：

```text
runtime_mode = NONE
```

### 13.4 cascade attention

如果本轮使用 cascade attention：

```text
invalid_modes={FULL}
```

结果：

```text
PIECEWISE 可用 → PIECEWISE
否则 → NONE
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3865`

### 13.5 encoder-decoder 首轮

如果本轮有 encoder output：

```text
禁用 FULL
```

如果本轮有 encoder input：

```text
skip_compiled=True
```

相关位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3839`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4312`

原因是 encoder input / cross attention metadata 更动态。

### 13.6 active LoRA

LoRA 会进入 `BatchDescriptor`：

```text
has_lora
num_active_loras
```

如果 `cudagraph_specialize_lora=True`，会为多个 active LoRA count 捕获不同 key。

如果运行时 active count 没有合适 key，可能 fallback。

相关位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:111`

### 13.7 超出 capture size

如果：

```text
num_tokens > max_cudagraph_capture_size
```

直接：

```text
runtime_mode = NONE
```

相关位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:274`

---

## 14. runtime mode 统计如何观察

如果开启：

```text
observability_config.cudagraph_metrics=True
```

`_determine_batch_execution_and_padding()` 会生成：

```text
CUDAGraphStat(
  num_unpadded_tokens,
  num_padded_tokens,
  num_paddings,
  runtime_mode,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907`

这个统计最终进入 `ModelRunnerOutput.cudagraph_stats`，再被 logger 聚合。

它可以回答：

```text
- 本轮到底是 FULL / PIECEWISE / NONE？
- 真实 token 数是多少？
- padding 到多少？
- padding waste 有多大？
- 某个 workload 为什么 graph 命中率低？
```

---

## 15. 容易混淆的点

### 15.1 `CompilationMode.NONE` 是否等于没有 CUDA graph？

不一定。

```text
CompilationMode.NONE 只表示不 torch.compile；
FULL cudagraph 仍然可以包住 eager forward。
```

### 15.2 `CUDAGraphMode.NONE` 是否等于完全 eager？

不一定。

```text
CUDAGraphMode.NONE 只表示不 replay CUDA graph；
底层仍可能是 torch.compile / vLLM compile 后的 callable。
```

### 15.3 `FULL` 是否表示整个 Engine step 被 capture？

不是。

```text
FULL graph 指 full model forward，
不是 Scheduler + logits + sampler + output processor 的全链路。
```

### 15.4 配置了 FULL 为什么 runtime 不是 FULL？

可能原因：

```text
- attention backend 自动降级；
- 本轮不是 uniform decode；
- cascade attention 禁用 FULL；
- encoder output 禁用 FULL；
- BatchDescriptor key 不存在；
- 超出 capture size；
- DP rank 协调后降级。
```

### 15.5 capture sizes 是 request 数还是 token 数？

主要是 token size / batch shape 档位。

在 uniform decode 下：

```text
num_tokens = num_reqs * uniform_decode_query_len
```

所以看起来也和 request 数相关。

但 dispatch key 实际是 `BatchDescriptor`，其中既有 `num_tokens`，也可能有 `num_reqs` 和 `uniform`。

---

## 16. 最小伪代码

### 16.1 初始化配置

```text
config = CompilationConfig(...)
config.post_init_cudagraph_sizes()

if enforce_eager:
  config.mode = NONE
  config.cudagraph_mode = NONE

if piecewise_requested_but_no_vllm_compile:
  config.cudagraph_mode = NONE

attn_support = min(attention_backend.get_cudagraph_support())
config.resolve_cudagraph_mode_and_sizes(attn_support, ...)

dispatcher.initialize_cudagraph_keys(
  config.cudagraph_mode,
  uniform_decode_query_len,
)
```

### 16.2 每轮 dispatch

```text
num_tokens = scheduler_output.total_num_scheduled_tokens
num_reqs = input_batch.num_reqs
uniform_decode = _is_uniform_decode(...)

num_tokens = _pad_for_sequence_parallelism(num_tokens)

valid_modes = {NONE} if force_eager else None
invalid_modes = {FULL} if use_cascade_attn or has_encoder_output else None

cudagraph_mode, batch_desc = dispatcher.dispatch(
  num_tokens,
  num_reqs,
  uniform_decode,
  has_lora,
  num_active_loras,
  valid_modes,
  invalid_modes,
)

if data_parallel_size > 1:
  synced_mode, num_tokens_across_dp = coordinate_batch_across_dp(...)
  cudagraph_mode, batch_desc = dispatcher.dispatch(
    synced_padded_tokens,
    valid_modes={synced_mode},
  )
```

### 16.3 forward context 和 wrapper

```text
with set_forward_context(
  attn_metadata,
  cudagraph_runtime_mode=cudagraph_mode,
  batch_descriptor=batch_desc,
  slot_mapping=slot_mappings,
  skip_compiled=has_encoder_input,
):
  hidden_states = self.model(...)
```

```text
# CUDAGraphWrapper.__call__
ctx = get_forward_context()

if ctx.mode == NONE or ctx.mode != self.runtime_mode:
  return runnable(...)

entry = graphs[ctx.batch_descriptor]

if entry.cudagraph is None:
  capture runnable
else:
  entry.cudagraph.replay()

return entry.output
```

---

## 17. 一句话总结

`CompilationConfig` 决定 vLLM 的优化上限：是否 compile、是否 cudagraph、capture 哪些 shape、哪些 LoRA / spec / SP / backend 组合合法；运行时 `GPUModelRunner` 再根据本轮 batch、attention backend、LoRA、encoder、cascade、DP 协调等因素，把真实执行归一成 `CUDAGraphMode + BatchDescriptor`，通过 `ForwardContext` 交给 wrapper。

最核心的结论是：

```text
compile mode 决定底层 runnable，cudagraph runtime mode 决定是否 replay；二者相关但不是同一个开关。
```
