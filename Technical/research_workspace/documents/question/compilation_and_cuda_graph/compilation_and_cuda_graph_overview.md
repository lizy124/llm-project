# vLLM V1 Compilation / CUDA Graph 逻辑梳理

源码位置：

- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/config/observability.py`
- `code/vllm/vllm/compilation/`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/v1/cudagraph_dispatcher.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py`
- `code/vllm/vllm/v1/worker/dp_utils.py`
- `code/vllm/vllm/v1/worker/ubatch_utils.py`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `code/vllm/vllm/v1/attention/backend.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/layers/`
- `code/vllm/docs/design/torch_compile.md`
- `code/vllm/docs/design/cuda_graphs.md`

本文按“先定边界，再走初始化 capture，再走 runtime dispatch，最后拆 padding、attention metadata、fallback 和调试”的方式，梳理 vLLM V1 中 `torch.compile`、vLLM compile、CUDA graph capture / replay 如何嵌入 `GPUModelRunner.execute_model()` 主链路。

它和 `executor_worker_model_runner` 的执行链路关系很密切。可以先把它理解成执行层里的性能优化支线：

```text
SchedulerOutput
  → ModelRunner 准备输入
  → 判断本轮是否可用 CUDA graph
  → 必要时 padding 到固定 batch shape
  → 构造 capture / replay 兼容的 attention metadata
  → set_forward_context(...)
  → model forward
      → eager / compiled / cudagraph replay
  → logits / sampler / output
```

本文不把 CUDA graph 讲成一个孤立模块，而是围绕一个核心问题展开：

```text
动态 serving batch 如何被整理成可编译、可 capture、可 replay 的稳定执行形态？
```

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的写法，本目录按“角色定位 → 配置 → 初始化 → runtime → 边界和调试”的顺序组织。

要回答的问题分成 11 组：

```text
1. Compilation 和 CUDA graph 在 vLLM 中分别解决什么问题？
2. CompileConfig、cudagraph 配置和 runtime mode 如何决定执行路径？
3. Worker warmup / compile / CUDA graph capture 生命周期是什么？
4. 动态 batch 如何通过 padding / shape bucket 变成可 replay 形态？
5. ModelRunner 每轮如何选择 eager、compiled、cudagraph replay？
6. Attention metadata 在 capture 和 replay 时有什么特殊路径？
7. model forward 如何被 compile wrapper / graph runner 包装？
8. sampler / output 是否也被 capture，和 forward graph 的边界在哪里？
9. TP / PP / DP / DBO / MoE 等并行如何影响 compilation / cudagraph？
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

如果只想先抓住主线，可以先读总览，再读 `01`、`02`、`03`、`04`、`05`。

---

## 1. 一句话回答

Compilation / CUDA graph 都是 vLLM 执行层的性能优化机制，但优化对象不同。

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
LLM serving 的 batch 是动态的，
但 CUDA graph replay 需要固定 shape、固定地址、固定 kernel launch 形态。
```

因此 vLLM 需要把：

```text
动态请求 / 动态 token / 动态 attention metadata
```

整理成：

```text
BatchDescriptor + padded buffers + capture-compatible metadata + ForwardContext
```

再交给 compiled forward 或 CUDA graph wrapper 执行。

一句话记忆：

```text
compile 优化 forward 图本身，CUDA graph 复用固定 shape 的 GPU launch 序列；二者都不改变调度和模型语义，只改变执行路径。
```

---

## 2. 总体流程图

### 2.1 初始化阶段

初始化阶段的目标是把模型、KV cache、compile graph、CUDA graph 都准备好。

```text
GPUWorker.init_device()
  → 初始化 distributed / device / random seed
  → 创建 GPUModelRunner

GPUWorker.load_model()
  → GPUModelRunner.load_model()
  → 加载模型权重
  → 根据 CompilationConfig 包装 model / compile wrapper / full graph wrapper

GPUWorker.initialize_from_config()
  → allocate KV cache
  → compile_or_warm_up_model()
      → profile_run / dummy run
      → 触发 torch.compile / vLLM compile
      → 初始化 attention backend workspace
      → capture_model()
          → CudagraphDispatcher.get_capture_descs()
          → _capture_cudagraphs()
          → _warmup_and_capture()
          → _dummy_run(..., mode=NONE) warmup
          → _dummy_run(..., mode=FULL/PIECEWISE, is_graph_capturing=True)
          → CUDAGraphWrapper capture
```

关键源码位置：

- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5142`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6227`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6584`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6651`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6686`

### 2.2 运行阶段

真实请求执行时，CUDA graph dispatch 发生在 forward 前。

```text
SchedulerOutput
  → GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
      → input_ids / positions / logits_indices / query_start_loc
  → _compute_cascade_attn_prefix_lens()
  → _determine_batch_execution_and_padding()
      → 判断 uniform_decode
      → 处理 LoRA / encoder output / cascade attention 限制
      → sequence parallel padding
      → CudagraphDispatcher.dispatch()
      → DP rank 协调
      → 得到 cudagraph_mode + BatchDescriptor
  → _get_slot_mappings()
      → padding token slot 填 -1
  → _build_attention_metadata()
      → FULL 时按 padded shape 构造 metadata
  → _preprocess()
      → input_ids / inputs_embeds / model_kwargs
  → set_forward_context(...)
      → cudagraph_runtime_mode
      → batch_descriptor
      → attn_metadata
      → slot_mapping
      → ubatch_slices
      → skip_compiled
  → _model_forward()
      → self.model(...)
      → CUDAGraphWrapper / compiled wrapper / attention backend 读取 ForwardContext
      → eager / compiled / cudagraph replay
  → hidden_states / IntermediateTensors
  → compute_logits / pooling
  → sample_tokens / output
```

关键源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3960`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4196`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`
- `code/vllm/vllm/v1/cudagraph_dispatcher.py:235`
- `code/vllm/vllm/compilation/cuda_graph.py:233`

---

## 3. 核心概念

### 3.1 `CompilationMode`

定义位置：`code/vllm/vllm/config/compilation.py:37`

它描述 torch compile / vLLM compile 层怎么工作：

```text
NONE：
  不使用 torch.compile。

STOCK_TORCH_COMPILE：
  使用 PyTorch 原生 nn.Module.compile / torch.compile 路径。

DYNAMO_TRACE_ONCE：
  做一次 Dynamo trace，避免后续反复 recompile。

VLLM_COMPILE：
  使用 vLLM 自定义 backend、piecewise compilation、custom pass、compile cache。
```

### 3.2 `CUDAGraphMode`

定义位置：`code/vllm/vllm/config/compilation.py:53`

它描述 CUDA graph 的 capture / replay 模式：

```text
NONE：
  不使用 CUDA graph。

PIECEWISE：
  只 capture / replay compiled 后的 piecewise 子图。

FULL：
  capture / replay 整个 model forward。

FULL_DECODE_ONLY：
  decode batch 尽量走 FULL，非 decode 场景不走 FULL。

FULL_AND_PIECEWISE：
  decode 能 FULL 就 FULL，mixed / prefill 尽量 PIECEWISE，否则 NONE。
```

实际运行时每轮最终会落到：

```text
FULL / PIECEWISE / NONE
```

配置里的组合模式会在 dispatch 时拆成具体 runtime mode。

### 3.3 `CompilationConfig`

定义位置：`code/vllm/vllm/config/compilation.py:379`

它控制：

```text
- compile mode；
- compile backend；
- compile sizes / ranges；
- splitting ops；
- cudagraph mode；
- cudagraph capture sizes；
- max cudagraph capture size；
- cudagraph warmup runs；
- cudagraph copy inputs；
- LoRA cudagraph specialization；
- multimodal encoder compile / cudagraph；
- debug dump / compile cache。
```

可以记成：

```text
CompilationConfig 是启动时“能不能 compile / 能不能 graph / capture 哪些 shape”的配置入口。
```

### 3.4 `BatchDescriptor`

定义位置：`code/vllm/vllm/forward_context.py:30`

它是 CUDA graph dispatch 的 key：

```text
num_tokens
num_reqs
uniform
has_lora
num_active_loras
```

含义：

```text
num_tokens：通常是 padded 后 token 数；
num_reqs：FULL graph 下 attention metadata 可能依赖请求数；
uniform：是否 uniform decode；
has_lora / num_active_loras：LoRA graph key 维度。
```

`CUDAGraphWrapper` 不是按“当前请求 id”查 graph，而是按 `BatchDescriptor` 查已经 capture 的 graph。

### 3.5 `ForwardContext`

定义位置：`code/vllm/vllm/forward_context.py:129`

它是 runtime 控制总线，保存：

```text
attn_metadata
slot_mapping
dp_metadata
cudagraph_runtime_mode
batch_descriptor
ubatch_slices
skip_compiled
```

`GPUModelRunner.execute_model()` 在 forward 前调用：

```text
set_forward_context(...)
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`

然后：

```text
CUDAGraphWrapper：读取 cudagraph_runtime_mode / batch_descriptor；
compile wrapper：读取 skip_compiled；
attention layer：读取 attn_metadata / slot_mapping；
MoE / DP / ubatch：读取 dp_metadata / ubatch_slices。
```

---

## 4. Compilation 和 CUDA graph 的分层关系

这里最容易混淆：

```text
CUDAGraphMode.NONE 不等于没有 torch.compile。
```

要分两层看。

### 4.1 第一层：底层 runnable 是什么

底层 runnable 可能是：

```text
- 原始 eager model.forward；
- PyTorch stock torch.compile 后 callable；
- vLLM VllmBackend 编译后的 runtime callable；
- PiecewiseBackend 子图；
- 带 static buffer copy 的 compiled callable。
```

这由 `CompilationMode` 和 model wrapper 决定。

### 4.2 第二层：这一轮是否 replay CUDA graph

runtime mode 可能是：

```text
FULL：
  外层 full CUDAGraphWrapper 生效。

PIECEWISE：
  内层 piecewise CUDAGraphWrapper 生效。

NONE：
  所有 cudagraph wrapper pass-through。
```

这由 `CudagraphDispatcher.dispatch()` 和 `ForwardContext` 决定。

所以可能出现：

```text
compiled callable + cudagraph_runtime_mode=NONE：
  不 replay CUDA graph，但仍然是 compiled forward。

eager runnable + cudagraph_runtime_mode=FULL：
  不 torch.compile，但整段 eager forward 被 full CUDA graph replay。

VLLM_COMPILE + FULL_AND_PIECEWISE：
  decode 走 full graph，mixed batch 走 piecewise graph，不满足条件走 compiled/eager fallback。
```

---

## 5. Runtime dispatch 的主决策点

运行时选择 eager / compiled / cudagraph replay 的核心入口是：

`code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`

```text
GPUModelRunner._determine_batch_execution_and_padding()
```

它负责：

```text
1. 判断本轮是否 uniform decode；
2. 判断是否有 encoder output、cascade attention、LoRA 等限制；
3. 先做 sequence parallelism padding；
4. 调 CudagraphDispatcher.dispatch()；
5. DP 场景下跨 rank 同步 mode 和 padding；
6. 返回 cudagraph_mode、BatchDescriptor、should_ubatch、num_tokens_across_dp、cudagraph_stats。
```

`CudagraphDispatcher.dispatch()` 的入口是：

`code/vllm/vllm/v1/cudagraph_dispatcher.py:235`

它的选择顺序是：

```text
1. 如果配置或输入不允许 graph，返回 NONE；
2. 把真实 num_tokens pad 到 capture size；
3. 尝试命中 FULL key；
4. 尝试命中 PIECEWISE key；
5. 都不命中则返回 NONE。
```

可以记成：

```text
FULL 优先，PIECEWISE 兜底，NONE 保证正确性。
```

---

## 6. CUDA graph 为什么需要 padding

CUDA graph replay 要求固定 shape。真实 serving batch 每轮都可能变化：

```text
- request 数不同；
- num scheduled tokens 不同；
- prefill / decode / mixed batch 不同；
- spec decode query length 不同；
- LoRA active count 不同；
- DP rank 上本地 token 数不同；
- attention metadata shape 不同。
```

vLLM 的做法是：

```text
真实 num_tokens
  → _pad_for_sequence_parallelism()
  → CudagraphDispatcher._create_padded_batch_descriptor()
  → pad 到某个 cudagraph_capture_size
  → 后续 input buffers / slot_mapping / attention metadata 按 padded shape 准备
```

关键源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3407`
- `code/vllm/vllm/v1/cudagraph_dispatcher.py:72`
- `code/vllm/vllm/v1/cudagraph_dispatcher.py:132`

padding 不是简单多算几个 token，它必须保证：

```text
- padding token 不写入真实 KV cache；
- padding request 不读到真实 block；
- padding 位置不参与 logits / sampling；
- attention metadata 的 shape 固定但语义安全；
- output 只切回真实 token / request。
```

最关键的保护是：

```text
padding token 的 slot_mapping = -1；
padding request 的 block_table = NULL_BLOCK_ID。
```

---

## 7. Attention metadata 为什么是 CUDA graph 的核心约束

FULL CUDA graph 包含 attention，因此不只是 `input_ids` shape 固定，attention metadata 也必须能固定。

`_build_attention_metadata()` 入口：

`code/vllm/vllm/v1/worker/gpu_model_runner.py:2208`

它构造的 `CommonAttentionMetadata` 会包含：

```text
query_start_loc
seq_lens
num_reqs
num_actual_tokens
max_query_len
max_seq_len
block_table_tensor
slot_mapping
positions
encoder_seq_lens
dcp_local_seq_lens
logits_indices_padded
```

FULL graph 下：

```text
pad_attn = cudagraph_mode == CUDAGraphMode.FULL
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4196`

如果 `pad_attn=True`，后续会按 padded token / request 形态构造：

```text
slot_mapping
query_start_loc
seq_lens
block_table
positions
ubatch slices
attention backend metadata
```

attention backend 还通过 `AttentionCGSupport` 声明自己支持什么程度的 full graph：

```text
ALWAYS
UNIFORM_BATCH
UNIFORM_SINGLE_TOKEN_DECODE
NEVER
```

定义位置：`code/vllm/vllm/v1/attention/backend.py:516`

如果 backend 能力不足，配置阶段或 attention 初始化阶段会自动降级：

```text
FULL → FULL_AND_PIECEWISE / FULL_DECODE_ONLY / PIECEWISE / NONE
```

---

## 8. capture 生命周期

CUDA graph 通常不是第一轮真实请求临时 capture，而是在 Worker 初始化 / warmup 阶段提前 capture 一组固定 shape。

主入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6584`

```text
GPUModelRunner.capture_model()
  → 如果 cudagraph_mode == NONE，跳过
  → set_cudagraph_capturing_enabled(True)
  → for runtime_mode, batch_descs in cudagraph_dispatcher.get_capture_descs():
        _capture_cudagraphs(batch_descs, runtime_mode)
  → set_cudagraph_capturing_enabled(False)
```

`get_capture_descs()` 返回顺序：

```text
PIECEWISE first
FULL second
```

每组内部按大 shape 优先 capture。

目的：

```text
先 capture 大图，让小图尽量复用 CUDA graph memory pool，降低额外显存占用。
```

每个 descriptor 的 capture 又分两步：

```text
_warmup_and_capture(desc, runtime_mode)
  → 多次 _dummy_run(..., cudagraph_runtime_mode=NONE)
  → 一次 _dummy_run(..., cudagraph_runtime_mode=runtime_mode, is_graph_capturing=True)
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6651`

warmup 用 `NONE`，是为了预热 kernel、workspace、attention backend；真正 capture 时才让 wrapper 根据 `ForwardContext` 进入 CUDA graph capture。

---

## 9. CUDAGraphWrapper 的边界

`CUDAGraphWrapper` 定义在：

`code/vllm/vllm/compilation/cuda_graph.py:145`

它不负责准备 input buffers，也不负责判断本轮 batch 是否应该 graph。

源码注释说明：

```text
CUDAGraphWrapper does not store persistent buffers or copy runtime inputs.
We assume implementing them is done outside of the wrapper.
```

因此边界是：

```text
ModelRunner / dispatcher：
  决定本轮 runtime mode 和 padded BatchDescriptor；
  准备持久 input buffer、slot_mapping、attention metadata。

CUDAGraphWrapper：
  从 ForwardContext 读取 runtime mode 和 BatchDescriptor；
  mode 不匹配就 pass-through；
  key 未 capture 时 capture；
  key 已 capture 时 replay。
```

`__call__()` 的核心逻辑在：`code/vllm/vllm/compilation/cuda_graph.py:233`

```text
ctx = get_forward_context()
mode = ctx.cudagraph_runtime_mode
batch_descriptor = ctx.batch_descriptor

if mode == NONE or mode != self.runtime_mode:
  return runnable(...)

entry = graphs[batch_descriptor]
if entry.cudagraph is None:
  validate_cudagraph_capturing_enabled()
  capture runnable into torch.cuda.graph
else:
  entry.cudagraph.replay()
return entry.output
```

---

## 10. 和 model forward / logits / sampler 的边界

CUDA graph 的主要边界是：

```text
with set_forward_context(...):
    hidden_states = self._model_forward(...)
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3757`

`_model_forward()` 本身很薄：

```text
return self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)
```

CUDA graph / compile wrapper 主要包住的是：

```text
model backbone forward
  → attention / KV update / layers / MLP / norm
  → hidden_states 或 IntermediateTensors
```

通常不包含：

```text
hidden_states[logits_indices]
compute_logits()
grammar bitmask
Sampler / RejectionSampler
logprobs / prompt_logprobs
ModelRunnerOutput
Scheduler.update_from_output()
OutputProcessor / detokenize / RequestOutput
```

因此：

```text
FULL graph ≈ full model forward，
不是 full Engine step。
```

---

## 11. fallback 是正常路径

CUDA graph 是优化路径，不是正确性路径。

只要本轮不满足固定 shape、稳定地址、稳定 metadata、稳定控制流或 backend 支持条件，vLLM 就会 fallback。

常见 fallback 层级是：

```text
FULL cudagraph
  → PIECEWISE cudagraph
  → NONE / eager or compiled pass-through
```

常见原因包括：

```text
- enforce_eager=True；
- cudagraph_mode 被配置或自动降级为 NONE；
- attention backend 不支持目标 full graph；
- num_tokens 超过 max_cudagraph_capture_size；
- BatchDescriptor key 未 capture；
- 本轮不是 uniform decode；
- cascade attention 禁用 FULL；
- encoder-decoder 首轮有 encoder input / output；
- KV scale 计算；
- LoRA active count 没有对应 graph key；
- DP rank 协调后降级；
- multimodal encoder budget 不命中；
- certain quantization / MoE / DeepEP / tracing 路径不兼容。
```

fallback 行为应该保证：

```text
输出语义一致；
KV cache 写入一致；
attention metadata 正确；
sampler 不感知前面是否 graph replay；
collective 顺序不死锁。
```

---

## 12. 并行机制为什么会影响 CUDA graph

并行场景下，CUDA graph 不是单卡固定 shape replay。

它还要求：

```text
- 每个 rank 的本地 tensor shape 可稳定描述；
- 所有参与通信的 rank 走相同 collective 顺序；
- DP rank 对 cudagraph mode / padding / ubatch 达成一致；
- TP / SP 对 token 数有对齐要求；
- PP 每个 stage 捕获自己的本地 forward；
- MoE / EP 的 All2All backend 支持对应 capture；
- DBO / microbatching 的 ubatch slices 和 context 稳定。
```

典型影响：

```text
TP / SP：
  num_tokens 可能要 pad 到 TP size 倍数。

DP：
  本地先 dispatch，再 all_reduce 同步，mode 取最保守值。

PP：
  不同 PP rank 输入输出不同，每个 rank capture 自己的 stage。

DBO / ubatching：
  一个 batch 被拆成多个 ubatch，每个 ubatch 有自己的 ForwardContext。

MoE / EP：
  token routing 和 All2All 通信形态影响 graph 兼容性。
```

因此并行场景下的核心问题不是“某个 rank 能不能 graph”，而是：

```text
所有相关 rank 能不能用一致的 shape、metadata 和通信图 replay。
```

---

## 13. 调试思路

调试 compile / CUDA graph 问题时，不要只看“有没有打开 cudagraph”，而要逐层确认：

```text
1. 最终 CompilationConfig 是什么？
2. attention backend 是否自动降级 cudagraph_mode？
3. CudagraphDispatcher 初始化了哪些 FULL / PIECEWISE keys？
4. capture_model() 是否真的 capture，耗时和显存是多少？
5. execute_model() 每轮 dispatch 到 FULL / PIECEWISE / NONE 哪个？
6. BatchDescriptor 是什么，padding 了多少？
7. ForwardContext 传了什么 cudagraph_runtime_mode 和 batch_descriptor？
8. CUDAGraphWrapper 是 replay、capture 还是 pass-through？
9. 输出异常时，padding token、slot_mapping、block_table、attention metadata 是否安全？
```

最有用的运行时指标是 `cudagraph_metrics`。

它记录：

```text
num_unpadded_tokens
num_padded_tokens
num_paddings
runtime_mode
```

生成位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907`

这些指标可以回答：

```text
为什么 runtime_mode 经常是 NONE？
为什么 FULL 很少？
为什么 graph 命中但 padding waste 很大？
为什么 capture 了 graph 但 workload 没收益？
```

---

## 14. 关键对象关系

```text
CompilationConfig
  启动期配置：compile mode、cudagraph mode、capture sizes、compile sizes、warmup runs。

CompilationMode
  决定底层 runnable 是 eager、stock torch.compile、Dynamo trace once 还是 vLLM compile。

CUDAGraphMode
  决定 CUDA graph 运行模式：NONE / PIECEWISE / FULL / 组合模式。

CudagraphDispatcher
  根据 capture sizes、LoRA cases、uniform decode 等生成合法 BatchDescriptor key，运行时 dispatch。

BatchDescriptor
  CUDA graph capture/replay key，描述 padded token、request、uniform decode、LoRA case。

ForwardContext
  runtime 控制总线，把 cudagraph mode、BatchDescriptor、attention metadata、slot mapping、DP metadata 传给模型内部。

CUDAGraphWrapper
  读取 ForwardContext，执行 pass-through / capture / replay。

GPUModelRunner
  负责准备 persistent buffers、padding、slot mapping、attention metadata、preprocess、set_forward_context 和 forward 调用。

AttentionMetadataBuilder
  把 CommonAttentionMetadata 转成 backend-specific metadata，并声明 full cudagraph 支持能力。

ModelRunnerOutput.cudagraph_stats
  把本轮 graph dispatch 统计带回 scheduler / metrics。
```

---

## 15. 和其他专题的关系

```text
executor_worker_model_runner：
  解释 ModelRunner 主执行链路，本专题细化其中 compile / cudagraph 分支。

attention：
  attention metadata builder 需要支持 cudagraph capture / replay。

scheduler：
  SchedulerOutput 决定本轮 num_tokens / num_reqs / prefill-decode 形态。

parallelism：
  TP / PP / DP / DBO / MoE 下 graph capture 要考虑 rank、collective 和 intermediate tensors。

multimodal：
  多模态 encoder 有单独的 compile / cudagraph 配置和 fallback 逻辑。

sampling_and_output：
  sampler 通常在 forward graph 边界之外，但 logits shape、async output 和 cudagraph_stats 会受影响。

quantization / lora：
  某些量化 kernel、动态 LoRA mapping 可能限制 cudagraph 或影响 BatchDescriptor key。
```

---

## 16. 推荐阅读路线

### 16.1 快速建立全局印象

```text
compilation_and_cuda_graph_overview.md
  → 01_compilation_cuda_graph_role.md
  → 02_compile_config_and_runtime_modes.md
```

### 16.2 按初始化和 capture 阅读

```text
compilation_and_cuda_graph_overview.md
  → 03_warmup_and_capture_lifecycle.md
  → 07_model_forward_compile_wrapper.md
```

### 16.3 按 runtime dispatch 阅读

```text
compilation_and_cuda_graph_overview.md
  → 04_batch_padding_and_shape_stability.md
  → 05_cudagraph_dispatch_flow.md
  → 06_attention_metadata_capture.md
```

### 16.4 按边界和 fallback 阅读

```text
compilation_and_cuda_graph_overview.md
  → 08_sampler_and_output_interaction.md
  → 09_parallelism_and_cudagraph.md
  → 10_limitations_and_fallbacks.md
  → 11_debugging_and_metrics.md
```

---

## 17. 文档定位

```text
compilation_and_cuda_graph_overview.md：
  总览主文档，适合快速建立 compile / cudagraph 全局图。

01-11：
  按问题拆开的专题文档，适合逐段精读职责、配置、capture、padding、dispatch、attention metadata、wrapper、sampler 边界、parallel、fallback 和调试。
```

---

## 18. 一句话总结

Compilation / CUDA graph 是 vLLM 在执行层降低运行时开销的机制。

最核心的主线是：

```text
动态 SchedulerOutput
  → ModelRunner 准备输入
  → CudagraphDispatcher 归一成 BatchDescriptor
  → padding 到 captured shape
  → 构造 capture-compatible attention metadata
  → ForwardContext 写入 runtime mode
  → CUDAGraphWrapper / compiled wrapper 执行 replay 或 pass-through
  → hidden_states
  → graph 外 logits / sampler / output
```

再压缩成一句话：

```text
vLLM 通过 compile、fixed shape、padding、static buffers、attention metadata 约束和 graph replay，把动态 serving 尽量转成可复用的高效 GPU forward；不满足条件时安全回退到 PIECEWISE 或 eager，保证正确性优先。
```
