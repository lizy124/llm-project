# 15 compilation_and_cuda_graph 背诵文档

## 1. 专题定位

`compilation_and_cuda_graph` 讲的是 vLLM V1 如何在执行层用 torch.compile、vLLM compile 和 CUDA graph 降低 forward 开销。

它不是改变模型语义的模块。

一句话：

```text
Compilation / CUDA graph 是 vLLM 在执行层降低运行时开销的性能优化机制。
```

## 2. 最小心智模型

主链路：

```text
动态 SchedulerOutput
  → ModelRunner 准备输入
  → 判断本轮能否用 CUDA graph
  → padding 到固定 batch shape
  → 构造 capture-compatible attention metadata
  → set_forward_context(...)
  → model forward
      → eager / compiled / cudagraph replay
  → hidden_states
  → graph 外 logits / sampler / output
```

要背住：

```text
compile 优化 forward 图本身；CUDA graph 复用固定 shape 的 GPU launch 序列；二者不改变调度和模型语义，只改变执行路径。
```

## 3. 核心矛盾

LLM serving 的 batch 是动态的：

```text
request 数不同
token 数不同
prefill / decode / mixed batch 不同
spec decode query length 不同
LoRA active count 不同
DP rank 本地 token 数不同
attention metadata shape 不同
```

但 CUDA graph replay 需要：

```text
固定 shape
固定 tensor 地址
固定 kernel launch 序列
稳定控制流
```

所以 vLLM 要把动态 batch 整理成：

```text
BatchDescriptor + padded buffers + capture-compatible metadata + ForwardContext
```

## 4. compile 和 CUDA graph 的区别

### torch.compile / vLLM compile

优化对象：

```text
Python / Dynamo / Inductor / custom pass 路径
```

目标：

```text
把 forward 转成更稳定、更高效的 compiled graph 或 kernel 组合。
```

### CUDA graph

优化对象：

```text
固定 shape、固定地址、固定 kernel launch 序列
```

目标：

```text
运行时 replay，减少每步 decode / small batch 的 kernel launch overhead。
```

一句话：

```text
compile 优化“图怎么执行”；CUDA graph 优化“launch 序列如何复用”。
```

## 5. 初始化阶段总览

初始化阶段：

```text
GPUWorker.init_device()
  → 初始化 distributed / device / seed
  → 创建 GPUModelRunner

GPUWorker.load_model()
  → GPUModelRunner.load_model()
  → 加载模型权重
  → 根据 CompilationConfig 包装 model / compile wrapper / graph wrapper

GPUWorker.initialize_from_config()
  → allocate KV cache
  → compile_or_warm_up_model()
      → dummy run / profile run
      → 触发 torch.compile / vLLM compile
      → 初始化 attention backend workspace
      → capture_model()
```

capture_model：

```text
CudagraphDispatcher.get_capture_descs()
  → _capture_cudagraphs()
  → _warmup_and_capture()
  → _dummy_run(..., mode=NONE) warmup
  → _dummy_run(..., mode=FULL/PIECEWISE, is_graph_capturing=True)
  → CUDAGraphWrapper capture
```

## 6. 运行阶段总览

真实请求执行：

```text
SchedulerOutput
  → GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
  → _compute_cascade_attn_prefix_lens()
  → _determine_batch_execution_and_padding()
      → 判断 uniform_decode
      → 处理 LoRA / encoder / cascade 限制
      → sequence parallel padding
      → CudagraphDispatcher.dispatch()
      → DP rank 协调
      → 得到 cudagraph_mode + BatchDescriptor
  → _get_slot_mappings()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(...)
  → _model_forward()
  → eager / compiled / cudagraph replay
  → hidden_states / IntermediateTensors
  → compute_logits / pooling
  → sample_tokens / output
```

## 7. CompilationMode

`CompilationMode` 描述 compile 层怎么工作。

常见模式：

```text
NONE：
  不使用 torch.compile。

STOCK_TORCH_COMPILE：
  使用 PyTorch 原生 torch.compile / nn.Module.compile。

DYNAMO_TRACE_ONCE：
  做一次 Dynamo trace，避免后续反复 recompile。

VLLM_COMPILE：
  使用 vLLM 自定义 backend、piecewise compilation、custom pass、compile cache。
```

一句话：

```text
CompilationMode 决定底层 forward callable 是 eager 还是 compiled。
```

## 8. CUDAGraphMode

`CUDAGraphMode` 描述 CUDA graph capture / replay 模式。

```text
NONE：
  不使用 CUDA graph。

PIECEWISE：
  只 capture / replay compiled 后的 piecewise 子图。

FULL：
  capture / replay 整个 model forward。

FULL_DECODE_ONLY：
  decode batch 尽量走 FULL，非 decode 不走 FULL。

FULL_AND_PIECEWISE：
  decode 能 FULL 就 FULL，mixed / prefill 尽量 PIECEWISE，否则 NONE。
```

实际每轮 runtime mode 最终落到：

```text
FULL / PIECEWISE / NONE
```

## 9. CompilationConfig

`CompilationConfig` 是启动时配置入口。

它控制：

```text
compile mode
compile backend
compile sizes / ranges
splitting ops
cudagraph mode
cudagraph capture sizes
max cudagraph capture size
cudagraph warmup runs
cudagraph copy inputs
LoRA cudagraph specialization
multimodal encoder compile / cudagraph
debug dump / compile cache
```

一句话：

```text
CompilationConfig 决定能不能 compile、能不能 graph、capture 哪些 shape。
```

## 10. BatchDescriptor

`BatchDescriptor` 是 CUDA graph dispatch key。

包含：

```text
num_tokens
num_reqs
uniform
has_lora
num_active_loras
```

含义：

```text
num_tokens：通常是 padded 后 token 数。
num_reqs：FULL graph 下 attention metadata 可能依赖请求数。
uniform：是否 uniform decode。
has_lora / num_active_loras：LoRA graph key 维度。
```

重要点：

```text
CUDAGraphWrapper 不是按 request id 查 graph，而是按 BatchDescriptor 查 captured graph。
```

## 11. ForwardContext

`ForwardContext` 是 runtime 控制总线。

保存：

```text
attn_metadata
slot_mapping
dp_metadata
cudagraph_runtime_mode
batch_descriptor
ubatch_slices
skip_compiled
```

ModelRunner 在 forward 前：

```text
set_forward_context(...)
```

之后：

```text
CUDAGraphWrapper 读取 cudagraph_runtime_mode / batch_descriptor。
compile wrapper 读取 skip_compiled。
attention layer 读取 attn_metadata / slot_mapping。
MoE / DP / ubatch 读取 dp_metadata / ubatch_slices。
```

## 12. compile 和 CUDA graph 的分层关系

要特别背：

```text
CUDAGraphMode.NONE 不等于没有 torch.compile。
```

两层看：

### 第一层：底层 runnable

可能是：

```text
原始 eager model.forward
PyTorch stock torch.compile callable
vLLM VllmBackend 编译后的 callable
PiecewiseBackend 子图
带 static buffer copy 的 compiled callable
```

由 `CompilationMode` 决定。

### 第二层：本轮是否 replay CUDA graph

runtime mode：

```text
FULL
PIECEWISE
NONE
```

由 `CudagraphDispatcher.dispatch()` 和 `ForwardContext` 决定。

## 13. 可能组合

```text
compiled callable + cudagraph_runtime_mode=NONE：
  不 replay CUDA graph，但仍是 compiled forward。

eager runnable + cudagraph_runtime_mode=FULL：
  不 torch.compile，但整段 eager forward 被 full CUDA graph replay。

VLLM_COMPILE + FULL_AND_PIECEWISE：
  decode 走 full graph，mixed batch 走 piecewise graph，不满足走 compiled/eager fallback。
```

## 14. Runtime dispatch 决策点

核心入口：

```text
GPUModelRunner._determine_batch_execution_and_padding()
```

它负责：

```text
1. 判断本轮是否 uniform decode。
2. 判断是否有 encoder output、cascade attention、LoRA 等限制。
3. 先做 sequence parallelism padding。
4. 调 CudagraphDispatcher.dispatch()。
5. DP 场景跨 rank 同步 mode 和 padding。
6. 返回 cudagraph_mode、BatchDescriptor、should_ubatch、num_tokens_across_dp、cudagraph_stats。
```

## 15. CudagraphDispatcher.dispatch

选择顺序：

```text
1. 如果配置或输入不允许 graph，返回 NONE。
2. 把真实 num_tokens pad 到 capture size。
3. 尝试命中 FULL key。
4. 尝试命中 PIECEWISE key。
5. 都不命中则返回 NONE。
```

要背住：

```text
FULL 优先，PIECEWISE 兜底，NONE 保证正确性。
```

## 16. 为什么需要 padding

CUDA graph replay 要固定 shape。

vLLM 把：

```text
真实 num_tokens
  → sequence parallel padding
  → CudagraphDispatcher padding
  → 某个 cudagraph_capture_size
```

padding 必须保证语义安全：

```text
padding token 不写真实 KV cache。
padding request 不读真实 block。
padding 位置不参与 logits / sampling。
attention metadata shape 固定但语义安全。
output 只切回真实 token / request。
```

关键保护：

```text
padding token 的 slot_mapping = -1。
padding request 的 block_table = NULL_BLOCK_ID。
```

## 17. Attention metadata 是核心约束

FULL CUDA graph 包含 attention。

因此不只是 input_ids shape 固定，attention metadata 也必须固定或可 capture。

CommonAttentionMetadata 包含：

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

FULL graph 下需要按 padded shape 构造 metadata。

backend 还要声明支持级别：

```text
ALWAYS
UNIFORM_BATCH
UNIFORM_SINGLE_TOKEN_DECODE
NEVER
```

## 18. capture 生命周期

CUDA graph 通常在 Worker 初始化 / warmup 阶段提前 capture。

主流程：

```text
capture_model()
  → 如果 cudagraph_mode == NONE，跳过。
  → set_cudagraph_capturing_enabled(True)
  → for runtime_mode, batch_descs in get_capture_descs():
        _capture_cudagraphs(batch_descs, runtime_mode)
  → set_cudagraph_capturing_enabled(False)
```

capture 顺序：

```text
PIECEWISE first
FULL second
```

每组内部通常大 shape 优先。

目的：

```text
先 capture 大图，让小图尽量复用 CUDA graph memory pool，降低额外显存。
```

## 19. warmup 和 capture

每个 descriptor 的 capture 分两步：

```text
_warmup_and_capture(desc, runtime_mode)
  → 多次 _dummy_run(..., cudagraph_runtime_mode=NONE)
  → 一次 _dummy_run(..., cudagraph_runtime_mode=runtime_mode, is_graph_capturing=True)
```

为什么 warmup 用 NONE：

```text
先预热 kernel、workspace、attention backend；真正 capture 时才进入 CUDA graph capture。
```

## 20. CUDAGraphWrapper 的边界

`CUDAGraphWrapper` 不负责：

```text
准备 input buffers
判断 batch 是否应该 graph
padding
构造 attention metadata
```

它只负责：

```text
从 ForwardContext 读取 runtime mode 和 BatchDescriptor
mode 不匹配时 pass-through
key 未 capture 时 capture
key 已 capture 时 replay
返回 graph output
```

边界：

```text
ModelRunner / dispatcher 准备可 replay 的输入和 metadata。
CUDAGraphWrapper 负责 capture / replay runnable。
```

## 21. forward / logits / sampler 边界

CUDA graph 主要包住：

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

一句话：

```text
FULL graph 约等于 full model forward，不是 full Engine step。
```

## 22. fallback 是正常路径

CUDA graph 是优化路径，不是正确性路径。

不满足条件时会 fallback：

```text
FULL cudagraph
  → PIECEWISE cudagraph
  → NONE / eager or compiled pass-through
```

常见 fallback 原因：

```text
enforce_eager=True
cudagraph_mode 配置为 NONE
attention backend 不支持 full graph
num_tokens 超过 max_cudagraph_capture_size
BatchDescriptor key 未 capture
本轮不是 uniform decode
cascade attention 禁用 FULL
encoder-decoder 首轮有 encoder input / output
KV scale 计算
LoRA active count 无对应 graph key
DP rank 协调后降级
某些 quantization / MoE / DeepEP 路径不兼容
```

重要点：

```text
fallback 应保证输出语义、KV cache、attention metadata、sampler 和 collective 顺序正确。
```

## 23. 并行为什么影响 CUDA graph

并行下 CUDA graph 还要求：

```text
每个 rank 的本地 tensor shape 可稳定描述。
所有通信 rank 走相同 collective 顺序。
DP rank 对 cudagraph mode / padding / ubatch 达成一致。
TP / SP 对 token 数有对齐要求。
PP 每个 stage capture 自己本地 forward。
MoE / EP 的 all2all backend 支持 capture。
DBO / microbatching 的 ubatch slices 稳定。
```

一句话：

```text
并行场景下，不是某个 rank 能 graph 就够，而是相关 rank 必须用一致 shape、metadata 和通信图 replay。
```

## 24. TP / DP / PP / MoE 影响

```text
TP / SP：
  num_tokens 可能要 pad 到 TP size 倍数。

DP：
  本地先 dispatch，再 all_reduce 同步，mode 取最保守值。

PP：
  不同 PP rank 输入输出不同，每个 rank capture 自己的 stage。

DBO / ubatching：
  一个 batch 拆成多个 ubatch，每个 ubatch 有自己的 ForwardContext。

MoE / EP：
  token routing 和 all2all 通信形态影响 graph 兼容性。
```

## 25. 调试思路

排查 compile / CUDA graph 问题时按层确认：

```text
1. 最终 CompilationConfig 是什么。
2. attention backend 是否自动降级 cudagraph_mode。
3. CudagraphDispatcher 初始化了哪些 FULL / PIECEWISE keys。
4. capture_model 是否真的 capture，耗时和显存是多少。
5. execute_model 每轮 dispatch 到 FULL / PIECEWISE / NONE。
6. BatchDescriptor 是什么，padding 多少。
7. ForwardContext 传了什么 mode 和 descriptor。
8. CUDAGraphWrapper 是 replay、capture 还是 pass-through。
9. padding token、slot_mapping、block_table、attention metadata 是否安全。
```

有用指标：

```text
num_unpadded_tokens
num_padded_tokens
num_paddings
runtime_mode
```

## 26. 关键对象关系

```text
CompilationConfig：启动期配置。
CompilationMode：决定底层 runnable 是 eager / compile / vLLM compile。
CUDAGraphMode：决定 CUDA graph 运行模式。
CudagraphDispatcher：生成合法 BatchDescriptor key，运行时 dispatch。
BatchDescriptor：capture / replay key。
ForwardContext：runtime 控制总线。
CUDAGraphWrapper：pass-through / capture / replay。
GPUModelRunner：准备 buffers、padding、slot mapping、attention metadata、forward。
AttentionMetadataBuilder：构造 backend-specific metadata，并声明 graph 支持。
ModelRunnerOutput.cudagraph_stats：把本轮 graph 统计带回。
```

## 27. 常见易混点

### CUDAGraphMode.NONE 不等于 eager

```text
可能仍然使用 compiled callable，只是不 replay CUDA graph。
```

### FULL graph 不是全 Engine step

```text
通常只覆盖 model forward，不覆盖 sampler、Scheduler、OutputProcessor。
```

### padding 不是随便多算

```text
padding token 必须不写 KV、不参与 logits、不污染输出。
```

### fallback 不是错误

```text
fallback 是保证正确性的正常路径，只是可能影响性能。
```

## 28. 与其他专题的关系

```text
executor_worker_model_runner：ModelRunner 主执行链路，本专题细化 compile / graph 分支。
attention：attention metadata builder 必须支持 capture / replay。
scheduler：SchedulerOutput 决定本轮 num_tokens / num_reqs / batch 形态。
parallelism：TP / PP / DP / MoE 下 graph 要考虑 rank 和 communication。
multimodal：multimodal encoder 有独立 compile / cudagraph 配置和 fallback。
sampling_and_output：sampler 通常在 forward graph 外。
quantization / lora：量化 kernel和 active LoRA 数量可能限制 graph。
```

## 29. 背诵总结

背这一段：

```text
Compilation / CUDA graph 是 vLLM 执行层的性能优化机制。CompilationMode 决定底层 forward callable 是 eager、stock torch.compile 还是 vLLM compile；CUDAGraphMode 决定每轮是否以 FULL、PIECEWISE 或 NONE 方式 replay CUDA graph。由于 serving batch 动态变化，ModelRunner 必须先通过 _determine_batch_execution_and_padding 和 CudagraphDispatcher 把真实 batch 归一成 BatchDescriptor，padding 到 captured shape，保证 padding token 不写 KV、不参与 logits，再构造 capture-compatible attention metadata，通过 ForwardContext 把 runtime mode、BatchDescriptor、metadata 和 slot_mapping 传给模型内部。CUDAGraphWrapper 只负责根据 ForwardContext pass-through、capture 或 replay。CUDA graph 优化的是 model forward，不是完整 Engine step；不满足 shape、backend、LoRA、并行或 metadata 条件时安全 fallback 到 PIECEWISE 或 NONE。
```
