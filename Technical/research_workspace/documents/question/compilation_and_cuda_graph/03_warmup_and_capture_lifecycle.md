# 03. Worker warmup 和 CUDA graph capture 生命周期是什么？

源码位置：

- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/compilation/cuda_graph.py`
- `code/vllm/vllm/compilation/monitor.py`
- `code/vllm/vllm/compilation/counter.py`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/v1/cudagraph_dispatcher.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `code/vllm/vllm/v1/attention/backend.py`

本问题关注：Worker 初始化后，模型如何 warmup、compile 和 capture CUDA graphs；哪些逻辑发生在真实请求之前；为什么 CUDA graph 不是第一轮请求临时捕获；capture 阶段如何复用 `GPUModelRunner.execute_model()` 的输入准备、attention metadata 和 `ForwardContext` 机制。

---

## 0. 梳理规划

本篇按“Worker 生命周期 → compile / warmup → dispatcher key → dummy run → capture wrapper → runtime 复用”的顺序梳理。

要回答的问题分成 11 组：

```text
1. Worker 初始化时 compilation / cudagraph 发生在哪个阶段？
2. load_model() 和 compile_or_warm_up_model() 分别做什么？
3. profile_run / dummy_run / warmup / capture 有什么区别？
4. CudagraphDispatcher 什么时候初始化合法 key？
5. capture_model() 如何枚举要捕获的 FULL / PIECEWISE graph？
6. 为什么 capture 前要先 warmup？
7. _dummy_run() 如何构造固定 shape 的假 batch？
8. capture 时需要准备哪些 static buffer / metadata / KV cache 状态？
9. CUDAGraphWrapper 如何被 dummy forward 触发 capture？
10. capture 后运行时如何 replay，什么情况下 fallback？
11. 如何观察 capture 耗时、显存和 graph 数量？
```

---

## 1. 一句话回答

CUDA graph 通常不是第一轮真实请求临时捕获的，而是在 Worker 初始化 / warmup 阶段，针对 `CudagraphDispatcher` 预先计算的一组 `BatchDescriptor`，用 `_dummy_run()` 构造固定 shape 假输入，经过 `set_forward_context()` 触发 `CUDAGraphWrapper` capture。

最小主线是：

```text
GPUWorker.initialize_from_config()
  → 分配 KV cache
  → GPUModelRunner.compile_or_warm_up_model()
      → profile / dummy warmup
      → 触发 torch.compile / vLLM compile
      → 初始化 attention backend workspace
      → GPUModelRunner.capture_model()
          → dispatcher.get_capture_descs()
          → _capture_cudagraphs()
          → _warmup_and_capture()
              → _dummy_run(..., mode=NONE) 多次 warmup
              → _dummy_run(..., mode=FULL/PIECEWISE, is_graph_capturing=True)
                  → set_forward_context(...)
                  → self.model(...)
                  → CUDAGraphWrapper capture
```

一句话记忆：

```text
warmup / capture 阶段的目标，是在真实请求到来前，为常见固定 shape 准备可 replay 的 GPU forward 执行图。
```

---

## 2. Worker 初始化中的位置

Compilation / CUDA graph 位于 Worker 设备初始化和模型加载之后，真实 serving 之前。

整体顺序可以写成：

```text
GPUWorker.init_device()
  → 初始化 device / distributed / seed / memory snapshot
  → 创建 GPUModelRunner

GPUWorker.load_model()
  → GPUModelRunner.load_model()
  → 加载模型权重
  → 根据 CompilationConfig 包装 model / compile wrapper / cudagraph wrapper

GPUWorker.initialize_from_config()
  → 初始化 KV cache tensors
  → 调用 compile_or_warm_up_model()
  → profile / warmup / capture
  → Worker ready for requests
```

这里要区分三个阶段：

```text
load_model：
  加载权重，创建或包装 model 对象。

compile_or_warm_up_model：
  用 dummy run 触发 compile、warmup kernel、capture CUDA graphs。

execute_model：
  真实请求运行时使用已准备好的 compile / cudagraph 路径。
```

---

## 3. load_model() 阶段做哪些 wrapper 准备

`GPUModelRunner.load_model()` 是模型加载和 wrapper 安装阶段。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5142`

它会：

```text
1. 调 model_loader.load_model() 实例化模型并加载权重；
2. 初始化 LoRA / drafter / MoE / communication buffer 等；
3. 如果是 STOCK_TORCH_COMPILE，调用 self.model.compile(...)；
4. 否则根据 cudagraph_mode 包装：
   - BreakableCUDAGraphWrapper；
   - CUDAGraphWrapper(FULL)；
   - UBatchWrapper；
5. 后续 _model_forward() 统一调用 self.model(...)。
```

关键点：

```text
load_model() 阶段通常只是把 wrapper 装好，
真正 torch.compile tracing 或 CUDA graph capture 多数要等 dummy forward 触发。
```

例如：

```text
@support_torch_compile 注入的 __call__：
  第一次 forward 时触发 Dynamo trace / backend compile。

CUDAGraphWrapper：
  第一次在允许 capture 的 context 中看到某个 BatchDescriptor 时 capture。
```

---

## 4. compile_or_warm_up_model() 的定位

`compile_or_warm_up_model()` 是 Worker 初始化里连接 compile、warmup 和 cudagraph capture 的入口。

相关调用位置在 Worker 初始化流程中。

它要完成几类工作：

```text
- 做 profile / dummy forward，让模型和 kernel 完成冷启动；
- 触发 torch.compile / vLLM compile 的第一次 trace / compile；
- 初始化 attention backend workspace / metadata builder；
- warmup sampler / logits / pooler 相关路径；
- capture CUDA graph；
- 记录 compilation_time / capture memory / capture counters。
```

这里的关键思想是：

```text
服务请求期间不希望突然开始 Dynamo compile 或 CUDA graph capture。
```

因此 vLLM 尽量把这些成本前置到 Worker 初始化阶段。

---

## 5. attention backend 初始化和 cudagraph mode 最终确定

CUDA graph mode 不能只由配置决定，因为 FULL graph 是否可用依赖 attention backend。

初始化 attention backend 后，`GPUModelRunner` 会调用：

```text
_check_and_update_cudagraph_mode()
```

相关位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6831`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6877`

它会检查每个 attention group 的 `AttentionCGSupport`：

```text
ALWAYS
UNIFORM_BATCH
UNIFORM_SINGLE_TOKEN_DECODE
NEVER
```

定义位置：`code/vllm/vllm/v1/attention/backend.py:516`

然后调用：

```text
CompilationConfig.resolve_cudagraph_mode_and_sizes(...)
```

位置：`code/vllm/vllm/config/compilation.py:1316`

这一步可能把用户配置修正为：

```text
FULL → FULL_AND_PIECEWISE
FULL → FULL_DECODE_ONLY
FULL → PIECEWISE
FULL → NONE
```

所以 `capture_model()` 使用的是“backend 修正后的最终 cudagraph_mode”，不是用户原始配置值。

---

## 6. CudagraphDispatcher 什么时候准备合法 key

`CudagraphDispatcher` 定义在：`code/vllm/vllm/v1/cudagraph_dispatcher.py:15`

创建位置通常在 `GPUModelRunner.__init__()` 中。

它初始化时还不能最终确定所有 key，因为：

```text
attention backend 还没初始化，
FULL graph 支持能力还不确定。
```

因此最终 key 初始化发生在 backend 能力确定之后：

```text
cudagraph_dispatcher.initialize_cudagraph_keys(...)
```

入口：`code/vllm/vllm/v1/cudagraph_dispatcher.py:166`

它会创建：

```text
cudagraph_keys[PIECEWISE]
cudagraph_keys[FULL]
```

这些 key 是运行时能 dispatch 的唯一合法集合。

初始化逻辑包括：

```text
1. 如果 cudagraph_mode == NONE，不创建 key；
2. 计算真实 batch size → padded graph size 映射；
3. 生成 LoRA capture cases；
4. 为 mixed mode 创建 PIECEWISE / FULL key；
5. 为 uniform decode routine 创建 FULL key；
6. 标记 keys_initialized=True。
```

---

## 7. capture_model() 主入口

`GPUModelRunner.capture_model()` 是 CUDA graph capture 的主入口。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6584`

可以压缩成伪代码：

```text
capture_model():
  if compilation_config.cudagraph_mode == NONE:
    logger warning / skip
    return 0

  compilation_counter.num_gpu_runner_capture_triggers += 1

  start_time = now
  start_free_gpu_memory = free memory

  set_cudagraph_capturing_enabled(True)
  try:
    for runtime_mode, batch_descs in dispatcher.get_capture_descs():
      _capture_cudagraphs(batch_descs, runtime_mode)
  finally:
    set_cudagraph_capturing_enabled(False)

  elapsed = now - start_time
  graph_memory = start_free_gpu_memory - end_free_gpu_memory
  log "Graph capturing finished ..."
```

几个关键点：

```text
1. capture_model() 只在 cudagraph_mode != NONE 时工作；
2. capture 前会打开全局 capture 许可；
3. capture 后必须关闭许可；
4. 真正 capture 不是 capture_model() 手写 torch.cuda.graph，
   而是 dummy forward 触发 CUDAGraphWrapper；
5. capture 耗时和显存会被记录到日志。
```

---

## 8. get_capture_descs()：capture 哪些 graph

`CudagraphDispatcher.get_capture_descs()` 定义在：

`code/vllm/vllm/v1/cudagraph_dispatcher.py:326`

它返回：

```text
list[tuple[CUDAGraphMode, list[BatchDescriptor]]]
```

顺序是：

```text
PIECEWISE first
FULL second
```

每组内部按大 shape 优先。

设计原因：

```text
先 capture 大 shape，让小 shape 尽量复用大 graph 的 CUDA memory pool，降低额外显存开销。
```

capture descriptors 来自 dispatcher 初始化时建立的合法 key，所以：

```text
capture_model() 不自己猜测要 capture 哪些 batch size；
它只 capture dispatcher 认为 runtime 可能合法 dispatch 的 BatchDescriptor。
```

这保证：

```text
初始化 capture key 和运行时 dispatch key 使用同一套规则。
```

---

## 9. _capture_cudagraphs()：按 runtime mode 捕获一组 graph

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6686`

它负责遍历某一组 `BatchDescriptor`：

```text
_capture_cudagraphs(batch_descs, runtime_mode)
  → tqdm 显示 Capturing CUDA graphs (..., FULL/PIECEWISE)
  → for desc in batch_descs:
       _warmup_and_capture(desc, runtime_mode)
```

它通常会在 global first rank 打进度条，例如：

```text
Capturing CUDA graphs (decode, FULL)
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
```

进度条能直接告诉你当前 capture 的是：

```text
- decode 还是 mixed prefill-decode；
- FULL 还是 PIECEWISE；
- 多少个 descriptor。
```

---

## 10. _warmup_and_capture()：为什么 capture 前要 warmup

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6651`

核心流程：

```text
_warmup_and_capture(desc, runtime_mode):
  for _ in range(cudagraph_num_of_warmups):
    _dummy_run(desc, cudagraph_runtime_mode=NONE, is_graph_capturing=False)

  _dummy_run(desc, cudagraph_runtime_mode=runtime_mode, is_graph_capturing=True)
```

warmup 使用 `NONE`，不是目标 runtime mode。

原因：

```text
- 让 CUDA kernel / Triton kernel / attention workspace 先初始化；
- 避免把一次性初始化开销 capture 进 graph；
- 触发 torch.compile / Inductor 编译；
- 预热 allocator / stream / backend wrapper；
- 检查 shape 和 metadata 构造是否正常。
```

真正 capture 只发生在最后一次 dummy run。

这也解释了：

```text
warmup 慢不一定是 CUDA graph capture 慢；
可能是 torch.compile、Triton JIT、attention workspace 或 kernel 冷启动。
```

---

## 11. _dummy_run() 是 capture 生命周期的核心

`_dummy_run()` 是 warmup / compile / capture 共用的假执行入口。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5658`

它的目标是：

```text
构造一个和目标 BatchDescriptor 一致的 dummy batch，
走和真实 execute_model() 尽量一致的输入准备、metadata 构造和 model forward。
```

### 11.1 dummy run 构造哪些输入

`_dummy_run()` 会构造：

```text
- dummy input_ids；
- dummy positions；
- dummy num_scheduled_tokens；
- dummy query_start_loc；
- dummy seq_lens；
- dummy slot_mapping；
- dummy block_table；
- dummy attention metadata；
- dummy model_kwargs；
- dummy intermediate_tensors（PP 场景）；
- dummy LoRA / multimodal / encoder 相关状态（按需要）。
```

它不需要真实请求语义，但必须满足：

```text
shape 和 capture descriptor 一致；
内存地址稳定；
attention backend 看到的 routine 和真实 runtime 一致；
padding token 不污染真实 KV cache。
```

### 11.2 dummy run 也走 dispatch / padding

`_dummy_run()` 内部也会调用：

```text
_determine_batch_execution_and_padding(...)
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5762`

这很重要：

```text
capture 用的 BatchDescriptor 和真实 runtime dispatch 用的是同一套逻辑。
```

否则就可能出现：

```text
初始化 capture 的 key 和运行时查找的 key 不一致。
```

### 11.3 dummy run 构造 attention metadata

如果是 FULL graph，attention 在 graph 内，dummy run 必须构造 attention metadata。

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5831`

逻辑可以理解为：

```text
if force_attention or cudagraph_runtime_mode == FULL:
  _build_attention_metadata(..., for_cudagraph_capture=is_graph_capturing)
```

FULL capture 时：

```text
for_cudagraph_capture=True
```

然后 attention builder 可以走：

```text
build_for_cudagraph_capture(common_attn_metadata)
```

接口位置：`code/vllm/vllm/v1/attention/backend.py:634`

### 11.4 dummy run 的 slot_mapping 必须安全

dummy run 没有真实 KV slot，所以它会把 slot mapping 填成安全值。

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5821`

语义：

```text
Dummy runs have no real slot assignments；
slot_mapping 填 -1，让 concat_and_cache 等 kernel 跳过 KV 写入。
```

这是避免 capture / warmup 污染真实 KV cache 的关键。

---

## 12. capture 时需要准备什么

CUDA graph capture 要求的不只是固定 `input_ids`。

需要稳定的是一整套 forward 环境。

### 12.1 固定 shape 的输入 buffer

包括：

```text
input_ids
positions
inputs_embeds
intermediate_tensors
slot_mapping
block_table
query_start_loc
seq_lens
attention metadata tensors
```

capture 和 replay 时这些 tensor 的：

```text
shape
地址
参与 kernel 的分支
```

都要稳定。

### 12.2 KV cache 地址稳定

CUDA graph replay 中 attention / KV update 会写读 KV cache。

KV cache tensor 必须在 capture 前已经分配，并且后续 replay 使用同一组 cache buffer。

因此 capture 发生在：

```text
KV cache 初始化之后。
```

### 12.3 attention metadata 结构稳定

FULL graph 下 attention metadata 要固定：

```text
- dataclass 类型；
- tensor shape；
- backend wrapper / plan；
- max_query_len 对应的 routine；
- block_table / slot_mapping shape；
- padding 区域语义。
```

不同 backend 可以通过 `build_for_cudagraph_capture()` 定制 capture metadata。

### 12.4 collective 顺序稳定

TP / PP / DP / MoE 场景下，capture 还要求：

```text
- 每个 rank 的 communication op 顺序稳定；
- DP ranks 的 cudagraph mode 和 padding 协调一致；
- PP 每个 stage 本地 forward 形态稳定；
- MoE / All2All backend 支持 capture。
```

### 12.5 wrapper 层级稳定

capture 的不是裸 forward，而是当前 `self.model` 的 wrapper 层级：

```text
CUDAGraphWrapper(FULL)
  → model.__call__
      → compile wrapper
          → VllmBackend / PiecewiseBackend
              → CUDAGraphWrapper(PIECEWISE)
```

运行时 replay 时，wrapper 层级和 `ForwardContext` 的 mode 必须与 capture 时对应。

---

## 13. CUDAGraphWrapper 如何被触发 capture

`CUDAGraphWrapper` 入口：`code/vllm/vllm/compilation/cuda_graph.py:233`

capture 时，dummy run 会设置：

```text
cudagraph_runtime_mode = FULL 或 PIECEWISE
batch_descriptor = 当前 desc
is_graph_capturing=True
```

然后进入：

```text
with set_forward_context(...):
  self.model(...)
```

`CUDAGraphWrapper.__call__()` 读取 context：

```text
forward_context.cudagraph_runtime_mode
forward_context.batch_descriptor
```

如果 mode 和 wrapper 的 `runtime_mode` 匹配：

```text
1. 找到或创建 entry；
2. 如果 entry.cudagraph is None：
   - validate_cudagraph_capturing_enabled()
   - with torch.cuda.graph(...): runnable(...)
   - 保存 output 和 cudagraph；
3. 后续同 key replay。
```

关键点：

```text
capture_model() 不直接调用 wrapper.capture()；
它通过 dummy forward + ForwardContext 触发 wrapper 普通 __call__ 分支。
```

这样 capture 和真实 runtime replay 使用同一个执行入口。

---

## 14. capture 顺序：PIECEWISE 为什么先于 FULL

`get_capture_descs()` 返回顺序是：

```text
PIECEWISE first
FULL second
```

这看起来反直觉，但有两个原因。

### 14.1 内层 graph 先准备

在 `FULL_AND_PIECEWISE` 模式下，模型可能同时有：

```text
外层 CUDAGraphWrapper(FULL)
内层 CUDAGraphWrapper(PIECEWISE)
```

先 capture piecewise 可以让 compiled subgraph 和内部 wrapper 完成初始化。

### 14.2 大 shape 优先复用 graph pool

每组 descriptor 内部会按大 shape 优先。

原因：

```text
CUDA graph memory pool 可以复用；
先 capture 大 shape，后 capture 小 shape，通常有利于降低额外 graph memory。
```

这也是为什么 capture 进度里常看到大 batch size 先出现。

---

## 15. capture 失败或跳过时会怎样

### 15.1 cudagraph_mode 为 NONE

如果最终：

```text
cudagraph_mode == NONE
```

`capture_model()` 会跳过 capture。

这不是错误，说明配置或 backend 已经决定不使用 CUDA graph。

### 15.2 capture key 不存在

如果 dispatcher 没有初始化某个 key，运行时不会 dispatch 到它。

所以不会出现：

```text
runtime 想 replay 一个根本不合法的 key。
```

### 15.3 runtime 遇到未 capture key

正常情况下，初始化 capture 过的 key 才会运行时命中。

如果某个 wrapper 在运行时 mode 匹配但 entry 没有 graph，会尝试 capture；但这需要：

```text
set_cudagraph_capturing_enabled(True)
```

否则会触发 `validate_cudagraph_capturing_enabled()` 的错误。

相关位置：

- `code/vllm/vllm/compilation/monitor.py:90`
- `code/vllm/vllm/compilation/cuda_graph.py:276`

这个设计是为了防止线上真实请求意外触发 lazy capture。

### 15.4 runtime fallback

如果运行时 batch 不满足已 capture key，`CudagraphDispatcher.dispatch()` 会返回 `NONE`。

这时 wrapper pass-through，底层 runnable 正常执行。

也就是说：

```text
graph miss 是正常 fallback，不应该中断请求。
```

---

## 16. profile_run、warmup、capture 的区别

这几个词容易混淆。

### 16.1 profile_run

`profile_run()` 入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6227`

主要用于：

```text
- 显存 profile；
- 估算可用 KV cache；
- 触发最大形态 dummy run；
- 检查模型路径可执行。
```

它可能也会 warm up 一些 kernel，但目标不是录 CUDA graph。

### 16.2 warmup

warmup 指 capture 前的普通 dummy forward：

```text
_dummy_run(..., cudagraph_runtime_mode=NONE)
```

目标是：

```text
- 触发 compile；
- 初始化 workspace；
- 避免把一次性初始化 capture 进 graph；
- 让后续 capture 更稳定。
```

### 16.3 capture

capture 是：

```text
_dummy_run(..., cudagraph_runtime_mode=FULL/PIECEWISE, is_graph_capturing=True)
```

目标是：

```text
记录固定 shape 的 GPU launch 序列，保存到 CUDAGraphWrapper entry 中。
```

### 16.4 replay

replay 发生在真实请求运行时：

```text
entry.cudagraph.replay()
return entry.output
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:360`

---

## 17. capture 后运行时如何复用

真实请求进来后：

```text
GPUModelRunner.execute_model()
  → _determine_batch_execution_and_padding()
  → 得到 cudagraph_mode + BatchDescriptor
  → set_forward_context(...)
  → self.model(...)
```

`CUDAGraphWrapper` 读取 `BatchDescriptor`：

```text
if entry.cudagraph exists:
  entry.cudagraph.replay()
  return entry.output
```

注意：

```text
entry.output 是 capture 时保存的输出 tensor 引用；
replay 会把新结果写进同一块输出 tensor；
调用方拿到的是同一结构的 output 引用。
```

这就是为什么 input/output buffer 地址稳定非常重要。

DEBUG 日志下，wrapper 会检查 replay 输入 tensor 地址是否和 capture 时一致。

相关位置：`code/vllm/vllm/compilation/cuda_graph.py:346`

---

## 18. capture 和 torch.compile 的关系

CUDA graph capture 和 torch.compile 不是同一件事。

### 18.1 compile 可能在 warmup 时触发

模型类如果有 `@support_torch_compile`，第一次 dummy forward 会触发：

```text
Dynamo trace
VllmBackend compile
PiecewiseBackend compile
Inductor compile
```

这通常发生在 warmup 阶段，而不是 capture 阶段。

### 18.2 capture 记录的是 compiled 或 eager runnable 的 GPU work

capture 时 wrapper 录的是当前 runnable 的 GPU launch 序列。

这个 runnable 可能是：

```text
- eager forward；
- stock torch.compile forward；
- vLLM compiled runtime callable；
- piecewise compiled subgraph。
```

因此：

```text
compile 是构造更好的 runnable；
cudagraph 是记录 runnable 在固定 shape 下的 GPU 执行序列。
```

### 18.3 piecewise cudagraph 依赖 compile boundary

PIECEWISE graph 依赖 vLLM compile 把模型 forward 切成子图。

所以：

```text
PIECEWISE cudagraph requires piecewise compilation；
FULL cudagraph can work with or without compilation。
```

相关说明在 `CompilationConfig` 注释附近：`code/vllm/vllm/config/compilation.py:615`

---

## 19. capture 统计和日志

### 19.1 capture 耗时和显存

`capture_model()` 会记录：

```text
Graph capturing finished in %.0f secs, took %.2f GiB
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6640`

它回答：

```text
capture 总耗时；
CUDA graph 额外占用显存。
```

### 19.2 capture 计数器

`compilation_counter` 定义在：`code/vllm/vllm/compilation/counter.py:20`

相关字段：

```text
num_gpu_runner_capture_triggers
num_cudagraph_captured
```

更新位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6595`
- `code/vllm/vllm/compilation/cuda_graph.py:339`

### 19.3 compile 耗时

`monitor_torch_compile()` 记录 compile 时间。

位置：`code/vllm/vllm/compilation/monitor.py:17`

它会累加到：

```text
CompilationConfig.compilation_time
CompilationConfig.encoder_compilation_time
```

### 19.4 runtime metrics

capture 完成不代表运行时一定命中 graph。

运行时要看：

```text
cudagraph_metrics
```

它记录：

```text
num_unpadded_tokens
num_padded_tokens
num_paddings
runtime_mode
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907`

---

## 20. 常见问题

### 20.1 CUDA graph 是不是第一轮请求才 capture？

通常不是。

vLLM 会在 Worker 初始化 / capture_model() 阶段提前 capture。

如果运行时意外 lazy capture，通常会被 `validate_cudagraph_capturing_enabled()` 阻止。

### 20.2 warmup 是不是 capture？

不是。

warmup 用：

```text
cudagraph_runtime_mode=NONE
```

capture 用：

```text
cudagraph_runtime_mode=FULL/PIECEWISE
is_graph_capturing=True
```

### 20.3 capture 的 graph 是按 request 数还是 token 数？

核心 key 是 `BatchDescriptor`。

它包含：

```text
num_tokens
num_reqs
uniform
has_lora
num_active_loras
```

所以既有 token shape，也可能有 request shape 和 LoRA 维度。

### 20.4 capture 完成后为什么 runtime 还会 NONE？

因为运行时 batch 可能不满足 capture key：

```text
- num_tokens 超过 max capture size；
- 非 uniform decode；
- cascade attention；
- encoder output；
- LoRA case 不匹配；
- DP rank 协调降级；
- attention backend 只支持更窄场景。
```

### 20.5 capture OOM 怎么理解？

每个 CUDA graph 都可能占用 graph memory pool。

capture sizes 太多、LoRA cases 太多、FULL + PIECEWISE 都启用、encoder graph 也启用时，启动显存压力会增加。

调试时优先看：

```text
cudagraph_capture_sizes
max_cudagraph_capture_size
cudagraph_num_of_warmups
cudagraph_specialize_lora
Graph capturing finished ... took X GiB
```

---

## 21. 最小伪代码

### 21.1 Worker 初始化

```text
worker.init_device()
worker.load_model()
worker.initialize_from_config(kv_cache_config):
  allocate_kv_cache()
  model_runner.compile_or_warm_up_model()
```

### 21.2 compile / warmup / capture

```text
compile_or_warm_up_model():
  profile_run()
  initialize_attn_backend()
  cudagraph_dispatcher.initialize_cudagraph_keys(...)
  capture_model()
```

### 21.3 capture_model

```text
capture_model():
  if cudagraph_mode == NONE:
    return

  set_cudagraph_capturing_enabled(True)
  for runtime_mode, descs in dispatcher.get_capture_descs():
    for desc in descs:
      _warmup_and_capture(desc, runtime_mode)
  set_cudagraph_capturing_enabled(False)
```

### 21.4 warmup_and_capture

```text
_warmup_and_capture(desc, runtime_mode):
  for _ in range(cudagraph_num_of_warmups):
    _dummy_run(desc, cudagraph_runtime_mode=NONE)

  _dummy_run(
    desc,
    cudagraph_runtime_mode=runtime_mode,
    is_graph_capturing=True,
  )
```

### 21.5 dummy_run

```text
_dummy_run(desc, mode, is_graph_capturing):
  construct dummy num_scheduled_tokens
  cudagraph_mode, batch_desc = _determine_batch_execution_and_padding(...)
  slot_mapping = fill(-1)
  attn_metadata = _build_attention_metadata(
    for_cudagraph_capture=is_graph_capturing,
  )

  with set_forward_context(
    attn_metadata,
    cudagraph_runtime_mode=mode,
    batch_descriptor=batch_desc,
    slot_mapping=slot_mapping,
  ):
    self.model(...)
```

---

## 22. 一句话总结

Worker warmup / capture 生命周期的核心是：

```text
先在初始化阶段用 dummy batch 触发 compile 和 kernel warmup，再对 dispatcher 认为合法的 BatchDescriptor 执行 CUDA graph capture；真实请求只负责 dispatch 到已 capture 的 key 并 replay，命不中就 fallback。
```

最核心的流程是：

```text
KV cache ready
  → attention backend ready
  → dispatcher keys ready
  → warmup dummy forward
  → capture dummy forward
  → CUDAGraphWrapper 保存 graph
  → runtime dispatch
  → replay or pass-through
```
