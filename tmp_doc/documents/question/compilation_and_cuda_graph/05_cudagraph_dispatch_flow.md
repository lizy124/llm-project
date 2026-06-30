# 05. ModelRunner 每轮如何选择 eager / compiled / cudagraph replay？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\cudagraph_dispatcher.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\cuda_graph.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\forward_context.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backend.py`
- `D:\lzy\project\kv_pool\code\vllm\docs\design\cuda_graphs.md`

本问题关注：真实请求执行时，`GPUModelRunner.execute_model()` 如何判断本轮是否可以使用 CUDA Graph，如何把判断结果放进 `ForwardContext`，以及最终如何 dispatch 到 eager / compiled / cudagraph capture / cudagraph replay 路径。

---

## 1. 一句话回答

vLLM V1 的 cudagraph dispatch 不是在 `_model_forward()` 里临时判断，而是在 forward 前先由 `GPUModelRunner._determine_batch_execution_and_padding()` 计算本轮的 `CUDAGraphMode` 和 `BatchDescriptor`，再通过 `set_forward_context()` 写入全局 forward context；模型或编译子图外层的 `CUDAGraphWrapper` 读取这个 context，命中已 capture 的 key 就 replay，没有命中或 mode 为 `NONE` 就普通调用底层 runnable。

主链路是：

```text
GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
  → _determine_batch_execution_and_padding()
      → 判断 uniform decode / mixed prefill-decode
      → 计算 LoRA dispatch case
      → 按 capture sizes 做 padding
      → 调 CudagraphDispatcher.dispatch()
      → 必要时做 DP rank 间协调
      → 得到 cudagraph_mode + BatchDescriptor
  → _get_slot_mappings()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(..., cudagraph_runtime_mode, batch_descriptor, ...)
  → _model_forward()
      → self.model(...)
      → FULL wrapper / PIECEWISE wrapper 读 ForwardContext
      → cudagraph replay / cudagraph capture / pass-through
```

所以：

```text
CudagraphDispatcher 决定“本轮应该用哪个 graph key”；
ForwardContext 负责把决定传给模型内部；
CUDAGraphWrapper 负责真正 capture / replay / fallback；
_model_forward() 本身只调用 self.model，不直接分支。
```

---

## 2. 先给结论：每轮 dispatch 看哪些东西

每轮 cudagraph dispatch 的核心输入不是“请求数量”一个字段，而是下面几类信息组合起来的 batch 描述。

### 2.1 token shape

来自 `SchedulerOutput` 和 `InputBatch`：

- `num_tokens`：本轮真实 scheduled token 总数
- `num_reqs`：当前 batch 的请求数
- `max_num_scheduled_tokens`：单个请求本轮最多执行多少 token
- `uniform_decode`：是否所有请求都是同一个 decode query length

其中 `uniform_decode` 的判断在 `_is_uniform_decode()`：

```text
max_num_scheduled_tokens == uniform_decode_query_len
并且
num_tokens == max_num_scheduled_tokens * num_reqs
```

`uniform_decode_query_len` 通常是：

```text
1 + num_speculative_tokens
```

所以它既覆盖普通 decode，也覆盖 speculative decode 的验证步。

### 2.2 cudagraph 配置

来自 `compilation_config`：

- `cudagraph_mode`
- `cudagraph_capture_sizes`
- `max_cudagraph_capture_size`
- `compile_sizes`
- `cudagraph_specialize_lora`
- `cudagraph_num_of_warmups`

这些配置决定：

```text
哪些 batch size 会被 capture；
小 batch 要 pad 到哪个 captured size；
FULL / PIECEWISE / FULL_DECODE_ONLY / FULL_AND_PIECEWISE 是否可用；
LoRA 是否要按 active adapter 数拆 graph key。
```

### 2.3 attention backend 能力

attention backend 通过 `AttentionCGSupport` 表达 full cudagraph 支持程度：

```text
ALWAYS
UNIFORM_BATCH
UNIFORM_SINGLE_TOKEN_DECODE
NEVER
```

`GPUModelRunner.initialize_attn_backend()` 初始化 attention backend 后，会调用 `_check_and_update_cudagraph_mode()` 修正最终的 `compilation_config.cudagraph_mode`。

典型含义是：

```text
如果 backend 不支持 full cudagraph，就不能强行走 FULL；
如果只支持 uniform decode full graph，就只能在 uniform decode 场景走 FULL；
如果 piecewise 编译可用，可以把不支持 full 的部分退到 PIECEWISE；
否则退到 NONE。
```

### 2.4 runtime feature 限制

即使配置允许 cudagraph，本轮也可能因为动态特性被降级：

- `force_eager=True`：profile / warmup / 显式 eager
- `use_cascade_attn=True`：禁用 `FULL`，可退到 `PIECEWISE` 或 `NONE`
- encoder-decoder 首轮带 encoder output：禁用 `FULL`，并在 forward context 里 `skip_compiled=True`
- `calculate_kv_scales=True`：本轮 forward 前强制 `CUDAGraphMode.NONE`
- `num_tokens > max_cudagraph_capture_size`：超出 capture 范围，返回 `NONE`
- batch key 不存在：返回 `NONE`
- DP ranks 需要统一 padding / mode：任一 rank eager 可能带动整体 eager

### 2.5 LoRA dispatch key

LoRA 会进入 `BatchDescriptor`：

```text
has_lora
num_active_loras
```

如果 `cudagraph_specialize_lora` 开启，dispatcher 会把真实 active LoRA 数映射到已 capture 的 bucket，例如捕获 powers of 2 或 `max_loras + 1` 这类 case。

---

## 3. 初始化阶段：先准备“哪些 key 是合法的”

运行时不是看到某个 batch size 才随便决定 capture。`CudagraphDispatcher` 会先维护两组合法 dispatch key：

```text
cudagraph_keys[PIECEWISE]
cudagraph_keys[FULL]
```

### 3.1 CudagraphDispatcher 创建

`GPUModelRunner.__init__()` 中创建：

```text
self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:819`

dispatcher 初始化时会记录：

```text
compilation_config
uniform_decode_query_len = 1 + num_speculative_tokens
specialize_lora_count
cudagraph_mode = NONE  # 默认先关闭，等 attention backend 初始化后再启用
```

### 3.2 attention backend 初始化后才能确定最终 mode

`initialize_attn_backend()` 会先收集每个 KV cache group / attention group 的 backend，然后调用：

```text
_check_and_update_cudagraph_mode(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6831`

原因是 `FULL` cudagraph 能否使用依赖 attention backend；backend 未初始化时，不能最终决定是否支持 full graph。

### 3.3 initialize_cudagraph_keys() 建立合法 key 集合

`CudagraphDispatcher.initialize_cudagraph_keys()` 的职责是：

```text
输入：最终 cudagraph_mode + uniform_decode_query_len
输出：FULL / PIECEWISE 各自有哪些 BatchDescriptor 可以被 dispatch
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:166`

它会：

1. 如果 mode 是 `NONE`，直接标记 initialized，然后不创建任何 key。
2. 预计算 `batch size → padded graph size` 的映射。
3. 根据 LoRA 配置生成需要 capture 的 LoRA cases。
4. 为 mixed mode 创建 `PIECEWISE` 或 `FULL` 的 key。
5. 如果 decode mode 是 `FULL` 且是 separate routine，再为 uniform decode 创建 `FULL` key。

### 3.4 BatchDescriptor 是 cudagraph dispatch 的 key

定义在 `forward_context.py`：

```text
BatchDescriptor(
  num_tokens,
  num_reqs=None,
  uniform=False,
  has_lora=False,
  num_active_loras=0,
)
```

位置：`code/vllm/vllm/forward_context.py:29`

字段含义：

- `num_tokens`：通常是 padded 后的 token 数
- `num_reqs`：请求数；`PIECEWISE` 可为 `None`，表示不按请求数区分
- `uniform`：是否 uniform decode
- `has_lora`：是否存在 active LoRA
- `num_active_loras`：用于 LoRA specialization

### 3.5 PIECEWISE key 更“宽松”

dispatcher 创建 `PIECEWISE` key 时会把：

```text
num_reqs=None
uniform=False
```

这表示 piecewise graph 只需要按 token shape 匹配，不需要精确区分请求数和 uniform decode routine。

而 `FULL` graph 通常需要更精确，因为 full graph 包含 attention，attention 的调度 metadata 可能依赖请求数、uniform decode routine 等。

---

## 4. capture 阶段：先把候选 graph 录好

CUDA Graph capture 发生在模型加载 / warmup 阶段，不是在每个真实请求里重新录图。

### 4.1 capture_model() 入口

`GPUModelRunner.capture_model()`：

```text
if cudagraph_mode == NONE:
  跳过 capture
else:
  set_cudagraph_capturing_enabled(True)
  for runtime_mode, batch_descs in cudagraph_dispatcher.get_capture_descs():
      _capture_cudagraphs(batch_descs, runtime_mode)
  set_cudagraph_capturing_enabled(False)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6583`

### 4.2 capture 顺序

`get_capture_descs()` 返回的顺序是：

```text
PIECEWISE first
FULL second
```

每组内部按 `num_tokens` 从大到小排序。

这样做的目的：

```text
先 capture 大 shape，让小 shape 可以复用 CUDA graph memory pool，降低额外显存占用。
```

### 4.3 warmup 和 capture 分开

每个 `BatchDescriptor` 会先 warmup，再 capture：

```text
_warmup_and_capture(desc, runtime_mode)
  → 多次 _dummy_run(..., cudagraph_runtime_mode=NONE)
  → 一次 _dummy_run(..., cudagraph_runtime_mode=runtime_mode, is_graph_capturing=True)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6651`

warmup 使用 `NONE`，目的是让 kernel / workspace / attention backend 先初始化；真正 capture 时才使用 `FULL` 或 `PIECEWISE`。

### 4.4 dummy run 也走同一套 dispatch 逻辑

`_dummy_run()` 内部也调用 `_determine_batch_execution_and_padding()`：

```text
_dummy_run()
  → 构造 dummy num_scheduled_tokens
  → _determine_batch_execution_and_padding(...)
  → _build_attention_metadata(...)
  → set_forward_context(...)
  → self.model(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5657`

这保证 capture 用的 `BatchDescriptor` 和真实运行时 dispatch 用的是同一套规则。

### 4.5 FULL capture 需要 attention metadata

`FULL` graph 包含整个 model forward，包括 attention，所以 `_dummy_run()` 在 `cudagraph_runtime_mode == FULL` 时会构造 attention metadata。

关键点：

```text
uniform_decode=True  → max_query_len = uniform_decode_query_len
uniform_decode=False → max_query_len = num_tokens
```

这样 attention backend 可以在 capture 时进入和真实运行一致的 kernel routine，例如 decode routine 或 mixed prefill-decode routine。

---

## 5. 真实执行：execute_model() 中的 dispatch 位置

真实请求进入 `GPUModelRunner.execute_model()` 后，dispatch 位于输入准备和 attention metadata 构建之间。

核心顺序：

```text
_update_states(scheduler_output)
_prepare_inputs(...)
_compute_cascade_attn_prefix_lens(...)
_determine_batch_execution_and_padding(...)
_get_slot_mappings(...)
_build_attention_metadata(...)
_preprocess(...)
set_forward_context(...)
_model_forward(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`

### 5.1 为什么 dispatch 在 attention metadata 前

因为 `FULL` graph 需要 padded attention metadata：

```text
pad_attn = cudagraph_mode == CUDAGraphMode.FULL
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4196`

如果本轮 dispatch 到 `FULL`，那么：

- `slot_mapping` 要按 padded token/request 维度准备
- attention metadata 要传入 `num_tokens_padded`
- padding token 的 slot 要填 `-1`
- ubatch slicing 也要按 padded shape 对齐

如果本轮不是 `FULL`，attention 通常按真实 token 数构造，piecewise graph 或 eager 路径不要求 full forward 的 padded attention 形状。

### 5.2 _determine_batch_execution_and_padding() 做什么

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`

它做五件事：

```text
1. 判断 uniform_decode
2. 判断是否有 encoder output、LoRA、cascade attention 等限制
3. 先做 sequence parallelism padding
4. 调 CudagraphDispatcher.dispatch() 得到 runtime mode + BatchDescriptor
5. 如有 DP，再跨 rank 协调 padding 和 cudagraph mode
```

输出是：

```text
cudagraph_mode
batch_descriptor
should_ubatch
num_tokens_across_dp
cudagraph_stats
```

### 5.3 uniform_decode 的意义

`uniform_decode` 表示本轮所有请求 query length 一样。

普通 decode：

```text
每个请求 1 个 token
num_tokens == num_reqs
max_num_scheduled_tokens == 1
```

spec decode 验证步：

```text
每个请求 1 + num_speculative_tokens 个 token
num_tokens == num_reqs * (1 + num_speculative_tokens)
```

很多 attention backend 只在这种 uniform batch 下支持 full cudagraph，所以它直接影响 `FULL_DECODE_ONLY` 和 `FULL_AND_PIECEWISE` 的分流。

### 5.4 cascade attention 会禁用 FULL

真实 dispatch 时：

```text
disable_full = use_cascade_attn or has_encoder_output
```

然后传给 dispatcher：

```text
invalid_modes={CUDAGraphMode.FULL}
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3865`

含义是：

```text
如果可用，退到 PIECEWISE；
如果没有 PIECEWISE key，则退到 NONE；
不会因为 FULL 不可用就报错。
```

### 5.5 profile / warmup 会强制 eager

`force_eager=True` 时，dispatcher 收到：

```text
valid_modes={CUDAGraphMode.NONE}
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3861`

所以无论 key 是否存在，本轮都返回 `NONE`。

---

## 6. CudagraphDispatcher.dispatch() 如何选 mode

`dispatch()` 是运行时选择 CUDA Graph 的中心逻辑。

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:235`

### 6.1 先计算 allowed_modes

```text
allowed_modes = valid_modes or valid_runtime_modes()
if invalid_modes:
    allowed_modes -= invalid_modes
```

所以调用方可以表达两类约束：

```text
valid_modes={NONE}       → 强制 eager
invalid_modes={FULL}     → 禁用 full，允许 piecewise / none
valid_modes={某个 DP 同步 mode} → DP 协调后所有 rank 走同一种 mode
```

### 6.2 快速 fallback 到 NONE

以下情况直接返回：

```text
CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
```

条件包括：

- keys 尚未初始化
- 全局 cudagraph mode 是 `NONE`
- `max_cudagraph_capture_size` 为空
- `num_tokens > max_cudagraph_capture_size`
- allowed modes 只剩 `NONE`

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:274`

### 6.3 把真实 token 数 pad 到 capture size

dispatcher 内部用 `_create_padded_batch_descriptor()` 把真实 token 数映射到 captured size。

例如 `cudagraph_capture_sizes=[1, 8, 16]` 时，可能出现：

```text
真实 num_tokens=5  → padded num_tokens=8
真实 num_tokens=9  → padded num_tokens=16
```

注意这里返回的是 padded 后的 `BatchDescriptor`，后续 attention metadata / input buffer / slot mapping 都会按这个 padded shape 处理。

### 6.4 FULL 优先，然后 PIECEWISE，最后 NONE

dispatch 搜索顺序：

```text
1. 如果 FULL allowed，检查 FULL key 是否存在
2. 如果 PIECEWISE allowed，检查 relaxed PIECEWISE key 是否存在
3. 否则返回 NONE
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:307`

这就是 `FULL_AND_PIECEWISE` 的核心：

```text
uniform decode 能命中 FULL → 用 FULL；
非 uniform / mixed prefill-decode → 通常退到 PIECEWISE；
没有匹配 key → NONE。
```

### 6.5 PIECEWISE 使用 relaxed key

PIECEWISE 检查时会把 descriptor 改成：

```text
replace(batch_desc, num_reqs=None, uniform=False)
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:316`

因此同一个 token size 的 piecewise graph 可以服务更多 batch 形态。

---

## 7. set_forward_context() 如何把 dispatch 结果传下去

`execute_model()` 在真正 forward 前调用：

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

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`

`ForwardContext` 中和 cudagraph dispatch 直接相关的字段是：

```text
cudagraph_runtime_mode
batch_descriptor
skip_compiled
attn_metadata
slot_mapping
ubatch_slices
dp_metadata
```

位置：`code/vllm/vllm/forward_context.py:128`

关键点：

```text
_model_forward() 不把 cudagraph_mode 作为显式参数传给模型；
模型内部的 CUDAGraphWrapper / attention / compiled graph 都从 ForwardContext 读运行态。
```

---

## 8. _model_forward() 为什么看起来没有分支

`_model_forward()` 实现非常简单：

```text
return self.model(
  input_ids=input_ids,
  positions=positions,
  intermediate_tensors=intermediate_tensors,
  inputs_embeds=inputs_embeds,
  **model_kwargs,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3757`

这不是说明没有 dispatch，而是说明 dispatch 被下沉到：

```text
模型外层 wrapper
compiled callable
piecewise backend wrapper
attention backend
ForwardContext
```

换句话说：

```text
execute_model() 负责“决定”；
_model_forward() 负责“调用”；
CUDAGraphWrapper 负责“执行路径分发”。
```

---

## 9. CUDAGraphWrapper 如何 capture / replay / fallback

`CUDAGraphWrapper` 是最终执行分支所在。

位置：`code/vllm/vllm/compilation/cuda_graph.py:145`

### 9.1 wrapper 绑定 runtime mode

每个 wrapper 初始化时绑定一个 mode：

```text
runtime_mode = FULL 或 PIECEWISE
```

运行时它从 `ForwardContext` 读取：

```text
forward_context.cudagraph_runtime_mode
forward_context.batch_descriptor
```

### 9.2 mode 不匹配就 pass-through

如果：

```text
cudagraph_runtime_mode == NONE
或
cudagraph_runtime_mode != wrapper.runtime_mode
```

则直接：

```text
return self.runnable(*args, **kwargs)
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:244`

这就是 eager / compiled fallback 的关键：

```text
fallback 不需要抛异常；wrapper 只是不启用 cudagraph，继续调用底层 runnable。
```

### 9.3 key 首次出现时 capture

如果 mode 匹配，但 `batch_descriptor` 不在 `concrete_cudagraph_entries` 中：

```text
创建 CUDAGraphEntry
validate_cudagraph_capturing_enabled()
with torch.cuda.graph(...):
    output = runnable(...)
保存 output 和 cudagraph
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:256`

正常情况下这些 key 在 `capture_model()` 阶段已经 capture；如果未来支持 lazy capture，这里也能承接。

### 9.4 key 已存在时 replay

如果 entry 已有 `cudagraph`：

```text
entry.cudagraph.replay()
return entry.output
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:357`

debug 模式下还会检查输入 tensor 地址是否和 capture 时一致，避免 replay 时输入地址变化导致错误。

### 9.5 nested wrapper 如何区分 FULL 和 PIECEWISE

vLLM 可以同时存在：

```text
外层 FULL wrapper：包住整个 model forward
内层 PIECEWISE wrapper：包住 compiled / split 后的子图
```

运行时靠 `cudagraph_runtime_mode` 区分：

- mode 为 `FULL`：外层 FULL wrapper 生效，内层 PIECEWISE wrapper pass-through
- mode 为 `PIECEWISE`：外层 FULL wrapper pass-through，内层 PIECEWISE wrapper 生效
- mode 为 `NONE`：两个 wrapper 都 pass-through

这让 `FULL_AND_PIECEWISE` 可以共存，而不需要在 `_model_forward()` 里写复杂分支。

---

## 10. eager / compiled / cudagraph 的真实关系

这里容易混淆：`CUDAGraphMode.NONE` 不等价于“完全没有 torch.compile”。

可以分成两层看：

```text
第一层：是否使用 cudagraph replay
  - FULL
  - PIECEWISE
  - NONE

第二层：底层 runnable 是什么
  - eager model
  - torch.compile 后的 model
  - piecewise compiled 子图
```

因此：

```text
cudagraph_runtime_mode=NONE
```

只表示本轮不 capture/replay CUDA Graph；底层仍可能是 compiled callable。

`skip_compiled=True` 才表示绕过 compiled path，例如 encoder-decoder 带 encoder input 的首轮：

```text
skip_compiled=has_encoder_input
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4312`

---

## 11. 几个典型 runtime 场景

### 11.1 普通 decode，FULL_AND_PIECEWISE

```text
num_reqs = N
num_tokens = N
max_num_scheduled_tokens = 1
uniform_decode = True
```

dispatch 倾向：

```text
FULL key 存在 → CUDAGraphMode.FULL
否则 PIECEWISE key 存在 → CUDAGraphMode.PIECEWISE
否则 NONE
```

forward 时：

```text
FULL wrapper replay 整个 model forward
attention metadata 使用 padded shape
```

### 11.2 spec decode 验证步

```text
uniform_decode_query_len = 1 + num_speculative_tokens
num_tokens = num_reqs * uniform_decode_query_len
uniform_decode = True
```

如果 attention backend 支持 uniform batch full graph，则可走 `FULL`；否则走 `PIECEWISE` 或 `NONE`。

### 11.3 mixed prefill-decode

```text
部分请求 prefill 多 token
部分请求 decode 1 token
uniform_decode = False
```

在 `FULL_AND_PIECEWISE` 下通常：

```text
FULL decode key 不匹配
PIECEWISE key 命中 → CUDAGraphMode.PIECEWISE
```

如果没有 piecewise compilation / piecewise key，则退到 `NONE`。

### 11.4 cascade attention

```text
use_cascade_attn = True
invalid_modes = {FULL}
```

结果：

```text
PIECEWISE 可用 → PIECEWISE
否则 → NONE
```

### 11.5 encoder-decoder 首轮

如果本轮有 encoder input / encoder output：

```text
has_encoder_output=True → dispatch 禁用 FULL
has_encoder_input=True  → set_forward_context(skip_compiled=True)
```

含义：

```text
首轮带 encoder 的路径动态性更强，不走 full graph；
纯 decoder step 后续才可能使用 cudagraph / compiled path。
```

### 11.6 active LoRA

如果当前 batch 有 LoRA：

```text
has_lora=True
num_active_loras=len(active adapters)
```

dispatcher 会把它映射到 capture 时准备的 LoRA case，形成不同 `BatchDescriptor`。

如果对应 LoRA key 不存在，则 fallback 到 `NONE`。

### 11.7 超出 capture size

如果：

```text
num_tokens > max_cudagraph_capture_size
```

则：

```text
CUDAGraphMode.NONE
BatchDescriptor(num_tokens)
```

不会报错，只是不使用 cudagraph。

---

## 12. DP 场景下为什么要二次协调

在 data parallel 场景，多个 DP rank 的本地 batch 可能不同。

`_determine_batch_execution_and_padding()` 会先本地 dispatch，然后调用 DP 协调逻辑：

```text
coordinate_batch_across_dp(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3882`

协调后可能得到：

```text
should_ubatch
num_tokens_across_dp
synced_cudagraph_mode
```

如果 `num_tokens_across_dp` 不为空，当前 rank 会用同步后的 token 数再次 dispatch：

```text
cudagraph_mode, batch_descriptor = dispatch_cudagraph(
  num_tokens_padded,
  valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3898`

原因是：

```text
MoE / DP / SP 场景下，各 rank 需要一致的通信和 graph shape；
不能 rank0 replay graph、rank1 eager，或者各自 padding 到不同 shape。
```

---

## 13. padding 后哪些数据也要跟着变

dispatch 返回的 `batch_descriptor.num_tokens` 可能大于真实 `num_tokens`。

后续会用 padded size 影响：

```text
num_tokens_padded
num_reqs_padded
ubatch_slices_padded
slot_mapping
attention metadata
input_ids / positions / inputs_embeds 切片长度
ForwardContext.num_tokens
```

### 13.1 slot_mapping 的 padding

`_get_slot_mappings()` 会把 padding token 的 slot 填为 `-1`：

```text
slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4008`

这对 full cudagraph 很重要，因为 reshape/cache 类 kernel 看到固定 shape，但 padding token 不能写入真实 KV cache。

### 13.2 FULL 才需要 pad attention

```text
pad_attn = cudagraph_mode == CUDAGraphMode.FULL
```

`FULL` graph 包含 attention，所以 metadata 也必须固定 shape；`PIECEWISE` graph 通常不 capture attention，因此不一定按 full padded attention metadata 构造。

---

## 14. fallback 语义

CUDA Graph dispatch 的 fallback 是正常路径，不是异常路径。

典型 fallback 原因：

```text
cudagraph_mode 被配置为 NONE
attention backend 不支持目标 mode
num_tokens 超过 max capture size
对应 BatchDescriptor key 不存在
本轮 force_eager/profile/warmup
cascade attention 禁用 FULL 且没有 PIECEWISE
encoder-decoder 动态首轮
KV scale calculation 本轮不兼容
LoRA case 未 capture
```

fallback 行为：

```text
CudagraphDispatcher 返回 CUDAGraphMode.NONE；
ForwardContext 写入 NONE；
CUDAGraphWrapper 看到 NONE 后 pass-through；
底层 runnable 正常执行；
输出语义保持一致，只是没有 cudagraph replay 加速。
```

如果开启 `observability_config.cudagraph_metrics`，`_determine_batch_execution_and_padding()` 会记录：

```text
num_unpadded_tokens
num_padded_tokens
num_paddings
runtime_mode
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907`

---

## 15. 最小伪代码

把真实链路压缩成伪代码，可以这样理解：

```text
# execute_model()
_update_states(scheduler_output)
logits_indices, spec_decode_metadata = _prepare_inputs(...)

cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, stats = \
    _determine_batch_execution_and_padding(
        num_tokens=total_num_scheduled_tokens,
        num_reqs=input_batch.num_reqs,
        max_num_scheduled_tokens=max(...),
        use_cascade_attn=...,
        num_encoder_reqs=...,
    )

num_tokens_padded = batch_desc.num_tokens
pad_attn = cudagraph_mode == FULL

slot_mappings = _get_slot_mappings(
    num_tokens_padded if pad_attn else num_tokens_unpadded,
    ...,
)

attn_metadata = _build_attention_metadata(
    num_tokens=num_tokens_unpadded,
    num_tokens_padded=num_tokens_padded if pad_attn else None,
    ...,
)

input_ids, inputs_embeds, positions, model_kwargs = _preprocess(...)

with set_forward_context(
    attn_metadata,
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    slot_mapping=slot_mappings,
    skip_compiled=has_encoder_input,
):
    output = self.model(...)
```

`CUDAGraphWrapper.__call__()` 则可以简化为：

```text
ctx = get_forward_context()

if ctx.cudagraph_runtime_mode == NONE:
    return runnable(...)

if ctx.cudagraph_runtime_mode != self.runtime_mode:
    return runnable(...)

entry = graphs[ctx.batch_descriptor]

if entry.cudagraph is None:
    capture runnable into CUDA graph
    return output

entry.cudagraph.replay()
return entry.output
```

---

## 16. 一句话总结

```text
vLLM 的 cudagraph dispatch 核心是：先把本轮 batch 归一成 BatchDescriptor，再由 CudagraphDispatcher 选择 FULL / PIECEWISE / NONE，最后通过 ForwardContext 让 CUDAGraphWrapper 自动 replay、capture 或 pass-through；所有不满足固定 shape 或 backend 能力的场景都安全退回普通执行路径。
```
