# 10. 算子如何和 CUDA Graph / torch compile 协同？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\cudagraph_dispatcher.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\cuda_graph.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\breakable_cudagraph.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\decorators.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\monitor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\forward_context.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\selector.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\encoder_cudagraph.py`

这个问题关注：算子在 CUDA Graph capture、torch compile、warmup、padding、static buffer、shape stability 场景下有什么要求，以及 fallback 如何影响图捕获和编译路径。

---

## 1. 一句话回答

CUDA Graph / torch compile 要求执行路径、张量 shape、内存地址和控制流尽量稳定，因此 vLLM 会把动态 batch padding 到预先 capture 的 batch descriptor，并通过 cudagraph dispatcher、forward context、attention metadata builder、static buffer 和 backend capability，把算子约束到可 replay / 可 compile 的路径上。

最小链路是：

```text
CompilationConfig
  → attention backend capability
  → cudagraph capture sizes / compile sizes
  → CudagraphDispatcher keys
  → runtime batch padding
  → set_forward_context(..., cudagraph_runtime_mode, batch_descriptor)
  → model forward with selected operators
  → capture or replay CUDA graph
```

---

## 2. 先区分两条机制

vLLM 里经常同时提到 CUDA Graph 和 torch compile，但它们不是一回事。

```text
torch compile
  → 让 PyTorch / Dynamo / Inductor 编译 Python 模型图或子图；
  → 关注 graph break、dynamic shape、FX graph、backend compiler。

CUDA Graph
  → 录制一段 GPU kernel launch 序列并 replay；
  → 关注 shape、内存地址、kernel 顺序、capture 时是否有非法操作。
```

在 V1 GPUModelRunner 中：

```text
STOCK_TORCH_COMPILE 模式：直接 self.model.compile(fullgraph=True, backend=...)
其他 vLLM compile 模式：由 CUDAGraphWrapper / BreakableCUDAGraphWrapper / UBatchWrapper / CudagraphDispatcher 控制 cudagraph。
```

位置：`vllm/v1/worker/gpu_model_runner.py:5273`

---

## 3. 模型初始化时如何接入 compile / cudagraph

模型加载完成后，`GPUModelRunner` 会根据 `compilation_config.mode` 和 `cudagraph_mode` 包装模型。

关键逻辑：

```python
if compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
    self.model.compile(fullgraph=True, backend=backend)
    return

if is_breakable_cudagraph_enabled() and cudagraph_mode != CUDAGraphMode.NONE:
    self.model = BreakableCUDAGraphWrapper(...)
elif cudagraph_mode.has_full_cudagraphs():
    self.model = CUDAGraphWrapper(..., runtime_mode=CUDAGraphMode.FULL)
elif parallel_config.use_ubatching:
    self.model = UBatchWrapper(...)
```

位置：`vllm/v1/worker/gpu_model_runner.py:5273`

这说明：

```text
stock torch compile 是一种独立模式；
vLLM 自己的 cudagraph runtime 通过 wrapper 和 dispatcher 管理；
ubatching 又会引入 UBatchWrapper。
```

---

## 4. CUDAGraphMode 的运行时含义

从 `GPUModelRunner.execute_model()` 的注释可见，runtime mode 常见值是：

```text
NONE：不使用 CUDA graph，直接 eager / compiled forward；
PIECEWISE：只对部分子图或分段 forward 使用 CUDA graph；
FULL：对完整可捕获 forward 路径使用 CUDA graph。
```

位置：`vllm/v1/worker/gpu_model_runner.py:5661`

FULL 模式要求最高：

```text
attention metadata 更稳定；
slot mapping / KV cache update 需要适配 padded shape；
某些功能如 cascade attention、encoder input、动态 KV scale 会禁用 full graph。
```

PIECEWISE 要求相对低，但仍然需要选中的 attention backend 和 compile splitting 支持。

---

## 5. attention backend 会决定 cudagraph 支持上限

attention backend 不是只影响 attention kernel，也会影响 cudagraph mode。

`GPUModelRunner` 在初始化 attention metadata builder 后会解析 cudagraph mode：

```python
cg_support = builder_cls.get_cudagraph_support(...)
cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(...)
self.cudagraph_dispatcher.initialize_cudagraph_keys(...)
```

位置：`vllm/v1/worker/gpu_model_runner.py:6877`

所以整体顺序是：

```text
select attention backend
  → 创建 metadata builder
  → 查询 backend/builder 的 cudagraph support
  → resolve cudagraph mode 和 capture sizes
  → 初始化 cudagraph dispatch keys
```

这也是为什么 backend fallback 会影响 cudagraph：一个 backend 支持 full cudagraph，另一个 backend 可能只支持 piecewise 或 none。

---

## 6. CudagraphDispatcher 做什么

`CudagraphDispatcher` 定义在：`vllm/v1/cudagraph_dispatcher.py:15`

它保存两组 key：

```text
CUDAGraphMode.PIECEWISE → set[BatchDescriptor]
CUDAGraphMode.FULL → set[BatchDescriptor]
```

位置：`vllm/v1/cudagraph_dispatcher.py:39`

注释中说得很直接：

```text
dispatcher 是 runtime cudagraph dispatching keys 的唯一真实来源；
runtime 会根据 input key 生成 cudagraph mode 和 batch descriptor；
wrapper 根据 forward context 中的 dispatch key capture / replay / passthrough。
```

位置：`vllm/v1/cudagraph_dispatcher.py:15`

可以把它理解成：

```text
真实 batch shape
  → padding / normalization
  → BatchDescriptor
  → 是否命中已 capture graph
  → FULL / PIECEWISE / NONE
```

---

## 7. BatchDescriptor 解决什么问题

CUDA Graph 需要固定 shape 和固定执行路径。vLLM 用 `BatchDescriptor` 描述可 capture / replay 的 batch：

```text
num_tokens
num_reqs
uniform
has_lora
num_active_loras
```

相关创建逻辑：`vllm/v1/cudagraph_dispatcher.py:132`

`num_tokens` 会被 padding 到 capture size：

```python
num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]
```

位置：`vllm/v1/cudagraph_dispatcher.py:141`

uniform decode 场景下，还会根据 `uniform_decode_query_len` 推导 `num_reqs`：

```python
num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
```

位置：`vllm/v1/cudagraph_dispatcher.py:143`

所以 batch descriptor 不是简单 batch size，而是“这个 graph 能 replay 的执行形态”。

---

## 8. capture sizes 如何映射到 padding

`_compute_bs_to_padded_graph_size()` 会预先建立：

```text
真实 num_tokens → padded graph size
```

位置：`vllm/v1/cudagraph_dispatcher.py:72`

逻辑是：

```text
capture_sizes = [s1, s2, s3, ...]
max_cudagraph_capture_size = M

num_tokens 落在某段区间时，pad 到该段右边界 capture size；
如果刚好等于 capture size，则不变。
```

同时它会校验：

```text
compile_sizes 中的值不能被 cudagraph padding 改变；
否则 torch compile size 和 cudagraph replay size 不一致。
```

位置：`vllm/v1/cudagraph_dispatcher.py:93`

这解释了一个常见报错：

```text
compile_sizes contains X which would be padded to Y.
```

解决方向不是随便改算子，而是让 compile_sizes 使用 cudagraph_capture_sizes 中不会被 padding 的值。

---

## 9. dispatch 运行时如何决定 FULL / PIECEWISE / NONE

运行时入口：

```python
def dispatch(num_tokens, uniform_decode=False, has_lora=False, ...)
```

位置：`vllm/v1/cudagraph_dispatcher.py:235`

关键条件：

```text
keys 尚未初始化 → NONE；
cudagraph_mode == NONE → NONE；
num_tokens > max_cudagraph_capture_size → NONE；
当前功能只允许 NONE → NONE；
FULL key 命中 → FULL；
PIECEWISE relaxed key 命中 → PIECEWISE；
都不命中 → NONE。
```

位置：`vllm/v1/cudagraph_dispatcher.py:274`

FULL key 更严格，PIECEWISE 可以把 key 放宽成：

```python
replace(batch_desc, num_reqs=None, uniform=False)
```

位置：`vllm/v1/cudagraph_dispatcher.py:313`

这就是为什么有些 batch 不能 full graph，但还能 piecewise graph。

---

## 10. runtime batch padding 在哪里发生

`GPUModelRunner._determine_batch_execution_and_padding()` 负责 runtime 决策。

位置：`vllm/v1/worker/gpu_model_runner.py:3810`

它会计算：

```text
uniform_decode
has_encoder_output
has_lora / num_active_loras
num_tokens_padded
cudagraph_mode
batch_descriptor
should_ubatch
num_tokens_across_dp
cudagraph_stats
```

核心调用：

```python
cudagraph_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
    num_tokens=num_tokens,
    has_lora=has_lora,
    uniform_decode=uniform_decode,
    num_active_loras=num_active_loras,
    valid_modes=...,
    invalid_modes=...,
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3855`

如果启用了 sequence parallelism，还会先 pad 到 tensor parallel size 的倍数：

```python
return round_up(num_scheduled_tokens, tp_size)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3407`

---

## 11. 哪些功能会禁用或限制 full cudagraph

在 runtime 决策和 forward 前，代码会按功能降级 cudagraph mode。

常见情况：

```text
cascade attention：disable FULL；
encoder-decoder 首次带 encoder input：skip compiled / eager；
calculate_kv_scales：动态计算 KV scales，强制 NONE；
num_tokens 超过 max_cudagraph_capture_size：NONE；
当前 batch descriptor 没有 capture key：NONE；
某些 LoRA specialize 场景没有匹配 graph：降级；
某些 spec decode drafter 只支持 PIECEWISE。
```

示例：cascade attention / encoder output 禁用 full：

```python
dispatch_cudagraph(
    num_tokens_padded,
    disable_full=use_cascade_attn or has_encoder_output,
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3865`

动态 KV scale 禁用 cudagraph：

```python
if self.calculate_kv_scales:
    cudagraph_mode = CUDAGraphMode.NONE
```

位置：`vllm/v1/worker/gpu_model_runner.py:4282`

encoder input 跳过 compiled：

```python
skip_compiled=has_encoder_input
```

位置：`vllm/v1/worker/gpu_model_runner.py:4312`

---

## 12. padding 后 attention metadata 怎么保持一致

padding 不只是把 input_ids 变长，还会影响 attention metadata、slot mapping、KV cache update。

`execute_model()` 中会判断：

```python
pad_attn = cudagraph_mode == CUDAGraphMode.FULL
```

位置：`vllm/v1/worker/gpu_model_runner.py:4196`

如果 full cudagraph，需要用 padded dimensions 构造 slot mappings 和 attention metadata：

```python
slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
    num_tokens_padded=num_tokens_padded if pad_attn or has_separate_kv_update else num_tokens_unpadded,
    num_reqs_padded=num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs,
    ...
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4244`

attention metadata builder 也会拿到 padded 参数：

```python
_build_attention_metadata(
    num_tokens=num_tokens_unpadded,
    num_tokens_padded=num_tokens_padded if pad_attn else None,
    num_reqs=num_reqs,
    num_reqs_padded=num_reqs_padded if pad_attn else None,
    ...
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4255`

所以算子看到的 shape 可能是 padded shape，但有效 token 数仍由 metadata 区分。

---

## 13. forward context 如何把 cudagraph 信息传给算子

模型 forward 前会设置 forward context：

```python
with set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    ubatch_slices=ubatch_slices_padded,
    slot_mapping=slot_mappings,
    skip_compiled=has_encoder_input,
):
    model_output = self._model_forward(...)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4303`

这一步非常关键：

```text
attention layer、KV cache op、compile wrapper、cudagraph wrapper 都通过 forward context 知道当前 runtime mode 和 batch descriptor。
```

没有这个 context，底层算子无法知道自己应该：

```text
使用 padded shape；
执行 graph replay；
走 eager fallback；
跳过 compiled path；
使用哪组 attention metadata。
```

---

## 14. capture_model 做了什么

`capture_model()` 是真正 capture CUDA graph 的入口。

位置：`vllm/v1/worker/gpu_model_runner.py:6584`

核心流程：

```text
1. 如果 cudagraph_mode == NONE，直接报错；
2. 初始化 encoder cudagraph manager；
3. 创建 graph memory pool；
4. set_cudagraph_capturing_enabled(True)；
5. 遍历 cudagraph_dispatcher.get_capture_descs()；
6. 对每个 runtime mode 和 batch descriptor 调 _capture_cudagraphs()；
7. 如有 multimodal encoder graph，也 capture encoder graph；
8. set_cudagraph_capturing_enabled(False)；
9. 返回 capture 消耗的内存。
```

位置：`vllm/v1/worker/gpu_model_runner.py:6602`

`get_capture_descs()` 返回的 descriptor 是按 token 数从大到小排序的，方便内存复用和 profiling。

位置：`vllm/v1/cudagraph_dispatcher.py:326`

---

## 15. _capture_cudagraphs 如何模拟 batch

`_capture_cudagraphs()` 会对每个 capture batch descriptor 做 warmup 和 capture。

位置：`vllm/v1/worker/gpu_model_runner.py:6686`

上层调用中可以看到，capture 期间会强制传入：

```text
cudagraph_runtime_mode
batch_descriptor
force_uniform_decode
force_has_lora
force_num_active_loras
```

相关调用：`vllm/v1/worker/gpu_model_runner.py:6654`

这保证 capture 时的 batch 形态和 runtime dispatcher 之后能命中的 key 一致。

换句话说：

```text
capture 录的是“未来可能 replay 的 padded batch descriptor”，不是某个真实用户请求。
```

---

## 16. 算子进入 CUDA Graph 需要满足什么

一个算子想稳定进入 CUDA Graph replay，通常需要满足：

```text
1. 输入 tensor shape 在 capture / replay 间一致；
2. tensor 地址稳定，不能每轮重新分配不同 buffer；
3. kernel launch 序列一致；
4. 不能在 capture 中做不支持的 CPU sync / cuda malloc / random host control；
5. 不能依赖 Python 动态分支改变执行路径；
6. backend 本身声明支持对应 cudagraph mode；
7. attention metadata / slot mapping 使用 padded shape 对齐。
```

vLLM 为此做了几类工程化处理：

```text
static input buffers；
padded batch descriptor；
prebuilt attention metadata；
ForwardContext；
CUDAGraphWrapper；
BreakableCUDAGraphWrapper；
CUDA graph capture sizes；
async output copy 与 forward 解耦。
```

---

## 17. 为什么有些算子需要 static buffer

CUDA Graph replay 复用 capture 时的内存地址。如果每轮 forward 都创建新 tensor，地址可能变化，graph 不能安全 replay。

因此 GPUModelRunner 会维护大量 persistent buffers，例如：

```text
input_ids.gpu
inputs_embeds.gpu
positions
slot mappings
block tables
attention metadata buffer
mamba state buffer
LoRA mapping buffer
```

模型 forward 前只是把本轮数据 copy / slice 到这些 buffer，再用 padded view 执行。

这也是为什么 `_preprocess()` 中常见：

```python
self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(...)
input_ids = self.input_ids.gpu[:num_input_tokens]
positions = self.positions[:num_input_tokens]
```

位置：`vllm/v1/worker/gpu_model_runner.py:3426`

---

## 18. attention 算子与 cudagraph 的特殊关系

attention 是最复杂的 cudagraph 参与者，因为它依赖：

```text
query length
KV cache block table
slot mapping
prefix / decode / prefill metadata
cascade attention metadata
KV cache layout
backend-specific workspace
```

FULL cudagraph 下，attention metadata 也需要按 padded shape 构造。

PIECEWISE cudagraph 下，attention 往往作为可分割 op 或 graph break 边界处理，由 compile config 的 splitting ops 和 backend support 决定。

所以 attention backend selection、metadata builder、cudagraph mode 三者必须一致。

---

## 19. sampling / logits 算子是否进入 CUDA Graph

sampling / logits 相关算子和 CUDA Graph 的关系更细：

```text
lm_head / logits processor 可能处于 forward 或 postprocess 边界；
Sampler、top-k/top-p、logprobs gather 常有动态 request metadata；
D2H copy / Python list 转换不适合放进 graph；
spec decode rejection sampler 有 Triton kernel，但输出长度和 metadata 更动态。
```

因此 vLLM 更强调：

```text
模型 forward 侧用 cudagraph 稳定主干性能；
sampling 输出侧减少同步、减少分配、使用 optimized sampler backend。
```

例如 `Sampler.gather_logprobs()` 中使用：

```python
torch._dynamo.decorators.mark_unbacked(logprobs, 0)
```

位置：`vllm/v1/sample/sampler.py:345`

这是为了避免 compile 对 batch dimension 过度 specialization。

---

## 20. torch compile 下的 graph break 风险

torch compile 关注 Python / Dynamo 图是否能稳定 trace。

风险来源包括：

```text
Python side control flow 依赖 tensor 值；
动态 shape 过多；
unsupported custom op；
CPU sync；
hooks / NVTX tracing；
动态创建 tensor / list；
不同 batch size 导致 specialization recompile。
```

代码中对 NVTX hooks 有专门处理：

```text
STOCK_TORCH_COMPILE 模式下不注册 layerwise hooks；
因为 hook functions 会被 Dynamo trace，而 nvtx.range_push/pop 不可 trace。
```

位置：`vllm/v1/worker/gpu_model_runner.py:3941`

这说明一些调试工具本身也可能破坏 compile path。

---

## 21. fallback 如何影响 cudagraph / compile

backend fallback 可能有三种影响：

### 21.1 完全不影响正确性，只影响速度

例如：

```text
FlashInfer sampler fallback 到 native；
小 batch top-k/top-p fallback 到 PyTorch；
某些基础算子 fallback 到 torch op。
```

结果通常一致，但 kernel 数量和同步点可能增加。

### 21.2 降低 cudagraph mode

例如：

```text
attention backend 不支持 FULL；
KV cache layout / metadata builder 不支持 full graph；
cascade attention 本轮启用；
encoder input 存在；
KV scale 动态计算。
```

这时 runtime mode 可能从 FULL 变 PIECEWISE 或 NONE。

### 21.3 触发 compile graph break 或 recompile

例如：

```text
fallback 到未被 compile 支持的 op；
shape 没有 pad 到 compile_sizes；
batch size 从 1 到 N 触发 specialization；
动态 per-request metadata 改变 Python 分支。
```

这类问题通常表现为首次请求慢、重复 recompile、graph break 日志或 cudagraph miss。

---

## 22. CUDA Graph 与 async output copy 的边界

CUDA Graph 主要覆盖 forward compute，而输出 CPU 化通常在 graph 外处理。

async scheduling 下，采样输出会包装成 `AsyncGPUModelRunnerOutput`：

```text
sampled_token_ids / logprobs / routed_experts
  → 在独立 copy stream 上 D2H
  → event 标记完成
  → get_output() 等待并转 Python 结构
```

位置：`vllm/v1/worker/gpu_model_runner.py:239`

这样做是为了：

```text
让 D2H copy 和下一轮 GPU compute 重叠；
避免输出解析阻塞 cudagraph replay 主路径。
```

---

## 23. multimodal encoder cudagraph

V1 还支持部分 multimodal encoder cudagraph。

在 `_execute_mm_encoder()` 中：

```python
if self.encoder_cudagraph_manager is not None and self.encoder_cudagraph_manager.supports_modality(modality):
    cudagraph_output = self.encoder_cudagraph_manager.execute(mm_kwargs_batch)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3073`

初始化入口：

```python
_create_encoder_cudagraph_manager()
_maybe_init_encoder_cudagraph_manager()
```

位置：`vllm/v1/worker/gpu_model_runner.py:6399`

capture_model 中也会 capture encoder graph：

```python
encoder_cudagraph_manager.capture(graph_pool=encoder_graph_pool)
```

位置：`vllm/v1/worker/gpu_model_runner.py:6621`

这说明 cudagraph 不只服务 decoder-only LLM forward，也可以覆盖稳定 shape 的 multimodal encoder 子图。

---

## 24. spec decode 与 cudagraph

spec decode 会改变每轮执行形态：

```text
主模型 target logits；
draft model / EAGLE / ngram proposal；
rejection sampler；
bonus token；
accepted token 数不固定。
```

因此 spec decode 需要额外的 cudagraph 约束。

代码中可以看到：

```text
Eagle currently only supports PIECEWISE cudagraphs.
```

位置：`vllm/v1/worker/gpu_model_runner.py:5974`

另外 `uniform_decode_query_len` 会包含 speculative tokens：

```python
self.uniform_decode_query_len = 1 + self.vllm_config.num_speculative_tokens
```

位置：`vllm/v1/cudagraph_dispatcher.py:37`

这会影响 full decode graph 的 `num_tokens → num_reqs` 计算。

---

## 25. LoRA 与 cudagraph

LoRA 会改变执行路径和 active adapter 数量，所以 cudagraph key 里包含：

```text
has_lora
num_active_loras
```

`CudagraphDispatcher._get_lora_cases()` 会决定 capture 哪些 LoRA case：

```text
没有 LoRA：只 capture no-LoRA；
启用 cudagraph_specialize_lora：capture no-LoRA + 部分 active LoRA counts；
不 specialize：capture max_loras + 1 的泛化 LoRA case。
```

位置：`vllm/v1/cudagraph_dispatcher.py:111`

runtime dispatch 时，如果实际 active LoRA 数不在 capture list 中，会找一个更大的 captured count 来匹配：

```python
idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
```

位置：`vllm/v1/cudagraph_dispatcher.py:283`

这避免为每一种 LoRA 数量都 capture 一张 graph。

---

## 26. data parallel / ubatching 对 cudagraph 的影响

`_determine_batch_execution_and_padding()` 中，data parallel 会协调各 rank 的 batch：

```python
coordinate_batch_across_dp(...)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3882`

如果不同 DP rank 的 token 数不同，需要同步 padded token count 和 cudagraph mode，避免某些 rank replay graph、另一些 rank eager 导致 collective 不匹配。

ubatching 则可能使用 `UBatchWrapper`，并把 `ubatch_slices` 放入 forward context。

位置：`vllm/v1/worker/gpu_model_runner.py:4171`

这说明 cudagraph 不只看单卡 shape，还要看 distributed execution 的一致性。

---

## 27. profiler / metrics 如何观察 cudagraph

如果启用 cudagraph metrics，会构造：

```python
CUDAGraphStat(
    num_unpadded_tokens=num_tokens,
    num_padded_tokens=batch_descriptor.num_tokens,
    num_paddings=batch_descriptor.num_tokens - num_tokens,
    runtime_mode=str(cudagraph_mode),
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3907`

`ModelRunnerOutput` 中也有：

```text
cudagraph_stats: CUDAGraphStat | None
```

位置：`vllm/v1/outputs.py`

排查时重点看：

```text
runtime_mode 是否频繁 NONE；
num_paddings 是否过多；
是否命中预期 capture size；
是否因 encoder / cascade / KV scale / LoRA 降级；
profiler 中 kernel launch 是否明显减少。
```

---

## 28. capture 失败时如何定位

可以按下面顺序排查：

```text
1. 确认 cudagraph_mode 不是 NONE；
2. 确认 attention backend 支持目标 cudagraph mode；
3. 确认 cudagraph_capture_sizes 和 compile_sizes 不冲突；
4. 确认 batch token 数不超过 max_cudagraph_capture_size；
5. 看 runtime 是否因 cascade / encoder input / calc_kv_scales 禁用 FULL；
6. 看是否 LoRA active count 没有匹配 captured key；
7. 看 backend fallback 是否换成不支持 capture 的实现；
8. 打开 graph break / compile debug 日志定位具体 op；
9. 用 profiler 看是否真的 replay，而不是 eager kernel launch。
```

最常见的误判是：

```text
配置里启用了 cudagraph，不代表每个 batch 都会 replay cudagraph。
```

runtime dispatcher 每轮都会重新判断。

---

## 29. 一个完整 runtime 时间线

```text
1. Scheduler 给出本轮 scheduled tokens；
2. GPUModelRunner 统计 num_tokens / num_reqs / max_query_len；
3. 判断 uniform decode、LoRA、encoder input、cascade attention；
4. CudagraphDispatcher.dispatch() 返回 runtime mode 和 batch descriptor；
5. 按 batch descriptor padding input / attention metadata / slot mapping；
6. set_forward_context 写入 cudagraph_runtime_mode 和 batch_descriptor；
7. CUDAGraphWrapper / BreakableCUDAGraphWrapper 决定 replay 或 passthrough；
8. attention / MLP / norm / quantized linear 等算子按 selected backend 执行；
9. forward 输出 hidden states / logits；
10. sampling / bookkeeping / async copy 处理输出；
11. cudagraph_stats 随 ModelRunnerOutput 返回。
```

---

## 30. 容易疑惑的点

### 30.1 padding 会不会改变模型结果？

不会。padding token / request 只是为了匹配 graph shape。attention metadata、slot mapping、有效 token 数会区分真实 token 和 padding token。

### 30.2 为什么有时启用了 CUDA Graph 仍然走 eager？

因为 runtime dispatcher 没命中 valid key，或本轮功能不允许 graph，例如 token 数超上限、encoder input、cascade attention、动态 KV scale、LoRA case 不匹配。

### 30.3 FULL 和 PIECEWISE 哪个更快？

通常 FULL launch overhead 更低，但要求更高；PIECEWISE 更容易覆盖动态场景。实际取决于 attention backend、batch shape、模型结构和功能组合。

### 30.4 fallback 到 torch op 是否一定破坏 CUDA Graph？

不一定。某些 torch op 可以被 capture 或 compile；问题在于它是否引入动态分配、CPU sync、unsupported operation 或改变 graph shape。

### 30.5 compile_sizes 和 cudagraph_capture_sizes 为什么要对齐？

因为 torch compile 产生的图和 CUDA Graph replay 的 padded shape 必须一致。compile size 如果会被 padding 到另一个 capture size，就会造成编译图和 replay 图不一致。

---

## 31. 总结

CUDA Graph / compile 协同链路可以压缩成：

```text
backend capability
  → resolve cudagraph mode
  → initialize capture keys
  → pad runtime batch to BatchDescriptor
  → set ForwardContext
  → wrapper capture / replay
  → selected operators execute on stable shapes
  → fallback to eager when key or capability mismatch
```

如果只记住一句话：

```text
vLLM 不是让任意动态 batch 直接进入 CUDA Graph，而是先把 batch、attention metadata、LoRA 状态和 backend 能力归一成可 capture 的 BatchDescriptor，再让算子在稳定 shape 和稳定执行路径下 replay。
```
