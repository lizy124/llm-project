# 07. Model forward 如何被 compile wrapper / graph runner 包装？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\decorators.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\wrapper.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\backends.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\piecewise_backend.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\cuda_graph.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\breakable_cudagraph.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_ubatch_wrapper.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\forward_context.py`
- `D:\lzy\project\kv_pool\code\vllm\docs\design\torch_compile.md`
- `D:\lzy\project\kv_pool\code\vllm\docs\design\cuda_graphs.md`

本问题关注：`GPUModelRunner._model_forward()` 看起来只是 `self.model(...)`，但这个 `self.model` 在加载、编译、warmup 和 CUDA Graph capture 后可能已经被多层 wrapper 包装。本文梳理普通 eager forward、`@support_torch_compile` 注入的 compile wrapper、vLLM 自定义 `VllmBackend/PiecewiseBackend`、`CUDAGraphWrapper`、`BreakableCUDAGraphWrapper`、`UBatchWrapper` 之间的关系，以及运行时如何通过 `ForwardContext` 在 compiled / cudagraph replay / fallback 之间切换。

---

## 1. 一句话回答

`GPUModelRunner._model_forward()` 始终只调用统一接口：

```text
self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)
```

但 `self.model` 可能已经是：

```text
原始 nn.Module
@support_torch_compile 注入了 TorchCompileWithNoGuardsWrapper 的 nn.Module
外层 CUDAGraphWrapper(FULL)
外层 BreakableCUDAGraphWrapper
外层 UBatchWrapper
内部由 VllmBackend 拆出来的 PiecewiseBackend 子图
内部再包 CUDAGraphWrapper(PIECEWISE)
```

运行时并不是 `_model_forward()` 自己判断走 eager / compile / replay，而是：

```text
load_model() / @support_torch_compile / torch.compile 后端
  → 决定 self.model 和子图被哪些 wrapper 包起来

execute_model() / set_forward_context()
  → 写入 cudagraph_runtime_mode、BatchDescriptor、skip_compiled

wrapper.__call__()
  → 根据 ForwardContext 选择 replay、capture 或 pass-through
```

---

## 2. 最小主链路

### 2.1 运行时 forward 主链路

```text
GPUModelRunner.execute_model()
  → _prepare_inputs()
  → _determine_batch_execution_and_padding()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(
        cudagraph_runtime_mode,
        batch_descriptor,
        skip_compiled,
        attn_metadata,
        slot_mapping,
    )
  → _model_forward()
      → self.model(...)
          → 外层 FULL CUDAGraphWrapper / UBatchWrapper / BreakableCUDAGraphWrapper
          → @support_torch_compile 注入的 __call__
          → torch.compile compiled callable
          → VllmBackend runtime callable
          → PiecewiseBackend 子图
          → PIECEWISE CUDAGraphWrapper
          → 原始 forward / attention / custom ops
```

### 2.2 模型加载和包装主链路

```text
GPUModelRunner.load_model()
  → model_loader.load_model()
      → 实例化模型类
      → 如果模型类有 @support_torch_compile，__init__ 中准备 compile wrapper
  → LoRA / drafter / MoE / comm buffer 初始化
  → 如果 STOCK_TORCH_COMPILE：self.model.compile(fullgraph=True, backend=...)
  → 否则：
      如果 breakable cudagraph 开启：self.model = BreakableCUDAGraphWrapper(self.model)
      elif cudagraph_mode has FULL：self.model = CUDAGraphWrapper(self.model, FULL)
      elif ubatching：self.model = UBatchWrapper(self.model, ...)
  → 后续 _model_forward() 只调用 self.model(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5142`

---

## 3. _model_forward() 本身做什么

`_model_forward()` 定义在：

`code/vllm/vllm/v1/worker/gpu_model_runner.py:3757`

实现非常薄：

```python
return self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

它不做：

```text
不判断 cudagraph mode；
不判断 compile mode；
不构造 attention metadata；
不做 logits；
不做 sampling；
不做 graph replay。
```

它的意义是给所有 runner 子类保留一个最小可覆盖点：

```text
只看模型执行本身，不把 execute_model() 的调度、输入准备、采样逻辑揉进去。
```

---

## 4. 第一层包装：模型类上的 @support_torch_compile

vLLM 很多模型类会加：

```python
@support_torch_compile
class XxxModel(nn.Module):
    ...
```

定义位置：`code/vllm/vllm/compilation/decorators.py:118`

### 4.1 decorator 改了什么

`support_torch_compile()` 最重要的动作是：

```text
把 TorchCompileWithNoGuardsWrapper 加到模型类的 bases 里；
替换模型类 __init__；
替换模型类 __call__；
记录 dynamic_arg_dims；
在第一次调用时触发 torch.compile。
```

核心位置：`code/vllm/vllm/compilation/decorators.py:331`

关键代码行为：

```text
cls.__bases__ = cls.__bases__ + (TorchCompileWithNoGuardsWrapper,)
cls.__init__ = wrapped_init
cls.__call__ = wrapped_call
```

所以它不是在外面包一个普通 Python 对象，而是让模型实例本身也具备 compile wrapper 的行为。

### 4.2 dynamic_arg_dims 的含义

如果 decorator 没显式传 `dynamic_arg_dims`，会从 `forward()` 的类型注解推断：

```text
torch.Tensor / torch.Tensor | None → 默认第 0 维动态
IntermediateTensors → 每个内部 tensor 第 0 维动态
```

位置：`code/vllm/vllm/compilation/decorators.py:153`

这和 vLLM 的 batch 表达一致：

```text
大多数模型 forward 的动态维就是 num_tokens。
```

例如：

```text
input_ids.shape[0]
positions.shape[0]
inputs_embeds.shape[0]
intermediate_tensors 中 hidden state 的 shape[0]
```

### 4.3 do_not_compile 何时为 True

模型实例初始化时会设置：

```text
do_not_compile =
  compilation_config.mode in [NONE, STOCK_TORCH_COMPILE]
  or ignore_torch_compile
  or enable_if(vllm_config) is False
```

位置：`code/vllm/vllm/compilation/decorators.py:385`

含义：

```text
CompilationMode.NONE：完全 eager，不走 decorator compile；
STOCK_TORCH_COMPILE：由 GPUModelRunner.load_model() 顶层调用 self.model.compile()；
某些模型/子模块显式 ignore：跳过；
encoder / multimodal 等可通过 enable_if 控制。
```

### 4.4 decorator 注入的 __call__ 做什么

被注入的 `__call__()` 位于：

`code/vllm/vllm/compilation/decorators.py:502`

流程：

```text
if do_not_compile or torch.compiler.is_compiling():
    return forward(...)

if ForwardContext.skip_compiled:
    return forward(...)

if aot_compiled_fn exists:
    return aot_compiled_fn(...)

if already compiled:
    return TorchCompileWithNoGuardsWrapper.__call__(...)

first call:
    mark dynamic inputs
    patch Dynamo / Inductor configs
    call TorchCompileWithNoGuardsWrapper.__call__(...)
    set compiled=True
```

也就是说：

```text
第一次真实/dummy forward 会触发 Dynamo tracing 和 backend compilation；
之后直接调用 compiled callable；
如果本轮 ForwardContext 要求 skip_compiled，就绕回原始 forward。
```

### 4.5 skip_compiled 是什么

`ForwardContext.skip_compiled` 定义在：

`code/vllm/vllm/forward_context.py:150`

`GPUModelRunner.execute_model()` 会在 encoder-decoder 首轮带 encoder input 时设置：

```text
skip_compiled=has_encoder_input
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4312`

含义：

```text
某些调用的 tensor 类型/shape/分支变化太大，不能复用一个 compiled graph；
这时保留 outer wrapper，但本轮直接走原始 forward。
```

---

## 5. 第二层包装：TorchCompileWithNoGuardsWrapper

定义位置：`code/vllm/vllm/compilation/wrapper.py:47`

它是 `@support_torch_compile` 注入的基类，核心职责是：

```text
创建 torch.compile(self.forward, fullgraph=True, dynamic=False, backend=backend)
运行时调用 _compiled_callable；
在非 STOCK_TORCH_COMPILE 模式下丢弃 guards，避免反复 Dynamo recompile；
可选使用 bytecode hook 直接 dispatch 到 compiled code；
提供 NVTX tracing 包裹。
```

### 5.1 初始化时创建 compiled callable

`TorchCompileWithNoGuardsWrapper.__init__()` 中：

```text
backend = compilation_config.init_backend(vllm_config, prefix=..., is_encoder=...)
self._compiled_callable = torch.compile(
    self.forward,
    fullgraph=True,
    dynamic=False,
    backend=backend,
    options=options,
)
```

位置：`code/vllm/vllm/compilation/wrapper.py:90`

注意：这里创建的是 torch.compile wrapper，真正 Dynamo trace 通常在第一次调用 `_compiled_callable` 时发生。

### 5.2 为什么叫 NoGuards

对于非 `STOCK_TORCH_COMPILE` 模式，它会设置 `guard_filter_fn`：

```text
如果 evaluate_guards=False：跳过所有 guards
如果 evaluate_guards=True：只保留 shape env 相关 guards
```

位置：`code/vllm/vllm/compilation/wrapper.py:105`

目的：

```text
vLLM 希望启动/warmup 阶段完成编译；
服务请求期间不因为新 batch shape 触发 Dynamo 重新编译。
```

### 5.3 __call__ 如何执行 compiled callable

位置：`code/vllm/vllm/compilation/wrapper.py:171`

大致逻辑：

```text
if 使用 bytecode hook:
    第一次触发 compilation
    后续临时替换 forward.__code__ 到 compiled bytecode
else:
    with _compilation_context():
        return _compiled_callable(...)
```

`_compilation_context()` 会临时提高 Dynamo cache limit。

---

## 6. 第三层：CompilationMode 决定 backend 形态

定义位置：`code/vllm/vllm/config/compilation.py:37`

```text
NONE = 0
STOCK_TORCH_COMPILE = 1
DYNAMO_TRACE_ONCE = 2
VLLM_COMPILE = 3
```

### 6.1 NONE

```text
不应用 torch.compile；
@support_torch_compile 注入的 __call__ 直接 forward；
如果 cudagraph_mode 也为 NONE，就是完全 eager forward。
```

注意：即使不 compile，仍可能有 full `CUDAGraphWrapper`，因为 full cudagraph 和 compilation 在新设计中基本正交。

### 6.2 STOCK_TORCH_COMPILE

`GPUModelRunner.load_model()` 中专门处理：

```text
backend = compilation_config.init_backend(...)
self.model.compile(fullgraph=True, backend=backend)
return
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5273`

这条路径使用 PyTorch 原生 `nn.Module.compile()`，不会继续走后面的 vLLM cudagraph wrapper 包装逻辑。

### 6.3 DYNAMO_TRACE_ONCE

仍然走 `TorchCompileWithNoGuardsWrapper`，但 backend 是普通 torch backend 或自定义 backend，不走 vLLM 的 piecewise backend。

特点：

```text
单次 Dynamo trace；
避免后续 recompile；
不做 vLLM 自定义 piecewise 子图编译。
```

### 6.4 VLLM_COMPILE

这是 V1 默认重点路径。

特点：

```text
使用 vLLM 自定义 VllmBackend；
Dynamo 先捕获模型 forward 的 FX graph；
VllmBackend 再按 splitting_ops 拆成 piecewise graphs；
每个可编译 subgraph 交给 PiecewiseBackend；
PiecewiseBackend 再调用 CompilerManager / Inductor 编译多个 shape range；
必要时给 subgraph 外面包 PIECEWISE CUDAGraphWrapper。
```

---

## 7. VllmBackend：vLLM 自定义 torch.compile backend

定义位置：`code/vllm/vllm/compilation/backends.py:800`

它是 `CompilationMode.VLLM_COMPILE` 下 `torch.compile(..., backend=VllmBackend(...))` 的 backend。

### 7.1 VllmBackend 做什么

当 Dynamo 完成 bytecode transform 后，会调用 backend：

```text
VllmBackend.__call__(graph, example_inputs)
```

位置：`code/vllm/vllm/compilation/backends.py:1014`

核心流程：

```text
1. 计算 compile cache key；
2. 初始化 cache 目录；
3. split_graph(graph, splitting_ops)；
4. 找出需要编译的 submodule；
5. PiecewiseCompileInterpreter 运行 split_gm；
6. 对每个可编译 submodule 创建 PiecewiseBackend；
7. 保存 compile cache；
8. 生成 stitching execution_code；
9. 返回 VllmSerializableFunction(runtime_callable)。
```

### 7.2 split_graph 如何切分

`split_graph()` 按 `splitting_ops` 切分 FX graph。

位置：`code/vllm/vllm/compilation/backends.py:548`

默认 `splitting_ops` 通常是 attention / KV cache update 等 cudagraph 不安全或不希望编译进 piecewise graph 的 op。

切分结果是：

```text
split_gm
piecewise_graphs: list[SplitItem]
```

每个 `SplitItem` 标记：

```text
submod_name
is_splitting_graph
fx.GraphModule
```

其中：

```text
is_splitting_graph=True  → 通常是 attention/custom op 本身，保留 eager 或特殊 op 调用；
is_splitting_graph=False → 两个 splitting op 之间的普通计算，可编译。
```

### 7.3 为什么 attention 仍能出现在 Dynamo full graph 中

`docs/design/torch_compile.md` 里说明：attention 内部很复杂，vLLM 会把它包装成 custom op，例如 `vllm.unified_attention_with_output`，让 Dynamo 从外部看它只是一个 op。

位置：`code/vllm/docs/design/torch_compile.md:167`

这样：

```text
Dynamo 可以捕获完整 forward；
但 VllmBackend 可以在后处理阶段按 attention op 切分；
attention 内部仍由 vLLM runtime / backend 处理。
```

---

## 8. PiecewiseBackend：按 runtime shape 选择已编译子图

定义位置：`code/vllm/vllm/compilation/piecewise_backend.py:86`

### 8.1 它解决什么问题

vLLM 的 batch 动态维通常是 `num_tokens`。

Inductor 可以编译：

```text
通用 symbolic shape graph
特定 compile_sizes 的 static shape graph
若干 compile_ranges 的 graph
```

`PiecewiseBackend` 就是每个可编译 subgraph 的 runtime dispatcher：

```text
根据 runtime_shape 找到对应 RangeEntry；
调用对应 compiled runnable。
```

### 8.2 初始化时提前编译所有 range

构造时：

```text
self.compile_ranges = compilation_config.get_compile_ranges()
self.compile_sizes = compilation_config.compile_sizes
self.compile_all_ranges()
```

位置：`code/vllm/vllm/compilation/piecewise_backend.py:137`

这符合 vLLM 的启动策略：

```text
服务请求不应该触发新的编译；
所有 compilation 在 warmup/profile 阶段完成。
```

### 8.3 runtime __call__ 如何选择 compiled graph

位置：`code/vllm/vllm/compilation/piecewise_backend.py:358`

逻辑：

```text
runtime_shape = args[sym_shape_indices[0]]
range_entry = _find_range_for_shape(runtime_shape)
return range_entry.runnable(*args)
```

`_find_range_for_shape()` 优先找 exact compile size：

```text
如果 runtime_shape in compile_sizes → 用 [size, size] graph
否则 → 找包含 runtime_shape 的 compile_range
```

位置：`code/vllm/vllm/compilation/piecewise_backend.py:343`

### 8.4 CompilerManager 负责 cache / compile / load

`CompilerManager` 定义在 `backends.py`。

位置：`code/vllm/vllm/compilation/backends.py:124`

职责：

```text
管理 cache 文件；
根据 compile_range 和 graph_index 查缓存；
缓存命中直接 load；
未命中调用 Inductor/Eager/custom compiler 编译；
把 handle 写回 vllm_compile_cache.py。
```

---

## 9. PIECEWISE CUDAGraphWrapper 如何包住 compiled 子图

VllmBackend 在创建 `PiecewiseBackend` 后，会调用：

```text
wrap_with_cudagraph_if_needed(...)
```

位置：`code/vllm/vllm/compilation/backends.py:628`

条件：

```text
compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
并且
not compilation_config.use_inductor_graph_partition
```

满足时返回：

```text
CUDAGraphWrapper(
  runnable=piecewise_backend,
  runtime_mode=CUDAGraphMode.PIECEWISE,
  cudagraph_options=...
)
```

位置：`code/vllm/vllm/compilation/backends.py:670`

重要点：

```text
PIECEWISE wrapper 绑定 runtime_mode=PIECEWISE；
无论 FX graph 是 full 还是 piecewise，它只在 ForwardContext 说 PIECEWISE 时生效；
FULL 模式下它会 pass-through，避免和外层 FULL wrapper 冲突。
```

这就是 nested wrapper 设计的核心。

---

## 10. FULL CUDAGraphWrapper 如何包住整个模型

`GPUModelRunner.load_model()` 在非 STOCK_TORCH_COMPILE 路径下处理 full cudagraph。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5287`

逻辑：

```text
cudagraph_mode = self.compilation_config.cudagraph_mode

if breakable enabled and cudagraph_mode != NONE:
    self.model = BreakableCUDAGraphWrapper(self.model, ...)
elif cudagraph_mode.has_full_cudagraphs() and not use_ubatching:
    self.model = CUDAGraphWrapper(self.model, runtime_mode=FULL)
elif use_ubatching:
    self.model = UBatchWrapper(...)
```

### 10.1 FULL wrapper 的位置

FULL wrapper 在最外层：

```text
GPUModelRunner._model_forward()
  → CUDAGraphWrapper(FULL).__call__
      → underlying model.__call__
          → @support_torch_compile __call__
              → compiled callable / VllmBackend runtime callable
                  → PiecewiseBackend / attention / etc.
```

所以 FULL cudagraph capture 的范围是：

```text
整个 model forward，包括 attention、compiled subgraphs、custom ops。
```

前提是本轮 backend/metadata 支持 full cudagraph。

### 10.2 FULL wrapper 与 compilation 正交

`CompilationConfig` 注释明确说明：

```text
piecewise cudagraph requires piecewise compilation；
full cudagraphs are supported with and without compilation。
```

位置：`code/vllm/vllm/config/compilation.py:615`

因此可以有：

```text
不 compile，但 full cudagraph replay 整个 eager forward；
compile 后，full cudagraph replay 整个 compiled forward；
FULL_AND_PIECEWISE 下，decode 用 FULL wrapper，mixed batch 用 PIECEWISE wrappers。
```

---

## 11. CUDAGraphWrapper 的 runtime 分支

定义位置：`code/vllm/vllm/compilation/cuda_graph.py:145`

### 11.1 输入来自 ForwardContext

每次 `__call__()` 读取：

```text
forward_context.cudagraph_runtime_mode
forward_context.batch_descriptor
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:240`

这些字段由 `GPUModelRunner.execute_model()` 的：

```text
set_forward_context(...)
```

写入。

### 11.2 mode 不匹配就 pass-through

```text
if cudagraph_runtime_mode == NONE
or cudagraph_runtime_mode != self.runtime_mode:
    return self.runnable(*args, **kwargs)
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:244`

因此：

```text
FULL wrapper 在 PIECEWISE 模式下不生效；
PIECEWISE wrapper 在 FULL 模式下不生效；
NONE 模式下所有 cudagraph wrapper 都 pass-through。
```

### 11.3 key 首次出现时 capture

如果 mode 匹配、但 entry 中还没有 graph：

```text
validate_cudagraph_capturing_enabled()
with torch.cuda.graph(...):
    output = runnable(...)
entry.output = weak_ref_tensors(output)
entry.cudagraph = cudagraph
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:265`

通常这些 capture 会在 `GPUModelRunner.capture_model()` 的 dummy run 阶段完成。

### 11.4 key 已存在时 replay

```text
entry.cudagraph.replay()
return entry.output
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:357`

debug 模式下还会检查输入 tensor 地址是否和 capture 时一致。

### 11.5 它不管理 static input buffers

`CUDAGraphWrapper` 注释明确说明：

```text
它不保存 persistent input buffers；
也不把 runtime inputs copy 到 static buffers；
这些由外部逻辑保证。
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:161`

这点和一些传统 graph runner 不同。

vLLM 的做法是：

```text
ModelRunner 本身维护 input_ids / positions / buffers；
每轮把真实请求写到这些持久 buffer 中；
wrapper 只负责按 batch_descriptor capture/replay。
```

---

## 12. cudagraph_copy_inputs 是什么

`CompilationConfig.cudagraph_copy_inputs` 定义：

位置：`code/vllm/vllm/config/compilation.py:633`

含义：

```text
只对 PIECEWISE 有效；
如果 caller 不能保证输入 buffer 地址稳定，可以让 compiler 管理 static input buffers；
compiler 会把 runtime tensors copy 到内部 buffer，再调用 compiled graph。
```

VllmBackend 返回 runtime callable 时，如果：

```text
cudagraph_mode != NONE
and cudagraph_copy_inputs=True
```

会构造：

```text
copy_and_call = make_copy_and_call(sym_tensor_indices, static_buffers, runtime_callable)
```

位置：`code/vllm/vllm/compilation/backends.py:1299`

`make_copy_and_call()` 会：

```text
对 symbolic input tensor：
  static_tensor = input_buffer[:runtime_shape]
  static_tensor.copy_(runtime_tensor)
  用 static_tensor 替换参数
再调用 compiled callable
```

位置：`code/vllm/vllm/compilation/backends.py:59`

---

## 13. BreakableCUDAGraphWrapper：不用 torch.compile split 的分段 capture

定义位置：`code/vllm/vllm/compilation/breakable_cudagraph.py:246`

它是 `CUDAGraphWrapper` 的替代方案，通过环境变量开启：

```text
VLLM_USE_BREAKABLE_CUDAGRAPH
```

`load_model()` 中优先级高于普通 FULL wrapper。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5290`

### 13.1 它解决什么问题

普通 piecewise cudagraph 依赖 torch.compile / FX split：

```text
Dynamo 捕获 graph
按 attention/custom op split
对可 capture 子图包 CUDAGraphWrapper
```

Breakable CUDAGraph 的思路是：

```text
不预先用 torch.compile 拆图；
运行时进入一个 stream capture context；
遇到被 @eager_break_during_capture 标记的 op 时结束当前 graph segment；
该 op eager 执行；
然后开始下一个 graph segment；
最终 replay 时按 segments 顺序执行。
```

### 13.2 runtime dispatch 规则

`BreakableCUDAGraphWrapper.__call__()` 和普通 wrapper 一样读 `ForwardContext`：

```text
batch_descriptor
cudagraph_runtime_mode
```

但它不区分 FULL / PIECEWISE 的 wrapper runtime mode：

```text
只要 cudagraph_runtime_mode != NONE，就按 batch_descriptor capture/replay。
```

因为 breakable capture 的 artifact 本身由 graph segments + eager breaks 组成。

---

## 14. UBatchWrapper：microbatching 场景的外层 wrapper

定义位置：`code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:113`

当 `parallel_config.use_ubatching=True` 时，`load_model()` 会包装：

```text
self.model = UBatchWrapper(self.model, vllm_config, runtime_mode, device)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5308`

它的职责是：

```text
把一个 batch 切成多个 ubatch；
每个 ubatch 在线程和独立 stream 上执行；
必要时 capture 整个 ubatch 调度过程；
最后把各 ubatch 输出 cat 回来。
```

如果 runtime mode 不是 `NONE`，内部还会持有一个：

```text
self.cudagraph_wrapper = CUDAGraphWrapper(runnable, runtime_mode=FULL)
```

位置：`code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:132`

注意：ubatch 线程会手动管理 forward context，因此 capture 时会临时：

```text
override_forward_context(None)
```

位置：`code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:247`

---

## 15. ForwardContext 是所有 wrapper 的控制面

`ForwardContext` 定义在：

`code/vllm/vllm/forward_context.py:128`

和 compile / cudagraph wrapper 直接相关的字段：

```text
cudagraph_runtime_mode
batch_descriptor
skip_compiled
attn_metadata
slot_mapping
ubatch_slices
dp_metadata
```

### 15.1 set_forward_context 写入运行态

`GPUModelRunner.execute_model()` 在 forward 前：

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

### 15.2 wrapper 分别读哪些字段

```text
CUDAGraphWrapper：
  cudagraph_runtime_mode
  batch_descriptor

BreakableCUDAGraphWrapper：
  cudagraph_runtime_mode
  batch_descriptor

@support_torch_compile 注入的 __call__：
  skip_compiled

attention/custom ops：
  attn_metadata
  slot_mapping
  dp_metadata

UBatchWrapper：
  ubatch_slices / ubatch context
```

这样一来，`_model_forward()` 不需要显式传 mode 参数，所有下游逻辑都从同一个 context 读取。

---

## 16. 几种典型组合

### 16.1 完全 eager

配置近似：

```text
compilation_mode = NONE
cudagraph_mode = NONE
```

调用链：

```text
_model_forward()
  → 原始 self.model.forward()
```

如果模型类有 `@support_torch_compile`，其 `__call__()` 也会因为 `do_not_compile=True` 直接 forward。

### 16.2 vLLM compile，无 cudagraph

```text
compilation_mode = VLLM_COMPILE
cudagraph_mode = NONE
```

调用链：

```text
_model_forward()
  → model.__call__  # support_torch_compile 注入
      → TorchCompileWithNoGuardsWrapper.__call__
          → torch.compile(..., backend=VllmBackend)
              → VllmBackend runtime callable
                  → PiecewiseBackend compiled subgraphs
                  → splitting ops / attention eager/custom op
```

没有 `CUDAGraphWrapper` 生效。

### 16.3 PIECEWISE cudagraph

```text
compilation_mode = VLLM_COMPILE
cudagraph_mode = PIECEWISE
```

调用链：

```text
_model_forward()
  → model.__call__
      → compiled runtime callable
          → CUDAGraphWrapper(PIECEWISE) around PiecewiseBackend
              → replay/capture compiled subgraph
          → attention/custom op pass-through outside graph
```

特点：

```text
attention 不进入 piecewise cudagraph；
普通 FFN/RMSNorm/Linear 等 compiled subgraph 可以 replay；
更兼容，但不是完整 forward graph。
```

### 16.4 FULL cudagraph，无 piecewise

```text
cudagraph_mode = FULL 或 FULL_DECODE_ONLY
```

调用链：

```text
_model_forward()
  → CUDAGraphWrapper(FULL)
      → model.__call__
          → compiled 或 eager forward
              → attention 也在其中
```

特点：

```text
FULL wrapper 包住整个 model；
要求 attention backend / metadata / batch shape 支持 full capture；
不要求必须 VLLM_COMPILE。
```

### 16.5 FULL_AND_PIECEWISE

```text
compilation_mode = VLLM_COMPILE
cudagraph_mode = FULL_AND_PIECEWISE
```

调用链里同时有：

```text
外层 CUDAGraphWrapper(FULL)
内层 CUDAGraphWrapper(PIECEWISE)
```

运行时：

```text
uniform decode → dispatcher 返回 FULL
  → 外层 FULL wrapper replay
  → 内层 PIECEWISE wrapper pass-through

mixed prefill-decode → dispatcher 返回 PIECEWISE
  → 外层 FULL wrapper pass-through
  → 内层 PIECEWISE wrapper replay

不满足任何 key → dispatcher 返回 NONE
  → 两层 wrapper 都 pass-through
```

这就是 nested wrapper 设计。

---

## 17. capture_model() 和 wrapper 的关系

`GPUModelRunner.capture_model()` 会遍历 dispatcher 中的 capture descriptors：

```text
for runtime_mode, batch_descs in cudagraph_dispatcher.get_capture_descs():
    _capture_cudagraphs(batch_descs, runtime_mode)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6583`

每个 capture 最终都会走：

```text
_dummy_run(..., cudagraph_runtime_mode=runtime_mode, is_graph_capturing=True)
  → set_forward_context(..., cudagraph_runtime_mode, batch_descriptor)
  → self.model(...)
      → 对应 CUDAGraphWrapper 看到 mode 匹配
      → 进入 torch.cuda.graph capture
```

所以：

```text
capture_model() 不直接调用 wrapper.capture()；
它通过 dummy forward + ForwardContext 触发 wrapper 的普通 __call__ 分支。
```

---

## 18. 编译失败 / fallback 如何理解

这里要区分几类 fallback。

### 18.1 cudagraph fallback

如果 runtime mode 是 `NONE`，或 wrapper mode 不匹配：

```text
CUDAGraphWrapper → runnable(*args, **kwargs)
```

这只是不用 CUDA Graph，底层仍可能是 compiled forward。

### 18.2 compiled fallback

如果：

```text
do_not_compile=True
或
ForwardContext.skip_compiled=True
```

模型 `__call__()` 会：

```text
return self.forward(*args, **kwargs)
```

也就是不用 compiled callable。

### 18.3 cache fallback

`CompilerManager` 如果 cache miss：

```text
重新调用 compiler.compile()
```

如果 cache 文件损坏或源码变更，AOT load 失败会 warning，然后重新编译。

### 18.4 不是所有 fallback 都意味着 eager

需要特别注意：

```text
cudagraph fallback 到 NONE ≠ torch.compile 关闭；
compiled fallback 到 forward ≠ attention 没有 custom op；
PIECEWISE pass-through ≠ 外层 FULL 不会生效。
```

最终实际路径要看：

```text
compilation_mode
cudagraph_runtime_mode
wrapper 层级
skip_compiled
backend support
```

---

## 19. 为什么 lm head / logits 通常不在 compiled graph 里

`docs/design/torch_compile.md` 中说明，计算图输入主要是 token / position / weights / buffers，输出是 hidden states；lm head projection 和 sampling 不在该 graph 中。

位置：`code/vllm/docs/design/torch_compile.md:167`

在 `GPUModelRunner.execute_model()` 中，forward 后才做：

```text
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

这意味着本文讨论的 compile wrapper / cudagraph wrapper 主要包住的是：

```text
模型 backbone forward → hidden_states
```

而不是完整的：

```text
forward + logits + sampler + scheduler update
```

---

## 20. 最小伪代码

### 20.1 load_model 包装伪代码

```text
self.model = model_loader.load_model(...)

# 模型类如果有 @support_torch_compile，实例初始化时已经注入 compile 行为

if compilation_mode == STOCK_TORCH_COMPILE:
    backend = compilation_config.init_backend(...)
    self.model.compile(fullgraph=True, backend=backend)
    return

if breakable_cudagraph_enabled and cudagraph_mode != NONE:
    self.model = BreakableCUDAGraphWrapper(self.model)
elif cudagraph_mode.has_full_cudagraphs() and not use_ubatching:
    self.model = CUDAGraphWrapper(self.model, runtime_mode=FULL)
elif use_ubatching:
    self.model = UBatchWrapper(self.model, runtime_mode=FULL or NONE)
```

### 20.2 model.__call__ 伪代码

```text
# support_torch_compile 注入
if do_not_compile or torch.compiler.is_compiling():
    return forward(...)

if get_forward_context().skip_compiled:
    return forward(...)

if aot_compiled_fn exists:
    return aot_compiled_fn(...)

if compiled:
    return compiled_callable(...)

mark_dynamic_inputs(...)
compiled_callable = torch.compile(forward, backend=VllmBackend or other backend)
output = compiled_callable(...)
compiled = True
return output
```

### 20.3 VllmBackend 伪代码

```text
VllmBackend.__call__(fx_graph, example_inputs):
    split_gm, pieces = split_graph(fx_graph, splitting_ops)

    for each non-splitting submodule:
        backend = PiecewiseBackend(submodule)
        if piecewise cudagraph enabled:
            backend = CUDAGraphWrapper(backend, PIECEWISE)
        replace split_gm.submodule with backend

    runtime_callable = compile_execution_fn(split_gm)
    return runtime_callable
```

### 20.4 CUDAGraphWrapper 伪代码

```text
ctx = get_forward_context()

if ctx.cudagraph_runtime_mode == NONE:
    return runnable(...)

if ctx.cudagraph_runtime_mode != self.runtime_mode:
    return runnable(...)

entry = entries[ctx.batch_descriptor]

if entry.cudagraph is None:
    with torch.cuda.graph(...):
        output = runnable(...)
    save entry
    return output

entry.cudagraph.replay()
return entry.output
```

---

## 21. 一句话总结

```text
vLLM 把模型执行拆成两层稳定接口：ModelRunner 永远调用 self.model(...)，而模型对象和编译子图在加载/首次 forward 时被 @support_torch_compile、VllmBackend、PiecewiseBackend、CUDAGraphWrapper 等多层包装；运行时再由 ForwardContext 控制这些 wrapper 是 replay、compiled call、pass-through 还是原始 eager forward。
```
