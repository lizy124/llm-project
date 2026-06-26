# 10. Attention 与 CUDA graph / torch.compile 的关系

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\config\compilation.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\vllm.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\forward_context.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\cuda_graph.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\breakable_cudagraph.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\piecewise_backend.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\cudagraph_dispatcher.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backend.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\attention.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\layers\attention\kv_transfer_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\kv_connector_model_runner_mixin.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\base.py`

本文用于梳理 CUDA graph、torch.compile、piecewise compilation 如何影响 attention 执行路径，以及 attention metadata、batch padding、KV cache update、KV connector layer hook、dynamic metadata 对图捕获的约束。前几篇已经分别讲了 metadata builder、attention forward、KV connector；本文把它们放到 “compile / graph capture / replay” 这一层统一看。

---

## 1. 本文要回答的问题

```text
CUDA graph 和 torch.compile 在 vLLM V1 中分别解决什么问题？
FULL / PIECEWISE / NONE cudagraph runtime mode 有什么差异？
FULL_DECODE_ONLY / FULL_AND_PIECEWISE 为什么存在？
Attention backend 的 CUDA graph 支持如何约束全局 cudagraph mode？
ModelRunner 如何决定 cudagraph_mode、padding、BatchDescriptor、ubatching？
attention metadata 的动态性如何影响 CUDA graph capture？
set_forward_context 如何把 graph 信息传给模型 forward？
CUDAGraphWrapper / BreakableCUDAGraphWrapper 如何 capture / replay？
piecewise compile 和 attention custom op 边界有什么关系？
KV connector 的 layer hook 为什么可能要求 piecewise / eager break？
cudagraph stats 如何回到 ModelRunnerOutput？
```

---

## 2. 一句话回答

CUDA graph / torch.compile 不改变 attention 的语义主链路，但会改变 “模型 forward 被如何包装、哪些 shape 可以复用、哪些 op 必须成为图边界”。

最小链路是：

```text
CompilationConfig
  → cudagraph_mode / compile mode / splitting_ops / capture sizes
  → attention backend 声明 AttentionCGSupport
  → GPUModelRunner._check_and_update_cudagraph_mode()
  → CudagraphDispatcher.initialize_cudagraph_keys(...)

每轮执行：
  SchedulerOutput
    → _prepare_inputs()
    → _determine_batch_execution_and_padding()
        → cudagraph_runtime_mode + BatchDescriptor + padding
    → _build_attention_metadata(... padded shape ...)
    → set_forward_context(cudagraph_runtime_mode, batch_descriptor, ...)
    → compiled model / CUDAGraphWrapper / BreakableCUDAGraphWrapper
    → Attention.forward()
    → unified_attention_with_output()
    → backend impl.forward(...)
```

如果只记住一句话：

```text
CUDA graph 依赖稳定的 batch descriptor 和 tensor address，torch.compile 依赖稳定的 graph boundary；attention metadata、KV cache update、KV connector hook 这些动态信息必须通过 padding、custom op、piecewise split 或 eager break 被稳定表达。
```

---

## 3. 先区分 compile 和 CUDA graph

### 3.1 torch.compile 解决什么

`torch.compile` / vLLM compile 主要解决：

```text
Python 调度开销；
图级优化；
Inductor 编译；
custom op / fusion / piecewise graph split；
静态 shape specialization；
编译缓存。
```

vLLM 的编译模式定义在：`code/vllm/vllm/config/compilation.py:37`

```text
CompilationMode.NONE
  完全 eager。

STOCK_TORCH_COMPILE
  标准 torch.compile。

DYNAMO_TRACE_ONCE
  单次 Dynamo trace。

VLLM_COMPILE
  vLLM 自定义 Inductor backend，支持 caching、piecewise compilation、shape specialization、custom passes。
```

位置：`code/vllm/vllm/config/compilation.py:37` 到 `code/vllm/vllm/config/compilation.py:50`

### 3.2 CUDA graph 解决什么

CUDA graph 主要解决：

```text
重复执行同一 shape / 同一地址的 GPU kernel launch overhead。
```

它的限制也更强：

```text
输入输出 tensor 地址要稳定；
kernel launch 序列要稳定；
shape 要匹配 capture key；
不能把不适合 replay 的 host-side side effect 录进图；
动态 metadata / KV transfer 需要小心处理。
```

`CUDAGraphWrapper` 的注释把流程总结成：

```text
1. wrapper 初始化时指定 runtime mode：FULL 或 PIECEWISE；
2. runtime 从 forward context 读取 cudagraph_runtime_mode 和 batch_descriptor；
3. 如果 mode 不匹配或为 NONE，则直接 eager 调用；
4. 如果 mode 匹配，第一次遇到 batch_descriptor 时 capture，之后 replay。
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:145` 到 `code/vllm/vllm/compilation/cuda_graph.py:160`

### 3.3 两者的关系

可以理解为：

```text
torch.compile 决定“模型图怎么被切分和编译”；
CUDA graph 决定“某个已稳定的执行片段如何 capture / replay”；
piecewise compilation 决定“attention / KV update 这类动态边界是否能成为可 graph 的片段”。
```

所以它们不是二选一，而是叠加关系。

---

## 4. CUDAGraphMode 有哪些模式

定义位置：`code/vllm/vllm/config/compilation.py:53`

```text
NONE = 0
PIECEWISE = 1
FULL = 2
FULL_DECODE_ONLY = (FULL, NONE)
FULL_AND_PIECEWISE = (FULL, PIECEWISE)
```

位置：`code/vllm/vllm/config/compilation.py:53` 到 `code/vllm/vllm/config/compilation.py:64`

### 4.1 runtime mode 只有三种

真正运行时有效的 runtime mode 是：

```text
NONE
PIECEWISE
FULL
```

位置：`code/vllm/vllm/config/compilation.py:92` 到 `code/vllm/vllm/config/compilation.py:97`

### 4.2 FULL

`FULL` 表示尽量把整个模型 forward 作为一个完整 CUDA graph capture / replay 单元。

特点：

```text
要求最强；
shape 和控制流最需要稳定；
attention backend 必须支持对应 batch 形态；
不适合 cascade attention / 某些 encoder output / 某些动态 hooks。
```

### 4.3 PIECEWISE

`PIECEWISE` 表示把模型 forward 切成多个片段，每个片段单独 compile / capture。

特点：

```text
要求相对弱；
attention custom op 可以作为 split boundary；
KV connector layer hook 可以在 eager break 中执行；
更适合 attention backend 不支持 full graph 但可在 piecewise 边界运行的场景。
```

### 4.4 FULL_DECODE_ONLY

`FULL_DECODE_ONLY = (FULL, NONE)` 表示：

```text
uniform decode batch 可以走 FULL；
mixed prefill/decode batch 不走 graph。
```

它存在的原因是很多 backend 只支持 decode-only full graph，不支持 mixed batch full graph。

### 4.5 FULL_AND_PIECEWISE

`FULL_AND_PIECEWISE = (FULL, PIECEWISE)` 表示：

```text
uniform decode batch 走 FULL；
mixed batch 走 PIECEWISE。
```

这是一个折中模式：

```text
decode 热路径尽量 full graph；
prefill / mixed 动态路径退到 piecewise。
```

---

## 5. AttentionCGSupport：backend 如何声明 CUDA graph 能力

定义位置：`code/vllm/vllm/v1/attention/backend.py:516`

```text
ALWAYS = 3
  Cudagraph always supported; supports mixed-prefill-decode。

UNIFORM_BATCH = 2
  只支持 query length 一致的 batch，可用于 spec decode。

UNIFORM_SINGLE_TOKEN_DECODE = 1
  只支持 query_len == 1 的 uniform decode。

NEVER = 0
  不支持 CUDA graph。
```

位置：`code/vllm/vllm/v1/attention/backend.py:516` 到 `code/vllm/vllm/v1/attention/backend.py:530`

每个 `AttentionMetadataBuilder` 有：

```text
_cudagraph_support
get_cudagraph_support(vllm_config, kv_cache_spec)
```

位置：`code/vllm/vllm/v1/attention/backend.py:533` 到 `code/vllm/vllm/v1/attention/backend.py:565`

这说明：

```text
CUDA graph 支持能力不是由 Attention impl 单独决定，而是由 metadata builder 声明。
```

原因是 graph 能不能捕获，不只看 kernel，还要看 metadata build 后的执行计划是否稳定。

---

## 6. GPUModelRunner 如何收集 backend CUDA graph 能力

入口：`GPUModelRunner._check_and_update_cudagraph_mode()`

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6877`

它做三件事。

### 6.1 遍历所有 attention backend

```text
for attn_backend in attention_backends:
    builder_cls = attn_backend.get_builder_cls()
    cg_support = builder_cls.get_cudagraph_support(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6892` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:6900`

### 6.2 取最弱支持等级

```text
min_cg_support
min_cg_attn_backend
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6889` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:6903`

这意味着：

```text
一个模型有多个 attention group 时，全局 cudagraph mode 受最弱 backend 约束。
```

### 6.3 交给 CompilationConfig resolve

```text
self.compilation_config.resolve_cudagraph_mode_and_sizes(
    min_cg_support,
    min_cg_attn_backend,
    uniform_decode_query_len,
    tensor_parallel_size,
    kv_cache_config,
    max_num_reqs,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6904` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:6912`

resolve 后再初始化 dispatcher keys：

```text
self.cudagraph_dispatcher.initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6913` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:6917`

---

## 7. resolve_cudagraph_mode_and_sizes() 如何降级

入口：`code/vllm/vllm/config/compilation.py:1316`

它根据：

```text
用户配置的 cudagraph_mode；
最弱 attention backend 的 AttentionCGSupport；
spec decode uniform_decode_query_len；
是否有 splitting_ops；
是否是 VLLM_COMPILE；
Mamba cache blocks；
sequence parallelism；
```

决定最终模式。

### 7.1 mixed batch full graph 不被支持时

如果用户希望 mixed batch 走 FULL，但最弱 backend 不是 `ALWAYS`：

```text
cudagraph_mode.mixed_mode() == FULL
and min_cg_support != ALWAYS
```

位置：`code/vllm/vllm/config/compilation.py:1333` 到 `code/vllm/vllm/config/compilation.py:1358`

处理逻辑：

```text
如果 min_cg_support == NEVER：直接报错，建议 PIECEWISE；
如果 splitting_ops 包含 attention：降为 FULL_AND_PIECEWISE；
否则降为 FULL_DECODE_ONLY。
```

### 7.2 decode full graph 也不支持时

如果 decode FULL 也不支持：

```text
cudagraph_mode.decode_mode() == FULL
and min_cg_support == NEVER
```

位置：`code/vllm/vllm/config/compilation.py:1360` 到 `code/vllm/vllm/config/compilation.py:1385`

处理逻辑：

```text
如果 VLLM_COMPILE 且 attention 被 piecewise 编译：降为 PIECEWISE；
否则降为 NONE。
```

### 7.3 spec decode 的特殊限制

spec decode 下：

```text
uniform_decode_query_len = 1 + num_speculative_tokens
```

如果 decode FULL 开启，且 `uniform_decode_query_len > 1`，backend 至少需要：

```text
AttentionCGSupport.UNIFORM_BATCH
```

否则：

```text
有 splitting_ops：降为 PIECEWISE；
否则：降为 NONE。
```

位置：`code/vllm/vllm/config/compilation.py:1387` 到 `code/vllm/vllm/config/compilation.py:1405`

### 7.4 spec decode capture sizes 要对齐

如果 decode FULL + spec decode，capture sizes 会被调整到 `uniform_decode_query_len` 的倍数。

位置：`code/vllm/vllm/config/compilation.py:1421` 到 `code/vllm/vllm/config/compilation.py:1433`

函数：

```text
adjust_cudagraph_sizes_for_spec_decode(...)
```

位置：`code/vllm/vllm/config/compilation.py:1462`

如果开启 sequence parallel，还要同时满足 TP size 对齐。

### 7.5 Mamba cache blocks 限制

Mamba 模型使用 FULL decode cudagraph 时，每个 decode sequence 需要一个 Mamba cache block。

如果：

```text
max_num_reqs > kv_cache_config.num_blocks
```

会报错，要求降低 `max_num_seqs` 或增加 `gpu_memory_utilization`。

位置：`code/vllm/vllm/config/compilation.py:1435` 到 `code/vllm/vllm/config/compilation.py:1457`

---

## 8. CudagraphDispatcher 如何生成 runtime mode 和 BatchDescriptor

定义位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:15`

它是运行时 graph 选择的核心。

源码注释说明：

```text
dispatcher 保存两组 dispatch keys：PIECEWISE 和 FULL；
这些 key 在 attention support 和 CompilationConfig resolve 后初始化；
runtime 根据输入条件生成 cudagraph_runtime_mode 和 BatchDescriptor；
CUDAGraphWrapper 会信任 forward context 中的 dispatch key。
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:15` 到 `code/vllm/vllm/v1/cudagraph_dispatcher.py:31`

### 8.1 cudagraph keys 何时初始化

初始化入口：

```text
initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:166`

如果 cudagraph disabled：

```text
keys_initialized = True
return
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:173` 到 `code/vllm/vllm/v1/cudagraph_dispatcher.py:176`

否则会：

```text
1. 预计算 batch size → padded graph size；
2. 生成 LoRA capture cases；
3. 为 mixed mode 添加 PIECEWISE / FULL keys；
4. 为 decode FULL 添加 uniform decode keys。
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:178` 到 `code/vllm/vllm/v1/cudagraph_dispatcher.py:233`

### 8.2 BatchDescriptor 包含什么

`BatchDescriptor` 定义在：`code/vllm/vllm/forward_context.py` 附近，由 `gpu_model_runner.py` 导入使用。

它最关键的字段在 dispatcher 中体现：

```text
num_tokens
num_reqs
uniform
has_lora
num_active_loras
```

`_create_padded_batch_descriptor()` 中会构造：

```text
BatchDescriptor(
    num_tokens=num_tokens_padded,
    num_reqs=num_reqs,
    uniform=uniform_decode,
    has_lora=has_lora,
    num_active_loras=num_active_loras,
)
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:132` 到 `code/vllm/vllm/v1/cudagraph_dispatcher.py:156`

### 8.3 PIECEWISE key 会放松 num_reqs / uniform

对 mixed mode 的 PIECEWISE：

```text
batch_desc = replace(batch_desc, num_reqs=None, uniform=False)
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:199` 到 `code/vllm/vllm/v1/cudagraph_dispatcher.py:203`

含义是：

```text
piecewise cudagraph 更关注 token shape；
不要求 request 数和 uniform decode 形态完全固定。
```

### 8.4 dispatch() 的运行时决策

入口：`code/vllm/vllm/v1/cudagraph_dispatcher.py:235`

如果未初始化、关闭、超过最大 capture size 或只允许 NONE：

```text
return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:274` 到 `code/vllm/vllm/v1/cudagraph_dispatcher.py:281`

否则：

```text
1. 根据 LoRA 状态归一化 num_active_loras；
2. 创建 padded BatchDescriptor；
3. 优先匹配 FULL key；
4. 再匹配 PIECEWISE relaxed key；
5. 都不匹配则 NONE。
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:283` 到 `code/vllm/vllm/v1/cudagraph_dispatcher.py:324`

---

## 9. _determine_batch_execution_and_padding() 每轮做什么

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`

它发生在：

```text
_prepare_inputs()
  → _determine_batch_execution_and_padding()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → set_forward_context(...)
```

它返回：

```text
cudagraph_mode
batch_descriptor
should_ubatch
num_tokens_across_dp
cudagraph_stats
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3825` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3831`

### 9.1 判断是否 uniform decode

```text
uniform_decode = _is_uniform_decode(
    max_num_scheduled_tokens,
    uniform_decode_query_len,
    num_tokens,
    num_reqs,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3832` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3838`

普通 decode 时：

```text
uniform_decode_query_len = 1
```

spec decode 时：

```text
uniform_decode_query_len = 1 + num_spec_tokens
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:816`

### 9.2 encoder output / cascade 会禁用 FULL

dispatch 时：

```text
disable_full = use_cascade_attn or has_encoder_output
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3865` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3867`

也就是说：

```text
cascade attention 当前不支持 full cudagraph；
encoder-decoder 中带 encoder output 的 step 也禁用 full cudagraph。
```

### 9.3 sequence parallelism padding

在 dispatch 前先：

```text
num_tokens_padded = _pad_for_sequence_parallelism(num_tokens)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3853`

如果启用 SP，还要求：

```text
batch_descriptor.num_tokens % tensor_parallel_size == 0
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3869` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3877`

### 9.4 data parallel 需要跨 rank 协调

如果 data parallel size > 1，会调用：

```text
coordinate_batch_across_dp(...)
```

并根据 DP 同步后的 token 数重新 dispatch。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3879` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3905`

这保证不同 DP rank 的 graph shape / mode 对齐。

### 9.5 cudagraph_stats

如果启用：

```text
observability_config.cudagraph_metrics
```

会构造：

```text
CUDAGraphStat(
  num_unpadded_tokens=num_tokens,
  num_padded_tokens=batch_descriptor.num_tokens,
  num_paddings=batch_descriptor.num_tokens - num_tokens,
  runtime_mode=str(cudagraph_mode),
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3914`

---

## 10. FULL graph 下为什么需要 padding

CUDA graph replay 要求 shape / address / launch pattern 稳定。

所以运行时真实 token 数：

```text
num_tokens_unpadded
```

会被 dispatcher pad 到某个 capture size：

```text
batch_descriptor.num_tokens
```

然后后续所有 attention 相关张量都要按 padded shape 准备。

在 attention metadata 中表现为：

```text
num_tokens_padded
num_reqs_padded
query_start_loc padded rows
block_table padded rows
slot_mapping padded entries
seq_lens padded rows
```

`_build_attention_metadata()` 中：

```text
num_tokens_padded = num_tokens_padded or num_tokens
num_reqs_padded = num_reqs_padded or num_reqs
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2231` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2233`

常见安全填充：

```text
block_table padded rows 填 NULL_BLOCK_ID；
is_prefilling padded rows 置 False；
slot_mapping padded token 填 -1 / PAD_SLOT_ID；
query_start_loc 保持非递减。
```

后端在 forward 中通常依赖：

```text
attn_metadata.num_actual_tokens
```

裁掉真实 token 区域，避免把 padding 当成真实 token。

---

## 11. build_for_cudagraph_capture() 的作用

`AttentionMetadataBuilder` 提供：

```text
build_for_cudagraph_capture(common_attn_metadata)
```

位置：`code/vllm/vllm/v1/attention/backend.py:634`

默认实现只是：

```text
return self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
```

位置：`code/vllm/vllm/v1/attention/backend.py:634` 到 `code/vllm/vllm/v1/attention/backend.py:644`

为什么需要这个接口？

```text
普通 build 可能做动态 plan、动态 mask、CPU/GPU sync；
CUDA graph capture 希望 metadata shape 更固定；
某些 backend 可以 override 这个方法，构造更 graph-safe 的 metadata。
```

ModelRunner 中如果是 capture 路径，会调用：

```text
builder.build_for_cudagraph_capture(common_attn_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2417` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2420`

普通路径则调用：

```text
builder.build(common_prefix_len, common_attn_metadata, ...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2431` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2435`

---

## 12. set_forward_context 如何传递 graph 信息

`ForwardContext` 定义在：`code/vllm/vllm/forward_context.py:128`

和 CUDA graph 相关的字段：

```text
cudagraph_runtime_mode: CUDAGraphMode
batch_descriptor: BatchDescriptor | None
ubatch_slices: UBatchSlices | None
skip_compiled: bool
```

位置：`code/vllm/vllm/forward_context.py:141` 到 `code/vllm/vllm/forward_context.py:151`

`set_forward_context()` 入口：`code/vllm/vllm/forward_context.py:249`

当：

```text
cudagraph_runtime_mode != NONE
and num_tokens is not None
```

会自动补：

```text
batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
```

如果调用方已经传入更完整的 descriptor，就使用调用方的。

位置：`code/vllm/vllm/forward_context.py:292` 到 `code/vllm/vllm/forward_context.py:297`

然后它把这些值放进 `ForwardContext`：

```text
cudagraph_runtime_mode
batch_descriptor
attn_metadata
slot_mapping
skip_compiled
```

位置：`code/vllm/vllm/forward_context.py:204` 到 `code/vllm/vllm/forward_context.py:231`

这就是 CUDAGraphWrapper 和 Attention.forward 能感知当前 graph 状态的通道。

---

## 13. GPUModelRunner.execute_model 中的 forward context

在模型 forward 前，ModelRunner 会调用：

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

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4303` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4313`

几个关键点：

```text
num_tokens 使用 padded token 数；
cudagraph_mode 是本轮 runtime mode，不一定等于用户配置；
batch_desc 是 dispatcher 归一化后的 capture/replay key；
slot_mapping 也已经按 padded / ubatch 情况处理；
has_encoder_input 时 skip_compiled=True，绕过 compiled model call。
```

`set_forward_context()` 外面还包了：

```text
maybe_get_kv_connector_output(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4315` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4318`

所以 KV connector 的 lifecycle 也是在 forward context 作用域内执行。

---

## 14. CUDAGraphWrapper 如何 capture / replay

入口：`code/vllm/vllm/compilation/cuda_graph.py:145`

### 14.1 运行时 dispatch

`CUDAGraphWrapper.__call__()` 会读取：

```text
forward_context.batch_descriptor
forward_context.cudagraph_runtime_mode
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:233` 到 `code/vllm/vllm/compilation/cuda_graph.py:243`

如果：

```text
没有 forward context；
runtime mode 是 NONE；
runtime mode 和 wrapper 自己的 mode 不匹配；
```

则直接调用原始 runnable。

位置：`code/vllm/vllm/compilation/cuda_graph.py:233` 到 `code/vllm/vllm/compilation/cuda_graph.py:254`

### 14.2 第一次遇到 key：capture

如果该 `BatchDescriptor` 没有 entry，就创建 entry。

如果 entry 还没有 cudagraph，就：

```text
validate_cudagraph_capturing_enabled()
记录输入 tensor addresses
创建 torch.cuda.CUDAGraph
with torch.cuda.graph(...):
    output = runnable(*args, **kwargs)
保存 output weak ref
保存 cudagraph
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:256` 到 `code/vllm/vllm/compilation/cuda_graph.py:344`

### 14.3 后续相同 key：replay

如果 entry 已有 cudagraph：

```text
DEBUG 下检查输入地址是否一致；
同步 offloader；
entry.cudagraph.replay();
return entry.output
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:346` 到 `code/vllm/vllm/compilation/cuda_graph.py:361`

关键限制：

```text
CUDA graph replay 不重新追踪 Python；
它假设输入地址和 capture 时一致；
BatchDescriptor 是选择 graph 的 key。
```

---

## 15. BreakableCUDAGraphWrapper 和 eager break

Breakable CUDA graph 是为了处理 “图中间需要 eager side effect” 的情况。

典型 side effect：

```text
KV connector.wait_for_layer_load(layer_name)
KV connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

这些操作不能简单录入 CUDA graph replay，否则可能 replay 时 hang 或重复执行错误的 host-side 行为。

### 15.1 eager_break_during_capture

定义位置：`code/vllm/vllm/compilation/breakable_cudagraph.py:59`

装饰器语义：

```text
正常情况下直接运行函数；
如果处于 BreakableCUDAGraphCapture 中：
  结束当前 graph segment；
  eager 运行这个函数；
  记录 replay callable；
  开启新 graph segment。
```

位置：`code/vllm/vllm/compilation/breakable_cudagraph.py:59` 到 `code/vllm/vllm/compilation/breakable_cudagraph.py:115`

### 15.2 装饰器顺序为什么重要

源码注释明确要求：

```python
@eager_break_during_capture   # outermost
@maybe_transfer_kv_layer
def unified_attention_with_output(...):
    ...
```

位置：`code/vllm/vllm/compilation/breakable_cudagraph.py:74` 到 `code/vllm/vllm/compilation/breakable_cudagraph.py:89`

原因是：

```text
maybe_transfer_kv_layer 引入 host-side side effect；
这些 side effect 必须在 eager segment 中运行；
不能被录进 cudagraph。
```

实际 attention custom op 也是这样写的：

```text
@eager_break_during_capture
@maybe_transfer_kv_layer
def unified_attention_with_output(...)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:734` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:736`

### 15.3 BreakableCUDAGraphCapture 的执行方式

`BreakableCUDAGraphCapture.add_eager()` 会：

```text
_end_segment()
result = fn()
segments.append(fn)
_begin_segment()
```

位置：`code/vllm/vllm/compilation/breakable_cudagraph.py:195` 到 `code/vllm/vllm/compilation/breakable_cudagraph.py:208`

replay 时：

```text
for r in segments:
    r()
```

位置：`code/vllm/vllm/compilation/breakable_cudagraph.py:212` 到 `code/vllm/vllm/compilation/breakable_cudagraph.py:214`

也就是说：

```text
breakable graph = graph segment + eager segment + graph segment + ...
```

---

## 16. unified_attention_with_output 是 attention 的 compile / graph 边界

定义位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:734`

它是一个 custom op，并注册为：

```text
op_name="unified_attention_with_output"
mutates_args=["output", "output_block_scale"]
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:779` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:784`

它内部做：

```text
get_attention_context(layer_name)
  → attn_metadata
  → self / Attention layer
  → kv_cache

self.impl.forward(
  query, key, value, kv_cache, attn_metadata, output=output, ...
)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:750` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:763`

它成为边界的原因：

```text
1. attention backend 内部高度动态；
2. attn_metadata 是 Python / dataclass / tensor 混合结构；
3. KV cache update / connector hook 可能有 side effect；
4. output 是外部传入并原地写入，方便 custom op 和 cudagraph replay。
```

---

## 17. unified_kv_cache_update 和 ordering dependency

有些 backend：

```text
forward_includes_kv_cache_update = False
```

这时 `Attention.forward()` 会先调用：

```text
unified_kv_cache_update(key, value, layer_name)
```

然后再调用：

```text
unified_attention_with_output(..., kv_cache_dummy_dep=...)
```

`kv_cache_dummy_dep` 的注释说明：

```text
它不被使用，但作为数据依赖，确保 torch.compile 保持 KV cache update 在 attention forward 之前。
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:746` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:748`

`unified_kv_cache_update` 也是 custom op：

```text
op_name="unified_kv_cache_update"
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:726` 到 `code/vllm/vllm/model_executor/layers/attention/attention.py:731`

编译配置里也会把 KV cache update op 纳入 splitting 判断：

```text
vllm::unified_kv_cache_update
vllm::unified_mla_kv_cache_update
```

位置：`code/vllm/vllm/config/compilation.py:1229` 到 `code/vllm/vllm/config/compilation.py:1250`

---

## 18. piecewise compile 与 splitting_ops

`CompilationConfig` 里有：

```text
splitting_ops
```

它表示哪些 op 用作 graph split boundary。

### 18.1 attention ops 是否被 piecewise 编译

判断函数：

```text
is_attention_compiled_piecewise()
```

位置：`code/vllm/vllm/config/compilation.py:1252`

逻辑：

```text
如果 splitting_ops 不包含 attention ops → False；
如果不是 Inductor partition，则要求 CompilationMode.VLLM_COMPILE；
如果是 Inductor partition，则要求 backend == "inductor" 且 mode != NONE。
```

位置：`code/vllm/vllm/config/compilation.py:1252` 到 `code/vllm/vllm/config/compilation.py:1261`

### 18.2 空 splitting_ops 会影响 PIECEWISE

如果 piecewise cudagraph 配置了，但 splitting_ops 为空：

```text
PIECEWISE → NONE
FULL_AND_PIECEWISE → FULL
```

位置：`code/vllm/vllm/config/compilation.py:1140` 到 `code/vllm/vllm/config/compilation.py:1165`

含义是：

```text
没有 split boundary，就没有真正的 piecewise graph。
```

### 18.3 PiecewiseBackend 做什么

`PiecewiseBackend` 定义在：`code/vllm/vllm/compilation/piecewise_backend.py:86`

它处理：

```text
1. 拿到某个 FX subgraph；
2. 根据 compile ranges / compile sizes 编译多个 shape range；
3. runtime 根据输入 shape dispatch 到对应 compiled runnable；
4. 支持从 cache / AOT artifacts 加载预编译 callable。
```

位置：`code/vllm/vllm/compilation/piecewise_backend.py:86` 到 `code/vllm/vllm/compilation/piecewise_backend.py:192`

这和 CUDA graph 的关系是：

```text
piecewise compile 先把模型拆成可编译片段；
piecewise cudagraph 再对这些片段按 BatchDescriptor capture / replay。
```

---

## 19. KV connector 为什么影响 CUDA graph / compile

KV connector 有两层 hook。

### 19.1 forward 外层 connector context

`KVConnectorModelRunnerMixin._get_kv_connector_output()` 必须在 active forward context 内使用。

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:74`

它会：

```text
kv_connector.bind_connector_metadata(...)
kv_connector.start_load_kv(get_forward_context())
...
kv_connector.wait_for_save()
kv_connector.get_finished(...)
kv_connector.get_block_ids_with_load_errors()
kv_connector.clear_connector_metadata()
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:85` 到 `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:112`

这要求 forward context 中有：

```text
attn_metadata
slot_mapping
cudagraph_runtime_mode
batch_descriptor
```

因为 connector 可能要根据当前 layer / metadata load 或 save KV。

### 19.2 attention layer 内层 hook

`maybe_transfer_kv_layer` 包住 `unified_attention_with_output()`。

位置：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:15`

它在 attention 前后执行：

```text
connector.wait_for_layer_load(layer_name)
result = attention(...)
connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

位置：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:37` 到 `code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:59`

这些是 host-side / communication side effect，不适合直接录进 monolithic CUDA graph。

### 19.3 requires_piecewise_for_cudagraph

在 `VllmConfig` 初始化中，会检查 connector class：

```text
connector_cls.requires_piecewise_for_cudagraph(extra_config)
```

位置：`code/vllm/vllm/config/vllm.py:1280` 到 `code/vllm/vllm/config/vllm.py:1286`

例如 LMCache connector 有：

```text
requires_piecewise_for_cudagraph(...)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:74`

MultiConnector 会在任一 child connector 需要时返回 True。

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:139` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:151`

直观理解：

```text
如果 connector 的 layerwise load/save 必须在 attention 边界外以 eager 方式执行，
就需要 piecewise / breakable graph，而不能简单 full graph replay。
```

---

## 20. ModelRunner 何时包装 CUDAGraphWrapper

`GPUModelRunner` 在模型 load / compile 后，会根据最终 `cudagraph_mode` 包装模型。

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5291`

逻辑大致是：

```text
如果 breakable cudagraph enabled 且 cudagraph_mode != NONE 且没有 ubatching：
  self.model = BreakableCUDAGraphWrapper(self.model, vllm_config)

否则如果 cudagraph_mode.has_full_cudagraphs() 且没有 ubatching：
  self.model = CUDAGraphWrapper(self.model, vllm_config, runtime_mode=FULL)

如果 use_ubatching：
  使用 gpu_ubatch_wrapper 中的 CUDAGraphWrapper。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5291` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5309`

这说明：

```text
是否包装模型，不只看用户配置；
还看 resolve 后的 cudagraph_mode、是否 breakable、是否 ubatching。
```

---

## 21. UBatch / microbatch 和 CUDA graph

如果启用 ubatching，runner 不直接用普通模型 wrapper，而是走 ubatch wrapper。

`gpu_ubatch_wrapper.py` 中：

```text
if runtime_mode is not NONE:
    self.cudagraph_wrapper = CUDAGraphWrapper(runnable, vllm_config, runtime_mode=runtime_mode)
```

位置：`code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:130` 到 `code/vllm/vllm/v1/worker/gpu_ubatch_wrapper.py:136`

对 attention metadata：

```text
ubatch_slices 会让 attn_metadata / slot_mapping 变成 list[dict[layer_name, ...]]；
set_forward_context 会把 ubatch_slices 放入 ForwardContext；
每个 ubatch 只看到自己的 token slice。
```

这点和前文 `06_attention_forward_flow.md` 中 UBatch 部分衔接。

---

## 22. attention metadata 动态性如何被 graph-safe 表达

attention metadata 很动态：

```text
query_start_loc
seq_lens
block_table
slot_mapping
positions
max_query_len
max_seq_len
prefill/decode split
spec decode logits indices
cascade prefix
encoder seq lens
```

CUDA graph 不能要求这些业务值完全不变，但要求：

```text
shape 稳定；
tensor address 稳定；
control-flow / launch pattern 稳定；
padded 区域安全；
backend 能用 tensor 内容表达动态信息，而不是捕获时固定 Python 分支。
```

vLLM 的做法是：

```text
1. 用 BatchDescriptor 选择 capture shape；
2. 用 padding 把 token / request 数补到 capture size；
3. 用 CommonAttentionMetadata 保存 padded tensor；
4. 用 num_actual_tokens 区分真实 token 和 padded token；
5. builder 可用 build_for_cudagraph_capture 构造 graph-safe metadata；
6. attention custom op 作为 compile / eager break 边界；
7. backend 自己声明能支持哪类 graph batch。
```

---

## 23. 为什么 cascade attention 会禁用 FULL

在 `_determine_batch_execution_and_padding()` 中：

```text
dispatch_cudagraph(..., disable_full=use_cascade_attn or has_encoder_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3865` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3867`

原因是 cascade attention 的 metadata 和执行计划依赖：

```text
common_prefix_len
prefix / suffix split
shared prefix kernel
per-request suffix kernel
```

这些通常比普通 decode/prefill 更动态。

同时 `AttentionCGSupport` 注释中也说：

```text
Here we do not consider the cascade attention, as currently it is never cudagraph supported.
```

位置：`code/vllm/vllm/v1/attention/backend.py:516` 到 `code/vllm/vllm/v1/attention/backend.py:519`

所以：

```text
cascade attention 可以继续正常执行；
但本轮不会走 FULL cudagraph。
```

---

## 24. cudagraph stats 如何回到输出

`_determine_batch_execution_and_padding()` 会生成 `CUDAGraphStat`。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3914`

字段定义：

```text
num_unpadded_tokens
num_padded_tokens
num_paddings
runtime_mode
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:32` 到 `code/vllm/vllm/compilation/cuda_graph.py:37`

生成类模型中，`cudagraph_stats` 会被保存进 `ExecuteModelState`，后续 sample_tokens 构造 `ModelRunnerOutput` 时带回。

`ModelRunnerOutput` 中也有：

```text
cudagraph_stats=cudagraph_stats
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4609` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4623`

日志聚合由 `CUDAGraphLogging` 完成：

```text
Unpadded Tokens
Padded Tokens
Num Paddings
Runtime Mode
Count
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:40` 到 `code/vllm/vllm/compilation/cuda_graph.py:123`

---

## 25. 一个完整例子：普通 decode FULL cudagraph

假设：

```text
backend 支持 UNIFORM_SINGLE_TOKEN_DECODE；
cudagraph_mode = FULL_DECODE_ONLY 或 FULL；
本轮是纯 decode；
每个 request 1 个 token；
没有 cascade；
没有 encoder output；
没有需要 eager break 的 connector。
```

链路：

```text
1. _prepare_inputs()
   max_num_scheduled_tokens = 1
   num_tokens = num_reqs

2. _determine_batch_execution_and_padding()
   uniform_decode = True
   dispatcher 根据 num_tokens 找到 FULL BatchDescriptor
   num_tokens 被 pad 到 capture size

3. _get_slot_mappings()
   按 padded tokens 准备 slot_mapping

4. _build_attention_metadata()
   num_tokens_padded / num_reqs_padded 进入 metadata
   padded rows / padded slots 安全填充

5. set_forward_context(...)
   cudagraph_runtime_mode = FULL
   batch_descriptor = FULL key

6. CUDAGraphWrapper.__call__()
   第一次 capture；之后相同 key replay

7. Attention.forward()
   通过 custom op 调 backend impl.forward
```

结果：

```text
decode 热路径减少 Python 和 kernel launch overhead。
```

---

## 26. 一个完整例子：mixed batch 降为 FULL_AND_PIECEWISE

假设：

```text
用户配置 FULL；
backend 支持 UNIFORM_SINGLE_TOKEN_DECODE，但不支持 mixed batch full graph；
attention splitting_ops 存在；
本轮有 prefill + decode mixed batch。
```

resolve 阶段：

```text
mixed_mode FULL 不满足 min_cg_support == ALWAYS；
因为 splitting_ops 包含 attention；
最终 cudagraph_mode = FULL_AND_PIECEWISE。
```

运行时：

```text
pure decode batch：dispatch 到 FULL；
mixed batch：dispatch 到 PIECEWISE；
如果没有匹配 key：fallback NONE。
```

这就是 `FULL_AND_PIECEWISE` 的价值：

```text
decode 用最强 full graph；
mixed 动态 batch 用 piecewise 保留部分 graph 优化。
```

---

## 27. 一个完整例子：KV connector + breakable graph

假设启用 KV connector，并且 connector 有 layerwise load/save。

attention custom op 上有：

```text
@eager_break_during_capture
@maybe_transfer_kv_layer
```

运行时：

```text
1. BreakableCUDAGraphWrapper 开始 capture graph segment。
2. 执行到 unified_attention_with_output。
3. eager_break_during_capture 结束当前 graph segment。
4. maybe_transfer_kv_layer 在 eager 段执行 wait_for_layer_load。
5. attention impl.forward 执行。
6. maybe_transfer_kv_layer 执行 save_kv_layer。
7. 开启下一个 graph segment。
8. replay 时按 segments 顺序执行：graph → eager attention/hook → graph。
```

关键点：

```text
KV connector side effect 不会被错误录进 monolithic CUDA graph；
attention op 成为 eager break 边界；
仍然保留 attention 前后其他片段的 graph replay 能力。
```

---

## 28. 容易疑惑的点

### 28.1 cudagraph_mode 配置值等于每轮 runtime mode 吗？

不一定。

用户配置可能是：

```text
FULL_AND_PIECEWISE
FULL_DECODE_ONLY
FULL
PIECEWISE
```

每轮 runtime mode 由 dispatcher 决定，只会是：

```text
FULL
PIECEWISE
NONE
```

### 28.2 FULL_DECODE_ONLY 是不是禁用 prefill？

不是。

它只是表示：

```text
prefill / mixed batch 不走 CUDA graph；
仍然正常 eager / compiled 执行。
```

### 28.3 为什么 backend 不支持 mixed full graph 时还能跑？

因为 resolve 会降级到：

```text
FULL_DECODE_ONLY
FULL_AND_PIECEWISE
PIECEWISE
NONE
```

只要不是强制非法组合，业务执行仍然可以 fallback。

### 28.4 BatchDescriptor 为什么要包含 LoRA 信息？

LoRA 是否激活、激活多少 adapter 会影响 forward 中的执行路径和 shape / specialization。

因此 cudagraph key 需要区分：

```text
has_lora
num_active_loras
```

### 28.5 为什么 piecewise key 可以 num_reqs=None？

因为 piecewise graph 更关注 token shape，request 数变化可以通过 metadata tensor 内容表达。

FULL graph 对某些 backend，例如 FA3 scheduler metadata，可能依赖精确 request 数，所以不能放松。

### 28.6 attention metadata 是不是被 torch.compile 编进图里？

不是简单地整体编进去。

metadata 通过 forward context 和 custom op 进入 attention backend；大量动态字段是 runtime tensor / dataclass。custom op / splitting op 边界避免 compiler 试图追踪所有 backend 内部动态逻辑。

### 28.7 build_for_cudagraph_capture 是否一定不同于 build？

不一定。

默认实现复用普通 `build()`。只有 backend 需要更稳定 metadata 或特殊 capture plan 时才 override。

### 28.8 KV connector 为什么不能直接放进 full graph？

因为它包含 host-side communication / wait / save side effects：

```text
wait_for_layer_load
save_kv_layer
wait_for_save
clear_connector_metadata
```

这些操作不应该被 CUDA graph replay 固化。

### 28.9 CUDA graph 会改变 attention 结果吗？

不应该。

它只改变执行包装和 replay 方式。正确性依赖于：

```text
padding 安全；
metadata 区分真实 token；
slot_mapping padded 区域不写 KV；
backend 声明的 CG support 真实可靠；
side effects 不被错误 capture。
```

---

## 29. 调试入口

调试 attention 与 CUDA graph / compile 的交互，建议按下面顺序看：

```text
1. CompilationConfig
   看 mode、cudagraph_mode、splitting_ops、capture sizes。

2. VllmConfig post init
   看 connector 是否要求 piecewise_for_cudagraph。

3. GPUModelRunner._check_and_update_cudagraph_mode()
   看 backend 的 AttentionCGSupport 和最终 cudagraph_mode。

4. CudagraphDispatcher.initialize_cudagraph_keys()
   看 FULL / PIECEWISE keys 和 BatchDescriptor。

5. GPUModelRunner._determine_batch_execution_and_padding()
   看本轮 uniform_decode、padding、runtime mode、cudagraph_stats。

6. GPUModelRunner._build_attention_metadata()
   看 num_tokens_padded / num_reqs_padded / build_for_cudagraph_capture。

7. set_forward_context()
   看 cudagraph_runtime_mode、batch_descriptor、slot_mapping 是否进入 ForwardContext。

8. CUDAGraphWrapper / BreakableCUDAGraphWrapper
   看是否 capture / replay / eager break。

9. unified_attention_with_output()
   看 attention custom op 是否作为 split / eager break 边界。

10. maybe_transfer_kv_layer
   看 connector layer hook 是否在 attention 前后执行。
```

常用源码位置：

```text
code/vllm/vllm/config/compilation.py:53
code/vllm/vllm/config/compilation.py:1316
code/vllm/vllm/v1/cudagraph_dispatcher.py:15
code/vllm/vllm/v1/worker/gpu_model_runner.py:3810
code/vllm/vllm/v1/worker/gpu_model_runner.py:6877
code/vllm/vllm/forward_context.py:249
code/vllm/vllm/compilation/cuda_graph.py:145
code/vllm/vllm/compilation/breakable_cudagraph.py:59
code/vllm/vllm/model_executor/layers/attention/attention.py:734
```

---

## 30. 最终可以记成一张表

| 阶段 | 主要函数 / 类 | 核心输入 | 核心输出 | 作用 |
|---|---|---|---|---|
| 配置 | `CompilationConfig` | compile mode、cudagraph mode、capture sizes | 编译 / graph 策略 | 声明用户期望 |
| backend 能力 | `AttentionCGSupport` | backend builder | CG support level | 约束 full / piecewise 能力 |
| 模式解析 | `resolve_cudagraph_mode_and_sizes()` | 用户模式 + backend 支持 | 最终 cudagraph_mode | 自动降级或报错 |
| key 初始化 | `CudagraphDispatcher.initialize_cudagraph_keys()` | capture sizes、LoRA、uniform decode | FULL / PIECEWISE keys | 定义可 replay 的 batch descriptors |
| 每轮决策 | `_determine_batch_execution_and_padding()` | num_tokens、uniform_decode、DP、cascade | runtime mode、BatchDescriptor、padding | 决定本轮是否 graph |
| metadata 构造 | `_build_attention_metadata()` | padded token / req 数 | padded metadata | 保证 graph shape 稳定 |
| context 注入 | `set_forward_context()` | runtime mode、descriptor、metadata | `ForwardContext` | 让 wrapper / attention 感知 graph 状态 |
| graph wrapper | `CUDAGraphWrapper` | batch descriptor | capture / replay | full / piecewise graph 执行 |
| breakable graph | `BreakableCUDAGraphWrapper` | eager break custom op | graph + eager segments | 避免 side effect 被 capture |
| attention 边界 | `unified_attention_with_output()` | Q/K/V、metadata、KV cache | backend output | custom op / split / eager break 边界 |
| connector hook | `maybe_transfer_kv_layer` | layer_name、metadata、kv_cache | wait/save KV | attention layer 前后处理 KV transfer |
| 观测 | `CUDAGraphStat` | unpadded / padded tokens | cudagraph stats | 统计 padding 和 runtime mode |

---

## 31. 总结

Attention 与 CUDA graph / torch.compile 的关系可以压缩成：

```text
配置层：
  CompilationConfig 决定想用哪种 compile / cudagraph mode。

能力层：
  Attention backend builder 声明 AttentionCGSupport。

解析层：
  resolve_cudagraph_mode_and_sizes() 根据 backend 能力自动降级或报错。

调度层：
  CudagraphDispatcher 用 BatchDescriptor 决定本轮 FULL / PIECEWISE / NONE。

准备层：
  ModelRunner padding inputs、slot_mapping、attention metadata。

上下文层：
  set_forward_context 把 runtime mode 和 descriptor 注入模型 forward。

执行层：
  CUDAGraphWrapper / BreakableCUDAGraphWrapper capture 或 replay。

attention 边界：
  unified_attention_with_output 作为 custom op、split op、eager break 点。

connector 边界：
  maybe_transfer_kv_layer 把 KV load/save 放在 attention 前后。
```

如果只记住最小心智模型：

```text
CUDA graph 要稳定 shape 和地址；
torch.compile 要稳定 graph 边界；
attention 的动态 metadata 和 KV connector side effects 通过 padding、BatchDescriptor、custom op、piecewise split、eager break 被隔离和表达。
```

再压缩成一句话：

```text
vLLM V1 的 attention graph 优化不是把 attention 简单塞进一个大图，而是由 backend 能力声明、dispatcher key、padded metadata、forward context、custom op 边界和 KV connector eager hook 共同维持“可 capture/replay”与“动态调度正确性”的平衡。
```
