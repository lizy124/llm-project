# 11. 如何调试 compile / CUDA graph 问题？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\config\compilation.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\observability.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\cuda_graph.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\monitor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\counter.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\cudagraph_dispatcher.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_worker.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\metrics\loggers.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\logger.py`

本问题关注：compile overhead、CUDA graph capture 失败、runtime graph miss、shape / metadata mismatch、fallback、padding waste、性能不达预期时如何定位。

---

## 1. 一句话回答

调试 compilation / CUDA graph 问题，不要只看“有没有打开 cudagraph”，而要沿着这条链路逐层确认：

```text
CompilationConfig
  → attention backend 支持度解析
  → CudagraphDispatcher 初始化合法 key
  → capture_model() 是否真的 capture
  → execute_model() 每轮 runtime dispatch
  → ForwardContext 传入的 cudagraph_runtime_mode / BatchDescriptor
  → CUDAGraphWrapper 是 replay、capture 还是 pass-through
  → ModelRunnerOutput / SchedulerStats / 日志指标
```

最核心的问题永远是两个：

```text
1. 启动时到底 capture 了哪些 graph key？
2. 运行时每轮 batch 到底 dispatch 到 FULL / PIECEWISE / NONE 中哪一个，为什么？
```

---

## 2. 最小定位路线

遇到 compile / cudagraph 性能或正确性问题时，可以先按这个顺序查：

```text
1. 看最终 CompilationConfig：mode、cudagraph_mode、capture_sizes、max_cudagraph_capture_size。
2. 看 attention backend 是否把 cudagraph_mode 降级。
3. 看 cudagraph dispatcher 是否初始化了 FULL / PIECEWISE keys。
4. 看 capture_model() 是否执行、耗时多少、占用多少显存。
5. 看 execute_model() 每轮 dispatch 得到的 cudagraph_mode 和 BatchDescriptor。
6. 看是否因为 padding、LoRA、cascade attention、encoder input、DP 协调等导致 fallback。
7. 看 CUDAGraphWrapper 是否 replay；DEBUG 下还会检查 replay 输入地址是否一致。
8. 看 cudagraph_metrics 表，确认 runtime_mode 频率和 padding waste。
9. 看 compilation_time / encoder_compilation_time，区分 compile 慢还是 runtime 慢。
10. 如有输出异常，再查 padded token、slot_mapping、attention metadata、KV cache 写入是否一致。
```

一句话：

```text
先定位“路径”，再定位“性能”，最后定位“正确性”。
```

---

## 3. 先区分几个常见问题

### 3.1 compile 慢

表现：

```text
- 模型启动很久；
- 日志里 torch.compile took 很长；
- 第一次请求 TTFT 高；
- CPU 占用高，但不一定在 capture graph。
```

主要看：

- `CompilationConfig.mode`
- `CompilationConfig.backend`
- `CompilationConfig.compile_sizes`
- `CompilationConfig.compile_ranges_endpoints`
- `CompilationConfig.compilation_time`
- `CompilationConfig.encoder_compilation_time`
- `compilation_counter.num_backend_compilations`
- `compilation_counter.num_inductor_compiles`
- `compilation_counter.num_cache_entries_updated`

计时入口在 `monitor_torch_compile()`：

```text
monitor_torch_compile()
  → 记录 torch_compile_start_time
  → 正常结束后累加 compilation_time / encoder_compilation_time
  → logger.info_once("torch.compile took %.2f s in total")
```

位置：`code/vllm/vllm/compilation/monitor.py:17`

### 3.2 CUDA graph capture 慢或 OOM

表现：

```text
- 启动阶段卡在 Capturing CUDA graphs；
- 日志显示 Graph capturing finished in N secs, took X GiB；
- capture 阶段 OOM；
- capture sizes 太多导致启动慢。
```

主要看：

- `cudagraph_mode`
- `cudagraph_capture_sizes`
- `max_cudagraph_capture_size`
- `cudagraph_num_of_warmups`
- LoRA capture cases
- FULL / PIECEWISE graph 数量
- encoder CUDA graph 是否开启

capture 主入口：

```text
GPUModelRunner.capture_model()
  → cudagraph_dispatcher.get_capture_descs()
  → _capture_cudagraphs()
  → _warmup_and_capture()
  → _dummy_run(..., is_graph_capturing=True)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6583`

### 3.3 graph miss / fallback 太多

表现：

```text
- 已经 capture 了 graph，但运行时 runtime_mode 经常是 NONE；
- throughput 没提升；
- cudagraph_metrics 里 NONE 占比很高；
- 某些 batch size 命中，另一些不命中。
```

主要看：

- 运行时 `num_tokens` 是否超过 `max_cudagraph_capture_size`
- 是否命中 `cudagraph_capture_sizes` padding bucket
- `uniform_decode` 是否为 True
- attention backend 是否支持 FULL
- 本轮是否有 cascade attention / encoder input / KV scale calculation
- LoRA active 数是否有对应 capture key
- DP 协调后是否强制同一个 mode

runtime dispatch 在：

```text
GPUModelRunner._determine_batch_execution_and_padding()
  → CudagraphDispatcher.dispatch()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`，`code/vllm/vllm/v1/cudagraph_dispatcher.py:235`

### 3.4 输出异常 / shape mismatch

表现：

```text
- graph replay 后输出错误；
- CUDA illegal memory access；
- attention shape mismatch；
- padding token 污染 KV cache；
- spec decode 下 logits / draft token 对不上。
```

主要看：

- `BatchDescriptor.num_tokens` 和真实 token 数是否区分清楚
- `slot_mapping` padding 区是否填 `-1`
- `attention metadata` 是否按 FULL graph padded shape 构造
- `positions` / `seq_lens` / `query_start_loc` 是否使用同一套 padded / unpadded 语义
- CUDAGraph replay 的输入 tensor 地址是否稳定

DEBUG 模式下，`CUDAGraphWrapper` 会记录 capture 时输入 tensor 地址，replay 时断言地址一致：

```text
Expected entry.input_addresses, got new_input_addresses
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:346`

---

## 4. 配置层：先确认最终打开了什么

### 4.1 mode 和 cudagraph_mode 是两层概念

`CompilationConfig` 里有两组容易混淆的配置。

第一组是 torch.compile / vLLM compile：

```text
CompilationMode.NONE
CompilationMode.STOCK_TORCH_COMPILE
CompilationMode.DYNAMO_TRACE_ONCE
CompilationMode.VLLM_COMPILE
```

位置：`code/vllm/vllm/config/compilation.py:37`

第二组是 CUDA graph runtime mode：

```text
CUDAGraphMode.NONE
CUDAGraphMode.PIECEWISE
CUDAGraphMode.FULL
CUDAGraphMode.FULL_DECODE_ONLY
CUDAGraphMode.FULL_AND_PIECEWISE
```

位置：`code/vllm/vllm/config/compilation.py:53`

它们的关系是：

```text
compile 决定底层 runnable 是 eager / torch.compile / vLLM compile；
cudagraph 决定这一轮是否 replay CUDA graph。
```

所以：

```text
cudagraph_runtime_mode = NONE
```

不等于“没有 compile”，只表示这一轮不使用 CUDA graph replay。

### 4.2 默认推荐不是强行 FULL

`FULL_AND_PIECEWISE` 的语义是：

```text
decode batch 尽量 FULL graph；
prefill / mixed prefill-decode 尽量 PIECEWISE graph；
不满足条件时退到 NONE。
```

配置说明在 `CompilationConfig.cudagraph_mode`：`code/vllm/vllm/config/compilation.py:587`

调试时不要只问：

```text
为什么不是 FULL？
```

而要问：

```text
本轮是 uniform decode 吗？
attention backend 支持 FULL 吗？
有没有 cascade attention / encoder input 等禁用 FULL 的因素？
PIECEWISE 是否可用？
```

### 4.3 capture size 决定能命中哪些 shape

关键字段：

```text
cudagraph_capture_sizes
max_cudagraph_capture_size
compile_sizes
```

位置：`code/vllm/vllm/config/compilation.py:629`，`code/vllm/vllm/config/compilation.py:673`

规则：

```text
CUDA graph 必须按固定 shape replay；
真实 num_tokens 会被 pad 到某个 captured size；
超过 max_cudagraph_capture_size 直接 runtime_mode=NONE。
```

`post_init_cudagraph_sizes()` 会对 capture sizes 做排序，并保证最大值和 `max_cudagraph_capture_size` 一致：

位置：`code/vllm/vllm/config/compilation.py:1070`

### 4.4 compile_sizes 不能被 cudagraph padding 改写

`CudagraphDispatcher._compute_bs_to_padded_graph_size()` 会检查：

```text
如果 compile_sizes 中某个 size 会被 padding 到另一个 cudagraph size，直接 ValueError。
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:93`

原因是：

```text
compile_sizes 是 Inductor 编译的 shape；
cudagraph padding 会改变实际 runtime shape；
两者不一致会造成调试和性能判断混乱。
```

### 4.5 LoRA 会增加 graph key 维度

如果启用 LoRA，dispatcher 会把 LoRA 状态放进 `BatchDescriptor`：

```text
has_lora
num_active_loras
```

LoRA capture cases 来自 `_get_lora_cases()`：

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:111`

如果 `cudagraph_specialize_lora=True`，会分别捕获无 LoRA / 有 LoRA 的若干 active count bucket；如果关闭 specialization，则可能统一用 `max_loras + 1` 的 key。

调试 LoRA graph miss 时要看：

```text
当前 batch active LoRA 数；
dispatcher captured_lora_counts；
BatchDescriptor.num_active_loras；
是否对应已有 FULL / PIECEWISE key。
```

---

## 5. attention backend 会改写 cudagraph_mode

### 5.1 为什么配置不是最终结果

用户配置的 `cudagraph_mode` 不是最终 runtime 可用 mode。初始化 attention backend 后，`GPUModelRunner` 会检查所有 attention backend 的 CUDA graph 支持度：

```text
GPUModelRunner.initialize_attn_backend()
  → _check_and_update_cudagraph_mode()
  → CompilationConfig.resolve_cudagraph_mode_and_sizes()
  → cudagraph_dispatcher.initialize_cudagraph_keys()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6736`，`code/vllm/vllm/v1/worker/gpu_model_runner.py:6877`

### 5.2 支持度取最弱 backend

`_check_and_update_cudagraph_mode()` 会遍历每个 KV cache group / attention group 的 backend：

```text
builder_cls.get_cudagraph_support(vllm_config, kv_cache_spec)
```

然后取最小支持度：

```text
min_cg_support
min_cg_attn_backend
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6889`

这意味着：

```text
只要一个 attention group 不支持 FULL，整体 FULL 能力就可能被降级。
```

### 5.3 常见自动降级

`resolve_cudagraph_mode_and_sizes()` 里有几类降级：

1. mixed batch FULL 不支持：

```text
FULL → FULL_AND_PIECEWISE 或 FULL_DECODE_ONLY
```

位置：`code/vllm/vllm/config/compilation.py:1333`

2. decode FULL 完全不支持：

```text
如果 attention 可 piecewise compile → PIECEWISE
否则 → NONE
```

位置：`code/vllm/vllm/config/compilation.py:1361`

3. spec decode 的 uniform query length 大于 1，但 backend 不支持 uniform batch FULL：

```text
FULL decode → PIECEWISE 或 NONE
```

位置：`code/vllm/vllm/config/compilation.py:1387`

4. Mamba cache blocks 不够时直接报错：

```text
max_num_seqs > available Mamba cache blocks
```

位置：`code/vllm/vllm/config/compilation.py:1435`

### 5.4 调试提示

如果你配置了 `FULL`，但实际不是 FULL，优先找 warning：

```text
CUDAGraphMode.X is not supported with Y backend ... setting cudagraph_mode=...
```

这类 warning 来自 `resolve_cudagraph_mode_and_sizes()`。

---

## 6. dispatcher 层：确认合法 key 有没有初始化

### 6.1 dispatcher 是 runtime dispatch 的唯一 key 源

`CudagraphDispatcher` 保存两组 key：

```text
cudagraph_keys[PIECEWISE]
cudagraph_keys[FULL]
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:39`

注释里明确说：

```text
The keys stored in dispatcher are the only source of truth for valid cudagraphs
that can be dispatched at runtime.
```

所以调试 graph miss 时，不要只看 wrapper 里有没有 graph，要先看 dispatcher 是否认为这个 key 合法。

### 6.2 initialize_cudagraph_keys() 做什么

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:166`

它会：

```text
1. 如果 cudagraph_mode=NONE，则不创建 key。
2. 预计算 batch size → padded graph size 映射。
3. 根据 LoRA 配置生成 capture cases。
4. 为 mixed mode 创建 PIECEWISE 或 FULL key。
5. 为 separate decode FULL 创建 uniform decode FULL key。
6. 标记 keys_initialized=True。
```

### 6.3 get_capture_descs() 决定 capture 哪些 graph

`capture_model()` 不自己枚举 batch sizes，而是调用：

```text
cudagraph_dispatcher.get_capture_descs()
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:326`

返回顺序是：

```text
PIECEWISE first
FULL second
```

每组内部按：

```text
(num_tokens, num_active_loras) descending
```

排序。

这么做是为了：

```text
先 capture 大 shape，让小 graph 尽量复用大 graph 的 memory pool。
```

---

## 7. capture 阶段：怎么确认真的录图了

### 7.1 capture_model() 的关键日志

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6583`

如果 CUDA graph 被关闭，会看到 warning：

```text
Skipping CUDA graph capture. To turn on CUDA graph capture, ensure `cudagraph_mode` was not manually set to `NONE`
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6585`

正常完成会看到：

```text
Graph capturing finished in %.0f secs, took %.2f GiB
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6644`

这条日志回答的是：

```text
capture 总耗时多少；
capture 后图池大概占了多少显存。
```

### 7.2 capture 前后会打开全局 capture 许可

`capture_model()` 里：

```text
set_cudagraph_capturing_enabled(True)
...
set_cudagraph_capturing_enabled(False)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6602`，`code/vllm/vllm/v1/worker/gpu_model_runner.py:6631`

`CUDAGraphWrapper` 真正 capture 前会调用：

```text
validate_cudagraph_capturing_enabled()
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:276`

如果在不该 capture 的运行时发生 lazy capture，会报：

```text
CUDA graph capturing detected at an inappropriate time. This operation is currently disabled.
```

位置：`code/vllm/vllm/compilation/monitor.py:90`

这对调试“为什么线上突然 capture / 卡顿”很有用。

### 7.3 warmup 和 capture 是分开的

`_warmup_and_capture()`：

```text
for _ in range(cudagraph_num_of_warmups):
    _dummy_run(..., cudagraph_runtime_mode=NONE)

_dummy_run(..., cudagraph_runtime_mode=FULL/PIECEWISE, is_graph_capturing=True)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6651`

所以：

```text
warmup 用 NONE；
真正 capture 用 FULL / PIECEWISE。
```

如果你看到 warmup 很慢，不一定是 graph capture 慢，也可能是 kernel 初始化、attention workspace 初始化、sampler warmup 等。

### 7.4 capture 进度条

`_capture_cudagraphs()` 会在 global first rank 打 tqdm：

```text
Capturing CUDA graphs (decode, FULL)
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6702`

这个字符串能直接告诉你当前 capture 的 batch 类型和 runtime mode。

### 7.5 compilation_counter 能看 capture 次数

`compilation_counter` 记录：

```text
num_gpu_runner_capture_triggers
num_cudagraph_captured
```

定义位置：`code/vllm/vllm/compilation/counter.py:20`

更新点：

```text
capture_model()                         → num_gpu_runner_capture_triggers += 1
CUDAGraphWrapper 第一次 capture entry      → num_cudagraph_captured += 1
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6595`，`code/vllm/vllm/compilation/cuda_graph.py:339`

---

## 8. runtime dispatch：每轮到底走哪条路

### 8.1 execute_model() 中 dispatch 的位置

真实请求执行时，主链路是：

```text
GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
  → _compute_cascade_attn_prefix_lens()
  → _determine_batch_execution_and_padding()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(...)
  → _model_forward()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`

runtime dispatch 发生在：

```text
_determine_batch_execution_and_padding()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`

### 8.2 函数输出就是调试重点

`_determine_batch_execution_and_padding()` 返回：

```text
cudagraph_mode
batch_descriptor
should_ubatch
num_tokens_across_dp
cudagraph_stats
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3916`

`execute_model()` 里有 debug 日志：

```text
Running batch with cudagraph_mode: %s, batch_descriptor: %s, should_ubatch: %s, num_tokens_across_dp: %s
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4158`

调试时开启 DEBUG 日志后，这条通常是最直接的 runtime 路径证据。

### 8.3 dispatch 的 fallback 条件

`CudagraphDispatcher.dispatch()` 中，下面情况直接返回 `NONE`：

```text
keys 未初始化；
全局 cudagraph_mode 是 NONE；
max_cudagraph_capture_size 为空；
num_tokens > max_cudagraph_capture_size；
allowed_modes 只允许 NONE。
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:274`

如果没有命中 FULL / PIECEWISE key，也会返回：

```text
CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:320`

### 8.4 FULL 优先，再 PIECEWISE，最后 NONE

搜索顺序：

```text
1. FULL allowed 且 FULL key 存在 → FULL
2. PIECEWISE allowed 且 relaxed PIECEWISE key 存在 → PIECEWISE
3. 否则 → NONE
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:307`

PIECEWISE 会把 key 放宽成：

```text
num_reqs=None
uniform=False
```

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:316`

这解释了为什么：

```text
FULL graph 对 batch 形态更敏感；
PIECEWISE graph 更容易命中。
```

---

## 9. 几个最常见的 runtime fallback 原因

### 9.1 num_tokens 超出 capture 范围

如果：

```text
num_tokens > max_cudagraph_capture_size
```

dispatcher 直接返回 `NONE`。

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:274`

解决方向：

```text
- 增大 max_cudagraph_capture_size；
- 调整 cudagraph_capture_sizes；
- 接受大 prefill eager；
- 降低 max_num_batched_tokens 或让 decode/prefill 形态更稳定。
```

### 9.2 capture_sizes 太稀导致 padding 浪费或 miss

真实 token 数会通过 `_bs_to_padded_graph_size` pad 到捕获 size：

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:72`

典型现象：

```text
num_unpadded_tokens=65
num_padded_tokens=128
num_paddings=63
runtime_mode=FULL/PIECEWISE
```

这说明命中了 graph，但 padding waste 很大，吞吐可能不升反降。

### 9.3 本轮不是 uniform decode

FULL decode graph 通常要求：

```text
max_num_scheduled_tokens == uniform_decode_query_len
num_tokens == max_num_scheduled_tokens * num_reqs
```

判断在 `_is_uniform_decode()`：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3789`

如果 batch 是 mixed prefill-decode，通常无法命中 decode FULL，只能走 PIECEWISE 或 NONE。

### 9.4 cascade attention 禁用 FULL

`_determine_batch_execution_and_padding()` dispatch 时：

```text
disable_full = use_cascade_attn or has_encoder_output
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3865`

含义：

```text
cascade attention 可以继续用 PIECEWISE，但不能用 FULL。
```

### 9.5 encoder-decoder 首轮禁用 FULL / compiled

如果 encoder-decoder 本轮有 encoder input：

```text
has_encoder_output=True → dispatch 禁用 FULL
has_encoder_input=True  → set_forward_context(skip_compiled=True)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3841`，`code/vllm/vllm/v1/worker/gpu_model_runner.py:4312`

### 9.6 KV scales 第一轮强制 NONE

如果需要计算 KV scales：

```text
if self.calculate_kv_scales:
    cudagraph_mode = CUDAGraphMode.NONE
    self.calculate_kv_scales = False
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4282`

因此第一轮可能不走 graph，后续才恢复。

### 9.7 DP rank 需要统一 mode / padding

data parallel 下会调用：

```text
coordinate_batch_across_dp(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3882`

协调后可能用 `synced_cudagraph_mode` 重新 dispatch：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3898`

所以单 rank 看起来能命中，不代表 DP 整体能按本 rank 原始 shape 运行。

### 9.8 LoRA active count 不匹配

如果有 LoRA，dispatch 会把实际 active count 映射到 captured bucket：

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:283`

如果没有匹配 key，则 fallback。

### 9.9 spec decode shape 需要按 query length 对齐

spec decode 下：

```text
uniform_decode_query_len = 1 + num_speculative_tokens
```

如果 FULL decode graph 开启且 query length > 1，capture sizes 会被调整为该长度的倍数：

位置：`code/vllm/vllm/config/compilation.py:1421`，`code/vllm/vllm/config/compilation.py:1462`

如果没有合法 size，会报错提示调整 `num_speculative_tokens`、`max_cudagraph_capture_size` 或 `cudagraph_capture_sizes`。

---

## 10. ForwardContext 和 wrapper 层怎么判断 replay

### 10.1 set_forward_context 是运行态传递点

`execute_model()` forward 前写入：

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

`_model_forward()` 自己没有 cudagraph 分支：

```text
return self.model(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3757`

真正的分支在 wrapper / compiled callable / attention backend 内部通过 `ForwardContext` 读取。

### 10.2 CUDAGraphWrapper 的三种行为

入口：`code/vllm/vllm/compilation/cuda_graph.py:233`

1. 没有 forward context：

```text
直接 runnable(*args, **kwargs)
```

典型是 vision encoder 等不在普通 LM forward context 中的调用。

2. mode 不匹配或 NONE：

```text
cudagraph_runtime_mode == NONE
或 cudagraph_runtime_mode != self.runtime_mode
→ pass-through
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:244`

3. mode 匹配：

```text
BatchDescriptor 无 entry → 创建 entry 并 capture
entry 已有 cudagraph → replay
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:256`，`code/vllm/vllm/compilation/cuda_graph.py:357`

### 10.3 DEBUG 模式下检查输入地址

`CUDAGraphWrapper` 初始化时：

```text
self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:190`

capture 时记录：

```text
entry.input_addresses = [x.data_ptr() for tensor args]
```

replay 时断言：

```text
new_input_addresses == entry.input_addresses
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:346`

如果这里报错，通常说明：

```text
用于 replay 的输入 tensor 不是同一块持久 buffer；
或某个路径意外创建了新 tensor；
或 wrapper capture 的参数列表和 replay 参数列表不稳定。
```

---

## 11. cudagraph_metrics：最有用的运行时表

### 11.1 如何开启

`ObservabilityConfig` 里有：

```text
cudagraph_metrics: bool = False
```

说明：

```text
Enable CUDA graph metrics (number of padded/unpadded tokens, runtime cudagraph dispatch modes, and their observed frequencies at every logging interval).
```

位置：`code/vllm/vllm/config/observability.py:56`

CLI 参数在 `arg_utils.py` 中暴露为 observability 参数；打开后，`LoggingStatLogger` 会创建 `CUDAGraphLogging`：

位置：`code/vllm/vllm/v1/metrics/loggers.py:119`

### 11.2 每轮 stats 在哪里生成

`_determine_batch_execution_and_padding()` 中：

```text
if observability_config.cudagraph_metrics:
    cudagraph_stats = CUDAGraphStat(
        num_unpadded_tokens=num_tokens,
        num_padded_tokens=batch_descriptor.num_tokens,
        num_paddings=batch_descriptor.num_tokens - num_tokens,
        runtime_mode=str(cudagraph_mode),
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907`

这个 `cudagraph_stats` 最终放进 `ModelRunnerOutput`：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4621`

再进入 scheduler stats / logger。

### 11.3 表格长什么样

`CUDAGraphLogging` 的列是：

```text
Unpadded Tokens
Padded Tokens
Num Paddings
Runtime Mode
Count
```

位置：`code/vllm/vllm/compilation/cuda_graph.py:40`

它会按相同 `CUDAGraphStat` 聚合频次，输出一个表。

### 11.4 如何解读

#### 情况 A：NONE 很多

```text
Runtime Mode = NONE 占比高
```

说明大部分 batch 没走 graph。优先排查：

```text
cudagraph_mode 是否被降级为 NONE；
num_tokens 是否超过 max capture size；
FULL/PIECEWISE key 是否不存在；
cascade / encoder input / DP 协调是否禁用；
```

#### 情况 B：padding 很多

```text
Unpadded Tokens = 33
Padded Tokens   = 64
Num Paddings    = 31
Runtime Mode    = FULL
```

说明命中了 graph，但 padding waste 大。可能吞吐不升反降。

解决方向：

```text
调整 capture_sizes；
减少过大的 bucket gap；
按实际 workload 选择 capture sizes；
观察常见 batch token 数分布。
```

#### 情况 C：FULL 很少，PIECEWISE 很多

这不一定是坏事。可能 workload 是 mixed prefill-decode，或者 attention backend 只适合 decode FULL。

要结合：

```text
num_scheduled_tokens 分布；
uniform_decode 判断；
TTFT / TPOT；
```

来看。

#### 情况 D：FULL 命中但性能差

优先看：

```text
padding waste；
batch 太小导致 graph replay 收益小；
是否启用了过多 LoRA graph variants；
是否有 DP / MoE 通信瓶颈；
是否 sampler / output copy 成为瓶颈。
```

---

## 12. compile / capture / runtime 三类时间指标

### 12.1 compilation_time

`CompilationConfig` 中记录：

```text
compilation_time
encoder_compilation_time
```

位置：`code/vllm/vllm/config/compilation.py:729`

更新在 `monitor_torch_compile()`：

位置：`code/vllm/vllm/compilation/monitor.py:47`

Executor 会聚合 workers 的编译时间，取最大值写回全局 config：

位置：`code/vllm/vllm/v1/executor/abstract.py:124`

EngineCore 初始化日志里也会读取：

位置：`code/vllm/vllm/v1/engine/core.py:324`

### 12.2 capture time / graph memory

`capture_model()` 记录：

```text
elapsed_time = end_time - start_time
cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
```

并打印：

```text
Graph capturing finished in %.0f secs, took %.2f GiB
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6640`

### 12.3 runtime mode / padding

runtime 由 `cudagraph_metrics` 记录：

```text
num_unpadded_tokens
num_padded_tokens
num_paddings
runtime_mode
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907`

这比单纯看吞吐更能解释“为什么没有加速”。

### 12.4 profiler annotation

`GPUWorker.annotate_profile()` 会给 profiler 加 iteration 级 annotation：

```text
execute_context_{num_ctx_requests}({num_ctx_tokens})_generation_{num_generation_requests}({num_generation_tokens})
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:775`

如果用 torch profiler / CUDA profiler，这个 annotation 可以帮助区分 prefill 和 generation step。

---

## 13. 正确性调试：shape / metadata / padding

### 13.1 padded 和 unpadded 要分清

`execute_model()` 里有两个关键值：

```text
num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
num_tokens_padded = batch_desc.num_tokens
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4126`，`code/vllm/vllm/v1/worker/gpu_model_runner.py:4167`

如果是 FULL graph：

```text
pad_attn = cudagraph_mode == CUDAGraphMode.FULL
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4196`

FULL graph 包含 attention，所以 attention metadata 也要按 padded shape 构造。

### 13.2 slot_mapping padding 区必须是 -1

`_get_slot_mappings()` 中：

```text
slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4008`

这防止 padding token 写入真实 KV cache。

如果怀疑 graph replay 污染 KV cache，优先查这里。

### 13.3 dummy run 也会把 slot_mapping 填 -1

`_dummy_run()` 中：

```text
Dummy runs have no real slot assignments — fill with -1 so concat_and_cache kernels skip the KV write.
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5821`

这说明 capture / profile 的 dummy graph 不应该写入真实 KV cache。

### 13.4 attention metadata capture path

`_build_attention_metadata()` 支持：

```text
for_cudagraph_capture=True
```

这会让 builder 调用：

```text
builder.build_for_cudagraph_capture(common_attn_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2417`

如果 capture 失败或 replay shape 不一致，要看 backend 的 `build_for_cudagraph_capture()` 和普通 `build()` 是否构造了兼容 metadata。

### 13.5 seq_lens / query_start_loc / positions 必须同一语义

`_prepare_inputs()` 里会构造：

```text
query_start_loc
optimistic_seq_lens_cpu
num_computed_tokens
positions
seq_lens
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2000`

`_build_attention_metadata()` 会把它们放进 `CommonAttentionMetadata`：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2330`

如果出现 attention shape mismatch，通常要沿着这几项检查。

---

## 14. 输出异常和 NaN 调试

### 14.1 统计 logits 中 NaN

环境变量：

```text
VLLM_COMPUTE_NANS_IN_LOGITS=1
```

对应代码：

```text
if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
    num_nans_in_logits = self._get_nans_in_logits(logits)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3618`

`_get_nans_in_logits()` 会按 request id 统计 logits NaN 数：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5562`

输出会进入 `ModelRunnerOutput.num_nans_in_logits`：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4620`

logger 里也会在开启该环境变量时输出 corrupted request 统计：

位置：`code/vllm/vllm/v1/metrics/loggers.py:263`

### 14.2 eager vs cudagraph 对比

如果怀疑 graph replay 导致输出异常，最直接方法是对比：

```text
cudagraph_mode=NONE
vs
cudagraph_mode=FULL_AND_PIECEWISE / FULL / PIECEWISE
```

重点比较：

```text
sampled_token_ids
logprobs
num_nans_in_logits
KV cache 后续输出是否漂移
```

如果 eager 正常、graph 异常，优先查：

```text
持久输入 buffer 地址；
padding slot_mapping；
attention metadata capture path；
backend 是否声明了过高的 cudagraph support；
某个自定义 op 是否 cudagraph unsafe。
```

---

## 15. 日志和调试开关

### 15.1 VLLM_LOGGING_LEVEL=DEBUG

vLLM logger 由 `logger.py` 初始化，日志格式包含文件和行号：

位置：`code/vllm/vllm/logger.py:22`

开启 DEBUG 后常见收益：

```text
- CUDAGraphWrapper 检查 replay 输入地址；
- cudagraph capture 会打印 Capturing a cudagraph on (..., BatchDescriptor)；
- custom ops enabled / disabled 日志更详细；
- 一些 backend unsupported reason 可能更完整。
```

CUDAGraphWrapper DEBUG 行为：`code/vllm/vllm/compilation/cuda_graph.py:190`

### 15.2 debug_dump_path / depyf

`CompilationConfig.debug_dump_path` 用于 dump compile debug 信息：

位置：`code/vllm/vllm/config/compilation.py:442`

`monitor_torch_compile()` 中，如果：

```text
mode == VLLM_COMPILE
且 compile_debug_dump_path() 不为空
```

会启用 depyf debug dump：

位置：`code/vllm/vllm/compilation/monitor.py:33`

用途：

```text
查看 Dynamo / FX / graph split / Inductor pass 相关产物；
定位 graph break、guard、unexpected recompilation。
```

### 15.3 VLLM_TRACE_FUNCTION

`VLLM_TRACE_FUNCTION` 会启用 Python 函数调用 tracing。

`VllmConfig` 中会根据该环境变量创建 trace 文件路径；`logger.enable_trace_function_call()` 会记录 root_dir 下函数 call / return。

位置：`code/vllm/vllm/logger.py:294`

警告：

```text
它会显著拖慢运行，只适合定位 hang / crash，不适合常规性能测试。
```

### 15.4 layerwise NVTX tracing

`ObservabilityConfig.enable_layerwise_nvtx_tracing`：

```text
Enable layerwise NVTX tracing.
Noted that this doesn't work with CUDA graphs enabled.
```

位置：`code/vllm/vllm/config/observability.py:60`

`GPUModelRunner._register_layerwise_nvtx_hooks()` 会在 dummy run 后注册 hooks，但如果使用 CUDA graph 或 STOCK_TORCH_COMPILE 会有额外限制：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3924`

调试建议：

```text
如果要看逐层 NVTX，先把 cudagraph_mode 关掉或接受 graph 路径下 marker 不完整。
```

### 15.5 jit_monitor_verbose

`ObservabilityConfig.jit_monitor_verbose`：

```text
Log every Triton JIT compile with its dispatch key.
```

位置：`code/vllm/vllm/config/observability.py:79`

用途：

```text
定位运行过程中是否仍在发生 Triton JIT compile；
判断慢请求是否是 JIT 冷启动而不是 cudagraph miss。
```

---

## 16. 性能定位：从 TTFT / TPOT 回到 graph 路径

### 16.1 TTFT 高

优先区分：

```text
启动阶段慢：compile / capture 慢；
首请求慢：JIT / warmup / graph 未命中 / prefill 太大；
请求排队慢：scheduler / KV cache / preemption；
```

compile / capture 相关要看：

```text
compilation_time
encoder_compilation_time
Graph capturing finished in ...
Initial profiling/warmup run took ...
```

### 16.2 TPOT 高

decode 阶段 TPOT 高时优先看：

```text
runtime_mode 是否 FULL；
FULL 的 padding waste 是否过大；
是否因为 LoRA / DP / MoE 通信导致 graph 收益被吞掉；
是否 sampler / async output copy 成为瓶颈；
```

如果 decode 仍大量 `NONE`，说明 graph 没命中；如果 decode 命中 `FULL` 但 TPOT 仍高，就要看 kernel / 通信 / padding / batch size。

### 16.3 graph 命中但吞吐下降

常见原因：

```text
- padding 太多；
- capture sizes 太粗；
- batch 太小，graph replay 收益不足；
- FULL graph 包含了一些本轮不划算的固定开销；
- LoRA specialized graph 太多，启动成本和显存变高；
- DP/MoE 通信瓶颈主导。
```

这时不要盲目追求 FULL，可以比较：

```text
FULL_AND_PIECEWISE
PIECEWISE
NONE
```

在相同 workload 下的 TTFT / TPOT / throughput / graph memory。

---

## 17. 常见问题对照表

| 现象 | 优先检查 | 关键位置 |
|---|---|---|
| 配了 FULL 但实际不是 FULL | attention backend 支持度、自动降级 warning | `config/compilation.py:1316` |
| 启动 capture 很久 | capture sizes、LoRA cases、encoder graphs、warmups | `gpu_model_runner.py:6583` |
| capture OOM | graph memory、max capture size、capture graph 数 | `gpu_model_runner.py:6435` |
| runtime_mode 经常 NONE | num_tokens 超上限、key 不存在、feature 禁用 FULL | `cudagraph_dispatcher.py:235` |
| FULL 少、PIECEWISE 多 | batch 是否 uniform decode、mixed prefill-decode 是否多 | `gpu_model_runner.py:3789` |
| graph 命中但性能差 | padding waste、batch 太小、通信/采样瓶颈 | `CUDAGraphLogging` |
| replay 输出异常 | 输入地址、slot_mapping=-1、attention metadata capture path | `cuda_graph.py:346` |
| 第一轮不走 graph | `calculate_kv_scales`、profile/warmup、encoder input | `gpu_model_runner.py:4282` |
| spec decode 下 shape 错 | capture sizes 是否为 `1 + num_spec_tokens` 倍数 | `compilation.py:1462` |
| LoRA 下 miss | `num_active_loras` 是否有 captured bucket | `cudagraph_dispatcher.py:283` |

---

## 18. 推荐的调试流程示例

### 18.1 先确认最终配置

看最终运行配置中的：

```text
compilation_config.mode
compilation_config.backend
compilation_config.cudagraph_mode
compilation_config.cudagraph_capture_sizes
compilation_config.max_cudagraph_capture_size
compilation_config.compile_sizes
compilation_config.splitting_ops
observability_config.cudagraph_metrics
```

如果最终 `cudagraph_mode=NONE`，后面不用继续查 replay。

### 18.2 再确认初始化是否降级

搜索日志：

```text
CUDAGraphMode.* is not supported
setting cudagraph_mode=...
Skipping CUDA graph capture
Graph capturing finished
```

这一步回答：

```text
启动时到底准备了什么。
```

### 18.3 再看 runtime dispatch

打开 DEBUG 或 cudagraph metrics，看：

```text
Running batch with cudagraph_mode: ..., batch_descriptor: ...
```

以及 cudagraph metrics 表。

这一步回答：

```text
真实 workload 到底命中了什么。
```

### 18.4 最后对照性能

对比：

```text
TTFT
TPOT
prompt throughput
generation throughput
GPU utilization
kernel launch 数
padding waste
runtime_mode 分布
```

如果性能问题和 graph miss 对不上，就说明瓶颈可能不在 cudagraph，而在：

```text
scheduler；
KV cache；
attention backend kernel；
MoE / DP 通信；
sampler；
output D2H copy；
multimodal encoder。
```

---

## 19. 最小心智模型

把调试链路压缩成伪代码：

```text
# 初始化阶段
compilation_config = final_config()
attn_support = min(attn_backend.get_cudagraph_support())
cudagraph_mode = resolve_cudagraph_mode_and_sizes(compilation_config, attn_support)
dispatcher.initialize_cudagraph_keys(cudagraph_mode)

# capture 阶段
for runtime_mode, batch_descs in dispatcher.get_capture_descs():
    for desc in batch_descs:
        warmup(desc, mode=NONE)
        capture(desc, mode=runtime_mode)

# runtime 阶段
num_tokens = scheduler_output.total_num_scheduled_tokens
uniform_decode = is_uniform_decode(...)

cudagraph_mode, batch_desc = dispatcher.dispatch(
    num_tokens,
    uniform_decode=uniform_decode,
    has_lora=...,
    num_active_loras=...,
    invalid_modes={FULL} if cascade_or_encoder else None,
)

with set_forward_context(
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    attn_metadata=...,
):
    model(...)

# wrapper 内部
if context.mode == NONE or context.mode != wrapper.runtime_mode:
    runnable(...)
elif batch_desc not captured:
    capture_or_error_if_capture_disabled(...)
else:
    cudagraph.replay()
```

---

## 20. 一句话总结

```text
compile / CUDA graph 调试的核心不是猜测“为什么慢”，而是把每一轮执行还原成明确证据：最终配置是什么、backend 是否降级、启动 capture 了哪些 BatchDescriptor、运行时 dispatch 到 FULL/PIECEWISE/NONE 哪个 mode、padding 浪费多少、wrapper 是 replay 还是 pass-through；这些证据齐了，compile overhead、graph miss、capture OOM、shape mismatch 和性能不达预期都能定位到具体层。
```
