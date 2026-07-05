# 04. 动态 batch 如何通过 padding 获得 shape 稳定性？

源码位置：

- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/v1/cudagraph_dispatcher.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/ubatch_utils.py`
- `code/vllm/vllm/v1/worker/dp_utils.py`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `code/vllm/vllm/v1/attention/backend.py`
- `code/vllm/vllm/v1/attention/backends/`

本问题关注：vLLM 如何把每轮动态变化的 request / token / attention metadata 映射到 CUDA graph 可 replay 的固定 shape；padding 到 capture bucket 后，`input_ids`、`positions`、`slot_mapping`、`block_table`、`query_start_loc`、`seq_lens`、`ubatch_slices`、logits / output 如何保证只影响真实 token，不让 padding token 污染 KV cache 或输出。

---

## 0. 梳理规划

本篇按“为什么需要 padding → padding bucket 如何选 → padded shape 影响哪些对象 → padding 如何保证语义安全”的顺序梳理。

要回答的问题分成 11 组：

```text
1. CUDA graph 为什么需要 shape 稳定？
2. vLLM 的动态 batch 到底动态在哪里？
3. cudagraph_capture_sizes 和 BatchDescriptor 如何形成 shape bucket？
4. _determine_batch_execution_and_padding() 如何决定 padded shape？
5. sequence parallelism / DP / DBO 如何额外影响 padding？
6. input_ids / positions / inputs_embeds 如何按 padded size 使用？
7. slot_mapping padding 为什么必须填 -1？
8. block_table / query_start_loc / seq_lens / attention metadata 如何按 padded request 处理？
9. logits_indices / sampler / output 如何只消费真实 token？
10. dummy run / capture 时 padding 如何避免污染 KV cache？
11. padding waste 如何通过 cudagraph_metrics 观察？
```

---

## 1. 一句话回答

CUDA graph replay 要求 shape、内存地址和 kernel launch 形态稳定；vLLM 通过 `CudagraphDispatcher` 把真实 `num_tokens / num_reqs / uniform / LoRA` 映射到已 capture 的 `BatchDescriptor`，再把输入 buffer、slot mapping、attention metadata、ubatch slices 等补到 padded shape，同时用 `slot_mapping=-1`、`NULL_BLOCK_ID`、真实 `logits_indices` 和 output slicing 保证 padding token 不影响 KV cache、attention、sampling 和最终输出。

最小主线是：

```text
真实 SchedulerOutput
  → num_tokens_unpadded / num_reqs
  → _pad_for_sequence_parallelism()
  → CudagraphDispatcher.dispatch()
  → BatchDescriptor(num_tokens=padded, num_reqs=padded, ...)
  → pad_attn = cudagraph_mode == FULL
  → _get_slot_mappings(... padded ...)
  → _build_attention_metadata(... padded ...)
  → _preprocess(... padded input buffers ...)
  → set_forward_context(num_tokens=padded, batch_descriptor=...)
  → model forward / graph replay
  → hidden_states / logits 只按真实 logits_indices 和真实 request 输出
```

一句话记忆：

```text
padding 是把动态 serving batch 变成 CUDA graph 固定 shape 的桥；安全填充和输出裁剪保证它不改变模型语义。
```

---

## 2. 为什么需要 padding

CUDA graph capture 记录的是一段 GPU work graph。

它要求：

```text
- kernel launch 序列稳定；
- tensor shape 稳定；
- tensor 地址稳定；
- backend metadata 对象结构稳定；
- attention routine 分支稳定；
- collective / communication 顺序稳定。
```

但 vLLM serving 的 batch 每轮都在变。

### 2.1 request 数动态

每轮 Scheduler 可能调度不同数量的请求：

```text
上一轮：32 个 decode request
下一轮：29 个 decode request + 1 个 prefill request
再下一轮：5 个大 prefill request
```

`num_reqs` 会影响：

```text
query_start_loc 长度；
seq_lens 长度；
block_table 行数；
FULL attention metadata shape；
BatchDescriptor.num_reqs。
```

### 2.2 token 数动态

每个请求本轮 scheduled token 数不同。

例如：

```text
普通 decode：每个 request 1 token；
spec decode 验证：每个 request 1 + draft token 数；
chunked prefill：某些 request 一次处理几十到几千 token；
mixed batch：prefill 和 decode 混合。
```

`num_tokens` 会影响：

```text
input_ids / positions 长度；
slot_mapping 长度；
attention metadata 的 num_actual_tokens；
hidden_states shape；
CUDA graph key。
```

### 2.3 attention metadata 动态

attention metadata 不只是几个数字，而是一整套：

```text
query_start_loc
seq_lens
block_table
slot_mapping
max_query_len
max_seq_len
num_reqs
positions
encoder_seq_lens
backend wrapper / plan / scheduler metadata
```

FULL graph 包含 attention，所以这些对象的 shape 和结构必须与 capture 时一致。

### 2.4 LoRA / spec decode / multimodal / DP 也会改变 shape

额外动态因素包括：

```text
LoRA active adapter 数；
spec decode 的 uniform_decode_query_len；
multimodal / encoder output；
DP rank 上本地 token 数；
DBO / ubatch 切分；
sequence parallelism token 对齐；
PP / intermediate_tensors shape。
```

所以 vLLM 不能简单按真实 batch 直接 replay graph，而要先把真实 batch 映射到固定 bucket。

---

## 3. BatchDescriptor：padding 后的 graph key

`BatchDescriptor` 定义在：`code/vllm/vllm/forward_context.py:30`

字段：

```text
num_tokens
num_reqs
uniform
has_lora
num_active_loras
```

它是 `CUDAGraphWrapper` 的 capture / replay key。

### 3.1 num_tokens

`num_tokens` 通常是 padded 后的 token 数。

例如：

```text
真实 num_tokens = 17
capture sizes = [1, 2, 4, 8, 16, 32]
→ padded num_tokens = 32
```

### 3.2 num_reqs

`num_reqs` 对 FULL graph 重要，因为 attention metadata 可能依赖 request 数。

PIECEWISE graph 通常更宽松，可以把：

```text
num_reqs=None
uniform=False
```

这样同一个 token size 的 piecewise graph 可以复用给更多 batch 形态。

### 3.3 uniform

`uniform` 表示是否所有 request query length 一致。

普通 decode：

```text
uniform=True
max_query_len=1
```

spec decode 验证步：

```text
uniform=True
max_query_len=1 + num_speculative_tokens
```

mixed prefill-decode：

```text
uniform=False
```

FULL graph 尤其依赖这个字段，因为 attention routine 可能完全不同。

### 3.4 LoRA 字段

```text
has_lora
num_active_loras
```

用于区分 LoRA kernel / mapping 的 graph key。

如果 LoRA active count 不同，kernel launch 或 buffer 形态可能不同。

---

## 4. capture sizes 如何形成 padding bucket

`cudagraph_capture_sizes` 定义在：`code/vllm/vllm/config/compilation.py:629`

它表示 vLLM 会提前 capture 哪些 token size。

`max_cudagraph_capture_size` 定义在：`code/vllm/vllm/config/compilation.py:673`

它表示最大可 graph 的 token size。

### 4.1 真实 token 数到 padded graph size

`CudagraphDispatcher._compute_bs_to_padded_graph_size()` 负责建立映射。

位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:72`

规则：

```text
如果真实 size 正好在 capture sizes 中：
  padded size = 真实 size

如果真实 size 落在两个 capture sizes 中间：
  padded size = 下一个更大的 capture size

如果真实 size > max_cudagraph_capture_size：
  不能 graph，runtime_mode=NONE
```

举例：

```text
capture sizes = [1, 2, 4, 8, 16]

真实 num_tokens = 5
→ padded num_tokens = 8

真实 num_tokens = 16
→ padded num_tokens = 16

真实 num_tokens = 17
→ 超出最大 capture size，NONE
```

### 4.2 _create_padded_batch_descriptor()

入口：`code/vllm/vllm/v1/cudagraph_dispatcher.py:132`

它根据真实 batch 和 LoRA 信息构造 padded descriptor。

输出是：

```text
BatchDescriptor(num_tokens=padded_size, num_reqs=..., uniform=..., has_lora=..., num_active_loras=...)
```

### 4.3 compile_sizes 不能被 padding 改写

`compile_sizes` 如果包含某个 shape，就要求这个 shape 不会在运行时被 cudagraph padding 到另一个 shape。

检查位置：`code/vllm/vllm/v1/cudagraph_dispatcher.py:93`

原因：

```text
compile graph 的 static shape 和 CUDA graph replay shape 必须一致。
```

否则会出现：

```text
Inductor 编译了 size=8，
但 runtime 被 padding 到 size=16，
底层 compiled runnable 和 graph key 语义不一致。
```

---

## 5. _determine_batch_execution_and_padding() 做什么

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`

它是运行时 shape 归一的中心。

核心步骤：

```text
1. 根据 num_tokens、num_reqs、max_num_scheduled_tokens 判断 uniform_decode；
2. 统计当前 active LoRA；
3. 判断本轮是否有 encoder output；
4. 先调用 _pad_for_sequence_parallelism(num_tokens)；
5. 调 CudagraphDispatcher.dispatch()；
6. 如果 use_cascade_attn 或 has_encoder_output，禁用 FULL；
7. 如果 force_eager，强制 valid_modes={NONE}；
8. DP 场景下 coordinate_batch_across_dp()；
9. DP 协调后可能再次 dispatch；
10. 返回 cudagraph_mode、BatchDescriptor、should_ubatch、num_tokens_across_dp、cudagraph_stats。
```

输出会影响后续：

```text
num_tokens_padded = batch_descriptor.num_tokens
num_reqs_padded = batch_descriptor.num_reqs or num_reqs
pad_attn = cudagraph_mode == FULL
```

这些值决定：

```text
slot_mapping 是否 padded；
attention metadata 是否 padded；
ubatch_slices 是否使用 padded 版本；
ForwardContext.num_tokens；
CUDAGraphWrapper replay key。
```

---

## 6. sequence parallelism 的预 padding

`_pad_for_sequence_parallelism()` 定义在：

`code/vllm/vllm/v1/worker/gpu_model_runner.py:3407`

逻辑：

```text
if pass_config.enable_sp and tensor_parallel_size > 1:
  return round_up(num_scheduled_tokens, tensor_parallel_size)
return num_scheduled_tokens
```

原因：

```text
sequence parallelism 需要把 token 维切到 TP ranks 上，
因此 token 数要能被 TP size 对齐。
```

这一步发生在 cudagraph dispatcher 之前。

所以真实流程是：

```text
真实 num_tokens
  → SP padding 到 TP size 倍数
  → cudagraph padding 到 capture bucket
```

如果 spec decode 也开启，还要满足：

```text
capture size 是 uniform_decode_query_len 的倍数；
同时满足 SP / TP size 对齐。
```

相关逻辑：`code/vllm/vllm/config/compilation.py:1462`

---

## 7. DP padding 和 mode 协调

Data Parallel 下，每个 DP rank 本地 batch 可能不同。

例如：

```text
rank0：num_tokens = 17
rank1：num_tokens = 24
rank2：num_tokens = 3
```

如果各 rank 独立决定 graph mode，可能出现通信顺序不一致。

因此 `_determine_batch_execution_and_padding()` 在 DP 场景下会调用：

```text
coordinate_batch_across_dp(...)
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3882`

DP 协调会收集每个 rank 的：

```text
orig_num_tokens_per_ubatch
padded_num_tokens_per_ubatch
should_ubatch
cudagraph_mode
```

核心语义：

```text
如果任一 rank 是 NONE，整体 mode 可能降级到 NONE；
如果 cudagraph 或 ubatching 启用，各 rank 可能 pad 到相同 token 数；
协调后用 synced_cudagraph_mode 再 dispatch。
```

这样保证：

```text
所有 DP rank 的 graph mode、padding、communication shape 一致。
```

---

## 8. DBO / ubatching 的 padded slices

如果启用 DBO / microbatching，一个 batch 会被拆成多个 ubatch。

相关工具：`code/vllm/vllm/v1/worker/ubatch_utils.py`

`maybe_create_ubatch_slices()` 会创建两套 slices：

```text
ubatch_slices：
  按真实 token / request 切分。

ubatch_slices_padded：
  最后一个 ubatch 扩展到 padded token / request shape。
```

在 FULL graph 下：

```text
ubatch_slices_attn = ubatch_slices_padded
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4242`

原因：

```text
FULL graph 包含 attention；
attention metadata 和 ubatch slices 必须与 padded graph shape 对齐。
```

---

## 9. slot_mapping 如何 padding

`slot_mapping` 描述：

```text
本轮每个 token 要写入 KV cache 的哪个 slot。
```

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3960`

如果需要 padded token shape，`_get_slot_mappings()` 会把 padding token 的 slot 填为：

```text
-1
```

关键位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4008`

语义：

```text
padding token 不属于任何真实 request，不能写入真实 KV cache。
```

很多 KV update / concat_and_cache kernel 会把 `slot_mapping=-1` 视为跳过。

这非常关键：

```text
如果 padding token 写入了真实 KV cache，后续 decode 会读到污染的 key/value，输出可能漂移。
```

### 9.1 FULL graph 为什么更依赖 padded slot_mapping

在 `execute_model()` 中：

```text
pad_attn = cudagraph_mode == CUDAGraphMode.FULL
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4196`

如果 `pad_attn=True`，`_get_slot_mappings()` 会按：

```text
num_tokens_padded
num_reqs_padded
```

准备 slot mapping。

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4244`

### 9.2 dummy run 也填 -1

dummy run 没有真实 slot assignment，因此也会填 `-1`。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5821`

这样 capture / warmup 不会写入真实 KV cache。

---

## 10. block_table 如何 padding

`block_table` 描述：

```text
每个 request 的 logical block → physical KV block id 映射。
```

FULL graph 下，如果 request 数被 padding，padding request 的 block table 不能继承旧数据。

`_get_block_table()` 会把 padding 行填成：

```text
NULL_BLOCK_ID
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2249`

语义：

```text
padding request 不应该读取任何真实 KV block。
```

这和 `slot_mapping=-1` 是一对保护：

```text
slot_mapping=-1：防止 padding token 写 KV；
block_table=NULL_BLOCK_ID：防止 padding request 读 KV。
```

---

## 11. query_start_loc / seq_lens 如何 padding

attention backend 需要知道每个 request 的 token 边界和序列长度。

常见字段：

```text
query_start_loc
seq_lens
num_reqs
num_actual_tokens
max_query_len
max_seq_len
```

在 `_build_attention_metadata()` 中，FULL graph 下会使用 padded 长度。

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2208`

关键语义：

```text
query_start_loc = self.query_start_loc.gpu[:num_reqs_padded + 1]
seq_lens = self.seq_lens[:num_reqs_padded]
num_reqs = num_reqs_padded
num_actual_tokens = num_tokens_padded
positions = self.positions[:num_tokens_padded]
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2330`

注意：

```text
这里的 num_actual_tokens 在 FULL graph 下可能包含 padding token。
```

它表达的是 attention metadata 暴露给 kernel 的 fixed shape，而不是最终真实 output token 数。

真实输出仍然要靠 logits_indices / output slicing 过滤。

---

## 12. max_query_len 为什么重要

`max_query_len` 决定 attention backend 走哪类 routine。

```text
普通 decode：
  max_query_len = 1

spec decode 验证步：
  max_query_len = 1 + num_speculative_tokens

mixed prefill-decode：
  max_query_len = 本轮最长 prefill chunk
```

FULL graph 的 capture 和 replay 必须保持同一种 routine。

例如：

```text
不能用普通 decode max_query_len=1 的 graph，
去 replay mixed prefill max_query_len=128 的 batch。
```

因此 `BatchDescriptor.uniform` 和 `max_query_len` 是 shape 稳定性的关键组成部分。

---

## 13. input_ids / positions / inputs_embeds 如何使用 padded size

ModelRunner 内部通常维护持久 buffer，例如：

```text
input_ids
positions
inputs_embeds
```

真实请求只写前 `num_tokens_unpadded` 部分；如果需要 CUDA graph replay，forward 会按 `num_tokens_padded` 切片传入。

概念上是：

```text
input_ids_for_forward = input_ids[:num_tokens_padded]
positions_for_forward = positions[:num_tokens_padded]
inputs_embeds_for_forward = inputs_embeds[:num_tokens_padded]
```

padding 区域必须是安全值。

对于 text-only 路径：

```text
padding token 的 input id / position 不应该被采样输出消费；
attention / KV 写入由 slot_mapping=-1 等字段保护。
```

对于 multimodal / prompt_embeds 路径：

```text
inputs_embeds buffer 也必须与 padded num_tokens 对齐；
但 mm embedding gather 只针对真实 token window。
```

---

## 14. logits_indices 和 output 如何只消费真实 token

padding 后 forward 可能产生：

```text
hidden_states.shape[0] = num_tokens_padded
```

但真实 logits 只应该对 `logits_indices` 指定的位置计算。

在 `execute_model()` 中：

```text
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354`

`logits_indices` 在 `_prepare_inputs()` 中基于真实请求和采样需求构造，不应该指向 padding token。

因此：

```text
padding hidden states 可以存在，
但不会进入 logits / sampler / output。
```

最终 `sample_tokens()` 和 `ModelRunnerOutput` 也只按真实 request 输出。

这保证了：

```text
padding 改变执行形状，不改变用户可见输出。
```

---

## 15. FULL 和 PIECEWISE 对 padding 的要求不同

### 15.1 FULL graph

FULL graph 包含整个 model forward，通常包括 attention。

因此 FULL 下需要：

```text
- input_ids / positions / inputs_embeds 按 padded num_tokens；
- slot_mapping 按 padded num_tokens；
- block_table 按 padded num_reqs；
- query_start_loc / seq_lens 按 padded num_reqs；
- attention metadata 按 padded shape；
- ForwardContext.num_tokens = padded；
- BatchDescriptor = padded key。
```

### 15.2 PIECEWISE graph

PIECEWISE 通常只 capture compiled 子图，attention 可能在 graph 外。

因此它对 attention metadata padding 的要求低一些。

在真实执行中：

```text
pad_attn = cudagraph_mode == FULL
```

所以 PIECEWISE 下很多 attention metadata 可以按真实 unpadded shape 构造。

但 piecewise 子图自己的 compiled shape / graph key 仍可能需要 token size padding。

### 15.3 NONE

`CUDAGraphMode.NONE` 下，不需要为了 full graph 固定 attention metadata。

通常会按真实 shape 执行。

但仍可能存在：

```text
sequence parallelism padding；
DP / ubatching 需要的 padding；
compiled callable 的 shape range。
```

所以 NONE 不等于完全没有任何 padding，只是没有 CUDA graph replay 相关的 fixed bucket padding。

---

## 16. capture / dummy run 中的 padding

capture 阶段也走同样的 padding 逻辑。

`_dummy_run()` 会调用：

```text
_determine_batch_execution_and_padding(...)
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5762`

这保证：

```text
capture 时的 BatchDescriptor 和 runtime dispatch 生成的 BatchDescriptor 一致。
```

dummy run 还会：

```text
- 把 slot_mapping 填 -1；
- 构造 for_cudagraph_capture 的 attention metadata；
- 设置 set_forward_context(cudagraph_runtime_mode, batch_descriptor)；
- 触发 CUDAGraphWrapper capture。
```

如果 capture 和 runtime 使用不同 padding 规则，会导致：

```text
capture graph key 存在，runtime 却不能安全 replay。
```

---

## 17. padding waste 如何观测

如果开启：

```text
observability_config.cudagraph_metrics=True
```

`_determine_batch_execution_and_padding()` 会创建：

```text
CUDAGraphStat(
  num_unpadded_tokens=num_tokens,
  num_padded_tokens=batch_descriptor.num_tokens,
  num_paddings=batch_descriptor.num_tokens - num_tokens,
  runtime_mode=str(cudagraph_mode),
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3907`

这些统计会被 logger 聚合成表。

可以观察：

```text
Unpadded Tokens
Padded Tokens
Num Paddings
Runtime Mode
Count
```

如果出现：

```text
Unpadded Tokens = 33
Padded Tokens   = 128
Num Paddings    = 95
Runtime Mode    = FULL
```

说明 graph 命中了，但 padding waste 很大，性能未必更好。

调优方向：

```text
- 调整 cudagraph_capture_sizes；
- 增加常见 workload 附近的 bucket；
- 降低过大的 max capture size；
- 接受大 prefill eager、小 decode graph；
- 观察真实 num_tokens 分布。
```

---

## 18. padding 的正确性要求

padding 的核心目标是：

```text
改变执行形状，不改变语义。
```

必须满足：

```text
1. padding token 不写入 KV cache；
2. padding request 不读取真实 KV block；
3. padding token 不参与 logits 计算；
4. padding token 不进入 sampler；
5. padding request 不出现在 ModelRunnerOutput；
6. attention backend 对 padding metadata 的处理安全；
7. DP / TP / PP / MoE 通信 shape 不分叉；
8. capture 和 replay 的 tensor 地址和 shape 一致。
```

关键保护：

```text
slot_mapping padding = -1
block_table padding = NULL_BLOCK_ID
logits_indices only real tokens
request output only real req_ids
DP ranks synced mode / padding
```

---

## 19. 常见错误模式

### 19.1 padding token 污染 KV cache

症状：

```text
CUDA graph replay 后后续 token 输出异常；
长请求输出漂移；
偶发 illegal memory access。
```

优先检查：

```text
slot_mapping[num_tokens_unpadded:num_tokens_padded] 是否填 -1。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4008`

### 19.2 padding request 读到旧 block

症状：

```text
attention 输出异常；
只有 FULL graph 下复现；
换 eager 后正常。
```

优先检查：

```text
block_table[num_reqs:num_reqs_padded] 是否填 NULL_BLOCK_ID。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2249`

### 19.3 metadata padded / unpadded 混用

症状：

```text
shape mismatch；
attention backend assert；
spec decode 下 drafter metadata 错误。
```

检查：

```text
pad_attn 是否只在 FULL 下为 True；
_build_attention_metadata 是否传入正确 num_tokens_padded / num_reqs_padded；
spec_decode_common_attn_metadata 是否需要 unpadded。
```

### 19.4 logits_indices 指向 padding token

症状：

```text
采样出无效 token；
logprobs shape 对不上；
ModelRunnerOutput 多出请求或 token。
```

检查：

```text
_prepare_inputs() 生成 logits_indices 时是否只基于真实 scheduled tokens。
```

### 19.5 DP rank padding 不一致

症状：

```text
DP / MoE 场景 hang；
collective mismatch；
某些 rank graph replay，某些 rank eager。
```

检查：

```text
coordinate_batch_across_dp() 是否同步了 mode 和 token 数；
synced_cudagraph_mode 是否一致；
num_tokens_across_dp 是否传入 ForwardContext。
```

---

## 20. 几个典型场景

### 20.1 普通 decode，batch size 命中 capture size

```text
num_reqs = 16
num_tokens = 16
capture sizes 包含 16
uniform_decode = True
```

结果：

```text
BatchDescriptor.num_tokens = 16
num_paddings = 0
runtime_mode 可能 FULL
```

这是最理想的 graph replay 场景。

### 20.2 普通 decode，batch size 不命中但可 padding

```text
num_reqs = 13
num_tokens = 13
capture sizes = [1, 2, 4, 8, 16]
```

结果：

```text
BatchDescriptor.num_tokens = 16
num_paddings = 3
slot_mapping[13:16] = -1
block_table padding rows = NULL_BLOCK_ID
```

输出仍然只包含 13 个真实请求。

### 20.3 mixed prefill-decode

```text
num_tokens = 96
num_reqs = 12
uniform_decode = False
```

可能结果：

```text
FULL key 不匹配；
PIECEWISE key 命中 → runtime_mode=PIECEWISE；
attention metadata 多数按真实 shape；
compiled子图可能按 padded token size replay。
```

### 20.4 超出最大 capture size

```text
num_tokens = 4096
max_cudagraph_capture_size = 1024
```

结果：

```text
runtime_mode = NONE
BatchDescriptor.num_tokens = 4096
按真实 shape 执行，不为了 graph padding。
```

这是正常 fallback。

### 20.5 DP + MoE

```text
rank0 padded=32
rank1 padded=64
rank2 padded=16
```

DP 协调后可能：

```text
所有 rank pad 到 64；
或某个 rank NONE 导致所有 rank NONE。
```

目的：

```text
保证 MoE / DP communication shape 和 graph mode 一致。
```

### 20.6 spec decode

```text
num_speculative_tokens = 4
uniform_decode_query_len = 5
num_reqs = 8
num_tokens = 40
```

capture sizes 需要满足：

```text
是 5 的倍数；
如果 SP 开启，还要满足 TP size 对齐。
```

否则可能配置阶段报错或运行时 fallback。

---

## 21. 和 attention metadata 文档的关系

本篇关注 padding 的总原则和对象流。

更细的 attention backend 细节见：

```text
06_attention_metadata_capture.md
```

那里会展开：

```text
- CommonAttentionMetadata 字段；
- build_for_cudagraph_capture()；
- FlashAttention / TritonAttention / FlashInfer / MLA 的特殊处理；
- scheduler_metadata / wrapper / plan / workspace 的 shape 稳定性；
- spec decode metadata unpad；
- cascade attention 为什么禁用 FULL。
```

---

## 22. 最小伪代码

### 22.1 dispatch 和 padding

```text
num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
num_reqs = input_batch.num_reqs

cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, stats = \
  _determine_batch_execution_and_padding(
    num_tokens=num_tokens_unpadded,
    num_reqs=num_reqs,
    max_num_scheduled_tokens=max(...),
    use_cascade_attn=...,
    num_encoder_reqs=...,
  )

num_tokens_padded = batch_desc.num_tokens
num_reqs_padded = batch_desc.num_reqs or num_reqs
pad_attn = cudagraph_mode == FULL
```

### 22.2 slot mapping

```text
slot_mappings = _get_slot_mappings(
  num_tokens_padded = num_tokens_padded if pad_attn else num_tokens_unpadded,
  num_reqs_padded = num_reqs_padded if pad_attn else num_reqs,
  num_tokens_unpadded = num_tokens_unpadded,
)

slot_mapping[num_tokens_unpadded:num_tokens_padded] = -1
```

### 22.3 attention metadata

```text
attn_metadata = _build_attention_metadata(
  num_tokens = num_tokens_unpadded,
  num_tokens_padded = num_tokens_padded if pad_attn else None,
  num_reqs = num_reqs,
  num_reqs_padded = num_reqs_padded if pad_attn else None,
  max_query_len = max_num_scheduled_tokens,
  slot_mappings = slot_mappings_by_group,
)
```

### 22.4 forward 和 output

```text
with set_forward_context(
  attn_metadata,
  num_tokens=num_tokens_padded,
  cudagraph_runtime_mode=cudagraph_mode,
  batch_descriptor=batch_desc,
  slot_mapping=slot_mappings,
):
  hidden_states = self.model(
    input_ids[:num_tokens_padded],
    positions[:num_tokens_padded],
    ...,
  )

sample_hidden_states = hidden_states[logits_indices]  # only real positions
logits = compute_logits(sample_hidden_states)
sampler_output = sample(logits)                       # only real requests
```

---

## 23. 一句话总结

动态 batch padding 的核心不是“随便补几个 token”，而是：

```text
用 BatchDescriptor 把真实 batch 映射到已 capture 的固定 shape，
再让 input buffer、slot_mapping、block_table、attention metadata、ubatch slices 与这个 fixed shape 对齐，
同时通过 -1 slot、NULL block、真实 logits_indices 和 output slicing 保证 padding 不改变语义。
```

最小心智模型：

```text
真实 batch 决定语义，padded batch 决定 graph shape；
所有 padding 位置都必须对 KV、attention、logits、output 安全不可见。
```
