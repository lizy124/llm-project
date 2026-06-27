# 03. hidden states 如何变成 logits / logprobs？

源码位置：

- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu_input_batch.py`
- `vllm/vllm/model_executor/layers/logits_processor.py`
- `vllm/vllm/v1/sample/sampler.py`
- `vllm/vllm/v1/sample/metadata.py`
- `vllm/vllm/v1/outputs.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/engine/logprobs.py`
- `vllm/vllm/v1/engine/output_processor.py`
- `vllm/vllm/logprobs.py`
- `vllm/vllm/outputs.py`

本问题关注：模型 forward 产生 hidden states 后，哪些位置会被拿去算 logits；`logits_indices` 如何决定采样位置；`LogitsProcessor` 如何把 hidden states 变成 vocab logits；sample logprobs 和 prompt logprobs 分别在哪里计算、如何穿过 `ModelRunnerOutput / Scheduler / EngineCoreOutput / OutputProcessor`，最后进入 `RequestOutput / CompletionOutput`。

---

## 1. 一句话回答

不是所有 hidden states 都会算 logits。

vLLM V1 的生成路径是：

```text
GPUModelRunner._prepare_inputs()
  → 计算 logits_indices
  → model forward 得到 hidden_states
  → hidden_states[logits_indices]
  → model.compute_logits(...)
  → LogitsProcessor + lm_head 得到 logits
  → Sampler 消费 logits 采样 token
  → Sampler 可选计算 sample logprobs
  → GPUModelRunner 可选额外计算 prompt logprobs
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → OutputProcessor / LogprobsProcessor
  → RequestOutput / CompletionOutput
```

所以可以压缩成一句：

```text
logits 是 sampler 的输入；logprobs 是输出附加信息；二者都依赖 logits_indices 和请求的 sampling_params。
```

---

## 2. 先区分三种东西

### 2.1 logits

`logits` 是模型对词表中每个 token 的未归一化分数。

在 vLLM 中：

```text
hidden_states
  → lm_head / logits processor
  → logits: [num_sample_positions, vocab_size]
```

它主要给 sampler 使用。

### 2.2 sample logprobs

`sample logprobs` 是生成 token 位置的 log probability 信息。

用户请求中：

```text
sampling_params.logprobs
```

决定是否返回以及返回多少个 top logprobs。

它对应最终输出里的：

```text
CompletionOutput.logprobs
CompletionOutput.cumulative_logprob
```

### 2.3 prompt logprobs

`prompt_logprobs` 是 prompt token 自身在模型下的 log probability。

用户请求中：

```text
sampling_params.prompt_logprobs
```

决定是否计算。

它对应最终输出里的：

```text
RequestOutput.prompt_logprobs
```

注意：

```text
prompt_logprobs 不来自生成 token 的 sampled logits；
它是在 prefill hidden states 上额外 compute_logits，
用 prompt 第 i 个位置预测 prompt 第 i+1 个 token。
```

---

## 3. logits_indices 的意义

### 3.1 为什么需要 logits_indices

模型 forward 通常返回本轮所有 scheduled tokens 的 hidden states。

但生成模型并不一定需要对每个 token 都计算 vocab logits。

```text
普通 decode：
  每个 request 通常只需要最后一个 token 的 logits。

chunked prefill：
  部分 prefill token 只是为了填 KV cache，不一定要采样。

spec decode：
  需要 draft token / bonus token 对应的一组 logits。

pooling / embedding：
  可能根本不走 generation logits。

prompt logprobs：
  需要额外对 prompt hidden states 算 logits，但这是单独路径。
```

所以 `logits_indices` 的作用是：

```text
从本轮 hidden_states 中挑出真正需要 compute_logits 的 row。
```

### 3.2 普通非 spec decode 的 logits_indices

在 `GPUModelRunner._prepare_inputs()` 中：

位置：`vllm/v1/worker/gpu_model_runner.py:1889`

非 speculative decoding 时：

位置：`vllm/v1/worker/gpu_model_runner.py:2161` 到 `vllm/v1/worker/gpu_model_runner.py:2169`

```text
logits_indices = query_start_loc[1:] - 1
```

含义：

```text
query_start_loc 记录每个 request 本轮 query token 的起止位置；
query_start_loc[1:] - 1 就是每个 request 本轮最后一个 query token 的 hidden-state row。
```

这正好对应普通生成：

```text
用本轮最后一个 token 的 hidden state 预测下一个 token。
```

源码里也有一个 TODO：

```text
chunked prefills 中 partial requests 理论上不应该采样，
当前为了简单仍会采样，后续忽略这些 sampled tokens。
```

位置：`vllm/v1/worker/gpu_model_runner.py:2163` 到 `vllm/v1/worker/gpu_model_runner.py:2167`。

### 3.3 spec decode 的 logits_indices

如果本轮有 speculative decoding：

```text
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
```

则会构造 `spec_decode_metadata`：

位置：`vllm/v1/worker/gpu_model_runner.py:2171` 到 `vllm/v1/worker/gpu_model_runner.py:2195`

```text
spec_decode_metadata = self._calc_spec_decode_metadata(...)
logits_indices = spec_decode_metadata.logits_indices
num_sampled_tokens = num_draft_tokens + 1
```

含义：

```text
spec decode 不只是每个 request 取一个最后位置；
它需要为 draft tokens / bonus token 准备多个 logits 位置，
用于验证和采样。
```

---

## 4. hidden states 到 logits 的执行路径

### 4.1 GPUModelRunner 中选 hidden states

模型 forward 后，普通生成路径会执行：

位置：`vllm/v1/worker/gpu_model_runner.py:4357`

```text
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

如果是 pipeline parallel 非最后 rank：

```text
非 last PP rank 返回 IntermediateTensors，不算最终 logits。
last PP rank 才 compute_logits。
```

相关位置：

```text
非 last PP rank 返回 intermediate：vllm/v1/worker/gpu_model_runner.py:4340
last rank compute_logits：vllm/v1/worker/gpu_model_runner.py:4377
```

### 4.2 broadcast_pp_output 稀有路径

如果 `broadcast_pp_output=True`，最后 PP rank 算出 logits 后会通过 PP group broadcast 给其他 rank。

位置：`vllm/v1/worker/gpu_model_runner.py:4379` 到 `vllm/v1/worker/gpu_model_runner.py:4387`

```text
model_output_broadcast_data["logits"] = logits.contiguous()
broadcasted = get_pp_group().broadcast_tensor_dict(...)
logits = broadcasted["logits"]
```

普通情况下不走这个路径。

---

## 5. model.compute_logits 背后的 LogitsProcessor

`model.compute_logits()` 最终会走模型的 lm head / logits processor。

核心实现：`vllm/model_executor/layers/logits_processor.py`

### 5.1 LogitsProcessor 的职责

类定义位置：`vllm/model_executor/layers/logits_processor.py:18`

它做三件事：

```text
1. 用 lm_head 把 hidden states 投影到 vocab logits；
2. 在 TP 下 gather / all-gather vocab 分片；
3. 应用 soft cap / scale，并裁掉 padded vocab。
```

源码注释位置：`vllm/model_executor/layers/logits_processor.py:19` 到 `vllm/model_executor/layers/logits_processor.py:26`。

### 5.2 forward 主逻辑

位置：`vllm/model_executor/layers/logits_processor.py:54`

```text
if logits_as_input:
  logits = hidden_states
else:
  logits = self._get_logits(hidden_states, lm_head, embedding_bias)

if soft_cap is not None:
  logits = tanh(logits / soft_cap) * soft_cap

if scale != 1.0:
  logits *= scale
```

### 5.3 TP 下 logits 如何 gather

位置：`vllm/model_executor/layers/logits_processor.py:75`

```text
if self.use_all_gather:
  logits = tensor_model_parallel_all_gather(logits)
else:
  logits = tensor_model_parallel_gather(logits)
```

含义：

```text
lm_head 通常是 vocab parallel；
每个 TP rank 只算词表 shard 的 logits；
需要 gather 或 all-gather 得到完整 vocab logits。
```

`use_all_gather` 由平台决定，例如 TPU / XLA 严格 SPMD 场景可能需要所有 rank 都执行同样操作。

### 5.4 裁掉 vocab padding

位置：`vllm/model_executor/layers/logits_processor.py:101`

```text
logits = logits[..., : self.org_vocab_size]
```

原因：

```text
vocab parallel / padding 可能让实际 lm_head vocab size 大于原始 vocab size；
最终 logits 只保留原始 vocab。
```

---

## 6. 两类 LogitsProcessor 不要混淆

vLLM 里有两个容易混淆的名字。

### 6.1 model executor 层 LogitsProcessor

文件：

```text
vllm/model_executor/layers/logits_processor.py
```

职责：

```text
hidden states → vocab logits
```

### 6.2 sampling 阶段 logits processors

文件：

```text
vllm/v1/sample/logits_processor/
```

职责：

```text
在采样前修改 logits，例如：
  min tokens
  logit bias
  min-p
  allowed token ids
  bad words
  guided / structured output mask
  自定义 logits processor
```

这些 processor 是 sampler 的一部分，不负责从 hidden states 计算 lm_head logits。

---

## 7. SamplingMetadata 里和 logprobs 相关的字段

`SamplingMetadata` 定义位置：`vllm/v1/sample/metadata.py:14`

和 logprobs 直接相关的是：

```text
max_num_logprobs：
  None 表示不返回 logprobs；
  0 表示只返回 sampled token 的 logprob；
  N 表示返回 sampled token + top N logprobs；
  -1 在 sampler 中表示返回 full vocab logprobs。

logprob_token_ids：
  指定只计算某些 token ids 的 logprobs；
  用于 generative scoring 之类场景，避免 materialize full vocab logprobs。
```

位置：`vllm/v1/sample/metadata.py:25` 到 `vllm/v1/sample/metadata.py:49`。

这些字段来自 `InputBatch` 对请求 `sampling_params` 的整理。

例如：

位置：`vllm/v1/worker/gpu_input_batch.py:416` 到 `vllm/v1/worker/gpu_input_batch.py:425`

```text
sampling_params.logprobs
  → self.num_logprobs[req_id]

sampling_params.logprob_token_ids
  → self.logprob_token_ids[req_id]
```

---

## 8. Sampler 如何计算 sample logprobs

核心文件：`vllm/v1/sample/sampler.py`

### 8.1 Sampler.forward 的顺序

`Sampler` 类定义位置：`vllm/v1/sample/sampler.py:20`

源码注释已经列出完整顺序，位置：`vllm/v1/sample/sampler.py:20` 到 `vllm/v1/sample/sampler.py:59`。

和 logprobs 最相关的是：

```text
1. 如果请求 logprobs：
   在 logits 还没有被 penalties / temperature / top-p/top-k 改写前，
   先计算 raw logprobs 或 raw logits。

2. logits 转 float32。

3. 应用 allowed_token_ids、bad words、logits processors、penalties。

4. 执行采样。

5. 如果需要 logprobs，收集 sampled token 和 top-k token 的 logprobs。
```

### 8.2 raw logprobs 在采样前计算

位置：`vllm/v1/sample/sampler.py:72`

关键逻辑：

```text
num_logprobs = sampling_metadata.max_num_logprobs

if num_logprobs is not None or sampling_metadata.logprob_token_ids:
  if logprobs_mode == "raw_logprobs":
    raw_logprobs = logits.log_softmax(...)
  elif logprobs_mode == "raw_logits":
    raw_logprobs = logits.clone() / logits.to(float32)
```

对应位置：`vllm/v1/sample/sampler.py:84` 到 `vllm/v1/sample/sampler.py:94`。

这里有个重要点：

```text
V1 sampler 默认用原始 logits 的 logprobs 作为返回 logprobs，
不同于 V0 使用采样后经过 penalties / temperature 的 logits。
```

源码注释位置：`vllm/v1/sample/sampler.py:80` 到 `vllm/v1/sample/sampler.py:83`。

### 8.3 compute_logprobs

位置：`vllm/v1/sample/sampler.py:304`

```text
logits.log_softmax(dim=-1, dtype=torch.float32)
```

这一步得到完整 vocab 的 logprobs tensor。

### 8.4 gather_logprobs

位置：`vllm/v1/sample/sampler.py:309`

输入：

```text
logprobs: [num_positions, vocab]
num_logprobs: 每个位置保留多少 top logprobs
token_ids: sampled token ids 或 prompt target token ids
```

输出：

```text
LogprobsTensors(
  logprob_token_ids: [num_positions, num_logprobs + 1],
  logprobs: [num_positions, num_logprobs + 1],
  selected_token_ranks: [num_positions],
)
```

实现细节：

```text
topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs)
token_logprobs = logprobs.gather(-1, token_ids)
token_ranks = batched_count_greater_than(logprobs, token_logprobs)
indices = cat(sampled_or_prompt_token_id, topk_indices)
logprobs = cat(sampled_or_prompt_token_logprob, topk_logprobs)
```

位置：`vllm/v1/sample/sampler.py:333` 到 `vllm/v1/sample/sampler.py:356`。

所以第 0 列有特殊含义：

```text
第 0 列永远是实际 sampled token 或 prompt target token；
后续列才是 top-k token。
```

这个约定在 engine logprobs processor 中也会用到。

### 8.5 logprob_token_ids 特殊路径

如果请求指定了 `logprob_token_ids`，sampler 会调用：

```text
gather_specific_token_logprobs(...)
```

位置：`vllm/v1/sample/sampler.py:151`

作用：

```text
只 gather 指定 token ids 的 logprob，
避免为返回少量特定 token 的分数而处理 top-k/full vocab 输出。
```

---

## 9. LogprobsTensors / LogprobsLists / Python Logprob 容器

### 9.1 Worker / Scheduler 之间的 tensor 和 list 容器

定义文件：`vllm/v1/outputs.py`

`LogprobsTensors`：位置 `vllm/v1/outputs.py:52`

```text
GPU / CPU tensor 侧容器：
  logprob_token_ids
  logprobs
  selected_token_ranks
  cu_num_generated_tokens
```

`LogprobsLists`：位置 `vllm/v1/outputs.py:27`

```text
CPU / numpy 侧容器：
  logprob_token_ids
  logprobs
  sampled_token_ranks
  cu_num_generated_tokens
```

为什么有两套？

```text
sampler 先在 GPU 上产生 LogprobsTensors；
ModelRunnerOutput 要跨进程 / 交给 Scheduler，torch.Tensor 序列化贵；
所以 sample logprobs 通常会转成 numpy-backed LogprobsLists。
```

`LogprobsTensors.tolists()` 位置：`vllm/v1/outputs.py:62`。

### 9.2 最终 RequestOutput 使用的 Python 容器

文件：`vllm/logprobs.py`

核心结构：

```text
Logprob：
  单个 token 的 logprob、rank、decoded_token。

PromptLogprobs：
  prompt 每个位置的 logprobs。

SampleLogprobs：
  生成 token 每个位置的 logprobs。

FlatLogprobs：
  扁平化存储，减少大量 dict/list 对象带来的 GC 开销。
```

`Logprob` 定义位置：`vllm/logprobs.py:13`。

`PromptLogprobs / SampleLogprobs` 定义位置：`vllm/logprobs.py:155`。

`create_prompt_logprobs()` 位置：`vllm/logprobs.py:162`。

这里有一个重要约定：

```text
prompt 第一个 token 没有上文，logprob 固定为 None。
```

位置：`vllm/logprobs.py:165` 到 `vllm/logprobs.py:166`。

`append_logprobs_for_next_position()` 位置：`vllm/logprobs.py:175`。

它同样遵守：

```text
第一个 token id / logprob 是实际 sampled token 或 prompt token，
后面才是 top-k token。
```

---

## 10. prompt_logprobs 如何计算

prompt logprobs 不是 sampler 的普通 sample logprobs。

它在 `GPUModelRunner` 中有专门路径：

```text
_get_prompt_logprobs_dict(hidden_states, num_scheduled_tokens)
```

位置：`vllm/v1/worker/gpu_model_runner.py:5461`。

### 10.1 什么时候计算

函数开头先看：

```text
self.num_prompt_logprobs
```

如果为空，直接返回 `{}`。

位置：`vllm/v1/worker/gpu_model_runner.py:5466` 到 `vllm/v1/worker/gpu_model_runner.py:5468`。

`self.num_prompt_logprobs` 来自新请求的：

```text
sampling_params.prompt_logprobs
```

### 10.2 chunked prefill 下如何处理

prompt logprobs 可能跨多个 prefill chunk 逐步计算。

源码会在 request 上维护：

```text
request.in_progress_prompt_logprobs_cpu
```

如果还没有，就创建完整 prompt 长度的 CPU `LogprobsTensors.empty_cpu(...)`。

位置：`vllm/v1/worker/gpu_model_runner.py:5492` 到 `vllm/v1/worker/gpu_model_runner.py:5501`。

每个 chunk 只填当前 chunk 对应 slice。

### 10.3 用哪个 hidden state 预测哪个 prompt token

核心逻辑：

位置：`vllm/v1/worker/gpu_model_runner.py:5524` 到 `vllm/v1/worker/gpu_model_runner.py:5541`

```text
prompt_hidden_states = hidden_states[offset : offset + num_logits]
logits = self.model.compute_logits(prompt_hidden_states)

tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]
logprobs = self.sampler.compute_logprobs(logits)
token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
  logprobs,
  num_prompt_logprobs,
  tgt_token_ids,
)
```

含义：

```text
prompt 位置 i 的 hidden state 用来预测 prompt 位置 i+1 的 token；
所以 target token 是 shifted prompt token。
```

这也是为什么最终 prompt 第一个 token 的 logprob 是 None。

### 10.4 prompt logprobs 如何返回

每个 chunk 的结果会 copy 到 CPU tensor：

位置：`vllm/v1/worker/gpu_model_runner.py:5543` 到 `vllm/v1/worker/gpu_model_runner.py:5551`

如果某个 request 完成 prefill，会放入：

```text
prompt_logprobs_dict[req_id] = logprobs_tensors
```

位置：`vllm/v1/worker/gpu_model_runner.py:5513` 到 `vllm/v1/worker/gpu_model_runner.py:5517`。

最后如果有 prompt logprobs，要同步 GPU→CPU 拷贝：

位置：`vllm/v1/worker/gpu_model_runner.py:5559` 到 `vllm/v1/worker/gpu_model_runner.py:5561`。

---

## 11. ModelRunnerOutput 中如何承载 logprobs

定义位置：`vllm/v1/outputs.py:233`

相关字段：

```text
sampled_token_ids：
  本轮每个 request 生成的 token ids。

logprobs：
  sample logprobs，类型是 LogprobsLists | None。

prompt_logprobs_dict：
  req_id -> LogprobsTensors | None。

num_nans_in_logits：
  req_id -> logits 中 NaN 数量，用于诊断。
```

位置：`vllm/v1/outputs.py:240` 到 `vllm/v1/outputs.py:267`。

为什么 sample logprobs 是 `LogprobsLists`，prompt logprobs 还是 `LogprobsTensors`？

```text
sample logprobs 是按本轮 generated tokens 的 batch 输出，通常在 worker bookkeeping 中转成 numpy/list 形式给 scheduler 切片。

prompt logprobs 是按 req_id 存放的完整 prompt tensor，可能跨 chunk 累积，完成 prefill 后按 request 交给 scheduler。
```

---

## 12. Scheduler 如何把 batch 级 logprobs 拆回 request

`EngineCore.step()` 会执行：

```text
scheduler.update_from_output(scheduler_output, model_output)
```

位置：`vllm/v1/engine/core.py:504`。

Scheduler 中处理 logprobs 的关键位置：`vllm/v1/core/sched/scheduler.py:1464`

### 12.1 取 batch 级输出

Scheduler 从 `ModelRunnerOutput` 取：

```text
sampled_token_ids
logprobs
prompt_logprobs_dict
```

位置：`vllm/v1/core/sched/scheduler.py:1469` 到 `vllm/v1/core/sched/scheduler.py:1471`。

### 12.2 sample logprobs 按 request 切片

如果请求需要 logprobs：

位置：`vllm/v1/core/sched/scheduler.py:1670` 到 `vllm/v1/core/sched/scheduler.py:1676`

```text
new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

这里 `len(new_token_ids)` 很重要：

```text
spec decode / jump decoding 下一个 request 本轮可能产生多个 token；
不同 request 新增 token 数也可能不同；
所以 LogprobsLists 支持 cu_num_generated_tokens 来定位每个 request 的 slice。
```

`slice_request()` 定义位置：`vllm/v1/outputs.py:40`。

### 12.3 prompt logprobs 按 req_id 取

位置：`vllm/v1/core/sched/scheduler.py:1681` 到 `vllm/v1/core/sched/scheduler.py:1682`

```text
prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
```

### 12.4 放入 EngineCoreOutput

位置：`vllm/v1/core/sched/scheduler.py:1691` 到 `vllm/v1/core/sched/scheduler.py:1697`

```text
EngineCoreOutput(
  new_token_ids=new_token_ids,
  new_logprobs=new_logprobs,
  new_prompt_logprobs_tensors=prompt_logprobs_tensors,
  ...
)
```

至此，logprobs 从 worker batch 输出拆回了 request 粒度。

---

## 13. OutputProcessor / LogprobsProcessor 如何形成最终输出

### 13.1 OutputProcessor 调用 LogprobsProcessor

`OutputProcessor.process_outputs()` 中：

位置：`vllm/v1/engine/output_processor.py:646` 到 `vllm/v1/engine/output_processor.py:648`

```text
req_state.logprobs_processor.update_from_output(engine_core_output)
```

之后才创建 RequestOutput：

位置：`vllm/v1/engine/output_processor.py:650`。

### 13.2 LogprobsProcessor 初始化

`LogprobsProcessor.from_new_request()` 位置：`vllm/v1/engine/logprobs.py:42`

它根据：

```text
sampling_params.num_logprobs
sampling_params.prompt_logprobs
sampling_params.flat_logprobs
```

创建：

```text
self.logprobs
self.prompt_logprobs
self.cumulative_logprob
```

位置：`vllm/v1/engine/logprobs.py:48` 到 `vllm/v1/engine/logprobs.py:67`。

### 13.3 更新 sample logprobs

`_update_sample_logprobs()` 位置：`vllm/v1/engine/logprobs.py:69`

它会：

```text
1. 遍历 EngineCoreOutput.new_logprobs 的每个 generated position。
2. 第 0 列视为 sampled token logprob。
3. sampled token logprob 累加到 cumulative_logprob。
4. 调 append_logprobs_for_next_position() 追加到 request 的 logprobs 容器。
5. 如有 tokenizer，顺便把 token ids 转 decoded_token，并修正 UTF-8 边界。
```

第 0 列 sampled token 约定的位置：`vllm/v1/engine/logprobs.py:107`。

累加 cumulative logprob 的位置：`vllm/v1/engine/logprobs.py:109`。

### 13.4 更新 prompt logprobs

`_update_prompt_logprobs()` 位置：`vllm/v1/engine/logprobs.py:121`

它会：

```text
1. 把 LogprobsTensors 转成 Python list。
2. 遍历每个 prompt position。
3. 调 append_logprobs_for_next_position() 写入 prompt_logprobs。
4. 处理 tokenizer decoded token 和 UTF-8 修正。
```

追加位置：`vllm/v1/engine/logprobs.py:179` 到 `vllm/v1/engine/logprobs.py:187`。

### 13.5 update_from_output 总入口

位置：`vllm/v1/engine/logprobs.py:348`

```text
if output.new_logprobs is not None:
  self._update_sample_logprobs(output.new_logprobs)

if output.new_prompt_logprobs_tensors is not None:
  self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
```

---

## 14. RequestOutput / CompletionOutput 如何带上 logprobs

### 14.1 CompletionOutput

定义位置：`vllm/outputs.py:21`

字段：

```text
cumulative_logprob
logprobs
```

位置：`vllm/outputs.py:40` 到 `vllm/outputs.py:44`。

`RequestState._new_completion_output()` 构造它：

位置：`vllm/v1/engine/output_processor.py:376`

```text
logprobs = self.logprobs_processor.logprobs
if delta and logprobs:
  logprobs = logprobs[-len(token_ids):]

CompletionOutput(
  token_ids=token_ids,
  logprobs=logprobs,
  cumulative_logprob=self.logprobs_processor.cumulative_logprob,
  ...
)
```

对应位置：`vllm/v1/engine/output_processor.py:392` 到 `vllm/v1/engine/output_processor.py:408`。

### 14.2 RequestOutput

定义位置：`vllm/outputs.py:85`

字段：

```text
prompt_logprobs
outputs: list[CompletionOutput]
```

位置：`vllm/outputs.py:109` 到 `vllm/outputs.py:136`。

`RequestState._new_request_output()` 构造它：

位置：`vllm/v1/engine/output_processor.py:333`

关键逻辑：

```text
if output_kind == DELTA:
  prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
else:
  prompt_logprobs = self.logprobs_processor.prompt_logprobs

RequestOutput(prompt_logprobs=prompt_logprobs, outputs=...)
```

位置：`vllm/v1/engine/output_processor.py:356` 到 `vllm/v1/engine/output_processor.py:368`。

### 14.3 DELTA 模式下 prompt_logprobs 只输出一次

`pop_prompt_logprobs()` 位置：`vllm/v1/engine/logprobs.py:189`

源码注释说明：

```text
DELTA 语义下，prompt logprobs 在 prefill 结束时一次性返回，
返回后 LogprobsProcessor 会忘记它们。
```

位置：`vllm/v1/engine/logprobs.py:189` 到 `vllm/v1/engine/logprobs.py:206`。

---

## 15. 容易混淆点

### 15.1 logits 不是最终输出

```text
logits 是 sampler 输入；
一般不会直接出现在 RequestOutput 中。
```

用户看到的是：

```text
token_ids
text
logprobs
prompt_logprobs
cumulative_logprob
```

### 15.2 logprobs 不是 sampler 必须字段

如果：

```text
sampling_params.logprobs is None
sampling_metadata.logprob_token_ids is None
```

sampler 可以不计算 logprobs。

### 15.3 prompt_logprobs 和 sample logprobs 位置不同

```text
sample logprobs：
  生成 token 位置，来自 sampler 对 sampled logits 的处理。

prompt_logprobs：
  prompt token 位置，来自 prefill hidden states 额外 compute_logits。
```

### 15.4 第 0 列不是 top-1，而是实际 token

`gather_logprobs()` 返回时：

```text
第 0 列：实际 sampled token 或 prompt target token；
后续列：top-k token。
```

如果实际 token 正好在 top-k 里，最终 Python dict 会自然去重。

### 15.5 raw logprobs 默认基于采样前 logits

V1 sampler 默认在 penalties / temperature / top-p/top-k 前计算 raw logprobs。

这和“采样使用的处理后概率分布”不是一回事。

### 15.6 spec decode 下每个 request 可能有多个 logprobs 位置

spec decode 可能一次接受多个 token。

因此：

```text
ModelRunnerOutput.sampled_token_ids 是 list[list[int]]；
LogprobsLists 支持 cu_num_generated_tokens；
Scheduler.slice_request(req_index, len(new_token_ids)) 用新增 token 数切片。
```

---

## 16. 最小源码阅读路线

```text
1. vllm/v1/worker/gpu_model_runner.py
   看 _prepare_inputs() 如何生成 logits_indices，forward 后如何 compute_logits。

2. vllm/model_executor/layers/logits_processor.py
   看 hidden_states 如何通过 lm_head、TP gather、soft cap / scale 变成 logits。

3. vllm/v1/sample/metadata.py
   看 SamplingMetadata 中 max_num_logprobs / logprob_token_ids。

4. vllm/v1/sample/sampler.py
   看 raw_logprobs、sample、gather_logprobs。

5. vllm/v1/outputs.py
   看 LogprobsTensors / LogprobsLists / ModelRunnerOutput。

6. vllm/v1/worker/gpu_model_runner.py 的 _get_prompt_logprobs_dict()
   看 prompt_logprobs 如何额外计算。

7. vllm/v1/core/sched/scheduler.py
   看 Scheduler 如何把 batch logprobs 拆回 request。

8. vllm/v1/engine/logprobs.py
   看 LogprobsProcessor 如何累计 sample / prompt logprobs。

9. vllm/v1/engine/output_processor.py
   看 RequestOutput / CompletionOutput 如何组装。

10. vllm/logprobs.py 和 vllm/outputs.py
    看最终用户可见的数据结构。
```

---

## 17. 一句话总结

```text
logits_indices 决定哪些 hidden states 会进入 lm_head；
LogitsProcessor 负责 hidden states → vocab logits；
Sampler 负责 token 采样和 sample logprobs；
prompt_logprobs 是 prefill hidden states 的额外评分路径；
Scheduler 把 batch 级 logprobs 拆回 request；
LogprobsProcessor 把 tensor/numpy 结果变成最终 RequestOutput / CompletionOutput 中的 Python logprobs。
```
