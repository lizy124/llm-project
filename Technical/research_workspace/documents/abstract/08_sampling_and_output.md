# 08 sampling_and_output 背诵文档

## 1. 专题定位

`sampling_and_output` 讲的是模型 forward 之后，vLLM 如何从 hidden states / logits 走到用户可见输出。

它连接三个层次：

```text
ModelRunner / Sampler：从 logits 采样 token。
Scheduler：把 token 回写到 Request 状态和资源账本。
OutputProcessor：把内部 token 输出变成用户可见 RequestOutput。
```

一句话：

```text
Sampling / Output 负责把模型内部计算结果变成上层可见输出。
```

## 2. 最小心智模型

主链路：

```text
hidden states
  → logits
  → grammar bitmask / logits processors
  → sampler / rejection sampler
  → sampled_token_ids / logprobs / prompt_logprobs
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / CompletionOutput / PoolingRequestOutput
```

要背住：

```text
Sampling 解决“选哪个 token”；Scheduler update 解决“这个 token 如何改变请求状态”；OutputProcessor 解决“这个状态如何展示给用户”。
```

## 3. 三层边界

### ModelRunner / Sampler 层

输入：

```text
hidden states
logits
InputBatch
sampling state
grammar bitmask
spec decode metadata
```

输出：

```text
sampled_token_ids
logprobs
prompt_logprobs
pooler_output
connector output
routed experts
```

它负责：

```text
计算 logits
应用 grammar bitmask
应用 SamplingParams
temperature / top-k / top-p / min-p / penalties
spec decode rejection sampling
sample logprobs
prompt logprobs
GPU → CPU 异步拷贝
构造 ModelRunnerOutput
```

它不负责 detokenize，也不负责释放 KV block。

### Scheduler update 层

输入：

```text
SchedulerOutput + ModelRunnerOutput
```

输出：

```text
更新后的 Request 状态 + EngineCoreOutputs
```

它负责：

```text
把 sampled token append 到 request
检查 stop
处理 spec decode accepted / rejected
推进 grammar 状态
切分 logprobs / prompt_logprobs
处理 pooling output
释放 KV blocks / encoder cache
构造 EngineCoreOutput
```

它不重新采样，不 detokenize。

### OutputProcessor 层

输入：

```text
EngineCoreOutputs
frontend RequestState
tokenizer
output_kind
```

输出：

```text
RequestOutput
CompletionOutput
PoolingRequestOutput
streaming queue item
```

它负责：

```text
detokenize
stop string 检查
DELTA / CUMULATIVE / FINAL_ONLY
logprobs 格式化
finish_reason / stop_reason
异步 queue 推送
同步 list 返回
frontend RequestState 清理
```

## 4. 总体流程图

完整链路：

```text
用户输入 SamplingParams
  → InputProcessor / EngineCoreRequest 保存参数
  → Scheduler 持有 request.sampling_params
  → SchedulerOutput 告诉 Worker 本轮跑哪些 token
  → ModelRunner 把 SamplingParams 写入 sampler state
  → model forward 得到 hidden_states
  → hidden_states[logits_indices]
  → model.compute_logits()
  → grammar bitmask 屏蔽非法 token
  → Sampler 或 RejectionSampler
  → sampled_token_ids
  → sample logprobs
  → prompt logprobs
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / CompletionOutput
```

EngineCore 中的关键连接点：

```text
scheduler_output = scheduler.schedule()
future = model_executor.execute_model(scheduler_output)
grammar_output = scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = model_executor.sample_tokens(grammar_output)
engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)
```

## 5. 为什么要分层

### ModelRunner 适合采样

因为它持有：

```text
hidden states / logits
InputBatch
request state GPU buffer
sampler state
speculator / rejection sampler
prompt logprobs worker
structured output worker
GPU stream / copy stream
```

采样放在 GPU 侧可以避免把大量 logits 搬到 Scheduler。

### Scheduler 适合状态对账

因为它知道：

```text
请求是否 running / waiting / finished
本轮 SchedulerOutput 原计划
KV blocks / encoder cache 状态
spec decode 调度了哪些 draft tokens
stop 后如何释放资源
EngineCoreOutputs 如何按 client_index 分组
```

### OutputProcessor 适合用户输出

因为它靠近 frontend，持有：

```text
tokenizer
RequestState
output_kind
streaming queue
external request id
```

## 6. SamplingParams 是源头

`SamplingParams` 是用户采样参数。

核心字段：

```text
presence_penalty
frequency_penalty
repetition_penalty
temperature
top_p
top_k
min_p
seed
stop / stop_token_ids
ignore_eos
max_tokens / min_tokens
logprobs
prompt_logprobs
logprob_token_ids
detokenize / output_kind
structured_outputs
logit_bias
allowed_token_ids
bad_words
repetition_detection
```

它会做规范化：

```text
seed == -1 → None
logprobs=True → 1
prompt_logprobs=True → 1
temperature 接近 0 → greedy
top_p / top_k / min_p / penalty 做范围校验
```

## 7. SamplingParams 如何进入 sampler

新 GPU sampler 路径：

```text
SamplingParams
  → Sampler.add_request(req_index, prompt_len, sampling_params)
  → SamplingStates
  → PenaltiesState
  → LogitBiasState
  → BadWordsState
  → LogprobTokenIdsState
```

这些状态把请求级参数变成 batch/GPU 可消费形态。

旧路径：

```text
SamplingParams
  → SamplingMetadata
  → Sampler.forward()
```

目标相同：

```text
把每个 request 的采样规则应用到当前 batch 的 logits 上。
```

## 8. logits 如何计算

ModelRunner 不一定对所有 hidden states 计算 logits。

通常只取：

```text
hidden_states[input_batch.logits_indices]
  → model.compute_logits()
```

原因：

```text
prefill 可能只需要最后位置 logits。
chunked prefill 的非最后 chunk 可能不输出 token。
decode 通常每个 request 一个 logits。
spec decode 需要多个 target / bonus logits。
prompt_logprobs 可能需要额外 prompt 位置 logits。
```

一句话：

```text
logits_indices 决定哪些 hidden states 需要进入 lm_head。
```

## 9. sample logprobs 和 prompt logprobs

### sample logprobs

sample logprobs 绑定生成 token。

它根据：

```text
logprobs
logprob_token_ids
logprobs_mode
```

返回 sampled token、top-k token 或指定 token 的 logprobs。

### prompt logprobs

prompt logprobs 是 prompt token 在上下文中的概率。

它不是 sampled token 的 logprobs。

由 PromptLogprobsWorker 根据 prompt 位置和 hidden states 计算。

要区分：

```text
sample logprobs：针对新生成 token。
prompt logprobs：针对 prompt token。
```

## 10. Sampler 主流程

Sampler 输入：

```text
logits
InputBatch
sampling states
```

处理顺序：

```text
raw logits
  → copy 成 FP32 processed logits
  → logit_bias / allowed_token_ids / min_tokens
  → repetition / frequency / presence penalties
  → bad words mask
  → temperature
  → min_p
  → top_k / top_p
  → greedy 或 random sample
```

背诵理解：

```text
约束类 mask / bias 先决定候选空间；penalty 根据历史调整分数；temperature / min_p / top-k / top-p 最后塑造采样分布。
```

## 11. greedy 和 random

当 `temperature == 0`：

```text
不加随机噪声，argmax，就是 greedy。
```

当 `temperature > 0`：

```text
按处理后的分布随机采样。
```

有些路径用 Gumbel-max 统一 greedy 和 random：

```text
greedy：无 gumbel noise。
random：加 gumbel noise 再 argmax。
```

## 12. structured output / grammar bitmask

结构化输出不是采样后修正。

它在采样前限制 logits。

流程：

```text
Scheduler.get_grammar_bitmask(scheduler_output)
  → GrammarOutput
  → ModelRunner.apply_grammar_bitmask(logits)
  → 不允许 token 的 logits = -inf
  → Sampler
```

采样后：

```text
Scheduler.update_from_output()
  → grammar.accept_tokens(req_id, new_token_ids)
```

边界：

```text
Scheduler 负责 grammar FSM 状态。
ModelRunner 负责把 grammar bitmask 应用到 logits。
```

## 13. spec decode 对 sampling 的影响

普通 decode：

```text
target logits
  → Sampler
  → 1 个 token
```

spec decode：

```text
draft tokens + draft probs
target model logits
  → RejectionSampler
  → accepted tokens / recovered token / bonus token
```

术语：

```text
accepted tokens：被接受的 draft token 前缀。
recovered token：拒绝后从修正分布采出的替代 token。
bonus token：全部 draft 接受后额外补的 token。
```

所以 spec decode 下：

```text
一个 request 一轮可能输出多个 token。
```

## 14. RejectionSampler 输出

RejectionSampler 返回的 sampled token 是真实要提交给 Scheduler 的 token。

不是原始 draft token。

输出规则：

```text
从左到右验证 draft tokens：
  接受：输出该 draft token，继续验证下一个。
  拒绝：输出 recovered token，后续 draft 不再输出。
  全部接受：输出所有 draft tokens，再追加 bonus token。
```

Scheduler 再根据原本 scheduled 的 draft tokens 和返回 token 数，计算 accepted / rejected。

## 15. ModelRunnerOutput 是什么

`ModelRunnerOutput` 是 Worker 返回 Scheduler 的 batch 级内部协议对象。

字段：

```text
req_ids：当前 batch request id 顺序。
req_id_to_index：request id → batch index。
sampled_token_ids：每个 request 本轮生成的 token ids。
logprobs：生成 token 的 logprobs。
prompt_logprobs_dict：prompt token 的 logprobs。
pooler_output：pooling / embedding / classify / reward 输出。
kv_connector_output：KV transfer 输出。
ec_connector_output：encoder cache 输出。
num_nans_in_logits：logits NaN 统计。
cudagraph_stats：CUDA graph 统计。
routed_experts：MoE routed experts 信息。
```

它不是用户输出。

原因：

```text
token ids 还没 detokenize。
stop string 还没检查。
logprobs 还不是用户 API 格式。
streaming delta 还没切分。
RequestOutput 还没构造。
```

## 16. AsyncOutput / GPU 到 CPU 拷贝

采样后很多结果仍在 GPU：

```text
sampled_token_ids
logprobs tensors
prompt_logprobs tensors
num_nans
pooler_output
routed_experts
```

为了不阻塞主计算流，新路径使用异步输出拷贝：

```text
main stream 计算完成
copy stream 等待 main stream
异步拷贝 sampled_token_ids / logprobs / prompt_logprobs
记录 copy_event
get_output() 等待 copy_event 并转成 CPU / Python 结构
```

一句话：

```text
AsyncOutput 用 copy stream 把 GPU 采样结果异步转成 Scheduler 可消费的 CPU 结构。
```

## 17. Scheduler.update_from_output 做什么

Scheduler 消费：

```text
SchedulerOutput + ModelRunnerOutput
```

处理：

```text
根据 req_id_to_index 找输出
取 sampled_token_ids
处理 spec decode accepted / rejected
append output token ids
检查 stop condition
推进 grammar
处理 logprobs / prompt_logprobs
处理 pooling output
处理 KV / EC connector output
释放 finished request resources
构造 EngineCoreOutput
```

一句话：

```text
Scheduler 把 worker 的 batch 输出重新对回 request 状态机。
```

## 18. stop 条件在哪里处理

token-id 级 stop 在 Scheduler。

例如：

```text
EOS
stop_token_ids
max_tokens
max_model_len
repetition detection
```

文本级 stop string 在 OutputProcessor。

因为它需要 detokenized text。

如果 OutputProcessor 检测到 stop string，但 Scheduler 还没结束请求：

```text
OutputProcessor 返回 reqs_to_abort
Engine 通知 EngineCore / Scheduler abort
```

## 19. OutputProcessor 做什么

OutputProcessor 输入：

```text
EngineCoreOutputs
```

它维护 frontend `RequestState`，并做：

```text
detokenizer.update(new_token_ids)
stop string check
logprobs_processor.update_from_output()
pooling output 包装
CompletionOutput 构造
RequestOutput 构造
streaming queue 推送
finished request 清理
```

输出：

```text
RequestOutput
PoolingRequestOutput
```

## 20. streaming 输出

用户可见输出有几种模式：

```text
DELTA：只返回本轮新增文本 / token。
CUMULATIVE：返回从开始到当前的累计输出。
FINAL_ONLY：只在结束时返回最终结果。
```

同步路径：

```text
OutputProcessor.process_outputs()
  → 返回 request_outputs list
```

异步路径：

```text
OutputProcessor.process_outputs()
  → queue.put(request_output)
  → AsyncLLM.generate() yield
```

## 21. Pooling / embedding / rerank 输出

非 generation 模型不走 token sampling。

典型链路：

```text
model forward
  → pooler_output
  → ModelRunnerOutput.pooler_output
  → Scheduler.update_from_output()
  → EngineCoreOutput.pooling_output
  → OutputProcessor
  → PoolingRequestOutput
```

区别：

```text
generation 输出 token ids。
pooling 输出 embedding / score / pooled tensor。
```

## 22. logprobs 的三个层级

要区分：

```text
logits：模型输出的 vocab 分数，仍在 worker / GPU 侧。
logprobs tensors：sampler / prompt worker 计算出的概率信息。
用户 logprobs：OutputProcessor 格式化后的 API 输出。
```

不要把 logits 当成用户 logprobs。

## 23. grammar、sampler、Scheduler 的边界

```text
Scheduler：生成 grammar bitmask，推进 grammar 状态。
ModelRunner：把 bitmask 应用到 logits。
Sampler：在被 mask 后的 logits 上采样。
Scheduler：接收采样结果后 grammar.accept_tokens。
```

这样 grammar 状态只保存在 Scheduler 侧，而 logits 过滤发生在 GPU 侧。

## 24. 常见易混点

### sampled_token_ids 不是最终文本

```text
它还是 token ids，需要 OutputProcessor detokenize。
```

### ModelRunnerOutput 不是 EngineCoreOutput

```text
ModelRunnerOutput 是 Worker 返回 Scheduler。
EngineCoreOutput 是 Scheduler 返回 Engine。
```

### stop string 不在 Scheduler 处理

```text
stop string 需要文本，因此在 OutputProcessor。
```

### prompt_logprobs 不是生成 token logprobs

```text
prompt_logprobs 针对 prompt token；sample logprobs 针对 sampled token。
```

### spec decode 下输出 token 数不固定

```text
一轮可能输出 accepted + recovered / bonus 多个 token。
```

## 25. 与其他专题的关系

```text
executor_worker_model_runner：解释 forward / logits / sample_tokens 在 ModelRunner 哪里发生。
scheduler：解释 update_from_output 如何更新请求状态。
engine：解释 OutputProcessor 如何返回 RequestOutput。
spec_decode：展开 RejectionSampler 和 accepted / rejected token 回账。
attention：attention 输出 hidden states，后续才有 logits。
operators：sampler / logprobs / top-k / top-p 底层 kernel。
structured output：grammar bitmask 和 FSM 状态。
```

## 26. 背诵总结

背这一段：

```text
Sampling / Output 是 vLLM 从模型数值结果到用户输出的后半段。ModelRunner 先用 logits_indices 从 hidden states 中选出需要的位置，调用 compute_logits 得到 logits，再应用 grammar bitmask 和 SamplingParams，通过 Sampler 或 RejectionSampler 产生 sampled_token_ids、logprobs 和 prompt_logprobs，并封装为 ModelRunnerOutput。Scheduler.update_from_output 用 SchedulerOutput 作为计划账本，把 ModelRunnerOutput 对回每个 Request，append token、检查 stop、处理 spec decode 回滚、推进 grammar、释放资源，并生成 EngineCoreOutputs。OutputProcessor 最后 detokenize、检查 stop string、格式化 logprobs 和 streaming delta，构造 RequestOutput 或 PoolingRequestOutput。
```
