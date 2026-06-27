# 01. Sampling / Output 在 vLLM 中负责什么？

源码位置：

- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py`
- `code/vllm/vllm/v1/sample/sampler.py`
- `code/vllm/vllm/v1/sample/metadata.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/sample/sampler.py`
- `code/vllm/vllm/v1/worker/gpu/sample/output.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/engine/__init__.py`
- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/v1/engine/logprobs.py`
- `code/vllm/vllm/outputs.py`

本问题关注：模型 forward 之后，vLLM 内部如何从 hidden states / logits 走到用户可见输出；哪些逻辑属于 ModelRunner，哪些属于 Scheduler，哪些属于 OutputProcessor；generation、pooling、structured output、spec decode、logprobs、async output 在这条链路中分别挂在哪里。

---

## 1. 一句话回答

Sampling / Output 不是单一函数，而是三层协作：

```text
ModelRunner / Sampler：
  负责从 hidden states / logits 得到 token、logprobs 或 pooling output，产出 ModelRunnerOutput。

Scheduler.update_from_output：
  负责把 ModelRunnerOutput 消化回 Request 状态机，推进 token、stop、grammar、spec decode、资源释放，并产出 EngineCoreOutputs。

OutputProcessor：
  负责把 EngineCoreOutputs 转成用户可见 RequestOutput / CompletionOutput / PoolingRequestOutput，包括 detokenize、streaming delta、logprobs 格式化。
```

最小主线是：

```text
GPUModelRunner._model_forward()
  → hidden_states
  → compute_logits()
  → sample_tokens()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

一句话记忆：

```text
Sampling 解决“模型这一步输出了什么 token / pooling 结果”，Output 解决“这个内部结果如何变成用户看见的响应”。
```

---

## 2. Sampling / Output 在整体链路中的位置

一次 generation 请求从执行到输出，可以压缩成：

```text
EngineCore.step()
  → Scheduler.schedule()
  → Executor.execute_model()
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
      → _prepare_inputs()
      → _build_attention_metadata()
      → _model_forward()
      → compute_logits()
      → 保存 ExecuteModelState
      → return None
  → EngineCore 获取 grammar_output
  → Executor.sample_tokens(grammar_output)
  → Worker.sample_tokens(grammar_output)
  → GPUModelRunner.sample_tokens(grammar_output)
      → apply_grammar_bitmask()
      → Sampler / RejectionSampler
      → logprobs / prompt_logprobs / bookkeeping
      → ModelRunnerOutput
  → Scheduler.update_from_output()
      → 更新 Request 状态
      → stop / finish 判断
      → EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs()
      → detokenize
      → logprobs 转换
      → RequestOutput / CompletionOutput
```

这里有三个“输出对象层级”：

```text
SamplerOutput：
  GPU / worker 内部的采样结果，偏 tensor 形态。

ModelRunnerOutput：
  Worker 返回给 Scheduler 的内部协议对象。

EngineCoreOutput / EngineCoreOutputs：
  Scheduler 返回给 Engine / OutputProcessor 的内部协议对象。

RequestOutput / CompletionOutput / PoolingRequestOutput：
  用户可见输出对象。
```

因此不要把 `sampled_token_ids`、`ModelRunnerOutput`、`RequestOutput` 混为一谈。

---

## 3. 三层职责边界

### 3.1 ModelRunner / Sampler 负责什么

ModelRunner 是 sampling 的设备侧执行层。

它负责：

```text
- 接收 execute_model() 阶段保存的 hidden states / logits / scheduler_output；
- 在 generation 场景下从 hidden states 计算 logits；
- 根据 logits_indices 只对需要采样的位置计算 logits；
- 接收 grammar bitmask 并在采样前屏蔽非法 token；
- 调用 Sampler 或 RejectionSampler；
- 处理 temperature / top-p / top-k / penalties / bad words / allowed token ids；
- 计算 sample logprobs；
- 计算 prompt logprobs；
- 处理 speculative decoding 的 draft proposal / accepted token 状态；
- 处理 NaN logits 统计；
- 处理 routed experts 输出；
- 处理 KV connector / EC connector 输出；
- 在 async scheduling 下包装异步 GPU→CPU 拷贝；
- 构造 ModelRunnerOutput。
```

它不负责：

```text
- 判断请求是否最终 finished；
- 把 sampled token 写入 Scheduler 的 Request；
- 释放 Scheduler 管理的 KV blocks；
- detokenize；
- 组装用户可见 RequestOutput；
- 处理 OpenAI API response 格式。
```

一句话：

```text
ModelRunner 负责“把模型数值输出变成 Scheduler 能消费的内部结果”。
```

### 3.2 Scheduler.update_from_output 负责什么

`Scheduler.update_from_output()` 是 Worker 输出进入请求状态机的入口。

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1464`

它负责：

```text
- 根据 ModelRunnerOutput.req_id_to_index 找到每个 request 的 batch index；
- 读取 sampled_token_ids；
- 把 generated token append 到 Request；
- 更新 request.num_computed_tokens；
- 处理 speculative decoding 的 accepted / rejected tokens；
- 必要时回退 spec decode 占位 token；
- 推进 structured output grammar 状态；
- 切分 sample logprobs；
- 取出 prompt_logprobs；
- 处理 pooling output；
- 处理 stop condition / finish_reason / stop_reason；
- 处理 KV connector / EC connector 输出；
- 释放 finished request 的 KV blocks / encoder cache / connector 状态；
- 构造 EngineCoreOutput；
- 按 client_index 汇总为 EngineCoreOutputs。
```

它不负责：

```text
- 重新采样；
- 重新计算 logits；
- detokenize；
- 生成 CompletionOutput；
- 格式化 OpenAI response。
```

一句话：

```text
Scheduler 负责“把模型输出对账回请求状态和资源账本”。
```

### 3.3 OutputProcessor 负责什么

`OutputProcessor` 是内部输出到用户输出的最后一层适配器。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:417`

它负责：

```text
- 接收 EngineCoreOutputs；
- 根据 request_id 找到 frontend 侧 RequestState；
- 更新请求统计信息；
- 对 generation 输出执行增量 detokenize；
- 检查 stop string；
- 更新 sample logprobs / prompt logprobs；
- 处理 streaming delta 或 cumulative output；
- 构造 RequestOutput / CompletionOutput；
- 对 pooling / embedding 请求构造 PoolingRequestOutput；
- 对 AsyncLLM，把输出放入请求队列；
- 对 LLMEngine，同步返回输出列表；
- 清理 finished request 的 frontend 状态。
```

它不负责：

```text
- 调用 sampler；
- 修改 logits；
- 决定 Scheduler 调度；
- 分配或释放 KV block；
- 推进 structured output grammar FSM。
```

一句话：

```text
OutputProcessor 负责“把内部 Engine 输出翻译成用户 API 输出”。
```

---

## 4. execute_model 和 sample_tokens 的边界

### 4.1 generation 场景为什么 execute_model 常返回 None

在 generation 模型中，`GPUModelRunner.execute_model()` 通常完成：

```text
input 准备
  → attention metadata
  → model forward
  → hidden_states
  → hidden_states[logits_indices]
  → model.compute_logits()
  → 保存 ExecuteModelState
  → return None
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4047`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4323`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4358`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4389`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4408`

`None` 的含义不是“没有输出”，而是：

```text
forward / logits 已经完成，真正采样结果需要后续 sample_tokens() 生成。
```

EngineCore 里对应逻辑是：

```text
model_output = future.result()
if model_output is None:
    model_output = model_executor.sample_tokens(grammar_output)
```

源码位置：`code/vllm/vllm/v1/engine/core.py:497` 到 `code/vllm/vllm/v1/engine/core.py:498`

### 4.2 ExecuteModelState 是 execute_model 和 sample_tokens 的桥

`execute_model()` 保存的临时状态包括：

```text
scheduler_output
logits
spec_decode_metadata
spec_decode_common_attn_metadata
hidden_states
sample_hidden_states
aux_hidden_states
ec_connector_output
cudagraph_stats
slot_mappings
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4389`

`sample_tokens()` 会解包这些状态，然后继续采样。

所以可以记成：

```text
execute_model() 负责“算出采样所需的 logits 和上下文”；
sample_tokens() 负责“消费这些上下文并构造 ModelRunnerOutput”。
```

### 4.3 sample_tokens 的主步骤

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4426`

核心步骤：

```text
1. 检查并解包 self.execute_model_state；
2. 如果有 grammar_output，调用 apply_grammar_bitmask()；
3. 调用 self._sample(logits, spec_decode_metadata)；
4. 调用 _update_states_after_model_execute()；
5. 处理 async scheduling / pipeline parallel 的 token 通信；
6. 处理 speculative decoding draft token 生成；
7. 做 bookkeeping，整理 sampled tokens / logprobs / prompt logprobs；
8. finalize KV connector；
9. 附加 routed experts / cudagraph stats / connector output；
10. 构造 ModelRunnerOutput；
11. async scheduling 下包装为 AsyncGPUModelRunnerOutput。
```

### 4.4 V2 GPU ModelRunner 的差异

较新的 `code/vllm/vllm/v1/worker/gpu/model_runner.py` 也有类似边界，但组织方式略不同：

```text
execute_model()
  → 保存 hidden states / model runner state

sample_tokens()
  → 从 hidden states 计算 logits
  → sample()
  → 构造 ModelRunnerOutput
```

对应源码位置：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1101`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1310`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1326`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1360`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1384`

从文档视角看，两者的共同点是：

```text
forward 和 sampling 可以拆阶段；
ModelRunnerOutput 是 worker 给 scheduler 的结果协议；
用户可见输出仍要经过 Scheduler 和 OutputProcessor。
```

---

## 5. Sampler / SamplingMetadata / SamplerOutput 的角色

### 5.1 SamplingMetadata 是采样参数的 batch 形态

源码位置：`code/vllm/vllm/v1/sample/metadata.py:15`

`SamplingMetadata` 把每个请求的采样参数整理成 Sampler 可消费的 batch 结构。

它包含：

```text
temperature
top_p
top_k
all_greedy / all_random
generators
max_num_logprobs
no_penalties / prompt_token_ids / frequency_penalties / presence_penalties / repetition_penalties
output_token_ids
min_tokens
allowed_token_ids_mask
bad_words_token_ids
logitsprocs
logprob_token_ids
spec_token_ids
thinking budget state
```

它来自 `InputBatch` 维护的 sampling 参数和当前 batch 状态。

一句话：

```text
SamplingMetadata 说明“这一批请求该按什么规则采样”。
```

### 5.2 Sampler 做真正 token 选择

旧 V1 Sampler 源码位置：`code/vllm/vllm/v1/sample/sampler.py:20`

Sampler 的核心流程是：

```text
1. 如果需要 logprobs，先基于原始 logits 计算 raw_logprobs 或保留 raw_logits；
2. 将 logits 转成 float32；
3. 应用 allowed token ids；
4. 应用 bad words；
5. 应用 logits processors；
6. 应用 penalties；
7. 根据 temperature / greedy / random 执行采样；
8. 计算 top-k / sampled token / 指定 token 的 logprobs；
9. 返回 SamplerOutput。
```

源码位置：

- `code/vllm/vllm/v1/sample/sampler.py:72`
- `code/vllm/vllm/v1/sample/sampler.py:84`
- `code/vllm/vllm/v1/sample/sampler.py:98`
- `code/vllm/vllm/v1/sample/sampler.py:102`
- `code/vllm/vllm/v1/sample/sampler.py:129`
- `code/vllm/vllm/v1/sample/sampler.py:142`

V2 optimized GPU sampler 在：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:30`

它更贴近 GPU input batch，减少 Python 侧 metadata 构造，但职责仍然是：

```text
logits + sampling state → sampled_token_ids + logprobs_tensors
```

### 5.3 SamplerOutput 是 worker 内部采样结果

旧 V1 `SamplerOutput` 定义在：`code/vllm/vllm/v1/outputs.py:186`

主要字段：

```text
sampled_token_ids：
  GPU tensor，表示采样得到的 token ids。

logprobs_tensors：
  可选的 logprobs tensor。
```

V2 GPU sampler output 在：`code/vllm/vllm/v1/worker/gpu/sample/output.py:11`

额外包含：

```text
num_nans
num_sampled
num_rejected
```

注意：

```text
SamplerOutput 还不是 Worker → Scheduler 的最终结果；
它要经过 bookkeeping、CPU 化、logprobs 整理和 connector 输出合并，才变成 ModelRunnerOutput。
```

---

## 6. Structured Output / grammar 在哪里接入

### 6.1 grammar bitmask 由 Scheduler 侧准备

结构化输出约束不是 OutputProcessor 后处理，而是在采样前改变 logits 可选空间。

大致链路：

```text
StructuredOutputManager 编译 grammar
  → Scheduler.get_grammar_bitmask(scheduler_output)
  → GrammarOutput
  → EngineCore.sample_tokens(grammar_output)
  → ModelRunner.apply_grammar_bitmask()
  → Sampler.sample()
```

源码位置：

- `code/vllm/vllm/v1/structured_output/__init__.py:36`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1440`
- `code/vllm/vllm/v1/core/sched/output.py:263`
- `code/vllm/vllm/v1/engine/core.py:492`

### 6.2 bitmask 在采样前应用

旧 V1 runner 中：

```text
sample_tokens()
  → apply_grammar_bitmask(scheduler_output, grammar_output, input_batch, logits)
  → _sample()
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4455`

V2 runner 中也在 sample 阶段计算 logits 后应用 grammar bitmask。

源码位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1045`

### 6.3 grammar 状态在 Scheduler 中推进

Scheduler 收到 sampled tokens 后，会推进 grammar FSM。

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1599`

这意味着：

```text
Worker 负责按 bitmask 限制采样；
Scheduler 负责根据采样结果推进 grammar 状态。
```

---

## 7. Speculative decoding 在 sampling/output 中的角色

Spec decode 会让“一轮一个请求输出几个 token”变得不固定。

### 7.1 SchedulerOutput 携带 draft tokens

Scheduler 会在 `SchedulerOutput.scheduled_spec_decode_tokens` 中告诉 Worker，本轮 target model 需要验证哪些 draft token。

源码位置：`code/vllm/vllm/v1/core/sched/output.py:197`

### 7.2 ModelRunner 使用 RejectionSampler

`GPUModelRunner._sample()` 中，如果没有 spec decode metadata，就走普通 sampler；如果有 spec decode metadata，则调用 rejection sampler。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3573`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3583`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3596`
- `code/vllm/vllm/v1/sample/rejection_sampler.py:37`

RejectionSampler 的输出仍然会包装成 `SamplerOutput`，但每个请求可能输出多个 accepted tokens，或者在拒绝位置重新采样一个 token。

### 7.3 Scheduler 修正 accepted / rejected 状态

Scheduler 消费 `ModelRunnerOutput.sampled_token_ids` 后，会和原本 scheduled 的 draft tokens 对账：

```text
scheduled draft tokens
  + sampled token ids
  → accepted / rejected 数量
  → 修正 num_computed_tokens
  → 必要时回退 output placeholders
  → 更新 request 输出 token
```

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py:1548`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1554`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1564`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1567`

### 7.4 draft proposal 属于下一轮准备

`sample_tokens()` 还可能根据当前 sampled tokens、hidden states 或 ngram 生成下一轮 draft tokens。

这部分不是用户输出本身，而是为了下一轮 scheduler 能调度 spec decode。

可以记成：

```text
当前轮 target model 输出
  → Scheduler 更新 accepted tokens
  → Worker 侧生成下一轮 draft tokens
  → 下一轮 SchedulerOutput.scheduled_spec_decode_tokens
```

---

## 8. logprobs / prompt_logprobs 在哪里产生

### 8.1 sample logprobs

sample logprobs 和生成 token 绑定。

采样阶段会根据用户请求的：

```text
logprobs
logprob_token_ids
logprobs_mode
```

保留或计算 raw logprobs / processed logprobs，并收集 top-k 或指定 token 的 logprobs。

源码位置：

- `code/vllm/vllm/v1/sample/sampler.py:84`
- `code/vllm/vllm/v1/sample/sampler.py:87`
- `code/vllm/vllm/v1/sample/sampler.py:129`
- `code/vllm/vllm/v1/outputs.py:249`

之后 Scheduler 会把 batch 级 logprobs 切成 request 级：

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1671`

OutputProcessor 再通过 `LogprobsProcessor` 转成用户输出格式。

源码位置：

- `code/vllm/vllm/v1/engine/logprobs.py:69`
- `code/vllm/vllm/v1/engine/logprobs.py:348`
- `code/vllm/vllm/v1/engine/output_processor.py:648`

### 8.2 prompt_logprobs

prompt logprobs 不是“采样出来的 token”的 logprobs，而是 prompt token 在模型上下文中的 logprobs。

旧 V1 中通常在 ModelRunner bookkeeping 阶段计算并放进：

```text
ModelRunnerOutput.prompt_logprobs_dict
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3730`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4618`
- `code/vllm/vllm/v1/outputs.py:255`

Scheduler 再把它放到 `EngineCoreOutput.new_prompt_logprobs_tensors`。

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1682`

OutputProcessor 最终写入 `RequestOutput.prompt_logprobs`。

源码位置：

- `code/vllm/vllm/v1/engine/logprobs.py:121`
- `code/vllm/vllm/v1/engine/output_processor.py:359`
- `code/vllm/vllm/v1/engine/output_processor.py:368`

---

## 9. ModelRunnerOutput 是什么

`ModelRunnerOutput` 是 Worker / ModelRunner 返回给 Scheduler 的 batch 级内部结果。

源码位置：`code/vllm/vllm/v1/outputs.py:234`

它的设计目标是：

```text
跨进程可传输；
便于 Scheduler 按 request 对账；
携带 token、logprobs、pooling、connector、debug/metrics 等所有执行结果。
```

### 9.1 req_ids / req_id_to_index

```text
req_ids：
  当前 batch 中 request id 的顺序。

req_id_to_index：
  request id → batch index。
```

Scheduler 用它把 `sampled_token_ids[index]` 对回具体 request。

### 9.2 sampled_token_ids

```text
sampled_token_ids: list[list[int]]
```

每个请求一个 list。

为什么是 `list[list[int]]`？

```text
普通 decode：每个请求通常 1 个 token；
spec decode / jump decoding：每个请求一轮可能多个 token；
某些 invalid / partial 情况可能是空 list。
```

### 9.3 logprobs

```text
logprobs: LogprobsLists | None
```

用于生成 token 的 logprobs。

`LogprobsLists` 通常包含：

```text
logprob_token_ids
logprobs
sampled_token_ranks
cu_num_generated_tokens
```

源码位置：`code/vllm/vllm/v1/outputs.py:27`

### 9.4 prompt_logprobs_dict

```text
prompt_logprobs_dict: dict[str, LogprobsTensors | None]
```

按 request id 保存 prompt logprobs。

### 9.5 pooler_output

```text
pooler_output: list[torch.Tensor | None] | None
```

用于 pooling / embedding / classification / reward 等非 generation 任务。

### 9.6 connector / stats / routed experts

还包括：

```text
kv_connector_output：KV transfer 输出；
ec_connector_output：encoder cache / multimodal connector 输出；
num_nans_in_logits：logits NaN 统计；
cudagraph_stats：CUDA graph 执行统计；
routed_experts：MoE routed experts 信息。
```

一句话：

```text
ModelRunnerOutput 是 Worker 返回 Scheduler 的“本轮执行对账凭证”。
```

---

## 10. Scheduler 如何把 ModelRunnerOutput 变成 EngineCoreOutputs

`Scheduler.update_from_output()` 的输入是：

```text
scheduler_output：
  本轮原计划。

model_runner_output：
  Worker 实际执行结果。
```

源码位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1464`

它会逐个 request 对账：

```text
for req in scheduled requests:
  req_index = model_runner_output.req_id_to_index[req_id]
  generated_token_ids = sampled_token_ids[req_index]
  处理 spec decode accepted / rejected
  更新 request.output_token_ids
  检查 stop condition
  推进 structured output grammar
  切分 sample logprobs
  读取 prompt logprobs
  构造 EngineCoreOutput
```

关键输出对象是 `EngineCoreOutput`。

源码位置：`code/vllm/vllm/v1/engine/__init__.py:175`

它包含：

```text
request_id
new_token_ids
new_logprobs
new_prompt_logprobs_tensors
pooling_output
finish_reason
stop_reason
events
kv_transfer_params
num_cached_tokens
routed_experts
num_nans_in_logits
```

多个 `EngineCoreOutput` 会按 client 分组，形成 `EngineCoreOutputs`。

源码位置：`code/vllm/vllm/v1/engine/__init__.py:220`

一句话：

```text
ModelRunnerOutput 是 batch 级执行结果，EngineCoreOutput 是 request 级引擎输出。
```

---

## 11. OutputProcessor 如何生成用户可见输出

### 11.1 process_outputs 的主流程

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:576`

主流程：

```text
OutputProcessor.process_outputs(engine_core_outputs)
  → 遍历 EngineCoreOutput
  → 找到 RequestState
  → 更新 stats
  → 如果是 generation：
       detokenize new_token_ids
       检查 stop string
       更新 logprobs / prompt_logprobs
       make_request_output()
  → 如果是 pooling：
       make_pooling_request_output()
  → 如果是 AsyncLLM：
       放入 request queue
  → 如果是 LLMEngine：
       返回 RequestOutput list
  → finished 后清理 RequestState
```

关键源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:606`
- `code/vllm/vllm/v1/engine/output_processor.py:614`
- `code/vllm/vllm/v1/engine/output_processor.py:635`
- `code/vllm/vllm/v1/engine/output_processor.py:639`
- `code/vllm/vllm/v1/engine/output_processor.py:648`
- `code/vllm/vllm/v1/engine/output_processor.py:651`
- `code/vllm/vllm/v1/engine/output_processor.py:662`
- `code/vllm/vllm/v1/engine/output_processor.py:669`

### 11.2 RequestOutput / CompletionOutput

用户可见 generation 输出定义在 `code/vllm/vllm/outputs.py`。

`CompletionOutput` 表示一个 completion 分支：

```text
index
text
token_ids
cumulative_logprob
logprobs
finish_reason
stop_reason
```

源码位置：`code/vllm/vllm/outputs.py:21`

`RequestOutput` 表示一个用户请求：

```text
request_id
prompt
prompt_token_ids
prompt_logprobs
outputs: list[CompletionOutput]
finished
metrics
kv_transfer_params
```

源码位置：`code/vllm/vllm/outputs.py:85`

如果是 streaming，`RequestOutput.add()` 还可以合并增量输出。

源码位置：`code/vllm/vllm/outputs.py:145`

### 11.3 为什么 detokenize 不在 Scheduler 做

Scheduler 只维护 token id 和 request 状态，不负责文本协议。

原因是：

```text
- Scheduler 应保持内部状态机简单；
- detokenize 与 tokenizer、streaming、stop string、OpenAI 输出格式关系更近；
- frontend 可能需要 delta / cumulative 两种模式；
- pooling / embedding 不需要普通 detokenize。
```

所以 detokenize 放在 OutputProcessor / frontend 侧。

---

## 12. generation 和 pooling / embedding 的差异

### 12.1 generation 路径

```text
model forward
  → hidden_states
  → compute_logits(hidden_states[logits_indices])
  → grammar bitmask
  → Sampler / RejectionSampler
  → sampled_token_ids / logprobs
  → ModelRunnerOutput
  → Scheduler append tokens
  → OutputProcessor detokenize
  → RequestOutput / CompletionOutput
```

关键特点：

```text
- 有 logits；
- 有 sampling；
- 有 output_token_ids；
- 有 stop token / stop string / max_tokens；
- 用户输出是文本 completion。
```

### 12.2 pooling / embedding 路径

```text
model forward
  → hidden_states
  → model.pooler(hidden_states, pooling_metadata)
  → pooler_output
  → ModelRunnerOutput.pooler_output
  → Scheduler 标记 request stopped
  → OutputProcessor 构造 PoolingRequestOutput / EmbeddingOutput
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3346`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3370`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3385`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1594`
- `code/vllm/vllm/v1/engine/output_processor.py:312`
- `code/vllm/vllm/outputs.py:204`
- `code/vllm/vllm/outputs.py:240`

关键特点：

```text
- 不走普通 Sampler；
- 不生成 sampled_token_ids；
- 通常不做 completion detokenize；
- 输出是向量、score、classification 或 pooling 结果。
```

### 12.3 Pipeline Parallel 下的差异

非最后 PP rank：

```text
model forward
  → IntermediateTensors
  → 发给下一 PP stage
```

最后 PP rank：

```text
hidden_states
  → generation: logits / sampling
  → pooling: pooler_output
```

所以不是所有 Worker 都会真正构造最终 `ModelRunnerOutput`。

---

## 13. async output / async scheduling 的角色

### 13.1 为什么需要 async output

采样后，很多结果最初在 GPU tensor 中：

```text
sampled_token_ids
logprobs_tensors
prompt_logprobs
routed_experts
pooling_output
```

如果每步都在主流上同步拷贝到 CPU，会阻塞后续计算。

async output 的目标是：

```text
把 GPU → CPU 拷贝放到 copy stream，和下一步 compute 重叠。
```

### 13.2 旧 V1 AsyncGPUModelRunnerOutput

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:241`

`sample_tokens()` 构造普通 `ModelRunnerOutput` 后，如果启用 async scheduling，会包装成 `AsyncGPUModelRunnerOutput`。

流程：

```text
AsyncGPUModelRunnerOutput.__init__()
  → copy stream 上发起 sampled_token_ids / logprobs / routed_experts D2H
  → 记录 event

get_output()
  → 等待 event
  → sampled_token_ids 转 list[list[int]]
  → logprobs tensor 转 LogprobsLists
  → 回填 ModelRunnerOutput
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:266`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:285`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:311`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4666`

### 13.3 V2 AsyncOutput / AsyncPoolingOutput

V2 路径定义在：`code/vllm/vllm/v1/worker/gpu/async_utils.py:12`

对应 generation 和 pooling 两类：

```text
AsyncOutput：
  sampled tokens / logprobs / prompt logprobs。

AsyncPoolingOutput：
  pooling output。
```

最终仍然通过 `get_output()` 变回普通 `ModelRunnerOutput`。

---

## 14. 关键对象关系

### 14.1 `SamplerOutput`

worker 内部采样结果。

```text
sampled_token_ids: Tensor
logprobs_tensors: LogprobsTensors | None
```

### 14.2 `ModelRunnerOutput`

Worker → Scheduler 的 batch 级协议对象。

```text
req_ids
req_id_to_index
sampled_token_ids
logprobs
prompt_logprobs_dict
pooler_output
kv_connector_output
ec_connector_output
num_nans_in_logits
cudagraph_stats
routed_experts
```

### 14.3 `EngineCoreOutput`

Scheduler → OutputProcessor 的 request 级协议对象。

```text
request_id
new_token_ids
new_logprobs
new_prompt_logprobs_tensors
pooling_output
finish_reason
stop_reason
```

### 14.4 `EngineCoreOutputs`

一个 client 的一批输出。

```text
engine_index
outputs: list[EngineCoreOutput]
scheduler_stats
finished_requests
timestamp
```

### 14.5 `CompletionOutput`

用户可见的一个 completion。

```text
index
text
token_ids
logprobs
finish_reason
stop_reason
```

### 14.6 `RequestOutput`

用户可见的 generation 请求输出。

```text
request_id
prompt
prompt_token_ids
prompt_logprobs
outputs: list[CompletionOutput]
finished
metrics
```

### 14.7 `PoolingRequestOutput`

用户可见的 pooling / embedding 类请求输出。

```text
request_id
outputs / data
finished
```

---

## 15. 和其他专题的关系

Sampling / Output 和以下文档强相关：

```text
../executor_worker_model_runner/07_model_forward_and_logits.md
  解释 hidden_states 和 logits 在哪里产生。

../executor_worker_model_runner/08_sampling_and_model_runner_output.md
  解释 sample_tokens() 如何构造 ModelRunnerOutput。

../scheduler/08_update_after_worker_output.md
  解释 Scheduler 如何消费 ModelRunnerOutput 并更新请求状态。

../attention/attention_overview.md
  解释 logits 之前的 model forward 和 attention metadata。

../config_and_model_loading/08_model_layers_and_execution_interface.md
  解释模型 forward / compute_logits / pooler 接口。
```

它们可以串成：

```text
attention / model forward
  → logits
  → sampling
  → Scheduler update
  → output processor
  → user output
```

---

## 16. 容易混淆的点

### 16.1 ModelRunnerOutput 是用户输出吗？

不是。

它是 Worker 给 Scheduler 的内部结果，还没有 detokenize，也没有转成用户 API 结构。

### 16.2 sampled_token_ids 是文本吗？

不是。

它只是 token id。文本要由 OutputProcessor / detokenizer 生成。

### 16.3 sample_tokens 是不是只做采样？

不是。

它还处理：

```text
grammar bitmask
spec decode
logprobs
prompt logprobs
bookkeeping
KV connector finalize
routed experts
async output packaging
```

### 16.4 Scheduler.update_from_output 是不是只转发 token？

不是。

它会更新 Request 状态、stop condition、grammar 状态、spec decode 状态，并释放资源。

### 16.5 OutputProcessor 会重新采样吗？

不会。

它只做输出适配：detokenize、logprobs 格式化、RequestOutput 构造、streaming queue 推送。

### 16.6 pooling / embedding 会走普通 token sampling 吗？

不会。

pooling / embedding 通常走 `pooler_output`，不走普通 `Sampler.sample()`。

### 16.7 structured output 是采样后修正吗？

不是。

structured output 的 grammar bitmask 在采样前应用，保证采样空间被约束。

### 16.8 为什么 sampled_token_ids 是 list[list[int]]？

因为 spec decode / jump decoding 下，每个请求一轮可能生成多个 token；也可能因为 invalid / partial 情况输出空列表。

---

## 17. 总结

Sampling / Output 的完整关系可以压缩成：

```text
hidden_states
  → logits
  → grammar bitmask
  → Sampler / RejectionSampler
  → SamplerOutput
  → bookkeeping / logprobs / prompt_logprobs / connector output
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / CompletionOutput / PoolingRequestOutput
```

如果只记住一句话：

```text
ModelRunner 负责生成内部执行结果，Scheduler 负责把结果写回请求状态机，OutputProcessor 负责把内部状态变成用户可见输出。
```

再压缩一层：

```text
Sampling 解决“选哪个 token”，Scheduler update 解决“这个 token 如何改变请求状态”，OutputProcessor 解决“这个状态如何展示给用户”。
```
