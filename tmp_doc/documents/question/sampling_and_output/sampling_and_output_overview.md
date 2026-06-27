# vLLM V1 Sampling / Output 逻辑梳理

源码位置：

- `vllm/vllm/sampling_params.py`
- `vllm/vllm/outputs.py`
- `vllm/vllm/logprobs.py`
- `vllm/vllm/v1/outputs.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/engine/__init__.py`
- `vllm/vllm/v1/engine/output_processor.py`
- `vllm/vllm/v1/engine/logprobs.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/sched/output.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu/model_runner.py`
- `vllm/vllm/v1/worker/gpu/async_utils.py`
- `vllm/vllm/v1/worker/gpu/structured_outputs.py`
- `vllm/vllm/v1/worker/gpu/sample/sampler.py`
- `vllm/vllm/v1/worker/gpu/sample/output.py`
- `vllm/vllm/v1/worker/gpu/sample/logprob.py`
- `vllm/vllm/v1/worker/gpu/sample/prompt_logprob.py`
- `vllm/vllm/v1/worker/gpu/sample/logit_bias.py`
- `vllm/vllm/v1/worker/gpu/sample/penalties.py`
- `vllm/vllm/v1/worker/gpu/sample/bad_words.py`
- `vllm/vllm/v1/worker/gpu/sample/min_p.py`
- `vllm/vllm/v1/worker/gpu/sample/gumbel.py`
- `vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- `vllm/vllm/v1/sample/sampler.py`
- `vllm/vllm/v1/sample/metadata.py`
- `vllm/vllm/v1/sample/rejection_sampler.py`
- `vllm/vllm/v1/sample/ops/topk_topp_sampler.py`
- `vllm/vllm/v1/sample/ops/logprobs.py`

本文按“先定边界，再走主链路，再拆关键阶段，最后总结接口和数据结构”的方式，梳理 vLLM V1 中 sampling 与 output 的关系。

它接在 `executor_worker_model_runner` 的 `07_model_forward_and_logits.md`、`08_sampling_and_model_runner_output.md` 后面，专门展开模型 forward 之后的后半段：

```text
hidden states / logits
  → grammar bitmask / logits processors
  → sampler / rejection sampler
  → sampled token ids / logprobs / prompt logprobs
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / CompletionOutput / streaming chunk
```

---

## 0. 梳理规划

本目录要回答的问题分成 11 组：

```text
1. Sampling / Output 在 vLLM 全链路中处于哪一段？
2. SamplingParams 如何从用户参数变成 worker / sampler 可消费的 metadata 或 state？
3. logits、sample logprobs、prompt logprobs 分别在哪里计算？
4. sampler 如何处理 temperature / top-k / top-p / min-p / penalties / seed？
5. speculative decoding 如何改变采样、接受/拒绝和输出回收？
6. structured output / grammar bitmask 如何限制采样？
7. ModelRunnerOutput 是什么？为什么它还不是用户输出？
8. Scheduler.update_from_output() 如何消费 sampled tokens 并更新 request 状态？
9. OutputProcessor 如何把内部状态转换成 RequestOutput？
10. streaming 输出如何决定本轮返回哪些 token / logprobs / finish reason？
11. pooling / embedding / rerank 输出如何区别于生成式输出？
```

阅读顺序建议：

```text
sampling_and_output_overview.md
  → 01_sampling_output_role.md
  → 02_sampling_params_and_metadata.md
  → 03_logits_and_logprobs.md
  → 04_sampler_flow.md
  → 05_spec_decode_sampling.md
  → 06_structured_output_and_grammar.md
  → 07_model_runner_output.md
  → 08_scheduler_update_output.md
  → 09_output_processor_request_output.md
  → 10_streaming_and_client_outputs.md
  → 11_pooling_and_embedding_outputs.md
```

如果只想先抓一条主线，可以先读总览，再读 `03`、`04`、`07`、`08`、`09`。

---

## 1. 一句话回答

Sampling / Output 这段链路负责把模型内部计算结果变成上层可见输出。

可以压缩成：

```text
ModelRunner / Sampler 负责“从 hidden states / logits 采样出 token 或 pooling output”；
Scheduler 负责“把 worker 输出消化回 request 状态机和资源账本”；
OutputProcessor 负责“把内部 request 状态转换成用户可见输出”。
```

最小主线是：

```text
EngineCore.step()
  → Scheduler.schedule()
  → Executor.execute_model()
  → GPUModelRunner.execute_model()
  → model forward
  → compute_logits()
  → sample_tokens(grammar_output)
  → Sampler / RejectionSampler
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / CompletionOutput / PoolingRequestOutput
```

对应关键入口：

- `vllm/vllm/v1/engine/core.py:479`
- `vllm/vllm/v1/worker/gpu/model_runner.py:1101`
- `vllm/vllm/v1/worker/gpu/model_runner.py:1326`
- `vllm/vllm/v1/core/sched/scheduler.py:1464`
- `vllm/vllm/v1/engine/output_processor.py:576`

一句话记忆：

```text
Sampling 解决“选哪个 token”，Scheduler update 解决“这个 token 如何改变请求状态”，OutputProcessor 解决“这个状态如何展示给用户”。
```

---

## 2. 三层边界

### 2.1 ModelRunner / Sampler 层

这一层负责设备侧采样和执行层输出组织。

```text
输入：
  hidden states / logits / InputBatch / sampling state / grammar bitmask / spec decode metadata

输出：
  sampled_token_ids / logprobs / prompt_logprobs / pooler_output / connector output / routed experts
```

它关心的是：

```text
- 哪些 hidden states 需要计算 logits；
- structured output grammar bitmask 是否要先屏蔽非法 token；
- SamplingParams 如何应用到 logits；
- temperature / top-k / top-p / min-p / penalties / seed 如何参与采样；
- spec decode 下 draft tokens 如何接受、拒绝和补采；
- sample logprobs / prompt logprobs 是否要返回；
- GPU tensor 如何异步复制到 CPU；
- 如何构造 Scheduler 能消费的 ModelRunnerOutput。
```

它不负责：

```text
- 决定请求是否 finished；
- 把 token append 到 Scheduler 的 Request；
- 释放 Scheduler 管理的 KV blocks；
- detokenize；
- 构造用户可见 RequestOutput。
```

一句话：

```text
ModelRunner / Sampler 负责“把模型数值输出变成 Scheduler 能对账的内部结果”。
```

### 2.2 Scheduler update 层

这一层负责把执行层输出回收到调度状态机。

```text
输入：
  SchedulerOutput + ModelRunnerOutput

输出：
  更新后的 request 状态 + EngineCoreOutputs
```

它关心的是：

```text
- 根据 req_id_to_index 找回每个 request 的输出；
- sampled token 是否 append 到 request；
- stop token / EOS / max tokens / repetition detection 是否触发；
- spec decode accepted / rejected token 如何修正 num_computed_tokens；
- structured output grammar 状态如何推进；
- logprobs / prompt_logprobs 如何切成 request 级；
- pooling output 如何使请求结束；
- KV blocks / encoder cache / connector metadata 如何释放或更新；
- 本轮哪些 request 需要产生 EngineCoreOutput。
```

它不负责：

```text
- 重新计算 logits；
- 重新采样；
- detokenize；
- 输出 OpenAI API 格式。
```

一句话：

```text
Scheduler 负责“把模型输出对账回请求状态和资源账本”。
```

### 2.3 OutputProcessor / client output 层

这一层负责把 EngineCore 内部输出转换为用户 API 可见结构。

```text
输入：
  EngineCoreOutputs / frontend RequestState / tokenizer / output_kind

输出：
  RequestOutput / CompletionOutput / PoolingRequestOutput / streaming queue item
```

它关心的是：

```text
- detokenize token ids；
- 检查 stop string；
- 处理 DELTA / CUMULATIVE / FINAL_ONLY 输出模式；
- sample logprobs / prompt logprobs 格式化；
- finish_reason / stop_reason 如何写入 CompletionOutput；
- AsyncLLM streaming 时是否放入 per-request queue；
- LLMEngine 同步模式下是否返回 RequestOutput list；
- finished 后清理 frontend RequestState。
```

它不负责：

```text
- sampler；
- logits processors；
- Scheduler 调度；
- KV block 分配和释放；
- structured output grammar FSM 推进。
```

一句话：

```text
OutputProcessor 负责“把内部 Engine 输出翻译成用户 API 输出”。
```

---

## 3. 总体流程图

```text
用户输入 SamplingParams
  → InputProcessor / EngineCoreRequest 保存请求参数
  → Scheduler 持有 request.sampling_params
  → SchedulerOutput 告诉 Worker 本轮跑哪些 token
  → ModelRunner.add_requests() 把 SamplingParams 写入 sampler state
  → ModelRunner.execute_model()
      → prepare_inputs
      → prepare_attn
      → set_forward_context
      → model forward
      → hidden_states / IntermediateTensors
  → ModelRunner.sample_tokens(grammar_output)
      → hidden_states[input_batch.logits_indices]
      → model.compute_logits()
      → structured output grammar bitmask
      → Sampler 或 RejectionSampler
      → sampled_token_ids
      → sample logprobs
      → prompt logprobs
      → AsyncOutput / ModelRunnerOutput
      → postprocess_sampled / speculator proposal
  → Scheduler.update_from_output()
      → append output tokens
      → update num_computed_tokens
      → check stop
      → advance grammar
      → slice logprobs
      → free finished request resources
      → EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs()
      → detokenize
      → stop string check
      → logprobs format
      → make RequestOutput / CompletionOutput / PoolingRequestOutput
      → streaming queue 或同步返回
```

EngineCore 的关键连接点在：

```python
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```

源码位置：`vllm/vllm/v1/engine/core.py:490` 到 `vllm/vllm/v1/engine/core.py:505`

这里的关键点是：

```text
execute_model 先完成 forward；
grammar_output 在 Scheduler 侧生成；
如果 execute_model 返回 None，则再显式调用 sample_tokens；
Scheduler 最后消费 ModelRunnerOutput，而不是直接消费 logits 或 SamplerOutput。
```

---

## 4. 这段链路为什么要分层

### 4.1 ModelRunner 适合放“数值输出到 token”

ModelRunner 已经持有：

```text
- 当前 batch 的 InputBatch；
- hidden states / logits；
- logits_indices；
- request state GPU backing buffer；
- sampler state；
- speculator / rejection sampler；
- prompt logprobs worker；
- structured outputs worker；
- copy stream / main stream。
```

所以采样放在 ModelRunner 侧，可以直接使用 GPU tensor，避免把 logits 搬到 Scheduler 再处理。

### 4.2 Scheduler 适合放“请求状态和资源账本”

Scheduler 才知道：

```text
- request 是否 running / waiting / finished；
- 本轮 SchedulerOutput 原计划；
- KV blocks / encoder cache / connector 状态；
- spec decode 原本调度了哪些 draft tokens；
- stop condition 触发后如何释放资源；
- EngineCoreOutputs 按 client_index 如何分组。
```

所以 worker 只返回结果，Scheduler 再做状态闭环。

### 4.3 OutputProcessor 适合放“用户输出适配”

OutputProcessor 更靠近 frontend，它持有 tokenizer 和 frontend RequestState。

它适合处理：

```text
- detokenize；
- stop string；
- DELTA / CUMULATIVE / FINAL_ONLY；
- RequestOutput / CompletionOutput 构造；
- AsyncLLM queue；
- LLMEngine 同步返回。
```

这样 Scheduler 可以保持 token-id 状态机，不和文本协议、OpenAI response、streaming delta 强耦合。

---

## 5. SamplingParams 如何进入 sampler

### 5.1 SamplingParams 是用户采样参数的源头

`SamplingParams` 定义在：`vllm/vllm/sampling_params.py:224`

核心字段包括：

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
routed_experts_prompt_start
repetition_detection
```

初始化后会做规范化和校验：

```text
- seed == -1 → None；
- stop / stop_token_ids / bad_words 的 None 归一化；
- logprobs=True → 1；
- prompt_logprobs=True → 1；
- stop string 输出需要保留 buffer；
- temperature 接近 0 时转 greedy，并重置 top_p/top_k/min_p；
- 校验 top_p、top_k、min_p、penalty、max_tokens 等范围。
```

源码位置：`vllm/vllm/sampling_params.py:429` 到 `vllm/vllm/sampling_params.py:551`

### 5.2 新 GPU sampler 用 stateful 方式保存参数

在新的 `vllm/vllm/v1/worker/gpu/model_runner.py` 路径中，新请求加入 batch 时：

```python
self.sampler.add_request(req_index, prompt_len, new_req_data.sampling_params)
self.prompt_logprobs_worker.add_request(req_id, req_index, new_req_data.sampling_params)
```

源码位置：`vllm/vllm/v1/worker/gpu/model_runner.py:801` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:809`

`Sampler.add_request()` 会把一个 `SamplingParams` 拆到多个状态对象：

```text
SamplingStates：
  temperature / top_p / top_k / min_p / seed / num_logprobs

PenaltiesState：
  repetition_penalty / frequency_penalty / presence_penalty

LogitBiasState：
  allowed_token_ids / logit_bias / min_tokens + stop_token_ids

BadWordsState：
  bad_words token 序列状态

LogprobTokenIdsState：
  用户指定 token ids 的 logprobs 请求
```

源码位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:56` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:70`

一句话：

```text
SamplingParams 是请求级配置，Sampler state 是 batch/GPU 可消费的运行时形态。
```

### 5.3 旧 SamplingMetadata 路径

旧 `vllm/vllm/v1/sample/sampler.py` 路径使用 `SamplingMetadata`。

`SamplingMetadata` 把每个请求的采样参数整理成一次 forward 可用的 batch 结构，包括：

```text
temperature / top_p / top_k
all_greedy / all_random
generators
max_num_logprobs
penalties
prompt_token_ids / output_token_ids
min_tokens
allowed_token_ids_mask
bad_words_token_ids
logitsprocs
logprob_token_ids
spec_token_ids
```

它和新 GPU sampler 的关系是：

```text
旧路径：SamplingParams → SamplingMetadata → Sampler.forward()
新路径：SamplingParams → sampler state → Sampler.__call__(logits, input_batch)
```

二者组织方式不同，但目标相同：

```text
把每个 request 的采样规则应用到当前 batch 的 logits 上。
```

---

## 6. logits / logprobs / prompt logprobs 的位置

### 6.1 logits 从 hidden states 计算

新的 GPU ModelRunner 中，采样前先取需要 logits 的 hidden states：

```python
sample_hidden_states = hidden_states[input_batch.logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

源码位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1043` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1044`

这说明：

```text
不是所有 hidden states 都计算 logits；
只对 input_batch.logits_indices 指定的位置计算 logits。
```

这样可以覆盖：

```text
- prefill：可能只需要最后位置 logits；
- chunked prefill：非最后 chunk 可能不输出 token；
- decode：通常每个 request 一个 logits；
- spec decode：target model 可能要验证多个 draft 位置；
- prompt_logprobs：可能需要额外 prompt 位置 logits。
```

### 6.2 sample logprobs 属于 sampled token

sample logprobs 和生成 token 绑定。

Sampler 根据用户请求的：

```text
logprobs
logprob_token_ids
logprobs_mode
```

计算 sampled token、top-k token 或指定 token 的 logprobs。

新 GPU sampler 中，判断是否需要 logprobs 的位置是：

```python
max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
max_per_req_token_ids = self.logprob_token_ids_state.max_num_token_ids(idx_mapping_np)
return_logprobs = max_num_logprobs != NO_LOGPROBS or max_per_req_token_ids > 0
```

源码位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:88` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:92`

之后通过 `compute_topk_logprobs()` 只计算需要返回的部分。

源码位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:104` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:118`

### 6.3 prompt logprobs 属于 prompt token

prompt logprobs 不是 sampled token 的 logprobs，而是 prompt token 在模型上下文中的概率。

新的 GPU ModelRunner 中，采样后会调用：

```python
prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
    self.model.compute_logits,
    hidden_states,
    input_batch,
    self.req_states.all_token_ids.gpu,
    self.req_states.num_computed_tokens.gpu,
    self.req_states.prompt_len.np,
)
```

源码位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1373` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1381`

所以要区分：

```text
sample logprobs：
  针对本轮新生成 token，由 sampler 输出。

prompt logprobs：
  针对 prompt token，由 PromptLogprobsWorker 计算，放入 ModelRunnerOutput.prompt_logprobs_dict。
```

---

## 7. Sampler 主流程

### 7.1 新 GPU sampler 的入口

新 GPU sampler 定义在：`vllm/vllm/v1/worker/gpu/sample/sampler.py:30`

入口：

```python
def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
```

源码位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:72`

它从 `InputBatch` 取出：

```text
expanded_idx_mapping：logits row 到 request state idx 的映射；
idx_mapping_np：当前 batch 的 request state idx；
cu_num_logits_np：expanded logits 的累积边界；
expanded_local_pos：expanded logits 内每个 token 的局部位置；
positions[input_batch.logits_indices]：当前 logits 对应位置；
input_ids[input_batch.logits_indices]：当前 logits 对应输入 token。
```

源码位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:77` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:82`

### 7.2 apply_sampling_params 的顺序

真正修改 logits 的顺序在 `apply_sampling_params()` 中。

源码位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:146`

处理顺序是：

```text
raw logits
  → copy 成 FP32 processed logits
  → logit_bias / allowed_token_ids / min_tokens
  → repetition / frequency / presence penalties
  → bad words mask
  → temperature
  → min_p
  → top_k / top_p
```

对应源码位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:156` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:196`

这个顺序可以理解为：

```text
约束类 mask / bias 先决定候选空间；
penalty 根据 prompt/output 历史调整分数；
temperature / min_p / top-k / top-p 最后塑造随机采样分布。
```

### 7.3 sample 的最终选择

新 GPU sampler 的 `sample()` 先处理除 top-k/top-p 以外的参数，再选择 FlashInfer 或原生路径：

```text
processed_logits = apply_sampling_params(..., skip_top_k_top_p=True)
  → get_top_k_top_p()
  → 如果满足条件，使用 flashinfer_sample()
  → 否则 apply_top_k_top_p() + gumbel_sample()
```

源码位置：`vllm/vllm/v1/worker/gpu/sample/sampler.py:198` 到 `vllm/vllm/v1/worker/gpu/sample/sampler.py:244`

关键判断：

```text
FlashInfer 路径不适合以下情况：
- 没有 top_k/top_p；
- 需要 processed_logprobs；
- batch 中有 greedy request；
- batch 中有显式 per-request seed。
```

### 7.4 greedy 和 random 的统一

新 GPU sampler 最终用 Gumbel-max 风格的 kernel：

```text
temperature == 0：
  不加 gumbel noise，argmax 就是 greedy。

temperature > 0：
  加 gumbel noise，再 argmax，相当于按分布随机采样。
```

旧 `vllm/vllm/v1/sample/sampler.py` 路径则更显式：

```text
1. 如果不是 all_random，先 greedy argmax；
2. 如果 all_greedy，直接返回；
3. 否则应用 temperature；
4. 应用 min_p / top_k / top_p；
5. 随机采样；
6. 对 batch 中 greedy 行和 random 行做混合。
```

源码位置：`vllm/vllm/v1/sample/sampler.py:243` 到 `vllm/vllm/v1/sample/sampler.py:302`

---

## 8. structured output / grammar bitmask

结构化输出不是采样后修正，而是在采样前限制 logits 的可选空间。

### 8.1 grammar bitmask 由 Scheduler 生成

Scheduler 在执行结果返回前，先根据本轮 `SchedulerOutput` 生成 grammar bitmask：

```python
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
```

源码位置：`vllm/vllm/v1/engine/core.py:492`

`Scheduler.get_grammar_bitmask()` 会：

```text
1. 找出本轮 scheduled 且使用 structured output 的 request；
2. 排除仍在 prefill chunk 中不需要采样的请求；
3. 调用 structured_output_manager.grammar_bitmask()；
4. 返回 GrammarOutput(structured_output_request_ids, bitmask)。
```

源码位置：`vllm/vllm/v1/core/sched/scheduler.py:1440` 到 `vllm/vllm/v1/core/sched/scheduler.py:1462`

### 8.2 bitmask 在采样前应用

新 GPU ModelRunner 中：

```python
if grammar_output is not None:
    self.structured_outputs_worker.apply_grammar_bitmask(
        logits,
        input_batch,
        grammar_output.structured_output_request_ids,
        grammar_output.grammar_bitmask,
    )
```

源码位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1045` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1053`

`StructuredOutputsWorker.apply_grammar_bitmask()` 会把 CPU bitmask 异步复制到 GPU，再把不允许的 token logits 写成 `-inf`。

源码位置：`vllm/vllm/v1/worker/gpu/structured_outputs.py:23` 到 `vllm/vllm/v1/worker/gpu/structured_outputs.py:80`

因此顺序是：

```text
compute_logits()
  → grammar bitmask 屏蔽非法 token
  → sampler 内部 allowed_token_ids / bad_words / penalties / top-k / top-p
  → sampled_token_ids
```

### 8.3 grammar 状态由 Scheduler 推进

Worker 只负责“按 bitmask 采样”。

采样结果返回后，Scheduler 会调用 grammar 接受新 token：

```text
new_token_ids
  → structured_output_manager.should_advance(request)
  → grammar.accept_tokens(req_id, new_token_ids)
```

源码位置：`vllm/vllm/v1/core/sched/scheduler.py:1599` 到 `vllm/vllm/v1/core/sched/scheduler.py:1615`

也就是说：

```text
Scheduler 负责 grammar FSM 状态；
ModelRunner 负责把本轮 grammar bitmask 应用到 logits。
```

---

## 9. speculative decoding 对 sampling / output 的影响

Spec decode 会让“一轮一个请求输出几个 token”变得不固定。

### 9.1 普通 sampler 和 rejection sampler 的分叉

新 GPU ModelRunner 中：

```python
if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
    sampler_output = self.sampler(logits, input_batch)
else:
    sampler_output = self.rejection_sampler(
        logits,
        input_batch,
        self.speculator.draft_logits,
    )
```

源码位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1055` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1067`

含义是：

```text
普通采样：
  target logits 直接选下一个 token。

spec decode：
  target logits 用于验证 draft tokens，接受一部分，拒绝后可能重采 recovered token，还可能补 bonus token。
```

旧 `RejectionSampler` 的注释把术语说得很清楚：

```text
accepted tokens：按 draft / target 概率关系被接受的 token；
recovered tokens：拒绝后从调整分布中重新采样的 token；
bonus tokens：全部 draft 接受后额外补的 token；
output tokens = accepted + recovered + bonus。
```

源码位置：`vllm/vllm/v1/sample/rejection_sampler.py:37` 到 `vllm/vllm/v1/sample/rejection_sampler.py:58`

### 9.2 Scheduler 修正 accepted / rejected 状态

Scheduler 消费输出时，会把本轮原本 scheduled 的 draft tokens 和实际 generated tokens 对账：

```text
scheduled_spec_token_ids
  + generated_token_ids
  → num_accepted
  → num_rejected
  → 修正 request.num_computed_tokens
  → 修正 request.num_output_placeholders
  → 记录 spec decoding stats
```

源码位置：`vllm/vllm/v1/core/sched/scheduler.py:1548` 到 `vllm/vllm/v1/core/sched/scheduler.py:1575`

这解释了为什么 `ModelRunnerOutput.sampled_token_ids` 是 `list[list[int]]`：

```text
普通 decode：每个 request 通常 1 个 token；
spec decode：每个 request 一轮可能接受多个 token；
chunked prefill / partial 情况：也可能没有输出 token。
```

### 9.3 draft proposal 是下一轮准备

采样后，ModelRunner 还可能调用 speculator 生成下一轮 draft tokens：

```text
current sampled tokens / hidden states
  → speculator.propose(...)
  → req_states.draft_tokens
  → EngineCore.post_step() / Scheduler.update_draft_token_ids()
  → 下一轮 SchedulerOutput.scheduled_spec_decode_tokens
```

源码位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1429` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1461`

这部分不是用户输出本身，而是下一轮 spec decode 的输入准备。

---

## 10. ModelRunnerOutput 是什么

`ModelRunnerOutput` 是 Worker / ModelRunner 返回给 Scheduler 的 batch 级内部协议对象。

源码位置：`vllm/vllm/v1/outputs.py:234`

它的设计目标是：

```text
- 跨进程可传输；
- 避免直接传大量 GPU tensor；
- 便于 Scheduler 按 request 对账；
- 同时携带 generation、pooling、connector、debug、metrics 等执行结果。
```

主要字段：

```text
req_ids：
  当前 batch 中 request id 的顺序。

req_id_to_index：
  request id → batch index。

sampled_token_ids：
  list[list[int]]，每个 request 本轮生成的 token ids。

logprobs：
  LogprobsLists | None，生成 token 的 logprobs。

prompt_logprobs_dict：
  req_id → LogprobsTensors | None，prompt token 的 logprobs。

pooler_output：
  pooling / embedding / classification / reward 等非 generation 输出。

kv_connector_output：
  KV transfer / connector 输出。

ec_connector_output：
  encoder cache / multimodal connector 输出。

num_nans_in_logits：
  logits NaN 统计。

cudagraph_stats：
  CUDA graph 执行统计。

routed_experts：
  MoE routed experts 信息。
```

源码位置：`vllm/vllm/v1/outputs.py:235` 到 `vllm/vllm/v1/outputs.py:281`

它有两个重要特点：

### 10.1 它是“Worker → Scheduler 的对账凭证”

Scheduler 使用：

```text
req_id_to_index[req_id]
  → sampled_token_ids[index]
  → logprobs.slice_request(index, num_tokens)
  → prompt_logprobs_dict[req_id]
  → pooler_output[index]
```

把 batch 级 worker 输出重新对回 request。

### 10.2 它不是用户输出

`ModelRunnerOutput` 仍然只是内部结果：

```text
- token ids 还没有 detokenize；
- stop string 还没在 frontend 侧检查；
- logprobs 还不是用户 API 格式；
- RequestOutput / CompletionOutput 还没有构造；
- streaming delta 还没有切分。
```

真正用户可见的输出还要经过：

```text
Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput
```

---

## 11. AsyncOutput / GPU 到 CPU 拷贝

采样后，很多结果仍在 GPU 上：

```text
sampled_token_ids
logprobs_tensors
prompt_logprobs_tensors
num_nans
pooler_output
routed_experts
```

如果每轮都在主流上同步拷贝，会阻塞后续计算。

新的 GPU path 使用 `AsyncOutput`：

```python
async_output = AsyncOutput(
    model_runner_output=model_runner_output,
    sampler_output=sampler_output,
    num_sampled_tokens=num_sampled,
    main_stream=self.main_stream,
    copy_stream=self.output_copy_stream,
)
```

源码位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1392` 到 `vllm/vllm/v1/worker/gpu/model_runner.py:1399`

`AsyncOutput` 做的事情：

```text
__init__：
  在 copy stream 上等待 main stream；
  异步拷贝 sampled_token_ids / logprobs / num_nans / prompt_logprobs；
  记录 copy_event。

get_output()：
  等待 copy_event；
  sampled_token_ids 转 list[list[int]]；
  按 num_sampled_tokens 删除 padding；
  logprobs_tensors 转 LogprobsLists；
  回填 ModelRunnerOutput。
```

源码位置：`vllm/vllm/v1/worker/gpu/async_utils.py:12` 到 `vllm/vllm/v1/worker/gpu/async_utils.py:69`

一句话：

```text
AsyncOutput 不是新的语义层，它只是让 GPU→CPU 输出拷贝和后续工作重叠，最终仍然返回 ModelRunnerOutput。
```

---

## 12. Scheduler.update_from_output 如何闭环

`Scheduler.update_from_output()` 是 worker 输出进入 request 状态机的入口。

源码位置：`vllm/vllm/v1/core/sched/scheduler.py:1464`

输入：

```text
scheduler_output：
  本轮原计划，包括 num_scheduled_tokens、scheduled_spec_decode_tokens、scheduled_encoder_inputs 等。

model_runner_output：
  Worker 实际执行结果，包括 sampled_token_ids、logprobs、prompt_logprobs、pooler_output、connector 输出等。
```

主流程可以压缩成：

```text
1. 取出 sampled_token_ids / logprobs / prompt_logprobs / pooler_output；
2. 处理 KV load failure 和 routed experts；
3. 遍历本轮 num_scheduled_tokens；
4. 用 req_id_to_index 找到 request 对应输出；
5. 如果有 spec decode，计算 accepted / rejected 并修正状态；
6. 释放本轮用完的 encoder inputs；
7. append generated tokens 并 check_stop；
8. pooling output 直接标记 request stopped；
9. 推进 structured output grammar；
10. stopped 时计算 finish_reason 并释放 request 资源；
11. 切分 sample logprobs；
12. 取 prompt_logprobs；
13. 构造 EngineCoreOutput；
14. 更新 KV connector 状态和 KV events；
15. 按 client_index 构造 EngineCoreOutputs。
```

关键源码位置：

- 遍历 scheduled requests：`vllm/vllm/v1/core/sched/scheduler.py:1527`
- 查找 request 输出 index：`vllm/vllm/v1/core/sched/scheduler.py:1543`
- spec decode 修正：`vllm/vllm/v1/core/sched/scheduler.py:1548`
- append token / stop check：`vllm/vllm/v1/core/sched/scheduler.py:1589`
- grammar advance：`vllm/vllm/v1/core/sched/scheduler.py:1599`
- logprobs slice：`vllm/vllm/v1/core/sched/scheduler.py:1671`
- EngineCoreOutput 构造：`vllm/vllm/v1/core/sched/scheduler.py:1690`
- EngineCoreOutputs 构造：`vllm/vllm/v1/core/sched/scheduler.py:1770`

`EngineCoreOutput` 是 request 级内部输出。

源码位置：`vllm/vllm/v1/engine/__init__.py:175`

主要字段：

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
prefill_stats
routed_experts
num_nans_in_logits
```

`EngineCoreOutputs` 是按 client 分组的一批输出。

源码位置：`vllm/vllm/v1/engine/__init__.py:220`

一句话：

```text
ModelRunnerOutput 是 batch 级执行结果，EngineCoreOutput 是 request 级引擎输出。
```

---

## 13. OutputProcessor 如何生成用户输出

`OutputProcessor` 是内部输出到用户输出的最后一层适配器。

源码位置：`vllm/vllm/v1/engine/output_processor.py:417`

主入口：`vllm/vllm/v1/engine/output_processor.py:576`

它的核心流程是：

```text
OutputProcessor.process_outputs(engine_core_outputs)
  → 遍历每个 EngineCoreOutput
  → 找到 frontend RequestState
  → 更新 stats
  → 如果是 generation：
       detokenizer.update(new_token_ids, ...)
       检查 stop string
       logprobs_processor.update_from_output(...)
       make_request_output(...)
  → 如果是 pooling：
       make_pooling_request_output(...)
  → 如果是 AsyncLLM：
       request_output 放入 per-request queue
  → 如果是 LLMEngine：
       加入同步返回列表
  → finished 后清理 RequestState
```

关键源码位置：

- 取 request state：`vllm/vllm/v1/engine/output_processor.py:606`
- detokenize：`vllm/vllm/v1/engine/output_processor.py:635`
- stop string check：`vllm/vllm/v1/engine/output_processor.py:639`
- logprobs update：`vllm/vllm/v1/engine/output_processor.py:648`
- make request output：`vllm/vllm/v1/engine/output_processor.py:651`
- AsyncLLM queue：`vllm/vllm/v1/engine/output_processor.py:661`
- LLMEngine return list：`vllm/vllm/v1/engine/output_processor.py:665`
- finish cleanup：`vllm/vllm/v1/engine/output_processor.py:668`

### 13.1 RequestOutput / CompletionOutput

用户可见 generation 输出定义在 `vllm/vllm/outputs.py`。

`CompletionOutput` 表示一个 completion 分支。

源码位置：`vllm/vllm/outputs.py:21`

主要字段：

```text
index
text
token_ids
cumulative_logprob
logprobs
routed_experts
finish_reason
stop_reason
lora_request
```

`RequestOutput` 表示一个用户请求。

源码位置：`vllm/vllm/outputs.py:85`

主要字段：

```text
request_id
prompt
prompt_token_ids
prompt_logprobs
outputs: list[CompletionOutput]
finished
metrics
lora_request
encoder_prompt
encoder_prompt_token_ids
num_cached_tokens
kv_transfer_params
```

### 13.2 output_kind 如何影响 streaming

`RequestState.make_request_output()` 会根据 `output_kind` 决定是否返回输出，以及返回 delta 还是累计值。

关键逻辑：

```text
FINAL_ONLY：
  未 finished 时不返回 RequestOutput。

stream_interval > 1：
  只有 finished、首 token、或达到 stream interval 时才返回。

DELTA：
  只返回 sent_tokens_offset 之后的新 token / text / logprobs。

CUMULATIVE：
  返回累计 token / text。
```

源码位置：`vllm/vllm/v1/engine/output_processor.py:280` 到 `vllm/vllm/v1/engine/output_processor.py:319`

`RequestOutput.add()` 还支持把后续增量输出合并到已有输出中。

源码位置：`vllm/vllm/outputs.py:145`

---

## 14. generation 和 pooling / embedding 的差异

### 14.1 generation 路径

```text
model forward
  → hidden_states
  → compute_logits(hidden_states[logits_indices])
  → grammar bitmask
  → Sampler / RejectionSampler
  → sampled_token_ids / logprobs
  → ModelRunnerOutput
  → Scheduler append tokens / check stop
  → OutputProcessor detokenize
  → RequestOutput / CompletionOutput
```

特点：

```text
- 有 logits；
- 有 sampling；
- 有 output_token_ids；
- 有 stop token / stop string / max_tokens；
- 用户输出是文本 completion。
```

### 14.2 pooling / embedding 路径

```text
model forward
  → hidden_states
  → pooler / pooling runner
  → pooler_output
  → ModelRunnerOutput.pooler_output
  → Scheduler 标记 request stopped
  → OutputProcessor 构造 PoolingRequestOutput
  → 上层可转换为 EmbeddingRequestOutput / ClassificationRequestOutput / ScoringRequestOutput 等
```

用户可见的 pooling 基类在：`vllm/vllm/outputs.py:204`

embedding 输出在：`vllm/vllm/outputs.py:240`

特点：

```text
- 不走普通 Sampler；
- 不生成 sampled_token_ids；
- 通常不做 completion detokenize；
- 输出是向量、score、classification 或 pooling 结果。
```

### 14.3 Pipeline Parallel 下的差异

非最后 PP rank：

```text
model forward
  → IntermediateTensors
  → 发给下一 PP stage
  → sample_tokens 阶段接收最后 rank 广播的 sampled tokens 并更新本地状态
```

最后 PP rank：

```text
hidden_states
  → generation: logits / sampling
  → pooling: pooler_output
```

所以不是所有 rank 都会真正计算 logits 或构造完整 sampling 输出。

---

## 15. 关键对象关系

### 15.1 `SamplingParams`

用户请求级采样配置。

```text
temperature / top_p / top_k / min_p / penalties / seed / stop / logprobs / structured_outputs / bad_words / allowed_token_ids ...
```

### 15.2 `InputBatch`

ModelRunner 侧的 batch 张量视图。

它告诉 sampler：

```text
logits 的每一行属于哪个 request；
每个 request 当前处于哪个位置；
哪些 token 需要 logits；
prefill / decode / spec decode 的边界在哪里。
```

### 15.3 `SamplerOutput`

worker 内部采样结果，偏 GPU tensor 形态。

新 GPU sampler 输出定义在：`vllm/vllm/v1/worker/gpu/sample/output.py:10`

```text
sampled_token_ids
logprobs_tensors
num_nans
num_sampled
num_rejected
```

### 15.4 `ModelRunnerOutput`

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

### 15.5 `EngineCoreOutput`

Scheduler → OutputProcessor 的 request 级协议对象。

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
```

### 15.6 `EngineCoreOutputs`

一个 client 的一批输出。

```text
engine_index
outputs: list[EngineCoreOutput]
scheduler_stats
finished_requests
timestamp
```

### 15.7 `CompletionOutput`

用户可见的一个 completion。

```text
index
text
token_ids
logprobs
cumulative_logprob
finish_reason
stop_reason
```

### 15.8 `RequestOutput`

用户可见的 generation 请求输出。

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

### 15.9 `PoolingRequestOutput`

用户可见的 pooling / embedding 类请求输出。

```text
request_id
outputs
prompt_token_ids
num_cached_tokens
finished
```

---

## 16. 常见疑惑

### 16.1 ModelRunnerOutput 是用户输出吗？

不是。

它是 Worker 返回 Scheduler 的内部结果，还没有 detokenize，也没有转成用户 API 结构。

### 16.2 sampled_token_ids 是文本吗？

不是。

它只是 token id。文本要由 OutputProcessor / Detokenizer 生成。

### 16.3 sample_tokens 是不是只做采样？

不是。

它还处理：

```text
grammar bitmask
spec decode
prompt logprobs
AsyncOutput
postprocess_sampled
speculator proposal
KV connector post_forward
ModelRunnerOutput 构造
```

### 16.4 Scheduler.update_from_output 是不是只转发 token？

不是。

它会更新 Request 状态、stop condition、grammar 状态、spec decode 状态、KV/encoder/connector 状态，并释放资源。

### 16.5 OutputProcessor 会重新采样吗？

不会。

它只做输出适配：detokenize、stop string、logprobs 格式化、RequestOutput 构造、streaming queue 推送。

### 16.6 structured output 是采样后修正吗？

不是。

grammar bitmask 在采样前应用，保证采样空间已经排除了非法 token。

### 16.7 为什么 sampled_token_ids 是 list[list[int]]？

因为每个 request 一轮输出 token 数可能不同：

```text
普通 decode：通常 1 个；
spec decode：可能多个 accepted/recovered/bonus token；
chunked prefill：可能 0 个；
invalid / partial 情况：也可能为空。
```

### 16.8 为什么 detokenize 不放在 Scheduler？

因为 detokenize 和 tokenizer、stop string、streaming delta、OpenAI 输出格式更接近 frontend；Scheduler 只需要维护 token id 状态机和资源账本。

---

## 17. 推荐阅读路线

### 17.1 快速建立全局印象

```text
sampling_and_output_overview.md
  → 01_sampling_output_role.md
  → 07_model_runner_output.md
  → 09_output_processor_request_output.md
```

### 17.2 按模型输出链路阅读

```text
../executor_worker_model_runner/07_model_forward_and_logits.md
  → 03_logits_and_logprobs.md
  → 04_sampler_flow.md
  → 07_model_runner_output.md
  → 08_scheduler_update_output.md
  → 09_output_processor_request_output.md
```

### 17.3 按高级能力阅读

```text
04_sampler_flow.md
  → 05_spec_decode_sampling.md
  → 06_structured_output_and_grammar.md
  → 10_streaming_and_client_outputs.md
  → 11_pooling_and_embedding_outputs.md
```

### 17.4 和 Scheduler / Executor 联动阅读

```text
../executor_worker_model_runner/executor_worker_model_runner_overview.md
  → ../executor_worker_model_runner/07_model_forward_and_logits.md
  → ../executor_worker_model_runner/08_sampling_and_model_runner_output.md
  → ../scheduler/08_update_after_worker_output.md
  → sampling_and_output_overview.md
```

---

## 18. 文档定位

```text
sampling_and_output_overview.md：
  总览主文档，建立从 hidden states / logits 到用户输出的全局图。

01-11：
  按问题拆开的专题文档，适合逐段精读 sampling、ModelRunnerOutput、Scheduler update、OutputProcessor 和 streaming / pooling 输出。
```

---

## 19. 一句话总结

Sampling / Output 是 vLLM 推理链路的“后半段闭环”：

```text
ModelRunner / Sampler 把模型结果变成 sampled tokens 或 pooling output，
Scheduler 把这些内部结果写回 request 状态机并释放资源，
OutputProcessor 把 request 状态变成用户可见输出。
```

最核心的主线是：

```text
hidden_states
  → compute_logits
  → grammar bitmask
  → Sampler / RejectionSampler
  → SamplerOutput
  → AsyncOutput / ModelRunnerOutput
  → Scheduler.update_from_output
  → EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs
  → RequestOutput / CompletionOutput / PoolingRequestOutput
```
