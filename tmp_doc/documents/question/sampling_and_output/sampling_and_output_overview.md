# vLLM V1 Sampling / Output 逻辑梳理

源码位置：

- `code/vllm/vllm/sampling_params.py`
- `code/vllm/vllm/outputs.py`
- `code/vllm/vllm/logprobs.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/v1/engine/logprobs.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/sample/sampler.py`
- `code/vllm/vllm/v1/worker/gpu/sample/logprob.py`
- `code/vllm/vllm/v1/worker/gpu/sample/prompt_logprob.py`
- `code/vllm/vllm/v1/worker/gpu/sample/output.py`
- `code/vllm/vllm/v1/sample/sampler.py`
- `code/vllm/vllm/v1/sample/ops/topk_topp_sampler.py`
- `code/vllm/vllm/v1/sample/ops/logprobs.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/structured_outputs.py`

本文按“先定边界，再走主链路，再拆关键阶段，最后总结接口和数据结构”的方式，梳理 vLLM V1 中 sampling 与 output 的关系。

它接在 `executor_worker_model_runner` 的 `07_model_forward_and_logits.md`、`08_sampling_and_model_runner_output.md` 后面，专门展开模型 forward 之后的后半段：

```text
hidden states / logits
  → logprobs / prompt logprobs
  → sampler
  → sampled token ids
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor
  → RequestOutput / CompletionOutput / Chat output / streaming chunk
```

---

## 0. 梳理规划

本目录要回答的问题分成 11 组：

```text
1. Sampling / Output 在 vLLM 全链路中处于哪一段？
2. SamplingParams 如何从用户参数变成 worker / sampler 可消费的 metadata？
3. logits、logprobs、prompt logprobs 分别在哪里计算？
4. sampler 如何处理 temperature / top-k / top-p / penalties / seed？
5. spec decode 如何改变采样和输出回收？
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

---

## 1. 一句话回答

Sampling / Output 这段链路负责把模型内部计算结果变成上层可见输出。

可以压缩成：

```text
ModelRunner 负责“从 hidden states 采样出 token”；
Scheduler 负责“把 token 消化回 request 状态机”；
OutputProcessor 负责“把内部 request 状态转换成用户可见输出”。
```

最小主线是：

```text
GPUModelRunner.execute_model()
  → _model_forward()
  → compute_logits()
  → sample_tokens()
  → sampler(...)
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / streaming output
```

---

## 2. 三层边界

### 2.1 ModelRunner / sampler 层

这一层负责设备侧采样。

```text
输入：
  hidden states / logits / sampling metadata / grammar bitmask / spec decode metadata

输出：
  sampled_token_ids / logprobs / prompt_logprobs / pooler_output / routed experts 等
```

它关心的是：

```text
- 哪些位置要算 logits；
- 哪些 logits 要参与采样；
- sampling params 如何应用；
- grammar bitmask 是否限制候选 token；
- spec decode 是否要做接受 / 拒绝；
- logprobs / prompt logprobs 是否要返回。
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
- sampled token 是否 append 到 request；
- stop condition 是否触发；
- spec decode token 如何接受 / 回退；
- KV blocks / encoder cache / connector metadata 是否释放；
- grammar / structured output 状态如何推进；
- 本轮哪些 request 有增量输出。
```

### 2.3 OutputProcessor / client output 层

这一层负责把 EngineCore 内部输出转换为用户 API 可见结构。

```text
输入：
  EngineCoreOutputs / request state / tokenizer / output kind

输出：
  RequestOutput / CompletionOutput / EmbeddingRequestOutput / streaming chunks
```

它关心的是：

```text
- detokenize；
- incremental output；
- finished reason；
- logprobs 格式；
- prompt logprobs 格式；
- streaming 时本轮返回哪些 delta；
- OpenAI API 兼容层最终如何消费。
```

---

## 3. 总体流程图

```text
用户输入 SamplingParams
  → Engine / InputProcessor 校验和归一化
  → Scheduler 持有 request.sampling_params
  → SchedulerOutput 告诉 Worker 本轮跑哪些 token
  → ModelRunner.forward 得到 hidden states
  → compute_logits(hidden_states, logits_indices)
  → sample_tokens(grammar_bitmask)
      → SamplingMetadata
      → penalties / temperature / top-k / top-p / min-p
      → structured output grammar mask
      → sample / greedy / random seed
      → logprobs / prompt logprobs
      → spec decode rejection / acceptance
  → ModelRunnerOutput
      → sampled_token_ids
      → logprobs
      → prompt_logprobs_dict
      → pooler_output
      → kv_connector_output
  → Scheduler.update_from_output()
      → append token
      → update num_computed_tokens
      → check stop
      → update grammar / spec decode / KV states
      → produce EngineCoreOutputs
  → OutputProcessor
      → detokenize
      → build RequestOutput
      → streaming / final output
```

---

## 4. 后续专题占位

```text
01_sampling_output_role.md：
  定义 sampling / output 在全链路中的边界，说明它和 EngineCore、Scheduler、ModelRunner、OutputProcessor 的关系。

02_sampling_params_and_metadata.md：
  梳理 SamplingParams、SamplingMetadata、per-request sampling 参数如何进入 worker 和 sampler。

03_logits_and_logprobs.md：
  梳理 hidden states → logits → logprobs / prompt logprobs 的计算路径。

04_sampler_flow.md：
  梳理 sampler 如何应用 temperature、top-k、top-p、min-p、penalties、seed，并产出 token。

05_spec_decode_sampling.md：
  梳理 speculative decoding 下 draft / target / bonus logits、rejection sampler 和输出修正。

06_structured_output_and_grammar.md：
  梳理 structured output / grammar bitmask 如何限制采样以及状态如何推进。

07_model_runner_output.md：
  梳理 ModelRunnerOutput 的字段、语义，以及为什么它还不是用户输出。

08_scheduler_update_output.md：
  梳理 Scheduler.update_from_output() 如何消费 worker 输出并推进 request 状态机。

09_output_processor_request_output.md：
  梳理 OutputProcessor 如何构造 RequestOutput、CompletionOutput、logprobs、finished reason。

10_streaming_and_client_outputs.md：
  梳理 streaming / delta output / final output / OpenAI server 输出的关系。

11_pooling_and_embedding_outputs.md：
  梳理 pooling、embedding、rerank 等非生成式输出与采样输出的分叉。
```

---

## 5. 一句话总结

Sampling / Output 是 vLLM 推理链路的“后半段闭环”：

```text
ModelRunner 把模型结果变成 sampled tokens，
Scheduler 把 sampled tokens 变成 request 状态推进，
OutputProcessor 把 request 状态变成用户可见输出。
```
