# vLLM Sampling 与 Output 完整梳理

本目录用于系统梳理 `D:/lzy/project/kv_pool/code/vllm` 中 vLLM 的采样参数、GPU 侧 sampling、structured output、logprobs、scheduler 输出整理、frontend OutputProcessor、最终 RequestOutput/OpenAI response 的完整链路。

## 文档导航

建议按以下顺序阅读：

1. [01_sampling_and_output_overview.md](01_sampling_and_output_overview.md)
   - 先建立全局分层：请求参数、EngineCore、GPU sampler、Scheduler、OutputProcessor、OpenAI serving 的边界。

2. [02_sampling_params_and_protocol_conversion.md](02_sampling_params_and_protocol_conversion.md)
   - 梳理 OpenAI Completion/Chat/Responses 请求如何转为 `SamplingParams`，以及字段校验、默认值、结构化输出参数。

3. [03_request_enqueue_and_parallel_sampling.md](03_request_enqueue_and_parallel_sampling.md)
   - 梳理 `InputProcessor`、`EngineCoreRequest`、内部 `Request`，以及 `n > 1` parallel sampling 如何拆 child request。

4. [04_gpu_sampling_execution_flow.md](04_gpu_sampling_execution_flow.md)
   - 梳理模型 forward 后 hidden states 如何变 logits，如何进入普通 sampler 或 spec decode rejection sampler。

5. [05_LogitsProcessor_Logprobs_StructuredOutput.md](05_LogitsProcessor_Logprobs_StructuredOutput.md)
   - 梳理 logits processor、allowed token、bad words、penalty、logprobs、grammar bitmask、reasoning gate。

6. [06_model_runner_output_to_engine_core_outputs.md](06_model_runner_output_to_engine_core_outputs.md)
   - 梳理 GPU/worker 输出如何进入 scheduler，并被整理成 `EngineCoreOutput(s)`。

7. [07_output_processor_and_request_output.md](07_output_processor_and_request_output.md)
   - 梳理 frontend 输出处理、detokenize、stop string、流式/非流式、logprobs 累积、collector。

8. [08_openai_returns_and_debug_map.md](08_openai_returns_and_debug_map.md)
   - 梳理 Completion/Chat serving 如何消费 `RequestOutput`，并给出按问题定位文件的调试地图。

9. [question_and_answer.md](question_and_answer.md)
   - 常见问题问答：为什么输出分三层、stop 在哪里判断、structured output 为什么影响 sampler 等。

## 一句话主链路

```text
OpenAI/Python 请求
  -> CompletionRequest/ChatCompletionRequest.to_sampling_params()
  -> SamplingParams
  -> InputProcessor.process_inputs()
  -> EngineCoreRequest
  -> Request
  -> Scheduler.schedule()
  -> GPUModelRunner.execute_model()
  -> hidden_states
  -> model.compute_logits()
  -> structured output grammar bitmask
  -> Sampler/RejectionSampler
  -> ModelRunnerOutput
  -> Scheduler.update_from_output()
  -> EngineCoreOutputs
  -> OutputProcessor.process_outputs()
  -> RequestOutput/CompletionOutput
  -> OpenAI-compatible response 或 LLM/AsyncLLM 输出
```

## 核心结论

- `SamplingParams` 是协议层到 engine 层的采样语义载体，定义在 `code/vllm/vllm/sampling_params.py:199`。
- V1 中用户可见输出不是 worker 直接产生的；worker 只产生 `ModelRunnerOutput`，还要经过 scheduler 与 frontend output processor。
- 输出链路至少有三层：`ModelRunnerOutput`、`EngineCoreOutputs`、`RequestOutput`。
- stop token / length / repetition 主要在 scheduler 检查；stop string 主要在 frontend detokenizer/output processor 检查。
- structured output 不是在文本输出后校验，而是在采样前通过 grammar bitmask 直接屏蔽非法 token logits。
- `n > 1` 在 V1 中会拆成多个 child request，再由 parent request 聚合输出。

## 重要代码锚点

- `code/vllm/vllm/sampling_params.py:199`：`SamplingParams`。
- `code/vllm/vllm/entrypoints/openai/completion/protocol.py:244`：Completion 请求转 `SamplingParams`。
- `code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:549`：Chat 请求转 `SamplingParams`。
- `code/vllm/vllm/v1/engine/input_processor.py:242`：`InputProcessor.process_inputs()`。
- `code/vllm/vllm/v1/engine/parallel_sampling.py:13`：`ParentRequest`。
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1038`：新 GPU runner 的 `sample()`。
- `code/vllm/vllm/v1/worker/gpu/sample/sampler.py:72`：新 GPU sampler 入口。
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:43`：spec decode rejection sampler。
- `code/vllm/vllm/v1/outputs.py:234`：`ModelRunnerOutput`。
- `code/vllm/vllm/v1/core/sched/scheduler.py:1463`：`Scheduler.update_from_output()`。
- `code/vllm/vllm/v1/engine/output_processor.py:576`：`OutputProcessor.process_outputs()`。
- `code/vllm/vllm/outputs.py:21`：`CompletionOutput`。
- `code/vllm/vllm/outputs.py:85`：`RequestOutput`。
