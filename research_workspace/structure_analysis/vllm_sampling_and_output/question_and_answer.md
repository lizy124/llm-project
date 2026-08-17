# Sampling 与 Output 常见问题问答

## Q1：vLLM 中真正执行采样的是哪里？

真正执行采样的是 worker/GPU 侧 sampler。

新 GPU worker 主链路：

```text
GPUModelRunner.sample()
  -> model.compute_logits()
  -> StructuredOutputsWorker.apply_grammar_bitmask()
  -> Sampler 或 RejectionSampler
```

关键位置：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1038`
- `code/vllm/vllm/v1/worker/gpu/sample/sampler.py:72`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:43`

`OutputProcessor` 不采样，它只处理输出转换。

## Q2：`SamplingParams` 是不是最终采样时直接使用的对象？

不是完全直接。

`SamplingParams` 先由协议层构造，然后在 `InputProcessor.process_inputs()` 中 clone、校验和补全，最后进入 `EngineCoreRequest` 和内部 `Request`。

后续 worker 会把 `Request.sampling_params` 拆成 batch/state 级结构，例如：

- 新 GPU worker 的 `SamplingStates`、`LogitBiasState`、`PenaltiesState`、`BadWordsState`；
- 旧路径的 `SamplingMetadata`。

相关位置：

- `code/vllm/vllm/sampling_params.py:199`
- `code/vllm/vllm/v1/engine/input_processor.py:242`
- `code/vllm/vllm/v1/worker/gpu/sample/states.py:17`
- `code/vllm/vllm/v1/sample/metadata.py:14`

## Q3：OpenAI Chat 的 `logprobs` 为什么不是直接传数字？

Chat Completion 中 OpenAI 语义通常是：

- `logprobs`：布尔值，表示是否返回；
- `top_logprobs`：返回多少个 top logprobs。

所以 chat 协议转换时，传给 `SamplingParams.logprobs` 的是：

```text
top_logprobs if logprobs else None
```

位置：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:629`

## Q4：Completion 和 Chat 的 streaming 为什么返回增量？

协议转换时会设置 `SamplingParams.output_kind`：

- streaming：`RequestOutputKind.DELTA`；
- non-streaming：`RequestOutputKind.FINAL_ONLY`。

位置：

- completion：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:337`
- chat：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:637`

OutputProcessor 根据 `output_kind` 决定返回增量还是最终输出。

## Q5：为什么 non-streaming 中间没有输出？

因为 non-streaming 通常设置为 `FINAL_ONLY`。

`RequestState.make_request_output()` 中，如果请求未完成且 output kind 是 `FINAL_ONLY`，会直接返回 None。

位置：`code/vllm/vllm/v1/engine/output_processor.py:280`

## Q6：`ModelRunnerOutput`、`EngineCoreOutputs`、`RequestOutput` 有什么区别？

三者层级不同：

1. `ModelRunnerOutput`：worker/model runner 的底层输出，包含 sampled token ids、logprobs、pooler output 等。
2. `EngineCoreOutputs`：scheduler 整理后的内部输出，按 request/client 分组，带 finish reason。
3. `RequestOutput`：用户/API 可见输出，带 detokenized text、CompletionOutput、metrics 等。

位置：

- `ModelRunnerOutput`：`code/vllm/vllm/v1/outputs.py:234`
- `EngineCoreOutputs`：`code/vllm/vllm/v1/engine/__init__.py:218`
- `RequestOutput`：`code/vllm/vllm/outputs.py:85`

## Q7：finish_reason 在哪里决定？

大多数 token 级 finish reason 由 scheduler 决定。

流程：

```text
check_stop()
  -> RequestStatus
  -> Request.get_finished_reason()
  -> EngineCoreOutput.finish_reason
  -> CompletionOutput.finish_reason
  -> OpenAI response finish_reason
```

关键位置：

- `check_stop()`：`code/vllm/vllm/v1/core/sched/utils.py:94`
- `Request.get_finished_reason()`：`code/vllm/vllm/v1/request.py:353`
- `EngineCoreOutput.finish_reason`：`code/vllm/vllm/v1/engine/__init__.py:187`

但 stop string 是例外，它由 frontend detokenizer/output processor 在文本层检查。

## Q8：stop token 和 stop string 的区别是什么？

stop token 是 token id 层面的停止条件，scheduler 可以直接判断。

stop string 是文本层面的停止条件，需要 detokenize 后才能判断，所以在 frontend 处理。

位置：

- stop token ids：`code/vllm/vllm/v1/core/sched/utils.py:108`
- stop string：`code/vllm/vllm/v1/engine/detokenizer.py:309`
- OutputProcessor 处理 stop string：`code/vllm/vllm/v1/engine/output_processor.py:639`

## Q9：`min_tokens` 如何防止过早停止？

它有两层影响：

1. sampler 前会 mask EOS/stop token ids，避免采样到这些 token；
2. scheduler 的 `check_stop()` 在未达到 `min_tokens` 时也不会停止。

位置：

- sampler mask：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:40`
- scheduler check：`code/vllm/vllm/v1/core/sched/utils.py:100`

## Q10：structured output 是输出后校验 JSON 吗？

不是。

structured output 的核心是在采样前生成 grammar bitmask，然后把非法 token 的 logits 置为 `-inf`。

流程：

```text
StructuredOutputsParams
  -> StructuredOutputRequest
  -> StructuredOutputManager.grammar_bitmask()
  -> GrammarOutput
  -> StructuredOutputsWorker.apply_grammar_bitmask()
  -> Sampler
```

位置：

- 参数：`code/vllm/vllm/sampling_params.py:71`
- request：`code/vllm/vllm/v1/structured_output/request.py:21`
- bitmask：`code/vllm/vllm/v1/structured_output/__init__.py:204`
- GPU 应用：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:23`

## Q11：structured output 和 spec decode 如何配合？

spec decode 下，一个请求可能一次验证多个 draft token，所以 grammar bitmask 也需要覆盖多个位置。

`StructuredOutputManager.grammar_bitmask()` 会为 speculative token positions 和 bonus token position 生成 mask，并在内部临时推进 grammar 后 rollback。

位置：

- spec decode logits row 展开：`code/vllm/vllm/v1/worker/gpu/model_runner.py:879`
- grammar speculative 处理：`code/vllm/vllm/v1/structured_output/__init__.py:275`
- rejection sampler：`code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:43`

## Q12：reasoning 为什么会影响 structured output？

有些模型会先输出 reasoning，再输出最终结构化内容。此时如果一开始就套 JSON/grammar 约束，可能会禁止 reasoning token。

所以 `StructuredOutputManager` 可以通过 reasoning parser 判断：

- reasoning 阶段是否填 grammar bitmask；
- reasoning 结束后何时推进 grammar FSM。

位置：

- reasoning parser：`code/vllm/vllm/reasoning/abs_reasoning_parsers.py:26`
- `should_fill_bitmask()`：`code/vllm/vllm/v1/structured_output/__init__.py:305`
- `should_advance()`：`code/vllm/vllm/v1/structured_output/__init__.py:325`

## Q13：`n > 1` 是怎么实现的？

V1 中 `n > 1` 通常由 `ParentRequest` fan-out 为多个 child request。

每个 child：

- `n` 被改成 1；
- request id 形如 `{index}_{parent_request_id}`；
- 如果 parent 有 seed，则 child seed 是 `seed + index`。

位置：

- `ParentRequest`：`code/vllm/vllm/v1/engine/parallel_sampling.py:13`
- child params：`code/vllm/vllm/v1/engine/parallel_sampling.py:52`
- child id：`code/vllm/vllm/v1/engine/parallel_sampling.py:83`
- 聚合输出：`code/vllm/vllm/v1/engine/parallel_sampling.py:100`

## Q14：logprobs 是在哪里算的？

新 GPU worker 中，logprobs 在 GPU sampler 侧计算。

位置：

- `compute_token_logprobs()`：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:78`
- `compute_topk_logprobs()`：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:101`

然后：

1. GPU `LogprobsTensors` 异步拷到 CPU；
2. scheduler slice 每个 request 的 logprobs；
3. frontend `LogprobsProcessor` 转成用户可见格式。

位置：

- D2H：`code/vllm/vllm/v1/worker/gpu/async_utils.py:34`
- scheduler slice：`code/vllm/vllm/v1/core/sched/scheduler.py:1669`
- frontend：`code/vllm/vllm/v1/engine/logprobs.py:29`

## Q15：为什么不直接计算完整 vocab logprobs？

完整 `[batch_size, vocab_size]` logprobs 显存和带宽成本高。

vLLM 尽量只计算：

- sampled token；
- top-k token；
- 用户指定 `logprob_token_ids`。

相关实现：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:81`

## Q16：`allowed_token_ids` 和 structured output 有什么区别？

二者都会限制可采样 token，但来源和粒度不同：

- `allowed_token_ids` 是用户直接给的一组 token id，通常每步固定；
- structured output 是 grammar/FSM 根据当前已生成内容动态给出 bitmask。

位置：

- allowed token ids：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:19`
- structured grammar bitmask：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:23`

## Q17：OpenAI Chat tool call 的 finish_reason 从哪里来？

底层 `CompletionOutput.finish_reason` 通常来自 engine/output processor。

但 Chat serving 层如果识别到工具调用，可能把最终 OpenAI response 的 finish reason 改为 `tool_calls`。

位置：

- chat streaming：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:677`
- chat non-streaming：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:1005`

## Q18：排查“采样参数没生效”先看哪里？

建议顺序：

1. 协议转换：
   - `code/vllm/vllm/entrypoints/openai/completion/protocol.py:244`
   - `code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:549`
2. `SamplingParams.__post_init__()` 是否改写：
   - `code/vllm/vllm/sampling_params.py:429`
3. `InputProcessor.process_inputs()` 是否补全/更新：
   - `code/vllm/vllm/v1/engine/input_processor.py:313`
4. GPU sampler 是否应用：
   - `code/vllm/vllm/v1/worker/gpu/sample/sampler.py:146`

## Q19：排查“生成了 token 但 API 没返回”先看哪里？

建议顺序：

1. `ModelRunnerOutput.sampled_token_ids`：`code/vllm/vllm/v1/outputs.py:240`
2. `Scheduler.update_from_output()`：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`
3. `EngineCoreOutput` 构造：`code/vllm/vllm/v1/core/sched/scheduler.py:1688`
4. `OutputProcessor.process_outputs()`：`code/vllm/vllm/v1/engine/output_processor.py:576`
5. `RequestState.make_request_output()`：`code/vllm/vllm/v1/engine/output_processor.py:272`
6. OpenAI serving response 构造：
   - completion：`code/vllm/vllm/entrypoints/openai/completion/serving.py:393`
   - chat：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:677`

## Q20：一句话理解完整链路？

`SamplingParams` 描述怎么采样，GPU sampler 真正选 token，scheduler 把 worker token 更新进请求状态并判断 token 级停止，OutputProcessor detokenize 并处理 stop string/logprobs/streaming，OpenAI serving 最后把 `RequestOutput` 包装成协议响应。
