# 08 OpenAI 返回与调试地图

本篇梳理 OpenAI serving 层如何消费 `RequestOutput`，以及排查 sampling/output 问题时应该从哪些文件入手。

## 1. OpenAI serving 层的位置

OpenAI-compatible API 最终消费的是 `RequestOutput` / `CompletionOutput`，不是 `ModelRunnerOutput` 或 `EngineCoreOutput`。

常见文件：

- Completion serving：`code/vllm/vllm/entrypoints/openai/completion/serving.py`
- Chat serving：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py`
- Responses serving：`code/vllm/vllm/entrypoints/openai/responses/serving.py`

## 2. Completion streaming

文件：`code/vllm/vllm/entrypoints/openai/completion/serving.py`

streaming 输出构造位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:393`

它会读取：

- `output.text`；
- `output.token_ids`；
- `output.logprobs`；
- `output.finish_reason`；
- `output.stop_reason`。

然后构造 `CompletionResponseStreamChoice`。

## 3. Completion non-streaming

non-streaming 会等待 final `RequestOutput`，再构造完整 response。

参数侧：Completion 请求 non-streaming 时 `output_kind=FINAL_ONLY`，因此 `OutputProcessor` 中间不会持续返回输出。

转换入口：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:244`

serving 主流程：`code/vllm/vllm/entrypoints/openai/completion/serving.py:129`

## 4. Chat streaming

文件：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py`

streaming 输出构造位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:677`

逻辑：

- 未 finished：
  - `finish_reason=None`；
  - 返回 delta message；
  - 可附带 logprobs/token ids。
- finished：
  - 如果工具调用，finish reason 可能改为 `tool_calls`；
  - 否则使用 `output.finish_reason`；
  - 透传 `output.stop_reason`。

## 5. Chat non-streaming

位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:1005`

逻辑：

- 构造 `ChatCompletionResponseChoice`；
- tool call 时 finish reason 为 `tool_calls`；
- 否则使用 `output.finish_reason`；
- 如果 finish reason 为空，兜底 `stop`；
- 透传 stop reason、token ids、routed experts。

## 6. Chat reasoning 输出

Chat serving 会在 OpenAI response 层处理 reasoning/content 拆分。

Reasoning parser 抽象：`code/vllm/vllm/reasoning/abs_reasoning_parsers.py:26`

核心方法：

- `extract_reasoning()`：`code/vllm/vllm/reasoning/abs_reasoning_parsers.py:146`
- `extract_reasoning_streaming()`：`code/vllm/vllm/reasoning/abs_reasoning_parsers.py:167`

Chat 请求生成前还会向 engine 传 reasoning 状态：

- 计算 `reasoning_ended`：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:346`
- 调用 generate：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:358`

这部分既影响 structured output 是否在 reasoning 阶段启用，也影响 OpenAI response 中 reasoning/content 的展示。

## 7. finish_reason 端到端来源

```text
Scheduler.check_stop()
  -> RequestStatus
  -> Request.get_finished_reason()
  -> EngineCoreOutput.finish_reason
  -> OutputProcessor
  -> CompletionOutput.finish_reason
  -> OpenAI response finish_reason
```

关键锚点：

- token stop 检查：`code/vllm/vllm/v1/core/sched/utils.py:94`
- RequestStatus 映射：`code/vllm/vllm/v1/request.py:353`
- `EngineCoreOutput.finish_reason`：`code/vllm/vllm/v1/engine/__init__.py:187`
- `CompletionOutput.finish_reason`：`code/vllm/vllm/outputs.py:21`

特殊情况：

- stop string 在 frontend 检测，可能由 `OutputProcessor` 修改 finish reason；
- tool call 在 Chat serving 层可能把 finish reason 改为 `tool_calls`；
- non-streaming chat 若 finish reason 为空可能兜底为 `stop`。

## 8. stop_reason 端到端来源

stop reason 可能是：

- stop token id；
- stop string；
- repetition_detected；
- None。

来源：

- stop token ids：`code/vllm/vllm/v1/core/sched/utils.py:108`
- repetition：`code/vllm/vllm/v1/core/sched/utils.py:119`
- stop string：`code/vllm/vllm/v1/engine/detokenizer.py:309`
- OutputProcessor 设置：`code/vllm/vllm/v1/engine/output_processor.py:639`

## 9. 调试地图：采样结果不符合预期

### 9.1 temperature/top_p/top_k/min_p 不生效

优先看：

1. 请求协议是否正确转成 `SamplingParams`：
   - completion：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:244`
   - chat：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:549`
2. `SamplingParams.__post_init__()` 是否把 greedy 模式强制改写 top_p/top_k/min_p：
   - `code/vllm/vllm/sampling_params.py:429`
3. GPU sampler 是否应用：
   - `code/vllm/vllm/v1/worker/gpu/sample/sampler.py:146`

### 9.2 seed / n > 1 行为不对

看：

- `SamplingParams.sampling_type`：`code/vllm/vllm/sampling_params.py:679`
- `ParentRequest._get_child_sampling_params()`：`code/vllm/vllm/v1/engine/parallel_sampling.py:52`
- GPU sampling state seeds：`code/vllm/vllm/v1/worker/gpu/sample/states.py:17`

重点：`n > 1` 且有 seed 时 child seed 会使用 `seed + index`。

### 9.3 allowed_token_ids / logit_bias / min_tokens 不对

看：

- `SamplingParams` 字段：`code/vllm/vllm/sampling_params.py:316`
- 参数校验：`code/vllm/vllm/sampling_params.py:717`
- GPU logit bias state：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:15`
- sampler 应用位置：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:159`

### 9.4 bad_words 不生效

看：

- tokenizer 更新：`code/vllm/vllm/v1/engine/input_processor.py:313`
- bad words state：`code/vllm/vllm/v1/worker/gpu/sample/bad_words.py:15`
- sampler 应用位置：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:173`

### 9.5 presence/frequency/repetition penalty 不对

看：

- `SamplingParams` penalty 字段：`code/vllm/vllm/sampling_params.py:224`
- GPU penalties state：`code/vllm/vllm/v1/worker/gpu/sample/penalties.py:14`
- 应用入口：`code/vllm/vllm/v1/worker/gpu/sample/penalties.py:81`
- sampler 调用：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:164`

## 10. 调试地图：输出不符合预期

### 10.1 token ids 已生成但前端没看到文本

看：

1. `ModelRunnerOutput.sampled_token_ids`：`code/vllm/vllm/v1/outputs.py:240`
2. `Scheduler.update_from_output()` 是否生成 `EngineCoreOutput`：`code/vllm/vllm/v1/core/sched/scheduler.py:1688`
3. `OutputProcessor.process_outputs()`：`code/vllm/vllm/v1/engine/output_processor.py:576`
4. `RequestState.make_request_output()` 是否因 FINAL_ONLY 或 stream_interval 返回 None：`code/vllm/vllm/v1/engine/output_processor.py:272`

### 10.2 streaming 不是增量或重复

看：

- `RequestOutputKind.DELTA`：`code/vllm/vllm/sampling_params.py:182`
- 协议转换 output_kind：
  - completion：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:337`
  - chat：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:637`
- `RequestState.make_request_output()`：`code/vllm/vllm/v1/engine/output_processor.py:272`
- detokenizer delta：`code/vllm/vllm/v1/engine/detokenizer.py:148`
- `RequestOutput.add()` 聚合：`code/vllm/vllm/outputs.py:145`

### 10.3 non-streaming 中间没有输出

这是正常行为。non-streaming 通常设置 `output_kind=FINAL_ONLY`。

检查：

- completion output kind：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:337`
- chat output kind：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:637`
- FINAL_ONLY 过滤：`code/vllm/vllm/v1/engine/output_processor.py:280`

### 10.4 stop string 没有按预期截断

看：

- `SamplingParams.stop`：`code/vllm/vllm/sampling_params.py:252`
- `include_stop_str_in_output`：`code/vllm/vllm/sampling_params.py:252`
- detokenizer stop string：`code/vllm/vllm/v1/engine/detokenizer.py:309`
- OutputProcessor 检测：`code/vllm/vllm/v1/engine/output_processor.py:639`

### 10.5 stop token / EOS 没有按预期停止

看：

- `SamplingParams.stop_token_ids` / `ignore_eos`：`code/vllm/vllm/sampling_params.py:252`
- generation config 注入：`code/vllm/vllm/v1/engine/input_processor.py:313`
- `check_stop()`：`code/vllm/vllm/v1/core/sched/utils.py:94`

### 10.6 logprobs 缺失或数量不对

看：

- `SamplingParams.logprobs`：`code/vllm/vllm/sampling_params.py:267`
- chat logprobs/top_logprobs 转换：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:629`
- GPU logprob 计算：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:101`
- scheduler slice：`code/vllm/vllm/v1/core/sched/scheduler.py:1669`
- frontend logprobs processor：`code/vllm/vllm/v1/engine/logprobs.py:29`

## 11. 调试地图：structured output 不符合预期

### 11.1 JSON/regex/choice 约束不生效

看：

- 请求转 `StructuredOutputsParams`：
  - completion：`code/vllm/vllm/entrypoints/openai/completion/protocol.py:281`
  - chat：`code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:581`
- `StructuredOutputsParams` 校验：`code/vllm/vllm/sampling_params.py:71`
- `StructuredOutputRequest.from_sampling_params()`：`code/vllm/vllm/v1/structured_output/request.py:31`
- grammar 初始化：`code/vllm/vllm/v1/structured_output/__init__.py:115`
- scheduler bitmask：`code/vllm/vllm/v1/core/sched/scheduler.py:1439`
- GPU mask：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:23`

### 11.2 reasoning 阶段 structured output 行为异常

看：

- reasoning parser：`code/vllm/vllm/reasoning/abs_reasoning_parsers.py:26`
- chat serving 传 reasoning 状态：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:346`
- `should_fill_bitmask()`：`code/vllm/vllm/v1/structured_output/__init__.py:305`
- `should_advance()`：`code/vllm/vllm/v1/structured_output/__init__.py:325`

### 11.3 spec decode + structured output 异常

看：

- spec decode logits row 展开：`code/vllm/vllm/v1/worker/gpu/model_runner.py:879`
- grammar bitmask speculative 处理：`code/vllm/vllm/v1/structured_output/__init__.py:275`
- rejection sampler：`code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:43`

## 12. 推荐阅读顺序

如果只想理解主线：

1. `code/vllm/vllm/sampling_params.py`
2. `code/vllm/vllm/v1/engine/input_processor.py`
3. `code/vllm/vllm/v1/worker/gpu/model_runner.py`
4. `code/vllm/vllm/v1/worker/gpu/sample/sampler.py`
5. `code/vllm/vllm/v1/core/sched/scheduler.py`
6. `code/vllm/vllm/v1/engine/output_processor.py`
7. `code/vllm/vllm/outputs.py`

如果专门看 OpenAI API：

1. `code/vllm/vllm/entrypoints/openai/completion/protocol.py`
2. `code/vllm/vllm/entrypoints/openai/chat_completion/protocol.py`
3. `code/vllm/vllm/entrypoints/openai/completion/serving.py`
4. `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py`

如果专门看 structured output：

1. `code/vllm/vllm/sampling_params.py`
2. `code/vllm/vllm/v1/structured_output/request.py`
3. `code/vllm/vllm/v1/structured_output/__init__.py`
4. `code/vllm/vllm/v1/core/sched/scheduler.py`
5. `code/vllm/vllm/v1/worker/gpu/structured_outputs.py`

## 13. 本篇小结

OpenAI serving 层主要是 response schema 组装和 OpenAI 兼容逻辑。真正采样在 GPU sampler，token/length stop 在 scheduler，stop string 和 detokenize 在 OutputProcessor，最终 OpenAI response 只是消费 `RequestOutput` 并做协议适配。
