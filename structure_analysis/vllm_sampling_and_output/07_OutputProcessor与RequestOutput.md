# 07 OutputProcessor 与 RequestOutput

本篇梳理 frontend 侧如何把 `EngineCoreOutputs` 转成用户可见的 `RequestOutput`，包括 detokenize、stop string、logprobs、流式/非流式聚合。

## 1. OutputProcessor 定位

文件：`code/vllm/vllm/v1/engine/output_processor.py`

类定义：`code/vllm/vllm/v1/engine/output_processor.py:417`

它位于 frontend/engine client 一侧，不执行模型、不采样，只负责把 engine core 输出变成上层 API 可消费的输出。

主入口：`process_outputs()`，`code/vllm/vllm/v1/engine/output_processor.py:576`

## 2. OutputProcessor 维护的对象

### 2.1 RequestOutputCollector

定义：`code/vllm/vllm/v1/engine/output_processor.py:45`

用于 AsyncLLM 每请求输出队列。

特点：

- `put()` 放入输出；
- `get()` 异步等待输出；
- DELTA 模式下如果 producer 快于 consumer，会合并多个 delta；
- 合并通过 `RequestOutput.add()` 完成。

### 2.2 StreamingUpdate

定义：`code/vllm/vllm/v1/engine/output_processor.py:116`

用于 streaming input 相关状态更新。

### 2.3 RequestState

定义：`code/vllm/vllm/v1/engine/output_processor.py:129`

每个 frontend 请求一个状态对象，保存：

- external/internal request id；
- parent request；
- prompt / prompt token ids / prompt embeds；
- `LogprobsProcessor`；
- `IncrementalDetokenizer`；
- output kind；
- stats；
- stream interval；
- routed experts chunks；
- streaming input 状态。

## 3. RequestOutput 与 CompletionOutput

文件：`code/vllm/vllm/outputs.py`

### 3.1 CompletionOutput

定义：`code/vllm/vllm/outputs.py:21`

单条 completion 输出，字段包括：

- `index`；
- `text`；
- `token_ids`；
- `cumulative_logprob`；
- `logprobs`；
- `routed_experts`；
- `finish_reason`；
- `stop_reason`；
- `lora_request`。

### 3.2 RequestOutput

定义：`code/vllm/vllm/outputs.py:85`

一个请求的最终输出，字段包括：

- `request_id`；
- `prompt`；
- `prompt_token_ids`；
- `prompt_logprobs`；
- `outputs: list[CompletionOutput]`；
- `finished`；
- `metrics`；
- `lora_request`；
- `encoder_prompt`；
- `encoder_prompt_token_ids`；
- `num_cached_tokens`；
- `kv_transfer_params`。

### 3.3 RequestOutput.add()

定义：`code/vllm/vllm/outputs.py:145`

用于把后续输出合并进已有输出。

在 DELTA 模式下，如果 output producer 比 consumer 快，collector 会把多个 delta 合并：

- 拼接 text；
- 拼接 token ids；
- 拼接 logprobs；
- 更新 finish reason / stop reason；
- 更新 metrics。

## 4. process_outputs() 主流程

入口：`code/vllm/vllm/v1/engine/output_processor.py:576`

每个 `EngineCoreOutput` 的处理顺序：

### 4.1 找到 RequestState

位置：`code/vllm/vllm/v1/engine/output_processor.py:607`

如果请求已经 abort，输出可能被忽略。

### 4.2 更新 stats

位置：`code/vllm/vllm/v1/engine/output_processor.py:613`

统计信息包括 prefill/decode 时间、cached tokens 等。

### 4.3 读取核心字段

位置：`code/vllm/vllm/v1/engine/output_processor.py:618`

读取：

- `new_token_ids`；
- `pooling_output`；
- `finish_reason`；
- `stop_reason`。

### 4.4 routed experts chunks

位置：`code/vllm/vllm/v1/engine/output_processor.py:623`

保存本次输出的 routed experts，最终拼到 completion output。

### 4.5 cached token 统计

位置：`code/vllm/vllm/v1/engine/output_processor.py:628`

第一次 prefill 后记录 `num_cached_tokens`。

### 4.6 detokenize

位置：`code/vllm/vllm/v1/engine/output_processor.py:635`

对非 pooling 输出，调用 `IncrementalDetokenizer.update()`。

### 4.7 stop string 检测

位置：`code/vllm/vllm/v1/engine/output_processor.py:639`

如果 frontend 检测到 stop string：

- `finish_reason = FinishReason.STOP`；
- `stop_reason = stop_string`；
- 如果 EngineCore 尚未 finished，需要通知 engine core abort 该请求。

### 4.8 logprobs 更新

位置：`code/vllm/vllm/v1/engine/output_processor.py:648`

调用 `LogprobsProcessor.update_from_output()`。

### 4.9 构造 RequestOutput

位置：`code/vllm/vllm/v1/engine/output_processor.py:651`

调用 `RequestState.make_request_output()`。

### 4.10 输出分发

位置：`code/vllm/vllm/v1/engine/output_processor.py:661`

- AsyncLLM：放入 request queue；
- LLMEngine：加入返回 list。

### 4.11 完成清理

位置：

- `code/vllm/vllm/v1/engine/output_processor.py:668`
- `code/vllm/vllm/v1/engine/output_processor.py:678`

如果请求完成，清理 request state；如果是 frontend stop string 触发完成，还需要 abort engine core 请求。

## 5. Output kind：DELTA / FINAL_ONLY / CUMULATIVE

定义：`code/vllm/vllm/sampling_params.py:182`

- `CUMULATIVE`：每次返回完整输出；
- `DELTA`：每次只返回增量；
- `FINAL_ONLY`：只返回最终输出。

OpenAI serving 中通常：

- streaming -> `DELTA`；
- non-streaming -> `FINAL_ONLY`。

## 6. RequestState.make_request_output()

入口：`code/vllm/vllm/v1/engine/output_processor.py:272`

关键逻辑：

1. 如果未完成且 `FINAL_ONLY`，返回 None；
2. 如果设置 `stream_interval > 1`，只在特定条件输出；
3. DELTA 模式只返回新 token 和新文本；
4. 非 DELTA 模式返回累计 token/text；
5. 构造 `CompletionOutput` 与 `RequestOutput`。

锚点：

- FINAL_ONLY 过滤：`code/vllm/vllm/v1/engine/output_processor.py:280`
- stream interval：`code/vllm/vllm/v1/engine/output_processor.py:287`
- DELTA token ids：`code/vllm/vllm/v1/engine/output_processor.py:302`

## 7. _new_completion_output()

入口：`code/vllm/vllm/v1/engine/output_processor.py:376`

关键逻辑：

- `delta = output_kind == DELTA`；
- `text = detokenizer.get_next_output_text(finished, delta)`；
- 非 DELTA 模式 token ids 使用完整 `detokenizer.output_token_ids`；
- DELTA 模式 logprobs 只截取最后 `len(token_ids)` 个；
- finished 时拼接 routed experts chunks。

锚点：`code/vllm/vllm/v1/engine/output_processor.py:387`

## 8. IncrementalDetokenizer

文件：`code/vllm/vllm/v1/engine/detokenizer.py`

`get_next_output_text()`：`code/vllm/vllm/v1/engine/detokenizer.py:148`

行为：

- `delta=False`：返回完整 output text；
- `delta=True`：只返回自上次输出后的新文本；
- 如果未 finished 且有 stop string，可能保留 stop buffer，避免提前发送可能属于 stop string 的后缀。

## 9. stop string 检测

入口：`code/vllm/vllm/v1/engine/detokenizer.py:309`

`check_stop_strings()` 会在 detokenized text 上查找 stop string，并按 `include_stop_str_in_output` 决定：

- 截断到 stop string 前；
- 或截断到 stop string 后。

为什么放在 frontend：字符串 stop 需要文本，不是 token id 层面能完全准确处理的。

## 10. FinishReason

定义：`code/vllm/vllm/v1/engine/__init__.py:42`

取值：

- `STOP` -> `stop`；
- `LENGTH` -> `length`；
- `ABORT` -> `abort`；
- `ERROR` -> `error`；
- `REPETITION` -> `repetition`。

位置：`code/vllm/vllm/v1/engine/__init__.py:28`

## 11. LogprobsProcessor 在 frontend 的作用

文件：`code/vllm/vllm/v1/engine/logprobs.py`

类：`LogprobsProcessor`，`code/vllm/vllm/v1/engine/logprobs.py:29`

更新入口：`code/vllm/vllm/v1/engine/logprobs.py:348`

职责：

- 把 scheduler 传来的 `new_logprobs` 变成 `CompletionOutput.logprobs`；
- 累加 `cumulative_logprob`；
- 处理 prompt logprobs；
- 在 DELTA 模式下只返回新增部分；
- 对 token 文本做必要 detokenize。

## 12. PoolingRequestOutput

如果是 embedding/pooling/classification/scoring 类请求，`EngineCoreOutput.pooling_output` 会被 `OutputProcessor` 包装成 pooling 输出。

这类输出不走普通 completion text/token_ids 路径。

相关处理入口仍在 `process_outputs()`：`code/vllm/vllm/v1/engine/output_processor.py:576`

## 13. AsyncLLM 输出队列

`RequestOutputCollector` 负责 AsyncLLM 请求级队列。

AsyncLLM 后台 output handler 从 engine core client 取 `EngineCoreOutputs`，调用 `OutputProcessor.process_outputs()`，然后把每个请求输出放入对应 collector。

相关 AsyncLLM 输出 handler：`code/vllm/vllm/v1/engine/async_llm.py:656`

## 14. LLMEngine 同步返回

同步 `LLMEngine.step()` 会调用 output processor，并直接返回 `RequestOutput` list。

锚点：`code/vllm/vllm/v1/engine/llm_engine.py:296`

## 15. ParentRequest 输出聚合

当 `n > 1` 时，child request 的 `RequestOutput` 还要经过 parent request 聚合。

相关文件：`code/vllm/vllm/v1/engine/parallel_sampling.py`

聚合入口：`code/vllm/vllm/v1/engine/parallel_sampling.py:100`

行为：

- FINAL_ONLY：等所有 child 完成后输出；
- DELTA：child 增量可以转为 parent 增量；
- child index 对应最终 `CompletionOutput.index`。

## 16. 本篇小结

`OutputProcessor` 是“内部输出到用户输出”的关键转换层。它负责：

- detokenize；
- stop string 检测；
- logprobs 累积和格式化；
- DELTA / FINAL_ONLY / CUMULATIVE 输出模式；
- AsyncLLM 队列；
- RequestOutput/CompletionOutput 构造；
- 完成后清理与 abort 通知。

如果看到 `ModelRunnerOutput` 里已经有 token ids，不代表用户马上会看到文本；它必须经过 scheduler 和 output processor。
