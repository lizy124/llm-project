# 09. OutputProcessor 如何生成 RequestOutput？

源码位置：

- `code/vllm/vllm/v1/sample/metadata.py`
- `code/vllm/vllm/v1/sample/sampler.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/engine/__init__.py`
- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/v1/engine/logprobs.py`
- `code/vllm/vllm/v1/engine/detokenizer.py`
- `code/vllm/vllm/v1/engine/parallel_sampling.py`
- `code/vllm/vllm/v1/engine/llm_engine.py`
- `code/vllm/vllm/v1/engine/async_llm.py`
- `code/vllm/vllm/outputs.py`
- `code/vllm/vllm/logprobs.py`

本问题关注：worker / scheduler 生成的 token、logprobs、pooling output、finish reason，如何经过 `OutputProcessor`、`RequestState`、detokenizer 和 logprobs processor，最终变成 API 层可见的 `RequestOutput`、`CompletionOutput`、`PoolingRequestOutput`、`EmbeddingRequestOutput`。

---

## 1. 一句话回答

`OutputProcessor` 是 vLLM v1 中 **EngineCore 内部输出到用户可见输出对象的转换层**。

核心链路是：

```text
SamplerOutput
  → ModelRunnerOutput
  → EngineCoreOutput
  → OutputProcessor.process_outputs()
  → RequestState
  → IncrementalDetokenizer
  → LogprobsProcessor
  → CompletionOutput / PoolingOutput
  → RequestOutput / PoolingRequestOutput
```

如果只记一句话：

```text
Sampler 决定“下一个 token 是什么”，OutputProcessor 决定“这个 token 怎么变成用户看到的 text / logprobs / finish reason”。
```

---

## 2. RequestOutput 和内部输出的区别

内部输出更关注执行和调度：

```text
req_id
new_token_ids
new_logprobs
prompt_logprobs tensors
pooling tensor
finish_reason
stop_reason
KV connector params
routed experts
prefill stats
scheduler stats
```

用户输出更关注 API 语义：

```text
request_id
prompt / prompt_token_ids
prompt_logprobs
outputs[]
  → text
  → token_ids
  → cumulative_logprob
  → logprobs
  → finish_reason
  → stop_reason
finished
metrics
num_cached_tokens
kv_transfer_params
```

所以二者不是同一个层次：

```text
EngineCoreOutput 是“本轮 engine core 对某个请求产生了什么”；
RequestOutput 是“用户这次应该看到什么”。
```

---

## 3. 总体主链路

从采样到 API 输出的完整链路：

```text
SamplingMetadata
  → Sampler.forward()
      → SamplerOutput(sampled_token_ids, logprobs_tensors)

GPUModelRunner
  → _bookkeeping_sync()
      → valid_sampled_token_ids
      → LogprobsLists
      → prompt_logprobs_dict
  → ModelRunnerOutput

Scheduler / EngineCore
  → update_from_output()
  → EngineCoreOutput
      → new_token_ids
      → new_logprobs
      → new_prompt_logprobs_tensors
      → pooling_output
      → finish_reason
      → stop_reason

Frontend engine
  → OutputProcessor.process_outputs()
      → detokenizer.update()
      → logprobs_processor.update_from_output()
      → RequestState.make_request_output()

最终输出
  → CompletionOutput
  → RequestOutput
  或
  → PoolingOutput
  → PoolingRequestOutput
```

这条链路里，`OutputProcessor` 位于 EngineCore 与前端 API 之间。

---

## 4. 几个输出对象的层次

### 4.1 SamplerOutput

`SamplerOutput` 是 sampler 的直接输出：

```text
sampled_token_ids: Tensor[num_reqs, max_num_generated_tokens]
logprobs_tensors: LogprobsTensors | None
```

它仍是 worker / GPU 侧结构，偏 tensor。

### 4.2 ModelRunnerOutput

`ModelRunnerOutput` 是 worker 发回 scheduler / engine core 的输出：

```text
req_ids
req_id_to_index
sampled_token_ids: list[list[int]]
logprobs: LogprobsLists | None
prompt_logprobs_dict
pooler_output
kv_connector_output
ec_connector_output
routed_experts
num_nans_in_logits
```

它已经把 GPU tensor 输出尽量转成 CPU / list / numpy 形式，方便跨进程传输。

### 4.3 EngineCoreOutput

`EngineCoreOutput` 是 engine core 协议层的单请求输出：

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

它是 `OutputProcessor` 的输入。

### 4.4 CompletionOutput

`CompletionOutput` 是一个 completion 分支的输出：

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

如果 `n > 1`，一个 `RequestOutput.outputs` 中可以有多个 `CompletionOutput`。

### 4.5 RequestOutput

`RequestOutput` 是普通生成请求最终返回给用户的对象：

```text
request_id
prompt
prompt_token_ids
prompt_logprobs
outputs: list[CompletionOutput]
finished
metrics
lora_request
encoder_prompt / encoder_prompt_token_ids
num_cached_tokens
kv_transfer_params
```

### 4.6 PoolingRequestOutput / EmbeddingRequestOutput

Pooling 类请求不生成 token 文本，返回的是 hidden states / embedding / score 等：

```text
PoolingRequestOutput
  → request_id
  → outputs: PoolingOutput
  → prompt_token_ids
  → num_cached_tokens
  → finished
```

`EmbeddingRequestOutput`、`ClassificationRequestOutput`、`ScoringRequestOutput` 都是从 `PoolingRequestOutput` 转换出来的类型化视图。

---

## 5. SamplerOutput 是怎么来的

`Sampler.forward()` 的输入是：

```text
logits
SamplingMetadata
```

`SamplingMetadata` 中会携带：

```text
temperature
top_p
top_k
generators
max_num_logprobs
prompt_token_ids
output_token_ids
allowed_token_ids_mask
bad_words_token_ids
logitsprocs
logprob_token_ids
spec_token_ids
```

Sampler 大致做：

```text
1. 如需要 logprobs，先保存 raw logprobs / raw logits；
2. logits 转 float32；
3. apply_logits_processors()；
4. sample() 得到 sampled token；
5. gather top-k logprobs 或指定 token logprobs；
6. 返回 SamplerOutput。
```

这里输出的 token 仍只是 token id，不包含文本。

---

## 6. GPUModelRunner 如何把 SamplerOutput 变成 ModelRunnerOutput

`GPUModelRunner._sample()` 调用 sampler：

```text
self.sampler(logits, sampling_metadata)
```

如果有 speculative decoding，则会走 rejection sampler，把 draft token 和 target logits 结合处理。

随后 `_bookkeeping_sync()` 会把 `SamplerOutput` 转成更适合 scheduler 的结构：

```text
sampled_token_ids tensor
  → valid_sampled_token_ids: list[list[int]]

logprobs_tensors
  → LogprobsLists

prompt logprobs tensors
  → prompt_logprobs_dict[req_id]
```

最后 `sample_tokens()` 组装：

```text
ModelRunnerOutput(
  req_ids=...,
  req_id_to_index=...,
  sampled_token_ids=...,
  logprobs=...,
  prompt_logprobs_dict=...,
  pooler_output=...,
  kv_connector_output=...,
  routed_experts=...,
)
```

这一步仍然不是用户输出，而是 worker → engine core 的内部结果。

---

## 7. EngineCoreOutput 是 OutputProcessor 的直接输入

`EngineCoreOutput` 是单个请求在一次 engine step 中的输出。

关键字段：

```text
request_id：内部请求 id
new_token_ids：本轮新生成 token ids
new_logprobs：本轮采样 token 的 logprobs
new_prompt_logprobs_tensors：prompt logprobs
pooling_output：pooling 模型输出 tensor
finish_reason：请求是否结束以及结束原因
stop_reason：具体 stop token id 或 stop string
kv_transfer_params：KV transfer 返回给用户的参数
prefill_stats：prefill 统计，如 num_cached_tokens
routed_experts：MoE routed experts 输出
num_nans_in_logits：logits NaN 检测
```

`OutputProcessor.process_outputs()` 只消费 `EngineCoreOutput`，不直接碰 sampler 或 model runner。

---

## 8. OutputProcessor 的核心职责

`OutputProcessor` 的职责可以分成八类：

```text
1. 为新请求创建 RequestState；
2. 根据 request id 找到对应 RequestState；
3. 更新 stats / num_cached_tokens；
4. detokenize 新 token ids；
5. 检查 stop strings，并在必要时修正 finish_reason / stop_reason；
6. 更新 sample logprobs / prompt logprobs；
7. 根据 output_kind / stream_interval 生成 RequestOutput；
8. 请求结束时清理状态，必要时通知 EngineCore abort。
```

它不负责：

```text
- 采样 token；
- 决定调度；
- 分配 KV cache；
- 执行模型 forward。
```

它负责的是：

```text
把内部 token 级输出转成 API 级输出。
```

---

## 9. RequestState 是输出处理的状态机

每个请求在前端 output processor 里都有一个 `RequestState`。

它保存：

```text
request_id：内部请求 id
external_req_id：用户提供的 request id
parent_req：parallel sampling 的父请求
request_index：n > 1 时当前 child 的 index
lora_request
output_kind
prompt / prompt_token_ids / prompt_embeds
logprobs_processor
detokenizer
max_tokens_param
stats
stream_interval
sent_tokens_offset
streaming_input / input_chunk_queue
routed_experts_chunks
num_cached_tokens
```

可以把它理解成：

```text
EngineCore 只给“这一轮新增了什么”；
RequestState 记住“到目前为止已经输出了什么、还应该怎么输出”。
```

---

## 10. 新请求如何进入 OutputProcessor

前端添加请求时会调用：

```text
OutputProcessor.add_request(...)
```

如果是新 request，会调用：

```text
RequestState.from_new_request(...)
```

对生成请求：

```text
sampling_params 存在
  → 根据 sampling_params.detokenize 决定是否使用 tokenizer
  → 创建 LogprobsProcessor
  → 创建 IncrementalDetokenizer
  → output_kind = sampling_params.output_kind
  → max_tokens / top_p / n / temperature 等用于 stats / tracing
```

对 pooling 请求：

```text
pooling_params 存在
  → 不创建 detokenizer
  → 不创建 logprobs_processor
  → output_kind = pooling_params.output_kind
```

这决定了后续走 completion 分支还是 pooling 分支。

---

## 11. process_outputs 的主循环

`OutputProcessor.process_outputs()` 是输出转换的核心循环。

它对每个 `EngineCoreOutput` 做：

```text
1. 根据 request_id 找 RequestState；
2. 更新 iteration stats；
3. 读取 new_token_ids / pooling_output / finish_reason / stop_reason；
4. 记录 routed_experts；
5. 如果是 prefill 后的第一轮，记录 num_cached_tokens；
6. completion 分支：
     detokenizer.update(new_token_ids, stop_terminated)
     如果发现 stop string，设置 finish_reason=STOP、stop_reason=stop_string
     logprobs_processor.update_from_output(engine_core_output)
7. RequestState.make_request_output(...)
8. 如果 AsyncLLM，有 queue 则 put 到 queue；
   如果 LLMEngine，无 queue 则 append 到返回列表；
9. 如果请求完成，清理 RequestState；
10. 如果 stop string 是 detokenizer 发现的但 EngineCore 还没结束，加入 reqs_to_abort。
```

最后返回：

```text
OutputProcessorOutput(
  request_outputs=list[RequestOutput | PoolingRequestOutput],
  reqs_to_abort=list[str],
)
```

---

## 12. detokenizer 如何生成 text

生成请求会创建 `IncrementalDetokenizer`。

如果：

```text
tokenizer is None 或 sampling_params.detokenize=False
```

则使用基础 `IncrementalDetokenizer`，只保存 token ids，不生成文本。

否则：

```text
fast tokenizer + tokenizers >= 0.22.0
  → FastIncrementalDetokenizer

其他情况
  → SlowIncrementalDetokenizer
```

detokenizer 每轮做：

```text
1. 接收 new_token_ids；
2. 增量 decode；
3. 追加到 output_text；
4. 检查 stop strings；
5. 必要时截断 output_text；
6. 在 get_next_output_text() 中根据 delta / final 输出文本。
```

---

## 13. include_stop_str_in_output 如何生效

`BaseIncrementalDetokenizer` 初始化时读取：

```text
sampling_params.stop
sampling_params.min_tokens
sampling_params.include_stop_str_in_output
```

如果有 stop string 且不包含 stop string：

```text
stop_buffer_length = max_len(stop_strings) - 1
```

流式输出时会暂存末尾字符，避免把可能组成 stop string 的字符提前发给用户。

当 `check_stop_strings()` 命中 stop string：

```text
include_stop_str_in_output=True：
  输出文本截断到 stop string 末尾，stop string 保留。

include_stop_str_in_output=False：
  输出文本截断到 stop string 起点，stop string 不保留。
```

此外，如果 EngineCore 因 stop token 结束：

```text
stop_terminated=True
```

且 `include_stop_str_in_output=False`，最后一个 stop token 不参与 detokenization，但仍会被加入 token id 列表。

---

## 14. stop string 为什么可能触发 abort

stop string 的检测发生在前端 detokenizer 中。

也就是说：

```text
EngineCore 可能还不知道请求已经应该因为 stop string 停止；
OutputProcessor 在 detokenize 后才发现 stop string。
```

因此 `process_outputs()` 中有这个逻辑：

```text
如果 finish_reason is not None：
  清理请求
  如果 engine_core_output.finished 为 False：
    reqs_to_abort.append(req_id)
```

含义：

```text
前端已经决定输出结束；
但 EngineCore 侧还没停止该请求；
需要再通知 EngineCore abort。
```

这就是为什么 `OutputProcessorOutput` 不只返回 `request_outputs`，还返回 `reqs_to_abort`。

---

## 15. LogprobsProcessor 做什么

`LogprobsProcessor` 负责把 EngineCore 的 logprobs tensors/lists 转成 API 侧的 logprobs 结构。

它维护：

```text
logprobs: SampleLogprobs | None
prompt_logprobs: PromptLogprobs | None
cumulative_logprob: float | None
num_logprobs
num_prompt_logprobs
```

初始化时根据 sampling params：

```text
num_logprobs = sampling_params.num_logprobs
num_prompt_logprobs = sampling_params.prompt_logprobs
```

如果用户没请求 logprobs，对应字段就是 None。

---

## 16. sample logprobs 如何更新

当 `EngineCoreOutput.new_logprobs` 不为空，调用：

```text
_update_sample_logprobs(logprobs_lists)
```

它会对每个生成位置：

```text
1. 取出 token_ids / logprobs / ranks；
2. 如果有 tokenizer，非增量地把候选 token ids 转成 token 字符串；
3. 修正 byte fallback / UTF-8 replacement char；
4. 把 sampled token 的 logprob 累加到 cumulative_logprob；
5. append_logprobs_for_next_position() 写入 API logprobs 容器。
```

注意：

```text
sampler 会把 sampled token 的 logprob 放在第一位。
```

所以 cumulative logprob 用的是 `logprobs[0]`。

---

## 17. prompt logprobs 如何更新

当 `EngineCoreOutput.new_prompt_logprobs_tensors` 不为空，调用：

```text
_update_prompt_logprobs(prompt_logprobs_tensors)
```

它会：

```text
1. 从 tensor 中取 token_ids / logprobs / ranks；
2. 展平成 token ids 做非增量 detokenize；
3. 按 prompt position 重建 Logprob 容器；
4. 修正 UTF-8 / byte fallback；
5. append 到 self.prompt_logprobs。
```

prompt logprobs 和 sample logprobs 是两条独立链路：

```text
prompt_logprobs → RequestOutput.prompt_logprobs
sample logprobs → CompletionOutput.logprobs
```

---

## 18. DELTA 模式下 logprobs 怎么办

如果 `output_kind == RequestOutputKind.DELTA`：

```text
CompletionOutput.text 只返回新增文本；
CompletionOutput.token_ids 只返回新增 token ids；
CompletionOutput.logprobs 只返回新增 token 对应的 logprobs；
RequestOutput.prompt_logprobs 通过 pop_prompt_logprobs() 一次性返回后清空。
```

为什么 prompt logprobs 要 pop？

```text
prompt logprobs 可能跨多个 prefill chunk 聚合；
DELTA 语义下不能每次重复返回全部 prompt logprobs；
因此在合适时机一次性吐出并清空。
```

---

## 19. RequestState.make_request_output 的分支

`make_request_output()` 是从状态生成输出对象的入口。

主要逻辑：

```text
finished = finish_reason is not None
final_only = output_kind == FINAL_ONLY

如果未 finished 且 FINAL_ONLY：
  return None

如果 stream_interval > 1：
  未达到输出间隔则 return None
  DELTA 模式下只取 sent_tokens_offset 之后的新 token

如果 pooling_output is not None：
  返回 PoolingRequestOutput

否则：
  构造 CompletionOutput
  如果没有 parent_req：
    outputs = [completion]
  如果有 parent_req：
    交给 ParentRequest.get_outputs() 聚合
  最后构造 RequestOutput
```

这说明：

```text
OutputProcessor 不是每个 engine step 都一定返回用户输出；
它会受 output_kind、stream_interval、parallel sampling 聚合状态影响。
```

---

## 20. CompletionOutput 如何组装

`_new_completion_output()` 会构造：

```text
CompletionOutput(
  index=request_index,
  text=...,
  token_ids=...,
  routed_experts=...,
  logprobs=...,
  cumulative_logprob=...,
  finish_reason=str(finish_reason) if finished else None,
  stop_reason=stop_reason if finished else None,
)
```

关键细节：

```text
text 来自 detokenizer.get_next_output_text(finished, delta)

非 DELTA：
  token_ids = detokenizer.output_token_ids

DELTA：
  token_ids = 当前新增 token_ids
  logprobs = logprobs[-len(token_ids):]

finished 且 routed_experts_chunks 不为空：
  routed_experts = concatenate(chunks)
```

因此 `CompletionOutput` 是：

```text
单条 completion 的文本、token、logprobs、结束信息。
```

---

## 21. RequestOutput 如何组装

`_new_request_output()` 会处理 prompt 和 request 级字段。

对 completion 请求：

```text
RequestOutput(
  request_id=external_req_id,
  prompt=prompt,
  prompt_token_ids=prompt_token_ids,
  prompt_logprobs=prompt_logprobs,
  outputs=list[CompletionOutput],
  finished=finished,
  metrics=stats,
  lora_request=lora_request,
  num_cached_tokens=num_cached_tokens,
  kv_transfer_params=kv_transfer_params,
)
```

注意：

```text
request_id 使用 external_req_id，
也就是用户传入的 request id，
不是 EngineCore 内部 child request id。
```

如果请求使用 `prompt_embeds` 而没有 prompt token ids：

```text
prompt_token_ids = [0] * len(prompt_embeds)
```

这是为了 API 输出结构仍然有 prompt_token_ids 字段。

---

## 22. PoolingRequestOutput 如何组装

如果 `EngineCoreOutput.pooling_output is not None`，走 pooling 分支。

输出对象是：

```text
PoolingRequestOutput(
  request_id=external_req_id,
  outputs=PoolingOutput(data=pooling_output),
  prompt_token_ids=prompt_token_ids,
  num_cached_tokens=num_cached_tokens,
  finished=finished,
)
```

这类请求没有：

```text
detokenizer
logprobs_processor
CompletionOutput
finish text
```

上层如果需要 embedding / classification / scoring，会再从 `PoolingRequestOutput` 转：

```text
EmbeddingRequestOutput.from_base()
ClassificationRequestOutput.from_base()
ScoringRequestOutput.from_base()
```

---

## 23. FinishReason 和 stop_reason 的区别

`FinishReason` 是枚举：

```text
STOP
LENGTH
ABORT
ERROR
REPETITION
```

它表示结束类别。

`stop_reason` 表示更具体的停止原因：

```text
stop token id
stop string
None
```

典型情况：

| 场景 | finish_reason | stop_reason |
|---|---|---|
| 命中 EOS / stop token | STOP | token id 或 None |
| 命中 stop string | STOP | stop string |
| 达到 max_tokens / max_model_len | LENGTH | None |
| 用户 abort | ABORT | None |
| 内部可重试错误 | ERROR | None |
| 重复模式检测 | REPETITION | None |

`CompletionOutput.finish_reason` 最终是字符串：

```text
str(finish_reason)
```

例如：

```text
"stop"
"length"
"abort"
"error"
"repetition"
```

---

## 24. RequestOutputKind 对输出有什么影响

生成请求的输出模式由：

```text
sampling_params.output_kind
```

决定。

典型语义：

```text
FINAL_ONLY：
  只在请求完成时返回最终完整输出。

DELTA：
  流式返回增量 text / token ids / logprobs。

其他非 delta 模式：
  可以返回当前累积完整输出。
```

`OutputProcessor` 中直接体现为：

```text
未 finished 且 FINAL_ONLY：return None
DELTA：只取新增 token 和新增文本
```

---

## 25. stream_interval 如何节流输出

`RequestState.stream_interval` 控制每隔多少 token 才输出一次。

如果 `stream_interval > 1`，只有满足以下条件才输出：

```text
1. 请求 finished；
2. 或这是第一个 token；
3. 或从上次发送后新增 token 数达到 stream_interval。
```

DELTA 模式下，还会更新：

```text
sent_tokens_offset
```

保证下次只返回后续新增 token。

---

## 26. n > 1 / parallel sampling 如何聚合

如果 `sampling_params.n > 1`，vLLM 会创建 `ParentRequest`。

Parent request 会为每个 child 生成：

```text
child_request_id = f"{index}_{parent_request_id}"
child_sampling_params.n = 1
如果 seed 不为空，每个 child seed = seed + index
```

每个 child 都有自己的 `RequestState`，但共享一个 `ParentRequest`。

当 child 输出 `CompletionOutput` 时：

```text
ParentRequest.get_outputs(child_request_id, completion_output)
```

会根据输出模式决定：

```text
非 FINAL_ONLY：
  streaming 时直接返回当前 child output。

FINAL_ONLY：
  聚合到 output_aggregator[index]；
  直到所有 child 完成才返回完整 outputs。
```

所以用户看到的仍是一个 external request id，对应多个 completion outputs。

---

## 27. RequestOutputCollector 在 AsyncLLM 中做什么

`AsyncLLM` 不能直接从 `process_outputs()` 返回列表，因为每个请求是独立异步 generator。

因此每个请求有一个：

```text
RequestOutputCollector
```

它负责：

```text
OutputProcessor 后台 put(output)
AsyncLLM.generate() 前台 await get()
```

如果 output_kind 是 DELTA：

```text
collector.aggregate = True
```

当生产速度超过消费速度时，collector 会把多个 `RequestOutput` 合并：

```text
RequestOutput.add(next_output, aggregate=True)
```

这样可以减少队列堆积，并保持 delta 输出语义。

---

## 28. LLMEngine.step 的同步输出路径

同步 `LLMEngine.step()` 做：

```text
1. outputs = engine_core.get_output()
2. processed_outputs = output_processor.process_outputs(outputs.outputs, ...)
3. engine_core.abort_requests(processed_outputs.reqs_to_abort)
4. 记录 stats
5. return processed_outputs.request_outputs
```

这条路径中：

```text
RequestState.queue is None
```

所以 `process_outputs()` 生成的 `RequestOutput` 会直接 append 到返回列表。

---

## 29. AsyncLLM.generate 的异步输出路径

`AsyncLLM.generate()` 做：

```text
1. add_request() 创建 RequestOutputCollector；
2. 后台 output_handler 不断从 EngineCore 拉 outputs；
3. output_handler 调 output_processor.process_outputs()；
4. 因为 RequestState.queue 不为空，RequestOutput 被 put 到 queue；
5. generate() 前台循环 q.get_nowait() or await q.get()；
6. yield RequestOutput；
7. out.finished=True 时结束。
```

后台 `_run_output_handler()` 还会按 chunk 处理 output：

```text
VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
```

避免一次处理太多 outputs 阻塞 event loop。

---

## 30. abort 请求如何产生最终输出

`OutputProcessor.abort_requests()` 会处理 external 或 internal request id。

如果找到对应 `RequestState`，会：

```text
1. 从 request_states 中移除；
2. 标记 LoRA request finished；
3. 生成一个 finish_reason=ABORT 的最终输出；
4. 如果是 AsyncLLM，把 abort output put 到 queue。
```

completion 请求：

```text
pooling_output = None
finish_reason = ABORT
```

pooling 请求：

```text
pooling_output = EMPTY_CPU_TENSOR
finish_reason = ABORT
```

这样前端 generator 可以收到 finished=True 的结束信号。

---

## 31. metrics / num_cached_tokens 从哪里来

`RequestState.stats` 在创建请求时初始化：

```text
RequestStateStats(arrival_time=arrival_time)
```

每轮输出时：

```text
_update_stats_from_output(...)
```

会根据 `EngineCoreOutput` 和 engine core timestamp 更新：

```text
first token latency
last token timestamp
prefill / decode timing
num_generation_tokens
queue / schedule 事件
```

`num_cached_tokens` 来自：

```text
engine_core_output.prefill_stats.num_cached_tokens
```

只在 request 还处于 prefill 状态时记录一次。

最终它们进入：

```text
RequestOutput.metrics
RequestOutput.num_cached_tokens
PoolingRequestOutput.num_cached_tokens
```

---

## 32. kv_transfer_params 如何进入用户输出

`EngineCoreOutput` 可能携带：

```text
kv_transfer_params
```

`process_outputs()` 读取后传给：

```text
RequestState.make_request_output(..., kv_transfer_params)
```

最终进入：

```text
RequestOutput.kv_transfer_params
```

这用于 P/D、remote KV transfer 等场景，让 connector 产生的参数能跟随请求输出返回给调用方。

---

## 33. routed_experts 如何进入 CompletionOutput

如果启用了 routed experts 返回，`EngineCoreOutput.routed_experts` 会在每轮被追加到：

```text
req_state.routed_experts_chunks
```

只有在请求完成时：

```text
finished and routed_experts_chunks
```

才会：

```text
np.concatenate(routed_experts_chunks, axis=0)
```

然后放进：

```text
CompletionOutput.routed_experts
```

因此 routed experts 不是每个 streaming chunk 都完整返回，而是在 finish 时汇总。

---

## 34. streaming input 的特殊处理

`RequestState` 还支持 streaming input / resumable request。

如果同一个 request id 再次 `add_request()`，已有 `RequestState` 会走：

```text
_update_streaming_request_state()
```

而不是创建新状态。

当当前 chunk 完成：

```text
如果 input_chunk_queue 里还有 update：
  apply_streaming_update(update)
  继续下一段输入

否则：
  input_chunk_queue = None
```

如果最终输入流结束，会发送：

```text
STREAM_FINISHED
```

用于解除 `generate()` 等待。

---

## 35. logprobs 数据结构的层次

worker / engine 内部：

```text
LogprobsTensors
  → torch.Tensor
  → 适合 GPU / CPU tensor 传输

LogprobsLists
  → numpy.ndarray
  → 适合 ModelRunnerOutput / EngineCoreOutput
```

API 输出：

```text
SampleLogprobs
PromptLogprobs
FlatLogprobs
Logprob objects
```

转换发生在：

```text
LogprobsProcessor._update_sample_logprobs()
LogprobsProcessor._update_prompt_logprobs()
```

这一步会把 token id、logprob、rank、decoded token 文本整合成用户能理解的 logprobs。

---

## 36. prompt_embeds 请求的输出处理

如果请求使用 `prompt_embeds`，可能没有原始 `prompt_token_ids`。

但 `RequestOutput` / `PoolingRequestOutput` 需要 prompt token ids 字段。

因此 `_new_request_output()` 中有特殊处理：

```text
if prompt_token_ids is None and prompt_embeds is not None:
  prompt_token_ids = [0] * len(prompt_embeds)
```

这不是说真实 token id 是 0，而是输出结构的占位。

---

## 37. 最容易混淆的点

### 37.1 OutputProcessor 会决定采样结果吗？

不会。

采样结果在 sampler / model runner / scheduler 阶段已经确定。OutputProcessor 只负责转换和呈现。

### 37.2 stop string 是 Scheduler 判断的吗？

不完全是。

stop token / EOS / length 等可以由 EngineCore / Scheduler 标记；但 stop string 需要 detokenized text，因此在 OutputProcessor 的 detokenizer 中检查。

### 37.3 finish_reason 和 stop_reason 是一回事吗？

不是。

```text
finish_reason：结束类别；
stop_reason：具体 stop token id 或 stop string。
```

### 37.4 RequestOutput.request_id 为什么不是内部 req_id？

因为用户应该看到自己传入的 request id。

内部为了 parallel sampling / child request / scheduling 可能改写 request id，最终 `_new_request_output()` 会使用 `external_req_id`。

### 37.5 DELTA 模式下 token_ids 是全量还是增量？

增量。

非 DELTA 模式下通常返回 detokenizer 当前完整 output token ids。

### 37.6 PoolingRequestOutput 有没有 CompletionOutput？

没有。

Pooling 输出直接是 `PoolingOutput(data=tensor)`，再由上层转成 embedding / classification / scoring 等具体格式。

---

## 38. 最终可以记成一张表

| 阶段 | 主要对象 | 核心字段 | 作用 |
|---|---|---|---|
| 采样输入 | `SamplingMetadata` | temperature / top_p / logprobs / logitsprocs | 控制 sampler |
| 采样输出 | `SamplerOutput` | sampled_token_ids / logprobs_tensors | GPU sampler 结果 |
| Worker 输出 | `ModelRunnerOutput` | sampled_token_ids / logprobs / pooler_output | worker 返回给 scheduler |
| Engine 输出 | `EngineCoreOutput` | new_token_ids / finish_reason / stop_reason | OutputProcessor 的输入 |
| 请求状态 | `RequestState` | detokenizer / logprobs_processor / stats | 前端输出状态机 |
| 文本转换 | `IncrementalDetokenizer` | output_text / token_ids / stop strings | token ids → text |
| logprobs 转换 | `LogprobsProcessor` | logprobs / prompt_logprobs / cumulative_logprob | 内部 logprobs → API logprobs |
| 单 completion | `CompletionOutput` | text / token_ids / logprobs / finish_reason | 一个候选输出 |
| 生成请求输出 | `RequestOutput` | prompt / outputs / metrics / finished | 用户可见生成输出 |
| pooling 输出 | `PoolingRequestOutput` | PoolingOutput / prompt_token_ids | embedding / pooling 系列输出 |

---

## 39. 总结

`OutputProcessor` 的主链路可以压缩成：

```text
EngineCoreOutput
  → RequestState
  → detokenizer.update(new_token_ids)
  → stop string check
  → logprobs_processor.update_from_output()
  → make_request_output()
      → CompletionOutput
      → RequestOutput
  → finished cleanup / reqs_to_abort
```

如果只记住三句话：

```text
1. OutputProcessor 不决定生成什么 token，它决定 token 如何变成用户可见输出。
2. detokenizer 负责 text / stop string，LogprobsProcessor 负责 logprobs，RequestState 负责流式状态。
3. RequestOutput 使用 external request id，内部 child request / EngineCoreOutput 只是中间执行细节。
```

因此，`RequestOutput` 是整个采样和调度链路的最终 API 表达：

```text
内部的 token ids、logprobs、finish reason、pooling tensor、stats
最终都在 OutputProcessor 这里被整理成用户能消费的输出对象。
```
