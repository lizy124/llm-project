# 10. Streaming 输出和客户端可见输出如何组织？

源码位置：

- `code/vllm/vllm/sampling_params.py`
- `code/vllm/vllm/outputs.py`
- `code/vllm/vllm/v1/engine/__init__.py`
- `code/vllm/vllm/v1/engine/output_processor.py`
- `code/vllm/vllm/v1/engine/detokenizer.py`
- `code/vllm/vllm/v1/engine/logprobs.py`
- `code/vllm/vllm/v1/engine/async_llm.py`
- `code/vllm/vllm/v1/engine/llm_engine.py`
- `code/vllm/vllm/v1/engine/parallel_sampling.py`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py`
- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py`
- `code/vllm/vllm/engine/protocol.py`

本问题关注：streaming 模式下，每轮 `EngineCoreOutputs` 如何变成增量输出；非 streaming / final-only 输出又如何组织；AsyncLLM、LLMEngine、OpenAI-compatible server、offline LLM 分别如何消费 `RequestOutput`。

---

## 1. 一句话回答

Streaming 输出的核心不是重新生成 token，而是决定：

```text
本轮新增了哪些 token / text / logprobs，
哪些应该立即返回给用户，
哪些应该缓存在 RequestState 里等待后续输出，
完成时如何附带 finish_reason / stop_reason / usage / metrics。
```

最小链路是：

```text
Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
  → LLMEngine.step() 直接返回
    或 AsyncLLM output_handler 放入 RequestOutputCollector
  → AsyncLLM.generate() yield
  → OpenAI server 转成 SSE chunk / final response
```

一句话记忆：

```text
streaming 解决的是“每轮把多少已生成内容暴露给用户”，不是“模型每轮如何生成”。
```

---

## 2. 输出模式：DELTA / CUMULATIVE / FINAL_ONLY

vLLM 内部用 `RequestOutputKind` 表示请求希望怎样返回输出。

源码位置：`code/vllm/vllm/sampling_params.py:182`

```text
CUMULATIVE：
  每次 RequestOutput 返回当前为止的完整输出。

DELTA：
  每次 RequestOutput 只返回本轮新增的 text / token_ids / logprobs。

FINAL_ONLY：
  中间轮不返回 RequestOutput，只在完成时返回最终输出。
```

对应枚举：

```text
RequestOutputKind.CUMULATIVE = 0
RequestOutputKind.DELTA = 1
RequestOutputKind.FINAL_ONLY = 2
```

`SamplingParams.output_kind` 保存这个模式。

源码位置：`code/vllm/vllm/sampling_params.py:301`

注意：

```text
output_kind 控制的是 vLLM engine 层的 RequestOutput 形态；
OpenAI server 还会在此基础上再包装成 ChatCompletionChunk / CompletionChunk / final JSON。
```

---

## 3. 从 EngineCoreOutputs 到 RequestOutput 的主链路

### 3.1 EngineCoreOutput 是 request 级内部输出

Scheduler 生成的 `EngineCoreOutput` 定义在：

`code/vllm/vllm/v1/engine/__init__.py:175`

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
prefill_stats
routed_experts
num_nans_in_logits
```

`EngineCoreOutputs` 是一批 `EngineCoreOutput`。

源码位置：`code/vllm/vllm/v1/engine/__init__.py:220`

它包含：

```text
engine_index
outputs: list[EngineCoreOutput]
scheduler_stats
timestamp
utility_output
finished_requests
```

### 3.2 OutputProcessor 是转换核心

`OutputProcessor.process_outputs()` 是内部输出到客户端输出的关键入口。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:576`

主流程：

```text
for engine_core_output in engine_core_outputs:
  → 根据 request_id 找 RequestState
  → 更新 stats
  → 读取 new_token_ids / pooling_output / finish_reason / stop_reason
  → generation：detokenizer.update(new_token_ids)
  → generation：logprobs_processor.update_from_output()
  → RequestState.make_request_output()
  → AsyncLLM：放入 per-request queue
  → LLMEngine：加入本轮返回列表
  → finished 后清理 RequestState
```

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:606`
- `code/vllm/vllm/v1/engine/output_processor.py:618`
- `code/vllm/vllm/v1/engine/output_processor.py:635`
- `code/vllm/vllm/v1/engine/output_processor.py:639`
- `code/vllm/vllm/v1/engine/output_processor.py:648`
- `code/vllm/vllm/v1/engine/output_processor.py:651`
- `code/vllm/vllm/v1/engine/output_processor.py:661`
- `code/vllm/vllm/v1/engine/output_processor.py:669`

一句话：

```text
OutputProcessor 是唯一应该全量遍历 EngineCoreOutputs 的 Python 层输出处理点。
```

源码注释也强调：为了减少 Python overhead，vLLM V1 尽量只在 `process_outputs()` 里循环处理 batch。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:594` 到 `code/vllm/vllm/v1/engine/output_processor.py:601`

---

## 4. RequestState 如何决定本轮是否返回

`RequestState` 是 frontend/output processor 侧保存每个请求输出状态的对象。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:129`

它保存：

```text
request_id / external_req_id
parent_req / request_index
output_kind
prompt / prompt_token_ids / prompt_embeds
logprobs_processor
detokenizer
max_tokens_param
queue
stats
stream_interval
sent_tokens_offset
streaming_input 状态
routed_experts_chunks
```

### 4.1 `make_request_output()` 是输出节流点

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:272`

它决定：

```text
- 当前是否 finished；
- FINAL_ONLY 是否要跳过中间输出；
- stream_interval 是否允许本轮输出；
- DELTA 模式下本轮 token 范围；
- generation 还是 pooling；
- n>1 parent request 如何聚合；
- 最终构造 RequestOutput 还是 PoolingRequestOutput。
```

### 4.2 FINAL_ONLY：未完成时不输出

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:280` 到 `code/vllm/vllm/v1/engine/output_processor.py:285`

逻辑是：

```text
if not finished and output_kind == FINAL_ONLY:
  return None
```

也就是说：

```text
FINAL_ONLY 不是模型不生成中间 token，
而是 OutputProcessor 不把中间 RequestOutput 暴露给客户端。
```

### 4.3 stream_interval：控制几 token 输出一次

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:287` 到 `code/vllm/vllm/v1/engine/output_processor.py:300`

当 `stream_interval > 1` 时，只有满足以下条件才输出：

```text
1. 请求 finished；
2. 这是第一次输出；
3. 距离上次输出的 token 数达到 stream_interval。
```

否则返回 `None`，本轮输出被缓存到 detokenizer / logprobs processor 的状态里。

`stream_interval` 从 scheduler config 传入 `OutputProcessor`：

- `code/vllm/vllm/v1/engine/llm_engine.py:97` 到 `code/vllm/vllm/v1/engine/llm_engine.py:102`
- `code/vllm/vllm/v1/engine/async_llm.py:138` 到 `code/vllm/vllm/v1/engine/async_llm.py:143`

### 4.4 DELTA：只返回上次以后新增内容

在 `make_request_output()` 中，DELTA 模式会按 `sent_tokens_offset` 取新增 token。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:302` 到 `code/vllm/vllm/v1/engine/output_processor.py:309`

逻辑是：

```text
new_token_ids = detokenizer.output_token_ids[sent_tokens_offset:]
sent_tokens_offset = detokenizer.num_output_tokens()
```

之后 `_new_completion_output()` 会调用：

```text
detokenizer.get_next_output_text(finished, delta=True)
```

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:376` 到 `code/vllm/vllm/v1/engine/output_processor.py:389`

所以 DELTA 模式下：

```text
CompletionOutput.text：本轮新增 text
CompletionOutput.token_ids：本轮新增 token ids
CompletionOutput.logprobs：本轮新增 logprobs
```

### 4.5 CUMULATIVE：每轮返回完整输出

如果不是 DELTA，则 `_new_completion_output()` 会：

```text
text = 当前完整 output_text
token_ids = detokenizer.output_token_ids
logprobs = logprobs_processor.logprobs
```

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:388`
- `code/vllm/vllm/v1/engine/output_processor.py:389`
- `code/vllm/vllm/v1/engine/output_processor.py:392`

所以 CUMULATIVE 模式下：

```text
每个 RequestOutput 都可以单独代表“到当前为止”的完整结果。
```

---

## 5. token、text、delta_text 的关系

### 5.1 四类 token/text 状态

```text
new_token_ids：
  Scheduler 本轮返回给 OutputProcessor 的新 token ids，来自 EngineCoreOutput。

output_token_ids：
  detokenizer 当前累计的输出 token ids。

text / output_text：
  detokenizer 当前累计的输出文本。

delta_text：
  DELTA / OpenAI streaming 场景本轮要发给客户端的新文本。
```

### 5.2 detokenizer.update() 先更新内部状态

generation 输出会先调用：

```text
detokenizer.update(new_token_ids, finish_reason == FinishReason.STOP)
```

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:639`

`IncrementalDetokenizer.update()` 会：

```text
1. 把 new_token_ids 加入 token_ids；
2. 增量 decode 成 text；
3. 检查 stop strings；
4. 必要时截断 stop string；
5. 返回匹配到的 stop string。
```

源码位置：`code/vllm/vllm/v1/engine/detokenizer.py:95` 到 `code/vllm/vllm/v1/engine/detokenizer.py:142`

如果 detokenizer 发现 stop string，会把 `finish_reason` 改成 `STOP`，并把 stop string 作为 `stop_reason`。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:642` 到 `code/vllm/vllm/v1/engine/output_processor.py:644`

### 5.3 get_next_output_text() 决定输出多少 text

源码位置：`code/vllm/vllm/v1/engine/detokenizer.py:148`

```text
CUMULATIVE：
  返回完整 output_text。

DELTA：
  返回上次调用以来新增的 text。

未 finished 且有 stop string buffer：
  暂时保留最后若干字符，避免 stop string 被过早 streamed 出去。
```

stop string buffer 的来源：

```text
如果配置了 stop strings 且不 include_stop_str_in_output，
就保留 max_stop_string_len - 1 个字符。
```

源码位置：`code/vllm/vllm/v1/engine/detokenizer.py:84` 到 `code/vllm/vllm/v1/engine/detokenizer.py:89`

### 5.4 Fast / Slow detokenizer

`IncrementalDetokenizer.from_new_request()` 会根据 tokenizer 类型选择：

```text
FastIncrementalDetokenizer：
  对 PreTrainedTokenizerFast 使用 tokenizers DecodeStream。

SlowIncrementalDetokenizer：
  使用 Python 侧 detokenize_incrementally。

空 tokenizer：
  不做文本 detokenize，只保留 token ids。
```

源码位置：`code/vllm/vllm/v1/engine/detokenizer.py:49` 到 `code/vllm/vllm/v1/engine/detokenizer.py:65`

Fast detokenizer 处理：

```text
skip_special_tokens
spaces_between_special_tokens
特殊 token 连续出现时的空格处理
invalid prefix 恢复
```

源码位置：`code/vllm/vllm/v1/engine/detokenizer.py:167` 到 `code/vllm/vllm/v1/engine/detokenizer.py:247`

Slow detokenizer 处理：

```text
prompt token 预填充
prefix_offset / read_offset
skip_special_tokens
spaces_between_special_tokens
```

源码位置：`code/vllm/vllm/v1/engine/detokenizer.py:250` 到 `code/vllm/vllm/v1/engine/detokenizer.py:306`

---

## 6. RequestOutput / CompletionOutput 如何组织

用户可见 generation 输出定义在：`code/vllm/vllm/outputs.py`

### 6.1 CompletionOutput

源码位置：`code/vllm/vllm/outputs.py:21`

`CompletionOutput` 表示一个 completion 分支。

字段：

```text
index：
  第几个 completion，n > 1 时用于区分多个输出。

text：
  输出文本。DELTA 时是增量文本，CUMULATIVE / FINAL_ONLY 时是完整文本。

token_ids：
  输出 token ids。DELTA 时是增量 token ids，CUMULATIVE / FINAL_ONLY 时是完整 token ids。

cumulative_logprob：
  当前累计 logprob。

logprobs：
  输出 token 对应的 logprobs。DELTA 时是增量 logprobs。

routed_experts：
  可选 MoE 路由信息，通常 finished 时合并。

finish_reason / stop_reason：
  完成原因和触发 stop 的字符串 / token id。

lora_request：
  当前输出使用的 LoRA。
```

`CompletionOutput.finished()` 判断 `finish_reason is not None`。

源码位置：`code/vllm/vllm/outputs.py:50`

### 6.2 RequestOutput

源码位置：`code/vllm/vllm/outputs.py:85`

`RequestOutput` 表示一个用户请求的输出。

字段：

```text
request_id：外部 request id；
prompt：原 prompt 文本；
prompt_token_ids：prompt token ids；
prompt_logprobs：prompt token logprobs；
outputs：list[CompletionOutput]；
finished：整个请求是否完成；
metrics：请求统计；
lora_request：使用的 LoRA；
encoder_prompt / encoder_prompt_token_ids：encoder-decoder 场景；
num_cached_tokens：prefix cache 命中 token 数；
kv_transfer_params：KV transfer 返回给客户端的参数。
```

### 6.3 RequestOutput.add() 用于合并增量

源码位置：`code/vllm/vllm/outputs.py:145`

`RequestOutput.add(next_output, aggregate)` 用于把多个 `RequestOutput` 合并。

如果 `aggregate=True`：

```text
- text 追加；
- token_ids extend；
- logprobs extend；
- cumulative_logprob 使用最新值；
- finish_reason / stop_reason 使用最新值。
```

如果 `aggregate=False`：

```text
同 index 的 CompletionOutput 直接被替换为新输出。
```

这个机制主要给 `RequestOutputCollector` 使用：当 producer 比 consumer 快时，可以把多个 delta 合并，减少队列积压。

---

## 7. AsyncLLM：每个请求一个输出队列

### 7.1 RequestOutputCollector 是 per-request queue

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:45`

`RequestOutputCollector` 是 AsyncLLM 场景下每个请求的输出收集器。

它包含：

```text
output_kind
request_id
output
ready event
input stream task
```

`put()` 是非阻塞写入：

```text
- 如果当前没有缓存输出，直接保存并 set event；
- 如果已有 RequestOutput，调用 RequestOutput.add() 合并；
- 如果是 PoolingRequestOutput，直接替换；
- 如果是 Exception，保存异常并唤醒 consumer。
```

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:62` 到 `code/vllm/vllm/v1/engine/output_processor.py:77`

`get()` 会阻塞等待 output，`get_nowait()` 尝试立即取出。

源码位置：

- `code/vllm/vllm/v1/engine/output_processor.py:78`
- `code/vllm/vllm/v1/engine/output_processor.py:88`

### 7.2 AsyncLLM.add_request() 创建 collector

`AsyncLLM.add_request()` 会：

```text
1. 把外部 input 转成 EngineCoreRequest；
2. 启动 output_handler；
3. 创建 RequestOutputCollector；
4. OutputProcessor.add_request() 注册 RequestState；
5. engine_core.add_request_async() 送入 EngineCore。
```

源码位置：

- `code/vllm/vllm/v1/engine/async_llm.py:280`
- `code/vllm/vllm/v1/engine/async_llm.py:368`
- `code/vllm/vllm/v1/engine/async_llm.py:373`
- `code/vllm/vllm/v1/engine/async_llm.py:376`
- `code/vllm/vllm/v1/engine/async_llm.py:400`

### 7.3 output_handler 从 EngineCore 拉输出并放入队列

`AsyncLLM._run_output_handler()` 启动后台任务。

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:637`

后台任务循环：

```text
while True:
  outputs = await engine_core.get_output_async()
  按 chunk_size 切分 outputs.outputs
  output_processor.process_outputs(...)
  process_outputs 内部把 RequestOutput 放入对应 queue
  如果 stop string 需要 abort，则调用 engine_core.abort_requests_async()
  更新 scheduler stats / logging
```

源码位置：

- `code/vllm/vllm/v1/engine/async_llm.py:656`
- `code/vllm/vllm/v1/engine/async_llm.py:660`
- `code/vllm/vllm/v1/engine/async_llm.py:671`
- `code/vllm/vllm/v1/engine/async_llm.py:675`
- `code/vllm/vllm/v1/engine/async_llm.py:685`
- `code/vllm/vllm/v1/engine/async_llm.py:691`

### 7.4 AsyncLLM.generate() yield RequestOutput

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:524`

主流程：

```text
q = await add_request(...)
while not finished:
  out = q.get_nowait() or await q.get()
  finished = out.finished
  if out is not STREAM_FINISHED:
    yield out
```

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:557` 到 `code/vllm/vllm/v1/engine/async_llm.py:586`

如果 client 断开、generator 被取消，AsyncLLM 会 abort request。

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:588` 到 `code/vllm/vllm/v1/engine/async_llm.py:596`

一句话：

```text
AsyncLLM 的 streaming 是：EngineCore outputs → OutputProcessor → per-request queue → async generator yield。
```

---

## 8. LLMEngine / offline LLM：同步拉取最终或中间输出

### 8.1 LLMEngine.step() 返回本轮 RequestOutput 列表

源码位置：`code/vllm/vllm/v1/engine/llm_engine.py:296`

主流程：

```text
outputs = engine_core.get_output()
processed_outputs = output_processor.process_outputs(outputs.outputs, ...)
engine_core.abort_requests(processed_outputs.reqs_to_abort)
record stats
return processed_outputs.request_outputs
```

源码位置：

- `code/vllm/vllm/v1/engine/llm_engine.py:302`
- `code/vllm/vllm/v1/engine/llm_engine.py:307`
- `code/vllm/vllm/v1/engine/llm_engine.py:317`
- `code/vllm/vllm/v1/engine/llm_engine.py:334`

LLMEngine 没有 per-request async queue。

```text
它每 step 拉一批 EngineCoreOutputs，处理后同步返回 RequestOutput list。
```

### 8.2 Offline LLM.generate() 通常返回最终 list

Offline `LLM.generate()` 返回：

```text
list[RequestOutput]
```

源码位置：`code/vllm/vllm/entrypoints/llm.py:422` 到 `code/vllm/vllm/entrypoints/llm.py:485`

它底层会通过 offline mixin 把请求加入 engine，然后不断 step，直到所有请求完成。

从使用者视角看：

```text
LLM.generate() 通常不是 SSE streaming 接口；
它最终返回按输入顺序排列的 RequestOutput 列表。
```

如果用户想显式先入队再等待，可以用：

```text
enqueue()
wait_for_completion()
```

源码位置：

- `code/vllm/vllm/entrypoints/llm.py:487`
- `code/vllm/vllm/entrypoints/llm.py:547`

---

## 9. OpenAI Completion streaming 如何包装 RequestOutput

OpenAI-compatible Completion API 的 streaming 入口在：

`code/vllm/vllm/entrypoints/openai/completion/serving.py:280`

### 9.1 请求先变成 AsyncLLM result_generator

Completion 请求会：

```text
render request
  → 构造 SamplingParams
  → engine_client.generate(...)
  → result_generator: AsyncIterator[RequestOutput]
```

源码位置：

- `code/vllm/vllm/entrypoints/openai/completion/serving.py:139`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py:170`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py:205`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py:217`

如果 `request.stream` 为真，则进入 `completion_stream_generator()`。

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:225`

### 9.2 Completion stream chunk 处理

`completion_stream_generator()` 会遍历：

```text
async for prompt_idx, res in result_generator:
  for output in res.outputs:
    根据 echo / return_token_ids / logprobs 计算 delta_text / delta_token_ids
    构造 CompletionStreamResponse
    yield f"data: {json}\n\n"
```

源码位置：

- `code/vllm/vllm/entrypoints/openai/completion/serving.py:305`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py:326`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py:335`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py:359`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py:398`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py:436`

它会跳过 chunked prefill 产生的空 chunk：

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:370` 到 `code/vllm/vllm/entrypoints/openai/completion/serving.py:376`

### 9.3 usage 和 [DONE]

如果启用 continuous usage，stream 中每个 chunk 可以携带 usage。

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:427` 到 `code/vllm/vllm/entrypoints/openai/completion/serving.py:434`

如果启用 include_usage，则最后额外发一个 usage chunk。

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:452` 到 `code/vllm/vllm/entrypoints/openai/completion/serving.py:464`

最后发送：

```text
data: [DONE]\n\n
```

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:475`

### 9.4 非 streaming completion

非 streaming 会消费完整 `result_generator`，保存最后的 `RequestOutput`，再调用 `request_output_to_completion_response()` 组装最终 JSON。

源码位置：

- `code/vllm/vllm/entrypoints/openai/completion/serving.py:238`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py:255`
- `code/vllm/vllm/entrypoints/openai/completion/serving.py:477`

最终 response 会包含：

```text
choices
usage
system_fingerprint
kv_transfer_params
```

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:596` 到 `code/vllm/vllm/entrypoints/openai/completion/serving.py:604`

---

## 10. OpenAI Chat streaming 如何包装 RequestOutput

Chat Completion 的 streaming 入口在：

`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:409`

### 10.1 第一帧先发送 role

Chat streaming 会在第一轮先发送 role chunk：

```text
DeltaMessage(role=role, content="")
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:478` 到 `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:535`

这和 Completion API 不同：

```text
Chat stream 的第一帧通常不是模型 token，而是 assistant role 初始化。
```

### 10.2 正常 token chunk

之后每个 `RequestOutput` 会遍历 `res.outputs`：

```text
output.text / output.token_ids
  → 可选 parser.parse_delta()
  → DeltaMessage(content / reasoning / tool_calls)
  → ChatCompletionStreamResponse
  → yield SSE data
```

源码位置：

- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:573`
- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:591`
- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:603`
- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:627`
- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:677`
- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:724`
- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:751`

如果没有 delta_text / token_ids 且是 chunked prefill 的空输出，也会跳过。

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:593` 到 `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:599`

### 10.3 reasoning / tool calls parser

如果配置了 parser，Chat stream 不直接把所有 text 当 content，而是：

```text
parser.parse_delta(
  delta_text,
  delta_token_ids,
  request,
  prompt_token_ids,
  finished,
)
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:603` 到 `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:610`

parser 可以把模型输出拆成：

```text
content
reasoning
tool_calls
```

如果 parser 暂时不产生可发送内容，则可能跳过当前 token chunk；如果 `return_token_ids` 启用，则仍可能发送 token ids。

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:634` 到 `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:647`

### 10.4 finish chunk / usage / DONE

当 `output.finish_reason` 非空时，Chat stream 会发送带 finish_reason 的最后一个 choice chunk。

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:691` 到 `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:721`

如果产生了 tool calls，finish_reason 可能转换为：

```text
tool_calls
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:697` 到 `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:707`

如果启用 usage，则发送额外 usage chunk。

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:754` 到 `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:781`

最后发送：

```text
data: [DONE]\n\n
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:815` 到 `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:816`

---

## 11. n > 1 / parallel sampling 的输出组织

当 `SamplingParams.n > 1` 时，vLLM 会把一个外部请求拆成多个内部 child request。

`SamplingParams` 注释说明：

```text
AsyncLLM 默认 streaming outputs；n > 1 时所有 n 个输出会被生成并按 request 进行 streaming。
如果希望完成时一次看到所有 n 个 outputs，可使用 FINAL_ONLY。
```

源码位置：`code/vllm/vllm/sampling_params.py:213` 到 `code/vllm/vllm/sampling_params.py:223`

### 11.1 ParentRequest 管理 child request

源码位置：`code/vllm/vllm/v1/engine/parallel_sampling.py:13`

`ParentRequest` 负责：

```text
- 保存外部 parent request id；
- 给每个 child 分配 request id；
- 为每个 child 复制 SamplingParams，并设置 n=1；
- 如果 seed 存在，则给每个 child 使用不同 seed；
- 追踪哪些 child 已完成；
- FINAL_ONLY 时聚合所有 child 的 CompletionOutput。
```

关键源码位置：

- `code/vllm/vllm/v1/engine/parallel_sampling.py:36`
- `code/vllm/vllm/v1/engine/parallel_sampling.py:83`
- `code/vllm/vllm/v1/engine/parallel_sampling.py:100`

### 11.2 RequestState 根据 ParentRequest 聚合输出

在 `RequestState.make_request_output()` 中，如果有 `parent_req`，会调用：

```text
parent_req.get_outputs(self.request_id, output)
```

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:321` 到 `code/vllm/vllm/v1/engine/output_processor.py:327`

`ParentRequest.get_outputs()` 的规则：

```text
非 FINAL_ONLY：
  当前 child 有输出就返回当前 completion。

FINAL_ONLY：
  先聚合各 child 的最终 CompletionOutput；
  等所有 child finished 后一次性返回完整 outputs。
```

源码位置：`code/vllm/vllm/v1/engine/parallel_sampling.py:100` 到 `code/vllm/vllm/v1/engine/parallel_sampling.py:126`

---

## 12. streaming input 和 STREAM_FINISHED

vLLM 还有一种“输入流式追加”的场景，用 `StreamingInput` 表示。

源码位置：`code/vllm/vllm/engine/protocol.py:28`

它用于多轮 streaming session，输入通过 async generator 持续给 engine。

### 12.1 AsyncLLM 处理 streaming input

`AsyncLLM.generate()` 可以接收：

```text
AsyncGenerator[StreamingInput, None]
```

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:524` 到 `code/vllm/vllm/v1/engine/async_llm.py:541`

如果 prompt 是 async generator，会进入 `_add_streaming_input_request()`。

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:316` 到 `code/vllm/vllm/v1/engine/async_llm.py:331`

这个路径会：

```text
- 为每个输入 chunk 创建 resumable EngineCoreRequest；
- 使用同一个 internal request id；
- 把更新交给 OutputProcessor；
- 输入流结束后发送一个 final request 表示输入结束。
```

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:458` 到 `code/vllm/vllm/v1/engine/async_llm.py:495`

### 12.2 OutputProcessor 处理 streaming update

如果 `OutputProcessor.add_request()` 发现 request state 已存在，会调用 `_update_streaming_request_state()`。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:520` 到 `code/vllm/vllm/v1/engine/output_processor.py:524`

streaming update 会进入队列，等当前 sub-request 完成后再应用。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:543` 到 `code/vllm/vllm/v1/engine/output_processor.py:575`

当输入最终结束且需要 unblock generate loop 时，OutputProcessor 可能发送 `STREAM_FINISHED`。

源码位置：

- `code/vllm/vllm/outputs.py:191`
- `code/vllm/vllm/v1/engine/output_processor.py:551` 到 `code/vllm/vllm/v1/engine/output_processor.py:555`

AsyncLLM.generate() 会识别并跳过这个 sentinel，不把它 yield 给用户。

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:581` 到 `code/vllm/vllm/v1/engine/async_llm.py:586`

---

## 13. stop string、finish_reason、stop_reason 的组织

### 13.1 finish_reason 来源

`FinishReason` 定义在：`code/vllm/vllm/v1/engine/__init__.py:42`

可能值：

```text
STOP：stop string / stop token / EOS 等停止；
LENGTH：达到 max_tokens 或 max_model_len；
ABORT：请求被 abort；
ERROR：请求级内部错误；
REPETITION：重复模式检测触发。
```

外部字符串映射定义在：`code/vllm/vllm/v1/engine/__init__.py:28` 到 `code/vllm/vllm/v1/engine/__init__.py:30`

### 13.2 stop string 可能在 OutputProcessor 侧发现

Scheduler / EngineCore 可能已经给出 `finish_reason`，但 stop string 检查还会在 detokenizer 中进行。

如果 OutputProcessor 发现 stop string：

```text
finish_reason = STOP
stop_reason = stop_string
```

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:639` 到 `code/vllm/vllm/v1/engine/output_processor.py:644`

如果 OutputProcessor 侧发现 stop string，但 EngineCore 还没 finished，需要 abort EngineCore 中的请求。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:677` 到 `code/vllm/vllm/v1/engine/output_processor.py:681`

LLMEngine 和 AsyncLLM 都会把 `reqs_to_abort` 转发给 EngineCore：

- `code/vllm/vllm/v1/engine/llm_engine.py:317`
- `code/vllm/vllm/v1/engine/async_llm.py:685` 到 `code/vllm/vllm/v1/engine/async_llm.py:689`

### 13.3 CompletionOutput 最终携带 finish_reason / stop_reason

`RequestState._new_completion_output()` 会把 finished 状态写入 `CompletionOutput`：

```text
finish_reason = str(finish_reason) if finished else None
stop_reason = stop_reason if finished else None
```

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:402` 到 `code/vllm/vllm/v1/engine/output_processor.py:410`

---

## 14. Pooling / Embedding 输出组织

Pooling 请求不走 completion detokenize。

在 `RequestState.make_request_output()` 中，如果 `pooling_output is not None`，直接创建 pooling 输出。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:312` 到 `code/vllm/vllm/v1/engine/output_processor.py:317`

`_new_request_output()` 会识别 `PoolingOutput` 并返回 `PoolingRequestOutput`。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:346` 到 `code/vllm/vllm/v1/engine/output_processor.py:355`

用户可见 pooling 类型定义在：

- `code/vllm/vllm/outputs.py:66`
- `code/vllm/vllm/outputs.py:204`
- `code/vllm/vllm/outputs.py:240`

包括：

```text
PoolingOutput
PoolingRequestOutput
EmbeddingOutput / EmbeddingRequestOutput
ClassificationOutput / ClassificationRequestOutput
ScoringOutput / ScoringRequestOutput
```

一句话：

```text
generation 输出走 token → text，pooling 输出走 tensor → PoolingOutput，不走普通 streaming token chunk。
```

---

## 15. 客户端可见输出有哪些层次

可以把客户端输出分成三层：

```text
1. vLLM Python 对象层：
   RequestOutput
   CompletionOutput
   PoolingRequestOutput
   EmbeddingRequestOutput

2. AsyncLLM generator 层：
   async for out in engine.generate(...):
     yield RequestOutput

3. OpenAI-compatible HTTP 层：
   non-streaming JSON response
   streaming SSE chunk：data: {...}\n\n
   terminal event：data: [DONE]\n\n
```

关系是：

```text
EngineCoreOutput
  → OutputProcessor
  → RequestOutput
  → AsyncLLM.generate() yield
  → OpenAI serving stream generator
  → ChatCompletionStreamResponse / CompletionStreamResponse
  → SSE data chunk
```

---

## 16. 容易混淆的点

### 16.1 streaming 是否影响模型生成？

通常不影响。

streaming 主要影响：

```text
RequestOutput 何时返回；
返回 DELTA 还是 CUMULATIVE；
OpenAI server 是否包装成 SSE chunk。
```

模型仍然按 Scheduler / Worker 的 step 生成 token。

### 16.2 DELTA 和 OpenAI SSE chunk 是一回事吗？

不是。

```text
DELTA 是 vLLM RequestOutput 的 output_kind；
SSE chunk 是 OpenAI HTTP server 的传输格式。
```

OpenAI stream generator 会把 `RequestOutput` 再转换成协议对象。

### 16.3 FINAL_ONLY 是不是不保存中间 token？

不是。

中间 token 仍然在 engine / request state 中推进，只是 `make_request_output()` 在未 finished 时返回 `None`。

### 16.4 chunked prefill 为什么可能不返回 chunk？

chunked prefill 可能产生空 text / 空 token 的中间输出。OpenAI stream generator 会跳过这种空 chunk。

源码位置：

- `code/vllm/vllm/entrypoints/openai/completion/serving.py:370`
- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:593`

### 16.5 stop string 是 Scheduler 判断的吗？

不完全是。

Scheduler 处理 token 级 stop / length 等状态；字符串 stop 需要 detokenized text，因此在 OutputProcessor / detokenizer 侧检查。

### 16.6 AsyncLLM 的 queue 会无限堆积吗？

`RequestOutputCollector.put()` 会在 producer 快于 consumer 时合并同一个 request 的输出；DELTA 模式下会 aggregate，避免每个小 chunk 都堆成独立对象。

源码位置：`code/vllm/vllm/v1/engine/output_processor.py:62` 到 `code/vllm/vllm/v1/engine/output_processor.py:77`

### 16.7 Chat stream 的第一帧为什么没有 token？

Chat Completion API 需要先发送 role，因此第一帧通常是：

```text
DeltaMessage(role="assistant", content="")
```

这不是模型新生成的 token，而是协议层初始化 chunk。

### 16.8 n > 1 时输出为什么有多个 index？

因为一个外部请求会拆成多个内部 child request，每个 child 的 completion 用 `CompletionOutput.index` 区分。

---

## 17. 总结

Streaming / Client Output 的完整链路可以压缩为：

```text
EngineCoreOutput(new_token_ids, finish_reason, ...)
  → OutputProcessor.process_outputs()
  → RequestState.detokenizer.update()
  → RequestState.make_request_output()
      → DELTA / CUMULATIVE / FINAL_ONLY
      → stream_interval 节流
      → RequestOutput / PoolingRequestOutput
  → LLMEngine.step() 返回 list
    或 AsyncLLM output_handler 放入 RequestOutputCollector
  → AsyncLLM.generate() yield RequestOutput
  → OpenAI server 转成 final JSON 或 SSE chunks
  → client
```

如果只记住一句话：

```text
OutputProcessor 决定 RequestOutput 的粒度，AsyncLLM / LLMEngine 决定如何把 RequestOutput 交给调用者，OpenAI serving 决定如何把它包装成 HTTP response 或 SSE chunk。
```

再压缩一层：

```text
模型生成 token，Scheduler 生成 EngineCoreOutput，OutputProcessor 生成 RequestOutput，entrypoints 生成客户端协议输出。
```
