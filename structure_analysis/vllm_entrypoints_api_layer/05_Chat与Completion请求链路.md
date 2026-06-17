# 05 Chat 与 Completion 请求链路

## 1. 两条最核心的在线请求链路

vLLM OpenAI-compatible server 最常用的两个接口是：

```http
POST /v1/chat/completions
POST /v1/completions
```

它们的共同特点是：

1. FastAPI router 接收 Pydantic request。
2. serving 对象做模型校验、输入渲染、参数转换。
3. 调用 `engine_client.generate()`。
4. 根据 `stream` 决定 SSE 流式输出或普通 JSON 输出。

## 2. Chat Completions 总链路

```text
HTTP POST /v1/chat/completions
  -> create_chat_completion()
  -> state.openai_serving_chat
  -> OpenAIServingChat.create_chat_completion()
  -> OpenAIServingChat._create_chat_completion()
  -> render_chat_request()
  -> OpenAIServingRender.render_chat()
  -> request.to_sampling_params() / to_beam_search_params()
  -> engine_client.generate()
  -> AsyncLLM.generate()
  -> AsyncLLM.add_request()
  -> InputProcessor.process_inputs()
  -> EngineCoreClient.add_request_async()
  -> output_handler 收集 EngineCore 输出
  -> ChatCompletionResponse 或 SSE chunk
```

## 3. Chat router 层

Router 文件：

```text
vllm/entrypoints/openai/chat_completion/api_router.py
```

接口定义：

```python
@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request)
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/api_router.py:40-53`。

处理逻辑：

1. 获取 endpoint load metrics header。
2. 获取 handler：`request.app.state.openai_serving_chat`。
3. 调用 `handler.create_chat_completion()`。
4. 根据返回类型转响应。

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/api_router.py:53-74`。

## 4. OpenAIServingChat 初始化

`OpenAIServingChat` 持有这些关键依赖：

- `engine_client`
- `OpenAIServingModels`
- `OpenAIServingRender`
- `RequestLogger`
- chat template
- tool parser
- reasoning parser
- 默认 sampling params

构造函数源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:108-185`。

它在 `init_generate_state()` 里创建并放入：

```python
state.openai_serving_chat
```

源码位置：`code/vllm/vllm/entrypoints/generate/api_router.py:115-145`。

## 5. Chat 请求预处理

### 5.1 入口

`create_chat_completion()` 只是包装异常和 KV transfer cleanup，实际进入 `_create_chat_completion()`：

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:235-249`。

### 5.2 render_chat_request

`render_chat_request()` 做两件关键事：

1. `_check_model(request)`：确认请求中的 model 是否可用。
2. 检查 engine 是否已经 dead。
3. 调用 `openai_serving_render.render_chat(request)` 做真正渲染。

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:208-233`。

返回：

```python
(conversation, engine_inputs)
```

其中 `engine_inputs` 是 engine 可消费的输入。

### 5.3 采样参数转换

在 `_create_chat_completion()` 中，每个 `engine_input` 会计算：

- prompt token ids
- 多模态 token 数
- request id
- max_tokens
- sampling params 或 beam search params
- trace headers
- data parallel rank
- reasoning state

关键源码：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:289-372`。

核心转换：

```python
sampling_params = request.to_sampling_params(max_tokens, self.default_sampling_params)
```

或：

```python
sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:313-322`。

### 5.4 调用 engine

普通采样路径调用：

```python
generator = self.engine_client.generate(
    engine_input,
    sampling_params,
    sub_request_id,
    ...
)
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:358-372`。

这就是 API 层和 engine 层的关键边界。

## 6. Chat 响应生成

如果 `request.stream == True`：

```python
return self.chat_completion_stream_generator(...)
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:379-390`。

否则：

```python
return await self.chat_completion_full_generator(...)
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:392-402`。

### 6.1 流式响应

`chat_completion_stream_generator()`：

1. 初始化 chunk 元信息。
2. 遍历 `result_generator`。
3. 对每个 `RequestOutput` 中的 outputs 生成 delta。
4. 如果启用 parser，则解析 reasoning/tool calls。
5. 组装 `ChatCompletionStreamResponse`。
6. yield `data: {json}\n\n`。
7. 结束时 yield `data: [DONE]\n\n`。

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:409-816`。

### 6.2 非流式响应

`chat_completion_full_generator()`：

1. 消费完整 `result_generator`。
2. 获取最后的 `RequestOutput`。
3. 遍历 choices。
4. 解析 tool calls/reasoning。
5. 组装 `ChatCompletionResponse`。

入口源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:818-900`。

## 7. Completion 总链路

```text
HTTP POST /v1/completions
  -> create_completion()
  -> state.openai_serving_completion
  -> OpenAIServingCompletion.create_completion()
  -> OpenAIServingCompletion._create_completion()
  -> render_completion_request()
  -> OpenAIServingRender.render_completion()
  -> request.to_sampling_params() / to_beam_search_params()
  -> engine_client.generate()
  -> merge_async_iterators()
  -> CompletionResponse 或 SSE chunk
```

## 8. Completion router 层

Router 文件：

```text
vllm/entrypoints/openai/completion/api_router.py
```

接口定义：

```python
@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request)
```

源码位置：`code/vllm/vllm/entrypoints/openai/completion/api_router.py:34-46`。

响应选择：

- `ErrorResponse` -> JSON error
- `CompletionResponse` -> JSON
- Async generator -> `StreamingResponse(text/event-stream)`

源码位置：`code/vllm/vllm/entrypoints/openai/completion/api_router.py:54-66`。

## 9. OpenAIServingCompletion 处理

`OpenAIServingCompletion` 初始化持有：

- `engine_client`
- `OpenAIServingModels`
- `OpenAIServingRender`
- `RequestLogger`
- 默认 sampling params
- max token override

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:55-85`。

`render_completion_request()`：

1. `_check_model(request)`。
2. 检查 engine 是否 dead。
3. 调用 `openai_serving_render.render_completion(request)`。

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:86-109`。

## 10. Completion 调用 engine

在 `_create_completion()` 中：

1. 渲染 request 得到多个 `engine_inputs`。
2. 为每个 prompt 创建独立 request id。
3. 计算 max_tokens。
4. 构造 `SamplingParams` 或 `BeamSearchParams`。
5. 调用 `engine_client.generate()`。
6. 多 prompt 情况下用 `merge_async_iterators()` 合并多个异步生成器。

关键源码：`code/vllm/vllm/entrypoints/openai/completion/serving.py:139-217`。

engine 调用源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:205-213`。

## 11. Completion 响应

### 11.1 流式响应

如果 `request.stream` 为 True：

```python
return self.completion_stream_generator(...)
```

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:222-236`。

`completion_stream_generator()` 会逐步把 engine 输出转换成：

```text
CompletionStreamResponse
```

并 yield SSE：

```text
data: {...}\n\n
data: [DONE]\n\n
```

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:280-475`。

### 11.2 非流式响应

非流式时会收集每个 prompt 的最终结果，然后调用：

```python
request_output_to_completion_response(...)
```

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:238-278`。

响应组装函数源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:477-604`。

## 12. Chat 与 Completion 的差异

| 维度 | Chat Completions | Completions |
|---|---|---|
| 输入协议 | messages 数组 | prompt 字符串或 token |
| 预处理 | chat template/render | completion render |
| tool calls | 支持，依赖 parser | 不作为主路径 |
| reasoning | 支持 reasoning parser | 较少涉及 |
| 多 prompt | 通常单 conversation | 可以多个 prompt，merge async iterators |
| 流式输出 | delta message | delta text |
| response object | `ChatCompletionResponse` | `CompletionResponse` |

## 13. API 层到 Engine 的最小公共核心

无论 chat 还是 completion，最终都收敛到：

```python
engine_client.generate(
    engine_input,
    sampling_params,
    request_id,
    lora_request=...,
    trace_headers=...,
    priority=...,
    data_parallel_rank=...,
)
```

这说明 API 层最核心的职责就是：

```text
外部协议 request
  -> EngineInput + SamplingParams + request_id + metadata
```

## 14. 关键结论

Chat 和 Completion 的主体差别在“请求渲染与响应包装”，不是 engine 调用。engine 看到的是统一的 `EngineInput` 和 `SamplingParams`。因此 vLLM 的 OpenAI API 层是协议适配层，而不是调度或执行层。