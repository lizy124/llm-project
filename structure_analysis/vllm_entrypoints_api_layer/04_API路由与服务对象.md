# 04 API 路由与服务对象

## 1. 路由层与服务对象层的分工

vLLM API 层通常拆成两类文件：

| 文件类型 | 命名 | 作用 |
|---|---|---|
| Router | `api_router.py` | 声明 HTTP path、method、依赖、返回类型，从 `app.state` 取 handler |
| Serving | `serving.py` | 真正处理请求，做校验、渲染、参数转换、调用 engine、组装响应 |
| Protocol | `protocol.py` | Pydantic request/response model，定义 OpenAI/Anthropic/vLLM 协议结构 |

典型例子：

```text
vllm/entrypoints/openai/chat_completion/
  ├─ api_router.py
  ├─ protocol.py
  ├─ serving.py
  └─ batch_serving.py
```

## 2. OpenAI-compatible generate API

### 2.1 Chat Completions

路径：

```http
POST /v1/chat/completions
```

Router 文件：

```text
vllm/entrypoints/openai/chat_completion/api_router.py
```

核心函数：

```python
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request)
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/api_router.py:40-74`。

处理逻辑：

1. 从 `raw_request.app.state.openai_serving_chat` 获取 handler。
2. 如果 handler 为 None，说明模型不支持 Chat Completions。
3. 调用 `handler.create_chat_completion(request, raw_request)`。
4. 如果返回 `ErrorResponse`，返回错误 JSON。
5. 如果返回 `ChatCompletionResponse`，返回普通 JSON。
6. 否则返回 `StreamingResponse`，媒体类型为 `text/event-stream`。

服务对象：

```text
OpenAIServingChat
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:108-185`。

### 2.2 Batch Chat Completions

路径：

```http
POST /v1/chat/completions/batch
```

源码位置：`code/vllm/vllm/entrypoints/openai/chat_completion/api_router.py:77-102`。

handler：

```text
state.openai_serving_chat_batch
```

### 2.3 Completions

路径：

```http
POST /v1/completions
```

Router 文件：

```text
vllm/entrypoints/openai/completion/api_router.py
```

源码位置：`code/vllm/vllm/entrypoints/openai/completion/api_router.py:34-70`。

处理逻辑与 chat 类似：

1. 从 `app.state.openai_serving_completion` 获取 handler。
2. 调用 `handler.create_completion()`。
3. 根据返回值选择 JSON 或 SSE。

服务对象：

```text
OpenAIServingCompletion
```

源码位置：`code/vllm/vllm/entrypoints/openai/completion/serving.py:55-85`。

### 2.4 Responses API

路径：

```http
POST /v1/responses
GET  /v1/responses/{response_id}
POST /v1/responses/{response_id}/cancel
```

Router 文件：

```text
vllm/entrypoints/openai/responses/api_router.py
```

源码位置：

- create：`code/vllm/vllm/entrypoints/openai/responses/api_router.py:48-77`
- retrieve：`code/vllm/vllm/entrypoints/openai/responses/api_router.py:80-107`
- cancel：`code/vllm/vllm/entrypoints/openai/responses/api_router.py:110-124`

服务对象：

```text
OpenAIServingResponses
```

它支持更接近 OpenAI 新版 Responses API 的事件流格式，流式输出会被 `_convert_stream_to_sse_events()` 转成 SSE event：`code/vllm/vllm/entrypoints/openai/responses/api_router.py:34-45`。

### 2.5 Models API

路径：

```http
GET /v1/models
```

Router 文件：

```text
vllm/entrypoints/openai/models/api_router.py
```

源码位置：`code/vllm/vllm/entrypoints/openai/models/api_router.py:20-29`。

服务对象：

```text
OpenAIServingModels
```

它在 `init_app_state()` 中初始化：`code/vllm/vllm/entrypoints/openai/api_server.py:356-361`。

## 3. Anthropic-compatible API

Router 文件：

```text
vllm/entrypoints/anthropic/api_router.py
```

主要接口：

```http
POST /v1/messages
POST /v1/messages/count_tokens
```

源码位置：

- messages：`code/vllm/vllm/entrypoints/anthropic/api_router.py:49`
- count tokens：`code/vllm/vllm/entrypoints/anthropic/api_router.py:95`

服务对象：

```text
AnthropicServingMessages
```

它在 generate state 中初始化：`code/vllm/vllm/entrypoints/generate/api_router.py:158-177`。

## 4. Pooling API

Pooling 相关 router 由：

```text
vllm/entrypoints/pooling/factories.py
```

统一注册。

入口函数：

```python
register_pooling_api_routers(app, supported_tasks, model_config)
```

源码位置：`code/vllm/vllm/entrypoints/pooling/factories.py:104-135`。

根据模型支持的任务动态注册：

| 任务 | Router | 典型路径 | Serving 对象 |
|---|---|---|---|
| embed | `pooling/embed/api_router.py` | `/v1/embeddings`、`/embed` | `ServingEmbedding` |
| classify | `pooling/classify/api_router.py` | `/classify` | `ServingClassification` |
| scoring/rerank | `pooling/scoring/api_router.py` | `/score`、`/rerank` 等 | `ServingScores` |
| generic pooling | `pooling/pooling/api_router.py` | `/pooling` | `ServingPooling` |

`init_pooling_state()` 会创建对应 serving 对象：`code/vllm/vllm/entrypoints/pooling/factories.py:137-211`。

## 5. vLLM serve 管理类 API

通用 serve router 由：

```text
vllm/entrypoints/serve/__init__.py
```

注册：

```python
register_vllm_serve_api_routers(app)
```

源码位置：`code/vllm/vllm/entrypoints/serve/__init__.py:11-32`。

包含：

### 5.1 Instrumentator

注册文件：`vllm/entrypoints/serve/instrumentator/__init__.py`

接口包括：

- basic endpoints
- health endpoints
- metrics endpoints
- offline docs endpoints

源码位置：`code/vllm/vllm/entrypoints/serve/instrumentator/__init__.py:7-22`。

### 5.2 LoRA 管理

Router：

```text
vllm/entrypoints/serve/lora/api_router.py
```

接口：

```http
POST /v1/load_lora_adapter
POST /v1/unload_lora_adapter
```

源码位置：`code/vllm/vllm/entrypoints/serve/lora/api_router.py:43`、`code/vllm/vllm/entrypoints/serve/lora/api_router.py:59`。

### 5.3 Profile

Router：

```text
vllm/entrypoints/serve/profile/api_router.py
```

接口：

```http
POST /start_profile
POST /stop_profile
```

源码位置：`code/vllm/vllm/entrypoints/serve/profile/api_router.py:21`、`code/vllm/vllm/entrypoints/serve/profile/api_router.py:29`。

### 5.4 Tokenize / Detokenize

Router：

```text
vllm/entrypoints/serve/tokenize/api_router.py
```

接口包括：

```http
POST /tokenize
POST /detokenize
GET  /tokenizer_info
```

源码位置：

- `code/vllm/vllm/entrypoints/serve/tokenize/api_router.py:38`
- `code/vllm/vllm/entrypoints/serve/tokenize/api_router.py:64`
- `code/vllm/vllm/entrypoints/serve/tokenize/api_router.py:100`

服务对象：

```text
OpenAIServingTokenization
```

在 `init_app_state()` 中创建：`code/vllm/vllm/entrypoints/openai/api_server.py:379-388`。

## 6. Disaggregated / Tokens API

Router：

```text
vllm/entrypoints/serve/disagg/api_router.py
```

接口包括：

```http
POST /v1/completions/tokens
POST /abort_requests
```

源码位置：

- `code/vllm/vllm/entrypoints/serve/disagg/api_router.py:49`
- `code/vllm/vllm/entrypoints/serve/disagg/api_router.py:82`

服务对象：

```text
ServingTokens
```

初始化位置：`code/vllm/vllm/entrypoints/generate/api_router.py:178-191`。

## 7. Elastic EP API

Router：

```text
vllm/entrypoints/serve/elastic_ep/api_router.py
```

接口包括：

```http
POST /scale_elastic_ep
POST /is_scaling_elastic_ep
```

源码位置：

- `code/vllm/vllm/entrypoints/serve/elastic_ep/api_router.py:32`
- `code/vllm/vllm/entrypoints/serve/elastic_ep/api_router.py:90`

## 8. Render API

Router：

```text
vllm/entrypoints/serve/render/api_router.py
```

用于只做 prompt render/tokenization 相关预处理，不一定执行生成。

接口源码位置：

- `code/vllm/vllm/entrypoints/serve/render/api_router.py:35`
- `code/vllm/vllm/entrypoints/serve/render/api_router.py:61`
- `code/vllm/vllm/entrypoints/serve/render/api_router.py:84`
- `code/vllm/vllm/entrypoints/serve/render/api_router.py:109`

服务对象：

```text
OpenAIServingRender
```

初始化位置：`code/vllm/vllm/entrypoints/openai/api_server.py:363-377`。

## 9. Dev API

只有 `VLLM_SERVER_DEV_MODE` 开启时才注册。

注册函数：

```python
register_vllm_dev_api_routers(app)
```

源码位置：`code/vllm/vllm/entrypoints/serve/__init__.py:35-61`。

源码中明确打印安全警告：开发 endpoints 不应在生产使用：`code/vllm/vllm/entrypoints/serve/__init__.py:36-39`。

包含：

- cache reset
- RLHF pause/resume/weight update
- collective RPC
- server info
- sleep/wake_up

## 10. Router 的通用模式

几乎所有 router 都遵循同样模式：

```text
FastAPI route
  -> validate_json_request / with_cancellation / load_aware_call
  -> 从 request.app.state 取 serving handler
  -> handler.create_xxx(...)
  -> ErrorResponse? JSONResponse
  -> 正常 Response? JSONResponse
  -> AsyncGenerator? StreamingResponse
```

以 Chat Completions 为例：

```text
/v1/chat/completions
  -> create_chat_completion
  -> state.openai_serving_chat
  -> OpenAIServingChat.create_chat_completion
  -> ErrorResponse / ChatCompletionResponse / AsyncGenerator
```

## 11. 关键结论

API 路由层很薄，主要负责协议入口和响应类型选择；真正的业务逻辑集中在 serving 对象中。serving 对象再通过 `EngineClient` 把请求交给 engine。