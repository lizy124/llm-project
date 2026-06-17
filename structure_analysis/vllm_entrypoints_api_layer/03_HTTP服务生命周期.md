# 03 HTTP 服务生命周期

## 1. 核心文件

OpenAI-compatible HTTP 服务的核心文件是：

```text
vllm/entrypoints/openai/api_server.py
```

它是 API server 生命周期的中枢，负责：

- 解析和校验 API server 参数
- 创建监听 socket
- 创建 `EngineClient`
- 构建 FastAPI app
- 注册 routers
- 初始化 app.state 中的 serving 对象
- 启动 uvicorn HTTP 服务
- 处理异常、中间件、鉴权、CORS、request id、metrics 等

## 2. 单 API server 启动链路

最常见的 `vllm serve <model>` 最终会进入：

```python
run_server(args)
```

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:645-658`。

流程：

```text
run_server(args)
  -> decorate_logs("APIServer")
  -> setup_server(args)
  -> run_server_worker(listen_address, sock, args)
```

`run_server_worker()` 再进入：

```text
build_async_engine_client(args)
  -> build_and_serve(engine_client, listen_address, sock, args)
```

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:661-683`。

## 3. setup_server：服务启动前准备

`setup_server(args)` 负责在模型加载前完成服务端准备。

主要步骤：

1. 打印版本和模型信息。
2. 打印非默认参数。
3. 加载 tool parser plugin。
4. 加载 reasoning parser plugin。
5. 校验 API server 参数。
6. 创建 TCP 或 Unix domain socket。
7. 设置 ulimit。
8. 返回 `listen_address, sock`。

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:514-549`。

为什么先 bind socket 再初始化 engine？源码注释说明这是为了避免和 Ray 相关的 race condition：`code/vllm/vllm/entrypoints/openai/api_server.py:529-531`。

## 4. EngineClient 创建

API server 不直接使用具体 engine，而是通过 `EngineClient` 协议。

### 4.1 build_async_engine_client

`build_async_engine_client(args)`：

1. 处理 `forkserver` 多进程启动方式。
2. 从 CLI args 创建 `AsyncEngineArgs`。
3. 调用 `build_async_engine_client_from_engine_args()`。

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:76-104`。

### 4.2 build_async_engine_client_from_engine_args

`build_async_engine_client_from_engine_args(engine_args)`：

1. `engine_args.create_engine_config()` 创建 `VllmConfig`。
2. import V1 `AsyncLLM`。
3. 调用 `AsyncLLM.from_vllm_config()`。
4. 清理多模态 cache。
5. yield `async_llm`。
6. finally 中 shutdown engine。

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:107-153`。

关键点：在线 HTTP 服务当前主要落到 V1 `AsyncLLM`：

```python
from vllm.v1.engine.async_llm import AsyncLLM
async_llm = AsyncLLM.from_vllm_config(...)
```

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:123-145`。

## 5. build_and_serve：构建 app 并启动 HTTP

`build_and_serve()`：

1. 获取 uvicorn 日志配置。
2. 调用 `engine_client.get_supported_tasks()`。
3. 获取 `model_config`。
4. 调用 `build_app()` 创建 FastAPI app。
5. 调用 `init_app_state()` 初始化服务对象。
6. 调用 `serve_http()` 启动 HTTP server。

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:552-597`。

链路：

```text
build_and_serve
  -> supported_tasks = await engine_client.get_supported_tasks()
  -> app = build_app(args, supported_tasks, model_config)
  -> await init_app_state(engine_client, app.state, args, supported_tasks)
  -> serve_http(app, sock=..., host=..., port=...)
```

## 6. build_app：FastAPI app 构建与路由注册

`build_app()` 根据参数创建 FastAPI app：

- `--disable-fastapi-docs`：关闭 OpenAPI/Docs。
- `--enable-offline-docs`：启用离线 docs。
- 默认：正常 FastAPI。

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:156-179`。

然后注册不同 router：

### 6.1 通用 vLLM serve router

```python
register_vllm_serve_api_routers(app)
```

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:181-183`。

它包含：

- basic/health/metrics/offline docs
- LoRA 管理
- profile
- tokenize

注册逻辑在：`code/vllm/vllm/entrypoints/serve/__init__.py:11-32`。

### 6.2 models router

```python
/v1/models
```

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:185-189`。

### 6.3 SageMaker router

用于 SageMaker 标准接口：`/ping`、`/invocations` 等。

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:191-195`。

### 6.4 generate 任务 router

如果模型支持 `generate`，注册：

- `/v1/chat/completions`
- `/v1/chat/completions/batch`
- `/v1/responses`
- `/v1/completions`
- Anthropic messages
- generative scoring
- disaggregated serve
- elastic EP

入口：`code/vllm/vllm/entrypoints/openai/api_server.py:202-219`。

具体 generate router 注册：`code/vllm/vllm/entrypoints/generate/api_router.py:19-47`。

### 6.5 render router

如果支持 `generate` 或 `render`，注册 render API：

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:221-226`。

### 6.6 speech to text router

如果支持 `transcription` 或 `realtime`，注册语音相关 router：

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:228-233`。

### 6.7 pooling router

如果支持 pooling tasks，注册：

- embedding
- classify
- score/rerank
- pooling

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:235-238`。

## 7. 中间件与异常处理

`build_app()` 会配置：

- CORS middleware
- HTTPException handler
- RequestValidationError handler
- EngineGenerateError / EngineDeadError handler
- GenerationError handler
- VLLMValidationError handler
- 全局 Exception handler
- API key 鉴权 middleware
- X-Request-Id middleware
- ScalingMiddleware
- WebSocket metrics middleware
- 自定义 ASGI middleware
- SageMaker bootstrap

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:240-300`。

## 8. init_app_state：把 engine 和 serving 对象放进 app.state

`init_app_state()` 是 API server 的另一个核心函数。

它会初始化：

- `state.engine_client`
- `state.vllm_config`
- `state.args`
- `state.openai_serving_models`
- `state.openai_serving_render`
- `state.openai_serving_tokenization`
- generate state
- speech-to-text state
- pooling state
- server load metrics

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:303-410`。

其中 `OpenAIServingModels` 用于模型列表、base model、LoRA model registry：`code/vllm/vllm/entrypoints/openai/api_server.py:356-361`。

`OpenAIServingRender` 用于 chat/completion/tokenization/render 的预处理：`code/vllm/vllm/entrypoints/openai/api_server.py:363-377`。

`OpenAIServingTokenization` 用于 tokenize/detokenize：`code/vllm/vllm/entrypoints/openai/api_server.py:379-388`。

## 9. generate state 初始化

如果支持 `generate`，`init_app_state()` 调用：

```python
init_generate_state(engine_client, state, args, request_logger, supported_tasks)
```

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:390-395`。

`init_generate_state()` 会创建：

- `state.openai_serving_responses`
- `state.openai_serving_chat`
- `state.openai_serving_chat_batch`
- `state.openai_serving_completion`
- `state.anthropic_serving_messages`
- `state.serving_tokens`
- `state.serving_generative_scoring`

源码位置：`code/vllm/vllm/entrypoints/generate/api_router.py:49-200`。

## 10. HTTP 服务生命周期总结

```text
vllm serve
  -> openai.api_server.run_server
  -> setup_server
      -> validate args
      -> bind socket
      -> set ulimit
  -> run_server_worker
      -> build_async_engine_client
          -> AsyncEngineArgs.from_cli_args
          -> create VllmConfig
          -> AsyncLLM.from_vllm_config
      -> build_and_serve
          -> engine_client.get_supported_tasks
          -> build_app
              -> register routers
              -> add middleware
              -> add exception handlers
          -> init_app_state
              -> create serving objects
          -> serve_http / uvicorn
```

## 11. 关键结论

API server 的生命周期可以理解为：

1. 先准备网络监听。
2. 再创建 engine client。
3. 再根据 engine 支持的任务动态注册接口。
4. 再把各种 serving handler 放到 `app.state`。
5. 请求到来时 router 从 `app.state` 取 handler，并调用 handler 处理请求。