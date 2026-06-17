# vLLM 服务入口与 API 层技术点问答

本文基于本目录已有的入口层分析文档，整理适合考察的技术点问题与参考回答。重点覆盖 Python API、CLI、HTTP 服务生命周期、路由与 serving 对象、Chat/Completion 请求链路、离线 LLM API、Engine 边界、接口清单与常见设计取舍。

## 1. 总体架构与入口层定位

### Q1：vLLM 的“服务入口与 API 层”主要解决什么问题？

答：它主要解决四类问题：

1. 用户从哪里进入 vLLM：包括 Python API、CLI、HTTP OpenAI-compatible API、gRPC。
2. 外部协议如何转成 vLLM 内部请求：通过 argparse、FastAPI route、Pydantic request model、chat template、tokenizer、多模态处理、SamplingParams/PoolingParams 等完成转换。
3. API 层如何连接 engine：在线服务通过 `EngineClient` 协议连接当前 V1 `AsyncLLM`。
4. engine 输出如何返回给外部协议：非流式收集最终 `RequestOutput` 转 JSON，流式则把增量输出转成 SSE `data: ...`。

核心结论是：入口层是协议适配层，不负责调度、KV Cache、模型执行。

### Q2：vLLM 有哪些外部入口？

答：主要有四类：

1. Python 包级 API：`from vllm import LLM`。
2. CLI：`vllm serve`、`vllm chat`、`vllm complete` 等。
3. HTTP：OpenAI-compatible REST API，比如 `/v1/chat/completions`、`/v1/completions`。
4. gRPC：通过 `vllm serve --grpc` 启动。

### Q3：入口层在 vLLM 整体框架中处于什么位置？

答：入口层位于外部用户协议和 engine 之间。它接收 HTTP、CLI、Python 或 gRPC 请求，完成参数解析、协议校验、输入渲染、tokenization、多模态处理、LoRA 选择、tool/reasoning parser、SamplingParams/PoolingParams 构造，然后调用 `EngineClient.generate()` 或 `EngineClient.encode()`。请求进入 EngineCore 后，调度、KV Cache、worker 执行等就属于 engine 内部。

### Q4：入口层不负责哪些事情？

答：入口层不负责：

- scheduler 调度算法；
- KV Cache block 分配；
- GPU model runner 执行；
- attention kernel；
- worker/executor 细节；
- EngineCore 内部请求调度和执行。

这些由 V1 engine、core、worker、executor、model runner 和底层 kernel 完成。

### Q5：为什么说 vLLM 的 API 层是“协议适配层”？

答：因为 Chat Completions、Completions、Responses、Anthropic Messages、Embedding、Pooling、Tokenize、Render 等外部接口虽然协议不同，但最终都会被转换为 engine 能理解的统一输入形式，例如：

- `EngineInput`；
- `SamplingParams` 或 `PoolingParams`；
- request id；
- LoRA、trace、priority、data parallel rank 等 metadata。

API 层只做转换和响应包装，不直接做模型调度和执行。

## 2. Python 包级公开 API

### Q6：`vllm/__init__.py` 的作用是什么？

答：`vllm/__init__.py` 是 vLLM Python 包的门面。它通过 `MODULE_ATTRS` 懒加载公开对象，并通过 `__all__` 明确声明包级别可导出的 API。用户可以直接写：

```python
from vllm import LLM, SamplingParams, RequestOutput
```

而不需要关心这些类真实位于哪个子模块。

### Q7：vLLM 包级别通常导出哪些重要对象？

答：主要包括：

- `LLM`：离线推理高级 API；
- `LLMEngine`：同步 engine 兼容导出；
- `AsyncLLMEngine`：异步 engine 兼容导出；
- `EngineArgs`、`AsyncEngineArgs`：engine 参数对象；
- `SamplingParams`：生成采样参数；
- `PoolingParams`：pooling 参数；
- `RequestOutput`、`PoolingRequestOutput`：生成和 pooling 输出对象。

### Q8：为什么包级 API 要做懒加载？

答：懒加载可以避免 import `vllm` 时立刻加载大量依赖，降低初始化开销，也能减少平台差异、可选依赖或重型模块在 import 阶段引起的问题。只有用户真正访问 `LLM`、`SamplingParams` 等对象时，才导入对应模块。

## 3. CLI 入口与命令注册

### Q9：`vllm` 命令是如何注册的？

答：在 `pyproject.toml` 中通过 project scripts 注册：

```toml
[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"
```

因此执行 `vllm ...` 实际会进入 `vllm.entrypoints.cli.main:main()`。

### Q10：CLI 主入口 `main()` 的职责是什么？

答：CLI 主入口主要做：

1. 延迟 import 子命令模块；
2. 执行 CLI 环境初始化；
3. 创建 `FlexibleArgumentParser`；
4. 注册子命令；
5. 解析参数；
6. 调用子命令的 `validate()`；
7. 调用子命令的 `cmd()` 执行实际逻辑。

### Q11：为什么 CLI 主入口要延迟 import 子命令模块？

答：延迟 import 可以避免 eager import 导致平台相关问题或依赖问题。vLLM 子命令很多，有些可能依赖特定后端、分布式组件或额外库。只有实际使用对应子命令时才加载，可以降低启动风险和开销。

### Q12：vLLM CLI 有哪些主要子命令？

答：主要包括：

| 命令 | 作用 |
|---|---|
| `vllm serve` | 启动本地 OpenAI-compatible、gRPC、headless 或多 API server 服务 |
| `vllm chat` | 作为客户端连接运行中的 OpenAI-compatible server，发送 chat 请求 |
| `vllm complete` | 作为客户端连接运行中的 OpenAI-compatible server，发送 completion 请求 |
| `vllm bench` | benchmark |
| `vllm collect-env` | 采集环境信息 |
| `vllm run-batch` | 运行 batch 请求 |

### Q13：所有 CLI 子命令是如何统一注册的？

答：每个子命令模块实现 `cmd_init()`，返回 `CLISubcommand` 对象。CLI 主入口加载子命令模块后，调用这些对象的注册逻辑，把它们挂到统一 parser 下。

## 4. `vllm serve` 启动模式

### Q14：`vllm serve` 对应哪个类？

答：对应 `vllm.entrypoints.cli.serve.ServeSubcommand`，其 `name = "serve"`，因此用户执行 `vllm serve ...` 时会进入这个子命令。

### Q15：`vllm serve` 的参数是在哪里添加的？

答：`ServeSubcommand.subparser_init()` 会创建 `serve` 子 parser，然后调用 `vllm.entrypoints.openai.cli_args.make_arg_parser()` 注入 OpenAI server 和 engine 参数。包括：

- `model_tag`；
- `--headless`；
- `--api-server-count`；
- `--config`；
- `--grpc`；
- FrontendArgs 参数；
- AsyncEngineArgs 参数。

### Q16：`vllm serve` 会做哪些参数校验？

答：`ServeSubcommand.validate()` 调用 `validate_parsed_serve_args(args)`，主要校验：

- chat template 是否有效；
- `--enable-auto-tool-choice` 必须配合 `--tool-call-parser`；
- `--enable-log-outputs` 必须配合 `--enable-log-requests`；
- data parallel multi-port external LB 参数是否合法。

### Q17：`vllm serve Qwen/Qwen3-0.6B` 中的位置参数模型名如何处理？

答：如果 CLI 中指定了 positional `model_tag`，`ServeSubcommand.cmd()` 会把它写入 `args.model`。也就是说：

```text
vllm serve Qwen/Qwen3-0.6B
```

最终等价于把模型名传给 engine 参数中的 `model`。

### Q18：`vllm serve` 有哪些启动分支？

答：主要有：

1. gRPC 模式：指定 `--grpc`，进入 `serve_grpc(args)`。
2. headless 模式：指定 `--headless`，只启动 engine/worker，不启动 API server。
3. 多 API server 模式：`api_server_count > 1` 或启用 Rust frontend 时进入 `run_multi_api_server(args)`。
4. 单 API server 模式：默认最常见路径，进入 OpenAI-compatible HTTP server 的 `run_server(args)`。

### Q19：`vllm serve --grpc` 做什么？

答：它不启动 OpenAI-compatible HTTP server，而是调用 `vllm.entrypoints.grpc_server.serve_grpc(args)`。gRPC server 会创建 `AsyncEngineArgs`、`VllmConfig`、V1 `AsyncLLM`，再创建 `VllmEngineServicer`，注册 health service 和 reflection，最后监听 host/port。

### Q20：`--headless` 模式适合什么场景？

答：`--headless` 表示不启动 API server，只启动 engine/worker。它适合多节点、数据并行或外部负载均衡场景，由其他 API server 或 frontend 连接这些 engine 进程。

### Q21：多 API server 模式为什么要先 bind HTTP socket？

答：多 API server 模式需要多个 API server 子进程共享或协调监听资源。先 bind socket 可以提前确定监听地址，并避免后续 engine 初始化或分布式组件启动后才发现端口冲突。同时单 API server 的 `setup_server()` 也会先 bind socket，源码注释提到这有助于避免 Ray 相关 race condition。

### Q22：`vllm serve` 是否直接创建模型执行器？

答：不是。`vllm serve` 先完成服务形态选择，例如 HTTP、gRPC、headless、多 API server、Rust frontend、DP supervisor。真正的 engine 创建发生在后续的 `openai.api_server.build_async_engine_client()` 或 `grpc_server.serve_grpc()` 中。

## 5. `vllm chat` 与 `vllm complete`

### Q23：`vllm chat` 和 `vllm complete` 是 server 还是 client？

答：它们是客户端命令，不启动模型服务。它们连接已经运行的 OpenAI-compatible server，默认 URL 是 `http://localhost:8000/v1`。

### Q24：`vllm chat` 如何发送请求？

答：`vllm chat` 会创建 OpenAI Python client，然后调用：

```python
client.chat.completions.create(..., stream=True)
```

它以流式方式消费 server 返回的 chat completions。

### Q25：`vllm complete` 如何发送请求？

答：`vllm complete` 也创建 OpenAI Python client，然后调用：

```python
client.completions.create(..., stream=True)
```

它面向 `/v1/completions` 接口。

### Q26：如果 `vllm chat` 未指定 model，它如何确定模型？

答：它会调用 OpenAI-compatible server 的 `/v1/models`，获取模型列表并选择第一个模型作为默认模型。

## 6. HTTP API Server 生命周期

### Q27：OpenAI-compatible HTTP 服务的核心文件是什么？

答：核心文件是：

```text
vllm/entrypoints/openai/api_server.py
```

它负责参数校验、socket 创建、EngineClient 创建、FastAPI app 构建、router 注册、app.state 初始化、uvicorn 启动、中间件、异常处理、鉴权、CORS、request id、metrics 等。

### Q28：单 API server 的启动链路是什么？

答：典型链路是：

```text
vllm serve
  -> openai.api_server.run_server(args)
  -> setup_server(args)
  -> run_server_worker(listen_address, sock, args)
  -> build_async_engine_client(args)
  -> build_and_serve(engine_client, listen_address, sock, args)
```

### Q29：`setup_server(args)` 做哪些事情？

答：它在模型加载前完成服务端准备：

1. 打印版本和模型信息；
2. 打印非默认参数；
3. 加载 tool parser plugin；
4. 加载 reasoning parser plugin；
5. 校验 API server 参数；
6. 创建 TCP 或 Unix domain socket；
7. 设置 ulimit；
8. 返回 `listen_address, sock`。

### Q30：为什么 HTTP 服务要先 bind socket 再初始化 engine？

答：主要是为了避免服务启动过程中和 Ray 等分布式组件产生 race condition，也可以尽早发现端口占用等网络监听问题，避免模型和 engine 已经加载后才失败。

### Q31：`build_async_engine_client(args)` 的作用是什么？

答：它负责把 CLI args 转成在线 engine client：

1. 处理 `forkserver` 多进程启动方式；
2. 从 CLI args 创建 `AsyncEngineArgs`；
3. 调用 `build_async_engine_client_from_engine_args()`；
4. 创建 V1 `AsyncLLM` 并作为 `EngineClient` 使用。

### Q32：HTTP API server 当前主要使用哪个 engine client？

答：当前主要使用 V1 `AsyncLLM`：

```python
from vllm.v1.engine.async_llm import AsyncLLM
async_llm = AsyncLLM.from_vllm_config(...)
```

它实现了 `EngineClient` 协议。

### Q33：`build_and_serve()` 做什么？

答：它负责构建 app 并启动 HTTP 服务：

1. 获取 uvicorn 日志配置；
2. 调用 `engine_client.get_supported_tasks()`；
3. 获取 `model_config`；
4. 调用 `build_app()` 创建 FastAPI app；
5. 调用 `init_app_state()` 初始化 serving 对象；
6. 调用 `serve_http()` 启动 HTTP server。

### Q34：为什么 router 注册要依赖 `supported_tasks`？

答：不同模型支持的任务不同，例如 generate、render、embedding、classify、score、pooling、transcription、realtime 等。API server 会根据 `engine_client.get_supported_tasks()` 动态注册对应接口，避免模型不支持的接口被错误暴露。

### Q35：`build_app()` 如何创建 FastAPI app？

答：它根据参数决定 FastAPI docs 的启用方式：

- `--disable-fastapi-docs`：关闭 OpenAPI/Docs；
- `--enable-offline-docs`：启用离线 docs；
- 默认：正常 FastAPI docs。

然后注册通用 serve router、models router、SageMaker router、generate router、render router、speech-to-text router、pooling router，并配置中间件和异常处理。

### Q36：`build_app()` 中常见中间件和异常处理有哪些？

答：包括：

- CORS middleware；
- HTTPException handler；
- RequestValidationError handler；
- EngineGenerateError / EngineDeadError handler；
- GenerationError handler；
- VLLMValidationError handler；
- 全局 Exception handler；
- API key 鉴权 middleware；
- X-Request-Id middleware；
- ScalingMiddleware；
- WebSocket metrics middleware；
- 自定义 ASGI middleware；
- SageMaker bootstrap。

### Q37：`init_app_state()` 的核心作用是什么？

答：它把 engine client、配置和各种 serving handler 放入 `app.state`，让 router 在请求到来时可以从 `request.app.state` 中取出对应 handler 处理请求。

### Q38：`app.state` 中通常会保存哪些对象？

答：包括：

- `state.engine_client`；
- `state.vllm_config`；
- `state.args`；
- `state.openai_serving_models`；
- `state.openai_serving_render`；
- `state.openai_serving_tokenization`；
- generate 相关 serving 对象；
- speech-to-text 相关对象；
- pooling 相关对象；
- server load metrics。

### Q39：generate state 初始化会创建哪些对象？

答：如果模型支持 `generate`，`init_generate_state()` 会创建：

- `state.openai_serving_responses`；
- `state.openai_serving_chat`；
- `state.openai_serving_chat_batch`；
- `state.openai_serving_completion`；
- `state.anthropic_serving_messages`；
- `state.serving_tokens`；
- `state.serving_generative_scoring`。

## 7. Router、Serving、Protocol 分层

### Q40：vLLM API 层中 Router、Serving、Protocol 分别负责什么？

答：

| 类型 | 典型文件 | 职责 |
|---|---|---|
| Router | `api_router.py` | 声明 HTTP path、method、依赖、返回类型，从 `app.state` 取 handler |
| Serving | `serving.py` | 真正处理请求：校验、渲染、参数转换、调用 engine、组装响应 |
| Protocol | `protocol.py` | 定义 Pydantic request/response model，描述 OpenAI/Anthropic/vLLM 协议结构 |

### Q41：为什么 router 层设计得很薄？

答：router 主要处理 HTTP 协议入口和响应类型选择。真正业务逻辑集中在 serving 对象中。这样可以让路由定义、协议模型、请求处理逻辑解耦，也便于不同接口复用 serving 能力。

### Q42：Router 的通用处理模式是什么？

答：大多数 router 遵循：

```text
FastAPI route
  -> validate_json_request / with_cancellation / load_aware_call
  -> 从 request.app.state 取 serving handler
  -> handler.create_xxx(...)
  -> ErrorResponse? JSONResponse
  -> 正常 Response? JSONResponse
  -> AsyncGenerator? StreamingResponse
```

### Q43：Serving 对象通常持有哪些依赖？

答：不同接口略有不同，但通常包括：

- `engine_client`；
- `OpenAIServingModels`；
- `OpenAIServingRender`；
- `RequestLogger`；
- 默认 sampling params 或 pooling params；
- chat template；
- tool parser；
- reasoning parser；
- tokenizer/render/input processor 相关能力。

## 8. OpenAI-compatible API

### Q44：Chat Completions 的 HTTP 路径是什么？

答：

```http
POST /v1/chat/completions
```

对应 router 是 `openai/chat_completion/api_router.py`，handler 是 `OpenAIServingChat.create_chat_completion()`。

### Q45：Chat Completions router 收到请求后做什么？

答：它会：

1. 获取 endpoint load metrics header；
2. 从 `request.app.state.openai_serving_chat` 获取 handler；
3. 如果 handler 不存在，说明模型不支持该接口；
4. 调用 `handler.create_chat_completion(request, raw_request)`；
5. 根据返回类型选择 JSON error、普通 JSON 或 `StreamingResponse(text/event-stream)`。

### Q46：Completions 的 HTTP 路径是什么？

答：

```http
POST /v1/completions
```

对应 router 是 `openai/completion/api_router.py`，handler 是 `OpenAIServingCompletion.create_completion()`。

### Q47：Responses API 包含哪些接口？

答：主要包括：

```http
POST /v1/responses
GET  /v1/responses/{response_id}
POST /v1/responses/{response_id}/cancel
```

对应服务对象是 `OpenAIServingResponses`，更接近 OpenAI 新版 Responses API，流式输出会转换成 SSE event。

### Q48：Models API 的作用是什么？

答：`GET /v1/models` 用于返回当前 server 可用模型列表。对应 handler 是 `OpenAIServingModels.show_available_models()`。它也会管理 base model 和 LoRA model registry。

### Q49：Batch Chat Completions 是什么？

答：路径是：

```http
POST /v1/chat/completions/batch
```

它由 `state.openai_serving_chat_batch` 处理，适合 batch 形式的 chat completion 请求。

## 9. Anthropic-compatible API

### Q50：vLLM 支持哪些 Anthropic-compatible API？

答：主要包括：

```http
POST /v1/messages
POST /v1/messages/count_tokens
```

服务对象是 `AnthropicServingMessages`，在 generate state 中初始化。

### Q51：Anthropic Messages 和 OpenAI Chat Completions 的共同点是什么？

答：它们都是外部协议层。虽然请求/响应格式不同，但最终都会经过 serving 对象转换成 engine 输入，并调用 `EngineClient.generate()` 进入统一生成链路。

## 10. Pooling、Embedding、Classification、Scoring API

### Q52：Pooling 相关 API 是如何注册的？

答：由 `register_pooling_api_routers(app, supported_tasks, model_config)` 统一注册。它根据模型支持的任务动态注册 embedding、classify、score/rerank、generic pooling 等 router。

### Q53：Pooling 相关接口有哪些？

答：常见包括：

| 类型 | 典型路径 | Serving 对象 |
|---|---|---|
| embedding | `/v1/embeddings`、`/embed` | `ServingEmbedding` |
| classify | `/classify` | `ServingClassification` |
| score/rerank | `/score`、`/rerank` 等 | `ServingScores` |
| generic pooling | `/pooling` | `ServingPooling` |

### Q54：Embedding/Pooling 类接口最终调用 engine 的哪个方法？

答：它们最终调用 `EngineClient.encode()`，并传入 `PoolingParams`，返回 `PoolingRequestOutput`。

### Q55：`generate()` 和 `encode()` 的区别是什么？

答：

- `generate()` 面向文本生成类任务，如 Chat、Completion、Responses、Anthropic Messages，输入采样参数 `SamplingParams`，输出 `RequestOutput`。
- `encode()` 面向 embedding/pooling/classify/score 等非生成任务，输入 `PoolingParams`，输出 `PoolingRequestOutput`。

## 11. vLLM serve 管理接口

### Q56：通用 serve router 包含哪些管理能力？

答：由 `register_vllm_serve_api_routers(app)` 注册，包含：

- basic endpoints；
- health endpoints；
- metrics endpoints；
- offline docs endpoints；
- LoRA 管理；
- profile；
- tokenize/detokenize；
- render；
- disaggregated/tokens；
- elastic EP 等。

### Q57：LoRA 管理接口有哪些？

答：

```http
POST /v1/load_lora_adapter
POST /v1/unload_lora_adapter
```

用于动态加载和卸载 LoRA adapter。

### Q58：Profile 接口有哪些？

答：

```http
POST /start_profile
POST /stop_profile
```

用于启动和停止 profiling。

### Q59：Tokenize/Detokenize 接口有哪些？

答：

```http
POST /tokenize
POST /detokenize
GET  /tokenizer_info
```

服务对象是 `OpenAIServingTokenization`。

### Q60：Render API 的作用是什么？

答：Render API 用于只做 prompt render/tokenization 相关预处理，不一定执行生成。它可以把 OpenAI/chat/completion 风格请求渲染为 prompt 或 token 结构，服务对象是 `OpenAIServingRender`。

### Q61：Disaggregated / Tokens API 包含什么？

答：主要包括：

```http
POST /v1/completions/tokens
POST /abort_requests
```

其中 `/v1/completions/tokens` 是 tokens in/out 形式的生成接口，`/abort_requests` 用于中止请求。

### Q62：Elastic EP API 的作用是什么？

答：Elastic EP 面向弹性 expert parallel 扩缩，主要接口：

```http
POST /scale_elastic_ep
POST /is_scaling_elastic_ep
```

分别用于发起扩缩和查询是否正在扩缩。

### Q63：Dev Mode API 有什么特点？

答：Dev Mode API 只有 `VLLM_SERVER_DEV_MODE` 开启时才注册。源码中明确提示开发 endpoints 不应在生产环境使用。它包括 cache reset、sleep/wake_up、RLHF pause/resume/weight update、collective RPC、server info 等接口。

### Q64：Dev Mode 下的 cache reset 接口有哪些？

答：

```http
POST /reset_prefix_cache
POST /reset_mm_cache
POST /reset_encoder_cache
```

它们最终会调用 engine client 的对应 cache reset 控制方法。

### Q65：Dev Mode 下 sleep/wake_up 接口有哪些？

答：

```http
POST /sleep
POST /wake_up
GET  /is_sleeping
```

用于控制 engine 睡眠、唤醒和查询睡眠状态。

### Q66：SageMaker API 有哪些接口？

答：主要包括：

```http
GET/POST /ping
POST     /invocations
```

`/ping` 用于 SageMaker health check，`/invocations` 用于 SageMaker invocation endpoint。

## 12. Chat Completions 请求链路

### Q67：Chat Completions 的完整请求链路是什么？

答：核心链路如下：

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

### Q68：`OpenAIServingChat` 初始化时持有哪些关键依赖？

答：包括：

- `engine_client`；
- `OpenAIServingModels`；
- `OpenAIServingRender`；
- `RequestLogger`；
- chat template；
- tool parser；
- reasoning parser；
- 默认 sampling params。

### Q69：`render_chat_request()` 做什么？

答：它主要做：

1. `_check_model(request)`：确认请求中的 model 是否可用；
2. 检查 engine 是否已经 dead；
3. 调用 `openai_serving_render.render_chat(request)` 做真正渲染；
4. 返回 `(conversation, engine_inputs)`。

其中 `engine_inputs` 是 engine 可消费的输入。

### Q70：Chat 请求如何转换采样参数？

答：在 `_create_chat_completion()` 中，会根据请求计算 max_tokens、prompt token ids、多模态 token 数等，然后调用：

```python
sampling_params = request.to_sampling_params(max_tokens, self.default_sampling_params)
```

如果是 beam search，则调用：

```python
sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
```

### Q71：Chat Completions 在哪里进入 engine？

答：在 `OpenAIServingChat._create_chat_completion()` 中调用：

```python
generator = self.engine_client.generate(
    engine_input,
    sampling_params,
    sub_request_id,
    ...
)
```

这是 API 层和 engine 层的关键边界。

### Q72：Chat Completions 流式响应如何生成？

答：如果 `request.stream == True`，返回 `chat_completion_stream_generator()`。它会遍历 engine 的 `RequestOutput`，为每个增量输出生成 delta，处理 reasoning/tool calls parser，组装 `ChatCompletionStreamResponse`，然后 yield：

```text
data: {json}\n\n
data: [DONE]\n\n
```

### Q73：Chat Completions 非流式响应如何生成？

答：非流式时使用 `chat_completion_full_generator()`，它会完整消费 `result_generator`，获取最终 `RequestOutput`，遍历 choices，解析 tool calls/reasoning，组装 `ChatCompletionResponse`。

### Q74：Chat Completions 中 tool calls 和 reasoning 是在哪层处理的？

答：主要在 serving 响应生成阶段处理。API 层会加载 tool parser 和 reasoning parser plugin，并在流式或非流式响应组装时解析模型输出中的 tool calls/reasoning 内容，再包装成 OpenAI-compatible response。

## 13. Completions 请求链路

### Q75：Completions 的完整请求链路是什么？

答：核心链路如下：

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

### Q76：`OpenAIServingCompletion` 持有哪些关键依赖？

答：包括：

- `engine_client`；
- `OpenAIServingModels`；
- `OpenAIServingRender`；
- `RequestLogger`；
- 默认 sampling params；
- max token override。

### Q77：`render_completion_request()` 做什么？

答：它会：

1. `_check_model(request)`；
2. 检查 engine 是否 dead；
3. 调用 `openai_serving_render.render_completion(request)`；
4. 返回一个或多个 `engine_inputs`。

### Q78：Completions 如何处理多个 prompt？

答：`_create_completion()` 会为每个 prompt 创建独立 request id，分别调用 `engine_client.generate()` 得到多个异步生成器，然后通过 `merge_async_iterators()` 合并多个异步生成器的输出。

### Q79：Completions 流式响应如何输出？

答：如果 `request.stream == True`，返回 `completion_stream_generator()`。它逐步把 engine 输出转换成 `CompletionStreamResponse`，并通过 SSE 返回：

```text
data: {...}\n\n
data: [DONE]\n\n
```

### Q80：Completions 非流式响应如何输出？

答：非流式时会收集每个 prompt 的最终结果，然后调用 `request_output_to_completion_response(...)` 组装 `CompletionResponse`。

### Q81：Chat Completions 和 Completions 的主要差异是什么？

答：

| 维度 | Chat Completions | Completions |
|---|---|---|
| 输入协议 | messages 数组 | prompt 字符串或 token |
| 预处理 | chat template/render | completion render |
| tool calls | 支持，依赖 parser | 不是主路径 |
| reasoning | 支持 reasoning parser | 较少涉及 |
| 多 prompt | 通常单 conversation | 可以多个 prompt，并 merge async iterators |
| 流式输出 | delta message | delta text |
| response object | `ChatCompletionResponse` | `CompletionResponse` |

### Q82：Chat 和 Completion 的共同核心是什么？

答：两者最终都会收敛到：

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

区别主要在请求渲染和响应包装，而不是 engine 调用。

## 14. SSE 与流式响应

### Q83：vLLM 为什么使用 SSE 做流式输出？

答：OpenAI-compatible Chat/Completion 流式接口通常使用 Server-Sent Events。vLLM 将 engine 的增量 `RequestOutput` 包装为 SSE chunk，格式为：

```text
data: {json}\n\n
```

结束时发送：

```text
data: [DONE]\n\n
```

这样可以兼容 OpenAI 客户端生态。

### Q84：流式和非流式在 engine 调用上有区别吗？

答：通常没有本质区别。两者都调用 `engine_client.generate()` 得到异步输出。区别在于 serving 层如何消费输出：

- 流式：每拿到增量输出就包装成 SSE chunk 返回；
- 非流式：完整消费生成器，取最终输出组装 JSON response。

### Q85：Responses API 的流式输出和 Chat/Completion 有什么不同？

答：Responses API 更接近 OpenAI 新版 Responses 事件流格式。它会把内部 stream 转换成 SSE event，而不仅仅是 Chat/Completion 的 chunk 格式。

## 15. EngineClient 边界

### Q86：什么是 `EngineClient`？

答：`EngineClient` 是在线 API server 依赖的抽象协议，定义 API 层和 engine 层之间的边界。API server 不直接依赖具体 engine 实现，而是通过 `EngineClient` 调用生成、编码和控制方法。

### Q87：`EngineClient` 有哪些关键属性？

答：包括：

- `vllm_config`；
- `model_config`；
- `renderer`；
- `input_processor`；
- `is_running`；
- `is_stopped`；
- `errored`；
- `dead_error`。

### Q88：`EngineClient.generate()` 的作用是什么？

答：它是文本生成类接口最终调用的核心方法。OpenAI Chat、OpenAI Completion、Responses、Anthropic Messages 等生成类请求，最终都会调用 `generate(prompt, sampling_params, request_id, ...)`，返回 `AsyncGenerator[RequestOutput, None]`。

### Q89：`EngineClient.encode()` 的作用是什么？

答：它是 embedding、pooling、classify、score 等非生成类接口最终调用的核心方法。它接收 prompt、`PoolingParams`、request id 等，返回 `AsyncGenerator[PoolingRequestOutput, None]`。

### Q90：`EngineClient` 还定义了哪些控制类方法？

答：包括：

- `abort()`；
- `check_health()`；
- `start_profile()` / `stop_profile()`；
- `reset_mm_cache()`；
- `reset_encoder_cache()`；
- `reset_prefix_cache()`；
- `sleep()` / `wake_up()` / `is_sleeping()`；
- `add_lora()`；
- `pause_generation()` / `resume_generation()` / `is_paused()`；
- `shutdown()`。

这些方法解释了为什么 profile、LoRA、cache reset、sleep 等 HTTP 管理接口可以统一调用 engine。

### Q91：API 层进入 engine 的“最后一站”是什么？

答：对于生成类请求，是 `EngineClient.generate()`；对于 pooling/embedding 类请求，是 `EngineClient.encode()`。更深入地说，请求进入 `AsyncLLM._add_request()` 并调用 `engine_core.add_request_async()` 后，就进入 EngineCore 内部，不再属于 API 层。

## 16. 在线服务的 AsyncLLM

### Q92：在线 HTTP 服务中 `EngineClient` 的具体实现是什么？

答：主要是 V1 `AsyncLLM`。它继承或实现 `EngineClient`，通过 `AsyncLLM.from_vllm_config(...)` 创建。

### Q93：`AsyncLLM.__init__()` 初始化了哪些关键组件？

答：主要包括：

1. `self.vllm_config`；
2. `self.model_config`；
3. tracing；
4. `self.renderer = renderer_from_config(...)`；
5. `self.input_processor = InputProcessor(...)`；
6. `self.output_processor = OutputProcessor(...)`；
7. `self.engine_core = EngineCoreClient.make_async_mp_client(...)`；
8. stats logger；
9. output handler task。

### Q94：`AsyncLLM.from_vllm_config()` 如何选择 executor？

答：它根据 `VllmConfig` 调用 `Executor.get_class(vllm_config)` 选择 executor。说明 API 层不直接决定使用什么 executor，executor 选择由 engine config 决定。

### Q95：`AsyncLLM.generate()` 的核心执行流程是什么？

答：流程是：

1. 创建与 request 对应的 async stream/collector；
2. 处理输入；
3. 添加 request 到 detokenizer/output processor；
4. 添加 request 到 EngineCore；
5. 后台 output handler 从 EngineCore 拉输出并放入 request queue；
6. `generate()` 迭代 queue，把 `RequestOutput` yield 给 API server。

### Q96：`AsyncLLM.add_request()` 在边界转换中做什么？

答：它是 API 输入进入 EngineCore 前的关键转换点，主要做：

1. 检查 engine 是否 errored；
2. 判断是否 pooling；
3. 校验 fast prefill 与 prompt logprobs 的兼容性；
4. 处理 streaming input；
5. 调用 `self.input_processor.process_inputs(...)`；
6. 设置 reasoning 状态；
7. 分配 request id；
8. 启动 output handler；
9. 创建 `RequestOutputCollector`；
10. 对 `n > 1` fan out 子请求；
11. 调用 `_add_request()`。

### Q97：`AsyncLLM._add_request()` 做什么？

答：它先把 request 加到 `OutputProcessor`，然后调用：

```python
await self.engine_core.add_request_async(request)
```

从这一步开始，请求进入 EngineCore，后续由 scheduler、executor、worker 等 engine 内部组件处理。

### Q98：为什么 `n > 1` 需要 fan out？

答：当请求要求生成多个候选输出时，engine 内部可以把一个用户请求拆成多个子请求，每个子请求独立生成，再由 output processor/collector 聚合回用户可见的结果。这就是 `n > 1` fan out 的意义。

## 17. 离线 LLM API

### Q99：离线推理的入口是什么？

答：通常是：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="...")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=32))
```

其中 `LLM` 来自 `vllm.entrypoints.llm.LLM`。

### Q100：`LLM` 类的定位是什么？

答：`LLM` 是离线推理高级封装，适合 offline inference/batch 场景。它封装 tokenizer、language model、GPU KV cache、batching mechanism、内存管理、请求入队、执行循环和输出收集。源码注释明确说明 online serving 应使用 async engine 类。

### Q101：当前 `LLM` 实际接入哪个 engine？

答：当前接入 V1 `LLMEngine`：

```python
from vllm.v1.engine.llm_engine import LLMEngine
```

离线 `LLM` 是 `LLMEngine` 的高级封装。

### Q102：`LLM.__init__()` 参数主要覆盖哪些方面？

答：包括：

- model/tokenizer；
- tokenizer mode；
- trust remote code；
- tensor parallel size；
- dtype/quantization；
- revision/tokenizer_revision；
- chat template；
- seed；
- GPU memory utilization；
- KV cache memory bytes；
- CPU offload；
- eager/CUDA graph；
- HF token/overrides；
- multimodal processor kwargs；
- pooling config；
- structured outputs config；
- profiler config；
- attention config；
- compilation config；
- speculative decoding alias 参数；
- 其他 EngineArgs 参数。

### Q103：离线 `LLM` 构造函数内部做了哪些关键归一化？

答：包括：

1. 如果未传 `disable_log_stats`，离线默认设置为 True；
2. 如果 `worker_cls` 是 Python class，用 `cloudpickle.dumps()` 序列化；
3. 如果 `kv_transfer_config` 是 dict，转成 `KVTransferConfig`；
4. 把 dict/None/实例统一转成 `CompilationConfig`、`StructuredOutputsConfig`、`ProfilerConfig`、`AttentionConfig` 等 config instance；
5. 限制离线单进程 `data_parallel_size > 1`，避免 hang；
6. 汇总参数构造 `EngineArgs`；
7. 调用 `LLMEngine.from_engine_args(...)` 创建 engine。

### Q104：`LLM.generate()` 的逻辑是什么？

答：它会：

1. 检查 `runner_type == "generate"`；
2. 如果未传 sampling params，使用模型默认 sampling params；
3. 调用 `_run_completion()`；
4. 由 `OfflineInferenceMixin` 执行请求添加、engine step 循环和结果收集。

### Q105：`LLM.chat()` 的逻辑是什么？

答：`LLM.chat()` 接收 OpenAI 风格 messages，通过 chat template 转成 prompt，然后调用 `_run_chat()`，底层仍然进入离线生成流程。它支持 tools、chat_template、add_generation_prompt、continue_final_message、tokenization/multimodal kwargs 等参数。

### Q106：`LLM.enqueue()` 和 `LLM.wait_for_completion()` 有什么作用？

答：它们允许用户把“请求入队”和“执行等待”拆开：

```python
request_ids = llm.enqueue(prompts, sampling_params)
outputs = llm.wait_for_completion()
```

`enqueue()` 只把请求加入 engine 队列，不开始处理；`wait_for_completion()` 调用 `_run_engine()` 等待所有已入队请求完成。

### Q107：离线 `LLM` 提供哪些控制类 API？

答：包括：

- `collective_rpc()`；
- `apply_model()`；
- `start_profile()`；
- `stop_profile()`；
- `reset_prefix_cache()`；
- `sleep()`；
- `wake_up()`；
- `get_metrics()`。

### Q108：离线 `LLM` 中权重更新 / RLHF 相关 API 有哪些？

答：包括：

- `init_weight_transfer_engine()`；
- `start_weight_update()`；
- `update_weights()`；
- `finish_weight_update()`。

这些方法底层通常通过 `self.llm_engine.collective_rpc(...)` 向 worker 广播控制命令。

### Q109：离线 `LLM` 和在线 OpenAI server 的区别是什么？

答：

| 维度 | 离线 `LLM` | 在线 OpenAI server |
|---|---|---|
| 入口 | Python class | HTTP/CLI |
| 主要文件 | `entrypoints/llm.py` | `entrypoints/openai/api_server.py` |
| Engine | `LLMEngine` | `AsyncLLM` as `EngineClient` |
| 请求方式 | 同步批量 | 异步请求/流式 |
| 输出 | `list[RequestOutput]` | JSON/SSE |
| 适用 | offline inference/batch | online serving |

### Q110：离线 LLM API 的调用链是什么？

答：

```text
from vllm import LLM
  -> vllm.__getattr__("LLM")
  -> vllm.entrypoints.llm.LLM.__init__
  -> EngineArgs(...)
  -> LLMEngine.from_engine_args(...)
  -> LLM.generate / LLM.chat
  -> OfflineInferenceMixin._run_completion / _run_chat
  -> LLMEngine.add_request
  -> LLMEngine.step
  -> RequestOutput
```

## 18. LLMEngine 离线同步 engine

### Q111：`LLMEngine` 的定位是什么？

答：`LLMEngine` 是 V1 中用于离线同步执行的 engine 封装，源码描述为 legacy LLMEngine for backwards compatibility。离线 `LLM` 通过它完成同步请求入队、step 执行和结果收集。

### Q112：`LLMEngine.__init__()` 初始化了哪些组件？

答：与 `AsyncLLM` 类似，它会初始化：

- `vllm_config`；
- `model_config`；
- renderer；
- input_processor；
- output_processor；
- engine_core；
- stat logger。

区别是它是同步 engine 包装。

### Q113：`LLMEngine.from_engine_args()` 做什么？

答：它会：

1. 调用 `engine_args.create_engine_config()` 创建 `VllmConfig`；
2. 调用 `Executor.get_class(vllm_config)` 选择 executor；
3. 创建 `LLMEngine`。

### Q114：`LLMEngine.add_request()` 做什么？

答：它会：

1. 校验 request_id；
2. 把 prompt/params 处理为 request；
3. 分配 request id；
4. 添加到 `OutputProcessor`；
5. 添加到 `engine_core`；
6. 如果 `n > 1`，fan out 子请求。

### Q115：`LLMEngine.step()` 的作用是什么？

答：`step()` 是同步离线执行循环的核心：

1. 从 EngineCore 获取 output；
2. 由 `OutputProcessor` 转成 request outputs；
3. abort stop string 完成的 request；
4. 记录 stats；
5. 返回 `RequestOutput` 或 `PoolingRequestOutput`。

## 19. 输入渲染、tokenization、多模态与参数转换

### Q116：Chat request 为什么需要 render？

答：外部 Chat API 输入是 messages 数组，不是模型可直接执行的 token 序列。render 会根据 chat template 把 messages、system/user/assistant 角色、多模态内容、tools、generation prompt 等转换为 prompt 或 engine input。

### Q117：Completion request 为什么也需要 render？

答：Completion 输入可能是字符串、token ids 或多个 prompt。render 层负责把这些协议输入统一成 engine 可消费的 `engine_inputs`，同时处理 tokenization 和相关 metadata。

### Q118：`OpenAIServingRender` 的作用是什么？

答：它统一负责 chat、completion、tokenization、render 等预处理能力。Chat/Completion serving 对象通过它把外部 request 渲染成 engine inputs。

### Q119：SamplingParams 是在哪里构造的？

答：通常在 serving 对象中由 request model 的方法构造，例如：

```python
request.to_sampling_params(max_tokens, self.default_sampling_params)
```

或 beam search 场景：

```python
request.to_beam_search_params(max_tokens, self.default_sampling_params)
```

### Q120：PoolingParams 是在哪里使用的？

答：Pooling/Embedding/Classify/Score 等接口的 serving 逻辑会把外部请求参数转换成 `PoolingParams`，然后调用 `EngineClient.encode()`。

### Q121：LoRA adapter selection 属于哪一层？

答：属于 API/serving 的预处理和 metadata 组装层。请求会根据 model 或 LoRA 参数确定 `lora_request`，然后随 `engine_client.generate()` 或 `encode()` 调用传入 engine。

### Q122：tool parser 和 reasoning parser 在启动时如何处理？

答：`setup_server(args)` 会加载 tool parser plugin 和 reasoning parser plugin。之后 Chat serving 在响应生成阶段使用这些 parser 解析模型输出中的 tool call 或 reasoning 内容。

## 20. 请求 ID、日志、trace 与优先级

### Q123：为什么请求需要 request id？

答：request id 用于在 API 层、AsyncLLM、OutputProcessor、EngineCore、日志和流式队列之间关联同一个请求。多 prompt 或 `n > 1` 时还可能生成子 request id。

### Q124：trace headers 的作用是什么？

答：trace headers 用于把上游 tracing 信息传递到 engine 请求链路，方便分布式追踪、性能分析和请求级诊断。

### Q125：priority 参数有什么意义？

答：priority 会随请求传入 engine，供 engine/scheduler 在支持优先级调度时参考。API 层只负责提取和传递，不实现调度策略。

### Q126：data_parallel_rank 参数有什么意义？

答：它用于数据并行场景中指定请求进入哪个 data parallel rank。API 层根据请求或服务配置传递该 metadata，具体执行由 engine 和分布式组件完成。

## 21. 健康检查、鉴权、指标与运维

### Q127：API key 鉴权在哪一层实现？

答：在 FastAPI app 的 middleware 层实现。`build_app()` 会配置 API key 鉴权 middleware，对请求进行统一鉴权。

### Q128：CORS 在哪里配置？

答：在 `build_app()` 中通过 CORS middleware 配置，属于 HTTP API server 层能力。

### Q129：X-Request-Id middleware 的作用是什么？

答：它用于为请求维护或生成请求 ID，便于日志追踪、链路诊断和响应头透传。

### Q130：metrics 接口属于哪类接口？

答：属于 serve 通用管理接口，由 instrumentator 相关 router 注册，用于暴露 Prometheus metrics 等监控数据。

### Q131：health check 的价值是什么？

答：health check 用于外部负载均衡、Kubernetes、SageMaker 等平台判断服务是否可用。它通常会间接检查 API server 或 engine 的健康状态。

## 22. gRPC 与 Legacy API Server

### Q132：gRPC server 的启动链路是什么？

答：

```text
vllm serve <model> --grpc
  -> ServeSubcommand.cmd
  -> serve_grpc(args)
  -> AsyncEngineArgs
  -> VllmConfig
  -> AsyncLLM
  -> VllmEngineServicer
  -> gRPC health service
  -> reflection
  -> listen host/port
```

### Q133：gRPC 和 HTTP server 的共同点是什么？

答：二者都是在线 serving 入口，最终都会创建 V1 `AsyncLLM` 或等价 engine client，并把外部协议请求转换成 engine 请求。

### Q134：legacy demo API server 是什么？

答：文件是 `vllm/entrypoints/api_server.py`，源码注释明确说明它仅用于演示 AsyncEngine 用法和简单性能 benchmark，不用于生产。生产应使用 OpenAI-compatible server。

### Q135：legacy demo API server 有哪些接口？

答：主要包括：

```http
GET  /health
POST /generate
```

它不是生产级 OpenAI-compatible server。

## 23. 常见设计取舍题

### Q136：为什么在线服务使用 `AsyncLLM`，离线 API 使用 `LLMEngine`？

答：在线服务需要支持异步 HTTP 请求、并发请求、流式输出、取消、后台 output handler、请求队列等能力，因此使用异步 `AsyncLLM` 作为 `EngineClient`。离线 API 更关注同步批处理和易用性，使用 `LLMEngine` 进行 add_request + step 的同步循环。

### Q137：为什么 API server 不直接依赖 `AsyncLLM` 的具体实现，而是依赖 `EngineClient`？

答：依赖协议可以解耦 API 层和 engine 实现。API 层只需要知道 `generate()`、`encode()`、健康检查、profile、LoRA、cache reset 等抽象方法，不需要关心 engine 内部如何调度、执行或通信。这样有利于替换 engine 实现和维护边界清晰。

### Q138：为什么路由是动态注册的？

答：因为不同模型支持的任务不同。动态注册可以让 API server 只暴露当前模型真正支持的接口，减少错误调用，也可以根据 generate、render、pooling、transcription、realtime 等能力组合出不同服务形态。

### Q139：为什么要把 serving handler 放到 `app.state`？

答：FastAPI router 本身应该保持薄逻辑。把 engine client 和 serving 对象放入 `app.state` 后，router 可以在每个请求中取出对应 handler，避免全局变量，也便于生命周期初始化、测试和多种 handler 管理。

### Q140：为什么需要 `OpenAIServingModels`？

答：它不仅用于 `/v1/models` 返回模型列表，还维护 base model、LoRA model registry 等模型可用性信息。Chat/Completion 等 serving 对象会通过它检查请求中的 model 是否可用。

### Q141：为什么 Chat/Completion 要先 `_check_model()`？

答：因为请求中的 `model` 必须是当前 server 可服务的 base model 或 LoRA model。提前检查可以返回协议层错误，而不是把无效请求传入 engine。

### Q142：为什么 API 层要检查 engine 是否 dead？

答：如果 engine 已经进入 dead/errored 状态，继续接收请求只会失败或阻塞。API 层提前检查可以快速返回错误，避免请求进入不可用 engine。

### Q143：为什么 streaming output 需要 output handler？

答：engine core 输出是异步产生的。`AsyncLLM` 通过后台 output handler 从 EngineCore 拉取输出，并分发到每个 request 的 queue。`generate()` 再从对应 queue 中 yield `RequestOutput` 给 API 层，实现并发请求下的异步流式输出。

### Q144：为什么多 prompt completion 要 merge async iterators？

答：每个 prompt 会对应一个独立的 engine request 和异步输出流。为了对外表现为一个 completion response，需要把多个异步生成器合并，并按协议组装 choices。

### Q145：为什么 API 层要支持 cancellation/abort？

答：在线请求可能因为客户端断开、超时或用户主动取消而不再需要生成。API 层通过 cancellation/abort 机制把取消信号传递给 engine，释放资源并避免无意义计算。

### Q146：为什么管理接口也通过 `EngineClient`？

答：profile、LoRA、cache reset、sleep/wake_up、pause/resume 等管理操作都作用于 engine。统一通过 `EngineClient` 可以保持 API 层不直接接触 engine 内部对象，同时支持不同 engine client 实现。

### Q147：为什么 Dev Mode API 不应在生产使用？

答：Dev Mode API 暴露 cache reset、sleep、RLHF weight update、collective RPC 等强控制能力，可能影响服务稳定性、安全性和一致性。因此只有开启 `VLLM_SERVER_DEV_MODE` 才注册，并且源码有安全警告。

### Q148：为什么 `LLM(data_parallel_size > 1)` 在离线单进程中被限制？

答：离线单进程数据并行容易导致进程协调和通信 hang。源码提示应使用 explicit multi-process data parallel example，而不是直接在单进程 `LLM` 中开启 `data_parallel_size > 1`。

### Q149：为什么 `worker_cls` 传入 Python class 时要用 `cloudpickle` 序列化？

答：worker 可能跨进程传递。普通 pickle 对动态 class 或复杂对象支持有限，`cloudpickle.dumps()` 可以降低 pickling 问题，提高多进程环境下 worker class 传递的可靠性。

### Q150：为什么离线 LLM 默认关闭 stats log？

答：离线推理通常是脚本或批处理使用场景，默认关闭 stats log 可以减少日志噪音和额外开销。在线服务则更需要 metrics 和日志用于运维。

## 24. 接口清单考察题

### Q151：OpenAI-compatible generate 相关接口有哪些？

答：主要包括：

```http
GET  /v1/models
POST /v1/chat/completions
POST /v1/chat/completions/batch
POST /v1/completions
POST /v1/responses
GET  /v1/responses/{response_id}
POST /v1/responses/{response_id}/cancel
```

### Q152：Anthropic-compatible 接口有哪些？

答：

```http
POST /v1/messages
POST /v1/messages/count_tokens
```

### Q153：Pooling/Embedding/Scoring 类接口有哪些？

答：典型包括：

```http
POST /v1/embeddings
POST /embed
POST /classify
POST /score
POST /rerank
POST /pooling
```

具体是否注册取决于模型支持的任务。

### Q154：通用管理接口有哪些？

答：包括 health、metrics、offline docs、LoRA、profile、tokenize/detokenize、render、disaggregated/tokens、elastic EP，以及 Dev Mode 下的 cache、sleep、RLHF、RPC、server info 等。

### Q155：vLLM 有哪些非 HTTP 接口？

答：主要有：

1. Python 包级 API：`LLM`、`SamplingParams`、`PoolingParams` 等；
2. CLI：`vllm serve/chat/complete/bench/collect-env/run-batch`；
3. gRPC：`vllm serve --grpc`。

## 25. 端到端链路复述题

### Q156：请复述 `vllm serve <model>` 到 HTTP 服务可用的完整链路。

答：

```text
vllm serve <model>
  -> pyproject.toml project.scripts
  -> vllm.entrypoints.cli.main:main
  -> 加载 vllm.entrypoints.cli.serve
  -> ServeSubcommand.subparser_init
  -> openai.cli_args.make_arg_parser
  -> parser.parse_args
  -> validate_parsed_serve_args
  -> ServeSubcommand.cmd
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

### Q157：请复述 `/v1/chat/completions` 从 HTTP 到 engine 的链路。

答：

```text
POST /v1/chat/completions
  -> FastAPI router create_chat_completion
  -> request.app.state.openai_serving_chat
  -> OpenAIServingChat.create_chat_completion
  -> _create_chat_completion
  -> render_chat_request
  -> OpenAIServingRender.render_chat
  -> request.to_sampling_params / to_beam_search_params
  -> engine_client.generate
  -> AsyncLLM.generate
  -> AsyncLLM.add_request
  -> InputProcessor.process_inputs
  -> AsyncLLM._add_request
  -> OutputProcessor.add_request
  -> EngineCoreClient.add_request_async
  -> EngineCore / Scheduler / Executor / Worker
```

### Q158：请复述 `/v1/completions` 从 HTTP 到 engine 的链路。

答：

```text
POST /v1/completions
  -> FastAPI router create_completion
  -> request.app.state.openai_serving_completion
  -> OpenAIServingCompletion.create_completion
  -> _create_completion
  -> render_completion_request
  -> OpenAIServingRender.render_completion
  -> request.to_sampling_params / to_beam_search_params
  -> engine_client.generate
  -> merge_async_iterators for multi prompt
  -> AsyncLLM.generate
  -> AsyncLLM.add_request
  -> EngineCoreClient.add_request_async
```

### Q159：请复述离线 `LLM.generate()` 的链路。

答：

```text
from vllm import LLM
  -> vllm.__getattr__("LLM")
  -> vllm.entrypoints.llm.LLM
  -> LLM.__init__
      -> EngineArgs
      -> LLMEngine.from_engine_args
  -> LLM.generate
  -> OfflineInferenceMixin._run_completion
  -> LLMEngine.add_request
  -> LLMEngine.step
  -> OutputProcessor
  -> RequestOutput
```

### Q160：请复述 API 层与 Engine 层的边界图。

答：

```text
HTTP / Python / gRPC / CLI
  -> entrypoints
  -> protocol model / argparse args
  -> serving object
  -> render/tokenize/validate
  -> EngineInput + SamplingParams/PoolingParams
  -> EngineClient.generate / encode
  -> AsyncLLM.add_request / LLMEngine.add_request
  -> InputProcessor.process_inputs
  -> OutputProcessor.add_request
  -> EngineCoreClient.add_request_async / add_request
  -> EngineCore / Scheduler / Executor / Worker
```

## 26. 高频面试总结题

### Q161：一句话总结 vLLM 入口层的核心职责。

答：vLLM 入口层负责把 Python、CLI、HTTP、gRPC 等外部协议请求转换为 engine 可理解的统一输入，并把 engine 输出包装成对应协议响应；它不负责调度、KV Cache 和模型执行。

### Q162：一句话总结 `vllm serve` 的职责。

答：`vllm serve` 负责选择服务形态、解析和校验参数，并启动对应的 HTTP/gRPC/headless/multi API server 流程，真正 engine 创建发生在后续 server 初始化中。

### Q163：一句话总结 FastAPI router 的职责。

答：FastAPI router 只负责声明接口、取出 `app.state` 中的 serving handler、调用 handler，并根据返回类型选择 JSON 或 SSE response。

### Q164：一句话总结 serving 对象的职责。

答：serving 对象负责请求校验、模型检查、输入渲染、SamplingParams/PoolingParams 转换、调用 engine client、组装协议响应。

### Q165：一句话总结 `EngineClient` 的意义。

答：`EngineClient` 是 API 层和 engine 层的抽象边界，使 API server 可以通过统一协议调用生成、编码和控制能力，而不依赖具体 engine 内部实现。

### Q166：一句话总结 `AsyncLLM` 的意义。

答：`AsyncLLM` 是在线服务使用的异步 engine client，实现请求入队、异步输出分发、流式生成和 EngineCore 通信。

### Q167：一句话总结 `LLM` 的意义。

答：`LLM` 是离线推理高级 Python API，把 engine 配置、请求入队、同步执行循环和输出收集封装成易用的 `generate()`、`chat()` 等方法。

### Q168：一句话总结 Chat 和 Completion 的区别。

答：Chat 和 Completion 的主要差异在输入渲染和响应包装，前者面向 messages、tool calls、reasoning 和 delta message，后者面向 prompt/text 和 delta text；两者最终都调用 `engine_client.generate()`。

### Q169：一句话总结 OpenAI API 和 Anthropic API 的关系。

答：它们是不同外部协议适配，最终都会转换为 vLLM engine 的统一生成请求。

### Q170：一句话总结管理接口如何作用到 engine。

答：LoRA、profile、cache reset、sleep、pause/resume 等管理接口通过 `EngineClient` 的控制方法作用到 engine，而不是 router 直接操作 engine 内部对象。
