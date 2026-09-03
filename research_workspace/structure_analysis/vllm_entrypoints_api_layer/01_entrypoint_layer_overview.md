# 01 入口层总览

## 1. 入口层回答的问题

vLLM 的“服务入口与 API 层”主要回答四类问题：

1. 用户从哪里进入 vLLM？
   - Python：`from vllm import LLM`
   - CLI：`vllm serve`、`vllm chat`、`vllm complete`
   - HTTP：OpenAI-compatible REST API
   - gRPC：`vllm serve --grpc`
2. 外部协议如何转成 vLLM 内部请求？
   - Pydantic request model 校验 HTTP body。
   - chat template/render/tokenize 将文本、多模态、工具调用等输入转成 `EngineInput`。
   - sampling/pooling 参数转成 `SamplingParams` 或 `PoolingParams`。
3. API 层如何连接 engine？
   - 在线服务统一依赖 `EngineClient` 协议。
   - 当前 OpenAI API Server 走 V1 `AsyncLLM`。
4. engine 输出如何回到外部协议？
   - 非流式：收集最终 `RequestOutput`，转成 JSON response。
   - 流式：把 `RequestOutput` 增量转成 SSE `data: ...`。

## 2. 顶层公开 API

`vllm/__init__.py` 是 vLLM Python 包的门面。它用 `MODULE_ATTRS` 懒加载公开对象，包括：

- `LLM`
- `LLMEngine`
- `AsyncLLMEngine`
- `EngineArgs`
- `AsyncEngineArgs`
- `SamplingParams`
- `PoolingParams`
- 各类输出对象

源码位置：`code/vllm/vllm/__init__.py:16-39`。

`__all__` 明确声明了包级别对外暴露对象，说明用户可以直接 `from vllm import LLM, SamplingParams, RequestOutput` 等：`code/vllm/vllm/__init__.py:76-101`。

## 3. CLI 入口

项目安装后注册一个命令：

```toml
[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"
```

源码位置：`code/vllm/pyproject.toml:43-44`。

也就是说，用户执行 `vllm ...` 时，实际进入的是：

```text
vllm.entrypoints.cli.main:main
```

CLI 主入口再动态加载子命令模块：

- `vllm.entrypoints.cli.openai`
- `vllm.entrypoints.cli.serve`
- `vllm.entrypoints.cli.launch`
- `vllm.entrypoints.cli.benchmark.main`
- `vllm.entrypoints.cli.collect_env`
- `vllm.entrypoints.cli.run_batch`

源码位置：`code/vllm/vllm/entrypoints/cli/main.py:17-37`。

## 4. 在线服务入口

主要入口是：

```text
vllm.entrypoints.cli.serve.ServeSubcommand
```

`ServeSubcommand.name = "serve"`，即对应 `vllm serve`：`code/vllm/vllm/entrypoints/cli/serve.py:44-48`。

`vllm serve` 默认启动 OpenAI-compatible API server；如果指定 `--grpc`，则启动 gRPC server：`code/vllm/vllm/entrypoints/cli/serve.py:55-59`。

在线服务的核心文件是：

```text
vllm/entrypoints/openai/api_server.py
```

它负责：

- 创建 socket
- 构建 FastAPI app
- 注册 routers
- 初始化 app.state
- 创建 `AsyncLLM` engine client
- 启动 uvicorn/HTTP server

## 5. 离线 Python API 入口

离线推理入口是：

```text
vllm.entrypoints.llm.LLM
```

`LLM` 类描述自己是“给定 prompt 和 sampling params 生成文本的 LLM”，包含 tokenizer、模型、KV cache、智能 batching 和内存管理：`code/vllm/vllm/entrypoints/llm.py:66-74`。

它适合 offline inference；源码注释明确提示 online serving 应使用 Async engine 类：`code/vllm/vllm/entrypoints/llm.py:171-174`。

## 6. API 层的核心分层

可以把入口层拆成五层：

```text
外部入口
  ├─ Python import: vllm.LLM
  ├─ CLI: vllm serve/chat/complete
  ├─ HTTP: /v1/chat/completions 等
  └─ gRPC: vllm serve --grpc

协议解析层
  ├─ argparse CLI 参数
  ├─ FastAPI route + Pydantic request model
  └─ OpenAI/Anthropic/Pooling 协议对象

服务对象层
  ├─ OpenAIServingChat
  ├─ OpenAIServingCompletion
  ├─ OpenAIServingResponses
  ├─ OpenAIServingModels
  ├─ OpenAIServingTokenization
  ├─ ServingEmbedding / ServingPooling / ServingScores
  └─ AnthropicServingMessages

渲染与预处理层
  ├─ chat template
  ├─ tokenizer
  ├─ multimodal input
  ├─ LoRA adapter selection
  ├─ tool call / reasoning parser
  └─ SamplingParams / PoolingParams

Engine 边界层
  ├─ EngineClient protocol
  ├─ AsyncLLM
  └─ LLMEngine
```

## 7. 最重要的调用方向

### 在线 OpenAI API

```text
vllm serve
  -> ServeSubcommand.cmd
  -> run_server / run_multi_api_server / run_headless / serve_grpc
  -> openai.api_server.run_server
  -> build_async_engine_client
  -> AsyncLLM.from_vllm_config
  -> build_app + init_app_state
  -> register routers
  -> HTTP request
  -> OpenAIServingXXX
  -> engine_client.generate / encode
  -> AsyncLLM.add_request
  -> EngineCoreClient
```

### 离线 LLM API

```text
from vllm import LLM
  -> vllm.__getattr__("LLM")
  -> vllm.entrypoints.llm.LLM
  -> EngineArgs
  -> LLMEngine.from_engine_args
  -> LLM.generate / LLM.chat
  -> OfflineInferenceMixin
  -> LLMEngine.add_request + step
```

## 8. 入口层不负责什么

入口层本身不负责：

- 具体 scheduler 调度算法
- KV Cache block 分配
- GPU model runner 执行
- attention kernel
- worker/executor 细节

这些工作在 V1 engine、core、worker、executor、model runner、csrc 中完成。入口层只负责把外部协议转换成 engine 能理解的请求，并把 engine 输出转换成协议响应。