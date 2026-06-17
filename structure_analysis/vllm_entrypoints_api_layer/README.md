# vLLM 服务入口与 API 层梳理

本目录梳理 `D:/lzy/project/kv_pool/code/vllm` 中 vLLM 的服务入口与 API 层，重点覆盖：

- Python 包对外 API：`vllm/__init__.py`、`vllm.entrypoints.llm.LLM`
- CLI 入口：`vllm` 命令、`vllm serve`、`vllm chat`、`vllm complete`
- OpenAI 兼容 HTTP 服务：FastAPI app、router 注册、请求处理、SSE 流式返回
- Anthropic、Responses、Pooling、Tokenize、LoRA、Profile、Disaggregated 等 API
- API 层与 V1 Engine 的边界：`EngineClient`、`AsyncLLM`、`LLMEngine`

## 文档索引

1. [01_入口层总览.md](01_入口层总览.md)
   - 服务入口与 API 层在 vLLM 整体框架中的位置。
2. [02_CLI入口与serve命令.md](02_CLI入口与serve命令.md)
   - `vllm` 命令、子命令注册、`vllm serve` 启动模式。
3. [03_HTTP服务生命周期.md](03_HTTP服务生命周期.md)
   - OpenAI API Server 的启动、FastAPI app 构建、EngineClient 初始化。
4. [04_API路由与服务对象.md](04_API路由与服务对象.md)
   - OpenAI、Anthropic、Pooling、Serve 管理类接口的 router 与 serving 对象。
5. [05_Chat与Completion请求链路.md](05_Chat与Completion请求链路.md)
   - `/v1/chat/completions` 与 `/v1/completions` 从请求到 engine 的详细链路。
6. [06_离线LLM_API.md](06_离线LLM_API.md)
   - `from vllm import LLM` 离线推理 API 的构造、generate、chat。
7. [07_API层与Engine边界.md](07_API层与Engine边界.md)
   - API 层与 `EngineClient`、`AsyncLLM`、`LLMEngine` 的职责边界。
8. [08_接口清单.md](08_接口清单.md)
   - 主要 HTTP/gRPC/CLI 接口清单。

## 一句话总结

vLLM 的服务入口与 API 层是“外部请求协议适配层”：它负责把 CLI、HTTP、gRPC、Python API 请求解析成统一的 engine 输入，并把 engine 输出重新包装成 OpenAI/Anthropic/vLLM 自己的响应格式；真正的调度、KV Cache、模型执行不在入口层完成，而是通过 `EngineClient` 交给 V1 engine。