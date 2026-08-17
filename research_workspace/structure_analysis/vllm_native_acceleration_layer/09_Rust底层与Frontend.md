# 09 Rust 底层与 Frontend

## 1. Rust 在 vLLM 中的定位

`rust/README.md` 明确说明：

> 这是 vLLM 的 Rust drop-in alternative frontend，目标是用 Rust 重建 northbound serving layer，同时通过已有 engine boundary 用 ZMQ 与 core Python vLLM engine process 通信。

源码位置：`code/vllm/rust/README.md:1-7`。

所以 Rust 层不是 CUDA kernel 层，而是：

- HTTP/gRPC frontend
- OpenAI-compatible API
- chat template rendering
- structured assistant events
- reasoning/tool parsing
- tokenizer / incremental detokenizer
- token-in/token-out LLM facade
- ZMQ + MessagePack engine-core client
- managed Python engine process

## 2. Rust workspace 结构

workspace 成员：`code/vllm/rust/Cargo.toml:1-16`。

| Crate | 作用 |
|---|---|
| `vllm-chat` | chat template、structured output、reasoning/tool parsing、多模态处理 |
| `vllm-cmd` | `vllm-rs` CLI 入口 |
| `vllm-engine-core-client` | ZMQ transport + MessagePack protocol，连接 Python engine core |
| `vllm-llm` | token-in/token-out 轻量 LLM facade |
| `vllm-managed-engine` | 管理 Python headless engine 进程 |
| `vllm-metrics` | metrics |
| `vllm-mock-engine` | 测试/mock engine |
| `vllm-reasoning-parser` | reasoning parser |
| `vllm-server` | OpenAI-compatible HTTP/gRPC server，基于 axum/tonic |
| `vllm-text` | tokenizer / detokenizer / text generation facade |
| `vllm-tokenizer` | tokenizer 相关 |
| `vllm-tool-parser` | tool parser |
| `vllm-tool-parser/python` | PyO3 Python 模块 |

`rust/README.md` 给出的架构图也说明自上而下是：

```text
vllm-cmd / vllm-rs
  -> vllm-server
  -> vllm-chat
  -> vllm-text
  -> vllm-llm
  -> vllm-engine-core-client
```

源码位置：`code/vllm/rust/README.md:9-33`。

## 3. Rust 构建输出

`tools/build_rust.py` 定义两个 RustExtension：

1. `vllm.vllm-rs`
   - path：`rust/src/cmd/Cargo.toml`
   - args：`--bin vllm-rs`
   - binding：`Exec`
2. `vllm._rust_tool_parser`
   - path：`rust/src/tool-parser/python/Cargo.toml`
   - binding：`PyO3`
   - py_limited_api：True

源码位置：`code/vllm/tools/build_rust.py:18-36`。

## 4. Rust frontend 启动方式

README 中示例：

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B
```

源码位置：`code/vllm/rust/README.md:39-43`。

也可以独立运行 Rust frontend-only server，连接外部 headless Python engines：`code/vllm/rust/README.md:45-68`。

## 5. `vllm-rs` CLI：`vllm-cmd`

入口文件：

```text
rust/src/cmd/src/main.rs
```

它使用 mimalloc 作为全局 allocator：`code/vllm/rust/src/cmd/src/main.rs:14-15`。

启动时：

1. 初始化 tracing。
2. 解析 CLI。
3. 构建 Tokio runtime。
4. 限制默认 Tokio worker threads，避免大 CPU 机器过度线程切换。
5. 根据 command 分发。

源码位置：`code/vllm/rust/src/cmd/src/main.rs:81-100`。

### 5.1 Frontend 模式

```rust
Command::Frontend(args) => vllm_server::serve(args.into_config(), shutdown_signal()).await
```

源码位置：`code/vllm/rust/src/cmd/src/main.rs:97-100`。

### 5.2 Serve 模式

`Command::Serve(args)` 支持两种模式：

1. `data_parallel_size_local == Some(0)`：Rust frontend 不管理本地 Python engine，只连接外部 engine。
2. 默认：spawn managed Python headless engine，同时启动 Rust OpenAI server。

源码位置：`code/vllm/rust/src/cmd/src/main.rs:100-190`。

关键调用：

- `ManagedEngineHandle::spawn(engine_config)`：`code/vllm/rust/src/cmd/src/main.rs:122-124`
- `vllm_server::serve(config, shutdown)`：`code/vllm/rust/src/cmd/src/main.rs:136-144`

## 6. Rust server：`vllm-server`

入口文件：

```text
rust/src/server/src/lib.rs
```

文件说明：

```rust
//! Minimal OpenAI-compatible HTTP server above [`vllm_chat`].
```

源码位置：`code/vllm/rust/src/server/src/lib.rs:1`。

### 6.1 build_state

`build_state(config)` 做：

1. 解析 served model names。
2. 加载 text/chat backends。
3. 根据模型是否 MoE 选择 coordinator mode。
4. 创建 `EngineCoreClient`。
5. 创建 `Llm`。
6. 创建 `TextLlm`。
7. 创建 `ChatLlm`。
8. 构造 `AppState`。

源码位置：`code/vllm/rust/src/server/src/lib.rs:47-106`。

核心连接 engine 的代码：

```rust
let client = EngineCoreClient::connect(EngineCoreClientConfig { ... }).await?
```

源码位置：`code/vllm/rust/src/server/src/lib.rs:83-90`。

### 6.2 serve

`serve(config, shutdown)` 调用 `serve_with_router_extension()`：`code/vllm/rust/src/server/src/lib.rs:108-115`。

`serve_with_router_extension()`：

1. validate config。
2. build_state。
3. bind listener。
4. build router。
5. 可选启动 gRPC server。
6. axum serve HTTP。
7. shutdown 时 drain 并关闭 state。

源码位置：`code/vllm/rust/src/server/src/lib.rs:121-268`。

### 6.3 HTTP + gRPC 并发

Rust server 可同时跑：

- HTTP：axum
- gRPC：tonic

gRPC 可选端口 `grpc_port`，绑定逻辑在：`code/vllm/rust/src/server/src/lib.rs:143-163`。

HTTP serve 逻辑：`code/vllm/rust/src/server/src/lib.rs:206-227`。

gRPC serve 逻辑：`code/vllm/rust/src/server/src/lib.rs:229-258`。

## 7. Rust chat：`vllm-chat`

入口文件：

```text
rust/src/chat/src/lib.rs
```

文件注释说明它是 `vllm_text` 之上的 minimal chat facade：

```text
messages -> rendered prompt -> tokenized prompt -> engine request -> streamed structured assistant events
```

源码位置：`code/vllm/rust/src/chat/src/lib.rs:1-8`。

### 7.1 导出能力

`vllm-chat` 导出：

- `ChatBackend`
- `ChatRenderer`
- `DeepSeekV4ChatRenderer`
- `DeepSeekV32ChatRenderer`
- `ParserSelection`
- `ReasoningParser`
- `ToolParser`
- `ChatRequest`
- `SamplingParams`
- `ChatEventStream`
- `FinishReason`

源码位置：`code/vllm/rust/src/chat/src/lib.rs:10-40`。

### 7.2 ChatLlm

`ChatLlm` 持有：

- `TextLlm`
- chat backend
- model dtype
- tool call parser selection
- reasoning parser selection

源码位置：`code/vllm/rust/src/chat/src/lib.rs:87-101`。

### 7.3 ChatLlm.chat

`chat()` 流程：

1. validate request。
2. 创建 output processor。
3. chat renderer 渲染 prompt。
4. 多模态 finalize。
5. 构造 `TextRequest`。
6. 调用 `self.text.generate(text_request)`。
7. output processor 把 decoded stream 转成 structured stream。
8. 返回 `ChatEventStream`。

源码位置：`code/vllm/rust/src/chat/src/lib.rs:170-209`。

这和 Python OpenAI server 的 `render_chat -> engine_client.generate -> stream response` 是同构的。

## 8. Rust LLM facade：`vllm-llm`

入口：

```text
rust/src/llm/src/lib.rs
```

文件说明：

> Thin generate-and-abort facade over EngineCoreClient.

源码位置：`code/vllm/rust/src/llm/src/lib.rs:23-30`。

`Llm` 持有：

- `EngineCoreClient`
- request id randomization flag
- stats logger
- inflight requests map

源码位置：`code/vllm/rust/src/llm/src/lib.rs:30-35`。

### 8.1 Llm.generate

流程：

1. `req.prepare()`。
2. 获取 prompt token ids。
3. 记录 external/internal request id。
4. 创建 request metrics tracker。
5. 调用 `self.client.call(prepared.engine_request).await?`。
6. track inflight ids。
7. 返回 `GenerateOutputStream`。

源码位置：`code/vllm/rust/src/llm/src/lib.rs:76-107`。

这对应 Python `AsyncLLM.generate()` 的较窄 Rust 版本。

### 8.2 abort

`abort()` 将 external request ids 映射到 internal engine ids，然后调用 engine core client abort。

源码位置：`code/vllm/rust/src/llm/src/lib.rs:109-123`。

## 9. EngineCoreClient：Rust 到 Python engine 的边界

crate：

```text
rust/src/engine-core-client
```

入口：`code/vllm/rust/src/engine-core-client/src/lib.rs:1-16`。

导出：

- `EngineCoreClient`
- `EngineCoreClientConfig`
- `EngineCoreOutputStream`
- `TransportMode`
- `CoordinatorMode`
- `AbortCause`
- protocol types

核心子目录：

| 路径 | 作用 |
|---|---|
| `client/` | client 实现、状态、stream |
| `coordinator/` | bootstrap/external/inproc/handle coordinator |
| `protocol/` | MessagePack 协议对象，dtype/logprobs/lora/multimodal/stats/tensor/utility |
| `transport.rs` | ZMQ transport |
| `metrics.rs` | metrics |
| `mock_engine.rs` | 测试 mock engine |

`rust/README.md` 明确说明这一层是：

```text
ZMQ transport + MessagePack protocol for the headless vLLM engine
```

源码位置：`code/vllm/rust/README.md:27-32`。

## 10. Rust server routes

`rust/src/server/src/routes` 包含：

- health
- metrics
- lora
- cache
- pause/sleep/server_info
- tokenize
- OpenAI chat completions
- OpenAI completions
- OpenAI models
- inference generate
- abort_requests
- collective_rpc

路径清单可见 Glob 结果对应源码目录：

```text
rust/src/server/src/routes/openai/chat_completions.rs
rust/src/server/src/routes/openai/completions.rs
rust/src/server/src/routes/openai/models.rs
rust/src/server/src/routes/tokenize.rs
rust/src/server/src/routes/health.rs
rust/src/server/src/routes/metrics.rs
...
```

这说明 Rust frontend 正在复刻 Python OpenAI-compatible server 的一部分能力。

## 11. Rust tool parser Python 模块

`tools/build_rust.py` 构建：

```text
vllm._rust_tool_parser
```

源码位置：`code/vllm/tools/build_rust.py:28-35`。

它来自：

```text
rust/src/tool-parser/python/Cargo.toml
```

用途：把 Rust tool parser 能力作为 Python extension 暴露给 Python 侧。

## 12. Rust 与 Python frontend 的关系

Rust frontend 不是完全替代整个 vLLM，而是替代“北向 serving layer”：

```text
Python 默认路径：
FastAPI OpenAI server
  -> AsyncLLM
  -> EngineCore

Rust frontend 路径：
axum/tonic server
  -> ChatLlm/TextLlm/Llm
  -> EngineCoreClient
  -> ZMQ/MessagePack
  -> Python EngineCore
```

Rust frontend 的优势主要在：

- 更低 HTTP/server overhead。
- 更强类型化协议处理。
- chat/template/parser/tokenizer 的高性能实现。
- 更容易做独立 frontend-only 部署。

## 13. 关键结论

Rust 层属于“底层 frontend/runtime 加速”，不是 GPU kernel 加速。它和 C++/CUDA 的关系是并列的：

- C++/CUDA/HIP/CPU：加速 tensor compute。
- Rust：加速服务入口、协议转换、chat rendering、tokenizer/parser、engine-core transport。

两者都绕开了纯 Python 的性能瓶颈，但作用位置不同。