# 08 接口清单

## 1. Python 包级 API

来源：`vllm/__init__.py`

| API | 来源 | 作用 |
|---|---|---|
| `LLM` | `.entrypoints.llm:LLM` | 离线推理高级 API |
| `LLMEngine` | `.engine.llm_engine:LLMEngine` | engine 兼容导出 |
| `AsyncLLMEngine` | `.engine.async_llm_engine:AsyncLLMEngine` | async engine 兼容导出 |
| `EngineArgs` | `.engine.arg_utils:EngineArgs` | 离线/sync engine 参数 |
| `AsyncEngineArgs` | `.engine.arg_utils:AsyncEngineArgs` | 在线/async engine 参数 |
| `SamplingParams` | `.sampling_params:SamplingParams` | 生成采样参数 |
| `PoolingParams` | `.pooling_params:PoolingParams` | pooling 参数 |
| `RequestOutput` | `.outputs:RequestOutput` | 生成输出 |
| `PoolingRequestOutput` | `.outputs:PoolingRequestOutput` | pooling 输出 |

源码位置：`code/vllm/vllm/__init__.py:16-39`、`code/vllm/vllm/__init__.py:76-101`。

## 2. CLI 接口

注册入口：`code/vllm/pyproject.toml:43-44`。

```bash
vllm [subcommand]
```

主要子命令：

| 命令 | 文件 | 作用 |
|---|---|---|
| `vllm serve` | `vllm/entrypoints/cli/serve.py` | 启动本地 OpenAI-compatible/gRPC/headless 服务 |
| `vllm chat` | `vllm/entrypoints/cli/openai.py` | 连接运行中的 OpenAI-compatible server，发 chat 请求 |
| `vllm complete` | `vllm/entrypoints/cli/openai.py` | 连接运行中的 OpenAI-compatible server，发 completion 请求 |
| `vllm bench` | `vllm/entrypoints/cli/benchmark/main.py` | benchmark |
| `vllm collect-env` | `vllm/entrypoints/cli/collect_env.py` | 环境信息采集 |
| `vllm run-batch` | `vllm/entrypoints/cli/run_batch.py` | batch 请求运行 |

CLI 主入口注册子命令：`code/vllm/vllm/entrypoints/cli/main.py:30-37`。

## 3. OpenAI-compatible API

### 3.1 Models

| Method | Path | Router | Handler |
|---|---|---|---|
| GET | `/v1/models` | `openai/models/api_router.py` | `OpenAIServingModels.show_available_models()` |

源码位置：`code/vllm/vllm/entrypoints/openai/models/api_router.py:20-29`。

### 3.2 Chat Completions

| Method | Path | Router | Handler |
|---|---|---|---|
| POST | `/v1/chat/completions` | `openai/chat_completion/api_router.py` | `OpenAIServingChat.create_chat_completion()` |
| POST | `/v1/chat/completions/batch` | `openai/chat_completion/api_router.py` | `OpenAIServingChatBatch.create_batch_chat_completion()` |

源码位置：

- `code/vllm/vllm/entrypoints/openai/chat_completion/api_router.py:40-74`
- `code/vllm/vllm/entrypoints/openai/chat_completion/api_router.py:77-102`

### 3.3 Completions

| Method | Path | Router | Handler |
|---|---|---|---|
| POST | `/v1/completions` | `openai/completion/api_router.py` | `OpenAIServingCompletion.create_completion()` |

源码位置：`code/vllm/vllm/entrypoints/openai/completion/api_router.py:34-70`。

### 3.4 Responses

| Method | Path | Router | Handler |
|---|---|---|---|
| POST | `/v1/responses` | `openai/responses/api_router.py` | `OpenAIServingResponses.create_responses()` |
| GET | `/v1/responses/{response_id}` | `openai/responses/api_router.py` | `OpenAIServingResponses.retrieve_responses()` |
| POST | `/v1/responses/{response_id}/cancel` | `openai/responses/api_router.py` | `OpenAIServingResponses.cancel_responses()` |

源码位置：`code/vllm/vllm/entrypoints/openai/responses/api_router.py:48-128`。

## 4. Anthropic-compatible API

| Method | Path | Router | Handler |
|---|---|---|---|
| POST | `/v1/messages` | `anthropic/api_router.py` | `AnthropicServingMessages.create_messages()` |
| POST | `/v1/messages/count_tokens` | `anthropic/api_router.py` | token counting handler |

源码位置：

- `code/vllm/vllm/entrypoints/anthropic/api_router.py:49`
- `code/vllm/vllm/entrypoints/anthropic/api_router.py:95`

## 5. Pooling / Embedding / Classification / Scoring API

这些接口根据 `supported_tasks` 和 `model_config` 动态注册。

注册入口：`code/vllm/vllm/entrypoints/pooling/factories.py:104-135`。

| 类型 | 典型路径 | Router | Serving |
|---|---|---|---|
| embedding | `/v1/embeddings`、`/embed` | `pooling/embed/api_router.py` | `ServingEmbedding` |
| classify | `/classify` | `pooling/classify/api_router.py` | `ServingClassification` |
| score/rerank | `/score`、`/rerank` 等 | `pooling/scoring/api_router.py` | `ServingScores` |
| generic pooling | `/pooling` | `pooling/pooling/api_router.py` | `ServingPooling` |

相关 router 装饰器位置：

- `code/vllm/vllm/entrypoints/pooling/classify/api_router.py:23`
- `code/vllm/vllm/entrypoints/pooling/embed/api_router.py:25`
- `code/vllm/vllm/entrypoints/pooling/embed/api_router.py:46`
- `code/vllm/vllm/entrypoints/pooling/scoring/api_router.py:31`
- `code/vllm/vllm/entrypoints/pooling/scoring/api_router.py:49`
- `code/vllm/vllm/entrypoints/pooling/scoring/api_router.py:68`
- `code/vllm/vllm/entrypoints/pooling/scoring/api_router.py:86`
- `code/vllm/vllm/entrypoints/pooling/scoring/api_router.py:105`
- `code/vllm/vllm/entrypoints/pooling/pooling/api_router.py:24`

## 6. vLLM serve 通用管理接口

通用注册入口：`code/vllm/vllm/entrypoints/serve/__init__.py:11-32`。

### 6.1 Health / Metrics / Basic / Docs

注册入口：`code/vllm/vllm/entrypoints/serve/instrumentator/__init__.py:7-22`。

包含：

- health
- metrics
- basic endpoints
- offline docs

### 6.2 LoRA

| Method | Path | 作用 |
|---|---|---|
| POST | `/v1/load_lora_adapter` | 动态加载 LoRA adapter |
| POST | `/v1/unload_lora_adapter` | 卸载 LoRA adapter |

源码位置：`code/vllm/vllm/entrypoints/serve/lora/api_router.py:43`、`code/vllm/vllm/entrypoints/serve/lora/api_router.py:59`。

### 6.3 Profile

| Method | Path | 作用 |
|---|---|---|
| POST | `/start_profile` | 开始 profiling |
| POST | `/stop_profile` | 停止 profiling |

源码位置：`code/vllm/vllm/entrypoints/serve/profile/api_router.py:21`、`code/vllm/vllm/entrypoints/serve/profile/api_router.py:29`。

### 6.4 Tokenize / Detokenize

| Method | Path | 作用 |
|---|---|---|
| POST | `/tokenize` | 文本转 token |
| POST | `/detokenize` | token 转文本 |
| GET | `/tokenizer_info` | 获取 tokenizer 信息，可由参数控制是否启用 |

源码位置：

- `code/vllm/vllm/entrypoints/serve/tokenize/api_router.py:38`
- `code/vllm/vllm/entrypoints/serve/tokenize/api_router.py:64`
- `code/vllm/vllm/entrypoints/serve/tokenize/api_router.py:100`

### 6.5 Render

Router：`vllm/entrypoints/serve/render/api_router.py`

用于将 OpenAI/chat/completion 请求渲染为 prompt 或 token 结构。

源码位置：

- `code/vllm/vllm/entrypoints/serve/render/api_router.py:35`
- `code/vllm/vllm/entrypoints/serve/render/api_router.py:61`
- `code/vllm/vllm/entrypoints/serve/render/api_router.py:84`
- `code/vllm/vllm/entrypoints/serve/render/api_router.py:109`

## 7. Disaggregated / Tokens API

| Method | Path | 作用 |
|---|---|---|
| POST | `/v1/completions/tokens` | tokens in/out 形式的生成接口 |
| POST | `/abort_requests` | abort 请求 |

源码位置：

- `code/vllm/vllm/entrypoints/serve/disagg/api_router.py:49`
- `code/vllm/vllm/entrypoints/serve/disagg/api_router.py:82`

## 8. Elastic EP API

| Method | Path | 作用 |
|---|---|---|
| POST | `/scale_elastic_ep` | 弹性 expert parallel 扩缩 |
| POST | `/is_scaling_elastic_ep` | 查询是否正在扩缩 |

源码位置：

- `code/vllm/vllm/entrypoints/serve/elastic_ep/api_router.py:32`
- `code/vllm/vllm/entrypoints/serve/elastic_ep/api_router.py:90`

## 9. Dev Mode API

只有 `VLLM_SERVER_DEV_MODE` 开启才注册。

注册位置：`code/vllm/vllm/entrypoints/serve/__init__.py:35-61`。

### 9.1 Cache

| Method | Path |
|---|---|
| POST | `/reset_prefix_cache` |
| POST | `/reset_mm_cache` |
| POST | `/reset_encoder_cache` |

源码位置：

- `code/vllm/vllm/entrypoints/serve/dev/cache/api_router.py:20`
- `code/vllm/vllm/entrypoints/serve/dev/cache/api_router.py:46`
- `code/vllm/vllm/entrypoints/serve/dev/cache/api_router.py:57`

### 9.2 Sleep

| Method | Path |
|---|---|
| POST | `/sleep` |
| POST | `/wake_up` |
| GET | `/is_sleeping` |

源码位置：

- `code/vllm/vllm/entrypoints/serve/dev/sleep/api_router.py:21`
- `code/vllm/vllm/entrypoints/serve/dev/sleep/api_router.py:32`
- `code/vllm/vllm/entrypoints/serve/dev/sleep/api_router.py:45`

### 9.3 RLHF / weight update

| Method | Path |
|---|---|
| POST | `/pause` |
| POST | `/resume` |
| GET | `/is_paused` |
| POST | `/init_weight_transfer_engine` |
| POST | `/start_weight_update` |
| POST | `/update_weights` |
| POST | `/finish_weight_update` |
| GET | `/get_world_size` |

源码位置：`code/vllm/vllm/entrypoints/serve/dev/rlhf/api_router.py:29-167`。

### 9.4 RPC / Server info

| Method | Path |
|---|---|
| POST | `/collective_rpc` |
| GET | `/server_info` |

源码位置：

- `code/vllm/vllm/entrypoints/serve/dev/rpc/api_router.py:23`
- `code/vllm/vllm/entrypoints/serve/dev/server_info/api_router.py:43`

## 10. SageMaker API

Router：`vllm/entrypoints/serve/sagemaker/api_router.py`

| Method | Path | 作用 |
|---|---|---|
| GET/POST | `/ping` | SageMaker health check |
| POST | `/invocations` | SageMaker invocation endpoint |

源码位置：

- `code/vllm/vllm/entrypoints/serve/sagemaker/api_router.py:48-49`
- `code/vllm/vllm/entrypoints/serve/sagemaker/api_router.py:55`

## 11. gRPC 接口

启动方式：

```bash
vllm serve <model> --grpc
```

CLI 分支位置：`code/vllm/vllm/entrypoints/cli/serve.py:55-59`。

服务实现：

```text
vllm/entrypoints/grpc_server.py
```

`serve_grpc(args)`：

1. 创建 `AsyncEngineArgs`。
2. 创建 `VllmConfig`。
3. 创建 V1 `AsyncLLM`。
4. 创建 `VllmEngineServicer`。
5. 注册 gRPC health service。
6. 启用 reflection。
7. 监听 host/port。

源码位置：`code/vllm/vllm/entrypoints/grpc_server.py:56-165`。

## 12. Legacy demo API server

文件：

```text
vllm/entrypoints/api_server.py
```

源码注释明确说明：

- 仅用于演示 AsyncEngine 用法和简单性能 benchmark。
- 不用于生产。
- 生产请使用 OpenAI-compatible server。

源码位置：`code/vllm/vllm/entrypoints/api_server.py:3-9`。

接口：

| Method | Path | 作用 |
|---|---|---|
| GET | `/health` | health check |
| POST | `/generate` | 简单生成接口 |

源码位置：`code/vllm/vllm/entrypoints/api_server.py:40-95`。

## 13. 总结

vLLM 的接口层非常丰富，但可以分成五类：

1. Python 包级 API：离线使用。
2. CLI：启动服务、客户端调用、benchmark、运维。
3. OpenAI/Anthropic HTTP API：在线 serving 主入口。
4. Pooling/Embedding/Scoring API：非生成类任务。
5. 管理接口：health、metrics、LoRA、profile、tokenize、cache、sleep、RLHF、elastic EP。