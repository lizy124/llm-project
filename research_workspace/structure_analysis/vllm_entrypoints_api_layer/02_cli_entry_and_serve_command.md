# 02 CLI 入口与 serve 命令

## 1. 命令注册

vLLM 的 CLI 命令由 `pyproject.toml` 注册：

```toml
[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"
```

源码位置：`code/vllm/pyproject.toml:43-44`。

因此用户执行：

```bash
vllm serve Qwen/Qwen3-0.6B
```

实际进入：

```text
vllm/entrypoints/cli/main.py:main()
```

## 2. CLI 主入口职责

`vllm.entrypoints.cli.main.main()` 做几件事：

1. 延迟 import 子命令模块，避免 eager import 导致平台或依赖问题。
2. 执行 CLI 环境初始化。
3. 创建 `FlexibleArgumentParser`。
4. 注册子命令。
5. 解析参数。
6. 调用子命令的 `validate()`。
7. 调用子命令的 `cmd()`。

关键源码：

- 延迟 import：`code/vllm/vllm/entrypoints/cli/main.py:17-28`
- 子命令模块列表：`code/vllm/vllm/entrypoints/cli/main.py:30-37`
- 创建 parser 与注册子命令：`code/vllm/vllm/entrypoints/cli/main.py:73-89`
- 执行 validate 和 dispatch：`code/vllm/vllm/entrypoints/cli/main.py:90-97`

## 3. CLI 子命令结构

主入口加载的子命令模块包括：

| 子命令模块 | 作用 |
|---|---|
| `vllm.entrypoints.cli.openai` | `vllm chat`、`vllm complete`，作为 OpenAI-compatible server 的客户端 |
| `vllm.entrypoints.cli.serve` | `vllm serve`，启动本地服务 |
| `vllm.entrypoints.cli.launch` | 多进程/分布式 launch 相关 |
| `vllm.entrypoints.cli.benchmark.main` | benchmark 子命令 |
| `vllm.entrypoints.cli.collect_env` | 环境采集 |
| `vllm.entrypoints.cli.run_batch` | batch 请求运行 |

所有子命令都实现 `cmd_init()`，返回 `CLISubcommand` 对象，然后由主入口统一注册。

## 4. `vllm serve` 子命令

`vllm serve` 对应类：

```python
class ServeSubcommand(CLISubcommand):
    name = "serve"
```

源码位置：`code/vllm/vllm/entrypoints/cli/serve.py:44-48`。

### 4.1 参数解析

`ServeSubcommand.subparser_init()`：

1. 创建 `serve` 子 parser。
2. 调用 `make_arg_parser()` 注入 OpenAI server 和 engine 参数。
3. 设置帮助文本。

源码位置：`code/vllm/vllm/entrypoints/cli/serve.py:153-166`。

`make_arg_parser()` 来自：

```text
vllm.entrypoints.openai.cli_args
```

它添加：

- `model_tag`
- `--headless`
- `--api-server-count`
- `--config`
- `--grpc`
- FrontendArgs 参数
- AsyncEngineArgs 参数

源码位置：`code/vllm/vllm/entrypoints/openai/cli_args.py:340-384`。

### 4.2 参数校验

`ServeSubcommand.validate()` 调用：

```python
validate_parsed_serve_args(args)
```

源码位置：`code/vllm/vllm/entrypoints/cli/serve.py:150-151`。

校验内容包括：

- chat template 是否有效
- `--enable-auto-tool-choice` 必须配合 `--tool-call-parser`
- `--enable-log-outputs` 必须配合 `--enable-log-requests`
- data parallel multi-port external LB 参数校验

源码位置：`code/vllm/vllm/entrypoints/openai/cli_args.py:387-407`。

## 5. `vllm serve` 的启动分支

`ServeSubcommand.cmd()` 是服务启动分发中枢。

### 5.1 positional model 覆盖

如果 CLI 中指定了 positional `model_tag`，则写入 `args.model`：

```python
if hasattr(args, "model_tag") and args.model_tag is not None:
    args.model = args.model_tag
```

源码位置：`code/vllm/vllm/entrypoints/cli/serve.py:50-53`。

### 5.2 gRPC 模式

如果传入 `--grpc`：

```text
vllm serve --grpc
```

则进入：

```python
from vllm.entrypoints.grpc_server import serve_grpc
uvloop.run(serve_grpc(args))
```

源码位置：`code/vllm/vllm/entrypoints/cli/serve.py:55-59`。

### 5.3 headless 模式

`--headless` 表示不启动 API server，只启动 engine/worker。它用于多节点或数据并行场景。

入口：

```python
run_headless(args)
```

源码位置：`code/vllm/vllm/entrypoints/cli/serve.py:141-142`。

`run_headless()` 会：

1. 根据 CLI 参数构建 `AsyncEngineArgs`。
2. 创建 `VllmConfig`。
3. 根据 data parallel 配置启动本地 engine 进程。
4. 通过 `CoreEngineProcManager` 监控 engine 生命周期。

关键源码：`code/vllm/vllm/entrypoints/cli/serve.py:173-255`。

### 5.4 多 API server 模式

如果 `api_server_count > 1` 或启用 Rust frontend，则进入：

```python
run_multi_api_server(args)
```

源码位置：`code/vllm/vllm/entrypoints/cli/serve.py:143-144`。

`run_multi_api_server()` 会：

1. 先 bind HTTP socket。
2. 创建 engine config。
3. 启动 core engines。
4. 启动多个 API server 子进程，或启动 Rust frontend。
5. 等待 API server / engine / coordinator 的完成或失败。

关键源码：

- socket setup：`code/vllm/vllm/entrypoints/cli/serve.py:284-291`
- engine 地址分配：`code/vllm/vllm/entrypoints/cli/serve.py:309-321`
- 启动 core engines：`code/vllm/vllm/entrypoints/cli/serve.py:323-325`
- 启动 API server manager：`code/vllm/vllm/entrypoints/cli/serve.py:349-359`
- 监控与关闭：`code/vllm/vllm/entrypoints/cli/serve.py:371-394`

### 5.5 单 API server 模式

最常见情况：

```bash
vllm serve <model>
```

会走：

```python
uvloop.run(run_server(args))
```

源码位置：`code/vllm/vllm/entrypoints/cli/serve.py:145-148`。

`run_server()` 来自：

```text
vllm.entrypoints.openai.api_server
```

## 6. `vllm chat` 与 `vllm complete`

这两个命令不是启动 server，而是连接已经运行的 OpenAI-compatible server。

源码文件：

```text
vllm/entrypoints/cli/openai.py
```

### 6.1 共同初始化

`_interactive_cli()`：

1. 注册信号处理。
2. 读取 `--url`，默认 `http://localhost:8000/v1`。
3. 读取 API key，默认 `OPENAI_API_KEY` 或 `EMPTY`。
4. 创建 OpenAI Python client。
5. 如果未指定 model，则调用 `/v1/models` 获取第一个模型。

源码位置：`code/vllm/vllm/entrypoints/cli/openai.py:29-44`。

### 6.2 chat

`ChatCommand.name = "chat"`，对应：

```bash
vllm chat
```

它会调用 OpenAI client：

```python
client.chat.completions.create(..., stream=True)
```

源码位置：`code/vllm/vllm/entrypoints/cli/openai.py:121-158`。

### 6.3 complete

`CompleteCommand.name = "complete"`，对应：

```bash
vllm complete
```

它会调用：

```python
client.completions.create(..., stream=True)
```

源码位置：`code/vllm/vllm/entrypoints/cli/openai.py:193-221`。

## 7. CLI 到服务启动的完整链路

```text
用户命令：vllm serve Qwen/Qwen3-0.6B
  -> pyproject.toml project.scripts
  -> vllm.entrypoints.cli.main:main
  -> 加载 vllm.entrypoints.cli.serve
  -> ServeSubcommand.subparser_init
  -> openai.cli_args.make_arg_parser
  -> parser.parse_args
  -> validate_parsed_serve_args
  -> ServeSubcommand.cmd
  -> run_server / run_multi_api_server / run_headless / serve_grpc
```

## 8. 关键结论

`vllm serve` 不是直接创建模型执行器，而是先完成“服务形态选择”：

- HTTP OpenAI server
- gRPC server
- headless engine
- multi API server
- Rust frontend
- DP supervisor

真正的 engine 创建发生在后面的 `openai.api_server.build_async_engine_client()` 或 `grpc_server.serve_grpc()` 中。