# 02 请求生命周期：从 API 到 EngineCore

本篇梳理请求进入 vLLM 推理引擎后的生命周期，重点是 API 层、`AsyncLLM`、`InputProcessor`、`EngineCoreClient`、`OutputProcessor` 之间的关系。

## 1. 请求生命周期总览

```text
HTTP / Python API 请求
  ↓
entrypoints 层解析请求
  ↓
AsyncLLM.generate() / AsyncLLM.add_request()
  ↓
InputProcessor.process_inputs()
  ↓
EngineCoreRequest
  ↓
EngineCoreClient.add_request_async()
  ↓
EngineCore.add_request()
  ↓
Scheduler.add_request()
  ↓
EngineCore.step() 推进执行
  ↓
EngineCoreOutputs
  ↓
OutputProcessor.process_outputs()
  ↓
RequestOutput / PoolingRequestOutput
  ↓
API 流式或非流式返回
```

## 2. API/入口层职责

入口层通常在：

```text
vllm/entrypoints/
vllm/entrypoints/openai/
```

职责：

- 启动 FastAPI server 或 CLI；
- 解析 OpenAI-compatible 请求；
- 将 HTTP schema 转换成 vLLM 内部参数；
- 构造 `AsyncEngineArgs` / `VllmConfig`；
- 创建 `AsyncLLM`；
- 调用 `AsyncLLM.generate()`、`encode()` 等接口；
- 把结果包装成 OpenAI-compatible response。

入口层不负责实际调度和模型执行。它只负责协议、参数和服务生命周期。

## 3. AsyncLLM：前台异步引擎

`AsyncLLM` 定义在 `code/vllm/vllm/v1/engine/async_llm.py:70`。

初始化阶段中，它做几件关键事：

1. 保存 `VllmConfig`；
2. 创建 renderer/tokenizer 相关组件；
3. 创建 `InputProcessor`；
4. 创建 `OutputProcessor`；
5. 通过 `EngineCoreClient.make_async_mp_client()` 创建 EngineCore 通信客户端；
6. 创建统计日志管理器；
7. 启动 output handler。

代码锚点：

- `AsyncLLM.__init__` 从 `code/vllm/vllm/v1/engine/async_llm.py:73` 开始。
- `self.input_processor = InputProcessor(...)` 在 `code/vllm/vllm/v1/engine/async_llm.py:135`。
- `self.output_processor = OutputProcessor(...)` 在 `code/vllm/vllm/v1/engine/async_llm.py:138`。
- `self.engine_core = EngineCoreClient.make_async_mp_client(...)` 在 `code/vllm/vllm/v1/engine/async_llm.py:146`。

### AsyncLLM 的主要职责

| 方法 | 作用 |
|---|---|
| `from_vllm_config()` | 根据 VllmConfig 和 Executor class 创建 AsyncLLM |
| `from_engine_args()` | 从 EngineArgs 创建 VllmConfig，再创建 AsyncLLM |
| `add_request()` | 异步加入请求 |
| `_add_request()` | 具体处理请求、构造 collector、送入 EngineCore |
| `_add_streaming_input_request()` | 处理 streaming input |
| `generate()` | 用户侧主要生成接口，返回异步生成器 |
| `abort()` | 取消请求 |
| `encode()` | pooling/embedding 类任务入口 |
| `sleep()/wake_up()` | 显存睡眠/唤醒控制 |
| `reset_prefix_cache()` | 清理 prefix cache |

## 4. InputProcessor：用户输入到 EngineCoreRequest

`InputProcessor` 在 `code/vllm/vllm/v1/engine/input_processor.py:36`。

它的核心任务是把外部输入转换为 engine core 可调度的数据结构。

典型处理内容：

- 校验 sampling params / pooling params；
- 校验 LoRA；
- 处理 prompt、tokens、multi-modal 输入；
- 处理 tokenizer 相关逻辑；
- 处理 prompt 长度；
- 处理 request id；
- 生成 `EngineCoreRequest`。

关键方法：

| 方法 | 作用 |
|---|---|
| `_validate_params()` | 校验 sampling/pooling 参数 |
| `_validate_lora()` | 校验 LoRA 请求 |
| `inject_into_mm_cache()` | 多模态输入缓存注入 |
| `assign_request_id()` | 给请求分配或确认 request_id |
| `process_inputs()` | 输入处理主函数 |
| `_validate_prompt_len()` | 校验 prompt 长度 |
| `_validate_model_input()` | 校验模型输入 |

`process_inputs()` 是这层最重要的入口，位于 `code/vllm/vllm/v1/engine/input_processor.py:242`。

## 5. EngineCoreClient：前台和 EngineCore 的通信边界

`EngineCoreClient` 定义在 `code/vllm/vllm/v1/engine/core_client.py:71`。

它抽象了前台和 engine core 的通信方式。不同模式下可能是：

- in-process：`InprocClient`；
- multi-process：`MPClient`；
- sync：`SyncMPClient`；
- async：`AsyncMPClient`；
- data parallel async：`DPAsyncMPClient`。

### 为什么需要 EngineCoreClient

`AsyncLLM` 通常运行在 API/server 进程中，而 `EngineCore` 可能运行在后台进程或多个进程中。`EngineCoreClient` 负责隐藏这些差异：

```text
AsyncLLM
  ↓ EngineCoreClient.add_request_async()
EngineCore process
```

### 重要接口

| 接口 | 作用 |
|---|---|
| `make_client()` | 创建同步/异步 client |
| `make_async_mp_client()` | 创建异步多进程 client |
| `get_output()` / `get_output_async()` | 获取 EngineCoreOutputs |
| `add_request()` / `add_request_async()` | 添加请求 |
| `abort_requests()` | 取消请求 |
| `profile()` | 控制 profiling |
| `reset_prefix_cache()` | 重置 prefix cache |
| `sleep()/wake_up()` | 控制 worker 显存状态 |
| `collective_rpc()` | 向 worker 发控制 RPC |

`core_client.py` 的类结构可参考：

- `EngineCoreClient`：`code/vllm/vllm/v1/engine/core_client.py:71`
- `InprocClient`：`code/vllm/vllm/v1/engine/core_client.py:276`
- `MPClient`：`code/vllm/vllm/v1/engine/core_client.py:467`
- `SyncMPClient`：`code/vllm/vllm/v1/engine/core_client.py:779`
- `AsyncMPClient`：`code/vllm/vllm/v1/engine/core_client.py:950`

## 6. OutputProcessor：EngineCoreOutputs 到 RequestOutput

`OutputProcessor` 在 `code/vllm/vllm/v1/engine/output_processor.py:417`。

它负责把 engine core 的内部输出转换成用户侧结果。

### 它维护哪些状态

`OutputProcessor` 内部维护 request state 和 collector：

- `RequestOutputCollector`：异步收集输出，定义在 `code/vllm/vllm/v1/engine/output_processor.py:45`；
- `RequestState`：保存单个请求的输出状态，定义在 `code/vllm/vllm/v1/engine/output_processor.py:129`；
- `StreamingUpdate`：处理流式增量，定义在 `code/vllm/vllm/v1/engine/output_processor.py:116`。

### 主要职责

| 方法 | 作用 |
|---|---|
| `add_request()` | 注册 request state 和 collector |
| `process_outputs()` | 处理 EngineCoreOutputs 主函数 |
| `_update_streaming_request_state()` | 更新流式请求状态 |
| `_finish_request()` | 标记请求完成并清理 |
| `abort_requests()` | 处理取消请求 |
| `propagate_error()` | 把错误传播给请求输出 |
| `do_tracing()` | tracing 处理 |
| `_update_stats_from_output()` | 统计信息更新 |

## 7. 一个请求的状态演化

一个 generation 请求通常会经历：

```text
客户端请求
  ↓
AsyncLLM.generate()
  ↓
InputProcessor.process_inputs()
  ↓
EngineCoreRequest
  ↓
OutputProcessor.add_request() 注册 collector
  ↓
EngineCoreClient.add_request_async()
  ↓
EngineCore.preprocess_add_request()
  ↓
Request 对象
  ↓
Scheduler.add_request()
  ↓
WAITING
  ↓ Scheduler.schedule()
RUNNING
  ↓ Worker/GPUModelRunner 执行
ModelRunnerOutput
  ↓ Scheduler.update_from_output()
EngineCoreOutputs
  ↓ OutputProcessor.process_outputs()
RequestOutput
  ↓
客户端收到 token / final output
```

## 8. 前台和后台的分工

| 模块 | 所在位置 | 是否执行模型 | 主要职责 |
|---|---|---|---|
| API server | 前台 | 否 | HTTP 协议、请求解析、response 格式 |
| AsyncLLM | 前台 | 否 | 异步接口、输入/输出处理、client 管理 |
| EngineCoreClient | 前台/通信层 | 否 | 进程间通信、utility RPC |
| EngineCore | 后台 engine core | 间接 | 调度主循环、调用 executor |
| Executor | 后台/worker 管理层 | 间接 | 调 worker 执行 |
| Worker/GPUModelRunner | worker 进程 | 是 | 设备执行、模型 forward、采样 |

## 9. 理解这层的重点

读这层代码时要重点区分三种输出：

1. `ModelRunnerOutput`：worker/model runner 产生的底层输出，包含 sampled tokens、logprobs、KV connector output 等。
2. `EngineCoreOutputs`：engine core/scheduler 整理后的内部输出，按 request 返回。
3. `RequestOutput`：API/user 可见输出。

因此，不要把 `GPUModelRunner` 的输出直接理解成最终用户输出。最终用户输出一定要经过 scheduler 和 output processor。
