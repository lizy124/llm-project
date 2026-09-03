# 07 API 层与 Engine 边界

## 1. 为什么要看边界

vLLM 的入口层看起来有很多协议：

- OpenAI Chat Completions
- OpenAI Completions
- Responses API
- Anthropic Messages
- Embeddings / Pooling / Classify / Score
- Tokenize / Render
- gRPC
- Python LLM API

但这些入口最终都需要进入 engine。理解边界的关键是：

```text
API 层负责协议适配；Engine 层负责请求调度、KV Cache、模型执行。
```

## 2. EngineClient 协议

在线 API server 依赖抽象协议：

```python
class EngineClient(ABC)
```

源码位置：`code/vllm/vllm/engine/protocol.py:40`。

它要求实现的关键属性：

- `vllm_config`
- `model_config`
- `renderer`
- `input_processor`
- `is_running`
- `is_stopped`
- `errored`
- `dead_error`

源码位置：`code/vllm/vllm/engine/protocol.py:40-62`。

## 3. EngineClient 的核心方法

### 3.1 generate

```python
def generate(
    prompt,
    sampling_params,
    request_id,
    *,
    prompt_text=None,
    lora_request=None,
    tokenization_kwargs=None,
    trace_headers=None,
    priority=0,
    data_parallel_rank=None,
    reasoning_ended=None,
    reasoning_parser_kwargs=None,
) -> AsyncGenerator[RequestOutput, None]
```

源码位置：`code/vllm/vllm/engine/protocol.py:64-84`。

这是 OpenAI Chat/Completion/Responses/Anthropic 文本生成类接口最终调用的核心方法。

### 3.2 encode

```python
def encode(
    prompt,
    pooling_params,
    request_id,
    ...
) -> AsyncGenerator[PoolingRequestOutput, None]
```

源码位置：`code/vllm/vllm/engine/protocol.py:86-99`。

这是 embedding/pooling/classify/score 类接口最终调用的核心方法。

### 3.3 控制类方法

`EngineClient` 还定义：

- `abort()`
- `check_health()`
- `start_profile()` / `stop_profile()`
- `reset_mm_cache()`
- `reset_encoder_cache()`
- `reset_prefix_cache()`
- `sleep()` / `wake_up()` / `is_sleeping()`
- `add_lora()`
- `pause_generation()` / `resume_generation()` / `is_paused()`
- `shutdown()`

源码位置：`code/vllm/vllm/engine/protocol.py:101-218`。

这解释了为什么 profile、LoRA、cache reset、sleep 等 HTTP 管理接口可以统一调用 engine。

## 4. 在线服务使用的 EngineClient：AsyncLLM

OpenAI API server 创建 engine client 的地方：

```python
from vllm.v1.engine.async_llm import AsyncLLM
async_llm = AsyncLLM.from_vllm_config(...)
```

源码位置：`code/vllm/vllm/entrypoints/openai/api_server.py:123-145`。

`AsyncLLM` 继承：

```python
class AsyncLLM(EngineClient)
```

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:70`。

## 5. AsyncLLM 初始化时做了什么

`AsyncLLM.__init__()` 主要构建：

1. `self.vllm_config`
2. `self.model_config`
3. tracing
4. `self.renderer = renderer_from_config(...)`
5. `self.input_processor = InputProcessor(...)`
6. `self.output_processor = OutputProcessor(...)`
7. `self.engine_core = EngineCoreClient.make_async_mp_client(...)`
8. stats logger
9. output handler task

关键源码：

- config/model/tracing：`code/vllm/vllm/v1/engine/async_llm.py:110-124`
- renderer/input/output processor：`code/vllm/vllm/v1/engine/async_llm.py:132-143`
- EngineCoreClient：`code/vllm/vllm/v1/engine/async_llm.py:145-153`
- logger：`code/vllm/vllm/v1/engine/async_llm.py:155-166`

## 6. AsyncLLM.from_vllm_config

`AsyncLLM.from_vllm_config()` 根据 `VllmConfig` 创建 AsyncLLM，并通过 `Executor.get_class(vllm_config)` 选择 executor。

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:202-229`。

这说明 API 层不直接决定用什么 executor；executor 选择由 engine config 决定。

## 7. AsyncLLM.generate 的执行链路

`AsyncLLM.generate()` 是 API server 最常调用的方法。

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:524-636`。

源码注释直接说明其步骤：

1. 创建与 request 对应的 AsyncStream / collector。
2. 处理输入。
3. 添加 request 到 detokenizer/output processor。
4. 添加 request 到 EngineCore。
5. 后台 output_handler 从 EngineCore 拉输出并放入每个 request 的 queue。
6. `generate()` 迭代 queue，把 `RequestOutput` yield 给 API server。

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:541-555`。

核心代码：

```text
AsyncLLM.generate
  -> await self.add_request(...)
  -> while not finished:
       out = q.get_nowait() or await q.get()
       yield out
```

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:558-587`。

## 8. AsyncLLM.add_request 的职责

`add_request()` 是从 API 层输入进入 engine core 前的关键转换点。

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:280-398`。

它做：

1. 检查 engine 是否 errored。
2. 判断是否 pooling。
3. 校验 fast prefill 与 prompt logprobs 的兼容性。
4. 如果是 streaming input，走 `_add_streaming_input_request()`。
5. 如果不是 `EngineCoreRequest`，调用：

```python
self.input_processor.process_inputs(...)
```

6. 设置 reasoning 状态。
7. 分配 request id。
8. 启动 output handler。
9. 创建 `RequestOutputCollector`。
10. 对 `n > 1` fan out 子请求。
11. 调用 `_add_request()`。

关键源码：

- 输入处理：`code/vllm/vllm/v1/engine/async_llm.py:333-361`
- request id 与 output handler：`code/vllm/vllm/v1/engine/async_llm.py:368-376`
- pooling 或 n=1：`code/vllm/vllm/v1/engine/async_llm.py:381-383`
- n>1 fan out：`code/vllm/vllm/v1/engine/async_llm.py:388-397`

## 9. 真正进入 EngineCore

`AsyncLLM._add_request()`：

1. 先把 request 加到 `OutputProcessor`。
2. 再调用：

```python
await self.engine_core.add_request_async(request)
```

源码位置：`code/vllm/vllm/v1/engine/async_llm.py:400-415`。

这一步之后，入口层/API 层就不再控制调度细节了，后续由 EngineCore、scheduler、executor、worker 处理。

## 10. 离线 API 使用的 Engine：LLMEngine

`LLM` 离线 API 使用：

```python
from vllm.v1.engine.llm_engine import LLMEngine
```

源码位置：`code/vllm/vllm/entrypoints/llm.py:54-55`。

`LLMEngine` 类定义：

```python
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""
```

源码位置：`code/vllm/vllm/v1/engine/llm_engine.py:48-49`。

## 11. LLMEngine 初始化

`LLMEngine.__init__()` 与 `AsyncLLM` 类似，也构建：

- `vllm_config`
- `model_config`
- renderer
- input_processor
- output_processor
- engine_core
- stat logger

但区别是它是同步 engine 包装。

关键源码：

- renderer/input/output processor：`code/vllm/vllm/v1/engine/llm_engine.py:91-102`
- EngineCoreClient：`code/vllm/vllm/v1/engine/llm_engine.py:104-111`
- stats logger：`code/vllm/vllm/v1/engine/llm_engine.py:113-121`

## 12. LLMEngine.from_engine_args

离线 `LLM` 创建 engine 时调用：

```python
LLMEngine.from_engine_args(engine_args, usage_context=UsageContext.LLM_CLASS)
```

`from_engine_args()`：

1. `engine_args.create_engine_config()`。
2. `Executor.get_class(vllm_config)`。
3. 创建 `LLMEngine`。

源码位置：`code/vllm/vllm/v1/engine/llm_engine.py:160-186`。

## 13. LLMEngine.add_request 与 step

### 13.1 add_request

`LLMEngine.add_request()`：

1. 校验 request_id。
2. 把 prompt/params 处理为 request。
3. 分配 request id。
4. 添加到 `OutputProcessor`。
5. 添加到 `engine_core`。
6. 如果 `n > 1`，fan out 子请求。

源码位置：`code/vllm/vllm/v1/engine/llm_engine.py:218-294`。

### 13.2 step

`LLMEngine.step()` 是同步离线执行循环的核心：

1. 从 EngineCore 获取 output。
2. `OutputProcessor` 转成 request outputs。
3. abort stop string 完成的 request。
4. 记录 stats。
5. 返回 `RequestOutput` 或 `PoolingRequestOutput`。

源码位置：`code/vllm/vllm/v1/engine/llm_engine.py:296-334`。

## 14. API 层与 Engine 层边界图

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

## 15. 关键边界对象

| 对象 | 位置 | 职责 |
|---|---|---|
| `EngineClient` | `vllm/engine/protocol.py` | 在线 API server 与 engine 的抽象协议 |
| `AsyncLLM` | `vllm/v1/engine/async_llm.py` | 在线异步 engine client 实现 |
| `LLMEngine` | `vllm/v1/engine/llm_engine.py` | 离线同步 engine 封装 |
| `InputProcessor` | `vllm/v1/engine/input_processor.py` | 把输入转成 EngineCoreRequest |
| `OutputProcessor` | `vllm/v1/engine/output_processor.py` | 把 EngineCore 输出转成 RequestOutput |
| `EngineCoreClient` | `vllm/v1/engine/core_client.py` | API/engine wrapper 与 EngineCore 的通信桥 |

## 16. 关键结论

入口层的“最后一站”是 `EngineClient.generate()` 或 `EngineClient.encode()`。一旦请求进入 `AsyncLLM._add_request()` 并调用 `engine_core.add_request_async()`，调度、KV cache、worker 执行就进入 engine 内部，不再属于 API 层。