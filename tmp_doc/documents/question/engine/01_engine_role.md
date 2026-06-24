# 01. Engine 在 vLLM V1 里指什么？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/engine/__init__.py`

本问题关注：`Engine` 到底是不是 `LLMEngine` / `AsyncLLM`，以及外层 Engine 和内部 `EngineCore` 的关系。

---

## 1. 一句话回答

在 vLLM V1 的这组代码里，`Engine` 更像一个架构层面的泛称，不是一个唯一固定的具体类。

可以理解为：

```text
Engine = vLLM 对外提供推理能力的外层引擎体系。
```

在具体代码里，常见的外层 Engine 形态主要是：

```text
LLMEngine：
  同步 / legacy 兼容路径里的外层 Engine。

AsyncLLM：
  异步 / API server 常用路径里的外层 Engine。
```

而 `EngineCore` 不是外层 Engine 本身，它是外层 Engine 背后的内部执行核心。

源码里 `EngineCore` 的定位非常直接：

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
```

位置：`vllm/vllm/v1/engine/core.py:96` 到 `vllm/vllm/v1/engine/core.py:97`

这句话说明：

```text
EngineCore 是 Engine 的 inner loop；
所以 EngineCore 更靠内，Engine 更靠外。
```

一句话总结：

```text
Engine 是外层引擎体系的泛称；
LLMEngine 和 AsyncLLM 是这个体系的两种具体外层对象；
EngineCore 是它们通过 EngineCoreClient 驱动的内部执行主循环。
```

---

## 2. Engine 是具体类还是泛称

严格说，在这里讨论的 `Engine` 不是一个必须精确对应到某个单一类名的对象。

更准确的理解是：

```text
Engine 是一层职责边界：
  对外接收请求；
  做输入处理；
  把请求送入 EngineCore；
  从 EngineCore 取回内部输出；
  做输出处理；
  向调用方返回 RequestOutput / PoolingRequestOutput。
```

所以当文档里说：

```text
Engine 和 EngineCore 的关系
```

通常不是在问：

```text
某个名叫 Engine 的类和 EngineCore 的继承关系是什么？
```

而是在问：

```text
外层引擎对象和内部执行核心的职责边界是什么？
```

在 V1 代码中，最典型的外层引擎对象是：

```text
同步路径：LLMEngine
异步路径：AsyncLLM
```

它们都做类似的外层工作：

```text
InputProcessor：
  EngineInput / PromptType → EngineCoreRequest

EngineCoreClient：
  EngineCoreRequest → EngineCore
  EngineCoreOutputs ← EngineCore

OutputProcessor：
  EngineCoreOutputs → RequestOutput / PoolingRequestOutput
```

因此可以画成：

```text
Engine（概念层 / 外层引擎体系）
  ├─ LLMEngine：同步形态
  └─ AsyncLLM：异步形态

EngineCore：
  Engine 内部真正执行 schedule → execute → update → output 的 inner loop
```

---

## 3. LLMEngine 和 Engine 的关系

`LLMEngine` 是同步路径里的外层 Engine。

源码里类定义是：

```python
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""
```

位置：`vllm/vllm/v1/engine/llm_engine.py:48` 到 `vllm/vllm/v1/engine/llm_engine.py:49`

这里说它是 `Legacy LLMEngine for backwards compatibility`，说明它保留了传统同步 `LLMEngine` 接口形态。

### 3.1 LLMEngine 初始化了外层输入输出处理器

`LLMEngine.__init__()` 里先创建 renderer，然后创建 `InputProcessor`：

```python
self.renderer = renderer = renderer_from_config(self.vllm_config)

# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:91` 到 `vllm/vllm/v1/engine/llm_engine.py:94`

接着创建 `OutputProcessor`：

```python
# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:96` 到 `vllm/vllm/v1/engine/llm_engine.py:102`

这两个注释已经把边界说得很清楚：

```text
InputProcessor：
  EngineInput → EngineCoreRequest

OutputProcessor：
  EngineCoreOutputs → RequestOutput
```

所以 `LLMEngine` 不是直接做模型 forward 的组件，它首先是一个输入输出编排层。

### 3.2 LLMEngine 持有的是 EngineCoreClient

`LLMEngine` 初始化时还会创建：

```python
# EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
self.engine_core = EngineCoreClient.make_client(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:104` 到 `vllm/vllm/v1/engine/llm_engine.py:111`

注意这里容易误解：

```text
字段名叫 self.engine_core；
但它的实际类型是 EngineCoreClient；
它不一定就是 EngineCore 对象本身。
```

如果是 in-process 模式，这个 client 里面会直接持有 `EngineCore`。

如果是 multi-process 模式，它会通过 ZMQ 和后台 `EngineCoreProc` 通信。

所以同步路径的对象关系是：

```text
LLMEngine
  → self.input_processor
  → self.output_processor
  → self.engine_core: EngineCoreClient
      → InprocClient / SyncMPClient
          → EngineCore / EngineCoreProc
```

### 3.3 LLMEngine.add_request() 是同步请求入口

同步 `add_request()` 中，如果传入的不是已经构造好的 `EngineCoreRequest`，会先调用 `InputProcessor`：

```python
request = self.input_processor.process_inputs(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:250` 到 `vllm/vllm/v1/engine/llm_engine.py:260`

然后登记到 `OutputProcessor`：

```python
self.output_processor.add_request(request, prompt_text, None, 0)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:272` 到 `vllm/vllm/v1/engine/llm_engine.py:274`

最后送入 EngineCoreClient：

```python
self.engine_core.add_request(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:275` 到 `vllm/vllm/v1/engine/llm_engine.py:276`

同步请求入口可以概括为：

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
```

### 3.4 LLMEngine.step() 是同步输出入口

同步 `step()` 的第一步是从 EngineCoreClient 拉输出：

```python
outputs = self.engine_core.get_output()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:302` 到 `vllm/vllm/v1/engine/llm_engine.py:304`

然后交给 `OutputProcessor`：

```python
processed_outputs = self.output_processor.process_outputs(
    outputs.outputs,
    engine_core_timestamp=outputs.timestamp,
    iteration_stats=iteration_stats,
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:306` 到 `vllm/vllm/v1/engine/llm_engine.py:313`

最后返回用户可见输出：

```python
return processed_outputs.request_outputs
```

位置：`vllm/vllm/v1/engine/llm_engine.py:334`

所以同步输出路径是：

```text
LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

这就是为什么 `LLMEngine` 可以理解为同步外层 Engine：

```text
它负责同步 add_request / step；
但内部真正 schedule / execute / update 在 EngineCore。
```

---

## 4. AsyncLLM 和 Engine 的关系

`AsyncLLM` 是异步路径里的外层 Engine。

源码里类定义是：

```python
class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""
```

位置：`vllm/vllm/v1/engine/async_llm.py:70` 到 `vllm/vllm/v1/engine/async_llm.py:71`

这句注释也说明它是一个异步 wrapper：

```text
AsyncLLM 是 vLLM engine 的异步外壳。
```

### 4.1 AsyncLLM 也有 InputProcessor / OutputProcessor / EngineCoreClient

`AsyncLLM.__init__()` 里和 `LLMEngine` 类似，也创建了 `InputProcessor`：

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)
```

位置：`vllm/vllm/v1/engine/async_llm.py:132` 到 `vllm/vllm/v1/engine/async_llm.py:135`

然后创建 `OutputProcessor`：

```python
# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:137` 到 `vllm/vllm/v1/engine/async_llm.py:143`

再创建异步 multi-process client：

```python
# EngineCore (starts the engine in background process).
self.engine_core = EngineCoreClient.make_async_mp_client(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:145` 到 `vllm/vllm/v1/engine/async_llm.py:153`

所以 `AsyncLLM` 和 `LLMEngine` 的核心边界是一样的：

```text
外层：InputProcessor / OutputProcessor / EngineCoreClient
内层：EngineCore / Scheduler / Worker / ModelRunner
```

区别主要是接口形态：

```text
LLMEngine：
  调用方主动 step() 拉输出。

AsyncLLM：
  后台 output_handler 持续从 EngineCore 拉输出，
  调用方迭代 async generator 获取结果。
```

### 4.2 AsyncLLM.add_request() 是异步请求入口

异步 `add_request()` 中，普通输入也会先走 `InputProcessor`：

```python
request = self.input_processor.process_inputs(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:349` 到 `vllm/vllm/v1/engine/async_llm.py:360`

然后启动 output handler：

```python
self._run_output_handler()
```

位置：`vllm/vllm/v1/engine/async_llm.py:370` 到 `vllm/vllm/v1/engine/async_llm.py:373`

真正把请求送进 EngineCore 的代码在 `_add_request()`：

```python
self.output_processor.add_request(request, prompt, parent_req, index, queue)
await self.engine_core.add_request_async(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:400` 到 `vllm/vllm/v1/engine/async_llm.py:412`

异步请求入口可以概括为：

```text
AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → _run_output_handler()
  → RequestOutputCollector
  → OutputProcessor.add_request(..., queue)
  → EngineCoreClient.add_request_async()
  → 后台 EngineCoreProc
```

### 4.3 AsyncLLM.generate() 对外返回 async generator

`generate()` 的注释把异步路径说得很清楚：

```python
"""
Main function called by the API server to kick off a request
    * 1) Making an AsyncStream corresponding to the Request.
    * 2) Processing the Input.
    * 3) Adding the Request to the Detokenizer.
    * 4) Adding the Request to the EngineCore (separate process).

A separate output_handler loop runs in a background AsyncIO task,
pulling outputs from EngineCore and putting them into the
per-request AsyncStream.

The caller of generate() iterates the returned AsyncGenerator,
returning the RequestOutput back to the caller.
"""
```

位置：`vllm/vllm/v1/engine/async_llm.py:541` 到 `vllm/vllm/v1/engine/async_llm.py:554`

所以异步对外路径是：

```text
API server / caller
  → AsyncLLM.generate()
  → AsyncLLM.add_request()
  → EngineCoreClient.add_request_async()
  → output_handler 后台拉输出
  → RequestOutputCollector queue
  → async generator yield RequestOutput
```

### 4.4 AsyncLLM 的 output_handler 是异步输出消费循环

`_run_output_handler()` 里定义了后台 `output_handler()`。

它不断从 EngineCoreClient 拉取 `EngineCoreOutputs`：

```python
outputs = await engine_core.get_output_async()
```

位置：`vllm/vllm/v1/engine/async_llm.py:656` 到 `vllm/vllm/v1/engine/async_llm.py:660`

然后分片交给 `OutputProcessor`：

```python
processed_outputs = output_processor.process_outputs(
    outputs_slice, outputs.timestamp, iteration_stats
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:674` 到 `vllm/vllm/v1/engine/async_llm.py:677`

如果 stop string 触发了需要 abort 的请求，还会异步通知 EngineCore：

```python
await engine_core.abort_requests_async(processed_outputs.reqs_to_abort)
```

位置：`vllm/vllm/v1/engine/async_llm.py:685` 到 `vllm/vllm/v1/engine/async_llm.py:689`

所以异步输出路径是：

```text
EngineCoreProc
  → EngineCoreOutputs
  → AsyncMPClient.get_output_async()
  → AsyncLLM.output_handler
  → OutputProcessor.process_outputs()
  → RequestOutputCollector
  → AsyncLLM.generate() yield
```

---

## 5. Engine 和 EngineCore 的关系

`EngineCore` 是外层 Engine 的内部执行核心。

最关键的源码注释是：

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
```

位置：`vllm/vllm/v1/engine/core.py:96` 到 `vllm/vllm/v1/engine/core.py:97`

可以理解为：

```text
Engine = 外层引擎接口和输入输出编排层；
EngineCore = Engine 内部的执行主循环。
```

### 5.1 外层 Engine 不直接操作 Scheduler

外层 `LLMEngine` / `AsyncLLM` 不是直接把请求塞进 `Scheduler`。

它们传给 EngineCoreClient 的对象是 `EngineCoreRequest`。

`EngineCoreRequest` 定义在：

```python
class EngineCoreRequest(...):
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    ...
```

位置：`vllm/vllm/v1/engine/__init__.py:88` 到 `vllm/vllm/v1/engine/__init__.py:137`

之后由 EngineCore 转成 Scheduler 内部的 `Request`：

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
```

位置：`vllm/vllm/v1/engine/core.py:853` 到 `vllm/vllm/v1/engine/core.py:875`

再交给 Scheduler：

```python
self.scheduler.add_request(request)
```

位置：`vllm/vllm/v1/engine/core.py:372` 到 `vllm/vllm/v1/engine/core.py:403`

所以请求边界是：

```text
LLMEngine / AsyncLLM：
  EngineCoreRequest

EngineCore：
  EngineCoreRequest → Request

Scheduler：
  Request
```

### 5.2 EngineCore 负责内部 step 闭环

`EngineCore.step()` 的注释是：

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output.
```

位置：`vllm/vllm/v1/engine/core.py:479` 到 `vllm/vllm/v1/engine/core.py:484`

核心代码主线是：

```python
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
...
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
...
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:486` 到 `vllm/vllm/v1/engine/core.py:508`

这说明 EngineCore 的主职责是：

```text
schedule：
  调用 Scheduler.schedule() 得到 SchedulerOutput。

execute：
  调用 model_executor.execute_model()，把计划交给 Worker / ModelRunner。

make output：
  调用 Scheduler.update_from_output()，把 ModelRunnerOutput 回收成 EngineCoreOutputs。
```

也就是：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model()
  → Worker / ModelRunner
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
```

### 5.3 EngineCore 输出的还不是最终用户输出

`EngineCoreOutput` 是单个请求的内部增量输出：

```python
class EngineCoreOutput(...):
    request_id: str
    new_token_ids: list[int]
    ...
    finish_reason: FinishReason | None = None
```

位置：`vllm/vllm/v1/engine/__init__.py:175` 到 `vllm/vllm/v1/engine/__init__.py:205`

`EngineCoreOutputs` 是一批内部输出：

```python
class EngineCoreOutputs(...):
    engine_index: int = 0
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0
    ...
```

位置：`vllm/vllm/v1/engine/__init__.py:220` 到 `vllm/vllm/v1/engine/__init__.py:248`

它们还不是用户最终看到的 `RequestOutput`。

外层 Engine 还要通过 `OutputProcessor` 转换：

```text
EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

所以输出边界是：

```text
EngineCore：
  产出 EngineCoreOutputs

LLMEngine / AsyncLLM：
  消费 EngineCoreOutputs
  产出 RequestOutput / PoolingRequestOutput
```

---

## 6. EngineCoreClient 为什么夹在中间

外层 Engine 通常不是直接持有 `EngineCore`，而是持有 `EngineCoreClient`。

`EngineCoreClient` 的注释写得很明确：

```python
class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """
```

位置：`vllm/vllm/v1/engine/core_client.py:71` 到 `vllm/vllm/v1/engine/core_client.py:80`

也就是说，`EngineCoreClient` 的作用是屏蔽运行方式差异：

```text
InprocClient：
  EngineCore 和 LLMEngine 在同一个进程里。

SyncMPClient：
  LLMEngine 通过 ZMQ 和后台 EngineCoreProc 通信。

AsyncMPClient：
  AsyncLLM 通过 asyncio + ZMQ 和后台 EngineCoreProc 通信。
```

### 6.1 make_client 决定使用哪种 client

`EngineCoreClient.make_client()` 根据 `multiprocess_mode` 和 `asyncio_mode` 决定返回什么：

```python
if multiprocess_mode and asyncio_mode:
    return EngineCoreClient.make_async_mp_client(...)

if multiprocess_mode and not asyncio_mode:
    return SyncMPClient(...)

return InprocClient(...)
```

位置：`vllm/vllm/v1/engine/core_client.py:82` 到 `vllm/vllm/v1/engine/core_client.py:105`

所以外层 `LLMEngine` 只管调用：

```text
self.engine_core.add_request(...)
self.engine_core.get_output()
self.engine_core.abort_requests(...)
```

但背后可能是：

```text
直接方法调用；
也可能是 ZMQ 消息；
也可能是 asyncio + ZMQ。
```

### 6.2 InprocClient：同进程直接调用 EngineCore

`InprocClient` 的注释：

```python
InprocClient: client for in-process EngineCore.
...
* pushes EngineCoreRequest directly into the EngineCore
* pulls EngineCoreOutputs by stepping the EngineCore
```

位置：`vllm/vllm/v1/engine/core_client.py:276` 到 `vllm/vllm/v1/engine/core_client.py:284`

它初始化时直接创建 `EngineCore`：

```python
self.engine_core = EngineCore(*args, **kwargs)
```

位置：`vllm/vllm/v1/engine/core_client.py:286` 到 `vllm/vllm/v1/engine/core_client.py:287`

拉输出时直接调用 `step_fn()`：

```python
outputs, model_executed = self.engine_core.step_fn()
self.engine_core.post_step(model_executed=model_executed)
return outputs and outputs.get(0) or EngineCoreOutputs()
```

位置：`vllm/vllm/v1/engine/core_client.py:289` 到 `vllm/vllm/v1/engine/core_client.py:292`

添加请求时会先预处理，再加入 EngineCore：

```python
req, request_wave = self.engine_core.preprocess_add_request(request)
self.engine_core.add_request(req, request_wave)
```

位置：`vllm/vllm/v1/engine/core_client.py:297` 到 `vllm/vllm/v1/engine/core_client.py:299`

所以 Inproc 模式是：

```text
LLMEngine
  → InprocClient.add_request()
  → EngineCore.preprocess_add_request()
  → EngineCore.add_request()

LLMEngine.step()
  → InprocClient.get_output()
  → EngineCore.step_fn()
  → EngineCoreOutputs
```

### 6.3 MPClient：后台进程里的 EngineCore

`MPClient` 的注释：

```python
MPClient: base client for multi-proc EngineCore.
    EngineCore runs in a background process busy loop, getting
    new EngineCoreRequests and returning EngineCoreOutputs

    * pushes EngineCoreRequests via input_socket
    * pulls EngineCoreOutputs via output_socket
```

位置：`vllm/vllm/v1/engine/core_client.py:467` 到 `vllm/vllm/v1/engine/core_client.py:478`

后台进程里的对象是 `EngineCoreProc`：

```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
```

位置：`vllm/vllm/v1/engine/core.py:894` 到 `vllm/vllm/v1/engine/core.py:895`

所以多进程关系是：

```text
LLMEngine / AsyncLLM
  → SyncMPClient / AsyncMPClient
  → ZMQ input_socket
  → EngineCoreProc input_queue
  → EngineCore.step_fn()
  → output_queue
  → ZMQ output_socket
  → SyncMPClient / AsyncMPClient
  → OutputProcessor
```

### 6.4 EngineCoreClient 是不是 EngineCore？有没有 EngineCoreServer？

严格说：

```text
EngineCoreClient 不是 EngineCore 本体。
```

`EngineCoreClient` 是外层 Engine 访问内部 EngineCore 的客户端抽象。

源码里 `LLMEngine.__init__()` 的字段名容易让人误解：

```python
self.engine_core = EngineCoreClient.make_client(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:104` 到 `vllm/vllm/v1/engine/llm_engine.py:111`

这里变量名叫 `self.engine_core`，但实际创建的是 `EngineCoreClient`。

更准确的理解是：

```text
self.engine_core 这个字段代表“外层 Engine 用来访问 EngineCore 的入口”；
它的具体类型是 EngineCoreClient 的某个子类，
不一定是真正的 EngineCore 对象。
```

分两种情况看：

```text
1. in-process 模式

LLMEngine
  → InprocClient
  → InprocClient.engine_core = EngineCore(...)

这时 InprocClient 内部确实直接持有一个 EngineCore 对象。
所以从调用效果看，它像是在直接调用 EngineCore。
```

```text
2. multi-process 模式

LLMEngine / AsyncLLM
  → SyncMPClient / AsyncMPClient
  → ZMQ
  → 后台 EngineCoreProc
  → EngineCoreProc 继承 EngineCore，内部跑 busy loop

这时 Client 不持有 EngineCore 本体，
只通过 socket 和后台 EngineCoreProc 通信。
```

源码里没有一个叫 `EngineCoreServer` 的类。

如果按 client / server 视角类比，可以这样理解：

```text
EngineCoreClient：
  client / proxy，外层 Engine 通过它发送请求、拉取输出、调用 utility。

EngineCoreProc：
  server-side process wrapper，后台进程里的 EngineCore 包装器，
  负责 ZMQ 输入输出、input_queue / output_queue、busy loop 和 shutdown。

EngineCore：
  真正的内部执行核心，负责 schedule → execute → update → output。
```

所以更准确的对象关系是：

```text
外层 Engine：LLMEngine / AsyncLLM
  → 持有 EngineCoreClient
      → InprocClient：同进程直接持有 EngineCore
      → SyncMPClient / AsyncMPClient：通过 ZMQ 访问后台 EngineCoreProc
          → EngineCoreProc 继承 EngineCore
              → 内部执行 Scheduler + model_executor 闭环
```

一句话：

```text
EngineCoreClient 不是 EngineCore；
它是访问 EngineCore 的客户端桥接层。
源码中没有 EngineCoreServer，multi-process 模式下承担“服务端”角色的是 EngineCoreProc。
```

---

## 7. 外层 Engine 负责什么

综合 `LLMEngine` 和 `AsyncLLM`，外层 Engine 主要负责这些事情。

### 7.1 对外接口

同步路径：

```text
LLMEngine.add_request()
LLMEngine.step()
LLMEngine.abort_request()
LLMEngine.sleep()
LLMEngine.wake_up()
```

异步路径：

```text
AsyncLLM.generate()
AsyncLLM.encode()
AsyncLLM.add_request()
AsyncLLM.abort()
AsyncLLM.pause_generation()
AsyncLLM.resume_generation()
```

这些都是用户 / API server 更容易接触到的外层入口。

### 7.2 输入处理

外层 Engine 负责调用：

```text
InputProcessor.process_inputs()
```

把用户侧输入转换成：

```text
EngineCoreRequest
```

这一步处理的是外层输入协议，而不是内部调度。

### 7.3 输出处理

外层 Engine 负责调用：

```text
OutputProcessor.process_outputs()
```

把内部输出转换成：

```text
RequestOutput / PoolingRequestOutput
```

这一步处理 detokenize、stop string、输出聚合、异步 queue 推送等用户可见输出逻辑。

### 7.4 通过 EngineCoreClient 访问内部 EngineCore

外层 Engine 不关心 EngineCore 是同进程还是后台进程。

它只使用统一接口：

```text
add_request / add_request_async
get_output / get_output_async
abort_requests / abort_requests_async
profile / sleep / wake_up / reset / LoRA / collective_rpc
```

这些由 `EngineCoreClient` 转发到真正的 EngineCore / EngineCoreProc。

### 7.5 统计、tracing、profile、生命周期控制

`LLMEngine` 和 `AsyncLLM` 还负责一部分外层管理能力，例如：

```text
统计日志；
tracing 初始化；
profile 入口；
多模态 renderer cache 清理；
shutdown；
LoRA 控制接口转发；
sleep / wake_up 控制接口转发。
```

这些不是 `EngineCore.step()` 主执行链路本身，但属于外层 Engine 的编排职责。

---

## 8. EngineCore 负责什么

和外层 Engine 相比，`EngineCore` 的职责更靠内。

它负责：

```text
1. 创建 model_executor；
2. 初始化 / profile KV cache；
3. 创建 Scheduler；
4. 把 EngineCoreRequest 转成 Request；
5. 把 Request 交给 Scheduler；
6. 每轮 step 调用 Scheduler.schedule()；
7. 调用 model_executor.execute_model()；
8. 调用 Scheduler.update_from_output()；
9. 返回 EngineCoreOutputs；
10. 转发部分 profile / reset / sleep / wake_up / LoRA / collective_rpc 到 Scheduler 或 model_executor。
```

从执行主线看，EngineCore 是：

```text
schedule → execute → update → output
```

的闭环总控。

但它不负责：

```text
不直接接收用户原始 prompt；
不直接构造最终 RequestOutput；
不直接 detokenize；
不直接实现 token 级调度策略；
不直接执行模型 forward。
```

这些分别属于：

```text
InputProcessor / OutputProcessor；
Scheduler；
model_executor / Worker / ModelRunner。
```

---

## 9. 容易混淆的点

### 9.1 Engine 是不是等于 LLMEngine？

不完全等于。

可以说：

```text
LLMEngine 是一种具体的外层 Engine。
```

但不能说：

```text
Engine 永远只等于 LLMEngine。
```

因为异步路径中外层 Engine 是 `AsyncLLM`。

更准确是：

```text
Engine 是外层引擎体系的泛称；
LLMEngine 是同步实现 / 同步形态。
```

### 9.2 Engine 是不是等于 AsyncLLM？

也不完全等于。

可以说：

```text
AsyncLLM 是一种具体的异步外层 Engine。
```

但 `Engine` 这个概念还可以覆盖同步 `LLMEngine`。

### 9.3 Engine 是不是等于 EngineCore？

不是。

源码直接说：

```text
EngineCore 是 vLLM Engine 的 inner loop。
```

所以关系是：

```text
Engine 更靠外；
EngineCore 更靠内。
```

更直观地说：

```text
Engine：
  负责用户输入输出、同步/异步接口、请求和输出编排。

EngineCore：
  负责内部调度执行闭环。
```

### 9.4 LLMEngine.self.engine_core 是不是一定是真 EngineCore？

不一定。

`LLMEngine.__init__()` 里：

```python
self.engine_core = EngineCoreClient.make_client(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:104` 到 `vllm/vllm/v1/engine/llm_engine.py:111`

所以 `self.engine_core` 这个字段名容易误导。

它实际是一个 `EngineCoreClient`：

```text
in-process：
  client 内部直接持有 EngineCore。

multi-process：
  client 通过 ZMQ 访问后台 EngineCoreProc。
```

### 9.5 InputProcessor / OutputProcessor 是 EngineCore 的一部分吗？

不是。

它们属于外层 Engine。

从代码位置看，`LLMEngine` / `AsyncLLM` 创建它们：

```text
LLMEngine / AsyncLLM
  → InputProcessor
  → OutputProcessor
  → EngineCoreClient
```

而 `EngineCore` 内部主要创建：

```text
model_executor
Scheduler
StructuredOutputManager
batch_queue
```

所以边界是：

```text
InputProcessor / OutputProcessor：
  外层 Engine 的输入输出适配组件。

EngineCore：
  内层执行闭环组件。
```

### 9.6 EngineCoreOutputs 是不是最终用户输出？

不是。

`EngineCoreOutputs` 是内部输出协议。

最终用户看到的是：

```text
RequestOutput / PoolingRequestOutput
```

转换关系是：

```text
EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

## 10. 最关键的层级图

### 10.1 总体层级

```text
用户 / API server / LLM
  ↓
LLMEngine / AsyncLLM
  ↓
InputProcessor
  ↓
EngineCoreRequest
  ↓
EngineCoreClient
  ↓
EngineCore / EngineCoreProc
  ↓
Scheduler + model_executor
  ↓
Worker / ModelRunner
  ↓
EngineCoreOutputs
  ↓
EngineCoreClient
  ↓
OutputProcessor
  ↓
RequestOutput / PoolingRequestOutput
```

### 10.2 同步路径

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → EngineCore.add_request()
  → Scheduler.add_request()

LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCore.step_fn()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

### 10.3 异步路径

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → RequestOutputCollector
  → OutputProcessor.add_request(..., queue)
  → EngineCoreClient.add_request_async()
  → 后台 EngineCoreProc

AsyncLLM.output_handler
  → EngineCoreClient.get_output_async()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutputCollector queue
  → generate() async generator yield
```

### 10.4 内部 EngineCore 路径

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model()
  → Worker / ModelRunner
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
```

---

## 11. 从“回答问题”的角度总结

如果问：

```text
Engine 到底是不是 LLMEngine 或者 AsyncLLM？
它们之间是什么关系？
```

可以回答：

```text
Engine 在这里更像 vLLM 外层引擎体系的泛称，
不是一个必须唯一对应到某个具体类的名字。

在 V1 代码里，常见的外层 Engine 形态有两个：
同步路径里的 LLMEngine，以及异步路径里的 AsyncLLM。
它们都负责对外接收请求、调用 InputProcessor 生成 EngineCoreRequest、
通过 EngineCoreClient 把请求送进 EngineCore、再把 EngineCoreOutputs
交给 OutputProcessor 转成 RequestOutput / PoolingRequestOutput。

EngineCore 则不是外层 Engine 本身，而是 Engine 的内部执行主循环。
它负责 schedule → execute → update → output 的内部闭环：
调用 Scheduler.schedule() 生成 SchedulerOutput，
调用 model_executor / Worker / ModelRunner 执行模型，
再调用 Scheduler.update_from_output() 生成 EngineCoreOutputs。
```

可以压缩成一句话：

```text
LLMEngine / AsyncLLM 是外层 Engine 的具体形态；
EngineCore 是它们背后的内部执行核心；
EngineCoreClient 是两者之间的桥。
```

最小心智模型：

```text
Engine = 外层输入输出编排层。
EngineCore = 内层执行闭环。
LLMEngine = 同步 Engine。
AsyncLLM = 异步 Engine。
EngineCoreClient = 外层 Engine 访问内层 EngineCore 的客户端桥接层。
```
