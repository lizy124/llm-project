# 03. AsyncLLM 负责什么？

源码位置：`vllm/v1/engine/async_llm.py`

相关源码：

- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/engine/input_processor.py`
- `vllm/v1/engine/output_processor.py`
- `vllm/v1/engine/__init__.py`

本问题关注：异步接口里的 `AsyncLLM` 如何作为外层 Engine，异步地提交请求、消费输出，并对外提供 async generator 风格的结果。

---

## 1. 一句话回答

`AsyncLLM` 是 vLLM V1 异步路径里的外层 Engine。

源码定义是：

```python
class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""
```

位置：`vllm/v1/engine/async_llm.py:70` 到 `vllm/v1/engine/async_llm.py:71`

可以理解为：

```text
AsyncLLM = vLLM Engine 的异步外壳。
```

它本身不直接做模型 forward，也不直接做 Scheduler 的 token 级调度。

它主要负责：

```text
1. 接收 API server / 用户侧异步请求；
2. 调用 InputProcessor 把输入转成 EngineCoreRequest；
3. 创建 RequestOutputCollector 作为每个请求的异步输出队列；
4. 通过 AsyncMPClient / EngineCoreClient 把请求发给后台 EngineCoreProc；
5. 启动后台 output_handler 持续拉取 EngineCoreOutputs；
6. 调用 OutputProcessor 把 EngineCoreOutput 转成 RequestOutput / PoolingRequestOutput；
7. 把结果放回对应 RequestOutputCollector；
8. 通过 async generator 返回给调用方。
```

一句话：

```text
AsyncLLM 是异步外层 Engine；
EngineCore 负责内部执行闭环，AsyncLLM 负责异步请求入口、异步输出分发和用户可见结果流。
```

---

## 2. AsyncLLM 的定位

`AsyncLLM` 和同步路径里的 `LLMEngine` 处在同一层：都是外层 Engine。

它们的共同边界是：

```text
外层 Engine：
  InputProcessor / OutputProcessor / EngineCoreClient

内部 EngineCore：
  Scheduler / model_executor / Worker / ModelRunner
```

不同点在于输出驱动方式：

```text
LLMEngine：
  调用方显式调用 step()，同步地从 EngineCore 拉取一批输出。

AsyncLLM：
  后台 output_handler 持续 await engine_core.get_output_async()，
  再把输出投递到每个请求自己的 RequestOutputCollector。
```

所以 `AsyncLLM` 可以理解为：

```text
LLMEngine 的异步化外层形态，
但不是简单把 step() 改成 await；
它额外引入了后台输出循环和每请求输出队列。
```

最小层级图：

```text
API server / caller
  → AsyncLLM.generate() / encode()
  → AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → RequestOutputCollector
  → OutputProcessor.add_request(..., queue)
  → AsyncMPClient.add_request_async()
  → EngineCoreProc / EngineCore
  → Scheduler / Worker / ModelRunner
  → EngineCoreOutputs
  → AsyncMPClient.get_output_async()
  → AsyncLLM._run_output_handler() 创建的后台 output_handler task
  → OutputProcessor.process_outputs()
  → RequestOutputCollector.put(...)
  → async generator yield
```

---

## 3. 初始化阶段

`AsyncLLM.__init__()` 的参数说明了它是外层 Engine：

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    log_requests: bool = True,
    start_engine_loop: bool = True,
    stat_loggers: list[StatLoggerFactory] | None = None,
    aggregate_engine_logging: bool = False,
    client_addresses: dict[str, Any] | None = None,
    client_count: int = 1,
    client_index: int = 0,
) -> None:
```

位置：`vllm/v1/engine/async_llm.py:73` 到 `vllm/v1/engine/async_llm.py:87`

它接收的是：

```text
vllm_config：整体配置；
executor_class：内部 EngineCore 创建 executor 时使用；
log_stats / stat_loggers：外层统计；
client_count / client_index：多 frontend client 输出路由；
client_addresses：多进程 EngineCore 通信地址。
```

### 3.1 注册配置序列化

初始化开头会注册 transformer config 的序列化能力：

```python
maybe_register_config_serialize_by_value()
```

位置：`vllm/v1/engine/async_llm.py:107` 到 `vllm/v1/engine/async_llm.py:108`

这是因为异步 V1 通常要通过多进程 / ZMQ 把配置、请求等对象发到后台 EngineCoreProc，序列化能力是基础设施。

### 3.2 保存配置和 tracing 设置

`AsyncLLM` 保存全局配置、模型配置和 observability 配置：

```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.observability_config = vllm_config.observability_config
```

位置：`vllm/v1/engine/async_llm.py:110` 到 `vllm/v1/engine/async_llm.py:112`

如果配置了 OTLP tracing endpoint，会初始化 tracing：

```python
tracing_endpoint = self.observability_config.otlp_traces_endpoint
if tracing_endpoint is not None:
    init_tracer("vllm.llm_engine", tracing_endpoint)
```

位置：`vllm/v1/engine/async_llm.py:114` 到 `vllm/v1/engine/async_llm.py:116`

这说明 `AsyncLLM` 还负责外层观测能力的接入。

### 3.3 创建 renderer

```python
self.renderer = renderer = renderer_from_config(self.vllm_config)
```

位置：`vllm/v1/engine/async_llm.py:132`

renderer 负责和 tokenizer、多模态渲染等外层输入输出处理相关的工作。

### 3.4 创建 InputProcessor

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)
```

位置：`vllm/v1/engine/async_llm.py:134` 到 `vllm/v1/engine/async_llm.py:135`

注释很关键：

```text
InputProcessor：EngineInput → EngineCoreRequest
```

也就是说，`AsyncLLM` 不直接把用户输入交给 EngineCore，而是先在外层处理成 EngineCore 能接受的内部请求协议。

### 3.5 创建 OutputProcessor

```python
# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(
    renderer.tokenizer,
    log_stats=self.log_stats,
    stream_interval=self.vllm_config.scheduler_config.stream_interval,
    tracing_enabled=tracing_endpoint is not None,
)
```

位置：`vllm/v1/engine/async_llm.py:137` 到 `vllm/v1/engine/async_llm.py:143`

注释同样关键：

```text
OutputProcessor：EngineCoreOutputs → RequestOutput
```

也就是说，`EngineCore` 返回的还不是最终用户输出，最终输出由 `AsyncLLM` 侧的 `OutputProcessor` 转换。

### 3.6 创建 EngineCoreClient：AsyncMPClient

`AsyncLLM` 使用的是异步多进程 client：

```python
# EngineCore (starts the engine in background process).
self.engine_core = EngineCoreClient.make_async_mp_client(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
    client_addresses=client_addresses,
    client_count=client_count,
    client_index=client_index,
)
```

位置：`vllm/v1/engine/async_llm.py:145` 到 `vllm/v1/engine/async_llm.py:153`

这说明异步路径默认把 EngineCore 放在后台进程里跑。

因此：

```text
AsyncLLM.engine_core 字段名虽然叫 engine_core，
但实际是 EngineCoreClient / AsyncMPClient，
不是直接的 EngineCore 对象。
```

### 3.7 初始化统计日志

如果开启 stats，`AsyncLLM` 会创建 `StatLoggerManager`：

```python
self.logger_manager = StatLoggerManager(
    vllm_config=vllm_config,
    engine_idxs=self.engine_core.engine_ranks_managed,
    custom_stat_loggers=custom_stat_loggers,
    enable_default_loggers=log_stats,
    client_count=client_count,
    aggregate_engine_logging=aggregate_engine_logging,
)
self.logger_manager.log_engine_initialized()
```

位置：`vllm/v1/engine/async_llm.py:155` 到 `vllm/v1/engine/async_llm.py:166`

这里用到了 `engine_core.engine_ranks_managed`，说明异步路径可以管理多个后台 EngineCore rank，尤其是 data parallel 场景。

### 3.8 准备 output_handler

初始化时：

```python
self.output_handler: asyncio.Task | None = None
try:
    # Start output handler eagerly if we are in the asyncio eventloop.
    asyncio.get_running_loop()
    self._run_output_handler()
except RuntimeError:
    pass
```

位置：`vllm/v1/engine/async_llm.py:170` 到 `vllm/v1/engine/async_llm.py:176`

含义是：

```text
如果当前已经在 asyncio event loop 中，就立即启动 output_handler；
如果还没有 event loop，就先不启动，等第一次 add_request() 时再启动。
```

后面 `add_request()` 也会调用 `_run_output_handler()`，确保输出循环一定启动。

### 3.9 profiler

如果配置了 torch profiler，并且没有忽略 frontend，`AsyncLLM` 会创建前端 CPU profiler：

```python
self.profiler = torch.profiler.profile(...)
```

位置：`vllm/v1/engine/async_llm.py:178` 到 `vllm/v1/engine/async_llm.py:200`

这也是外层 Engine 的观测和调试职责。

### 3.10 初始化链路总结

```text
AsyncLLM.__init__()
  → 保存 vllm_config / model_config / observability_config
  → 初始化 tracing / stats 配置
  → renderer_from_config()
  → InputProcessor(vllm_config, renderer)
  → OutputProcessor(tokenizer, ...)
  → EngineCoreClient.make_async_mp_client(...)
  → 创建 logger_manager
  → 准备 output_handler
  → 可选初始化 frontend profiler
```

---

## 4. from_vllm_config / from_engine_args

`AsyncLLM` 有两个常用构造入口。

### 4.1 from_vllm_config

```python
@classmethod
def from_vllm_config(...):
    return cls(
        vllm_config=vllm_config,
        executor_class=Executor.get_class(vllm_config),
        ...
    )
```

位置：`vllm/v1/engine/async_llm.py:202` 到 `vllm/v1/engine/async_llm.py:229`

这里的关键点是：

```text
外层 AsyncLLM 根据 vllm_config 选择 executor_class；
但 executor 的真正创建发生在 EngineCore 里。
```

### 4.2 from_engine_args

```python
vllm_config = engine_args.create_engine_config(usage_context)
executor_class = Executor.get_class(vllm_config)
```

位置：`vllm/v1/engine/async_llm.py:231` 到 `vllm/v1/engine/async_llm.py:254`

这条路径用于从 CLI / server 参数构造异步 Engine：

```text
AsyncEngineArgs
  → create_engine_config()
  → Executor.get_class(vllm_config)
  → AsyncLLM(...)
```

---

## 5. 异步请求入口：add_request()

`AsyncLLM.add_request()` 是异步请求进入外层 Engine 的核心入口。

入口定义：

```python
async def add_request(
    self,
    request_id: str,
    prompt: EngineCoreRequest | PromptType | EngineInput | AsyncGenerator[StreamingInput, None],
    params: SamplingParams | PoolingParams,
    ...
) -> RequestOutputCollector:
```

位置：`vllm/v1/engine/async_llm.py:280` 到 `vllm/v1/engine/async_llm.py:297`

它返回的不是 `RequestOutput`，而是：

```text
RequestOutputCollector
```

这个 collector 是每个请求自己的异步输出队列，后续 `generate()` / `encode()` 会从里面取结果并 yield。

### 5.1 Engine dead 检查

入口处先检查 Engine 是否已经出错：

```python
if self.errored:
    raise EngineDeadError()
```

位置：`vllm/v1/engine/async_llm.py:300` 到 `vllm/v1/engine/async_llm.py:301`

`errored` 的定义是：

```python
return self.engine_core.resources.engine_dead or not self.is_running
```

位置：`vllm/v1/engine/async_llm.py:1053` 到 `vllm/v1/engine/async_llm.py:1055`

这说明 `AsyncLLM` 会把后台 EngineCoreProc 的死亡状态传播到前端。

### 5.2 pooling / kv sharing 校验

```python
is_pooling = isinstance(params, PoolingParams)
```

位置：`vllm/v1/engine/async_llm.py:303`

如果打开 `kv_sharing_fast_prefill` 且请求需要 prompt logprobs，会报错：

```python
if self.vllm_config.cache_config.kv_sharing_fast_prefill and not is_pooling and params.prompt_logprobs:
    raise ValueError(...)
```

位置：`vllm/v1/engine/async_llm.py:305` 到 `vllm/v1/engine/async_llm.py:314`

这是外层 Engine 做请求合法性检查的例子。

### 5.3 streaming input 特殊路径

如果 prompt 是 `AsyncGenerator`，说明是 streaming input：

```python
if isinstance(prompt, AsyncGenerator):
    ...
    return await self._add_streaming_input_request(...)
```

位置：`vllm/v1/engine/async_llm.py:316` 到 `vllm/v1/engine/async_llm.py:331`

普通请求则继续走 `InputProcessor.process_inputs()`。

### 5.4 普通输入转 EngineCoreRequest

如果传入的是已经构造好的 `EngineCoreRequest`，会直接使用，但这条路径已经标记为 deprecated：

```python
if isinstance(prompt, EngineCoreRequest):
    logger.warning_once(...)
    request = prompt
```

位置：`vllm/v1/engine/async_llm.py:333` 到 `vllm/v1/engine/async_llm.py:347`

否则走输入处理：

```python
request = self.input_processor.process_inputs(
    request_id,
    prompt,
    params,
    supported_tasks=await self.get_supported_tasks(),
    arrival_time=arrival_time,
    lora_request=lora_request,
    tokenization_kwargs=tokenization_kwargs,
    trace_headers=trace_headers,
    priority=priority,
    data_parallel_rank=data_parallel_rank,
)
```

位置：`vllm/v1/engine/async_llm.py:349` 到 `vllm/v1/engine/async_llm.py:360`

这里完成：

```text
用户输入 / PromptType / EngineInput
  → EngineCoreRequest
```

同时提取 prompt_text：

```python
prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)
```

位置：`vllm/v1/engine/async_llm.py:361`

### 5.5 reasoning 字段补充

```python
if reasoning_ended is not None:
    request.reasoning_ended = reasoning_ended
if reasoning_parser_kwargs is not None:
    request.reasoning_parser_kwargs = reasoning_parser_kwargs
```

位置：`vllm/v1/engine/async_llm.py:363` 到 `vllm/v1/engine/async_llm.py:366`

这说明 `AsyncLLM` 还会补充部分外层请求元信息。

### 5.6 分配内部 request id

```python
self.input_processor.assign_request_id(request)
```

位置：`vllm/v1/engine/async_llm.py:368`

这一步用于处理内部 request_id 和用户传入 external request id 的关系。

### 5.7 确保 output_handler 启动

```python
self._run_output_handler()
```

位置：`vllm/v1/engine/async_llm.py:370` 到 `vllm/v1/engine/async_llm.py:373`

注释说明：

```text
第一次 add_request() 时启动 output_handler，
这样 __init__ 可以在 event loop 之前调用，便于 OpenAI server 处理启动失败。
```

### 5.8 创建 RequestOutputCollector

```python
queue = RequestOutputCollector(params.output_kind, request.request_id)
```

位置：`vllm/v1/engine/async_llm.py:375` 到 `vllm/v1/engine/async_llm.py:377`

这个 queue 是异步输出路径的关键。

它把：

```text
后台 output_handler 生产的 RequestOutput
```

和：

```text
generate() / encode() async generator 消费的 RequestOutput
```

连接起来。

### 5.9 单请求与 n>1 fan out

如果是 pooling，或者 `params.n == 1`，直接添加请求：

```python
if is_pooling or params.n == 1:
    await self._add_request(request, prompt_text, None, 0, queue)
    return queue
```

位置：`vllm/v1/engine/async_llm.py:381` 到 `vllm/v1/engine/async_llm.py:383`

如果 `n > 1`，会创建 `ParentRequest` 并 fan out 成多个 child request：

```python
parent_request = ParentRequest(request)
for idx in range(parent_params.n):
    request_id, child_params = parent_request.get_child_info(idx)
    child_request = request if idx == parent_params.n - 1 else copy(request)
    child_request.request_id = request_id
    child_request.sampling_params = child_params
    await self._add_request(child_request, prompt_text, parent_request, idx, queue)
```

位置：`vllm/v1/engine/async_llm.py:385` 到 `vllm/v1/engine/async_llm.py:397`

也就是说，对用户来说是一个请求，但内部会拆成多个 child request 进入 EngineCore。

### 5.10 add_request 主链路总结

```text
AsyncLLM.add_request()
  → 检查 engine 是否 alive
  → 判断 pooling / generation
  → streaming input 特殊路径
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → 补充 reasoning 字段
  → InputProcessor.assign_request_id()
  → _run_output_handler()
  → RequestOutputCollector
  → _add_request()
      → OutputProcessor.add_request(..., queue)
      → EngineCoreClient.add_request_async(request)
  → 返回 RequestOutputCollector
```

---

## 6. _add_request()：同时登记 OutputProcessor 和 EngineCore

`_add_request()` 是真正把请求登记到输出侧和执行侧的方法：

```python
async def _add_request(
    self,
    request: EngineCoreRequest,
    prompt: str | None,
    parent_req: ParentRequest | None,
    index: int,
    queue: RequestOutputCollector,
):
    # Add the request to OutputProcessor (this process).
    self.output_processor.add_request(request, prompt, parent_req, index, queue)

    # Add the EngineCoreRequest to EngineCore (separate process).
    await self.engine_core.add_request_async(request)
```

位置：`vllm/v1/engine/async_llm.py:400` 到 `vllm/v1/engine/async_llm.py:412`

这里有一个非常重要的边界：

```text
OutputProcessor 在 AsyncLLM 所在进程；
EngineCore 在单独后台进程；
请求需要同时登记到两边。
```

为什么先 `OutputProcessor.add_request()`？

因为后续 output_handler 收到 `EngineCoreOutput` 时，需要能根据 request_id 找到对应的输出状态和 queue。

可以理解为：

```text
OutputProcessor.add_request()：
  先在前端准备好“如何组装输出、往哪个 queue 放”。

EngineCoreClient.add_request_async()：
  再把请求送进后台 EngineCore 执行。
```

---

## 7. generate()：对外的文本生成异步接口

`generate()` 是 API server 常用的入口。

当前实现中，`generate()` 的实际职责可以概括为：

```text
generate() 实际先调用 add_request() 创建 RequestOutputCollector；
_add_request() 先注册到 OutputProcessor，再 await engine_core.add_request_async(request)；
后台 output_handler 把处理后的 RequestOutput 放入 collector；
generate() 从 collector 中取出并 yield RequestOutput。
```

关键位置：`vllm/v1/engine/async_llm.py:557`、`vllm/v1/engine/async_llm.py:375`、`vllm/v1/engine/async_llm.py:409`

### 7.1 generate() 先调用 add_request()

```python
q = await self.add_request(
    request_id,
    prompt,
    sampling_params,
    ...
)
```

位置：`vllm/v1/engine/async_llm.py:557` 到 `vllm/v1/engine/async_llm.py:571`

这里的 `q` 就是 `RequestOutputCollector`。

### 7.2 generate() 从 queue 中取输出并 yield

```python
finished = False
while not finished:
    out = q.get_nowait() or await q.get()
    assert isinstance(out, RequestOutput)
    finished = out.finished
    if out is not STREAM_FINISHED:
        yield out
```

位置：`vllm/v1/engine/async_llm.py:573` 到 `vllm/v1/engine/async_llm.py:586`

也就是说，`generate()` 自己不直接从 EngineCore 拿输出。

它消费的是：

```text
output_handler → OutputProcessor → RequestOutputCollector
```

已经处理好的用户可见输出。

### 7.3 generate() 的取消与错误处理

如果客户端断开、generator 被取消或垃圾回收，会 abort 请求：

```python
except (asyncio.CancelledError, GeneratorExit):
    if q is not None:
        await self.abort(q.request_id, internal=True)
    ...
    raise
```

位置：`vllm/v1/engine/async_llm.py:588` 到 `vllm/v1/engine/async_llm.py:596`

如果 Engine 已经死了，直接传播 `EngineDeadError`：

```python
except EngineDeadError:
    ...
    raise
```

位置：`vllm/v1/engine/async_llm.py:598` 到 `vllm/v1/engine/async_llm.py:602`

如果是输入流错误，会 abort 并抛出原始 cause：

```python
except InputStreamError as e:
    if q is not None:
        await self.abort(q.request_id, internal=True)
    ...
    raise e.cause from e
```

位置：`vllm/v1/engine/async_llm.py:610` 到 `vllm/v1/engine/async_llm.py:616`

其它异常会包装成 `EngineGenerateError`：

```python
except Exception as e:
    if q is not None:
        await self.abort(q.request_id, internal=True)
    ...
    raise EngineGenerateError() from e
```

位置：`vllm/v1/engine/async_llm.py:618` 到 `vllm/v1/engine/async_llm.py:632`

最后会关闭 collector：

```python
finally:
    if q is not None:
        q.close()
```

位置：`vllm/v1/engine/async_llm.py:633` 到 `vllm/v1/engine/async_llm.py:635`

### 7.4 generate() 主链路

```text
AsyncLLM.generate()
  → add_request()
  → RequestOutputCollector q
  → while not finished:
      q.get_nowait() or await q.get()
      yield RequestOutput
  → cancelled / error 时 abort
  → finally close q
```

注意：

```text
generate() 是用户输出消费端；
output_handler 是用户输出生产端；
两者通过 RequestOutputCollector 解耦。
```

---

## 8. encode()：异步 pooling / embedding 类接口

`encode()` 和 `generate()` 类似，但面向 pooling 输出：

```python
async def encode(
    self,
    prompt: PromptType | EngineInput,
    pooling_params: PoolingParams,
    request_id: str,
    ...
) -> AsyncGenerator[PoolingRequestOutput, None]:
```

位置：`vllm/v1/engine/async_llm.py:803` 到 `vllm/v1/engine/async_llm.py:813`

注释也强调它会：

```text
处理输入；
添加请求到 EngineCore；
由后台 output_handler 拉取输出；
调用方迭代 async generator 获取 PoolingRequestOutput。
```

位置：`vllm/v1/engine/async_llm.py:814` 到 `vllm/v1/engine/async_llm.py:826`

核心逻辑：

```python
q = await self.add_request(..., pooling_params, ...)
...
out = q.get_nowait() or await q.get()
assert isinstance(out, PoolingRequestOutput)
finished = out.finished
yield out
```

位置：`vllm/v1/engine/async_llm.py:828` 到 `vllm/v1/engine/async_llm.py:852`

所以：

```text
generate()：返回 RequestOutput
encode()：返回 PoolingRequestOutput
```

但它们共享同一套异步 Engine 主链路。

---

## 9. streaming input 请求

`AsyncLLM` 支持一种特殊输入：

```text
prompt 是 AsyncGenerator[StreamingInput, None]
```

入口在 `add_request()`：

```python
if isinstance(prompt, AsyncGenerator):
    ...
    return await self._add_streaming_input_request(...)
```

位置：`vllm/v1/engine/async_llm.py:316` 到 `vllm/v1/engine/async_llm.py:331`

### 9.1 参数限制

streaming input 会先校验 sampling params：

```python
self._validate_streaming_input_sampling_params(sampling_params)
```

位置：`vllm/v1/engine/async_llm.py:429`

限制逻辑是：

```python
if (
    not isinstance(params, SamplingParams)
    or params.n > 1
    or params.output_kind == RequestOutputKind.FINAL_ONLY
    or params.stop
):
    raise ValueError(...)
```

位置：`vllm/v1/engine/async_llm.py:503` 到 `vllm/v1/engine/async_llm.py:517`

也就是说，streaming input 当前不支持：

```text
pooling；
n > 1；
FINAL_ONLY；
stop strings。
```

### 9.2 创建 final request 作为结束信号

streaming input 会先创建一个 `final_req`：

```python
final_req = self.input_processor.process_inputs(
    request_id=request_id,
    prompt=TokensPrompt(prompt_token_ids=[0]),
    params=sampling_params,
    **inputs,
)
self.input_processor.assign_request_id(final_req)
internal_req_id = final_req.request_id
```

位置：`vllm/v1/engine/async_llm.py:445` 到 `vllm/v1/engine/async_llm.py:454`

这个 final request 既用于校验，也用于 input stream 关闭后的 finished signal。

### 9.3 每个 input chunk 都会变成一个 EngineCoreRequest

内部 `handle_inputs()` 会遍历 input stream：

```python
async for input_chunk in input_stream:
    ...
    req = self.input_processor.process_inputs(
        request_id=internal_req_id,
        prompt=input_chunk.prompt,
        params=sp,
        resumable=True,
        **inputs,
    )
```

位置：`vllm/v1/engine/async_llm.py:458` 到 `vllm/v1/engine/async_llm.py:474`

注意这里：

```text
request_id 使用同一个 internal_req_id；
resumable=True；
表示同一个请求可以持续追加输入。
```

然后发送到 EngineCore：

```python
await self._add_request(req, prompt_text, None, 0, queue)
```

位置：`vllm/v1/engine/async_llm.py:480` 到 `vllm/v1/engine/async_llm.py:483`

### 9.4 input stream 结束后发送 final request

```python
finally:
    ...
    if not cancelled:
        await self._add_request(final_req, None, None, 0, queue)
```

位置：`vllm/v1/engine/async_llm.py:490` 到 `vllm/v1/engine/async_llm.py:495`

这表示输入流结束后，会发送一个带 dummy token 的 final request / 结束信号 request，告诉 EngineCore 输入已经结束。这个 dummy token 不作为真实输入 chunk 追加。

### 9.5 streaming input 链路

```text
AsyncLLM.add_request(prompt=AsyncGenerator)
  → _add_streaming_input_request()
  → 创建 final_req / RequestOutputCollector
  → create_task(handle_inputs())
      → async for input_chunk
          → InputProcessor.process_inputs(..., resumable=True)
          → _add_request(req)
      → stream 结束后 _add_request(final_req)
  → output_handler 正常处理输出
  → generate() 从 collector yield RequestOutput
```

---

## 10. 后台输出处理：_run_output_handler()

`AsyncLLM` 最核心的异步机制是 `output_handler`。

入口：

```python
def _run_output_handler(self):
    """Background loop: pulls from EngineCore and pushes to AsyncStreams."""
```

位置：`vllm/v1/engine/async_llm.py:637` 到 `vllm/v1/engine/async_llm.py:638`

这句注释可以直接翻译成：

```text
后台循环：从 EngineCore 拉输出，并推送到每个请求的异步流。
```

### 10.1 防止重复启动

```python
if self.output_handler is not None:
    return
```

位置：`vllm/v1/engine/async_llm.py:640` 到 `vllm/v1/engine/async_llm.py:641`

也就是说，一个 `AsyncLLM` 只有一个 output_handler task。

### 10.2 避免 task 强引用 AsyncLLM

```python
engine_core = self.engine_core
output_processor = self.output_processor
log_stats = self.log_stats
...
renderer = self.renderer
chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
```

位置：`vllm/v1/engine/async_llm.py:643` 到 `vllm/v1/engine/async_llm.py:654`

注释说明：

```text
避免 task 持有 AsyncLLM 对象的循环引用，影响回收和清理。
```

### 10.3 从 EngineCore 异步拉输出

后台循环中第一步：

```python
outputs = await engine_core.get_output_async()
num_outputs = len(outputs.outputs)
```

位置：`vllm/v1/engine/async_llm.py:656` 到 `vllm/v1/engine/async_llm.py:661`

这一步拿到的是：

```text
EngineCoreOutputs
```

还不是 `RequestOutput`。

### 10.4 分片处理输出，避免阻塞 event loop

```python
engine_core_outputs = outputs.outputs
for start in range(0, num_outputs, chunk_size):
    end = start + chunk_size
    outputs_slice = engine_core_outputs[start:end]
```

位置：`vllm/v1/engine/async_llm.py:667` 到 `vllm/v1/engine/async_llm.py:673`

如果输出很多，分片处理可以避免长时间占用 event loop。

中间会主动让出 event loop：

```python
if end < num_outputs:
    await asyncio.sleep(0)
```

位置：`vllm/v1/engine/async_llm.py:681` 到 `vllm/v1/engine/async_llm.py:683`

### 10.5 调用 OutputProcessor 转换输出

```python
processed_outputs = output_processor.process_outputs(
    outputs_slice, outputs.timestamp, iteration_stats
)
```

位置：`vllm/v1/engine/async_llm.py:674` 到 `vllm/v1/engine/async_llm.py:677`

这里有一个和同步 `LLMEngine.step()` 不同的点：

```python
assert not processed_outputs.request_outputs
```

位置：`vllm/v1/engine/async_llm.py:678` 到 `vllm/v1/engine/async_llm.py:679`

原因是异步路径下，`OutputProcessor` 会把 RequestOutput 推到对应的 `RequestOutputCollector`，而不是直接返回给 `output_handler`。

所以：

```text
同步 LLMEngine.step()：
  OutputProcessor.process_outputs() 返回 request_outputs。

异步 `_run_output_handler()` 创建的后台 `output_handler()` task：
  OutputProcessor.process_outputs() 把结果放进 collector queue，
  processed_outputs.request_outputs 应为空。
```

### 10.6 stop string 触发的 abort

如果 `OutputProcessor` 发现 stop string 导致需要 abort 的请求：

```python
if processed_outputs.reqs_to_abort:
    await engine_core.abort_requests_async(processed_outputs.reqs_to_abort)
```

位置：`vllm/v1/engine/async_llm.py:685` 到 `vllm/v1/engine/async_llm.py:689`

这说明 stop string 的一部分处理在外层 OutputProcessor，但真正取消内部请求仍要通知 EngineCore。

### 10.7 更新 scheduler stats

```python
output_processor.update_scheduler_stats(outputs.scheduler_stats)
```

位置：`vllm/v1/engine/async_llm.py:691`

如果有 logger，会记录统计：

```python
logger_ref[0].record(
    engine_idx=outputs.engine_index,
    scheduler_stats=outputs.scheduler_stats,
    iteration_stats=iteration_stats,
    mm_cache_stats=renderer.stat_mm_cache(),
)
```

位置：`vllm/v1/engine/async_llm.py:693` 到 `vllm/v1/engine/async_llm.py:702`

### 10.8 output_handler 异常传播

如果 output_handler 抛异常：

```python
except Exception as e:
    logger.exception("AsyncLLM output_handler failed.")
    output_processor.propagate_error(e)
```

位置：`vllm/v1/engine/async_llm.py:703` 到 `vllm/v1/engine/async_llm.py:705`

这会把错误传播到所有相关请求的输出队列，让等待 `generate()` 的调用方感知失败。

### 10.9 创建后台 task

```python
self.output_handler = asyncio.create_task(output_handler())
```

位置：`vllm/v1/engine/async_llm.py:707`

### 10.10 output_handler 主链路

```text
AsyncLLM._run_output_handler()
  → create_task(output_handler)

output_handler loop:
  → await engine_core.get_output_async()
  → EngineCoreOutputs
  → 按 chunk 切分 outputs.outputs
  → OutputProcessor.process_outputs(outputs_slice, timestamp, stats)
  → RequestOutputCollector.put(...)
  → 必要时 engine_core.abort_requests_async(reqs_to_abort)
  → update_scheduler_stats
  → record stats
```

---

## 11. EngineCoreClient / AsyncMPClient 在异步路径中的作用

`AsyncLLM` 不是直接访问 `EngineCore`，而是通过 `EngineCoreClient.make_async_mp_client()` 创建异步 client。

`EngineCoreClient` 的注释说明：

```python
EngineCoreClient: subclasses handle different methods for pushing
    and pulling from the EngineCore for asyncio / multiprocessing.

Subclasses:
* InprocClient: In process EngineCore (for V0-style LLMEngine use)
* SyncMPClient: ZMQ + background proc EngineCore (for LLM)
* AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
```

位置：`vllm/v1/engine/core_client.py:71` 到 `vllm/v1/engine/core_client.py:80`

异步路径使用：

```text
AsyncMPClient：ZMQ + background proc EngineCore w/ asyncio
```

### 11.1 make_async_mp_client 选择 client

```python
if parallel_config.data_parallel_size > 1:
    if parallel_config.data_parallel_external_lb:
        return DPAsyncMPClient(*client_args)
    return DPLBAsyncMPClient(*client_args)
return AsyncMPClient(*client_args)
```

位置：`vllm/v1/engine/core_client.py:107` 到 `vllm/v1/engine/core_client.py:132`

含义是：

```text
普通异步路径：AsyncMPClient
DP 外部负载均衡：DPAsyncMPClient
DP 内部负载均衡：DPLBAsyncMPClient
```

所以 `AsyncLLM` 不需要自己处理 data parallel 选哪个 EngineCore，交给 client 层。

### 11.2 AsyncMPClient 的输出队列

`AsyncMPClient` 初始化时创建：

```python
self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()
```

位置：`vllm/v1/engine/core_client.py:971` 到 `vllm/v1/engine/core_client.py:973`

它还会在 event loop 中确保输出 socket 读取 task 启动：

```python
asyncio.get_running_loop()
self._ensure_output_queue_task()
```

位置：`vllm/v1/engine/core_client.py:974` 到 `vllm/v1/engine/core_client.py:981`

### 11.3 AsyncMPClient 从 ZMQ output_socket 读 EngineCoreOutputs

`_ensure_output_queue_task()` 里有一个 `process_outputs_socket()`：

```python
frames = await output_socket.recv_multipart(copy=False)
resources.validate_alive(frames)
outputs: EngineCoreOutputs = decoder.decode(frames)
```

位置：`vllm/v1/engine/core_client.py:1005` 到 `vllm/v1/engine/core_client.py:1010`

如果是普通输出，会放进 `outputs_queue`：

```python
if outputs.outputs or outputs.scheduler_stats:
    outputs_queue.put_nowait(outputs)
```

位置：`vllm/v1/engine/core_client.py:1042` 到 `vllm/v1/engine/core_client.py:1043`

如果出错，则把异常放入队列：

```python
except Exception as e:
    outputs_queue.put_nowait(e)
except asyncio.CancelledError:
    outputs_queue.put_nowait(EngineDeadError())
```

位置：`vllm/v1/engine/core_client.py:1044` 到 `vllm/v1/engine/core_client.py:1047`

### 11.4 AsyncMPClient.get_output_async()

`_run_output_handler()` 中的后台 `output_handler()` task 调用的 `get_output_async()` 在这里：

```python
async def get_output_async(self) -> EngineCoreOutputs:
    self._ensure_output_queue_task()
    ...
    outputs = await self.outputs_queue.get()
    if isinstance(outputs, Exception):
        raise self._format_exception(outputs) from None
    return outputs
```

位置：`vllm/v1/engine/core_client.py:1053` 到 `vllm/v1/engine/core_client.py:1062`

所以：

```text
AsyncLLM._run_output_handler() 创建的后台 output_handler task
  → AsyncMPClient.get_output_async()
  → outputs_queue.get()
  → EngineCoreOutputs 或异常
```

### 11.5 AsyncMPClient.add_request_async()

请求发送入口：

```python
async def add_request_async(self, request: EngineCoreRequest) -> None:
    request.client_index = self.client_index
    await self._send_input(EngineCoreRequestType.ADD, request)
    self._ensure_output_queue_task()
```

位置：`vllm/v1/engine/core_client.py:1121` 到 `vllm/v1/engine/core_client.py:1124`

这里的 `client_index` 很重要：

```text
它让后台 EngineCoreProc 返回输出时，能把 EngineCoreOutputs 路由回正确的 frontend client。
```

### 11.6 AsyncMPClient 的 utility 调用

控制类接口通过 `call_utility_async()` 发到 EngineCore：

```python
async def call_utility_async(self, method: str, *args) -> Any:
    return await self._call_utility_async(method, *args, engine=self.core_engine)
```

位置：`vllm/v1/engine/core_client.py:1101` 到 `vllm/v1/engine/core_client.py:1102`

内部消息类型是 `UTILITY`：

```python
message = (
    EngineCoreRequestType.UTILITY.value,
    *self.encoder.encode((self.client_index, call_id, method, args)),
)
```

位置：`vllm/v1/engine/core_client.py:1110` 到 `vllm/v1/engine/core_client.py:1113`

例如：

```text
profile_async → call_utility_async("profile", ...)
reset_mm_cache_async → call_utility_async("reset_mm_cache")
sleep_async → call_utility_async("sleep", ...)
wake_up_async → call_utility_async("wake_up", ...)
```

---

## 12. abort 请求

`AsyncLLM.abort()` 用于取消请求。

```python
async def abort(
    self, request_id: str | Iterable[str], internal: bool = False
) -> None:
```

位置：`vllm/v1/engine/async_llm.py:709` 到 `vllm/v1/engine/async_llm.py:711`

它先处理 request_id 列表：

```python
request_ids = (
    (request_id,) if isinstance(request_id, str) else as_list(request_id)
)
```

位置：`vllm/v1/engine/async_llm.py:714` 到 `vllm/v1/engine/async_llm.py:716`

然后先通知 `OutputProcessor`：

```python
all_request_ids = self.output_processor.abort_requests(request_ids, internal)
```

位置：`vllm/v1/engine/async_llm.py:717`

再通知 EngineCore：

```python
await self.engine_core.abort_requests_async(all_request_ids)
```

位置：`vllm/v1/engine/async_llm.py:718`

所以 abort 链路是：

```text
AsyncLLM.abort()
  → OutputProcessor.abort_requests()
  → EngineCoreClient.abort_requests_async()
  → AsyncMPClient 发送 ABORT 消息
  → EngineCoreProc
  → EngineCore.abort_requests()
  → Scheduler.finish_requests(... FINISHED_ABORTED)
```

为什么先处理 OutputProcessor？

```text
OutputProcessor 维护前端输出状态、collector queue、parent/child request 映射；
EngineCore 维护内部 Scheduler 状态；
两边都要清理。
```

---

## 13. pause / resume / sleep / wake_up

`AsyncLLM` 还暴露一组控制类异步接口。

### 13.1 pause_generation()

入口：

```python
async def pause_generation(
    self,
    *,
    mode: PauseMode = "abort",
    wait_for_inflight_requests: bool | None = None,
    clear_cache: bool = True,
) -> None:
```

位置：`vllm/v1/engine/async_llm.py:750` 到 `vllm/v1/engine/async_llm.py:756`

注释说明：

```text
暂停 generation 以便模型权重更新。
mode 处理 in-flight requests：abort / wait / keep。
新 generation / encoding 请求在 resume 前不会被调度。
```

位置：`vllm/v1/engine/async_llm.py:757` 到 `vllm/v1/engine/async_llm.py:773`

如果需要清 cache，会先清 renderer 的多模态 cache：

```python
if clear_cache:
    await self.renderer.clear_mm_cache_async()
```

位置：`vllm/v1/engine/async_llm.py:784` 到 `vllm/v1/engine/async_llm.py:785`

然后通知 EngineCore：

```python
await self.engine_core.pause_scheduler_async(mode=mode, clear_cache=clear_cache)
```

位置：`vllm/v1/engine/async_llm.py:786`

最后短暂 sleep，帮助 final outputs 有更直观的返回顺序：

```python
await asyncio.sleep(0.02)
```

位置：`vllm/v1/engine/async_llm.py:787` 到 `vllm/v1/engine/async_llm.py:793`

### 13.2 resume_generation()

```python
async def resume_generation(self) -> None:
    await self.engine_core.resume_scheduler_async()
```

位置：`vllm/v1/engine/async_llm.py:795` 到 `vllm/v1/engine/async_llm.py:797`

### 13.3 is_paused()

```python
async def is_paused(self) -> bool:
    return await self.engine_core.is_scheduler_paused_async()
```

位置：`vllm/v1/engine/async_llm.py:799` 到 `vllm/v1/engine/async_llm.py:801`

### 13.4 sleep()

```python
async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
    if level >= 1:
        await self.renderer.clear_mm_cache_async()
    await self.engine_core.sleep_async(level, mode)
```

位置：`vllm/v1/engine/async_llm.py:931` 到 `vllm/v1/engine/async_llm.py:934`

`sleep` 比 `pause_generation` 更偏执行资源管理：

```text
level 0：暂停调度；
level 1+：还会让 EngineCore / executor 管理 GPU memory。
```

### 13.5 wake_up()

```python
async def wake_up(self, tags: list[str] | None = None) -> None:
    await self.engine_core.wake_up_async(tags)
```

位置：`vllm/v1/engine/async_llm.py:939` 到 `vllm/v1/engine/async_llm.py:940`

### 13.6 控制类接口的共同特点

这些方法都体现同一条边界：

```text
AsyncLLM 暴露异步外层 API；
真正控制 Scheduler / model_executor 的动作在 EngineCore；
中间通过 AsyncMPClient 的 utility 消息转发。
```

---

## 14. profile / cache reset / LoRA / collective_rpc

`AsyncLLM` 还提供多个异步控制接口。

### 14.1 profile

```python
async def start_profile(self, profile_prefix: str | None = None) -> None:
    coros = [self.engine_core.profile_async(True, profile_prefix)]
    if self.profiler is not None:
        coros.append(asyncio.to_thread(self.profiler.start))
    await asyncio.gather(*coros)
```

位置：`vllm/v1/engine/async_llm.py:905` 到 `vllm/v1/engine/async_llm.py:909`

停止 profile：

```python
async def stop_profile(self) -> None:
    coros = [self.engine_core.profile_async(False)]
    if self.profiler is not None:
        coros.append(asyncio.to_thread(self.profiler.stop))
    await asyncio.gather(*coros)
```

位置：`vllm/v1/engine/async_llm.py:911` 到 `vllm/v1/engine/async_llm.py:915`

这里同时处理：

```text
EngineCore / worker 侧 profile；
AsyncLLM frontend CPU profiler。
```

### 14.2 reset_mm_cache

```python
async def reset_mm_cache(self) -> None:
    await self.renderer.clear_mm_cache_async()
    await self.engine_core.reset_mm_cache_async()
```

位置：`vllm/v1/engine/async_llm.py:917` 到 `vllm/v1/engine/async_llm.py:919`

说明多模态 cache 同时可能存在于：

```text
AsyncLLM renderer 侧；
EngineCore / Worker 侧。
```

### 14.3 reset_prefix_cache / reset_encoder_cache

```python
async def reset_prefix_cache(...):
    return await self.engine_core.reset_prefix_cache_async(...)
```

位置：`vllm/v1/engine/async_llm.py:921` 到 `vllm/v1/engine/async_llm.py:926`

```python
async def reset_encoder_cache(self) -> None:
    await self.engine_core.reset_encoder_cache_async()
```

位置：`vllm/v1/engine/async_llm.py:928` 到 `vllm/v1/engine/async_llm.py:929`

prefix cache / encoder cache 的核心状态在 EngineCore / Scheduler / Worker 侧，所以外层只转发。

### 14.4 LoRA

```python
async def add_lora(self, lora_request: LoRARequest) -> bool:
    return await self.engine_core.add_lora_async(lora_request)
```

位置：`vllm/v1/engine/async_llm.py:948` 到 `vllm/v1/engine/async_llm.py:950`

类似还有：

```text
remove_lora_async
list_loras_async
pin_lora_async
```

位置：`vllm/v1/engine/async_llm.py:952` 到 `vllm/v1/engine/async_llm.py:962`

### 14.5 collective_rpc

```python
return await self.engine_core.collective_rpc_async(
    method, timeout, args, kwargs
)
```

位置：`vllm/v1/engine/async_llm.py:964` 到 `vllm/v1/engine/async_llm.py:976`

这允许外层 Engine 发起 worker 侧 collective RPC。

---

## 15. shutdown 和生命周期

`AsyncLLM.__del__()` 会调用 shutdown：

```python
def __del__(self):
    self.shutdown()
```

位置：`vllm/v1/engine/async_llm.py:256` 到 `vllm/v1/engine/async_llm.py:257`

`shutdown()` 做三件事：

```python
shutdown_prometheus()

if renderer := getattr(self, "renderer", None):
    renderer.shutdown()

if engine_core := getattr(self, "engine_core", None):
    engine_core.shutdown(timeout=timeout)

handler = getattr(self, "output_handler", None)
if handler is not None:
    cancel_task_threadsafe(handler)
```

位置：`vllm/v1/engine/async_llm.py:259` 到 `vllm/v1/engine/async_llm.py:271`

所以 shutdown 链路是：

```text
AsyncLLM.shutdown()
  → shutdown_prometheus()
  → renderer.shutdown()
  → EngineCoreClient.shutdown()
      → 后台 EngineCoreProc / sockets / resources 清理
  → cancel output_handler task
```

这体现：

```text
AsyncLLM 负责前端异步资源和后台 EngineCoreClient 的生命周期收口。
```

---

## 16. 和 LLMEngine 的异同

### 16.1 相同点

`AsyncLLM` 和同步 `LLMEngine` 都属于外层 Engine。

共同结构是：

```text
InputProcessor：
  EngineInput / PromptType → EngineCoreRequest

EngineCoreClient：
  把 EngineCoreRequest 送入 EngineCore
  从 EngineCore 拉取 EngineCoreOutputs

OutputProcessor：
  EngineCoreOutputs → RequestOutput / PoolingRequestOutput
```

共同职责是：

```text
接收用户请求；
处理输入；
登记输出状态；
把请求交给 EngineCore；
消费 EngineCoreOutputs；
生成用户可见输出；
处理 abort / profile / sleep / cache reset / LoRA 等控制接口。
```

### 16.2 不同点

最核心区别是输出驱动方式。

同步 `LLMEngine`：

```text
调用方循环：
  while engine.has_unfinished_requests():
      outputs = engine.step()
```

`LLMEngine.step()` 内部同步执行：

```text
EngineCoreClient.get_output()
  → OutputProcessor.process_outputs()
  → return request_outputs
```

异步 `AsyncLLM`：

```text
后台 output_handler：
  while True:
      outputs = await engine_core.get_output_async()
      output_processor.process_outputs(...)
      push to RequestOutputCollector

调用方：
  async for output in engine.generate(...):
      yield output
```

所以：

```text
LLMEngine：step 返回输出。
AsyncLLM：output_handler 分发输出，generate/encode 从 queue 取输出。
```

### 16.3 EngineCore 运行方式不同

同步 `LLMEngine` 可以使用：

```text
InprocClient：同进程 EngineCore；
SyncMPClient：多进程 EngineCore。
```

异步 `AsyncLLM` 通过 `EngineCoreClient.make_async_mp_client()` 创建异步 client：

```text
DP=1：
  AsyncMPClient。

data_parallel_external_lb=True：
  DPAsyncMPClient。

其它 DP>1 场景：
  DPLBAsyncMPClient。
```

因此异步路径天然更偏：

```text
frontend asyncio loop
  ↔ AsyncMPClient / ZMQ
  ↔ background EngineCoreProc
```

---

## 17. 容易混淆的点

### 17.1 AsyncLLM 是不是 EngineCore？

不是。

`AsyncLLM` 是外层异步 Engine；`EngineCore` 是内部执行核心。

```text
AsyncLLM：
  处理异步 API、输入输出、collector、output_handler。

EngineCore：
  处理 schedule → execute → update → output。
```

### 17.2 AsyncLLM.engine_core 是不是 EngineCore 对象？

不是直接的 EngineCore。

`AsyncLLM.__init__()` 里：

```python
self.engine_core = EngineCoreClient.make_async_mp_client(...)
```

位置：`vllm/v1/engine/async_llm.py:145` 到 `vllm/v1/engine/async_llm.py:153`

所以这里的 `engine_core` 字段实际是：

```text
AsyncMPClient / DPAsyncMPClient / DPLBAsyncMPClient
```

它通过 ZMQ 和后台 `EngineCoreProc` 通信。

### 17.3 output_handler 是不是执行模型的 loop？

不是。

`output_handler` 只负责：

```text
从 EngineCoreClient 拉 EngineCoreOutputs；
调用 OutputProcessor；
把 RequestOutput 推到 collector queue。
```

真正执行模型的 loop 在后台 `EngineCoreProc` / `EngineCore.step()`。

### 17.4 generate() 是不是直接拿 EngineCoreOutputs？

不是。

`generate()` 从 `RequestOutputCollector` 取的是已经处理好的 `RequestOutput`：

```text
EngineCoreOutputs
  → output_handler
  → OutputProcessor.process_outputs()
  → RequestOutputCollector
  → generate() yield RequestOutput
```

### 17.5 OutputProcessor.process_outputs() 为什么不返回 request_outputs？

异步路径下，`OutputProcessor` 有 queue，因此会直接把输出投递到对应 collector。

源码里明确断言：

```python
assert not processed_outputs.request_outputs
```

位置：`vllm/v1/engine/async_llm.py:678` 到 `vllm/v1/engine/async_llm.py:679`

同步路径中，`OutputProcessor.process_outputs()` 返回 `request_outputs`；异步路径中，它推送到 queue。

### 17.6 abort 为什么要同时通知 OutputProcessor 和 EngineCore？

因为两边维护不同状态：

```text
OutputProcessor：
  前端输出状态、collector queue、parent/child request 映射。

EngineCore / Scheduler：
  内部请求状态、KV block、running/waiting 队列。
```

所以 abort 链路是：

```text
OutputProcessor.abort_requests()
  → EngineCoreClient.abort_requests_async()
```

---

## 18. 最关键的关系图

### 18.1 初始化

```text
AsyncLLM.__init__()
  → renderer_from_config()
  → InputProcessor
  → OutputProcessor
  → EngineCoreClient.make_async_mp_client()
      → AsyncMPClient / DPAsyncMPClient / DPLBAsyncMPClient
      → 后台 EngineCoreProc
  → logger_manager
  → output_handler 准备 / 启动
```

### 18.2 请求进入

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → input_processor.assign_request_id()
  → RequestOutputCollector
  → OutputProcessor.add_request(request, queue)
  → AsyncMPClient.add_request_async(request)
  → ZMQ ADD
  → EngineCoreProc
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → Scheduler.add_request()
```

### 18.3 输出返回

```text
EngineCoreProc._process_engine_step()
  → self.step_fn()
      → EngineCore.step() 或 EngineCore.step_with_batch_queue()
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → EngineCoreProc output socket
  → AsyncMPClient process_outputs_socket()
  → AsyncMPClient.outputs_queue
  → AsyncLLM._run_output_handler() 创建的后台 output_handler task
  → OutputProcessor.process_outputs()
  → RequestOutputCollector
  → AsyncLLM.generate() / encode()
  → yield RequestOutput / PoolingRequestOutput
```

### 18.4 generate 消费端

```text
AsyncLLM.generate()
  → q = await add_request(...)
  → while not finished:
      out = q.get_nowait() or await q.get()
      yield out
  → cancel / error 时 abort
  → finally q.close()
```

### 18.5 control / utility

```text
AsyncLLM.pause_generation() / sleep() / profile() / reset_xxx() / add_lora()
  → AsyncMPClient.call_utility_async(method, args)
  → ZMQ UTILITY
  → EngineCoreProc._handle_client_request()
  → EngineCore method
  → Scheduler / model_executor
  → UtilityOutput
  → AsyncMPClient future result
```

---

## 19. 从“回答问题”的角度总结

如果问：

```text
AsyncLLM 负责什么？
```

可以回答：

```text
AsyncLLM 是 vLLM V1 异步路径里的外层 Engine。

它负责接收异步 generate / encode 请求，调用 InputProcessor 把用户输入转换成 EngineCoreRequest，
创建 RequestOutputCollector 作为每个请求的异步输出队列，
并通过 AsyncMPClient 把请求发送到后台 EngineCoreProc。

后台 EngineCore 完成 schedule、execute、update 后返回 EngineCoreOutputs。
AsyncLLM 的 output_handler 会持续 await engine_core.get_output_async()，
拿到 EngineCoreOutputs 后调用 OutputProcessor.process_outputs()，
把内部输出转换成 RequestOutput / PoolingRequestOutput 并推入对应 collector。
调用方最终通过 async generator 从 collector 中取出结果。
```

它和 `LLMEngine` 的区别可以概括为：

```text
LLMEngine：同步外层 Engine，调用方通过 step() 主动拉输出。
AsyncLLM：异步外层 Engine，后台 output_handler 持续拉输出，调用方通过 async generator 消费结果。
```

它和 `EngineCore` 的区别可以概括为：

```text
AsyncLLM：异步输入输出编排层。
EngineCore：内部执行闭环总控。
```

最小心智模型：

```text
AsyncLLM = InputProcessor + AsyncMPClient + OutputProcessor + output_handler + RequestOutputCollector。
```

展开就是：

```text
输入侧：
  用户请求 → InputProcessor → EngineCoreRequest → AsyncMPClient → EngineCore

输出侧：
  EngineCoreOutputs → AsyncMPClient → output_handler → OutputProcessor → RequestOutputCollector → async generator
```
