# vLLM V1 Engine 逻辑梳理

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/input_processor.py`
- `vllm/vllm/v1/engine/output_processor.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/engine/__init__.py`
- `vllm/vllm/v1/request.py`

本文用于总览 vLLM V1 外层 `Engine` 的职责、`LLMEngine` / `AsyncLLM` 的关系、输入输出处理边界，以及外层 Engine 如何通过 `EngineCoreClient` 驱动内部 `EngineCore`。

---

## 1. Engine 是什么

在 vLLM V1 里，`Engine` 更像一个架构层面的泛称，不是一个唯一固定的具体类。

可以理解为：

```text
Engine = vLLM 对外提供推理能力的外层引擎体系。
```

具体代码里，常见的外层 Engine 形态是：

```text
LLMEngine：
  同步 / legacy 兼容路径里的外层 Engine。

AsyncLLM：
  异步 / API server 常用路径里的外层 Engine。
```

而 `EngineCore` 不是外层 Engine 本身，它是外层 Engine 背后的内部执行核心。

源码中 `EngineCore` 的定位非常直接：

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
```

位置：`vllm/vllm/v1/engine/core.py:96` 到 `vllm/vllm/v1/engine/core.py:97`

也就是说：

```text
EngineCore 是 Engine 的内部主循环。
```

因此，最重要的边界是：

```text
Engine：
  负责用户接口、输入处理、输出处理、同步/异步编排。

EngineCore：
  负责内部 schedule → execute → update → output 执行闭环。
```

一句话：

```text
LLMEngine / AsyncLLM 是外层 Engine 的具体形态；
EngineCore 是它们驱动的内部执行核心；
EngineCoreClient 是两者之间的桥。
```

---

## 2. Engine 总体主链路

从用户请求到用户输出，完整主链路可以概括为：

```text
用户 / API server / LLM
  → LLMEngine / AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request() / add_request_async()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
  → EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output)
  → Worker / ModelRunner forward / sample
  → ModelRunnerOutput
  → Scheduler.update_from_output(scheduler_output, model_output)
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

这条链路可以拆成三大段：

```text
输入段：
  用户输入 → InputProcessor → EngineCoreRequest → EngineCore → Request → Scheduler

执行段：
  Scheduler → SchedulerOutput → Worker / ModelRunner → ModelRunnerOutput → Scheduler

输出段：
  EngineCoreOutputs → OutputProcessor → RequestOutput / PoolingRequestOutput
```

其中：

```text
输入段和输出段主要属于外层 Engine；
执行段主要属于 EngineCore / Scheduler / Worker。
```

最小心智模型：

```text
Engine 管输入输出；
EngineCore 管内部执行闭环；
Scheduler 管调度账本；
Worker / ModelRunner 管实际模型计算。
```

---

## 3. LLMEngine：同步外层 Engine

`LLMEngine` 是同步路径里的外层 Engine。

源码定义：

```python
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""
```

位置：`vllm/vllm/v1/engine/llm_engine.py:48` 到 `vllm/vllm/v1/engine/llm_engine.py:49`

它主要提供同步接口：

```text
add_request()：添加请求；
step()：拉取并处理一轮输出；
has_unfinished_requests()：判断是否还有未完成请求；
abort_request()：取消请求；
profile / reset / sleep / wake_up / LoRA 等控制接口。
```

### 3.1 初始化阶段

`LLMEngine.__init__()` 中创建三个关键组件：

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)

# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(...)

# EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
self.engine_core = EngineCoreClient.make_client(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:93` 到 `vllm/vllm/v1/engine/llm_engine.py:111`

这里的注释已经说明了同步外层 Engine 的核心边界：

```text
InputProcessor：
  EngineInput → EngineCoreRequest

EngineCoreClient：
  EngineCoreRequest → EngineCore
  EngineCoreOutputs ← EngineCore

OutputProcessor：
  EngineCoreOutputs → RequestOutput
```

注意：`LLMEngine.engine_core` 字段名虽然叫 `engine_core`，但实际是 `EngineCoreClient`。

```text
in-process：
  EngineCoreClient 是 InprocClient，内部直接持有 EngineCore。

multi-process：
  EngineCoreClient 是 SyncMPClient，通过 ZMQ 访问后台 EngineCoreProc。
```

### 3.2 同步请求入口 add_request()

普通请求进入 `LLMEngine.add_request()` 后，会先走输入处理：

```python
request = self.input_processor.process_inputs(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:250` 到 `vllm/vllm/v1/engine/llm_engine.py:260`

然后分配内部 request id：

```python
self.input_processor.assign_request_id(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:263`

普通 `n == 1` 时：

```python
self.output_processor.add_request(request, prompt_text, None, 0)
self.engine_core.add_request(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:272` 到 `vllm/vllm/v1/engine/llm_engine.py:276`

所以同步请求路径是：

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → Scheduler.add_request()
```

如果 `SamplingParams.n > 1`，`LLMEngine` 会通过 `ParentRequest` fan out 多个 child request：

```text
用户一个请求
  → ParentRequest
  → child request 0
  → child request 1
  → ...
  → 每个 child 都进入 EngineCore
  → OutputProcessor / ParentRequest 聚合回一个外部 RequestOutput
```

### 3.3 同步输出入口 step()

`LLMEngine.step()` 先从 EngineCoreClient 拉输出：

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

如果 detokenize 后发现 stop string，需要通知 EngineCore abort：

```python
self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:316` 到 `vllm/vllm/v1/engine/llm_engine.py:318`

最后返回用户输出：

```python
return processed_outputs.request_outputs
```

位置：`vllm/vllm/v1/engine/llm_engine.py:334`

同步输出路径：

```text
LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

## 4. AsyncLLM：异步外层 Engine

`AsyncLLM` 是异步路径里的外层 Engine。

源码定义：

```python
class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""
```

位置：`vllm/vllm/v1/engine/async_llm.py:70` 到 `vllm/vllm/v1/engine/async_llm.py:71`

可以理解为：

```text
AsyncLLM = vLLM Engine 的异步外壳。
```

它和 `LLMEngine` 的核心边界相同：

```text
InputProcessor：输入转换；
EngineCoreClient：访问 EngineCore；
OutputProcessor：输出转换。
```

区别在于：

```text
LLMEngine：调用方主动 step() 拉输出。
AsyncLLM：后台 output_handler 持续拉输出，调用方通过 async generator 消费结果。
```

### 4.1 初始化阶段

`AsyncLLM.__init__()` 中创建：

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)

# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(...)

# EngineCore (starts the engine in background process).
self.engine_core = EngineCoreClient.make_async_mp_client(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:134` 到 `vllm/vllm/v1/engine/async_llm.py:153`

异步路径通常使用：

```text
AsyncMPClient / DPAsyncMPClient / DPLBAsyncMPClient
```

它们通过 ZMQ 和后台 `EngineCoreProc` 通信。

### 4.2 异步请求入口 add_request()

普通请求在 `AsyncLLM.add_request()` 中也会先走：

```python
request = self.input_processor.process_inputs(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:349` 到 `vllm/vllm/v1/engine/async_llm.py:360`

然后：

```python
self.input_processor.assign_request_id(request)
self._run_output_handler()
queue = RequestOutputCollector(params.output_kind, request.request_id)
```

位置：`vllm/vllm/v1/engine/async_llm.py:368` 到 `vllm/vllm/v1/engine/async_llm.py:377`

真正登记请求的方法是 `_add_request()`：

```python
self.output_processor.add_request(request, prompt, parent_req, index, queue)
await self.engine_core.add_request_async(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:400` 到 `vllm/vllm/v1/engine/async_llm.py:412`

异步请求路径：

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → _run_output_handler()
  → RequestOutputCollector
  → OutputProcessor.add_request(..., queue)
  → EngineCoreClient.add_request_async()
  → EngineCoreProc
  → EngineCore.preprocess_add_request()
  → Scheduler.add_request()
```

### 4.3 generate() 的 async generator

`generate()` 的源码注释说明：

```text
A separate output_handler loop runs in a background AsyncIO task,
pulling outputs from EngineCore and putting them into the
per-request AsyncStream.

The caller of generate() iterates the returned AsyncGenerator,
returning the RequestOutput back to the caller.
```

位置：`vllm/vllm/v1/engine/async_llm.py:549` 到 `vllm/vllm/v1/engine/async_llm.py:554`

`generate()` 本身从 `RequestOutputCollector` 取输出；generation 路径会过滤 streaming input 的 `STREAM_FINISHED` sentinel：

```python
out = q.get_nowait() or await q.get()
...
if out is not STREAM_FINISHED:
    yield out
```

位置：`vllm/vllm/v1/engine/async_llm.py:573` 到 `vllm/vllm/v1/engine/async_llm.py:586`

也就是说：

```text
generate() 不直接消费 EngineCoreOutputs；
它消费 OutputProcessor 已经处理好的 RequestOutput。
```

### 4.4 output_handler 后台输出循环

`AsyncLLM._run_output_handler()` 中的后台循环：

```python
outputs = await engine_core.get_output_async()
...
processed_outputs = output_processor.process_outputs(
    outputs_slice, outputs.timestamp, iteration_stats
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:656` 到 `vllm/vllm/v1/engine/async_llm.py:677`

异步路径中，`OutputProcessor` 不直接返回 request_outputs，而是推入 queue：

```python
# NOTE: RequestOutputs are pushed to their queues.
assert not processed_outputs.request_outputs
```

位置：`vllm/vllm/v1/engine/async_llm.py:678` 到 `vllm/vllm/v1/engine/async_llm.py:679`

异步输出路径：

```text
EngineCoreProc
  → EngineCoreOutputs
  → AsyncMPClient.get_output_async()
  → AsyncLLM.output_handler
  → OutputProcessor.process_outputs()
  → RequestOutputCollector.put(...)
  → AsyncLLM.generate() / encode()
  → yield RequestOutput / PoolingRequestOutput
```

---

## 5. InputProcessor 的位置和职责

`InputProcessor` 属于外层 Engine。

它的作用是把用户输入处理成 `EngineCoreRequest`。

### 5.1 初始化

`InputProcessor.__init__()` 保存各种配置：

```python
self.vllm_config = vllm_config
self.model_config = model_config = vllm_config.model_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
self.scheduler_config = vllm_config.scheduler_config
self.speculative_config = vllm_config.speculative_config
self.structured_outputs_config = vllm_config.structured_outputs_config
self.observability_config = vllm_config.observability_config
self.use_v2_model_runner = vllm_config.use_v2_model_runner
```

位置：`vllm/vllm/v1/engine/input_processor.py:44` 到 `vllm/vllm/v1/engine/input_processor.py:52`

它还会根据多模态能力初始化 `mm_encoder_cache_size` / `skip_prompt_length_check`，并创建 `InputPreprocessor`：

```python
if self.supports_mm_inputs:
    mm_budget = MultiModalBudget(vllm_config, mm_registry)
    self.mm_encoder_cache_size = mm_budget.encoder_cache_size
    self.skip_prompt_length_check = mm_budget.processor.info.skip_prompt_length_check
...
self.input_preprocessor = InputPreprocessor(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:58` 到 `vllm/vllm/v1/engine/input_processor.py:73`

### 5.2 process_inputs()

`process_inputs()` 会做：

```text
校验 SamplingParams / PoolingParams；
校验 LoRA；
校验 data_parallel_rank；
预处理 raw prompt；
校验平台和模型输入；
拆分 encoder / decoder input；
提取 prompt_token_ids / prompt_embeds；
克隆并补全 SamplingParams / PoolingParams；
整理多模态 mm_features；
构造 EngineCoreRequest。
```

最终返回：

```python
return EngineCoreRequest(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:370` 到 `vllm/vllm/v1/engine/input_processor.py:385`

### 5.3 assign_request_id()

外层 Engine 调用：

```python
InputProcessor.assign_request_id(request)
```

它会把用户 request id 保存为 external id；默认追加 8 位随机后缀生成内部 request id，但如果启用了 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`，源码只告警且不改写 `request_id`：

```python
request.external_req_id = request.request_id
if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
    logger.warning_once(...)
else:
    request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
```

位置：`vllm/vllm/v1/engine/input_processor.py:222` 到 `vllm/vllm/v1/engine/input_processor.py:240`

所以请求 ID 分两层：

```text
external_req_id：
  用户传入的 request_id，最终 RequestOutput 里会展示它。

request_id：
  默认是 vLLM 内部唯一 request_id，用于 EngineCore / Scheduler / OutputProcessor 状态对齐；禁用随机化时可能仍等于外部 id。
```

---

## 6. OutputProcessor 的位置和职责

`OutputProcessor` 也属于外层 Engine。

源码定位：

```python
class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""
```

位置：`vllm/vllm/v1/engine/output_processor.py:424` 到 `vllm/vllm/v1/engine/output_processor.py:425`

它负责把内部输出协议转换成用户输出协议。

### 6.1 add_request()

请求进入 EngineCore 之前，外层 Engine 会先调用：

```text
OutputProcessor.add_request()
```

它会建立输出侧状态 `RequestState`：

```python
req_state = RequestState.from_new_request(...)
self.request_states[request_id] = req_state
```

位置：`vllm/vllm/v1/engine/output_processor.py:534` 到 `vllm/vllm/v1/engine/output_processor.py:544`

并登记 external 到 internal 的映射：

```python
self.external_req_ids[req_state.external_req_id].append(request_id)
```

位置：`vllm/vllm/v1/engine/output_processor.py:548` 到 `vllm/vllm/v1/engine/output_processor.py:549`

这一步必须发生在 EngineCore 返回输出之前，因为 EngineCoreOutput 只带内部增量信息。

### 6.2 process_outputs()

`process_outputs()` 是输出转换核心。

源码注释说明它做三件事：

```text
1. Compute stats for logging；
2. Detokenize；
3. Create and handle RequestOutput objects。
```

位置：`vllm/vllm/v1/engine/output_processor.py:584` 到 `vllm/vllm/v1/engine/output_processor.py:610`

主流程：

```text
for each EngineCoreOutput:
  → 找到 request_states[request_id]
  → 更新 stats
  → generation：detokenizer.update(new_token_ids)
  → stop string 检查
  → logprobs_processor.update_from_output()
  → pooling：直接使用 pooling_output
  → RequestState.make_request_output()
  → 同步路径：append 到 request_outputs
  → 异步路径：queue.put(request_output)
  → 非 streaming input 请求 finished 后清理 RequestState
  → streaming input 请求可能应用下一段 StreamingUpdate 或等待 / 发送 STREAM_FINISHED sentinel
  → 必要时 reqs_to_abort.append(request_id)
```

### 6.3 stop string 和 reqs_to_abort

如果 OutputProcessor 通过 detokenize 发现 stop string，但 EngineCore 还没认为请求 finished，会返回 `reqs_to_abort`：

```text
OutputProcessor 检测文本级 stop
  → reqs_to_abort
  → LLMEngine / AsyncLLM 通知 EngineCore abort
  → Scheduler.finish_requests(... FINISHED_ABORTED)
```

这说明文本级 stop string 属于外层输出处理，但内部请求取消仍要通知 EngineCore。

### 6.4 同步和异步差异

同步路径中：

```text
RequestState.queue is None
process_outputs() 返回 request_outputs
LLMEngine.step() 直接 return
```

异步路径中：

```text
RequestState.queue is RequestOutputCollector
process_outputs() 把输出推入 queue
AsyncLLM.generate() / encode() 从 queue 取输出
```

---

## 7. EngineCoreClient 的桥接作用

外层 Engine 通常不直接持有 `EngineCore`，而是持有 `EngineCoreClient`。

源码注释：

```python
EngineCoreClient: subclasses handle different methods for pushing
    and pulling from the EngineCore for asyncio / multiprocessing.

Subclasses:
* InprocClient: In process EngineCore (for V0-style LLMEngine use)
* SyncMPClient: ZMQ + background proc EngineCore (for LLM)
* AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
```

位置：`vllm/vllm/v1/engine/core_client.py:71` 到 `vllm/vllm/v1/engine/core_client.py:80`

### 7.1 make_client()

同步路径通过：

```python
EngineCoreClient.make_client(
    multiprocess_mode=multiprocess_mode,
    asyncio_mode=False,
    ...
)
```

选择：

```text
multiprocess_mode=False：InprocClient
multiprocess_mode=True：SyncMPClient
```

### 7.2 make_async_mp_client()

异步路径通过：

```python
EngineCoreClient.make_async_mp_client(...)
```

选择：

```text
无 DP：AsyncMPClient
DP + external load balancer：DPAsyncMPClient
DP + internal load balancer：DPLBAsyncMPClient
```

### 7.3 InprocClient

同进程模式直接创建 EngineCore：

```python
self.engine_core = EngineCore(*args, **kwargs)
```

位置：`vllm/vllm/v1/engine/core_client.py:286` 到 `vllm/vllm/v1/engine/core_client.py:287`

添加请求时：

```python
req, request_wave = self.engine_core.preprocess_add_request(request)
self.engine_core.add_request(req, request_wave)
```

位置：`vllm/vllm/v1/engine/core_client.py:297` 到 `vllm/vllm/v1/engine/core_client.py:299`

拉输出时：

```python
outputs, model_executed = self.engine_core.step_fn()
self.engine_core.post_step(model_executed=model_executed)
```

位置：`vllm/vllm/v1/engine/core_client.py:289` 到 `vllm/vllm/v1/engine/core_client.py:291`

### 7.4 MPClient / AsyncMPClient

多进程模式中，EngineCore 在后台 `EngineCoreProc` 中运行。

前端 client 通过：

```text
ZMQ input_socket：发送 ADD / ABORT / UTILITY；
ZMQ output_socket：接收 EngineCoreOutputs；UtilityOutput 通过 EngineCoreOutputs.utility_output 字段承载。
```

`AsyncMPClient.add_request_async()` 会设置 client index 并发送 ADD：

```python
request.client_index = self.client_index
await self._send_input(EngineCoreRequestType.ADD, request)
```

位置：`vllm/vllm/v1/engine/core_client.py:1121` 到 `vllm/vllm/v1/engine/core_client.py:1124`

`AsyncMPClient.get_output_async()` 从 outputs queue 取输出：

```python
outputs = await self.outputs_queue.get()
```

位置：`vllm/vllm/v1/engine/core_client.py:1053` 到 `vllm/vllm/v1/engine/core_client.py:1062`

---

## 8. Engine 和 EngineCore 的边界

### 8.1 Engine 负责什么

```text
1. 对外提供同步 / 异步接口；
2. 接收用户 request_id、prompt、params；
3. 调用 InputProcessor 处理输入；
4. 调用 OutputProcessor 登记输出状态；
5. 通过 EngineCoreClient 访问 EngineCore；
6. 从 EngineCoreClient 拉取 EngineCoreOutputs；
7. 调用 OutputProcessor 构造用户输出；
8. 处理 abort、profile、sleep、wake_up、LoRA 等外层 API 转发；
9. 管理 stats、tracing、frontend profiler、output_handler 等外层生命周期。
```

### 8.2 EngineCore 负责什么

```text
1. 创建 model_executor；
2. profile / 初始化 KV cache；
3. 创建 StructuredOutputManager；
4. 创建 Scheduler；
5. 把 EngineCoreRequest 转成 Request；
6. 把 Request 加入 Scheduler；
7. 每轮 step 调 Scheduler.schedule()；
8. 调 model_executor.execute_model()，必要时再调用 sample_tokens()；
9. 调 Scheduler.update_from_output()；
10. 返回 EngineCoreOutputs；
11. 协调 Scheduler / executor 的 profile、reset、sleep、wake_up、LoRA 等内部控制能力。
```

### 8.3 Engine 不负责什么

外层 Engine 不负责：

```text
token 级调度；
waiting / running 队列管理；
KV block 分配和释放；
prefix cache 命中；
preemption；
Scheduler.update_from_output() 的状态对账；
Worker / ModelRunner forward；
sampling 的实际执行；
KV / encoder 物理缓存操作。
```

这些属于 `EngineCore`、`Scheduler`、`model_executor`、`Worker / ModelRunner`。

### 8.4 EngineCore 不负责什么

EngineCore 不负责：

```text
用户原始 prompt 预处理；
用户 request id 到内部 request id 的外层语义管理；
detokenize；
stop string 文本级检查；
最终 RequestOutput / PoolingRequestOutput 构造；
异步 RequestOutputCollector 分发；
API server async generator。
```

这些属于 `LLMEngine / AsyncLLM`、`InputProcessor`、`OutputProcessor`。

---

## 9. EngineCore 内部初始化与执行

外层 Engine 初始化时，会通过 `EngineCoreClient` 传入：

```text
vllm_config；
executor_class；
log_stats。
```

真正内部初始化在 `EngineCore.__init__()`。

### 9.1 创建 model_executor

```python
self.model_executor = executor_class(vllm_config)
```

位置：`vllm/vllm/v1/engine/core.py:122` 到 `vllm/vllm/v1/engine/core.py:123`

`executor_class` 来自：

```python
Executor.get_class(vllm_config)
```

它会根据 `parallel_config.distributed_executor_backend` 选择：

```text
uni → UniProcExecutor
mp → MultiprocExecutor
ray → RayDistributedExecutor / RayExecutorV2
external_launcher → ExecutorWithExternalLauncher
自定义 qualname → 自定义 Executor
```

### 9.2 初始化 KV cache

```python
kv_cache_config = self._initialize_kv_caches(vllm_config)
```

位置：`vllm/vllm/v1/engine/core.py:132` 到 `vllm/vllm/v1/engine/core.py:133`

内部会：

```text
获取 KV cache specs；
如果发现 non_causal KV cache spec，禁用 chunked prefill / prefix caching；
profile 可用 GPU memory，attention-free 模型则可用 KV memory 记为 0；
生成 worker KV cache config；
必要时把 auto-fit 后的 max_model_len 同步给 workers；
生成 scheduler KV cache config；
更新 cache_config；
调用 model_executor.initialize_from_config() 初始化 worker 侧 KV cache 并 warmup model。
```

### 9.3 创建 Scheduler

```python
Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
...
self.scheduler: SchedulerInterface = Scheduler(
    vllm_config=vllm_config,
    kv_cache_config=kv_cache_config,
    structured_output_manager=self.structured_output_manager,
    ...
)
```

位置：`vllm/vllm/v1/engine/core.py:136` 到 `vllm/vllm/v1/engine/core.py:158`

注意顺序：

```text
先初始化 KV cache；
再创建 Scheduler。
```

因为 Scheduler 需要 `kv_cache_config` 和 block size 等信息。

### 9.4 step_fn

如果没有 batch queue：

```text
step_fn = step
```

如果启用了 batch queue：

```text
step_fn = step_with_batch_queue
```

对应位置：`vllm/vllm/v1/engine/core.py:221` 到 `vllm/vllm/v1/engine/core.py:223`

---

## 10. 请求生命周期总览

### 10.1 同步请求进入

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → assign_request_id()
  → OutputProcessor.add_request(queue=None)
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
```

### 10.2 异步请求进入

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → assign_request_id()
  → _run_output_handler()
  → RequestOutputCollector
  → OutputProcessor.add_request(queue)
  → AsyncMPClient.add_request_async()
  → ZMQ ADD
  → EngineCoreProc.process_input_sockets()
  → EngineCore.preprocess_add_request()
  → input_queue 放入 (ADD, (Request, request_wave))
  → EngineCore.add_request()
  → Scheduler.add_request()
```

### 10.3 EngineCoreRequest → Request

`EngineCore.preprocess_add_request()`：

```text
EngineCoreRequest
  → mm_receiver_cache 处理多模态特征
  → Request.from_engine_core_request()
  → structured_output_manager.grammar_init(req)  # 如果需要
  → Request
```

### 10.4 Request 进入 Scheduler

`EngineCore.add_request()`：

```text
Request
  → pooling task 校验
  → KV / EC transfer 检查
  → Scheduler.add_request()
```

进入 Scheduler 后，请求由 Scheduler 管理：

```text
requests；
waiting；
skipped_waiting；
running；
finished_req_ids / finished_req_ids_dict；
_free_request() / _free_blocks() 释放路径。
```

---

## 11. 输出生命周期总览

### 11.1 内部输出产生

```text
EngineCore.step()
  → Scheduler.schedule()
  → model_executor.execute_model(scheduler_output, non_block=True)
  → future.result()
  → 如果 model_output is None，则 model_executor.sample_tokens(grammar_output)
  → Scheduler.update_from_output()
  → dict[client_index, EngineCoreOutputs]
```

### 11.2 同步输出返回

```text
LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
  → return list
```

### 11.3 异步输出返回

```text
EngineCoreProc
  → ZMQ output socket
  → AsyncMPClient.outputs_queue
  → AsyncLLM.output_handler
  → OutputProcessor.process_outputs()
  → RequestOutputCollector.put()
  → AsyncLLM.generate() / encode()
  → yield output
```

### 11.4 OutputProcessor 的转换

```text
EngineCoreOutput
  → RequestState
  → detokenizer.update(new_token_ids)
  → stop string check
  → logprobs_processor.update_from_output()
  → CompletionOutput / PoolingOutput
  → RequestOutput / PoolingRequestOutput
```

---

## 12. abort / stop / lifecycle 控制

### 12.1 用户主动 abort

同步：

```text
LLMEngine.abort_request()
  → OutputProcessor.abort_requests()
  → EngineCoreClient.abort_requests()
  → EngineCore.abort_requests()
  → Scheduler.finish_requests(... FINISHED_ABORTED)
```

异步：

```text
AsyncLLM.abort()
  → OutputProcessor.abort_requests()
  → EngineCoreClient.abort_requests_async()
  → ZMQ ABORT
  → EngineCoreProc
  → Scheduler.finish_requests(... FINISHED_ABORTED)
```

### 12.2 stop string 自动 abort

文本级 stop string 在 OutputProcessor 中处理。

如果 OutputProcessor 检测到 stop string，但 EngineCore 还没 finished：

```text
OutputProcessor.process_outputs()
  → reqs_to_abort
  → LLMEngine / AsyncLLM 通知 EngineCore abort
```

### 12.3 控制类 API

外层 Engine 暴露：

```text
profile；
reset_mm_cache；
reset_prefix_cache；
reset_encoder_cache；
pause_generation / resume_generation；
sleep；
wake_up；
add_lora / remove_lora / list_loras / pin_lora；
collective_rpc / apply_model；
scale_elastic_ep。
```

这些通常通过 `EngineCoreClient` 转发给 `EngineCore`，再由 EngineCore 操作：

```text
Scheduler；
model_executor；
Worker / ModelRunner。
```

---

## 13. 同步、多进程、异步模式对比

### 13.1 InprocClient

```text
LLMEngine
  → InprocClient
  → EngineCore 同进程对象
```

特点：

```text
add_request 直接调用 EngineCore.preprocess_add_request()；
get_output 直接调用 EngineCore.step_fn()；
适合 V0-style LLMEngine 兼容路径。
```

### 13.2 SyncMPClient

```text
LLMEngine
  → SyncMPClient
  → ZMQ
  → EngineCoreProc 后台进程
```

特点：

```text
同步接口不变；
EngineCore 在后台 busy loop 中运行；
前端通过 output queue / socket 拉输出。
```

### 13.3 AsyncMPClient

```text
AsyncLLM
  → AsyncMPClient
  → asyncio + ZMQ
  → EngineCoreProc 后台进程
```

特点：

```text
add_request_async 异步发送请求；
get_output_async 异步拉输出；
AsyncLLM.output_handler 后台分发 RequestOutput。
```

### 13.4 DP client

DP 场景下可能使用：

```text
DPAsyncMPClient：外部负载均衡；
DPLBAsyncMPClient：内部负载均衡。
```

它们会根据 DP rank、负载统计、client_index、current_wave 等信息选择具体 EngineCore。

---

## 14. 对象层级速查

```text
EngineInput / PromptType：
  用户侧输入。

EngineCoreRequest：
  InputProcessor 生成，外层 Engine 传给 EngineCore 的请求协议。

Request：
  EngineCore 转换后交给 Scheduler 的内部请求对象，包含调度状态。

SchedulerOutput：
  Scheduler 生成的一轮执行计划。

ModelRunnerOutput：
  Worker / ModelRunner 返回的模型执行结果。

EngineCoreOutput：
  Scheduler.update_from_output() 生成的单请求内部增量输出。

EngineCoreOutputs：
  EngineCore 返回给某个 client 的一批内部输出。

RequestOutput / PoolingRequestOutput：
  OutputProcessor 转换后的用户可见输出。
```

---

## 15. 组件职责速查

```text
LLMEngine：
  同步外层 Engine，负责 add_request / step / 同步控制 API。

AsyncLLM：
  异步外层 Engine，负责 generate / encode / output_handler / async generator。

InputProcessor：
  把用户输入、params、多模态输入转换成 EngineCoreRequest。

OutputProcessor：
  把 EngineCoreOutputs 转成 RequestOutput / PoolingRequestOutput。

RequestOutputCollector：
  AsyncLLM 中每个请求的异步输出队列。

EngineCoreClient：
  屏蔽 in-process / multi-process / async 通信差异。

EngineCore：
  内部执行闭环总控。

EngineCoreProc：
  后台进程版 EngineCore，额外管理 ZMQ、input/output queue、busy loop。

Scheduler：
  请求队列、token budget、KV block、状态账本、输出回收。

model_executor：
  执行器抽象，负责把 execute_model / sample_tokens 分发给 Worker。

Worker / ModelRunner：
  准备模型输入、执行 forward、logits、sampling、pooling、KV/encoder 实际操作。
```

---

## 16. 容易混淆的点

### 16.1 Engine 是不是 LLMEngine？

不完全是。

```text
LLMEngine 是同步外层 Engine 的具体形态；
AsyncLLM 是异步外层 Engine 的具体形态；
Engine 是外层引擎体系的泛称。
```

### 16.2 Engine 是不是 EngineCore？

不是。

```text
EngineCore 是 Engine 的 inner loop，
更靠内，负责 schedule → execute → update → output。
```

### 16.3 InputProcessor / OutputProcessor 属于 EngineCore 吗？

不属于。

它们在 `LLMEngine` / `AsyncLLM` 初始化时创建，属于外层 Engine。

### 16.4 LLMEngine.engine_core 是不是 EngineCore？

字段名容易误导。

```text
LLMEngine.engine_core 实际是 EngineCoreClient。
```

只有 InprocClient 内部才直接持有 `EngineCore`。

### 16.5 EngineCoreOutputs 是不是最终用户输出？

不是。

```text
EngineCoreOutputs 是内部输出协议；
RequestOutput / PoolingRequestOutput 才是用户可见输出。
```

### 16.6 OutputProcessor.add_request() 为什么在有输出前调用？

因为它不是处理已有输出，而是提前登记输出状态。

后续 EngineCoreOutput 只带 request_id 和增量信息，必须提前有 RequestState 才能构造完整用户输出。

### 16.7 stop string 在哪里处理？

文本级 stop string 在 OutputProcessor 的 detokenizer 中处理。

如果因此需要终止内部请求，会通过 `reqs_to_abort` 通知 EngineCore。

### 16.8 Scheduler 是外层 Engine 创建的吗？

不是。

Scheduler 在 `EngineCore.__init__()` 中创建。

外层 Engine 只创建 EngineCoreClient。

---

## 17. 最关键的关系图

### 17.1 总体层级

```text
用户 / API server
  ↓
LLMEngine / AsyncLLM                  # 外层 Engine
  ↓
InputProcessor                        # 输入适配
  ↓
EngineCoreRequest                     # 外层到内层请求协议
  ↓
EngineCoreClient                      # 通信桥
  ↓
EngineCore / EngineCoreProc           # 内部执行核心
  ↓
Scheduler + model_executor            # 调度 + 执行分发
  ↓
Worker / ModelRunner                  # 实际模型执行
  ↓
EngineCoreOutputs                     # 内部输出协议
  ↓
OutputProcessor                       # 输出适配
  ↓
RequestOutput / PoolingRequestOutput  # 用户可见输出
```

### 17.2 同步 LLMEngine 主链路

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()

LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

### 17.3 异步 AsyncLLM 主链路

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → InputProcessor.assign_request_id()
  → _run_output_handler()
  → RequestOutputCollector
  → OutputProcessor.add_request(..., queue)
  → EngineCoreClient.add_request_async()
  → EngineCoreProc.process_input_sockets()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCoreProc

AsyncLLM._run_output_handler() 内创建的 output_handler 后台任务
  → EngineCoreClient.get_output_async()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutputCollector.put()
  → generate() 过滤 STREAM_FINISHED sentinel 后 yield
```

### 17.4 EngineCore 内部主链路

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(..., non_block=True)
  → future.result()
  → 如果 model_output is None，则 model_executor.sample_tokens(grammar_output)
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
```

### 17.5 初始化主链路

```text
EngineArgs / AsyncEngineArgs
  → VllmConfig
  → Executor.get_class(vllm_config)
  → LLMEngine / AsyncLLM
      → renderer
      → InputProcessor
      → OutputProcessor
      → EngineCoreClient
          → InprocClient / SyncMPClient / AsyncMPClient
          → EngineCore / EngineCoreProc
              → executor_class(vllm_config)
              → KV cache init
              → Scheduler init
```

---

## 18. 推荐阅读顺序

如果第一次看 Engine，可以按下面顺序：

```text
01_engine_role.md
  → 11_engine_vs_engine_core_boundaries.md
  → 02_llm_engine_sync.md
  → 03_async_llm.md
  → 07_request_lifecycle.md
  → 05_output_processor.md
  → 09_initialization_and_config.md
```

如果重点想理解请求从外层到内层：

```text
01_engine_role.md
  → 04_input_processor.md
  → 07_request_lifecycle.md
  → 06_engine_core_client_bridge.md
  → ../engine_core/02_request_entry.md
```

如果重点想理解输出从内层回外层：

```text
../engine_core/07_engine_core_outputs.md
  → 08_output_lifecycle.md
  → 05_output_processor.md
  → 03_async_llm.md
```

如果想和 EngineCore 文档联动：

```text
11_engine_vs_engine_core_boundaries.md
  → ../engine_core/01_engine_core_role.md
  → ../engine_core/03_step_loop.md
  → ../engine_core/05_worker_execution.md
  → ../engine_core/07_engine_core_outputs.md
```

---

## 19. 从“回答问题”的角度总结

如果问：

```text
vLLM V1 Engine 负责什么？
```

可以回答：

```text
vLLM V1 中的 Engine 可以理解为外层引擎体系，具体形态包括同步的 LLMEngine 和异步的 AsyncLLM。
它们负责对外接收请求、调用 InputProcessor 把用户输入转换成 EngineCoreRequest，
通过 EngineCoreClient 把请求送入内部 EngineCore，
再从 EngineCoreClient 拉取 EngineCoreOutputs，
最后调用 OutputProcessor 把内部输出转换成用户可见的 RequestOutput / PoolingRequestOutput。
```

如果问：

```text
Engine 和 EngineCore 是什么关系？
```

可以回答：

```text
Engine 是外层输入输出编排层，EngineCore 是内部执行闭环。
EngineCore 是 Engine 的 inner loop，它不直接处理用户原始输入，也不直接构造最终用户输出；
这些由 LLMEngine / AsyncLLM 侧的 InputProcessor 和 OutputProcessor 完成。
EngineCore 负责把 Scheduler 和 model_executor / Worker / ModelRunner 串起来，
完成 schedule → execute → update → output 的内部主循环。
```

如果问：

```text
LLMEngine 和 AsyncLLM 有什么区别？
```

可以回答：

```text
二者都属于外层 Engine，都使用 InputProcessor、OutputProcessor 和 EngineCoreClient。
区别在于输出驱动方式：LLMEngine 是同步接口，调用方通过 step() 主动拉取一批输出；
AsyncLLM 是异步接口，后台 output_handler 持续从 EngineCore 拉取输出，
并把 RequestOutput 推入每个请求的 RequestOutputCollector，调用方通过 async generator 消费。
```

最终最小心智模型：

```text
Engine = InputProcessor + EngineCoreClient + OutputProcessor + 同步/异步外层接口。

EngineCore = Scheduler + model_executor 的内部执行闭环总控。
```

压缩成一句话：

```text
Engine 管用户侧输入输出，EngineCore 管内部调度执行。
```
