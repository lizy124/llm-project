# 11. Engine 和 EngineCore 的边界是什么？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/input_processor.py`
- `vllm/vllm/v1/engine/output_processor.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/request.py`

本问题关注：外层 `Engine` 和内部 `EngineCore` 的职责边界，避免把输入输出处理、调度、模型执行混在一起理解。

---

## 1. 一句话回答

`Engine` 是外层输入输出编排层，`EngineCore` 是内部执行闭环总控。

在 vLLM V1 里，说 `Engine` 时，通常不是指某个唯一的类，而是指外层引擎体系。具体代码里常见形态是：

```text
LLMEngine：
  同步外层 Engine。

AsyncLLM：
  异步外层 Engine。
```

而 `EngineCore` 的源码注释非常明确：

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
```

位置：`vllm/vllm/v1/engine/core.py:96` 到 `vllm/vllm/v1/engine/core.py:97`

这句话说明：

```text
EngineCore 是 Engine 的内部主循环，而不是外层 Engine 本身。
```

最小边界可以这样记：

```text
Engine：
  负责用户接口、输入处理、输出处理、同步/异步编排。

EngineCore：
  负责内部 schedule → execute → update → output 执行闭环。
```

完整链路是：

```text
用户 / API server
  → LLMEngine / AsyncLLM
  → InputProcessor
  → EngineCoreRequest
  → EngineCoreClient
  → EngineCore
  → Scheduler + model_executor
  → Worker / ModelRunner
  → EngineCoreOutputs
  → OutputProcessor
  → RequestOutput / PoolingRequestOutput
```

一句话总结：

```text
EngineCore 不负责用户输入输出；Engine 不负责内部调度执行细节。
```

---

## 2. Engine 是哪一层

这里的 `Engine` 可以理解为外层引擎体系，而不是一个固定类名。

它主要包括：

```text
LLMEngine：同步接口；
AsyncLLM：异步接口；
InputProcessor：输入转换；
OutputProcessor：输出转换；
EngineCoreClient：访问 EngineCore 的桥。
```

### 2.1 LLMEngine 是同步外层 Engine

`LLMEngine` 的类定义：

```python
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""
```

位置：`vllm/vllm/v1/engine/llm_engine.py:48` 到 `vllm/vllm/v1/engine/llm_engine.py:49`

它初始化时创建：

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)

# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(...)

# EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
self.engine_core = EngineCoreClient.make_client(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:93` 到 `vllm/vllm/v1/engine/llm_engine.py:111`

这三个注释已经说明外层 Engine 的边界：

```text
输入：EngineInput → EngineCoreRequest
执行桥：EngineCoreClient
输出：EngineCoreOutputs → RequestOutput
```

### 2.2 AsyncLLM 是异步外层 Engine

`AsyncLLM` 的类定义：

```python
class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""
```

位置：`vllm/vllm/v1/engine/async_llm.py:70` 到 `vllm/vllm/v1/engine/async_llm.py:71`

它初始化时也创建同样的外层组件：

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)

# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(...)

# EngineCore (starts the engine in background process).
self.engine_core = EngineCoreClient.make_async_mp_client(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:134` 到 `vllm/vllm/v1/engine/async_llm.py:153`

所以同步 / 异步外层 Engine 的共同结构是：

```text
LLMEngine / AsyncLLM
  → InputProcessor
  → OutputProcessor
  → EngineCoreClient
```

不同点只是输出驱动方式：

```text
LLMEngine：
  调用方通过 step() 同步拉输出。

AsyncLLM：
  后台 output_handler 持续异步拉输出，generate()/encode() 从 queue 中 yield。
```

---

## 3. EngineCore 是哪一层

`EngineCore` 更靠内，是执行闭环主控。

它初始化时创建内部执行组件：

```python
self.model_executor = executor_class(vllm_config)
...
kv_cache_config = self._initialize_kv_caches(vllm_config)
self.structured_output_manager = StructuredOutputManager(vllm_config)
...
self.scheduler: SchedulerInterface = Scheduler(...)
```

位置：`vllm/vllm/v1/engine/core.py:122` 到 `vllm/vllm/v1/engine/core.py:158`

它的核心方法是 `step()`：

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output.
```

位置：`vllm/vllm/v1/engine/core.py:479` 到 `vllm/vllm/v1/engine/core.py:484`

普通 `step()` 的主线是：

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

这说明 `EngineCore` 的核心职责是：

```text
Schedule：
  调 Scheduler.schedule() 得到 SchedulerOutput。

Execute：
  调 model_executor.execute_model()，把计划交给 Worker / ModelRunner。

Update / Make output：
  调 Scheduler.update_from_output()，用 ModelRunnerOutput 生成 EngineCoreOutputs。
```

所以 `EngineCore` 的最小心智模型是：

```text
EngineCore = schedule → execute → update → output 的内部闭环控制器。
```

---

## 4. Engine 负责什么

外层 Engine 主要负责用户侧和 EngineCore 之间的适配。

### 4.1 接收用户 / API 请求

同步路径入口：

```text
LLMEngine.add_request()
```

对应位置：`vllm/vllm/v1/engine/llm_engine.py:218`

异步路径入口：

```text
AsyncLLM.generate()
AsyncLLM.encode()
AsyncLLM.add_request()
```

对应位置：

- `vllm/vllm/v1/engine/async_llm.py:280`
- `vllm/vllm/v1/engine/async_llm.py:524`
- `vllm/vllm/v1/engine/async_llm.py:803`

这一层接收的是用户侧参数：

```text
request_id；
prompt / EngineInput；
SamplingParams / PoolingParams；
LoRARequest；
trace headers；
priority；
data_parallel_rank；
```

### 4.2 输入处理：调用 InputProcessor

同步 `LLMEngine.add_request()`：

```python
request = self.input_processor.process_inputs(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:250` 到 `vllm/vllm/v1/engine/llm_engine.py:260`

异步 `AsyncLLM.add_request()`：

```python
request = self.input_processor.process_inputs(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:349` 到 `vllm/vllm/v1/engine/async_llm.py:360`

这一步完成：

```text
PromptType / EngineInput / params
  → EngineCoreRequest
```

`InputProcessor.process_inputs()` 最后返回 `EngineCoreRequest`：

```python
return EngineCoreRequest(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:370` 到 `vllm/vllm/v1/engine/input_processor.py:385`

### 4.3 request_id 外部 / 内部映射

外层 Engine 会调用：

```python
self.input_processor.assign_request_id(request)
```

同步位置：`vllm/vllm/v1/engine/llm_engine.py:263`

异步位置：`vllm/vllm/v1/engine/async_llm.py:368`

`assign_request_id()` 会把用户传入 id 保存成 `external_req_id`，再生成内部唯一 id：

```python
request.external_req_id = request.request_id
request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
```

位置：`vllm/vllm/v1/engine/input_processor.py:222` 到 `vllm/vllm/v1/engine/input_processor.py:240`

这属于外层 Engine 的输入协议管理职责。

### 4.4 输出侧登记：调用 OutputProcessor.add_request()

同步路径：

```python
self.output_processor.add_request(request, prompt_text, None, 0)
self.engine_core.add_request(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:272` 到 `vllm/vllm/v1/engine/llm_engine.py:276`

异步路径：

```python
self.output_processor.add_request(request, prompt, parent_req, index, queue)
await self.engine_core.add_request_async(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:400` 到 `vllm/vllm/v1/engine/async_llm.py:412`

这一步不是处理已有输出，而是提前建立输出侧状态：

```text
request_id → RequestState
external_req_id → internal request_ids
parent request / child request
异步 RequestOutputCollector queue
Detokenizer / LogprobsProcessor
```

### 4.5 通过 EngineCoreClient 访问 EngineCore

外层 Engine 持有的是 `EngineCoreClient`，不是一定直接持有 `EngineCore`。

`EngineCoreClient` 注释：

```python
EngineCoreClient: subclasses handle different methods for pushing
    and pulling from the EngineCore for asyncio / multiprocessing.
```

位置：`vllm/vllm/v1/engine/core_client.py:71` 到 `vllm/vllm/v1/engine/core_client.py:80`

它屏蔽：

```text
InprocClient：同进程直接调用 EngineCore；
SyncMPClient：同步 ZMQ + 后台 EngineCoreProc；
AsyncMPClient：asyncio ZMQ + 后台 EngineCoreProc。
```

### 4.6 输出处理：调用 OutputProcessor.process_outputs()

同步 `LLMEngine.step()`：

```python
outputs = self.engine_core.get_output()
processed_outputs = self.output_processor.process_outputs(
    outputs.outputs,
    engine_core_timestamp=outputs.timestamp,
    iteration_stats=iteration_stats,
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:302` 到 `vllm/vllm/v1/engine/llm_engine.py:313`

异步 `AsyncLLM.output_handler`：

```python
outputs = await engine_core.get_output_async()
...
processed_outputs = output_processor.process_outputs(
    outputs_slice, outputs.timestamp, iteration_stats
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:656` 到 `vllm/vllm/v1/engine/async_llm.py:677`

`OutputProcessor` 的源码定位：

```python
class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""
```

位置：`vllm/vllm/v1/engine/output_processor.py:417` 到 `vllm/vllm/v1/engine/output_processor.py:418`

所以外层 Engine 负责把：

```text
EngineCoreOutputs
  → RequestOutput / PoolingRequestOutput
```

### 4.7 对外同步 / 异步输出接口

同步路径：

```python
return processed_outputs.request_outputs
```

位置：`vllm/vllm/v1/engine/llm_engine.py:334`

异步路径中，`OutputProcessor` 把输出推入 queue，`generate()` 从 queue 里取；generation 路径会过滤 streaming input 的 `STREAM_FINISHED` sentinel：

```python
out = q.get_nowait() or await q.get()
...
if out is not STREAM_FINISHED:
    yield out
```

位置：`vllm/vllm/v1/engine/async_llm.py:573` 到 `vllm/vllm/v1/engine/async_llm.py:586`

---

## 5. EngineCore 负责什么

`EngineCore` 负责内部执行和调度闭环。

### 5.1 创建内部执行组件

`EngineCore.__init__()` 会创建：

```text
model_executor；
KV cache；
StructuredOutputManager；
Scheduler；
mm_receiver_cache；
batch_queue / step_fn；
request_block_hasher；
async_scheduling 标志；
aborts_queue。
```

关键代码：

```python
self.model_executor = executor_class(vllm_config)
kv_cache_config = self._initialize_kv_caches(vllm_config)
self.structured_output_manager = StructuredOutputManager(vllm_config)
Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
self.scheduler: SchedulerInterface = Scheduler(...)
```

位置：`vllm/vllm/v1/engine/core.py:122` 到 `vllm/vllm/v1/engine/core.py:158`

### 5.2 接收 EngineCoreRequest 并转成 Request

EngineCore 的请求预处理入口：

```python
def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
```

位置：`vllm/vllm/v1/engine/core.py:853`

核心转换：

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
```

位置：`vllm/vllm/v1/engine/core.py:867`

如果是结构化输出请求，还会初始化 grammar：

```python
if req.use_structured_output:
    self.structured_output_manager.grammar_init(req)
```

位置：`vllm/vllm/v1/engine/core.py:868` 到 `vllm/vllm/v1/engine/core.py:874`

转换边界是：

```text
EngineCoreRequest：外层 Engine 传入的请求协议。
Request：Scheduler 内部使用的请求状态对象。
```

### 5.3 把 Request 交给 Scheduler

```python
self.scheduler.add_request(request)
```

位置：`vllm/vllm/v1/engine/core.py:403`

这一步之后，请求进入 Scheduler 的：

```text
requests dict；
waiting / skipped_waiting；
running；
```

等内部状态体系。

### 5.4 调度执行闭环

`EngineCore.step()` 负责把 Scheduler 和 model_executor 串起来：

```text
Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
```

这也是 `EngineCore` 和外层 Engine 的最大区别：

```text
外层 Engine 只是送请求和拿输出；
EngineCore 负责每一轮内部执行闭环。
```

### 5.5 控制类能力转发

`EngineCore` 还负责内部控制能力，例如：

```text
profile；
reset_mm_cache；
reset_prefix_cache；
reset_encoder_cache；
sleep / wake_up；
LoRA；
collective_rpc。
```

这些通常由外层 Engine 暴露 API，再通过 `EngineCoreClient` 转发到 EngineCore。

例如 EngineCore 中：

```python
def reset_prefix_cache(...):
    return self.scheduler.reset_prefix_cache(...)
```

位置：`vllm/vllm/v1/engine/core.py:680` 到 `vllm/vllm/v1/engine/core.py:685`

```python
def sleep(self, level: int = 1, mode: PauseMode = "abort"):
    ...
    pause_future = self.pause_scheduler(...)
    ...
    model_executor.sleep(level)
```

位置：`vllm/vllm/v1/engine/core.py:761` 到 `vllm/vllm/v1/engine/core.py:797`

这说明 EngineCore 是内部资源和执行系统的控制层。

---

## 6. Engine 不负责什么

外层 Engine 不负责内部执行细节。

### 6.1 不负责 token 级调度

外层 Engine 不判断：

```text
哪些请求本轮调度；
每个请求调度多少 token；
waiting / running 如何转移；
是否需要抢占；
是否启用 chunked prefill；
spec decode token 如何接受 / 拒绝。
```

这些由 `Scheduler` 负责。

外层 Engine 只是：

```text
add_request → get_output / output_handler
```

### 6.2 不负责 KV block 管理

外层 Engine 不分配 / 释放 KV block。

KV block 相关逻辑属于：

```text
EngineCore 初始化 KV cache；
Scheduler / KVCacheManager 调度和释放 block；
Worker / ModelRunner 使用物理 KV cache。
```

外层 Engine 最多提供：

```text
reset_prefix_cache / reset_encoder_cache / sleep
```

这类控制入口，但真正执行在 EngineCore / Scheduler / executor 侧。

### 6.3 不负责模型 forward

外层 Engine 不调用模型本体。

模型执行链路是：

```text
EngineCore
  → model_executor.execute_model()
  → Executor 实现分发到 Worker / ModelRunner
  → model forward / sample
```

外层 Engine 只等待 `EngineCoreOutputs`。

### 6.4 不负责 Scheduler.update_from_output()

模型执行结果如何回收、如何更新请求状态、如何释放资源，属于 Scheduler：

```text
Scheduler.update_from_output(scheduler_output, model_output)
```

外层 Engine 不参与这一步。

---

## 7. EngineCore 不负责什么

`EngineCore` 虽然是内部执行核心，但它也有明确边界。

### 7.1 不直接处理用户原始 prompt

用户原始 prompt 由外层 `InputProcessor` 处理。

InputProcessor 的输出是：

```text
EngineCoreRequest
```

EngineCore 接收的是已经处理好的 `EngineCoreRequest`，不是用户原始 prompt。

### 7.2 不负责 tokenizer / detokenize 输出

EngineCore 输出的是：

```text
EngineCoreOutput.new_token_ids
```

最终文本由 `OutputProcessor` 的 detokenizer 处理：

```text
new_token_ids
  → IncrementalDetokenizer.update()
  → CompletionOutput.text
```

所以：

```text
EngineCore 不负责把 token ids 变成最终 text。
```

### 7.3 不负责最终 RequestOutput 构造

EngineCore 返回的是 `EngineCoreOutputs`。

最终用户输出是：

```text
RequestOutput / PoolingRequestOutput
```

转换发生在：

```text
OutputProcessor.process_outputs()
```

### 7.4 不实现 token 级调度策略

EngineCore 调用：

```text
Scheduler.schedule()
```

但具体策略由 Scheduler 实现。

也就是说：

```text
EngineCore 调 Scheduler；
Scheduler 决定怎么调。
```

### 7.5 不直接实现模型 forward

EngineCore 调用：

```text
model_executor.execute_model(scheduler_output)
```

但真正 forward 在 Worker / ModelRunner。

也就是说：

```text
EngineCore 发起执行；
Worker / ModelRunner 实现执行。
```

---

## 8. 输入边界：EngineInput、EngineCoreRequest、Request

理解 Engine 和 EngineCore 的边界，关键是看请求对象如何变化。

### 8.1 EngineInput / PromptType

这是外层用户输入。

进入：

```text
LLMEngine.add_request()
AsyncLLM.add_request()
```

### 8.2 EngineCoreRequest

`InputProcessor.process_inputs()` 返回：

```text
EngineCoreRequest
```

它是外层 Engine 交给 EngineCore 的请求协议。

包含：

```text
request_id；
prompt_token_ids / prompt_embeds / prompt_is_token_ids；
mm_features；
sampling_params / pooling_params；
arrival_time；
lora_request；
cache_salt；
data_parallel_rank；
trace_headers；
client_index；
current_wave；
priority；
resumable；
external_req_id；
reasoning_ended / reasoning_parser_kwargs；
abort_immediately。
```

### 8.3 Request

`EngineCore.preprocess_add_request()` 中调用：

```text
Request.from_engine_core_request()
```

得到 Scheduler 内部 `Request`。

`Request` 额外有：

```text
status；
stop_reason；
kv_transfer_params；
num_computed_tokens；
output_token_ids；
all_token_ids；
spec_token_ids；
num_output_placeholders / async_tokens_to_discard；
next_decode_eligible_step / last_sched_seq；
block_hashes；
prefill_stats；
num_preemptions；
streaming_queue；
abort_immediately。
```

所以请求边界是：

```text
外层 Engine：EngineInput / PromptType → EngineCoreRequest
EngineCore：EngineCoreRequest → Request
Scheduler：Request
```

---

## 9. 输出边界：EngineCoreOutputs、RequestOutput

输出对象也分层。

### 9.1 EngineCoreOutput / EngineCoreOutputs

`EngineCoreOutput` 是内部增量输出：

```text
request_id；
new_token_ids；
new_logprobs / new_prompt_logprobs_tensors；
pooling_output；
finish_reason；
stop_reason；
events；
kv_transfer_params；
trace_headers；
prefill_stats；
routed_experts；
num_nans_in_logits。
```

`EngineCoreOutputs` 是一批输出和 stats：

```text
engine_index
outputs: list[EngineCoreOutput]
scheduler_stats
timestamp
utility_output
finished_requests
wave_complete / start_wave
```

它们由 Scheduler.update_from_output() 生成，并经 EngineCoreClient 返回给外层 Engine。

### 9.2 RequestOutput / PoolingRequestOutput

这是用户可见输出。

由外层 `OutputProcessor` 构造：

```text
EngineCoreOutput
  → detokenize / logprobs / stop string / aggregation
  → CompletionOutput / PoolingOutput
  → RequestOutput / PoolingRequestOutput
```

### 9.3 输出边界图

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

## 10. 同步路径边界

同步路径中，外层 Engine 是 `LLMEngine`。

### 10.1 请求进入

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request
  → Scheduler.add_request()
```

### 10.2 输出返回

```text
LLMEngine.step()
  → EngineCoreClient.get_output()
      → InprocClient: EngineCore.step_fn()
      → SyncMPClient: outputs_queue.get()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

### 10.3 同步边界总结

```text
LLMEngine：
  负责 add_request / step 这套同步外层接口。

EngineCore：
  负责 step_fn() 内部执行闭环。
```

---

## 11. 异步路径边界

异步路径中，外层 Engine 是 `AsyncLLM`。

### 11.1 请求进入

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → assign_request_id()
  → RequestOutputCollector
  → OutputProcessor.add_request(..., queue)
  → EngineCoreClient.add_request_async()
  → EngineCoreProc
  → Scheduler.add_request()
```

### 11.2 输出返回

```text
EngineCoreProc
  → EngineCoreOutputs
  → AsyncMPClient.get_output_async()
  → AsyncLLM.output_handler
  → OutputProcessor.process_outputs()
  → RequestOutputCollector.put()
  → AsyncLLM.generate() yield
```

### 11.3 异步边界总结

```text
AsyncLLM：
  负责异步请求、output_handler、collector queue、async generator。

EngineCoreProc / EngineCore：
  负责后台执行闭环。
```

---

## 12. Control / utility 的边界

一些控制类 API 看起来由外层 Engine 暴露，但实际执行在 EngineCore 或 executor / Scheduler。

### 12.1 外层 Engine 暴露接口

同步：

```text
LLMEngine.start_profile()
LLMEngine.reset_prefix_cache()
LLMEngine.sleep()
LLMEngine.wake_up()
LLMEngine.add_lora()
```

异步：

```text
AsyncLLM.start_profile()
AsyncLLM.pause_generation()
AsyncLLM.sleep()
AsyncLLM.wake_up()
AsyncLLM.add_lora()
```

### 12.2 EngineCoreClient 转发

这些调用通常通过 `EngineCoreClient` 转发。

例如异步 client 会通过 `UTILITY` 消息发给后台 EngineCoreProc。

### 12.3 EngineCore 内部执行

EngineCore 再转给：

```text
Scheduler：reset_prefix_cache / pause_scheduler / resume_scheduler
model_executor：profile / sleep / wake_up / add_lora / collective_rpc
```

所以边界是：

```text
外层 Engine：提供用户可调用 API。
EngineCoreClient：传递调用。
EngineCore：协调 Scheduler / executor 真正执行。
```

---

## 13. 容易混淆的点

### 13.1 Engine 是不是 EngineCore？

不是。

```text
Engine：外层引擎体系。
EngineCore：Engine 的内部主循环。
```

### 13.2 LLMEngine / AsyncLLM 是不是 Engine？

可以理解为具体外层 Engine 形态。

```text
LLMEngine：同步 Engine。
AsyncLLM：异步 Engine。
```

### 13.3 LLMEngine.engine_core 是不是 EngineCore？

不一定。

`LLMEngine.engine_core` 实际是 `EngineCoreClient`。

```text
InprocClient 内部才直接持有 EngineCore；
SyncMPClient / AsyncMPClient 通过 ZMQ 访问 EngineCoreProc。
```

### 13.4 InputProcessor / OutputProcessor 是不是 EngineCore 的一部分？

不是。

它们属于外层 Engine。

```text
InputProcessor：用户输入 → EngineCoreRequest
OutputProcessor：EngineCoreOutputs → RequestOutput
```

### 13.5 Scheduler 是不是 Engine 的一部分？

Scheduler 在 `EngineCore` 内部创建和持有。

从外层 Engine 的视角，它只通过 EngineCore 间接发挥作用。

### 13.6 Worker / ModelRunner 是不是 EngineCore？

不是。

```text
EngineCore：调用 model_executor。
model_executor：分发到 Worker。
Worker / ModelRunner：实际 forward / sample。
```

### 13.7 EngineCoreOutputs 是不是最终输出？

不是。

```text
EngineCoreOutputs：内部输出协议。
RequestOutput / PoolingRequestOutput：用户可见输出。
```

---

## 14. 最关键的关系图

### 14.1 总体层级

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

### 14.2 Engine 负责的部分

```text
LLMEngine / AsyncLLM
  → 接收用户请求
  → InputProcessor.process_inputs()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCoreClient.get_output()
  → OutputProcessor.process_outputs()
  → 返回 / yield RequestOutput
```

### 14.3 EngineCore 负责的部分

```text
EngineCore
  → preprocess_add_request()
  → Request.from_engine_core_request()
  → Scheduler.add_request()
  → step()
      → Scheduler.schedule()
      → model_executor.execute_model()
      → 如果 execute_model 返回 None，则 model_executor.sample_tokens(grammar_output)
      → Scheduler.update_from_output()
      → EngineCoreOutputs
```

### 14.4 Scheduler / Worker 边界

```text
Scheduler：
  请求队列、token budget、KV block、状态账本、输出回收。

Worker / ModelRunner：
  模型输入准备、forward、logits、sampling、pooling、KV/encoder 实际执行。
```

---

## 15. 从“回答问题”的角度总结

如果问：

```text
Engine 和 EngineCore 的边界是什么？
```

可以回答：

```text
Engine 是外层引擎体系，具体形态包括同步的 LLMEngine 和异步的 AsyncLLM。
它负责对外接收请求、调用 InputProcessor 把用户输入转成 EngineCoreRequest，
通过 EngineCoreClient 把请求送进 EngineCore，再调用 OutputProcessor 把 EngineCoreOutputs
转成用户可见的 RequestOutput / PoolingRequestOutput。

EngineCore 是 Engine 的内部主循环。它接收 EngineCoreRequest，转换成 Scheduler 使用的 Request，
然后在每轮 step 中调用 Scheduler.schedule() 生成 SchedulerOutput，
调用 model_executor / Worker / ModelRunner 执行模型，再调用 Scheduler.update_from_output()
生成 EngineCoreOutputs 返回给外层 Engine。
```

可以进一步压缩为：

```text
Engine 管输入输出和接口；EngineCore 管内部执行闭环。
```

最小心智模型：

```text
Engine：
  user input → EngineCoreRequest
  EngineCoreOutputs → user output

EngineCore：
  Request → SchedulerOutput → ModelRunnerOutput → EngineCoreOutputs
```

所以不要把几类职责混在一起：

```text
InputProcessor / OutputProcessor：外层 Engine。
Scheduler：EngineCore 内部调度组件。
Worker / ModelRunner：模型实际执行组件。
EngineCore：把 Scheduler 和 Worker 串起来的内部主循环。
```
