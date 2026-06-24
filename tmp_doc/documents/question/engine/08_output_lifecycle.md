# 08. 一次输出如何从 EngineCore 回到用户？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/output_processor.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/engine/__init__.py`

本问题关注：`EngineCoreOutputs` 如何被外层 Engine 消费，并转换成最终用户可见输出。

---

## 1. 一句话回答

一次输出从 EngineCore 回到用户的主线是：

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
  → 同步 step 返回 / 异步 queue yield
```

一句话：

```text
EngineCore 只返回内部 EngineCoreOutputs；
LLMEngine / AsyncLLM 通过 OutputProcessor 把它转换成用户可见的 RequestOutput / PoolingRequestOutput。
```

其中：

```text
Scheduler：
  把 Worker 的真实结果整理成 EngineCoreOutput。

EngineCoreClient：
  把 EngineCoreOutputs 从同进程 EngineCore 或后台 EngineCoreProc 取回前端。

OutputProcessor：
  负责 detokenize、logprobs、stop string、parallel sampling 合并、streaming queue 分发。
```

---

## 2. 输出对象层级

理解输出生命周期前，要先区分几层对象。

```text
ModelRunnerOutput
  Worker / ModelRunner 的原始执行结果。

EngineCoreOutput
  Scheduler 针对单个 request 构造的内部输出。

EngineCoreOutputs
  一批 EngineCoreOutput 的容器，按 client_index 返回给前端。

RequestOutput / PoolingRequestOutput
  vLLM 对用户暴露的最终输出对象。
```

完整层级是：

```text
Worker / ModelRunner
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → CompletionOutput / PoolingOutput
  → RequestOutput / PoolingRequestOutput
```

---

## 3. EngineCoreOutput：单个请求的内部输出

`EngineCoreOutput` 定义在：

```python
class EngineCoreOutput(msgspec.Struct, ...):
```

位置：`vllm/vllm/v1/engine/__init__.py:175` 到 `vllm/vllm/v1/engine/__init__.py:180`

主要字段包括：

```python
request_id: str
new_token_ids: list[int]

new_logprobs: LogprobsLists | None = None
new_prompt_logprobs_tensors: LogprobsTensors | None = None

pooling_output: torch.Tensor | None = None

finish_reason: FinishReason | None = None
stop_reason: int | str | None = None
events: list[EngineCoreEvent] | None = None
kv_transfer_params: dict[str, Any] | None = None

trace_headers: Mapping[str, str] | None = None

prefill_stats: PrefillStats | None = None

routed_experts: np.ndarray | None = None
num_nans_in_logits: int = 0
```

位置：`vllm/vllm/v1/engine/__init__.py:181` 到 `vllm/vllm/v1/engine/__init__.py:201`

可以分成几类：

| 字段 | 含义 |
|---|---|
| `request_id` | 内部 request id |
| `new_token_ids` | 本轮新增 token ids |
| `new_logprobs` | 本轮 sample logprobs |
| `new_prompt_logprobs_tensors` | prompt logprobs 相关 tensor |
| `pooling_output` | pooling 请求的 tensor 输出 |
| `finish_reason` | 请求是否结束以及结束原因 |
| `stop_reason` | stop token / stop string 等停止细节 |
| `events` | EngineCore 事件，用于 metrics / tracing |
| `kv_transfer_params` | KV transfer 相关输出参数 |
| `trace_headers` | tracing 上下文 |
| `prefill_stats` | prefill / cache 命中统计 |
| `routed_experts` | MoE routed experts 信息 |
| `num_nans_in_logits` | logits 中 NaN 数量 |

它有一个便捷属性：

```python
@property
def finished(self) -> bool:
    return self.finish_reason is not None
```

位置：`vllm/vllm/v1/engine/__init__.py:203` 到 `vllm/vllm/v1/engine/__init__.py:205`

所以：

```text
EngineCoreOutput.finished == True
等价于
finish_reason is not None
```

---

## 4. EngineCoreOutputs：一批内部输出的容器

`EngineCoreOutputs` 定义在：

```python
class EngineCoreOutputs(msgspec.Struct, ...):
```

位置：`vllm/vllm/v1/engine/__init__.py:220` 到 `vllm/vllm/v1/engine/__init__.py:225`

字段包括：

```python
engine_index: int = 0
outputs: list[EngineCoreOutput] = []
scheduler_stats: SchedulerStats | None = None
timestamp: float = 0.0

utility_output: UtilityOutput | None = None
finished_requests: set[str] | None = None

wave_complete: int | None = None
start_wave: int | None = None
```

位置：`vllm/vllm/v1/engine/__init__.py:229` 到 `vllm/vllm/v1/engine/__init__.py:244`

各字段含义是：

| 字段 | 含义 |
|---|---|
| `engine_index` | 多 engine / DP 场景下输出来自哪个 engine |
| `outputs` | 本批次的单请求 `EngineCoreOutput` 列表 |
| `scheduler_stats` | Scheduler 统计信息 |
| `timestamp` | 输出创建时间，用于 metrics |
| `utility_output` | utility 调用返回值，不是普通请求输出 |
| `finished_requests` | 已结束请求 id 集合，常用于 DP/LB 路由清理 |
| `wave_complete` | DP wave 完成信号 |
| `start_wave` | DP 下一 wave 启动信号 |

`timestamp` 如果为 0，会自动设置为 monotonic 时间：

```python
def __post_init__(self):
    if self.timestamp == 0.0:
        self.timestamp = time.monotonic()
```

位置：`vllm/vllm/v1/engine/__init__.py:246` 到 `vllm/vllm/v1/engine/__init__.py:248`

---

## 5. EngineCoreOutputs 和 RequestOutput 的区别

这两类对象处在不同边界。

```text
EngineCoreOutputs：
  EngineCore / Scheduler 内部输出协议。
  使用内部 request_id。
  携带 token ids、pooling tensor、logprobs、stats、KV transfer、finished ids 等。

RequestOutput / PoolingRequestOutput：
  用户可见输出协议。
  使用 external request id。
  携带 detokenized text、CompletionOutput、PoolingOutput、metrics 等。
```

转换发生在：

```python
OutputProcessor.process_outputs(...)
```

位置：`vllm/vllm/v1/engine/output_processor.py:576`

所以可以记成：

```text
EngineCoreOutputs 是“内部执行结果”；
RequestOutput 是“外部 API 输出”。
```

---

## 6. 输出生命周期从哪里开始

输出生命周期的上游是 Scheduler 回收 Worker 结果。

在 EngineCore 的 step 闭环中：

```text
EngineCore.step()
  → Scheduler.schedule()
  → model_executor.execute_model()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → dict[int, EngineCoreOutputs]
```

`dict[int, EngineCoreOutputs]` 的 key 是 `client_index`。

也就是说 EngineCore 返回的输出天然按前端 client 分组：

```text
client_index 0 → EngineCoreOutputs
client_index 1 → EngineCoreOutputs
...
```

这在多 API server / 多 client / DP 场景下用于把输出路由回正确前端。

---

## 7. 同进程输出如何进入 LLMEngine

同进程模式下，`LLMEngine.step()` 调用：

```python
outputs = self.engine_core.get_output()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:302` 到 `vllm/vllm/v1/engine/llm_engine.py:304`

如果底层是 `InprocClient`，`get_output()` 会直接驱动 EngineCore 一轮执行：

```python
def get_output(self) -> EngineCoreOutputs:
    outputs, model_executed = self.engine_core.step_fn()
    self.engine_core.post_step(model_executed=model_executed)
    return outputs and outputs.get(0) or EngineCoreOutputs()
```

位置：`vllm/vllm/v1/engine/core_client.py:289` 到 `vllm/vllm/v1/engine/core_client.py:292`

因此同进程同步路径是：

```text
LLMEngine.step()
  → InprocClient.get_output()
  → EngineCore.step_fn()
  → Scheduler.update_from_output()
  → dict[int, EngineCoreOutputs]
  → 取 client 0 的 EngineCoreOutputs
  → OutputProcessor.process_outputs()
```

---

## 8. 多进程输出如何进入 LLMEngine / AsyncLLM

多进程模式下，EngineCore 在后台 `EngineCoreProc` 中运行。

后台 busy loop 产出输出后，会把它放入 output queue：

```python
outputs, model_executed = self.step_fn()
# Put EngineCoreOutputs into the output queue.
for output in outputs.items() if outputs else ():
    self.output_queue.put_nowait(output)
self.post_step(model_executed)
```

位置：`vllm/vllm/v1/engine/core.py:1301` 到 `vllm/vllm/v1/engine/core.py:1307`

这里的 `output` 是：

```text
(client_index, EngineCoreOutputs)
```

output socket 线程再按 `client_index` 发回前端：

```python
client_index, outputs = output
outputs.engine_index = engine_index
...
tracker = sockets[client_index].send_multipart(
    buffers, copy=False, track=True
)
```

位置：`vllm/vllm/v1/engine/core.py:1628` 到 `vllm/vllm/v1/engine/core.py:1646`

所以多进程路径是：

```text
EngineCoreProc.step_fn()
  → EngineCoreOutputs
  → output_queue
  → output socket
  → EngineCoreClient output thread / task
  → frontend queue
  → LLMEngine / AsyncLLM
```

---

## 9. 同步输出路径：LLMEngine.step()

同步输出入口是：

```python
def step(self) -> list[RequestOutput | PoolingRequestOutput]:
```

位置：`vllm/vllm/v1/engine/llm_engine.py:296`

主流程：

```text
LLMEngine.step()
  → 如需 DP dummy batch，执行并返回 []
  → EngineCoreClient.get_output()
  → OutputProcessor.process_outputs()
  → OutputProcessor.update_scheduler_stats()
  → 对 stop string 触发的请求调用 EngineCore abort
  → logger_manager.record()
  → return processed_outputs.request_outputs
```

源码对应：

```python
# 1) Get EngineCoreOutput from the EngineCore.
outputs = self.engine_core.get_output()

# 2) Process EngineCoreOutputs.
iteration_stats = IterationStats() if self.log_stats else None
processed_outputs = self.output_processor.process_outputs(
    outputs.outputs,
    engine_core_timestamp=outputs.timestamp,
    iteration_stats=iteration_stats,
)
self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

# 3) Abort any reqs that finished due to stop strings.
self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

# 4) Record stats
...

return processed_outputs.request_outputs
```

位置：`vllm/vllm/v1/engine/llm_engine.py:302` 到 `vllm/vllm/v1/engine/llm_engine.py:334`

同步路径的特点是：

```text
OutputProcessor.process_outputs() 直接返回 request_outputs 列表；
LLMEngine.step() 把这个列表返回给调用方。
```

---

## 10. 异步输出路径：AsyncLLM output_handler

异步路径不是用户主动调用 step，而是后台 output handler 持续拉取 EngineCoreOutputs。

入口：

```python
def _run_output_handler(self):
    """Background loop: pulls from EngineCore and pushes to AsyncStreams."""
```

位置：`vllm/vllm/v1/engine/async_llm.py:637` 到 `vllm/vllm/v1/engine/async_llm.py:638`

output handler 主循环：

```python
while True:
    # 1) Pull EngineCoreOutputs from the EngineCore.
    outputs = await engine_core.get_output_async()
    num_outputs = len(outputs.outputs)
    ...
    for start in range(0, num_outputs, chunk_size):
        outputs_slice = engine_core_outputs[start:end]
        # 2) Process EngineCoreOutputs.
        processed_outputs = output_processor.process_outputs(
            outputs_slice, outputs.timestamp, iteration_stats
        )
        assert not processed_outputs.request_outputs
        ...
        # 3) Abort any reqs that finished due to stop strings.
        if processed_outputs.reqs_to_abort:
            await engine_core.abort_requests_async(
                processed_outputs.reqs_to_abort
            )
    output_processor.update_scheduler_stats(outputs.scheduler_stats)
    ... logging ...
```

位置：`vllm/vllm/v1/engine/async_llm.py:656` 到 `vllm/vllm/v1/engine/async_llm.py:702`

异步路径的特点是：

```text
OutputProcessor.process_outputs() 不返回给 output_handler；
它会把 RequestOutput 放进每个请求自己的 RequestOutputCollector queue。
```

因此这里有断言：

```python
assert not processed_outputs.request_outputs
```

位置：`vllm/vllm/v1/engine/async_llm.py:678` 到 `vllm/vllm/v1/engine/async_llm.py:679`

---

## 11. 异步路径为什么要分 chunk 处理输出

AsyncLLM output handler 处理输出时会分片：

```python
chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
...
for start in range(0, num_outputs, chunk_size):
    end = start + chunk_size
    outputs_slice = engine_core_outputs[start:end]
    processed_outputs = output_processor.process_outputs(...)
    ...
    if end < num_outputs:
        await asyncio.sleep(0)
```

位置：`vllm/vllm/v1/engine/async_llm.py:653` 到 `vllm/vllm/v1/engine/async_llm.py:683`

注释说明：

```python
# Split outputs into chunks of at most
# VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
# event loop for too long.
```

位置：`vllm/vllm/v1/engine/async_llm.py:667` 到 `vllm/vllm/v1/engine/async_llm.py:669`

所以异步路径为了避免一次处理过多输出阻塞 event loop，会：

```text
把 EngineCoreOutputs.outputs 切成小块；
每块调用一次 OutputProcessor.process_outputs()；
块之间 await asyncio.sleep(0) 让出事件循环。
```

---

## 12. OutputProcessor 的职责

`OutputProcessor` 定义：

```python
class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""
```

位置：`vllm/vllm/v1/engine/output_processor.py:417` 到 `vllm/vllm/v1/engine/output_processor.py:418`

它初始化时维护这些状态：

```python
self.request_states: dict[str, RequestState] = {}
self.parent_requests: dict[str, ParentRequest] = {}
self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
self.lora_states = LoRARequestStates(log_stats)
```

位置：`vllm/vllm/v1/engine/output_processor.py:431` 到 `vllm/vllm/v1/engine/output_processor.py:434`

这些状态的作用是：

```text
request_states：
  internal request_id → RequestState，用于把内部输出转成外部输出。

parent_requests：
  parallel sampling 父请求状态，用于 n > 1 输出合并。

external_req_ids：
  external request id → internal request ids，用于 abort 和输出 id 映射。

lora_states：
  统计 LoRA request 状态。
```

---

## 13. RequestState：外层输出状态

每个请求在进入 EngineCore 前，会先在 `OutputProcessor` 中注册为 `RequestState`。

`RequestState` 保存：

```python
request_id: str
external_req_id: str
parent_req: ParentRequest | None
request_index: int
lora_request: LoRARequest | None
output_kind: RequestOutputKind
prompt: str | None
prompt_token_ids: list[int] | None
prompt_embeds: torch.Tensor | None
logprobs_processor: LogprobsProcessor | None
detokenizer: IncrementalDetokenizer | None
max_tokens_param: int | None
arrival_time: float
queue: RequestOutputCollector | None
log_stats: bool
stream_interval: int
```

位置：`vllm/vllm/v1/engine/output_processor.py:129` 到 `vllm/vllm/v1/engine/output_processor.py:151`

它还维护运行中的输出状态：

```text
is_prefilling：是否仍处于第一次 prefill 输出阶段；
num_cached_tokens：prefix cache 命中 token 数；
routed_experts_chunks：累积 routed experts；
sent_tokens_offset：stream interval / delta 输出偏移；
input_chunk_queue：streaming input 后续 chunk 队列。
```

相关位置：`vllm/vllm/v1/engine/output_processor.py:172` 到 `vllm/vllm/v1/engine/output_processor.py:189`

所以：

```text
EngineCore 只知道内部 request 状态；
OutputProcessor.RequestState 保存的是构造外部输出所需的状态。
```

---

## 14. RequestOutputCollector：异步输出队列

异步请求会创建 `RequestOutputCollector`。

定义：

```python
class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """
```

位置：`vllm/vllm/v1/engine/output_processor.py:45` 到 `vllm/vllm/v1/engine/output_processor.py:52`

它保存：

```python
self.aggregate = output_kind == RequestOutputKind.DELTA
self.request_id = request_id
self.output: RequestOutput | PoolingRequestOutput | Exception | None = None
self.ready = asyncio.Event()
```

位置：`vllm/vllm/v1/engine/output_processor.py:54` 到 `vllm/vllm/v1/engine/output_processor.py:58`

`put()` 会把输出放进 collector：

```python
if self.output is None or isinstance(output, Exception):
    self.output = output
    self.ready.set()
elif isinstance(self.output, RequestOutput) and isinstance(output, RequestOutput):
    self.output.add(output, aggregate=self.aggregate)
elif isinstance(self.output, PoolingRequestOutput) and isinstance(output, PoolingRequestOutput):
    self.output = output
```

位置：`vllm/vllm/v1/engine/output_processor.py:62` 到 `vllm/vllm/v1/engine/output_processor.py:76`

含义是：

```text
如果消费者来不及取，新的 RequestOutput 可以和已有 output 合并；
DELTA 模式下按 delta 聚合；
Exception 会直接传给消费者。
```

消费者通过：

```python
async def get(self) -> RequestOutput | PoolingRequestOutput:
```

位置：`vllm/vllm/v1/engine/output_processor.py:78`

等待输出。

---

## 15. OutputProcessor.add_request()：输出生命周期的注册起点

输出生命周期不是从 `process_outputs()` 才开始，而是在请求进入 EngineCore 前就开始。

`OutputProcessor.add_request()` 入口：

```python
def add_request(
    self,
    request: EngineCoreRequest,
    prompt: str | None,
    parent_req: ParentRequest | None = None,
    request_index: int = 0,
    queue: RequestOutputCollector | None = None,
) -> None:
```

位置：`vllm/vllm/v1/engine/output_processor.py:512` 到 `vllm/vllm/v1/engine/output_processor.py:519`

普通新请求会创建 `RequestState`：

```python
req_state = RequestState.from_new_request(
    tokenizer=self.tokenizer,
    request=request,
    prompt=prompt,
    parent_req=parent_req,
    request_index=request_index,
    queue=queue,
    log_stats=self.log_stats,
    stream_interval=self.stream_interval,
)
self.request_states[request_id] = req_state
```

位置：`vllm/vllm/v1/engine/output_processor.py:526` 到 `vllm/vllm/v1/engine/output_processor.py:536`

并记录外部 id 到内部 id 的映射：

```python
self.external_req_ids[req_state.external_req_id].append(request_id)
```

位置：`vllm/vllm/v1/engine/output_processor.py:540` 到 `vllm/vllm/v1/engine/output_processor.py:541`

所以：

```text
OutputProcessor 必须先 add_request，后续才能 process EngineCoreOutput。
```

否则 `process_outputs()` 找不到 `RequestState`，会忽略该输出。

---

## 16. RequestState.from_new_request() 如何初始化输出状态

`RequestState.from_new_request()` 会根据请求类型初始化不同状态。

如果是 generation 请求：

```python
if sampling_params := request.sampling_params:
    if not sampling_params.detokenize:
        tokenizer = None
    output_kind = sampling_params.output_kind
    logprobs_processor = LogprobsProcessor.from_new_request(...)
    detokenizer = IncrementalDetokenizer.from_new_request(...)
    max_tokens_param = sampling_params.max_tokens
    top_p = sampling_params.top_p
    n = sampling_params.n
    temperature = sampling_params.temperature
```

位置：`vllm/vllm/v1/engine/output_processor.py:222` 到 `vllm/vllm/v1/engine/output_processor.py:237`

如果是 pooling 请求：

```python
else:
    logprobs_processor = None
    detokenizer = None
    max_tokens_param = None
    top_p = None
    n = None
    temperature = None
    assert request.pooling_params is not None
    output_kind = request.pooling_params.output_kind
```

位置：`vllm/vllm/v1/engine/output_processor.py:238` 到 `vllm/vllm/v1/engine/output_processor.py:246`

所以：

```text
generation request：
  需要 detokenizer 和 logprobs_processor。

pooling request：
  不需要 detokenizer，直接输出 PoolingOutput。
```

`RequestState` 必须保存 `external_req_id`：

```python
assert request.external_req_id is not None
```

位置：`vllm/vllm/v1/engine/output_processor.py:248`

因为最终 `RequestOutput.request_id` 使用的是用户外部 id，而不是内部随机 id。

---

## 17. process_outputs() 总流程

核心函数是：

```python
def process_outputs(
    self,
    engine_core_outputs: list[EngineCoreOutput],
    engine_core_timestamp: float | None = None,
    iteration_stats: IterationStats | None = None,
) -> OutputProcessorOutput:
```

位置：`vllm/vllm/v1/engine/output_processor.py:576` 到 `vllm/vllm/v1/engine/output_processor.py:581`

源码注释概括它做三件事：

```python
"""
Process the EngineCoreOutputs:
1) Compute stats for logging
2) Detokenize
3) Create and handle RequestOutput objects:
    * If there is a queue (for usage with AsyncLLM),
      put the RequestOutput objects into the queue for
      handling by the per-request generate() tasks.

    * If there is no queue (for usage with LLMEngine),
      return a list of RequestOutput objects.
"""
```

位置：`vllm/vllm/v1/engine/output_processor.py:582` 到 `vllm/vllm/v1/engine/output_processor.py:592`

完整流程可以概括为：

```text
process_outputs()
  → 初始化 request_outputs / reqs_to_abort
  → 遍历每个 EngineCoreOutput
  → 用 internal request_id 找 RequestState
  → 更新 stats
  → 读取 new_token_ids / pooling_output / finish_reason / stop_reason / kv_transfer_params
  → 累积 routed experts
  → 记录 prefill num_cached_tokens
  → generation 请求 detokenize 并检查 stop string
  → 更新 logprobs
  → RequestState.make_request_output()
  → 同步：append 到 request_outputs
  → 异步：put 到 RequestOutputCollector
  → finish_reason != None 时处理完成 / streaming continuation
  → 返回 OutputProcessorOutput(request_outputs, reqs_to_abort)
```

---

## 18. process_outputs() 为什么只循环一次

`process_outputs()` 注释强调：

```python
NOTE FOR DEVELOPERS

vLLM V1 minimizes the number of python loops over the full
batch to ensure system overheads are minimized. This is the
only function that should loop over EngineCoreOutputs.
```

位置：`vllm/vllm/v1/engine/output_processor.py:594` 到 `vllm/vllm/v1/engine/output_processor.py:598`

含义是：

```text
为了降低 Python 侧批处理开销，所有需要逐请求处理 EngineCoreOutput 的工作，都应该合并在 process_outputs() 的同一个循环里完成。
```

所以它同时处理：

```text
stats；
detokenize；
logprobs；
stop string；
routed experts；
prefill stats；
RequestOutput 构造；
finish 清理。
```

---

## 19. 第一步：根据 request_id 找 RequestState

循环开头：

```python
for engine_core_output in engine_core_outputs:
    req_id = engine_core_output.request_id
    req_state = self.request_states.get(req_id)
    if req_state is None:
        # Ignore output for already-aborted request.
        continue
```

位置：`vllm/vllm/v1/engine/output_processor.py:606` 到 `vllm/vllm/v1/engine/output_processor.py:611`

这里用的是内部 request id。

如果找不到 `RequestState`，说明外层已经 abort 或清理过该请求，直接忽略输出。

这解释了一个重要边界：

```text
EngineCore 可能仍返回某个请求的输出；
但如果外层 OutputProcessor 已经删除 RequestState，则该输出不会再返回给用户。
```

---

## 20. 第二步：更新 iteration stats

如果开启 stats，会调用：

```python
self._update_stats_from_output(
    req_state, engine_core_output, engine_core_timestamp, iteration_stats
)
```

位置：`vllm/vllm/v1/engine/output_processor.py:613` 到 `vllm/vllm/v1/engine/output_processor.py:616`

内部会调用：

```python
iteration_stats.update_from_output(
    engine_core_output,
    engine_core_timestamp,
    req_state.is_prefilling,
    req_state.stats,
    self.lora_states,
    req_state.lora_name,
)
```

位置：`vllm/vllm/v1/engine/output_processor.py:786` 到 `vllm/vllm/v1/engine/output_processor.py:793`

这一步把 EngineCore 输出中的事件、时间戳、LoRA 信息、prefill/decode 阶段信息合并到前端 stats。

---

## 21. 第三步：读取 EngineCoreOutput 字段

`process_outputs()` 从内部输出中取出本轮字段：

```python
new_token_ids = engine_core_output.new_token_ids
pooling_output = engine_core_output.pooling_output
finish_reason = engine_core_output.finish_reason
stop_reason = engine_core_output.stop_reason
kv_transfer_params = engine_core_output.kv_transfer_params
```

位置：`vllm/vllm/v1/engine/output_processor.py:618` 到 `vllm/vllm/v1/engine/output_processor.py:622`

如果有 routed experts，会先累积起来：

```python
if engine_core_output.routed_experts is not None:
    req_state.routed_experts_chunks.append(
        engine_core_output.routed_experts
    )
```

位置：`vllm/vllm/v1/engine/output_processor.py:623` 到 `vllm/vllm/v1/engine/output_processor.py:626`

routed experts 最终在请求 finished 时拼接到 `CompletionOutput`。

---

## 22. 第四步：记录 prefill_stats / num_cached_tokens

如果 `RequestState` 仍处于 prefilling 阶段：

```python
if req_state.is_prefilling:
    if engine_core_output.prefill_stats is not None:
        req_state.num_cached_tokens = (
            engine_core_output.prefill_stats.num_cached_tokens
        )
    req_state.is_prefilling = False
```

位置：`vllm/vllm/v1/engine/output_processor.py:628` 到 `vllm/vllm/v1/engine/output_processor.py:633`

含义是：

```text
第一次 EngineCoreOutput 到来时，如果带有 prefill_stats，
OutputProcessor 会记录 num_cached_tokens，
后续放入 RequestOutput / PoolingRequestOutput。
```

这让用户可见输出能知道 prefix cache 命中 token 数。

---

## 23. 第五步：generation 输出 detokenize 和 stop string 检查

如果不是 pooling output，就按 generation 处理：

```python
if pooling_output is None:
    assert req_state.detokenizer is not None
    assert req_state.logprobs_processor is not None
    # 2) Detokenize the token ids into text and perform stop checks.
    stop_string = req_state.detokenizer.update(
        new_token_ids, finish_reason == FinishReason.STOP
    )
    if stop_string:
        finish_reason = FinishReason.STOP
        stop_reason = stop_string
```

位置：`vllm/vllm/v1/engine/output_processor.py:635` 到 `vllm/vllm/v1/engine/output_processor.py:645`

这一步很关键：

```text
Scheduler 能处理 token 级 stop；
OutputProcessor 才能处理文本级 stop string。
```

如果 detokenizer 检出 stop string，会把：

```text
finish_reason = STOP
stop_reason = stop string
```

但此时 EngineCore 内部不一定已经知道这个请求要停止。

因此后面可能会产生 `reqs_to_abort`，让外层 Engine 反向通知 EngineCore abort。

---

## 24. 第六步：更新 logprobs

generation 输出会更新 logprobs：

```python
req_state.logprobs_processor.update_from_output(engine_core_output)
```

位置：`vllm/vllm/v1/engine/output_processor.py:646` 到 `vllm/vllm/v1/engine/output_processor.py:648`

`LogprobsProcessor` 会根据 `EngineCoreOutput.new_logprobs` 和 `new_prompt_logprobs_tensors` 更新请求级 logprobs 状态。

最终 `RequestOutput` 中的：

```text
prompt_logprobs
CompletionOutput.logprobs
CompletionOutput.cumulative_logprob
```

都来自这里维护的状态。

---

## 25. 第七步：RequestState.make_request_output()

核心转换发生在：

```python
if request_output := req_state.make_request_output(
    new_token_ids,
    pooling_output,
    finish_reason,
    stop_reason,
    kv_transfer_params,
):
    ...
```

位置：`vllm/vllm/v1/engine/output_processor.py:650` 到 `vllm/vllm/v1/engine/output_processor.py:657`

`make_request_output()` 会根据输出类型决定是否返回用户可见输出。

入口：

```python
def make_request_output(
    self,
    new_token_ids: list[int],
    pooling_output: torch.Tensor | None,
    finish_reason: FinishReason | None,
    stop_reason: int | str | None,
    kv_transfer_params: dict[str, Any] | None = None,
) -> RequestOutput | PoolingRequestOutput | None:
```

位置：`vllm/vllm/v1/engine/output_processor.py:272` 到 `vllm/vllm/v1/engine/output_processor.py:279`

---

## 26. FINAL_ONLY 输出策略

如果用户要求 final only，并且请求还没结束：

```python
finished = finish_reason is not None
final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

if not finished and final_only:
    # Only the final output is required in FINAL_ONLY mode.
    return None
```

位置：`vllm/vllm/v1/engine/output_processor.py:280` 到 `vllm/vllm/v1/engine/output_processor.py:285`

含义是：

```text
FINAL_ONLY 模式下，中间 token 不返回给用户；
只有最终 finish 时才返回完整 RequestOutput。
```

---

## 27. stream_interval 输出节流

如果设置了 `stream_interval > 1`：

```python
if self.stream_interval > 1:
    assert self.detokenizer is not None

    # Send output request only when
    # 1. It has finished, or
    # 2. It is the first token, or
    # 3. It has reached the stream interval number of tokens
    if not (
        finished
        or self.sent_tokens_offset == 0
        or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
        >= self.stream_interval
    ):
        return None
```

位置：`vllm/vllm/v1/engine/output_processor.py:287` 到 `vllm/vllm/v1/engine/output_processor.py:300`

这说明输出不是每个 token 都一定返回。

返回条件是：

```text
请求 finished；
或这是第一次输出；
或自上次发送后累计 token 数达到 stream_interval。
```

如果是 DELTA 模式，还会只发送 offset 后的新 token：

```python
if self.output_kind == RequestOutputKind.DELTA:
    new_token_ids = self.detokenizer.output_token_ids[
        self.sent_tokens_offset :
    ]
    self.sent_tokens_offset = self.detokenizer.num_output_tokens()
```

位置：`vllm/vllm/v1/engine/output_processor.py:302` 到 `vllm/vllm/v1/engine/output_processor.py:308`

---

## 28. Pooling 输出如何构造

如果 `pooling_output is not None`：

```python
return self._new_request_output(
    external_req_id,
    [self._new_pooling_output(pooling_output)],
    finished,
)
```

位置：`vllm/vllm/v1/engine/output_processor.py:312` 到 `vllm/vllm/v1/engine/output_processor.py:317`

`_new_pooling_output()` 很简单：

```python
def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
    return PoolingOutput(data=pooling_output)
```

位置：`vllm/vllm/v1/engine/output_processor.py:413` 到 `vllm/vllm/v1/engine/output_processor.py:414`

最终 `_new_request_output()` 会返回 `PoolingRequestOutput`：

```python
return PoolingRequestOutput(
    request_id=external_req_id,
    outputs=first_output,
    num_cached_tokens=self.num_cached_tokens,
    prompt_token_ids=prompt_token_ids,
    finished=finished,
)
```

位置：`vllm/vllm/v1/engine/output_processor.py:347` 到 `vllm/vllm/v1/engine/output_processor.py:355`

所以 pooling 路径是：

```text
EngineCoreOutput.pooling_output: torch.Tensor
  → PoolingOutput(data=tensor)
  → PoolingRequestOutput
```

---

## 29. Generation 输出如何构造

普通 generation 会先构造 `CompletionOutput`：

```python
output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)
```

位置：`vllm/vllm/v1/engine/output_processor.py:319`

`_new_completion_output()` 内部会取 detokenized text：

```python
text = self.detokenizer.get_next_output_text(finished, delta)
if not delta:
    token_ids = self.detokenizer.output_token_ids
```

位置：`vllm/vllm/v1/engine/output_processor.py:387` 到 `vllm/vllm/v1/engine/output_processor.py:390`

如果是 DELTA 模式：

```text
text 是本次新增文本；
token_ids 是本次新增 token ids。
```

如果不是 DELTA：

```text
text 通常是累计文本；
token_ids 会替换为累计 output_token_ids。
```

logprobs 也按 delta 裁剪：

```python
logprobs = self.logprobs_processor.logprobs
if delta and logprobs:
    logprobs = logprobs[-len(token_ids) :]
```

位置：`vllm/vllm/v1/engine/output_processor.py:392` 到 `vllm/vllm/v1/engine/output_processor.py:395`

最后返回：

```python
return CompletionOutput(
    index=self.request_index,
    text=text,
    token_ids=token_ids,
    routed_experts=routed_experts,
    logprobs=logprobs,
    cumulative_logprob=self.logprobs_processor.cumulative_logprob,
    finish_reason=str(finish_reason) if finished else None,
    stop_reason=stop_reason if finished else None,
)
```

位置：`vllm/vllm/v1/engine/output_processor.py:402` 到 `vllm/vllm/v1/engine/output_processor.py:411`

---

## 30. RequestOutput 如何使用 external request id

`_new_request_output()` 返回 `RequestOutput` 时：

```python
return RequestOutput(
    request_id=external_req_id,  # request_id is what was provided externally
    lora_request=self.lora_request,
    prompt=self.prompt,
    prompt_token_ids=prompt_token_ids,
    prompt_logprobs=prompt_logprobs,
    outputs=cast(list[CompletionOutput], outputs),
    finished=finished,
    kv_transfer_params=kv_transfer_params,
    num_cached_tokens=self.num_cached_tokens,
    metrics=self.stats,
)
```

位置：`vllm/vllm/v1/engine/output_processor.py:363` 到 `vllm/vllm/v1/engine/output_processor.py:374`

这里特别重要：

```text
EngineCoreOutput.request_id 是内部 request id；
RequestOutput.request_id 是 external_req_id，也就是用户传入的 id。
```

这就是 OutputProcessor 必须保存 external / internal id 映射的原因。

---

## 31. prompt_embeds 输出里的 prompt_token_ids 占位

如果请求使用 prompt embeds，可能没有真实 prompt token ids。

`_new_request_output()` 会处理：

```python
prompt_token_ids = self.prompt_token_ids
if prompt_token_ids is None and self.prompt_embeds is not None:
    prompt_token_ids = [0] * len(self.prompt_embeds)
assert prompt_token_ids is not None
```

位置：`vllm/vllm/v1/engine/output_processor.py:340` 到 `vllm/vllm/v1/engine/output_processor.py:344`

所以用户输出中的 `prompt_token_ids` 在 prompt embeds 场景下可能是占位 0 列表。

---

## 32. parallel sampling 输出如何合并

如果请求来自 `n > 1` parallel sampling，会有 `parent_req`：

```python
if self.parent_req is None:
    outputs = [output]
else:
    outputs, finished = self.parent_req.get_outputs(self.request_id, output)
    if not outputs:
        return None
    external_req_id = self.parent_req.external_req_id
```

位置：`vllm/vllm/v1/engine/output_processor.py:321` 到 `vllm/vllm/v1/engine/output_processor.py:327`

含义是：

```text
EngineCore / Scheduler 看到的是多个 child request；
OutputProcessor / ParentRequest 负责把多个 child 的 CompletionOutput 合并成一个外部 RequestOutput。
```

如果当前 child 的输出还不足以形成对外输出，`get_outputs()` 可能返回空，`make_request_output()` 返回 None。

---

## 33. 同步和异步如何分发 RequestOutput

`process_outputs()` 得到 `request_output` 后：

```python
if req_state.streaming_input:
    request_output.finished = False

if req_state.queue is not None:
    # AsyncLLM: put into queue for handling by generate().
    req_state.queue.put(request_output)
else:
    # LLMEngine: return list of RequestOutputs.
    request_outputs.append(request_output)
```

位置：`vllm/vllm/v1/engine/output_processor.py:658` 到 `vllm/vllm/v1/engine/output_processor.py:666`

所以分发规则是：

```text
同步 LLMEngine：
  RequestState.queue is None
  → append 到 request_outputs
  → LLMEngine.step() 返回给用户。

异步 AsyncLLM：
  RequestState.queue is RequestOutputCollector
  → queue.put(request_output)
  → async generator 从 collector.get() 取出并 yield。
```

streaming input 场景下，即使当前 chunk 结束，也会把 `request_output.finished` 改成 False，因为整个 streaming session 可能还没结束。

---

## 34. finish 后如何清理 RequestState

`process_outputs()` 在每个输出处理完后检查：

```python
if finish_reason is not None:
    if req_state.streaming_input:
        if req_state.input_chunk_queue:
            update = req_state.input_chunk_queue.popleft()
            req_state.apply_streaming_update(update)
        else:
            req_state.input_chunk_queue = None
    else:
        self._finish_request(req_state)
        if not engine_core_output.finished:
            # If req not finished in EngineCore, but Detokenizer
            # detected stop string, abort needed in EngineCore.
            reqs_to_abort.append(req_id)
        ... stats / tracing ...
```

位置：`vllm/vllm/v1/engine/output_processor.py:668` 到 `vllm/vllm/v1/engine/output_processor.py:688`

这说明 finish 后有两条路径：

```text
普通请求：
  _finish_request() 清理 OutputProcessor 状态。

streaming input 请求：
  如果还有 input chunk，应用下一段输入；
  如果没有 chunk，等待后续输入或 final request。
```

---

## 35. _finish_request() 清理哪些状态

`_finish_request()`：

```python
def _finish_request(self, req_state: RequestState) -> None:
    req_id = req_state.request_id
    self.request_states.pop(req_id)

    internal_ids = self.external_req_ids[req_state.external_req_id]
    internal_ids.remove(req_id)
    if not internal_ids:
        del self.external_req_ids[req_state.external_req_id]

    # Remove parent request if applicable.
    parent_req = req_state.parent_req
    if parent_req and not parent_req.child_requests:
        self.parent_requests.pop(parent_req.request_id, None)
```

位置：`vllm/vllm/v1/engine/output_processor.py:695` 到 `vllm/vllm/v1/engine/output_processor.py:707`

它清理三类外层状态：

```text
request_states：
  删除 internal request id 对应的 RequestState。

external_req_ids：
  从 external → internal 映射中删除该 internal id。

parent_requests：
  如果 parallel sampling 的 parent 已没有 child，删除 parent 状态。
```

注意：

```text
这只是 OutputProcessor 外层状态清理；
Scheduler 内部 Request / KV blocks 的释放由 Scheduler.update_from_output() / EngineCore 负责。
```

---

## 36. stop string 触发后为什么要 abort EngineCore

一个容易混淆的路径是：

```python
if not engine_core_output.finished:
    # If req not finished in EngineCore, but Detokenizer
    # detected stop string, abort needed in EngineCore.
    reqs_to_abort.append(req_id)
```

位置：`vllm/vllm/v1/engine/output_processor.py:677` 到 `vllm/vllm/v1/engine/output_processor.py:681`

原因是：

```text
文本级 stop string 只能在 detokenize 后判断；
Scheduler / EngineCore 可能只知道 token 级 stop 条件；
因此 OutputProcessor 可能先发现“用户语义上该停了”。
```

此时外层需要通知 EngineCore：

同步路径：

```python
self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:316` 到 `vllm/vllm/v1/engine/llm_engine.py:318`

异步路径：

```python
if processed_outputs.reqs_to_abort:
    await engine_core.abort_requests_async(
        processed_outputs.reqs_to_abort
    )
```

位置：`vllm/vllm/v1/engine/async_llm.py:685` 到 `vllm/vllm/v1/engine/async_llm.py:689`

所以 stop string 的输出生命周期有一个反向控制流：

```text
EngineCoreOutput
  → OutputProcessor detokenize
  → 发现 stop string
  → 对用户返回 finished output
  → reqs_to_abort
  → EngineCore.abort_requests()
```

---

## 37. streaming input 输出生命周期

streaming input 请求的输出生命周期更复杂。

当 `OutputProcessor.add_request()` 收到同一个 request id 的后续 request 时：

```python
req_state = self.request_states.get(request_id)
if req_state is not None:
    self._update_streaming_request_state(req_state, request, prompt)
    return
```

位置：`vllm/vllm/v1/engine/output_processor.py:520` 到 `vllm/vllm/v1/engine/output_processor.py:524`

如果是 resumable chunk，会构造 `StreamingUpdate`：

```python
update = StreamingUpdate(
    prompt=prompt,
    prompt_token_ids=request.prompt_token_ids,
    arrival_time=request.arrival_time,
)
```

位置：`vllm/vllm/v1/engine/output_processor.py:562` 到 `vllm/vllm/v1/engine/output_processor.py:566`

如果上一段已经结束，会立即 apply：

```python
if req_state.input_chunk_queue is None:
    req_state.apply_streaming_update(update)
    req_state.input_chunk_queue = deque()
else:
    req_state.input_chunk_queue.append(update)
```

位置：`vllm/vllm/v1/engine/output_processor.py:568` 到 `vllm/vllm/v1/engine/output_processor.py:574`

当某个 chunk 的 output finish 后：

```text
如果 input_chunk_queue 有下一段：
  apply_streaming_update(update)，继续这个 session。

如果 input_chunk_queue 为空：
  input_chunk_queue = None，等待后续输入。

如果 final request 到达且 engine 已结束：
  _finish_request() 并向 queue put STREAM_FINISHED。
```

相关位置：`vllm/vllm/v1/engine/output_processor.py:543` 到 `vllm/vllm/v1/engine/output_processor.py:560`，`vllm/vllm/v1/engine/output_processor.py:668` 到 `vllm/vllm/v1/engine/output_processor.py:675`

---

## 38. abort 输出生命周期

外部 abort 会先经过 `OutputProcessor.abort_requests()`：

```python
def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
```

位置：`vllm/vllm/v1/engine/output_processor.py:450`

它支持 external id 和 internal id：

```text
internal=False：
  request_ids 是用户外部 id；
  通过 external_req_ids 找到所有 internal ids。

internal=True：
  request_ids 是内部 id，也可能是 parent id。
```

如果请求有 async queue，会生成最终 abort output：

```python
if req_state.queue is not None and (
    request_output := req_state.make_request_output(
        new_token_ids=[],
        pooling_output=EMPTY_CPU_TENSOR
        if req_state.detokenizer is None
        else None,
        finish_reason=FinishReason.ABORT,
        stop_reason=None,
        kv_transfer_params=None,
    )
):
    req_state.queue.put(request_output)
```

位置：`vllm/vllm/v1/engine/output_processor.py:489` 到 `vllm/vllm/v1/engine/output_processor.py:502`

这说明：

```text
异步 abort 可以立即向请求队列放入一个 finished=abort 的最终输出；
同时返回 internal request ids，让 EngineCore 也取消内部请求。
```

同步 abort 路径主要清理 OutputProcessor 状态，并由外层调用 EngineCore abort。

---

## 39. tracing 和 stats 何时完成

请求真正 finish 后，`process_outputs()` 会更新 finished stats：

```python
self._update_stats_from_finished(
    req_state, finish_reason, iteration_stats
)
if self.tracing_enabled:
    self.do_tracing(engine_core_output, req_state, iteration_stats)
```

位置：`vllm/vllm/v1/engine/output_processor.py:683` 到 `vllm/vllm/v1/engine/output_processor.py:688`

`_update_stats_from_finished()` 会记录：

```python
iteration_stats.update_from_finished_request(
    finish_reason=finish_reason,
    request_id=req_state.external_req_id,
    num_prompt_tokens=req_state.prompt_len,
    max_tokens_param=req_state.max_tokens_param,
    req_stats=req_state.stats,
    num_cached_tokens=req_state.num_cached_tokens,
)
self.lora_states.request_finished(req_state.request_id, req_state.lora_name)
```

位置：`vllm/vllm/v1/engine/output_processor.py:804` 到 `vllm/vllm/v1/engine/output_processor.py:814`

tracing 会使用 `trace_headers` 和请求 stats 构造 span：

```python
trace_context = extract_trace_context(engine_core_output.trace_headers)
...
instrument_manual(
    span_name="llm_request",
    start_time=arrival_time_ns,
    attributes=attributes,
    context=trace_context,
    kind=SpanKind.SERVER,
)
```

位置：`vllm/vllm/v1/engine/output_processor.py:721` 到 `vllm/vllm/v1/engine/output_processor.py:772`

---

## 40. 同步输出完整流程图

```text
LLMEngine.step()
  │
  ├─ EngineCoreClient.get_output()
  │    ├─ InprocClient:
  │    │    └─ EngineCore.step_fn() → EngineCoreOutputs
  │    └─ SyncMPClient:
  │         └─ outputs_queue.get() → EngineCoreOutputs
  │
  ├─ OutputProcessor.process_outputs(outputs.outputs)
  │    ├─ for EngineCoreOutput in outputs:
  │    │    ├─ request_id → RequestState
  │    │    ├─ update stats
  │    │    ├─ detokenize new_token_ids
  │    │    ├─ check stop string
  │    │    ├─ update logprobs
  │    │    ├─ make RequestOutput / PoolingRequestOutput
  │    │    ├─ append to request_outputs
  │    │    └─ finish ? cleanup RequestState
  │    └─ return OutputProcessorOutput
  │
  ├─ OutputProcessor.update_scheduler_stats()
  ├─ EngineCoreClient.abort_requests(reqs_to_abort)
  ├─ logger_manager.record()
  └─ return list[RequestOutput | PoolingRequestOutput]
```

---

## 41. 异步输出完整流程图

```text
AsyncLLM.add_request()
  → RequestOutputCollector
  → OutputProcessor.add_request(..., queue)
  → EngineCoreClient.add_request_async()

AsyncLLM output_handler task
  │
  ├─ await EngineCoreClient.get_output_async()
  │    └─ EngineCoreOutputs
  │
  ├─ split outputs by VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
  │
  ├─ OutputProcessor.process_outputs(outputs_slice)
  │    ├─ request_id → RequestState
  │    ├─ detokenize / logprobs / stop string
  │    ├─ make RequestOutput / PoolingRequestOutput
  │    └─ queue.put(request_output)
  │
  ├─ await engine_core.abort_requests_async(reqs_to_abort)
  ├─ OutputProcessor.update_scheduler_stats()
  └─ logger_manager.record()

Async generator
  → await RequestOutputCollector.get()
  → yield RequestOutput / PoolingRequestOutput
```

---

## 42. 输出对象转换完整流程图

```text
EngineCoreOutput
  ├─ request_id
  ├─ new_token_ids
  ├─ new_logprobs
  ├─ pooling_output
  ├─ finish_reason
  ├─ stop_reason
  ├─ prefill_stats
  ├─ kv_transfer_params
  └─ routed_experts

OutputProcessor.process_outputs()
  ├─ request_id → RequestState
  ├─ update stats
  ├─ prefill_stats → num_cached_tokens
  ├─ pooling_output ?
  │    └─ PoolingOutput → PoolingRequestOutput
  └─ generation ?
       ├─ detokenizer.update(new_token_ids)
       ├─ stop string check
       ├─ logprobs_processor.update_from_output()
       ├─ _new_completion_output()
       │    ├─ text
       │    ├─ token_ids
       │    ├─ logprobs
       │    ├─ cumulative_logprob
       │    ├─ finish_reason
       │    └─ stop_reason
       └─ _new_request_output()
            ├─ external request_id
            ├─ prompt / prompt_token_ids
            ├─ prompt_logprobs
            ├─ outputs: list[CompletionOutput]
            ├─ finished
            ├─ kv_transfer_params
            ├─ num_cached_tokens
            └─ metrics
```

---

## 43. 一个完整例子：普通 decode token 返回

假设某个请求内部 id 是：

```text
req-a-12345678
```

外部 id 是：

```text
req-a
```

EngineCore 本轮返回：

```text
EngineCoreOutput(
  request_id="req-a-12345678",
  new_token_ids=[42],
  finish_reason=None,
)
```

OutputProcessor 处理：

```text
1. 用 req-a-12345678 找到 RequestState；
2. detokenizer.update([42])；
3. 没有 stop string；
4. logprobs_processor 更新；
5. _new_completion_output() 生成 CompletionOutput；
6. _new_request_output() 生成 RequestOutput；
7. RequestOutput.request_id = "req-a"；
8. finished=False。
```

同步路径返回：

```text
LLMEngine.step() → [RequestOutput(request_id="req-a", ...)]
```

异步路径则：

```text
RequestOutputCollector.put(RequestOutput(...))
async generator await get() 后 yield。
```

---

## 44. 一个完整例子：stop string 触发结束

假设 EngineCore 返回 token 后：

```text
engine_core_output.finished == False
finish_reason == None
```

但是 detokenizer 发现文本命中了 stop string。

OutputProcessor 会：

```text
1. stop_string = detokenizer.update(...)
2. finish_reason = FinishReason.STOP
3. stop_reason = stop_string
4. 构造 finished=True 的 RequestOutput
5. _finish_request(req_state)
6. 因为 engine_core_output.finished == False，将 req_id 加入 reqs_to_abort
```

随后外层：

```text
LLMEngine.step()
  → engine_core.abort_requests(reqs_to_abort)

AsyncLLM output_handler
  → await engine_core.abort_requests_async(reqs_to_abort)
```

这说明：

```text
用户可见输出可以先 finished；
然后外层再通知 EngineCore 清理内部还未停止的请求。
```

---

## 45. 一个完整例子：pooling output 返回

假设 EngineCore 返回：

```text
EngineCoreOutput(
  request_id="pool-xxx",
  new_token_ids=[],
  pooling_output=tensor(...),
  finish_reason=STOP,
)
```

OutputProcessor 会：

```text
1. 不走 detokenizer；
2. _new_pooling_output(tensor) → PoolingOutput；
3. _new_request_output() → PoolingRequestOutput；
4. 如果 finished，清理 RequestState。
```

返回给用户的是：

```text
PoolingRequestOutput(
  request_id=external_req_id,
  outputs=PoolingOutput(data=tensor),
  finished=True,
)
```

---

## 46. 容易疑惑的点

### 46.1 EngineCoreOutputs 是不是用户最终看到的输出？

不是。

它是 EngineCore 内部输出协议。

用户最终看到的是：

```text
RequestOutput / PoolingRequestOutput
```

中间必须经过：

```text
OutputProcessor.process_outputs()
```

### 46.2 EngineCoreOutput.request_id 为什么不是用户传入的 id？

因为 EngineCore / Scheduler 使用内部随机化后的 request id。

OutputProcessor 通过 `RequestState.external_req_id` 把它转换回用户 id：

```python
request_id=external_req_id
```

位置：`vllm/vllm/v1/engine/output_processor.py:363` 到 `vllm/vllm/v1/engine/output_processor.py:365`

### 46.3 为什么 OutputProcessor 要在请求进入 EngineCore 前 add_request？

因为它需要提前保存：

```text
external_req_id；
prompt；
detokenizer；
logprobs processor；
parent request；
streaming queue；
stats。
```

没有这些状态，EngineCoreOutput 无法转换成 RequestOutput。

### 46.4 为什么有些 EngineCoreOutput 被忽略？

如果 `request_states` 里找不到对应 request id：

```python
if req_state is None:
    # Ignore output for already-aborted request.
    continue
```

位置：`vllm/vllm/v1/engine/output_processor.py:608` 到 `vllm/vllm/v1/engine/output_processor.py:611`

说明外层已经 abort / 清理该请求，因此不再向用户返回输出。

### 46.5 stop string 为什么在 OutputProcessor 处理？

因为 stop string 是文本级条件，必须 detokenize 后才能知道。

Scheduler 更偏 token 级停止条件；OutputProcessor 才有文本增量状态。

### 46.6 为什么 stop string 后还要 abort EngineCore？

因为 EngineCore 可能还不知道文本级 stop string 已触发。

OutputProcessor 会生成 `reqs_to_abort`，外层 Engine 再通知 EngineCore 取消内部请求。

### 46.7 FINAL_ONLY 为什么中间没有输出？

`make_request_output()` 中：

```text
not finished and output_kind == FINAL_ONLY
  → return None
```

所以只有最后 finished 时才返回。

### 46.8 DELTA 和非 DELTA 的区别是什么？

DELTA 模式：

```text
返回本次新增 text / token_ids / logprobs。
```

非 DELTA 模式：

```text
返回累计 text / token_ids / logprobs 状态。
```

### 46.9 AsyncLLM 为什么 process_outputs() 后没有 request_outputs？

因为异步路径每个请求有自己的 `RequestOutputCollector`。

`process_outputs()` 会直接：

```text
req_state.queue.put(request_output)
```

所以 output_handler 里断言：

```python
assert not processed_outputs.request_outputs
```

### 46.10 OutputProcessor 清理 RequestState 是否等于 Scheduler 释放 KV blocks？

不是。

```text
OutputProcessor._finish_request()：
  清理前端输出状态。

Scheduler._free_request() / _free_blocks()：
  清理内部 Request、KV blocks、encoder cache、connector 状态。
```

两者属于不同层。

---

## 47. 从“回答问题”的角度总结

如果要问：

```text
一次输出如何从 EngineCore 回到用户？
```

可以回答：

```text
Worker / ModelRunner 执行后返回 ModelRunnerOutput，
Scheduler.update_from_output() 把它整理成按 request 划分的 EngineCoreOutput，
再封装成按 client_index 分组的 EngineCoreOutputs。

同步 LLMEngine.step() 通过 EngineCoreClient.get_output() 取回 EngineCoreOutputs；
异步 AsyncLLM 的 output_handler 通过 get_output_async() 持续拉取 EngineCoreOutputs。

随后 OutputProcessor.process_outputs() 根据 internal request_id 找到 RequestState，
更新 stats，detokenize token ids，处理 stop string 和 logprobs，
把 pooling tensor 或 completion token 转成 PoolingRequestOutput / RequestOutput。
同步路径把 RequestOutput 列表返回给 step() 调用方；
异步路径把 RequestOutput 放入每个请求的 RequestOutputCollector，由 async generator yield 给用户。
```

核心公式是：

```text
内部输出：
ModelRunnerOutput
  → EngineCoreOutput
  → EngineCoreOutputs

外部输出：
EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

## 48. 最关键流程图

```text
Worker / ModelRunner
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput(request_id=internal_id, new_token_ids, finish_reason, ...)
  → EngineCoreOutputs(outputs=[...], scheduler_stats, timestamp)
```

```text
同步 LLMEngine

LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
       ├─ internal request_id → RequestState
       ├─ detokenizer.update()
       ├─ stop string check
       ├─ logprobs_processor.update_from_output()
       ├─ CompletionOutput / PoolingOutput
       ├─ RequestOutput / PoolingRequestOutput
       └─ finish cleanup / reqs_to_abort
  → EngineCoreClient.abort_requests(reqs_to_abort)
  → return list[RequestOutput | PoolingRequestOutput]
```

```text
异步 AsyncLLM

AsyncLLM output_handler
  → await EngineCoreClient.get_output_async()
  → EngineCoreOutputs
  → split chunks
  → OutputProcessor.process_outputs()
       └─ RequestOutputCollector.put(request_output)
  → await EngineCoreClient.abort_requests_async(reqs_to_abort)

per-request async generator
  → await RequestOutputCollector.get()
  → yield RequestOutput / PoolingRequestOutput
```

---

## 49. 和其它文档的关系

`engine_core/03_step_loop.md` 解释的是：

```text
EngineCore.step() 如何 schedule → execute → update，并得到 EngineCoreOutputs。
```

`scheduler/08_update_after_worker_output.md` 解释的是：

```text
Scheduler.update_from_output() 如何把 ModelRunnerOutput 转成 EngineCoreOutput / EngineCoreOutputs。
```

`06_engine_core_client_bridge.md` 解释的是：

```text
EngineCoreOutputs 如何通过 InprocClient / SyncMPClient / AsyncMPClient 回到前端。
```

本篇关注最后一段：

```text
EngineCoreOutputs 如何被 LLMEngine / AsyncLLM 消费，
再由 OutputProcessor 转成用户可见的 RequestOutput / PoolingRequestOutput。
```
