# 05. OutputProcessor 如何把 EngineCoreOutputs 转成用户输出？

源码位置：`vllm/v1/engine/output_processor.py`

相关源码：

- `vllm/v1/engine/llm_engine.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/engine/detokenizer.py`
- `vllm/v1/engine/logprobs.py`

本问题关注：`OutputProcessor` 在外层 Engine 中的位置，以及它如何把内部输出协议转换成用户可见的 `RequestOutput` / `PoolingRequestOutput`。

---

## 1. 一句话回答

`OutputProcessor` 是外层 Engine 的输出处理组件。

源码定位非常直接：

```python
class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""
```

位置：`vllm/v1/engine/output_processor.py:424` 到 `vllm/v1/engine/output_processor.py:425`

可以理解为：

```text
OutputProcessor = EngineCoreOutputs → RequestOutput / PoolingRequestOutput 的转换器。
```

它不负责模型执行，也不负责 Scheduler 状态更新。

它主要负责：

```text
1. 维护外层请求输出状态 RequestState；
2. 接收 EngineCoreOutput；
3. 对生成 token 做 detokenize；
4. 处理 stop string；
5. 处理 logprobs / prompt logprobs；
6. 构造 CompletionOutput / PoolingOutput；
7. 构造 RequestOutput / PoolingRequestOutput；
8. 在同步 LLMEngine 中返回 request_outputs；
9. 在异步 AsyncLLM 中把输出推入 RequestOutputCollector；
10. 必要时返回 reqs_to_abort，让外层 Engine 通知 EngineCore 取消请求。
```

一句话：

```text
EngineCore 产出内部增量输出；OutputProcessor 把这些内部输出变成用户可见输出。
```

---

## 2. OutputProcessor 位于哪一层

`OutputProcessor` 不在 `EngineCore` 内部，而是在外层 Engine 中创建和使用。

### 2.1 LLMEngine 创建 OutputProcessor

同步 `LLMEngine` 初始化时：

```python
# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(
    renderer.tokenizer,
    log_stats=self.log_stats,
    stream_interval=self.vllm_config.scheduler_config.stream_interval,
    tracing_enabled=tracing_endpoint is not None,
)
```

位置：`vllm/v1/engine/llm_engine.py:96` 到 `vllm/v1/engine/llm_engine.py:102`

注释已经说明边界：

```text
EngineCoreOutputs → RequestOutput
```

### 2.2 AsyncLLM 也创建 OutputProcessor

异步 `AsyncLLM` 初始化时也创建：

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

所以无论同步还是异步，层级都是：

```text
LLMEngine / AsyncLLM
  ├─ OutputProcessor
  └─ EngineCoreClient → EngineCore

add_request 路径：OutputProcessor.add_request(...) → EngineCoreClient.add_request(...)
output 路径：EngineCoreClient.get_output*() → OutputProcessor.process_outputs(...)
```

而不是：

```text
EngineCore
  → OutputProcessor
```

### 2.3 EngineCore 返回的是内部输出协议

EngineCore 返回给外层的是 `EngineCoreOutputs`，其中包含多个 `EngineCoreOutput`。

`EngineCoreOutput` 定义在：

```python
class EngineCoreOutput(...):
    request_id: str
    new_token_ids: list[int]
    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None
    pooling_output: torch.Tensor | None = None
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    ...
```

位置：`vllm/v1/engine/__init__.py:175` 到 `vllm/v1/engine/__init__.py:203`

`EngineCoreOutputs` 定义在：

```python
class EngineCoreOutputs(...):
    engine_index: int = 0
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0
    ...
```

位置：`vllm/v1/engine/__init__.py:221` 到 `vllm/v1/engine/__init__.py:249`

这些对象还不是最终用户输出。

最终输出类型来自 `vllm.outputs`：

```python
from vllm.outputs import (
    STREAM_FINISHED,
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
)
```

位置：`vllm/v1/engine/output_processor.py:14` 到 `vllm/v1/engine/output_processor.py:20`

因此输出分层是：

```text
EngineCoreOutput / EngineCoreOutputs：
  EngineCore 返回给外层 Engine 的内部输出协议。

RequestOutput / PoolingRequestOutput：
  外层 Engine 返回给用户 / API server 的用户可见输出协议。
```

---

## 3. 输出对象转换关系

完整转换链路是：

```text
ModelRunnerOutput
  → Scheduler.update_from_output(scheduler_output, model_output)
  → dict[int, EngineCoreOutputs]
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → CompletionOutput / PoolingOutput
  → RequestOutput / PoolingRequestOutput
```

可以拆成两段：

```text
内部执行段：
  ModelRunnerOutput → Scheduler.update_from_output() → EngineCoreOutputs

外层输出段：
  EngineCoreOutputs → OutputProcessor.process_outputs() → RequestOutput
```

`OutputProcessor` 只处理第二段。

它不关心：

```text
Scheduler 怎么调度；
Worker 怎么 forward；
ModelRunner 怎么 sampling；
KV block 怎么释放；
请求如何从 waiting 变 running。
```

它关心的是：

```text
这个 request_id 对应哪个 RequestState；
新增 token ids 如何转成文本；
是否触发 stop string；
logprobs 如何组装；
当前输出是 delta / cumulative / final only；
输出应该直接返回，还是推入异步 queue；
请求完成后如何清理外层状态。
```

---

## 4. OutputProcessor 的核心状态

`OutputProcessor.__init__()` 初始化了几个关键字段：

```python
self.log_stats = log_stats
self.tokenizer = tokenizer
self.stream_interval = stream_interval
self.request_states: dict[str, RequestState] = {}
self.parent_requests: dict[str, ParentRequest] = {}
self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
self.lora_states = LoRARequestStates(log_stats)
self.tracing_enabled = tracing_enabled
```

位置：`vllm/v1/engine/output_processor.py:427` 到 `vllm/v1/engine/output_processor.py:442`

### 4.1 request_states

```python
self.request_states: dict[str, RequestState] = {}
```

位置：`vllm/v1/engine/output_processor.py:438`

这是最核心的外层输出状态表。

含义：

```text
key：内部 request_id
value：RequestState
```

每个还没完全结束的请求，都会在这里有一个 `RequestState`。

EngineCore 每次返回 `EngineCoreOutput` 时，OutputProcessor 会通过 `request_id` 找到对应状态：

```python
req_id = engine_core_output.request_id
req_state = self.request_states.get(req_id)
```

位置：`vllm/v1/engine/output_processor.py:615` 到 `vllm/v1/engine/output_processor.py:616`

如果找不到：

```python
if req_state is None:
    # Ignore output for already-aborted request.
    continue
```

位置：`vllm/v1/engine/output_processor.py:617` 到 `vllm/v1/engine/output_processor.py:619`

说明：已经在外层 abort / 清理掉的请求，如果后续还有迟到的 EngineCoreOutput，会被忽略。

### 4.2 parent_requests

```python
self.parent_requests: dict[str, ParentRequest] = {}
```

位置：`vllm/v1/engine/output_processor.py:439`

用于处理 `n > 1` 的 parallel sampling。

一个用户请求可能被拆成多个 child request 进入 EngineCore，`ParentRequest` 负责把多个 child 输出聚合回一个外部请求。

在 `RequestState.make_request_output()` 中：

```python
if self.parent_req is None:
    outputs = [output]
else:
    outputs, finished = self.parent_req.get_outputs(self.request_id, output)
    if not outputs:
        return None
    external_req_id = self.parent_req.external_req_id
```

位置：`vllm/v1/engine/output_processor.py:322` 到 `vllm/v1/engine/output_processor.py:328`

含义是：

```text
单请求：一个 CompletionOutput 直接成为 RequestOutput.outputs。
n>1：多个 child CompletionOutput 通过 ParentRequest 聚合。
```

### 4.3 external_req_ids

```python
self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
```

位置：`vllm/v1/engine/output_processor.py:440`

它维护：

```text
外部 request_id → 内部 request_id 列表
```

为什么需要？

因为用户传入的 request_id 和 EngineCore 内部 request_id 可能不同，尤其在：

```text
parallel sampling；
内部生成 request_id；
streaming input；
用户按外部 request id abort。
```

`add_request()` 中会登记映射：

```python
self.external_req_ids[req_state.external_req_id].append(request_id)
```

位置：`vllm/v1/engine/output_processor.py:548` 到 `vllm/v1/engine/output_processor.py:549`

`abort_requests()` 则会根据 external id 找到所有 internal ids：

```python
elif internal_ids := self.external_req_ids.pop(request_id, []):
    internal_req_ids.extend(internal_ids)
```

位置：`vllm/v1/engine/output_processor.py:485` 到 `vllm/v1/engine/output_processor.py:487`

### 4.4 lora_states

```python
self.lora_states = LoRARequestStates(log_stats)
```

位置：`vllm/v1/engine/output_processor.py:441`

用于统计 LoRA 请求状态。

Scheduler stats 更新时：

```python
def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
    self.lora_states.update_scheduler_stats(scheduler_stats)
```

位置：`vllm/v1/engine/output_processor.py:719` 到 `vllm/v1/engine/output_processor.py:720`

请求结束时也会更新：

```python
self.lora_states.request_finished(req_state.request_id, req_state.lora_name)
```

位置：`vllm/v1/engine/output_processor.py:824`

### 4.5 stream_interval

```python
self.stream_interval = stream_interval
```

位置：`vllm/v1/engine/output_processor.py:437`

它控制流式输出频率。

如果 `stream_interval > 1`，`RequestState.make_request_output()` 中会限制不是每个 token 都发：

```python
if self.stream_interval > 1:
    ...
    if not (
        finished
        or self.sent_tokens_offset == 0
        or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
        >= self.stream_interval
    ):
        return None
```

位置：`vllm/v1/engine/output_processor.py:288` 到 `vllm/v1/engine/output_processor.py:301`

含义是：

```text
只有以下情况才发输出：
1. 请求 finished；
2. 这是第一次输出；
3. 距离上次发送已经累计到 stream_interval 个 token。
```

---

## 5. RequestOutputCollector：异步路径的 per-request queue

`RequestOutputCollector` 用于 AsyncLLM。

源码注释：

```python
class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """
```

位置：`vllm/v1/engine/output_processor.py:45` 到 `vllm/v1/engine/output_processor.py:52`

它的作用是：

```text
output_handler 生产 RequestOutput；
generate() / encode() 消费 RequestOutput；
RequestOutputCollector 是两者之间的异步队列。
```

### 5.1 初始化

```python
self.aggregate = output_kind == RequestOutputKind.DELTA
self.request_id = request_id
self.output: RequestOutput | PoolingRequestOutput | Exception | None = None
self.ready = asyncio.Event()
```

位置：`vllm/v1/engine/output_processor.py:54` 到 `vllm/v1/engine/output_processor.py:58`

其中：

```text
aggregate：DELTA 模式下，如果生产速度快于消费速度，可以合并输出。
ready：异步消费者等待输出的 Event。
```

### 5.2 put()

```python
def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None:
```

位置：`vllm/v1/engine/output_processor.py:62`

如果当前没有缓存输出，或者放入的是异常：

```python
self.output = output
self.ready.set()
```

位置：`vllm/v1/engine/output_processor.py:64` 到 `vllm/v1/engine/output_processor.py:66`

如果缓存和新输出都是 `RequestOutput`，会调用：

```python
self.output.add(output, aggregate=self.aggregate)
```

位置：`vllm/v1/engine/output_processor.py:67` 到 `vllm/v1/engine/output_processor.py:72`

其中 `self.aggregate` 只在 `output_kind == DELTA` 时为 `True`：

```text
DELTA：
  拼接同 index completion 的 text / token_ids / logprobs。

非 DELTA：
  用新 completion 替换同 index completion。

PoolingRequestOutput：
  直接用新输出替换缓存输出。
```

这避免了 output_handler 生成太快、generate() 消费太慢时堆积大量小 delta，同时保留非 DELTA 输出的替换语义。

### 5.3 get() / get_nowait()

异步等待：

```python
async def get(self) -> RequestOutput | PoolingRequestOutput:
    while (output := self.output) is None:
        await self.ready.wait()
    ...
```

位置：`vllm/v1/engine/output_processor.py:78` 到 `vllm/v1/engine/output_processor.py:86`

非阻塞获取：

```python
def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:
```

位置：`vllm/v1/engine/output_processor.py:88` 到 `vllm/v1/engine/output_processor.py:96`

`AsyncLLM.generate()` 中就是这样消费：

```python
out = q.get_nowait() or await q.get()
```

位置：`vllm/v1/engine/async_llm.py:576` 到 `vllm/v1/engine/async_llm.py:580`

---

## 6. RequestState：单请求的输出状态

`RequestState` 是 OutputProcessor 处理输出时最重要的请求级状态。

初始化字段包括：

```python
self.request_id = request_id
self.external_req_id = external_req_id
self.parent_req = parent_req
self.request_index = request_index
self.lora_request = lora_request
self.output_kind = output_kind
self.prompt = prompt
self.prompt_token_ids = prompt_token_ids
self.prompt_embeds = prompt_embeds
self.prompt_len = length_from_prompt_token_ids_or_embeds(...)
self.logprobs_processor = logprobs_processor
self.detokenizer = detokenizer
self.max_tokens_param = max_tokens_param
self.queue = queue
self.num_cached_tokens = 0
self.stats = RequestStateStats(...) if log_stats else None
```

位置：`vllm/v1/engine/output_processor.py:129` 到 `vllm/v1/engine/output_processor.py:190`

可以理解为：

```text
RequestState = 输出侧的请求账本。
```

它保存：

```text
外部 request_id；
内部 request_id；
prompt 文本 / token ids / embeds；
是否有 parent request；
输出 index；
detokenizer；
logprobs processor；
输出模式 DELTA / CUMULATIVE / FINAL_ONLY；
stream_interval；
异步 queue；
统计信息；
streaming input 状态。
```

### 6.1 from_new_request()

`RequestState.from_new_request()` 根据 `EngineCoreRequest` 创建状态。

如果是 generation 请求：

```python
if sampling_params := request.sampling_params:
    if not sampling_params.detokenize:
        tokenizer = None
    output_kind = sampling_params.output_kind
    logprobs_processor = LogprobsProcessor.from_new_request(...)
    detokenizer = IncrementalDetokenizer.from_new_request(...)
    max_tokens_param = sampling_params.max_tokens
```

位置：`vllm/v1/engine/output_processor.py:222` 到 `vllm/v1/engine/output_processor.py:238`

如果是 pooling 请求：

```python
else:
    logprobs_processor = None
    detokenizer = None
    max_tokens_param = None
    ...
    assert request.pooling_params is not None
    output_kind = request.pooling_params.output_kind
```

位置：`vllm/v1/engine/output_processor.py:239` 到 `vllm/v1/engine/output_processor.py:246`

所以：

```text
generation 请求：有 detokenizer / logprobs_processor。
pooling 请求：没有 detokenizer / logprobs_processor，直接处理 pooling_output。
```

### 6.2 output_kind

`output_kind` 来自 SamplingParams / PoolingParams。

主要影响：

```text
DELTA：只返回增量 token / 文本；
CUMULATIVE：返回累计输出；
FINAL_ONLY：只在 finished 时返回最终输出。
```

`FINAL_ONLY` 的处理：

```python
if not finished and final_only:
    return None
```

位置：`vllm/v1/engine/output_processor.py:281` 到 `vllm/v1/engine/output_processor.py:286`

DELTA 的处理：

```python
if self.output_kind == RequestOutputKind.DELTA:
    new_token_ids = self.detokenizer.output_token_ids[self.sent_tokens_offset:]
    self.sent_tokens_offset = self.detokenizer.num_output_tokens()
```

位置：`vllm/v1/engine/output_processor.py:303` 到 `vllm/v1/engine/output_processor.py:309`

### 6.3 detokenizer

generation 请求会创建 `IncrementalDetokenizer`：

```python
detokenizer = IncrementalDetokenizer.from_new_request(
    tokenizer=tokenizer,
    request=request,
)
```

位置：`vllm/v1/engine/output_processor.py:230` 到 `vllm/v1/engine/output_processor.py:233`

`process_outputs()` 中会用它更新 token 并做 stop string 检查：

```python
stop_string = req_state.detokenizer.update(
    new_token_ids, finish_reason == FinishReason.STOP
)
```

位置：`vllm/v1/engine/output_processor.py:648` 到 `vllm/v1/engine/output_processor.py:650`

如果返回了 stop string：

```python
if stop_string:
    finish_reason = FinishReason.STOP
    stop_reason = stop_string
```

位置：`vllm/v1/engine/output_processor.py:651` 到 `vllm/v1/engine/output_processor.py:653`

这意味着：

```text
EngineCore / Scheduler 可以因为 token-level stop 结束请求；
OutputProcessor 也可能因为 detokenize 后发现 stop string 而要求结束请求。
```

后者需要外层 Engine 再通知 EngineCore abort。

### 6.4 logprobs_processor

generation 请求也会创建 `LogprobsProcessor`：

```python
logprobs_processor = LogprobsProcessor.from_new_request(
    tokenizer=tokenizer,
    request=request,
)
```

位置：`vllm/v1/engine/output_processor.py:226` 到 `vllm/v1/engine/output_processor.py:229`

每次有输出时更新：

```python
req_state.logprobs_processor.update_from_output(engine_core_output)
```

位置：`vllm/v1/engine/output_processor.py:655` 到 `vllm/v1/engine/output_processor.py:657`

构造 `RequestOutput` 时，会把 prompt logprobs 和 sample logprobs 放进去。

DELTA 模式下 prompt logprobs 会被 pop：

```python
if self.output_kind == RequestOutputKind.DELTA:
    prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
else:
    prompt_logprobs = self.logprobs_processor.prompt_logprobs
```

位置：`vllm/v1/engine/output_processor.py:363` 到 `vllm/v1/engine/output_processor.py:367`

sample logprobs 在 `CompletionOutput` 中：

```python
logprobs = self.logprobs_processor.logprobs
if delta and logprobs:
    logprobs = logprobs[-len(token_ids) :]
```

位置：`vllm/v1/engine/output_processor.py:400` 到 `vllm/v1/engine/output_processor.py:402`

---

## 7. add_request()：登记输出状态

外层 Engine 在把请求送进 EngineCore 前，会先调用 `OutputProcessor.add_request()`。

同步路径：

```python
self.output_processor.add_request(request, prompt_text, None, 0)
self.engine_core.add_request(request)
```

位置：`vllm/v1/engine/llm_engine.py:272` 到 `vllm/v1/engine/llm_engine.py:276`

异步路径：

```python
self.output_processor.add_request(request, prompt, parent_req, index, queue)
await self.engine_core.add_request_async(request)
```

位置：`vllm/v1/engine/async_llm.py:400` 到 `vllm/v1/engine/async_llm.py:412`

### 7.1 为什么要先登记 OutputProcessor

因为 EngineCore 后续返回的是：

```text
EngineCoreOutput(request_id=...)
```

OutputProcessor 必须提前知道：

```text
这个 request_id 对应哪个 prompt；
是否需要 detokenize；
输出模式是什么；
是否属于 parent request；
异步输出要放进哪个 queue；
用户看到的 external request id 是什么。
```

否则收到 EngineCoreOutput 时无法构造用户输出。

### 7.2 add_request() 的逻辑

```python
request_id = request.request_id
req_state = self.request_states.get(request_id)
if req_state is not None:
    self._update_streaming_request_state(req_state, request, prompt)
    return
```

位置：`vllm/v1/engine/output_processor.py:528` 到 `vllm/v1/engine/output_processor.py:532`

如果 request_id 已经存在，说明可能是 streaming input 的追加输入，走 `_update_streaming_request_state()`。

否则创建新的 `RequestState`：

```python
req_state = RequestState.from_new_request(...)
self.request_states[request_id] = req_state
```

位置：`vllm/v1/engine/output_processor.py:534` 到 `vllm/v1/engine/output_processor.py:544`

如果有 parent request，登记 parent：

```python
if parent_req:
    self.parent_requests[parent_req.request_id] = parent_req
```

位置：`vllm/v1/engine/output_processor.py:545` 到 `vllm/v1/engine/output_processor.py:546`

最后登记 external 到 internal 的映射：

```python
self.external_req_ids[req_state.external_req_id].append(request_id)
```

位置：`vllm/v1/engine/output_processor.py:548` 到 `vllm/v1/engine/output_processor.py:549`

### 7.3 add_request() 主链路

```text
OutputProcessor.add_request()
  → 如果 request_id 已存在：更新 streaming request state
  → 否则 RequestState.from_new_request()
      → 创建 detokenizer / logprobs_processor
      → 保存 prompt / params / queue / stats
  → self.request_states[request_id] = req_state
  → 如果 n>1：登记 parent_requests
  → external_req_ids[external_req_id].append(internal_request_id)
```

---

## 8. process_outputs() 总览

`process_outputs()` 是 OutputProcessor 最核心的方法。

入口：

```python
def process_outputs(
    self,
    engine_core_outputs: list[EngineCoreOutput],
    engine_core_timestamp: float | None = None,
    iteration_stats: IterationStats | None = None,
) -> OutputProcessorOutput:
```

位置：`vllm/v1/engine/output_processor.py:584` 到 `vllm/v1/engine/output_processor.py:589`

源码注释说明它做三件事：

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

位置：`vllm/v1/engine/output_processor.py:590` 到 `vllm/v1/engine/output_processor.py:600`

开发者注释也很重要：

```python
vLLM V1 minimizes the number of python loops over the full
batch to ensure system overheads are minimized. This is the
only function that should loop over EngineCoreOutputs.
```

位置：`vllm/v1/engine/output_processor.py:602` 到 `vllm/v1/engine/output_processor.py:606`

也就是说：

```text
process_outputs() 是外层输出处理唯一应该全 batch 遍历 EngineCoreOutputs 的地方。
```

### 8.1 返回值 OutputProcessorOutput

返回类型是：

```python
@dataclass
class OutputProcessorOutput:
    request_outputs: list[RequestOutput | PoolingRequestOutput]
    reqs_to_abort: list[str]
```

位置：`vllm/v1/engine/output_processor.py:109` 到 `vllm/v1/engine/output_processor.py:112`

含义：

```text
request_outputs：
  同步 LLMEngine 需要直接返回给调用方的用户输出。

reqs_to_abort：
  OutputProcessor 在 detokenize 后发现需要停止，但 EngineCore 还没停止的请求。
  外层 Engine 需要把这些 request_id 再发给 EngineCore abort。
```

### 8.2 process_outputs 主流程

源码主循环：

```python
request_outputs: list[RequestOutput | PoolingRequestOutput] = []
reqs_to_abort: list[str] = []
for engine_core_output in engine_core_outputs:
    req_id = engine_core_output.request_id
    req_state = self.request_states.get(req_id)
    ...
```

位置：`vllm/v1/engine/output_processor.py:612` 到 `vllm/v1/engine/output_processor.py:616`

可以概括为：

```text
process_outputs()
  → 初始化 request_outputs / reqs_to_abort
  → 遍历每个 EngineCoreOutput
      → 找 RequestState
      → 更新 stats
      → 取 new_token_ids / pooling_output / finish_reason
      → generation：detokenize + stop string 检查 + logprobs 更新
      → pooling：跳过 detokenize，直接构造 PoolingOutput
      → RequestState.make_request_output()
      → 同步路径：append 到 request_outputs
      → 异步路径：put 到 RequestOutputCollector
      → 如果 finished：清理 RequestState 或处理 streaming input 下一段
      → 如果 OutputProcessor 检测 stop 但 EngineCore 未 finished：加入 reqs_to_abort
  → 返回 OutputProcessorOutput
```

---

## 9. process_outputs 详细步骤

### 9.1 找到对应 RequestState

```python
req_id = engine_core_output.request_id
req_state = self.request_states.get(req_id)
if req_state is None:
    # Ignore output for already-aborted request.
    continue
```

位置：`vllm/v1/engine/output_processor.py:607` 到 `vllm/v1/engine/output_processor.py:611`

如果请求已经被前端 abort 或清理，后续迟到输出会被忽略。

### 9.2 更新本轮统计

```python
self._update_stats_from_output(
    req_state, engine_core_output, engine_core_timestamp, iteration_stats
)
```

位置：`vllm/v1/engine/output_processor.py:621` 到 `vllm/v1/engine/output_processor.py:624`

如果没有开启 stats，内部会直接返回：

```python
if iteration_stats is None:
    return
```

位置：`vllm/v1/engine/output_processor.py:791` 到 `vllm/v1/engine/output_processor.py:792`

### 9.3 取 EngineCoreOutput 字段

```python
new_token_ids = engine_core_output.new_token_ids
pooling_output = engine_core_output.pooling_output
finish_reason = engine_core_output.finish_reason
stop_reason = engine_core_output.stop_reason
kv_transfer_params = engine_core_output.kv_transfer_params
ec_transfer_params = engine_core_output.ec_transfer_params
```

位置：`vllm/v1/engine/output_processor.py:626` 到 `vllm/v1/engine/output_processor.py:631`

如果有 routed experts，也会累积：

```python
if engine_core_output.routed_experts is not None:
    req_state.routed_experts_chunks.append(engine_core_output.routed_experts)
```

位置：`vllm/v1/engine/output_processor.py:632` 到 `vllm/v1/engine/output_processor.py:635`

### 9.4 记录 prefill 阶段 cached tokens

```python
if req_state.is_prefilling:
    if engine_core_output.prefill_stats is not None:
        req_state.num_cached_tokens = (
            engine_core_output.prefill_stats.num_cached_tokens
        )
    req_state.is_prefilling = False
```

位置：`vllm/v1/engine/output_processor.py:637` 到 `vllm/v1/engine/output_processor.py:642`

含义：

```text
第一次从 prefill 阶段收到输出时，记录 prefix cache 命中的 token 数；
后续 RequestOutput / PoolingRequestOutput 会带 num_cached_tokens。
```

### 9.5 generation 输出：detokenize + stop string

如果不是 pooling 输出：

```python
if pooling_output is None:
    assert req_state.detokenizer is not None
    assert req_state.logprobs_processor is not None
```

位置：`vllm/v1/engine/output_processor.py:644` 到 `vllm/v1/engine/output_processor.py:646`

然后 detokenize：

```python
stop_string = req_state.detokenizer.update(
    new_token_ids, finish_reason == FinishReason.STOP
)
```

位置：`vllm/v1/engine/output_processor.py:648` 到 `vllm/v1/engine/output_processor.py:650`

如果 detokenizer 检测到 stop string：

```python
if stop_string:
    finish_reason = FinishReason.STOP
    stop_reason = stop_string
```

位置：`vllm/v1/engine/output_processor.py:651` 到 `vllm/v1/engine/output_processor.py:653`

这一步非常关键：

```text
stop string 是文本级条件，必须 detokenize 后才能可靠判断。
因此它属于 OutputProcessor，而不是 Scheduler 的纯 token 级逻辑。
```

### 9.6 generation 输出：logprobs

```python
req_state.logprobs_processor.update_from_output(engine_core_output)
```

位置：`vllm/v1/engine/output_processor.py:655` 到 `vllm/v1/engine/output_processor.py:657`

这一步把 EngineCoreOutput 中的：

```text
new_logprobs；
new_prompt_logprobs_tensors；
```

转成用户输出里的 logprobs / prompt_logprobs。

### 9.7 构造用户输出

```python
if request_output := req_state.make_request_output(
    new_token_ids,
    pooling_output,
    finish_reason,
    stop_reason,
    kv_transfer_params,
    ec_transfer_params,
):
    ...
```

位置：`vllm/v1/engine/output_processor.py:660` 到 `vllm/v1/engine/output_processor.py:667`

`make_request_output()` 可能返回 `None`，常见原因：

```text
FINAL_ONLY 模式但请求还没 finished；
stream_interval 未达到发送条件；
n>1 parent request 还没准备好对外输出。
```

### 9.8 异步 streaming input 特殊处理

```python
if req_state.streaming_input:
    request_output.finished = False
```

位置：`vllm/v1/engine/output_processor.py:679`

含义：

```text
streaming input 只有在 req_state.streaming_input 当前仍为 True 时，
才会把本次 request_output.finished 强制改为 False。
如果 final request 已经把 req_state.streaming_input 置为 False，
当前 chunk 的完成输出可以保持 finished=True。
```

### 9.9 同步与异步输出分发

如果 `req_state.queue` 不为空，说明是 AsyncLLM：

```python
if req_state.queue is not None:
    # AsyncLLM: put into queue for handling by generate().
    req_state.queue.put(request_output)
```

位置：`vllm/v1/engine/output_processor.py:671` 到 `vllm/v1/engine/output_processor.py:673`

否则是 LLMEngine：

```python
else:
    # LLMEngine: return list of RequestOutputs.
    request_outputs.append(request_output)
```

位置：`vllm/v1/engine/output_processor.py:674` 到 `vllm/v1/engine/output_processor.py:676`

这是同步 / 异步输出处理的核心分叉。

### 9.10 finished 后的清理

如果 `finish_reason is not None`，说明该请求本次输出后结束：

```python
if finish_reason is not None:
```

位置：`vllm/v1/engine/output_processor.py:679`

当 `finish_reason is not None` 时，streaming input 会按当前状态分支处理：

```python
if req_state.streaming_input:
    if req_state.input_chunk_queue:
        update = req_state.input_chunk_queue.popleft()
        req_state.apply_streaming_update(update)
    else:
        req_state.input_chunk_queue = None
```

位置：`vllm/v1/engine/output_processor.py:680` 到 `vllm/v1/engine/output_processor.py:685`

```text
req_state.streaming_input 为 True：
  应用队列中的下一段 StreamingUpdate；没有下一段则将 input_chunk_queue 置为 None，等待后续输入或 final request。

req_state.streaming_input 为 False：
  调用 _finish_request(req_state)，最终输出可以保持 finished=True。
```

非 streaming 或 final request 已结束 streaming 状态时会清理请求：

```python
else:
    self._finish_request(req_state)
```

位置：`vllm/v1/engine/output_processor.py:686` 到 `vllm/v1/engine/output_processor.py:687`

### 9.11 stop string 导致的 reqs_to_abort

如果 OutputProcessor 认为请求 finished，但 EngineCoreOutput 自己没有标记 finished：

```python
if not engine_core_output.finished:
    # If req not finished in EngineCore, but Detokenizer
    # detected stop string, abort needed in EngineCore.
    reqs_to_abort.append(req_id)
```

位置：`vllm/v1/engine/output_processor.py:688` 到 `vllm/v1/engine/output_processor.py:691`

这就是 `reqs_to_abort` 的来源。

含义是：

```text
OutputProcessor 在文本层发现 stop string；
但 EngineCore / Scheduler 还不知道这个文本级 stop；
外层 Engine 需要再调用 engine_core.abort_requests(reqs_to_abort)。
```

同步 `LLMEngine.step()` 中会这样处理：

```python
self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
```

位置：`vllm/v1/engine/llm_engine.py:316` 到 `vllm/v1/engine/llm_engine.py:318`

异步 `AsyncLLM.output_handler` 中会这样处理：

```python
if processed_outputs.reqs_to_abort:
    await engine_core.abort_requests_async(processed_outputs.reqs_to_abort)
```

位置：`vllm/v1/engine/async_llm.py:686` 到 `vllm/v1/engine/async_llm.py:689`

---

## 10. RequestState.make_request_output()

`RequestState.make_request_output()` 负责从请求状态和本次增量构造最终输出对象。

入口：

```python
def make_request_output(
    self,
    new_token_ids: list[int],
    pooling_output: torch.Tensor | None,
    finish_reason: FinishReason | None,
    stop_reason: int | str | None,
    kv_transfer_params: dict[str, Any] | None = None,
    ec_transfer_params: dict[str, Any] | None = None,
) -> RequestOutput | PoolingRequestOutput | None:
```

位置：`vllm/v1/engine/output_processor.py:272` 到 `vllm/v1/engine/output_processor.py:280`

### 10.1 FINAL_ONLY

```python
finished = finish_reason is not None
final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

if not finished and final_only:
    return None
```

位置：`vllm/v1/engine/output_processor.py:281` 到 `vllm/v1/engine/output_processor.py:286`

如果用户只要最终输出，中间增量会被丢弃。

### 10.2 stream_interval

如果设置了 `stream_interval > 1`，未达到输出间隔时返回 `None`。

位置：`vllm/v1/engine/output_processor.py:288` 到 `vllm/v1/engine/output_processor.py:301`

### 10.3 pooling 输出

如果 `pooling_output is not None`：

```python
return self._new_request_output(
    external_req_id,
    [self._new_pooling_output(pooling_output)],
    finished,
)
```

位置：`vllm/v1/engine/output_processor.py:313` 到 `vllm/v1/engine/output_processor.py:318`

`_new_pooling_output()` 很简单：

```python
def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
    return PoolingOutput(data=pooling_output)
```

位置：`vllm/v1/engine/output_processor.py:420` 到 `vllm/v1/engine/output_processor.py:421`

最终会构造 `PoolingRequestOutput`。

### 10.4 generation 输出

如果不是 pooling，先构造 `CompletionOutput`：

```python
output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)
```

位置：`vllm/v1/engine/output_processor.py:320`

然后如果没有 parent request，直接包装：

```python
if self.parent_req is None:
    outputs = [output]
```

位置：`vllm/v1/engine/output_processor.py:322` 到 `vllm/v1/engine/output_processor.py:323`

如果有 parent request，则交给 parent 聚合：

```python
outputs, finished = self.parent_req.get_outputs(self.request_id, output)
```

位置：`vllm/v1/engine/output_processor.py:325` 到 `vllm/v1/engine/output_processor.py:328`

最后构造 `RequestOutput`：

```python
return self._new_request_output(
    external_req_id,
    outputs,
    finished,
    kv_transfer_params,
    ec_transfer_params,
)
```

位置：`vllm/v1/engine/output_processor.py:330` 到 `vllm/v1/engine/output_processor.py:336`

---

## 11. CompletionOutput 如何构造

`_new_completion_output()` 负责构造单个 completion choice。

入口：

```python
def _new_completion_output(
    self,
    token_ids: list[int],
    finish_reason: FinishReason | None,
    stop_reason: int | str | None,
) -> CompletionOutput:
```

位置：`vllm/v1/engine/output_processor.py:383` 到 `vllm/v1/engine/output_processor.py:388`

### 11.1 text

```python
text = self.detokenizer.get_next_output_text(finished, delta)
```

位置：`vllm/v1/engine/output_processor.py:395`

如果不是 DELTA 模式，token_ids 使用累计输出：

```python
if not delta:
    token_ids = self.detokenizer.output_token_ids
```

位置：`vllm/v1/engine/output_processor.py:396` 到 `vllm/v1/engine/output_processor.py:397`

### 11.2 logprobs

```python
logprobs = self.logprobs_processor.logprobs
if delta and logprobs:
    logprobs = logprobs[-len(token_ids) :]
```

位置：`vllm/v1/engine/output_processor.py:400` 到 `vllm/v1/engine/output_processor.py:402`

DELTA 模式只返回本次新增 token 对应的 logprobs。

### 11.3 routed_experts

```python
routed_experts = None
if finished and self.routed_experts_chunks:
    routed_experts = np.concatenate(self.routed_experts_chunks, axis=0)
```

位置：`vllm/v1/engine/output_processor.py:405` 到 `vllm/v1/engine/output_processor.py:407`

`routed_experts` 会在请求结束时聚合。

### 11.4 CompletionOutput 字段

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

位置：`vllm/v1/engine/output_processor.py:409` 到 `vllm/v1/engine/output_processor.py:418`

这里把内部 `FinishReason` 转成用户输出中的字符串：

```text
stop / length / abort / error / repetition
```

---

## 12. RequestOutput / PoolingRequestOutput 如何构造

`_new_request_output()` 根据输出类型构造最终用户可见对象。

入口：

```python
def _new_request_output(
    self,
    external_req_id: str,
    outputs: list[CompletionOutput] | list[PoolingOutput],
    finished: bool,
    kv_transfer_params: dict[str, Any] | None = None,
    ec_transfer_params: dict[str, Any] | None = None,
) -> RequestOutput | PoolingRequestOutput:
```

位置：`vllm/v1/engine/output_processor.py:338` 到 `vllm/v1/engine/output_processor.py:345`

### 12.1 prompt_embeds 占位 token ids

如果使用了 prompt embeds，没有 prompt token ids，会构造占位：

```python
prompt_token_ids = self.prompt_token_ids
if prompt_token_ids is None and self.prompt_embeds is not None:
    prompt_token_ids = [0] * len(self.prompt_embeds)
assert prompt_token_ids is not None
```

位置：`vllm/v1/engine/output_processor.py:347` 到 `vllm/v1/engine/output_processor.py:350`

### 12.2 PoolingRequestOutput

如果第一个 output 是 `PoolingOutput`：

```python
return PoolingRequestOutput(
    request_id=external_req_id,
    outputs=first_output,
    num_cached_tokens=self.num_cached_tokens,
    prompt_token_ids=prompt_token_ids,
    finished=finished,
)
```

位置：`vllm/v1/engine/output_processor.py:355` 到 `vllm/v1/engine/output_processor.py:361`

### 12.3 RequestOutput

generation 请求构造：

```python
return RequestOutput(
    request_id=external_req_id,
    lora_request=self.lora_request,
    prompt=self.prompt,
    prompt_token_ids=prompt_token_ids,
    prompt_logprobs=prompt_logprobs,
    outputs=cast(list[CompletionOutput], outputs),
    finished=finished,
    kv_transfer_params=kv_transfer_params,
    ec_transfer_params=ec_transfer_params,
    num_cached_tokens=self.num_cached_tokens,
    metrics=self.stats,
)
```

位置：`vllm/v1/engine/output_processor.py:369` 到 `vllm/v1/engine/output_processor.py:381`

注意这里使用的是 `external_req_id`：

```python
request_id=external_req_id  # request_id is what was provided externally
```

位置：`vllm/v1/engine/output_processor.py:369` 到 `vllm/v1/engine/output_processor.py:370`

也就是说，用户最终看到的是外部 request id，而不是内部 child request id。

---

## 13. 同步路径：LLMEngine.step()

同步 `LLMEngine.step()` 中，OutputProcessor 的使用方式是：

```python
outputs = self.engine_core.get_output()
...
processed_outputs = self.output_processor.process_outputs(
    outputs.outputs,
    engine_core_timestamp=outputs.timestamp,
    iteration_stats=iteration_stats,
)
self.output_processor.update_scheduler_stats(outputs.scheduler_stats)
...
self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
...
return processed_outputs.request_outputs
```

位置：`vllm/v1/engine/llm_engine.py:302` 到 `vllm/v1/engine/llm_engine.py:334`

同步路径可以概括为：

```text
LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → OutputProcessorOutput.request_outputs
  → engine_core.abort_requests(reqs_to_abort)
  → return request_outputs
```

同步路径的特点：

```text
RequestState.queue is None；
process_outputs() 把 RequestOutput append 到 request_outputs；
LLMEngine.step() 直接 return request_outputs。
```

对应代码：

```python
if req_state.queue is not None:
    req_state.queue.put(request_output)
else:
    request_outputs.append(request_output)
```

位置：`vllm/v1/engine/output_processor.py:661` 到 `vllm/v1/engine/output_processor.py:666`

---

## 14. 异步路径：AsyncLLM output_handler

异步 `AsyncLLM` 中，OutputProcessor 由后台 output_handler 调用。

```python
outputs = await engine_core.get_output_async()
...
processed_outputs = output_processor.process_outputs(
    outputs_slice, outputs.timestamp, iteration_stats
)
# NOTE: RequestOutputs are pushed to their queues.
assert not processed_outputs.request_outputs
```

位置：`vllm/v1/engine/async_llm.py:656` 到 `vllm/v1/engine/async_llm.py:679`

如果有 `reqs_to_abort`：

```python
if processed_outputs.reqs_to_abort:
    await engine_core.abort_requests_async(
        processed_outputs.reqs_to_abort
    )
```

位置：`vllm/v1/engine/async_llm.py:686` 到 `vllm/v1/engine/async_llm.py:689`

异步路径可以概括为：

```text
AsyncLLM.output_handler
  → await engine_core.get_output_async()
  → EngineCoreOutputs
  → 按 chunk 切分 outputs.outputs
  → OutputProcessor.process_outputs()
  → RequestOutputCollector.put(request_output)
  → 如果 processed_outputs.reqs_to_abort 非空，立即 await engine_core.abort_requests_async(...)
  → 更新 scheduler stats / logging

AsyncLLM.generate() / encode()
  → 并发地从 RequestOutputCollector 执行 q.get_nowait() or await q.get()
  → yield 输出
```

异步路径的特点：

```text
RequestState.queue is not None；
process_outputs() 不返回 request_outputs；
而是把 RequestOutput 放进 RequestOutputCollector；
generate() / encode() async generator 再消费 collector。
```

---

## 15. abort_requests()：外层输出状态清理

`OutputProcessor.abort_requests()` 负责清理输出侧状态，并返回需要通知 EngineCore 的内部 request ids。

入口：

```python
def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
```

位置：`vllm/v1/engine/output_processor.py:457`

源码注释说明：

```python
The request_ids may be either external request IDs ... or internal request IDs ...
If an external request ID is provided, and that external request ID
was used for multiple requests, all requests associated with that external
request ID are aborted.
In the case of parallel sampling, a request ID may be used to identify
a parent request, in which case the associated child requests are aborted also.
```

位置：`vllm/v1/engine/output_processor.py:458` 到 `vllm/v1/engine/output_processor.py:470`

### 15.1 internal=True

如果传入的是内部 request id：

```python
if internal:
    internal_req_ids.append(request_id)
    ...
```

位置：`vllm/v1/engine/output_processor.py:474` 到 `vllm/v1/engine/output_processor.py:484`

### 15.2 internal=False

如果传入的是外部 request id：

```python
elif internal_ids := self.external_req_ids.pop(request_id, []):
    internal_req_ids.extend(internal_ids)
```

位置：`vllm/v1/engine/output_processor.py:485` 到 `vllm/v1/engine/output_processor.py:487`

这会把一个外部 request id 映射到多个内部 child request id。

### 15.3 生成 abort final output

对于异步队列，如果请求被 abort，会生成一个最终 abort 输出：

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

位置：`vllm/v1/engine/output_processor.py:496` 到 `vllm/v1/engine/output_processor.py:510`

这里有一个细节：

```python
EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")
```

位置：`vllm/v1/engine/output_processor.py:41` 到 `vllm/v1/engine/output_processor.py:42`

对于 pooling 请求，abort 时需要传一个非 None 的 pooling_output，才能走 pooling 输出分支。

### 15.4 parent request abort

如果 abort 的是 parent request：

```python
elif parent := self.parent_requests.get(request_id):
    if parent.child_requests:
        child_reqs = list(parent.child_requests)
        child_reqs = self.abort_requests(child_reqs, internal=True)
        request_ids_to_abort.extend(child_reqs)
    self.parent_requests.pop(request_id, None)
```

位置：`vllm/v1/engine/output_processor.py:511` 到 `vllm/v1/engine/output_processor.py:517`

所以 parallel sampling 的父请求 abort 会递归 abort 所有 child。

### 15.5 abort 链路

同步路径：

```text
LLMEngine.abort_request()
  → OutputProcessor.abort_requests()
  → EngineCoreClient.abort_requests(internal_ids)
  → EngineCore / Scheduler
```

异步路径：

```text
AsyncLLM.abort()
  → OutputProcessor.abort_requests()
  → EngineCoreClient.abort_requests_async(internal_ids)
  → EngineCore / Scheduler
```

---

## 16. streaming input 输出状态

`OutputProcessor` 支持 streaming input 请求的输出状态更新。

### 16.1 StreamingUpdate

```python
@dataclass
class StreamingUpdate:
    prompt: str | None
    prompt_token_ids: list[int] | None
    arrival_time: float
    final: bool = False
```

位置：`vllm/v1/engine/output_processor.py:115` 到 `vllm/v1/engine/output_processor.py:126`

它代表 streaming input 的一个输入更新。

### 16.2 RequestState 中的 streaming 字段

```python
self.streaming_input = stream_input
self.input_chunk_queue: deque[StreamingUpdate] | None = (
    deque() if stream_input else None
)
```

位置：`vllm/v1/engine/output_processor.py:185` 到 `vllm/v1/engine/output_processor.py:189`

### 16.3 apply_streaming_update()

```python
self.streaming_input = not update.final
...
if update.prompt:
    self.prompt = ((self.prompt + update.prompt) if self.prompt else update.prompt)
...
self.prompt_len = len(self.prompt_token_ids)
...
self.is_prefilling = True
```

位置：`vllm/v1/engine/output_processor.py:191` 到 `vllm/v1/engine/output_processor.py:208`

这表示每个新 input chunk 到来时，OutputProcessor 需要更新 prompt、prompt_token_ids、arrival_time，并把状态重新视为 prefill。

### 16.4 _update_streaming_request_state()

当 `add_request()` 发现 request_id 已存在，会调用：

```python
self._update_streaming_request_state(req_state, request, prompt)
```

位置：`vllm/v1/engine/output_processor.py:521` 到 `vllm/v1/engine/output_processor.py:524`

如果不是 resumable，说明是 final request：

```python
if not request.resumable:
    # Final request - just mark completion, don't add its dummy tokens.
```

位置：`vllm/v1/engine/output_processor.py:555` 到 `vllm/v1/engine/output_processor.py:556`

如果 Engine 已经结束，会发一个 `STREAM_FINISHED` 解除 generate loop：

```python
req_state.queue.put(STREAM_FINISHED)
```

位置：`vllm/v1/engine/output_processor.py:557` 到 `vllm/v1/engine/output_processor.py:563`

如果是新的 streaming update，则排队：

```python
update = StreamingUpdate(...)
...
req_state.input_chunk_queue.append(update)
```

位置：`vllm/v1/engine/output_processor.py:570` 到 `vllm/v1/engine/output_processor.py:582`

### 16.5 process_outputs 中处理 streaming input finished

当某个 chunk finished：

```python
if req_state.streaming_input:
    if req_state.input_chunk_queue:
        update = req_state.input_chunk_queue.popleft()
        req_state.apply_streaming_update(update)
    else:
        req_state.input_chunk_queue = None
```

位置：`vllm/v1/engine/output_processor.py:680` 到 `vllm/v1/engine/output_processor.py:685`

含义：

```text
一个 chunk 的 EngineCore 请求 finished 后，
如果还有后续 input chunk，就把下一段输入应用到同一个 RequestState，
继续等待后续 EngineCoreOutput。
```

---

## 17. stats 和 tracing

`OutputProcessor` 还负责一部分外层统计和 tracing。

### 17.1 _update_stats_from_output()

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

位置：`vllm/v1/engine/output_processor.py:784` 到 `vllm/v1/engine/output_processor.py:803`

它用 EngineCoreOutput 和时间戳更新本轮 stats。

### 17.2 _update_stats_from_finished()

请求结束时：

```python
iteration_stats.update_from_finished_request(
    finish_reason=finish_reason,
    request_id=req_state.external_req_id,
    num_prompt_tokens=req_state.prompt_len,
    max_tokens_param=req_state.max_tokens_param,
    req_stats=req_state.stats,
    num_cached_tokens=req_state.num_cached_tokens,
)
```

位置：`vllm/v1/engine/output_processor.py:816` 到 `vllm/v1/engine/output_processor.py:823`

并更新 LoRA 和 parent request 统计：

```python
self.lora_states.request_finished(req_state.request_id, req_state.lora_name)
ParentRequest.observe_finished_request(...)
```

位置：`vllm/v1/engine/output_processor.py:824` 到 `vllm/v1/engine/output_processor.py:828`

### 17.3 do_tracing()

如果 tracing 开启，请求结束时会记录 span：

```python
if self.tracing_enabled:
    self.do_tracing(engine_core_output, req_state, iteration_stats)
```

位置：`vllm/v1/engine/output_processor.py:697` 到 `vllm/v1/engine/output_processor.py:698`

`do_tracing()` 中会记录：

```text
TTFT；
E2E latency；
time in queue；
prompt tokens；
completion tokens；
prefill / decode / inference time；
request id；
top_p / max_tokens / temperature / n。
```

对应字段位置：`vllm/v1/engine/output_processor.py:731` 到 `vllm/v1/engine/output_processor.py:782`

---

## 18. 请求完成清理 _finish_request()

请求完成后，OutputProcessor 会清理外层状态。

```python
def _finish_request(self, req_state: RequestState) -> None:
    req_id = req_state.request_id
    self.request_states.pop(req_id)
```

位置：`vllm/v1/engine/output_processor.py:695` 到 `vllm/v1/engine/output_processor.py:697`

然后清理 external id 映射：

```python
internal_ids = self.external_req_ids[req_state.external_req_id]
internal_ids.remove(req_id)
if not internal_ids:
    del self.external_req_ids[req_state.external_req_id]
```

位置：`vllm/v1/engine/output_processor.py:699` 到 `vllm/v1/engine/output_processor.py:702`

如果有 parent request 且 child 都结束，也清理 parent：

```python
parent_req = req_state.parent_req
if parent_req and not parent_req.child_requests:
    self.parent_requests.pop(parent_req.request_id, None)
```

位置：`vllm/v1/engine/output_processor.py:704` 到 `vllm/v1/engine/output_processor.py:707`

这说明外层输出状态和 EngineCore 内部请求状态是分开的：

```text
EngineCore / Scheduler 负责释放内部调度资源；
OutputProcessor 负责释放外层输出状态。
```

---

## 19. 容易混淆的点

### 19.1 OutputProcessor 是不是 EngineCore 的一部分？

不是。

`OutputProcessor` 在 `LLMEngine` / `AsyncLLM` 中创建，属于外层 Engine。

```text
EngineCore：
  输出 EngineCoreOutputs。

OutputProcessor：
  把 EngineCoreOutputs 转成用户输出。
```

### 19.2 EngineCoreOutputs 是不是 RequestOutput？

不是。

```text
EngineCoreOutputs：
  内部输出协议。

RequestOutput / PoolingRequestOutput：
  用户可见输出协议。
```

中间必须经过：

```text
OutputProcessor.process_outputs()
```

### 19.3 detokenize 在哪里做？

在 `OutputProcessor` 侧，通过 `IncrementalDetokenizer`。

```text
EngineCoreOutput.new_token_ids
  → IncrementalDetokenizer.update()
  → text / stop string
  → CompletionOutput.text
```

### 19.4 stop string 在哪里处理？

文本级 stop string 在 `OutputProcessor` 处理。

如果 detokenizer 发现 stop string，但 EngineCore 还没 finished：

```text
OutputProcessor 返回 reqs_to_abort；
LLMEngine / AsyncLLM 再通知 EngineCore abort。
```

### 19.5 同步和异步 OutputProcessor 有两个实现吗？

没有。

同步和异步共用同一个 `OutputProcessor`。

区别只在 `RequestState.queue`：

```text
queue is None：
  LLMEngine，同步返回 request_outputs。

queue is not None：
  AsyncLLM，推入 RequestOutputCollector。
```

### 19.6 OutputProcessor.abort_requests() 是否等于 EngineCore abort？

不是。

`OutputProcessor.abort_requests()` 只处理外层输出状态，并返回内部 request ids。

真正取消 EngineCore / Scheduler 里的请求，还需要：

```text
EngineCoreClient.abort_requests(...)
```

### 19.7 Pooling 输出需要 detokenize 吗？

不需要。

pooling 请求没有 detokenizer / logprobs_processor，直接把 `pooling_output` 包成 `PoolingOutput` 和 `PoolingRequestOutput`。

---

## 20. 最关键的关系图

### 20.1 总体输出链路

```text
Worker / ModelRunner
  → ModelRunnerOutput
  → Scheduler.update_from_output(scheduler_output, model_output)
  → dict[int, EngineCoreOutputs]
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

### 20.2 OutputProcessor 内部处理

```text
OutputProcessor.process_outputs(engine_core_outputs)
  → for each EngineCoreOutput
      → request_states[request_id]
      → update stats
      → generation:
          → detokenizer.update(new_token_ids)
          → stop string check
          → logprobs_processor.update_from_output()
      → pooling:
          → use pooling_output directly
      → RequestState.make_request_output()
          → CompletionOutput / PoolingOutput
          → RequestOutput / PoolingRequestOutput
      → queue.put(output) 或 request_outputs.append(output)
      → finished 后 _finish_request()
      → 必要时 reqs_to_abort.append(request_id)
  → OutputProcessorOutput(request_outputs, reqs_to_abort)
```

### 20.3 同步 LLMEngine 路径

```text
LLMEngine.step()
  → engine_core.get_output()
  → EngineCoreOutputs
  → output_processor.process_outputs()
  → request_outputs
  → engine_core.abort_requests(reqs_to_abort)
  → return request_outputs
```

### 20.4 异步 AsyncLLM 路径

```text
AsyncLLM.output_handler
  → await engine_core.get_output_async()
  → EngineCoreOutputs
  → output_processor.process_outputs()
  → RequestOutputCollector.put(output)
  → 如果有 reqs_to_abort，立即 await engine_core.abort_requests_async(...)
  → update scheduler stats / logging

AsyncLLM.generate()
  → 并发地 q.get_nowait() or await q.get()
  → yield RequestOutput
```

### 20.5 请求状态关系

```text
OutputProcessor
  → request_states[internal_req_id] = RequestState
  → external_req_ids[external_req_id] = [internal_req_id, ...]
  → parent_requests[parent_req_id] = ParentRequest

RequestState
  → detokenizer
  → logprobs_processor
  → queue（AsyncLLM）
  → prompt / prompt_token_ids / stats
  → output_kind / stream_interval
```

---

## 21. 从“回答问题”的角度总结

如果问：

```text
OutputProcessor 如何把 EngineCoreOutputs 转成用户输出？
```

可以回答：

```text
OutputProcessor 是 LLMEngine / AsyncLLM 这一层的输出处理组件。
EngineCore 返回的 EngineCoreOutputs 只是内部输出协议，里面主要是每个请求的新 token ids、logprobs、pooling output、finish reason 等。
OutputProcessor 会根据 request_id 找到对应的 RequestState，对 generation 请求执行增量 detokenize、stop string 检查和 logprobs 组装；对 pooling 请求则直接包装 pooling_output。
随后它通过 RequestState.make_request_output() 构造 CompletionOutput / PoolingOutput，并进一步构造 RequestOutput / PoolingRequestOutput。
```

同步和异步的差异是：

```text
在 LLMEngine 中，RequestState 没有 queue，process_outputs() 会把 RequestOutput 放到返回列表里，LLMEngine.step() 直接返回这些输出。

在 AsyncLLM 中，RequestState 有 RequestOutputCollector queue，process_outputs() 会把 RequestOutput 推入 queue，generate() / encode() 再从 queue 中异步取出并 yield 给调用方。
```

它的核心边界可以概括为：

```text
EngineCore：负责生成内部 EngineCoreOutputs。
OutputProcessor：负责把内部输出协议转换成用户可见输出协议。
```

最小心智模型：

```text
OutputProcessor = request_id 对账 + detokenize + stop string + logprobs + output aggregation + sync/async 分发。
```
