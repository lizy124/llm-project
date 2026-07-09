# vLLM V1 EngineCore 逻辑梳理

源码位置：`vllm/vllm/v1/engine/core.py`

本文用于总览 `EngineCore` 的职责、主执行循环、Scheduler / Worker 的协作关系、同步 / 异步 / 多进程模式，以及 `EngineCoreOutputs` 如何返回给上层。

---

## 1. EngineCore 是什么

`EngineCore` 是 vLLM V1 Engine 内部的执行核心。

源码里的定位非常直接：

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
```

位置：`vllm/vllm/v1/engine/core.py:96`

一句话：

```text
EngineCore 是 vLLM V1 的内部执行闭环总控。
```

它主要负责把下面几件事串成一轮一轮的执行：

```text
接收外层已经处理好的 EngineCoreRequest；
转换成 Scheduler 使用的内部 Request；
把 Request 交给 Scheduler；
调用 Scheduler.schedule() 生成 SchedulerOutput；
把 SchedulerOutput 交给 model_executor / Worker / ModelRunner 执行；
拿到 ModelRunnerOutput；
调用 Scheduler.update_from_output() 更新请求状态；
得到 EngineCoreOutputs；
返回给外层 LLMEngine / AsyncLLM。
```

它不负责：

```text
不直接处理用户原始 prompt；
不直接 detokenize token；
不直接实现 token 级调度；
不直接执行模型 forward；
不直接构造最终 RequestOutput。
```

这些职责分别属于：

```text
InputProcessor：
  把用户输入转成 EngineCoreRequest。

Scheduler：
  管理请求状态、token budget、KV block、抢占、stop、资源释放。

model_executor / Worker / ModelRunner：
  准备模型输入、执行 forward / sample / pooling。

OutputProcessor：
  把 EngineCoreOutputs 转成 RequestOutput / PoolingRequestOutput。
```

所以 EngineCore 的核心边界是：

```text
它不是 Scheduler；
它不是 Worker；
它是 Scheduler 和 Worker 之间的一轮执行闭环编排者。
```

---

## 2. EngineCore 主链路

EngineCore 的主链路可以概括为：

```text
Engine / API
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
  → EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output)
  → Worker / ModelRunner forward
  → ModelRunnerOutput
  → Scheduler.update_from_output(scheduler_output, model_output)
  → EngineCoreOutputs
  → EngineCoreClient.get_output()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

最核心的是 `EngineCore.step()`。

源码注释直接说明：

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output.

    Returns tuple of outputs and a flag indicating whether the model
    was executed.
    """
```

位置：`vllm/vllm/v1/engine/core.py:479` 到 `vllm/vllm/v1/engine/core.py:484`

普通 `step()` 的代码主线是：

```python
if not self.scheduler.has_requests():
    return {}, False
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
...
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
...
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)

return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

位置：`vllm/vllm/v1/engine/core.py:486` 到 `vllm/vllm/v1/engine/core.py:508`

这一段可以拆成三步：

```text
Schedule：
  Scheduler.schedule() 生成本轮执行计划 SchedulerOutput。

Execute：
  model_executor.execute_model() 把计划交给 Worker / ModelRunner 执行。

Make output：
  Scheduler.update_from_output() 用 Worker 结果更新请求状态并生成 EngineCoreOutputs。
```

所以：

```text
SchedulerOutput 是计划；
ModelRunnerOutput 是结果；
EngineCoreOutputs 是回给外层 Engine 的内部输出。
```

---

## 3. 请求如何进入 EngineCore

外层 `LLMEngine` / `AsyncLLM` 不直接把用户输入塞给 Scheduler。

请求入口先经过 `InputProcessor`：

```text
用户输入 prompt / params
  → InputProcessor.process_inputs()
  → EngineCoreRequest
```

同步 `LLMEngine.add_request()` 中，主线是：

```python
request = self.input_processor.process_inputs(...)
...
self.output_processor.add_request(request, prompt_text, None, 0)
self.engine_core.add_request(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:250`、`vllm/vllm/v1/engine/llm_engine.py:274` 到 `vllm/vllm/v1/engine/llm_engine.py:276`

EngineCore 内部再把 `EngineCoreRequest` 转成 Scheduler 使用的 `Request`：

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
```

位置：`vllm/vllm/v1/engine/core.py:867`

如果请求使用结构化输出，还会初始化 grammar：

```python
if req.use_structured_output:
    self.structured_output_manager.grammar_init(req)
```

位置：`vllm/vllm/v1/engine/core.py:868` 到 `vllm/vllm/v1/engine/core.py:874`

最后进入 Scheduler：

```python
self.scheduler.add_request(request)
```

位置：`vllm/vllm/v1/engine/core.py:403`

所以请求对象分两层：

```text
EngineCoreRequest：
  外层 Engine 交给 EngineCore 的请求对象。

Request：
  EngineCore 转换后交给 Scheduler 的内部请求对象，带有更多调度状态。
```

边界是：

```text
外层 Engine / AsyncLLM：EngineCoreRequest
EngineCore 内部：Request
Scheduler：Request
```

---

## 4. EngineCore 和 Scheduler 的关系

`Scheduler` 是 EngineCore 内部的调度和状态管理组件。

EngineCore 初始化时会创建 Scheduler：

```python
self.scheduler: SchedulerInterface = Scheduler(
    vllm_config=vllm_config,
    kv_cache_config=kv_cache_config,
    structured_output_manager=self.structured_output_manager,
    include_finished_set=include_finished_set,
    log_stats=self.log_stats,
    block_size=scheduler_block_size,
    hash_block_size=hash_block_size,
)
```

位置：`vllm/vllm/v1/engine/core.py:150` 到 `vllm/vllm/v1/engine/core.py:158`

初始化顺序是：

```text
EngineCore
  → 创建 model_executor
  → 初始化 / profile KV cache
  → 得到 kv_cache_config
  → 创建 structured_output_manager
  → 创建 Scheduler
```

Scheduler 需要 `kv_cache_config`，所以必须在 KV cache 初始化之后创建。

在一轮执行中，Scheduler 做两件核心事：

```text
1. schedule()：
   根据 waiting / running 请求、KV block、token budget、spec decode、encoder input 等状态，
   生成本轮执行计划 SchedulerOutput。

2. update_from_output()：
   用 Worker 返回的 ModelRunnerOutput 更新请求状态，
   处理 stop、spec decode 接受/拒绝、logprobs、pooling、KV transfer、资源释放，
   最终生成 EngineCoreOutputs。
```

对应 EngineCore 调用是：

```python
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
```

位置：`vllm/vllm/v1/engine/core.py:490`

以及：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:504` 到 `vllm/vllm/v1/engine/core.py:506`

可以这样理解：

```text
EngineCore 负责调用 Scheduler；
Scheduler 负责决定怎么调度，以及如何根据结果更新状态。
```

EngineCore 不直接判断：

```text
每个请求调度多少 token；
KV block 是否够；
是否需要抢占；
prefix cache 命中多少；
spec token 接受 / 拒绝后如何回退；
请求是否 finished；
何时释放 block。
```

这些都属于 Scheduler。

---

## 5. SchedulerOutput 在 EngineCore 中如何流动

`SchedulerOutput` 是 Scheduler 生成的一轮执行计划。

它定义在：

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]
    finished_req_ids: set[str]
    free_encoder_mm_hashes: list[str]
    preempted_req_ids: set[str] | None = None
    has_structured_output_requests: bool = False
    pending_structured_output_tokens: bool = False
    num_invalid_spec_tokens: dict[str, int] | None = None
    kv_connector_metadata: KVConnectorMetadata | None = None
    ec_connector_metadata: ECConnectorMetadata | None = None
    new_block_ids_to_zero: list[int] | None = None
    num_spec_tokens_to_schedule: int = 0
```

位置：`vllm/vllm/v1/core/sched/output.py:180` 到 `vllm/vllm/v1/core/sched/output.py:245`

它的作用可以分成两层：

```text
向下：
  告诉 Worker / ModelRunner 本轮要执行哪些请求、多少 token、哪些 block、哪些 encoder input。

向上：
  在 Worker 返回后，作为本轮计划账本交给 Scheduler.update_from_output() 对账。
```

EngineCore 拿到它后会做三件事：

```text
1. execute_model(scheduler_output)：
   发给 model_executor / Worker 执行。

2. get_grammar_bitmask(scheduler_output)：
   为结构化输出计算 grammar bitmask。

3. update_from_output(scheduler_output, model_output)：
   和 Worker 真实结果一起交给 Scheduler 回收。
```

所以 `SchedulerOutput` 不是一次性发给 Worker 就结束。

它还是后续 `update_from_output()` 的对账凭证。

---

## 6. EngineCore 和 ModelExecutor / Worker / ModelRunner 的关系

EngineCore 不直接执行模型 forward。

它只调用：

```python
future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

位置：`vllm/vllm/v1/engine/core.py:491`

`model_executor` 是 Executor 抽象，负责把调用分发到 Worker。

Executor 抽象里说明：

```python
class Executor(ABC):
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
```

位置：`vllm/vllm/v1/executor/abstract.py:37` 到 `vllm/vllm/v1/executor/abstract.py:42`

Executor 的 `execute_model()` 本质上是调用 Worker 的 `execute_model`：

```python
output = self.collective_rpc(
    "execute_model", args=(scheduler_output,), non_block=non_block
)
return output[0]
```

位置：`vllm/vllm/v1/executor/abstract.py:221` 到 `vllm/vllm/v1/executor/abstract.py:227`

GPU Worker 再调用 ModelRunner：

```python
output = self.model_runner.execute_model(
    scheduler_output, intermediate_tensors
)
```

位置：`vllm/vllm/v1/worker/gpu_worker.py:867` 到 `vllm/vllm/v1/worker/gpu_worker.py:870`

GPU ModelRunner 里才是真正的模型输入准备和 forward：

```text
_update_states(scheduler_output)
_prepare_inputs(...)
_build_attention_metadata(...)
_preprocess(...)
_model_forward(...)
compute_logits / pooling
sample_tokens(...)
ModelRunnerOutput
```

真正 forward 发生在：

```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4320` 到 `vllm/vllm/v1/worker/gpu_model_runner.py:4326`

对于 generation 模型，`execute_model()` 可能只完成 forward / logits，然后返回 `None`。

EngineCore 看到 `None` 后会调用：

```python
model_output = self.model_executor.sample_tokens(grammar_output)
```

位置：`vllm/vllm/v1/engine/core.py:498` 到 `vllm/vllm/v1/engine/core.py:499`

这一步会应用 grammar bitmask、执行采样，并构造 `ModelRunnerOutput`。

所以执行链路是：

```text
EngineCore
  → model_executor
  → Worker
  → ModelRunner
  → model forward / sample
  → ModelRunnerOutput
```

---

## 7. ModelRunnerOutput 如何回到 Scheduler

`ModelRunnerOutput` 是 Worker / ModelRunner 返回的 batch 级执行结果。

它包含：

```python
@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    logprobs: LogprobsLists | None = None
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(default_factory=dict)
    pooler_output: list[torch.Tensor | None] | None = None
    kv_connector_output: KVConnectorOutput | None = None
    ec_connector_output: ECConnectorOutput | None = None
    num_nans_in_logits: dict[str, int] | None = None
    cudagraph_stats: CUDAGraphStat | None = None
    routed_experts: RoutedExpertsLists | None = None
```

位置：`vllm/vllm/v1/outputs.py:233` 到 `vllm/vllm/v1/outputs.py:281`

EngineCore 不自己解释它，而是调用：

```python
self.scheduler.update_from_output(scheduler_output, model_output)
```

位置：`vllm/vllm/v1/engine/core.py:504` 到 `vllm/vllm/v1/engine/core.py:506`

为什么要同时传 `SchedulerOutput` 和 `ModelRunnerOutput`？

```text
SchedulerOutput：
  本轮计划账本，说明哪些请求被调度、每个请求调度多少 token、哪些 spec tokens / encoder inputs 被处理。

ModelRunnerOutput：
  真实执行结果，说明实际生成了哪些 token、logprobs / pooling output / connector output 是什么。
```

`update_from_output()` 会把两者对齐：

```text
用 SchedulerOutput.num_scheduled_tokens 遍历本轮请求；
用 ModelRunnerOutput.req_id_to_index 找到请求对应输出；
append sampled token；
检查 stop；
处理 spec decode 接受 / 拒绝；
处理 logprobs / prompt logprobs；
处理 pooling output；
处理 routed experts；
处理 KV connector output / KV stats / KV events；
释放 finished request 资源；
构造 EngineCoreOutput；
按 client_index 组装 EngineCoreOutputs。
```

所以：

```text
schedule() 是发任务；
execute_model() 是执行任务；
update_from_output() 是收结果并对账。
```

---

## 8. EngineCoreOutputs 如何返回上层

`Scheduler.update_from_output()` 返回：

```text
dict[int, EngineCoreOutputs]
```

其中 key 是 `client_index`。

`EngineCoreOutput` 是单个请求的一次增量输出：

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

位置：`vllm/vllm/v1/engine/__init__.py:175` 到 `vllm/vllm/v1/engine/__init__.py:201`

`EngineCoreOutputs` 是一组输出和统计信息：

```python
class EngineCoreOutputs(...):
    engine_index: int = 0
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0
    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None
    wave_complete: int | None = None
    start_wave: int | None = None
```

位置：`vllm/vllm/v1/engine/__init__.py:220` 到 `vllm/vllm/v1/engine/__init__.py:244`

同步 `LLMEngine.step()` 消费方式是：

```python
outputs = self.engine_core.get_output()
processed_outputs = self.output_processor.process_outputs(
    outputs.outputs,
    engine_core_timestamp=outputs.timestamp,
    iteration_stats=iteration_stats,
)
self.output_processor.update_scheduler_stats(outputs.scheduler_stats)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:302` 到 `vllm/vllm/v1/engine/llm_engine.py:315`

异步 `AsyncLLM` 则在后台 `output_handler()` 中消费：

```python
outputs = await engine_core.get_output_async()
...
processed_outputs = output_processor.process_outputs(
    outputs_slice, outputs.timestamp, iteration_stats
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:656` 到 `vllm/vllm/v1/engine/async_llm.py:677`

所以输出层级是：

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

注意：

```text
EngineCoreOutputs 不是最终用户输出；
它是 EngineCore 返回给外层 Engine 的内部输出协议。
```

---

## 9. 同步、异步和多进程模式

外层 Engine 通常不是直接持有 `EngineCore`，而是持有 `EngineCoreClient`。

`EngineCoreClient` 的注释说明：

```python
EngineCoreClient: subclasses handle different methods for pushing
    and pulling from the EngineCore for asyncio / multiprocessing.

Subclasses:
* InprocClient: In process EngineCore (for V0-style LLMEngine use)
* SyncMPClient: ZMQ + background proc EngineCore (for LLM)
* AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
```

位置：`vllm/vllm/v1/engine/core_client.py:71` 到 `vllm/vllm/v1/engine/core_client.py:80`

### 9.1 InprocClient

in-process 模式下，Client 直接创建 EngineCore：

```python
self.engine_core = EngineCore(*args, **kwargs)
```

位置：`vllm/vllm/v1/engine/core_client.py:286` 到 `vllm/vllm/v1/engine/core_client.py:287`

拉输出时直接 step：

```python
outputs, model_executed = self.engine_core.step_fn()
self.engine_core.post_step(model_executed=model_executed)
return outputs and outputs.get(0) or EngineCoreOutputs()
```

位置：`vllm/vllm/v1/engine/core_client.py:289` 到 `vllm/vllm/v1/engine/core_client.py:292`

### 9.2 EngineCoreProc

多进程模式下，后台运行的是：

```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
```

位置：`vllm/vllm/v1/engine/core.py:894` 到 `vllm/vllm/v1/engine/core.py:895`

它有自己的 input / output queue：

```python
self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
```

位置：`vllm/vllm/v1/engine/core.py:913` 到 `vllm/vllm/v1/engine/core.py:914`

后台 busy loop 是：

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        self._process_input_queue()
        self._process_engine_step()

    raise SystemExit
```

位置：`vllm/vllm/v1/engine/core.py:1257` 到 `vllm/vllm/v1/engine/core.py:1265`

`_process_engine_step()` 会把输出放进 output queue：

```python
outputs, model_executed = self.step_fn()
for output in outputs.items() if outputs else ():
    self.output_queue.put_nowait(output)
self.post_step(model_executed)
```

位置：`vllm/vllm/v1/engine/core.py:1301` 到 `vllm/vllm/v1/engine/core.py:1307`

输出线程再按 `client_index` 发给对应 client socket：

```python
client_index, outputs = output
outputs.engine_index = engine_index
...
tracker = sockets[client_index].send_multipart(
    buffers, copy=False, track=True
)
```

位置：`vllm/vllm/v1/engine/core.py:1628` 到 `vllm/vllm/v1/engine/core.py:1645`

所以多进程路径是：

```text
Frontend EngineCoreClient
  → ZMQ input socket
  → EngineCoreProc input thread
  → input_queue
  → run_busy_loop()
  → EngineCore.step_fn()
  → output_queue
  → output thread
  → ZMQ output socket
  → Frontend EngineCoreClient
```

---

## 10. batch queue 和异步调度

EngineCore 支持 batch queue，用于 pipeline parallel 等场景。

初始化时：

```python
self.batch_queue_size = vllm_config.max_concurrent_batches
self.batch_queue = None
if self.batch_queue_size > 1:
    self.batch_queue = deque(maxlen=self.batch_queue_size)
```

位置：`vllm/vllm/v1/engine/core.py:192` 到 `vllm/vllm/v1/engine/core.py:203`

然后选择：

```python
self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)
self.async_scheduling = vllm_config.scheduler_config.async_scheduling
```

位置：`vllm/vllm/v1/engine/core.py:221` 到 `vllm/vllm/v1/engine/core.py:224`

普通 `step()` 是同步闭环：

```text
schedule → execute → result → update_from_output
```

`step_with_batch_queue()` 则允许：

```text
先 schedule 新 batch 入队；
不一定立刻等 Worker 输出；
等队列里最早的 future 完成后再 update_from_output。
```

这让调度和模型执行可以更好地 overlap，尤其用于 pipeline parallel 减少 pipeline bubble。

如果本轮存在 `pending_structured_output_tokens`，当前源码会把 sampling 延后，等前一个 batch 的输出处理完成后，再计算 grammar bitmask 并调用 `sample_tokens()`：

```python
if not scheduler_output.pending_structured_output_tokens:
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    future = self.model_executor.sample_tokens(grammar_output, non_block=True)
else:
    deferred_scheduler_output = scheduler_output
```

位置：`vllm/vllm/v1/engine/core.py:559` 到 `vllm/vllm/v1/engine/core.py:572`

核心区别：

```text
普通 step：SchedulerOutput 和 ModelRunnerOutput 通常在同一次 step 中配对。

batch queue：SchedulerOutput 会和 future 一起暂存在 batch_queue，
稍后 Worker 返回时再和对应 ModelRunnerOutput 配对。
```

---

## 11. abort、utility、sleep、shutdown

EngineCore 除了主执行链路，还负责控制类能力。

### 11.1 abort

EngineCore 有 abort queue：

```python
self.aborts_queue = queue.Queue[list[str]]()
```

位置：`vllm/vllm/v1/engine/core.py:226`

普通 step 在处理 Worker 输出前先处理 abort：

```python
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:501` 到 `vllm/vllm/v1/engine/core.py:506`

这样可以保证执行期间到达的取消请求，在 Scheduler 回收输出前生效。

### 11.2 utility

后台进程中，client request 会按类型分发：

```text
ADD：添加请求；
ABORT：取消请求；
UTILITY：profile / reset / sleep / LoRA 等工具调用；
EXECUTOR_FAILED：executor 异常；
WAKEUP：唤醒 loop。
```

对应位置：`vllm/vllm/v1/engine/core.py:1375` 到 `vllm/vllm/v1/engine/core.py:1399`

utility 输出通过 `EngineCoreOutputs(utility_output=...)` 返回，而不是走请求输出列表。

### 11.3 profile / reset / sleep / wakeup

`profile()` 直接转发到 executor：

```python
self.model_executor.profile(is_start, profile_prefix)
```

位置：`vllm/vllm/v1/engine/core.py:662` 到 `vllm/vllm/v1/engine/core.py:663`

`reset_mm_cache()` 清 EngineCore 侧 MM cache，并通知 Worker：

```python
if self.mm_receiver_cache is not None:
    self.mm_receiver_cache.clear_cache()

self.model_executor.reset_mm_cache()
```

位置：`vllm/vllm/v1/engine/core.py:674` 到 `vllm/vllm/v1/engine/core.py:678`

`reset_prefix_cache()` 交给 Scheduler：

```python
return self.scheduler.reset_prefix_cache(
    reset_running_requests, reset_connector
)
```

位置：`vllm/vllm/v1/engine/core.py:680` 到 `vllm/vllm/v1/engine/core.py:685`

`reset_encoder_cache()` 同时清 Scheduler 逻辑状态和 Worker 物理缓存：

```python
self.scheduler.reset_encoder_cache()
self.model_executor.reset_encoder_cache()
```

位置：`vllm/vllm/v1/engine/core.py:702` 到 `vllm/vllm/v1/engine/core.py:705`

`sleep()` 先暂停 Scheduler，再根据 level 让 executor 管理 GPU 内存：

```python
pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
...
model_executor.sleep(level)
```

位置：`vllm/vllm/v1/engine/core.py:774` 到 `vllm/vllm/v1/engine/core.py:784`

`wake_up()` 会先剔除只用于恢复调度的 `"scheduling"` tag，再按需唤醒 executor，最后恢复 Scheduler：

```python
if tags is not None and "scheduling" in tags:
    tags = [t for t in tags if t != "scheduling"]

if tags is None or tags:
    self.model_executor.wake_up(tags)

self.resume_scheduler()
```

位置：`vllm/vllm/v1/engine/core.py:799` 到 `vllm/vllm/v1/engine/core.py:813`

### 11.4 shutdown

普通 EngineCore shutdown 会释放：

```text
structured output backend；
model_executor / Worker；
Scheduler；
GC freeze 状态；
distributed environment / cached memory。
```

对应代码：

```python
self.structured_output_manager.clear_backend()
if self.model_executor:
    self.model_executor.shutdown()
if self.scheduler:
    self.scheduler.shutdown()
gc.unfreeze()
cleanup_dist_env_and_memory()
```

位置：`vllm/vllm/v1/engine/core.py:644` 到 `vllm/vllm/v1/engine/core.py:660`

后台进程模式还有 shutdown state：

```python
class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2
```

位置：`vllm/vllm/v1/engine/core.py:888` 到 `vllm/vllm/v1/engine/core.py:891`

`_handle_shutdown()` 会根据 `shutdown_timeout` 决定 abort 还是 drain。

---

## 12. 容易混淆的几个边界

### 12.1 EngineCore 是不是 Scheduler

不是。

```text
EngineCore：
  调用 Scheduler，编排执行闭环。

Scheduler：
  管理请求状态、调度策略、KV block、stop、资源释放。
```

### 12.2 EngineCore 是不是 Worker

不是。

```text
EngineCore：
  调用 model_executor.execute_model()。

Worker / ModelRunner：
  真正准备输入并执行模型 forward / sample / pooling。
```

### 12.3 EngineCoreOutputs 是不是最终用户输出

不是。

```text
EngineCoreOutputs：
  EngineCore 返回给外层 Engine 的内部输出。

RequestOutput / PoolingRequestOutput：
  OutputProcessor 转换后的上层用户可见输出。
```

### 12.4 SchedulerOutput 和 ModelRunnerOutput 的区别

```text
SchedulerOutput：
  Scheduler 发出的计划。

ModelRunnerOutput：
  Worker 返回的结果。

update_from_output()：
  把计划和结果对账。
```

### 12.5 InprocClient 和 EngineCoreProc 的区别

```text
InprocClient：
  同进程直接调用 EngineCore。

EngineCoreProc：
  后台进程运行 EngineCore，额外管理 ZMQ、input/output queue、busy loop、shutdown。
```

---

## 13. 从“回答问题”的角度总结

如果问：

```text
EngineCore 在 vLLM V1 里负责什么？
```

可以回答：

```text
EngineCore 是 vLLM V1 Engine 内部的执行闭环总控。

它接收外层 Engine 处理好的 EngineCoreRequest，
转换成内部 Request 后交给 Scheduler；
每轮 step 中调用 Scheduler.schedule() 得到 SchedulerOutput，
再把 SchedulerOutput 交给 model_executor / Worker / ModelRunner 执行；
Worker 返回 ModelRunnerOutput 后，
EngineCore 把 SchedulerOutput 和 ModelRunnerOutput 一起交给 Scheduler.update_from_output()，
由 Scheduler 更新请求状态并生成 EngineCoreOutputs；
最后 EngineCoreOutputs 被 EngineCoreClient 返回给 LLMEngine / AsyncLLM，
再由 OutputProcessor 转成最终 RequestOutput / PoolingRequestOutput。
```

角色关系可以概括为：

```text
LLMEngine / AsyncLLM：
  外层用户接口、输入处理、输出处理。

EngineCoreClient：
  屏蔽 in-process / multi-process / async 通信差异。

EngineCore：
  内部执行闭环总控。

Scheduler：
  请求队列、调度决策、KV block、请求状态账本、输出回收。

model_executor / Worker / ModelRunner：
  模型 forward、sampling、pooling、KV / encoder 实际执行。

OutputProcessor：
  detokenize、stop string、RequestOutput 构造、异步队列推送。
```

---

## 14. 最关键的关系图

### 14.1 请求进入

```text
LLMEngine / AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
  → waiting / skipped_waiting / running
```

### 14.2 一轮 EngineCore.step

```text
EngineCore.step()
  → scheduler.has_requests()
  → scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output)
  → scheduler.get_grammar_bitmask(scheduler_output)
  → future.result()
  → model_executor.sample_tokens(grammar_output)  # 如果 execute_model 返回 None
  → _process_aborts_queue()
  → scheduler.update_from_output(scheduler_output, model_output)
  → dict[client_index, EngineCoreOutputs]
```

### 14.3 Worker 执行

```text
model_executor.execute_model()
  → Executor 实现分发到 Worker / ModelRunner
  → ModelRunner.execute_model()
  → _update_states(scheduler_output)
  → _prepare_inputs()
  → _build_attention_metadata()
  → _model_forward()
  → compute_logits / pooling
  → sample_tokens(grammar_output)
  → ModelRunnerOutput
```

### 14.4 输出返回

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → LLMEngine.step() / AsyncLLM output_handler
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

### 14.5 多进程后台模式

```text
Frontend EngineCoreClient
  → ZMQ input socket
  → EngineCoreProc input thread
  → input_queue
  → run_busy_loop()
  → _process_input_queue()
  → _process_engine_step()
  → output_queue
  → output thread
  → ZMQ output socket
  → Frontend EngineCoreClient
```

---

## 15. 推荐阅读顺序

如果第一次看 EngineCore，可以按下面顺序读：

```text
01_engine_core_role.md
  → 02_request_entry.md
  → 03_step_loop.md
  → 04_scheduler_output_flow.md
  → 05_worker_execution.md
  → 06_update_from_output_bridge.md
  → 07_engine_core_outputs.md
  → 08_lifecycle_and_async.md
```

如果重点想理解主执行链路：

```text
03_step_loop.md
  → 04_scheduler_output_flow.md
  → 05_worker_execution.md
  → 06_update_from_output_bridge.md
  → 07_engine_core_outputs.md
```

如果想和 Scheduler 文档联动：

```text
01_engine_core_role.md
  → ../scheduler/vllm_scheduler.md
  → 04_scheduler_output_flow.md
  → ../scheduler/08_update_after_worker_output.md
  → 07_engine_core_outputs.md
```

核心记忆点：

```text
EngineCore 负责编排一轮执行闭环；
Scheduler 负责调度决策和请求状态账本；
Worker / ModelRunner 负责真正 forward / sampling / pooling；
EngineCoreOutputs 是返回给上层 Engine 的内部结果；
OutputProcessor 才把内部结果变成用户可见输出。
```
