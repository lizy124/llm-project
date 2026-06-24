# 07. EngineCoreOutputs 如何返回给上层？

源码位置：`vllm/vllm/v1/engine/core.py`

本问题关注：`Scheduler.update_from_output()` 产生的 `EngineCoreOutputs` 如何被 EngineCore 返回给上层 `LLMEngine` / `AsyncLLM`，以及它和最终用户可见的 `RequestOutput` 是什么关系。

---

## 1. 一句话回答

`EngineCoreOutputs` 是 EngineCore 一轮 `step()` 返回给外层 Engine 的内部输出。

它由 Scheduler 在 `update_from_output()` 中生成：

```text
Scheduler.update_from_output()
  → EngineCoreOutput          # 单个请求的一次增量输出
  → EngineCoreOutputs         # 某个 client 本轮的一组输出
  → dict[int, EngineCoreOutputs]
```

EngineCore 拿到它后直接返回：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)

return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

位置：`vllm/vllm/v1/engine/core.py:504` 到 `vllm/vllm/v1/engine/core.py:508`

一句话：

```text
EngineCoreOutputs 是 EngineCore 和外层 Engine 之间的内部输出协议；
它不是最终用户输出，
外层还要经过 OutputProcessor 才会变成 RequestOutput / PoolingRequestOutput。
```

---

## 2. EngineCoreOutput 和 EngineCoreOutputs 的关系

先区分两个对象：

```text
EngineCoreOutput：
  单个 request 在某一轮 step 中产生的一条增量输出。

EngineCoreOutputs：
  某个 client 在某一轮 step 中收到的一组 EngineCoreOutput，
  以及本轮 scheduler_stats / timestamp / finished_requests 等附加信息。
```

### 2.1 EngineCoreOutput：单个请求输出

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
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None

    prefill_stats: PrefillStats | None = None

    routed_experts: np.ndarray | None = None
    num_nans_in_logits: int = 0
```

位置：`vllm/vllm/v1/engine/__init__.py:175` 到 `vllm/vllm/v1/engine/__init__.py:201`

核心字段含义：

```text
request_id：
  这条输出属于哪个内部请求。

new_token_ids：
  本轮新增 token ids。

new_logprobs：
  本轮新增 token 的 logprobs。

new_prompt_logprobs_tensors：
  prompt logprobs。

pooling_output：
  pooling / embedding 类任务的输出。

finish_reason / stop_reason：
  请求是否结束，以及结束原因。

events：
  请求级事件，用于统计和观测。

kv_transfer_params：
  KV transfer 相关返回参数。

trace_headers：
  tracing 相关 headers。

prefill_stats：
  prefill 阶段统计。

routed_experts：
  MoE routed experts 信息。

num_nans_in_logits：
  logits 中 NaN 数量，非 0 表示输出可能损坏。
```

它还有一个便捷属性：

```python
@property
def finished(self) -> bool:
    return self.finish_reason is not None
```

位置：`vllm/vllm/v1/engine/__init__.py:203` 到 `vllm/vllm/v1/engine/__init__.py:205`

也就是说：

```text
EngineCoreOutput.finished 为 True，表示这个请求在 EngineCore 层已经结束。
```

---

### 2.2 EngineCoreOutputs：一组请求输出

`EngineCoreOutputs` 定义在：

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

核心字段含义：

```text
engine_index：
  多 engine / DP 场景中标识输出来自哪个 engine。

outputs：
  本轮请求输出列表，每个元素是 EngineCoreOutput。

scheduler_stats：
  Scheduler 本轮统计信息。

timestamp：
  EngineCoreOutputs 创建时间，用于外层统计延迟。

utility_output：
  profile / reset / sleep / LoRA 等 utility 调用的返回。

finished_requests：
  一批已经结束的 request ids，用于多 engine / 多 client 生命周期追踪。

wave_complete / start_wave：
  DP wave 控制信息。
```

如果创建时没有传 `timestamp`，会自动填当前 monotonic 时间：

```python
def __post_init__(self):
    if self.timestamp == 0.0:
        self.timestamp = time.monotonic()
```

位置：`vllm/vllm/v1/engine/__init__.py:246` 到 `vllm/vllm/v1/engine/__init__.py:248`

可以理解为：

```text
EngineCoreOutput 是“请求级输出”；
EngineCoreOutputs 是“EngineCore 本轮返回给某个 client 的批量输出”。
```

---

## 3. EngineCoreOutputs 是谁生成的

`EngineCoreOutputs` 不是 EngineCore 自己手动拼出来的，而是 Scheduler 在 `update_from_output()` 中生成。

Scheduler 先按 `client_index` 收集单请求输出：

```python
outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1488`

当某个请求有新 token、pooling output、KV transfer params 或 stopped 时，Scheduler 添加一条 `EngineCoreOutput`：

```python
outputs[request.client_index].append(
    EngineCoreOutput(
        request_id=req_id,
        new_token_ids=new_token_ids,
        finish_reason=finish_reason,
        new_logprobs=new_logprobs,
        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
        pooling_output=pooler_output,
        stop_reason=request.stop_reason,
        events=request.take_events(),
        prefill_stats=request.take_prefill_stats(),
        kv_transfer_params=kv_transfer_params,
        trace_headers=request.trace_headers,
        routed_experts=routed_experts,
        num_nans_in_logits=request.num_nans_in_logits,
    )
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1689` 到 `vllm/vllm/v1/core/sched/scheduler.py:1706`

最后把每个 client 的输出列表包装成 `EngineCoreOutputs`：

```python
engine_core_outputs = {
    client_index: EngineCoreOutputs(outputs=outs)
    for client_index, outs in outputs.items()
}
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1770` 到 `vllm/vllm/v1/core/sched/scheduler.py:1775`

所以返回给 EngineCore 的是：

```text
dict[int, EngineCoreOutputs]
```

其中：

```text
key：client_index
value：这个 client 本轮要收到的 EngineCoreOutputs
```

---

## 4. 为什么按 client_index 分组

`EngineCore.step()` 返回的是：

```python
tuple[dict[int, EngineCoreOutputs], bool]
```

位置：`vllm/vllm/v1/engine/core.py:479`

为什么不是直接返回一个 `EngineCoreOutputs`？

因为 EngineCore 可能同时服务多个前端 client。

请求进入 EngineCore 时，`EngineCoreRequest` 带有 `client_index`，Scheduler 内部 Request 也保留这个信息。

在生成输出时，Scheduler 使用：

```python
outputs[request.client_index].append(...)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1690`

这样可以保证：

```text
client 0 的请求输出只回到 client 0；
client 1 的请求输出只回到 client 1；
多客户端共用 EngineCore 时，输出不会混淆。
```

### 4.1 InprocClient 只取 client 0

in-process 模式下，通常只有一个 client。

`InprocClient.get_output()` 会取 `outputs.get(0)`：

```python
def get_output(self) -> EngineCoreOutputs:
    outputs, model_executed = self.engine_core.step_fn()
    self.engine_core.post_step(model_executed=model_executed)
    return outputs and outputs.get(0) or EngineCoreOutputs()
```

位置：`vllm/vllm/v1/engine/core_client.py:289` 到 `vllm/vllm/v1/engine/core_client.py:292`

也就是说：

```text
同步 inproc 场景只关心 client_index = 0 的输出。
```

如果没有输出，会返回一个空的 `EngineCoreOutputs()`。

### 4.2 MPClient 按 client socket 返回

多进程模式下，`EngineCoreProc` 会把 `(client_index, EngineCoreOutputs)` 放入 output queue：

```python
outputs, model_executed = self.step_fn()
for output in outputs.items() if outputs else ():
    self.output_queue.put_nowait(output)
```

位置：`vllm/vllm/v1/engine/core.py:1301` 到 `vllm/vllm/v1/engine/core.py:1305`

输出线程取出时会按 `client_index` 发送到对应 socket：

```python
client_index, outputs = output
outputs.engine_index = engine_index
...
tracker = sockets[client_index].send_multipart(
    buffers, copy=False, track=True
)
```

位置：`vllm/vllm/v1/engine/core.py:1628` 到 `vllm/vllm/v1/engine/core.py:1645`

所以多进程场景中，`client_index` 是路由输出的关键字段。

---

## 5. EngineCoreOutputs 如何回到同步 LLMEngine

同步 `LLMEngine.step()` 的第一步就是从 EngineCoreClient 拉输出：

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
self.output_processor.update_scheduler_stats(outputs.scheduler_stats)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:306` 到 `vllm/vllm/v1/engine/llm_engine.py:315`

也就是说，同步路径是：

```text
LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs(outputs.outputs)
  → RequestOutput / PoolingRequestOutput
```

如果某些请求因为 stop string 在 OutputProcessor 中被判定需要 abort，LLMEngine 会再通知 EngineCore：

```python
self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:317` 到 `vllm/vllm/v1/engine/llm_engine.py:318`

最终同步 `LLMEngine.step()` 返回的是：

```python
return processed_outputs.request_outputs
```

位置：`vllm/vllm/v1/engine/llm_engine.py:334`

所以：

```text
EngineCoreOutputs 是同步 Engine 的内部输入；
RequestOutput / PoolingRequestOutput 才是 LLMEngine.step() 返回给调用方的输出。
```

---

## 6. EngineCoreOutputs 如何回到异步 AsyncLLM

异步路径中，`AsyncLLM` 有一个后台 `output_handler()`。

它循环从 EngineCoreClient 拉输出：

```python
outputs = await engine_core.get_output_async()
num_outputs = len(outputs.outputs)
```

位置：`vllm/vllm/v1/engine/async_llm.py:656` 到 `vllm/vllm/v1/engine/async_llm.py:661`

然后分 chunk 交给 OutputProcessor：

```python
processed_outputs = output_processor.process_outputs(
    outputs_slice, outputs.timestamp, iteration_stats
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:671` 到 `vllm/vllm/v1/engine/async_llm.py:677`

异步模式下，`OutputProcessor.process_outputs()` 一般不会把 `RequestOutput` 直接返回给 `output_handler()`，而是推入每个请求自己的队列：

```python
# NOTE: RequestOutputs are pushed to their queues.
assert not processed_outputs.request_outputs
```

位置：`vllm/vllm/v1/engine/async_llm.py:678` 到 `vllm/vllm/v1/engine/async_llm.py:679`

如果有 stop string 触发的 abort，也会异步通知 EngineCore：

```python
await engine_core.abort_requests_async(
    processed_outputs.reqs_to_abort
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:685` 到 `vllm/vllm/v1/engine/async_llm.py:689`

异步路径可以概括为：

```text
AsyncLLM output_handler
  → await EngineCoreClient.get_output_async()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → 每个请求的 async queue
  → async generator yield RequestOutput
```

所以同步和异步的区别是：

```text
同步 LLMEngine：process_outputs() 返回 RequestOutput 列表；
异步 AsyncLLM：process_outputs() 把 RequestOutput 推进请求队列，由 generate() 异步消费。
```

但二者共同点是：

```text
都先消费 EngineCoreOutputs，
再由 OutputProcessor 转成上层输出。
```

---

## 7. MPClient 如何接收 EngineCoreOutputs

多进程同步客户端 `SyncMPClient` 有一个输出线程，负责从 ZMQ socket 读 `EngineCoreOutputs`：

```python
frames = out_socket.recv_multipart(copy=False)
resources.validate_alive(frames)
outputs: EngineCoreOutputs = decoder.decode(frames)
if outputs.utility_output:
    _process_utility_output(outputs.utility_output, utility_results)
else:
    outputs_queue.put_nowait(outputs)
```

位置：`vllm/vllm/v1/engine/core_client.py:824` 到 `vllm/vllm/v1/engine/core_client.py:830`

同步 `get_output()` 再从队列里取：

```python
outputs = self.outputs_queue.get()
...
return outputs
```

位置：`vllm/vllm/v1/engine/core_client.py:849` 到 `vllm/vllm/v1/engine/core_client.py:859`

异步 `AsyncMPClient` 也有类似任务，从 socket 接收并 decode：

```python
frames = await output_socket.recv_multipart(copy=False)
resources.validate_alive(frames)
outputs: EngineCoreOutputs = decoder.decode(frames)
```

位置：`vllm/vllm/v1/engine/core_client.py:1005` 到 `vllm/vllm/v1/engine/core_client.py:1010`

然后只把真正需要上层处理的输出放入队列：

```python
if outputs.outputs or outputs.scheduler_stats:
    outputs_queue.put_nowait(outputs)
```

位置：`vllm/vllm/v1/engine/core_client.py:1042` 到 `vllm/vllm/v1/engine/core_client.py:1043`

这说明：

```text
EngineCoreOutputs 是 EngineCoreProc 跨进程返回给 EngineCoreClient 的序列化消息。
```

---

## 8. 没有 token 的 prefill step 是否返回输出

不一定。

Scheduler 在构造 `EngineCoreOutput` 时有条件判断：

```python
if (
    new_token_ids
    or pooler_output is not None
    or kv_transfer_params
    or stopped
):
    outputs[request.client_index].append(EngineCoreOutput(...))
else:
    # Invariant: EngineCore returns no partial prefill outputs.
    assert not prompt_logprobs_tensors
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1683` 到 `vllm/vllm/v1/core/sched/scheduler.py:1709`

这里最关键的是注释：

```text
EngineCore returns no partial prefill outputs.
```

也就是说：

```text
如果某个 prefill chunk 没有生成 token、没有 pooling output、没有 stop、没有 KV transfer params，
那这一轮不会给上层返回这个请求的 EngineCoreOutput。
```

为什么？

因为上层用户关心的是：

```text
新生成 token；
请求结束；
pooling 输出；
必要的 KV transfer / 状态事件。
```

单纯的中间 prefill 进度不需要作为用户输出返回。

但是没有 token 不代表一定没有 `EngineCoreOutputs`。

下面几种情况仍可能返回：

```text
pooling_output 不为空；
请求 stopped / finished；
kv_transfer_params 不为空；
scheduler_stats 需要返回；
finished_requests 需要通知；
utility_output 需要返回。
```

所以要区分：

```text
没有 token：不一定没有 EngineCoreOutputs；
没有任何输出 / stats / utility：才可能返回空或不入队。
```

---

## 9. EngineCoreOutputs 和 RequestOutput 的区别

容易混淆的一点是：`EngineCoreOutputs` 不是最终用户看到的输出。

它还没有做完整的上层处理，例如：

```text
token detokenization；
stop string 判断；
RequestOutput 对象构造；
async queue 推送；
外部 request_id 映射；
输出统计更新。
```

这些由 `OutputProcessor.process_outputs()` 完成。

`OutputProcessor.process_outputs()` 的注释说明它会：

```python
"""
Process the EngineCoreOutputs:
1) Compute stats for logging
2) Detokenize
3) Create and handle RequestOutput objects:
    * If there is a queue (for usage with AsyncLLM),
      put the RequestOutput objects into the queue ...

    * If there is no queue (for usage with LLMEngine),
      return a list of RequestOutput objects.
"""
```

位置：`vllm/vllm/v1/engine/output_processor.py:576` 到 `vllm/vllm/v1/engine/output_processor.py:593`

所以对象层级是：

```text
ModelRunnerOutput
  Worker / ModelRunner 返回的 batch 级模型结果。

EngineCoreOutput
  Scheduler 转换后的单请求内部输出。

EngineCoreOutputs
  EngineCore 返回给某个 client 的一组内部输出。

RequestOutput / PoolingRequestOutput
  OutputProcessor 转换后的上层用户可见输出。
```

---

## 10. 总结

`EngineCoreOutputs` 的返回路径是：

```text
ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput
  → dict[client_index, EngineCoreOutputs]
  → EngineCore.step() return
  → EngineCoreClient.get_output() / get_output_async()
  → LLMEngine.step() / AsyncLLM output_handler
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

关键点：

```text
EngineCoreOutput 是单请求增量输出；
EngineCoreOutputs 是某个 client 的一批内部输出；
EngineCore.step() 返回 dict[int, EngineCoreOutputs]，用于多 client 路由；
InprocClient 通常只取 client 0；
MPClient 会按 client_index 把输出发回对应前端；
OutputProcessor 才会把 EngineCoreOutputs 转成最终 RequestOutput。
```

如果要回答：

```text
EngineCoreOutputs 如何返回给上层？
```

可以概括为：

```text
Scheduler.update_from_output() 会把 Worker 的 ModelRunnerOutput 转成请求级 EngineCoreOutput，
再按 request.client_index 分组为 dict[int, EngineCoreOutputs] 返回给 EngineCore。
EngineCore.step() 把这个 dict 返回给 EngineCoreClient；
in-process 模式下 InprocClient 取 client 0 的 EngineCoreOutputs，
多进程模式下 EngineCoreProc 通过 output_queue 和 ZMQ 按 client_index 发回前端。
上层 LLMEngine / AsyncLLM 拿到 EngineCoreOutputs 后，
再交给 OutputProcessor.process_outputs() 转成 RequestOutput / PoolingRequestOutput。
```
