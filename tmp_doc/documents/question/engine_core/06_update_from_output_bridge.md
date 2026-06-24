# 06. EngineCore 如何使用 Scheduler.update_from_output()？

源码位置：`vllm/vllm/v1/engine/core.py`

本问题关注：EngineCore 在拿到 Worker 返回的 `ModelRunnerOutput` 后，为什么要把 `SchedulerOutput` 和 `ModelRunnerOutput` 一起交回 `Scheduler.update_from_output()`，以及这个调用如何把“模型执行结果”转换成上层可消费的 `EngineCoreOutputs`。

---

## 1. 一句话回答

`SchedulerOutput` 是本轮执行计划，`ModelRunnerOutput` 是 Worker / ModelRunner 的真实执行结果。

EngineCore 在一轮 `step()` 末尾调用：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:504` 到 `vllm/vllm/v1/engine/core.py:506`

它的作用是：

```text
把本轮“发出去的计划”和“实际执行结果”合并，
交给 Scheduler 更新请求状态、处理 stop / spec decode / logprobs / pooling / KV transfer / 资源释放，
最后生成 EngineCoreOutputs 返回给 EngineCore。
```

主线是：

```text
EngineCore.step()
  → scheduler.schedule()
  → SchedulerOutput  # 本轮计划
  → model_executor.execute_model(scheduler_output)
  → ModelRunnerOutput  # 真实结果
  → scheduler.update_from_output(scheduler_output, model_output)
  → dict[int, EngineCoreOutputs]
  → EngineCore 返回给上层 EngineCoreClient
```

一句话：

```text
EngineCore 是桥；
它不自己解释 Worker 输出，
而是把 SchedulerOutput 和 ModelRunnerOutput 一起交回 Scheduler，
让 Scheduler 完成状态对账和输出构造。
```

---

## 2. update_from_output 在 EngineCore 中的位置

普通 `EngineCore.step()` 的核心代码是：

```python
if not self.scheduler.has_requests():
    return {}, False
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
with (...):
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)

return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

位置：`vllm/vllm/v1/engine/core.py:486` 到 `vllm/vllm/v1/engine/core.py:508`

这个顺序很重要：

```text
1. schedule()：Scheduler 先发出计划；
2. execute_model() / sample_tokens()：Worker 按计划执行；
3. _process_aborts_queue()：先处理执行期间到达的 abort；
4. update_from_output()：最后用 Worker 结果更新 Scheduler 状态。
```

为什么 abort 要在 `update_from_output()` 前处理？

因为模型执行期间，外部可能取消某些请求。

如果先处理 Worker 输出，再处理 abort，Scheduler 可能会给已经取消的请求继续生成输出或继续保留状态。

所以 EngineCore 在进入 `update_from_output()` 前先调用：

```python
self._process_aborts_queue()
```

位置：`vllm/vllm/v1/engine/core.py:501` 到 `vllm/vllm/v1/engine/core.py:503`

这样 Scheduler 回收输出时可以看到最新的请求状态。

---

## 3. 为什么需要 SchedulerOutput

`SchedulerOutput` 是本轮执行计划。

它告诉 Scheduler：

```text
这一轮原本调度了哪些请求；
每个请求原本调度了多少 token；
哪些 spec decode tokens 被送去执行；
哪些 encoder inputs 被处理；
是否有 KV / EC connector metadata；
哪些请求是本轮通知 Worker 清理的 finished request。
```

`Scheduler.update_from_output()` 的签名是：

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1464` 到 `vllm/vllm/v1/core/sched/scheduler.py:1468`

函数一开始就取出 SchedulerOutput 的关键字段：

```python
num_scheduled_tokens = scheduler_output.num_scheduled_tokens
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1472`

这个字段是后续遍历请求的基础：

```python
for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
    assert num_tokens_scheduled > 0
    ...
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1527` 到 `vllm/vllm/v1/core/sched/scheduler.py:1528`

也就是说，Scheduler 不是遍历 Worker 返回的所有字段来决定谁被调度，而是以 `SchedulerOutput.num_scheduled_tokens` 为准。

原因是：

```text
SchedulerOutput 才是 Scheduler 自己发出去的账本；
ModelRunnerOutput 只是 Worker 返回的结果。
```

### 3.1 对 spec decode 的对账

Spec decode 场景下，Scheduler 需要知道本轮原本发出了哪些 draft tokens：

```python
scheduled_spec_token_ids = (
    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1548` 到 `vllm/vllm/v1/core/sched/scheduler.py:1550`

然后根据 Worker 返回的 `generated_token_ids` 判断接受 / 拒绝数量：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1554` 到 `vllm/vllm/v1/core/sched/scheduler.py:1557`

如果有 draft token 被拒绝，Scheduler 会把之前乐观推进的进度扣回来：

```python
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1563` 到 `vllm/vllm/v1/core/sched/scheduler.py:1568`

这里必须依赖 `SchedulerOutput`。

因为只有它知道：

```text
本轮到底发出了几个 scheduled_spec_decode_tokens。
```

### 3.2 对 encoder input 的释放

Scheduler 还会在确认本轮执行后释放已经安全的 encoder inputs：

```python
if request.has_encoder_inputs:
    self._free_encoder_inputs(request)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1577` 到 `vllm/vllm/v1/core/sched/scheduler.py:1579`

释放是否安全，依赖请求当前进度，而这个进度与本轮 `SchedulerOutput` 中的 token 调度密切相关。

### 3.3 对 deferred free 的推进

如果开启 deferred block free，Scheduler 会根据本轮是否实际执行 token 来推进 fence：

```python
if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
    self.processed_step_seq += 1
    self._drain_deferred_frees()
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1478` 到 `vllm/vllm/v1/core/sched/scheduler.py:1482`

这也必须依赖 `SchedulerOutput.total_num_scheduled_tokens`。

含义是：

```text
只有本轮确实有 GPU 写入完成后，
之前延迟释放的 KV blocks 才可能变得安全。
```

### 3.4 对性能统计的生成

Scheduler 还可能用 `SchedulerOutput` 生成 step 级性能统计：

```python
perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1484` 到 `vllm/vllm/v1/core/sched/scheduler.py:1486`

所以 `SchedulerOutput` 不只是 Worker 的输入，也是 Scheduler 回收阶段的依据。

可以概括为：

```text
没有 SchedulerOutput，Scheduler 就不知道自己上一轮到底发出了什么任务；
也就无法正确修正进度、处理 spec decode、释放资源和生成统计。
```

---

## 4. 为什么需要 ModelRunnerOutput

`ModelRunnerOutput` 是 Worker / ModelRunner 的真实执行结果。

`update_from_output()` 开头会取出这些字段：

```python
sampled_token_ids = model_runner_output.sampled_token_ids
logprobs = model_runner_output.logprobs
prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
pooler_outputs = model_runner_output.pooler_output
num_nans_in_logits = model_runner_output.num_nans_in_logits
kv_connector_output = model_runner_output.kv_connector_output
cudagraph_stats = model_runner_output.cudagraph_stats
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1469` 到 `vllm/vllm/v1/core/sched/scheduler.py:1476`

这些信息回答的是：

```text
Worker 实际生成了哪些 token；
这些 token 的 logprobs 是什么；
prompt logprobs 是什么；
pooling 模型是否有 pooling output；
KV connector 是否完成 / 失败；
logits 是否出现 NaN；
CUDA graph 执行统计是什么。
```

### 4.1 取出每个请求的输出

Scheduler 遍历 `num_scheduled_tokens` 中的请求后，会用 `req_id_to_index` 找到这个请求在 Worker 输出中的位置：

```python
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = (
    sampled_token_ids[req_index] if sampled_token_ids else []
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1543` 到 `vllm/vllm/v1/core/sched/scheduler.py:1546`

这里体现了两个输入的配合：

```text
SchedulerOutput.num_scheduled_tokens：告诉 Scheduler 本轮哪些请求需要处理；
ModelRunnerOutput.req_id_to_index：告诉 Scheduler 去 Worker 输出的哪个位置取结果。
```

### 4.2 更新请求 token 并检查停止

对于生成出来的新 token，Scheduler 调用：

```python
new_token_ids, stopped = self._update_request_with_output(
    request, new_token_ids
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1590` 到 `vllm/vllm/v1/core/sched/scheduler.py:1593`

内部会 append token，并检查 stop：

```python
request.append_output_token_ids(output_token_id)
stopped = check_stop(request, self.max_model_len)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1856` 到 `vllm/vllm/v1/core/sched/scheduler.py:1861`

所以 stop 判断不是 Worker 做最终决定，而是 Scheduler 用 Worker 结果更新请求状态后判断。

### 4.3 pooling 输出

如果是 pooling 模型，Scheduler 会从 `pooler_outputs` 取结果：

```python
pooler_output = pooler_outputs[req_index] if pooler_outputs else None
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1584`

如果请求是 pooling request 且有输出，则直接停止：

```python
elif request.pooling_params and pooler_output is not None:
    request.status = RequestStatus.FINISHED_STOPPED
    stopped = True
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1594` 到 `vllm/vllm/v1/core/sched/scheduler.py:1597`

### 4.4 logprobs 和 prompt logprobs

如果请求需要 sample logprobs，Scheduler 会切出对应请求的 logprobs：

```python
new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1670` 到 `vllm/vllm/v1/core/sched/scheduler.py:1676`

prompt logprobs 则通过 req_id 查：

```python
prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1681` 到 `vllm/vllm/v1/core/sched/scheduler.py:1682`

### 4.5 KV connector 输出

如果 Worker 返回了 KV connector 输出，Scheduler 会处理 KV transfer 完成状态：

```python
if kv_connector_output:
    self._update_from_kv_xfer_finished(kv_connector_output)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1732` 到 `vllm/vllm/v1/core/sched/scheduler.py:1734`

如果有外部 KV load 失败，还会先处理 invalid blocks：

```python
if kv_connector_output and kv_connector_output.invalid_block_ids:
    failed_kv_load_req_ids = self._handle_invalid_blocks(
        kv_connector_output.invalid_block_ids,
        num_scheduled_tokens,
    )
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1491` 到 `vllm/vllm/v1/core/sched/scheduler.py:1499`

所以 `ModelRunnerOutput` 是真实结果来源。

可以概括为：

```text
没有 ModelRunnerOutput，Scheduler 只有计划，没有结果；
它不知道生成了什么 token、是否有 pooling output、KV transfer 是否失败、logprobs 是什么。
```

---

## 5. update_from_output 如何生成 EngineCoreOutput

在 `update_from_output()` 内部，Scheduler 会按 `client_index` 收集输出：

```python
outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1488`

当某个请求有新 token、pooling output、KV transfer params、或者 stopped 时，会构造 `EngineCoreOutput`：

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

这里的 `EngineCoreOutput` 是单个请求的一次输出。

它的定义包括：

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

可以理解为：

```text
ModelRunnerOutput 是 Worker 维度的原始执行结果；
EngineCoreOutput 是面向某个请求的增量输出。
```

Scheduler 在这里完成了从“batch 输出”到“request 输出”的转换。

---

## 6. update_from_output 返回什么

`update_from_output()` 返回：

```python
dict[int, EngineCoreOutputs]
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1464` 到 `vllm/vllm/v1/core/sched/scheduler.py:1468`

其中：

```text
key：client_index
value：这个 client 本轮要收到的 EngineCoreOutputs
```

最终构造发生在：

```python
engine_core_outputs = {
    client_index: EngineCoreOutputs(outputs=outs)
    for client_index, outs in outputs.items()
}
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1770` 到 `vllm/vllm/v1/core/sched/scheduler.py:1775`

`EngineCoreOutputs` 的定义是：

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

### 6.1 为什么按 client_index 分组

因为 EngineCore 可能服务多个前端 client。

请求进入 EngineCore 时，`EngineCoreRequest` 里有：

```text
client_index
```

Scheduler 在构造输出时使用：

```python
outputs[request.client_index].append(...)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1690`

所以返回值按 client 分组：

```text
client 0 的请求输出返回给 client 0；
client 1 的请求输出返回给 client 1；
不同 client 的输出不会混在同一个 EngineCoreOutputs 里。
```

### 6.2 finished_requests

如果 Scheduler 启用了 `include_finished_set`，还会把 finished request ids 放进 `EngineCoreOutputs`：

```python
if (eco := engine_core_outputs.get(client_index)) is not None:
    eco.finished_requests = finished_set
else:
    engine_core_outputs[client_index] = EngineCoreOutputs(
        finished_requests=finished_set
    )
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1777` 到 `vllm/vllm/v1/core/sched/scheduler.py:1789`

这用于多 engine / 多 client 场景下高效追踪请求生命周期。

### 6.3 scheduler_stats

如果本轮产生 stats，Scheduler 会把 stats 放到其中一个 `EngineCoreOutputs`：

```python
if (stats := self.make_stats(...)) is not None:
    if (eco := next(iter(engine_core_outputs.values()), None)) is None:
        engine_core_outputs[0] = eco = EngineCoreOutputs()
    eco.scheduler_stats = stats
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1791` 到 `vllm/vllm/v1/core/sched/scheduler.py:1801`

注意：

```text
即使本轮没有 request output，也可能为了返回 stats 构造一个空 EngineCoreOutputs。
```

---

## 7. EngineCore 自己做什么，不做什么

在这个桥接点上，EngineCore 做的事情很少，但很关键。

它做：

```text
保存 scheduler_output；
调用 Worker 得到 model_output；
处理执行期间到达的 abort；
调用 scheduler.update_from_output(scheduler_output, model_output)；
把返回的 EngineCoreOutputs 原样交给上层。
```

它不做：

```text
不自己 append token 到 Request；
不自己判断 stop；
不自己释放 KV blocks；
不自己处理 spec decode 接受 / 拒绝；
不自己切 logprobs；
不自己把 batch output 拆成 request output；
不自己按 client_index 分组。
```

这些都由 Scheduler 在 `update_from_output()` 里完成。

所以 EngineCore 的职责边界是：

```text
EngineCore：桥接计划和结果，驱动闭环；
Scheduler：维护请求状态账本，做调度后对账；
Worker / ModelRunner：真实执行模型计算并返回结果。
```

---

## 8. 和 scheduler/08 文档的关系

本文重点讲 EngineCore 视角的桥接关系。

也就是回答：

```text
EngineCore 为什么要调用 update_from_output？
为什么要同时传 SchedulerOutput 和 ModelRunnerOutput？
update_from_output 返回的 EngineCoreOutputs 如何回到 EngineCore？
```

更细的 Scheduler 内部逻辑见：

```text
../scheduler/08_update_after_worker_output.md
```

那里关注的是 `Scheduler.update_from_output()` 内部细节，例如：

```text
deferred free；
KV load failure / invalid blocks；
spec decode 接受 / 拒绝；
structured output grammar 推进；
stop 检查；
_free_request()；
KV Connector 状态更新；
EngineCoreOutputs 组装；
stats / KV events 发布。
```

可以这样区分：

```text
engine_core/06：讲 EngineCore 如何把计划和结果交回 Scheduler；
scheduler/08：讲 Scheduler 拿到计划和结果后内部怎么处理。
```

---

## 9. 总结

`Scheduler.update_from_output()` 是 EngineCore 一轮执行闭环的“收结果”阶段。

完整链路是：

```text
Scheduler.schedule()
  → SchedulerOutput
  → EngineCore 保存这份计划
  → model_executor.execute_model(scheduler_output)
  → Worker / ModelRunner 执行
  → ModelRunnerOutput
  → EngineCore._process_aborts_queue()
  → Scheduler.update_from_output(scheduler_output, model_output)
  → 更新 Request / Scheduler 状态
  → 构造 dict[client_index, EngineCoreOutputs]
  → EngineCore 返回上层
```

关键点：

```text
SchedulerOutput 是计划账本；
ModelRunnerOutput 是实际结果；
update_from_output() 是对账和状态更新；
EngineCoreOutputs 是返回给外层 Engine 的内部输出。
```

如果要回答：

```text
EngineCore 如何使用 Scheduler.update_from_output()？
```

可以概括为：

```text
EngineCore 在每轮 step 中先调用 Scheduler.schedule() 得到 SchedulerOutput，
再把它交给 model_executor / Worker 执行得到 ModelRunnerOutput。
执行结束后，EngineCore 先处理执行期间到达的 abort，
然后把 SchedulerOutput 和 ModelRunnerOutput 一起传给 Scheduler.update_from_output()。
Scheduler 以 SchedulerOutput 为本轮计划账本，以 ModelRunnerOutput 为真实执行结果，
更新请求状态、处理 stop / spec decode / logprobs / pooling / KV transfer，
最后按 client_index 返回 EngineCoreOutputs。
```
