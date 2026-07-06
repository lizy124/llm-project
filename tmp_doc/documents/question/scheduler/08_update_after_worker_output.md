# 08. Worker 执行完后，如何更新请求状态、释放 block、返回输出？

源码位置：`vllm/vllm/v1/core/sched/scheduler.py`

本问题关注：Scheduler 发出 `SchedulerOutput` 后，Worker / ModelRunner 执行完返回 `ModelRunnerOutput`，Scheduler 如何处理 sampled token、logprobs、pooling output、spec decode 接受/拒绝、structured output grammar、KV Connector output、请求停止、资源释放，并最终返回 `EngineCoreOutputs`。

一句话概括：

```text
schedule() 是“发任务”；
update_from_output() 是“收结果”。

Scheduler 在 update_from_output() 中把 Worker 的真实执行结果合并回请求状态，
修正 schedule 阶段的乐观进度，处理 stop / spec / grammar / KV transfer，
释放或延迟释放资源，最后按 client_index 返回 EngineCoreOutputs。
```

---

## 1. 一句话回答

Worker 执行完后，Scheduler 通过：

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`

来回收这一轮执行结果。

整体流程可以概括为：

```text
update_from_output()
  → 处理 deferred free
  → 处理 KV load 失败 / invalid blocks
  → 遍历本轮 num_scheduled_tokens 中的请求
  → 从 ModelRunnerOutput 取 sampled token / pooling output / logprobs
  → 修正 spec decode 接受 / 拒绝
  → 释放已经安全的 encoder inputs
  → append 输出 token 并检查 stop
  → 推进 structured output grammar
  → 请求停止后处理 streaming / resumable
  → 请求真正结束则 _free_request()
  → 从 running / waiting 移除 stopped 请求
  → 更新 KV Connector finished_recving / finished_sending
  → 收集 KV events / stats
  → 返回 dict[client_index, EngineCoreOutputs]
```

返回值是：

```text
dict[int, EngineCoreOutputs]
```

其中 key 是 `client_index`，value 是这个 client 本轮要收到的输出集合。

---

## 2. `schedule()` 和 `update_from_output()` 的关系

`Scheduler.schedule()` 负责生成一轮执行计划：

```text
哪些请求执行；
每个请求执行多少 token；
需要哪些 KV blocks；
需要哪些 encoder inputs；
需要哪些 connector metadata。
```

这些信息被打包成 `SchedulerOutput` 发给 Worker / ModelRunner。

但是 `schedule()` 在构造完输出后，会立即调用：

```python
self._update_after_schedule(scheduler_output)
```

位置：`scheduler.py:1096`

其中会先推进请求进度：

```python
request.num_computed_tokens += num_scheduled_token
```

位置：`scheduler.py:1139`

这是一种“乐观推进”。

为什么要这样？

```text
async scheduling / pipeline parallel 下，
下一轮 schedule() 可能在上一轮 Worker 输出回来之前就发生。
如果不先更新 num_computed_tokens，Scheduler 可能重复调度同一段 token。
```

`update_from_output()` 的作用就是：

```text
把 Worker 的真实结果和 Scheduler 的乐观状态对齐。
```

例如：

```text
spec decode 中有 draft token 被拒绝，
update_from_output() 会把 num_computed_tokens 扣回来。

外部 KV load 失败，
update_from_output() 会把请求进度回退到失败 block 之前。
```

---

## 3. `update_from_output()` 的输入输出

入口参数有两个：

```python
scheduler_output: SchedulerOutput
model_runner_output: ModelRunnerOutput
```

位置：`scheduler.py:1465`

可以理解为：

```text
SchedulerOutput：
  Scheduler 本轮发出去的“计划”。

ModelRunnerOutput：
  Worker / ModelRunner 本轮执行后的“结果”。
```

`update_from_output()` 开头会取出这些字段：

```python
sampled_token_ids = model_runner_output.sampled_token_ids
logprobs = model_runner_output.logprobs
prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
num_scheduled_tokens = scheduler_output.num_scheduled_tokens
pooler_outputs = model_runner_output.pooler_output
num_nans_in_logits = model_runner_output.num_nans_in_logits
kv_connector_output = model_runner_output.kv_connector_output
cudagraph_stats = model_runner_output.cudagraph_stats
```

位置：`scheduler.py:1468`

这里把“计划”和“实际输出”汇合到一起：

```text
num_scheduled_tokens 告诉 Scheduler 本轮哪些请求被执行了；
sampled_token_ids / pooler_output 告诉 Scheduler Worker 实际产出了什么；
kv_connector_output 告诉 Scheduler 外部 KV transfer 是否完成或失败。
```

---

## 4. deferred free：先释放已经安全的 blocks

`update_from_output()` 开头先处理 deferred free：

```python
if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
    self.processed_step_seq += 1
    self._drain_deferred_frees()
```

位置：`scheduler.py:1477`

含义是：

```text
之前某些请求结束或被抢占时，KV blocks 可能没有立即归还 block pool。
因为 GPU 上还有 in-flight step 可能正在写这些 blocks。
```

现在 Worker 返回了这一轮输出，说明对应 GPU write 已经完成，可以推进：

```text
processed_step_seq
```

并尝试释放已经安全的 deferred blocks。

`_drain_deferred_frees()` 会检查 fence：

```python
while self.deferred_frees:
    fence, _ = self.deferred_frees[0]
    if fence > self.processed_step_seq:
        break
    _, blocks = self.deferred_frees.popleft()
    self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
```

位置：`scheduler.py:2092`

这和 `06_kv_block_allocation_and_preemption.md` 中的 deferred free 逻辑形成闭环。

---

## 5. KV load failure / invalid blocks 处理

外部 KV load 可能失败。Worker 会通过 `kv_connector_output.invalid_block_ids` 报告无效 blocks。

`update_from_output()` 中先处理这类失败：

```python
failed_kv_load_req_ids = None
if kv_connector_output and kv_connector_output.invalid_block_ids:
    failed_kv_load_req_ids = self._handle_invalid_blocks(
        kv_connector_output.invalid_block_ids,
        num_scheduled_tokens,
    )
```

位置：`scheduler.py:1490`

源码注释解释：

```python
# These blocks contain externally computed tokens that failed to
# load. Identify affected requests and adjust their computed token
# count to trigger recomputation of the invalid blocks.
```

位置：`scheduler.py:1491`

含义是：

```text
某些 external KV blocks load 失败后，
Scheduler 要找出受影响请求，
并把请求的 num_computed_tokens 回退到失败 block 之前，
让后续 schedule 重新计算这些 token。
```

如果配置不允许 recompute KV load failure，后面会把这些请求直接结束为 error：

```python
if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
    requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
    self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
```

位置：`scheduler.py:1717`

所以 KV load failure 有两类处理方式：

```text
recompute：
  回退 num_computed_tokens，后续重新计算失败部分。

fail：
  直接将请求标记为 FINISHED_ERROR，并返回错误结束输出。
```

---

## 6. 只遍历本轮真正被调度的请求

`update_from_output()` 的核心循环是：

```python
for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
```

位置：`scheduler.py:1526`

这里的 `num_scheduled_tokens` 来自 `SchedulerOutput`。

这说明：

```text
update_from_output() 只处理本轮真正调度了 token 的请求。
```

如果某个 request 在 `self.running` 中，但本轮因为 PP cadence、DP prefill balancing、encoder budget、Mamba 对齐等原因没有被调度，它不会出现在 `num_scheduled_tokens` 中，因此本轮也没有 Worker 输出要处理。

循环开头还有两个跳过条件：

```python
if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
    continue
request = self.requests.get(req_id)
if request is None or request.is_finished():
    continue
```

位置：`scheduler.py:1528`

第二个条件很重要：

```text
请求可能在 Worker 执行期间被外部 abort；
或者因为 async / pipeline 场景，Scheduler 收到输出时请求已经 finished。
```

这种情况下直接跳过，不再 append 输出。

---

## 7. 从 ModelRunnerOutput 取 sampled token

每个请求通过 `req_id_to_index` 找到自己在 model runner batch 中的行：

```python
req_index = model_runner_output.req_id_to_index[req_id]
generated_token_ids = (
    sampled_token_ids[req_index] if sampled_token_ids else []
)
```

位置：`scheduler.py:1542`

含义：

```text
ModelRunnerOutput 是按 batch row 返回 sampled tokens；
Scheduler 通过 req_id_to_index 把 batch row 映射回 request_id。
```

不同请求类型的输出可能不同：

```text
prefill chunk：
  通常没有 sampled token，只是写 KV。

普通 decode：
  通常返回一个 sampled token。

spec decode：
  可能返回多个 token，包含接受的 draft token 和 target sampled token。

pooling request：
  可能没有 sampled token，但有 pooler_output。
```

---

## 8. Spec decode 接受 / 拒绝修正

如果本轮调度了 spec tokens：

```python
scheduled_spec_token_ids = (
    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
)
```

位置：`scheduler.py:1547`

则根据 Worker 返回的 `generated_token_ids` 判断接受了多少 draft tokens：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

位置：`scheduler.py:1553`

普通自回归场景中：

```text
num_sampled_tokens_per_step = 1
```

所以：

```text
num_accepted = len(generated_token_ids) - 1
```

也就是：

```text
Worker 返回 token 数 = accepted draft tokens + target sampled token
```

如果有 draft token 被拒绝，Scheduler 要回退之前乐观推进的进度：

```python
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected
```

位置：`scheduler.py:1562`

如果 async scheduling 中 output placeholders 也包含 spec tokens，也要同步回退：

```python
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

位置：`scheduler.py:1566`

这说明 spec decode 的状态更新是两阶段：

```text
schedule 阶段：
  先乐观地把 draft tokens 算进 num_computed_tokens。

update_from_output 阶段：
  根据实际接受 / 拒绝数，把 rejected tokens 扣回来。
```

---

## 9. 释放已经安全的 encoder inputs

在处理 token 输出前，Scheduler 会释放已经不再需要的 encoder input：

```python
if request.has_encoder_inputs:
    self._free_encoder_inputs(request)
```

位置：`scheduler.py:1576`

为什么不能在 schedule 阶段马上释放？

```text
schedule 阶段只是发出执行计划；
Worker 还没真正执行完。
只有 update_from_output() 收到结果后，才能确认本轮相关 encoder output 已被使用。
```

`_free_encoder_inputs()` 会遍历 encoder cache 中该请求关联的 input：

```python
cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
    request
)
```

位置：`scheduler.py:1866`

如果是 encoder-decoder 模型，并且已经生成过 decoder token：

```python
if self.is_encoder_decoder and request.num_computed_tokens > 0:
    self.encoder_cache_manager.free_encoder_input(request, input_id)
```

位置：`scheduler.py:1880`

对于普通多模态 placeholder，如果 decoder 进度已经越过该 placeholder，并且不会被 pending draft rejection 回退到里面，也会释放：

```python
elif (
    start_pos + num_tokens
    <= request.num_computed_tokens - request.num_output_placeholders
):
    self.encoder_cache_manager.free_encoder_input(request, input_id)
```

位置：`scheduler.py:1885`

---

## 10. append 输出 token 并检查 stop

核心处理函数是：

```python
new_token_ids, stopped = self._update_request_with_output(
    request, new_token_ids
)
```

位置：`scheduler.py:1590`

内部逻辑：

```python
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)
    stopped = check_stop(request, self.max_model_len)
    if stopped:
        del new_token_ids[num_new:]
        break
```

位置：`scheduler.py:1855`

也就是说：

```text
Scheduler 把 Worker 返回的新 token 逐个 append 到 request；
每 append 一个 token，就检查是否触发 stop；
如果触发 stop，裁剪掉后面多余 token。
```

stop 条件可能包括：

```text
EOS；
stop token；
max_tokens；
max_model_len；
其它 sampling / request 级停止条件。
```

源码注释还说明：

```python
# a request is still being prefilled, we expect the model runner
# to return empty token ids for the request.
```

位置：`scheduler.py:1851`

所以 prefill chunk 中间阶段通常不会产生 EngineCoreOutput token。

---

## 11. pooling request 如何结束

如果没有 `new_token_ids`，但请求是 pooling request，并且 Worker 返回了 pooler output：

```python
elif request.pooling_params and pooler_output is not None:
    # Pooling stops as soon as there is output.
    request.status = RequestStatus.FINISHED_STOPPED
    stopped = True
```

位置：`scheduler.py:1593`

含义是：

```text
pooling 请求不是自回归生成 token；
只要 pooler output 已经返回，请求就可以结束。
```

这种请求的输出会通过 `EngineCoreOutput.pooling_output` 返回给上层。

---

## 12. Structured output grammar 如何推进

如果请求使用结构化输出，生成 token 后还要推进 grammar 状态：

```python
if new_token_ids and self.structured_output_manager.should_advance(request):
    struct_output_request = request.structured_output_request
    assert struct_output_request is not None
    assert struct_output_request.grammar is not None
    if not struct_output_request.grammar.accept_tokens(
        req_id, new_token_ids
    ):
        request.status = RequestStatus.FINISHED_ERROR
        request.resumable = False
        stopped = True
```

位置：`scheduler.py:1598`

含义：

```text
结构化输出不只是 decode 前生成 grammar bitmask；
decode 后也要把实际生成的 token 喂给 grammar，推进 grammar 状态。
```

如果 grammar 拒绝这些 token：

```text
说明输出违反结构约束；
Scheduler 会把请求标记为 FINISHED_ERROR，并停止请求。
```

---

## 13. Routed experts 信息如何处理

如果启用了返回 routed experts，并且 Worker 返回了 routing data：

```python
if model_runner_output.routed_experts is not None:
    re = model_runner_output.routed_experts
    self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
```

位置：`scheduler.py:1507`

后面在每个请求里，根据 prefill / decode / spec decode 的不同情况取 routed experts：

```python
if self.enable_return_routed_experts and routing_data is not None and new_token_ids:
    ...
```

位置：`scheduler.py:1615`

这不是 Scheduler 主调度逻辑的核心，但它体现了一个模式：

```text
Scheduler 会把 Worker 返回的附加信息按 request_id 重新对齐，
再封装进 EngineCoreOutput。
```

---

## 14. 请求停止后如何处理 resumable / streaming

如果 `stopped=True`，Scheduler 会先记录 finish reason：

```python
finish_reason = request.get_finished_reason()
```

位置：`scheduler.py:1659`

然后调用：

```python
finished = self._handle_stopped_request(request)
```

位置：`scheduler.py:1660`

`_handle_stopped_request()` 的逻辑是：

```python
if not request.resumable:
    return True

if request.streaming_queue:
    update = request.streaming_queue.popleft()
    if update is None:
        # Streaming request finished.
        return True
    self._update_request_as_session(request, update)
else:
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    self.num_waiting_for_streaming_input += 1

self._enqueue_waiting_request(request)
return False
```

位置：`scheduler.py:1830`

含义：

```text
非 resumable 请求：
  stop 就是真的 finished。

resumable / streaming 请求：
  如果 streaming_queue 里已经有下一段输入，就更新 session，继续调度；
  如果队列为空，就进入 WAITING_FOR_STREAMING_REQ，等待用户继续输入；
  如果取到 None，说明 streaming 真正结束。
```

所以 stop 不一定等于整个请求结束。

对于 streaming request，可能只是当前 chunk 暂停，后续还会继续。

---

## 15. 请求真正结束后调用 `_free_request()`

如果 `_handle_stopped_request()` 返回 True，说明请求真的结束，可以释放资源：

```python
if finished:
    kv_transfer_params = self._free_request(request)
```

位置：`scheduler.py:1661`

`_free_request()` 核心逻辑是：

```python
self._inflight_prefills.discard(request)
connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
self.encoder_cache_manager.free(request)
request_id = request.request_id
self.finished_req_ids.add(request_id)
...
delay_free_blocks |= connector_delay_free_blocks
if not delay_free_blocks:
    self._free_blocks(request)

return kv_xfer_params
```

位置：`scheduler.py:2046`

它做了几件事：

```text
1. 从 _inflight_prefills 移除；
2. 通知 KV Connector 请求结束；
3. 释放 encoder cache；
4. 记录 finished request id；
5. 如果不需要延迟释放 blocks，则真正 free blocks；
6. 返回 kv_transfer_params，随 EngineCoreOutput 传给上层。
```

---

## 16. `_connector_finished()`：请求结束时可能保存 / 发送 KV

`_free_request()` 中会调用：

```python
connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
```

位置：`scheduler.py:2052`

`_connector_finished()` 会先释放 out-of-window prefix blocks：

```python
self.kv_cache_manager.remove_skipped_blocks(
    request_id=request.request_id,
    total_computed_tokens=request.num_computed_tokens,
)
```

位置：`scheduler.py:2313`

然后取 block ids：

```python
block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
```

位置：`scheduler.py:2318`

再调用 connector：

```python
return self.connector.request_finished_all_groups(request, block_ids)
```

位置：`scheduler.py:2328`

这一步的意义是：

```text
请求结束时，KV Connector 可以决定是否把该请求的 KV 保存 / 发送到外部系统。
```

如果 connector 需要异步发送 KV，就可能返回：

```text
connector_delay_free_blocks = True
```

这时请求虽然 finished，但 KV blocks 不能马上释放。

---

## 17. 为什么 finished 请求可能还留在 `self.requests`

真正删除请求索引发生在：

```python
def _free_blocks(self, request: Request):
    assert request.is_finished()
    self._free_request_blocks(request)
    del self.requests[request.request_id]
```

位置：`scheduler.py:2065`

如果 `_free_request()` 中因为 connector 或其它原因设置了 `delay_free_blocks=True`，就不会立即调用 `_free_blocks()`。

因此：

```text
请求可能已经 finished，
也已经从 running / waiting 队列移除，
但仍然保留在 self.requests 中，
通常是在等待 KV Connector finished_sending，使 `_free_request()` 暂时不能调用 `_free_blocks()`。deferred free 只延迟 block 归还 block pool；一旦执行 `_free_blocks()`，请求索引已经会从 `self.requests` 删除。
```

这和 `01_request_states.md` 中的结论一致：

```text
self.requests 不是 unfinished requests 的简单集合。
```

---

## 18. 从 running / waiting 队列移除 stopped 请求

`update_from_output()` 在循环中不会直接修改 `self.running`，而是先收集：

```python
stopped_running_reqs: set[Request] = set()
stopped_preempted_reqs: set[Request] = set()
```

位置：`scheduler.py:1524`

请求 stop 后，根据 stop 前状态加入不同集合：

```python
if status_before_stop == RequestStatus.RUNNING:
    stopped_running_reqs.add(request)
else:
    stopped_preempted_reqs.add(request)
```

位置：`scheduler.py:1664`

循环结束后批量移除：

```python
if stopped_running_reqs:
    self.running = remove_all(self.running, stopped_running_reqs)
if stopped_preempted_reqs:
    self.waiting.remove_requests(stopped_preempted_reqs)
```

位置：`scheduler.py:1710`

这样做可以避免在遍历过程中修改 running / waiting 队列。

---

## 19. 构造 EngineCoreOutput

如果本轮有可返回内容，Scheduler 会构造：

```python
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
```

位置：`scheduler.py:1688`

触发条件是：

```python
if (
    new_token_ids
    or pooler_output is not None
    or kv_transfer_params
    or stopped
):
```

位置：`scheduler.py:1682`

也就是说：

```text
有新 token；
或有 pooling output；
或有 KV transfer params；
或请求停止；
才会产生 EngineCoreOutput。
```

如果只是中间 prefill chunk，没有 token、没有 pooling、没有 stop，就不会返回 partial prefill output：

```python
# Invariant: EngineCore returns no partial prefill outputs.
assert not prompt_logprobs_tensors
```

位置：`scheduler.py:1706`

---

## 20. logprobs / prompt logprobs / NaN 统计

如果请求需要 sample logprobs：

```python
if (
    request.sampling_params is not None
    and request.sampling_params.num_logprobs is not None
    and logprobs
):
    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

位置：`scheduler.py:1670`

prompt logprobs 通过：

```python
prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
```

位置：`scheduler.py:1680`

如果 Worker 报告 logits 中 NaN 数量：

```python
if num_nans_in_logits is not None and req_id in num_nans_in_logits:
    request.num_nans_in_logits = num_nans_in_logits[req_id]
```

位置：`scheduler.py:1677`

这些信息都会被封装进 `EngineCoreOutput`。

---

## 21. KV Connector finished_recving / finished_sending

处理完请求输出后，如果 Worker 返回了 KV connector output：

```python
if kv_connector_output:
    self._update_from_kv_xfer_finished(kv_connector_output)
```

位置：`scheduler.py:1731`

`_update_from_kv_xfer_finished()` 中，如果有远端 KV load 完成：

```python
for req_id in kv_connector_output.finished_recving or ():
    assert req_id in self.requests
    req = self.requests[req_id]
    if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        self.finished_recving_kv_req_ids.add(req_id)
    else:
        assert RequestStatus.is_finished(req.status)
        self._free_blocks(self.requests[req_id])
```

位置：`scheduler.py:2432`

含义是：

```text
Worker 完成 remote KV load 后，Scheduler 不会立刻把请求 append 到 running；
它只是把 req_id 加入 finished_recving_kv_req_ids。
```

下一轮 `schedule()` 调度 waiting / skipped_waiting 时，会在：

```python
_try_promote_blocked_waiting_request()
```

中看到这个 req_id，然后调用 `_update_waiting_for_remote_kv()`，把请求从 `WAITING_FOR_REMOTE_KVS` 恢复成 `WAITING` 或 `PREEMPTED`。

如果是 KV save / send 完成：

```python
for req_id in kv_connector_output.finished_sending or ():
    self._free_blocks(self.requests[req_id])
```

位置：`scheduler.py:2441`

这会释放之前因 connector async send 而延迟释放的 blocks，并从 `self.requests` 删除请求。

---

## 22. KV cache events 和 connector stats

`update_from_output()` 还会聚合 KV connector stats。

Worker 侧 stats：

```python
kv_connector_stats = (
    kv_connector_output.kv_connector_stats if kv_connector_output else None
)
```

位置：`scheduler.py:1735`

Scheduler 侧 stats：

```python
scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
```

位置：`scheduler.py:1741`

然后聚合。

KV cache events 来自 KV cache manager：

```python
events = self.kv_cache_manager.take_events()
```

位置：`scheduler.py:1752`

也可能来自 connector：

```python
connector_events = self.connector.take_events()
```

位置：`scheduler.py:1756`

最后发布：

```python
batch = KVEventBatch(ts=time.time(), events=events)
self.kv_event_publisher.publish(batch)
```

位置：`scheduler.py:1765`

---

## 23. 按 client_index 返回 EngineCoreOutputs

循环里每个 request 的输出先放入：

```python
outputs[request.client_index].append(...)
```

位置：`scheduler.py:1689`

最后构造：

```python
engine_core_outputs = {
    client_index: EngineCoreOutputs(outputs=outs)
    for client_index, outs in outputs.items()
}
```

位置：`scheduler.py:1769`

如果启用了 `finished_req_ids_dict`，还会给每个 client 附加 finished request ids：

```python
eco.finished_requests = finished_set
```

位置：`scheduler.py:1782`

最后如果有 scheduler stats，会附加到其中一个 `EngineCoreOutputs`：

```python
eco.scheduler_stats = stats
```

位置：`scheduler.py:1800`

最终返回：

```python
return engine_core_outputs
```

位置：`scheduler.py:1802`

---

## 24. 一个完整例子：普通 decode token 返回

假设：

```text
req-a 在 running 中；
本轮 SchedulerOutput 中 num_scheduled_tokens[req-a] = 1；
Worker 返回 sampled token [42]。
```

`update_from_output()` 过程：

```text
1. 从 req_id_to_index 找到 req-a 的 batch row；
2. generated_token_ids = [42]；
3. 没有 scheduled_spec_token_ids，不做 spec 回退；
4. _update_request_with_output(req-a, [42])；
5. append token 42；
6. check_stop 返回 False；
7. 构造 EngineCoreOutput(new_token_ids=[42])；
8. req-a 继续留在 self.running。
```

结果：

```text
客户端收到 token 42；
请求继续等待下一轮 decode。
```

---

## 25. 一个完整例子：达到 stop 后释放

假设：

```text
req-b 本轮返回 EOS token；
check_stop 返回 True；
req-b 不是 resumable 请求。
```

流程：

```text
1. append EOS token；
2. check_stop 设置 stopped=True；
3. finish_reason = request.get_finished_reason()；
4. _handle_stopped_request(req-b) 返回 True；
5. _free_request(req-b)：
   - _inflight_prefills.discard(req-b)
   - _connector_finished(req-b)
   - encoder_cache_manager.free(req-b)
   - finished_req_ids.add(req-b)
   - 如果不延迟释放，_free_blocks(req-b)
6. req-b 加入 stopped_running_reqs；
7. 循环结束后从 self.running 移除；
8. 返回带 finish_reason 的 EngineCoreOutput。
```

状态迁移：

```text
RUNNING
  → FINISHED_STOPPED / FINISHED_LENGTH / 其它 finished status
  → 释放资源
  → 从 running 移除
  → 从 self.requests 删除，或等待 connector async send 完成后删除
```

---

## 26. 一个完整例子：spec decode 部分接受

假设：

```text
本轮 scheduled_spec_decode_tokens[req-c] = 4 个 draft tokens；
num_sampled_tokens_per_step = 1；
Worker 返回 generated_token_ids 长度为 3。
```

则：

```text
num_draft_tokens = 4
num_accepted = len(generated_token_ids) - 1 = 2
num_rejected = 4 - 2 = 2
```

Scheduler 会执行：

```text
request.num_computed_tokens -= 2
如果 num_output_placeholders > 0：
  request.num_output_placeholders -= 2
```

然后再把 Worker 返回的 3 个 token append 到请求输出中。

这说明：

```text
spec decode 中 schedule 阶段先乐观推进，
update_from_output 阶段再按实际接受数修正。
```

---

## 27. 一个完整例子：async KV load 完成

假设：

```text
req-d 当前状态是 WAITING_FOR_REMOTE_KVS；
Worker 完成远端 KV load；
kv_connector_output.finished_recving = {req-d}。
```

`update_from_output()` 中：

```text
1. 调用 _update_from_kv_xfer_finished(kv_connector_output)；
2. 看到 req-d.status == WAITING_FOR_REMOTE_KVS；
3. self.finished_recving_kv_req_ids.add(req-d)。
```

本轮不会马上把 req-d 放入 running。

下一轮 `schedule()` 中：

```text
1. 从 skipped_waiting 选中 req-d；
2. _try_promote_blocked_waiting_request(req-d)；
3. req-d 在 finished_recving_kv_req_ids 中；
4. _update_waiting_for_remote_kv(req-d)；
5. status 恢复为 WAITING 或 PREEMPTED；
6. 后续再按 waiting → running 流程调度。
```

---

## 28. 一个完整例子：streaming request 等下一段输入

假设：

```text
req-e 是 resumable / streaming request；
当前 chunk 生成到了 stop；
streaming_queue 暂时为空。
```

`_handle_stopped_request()` 会：

```text
1. request.status = WAITING_FOR_STREAMING_REQ；
2. num_waiting_for_streaming_input += 1；
3. _enqueue_waiting_request(request)；
4. 因为是 blocked waiting status，进入 skipped_waiting；
5. return False，表示请求还没有真正 finished。
```

所以：

```text
这类 stop 不是最终结束；
请求会等待 add_request() 收到同 request_id 的下一段输入后继续。
```

---

## 29. 容易疑惑的点

### 29.1 `update_from_output()` 会处理所有 running 请求吗？

不会。

它只处理：

```python
scheduler_output.num_scheduled_tokens
```

中的请求。

仍在 `self.running` 里但本轮没有被调度的请求，没有 Worker 输出要处理。

### 29.2 prefill chunk 为什么没有输出？

prefill chunk 的主要作用是计算 KV Cache。

中间 prefill chunk 通常不会 sampled token，因此不会返回 partial prefill output。

源码中也有断言：

```text
EngineCore returns no partial prefill outputs.
```

位置：`scheduler.py:1706`

### 29.3 为什么 schedule 先增加 `num_computed_tokens`，update 再回退？

为了避免 async / PP 场景重复调度。

Scheduler 先乐观认为发出去的 token 已经进入计算进度；如果后续发现 spec token 被拒绝或 KV load 失败，再在 `update_from_output()` 中修正。

### 29.4 请求 stop 后一定马上从 `self.requests` 删除吗？

不一定。

如果 KV Connector 需要异步保存 / 发送 KV，或者 block 释放需要 deferred free，请求可能 finished 但仍暂留在 `self.requests`。

真正删除发生在：

```python
del self.requests[request.request_id]
```

位置：`scheduler.py:2068`

### 29.5 `finished_recving` 后为什么不立刻进入 running？

`finished_recving` 只表示远端 KV load 完成。

Scheduler 会先记录：

```python
finished_recving_kv_req_ids.add(req_id)
```

下一轮 schedule 再通过 `_try_promote_blocked_waiting_request()` 把请求从 `WAITING_FOR_REMOTE_KVS` 恢复为 `WAITING` / `PREEMPTED`。

### 29.6 stop 是否一定表示请求结束？

不一定。

对于 resumable / streaming 请求，stop 可能只是当前 input chunk 结束。

如果还有下一段输入，Scheduler 会更新 session；如果没有，就进入 `WAITING_FOR_STREAMING_REQ`。

---

## 30. 从“回答问题”的角度总结

如果要问：

```text
Worker 执行完后，Scheduler 如何更新请求状态、释放 block、返回输出？
```

Scheduler 的回答是：

```text
通过 update_from_output() 把 SchedulerOutput 和 ModelRunnerOutput 对齐。

它会处理 Worker 返回的 sampled tokens、pooling outputs、logprobs、KV connector outputs，
修正 spec decode 中被拒绝的 draft tokens，
释放已经安全的 encoder inputs 和 deferred blocks，
append 输出 token 并检查 stop，
对真正 finished 的请求调用 _free_request()，
最后构造按 client_index 分组的 EngineCoreOutputs 返回给上层。
```

状态闭环是：

```text
schedule()
  → _update_after_schedule() 乐观推进 num_computed_tokens
  → Worker forward
  → update_from_output()
  → append token / spec 回退 / grammar 推进 / KV transfer 状态更新
  → stop 则 _handle_stopped_request()
  → finished 则 _free_request()
  → 返回 EngineCoreOutputs
```

---

## 31. 最关键流程图

```text
update_from_output()
  │
  ├─ 读取 ModelRunnerOutput
  │    ├─ sampled_token_ids
  │    ├─ logprobs
  │    ├─ pooler_output
  │    └─ kv_connector_output
  │
  ├─ deferred free
  │    └─ _drain_deferred_frees()
  │
  ├─ KV load failure
  │    └─ _handle_invalid_blocks()
  │
  ├─ for req_id in scheduler_output.num_scheduled_tokens
  │    ├─ 跳过已 finished / aborted 请求
  │    ├─ 取 generated_token_ids
  │    ├─ spec decode accepted / rejected 修正
  │    ├─ _free_encoder_inputs()
  │    ├─ _update_request_with_output()
  │    │    ├─ append_output_token_ids
  │    │    └─ check_stop
  │    ├─ pooling output 结束 pooling request
  │    ├─ grammar.accept_tokens()
  │    ├─ stopped ?
  │    │    ├─ _handle_stopped_request()
  │    │    └─ finished ? _free_request()
  │    ├─ 提取 logprobs / prompt logprobs / routed experts
  │    └─ 构造 EngineCoreOutput
  │
  ├─ 批量移除 stopped running / waiting 请求
  ├─ 处理 failed KV load error 输出
  ├─ _update_from_kv_xfer_finished()
  │    ├─ finished_recving → finished_recving_kv_req_ids
  │    └─ finished_sending → _free_blocks()
  ├─ 收集 KV events / connector stats / perf stats
  └─ 返回 dict[client_index, EngineCoreOutputs]
```

---

## 32. 最关键的判断公式

```text
本轮处理哪些请求：
  for req_id in scheduler_output.num_scheduled_tokens

取 Worker 输出：
  req_index = model_runner_output.req_id_to_index[req_id]
  generated_token_ids = sampled_token_ids[req_index]

spec decode：
  num_accepted = max(len(generated_token_ids) - num_sampled_tokens_per_step, 0)
  num_rejected = len(scheduled_spec_token_ids) - num_accepted
  request.num_computed_tokens -= num_rejected
  request.num_output_placeholders -= num_rejected  # if > 0

append token：
  request.append_output_token_ids(output_token_id)
  stopped = check_stop(request, max_model_len)

streaming / resumable：
  stopped and resumable and no next input
    → WAITING_FOR_STREAMING_REQ
    → skipped_waiting

真正 finished：
  finished = _handle_stopped_request(request)
  if finished:
      kv_transfer_params = _free_request(request)

删除请求：
  _free_blocks(request)
    → _free_request_blocks(request)
    → del self.requests[request_id]

远端 KV load 完成：
  kv_connector_output.finished_recving
    → finished_recving_kv_req_ids.add(req_id)
    → 下一轮 schedule 恢复 WAITING_FOR_REMOTE_KVS

远端 KV send 完成：
  kv_connector_output.finished_sending
    → _free_blocks(request)
```

---

## 33. 和前面问题的关系

前几篇讲的是 Scheduler 如何把请求发出去：

```text
01_request_states.md：请求在哪些队列和状态中
02_token_budget.md：本轮最多能调度多少 token
03_running_decode_prefill.md：running 请求如何继续推进
04_waiting_to_running.md：waiting 请求如何进入 running
05_prefix_and_external_kv_hits.md：cache 命中如何减少计算
06_kv_block_allocation_and_preemption.md：KV block 如何分配与抢占
07_auxiliary_scheduling_features.md：encoder / grammar / spec / LoRA 等如何插入调度
```

本篇讲的是 Scheduler 如何把结果收回来：

```text
Worker 执行 SchedulerOutput 后，
Scheduler 如何处理真实输出、修正乐观状态、推进请求状态机、释放资源、返回 EngineCoreOutputs。
```

至此，一个请求的主生命周期闭环就是：

```text
add_request
  → waiting
  → schedule
  → running
  → SchedulerOutput
  → Worker / ModelRunner
  → ModelRunnerOutput
  → update_from_output
  → 继续 running / 回到 waiting / finished free
```
