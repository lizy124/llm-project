# 10. KVPool 端到端如何贯穿 Scheduler 和 Worker？

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/utils.py`

本文用于从 KVPool / 外部 KV Cache 视角串起完整端到端链路：查询命中、分配 block、load KV、执行 forward、save KV、完成通知、失败回退和资源释放。

这里的“KVPool”不是 Scheduler 里的一个固定类名，而是外部 KV cache / KV connector 这一类能力的心智模型：Scheduler 侧 connector 负责查外部命中、规划 load / save、维护请求生命周期；Worker 侧 connector 负责真正把 KV 在外部系统和本地 paged KV cache 之间搬运。

---

## 1. 一句话回答

KVPool 端到端贯穿 vLLM V1 的方式是：

```text
Scheduler 侧：
  本地 prefix cache 查询
  → 外部 KVPool 命中查询
  → 分配本地 KV blocks
  → 构造 kv_connector_metadata
  → 生成 SchedulerOutput

Worker 侧：
  接收 SchedulerOutput
  → bind kv_connector_metadata
  → start_load_kv / attention 中 wait_for_layer_load
  → forward 时 save_kv_layer
  → wait_for_save / get_finished / invalid_block_ids
  → 生成 ModelRunnerOutput.kv_connector_output

Scheduler 回收：
  update_from_output()
  → 处理 invalid blocks
  → 处理 finished_recving / finished_sending
  → 恢复等待请求或释放 block
```

所以最核心的桥梁只有两个对象：

```text
SchedulerOutput.kv_connector_metadata
  Scheduler → Worker：告诉 Worker 本轮要 load / save 哪些 KV。

ModelRunnerOutput.kv_connector_output
  Worker → Scheduler：告诉 Scheduler 哪些 load / save 完成，哪些 block load 失败。
```

---

## 2. 端到端总览图

把一次请求从进入系统到结束后的外部 KV 保存串起来，可以看成下面这条链路：

```text
new request
  → Scheduler waiting 阶段
  → KVCacheManager 查询本地 prefix cache
  → connector.get_num_new_matched_tokens() 查询 KVPool 外部命中
  → 计算 num_computed_tokens = local hit + external hit
  → KVCacheManager.allocate_slots() 分配本地 blocks
  → connector.update_state_after_alloc() 记录 load 计划
  → SchedulerOutput.kv_connector_metadata
  → Executor / Worker / GPUModelRunner.execute_model()
  → Worker connector bind metadata
  → legacy context 或新版 ActiveKVConnector.pre_forward() 调 start_load_kv()
  → attention/forward 使用本地 paged KV cache
  → save_kv_layer() 可异步保存新 KV
  → wait_for_save()
  → get_finished() / get_block_ids_with_load_errors()
  → ModelRunnerOutput.kv_connector_output
  → Scheduler.update_from_output()
  → _handle_invalid_blocks() / _update_from_kv_xfer_finished()
  → 请求继续调度、重算、完成或释放 block
```

这条链路的关键点是：

```text
KVPool 命中发生在 Scheduler 决策阶段；
KVPool 搬运发生在 Worker 执行阶段；
KVPool 完成 / 失败结果回到 Scheduler update 阶段。
```

---

## 3. 参与端到端链路的核心对象

### 3.1 Request

`Request` 是请求状态的载体。KVPool 链路主要读写：

```text
request.request_id
request.num_tokens
request.num_computed_tokens
request.block_hashes
request.status
request.num_preemptions
```

其中 `num_computed_tokens` 是最关键的字段。

它表示：

```text
当前请求已经可以认为“算过 / 可复用”的 token 数。
```

本地 prefix cache 命中、外部 KVPool 命中、异步 load 完成、invalid block 回退，最终都会落到这个字段上。

### 3.2 KVCacheManager / BlockPool

Scheduler 不直接操作 GPU KV tensor，而是通过 KVCacheManager 管理 block 账本：

```text
get_computed_blocks()
allocate_slots()
cache_blocks()
free()
evict_blocks()
get_block_ids()
```

BlockPool / KVCacheManager 负责的是：

```text
本地 KV block 是否已经存在；
哪些 block 可复用；
哪些 block 要新分配；
哪些 block 要进入 prefix cache；
哪些 block 要释放或驱逐。
```

### 3.3 KVConnectorBase_V1

KV connector 在 Scheduler 和 Worker 两侧都有角色。

Scheduler 侧基础接口中，`request_finished()` 属于 `KVConnectorBase_V1`；`request_finished_all_groups()` 属于支持 HMA 的 `SupportsHMA` 扩展，Scheduler 会按 connector 类型选择调用。

Scheduler 侧基础接口：

```text
get_num_new_matched_tokens()
update_state_after_alloc()
build_connector_meta()
request_finished() / SupportsHMA.request_finished_all_groups()
update_connector_output()
has_pending_push_work()
```

部分 connector 还会提供 `reset_cache()` 这类可选方法，Scheduler 侧控制接口会按需动态调用。

Worker 侧主要接口：

```text
bind_connector_metadata()
start_load_kv()
wait_for_layer_load()
save_kv_layer()
wait_for_save()
get_finished()
get_block_ids_with_load_errors()
build_connector_worker_meta()
clear_connector_metadata()
```

因此 connector 本身就是 KVPool 端到端协议的抽象层。

### 3.4 SchedulerOutput

`SchedulerOutput` 是 Scheduler 发给 Worker 的本轮执行计划。

和 KVPool 直接相关的字段是：

```text
num_scheduled_tokens
finished_req_ids
kv_connector_metadata
new_block_ids_to_zero
```

定义位置：`vllm/v1/core/sched/output.py:181`

其中：

```text
kv_connector_metadata：本轮 KV connector 的不透明元数据；
finished_req_ids：告诉 Worker 哪些请求已经结束，可用于 connector 完成通知；
new_block_ids_to_zero：Worker 使用新 block 前需要清零。
```

### 3.5 KVConnectorOutput / ModelRunnerOutput

Worker 回 Scheduler 的 KVPool 结果放在：

```text
ModelRunnerOutput.kv_connector_output
```

`KVConnectorOutput` 字段包括：

```text
finished_sending
finished_recving
kv_connector_stats
kv_cache_events
kv_connector_worker_meta
invalid_block_ids
expected_finished_count
```

定义位置：`vllm/v1/outputs.py:196`

其中最影响调度的是：

```text
finished_recving：外部 KV load 完成，请求可恢复调度；
finished_sending：外部 KV save / send 完成，可以释放延迟保留的 blocks；
invalid_block_ids：外部 KV load 失败，需要 fail 或回退重算。
```

---

## 4. Scheduler waiting 阶段如何查询 KVPool 命中

KVPool 查询发生在 waiting 请求第一次进入调度时。

主逻辑在 `Scheduler.schedule()` 的 waiting 阶段：

```text
1. 先处理 skipped_waiting / waiting 队列；
2. 如果请求还没计算过 token，则先查本地 prefix cache；
3. 如果配置了 connector，再查外部 KVPool 命中；
4. local hit + external hit 合成 num_computed_tokens；
5. 再决定本轮要不要 forward 以及 forward 多少 token。
```

关键代码位置：`vllm/v1/core/sched/scheduler.py:671`

### 4.1 先查本地 prefix cache

当 `request.num_computed_tokens == 0` 时，Scheduler 先查本地缓存：

```text
KVCacheManager.get_computed_blocks(request)
  → new_computed_blocks
  → num_new_local_computed_tokens
```

位置：`vllm/v1/core/sched/scheduler.py:710`

这一步回答：

```text
这个请求有多少 token 已经在本地 KV cache 中了？
```

### 4.2 再查外部 KVPool

如果有 connector：

```python
ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
    request, num_new_local_computed_tokens
)
```

位置：`vllm/v1/core/sched/scheduler.py:723` 到 `vllm/v1/core/sched/scheduler.py:728`

这个接口回答两个问题：

```text
ext_tokens：外部 KVPool 还能额外命中多少 token；
load_kv_async：这些外部 KV 是否要异步 load。
```

注意这里传入的是 `num_new_local_computed_tokens`。

含义是：

```text
外部 KVPool 只需要补本地没有命中的部分，避免重复搬运本地已有 KV。
```

### 4.3 外部命中为 None 时为什么跳过调度

如果 connector 返回：

```text
ext_tokens is None
```

Scheduler 会把请求放回 skipped waiting：

```text
request cannot be scheduled because connector couldn't determine matched tokens
```

位置：`vllm/v1/core/sched/scheduler.py:729`

这表示：

```text
KVPool 查询本身还没准备好，Scheduler 不贸然分配和执行，下一轮再试。
```

### 4.4 计算总 computed tokens

Scheduler 合成：

```text
num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
```

位置：`vllm/v1/core/sched/scheduler.py:745`

从这一步开始，KVPool 命中就变成了调度可理解的普通 computed token 数。

---

## 5. KVPool 命中如何减少本轮 forward token

外部命中不直接让 Worker 少跑 token，而是通过 Scheduler 改变：

```text
num_computed_tokens
num_new_tokens
num_scheduled_tokens
```

正常同步 load / 可立即执行路径：

```text
num_new_tokens = request.num_tokens - num_computed_tokens
num_new_tokens = min(num_new_tokens, token_budget)
```

位置：`vllm/v1/core/sched/scheduler.py:792` 到 `vllm/v1/core/sched/scheduler.py:812`

因此：

```text
外部 KVPool 命中越多，num_computed_tokens 越大；
num_computed_tokens 越大，本轮要 forward 的 num_new_tokens 越少。
```

举例：

```text
prompt 长度 = 1000
本地 prefix cache 命中 = 200
KVPool 外部命中 = 600
num_computed_tokens = 800
本轮 prefill 只需要从 token 800 开始补算剩余 200 个 token
```

最后 Scheduler 会写入：

```text
num_scheduled_tokens[request_id] = num_new_tokens
request.num_computed_tokens = num_computed_tokens
request.status = RUNNING
```

位置：`vllm/v1/core/sched/scheduler.py:954` 到 `vllm/v1/core/sched/scheduler.py:960`

---

## 6. block 分配如何同时服务本地 prefix 和外部 KV

确定命中后，Scheduler 通过 `allocate_slots()` 分配本地 blocks：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    full_sequence_must_fit=self.scheduler_reserve_full_isl,
    reserved_blocks=reserved_blocks,
    has_scheduled_reqs=bool(self.running),
    ...
)
```

位置：`vllm/v1/core/sched/scheduler.py:874` 到 `vllm/v1/core/sched/scheduler.py:886`

这里有几个关键参数：

```text
num_new_computed_tokens：本地 prefix cache 命中的 token 数；
new_computed_blocks：本地已经命中的 blocks；
num_external_computed_tokens：外部 KVPool 命中的 token 数；
delay_cache_blocks：异步 load 时先分配 block，但暂不把它们 cache 为可复用。
```

可以理解为：

```text
本地命中：直接复用已有 block；
外部命中：先分配本地目标 block，Worker 后面把外部 KV load 进去；
未命中 token：本轮 forward 计算并写入后续 block。
```

---

## 7. connector.update_state_after_alloc() 做什么

分配完 blocks 后，Scheduler 会通知 connector：

```python
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

位置：`vllm/v1/core/sched/scheduler.py:901` 到 `vllm/v1/core/sched/scheduler.py:906`

这一步的含义是：

```text
Scheduler 已经为这个请求分配好了本地目标 block；
connector 可以记录“把外部 KV load 到哪些本地 block”这件事；
后续 build_connector_meta() 会把这些计划打包进 SchedulerOutput。
```

也就是说：

```text
get_num_new_matched_tokens() 只回答“命中了多少”；
update_state_after_alloc() 才知道“这些命中要落到哪些本地 block”。
```

---

## 8. 同步 load 与异步 load 的调度差异

KVPool load 有两类路径：

```text
同步 / 本轮可执行 load：
  load_kv_async = False
  → 分配 blocks
  → 请求进入 RUNNING
  → 本轮 forward 剩余 token

异步 load：
  load_kv_async = True
  → 分配 blocks
  → num_new_tokens = 0
  → 请求进入 WAITING_FOR_REMOTE_KVS
  → 本轮不 forward，等待 finished_recving
```

### 8.1 同步 load 路径

如果 `load_kv_async = False`，Scheduler 会计算真正要 forward 的 token 数：

```text
num_new_tokens = request.num_tokens - num_computed_tokens
```

然后请求进入 running：

```text
self.running.append(request)
request.status = RequestStatus.RUNNING
```

位置：`vllm/v1/core/sched/scheduler.py:940` 到 `vllm/v1/core/sched/scheduler.py:960`

Worker 在本轮 forward 前触发 connector load，attention 层使用已经 load 好的 KV。

### 8.2 异步 load 路径

如果 `load_kv_async = True`：

```text
num_new_tokens = 0
allocate_slots(..., delay_cache_blocks=True)
request.status = WAITING_FOR_REMOTE_KVS
step_skipped_waiting.prepend_request(request)
request.num_computed_tokens = num_computed_tokens
```

位置：`vllm/v1/core/sched/scheduler.py:782` 到 `vllm/v1/core/sched/scheduler.py:785`，以及 `vllm/v1/core/sched/scheduler.py:918` 到 `vllm/v1/core/sched/scheduler.py:938`

这表示：

```text
Scheduler 只启动外部 KV load，不让这个请求本轮 forward。
```

这类请求之后会从 skipped waiting 中反复被检查，直到 Worker 回传 `finished_recving`。

---

## 9. SchedulerOutput.kv_connector_metadata 如何生成

Scheduler 完成本轮调度后构造 `SchedulerOutput`。

位置：`vllm/v1/core/sched/scheduler.py:1057`

随后如果有 connector：

```python
meta = self._build_kv_connector_meta(self.connector, scheduler_output)
scheduler_output.kv_connector_metadata = meta
```

位置：`vllm/v1/core/sched/scheduler.py:1080`

而 `_build_kv_connector_meta()` 本质调用：

```python
connector.build_connector_meta(scheduler_output)
```

位置：`vllm/v1/core/sched/scheduler.py:1100`

这一步把 Scheduler 侧累计的 load / save / preemption / request lifecycle 状态封装成一个不透明 metadata。

重要点：

```text
Scheduler 不要求自己理解每个 connector 的传输细节；
Worker 也不重新做 KVPool 查询；
两边通过 connector 自己定义的 KVConnectorMetadata 对齐。
```

---

## 10. SchedulerOutput 如何进入 Worker

EngineCore 侧主链路是：

```text
EngineCore.step()
  → scheduler.schedule()
  → model_executor.execute_model(scheduler_output)
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
```

Worker 侧拿到的核心输入仍然是同一个 `SchedulerOutput`。

KVPool 信息不单独传一条 RPC，而是挂在：

```text
scheduler_output.kv_connector_metadata
```

这保证了：

```text
调度计划、block 分配、KV transfer metadata 在同一轮 step 中保持一致。
```

---

## 11. Worker 如何使用 kv_connector_metadata

在 `GPUModelRunner.execute_model()` 开始阶段，如果存在 KV transfer group：

```python
kv_connector_metadata = scheduler_output.kv_connector_metadata
get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4075`

这一步用于处理：

```text
被抢占请求；
即将被覆盖 / 释放但 connector 仍需要保存的 blocks；
connector 需要在真正 forward 前完成的清理或保护动作。
```

然后 ModelRunner 更新 batch 状态：

```text
_update_states(scheduler_output)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4085`

这会把 Scheduler 分配的 block ids 写入 Worker 侧 InputBatch / block table。

---

## 12. 0-token step 为什么仍然可能触发 KVPool

如果本轮没有 scheduled tokens：

```python
if not num_scheduled_tokens:
    if not has_kv_transfer_group():
        return EMPTY_MODEL_RUNNER_OUTPUT
    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4096`

含义是：

```text
没有 forward token，不代表没有 KVPool 工作。
```

典型场景：

```text
- 异步 load 只需要启动 / 轮询传输；
- 请求已经 finished，但 save 还在进行；
- connector 需要返回 finished_sending / finished_recving；
- 上一轮异步任务完成，需要把状态带回 Scheduler。
```

`kv_connector_no_forward()` 会：

```text
set_forward_context(None)
→ bind metadata
→ start_load_kv()
→ get_finished()
→ get_block_ids_with_load_errors()
→ 返回只有 kv_connector_output 的 ModelRunnerOutput
```

位置：`vllm/v1/worker/kv_connector_model_runner_mixin.py:36`

---

## 13. forward 前后 KVPool 如何包进执行上下文

legacy GPU model runner 路径中，KV connector 被包在 forward context 里：

```python
with (
    set_forward_context(...),
    self.maybe_get_kv_connector_output(
        scheduler_output,
        defer_finalize=defer_kv_connector_finalize,
    ) as kv_connector_output,
):
    model_output = self._model_forward(...)
```

位置：`vllm/v1/worker/gpu_model_runner.py:4302`

新版 GPU model runner 路径中，`v1/worker/gpu/model_runner.py` 在设置 forward context 后调用 `self.kv_connector.pre_forward(scheduler_output)`，在 `sample_tokens()` / `pool()` 阶段调用 `self.kv_connector.post_forward(finished_req_ids)` 收集 `KVConnectorOutput`。

这说明：

```text
KVPool load / save 不是 forward 之后单独补的一步，
而是嵌入 forward 上下文或其前后钩子，让 attention 层可以按 layer 与 KV transfer 协同。
```

---

## 14. KVConnectorModelRunnerMixin 的生命周期

`maybe_get_kv_connector_output()` 最终进入 `_get_kv_connector_output()`。

核心顺序是：

```text
1. 创建 KVConnectorOutput；
2. get_kv_transfer_group()；
3. bind_connector_metadata(scheduler_output.kv_connector_metadata)；
4. start_load_kv(get_forward_context())；
5. yield 给 forward；
6. wait_for_save()；
7. get_finished(scheduler_output.finished_req_ids)；
8. get_block_ids_with_load_errors()；
9. get stats / events / worker_meta；
10. clear_connector_metadata()。
```

这是普通非 speculative 路径的顺序。开启 speculative decoding 时，`defer_finalize=True` 会让 `_get_kv_connector_output()` 跳过 `wait_for_save()` 和 `clear_connector_metadata()`，但仍然收集 `get_finished()`、invalid blocks、stats / events / worker_meta；后续 `sample_tokens()` 中的 `finalize_kv_connector()` 再执行 wait 和 clear。

位置：`vllm/v1/worker/kv_connector_model_runner_mixin.py:77`

其中：

```text
start_load_kv：把外部 KV load 到本地 paged KV buffer；
save_kv_layer：attention layer 内部把本地 KV 保存到外部系统；
wait_for_save：保证异步 save 完成到安全点；
get_finished：把异步 send / recv 完成的 request id 带回 Scheduler；
get_block_ids_with_load_errors：把失败 block id 带回 Scheduler。
```

---

## 15. KVPool load 到本地 block 的真实语义

Worker 侧 connector 的抽象接口说明了 load 的语义：

```text
start_load_kv(forward_context)
  从 connector / 外部 KVPool 开始加载 KV 到 vLLM paged KV buffer。

wait_for_layer_load(layer_name)
  attention layer 内等待某一层 KV load 完成。
```

位置：`vllm/distributed/kv_transfer/kv_connector/v1/base.py:293`

这说明外部 KVPool 的 load 目标不是一个独立 cache，而是 Scheduler 已经分配好的本地 paged KV blocks。

因此链路是：

```text
Scheduler 分配目标 block ids
  → connector metadata 携带 load 计划
  → Worker connector start_load_kv()
  → KV 被写入本地 paged KV cache
  → attention backend 按 block table / slot mapping 使用这些 KV
```

---

## 16. 请求结束后如何 save 到 KVPool

请求停止时，Scheduler 会释放请求，但释放前先通知 connector。

主入口：

```text
Scheduler.update_from_output()
  → 请求 stop
  → _free_request()
  → _connector_finished()
```

`_free_request()` 中：

```python
connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
...
delay_free_blocks |= connector_delay_free_blocks
if not delay_free_blocks:
    self._free_blocks(request)
```

位置：`vllm/v1/core/sched/scheduler.py:2046`

`_connector_finished()` 会：

```text
1. remove_skipped_blocks()
2. get_block_ids(request_id)
3. connector.request_finished() 或 request_finished_all_groups()
```

位置：`vllm/v1/core/sched/scheduler.py:2299`

connector 的 `request_finished*()` 可以返回：

```text
True：请求的 KV 还在异步保存 / 发送，Scheduler 暂时不能释放 blocks；
False：不需要延迟释放，Scheduler 可以立即 free blocks；
kv_transfer_params：可选地带回给请求输出。
```

这就是“请求已经结束，但 block 还不能马上释放”的来源。

---

## 17. finished_sending 如何让 Scheduler 最终释放 block

如果 connector 在 `request_finished()` 返回 True，Scheduler 会保留 request 和 blocks。

之后 Worker 侧 connector 在某一轮返回：

```text
KVConnectorOutput.finished_sending = {req_id, ...}
```

Scheduler 在 `_update_from_kv_xfer_finished()` 中处理：

```python
for req_id in kv_connector_output.finished_sending or ():
    self._free_blocks(self.requests[req_id])
```

位置：`vllm/v1/core/sched/scheduler.py:2441`

因此 `finished_sending` 的语义是：

```text
这个请求的外部 KV save / send 已经完成，
本地 blocks 不再需要为 connector 保留，
Scheduler 可以真正释放 request blocks。
```

---

## 18. finished_recving 如何让异步 load 请求恢复调度

异步 load 请求会处于：

```text
RequestStatus.WAITING_FOR_REMOTE_KVS
```

位置：`vllm/v1/core/sched/scheduler.py:920`

Worker 侧 load 完成后返回：

```text
KVConnectorOutput.finished_recving = {req_id, ...}
```

Scheduler 在 `_update_from_kv_xfer_finished()` 中：

```python
if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    self.finished_recving_kv_req_ids.add(req_id)
```

位置：`vllm/v1/core/sched/scheduler.py:2432`

下一轮 waiting 阶段遇到这个请求时：

```text
_try_promote_blocked_waiting_request()
  → _update_waiting_for_remote_kv()
  → status 改回 WAITING 或 PREEMPTED
```

位置：`vllm/v1/core/sched/scheduler.py:2384`

`_update_waiting_for_remote_kv()` 会把已经 load 好的 blocks cache 起来：

```python
self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
```

位置：`vllm/v1/core/sched/scheduler.py:2375`

因此异步 load 的完整恢复链路是：

```text
Scheduler 发起 async load
  → request.status = WAITING_FOR_REMOTE_KVS
  → Worker connector 完成 load
  → ModelRunnerOutput.kv_connector_output.finished_recving
  → Scheduler.finished_recving_kv_req_ids.add(req_id)
  → 下一轮 promote blocked waiting request
  → cache_blocks()
  → 请求回到 WAITING / PREEMPTED
  → 再次进入正常调度
```

---

## 19. full hit 为什么要回退最后一个 token

本地 prefix cache 查询时会预先把最大命中长度限制为 `request.num_tokens - 1`，避免直接把整个 prompt 都当成本地 cache hit。

位置：`vllm/v1/core/kv_cache_manager.py:221` 到 `vllm/v1/core/kv_cache_manager.py:227`

异步外部 KVPool load 完成后，如果出现整个 prompt 都来自外部 KV：

```text
request.num_computed_tokens == request.num_tokens
```

`_update_waiting_for_remote_kv()` 中会做：

```python
if request.num_computed_tokens == request.num_tokens:
    request.num_computed_tokens = request.num_tokens - 1
```

位置：`vllm/v1/core/sched/scheduler.py:2377`

原因是：

```text
即使 prompt KV 全部命中，生成下一个 token 仍需要一次模型执行来得到 logits / sample。
```

如果不回退最后一个 token，就会出现：

```text
没有 token 需要 forward，但又需要采样下一个 token。
```

所以 full hit 的实际执行模型是：

```text
异步 KVPool load 命中完整 prompt
  → 恢复调度时把 num_computed_tokens 回退到 prompt_len - 1
  → 后续 forward 最后一个 prompt token
  → 产生 logits
  → sample 第一个 output token
```

---

## 20. invalid_block_ids 如何触发回退或失败

Worker connector 如果发现外部 KV load 失败，会返回：

```text
KVConnectorOutput.invalid_block_ids
```

位置：`vllm/v1/outputs.py:203`

Scheduler 在 `update_from_output()` 最开始处理：

```python
if kv_connector_output and kv_connector_output.invalid_block_ids:
    failed_kv_load_req_ids = self._handle_invalid_blocks(
        kv_connector_output.invalid_block_ids,
        num_scheduled_tokens,
    )
```

位置：`vllm/v1/core/sched/scheduler.py:1491`

`_handle_invalid_blocks()` 会分别处理：

```text
async load 请求：在 skipped_waiting / WAITING_FOR_REMOTE_KVS 中找受影响请求；
sync load 请求：在 running 中找受影响请求。
```

位置：`vllm/v1/core/sched/scheduler.py:2549`

### 20.1 如何找到受影响请求

`_update_requests_with_invalid_blocks()` 会扫描请求的 block ids：

```text
如果某个 computed block 在 invalid_block_ids 中，
就把 request.num_computed_tokens 截断到这个 block 之前。
```

核心逻辑：

```python
request.num_computed_tokens = idx * self.block_size
```

位置：`vllm/v1/core/sched/scheduler.py:2523`

这意味着：

```text
失败 block 以及它之后的 token 都要重算。
```

### 20.2 failure_policy=fail 与 recompute

Scheduler 初始化时读取：

```text
kv_transfer_config.kv_load_failure_policy
```

如果策略是 `recompute`：

```text
调整 num_computed_tokens，让后续调度从最长有效前缀之后重算。
```

如果策略不是 recompute：

```text
把相关请求标记为失败。
```

位置：`vllm/v1/core/sched/scheduler.py:2558`

---

## 21. 多 Worker / 多 rank 下 KVConnectorOutput 如何聚合

多进程 / 多 rank 场景下，不是每个 Worker 返回的 KV connector 输出都能直接给 Scheduler。

`KVOutputAggregator` 会聚合所有 worker 的：

```text
finished_sending
finished_recving
kv_connector_stats
kv_connector_worker_meta
kv_cache_events
invalid_block_ids
```

位置：`vllm/distributed/kv_transfer/kv_connector/utils.py:71`

关键逻辑是：

```text
finished_sending / finished_recving：
  根据 expected_finished_count 做计数，所有需要的 worker 都完成后才对 Scheduler 暴露。

invalid_block_ids：
  对所有 worker 的 invalid_block_ids 求 union。
```

位置：`vllm/distributed/kv_transfer/kv_connector/utils.py:111` 和 `vllm/distributed/kv_transfer/kv_connector/utils.py:154`

因此 Scheduler 看到的是聚合后的 connector output，而不是单个 rank 的局部状态。

---

## 22. KVPool 与 prefix cache 的关系

本地 prefix cache 和外部 KVPool 不是互斥关系，而是串联关系：

```text
先查本地 prefix cache；
再用本地命中长度作为起点查外部 KVPool；
最后合并成 num_computed_tokens。
```

可以理解为：

```text
本地 prefix cache：避免重复计算和重复远端传输；
外部 KVPool：补充本地没有的历史 KV；
BlockPool：为外部 KV 和新计算 KV 提供本地物理落点。
```

一个典型状态：

```text
tokens:       [0 ........ 199][200 ........ 799][800 ........ 999]
来源:         local prefix     external KVPool  local forward
Scheduler:    local hit=200    external hit=600 num_new_tokens=200
Worker:       复用本地 block    load 到本地 block  forward 写新 block
```

---

## 23. KVPool 与 Worker block table / slot mapping 的关系

KVPool 不直接改变 attention 的调用方式。

它只改变：

```text
哪些本地 block 已经被视为 computed；
哪些 block 需要从外部 load；
哪些 token 本轮还要 forward。
```

Worker 侧最终仍然使用统一机制：

```text
InputBatch.block_table
slot mapping
attention metadata
paged KV cache tensor
```

也就是说：

```text
不管 KV 来自本地 prefix cache、外部 KVPool，还是本轮 forward，
attention backend 最后看到的都是本地 paged KV cache + block table + slot mapping。
```

这也是为什么外部 KV 必须先落到 Scheduler 分配的本地 blocks。

---

## 24. 请求结束后的资源释放有几种情况

KVPool 会让资源释放多出“延迟释放”分支。

### 24.1 没有 connector 或不需要保存

```text
request finished
  → _free_request()
  → _connector_finished() 返回 False
  → _free_blocks()
  → blocks 回到 BlockPool
```

### 24.2 connector 需要异步 save

```text
request finished
  → connector.request_finished() 返回 True
  → Scheduler 不立即 free blocks
  → Worker connector 继续 save / send
  → finished_sending 回来
  → Scheduler._free_blocks()
```

### 24.3 async KV load 请求被 abort

如果请求还在 `WAITING_FOR_REMOTE_KVS` 就被终止：

```text
如果还没 finished_recving，则 delay_free_blocks=True；
等 connector 后续回 finished_recving / cleanup 后再释放。
```

位置：`vllm/v1/core/sched/scheduler.py:2033`

### 24.4 多 inflight batch 下 deferred free

如果启用 async scheduling / PP 等多 in-flight batch，并且当前是 KV consumer：

```text
self.defer_block_free = True
```

位置：`vllm/v1/core/sched/scheduler.py:149`

这会让已经逻辑释放的 blocks 等到对应 GPU step 完成后再真正回到 BlockPool，避免：

```text
某个 in-flight step 仍在写旧 request 的 KV，
另一个 async load 又把同一 block 分配给新 request。
```

---

## 25. 六个典型场景串起来看

### 25.1 场景一：无 KVPool 命中，全部本地 prefill

```text
get_computed_blocks() = 0
connector.get_num_new_matched_tokens() = 0
num_computed_tokens = 0
num_new_tokens = prompt_len
allocate_slots()
request RUNNING
Worker forward 全部 prompt
请求结束后 connector 可选择 save KV
```

### 25.2 场景二：KVPool 部分命中，本地补算剩余 token

```text
local hit = 0
external hit = 800
prompt_len = 1000
num_computed_tokens = 800
num_new_tokens = 200
allocate_slots(num_external_computed_tokens=800)
Worker load 前 800 token 的 KV
Worker forward 后 200 token
sample / update
```

### 25.3 场景三：本地 prefix + KVPool 联合命中

```text
local hit = 200
external hit = 600
num_computed_tokens = 800
本地 blocks 直接复用
外部 blocks load 到本地目标 block
剩余 200 token forward
```

### 25.4 场景四：KVPool full hit

```text
external/local 总命中 = prompt_len
本地 prefix cache 查询会把最大本地 hit 限制到 prompt_len - 1
如果异步外部 load 后 num_computed_tokens == prompt_len：finished_recving 后 cache_blocks()
_update_waiting_for_remote_kv() 把 num_computed_tokens 回退到 prompt_len - 1
下一轮 forward 最后一个 prompt token
sample 第一个 output token
```

### 25.5 场景五：load_kv_async=True

```text
Scheduler 查到外部命中且 connector 要异步 load
num_new_tokens = 0
allocate_slots(delay_cache_blocks=True)
request.status = WAITING_FOR_REMOTE_KVS
如果同轮还有其他 scheduled tokens，则在普通 forward step 中随其他请求推进 load
只有整个 batch total_num_scheduled_tokens=0 时才走 no-forward
finished_recving 回 Scheduler
请求恢复 WAITING / PREEMPTED
下一轮再调度 forward
```

### 25.6 场景六：invalid block，回退重算

```text
Worker connector load 失败
KVConnectorOutput.invalid_block_ids = {...}
Scheduler._handle_invalid_blocks()
找到受影响 request
request.num_computed_tokens 截断到失败 block 前
failure_policy=recompute：后续重算失败 block 之后 tokens
failure_policy=fail：请求失败
```

### 25.7 场景七：请求结束后异步 save，延迟释放 block

```text
request finished
_connector_finished()
connector.request_finished() 返回 True
Scheduler 保留 blocks
Worker connector save / send 完成
finished_sending 回 Scheduler
Scheduler._free_blocks()
```

---

## 26. 与 SchedulerOutput / ModelRunnerOutput 的对账关系

KVPool 链路的对账可以压缩成：

```text
SchedulerOutput：
  这一轮计划做什么，包括 connector metadata。

ModelRunnerOutput：
  这一轮实际做完什么，包括 connector output。

Scheduler.update_from_output()：
  用两者对账，更新请求状态和 KV block 生命周期。
```

尤其是：

```text
SchedulerOutput.finished_req_ids
  → Worker connector.get_finished(finished_req_ids)
  → ModelRunnerOutput.kv_connector_output.finished_sending
  → Scheduler 释放保存完成的 blocks
```

以及：

```text
SchedulerOutput.kv_connector_metadata
  → Worker start_load_kv()
  → ModelRunnerOutput.kv_connector_output.finished_recving
  → Scheduler 恢复 WAITING_FOR_REMOTE_KVS 请求
```

---

## 27. 容易疑惑的点

### 27.1 KVPool 命中是不是 Worker 发现的？

不是。

KVPool 外部命中数量由 Scheduler 侧 connector 查询：

```text
connector.get_num_new_matched_tokens()
```

Worker 不重新判断命中了多少，只按 `kv_connector_metadata` 执行 load / save。

### 27.2 KVPool 命中是不是直接跳过整个请求？

不是。

命中只会增加 `num_computed_tokens`，从而减少本轮 `num_new_tokens`。

如果 full hit，还需要回退最后一个 token 做一次 forward 来产生 logits。

### 27.3 外部 KV load 后是不是变成普通本地 KV？

是。

load 完成后，KV 被放进本地 paged KV cache blocks，attention backend 通过 block table / slot mapping 使用它。

### 27.4 0-token step 是不是可以跳过 Worker？

不能简单跳过。

如果有 KV connector，0-token step 仍可能推进异步 load/save，并返回 `finished_recving`、`finished_sending`、`invalid_block_ids`。

### 27.5 finished_recving 和 finished_sending 有什么区别？

```text
finished_recving：外部 KV 已经 load 到本地，可以恢复等待请求；
finished_sending：本地 KV 已经 save/send 到外部，可以释放延迟保留的本地 blocks。
```

### 27.6 invalid_block_ids 为什么按 block 而不是 req_id 回传？

因为失败发生在外部 KV block load 层面，一个 block 可能被多个请求共享。

Scheduler 需要扫描请求 block table，找出受影响请求，并尽量保留最长有效前缀。

### 27.7 KVPool 和 prefix cache 谁优先？

本地 prefix cache 先查，外部 KVPool 后查。

这样可以避免远端重复搬运本地已经有的 blocks。

---

## 28. 总结

KVPool 端到端链路可以压缩为：

```text
Scheduler 查命中：
  local prefix cache → external KVPool

Scheduler 做账本：
  num_computed_tokens → allocate_slots → kv_connector_metadata

Worker 做物理传输：
  bind metadata → start_load_kv → forward/save_kv_layer → get_finished

Scheduler 做回收：
  finished_recving 恢复请求
  finished_sending 释放 blocks
  invalid_block_ids fail / recompute
```

如果只记住一句话：

```text
KVPool 在 Scheduler 侧决定“哪些 KV 可以复用、要分配哪些本地 block”，在 Worker 侧完成“把外部 KV 搬进/搬出本地 paged KV cache”，再通过 ModelRunnerOutput 把完成或失败状态回传给 Scheduler。
```

再压缩成最小心智模型：

```text
本地 prefix cache 解决“本地有没有”；
KVPool connector 解决“外部有没有”；
BlockPool 解决“本地放哪里”；
SchedulerOutput 解决“告诉 Worker 做什么”；
Worker connector 解决“真正怎么搬”；
ModelRunnerOutput 解决“告诉 Scheduler 搬完没有”。
```
