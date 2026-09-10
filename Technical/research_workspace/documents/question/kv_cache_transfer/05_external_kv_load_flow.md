# 05. 外部 KV Cache 如何 load 回本地？

源码位置：

- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/kv_cache_coordinator.py`
- `code/vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/kv_connector.py`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py`
- `code/vllm/vllm/v1/outputs.py`

本问题关注：外部 KV Cache 命中后，Scheduler 如何把“远端已有 KV”转成本地 block 分配和 load 计划；`load_kv_async=True` 为什么会让本轮不 forward；`delay_cache_blocks=True` 为什么必须存在；Worker / ModelRunner 如何发起 load；`finished_recving` 如何回到 Scheduler 并让 `WAITING_FOR_REMOTE_KVS` 请求恢复调度；以及 load 失败时如何进入 invalid blocks / recompute 链路。

---

## 1. 一句话回答

外部 KV load 的本质是：

```text
把原本需要本地 prefill forward 的一段 prompt token，
转换成“先分配本地 KV blocks，再由 Worker connector 从外部系统把 KV 写回这些 blocks”。
```

如果 connector 返回：

```text
(num_external_computed_tokens > 0, load_kv_async=True)
```

Scheduler 会：

```text
1. 给这些外部 KV token 分配本地 block；
2. 构造 SchedulerOutput.kv_connector_metadata；
3. 把请求置为 WAITING_FOR_REMOTE_KVS；
4. 本轮不把它加入 running，也不 forward；
5. 等 Worker 报告 finished_recving 后，再把请求恢复成 WAITING / PREEMPTED 继续调度。
```

最小心智模型：

```text
外部 KV load = Scheduler 先预留本地 KV block，Worker 异步填充这些 block，Scheduler 收到 finished_recving 后再承认这些 token 已经 computed。
```

---

## 2. 总体主链路

完整链路可以压缩成：

```text
WAITING request
  → Scheduler 查询本地 prefix cache
  → connector.get_num_new_matched_tokens(request, local_hit_tokens)
  → 得到 num_external_computed_tokens / load_kv_async
  → KVCacheManager.allocate_slots(... num_external_computed_tokens, delay_cache_blocks=load_kv_async)
  → 为远端 KV 分配本地 blocks
  → connector.update_state_after_alloc(request, blocks, num_external_tokens)
  → connector.build_connector_meta(scheduler_output)
  → SchedulerOutput.kv_connector_metadata
  → Executor / Worker / ModelRunner.execute_model()
  → kv_connector.bind_connector_metadata(...)
  → kv_connector.start_load_kv(forward_context)
  → Worker connector 发起外部 KV load
  → kv_connector.get_finished(...)
  → ModelRunnerOutput.kv_connector_output.finished_recving
  → Scheduler.update_from_output()
  → _update_from_kv_xfer_finished()
  → finished_recving_kv_req_ids.add(req_id)
  → 下一轮 schedule() promote WAITING_FOR_REMOTE_KVS
  → _update_waiting_for_remote_kv()
  → cache_blocks() / 调整 num_computed_tokens
  → 请求重新进入 WAITING / PREEMPTED
  → 后续按普通请求继续 schedule
```

关键点是：

```text
load 的发起发生在 Worker / ModelRunner 侧；
load 的完成闭环发生在 Scheduler.update_from_output() 之后的下一轮 schedule()。
```

---

## 3. 涉及的核心对象

### 3.1 Request.kv_transfer_params

外部 KV load 通常由请求上的 `kv_transfer_params` 驱动。V1 `Request` 初始化时会从 `sampling_params.extra_args["kv_transfer_params"]` 读取它；pooling params 目前不会设置这个字段。

位置：`code/vllm/vllm/v1/request.py:101` 到 `code/vllm/vllm/v1/request.py:123`

以 NIXL pull 为例，常见字段包括：

```text
do_remote_prefill
remote_block_ids
remote_engine_id
remote_request_id
remote_host
remote_port
remote_num_tokens
do_remote_decode
```

这些字段告诉 Scheduler 侧 connector：

```text
远端是否有可读 KV；
远端请求是谁；
远端 block ids 是什么；
需要从哪个 engine / host / port 读取。
```

### 3.2 Scheduler 侧 connector

抽象接口在：`kv_connector/v1/base.py:453`

与 load 直接相关的是：

```text
get_num_new_matched_tokens(request, num_computed_tokens)
update_state_after_alloc(request, blocks, num_external_tokens)
build_connector_meta(scheduler_output)
update_connector_output(connector_output)
```

Scheduler 侧 connector 不真正搬 KV 数据。它只做：

```text
命中判断、状态记录、metadata 构造、worker 回报处理。
```

### 3.3 Worker 侧 connector

抽象接口在：`kv_connector/v1/base.py:292`

与 load 直接相关的是：

```text
handle_preemptions(kv_connector_metadata)
bind_connector_metadata()
start_load_kv(forward_context)
get_finished(finished_req_ids)
get_block_ids_with_load_errors()
build_connector_worker_meta()
clear_connector_metadata()
```

Worker 侧 connector 负责真正把外部 KV 写入本地 paged KV buffer。

### 3.4 SchedulerOutput.kv_connector_metadata

定义入口在：`SchedulerOutput`，`sched/output.py:182` 起；`kv_connector_metadata` 字段在 `sched/output.py:234`。

它是：

```text
Scheduler connector → Worker connector 的一轮 KV transfer 计划。
```

不同 connector 的 metadata 类型不同。例如 NIXL 使用 `NixlConnectorMetadata`，Offloading 使用 `OffloadingConnectorMetadata`。

### 3.5 KVConnectorOutput

定义在：`outputs.py:196`

与 load 相关的字段：

```text
finished_recving       哪些请求的 async load 完成
invalid_block_ids      哪些外部 KV block load 失败
kv_connector_stats     worker 侧 connector 统计
kv_cache_events        connector 产生的 KV cache events
kv_connector_worker_meta  worker → scheduler 的附加 metadata
```

---

## 4. Scheduler 什么时候查询外部 KV

外部 KV 查询发生在 WAITING 请求调度阶段。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:673` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1006`

核心前提是：

```text
request.num_computed_tokens == 0
```

这说明这是一个尚未在本地跑起来的新请求，或者是被抢占后重新等待的请求，需要重新判断本地 / 外部 prefix 命中。async load 完成后回到 WAITING 的请求通常已经带着 `num_computed_tokens > 0`，会走“KVTransfer: WAITING reqs have num_computed_tokens > 0”分支，不再重新查询 connector。

流程是：

```text
1. 先查本地 prefix cache；
2. 再把本地命中的 token 数传给 connector；
3. connector 只返回“本地已 computed 之后，外部还能补多少 token”。
```

对应代码：

```text
new_computed_blocks, num_new_local_computed_tokens =
    kv_cache_manager.get_computed_blocks(request)

ext_tokens, load_kv_async =
    connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)

num_external_computed_tokens = ext_tokens
num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:723` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:800`

---

## 5. 本地 prefix hit 和外部 KV hit 如何合并

Scheduler 把已计算 token 分成两段：

```text
local computed tokens：
  本地 prefix cache 已命中的 token。

external computed tokens：
  外部 KV cache 命中的 token，需要 load 回本地 block。
```

最终：

```text
num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
```

注意：

```text
connector.get_num_new_matched_tokens() 的 num_computed_tokens 参数是本地已经命中的 token 数。
```

也就是说，connector 不能重复计算本地 prefix cache 已经命中的部分，它只负责回答：

```text
从 local_hit_tokens 之后开始，外部还能提供多少 token 的 KV？
```

抽象接口说明在：`kv_connector/v1/base.py:454` 到 `kv_connector/v1/base.py:485`

---

## 6. get_num_new_matched_tokens() 返回什么

接口返回：

```text
(tuple[int | None, bool])
```

含义是：

```text
第一个值：外部 KV cache 可提供的新增 token 数。
  0     → 没有外部命中；
  N>0   → 有 N 个 token 可以从外部 load；
  None  → connector 暂时无法确定，Scheduler 稍后再查。

第二个值：是否异步 load。
  True  → 这些 token 会在 scheduler step 之间异步加载；
  False → 不需要进入 WAITING_FOR_REMOTE_KVS。
```

接口约束：

```text
如果第一个值是 0，第二个值必须是 False。
```

位置：`kv_connector/v1/base.py:454` 到 `kv_connector/v1/base.py:485`

### 6.1 返回 None 的含义

如果返回：

```text
ext_tokens is None
```

Scheduler 会把请求从当前队列取出，放入本轮 `step_skipped_waiting`，稍后再尝试。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:781` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:787`

这表示 connector 还没准备好判断命中，而不是请求失败。

### 6.2 NIXL pull 示例

`NixlPullConnectorScheduler.get_num_new_matched_tokens()`：`nixl/pull_scheduler.py:34`

remote prefill 场景：

```text
如果 kv_transfer_params.do_remote_prefill=True：
  actual = _mamba_prefill_token_count(len(prompt_token_ids))
  count = actual - local_computed_tokens
  如果 count > 0：返回 (count, True)
```

其中 Mamba 模型会把 `actual` 设为 `num_prompt_tokens - 1`，因为 decoder 侧需要重算最后一个 prompt token。

remote decode / block ids 场景：

```text
如果有 remote_block_ids 和 remote metadata：
  count = min(remote_num_tokens, prompt_tokens) - local_computed_tokens
  如果 count 低于 kv_recompute_threshold，可以返回 (0, False) 选择本地重算；
  否则返回 (count, True)。
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:60` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:110`

### 6.3 Offloading 示例

`OffloadingConnectorScheduler.get_num_new_matched_tokens()`：`offloading/scheduler.py:692`

它会：

```text
1. 如果请求仍有 in-flight transfers，返回 (None, False) 延后重试；
2. 更新请求 offload keys；
3. 记录本地 computed tokens；
4. 查 offload manager 是否命中；
5. touch 命中的 offloaded entries；
6. 返回 (num_hit_tokens, bool(num_hit_tokens))。
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:718` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:747`

这说明 offloading connector 的外部 KV load 也是异步 load：只要命中 token 数大于 0，就返回 `load_kv_async=True`。

---

## 7. load_kv_async=True 表示什么

当：

```text
load_kv_async=True
```

Scheduler 的语义是：

```text
这批外部 KV 不能在当前 schedule() 内同步视为就绪；
必须先分配本地 blocks，发起 Worker 侧 load，
等后续 finished_recving 回来后才能继续调度这个请求。
```

对应分支：

```python
if load_kv_async:
    assert num_external_computed_tokens > 0
    num_new_tokens = 0
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:834` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:837`

这就是为什么异步 load 时本轮不 forward：

```text
本轮只有“为外部 KV 准备落地点 + 发起 load”的意义，
没有本地要计算的新 token。
```

---

## 8. 为什么 load_kv_async 时 num_new_tokens = 0

因为外部 KV 命中的 token 已经在远端算过，它们本轮不需要本地模型 forward。

Scheduler 把本轮动作拆成：

```text
num_external_computed_tokens：
  需要分配本地 KV slot，并由 connector load 内容。

num_new_tokens：
  需要本地 forward 计算的 token。
```

异步 load 的第一步只做前者，所以：

```text
num_new_tokens = 0
```

这样有两个结果：

```text
1. allocate_slots() 仍然允许执行，因为 num_external_computed_tokens > 0；
2. SchedulerOutput.total_num_scheduled_tokens 不会因为这个请求增加。
```

`KVCacheManager.allocate_slots()` 特意允许这种情况：

```python
if num_new_tokens == 0 and num_external_computed_tokens == 0:
    raise ValueError(...)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:365` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:371`

也就是说：

```text
num_new_tokens == 0 是合法的，前提是存在 external computed tokens。
```

---

## 9. allocate_slots() 如何为外部 KV 分配本地 block

调用位置：`code/vllm/vllm/v1/core/sched/scheduler.py:914` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:954`

异步 load 时关键参数是：

```text
num_new_tokens=0
num_new_computed_tokens=num_new_local_computed_tokens
new_computed_blocks=new_computed_blocks
num_external_computed_tokens=num_external_computed_tokens
delay_cache_blocks=True
num_lookahead_tokens=0 if load_kv_async and num_lookahead_tokens > 0 else normal lookahead
full_sequence_must_fit=scheduler_reserve_full_isl
reserved_blocks=_inflight_prefill_reserved_blocks()
has_scheduled_reqs=bool(self.running)
```

`allocate_slots()` 内部会：

```text
1. 计算 total_computed_tokens = local + external；
2. remove_skipped_blocks() 释放窗口外 blocks；
3. get_num_blocks_to_allocate() 计算需要分配多少本地 block；
4. allocate_new_computed_blocks()：
   - touch 本地 prefix-hit blocks；
   - 为 external computed tokens 分配本地 blocks；
5. allocate_new_blocks()：本轮 new token 为 0 时通常不再为 forward token 分配；
6. 因 delay_cache_blocks=True，直接返回新 blocks，不 cache。
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:353` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:477`

---

## 10. external computed tokens 对 block 的含义

`num_external_computed_tokens` 表示：

```text
这些 token 的 KV 不在 vLLM 本地 prefix cache 中，
但外部 connector 认为它们可用。
```

因此 Scheduler 需要做两件事：

```text
1. 给它们分配本地 block ids；
2. 让 Worker connector 知道应该把远端 KV 写入哪些本地 block ids。
```

真正分配本地 blocks 的路径是：

```text
KVCacheCoordinator.allocate_new_computed_blocks()
  → 第一阶段：对所有 KV group 调 SingleTypeKVCacheManager.add_local_computed_blocks()
  → 第二阶段：对所有 KV group 调 SingleTypeKVCacheManager.allocate_external_computed_blocks()
```

这里采用 two-phase allocation：先 touch 每个 group 的本地 prefix-hit blocks，再给每个 group 分配 external blocks，避免前一个 group 分配 external blocks 时驱逐后一个 group 尚未 touch 的 cache-hit blocks。

其中 `allocate_external_computed_blocks()` 会：

```text
1. 根据 local + external computed tokens 计算需要多少 blocks；
2. block_pool.get_new_blocks(...) 分配本地 blocks；
3. req_to_blocks[request_id].extend(allocated_blocks)；
4. 对部分 attention 类型记录 new_block_ids。
```

位置：`code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:282` 到 `code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:319`

---

## 11. delay_cache_blocks=True 的含义

`delay_cache_blocks=True` 只在 KV transfer 这类场景下使用。

它表示：

```text
这些 blocks 虽然已经分配给请求，
但其 KV 内容还没有真正 load 到本地 GPU cache，
因此不能立刻写入 prefix cache hash map。
```

`KVCacheManager.allocate_slots()` 中：

```python
if not self.enable_caching or delay_cache_blocks:
    return self.create_kv_cache_blocks(new_blocks)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:474` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:477`

如果没有这个延迟，可能出现错误：

```text
Scheduler 把尚未填充完成的 blocks 标记为 prefix cache；
后续其他请求命中这些 blocks；
Worker 读取到未完成或错误的 KV 内容。
```

所以外部 KV load 的缓存提交必须等：

```text
Worker finished_recving
  → Scheduler._update_waiting_for_remote_kv()
  → kv_cache_manager.cache_blocks(...)
```

---

## 12. connector.update_state_after_alloc() 做什么

Scheduler 分配完本地 blocks 后，会调用：

```python
connector.update_state_after_alloc(
    request,
    kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:969` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:974`

这个调用非常关键，因为 connector 只有在 block 分配之后才知道：

```text
远端 KV 应该 load 到哪些本地 block ids。
```

### 12.1 NIXL pull 示例

`NixlPullConnectorScheduler.update_state_after_alloc()`：`nixl/pull_scheduler.py:112`

它会把请求加入：

```text
_reqs_need_recv[request_id] = (request, local_block_ids)
```

其中 `local_block_ids` 的生成方式是：

```text
如果 num_external_tokens > 0：
  unhashed_local_block_ids = blocks.get_unhashed_block_ids_all_groups()
否则：
  unhashed_local_block_ids = ()

local_block_ids = get_sw_clipped_blocks(unhashed_local_block_ids)
```

含义是：

```text
只拉取那些还没有本地 prefix cache hash 的 blocks，并按滑动窗口裁剪。
```

如果本地已经 full prefix cache hit，`num_external_tokens=0`，会保留一个空 recv 任务，用于后续 `send_notif` 通知远端释放。

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:126` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:179`

### 12.2 Offloading 示例

`OffloadingConnectorScheduler.update_state_after_alloc()`：`offloading/scheduler.py:749`

它会构造：

```text
keys_to_load      外部存储中的 KV keys
dst_block_ids     本地 GPU block ids
group_sizes       每个 KV group 要 load 的 block 数
block_indices     每个 group 从哪个 block index 开始写
load_job_id       这次 load 任务 id
```

然后写入：

```text
_current_batch_load_jobs[load_job_id] = TransferJob(...)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:749` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:846`

这就是后续 `build_connector_meta()` 发送给 Worker 的 load job。

---

## 13. WAITING_FOR_REMOTE_KVS 如何进入 skipped_waiting

`update_state_after_alloc()` 后，Scheduler 会把请求从 waiting 队列取出。

如果 `load_kv_async=True`：

```python
request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
step_skipped_waiting.prepend_request(request)
request.num_computed_tokens = num_computed_tokens
self._inflight_prefills.add(request)
continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:985` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1006`

几个关键点：

```text
1. 请求不进入 self.running；
2. 不加入 scheduled_new_reqs / scheduled_cached_reqs；
3. 不消耗 token_budget，因为 num_new_tokens=0；
4. request.num_computed_tokens 先设置成 local + external；
5. 这个值在 transfer 完成前不会被用于 forward；
6. 请求进入 skipped_waiting，后续每轮尝试 promote。
```

这里的 `num_computed_tokens` 是“预期完成后”的进度，而不是“当前本地已经可用”的进度。

源码注释也强调：

```text
request.num_computed_tokens will not be used anywhere until the request finished the KV transfer.
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:991` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1004`

---

## 14. build_connector_meta() 如何把 load 计划放进 SchedulerOutput

Scheduler 在构造完 `SchedulerOutput` 后调用：

```python
meta = self._build_kv_connector_meta(self.connector, scheduler_output)
scheduler_output.kv_connector_metadata = meta
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1162` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1168`

`_build_kv_connector_meta()` 本质就是：

```python
return connector.build_connector_meta(scheduler_output)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1186` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1189`

### 14.1 NIXL metadata

`NixlBaseConnectorScheduler.build_connector_meta()`：`nixl/base_scheduler.py:397`

它会把 `_reqs_need_recv` 写入：

```text
NixlConnectorMetadata.reqs_to_recv
```

并附带：

```text
request_id
local_block_ids
kv_transfer_params
remote engine/request/host/port 信息
```

随后清空：

```text
_reqs_need_recv
_reqs_in_batch
_reqs_not_processed
_reqs_need_send
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:397` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:430`

### 14.2 Offloading metadata

`OffloadingConnectorScheduler.build_connector_meta()`：`offloading/scheduler.py:1088`

它会构造：

```text
OffloadingConnectorMetadata(
  load_jobs=current_batch_load_jobs,
  store_jobs=...,
  jobs_to_flush=...
)
```

然后清空当前批次的 load jobs。

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1121` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1128`

---

## 15. 0-token step 如何也能驱动 KV load

异步 load 请求本身不产生 scheduled tokens，但 Scheduler 仍然可能输出带有 `kv_connector_metadata` 的 `SchedulerOutput`。

Worker / ModelRunner 对 0-token step 有专门路径。

在 legacy `GPUModelRunner.execute_model()` 中：

```python
if not num_scheduled_tokens:
    if not has_kv_transfer_group():
        return EMPTY_MODEL_RUNNER_OUTPUT
    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4096` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4165`

在新版 GPU model runner 中，同样会在 `scheduler_output.total_num_scheduled_tokens == 0` 时走 connector 的 `no_forward()`：

```python
if scheduler_output.total_num_scheduled_tokens == 0:
    empty_output = self.kv_connector.no_forward(scheduler_output)
    return empty_output
```

位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1145` 到 `code/vllm/vllm/v1/worker/gpu/model_runner.py:1148`

这表示：

```text
即使本轮没有模型 forward，
只要有 KV connector，Worker 仍然会消费 connector metadata，
发起或推进 KV load / save。
```

legacy `kv_connector_no_forward()` 的实现位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:35`

它会：

```text
set_forward_context(None, vllm_config)
_get_kv_connector_output(... wait_for_save=False)
返回只包含 kv_connector_output 的 ModelRunnerOutput
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:36` 到 `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:48`

新版 `ActiveKVConnector.no_forward()` 则会直接调用 `pre_forward()` 和 `post_forward(wait_for_save=False)`，同样只推进 connector，不跑模型。

位置：`code/vllm/vllm/v1/worker/gpu/kv_connector.py:98` 到 `code/vllm/vllm/v1/worker/gpu/kv_connector.py:105`

---

## 16. Worker / ModelRunner 如何发起 load

legacy Worker 侧 connector 生命周期被包在：

```python
KVConnectorModelRunnerMixin._get_kv_connector_output(...)
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:77`

核心步骤：

```text
1. 创建 KVConnectorOutput；
2. get_kv_transfer_group() 取 Worker 侧 connector；
3. bind_connector_metadata(scheduler_output.kv_connector_metadata)；
4. start_load_kv(get_forward_context())；
5. yield 给 forward 或 no-forward 主体；
6. finally 中 wait_for_save；
7. get_finished(finished_req_ids)；
8. get_block_ids_with_load_errors()；
9. 收集 stats / events / worker_meta；
10. clear_connector_metadata()。

新版 `ActiveKVConnector.pre_forward()` 在 bind metadata 前还会调用 `handle_preemptions(kv_connector_metadata)`，用于让需要处理异步 save / evict 的 connector 在 block 被覆盖前完成预处理。
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:83` 到 `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:112`

新版 GPU runner 的 `ActiveKVConnector` 拆成 `pre_forward()` / `post_forward()`：`pre_forward()` 绑定 metadata 并启动 load，`post_forward()` 等待 save、收集 finished_recving / invalid_block_ids / stats / worker_meta 并清理 metadata。

位置：`code/vllm/vllm/v1/worker/gpu/kv_connector.py:61` 到 `code/vllm/vllm/v1/worker/gpu/kv_connector.py:95`

最关键的是：

```python
kv_connector.start_load_kv(get_forward_context())
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:91` 到 `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:95`

新版路径对应的是：

```python
self.kv_connector.start_load_kv(get_forward_context())
```

位置：`code/vllm/vllm/v1/worker/gpu/kv_connector.py:70` 到 `code/vllm/vllm/v1/worker/gpu/kv_connector.py:75`

这说明 load 是在 forward context 建好后、模型 forward 前启动的；如果是 0-token no-forward 路径，则会用空 forward context 驱动 connector。

---

## 17. NIXL Worker 如何执行 load

NIXL pull 的 Worker 入口：

```python
NixlPullConnector.start_load_kv()
  → NixlPullConnectorWorker.start_load_kv(metadata)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:343` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:347`

Worker 侧会遍历：

```text
metadata.reqs_to_recv
```

对每个请求：

```text
1. 把 local logical block ids 转成 kernel block ids；
2. 保存 recving metadata 供失败恢复；
3. 如远端 agent 未 handshake，则发起 background handshake；
4. handshake 已完成则 _read_blocks_for_req()；
5. 处理 ready_requests 中 handshake 刚完成的请求；
6. 发送 heartbeats / 处理 reqs_to_send 等控制信息。
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py:40` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py:99`

真正 READ transfer 在：

```text
_read_blocks_for_req()
  → 计算 TP mapping / remote ranks / local_block_ids / remote_block_ids
  → _read_blocks(...)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py:101` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py:178`

这条路径说明：

```text
Scheduler 只传 logical block ids；
Worker 会根据本地/远端 block size、TP mapping、物理布局把它们转成真实传输计划。
```

---

## 18. Offloading Worker 如何执行 load

Offloading 的 Worker 侧入口先经过 wrapper：

```text
OffloadingConnector.start_load_kv()
  → OffloadingConnectorWorker.start_kv_transfers(metadata)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:89` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:92`

Worker 在 metadata 中看到：

```text
metadata.load_jobs
```

`OffloadingConnectorWorker.start_kv_transfers()` 位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:281`

它会：

```text
1. 先提交上一轮未提交的 store jobs；
2. 遍历 metadata.load_jobs；
3. 记录 _load_jobs[job_id] = req_id；
4. worker.submit_load(job_id, src_spec, dst_spec) 发起异步 load。
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:281` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:292`

load 完成后，wrapper 的 `OffloadingConnector.get_finished()` 会先把 store jobs 延迟提交给下一步的 `start_kv_transfers()`，再从底层 worker 拉取 finished transfer：

```text
OffloadingConnector.get_finished()
  → connector_worker.prepare_store_kv(metadata)
  → connector_worker.get_finished(finished_req_ids)
  → 如果 job_id 属于 _load_jobs
  → finished_recving.add(req_id)
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:111` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:120`

底层 worker 位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:304` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:339`

---

## 19. finished_recving 从哪里回来

Worker / ModelRunner 在 `_get_kv_connector_output()` 的 finally 中调用：

```python
output.finished_sending, output.finished_recving = (
    kv_connector.get_finished(scheduler_output.finished_req_ids)
)
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:102` 到 `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:104`

新版 GPU connector 对应位置：`code/vllm/vllm/v1/worker/gpu/kv_connector.py:83` 到 `code/vllm/vllm/v1/worker/gpu/kv_connector.py:95`

然后这个 output 被放进：

```text
ModelRunnerOutput.kv_connector_output
```

`KVConnectorOutput.finished_recving` 定义在：`outputs.py:196` 到 `outputs.py:199`

含义是：

```text
这些请求的外部 KV load 已经在 Worker 侧完成，可以由 Scheduler 恢复调度。
```

注意：

```text
finished_recving 不等于本轮请求已经 forward；
它只表示本地 blocks 已经被外部 KV 填好。
```

---

## 20. Scheduler 如何处理 finished_recving

Scheduler 在 `update_from_output()` 末尾处理 connector output：

```python
if kv_connector_output:
    self._update_from_kv_xfer_finished(kv_connector_output)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1837` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1839`

`_update_from_kv_xfer_finished()` 中：

```python
for req_id in kv_connector_output.finished_recving or ():
    req = self.requests[req_id]
    if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        self.finished_recving_kv_req_ids.add(req_id)
    else:
        assert RequestStatus.is_finished(req.status)
        self._free_blocks(self.requests[req_id])
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2559` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2586`

这说明：

```text
如果请求还在 WAITING_FOR_REMOTE_KVS，先只记录 finished_recving；
真正把请求状态恢复，发生在下一次 schedule() 遍历 waiting/skipped_waiting 时。
```

为什么不在 update_from_output() 里直接恢复？

```text
因为 Scheduler 的 admission、token budget、LoRA 限制、block 分配等统一发生在 schedule()；
把恢复动作放在 schedule() 开头，可以复用同一套调度入口。
```

---

## 21. WAITING_FOR_REMOTE_KVS 如何恢复调度

下一轮 `Scheduler.schedule()` 遍历 waiting / skipped_waiting 时，先尝试 promote blocked status：

```python
if self._is_blocked_waiting_status(request.status) and not self._try_promote_blocked_waiting_request(request):
    ...
    step_skipped_waiting.prepend_request(request)
    continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:691` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:701`

`WAITING_FOR_REMOTE_KVS` 属于 blocked waiting status：

```python
return status in (
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
    WAITING_FOR_REMOTE_KVS,
    WAITING_FOR_STREAMING_REQ,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1911` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1916`

恢复逻辑在：`_try_promote_blocked_waiting_request()`，位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2526`

```text
如果 req_id 不在 finished_recving_kv_req_ids：
  返回 False，请求继续留在 skipped_waiting。

如果 req_id 已 finished_recving：
  调 _update_waiting_for_remote_kv(request)；
  如果 request.num_preemptions > 0，状态置 PREEMPTED；
  否则状态置 WAITING；
  返回 True，继续走普通 WAITING 调度逻辑。
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2530` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2541`

---

## 22. _update_waiting_for_remote_kv() 做了什么

`_update_waiting_for_remote_kv()` 是外部 KV load 完成后的提交点。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2492`

正常成功路径：

```text
1. kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
2. 如果 request.num_computed_tokens == request.num_tokens：
     request.num_computed_tokens = request.num_tokens - 1
3. finished_recving_kv_req_ids.remove(request_id)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2515` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2524`

### 22.1 为什么这里才 cache_blocks()

因为只有到 `finished_recving` 后，Worker 才确认外部 KV 已经写入本地 blocks。

这时才可以：

```text
给这些 full blocks 设置 block_hash；
放入本地 prefix cache；
让后续请求复用。
```

### 22.2 为什么 full prompt hit 要回退最后一个 token

如果：

```text
request.num_computed_tokens == request.num_tokens
```

说明整个 prompt 都来自外部 KV。

但生成下一个 token 仍需要最后一个 prompt token 的 logits，所以 vLLM 会把进度改成：

```text
request.num_tokens - 1
```

让下一轮保留最后 token 的 forward，用于产生 next-token logits。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2519` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2522`

这个逻辑与本地 prefix cache full hit 的处理一致。

---

## 23. load 失败时如何处理

Worker connector 可以通过：

```text
KVConnectorOutput.invalid_block_ids
```

报告 load 失败的 block。

字段定义在：`outputs.py:203` 到 `outputs.py:205`

Worker 侧采集位置：

```python
output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
```

位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:105`

新版 GPU connector 对应位置：`code/vllm/vllm/v1/worker/gpu/kv_connector.py:89`

Scheduler 在 `update_from_output()` 开头处理：

```python
if kv_connector_output and kv_connector_output.invalid_block_ids:
    failed_kv_load_req_ids = self._handle_invalid_blocks(...)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1578` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1586`

### 23.1 async load 失败

`_handle_invalid_blocks()` 会先扫描：

```text
self.skipped_waiting 中 status == WAITING_FOR_REMOTE_KVS 的请求
```

并调用：

```text
_update_requests_with_invalid_blocks(... evict_blocks=False)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2702` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2715`

这会把受影响请求的 `num_computed_tokens` 回退到第一个失败 block 之前。

### 23.2 finished_recving 后如何提交失败结果

如果启用了 recompute policy（`recompute_kv_load_failures=True`），async 请求有失败 block 时，Scheduler 会记录：

```text
failed_recving_kv_req_ids |= async_failed_req_ids
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2757` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2759`

随后请求仍要等 `finished_recving`，因为 connector 必须完成/收尾这次 transfer。

等 `_update_waiting_for_remote_kv()` 执行时：

```text
如果 request_id 在 failed_recving_kv_req_ids：
  如果 request.num_computed_tokens > 0：cache 有效 prefix；
  否则：kv_cache_manager.free(request) 释放已分配 blocks；
  从 failed_recving_kv_req_ids 移除。
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2502` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2513`

也就是说，在 recompute policy 下：

```text
load 失败不是立刻让请求消失；
Scheduler 会尽量保留失败 block 之前的有效 prefix，然后后续本地重算失败部分。
```

如果配置的 failure policy 是 fail（`recompute_kv_load_failures=False`），`_handle_invalid_blocks()` 会直接返回受影响请求，`update_from_output()` 随后用 `FINISHED_ERROR` 结束它们；相关细节见 `08_invalid_blocks_and_recompute.md`。

---

## 24. 同步 load 和异步 load 的区别

从 Scheduler 角度：

```text
load_kv_async=True：
  本轮只分配 blocks + 发起 load；
  请求进入 WAITING_FOR_REMOTE_KVS；
  finished_recving 后下一轮恢复。

load_kv_async=False：
  不进入 WAITING_FOR_REMOTE_KVS；
  请求可以继续作为普通 scheduled request 进入 running；
  connector 可能在本轮 forward 前同步/半同步完成必要 load。
```

抽象接口允许两种模式，但 NIXL pull / Offloading 的外部命中通常走 async。

判断关键不是 `num_external_computed_tokens` 是否大于 0，而是 connector 返回的第二个值：

```text
load_kv_async
```

---

## 25. 为什么 async load 要 reserved_blocks

async load 会占用本地 blocks，但请求暂时不 forward，也不容易通过普通 preemption 释放进度。

为了避免它吃掉其他 in-flight prefill 继续执行所需的 blocks，Scheduler 会在 async load 时计算：

```python
reserved_blocks = self._inflight_prefill_reserved_blocks()
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:934` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:940`

`allocate_slots()` 会用：

```text
available_blocks = block_pool.get_num_free_blocks() - reserved_blocks
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:446` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:450`

这样可以避免：

```text
async load 请求占住大量 KV blocks，
导致已经在进行的 prefill 请求无法完成，
进而产生死锁或频繁抢占。
```

---

## 26. 与 preemption 的关系

async load 请求进入：

```text
WAITING_FOR_REMOTE_KVS + skipped_waiting
```

它不在 `self.running` 中，因此不会作为普通 running request 被 `_preempt_request()` 抢占。

但它仍然持有已经分配的 blocks。

如果外部中止 / finish 请求时，`finish_requests()` 会特殊判断：

```text
如果 request.status == WAITING_FOR_REMOTE_KVS：
  delay_free_blocks = request_id not in finished_recving_kv_req_ids
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2141` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2152`

含义是：

```text
如果 load 还没完成，不能随便释放 blocks；
要等 connector 完成并回报，否则 Worker 可能还在写这些 blocks。
```

---

## 27. 与 Worker forward 的关系

外部 KV load 并不等价于本轮模型 forward。

三种常见情况：

```text
1. 只有 async load，没有 scheduled tokens：
   GPUModelRunner 走 kv_connector_no_forward()，不跑模型。

2. 有其他请求 scheduled tokens，同时某个请求在 async load：
   Worker 正常 forward 其他请求；connector.start_load_kv() 同时发起/推进 load。

3. load 完成后的下一轮：
   请求恢复 WAITING / PREEMPTED；Scheduler 重新计算 num_new_tokens；
   后续才可能进入 running 并 forward suffix / last token。
```

所以外部 KV load 是独立于 forward 的 transfer 子流程，但它借用了 execute_model 这一轮的 Worker 调用来推进。

---

## 28. 外部 KV load 与本地 prefix cache 的关系

外部 KV load 完成前：

```text
blocks 已分配给请求，但 delay_cache_blocks=True，不能进入本地 prefix cache。
```

外部 KV load 完成后：

```text
_update_waiting_for_remote_kv()
  → kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
  → full blocks 写入本地 prefix cache
```

因此，一个远端 KV 命中请求会在完成 load 后反过来“暖”本地 prefix cache。

后续相同 prefix 的请求可能不再需要外部 KV load，而是直接走本地 `get_computed_blocks()` 命中。

---

## 29. 与 SchedulerOutput 的关系

在 async load 发起那一轮：

```text
该 load 请求通常不会出现在 scheduled_new_reqs / scheduled_cached_reqs 中，
因为它没有 scheduled tokens，也没有进入 running。
```

但它的 load 计划会出现在：

```text
scheduler_output.kv_connector_metadata
```

这也是为什么 Worker 侧必须在 0-token step 仍然处理 connector metadata。

对于 Worker 来说：

```text
SchedulerOutput 的 token 调度部分可能为空，
但 KV connector metadata 仍然有实际工作。
```

---

## 30. 一个完整例子

假设请求 prompt 长度 1024，本地 prefix cache 命中 256 token，外部 KV cache 可提供后续 768 token。

### 30.1 Scheduler 查询命中

```text
local_hit = 256
connector.get_num_new_matched_tokens(request, 256)
  → (768, True)
num_computed_tokens = 256 + 768 = 1024
```

### 30.2 Scheduler 分配 blocks

```text
allocate_slots(
  num_new_tokens=0,
  num_new_computed_tokens=256,
  new_computed_blocks=<local hit blocks>,
  num_external_computed_tokens=768,
  delay_cache_blocks=True,
)
```

结果：

```text
本地 prefix-hit blocks 被 touch；
外部 KV 对应的本地 blocks 被分配；
但不 cache 这些 external blocks。
```

### 30.3 Connector 构造 load 计划

```text
update_state_after_alloc()
  → 记录 req_id、remote info、本地 block ids
build_connector_meta()
  → SchedulerOutput.kv_connector_metadata
```

### 30.4 请求进入等待

```text
request.status = WAITING_FOR_REMOTE_KVS
request.num_computed_tokens = 1024
step_skipped_waiting.prepend_request(request)
```

### 30.5 Worker 执行 load

```text
ModelRunner.execute_model()
  → bind_connector_metadata()
  → start_load_kv()
  → NIXL READ / offloading submit_load
  → get_finished()
  → finished_recving={req_id}
```

### 30.6 Scheduler 恢复请求

```text
update_from_output()
  → finished_recving_kv_req_ids.add(req_id)
下一轮 schedule()
  → _try_promote_blocked_waiting_request()
  → _update_waiting_for_remote_kv()
  → cache_blocks(request, 1024)
  → num_computed_tokens = 1023
  → status = WAITING
```

下一轮普通调度会保留最后 token 的 forward 来计算 next-token logits，然后继续 decode。

---

## 31. 容易疑惑的点

### 31.1 外部 KV 命中后是不是马上能进入 running？

不一定。

如果 `load_kv_async=True`，请求会先进入 `WAITING_FOR_REMOTE_KVS`，等 Worker 回报 `finished_recving` 后才恢复调度。

### 31.2 load_kv_async=True 时为什么 num_new_tokens=0？

因为本轮不做本地 forward，只为外部 KV 分配本地 block 并发起 load。

### 31.3 request.num_computed_tokens 提前设置会不会误用？

Scheduler 注释明确说明，这个值在 KV transfer 完成前不会被用于 forward。请求处于 `WAITING_FOR_REMOTE_KVS`，会被 blocked status 逻辑跳过。

### 31.4 delay_cache_blocks=True 是不是不分配 block？

不是。

它仍然分配 block，只是不把这些 block 立刻写入本地 prefix cache。

### 31.5 finished_recving 是否表示请求完成？

不是。

它只表示外部 KV load 完成。请求还需要重新进入 Scheduler，可能保留最后 token 的 forward，继续 prefill suffix 或 decode。

### 31.6 为什么 0-token step 还要调用 Worker？

因为 KV connector 的 load/save 可以独立于模型 forward 推进。没有 scheduled tokens 时，`kv_connector_no_forward()` 仍会消费 metadata 并返回 connector output。

### 31.7 load 失败后为什么还要等 finished_recving？

因为 connector 需要完成 transfer 收尾并报告哪些 block 失败。Scheduler 会先记录 invalid blocks，再在 finished_recving 后提交可用 prefix 或释放 blocks。

### 31.8 外部 KV load 后会不会进入本地 prefix cache？

会，但必须等 `finished_recving` 后由 `_update_waiting_for_remote_kv()` 调 `cache_blocks()`，不能在分配 block 时提前 cache。

---

## 32. 最小心智模型

外部 KV load 的主线可以记成：

```text
Scheduler 查到外部 KV 命中
  → 分配本地 blocks 作为落地点
  → connector metadata 告诉 Worker 从哪里读、写到哪里
  → 请求进入 WAITING_FOR_REMOTE_KVS
  → Worker connector 异步 load KV
  → finished_recving 回到 Scheduler
  → Scheduler cache loaded blocks 并恢复请求
```

再压缩成一句话：

```text
外部 KV load 不是“跳过 Scheduler”，而是 Scheduler 先把远端 KV 映射成本地 block 账本，Worker 再把真实 KV 内容填进去，最后 Scheduler 根据 finished_recving 把请求重新放回普通调度链路。
```