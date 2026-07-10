# 09. deferred free 如何避免异步 KV 竞态？

源码位置：

- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/kv_cache_coordinator.py`
- `code/vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `code/vllm/vllm/v1/core/block_pool.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py`

本文梳理 vLLM V1 中 deferred free 的安全语义：为什么异步调度 / pipeline parallel / KV consumer 场景下不能立即把 KV block 还给 `BlockPool`，`last_sched_seq / processed_step_seq / deferred_frees` 如何构成 step fence，以及它和 KV connector delayed-free、offloading flush 的区别。

---

## 1. 一句话回答

`deferred free` 是 Scheduler 在“可能还有 in-flight GPU step 正在写 KV block”时，把 block 从请求 bookkeeping 里摘出来，但暂时不还给 `BlockPool` 的延迟归还机制。

主链路是：

```text
启用条件：KV consumer + max_concurrent_batches > 1

schedule() 产生非空 step
  → sched_step_seq += 1
  → _update_after_schedule()
      → request.last_sched_seq = sched_step_seq

request finished / preempted / finished_sending
  → _free_blocks() / _preempt_request()
  → _free_request_blocks(request)
      → 如果 request.last_sched_seq <= processed_step_seq：立即 free
      → 否则：pop_blocks_for_free()，放入 deferred_frees

update_from_output(scheduler_output, model_output)
  → 该非空 step 的 GPU 写已完成
  → processed_step_seq += 1
  → _drain_deferred_frees()
      → fence <= processed_step_seq 的 blocks 真正归还 BlockPool
```

所以：

```text
deferred free 保护的是“block 归还给 BlockPool 的时间”；
它不阻止请求从 running/waiting 中移除，也不等价于 connector 的异步 save delayed-free。
```

---

## 2. 本文要回答的问题

```text
为什么某些 block 不能立即释放？
defer_block_free 什么时候开启？
max_concurrent_batches 和 async / PP 有什么关系？
sched_step_seq / processed_step_seq / last_sched_seq 分别表示什么？
deferred_frees 里保存什么？
_free_request_blocks() 如何判断是否安全？
_drain_deferred_frees() 什么时候执行？
它和 finished_sending / WAITING_FOR_REMOTE_KVS / offloading flush 有什么区别？
```

---

## 3. 最小心智模型

可以把 block 释放分成两步：

```text
1. 从 request bookkeeping 中摘掉：
   req_to_blocks 不再认为这个 request 持有这些 blocks。

2. 还给 BlockPool：
   block.ref_cnt 减少，ref_cnt 为 0 的 block 进入 free queue，之后可被新请求复用。
```

普通同步场景这两步可以连续发生：

```text
request finished
  → kv_cache_manager.free(request)
  → pop req_to_blocks
  → block_pool.free_blocks(...)
```

但异步 / PP / KV consumer 场景下可能存在：

```text
Scheduler 已经决定释放 request A 的 block；
但某个 in-flight GPU step 仍在写 request A 的 KV；
同时 KV consumer 又可能把这些刚释放的 block 分配给 request B，准备 load remote KV；
如果 load 写入和旧 step 写入没有顺序约束，就会发生 block 数据竞态。
```

`deferred free` 的做法是：

```text
request A 的 bookkeeping 先清掉；
但 blocks 暂时不进入 BlockPool free queue；
等对应 in-flight step 的 output 被 update_from_output() 处理后，才真正归还。
```

---

## 4. 为什么会有异步释放竞态

### 4.1 多个 batch 可能同时 in-flight

`max_concurrent_batches` 定义在：`code/vllm/vllm/config/vllm.py:494`

```python
@property
def max_concurrent_batches(self) -> int:
    # PP requires PP-size concurrent batches to fill the pipeline.
    # Async scheduling requires 2 concurrent batches to overlap.
    pp_size = self.parallel_config.pipeline_parallel_size
    if self.scheduler_config.async_scheduling:
        if self.use_v2_model_runner:
            return pp_size + 1
        # V1 Model Runner does not fully support async scheduling with PP.
        if pp_size <= 1:
            return 2
    return pp_size
```

位置：`code/vllm/vllm/config/vllm.py:494` 到 `code/vllm/vllm/config/vllm.py:505`

含义：

```text
PP：为了填满 pipeline，允许 PP-size 个 batch 在不同 stage 中重叠；
async scheduling：为了 schedule / execute 重叠，至少需要 2 个 batch；
V2 + async + PP：可能是 pp_size + 1。
```

只要 `max_concurrent_batches > 1`，Scheduler 就不能再假设：

```text
我处理当前输出时，之前所有 GPU 写都已经完全结束，且不会再碰旧 block。
```

### 4.2 KV consumer 让“释放后立即复用”更危险

Scheduler 初始化 connector 时有注释：

```python
# With overlapping batches (async scheduling or PP), a step may
# still be writing a freed request's KV blocks. A consumer KV
# Connector can reallocate and fill those blocks via a load that
# isn't ordered against that write, so defer freeing them.
multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
    self.defer_block_free = True
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:145` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:151`

这段注释已经把问题说清楚：

```text
旧 step 仍可能写 block X；
Scheduler 如果提前释放 block X；
KV consumer 可能立刻把 block X 分给另一个请求并 load remote KV；
旧 step 写入和 remote KV load 写入之间没有天然 ordering；
结果就是 KV 数据被覆盖或污染。
```

因此 deferred free 只在这个组合下开启：

```text
kv_transfer_config 存在；
当前实例是 KV consumer；
max_concurrent_batches > 1。
```

---

## 5. defer_block_free 什么时候开启

Scheduler 初始化里：

```python
self.defer_block_free = False
kv_transfer_config = self.vllm_config.kv_transfer_config
if kv_transfer_config is not None:
    ...
    multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
    if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
        self.defer_block_free = True
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:128` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:151`

可以整理成表：

| 场景 | `defer_block_free` |
|---|---|
| 没有 KV connector | False |
| KV producer，但不是 consumer | False |
| KV consumer，单 in-flight batch | False |
| KV consumer，async scheduling / PP 导致多 in-flight batch | True |

注意：

```text
deferred free 不是所有 async scheduling 都开启；
它专门保护 KV consumer 场景下 remote KV load 与旧 GPU write 的竞态。
```

---

## 6. deferred free 的核心状态

Scheduler 初始化时创建三个字段：

```python
# Counts of non-empty steps scheduled / processed. update_from_output
# is called once per scheduled step in FIFO order, so these stay in sync.
self.sched_step_seq = 0
self.processed_step_seq = 0
# FIFO of (fence_seq, blocks): blocks become safe to free once
# processed_step_seq >= fence_seq.
self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:292` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:298`

### 6.1 `sched_step_seq`

表示 Scheduler 已经发出去的非空执行 step 序号。

```text
非空 step：total_num_scheduled_tokens > 0；
只有真正会写 KV / 执行模型的 step 才推进 fence。
```

### 6.2 `processed_step_seq`

表示 Scheduler 已经处理完输出的非空 step 序号。

```text
update_from_output() 处理完某个非空 SchedulerOutput 后推进；
意味着这一步以及更早步骤的 GPU 写入已经完成。
```

### 6.3 `request.last_sched_seq`

记录某个 request 最近一次被调度到哪个非空 step。

```text
如果 request.last_sched_seq > processed_step_seq，说明它最近一次调度的 GPU 写还没被 Scheduler 确认完成；
此时不能把它的 blocks 还给 BlockPool。
```

字段定义在 `Request`：

```python
# Seq of the most recent step this request was scheduled in; fences
deferred block freeing (see Scheduler._free_request_blocks).
self.last_sched_seq = 0
```

位置：`code/vllm/vllm/v1/request.py:148` 到 `code/vllm/vllm/v1/request.py:150`

### 6.4 `deferred_frees`

保存：

```text
(fence_seq, blocks)
```

其中：

```text
fence_seq：这些 blocks 最早可以归还的 step 序号；
blocks：已经从 req_to_blocks 摘出来，但还没有 free_blocks() 的 KVCacheBlock 列表。
```

---

## 7. schedule() 如何建立 fence

### 7.1 构造 SchedulerOutput 后推进 sched_step_seq

`Scheduler.schedule()` 末尾构造 `SchedulerOutput` 后：

```python
# Advance the fence only for non-empty steps (those that actually
# write KV and have their output processed later in update_from_output).
if self.defer_block_free and total_num_scheduled_tokens > 0:
    self.sched_step_seq += 1
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1091` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1094`

关键点：

```text
只有 total_num_scheduled_tokens > 0 才推进；
因为 0-token step 不执行模型 forward，也不会写新的 KV。
```

### 7.2 _update_after_schedule 记录 request.last_sched_seq

之后进入 `_update_after_schedule()`：

```python
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
    if self.defer_block_free:
        # Record the in-flight step, to fence deferred block freeing.
        request.last_sched_seq = self.sched_step_seq
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1138` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1144`

这一步建立了 request 与 step fence 的关系：

```text
request 最近一次被安排写 KV 的 step = 当前 sched_step_seq。
```

如果这个 request 在该 step output 回来前被 abort / preempt / 结束，那么 `_free_request_blocks()` 会看到：

```text
request.last_sched_seq > processed_step_seq
```

于是进入 deferred free。

---

## 8. _free_request_blocks() 如何判断是否安全

入口：`code/vllm/vllm/v1/core/sched/scheduler.py:2077`

```python
def _free_request_blocks(self, request: Request):
    """Free the request's KV blocks, deferring the return to the block
    pool when an in-flight GPU step may still write them.
    """
    if not self.defer_block_free or (
        # Last scheduled step already processed: no in-flight write remains
        # (always the case for a normal finish), so free now.
        request.last_sched_seq <= self.processed_step_seq
    ):
        self.kv_cache_manager.free(request)
        return
    blocks = self.kv_cache_manager.pop_blocks_for_free(request)
    if blocks:
        self.deferred_frees.append((self.sched_step_seq, blocks))
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2077` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2090`

它有两条路径。

### 8.1 安全路径：立即释放

满足任一条件就立即释放：

```text
defer_block_free == False；
或 request.last_sched_seq <= processed_step_seq。
```

调用：

```python
self.kv_cache_manager.free(request)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2086`

`KVCacheManager.free()` 最终会释放 request blocks：

```python
def free(self, request: Request) -> None:
    self.coordinator.free(request.request_id)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:460` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:468`

`SingleTypeKVCacheManager.free()` 会：

```python
self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))
```

位置：`code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:399` 到 `code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:407`

### 8.2 不安全路径：先 pop，后延迟归还

如果：

```text
defer_block_free == True；
且 request.last_sched_seq > processed_step_seq。
```

说明该 request 最近调度的 step 还没确认完成。

这时：

```python
blocks = self.kv_cache_manager.pop_blocks_for_free(request)
if blocks:
    self.deferred_frees.append((self.sched_step_seq, blocks))
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2088` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2090`

`pop_blocks_for_free()` 的语义是：

```text
从 request 的 req_to_blocks bookkeeping 中移除 blocks；
但不调用 block_pool.free_blocks()；
调用方必须稍后自己归还。
```

`KVCacheManager.pop_blocks_for_free()`：

```python
def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
    """Pop the request's bookkeeping and return its blocks without
    returning them to the block pool. The caller must eventually free
    them in reverse order (so that tail blocks are evicted first).
    """
    return self.coordinator.pop_blocks_for_free(request.request_id)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:483` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:494`

`SingleTypeKVCacheManager.pop_blocks_for_free()`：

```python
req_blocks = self.req_to_blocks.pop(request_id, [])
self.num_cached_block.pop(request_id, None)
return req_blocks
```

位置：`code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:381` 到 `code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:397`

这就是 deferred free 的核心：

```text
request 已经不再持有这些 blocks；
但 blocks 也还没进入 BlockPool free queue；
因此不会被新请求重新分配。
```

---

## 9. _drain_deferred_frees() 什么时候释放

### 9.1 update_from_output 确认 GPU 写完成

`Scheduler.update_from_output()` 开头：

```python
# Every GPU write enqueued by this and earlier steps has completed, so it is
# safe to return deferred-free blocks to the pool.
if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
    self.processed_step_seq += 1
    self._drain_deferred_frees()
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1477` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1481`

含义：

```text
只要某个非空 SchedulerOutput 的 ModelRunnerOutput 回来了，说明这个 step 的 GPU 写已经完成；
processed_step_seq 前进；
所有 fence <= processed_step_seq 的 deferred blocks 都可以归还。
```

### 9.2 _drain_deferred_frees 按 FIFO drain

```python
def _drain_deferred_frees(self):
    """Return deferred blocks whose fence step has completed.

    Entries are appended with monotonically non-decreasing fences, so
    stop at the first one that is still pending.
    """
    while self.deferred_frees:
        fence, _ = self.deferred_frees[0]
        if fence > self.processed_step_seq:
            break
        _, blocks = self.deferred_frees.popleft()
        # Free in reverse order so that the tail blocks are evicted first.
        self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2092` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2104`

这里有几个关键细节：

```text
1. deferred_frees 是 FIFO；
2. fence 单调不下降；
3. 遇到第一个未完成 fence 就停止；
4. 真正归还时直接调用 block_pool.free_blocks(reversed(blocks))；
5. 不再经过 req_to_blocks，因为 request bookkeeping 已经在 pop_blocks_for_free() 中移除了。
```

---

## 10. BlockPool.free_blocks 真正做什么

入口：`code/vllm/vllm/v1/core/block_pool.py:614`

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
```

核心逻辑：

```python
blocks_with_hash = []
blocks_without_hash = []
for block in ordered_blocks:
    block.ref_cnt -= 1
    if block.ref_cnt == 0 and not block.is_null:
        if block.block_hash is None:
            blocks_without_hash.append(block)
        else:
            blocks_with_hash.append(block)

# Blocks without hash always get evicted first - prepend them last to the tail
self.free_block_queue.prepend_n(blocks_without_hash)
self.free_block_queue.append_n(blocks_with_hash)
```

位置：`code/vllm/vllm/v1/core/block_pool.py:622` 到 `code/vllm/vllm/v1/core/block_pool.py:635`

这说明：

```text
free_blocks 只是减少 ref_cnt；
只有 ref_cnt 归零，且不是 null block，才进入 free queue；
带 block_hash 的 cached blocks 会保留 hash，可作为 prefix cache eviction candidate；
不带 hash 的 blocks 更优先被重新分配。
```

因此 deferred free 延迟的是：

```text
ref_cnt-- 和进入 free queue 的时间。
```

它不是简单延迟删除 Python 对象，而是延迟 block 可复用性。

---

## 11. 哪些路径会触发 _free_request_blocks()

### 11.1 正常请求结束

`update_from_output()` 检测 stopped 后：

```python
if stopped:
    finished = self._handle_stopped_request(request)
    if finished:
        kv_transfer_params = self._free_request(request)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1655` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1663`

`_free_request()` 中：

```python
if not delay_free_blocks:
    self._free_blocks(request)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2059` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2061`

`_free_blocks()` 会调用：

```python
self._free_request_blocks(request)
del self.requests[request.request_id]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2065` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2068`

### 11.2 外部 abort / finish_requests

`finish_requests()` 设置状态后调用：

```python
request.status = finished_status
self._free_request(request, delay_free_blocks=delay_free_blocks)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2041` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2042`

如果请求正在 `WAITING_FOR_REMOTE_KVS`，可能会额外设置 `delay_free_blocks=True`。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2033` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2039`

### 11.3 preemption

`_preempt_request()` 中：

```python
self._free_request_blocks(request)
self.encoder_cache_manager.free(request)
self._inflight_prefills.discard(request)
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1105` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1118`

这说明 preemption 也会释放 request 当前 KV blocks。

如果该 request 最近调度的 step 还没完成，释放同样会进入 deferred free。

### 11.4 KV transfer 完成后的 delayed request 释放

`_update_from_kv_xfer_finished()` 中，`finished_sending` 会触发：

```python
for req_id in kv_connector_output.finished_sending or ():
    logger.debug("Finished sending KV transfer for request %s", req_id)
    assert req_id in self.requests
    self._free_blocks(self.requests[req_id])
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2441` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2444`

`finished_recving` 如果对应请求已经 finished，也会：

```python
assert RequestStatus.is_finished(req.status)
self._free_blocks(self.requests[req_id])
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2438` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2440`

这意味着：

```text
connector delayed-free 完成后，最终释放仍然会经过 _free_request_blocks()；
如果 step fence 仍未满足，依旧可能进入 deferred_frees。
```

---

## 12. deferred free 和 connector delayed-free 的区别

这两个概念很容易混淆。

### 12.1 connector delayed-free

来源：

```text
connector.request_finished*() 返回 delay_free_blocks=True。
```

作用：

```text
请求结束后，connector 还要异步 save/send 这个 request 的 KV blocks；
Scheduler 暂时不调用 _free_blocks()；
request 继续留在 self.requests；
等 Worker 返回 finished_sending / finished_recving 后再释放。
```

典型：

```text
NIXL push P-side WRITE。
```

### 12.2 deferred free

来源：

```text
_free_request_blocks() 判断 request.last_sched_seq > processed_step_seq。
```

作用：

```text
Scheduler 已经决定释放 blocks；
request bookkeeping 已经 pop；
但 blocks 暂时不还给 BlockPool；
等对应 in-flight GPU step output 被处理后再 free_blocks()。
```

### 12.3 二者关系

顺序上可能叠加：

```text
request finished
  → connector_delay_free_blocks=True
  → 暂时不进入 _free_blocks()
  → later finished_sending
  → _free_blocks()
      → _free_request_blocks()
          → 如果 step fence 未完成，再进入 deferred_frees
```

所以：

```text
connector delayed-free：推迟“开始释放 request blocks”；
deferred free：推迟“把已摘出的 blocks 归还 BlockPool”。
```

---

## 13. deferred free 和 WAITING_FOR_REMOTE_KVS 的关系

异步 external KV load 会让请求进入：

```text
RequestStatus.WAITING_FOR_REMOTE_KVS
```

调度时：

```python
if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    step_skipped_waiting.prepend_request(request)
    request.num_computed_tokens = num_computed_tokens
    self._inflight_prefills.add(request)
    continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:916` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:937`

如果这个请求在 remote KV load 完成前被 abort：

```python
if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    delay_free_blocks = (
        request.request_id not in self.finished_recving_kv_req_ids
    )
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2033` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2037`

含义：

```text
remote KV load 还没完成时，也不能释放目标 blocks；
否则 worker / connector 可能还在把远端 KV 写入这些 blocks。
```

load 完成后：

```python
for req_id in kv_connector_output.finished_recving or ():
    if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        self.finished_recving_kv_req_ids.add(req_id)
    else:
        assert RequestStatus.is_finished(req.status)
        self._free_blocks(self.requests[req_id])
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2431` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2440`

这里和 deferred free 的关系是：

```text
WAITING_FOR_REMOTE_KVS 的 delay_free_blocks 保护的是“remote load 正在写”；
deferred free 保护的是“旧 GPU step 可能仍在写”；
两者都可能阻止 blocks 过早进入 BlockPool。
```

---

## 14. deferred free 和 offloading flush 的区别

Offloading connector 还有另一套安全机制：

```text
_block_id_to_pending_jobs
_current_batch_jobs_to_flush
jobs_to_flush
```

`OffloadingConnectorScheduler.build_connector_meta()` 中：

```python
# Flush jobs that contain re-allocated blocks.
if (
    self._block_id_to_pending_jobs
    and not self._block_id_to_pending_jobs.keys().isdisjoint(
        self._current_batch_allocated_block_ids
    )
):
    self._current_batch_jobs_to_flush.update(
        jid
        for bid in self._current_batch_allocated_block_ids
        if bid in self._block_id_to_pending_jobs
        for jid in self._block_id_to_pending_jobs[bid]
    )
```

位置：`code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1005` 到 `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1017`

这个机制解决的是：

```text
某个 offload store job 还没完成；
它依赖的 block 即将被重新分配；
那就让 worker flush 对应 job，确保 store 完成后再复用。
```

它和 deferred free 的区别：

```text
deferred free：阻止 block 进入 BlockPool free queue；
offloading flush：block 可能已经要被复用，但复用前强制完成依赖它的 store job。
```

二者可以共同存在，但保护的竞态不同。

---

## 15. 为什么 _drain_deferred_frees 在 update_from_output 开头执行

`update_from_output()` 一开始就：

```python
if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
    self.processed_step_seq += 1
    self._drain_deferred_frees()
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1477` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1481`

原因是：

```text
拿到 model_runner_output，说明这个 scheduler_output 对应的 worker 执行已经结束；
此时该 step 写过的 KV blocks 不会再被旧 step 写入；
可以尽早归还满足 fence 的 blocks，增加本轮后续调度可用 block 数。
```

注意：

```text
update_from_output 是按 scheduled step FIFO 调用；
sched_step_seq / processed_step_seq 才能用简单单调计数作为 fence。
```

源码注释：

```text
update_from_output is called once per scheduled step in FIFO order, so these stay in sync.
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:292` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:293`

---

## 16. 为什么 append 的 fence 用 sched_step_seq

不安全释放时：

```python
self.deferred_frees.append((self.sched_step_seq, blocks))
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2088` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2090`

直觉上可能会问：为什么不用 `request.last_sched_seq`？

源码使用 `sched_step_seq` 的效果是更保守：

```text
如果 request 最近写入在 step N；
当前 Scheduler 已经发出了 step M（M >= N）；
把 fence 设为 M，意味着等到当前已发出的更晚 step 也 processed 后再归还；
这样不会早于 request.last_sched_seq，安全但可能稍微晚一点释放。
```

因此时间线示例里的 `deferred_frees.append((1, A_blocks))` 只对应“当前已发出的最新非空 step 仍是 1”的情况；如果释放发生时 Scheduler 已经发出后续非空 step，实际 fence 会是更大的 `sched_step_seq`。

这和注释保持一致：

```text
FIFO of (fence_seq, blocks): blocks become safe to free once processed_step_seq >= fence_seq.
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:296` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:298`

---

## 17. 常见时间线示例

### 17.1 单 batch，同步执行

```text
defer_block_free = False

step 1 schedule request A
step 1 output returns
request A finished
_free_request_blocks(A)
  → kv_cache_manager.free(A)
  → blocks 立即进入 BlockPool
```

没有 deferred free。

### 17.2 KV consumer + async scheduling，abort in-flight 请求

```text
defer_block_free = True
processed_step_seq = 0

schedule step 1:
  sched_step_seq = 1
  request A.last_sched_seq = 1
  Worker 正在执行 step 1

用户 abort A:
  finish_requests(A)
  _free_request_blocks(A)
  A.last_sched_seq=1 > processed_step_seq=0
  → pop_blocks_for_free(A)
  → deferred_frees.append((1, A_blocks))
  → A_blocks 不在 req_to_blocks，也不在 BlockPool free queue

step 1 output returns:
  update_from_output(step 1)
  processed_step_seq = 1
  _drain_deferred_frees()
  fence=1 <= processed_step_seq=1
  → block_pool.free_blocks(A_blocks)
```

核心安全点：

```text
在 step 1 output 回来前，A_blocks 不会被 request B 的 external KV load 复用。
```

### 17.3 PP 场景下 preempt running 请求

```text
PP 允许多个 batch 处在不同 pipeline stage；
request A 被 preempt；
_preempt_request(A)
  → _free_request_blocks(A)
  → 如果 A 最近 step 还没 processed，则进入 deferred_frees；
后续按 processed_step_seq drain。
```

这避免：

```text
前一 pipeline stage 还可能写 A 的 KV；
但 block 已经被后续 batch 复用。
```

### 17.4 connector delayed-free 后再 deferred free

```text
request A finished
connector.request_finished*() 返回 delay_free_blocks=True
Scheduler 暂不 _free_blocks(A)

later Worker 返回 finished_sending={A}
_update_from_kv_xfer_finished()
  → _free_blocks(A)
  → _free_request_blocks(A)
      → 如果 A.last_sched_seq 仍未 processed，则进入 deferred_frees
      → 否则立即归还 BlockPool
```

这说明：

```text
finished_sending 只是 connector transfer 完成；
最终能否立刻归还 BlockPool，还要看 step fence。
```

---

## 18. 容易疑惑的点

### 18.1 deferred free 是否会删除 request？

`_free_blocks()` 会删除 request：

```python
self._free_request_blocks(request)
del self.requests[request.request_id]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2065` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2068`

即使 blocks 进入 `deferred_frees`，request 也会从 `self.requests` 删除。

所以：

```text
request 生命周期可以结束；
block 归还 BlockPool 可以延后。
```

### 18.2 deferred_frees 里的 blocks 还属于 request 吗？

不属于。

`pop_blocks_for_free()` 已经把它们从 `req_to_blocks` 中移除。

它们处于中间状态：

```text
不属于 request；
也不在 BlockPool free queue；
只由 deferred_frees 持有，等待 fence 完成。
```

### 18.3 为什么不直接等 output 回来再释放 request？

因为 request 的调度状态、队列状态、输出回收可以先完成。

只要 block 不被复用，就能避免 KV 数据竞态。

```text
deferred free 只延迟资源归还，不延迟整个请求状态机。
```

### 18.4 0-token step 会推进 fence 吗？

不会。

`schedule()` 里：

```python
if self.defer_block_free and total_num_scheduled_tokens > 0:
    self.sched_step_seq += 1
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1091` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1094`

`update_from_output()` 里也只有非空 step 才：

```python
if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
    self.processed_step_seq += 1
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1477` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1481`

因为 0-token step 不写 KV。

### 18.5 deferred free 是否会减少可用 blocks，导致调度更保守？

会短暂减少。

进入 `deferred_frees` 的 blocks 不在 free queue 中，因此 `block_pool.get_num_free_blocks()` 看不到它们。

这是有意的：

```text
宁愿短暂少用一些 blocks，也不能把仍可能被旧 GPU step 写入的 blocks 分给新请求。
```

### 18.6 为什么 free 时要 reversed(blocks)？

`KVCacheManager.free()` 注释：

```text
We free the blocks in reverse order so that the tail blocks are evicted first when caching is enabled.
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:460` 到 `code/vllm/vllm/v1/core/kv_cache_manager.py:468`

`_drain_deferred_frees()` 也保持同样规则：

```python
self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2102` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2104`

### 18.7 has_requests 为什么还要看 finished requests / pending push work？

Scheduler 里：

```python
return (
    self.has_unfinished_requests()
    or self.has_finished_requests()
    or (self.connector is not None and self.connector.has_pending_push_work())
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2130` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2141`

原因：

```text
请求可能已经 finished，但 connector delayed-free 还没结束；
也可能 NIXL push 还有 pending work；
engine loop 必须继续 step，Worker 才能推进 transfer 并返回 finished_sending/finished_recving。
```

这不是 deferred free 本身，但会影响 delayed blocks 最终能不能被释放。

---

## 19. 总结

`deferred free` 主链路可以压缩成：

```text
KV consumer + 多 in-flight batch
  → defer_block_free=True
  → schedule 非空 step 时 sched_step_seq++
  → scheduled request 记录 last_sched_seq
  → request finished / preempted / transfer done 时尝试 free
  → 如果 last_sched_seq 尚未 processed：
        pop_blocks_for_free()
        deferred_frees.append((fence, blocks))
    否则：
        kv_cache_manager.free()
  → update_from_output 非空 step 时 processed_step_seq++
  → drain fence <= processed_step_seq 的 blocks
  → block_pool.free_blocks(reversed(blocks))
```

最关键的边界是：

```text
deferred free 不是“请求不结束”；
也不是“connector save 还没完成”；
它是“blocks 已经从 request 中摘掉，但暂时不能进入 BlockPool free queue”。
```

如果只记一句话：

```text
在 KV consumer 的 async / PP 场景下，deferred free 用 step fence 保证旧 in-flight GPU 写完成之后，才把已释放请求的 KV blocks 重新交给 BlockPool，从而避免 remote KV load 和旧 batch 写入复用同一 block 的竞态。
```
