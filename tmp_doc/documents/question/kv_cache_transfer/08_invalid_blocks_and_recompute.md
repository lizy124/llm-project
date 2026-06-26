# 08. 外部 KV load 失败后如何回退重算？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\outputs.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\kv_connector_model_runner_mixin.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\base.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\nixl\connector.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\nixl\base_worker.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\v1\lmcache_connector.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\distributed\kv_transfer\kv_connector\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\kv_transfer.py`
- `D:\lzy\project\kv_pool\code\vllm\tests\v1\kv_connector\unit\test_kv_load_failure_recovery.py`
- `D:\lzy\project\kv_pool\code\vllm\tests\v1\kv_connector\unit\test_error_propagation.py`

本问题关注：外部 KV load 失败时，Worker connector 如何把失败 block ids 回传给 Scheduler；Scheduler 如何区分 async load 和 sync load；如何根据 block id 找到受影响请求和 token 位置；`num_computed_tokens` 为什么可以回退到第一个失败 block 之前；`kv_load_failure_policy=recompute/fail` 如何影响后续请求状态；以及为什么 async load 失败仍要等 `finished_recving` 才做最终提交。

---

## 1. 一句话回答

外部 KV load 失败时，Worker 不直接决定请求失败或重算，而是把失败的本地 KV block id 放进：

```text
ModelRunnerOutput.kv_connector_output.invalid_block_ids
```

Scheduler 在 `update_from_output()` 一开始处理这些 block ids：

```text
invalid block id
  → 找到引用这些 block 的请求
  → 定位第一个失败 block 的 block index
  → request.num_computed_tokens 回退到 block_index * block_size
  → 根据 kv_load_failure_policy 决定：
      recompute：后续重新本地计算失败 block 及其后续 tokens
      fail：请求直接 FINISHED_ERROR
```

最小心智模型：

```text
invalid_block_ids 是 Worker 对“这些外部 KV block 不可信”的回报；Scheduler 通过回退 num_computed_tokens，把不可信的 KV 从“已 computed”集合中剔除，让后续调度重新计算或直接失败。
```

---

## 2. 总体主链路

完整链路可以压缩成：

```text
Worker connector load 外部 KV
  → 某些本地 block load 失败
  → connector.get_block_ids_with_load_errors()
  → KVConnectorOutput.invalid_block_ids
  → ModelRunnerOutput.kv_connector_output
  → Scheduler.update_from_output()
  → _handle_invalid_blocks(invalid_block_ids, num_scheduled_tokens)
  → 扫描 async WAITING_FOR_REMOTE_KVS 请求
  → 扫描 sync running 请求
  → _update_requests_with_invalid_blocks()
  → request.num_computed_tokens 回退
  → failure_policy == recompute：保留/恢复请求，后续重算
  → failure_policy == fail：finish_requests(... FINISHED_ERROR)
```

其中最核心的分界是：

```text
async load：请求在 skipped_waiting，状态是 WAITING_FOR_REMOTE_KVS；
sync load：请求已经在 running，本轮也出现在 SchedulerOutput.num_scheduled_tokens 中。
```

---

## 3. invalid_block_ids 是什么

`invalid_block_ids` 定义在：`vllm/v1/outputs.py:196`

字段说明：

```text
IDs of externally computed KV blocks that failed to load.
Requests referencing these blocks should be rescheduled to recompute them.
```

位置：`outputs.py:203` 到 `outputs.py:205`

它表示的是：

```text
本地 GPU KV block id。
```

不是：

```text
token id；
远端 block id；
request id；
hash key。
```

Scheduler 需要把这些本地 block id 反查到请求当前持有的 block table 中，从而推导失败 token 位置。

---

## 4. invalid_block_ids 从哪里来

Worker / ModelRunner 侧统一采集点在：

```python
KVConnectorModelRunnerMixin._get_kv_connector_output()
```

位置：`kv_connector_model_runner_mixin.py:77`

在 finally 中：

```python
output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
```

位置：`kv_connector_model_runner_mixin.py:102` 到 `kv_connector_model_runner_mixin.py:106`

这说明：

```text
具体 connector 负责发现 load 错误；
ModelRunner 只负责把错误 block ids 塞进 KVConnectorOutput。
```

---

## 5. connector 抽象接口如何定义 load failure

接口在：`kv_connector/v1/base.py:375`

```python
def get_block_ids_with_load_errors(self) -> set[int]:
```

接口说明强调：

```text
- 适用于 sync loading 和 async loading；
- async loading 的失败 block 可以在任意 forward pass 中报告；
- 最迟必须在该请求通过 get_finished() 返回 finished_recving 的同一 pass 报告；
- 即使失败，请求也必须通过 get_finished() 报告完成；
- sync loading 的失败应在发现失败的那个 forward pass 报告。
```

位置：`kv_connector/v1/base.py:375` 到 `kv_connector/v1/base.py:393`

这几个约束很重要：

```text
invalid_block_ids 告诉 Scheduler “哪些 block 不可信”；
finished_recving 告诉 Scheduler “这次 async transfer 已经收尾，可以推进状态”。
```

二者不是互斥关系，async load 失败时通常两者都需要出现，只是可能不在同一轮出现。

---

## 6. NIXL 如何收集失败 block

NIXL facade：`nixl/connector.py:244`

```python
def get_block_ids_with_load_errors(self) -> set[int]:
    return self.connector_worker.get_block_ids_with_load_errors()
```

Worker 实现在：`nixl/base_worker.py:2191`

```text
1. NIXL worker 内部维护 _invalid_block_ids 队列；
2. load / postprocess 失败时把本地 block ids 放入该队列；
3. get_block_ids_with_load_errors() drain 队列并返回 set[int]。
```

位置：`nixl/base_worker.py:2191` 到 `nixl/base_worker.py:2205`

关键点：

```text
返回后会清空这批 invalid block ids，避免重复上报。
```

---

## 7. LMCache / FlexKV / MultiConnector 如何处理

不同 connector 的实现不同，但都遵循同一个抽象。

### 7.1 LMCache

`LMCacheConnector.get_block_ids_with_load_errors()`：`lmcache_connector.py:215`

它会尝试调用底层 engine 的：

```text
get_block_ids_with_load_errors()
```

如果底层旧版本没有这个方法，则返回空 set。

位置：`lmcache_connector.py:215` 到 `lmcache_connector.py:228`

### 7.2 FlexKV

`FlexKVConnector.get_block_ids_with_load_errors()`：`flexkv_connector.py:258`

直接委托给底层 FlexKV connector。

### 7.3 MultiConnector / KVOutputAggregator

多 worker / 多 connector 场景会合并 invalid ids。

`kv_connector/utils.py` 中聚合逻辑：

```python
invalid_block_ids |= kv_output.invalid_block_ids
```

最终写入聚合后的：

```text
KVConnectorOutput.invalid_block_ids
```

位置：`kv_connector/utils.py:154` 到 `kv_connector/utils.py:168`

也就是说，只要任意 worker / connector 报告某个 block 失败，Scheduler 就会看到这个 block id。

---

## 8. Scheduler 什么时候处理 invalid blocks

处理入口在：

```python
Scheduler.update_from_output(scheduler_output, model_runner_output)
```

位置：`scheduler.py:1463`

它在主循环处理每个请求输出之前，先处理 invalid blocks：

```python
failed_kv_load_req_ids = None
if kv_connector_output and kv_connector_output.invalid_block_ids:
    failed_kv_load_req_ids = self._handle_invalid_blocks(
        kv_connector_output.invalid_block_ids,
        num_scheduled_tokens,
    )
```

位置：`scheduler.py:1490` 到 `scheduler.py:1498`

为什么要这么早？

```text
因为一旦发现某些 externally computed KV 不可信，
后续对这些请求的 sampled tokens / stop / output 更新都不能按正常路径继续处理。
```

后面主循环里会跳过受影响请求：

```python
if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
    continue
```

位置：`scheduler.py:1526` 到 `scheduler.py:1530`

---

## 9. failure policy 在哪里配置

配置字段：

```python
KVTransferConfig.kv_load_failure_policy: Literal["recompute", "fail"] = "fail"
```

位置：`config/kv_transfer.py:69`

含义：

```text
recompute：回退请求进度，后续重算失败 blocks；
fail：直接把受影响请求标记为 FINISHED_ERROR。
```

Scheduler 初始化时读取：

```python
kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"
```

位置：`scheduler.py:142` 到 `scheduler.py:143`

如果没有 KV connector，默认值虽然初始化为 True，但这条逻辑不会实际触发。

---

## 10. _handle_invalid_blocks() 的职责

入口：`scheduler.py:2549`

它负责：

```text
1. 根据 policy 判断 should_fail；
2. 处理 async load 请求；
3. 处理 sync load 请求；
4. 汇总失败请求数和受影响 token 数；
5. fail policy 下返回所有失败请求 id；
6. recompute policy 下标记 async failed req，返回 sync failed req。
```

核心变量：

```text
should_fail = not self.recompute_kv_load_failures
```

位置：`scheduler.py:2558`

---

## 11. async load 失败如何扫描

async load 请求不在 `self.running`，而是在：

```text
self.skipped_waiting
```

并且状态是：

```text
RequestStatus.WAITING_FOR_REMOTE_KVS
```

`_handle_invalid_blocks()` 先构造：

```python
async_load_reqs = (
    req
    for req in self.skipped_waiting
    if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
)
```

位置：`scheduler.py:2560` 到 `scheduler.py:2565`

然后调用：

```python
_update_requests_with_invalid_blocks(
    async_load_reqs,
    invalid_block_ids,
    num_scheduled_tokens,
    evict_blocks=False,
)
```

位置：`scheduler.py:2566` 到 `scheduler.py:2573`

为什么 `evict_blocks=False`？

```text
因为 async load 的 blocks 还没有 cache_blocks()，
还没有正式进入本地 prefix cache，
只需要回退请求进度，不需要从 prefix cache hash map 驱逐。
```

---

## 12. sync load 失败如何扫描

sync load 请求已经进入：

```text
self.running
```

`_handle_invalid_blocks()` 对 running 请求再次调用：

```python
_update_requests_with_invalid_blocks(
    self.running,
    invalid_block_ids,
    num_scheduled_tokens,
    evict_blocks=True,
)
```

位置：`scheduler.py:2578` 到 `scheduler.py:2583`

为什么 `evict_blocks=True`？

```text
sync load 的 blocks 可能已经被 cache，
如果后续 policy=fail，就要把失败 block 及其后续依赖 blocks 从 prefix cache 中驱逐，
避免污染后续请求。
```

实际驱逐在：

```python
if sync_blocks_to_evict and not self.recompute_kv_load_failures:
    self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)
```

位置：`scheduler.py:2591` 到 `scheduler.py:2595`

---

## 13. 如何根据 block id 定位失败 token 位置

核心函数：

```python
_update_requests_with_invalid_blocks(...)
```

位置：`scheduler.py:2446`

它对每个请求做：

```python
(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
req_num_computed_tokens = request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
req_num_computed_blocks = (req_num_computed_tokens + block_size - 1) // block_size
```

位置：`scheduler.py:2488` 到 `scheduler.py:2498`

然后遍历：

```python
for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
    if block_id not in invalid_block_ids:
        continue
```

位置：`scheduler.py:2499` 到 `scheduler.py:2501`

一旦发现第一个失败 block：

```python
request.num_computed_tokens = idx * self.block_size
```

位置：`scheduler.py:2521` 到 `scheduler.py:2524`

也就是说：

```text
失败 block index = idx
安全 computed token 数 = idx * block_size
```

因为第 idx 个 block 里的 KV 已经不可信，所以只能保留它之前的完整 blocks。

---

## 14. 为什么要减掉本轮 num_scheduled_tokens

计算失败影响范围时使用：

```python
req_num_computed_tokens = request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
```

位置：`scheduler.py:2492` 到 `scheduler.py:2494`

原因是：

```text
update_from_output() 执行时，request.num_computed_tokens 可能已经在 schedule() 后被提前推进；
但 invalid blocks 针对的是外部 computed KV 的已加载部分，
不应该把本轮刚 schedule 的新 token 也算进外部 load 失败定位里。
```

所以要扣掉本轮 scheduled tokens，得到：

```text
进入本轮 forward 之前，被认为已经 computed 的外部 / prefix 范围。
```

---

## 15. 为什么可以回退 num_computed_tokens 后重算

vLLM Scheduler 的基本模型是：

```text
request.num_computed_tokens 表示该请求已有多少 token 的 KV 可以被信任并复用。
```

如果某个外部 block load 失败，那么从这个 block 开始的 KV 都不能再信任。

所以回退到：

```text
第一个失败 block 的起始 token
```

就等价于告诉 Scheduler：

```text
从这里开始还没算过，下一次 schedule 时重新分配/调度这些 token。
```

后续调度会按普通逻辑计算：

```text
num_new_tokens = request.num_tokens - request.num_computed_tokens
```

这样失败部分自然回到本地 prefill / decode 路径。

---

## 16. downstream dependent blocks 为什么也要处理

如果 block i 失败，那么 i 之后的 blocks 也不能直接信任。

原因是 prefix cache block hash 是链式依赖：

```text
block_hash[i] = hash(parent_block_hash[i-1], block_tokens[i], extra_keys)
```

后续 block 的语义依赖前缀成立。

所以 sync load 且需要 evict 时，会收集：

```python
blocks_to_evict.update(req_block_ids[idx:])
```

位置：`scheduler.py:2529` 到 `scheduler.py:2531`

这表示：

```text
失败 block 以及它后面所有依赖 block 都要被视为不可信。
```

---

## 17. shared block 如何避免重复重算

`_update_requests_with_invalid_blocks()` 使用：

```python
marked_invalid_block_ids: set[int] = set()
```

位置：`scheduler.py:2479` 到 `scheduler.py:2483`

如果多个请求共享同一个 invalid block：

```text
第一个请求会被标记为从该 block 开始重算；
后续请求遇到同一个 block 时，知道它已经由前面的请求负责重算。
```

对应逻辑：

```python
if block_id in marked_invalid_block_ids:
    continue
```

位置：`scheduler.py:2505` 到 `scheduler.py:2512`

如果一个请求的 invalid blocks 全部都是“已由前面请求标记”的共享 block，则它会回退到：

```python
request.num_computed_tokens = req_num_computed_tokens
```

位置：`scheduler.py:2533` 到 `scheduler.py:2544`

源码注释说明：

```text
目前这个共享 block 优化只适用于 sync loading；async loading does not yet support block sharing。
```

位置：`scheduler.py:2508` 到 `scheduler.py:2511`，`scheduler.py:2537` 到 `scheduler.py:2539`

---

## 18. recompute policy 下 async load 失败如何恢复

当：

```text
kv_load_failure_policy = recompute
```

`_handle_invalid_blocks()` 不会直接失败请求，而是：

```python
self.failed_recving_kv_req_ids |= async_failed_req_ids
return sync_failed_req_ids
```

位置：`scheduler.py:2615` 到 `scheduler.py:2618`

对 async load 来说，请求仍然处在：

```text
WAITING_FOR_REMOTE_KVS
```

它必须等 connector 报告：

```text
finished_recving
```

之后才进入 `_update_waiting_for_remote_kv()` 提交状态。

---

## 19. failed_recving_kv_req_ids 的作用

`failed_recving_kv_req_ids` 是 Scheduler 记录 async load 失败请求的集合。

定义位置：`scheduler.py:195` 到 `scheduler.py:198`

它的作用是：

```text
在 invalid block 已经使 num_computed_tokens 回退后，
等该请求后续 finished_recving 时，
走失败提交路径，而不是把全部外部 tokens 当成成功 load 并 cache。
```

提交逻辑在 `_update_waiting_for_remote_kv()`：`scheduler.py:2350`

如果请求在 `failed_recving_kv_req_ids` 中：

```text
1. request.num_computed_tokens > 0：
   cache 有效前缀 blocks；

2. request.num_computed_tokens == 0：
   释放该请求已经分配的 blocks；

3. 从 failed_recving_kv_req_ids 移除。
```

位置：`scheduler.py:2360` 到 `scheduler.py:2371`

这避免了一个严重错误：

```text
load 失败后还把原本的 external computed tokens 全部 cache 成可复用 prefix。
```

---

## 20. recompute policy 下 sync load 失败如何恢复

sync load 请求在 `running` 中。

如果发生 invalid block：

```text
_update_requests_with_invalid_blocks()
  → request.num_computed_tokens 回退
  → sync_failed_req_ids 返回给 update_from_output()
```

`update_from_output()` 主循环遇到这些请求会：

```text
continue
```

位置：`scheduler.py:1526` 到 `scheduler.py:1530`

这表示：

```text
本轮这个请求的 sampled tokens / stop 判断 / output 生成全部跳过，
请求保留在 running 中，下一轮按回退后的 num_computed_tokens 继续调度。
```

单测 `test_sync_load_failure()` 验证了：

```text
失败请求保留在 scheduler.running；
num_computed_tokens == min_invalid_block_idx * block_size；
其他未失败请求正常完成。
```

位置：`tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py:122` 起

---

## 21. fail policy 下会发生什么

当：

```text
kv_load_failure_policy = fail
```

Scheduler 初始化得到：

```text
self.recompute_kv_load_failures = False
```

`_handle_invalid_blocks()` 中：

```python
should_fail = True
return async_failed_req_ids | sync_failed_req_ids
```

位置：`scheduler.py:2558` 到 `scheduler.py:2606`

回到 `update_from_output()` 后：

```python
if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
    requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
    self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
    ... 生成 EngineCoreOutput(finish_reason=ERROR)
```

位置：`scheduler.py:1717` 到 `scheduler.py:1729`

单测覆盖：

```text
test_error_propagation_sync_load()
test_error_propagation_async_load()
```

位置：`tests/v1/kv_connector/unit/test_error_propagation.py:41` 和 `test_error_propagation.py:95`

预期结果是：

```text
request.status == FINISHED_ERROR
request.get_finished_reason() == FinishReason.ERROR
输出中 finish_reason == ERROR
```

---

## 22. fail policy 下为什么要 evict sync blocks

sync load 失败时，某些失败 blocks 可能已经进入本地 prefix cache。

如果 policy 是 fail，Scheduler 不会通过 recompute 修复这些 blocks，因此必须把它们从 prefix cache 中移除。

对应逻辑：

```python
if sync_blocks_to_evict and not self.recompute_kv_load_failures:
    self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)
```

位置：`scheduler.py:2591` 到 `scheduler.py:2595`

这里 evict 的范围是：

```text
失败 block + 后续 dependent blocks
```

这样可以防止：

```text
后续请求命中已经失败 / 不可信的 prefix cache block。
```

---

## 23. recompute policy 下为什么不立即 evict sync blocks

源码注释：

```text
evict invalid blocks and downstream dependent blocks from cache
only when not using recompute policy
(where blocks will be recomputed and reused by other requests sharing them)
```

位置：`scheduler.py:2591` 到 `scheduler.py:2594`

含义是：

```text
recompute policy 下，失败 block 所在请求会重新计算；
如果有共享请求，重算结果可继续修复/复用这条链；
因此不直接 evict，避免破坏共享重算路径。
```

这个逻辑主要服务 sync load + shared blocks 场景。

---

## 24. async load 失败为什么不能马上 free blocks

async load 失败请求仍处于：

```text
WAITING_FOR_REMOTE_KVS
```

Worker connector 可能还在完成 transfer 或收尾。

即使已经上报 `invalid_block_ids`，也必须等：

```text
finished_recving
```

原因是 connector 抽象要求：

```text
Async loading: failed blocks may be reported in any forward pass up to and including
where the request ID is returned by get_finished(); even if failures occur, the request
must still be reported via get_finished().
```

位置：`kv_connector/v1/base.py:383` 到 `kv_connector/v1/base.py:389`

所以 Scheduler 的动作分两段：

```text
invalid_block_ids 到达：
  回退 num_computed_tokens，记录 failed_recving_kv_req_ids。

finished_recving 到达：
  _update_waiting_for_remote_kv() 提交有效前缀或释放 blocks。
```

---

## 25. async load 失败后的下一轮调度

完成 `finished_recving` 后，下一轮 `schedule()` 会尝试 promote：

```text
_try_promote_blocked_waiting_request()
  → _update_waiting_for_remote_kv()
  → status = WAITING 或 PREEMPTED
```

位置：`scheduler.py:2384` 到 `scheduler.py:2399`

如果失败导致 `request.num_computed_tokens` 回退到 0：

```text
_update_waiting_for_remote_kv()
  → kv_cache_manager.free(request)
```

位置：`scheduler.py:2363` 到 `scheduler.py:2369`

这样后续请求会像普通新请求一样重新走：

```text
local prefix cache lookup
external KV lookup
allocate_slots
```

如果还有部分有效前缀：

```text
cache_blocks(request, request.num_computed_tokens)
```

后续只重算失败 block 之后的 token。

---

## 26. sync load 和 async load 的对比

```text
sync load 失败：
  请求已经在 running；
  invalid blocks 在 update_from_output 当前轮处理；
  recompute 下请求留在 running，跳过本轮输出处理；
  fail 下请求 FINISHED_ERROR；
  可能需要 evict prefix cache blocks。

async load 失败：
  请求在 skipped_waiting / WAITING_FOR_REMOTE_KVS；
  invalid blocks 先回退 num_computed_tokens；
  recompute 下记录 failed_recving_kv_req_ids；
  仍等待 finished_recving；
  promote 时 cache 有效前缀或释放 blocks。
```

两者共同点：

```text
都用 invalid block 在 req_block_ids 中的 index 来回退 num_computed_tokens。
```

不同点：

```text
sync load 失败发生在正在执行的一轮；
async load 失败发生在等待远端 KV 的 blocked 请求上。
```

---

## 27. 与 finished_recving 的关系

`finished_recving` 在：

```text
Scheduler._update_from_kv_xfer_finished()
```

中处理。

位置：`scheduler.py:2417` 到 `scheduler.py:2445`

如果请求仍是：

```text
WAITING_FOR_REMOTE_KVS
```

则加入：

```text
finished_recving_kv_req_ids
```

位置：`scheduler.py:2431` 到 `scheduler.py:2438`

这只是“transfer 完成”的标记，不代表没有失败。

是否失败由：

```text
failed_recving_kv_req_ids
```

决定。

最终在 `_update_waiting_for_remote_kv()` 中同时考虑：

```text
finished_recving_kv_req_ids：允许 promote；
failed_recving_kv_req_ids：走失败提交逻辑。
```

---

## 28. 与 request_finished / abort 的关系

如果外部请求被 abort 或 finish，而它还处于：

```text
WAITING_FOR_REMOTE_KVS
```

`finish_requests()` 会判断：

```python
if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    delay_free_blocks = request.request_id not in self.finished_recving_kv_req_ids
    self.finished_recving_kv_req_ids.discard(request.request_id)
    self.failed_recving_kv_req_ids.discard(request.request_id)
```

位置：`scheduler.py:2031` 到 `scheduler.py:2040`

含义是：

```text
如果 transfer 还没 finished_recving，不能立即释放 blocks；
同时要清理 finished / failed recving 集合，避免悬挂状态污染后续逻辑。
```

这和 invalid blocks 的恢复链路是一致的：

```text
异步 transfer 的 block 生命周期必须等 Worker connector 收尾。
```

---

## 29. 一个 async recompute 例子

假设：

```text
block_size = 16
外部 KV 命中 99 个 blocks
失败 block index = 50
policy = recompute
```

### 29.1 schedule 阶段

```text
connector.get_num_new_matched_tokens()
  → (99 * 16, True)

Scheduler:
  → allocate_slots(... num_external_computed_tokens=1584, delay_cache_blocks=True)
  → request.status = WAITING_FOR_REMOTE_KVS
  → request.num_computed_tokens = 1584
```

### 29.2 Worker 回报失败

```text
KVConnectorOutput.invalid_block_ids = {req_block_ids[50]}
finished_recving 暂时未返回或稍后返回
```

### 29.3 Scheduler 处理 invalid blocks

```text
_update_requests_with_invalid_blocks()
  → idx = 50
  → request.num_computed_tokens = 50 * 16 = 800
  → affected_req_ids = {request_id}

_handle_invalid_blocks()
  → failed_recving_kv_req_ids.add(request_id)
```

### 29.4 finished_recving 后 promote

```text
_update_waiting_for_remote_kv()
  → request_id 在 failed_recving_kv_req_ids
  → cache_blocks(request, 800)
  → status = WAITING
```

后续调度会从 token 800 开始本地重算。

---

## 30. 一个 sync fail 例子

假设：

```text
policy = fail
sync load 请求已经进入 running
失败 block index = 50
```

### 30.1 Worker 回报失败

```text
ModelRunnerOutput.kv_connector_output.invalid_block_ids = {req_block_ids[50]}
```

### 30.2 Scheduler 处理

```text
_handle_invalid_blocks()
  → 扫描 self.running
  → request.num_computed_tokens = 50 * block_size
  → sync_blocks_to_evict = req_block_ids[50:]
  → evict_blocks(sync_blocks_to_evict)
  → return {request_id}
```

### 30.3 update_from_output 失败请求

```text
finish_requests({request_id}, FINISHED_ERROR)
EngineCoreOutput.finish_reason = ERROR
```

这个行为由 `test_error_propagation_sync_load()` 覆盖。

---

## 31. 容易疑惑的点

### 31.1 invalid_block_ids 是远端 block id 吗？

不是。

它是 Worker 侧本地 KV block id。Scheduler 用它在本地 `req_to_blocks` / block table 中定位请求和 token 范围。

### 31.2 invalid_block_ids 会直接让请求失败吗？

不一定。

取决于 `kv_load_failure_policy`：

```text
recompute：回退并重算；
fail：FINISHED_ERROR。
```

### 31.3 async load 失败后为什么还要等 finished_recving？

因为 connector 需要完成 transfer 收尾；接口要求即使失败也必须通过 `get_finished()` 报告完成。

### 31.4 回退到失败 block 前是否会丢掉有效 KV？

不会丢掉失败 block 之前的完整有效 blocks。回退点是 `idx * block_size`，只保留失败 block 之前的完整 prefix。

### 31.5 为什么后续 blocks 也要 evict？

因为后续 prefix cache blocks 依赖前面的 block hash 链。前面的 block 失败，后续 block 的语义也不能直接信任。

### 31.6 sync load 下为什么要跳过本轮 output 更新？

因为这个请求本轮使用了不可信 KV，sampled tokens / stop 判断都不能作为有效结果。跳过后下一轮按回退后的进度重新执行。

### 31.7 async load 失败且回退到 0 时为什么 free request blocks？

因为没有任何有效外部 KV prefix 可以保留，已分配 blocks 没有可提交内容，释放后让请求重新从头走普通调度。

### 31.8 recompute policy 下为什么不总是 evict invalid blocks？

sync shared-block 场景下，失败 block 可能由某个请求重算后继续被共享。源码只在非 recompute policy 下 evict sync failed blocks。

---

## 32. 最小心智模型

invalid blocks 的恢复链路可以记成：

```text
Worker 发现外部 KV block load 失败
  → 上报本地 invalid block ids
  → Scheduler 找到引用这些 block 的请求
  → num_computed_tokens 回退到第一个失败 block 前
  → recompute：重新调度失败部分
  → fail：请求 FINISHED_ERROR
```

再压缩成一句话：

```text
invalid_block_ids 不是最终错误处理结果，而是 Scheduler 用来修正“哪些 KV 可以信任”的证据；真正的结果由 kv_load_failure_policy 决定。
```