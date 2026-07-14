# 06. KV Cache block 是否够用，不够时是否需要抢占？

源码位置：`vllm/vllm/v1/core/sched/scheduler.py`

本问题关注：Scheduler 在调度 running / waiting 请求时，如何向 `KVCacheManager` 申请 KV Cache block；申请失败时，哪些场景会触发抢占；被抢占请求如何回到 waiting；请求结束后 block 又如何释放或延迟释放。

一句话概括：

```text
Scheduler 自己不直接管理每个 KV block，
而是通过 kv_cache_manager.allocate_slots() 判断 block 是否够用。

running 请求分配失败时，可以抢占 running 队列中的请求来释放 block；
waiting 请求分配失败时，本轮 waiting 调度停止，不在 waiting 阶段主动抢占 running。
```

---

## 1. 一句话回答

KV block 是否够用，核心由：

```python
self.kv_cache_manager.allocate_slots(...)
```

判断。

running 阶段调用：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:572`

waiting 阶段调用：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    num_encoder_tokens=num_encoder_tokens,
    full_sequence_must_fit=self.scheduler_reserve_full_isl,
    reserved_blocks=reserved_blocks,
    has_scheduled_reqs=bool(self.running),
)
```

位置：`scheduler.py:942`

返回值含义：

```text
new_blocks is not None：
  block 分配成功，请求本轮可以继续调度。

new_blocks is None：
  block 不够，请求本轮不能按当前计划调度。
```

两类请求处理方式不同：

```text
running 请求 allocate_slots 失败：
  Scheduler 会尝试抢占某个 running 请求，释放 block 后重试。

waiting 请求 allocate_slots 失败：
  Scheduler 不在这里抢占 running，而是停止 waiting 阶段。
```

---

## 2. KV block 分配发生在哪两个阶段

`schedule()` 中有两个主要分配点。

### 2.1 running 阶段

running 请求已经在 `self.running` 中，本轮要继续 decode / chunked prefill / spec decode 时，需要为新增 token 申请 block：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

位置：`scheduler.py:572`

running 阶段的分配比较简单，因为请求已经有一部分 blocks 绑定在自己身上，本轮只需要追加新增 token / lookahead token 所需的 slots。

### 2.2 waiting 阶段

waiting 请求第一次进入 running，或者 PREEMPTED 请求重新恢复时，分配参数更多：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    num_encoder_tokens=num_encoder_tokens,
    full_sequence_must_fit=self.scheduler_reserve_full_isl,
    reserved_blocks=reserved_blocks,
    has_scheduled_reqs=bool(self.running),
)
```

位置：`scheduler.py:942`

waiting 阶段的分配要同时考虑：

```text
本地 prefix cache 命中的 blocks；
外部 KV cache 命中的 tokens；
本轮真正要计算的 tokens；
spec decode lookahead tokens；
encoder-decoder cross-attention blocks；
async KV load 的 delayed cache blocks；
为其它 in-flight 请求保留的 reserved blocks；
waiting admission watermark。
```

---

## 3. `allocate_slots()` 的输入参数含义

`KVCacheManager.allocate_slots()` 定义是：

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    reserved_blocks: int = 0,
    has_scheduled_reqs: bool = True,
) -> KVCacheBlocks | None:
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:269`

关键参数如下：

| 参数 | 含义 |
|---|---|
| `request` | 要分配 KV slots 的请求 |
| `num_new_tokens` | 本轮要新增计算的 token 数 |
| `num_new_computed_tokens` | 本地 prefix cache 新命中的 token 数，不含外部 KV |
| `new_computed_blocks` | 本地 prefix cache 命中的 blocks |
| `num_lookahead_tokens` | spec decode / EAGLE 等需要额外预留的 lookahead tokens |
| `num_external_computed_tokens` | 外部 KV Connector 命中的 token 数 |
| `delay_cache_blocks` | async KV load 时，分配 block 但暂不正式 cache |
| `num_encoder_tokens` | encoder-decoder cross-attention 需要的 block 数量依据 |
| `full_sequence_must_fit` | admission gate：要求完整序列能放下才接纳 |
| `reserved_blocks` | 需要留给其它 in-flight 请求的空闲 blocks 数 |
| `has_scheduled_reqs` | 本轮是否已经有 scheduled requests，影响 watermark 是否生效 |

---

## 4. block 布局：computed / external / new / lookahead

`allocate_slots()` 的注释给出了 block 布局：

```text
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                              |   < to be computed >     |
----------------------------------------------------------------------
                              |            < to be allocated >           |
----------------------------------------------------------------------
```

位置：`kv_cache_manager.py:315`

几个缩写含义：

```text
comp：
  request.num_computed_tokens，之前已经计算过的 token。

new_comp：
  num_new_computed_tokens，本轮新发现的本地 prefix cache 命中。

ext_comp：
  num_external_computed_tokens，外部 KV cache 命中。

new：
  num_new_tokens，本轮要实际计算的 token。

lookahead：
  num_lookahead_tokens，spec decode 预留 token。
```

所以一次 block 分配可能同时处理四类东西：

```text
1. 已经存在且可复用的本地 prefix cache blocks；
2. 外部 KV 命中但需要本地承接的 blocks；
3. 本轮 forward 要写入的新 blocks；
4. 投机解码需要提前预留的 lookahead blocks。
```

---

## 5. 为什么 `num_new_tokens = 0` 有时也能分配

`allocate_slots()` 开头有一个校验：

```python
if num_new_tokens == 0 and num_external_computed_tokens == 0:
    raise ValueError(
        "num_new_tokens must be greater than 0 when there are no "
        "external computed tokens"
    )
```

位置：`kv_cache_manager.py:365`

这说明：

```text
普通本地计算路径下，num_new_tokens 不能是 0；
但 async external KV load 场景可以 num_new_tokens = 0，
前提是 num_external_computed_tokens > 0。
```

原因是 async KV load 本轮不做模型 forward，但仍然要为外部命中的 KV 分配本地 blocks 来接收远端数据。

对应 waiting 阶段：

```python
if load_kv_async:
    assert num_external_computed_tokens > 0
    num_new_tokens = 0
```

位置：`scheduler.py:834`

---

## 6. `allocate_slots()` 内部的三阶段逻辑

`allocate_slots()` 注释明确说 allocation 有三阶段：

```python
# The allocation has three stages:
# - Free unnecessary blocks in `comp` and check
#    if we have sufficient free blocks (return None if not).
# - Handle prefix tokens (`comp + new_comp + ext_comp`):
#     - Free unnecessary blocks (e.g. outside sliding window)
#     - Allocate new blocks for `ext_comp` tokens inside
#       sliding window
# - Allocate new blocks for tokens to be computed (`new + lookahead`)
```

位置：`kv_cache_manager.py:353`

可以理解为：

```text
第一步：先释放 attention 已经不需要的旧 blocks，例如 sliding window 外的 blocks；
第二步：处理已经 computed 的 token，包括本地 prefix hit 和 external hit；
第三步：为本轮真正要计算的 token 和 lookahead token 分配新 blocks。
```

这说明 block 分配不只是简单 `free_blocks >= required_blocks`，它还会先尝试释放当前请求不再需要的 blocks，然后再计算是否足够。

---

## 7. computed tokens 与 total computed tokens

`allocate_slots()` 会先计算本地 computed token 数：

```python
num_local_computed_tokens = (
    request.num_computed_tokens + num_new_computed_tokens
)
```

位置：`kv_cache_manager.py:380`

然后合并外部 KV 命中：

```python
total_computed_tokens = min(
    num_local_computed_tokens + num_external_computed_tokens,
    self.max_model_len,
)
```

位置：`kv_cache_manager.py:383`

含义：

```text
num_local_computed_tokens：
  请求已有计算进度 + 本地 prefix cache 新命中。

total_computed_tokens：
  本地 computed + 外部 KV computed，且不超过 max_model_len。
```

这些值会影响：

```text
哪些 old blocks 可以被 remove_skipped_blocks 释放；
external computed tokens 需要分配多少承接 blocks；
本轮最终需要多少 total slots。
```

---

## 8. waiting admission watermark

`allocate_slots()` 会计算一个 watermark：

```python
watermark_blocks = 0
if has_scheduled_reqs and request.status in (
    RequestStatus.WAITING,
    RequestStatus.PREEMPTED,
):
    watermark_blocks = self.watermark_blocks
```

位置：`kv_cache_manager.py:388`

含义是：

```text
watermark 只应用于 waiting / preempted 请求，
并且只有当本轮已经有 scheduled requests 时才生效。
```

为什么 running 请求不应用这个 watermark？

```text
running 请求已经占有 KV blocks，继续推进它们通常可以尽快完成并释放资源；
waiting / preempted 请求是新接纳或重新接纳的请求，需要留一点 headroom，避免刚接纳就频繁触发抢占。
```

所以 waiting 请求即使理论上刚好能放下，也可能因为 watermark 要求保留一部分空闲 blocks 而分配失败。

---

## 9. `full_sequence_must_fit`：完整序列 admission gate

如果启用：

```python
full_sequence_must_fit=True
```

`allocate_slots()` 会先检查完整请求序列能否放下：

```python
if full_sequence_must_fit:
    full_num_tokens = min(request.num_tokens, self.max_model_len)

    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=full_num_tokens,
        ...
        apply_admission_cap=True,
    )
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > self.block_pool.get_num_free_blocks():
        return None
```

位置：`kv_cache_manager.py:397`

注释说明它用于：

```text
Only allocate blocks if the KV cache has enough free blocks to hold the full sequence,
accounting for prefix cache hits and sliding window.
```

位置：`kv_cache_manager.py:303`

作用是 admission gate：

```text
即使 chunked prefill 只需要先放下第一段，
也可以要求完整序列最终必须能放下，
避免过度接纳长请求导致后续必然抢占。
```

在 Scheduler waiting 阶段，它来自：

```python
full_sequence_must_fit=self.scheduler_reserve_full_isl
```

位置：`scheduler.py:951`

---

## 10. `remove_skipped_blocks()`：先释放不再需要的 blocks

在真正计算需要多少新 blocks 前，`allocate_slots()` 会调用：

```python
self.coordinator.remove_skipped_blocks(
    request.request_id,
    max(0, total_computed_tokens - request.num_in_flight_tokens),
    num_prompt_tokens=request.num_prompt_tokens,
)
```

位置：`kv_cache_manager.py:429`

注释说明：

```python
# Free the blocks that are skipped during the attention computation
# (e.g., tokens outside the sliding window).
# We can do this even if we cannot schedule this request due to
# insufficient free blocks.
# Should call this function before allocating new blocks to reduce
# the number of evicted blocks.
# Free on the processed-token basis: in-flight steps' attention windows
# still read blocks below the optimistic boundary, and rejected spec
# tokens can roll it back.
```

位置：`kv_cache_manager.py:420`

这一步的意义是：

```text
如果 sliding window 或其它机制已经不再需要某些旧 token 的 KV，
就先把这些 blocks 释放出来，
提高当前请求或其它请求的分配成功率。
```

注意：即使当前请求后续因为 block 不够无法调度，这些 skipped blocks 也可以先释放。但新 commit 不会按乐观的 `total_computed_tokens` 直接释放，而是扣掉 `request.num_in_flight_tokens`，避免仍在 in-flight step 中被 attention 读取的 blocks 被提前归还；spec token 后续被拒绝时也可能回滚 computed 进度。

---

## 11. 计算需要多少 blocks

接下来会计算本轮需要分配多少 blocks：

```python
num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
    request_id=request.request_id,
    num_tokens=num_tokens_need_slot,
    new_computed_blocks=new_computed_block_list,
    num_encoder_tokens=num_encoder_tokens,
    total_computed_tokens=num_local_computed_tokens
    + num_external_computed_tokens,
    num_tokens_main_model=num_tokens_main_model,
)
```

位置：`kv_cache_manager.py:467`

其中：

```python
num_tokens_main_model = total_computed_tokens + num_new_tokens
num_tokens_need_slot = min(
    num_tokens_main_model + num_lookahead_tokens,
    self.max_model_len,
)
```

位置：`kv_cache_manager.py:415`

含义：

```text
num_tokens_main_model：
  主模型本轮调度后需要覆盖到的位置。

num_tokens_need_slot：
  主模型 token + lookahead token 需要的 slot 总范围。
```

所以 spec decode 的 lookahead token 会增加 block 需求，即使这些 token 后续可能被拒绝。

---

## 12. `reserved_blocks` 与 `watermark_blocks`

真正判断是否够用时：

```python
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None
```

位置：`kv_cache_manager.py:446`

这里有两个额外约束。

### 12.1 reserved_blocks

`reserved_blocks` 是必须给其它 in-flight 请求留下的空闲 blocks。

Scheduler waiting 阶段在 async KV load 时设置：

```python
if load_kv_async:
    reserved_blocks = self._inflight_prefill_reserved_blocks()
```

位置：`scheduler.py:934`

原因是 async load 会持有 blocks，且在这里不可抢占。如果它把 block 占光，可能导致已经 in-flight 的 prefill 无法完成。

### 12.2 watermark_blocks

`watermark_blocks` 是 waiting / preempted admission 的额外 headroom。

所以最终判定不是：

```text
num_blocks_to_allocate <= free_blocks
```

而是：

```text
num_blocks_to_allocate + watermark_blocks
  <= free_blocks - reserved_blocks
```

这就是为什么看起来还有 free blocks，也可能分配失败。

---

## 13. prefix hit / external hit blocks 的分配

如果存在本地 prefix cache 命中 blocks，或者外部 KV 命中 tokens：

```python
if (
    new_computed_block_list is not self.empty_kv_cache_blocks.blocks
    or num_external_computed_tokens > 0
):
    self.coordinator.allocate_new_computed_blocks(
        request_id=request.request_id,
        new_computed_blocks=new_computed_block_list,
        num_local_computed_tokens=num_local_computed_tokens,
        num_external_computed_tokens=num_external_computed_tokens,
    )
```

位置：`kv_cache_manager.py:454`

这一步会：

```text
1. 把本地 prefix hit blocks 接到当前 request 上；
2. 为 external computed tokens 分配本地承接 blocks；
3. 维护 block 引用计数 / touch 状态，避免 cache-hit blocks 被误驱逐。
```

`05_prefix_and_external_kv_hits.md` 已经展开过这部分。本篇只需要记住：

```text
命中 token 也会影响 block 分配，尤其是 external KV hit 仍然需要本地 blocks 承接。
```

---

## 14. 为 new tokens / lookahead tokens 分配新 blocks

处理完 computed blocks 后，会为本轮要计算的 token 分配新 blocks：

```python
new_blocks = self.coordinator.allocate_new_blocks(
    request.request_id,
    num_tokens_need_slot,
    num_tokens_main_model,
    num_encoder_tokens,
)
```

位置：`kv_cache_manager.py:467`

返回的 `new_blocks` 会被包装成：

```python
return self.create_kv_cache_blocks(new_blocks)
```

位置：`kv_cache_manager.py:477` 或 `kv_cache_manager.py:490`

Scheduler 会把它记录到：

```python
req_to_new_blocks[request_id] = new_blocks
```

running 阶段位置：`scheduler.py:625`

waiting 阶段则记录完整 blocks：

```python
req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(request_id)
```

位置：`scheduler.py:1022`

---

## 15. 分配后是否立即 cache blocks

如果没有启用 caching，或者是 async KV load：

```python
if not self.enable_caching or delay_cache_blocks:
    return self.create_kv_cache_blocks(new_blocks)
```

位置：`kv_cache_manager.py:476`

`delay_cache_blocks=True` 主要用于 P/D 或 KV transfer：

```text
block 已经分配出来接收远端 KV，
但 KV 数据还没真正到达，
所以不能立即把这些 blocks 标记为可复用 prefix cache。
```

普通路径下，会 cache 到：

```python
num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,
)
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

位置：`kv_cache_manager.py:484`

这里会排除不可提交的 token，例如可能被拒绝的 draft token。

---

## 16. running 阶段 block 不够：触发抢占

running 请求分配失败时，Scheduler 会进入抢占循环：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(...)

    if new_blocks is not None:
        break

    # The request cannot be scheduled.
    # Preempt the lowest-priority request.
    ...
```

位置：`scheduler.py:570`

也就是说：

```text
running 阶段 allocate_slots 返回 None，并不立刻放弃；
Scheduler 会尝试抢占其它 running 请求释放 blocks，然后重试 allocate_slots。
```

这是 running 阶段和 waiting 阶段最大的不同。

---

## 17. PRIORITY 策略下抢占谁

如果策略是 `PRIORITY`：

```python
preempted_req = max(
    self.running,
    key=lambda r: (r.priority, r.arrival_time),
)
self.running.remove(preempted_req)
```

位置：`scheduler.py:584`

含义是：

```text
从当前 running 请求中选择最应该被抢占的请求。
```

这里使用 `(priority, arrival_time)` 作为 key。

可以理解为：

```text
数值更大的 `priority`（调度优先级更低）/ 更晚 `arrival_time` 的请求更可能被抢占。
```

具体优先级大小与队列比较实现有关，但核心语义是：

```text
PRIORITY 策略下，不是简单抢占队尾，而是根据优先级和到达时间选择 victim。
```

---

## 18. 非 PRIORITY 策略下抢占谁

如果不是 PRIORITY：

```python
preempted_req = self.running.pop()
```

位置：`scheduler.py:609`

也就是抢占 running 队尾请求。

这更接近 FCFS / 队列顺序语义：

```text
前面的 running 请求优先保留，后面的 running 请求先被抢占。
```

---

## 19. 抢占了本轮已经调度过的请求怎么办

PRIORITY 模式下，有可能抢占到本轮前面已经成功调度过的 running 请求。

这时 Scheduler 必须撤销它本轮的调度记录：

```python
if preempted_req in scheduled_running_reqs:
    preempted_req_id = preempted_req.request_id
    scheduled_running_reqs.remove(preempted_req)
    token_budget += num_scheduled_tokens.pop(preempted_req_id)
    req_to_new_blocks.pop(preempted_req_id)
    scheduled_spec_decode_tokens.pop(preempted_req_id, None)
```

位置：`scheduler.py:590`

如果它本轮还安排了 encoder input，还要恢复 encoder budget：

```python
preempted_encoder_inputs = scheduled_encoder_inputs.pop(
    preempted_req_id, None
)
if preempted_encoder_inputs:
    num_embeds_to_restore = sum(
        preempted_req.get_num_encoder_embeds(i)
        for i in preempted_encoder_inputs
    )
    encoder_compute_budget += num_embeds_to_restore
```

位置：`scheduler.py:596`

这说明抢占不是单纯释放 block，还必须保持本轮 `SchedulerOutput` 的一致性：

```text
已经被抢占的请求，不能继续出现在本轮 scheduled_running_reqs / num_scheduled_tokens / scheduled_encoder_inputs 中。
```

---

## 20. `_preempt_request()` 做了什么

真正的抢占处理在：

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
```

位置：`scheduler.py:1191`

核心逻辑：

```python
self._free_request_blocks(request)
self.encoder_cache_manager.free(request)
self._inflight_prefills.discard(request)
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
if request.spec_token_ids:
    request.spec_token_ids = []
request.num_preemptions += 1
self.waiting.prepend_request(request)
self.reset_preempted_req_ids.add(request.request_id)
```

位置：`scheduler.py:1200`

所以被抢占请求会发生这些变化：

```text
1. 从 self.running 中移除；
2. 释放或延迟释放 KV blocks；
3. 释放 encoder cache；
4. 从 _inflight_prefills 移除；
5. status 改为 PREEMPTED；
6. num_computed_tokens 重置为 0；
7. 清空 spec_token_ids；
8. num_preemptions 加 1；
9. 放回 self.waiting 队列头部；
10. 记录到 reset_preempted_req_ids，后续通过 SchedulerOutput 通知 Worker 侧重置相关状态。
```

状态迁移是：

```text
RUNNING
  → block 不够，被抢占
  → PREEMPTED
  → waiting 队列头部
```

---

## 21. 为什么抢占要把 `num_computed_tokens` 重置为 0

抢占时：

```python
request.num_computed_tokens = 0
```

位置：`scheduler.py:1204`

原因是：

```text
请求当前绑定的 KV blocks 已经释放，
Scheduler 不能再认为这些 token 仍然被该请求持有。
```

但这不意味着之前算过的 KV 一定全部浪费。

如果这些 blocks 已经进入 prefix cache，或者被外部 KV Connector 保存，后续 PREEMPTED 请求重新进入 waiting 调度时，会重新走：

```text
本地 prefix cache 查询；
外部 KV cache 查询；
block 重新分配。
```

因此抢占后的恢复路径是：

```text
PREEMPTED / waiting
  → prefix cache / external KV hit
  → allocate_slots
  → RUNNING
```

如果外部 KV 命中需要异步加载，恢复会多一个中间状态：

```text
PREEMPTED / waiting
  → get_num_new_matched_tokens 命中远端 KV
  → load_kv_async=True
  → allocate_slots(delay_cache_blocks=True)
  → WAITING_FOR_REMOTE_KVS
  → Worker finished_recving
  → PREEMPTED / waiting
  → 再次调度进入 RUNNING
```

也就是说，被抢占请求本地 blocks 被释放后，如果已计算 KV 被外部 KV 池保存，重新进入 running 前可以先从远端加载回来；未命中的部分才需要重新 prefill。

---

## 22. 抢占当前 request 时，说明无法调度它

抢占循环中有一段：

```python
if preempted_req == request:
    # No more request to preempt. Cannot schedule this request.
    break
```

位置：`scheduler.py:565`

含义是：

```text
如果为了给当前 request 分配 block，最后把当前 request 自己也抢占了，
说明已经没有其它 running 请求可释放资源，
当前请求本轮无法调度。
```

之后：

```python
if new_blocks is None:
    # Cannot schedule this request.
    break
```

位置：`scheduler.py:569`

running 阶段停止。

---

## 23. running 阶段发生抢占后，waiting 阶段不会执行

waiting 阶段入口是：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

位置：`scheduler.py:625`

只要 running 阶段发生过抢占，`preempted_reqs` 非空，本轮就不会再调度 waiting 请求。

这很重要：

```text
抢占说明 KV block 已经紧张；
此时再接纳新的 waiting 请求，只会加剧 block 压力。
```

因此，本轮如果抢占了 running 请求，Scheduler 会停止拉新，只返回当前已经确定的调度输出。

---

## 24. waiting 阶段 block 不够：不抢占，直接停止 waiting 调度

waiting 阶段如果 `allocate_slots()` 返回 None：

```python
if new_blocks is None:
    # The request cannot be scheduled.

    # NOTE: we need to untouch the request from the encode cache
    # manager
    if request.has_encoder_inputs:
        self.encoder_cache_manager.free(request)
    break
```

位置：`scheduler.py:887`

这里没有调用 `_preempt_request()`。

原因是：

```text
waiting 请求还没有进入 running，
它不是已经占有执行流的请求；
如果为了接纳一个 waiting 请求而抢占 running，
可能会明显伤害已有请求的 latency 和稳定性。
```

所以 waiting 阶段策略是：

```text
block 够 → 接纳 waiting 请求进入 running；
block 不够 → 停止 waiting 阶段，等待后续 running 请求完成或释放资源。
```

---

## 25. waiting async KV load 的 block 约束更严格

async KV load 会设置：

```python
reserved_blocks = self._inflight_prefill_reserved_blocks()
```

位置：`scheduler.py:934`

并传给：

```python
allocate_slots(..., reserved_blocks=reserved_blocks)
```

位置：`scheduler.py:883`

注释说：

```python
# An async load holds its blocks for the whole transfer with
# no forward progress and isn't preemptible here. Admit it
# only if it fits in (free - other in-flight reservations), to
# avoid deadlock and predictable preemptions.
```

位置：`scheduler.py:867`

这说明 async KV load 请求虽然不消耗本地 forward token budget，但它对 block 资源很敏感：

```text
它会占用 blocks 等远端 KV 到达；
在这里不能靠抢占它来快速释放；
所以 admission 时必须留足其它 in-flight prefill 的资源。
```

---

## 26. 成功分配后，running 与 waiting 记录不同

### 26.1 running 请求

running 请求成功分配后记录：

```python
scheduled_running_reqs.append(request)
request_id = request.request_id
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

位置：`scheduler.py:573`

这里 `req_to_new_blocks` 记录的是本轮新增分配的 blocks。

### 26.2 waiting 请求

waiting 请求成功分配并进入 running 后记录：

```python
req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
    request_id
)
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens
```

位置：`scheduler.py:1022`

这里记录的是请求当前完整 block 列表，因为它是新进入模型执行流的请求，需要给 Worker 足够的初始化信息。

### 26.3 为什么 resumed 请求不能当普通 running 请求追加 blocks

被抢占请求在 `_preempt_request()` 中会释放 KV blocks，并把 `num_computed_tokens` 重置为 0。之后这些旧 blocks 可能已经被其它请求复用。

所以 resumed 请求重新调度时，Worker 侧虽然可能还保留 request state，但旧 block table 已经失效：

```text
普通 running 请求：
  旧 blocks 仍属于该请求，新 blocks 可以 append。

resumed 请求：
  旧 blocks 已释放或复用，新 blocks 必须 replace 旧 block table。
```

旧 model runner 通过 `scheduled_cached_reqs.resumed_req_ids` 表达这个差异；Worker 看到请求在 `resumed_req_ids` 中，就用新的 `block_ids` 替换旧值，而不是追加。这样可以复用仍有效的 request state，只重建 KV cache 相关状态。

V2 model runner 路径下，Scheduler 会把 `scheduled_resumed_reqs` 合并进 `scheduled_new_reqs`，相当于重新下发完整 request 数据，由 V2 runner 按新请求路径处理。

---

## 27. 请求结束后如何释放 blocks

请求真正结束时会调用：

```python
def _free_request(self, request: Request, delay_free_blocks: bool = False)
```

位置：`scheduler.py:2046`

核心逻辑：

```python
self._inflight_prefills.discard(request)
connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
self.encoder_cache_manager.free(request)
self.finished_req_ids.add(request_id)
...
if not delay_free_blocks:
    self._free_blocks(request)
```

位置：`scheduler.py:2051`

这里有一个关键点：

```text
请求 finished 不一定立刻释放 KV blocks。
```

如果 connector 需要异步保存 / 发送 KV，`connector_delay_free_blocks` 可能为 True，这时会延迟释放 blocks。

---

## 28. `_free_blocks()` 才会从 `self.requests` 删除请求

真正释放请求 blocks 并删除请求索引的是：

```python
def _free_blocks(self, request: Request):
    assert request.is_finished()
    self._free_request_blocks(request)
    del self.requests[request.request_id]
```

位置：`scheduler.py:2065`

所以：

```text
_free_request() 标记请求完成并尝试释放资源；
_free_blocks() 才真正释放 KV blocks 并从 self.requests 删除。
```

这解释了为什么：

```text
finished 请求可能已经不在 running / waiting 中，
但仍然留在 self.requests，
通常是因为 KV Connector 还没完成异步发送，导致 `_free_request()` 暂时不能调用 `_free_blocks()`。deferred free 只延迟 block 归还 block pool；一旦执行 `_free_blocks()`，请求索引已经会从 `self.requests` 删除。
```

---

## 29. `_free_request_blocks()` 与 deferred free

释放 block 的底层入口是：

```python
def _free_request_blocks(self, request: Request):
```

位置：`scheduler.py:2077`

它会判断是否需要延迟释放：

```python
if not self.defer_block_free or (
    request.last_sched_seq <= self.processed_step_seq
):
    self.kv_cache_manager.free(request)
    return
```

位置：`scheduler.py:2081`

如果当前还有 in-flight GPU step 可能写这些 blocks，则不能立刻还给 block pool：

```python
blocks = self.kv_cache_manager.pop_blocks_for_free(request)
if blocks:
    self.deferred_frees.append((self.sched_step_seq, blocks))
```

位置：`scheduler.py:2088`

原因是 async scheduling / pipeline parallel 下，Scheduler 可能提前释放某个请求，但 GPU 上一轮写 KV 的操作还没真正完成。如果此时 block 被立即复用，可能产生写入竞态。

---

## 30. deferred frees 如何最终归还 block pool

后续 `update_from_output()` 开头会 drain deferred frees。

释放逻辑是：

```python
def _drain_deferred_frees(self):
    while self.deferred_frees:
        fence, _ = self.deferred_frees[0]
        if fence > self.processed_step_seq:
            break
        _, blocks = self.deferred_frees.popleft()
        self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
```

位置：`scheduler.py:2092`

含义是：

```text
只有当对应 fence step 已经被处理完，
这些 blocks 才能真正归还给 block pool。
```

这保证了：

```text
不会把仍可能被 GPU 写入的 KV blocks 提前复用。
```

---

## 31. 一个完整例子：running decode 分配成功

假设：

```text
running req-a:
  本轮 decode 1 token
  num_new_tokens = 1
  num_lookahead_tokens = 0
  KV block pool 还有空闲 block
```

流程：

```text
1. Scheduler 调用 allocate_slots(req-a, 1)
2. KVCacheManager 计算需要新增 blocks
3. free blocks 足够
4. allocate_new_blocks 成功
5. 返回 new_blocks
6. Scheduler 记录：
   scheduled_running_reqs.append(req-a)
   num_scheduled_tokens[req-a] = 1
   req_to_new_blocks[req-a] = new_blocks
```

结果：

```text
req-a 本轮继续 decode。
```

---

## 32. 一个完整例子：running block 不够导致抢占

假设：

```text
running = [req-a, req-b, req-c]
当前调度 req-a，需要新增 block
allocate_slots(req-a) 返回 None
policy = FCFS
```

Scheduler 会：

```text
1. preempted_req = self.running.pop()，即 req-c；
2. _preempt_request(req-c)：
   - 释放 req-c blocks；
   - req-c.status = PREEMPTED；
   - req-c.num_computed_tokens = 0；
   - req-c 放回 waiting 队列头部；
3. 重新尝试 allocate_slots(req-a)。
```

如果这次成功：

```text
req-a 本轮继续调度；
req-c 等后续轮次重新从 waiting 恢复。
```

如果仍然失败：

```text
继续抢占，直到成功或当前 req-a 自己也被抢占。
```

---

## 33. 一个完整例子：waiting 分配失败

假设：

```text
running 已经占用了大部分 KV blocks
waiting 队头 req-d 需要 4096 token 的 prefill blocks
allocate_slots(req-d) 返回 None
```

waiting 阶段处理：

```text
1. 不抢占 running；
2. 如果 req-d 有 encoder input，释放刚 touch/allocate 的 encoder cache；
3. break，结束 waiting 阶段。
```

结果：

```text
req-d 本轮不进入 running；
后面的 waiting 请求也不会被绕过调度；
等待后续 running 请求完成或释放 blocks。
```

---

## 34. 一个完整例子：外部 KV async load 分配 blocks

假设：

```text
waiting req-e:
  external KV 命中 8000 token
  load_kv_async = True
  num_new_tokens = 0
  num_external_computed_tokens = 8000
```

分配时：

```text
allocate_slots(
  req-e,
  num_new_tokens=0,
  num_external_computed_tokens=8000,
  delay_cache_blocks=True,
  reserved_blocks=...
)
```

如果分配成功：

```text
1. 为远端 KV load 分配本地接收 blocks；
2. connector.update_state_after_alloc() 记录这些 blocks；
3. req-e.status = WAITING_FOR_REMOTE_KVS；
4. req-e 进入 skipped_waiting；
5. 本轮不进入 running。
```

如果分配失败：

```text
waiting 阶段停止，不启动这次 async load。
```

---

## 35. 容易疑惑的点

### 35.1 token_budget 够，KV block 就一定够吗？

不一定。

`token_budget` 限制的是本轮模型计算 token 数；KV block 限制的是缓存容量。

可能出现：

```text
token_budget 还有很多，
但 KV block 不够，
请求仍然不能调度。
```

### 35.2 KV block 够，token_budget 就一定够吗？

也不一定。

KV block 够只能说明有地方存 KV，是否能调度还要看：

```text
token_budget；
max_model_len；
encoder budget；
Mamba alignment；
chunked prefill；
LoRA / pause 等其它约束。
```

### 35.3 waiting 请求 block 不够时为什么不抢占？

因为 waiting 请求还没有进入执行流，抢占 running 去接纳 waiting 会伤害已有请求 latency。

Scheduler 的策略是：

```text
优先保证 running 请求推进；
waiting 请求只有在资源允许时才接纳。
```

### 35.4 抢占会丢掉已经算过的 KV 吗？

当前请求绑定的 blocks 会被释放，但如果这些 blocks 已经进入 prefix cache 或外部 KV cache，后续可以重新命中。

因此抢占会把请求状态重置为：

```text
PREEMPTED / num_computed_tokens = 0
```

后续重新调度时再通过 prefix cache / external KV cache 找回可复用部分。

### 35.5 `new_blocks is None` 表示什么？

表示当前 block 分配条件不满足。

原因可能包括：

```text
free blocks 不够；
watermark 要求保留 headroom；
reserved_blocks 要求为 in-flight 请求保留空间；
full_sequence_must_fit admission gate 失败；
external KV hit 需要的承接 blocks 不够；
lookahead tokens 额外 block 需求太大。
```

### 35.6 finished 请求一定马上释放 block 吗？

不一定。

如果 KV Connector 需要异步发送 / 保存 KV，或者 async scheduling 下 GPU step 还没完成，block 释放可能延迟。

---

## 36. 从“回答问题”的角度总结

如果要问：

```text
KV Cache block 是否够用，不够时是否需要抢占？
```

Scheduler 的回答是：

```text
用 kv_cache_manager.allocate_slots() 尝试分配。

running 请求分配失败：
  通过抢占 running 请求释放 block，然后重试。

waiting 请求分配失败：
  不抢占，停止 waiting 阶段。

请求结束或被抢占：
  释放或延迟释放 KV blocks。
```

running 抢占迁移：

```text
RUNNING
  → allocate_slots 失败
  → 选择 victim
  → _preempt_request()
  → PREEMPTED
  → waiting 队列头部
```

waiting 分配失败：

```text
WAITING / PREEMPTED
  → allocate_slots 失败
  → 本轮不进入 running
  → waiting 阶段停止
```

---

## 37. 最关键的判断公式

```text
block 分配入口：
  new_blocks = kv_cache_manager.allocate_slots(...)

分配成功：
  new_blocks is not None

分配失败：
  new_blocks is None

running 阶段：
  while allocate_slots(...) is None:
      preempt one running request
      _preempt_request(preempted_req)
      retry allocate_slots(...)

waiting 阶段：
  if allocate_slots(...) is None:
      if request.has_encoder_inputs:
          encoder_cache_manager.free(request)
      break

allocate_slots 成功条件核心近似：
  required_blocks = num_blocks_to_allocate + watermark_blocks
  available_blocks = free_blocks - reserved_blocks
  required_blocks <= available_blocks

抢占结果：
  request.status = PREEMPTED
  request.num_computed_tokens = 0
  self.waiting.prepend_request(request)

释放结果：
  _free_request_blocks(request)
  del self.requests[request_id]  # 只有 _free_blocks() 中真正删除
```

---

## 38. 和前后问题的关系

`05_prefix_and_external_kv_hits.md` 解释了：

```text
Scheduler 如何得到 local / external computed tokens，
从而决定本轮还需要计算多少 token。
```

本篇解释的是：

```text
知道要计算多少 token 后，KV block 是否够，
不够时 running 请求如何抢占，waiting 请求为何停止。
```

接下来 `07_auxiliary_scheduling_features.md` 可以继续展开：

```text
encoder input、结构化输出、spec decode、LoRA、Mamba、pause、DP prefill balancing 等辅助调度能力，
如何嵌入上述 token budget 和 KV block 分配流程。
```
