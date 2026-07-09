# 03. 哪些 running 请求继续 decode / prefill？

源码位置：`vllm/vllm/v1/core/sched/scheduler.py`

本问题关注：Scheduler 每轮进入 `schedule()` 后，已经在 `self.running` 里的请求，哪些会继续推进 decode / prefill，哪些虽然仍然处于 running 但本轮会被跳过，以及 running 请求本轮具体会被安排多少 token。

这里的核心点是：vLLM V1 Scheduler 并没有把请求硬拆成“decode 阶段”和“prefill 阶段”两个独立流程。它统一用 `num_computed_tokens` 去追赶 `num_tokens_with_spec`。

源码注释里直接说明了这一点：

```python
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec.
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:389`

也就是说：

```text
running 请求是否继续 decode / prefill
  不是由一个显式的 phase 字段决定，
  而是由 num_computed_tokens 和 num_tokens_with_spec 的差值，
  再叠加 token budget、KV block、encoder、Mamba、PP/async 等限制共同决定。
```

---

## 1. 一句话回答

每轮 `schedule()` 会先扫描 `self.running`：

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
```

位置：`scheduler.py:431`

对于每个 running 请求，Scheduler 会依次判断：

```text
1. async / speculative 场景下是否已经确定达到 max_tokens；
2. PP / async cadence 是否允许本轮继续 decode；
3. DP prefill balancing 是否要求暂缓 prefill chunk；
4. 根据 num_tokens_with_spec + placeholders - num_computed_tokens 计算 num_new_tokens；
5. 用 long_prefill_token_threshold / token_budget / max_model_len 截断；
6. 如果有 encoder input，检查 encoder budget 和 encoder cache；
7. 如果有 Mamba block 对齐要求，继续裁剪 num_new_tokens；
8. 如果 num_new_tokens == 0，本轮跳过该 running 请求；
9. 尝试分配新增 KV block；
10. 分配成功则本轮调度它，分配失败则可能抢占 running 请求。
```

所以 running 请求本轮继续执行的必要条件可以概括为：

```text
请求在 self.running 中
and token_budget > 0
and 没有被 async/max_tokens/cadence/prefill throttle 跳过
and 计算出的 num_new_tokens > 0
and KV block 分配成功
```

调度成功后会记录到：

```python
scheduled_running_reqs.append(request)
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

位置：`scheduler.py:573`

---

## 2. running 请求的本质：已经进入执行流，但本轮不一定执行

`self.running` 保存的是已经进入模型执行流的请求。

初始化位置：

```python
self.running: list[Request] = []
```

位置：`scheduler.py:183`

一个请求进入 `self.running` 后，通常已经满足：

1. 被 Scheduler 从 waiting / skipped_waiting 队列取出；
2. 完成了必要的 prefix cache / external KV cache 判断；
3. 分配过 KV Cache block；
4. 进入过某一轮 `SchedulerOutput`；
5. `Request.status` 被设置为 `RUNNING`。

但是，“在 running 队列中”不等于“每一轮都会执行”。

原因是 running 请求每轮还要经过以下限制：

```text
async scheduling / max_tokens 限制
Pipeline Parallel decode cadence
DP prefill balancing
当前 token_budget
max_model_len
encoder compute budget
encoder cache 容量
Mamba block 对齐
KV block 是否可分配
```

所以更准确地说：

```text
self.running 表示请求已经处于可持续推进的执行流；
scheduled_running_reqs 表示本轮真正被安排执行的 running 请求。
```

两者不是一回事。

---

## 3. 为什么 Scheduler 不显式区分 decode 和 prefill

`schedule()` 开头有一段非常关键的注释：

```python
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec. num_tokens_with_spec =
# len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

位置：`scheduler.py:389`

这说明 vLLM V1 的 Scheduler 采用统一 token 追赶模型：

```text
目标 token 数：request.num_tokens_with_spec
已计算 token 数：request.num_computed_tokens
本轮需要补算：目标 token 数 - 已计算 token 数
```

因此：

```text
prefill：num_computed_tokens 还没追上 prompt / 当前输入 token；
decode：prompt 已经算完，当前只需要继续计算新生成 token；
chunked prefill：prompt 很长，一轮只计算其中一段；
spec decode：目标 token 数里还包含 draft/spec tokens；
async / PP：可能还有 output placeholders 参与计算。
```

这些都被统一进同一套公式，而不是写成多个完全分离的调度流程。

---

## 4. running 调度循环入口

每轮 `schedule()` 先初始化本轮状态：

```python
scheduled_running_reqs: list[Request] = []
preempted_reqs: list[Request] = []
req_to_new_blocks: dict[str, KVCacheBlocks] = {}
num_scheduled_tokens: dict[str, int] = {}
token_budget = self.max_num_scheduled_tokens
```

位置：`scheduler.py:402` 到 `scheduler.py:407`

如果当前 pause 状态是 `PAUSED_ALL`，则直接把预算清零：

```python
if self._pause_state == PauseState.PAUSED_ALL:
    token_budget = 0
```

位置：`scheduler.py:408`

running 阶段入口是：

```python
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
```

位置：`scheduler.py:430`

这说明：

```text
running 请求只有在 token_budget > 0 时才会被扫描。
```

如果预算一开始就是 0，或者前面的 running 请求已经耗尽预算，则后面的 running 请求本轮不会被考虑。

---

## 5. 第一类跳过：async / spec 场景下已经确定达到 max_tokens

running 循环里的第一个跳过条件是：

```python
if (
    request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens
):
    req_index += 1
    continue
```

位置：`scheduler.py:434`

这个逻辑主要服务于 async scheduling 和 speculative decoding。

### 5.1 `num_output_placeholders` 是什么

`num_output_placeholders` 表示已经被 Scheduler 预留、但 Worker 输出还没真正返回的 output token 位置。

在 async scheduling / PP 场景下，Scheduler 可能在上一轮 Worker 结果还没回来时，就先进入下一轮调度。如果不额外判断，就可能多调度一步 decode。

时间线可以理解为：

```text
T0: 第 N 轮 schedule 已经发出 decode / spec decode forward
T0: _update_after_schedule() 先把 num_computed_tokens 乐观推进
T0: AsyncScheduler 再增加 num_output_placeholders
T1: Worker 结果还没回来，Scheduler 进入第 N+1 轮
T1: request.num_output_placeholders > 0，表示已有输出占位在路上
T1: 如果这批输出回来后已经达到 max_tokens，本轮就不能再调度该请求
```

所以这个判断不是取消已经发出去的 forward，而是避免当前这轮再多发一步 forward。

### 5.2 为什么公式里有 `+ 2 - num_output_placeholders`

源码判断是：

```text
num_computed_tokens + 2 - num_output_placeholders
```

在这个判断点，`num_computed_tokens` 已经被 async schedule 乐观推进过，包含了在路上的 output placeholders 对应的计算进度。因此：

```text
num_computed_tokens - num_output_placeholders
```

会退回到“已确认输出对应的已计算输入位置”。对自回归 decode 来说，这个位置比当前已确认序列长度少 1，因为最后一个已确认 token 被 forward 后才会产生下一个 token。

所以需要再加 2：

```text
+1：从已计算输入位置回到当前已确认序列长度
+1：这次 in-flight target forward 至少会产生 1 个 sampled token
```

因此该公式表示：即使 speculative draft token 全部被拒绝，这个在路上的 target forward 返回后，请求至少会达到的序列长度。如果这个长度已经达到 `num_prompt_tokens + max_tokens`，就跳过该 running 请求。

### 5.3 这个跳过不会移出 running

这里执行的是：

```python
req_index += 1
continue
```

位置：`scheduler.py:447`

请求仍然留在 `self.running` 中，只是本轮不继续调度。

---

## 6. 第二类跳过：Pipeline Parallel / async decode cadence

第二个跳过条件是：

```python
if self.current_step < request.next_decode_eligible_step:
    req_index += 1
    continue
```

位置：`scheduler.py:450`

注释说明：

```python
# V2+PP+async: enforce `pp_size` steps between same-req decodes
# to match worker-side sampled-tokens broadcast slot ring cadence.
```

位置：`scheduler.py:451`

这表示在 V2 + Pipeline Parallel + async scheduling 场景下，同一个请求不能在任意连续 step 都 decode。

原因是 worker 侧 sampled token 的广播槽位有一个 ring cadence。Scheduler 需要保证同一个 request 的 decode 间隔满足 pipeline cadence，否则 worker 侧可能还没准备好消费这个请求的下一次 sampled token。

因此：

```text
current_step < next_decode_eligible_step
```

说明这个请求虽然在 running 中，但还没到下一次允许 decode 的 step。

同样，它不会被移出 `self.running`，只是本轮跳过。

---

## 7. 第三类跳过：DP prefill balancing 暂缓 prefill chunk

`schedule()` 支持一个参数：

```python
def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
```

位置：`scheduler.py:387`

如果开启 throttle，Scheduler 会计算：

```python
defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)
```

位置：`scheduler.py:425`

含义是：

```text
如果当前是被 throttle 的 step，
并且 prefill 容量没有被判定为瓶颈，
同时 running 里存在非 prefill chunk 的请求，
那么本轮先延后 prefill chunk，让 decode 请求填满这个 step。
```

running 阶段对应的跳过逻辑是：

```python
if defer_prefills and request.is_prefill_chunk:
    req_index += 1
    continue
```

位置：`scheduler.py:456`

这类请求仍然是 running，只是当前 step 不推进 prefill。

为什么要这么做？

```text
prefill 通常一次计算很多 token，容易打断 decode cadence；
DP prefill balancing 希望在某些非 cadence-aligned step 上优先保证 decode，
把 prefill chunk 延后到更合适的 step。
```

所以：

```text
有 running prefill chunk
并不表示它本轮一定继续 prefill。
```

---

## 8. running 请求本轮 token 数的核心公式

如果没有被前面几个条件跳过，Scheduler 开始计算本轮要给这个 running 请求安排多少 token。

核心公式是：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

位置：`scheduler.py:462`

可以理解为：

```text
num_new_tokens = 目标应计算 token 数 - 已经计算 token 数
```

各字段含义：

| 字段 | 含义 |
|---|---|
| `request.num_tokens_with_spec` | 当前请求 token 总数，包含 prompt、已生成 output、以及 speculative draft token |
| `request.num_output_placeholders` | async / PP 场景中已经预留但还没返回的输出 token |
| `request.num_computed_tokens` | Scheduler 认为已经完成计算的 token 数 |
| `num_new_tokens` | 本轮还需要补算的 token 数 |

### 8.1 普通 decode 的例子

假设：

```text
prompt = 100 token
已经生成 output = 5 token
没有 spec token
没有 placeholder
num_tokens_with_spec = 105
num_computed_tokens = 104
```

那么：

```text
num_new_tokens = 105 - 104 = 1
```

这就是普通 decode：本轮只需要计算 1 个 token 位置。

### 8.2 chunked prefill 的例子

假设：

```text
prompt = 10000 token
当前没有 output
num_tokens_with_spec = 10000
num_computed_tokens = 4096
```

那么：

```text
num_new_tokens = 10000 - 4096 = 5904
```

这个请求虽然已经在 running 中，但它仍然处于 prefill 没算完的阶段。后续还会被 token budget、long prefill threshold 等条件裁剪成 chunk。

### 8.3 speculative decode 的例子

假设：

```text
prompt + output = 105 token
spec_token_ids = 4 个 draft token
num_tokens_with_spec = 109
num_computed_tokens = 104
```

那么：

```text
num_new_tokens = 109 - 104 = 5
```

这通常表示本轮要让 target model 验证 4 个 draft token，并额外采样 1 个 token。

---

## 9. running 请求 token 数的第一层裁剪：long prefill threshold

计算出 `num_new_tokens` 后，先受到 `long_prefill_token_threshold` 限制：

```python
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
```

位置：`scheduler.py:467`

这个限制主要用于长 prefill / chunked prefill。

如果一个 running 请求还剩很多 prompt token 没计算，而一次性安排它会吃掉大量 batch budget，那么 long prefill threshold 会把它切小。

例如：

```text
num_new_tokens = 12000
long_prefill_token_threshold = 2048
```

裁剪后：

```text
num_new_tokens = 2048
```

这样可以避免一个长 prefill 请求独占本轮所有 token budget。

---

## 10. 第二层裁剪：本轮剩余 token_budget

接着会裁剪到当前剩余预算以内：

```python
num_new_tokens = min(num_new_tokens, token_budget)
```

位置：`scheduler.py:469`

这一步保证：

```text
单个 running 请求本轮安排的 token 数不能超过本轮剩余 token_budget。
```

例如：

```text
当前 token_budget = 512
request 还需要 num_new_tokens = 2048
```

裁剪后：

```text
num_new_tokens = 512
```

这也是为什么 running 请求之间存在预算竞争：

```text
越靠前被调度的 running 请求越先消耗 token_budget；
后面的 running 请求只能使用剩余预算。
```

---

## 11. 第三层裁剪：max_model_len

接着还要保证输入位置不超过模型最大长度：

```python
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len
    - request.num_computed_tokens
    - self.num_sampled_tokens_per_step,
)
```

位置：`scheduler.py:473`

这里减掉 `self.num_sampled_tokens_per_step` 是为了给采样 token 留位置。

普通自回归模型中：

```python
self.num_sampled_tokens_per_step = 1
```

位置：`scheduler.py:119`

所以 running 请求最多只能计算到：

```text
max_model_len - 1
```

因为本轮 forward 后还要采样下一个 token。

如果是某些不采样 token 的模型或路径，`num_sampled_tokens_per_step` 可能是 0。

这一步非常重要，尤其是 speculative decoding 场景。spec token 会让 `num_tokens_with_spec` 提前超过普通 decode 的 1 token 范围，如果不限制 max model length，就可能调度超过模型上下文上限的位置。

---

## 12. 第四层限制：encoder input 调度

如果 running 请求包含 encoder input，例如多模态 image/audio/video 输入，Scheduler 还要检查本轮 decoder token 范围内是否覆盖了必须处理的 encoder input。

判断入口：

```python
if request.has_encoder_inputs:
    (
        encoder_inputs_to_schedule,
        num_new_tokens,
        new_encoder_compute_budget,
        external_load_encoder_input,
    ) = self._try_schedule_encoder_inputs(...)
```

位置：`scheduler.py:484`

`_try_schedule_encoder_inputs()` 的职责是：

```text
判断当前本轮 decoder token 范围内，哪些 encoder input 必须先被计算或加载；
如果 encoder budget / encoder cache 不够，就缩短 num_new_tokens；
必要时只调度到那个无法处理的 encoder input 之前。
```

函数注释说明：

```python
# Determine which encoder inputs need to be scheduled in the current step,
# and update `num_new_tokens` and encoder token budget accordingly.
```

位置：`scheduler.py:1287`

它会查找本轮 token 窗口覆盖的多模态 feature：

```python
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
)
```

位置：`scheduler.py:1321`

如果本轮碰到了某个 encoder input，但 encoder compute budget 不够，或者 encoder cache 没空间，则 `num_new_tokens` 可能被缩短，甚至变成 0。

因此：

```text
running 请求 token_budget 够，不代表一定能推进；
如果前方 token 依赖一个无法安排的 encoder input，本轮也可能被截断或跳过。
```

---

## 13. 第五层限制：Mamba block 对齐

如果模型包含 Mamba 层，并且 cache 配置要求 block 对齐，Scheduler 还会执行：

```python
if self.need_mamba_block_aligned_split:
    num_new_tokens = self._mamba_block_aligned_split(
        request, num_new_tokens
    )
```

位置：`scheduler.py:498`

`_mamba_block_aligned_split()` 的核心逻辑是：

```python
num_new_tokens = num_new_tokens // block_size * block_size
```

位置：`scheduler.py:364`

目的：

```text
让 Mamba state cache 的 prefill chunk 尽量按 block size 对齐。
```

原因是 Mamba state cache 对 chunk 边界更敏感。如果 chunk 切得不对齐，可能导致 Mamba cache miss，或者无法缓存某些 state。

不过源码里也有例外：

```python
# As an exception, if `num_new_tokens` is less than `block_size`, the
# state is simply not cached, requiring no special handling.
```

位置：`scheduler.py:351`

所以 Mamba 对齐不是简单地永远向下取整，而是结合当前是否还处于 prefill、是否跨过最后 cache position、是否 Eagle 模式等条件决定。

当前源码还包含 Marconi cache admission 优化：如果未缓存公共前缀长度足够长，并且本轮 token 数超过该公共前缀，会把 `num_new_tokens` 截到公共前缀长度再按 block 对齐：

```python
if (
    num_uncached_common_prefix_tokens >= block_size
    and num_new_tokens > num_uncached_common_prefix_tokens
):
    num_new_tokens = num_uncached_common_prefix_tokens
    num_new_tokens = num_new_tokens // block_size * block_size
```

位置：`scheduler.py:376` 到 `scheduler.py:384`

对于 running 调度来说，结论是：

```text
Mamba block 对齐可能把 num_new_tokens 裁小；
如果裁剪后变成 0，本轮该 running 请求会被跳过。
```

---

## 14. `num_new_tokens == 0` 时为什么是 continue 而不是 break

所有裁剪之后，如果：

```python
if num_new_tokens == 0:
    req_index += 1
    continue
```

位置：`scheduler.py:503`

Scheduler 会跳过当前 running 请求，继续看后面的 running 请求。

源码注释列了几类原因：

```python
# 1. No new tokens to schedule. This may happen when
#    (1) PP>1 and we have already scheduled all prompt tokens
#    but they are not finished yet.
#    (2) Async scheduling and the request has reached to either
#    its max_total_tokens or max_model_len.
# 2. The encoder budget is exhausted.
# 3. The encoder cache is exhausted.
# 4. Insufficient budget for a block-aligned chunk in hybrid
#    models with mamba cache mode "align".
```

位置：`scheduler.py:503`

这里选择 `continue` 而不是 `break` 很重要。

源码注释解释：

```python
# NOTE(woosuk): Here, by doing `continue` instead of `break`,
# we do not strictly follow the FCFS scheduling policy and
# allow the lower-priority requests to be scheduled.
```

位置：`scheduler.py:515`

也就是说：

```text
如果当前 running 请求因为 encoder / Mamba / PP / async 等原因不能推进，
Scheduler 不会让它阻塞整个 running 队列，
而是继续尝试后面的 running 请求。
```

这点和 waiting 阶段某些 `break` 行为不同。running 请求已经占据执行流，跳过某个暂时不可推进的请求后，继续推进其他请求通常更有利于利用 GPU。

---

## 15. KV block 分配：running 请求真正能否执行的关键

如果 `num_new_tokens > 0`，Scheduler 接下来要为新增 token 分配 KV block：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

位置：`scheduler.py:524`

这里的含义是：

```text
这个 running 请求本轮要多计算 num_new_tokens 个 token，
因此需要确保 KV Cache 中有足够 block 容纳这些新增 token 的 KV。
```

如果 `allocate_slots()` 返回非 `None`，说明分配成功，请求可以被调度。

如果返回 `None`，说明 KV Cache 空间不足，需要进入抢占逻辑。

---

## 16. running 阶段的抢占逻辑

当当前 running 请求需要新增 block，但 block 不够时，Scheduler 会抢占某个 running 请求。

源码结构是：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(...)

    if new_blocks is not None:
        break

    # The request cannot be scheduled.
    # Preempt the lowest-priority request.
    ...
```

位置：`scheduler.py:523`

### 16.1 PRIORITY 策略下抢占谁

如果调度策略是 `PRIORITY`：

```python
preempted_req = max(
    self.running,
    key=lambda r: (r.priority, r.arrival_time),
)
self.running.remove(preempted_req)
```

位置：`scheduler.py:536`

这里会选择优先级最低的 running 请求。

注意：这里用 `max` 是因为在该实现中，排序键更大的请求表示更应该被抢占：

```text
数值更大的 `priority`（调度优先级更低）/ 更晚 `arrival_time` 的请求更可能被抢占。
```

### 16.2 非 PRIORITY 策略下抢占谁

如果不是 PRIORITY：

```python
preempted_req = self.running.pop()
```

位置：`scheduler.py:561`

通常就是 running 队尾请求。

### 16.3 如果抢占了本轮已经调度过的请求

PRIORITY 模式下，有可能抢占到本轮前面已经调度过的 running 请求。

这时 Scheduler 会撤销它本轮的调度记录：

```python
scheduled_running_reqs.remove(preempted_req)
token_budget += num_scheduled_tokens.pop(preempted_req_id)
req_to_new_blocks.pop(preempted_req_id)
scheduled_spec_decode_tokens.pop(preempted_req_id, None)
```

位置：`scheduler.py:542`

如果它本轮还安排过 encoder input，也要恢复 encoder compute budget：

```python
encoder_compute_budget += num_embeds_to_restore
```

位置：`scheduler.py:558`

这说明 running 阶段的抢占不是简单地“释放一个请求”，还要保证本轮已经构造的调度计划保持一致。

### 16.4 被抢占请求会回到 waiting

真正抢占通过：

```python
self._preempt_request(preempted_req, scheduled_timestamp)
```

位置：`scheduler.py:563`

`_preempt_request()` 会释放请求 block / encoder cache，并把状态改为 `PREEMPTED`，再放回 waiting 队列。

核心迁移是：

```text
self.running / RUNNING
  → KV block 不够，被抢占
  → self.waiting / PREEMPTED
```

如果被抢占的正是当前正在尝试调度的 request：

```python
if preempted_req == request:
    break
```

位置：`scheduler.py:565`

表示已经没有更多请求可以抢占来满足它了，当前请求本轮无法调度。

---

## 17. block 仍然分配失败时，running 阶段停止

如果抢占后还是拿不到 block：

```python
if new_blocks is None:
    # Cannot schedule this request.
    break
```

位置：`scheduler.py:569`

这里是 `break`，不是 `continue`。

原因是：

```text
KV block 已经不足到无法满足当前请求；
继续看后面的 running 请求通常也难以成功，
并且此时可能已经发生抢占，需要结束 running 阶段并进入后续输出构造。
```

另外，后续 waiting 阶段还有一个重要条件：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

位置：`scheduler.py:625`

也就是说：

```text
只要本轮发生过抢占，Scheduler 就不会继续接纳新的 waiting 请求进入 running。
```

这是为了避免刚因为 KV block 不足抢占了 running 请求，又继续拉入新的 waiting 请求，造成资源震荡。

---

## 18. running 请求调度成功后记录哪些信息

如果 KV block 分配成功，Scheduler 会记录本轮调度结果：

```python
scheduled_running_reqs.append(request)
request_id = request.request_id
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
req_index += 1
```

位置：`scheduler.py:573`

这几个字段作用不同：

| 字段 | 含义 |
|---|---|
| `scheduled_running_reqs` | 本轮真正继续执行的 running 请求 |
| `req_to_new_blocks` | 每个请求本轮新增分配的 KV block |
| `num_scheduled_tokens` | 每个请求本轮安排计算多少 token |
| `token_budget` | 本轮剩余 token 预算 |

注意：

```text
scheduled_running_reqs 是 self.running 的子集。
```

有些 running 请求可能因为 cadence、prefill throttle、encoder budget、Mamba、max_tokens 等原因被跳过，不会出现在 `scheduled_running_reqs` 中。

---

## 19. running 阶段的 speculative decode 记录

如果请求带有 speculative decode token：

```python
if request.spec_token_ids:
```

位置：`scheduler.py:582`

Scheduler 会计算本轮实际安排了多少 spec token：

```python
num_scheduled_spec_tokens = (
    num_new_tokens
    + request.num_computed_tokens
    - request.num_tokens
    - request.num_output_placeholders
)
```

位置：`scheduler.py:583`

这个公式可以理解为：

```text
本轮调度 token 中，超过 request.num_tokens / placeholders 之外的部分，
就是实际调度到的 draft/spec token。
```

如果大于 0，就把本轮用到的 spec token 写入：

```python
scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
```

位置：`scheduler.py:593`

然后清空请求上的旧 draft token：

```python
request.spec_token_ids = []
```

位置：`scheduler.py:597`

后续 Worker 返回后，如果部分 draft token 被拒绝，会在 `update_from_output()` 中回退：

```python
request.num_computed_tokens -= num_rejected
```

位置：`scheduler.py:1562`

如果 async scheduling 中 placeholder 也包含 spec token，还会同步回退：

```python
request.num_output_placeholders -= num_rejected
```

位置：`scheduler.py:1566`

所以 speculative decode 的完整闭环是：

```text
schedule()
  → 把 spec_token_ids 计入 num_tokens_with_spec
  → 本轮可能调度 draft tokens
  → 记录 scheduled_spec_decode_tokens
  → 清空 request.spec_token_ids
  → update_from_output() 根据接受/拒绝情况修正 num_computed_tokens
```

---

## 20. running 阶段的 encoder input 记录

如果 `_try_schedule_encoder_inputs()` 决定本轮要本地计算 encoder input：

```python
if encoder_inputs_to_schedule:
    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
    for i in encoder_inputs_to_schedule:
        self.encoder_cache_manager.allocate(request, i)
        if self.ec_connector is not None:
            self.ec_connector.update_state_after_alloc(request, i)
    encoder_compute_budget = new_encoder_compute_budget
```

位置：`scheduler.py:600`

这表示：

```text
本轮 SchedulerOutput 会告诉 Worker：
这些请求除了 decoder token 外，还要处理哪些 encoder input。
```

如果 encoder input 不需要本地计算，而是可以从外部 encoder cache 加载：

```python
if external_load_encoder_input:
    for i in external_load_encoder_input:
        self.encoder_cache_manager.allocate(request, i)
        if self.ec_connector is not None:
            self.ec_connector.update_state_after_alloc(request, i)
```

位置：`scheduler.py:608`

这类输入不会按普通本地 encoder compute 消耗预算，但仍要占用 encoder cache，并通知 ECConnector 构造元数据。

---

## 21. running 请求如何进入 SchedulerOutput

running 阶段结束后，Scheduler 会构造 `SchedulerOutput`。

对于本轮继续执行的 running 请求，它们不会作为 `scheduled_new_reqs` 发送，而是进入 cached request 数据：

```python
cached_reqs_data = self._make_cached_request_data(
    scheduled_running_reqs,
    scheduled_resumed_reqs,
    num_scheduled_tokens,
    scheduled_spec_decode_tokens,
    req_to_new_blocks,
)
```

位置：`scheduler.py:1031`

然后写入：

```python
scheduler_output = SchedulerOutput(
    scheduled_cached_reqs=cached_reqs_data,
    num_scheduled_tokens=num_scheduled_tokens,
    scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
    scheduled_encoder_inputs=scheduled_encoder_inputs,
    ...
)
```

位置：`scheduler.py:1057`

所以：

```text
running 请求本轮如果被继续调度，通常会体现在 scheduled_cached_reqs 中。
```

为什么叫 cached request？

因为这些请求通常已经在 ModelRunner 的 persistent batch / cache 状态里，不需要像新请求一样重新发送所有初始化信息。

---

## 22. `_make_cached_request_data()` 里 running 请求携带什么

`_make_cached_request_data()` 会为 scheduled running 请求构造：

```python
req_ids
new_token_ids
new_block_ids
all_token_ids
num_computed_tokens
num_output_tokens
```

位置：`scheduler.py:1219`

其中：

```python
req_ids.append(req_id)
```

位置：`scheduler.py:1237`

表示本轮有哪些 cached/running 请求要执行。

如果使用 Pipeline Parallel 且不是 async scheduling，需要把 sampled token 传回去：

```python
if self.use_pp and not self.scheduler_config.async_scheduling:
    token_ids = req.all_token_ids[
        req.num_computed_tokens : req.num_computed_tokens + num_tokens
    ]
    new_token_ids.append(token_ids)
```

位置：`scheduler.py:1242`

如果请求上一轮没有被调度过，就需要传完整 token ids：

```python
if not scheduled_in_prev_step:
    all_token_ids[req_id] = req.all_token_ids.copy()
```

位置：`scheduler.py:1259`

同时传递本轮新增 block：

```python
new_block_ids.append(
    req_to_new_blocks[req_id].get_block_ids(allow_none=True)
)
```

位置：`scheduler.py:1261`

以及调度前的计算进度：

```python
num_computed_tokens.append(req.num_computed_tokens)
num_output_tokens.append(
    req.num_output_tokens + req.num_output_placeholders
)
```

位置：`scheduler.py:1264`

这里有一个重要细节：

```text
SchedulerOutput 中给 Worker 的 num_computed_tokens，
是 _update_after_schedule() 之前的值。
```

因为 Worker 需要知道本轮 input 从哪里开始取。Scheduler 内部状态会在输出构造之后再前移。

---

## 23. 调度后如何更新 running 请求进度

构造完 `SchedulerOutput` 后，Scheduler 会调用：

```python
self._update_after_schedule(scheduler_output)
```

位置：`scheduler.py:1096`

核心逻辑是：

```python
request.num_computed_tokens += num_scheduled_token
```

位置：`scheduler.py:1139`

这表示：

```text
只要 Scheduler 本轮已经把 token 安排给 Worker，
它就先把这些 token 计入 num_computed_tokens。
```

即使 Worker 输出还没有返回，Scheduler 也要先推进内部进度。原因是 async scheduling / pipeline parallel 下，下一轮 `schedule()` 可能会在上一轮输出回来之前发生。如果不先推进，就可能重复调度同一段 token。

### 23.1 更新 `is_prefill_chunk`

同时会更新：

```python
request.is_prefill_chunk = request.num_computed_tokens < (
    request.num_tokens + request.num_output_placeholders
)
```

位置：`scheduler.py:1145`

这个字段表示请求当前是否还处于 prefill chunk 状态。

可以理解为：

```text
如果 num_computed_tokens 还没追上 request.num_tokens + placeholders，
说明仍然有已有输入 token 没算完，是 prefill chunk；
否则就不再是 prefill chunk。
```

这会影响下一轮 DP prefill balancing：

```python
if defer_prefills and request.is_prefill_chunk:
    continue
```

位置：`scheduler.py:456`

### 23.2 routed experts 的 block 快照

如果开启返回 routed experts，Scheduler 会在 forward 开始前保存本轮调度请求的 block ids。这样即使 async scheduling 后续抢占并释放了请求 block，`update_from_output()` 仍然能按本轮执行时的 block 布局读取 routed experts：

```python
if self.enable_return_routed_experts:
    gid = self.routed_experts_mgr.attn_gid
    self._re_block_ids.update(
        {
            rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
            for rid in num_scheduled_tokens
        }
    )
```

位置：`scheduler.py:1155` 到 `scheduler.py:1169`

### 23.3 从 `_inflight_prefills` 移除

如果请求已经不是 prefill chunk：

```python
if not request.is_prefill_chunk:
    self._inflight_prefills.discard(request)
```

位置：`scheduler.py:1152`

表示该请求不再被视为正在飞行中的 prefill。

---

## 24. running 请求本轮执行结果什么时候真正生效

`_update_after_schedule()` 只是 Scheduler 侧的“乐观推进”。真正的模型输出要等 Worker 返回后，在 `update_from_output()` 中处理。

对普通 decode 来说，Worker 返回 sampled token 后，会追加到 request 输出里：

```python
new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
```

位置：`scheduler.py:1589`

对 speculative decode 来说，如果 draft token 被拒绝，会回退 `num_computed_tokens`：

```python
request.num_computed_tokens -= num_rejected
```

位置：`scheduler.py:1562`

所以 running 请求的进度更新分两层：

```text
schedule 阶段：
  _update_after_schedule() 先把已调度 token 计入 num_computed_tokens

update_from_output 阶段：
  根据 Worker 实际输出修正 output token、stop 状态、spec 接受/拒绝、资源释放
```

这也是为什么 `num_computed_tokens` 不是简单等于“Worker 已经返回的 token 数”，而是 Scheduler 在 async/pipeline 场景下维护的调度进度。

---

## 25. running 请求继续 decode / prefill 的完整流程

把上面的逻辑串起来，running 阶段可以写成下面的伪代码：

```text
for request in self.running while token_budget > 0:
    if async placeholders show request will reach max_tokens:
        skip this request

    if current_step < request.next_decode_eligible_step:
        skip this request

    if defer_prefills and request.is_prefill_chunk:
        skip this request

    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )

    num_new_tokens = min(num_new_tokens, long_prefill_token_threshold)
    num_new_tokens = min(num_new_tokens, token_budget)
    num_new_tokens = min(num_new_tokens, max_model_len remaining space)

    if request.has_encoder_inputs:
        adjust num_new_tokens by encoder budget/cache constraints

    if need_mamba_block_aligned_split:
        adjust num_new_tokens by Mamba block alignment / common-prefix admission

    if num_new_tokens == 0:
        skip this request

    new_blocks = allocate_slots(request, num_new_tokens)
    while new_blocks is None:
        preempt one running request
        if no more request can be preempted:
            stop running scheduling
        retry allocate_slots

    record this request as scheduled_running_reqs
    record num_scheduled_tokens[request_id]
    record req_to_new_blocks[request_id]
    token_budget -= num_new_tokens

    if request has spec_token_ids:
        record scheduled_spec_decode_tokens

    if request has encoder inputs scheduled:
        allocate encoder cache and record scheduled_encoder_inputs
```

---

## 26. 一个具体例子：普通 decode 请求

假设：

```text
max_num_scheduled_tokens = 8192
本轮 token_budget = 8192
running:
  req-a:
    prompt = 100 token
    output = 5 token
    num_tokens_with_spec = 105
    num_computed_tokens = 104
    no placeholder
    no spec token
```

调度过程：

```text
1. req-a 没有被 async/max_tokens 跳过
2. cadence 允许 decode
3. 不是 prefill chunk，或没有 defer_prefills
4. num_new_tokens = 105 - 104 = 1
5. token_budget 够，max_model_len 够
6. allocate_slots(req-a, 1) 成功
7. num_scheduled_tokens[req-a] = 1
8. token_budget = 8191
9. req-a 进入 scheduled_running_reqs
```

这就是一个典型的 running decode step。

---

## 27. 一个具体例子：running chunked prefill

假设：

```text
max_num_scheduled_tokens = 8192
running:
  req-b:
    prompt = 20000 token
    output = 0
    num_tokens_with_spec = 20000
    num_computed_tokens = 4096
    long_prefill_token_threshold = 4096
```

初始计算：

```text
num_new_tokens = 20000 - 4096 = 15904
```

经过 long prefill threshold：

```text
num_new_tokens = 4096
```

如果 token_budget 够，KV block 也够，本轮会继续 prefill 4096 token。

调度后：

```text
_update_after_schedule():
  num_computed_tokens = 4096 + 4096 = 8192
```

因为：

```text
8192 < 20000
```

所以：

```text
request.is_prefill_chunk = True
```

下一轮它仍然是 running 中的 prefill chunk，可能继续被调度，也可能因为 DP prefill balancing 被暂缓。

---

## 28. 一个具体例子：running 请求本轮被跳过

假设：

```text
running:
  req-c:
    is_prefill_chunk = True
    当前 schedule(throttle_prefills=True)
    defer_prefills = True
```

则在 running 阶段：

```python
if defer_prefills and request.is_prefill_chunk:
    req_index += 1
    continue
```

位置：`scheduler.py:456`

结果：

```text
req-c 仍然留在 self.running；
req-c 本轮不进入 scheduled_running_reqs；
req-c 不消耗 token_budget；
req-c 的 num_computed_tokens 不会在本轮增加。
```

这类情况不能理解为请求阻塞或结束，只是 Scheduler 当前 step 选择不推进它。

---

## 29. 一个具体例子：KV block 不够导致抢占

假设：

```text
running:
  req-a: 正在 decode
  req-b: 正在 chunked prefill
  req-c: 正在 decode
```

当前调度 req-a 时需要新增 KV block，但：

```python
allocate_slots(req-a, num_new_tokens) == None
```

Scheduler 会抢占一个 running 请求。

FCFS / 非 PRIORITY 下：

```python
preempted_req = self.running.pop()
```

位置：`scheduler.py:561`

如果队尾是 req-c，则 req-c 被抢占：

```text
req-c 从 self.running 移除
req-c 释放 KV blocks
req-c.status = PREEMPTED
req-c.num_computed_tokens = 0
req-c 被放回 waiting 队列头部
```

然后 Scheduler 重新尝试给 req-a 分配 block。

如果这次成功，req-a 本轮继续执行；如果还是失败，可能继续抢占，直到成功或者把当前请求也抢占掉。

---

## 30. 容易疑惑的点

### 30.1 running 请求一定是 decode 吗？

不是。

running 请求可能处于：

```text
普通 decode
chunked prefill
speculative decode
等待 PP / async cadence
等待 encoder budget
Mamba 对齐后暂时无法推进
```

是否是 prefill chunk，主要看：

```python
request.is_prefill_chunk
```

这个字段会在 `_update_after_schedule()` 中更新。

### 30.2 running 请求本轮一定会出现在 SchedulerOutput 里吗？

不一定。

只有本轮真正调度成功的 running 请求，才会进入：

```text
scheduled_running_reqs
scheduled_cached_reqs
num_scheduled_tokens
```

被跳过的 running 请求仍留在 `self.running`，但本轮不会出现在 `num_scheduled_tokens` 中。

### 30.3 `continue` 和 `break` 有什么区别？

running 阶段中：

```text
num_new_tokens == 0 → continue
```

表示当前请求暂时不能推进，但后面的 running 请求可能可以推进。

而：

```text
KV block 分配失败且抢占后仍失败 → break
```

表示资源不足已经影响本轮 running 调度，继续扫描意义不大。

### 30.4 running 阶段发生抢占后，还会调度 waiting 吗？

不会。

waiting 阶段入口要求：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

位置：`scheduler.py:625`

只要本轮发生抢占，就不接纳新的 waiting 请求。

### 30.5 `num_computed_tokens` 是 Worker 已经返回的 token 数吗？

不完全是。

Scheduler 在 `_update_after_schedule()` 中会先把本轮已安排的 token 加到 `num_computed_tokens`：

```python
request.num_computed_tokens += num_scheduled_token
```

位置：`scheduler.py:1139`

这样做是为了 async / PP 场景下避免重复调度。Worker 返回后，如果 spec token 被拒绝，再回退。

---

## 31. 从“回答问题”的角度总结

如果要问：

```text
哪些 running 请求会继续 decode / prefill？
```

Scheduler 的答案不是简单地说“所有 self.running 请求都会继续执行”，而是：

```text
从 self.running 头到尾扫描；
跳过当前 step 不符合条件的请求；
为可推进请求计算 num_new_tokens；
经过 token budget / max_model_len / encoder / Mamba 等限制；
成功分配 KV block 后，才会把它加入 scheduled_running_reqs。
```

最终本轮真正执行的 running 请求是：

```python
scheduled_running_reqs
```

本轮每个请求执行多少 token 是：

```python
num_scheduled_tokens[request_id]
```

本轮新增的 KV block 是：

```python
req_to_new_blocks[request_id]
```

它们最终会进入：

```python
SchedulerOutput.scheduled_cached_reqs
SchedulerOutput.num_scheduled_tokens
SchedulerOutput.scheduled_spec_decode_tokens
SchedulerOutput.scheduled_encoder_inputs
```

---

## 32. 最关键的判断公式

```text
running 请求本轮候选 token 数：
  num_new_tokens = (
      request.num_tokens_with_spec
      + request.num_output_placeholders
      - request.num_computed_tokens
  )

本轮可调度条件：
  request in self.running
  and token_budget > 0
  and current_step >= request.next_decode_eligible_step
  and not (defer_prefills and request.is_prefill_chunk)
  and not async/max_tokens skip
  and num_new_tokens after all clipping > 0
  and allocate_slots(request, num_new_tokens) succeeds

调度成功后：
  scheduled_running_reqs.append(request)
  num_scheduled_tokens[request_id] = num_new_tokens
  req_to_new_blocks[request_id] = new_blocks
  token_budget -= num_new_tokens

调度后内部推进：
  request.num_computed_tokens += num_scheduled_tokens[request_id]
  request.is_prefill_chunk = (
      request.num_computed_tokens
      < request.num_tokens + request.num_output_placeholders
  )
```

---

## 33. 和前两个问题的关系

`01_request_states.md` 解释了：

```text
哪些请求在 self.running / self.waiting / self.skipped_waiting 中。
```

`02_token_budget.md` 解释了：

```text
一轮 schedule 总共最多能安排多少 token。
```

本篇进一步解释：

```text
在 self.running 里的请求，如何竞争并消耗本轮 token_budget；
哪些 running 请求真正进入本轮 SchedulerOutput；
哪些 running 请求虽然还在 running，但当前 step 被跳过。
```

所以三者连起来是：

```text
请求在哪里？
  → 01_request_states.md

本轮总预算是多少？
  → 02_token_budget.md

running 请求怎么用掉这些预算？
  → 03_running_decode_prefill.md
```
