# 04. 哪些 waiting 请求可以进入运行态？

源码位置：`D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`

本问题关注：Scheduler 在处理完 `self.running` 后，如何从 `self.waiting` / `self.skipped_waiting` 中选择请求，判断它是否可以进入 `self.running`，以及哪些条件会导致 waiting 请求本轮不能进入运行态。

一句话概括：

```text
waiting 请求能否进入 running，取决于：
本轮是否允许接纳新请求、running 数量是否未满、token budget 是否还有剩余、
blocked 状态能否恢复、LoRA / KV Connector / encoder / Mamba 等约束是否满足，
以及 KV Cache block 是否能分配成功。
```

---

## 1. 一句话回答

Scheduler 每轮先调度 running 请求，然后才考虑 waiting 请求。

waiting 阶段入口是：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:625`

这说明 waiting 请求能进入 running 的第一前提是：

```text
本轮 running 阶段没有发生抢占；
并且 Scheduler 当前允许接纳新请求。
```

进入 waiting 循环后：

```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
```

位置：`scheduler.py:628`

所以 waiting 请求进入 running 的必要条件可以概括为：

```text
1. 本轮没有 preempted_reqs；
2. pause_state == UNPAUSED；
3. token_budget > 0；
4. len(self.running) < self.max_num_running_reqs；
5. 能从 self.waiting / self.skipped_waiting 选出一个队头请求；
6. 如果请求是 blocked waiting status，需要先恢复成功；
7. 没有因为 LoRA 限制被本轮跳过；
8. prefix cache / external KV cache 查询能得到确定结果；
9. 如果有 encoder cache / ECConnector 约束，需要满足；
10. 能计算出本轮要调度的 num_new_tokens；
11. chunked prefill / token budget / encoder / Mamba 等限制后仍可调度；
12. KV Cache block 分配成功；
13. 如果不是 async remote KV load，才会 append 到 self.running 并设置 RUNNING。
```

调度成功后的核心动作是：

```python
request = request_queue.pop_request()
self.running.append(request)
request.status = RequestStatus.RUNNING
```

位置：`scheduler.py:916`、`scheduler.py:939`、`scheduler.py:958`

---

## 2. waiting 调度只发生在 running 阶段之后

`schedule()` 的主流程是：

```text
schedule()
  → 初始化 token_budget
  → 先调度 self.running
  → 如果没有抢占且未暂停新请求，再调度 waiting / skipped_waiting
  → 构造 SchedulerOutput
  → _update_after_schedule()
```

waiting 阶段在 running 阶段之后：

```python
# Next, schedule the WAITING requests.
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

位置：`scheduler.py:624`

这有两个重要含义。

### 2.1 running 请求优先使用 token_budget

在 waiting 阶段开始前，running 请求已经消耗了一部分甚至全部 `token_budget`。

因此 waiting 请求只能使用：

```text
running 阶段之后剩余的 token_budget。
```

这和 `02_token_budget.md` 中的结论一致：

```text
本轮预算先给 running，再给 waiting。
```

### 2.2 running 阶段发生抢占后，不再接纳 waiting

如果 running 阶段因为 KV Cache block 不够发生过抢占，`preempted_reqs` 非空，waiting 阶段不会执行。

这可以避免一种资源震荡：

```text
刚因为 block 不够把 running 请求抢占回 waiting，
又在同一轮把新的 waiting 请求拉进 running。
```

所以只要本轮发生抢占，本轮就不会再接纳新的 waiting 请求。

---

## 3. pause 状态对 waiting 调度的影响

waiting 阶段要求：

```python
self._pause_state == PauseState.UNPAUSED
```

位置：`scheduler.py:625`

Scheduler 有三种 pause 状态：

| 状态 | waiting 调度行为 |
|---|---|
| `UNPAUSED` | 可以调度 waiting 请求 |
| `PAUSED_NEW` | 不接纳新 waiting 请求，只允许已有 running 继续推进 |
| `PAUSED_ALL` | 不调度任何请求，`token_budget` 会被置为 0 |

`PAUSED_ALL` 在 `schedule()` 开头处理：

```python
if self._pause_state == PauseState.PAUSED_ALL:
    token_budget = 0
```

位置：`scheduler.py:408`

而 `PAUSED_NEW` 不会直接把 `token_budget` 置 0，因为 running 请求仍然可以继续推进。但它会让 waiting 阶段条件不成立。

因此：

```text
PAUSED_NEW：running 可以继续，waiting 不进入 running。
PAUSED_ALL：running 和 waiting 都不推进。
```

---

## 4. waiting 循环入口：必须有等待请求且 token_budget > 0

waiting 循环是：

```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
```

位置：`scheduler.py:628`

这说明 waiting 阶段继续运行需要两个条件：

```text
1. self.waiting 或 self.skipped_waiting 至少一个非空；
2. 本轮 token_budget 仍然大于 0。
```

如果 running 阶段已经把 token budget 用完，则 waiting 请求不会被扫描。

不过要注意：外部 KV async load 是一个特殊路径，它可能令 `num_new_tokens = 0`，不消耗 token budget，但 waiting 循环本身仍要求进入循环时 `token_budget > 0`。

---

## 5. running 数量限制：`max_num_running_reqs`

waiting 循环一开始会检查：

```python
if len(self.running) == self.max_num_running_reqs:
    break
```

位置：`scheduler.py:629`

这里限制的是同时 running 的请求数量，不是 token 数量。

`self.max_num_running_reqs` 来自：

```python
self.max_num_running_reqs = self.scheduler_config.max_num_seqs
```

位置：`scheduler.py:107`

因此：

```text
即使 token_budget 还有剩余，
只要 running 请求数已经达到 max_num_seqs，
Scheduler 就不会再从 waiting 中接纳新请求。
```

这体现了 Scheduler 同时受两个维度限制：

```text
token 维度：max_num_scheduled_tokens / token_budget
sequence 维度：max_num_running_reqs / max_num_seqs
```

---

## 6. 从 `waiting` 还是 `skipped_waiting` 里取请求

waiting 请求不只来自 `self.waiting`，也可能来自 `self.skipped_waiting`。

每次循环会先选择一个队列：

```python
request_queue = self._select_waiting_queue_for_scheduling()
assert request_queue is not None
request = request_queue.peek_request()
```

位置：`scheduler.py:632`

选择逻辑是：

```python
def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
    if self.policy == SchedulingPolicy.FCFS:
        return self.skipped_waiting or self.waiting or None

    if self.waiting and self.skipped_waiting:
        waiting_req = self.waiting.peek_request()
        skipped_req = self.skipped_waiting.peek_request()
        return self.waiting if waiting_req < skipped_req else self.skipped_waiting

    return self.waiting or self.skipped_waiting or None
```

位置：`scheduler.py:1818`

### 6.1 FCFS 策略

FCFS 下优先返回：

```python
self.skipped_waiting or self.waiting or None
```

位置：`scheduler.py:1819`

也就是说，如果 `skipped_waiting` 非空，会优先尝试之前被跳过的请求。

这样可以避免：

```text
某个请求因为一轮 LoRA / KV Connector / encoder cache 等条件被跳过后，
长期被新来的 waiting 请求插队。
```

### 6.2 PRIORITY 策略

PRIORITY 下，如果两个队列都有请求，会比较两个队头：

```python
waiting_req = self.waiting.peek_request()
skipped_req = self.skipped_waiting.peek_request()
return self.waiting if waiting_req < skipped_req else self.skipped_waiting
```

位置：`scheduler.py:1823`

这说明：

```text
PRIORITY 策略下，waiting / skipped_waiting 不是固定谁优先，
而是由队头请求的优先级比较决定。
```

### 6.3 `peek_request()` 而不是立即 `pop_request()`

这里先 `peek_request()`，不是马上 `pop_request()`。

原因是：

```text
后续还有很多检查可能导致 break / continue；
只有确认请求要进入 running，或确认要放入 step_skipped_waiting 时，才会真正 pop。
```

---

## 7. blocked waiting 请求要先尝试恢复

队头请求可能处于 blocked waiting 状态。

blocked 状态判断函数是：

```python
@staticmethod
def _is_blocked_waiting_status(status: RequestStatus) -> bool:
    return status in (
        RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_STREAMING_REQ,
    )
```

位置：`scheduler.py:1804`

waiting 循环中，如果队头请求是 blocked 状态，会先尝试恢复：

```python
if self._is_blocked_waiting_status(
    request.status
) and not self._try_promote_blocked_waiting_request(request):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

位置：`scheduler.py:638`

含义是：

```text
blocked 请求不会直接进入 running；
Scheduler 每轮会尝试把它恢复到 WAITING / PREEMPTED；
如果恢复失败，就继续放回 skipped_waiting，等待下一轮再试。
```

---

## 8. 三类 blocked waiting 状态如何恢复

恢复逻辑集中在：

```python
def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
```

位置：`scheduler.py:2384`

### 8.1 `WAITING_FOR_REMOTE_KVS`

远端 KV 异步加载完成前，请求不能进入 running。

判断条件是：

```python
if request.request_id not in self.finished_recving_kv_req_ids:
    return False
```

位置：`scheduler.py:2392`

如果 Worker 还没报告 finished_recving，则恢复失败，继续留在 skipped waiting。

如果已经完成，则：

```python
self._update_waiting_for_remote_kv(request)
if request.num_preemptions:
    request.status = RequestStatus.PREEMPTED
else:
    request.status = RequestStatus.WAITING
return True
```

位置：`scheduler.py:2394`

迁移关系：

```text
WAITING_FOR_REMOTE_KVS
  → Worker 报告 finished_recving
  → _update_waiting_for_remote_kv()
  → WAITING 或 PREEMPTED
```

### 8.2 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`

结构化输出请求需要 grammar 准备好。

```python
structured_output_req = request.structured_output_request
if not (structured_output_req and structured_output_req.grammar):
    return False
request.status = RequestStatus.WAITING
return True
```

位置：`scheduler.py:2401`

迁移关系：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
  → grammar ready
  → WAITING
```

### 8.3 `WAITING_FOR_STREAMING_REQ`

streaming input 等待不会在普通 schedule 循环中自动恢复：

```python
if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
    assert not request.streaming_queue
    return False
```

位置：`scheduler.py:2408`

它需要 `add_request()` 收到同一个 request id 的新 input chunk，再通过 `_update_request_as_session()` 改回 `WAITING`。

迁移关系：

```text
WAITING_FOR_STREAMING_REQ
  -- add_request 收到新输入 --> WAITING
```

---

## 9. LoRA 数量限制可能导致本轮跳过

如果启用了 LoRA，Scheduler 要保证一轮调度中的 LoRA 种类不超过 `max_loras`。

waiting 阶段检查：

```python
if (
    self.lora_config
    and request.lora_request
    and (
        len(scheduled_loras) == self.lora_config.max_loras
        and request.lora_request.lora_int_id not in scheduled_loras
    )
):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

位置：`scheduler.py:653`

含义是：

```text
如果本轮已经调度了 max_loras 种 LoRA，
而当前 waiting 请求使用的是一种新的 LoRA，
则本轮不能调度这个请求。
```

这类请求会被放入 `step_skipped_waiting`。

注意：这不是 blocked status。

请求的状态可能仍然是：

```python
RequestStatus.WAITING
```

只是本轮由于 LoRA 并发种类限制被临时跳过。

这也是为什么 `skipped_waiting` 不能简单等同于 blocked requests。

---

## 10. waiting 请求先查本地 prefix cache

waiting 请求被接纳进入 running 前，会先计算已经可复用的 token。

只有当：

```python
if request.num_computed_tokens == 0:
```

位置：`scheduler.py:672`

才会查询本地 prefix cache 和外部 KV cache。

普通路径下，本地 prefix cache 查询是：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

位置：`scheduler.py:709`

含义：

| 字段 | 含义 |
|---|---|
| `new_computed_blocks` | 本地 prefix cache 命中的 KV blocks |
| `num_new_local_computed_tokens` | 本地 prefix cache 命中的 token 数 |

如果是 Hybrid + Mamba + connector 场景，会走特殊路径：

```python
computed, per_group_hits = (
    self.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(...)
)
num_new_local_computed_tokens = max(per_group_hits)
```

位置：`scheduler.py:682`

这部分细节会在 `05_prefix_and_external_kv_hits.md` 中专门展开。这里先抓住主线：

```text
waiting 请求不是一定从第 0 个 token 开始 prefill；
Scheduler 会先用本地 prefix cache 命中来减少本轮需要计算的 token。
```

---

## 11. 再查外部 KV cache

如果配置了 KV Connector，Scheduler 还会查询外部 KV cache：

```python
if self.connector is not None:
    ext_tokens, load_kv_async = (
        self.connector.get_num_new_matched_tokens(
            request, num_new_local_computed_tokens
        )
    )
```

位置：`scheduler.py:722`

这里把本地命中的 token 数传给 connector。

原因是 connector 需要返回：

```text
外部 KV cache 相对本地 prefix cache 额外命中的 token 数。
```

例如：

```text
prompt 总长度 = 10000
本地 prefix cache 命中 = 3000
外部 KV cache 总共可命中 = 8000
那么 connector 应返回外部新增命中 = 5000
```

最后总命中是：

```python
num_computed_tokens = (
    num_new_local_computed_tokens + num_external_computed_tokens
)
```

位置：`scheduler.py:745`

并且必须满足：

```python
assert num_computed_tokens <= request.num_tokens
```

位置：`scheduler.py:748`

---

## 12. external KV 查询不确定时，waiting 请求会被跳过

KV Connector 可能暂时无法确定外部命中数量。

如果返回：

```python
ext_tokens is None
```

Scheduler 会执行：

```python
request_queue.pop_request()
step_skipped_waiting.prepend_request(request)
continue
```

位置：`scheduler.py:729`

含义是：

```text
当前请求本轮无法继续判断，
先放入 step_skipped_waiting，
不要阻塞整个 waiting 调度循环。
```

这类请求通常仍然可能是 `WAITING` 状态，不是 blocked status。

所以这里再次体现：

```text
skipped_waiting = blocked requests + temporarily skipped requests
```

---

## 13. ECConnector / encoder cache 预取不可用时跳过

在计算出 `num_computed_tokens` 后，如果配置了 ECConnector，并且请求有多模态特征，Scheduler 还会检查远端 encoder cache 是否可用：

```python
if (
    self.ec_connector is not None
    and request.mm_features
    and not self.ec_connector.ensure_cache_available(
        request, num_computed_tokens
    )
):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

位置：`scheduler.py:750`

含义是：

```text
如果当前请求依赖的 encoder cache / 多模态预取还没准备好，
本轮先跳过这个 waiting 请求，后续再尝试。
```

这同样是临时跳过，不一定改变 `Request.status`。

---

## 14. async remote KV load：特殊的 waiting 路径

外部 KV cache 命中后，connector 可能要求异步加载远端 KV。

此时：

```python
if load_kv_async:
    assert num_external_computed_tokens > 0
    num_new_tokens = 0
```

位置：`scheduler.py:781`

这表示：

```text
本轮不做本地模型 forward；
本轮只是为远端 KV load 分配本地 block，并生成 connector metadata。
```

后面分配 block 成功并 pop 请求后，会走特殊路径：

```python
if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    step_skipped_waiting.prepend_request(request)
    request.num_computed_tokens = num_computed_tokens
    self._inflight_prefills.add(request)
    continue
```

位置：`scheduler.py:917`

也就是说 async remote KV load 不会让请求进入 running。

它的迁移关系是：

```text
self.waiting / WAITING
  → 查询外部 KV 命中，load_kv_async=True
  → allocate_slots(delay_cache_blocks=True)
  → RequestStatus.WAITING_FOR_REMOTE_KVS
  → step_skipped_waiting / self.skipped_waiting
  → Worker 完成 KV load
  → 下轮 schedule 尝试恢复为 WAITING / PREEMPTED
```

这个路径很关键：

```text
分配了 KV block，不代表请求一定进入 running；
async remote KV load 会先进入 WAITING_FOR_REMOTE_KVS。
```

---

## 15. DP prefill balancing 可能停止接纳新 prefill

如果不是 async KV load，并且当前开启了 prefill defer：

```python
elif defer_prefills and request.num_computed_tokens == 0:
    break
```

位置：`scheduler.py:785`

注释说明：

```python
# DP prefill balancing: async KV loads (the branch above) are
# allowed to start even on throttled steps, but committing new
# prefill compute is deferred to a cadence-aligned step.
```

位置：`scheduler.py:786`

含义是：

```text
在被 throttle 的 step 上，Scheduler 可以启动 async KV load，
但不会提交新的本地 prefill 计算。
```

这里是 `break`，不是 `continue`。

也就是说：

```text
当前队头 waiting prefill 不调度时，
本轮 waiting 阶段停止，不绕过它去调度后面的 waiting 请求。
```

---

## 16. 普通 waiting 请求如何计算 `num_new_tokens`

如果不是 async KV load，也没有因为 defer prefill 停止，则计算：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

位置：`scheduler.py:795`

这里用的是 `request.num_tokens`，不是 `request.num_prompt_tokens`。

源码注释解释：

```python
# We use `request.num_tokens` instead of
# `request.num_prompt_tokens` to consider the resumed
# requests, which have output tokens.
```

位置：`scheduler.py:791`

原因是 PREEMPTED / resumed 请求可能已经有 output token。恢复时不只是重新 prefill prompt，还要把当前请求已有的 token 序列考虑进去。

因此：

```text
num_new_tokens = 当前请求已有 token 总数 - 本地/外部已计算 token 数
```

例子：

```text
request.num_tokens = 12000
本地 prefix cache 命中 = 4000
外部 KV cache 命中 = 3000
num_computed_tokens = 7000
num_new_tokens = 12000 - 7000 = 5000
```

Scheduler 本轮只需要安排剩余 5000 token。

---

## 17. waiting 请求 token 数的第一层裁剪：long prefill threshold

普通 waiting 请求算出 `num_new_tokens` 后，会先受 long prefill threshold 限制：

```python
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold
```

位置：`scheduler.py:796`

这和 running 阶段类似，用于避免单个长 prompt 一次吃掉太多 token budget。

例如：

```text
num_new_tokens = 12000
long_prefill_token_threshold = 4096
```

裁剪后：

```text
num_new_tokens = 4096
```

---

## 18. chunked prefill 对 waiting 调度的影响

接下来是一个非常重要的分支：

```python
if (
    not self.scheduler_config.enable_chunked_prefill
    and num_new_tokens > token_budget
):
    break
```

位置：`scheduler.py:802`

含义是：

```text
如果关闭 chunked prefill，
waiting 请求必须能在本轮完整安排它剩余的 prefill token；
如果剩余 token 超过当前 token_budget，本轮 waiting 调度停止。
```

如果启用了 chunked prefill，则允许只安排一部分：

```python
num_new_tokens = min(num_new_tokens, token_budget)
assert num_new_tokens > 0
```

位置：`scheduler.py:810`

### 18.1 为什么关闭 chunked prefill 时是 break

这里是 `break`，不是 `continue`。

原因是：

```text
队头 waiting 请求无法完整 prefill，
如果绕过它去调度后面的 waiting 请求，会破坏 FCFS / priority 队列语义，
也可能让长 prompt 长期饥饿。
```

所以 Scheduler 选择停止 waiting 阶段。

---

## 19. encoder input 可能缩短或阻止 waiting 调度

如果 waiting 请求有 encoder input，会调用：

```python
if request.has_encoder_inputs:
    (
        encoder_inputs_to_schedule,
        num_new_tokens,
        new_encoder_compute_budget,
        external_load_encoder_input,
    ) = self._try_schedule_encoder_inputs(...)
    if num_new_tokens == 0:
        break
```

位置：`scheduler.py:813`

`_try_schedule_encoder_inputs()` 会检查本轮 decoder token 范围是否覆盖 image/audio/video 等 encoder input。

如果覆盖到了某个必须先处理的 encoder input，但：

```text
encoder compute budget 不够；
或 encoder cache 没有空间；
或不允许切分多模态输入；
```

那么 `num_new_tokens` 可能被缩短到 encoder input 之前。

如果缩短后为 0，waiting 阶段停止。

所以：

```text
token_budget 够，不代表 waiting 请求一定能进入 running；
如果它前方依赖无法安排的 encoder input，本轮也不能调度。
```

---

## 20. Mamba block 对齐可能继续裁剪 waiting token 数

如果需要 Mamba block 对齐，且不是 async KV load：

```python
if self.need_mamba_block_aligned_split and not load_kv_async:
    num_new_tokens = self._mamba_block_aligned_split(
        request,
        num_new_tokens,
        num_new_local_computed_tokens,
        num_external_computed_tokens,
        num_uncached_common_prefix_tokens,
    )
    if num_new_tokens == 0:
        break
```

位置：`scheduler.py:832`

Mamba 对齐的目标是让 prefill chunk 尽量按 block 边界切分，便于 Mamba state cache。

如果对齐后 `num_new_tokens` 变成 0，Scheduler 会停止 waiting 调度。

这里同样是 `break`。

---

## 21. lookahead tokens 与 async KV load 的特殊处理

在分配 KV block 前，Scheduler 会计算本轮有效的 lookahead token 数：

```python
limit_lookahead_tokens = load_kv_async and self.use_eagle
effective_lookahead_tokens = (
    0 if limit_lookahead_tokens else self.num_lookahead_tokens
)
```

位置：`scheduler.py:848`

注释说明这是为处理 P/D Disaggregation + Spec Decoding 的边界情况：

```text
async KV load 与 Eagle/spec decode 同时存在时，
额外 lookahead block 可能导致本地和远端 block 数不匹配，
因此在该场景下限制 lookahead tokens。
```

普通 waiting 请求则使用：

```text
effective_lookahead_tokens = self.num_lookahead_tokens
```

---

## 22. encoder-decoder cross-attention block 数

对于 encoder-decoder 模型，如果本轮安排了 encoder input，还要计算 cross-attention 需要的 block：

```python
num_encoder_tokens = 0
if (
    self.is_encoder_decoder
    and request.has_encoder_inputs
    and encoder_inputs_to_schedule
):
    num_encoder_tokens = sum(
        request.get_num_encoder_embeds(i)
        for i in encoder_inputs_to_schedule
    )
```

位置：`scheduler.py:853`

这些 `num_encoder_tokens` 会传给 KV cache manager 的 `allocate_slots()`。

含义是：

```text
waiting 请求进入 running 前，不只要考虑 decoder token 的 KV block，
encoder-decoder 模型还要考虑 cross-attention 相关 block。
```

---

## 23. async KV load 的 reserved blocks

如果是 async KV load，Scheduler 会计算 reserved blocks：

```python
reserved_blocks = 0
if load_kv_async:
    reserved_blocks = self._inflight_prefill_reserved_blocks()
```

位置：`scheduler.py:865`

注释解释：

```python
# An async load holds its blocks for the whole transfer with
# no forward progress and isn't preemptible here. Admit it
# only if it fits in (free - other in-flight reservations), to
# avoid deadlock and predictable preemptions.
```

位置：`scheduler.py:867`

含义是：

```text
async load 会长期占用 block，且在这里不能像普通 running 请求一样被抢占；
因此 Scheduler 要预留其它 in-flight prefill 的 block，
避免 async load 占满空间后造成死锁或必然抢占。
```

---

## 24. waiting 请求分配 KV block

经过前面的所有判断后，Scheduler 调用：

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

位置：`scheduler.py:873`

waiting 阶段的 block 分配比 running 阶段复杂，因为它要同时处理：

| 参数 | 含义 |
|---|---|
| `num_new_tokens` | 本轮要实际计算的 token 数 |
| `num_new_computed_tokens` | 本地 prefix cache 命中的 token 数 |
| `new_computed_blocks` | 本地 prefix cache 命中的 block |
| `num_external_computed_tokens` | 外部 KV cache 命中的 token 数 |
| `delay_cache_blocks` | async KV load 时，先分配 block 但延迟正式 cache |
| `num_lookahead_tokens` | spec decode / lookahead 需要预留的 token |
| `num_encoder_tokens` | encoder-decoder cross-attention 相关 block |
| `reserved_blocks` | async load 场景下为其它 in-flight 请求预留的 block |
| `has_scheduled_reqs` | 当前是否已经有 scheduled/running 请求影响分配策略 |

如果分配失败：

```python
if new_blocks is None:
    if request.has_encoder_inputs:
        self.encoder_cache_manager.free(request)
    break
```

位置：`scheduler.py:887`

也就是说：

```text
KV block 分配失败时，当前 waiting 请求不能进入 running，
waiting 阶段停止。
```

注意：waiting 阶段分配失败不会像 running 阶段那样在这里抢占 running 请求。

---

## 25. 分配后通知 KV Connector

如果存在 KV Connector，分配 block 成功后会通知 connector：

```python
if self.connector is not None:
    self.connector.update_state_after_alloc(
        request,
        self.kv_cache_manager.get_blocks(request_id),
        num_external_computed_tokens,
    )
```

位置：`scheduler.py:900`

这一步非常关键。

它告诉 connector：

```text
1. 当前请求本地分配到了哪些 KV block；
2. 有多少 token 来自外部 KV cache；
3. connector 后续可以根据这些信息构造 load / save metadata。
```

对于 KVPool / P-D disaggregation 场景，这里就是 Scheduler 和外部 KV 系统对接的重要插入点。

---

## 26. async KV load 分配成功后不会进入 running

block 分配成功后，Scheduler 先把请求从原队列弹出：

```python
request = request_queue.pop_request()
```

位置：`scheduler.py:916`

如果是 async KV load：

```python
if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    step_skipped_waiting.prepend_request(request)
    request.num_computed_tokens = num_computed_tokens
    self._inflight_prefills.add(request)
    continue
```

位置：`scheduler.py:917`

这里有几个关键点：

```text
1. 请求已经从 waiting / skipped_waiting 原队列取出；
2. 请求状态变为 WAITING_FOR_REMOTE_KVS；
3. 请求被放入 step_skipped_waiting；
4. request.num_computed_tokens 先设置为命中 token 数；
5. 请求不会 append 到 self.running；
6. 本轮不会记录 num_scheduled_tokens；
7. 后续等 Worker 完成 KV load 后再恢复。
```

注释中特别说明：

```text
虽然 KV 还没真正加载完成，但先设置 num_computed_tokens；
如果后续 connector 报告 transfer error，会在 _update_requests_with_invalid_blocks 中修正；
transfer 完成后，_update_waiting_for_remote_kv 会只 cache 成功加载的 token。
```

这说明 async KV load 是一种介于 waiting 和 running 之间的特殊状态。

---

## 27. 普通 waiting 请求正式进入 running

如果不是 async KV load，则 block 分配成功后请求会进入 running：

```python
self.running.append(request)
```

位置：`scheduler.py:939`

如果开启 stats，会记录调度事件：

```python
request.record_event(
    EngineCoreEventType.SCHEDULED, scheduled_timestamp
)
```

位置：`scheduler.py:940`

然后根据原始状态区分新请求和恢复请求：

```python
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request)
else:
    raise RuntimeError(f"Invalid request status: {request.status}")
```

位置：`scheduler.py:944`

最后设置状态和调度记录：

```python
req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
    request_id
)
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens
```

位置：`scheduler.py:953`

迁移关系是：

```text
self.waiting / WAITING
  → block 分配成功
  → scheduled_new_reqs
  → self.running / RUNNING

self.waiting / PREEMPTED
  → block 分配成功
  → scheduled_resumed_reqs
  → self.running / RUNNING
```

如果请求来自 `skipped_waiting`，只要它已经恢复成 `WAITING` 或 `PREEMPTED`，也遵循同样逻辑。

---

## 28. 为什么先设置 `num_computed_tokens` 为 cache 命中数

进入 running 后：

```python
request.num_computed_tokens = num_computed_tokens
```

位置：`scheduler.py:959`

这里的 `num_computed_tokens` 是：

```text
本地 prefix cache 命中 token 数 + 外部 KV cache 命中 token 数
```

也就是说，Scheduler 先把已经可复用的 KV 计入请求进度。

然后本轮实际调度的新 token 会记录在：

```python
num_scheduled_tokens[request_id] = num_new_tokens
```

位置：`scheduler.py:956`

等 `SchedulerOutput` 构造完成后，`_update_after_schedule()` 会再执行：

```python
request.num_computed_tokens += num_scheduled_token
```

位置：`scheduler.py:1139`

所以 waiting 请求进入 running 的进度更新分两步：

```text
进入 running 时：
  request.num_computed_tokens = cache 命中 token 数

_update_after_schedule()：
  request.num_computed_tokens += 本轮实际调度 token 数
```

例子：

```text
prompt = 10000
本地 prefix cache 命中 = 3000
外部 KV cache 命中 = 4000
本轮调度 num_new_tokens = 3000
```

进入 running 时：

```text
num_computed_tokens = 7000
```

调度后：

```text
num_computed_tokens = 7000 + 3000 = 10000
```

这样下一轮 Scheduler 就知道 prompt 已经 prefill 完成，可以进入 decode。

---

## 29. `_inflight_prefills` 如何记录 waiting 请求

如果本轮调度后仍然没有 prefill 完：

```python
if num_computed_tokens + num_new_tokens < request.num_tokens:
    self._inflight_prefills.add(request)
```

位置：`scheduler.py:960`

含义：

```text
这个请求已经进入 running，
但当前仍然只是完成了部分 prefill，
后续还需要继续 chunked prefill。
```

async KV load 也会加入 `_inflight_prefills`：

```python
self._inflight_prefills.add(request)
```

位置：`scheduler.py:936`

但二者语义略有不同：

```text
普通 chunked prefill：请求已经在 running 中，并且本轮做了本地计算。
async KV load：请求在 WAITING_FOR_REMOTE_KVS 中，等待远端 KV load 完成。
```

---

## 30. waiting 请求的 encoder input 记录

如果本轮需要本地计算 encoder input：

```python
if encoder_inputs_to_schedule:
    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
    for i in encoder_inputs_to_schedule:
        self.encoder_cache_manager.allocate(request, i)
        if self.ec_connector is not None:
            self.ec_connector.update_state_after_alloc(request, i)
    encoder_compute_budget = new_encoder_compute_budget
```

位置：`scheduler.py:964`

如果 encoder input 来自外部 encoder cache：

```python
if external_load_encoder_input:
    for i in external_load_encoder_input:
        self.encoder_cache_manager.allocate(request, i)
        if self.ec_connector is not None:
            self.ec_connector.update_state_after_alloc(request, i)
```

位置：`scheduler.py:972`

这些信息会进入 `SchedulerOutput.scheduled_encoder_inputs` 或 ECConnector metadata，供 Worker / connector 执行。

---

## 31. 本轮 skipped 请求如何合并回 `self.skipped_waiting`

waiting 阶段开始时创建一个临时队列：

```python
step_skipped_waiting = create_request_queue(self.policy)
```

位置：`scheduler.py:626`

本轮临时跳过的请求会进入这个队列，例如：

```text
blocked 状态恢复失败；
LoRA 种类超过 max_loras；
KV Connector 暂时无法确定命中；
ECConnector / encoder cache 暂不可用；
async remote KV load 进入 WAITING_FOR_REMOTE_KVS。
```

waiting 阶段结束后合并回：

```python
if step_skipped_waiting:
    self.skipped_waiting.prepend_requests(step_skipped_waiting)
```

位置：`scheduler.py:979`

注释说明：

```python
# re-queue requests skipped in this pass ahead of older skipped items.
```

位置：`scheduler.py:979`

也就是说：

```text
本轮刚跳过的请求会被放回 self.skipped_waiting，
并且排在更早的 skipped items 前面。
```

下一轮 `schedule()` 会再次根据策略尝试它们。

---

## 32. DP prefill balancing 如何记录是否 capacity-bound

waiting 阶段最后还有一段：

```python
if not defer_prefills:
    self.prefill_capacity_bound = bool(self.waiting)
```

位置：`scheduler.py:983`

含义是：

```text
如果当前不是 defer_prefills 的 step，
则根据 waiting 队列是否仍有请求，记录 prefill 是否受容量限制。
```

如果 waiting 队列还有请求没被接纳，说明本轮可能是 capacity-bound。

这个状态会影响后续 DP prefill balancing 中：

```python
defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)
```

位置：`scheduler.py:425`

也就是说：

```text
waiting 阶段是否能接纳足够 prefill，会反过来影响后续 step 是否继续 defer prefill。
```

---

## 33. waiting 请求如何进入 SchedulerOutput

成功进入 running 的 waiting 请求会被记录到不同列表。

### 33.1 全新请求：`scheduled_new_reqs`

如果原状态是：

```python
RequestStatus.WAITING
```

则进入：

```python
scheduled_new_reqs.append(request)
```

位置：`scheduler.py:944`

这类请求通常是第一次进入模型执行流。

### 33.2 被抢占后恢复：`scheduled_resumed_reqs`

如果原状态是：

```python
RequestStatus.PREEMPTED
```

则进入：

```python
scheduled_resumed_reqs.append(request)
```

位置：`scheduler.py:946`

这类请求之前运行过，因为 KV block 不够等原因被抢占，后来重新进入 running。

它和普通 running 请求的关键区别是：抢占时旧 KV blocks 已经释放，甚至可能被其它请求复用；Worker 侧虽然可能还保留 request state，但旧 block table 不能继续使用。

```text
scheduled_running_reqs:
  旧 block table 有效，新 blocks 追加。

scheduled_resumed_reqs:
  旧 block table 失效，新 blocks 替换。
```

因此旧 model runner 会通过 `scheduled_cached_reqs.resumed_req_ids` 标记恢复请求，让 Worker 复用仍有效的 request state，同时替换 KV block table。

### 33.3 输出构造差异

构造 `SchedulerOutput` 时，普通新请求进入：

```python
scheduled_new_reqs=new_reqs_data
```

位置：`scheduler.py:1058`

running / resumed 请求会参与：

```python
scheduled_cached_reqs=cached_reqs_data
```

位置：`scheduler.py:1059`

不过在 V2 ModelRunner 下，会把 resumed 请求合并进 new requests：

```python
if self.use_v2_model_runner:
    scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
    scheduled_resumed_reqs = []
```

位置：`scheduler.py:1012`

语义上可以理解为：

```text
WAITING → scheduled_new_reqs → running
PREEMPTED → scheduled_resumed_reqs 或 V2 下合并进 scheduled_new_reqs → running
```

---

## 34. 一个完整例子：普通 waiting 进入 running

假设当前状态：

```text
token_budget = 8192
len(self.running) < max_num_running_reqs
preempted_reqs = []
pause_state = UNPAUSED

waiting 队头 req-a:
  status = WAITING
  request.num_tokens = 12000
  本地 prefix cache 命中 = 4000
  外部 KV cache 命中 = 0
  enable_chunked_prefill = True
```

调度过程：

```text
1. waiting 阶段入口条件成立；
2. running 数未达上限；
3. 选择 self.waiting 队列；
4. req-a 不是 blocked status；
5. LoRA 限制通过；
6. 本地 prefix cache 命中 4000；
7. num_computed_tokens = 4000；
8. num_new_tokens = 12000 - 4000 = 8000；
9. token_budget 够；
10. allocate_slots 成功；
11. request_queue.pop_request()；
12. self.running.append(req-a)；
13. scheduled_new_reqs.append(req-a)；
14. request.status = RUNNING；
15. request.num_computed_tokens = 4000；
16. num_scheduled_tokens[req-a] = 8000。
```

之后 `_update_after_schedule()` 会把：

```text
request.num_computed_tokens = 4000 + 8000 = 12000
```

这说明 prompt 已经完成 prefill，后续可以进入 decode。

---

## 35. 一个完整例子：关闭 chunked prefill 导致 waiting 停止

假设：

```text
token_budget = 4096
waiting 队头 req-b:
  request.num_tokens = 12000
  num_computed_tokens = 0
  enable_chunked_prefill = False
```

则：

```text
num_new_tokens = 12000
num_new_tokens > token_budget
not enable_chunked_prefill
```

触发：

```python
break
```

位置：`scheduler.py:802`

结果：

```text
req-b 本轮不进入 running；
后面的 waiting 请求也不会被绕过调度；
本轮 waiting 阶段结束。
```

如果开启 chunked prefill，则会变成：

```text
num_new_tokens = min(12000, 4096) = 4096
```

然后只 prefill 一部分。

---

## 36. 一个完整例子：external KV async load

假设：

```text
waiting 队头 req-c:
  status = WAITING
  request.num_tokens = 10000
  本地 prefix cache 命中 = 0
  外部 KV cache 命中 = 10000
  load_kv_async = True
```

则：

```text
num_external_computed_tokens = 10000
num_computed_tokens = 10000
num_new_tokens = 0
```

Scheduler 会：

```text
1. allocate_slots(..., delay_cache_blocks=True)；
2. connector.update_state_after_alloc(...)；
3. 从 waiting 队列 pop 出 req-c；
4. request.status = WAITING_FOR_REMOTE_KVS；
5. request.num_computed_tokens = 10000；
6. req-c 进入 step_skipped_waiting；
7. req-c 不进入 self.running；
8. 本轮不消耗 token_budget 做本地 forward。
```

后续：

```text
Worker 完成远端 KV load
  → update_from_output() 记录 finished_recving_kv_req_ids
  → 下一轮 schedule() 中 _try_promote_blocked_waiting_request()
  → 恢复为 WAITING / PREEMPTED
  → 再尝试进入 running
```

---

## 37. 一个完整例子：PREEMPTED 请求恢复 running

假设：

```text
req-d 之前在 running 中，后来因为 KV block 不够被抢占；
_preempt_request() 已经把它放回 waiting；
状态是 PREEMPTED；
num_preemptions > 0。
```

后续某轮 waiting 调度中：

```text
1. req-d 被选中；
2. prefix cache / external KV cache 重新查询；
3. 重新分配 KV block；
4. 分配成功；
5. self.running.append(req-d)；
6. 因为 request.status == PREEMPTED，进入 scheduled_resumed_reqs；
7. request.status = RUNNING。
```

迁移关系：

```text
RUNNING
  → 被抢占
  → PREEMPTED / waiting
  → 重新调度成功
  → RUNNING
```

注意：PREEMPTED 请求恢复时，也可能通过 prefix cache / external KV cache 复用之前已经算过的 KV。

---

## 38. 容易疑惑的点

### 38.1 waiting 阶段一定会执行吗？

不一定。

以下情况 waiting 阶段不会执行：

```text
1. running 阶段发生了抢占；
2. pause_state 是 PAUSED_NEW；
3. pause_state 是 PAUSED_ALL；
4. running 阶段已经耗尽 token_budget；
5. waiting / skipped_waiting 都为空。
```

### 38.2 `skipped_waiting` 里的请求会重新尝试吗？

会。

`_select_waiting_queue_for_scheduling()` 会同时考虑 `waiting` 和 `skipped_waiting`。

FCFS 下甚至优先选择：

```python
self.skipped_waiting or self.waiting
```

位置：`scheduler.py:1819`

### 38.3 cache 命中后是不是一定进入 running？

不是。

cache 命中只是减少 `num_new_tokens`。

请求仍然可能因为以下原因不能进入 running：

```text
external KV async load；
LoRA 限制；
encoder cache / ECConnector 不可用；
chunked prefill 关闭且 token_budget 不够；
Mamba 对齐后 num_new_tokens 为 0；
KV block 分配失败；
running 请求数达到 max_num_running_reqs。
```

### 38.4 async KV load 为什么不进入 running？

因为 async KV load 本轮不做本地模型 forward。

它只是：

```text
分配本地 block；
生成 connector metadata；
等待 Worker 从远端加载 KV；
加载完成后再恢复调度。
```

所以它进入：

```python
RequestStatus.WAITING_FOR_REMOTE_KVS
```

而不是：

```python
RequestStatus.RUNNING
```

### 38.5 为什么有些地方是 `continue`，有些地方是 `break`？

大致规则是：

```text
continue：
  当前请求只是临时跳过，不应该阻塞后面的 waiting 请求。
  例如 blocked 恢复失败、LoRA 限制、KV Connector 暂时无法确定命中。

break：
  当前队头请求无法调度，且不应该绕过它继续调度后面的请求，
  或者资源已经不足以继续 waiting 调度。
  例如 running 数达到上限、关闭 chunked prefill 且 budget 不够、
  encoder / Mamba 导致 num_new_tokens 为 0、KV block 分配失败。
```

### 38.6 PREEMPTED 请求算新请求吗？

不完全算。

它会重新从 waiting 进入 running，但记录到：

```python
scheduled_resumed_reqs
```

而不是普通的：

```python
scheduled_new_reqs
```

V2 ModelRunner 下打包方式可能合并，但语义上仍然表示“恢复请求”。

---

## 39. 从“回答问题”的角度总结

如果要问：

```text
哪些 waiting 请求可以进入运行态？
```

Scheduler 的回答是：

```text
在 running 阶段没有抢占、Scheduler 未暂停新请求、token_budget 还有剩余、
running 数量未达上限时，
从 waiting / skipped_waiting 中选择队头请求，
先恢复 blocked 状态，再检查 LoRA、prefix cache、external KV cache、encoder、Mamba、KV block 等约束。
只有普通本地计算路径分配 block 成功后，才会进入 self.running 并变成 RUNNING。
```

正式进入 running 的动作是：

```python
request = request_queue.pop_request()
self.running.append(request)
request.status = RequestStatus.RUNNING
```

同时记录：

```python
num_scheduled_tokens[request_id] = num_new_tokens
req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(request_id)
```

而以下请求不会在本轮进入 running：

```text
blocked 状态恢复失败的请求；
因 LoRA 限制临时跳过的请求；
KV Connector 无法确定命中数的请求；
ECConnector / encoder cache 暂不可用的请求；
async remote KV load 请求；
chunked prefill 关闭且本轮 budget 不够完整 prefill 的请求；
encoder / Mamba 限制后 num_new_tokens 为 0 的请求；
KV block 分配失败的请求。
```

---

## 40. 最关键的判断公式

```text
waiting 阶段入口：
  not preempted_reqs
  and self._pause_state == UNPAUSED
  and token_budget > 0
  and (self.waiting or self.skipped_waiting)

sequence 数量限制：
  len(self.running) < self.max_num_running_reqs

队列选择：
  FCFS:
    self.skipped_waiting or self.waiting
  PRIORITY:
    compare self.waiting.peek_request() and self.skipped_waiting.peek_request()

blocked 请求：
  if _is_blocked_waiting_status(request.status):
    must _try_promote_blocked_waiting_request(request) == True

cache 命中：
  num_computed_tokens = (
      num_new_local_computed_tokens
      + num_external_computed_tokens
  )

普通 waiting 请求本轮 token 数：
  num_new_tokens = request.num_tokens - num_computed_tokens

进入 running 的核心条件：
  not load_kv_async
  and num_new_tokens after clipping > 0
  and allocate_slots(...) succeeds

进入 running 后：
  request = request_queue.pop_request()
  self.running.append(request)
  request.status = RUNNING
  request.num_computed_tokens = num_computed_tokens
  num_scheduled_tokens[request_id] = num_new_tokens
  token_budget -= num_new_tokens
```

---

## 41. 和前几个问题的关系

`01_request_states.md` 解释了：

```text
waiting / running / skipped_waiting 分别是什么。
```

`02_token_budget.md` 解释了：

```text
一轮 schedule 总共有多少 token budget，以及 running / waiting 如何共同消耗它。
```

`03_running_decode_prefill.md` 解释了：

```text
已经在 running 里的请求如何继续 decode / prefill。
```

本篇解释的是：

```text
还没进入 running 的 waiting / skipped_waiting 请求，
如何被 Scheduler 接纳进入 running，
以及哪些条件会让它们本轮继续等待或被跳过。
```

接下来 `05_prefix_and_external_kv_hits.md` 可以继续深入：

```text
waiting 请求进入 running 前，
本地 prefix cache 和外部 KV cache 到底如何计算命中 token。
```
