# 01. 当前有哪些请求在等待、运行、阻塞？

源码位置：`D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`

本问题关注：Scheduler 如何知道当前系统里有哪些请求、哪些请求正在运行、哪些请求还在等待、哪些请求暂时阻塞，以及这些状态如何迁移。

---

## 1. 一句话回答

Scheduler 不是通过一个单独字段回答这个问题，而是通过“四类容器 + Request.status”共同回答：

```text
self.requests         保存 Scheduler 仍然持有的请求全集
self.running          保存已经进入执行流的请求
self.waiting          保存正常等待调度的请求
self.skipped_waiting  保存暂时跳过或阻塞的等待请求
Request.status        标识请求当前语义状态
```

所以：

```text
正在运行的请求 = self.running
正常等待的请求 = self.waiting
阻塞/跳过的请求 = self.skipped_waiting
所有未彻底释放的请求 = self.requests
```

但要特别注意：`self.skipped_waiting` 不完全等于“阻塞请求”，它还包含一些只是本轮因为调度约束被临时跳过的请求。

---

## 2. Scheduler 初始化时建立的请求容器

在 `Scheduler.__init__()` 中，请求相关状态主要初始化在以下位置。

### 2.1 `self.requests`：请求全集索引

```python
self.requests: dict[str, Request] = {}
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:171`

`self.requests` 是 `request_id -> Request` 的字典。

它的作用不是表示“正在调度的请求”，而是表示：

```text
Scheduler 当前还需要记住的所有请求
```

它包括：

1. 正常 waiting 的请求；
2. skipped / blocked waiting 的请求；
3. running 请求；
4. 已经 finished 但还不能完全删除的请求。

第 4 种容易被忽略：如果 KV Connector 还在异步发送 KV，或者请求结束但 block 释放被延迟，那么请求可能已经不在 `running` / `waiting` 队列里，但仍保留在 `self.requests` 中，直到 connector 通知可以释放。

因此：

```text
self.requests 是全集索引，不是调度队列。
```

---

### 2.2 `self.waiting`：正常等待队列

```python
self.waiting = create_request_queue(self.policy)
```

位置：`scheduler.py:180`

`self.waiting` 保存可以被正常尝试调度的请求。

典型状态包括：

```python
RequestStatus.WAITING
RequestStatus.PREEMPTED
```

含义：

- `WAITING`：新请求或恢复后的请求，等待进入 running；
- `PREEMPTED`：之前运行过，但因为 KV Cache block 不够被抢占，现在回到 waiting 等待重新调度。

---

### 2.3 `self.skipped_waiting`：阻塞或被临时跳过的等待队列

```python
self.skipped_waiting = create_request_queue(self.policy)
```

位置：`scheduler.py:182`

`skipped_waiting` 的名字很准确：它保存的是“当前还不能正常调度，需要跳过一下”的请求。

它包含两类请求。

第一类：真正阻塞的请求。例如：

```python
RequestStatus.WAITING_FOR_REMOTE_KVS
RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
RequestStatus.WAITING_FOR_STREAMING_REQ
```

第二类：不是语义阻塞，只是本轮受约束暂时跳过的请求。例如：

- LoRA 种类超过本轮上限；
- KV Connector 暂时返回 `None`，无法确定外部 KV 命中；
- ECConnector / encoder cache 暂时不可用；
- 某些调度预算导致本轮不能安排它。

所以：

```text
skipped_waiting = blocked requests + temporarily skipped waiting requests
```

---

### 2.4 `self.running`：运行中请求列表

```python
self.running: list[Request] = []
```

位置：`scheduler.py:183`

`running` 中的请求已经进入模型执行流。

通常它们已经：

1. 被 Scheduler 从 waiting 队列取出；
2. 分配过 KV Cache block；
3. 加入过某一轮 `SchedulerOutput`；
4. 状态设置为 `RequestStatus.RUNNING`。

每轮 `schedule()` 会先扫描 `self.running`，再考虑 waiting 请求。

位置：`scheduler.py:429` 到 `scheduler.py:614`

---

## 3. Request.status 如何参与判断

队列告诉 Scheduler “请求在哪里”，`Request.status` 告诉 Scheduler “请求为什么在这里”。

阻塞状态判断函数是：

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

这三个状态的含义如下。

| 状态 | 含义 | 一般所在队列 | 如何恢复 |
|---|---|---|---|
| `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` | 等结构化输出 grammar 准备完成 | `skipped_waiting` | grammar 构造好后变回 `WAITING` |
| `WAITING_FOR_REMOTE_KVS` | 等远端 KV Cache 加载完成 | `skipped_waiting` | Worker 报告 finished_recving 后恢复 |
| `WAITING_FOR_STREAMING_REQ` | 等 streaming input 下一段输入 | `skipped_waiting` | `add_request()` 收到同 request id 新输入后恢复 |

因此，严格判断“阻塞请求”时应该看：

```text
request in self.skipped_waiting
and _is_blocked_waiting_status(request.status)
```

而不能简单把 `skipped_waiting` 全部当成阻塞。

---

## 4. 新请求进入 Scheduler 时如何分类

入口：

```python
def add_request(self, request: Request) -> None:
```

位置：`scheduler.py:1959`

### 4.1 全新 request_id

如果 `request_id` 不存在：

```python
self._enqueue_waiting_request(request)
self.requests[request.request_id] = request
```

位置：`scheduler.py:1976`

`_enqueue_waiting_request()` 决定进入 `waiting` 还是 `skipped_waiting`：

```python
def _enqueue_waiting_request(self, request: Request) -> None:
    if self._is_blocked_waiting_status(request.status):
        self.skipped_waiting.add_request(request)
    else:
        self.waiting.add_request(request)
```

位置：`scheduler.py:1812`

也就是说：

```text
普通请求 → self.waiting
阻塞型请求 → self.skipped_waiting
所有请求 → self.requests
```

### 4.2 已存在 request_id：streaming input 场景

如果同一个 request id 再次进入：

```python
existing = self.requests.get(request.request_id)
```

位置：`scheduler.py:1960`

这一般表示 streaming input 的后续 chunk。

如果已有请求正在等输入，则：

```python
self._update_request_as_session(existing, update)
```

位置：`scheduler.py:1968`

`_update_request_as_session()` 最后会设置：

```python
session.status = RequestStatus.WAITING
```

位置：`scheduler.py:1214`

这表示请求从：

```text
WAITING_FOR_STREAMING_REQ → WAITING
```

然后后续会重新参与调度。

---

## 5. `schedule()` 每轮如何处理 running 请求

每轮调度先处理 running。

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
```

位置：`scheduler.py:431`

这说明：

```text
只要请求在 self.running 里，它就是 Scheduler 认为的运行中请求。
```

但 running 不代表本轮一定会执行。running 请求可能因为以下原因被跳过。

### 5.1 已经达到可确定的最大输出，不再调度

```python
if request.num_output_placeholders > 0 and ... >= request.num_prompt_tokens + request.max_tokens:
    req_index += 1
    continue
```

位置：`scheduler.py:434`

这是 async scheduling 下避免多调度一步。

### 5.2 Pipeline Parallel / async cadence 限制

```python
if self.current_step < request.next_decode_eligible_step:
    req_index += 1
    continue
```

位置：`scheduler.py:450`

表示这个 running 请求还没到下一次 decode 的节奏。

### 5.3 DP prefill balancing 暂缓 prefill chunk

```python
if defer_prefills and request.is_prefill_chunk:
    req_index += 1
    continue
```

位置：`scheduler.py:456`

这时请求仍在 `running`，只是当前 step 不推进。

### 5.4 本轮没有可调度 token

```python
if num_new_tokens == 0:
    req_index += 1
    continue
```

位置：`scheduler.py:503`

可能原因包括 encoder budget 不足、Mamba block 对齐后不能形成合法 chunk 等。

---

## 6. running 请求如何变成 waiting：抢占

当 running 请求申请 KV block 失败时，Scheduler 会抢占某个 running 请求。

申请 block：

```python
new_blocks = self.kv_cache_manager.allocate_slots(...)
```

位置：`scheduler.py:524`

失败后选择被抢占请求。

PRIORITY 策略：

```python
preempted_req = max(
    self.running,
    key=lambda r: (r.priority, r.arrival_time),
)
```

位置：`scheduler.py:536`

非 PRIORITY 策略：

```python
preempted_req = self.running.pop()
```

位置：`scheduler.py:561`

抢占处理：

```python
self._preempt_request(preempted_req, scheduled_timestamp)
```

位置：`scheduler.py:563`

`_preempt_request()` 做的事情：

```python
self._free_request_blocks(request)
self.encoder_cache_manager.free(request)
self._inflight_prefills.discard(request)
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
request.num_preemptions += 1
self.waiting.prepend_request(request)
```

位置：`scheduler.py:1114` 到 `scheduler.py:1126`

状态迁移：

```text
self.running / RUNNING
  → block 不够，被抢占
  → self.waiting / PREEMPTED
```

注意：抢占会重置 `num_computed_tokens = 0`。后续重新调度时，会重新走 prefix cache / external KV cache 查询，尽可能复用已缓存的 KV。

---

## 7. waiting / skipped_waiting 的选择逻辑

调度 waiting 请求前，Scheduler 会选择一个队列：

```python
request_queue = self._select_waiting_queue_for_scheduling()
```

位置：`scheduler.py:632`

选择函数：

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

解释：

- FCFS：优先处理 `skipped_waiting`，避免之前被跳过的请求长期饥饿；
- PRIORITY：比较两个队列头部请求的优先级；
- 只有一个队列非空时，就选那个。

---

## 8. 阻塞请求如何尝试恢复

在 waiting 调度循环中，如果队头请求是阻塞状态，会先尝试恢复：

```python
if self._is_blocked_waiting_status(request.status) and not self._try_promote_blocked_waiting_request(request):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

位置：`scheduler.py:638`

这段逻辑说明：

```text
阻塞请求每轮都会被检查；如果恢复条件不满足，就继续放回 skipped_waiting。
```

恢复逻辑集中在：

```python
def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
```

位置：`scheduler.py:2384`

### 8.1 远端 KV load 阻塞恢复

```python
if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    if request.request_id not in self.finished_recving_kv_req_ids:
        return False
    self._update_waiting_for_remote_kv(request)
    if request.num_preemptions:
        request.status = RequestStatus.PREEMPTED
    else:
        request.status = RequestStatus.WAITING
    return True
```

位置：`scheduler.py:2388`

含义：

1. Worker 还没报告 finished_recving：继续阻塞；
2. Worker 报告完成：把远端 KV 标记为可用；
3. 如果请求之前被抢占过，恢复为 `PREEMPTED`；否则恢复为 `WAITING`。

状态迁移：

```text
WAITING_FOR_REMOTE_KVS → WAITING / PREEMPTED
```

### 8.2 结构化输出 grammar 阻塞恢复

```python
if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
    structured_output_req = request.structured_output_request
    if not (structured_output_req and structured_output_req.grammar):
        return False
    request.status = RequestStatus.WAITING
    return True
```

位置：`scheduler.py:2401`

状态迁移：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING
```

### 8.3 streaming input 阻塞不会在 schedule 中自动恢复

```python
if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
    assert not request.streaming_queue
    return False
```

位置：`scheduler.py:2408`

它需要通过 `add_request()` 收到新的 streaming input chunk 后恢复。

状态迁移：

```text
WAITING_FOR_STREAMING_REQ --add_request 同 request_id 新输入--> WAITING
```

---

## 9. waiting 请求如何进入 running

waiting 请求被成功调度后，会从原队列弹出：

```python
request = request_queue.pop_request()
```

位置：`scheduler.py:916`

如果不是异步 KV load，则加入 running：

```python
self.running.append(request)
```

位置：`scheduler.py:939`

然后根据原状态记录为新请求或恢复请求：

```python
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request)
```

位置：`scheduler.py:944`

最后设置为 running：

```python
request.status = RequestStatus.RUNNING
```

位置：`scheduler.py:958`

状态迁移：

```text
self.waiting / WAITING
  → schedule 成功
  → self.running / RUNNING

self.waiting / PREEMPTED
  → schedule 成功
  → self.running / RUNNING
```

---

## 10. waiting 请求如何进入阻塞队列

### 10.1 外部 KV 异步加载

如果 KV Connector 表示外部 KV 命中，并且可以异步 load：

```python
if load_kv_async:
    num_new_tokens = 0
```

位置：`scheduler.py:781`

分配 block 后，不加入 running，而是：

```python
request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
step_skipped_waiting.prepend_request(request)
request.num_computed_tokens = num_computed_tokens
self._inflight_prefills.add(request)
continue
```

位置：`scheduler.py:917` 到 `scheduler.py:937`

状态迁移：

```text
WAITING
  → 远端 KV async load
  → WAITING_FOR_REMOTE_KVS
  → skipped_waiting
```

### 10.2 streaming input 等下一段输入

当请求是 resumable，当前输入 chunk 结束但后续输入还没到：

```python
request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
self.num_waiting_for_streaming_input += 1
self._enqueue_waiting_request(request)
```

位置：`scheduler.py:1842`

状态迁移：

```text
RUNNING
  → 当前 chunk 结束
  → WAITING_FOR_STREAMING_REQ
  → skipped_waiting
```

### 10.3 结构化输出 grammar 等待

结构化输出 grammar 未准备好时，请求状态会是：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

通过 `_enqueue_waiting_request()` 进入 `skipped_waiting`。

位置：`scheduler.py:1812`

---

## 11. 请求结束后如何从队列中移除

### 11.1 正常生成结束

在 `update_from_output()` 中，如果请求 stop：

```python
finished = self._handle_stopped_request(request)
if finished:
    kv_transfer_params = self._free_request(request)
```

位置：`scheduler.py:1656`

之后从 running 中批量删除：

```python
if stopped_running_reqs:
    self.running = remove_all(self.running, stopped_running_reqs)
```

位置：`scheduler.py:1711`

如果是 preempted/waiting 中停止，也会从 waiting 删除：

```python
self.waiting.remove_requests(stopped_preempted_reqs)
```

位置：`scheduler.py:1713`

### 11.2 外部 abort

外部取消请求时调用：

```python
def finish_requests(self, request_ids, finished_status)
```

位置：`scheduler.py:1983`

它会分别从 running 和 waiting/skipped_waiting 中移除：

```python
self.running = remove_all(self.running, running_requests_to_remove)
self.waiting.remove_requests(waiting_requests_to_remove)
self.skipped_waiting.remove_requests(waiting_requests_to_remove)
```

位置：`scheduler.py:2024`

然后设置 finished 状态并调用 `_free_request()`。

---

## 12. 为什么 finished 请求可能还在 self.requests 里

`_free_request()` 的关键逻辑：

```python
connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
...
delay_free_blocks |= connector_delay_free_blocks
if not delay_free_blocks:
    self._free_blocks(request)
```

位置：`scheduler.py:2051`

如果 connector 需要异步发送 KV，`delay_free_blocks=True`，则不会立即调用 `_free_blocks()`。

`_free_blocks()` 才会真正删除：

```python
del self.requests[request.request_id]
```

位置：`scheduler.py:2068`

所以：

```text
finished 但等待 connector 清理的请求：
  不在 running/waiting/skipped_waiting 中
  但还在 self.requests 中
```

这解释了为什么 `self.requests` 不能简单等同于 unfinished requests。

---

## 13. 对外统计接口如何回答数量

### 13.1 `get_request_counts()`

```python
def get_request_counts(self) -> tuple[int, int]:
    return len(self.running), len(self.waiting) + len(self.skipped_waiting)
```

位置：`scheduler.py:1955`

返回：

```text
running 数量 = len(self.running)
waiting 数量 = len(self.waiting) + len(self.skipped_waiting)
```

注意：这里的 waiting 数量包含阻塞/跳过请求。

### 13.2 `get_num_unfinished_requests()`

```python
def get_num_unfinished_requests(self) -> int:
```

位置：`scheduler.py:2106`

核心逻辑：

```python
if self._pause_state == PauseState.PAUSED_ALL:
    return 0
if self._pause_state == PauseState.PAUSED_NEW:
    return len(self.running)
num_waiting = (
    len(self.waiting)
    + len(self.skipped_waiting)
    - self.num_waiting_for_streaming_input
)
return num_waiting + len(self.running)
```

位置：`scheduler.py:2106` 到 `scheduler.py:2116`

这里有一个重要细节：

```text
WAITING_FOR_STREAMING_REQ 的请求会被减掉。
```

因为这类请求在等用户继续输入，不是 Scheduler 当前能够推进的计算任务。

---

## 14. 容易疑惑的点

### 14.1 `skipped_waiting` 是不是都阻塞？

不是。

真正阻塞要看 status 是否属于：

```python
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
WAITING_FOR_REMOTE_KVS
WAITING_FOR_STREAMING_REQ
```

`skipped_waiting` 还可能包含普通 `WAITING` 或 `PREEMPTED`，只是本轮因为 LoRA、KV 查询、encoder cache 等约束被跳过。

### 14.2 `running` 里的请求本轮一定会执行吗？

不一定。

它可能因为：

- PP/async cadence；
- prefill balancing；
- encoder budget；
- Mamba block 对齐；
- 已达到 max tokens；

在本轮被跳过，但仍留在 `running`。

### 14.3 `requests` 里的请求一定还没结束吗？

不一定。

如果 KV Connector 还有异步发送/清理，finished 请求也可能暂时留在 `self.requests`。

### 14.4 `waiting` 和 `PREEMPTED` 是什么关系？

`PREEMPTED` 请求通常在 `waiting` 队列中。

它表示这个请求之前运行过，但被抢占后回到 waiting，等待重新调度。

### 14.5 为什么 streaming request 要从 unfinished 计数里减掉？

因为它当前没有可执行的输入，Scheduler 无法推进它。它更像是在等外部用户输入，而不是等 GPU 资源。

---

## 15. 总结

Scheduler 回答“当前有哪些请求在等待、运行、阻塞”依赖以下映射：

```text
运行中：
  self.running

正常等待：
  self.waiting

阻塞或临时跳过：
  self.skipped_waiting

请求全集：
  self.requests

真正阻塞：
  request in self.skipped_waiting
  and request.status in {
      WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
      WAITING_FOR_REMOTE_KVS,
      WAITING_FOR_STREAMING_REQ,
  }
```

请求状态的主要迁移是：

```text
新请求：
  add_request → waiting → running → finished/free

抢占请求：
  running → PREEMPTED → waiting → running

远端 KV 异步加载：
  waiting → WAITING_FOR_REMOTE_KVS → skipped_waiting → waiting/running

结构化输出等待：
  WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → skipped_waiting → waiting

streaming input：
  running → WAITING_FOR_STREAMING_REQ → skipped_waiting → 收到新输入 → waiting
```

因此，Scheduler 的请求状态管理不是简单的一个状态机，而是：

```text
队列位置 + Request.status + connector/streaming/grammar 等外部条件
```

共同决定当前请求属于等待、运行、阻塞还是待清理。
