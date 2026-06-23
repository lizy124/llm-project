# Scheduler 如何回答：当前有哪些请求在等待、运行、阻塞？

源码位置：`D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`

这个问题的答案主要来自 `Scheduler` 内部维护的几个容器和 `Request.status` 状态。

一句话概括：

```text
Scheduler 用 self.running 表示正在运行的请求；
用 self.waiting 表示正常等待调度的请求；
用 self.skipped_waiting 表示暂时阻塞/跳过的等待请求；
用 self.requests 保存所有还没彻底释放的请求全集。
```

---

## 1. 最核心的四个容器

### 1.1 `self.requests`：请求全集

初始化位置：

```python
self.requests: dict[str, Request] = {}
```

源码：`code/vllm/vllm/v1/core/sched/scheduler.py:171`

它保存 Scheduler 当前知道的所有请求。

这里的“所有请求”不是指所有正在调度的请求，而是指所有还没有彻底释放的请求，包括：

1. 正在 waiting 的请求；
2. 正在 running 的请求；
3. 暂时 blocked / skipped 的请求；
4. 已经 finished 但因为 KV Connector 异步发送/接收还没完成，所以暂时不能删除的请求。

因此，`self.requests` 是全局索引表，不直接等价于“活跃请求队列”。

典型用途：

```python
request = self.requests.get(req_id)
```

比如 `update_from_output()` 里会根据 req_id 找回请求对象。

源码：`scheduler.py:1531`

---

### 1.2 `self.running`：正在运行的请求

初始化位置：

```python
self.running: list[Request] = []
```

源码：`scheduler.py:183`

`self.running` 表示已经被 Scheduler 接纳进入模型执行流的请求。

这些请求通常已经：

1. 分配过 KV Cache block；
2. 进入过某一轮 `SchedulerOutput`；
3. 处于 `RequestStatus.RUNNING`；
4. 后续 decode 或 chunked prefill 时会优先被继续调度。

`schedule()` 每轮首先调度 running 请求：

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
```

源码：`scheduler.py:431`

所以 Scheduler 回答“当前有哪些请求在运行”时，最直接依据就是：

```text
self.running
```

并且这些 request 的状态通常是：

```python
RequestStatus.RUNNING
```

---

### 1.3 `self.waiting`：正常等待调度的请求

初始化位置：

```python
self.waiting = create_request_queue(self.policy)
```

源码：`scheduler.py:180`

`self.waiting` 保存可以被正常尝试调度的请求。

这些请求一般满足：

1. 还没有进入 running；
2. 没有被阻塞在外部资源上；
3. 可以在下一轮 `schedule()` 中尝试分配 token 和 KV block。

新请求进入 Scheduler 时，如果不是阻塞状态，就会加入 `self.waiting`：

```python
self._enqueue_waiting_request(request)
self.requests[request.request_id] = request
```

源码：`scheduler.py:1976`

`_enqueue_waiting_request()` 的逻辑是：

```python
if self._is_blocked_waiting_status(request.status):
    self.skipped_waiting.add_request(request)
else:
    self.waiting.add_request(request)
```

源码：`scheduler.py:1812`

也就是说，正常 waiting 请求进入 `self.waiting`，阻塞型 waiting 请求进入 `self.skipped_waiting`。

---

### 1.4 `self.skipped_waiting`：被跳过或阻塞的等待请求

初始化位置：

```python
self.skipped_waiting = create_request_queue(self.policy)
```

源码：`scheduler.py:182`

`self.skipped_waiting` 是理解“阻塞请求”的关键。

它保存的不是 finished 请求，而是“目前还不能正常调度，但后续可能恢复”的请求。

常见来源包括：

1. 等远端 KV cache 加载完成；
2. 等结构化输出 grammar 准备好；
3. 等 streaming input 的下一段输入；
4. 因 LoRA 数量限制被本轮跳过；
5. 因 KV Connector 暂时无法判断命中而被本轮跳过；
6. 因 encoder cache / ECConnector 等条件不满足被跳过。

`schedule()` 中如果遇到暂时不能调度的 waiting 请求，会把它放到本轮临时队列：

```python
step_skipped_waiting.prepend_request(request)
```

最后再合并回：

```python
self.skipped_waiting.prepend_requests(step_skipped_waiting)
```

源码：`scheduler.py:980`

---

## 2. Scheduler 如何区分“等待”和“阻塞”

Scheduler 不是只靠队列名字区分，还会看 `Request.status`。

### 2.1 阻塞型 waiting 状态

阻塞状态判断函数：

```python
@staticmethod
def _is_blocked_waiting_status(status: RequestStatus) -> bool:
    return status in (
        RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_STREAMING_REQ,
    )
```

源码：`scheduler.py:1804`

这三个状态分别表示：

| 状态 | 含义 | 请求在哪里 |
|---|---|---|
| `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` | 等 grammar 构建完成 | `skipped_waiting` |
| `WAITING_FOR_REMOTE_KVS` | 等外部 KV cache 加载完成 | `skipped_waiting` |
| `WAITING_FOR_STREAMING_REQ` | 等 streaming input 的下一段输入 | `skipped_waiting` |

所以，“阻塞请求”在 Scheduler 里通常就是：

```text
在 self.skipped_waiting 中，并且 status 属于 _is_blocked_waiting_status() 的请求
```

不过要注意：`self.skipped_waiting` 里不一定全是长期阻塞状态，也可能有本轮因为 LoRA / KV 查询 / budget 等原因临时跳过的请求。

---

## 3. 新请求进入时，Scheduler 怎么分类？

入口方法：

```python
def add_request(self, request: Request) -> None:
```

源码：`scheduler.py:1959`

### 3.1 全新请求

如果 request id 不存在：

```python
self._enqueue_waiting_request(request)
self.requests[request.request_id] = request
```

源码：`scheduler.py:1976`

分类逻辑在 `_enqueue_waiting_request()`：

```python
if self._is_blocked_waiting_status(request.status):
    self.skipped_waiting.add_request(request)
else:
    self.waiting.add_request(request)
```

源码：`scheduler.py:1812`

因此：

```text
普通 WAITING 请求 → self.waiting
阻塞型 WAITING 请求 → self.skipped_waiting
所有请求 → self.requests
```

### 3.2 streaming input 的重复 request id

如果 request id 已经存在，说明可能是 streaming input 的后续 chunk：

```python
existing = self.requests.get(request.request_id)
```

源码：`scheduler.py:1960`

这时不会创建新请求，而是更新已有 session。

如果已有请求正在等 streaming input：

```python
elif update is not None:
    self._update_request_as_session(existing, update)
```

源码：`scheduler.py:1967`

`_update_request_as_session()` 会把状态改回：

```python
session.status = RequestStatus.WAITING
```

源码：`scheduler.py:1214`

这样它就可以重新进入正常调度流程。

---

## 4. `schedule()` 如何统计和处理 running 请求

每轮调度时，Scheduler 首先扫描 `self.running`。

入口：

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
```

源码：`scheduler.py:431`

running 请求可能出现以下情况。

### 4.1 当前不能 decode，跳过

Pipeline Parallel + async 场景中，同一个请求不能每一步都 decode，需要等 cadence：

```python
if self.current_step < request.next_decode_eligible_step:
    req_index += 1
    continue
```

源码：`scheduler.py:450`

这类请求仍然在 `self.running` 里，只是本轮不调度。

### 4.2 DP prefill balancing 下跳过 prefill chunk

```python
if defer_prefills and request.is_prefill_chunk:
    req_index += 1
    continue
```

源码：`scheduler.py:456`

这类请求也是 running，只是本轮暂不推进 prefill。

### 4.3 没有可调度 token，跳过

如果 `num_new_tokens == 0`：

```python
req_index += 1
continue
```

源码：`scheduler.py:503`

常见原因：

1. PP 场景下 prompt 已经发出但还没返回；
2. async scheduling 中已经达到 max tokens；
3. encoder budget 不足；
4. hybrid Mamba 对齐后本轮无法形成合法 chunk。

这些请求仍然算 running。

### 4.4 KV block 不够，可能抢占 running 请求

如果 running 请求申请新 block 失败：

```python
new_blocks = self.kv_cache_manager.allocate_slots(...)
if new_blocks is None:
    ... preempt ...
```

源码：`scheduler.py:524`

被抢占请求会从 `self.running` 移除，然后进入 waiting：

```python
self._preempt_request(preempted_req, scheduled_timestamp)
```

源码：`scheduler.py:563`

`_preempt_request()` 会执行：

```python
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
self.waiting.prepend_request(request)
```

源码：`scheduler.py:1117`

所以抢占会让请求从：

```text
running → waiting，状态 RUNNING → PREEMPTED
```

---

## 5. `schedule()` 如何统计和处理 waiting / skipped_waiting 请求

调度 waiting 请求的条件：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

源码：`scheduler.py:625`

也就是说，如果本轮发生了抢占，Scheduler 不再接纳新的 waiting 请求。

### 5.1 选择 waiting 还是 skipped_waiting

```python
request_queue = self._select_waiting_queue_for_scheduling()
```

源码：`scheduler.py:632`

选择逻辑：

```python
if self.policy == SchedulingPolicy.FCFS:
    return self.skipped_waiting or self.waiting or None
```

源码：`scheduler.py:1818`

FCFS 下优先看 `skipped_waiting`，避免之前被跳过的请求长期饥饿。

PRIORITY 下，如果两个队列都有请求，会比较队头：

```python
return self.waiting if waiting_req < skipped_req else self.skipped_waiting
```

源码：`scheduler.py:1822`

### 5.2 阻塞请求先尝试恢复

取到队头请求后，如果它是阻塞状态：

```python
if self._is_blocked_waiting_status(request.status) and not self._try_promote_blocked_waiting_request(request):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

源码：`scheduler.py:638`

这说明：

```text
阻塞请求不会直接丢弃，而是每轮尝试恢复；恢复失败就继续放回 skipped_waiting。
```

---

## 6. 三类阻塞状态怎么恢复

恢复逻辑集中在：

```python
def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
```

源码：`scheduler.py:2384`

### 6.1 `WAITING_FOR_REMOTE_KVS`

这是外部 KV cache 异步加载中的状态。

判断是否完成：

```python
if request.request_id not in self.finished_recving_kv_req_ids:
    return False
```

源码：`scheduler.py:2391`

如果 Worker 还没报告 KV load 完成，就不能恢复。

完成后调用：

```python
self._update_waiting_for_remote_kv(request)
```

源码：`scheduler.py:2394`

然后根据是否被抢占过恢复状态：

```python
if request.num_preemptions:
    request.status = RequestStatus.PREEMPTED
else:
    request.status = RequestStatus.WAITING
```

源码：`scheduler.py:2395`

即：

```text
WAITING_FOR_REMOTE_KVS → WAITING / PREEMPTED
```

### 6.2 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`

结构化输出请求需要 grammar 准备好。

如果 grammar 还没有：

```python
if not (structured_output_req and structured_output_req.grammar):
    return False
```

源码：`scheduler.py:2402`

如果 grammar 已经存在：

```python
request.status = RequestStatus.WAITING
return True
```

源码：`scheduler.py:2405`

即：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING
```

### 6.3 `WAITING_FOR_STREAMING_REQ`

这个状态表示请求正在等用户继续输入。

```python
if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
    assert not request.streaming_queue
    return False
```

源码：`scheduler.py:2408`

它不会在普通 schedule 循环里自动恢复，只能等 `add_request()` 收到同 request id 的新 streaming chunk，然后 `_update_request_as_session()` 把它改回 `WAITING`。

即：

```text
WAITING_FOR_STREAMING_REQ --收到新输入--> WAITING
```

---

## 7. waiting 请求怎样变成 running

waiting 请求成功调度后，会从队列中弹出：

```python
request = request_queue.pop_request()
```

源码：`scheduler.py:916`

如果不是异步 KV load，就加入 running：

```python
self.running.append(request)
```

源码：`scheduler.py:939`

然后设置状态：

```python
request.status = RequestStatus.RUNNING
```

源码：`scheduler.py:958`

所以正常迁移是：

```text
waiting / skipped_waiting
  → schedule 成功
  → self.running
  → RequestStatus.RUNNING
```

---

## 8. waiting 请求怎样变成阻塞

### 8.1 远端 KV 异步加载导致阻塞

如果 KV Connector 命中外部 KV cache，并选择 async load：

```python
if load_kv_async:
    num_new_tokens = 0
```

源码：`scheduler.py:781`

分配 block 后不会加入 running，而是：

```python
request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
step_skipped_waiting.prepend_request(request)
request.num_computed_tokens = num_computed_tokens
self._inflight_prefills.add(request)
continue
```

源码：`scheduler.py:917`

迁移关系：

```text
WAITING
  → 发现外部 KV async load
  → WAITING_FOR_REMOTE_KVS
  → skipped_waiting
```

### 8.2 结构化输出 grammar 未就绪

结构化输出 grammar 未完成时，请求状态可能是：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

进入 Scheduler 时 `_enqueue_waiting_request()` 会把它放入 `skipped_waiting`。

源码：`scheduler.py:1812`

### 8.3 streaming input 等下一段输入

当一个 resumable request 当前 chunk 结束，但后续 streaming input 还没到：

```python
request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
self.num_waiting_for_streaming_input += 1
self._enqueue_waiting_request(request)
```

源码：`scheduler.py:1842`

于是它进入：

```text
WAITING_FOR_STREAMING_REQ → skipped_waiting
```

---

## 9. `skipped_waiting` 不完全等于“阻塞”

需要特别注意：`self.skipped_waiting` 是“本轮不能正常调度”的集合，但里面不一定全是三种 blocked status。

例如以下情况也会进入 `skipped_waiting`。

### 9.1 LoRA 数量限制

如果本轮调度的 LoRA 种类已经达到上限：

```python
request_queue.pop_request()
step_skipped_waiting.prepend_request(request)
continue
```

源码：`scheduler.py:653`

这类请求可能状态仍是 `WAITING`，只是本轮因为 LoRA 限制跳过。

### 9.2 KV Connector 暂时无法判断命中

如果 connector 返回 `ext_tokens is None`：

```python
request_queue.pop_request()
step_skipped_waiting.prepend_request(request)
continue
```

源码：`scheduler.py:729`

这类请求也可能仍是 `WAITING`。

### 9.3 ECConnector / encoder cache 暂不可用

如果多模态 encoder cache 暂不可用：

```python
request_queue.pop_request()
step_skipped_waiting.prepend_request(request)
continue
```

源码：`scheduler.py:750`

所以严格来说：

```text
blocked requests = skipped_waiting 中 status 属于阻塞状态的请求
temporarily skipped requests = skipped_waiting 中 status 仍是 WAITING/PREEMPTED 但本轮条件不满足的请求
```

---

## 10. 请求结束后如何从 running / waiting 中移除

### 10.1 正常生成结束

`update_from_output()` 检查到请求 stop 后：

```python
finished = self._handle_stopped_request(request)
if finished:
    kv_transfer_params = self._free_request(request)
```

源码：`scheduler.py:1656`

然后从 running 删除：

```python
self.running = remove_all(self.running, stopped_running_reqs)
```

源码：`scheduler.py:1711`

### 10.2 外部取消 / abort

外部调用：

```python
finish_requests(request_ids, finished_status)
```

源码：`scheduler.py:1983`

会先收集 running 和 waiting 中要删除的请求：

```python
running_requests_to_remove.add(request)
waiting_requests_to_remove.append(request)
```

源码：`scheduler.py:2016`

然后批量删除：

```python
self.running = remove_all(self.running, running_requests_to_remove)
self.waiting.remove_requests(waiting_requests_to_remove)
self.skipped_waiting.remove_requests(waiting_requests_to_remove)
```

源码：`scheduler.py:2024`

---

## 11. Scheduler 如何给外部报告数量

### 11.1 running / waiting 数量

```python
def get_request_counts(self) -> tuple[int, int]:
    return len(self.running), len(self.waiting) + len(self.skipped_waiting)
```

源码：`scheduler.py:1955`

这里第二个值把 `waiting` 和 `skipped_waiting` 合并统计。

所以它回答的是：

```text
running 数量 = len(self.running)
waiting 总数量 = len(self.waiting) + len(self.skipped_waiting)
```

这里的 waiting 总数量包含阻塞请求。

### 11.2 unfinished 请求数量

```python
def get_num_unfinished_requests(self) -> int:
```

源码：`scheduler.py:2106`

逻辑：

```python
if PAUSED_ALL:
    return 0
if PAUSED_NEW:
    return len(self.running)
num_waiting = len(self.waiting) + len(self.skipped_waiting) - self.num_waiting_for_streaming_input
return num_waiting + len(self.running)
```

源码：`scheduler.py:2106`

注意：`WAITING_FOR_STREAMING_REQ` 的请求会被减掉，因为它在等待用户新输入，不应该算作当前可推进的 unfinished work。

---

## 12. 状态迁移总图

### 12.1 普通请求

```text
add_request
  → self.waiting / RequestStatus.WAITING
  → schedule 成功
  → self.running / RequestStatus.RUNNING
  → update_from_output 生成 token
  → 未结束：继续 self.running
  → 结束：_free_request → 从 self.running 和 self.requests 移除
```

### 12.2 被抢占请求

```text
self.running / RUNNING
  → KV block 不够
  → _preempt_request
  → self.waiting / PREEMPTED
  → 后续 schedule 恢复
  → self.running / RUNNING
```

### 12.3 远端 KV 异步加载请求

```text
self.waiting / WAITING
  → connector 命中外部 KV 且 async load
  → self.skipped_waiting / WAITING_FOR_REMOTE_KVS
  → Worker 报告 finished_recving
  → _try_promote_blocked_waiting_request
  → self.waiting 语义 / WAITING 或 PREEMPTED
  → schedule 成功
  → self.running / RUNNING
```

### 12.4 结构化输出 grammar 阻塞

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
  → self.skipped_waiting
  → grammar ready
  → WAITING
  → schedule 成功
  → RUNNING
```

### 12.5 streaming input 阻塞

```text
RUNNING
  → 当前 chunk 结束但请求 resumable
  → WAITING_FOR_STREAMING_REQ
  → self.skipped_waiting
  → add_request 收到同 request_id 新 chunk
  → _update_request_as_session
  → WAITING
  → 后续继续调度
```

---

## 13. 从“回答问题”的角度总结

如果要问 Scheduler：

```text
当前有哪些请求在等待、运行、阻塞？
```

它内部的答案是：

### 13.1 运行中的请求

```python
self.running
```

通常状态：

```python
RequestStatus.RUNNING
```

### 13.2 正常等待的请求

```python
self.waiting
```

通常状态：

```python
RequestStatus.WAITING 或 RequestStatus.PREEMPTED
```

### 13.3 阻塞或暂时跳过的请求

```python
self.skipped_waiting
```

其中真正阻塞的状态是：

```python
RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
RequestStatus.WAITING_FOR_REMOTE_KVS
RequestStatus.WAITING_FOR_STREAMING_REQ
```

### 13.4 所有还没释放的请求

```python
self.requests
```

这个全集可能包含已经 finished 但等待 KV Connector 异步清理的请求。

---

## 14. 最关键的判断公式

```text
running 请求：
  len(self.running)

waiting 请求：
  len(self.waiting)

skipped / blocked waiting 请求：
  len(self.skipped_waiting)

外部统计 waiting 总数：
  len(self.waiting) + len(self.skipped_waiting)

真正 blocked 请求：
  [r for r in self.skipped_waiting if _is_blocked_waiting_status(r.status)]

所有 scheduler 仍持有的请求：
  self.requests
```

---

## 15. 一个具体例子

假设当前 Scheduler 状态如下：

```text
self.running:
  req-a: RUNNING
  req-b: RUNNING

self.waiting:
  req-c: WAITING
  req-d: PREEMPTED

self.skipped_waiting:
  req-e: WAITING_FOR_REMOTE_KVS
  req-f: WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
  req-g: WAITING

self.requests:
  req-a, req-b, req-c, req-d, req-e, req-f, req-g, req-h
```

那么 Scheduler 的理解是：

```text
正在运行：req-a, req-b
正常等待：req-c, req-d
阻塞等待：req-e, req-f
临时跳过但不一定阻塞：req-g
仍被 Scheduler 持有但可能已 finished/待清理：req-h
```

`get_request_counts()` 会返回：

```text
running = 2
waiting = 5  # req-c/d/e/f/g
```

它不会单独区分 blocked，需要结合 `skipped_waiting` 和 `Request.status` 再判断。
