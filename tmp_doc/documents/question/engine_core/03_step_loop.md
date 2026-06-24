# 03. EngineCore.step() 如何驱动一轮执行？

源码位置：`vllm/vllm/v1/engine/core.py`

本问题关注：一次 EngineCore step 的主流程，以及 EngineCore 如何串起 Scheduler.schedule()、ModelRunner / Worker forward、Scheduler.update_from_output()。

---

## 1. 一句话回答

`EngineCore.step()` 是 EngineCore 最核心的一轮执行函数。

源码注释直接概括了它的职责：

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output.

    Returns tuple of outputs and a flag indicating whether the model
    was executed.
    """
```

位置：`vllm/vllm/v1/engine/core.py:479`

也就是说，一次 `step()` 做三件事：

```text
1. Schedule：调用 Scheduler.schedule()，决定本轮怎么跑；
2. Execute：把 SchedulerOutput 交给 model_executor / Worker / ModelRunner 执行；
3. Make output：调用 Scheduler.update_from_output()，把 Worker 结果变成 EngineCoreOutputs。
```

主流程是：

```text
EngineCore.step()
  → scheduler.has_requests()
  → scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output)
  → scheduler.get_grammar_bitmask(scheduler_output)
  → ModelRunnerOutput
  → _process_aborts_queue()
  → scheduler.update_from_output(scheduler_output, model_output)
  → EngineCoreOutputs
```

一句话：

```text
Scheduler 负责生成执行计划；
Worker / ModelRunner 负责执行计划；
EngineCore.step() 负责把计划、执行、结果回收串成一个闭环。
```

---

## 2. step 主流程图

普通 `step()` 可以画成：

```text
EngineCore.step()
  │
  ├─ 1. 检查 Scheduler 是否还有请求
  │    └─ if not scheduler.has_requests(): return {}, False
  │
  ├─ 2. 调度
  │    └─ scheduler_output = scheduler.schedule(...)
  │
  ├─ 3. 执行模型
  │    └─ future = model_executor.execute_model(scheduler_output, non_block=True)
  │
  ├─ 4. 生成 grammar bitmask
  │    └─ grammar_output = scheduler.get_grammar_bitmask(scheduler_output)
  │
  ├─ 5. 等待 Worker / ModelRunner 输出
  │    └─ model_output = future.result()
  │
  ├─ 6. 如有需要，执行 sample_tokens
  │    └─ model_output = model_executor.sample_tokens(grammar_output)
  │
  ├─ 7. 处理执行期间到达的 abort
  │    └─ _process_aborts_queue()
  │
  ├─ 8. 回收输出并更新请求状态
  │    └─ scheduler.update_from_output(scheduler_output, model_output)
  │
  └─ 9. 返回 EngineCoreOutputs 和 model_executed 标记
```

对应源码主线：

```python
if not self.scheduler.has_requests():
    return {}, False
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
with (...):
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)

return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

位置：`vllm/vllm/v1/engine/core.py:486` 到 `vllm/vllm/v1/engine/core.py:508`

---

## 3. step 入口：先检查 Scheduler 是否还有请求

`step()` 开头先检查：

```python
if not self.scheduler.has_requests():
    return {}, False
```

位置：`vllm/vllm/v1/engine/core.py:486` 到 `vllm/vllm/v1/engine/core.py:489`

这里的含义是：

```text
如果 Scheduler 已经没有任何需要处理的请求，
本轮 EngineCore 不调度、不执行模型，也不返回输出。
```

返回值是：

```text
({}, False)
```

其中：

```text
{}：没有 EngineCoreOutputs；
False：本轮没有执行模型。
```

注意这里检查的是 Scheduler 层是否还有请求，包括：

```text
waiting / skipped_waiting / running / finished 但尚未完全清理的请求等。
```

具体 `has_requests()` 的含义由 Scheduler 实现决定。

---

## 4. schedule 阶段：生成 SchedulerOutput

如果 Scheduler 还有请求，EngineCore 调用：

```python
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
```

位置：`vllm/vllm/v1/engine/core.py:490`

这一步会进入 Scheduler 主调度逻辑。

Scheduler 会决定：

```text
本轮哪些 running 请求继续执行；
哪些 waiting 请求可以进入 running；
每个请求本轮调度多少 token；
需要哪些 KV blocks；
是否发生抢占；
有哪些 encoder input；
有哪些 spec decode tokens；
需要哪些 KV / EC connector metadata。
```

返回的 `SchedulerOutput` 是本轮执行计划。

可以理解为：

```text
SchedulerOutput = 本轮 Worker / ModelRunner 应该如何执行的计划书。
```

### 4.1 `_should_throttle_prefills()` 是什么

`EngineCore.step()` 调用 Scheduler 时传了：

```python
self._should_throttle_prefills()
```

普通 EngineCore 中它默认返回 False：

```python
def _should_throttle_prefills(self) -> bool:
    """Whether to defer new prefills this step (DP prefill balancing).
    Overridden by the DP engine core; never throttles otherwise."""
    return False
```

位置：`vllm/vllm/v1/engine/core.py:474` 到 `vllm/vllm/v1/engine/core.py:477`

也就是说，普通 EngineCore 不主动 throttle prefill。

DP EngineCore 会覆盖它：

```python
def _should_throttle_prefills(self) -> bool:
    return (
        self.prefill_schedule_interval > 1
        and self.step_counter % self.prefill_schedule_interval != 0
    )
```

位置：`vllm/vllm/v1/engine/core.py:1914` 到 `vllm/vllm/v1/engine/core.py:1921`

这用于 DP prefill balancing：

```text
某些 step 延后新 prefill，避免 prefill 打乱 decode cadence。
```

### 4.2 schedule 之后 Scheduler 内部已经做了乐观更新

`Scheduler.schedule()` 内部会构造 SchedulerOutput，并调用 `_update_after_schedule()` 乐观推进请求状态。

所以 EngineCore 拿到 `scheduler_output` 时，Scheduler 内部已经记录：

```text
这些 token 已经被安排出去了；
下一轮不要重复调度同一段 token。
```

后续 `update_from_output()` 再用 Worker 的真实结果修正状态。

---

## 5. forward 执行阶段：execute_model

生成 `SchedulerOutput` 后，EngineCore 调用：

```python
future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

位置：`vllm/vllm/v1/engine/core.py:491`

这一步把本轮执行计划交给 `model_executor`。

`model_executor` 再负责分发给实际 Worker / ModelRunner。

可以理解为：

```text
EngineCore 不直接 forward；
EngineCore 调用 model_executor.execute_model()；
model_executor / Worker / ModelRunner 才真正执行模型。
```

### 5.1 为什么 non_block=True

这里传入：

```python
non_block=True
```

说明 `execute_model()` 可以返回一个 future。

EngineCore 随后可以先做一些轻量工作，例如获取 grammar bitmask：

```python
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
```

位置：`vllm/vllm/v1/engine/core.py:492`

然后再等待：

```python
model_output = future.result()
```

位置：`vllm/vllm/v1/engine/core.py:497`

这个结构允许：

```text
模型执行和部分 CPU 侧准备工作重叠。
```

### 5.2 execute_model 返回什么

`future.result()` 得到的是：

```text
ModelRunnerOutput 或 None
```

源码：

```python
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

位置：`vllm/vllm/v1/engine/core.py:497` 到 `vllm/vllm/v1/engine/core.py:499`

也就是说有两种情况：

```text
execute_model 直接返回 ModelRunnerOutput：
  forward 和 sampling 可能已经一起完成。

execute_model 返回 None：
  需要 EngineCore 再调用 sample_tokens(grammar_output) 完成采样。
```

这个设计让 forward 和 sampling 可以拆开，尤其是 structured output grammar bitmask 需要参与采样时。

---

## 6. grammar bitmask 在 step 里的位置

EngineCore 在等待模型输出前，调用：

```python
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
```

位置：`vllm/vllm/v1/engine/core.py:492`

这个 grammar bitmask 来自 Scheduler / structured output manager，用于限制结构化输出请求下一步可选 token。

为什么在这里调用？

```text
SchedulerOutput 已经知道本轮哪些请求会执行；
如果这些请求包含 structured output decode，
采样时需要 grammar bitmask。
```

所以 step 中 structured output 的协作关系是：

```text
Scheduler.schedule()
  → 确定本轮执行请求
  → EngineCore 调用 scheduler.get_grammar_bitmask(scheduler_output)
  → model_executor.sample_tokens(grammar_output)
  → Scheduler.update_from_output() 中 grammar.accept_tokens()
```

注意：

```text
get_grammar_bitmask() 不负责推进 grammar 状态；
它只生成采样约束。
真正根据输出 token 推进 grammar 状态发生在 Scheduler.update_from_output()。
```

---

## 7. log_error_detail 和 log_iteration_details

EngineCore 等待模型输出时包了两个 context manager：

```python
with (
    self.log_error_detail(scheduler_output),
    self.log_iteration_details(scheduler_output),
):
    model_output = future.result()
    ...
```

位置：`vllm/vllm/v1/engine/core.py:493` 到 `vllm/vllm/v1/engine/core.py:500`

### 7.1 `log_error_detail()`

如果模型执行失败，它会 dump 更详细的 scheduler / input 信息：

```python
dump_engine_exception(
    self.vllm_config, scheduler_output, self.scheduler.make_stats()
)
```

位置：`vllm/vllm/v1/engine/core.py:427` 到 `vllm/vllm/v1/engine/core.py:430`

作用是：

```text
模型 forward 出错时，记录当前 SchedulerOutput 和 Scheduler stats，方便定位。
```

### 7.2 `log_iteration_details()`

如果开启 iteration details 日志，会根据 `SchedulerOutput` 统计本轮：

```text
context requests
context tokens
generation requests
generation tokens
iteration elapsed time
```

对应逻辑位置：`vllm/vllm/v1/engine/core.py:433` 到 `vllm/vllm/v1/engine/core.py:472`

这属于观测能力，不改变 step 主逻辑。

---

## 8. update_from_output 阶段：用真实结果更新 Scheduler

模型执行完成后，EngineCore 先处理 abort：

```python
self._process_aborts_queue()
```

位置：`vllm/vllm/v1/engine/core.py:503`

然后调用：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:504` 到 `vllm/vllm/v1/engine/core.py:506`

这里非常关键：

```text
update_from_output() 需要同时拿到 SchedulerOutput 和 ModelRunnerOutput。
```

原因是：

```text
SchedulerOutput 是本轮计划：
  哪些请求被调度、每个请求调度多少 token、有哪些 spec tokens / encoder inputs / connector metadata。

ModelRunnerOutput 是本轮真实结果：
  sampled tokens、pooling output、logprobs、KV connector output、routed experts 等。

Scheduler.update_from_output() 用真实结果对照执行计划，更新请求状态并构造 EngineCoreOutputs。
```

具体会处理：

```text
append sampled token；
spec decode accepted / rejected 回退；
async placeholder 消耗；
structured output grammar accept_tokens；
stop / finish reason；
streaming / resumable 请求；
KV connector finished_recving / finished_sending；
block / encoder cache 释放；
EngineCoreOutput 构造。
```

这些细节在 `../scheduler/08_update_after_worker_output.md` 中展开。

---

## 9. 为什么要先处理 aborts_queue

普通 `step()` 中，模型执行完成后、`update_from_output()` 前会执行：

```python
self._process_aborts_queue()
```

位置：`vllm/vllm/v1/engine/core.py:501` 到 `vllm/vllm/v1/engine/core.py:503`

`_process_aborts_queue()` 会把执行期间收到的 abort 合并成一批：

```python
request_ids = []
while not self.aborts_queue.empty():
    ids = self.aborts_queue.get_nowait()
    request_ids.extend((ids,) if isinstance(ids, str) else ids)
self.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/core.py:634` 到 `vllm/vllm/v1/engine/core.py:642`

为什么要在 `update_from_output()` 前处理？

因为模型 forward 期间，客户端可能已经取消请求。

如果不先 abort，`update_from_output()` 可能会继续给已取消请求 append token 或返回普通输出。

所以顺序是：

```text
Worker 结果回来
  → 先处理执行期间到达的 abort
  → 再 update_from_output
```

这样 Scheduler 在处理 Worker 输出时能看到请求已经是 finished / aborted，从而跳过或返回正确的 abort 输出。

---

## 10. step 返回值

`step()` 返回：

```python
return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

位置：`vllm/vllm/v1/engine/core.py:508`

返回值类型是：

```python
tuple[dict[int, EngineCoreOutputs], bool]
```

其中：

```text
engine_core_outputs：
  dict[int, EngineCoreOutputs]
  key 是 client_index，value 是该 client 本轮要收到的输出。

model_executed：
  bool
  表示本轮是否真正执行了模型 token 计算。
```

`model_executed` 的判断是：

```text
scheduler_output.total_num_scheduled_tokens > 0
```

也就是说：

```text
本轮调度 token 数大于 0，才认为模型真正执行了 token 计算。
```

注意：

```text
有些 step 可能没有调度 token，但仍然有 Scheduler 工作，
例如等待 KV connector 状态、处理 finished / cleanup 等。
```

所以 `model_executed=False` 不一定等于 EngineCore 完全空闲。

---

## 11. InprocClient 如何调用 step

同进程模式下，外层 `LLMEngine.step()` 会调用：

```python
outputs = self.engine_core.get_output()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:304`

`InprocClient.get_output()` 内部调用：

```python
outputs, model_executed = self.engine_core.step_fn()
self.engine_core.post_step(model_executed=model_executed)
return outputs and outputs.get(0) or EngineCoreOutputs()
```

位置：`vllm/vllm/v1/engine/core_client.py:289` 到 `vllm/vllm/v1/engine/core_client.py:292`

这里有两个点：

```text
1. EngineCore.step_fn() 可能是 step，也可能是 step_with_batch_queue；
2. post_step() 在每轮后执行，用于处理 draft token ids 等后处理。
```

---

## 12. post_step 做什么

`post_step()` 入口：

```python
def post_step(self, model_executed: bool) -> None:
```

位置：`vllm/vllm/v1/engine/core.py:510`

主要逻辑是：

```python
if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
    draft_token_ids = self.model_executor.take_draft_token_ids()
    if draft_token_ids is not None:
        self.scheduler.update_draft_token_ids(draft_token_ids)
```

位置：`vllm/vllm/v1/engine/core.py:510` 到 `vllm/vllm/v1/engine/core.py:517`

含义是：

```text
非 async scheduling 下，如果启用 speculative decoding 或 diffusion 相关 draft token，
EngineCore 在 step 后从 model_executor 取 draft token ids，
再交给 Scheduler 更新到对应 request 上。
```

注意注释：

```python
# When using async scheduling we can't get draft token ids in advance,
# so we update draft token ids in the worker process and don't
# need to update draft token ids here.
```

位置：`vllm/vllm/v1/engine/core.py:510` 到 `vllm/vllm/v1/engine/core.py:513`

所以：

```text
同步 / 非 async scheduling：
  draft token ids 由 EngineCore.post_step() 从 model_executor 取出后交给 Scheduler。

async scheduling：
  draft token ids 在 worker 进程里更新，不走这里。
```

---

## 13. step_fn：普通 step 和 batch queue step

EngineCore 初始化时会选择：

```python
self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)
```

位置：`vllm/vllm/v1/engine/core.py:221` 到 `vllm/vllm/v1/engine/core.py:223`

而 `batch_queue` 是否启用取决于：

```python
self.batch_queue_size = vllm_config.max_concurrent_batches
...
if self.batch_queue_size > 1:
    self.batch_queue = deque(maxlen=self.batch_queue_size)
```

位置：`vllm/vllm/v1/engine/core.py:196` 到 `vllm/vllm/v1/engine/core.py:202`

所以有两种 step 模式：

```text
max_concurrent_batches == 1：
  使用普通 step()。

max_concurrent_batches > 1：
  使用 step_with_batch_queue()。
```

普通 `step()` 是：

```text
调度一批 → 执行一批 → 立即等结果 → update_from_output。
```

batch queue 模式是：

```text
尽量先调度新 batch 填满队列；
如果队列未满且还有可调度请求，可以先不等结果；
队列满或没有新 batch 可调度时，再取最早完成的一批做 update_from_output。
```

---

## 14. step_with_batch_queue 的作用

`step_with_batch_queue()` 注释说明：

```python
"""Schedule and execute batches with the batch queue.
Note that if nothing to output in this step, None is returned.

The execution flow is as follows:
1. Try to schedule a new batch if the batch queue is not full.
If a new batch is scheduled, directly return an empty engine core
output. In other words, fulfilling the batch queue has a higher priority
than getting model outputs.
2. If there is no new scheduled batch, meaning that the batch queue
is full or no other requests can be scheduled, we block until the first
batch in the job queue is finished.
3. Update the scheduler from the output.
"""
```

位置：`vllm/vllm/v1/engine/core.py:519` 到 `vllm/vllm/v1/engine/core.py:534`

它的目的主要是：

```text
支持 pipeline parallelism / async scheduling 下多个 in-flight batches，
尽量减少 pipeline bubble，提高吞吐。
```

初始化时也有注释：

```python
# Batch queue for scheduled batches. This enables us to asynchronously
# schedule and execute batches, and is required by pipeline parallelism
# to eliminate pipeline bubbles.
```

位置：`vllm/vllm/v1/engine/core.py:192` 到 `vllm/vllm/v1/engine/core.py:195`

### 14.1 batch queue 的基本流程

简化流程：

```text
step_with_batch_queue()
  │
  ├─ 如果 batch_queue 未满，并且 Scheduler 有请求：
  │    ├─ scheduler.schedule()
  │    ├─ model_executor.execute_model(..., non_block=True)
  │    ├─ 如可立即采样，则 sample_tokens(non_block=True)
  │    ├─ 把 future + scheduler_output 放进 batch_queue
  │    └─ 如果队列还没满且还有可调度请求，可以先 return None
  │
  ├─ 否则 / 或需要回收结果：
  │    ├─ 从 batch_queue pop 最早结果
  │    ├─ future.result()
  │    ├─ _process_aborts_queue()
  │    ├─ scheduler.update_from_output()
  │    └─ 返回 EngineCoreOutputs
```

对应核心代码：

```python
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
exec_future = self.model_executor.execute_model(
    scheduler_output, non_block=True
)
...
batch_queue.appendleft((future, scheduler_output, exec_future))
...
future, scheduler_output, exec_model_fut = batch_queue.pop()
model_output = future.result()
...
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:546` 到 `vllm/vllm/v1/engine/core.py:607`

### 14.2 为什么可能返回 None

`step_with_batch_queue()` 的返回类型是：

```python
tuple[dict[int, EngineCoreOutputs] | None, bool]
```

位置：`vllm/vllm/v1/engine/core.py:519` 到 `vllm/vllm/v1/engine/core.py:521`

如果本轮只是调度新 batch、提交执行，但还不回收任何 batch 输出，可能返回：

```python
return None, model_executed
```

位置：`vllm/vllm/v1/engine/core.py:576` 到 `vllm/vllm/v1/engine/core.py:581`

含义：

```text
本轮 EngineCore 做了调度 / 提交模型执行，
但暂时没有 EngineCoreOutputs 可以返回。
```

### 14.3 deferred structured output sampling

batch queue 模式下，如果有 structured output 依赖前一步 token 才能生成 bitmask，可能会 deferred sampling：

```python
if not scheduler_output.pending_structured_output_tokens:
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    future = self.model_executor.sample_tokens(
        grammar_output, non_block=True
    )
else:
    deferred_scheduler_output = scheduler_output
```

位置：`vllm/vllm/v1/engine/core.py:560` 到 `vllm/vllm/v1/engine/core.py:571`

后面在处理完前一个输出后，再为 deferred request 生成 grammar bitmask 并 sample：

```python
grammar_output = self.scheduler.get_grammar_bitmask(
    deferred_scheduler_output
)
future = self.model_executor.sample_tokens(grammar_output, non_block=True)
batch_queue.appendleft((future, deferred_scheduler_output, exec_future))
```

位置：`vllm/vllm/v1/engine/core.py:626` 到 `vllm/vllm/v1/engine/core.py:630`

这说明 batch queue 模式下，EngineCore 不只是简单排队 future，还要处理 structured output 和 speculative draft token 的时序依赖。

---

## 15. EngineCoreProc busy loop 如何驱动 step

在后台进程模式中，不是外部直接调用 `step()`，而是 EngineCoreProc 自己跑 busy loop。

入口：

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()
```

位置：`vllm/vllm/v1/engine/core.py:1257` 到 `vllm/vllm/v1/engine/core.py:1264`

主线是：

```text
run_busy_loop()
  → _process_input_queue()
  → _process_engine_step()
  → output_queue
  → output socket
```

### 15.1 什么时候认为有 work

`has_work()`：

```python
return (
    self.engines_running
    or self.scheduler.has_requests()
    or bool(self.batch_queue)
)
```

位置：`vllm/vllm/v1/engine/core.py:1245` 到 `vllm/vllm/v1/engine/core.py:1251`

也就是说，只要满足任一条件，就会继续 step：

```text
DP engines_running 为 True；
Scheduler 里还有请求；
batch_queue 里还有未回收的 batch。
```

### 15.2 _process_engine_step()

真正调用 step_fn 的地方：

```python
outputs, model_executed = self.step_fn()
for output in outputs.items() if outputs else ():
    self.output_queue.put_nowait(output)
self.post_step(model_executed)
```

位置：`vllm/vllm/v1/engine/core.py:1298` 到 `vllm/vllm/v1/engine/core.py:1307`

如果本轮没有模型执行，但 Scheduler 仍有请求：

```python
if not model_executed and self.scheduler.has_requests():
    time.sleep(0.001)
```

位置：`vllm/vllm/v1/engine/core.py:1309` 到 `vllm/vllm/v1/engine/core.py:1313`

注释说明：

```python
# If no model execution happened but there is still scheduler work
# (e.g. WAITING_FOR_REMOTE_KVS or delayed KV connector frees), yield
# the GIL briefly to allow background transfer threads to make progress.
```

位置：`vllm/vllm/v1/engine/core.py:1309` 到 `vllm/vllm/v1/engine/core.py:1311`

这说明：

```text
EngineCore 可能处于“没有模型 forward，但还有调度/connector 清理工作”的状态。
```

例如：

```text
WAITING_FOR_REMOTE_KVS 等远端 KV load 完成；
delayed KV connector frees 等后台线程完成。
```

---

## 16. DP EngineCore 的 step 特殊性

DP 场景使用 `DPEngineCoreProc`，它覆盖了部分行为。

### 16.1 DP prefill balancing

DP EngineCore 会覆盖 `_should_throttle_prefills()`：

```python
return (
    self.prefill_schedule_interval > 1
    and self.step_counter % self.prefill_schedule_interval != 0
)
```

位置：`vllm/vllm/v1/engine/core.py:1914` 到 `vllm/vllm/v1/engine/core.py:1921`

这会影响：

```python
self.scheduler.schedule(self._should_throttle_prefills())
```

也就是说 DP EngineCore 能在某些 step 告诉 Scheduler：

```text
本轮延后 prefill。
```

### 16.2 没有真实执行时可能执行 dummy batch

DP busy loop 中，如果本轮没有执行模型，但全局仍处于 running wave，可能执行 dummy batch：

```python
if not executed:
    if not local_unfinished_reqs and not self.engines_running:
        continue

    # We are in a running state and so must execute a dummy pass
    # if the model didn't execute any ready requests.
    with self.log_iteration_details(None):
        self.execute_dummy_batch()
```

位置：`vllm/vllm/v1/engine/core.py:1944` 到 `vllm/vllm/v1/engine/core.py:1953`

作用是：

```text
在 DP / MoE 等场景中，即使本 rank 没有 ready request，
也可能需要执行 dummy pass 来保持集体通信 / wave 同步。
```

### 16.3 全局 unfinished 判断

DP EngineCore 还会周期性同步全局是否还有未完成请求：

```python
self.engines_running = self._has_global_unfinished_reqs(
    local_unfinished_reqs
)
```

位置：`vllm/vllm/v1/engine/core.py:1955` 到 `vllm/vllm/v1/engine/core.py:1958`

`_has_global_unfinished_reqs()` 内部使用 DP group 同步：

```python
has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
    self.dp_group,
    has_unfinished=local_unfinished,
    pending_pause=self.pending_pause,
)
```

位置：`vllm/vllm/v1/engine/core.py:1988` 到 `vllm/vllm/v1/engine/core.py:1992`

所以 DP 模式下，一轮 step 不只是本地 Scheduler / Worker 闭环，还要参与全局 DP wave 状态同步。

---

## 17. 一轮 step 中几个核心对象的关系

### 17.1 SchedulerOutput

来源：

```python
scheduler_output = self.scheduler.schedule(...)
```

作用：

```text
本轮执行计划。
```

包含：

```text
scheduled_new_reqs
scheduled_cached_reqs
num_scheduled_tokens
scheduled_encoder_inputs
scheduled_spec_decode_tokens
kv_connector_metadata
ec_connector_metadata
finished_req_ids
```

### 17.2 ModelRunnerOutput

来源：

```python
model_output = future.result()
# 或
model_output = self.model_executor.sample_tokens(grammar_output)
```

作用：

```text
Worker / ModelRunner 真实执行结果。
```

包含：

```text
sampled_token_ids
logprobs
pooler_output
kv_connector_output
req_id_to_index
routed_experts
```

### 17.3 EngineCoreOutputs

来源：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

作用：

```text
EngineCore 返回给外层 Engine / AsyncLLM 的内部输出。
```

之后由外层 `OutputProcessor` 转成用户可见的：

```text
RequestOutput
PoolingRequestOutput
```

---

## 18. 容易疑惑的点

### 18.1 EngineCore.step() 是否每次都会执行模型 forward？

不一定。

如果 Scheduler 没有请求：

```text
直接 return {}, False。
```

如果 Scheduler 有请求但本轮没有可调度 token，例如等待远端 KV、connector cleanup 等，也可能：

```text
model_executed = False。
```

### 18.2 Scheduler.schedule() 之后，Worker 是否一定会返回 sampled token？

不一定。

可能情况：

```text
prefill chunk：通常没有 sampled token；
pooling request：可能返回 pooling output；
普通 decode：通常返回 sampled token；
spec decode：可能返回多个 token；
空调度 / connector-only step：可能没有 token。
```

### 18.3 为什么 update_from_output 需要 SchedulerOutput？

因为 SchedulerOutput 是本轮计划。

Worker 返回的 ModelRunnerOutput 只说明“实际返回了什么”，但 Scheduler 还需要知道：

```text
本轮哪些请求被调度；
每个请求调度了多少 token；
本轮发出了哪些 spec tokens；
本轮安排了哪些 encoder inputs；
哪些请求是 new / cached / resumed。
```

这些在 SchedulerOutput 里。

### 18.4 为什么 execute_model 后还要 sample_tokens？

因为某些路径会把 forward 和 sampling 拆开。

`execute_model()` 返回 `None` 时，EngineCore 会调用：

```python
model_output = self.model_executor.sample_tokens(grammar_output)
```

这允许 structured output grammar bitmask 在 sampling 阶段生效。

### 18.5 batch queue 模式为什么可能先不返回输出？

因为它优先填充 in-flight batch queue，减少 pipeline bubbles。

也就是说：

```text
本轮可能只是提交了一个新的 batch，
但没有等待任何 batch 完成，
所以没有 EngineCoreOutputs 返回。
```

### 18.6 post_step 为什么不在 step 里面？

因为不同客户端模式需要统一在每轮 step 后执行后处理。

例如 InprocClient：

```python
outputs, model_executed = self.engine_core.step_fn()
self.engine_core.post_step(model_executed=model_executed)
```

后台 EngineCoreProc 中也是 `_process_engine_step()` 在 step_fn 后调用：

```python
self.post_step(model_executed)
```

这样普通 step 和 batch queue step 都能复用同一个后处理逻辑。

---

## 19. 从“回答问题”的角度总结

如果要问：

```text
EngineCore.step() 如何驱动一轮执行？
```

可以回答：

```text
EngineCore.step() 是 EngineCore 的一轮执行闭环。
它先让 Scheduler.schedule() 生成 SchedulerOutput，
再把这个 SchedulerOutput 交给 model_executor / Worker / ModelRunner 执行，
拿到 ModelRunnerOutput 后，先处理执行期间到达的 abort，
再调用 Scheduler.update_from_output() 更新请求状态并生成 EngineCoreOutputs。
```

普通 step 是串行闭环：

```text
schedule → execute → output → update
```

batch queue step 是流水线闭环：

```text
尽量先 schedule / execute 新 batch，
再按需要回收较早 batch 的 output。
```

后台进程模式下，EngineCoreProc 用 busy loop 驱动 step：

```text
process input queue
  → step_fn
  → output_queue
  → output socket
```

---

## 20. 最关键流程图

```text
普通 EngineCore.step()

if not scheduler.has_requests():
    return {}, False

scheduler_output = scheduler.schedule(throttle_prefills)

future = model_executor.execute_model(
    scheduler_output,
    non_block=True,
)

grammar_output = scheduler.get_grammar_bitmask(scheduler_output)

model_output = future.result()
if model_output is None:
    model_output = model_executor.sample_tokens(grammar_output)

_process_aborts_queue()

engine_core_outputs = scheduler.update_from_output(
    scheduler_output,
    model_output,
)

return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

```text
batch queue 模式

if scheduler.has_requests() and batch_queue not full:
    scheduler_output = scheduler.schedule(...)
    exec_future = model_executor.execute_model(...)
    sample future if needed
    batch_queue.appendleft((future, scheduler_output, exec_future))
    if queue still not full and more work can be scheduled:
        return None, model_executed

future, scheduler_output, exec_model_fut = batch_queue.pop()
model_output = future.result()
_process_aborts_queue()
engine_core_outputs = scheduler.update_from_output(
    scheduler_output,
    model_output,
)
return engine_core_outputs, model_executed
```

```text
EngineCoreProc busy loop

while _handle_shutdown():
    _process_input_queue()
    outputs, model_executed = step_fn()
    for client_index, engine_core_outputs in outputs.items():
        output_queue.put_nowait((client_index, engine_core_outputs))
    post_step(model_executed)
```
