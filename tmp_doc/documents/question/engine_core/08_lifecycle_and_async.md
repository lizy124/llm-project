# 08. EngineCore 的异步、并发和生命周期管理

源码位置：`vllm/vllm/v1/engine/core.py`

本问题关注：EngineCore 除主执行链路外的异步、并发、Worker 生命周期、profile / reset / sleep / wakeup / shutdown 等辅助能力。

---

## 1. 一句话回答

`EngineCore` 不只是 `step()` 主循环。

它还负责：

```text
创建 model_executor / Scheduler；
初始化 KV cache；
管理 batch queue / async scheduling；
处理 abort 队列；
转发 profile / reset / sleep / wakeup / LoRA 等 utility 调用；
在后台进程模式下运行 ZMQ input / output 线程和 busy loop；
处理 executor failure 和 shutdown。
```

一句话：

```text
EngineCore 是 vLLM V1 内部执行闭环的生命周期管理者；
EngineCoreProc 则是在多进程模式下给 EngineCore 加上 ZMQ 通信、后台 busy loop、输入输出线程和 shutdown 协议。
```

主线可以分成两类：

```text
in-process：
  外层 EngineCoreClient 直接调用 EngineCore.step() / add_request() / utility 方法。

multi-process：
  EngineCoreProc 在后台进程 busy loop 中运行，
  前端通过 ZMQ 发送 ADD / ABORT / UTILITY，
  EngineCoreProc 通过 output_queue + ZMQ 返回 EngineCoreOutputs。
```

---

## 2. EngineCore 初始化时管理什么

`EngineCore` 初始化入口是：

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
```

位置：`vllm/vllm/v1/engine/core.py:96` 到 `vllm/vllm/v1/engine/core.py:97`

初始化时先创建 `model_executor`：

```python
self.model_executor = executor_class(vllm_config)
if executor_fail_callback is not None:
    self.model_executor.register_failure_callback(executor_fail_callback)
```

位置：`vllm/vllm/v1/engine/core.py:122` 到 `vllm/vllm/v1/engine/core.py:125`

然后初始化 KV cache：

```python
kv_cache_config = self._initialize_kv_caches(vllm_config)
self.structured_output_manager = StructuredOutputManager(vllm_config)
```

位置：`vllm/vllm/v1/engine/core.py:132` 到 `vllm/vllm/v1/engine/core.py:134`

再创建 Scheduler：

```python
self.scheduler: SchedulerInterface = Scheduler(
    vllm_config=vllm_config,
    kv_cache_config=kv_cache_config,
    structured_output_manager=self.structured_output_manager,
    include_finished_set=include_finished_set,
    log_stats=self.log_stats,
    block_size=scheduler_block_size,
    hash_block_size=hash_block_size,
)
```

位置：`vllm/vllm/v1/engine/core.py:150` 到 `vllm/vllm/v1/engine/core.py:158`

这说明初始化顺序是：

```text
EngineCore
  → load_general_plugins()
  → 创建 model_executor
  → 初始化 / profile KV cache
  → 创建 structured_output_manager
  → 从 scheduler_config 解析 Scheduler 类并创建 Scheduler
  → 初始化 connector / multimodal receiver cache / batch queue / EC consumer 标记
  → 初始化 prefix cache hashing、abort queue、GC / env cache 状态
```

如果 Scheduler 有 connector，会先让 model executor 初始化 KV output aggregator：

```python
if self.scheduler.connector is not None:
    self.model_executor.init_kv_output_aggregator(self.scheduler.connector)
```

位置：`vllm/vllm/v1/engine/core.py:163` 到 `vllm/vllm/v1/engine/core.py:164`

EngineCore 还会按多模态配置创建 engine 侧 receiver cache：

```python
self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
    vllm_config
)
```

位置：`vllm/vllm/v1/engine/core.py:166` 到 `vllm/vllm/v1/engine/core.py:169`

如果 Scheduler 有 KV connector，还会把 Worker 侧 handshake metadata 收集到 Scheduler connector：

```python
kv_connector = self.scheduler.get_kv_connector()
if kv_connector is not None:
    xfer_handshake_metadata = (
        self.model_executor.get_kv_connector_handshake_metadata()
    )
    ...
    kv_connector.set_xfer_handshake_metadata_pp_aware(content)
```

位置：`vllm/vllm/v1/engine/core.py:171` 到 `vllm/vllm/v1/engine/core.py:190`

所以 EngineCore 初始化不仅是创建对象，还要完成：

```text
模型执行环境初始化；
KV cache 配置探测；
Scheduler 状态账本初始化；
KV / EC / MM connector 上下文同步；
Worker 和 Scheduler 之间的元信息对齐。
```

---

## 3. 同步 EngineCore：InprocClient 直接调用

in-process 模式中，`EngineCore` 和外层 `LLMEngine` 在同一进程。

`InprocClient` 初始化时直接创建 EngineCore：

```python
self.engine_core = EngineCore(*args, **kwargs)
```

位置：`vllm/vllm/v1/engine/core_client.py:286` 到 `vllm/vllm/v1/engine/core_client.py:287`

拉输出时直接执行一步：

```python
def get_output(self) -> EngineCoreOutputs:
    outputs, model_executed = self.engine_core.step_fn()
    self.engine_core.post_step(model_executed=model_executed)
    return outputs and outputs.get(0) or EngineCoreOutputs()
```

位置：`vllm/vllm/v1/engine/core_client.py:289` 到 `vllm/vllm/v1/engine/core_client.py:292`

请求进入时也直接调用 EngineCore：

```python
req, request_wave = self.engine_core.preprocess_add_request(request)
self.engine_core.add_request(req, request_wave)
```

位置：`vllm/vllm/v1/engine/core_client.py:297` 到 `vllm/vllm/v1/engine/core_client.py:299`

所以同步 in-process 路径是：

```text
LLMEngine.step()
  → InprocClient.get_output()
  → EngineCore.step_fn()
  → EngineCore.post_step()
  → EngineCoreOutputs
```

这里没有后台 busy loop，也没有 ZMQ 输入输出线程。

---

## 4. batch queue 和 async scheduling

EngineCore 支持 batch queue，用于 pipeline parallel 等场景减少 pipeline bubble。

初始化时会读取：

```python
self.batch_queue_size = vllm_config.max_concurrent_batches
self.batch_queue = None
if self.batch_queue_size > 1:
    self.batch_queue = deque(maxlen=self.batch_queue_size)
```

位置：`vllm/vllm/v1/engine/core.py:192` 到 `vllm/vllm/v1/engine/core.py:203`

然后选择 step 函数：

```python
self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)
self.async_scheduling = vllm_config.scheduler_config.async_scheduling
```

位置：`vllm/vllm/v1/engine/core.py:221` 到 `vllm/vllm/v1/engine/core.py:224`

普通 `step()` 是：

```text
schedule → execute → wait result → update_from_output
```

而 `step_with_batch_queue()` 是：

```text
优先 schedule 新 batch 入队；
如果队列未满且不需要立刻返回输出，可以先返回 None；
之后再 pop 最早完成的 future，拿 model_output 做 update_from_output。
```

源码注释说明：

```python
"""Schedule and execute batches with the batch queue.
Note that if nothing to output in this step, None is returned.

The execution flow is as follows:
1. Try to schedule a new batch if the batch queue is not full.
...
2. If there is no new scheduled batch ... we block until the first
batch in the job queue is finished.
3. Update the scheduler from the output.
"""
```

位置：`vllm/vllm/v1/engine/core.py:528` 到 `vllm/vllm/v1/engine/core.py:543`

所以 batch queue 改变的是：

```text
SchedulerOutput 和 ModelRunnerOutput 不一定在同一次 step 调用中立即配对；
SchedulerOutput 会和 future 一起暂存在 batch_queue，
等对应 future 完成后再回到 update_from_output。
```

当前源码里还要区分 generation sampling 是否能立即执行：

```python
if not scheduler_output.pending_structured_output_tokens:
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    future = self.model_executor.sample_tokens(grammar_output, non_block=True)
else:
    deferred_scheduler_output = scheduler_output
```

位置：`vllm/vllm/v1/engine/core.py:568` 到 `vllm/vllm/v1/engine/core.py:580`

如果结构化输出需要等待前一个 batch 的结果，sampling 会被延后；当前 batch 出队并完成 `update_from_output()` 后，如果启用 draft tokens，会先用 `update_draft_token_ids_in_output()` 过滤 deferred scheduler output 中的无效 spec tokens，再计算 grammar bitmask 并调用 `sample_tokens()`：

```python
grammar_output = self.scheduler.get_grammar_bitmask(
    deferred_scheduler_output
)
future = self.model_executor.sample_tokens(grammar_output, non_block=True)
batch_queue.appendleft((future, deferred_scheduler_output, exec_future))
```

位置：`vllm/vllm/v1/engine/core.py:621` 到 `vllm/vllm/v1/engine/core.py:639`

---

## 5. 多进程 EngineCoreProc：后台 busy loop

多进程模式中，真正跑的是：

```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
```

位置：`vllm/vllm/v1/engine/core.py:905` 到 `vllm/vllm/v1/engine/core.py:906`

它在 EngineCore 外面加了：

```text
ZMQ input sockets；
ZMQ output sockets；
input_queue；
output_queue；
input thread；
output thread；
background busy loop；
shutdown state。
```

初始化时创建队列：

```python
self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
executor_fail_callback = lambda: self.input_queue.put_nowait(
    (EngineCoreRequestType.EXECUTOR_FAILED, b"")
)
```

位置：`vllm/vllm/v1/engine/core.py:924` 到 `vllm/vllm/v1/engine/core.py:928`

然后启动输入线程和输出线程：

```python
input_thread = threading.Thread(
    target=self.process_input_sockets,
    ...,
    daemon=True,
)
input_thread.start()

self.output_thread = threading.Thread(
    target=self.process_output_sockets,
    ...,
    daemon=True,
)
self.output_thread.start()
```

位置：`vllm/vllm/v1/engine/core.py:989` 到 `vllm/vllm/v1/engine/core.py:1010`

真正的后台主循环是：

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()

    raise SystemExit
```

位置：`vllm/vllm/v1/engine/core.py:1268` 到 `vllm/vllm/v1/engine/core.py:1276`

可以理解为：

```text
EngineCoreProc 自己不断循环：
  先处理前端输入；
  再执行一轮 EngineCore step；
  再把输出送回前端。
```

---

## 6. 输入队列如何驱动 step

`EngineCoreProc._process_input_queue()` 会在没有工作时阻塞等待输入：

```python
while not self.has_work() and self.is_running():
    ...
    req = self.input_queue.get(block=block)
    self._handle_client_request(*req)
```

位置：`vllm/vllm/v1/engine/core.py:1281` 到 `vllm/vllm/v1/engine/core.py:1295`

之后会把队列里剩余输入也处理掉：

```python
while not self.input_queue.empty():
    req = self.input_queue.get_nowait()
    self._handle_client_request(*req)
```

位置：`vllm/vllm/v1/engine/core.py:1304` 到 `vllm/vllm/v1/engine/core.py:1307`

处理完输入后，`_process_engine_step()` 调用 `step_fn()`：

```python
outputs, model_executed = self.step_fn()
for output in outputs.items() if outputs else ():
    self.output_queue.put_nowait(output)
self.post_step(model_executed)
```

位置：`vllm/vllm/v1/engine/core.py:1312` 到 `vllm/vllm/v1/engine/core.py:1318`

如果没有实际模型执行，但 Scheduler 还有请求，会短暂 sleep 让后台传输线程推进：

```python
if not model_executed and self.scheduler.has_requests():
    time.sleep(0.001)
```

位置：`vllm/vllm/v1/engine/core.py:1320` 到 `vllm/vllm/v1/engine/core.py:1324`

所以后台 loop 的节奏是：

```text
input_queue 有请求 / Scheduler 有请求 / batch_queue 有任务
  → has_work() 为 True
  → 执行 step_fn()
  → 产出 EngineCoreOutputs
  → output_queue
```

---

## 7. client request 如何分发

后台进程中，前端发来的请求会统一进入 `_handle_client_request()`：

```python
def _handle_client_request(
    self, request_type: EngineCoreRequestType, request: Any
) -> None:
```

位置：`vllm/vllm/v1/engine/core.py:1381` 到 `vllm/vllm/v1/engine/core.py:1383`

主要分发类型是：

```python
if request_type == EngineCoreRequestType.WAKEUP:
    return
elif request_type == EngineCoreRequestType.ADD:
    ...
    self.add_request(req, request_wave)
elif request_type == EngineCoreRequestType.ABORT:
    self.abort_requests(request)
elif request_type == EngineCoreRequestType.UTILITY:
    ...
    self._invoke_utility_method(...)
elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
    raise RuntimeError("Executor failed.")
```

位置：`vllm/vllm/v1/engine/core.py:1386` 到 `vllm/vllm/v1/engine/core.py:1410`

含义是：

```text
ADD：添加请求；
ABORT：取消请求；
UTILITY：profile / reset / sleep / wakeup / LoRA / collective_rpc 等辅助调用；
EXECUTOR_FAILED：Worker / executor 失败信号；
WAKEUP：唤醒 input queue。
```

如果是 utility 调用，EngineCoreProc 会把结果包装成 `UtilityOutput`，再放进 `EngineCoreOutputs` 发回客户端：

```python
enqueue_output = lambda out: self.output_queue.put_nowait(
    (client_idx, EngineCoreOutputs(utility_output=out))
)
```

位置：`vllm/vllm/v1/engine/core.py:1405` 到 `vllm/vllm/v1/engine/core.py:1407`

---

## 8. utility 调用如何支持 Future

有些 utility 调用可能立即返回，有些可能返回 `Future`。

`_invoke_utility_method()` 会统一处理：

```python
result = get_result()
if isinstance(result, Future):
    # Defer utility output handling until future completion.
    callback = lambda future: EngineCoreProc._invoke_utility_method(
        name, future.result, output, enqueue_output
    )
    result.add_done_callback(callback)
    return
output.result = UtilityResult(result)
```

位置：`vllm/vllm/v1/engine/core.py:1447` 到 `vllm/vllm/v1/engine/core.py:1456`

如果调用失败，会返回 failure message：

```python
except Exception as e:
    logger.exception("Invocation of %s method failed", name)
    output.failure_message = f"Call to {name} method failed: {str(e)}"
enqueue_output(output)
```

位置：`vllm/vllm/v1/engine/core.py:1457` 到 `vllm/vllm/v1/engine/core.py:1460`

这说明 utility 调用和普通 request 输出不同：

```text
普通 request 输出走 EngineCoreOutput / EngineCoreOutputs.outputs；
utility 输出走 EngineCoreOutputs.utility_output。
```

前端客户端会识别 `utility_output`，把结果填回等待的 future，而不是交给 OutputProcessor 当作请求输出处理。

---

## 9. profile / reset / sleep / wakeup

EngineCore 还暴露一组运行时控制能力。

### 9.1 profile

`profile()` 直接转发给 model executor：

```python
def profile(self, is_start: bool = True, profile_prefix: str | None = None):
    self.model_executor.profile(is_start, profile_prefix)
```

位置：`vllm/vllm/v1/engine/core.py:671` 到 `vllm/vllm/v1/engine/core.py:672`

也就是说：

```text
EngineCore 接收 profile utility；
真正 profiler 控制在 Worker / model_executor 侧执行。
```

### 9.2 reset_mm_cache

`reset_mm_cache()` 会先检查是否还有未完成请求：

```python
if self.scheduler.has_unfinished_requests():
    logger.warning(
        "Resetting the multi-modal cache when requests are "
        "in progress may lead to desynced internal caches."
    )
```

位置：`vllm/vllm/v1/engine/core.py:674` 到 `vllm/vllm/v1/engine/core.py:681`

然后清理 EngineCore 侧 cache，并通知 executor：

```python
if self.mm_receiver_cache is not None:
    self.mm_receiver_cache.clear_cache()

self.model_executor.reset_mm_cache()
```

位置：`vllm/vllm/v1/engine/core.py:683` 到 `vllm/vllm/v1/engine/core.py:687`

### 9.3 reset_prefix_cache

prefix cache reset 交给 Scheduler：

```python
def reset_prefix_cache(
    self, reset_running_requests: bool = False, reset_connector: bool = False
) -> bool:
    return self.scheduler.reset_prefix_cache(
        reset_running_requests, reset_connector
    )
```

位置：`vllm/vllm/v1/engine/core.py:689` 到 `vllm/vllm/v1/engine/core.py:694`

因为 prefix cache 的逻辑状态和 KV block 管理在 Scheduler / KV cache manager 侧。

### 9.4 reset_encoder_cache

encoder cache reset 同时清 Scheduler 侧逻辑状态和 Worker 侧物理存储：

```python
self.scheduler.reset_encoder_cache()
self.model_executor.reset_encoder_cache()
```

位置：`vllm/vllm/v1/engine/core.py:711` 到 `vllm/vllm/v1/engine/core.py:714`

文档注释也说明：

```text
Clears both the scheduler's cache manager and the GPU model runner's cache.
```

位置：`vllm/vllm/v1/engine/core.py:696` 到 `vllm/vllm/v1/engine/core.py:702`

---

## 10. pause / sleep / wakeup

`sleep()` 是更高层的暂停和显存管理接口。

源码注释说明三种 level：

```python
level: Sleep level.
    - Level 0: Pause scheduling only. Requests are still accepted
               but not processed. No GPU memory changes.
    - Level 1: Offload model weights to CPU, discard KV cache.
    - Level 2: Discard all GPU memory.
```

位置：`vllm/vllm/v1/engine/core.py:770` 到 `vllm/vllm/v1/engine/core.py:780`

`sleep()` 先 pause scheduler：

```python
clear_prefix_cache = level >= 1
pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
if level < 1:
    return pause_future
```

位置：`vllm/vllm/v1/engine/core.py:783` 到 `vllm/vllm/v1/engine/core.py:787`

level 1 以上再交给 executor 管理 GPU 内存：

```python
model_executor = self.model_executor
if pause_future is None:
    model_executor.sleep(level)
    return None
```

位置：`vllm/vllm/v1/engine/core.py:789` 到 `vllm/vllm/v1/engine/core.py:793`

如果 pause 需要等待，会在 Future 完成后再 sleep：

```python
def pause_complete(f: Future):
    try:
        f.result()
        future.set_result(model_executor.sleep(level))
    except Exception as e:
        future.set_exception(e)
```

位置：`vllm/vllm/v1/engine/core.py:795` 到 `vllm/vllm/v1/engine/core.py:806`

`wake_up()` 则反过来。它会先处理只恢复调度用的 `"scheduling"` tag，再按需唤醒 executor；只有 executor 已经完全不处于 sleeping 状态时，才恢复 scheduler，partial wake 会继续保持调度暂停：

```python
if tags is not None and "scheduling" in tags:
    tags = [t for t in tags if t != "scheduling"]

if tags is None or tags:
    self.model_executor.wake_up(tags)

if not self.model_executor.is_sleeping:
    self.resume_scheduler()
```

位置：`vllm/vllm/v1/engine/core.py:808` 到 `vllm/vllm/v1/engine/core.py:824`

`is_sleeping()` 会同时看 scheduler pause 状态和 executor sleeping 状态：

```python
return self.is_scheduler_paused() or self.model_executor.is_sleeping
```

位置：`vllm/vllm/v1/engine/core.py:826` 到 `vllm/vllm/v1/engine/core.py:828`

所以 sleep / wakeup 的边界是：

```text
Scheduler pause/resume：由 EngineCore 控制；
GPU memory sleep/wakeup：委托给 model_executor / Worker。
```

---

## 11. pause_scheduler 的模式

`pause_scheduler()` 支持不同暂停模式：

```python
- ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
  outputs to be sent ..., optionally clear caches.
- ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
  optionally clear caches.
- ``keep``: Set PAUSED_ALL; return a Future that completes when the
  output queue is empty.
```

位置：`vllm/vllm/v1/engine/core.py:731` 到 `vllm/vllm/v1/engine/core.py:746`

in-process 的普通 EngineCore 不支持 wait：

```python
if mode == "wait":
    raise ValueError("'wait' mode can't be used in inproc-engine mode")
```

位置：`vllm/vllm/v1/engine/core.py:749` 到 `vllm/vllm/v1/engine/core.py:750`

如果是 abort 模式，会让 Scheduler 结束所有请求：

```python
if mode == "abort":
    self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
```

位置：`vllm/vllm/v1/engine/core.py:752` 到 `vllm/vllm/v1/engine/core.py:753`

然后设置 pause state：

```python
pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
self.scheduler.set_pause_state(pause_state)
```

位置：`vllm/vllm/v1/engine/core.py:755` 到 `vllm/vllm/v1/engine/core.py:756`

可以理解为：

```text
PAUSED_NEW：新请求排队，但允许已有请求按模式处理；
PAUSED_ALL：全部暂停，不继续 step；
UNPAUSED：恢复正常调度。
```

---

## 12. abort 队列和并发安全

EngineCore 有一个 abort queue：

```python
self.aborts_queue = queue.Queue[list[str]]()
```

位置：`vllm/vllm/v1/engine/core.py:226`

普通 `step()` 在 Worker 执行完成后、处理输出前，会先处理执行期间到达的 abort：

```python
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:501` 到 `vllm/vllm/v1/engine/core.py:506`

`_process_aborts_queue()` 会批量取出 request ids：

```python
while not self.aborts_queue.empty():
    ids = self.aborts_queue.get_nowait()
    request_ids.extend((ids,) if isinstance(ids, str) else ids)
self.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/core.py:634` 到 `vllm/vllm/v1/engine/core.py:642`

这样做的原因是：

```text
Worker forward 期间，前端可能发来 abort；
这些 abort 不能打断正在执行的 GPU kernel，
但必须在 update_from_output() 前生效，
避免 Scheduler 给已取消请求继续产出正常输出。
```

后台进程空闲等待时也会清理 abort queue：

```python
# Drain aborts queue; all aborts are also processed via input_queue.
with self.aborts_queue.mutex:
    self.aborts_queue.queue.clear()
```

位置：`vllm/vllm/v1/engine/core.py:1274` 到 `vllm/vllm/v1/engine/core.py:1277`

---

## 13. shutdown 流程

普通 EngineCore 的资源释放在 `shutdown()`：

```python
def shutdown(self):
    logger.debug_once("[shutdown] EngineCore: tearing down local resources")
    self.structured_output_manager.clear_backend()
    if self.model_executor:
        self.model_executor.shutdown()
    if self.scheduler:
        self.scheduler.shutdown()
    gc.unfreeze()
    cleanup_dist_env_and_memory()
    logger.debug_once("[shutdown] EngineCore: local resource teardown complete")
```

位置：`vllm/vllm/v1/engine/core.py:644` 到 `vllm/vllm/v1/engine/core.py:660`

它释放：

```text
structured output backend；
model_executor / Worker；
Scheduler；
GC freeze 状态；
distributed environment / cached memory。
```

后台进程模式还有 shutdown state：

```python
class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2
```

位置：`vllm/vllm/v1/engine/core.py:888` 到 `vllm/vllm/v1/engine/core.py:891`

busy loop 每轮都会调用 `_handle_shutdown()`：

```python
while self._handle_shutdown():
    self._process_input_queue()
    self._process_engine_step()
```

位置：`vllm/vllm/v1/engine/core.py:1257` 到 `vllm/vllm/v1/engine/core.py:1263`

如果 shutdown 被请求，会根据 `shutdown_timeout` 决定 abort 还是 drain：

```python
shutdown_timeout = self.vllm_config.shutdown_timeout
mode = "abort" if shutdown_timeout == 0 else "drain"
```

位置：`vllm/vllm/v1/engine/core.py:1327` 到 `vllm/vllm/v1/engine/core.py:1329`

timeout 为 0 时，直接 abort 所有 in-flight 请求：

```python
aborted_reqs = self.scheduler.finish_requests(
    None, RequestStatus.FINISHED_ABORTED
)
self._send_abort_outputs(aborted_reqs)
```

位置：`vllm/vllm/v1/engine/core.py:1337` 到 `vllm/vllm/v1/engine/core.py:1347`

如果还有工作，就继续 loop；没有工作后退出：

```python
if not self.has_work():
    logger.info(
        "[shutdown] EngineCore: request processing complete; "
        "starting resource teardown"
    )
    return False
```

位置：`vllm/vllm/v1/engine/core.py:1360` 到 `vllm/vllm/v1/engine/core.py:1366`

---

## 14. 异常传播和 executor failure

EngineCoreProc 初始化时注册 executor failure callback：

```python
executor_fail_callback = lambda: self.input_queue.put_nowait(
    (EngineCoreRequestType.EXECUTOR_FAILED, b"")
)
```

位置：`vllm/vllm/v1/engine/core.py:924` 到 `vllm/vllm/v1/engine/core.py:928`

如果 executor 进入失败状态，会向 input_queue 放入 `EXECUTOR_FAILED`。

`_handle_client_request()` 收到后直接抛异常：

```python
elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
    raise RuntimeError("Executor failed.")
```

位置：`vllm/vllm/v1/engine/core.py:1398` 到 `vllm/vllm/v1/engine/core.py:1399`

utility 调用失败则不会直接当作普通请求输出，而是写入 `UtilityOutput.failure_message`：

```python
output.failure_message = f"Call to {name} method failed: {str(e)}"
```

位置：`vllm/vllm/v1/engine/core.py:1457` 到 `vllm/vllm/v1/engine/core.py:1460`

如果后台进程需要通知前端 EngineCore 已死，会发送特殊字节：

```python
self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)
```

位置：`vllm/vllm/v1/engine/core.py:1468` 到 `vllm/vllm/v1/engine/core.py:1472`

所以异常传播大致分为：

```text
Worker / executor permanent failure：
  executor_fail_callback → input_queue EXECUTOR_FAILED → RuntimeError。

utility 调用失败：
  UtilityOutput.failure_message → 前端等待的 future 抛异常。

EngineCoreProc 死亡：
  ENGINE_CORE_DEAD → 前端 EngineCoreClient 感知 EngineDeadError / 连接失效。
```

---

## 15. 输出线程和跨进程返回

后台 EngineCoreProc 的 step 结果先进 `output_queue`：

```python
for output in outputs.items() if outputs else ():
    self.output_queue.put_nowait(output)
```

位置：`vllm/vllm/v1/engine/core.py:1303` 到 `vllm/vllm/v1/engine/core.py:1305`

输出线程取出后设置 `engine_index` 并发给对应 client socket：

```python
client_index, outputs = output
outputs.engine_index = engine_index
...
tracker = sockets[client_index].send_multipart(
    buffers, copy=False, track=True
)
```

位置：`vllm/vllm/v1/engine/core.py:1626` 到 `vllm/vllm/v1/engine/core.py:1645`

如果 `client_index == -1`，输出会发给 DP coordinator socket，而不是普通前端 client：

```python
if client_index == -1:
    assert coord_socket is not None
    coord_socket.send_multipart(encoder.encode(outputs))
    continue
```

位置：`vllm/vllm/v1/engine/core.py:1629` 到 `vllm/vllm/v1/engine/core.py:1634`

所以跨进程返回路径是：

```text
EngineCore.step_fn()
  → dict[client_index, EngineCoreOutputs]
  → EngineCoreProc.output_queue
  → process_output_sockets thread
  → ZMQ socket for client_index
  → EngineCoreClient output queue
  → LLMEngine / AsyncLLM
```

这也是为什么前面文档里一直强调 `EngineCoreOutputs` 按 `client_index` 分组。

---

## 16. 总结

`EngineCore` 的生命周期和异步能力可以按层次理解：

```text
EngineCore 基础层：
  初始化 model_executor / KV cache / Scheduler；
  维护 step_fn、batch_queue、abort queue；
  转发 profile / reset / sleep / wakeup / LoRA 等 utility；
  shutdown 时释放 executor、scheduler、distributed state。

EngineCoreProc 后台层：
  增加 ZMQ handshakes、input/output socket threads、input_queue/output_queue；
  run_busy_loop() 中反复 process input + step engine；
  处理 ADD / ABORT / UTILITY / EXECUTOR_FAILED；
  通过 output_queue + ZMQ 返回 EngineCoreOutputs。

外层 Client 层：
  InprocClient 直接调用 EngineCore；
  SyncMPClient / AsyncMPClient 通过 ZMQ 与 EngineCoreProc 通信。
```

如果要回答：

```text
EngineCore 的异步、并发和生命周期管理是什么？
```

可以概括为：

```text
EngineCore 本身负责初始化执行环境、维护 Scheduler 和 model_executor、选择普通 step 或 batch queue step、处理 abort、转发 utility 调用并释放资源。
多进程场景下，EngineCoreProc 继承 EngineCore，并额外管理 ZMQ 输入输出线程、input_queue / output_queue、后台 run_busy_loop、client request 分发、executor failure 和 shutdown 状态。
同步模式下外层通过 InprocClient 直接调用 EngineCore；异步 / 多进程模式下外层通过 EngineCoreClient 与后台 EngineCoreProc 通信。
```
