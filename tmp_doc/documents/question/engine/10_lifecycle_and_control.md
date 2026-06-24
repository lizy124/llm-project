# 10. Engine 的 abort、profile、sleep、shutdown 如何接入？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/engine/core.py`

本问题关注：外层 Engine 除了 `add_request` / `step` 之外，如何接入取消请求、profile、reset、sleep、wake_up、pause、shutdown、LoRA、collective_rpc 等控制能力。

---

## 1. 一句话回答

外层 Engine 暴露控制类接口，但大部分真正动作不在外层完成，而是通过 `EngineCoreClient` 转发到 `EngineCore` / `EngineCoreProc`。

主线是：

```text
LLMEngine / AsyncLLM 控制 API
  → OutputProcessor 先处理外层请求状态（部分接口需要）
  → EngineCoreClient 同步或异步转发
  → InprocClient 直接调用 EngineCore
     或 MPClient 发送 UTILITY / ABORT 消息到 EngineCoreProc
  → EngineCore 操作 Scheduler / model_executor / cache / LoRA / distributed resources
```

一句话：

```text
外层 Engine 负责提供用户可调用的控制入口；
EngineCoreClient 负责屏蔽同进程、多进程、同步、异步差异；
EngineCore 负责真正操作 Scheduler、model_executor 和资源生命周期。
```

---

## 2. 控制接口分几类

从外层 Engine 看，控制类接口可以分成几类：

```text
请求控制：
  abort_request / abort
  notify_kv_transfer_request_rejected

运行状态控制：
  has_unfinished_requests
  get_num_unfinished_requests
  check_health
  is_running / is_stopped / errored

调度暂停控制：
  pause_generation
  resume_generation
  is_paused

资源控制：
  sleep
  wake_up
  is_sleeping
  reset_mm_cache
  reset_prefix_cache
  reset_encoder_cache

性能和观测：
  start_profile
  stop_profile
  get_metrics
  do_log_stats

模型扩展和执行器控制：
  add_lora / remove_lora / list_loras / pin_lora
  collective_rpc
  apply_model
  save_sharded_state

进程和后台任务生命周期：
  shutdown
  __del__
  EngineCoreClient.shutdown
  EngineCore.shutdown
```

核心模式是：

```text
外层 Engine 方法
  → EngineCoreClient 方法
  → EngineCore 方法
```

但也有一些外层额外逻辑，例如：

```text
abort：先清理 OutputProcessor；
reset_mm_cache / sleep(level >= 1)：先清理 renderer 的多模态 cache；
profile：AsyncLLM 还可能同时启动 Python profiler；
shutdown：AsyncLLM 还要关闭 renderer、EngineCoreClient、output_handler。
```

---

## 3. 同步 LLMEngine 的控制入口总览

同步 `LLMEngine` 的控制方法集中在 `llm_engine.py`。

### 3.1 请求状态

```python
def get_num_unfinished_requests(self) -> int:
    return self.output_processor.get_num_unfinished_requests()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:188` 到 `vllm/vllm/v1/engine/llm_engine.py:189`

```python
def has_unfinished_requests(self) -> bool:
    has_unfinished = self.output_processor.has_unfinished_requests()
    if self.dp_group is None:
        return has_unfinished or self.engine_core.dp_engines_running()
    return self.has_unfinished_requests_dp(has_unfinished)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:191` 到 `vllm/vllm/v1/engine/llm_engine.py:195`

同步 Engine 的 unfinished 主要看 `OutputProcessor` 是否还有外层请求状态；DP 场景还会考虑全局 engines 是否仍在运行。

### 3.2 abort

```python
def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
    """Remove request_ids from EngineCore and Detokenizer."""

    request_ids = self.output_processor.abort_requests(request_ids, internal)
    self.engine_core.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:212` 到 `vllm/vllm/v1/engine/llm_engine.py:216`

### 3.3 profile / reset / sleep / wake

```python
def start_profile(self, profile_prefix: str | None = None):
    self.engine_core.profile(True, profile_prefix)

def stop_profile(self):
    self.engine_core.profile(False)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:336` 到 `vllm/vllm/v1/engine/llm_engine.py:340`

```python
def reset_mm_cache(self):
    self.renderer.clear_mm_cache()
    self.engine_core.reset_mm_cache()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:342` 到 `vllm/vllm/v1/engine/llm_engine.py:344`

```python
def sleep(self, level: int = 1, mode: PauseMode = "abort"):
    if level >= 1:
        self.renderer.clear_mm_cache()
    self.engine_core.sleep(level, mode)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:361` 到 `vllm/vllm/v1/engine/llm_engine.py:364`

```python
def wake_up(self, tags: list[str] | None = None):
    self.engine_core.wake_up(tags)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:369` 到 `vllm/vllm/v1/engine/llm_engine.py:370`

### 3.4 LoRA / RPC

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    """Load a new LoRA adapter into the engine for future requests."""
    return self.engine_core.add_lora(lora_request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:403` 到 `vllm/vllm/v1/engine/llm_engine.py:405`

```python
def collective_rpc(...):
    return self.engine_core.collective_rpc(method, timeout, args, kwargs)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:419` 到 `vllm/vllm/v1/engine/llm_engine.py:426`

同步 `LLMEngine` 的控制接口大多是 thin wrapper。

---

## 4. 异步 AsyncLLM 的控制入口总览

`AsyncLLM` 的控制接口对应异步版本。

### 4.1 abort

```python
async def abort(
    self, request_id: str | Iterable[str], internal: bool = False
) -> None:
    """Abort RequestId in OutputProcessor and EngineCore."""

    request_ids = (
        (request_id,) if isinstance(request_id, str) else as_list(request_id)
    )
    all_request_ids = self.output_processor.abort_requests(request_ids, internal)
    await self.engine_core.abort_requests_async(all_request_ids)
```

位置：`vllm/vllm/v1/engine/async_llm.py:709` 到 `vllm/vllm/v1/engine/async_llm.py:718`

### 4.2 pause / resume

```python
async def pause_generation(
    self,
    *,
    mode: PauseMode = "abort",
    wait_for_inflight_requests: bool | None = None,
    clear_cache: bool = True,
) -> None:
```

位置：`vllm/vllm/v1/engine/async_llm.py:750` 到 `vllm/vllm/v1/engine/async_llm.py:756`

核心转发：

```python
if clear_cache:
    await self.renderer.clear_mm_cache_async()
await self.engine_core.pause_scheduler_async(mode=mode, clear_cache=clear_cache)
await asyncio.sleep(0.02)
```

位置：`vllm/vllm/v1/engine/async_llm.py:784` 到 `vllm/vllm/v1/engine/async_llm.py:793`

恢复和查询：

```python
async def resume_generation(self) -> None:
    """Resume generation after :meth:`pause_generation`."""
    await self.engine_core.resume_scheduler_async()

async def is_paused(self) -> bool:
    """Return whether the engine is currently paused."""
    return await self.engine_core.is_scheduler_paused_async()
```

位置：`vllm/vllm/v1/engine/async_llm.py:795` 到 `vllm/vllm/v1/engine/async_llm.py:801`

### 4.3 profile / reset / sleep / wake

```python
async def start_profile(self, profile_prefix: str | None = None) -> None:
    coros = [self.engine_core.profile_async(True, profile_prefix)]
    if self.profiler is not None:
        coros.append(asyncio.to_thread(self.profiler.start))
    await asyncio.gather(*coros)
```

位置：`vllm/vllm/v1/engine/async_llm.py:905` 到 `vllm/vllm/v1/engine/async_llm.py:909`

```python
async def reset_mm_cache(self) -> None:
    await self.renderer.clear_mm_cache_async()
    await self.engine_core.reset_mm_cache_async()
```

位置：`vllm/vllm/v1/engine/async_llm.py:917` 到 `vllm/vllm/v1/engine/async_llm.py:919`

```python
async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
    if level >= 1:
        await self.renderer.clear_mm_cache_async()
    await self.engine_core.sleep_async(level, mode)
```

位置：`vllm/vllm/v1/engine/async_llm.py:931` 到 `vllm/vllm/v1/engine/async_llm.py:934`

### 4.4 health / stopped 状态

```python
async def check_health(self) -> None:
    logger.debug("Called check_health.")
    if self.errored:
        raise self.dead_error
```

位置：`vllm/vllm/v1/engine/async_llm.py:900` 到 `vllm/vllm/v1/engine/async_llm.py:903`

```python
@property
def is_running(self) -> bool:
    # Is None before the loop is started.
    return self.output_handler is None or not self.output_handler.done()

@property
def is_stopped(self) -> bool:
    return self.errored

@property
def errored(self) -> bool:
    return self.engine_core.resources.engine_dead or not self.is_running
```

位置：`vllm/vllm/v1/engine/async_llm.py:1044` 到 `vllm/vllm/v1/engine/async_llm.py:1055`

---

## 5. abort 请求如何接入

abort 是最典型的跨层控制流。

### 5.1 同步 abort 主线

```text
LLMEngine.abort_request(request_ids, internal=False)
  → OutputProcessor.abort_requests(request_ids, internal)
  → 得到 internal request ids
  → EngineCoreClient.abort_requests(internal_ids)
  → EngineCore.abort_requests(internal_ids)
  → Scheduler.finish_requests(..., FINISHED_ABORTED)
```

同步外层入口：

```python
request_ids = self.output_processor.abort_requests(request_ids, internal)
self.engine_core.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:215` 到 `vllm/vllm/v1/engine/llm_engine.py:216`

### 5.2 异步 abort 主线

```text
AsyncLLM.abort(request_id, internal=False)
  → OutputProcessor.abort_requests()
  → EngineCoreClient.abort_requests_async()
  → EngineCoreProc / EngineCore.abort_requests()
  → Scheduler.finish_requests(..., FINISHED_ABORTED)
```

异步外层入口：

```python
all_request_ids = self.output_processor.abort_requests(request_ids, internal)
await self.engine_core.abort_requests_async(all_request_ids)
```

位置：`vllm/vllm/v1/engine/async_llm.py:717` 到 `vllm/vllm/v1/engine/async_llm.py:718`

### 5.3 为什么先经过 OutputProcessor

因为用户可能传的是 external request id。

`OutputProcessor.abort_requests()` 会处理：

```text
external id → internal ids；
parallel sampling parent id → child ids；
清理 RequestState；
异步 queue 中放入 abort final output。
```

然后才把内部 request ids 交给 EngineCore。

### 5.4 EngineCore.abort_requests()

EngineCore 内部入口是：

```python
def abort_requests(self, request_ids: list[str]):
    """Abort requests from the scheduler."""
    self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
```

位置：`vllm/vllm/v1/engine/core.py:409` 到 `vllm/vllm/v1/engine/core.py:415`

所以真正的请求状态迁移发生在 Scheduler：

```text
running / waiting / skipped_waiting
  → FINISHED_ABORTED
  → 释放或延迟释放资源
  → 返回 abort output 或清理状态
```

---

## 6. abort 在 EngineCoreClient 中如何转发

### 6.1 InprocClient

同进程模式直接调用 EngineCore：

```python
def abort_requests(self, request_ids: list[str]) -> None:
    if len(request_ids) > 0:
        self.engine_core.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/core_client.py:301` 到 `vllm/vllm/v1/engine/core_client.py:303`

### 6.2 SyncMPClient

多进程同步模式发送 `ABORT` 消息：

```python
def abort_requests(self, request_ids: list[str]) -> None:
    if request_ids and not self.resources.engine_dead:
        self._send_input(EngineCoreRequestType.ABORT, request_ids)
```

位置：`vllm/vllm/v1/engine/core_client.py:891` 到 `vllm/vllm/v1/engine/core_client.py:893`

### 6.3 AsyncMPClient

异步模式发送异步 `ABORT` 消息：

```python
async def abort_requests_async(self, request_ids: list[str]) -> None:
    if request_ids and not self.resources.engine_dead:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids)
```

位置：`vllm/vllm/v1/engine/core_client.py:1126` 到 `vllm/vllm/v1/engine/core_client.py:1128`

所以 abort 的 client 层差异是：

```text
Inproc：直接函数调用；
SyncMP：ZMQ ABORT；
AsyncMP：await ZMQ ABORT。
```

---

## 7. 执行期间 abort 的处理

多进程场景下，ABORT 进入 EngineCoreProc input socket 后，会同时进入两个队列：

```text
input_queue：
  保证和 ADD 等输入事件的顺序关系。

aborts_queue：
  允许模型 forward 期间更早处理 abort。
```

EngineCore step 在 Worker 输出回来后、`update_from_output()` 前调用：

```python
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:501` 到 `vllm/vllm/v1/engine/core.py:506`

`_process_aborts_queue()`：

```python
if not self.aborts_queue.empty():
    request_ids = []
    while not self.aborts_queue.empty():
        ids = self.aborts_queue.get_nowait()
        request_ids.extend((ids,) if isinstance(ids, str) else ids)
    self.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/core.py:634` 到 `vllm/vllm/v1/engine/core.py:642`

作用是：

```text
如果请求在模型执行期间被取消，
EngineCore 会先把 abort 应用到 Scheduler，
再处理 Worker 输出，避免继续给已取消请求 append token。
```

---

## 8. stop string 触发的内部 abort

除了用户主动 abort，还有一种由输出处理触发的 abort。

当 `OutputProcessor` detokenize 后发现 stop string，但 EngineCore 还没 finished：

```text
OutputProcessor.process_outputs()
  → reqs_to_abort.append(req_id)
```

同步 `LLMEngine.step()` 会：

```python
self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:316` 到 `vllm/vllm/v1/engine/llm_engine.py:318`

异步 output handler 会：

```python
if processed_outputs.reqs_to_abort:
    await engine_core.abort_requests_async(
        processed_outputs.reqs_to_abort
    )
```

位置：`vllm/vllm/v1/engine/async_llm.py:685` 到 `vllm/vllm/v1/engine/async_llm.py:689`

这条路径说明：

```text
输出层也可能产生控制信号，反向通知 EngineCore 停止请求。
```

---

## 9. notify_kv_transfer_request_rejected 特殊 abort_immediately 路径

异步 Engine 有一个特殊控制入口：

```python
async def notify_kv_transfer_request_rejected(...):
```

位置：`vllm/vllm/v1/engine/async_llm.py:723`

它会构造一个 `abort_immediately=True` 的 `EngineCoreRequest`：

```python
request = EngineCoreRequest(
    request_id=request_id,
    prompt_token_ids=[0],
    ...
    sampling_params=SamplingParams(
        max_tokens=1,
        extra_args={"kv_transfer_params": dict(kv_transfer_params)},
    ),
    ...
    abort_immediately=True,
)
await self.engine_core.add_request_async(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:733` 到 `vllm/vllm/v1/engine/async_llm.py:748`

注释说明：

```python
"""Submit a pre-aborted request so the connector's request_finished
hook runs to free any pre-admission KV-transfer resources ..."""
```

位置：`vllm/vllm/v1/engine/async_llm.py:730` 到 `vllm/vllm/v1/engine/async_llm.py:732`

EngineCore.add_request() 中看到 `abort_immediately` 会立即 abort：

```python
self.scheduler.add_request(request)
if request.abort_immediately:
    self.abort_requests([request.request_id])
```

位置：`vllm/vllm/v1/engine/core.py:403` 到 `vllm/vllm/v1/engine/core.py:407`

这类请求目的不是生成，而是：

```text
走标准 request_finished / connector cleanup 路径，
释放预接纳阶段占用的 KV transfer 资源。
```

---

## 10. profile 如何接入

### 10.1 同步 LLMEngine

同步 profile 入口：

```python
def start_profile(self, profile_prefix: str | None = None):
    self.engine_core.profile(True, profile_prefix)

def stop_profile(self):
    self.engine_core.profile(False)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:336` 到 `vllm/vllm/v1/engine/llm_engine.py:340`

### 10.2 异步 AsyncLLM

异步 profile 同时可能启动 EngineCore profile 和外层 profiler：

```python
async def start_profile(self, profile_prefix: str | None = None) -> None:
    coros = [self.engine_core.profile_async(True, profile_prefix)]
    if self.profiler is not None:
        coros.append(asyncio.to_thread(self.profiler.start))
    await asyncio.gather(*coros)
```

位置：`vllm/vllm/v1/engine/async_llm.py:905` 到 `vllm/vllm/v1/engine/async_llm.py:909`

停止类似：

```python
coros = [self.engine_core.profile_async(False)]
if self.profiler is not None:
    coros.append(asyncio.to_thread(self.profiler.stop))
await asyncio.gather(*coros)
```

位置：`vllm/vllm/v1/engine/async_llm.py:911` 到 `vllm/vllm/v1/engine/async_llm.py:915`

### 10.3 EngineCore.profile

EngineCore 内部只是转给 model executor：

```python
def profile(self, is_start: bool = True, profile_prefix: str | None = None):
    self.model_executor.profile(is_start, profile_prefix)
```

位置：`vllm/vllm/v1/engine/core.py:662` 到 `vllm/vllm/v1/engine/core.py:663`

所以 profile 链路是：

```text
LLMEngine.start_profile()
  → EngineCoreClient.profile()
  → EngineCore.profile()
  → model_executor.profile(True)
```

异步路径则多一层 `await profile_async()` 和可选外层 profiler。

---

## 11. reset_mm_cache 如何接入

同步路径：

```python
def reset_mm_cache(self):
    self.renderer.clear_mm_cache()
    self.engine_core.reset_mm_cache()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:342` 到 `vllm/vllm/v1/engine/llm_engine.py:344`

异步路径：

```python
async def reset_mm_cache(self) -> None:
    await self.renderer.clear_mm_cache_async()
    await self.engine_core.reset_mm_cache_async()
```

位置：`vllm/vllm/v1/engine/async_llm.py:917` 到 `vllm/vllm/v1/engine/async_llm.py:919`

EngineCore 内部：

```python
def reset_mm_cache(self):
    if self.scheduler.has_unfinished_requests():
        logger.warning(...)

    if self.mm_receiver_cache is not None:
        self.mm_receiver_cache.clear_cache()

    self.model_executor.reset_mm_cache()
```

位置：`vllm/vllm/v1/engine/core.py:665` 到 `vllm/vllm/v1/engine/core.py:678`

含义：

```text
外层 renderer 清理前端多模态 processor cache；
EngineCore 清理 receiver cache；
model_executor 清理 Worker / ModelRunner 侧 mm cache。
```

如果仍有未完成请求，会警告可能导致内部 cache 不同步。

---

## 12. reset_prefix_cache 如何接入

同步路径：

```python
def reset_prefix_cache(
    self, reset_running_requests: bool = False, reset_connector: bool = False
) -> bool:
    return self.engine_core.reset_prefix_cache(
        reset_running_requests, reset_connector
    )
```

位置：`vllm/vllm/v1/engine/llm_engine.py:346` 到 `vllm/vllm/v1/engine/llm_engine.py:351`

异步路径：

```python
async def reset_prefix_cache(
    self, reset_running_requests: bool = False, reset_connector: bool = False
) -> bool:
    return await self.engine_core.reset_prefix_cache_async(
        reset_running_requests, reset_connector
    )
```

位置：`vllm/vllm/v1/engine/async_llm.py:921` 到 `vllm/vllm/v1/engine/async_llm.py:926`

EngineCore 内部：

```python
def reset_prefix_cache(
    self, reset_running_requests: bool = False, reset_connector: bool = False
) -> bool:
    return self.scheduler.reset_prefix_cache(
        reset_running_requests, reset_connector
    )
```

位置：`vllm/vllm/v1/engine/core.py:680` 到 `vllm/vllm/v1/engine/core.py:685`

所以 prefix cache reset 主要由 Scheduler / KV cache manager 处理。

参数含义可以概括为：

```text
reset_running_requests：
  是否影响正在运行中的请求。

reset_connector：
  是否同时重置外部 KV connector 相关状态。
```

---

## 13. reset_encoder_cache 如何接入

同步路径：

```python
def reset_encoder_cache(self) -> None:
    """Reset the encoder cache to invalidate all cached encoder outputs.

    This should be called when model weights are updated to ensure
    stale vision embeddings computed with old weights are not reused.
    """
    self.engine_core.reset_encoder_cache()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:353` 到 `vllm/vllm/v1/engine/llm_engine.py:359`

异步路径：

```python
async def reset_encoder_cache(self) -> None:
    await self.engine_core.reset_encoder_cache_async()
```

位置：`vllm/vllm/v1/engine/async_llm.py:928` 到 `vllm/vllm/v1/engine/async_llm.py:929`

EngineCore 内部：

```python
def reset_encoder_cache(self) -> None:
    if self.scheduler.has_unfinished_requests():
        logger.warning(...)

    self.scheduler.reset_encoder_cache()
    self.model_executor.reset_encoder_cache()
```

位置：`vllm/vllm/v1/engine/core.py:687` 到 `vllm/vllm/v1/engine/core.py:705`

含义：

```text
scheduler.reset_encoder_cache()：
  清理逻辑 encoder cache 管理状态。

model_executor.reset_encoder_cache()：
  清理 GPU ModelRunner 侧物理 encoder cache。
```

---

## 14. _reset_caches：pause / sleep 中的组合重置

EngineCore 内部有组合重置方法：

```python
def _reset_caches(
    self,
    reset_running_requests: bool = True,
    reset_connector: bool = True,
) -> None:
    self.reset_prefix_cache(
        reset_running_requests=reset_running_requests,
        reset_connector=reset_connector,
    )
    self.reset_mm_cache()
    self.reset_encoder_cache()
```

位置：`vllm/vllm/v1/engine/core.py:707` 到 `vllm/vllm/v1/engine/core.py:720`

注释说明：

```text
reset_connector=True 时，外部 connector 会随本地 cache 一起清理，
符合 pause_generation(clear_cache=True) 的语义。
```

这个方法主要由 `pause_scheduler(clear_cache=True)` 和 sleep 相关路径使用。

---

## 15. pause_generation / pause_scheduler 如何接入

`pause_generation()` 是异步外层 API，`pause_scheduler()` 是 EngineCore 内部执行点。

### 15.1 AsyncLLM.pause_generation()

```python
async def pause_generation(
    self,
    *,
    mode: PauseMode = "abort",
    wait_for_inflight_requests: bool | None = None,
    clear_cache: bool = True,
) -> None:
```

位置：`vllm/vllm/v1/engine/async_llm.py:750` 到 `vllm/vllm/v1/engine/async_llm.py:756`

注释说明三种 mode：

```text
abort：立即 abort 所有 in-flight requests。
wait：等待 in-flight requests 完成。
keep：冻结队列中的请求，resume 后继续。
```

外层逻辑：

```python
if wait_for_inflight_requests:
    ...
    mode = "wait"
if clear_cache:
    await self.renderer.clear_mm_cache_async()
await self.engine_core.pause_scheduler_async(mode=mode, clear_cache=clear_cache)
await asyncio.sleep(0.02)
```

位置：`vllm/vllm/v1/engine/async_llm.py:775` 到 `vllm/vllm/v1/engine/async_llm.py:793`

### 15.2 EngineCore.pause_scheduler() 同进程版本

基础 EngineCore 中：

```python
def pause_scheduler(
    self, mode: PauseMode = "abort", clear_cache: bool = True
) -> Future | None:
```

位置：`vllm/vllm/v1/engine/core.py:722` 到 `vllm/vllm/v1/engine/core.py:724`

同进程版本不支持 wait：

```python
if mode == "wait":
    raise ValueError("'wait' mode can't be used in inproc-engine mode")
```

位置：`vllm/vllm/v1/engine/core.py:740` 到 `vllm/vllm/v1/engine/core.py:741`

abort 模式会结束所有请求：

```python
if mode == "abort":
    self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
```

位置：`vllm/vllm/v1/engine/core.py:743` 到 `vllm/vllm/v1/engine/core.py:744`

然后设置 pause state：

```python
pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
self.scheduler.set_pause_state(pause_state)
if clear_cache:
    self._reset_caches()
```

位置：`vllm/vllm/v1/engine/core.py:746` 到 `vllm/vllm/v1/engine/core.py:749`

### 15.3 EngineCoreProc.pause_scheduler() 多进程版本

多进程 `EngineCoreProc` 覆盖了 `pause_scheduler()`，支持 wait / 异步完成：

```python
def pause_scheduler(
    self, mode: PauseMode = "abort", clear_cache: bool = True
) -> Future | None:
```

位置：`vllm/vllm/v1/engine/core.py:1663` 到 `vllm/vllm/v1/engine/core.py:1665`

注释概括：

```text
abort：设置 PAUSED_NEW，abort 所有请求，等待 abort outputs 发送后完成；
wait：设置 PAUSED_NEW，继续 step，让 in-flight 请求 drain；
keep：设置 PAUSED_ALL，冻结请求，等 output queue 空。
```

位置：`vllm/vllm/v1/engine/core.py:1666` 到 `vllm/vllm/v1/engine/core.py:1677`

abort 模式会发送 abort outputs：

```python
if mode == "abort":
    aborted_reqs = self.scheduler.finish_requests(
        None, RequestStatus.FINISHED_ABORTED
    )
    self._send_abort_outputs(aborted_reqs)
```

位置：`vllm/vllm/v1/engine/core.py:1687` 到 `vllm/vllm/v1/engine/core.py:1691`

如果当前已经 pause complete，则可立即 reset cache 并返回：

```python
if self._pause_complete():
    if clear_cache:
        self._reset_caches()
    return None
```

位置：`vllm/vllm/v1/engine/core.py:1696` 到 `vllm/vllm/v1/engine/core.py:1699`

否则注册 idle callback，等 EngineCore 空闲后再清 cache、resolve Future：

```python
future = Future[Any]()
self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
return future
```

位置：`vllm/vllm/v1/engine/core.py:1701` 到 `vllm/vllm/v1/engine/core.py:1703`

---

## 16. pause_state 的语义

`pause_scheduler()` 最终设置 Scheduler 的 pause state：

```python
pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
self.scheduler.set_pause_state(pause_state)
```

位置：`vllm/vllm/v1/engine/core.py:746` 到 `vllm/vllm/v1/engine/core.py:747`，`vllm/vllm/v1/engine/core.py:1693` 到 `vllm/vllm/v1/engine/core.py:1694`

可以理解为：

| mode | pause_state | 行为 |
|---|---|---|
| `abort` | `PAUSED_NEW` | 不接纳新调度，abort 已有请求 |
| `wait` | `PAUSED_NEW` | 不接纳新请求调度，但允许已有请求 drain |
| `keep` | `PAUSED_ALL` | 所有请求冻结，resume 后继续 |

其中：

```text
PAUSED_NEW：
  running 可继续推进，waiting 不再进入 running。

PAUSED_ALL：
  Scheduler 不继续调度请求。
```

---

## 17. resume_generation / resume_scheduler 如何接入

异步外层恢复：

```python
async def resume_generation(self) -> None:
    """Resume generation after :meth:`pause_generation`."""
    await self.engine_core.resume_scheduler_async()
```

位置：`vllm/vllm/v1/engine/async_llm.py:795` 到 `vllm/vllm/v1/engine/async_llm.py:797`

EngineCore 恢复：

```python
def resume_scheduler(self) -> None:
    """Resume the scheduler and flush any requests queued while paused."""
    self.scheduler.set_pause_state(PauseState.UNPAUSED)
```

位置：`vllm/vllm/v1/engine/core.py:753` 到 `vllm/vllm/v1/engine/core.py:755`

查询 pause：

```python
def is_scheduler_paused(self) -> bool:
    """Return whether the scheduler is in any pause state."""
    return self.scheduler.pause_state != PauseState.UNPAUSED
```

位置：`vllm/vllm/v1/engine/core.py:757` 到 `vllm/vllm/v1/engine/core.py:759`

---

## 18. sleep 如何接入

sleep 是更强的资源控制接口，它先 pause scheduler，再按 level 处理 GPU 资源。

### 18.1 外层 sleep

同步：

```python
def sleep(self, level: int = 1, mode: PauseMode = "abort"):
    if level >= 1:
        self.renderer.clear_mm_cache()
    self.engine_core.sleep(level, mode)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:361` 到 `vllm/vllm/v1/engine/llm_engine.py:364`

异步：

```python
async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
    if level >= 1:
        await self.renderer.clear_mm_cache_async()
    await self.engine_core.sleep_async(level, mode)
```

位置：`vllm/vllm/v1/engine/async_llm.py:931` 到 `vllm/vllm/v1/engine/async_llm.py:934`

外层会记录 sleep state：

```python
self.logger_manager.record_sleep_state(1, level)
```

同步位置：`vllm/vllm/v1/engine/llm_engine.py:366` 到 `vllm/vllm/v1/engine/llm_engine.py:367`

异步位置：`vllm/vllm/v1/engine/async_llm.py:936` 到 `vllm/vllm/v1/engine/async_llm.py:937`

### 18.2 EngineCore.sleep()

EngineCore sleep 入口：

```python
def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
    """Put the engine to sleep at the specified level.
```

位置：`vllm/vllm/v1/engine/core.py:761` 到 `vllm/vllm/v1/engine/core.py:762`

注释说明 level：

```text
Level 0：只暂停 scheduling，不改变 GPU memory。
Level 1：offload model weights 到 CPU，丢弃 KV cache。
Level 2：丢弃所有 GPU memory。
```

位置：`vllm/vllm/v1/engine/core.py:765` 到 `vllm/vllm/v1/engine/core.py:770`

核心流程：

```python
clear_prefix_cache = level >= 1
pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
if level < 1:
    return pause_future

model_executor = self.model_executor
if pause_future is None:
    model_executor.sleep(level)
    return None
```

位置：`vllm/vllm/v1/engine/core.py:774` 到 `vllm/vllm/v1/engine/core.py:784`

如果 pause 需要异步等待，则 callback 中执行 executor sleep：

```python
def pause_complete(f: Future):
    try:
        f.result()
        future.set_result(model_executor.sleep(level))
    except Exception as e:
        future.set_exception(e)
...
pause_future.add_done_callback(pause_complete)
return future
```

位置：`vllm/vllm/v1/engine/core.py:786` 到 `vllm/vllm/v1/engine/core.py:797`

因此 sleep 的关键公式是：

```text
sleep(level)
  → pause_scheduler(mode, clear_cache=(level >= 1))
  → level 0: 只暂停调度
  → level >= 1: model_executor.sleep(level)
```

---

## 19. wake_up 如何接入

外层同步：

```python
def wake_up(self, tags: list[str] | None = None):
    self.engine_core.wake_up(tags)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:369` 到 `vllm/vllm/v1/engine/llm_engine.py:370`

外层异步：

```python
async def wake_up(self, tags: list[str] | None = None) -> None:
    await self.engine_core.wake_up_async(tags)
```

位置：`vllm/vllm/v1/engine/async_llm.py:939` 到 `vllm/vllm/v1/engine/async_llm.py:940`

EngineCore 内部：

```python
def wake_up(self, tags: list[str] | None = None):
    """Wake up the engine from sleep.

    Args:
        tags: Tags to wake up. Use ["scheduling"] for level 0 wake up.
    """
    if tags is not None and "scheduling" in tags:
        tags = [t for t in tags if t != "scheduling"]

    if tags is None or tags:
        self.model_executor.wake_up(tags)

    # Resume scheduling (applies to all levels)
    self.resume_scheduler()
```

位置：`vllm/vllm/v1/engine/core.py:799` 到 `vllm/vllm/v1/engine/core.py:813`

含义：

```text
如果 tags 包含 scheduling，只恢复调度；
如果 tags 还有其它资源 tag，则唤醒 model_executor 对应资源；
最后总是 resume_scheduler()。
```

`is_sleeping()`：

```python
def is_sleeping(self) -> bool:
    """Check if engine is sleeping at any level."""
    return self.is_scheduler_paused() or self.model_executor.is_sleeping
```

位置：`vllm/vllm/v1/engine/core.py:815` 到 `vllm/vllm/v1/engine/core.py:817`

---

## 20. control API 在 EngineCoreClient 中如何转发

### 20.1 InprocClient：直接调用 EngineCore

```python
def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
    self.engine_core.profile(is_start, profile_prefix)

def reset_mm_cache(self) -> None:
    self.engine_core.reset_mm_cache()

def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
    if mode == "wait":
        raise ValueError("'wait' pause mode is not supported in inproc-engine mode")
    result = self.engine_core.sleep(level, mode)
    assert result is None
```

位置：`vllm/vllm/v1/engine/core_client.py:308` 到 `vllm/vllm/v1/engine/core_client.py:328`

同进程模式下所有控制方法基本都是直接函数调用。

### 20.2 SyncMPClient：UTILITY 消息

同步 MP 通过 `call_utility()`：

```python
def call_utility(self, method: str, *args) -> Any:
    call_id = uuid.uuid1().int >> 64
    future: Future[Any] = Future()
    self.utility_results[call_id] = future
    self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

    return future.result()
```

位置：`vllm/vllm/v1/engine/core_client.py:875` 到 `vllm/vllm/v1/engine/core_client.py:881`

例如：

```python
def profile(...):
    self.call_utility("profile", is_start, profile_prefix)

def reset_prefix_cache(...):
    return self.call_utility("reset_prefix_cache", reset_running_requests, reset_connector)

def sleep(...):
    self.call_utility("sleep", level, mode)
```

位置：`vllm/vllm/v1/engine/core_client.py:895` 到 `vllm/vllm/v1/engine/core_client.py:930`

### 20.3 AsyncMPClient：异步 UTILITY 消息

异步 utility：

```python
async def _call_utility_async(
    self, method: str, *args, engine: EngineIdentity
) -> Any:
    call_id = uuid.uuid1().int >> 64
    future = asyncio.get_running_loop().create_future()
    self.utility_results[call_id] = future
    message = (
        EngineCoreRequestType.UTILITY.value,
        *self.encoder.encode((self.client_index, call_id, method, args)),
    )
    await self._send_input_message(message, engine, args)
    self._ensure_output_queue_task()
    return await future
```

位置：`vllm/vllm/v1/engine/core_client.py:1104` 到 `vllm/vllm/v1/engine/core_client.py:1116`

对应控制方法：

```python
async def pause_scheduler_async(...):
    await self.call_utility_async("pause_scheduler", mode, clear_cache)

async def resume_scheduler_async(self) -> None:
    await self.call_utility_async("resume_scheduler")

async def sleep_async(...):
    await self.call_utility_async("sleep", level, mode)
```

位置：`vllm/vllm/v1/engine/core_client.py:1130` 到 `vllm/vllm/v1/engine/core_client.py:1166`

所以 MP 控制接口统一模式是：

```text
frontend method
  → UTILITY(client_idx, call_id, method_name, args)
  → EngineCoreProc._handle_client_request(UTILITY)
  → getattr(self, method_name)(*args)
  → UtilityOutput(call_id, result)
  → Future.set_result()
```

---

## 21. LoRA 和 collective_rpc 如何接入

LoRA 控制接口外层只是转发。

同步：

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.engine_core.add_lora(lora_request)

def remove_lora(self, lora_id: int) -> bool:
    return self.engine_core.remove_lora(lora_id)

def list_loras(self) -> set[int]:
    return self.engine_core.list_loras()

def pin_lora(self, lora_id: int) -> bool:
    return self.engine_core.pin_lora(lora_id)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:403` 到 `vllm/vllm/v1/engine/llm_engine.py:417`

异步：

```python
async def add_lora(self, lora_request: LoRARequest) -> bool:
    return await self.engine_core.add_lora_async(lora_request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:948` 到 `vllm/vllm/v1/engine/async_llm.py:950`

EngineCore 内部转给 model executor：

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.model_executor.add_lora(lora_request)

def remove_lora(self, lora_id: int) -> bool:
    return self.model_executor.remove_lora(lora_id)

def list_loras(self) -> set[int]:
    return self.model_executor.list_loras()

def pin_lora(self, lora_id: int) -> bool:
    return self.model_executor.pin_lora(lora_id)
```

位置：`vllm/vllm/v1/engine/core.py:822` 到 `vllm/vllm/v1/engine/core.py:832`

collective_rpc 也类似：

```python
def collective_rpc(...):
    return self.model_executor.collective_rpc(method, timeout, args, kwargs)
```

位置：`vllm/vllm/v1/engine/core.py:844` 到 `vllm/vllm/v1/engine/core.py:851`

所以这类控制接口的最终执行者是 `model_executor`。

---

## 22. get_supported_tasks 和缓存

同步 `LLMEngine`：

```python
def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
    if not hasattr(self, "_supported_tasks"):
        # Cache the result
        self._supported_tasks = self.engine_core.get_supported_tasks()

    return self._supported_tasks
```

位置：`vllm/vllm/v1/engine/llm_engine.py:205` 到 `vllm/vllm/v1/engine/llm_engine.py:210`

异步 `AsyncLLM`：

```python
async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
    if not hasattr(self, "_supported_tasks"):
        # Cache the result
        self._supported_tasks = await self.engine_core.get_supported_tasks_async()

    return self._supported_tasks
```

位置：`vllm/vllm/v1/engine/async_llm.py:273` 到 `vllm/vllm/v1/engine/async_llm.py:278`

这说明 supported tasks 是从 EngineCore 查询后缓存在外层 Engine 的。

---

## 23. has_unfinished_requests 和 DP dummy batch

同步 `LLMEngine.has_unfinished_requests()`：

```python
has_unfinished = self.output_processor.has_unfinished_requests()
if self.dp_group is None:
    return has_unfinished or self.engine_core.dp_engines_running()
return self.has_unfinished_requests_dp(has_unfinished)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:191` 到 `vllm/vllm/v1/engine/llm_engine.py:195`

DP 场景：

```python
aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
    self.dp_group, has_unfinished
)
if not has_unfinished and aggregated_has_unfinished:
    self.should_execute_dummy_batch = True
return aggregated_has_unfinished
```

位置：`vllm/vllm/v1/engine/llm_engine.py:197` 到 `vllm/vllm/v1/engine/llm_engine.py:203`

这说明：

```text
本地没有 unfinished 请求，但 DP 全局仍有其它 rank 未完成时，
当前 rank 可能需要执行 dummy batch 来维持 collective / DP 同步。
```

对应 `LLMEngine.step()` 开头：

```python
if self.should_execute_dummy_batch:
    self.should_execute_dummy_batch = False
    self.engine_core.execute_dummy_batch()
    return []
```

位置：`vllm/vllm/v1/engine/llm_engine.py:297` 到 `vllm/vllm/v1/engine/llm_engine.py:300`

---

## 24. shutdown 如何接入：AsyncLLM 外层

`AsyncLLM` 有显式 shutdown：

```python
def shutdown(self, timeout: float | None = None) -> None:
    """Shutdown, cleaning up the background proc and IPC."""
    shutdown_prometheus()

    if renderer := getattr(self, "renderer", None):
        renderer.shutdown()

    if engine_core := getattr(self, "engine_core", None):
        engine_core.shutdown(timeout=timeout)

    handler = getattr(self, "output_handler", None)
    if handler is not None:
        cancel_task_threadsafe(handler)
```

位置：`vllm/vllm/v1/engine/async_llm.py:259` 到 `vllm/vllm/v1/engine/async_llm.py:271`

析构时也会调用：

```python
def __del__(self):
    self.shutdown()
```

位置：`vllm/vllm/v1/engine/async_llm.py:256` 到 `vllm/vllm/v1/engine/async_llm.py:257`

因此 AsyncLLM shutdown 会处理：

```text
Prometheus shutdown；
renderer shutdown；
EngineCoreClient shutdown；
output_handler task cancel。
```

---

## 25. shutdown 如何接入：EngineCoreClient

### 25.1 InprocClient.shutdown()

同进程：

```python
def shutdown(self, timeout: float | None = None) -> None:
    self.engine_core.shutdown()
```

位置：`vllm/vllm/v1/engine/core_client.py:305` 到 `vllm/vllm/v1/engine/core_client.py:306`

### 25.2 MPClient.shutdown()

多进程：

```python
def shutdown(self, timeout: float | None = None) -> None:
    """Shutdown engine manager under timeout and clean up resources."""
    if self._finalizer.detach() is not None:
        ...
        if self.resources.engine_manager is not None:
            self.resources.engine_manager.shutdown(timeout=timeout)
        self.resources()
```

位置：`vllm/vllm/v1/engine/core_client.py:651` 到 `vllm/vllm/v1/engine/core_client.py:662`

`BackgroundResources.__call__()` 会：

```text
engine_dead = True；
shutdown engine_manager；
shutdown coordinator；
关闭 socket；
取消 async output / stats tasks；
同步模式下通知 output thread 退出。
```

核心位置：`vllm/vllm/v1/engine/core_client.py:392` 到 `vllm/vllm/v1/engine/core_client.py:452`

所以多进程 shutdown 不只是关闭 EngineCore，还要关闭：

```text
后台进程 / actor；
ZMQ input / output sockets；
output queue task / thread；
DP coordinator；
stats update socket/task。
```

---

## 26. shutdown 如何接入：EngineCore 内部

EngineCore 内部 shutdown：

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
model_executor / worker 资源；
Scheduler 资源；
startup 阶段 freeze 的 GC 对象；
distributed process group / GPU memory cache 等环境资源。
```

DP EngineCoreProc 还会额外销毁 DP group：

```python
def shutdown(self):
    super().shutdown()
    if dp_group := getattr(self, "dp_group", None):
        stateless_destroy_torch_distributed_process_group(dp_group)
```

位置：`vllm/vllm/v1/engine/core.py:1811` 到 `vllm/vllm/v1/engine/core.py:1814`

---

## 27. EngineCoreProc 的 shutdown 状态机

后台 `EngineCoreProc` 有 shutdown 状态：

```python
class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2
```

位置：`vllm/vllm/v1/engine/core.py:888` 到 `vllm/vllm/v1/engine/core.py:891`

busy loop 中 `_handle_shutdown()` 会根据状态处理：

```text
RUNNING：继续运行；
REQUESTED：进入 abort 或 drain 模式；
SHUTTING_DOWN：等 has_work() 为 False 后退出。
```

在 shutdown timeout 为 0 时，会 abort 所有 in-flight 请求：

```python
aborted_reqs = self.scheduler.finish_requests(
    None, RequestStatus.FINISHED_ABORTED
)
self._send_abort_outputs(aborted_reqs)
```

位置：`vllm/vllm/v1/engine/core.py:1344` 到 `vllm/vllm/v1/engine/core.py:1347`

当没有 work 时退出：

```python
if not self.has_work():
    logger.info(...)
    return False
```

位置：`vllm/vllm/v1/engine/core.py:1360` 到 `vllm/vllm/v1/engine/core.py:1366`

这说明后台进程 shutdown 不是简单 kill，而是可以：

```text
abort in-flight；
或 drain in-flight；
等待 output / cleanup 完成；
再释放资源。
```

---

## 28. 异常和 health 状态如何传播

多进程 client 用 `engine_dead` 标记后台死亡。

`BackgroundResources.validate_alive()`：

```python
def validate_alive(self, frames: Sequence[zmq.Frame]):
    if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
        self.engine_dead = True
        raise EngineDeadError()
```

位置：`vllm/vllm/v1/engine/core_client.py:454` 到 `vllm/vllm/v1/engine/core_client.py:457`

发送前检查：

```python
def ensure_alive(self):
    if self.resources.engine_dead:
        raise EngineDeadError()
```

位置：`vllm/vllm/v1/engine/core_client.py:670` 到 `vllm/vllm/v1/engine/core_client.py:672`

AsyncLLM health：

```python
async def check_health(self) -> None:
    if self.errored:
        raise self.dead_error
```

位置：`vllm/vllm/v1/engine/async_llm.py:900` 到 `vllm/vllm/v1/engine/async_llm.py:903`

`errored` 判定：

```python
return self.engine_core.resources.engine_dead or not self.is_running
```

位置：`vllm/vllm/v1/engine/async_llm.py:1053` 到 `vllm/vllm/v1/engine/async_llm.py:1055`

所以异常传播是：

```text
EngineCoreProc dead / output task failed
  → EngineCoreClient.resources.engine_dead 或 output handler done
  → AsyncLLM.errored
  → check_health() 抛 EngineDeadError
```

---

## 29. scale_elastic_ep 控制入口

异步 Engine 还支持 elastic EP scaling。

外层入口：

```python
async def scale_elastic_ep(
    self, new_data_parallel_size: int, drain_timeout: int = 300
):
```

位置：`vllm/vllm/v1/engine/async_llm.py:994` 到 `vllm/vllm/v1/engine/async_llm.py:996`

如果设置了 drain，会先等待请求 drain：

```python
if envs.VLLM_ELASTIC_EP_DRAIN_REQUESTS:
    await self.wait_for_requests_to_drain(drain_timeout)
```

位置：`vllm/vllm/v1/engine/async_llm.py:1013` 到 `vllm/vllm/v1/engine/async_llm.py:1018`

然后调用 client：

```python
await self.engine_core.scale_elastic_ep(new_data_parallel_size)
self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
```

位置：`vllm/vllm/v1/engine/async_llm.py:1037` 到 `vllm/vllm/v1/engine/async_llm.py:1040`

`DPLBAsyncMPClient.scale_elastic_ep()` 负责真正扩缩 engine cores：

```python
async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
```

位置：`vllm/vllm/v1/engine/core_client.py:1545`

这类接口属于多进程 / DP 高级生命周期控制。

---

## 30. 完整流程图：abort

```text
用户调用 abort
  │
  ├─ LLMEngine.abort_request() / AsyncLLM.abort()
  │    ├─ OutputProcessor.abort_requests()
  │    │    ├─ external id → internal ids
  │    │    ├─ parent id → child ids
  │    │    ├─ 清理 RequestState
  │    │    └─ async queue 放 abort output（如有）
  │    │
  │    └─ EngineCoreClient.abort_requests(_async)
  │         ├─ InprocClient: EngineCore.abort_requests()
  │         └─ MPClient: ZMQ ABORT → EngineCoreProc
  │
  └─ EngineCore.abort_requests()
       └─ Scheduler.finish_requests(..., FINISHED_ABORTED)
```

---

## 31. 完整流程图：pause / sleep / wake

```text
AsyncLLM.pause_generation(mode, clear_cache)
  ├─ renderer.clear_mm_cache_async()  # clear_cache=True
  └─ EngineCoreClient.pause_scheduler_async()
       └─ UTILITY pause_scheduler(mode, clear_cache)
            └─ EngineCoreProc.pause_scheduler()
                 ├─ mode=abort: Scheduler.finish_requests(..., ABORTED)
                 ├─ set_pause_state(PAUSED_NEW / PAUSED_ALL)
                 ├─ wait/drain if needed
                 └─ clear caches when complete
```

```text
sleep(level, mode)
  ├─ 外层 level >= 1: renderer.clear_mm_cache()
  └─ EngineCore.sleep(level, mode)
       ├─ pause_scheduler(mode, clear_cache=(level >= 1))
       ├─ level 0: only pause scheduling
       └─ level >= 1: model_executor.sleep(level)
```

```text
wake_up(tags)
  └─ EngineCore.wake_up(tags)
       ├─ model_executor.wake_up(tags)  # if needed
       └─ resume_scheduler()
            └─ Scheduler.pause_state = UNPAUSED
```

---

## 32. 完整流程图：profile / reset / LoRA

```text
profile
  → LLMEngine.start_profile() / AsyncLLM.start_profile()
  → EngineCoreClient.profile(_async)
  → EngineCore.profile()
  → model_executor.profile()
```

```text
reset_mm_cache
  → outer renderer.clear_mm_cache()
  → EngineCore.reset_mm_cache()
       ├─ mm_receiver_cache.clear_cache()
       └─ model_executor.reset_mm_cache()
```

```text
reset_prefix_cache
  → EngineCore.reset_prefix_cache()
  → Scheduler.reset_prefix_cache()
```

```text
reset_encoder_cache
  → EngineCore.reset_encoder_cache()
       ├─ Scheduler.reset_encoder_cache()
       └─ model_executor.reset_encoder_cache()
```

```text
LoRA
  → Engine.add_lora / remove_lora / list_loras / pin_lora
  → EngineCoreClient utility
  → EngineCore
  → model_executor
```

---

## 33. 完整流程图：shutdown

```text
AsyncLLM.shutdown(timeout)
  ├─ shutdown_prometheus()
  ├─ renderer.shutdown()
  ├─ EngineCoreClient.shutdown(timeout)
  │    ├─ InprocClient:
  │    │    └─ EngineCore.shutdown()
  │    │         ├─ structured_output_manager.clear_backend()
  │    │         ├─ model_executor.shutdown()
  │    │         ├─ scheduler.shutdown()
  │    │         ├─ gc.unfreeze()
  │    │         └─ cleanup_dist_env_and_memory()
  │    │
  │    └─ MPClient:
  │         ├─ engine_manager.shutdown(timeout)
  │         ├─ coordinator.shutdown()
  │         ├─ close sockets
  │         ├─ cancel output / stats tasks
  │         └─ mark engine_dead=True
  │
  └─ cancel output_handler task
```

后台 EngineCoreProc：

```text
shutdown requested
  → _handle_shutdown()
  → abort or drain in-flight requests
  → wait until not has_work()
  → EngineCore.shutdown()
  → send ENGINE_CORE_DEAD if needed
  → process exits
```

---

## 34. 容易疑惑的点

### 34.1 外层 Engine 是否直接操作 Scheduler？

通常不直接操作。

外层通过：

```text
EngineCoreClient
  → EngineCore
  → Scheduler
```

只有 OutputProcessor 会先处理外层请求状态，例如 abort。

### 34.2 abort 为什么要先经过 OutputProcessor？

因为用户传入的可能是 external request id，而 Scheduler 使用的是 internal request id。

OutputProcessor 负责：

```text
external id → internal ids；
parent id → child ids；
清理外层 RequestState；
异步路径生成 abort output。
```

### 34.3 pause 和 sleep 有什么区别？

```text
pause：
  控制 Scheduler 是否继续调度请求，可选择 abort / wait / keep。

sleep：
  先 pause，再根据 level 释放或 offload GPU 资源。
```

### 34.4 sleep level 0 和 level 1/2 的区别？

```text
level 0：
  只暂停 scheduling，不改变 GPU memory。

level 1：
  offload model weights 到 CPU，丢弃 KV cache。

level 2：
  丢弃所有 GPU memory。
```

### 34.5 为什么 inproc 不支持 wait pause mode？

基础 `EngineCore.pause_scheduler()` 中明确拒绝：

```python
if mode == "wait":
    raise ValueError("'wait' mode can't be used in inproc-engine mode")
```

位置：`vllm/vllm/v1/engine/core.py:740` 到 `vllm/vllm/v1/engine/core.py:741`

多进程 `EngineCoreProc` 有 busy loop 和 idle callback，因此可以支持 wait / Future 完成。

### 34.6 reset cache 是否安全？

如果有 unfinished requests，`reset_mm_cache()` 和 `reset_encoder_cache()` 会警告，因为可能造成内部 cache 不同步。

所以这类接口更适合：

```text
调试；
模型权重更新后；
pause / drain 后；
确认没有 in-flight requests 时。
```

### 34.7 shutdown 是否等价于 abort？

不是。

shutdown 是进程 / executor / scheduler / socket / background task 生命周期清理。

它可能先 abort 或 drain 请求，但目标是释放整个 Engine 资源。

### 34.8 EngineCoreClient 的 utility 调用是什么？

多进程模式下，profile / reset / sleep / LoRA 等不是普通 Python 调用，而是：

```text
UTILITY(method_name, args)
  → EngineCoreProc._handle_client_request()
  → getattr(self, method_name)(*args)
  → UtilityOutput(call_id)
```

### 34.9 check_health 看什么？

AsyncLLM 的 health 看：

```text
engine_core.resources.engine_dead
或 output_handler 是否已经结束。
```

任何一个异常都会让 `check_health()` 抛 `EngineDeadError`。

### 34.10 OutputProcessor 清理和 Scheduler 清理是一回事吗？

不是。

```text
OutputProcessor.abort_requests() / _finish_request()
  → 清理外层输出状态。

Scheduler.finish_requests() / _free_request()
  → 清理内部调度状态和 KV / encoder 资源。
```

abort / finish 路径通常要两边都处理。

---

## 35. 从“回答问题”的角度总结

如果要问：

```text
Engine 的 abort、profile、sleep、shutdown 如何接入？
```

可以回答：

```text
外层 LLMEngine / AsyncLLM 暴露控制类 API，但真正执行大多通过 EngineCoreClient 转发到 EngineCore。

abort 比较特殊，会先经过 OutputProcessor，把用户 external request id 转成内部 request id，并清理外层输出状态；然后再调用 EngineCore.abort_requests()，最终由 Scheduler.finish_requests(..., FINISHED_ABORTED) 修改内部请求状态。

profile、reset、LoRA、collective_rpc 等接口通常直接转发给 EngineCore，EngineCore 再转给 model_executor 或 Scheduler。

pause / sleep / wake_up 属于资源和调度控制：pause 设置 Scheduler pause_state；sleep 先 pause，再按 level 释放或 offload GPU 资源；wake_up 唤醒 model_executor 并恢复 Scheduler。

shutdown 则负责释放完整生命周期资源：AsyncLLM 关闭 renderer、EngineCoreClient 和 output_handler；EngineCoreClient 关闭后台进程、socket 和任务；EngineCore 关闭 structured output backend、model_executor、Scheduler 和 distributed / GPU 相关资源。
```

核心公式是：

```text
请求控制：
  Engine abort
    → OutputProcessor
    → EngineCoreClient
    → EngineCore
    → Scheduler

资源控制：
  Engine profile/reset/sleep/wake
    → EngineCoreClient
    → EngineCore
    → Scheduler / model_executor

生命周期：
  Engine shutdown
    → EngineCoreClient.shutdown
    → EngineCore.shutdown 或 EngineCoreProc shutdown
    → executor / scheduler / sockets / tasks cleanup
```

---

## 36. 最关键流程图

```text
控制入口总览

LLMEngine / AsyncLLM
  ├─ abort
  │    ├─ OutputProcessor.abort_requests()
  │    └─ EngineCoreClient.abort_requests(_async)
  │         └─ EngineCore.abort_requests()
  │              └─ Scheduler.finish_requests(..., FINISHED_ABORTED)
  │
  ├─ profile
  │    └─ EngineCoreClient.profile(_async)
  │         └─ EngineCore.profile()
  │              └─ model_executor.profile()
  │
  ├─ reset_mm_cache
  │    ├─ renderer.clear_mm_cache()
  │    └─ EngineCore.reset_mm_cache()
  │         └─ model_executor.reset_mm_cache()
  │
  ├─ reset_prefix_cache
  │    └─ EngineCore.reset_prefix_cache()
  │         └─ Scheduler.reset_prefix_cache()
  │
  ├─ reset_encoder_cache
  │    └─ EngineCore.reset_encoder_cache()
  │         ├─ Scheduler.reset_encoder_cache()
  │         └─ model_executor.reset_encoder_cache()
  │
  ├─ pause_generation / sleep
  │    └─ EngineCore.pause_scheduler() / sleep()
  │         ├─ Scheduler.set_pause_state(...)
  │         ├─ optional Scheduler.finish_requests(...)
  │         ├─ optional _reset_caches()
  │         └─ optional model_executor.sleep(level)
  │
  ├─ wake_up
  │    └─ EngineCore.wake_up()
  │         ├─ model_executor.wake_up(tags)
  │         └─ Scheduler.set_pause_state(UNPAUSED)
  │
  └─ shutdown
       ├─ renderer.shutdown()
       ├─ EngineCoreClient.shutdown()
       │    ├─ Inproc: EngineCore.shutdown()
       │    └─ MP: engine_manager / sockets / tasks cleanup
       └─ output_handler cancel
```

---

## 37. 和其它文档的关系

`02_llm_engine_sync.md` 解释的是：

```text
同步 LLMEngine 如何接入请求和输出主路径。
```

本篇补充：

```text
主路径之外的控制接口如何进入 EngineCore / Scheduler / model_executor。
```

`06_engine_core_client_bridge.md` 解释的是：

```text
EngineCoreClient 如何屏蔽同进程、多进程、同步、异步差异。
```

本篇中的 profile / reset / sleep / LoRA 等控制接口在多进程下基本都依赖 `UTILITY` 消息机制。

`engine_core/02_request_entry.md` 解释的是：

```text
请求如何 add / abort 进入 EngineCore。
```

本篇展开 abort、pause、sleep、shutdown 等更完整生命周期控制。

`scheduler/01_request_states.md` 和 `scheduler/08_update_after_worker_output.md` 解释的是：

```text
Scheduler 中请求如何处在 waiting / running / skipped_waiting / finished，
以及 Worker 输出回来后如何释放资源。
```

本篇中的 abort / pause / shutdown 最终都会影响这些 Scheduler 状态和资源释放路径。
