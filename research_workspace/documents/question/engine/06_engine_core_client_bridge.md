# 06. EngineCoreClient 如何连接外层 Engine 和 EngineCore？

源码位置：`vllm/v1/engine/core_client.py`

本问题关注：为什么外层 `LLMEngine` / `AsyncLLM` 通常通过 `EngineCoreClient` 访问 `EngineCore`，以及不同 client 形态如何屏蔽同进程、多进程、同步和异步差异。

---

## 1. 一句话回答

`EngineCoreClient` 是外层 Engine 和内部 `EngineCore` 之间的桥接层。

它的作用是把外层统一的调用：

```text
add_request / add_request_async
get_output / get_output_async
abort_requests / abort_requests_async
utility calls
```

映射到不同的 EngineCore 运行形态：

```text
InprocClient：
  同进程直接调用 EngineCore 对象。

SyncMPClient：
  同步前端 + 后台 EngineCoreProc，多进程 ZMQ 通信。

AsyncMPClient：
  异步前端 + 后台 EngineCoreProc，多进程 asyncio ZMQ 通信。
```

一句话：

```text
EngineCoreClient 让 LLMEngine / AsyncLLM 不必关心 EngineCore 是本进程对象，还是后台进程 busy loop；
外层只和统一 client 接口交互。
```

主线是：

```text
LLMEngine / AsyncLLM
  → EngineCoreClient
  → Inproc EngineCore 或 EngineCoreProc
  → Scheduler / Worker
```

---

## 2. EngineCoreClient 在整体链路中的位置

请求进入 EngineCore 的链路是：

```text
外层 Engine / AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCoreClient.add_request() / add_request_async()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
```

其中 `EngineCoreClient` 位于：

```text
EngineCoreRequest 之后；
EngineCore / EngineCoreProc 之前。
```

输出方向则是：

```text
Scheduler.update_from_output()
  → EngineCoreOutputs
  → EngineCoreClient.get_output() / get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

所以 `EngineCoreClient` 同时连接：

```text
输入方向：EngineCoreRequest → EngineCore
输出方向：EngineCoreOutputs → 外层 Engine
```

---

## 3. EngineCoreClient 抽象接口

`EngineCoreClient` 定义在：

```python
class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """
```

位置：`vllm/v1/engine/core_client.py:71` 到 `vllm/v1/engine/core_client.py:80`

这段注释直接说明了三种 client 的定位。

抽象接口包含几类能力。

### 3.1 同步请求和输出接口

```python
def get_output(self) -> EngineCoreOutputs:
    raise NotImplementedError

def add_request(self, request: EngineCoreRequest) -> None:
    raise NotImplementedError

def abort_requests(self, request_ids: list[str]) -> None:
    raise NotImplementedError
```

位置：`vllm/v1/engine/core_client.py:137` 到 `vllm/v1/engine/core_client.py:175`

这类接口主要给同步 `LLMEngine` 使用。

### 3.2 异步请求和输出接口

```python
async def get_output_async(self) -> EngineCoreOutputs:
    raise NotImplementedError

async def add_request_async(self, request: EngineCoreRequest) -> None:
    raise NotImplementedError

async def abort_requests_async(self, request_ids: list[str]) -> None:
    raise NotImplementedError
```

位置：`vllm/v1/engine/core_client.py:212` 到 `vllm/v1/engine/core_client.py:246`

这类接口主要给 `AsyncLLM` 使用。

### 3.3 utility / 管理接口

`EngineCoreClient` 还统一暴露：

```text
get_supported_tasks
profile
reset_mm_cache
reset_prefix_cache
reset_encoder_cache
sleep / wake_up / is_sleeping
execute_dummy_batch
add_lora / remove_lora / list_loras / pin_lora
save_sharded_state
collective_rpc
pause_scheduler_async / resume_scheduler_async / is_scheduler_paused_async
dp_engines_running
scale_elastic_ep
```

其中抽象基类接口主要位于 `vllm/v1/engine/core_client.py:137` 到 `vllm/v1/engine/core_client.py:273`；`pause_scheduler_async`、`resume_scheduler_async`、`is_scheduler_paused_async` 是 `AsyncMPClient` 提供的异步控制方法，位置：`vllm/v1/engine/core_client.py:1130` 到 `vllm/v1/engine/core_client.py:1139`。

大多数管理方法在同进程模式下直接转发给 `EngineCore`，在多进程模式下通过 `UTILITY` 消息发给后台 `EngineCoreProc`。但也有 client 本地状态和 DP 特例：`dp_engines_running()` 在 `InprocClient` 固定返回 `False`，在 `MPClient` 返回 `self.engines_running`；`scale_elastic_ep()` 只在 `DPLBAsyncMPClient` 中实现，用 client 侧流程编排 scale up / down，不是简单 UTILITY 转发。

---

## 4. make_client：根据模式选择 client 类型

创建同步 client 的入口是：

```python
@staticmethod
def make_client(
    multiprocess_mode: bool,
    asyncio_mode: bool,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
) -> "EngineCoreClient":
```

位置：`vllm/v1/engine/core_client.py:82` 到 `vllm/v1/engine/core_client.py:89`

选择逻辑是：

```python
if asyncio_mode and not multiprocess_mode:
    raise NotImplementedError(...)

if multiprocess_mode and asyncio_mode:
    return EngineCoreClient.make_async_mp_client(...)

if multiprocess_mode and not asyncio_mode:
    return SyncMPClient(...)

return InprocClient(...)
```

位置：`vllm/v1/engine/core_client.py:90` 到 `vllm/v1/engine/core_client.py:105`

可以整理成表：

| multiprocess_mode | asyncio_mode | client |
|---|---|---|
| `False` | `False` | `InprocClient` |
| `True` | `False` | `SyncMPClient` |
| `True` | `True` | `AsyncMPClient` / DP variants |
| `False` | `True` | 暂不支持 |

这说明：

```text
异步 EngineCore 当前要求多进程；
同进程模式主要给同步 LLMEngine 使用。
```

---

## 5. LLMEngine 如何使用 EngineCoreClient

同步 `LLMEngine` 初始化中创建 client：

```python
# EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
self.engine_core = EngineCoreClient.make_client(
    multiprocess_mode=multiprocess_mode,
    asyncio_mode=False,
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
)
```

位置：`vllm/v1/engine/llm_engine.py:104` 到 `vllm/v1/engine/llm_engine.py:111`

同步请求送入：

```python
self.engine_core.add_request(request)
```

位置：`vllm/v1/engine/llm_engine.py:276`

同步 step 取输出：

```python
outputs = self.engine_core.get_output()
```

位置：`vllm/v1/engine/llm_engine.py:302` 到 `vllm/v1/engine/llm_engine.py:304`

同步 abort：

```python
self.engine_core.abort_requests(request_ids)
```

位置：`vllm/v1/engine/llm_engine.py:215` 到 `vllm/v1/engine/llm_engine.py:216`

所以同步生成主路径依赖三个动作：

```text
add_request()
get_output()
abort_requests()
```

但 `LLMEngine` 还会通过 `EngineCoreClient` 调用 `get_supported_tasks`、profile、cache reset、sleep / wake_up、LoRA、`collective_rpc`、`execute_dummy_batch` 等管理接口。

至于底层是同进程直接调用，还是 ZMQ 多进程通信，由 `EngineCoreClient` 屏蔽。

---

## 6. AsyncLLM 如何使用 EngineCoreClient

异步 `AsyncLLM` 初始化中直接创建 async MP client：

```python
# EngineCore (starts the engine in background process).
self.engine_core = EngineCoreClient.make_async_mp_client(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
    client_addresses=client_addresses,
    client_count=client_count,
    client_index=client_index,
)
```

位置：`vllm/v1/engine/async_llm.py:145` 到 `vllm/v1/engine/async_llm.py:153`

异步请求送入：

```python
await self.engine_core.add_request_async(request)
```

位置：`vllm/v1/engine/async_llm.py:411` 到 `vllm/v1/engine/async_llm.py:412`

异步输出 handler 中取输出：

```python
outputs = await engine_core.get_output_async()
```

位置：`vllm/v1/engine/async_llm.py:657` 到 `vllm/v1/engine/async_llm.py:660`

异步 abort：

```python
await self.engine_core.abort_requests_async(all_request_ids)
```

位置：`vllm/v1/engine/async_llm.py:717` 到 `vllm/v1/engine/async_llm.py:718`

所以异步路径对应关系是：

```text
AsyncLLM.add_request()
  → EngineCoreClient.add_request_async()

AsyncLLM output handler
  → EngineCoreClient.get_output_async()

AsyncLLM.abort()
  → EngineCoreClient.abort_requests_async()
```

---

## 7. InprocClient：同进程直接调用 EngineCore

`InprocClient` 定义：

```python
class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """
```

位置：`vllm/v1/engine/core_client.py:276` 到 `vllm/v1/engine/core_client.py:284`

它初始化时直接创建 `EngineCore`：

```python
def __init__(self, *args, **kwargs):
    self.engine_core = EngineCore(*args, **kwargs)
```

位置：`vllm/v1/engine/core_client.py:286` 到 `vllm/v1/engine/core_client.py:287`

所以同进程模式没有 ZMQ，没有后台 busy loop。

---

## 8. InprocClient.add_request()

同进程请求入口：

```python
def add_request(self, request: EngineCoreRequest) -> None:
    req, request_wave = self.engine_core.preprocess_add_request(request)
    self.engine_core.add_request(req, request_wave)
```

位置：`vllm/v1/engine/core_client.py:297` 到 `vllm/v1/engine/core_client.py:299`

这一步做了两件事：

```text
1. EngineCore.preprocess_add_request()
   EngineCoreRequest → Request

2. EngineCore.add_request()
   Request → Scheduler.add_request()
```

也就是说，同进程模式的请求入口是直接函数调用：

```text
LLMEngine.add_request()
  → InprocClient.add_request()
  → EngineCore.preprocess_add_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
```

没有序列化，也没有跨进程通信。

---

## 9. InprocClient.get_output()

同进程输出入口：

```python
def get_output(self) -> EngineCoreOutputs:
    outputs, model_executed = self.engine_core.step_fn()
    self.engine_core.post_step(model_executed=model_executed)
    return outputs and outputs.get(0) or EngineCoreOutputs()
```

位置：`vllm/v1/engine/core_client.py:289` 到 `vllm/v1/engine/core_client.py:292`

这说明同步同进程模式下：

```text
LLMEngine.step()
  → InprocClient.get_output()
  → EngineCore.step_fn()
  → Scheduler.schedule()
  → Worker / ModelRunner
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → EngineCore.post_step()
```

其中 `outputs` 类型是：

```text
dict[int, EngineCoreOutputs]
```

同进程 `LLMEngine` 只有一个 client，所以取：

```python
outputs.get(0)
```

如果本轮没有输出，则返回空的 `EngineCoreOutputs()`。

---

## 10. InprocClient 的管理方法

`InprocClient` 的很多方法都是直接转发给 `EngineCore`：

```python
def abort_requests(self, request_ids: list[str]) -> None:
    if len(request_ids) > 0:
        self.engine_core.abort_requests(request_ids)
```

位置：`vllm/v1/engine/core_client.py:301` 到 `vllm/v1/engine/core_client.py:303`

例如：

```python
def reset_prefix_cache(...):
    return self.engine_core.reset_prefix_cache(...)

def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.engine_core.add_lora(lora_request)

def collective_rpc(...):
    return self.engine_core.collective_rpc(...)
```

位置：`vllm/v1/engine/core_client.py:314` 到 `vllm/v1/engine/core_client.py:363`

同进程模式下 `dp_engines_running()` 固定返回 False：

```python
def dp_engines_running(self) -> bool:
    return False
```

位置：`vllm/v1/engine/core_client.py:365` 到 `vllm/v1/engine/core_client.py:366`

---

## 11. MPClient：多进程 client 基类

`MPClient` 是 `SyncMPClient` 和 `AsyncMPClient` 的基类：

```python
class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket

        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """
```

位置：`vllm/v1/engine/core_client.py:467` 到 `vllm/v1/engine/core_client.py:478`

多进程模式的核心变化是：

```text
EngineCore 不在前端进程直接 step；
EngineCoreProc 在后台进程跑 busy loop；
前端通过 ZMQ input_socket 发送请求；
后台通过 ZMQ output_socket 返回 EngineCoreOutputs。
```

---

## 12. MPClient 初始化：ZMQ、后台 EngineCore 和序列化

`MPClient.__init__()` 主要做几类事情。

### 12.1 创建 ZMQ context 和资源 finalizer

```python
sync_ctx = zmq.Context(io_threads=2)
self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx
self.resources = BackgroundResources(ctx=sync_ctx)
self._finalizer = weakref.finalize(self, self.resources)
```

位置：`vllm/v1/engine/core_client.py:490` 到 `vllm/v1/engine/core_client.py:498`

`BackgroundResources` 用于 client 被回收时清理 socket、后台进程、任务等资源。

### 12.2 创建 input / output socket

如果没有外部传入 `client_addresses`，client 会自己创建地址并启动 engine：

```python
addresses = get_engine_zmq_addresses(vllm_config)
self.input_socket = self.resources.input_socket = make_zmq_socket(
    self.ctx,
    addresses.inputs[0],
    zmq.ROUTER,
    bind=True,
    router_handover=enable_input_socket_handover,
)
self.resources.output_socket = make_zmq_socket(
    self.ctx, addresses.outputs[0], zmq.PULL
)
```

位置：`vllm/v1/engine/core_client.py:551` 到 `vllm/v1/engine/core_client.py:562`

也就是说前端侧：

```text
input_socket：ROUTER，用于发消息给 EngineCoreProc 的 DEALER。
output_socket：PULL，用于接收 EngineCoreProc 的 PUSH 输出。
```

### 12.3 启动后台 EngineCore

```python
with launch_core_engines(
    vllm_config, executor_class, log_stats, addresses
) as (engine_manager, coordinator, addresses, tensor_queue):
    self.resources.coordinator = coordinator
    self.resources.engine_manager = engine_manager
```

位置：`vllm/v1/engine/core_client.py:573` 到 `vllm/v1/engine/core_client.py:578`

这里会启动后台 `EngineCoreProc`。

### 12.4 序列化和多模态 tensor IPC

```python
self.encoder = MsgpackEncoder(oob_tensor_consumer=tensor_ipc_sender)
self.decoder = MsgpackDecoder(EngineCoreOutputs)
```

位置：`vllm/v1/engine/core_client.py:593` 到 `vllm/v1/engine/core_client.py:594`

如果启用了多模态 tensor IPC，会创建 `TensorIpcSender`：

```python
if mm_tensor_ipc == "torch_shm" and tensor_queue is not None:
    tensor_ipc_sender = TensorIpcSender(tensor_queue)
```

位置：`vllm/v1/engine/core_client.py:587` 到 `vllm/v1/engine/core_client.py:591`

这用于把多模态 tensor 等大对象通过辅助通道传给后台 EngineCore。

### 12.5 等待 EngineCore ready

MPClient 会等待每个 engine 发送 ready response：

```python
while identities:
    ...
    identity, payload = sync_input_socket.recv_multipart()
    identities.remove(identity)
    self._apply_ready_response(payload)
```

位置：`vllm/v1/engine/core_client.py:615` 到 `vllm/v1/engine/core_client.py:633`

ready response 会把后台初始化后的部分配置同步回前端，例如：

```text
max_model_len
num_gpu_blocks
block_size
kv_cache_size_tokens
kv_cache_max_concurrency
dp_stats_address
```

相关处理位置：`vllm/v1/engine/core_client.py:714` 到 `vllm/v1/engine/core_client.py:755`

---

## 13. EngineCoreRequestType：多进程消息类型

多进程通信使用 `EngineCoreRequestType` 区分消息类型：

```python
class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    EXECUTOR_FAILED = b"\x04"
    WAKEUP = b"\x05"
```

位置：`vllm/v1/engine/__init__.py:252` 到 `vllm/v1/engine/__init__.py:265`

消息来源和含义是：

| 类型 | 含义 |
|---|---|
| `ADD` | 前端发送的新请求进入 EngineCore |
| `ABORT` | 前端发送的取消请求 |
| `UTILITY` | 前端发送的 EngineCore 管理方法调用 |
| `START_DP_WAVE` | `DPCoordinator` 广播的 DP wave 协调消息 |
| `EXECUTOR_FAILED` | `EngineCoreProc.input_queue` 内部执行器失败 sentinel |
| `WAKEUP` | `EngineCoreProc.input_queue` 内部 shutdown / 队列唤醒 sentinel |

---

## 14. SyncMPClient：同步前端 + 后台 EngineCoreProc

`SyncMPClient` 定义：

```python
class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""
```

位置：`vllm/v1/engine/core_client.py:779` 到 `vllm/v1/engine/core_client.py:780`

初始化时调用 MPClient：

```python
super().__init__(
    asyncio_mode=False,
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=log_stats,
)
```

位置：`vllm/v1/engine/core_client.py:786` 到 `vllm/v1/engine/core_client.py:791`

并创建同步输出队列：

```python
self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()
```

位置：`vllm/v1/engine/core_client.py:793` 到 `vllm/v1/engine/core_client.py:794`

---

## 15. SyncMPClient 输出线程

`SyncMPClient` 会启动一个线程处理 output socket：

```python
self.output_queue_thread = Thread(
    target=process_outputs_socket,
    name="EngineCoreOutputQueueThread",
    daemon=True,
)
self.output_queue_thread.start()
```

位置：`vllm/v1/engine/core_client.py:838` 到 `vllm/v1/engine/core_client.py:844`

线程逻辑是：

```text
poll output_socket
  → recv_multipart()
  → validate_alive()
  → decoder.decode(frames)
  → 如果是 utility_output，设置对应 Future
  → 否则放入 outputs_queue
  → 异常也放入 outputs_queue
```

对应源码：

```python
frames = out_socket.recv_multipart(copy=False)
resources.validate_alive(frames)
outputs: EngineCoreOutputs = decoder.decode(frames)
if outputs.utility_output:
    _process_utility_output(outputs.utility_output, utility_results)
else:
    outputs_queue.put_nowait(outputs)
```

位置：`vllm/v1/engine/core_client.py:824` 到 `vllm/v1/engine/core_client.py:830`

这使得同步前端的 `get_output()` 可以只从本地 queue 阻塞读取。

---

## 16. SyncMPClient.get_output()

同步多进程输出入口：

```python
def get_output(self) -> EngineCoreOutputs:
    outputs = self.outputs_queue.get()

    if isinstance(outputs, Exception):
        raise self._format_exception(outputs) from None
    if outputs.wave_complete is not None:
        self.engines_running = False
    return outputs
```

位置：`vllm/v1/engine/core_client.py:849` 到 `vllm/v1/engine/core_client.py:859`

含义是：

```text
LLMEngine.step()
  → SyncMPClient.get_output()
  → 从 outputs_queue 阻塞取 EngineCoreOutputs
  → 交给 OutputProcessor.process_outputs()
```

注意，同步 MP 模式下：

```text
LLMEngine.step() 本身不调用 EngineCore.step_fn()；
EngineCoreProc 后台 busy loop 已经在独立进程中 step；
前端只是取输出。
```

---

## 17. SyncMPClient._send_input()

同步多进程发送消息入口：

```python
def _send_input(self, request_type: EngineCoreRequestType, request: Any):
    self.ensure_alive()
    self.free_pending_messages()
    # (Identity, RequestType, SerializedRequest)
    msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
```

位置：`vllm/v1/engine/core_client.py:861` 到 `vllm/v1/engine/core_client.py:865`

如果没有额外 tensor buffer：

```python
self.input_socket.send_multipart(msg, copy=False)
```

位置：`vllm/v1/engine/core_client.py:867` 到 `vllm/v1/engine/core_client.py:870`

如果有 tensor backing buffer，则用 tracker 跟踪并保留 request 引用：

```python
tracker = self.input_socket.send_multipart(msg, copy=False, track=True)
self.add_pending_message(tracker, request)
```

位置：`vllm/v1/engine/core_client.py:872` 到 `vllm/v1/engine/core_client.py:873`

这样避免 ZMQ 还没发送完，Python 对象或 tensor buffer 就被释放。

---

## 18. SyncMPClient.add_request() / abort_requests()

同步多进程 add：

```python
def add_request(self, request: EngineCoreRequest) -> None:
    if self.is_dp:
        self.engines_running = True
    self._send_input(EngineCoreRequestType.ADD, request)
```

位置：`vllm/v1/engine/core_client.py:886` 到 `vllm/v1/engine/core_client.py:889`

同步多进程 abort：

```python
def abort_requests(self, request_ids: list[str]) -> None:
    if request_ids and not self.resources.engine_dead:
        self._send_input(EngineCoreRequestType.ABORT, request_ids)
```

位置：`vllm/v1/engine/core_client.py:891` 到 `vllm/v1/engine/core_client.py:893`

所以同步 MP 请求入口是：

```text
LLMEngine.add_request()
  → SyncMPClient.add_request()
  → _send_input(ADD, EngineCoreRequest)
  → ZMQ ROUTER → EngineCoreProc DEALER
```

abort 入口是：

```text
LLMEngine.abort_request()
  → SyncMPClient.abort_requests()
  → _send_input(ABORT, request_ids)
```

---

## 19. SyncMPClient utility 调用

很多管理方法走 `call_utility()`：

```python
def call_utility(self, method: str, *args) -> Any:
    call_id = uuid.uuid1().int >> 64
    future: Future[Any] = Future()
    self.utility_results[call_id] = future
    self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

    return future.result()
```

位置：`vllm/v1/engine/core_client.py:875` 到 `vllm/v1/engine/core_client.py:881`

例如：

```python
def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
    return self.call_utility("get_supported_tasks")

def reset_prefix_cache(...):
    return self.call_utility("reset_prefix_cache", reset_running_requests, reset_connector)

def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.call_utility("add_lora", lora_request)
```

位置：`vllm/v1/engine/core_client.py:883` 到 `vllm/v1/engine/core_client.py:921`

也就是说同步 MP 的 utility 模式是：

```text
前端创建 call_id 和 Future
  → 发送 UTILITY(client_idx, call_id, method, args)
  → 后台调用 EngineCore 对应方法
  → 返回 UtilityOutput(call_id, result / failure)
  → 输出线程设置 Future
  → call_utility() 返回结果
```

---

## 20. AsyncMPClient：异步前端 + 后台 EngineCoreProc

`AsyncMPClient` 定义：

```python
class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""
```

位置：`vllm/v1/engine/core_client.py:950` 到 `vllm/v1/engine/core_client.py:951`

初始化时使用 asyncio 模式：

```python
super().__init__(
    asyncio_mode=True,
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=log_stats,
    client_addresses=client_addresses,
)
```

位置：`vllm/v1/engine/core_client.py:963` 到 `vllm/v1/engine/core_client.py:969`

并保存：

```python
self.client_count = client_count
self.client_index = client_index
self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()
```

位置：`vllm/v1/engine/core_client.py:971` 到 `vllm/v1/engine/core_client.py:973`

`client_index` 很关键：它用于多 API server / 多前端 client 场景下，把 EngineCore 输出路由回正确客户端。

---

## 21. AsyncMPClient 输出任务

异步输出任务由 `_ensure_output_queue_task()` 启动：

```python
def _ensure_output_queue_task(self):
    resources = self.resources
    if resources.output_queue_task is not None:
        return
    ...
    resources.output_queue_task = asyncio.create_task(
        process_outputs_socket(), name="EngineCoreOutputQueueTask"
    )
```

位置：`vllm/v1/engine/core_client.py:984` 到 `vllm/v1/engine/core_client.py:1051`

任务核心逻辑是：

```python
frames = await output_socket.recv_multipart(copy=False)
resources.validate_alive(frames)
outputs: EngineCoreOutputs = decoder.decode(frames)
if outputs.utility_output:
    ...
    continue
...
if outputs.outputs or outputs.scheduler_stats:
    outputs_queue.put_nowait(outputs)
```

位置：`vllm/v1/engine/core_client.py:1005` 到 `vllm/v1/engine/core_client.py:1043`

这说明异步 client 会在后台 task 中持续接收 EngineCoreProc 输出：收到 `utility_output` 时解析对应 Future；子类可以先通过 `process_engine_outputs` 处理 DP / LB 元数据；只有 `outputs.outputs` 或 `outputs.scheduler_stats` 非空时才放入 asyncio queue。

---

## 22. AsyncMPClient.get_output_async()

异步输出入口：

```python
async def get_output_async(self) -> EngineCoreOutputs:
    self._ensure_output_queue_task()
    assert self.outputs_queue is not None
    outputs = await self.outputs_queue.get()
    if isinstance(outputs, Exception):
        raise self._format_exception(outputs) from None
    return outputs
```

位置：`vllm/v1/engine/core_client.py:1053` 到 `vllm/v1/engine/core_client.py:1062`

这对应 `AsyncLLM` output handler：

```text
while True:
  outputs = await engine_core.get_output_async()
  output_processor.process_outputs(...)
```

---

## 23. AsyncMPClient 发送消息

异步发送入口：

```python
def _send_input(
    self,
    request_type: EngineCoreRequestType,
    request: Any,
    engine: EngineIdentity | None = None,
) -> Awaitable[Any]:
```

位置：`vllm/v1/engine/core_client.py:1064` 到 `vllm/v1/engine/core_client.py:1069`

它构造消息：

```python
message = (request_type.value, *self.encoder.encode(request))
return self._send_input_message(message, engine, request)
```

位置：`vllm/v1/engine/core_client.py:1073` 到 `vllm/v1/engine/core_client.py:1074`

真正发送时会拼上 engine identity：

```python
msg = (engine,) + message
...
return self.input_socket.send_multipart(msg, copy=False)
```

位置：`vllm/v1/engine/core_client.py:1086` 到 `vllm/v1/engine/core_client.py:1089`

如果有 tensor buffer，也会通过 tracker 保留对象引用：

```python
future = self.input_socket.send_multipart(msg, copy=False, track=True)
...
future.add_done_callback(add_pending)
```

位置：`vllm/v1/engine/core_client.py:1091` 到 `vllm/v1/engine/core_client.py:1099`

---

## 24. AsyncMPClient.add_request_async() / abort_requests_async()

异步 add：

```python
async def add_request_async(self, request: EngineCoreRequest) -> None:
    request.client_index = self.client_index
    await self._send_input(EngineCoreRequestType.ADD, request)
    self._ensure_output_queue_task()
```

位置：`vllm/v1/engine/core_client.py:1121` 到 `vllm/v1/engine/core_client.py:1124`

这里会设置：

```text
request.client_index = self.client_index
```

作用是：

```text
EngineCoreOutputs 返回时，EngineCoreProc 可以按 client_index 发回对应前端 client。
```

异步 abort：

```python
async def abort_requests_async(self, request_ids: list[str]) -> None:
    if request_ids and not self.resources.engine_dead:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids)
```

位置：`vllm/v1/engine/core_client.py:1126` 到 `vllm/v1/engine/core_client.py:1128`

---

## 25. AsyncMPClient utility 调用

异步 utility 入口：

```python
async def call_utility_async(self, method: str, *args) -> Any:
    return await self._call_utility_async(method, *args, engine=self.core_engine)
```

位置：`vllm/v1/engine/core_client.py:1101` 到 `vllm/v1/engine/core_client.py:1102`

核心逻辑：

```python
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

位置：`vllm/v1/engine/core_client.py:1104` 到 `vllm/v1/engine/core_client.py:1116`

异步 utility 和同步 utility 的差别是：

```text
同步：Future.result() 阻塞等待；
异步：await future。
```

---

## 26. EngineCoreProc 如何接收 ADD / ABORT / UTILITY

后台多进程模式下，EngineCoreProc 有 input socket IO thread。

入口：

```python
def process_input_sockets(...):
    """Input socket IO thread."""
```

位置：`vllm/v1/engine/core.py:1493` 到 `vllm/v1/engine/core.py:1500`

收到消息后先解析类型：

```python
type_frame, *data_frames = input_socket.recv_multipart(copy=False)
request_type = EngineCoreRequestType(bytes(type_frame.buffer))
```

位置：`vllm/v1/engine/core.py:1568` 到 `vllm/v1/engine/core.py:1574`

### 26.1 ADD：先反序列化 EngineCoreRequest，再预处理

```python
if request_type == EngineCoreRequestType.ADD:
    req: EngineCoreRequest = add_request_decoder.decode(data_frames)
    try:
        request = self.preprocess_add_request(req)
    except Exception:
        self._handle_request_preproc_error(req)
        continue
```

位置：`vllm/v1/engine/core.py:1578` 到 `vllm/v1/engine/core.py:1584`

注意这里的 `request` 会变成：

```text
(req: Request, request_wave: int)
```

也就是说多进程模式下：

```text
preprocess_add_request() 在 input socket 线程里执行。
```

这样可以和模型 forward 重叠，减少主 busy loop 的阻塞。

### 26.2 ABORT：同时进入 aborts_queue 和 input_queue

非 ADD 消息走 generic decoder：

```python
request = generic_decoder.decode(data_frames)

if request_type == EngineCoreRequestType.ABORT:
    self.aborts_queue.put_nowait(request)
```

位置：`vllm/v1/engine/core.py:1585` 到 `vllm/v1/engine/core.py:1593`

注释说明：

```python
# Aborts are added to *both* queues, allows us to eagerly
# process aborts while also ensuring ordering in the input
# queue to avoid leaking requests. This is ok because
# aborting in the scheduler is idempotent.
```

位置：`vllm/v1/engine/core.py:1588` 到 `vllm/v1/engine/core.py:1592`

最后所有请求都会进入 input queue：

```python
self.input_queue.put_nowait((request_type, request))
```

位置：`vllm/v1/engine/core.py:1595` 到 `vllm/v1/engine/core.py:1596`

---

## 27. EngineCoreProc busy loop 如何分发 client 请求

后台 busy loop 会先处理 input queue，再 step：

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()
```

位置：`vllm/v1/engine/core.py:1268` 到 `vllm/v1/engine/core.py:1275`

`_process_input_queue()` 里会调用：

```python
req = self.input_queue.get(block=block)
self._handle_client_request(*req)
```

位置：`vllm/v1/engine/core.py:1294` 到 `vllm/v1/engine/core.py:1295`

分发逻辑在 `_handle_client_request()`：

```python
if request_type == EngineCoreRequestType.WAKEUP:
    return
elif request_type == EngineCoreRequestType.ADD:
    req, request_wave = request
    if self._reject_add_in_shutdown(req):
        return
    self.add_request(req, request_wave)
elif request_type == EngineCoreRequestType.ABORT:
    self.abort_requests(request)
elif request_type == EngineCoreRequestType.UTILITY:
    client_idx, call_id, method_name, args = request
    ...
```

位置：`vllm/v1/engine/core.py:1381` 到 `vllm/v1/engine/core.py:1408`

所以后台收到的三类主消息分别是：

```text
ADD：
  EngineCore.add_request(req, request_wave)

ABORT：
  EngineCore.abort_requests(request_ids)

UTILITY：
  getattr(self, method_name)(*args)
  然后返回 UtilityOutput
```

---

## 28. EngineCoreProc 如何把输出发回 client

后台 step 产出 EngineCoreOutputs 后：

```python
outputs, model_executed = self.step_fn()
# Put EngineCoreOutputs into the output queue.
for output in outputs.items() if outputs else ():
    self.output_queue.put_nowait(output)
# Post-step hook.
self.post_step(model_executed)
```

位置：`vllm/v1/engine/core.py:1313` 到 `vllm/v1/engine/core.py:1318`

这里 `outputs.items()` 的元素是：

```text
(client_index, EngineCoreOutputs)
```

随后 output socket IO thread 从 `output_queue` 取出并发送：

```python
client_index, outputs = output
outputs.engine_index = engine_index
...
tracker = sockets[client_index].send_multipart(
    buffers, copy=False, track=True
)
```

位置：`vllm/v1/engine/core.py:1639` 到 `vllm/v1/engine/core.py:1657`

这说明多 client 场景下：

```text
EngineCoreOutputs 按 client_index 路由到对应前端 output socket。
```

如果 `client_index == -1`，则是 coordinator 消息：

```python
if client_index == -1:
    assert coord_socket is not None
    coord_socket.send_multipart(encoder.encode(outputs))
    continue
```

位置：`vllm/v1/engine/core.py:1642` 到 `vllm/v1/engine/core.py:1647`

---

## 29. EngineCoreProc utility 返回路径

后台处理 UTILITY 时会构造 `UtilityOutput`：

```python
output = UtilityOutput(call_id)
...
enqueue_output = lambda out: self.output_queue.put_nowait(
    (client_idx, EngineCoreOutputs(utility_output=out))
)
self._invoke_utility_method(method_name, get_result, output, enqueue_output)
```

位置：`vllm/v1/engine/core.py:1399` 到 `vllm/v1/engine/core.py:1408`

`_invoke_utility_method()` 会调用目标方法：

```python
result = get_result()
...
output.result = UtilityResult(result)
...
enqueue_output(output)
```

位置：`vllm/v1/engine/core.py:1447` 到 `vllm/v1/engine/core.py:1460`

前端收到后用 `_process_utility_output()` 找到对应 Future：

```python
future = utility_results.pop(output.call_id)
...
future.set_result(output.result.result)
```

位置：`vllm/v1/engine/core_client.py:757` 到 `vllm/v1/engine/core_client.py:768`

所以 utility 完整闭环是：

```text
client.call_utility(method, args)
  → 发送 UTILITY(client_idx, call_id, method, args)
  → EngineCoreProc._handle_client_request(UTILITY)
  → getattr(EngineCoreProc, method)(*args)
  → UtilityOutput(call_id, result)
  → EngineCoreOutputs(utility_output=...)
  → client output thread / task
  → utility_results[call_id].set_result()
```

---

## 30. BackgroundResources：清理后台资源

多进程 client 使用 `BackgroundResources` 做清理：

```python
@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""
```

位置：`vllm/v1/engine/core_client.py:369` 到 `vllm/v1/engine/core_client.py:372`

它保存：

```text
ctx
engine_manager
coordinator
input_socket / output_socket
async tasks
shutdown_path
engine_dead flag
```

相关字段位置：`vllm/v1/engine/core_client.py:374` 到 `vllm/v1/engine/core_client.py:390`

清理时会：

```text
标记 engine_dead=True；
shutdown engine_manager；
shutdown coordinator；
关闭 sockets；
取消 async output / stats tasks；
同步模式下通过 shutdown_path 通知输出线程退出。
```

相关逻辑位置：`vllm/v1/engine/core_client.py:392` 到 `vllm/v1/engine/core_client.py:452`

这说明 `EngineCoreClient` 不只是通信包装，也负责多进程资源生命周期的一部分。

---

## 31. 异常和 EngineDeadError 传播

多进程 client 通过 `engine_dead` 标记传播后台异常。

`validate_alive()` 会识别后台发来的 dead sentinel：

```python
def validate_alive(self, frames: Sequence[zmq.Frame]):
    if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
        self.engine_dead = True
        raise EngineDeadError()
```

位置：`vllm/v1/engine/core_client.py:454` 到 `vllm/v1/engine/core_client.py:457`

client 侧调用前会检查：

```python
def ensure_alive(self):
    if self.resources.engine_dead:
        raise EngineDeadError()
```

位置：`vllm/v1/engine/core_client.py:670` 到 `vllm/v1/engine/core_client.py:672`

如果后台线程 / task 捕获异常，会放入输出队列；`get_output()` / `get_output_async()` 取到后抛出：

```python
if isinstance(outputs, Exception):
    raise self._format_exception(outputs) from None
```

同步位置：`vllm/v1/engine/core_client.py:855` 到 `vllm/v1/engine/core_client.py:856`

异步位置：`vllm/v1/engine/core_client.py:1060` 到 `vllm/v1/engine/core_client.py:1061`

所以异常传播链路是：

```text
EngineCoreProc / output thread 发现异常
  → outputs_queue 放入 Exception
  → get_output() / get_output_async()
  → 抛给 LLMEngine / AsyncLLM
```

---

## 32. DP AsyncMPClient：多 DP engine 的扩展

`make_async_mp_client()` 会根据 DP 配置返回不同 async client：

```python
if parallel_config.data_parallel_size > 1:
    if parallel_config.data_parallel_external_lb:
        # External load balancer - client per DP rank.
        return DPAsyncMPClient(*client_args)
    # Internal load balancer - client balances to all DP ranks.
    return DPLBAsyncMPClient(*client_args)
return AsyncMPClient(*client_args)
```

位置：`vllm/v1/engine/core_client.py:126` 到 `vllm/v1/engine/core_client.py:132`

也就是说异步多进程还有两种 DP 变体：

```text
DPAsyncMPClient：
  外部负载均衡，通常 client per DP rank。

DPLBAsyncMPClient：
  内部负载均衡，一个 client 在多个 EngineCore 之间选 engine。
```

---

## 33. DPAsyncMPClient：current_wave 和 FIRST_REQ

`DPAsyncMPClient` 初始化时维护：

```python
self.current_wave = 0
```

位置：`vllm/v1/engine/core_client.py:1213`

add request 时会设置：

```python
request.current_wave = self.current_wave
request.client_index = self.client_index

chosen_engine = self.get_core_engine_for_request(request)
to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
if not self.engines_running:
    req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
    await self.first_req_send_socket.send(req_msg)

await to_await
self._ensure_output_queue_task()
```

位置：`vllm/v1/engine/core_client.py:1359` 到 `vllm/v1/engine/core_client.py:1374`

这说明 DP async client 还要处理：

```text
current_wave 标记；
engines_running 状态；
首个请求唤醒 coordinator / engine wave；
按 engine identity 发送请求。
```

这些差异都不暴露给 `AsyncLLM`。

---

## 34. DPLBAsyncMPClient：内部负载均衡和 abort 路由

`DPLBAsyncMPClient` 会记录每个请求发到了哪个 engine：

```python
self.reqs_in_flight: dict[str, EngineIdentity] = {}
```

位置：`vllm/v1/engine/core_client.py:1395` 到 `vllm/v1/engine/core_client.py:1396`

选 engine 时会看：

```python
if (eng_index := request.data_parallel_rank) is None and (
    eng_index := get_late_interaction_engine_index(
        request.pooling_params, len(self.core_engines)
    )
) is None:
    current_counts = self.lb_engines
    ...
    waiting, running = current_counts[idx]
    score = waiting * 4 + running
    ...
```

位置：`vllm/v1/engine/core_client.py:1413` 到 `vllm/v1/engine/core_client.py:1437`

选择后记录：

```python
chosen_engine = self.core_engines[eng_index]
self.reqs_in_flight[request.request_id] = chosen_engine
return chosen_engine
```

位置：`vllm/v1/engine/core_client.py:1444` 到 `vllm/v1/engine/core_client.py:1447`

abort 时按请求所属 engine 路由：

```python
if len(request_ids) == 1:
    if engine := self.reqs_in_flight.get(request_ids[0]):
        await self._abort_requests(request_ids, engine)
    return

by_engine = defaultdict[EngineIdentity, list[str]](list)
for req_id in request_ids:
    if engine := self.reqs_in_flight.get(req_id):
        by_engine[engine].append(req_id)
for engine, req_ids in by_engine.items():
    await self._abort_requests(req_ids, engine)
```

位置：`vllm/v1/engine/core_client.py:1529` 到 `vllm/v1/engine/core_client.py:1544`

这说明：

```text
多 DP engine 场景下，add 和 abort 都必须路由到正确 EngineCoreProc。
EngineCoreClient 负责隐藏这层路由复杂度。
```

---

## 35. Client 屏蔽了哪些差异

`EngineCoreClient` 主要屏蔽以下差异。

### 35.1 请求发送方式

```text
InprocClient：
  直接函数调用 EngineCore.preprocess_add_request() / add_request()。

SyncMPClient：
  ZMQ ROUTER 发送 ADD 消息。

AsyncMPClient：
  asyncio ZMQ 发送 ADD 消息。

DPLBAsyncMPClient：
  先选择 DP engine，再发送 ADD。
```

### 35.2 输出获取方式

```text
InprocClient：
  get_output() 直接调用 EngineCore.step_fn()。

SyncMPClient：
  后台线程从 output_socket 收消息，get_output() 从 queue 阻塞取。

AsyncMPClient：
  后台 asyncio task 从 output_socket 收消息，get_output_async() await queue。
```

### 35.3 EngineCore 生命周期

```text
InprocClient：
  直接持有 EngineCore 对象。

MPClient：
  负责启动 / 连接后台 EngineCoreProc，等待 ready，清理进程和 sockets。
```

### 35.4 utility 调用方式

```text
InprocClient：
  直接调用 EngineCore 方法。

MPClient：
  发送 UTILITY 消息，通过 call_id / Future 接收结果。
```

### 35.5 异常传播

```text
MPClient：
  监听 EngineCoreProc dead sentinel；
  output thread / task 捕获异常；
  get_output 抛出 EngineDeadError 或原始异常。
```

### 35.6 多 client 和 DP 路由

```text
AsyncMPClient：
  设置 request.client_index。

EngineCoreProc：
  按 client_index 把 EngineCoreOutputs 发回对应 socket。

DPLBAsyncMPClient：
  根据负载选择 engine，并记录 reqs_in_flight 用于 abort 路由。
```

---

## 36. Inproc 和 MP 的核心区别

可以用一张表理解：

| 维度 | InprocClient | SyncMPClient / AsyncMPClient |
|---|---|---|
| EngineCore 所在位置 | 前端同进程 | 后台 EngineCoreProc |
| 请求发送 | 直接函数调用 | ZMQ 消息 |
| 输出获取 | 调用 `step_fn()` | 从 output socket / queue 获取 |
| 是否有 busy loop | 没有独立 busy loop | EngineCoreProc 有 busy loop |
| preprocessing 位置 | add_request 调用栈内 | input socket 线程内 |
| utility | 直接调用方法 | UTILITY 消息 + Future |
| 异常传播 | 普通 Python 调用 | EngineDeadError / output queue |
| 多 client 路由 | 通常只有 client 0 | 依赖 client_index |

最关键区别：

```text
InprocClient.get_output() 会驱动 EngineCore.step_fn()；
MPClient.get_output() 只是取后台 EngineCoreProc 已经产生的输出。
```

---

## 37. 完整流程图：InprocClient

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → OutputProcessor.add_request()
  → InprocClient.add_request()
       ├─ EngineCore.preprocess_add_request()
       │    └─ EngineCoreRequest → Request
       └─ EngineCore.add_request()
            └─ Scheduler.add_request()

LLMEngine.step()
  → InprocClient.get_output()
       ├─ EngineCore.step_fn()
       │    ├─ Scheduler.schedule()
       │    ├─ model_executor.execute_model()
       │    └─ Scheduler.update_from_output()
       ├─ EngineCore.post_step()
       └─ EngineCoreOutputs for client 0
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

---

## 38. 完整流程图：SyncMPClient

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → OutputProcessor.add_request()
  → SyncMPClient.add_request()
       └─ _send_input(ADD, EngineCoreRequest)
            └─ ZMQ ROUTER → EngineCoreProc DEALER

EngineCoreProc input thread
  → process_input_sockets()
  → decode ADD EngineCoreRequest
  → EngineCore.preprocess_add_request()
  → input_queue.put((ADD, (Request, wave)))

EngineCoreProc busy loop
  → _process_input_queue()
  → _handle_client_request(ADD)
  → EngineCore.add_request()
  → Scheduler.add_request()
  → _process_engine_step()
  → EngineCore.step_fn()
  → output_queue.put((client_index, EngineCoreOutputs))

EngineCoreProc output thread
  → process_output_sockets()
  → send_multipart(EngineCoreOutputs)

SyncMPClient output thread
  → recv_multipart()
  → decoder.decode()
  → outputs_queue.put_nowait(outputs)

LLMEngine.step()
  → SyncMPClient.get_output()
  → outputs_queue.get()
  → OutputProcessor.process_outputs()
```

---

## 39. 完整流程图：AsyncMPClient

```text
AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → OutputProcessor.add_request(..., queue)
  → AsyncMPClient.add_request_async()
       ├─ request.client_index = self.client_index
       ├─ _send_input(ADD, request)
       └─ _ensure_output_queue_task()

EngineCoreProc
  → 和 SyncMPClient 一样：input socket → input_queue → busy loop → output_queue → output socket

AsyncMPClient output task
  → await output_socket.recv_multipart()
  → decode EngineCoreOutputs
  → utility? resolve Future
  → output_handler? process DP/LB metadata
  → outputs_queue.put_nowait(outputs)

AsyncLLM output handler
  → await engine_core.get_output_async()
  → OutputProcessor.process_outputs()
  → RequestOutputCollector.put()
```

---

## 40. 容易疑惑的点

### 40.1 EngineCoreClient 是 EngineCore 吗？

不是。

`EngineCoreClient` 是访问 `EngineCore` 的桥。

```text
InprocClient 里面确实直接持有 EngineCore；
MPClient 里面持有的是 socket、engine manager 和后台进程资源。
```

### 40.2 为什么同步 LLMEngine 不直接持有 EngineCore？

因为同步 `LLMEngine` 既要支持同进程，也要支持多进程。

统一持有 `EngineCoreClient` 后，`LLMEngine` 只需要调用：

```text
add_request()
get_output()
abort_requests()
```

不需要关心底层实现。

### 40.3 InprocClient.get_output() 为什么会调用 step_fn？

因为同进程模式没有后台 busy loop。

必须由前端 `LLMEngine.step()` 通过 `InprocClient.get_output()` 主动驱动一轮：

```text
EngineCore.step_fn()
```

### 40.4 SyncMPClient.get_output() 为什么不调用 step_fn？

因为多进程模式下 `EngineCoreProc` 自己运行：

```text
run_busy_loop()
  → _process_input_queue()
  → _process_engine_step()
```

前端只从 `outputs_queue` 取结果。

### 40.5 多进程 ADD 为什么在 input socket 线程里 preprocess？

`process_input_sockets()` 收到 ADD 后会立即：

```text
EngineCoreRequest → preprocess_add_request() → Request
```

这样可以让请求初始化和模型 forward 重叠，减少 busy loop 被输入预处理阻塞的时间。

### 40.6 abort 为什么在多进程下进两个队列？

ABORT 会进入：

```text
input_queue：保证和 ADD 等输入事件的顺序关系；
aborts_queue：允许模型执行期间更早处理 abort。
```

因为 Scheduler abort 是幂等的，所以重复处理可接受。

### 40.7 utility_output 为什么不进入普通 outputs_queue？

utility 调用是管理方法调用，不是用户请求输出。

前端 output thread / task 收到 utility output 后，会通过 call_id 找到等待的 Future，并设置结果。

普通请求输出才进入 `outputs_queue` 给 `get_output()` / `get_output_async()`。

### 40.8 client_index 是做什么的？

`client_index` 用来在多前端 client 场景下路由输出。

```text
AsyncMPClient.add_request_async()
  → request.client_index = self.client_index

EngineCoreProc output thread
  → sockets[client_index].send_multipart(...)
```

### 40.9 DPLBAsyncMPClient 为什么要记录 reqs_in_flight？

因为一个请求被负载均衡到哪个 EngineCore，后续 abort 必须发给同一个 engine。

所以它记录：

```text
request_id → EngineIdentity
```

finished 后再移除。

---

## 41. 从“回答问题”的角度总结

如果要问：

```text
EngineCoreClient 如何连接外层 Engine 和 EngineCore？
```

可以回答：

```text
EngineCoreClient 是 LLMEngine / AsyncLLM 和 EngineCore 之间的统一访问层。
外层 Engine 把 InputProcessor 生成的 EngineCoreRequest 交给 EngineCoreClient，
再通过 EngineCoreClient 取回 EngineCoreOutputs。

在同进程模式下，InprocClient 直接创建并调用 EngineCore，
add_request() 会直接执行 preprocess_add_request() 和 EngineCore.add_request()，
get_output() 会直接调用 EngineCore.step_fn()。

在多进程同步模式下，SyncMPClient 通过 ZMQ 把 ADD / ABORT / UTILITY 消息发给后台 EngineCoreProc，
后台 busy loop 负责 step，前端 get_output() 只从本地 outputs_queue 阻塞读取结果。

在多进程异步模式下，AsyncMPClient 用 asyncio ZMQ 发送请求并用后台 task 接收输出，
通过 get_output_async() / add_request_async() / abort_requests_async() 服务 AsyncLLM。
```

核心公式是：

```text
外层统一接口：
  add_request / get_output / abort_requests
  add_request_async / get_output_async / abort_requests_async

底层实现差异：
  InprocClient = direct call
  SyncMPClient = sync queue + ZMQ + background EngineCoreProc
  AsyncMPClient = asyncio queue + ZMQ + background EngineCoreProc
```

---

## 42. 最关键流程图

```text
EngineCoreClient.make_client()
  ├─ multiprocess=False, asyncio=False
  │    └─ InprocClient
  │         ├─ EngineCore(*args)
  │         ├─ add_request() → preprocess_add_request() → add_request()
  │         └─ get_output() → step_fn() → post_step()
  │
  ├─ multiprocess=True, asyncio=False
  │    └─ SyncMPClient
  │         ├─ MPClient 启动 / 连接 EngineCoreProc
  │         ├─ add_request() → ZMQ ADD
  │         ├─ output thread → outputs_queue
  │         └─ get_output() → outputs_queue.get()
  │
  └─ multiprocess=True, asyncio=True
       └─ AsyncMPClient / DPAsyncMPClient / DPLBAsyncMPClient
            ├─ MPClient 启动 / 连接 EngineCoreProc
            ├─ add_request_async() → ZMQ ADD
            ├─ output task → asyncio.Queue
            └─ get_output_async() → await outputs_queue.get()
```

```text
后台 EngineCoreProc 多进程闭环：

input socket thread
  → decode ADD / ABORT / UTILITY
  → preprocess ADD
  → input_queue

busy loop
  → _process_input_queue()
  → _handle_client_request()
  → EngineCore.add_request() / abort_requests() / utility method
  → _process_engine_step()
  → step_fn()
  → output_queue

output socket thread
  → output_queue.get()
  → send EngineCoreOutputs to sockets[client_index]
```

---

## 43. 和其它文档的关系

`02_llm_engine_sync.md` 解释的是：

```text
同步 LLMEngine 如何通过 EngineCoreClient 连接 InputProcessor、EngineCore 和 OutputProcessor。
```

本篇展开其中的：

```text
EngineCoreClient 这座桥如何适配同进程 / 多进程 / 异步模式。
```

`engine_core/02_request_entry.md` 解释的是：

```text
EngineCoreRequest 如何进入 EngineCore，再进入 Scheduler。
```

本篇解释的是：

```text
EngineCoreRequest 是如何通过 EngineCoreClient 到达 EngineCore / EngineCoreProc 的。
```

`engine_core/03_step_loop.md` 解释的是：

```text
EngineCore.step() 如何 schedule → execute → update。
```

本篇补充：

```text
同进程模式下，InprocClient.get_output() 直接触发 step_fn；
多进程模式下，EngineCoreProc busy loop 触发 step_fn，client 只取结果。
```
