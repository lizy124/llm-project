# 02. 请求如何进入 EngineCore？

源码位置：`vllm/vllm/v1/engine/core.py`

本问题关注：外部请求如何进入 EngineCore，EngineCore 如何把请求交给 Scheduler，以及 abort / finish request 等控制请求如何传入内部执行流。

---

## 1. 一句话回答

请求进入 EngineCore 的主线是：

```text
外层 Engine / AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCoreClient.add_request() / add_request_async()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
  → Scheduler waiting / skipped_waiting 队列
```

一句话：

```text
外层 Engine 负责把用户输入处理成 EngineCoreRequest；
EngineCore 负责把 EngineCoreRequest 转成内部 Request；
Scheduler 负责真正接管请求队列和调度状态。
```

所以请求不是直接从 API 进入 Scheduler，而是经过：

```text
InputProcessor
  → EngineCoreClient
  → EngineCore
  → Scheduler
```

---

## 2. 请求对象分两层：EngineCoreRequest 和 Request

理解请求入口前，要先区分两个对象。

### 2.1 `EngineCoreRequest`：EngineCore 的外部输入对象

`EngineCoreRequest` 定义在：

```python
class EngineCoreRequest(msgspec.Struct, ...):
```

位置：`vllm/vllm/v1/engine/__init__.py:88`

它是外层 Engine 交给 EngineCore 的请求对象。

主要字段包括：

```python
request_id: str
prompt_token_ids: list[int] | None
mm_features: list[MultiModalFeatureSpec] | None
sampling_params: SamplingParams | None
pooling_params: PoolingParams | None
arrival_time: float
lora_request: LoRARequest | None
cache_salt: str | None
data_parallel_rank: int | None
prompt_embeds: torch.Tensor | None = None
prompt_is_token_ids: list[bool] | None = None
client_index: int = 0
current_wave: int = 0
priority: int = 0
trace_headers: Mapping[str, str] | None = None
resumable: bool = False
external_req_id: str | None = None
reasoning_ended: bool | None = None
reasoning_parser_kwargs: dict[str, Any] | None = None
abort_immediately: bool = False
```

位置：`vllm/vllm/v1/engine/__init__.py:94` 到 `vllm/vllm/v1/engine/__init__.py:137`

可以理解为：

```text
EngineCoreRequest 是跨 Engine / EngineCore 边界传输的请求结构。
```

它已经包含：

```text
tokenized prompt
sampling / pooling 参数
LoRA 信息
多模态特征
优先级
client_index
DP wave 信息
streaming / resumable 标记
trace headers
```

### 2.2 `Request`：Scheduler 使用的内部请求对象

EngineCore 真正交给 Scheduler 的不是 `EngineCoreRequest`，而是内部 `Request`。

转换发生在：

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
```

位置：`vllm/vllm/v1/engine/core.py:878`

所以对象关系是：

```text
外层 Engine / AsyncLLM：EngineCoreRequest
EngineCore 内部：Request
Scheduler：Request
```

为什么要转成 `Request`？

因为 Scheduler 需要的状态更多，例如：

```text
Request.status
num_computed_tokens
num_output_placeholders
spec_token_ids
block_hashes
streaming_queue
prefill_stats
structured_output_request
```

这些是调度和状态机运行过程中维护的内部字段，不适合作为外层 API 请求对象直接暴露。

---

## 3. 同步 LLMEngine 请求入口

同步路径中，请求从 `LLMEngine.add_request()` 进入。

入口位置：

```python
def add_request(...):
```

位置：`vllm/vllm/v1/engine/llm_engine.py:218`

### 3.1 外部输入先经过 InputProcessor

如果用户传入的不是 `EngineCoreRequest`，`LLMEngine` 会调用：

```python
request = self.input_processor.process_inputs(
    request_id,
    prompt,
    params,
    supported_tasks=self.get_supported_tasks(),
    arrival_time=arrival_time,
    lora_request=lora_request,
    tokenization_kwargs=tokenization_kwargs,
    trace_headers=trace_headers,
    priority=priority,
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:250` 到 `vllm/vllm/v1/engine/llm_engine.py:260`

这一步负责把外部输入转换成 EngineCore 可以接收的结构。

典型处理包括：

```text
prompt tokenization / prompt embeds 处理
sampling_params / pooling_params 处理
多模态输入处理
LoRA 信息挂载
arrival_time / priority / trace headers 写入
```

### 3.2 assign_request_id

处理后会调用：

```python
self.input_processor.assign_request_id(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:263`

它会处理内部 request id 和外部 request id 的关系。

这对应 `EngineCoreRequest.external_req_id` 字段的注释：

```python
# The user-provided request ID. This field is set internally,
# copied from the provided request_id that's originally assigned
# to the request_id field, see InputProcessor.assign_request_id().
# Used in outputs and to support abort(req_id, internal=False).
external_req_id: str | None = None
```

位置：`vllm/vllm/v1/engine/__init__.py:124` 到 `vllm/vllm/v1/engine/__init__.py:128`

可以理解为：

```text
外部用户 request_id 和内部 request_id 可能不同；
OutputProcessor / abort 需要知道二者关系。
```

### 3.3 加入 OutputProcessor，再送入 EngineCore

如果 `n == 1`，同步路径会：

```python
self.output_processor.add_request(request, prompt_text, None, 0)
self.engine_core.add_request(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:274` 到 `vllm/vllm/v1/engine/llm_engine.py:276`

这里有两个动作：

```text
OutputProcessor.add_request：
  外层记录请求，用于后续把 EngineCoreOutputs 转成 RequestOutput。

EngineCoreClient.add_request：
  把请求真正送进 EngineCore 执行流。
```

所以同步请求入口不是只进 EngineCore，还会同时进入 OutputProcessor 的外层状态管理。

---

## 4. n > 1 时如何进入 EngineCore

如果 sampling params 中 `n > 1`，表示一个用户请求要生成多个候选。

同步路径会创建 `ParentRequest`，再 fan out 成多个 child request：

```python
parent_req = ParentRequest(request)
for idx in range(n):
    request_id, child_params = parent_req.get_child_info(idx)
    child_request = request if idx == n - 1 else copy(request)
    child_request.request_id = request_id
    child_request.sampling_params = child_params
    self.output_processor.add_request(child_request, prompt_text, parent_req, idx)
    self.engine_core.add_request(child_request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:279` 到 `vllm/vllm/v1/engine/llm_engine.py:292`

也就是说：

```text
一个外部请求 n > 1
  → 被拆成多个 EngineCoreRequest child
  → 每个 child 都进入 EngineCore / Scheduler
  → OutputProcessor 负责把多个 child 输出重新归并到 parent request
```

这说明 EngineCore 层看到的是多个独立请求；外层的 parent / child 合并逻辑在 OutputProcessor / ParentRequest 中处理。

---

## 5. 异步 AsyncLLM 请求入口

异步路径中，请求从 `AsyncLLM.add_request()` 进入。

入口位置：

```python
async def add_request(...):
```

位置：`vllm/vllm/v1/engine/async_llm.py:280`

### 5.1 普通异步请求

如果不是 streaming input，逻辑和同步类似：

```python
request = self.input_processor.process_inputs(...)
...
self.input_processor.assign_request_id(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:349` 到 `vllm/vllm/v1/engine/async_llm.py:368`

然后创建输出收集器：

```python
queue = RequestOutputCollector(params.output_kind, request.request_id)
```

位置：`vllm/vllm/v1/engine/async_llm.py:376`

再调用 `_add_request()`：

```python
await self._add_request(request, prompt_text, None, 0, queue)
```

位置：`vllm/vllm/v1/engine/async_llm.py:381` 到 `vllm/vllm/v1/engine/async_llm.py:383`

`_add_request()` 内部：

```python
self.output_processor.add_request(request, prompt, parent_req, index, queue)
await self.engine_core.add_request_async(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:408` 到 `vllm/vllm/v1/engine/async_llm.py:412`

所以异步普通请求主线是：

```text
AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → RequestOutputCollector
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request_async()
```

### 5.2 异步 n > 1

异步路径中 `n > 1` 也会 fan out child requests：

```python
parent_request = ParentRequest(request)
for idx in range(parent_params.n):
    request_id, child_params = parent_request.get_child_info(idx)
    child_request = request if idx == parent_params.n - 1 else copy(request)
    child_request.request_id = request_id
    child_request.sampling_params = child_params
    await self._add_request(child_request, prompt_text, parent_request, idx, queue)
```

位置：`vllm/vllm/v1/engine/async_llm.py:388` 到 `vllm/vllm/v1/engine/async_llm.py:397`

和同步路径一样：

```text
EngineCore 看到多个 child request；
外层 OutputProcessor / RequestOutputCollector 负责输出归并。
```

---

## 6. EngineCoreClient 如何把请求送进 EngineCore

外层 Engine 持有的是 `EngineCoreClient`，而不是一定直接持有 EngineCore。

`EngineCoreClient` 有三种主要实现：

```text
InprocClient：同进程 EngineCore。
SyncMPClient：同步多进程 EngineCore。
AsyncMPClient：异步多进程 EngineCore。
```

位置：`vllm/vllm/v1/engine/core_client.py:71` 到 `vllm/vllm/v1/engine/core_client.py:80`

### 6.1 InprocClient：直接预处理并 add_request

同进程模式下：

```python
req, request_wave = self.engine_core.preprocess_add_request(request)
self.engine_core.add_request(req, request_wave)
```

位置：`vllm/vllm/v1/engine/core_client.py:297` 到 `vllm/vllm/v1/engine/core_client.py:299`

这表示：

```text
EngineCoreRequest
  → EngineCore.preprocess_add_request()
  → Request
  → EngineCore.add_request()
```

没有 socket 通信。

### 6.2 SyncMPClient：通过 input_socket 发送 ADD

多进程同步模式下：

```python
self._send_input(EngineCoreRequestType.ADD, request)
```

位置：`vllm/vllm/v1/engine/core_client.py:886` 到 `vllm/vllm/v1/engine/core_client.py:889`

也就是说，前端进程把 `EngineCoreRequest` 序列化后通过 ZMQ 发给后台 EngineCoreProc。

### 6.3 AsyncMPClient：异步发送 ADD

异步多进程模式下：

```python
request.client_index = self.client_index
await self._send_input(EngineCoreRequestType.ADD, request)
self._ensure_output_queue_task()
```

位置：`vllm/vllm/v1/engine/core_client.py:1121` 到 `vllm/vllm/v1/engine/core_client.py:1124`

这里额外设置了：

```python
request.client_index = self.client_index
```

位置：`vllm/vllm/v1/engine/core_client.py:1122`

作用是：

```text
在多 API server / 多 client 场景下，EngineCoreOutputs 要能返回给正确的 client。
```

---

## 7. EngineCoreProc 如何接收 ADD 请求

后台进程模式下，请求会先进入 ZMQ input socket，再进入 `input_queue`。

### 7.1 input socket 线程反序列化 EngineCoreRequest

`process_input_sockets()` 中，如果请求类型是 ADD：

```python
if request_type == EngineCoreRequestType.ADD:
    req: EngineCoreRequest = add_request_decoder.decode(data_frames)
    try:
        request = self.preprocess_add_request(req)
    except Exception:
        self._handle_request_preproc_error(req)
        continue
```

位置：`vllm/vllm/v1/engine/core.py:1578` 到 `vllm/vllm/v1/engine/core.py:1584`

然后放入 input queue：

```python
self.input_queue.put_nowait((request_type, request))
```

位置：`vllm/vllm/v1/engine/core.py:1595` 到 `vllm/vllm/v1/engine/core.py:1596`

注意：

```text
多进程模式下，preprocess_add_request() 在 input socket 线程中执行，
这样可以和模型 forward 重叠，提高吞吐。
```

这一点在 `preprocess_add_request()` 注释里也写了：

```python
This function could be directly used in input processing thread to allow
request initialization running in parallel with Model forward
```

位置：`vllm/vllm/v1/engine/core.py:864` 到 `vllm/vllm/v1/engine/core.py:869`

### 7.2 busy loop 从 input_queue 取请求

EngineCoreProc 的 busy loop 会调用 `_process_input_queue()`：

```python
req = self.input_queue.get(block=block)
self._handle_client_request(*req)
```

位置：`vllm/vllm/v1/engine/core.py:1294` 到 `vllm/vllm/v1/engine/core.py:1296`

对于 ADD 请求：

```python
elif request_type == EngineCoreRequestType.ADD:
    req, request_wave = request
    if self._reject_add_in_shutdown(req):
        return
    self.add_request(req, request_wave)
```

位置：`vllm/vllm/v1/engine/core.py:1388` 到 `vllm/vllm/v1/engine/core.py:1392`

这里的 `req` 已经是内部 `Request`，不是原始 `EngineCoreRequest`。

---

## 8. EngineCore.preprocess_add_request() 做什么

预处理入口：

```python
def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
```

位置：`vllm/vllm/v1/engine/core.py:864`

它主要做三件事。

### 8.1 多模态 mm receiver cache 处理

如果有多模态特征，并且 EngineCore 有 `mm_receiver_cache`：

```python
if self.mm_receiver_cache is not None and request.mm_features:
    request.mm_features = self.mm_receiver_cache.get_and_update_features(
        request.mm_features
    )
```

位置：`vllm/vllm/v1/engine/core.py:873` 到 `vllm/vllm/v1/engine/core.py:876`

作用是：

```text
处理多模态特征在 EngineCore 侧的接收缓存，
避免跨进程 / tensor IPC 场景下重复传输或引用丢失。
```

### 8.2 EngineCoreRequest 转 Request

核心转换：

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
```

位置：`vllm/vllm/v1/engine/core.py:878`

这一步会把外层请求对象转成 Scheduler 使用的内部请求对象。

其中 `request_block_hasher` 来自 EngineCore 初始化：

```python
self.request_block_hasher = get_request_block_hasher(
    hash_block_size, caching_hash_fn
)
```

位置：`vllm/vllm/v1/engine/core.py:217` 到 `vllm/vllm/v1/engine/core.py:219`

它只在启用 prefix caching 或 KV connector 时创建：

```python
if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
```

位置：`vllm/vllm/v1/engine/core.py:210` 到 `vllm/vllm/v1/engine/core.py:211`

作用是：

```text
给请求生成 block_hashes，供 prefix cache / external KV cache 查询使用。
```

### 8.3 structured output grammar 初始化

如果请求使用结构化输出：

```python
if req.use_structured_output:
    self.structured_output_manager.grammar_init(req)
```

位置：`vllm/vllm/v1/engine/core.py:879` 到 `vllm/vllm/v1/engine/core.py:885`

注释说明：

```python
# grammar_init is only invoked in input processing thread.
# For structured_output_manager, each request is independent and
# grammar compilation is async. Scheduler always checks grammar
# compilation status before scheduling request.
```

位置：`vllm/vllm/v1/engine/core.py:880` 到 `vllm/vllm/v1/engine/core.py:884`

也就是说：

```text
grammar 初始化可以异步进行；
请求进入 Scheduler 后，如果 grammar 还没 ready，Scheduler 会把它放入 skipped_waiting，
状态是 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR。
```

最后返回：

```python
return req, request.current_wave
```

位置：`vllm/vllm/v1/engine/core.py:886`

`current_wave` 用于 DP wave 相关调度。

---

## 9. EngineCore.add_request() 如何交给 Scheduler

真正把内部 `Request` 加入 Scheduler 的入口是：

```python
def add_request(self, request: Request, request_wave: int = 0):
```

位置：`vllm/vllm/v1/engine/core.py:372`

### 9.1 request_id 类型检查

先检查 request id：

```python
if not isinstance(request.request_id, str):
    raise TypeError(...)
```

位置：`vllm/vllm/v1/engine/core.py:378` 到 `vllm/vllm/v1/engine/core.py:382`

### 9.2 pooling task 支持检查

如果是 pooling request，会检查 task 是否支持：

```python
if pooling_params := request.pooling_params:
    supported_pooling_tasks = [
        task for task in self.get_supported_tasks() if task in POOLING_TASKS
    ]

    if pooling_params.task not in supported_pooling_tasks:
        raise ValueError(...)
```

位置：`vllm/vllm/v1/engine/core.py:384` 到 `vllm/vllm/v1/engine/core.py:393`

这说明 EngineCore 会在入 Scheduler 前做一些模型能力校验。

### 9.3 KV / EC transfer params 检查

如果请求带了 KV transfer params，但 Scheduler 没有 KV connector：

```python
if request.kv_transfer_params is not None and (
    not self.scheduler.get_kv_connector()
):
    logger.warning(...)
```

位置：`vllm/vllm/v1/engine/core.py:395` 到 `vllm/vllm/v1/engine/core.py:401`

如果请求带了 EC transfer params，但 Scheduler 没有 EC connector，也会告警：

```python
if request.ec_transfer_params is not None and self.scheduler.get_ec_connector() is None:
    logger.warning(...)
```

位置：`vllm/vllm/v1/engine/core.py:403` 到 `vllm/vllm/v1/engine/core.py:410`

这说明 EngineCore 会检查请求携带的 KV / EC transfer 信息是否能被当前 Scheduler / Connector 支持。

### 9.4 调用 Scheduler.add_request()

最终核心动作：

```python
self.scheduler.add_request(request)
```

位置：`vllm/vllm/v1/engine/core.py:412`

从这里开始，请求就进入 Scheduler 的请求状态管理。

Scheduler 内部会把请求放入：

```text
self.requests
self.waiting 或 self.skipped_waiting
```

这个后续由 Scheduler 文档展开。

### 9.5 abort_immediately 特殊路径

如果请求设置了：

```python
abort_immediately: bool = False
```

位置：`vllm/vllm/v1/engine/__init__.py:133` 到 `vllm/vllm/v1/engine/__init__.py:137`

EngineCore.add_request() 在加进 Scheduler 后会立即 abort：

```python
if request.abort_immediately:
    self.abort_requests([request.request_id])
```

位置：`vllm/vllm/v1/engine/core.py:413` 到 `vllm/vllm/v1/engine/core.py:416`

注释解释：

```python
# Immediately abort so the connector's request_finished hook runs
# to free any pre-admission KV-transfer resources.
```

位置：`vllm/vllm/v1/engine/core.py:414` 到 `vllm/vllm/v1/engine/core.py:416`

这种请求不是为了正常生成，而是为了走标准 request_finished 清理路径，释放 KV transfer 相关资源。

---

## 10. Scheduler.add_request() 之后请求在哪里

EngineCore 只调用：

```python
self.scheduler.add_request(request)
```

后续具体入队由 Scheduler 管。

Scheduler.add_request() 的入口：

```python
def add_request(self, request: Request) -> None:
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:2069`

普通全新请求会：

```python
self._enqueue_waiting_request(request)
self.requests[request.request_id] = request
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:2086` 到 `vllm/vllm/v1/core/sched/scheduler.py:2087`

`_enqueue_waiting_request()` 根据请求状态选择队列：

```python
if self._is_blocked_waiting_status(request.status):
    self.skipped_waiting.add_request(request)
else:
    self.waiting.add_request(request)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1918` 到 `vllm/vllm/v1/core/sched/scheduler.py:1922`

所以请求进入 Scheduler 后可能在：

```text
普通请求：
  self.waiting

blocked waiting 请求：
  self.skipped_waiting

所有 Scheduler 仍持有的请求：
  self.requests
```

常见 blocked 状态包括：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
WAITING_FOR_REMOTE_KVS
WAITING_FOR_STREAMING_REQ
```

---

## 11. abort / finish request 如何进入 EngineCore

外部取消请求不会直接删除 EngineCore 里的对象，而是通过 EngineCore / Scheduler 的 finish path。

### 11.1 同步 LLMEngine abort

同步路径中：

```python
def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
    request_ids = self.output_processor.abort_requests(request_ids, internal)
    self.engine_core.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:212` 到 `vllm/vllm/v1/engine/llm_engine.py:216`

也就是说先让 OutputProcessor 清理外层状态，再通知 EngineCore。

### 11.2 异步 AsyncLLM abort

异步路径中：

```python
all_request_ids = self.output_processor.abort_requests(request_ids, internal)
await self.engine_core.abort_requests_async(all_request_ids)
```

位置：`vllm/vllm/v1/engine/async_llm.py:717` 到 `vllm/vllm/v1/engine/async_llm.py:718`

如果用户取消 async generator，也会调用 abort：

```python
await self.abort(q.request_id, internal=True)
```

位置：`vllm/vllm/v1/engine/async_llm.py:591` 到 `vllm/vllm/v1/engine/async_llm.py:593`

### 11.3 InprocClient abort

同进程 client 直接调用：

```python
self.engine_core.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/core_client.py:301` 到 `vllm/vllm/v1/engine/core_client.py:303`

### 11.4 MPClient abort

多进程同步路径通过 input socket 发 ABORT：

```python
self._send_input(EngineCoreRequestType.ABORT, request_ids)
```

位置：`vllm/vllm/v1/engine/core_client.py:891` 到 `vllm/vllm/v1/engine/core_client.py:893`

异步路径类似：

```python
await self._send_input(EngineCoreRequestType.ABORT, request_ids)
```

位置：`vllm/vllm/v1/engine/core_client.py:1126` 到 `vllm/vllm/v1/engine/core_client.py:1128`

后台 input socket 线程收到 ABORT 后，除了放入 input_queue，还会放入 aborts_queue：

```python
if request_type == EngineCoreRequestType.ABORT:
    self.aborts_queue.put_nowait(request)
```

位置：`vllm/vllm/v1/engine/core.py:1588` 到 `vllm/vllm/v1/engine/core.py:1594`

注释说明：

```python
# Aborts are added to *both* queues, allows us to eagerly
# process aborts while also ensuring ordering in the input
# queue to avoid leaking requests. This is ok because
# aborting in the scheduler is idempotent.
```

位置：`vllm/vllm/v1/engine/core.py:1588` 到 `vllm/vllm/v1/engine/core.py:1593`

### 11.5 EngineCore.abort_requests()

EngineCore 的 abort 入口是：

```python
def abort_requests(self, request_ids: list[str]):
    self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
```

位置：`vllm/vllm/v1/engine/core.py:418` 到 `vllm/vllm/v1/engine/core.py:424`

所以 EngineCore 自己不实现复杂取消逻辑，而是交给 Scheduler：

```text
EngineCore.abort_requests()
  → Scheduler.finish_requests(..., FINISHED_ABORTED)
  → Scheduler 从 running / waiting / skipped_waiting 移除请求
  → 释放资源或延迟释放
  → 后续返回 abort output
```

### 11.6 执行期间 abort 的处理

普通 `step()` 中，模型执行完成后、`update_from_output()` 前，会处理执行期间到达的 abort：

```python
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`vllm/vllm/v1/engine/core.py:510` 到 `vllm/vllm/v1/engine/core.py:515`

`_process_aborts_queue()` 会批量合并 abort ids：

```python
request_ids = []
while not self.aborts_queue.empty():
    ids = self.aborts_queue.get_nowait()
    request_ids.extend((ids,) if isinstance(ids, str) else ids)
self.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/core.py:643` 到 `vllm/vllm/v1/engine/core.py:651`

作用是：

```text
如果请求在模型执行期间被取消，
EngineCore 会在处理 Worker 输出前先把这些 abort 应用到 Scheduler，
避免已经取消的请求继续 append 输出。
```

---

## 12. streaming / resumable 请求入口

streaming input 主要出现在异步路径。

### 12.1 AsyncLLM 检测 AsyncGenerator 输入

如果 prompt 是 `AsyncGenerator`：

```python
if isinstance(prompt, AsyncGenerator):
    return await self._add_streaming_input_request(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:316` 到 `vllm/vllm/v1/engine/async_llm.py:331`

### 12.2 streaming input 先创建 final request

`_add_streaming_input_request()` 中，会先创建一个 final request，用作输入流结束信号：

```python
final_req = self.input_processor.process_inputs(
    request_id=request_id,
    prompt=TokensPrompt(prompt_token_ids=[0]),
    params=sampling_params,
    **inputs,
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:445` 到 `vllm/vllm/v1/engine/async_llm.py:452`

然后：

```python
self.input_processor.assign_request_id(final_req)
internal_req_id = final_req.request_id
```

位置：`vllm/vllm/v1/engine/async_llm.py:453` 到 `vllm/vllm/v1/engine/async_llm.py:454`

这说明 streaming 内部会使用一个 internal request id 来维持同一 session。

### 12.3 每个 input chunk 生成 resumable 请求

后台 `handle_inputs()` 中，每收到一个 chunk，就创建一个请求：

```python
req = self.input_processor.process_inputs(
    request_id=internal_req_id,
    prompt=input_chunk.prompt,
    params=sp,
    resumable=True,
    **inputs,
)
req.external_req_id = request_id
...
await self._add_request(req, prompt_text, None, 0, queue)
```

位置：`vllm/vllm/v1/engine/async_llm.py:468` 到 `vllm/vllm/v1/engine/async_llm.py:483`

关键点：

```text
每个 streaming chunk 使用同一个 internal request_id；
resumable=True；
external_req_id 保存用户看到的 request_id。
```

### 12.4 输入流结束时发送 final request

当 async generator 正常结束时：

```python
await self._add_request(final_req, None, None, 0, queue)
```

位置：`vllm/vllm/v1/engine/async_llm.py:491` 到 `vllm/vllm/v1/engine/async_llm.py:495`

这个 final request 告诉 EngineCore / Scheduler：

```text
这个 streaming input session 不会再有后续 chunk。
```

### 12.5 Scheduler 如何识别同一个 streaming request

EngineCore 仍然只是把请求交给 Scheduler。

Scheduler.add_request() 中会检查 request id 是否已存在：

```python
existing = self.requests.get(request.request_id)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:2070`

如果已存在，表示这是同一 streaming session 的后续输入。Scheduler 会用 `StreamingUpdate.from_request()` 构造更新：已有请求还没停在 `WAITING_FOR_STREAMING_REQ` 时，把 update 追加到 `existing.streaming_queue`；已有请求已经在等待新输入且 update 非空时，立即 `_update_request_as_session()`；final request 则结束该 streaming session。

所以 streaming 请求入口主线是：

```text
AsyncGenerator input
  → 每个 chunk 生成 resumable EngineCoreRequest
  → 使用同一个 internal request_id
  → EngineCore.add_request()
  → Scheduler.add_request()
      → existing request 仍在运行：append 到 streaming_queue
      → existing request 已等待输入：_update_request_as_session()
      → final request：finish streaming session
```

---

## 13. notify_kv_transfer_request_rejected 特殊入口

异步 Engine 里还有一个特殊入口：

```python
async def notify_kv_transfer_request_rejected(...):
```

位置：`vllm/vllm/v1/engine/async_llm.py:723`

它会构造一个带 `abort_immediately=True` 的 `EngineCoreRequest`：

```python
request = EngineCoreRequest(
    request_id=request_id,
    prompt_token_ids=[0],
    ...
    abort_immediately=True,
)
await self.engine_core.add_request_async(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:733` 到 `vllm/vllm/v1/engine/async_llm.py:748`

注释说明：

```python
# Submit a pre-aborted request so the connector's request_finished
# hook runs to free any pre-admission KV-transfer resources
```

位置：`vllm/vllm/v1/engine/async_llm.py:730` 到 `vllm/vllm/v1/engine/async_llm.py:732`

这类请求的目的不是执行模型，而是进入标准 Scheduler / connector 清理路径。

---

## 14. 请求入口的完整流程图

### 14.1 同步普通请求

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → InprocClient:
       EngineCore.preprocess_add_request()
       EngineCore.add_request()
    or MPClient:
       ZMQ ADD
       EngineCoreProc.process_input_sockets()
       EngineCore.preprocess_add_request()
       input_queue
       EngineCore._handle_client_request(ADD)
       EngineCore.add_request()
  → Scheduler.add_request()
  → waiting / skipped_waiting
```

### 14.2 异步普通请求

```text
AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → RequestOutputCollector
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request_async()
  → ZMQ ADD
  → EngineCoreProc.process_input_sockets()
  → EngineCore.preprocess_add_request()
  → input_queue
  → EngineCore._handle_client_request(ADD)
  → EngineCore.add_request()
  → Scheduler.add_request()
```

### 14.3 abort 请求

```text
LLMEngine.abort_request() / AsyncLLM.abort()
  → OutputProcessor.abort_requests()
  → EngineCoreClient.abort_requests() / abort_requests_async()
  → EngineCore.abort_requests()
  → Scheduler.finish_requests(..., FINISHED_ABORTED)
```

多进程下：

```text
ABORT message
  → input socket
  → input_queue + aborts_queue
  → _process_aborts_queue()
  → EngineCore.abort_requests()
```

### 14.4 streaming input 请求

```text
AsyncLLM.add_request(prompt=AsyncGenerator)
  → _add_streaming_input_request()
  → 每个 input chunk:
       InputProcessor.process_inputs(..., resumable=True)
       same internal request_id
       EngineCoreClient.add_request_async()
       Scheduler.add_request()
       existing request_id → append streaming_queue 或更新已有 session
  → input stream 结束:
       发送 final request，结束 streaming session
```

---

## 15. 容易疑惑的点

### 15.1 EngineCoreRequest 是不是 Scheduler 的 Request？

不是。

```text
EngineCoreRequest 是 EngineCore 边界上的输入对象；
Request 是 Scheduler 使用的内部状态对象。
```

转换发生在：

```python
Request.from_engine_core_request(...)
```

位置：`vllm/vllm/v1/engine/core.py:878`

### 15.2 请求是不是直接进入 Scheduler？

从外层看不是直接进入。

主线是：

```text
外部输入
  → InputProcessor
  → EngineCoreClient
  → EngineCore.preprocess_add_request
  → EngineCore.add_request
  → Scheduler.add_request
```

### 15.3 OutputProcessor 为什么也要 add_request？

因为 OutputProcessor 负责外层输出状态。

```text
EngineCore 只返回 EngineCoreOutputs；
OutputProcessor 要知道 request 的外部 id、prompt_text、parent request、stream collector 等信息，
才能把 EngineCoreOutputs 转成 RequestOutput / PoolingRequestOutput。
```

所以请求进入 EngineCore 前，外层通常也会在 OutputProcessor 注册一次。

### 15.4 abort 为什么既进 input_queue 又进 aborts_queue？

多进程 EngineCoreProc 中，ABORT 会进入两个队列：

```text
input_queue：保证和 ADD 等输入事件的顺序关系；
aborts_queue：允许模型执行期间更早处理 abort。
```

因为 Scheduler abort 是幂等的，所以重复进入是可接受的。

### 15.5 streaming input 为什么使用同一个 internal request_id？

因为它表示同一个会话的后续输入 chunk。

Scheduler.add_request() 发现 request id 已存在后，会更新已有 session，而不是创建独立请求。

---

## 16. 从“回答问题”的角度总结

如果要问：

```text
请求如何进入 EngineCore？
```

可以回答：

```text
外层 LLMEngine / AsyncLLM 先用 InputProcessor 把用户输入转成 EngineCoreRequest，
并在 OutputProcessor 中注册外层输出状态；
然后通过 EngineCoreClient 把 EngineCoreRequest 送入 EngineCore。

EngineCore 会在 preprocess_add_request() 中把 EngineCoreRequest 转成内部 Request，
处理多模态 receiver cache、block hash、structured output grammar 初始化等，
最后由 EngineCore.add_request() 调用 Scheduler.add_request()，
让 Scheduler 接管请求队列和调度状态。
```

请求入口的核心公式是：

```text
外层请求
  → EngineCoreRequest
  → Request
  → Scheduler.add_request()
  → waiting / skipped_waiting
```

abort 入口则是：

```text
外层 abort
  → OutputProcessor.abort_requests()
  → EngineCore.abort_requests()
  → Scheduler.finish_requests(..., FINISHED_ABORTED)
```

streaming input 则是：

```text
多个 resumable EngineCoreRequest
  → same internal request_id
  → Scheduler.add_request()
  → 追加到 streaming_queue 或更新已有 session；final request 结束 session
```
