# 02. LLMEngine 负责什么？

源码位置：`vllm/vllm/v1/engine/llm_engine.py`

本问题关注：同步接口里的 `LLMEngine` 如何作为外层 Engine，连接用户输入、`InputProcessor`、`EngineCoreClient` 和 `OutputProcessor`。

---

## 1. 一句话回答

`LLMEngine` 是 vLLM V1 同步路径里的外层兼容 Engine。

它的核心职责不是调度，也不是执行模型，而是：

```text
用户输入 / EngineInput / PromptType
  → InputProcessor
  → EngineCoreRequest
  → EngineCoreClient
  → EngineCore / Scheduler / Worker
  → EngineCoreOutputs
  → OutputProcessor
  → RequestOutput / PoolingRequestOutput
```

一句话：

```text
LLMEngine 负责同步 API 侧的请求接入和输出转换；
EngineCore 负责调度执行闭环；
Scheduler / Worker 才负责真正的调度、KV 管理和模型 forward。
```

源码里类定义也直接说明：

```python
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""
```

位置：`vllm/vllm/v1/engine/llm_engine.py:48` 到 `vllm/vllm/v1/engine/llm_engine.py:49`

所以可以把同步 `LLMEngine` 理解为：

```text
外部同步调用接口
  + 输入预处理入口
  + EngineCoreClient 包装层
  + 输出后处理入口
  + stats / profile / cache / LoRA 等管理方法的转发层
```

---

## 2. LLMEngine 在整体架构中的位置

同步路径里，请求不是直接进入 Scheduler。

更完整的位置是：

```text
调用方
  → LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request
  → Scheduler.add_request()
  → Scheduler.schedule()
  → Worker / ModelRunner
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → LLMEngine.step()
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

这里 `LLMEngine` 处在 EngineCore 外面。

它持有三个关键对象：

```text
InputProcessor：
  把外部输入转成 EngineCoreRequest。

EngineCoreClient：
  把 EngineCoreRequest 送入 EngineCore，并从 EngineCore 取 EngineCoreOutputs。

OutputProcessor：
  把 EngineCoreOutputs 转成用户可见的 RequestOutput / PoolingRequestOutput。
```

所以 `LLMEngine` 更像“同步前端门面”：

```text
LLMEngine 不直接管理 waiting / running；
LLMEngine 不直接分配 KV blocks；
LLMEngine 不直接执行 forward；
LLMEngine 负责把同步 API 和 EngineCore 内部执行流接起来。
```

---

## 3. 初始化阶段：创建三类核心组件

`LLMEngine.__init__()` 入口：

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    ...
) -> None:
```

位置：`vllm/vllm/v1/engine/llm_engine.py:51` 到 `vllm/vllm/v1/engine/llm_engine.py:61`

初始化主线可以概括为：

```text
LLMEngine.__init__()
  → 保存 vllm_config / model_config / observability_config
  → 如有 tracing endpoint，初始化 tracer
  → 如有 DP 同步需求，初始化 dp_group
  → 创建 renderer
  → 创建 InputProcessor
  → 创建 OutputProcessor
  → 创建 EngineCoreClient
  → 如开启 stats，创建 StatLoggerManager
  → 非多进程模式下暴露 model_executor 兼容 v0
  → 清理 multimodal cache
```

### 3.1 创建 renderer

初始化时先创建 renderer：

```python
self.renderer = renderer = renderer_from_config(self.vllm_config)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:91`

renderer 的作用主要是支持输入渲染、tokenizer、多模态 cache 等前端处理能力。

后面 `InputProcessor` 和 `OutputProcessor` 都会依赖它。

### 3.2 创建 InputProcessor

源码：

```python
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:93` 到 `vllm/vllm/v1/engine/llm_engine.py:94`

注释已经说明它的职责：

```text
EngineInput → EngineCoreRequest
```

也就是把外部 prompt、参数、LoRA、多模态输入等，规范化为 EngineCore 可以接收的 `EngineCoreRequest`。

### 3.3 创建 OutputProcessor

源码：

```python
# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(
    renderer.tokenizer,
    log_stats=self.log_stats,
    stream_interval=self.vllm_config.scheduler_config.stream_interval,
    tracing_enabled=tracing_endpoint is not None,
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:96` 到 `vllm/vllm/v1/engine/llm_engine.py:102`

注释也直接说明它的职责：

```text
EngineCoreOutputs → RequestOutput
```

OutputProcessor 会保存每个请求的外层状态，例如：

```text
external request id
internal request id
prompt text
detokenizer
logprobs processor
parallel sampling parent request
stream interval
stats
async queue（异步路径使用）
```

同步 `LLMEngine.step()` 最终返回的就是它处理后的 `RequestOutput` / `PoolingRequestOutput`。

### 3.4 创建 EngineCoreClient

源码：

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

位置：`vllm/vllm/v1/engine/llm_engine.py:104` 到 `vllm/vllm/v1/engine/llm_engine.py:111`

这说明同步 `LLMEngine` 并不总是直接持有 `EngineCore`，而是持有 `EngineCoreClient`。

`EngineCoreClient` 根据模式可能是：

```text
InprocClient：
  EngineCore 和 LLMEngine 在同一进程。

SyncMPClient：
  同步 API + 多进程 EngineCore。
```

两者都对 `LLMEngine` 暴露统一接口：

```text
add_request()
get_output()
abort_requests()
reset_prefix_cache()
add_lora()
...
```

---

## 4. 初始化时的 DP 和多进程差异

`LLMEngine` 初始化里有一段 DP 相关逻辑：

```python
self.external_launcher_dp = (
    parallel_config.data_parallel_size > 1
    and executor_backend == "external_launcher"
)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:75` 到 `vllm/vllm/v1/engine/llm_engine.py:78`

如果不是多进程，并且 DP size > 1，还会先初始化 DP group：

```python
if (
    not multiprocess_mode
    and parallel_config.data_parallel_size > 1
    and not self.external_launcher_dp
):
    self.dp_group = parallel_config.stateless_init_dp_group()
else:
    self.dp_group = None
```

位置：`vllm/vllm/v1/engine/llm_engine.py:81` 到 `vllm/vllm/v1/engine/llm_engine.py:88`

注释强调：

```python
# important: init dp group before init the engine_core
# In the decoupled engine case this is handled in EngineCoreProc.
```

位置：`vllm/vllm/v1/engine/llm_engine.py:79` 到 `vllm/vllm/v1/engine/llm_engine.py:80`

含义是：

```text
非 multiprocess 且 data_parallel_size > 1 且不是 external_launcher_dp：
  LLMEngine 在创建 EngineCoreClient 前调用 parallel_config.stateless_init_dp_group()。

external_launcher_dp：
  LLMEngine 在创建 EngineCoreClient 后复用 get_dp_group().cpu_group。

multiprocess / decoupled EngineCore：
  EngineCoreProc 自己处理 DP 初始化。
```

所以 `LLMEngine` 虽然不做调度，但它负责在同步前端侧协调一些运行模式初始化。

---

## 5. from_engine_args / from_vllm_config：构造入口

同步 `LLMEngine` 常见创建路径有两个。

### 5.1 from_vllm_config

```python
@classmethod
def from_vllm_config(...):
    return cls(
        vllm_config=vllm_config,
        executor_class=Executor.get_class(vllm_config),
        log_stats=(not disable_log_stats),
        ...
        multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
    )
```

位置：`vllm/vllm/v1/engine/llm_engine.py:143` 到 `vllm/vllm/v1/engine/llm_engine.py:158`

它从已有 `VllmConfig` 创建 `LLMEngine`，并根据环境变量决定是否启用 V1 multiprocessing。

### 5.2 from_engine_args

```python
def from_engine_args(...):
    vllm_config = engine_args.create_engine_config(usage_context)
    executor_class = Executor.get_class(vllm_config)
    ...
    return cls(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:160` 到 `vllm/vllm/v1/engine/llm_engine.py:186`

这一路径先把 CLI / API 层的 `EngineArgs` 转成 `VllmConfig`，再创建 Engine。

所以创建路径可以概括为：

```text
EngineArgs
  → create_engine_config()
  → VllmConfig
  → Executor.get_class()
  → LLMEngine.__init__()
```

---

## 6. add_request：同步请求入口

同步请求从：

```python
def add_request(
    self,
    request_id: str,
    prompt: EngineCoreRequest | PromptType | EngineInput,
    params: SamplingParams | PoolingParams,
    ...
) -> str:
```

位置：`vllm/vllm/v1/engine/llm_engine.py:218` 到 `vllm/vllm/v1/engine/llm_engine.py:229`

整体主线是：

```text
LLMEngine.add_request()
  → 校验 request_id 必须是 str
  → 如果 prompt 已经是 EngineCoreRequest，直接使用（兼容旧路径）
  → 否则调用 InputProcessor.process_inputs()
  → extract_prompt_components() 提取 prompt_text
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
  → 返回内部 req_id
```

源码入口先校验 `request_id`：

```python
if not isinstance(request_id, str):
    raise TypeError(f"request_id must be a string, got {type(request_id)}")
```

位置：`vllm/vllm/v1/engine/llm_engine.py:230` 到 `vllm/vllm/v1/engine/llm_engine.py:232`

也就是说同步路径要求外部请求 id 必须是字符串。

---

## 7. add_request 第一步：处理外部输入

`LLMEngine.add_request()` 支持两类输入。

### 7.1 已经是 EngineCoreRequest

如果传入的 `prompt` 已经是 `EngineCoreRequest`：

```python
if isinstance(prompt, EngineCoreRequest):
    logger.warning_once(...)
    request = prompt
```

位置：`vllm/vllm/v1/engine/llm_engine.py:235` 到 `vllm/vllm/v1/engine/llm_engine.py:242`

这是一条兼容旧用法的路径。

如果参数里的 `request_id` 和 `EngineCoreRequest.request_id` 不一致，会提示以后者为准：

```python
if request_id != request.request_id:
    logger.warning_once(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:243` 到 `vllm/vllm/v1/engine/llm_engine.py:248`

### 7.2 普通 PromptType / EngineInput

如果不是 `EngineCoreRequest`，就进入 `InputProcessor`：

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
prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:250` 到 `vllm/vllm/v1/engine/llm_engine.py:261`

这一步把用户输入变成 `EngineCoreRequest`。

从 `InputProcessor.process_inputs()` 看，它会做这些事情：

```text
1. 校验 SamplingParams / PoolingParams；
2. 校验 LoRA 是否启用；
3. 校验 data_parallel_rank 范围；
4. 如果输入已经是 EngineInput，直接使用；
5. 如果是 raw prompt，通过 InputPreprocessor 预处理；
6. 调用 current_platform.validate_request()；
7. 拆分 encoder / decoder inputs；
8. 校验模型输入长度和类型；
9. 复制并补全 SamplingParams / PoolingParams；
10. 提取多模态 mm_features；
11. 构造 EngineCoreRequest。
```

对应关键源码：

```python
self._validate_params(params, supported_tasks)
self._validate_lora(lora_request)
```

位置：`vllm/vllm/v1/engine/input_processor.py:256` 到 `vllm/vllm/v1/engine/input_processor.py:257`

raw prompt 会经过预处理：

```python
processed_inputs = self.input_preprocessor.preprocess(
    prompt,
    tokenization_kwargs=tokenization_kwargs,
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:291` 到 `vllm/vllm/v1/engine/input_processor.py:294`

然后校验并拆分输入：

```python
current_platform.validate_request(processed_inputs, params)
encoder_inputs, decoder_inputs = split_enc_dec_input(processed_inputs)
self._validate_model_inputs(encoder_inputs, decoder_inputs)
```

位置：`vllm/vllm/v1/engine/input_processor.py:296` 到 `vllm/vllm/v1/engine/input_processor.py:299`

最终返回 `EngineCoreRequest`：

```python
return EngineCoreRequest(
    request_id=request_id,
    prompt_token_ids=prompt_token_ids,
    prompt_embeds=prompt_embeds,
    prompt_is_token_ids=prompt_is_token_ids,
    mm_features=mm_features,
    sampling_params=sampling_params,
    pooling_params=pooling_params,
    arrival_time=arrival_time,
    lora_request=lora_request,
    cache_salt=decoder_inputs.get("cache_salt"),
    priority=priority,
    data_parallel_rank=data_parallel_rank,
    trace_headers=trace_headers,
    resumable=resumable,
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:370` 到 `vllm/vllm/v1/engine/input_processor.py:385`

因此，`LLMEngine` 本身不做 tokenization / multimodal 细节，而是委托给 `InputProcessor`。

---

## 8. EngineCoreRequest：LLMEngine 送入 EngineCore 的对象

`EngineCoreRequest` 定义在：

```python
class EngineCoreRequest(msgspec.Struct, ...):
```

位置：`vllm/vllm/v1/engine/__init__.py:86` 到 `vllm/vllm/v1/engine/__init__.py:91`

它的主要字段包括：

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

位置：`vllm/vllm/v1/engine/__init__.py:92` 到 `vllm/vllm/v1/engine/__init__.py:135`

可以理解为：

```text
EngineCoreRequest 是 LLMEngine / AsyncLLM 和 EngineCore 之间的边界对象。
```

它已经包含：

```text
tokenized prompt 或 prompt embeds
sampling / pooling params
多模态特征
LoRA 信息
arrival_time
priority
trace headers
DP rank / wave
client_index
external request id
```

但它还不是 Scheduler 内部的 `Request`。

对象层次是：

```text
LLMEngine 外层：EngineCoreRequest
EngineCore 内部：Request
Scheduler 状态机：Request
```

`EngineCoreRequest → Request` 的转换发生在 EngineCore 的 `preprocess_add_request()`，不是在 `LLMEngine` 里。

---

## 9. assign_request_id：外部 id 和内部 id 的分离

`InputProcessor.process_inputs()` 返回后，`LLMEngine.add_request()` 会调用：

```python
self.input_processor.assign_request_id(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:263`

`assign_request_id()` 的默认逻辑是：

```python
request.external_req_id = request.request_id
...
request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
```

位置：`vllm/vllm/v1/engine/input_processor.py:232` 到 `vllm/vllm/v1/engine/input_processor.py:240`

它的注释说明默认会给内部 request id 增加 8 位随机后缀，以保证内部唯一性；如果设置了 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`，则不会追加随机后缀。

所以同步路径中有两个 id：

```text
external_req_id：
  用户传入的 request_id，用于输出给用户，也用于外部 abort。

request_id：
  vLLM 内部 request id，默认会追加随机后缀，用于 EngineCore / Scheduler 内部状态。
```

`EngineCoreRequest` 里也有对应注释：

```python
# The user-provided request ID. This field is set internally,
# copied from the provided request_id that's originally assigned
# to the request_id field, see InputProcessor.assign_request_id().
# Used in outputs and to support abort(req_id, internal=False).
external_req_id: str | None = None
```

位置：`vllm/vllm/v1/engine/__init__.py:122` 到 `vllm/vllm/v1/engine/__init__.py:126`

这解释了一个常见疑问：

```text
LLMEngine.add_request() 返回的 req_id 是内部 id；
用户可见输出里的 request_id 通常是 external_req_id。
```

---

## 10. 普通 n == 1 请求如何进入 EngineCore

`LLMEngine.add_request()` 处理完输入后会取出：

```python
req_id = request.request_id
params = request.params
n = params.n if isinstance(params, SamplingParams) else 1
```

位置：`vllm/vllm/v1/engine/llm_engine.py:265` 到 `vllm/vllm/v1/engine/llm_engine.py:270`

如果 `n == 1`：

```python
if n == 1:
    # Make a new RequestState and queue.
    self.output_processor.add_request(request, prompt_text, None, 0)
    # Add the request to EngineCore.
    self.engine_core.add_request(request)
    return req_id
```

位置：`vllm/vllm/v1/engine/llm_engine.py:272` 到 `vllm/vllm/v1/engine/llm_engine.py:277`

这里一定要注意顺序：

```text
先 OutputProcessor.add_request()
再 EngineCoreClient.add_request()
```

原因是：

```text
EngineCore 后续只返回 EngineCoreOutput；
OutputProcessor 必须提前知道 request 的 prompt、外部 id、detokenizer、logprobs processor 等外层状态，
才能把 EngineCoreOutput 转回 RequestOutput。
```

所以普通请求进入同步 Engine 的主线是：

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()
```

---

## 11. OutputProcessor.add_request() 做什么

`OutputProcessor.add_request()` 入口：

```python
def add_request(
    self,
    request: EngineCoreRequest,
    prompt: str | None,
    parent_req: ParentRequest | None = None,
    request_index: int = 0,
    queue: RequestOutputCollector | None = None,
) -> None:
```

位置：`vllm/vllm/v1/engine/output_processor.py:512` 到 `vllm/vllm/v1/engine/output_processor.py:519`

如果 request 已存在，说明是 streaming update，会走更新逻辑：

```python
req_state = self.request_states.get(request_id)
if req_state is not None:
    self._update_streaming_request_state(req_state, request, prompt)
    return
```

位置：`vllm/vllm/v1/engine/output_processor.py:520` 到 `vllm/vllm/v1/engine/output_processor.py:524`

普通新请求会创建 `RequestState`：

```python
req_state = RequestState.from_new_request(
    tokenizer=self.tokenizer,
    request=request,
    prompt=prompt,
    parent_req=parent_req,
    request_index=request_index,
    queue=queue,
    log_stats=self.log_stats,
    stream_interval=self.stream_interval,
)
self.request_states[request_id] = req_state
```

位置：`vllm/vllm/v1/engine/output_processor.py:526` 到 `vllm/vllm/v1/engine/output_processor.py:536`

然后记录 external id 到 internal id 的映射：

```python
self.external_req_ids[req_state.external_req_id].append(request_id)
```

位置：`vllm/vllm/v1/engine/output_processor.py:540` 到 `vllm/vllm/v1/engine/output_processor.py:541`

所以 OutputProcessor 的请求注册不是“把请求送去执行”，而是建立输出处理需要的外层状态。

`RequestState.from_new_request()` 会根据请求类型创建不同状态：

```text
SamplingParams：
  创建 LogprobsProcessor、IncrementalDetokenizer，记录 max_tokens / top_p / n / temperature。

PoolingParams：
  不创建 detokenizer，记录 pooling output kind。
```

相关位置：`vllm/vllm/v1/engine/output_processor.py:211` 到 `vllm/vllm/v1/engine/output_processor.py:270`

---

## 12. EngineCoreClient.add_request()：真正送入 EngineCore

`LLMEngine` 调用的是：

```python
self.engine_core.add_request(request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:276`

这里的 `self.engine_core` 是 `EngineCoreClient`。

### 12.1 InprocClient

同进程模式下：

```python
def add_request(self, request: EngineCoreRequest) -> None:
    req, request_wave = self.engine_core.preprocess_add_request(request)
    self.engine_core.add_request(req, request_wave)
```

位置：`vllm/vllm/v1/engine/core_client.py:297` 到 `vllm/vllm/v1/engine/core_client.py:299`

也就是：

```text
EngineCoreRequest
  → EngineCore.preprocess_add_request()
  → Request
  → EngineCore.add_request()
  → Scheduler.add_request()
```

### 12.2 SyncMPClient

多进程同步模式下：

```python
def add_request(self, request: EngineCoreRequest) -> None:
    if self.is_dp:
        self.engines_running = True
    self._send_input(EngineCoreRequestType.ADD, request)
```

位置：`vllm/vllm/v1/engine/core_client.py:886` 到 `vllm/vllm/v1/engine/core_client.py:889`

也就是把 `EngineCoreRequest` 序列化后发给后台 EngineCoreProc。

同步 `LLMEngine` 不关心底层是同进程还是多进程，它只依赖统一接口：

```text
EngineCoreClient.add_request(request)
```

---

## 13. n > 1：parallel sampling 如何 fan out

如果 sampling params 中 `n > 1`，表示一个用户请求要生成多个候选。

`LLMEngine.add_request()` 会创建 `ParentRequest`，然后拆成多个 child request：

```python
# Fan out child requests (for n>1).
parent_req = ParentRequest(request)
for idx in range(n):
    request_id, child_params = parent_req.get_child_info(idx)
    child_request = request if idx == n - 1 else copy(request)
    child_request.request_id = request_id
    child_request.sampling_params = child_params

    # Make a new RequestState and queue.
    self.output_processor.add_request(
        child_request, prompt_text, parent_req, idx
    )
    # Add the request to EngineCore.
    self.engine_core.add_request(child_request)

return req_id
```

位置：`vllm/vllm/v1/engine/llm_engine.py:279` 到 `vllm/vllm/v1/engine/llm_engine.py:294`

含义是：

```text
一个外部请求 n > 1
  → ParentRequest 记录父请求
  → fan out 成 n 个 child EngineCoreRequest
  → 每个 child 都注册到 OutputProcessor
  → 每个 child 都送入 EngineCore
  → EngineCore / Scheduler 把它们当独立请求调度
  → OutputProcessor / ParentRequest 再把 child outputs 合并成一个外部 RequestOutput
```

所以：

```text
parallel sampling 的拆分发生在 LLMEngine 层；
parallel sampling 的执行发生在 EngineCore / Scheduler 层；
parallel sampling 的合并发生在 OutputProcessor / ParentRequest 层。
```

---

## 14. step：同步输出入口

同步 `LLMEngine.step()` 入口：

```python
def step(self) -> list[RequestOutput | PoolingRequestOutput]:
```

位置：`vllm/vllm/v1/engine/llm_engine.py:296`

主流程是：

```text
LLMEngine.step()
  → 如需要 DP dummy batch，则 execute_dummy_batch() 并返回 []
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → 如 stop string 由 detokenizer 检出，abort EngineCore 中对应请求
  → 记录 stats
  → 返回 RequestOutput / PoolingRequestOutput 列表
```

源码结构很清楚：

```python
# 1) Get EngineCoreOutput from the EngineCore.
outputs = self.engine_core.get_output()

# 2) Process EngineCoreOutputs.
processed_outputs = self.output_processor.process_outputs(...)

# 3) Abort any reqs that finished due to stop strings.
self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

# 4) Record stats
...

return processed_outputs.request_outputs
```

位置：`vllm/vllm/v1/engine/llm_engine.py:302` 到 `vllm/vllm/v1/engine/llm_engine.py:334`

一句话：

```text
LLMEngine.step() 不是直接跑模型；它从 EngineCoreClient 拉取 EngineCoreOutputs，再交给 OutputProcessor 转成同步 API 返回值。
```

---

## 15. step 第 0 步：DP dummy batch

`step()` 开头先处理一种 DP 特殊情况：

```python
if self.should_execute_dummy_batch:
    self.should_execute_dummy_batch = False
    self.engine_core.execute_dummy_batch()
    return []
```

位置：`vllm/vllm/v1/engine/llm_engine.py:297` 到 `vllm/vllm/v1/engine/llm_engine.py:300`

`should_execute_dummy_batch` 来自 `has_unfinished_requests_dp()`：

```python
if not has_unfinished and aggregated_has_unfinished:
    self.should_execute_dummy_batch = True
```

位置：`vllm/vllm/v1/engine/llm_engine.py:201` 到 `vllm/vllm/v1/engine/llm_engine.py:202`

含义是：

```text
本地 OutputProcessor 看起来没有 unfinished 请求，
但 DP 全局还有其它 rank 未完成请求，
当前 rank 可能需要执行 dummy batch 来维持 DP / collective 同步。
```

这和 EngineCore 文档里的 DP dummy batch 逻辑对应。

---

## 16. step 第 1 步：从 EngineCoreClient 获取 EngineCoreOutputs

同步 step 的第一步：

```python
with record_function_or_nullcontext("llm_engine step: get_output"):
    outputs = self.engine_core.get_output()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:302` 到 `vllm/vllm/v1/engine/llm_engine.py:304`

### 16.1 InprocClient.get_output()

同进程模式下：

```python
def get_output(self) -> EngineCoreOutputs:
    outputs, model_executed = self.engine_core.step_fn()
    self.engine_core.post_step(model_executed=model_executed)
    return outputs and outputs.get(0) or EngineCoreOutputs()
```

位置：`vllm/vllm/v1/engine/core_client.py:289` 到 `vllm/vllm/v1/engine/core_client.py:292`

也就是说：

```text
LLMEngine.step()
  → InprocClient.get_output()
  → EngineCore.step_fn()
  → Scheduler.schedule()
  → Worker / ModelRunner
  → Scheduler.update_from_output()
  → EngineCoreOutputs
```

同进程模式下，`LLMEngine.step()` 间接驱动了一轮 EngineCore 执行。

### 16.2 SyncMPClient.get_output()

多进程同步模式下：

```python
def get_output(self) -> EngineCoreOutputs:
    outputs = self.outputs_queue.get()
    ...
    if outputs.wave_complete is not None:
        self.engines_running = False
    return outputs
```

位置：`vllm/vllm/v1/engine/core_client.py:849` 到 `vllm/vllm/v1/engine/core_client.py:859`

也就是说：

```text
LLMEngine.step()
  → SyncMPClient.get_output()
  → 从 outputs_queue 阻塞读取后台 EngineCoreProc 已产生的 EngineCoreOutputs
```

后台 EngineCoreProc 自己用 busy loop 调用 `step_fn()`，同步前端只负责取输出。

### 16.3 两种模式对 LLMEngine 是透明的

对 `LLMEngine.step()` 来说，两种模式都只是：

```text
outputs = self.engine_core.get_output()
```

差异被封装在 `EngineCoreClient` 内部。

---

## 17. step 第 2 步：OutputProcessor.process_outputs()

拿到 `EngineCoreOutputs` 后，`LLMEngine.step()` 调用：

```python
iteration_stats = IterationStats() if self.log_stats else None
processed_outputs = self.output_processor.process_outputs(
    outputs.outputs,
    engine_core_timestamp=outputs.timestamp,
    iteration_stats=iteration_stats,
)
self.output_processor.update_scheduler_stats(outputs.scheduler_stats)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:308` 到 `vllm/vllm/v1/engine/llm_engine.py:314`

`OutputProcessor.process_outputs()` 的入口：

```python
def process_outputs(
    self,
    engine_core_outputs: list[EngineCoreOutput],
    engine_core_timestamp: float | None = None,
    iteration_stats: IterationStats | None = None,
) -> OutputProcessorOutput:
```

位置：`vllm/vllm/v1/engine/output_processor.py:576` 到 `vllm/vllm/v1/engine/output_processor.py:581`

源码注释概括它做三件事：

```python
"""
Process the EngineCoreOutputs:
1) Compute stats for logging
2) Detokenize
3) Create and handle RequestOutput objects:
...
"""
```

位置：`vllm/vllm/v1/engine/output_processor.py:582` 到 `vllm/vllm/v1/engine/output_processor.py:592`

更完整的流程是：

```text
OutputProcessor.process_outputs()
  → 遍历 EngineCoreOutput
  → 找到对应 RequestState
  → 更新 stats
  → 取 new_token_ids / pooling_output / finish_reason / stop_reason
  → 更新 routed experts 累积
  → 如果是第一次 prefill 完成，记录 num_cached_tokens
  → detokenizer.update() 把 token 转成 text
  → 检查 stop string
  → logprobs_processor.update_from_output()
  → RequestState.make_request_output()
  → 同步路径：append 到 request_outputs
  → 异步路径：put 到 queue
  → 如果请求 finished，清理 RequestState
  → 如果 Detokenizer 检出 stop string 但 EngineCore 未 finished，记录 reqs_to_abort
```

---

## 18. EngineCoreOutput 如何变成 RequestOutput

在 `process_outputs()` 循环里，先取出 EngineCore 输出字段：

```python
new_token_ids = engine_core_output.new_token_ids
pooling_output = engine_core_output.pooling_output
finish_reason = engine_core_output.finish_reason
stop_reason = engine_core_output.stop_reason
kv_transfer_params = engine_core_output.kv_transfer_params
```

位置：`vllm/vllm/v1/engine/output_processor.py:618` 到 `vllm/vllm/v1/engine/output_processor.py:622`

如果是 generation 输出，会先 detokenize：

```python
stop_string = req_state.detokenizer.update(
    new_token_ids, finish_reason == FinishReason.STOP
)
if stop_string:
    finish_reason = FinishReason.STOP
    stop_reason = stop_string
```

位置：`vllm/vllm/v1/engine/output_processor.py:639` 到 `vllm/vllm/v1/engine/output_processor.py:645`

然后更新 logprobs：

```python
req_state.logprobs_processor.update_from_output(engine_core_output)
```

位置：`vllm/vllm/v1/engine/output_processor.py:646` 到 `vllm/vllm/v1/engine/output_processor.py:648`

最后创建外部输出：

```python
if request_output := req_state.make_request_output(
    new_token_ids,
    pooling_output,
    finish_reason,
    stop_reason,
    kv_transfer_params,
):
    ...
```

位置：`vllm/vllm/v1/engine/output_processor.py:650` 到 `vllm/vllm/v1/engine/output_processor.py:657`

同步 `LLMEngine` 没有 queue，所以会加入返回列表：

```python
request_outputs.append(request_output)
```

位置：`vllm/vllm/v1/engine/output_processor.py:664` 到 `vllm/vllm/v1/engine/output_processor.py:666`

`RequestState.make_request_output()` 里会根据类型返回：

```text
pooling_output is not None：
  返回 PoolingRequestOutput。

普通 generation：
  返回 RequestOutput，里面包含 CompletionOutput。

parent_req is not None：
  通过 ParentRequest 合并多个 child outputs。
```

关键位置：`vllm/vllm/v1/engine/output_processor.py:272` 到 `vllm/vllm/v1/engine/output_processor.py:331`

---

## 19. Detokenizer 检出 stop string 后为什么还要 abort EngineCore

`LLMEngine.step()` 在 `process_outputs()` 后执行：

```python
with record_function_or_nullcontext("llm_engine step: abort_requests"):
    self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:317` 到 `vllm/vllm/v1/engine/llm_engine.py:319`

`reqs_to_abort` 来自 OutputProcessor：

```python
if not engine_core_output.finished:
    # If req not finished in EngineCore, but Detokenizer
    # detected stop string, abort needed in EngineCore.
    reqs_to_abort.append(req_id)
```

位置：`vllm/vllm/v1/engine/output_processor.py:678` 到 `vllm/vllm/v1/engine/output_processor.py:681`

为什么需要这样？

```text
EngineCore / Scheduler 能基于 stop token、EOS、max_tokens 等条件停止请求；
但 stop string 需要 detokenize 成文本后才能判断，发生在 OutputProcessor。
```

因此可能出现：

```text
Scheduler 认为请求还没 finished；
OutputProcessor detokenize 后发现文本命中了 stop string；
LLMEngine 必须反向通知 EngineCore abort 这个内部请求。
```

这就是同步 step 中“输出处理反向影响 EngineCore”的少数路径之一。

---

## 20. abort_request：同步取消入口

同步取消入口是：

```python
def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
    """Remove request_ids from EngineCore and Detokenizer."""

    request_ids = self.output_processor.abort_requests(request_ids, internal)
    self.engine_core.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:212` 到 `vllm/vllm/v1/engine/llm_engine.py:216`

这里也体现了 `LLMEngine` 的桥接职责：

```text
先清理 OutputProcessor 的外层状态；
再把需要取消的内部 request ids 发给 EngineCore。
```

`OutputProcessor.abort_requests()` 支持两种 id：

```text
internal=False：
  传入 external request id；
  OutputProcessor 根据 external_req_ids 找到所有 internal ids。

internal=True：
  传入 internal request id；
  也可能是 parent request id。
```

相关逻辑位置：`vllm/vllm/v1/engine/output_processor.py:450` 到 `vllm/vllm/v1/engine/output_processor.py:510`

取消时还会给异步 queue 生成 abort final output：

```python
req_state.queue.put(request_output)
```

位置：`vllm/vllm/v1/engine/output_processor.py:489` 到 `vllm/vllm/v1/engine/output_processor.py:502`

同步路径没有 queue，主要是清理状态并返回需要 EngineCore abort 的内部 ids。

---

## 21. has_unfinished_requests：同步循环如何判断是否继续 step

同步外层通常会循环调用：

```text
while engine.has_unfinished_requests():
    outputs = engine.step()
```

`LLMEngine.has_unfinished_requests()` 是：

```python
def has_unfinished_requests(self) -> bool:
    has_unfinished = self.output_processor.has_unfinished_requests()
    if self.dp_group is None:
        return has_unfinished or self.engine_core.dp_engines_running()
    return self.has_unfinished_requests_dp(has_unfinished)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:191` 到 `vllm/vllm/v1/engine/llm_engine.py:195`

这里判断的不是 Scheduler 内部状态，而是外层 OutputProcessor 是否还有未完成请求。

`OutputProcessor.has_unfinished_requests()` 很简单：

```python
def has_unfinished_requests(self) -> bool:
    return len(self.request_states) > 0
```

位置：`vllm/vllm/v1/engine/output_processor.py:440` 到 `vllm/vllm/v1/engine/output_processor.py:441`

所以同步 API 视角的 unfinished 是：

```text
OutputProcessor 里还有 RequestState 没被清理。
```

但 DP 场景会额外考虑全局未完成状态和 dummy batch。

---

## 22. get_num_unfinished_requests：只看 OutputProcessor

`get_num_unfinished_requests()`：

```python
def get_num_unfinished_requests(self) -> int:
    return self.output_processor.get_num_unfinished_requests()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:188` 到 `vllm/vllm/v1/engine/llm_engine.py:189`

`OutputProcessor` 返回：

```python
def get_num_unfinished_requests(self):
    return len(self.request_states)
```

位置：`vllm/vllm/v1/engine/output_processor.py:437` 到 `vllm/vllm/v1/engine/output_processor.py:438`

这再次说明：

```text
LLMEngine 的同步请求完成状态，是以外层输出状态为准。
```

Scheduler 内部可能还有 finished 但等待 deferred free / connector cleanup 的请求，这不一定等同于同步 API 视角的 unfinished request 数。

---

## 23. stats 记录：LLMEngine 只做外层聚合

`LLMEngine.step()` 处理完输出和 abort 后，会记录 stats：

```python
if (
    self.logger_manager is not None
    and outputs.scheduler_stats is not None
    and len(outputs.outputs) > 0
):
    self.logger_manager.record(
        scheduler_stats=outputs.scheduler_stats,
        iteration_stats=iteration_stats,
        mm_cache_stats=self.renderer.stat_mm_cache(),
    )
    self.do_log_stats_with_interval()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:321` 到 `vllm/vllm/v1/engine/llm_engine.py:332`

这里的 stats 来源包括：

```text
scheduler_stats：
  EngineCore / Scheduler 返回。

iteration_stats：
  OutputProcessor 处理输出时统计。

mm_cache_stats：
  renderer 的多模态 cache 统计。
```

所以 LLMEngine 不生产 Scheduler 的核心 stats，只是在同步前端侧聚合、记录。

---

## 24. cache / sleep / profile / LoRA 等方法：多数是转发

`LLMEngine` 还有一批管理方法，例如：

```python
def start_profile(self, profile_prefix: str | None = None):
    self.engine_core.profile(True, profile_prefix)

def stop_profile(self):
    self.engine_core.profile(False)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:336` 到 `vllm/vllm/v1/engine/llm_engine.py:340`

多模态 cache 重置：

```python
def reset_mm_cache(self):
    self.renderer.clear_mm_cache()
    self.engine_core.reset_mm_cache()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:342` 到 `vllm/vllm/v1/engine/llm_engine.py:344`

prefix cache 重置：

```python
def reset_prefix_cache(...):
    return self.engine_core.reset_prefix_cache(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:346` 到 `vllm/vllm/v1/engine/llm_engine.py:351`

sleep / wake up：

```python
def sleep(self, level: int = 1, mode: PauseMode = "abort"):
    if level >= 1:
        self.renderer.clear_mm_cache()
    self.engine_core.sleep(level, mode)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:361` 到 `vllm/vllm/v1/engine/llm_engine.py:364`

LoRA：

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.engine_core.add_lora(lora_request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:403` 到 `vllm/vllm/v1/engine/llm_engine.py:405`

这类方法可以概括为：

```text
LLMEngine 负责同步 API 暴露；
具体能力大多转发给 EngineCoreClient / EngineCore。
```

但有些外层资源也会一起处理，例如：

```text
reset_mm_cache / sleep(level >= 1)：
  先清理 renderer 侧 mm cache，再通知 EngineCore。
```

---

## 25. tokenizer 访问：来自 renderer

`LLMEngine` 的 tokenizer 属性：

```python
@property
def tokenizer(self) -> TokenizerLike | None:
    return self.renderer.tokenizer

def get_tokenizer(self) -> TokenizerLike:
    return self.renderer.get_tokenizer()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:382` 到 `vllm/vllm/v1/engine/llm_engine.py:387`

也就是说，同步 Engine 对 tokenizer 的访问来自 renderer，而不是 EngineCore。

这和输入 / 输出处理的分层一致：

```text
tokenization / detokenization 主要在外层 processor / renderer；
模型调度执行主要在 EngineCore / Scheduler / Worker。
```

---

## 26. LLMEngine 和 EngineCore 的职责边界

从同步路径看，可以把职责边界总结为：

| 组件 | 主要职责 |
|---|---|
| `LLMEngine` | 同步 API 门面，请求入口、输出入口、管理方法转发 |
| `InputProcessor` | 校验参数，预处理 prompt，构造 `EngineCoreRequest` |
| `OutputProcessor` | 保存外层请求状态，detokenize，构造 `RequestOutput` |
| `EngineCoreClient` | 屏蔽同进程 / 多进程差异，把请求送入 EngineCore，取回输出 |
| `EngineCore` | 调用 Scheduler / Worker，形成 schedule → execute → update 闭环 |
| `Scheduler` | 管理 waiting / running / block / stop / request 状态 |
| `Worker / ModelRunner` | 真正执行模型 forward / sampling / pooling |

一句话边界：

```text
LLMEngine 管外层同步 API；
EngineCore 管内部执行闭环；
Scheduler 管请求状态和资源；
Worker 管模型执行。
```

---

## 27. LLMEngine 不负责什么

`LLMEngine` 不直接负责以下事情。

### 27.1 不直接调度请求

它不会直接维护：

```text
self.waiting
self.running
self.skipped_waiting
```

这些是 Scheduler 的职责。

### 27.2 不直接分配 KV blocks

它不会调用：

```text
kv_cache_manager.allocate_slots()
kv_cache_manager.free_blocks()
```

KV block 分配、抢占、释放都在 Scheduler / KV cache manager 里。

### 27.3 不直接执行模型 forward

它不会直接调用模型 forward。

同步同进程下，`LLMEngine.step()` 最多通过：

```text
EngineCoreClient.get_output()
  → EngineCore.step_fn()
  → model_executor.execute_model()
```

间接驱动执行。

### 27.4 不直接生成 EngineCoreOutput

`EngineCoreOutput` 是 Scheduler 在 `update_from_output()` 中根据 Worker 输出构造的。

`LLMEngine` 只消费它：

```text
EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput
```

### 27.5 不直接处理 prefix cache / external KV cache 命中

这些逻辑发生在 Scheduler waiting 调度阶段：

```text
prefix cache hit
external KV cache hit
async remote KV load
KV connector metadata
```

`LLMEngine` 只在输出里透传可能的 `kv_transfer_params`。

---

## 28. 同步普通请求的完整流程图

```text
LLMEngine.add_request(request_id, prompt, params)
  │
  ├─ 校验 request_id 是 str
  │
  ├─ 如果 prompt 不是 EngineCoreRequest
  │    ├─ InputProcessor.process_inputs()
  │    │    ├─ validate params / LoRA
  │    │    ├─ preprocess raw prompt
  │    │    ├─ validate platform request
  │    │    ├─ split encoder / decoder inputs
  │    │    ├─ clone / update SamplingParams 或 PoolingParams
  │    │    ├─ collect mm_features
  │    │    └─ return EngineCoreRequest
  │    └─ extract_prompt_components()
  │
  ├─ InputProcessor.assign_request_id()
  │    ├─ external_req_id = 用户 request_id
  │    └─ 默认 request_id = external_req_id + random suffix
  │
  ├─ n == 1 ?
  │    ├─ OutputProcessor.add_request()
  │    │    └─ 创建 RequestState
  │    └─ EngineCoreClient.add_request()
  │         ├─ InprocClient: preprocess_add_request + EngineCore.add_request
  │         └─ SyncMPClient: 发送 ADD 到 EngineCoreProc
  │
  └─ return internal req_id
```

---

## 29. 同步 parallel sampling 的完整流程图

```text
LLMEngine.add_request(..., SamplingParams.n > 1)
  │
  ├─ InputProcessor.process_inputs()
  ├─ InputProcessor.assign_request_id()
  ├─ parent_req = ParentRequest(request)
  │
  ├─ for idx in range(n):
  │    ├─ parent_req.get_child_info(idx)
  │    ├─ child_request.request_id = child id
  │    ├─ child_request.sampling_params = child params
  │    ├─ OutputProcessor.add_request(child_request, parent_req, idx)
  │    └─ EngineCoreClient.add_request(child_request)
  │
  └─ return parent internal req_id
```

执行和输出上：

```text
EngineCore / Scheduler：
  看到的是多个 child requests。

OutputProcessor / ParentRequest：
  负责把多个 child 的 CompletionOutput 合并为一个外部 RequestOutput。
```

---

## 30. 同步 step 的完整流程图

```text
LLMEngine.step()
  │
  ├─ if should_execute_dummy_batch:
  │    ├─ engine_core.execute_dummy_batch()
  │    └─ return []
  │
  ├─ outputs = EngineCoreClient.get_output()
  │    ├─ InprocClient:
  │    │    ├─ EngineCore.step_fn()
  │    │    ├─ EngineCore.post_step()
  │    │    └─ EngineCoreOutputs
  │    └─ SyncMPClient:
  │         └─ outputs_queue.get()
  │
  ├─ OutputProcessor.process_outputs(outputs.outputs)
  │    ├─ stats
  │    ├─ detokenize
  │    ├─ logprobs
  │    ├─ stop string check
  │    ├─ RequestOutput / PoolingRequestOutput
  │    └─ reqs_to_abort
  │
  ├─ engine_core.abort_requests(reqs_to_abort)
  ├─ logger_manager.record(...)
  └─ return request_outputs
```

---

## 31. add_request 和 step 的配对关系

同步 `LLMEngine` 的使用方式通常是：

```text
add_request() 只负责把请求送进去；
step() 负责不断取回输出；
has_unfinished_requests() 决定是否继续 step。
```

三者关系是：

```text
LLMEngine.add_request()
  → OutputProcessor.request_states 增加请求
  → EngineCore / Scheduler 接管执行

LLMEngine.has_unfinished_requests()
  → 检查 OutputProcessor.request_states 是否还有请求

LLMEngine.step()
  → EngineCoreClient.get_output()
  → OutputProcessor.process_outputs()
  → 请求 finished 时从 OutputProcessor.request_states 清理
```

所以从同步 API 视角：

```text
请求生命周期开始：OutputProcessor.add_request()
请求生命周期结束：OutputProcessor.process_outputs() / abort_requests() 清理 RequestState
```

而 EngineCore 内部还有自己的请求生命周期：

```text
Scheduler.add_request()
  → waiting / running
  → update_from_output()
  → finished / free blocks
```

两条生命周期相关，但不完全等价。

---

## 32. 容易疑惑的点

### 32.1 LLMEngine 是否就是 EngineCore？

不是。

```text
LLMEngine 是同步外层 Engine；
EngineCore 是内部执行核心。
```

`LLMEngine` 持有的是：

```python
self.engine_core = EngineCoreClient.make_client(...)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:105` 到 `vllm/vllm/v1/engine/llm_engine.py:111`

也就是 `EngineCoreClient`，而不是一定直接持有 `EngineCore`。

### 32.2 LLMEngine.add_request() 是否直接把请求放进 Scheduler？

不是直接放。

主线是：

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request
  → EngineCore.add_request()
  → Scheduler.add_request()
```

### 32.3 为什么 add_request 要先调用 OutputProcessor.add_request()？

因为 EngineCore 返回的是内部输出 `EngineCoreOutput`。

OutputProcessor 需要提前保存：

```text
external_req_id
prompt text
prompt token ids
parent request
request index
detokenizer
logprobs processor
stream interval
stats
```

否则后续无法构造用户可见的 `RequestOutput`。

### 32.4 为什么 request_id 会被随机化？

为了内部唯一性。

`assign_request_id()` 默认会：

```text
external_req_id = 用户传入 id
request_id = external_req_id + 8 位随机后缀
```

位置：`vllm/vllm/v1/engine/input_processor.py:232` 到 `vllm/vllm/v1/engine/input_processor.py:240`

这样即使用户重复传入同一个 id，内部请求也更不容易冲突；如果设置了 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`，则不会追加随机后缀。

### 32.5 LLMEngine.step() 是否每次都会执行模型？

不一定。

同进程模式下，它会调用 `EngineCore.step_fn()`，但 EngineCore 可能没有真正执行模型 token 计算。

多进程模式下，`LLMEngine.step()` 只是从 `outputs_queue` 取后台 EngineCoreProc 已产生的输出。

另外 DP dummy batch 路径会返回空列表：

```python
self.engine_core.execute_dummy_batch()
return []
```

位置：`vllm/vllm/v1/engine/llm_engine.py:297` 到 `vllm/vllm/v1/engine/llm_engine.py:300`

### 32.6 stop string 为什么在 OutputProcessor 里处理？

因为 stop string 需要 detokenize 后才能判断。

Scheduler 更适合处理 token 级停止条件；OutputProcessor 能看到文本，因此它检测 stop string。

如果 OutputProcessor 检出 stop string，但 EngineCore 还没 finished，会让 `LLMEngine.step()` 反向 abort：

```text
OutputProcessor.process_outputs()
  → reqs_to_abort
  → LLMEngine.step()
  → EngineCoreClient.abort_requests()
```

### 32.7 n > 1 时 EngineCore 知道 parent request 吗？

EngineCore 主要看到的是多个 child request。

parent / child 的输出归并由外层处理：

```text
ParentRequest
OutputProcessor
RequestState.make_request_output()
```

所以 parallel sampling 的外部请求语义主要在 `LLMEngine` / `OutputProcessor` 层维护。

### 32.8 has_unfinished_requests 看的是 Scheduler 吗？

同步 API 主要看 `OutputProcessor.request_states`。

```python
has_unfinished = self.output_processor.has_unfinished_requests()
```

位置：`vllm/vllm/v1/engine/llm_engine.py:191` 到 `vllm/vllm/v1/engine/llm_engine.py:192`

DP / multiprocess 场景会额外考虑 `dp_engines_running()` 或 DP group 全局状态。

---

## 33. 从“回答问题”的角度总结

如果要问：

```text
LLMEngine 负责什么？
```

可以回答：

```text
同步 LLMEngine 是 vLLM V1 的外层同步 Engine。
它负责接收用户请求，调用 InputProcessor 把 prompt / params / LoRA / 多模态输入转换为 EngineCoreRequest；
然后在 OutputProcessor 中注册外层请求状态；
再通过 EngineCoreClient 把请求送入 EngineCore。

在输出方向，LLMEngine.step() 从 EngineCoreClient 获取 EngineCoreOutputs，
交给 OutputProcessor detokenize、处理 logprobs / stop string / pooling / parallel sampling 合并，
最后返回 RequestOutput 或 PoolingRequestOutput。
```

它的核心公式是：

```text
输入方向：
LLMEngine.add_request()
  → InputProcessor
  → EngineCoreRequest
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request()

输出方向：
LLMEngine.step()
  → EngineCoreClient.get_output()
  → EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
```

它不负责：

```text
Scheduler 调度；
Worker forward；
KV block 分配 / 抢占 / 释放；
prefix cache / external KV cache 命中；
structured output grammar 推进；
spec decode 接受 / 拒绝修正。
```

这些都发生在 EngineCore / Scheduler / Worker 层。

---

## 34. 最关键流程图

```text
同步 LLMEngine 总览

构造阶段：
LLMEngine.__init__()
  ├─ renderer_from_config()
  ├─ InputProcessor(vllm_config, renderer)
  ├─ OutputProcessor(renderer.tokenizer, ...)
  └─ EngineCoreClient.make_client(..., asyncio_mode=False)

请求入口：
LLMEngine.add_request()
  ├─ InputProcessor.process_inputs()
  │    └─ EngineCoreRequest
  ├─ InputProcessor.assign_request_id()
  │    ├─ external_req_id = 用户 id
  │    └─ request_id = 内部随机 id
  ├─ OutputProcessor.add_request()
  │    └─ RequestState
  └─ EngineCoreClient.add_request()
       ├─ InprocClient: EngineCore.preprocess_add_request() → EngineCore.add_request()
       └─ SyncMPClient: ZMQ ADD → EngineCoreProc

执行和输出：
LLMEngine.step()
  ├─ EngineCoreClient.get_output()
  │    └─ EngineCoreOutputs
  ├─ OutputProcessor.process_outputs()
  │    ├─ detokenize
  │    ├─ logprobs
  │    ├─ stop string
  │    ├─ parent request merge
  │    └─ RequestOutput / PoolingRequestOutput
  ├─ EngineCoreClient.abort_requests(reqs_to_abort)
  └─ return request_outputs
```

---

## 35. 和 engine_core / scheduler 文档的关系

`engine_core/02_request_entry.md` 解释的是：

```text
请求如何从外层 Engine / AsyncLLM 进入 EngineCore，再进入 Scheduler。
```

本文中的 `LLMEngine.add_request()` 正是同步路径的外层入口。

`engine_core/03_step_loop.md` 解释的是：

```text
EngineCore.step() 如何 schedule → execute → update。
```

本文中的 `LLMEngine.step()` 在同进程模式下会通过 `InprocClient.get_output()` 间接调用 `EngineCore.step_fn()`。

`scheduler/08_update_after_worker_output.md` 解释的是：

```text
Scheduler 如何把 ModelRunnerOutput 转成 EngineCoreOutputs。
```

本文中的 `OutputProcessor.process_outputs()` 则解释：

```text
LLMEngine 如何把 EngineCoreOutputs 继续转成用户可见的 RequestOutput / PoolingRequestOutput。
```

因此三层关系是：

```text
Scheduler：
  ModelRunnerOutput → EngineCoreOutputs

EngineCore：
  schedule → execute → update_from_output

LLMEngine：
  用户输入 → EngineCoreRequest
  EngineCoreOutputs → RequestOutput
```
