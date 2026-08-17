# 07. 一个请求在外层 Engine 中如何流动？

源码位置：

- `vllm/vllm/v1/engine/llm_engine.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/input_processor.py`
- `vllm/vllm/v1/engine/output_processor.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/request.py`

本问题关注：用户请求从外层 Engine 入口到进入 EngineCore / Scheduler 的完整路径。

---

## 1. 一句话回答

一个请求不会直接从用户入口进入 `Scheduler`。

它会先经过外层 Engine：

```text
用户 / API server
  → LLMEngine / AsyncLLM
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request() / add_request_async()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
```

一句话：

```text
外层 Engine 负责把用户输入变成 EngineCoreRequest，并登记输出状态；
EngineCore 负责把 EngineCoreRequest 变成 Scheduler 内部 Request；
Scheduler 才真正接管请求调度状态。
```

这里有两个很重要的请求对象边界：

```text
EngineCoreRequest：
  外层 Engine 传给 EngineCore 的请求协议。

Request：
  EngineCore 转换后交给 Scheduler 的内部请求状态对象。
```

也有两个很重要的状态登记点：

```text
OutputProcessor.add_request()：
  在外层登记输出侧状态，后续才能把 EngineCoreOutput 转成 RequestOutput。

Scheduler.add_request()：
  在内层登记调度侧状态，后续才能参与 schedule / execute / update。
```

---

## 2. 请求生命周期总览

从用户请求到 Scheduler 的完整链路可以分成五段。

### 2.1 外层入口

同步路径：

```text
LLMEngine.add_request()
```

异步路径：

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
```

或者：

```text
AsyncLLM.encode()
  → AsyncLLM.add_request()
```

这一层接收的是用户侧输入：

```text
同步 LLMEngine.add_request()：
  request_id、prompt、params、arrival_time、lora_request、tokenization_kwargs、trace_headers、priority、prompt_text。

异步 AsyncLLM.add_request() / generate()：
  还接收 data_parallel_rank、reasoning_ended、reasoning_parser_kwargs。
```

### 2.2 输入处理

外层 Engine 调用：

```text
InputProcessor.process_inputs()
```

把用户输入转成：

```text
EngineCoreRequest
```

### 2.3 输出侧登记

在请求真正进入 EngineCore 前，外层 Engine 会先调用：

```text
OutputProcessor.add_request()
```

这一步不是生成输出，而是提前登记输出状态。

### 2.4 发送给 EngineCore

外层 Engine 不直接操作 EngineCore，而是通过：

```text
EngineCoreClient
```

同步路径：

```text
EngineCoreClient.add_request()
```

异步路径：

```text
EngineCoreClient.add_request_async()
```

### 2.5 EngineCore 内部转换并交给 Scheduler

EngineCore 内部会把 `EngineCoreRequest` 转成 `Request`：

```text
EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
```

然后交给 Scheduler：

```text
EngineCore.add_request()
  → Scheduler.add_request()
```

---

## 3. 第一阶段：InputProcessor 把用户输入转成 EngineCoreRequest

`InputProcessor` 是外层 Engine 的输入处理组件。

初始化时会保存配置，并创建 `InputPreprocessor`：

```python
self.vllm_config = vllm_config
self.model_config = model_config = vllm_config.model_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
self.scheduler_config = vllm_config.scheduler_config
...
self.input_preprocessor = InputPreprocessor(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:36` 到 `vllm/vllm/v1/engine/input_processor.py:73`

### 3.1 参数校验

`process_inputs()` 开头会校验 `SamplingParams` / `PoolingParams`：

```python
self._validate_params(params, supported_tasks)
self._validate_lora(lora_request)
```

位置：`vllm/vllm/v1/engine/input_processor.py:242` 到 `vllm/vllm/v1/engine/input_processor.py:257`

`_validate_params()` 会区分 generation 和 pooling：

```python
if isinstance(params, SamplingParams):
    ...
elif isinstance(params, PoolingParams):
    ...
else:
    raise TypeError(...)
```

位置：`vllm/vllm/v1/engine/input_processor.py:82` 到 `vllm/vllm/v1/engine/input_processor.py:144`

所以一个请求在进入 EngineCore 前，已经在外层确认：

```text
这个模型是否支持 generation；
这个模型是否支持 pooling；
sampling / pooling params 是否合法；
LoRA 是否可用；
data_parallel_rank 是否在合法范围内。
```

### 3.2 原始 prompt 预处理

如果传入的 prompt 已经是带 `type` 的 `EngineInput`，直接使用：

```python
if isinstance(prompt, dict) and "type" in prompt:
    ...
    processed_inputs: EngineInput = prompt
```

位置：`vllm/vllm/v1/engine/input_processor.py:269` 到 `vllm/vllm/v1/engine/input_processor.py:280`

否则会调用 `InputPreprocessor.preprocess()`：

```python
processed_inputs = self.input_preprocessor.preprocess(
    prompt,
    tokenization_kwargs=tokenization_kwargs,
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:281` 到 `vllm/vllm/v1/engine/input_processor.py:294`

也就是说：

```text
PromptType / raw prompt
  → InputPreprocessor.preprocess()
  → EngineInput
```

### 3.3 平台和模型输入校验

```python
current_platform.validate_request(processed_inputs, params)
```

位置：`vllm/vllm/v1/engine/input_processor.py:296`

然后拆分 encoder / decoder 输入：

```python
encoder_inputs, decoder_inputs = split_enc_dec_input(processed_inputs)
self._validate_model_inputs(encoder_inputs, decoder_inputs)
```

位置：`vllm/vllm/v1/engine/input_processor.py:298` 到 `vllm/vllm/v1/engine/input_processor.py:299`

`_validate_model_inputs()` 会校验：

```text
prompt 是否为空；
prompt 长度是否超过 max_model_len；
encoder input 是否超过 encoder cache；
token id 是否越界；
多模态 placeholder 是否合法。
```

相关位置：`vllm/vllm/v1/engine/input_processor.py:387` 到 `vllm/vllm/v1/engine/input_processor.py:494`

### 3.4 提取 prompt_token_ids / prompt_embeds

如果 decoder 输入是 embeds：

```python
if decoder_inputs["type"] == "embeds":
    prompt_embeds = decoder_inputs["prompt_embeds"]
    prompt_token_ids = decoder_inputs.get("prompt_token_ids")
    prompt_is_token_ids = decoder_inputs.get("is_token_ids")
```

位置：`vllm/vllm/v1/engine/input_processor.py:301` 到 `vllm/vllm/v1/engine/input_processor.py:305`

否则普通 token 输入：

```python
else:
    prompt_token_ids = decoder_inputs["prompt_token_ids"]
    prompt_embeds = None
    prompt_is_token_ids = None
```

位置：`vllm/vllm/v1/engine/input_processor.py:306` 到 `vllm/vllm/v1/engine/input_processor.py:309`

### 3.5 克隆并补全 SamplingParams / PoolingParams

如果是 generation 请求：

```python
sampling_params = params.clone()
if sampling_params.max_tokens is None:
    seq_len = length_from_prompt_token_ids_or_embeds(
        prompt_token_ids, prompt_embeds
    )
    sampling_params.max_tokens = self.model_config.max_model_len - seq_len

sampling_params.update_from_generation_config(...)
if self.tokenizer is not None:
    sampling_params.update_from_tokenizer(self.tokenizer)
```

位置：`vllm/vllm/v1/engine/input_processor.py:311` 到 `vllm/vllm/v1/engine/input_processor.py:328`

如果是 pooling 请求：

```python
pooling_params = params.clone()
```

位置：`vllm/vllm/v1/engine/input_processor.py:329` 到 `vllm/vllm/v1/engine/input_processor.py:330`

这说明传给 EngineCore 的 params 已经是外层处理过的克隆对象。

### 3.6 多模态输入转换成 mm_features

如果 decoder 输入是 multimodal：

```python
if decoder_inputs["type"] == "multimodal":
    decoder_mm_inputs = decoder_inputs["mm_kwargs"]
    decoder_mm_positions = decoder_inputs["mm_placeholders"]
    decoder_mm_hashes = decoder_inputs["mm_hashes"]
```

位置：`vllm/vllm/v1/engine/input_processor.py:332` 到 `vllm/vllm/v1/engine/input_processor.py:338`

随后会按多模态 placeholder 的位置排序，并构造 `MultiModalFeatureSpec`：

```python
mm_features.append(
    MultiModalFeatureSpec(
        data=decoder_mm_inputs[modality][idx],
        modality=modality,
        identifier=self._get_mm_identifier(...),
        mm_position=decoder_mm_positions[modality][idx],
        mm_hash=base_mm_hash,
    )
)
```

位置：`vllm/vllm/v1/engine/input_processor.py:349` 到 `vllm/vllm/v1/engine/input_processor.py:368`

### 3.7 构造 EngineCoreRequest

`process_inputs()` 最后返回：

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

所以 InputProcessor 输出的是：

```text
EngineCoreRequest
```

而不是 Scheduler 的 `Request`。

---

## 4. EngineCoreRequest 是什么

`EngineCoreRequest` 定义在 `vllm/vllm/v1/engine/__init__.py`。

核心字段包括：

```python
class EngineCoreRequest(...):
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

位置：`vllm/vllm/v1/engine/__init__.py:88` 到 `vllm/vllm/v1/engine/__init__.py:145`

它可以理解为：

```text
外层 Engine 和 EngineCore 之间的请求协议。
```

它已经包含：

```text
prompt token ids / prompt embeds；
多模态特征；
sampling params / pooling params；
arrival_time；
LoRA；
cache_salt；
priority；
data_parallel_rank；
trace headers；
streaming input 标记；
client_index；
current_wave；
```

但它还不是 Scheduler 内部 `Request`。

---

## 5. request_id 的外部 ID 和内部 ID

请求进入 EngineCore 前，外层 Engine 会调用：

```python
self.input_processor.assign_request_id(request)
```

同步路径位置：`vllm/vllm/v1/engine/llm_engine.py:263`

异步路径位置：`vllm/vllm/v1/engine/async_llm.py:368`

`assign_request_id()` 的源码是：

```python
@staticmethod
def assign_request_id(request: EngineCoreRequest):
    """Replace the externally supplied request ID with an internal request ID
    that adds 8 random characters in order to ensure uniqueness.
    """
    if request.external_req_id is not None:
        raise ValueError(...)
    request.external_req_id = request.request_id
    if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
        ...
    else:
        request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
```

位置：`vllm/vllm/v1/engine/input_processor.py:222` 到 `vllm/vllm/v1/engine/input_processor.py:240`

这一步非常关键：

```text
external_req_id：用户传入的 request_id；
request_id：vLLM 内部使用的唯一 request_id，默认追加 8 位随机后缀。
```

为什么要这样做？

```text
避免用户传入重复 request_id 导致内部状态冲突；
支持 parallel sampling / streaming input / abort 映射；
最终输出仍然可以还原成用户的 external request_id。
```

OutputProcessor 会维护映射：

```text
external_req_id → internal request_id 列表
```

用户最终看到的 `RequestOutput.request_id` 仍然是 external id。

---

## 6. 同步请求路径：LLMEngine.add_request()

同步路径入口是 `LLMEngine.add_request()`。

### 6.1 入口参数

```python
def add_request(
    self,
    request_id: str,
    prompt: EngineCoreRequest | PromptType | EngineInput,
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    prompt_text: str | None = None,
) -> str:
```

位置：`vllm/vllm/v1/engine/llm_engine.py:218` 到 `vllm/vllm/v1/engine/llm_engine.py:229`

它接收用户侧参数，并最终返回 request id。

### 6.2 直接传 EngineCoreRequest 的特殊路径

如果传入的 prompt 已经是 `EngineCoreRequest`，会直接使用，但这是 deprecated 路径：

```python
if isinstance(prompt, EngineCoreRequest):
    logger.warning_once(...)
    request = prompt
```

位置：`vllm/vllm/v1/engine/llm_engine.py:234` 到 `vllm/vllm/v1/engine/llm_engine.py:249`

### 6.3 普通路径：InputProcessor.process_inputs()

普通输入会走：

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

然后提取 prompt_text：

```python
prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:261`

### 6.4 分配内部 request id

```python
self.input_processor.assign_request_id(request)
req_id = request.request_id
```

位置：`vllm/vllm/v1/engine/llm_engine.py:263` 到 `vllm/vllm/v1/engine/llm_engine.py:265`

### 6.5 n == 1：普通单请求

```python
if n == 1:
    # Make a new RequestState and queue.
    self.output_processor.add_request(request, prompt_text, None, 0)
    # Add the request to EngineCore.
    self.engine_core.add_request(request)
    return req_id
```

位置：`vllm/vllm/v1/engine/llm_engine.py:270` 到 `vllm/vllm/v1/engine/llm_engine.py:277`

这里有两个动作：

```text
OutputProcessor.add_request()：
  登记外层输出状态。

EngineCoreClient.add_request()：
  把请求送入 EngineCore。
```

注意顺序：先 OutputProcessor，后 EngineCore。

### 6.6 n > 1：parallel sampling fan out

如果 `SamplingParams.n > 1`，会拆成多个 child request：

```python
parent_req = ParentRequest(request)
for idx in range(n):
    request_id, child_params = parent_req.get_child_info(idx)
    child_request = request if idx == n - 1 else copy(request)
    child_request.request_id = request_id
    child_request.sampling_params = child_params
    ...
    self.output_processor.add_request(
        child_request, prompt_text, parent_req, idx
    )
    self.engine_core.add_request(child_request)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:279` 到 `vllm/vllm/v1/engine/llm_engine.py:292`

这说明：

```text
用户一个请求，如果 n > 1，内部会变成多个 child request 进入 EngineCore；
OutputProcessor / ParentRequest 负责后续把多个 child 输出聚合回一个外部 RequestOutput。
```

### 6.7 同步请求路径总结

```text
LLMEngine.add_request()
  → 校验 request_id 类型
  → 如果 prompt 是 EngineCoreRequest：直接使用
  → 否则 InputProcessor.process_inputs()
  → EngineCoreRequest
  → extract_prompt_components()
  → InputProcessor.assign_request_id()
  → 如果 n == 1：
      → OutputProcessor.add_request(request)
      → EngineCoreClient.add_request(request)
  → 如果 n > 1：
      → ParentRequest
      → 拆 child request
      → 每个 child：
          → OutputProcessor.add_request(child, parent)
          → EngineCoreClient.add_request(child)
```

---

## 7. 异步请求路径：AsyncLLM.generate() / add_request()

异步路径通常从 `AsyncLLM.generate()` 进入。

当前实现中，`generate()` 的主线是：

```text
generate() awaits add_request()，add_request() 返回 RequestOutputCollector。
后台 output_handler 拉取 EngineCoreOutputs，调用 OutputProcessor.process_outputs()。
OutputProcessor 把 RequestOutput 放入对应 collector。
generate() 再执行 q.get_nowait() / await q.get()，持续 yield RequestOutput 直到 finished。
```

关键位置：`vllm/vllm/v1/engine/async_llm.py:557` 到 `vllm/vllm/v1/engine/async_llm.py:586`、`vllm/vllm/v1/engine/output_processor.py:45` 到 `vllm/vllm/v1/engine/output_processor.py:86`

### 7.1 generate() 调用 add_request()

```python
q = await self.add_request(
    request_id,
    prompt,
    sampling_params,
    ...
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:557` 到 `vllm/vllm/v1/engine/async_llm.py:571`

返回的 `q` 是：

```text
RequestOutputCollector
```

它保存待消费的 `RequestOutput` / `PoolingRequestOutput`，并在 `DELTA` 模式下把生产端领先消费者的 `RequestOutput` 合并；后续 `generate()` 从它里面取 `RequestOutput` 并 yield。

### 7.2 AsyncLLM.add_request() 入口

```python
async def add_request(
    self,
    request_id: str,
    prompt: EngineCoreRequest | PromptType | EngineInput | AsyncGenerator[StreamingInput, None],
    params: SamplingParams | PoolingParams,
    ...
) -> RequestOutputCollector:
```

位置：`vllm/vllm/v1/engine/async_llm.py:280` 到 `vllm/vllm/v1/engine/async_llm.py:297`

异步路径相比同步路径多了：

```text
RequestOutputCollector；
output_handler；
add_request_async；
streaming input 支持。
```

### 7.3 Engine dead 检查

```python
if self.errored:
    raise EngineDeadError()
```

位置：`vllm/vllm/v1/engine/async_llm.py:300` 到 `vllm/vllm/v1/engine/async_llm.py:301`

这会在后台 EngineCoreProc 已经死亡时阻止继续添加请求。

### 7.4 streaming input 特殊路径

如果 prompt 是 `AsyncGenerator`：

```python
if isinstance(prompt, AsyncGenerator):
    ...
    return await self._add_streaming_input_request(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:316` 到 `vllm/vllm/v1/engine/async_llm.py:331`

普通请求则继续走 InputProcessor。

### 7.5 普通路径：InputProcessor.process_inputs()

```python
request = self.input_processor.process_inputs(
    request_id,
    prompt,
    params,
    supported_tasks=await self.get_supported_tasks(),
    arrival_time=arrival_time,
    lora_request=lora_request,
    tokenization_kwargs=tokenization_kwargs,
    trace_headers=trace_headers,
    priority=priority,
    data_parallel_rank=data_parallel_rank,
)
```

位置：`vllm/vllm/v1/engine/async_llm.py:349` 到 `vllm/vllm/v1/engine/async_llm.py:360`

### 7.6 reasoning 字段补充

```python
if reasoning_ended is not None:
    request.reasoning_ended = reasoning_ended
if reasoning_parser_kwargs is not None:
    request.reasoning_parser_kwargs = reasoning_parser_kwargs
```

位置：`vllm/vllm/v1/engine/async_llm.py:363` 到 `vllm/vllm/v1/engine/async_llm.py:366`

这些字段后续会在 `Request.from_engine_core_request()` 中传入内部 Request。

### 7.7 分配内部 request id 并启动 output_handler

```python
self.input_processor.assign_request_id(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:368`

然后：

```python
self._run_output_handler()
```

位置：`vllm/vllm/v1/engine/async_llm.py:370` 到 `vllm/vllm/v1/engine/async_llm.py:373`

异步路径必须保证后台 output_handler 已经启动，否则 EngineCore 返回的输出没人消费。

### 7.8 创建 RequestOutputCollector

```python
queue = RequestOutputCollector(params.output_kind, request.request_id)
```

位置：`vllm/vllm/v1/engine/async_llm.py:375` 到 `vllm/vllm/v1/engine/async_llm.py:377`

这个 queue 会传给 OutputProcessor。

后续 OutputProcessor 收到输出时，不是直接返回 list，而是：

```text
queue.put(RequestOutput)
```

### 7.9 单请求与 n > 1

单请求：

```python
if is_pooling or params.n == 1:
    await self._add_request(request, prompt_text, None, 0, queue)
    return queue
```

位置：`vllm/vllm/v1/engine/async_llm.py:381` 到 `vllm/vllm/v1/engine/async_llm.py:383`

parallel sampling：

```python
parent_request = ParentRequest(request)
for idx in range(parent_params.n):
    request_id, child_params = parent_request.get_child_info(idx)
    child_request = request if idx == parent_params.n - 1 else copy(request)
    child_request.request_id = request_id
    child_request.sampling_params = child_params
    await self._add_request(
        child_request, prompt_text, parent_request, idx, queue
    )
```

位置：`vllm/vllm/v1/engine/async_llm.py:385` 到 `vllm/vllm/v1/engine/async_llm.py:397`

### 7.10 _add_request()

异步路径真正登记请求的地方是：

```python
async def _add_request(...):
    # Add the request to OutputProcessor (this process).
    self.output_processor.add_request(request, prompt, parent_req, index, queue)

    # Add the EngineCoreRequest to EngineCore (separate process).
    await self.engine_core.add_request_async(request)
```

位置：`vllm/vllm/v1/engine/async_llm.py:400` 到 `vllm/vllm/v1/engine/async_llm.py:412`

这说明异步路径中：

```text
OutputProcessor 在前端进程；
EngineCore 在后台进程；
_add_request() 必须同时登记两边。
```

### 7.11 异步请求路径总结

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
  → Engine dead 检查
  → streaming input 特殊路径 或普通 InputProcessor.process_inputs()
  → EngineCoreRequest
  → reasoning 字段补充
  → InputProcessor.assign_request_id()
  → _run_output_handler()
  → RequestOutputCollector
  → _add_request()
      → OutputProcessor.add_request(request, queue)
      → EngineCoreClient.add_request_async(request)
  → generate() 从 queue 取 RequestOutput 并 yield
```

---

## 8. 为什么 OutputProcessor.add_request 在输出前出现

这一点非常容易误解。

`OutputProcessor.add_request()` 不是在“处理已经生成的输出”，而是在请求进入 EngineCore 前建立输出侧状态。

### 8.1 EngineCoreOutput 只有内部增量信息

EngineCore 后续返回的是 `EngineCoreOutput`，它大致包含：

```text
request_id；
new_token_ids；
new_logprobs；
pooling_output；
finish_reason；
stop_reason；
```

但它不包含完整的外层输出上下文，例如：

```text
用户原始 external request id；
prompt 文本；
prompt token ids；
输出模式 DELTA / CUMULATIVE / FINAL_ONLY；
是否 n > 1 parent request；
异步 queue；
detokenizer；
logprobs_processor；
```

这些都需要 `OutputProcessor.add_request()` 提前登记。

### 8.2 OutputProcessor.add_request() 创建 RequestState

`OutputProcessor.add_request()` 会创建 `RequestState`：

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

位置：`vllm/vllm/v1/engine/output_processor.py:534` 到 `vllm/vllm/v1/engine/output_processor.py:544`

然后登记 external 到 internal 的映射：

```python
self.external_req_ids[req_state.external_req_id].append(request_id)
```

位置：`vllm/vllm/v1/engine/output_processor.py:548` 到 `vllm/vllm/v1/engine/output_processor.py:549`

所以它的作用是：

```text
提前准备好“将来收到这个 request_id 的 EngineCoreOutput 时，应该怎么生成用户输出”。
```

### 8.3 如果没有提前登记会怎样

如果 EngineCore 已经返回输出，但 OutputProcessor 没有对应 `RequestState`，代码会直接忽略：

```python
req_state = self.request_states.get(req_id)
if req_state is None:
    # Ignore output for already-aborted request.
    continue
```

位置：`vllm/vllm/v1/engine/output_processor.py:615` 到 `vllm/vllm/v1/engine/output_processor.py:619`

因此必须先登记输出侧状态，再把请求送给 EngineCore。

---

## 9. EngineCoreClient：把请求送到 EngineCore

外层 Engine 不直接依赖 EngineCore 的运行方式，而是通过 `EngineCoreClient`。

`EngineCoreClient` 注释说明：

```python
EngineCoreClient: subclasses handle different methods for pushing
    and pulling from the EngineCore for asyncio / multiprocessing.

Subclasses:
* InprocClient: In process EngineCore (for V0-style LLMEngine use)
* SyncMPClient: ZMQ + background proc EngineCore (for LLM)
* AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
```

位置：`vllm/vllm/v1/engine/core_client.py:71` 到 `vllm/vllm/v1/engine/core_client.py:80`

### 9.1 InprocClient：同进程路径

`InprocClient.add_request()`：

```python
req, request_wave = self.engine_core.preprocess_add_request(request)
self.engine_core.add_request(req, request_wave)
```

位置：`vllm/vllm/v1/engine/core_client.py:297` 到 `vllm/vllm/v1/engine/core_client.py:299`

也就是说同进程路径中：

```text
EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → EngineCore.add_request()
```

### 9.2 SyncMPClient：同步多进程路径

`SyncMPClient.add_request()` 会把请求通过 ZMQ 发出去：

```python
self._send_input(EngineCoreRequestType.ADD, request)
```

位置：`vllm/vllm/v1/engine/core_client.py:886` 到 `vllm/vllm/v1/engine/core_client.py:889`

多进程路径中，前端不直接调用 `EngineCore.preprocess_add_request()`；后台 `EngineCoreProc.process_input_sockets()` 会先 decode `EngineCoreRequest`，调用 `EngineCore.preprocess_add_request()`，再把 `(ADD, (Request, request_wave))` 放入 `input_queue`，最后由 `_handle_client_request()` 调用 `EngineCore.add_request(Request, request_wave)`。

### 9.3 AsyncMPClient：异步多进程路径

`AsyncMPClient.add_request_async()`：

```python
async def add_request_async(self, request: EngineCoreRequest) -> None:
    request.client_index = self.client_index
    await self._send_input(EngineCoreRequestType.ADD, request)
    self._ensure_output_queue_task()
```

位置：`vllm/vllm/v1/engine/core_client.py:1121` 到 `vllm/vllm/v1/engine/core_client.py:1124`

这里会设置：

```text
request.client_index
```

用于多 frontend client 场景下，后台 EngineCoreProc 把输出发回正确 client。

### 9.4 DPAsyncMPClient：data parallel 请求路由

如果 data parallel size > 1，`make_async_mp_client()` 可能返回 `DPAsyncMPClient` 或 `DPLBAsyncMPClient`。

`DPAsyncMPClient.add_request_async()` 会设置：

```python
request.current_wave = self.current_wave
request.client_index = self.client_index
chosen_engine = self.get_core_engine_for_request(request)
to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
```

位置：`vllm/vllm/v1/engine/core_client.py:1359` 到 `vllm/vllm/v1/engine/core_client.py:1367`

这说明 DP 场景下，一个请求还会被路由到某个具体 EngineCore rank。

---

## 10. 多进程后台路径：EngineCoreProc input_queue

多进程模式下，后台运行的是：

```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
```

位置：`vllm/vllm/v1/engine/core.py:905` 到 `vllm/vllm/v1/engine/core.py:906`

它有 input queue：

```python
self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
```

位置：`vllm/vllm/v1/engine/core.py:924`

### 10.1 busy loop

后台主循环是：

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()
```

位置：`vllm/vllm/v1/engine/core.py:1268` 到 `vllm/vllm/v1/engine/core.py:1274`

也就是说后台 EngineCoreProc 一直循环做两件事：

```text
处理输入请求；
执行 EngineCore.step() 并返回输出。
```

### 10.2 处理 input_queue

```python
req = self.input_queue.get(block=block)
self._handle_client_request(*req)
```

位置：`vllm/vllm/v1/engine/core.py:1293` 到 `vllm/vllm/v1/engine/core.py:1295`

如果队列里还有更多请求，会继续处理：

```python
while not self.input_queue.empty():
    req = self.input_queue.get_nowait()
    self._handle_client_request(*req)
```

位置：`vllm/vllm/v1/engine/core.py:1304` 到 `vllm/vllm/v1/engine/core.py:1307`

### 10.3 分发 ADD 请求

```python
if request_type == EngineCoreRequestType.ADD:
    req, request_wave = request
    if self._reject_add_in_shutdown(req):
        return
    self.add_request(req, request_wave)
```

位置：`vllm/vllm/v1/engine/core.py:1388` 到 `vllm/vllm/v1/engine/core.py:1392`

这里进入的是 `EngineCore.add_request()`。

注意：在多进程路径中，input thread 通常会提前把 `EngineCoreRequest` 预处理成 `(Request, request_wave)` 再放入 input_queue，因此这里变量名是 `req`。

---

## 11. EngineCore.preprocess_add_request()：EngineCoreRequest → Request

EngineCore 内部请求转换入口是：

```python
def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
    """Preprocess the request.

    This function could be directly used in input processing thread to allow
    request initialization running in parallel with Model forward
    """
```

位置：`vllm/vllm/v1/engine/core.py:864` 到 `vllm/vllm/v1/engine/core.py:869`

### 11.1 多模态 receiver cache

如果有多模态特征，会先过 EngineCore 侧 receiver cache：

```python
if self.mm_receiver_cache is not None and request.mm_features:
    request.mm_features = self.mm_receiver_cache.get_and_update_features(
        request.mm_features
    )
```

位置：`vllm/vllm/v1/engine/core.py:873` 到 `vllm/vllm/v1/engine/core.py:876`

### 11.2 转成 Request

```python
req = Request.from_engine_core_request(request, self.request_block_hasher)
```

位置：`vllm/vllm/v1/engine/core.py:878`

这一步完成：

```text
EngineCoreRequest → Request
```

### 11.3 结构化输出 grammar 初始化

如果请求使用结构化输出：

```python
if req.use_structured_output:
    self.structured_output_manager.grammar_init(req)
```

位置：`vllm/vllm/v1/engine/core.py:879` 到 `vllm/vllm/v1/engine/core.py:885`

这说明结构化输出 grammar 初始化发生在 EngineCore 侧，而不是 InputProcessor 侧。

### 11.4 返回 request_wave

```python
return req, request.current_wave
```

位置：`vllm/vllm/v1/engine/core.py:886`

`current_wave` 用于 DP 场景。

---

## 12. Request.from_engine_core_request()

`Request` 是 Scheduler 内部使用的请求对象。

转换方法是：

```python
@classmethod
def from_engine_core_request(
    cls,
    request: EngineCoreRequest,
    block_hasher: Callable[["Request"], list["BlockHash"]] | None,
) -> "Request":
    return cls(
        request_id=request.request_id,
        client_index=request.client_index,
        prompt_token_ids=request.prompt_token_ids,
        prompt_embeds=request.prompt_embeds,
        prompt_is_token_ids=request.prompt_is_token_ids,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        priority=request.priority,
        trace_headers=request.trace_headers,
        block_hasher=block_hasher,
        resumable=request.resumable,
        reasoning_ended=request.reasoning_ended,
        reasoning_parser_kwargs=request.reasoning_parser_kwargs,
        abort_immediately=request.abort_immediately,
    )
```

位置：`vllm/vllm/v1/request.py:212` 到 `vllm/vllm/v1/request.py:237`

也就是说，`EngineCoreRequest` 的字段会被搬进内部 `Request`，同时增加更多内部状态。

### 12.1 Request 初始化内部调度状态

`Request.__init__()` 会设置：

```python
self.status = RequestStatus.WAITING
self.events: list[EngineCoreEvent] = []
self.stop_reason: int | str | None = None
```

位置：`vllm/vllm/v1/request.py:97` 到 `vllm/vllm/v1/request.py:99`

它还初始化 KV / EC transfer 参数字段：

```python
self.kv_transfer_params: dict[str, Any] | None = None
self.ec_transfer_params: dict[str, Any] | None = None
```

位置：`vllm/vllm/v1/request.py:101` 到 `vllm/vllm/v1/request.py:104`

如果是 pooling：

```python
if pooling_params is not None:
    self.max_tokens = 1
```

位置：`vllm/vllm/v1/request.py:106` 到 `vllm/vllm/v1/request.py:108`

如果是 generation：

```python
assert sampling_params.max_tokens is not None
self.max_tokens = sampling_params.max_tokens
if self.structured_output_request is not None:
    self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

位置：`vllm/vllm/v1/request.py:109` 到 `vllm/vllm/v1/request.py:114`

如果 sampling params 带 `extra_args`，还会从中取出 KV / EC transfer 参数和 KV cache 上报模式：

```python
self.kv_transfer_params = sampling_params.extra_args.get(
    "kv_transfer_params"
)
self.ec_transfer_params = sampling_params.extra_args.get(
    "ec_transfer_params"
)
self.kv_cache_report_mode = sampling_params.extra_args.get(
    "kv_cache_report_mode", "incremental"
)
```

位置：`vllm/vllm/v1/request.py:116` 到 `vllm/vllm/v1/request.py:127`

也就是说，如果请求有结构化输出，它不是普通 `WAITING`，而是先进入：

```text
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
```

### 12.2 Request 增加 token / cache / scheduler 状态

`Request` 内部还会初始化：

```python
self._output_token_ids: list[int] = []
self._all_token_ids: list[int] = (...)
self.num_output_placeholders = 0
self.async_tokens_to_discard = 0
self.num_in_flight_tokens = 0
self.next_decode_eligible_step = 0
self.last_sched_seq = 0
self.spec_token_ids: list[int] = []
self.num_computed_tokens = 0
self.mm_features = mm_features or []
self.is_prefill_chunk = False
self.num_preemptions = 0
self.prefill_stats: PrefillStats | None = PrefillStats()
self.block_hashes: list[BlockHash] = []
self.update_block_hashes()
```

位置：`vllm/vllm/v1/request.py:143` 到 `vllm/vllm/v1/request.py:199`

这些都是 Scheduler 后续调度、prefix cache、spec decode、prefill/decode 状态更新要用的内部字段。

所以：

```text
EngineCoreRequest 是外层请求协议；
Request 是带调度状态、输出 token 状态、cache 状态的内部请求对象。
```

---

## 13. EngineCore.add_request()：交给 Scheduler

EngineCore 收到内部 `Request` 后，会调用：

```python
def add_request(self, request: Request, request_wave: int = 0):
    """Add request to the scheduler.

    `request_wave`: indicate which wave of requests this is expected to
    belong to in DP case
    """
```

位置：`vllm/vllm/v1/engine/core.py:372` 到 `vllm/vllm/v1/engine/core.py:377`

### 13.1 request_id 类型校验

```python
if not isinstance(request.request_id, str):
    raise TypeError(...)
```

位置：`vllm/vllm/v1/engine/core.py:378` 到 `vllm/vllm/v1/engine/core.py:382`

### 13.2 pooling task 支持校验

如果请求是 pooling，会检查任务是否支持：

```python
if pooling_params := request.pooling_params:
    supported_pooling_tasks = [
        task for task in self.get_supported_tasks() if task in POOLING_TASKS
    ]

    if pooling_params.task not in supported_pooling_tasks:
        raise ValueError(...)
```

位置：`vllm/vllm/v1/engine/core.py:384` 到 `vllm/vllm/v1/engine/core.py:393`

### 13.3 KV / EC transfer 参数检查

如果请求带了 `kv_transfer_params`，但 Scheduler 没有 KVConnector，会警告并禁用：

```python
if request.kv_transfer_params is not None and (
    not self.scheduler.get_kv_connector()
):
    logger.warning(...)
```

位置：`vllm/vllm/v1/engine/core.py:395` 到 `vllm/vllm/v1/engine/core.py:401`

如果请求带了 `ec_transfer_params`，但 Scheduler 没有 ECConnector，也会警告并禁用：

```python
if (
    request.ec_transfer_params is not None
    and self.scheduler.get_ec_connector() is None
):
    logger.warning(...)
```

位置：`vllm/vllm/v1/engine/core.py:403` 到 `vllm/vllm/v1/engine/core.py:410`

### 13.4 进入 Scheduler

最终调用：

```python
self.scheduler.add_request(request)
```

位置：`vllm/vllm/v1/engine/core.py:412`

如果请求设置了 `abort_immediately`，立刻 abort：

```python
if request.abort_immediately:
    self.abort_requests([request.request_id])
```

位置：`vllm/vllm/v1/engine/core.py:413` 到 `vllm/vllm/v1/engine/core.py:416`

### 13.5 Scheduler 接管后的状态

进入 Scheduler 后，新请求会先进入 Scheduler 的等待侧状态：

```text
Scheduler.add_request()：
  对新请求设置 streaming_queue（仅 resumable）；
  调用 _enqueue_waiting_request() 将其放入 waiting；
  如果状态是 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR / WAITING_FOR_REMOTE_KVS / WAITING_FOR_STREAMING_REQ，则放入 skipped_waiting；
  随后记录到 requests。

Scheduler.schedule()：
  后续被选中执行时，才追加到 running 并把 status 设为 RUNNING。
```

这些由 Scheduler 管理，不再属于外层 Engine 的职责。

外层 Engine 只会继续：

```text
等待 EngineCoreOutputs；
调用 OutputProcessor 转用户输出；
必要时 abort。
```

---

## 14. streaming input 请求路径

异步路径支持 streaming input。

### 14.1 入口判断

在 `AsyncLLM.add_request()` 中，如果 prompt 是 async generator：

```python
if isinstance(prompt, AsyncGenerator):
    ...
    return await self._add_streaming_input_request(...)
```

位置：`vllm/vllm/v1/engine/async_llm.py:316` 到 `vllm/vllm/v1/engine/async_llm.py:331`

### 14.2 创建 final_req

`_add_streaming_input_request()` 会先创建一个 final request：

```python
final_req = self.input_processor.process_inputs(
    request_id=request_id,
    prompt=TokensPrompt(prompt_token_ids=[0]),
    params=sampling_params,
    **inputs,
)
self.input_processor.assign_request_id(final_req)
internal_req_id = final_req.request_id
```

位置：`vllm/vllm/v1/engine/async_llm.py:445` 到 `vllm/vllm/v1/engine/async_llm.py:454`

### 14.3 每个输入 chunk 变成 resumable EngineCoreRequest

```python
async for input_chunk in input_stream:
    ...
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

位置：`vllm/vllm/v1/engine/async_llm.py:461` 到 `vllm/vllm/v1/engine/async_llm.py:483`

这里的关键点：

```text
request_id 使用同一个 internal_req_id；
resumable=True；
每个 chunk 都作为同一个请求的延续进入 EngineCore。
```

### 14.4 输入流结束后发送 final_req

```python
if not cancelled:
    await self._add_request(final_req, None, None, 0, queue)
```

位置：`vllm/vllm/v1/engine/async_llm.py:490` 到 `vllm/vllm/v1/engine/async_llm.py:495`

final_req 用来告诉输出侧和内层：输入已经结束。这里的 final_req 是带 dummy token 的结束信号请求，OutputProcessor 侧不会把这个 dummy token 当作真实 streaming chunk 追加。

### 14.5 Scheduler 内部状态

`Request` 中有 streaming 相关字段：

```python
self.resumable = resumable
self.streaming_queue: deque[StreamingUpdate | None] | None = None
```

位置：`vllm/vllm/v1/request.py:203` 到 `vllm/vllm/v1/request.py:206`

Scheduler 会根据这些状态处理 `WAITING_FOR_STREAMING_REQ` 等状态。

---

## 15. 请求输出和请求输入为什么交错登记

请求进入 EngineCore 前，外层会先登记 OutputProcessor。

同步：

```text
OutputProcessor.add_request()
EngineCoreClient.add_request()
```

异步：

```text
OutputProcessor.add_request(..., queue)
EngineCoreClient.add_request_async()
```

这看起来像“输出处理器在还没有输出时被调用”，但其实它是在建立输出状态。

核心原因：

```text
EngineCore 后续只会返回 request_id + 增量输出；
OutputProcessor 必须提前知道这个 request_id 对应的外部请求信息和输出模式。
```

它提前登记：

```text
external request id；
internal request id；
prompt；
prompt token ids / embeds；
parent request；
request index；
异步 queue；
detokenizer；
logprobs processor；
stream interval；
```

所以请求生命周期实际上有两条并行状态线：

```text
输出侧状态线：
  OutputProcessor.add_request()
  → RequestState
  → process_outputs()
  → RequestOutput
  → _finish_request()

调度侧状态线：
  EngineCoreClient.add_request()
  → EngineCore.preprocess_add_request()
  → Request
  → Scheduler.add_request()
  → Scheduler.schedule()
  → Scheduler.update_from_output()
  → free request
```

---

## 16. abort 请求路径

abort 也需要同时处理输出侧和调度侧状态。

### 16.1 同步 LLMEngine.abort_request()

```python
def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
    """Remove request_ids from EngineCore and Detokenizer."""

    request_ids = self.output_processor.abort_requests(request_ids, internal)
    self.engine_core.abort_requests(request_ids)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:212` 到 `vllm/vllm/v1/engine/llm_engine.py:216`

链路是：

```text
LLMEngine.abort_request()
  → OutputProcessor.abort_requests()
      → 清理输出侧 RequestState
      → external id 转 internal id
      → 处理 parent/child request
  → EngineCoreClient.abort_requests(internal_ids)
      → EngineCore.abort_requests()
      → Scheduler.finish_requests(... FINISHED_ABORTED)
```

### 16.2 异步 AsyncLLM.abort()

```python
async def abort(
    self, request_id: str | Iterable[str], internal: bool = False
) -> None:
    request_ids = (...)
    all_request_ids = self.output_processor.abort_requests(request_ids, internal)
    await self.engine_core.abort_requests_async(all_request_ids)
```

位置：`vllm/vllm/v1/engine/async_llm.py:709` 到 `vllm/vllm/v1/engine/async_llm.py:718`

链路是：

```text
AsyncLLM.abort()
  → OutputProcessor.abort_requests()
  → AsyncMPClient.abort_requests_async()
  → ZMQ ABORT
  → EngineCoreProc._handle_client_request(ABORT)
  → EngineCore.abort_requests()
  → Scheduler.finish_requests(... FINISHED_ABORTED)
```

### 16.3 EngineCore.abort_requests()

EngineCore 侧 abort 是：

```python
def abort_requests(self, request_ids: list[str]):
    """Abort requests from the scheduler."""
    self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
```

位置：`vllm/vllm/v1/engine/core.py:409` 到 `vllm/vllm/v1/engine/core.py:415`

所以真正调度侧取消由 Scheduler 完成。

### 16.4 stop string 导致的自动 abort

还有一种特殊 abort：OutputProcessor detokenize 后发现 stop string，但 EngineCore 还没认为请求完成。

此时 `process_outputs()` 会返回 `reqs_to_abort`。

同步路径：

```python
self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
```

位置：`vllm/vllm/v1/engine/llm_engine.py:316` 到 `vllm/vllm/v1/engine/llm_engine.py:318`

异步路径：

```python
if processed_outputs.reqs_to_abort:
    await engine_core.abort_requests_async(processed_outputs.reqs_to_abort)
```

位置：`vllm/vllm/v1/engine/async_llm.py:685` 到 `vllm/vllm/v1/engine/async_llm.py:689`

这说明：

```text
文本级 stop string 属于 OutputProcessor；
内部请求取消仍要通知 EngineCore / Scheduler。
```

---

## 17. 请求进入 Scheduler 后发生什么

请求进入 Scheduler 后，外层请求生命周期的“进入阶段”就结束了。

后续由 EngineCore 主循环驱动：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model()
  → Worker / ModelRunner
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
```

外层 Engine 不直接参与：

```text
waiting / running 队列管理；
token budget 分配；
KV block 分配和释放；
prefix cache 命中；
preemption；
structured output grammar 调度；
spec decode 接受/拒绝；
```

这些属于 Scheduler / EngineCore / Worker。

外层 Engine 等待的是：

```text
EngineCoreOutputs
```

然后再交给 OutputProcessor 转成用户输出。

---

## 18. 容易混淆的点

### 18.1 用户请求是否直接进入 Scheduler？

不是。

用户请求先进入外层 Engine，再经过：

```text
InputProcessor → EngineCoreRequest → EngineCoreClient → EngineCore → Request → Scheduler
```

### 18.2 EngineCoreRequest 和 Request 是同一个对象吗？

不是。

```text
EngineCoreRequest：
  外层 Engine 和 EngineCore 之间的请求协议。

Request：
  EngineCore 转换后交给 Scheduler 的内部调度对象。
```

### 18.3 OutputProcessor.add_request 是不是处理输出？

不是处理已有输出，而是提前登记输出状态。

真正处理输出的是：

```text
OutputProcessor.process_outputs()
```

### 18.4 为什么 request_id 会变化？

`InputProcessor.assign_request_id()` 会把用户传入的 request_id 存为：

```text
external_req_id
```

再生成内部唯一：

```text
request_id = external_req_id + 随机后缀
```

这样避免内部 request id 冲突。

### 18.5 n > 1 时是一个请求还是多个请求？

用户视角是一个请求。

内部 EngineCore 视角是多个 child request。

```text
ParentRequest：
  负责把多个 child 输出聚合回一个外部 RequestOutput。
```

### 18.6 AsyncLLM.generate() 是否直接把请求送到 Scheduler？

不是。

`generate()` 调用 `add_request()`，然后：

```text
InputProcessor → OutputProcessor → AsyncMPClient → EngineCoreProc → EngineCore → Scheduler
```

### 18.7 abort 是否只需要通知 EngineCore？

不是。

abort 要先清理 OutputProcessor 输出侧状态，再通知 EngineCore / Scheduler 取消内部请求。

---

## 19. 最关键的关系图

### 19.1 总体请求进入链路

```text
用户 / API server
  → LLMEngine.add_request() / AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → InputProcessor.assign_request_id()
  → OutputProcessor.add_request()
  → EngineCoreClient.add_request() / add_request_async()
  → EngineCore.preprocess_add_request()
  → Request.from_engine_core_request()
  → EngineCore.add_request()
  → Scheduler.add_request()
```

### 19.2 同步路径

```text
LLMEngine.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → assign_request_id()
  → OutputProcessor.add_request(request, queue=None)
  → EngineCoreClient.add_request()
      → InprocClient:
          → EngineCore.preprocess_add_request()
          → EngineCore.add_request()
      → SyncMPClient:
          → ZMQ ADD
          → EngineCoreProc.process_input_sockets()
          → decode EngineCoreRequest
          → EngineCore.preprocess_add_request()
          → input_queue.put((ADD, (Request, request_wave)))
          → EngineCoreProc._handle_client_request()
          → EngineCore.add_request(Request, request_wave)
  → Scheduler.add_request()
```

### 19.3 异步路径

```text
AsyncLLM.generate()
  → AsyncLLM.add_request()
  → InputProcessor.process_inputs()
  → EngineCoreRequest
  → assign_request_id()
  → _run_output_handler()
  → RequestOutputCollector
  → OutputProcessor.add_request(request, queue)
  → AsyncMPClient.add_request_async()
      → request.client_index = client_index
      → ZMQ ADD
      → EngineCoreProc.process_input_sockets()
      → decode EngineCoreRequest
      → EngineCore.preprocess_add_request()
      → input_queue.put((ADD, (Request, request_wave)))
      → EngineCoreProc._handle_client_request()
      → EngineCore.add_request(Request, request_wave)
  → Scheduler.add_request()
```

### 19.4 EngineCore 内部转换

```text
EngineCore.preprocess_add_request(EngineCoreRequest)
  → mm_receiver_cache.get_and_update_features()
  → Request.from_engine_core_request()
  → structured_output_manager.grammar_init(req)  # 如果需要
  → (Request, current_wave)

EngineCore.add_request(Request)
  → pooling task 校验
  → KV / EC transfer 检查
  → Scheduler.add_request(Request)
```

### 19.5 abort 链路

```text
LLMEngine.abort_request() / AsyncLLM.abort()
  → OutputProcessor.abort_requests()
      → external id 转 internal id
      → 清理 RequestState / parent request
  → EngineCoreClient.abort_requests()
  → EngineCore.abort_requests()
  → Scheduler.finish_requests(... FINISHED_ABORTED)
```

---

## 20. 从“回答问题”的角度总结

如果问：

```text
一个请求在外层 Engine 中如何流动？
```

可以回答：

```text
请求首先进入外层 Engine，也就是同步路径里的 LLMEngine.add_request()，
或者异步路径里的 AsyncLLM.generate() / add_request()。

外层 Engine 会调用 InputProcessor.process_inputs()，把用户输入、prompt、
sampling params / pooling params、多模态输入、LoRA、priority 等信息处理成 EngineCoreRequest。
然后 InputProcessor.assign_request_id() 会把用户提供的 request_id 保存为 external_req_id，
并生成内部唯一 request_id。

在请求进入 EngineCore 前，外层 Engine 会先调用 OutputProcessor.add_request()，
登记输出侧 RequestState。这样后续 EngineCore 返回 EngineCoreOutput 时，
OutputProcessor 才知道如何 detokenize、处理 logprobs、聚合 n>1 输出，
以及同步返回还是异步推入 RequestOutputCollector。

随后请求通过 EngineCoreClient 发送到 EngineCore。
同进程模式下 InprocClient 直接调用 EngineCore.preprocess_add_request() 和 EngineCore.add_request()；
多进程模式下 SyncMPClient / AsyncMPClient 通过 ZMQ 把请求发给后台 EngineCoreProc，
由后台 input queue 和 busy loop 分发处理。

EngineCore.preprocess_add_request() 会把 EngineCoreRequest 转成 Scheduler 使用的内部 Request，
并为结构化输出请求初始化 grammar。最后 EngineCore.add_request() 调用 Scheduler.add_request()，
请求正式进入 Scheduler 的 waiting / running / requests 状态管理体系。
```

最小心智模型：

```text
用户请求
  → InputProcessor：变成 EngineCoreRequest
  → OutputProcessor：登记输出状态
  → EngineCoreClient：送入 EngineCore
  → EngineCore：变成 Request
  → Scheduler：进入调度队列
```
