# 02. LoRARequest 如何从用户请求进入执行链路？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\request.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\llm_engine.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\input_processor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\__init__.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\core.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\request.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\output.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_input_batch.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\lora_model_runner_mixin.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\worker_manager.py`

本问题关注：用户请求携带的 `LoRARequest` 如何进入 `EngineCoreRequest`、Scheduler 侧 `Request`、`SchedulerOutput.NewRequestData`，最终如何被 Worker / ModelRunner 转成当前 batch 的 LoRA mapping，并驱动 LoRA manager 加载和激活 adapter。

---

## 1. 一句话回答

`LoRARequest` 是请求级的 adapter 选择信息。

它不会在 Engine / Scheduler 里执行 LoRA，也不等于 adapter 已经加载到 GPU；它只是一路跟随请求状态流转，直到 Worker 构造本轮 batch 时，被转换成：

```text
req_index -> lora_int_id
lora_int_id -> LoRARequest
per-token LoRA mapping
```

然后 ModelRunner 在 forward 前调用 LoRA manager 激活本轮需要的 adapter。

主链路是：

```text
用户请求 / LLMEngine.add_request(lora_request=...)
  → InputProcessor._validate_lora()
  → EngineCoreRequest.lora_request
  → EngineCore.preprocess_add_request()
  → Request.lora_request
  → Scheduler.schedule()
  → NewRequestData.lora_request
  → GPUModelRunner._update_states()
  → CachedRequestState.lora_request
  → InputBatch.add_request()
  → request_lora_mapping / lora_id_to_lora_request
  → InputBatch.make_lora_inputs()
  → LoRAModelRunnerMixin.set_active_loras()
  → WorkerLoRAManager.set_active_adapters()
  → LoRA layer 按 mapping 执行
```

所以：

```text
Engine / Scheduler 负责携带 LoRARequest；
Worker / ModelRunner 负责把 LoRARequest 变成可执行的 adapter mapping；
LoRA manager 负责 adapter 的加载、缓存、激活和映射下发。
```

---

## 2. LoRARequest 是什么

`LoRARequest` 定义在：`request.py:8`

核心字段：

```python
class LoRARequest(msgspec.Struct, omit_defaults=True, array_like=True):
    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None
    load_inplace: bool = False
    is_3d_lora_weight: bool = False
```

位置：`lora/request.py:8` 到 `lora/request.py:37`

字段含义：

| 字段 | 含义 |
|---|---|
| `lora_name` | adapter 的逻辑名称，`__eq__` / `__hash__` 按它比较 |
| `lora_int_id` | adapter 的整数 ID，执行时写入 LoRA mapping，必须大于 0 |
| `lora_path` | adapter checkpoint 路径，加载 adapter 时使用，不能为空 |
| `base_model_name` | 可选 base model 标识，主要是元数据 |
| `tensorizer_config_dict` | tensorizer 加载相关配置 |
| `load_inplace` | True 时即使相同 ID 已存在，也强制重新加载并原地替换 |
| `is_3d_lora_weight` | MoE LoRA 权重格式标记，混合 MoE LoRA 格式时用于加载路由 |

初始化校验：

```python
def __post_init__(self):
    if self.lora_int_id < 1:
        raise ValueError(f"id must be > 0, got {self.lora_int_id}")

    assert self.lora_path, "lora_path cannot be empty"
```

位置：`lora/request.py:39` 到 `lora/request.py:44`

还有三个兼容属性：

```python
adapter_id -> lora_int_id
name       -> lora_name
path       -> lora_path
```

位置：`lora/request.py:46` 到 `lora/request.py:56`

容易忽略的一点：

```text
LoRARequest 的相等性和 hash 按 lora_name 判断，不按 lora_int_id / lora_path 判断。
```

对应代码：

```python
def __eq__(self, value: object) -> bool:
    return isinstance(value, self.__class__) and self.lora_name == value.lora_name

def __hash__(self) -> int:
    return hash(self.lora_name)
```

位置：`lora/request.py:58` 到 `lora/request.py:73`

这会影响 `set[LoRARequest]` 去重：同名 LoRA 会被认为是同一个请求对象。

---

## 3. 用户请求入口

V1 入口在 `LLMEngine.add_request()`：`llm_engine.py:218`

函数签名包含：

```python
def add_request(
    self,
    request_id: str,
    prompt: EngineCoreRequest | PromptType | EngineInput,
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    ...
) -> str:
```

位置：`llm_engine.py:218` 到 `llm_engine.py:229`

如果传进来的不是已经构造好的 `EngineCoreRequest`，会调用 `InputProcessor.process_inputs()`：

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

位置：`llm_engine.py:249` 到 `llm_engine.py:260`

然后：

```python
self.input_processor.assign_request_id(request)
...
self.engine_core.add_request(request)
```

位置：`llm_engine.py:263` 到 `llm_engine.py:276`

这里有一个重要细节：

```text
用户传入的 request_id 会被 assign_request_id() 改成内部 request_id；
LoRARequest 不会被改写，仍挂在 EngineCoreRequest 上。
```

---

## 4. InputProcessor 如何处理 LoRARequest

`InputProcessor.process_inputs()` 定义在：`input_processor.py:242`

开始时会校验 params 和 LoRA：

```python
self._validate_params(params, supported_tasks)
self._validate_lora(lora_request)
```

位置：`input_processor.py:256` 到 `input_processor.py:257`

### 4.1 _validate_lora()

定义位置：`input_processor.py:146`

```python
def _validate_lora(self, lora_request: LoRARequest | None) -> None:
    if lora_request is None:
        return

    # LoRA request passed in while LoRA is not enabled
    if not self.lora_config:
        raise ValueError(
            f"Got lora_request {lora_request} but LoRA is not enabled!"
        )

    if self.tokenizer is not None:
        logger.warning_once(...)
```

位置：`input_processor.py:146` 到 `input_processor.py:163`

它只做两件事：

```text
1. 如果没传 lora_request，直接返回；
2. 如果传了 lora_request 但 engine 没启用 LoRA，报错。
```

它不检查 adapter 文件是否存在，也不加载 adapter。adapter 的实际加载在 Worker 侧发生。

### 4.2 LoRA 对多模态 cache key 的影响

`InputProcessor` 还有一个和多模态相关的 LoRA 逻辑：

```python
def _get_mm_identifier(
    self,
    mm_hash: str,
    lora_request: LoRARequest | None,
) -> str:
    if (
        lora_request is None
        or self.lora_config is None
        or not self.lora_config.enable_tower_connector_lora
    ):
        return mm_hash
    return f"{lora_request.lora_name}:{mm_hash}"
```

位置：`input_processor.py:165` 到 `input_processor.py:181`

原因是：

```text
enable_tower_connector_lora=True 时，多模态 encoder / connector 输出会受 LoRA 影响。
同一张图片在不同 LoRA 下可能产生不同 embedding，不能共享同一个 mm cache key。
```

构造 `MultiModalFeatureSpec` 时会使用这个 identifier：

```python
identifier=self._get_mm_identifier(
    base_mm_hash,
    lora_request,
),
mm_hash=base_mm_hash,
```

位置：`input_processor.py:354` 到 `input_processor.py:368`

这说明 LoRARequest 虽然主要是 adapter 选择信息，但在 multimodal + tower connector LoRA 场景，也会影响多模态 cache key。

### 4.3 构造 EngineCoreRequest

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

位置：`input_processor.py:370` 到 `input_processor.py:385`

这里 LoRARequest 第一次进入 V1 engine core 的请求对象。

---

## 5. EngineCoreRequest 中的 lora_request

`EngineCoreRequest` 定义在：`v1/engine/__init__.py:86`

字段中包含：

```python
class EngineCoreRequest(msgspec.Struct, ...):
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    ...
```

位置：`v1/engine/__init__.py:86` 到 `v1/engine/__init__.py:100`

`EngineCoreRequest` 是跨 Engine 前端和 EngineCore 的轻量请求载体。它保存的是请求已预处理后的信息，包括 token、multimodal features、sampling / pooling 参数、LoRARequest 等。

---

## 6. EngineCore 如何转成 Scheduler Request

EngineCore 里添加请求的位置有两步。

### 6.1 preprocess_add_request()

定义在：`core.py:853`

```python
def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
    ...
    req = Request.from_engine_core_request(request, self.request_block_hasher)
    if req.use_structured_output:
        self.structured_output_manager.grammar_init(req)
    return req, request.current_wave
```

位置：`core.py:853` 到 `core.py:875`

这里把 `EngineCoreRequest` 转成 scheduler 使用的 `Request`。

### 6.2 Request.from_engine_core_request()

定义在：`request.py:197`

```python
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

位置：`request.py:197` 到 `request.py:222`

这一步是 LoRARequest 从 `EngineCoreRequest` 进入 Scheduler 侧 `Request` 的关键。

### 6.3 Request 保存 lora_request

`Request.__init__()` 中：

```python
self.lora_request = lora_request
```

位置：`request.py:70` 到 `request.py:86`

之后 Scheduler 看到的请求对象就带有：

```python
request.lora_request: LoRARequest | None
```

---

## 7. Scheduler 如何处理 LoRARequest

Scheduler 不执行 LoRA forward，也不加载 LoRA 权重。

它只在两个地方关心 LoRA：

```text
1. 调度约束：本轮 batch 中活跃 LoRA 数不能超过 max_loras；
2. 输出数据：首次调度请求时，把 request.lora_request 放进 NewRequestData。
```

### 7.1 RUNNING 请求记录 scheduled_loras

RUNNING 调度结束后会记录当前已调度请求用到的 LoRA：

```python
scheduled_loras: set[int] = set()
if self.lora_config:
    scheduled_loras = set(
        req.lora_request.lora_int_id
        for req in scheduled_running_reqs
        if req.lora_request and req.lora_request.lora_int_id > 0
    )
    assert len(scheduled_loras) <= self.lora_config.max_loras
```

位置：`scheduler.py:614` 到 `scheduler.py:622`

含义：

```text
当前已经进入本轮 batch 的 running 请求，可能已经占用了若干 LoRA slot。
```

### 7.2 WAITING 请求会受 max_loras 限制

在接纳 WAITING 请求时，Scheduler 会根据 `scheduled_loras` 和新请求的 `lora_request` 判断是否还能加入本轮 batch。

典型逻辑是：

```text
如果请求带 lora_request，且这个 lora_int_id 不在 scheduled_loras，
加入后会导致活跃 LoRA 数超过 lora_config.max_loras，
则当前请求不能进入本轮 batch。
```

这一步的目的不是加载 adapter，而是避免 Worker 本轮要同时激活超过配置上限的 LoRA。

### 7.3 SchedulerOutput.NewRequestData 携带 lora_request

`NewRequestData` 定义在：`output.py:30`

字段包含：

```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: torch.Tensor | None = None
    prompt_is_token_ids: list[bool] | None = None
```

位置：`output.py:30` 到 `output.py:41`

从 `Request` 构造时：

```python
lora_request=request.lora_request,
```

位置：`output.py:47` 到 `output.py:65`

也就是说，只有 `scheduled_new_reqs` 里会携带完整的 `lora_request`。

已经在 Worker 缓存过的请求后续进入 `scheduled_cached_reqs`，不会每轮重复发送 `LoRARequest`；Worker 会用自己的 `CachedRequestState` 保存它。

---

## 8. SchedulerOutput 到 Worker 的状态缓存

`SchedulerOutput` 进入 Worker / ModelRunner 后，`GPUModelRunner._update_states()` 会把 `scheduled_new_reqs` 转成 Worker 侧缓存状态。

相关代码位置：`gpu_model_runner.py:1194` 到 `gpu_model_runner.py:1238`

核心代码：

```python
req_state = CachedRequestState(
    req_id=req_id,
    prompt_token_ids=new_req_data.prompt_token_ids,
    prompt_embeds=new_req_data.prompt_embeds,
    prompt_is_token_ids=new_req_data.prompt_is_token_ids,
    mm_features=new_req_data.mm_features,
    sampling_params=sampling_params,
    pooling_params=pooling_params,
    generator=generator,
    block_ids=new_req_data.block_ids,
    num_computed_tokens=new_req_data.num_computed_tokens,
    output_token_ids=[],
    lora_request=new_req_data.lora_request,
)
self.requests[req_id] = req_state
```

位置：`gpu_model_runner.py:1224` 到 `gpu_model_runner.py:1238`

`CachedRequestState` 定义在：`gpu_input_batch.py:33`

字段包含：

```python
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    generator: torch.Generator | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]
    ...
    lora_request: LoRARequest | None = None
```

位置：`gpu_input_batch.py:33` 到 `gpu_input_batch.py:50`

因此 Worker 侧有两层请求状态：

```text
GPUModelRunner.requests[req_id].lora_request
  保存“这个请求绑定哪个 LoRA”。

InputBatch.request_lora_mapping[req_index]
  保存“当前 batch 第 req_index 行用哪个 LoRA id”。
```

---

## 9. InputBatch 如何维护 LoRA 状态

`InputBatch` 初始化时创建 LoRA 相关结构：

```python
self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
self.lora_id_to_request_ids: dict[int, set[str]] = {}
self.lora_id_to_lora_request: dict[int, LoRARequest] = {}
```

位置：`gpu_input_batch.py:244` 到 `gpu_input_batch.py:247`

三个结构分别表示：

| 结构 | 含义 |
|---|---|
| `request_lora_mapping[req_index]` | 当前 batch 中某一行请求使用的 LoRA id，0 表示无 LoRA |
| `lora_id_to_request_ids` | 某个 LoRA id 当前被哪些请求使用 |
| `lora_id_to_lora_request` | LoRA id 对应的完整 `LoRARequest`，用于加载 adapter |

### 9.1 add_request()

`InputBatch.add_request()` 定义在：`gpu_input_batch.py:335`

当请求加入当前执行 batch 时：

```python
if request.lora_request:
    lora_id = request.lora_request.lora_int_id
    if lora_id not in self.lora_id_to_request_ids:
        self.lora_id_to_request_ids[lora_id] = set()

    self.request_lora_mapping[req_index] = lora_id
    self.lora_id_to_request_ids[lora_id].add(request.req_id)
    self.lora_id_to_lora_request[lora_id] = request.lora_request
else:
    # No LoRA
    self.request_lora_mapping[req_index] = 0
```

位置：`gpu_input_batch.py:468` 到 `gpu_input_batch.py:479`

这一步把请求级对象压缩成 batch 级整数 mapping。

### 9.2 remove_request()

请求离开当前 batch 时，会清理 LoRA 反向索引：

```python
lora_id = self.request_lora_mapping[req_index]
if lora_id != 0:
    lora_req_ids = self.lora_id_to_request_ids[lora_id]
    lora_req_ids.discard(req_id)
    if not lora_req_ids:
        del self.lora_id_to_request_ids[lora_id]
        del self.lora_id_to_lora_request[lora_id]
    self.request_lora_mapping[req_index] = 0
```

位置：`gpu_input_batch.py:530` 到 `gpu_input_batch.py:538`

注意：这只是从当前 batch 状态中移除 LoRA 映射，不一定意味着 adapter 权重立刻从 LoRA manager 中卸载。是否卸载取决于具体 LoRA manager 的缓存策略。

### 9.3 make_lora_inputs()

真正执行前，InputBatch 会把每个 request 的 LoRA id 展开到 token 级：

```python
def make_lora_inputs(
    self, num_scheduled_tokens: np.ndarray, num_sampled_tokens: np.ndarray
) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
    req_lora_mapping = self.request_lora_mapping[: self.num_reqs]
    prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
    token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))

    active_lora_requests: set[LoRARequest] = set(
        self.lora_id_to_lora_request.values()
    )

    return prompt_lora_mapping, token_lora_mapping, active_lora_requests
```

位置：`gpu_input_batch.py:976` 到 `gpu_input_batch.py:999`

返回值含义：

| 返回值 | 含义 |
|---|---|
| `prompt_lora_mapping` | 每个 sampled token 使用哪个 LoRA id |
| `token_lora_mapping` | 每个 scheduled input token 使用哪个 LoRA id |
| `active_lora_requests` | 当前 batch 涉及的完整 LoRARequest 集合 |

`0` 表示这段 token 不使用 LoRA。

---

## 10. ModelRunner 何时激活 LoRA

`GPUModelRunner` 在准备输入 / 元数据时会设置当前 batch 的 LoRA。

相关代码：

```python
# Hot-Swap lora model
if self.lora_config:
    assert (
        np.sum(num_sampled_tokens)
        <= self.vllm_config.scheduler_config.max_num_batched_tokens
    )
    self.set_active_loras(
        self.input_batch, num_scheduled_tokens, num_sampled_tokens
    )
```

位置：`gpu_model_runner.py:2193` 到 `gpu_model_runner.py:2201`

`set_active_loras()` 来自 `LoRAModelRunnerMixin`。

### 10.1 LoRAModelRunnerMixin.set_active_loras()

定义在：`lora_model_runner_mixin.py:73`

```python
def set_active_loras(
    self,
    input_batch: InputBatch,
    num_scheduled_tokens: np.ndarray,
    num_sampled_tokens: np.ndarray | None = None,
    mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
) -> None:
    if num_sampled_tokens is None:
        num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

    prompt_lora_mapping, token_lora_mapping, lora_requests = (
        input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
    )
    return self._set_active_loras(
        prompt_lora_mapping, token_lora_mapping, lora_requests, mapping_type
    )
```

位置：`lora_model_runner_mixin.py:73` 到 `lora_model_runner_mixin.py:91`

### 10.2 _set_active_loras()

定义在：`lora_model_runner_mixin.py:48`

```python
lora_mapping = LoRAMapping(
    token_lora_mapping,
    prompt_lora_mapping,
    is_prefill=True,
    type=mapping_type,
)
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

位置：`lora_model_runner_mixin.py:48` 到 `lora_model_runner_mixin.py:68`

这里把 InputBatch 的 numpy / tuple 映射包装成 `LoRAMapping`，交给 LoRA manager。

`mapping_type` 默认是：

```python
LoRAMappingType.LANGUAGE
```

表示语言模型主体的 LoRA mapping。

---

## 11. LoRA manager 如何加载和激活 adapter

ModelRunner 加载模型时，如果启用 LoRA，会把基础模型包装成支持 LoRA 的模型。

`LoRAModelRunnerMixin.load_lora_model()`：

```python
if not supports_lora(model):
    raise ValueError(f"{model.__class__.__name__} does not support LoRA yet.")

self.lora_manager = LRUCacheWorkerLoRAManager(
    vllm_config,
    device,
    model.embedding_modules,
)
return self.lora_manager.create_lora_manager(model, vllm_config)
```

位置：`lora_model_runner_mixin.py:31` 到 `lora_model_runner_mixin.py:46`

`GPUModelRunner.load_model()` 中调用：

```python
if self.lora_config:
    self.model = self.load_lora_model(
        self.model, self.vllm_config, self.device
    )
```

位置：`gpu_model_runner.py:5167` 到 `gpu_model_runner.py:5169`

### 11.1 set_active_adapters()

`WorkerLoRAManager.set_active_adapters()` 定义在：`worker_manager.py:183`

```python
def set_active_adapters(self, requests: set[Any], mapping: Any | None) -> None:
    self._apply_adapters(requests)
    if mapping is not None:
        self._adapter_manager.set_adapter_mapping(mapping)
```

位置：`worker_manager.py:183` 到 `worker_manager.py:186`

它分两步：

```text
1. _apply_adapters(requests)：确保当前 batch 需要的 adapter 已加载 / 激活；
2. set_adapter_mapping(mapping)：把 token 到 LoRA slot 的映射交给底层 LoRA layer。
```

### 11.2 LRUCacheWorkerLoRAManager._apply_adapters()

V1 ModelRunner 使用的是 `LRUCacheWorkerLoRAManager`。

```python
loras_map = {
    lora_request.lora_int_id: lora_request
    for lora_request in lora_requests
    if lora_request
}
if len(loras_map) > self._adapter_manager.lora_slots:
    raise RuntimeError(...)
for lora in loras_map.values():
    self.add_adapter(lora)
```

位置：`worker_manager.py:258` 到 `worker_manager.py:271`

这里再次检查当前 batch 的活跃 LoRA 数不能超过 GPU LoRA slots。

### 11.3 add_adapter()

```python
if (
    lora_request.lora_int_id not in self.list_adapters()
    or lora_request.load_inplace
):
    lora = self._load_adapter(lora_request)
    self._adapter_manager.remove_adapter(lora.id)
    if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
        self._adapter_manager.remove_oldest_adapter()
    loaded = self._adapter_manager.add_adapter(lora)
else:
    loaded = (
        self._adapter_manager.get_adapter(lora_request.lora_int_id) is not None
    )
self._adapter_manager.activate_adapter(lora_request.lora_int_id)
return loaded
```

位置：`worker_manager.py:273` 到 `worker_manager.py:307`

含义：

```text
- adapter 不在缓存中：从 lora_path 加载；
- load_inplace=True：即使已有同 ID adapter，也重新加载并替换；
- 超过 CPU / manager 容量：按 LRU 淘汰旧 adapter；
- 最后 activate_adapter(lora_int_id)。
```

### 11.4 _load_adapter()

加载权重位置：`worker_manager.py:99`

关键步骤：

```text
1. 解析 lora_path 的绝对路径；
2. PEFTHelper.from_local_dir() 读取 adapter 配置；
3. validate_legal() 校验 LoRA 配置是否合法；
4. LoRAModel.from_local_checkpoint() 加载权重；
5. 设置 is_3d_lora_weight 标记。
```

对应代码位置：`worker_manager.py:99` 到 `worker_manager.py:162`

因此，`LoRARequest.lora_path` 真正被使用是在 Worker 侧加载 adapter 时，而不是 Engine / Scheduler 阶段。

---

## 12. 多模态 LoRA 的特殊路径

如果模型支持 tower / connector LoRA，encoder 侧也需要单独设置 LoRA mapping。

`GPUModelRunner._execute_mm_encoder()` 中：

```python
if self.lora_config and self.lora_manager.supports_tower_connector_lora():
    ...
    tower_mapping = LoRAMapping(
        tuple(token_lora_mapping),
        tuple(prompt_lora_mapping),
        is_prefill=True,
        type=LoRAMappingType.TOWER,
    )
    self.lora_manager.set_active_adapters(lora_requests, tower_mapping)
```

位置：`gpu_model_runner.py:2941` 到 `gpu_model_runner.py:2973`

如果模型还有 connector LoRA，会继续设置：

```python
connector_mapping = LoRAMapping(
    index_mapping=tuple(connector_token_mapping.tolist()),
    prompt_mapping=tuple(prompt_lora_mapping),
    is_prefill=True,
    type=LoRAMappingType.CONNECTOR,
)

self.lora_manager.set_active_adapters(
    lora_requests,
    connector_mapping,
)
```

位置：`gpu_model_runner.py:2975` 到 `gpu_model_runner.py:3008`

这条路径和语言模型主体的 LoRA mapping 不同：

```text
LANGUAGE mapping：用于主语言模型 token forward；
TOWER mapping：用于 multimodal encoder tower；
CONNECTOR mapping：用于 multimodal connector。
```

这也解释了前面 `InputProcessor._get_mm_identifier()` 为什么在 `enable_tower_connector_lora=True` 时把 `lora_name` 加进 multimodal identifier。

---

## 13. add_lora / remove_lora 控制面和请求流的区别

`LLMEngine` 暴露了 LoRA 管理接口：

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    """Load a new LoRA adapter into the engine for future requests."""
    return self.engine_core.add_lora(lora_request)

def remove_lora(self, lora_id: int) -> bool:
    """Remove an already loaded LoRA adapter."""
    return self.engine_core.remove_lora(lora_id)
```

位置：`llm_engine.py:403` 到 `llm_engine.py:409`

EngineCore 再转给 executor / worker：

```python
return self.model_executor.add_lora(lora_request)
return self.model_executor.remove_lora(lora_id)
return self.model_executor.pin_lora(lora_id)
```

位置：`core.py:823` 到 `core.py:832`

这条是控制面：

```text
显式加载 / 删除 / pin adapter。
```

请求里的 `lora_request` 是数据面：

```text
声明这个请求要使用哪个 adapter。
```

两者关系：

```text
- 请求可以携带 LoRARequest，Worker 在 set_active_adapters() 时按需加载；
- 也可以先通过 add_lora() 预加载 adapter，再让请求引用同一个 lora_int_id；
- remove_lora() / pin_lora() 影响 adapter manager 的缓存状态，不改变已经排队请求里的 LoRARequest 字段。
```

---

## 14. n > 1 时 LoRARequest 如何处理

`LLMEngine.add_request()` 支持 `SamplingParams.n > 1`。

当 `n == 1` 时，原请求直接进入 EngineCore：

```python
self.output_processor.add_request(request, prompt_text, None, 0)
self.engine_core.add_request(request)
```

位置：`llm_engine.py:270` 到 `llm_engine.py:277`

当 `n > 1` 时，会 fan out 子请求：

```python
parent_req = ParentRequest(request)
for idx in range(n):
    request_id, child_params = parent_req.get_child_info(idx)
    child_request = request if idx == n - 1 else copy(request)
    child_request.request_id = request_id
    child_request.sampling_params = child_params
    ...
    self.engine_core.add_request(child_request)
```

位置：`llm_engine.py:279` 到 `llm_engine.py:294`

这里 `copy(request)` 会保留 `lora_request` 引用，因此同一个用户请求 fan out 出来的多个 child request 会使用同一个 LoRARequest。

---

## 15. 一轮执行中 LoRA 的具体状态流

可以把 Worker 侧一轮执行简化为：

```python
# SchedulerOutput 进入 Worker
for new_req_data in scheduler_output.scheduled_new_reqs:
    req_state = CachedRequestState(
        ...,
        lora_request=new_req_data.lora_request,
    )
    self.requests[req_id] = req_state

# 本轮真正参与执行的请求加入 InputBatch
for req_state in reqs_to_add:
    input_batch.add_request(req_state)
    # request_lora_mapping[req_index] = req_state.lora_request.lora_int_id or 0

# 准备模型输入时
if self.lora_config:
    prompt_lora_mapping, token_lora_mapping, lora_requests = \
        input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)

    lora_mapping = LoRAMapping(
        token_lora_mapping,
        prompt_lora_mapping,
        is_prefill=True,
        type=LoRAMappingType.LANGUAGE,
    )

    lora_manager.set_active_adapters(lora_requests, lora_mapping)

# forward 时 LoRA layer 根据 mapping 选择 adapter
model_forward(...)
```

这个过程说明：

```text
LoRARequest 在请求生命周期内是稳定的；
LoRA mapping 在每个 scheduler step / batch 中重新生成；
adapter 权重是否需要加载由当前 active_lora_requests 和 LoRA manager 缓存状态决定。
```

---

## 16. 容易混淆的点

### 16.1 LoRARequest 不等于 adapter 已加载

`LoRARequest` 只是描述：

```text
这个请求想使用哪个 adapter，以及 adapter 从哪里加载。
```

实际加载发生在：

```text
WorkerLoRAManager._load_adapter()
```

位置：`worker_manager.py:99`

### 16.2 Scheduler 不执行 LoRA

Scheduler 最多做：

```text
- 限制本轮活跃 LoRA 数；
- 把 lora_request 放进 NewRequestData。
```

真正 LoRA 激活在 ModelRunner：

```text
GPUModelRunner -> set_active_loras() -> lora_manager.set_active_adapters()
```

### 16.3 scheduled_cached_reqs 不重复带 LoRARequest

`LoRARequest` 只在新请求首次调度时随 `NewRequestData` 发给 Worker。

后续 step 里，Worker 从：

```text
GPUModelRunner.requests[req_id].lora_request
```

读取缓存的 LoRARequest。

### 16.4 lora_int_id 是执行映射 ID

`InputBatch.request_lora_mapping`、`LoRAMapping`、底层 LoRA layer 都使用 `lora_int_id`。

因此：

```text
同一个 adapter 的 lora_int_id 必须全局稳定且唯一。
```

源码注释也说明：

```text
lora_int_id must be globally unique for a given adapter.
This is currently not enforced in vLLM.
```

位置：`lora/request.py:16` 到 `lora/request.py:17`

### 16.5 lora_name 用于对象去重和多模态 cache key

`LoRARequest.__eq__` / `__hash__` 按 `lora_name` 判断。

在 tower connector LoRA 场景，多模态 identifier 也会用：

```text
f"{lora_request.lora_name}:{mm_hash}"
```

所以 `lora_name` 不只是展示字段。

### 16.6 无 LoRA 请求使用 id 0

`InputBatch.add_request()` 中，无 LoRA 请求设置：

```python
self.request_lora_mapping[req_index] = 0
```

位置：`gpu_input_batch.py:477` 到 `gpu_input_batch.py:479`

这使同一个 batch 可以混合：

```text
LoRA 请求 + 非 LoRA 请求
```

底层 mapping 通过 0 区分无 LoRA token。

### 16.7 load_inplace 会强制替换同 ID adapter

如果：

```python
lora_request.load_inplace = True
```

`LRUCacheWorkerLoRAManager.add_adapter()` 会重新加载并替换已有同 ID adapter。

位置：`worker_manager.py:279` 到 `worker_manager.py:292`

这适合 adapter 内容变了但希望复用同一个 `lora_int_id` 的场景。

---

## 17. 总结

`LoRARequest` 的请求链路可以分成三段：

```text
入口段：
  LLMEngine.add_request(lora_request=...)
    → InputProcessor._validate_lora()
    → EngineCoreRequest.lora_request

调度段：
  EngineCore.preprocess_add_request()
    → Request.lora_request
    → Scheduler.schedule()
    → NewRequestData.lora_request

执行段：
  GPUModelRunner._update_states()
    → CachedRequestState.lora_request
    → InputBatch.request_lora_mapping
    → InputBatch.make_lora_inputs()
    → LoRAMapping
    → WorkerLoRAManager.set_active_adapters()
    → LoRA layer forward
```

最核心的理解是：

```text
LoRARequest 是请求级 adapter 绑定信息；
Scheduler 只把它作为请求属性携带和做 batch 约束；
Worker 才把它变成 token 级 LoRA mapping，并按需加载 / 激活 adapter。
```
