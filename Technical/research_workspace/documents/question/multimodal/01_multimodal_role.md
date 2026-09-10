# 01. Multimodal 在 vLLM V1 里负责什么？

源码位置：

- `code/vllm/vllm/config/multimodal.py`
- `code/vllm/vllm/multimodal/`
- `code/vllm/vllm/renderers/base.py`
- `code/vllm/vllm/inputs/preprocess.py`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/model_executor/models/`

本问题关注：Multimodal 子系统在 vLLM V1 中的定位和职责边界。它不是单独的模型执行器，也不是 Scheduler 的替代品，而是一条贯穿输入处理、调度、encoder 执行、embedding 拼接和缓存复用的辅助链路。

---

## 0. 梳理规划

本篇按“先定角色，再走主链路，再拆关键对象，最后总结边界”的方式梳理 vLLM V1 的多模态链路。

要回答的问题分成 12 组：

```text
1. Multimodal 在 vLLM V1 里是哪一层？
2. 它和 InputProcessor / Scheduler / ModelRunner / 模型实现的关系是什么？
3. 用户输入里的 image / audio / video / prompt_embeds 如何变成 MultiModalFeatureSpec？
4. placeholder token、mm_position、mm_hash、identifier 分别是什么？
5. Scheduler 如何决定本轮要不要执行 multimodal encoder？
6. EncoderCacheManager 管什么，不管什么？
7. GPUModelRunner 如何执行 _execute_mm_encoder()？
8. encoder output 如何进入 inputs_embeds？
9. 多模态 processor cache 和 encoder cache 有什么区别？
10. prompt_embeds / embedding-only 模态如何接入？
11. encoder-decoder 多模态模型有什么特殊路径？
12. 多模态链路不负责哪些事情？
```

阅读顺序建议：

```text
01_multimodal_role.md
  → input_processor / scheduler / model_runner 相关文档
  → executor_worker_model_runner/03_model_runner_role.md
  → executor_worker_model_runner/05_input_batch_and_state_update.md
  → executor_worker_model_runner/06_prepare_inputs_and_attention_metadata.md
```

本篇重点讲多模态总定位和端到端主链路，不会逐个模型展开 Llava、Qwen2-VL、Whisper、Voxtral 等具体实现。

---

## 1. 一句话回答

`Multimodal` 是 vLLM V1 中把非文本输入接入普通模型执行链路的输入准备与 encoder 辅助系统。

它负责：

```text
1. 识别模型是否支持多模态输入；
2. 根据模型注册的 processor 解析 image / audio / video / embeds；
3. 把多模态输入转换成模型特定的 mm_kwargs；
4. 维护多模态 placeholder 和 prompt token 位置关系；
5. 生成 MultiModalFeatureSpec；
6. 为 Scheduler 提供 encoder token / cache budget 依据；
7. 让 Scheduler 决定本轮要执行哪些 encoder input；
8. 让 GPUModelRunner 执行模型的 multimodal encoder；
9. 缓存 encoder output；
10. 把 encoder output 按 placeholder 位置拼回 inputs_embeds；
11. 支持 processor cache、encoder cache、EC connector、prompt_embeds、M-RoPE / XD-RoPE 等特殊能力。
```

它不负责：

```text
1. 不负责用户请求队列调度；
2. 不负责 decoder KV block 分配；
3. 不负责 attention backend 主链路选择；
4. 不负责最终 token sampling；
5. 不负责 detokenize；
6. 不负责 OpenAI response 协议包装；
7. 不负责模型权重加载本身；
8. 不负责把 ModelRunnerOutput 转成 RequestOutput。
```

最小主线是：

```text
multi_modal_data
  → Renderer / MultiModalProcessor
  → MultiModalInput
  → InputProcessor.process_inputs()
  → MultiModalFeatureSpec
  → EngineCoreRequest.mm_features
  → Request.mm_features
  → Scheduler._try_schedule_encoder_inputs()
  → SchedulerOutput.scheduled_encoder_inputs
  → GPUModelRunner._execute_mm_encoder()
  → GPUModelRunner.encoder_cache
  → GPUModelRunner._gather_mm_embeddings()
  → model.embed_input_ids(..., multimodal_embeddings=...)
  → inputs_embeds
  → decoder forward / logits / sampling
```

一句话压缩：

```text
Multimodal 负责把非文本输入变成 decoder forward 可以消费的 embeddings，并把这件事接入 vLLM V1 的调度、缓存和执行链路。
```

---

## 2. Multimodal 在整体链路中的位置

从用户请求到模型 forward，相关组件关系是：

```text
API / LLM.generate / chat
  → Renderer / InputPreprocessor
  → MultiModalProcessor
  → InputProcessor
  → EngineCoreRequest
  → Request
  → Scheduler
  → SchedulerOutput
  → Executor
  → Worker
  → GPUModelRunner
  → model.embed_multimodal()
  → model.embed_input_ids()
  → model.forward()
```

对应职责：

```text
Renderer / MultiModalProcessor：把原始多模态数据处理成 token ids、mm_kwargs、placeholder、hash；
InputProcessor：把处理结果整理成 EngineCoreRequest.mm_features；
Request：保存请求级 mm_features；
Scheduler：决定本轮哪些 mm_features 的 encoder output 必须可用；
EncoderCacheManager：管理 encoder output 的逻辑缓存账本；
GPUModelRunner：真正执行 multimodal encoder，缓存 output，并把 output 拼进 inputs_embeds；
模型实现：提供 embed_multimodal() 和 embed_input_ids() 等模型特定能力。
```

注意这条链路没有替代普通文本生成链路。

多模态只是在 prefill / prompt 阶段改变模型输入准备方式，后续仍然进入：

```text
_model_forward()
  → compute_logits()
  → sample_tokens()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor
```

---

## 3. MultiModalConfig：多模态行为由配置驱动

配置定义在：`code/vllm/vllm/config/multimodal.py:73`

```python
@config
class MultiModalConfig:
    """Controls the behavior of multimodal models."""
```

它控制的是多模态模型的输入、缓存和 encoder 执行策略，不是某个单独模型的 forward 实现。

### 3.1 基础开关和限额

关键字段包括：

```text
language_model_only
limit_per_prompt
enable_mm_embeds
media_io_kwargs
mm_processor_kwargs
```

位置：`code/vllm/vllm/config/multimodal.py:77` 到 `code/vllm/vllm/config/multimodal.py:122`

含义：

| 字段 | 作用 |
|---|---|
| `language_model_only` | 把多模态输入全部禁用，等价于每种 modality limit 为 0 |
| `limit_per_prompt` | 控制每个 prompt 中 image / video / audio 等 item 上限 |
| `enable_mm_embeds` | 允许传入预计算多模态 embeddings |
| `media_io_kwargs` | 控制媒体读取参数，例如 video num_frames |
| `mm_processor_kwargs` | 传给模型 processor 的额外参数 |

`get_limit_per_prompt()` 会处理 `language_model_only` 和默认限额：

```python
if self.language_model_only:
    return 0

limit_data = self.limit_per_prompt.get(modality)

if limit_data is None:
    return 999

return limit_data.count
```

位置：`code/vllm/vllm/config/multimodal.py:310` 到 `code/vllm/vllm/config/multimodal.py:324`

这说明未显式配置的 modality 默认不是 0，而是 999。

### 3.2 processor cache 配置

关键字段：

```text
mm_processor_cache_gb
mm_processor_cache_type
mm_shm_cache_max_object_size_mb
mm_tensor_ipc
```

位置：`code/vllm/vllm/config/multimodal.py:123` 到 `code/vllm/vllm/config/multimodal.py:199`

它们控制的是“多模态 processor / mapper 的输出缓存”，也就是避免重复执行图片预处理、音频预处理、HF processor 等 CPU 侧处理。

这不是 encoder output cache。

### 3.3 encoder 执行配置

关键字段：

```text
mm_encoder_only
mm_encoder_tp_mode
mm_encoder_attn_backend
mm_encoder_attn_dtype
mm_encoder_fp8_scale_path
mm_encoder_fp8_scale_save_path
mm_encoder_fp8_scale_save_margin
skip_mm_profiling
video_pruning_rate
mm_tensor_ipc
mm_ipc_gpu_memory_gb
```

位置：`code/vllm/vllm/config/multimodal.py:139` 到 `code/vllm/vllm/config/multimodal.py:209`

它们影响的是多模态 encoder 执行和多模态 tensor 传输：

```text
- 是否只跑 encoder；
- encoder TP 是切权重还是切数据；
- ViT encoder attention backend；
- encoder attention FP8 和 scale 读写；
- 是否跳过多模态 profiling；
- video token pruning；
- 多模态 tensor IPC 方式和前端 GPU 多模态内存预留。
```

### 3.4 compute_hash()

`MultiModalConfig.compute_hash()` 只把会影响计算图的字段加入 hash：

```python
factors: list[Any] = [
    self.mm_encoder_attn_backend.name
    if self.mm_encoder_attn_backend is not None
    else None,
    self.mm_encoder_tp_mode,
    self.mm_encoder_attn_dtype,
    self.mm_encoder_fp8_scale_path,
]
```

位置：`code/vllm/vllm/config/multimodal.py:287` 到 `code/vllm/vllm/config/multimodal.py:308`

这说明 vLLM 区分：

```text
影响 processor 行为的配置；
影响 encoder / forward 计算图的配置。
```

---

## 4. MultiModalRegistry：模型到 processor 的注册中心

注册中心定义在：`code/vllm/vllm/multimodal/registry.py:98`

```python
class MultiModalRegistry:
    """
    A registry that dispatches data processing according to the model.
    """
```

### 4.1 supports_multimodal_inputs()

入口：`code/vllm/vllm/multimodal/registry.py:103`

它判断一个模型是否真正启用多模态输入：

```python
if not model_config.is_multimodal_model:
    return False

mm_config = model_config.get_multimodal_config()
...
if all(mm_config.get_limit_per_prompt(modality) == 0
       for modality in info.supported_mm_limits):
    if mm_config.enable_mm_embeds:
        return True
    return False
```

位置：`code/vllm/vllm/multimodal/registry.py:103` 到 `code/vllm/vllm/multimodal/registry.py:140`

这里有三个关键点：

```text
1. model_config.is_multimodal_model 为 False，则直接文本模式；
2. 模型类必须注册 multimodal processor；
3. 所有 modality limit 都是 0 时，通常文本模式；但 enable_mm_embeds=True 时仍保留多模态基础设施。
```

### 4.2 register_processor()

模型通过 registry 注册 processor：

```python
def register_processor(
    self,
    processor,
    *,
    info,
    dummy_inputs,
):
```

位置：`code/vllm/vllm/multimodal/registry.py:142` 到 `code/vllm/vllm/multimodal/registry.py:174`

注册后，模型类会持有：

```text
_processor_factory.info
_processor_factory.dummy_inputs
_processor_factory.processor
```

这也是为什么不同 VLM 模型可以有不同输入处理逻辑。

### 4.3 create_processor()

真正创建 processor 的入口是：

```python
def create_processor(...):
```

位置：`code/vllm/vllm/multimodal/registry.py:211` 到 `code/vllm/vllm/multimodal/registry.py:230`

它会：

```text
1. 确认模型是多模态模型；
2. 找到模型类；
3. 取出模型注册的 processor factory；
4. 创建 InputProcessingContext；
5. build_processor()。
```

### 4.4 processor cache 工厂

`MultiModalRegistry` 也负责根据配置创建 processor cache：

```text
processor_cache_from_config()
processor_only_cache_from_config()
engine_receiver_cache_from_config()
worker_receiver_cache_from_config()
```

位置：`code/vllm/vllm/multimodal/registry.py:294` 到 `code/vllm/vllm/multimodal/registry.py:347`

其中 `_get_cache_type()` 会根据配置和并行模式返回：

```text
None
processor_only
lru
shm
```

位置：`code/vllm/vllm/multimodal/registry.py:268` 到 `code/vllm/vllm/multimodal/registry.py:292`

这说明 processor cache 是多进程 / API process / engine process 协作的一部分。

---

## 5. MultiModalDataParser：解析原始 image / audio / video / embeds

解析器定义在：`code/vllm/vllm/multimodal/parse.py:489`

```python
class MultiModalDataParser:
    """
    Parses MultiModalDataDict into MultiModalDataItems.
    """
```

它把用户传入的 `multi_modal_data` 标准化成内部 item 列表。

### 5.1 支持的内置 modality

`_get_subparsers()` 返回：

```python
return {
    "audio": self._parse_audio_data,
    "image": self._parse_image_data,
    "video": self._parse_video_data,
    "vision_chunk": self._parse_vision_chunk_data,
}
```

位置：`code/vllm/vllm/multimodal/parse.py:692` 到 `code/vllm/vllm/multimodal/parse.py:698`

如果输入 key 不在其中：

```python
raise ValueError(f"Unsupported modality: {k}")
```

位置：`code/vllm/vllm/multimodal/parse.py:700` 到 `code/vllm/vllm/multimodal/parse.py:710`

### 5.2 image / audio / video 的处理方向

它会把不同形式的输入规整为 item 列表：

```text
image：PIL / ndarray / torch.Tensor / list
video：帧序列 / ndarray / tensor / 带 metadata 的 tuple
音频：array / tensor / list / 带采样率的 tuple
```

关键位置：

```text
_parse_audio_data(): code/vllm/vllm/multimodal/parse.py:567
_parse_image_data(): code/vllm/vllm/multimodal/parse.py:606
_parse_video_data(): code/vllm/vllm/multimodal/parse.py:634
```

如果传入的是 embeddings，解析器会返回 embedding item：

```python
if self.is_embeddings(data):
    return ImageEmbeddingItems(data, self.expected_hidden_size)
```

位置：`code/vllm/vllm/multimodal/parse.py:613` 到 `code/vllm/vllm/multimodal/parse.py:614`

这就是 `enable_mm_embeds` 能接入的基础。

---

## 6. Renderer / InputPreprocessor：多模态输入如何进入 engine input

旧入口在 `InputPreprocessor`，新链路里 Renderer 负责更多工作。

`InputPreprocessor._process_multimodal()` 很薄：

```python
return self.renderer._process_multimodal(...)
```

位置：`code/vllm/vllm/inputs/preprocess.py:90` 到 `code/vllm/vllm/inputs/preprocess.py:109`

真正处理在：`code/vllm/vllm/renderers/base.py:728`

```python
def _process_multimodal(
    self,
    prompt,
    mm_data,
    mm_uuids,
    mm_processor_kwargs,
    tokenization_kwargs,
    *,
    skip_mm_cache=False,
) -> MultiModalInput:
```

### 6.1 Renderer 的处理步骤

主流程：

```python
mm_processor = self.get_mm_processor()
mm_data_items = mm_processor.info.parse_mm_data(mm_data)
mm_uuid_items = parse_mm_uuids(mm_uuids)
mm_uuid_items = self._process_mm_uuids(...)
mm_processor_inputs = MMProcessorInputs(...)
mm_inputs = mm_processor.apply(mm_processor_inputs, mm_timing_ctx)
return mm_inputs
```

位置：`code/vllm/vllm/renderers/base.py:740` 到 `code/vllm/vllm/renderers/base.py:766`

输出的 `mm_inputs` 是 `MultiModalInput`，通常包含：

```text
prompt_token_ids
mm_kwargs
mm_placeholders
mm_hashes
```

### 6.2 mm_uuids 和 hash

Renderer 会处理 `multi_modal_uuids`：

```python
mm_uuid_items = parse_mm_uuids(mm_uuids)
mm_uuid_items = self._process_mm_uuids(...)
```

位置：`code/vllm/vllm/renderers/base.py:745` 到 `code/vllm/vllm/renderers/base.py:750`

如果同时关闭 prefix caching 和 processor cache，Renderer 会用请求局部 id 覆盖 uuid：

```python
if mm_processor_cache_gb == 0 and not enable_prefix_caching:
    mm_uuid_items = {
        modality: [f"{mm_req_id}-{modality}-{i}" ...]
    }
```

位置：`code/vllm/vllm/renderers/base.py:699` 到 `code/vllm/vllm/renderers/base.py:725`

这说明多模态 hash/uuid 不只是为了 processor cache，也会影响 encoder output 复用。

---

## 7. MultiModalFeatureSpec：贯穿 V1 多模态链路的核心对象

定义位置：`code/vllm/vllm/multimodal/inputs.py:301`

```python
@dataclass
class MultiModalFeatureSpec:
    """
    Represents a single multimodal input with its processed data and metadata.
    """
```

字段：

```python
data: MultiModalKwargsItem | None
modality: str
identifier: str
mm_position: PlaceholderRange
mm_hash: str | None = None
```

位置：`code/vllm/vllm/multimodal/inputs.py:311` 到 `code/vllm/vllm/multimodal/inputs.py:332`

字段含义：

| 字段 | 含义 |
|---|---|
| `data` | 该 item 的 processor 输出，即模型 encoder 需要的 kwargs。缓存命中时可以是 None |
| `modality` | 输入模态，例如 image / audio / video / prompt_embeds |
| `identifier` | encoder output cache 的 key，可能带 LoRA 前缀 |
| `mm_position` | placeholder 在 prompt token 序列中的位置和长度 |
| `mm_hash` | processor output cache 的 hash，通常不带 LoRA 前缀 |

### 7.1 一个请求多个多模态 item

源码注释说明：

```text
A request containing multiple multimodal items will have one MultiModalFeatureSpec per item.
```

位置：`code/vllm/vllm/multimodal/inputs.py:304` 到 `code/vllm/vllm/multimodal/inputs.py:308`

因此一个 prompt 里有 2 张图和 1 段音频时，通常会有 3 个 `MultiModalFeatureSpec`。

### 7.2 data 可以为 None

注释说明：

```text
Can be None if the item is cached, to skip IPC between API server and engine core processes.
```

位置：`code/vllm/vllm/multimodal/inputs.py:311` 到 `code/vllm/vllm/multimodal/inputs.py:317`

也就是说，`MultiModalFeatureSpec` 不一定携带 tensor payload。它也可以只携带 cache key 和位置信息。

---

## 8. InputProcessor：把 MultiModalInput 变成 EngineCoreRequest.mm_features

入口：`code/vllm/vllm/v1/engine/input_processor.py:242`

```python
def process_inputs(...) -> EngineCoreRequest:
```

### 8.1 初始化时判断多模态能力

`InputProcessor.__init__()` 会判断当前模型是否支持多模态：

```python
self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:58`

如果支持，会计算 encoder budget：

```python
mm_budget = MultiModalBudget(vllm_config, mm_registry)
self.mm_encoder_cache_size = mm_budget.encoder_cache_size
self.skip_prompt_length_check = mm_budget.processor.info.skip_prompt_length_check
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:61` 到 `code/vllm/vllm/v1/engine/input_processor.py:67`

这说明输入校验阶段已经需要知道多模态 encoder cache size。

### 8.2 process_inputs() 中整理 mm_features

如果 decoder input 是 multimodal：

```python
if decoder_inputs["type"] == "multimodal":
    decoder_mm_inputs = decoder_inputs["mm_kwargs"]
    decoder_mm_positions = decoder_inputs["mm_placeholders"]
    decoder_mm_hashes = decoder_inputs["mm_hashes"]
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:332` 到 `code/vllm/vllm/v1/engine/input_processor.py:338`

随后把不同 modality 的字典压平成按 prompt 位置排序的列表：

```python
sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

mm_features = []
for modality, idx in sorted_mm_idxs:
    base_mm_hash = decoder_mm_hashes[modality][idx]
    mm_features.append(
        MultiModalFeatureSpec(...)
    )
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:349` 到 `code/vllm/vllm/v1/engine/input_processor.py:368`

最后放进 `EngineCoreRequest`：

```python
return EngineCoreRequest(
    ...
    mm_features=mm_features,
    ...
)
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:370` 到 `code/vllm/vllm/v1/engine/input_processor.py:385`

### 8.3 identifier 和 LoRA

`InputProcessor._get_mm_identifier()` 会在 tower connector LoRA 场景下把 LoRA 名字拼进 identifier：

```python
if lora_request is None or ...:
    return mm_hash
return f"{lora_request.lora_name}:{mm_hash}"
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:165` 到 `code/vllm/vllm/v1/engine/input_processor.py:181`

原因是：

```text
如果多模态 encoder / connector 会受 LoRA 影响，同一张图片在不同 LoRA 下的 encoder output 可能不同，不能复用同一个 encoder cache entry。
```

---

## 9. Request 和 SchedulerOutput 如何携带多模态状态

### 9.1 Request.mm_features

`Request` 初始化参数包含：

```python
mm_features: list[MultiModalFeatureSpec] | None = None
```

位置：`code/vllm/vllm/v1/request.py:70`

保存到请求状态：

```python
self.mm_features = mm_features or []
```

位置：`code/vllm/vllm/v1/request.py:156` 到 `code/vllm/vllm/v1/request.py:158`

这说明 `mm_features` 是请求生命周期状态的一部分。

### 9.2 NewRequestData.mm_features

Scheduler 发送新请求给 worker 时，会把 `mm_features` 放进 `NewRequestData`：

```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
```

位置：`code/vllm/vllm/v1/core/sched/output.py:32` 到 `code/vllm/vllm/v1/core/sched/output.py:43`

`from_request()` 直接从 request 搬运：

```python
mm_features=request.mm_features
```

位置：`code/vllm/vllm/v1/core/sched/output.py:55` 到 `code/vllm/vllm/v1/core/sched/output.py:67`

### 9.3 SchedulerOutput.scheduled_encoder_inputs

`SchedulerOutput` 中多模态最关键的字段是：

```python
scheduled_encoder_inputs: dict[str, list[int]]
```

位置：`code/vllm/vllm/v1/core/sched/output.py:203` 到 `code/vllm/vllm/v1/core/sched/output.py:206`

注释说明：

```text
req_id -> encoder input indices that need processing.
例如 request 有 [0, 1]，表示本轮要处理该请求的第 0 个和第 1 个 image。
```

还有一个释放字段：

```python
free_encoder_mm_hashes: list[str]
```

位置：`code/vllm/vllm/v1/core/sched/output.py:215` 到 `code/vllm/vllm/v1/core/sched/output.py:217`

它告诉 worker 哪些 encoder output 可以从物理 cache 中删除。

---

## 10. MultiModalBudget：encoder budget 和 cache size 从哪里来

定义位置：`code/vllm/vllm/multimodal/encoder_budget.py:44`

```python
class MultiModalBudget:
    """Helper class to calculate budget information for multi-modal models."""
```

### 10.1 计算每个 item 最大 token 数

入口：

```python
def get_mm_max_toks_per_item(...)
```

位置：`code/vllm/vllm/multimodal/encoder_budget.py:15` 到 `code/vllm/vllm/multimodal/encoder_budget.py:41`

它优先问 processor：

```python
max_tokens_per_item = processor.info.get_mm_max_tokens_per_item(...)
```

如果 processor 没提供，就构造 dummy multimodal input，再从 placeholder 计算：

```python
mm_inputs = mm_registry.get_dummy_mm_inputs(...)
return {
    modality: sum(item.get_num_embeds() for item in placeholders)
    for modality, placeholders in mm_inputs["mm_placeholders"].items()
}
```

位置：`code/vllm/vllm/multimodal/encoder_budget.py:25` 到 `code/vllm/vllm/multimodal/encoder_budget.py:40`

### 10.2 active modality

`MultiModalBudget` 会区分两类 modality：

```text
tower_modalities：limit > 0，需要走 multimodal encoder tower；
embed_only_modalities：enable_mm_embeds=True 且 limit == 0，只处理预计算 embeddings。
```

位置：`code/vllm/vllm/multimodal/encoder_budget.py:72` 到 `code/vllm/vllm/multimodal/encoder_budget.py:85`

### 10.3 encoder_compute_budget 和 encoder_cache_size

预算计算：

```python
encoder_compute_budget, encoder_cache_size = compute_mm_encoder_budget(
    scheduler_config,
    active_mm_max_toks_per_item,
)
```

位置：`code/vllm/vllm/multimodal/encoder_budget.py:115` 到 `code/vllm/vllm/multimodal/encoder_budget.py:123`

`compute_mm_encoder_budget()` 会取配置值和单 item 最大值的 max：

```python
encoder_compute_budget = max(
    scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
)
encoder_cache_size = max(
    scheduler_config.encoder_cache_size, max_tokens_per_mm_item
)
```

位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:269` 到 `code/vllm/vllm/v1/core/encoder_cache_manager.py:316`

这说明 vLLM 会保证至少能容纳一个最大多模态 item。

---

## 11. Scheduler：什么时候调度 multimodal encoder

Scheduler 初始化时会创建 encoder cache manager：

```python
supports_mm_inputs = mm_registry.supports_multimodal_inputs(...)
mm_budget = MultiModalBudget(...) if supports_mm_inputs else None
self.max_num_encoder_input_tokens = mm_budget.encoder_compute_budget if mm_budget else 0
encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:204` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:230`

核心判断在 `_try_schedule_encoder_inputs()`。

入口：`code/vllm/vllm/v1/core/sched/scheduler.py:1367`

```python
def _try_schedule_encoder_inputs(
    self,
    request,
    num_computed_tokens,
    num_new_tokens,
    encoder_compute_budget,
    shift_computed_tokens=0,
) -> tuple[list[int], int, int, list[int]]:
```

### 11.1 调度条件

源码注释直接列出条件：

```text
An encoder input will be scheduled if:
- Its output tokens overlap with the range of tokens being computed in this step.
- It is not already computed and stored in the encoder cache.
- It is not exist on remote encoder cache (via ECConnector).
- There is sufficient encoder token budget to process it.
- The encoder cache has space to store it.
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1379` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1387`

### 11.2 找到本轮 token window 覆盖的 mm_features

Scheduler 用 `get_mm_features_in_window()` 找到本轮涉及哪些多模态 item：

```python
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1409` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1413`

这说明多模态 encoder 调度是和“本轮要计算的 token 范围”绑定的。

### 11.3 encoder cache 命中则不调度

```python
if self.encoder_cache_manager.check_and_update_cache(request, i):
    continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1449` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1452`

如果 encoder output 已经缓存，只更新引用关系，不让 worker 重跑 encoder。

如果配置了 ECConnector，Scheduler 还会检查远端 encoder cache：

```python
if self.ec_connector is not None and self.ec_connector.has_cache_item(item_identifier):
    external_load_encoder_input.append(i)
    continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1507` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1513`

这种情况下本轮会分配逻辑账本并通过 EC connector 加载，而不是把该 item 放进本地 encoder 执行列表。

### 11.4 cache 或预算不够时会截断本轮 token

如果无法分配 encoder cache 或 compute budget：

```python
if not self.encoder_cache_manager.can_allocate(...):
    if num_computed_tokens + shift_computed_tokens < start_pos:
        num_new_tokens = start_pos - (...)
    else:
        num_new_tokens = 0
    break
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1470` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1489`

这点很重要：

```text
如果本轮 decoder token 会跨过一个尚未准备好的多模态 placeholder，但 encoder 预算或 cache 不够，Scheduler 会缩短本轮 num_new_tokens，避免 decoder forward 时缺 embedding。
```

### 11.5 写入 SchedulerOutput

调度成功后：

```python
scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
for i in encoder_inputs_to_schedule:
    self.encoder_cache_manager.allocate(request, i)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:648` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:656`

最后构造 `SchedulerOutput`：

```python
SchedulerOutput(
    ...
    scheduled_encoder_inputs=scheduled_encoder_inputs,
    free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
    ...
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1142` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1157`

---

## 12. EncoderCacheManager：管理 encoder output 的逻辑账本

定义位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`

```python
class EncoderCacheManager:
    """Manages caching of encoder outputs for multimodal models in vLLM V1."""
```

### 12.1 它管理什么

它的注释说明：

```text
- 管理 multimodal encoder output 生命周期；
- 以单个多模态 input item 为粒度；
- cache size 按 encoder embeddings 数量计；
- 通过 mm_hash / identifier 复用相同多模态数据的 encoder output；
- 没有 request 引用的旧 entry 可以在分配时被驱逐。
```

位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17` 到 `code/vllm/vllm/v1/core/encoder_cache_manager.py:65`

内部核心状态：

```python
self.cached: dict[str, set[str]] = {}
self.request_cached_ids: dict[str, set[int]] = {}
self.freeable: OrderedDict[str, int] = OrderedDict()
self.freed: list[str] = []
```

位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:72` 到 `code/vllm/vllm/v1/core/encoder_cache_manager.py:79`

### 12.2 它不存 tensor

`can_allocate()` 注释明确说：

```text
This method does not allocate physical memory for the encoder output but only the state of EncoderCacheManager.
```

位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:154` 到 `code/vllm/vllm/v1/core/encoder_cache_manager.py:157`

真正的 tensor 存储在 worker 侧 `GPUModelRunner.encoder_cache`。

所以边界是：

```text
EncoderCacheManager：Scheduler 侧逻辑账本；
GPUModelRunner.encoder_cache：Worker 侧物理 encoder output tensor cache。
```

### 12.3 check_and_update_cache()

```python
mm_hash = request.mm_features[input_id].identifier
if mm_hash not in self.cached:
    return False
...
self.cached[mm_hash].add(request.request_id)
self.request_cached_ids.setdefault(request.request_id, set()).add(input_id)
return True
```

位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:94` 到 `code/vllm/vllm/v1/core/encoder_cache_manager.py:121`

命中时只更新引用，不触发 worker encoder。

### 12.4 can_allocate() 和驱逐

如果空间不够，但 freeable 足够，会驱逐旧 entry：

```python
while num_embeds > self.num_free_slots:
    mm_hash, num_free_embeds = self.freeable.popitem(last=False)
    del self.cached[mm_hash]
    self.freed.append(mm_hash)
    self.num_free_slots += num_free_embeds
```

位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:174` 到 `code/vllm/vllm/v1/core/encoder_cache_manager.py:182`

被驱逐的 hash 会通过：

```python
get_freed_mm_hashes()
```

发给 worker。

位置：`code/vllm/vllm/v1/core/encoder_cache_manager.py:255` 到 `code/vllm/vllm/v1/core/encoder_cache_manager.py:266`

---

## 13. GPUModelRunner 初始化时的多模态状态

`GPUModelRunner.__init__()` 里保存多模态相关字段：

```python
self.mm_registry = MULTIMODAL_REGISTRY
self.uses_mrope = model_config.uses_mrope
self.uses_xdrope_dim = model_config.uses_xdrope_dim
self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(model_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:518` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:524`

还会创建 `MultiModalBudget`：

```python
self.mm_budget = (
    MultiModalBudget(self.vllm_config, self.mm_registry)
    if self.supports_mm_inputs
    else None
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:856` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:860`

对于输入 embeddings，会预分配 buffer：

```python
self.inputs_embeds = self._make_buffer(
    self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:786` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:791`

这说明多模态模型通常走 `inputs_embeds` 输入，而不是只传 `input_ids`。

---

## 14. _update_states()：worker 侧保存 mm_features

当 SchedulerOutput 进入 worker，`GPUModelRunner._update_states()` 会创建或更新 `CachedRequestState`。

新请求创建时：

```python
req_state = CachedRequestState(
    req_id=req_id,
    prompt_token_ids=new_req_data.prompt_token_ids,
    prompt_embeds=new_req_data.prompt_embeds,
    prompt_is_token_ids=new_req_data.prompt_is_token_ids,
    mm_features=new_req_data.mm_features,
    ...
)
self.requests[req_id] = req_state
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1265` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1279`

流式追加或恢复请求时也会更新：

```python
req_state.mm_features = new_req_data.mm_features
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1610` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1614`

同时，worker 会根据 scheduler 要求释放物理 encoder cache：

```python
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1199` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1201`

---

## 15. _batch_mm_inputs_from_scheduler()：从 SchedulerOutput 取本轮 encoder 输入

入口在：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2913`

```python
scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
if not scheduled_encoder_inputs:
    return [], [], []
```

它遍历：

```python
for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
    req_state = self.requests[req_id]
    for mm_input_id in encoder_input_ids:
        mm_feature = req_state.mm_features[mm_input_id]
        if mm_feature.data is None:
            continue
        mm_hashes.append(mm_feature.identifier)
        mm_kwargs.append((mm_feature.modality, mm_feature.data))
        mm_lora_refs.append((req_id, mm_feature.mm_position))
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2942` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2952`

输出三类信息：

```text
mm_hashes：每个 item 的 encoder cache key；
mm_kwargs：传给模型 embed_multimodal() 的 processor 输出；
mm_lora_refs：tower connector LoRA 需要的请求和 placeholder 位置信息。
```

---

## 16. _execute_mm_encoder()：真正执行多模态 encoder

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2956`

```python
def _execute_mm_encoder(self, scheduler_output) -> list[torch.Tensor]:
```

### 16.1 prompt_embeds 特殊路径

如果 modality 是 `prompt_embeds`：

```python
pe_tensor = mm_kwargs[i][1]["embedding"].data
self.encoder_cache[mm_hashes[i]] = pe_tensor.to(self.device)
self.maybe_save_ec_to_connector(self.encoder_cache, mm_hashes[i])
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2966` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2991`

这类输入已经在模型 embedding space 里，所以不跑 encoder，只直接放进 encoder cache，后续复用 `_gather_mm_embeddings()` 的统一路径。

### 16.2 按 modality 分组 batch

普通多模态输入会通过：

```python
for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(...):
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3080` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3082`

原因是不同 modality 或不同模型输入形态不能随便混在一个 encoder batch 里。

### 16.3 调用模型的 embed_multimodal()

真正执行 encoder：

```python
batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3147` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3150`

如果配置了 encoder CUDA graph，则可能走：

```python
cudagraph_output = self.encoder_cudagraph_manager.execute(mm_kwargs_batch)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3138` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3148`

输出检查：

```python
sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3152`

### 16.4 缓存 encoder output

执行完后：

```python
for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
    self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3157` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3161`

因此物理 tensor cache 在 worker 侧，key 是 `MultiModalFeatureSpec.identifier`。

---

## 17. _gather_mm_embeddings()：从 encoder_cache 按 placeholder 切片取 embedding

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3165`

```python
def _gather_mm_embeddings(
    self,
    scheduler_output,
    shift_computed_tokens=0,
) -> tuple[list[torch.Tensor], torch.Tensor]:
```

### 17.1 遍历当前 batch 的请求

它按 `input_batch.req_ids` 遍历当前 batch：

```python
for req_id in self.input_batch.req_ids:
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
    req_state = self.requests[req_id]
    num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3184` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3189`

然后找本轮 token window 中涉及的 mm_features：

```python
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_scheduled_tokens,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3191` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3196`

### 17.2 根据 mm_position 计算切片范围

每个 feature 都有 placeholder 范围：

```python
pos_info = mm_feature.mm_position
start_pos = pos_info.offset
num_encoder_tokens = pos_info.length
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3197` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3202`

如果本轮只覆盖该 placeholder 的一部分，会计算对应 embedding 区间：

```python
curr_embeds_start, curr_embeds_end = (
    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3209` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3211`

### 17.3 从 encoder_cache 获取 output

```python
mm_hash = mm_feature.identifier
encoder_output = self.encoder_cache.get(mm_hash, None)
if encoder_output is None:
    ...
    raise RuntimeError(f"Encoder cache miss for {mm_hash}.")
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3217` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3228`

这里如果 miss，通常说明 Scheduler 和 Worker 的 encoder cache 状态不一致，或者前面没正确执行 `_execute_mm_encoder()`；当前实现对 drafter 的前看位置会先跳过，其他情况抛 `RuntimeError`。

### 17.4 生成 is_mm_embed mask

`_gather_mm_embeddings()` 返回两个东西：

```text
mm_embeds：本轮需要插入的多模态 embeddings；
is_mm_embed：标记哪些 scheduled token 位置应替换成多模态 embedding。
```

相关代码：

```python
is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = True
mm_embeds_req.append(mm_embeds_item)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3236` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3246`

这一步是“placeholder token 转 embedding”的核心。

---

## 18. _preprocess()：把多模态 embedding 拼进模型输入

`_preprocess()` 是 ModelRunner 在 forward 前生成模型实参的地方。

多模态路径入口：

```python
if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
    with self.maybe_get_ec_connector_output(...):
        self._execute_mm_encoder(scheduler_output)
        mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3501` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3509`

### 18.1 为什么要在 _prepare_inputs 后 gather

源码注释：

```python
# _prepare_inputs may reorder the batch, so we must gather multi
# modal outputs after that to ensure the correct order
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3497` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3498`

这说明多模态 embedding 的拼接必须跟当前 batch row 顺序对齐，不能只按请求原始顺序处理。

### 18.2 调用 embed_input_ids()

多模态模型统一走 embeddings 输入。普通多模态路径会调用：

```python
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3538` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3542`

随后写入预分配 buffer：

```python
self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3544` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3547`

如果同一批里存在预计算 `prompt_embeds`，当前实现会先只 embed token-id 位置，再用 `torch.where()` 避免覆盖已有 prompt embeddings。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3513` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3536`

然后：

```python
input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
model_kwargs = {
    **self._init_model_kwargs(),
    **self._extract_mm_kwargs(scheduler_output),
}
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3549` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3553`

这里的关键点是：

```text
文本 token embedding 和多模态 embedding 的融合不是 Scheduler 做的，
也不是 OutputProcessor 做的，
而是在 ModelRunner._preprocess() 中通过模型的 embed_input_ids() 完成。
```

### 18.3 文本模型路径不同

如果不是多模态，也没有 prompt_embeds：

```python
input_ids = self.input_ids.gpu[:num_input_tokens]
inputs_embeds = None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3580` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3586`

所以：

```text
text-only：forward 通常吃 input_ids；
multimodal：forward 通常吃 inputs_embeds。
```

---

## 19. encoder-decoder 模型的特殊路径

对于 encoder-decoder 模型，`_preprocess()` 里有单独分支：

```python
if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
    encoder_outputs = self._execute_mm_encoder(scheduler_output)
    model_kwargs.update({"encoder_outputs": encoder_outputs})
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3605` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3612`

注释说明：

```text
For an encoder-decoder model, our processing here is a bit simpler, because the outputs are just passed to the decoder.
We are not doing any prompt replacement.
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3605` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3610`

也就是说：

```text
Decoder-only VLM：encoder output 要按 placeholder 拼进 inputs_embeds；
Encoder-decoder：encoder output 作为 encoder_outputs 传给 decoder，不做 prompt replacement。
```

Scheduler 侧也有特殊处理：

```python
if self.is_encoder_decoder:
    lo = 0
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1414` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1416`

并且 encoder-decoder 当前使用 `EncoderDecoderCacheManager`：

```python
self.encoder_cache_manager = (
    EncoderDecoderCacheManager(cache_size=encoder_cache_size)
    if self.is_encoder_decoder
    else EncoderCacheManager(cache_size=encoder_cache_size)
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:226` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:230`

---

## 20. processor cache 和 encoder cache 的区别

这是最容易混淆的点。

### 20.1 processor cache

位置：`code/vllm/vllm/multimodal/registry.py:294` 到 `code/vllm/vllm/multimodal/registry.py:347`

它缓存的是：

```text
Renderer / MultiModalProcessor 处理后的 mm_kwargs / processor outputs。
```

主要目的：

```text
避免重复 decode 图片、处理音频、调用 HF processor、生成 processor tensor。
```

控制配置：

```text
mm_processor_cache_gb
mm_processor_cache_type
mm_shm_cache_max_object_size_mb
mm_tensor_ipc
```

典型位置：

```text
API process / engine process / worker process 的 IPC 和共享缓存层。
```

### 20.2 encoder cache

逻辑管理在：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`

物理 tensor 存在：

```text
GPUModelRunner.encoder_cache
```

它缓存的是：

```text
model.embed_multimodal() 的输出，也就是 image/audio/video encoder embeddings。
```

主要目的：

```text
避免同一个多模态 item 在 chunked prefill、多个请求、prefix/cache 命中场景中重复跑 encoder。
```

控制字段：

```text
SchedulerOutput.scheduled_encoder_inputs
SchedulerOutput.free_encoder_mm_hashes
MultiModalFeatureSpec.identifier
EncoderCacheManager.cached/freeable/freed
```

### 20.3 二者关系

```text
processor cache 命中：可能不用重新传输 / 重新构造 mm_kwargs；
encoder cache 命中：不用重新执行 model.embed_multimodal()。
```

它们不是同一层缓存。

---

## 21. placeholder、mm_position、encoder token、embedding token

多模态输入进入 decoder-only VLM 时，prompt 中通常会出现 placeholder token 区间。

`MultiModalFeatureSpec.mm_position` 是 `PlaceholderRange`：

```text
offset：该 item 对应 placeholder 在 prompt token 序列中的起始位置；
length：placeholder token 区间长度；
get_num_embeds()：实际对应多少 encoder embeddings；
get_embeds_indices_in_range()：把 scheduled token 范围映射到 embedding 范围；
is_embed：部分位置是否是真正 embedding 的 mask。
```

在 Scheduler 中：

```python
start_pos = mm_feature.mm_position.offset
num_encoder_tokens = mm_feature.mm_position.length
num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1420` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1422`

在 ModelRunner 中：

```python
curr_embeds_start, curr_embeds_end = (
    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3209` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3211`

这说明：

```text
placeholder token 数量和真正 encoder embedding 数量不一定简单一一对应。
```

尤其在：

```text
M-RoPE / XD-RoPE；
视频 pruning；
use_audio_in_video；
prompt_embeds；
动态分辨率视觉模型；
```

这些场景里，必须通过 `PlaceholderRange` 做精确映射。

---

## 22. M-RoPE / XD-RoPE 和多模态位置

`GPUModelRunner` 初始化时记录：

```python
self.uses_mrope = model_config.uses_mrope
self.uses_xdrope_dim = model_config.uses_xdrope_dim
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:520` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:521`

在请求状态更新时，会根据 mm_features 计算位置。

M-RoPE 场景会过滤 `prompt_embeds`：

```python
mrope_features = [
    f for f in req_state.mm_features if f.modality != "prompt_embeds"
]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1636` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1641`

XD-RoPE 则会使用：

```python
req_state.xdrope_positions = xdrope_model.get_xdrope_input_positions(
    req_state.prompt_token_ids,
    req_state.mm_features,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1671` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1674`

这说明多模态不仅影响 embeddings，还可能影响 position ids。

---

## 23. 多模态链路和普通 generation 的完整例子

假设一个 decoder-only VLM 请求包含一张图片和一段文本。

完整链路：

```text
1. 用户传入：
   prompt + multi_modal_data={"image": ...}

2. Renderer._process_multimodal()
   → parse_mm_data()
   → mm_processor.apply()
   → 得到 prompt_token_ids / mm_kwargs / mm_placeholders / mm_hashes

3. InputProcessor.process_inputs()
   → 按 placeholder 位置排序
   → 构造 MultiModalFeatureSpec(data, modality, identifier, mm_position, mm_hash)
   → EngineCoreRequest.mm_features

4. Request
   → 保存 request.mm_features

5. Scheduler.schedule()
   → 根据本轮 token window 找到涉及的 mm_feature
   → 检查 EncoderCacheManager 是否命中
   → 检查 encoder_compute_budget 和 encoder cache 空间
   → SchedulerOutput.scheduled_encoder_inputs[req_id] = [0]

6. Executor / Worker
   → 把 SchedulerOutput 发给 GPUModelRunner

7. GPUModelRunner._update_states()
   → new_req_data.mm_features 保存到 CachedRequestState
   → 处理 free_encoder_mm_hashes

8. GPUModelRunner._preprocess()
   → _execute_mm_encoder()
      → model.embed_multimodal(**mm_kwargs)
      → encoder_cache[identifier] = image_embedding
   → _gather_mm_embeddings()
      → 根据 mm_position 从 encoder_cache 切片
      → 生成 mm_embeds 和 is_mm_embed
   → model.embed_input_ids(input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed)
   → 得到 inputs_embeds

9. GPUModelRunner._model_forward()
   → 用 inputs_embeds / positions / attention metadata 执行模型

10. generation 后续
   → compute_logits()
   → sample_tokens()
   → ModelRunnerOutput
```

这条链路里，多模态只改变 forward 输入准备，不改变采样和输出处理的基本框架。

---

## 24. prompt_embeds / embedding-only 模态的完整例子

如果用户传入的是预计算 embeddings：

```text
multi_modal_data 中的 item 已经是 tensor embeddings，或者 prompt 使用 prompt_embeds。
```

解析阶段：

```python
if self.is_embeddings(data):
    return ImageEmbeddingItems(data, self.expected_hidden_size)
```

位置：`code/vllm/vllm/multimodal/parse.py:613` 到 `code/vllm/vllm/multimodal/parse.py:614`

预算阶段：

```python
embed_only_modalities = {
    modality
    for modality in supported_mm_limits
    if enable_mm_embeds and mm_limits.get(modality, 0) == 0
}
```

位置：`code/vllm/vllm/multimodal/encoder_budget.py:79` 到 `code/vllm/vllm/multimodal/encoder_budget.py:85`

执行阶段：

```python
if modality == "prompt_embeds":
    self.encoder_cache[mm_hashes[i]] = pe_tensor.to(self.device)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2966` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2991`

也就是说：

```text
embedding-only 输入仍然走 MultiModalFeatureSpec / scheduled_encoder_inputs / encoder_cache / _gather_mm_embeddings 的统一框架，
但跳过真正的 multimodal encoder forward。
```

---

## 25. 和 Scheduler 的职责边界

### Scheduler 负责

```text
1. 保存 Request.mm_features；
2. 计算本轮 token window 是否覆盖某些 mm_features；
3. 判断 encoder cache 是否命中；
4. 判断 encoder compute budget 是否足够；
5. 判断 encoder cache 空间是否足够；
6. 必要时缩短本轮 num_new_tokens；
7. 在 SchedulerOutput.scheduled_encoder_inputs 中告诉 worker 本轮要跑哪些 encoder item；
8. 在 SchedulerOutput.free_encoder_mm_hashes 中告诉 worker 释放哪些 encoder output。
```

### Scheduler 不负责

```text
1. 不解析图片、音频、视频；
2. 不调用 HF processor；
3. 不执行 model.embed_multimodal()；
4. 不存储 encoder output tensor；
5. 不把 encoder output 拼进 inputs_embeds。
```

边界一句话：

```text
Scheduler 决定“哪些多模态 encoder output 本轮必须可用”，ModelRunner 决定“如何得到并使用这些 output”。
```

---

## 26. 和 ModelRunner 的职责边界

### ModelRunner 负责

```text
1. 接收 SchedulerOutput.scheduled_encoder_inputs；
2. 从 CachedRequestState.mm_features 找到 mm_kwargs；
3. 调用 model.embed_multimodal()；
4. 把 encoder output 存进 encoder_cache；
5. 根据 mm_position 从 encoder_cache 切片；
6. 生成 is_mm_embed mask；
7. 调用 model.embed_input_ids() 合并文本 token embedding 和多模态 embedding；
8. 生成 inputs_embeds；
9. 继续执行普通 forward / logits / sampling。
```

### ModelRunner 不负责

```text
1. 不决定哪些请求进入本轮 batch；
2. 不决定 encoder cache 逻辑驱逐策略；
3. 不处理用户原始图片和音频的解析；
4. 不构造 OpenAI 输出；
5. 不决定 max_num_encoder_input_tokens 或 encoder_cache_size。
```

边界一句话：

```text
ModelRunner 是多模态 encoder 执行和 embedding 拼接的落地点。
```

---

## 27. 和模型实现的职责边界

多模态模型类通常要提供模型特定能力：

```text
embed_multimodal(**mm_kwargs)
embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
get_num_mm_encoder_tokens(...)
get_num_mm_connector_tokens(...)
get_mm_mapping()
```

这些接口在不同模型里实现不同，例如 Qwen2-VL、LLaVA、Whisper、Voxtral、Pixtral 等。

vLLM 通用层不理解每个模型的视觉塔、音频塔、connector 细节。

通用层只约定：

```text
1. processor 输出 mm_kwargs；
2. model.embed_multimodal(**mm_kwargs) 返回每个 item 的 embeddings；
3. model.embed_input_ids() 能把 placeholder token 对应位置替换成 multimodal embeddings；
4. 后续 model.forward() 可以消费 input_ids 或 inputs_embeds。
```

因此：

```text
vLLM multimodal 通用框架负责路由、调度、缓存、拼接；
具体模型负责真正的 encoder 和 embedding 语义。
```

---

## 28. 和 Executor / Worker 的职责边界

Executor 不理解多模态细节。

它只负责：

```text
Executor.execute_model(scheduler_output)
  → Worker.execute_model(scheduler_output)
  → GPUModelRunner.execute_model(scheduler_output)
```

对于多模态，Executor 看到的仍然只是普通的 `SchedulerOutput`。

Worker 负责设备生命周期和持有 ModelRunner。

多模态真正落在：

```text
GPUModelRunner._update_states()
GPUModelRunner._execute_mm_encoder()
GPUModelRunner._gather_mm_embeddings()
GPUModelRunner._preprocess()
```

所以：

```text
Executor：分发 SchedulerOutput；
Worker：管理 device / lifecycle；
ModelRunner：处理多模态 encoder 和 embedding 拼接。
```

---

## 29. 和 decoder KV cache 的区别

多模态 encoder cache 和 decoder KV cache 不是一回事。

### decoder KV cache

```text
对象：decoder self-attention 的 key/value；
管理者：KVCacheManager / Scheduler / ModelRunner KV cache tensors；
粒度：block；
作用：加速 autoregressive decode；
进入 attention backend。
```

### multimodal encoder cache

```text
对象：image/audio/video encoder output embeddings；
逻辑管理者：EncoderCacheManager；
物理存储：GPUModelRunner.encoder_cache；
粒度：单个 multimodal item；
作用：避免重复执行 multimodal encoder，并在 prompt placeholder 位置拼接 inputs_embeds。
```

二者都叫 cache，但处在不同层。

关键区别：

```text
KV cache 是 decoder 历史 token 的 attention 状态；
encoder cache 是非文本输入经过 encoder 后的 embedding 结果。
```

---

## 30. 多模态链路和 chunked prefill 的关系

多模态输入常常很长，尤其是 image/video placeholder tokens。

Scheduler 的 `_try_schedule_encoder_inputs()` 会根据本轮 token window 判断是否触达某个多模态 item：

```python
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1409` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1413`

如果不允许 chunked multimodal input，且本轮只覆盖了 item 的一部分，会回退：

```python
if disable_chunked_mm_input and ...:
    num_new_tokens = max(0, start_pos - (...))
    break
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1454` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1469`

如果预算或 cache 不足，也可能截断：

```python
num_new_tokens = start_pos - (...)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1470` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1489`

因此 chunked prefill 下，多模态 encoder 调度和 decoder token 调度是耦合的。

---

## 31. 容易混淆的点

### 31.1 Multimodal 是不是一个模型执行器？

不是。

```text
模型执行器是 Worker / ModelRunner；
Multimodal 是输入处理、encoder 辅助、embedding 拼接和缓存机制。
```

### 31.2 MultiModalFeatureSpec.data 是不是一定有 tensor？

不是。

```text
data 可以为 None，表示 processor 输出已经缓存或不需要通过 IPC 重新发送。
```

位置：`code/vllm/vllm/multimodal/inputs.py:311` 到 `code/vllm/vllm/multimodal/inputs.py:317`

### 31.3 mm_hash 和 identifier 是不是同一个东西？

不总是。

```text
mm_hash：processor cache key，通常不带 LoRA 前缀；
identifier：encoder cache key，tower connector LoRA 场景下可能带 LoRA 前缀。
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:165` 到 `code/vllm/vllm/v1/engine/input_processor.py:181`

### 31.4 scheduled_encoder_inputs 里存的是 hash 吗？

不是。

```text
scheduled_encoder_inputs: req_id -> list[input_id]
```

这里的 input_id 是 `Request.mm_features` 中的索引。

位置：`code/vllm/vllm/v1/core/sched/output.py:203` 到 `code/vllm/vllm/v1/core/sched/output.py:206`

### 31.5 encoder cache hit 后还会执行 _execute_mm_encoder() 吗？

通常不会对该 item 执行。

Scheduler 命中 `EncoderCacheManager.check_and_update_cache()` 后不会把该 item 加进 `scheduled_encoder_inputs`。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1449` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1452`

### 31.6 多模态输入最终会改变输出对象类型吗？

通常不会。

对于 generation：

```text
多模态输入 → inputs_embeds → forward → logits → sampling → token output
```

最终仍然走 `ModelRunnerOutput`、`EngineCoreOutputs`、`RequestOutput`。

### 31.7 prompt_embeds 是否绕过所有多模态机制？

不是。

在多模态 embedding-only 场景，它会跳过 encoder，但仍可能进入：

```text
MultiModalFeatureSpec
scheduled_encoder_inputs
encoder_cache
_gather_mm_embeddings
embed_input_ids
```

---

## 32. 从“回答问题”的角度总结

如果要问：

```text
Multimodal 在 vLLM V1 里负责什么？
```

可以回答：

```text
Multimodal 是 vLLM V1 中把非文本输入接入语言模型执行链路的系统。

它在请求进入时通过 Renderer 和 MultiModalProcessor 把 image / audio / video / embeddings 处理成 prompt_token_ids、mm_kwargs、placeholder 位置和 hash；
InputProcessor 再把这些信息整理成 MultiModalFeatureSpec，放入 EngineCoreRequest 和 Request。

Scheduler 根据本轮 token window、encoder compute budget、encoder cache 状态决定哪些 multimodal item 需要在本轮执行 encoder，并通过 SchedulerOutput.scheduled_encoder_inputs 通知 worker。

GPUModelRunner 收到 SchedulerOutput 后，调用模型的 embed_multimodal() 执行多模态 encoder，把结果存入 encoder_cache，再根据 mm_position 从 cache 中取出对应 embeddings，调用 embed_input_ids() 把文本 token embedding 和多模态 embedding 合并为 inputs_embeds，最后进入普通 forward / logits / sampling 链路。
```

职责关系可以概括为：

```text
Renderer / Processor：解析和预处理原始多模态输入；
InputProcessor：构造 MultiModalFeatureSpec；
Scheduler：决定哪些 encoder input 本轮需要可用；
EncoderCacheManager：管理 encoder output 的逻辑缓存账本；
GPUModelRunner：执行 encoder、缓存 output、拼接 inputs_embeds；
模型实现：提供具体 embed_multimodal() / embed_input_ids() 语义；
Sampler / OutputProcessor：照常处理 logits、采样和最终输出。
```

---

## 33. 最关键流程图

```text
用户输入
  ├─ prompt / prompt_token_ids
  └─ multi_modal_data
       │
       ▼
Renderer._process_multimodal()
  ├─ MultiModalDataParser.parse_mm_data()
  ├─ mm_processor.apply()
  ├─ mm_kwargs
  ├─ mm_placeholders
  └─ mm_hashes
       │
       ▼
InputProcessor.process_inputs()
  ├─ argsort_mm_positions()
  └─ MultiModalFeatureSpec
       ├─ data
       ├─ modality
       ├─ identifier
       ├─ mm_position
       └─ mm_hash
       │
       ▼
EngineCoreRequest.mm_features
       │
       ▼
Request.mm_features
       │
       ▼
Scheduler.schedule()
  ├─ _try_schedule_encoder_inputs()
  │    ├─ get_mm_features_in_window()
  │    ├─ EncoderCacheManager.check_and_update_cache()
  │    ├─ EncoderCacheManager.can_allocate()
  │    └─ EncoderCacheManager.allocate()
  │
  └─ SchedulerOutput
       ├─ scheduled_encoder_inputs
       └─ free_encoder_mm_hashes
       │
       ▼
Executor / Worker
       │
       ▼
GPUModelRunner.execute_model()
  ├─ _update_states()
  │    ├─ 保存 req_state.mm_features
  │    └─ 删除 free_encoder_mm_hashes
  │
  ├─ _prepare_inputs()
  ├─ _build_attention_metadata()
  │
  ├─ _preprocess()
  │    ├─ _execute_mm_encoder()
  │    │    ├─ _batch_mm_inputs_from_scheduler()
  │    │    ├─ model.embed_multimodal(**mm_kwargs)
  │    │    └─ encoder_cache[identifier] = output
  │    │
  │    ├─ _gather_mm_embeddings()
  │    │    ├─ encoder_cache[identifier]
  │    │    ├─ mm_position 切片
  │    │    └─ is_mm_embed mask
  │    │
  │    ├─ model.embed_input_ids(
  │    │    input_ids,
  │    │    multimodal_embeddings=mm_embeds,
  │    │    is_multimodal=is_mm_embed,
  │    │  )
  │    │
  │    └─ inputs_embeds
  │
  ├─ _model_forward()
  ├─ compute_logits()
  └─ sample_tokens()
       │
       ▼
ModelRunnerOutput
```

---

## 34. 最关键对象关系

```text
MultiModalConfig
  多模态行为配置，包括 limit、processor cache、encoder TP、encoder attention、embedding-only 等。

MultiModalRegistry
  模型类到 processor / processing info / dummy inputs 的注册中心。

MultiModalDataParser
  把原始 image / audio / video / embeddings 解析成内部 item。

MultiModalProcessor
  模型特定 processor，输出 prompt_token_ids、mm_kwargs、placeholder、hash。

MultiModalFeatureSpec
  单个多模态 item 在 V1 链路中的核心描述对象。

PlaceholderRange / mm_position
  描述该多模态 item 在 prompt token 序列中的 placeholder 区间，以及 token 到 embedding 的映射。

EncoderCacheManager
  Scheduler 侧 encoder output 逻辑缓存账本。

SchedulerOutput.scheduled_encoder_inputs
  本轮需要 worker 执行的 encoder input 索引。

GPUModelRunner.encoder_cache
  Worker 侧 encoder output tensor cache。

model.embed_multimodal()
  模型特定多模态 encoder 执行入口。

model.embed_input_ids()
  把文本 token embedding 和多模态 embedding 合并成 inputs_embeds 的入口。
```

---

## 35. 最小心智模型

如果只记一条主线，可以记：

```text
多模态输入先被 processor 变成 mm_kwargs + placeholder + hash，
再被 InputProcessor 包成 MultiModalFeatureSpec，
Scheduler 按 token window 和 encoder cache 决定要不要跑 encoder，
ModelRunner 执行 embed_multimodal() 并缓存结果，
最后把 encoder output 按 placeholder 位置拼进 inputs_embeds，进入普通 decoder forward。
```

再压缩成一句话：

```text
Multimodal = 非文本输入处理 + encoder 调度 + encoder output 缓存 + inputs_embeds 拼接。
```
