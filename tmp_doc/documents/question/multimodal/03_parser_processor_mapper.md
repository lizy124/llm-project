# 03. 多模态 parser 和 processor 如何生成 feature？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\parse.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\inputs.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\processing\processor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\processing\inputs.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\registry.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\hasher.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\cache.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\input_processor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\worker_base.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\interfaces.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\qwen2_5_vl.py`

本问题关注：原始 image / audio / video 等多模态数据如何被解析、校验、送入模型专用 processor，如何生成模型 forward 需要的 kwargs / feature spec / placeholder ranges / hash，以及这些结果如何进入 EngineCore、Scheduler、Worker 和 ModelRunner。

---

## 0. 梳理规划

本篇按“先定对象边界，再走主链路，再拆关键阶段，最后给例子和流程图”的方式梳理。

要回答的问题分成 10 组：

```text
1. parser、processor、mapper、feature spec 分别是什么？
2. MultiModalDataParser 解析 raw image / audio / video 时做了什么？
3. embedding 输入为什么可以绕过 HF processor？
4. BaseMultiModalProcessor.apply() 的主流程是什么？
5. HF processor output 如何被拆成每个 item 的 kwargs？
6. prompt placeholder 如何被替换并映射到 PlaceholderRange？
7. mm hash / uuid / processor cache 在哪里接入？
8. MultiModalFeatureSpec 在哪里生成，它包含什么？
9. mapper output 是 tensor、kwargs 还是 feature？
10. Worker / ModelRunner 最终如何消费这些 feature？
```

阅读顺序建议：

```text
01_multimodal_overview.md
  → 02_input_and_placeholder.md
  → 03_parser_processor_mapper.md
  → 04_multimodal_scheduler_encoder_cache.md
  → 05_worker_model_runner_multimodal.md
```

如果只想记一条主线，可以先看第 1、2、7、12、18 节。

---

## 1. 一句话回答

vLLM V1 的多模态输入不是由 parser 一步直接变成模型 feature。

真实链路是：

```text
raw multi_modal_data
  → MultiModalDataParser.parse_mm_data()
  → MultiModalDataItems
  → model-specific BaseMultiModalProcessor.apply()
  → HF processor / passthrough embeddings
  → MultiModalKwargsItems + mm_hashes + PromptUpdate
  → prompt_token_ids + mm_placeholders
  → MultiModalInput
  → v1 InputProcessor flatten/sort
  → list[MultiModalFeatureSpec]
  → EngineCoreRequest.mm_features
  → SchedulerOutput.scheduled_new_reqs[*].mm_features
  → Worker / ModelRunner / multimodal encoder
```

最关键的边界是：

```text
MultiModalDataParser：把用户传入的数据规范化成按 modality 分组的 items。
BaseMultiModalProcessor：调用 HF processor，生成 prompt_token_ids、mm kwargs、hash、placeholder ranges。
MultiModalKwargsItems：按多模态 item 拆开的模型 kwargs。
MultiModalFeatureSpec：请求侧每个多模态 item 的执行规格。
ModelRunner / model.embed_multimodal：最终用 kwargs 运行多模态 encoder，得到 embeddings。
```

一句话压缩：

```text
Parser 负责“认出和规整输入”，Processor 负责“模型专用预处理和占位符对齐”，FeatureSpec 负责“把每个多模态 item 带进调度和执行”。
```

---

## 2. 先纠正一个容易误解的点

原始占位稿里写的是：

```text
MultiModalDataParser
  → processor
  → mapper
  → MultiModalFeatureSpec
```

这个方向大体对，但需要更精确：

```text
MultiModalDataParser 不直接生成 MultiModalFeatureSpec。
BaseMultiModalProcessor.apply() 也不直接返回 MultiModalFeatureSpec。
MultiModalFeatureSpec 是 V1 engine input_processor 在拿到 MultiModalInput 后生成的。
```

也就是说，`MultiModalFeatureSpec` 的生成点在 V1 请求预处理层：

```python
mm_features.append(
    MultiModalFeatureSpec(
        data=decoder_mm_inputs[modality][idx],
        modality=modality,
        identifier=self._get_mm_identifier(base_mm_hash, lora_request),
        mm_position=decoder_mm_positions[modality][idx],
        mm_hash=base_mm_hash,
    )
)
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:354` 到 `code/vllm/vllm/v1/engine/input_processor.py:368`

因此更准确的主链路是：

```text
parse raw data
  → process prompt + media
  → produce MultiModalInput
  → flatten MultiModalInput into MultiModalFeatureSpec
```

---

## 3. 核心对象关系

```text
MultiModalDataDict
  用户传入的原始多模态字典，例如 {"image": image}、{"audio": audio}、{"video": video}。

MultiModalDataItems
  parser 输出的规范化容器，内部每个 modality 对应一个 ModalityDataItems。

ModalityDataItems
  单个 modality 的 item 集合，提供 get_processor_data() 和 get_passthrough_data()。

ProcessorInputs
  BaseMultiModalProcessor.apply() 的输入封装，包含 prompt、mm_data_items、mm_uuid_items、processor kwargs。

MultiModalKwargsItems
  HF processor 或 passthrough 输入被拆成“每个多模态 item 一份 kwargs”。

PromptUpdate
  模型专用的 prompt 占位符插入 / 替换规则。

PlaceholderRange
  某个多模态 item 在 token prompt 中对应的占位符区间。

MultiModalInput
  processor 返回给上层的结构，包含 prompt_token_ids、mm_kwargs、mm_hashes、mm_placeholders。

MultiModalFeatureSpec
  V1 request 中每个多模态 item 的规格，包含 data、modality、identifier、mm_position、mm_hash。
```

对象之间的层次是：

```text
MultiModalDataItems
  → MultiModalKwargsItems
  → MultiModalInput
  → list[MultiModalFeatureSpec]
```

---

## 4. MultiModalDataParser 负责什么

`MultiModalDataParser` 定义在 `parse.py`：

```python
class MultiModalDataParser:
    """
    Parses MultiModalDataDict into MultiModalDataItems.
    """
```

位置：`code/vllm/vllm/multimodal/parse.py:489` 到 `code/vllm/vllm/multimodal/parse.py:504`

它初始化时保存几类解析参数：

```python
self.audio_resampler = AudioResampler(target_sr=target_sr, method=audio_resample_method)
self.target_channels = target_channels
self.video_needs_metadata = video_needs_metadata
self.expected_hidden_size = expected_hidden_size
```

位置：`code/vllm/vllm/multimodal/parse.py:506` 到 `code/vllm/vllm/multimodal/parse.py:523`

这些参数影响：

```text
audio：是否重采样，是否归一化 channel 数；
video：是否要求 metadata；
embeddings：是否校验 hidden_size；
所有 modality：是否把单个输入规整成 list-of-items。
```

主入口是：

```python
def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
    subparsers = self._get_subparsers()

    mm_items = MultiModalDataItems()
    for k, v in mm_data.items():
        if k not in subparsers:
            raise ValueError(f"Unsupported modality: {k}")

        if (parsed_data := subparsers[k](v)) is not None:
            mm_items[k] = parsed_data

    return mm_items
```

位置：`code/vllm/vllm/multimodal/parse.py:692` 到 `code/vllm/vllm/multimodal/parse.py:704`

支持的 modality 来自：

```python
return {
    "audio": self._parse_audio_data,
    "image": self._parse_image_data,
    "video": self._parse_video_data,
    "vision_chunk": self._parse_vision_chunk_data,
}
```

位置：`code/vllm/vllm/multimodal/parse.py:684` 到 `code/vllm/vllm/multimodal/parse.py:690`

因此 parser 的职责不是模型预处理，而是：

```text
校验 modality 是否支持；
把单个 item / batch / tensor / tuple 等输入统一成 items；
识别 embedding 输入；
为 audio 做 resample / channel normalize；
为 video 检查 metadata；
为后续 HF processor 和 passthrough 分流提供统一接口。
```

---

## 5. ModalityDataItems 的两个出口

所有 modality items 都继承 `ModalityDataItems`，它定义了两个关键抽象方法：

```python
@abstractmethod
def get_processor_data(self) -> Mapping[str, object]:
    """Get the data to pass to the HF processor."""

@abstractmethod
def get_passthrough_data(self) -> Mapping[str, object]:
    """Get the data to pass directly to the model."""
```

位置：`code/vllm/vllm/multimodal/parse.py:93` 到 `code/vllm/vllm/multimodal/parse.py:101`

这两个出口决定了后续是否走 HF processor。

普通 image/audio/video 输入走 `ProcessorBatchItems`：

```python
def get_processor_data(self) -> Mapping[str, object]:
    return {f"{self.modality}s": self.get_all()}

def get_passthrough_data(self) -> Mapping[str, object]:
    return {}
```

位置：`code/vllm/vllm/multimodal/parse.py:121` 到 `code/vllm/vllm/multimodal/parse.py:125`

embedding 输入走 `EmbeddingItems`：

```python
def get_processor_data(self) -> Mapping[str, object]:
    return {}

def get_passthrough_data(self) -> Mapping[str, object]:
    return {f"{self.modality}_embeds": self.data}
```

位置：`code/vllm/vllm/multimodal/parse.py:229` 到 `code/vllm/vllm/multimodal/parse.py:233`

这就是预计算 embedding 可以绕过 HF processor 的原因。

可以这样理解：

```text
raw image/audio/video：进入 HF processor；
image_embeds/audio_embeds/video_embeds：作为 passthrough kwargs 直接传给模型侧逻辑；
processor_data + passthrough_data：最后会合并成 processed_data。
```

---

## 6. image / audio / video 是否走相同 parser？

不完全相同。

它们共享 `MultiModalDataParser.parse_mm_data()` 的框架，但每个 modality 有自己的 subparser。

### 6.1 image parser

image 解析入口：

```python
def _parse_image_data(self, data: ModalityData[ImageItem]):
```

位置：`code/vllm/vllm/multimodal/parse.py:605`

逻辑是：

```text
None：忽略；
3D embedding / list[2D tensor]：ImageEmbeddingItems；
PIL image / MediaWithBytes / 3D ndarray / 3D tensor：单张 image；
更高维 ndarray / tensor：按第一维拆成多张 image；
其他序列：按用户提供的 list 处理。
```

关键代码：

```python
if self.is_embeddings(data):
    return ImageEmbeddingItems(data, self.expected_hidden_size)

if isinstance(data, (PILImage.Image, MediaWithBytes)) or (
    isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 3
):
    data_items = [data]
elif isinstance(data, (np.ndarray, torch.Tensor)):
    data_items = [elem for elem in data]
else:
    data_items = data

return ImageProcessorItems(data_items)
```

位置：`code/vllm/vllm/multimodal/parse.py:612` 到 `code/vllm/vllm/multimodal/parse.py:624`

### 6.2 audio parser

audio 解析入口：

```python
def _parse_audio_data(self, data: ModalityData[AudioItem]):
```

位置：`code/vllm/vllm/multimodal/parse.py:566`

额外处理包括：

```text
(audio, sampling_rate) tuple：按 orig_sr 重采样；
list[float] / 1D ndarray / 1D tensor：视为单条 audio；
更高维 ndarray / tensor：按第一维拆成多条 audio；
target_channels 非空：normalize_audio 到目标 channel 数。
```

关键代码：

```python
if orig_sr is None:
    new_audio = audio
else:
    new_audio = self.audio_resampler.resample(audio, orig_sr=orig_sr)

if self.target_channels is not None:
    spec = AudioSpec(target_channels=self.target_channels)
    new_audio = normalize_audio(new_audio, spec)
```

位置：`code/vllm/vllm/multimodal/parse.py:590` 到 `code/vllm/vllm/multimodal/parse.py:599`

### 6.3 video parser

video 解析入口：

```python
def _parse_video_data(self, data: ModalityData[VideoItem]):
```

位置：`code/vllm/vllm/multimodal/parse.py:626`

额外处理包括：

```text
4D ndarray / tensor：视为单个 video；
更高维 ndarray / tensor：按第一维拆成多个 video；
(video, metadata) tuple：保留 metadata；
video_needs_metadata=True 时没有 metadata 会报错。
```

关键代码：

```python
if self.video_needs_metadata:
    if metadata is None:
        raise ValueError(
            "Video metadata is required but not found in mm input. "
            "Please check your video input in `multi_modal_data`"
        )
    new_videos.append((video, metadata))
    metadata_lst.append(metadata)
else:
    new_videos.append(video)
```

位置：`code/vllm/vllm/multimodal/parse.py:652` 到 `code/vllm/vllm/multimodal/parse.py:661`

所以答案是：

```text
image / audio / video 共用 parser 框架，但 subparser 不同。
image 重点处理单图 / 多图 / embedding；
audio 额外处理 sampling rate 和 channel；
video 额外处理 frame batch 和 metadata。
```

---

## 7. BaseMultiModalProcessor.apply() 主流程

processor 的抽象基类是：

```python
class BaseMultiModalProcessor(ABC, Generic[_I]):
    """
    Abstract base class to process multi-modal inputs to be used in vLLM.

    Not to be confused with transformers.ProcessorMixin.
    """
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:972` 到 `code/vllm/vllm/multimodal/processing/processor.py:977`

注意这句话：

```text
BaseMultiModalProcessor 不是 Hugging Face ProcessorMixin，
而是 vLLM 自己的模型专用多模态处理器。
```

初始化时它保存：

```python
self.info = info
self.dummy_inputs = dummy_inputs
self.cache = cache
self.data_parser = self.info.get_data_parser()
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:979` 到 `code/vllm/vllm/multimodal/processing/processor.py:992`

主入口 `apply()` 的注释已经给出三步：

```python
"""
The main steps are:

1. Apply HF Processor on prompt text and multi-modal data together,
   outputting token IDs and processed tensors.
2. Find and update sequences in the token IDs with placeholder tokens.
   The number of placeholder tokens equals the feature size of the
   multi-modal data outputted by the multi-modal encoder.
3. Extract information about the placeholder tokens from the
   processed token IDs.
"""
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:1663` 到 `code/vllm/vllm/multimodal/processing/processor.py:1680`

实际代码是：

```python
(
    prompt_ids,
    mm_info,
    is_update_applied,
) = self._cached_apply_hf_processor(inputs, timing_ctx)

with timing_ctx.record("apply_prompt_updates"):
    prompt_ids, mm_placeholders = self._maybe_apply_prompt_updates(
        mm_items=inputs.mm_data_items,
        prompt_ids=prompt_ids,
        mm_kwargs=mm_info.kwargs,
        mm_prompt_updates=mm_info.prompt_updates,
        is_update_applied=is_update_applied,
    )

mm_placeholder_ranges = {
    modality: [item.to_range() for item in placeholders]
    for modality, placeholders in mm_placeholders.items()
}

return mm_input(
    prompt_token_ids=prompt_ids,
    mm_kwargs=mm_info.kwargs,
    mm_hashes=mm_info.hashes,
    mm_placeholders=mm_placeholder_ranges,
)
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:1681` 到 `code/vllm/vllm/multimodal/processing/processor.py:1707`

所以 processor 的输出不是裸 tensor，而是一个完整的 `MultiModalInput`：

```text
prompt_token_ids：已经和多模态 placeholder 对齐后的 token ids；
mm_kwargs：每个多模态 item 的模型输入 kwargs；
mm_hashes：每个 item 的 processor cache hash；
mm_placeholders：每个 item 在 prompt 中的 PlaceholderRange。
```

---

## 8. HF processor 是怎么被调用的

processor 先从 `MultiModalDataItems` 中拆出两类数据：

```python
def _get_hf_mm_data(self, mm_items: MultiModalDataItems):
    processor_data = dict[str, object]()
    passthrough_data = dict[str, object]()

    for items in mm_items.values():
        processor_data.update(items.get_processor_data())
        passthrough_data.update(items.get_passthrough_data())

    return processor_data, passthrough_data
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:1083` 到 `code/vllm/vllm/multimodal/processing/processor.py:1095`

调用 HF processor 的基础函数是：

```python
return self.info.ctx.call_hf_processor(
    self.info.get_hf_processor(**mm_kwargs),
    dict(text=prompt, **mm_data),
    dict(**mm_kwargs, **tok_kwargs),
)
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:1097` 到 `code/vllm/vllm/multimodal/processing/processor.py:1114`

文本和多模态一起处理时：

```python
processed_data = self._call_hf_processor(
    prompt=prompt_text,
    mm_data=processor_data,
    mm_kwargs=hf_processor_mm_kwargs,
    tok_kwargs=tokenization_kwargs,
)
processed_data.update(passthrough_data)

input_ids = processed_data.pop("input_ids")
(prompt_ids,) = input_ids
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:1153` 到 `code/vllm/vllm/multimodal/processing/processor.py:1165`

这里有两个关键点：

```text
1. HF processor 返回的是 batched BatchFeature，例如 input_ids、pixel_values、image_grid_thw。
2. passthrough_data 会被 update 到 processed_data，因此 embedding 输入也能进入后续 kwargs 拆分。
```

---

## 9. HF processor output 如何变成每个 item 的 kwargs

HF processor 常返回一个大 batch，例如：

```text
pixel_values: tensor for all images
image_grid_thw: tensor for all images
```

vLLM 需要把它拆成每个多模态 item 一份 `MultiModalKwargsItem`。

抽象配置是 `MultiModalFieldConfig`：

```python
@dataclass(frozen=True)
class MultiModalFieldConfig:
    field: BaseMultiModalField
    modality: str
```

位置：`code/vllm/vllm/multimodal/inputs.py:633` 到 `code/vllm/vllm/multimodal/inputs.py:844`

常见 field 类型有三种：

```text
MultiModalFieldConfig.batched：按 batch 第一维拆，每个 item 一个 elem；
MultiModalFieldConfig.flat：按 slice 从一个扁平 tensor 中切出每个 item；
MultiModalFieldConfig.shared：多个 item 共享同一个字段。
```

`MultiModalKwargsItems.from_hf_inputs()` 用 field config 执行拆分：

```python
for key, config in config_by_key.items():
    batch = hf_inputs.get(key)
    if batch is not None:
        elems = config.build_elems(key, batch)
        if len(elems) > 0:
            elems_by_key[key] = elems
            keys_by_modality[config.modality].add(key)
```

位置：`code/vllm/vllm/multimodal/inputs.py:919` 到 `code/vllm/vllm/multimodal/inputs.py:933`

然后按 modality 和 item index 组装：

```python
items_by_modality[modality] = [
    MultiModalKwargsItem({k: v[i] for k, v in elems_in_modality.items()})
    for i in range(batch_size)
]
```

位置：`code/vllm/vllm/multimodal/inputs.py:935` 到 `code/vllm/vllm/multimodal/inputs.py:952`

因此：

```text
HF processor output：batch 级 BatchFeature；
MultiModalFieldConfig：描述每个字段如何拆；
MultiModalKwargsItems：按 modality / item 拆好的模型 kwargs。
```

---

## 10. mapper output 是 tensor 还是 model kwargs？

如果把“mapper”理解成“把 processor output 映射成模型输入”的逻辑，那么在当前 vLLM 代码里它不是一个统一名为 Mapper 的类。

它分散在三处：

```text
1. 模型专用 BaseMultiModalProcessor._get_mm_fields_config()
   决定 HF output 的每个字段属于哪个 modality、怎么拆 item。

2. MultiModalKwargsItems.from_hf_inputs()
   按 field config 把 BatchFeature 拆成 MultiModalKwargsItem。

3. group_and_batch_mm_kwargs() / model.embed_multimodal()
   在执行前把多个 item 的 kwargs 重新合批，传给模型多模态 encoder。
```

所以 mapper 的输出不是单个 tensor，而是：

```text
单 item 层面：MultiModalKwargsItem
  例如 {"pixel_values": elem, "image_grid_thw": elem}

request 层面：MultiModalFeatureSpec.data
  即某个 item 的 MultiModalKwargsItem，加上 modality/hash/position。

执行层面：BatchedTensorInputs
  多个 item 合批后的 tensor kwargs，传给 embed_multimodal(**kwargs)。
```

从对象形态看：

```text
BatchFeature
  → MultiModalKwargsItems
  → MultiModalFeatureSpec.data
  → group_and_batch_mm_kwargs()
  → BatchedTensorInputs
  → model.embed_multimodal(**kwargs)
```

---

## 11. prompt placeholder 如何对齐 feature

多模态模型通常要求 prompt 中有 `<image>`、`<video>`、`<audio>` 之类占位符。

vLLM 用 `PromptUpdate` 描述如何插入或替换这些占位符。

抽象类是：

```python
@dataclass
class PromptUpdate(ABC):
    modality: str
    target: PromptUpdateTarget
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:297` 到 `code/vllm/vllm/multimodal/processing/processor.py:319`

两种模式：

```text
PromptInsertion：在 target 后插入 placeholder tokens；
PromptReplacement：把 target 替换成 placeholder tokens。
```

位置：

```text
code/vllm/vllm/multimodal/processing/processor.py:353 到 code/vllm/vllm/multimodal/processing/processor.py:420
code/vllm/vllm/multimodal/processing/processor.py:422 到 code/vllm/vllm/multimodal/processing/processor.py:497
```

processor 会根据模型专用 `_get_prompt_updates()` 得到规则：

```python
mm_prompt_updates = self._get_mm_prompt_updates(
    inputs.mm_data_items,
    inputs.hf_processor_mm_kwargs,
    mm_kwargs,
)
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:1427` 到 `code/vllm/vllm/multimodal/processing/processor.py:1431`

然后 `_maybe_apply_prompt_updates()` 处理两种情况：

```text
如果 HF processor 已经完成 prompt update：只查找 placeholder；
如果 HF processor 没有完成：vLLM 自己应用 PromptUpdate，再查找 placeholder。
```

代码：

```python
if is_update_applied:
    mm_placeholders = self._find_mm_placeholders(prompt_ids, mm_prompt_updates)
    self._validate_mm_placeholders(mm_placeholders, mm_item_counts)
else:
    prompt_ids, mm_placeholders = self._apply_prompt_updates(
        prompt_ids,
        mm_prompt_updates,
    )
    self._validate_mm_placeholders(mm_placeholders, mm_item_counts)
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:1648` 到 `code/vllm/vllm/multimodal/processing/processor.py:1659`

最终每个 placeholder 被转换成 `PlaceholderRange`：

```python
def to_range(self) -> PlaceholderRange:
    return PlaceholderRange(
        offset=self.start_idx,
        length=self.length,
        is_embed=self.is_embed,
    )
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:675` 到 `code/vllm/vllm/multimodal/processing/processor.py:694`

`PlaceholderRange` 本身定义为：

```python
@dataclass(frozen=True)
class PlaceholderRange:
    offset: int
    length: int
    is_embed: torch.Tensor | None = None
```

位置：`code/vllm/vllm/multimodal/inputs.py:118` 到 `code/vllm/vllm/multimodal/inputs.py:145`

它表示：

```text
offset：该多模态 item 的 placeholder 在 prompt_token_ids 中从哪里开始；
length：占了多少 token；
is_embed：哪些 token 位置要被多模态 embedding 替换。
```

---

## 12. MultiModalFeatureSpec 在哪里生成

`BaseMultiModalProcessor.apply()` 返回的是 `mm_input(...)`，不是 `MultiModalFeatureSpec`。

V1 engine input processor 在构造 `EngineCoreRequest` 时，把 `MultiModalInput` flatten 成 `mm_features`。

关键代码：

```python
if decoder_inputs["type"] == "multimodal":
    decoder_mm_inputs = decoder_inputs["mm_kwargs"]
    decoder_mm_positions = decoder_inputs["mm_placeholders"]
    decoder_mm_hashes = decoder_inputs["mm_hashes"]

    sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

    mm_features = []
    for modality, idx in sorted_mm_idxs:
        base_mm_hash = decoder_mm_hashes[modality][idx]
        mm_features.append(
            MultiModalFeatureSpec(
                data=decoder_mm_inputs[modality][idx],
                modality=modality,
                identifier=self._get_mm_identifier(base_mm_hash, lora_request),
                mm_position=decoder_mm_positions[modality][idx],
                mm_hash=base_mm_hash,
            )
        )
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:333` 到 `code/vllm/vllm/v1/engine/input_processor.py:368`

排序函数是：

```python
def argsort_mm_positions(mm_positions: MultiModalPlaceholders) -> list[tuple[str, int]]:
    ...
    sorted_flat_items = sorted(flat_items, key=lambda x: x[2].offset)
    return [(modality, idx) for modality, idx, _ in sorted_flat_items]
```

位置：`code/vllm/vllm/multimodal/utils.py:137` 到 `code/vllm/vllm/multimodal/utils.py:157`

这说明：

```text
mm_features 是按 prompt 中出现顺序排序的。
不同 modality 可以混排，例如 image、video、image。
Scheduler / ModelRunner 后续都可以假设 mm_features 已按 mm_position.offset 升序。
```

---

## 13. MultiModalFeatureSpec 包含什么

定义在 `inputs.py`：

```python
@dataclass
class MultiModalFeatureSpec:
    data: MultiModalKwargsItem | None
    modality: str
    identifier: str
    mm_position: PlaceholderRange
    mm_hash: str | None = None
```

位置：`code/vllm/vllm/multimodal/inputs.py:301` 到 `code/vllm/vllm/multimodal/inputs.py:332`

字段含义：

| 字段 | 含义 |
|---|---|
| `data` | 该 item 的模型 kwargs；如果已缓存，可能是 `None` |
| `modality` | `image` / `audio` / `video` 等 |
| `identifier` | encoder output cache 使用的 id，可能带 LoRA 前缀 |
| `mm_position` | 该 item 在 prompt token 中的 placeholder 范围 |
| `mm_hash` | processor output cache 使用的基础 hash，不带 LoRA 前缀 |

注意两个 hash 的语义不同：

```text
mm_hash：processor cache key，跨 LoRA 可共享；
identifier：encoder cache key，可能因为 LoRA 不同而不同。
```

`data` 可以为 `None` 的原因也很重要：

```python
Can be None if the item is cached, to skip IPC between API server
and engine core processes.
```

位置：`code/vllm/vllm/multimodal/inputs.py:311` 到 `code/vllm/vllm/multimodal/inputs.py:317`

---

## 14. feature hash 如何计算

hash 入口在 `ProcessorInputs.get_mm_hashes()`：

```python
def get_mm_hashes(self, model_id: str) -> MultiModalHashes:
```

位置：`code/vllm/vllm/multimodal/processing/inputs.py:25`

如果用户提供了 `mm_uuid_items`：

```python
if uuid_item is None or hf_processor_mm_kwargs:
    item = uuid_item if uuid_item is not None else item
    hashes.append(
        hasher.hash_kwargs(
            model_id=model_id,
            **{modality: item},
            **hf_processor_mm_kwargs,
        )
    )
else:
    hashes.append(uuid_item)
```

位置：`code/vllm/vllm/multimodal/processing/inputs.py:42` 到 `code/vllm/vllm/multimodal/processing/inputs.py:58`

如果没有 uuid：

```python
mm_hashes[modality] = [
    hasher.hash_kwargs(
        model_id=model_id,
        **{modality: item},
        **hf_processor_mm_kwargs,
    )
    for item in data_items
]
```

位置：`code/vllm/vllm/multimodal/processing/inputs.py:61` 到 `code/vllm/vllm/multimodal/processing/inputs.py:69`

所以 hash 输入包括：

```text
model_id；
modality 名称；
该 modality 的原始 item 或 uuid；
hf_processor_mm_kwargs。
```

为什么提供 uuid 后仍可能重新 hash？

```text
因为 hf_processor_mm_kwargs 会改变 processor output。
同一张图在不同 resize / crop / fps / processor 参数下，不能复用同一个 processed result。
```

底层 hasher 是 `MultiModalHasher.hash_kwargs()`：

```python
hasher_factory = _get_hasher_factory(envs.VLLM_MM_HASHER_ALGORITHM)
hasher = hasher_factory()

for k, v in sorted(kwargs.items(), key=lambda kv: kv[0]):
    for bytes_ in cls.iter_item_to_bytes(k, v):
        hasher.update(bytes_)

return hasher.hexdigest()
```

位置：`code/vllm/vllm/multimodal/hasher.py:153` 到 `code/vllm/vllm/multimodal/hasher.py:162`

支持的序列化对象包括：

```text
bytes / memoryview / str / int / float；
PIL.Image；
MediaWithBytes[Image]；
torch.Tensor；
numpy.ndarray；
list / tuple / dict；
其他对象 fallback 到 pickle。
```

相关位置：`code/vllm/vllm/multimodal/hasher.py:50` 到 `code/vllm/vllm/multimodal/hasher.py:151`

---

## 15. processor cache 在哪里接入

processor cache 的创建在 `MultiModalRegistry`。

判断 cache 类型：

```python
def _get_cache_type(self, vllm_config: VllmConfig) -> Literal[None, "processor_only", "lru", "shm"]:
```

位置：`code/vllm/vllm/multimodal/registry.py:268` 到 `code/vllm/vllm/multimodal/registry.py:292`

条件包括：

```text
模型必须支持 multimodal；
mm_processor_cache_gb 必须大于 0；
IPC cache 是否支持取决于 api process / data parallel 配置；
最终类型来自 mm_processor_cache_type，或者退化成 processor_only。
```

创建 sender cache：

```python
if cache_type is None:
    return None
elif cache_type == "processor_only":
    return MultiModalProcessorOnlyCache(vllm_config.model_config)
elif cache_type == "lru":
    return MultiModalProcessorSenderCache(vllm_config.model_config)
elif cache_type == "shm":
    return ShmObjectStoreSenderCache(vllm_config)
```

位置：`code/vllm/vllm/multimodal/registry.py:294` 到 `code/vllm/vllm/multimodal/registry.py:309`

processor 侧实际使用 cache 的入口是 `_cached_apply_hf_processor()`：

```python
cache = self.cache

_, passthrough_data = self._get_hf_mm_data(inputs.mm_data_items)
if cache is None or passthrough_data:
    return self._apply_hf_processor(inputs, timing_ctx)
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:1441` 到 `code/vllm/vllm/multimodal/processing/processor.py:1454`

注意：

```text
如果存在 passthrough_data，例如用户直接传 image_embeds，则不走 processor cache。
因为这类数据绕过 HF processor，cache processor output 意义不同。
```

cache miss 时只处理缺失 item：

```python
mm_is_cached, mm_missing_data_items = self._get_cache_missing_items(
    cache=cache,
    mm_data_items=inputs.mm_data_items,
    mm_hashes=mm_hashes,
)
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:1459` 到 `code/vllm/vllm/multimodal/processing/processor.py:1464`

然后合并 cached 和 missing 的 kwargs / prompt updates：

```python
mm_kwargs, mm_prompt_updates = self._merge_mm_kwargs(
    cache,
    mm_hashes=mm_hashes,
    mm_is_cached=mm_is_cached,
    mm_missing_kwargs=mm_missing_kwargs,
    mm_missing_prompt_updates=mm_missing_prompt_updates,
)
```

位置：`code/vllm/vllm/multimodal/processing/processor.py:1495` 到 `code/vllm/vllm/multimodal/processing/processor.py:1502`

所以 processor cache 缓存的是：

```text
每个多模态 item 的 MultiModalKwargsItem；
该 item 对应的 ResolvedPromptUpdate。
```

不是缓存最终文本输出，也不是缓存语言模型 hidden states。

---

## 16. receiver cache 在 Engine / Worker 侧如何接入

多进程 / IPC 场景下，API 进程和 engine / worker 进程之间还可能有 receiver cache。

EngineCore 添加请求时会更新 request 的 `mm_features`：

```python
if self.mm_receiver_cache is not None and request.mm_features:
    request.mm_features = self.mm_receiver_cache.get_and_update_features(
        request.mm_features
    )
```

位置：`code/vllm/vllm/v1/engine/core.py:862` 到 `code/vllm/vllm/v1/engine/core.py:864`

Worker 执行前也会对 scheduled new requests 应用 cache：

```python
def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
    mm_cache = self.mm_receiver_cache
    if mm_cache is None:
        return

    for req_data in scheduler_output.scheduled_new_reqs:
        req_data.mm_features = mm_cache.get_and_update_features(
            req_data.mm_features
        )
```

位置：`code/vllm/vllm/v1/worker/worker_base.py:330` 到 `code/vllm/vllm/v1/worker/worker_base.py:338`

receiver cache 更新 feature 的逻辑是：

```python
for feature in mm_features:
    cache_key = feature.mm_hash or feature.identifier
    self.touch_receiver_cache_item(cache_key, feature.data)

for feature in mm_features:
    cache_key = feature.mm_hash or feature.identifier
    feature.data = self.get_and_update_item(feature.data, cache_key)
return mm_features
```

位置：`code/vllm/vllm/multimodal/cache.py:589` 到 `code/vllm/vllm/multimodal/cache.py:608`

这里使用 `mm_hash` 优先，而不是 `identifier` 优先：

```text
mm_hash：跨 LoRA 共享 processor output；
identifier：兼容旧逻辑或用于 encoder output cache。
```

---

## 17. MultiModalRegistry 负责什么

`MultiModalRegistry` 是模型类到 processor 的分发表。

定义：

```python
class MultiModalRegistry:
    """
    A registry that dispatches data processing according to the model.
    """
```

位置：`code/vllm/vllm/multimodal/registry.py:98` 到 `code/vllm/vllm/multimodal/registry.py:101`

模型通过装饰器注册 processor：

```python
@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class Qwen2_5_VLForConditionalGeneration(...):
```

位置：`code/vllm/vllm/model_executor/models/qwen2_5_vl.py:1237` 到 `code/vllm/vllm/model_executor/models/qwen2_5_vl.py:1242`

注册时会把 factory 挂到模型类上：

```python
model_cls._processor_factory = _ProcessorFactories(
    info=info,
    dummy_inputs=dummy_inputs,
    processor=processor,
)
```

位置：`code/vllm/vllm/multimodal/registry.py:166` 到 `code/vllm/vllm/multimodal/registry.py:170`

创建 processor 时：

```python
model_cls = self._get_model_cls(model_config)
factories = model_cls._processor_factory
ctx = self._create_processing_ctx(model_config, tokenizer)
return factories.build_processor(ctx, cache=cache)
```

位置：`code/vllm/vllm/multimodal/registry.py:211` 到 `code/vllm/vllm/multimodal/registry.py:230`

因此：

```text
Registry 不直接处理 image / audio / video；
Registry 负责找到当前模型注册的 ProcessingInfo、DummyInputsBuilder、BaseMultiModalProcessor；
真正的字段映射和 placeholder 规则在模型专用 processor 中。
```

---

## 18. Qwen2.5-VL 例子：processor 如何定义 mapper 规则

以 Qwen2.5-VL 为例。

它的 processing info 返回 HF processor：

```python
class Qwen2_5_VLProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Qwen2_5_VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen2_5_VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
```

位置：`code/vllm/vllm/model_executor/models/qwen2_5_vl.py:1148` 到 `code/vllm/vllm/model_executor/models/qwen2_5_vl.py:1157`

它的 processor 增加一个 video 字段配置：

```python
class Qwen2_5_VLMultiModalProcessor(Qwen2VLMultiModalProcessor):
    def _get_mm_fields_config(...):
        return dict(
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            second_per_grid_ts=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )
```

位置：`code/vllm/vllm/model_executor/models/qwen2_5_vl.py:1160` 到 `code/vllm/vllm/model_executor/models/qwen2_5_vl.py:1169`

这说明：

```text
second_per_grid_ts 是 video modality 的字段；
按 batch 第一维拆分；
keep_on_cpu=True，说明该字段不会被普通 H2D 合批逻辑搬到 GPU。
```

它的 prompt update 逻辑根据 image/video 的 grid 计算 placeholder 数量：

```python
placeholder = {
    "image": vocab[hf_processor.image_token],
    "video": vocab[hf_processor.video_token],
}

merge_length = image_processor.merge_size**2

def get_replacement_qwen2vl(item_idx: int, modality: str):
    out_item = out_mm_kwargs[modality][item_idx]
    grid_thw = out_item[f"{modality}_grid_thw"].data
    num_tokens = int(grid_thw.prod()) // merge_length
    return [placeholder[modality]] * num_tokens
```

位置：`code/vllm/vllm/model_executor/models/qwen2_5_vl.py:1193` 到 `code/vllm/vllm/model_executor/models/qwen2_5_vl.py:1225`

返回的是两个 replacement 规则：

```python
return [
    PromptReplacement(
        modality=modality,
        target=[placeholder[modality]],
        replacement=partial(get_replacement_qwen2vl, modality=modality),
    )
    for modality in ("image", "video")
]
```

位置：`code/vllm/vllm/model_executor/models/qwen2_5_vl.py:1227` 到 `code/vllm/vllm/model_executor/models/qwen2_5_vl.py:1234`

这个例子说明模型专用 processor 负责两类关键信息：

```text
字段映射：HF output 哪些 key 属于 image/video/audio，怎么拆；
占位符映射：一个 <image>/<video> 要扩展成多少个 feature placeholder token。
```

---

## 19. 多个 image / video frame 如何组织

多张 image：

```text
parser 阶段：ImageProcessorItems 中有多个 item；
HF processor 阶段：可能返回 batched pixel_values / image_grid_thw；
field config 阶段：拆成 MultiModalKwargsItems["image"][0..N-1]；
feature spec 阶段：每张 image 一个 MultiModalFeatureSpec；
执行阶段：按 prompt 出现顺序和 modality 分组合批。
```

单个 video：

```text
parser 阶段：VideoProcessorItems 的一个 item，内部可能是多帧；
HF processor 阶段：可能输出 video_grid_thw、pixel_values_videos、second_per_grid_ts；
field config 阶段：这个 video item 对应一个 MultiModalKwargsItem；
placeholder 阶段：根据 T/H/W 计算它需要多少 video placeholder tokens；
feature spec 阶段：一个 video 对应一个 MultiModalFeatureSpec。
```

多个 video：

```text
VideoProcessorItems 中多个 item；
每个 item 有自己的 grid / metadata / placeholder range；
feature spec 按 prompt offset 排序，可以和 image 混排。
```

所以：

```text
video frame 不是每帧一个 MultiModalFeatureSpec；
通常是一个 video item 一个 MultiModalFeatureSpec，frames 是该 item data 内部的维度。
```

---

## 20. 执行前如何重新合批 kwargs

`MultiModalKwargsItems` 是按 item 拆开的，但模型执行时希望合批。

单 item 的数据结构是：

```python
class MultiModalKwargsItem(UserDict[str, MultiModalFieldElem]):
    """
    A dictionary of processed keyword arguments to pass to the model,
    corresponding to a single item in MultiModalDataItems.
    """
```

位置：`code/vllm/vllm/multimodal/inputs.py:854` 到 `code/vllm/vllm/multimodal/inputs.py:859`

合批函数是：

```python
def group_and_batch_mm_kwargs(
    mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[str, int, BatchedTensorInputs], None, None]:
```

位置：`code/vllm/vllm/multimodal/utils.py:236` 到 `code/vllm/vllm/multimodal/utils.py:272`

它的约束是：

```text
连续 items 才会被分组；
不同 modality 不会合在一个 batch；
字段集合不同不能合；
MultiModalSharedField 的 shared 值不同不能合。
```

注释说明：

```text
To simplify the implementation of embed_multimodal,
we add another restriction that the items in a batch must belong to the same modality.
```

位置：`code/vllm/vllm/multimodal/utils.py:242` 到 `code/vllm/vllm/multimodal/utils.py:252`

合批后的结果是 `BatchedTensorInputs`：

```text
{key: batched tensor or nested tensor}
```

然后模型的 `embed_multimodal(**kwargs)` 消费这些 kwargs。

接口定义：

```python
def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
    """
    Returns multimodal embeddings generated from multimodal kwargs
    to be merged with text embeddings.
    """
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:154` 到 `code/vllm/vllm/model_executor/models/interfaces.py:164`

接口还要求：

```text
返回的 multimodal embeddings 必须和它们在 prompt 中出现的多模态 item 顺序一致。
```

---

## 21. mm_features 在 Scheduler / ModelRunner 侧的作用

`EngineCoreRequest` 携带 `mm_features`：

```python
return EngineCoreRequest(
    ...
    mm_features=mm_features,
    ...
)
```

位置：`code/vllm/vllm/v1/engine/input_processor.py:370` 到 `code/vllm/vllm/v1/engine/input_processor.py:385`

V1 request 会保存它：

```python
self.mm_features = mm_features or []
```

位置：`code/vllm/vllm/v1/request.py:157`

Scheduler 输出新请求时，`NewRequestData` 也带着它：

```python
mm_features=request.mm_features,
```

位置：`code/vllm/vllm/v1/core/sched/output.py:52` 到 `code/vllm/vllm/v1/core/sched/output.py:56`

ModelRunner `_update_states()` 创建 worker 侧请求状态时保存：

```python
req_state = CachedRequestState(
    req_id=req_id,
    prompt_token_ids=new_req_data.prompt_token_ids,
    prompt_embeds=new_req_data.prompt_embeds,
    prompt_is_token_ids=new_req_data.prompt_is_token_ids,
    mm_features=new_req_data.mm_features,
    ...
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1224` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:1238`

后续 `mm_features` 会用于：

```text
判断某个 token window 内有哪些多模态 feature；
调度 encoder cache 输入；
构造 multimodal embeddings；
M-RoPE / XD-RoPE 位置计算；
把 encoder output scatter 到 text embeddings 对应位置。
```

例如 M-RoPE 接口直接接收 `mm_features`：

```python
def get_mrope_input_positions(
    self,
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
) -> tuple[torch.Tensor, int]:
```

位置：`code/vllm/vllm/model_executor/models/interfaces.py:1457` 到 `code/vllm/vllm/model_executor/models/interfaces.py:1462`

---

## 22. parser / processor / mapper / feature 的职责边界

### Parser 负责

```text
识别 modality；
规范化单个 item / batch item；
区分 raw media 和 embedding；
音频重采样 / channel normalize；
视频 metadata 检查；
提供 processor_data / passthrough_data 两个出口。
```

不负责：

```text
不调用 HF processor；
不生成 prompt_token_ids；
不替换 placeholder；
不生成 MultiModalFeatureSpec；
不执行模型 encoder。
```

### Processor 负责

```text
调用模型对应的 HF processor；
把 HF BatchFeature 转成 MultiModalKwargsItems；
计算 mm_hash；
生成模型专用 PromptUpdate；
确保 prompt 中 placeholder 数量和 feature 数量一致；
返回 MultiModalInput。
```

不负责：

```text
不做 Scheduler 调度；
不创建 EngineCoreRequest；
不运行 multimodal encoder；
不直接生成用户输出。
```

### Mapper 负责

当前代码里没有统一的 `Mapper` 类。

如果文档里使用 mapper 这个词，它实际对应：

```text
_get_mm_fields_config()
  + MultiModalFieldConfig
  + MultiModalKwargsItems.from_hf_inputs()
  + group_and_batch_mm_kwargs()
```

也就是：

```text
把 HF processor 的 batch output 映射为模型 forward / embed_multimodal 能消费的 kwargs。
```

### FeatureSpec 负责

```text
把单个多模态 item 的 data、modality、hash、placeholder 位置打包；
作为 EngineCoreRequest / SchedulerOutput / Worker request state 的多模态执行规格；
连接调度、encoder cache、ModelRunner 和模型接口。
```

---

## 23. 常见问题

### 23.1 image / audio / video 是否走相同 parser？

共用 `MultiModalDataParser` 框架，但 subparser 不同。

```text
image：处理 PIL / ndarray / tensor / MediaWithBytes / embeddings；
audio：额外处理 sampling rate 和 channel normalize；
video：额外处理 frame batch 和 metadata；
vision_chunk：处理统一 vision chunk，不支持 embedding。
```

### 23.2 processor cache 在哪里接入？

两处：

```text
API / processor 侧：BaseMultiModalProcessor._cached_apply_hf_processor()
Engine / Worker receiver 侧：BaseMultiModalReceiverCache.get_and_update_features()
```

processor cache 避免重复 HF processor 计算；receiver cache 减少进程间重复传 tensor。

### 23.3 feature hash 如何计算？

由 `ProcessorInputs.get_mm_hashes()` 调用 `MultiModalHasher.hash_kwargs()`。

输入通常包括：

```text
model_id + modality item + hf_processor_mm_kwargs
```

有 uuid 且没有 processor kwargs 时可以直接用 uuid；有 processor kwargs 时仍会重新 hash。

### 23.4 mapper output 是 tensor 还是 model kwargs？

单 item 是 `MultiModalKwargsItem`，里面每个 key 对应 `MultiModalFieldElem`。

执行前合批后变成 `BatchedTensorInputs`，也就是模型 `embed_multimodal(**kwargs)` 接收的 tensor kwargs。

### 23.5 多个 image / video frame 如何组织？

```text
多个 image：每张 image 一个 item，一个 MultiModalFeatureSpec；
一个 video：通常一个 video item，一个 MultiModalFeatureSpec，frames 在 item 内部；
多个 video：多个 video items，各自有 feature spec；
image/video/audio 可以按 prompt offset 混排。
```

### 23.6 MultiModalFeatureSpec 是最终 encoder output 吗？

不是。

它只是执行规格：

```text
MultiModalFeatureSpec.data 是 processor output kwargs；
多模态 encoder output 是后续 ModelRunner / model.embed_multimodal 运行后产生的 embeddings。
```

### 23.7 embedding 输入会不会走 HF processor？

不会。

embedding 输入通过 `EmbeddingItems.get_passthrough_data()` 进入 `processed_data`，绕过 HF processor 的媒体处理。

但它仍然要参与：

```text
hash；
prompt placeholder 对齐；
MultiModalKwargsItems；
MultiModalFeatureSpec；
执行侧合批 / scatter。
```

---

## 24. 从“回答问题”的角度总结

如果要问：

```text
多模态 parser 和 processor 如何生成 feature？
```

可以回答：

```text
vLLM 先用 MultiModalDataParser 把用户传入的 multi_modal_data 规范化成 MultiModalDataItems。
每个 modality item 会暴露 get_processor_data() 和 get_passthrough_data()，raw image/audio/video 进入 HF processor，预计算 embedding 则作为 passthrough data 直接进入后续 kwargs。

随后模型注册到 MultiModalRegistry 的 BaseMultiModalProcessor 调用对应 HF processor，得到 prompt token ids 和 BatchFeature；再通过模型专用 _get_mm_fields_config() 与 MultiModalFieldConfig，把 BatchFeature 拆成每个 item 的 MultiModalKwargsItem。
同时 processor 计算每个 item 的 mm_hash，并通过模型专用 _get_prompt_updates() 将 prompt 中的 <image>/<video>/<audio> 等占位符扩展或替换成与 encoder feature 数量一致的 placeholder tokens，最终得到 PlaceholderRange。

processor 返回 MultiModalInput，其中包含 prompt_token_ids、mm_kwargs、mm_hashes 和 mm_placeholders。
V1 engine input_processor 再把这些按 placeholder offset 排序并 flatten 成 list[MultiModalFeatureSpec]，放入 EngineCoreRequest.mm_features。
这些 feature spec 后续被 Scheduler、Worker、ModelRunner、encoder cache 和模型的 embed_multimodal() 使用，最终才运行多模态 encoder 并生成要 merge 到文本 embeddings 的 multimodal embeddings。
```

最短心智模型：

```text
Parser 产出 items；Processor 产出 MultiModalInput；InputProcessor 产出 FeatureSpec；ModelRunner 消费 FeatureSpec 运行 encoder。
```

---

## 25. 最关键流程图

```text
User request
  ├─ prompt
  ├─ multi_modal_data
  └─ multi_modal_uuids / hf_processor_mm_kwargs

      ↓

MultiModalDataParser.parse_mm_data()
  ├─ _parse_image_data()
  │    ├─ ImageProcessorItems
  │    └─ ImageEmbeddingItems
  ├─ _parse_audio_data()
  │    ├─ resample / normalize_audio
  │    ├─ AudioProcessorItems
  │    └─ AudioEmbeddingItems
  ├─ _parse_video_data()
  │    ├─ metadata check
  │    ├─ VideoProcessorItems
  │    └─ VideoEmbeddingItems
  └─ _parse_vision_chunk_data()

      ↓

MultiModalDataItems
  ├─ get_processor_data()
  │    └─ raw media for HF processor
  └─ get_passthrough_data()
       └─ precomputed embeddings / dict inputs

      ↓

BaseMultiModalProcessor.apply()
  ├─ _cached_apply_hf_processor()
  │    ├─ get_mm_hashes()
  │    ├─ processor cache lookup
  │    ├─ _apply_hf_processor_main()
  │    │    ├─ _call_hf_processor()
  │    │    ├─ processed_data.update(passthrough_data)
  │    │    └─ prompt_ids
  │    ├─ MultiModalKwargsItems.from_hf_inputs()
  │    │    └─ _get_mm_fields_config()
  │    ├─ _get_mm_prompt_updates()
  │    │    └─ _get_prompt_updates()
  │    └─ merge cached + missing items
  │
  ├─ _maybe_apply_prompt_updates()
  │    ├─ find or apply PromptInsertion / PromptReplacement
  │    ├─ _find_mm_placeholders()
  │    └─ PlaceholderRange(offset, length, is_embed)
  │
  └─ mm_input(
       prompt_token_ids,
       mm_kwargs,
       mm_hashes,
       mm_placeholders,
     )

      ↓

V1 engine input_processor
  ├─ argsort_mm_positions(mm_placeholders)
  ├─ flatten by prompt order
  └─ MultiModalFeatureSpec(
       data=MultiModalKwargsItem,
       modality="image" / "audio" / "video",
       identifier=encoder-cache id,
       mm_position=PlaceholderRange,
       mm_hash=processor-cache hash,
     )

      ↓

EngineCoreRequest.mm_features
  → Request.mm_features
  → SchedulerOutput.scheduled_new_reqs[*].mm_features
  → WorkerBase._apply_mm_cache()
  → GPUModelRunner._update_states()
  → CachedRequestState.mm_features

      ↓

ModelRunner / multimodal encoder
  ├─ get_mm_features_in_window()
  ├─ group_and_batch_mm_kwargs()
  ├─ model.embed_multimodal(**batched_kwargs)
  ├─ M-RoPE / XD-RoPE position logic uses mm_features
  └─ scatter multimodal embeddings into placeholder positions
```

---

## 26. 最关键对象速查

```text
MultiModalDataParser
  原始数据解析器，按 modality 归一化输入。

ModalityDataItems
  单 modality 的 item 容器，决定 processor_data / passthrough_data。

BaseMultiModalProcessor
  vLLM 模型专用 processor，编排 HF processor、hash、cache、prompt update。

MultiModalFieldConfig
  描述 HF processor output 中某个字段如何拆成每个 item 的 elem。

MultiModalKwargsItem
  单个多模态 item 的模型 kwargs。

MultiModalKwargsItems
  按 modality 分组的多个 MultiModalKwargsItem。

PromptUpdate
  模型专用 placeholder 插入 / 替换规则。

PlaceholderRange
  多模态 item 在 prompt token ids 中的占位符位置。

MultiModalInput
  processor 输出，包含 prompt_token_ids、mm_kwargs、mm_hashes、mm_placeholders。

MultiModalFeatureSpec
  V1 请求 / 调度 / 执行层携带的每个多模态 item 规格。

BatchedTensorInputs
  执行前把多个 MultiModalKwargsItem 合批后的 tensor kwargs。
```

---

## 27. 最小心智模型

```text
MultiModalDataParser
  = raw media / embeddings → normalized items

BaseMultiModalProcessor
  = normalized items + prompt → token ids + item kwargs + hashes + placeholder ranges

MultiModalFieldConfig / MultiModalKwargsItems
  = HF BatchFeature → per-item model kwargs

MultiModalFeatureSpec
  = per-item kwargs + modality + hash + prompt position

ModelRunner
  = feature spec → batched kwargs → embed_multimodal → multimodal embeddings
```
