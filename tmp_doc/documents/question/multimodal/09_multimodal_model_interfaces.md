# 09. 多模态模型类需要提供什么接口？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\interfaces.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\registry.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\registry.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\processing\processor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\model.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\llava.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\qwen2_vl.py`

本问题关注：vLLM 如何识别一个模型是多模态模型；多模态模型类需要提供哪些接口；processor 如何把原始 image / audio / video / prompt_embeds 转成模型 kwargs 和 placeholder；`GPUModelRunner` 如何调用模型侧 encoder、缓存 multimodal embeddings、把 embeddings 合并回文本 token 位置，并继续 decoder forward。

---

## 1. 一句话回答

vLLM 中多模态模型类的核心契约是实现 `SupportsMultiModal`：

```text
模型类继承 SupportsMultiModal
  → 通过 MULTIMODAL_REGISTRY.register_processor(...) 注册 processor / info / dummy builder
  → ModelRegistry.inspect_model_cls() 标记 supports_multimodal=True
  → ModelConfig 初始化 MultiModalConfig
  → MultiModalRegistry 判断是否真正启用多模态输入
  → GPUModelRunner._execute_mm_encoder() 调用 model.embed_multimodal(**mm_kwargs)
  → GPUModelRunner._gather_mm_embeddings() 取 encoder cache 中当前窗口需要的 embedding
  → model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
  → _merge_multimodal_embeddings() 把多模态 embedding scatter 到 placeholder token 位置
  → model.forward(..., inputs_embeds=inputs_embeds, **model_kwargs)
```

所以多模态模型类不只是多一个 vision tower。它至少要回答三类问题：

```text
1. 我是不是多模态模型？
   通过继承 SupportsMultiModal / supports_multimodal=True 暴露给 registry。

2. 原始媒体如何进入模型？
   通过 MULTIMODAL_REGISTRY.register_processor(...) 注册处理器，产出 mm_kwargs、mm_hashes、mm_placeholders。

3. 媒体特征如何变成 decoder 可吃的输入？
   通过 embed_multimodal() 生成 embedding，再通过 embed_input_ids() / _merge_multimodal_embeddings() 合并到文本 embedding。
```

---

## 2. 最小主链路

以 V1 `GPUModelRunner` 为主，执行链路可以记为：

```text
请求进入引擎前 / processor 阶段：
raw multimodal data
  → MultiModalProcessor.apply(...)
  → prompt_token_ids
  → mm_kwargs
  → mm_hashes
  → mm_placeholders / PlaceholderRange
  → MultiModalFeatureSpec

调度和 worker 阶段：
SchedulerOutput.scheduled_encoder_inputs
  → GPUModelRunner._execute_mm_encoder()
  → model.embed_multimodal(**mm_kwargs)
  → encoder_cache[mm_hash] = embedding

本轮 forward 前：
GPUModelRunner._gather_mm_embeddings()
  → 按当前 token window 从 encoder_cache 切片
  → 生成 mm_embeds
  → 生成 is_mm_embed bool mask

合并输入：
model.embed_input_ids(
    input_ids,
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
  → text embeddings
  → _merge_multimodal_embeddings()
  → inputs_embeds

真正 forward：
GPUModelRunner._model_forward(
    input_ids=None 或 raw input_ids,
    positions=positions,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
  → model.forward()
  → hidden_states
  → compute_logits() / pooler
```

关键分界点在 `_preprocess()`：

```text
支持多模态 + PP first rank + 非 encoder-decoder：
  跑多模态 encoder，合并 embeddings，forward 使用 inputs_embeds。

普通文本模型：
  不提前 embed，forward 使用 input_ids，embedding layer 可以留在 CUDA graph 内。

encoder-decoder 模型：
  也复用 _execute_mm_encoder()，但 encoder 输出通过 model_kwargs["encoder_outputs"] 传给 decoder，
  不走 prompt replacement 合并路径。
```

对应源码：`gpu_model_runner.py:3426` 到 `gpu_model_runner.py:3568`。

---

## 3. vLLM 如何识别模型是多模态模型

### 3.1 模型类层：`SupportsMultiModal`

接口定义在 `interfaces.py:94` 开始：

```python
@runtime_checkable
class SupportsMultiModal(Protocol):
    supports_multimodal: ClassVar[Literal[True]] = True
```

位置：`interfaces.py:94` 到 `interfaces.py:105`

这意味着模型类只要继承 `SupportsMultiModal`，`supports_multimodal(model_cls)` 就会返回 True。

判断函数非常直接：

```python
def supports_multimodal(model: type[object] | object, ...) -> ...:
    return getattr(model, "supports_multimodal", False)
```

位置：`interfaces.py:451` 到 `interfaces.py:462`

同时还有几个相关 capability flag：

- `supports_multimodal_raw_input_only`：模型支持多模态，但只吃 raw input，不走常规 embedding 合并路径。
- `supports_encoder_tp_data`：是否支持 `--mm-encoder-tp-mode data`。
- `requires_raw_input_tokens`：即使走多模态 `inputs_embeds`，模型 forward 仍需要原始 `input_ids`。

位置：`interfaces.py:107` 到 `interfaces.py:123`，辅助判断在 `interfaces.py:465` 到 `interfaces.py:474`。

### 3.2 registry 层：`_ModelInfo.supports_multimodal`

`ModelRegistry` inspect 模型类时会把接口能力固化到 `_ModelInfo`：

```python
supports_multimodal=supports_multimodal(model)
supports_multimodal_raw_input_only=supports_multimodal_raw_input_only(model)
requires_raw_input_tokens=requires_raw_input_tokens(model)
supports_multimodal_encoder_tp_data=supports_multimodal_encoder_tp_data(model)
```

位置：`registry.py:747` 到 `registry.py:796`

对外判断多模态模型的方法是：

```python
def is_multimodal_model(...):
    model_cls, _ = self.inspect_model_cls(...)
    return model_cls.supports_multimodal
```

位置：`registry.py:1296` 到 `registry.py:1302`

而模型类的解析链路是：

```text
HF config.architectures
  → ModelRegistry.inspect_model_cls(...)
  → _normalize_arch(...)
  → _try_inspect_model_cls(...)
  → _ModelInfo.from_model_cls(...)
```

或真正加载时：

```text
HF config.architectures
  → ModelRegistry.resolve_model_cls(...)
  → _try_load_model_cls(...)
```

位置：`registry.py:1174` 到 `registry.py:1278`。

### 3.3 `ModelConfig` 层：只有支持多模态才初始化 `MultiModalConfig`

`ModelConfig.__post_init__` 中会检查 `_model_info.supports_multimodal`。如果为 True，就初始化 `MultiModalConfig`：

```python
if self._model_info.supports_multimodal:
    mm_config_kwargs = dict(...)
    self.multimodal_config = MultiModalConfig(**mm_config_kwargs)
```

位置：`model.py:663` 到 `model.py:701`

`ModelConfig.is_multimodal_model` 不是再次 inspect 模型类，而是看 `multimodal_config` 是否存在：

```python
@property
def is_multimodal_model(self) -> bool:
    return self.multimodal_config is not None
```

位置：`model.py:1562` 到 `model.py:1564`

所以这里有两个层次：

```text
模型类能力：_model_info.supports_multimodal
运行时配置：model_config.multimodal_config is not None
```

模型类支持多模态，才会有 `MultiModalConfig`；有 `MultiModalConfig`，才会被运行时当作多模态模型配置处理。

### 3.4 `MultiModalRegistry` 层：支持不等于本次启用

`MultiModalRegistry.supports_multimodal_inputs()` 进一步判断本次运行是否真的需要多模态基础设施：

```python
if not model_config.is_multimodal_model:
    return False

info = self._create_processing_info(...)

if all(mm_config.get_limit_per_prompt(modality) == 0
       for modality in info.supported_mm_limits):
    if mm_config.enable_mm_embeds:
        return True
    return False

return True
```

位置：`multimodal/registry.py:103` 到 `multimodal/registry.py:140`

这段逻辑说明：

```text
模型类支持多模态
  不代表当前请求一定走多模态；

如果所有 supported modalities 的 limit 都是 0：
  默认按 text-only 跑；

但如果 enable_mm_embeds=True：
  即使 encoder 不跑，也要保留多模态基础设施，
  因为用户可能传入预计算 embeddings。
```

---

## 4. 多模态模型类的核心接口

### 4.1 `get_placeholder_str()`：聊天模板里的占位字符串

接口：

```python
@classmethod
def get_placeholder_str(cls, modality: str, i: int) -> str | None:
    ...
```

位置：`interfaces.py:147` 到 `interfaces.py:152`

它用于告诉上层渲染 / chat utils：某种 modality 的第 i 个 item 在 prompt 中应该用什么字符串占位。

LLaVA 示例：

```python
if modality.startswith("image"):
    return "<image>"
```

位置：`llava.py:527` 到 `llava.py:532`

Qwen2-VL 示例：

```python
if modality.startswith("image"):
    return "<|vision_start|><|image_pad|><|vision_end|>"
if modality.startswith("video"):
    return "<|vision_start|><|video_pad|><|vision_end|>"
```

位置：`qwen2_vl.py:1279` 到 `qwen2_vl.py:1286`

### 4.2 `embed_multimodal()`：模型侧 encoder 入口

接口：

```python
def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
    ...
```

位置：`interfaces.py:154` 到 `interfaces.py:164`

这是模型类最核心的多模态接口。`GPUModelRunner._execute_mm_encoder()` 最终调用的就是它：

```python
batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
```

位置：`gpu_model_runner.py:3082` 到 `gpu_model_runner.py:3085`

返回类型是 `MultiModalEmbeddings`：

```python
MultiModalEmbeddings = list[Tensor] | Tensor | tuple[Tensor, ...]
```

约定：

```text
1. 可以返回 list / tuple，每个元素是一个媒体 item 的 2D tensor；
2. 也可以返回一个 3D tensor，用 batch 维度表示多个媒体 item；
3. 返回顺序必须和媒体 item 在 prompt 中出现的顺序一致。
```

位置：`interfaces.py:64` 到 `interfaces.py:71`，顺序约束见 `interfaces.py:159` 到 `interfaces.py:163`。

### 4.3 `embed_input_ids()`：文本 embedding 和多模态 embedding 的统一入口

`SupportsMultiModal` 提供了默认实现：

```python
def embed_input_ids(
    self,
    input_ids: Tensor,
    multimodal_embeddings: MultiModalEmbeddings | None = None,
    *,
    is_multimodal: Tensor | None = None,
) -> Tensor:
    inputs_embeds = self._embed_text_input_ids(
        input_ids,
        self.get_language_model().embed_input_ids,
        is_multimodal=is_multimodal,
    )

    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds

    return _merge_multimodal_embeddings(...)
```

位置：`interfaces.py:343` 到 `interfaces.py:409`

它做两件事：

```text
1. 调 language model 的 embed_input_ids() 得到文本 embeddings；
2. 如果有 multimodal_embeddings，就按 is_multimodal mask 把它们覆盖到 placeholder 位置。
```

`GPUModelRunner._preprocess()` 中，多模态路径会调用：

```python
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

位置：`gpu_model_runner.py:3484` 到 `gpu_model_runner.py:3488`

注意：`embed_input_ids()` 的默认实现依赖 `get_language_model()` 能找到底层语言模型。

### 4.4 `get_language_model()`：找到真正的文本 decoder

默认实现位置：`interfaces.py:177` 到 `interfaces.py:213`。

它优先使用 `_mark_language_model()` 记录的 `_language_model_names`，找不到时 fallback 到子模块里第一个有 `embed_input_ids` 的模块：

```text
_language_model_names
  → common_prefix(...)
  → getattr(...)
  → hasattr(mod, "embed_input_ids")
  → 返回语言模型

fallback:
  遍历 children()
  → 找到有 embed_input_ids 的 child
```

如果模型类没有用 `_mark_language_model()`，也没有 child 暴露 `embed_input_ids`，默认 `embed_input_ids()` 会失败。

典型写法见 LLaVA：

```python
with self._mark_language_model(vllm_config):
    self.language_model = init_vllm_registered_model(...)
```

位置：`llava.py:578` 到 `llava.py:583`

Qwen2-VL 也是同样模式：`qwen2_vl.py:1306` 到 `qwen2_vl.py:1311`。

### 4.5 `_mark_tower_model()` / `_mark_language_model()`：让运行时知道哪些模块是 tower，哪些是 LLM

`SupportsMultiModal` 提供三个 context manager：

```text
_mark_language_model(...)
_mark_tower_model(...)
_mark_composite_model(...)
```

位置：`interfaces.py:215` 到 `interfaces.py:323`

它们的作用不是 forward，而是初始化和运行时裁剪：

```text
_mark_language_model：
  标记语言模型模块；
  在 --mm-encoder-only 模式下可以跳过 language model 初始化。

_mark_tower_model：
  标记视觉 / 音频 / 视频 tower；
  当某个 modality 的 --limit-mm-per-prompt 为 0 时，可以跳过对应 tower 初始化。

_mark_composite_model：
  同时标记 language model 和多个 modality tower。
```

LLaVA：

```text
_mark_tower_model(vllm_config, "image")
  → vision_tower
  → multi_modal_projector

_mark_language_model(vllm_config)
  → language_model
```

位置：`llava.py:562` 到 `llava.py:583`

Qwen2-VL：

```text
_mark_tower_model(vllm_config, {"image", "video"})
  → visual

_mark_language_model(vllm_config)
  → language_model
```

位置：`qwen2_vl.py:1298` 到 `qwen2_vl.py:1311`

### 4.6 `configure_mm_token_handling()`：处理 OOV 多模态 token

接口实现：

```python
def configure_mm_token_handling(self, vocab_size: int, mm_token_ids: list[int]):
    self._has_oov_mm_tokens = any(tok_id >= vocab_size for tok_id in mm_token_ids)
```

位置：`interfaces.py:166` 到 `interfaces.py:175`

如果 placeholder token id 超出文本 vocab，`embed_input_ids()` 不能直接把这些 id 喂给 embedding table。默认实现会在 `_embed_text_input_ids()` 中把多模态位置先 mask 成 0，再取文本 embedding：

```python
in_vocab_ids = input_ids.masked_fill(is_multimodal, 0)
return embed_input_ids(in_vocab_ids)
```

位置：`interfaces.py:355` 到 `interfaces.py:372`

随后多模态 embedding 会覆盖这些位置，所以 placeholder id 本身不会影响最终输入。

LLaVA 在初始化中调用：

```python
self.configure_mm_token_handling(
    vocab_size=config.text_config.vocab_size,
    mm_token_ids=[config.image_token_index],
)
```

位置：`llava.py:544` 到 `llava.py:547`

### 4.7 `forward()` / `compute_logits()` / `load_weights()`：仍然是普通模型契约

多模态模型最终仍然需要像普通 generation model 一样暴露：

```text
forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs)
compute_logits(hidden_states)
load_weights(weights)
```

LLaVA 的 forward 说明了关键点：prompt 中已经为图像 embedding 插入了足够的 placeholder token，因此 `positions` 和 attention metadata 与最终 embedding 序列长度一致。

位置：`llava.py:668` 到 `llava.py:720`

它实际 forward 给语言模型：

```python
hidden_states = self.language_model.model(
    input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
)
```

位置：`llava.py:716` 到 `llava.py:718`

logits 仍然委托给语言模型：

```python
return self.language_model.compute_logits(hidden_states)
```

位置：`llava.py:722` 到 `llava.py:726`

### 4.8 `get_mm_mapping()` / token count 接口：LoRA 和模块映射需要

部分多模态模型还会提供：

```text
get_mm_mapping()
get_num_mm_encoder_tokens(num_image_tokens)
get_num_mm_connector_tokens(num_vision_tokens)
```

它们不是 `SupportsMultiModal` 最小 forward 路径必需，但会被 LoRA / tower connector LoRA 等路径使用。

`GPUModelRunner._execute_mm_encoder()` 中，如果 LoRA manager 支持 tower / connector LoRA，会调用：

```python
self.model.get_num_mm_encoder_tokens(...)
self.model.get_mm_mapping()
self.model.get_num_mm_connector_tokens(...)
```

位置：`gpu_model_runner.py:2941` 到 `gpu_model_runner.py:3008`

LLaVA 示例：

```python
def get_mm_mapping(self) -> MultiModelKeys:
    return MultiModelKeys.from_string_field(
        language_model="language_model",
        connector="multi_modal_projector",
        tower_model="vision_tower",
    )
```

位置：`llava.py:732` 到 `llava.py:740`

它还说明 LLaVA 的 encoder token 和 connector token 都是一进一出：

```text
get_num_mm_encoder_tokens(num_image_tokens) = num_image_tokens
get_num_mm_connector_tokens(num_vision_tokens) = num_vision_tokens
```

位置：`llava.py:742` 到 `llava.py:756`

---

## 5. processor 注册：模型类还必须告诉 vLLM 怎么处理原始媒体

### 5.1 `MULTIMODAL_REGISTRY.register_processor(...)`

多模态模型类通常会被装饰器包住：

```python
@MULTIMODAL_REGISTRY.register_processor(
    processor_factory,
    info=processing_info_factory,
    dummy_inputs=dummy_inputs_builder_factory,
)
class SomeMultimodalModel(nn.Module, SupportsMultiModal, ...):
    ...
```

注册逻辑在 `multimodal/registry.py:142` 到 `multimodal/registry.py:174`。

这个装饰器会给模型类写入：

```python
model_cls._processor_factory = _ProcessorFactories(
    info=info,
    dummy_inputs=dummy_inputs,
    processor=processor,
)
```

位置：`multimodal/registry.py:157` 到 `multimodal/registry.py:172`

`MultiModalRegistry._get_model_cls()` 会要求多模态模型类有 `_processor_factory`：

```python
if not hasattr(model_cls, "_processor_factory"):
    raise ValueError("Model class ... has no registered multimodal processor")
```

位置：`multimodal/registry.py:176` 到 `multimodal/registry.py:186`

所以：

```text
继承 SupportsMultiModal
  只说明模型类有多模态能力；

注册 processor
  才说明 vLLM 知道如何把请求里的媒体数据转成这个模型的输入格式。
```

### 5.2 processor 产物：`prompt_token_ids` / `mm_kwargs` / `mm_hashes` / `mm_placeholders`

processor 的核心职责是：

```text
1. 调 HF processor / 自定义 processor，把原始媒体转成 pixel_values / grid_thw / audio features 等；
2. 修改 prompt，把 <image> / <video> 等占位符扩展成真实数量的 placeholder token；
3. 记录每个媒体 item 在 prompt token 序列中的 range；
4. 生成 hash，便于 worker 侧 encoder_cache 复用；
5. 生成 dummy inputs，用于 profiling / cache / memory planning。
```

`PromptUpdate` 定义了如何插入或替换 prompt 中的占位内容：

```python
class PromptUpdate(ABC):
    modality: str
    target: PromptUpdateTarget
    content: PromptUpdateContent
    mode: UpdateMode
```

位置：`processing/processor.py:297` 到 `processing/processor.py:319`

`PlaceholderFeaturesInfo` 最终可以转成 `PlaceholderRange`：

```python
return PlaceholderRange(
    offset=self.start_idx,
    length=self.length,
    is_embed=self.is_embed,
)
```

位置：`processing/processor.py:675` 到 `processing/processor.py:694`

其中 `is_embed` 很重要：某些模型的替换文本里并不是所有 token 都要被多模态 embedding 覆盖，`is_embed` 可以精确标记哪些位置是 embedding 位置。

`PromptUpdateDetails.is_embed` 的含义见 `processing/processor.py:205` 到 `processing/processor.py:222`。

### 5.3 dummy inputs：不是可选的边角功能

注册 processor 时必须提供 `dummy_inputs`。`MultiModalRegistry.get_dummy_mm_inputs()` 会用它生成 profiling 输入：

```python
processor_inputs = processor.dummy_inputs.get_dummy_processor_inputs(...)
mm_inputs = processor.apply(processor_inputs, ...)
```

位置：`multimodal/registry.py:232` 到 `multimodal/registry.py:266`

这影响：

```text
GPU memory profiling
max multimodal tokens 估计
encoder cache / CUDA graph capture 的容量规划
limit-mm-per-prompt 配置验证
```

因此新增一个多模态模型时，只实现 `embed_multimodal()` 不够，还要补齐 processing info 和 dummy input builder。

---

## 6. `GPUModelRunner` 如何调用这些接口

### 6.1 `_execute_mm_encoder()`：收集本轮需要跑 encoder 的媒体输入

入口：

```python
def _execute_mm_encoder(self, scheduler_output: SchedulerOutput) -> list[torch.Tensor]:
```

位置：`gpu_model_runner.py:2889` 到 `gpu_model_runner.py:3098`

它先从 scheduler output / request state 收集：

```text
mm_hashes
mm_kwargs
mm_lora_refs
```

位置：`gpu_model_runner.py:2892` 到 `gpu_model_runner.py:2897`

特殊情况：`prompt_embeds` 是 passthrough modality，已经是模型 embedding 空间的 tensor，不需要 encoder：

```python
self.encoder_cache[mm_hashes[i]] = pe_tensor.to(self.device)
```

位置：`gpu_model_runner.py:2899` 到 `gpu_model_runner.py:2924`

普通 image / audio / video 会经过分组 batching：

```python
for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(...):
    batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
```

位置：`gpu_model_runner.py:3013` 到 `gpu_model_runner.py:3085`

执行后会检查输出 item 数量并写入 encoder cache：

```python
sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
encoder_outputs.extend(batch_outputs)

for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
```

位置：`gpu_model_runner.py:3087` 到 `gpu_model_runner.py:3096`

### 6.2 为什么要分组 batching

注释说明了当前策略：

```text
如果 batch 中有多个 modality，或者和前一个 item 的 modality 不同，就分开处理，
这样能保持 item 顺序，同时尽量对同一 modality 做 batching。
```

位置：`gpu_model_runner.py:2932` 到 `gpu_model_runner.py:2938`

原因是 `embed_multimodal()` 的返回顺序必须和 prompt 中媒体出现顺序一致；如果任意重排 encoder batch，就必须额外重排 encoder outputs。

### 6.3 `_gather_mm_embeddings()`：只取当前 chunk 需要的 embedding

入口：

```python
def _gather_mm_embeddings(...) -> tuple[list[torch.Tensor], torch.Tensor]:
```

位置：`gpu_model_runner.py:3100` 到 `gpu_model_runner.py:3196`

它不是把整段 prompt 的所有多模态 embedding 都塞进本轮 forward，而是按当前调度窗口切片：

```text
req_state.num_computed_tokens
num_scheduled_tokens
mm_feature.mm_position.offset
mm_feature.mm_position.length
  → start_idx / end_idx
  → curr_embeds_start / curr_embeds_end
  → encoder_output[...] 切片
```

对应代码：`gpu_model_runner.py:3116` 到 `gpu_model_runner.py:3169`。

同时它会构造 `is_mm_embed`：

```python
is_mm_embed = torch.zeros(total_num_scheduled_tokens, dtype=torch.bool, device="cpu")
...
is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = True
```

位置：`gpu_model_runner.py:3107` 到 `gpu_model_runner.py:3110`，以及 `gpu_model_runner.py:3159` 到 `gpu_model_runner.py:3168`

这个 mask 是后面 `_merge_multimodal_embeddings()` 的 scatter 位置依据。

### 6.4 `_preprocess()`：把多模态路径折叠成统一 forward 输入

多模态路径在 `_preprocess()` 中：

```python
if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
    self._execute_mm_encoder(scheduler_output)
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
    inputs_embeds_scheduled = self.model.embed_input_ids(
        self.input_ids.gpu[:num_scheduled_tokens],
        multimodal_embeddings=mm_embeds,
        is_multimodal=is_mm_embed,
    )
    self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)
    input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
```

位置：`gpu_model_runner.py:3447` 到 `gpu_model_runner.py:3495`

`_prepare_mm_inputs()` 决定 forward 是否还传 `input_ids`：

```python
if self.model.requires_raw_input_tokens:
    input_ids = self.input_ids.gpu[:num_tokens]
else:
    input_ids = None

inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
```

位置：`gpu_model_runner.py:3415` 到 `gpu_model_runner.py:3424`

大多数多模态 decoder 只需要 `inputs_embeds`，所以 `input_ids=None`。少数模型声明 `requires_raw_input_tokens=True` 时，runner 会同时传 raw input ids。

### 6.5 真正的 forward 仍然由 `_model_forward()` 调用模型

多模态合并完成后，后续和普通模型一样进入：

```python
self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

这部分在执行文档的 `07_model_forward_and_logits.md` 中已经梳理过。这里要强调的是：多模态模型类的 `forward()` 必须能接受 `inputs_embeds`，并把它交给底层语言模型。

LLaVA 示例见 `llava.py:668` 到 `llava.py:720`。

---

## 7. embedding 合并的精确语义

### 7.1 `_merge_multimodal_embeddings()` 做的是覆盖，不是拼接

合并函数：

```python
def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
    return inputs_embeds
```

位置：`models/utils.py:479` 到 `models/utils.py:515`

它的语义是：

```text
prompt token 序列已经提前插入 placeholder token；
KV cache、position、attention metadata 都按插入后的 token 长度计算；
合并时只把 placeholder token 对应位置的 text embedding 覆盖为多模态 embedding；
不会在 forward 前再改变序列长度。
```

这也是为什么 processor 必须准确计算 placeholder token 数。

### 7.2 数量不匹配会直接报错

如果 `mm_embeds_flat` 的 token 数和 `is_multimodal.sum()` 不一致，会报：

```text
Attempted to assign ... multimodal tokens to ... placeholders
```

位置：`models/utils.py:498` 到 `models/utils.py:513`

这类错误通常说明：

```text
processor 计算的 placeholder 数量
  和
model.embed_multimodal() 实际输出的 feature token 数量
不一致。
```

排查时要同时看：

```text
ProcessingInfo / DummyInputsBuilder 的 feature token 估计
PromptReplacement / PromptInsertion 的 placeholder 展开
模型 vision/audio encoder 的输出 token 数
projector 是否改变 token 数
spatial merge / pixel shuffle / pruning 是否改变 token 数
```

### 7.3 `is_embed` 支持“只有部分 placeholder token 覆盖 embedding”

`PromptUpdateDetails.is_embed` 可以指定 full replacement 中哪些 token 真的由多模态 embedding 覆盖。

位置：`processing/processor.py:212` 到 `processing/processor.py:222`

runner 侧会读取 `pos_info.is_embed`：

```python
if (is_embed := pos_info.is_embed) is not None:
    is_embed = is_embed[start_idx:end_idx]
    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
else:
    mm_embeds_item = encoder_output[start_idx:end_idx]
```

位置：`gpu_model_runner.py:3153` 到 `gpu_model_runner.py:3158`

然后对 `is_mm_embed` 做 OR：

```python
is_mm_embed[...] |= is_embed
```

位置：`gpu_model_runner.py:3161` 到 `gpu_model_runner.py:3168`

这使得像 Qwen 系列那样包含 `<|vision_start|>` / `<|vision_end|>` 的结构可以只替换中间真正的视觉 pad token，而不是覆盖所有特殊 token。

---

## 8. 代表模型：LLaVA

LLaVA 是典型的 image-only 多模态 decoder 模型。

### 8.1 类声明和注册

```python
@MULTIMODAL_REGISTRY.register_processor(
    _build_llava_or_pixtral_hf_processor,
    info=_build_llava_or_pixtral_hf_info,
    dummy_inputs=LlavaDummyInputsBuilder,
)
class LlavaForConditionalGeneration(
    nn.Module,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
):
```

位置：`llava.py:499` 到 `llava.py:511`

这里同时完成：

```text
1. 注册 processor；
2. 标记 SupportsMultiModal；
3. 声明支持 LoRA / PP / Eagle 等其他能力。
```

### 8.2 模块组成

初始化中：

```text
_mark_tower_model("image"):
  vision_tower
  multi_modal_projector

_mark_language_model():
  language_model
```

位置：`llava.py:562` 到 `llava.py:583`

其中：

```text
vision_tower：CLIP / SigLIP / PixtralHF vision model
multi_modal_projector：把 vision hidden size 投影到 text hidden size
language_model：vLLM 注册的底层 CausalLM
```

### 8.3 输入解析和 encoder

LLaVA 支持两类 image 输入：

```text
pixel_values：需要 vision_tower + projector；
image_embeds：已经是 image embedding，直接进入后续处理。
```

解析位置：`llava.py:589` 到 `llava.py:621`

处理路径：

```text
pixel_values
  → _image_pixels_to_features()
  → vision_tower(...)
  → multi_modal_projector(...)
  → image_embeds

image_embeds
  → 直接返回 data
```

位置：`llava.py:623` 到 `llava.py:659`

`embed_multimodal()` 只是把这条路径包装成统一接口：

```python
def embed_multimodal(self, **kwargs):
    image_input = self._parse_and_validate_image_input(**kwargs)
    if image_input is None:
        return []
    return self._process_image_input(image_input)
```

位置：`llava.py:661` 到 `llava.py:666`

### 8.4 forward

LLaVA forward 不再自己合并 image embedding，因为 runner 已经通过 `embed_input_ids()` 合并好了。它只需要把 `inputs_embeds` 传给语言模型：

```python
hidden_states = self.language_model.model(
    input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
)
```

位置：`llava.py:716` 到 `llava.py:718`

---

## 9. 代表模型：Qwen2-VL

Qwen2-VL 是 image + video、多尺度视觉 token、M-RoPE 和 encoder cudagraph 更完整的例子。

### 9.1 支持 image / video placeholder

```python
if modality.startswith("image"):
    return "<|vision_start|><|image_pad|><|vision_end|>"
if modality.startswith("video"):
    return "<|vision_start|><|video_pad|><|vision_end|>"
```

位置：`qwen2_vl.py:1279` 到 `qwen2_vl.py:1286`

### 9.2 支持 encoder tensor parallel data mode

类上声明：

```python
supports_encoder_tp_data = True
```

位置：`qwen2_vl.py:1200`

`ModelConfig` 会在用户配置 `mm_encoder_tp_mode="data"` 时检查这个 flag。如果模型不支持，会 fallback 到 `weights`：

位置：`model.py:664` 到 `model.py:674`

### 9.3 image / video 输入解析

Qwen2-VL image 输入支持：

```text
pixel_values + image_grid_thw
image_embeds + image_grid_thw
```

位置：`qwen2_vl.py:1317` 到 `qwen2_vl.py:1340`

video 输入支持：

```text
pixel_values_videos + video_grid_thw
video_embeds + video_grid_thw
```

位置：`qwen2_vl.py:1341` 到 `qwen2_vl.py:1363`

`grid_thw` 决定每个 item 的输出 token 数。处理后会按每个 image / video item split：

```python
sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
return image_embeds.split(sizes)
```

位置：`qwen2_vl.py:1383` 到 `qwen2_vl.py:1386`

video 同理见 `qwen2_vl.py:1408` 到 `qwen2_vl.py:1411`。

### 9.4 `embed_multimodal()` 保持 modality 顺序

Qwen2-VL 会先按 kwargs 出现顺序解析 modalities：

```python
for input_key in kwargs:
    if input_key in ("pixel_values", "image_embeds") ...
    if input_key in ("pixel_values_videos", "video_embeds") ...
```

位置：`qwen2_vl.py:1413` 到 `qwen2_vl.py:1430`

然后按 `modalities` 的 key 顺序拼接 outputs：

```python
for modality in modalities:
    if modality == "images":
        multimodal_embeddings += tuple(image_embeddings)
    if modality == "videos":
        multimodal_embeddings += tuple(video_embeddings)
```

位置：`qwen2_vl.py:1432` 到 `qwen2_vl.py:1453`

这是为了满足 `SupportsMultiModal.embed_multimodal()` 的顺序约束：输出 embedding 顺序必须和 prompt 中对应媒体 item 出现顺序一致。

### 9.5 M-RoPE 位置

Qwen2-VL 会根据 image / video grid 计算 3 维位置：

```text
iter_mm_grid_thw(...)
  → offset, grid_t, grid_h, grid_w, t_factor
  → get_mrope_input_positions(...)
  → llm_positions: shape (3, seq_len)
  → mrope_position_delta
```

位置：`qwen2_vl.py:1202` 到 `qwen2_vl.py:1277`

这解释了为什么多模态接口不只关心 embedding：有些模型还需要 processor / feature spec 提供 grid 信息，以便 runner 或模型计算正确的 position ids。

### 9.6 Encoder CUDA Graph 可选接口

Qwen2-VL 还实现了 encoder cudagraph 相关方法：

```text
get_encoder_cudagraph_config()
get_input_modality(mm_kwargs)
get_max_frames_per_video()
get_encoder_cudagraph_budget_range(...)
get_encoder_cudagraph_item_specs(mm_kwargs)
select_encoder_cudagraph_items(mm_kwargs, indices)
prepare_encoder_cudagraph_capture_inputs(...)
```

位置：`qwen2_vl.py:1455` 开始。

这些不是所有多模态模型必需，但当模型希望 encoder 也被 cudagraph 捕获时，需要提供这些能力。

`GPUModelRunner._execute_mm_encoder()` 会优先尝试：

```python
if self.encoder_cudagraph_manager is not None
   and self.encoder_cudagraph_manager.supports_modality(modality):
    cudagraph_output = self.encoder_cudagraph_manager.execute(mm_kwargs_batch)

if cudagraph_output is not None:
    batch_outputs = cudagraph_output
else:
    batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
```

位置：`gpu_model_runner.py:3073` 到 `gpu_model_runner.py:3085`

---

## 10. 新增一个多模态模型时的接口清单

### 10.1 最小必需

```text
模型类：
  - 继承 nn.Module
  - 继承 SupportsMultiModal
  - 实现 / 继承 forward(..., input_ids, positions, inputs_embeds, ...)
  - 实现 compute_logits(hidden_states) 或对应 pooling 接口
  - 实现 load_weights(weights)

多模态接口：
  - get_placeholder_str(modality, i)
  - embed_multimodal(**kwargs) -> MultiModalEmbeddings
  - 能让默认 get_language_model() 找到底层 LLM，通常用 _mark_language_model(...)
  - 如果有 OOV 多模态 token，初始化时调用 configure_mm_token_handling(...)

processor 注册：
  - @MULTIMODAL_REGISTRY.register_processor(...)
  - ProcessingInfo：声明支持哪些 modality、每个媒体 item 有多少 feature tokens 等
  - MultiModalProcessor：把原始媒体转成 mm_kwargs，并插入 / 替换 prompt placeholders
  - DummyInputsBuilder：为 profiling 生成 dummy multimodal inputs
```

### 10.2 常见但非最小必需

```text
模块标记：
  - _mark_tower_model(vllm_config, modality)
  - _mark_composite_model(...)

LoRA / 模块映射：
  - get_mm_mapping()
  - get_num_mm_encoder_tokens(...)
  - get_num_mm_connector_tokens(...)

位置编码：
  - get_mrope_input_positions(...)
  - recompute_mrope_positions(...)  # SupportsMultiModalPruning

encoder cudagraph：
  - get_encoder_cudagraph_config()
  - get_encoder_cudagraph_item_specs(...)
  - select_encoder_cudagraph_items(...)
  - prepare_encoder_cudagraph_capture_inputs(...)

特殊执行：
  - supports_encoder_tp_data = True
  - supports_multimodal_raw_input_only = True
  - requires_raw_input_tokens = True
```

### 10.3 最容易出错的契约

```text
1. embed_multimodal() 返回顺序必须和 prompt 中媒体出现顺序一致。

2. embed_multimodal() 输出 token 数必须和 processor 插入的 embedding placeholder 数一致。

3. 如果 replacement 里有不该被 embedding 覆盖的特殊 token，必须正确设置 is_embed。

4. 如果多模态 token id 超出 vocab，必须调用 configure_mm_token_handling()，否则 embedding lookup 可能越界。

5. forward() 必须支持 inputs_embeds；多模态路径下 runner 默认会提前把 token ids 转成 embeddings。

6. ProcessingInfo / DummyInputsBuilder 的 token 估计必须和真实 encoder 输出一致，否则 profiling、调度预算、合并阶段都会出问题。

7. 如果模型有 image + video / audio + video 等多 modality，embed_multimodal() 不能随意重排输出。
```

---

## 11. 和旧接口命名的关系

早期文档或旧版本代码里常见这些说法：

```text
get_multimodal_embeddings
merge_multimodal_embeddings
get_input_embeddings
```

在当前代码中，对应关系更准确地写成：

```text
get_multimodal_embeddings
  → 当前模型侧统一接口是 SupportsMultiModal.embed_multimodal(**kwargs)

merge_multimodal_embeddings
  → 当前通用实现是 models.utils._merge_multimodal_embeddings(...)
  → 由 SupportsMultiModal.embed_input_ids(...) 调用

get_input_embeddings / get_input_embeddings(input_ids)
  → 很多底层语言模型仍有自己的 embedding 方法
  → 多模态统一入口更推荐看 SupportsMultiModal.embed_input_ids(...)
  → 它内部通过 get_language_model().embed_input_ids(...) 取文本 embedding
```

因此梳理当前 vLLM 多模态模型接口时，应以这些名字为准：

```text
SupportsMultiModal
MULTIMODAL_REGISTRY.register_processor
get_placeholder_str
embed_multimodal
embed_input_ids
get_language_model
_merge_multimodal_embeddings
_mark_language_model / _mark_tower_model
```

---

## 12. 总结

多模态模型类和普通语言模型类的差异不在 logits 侧，而在 forward 前的输入构造侧。

可以按三层理解：

```text
注册层：
  ModelRegistry 识别 SupportsMultiModal；
  ModelConfig 初始化 MultiModalConfig；
  MultiModalRegistry 绑定 processor。

处理层：
  processor 把原始媒体变成 mm_kwargs；
  processor 把 prompt 中的占位符扩展成准确数量的 placeholder token；
  PlaceholderRange 记录每个媒体 item 对应 token 区间。

执行层：
  GPUModelRunner 调 model.embed_multimodal() 跑 encoder；
  encoder_cache 按 mm_hash 缓存输出；
  _gather_mm_embeddings() 按本轮 chunk 取 embedding；
  model.embed_input_ids() 把文本 embedding 和多模态 embedding 合并；
  model.forward(..., inputs_embeds=...) 继续正常 decoder forward。
```

最终接口契约可以压缩成一句：

```text
多模态模型类必须让 vLLM 知道“哪里放媒体 token、如何把媒体转成 embedding、如何把 embedding 放回 decoder 输入序列”。
```
