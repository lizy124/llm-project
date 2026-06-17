# 08 多模态模型接口与典型实现

本篇梳理 vLLM 多模态模型需要实现的接口，以及典型模型族如何把 image/audio/video encoder 输出合并进语言模型。重点是理解 runtime 统一调用 `embed_multimodal()` 与 `embed_input_ids()`，而不同模型族在 tower、projector、placeholder 长度、位置编码上各不相同。

## 1. `SupportsMultiModal`

多模态模型核心接口在：`code/vllm/vllm/model_executor/models/vision.py:95`。

关键方法包括：

| 方法 | 作用 |
|---|---|
| `embed_multimodal(...)` | 把 image/audio/video/embedding 输入转成多模态 embeddings 或 encoder outputs。 |
| `get_language_model()` | 返回语言模型主体，供 embedding、LoRA、PP 等公共逻辑使用。 |
| `embed_input_ids(...)` | 把 token ids 转成 inputs_embeds，并把多模态 embeddings 覆写到 placeholder 位置。 |
| `get_num_mm_encoder_tokens(...)` | 返回 encoder 侧 token/embed 数。 |
| `get_num_mm_connector_tokens(...)` | 返回 connector/projector 后实际写回 LLM 的 token 数。 |

默认 `embed_input_ids()` 实现：`code/vllm/vllm/model_executor/models/vision.py:374`。

## 2. 公共 embedding 合并逻辑

合并函数：`code/vllm/vllm/model_executor/models/utils.py:479`。

主逻辑：

```text
inputs_embeds = language_model.get_input_embeddings(input_ids)
mm_embeds_flat = flatten(multimodal_embeddings)
mask = is_multimodal
inputs_embeds[mask] = mm_embeds_flat
```

这说明 decoder-only VLM 的主融合方式是“placeholder token embedding 覆写”，不是在 attention 层临时拼接 side input。

## 3. 常见扩展接口

### 3.1 `SupportsMRoPE`

位置：`code/vllm/vllm/model_executor/models/vision.py:1445`。

用于需要多维位置编码的模型，例如 Qwen2-VL。

### 3.2 `SupportsEncoderCudaGraph`

位置：`code/vllm/vllm/model_executor/models/vision.py:1544`。

用于多模态 encoder 独立做 cudagraph capture/replay。

### 3.3 `SupportsTranscription`

位置：`code/vllm/vllm/model_executor/models/vision.py:1074`。

用于 ASR / transcription 模型。

### 3.4 DP-sharded vision / MRoPE 辅助

相关接口：

- `code/vllm/vllm/model_executor/models/interfaces.py:33`
- `code/vllm/vllm/model_executor/models/interfaces.py:198`
- `code/vllm/vllm/model_executor/models/interfaces.py:281`
- `code/vllm/vllm/model_executor/models/interfaces.py:383`
- `code/vllm/vllm/model_executor/models/interfaces.py:573`

例如 `run_dp_sharded_mrope_vision_model(...)` 支持 Qwen2-VL 这类视觉 encoder 与 M-RoPE 强耦合的模型。

## 4. processor registry 与模型类关系

注册入口：`code/vllm/vllm/multimodal/registry.py:142`。

`@MULTIMODAL_REGISTRY.register_processor(...)` 会给模型类挂上 `_processor_factory`，其中包含：

- `info`；
- `dummy_inputs`；
- `processor`。

相关位置：

- registry 类：`code/vllm/vllm/multimodal/registry.py:98`
- 注册入口：`code/vllm/vllm/multimodal/registry.py:142`
- `_create_processing_info()`：`code/vllm/vllm/multimodal/registry.py:198`
- `create_processor()`：`code/vllm/vllm/multimodal/registry.py:211`

三件套分工：

| 组件 | 职责 |
|---|---|
| `ProcessingInfo` | 描述模型 HF config、HF processor、modality limits、token 规则。 |
| `DummyInputsBuilder` | 构造 profiling / budget 推导用假输入。 |
| `MultiModalProcessor` | 真正把用户输入转成 prompt tokens、mm kwargs、placeholder。 |

## 5. Transformers 通用多模态适配

文件：`code/vllm/vllm/model_executor/models/transformers/multimodal.py`。

关键位置：

- processing info：`code/vllm/vllm/model_executor/models/transformers/multimodal.py:59`
- processor：`code/vllm/vllm/model_executor/models/transformers/multimodal.py:263`
- mixin：`code/vllm/vllm/model_executor/models/transformers/multimodal.py:366`
- 其他桥接：`code/vllm/vllm/model_executor/models/transformers/multimodal.py:461`

这个通用适配层把 HF-style 模型的 `get_image_features(...)`、processor 输出、placeholder 信息与 vLLM 的 `embed_multimodal` 框架连接起来。

## 6. LLaVA：标准 vision tower + projector

关键位置：

- 模型结构：`code/vllm/vllm/model_executor/models/llava.py:504`
- 解析 image input：`code/vllm/vllm/model_executor/models/llava.py:643`
- 处理 image input：`code/vllm/vllm/model_executor/models/llava.py:661`
- forward/模型主链：`code/vllm/vllm/model_executor/models/llava.py:742`

典型流程：

```text
image input
  ↓
_parse_and_validate_image_input
  ↓
vision_tower
  ↓
multi_modal_projector
  ↓
image embeddings
  ↓
embed_input_ids 覆写 placeholder
```

LLaVA 是最标准的 decoder-only image 模型样本。

## 7. Qwen2-VL：image/video + grid_thw + M-RoPE

关键位置：

- vision 相关结构：`code/vllm/vllm/model_executor/models/qwen2_vl.py:1180`
- 模型主体：`code/vllm/vllm/model_executor/models/qwen2_vl.py:1240`
- embedding/visual 输出处理：`code/vllm/vllm/model_executor/models/qwen2_vl.py:1365`
- embed multimodal：`code/vllm/vllm/model_executor/models/qwen2_vl.py:1388`
- forward：`code/vllm/vllm/model_executor/models/qwen2_vl.py:1432`
- M-RoPE 支持：`code/vllm/vllm/model_executor/models/qwen2_vl.py:1740`

特点：

- 同时支持 image 和 video；
- 输入包含 `grid_thw`；
- patch 数会经过 merge size 缩减；
- 需要 M-RoPE 多维位置；
- 可能使用 DP-sharded vision；
- encoder token 数和 connector token 数不一定 1:1。

## 8. Qwen2-Audio：audio tower + effective length

关键位置：

- audio tower 结构：`code/vllm/vllm/model_executor/models/qwen2_audio.py:152`
- 音频处理：`code/vllm/vllm/model_executor/models/qwen2_audio.py:390`
- embeddings split：`code/vllm/vllm/model_executor/models/qwen2_audio.py:454`

流程：

```text
input_features + attention_mask
  ↓
audio_tower
  ↓
last_hidden_state
  ↓
multi_modal_projector
  ↓
feature_attention_mask 推导有效长度
  ↓
按 sample split
  ↓
embed_input_ids 合并
```

音频路径的关键是有效长度对齐，而不是固定 patch 数。

## 9. Whisper：encoder-decoder 例外路径

关键位置：

- 模型 forward：`code/vllm/vllm/model_executor/models/whisper.py:598`
- 模型类：`code/vllm/vllm/model_executor/models/whisper.py:798`
- `embed_multimodal()`：`code/vllm/vllm/model_executor/models/whisper.py:990`
- `embed_input_ids()`：`code/vllm/vllm/model_executor/models/whisper.py:998`

Whisper 的多模态结果不是覆写 decoder token embeddings，而是作为 encoder hidden states 通过 cross-attention 给 decoder 使用。

因此它是理解 vLLM 多模态非 decoder-only 路径的关键样本。

## 10. Pixtral

关键位置：

- 视觉结构：`code/vllm/vllm/model_executor/models/pixtral.py:215`
- 模型/adapter：`code/vllm/vllm/model_executor/models/pixtral.py:384`
- forward/embedding：`code/vllm/vllm/model_executor/models/pixtral.py:585`

特点：

- 图像 token 很大程度由 chat template 预插入；
- 不完全依赖普通 placeholder replacement；
- vision 侧可能有 patch merger/adapter；
- token 数换算更灵活。

## 11. Gemma3-MM

关键位置：

- projector / 视觉结构：`code/vllm/vllm/model_executor/models/gemma3_mm.py:160`
- 其他组件：`code/vllm/vllm/model_executor/models/gemma3_mm.py:262`
- 模型主体：`code/vllm/vllm/model_executor/models/gemma3_mm.py:469`
- 多模态处理：`code/vllm/vllm/model_executor/models/gemma3_mm.py:567`
- forward/合并：`code/vllm/vllm/model_executor/models/gemma3_mm.py:648`

特点：

- image-only；
- 支持 pan-and-scan / crops；
- 一张图可能展开成原图 + 多个 crop；
- projector 可能是 pooled projector；
- placeholder/newline token 有兼容逻辑。

## 12. 典型模型族对比

| 模型族 | 模态 | 架构 | 融合方式 | 特点 |
|---|---|---|---|---|
| LLaVA | image | decoder-only | placeholder embedding 覆写 | 标准 vision tower + projector。 |
| Qwen2-VL | image/video | decoder-only | placeholder embedding 覆写 | grid_thw、M-RoPE、video、merge size。 |
| Qwen2-Audio | audio | decoder-only | placeholder embedding 覆写 | audio mask 推导有效长度。 |
| Whisper | audio | encoder-decoder | cross-attention | 不走 decoder placeholder 覆写主路径。 |
| Pixtral | image | decoder-only | template/token 强相关 | chat template 预插图像 token。 |
| Gemma3-MM | image | decoder-only | placeholder embedding 覆写 | pan-and-scan、crop、pooled projector。 |

## 13. 一句话总结

vLLM 通过 `SupportsMultiModal` 把不同多模态模型统一成 `embed_multimodal` 和 `embed_input_ids` 两个关键接口：前者把媒体转成 embeddings，后者把 embeddings 覆写到 placeholder token 位置；不同模型族的复杂性主要体现在 processor 规则、encoder/tower 输出长度、projector/connector、位置编码和是否 encoder-decoder。
