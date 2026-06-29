# 10. 多模态模型架构如何接入？

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\multimodal\registry.py`
- `E:\lizy\code\vllm-project\vllm\vllm\multimodal\processing\processor.py`
- `E:\lizy\code\vllm-project\vllm\vllm\multimodal\inputs.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\core\sched\output.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\worker\gpu_model_runner.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\interfaces.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\utils.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\llava.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\qwen2_vl.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\qwen2_audio.py`

本问题关注：vision-language / audio-language / video-language 等多模态模型，如何在 vLLM 中把图像、视频、音频或预计算 embedding 接入语言模型 backbone；多模态 placeholder 如何生成；encoder 输出如何缓存；`GPUModelRunner` 如何把多模态 embedding 合入 `inputs_embeds`；模型类需要实现哪些接口；M-RoPE / XD-RoPE / load weights / LoRA / encoder cache 这些细节如何配合。

---

## 1. 一句话回答

vLLM 的多模态模型架构，本质上是把“非文本输入”先经过 processor 变成 placeholder token 和 `mm_features`，再由 `GPUModelRunner` 调用模型的 modality encoder / projector 得到 hidden-size 对齐的 multimodal embeddings，最后用 `embed_input_ids()` 把这些 embeddings 覆盖到 placeholder 对应位置，形成 `inputs_embeds` 送入语言模型 backbone。

主链路是：

```text
用户输入 text + image / video / audio
  → MultiModalRegistry 创建模型专属 processor
  → BaseMultiModalProcessor.apply()
  → HF processor 处理原始模态数据
  → prompt update / placeholder replacement
  → MultiModalInput(prompt_token_ids, mm_kwargs, mm_hashes, mm_placeholders)
  → Request.mm_features
  → SchedulerOutput.scheduled_encoder_inputs
  → GPUModelRunner._execute_mm_encoder()
  → model.embed_multimodal(**mm_kwargs)
  → encoder_cache[mm_hash] = multimodal embeddings
  → GPUModelRunner._gather_mm_embeddings()
  → model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
  → inputs_embeds
  → model.forward(inputs_embeds=...)
  → language_model.model(...)
  → compute_logits / pooler
```

所以多模态模型接入不是单点逻辑，而是四层协议：

```text
processor 层：决定 placeholder 和 mm_kwargs；
Scheduler 层：决定本轮哪些 encoder input 要执行；
ModelRunner 层：执行 encoder、缓存 embedding、合并 inputs_embeds；
Model class 层：定义 tower / projector / language_model / forward / load_weights。
```

---

## 2. 最小心智模型

多模态模型可以先压缩成这个结构：

```text
text tokens
  → token embedding

image / video / audio / prompt_embeds
  → modality encoder 或 passthrough
  → projector / merger
  → hidden-size 对齐的 multimodal embeddings

placeholder mask
  → 把 multimodal embeddings scatter 到 token embeddings 对应位置
  → inputs_embeds
  → language model backbone
  → hidden states
  → logits / pooling output
```

关键点：

```text
1. 多模态输入最终不是作为 input_ids 直接进入 LLM；
2. 语言模型看到的是已经融合好的 inputs_embeds；
3. placeholder token 的数量必须和 encoder/projector 输出 token 数一致；
4. KV cache / attention metadata 使用替换后的 token 序列长度；
5. 图像、视频、音频差异主要在 processor、tower、projector 和 position 计算中。
```

---

## 3. 多模态架构的典型组成

以大多数 VLM / ALM 为例，模型类通常包含三块。

```text
modality tower：
  CLIP / SigLIP / ViT / Qwen2VisionTransformer / audio encoder / video encoder

connector / projector：
  MLP projector / linear projector / merger / resampler

language model：
  Qwen / Llama / Mistral / Gemma / Phi 等 CausalLM backbone
```

在模型类里经常能看到：

```text
self.vision_tower / self.visual / self.audio_tower
self.multi_modal_projector / visual.merger
self.language_model
```

例如 LLaVA：

```text
vision_tower(pixel_values)
  → image_features
  → multi_modal_projector(image_features)
  → image_embeds
  → language_model.model(inputs_embeds=...)
```

对应源码：`llava.py:562` 到 `llava.py:576`、`llava.py:643` 到 `llava.py:666`、`llava.py:668` 到 `llava.py:720`

例如 Qwen2-VL：

```text
visual(pixel_values, grid_thw)
  → image/video embeddings
  → split by grid_thw
  → language_model.model(inputs_embeds=..., positions=mrope_positions)
```

对应源码：`qwen2_vl.py:1298` 到 `qwen2_vl.py:1311`、`qwen2_vl.py:1365` 到 `qwen2_vl.py:1453`、`qwen2_vl.py:1687` 到 `qwen2_vl.py:1718`

例如 Qwen2-Audio：

```text
audio_tower(input_features, attention_mask)
  → selected_audio_feature
  → multi_modal_projector
  → split by audio_output_lengths
  → language_model.model(inputs_embeds=...)
```

对应源码：`qwen2_audio.py:348` 到 `qwen2_audio.py:360`、`qwen2_audio.py:390` 到 `qwen2_audio.py:459`、`qwen2_audio.py:461` 到 `qwen2_audio.py:475`

---

## 4. 模型类必须实现的接口

多模态模型通常继承或满足 `SupportsMultiModal`。

源码：`interfaces.py:95` 到 `interfaces.py:410`

核心接口是：

```text
supports_multimodal = True
get_placeholder_str(modality, i)
embed_multimodal(**kwargs)
get_language_model()
embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
```

### 4.1 `get_placeholder_str()`

`get_placeholder_str()` 给 processor / chat template 一个“用户可见”的占位符形式。

LLaVA：

```python
def get_placeholder_str(cls, modality: str, i: int) -> str | None:
    if modality.startswith("image"):
        return "<image>"
```

位置：`llava.py:527` 到 `llava.py:532`

Qwen2-VL：

```python
if modality.startswith("image"):
    return "<|vision_start|><|image_pad|><|vision_end|>"
if modality.startswith("video"):
    return "<|vision_start|><|video_pad|><|vision_end|>"
```

位置：`qwen2_vl.py:1279` 到 `qwen2_vl.py:1286`

Qwen2-Audio：

```python
if modality.startswith("audio"):
    return f"Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>"
```

位置：`qwen2_audio.py:331` 到 `qwen2_audio.py:337`

注意：这个字符串不一定等于最终 prompt 里的所有 feature placeholder。最终会由 processor 根据 encoder 输出长度插入或替换成足够多的 placeholder token。

### 4.2 `embed_multimodal()`

`embed_multimodal()` 是 ModelRunner 调用多模态 encoder 的统一入口。

语义是：

```text
输入：HF processor 产出的 mm_kwargs；
输出：MultiModalEmbeddings，通常是 tuple/list/tensor，每个 item 对应一个 image/video/audio；
要求：输出顺序必须和 prompt 中多模态 item 出现顺序一致。
```

接口定义位置：`interfaces.py:155` 到 `interfaces.py:165`

LLaVA 的实现：

```text
_parse_and_validate_image_input()
  → _process_image_input()
  → vision_tower
  → multi_modal_projector
  → image embeddings
```

位置：`llava.py:589` 到 `llava.py:666`

Qwen2-VL 的实现：

```text
_parse_and_validate_multimodal_inputs()
  → _process_image_input() / _process_video_input()
  → visual(...)
  → split by grid_thw
```

位置：`qwen2_vl.py:1317` 到 `qwen2_vl.py:1453`

Qwen2-Audio 的实现：

```text
_parse_and_validate_audio_input()
  → _process_audio_input()
  → audio_tower
  → multi_modal_projector
  → split by audio_output_lengths
```

位置：`qwen2_audio.py:366` 到 `qwen2_audio.py:459`

### 4.3 `embed_input_ids()`

`embed_input_ids()` 是把文本 embedding 和多模态 embedding 合并的关键。

默认实现位置：`interfaces.py:375` 到 `interfaces.py:410`

流程是：

```text
1. 先调用 language_model.embed_input_ids(input_ids) 得到文本 embeddings；
2. 如果有 OOV multimodal token，先把对应 input_ids mask 成 0，避免 embedding lookup 越界；
3. 调用 _merge_multimodal_embeddings()；
4. 用 is_multimodal mask 把多模态 embeddings 覆盖到 placeholder 位置。
```

真正 scatter 的函数是 `_merge_multimodal_embeddings()`。

位置：`utils.py:479` 到 `utils.py:515`

关键逻辑：

```python
inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
```

如果数量对不上，会报：

```text
Attempted to assign ... multimodal tokens to ... placeholders
```

这就是多模态接入最常见的问题之一：

```text
placeholder token 数量 != encoder/projector 输出 embedding 数量。
```

### 4.4 `_mark_language_model()` 和 `_mark_tower_model()`

多模态模型构造时经常使用：

```text
with self._mark_tower_model(...):
  创建 vision/audio tower 和 connector

with self._mark_language_model(...):
  创建 language_model
```

位置：`interfaces.py:216` 到 `interfaces.py:293`

作用是：

```text
1. 标记哪些子模块是 language model；
2. 标记哪些子模块是 tower model；
3. 支持 --mm-encoder-only；
4. 支持 limit-mm-per-prompt=0 时跳过 tower 初始化；
5. 给权重加载、LoRA、模块分组提供结构信息。
```

---

## 5. Processor 层如何生成 placeholder 和 mm_features

多模态输入进入模型前，先经过 `MultiModalRegistry` 和模型注册的 processor。

注册入口：`registry.py:142` 到 `registry.py:174`

创建 processor：`registry.py:211` 到 `registry.py:230`

判断模型是否需要多模态基础设施：`registry.py:103` 到 `registry.py:140`

### 5.1 Registry 做什么

`MultiModalRegistry` 负责：

```text
1. 根据 model_config 找到模型类；
2. 找到模型类上注册的 _processor_factory；
3. 创建 ProcessingInfo / DummyInputsBuilder / BaseMultiModalProcessor；
4. 判断是否支持 multimodal inputs；
5. 创建 processor cache / receiver cache。
```

如果模型类没有注册 processor，vLLM 会把它当成 text-only 路径或报错。

### 5.2 BaseMultiModalProcessor.apply() 主流程

入口：`processor.py:1663` 到 `processor.py:1707`

代码注释已经概括了三步：

```text
1. 调用 HF Processor 处理 prompt text 和多模态数据；
2. 找到或插入 placeholder tokens；
3. 从 token ids 中提取 placeholder 位置信息。
```

更详细的链路是：

```text
BaseMultiModalProcessor.apply()
  → _cached_apply_hf_processor()
  → _apply_hf_processor_main()
  → MultiModalKwargsItems.from_hf_inputs()
  → _get_mm_prompt_updates()
  → _maybe_apply_prompt_updates()
  → _apply_prompt_updates() 或 _find_mm_placeholders()
  → mm_input(prompt_token_ids, mm_kwargs, mm_hashes, mm_placeholders)
```

位置：`processor.py:1398` 到 `processor.py:1439`、`processor.py:1528` 到 `processor.py:1579`、`processor.py:1636` 到 `processor.py:1661`

### 5.3 PromptUpdate 决定怎么改 prompt

抽象类：`processor.py:297` 到 `processor.py:350`

两种更新模式：

```text
PromptInsertion：把 placeholder 插到指定位置后面；
PromptReplacement：把用户 prompt 里的占位符替换成 feature placeholders。
```

位置：`processor.py:353` 到 `processor.py:420`、`processor.py:422` 到 `processor.py:496`

例如用户只写一个 `<image>`，但视觉 encoder 输出 576 个 image tokens，则 processor 会把它替换成足够多的 `<image>` feature placeholders。这样：

```text
prompt_token_ids 长度 = 文本 token 数 + 多模态 embedding token 数
attention / KV cache / positions 才能提前按真实长度分配。
```

### 5.4 PlaceholderRange 记录什么

`PlaceholderRange` 定义位置：`inputs.py:118` 到 `inputs.py:216`

它记录：

```text
offset：placeholder 在 prompt token 序列中的起点；
length：placeholder 覆盖的 token 数；
is_embed：可选 bool mask，表示其中哪些位置要被 embedding 覆盖。
```

`is_embed` 很重要，因为有些模型的 placeholder 包含 BOS/EOS/包裹 token：

```text
<|vision_start|> <|image_pad|> ... <|vision_end|>
```

其中可能只有 `<|image_pad|>` 对应视觉 embedding，start/end 仍然作为普通文本 token 或特殊 token 处理。

相关方法：

```text
get_num_embeds()
get_embeds_indices_in_range(start_idx, end_idx)
extract_embeds_range()
```

位置：`inputs.py:152` 到 `inputs.py:202`

---

## 6. SchedulerOutput 如何携带多模态信息

进入 V1 执行层后，多模态信息主要在 `SchedulerOutput` 里传给 Worker。

定义位置：`output.py:180` 到 `output.py:215`

和多模态直接相关的字段：

```text
scheduled_new_reqs[].mm_features
scheduled_encoder_inputs
free_encoder_mm_hashes
```

### 6.1 `NewRequestData.mm_features`

`NewRequestData` 定义位置：`output.py:30` 到 `output.py:65`

其中：

```text
prompt_token_ids：已经包含 placeholder 的 token ids；
mm_features：每个多模态 item 的 modality、identifier、mm_position、data；
prompt_embeds：可选的预计算 prompt embeddings；
prompt_is_token_ids：混合 token ids / embeddings 时的标记。
```

`GPUModelRunner._update_states()` 会把 `new_req_data.mm_features` 放进 `CachedRequestState`。

位置：`gpu_model_runner.py:1224` 到 `gpu_model_runner.py:1237`

### 6.2 `scheduled_encoder_inputs`

字段注释：`output.py:201` 到 `output.py:204`

含义是：

```text
req_id -> encoder input indices that need processing
```

例如：

```text
{"req-A": [0, 1]}
```

表示这个请求的第 0 个和第 1 个多模态 item 需要在本轮跑 encoder。

为什么不是每一步都跑？

```text
1. decode 阶段通常不需要再次处理图像/音频；
2. encoder 输出可以按 mm_hash 缓存在 GPU 侧；
3. chunked prefill 时，同一个 encoder 输出可能跨多个 prefill chunk 被消费。
```

### 6.3 `free_encoder_mm_hashes`

字段注释：`output.py:213` 到 `output.py:215`

`GPUModelRunner._update_states()` 会释放这些 encoder cache：

```python
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

位置：`gpu_model_runner.py:1158` 到 `gpu_model_runner.py:1160`

这说明多模态 encoder cache 的生命周期由 Scheduler 决定，Worker 只按指令释放。

---

## 7. GPUModelRunner 中的多模态主链路

`GPUModelRunner.execute_model()` 中，多模态真正进入模型的位置在 `_preprocess()`。

主调用位置：`gpu_model_runner.py:4274` 到 `gpu_model_runner.py:4283`

```text
execute_model()
  → _update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
  → _model_forward()
```

多模态相关细节集中在：

```text
_update_states()
_batch_mm_inputs_from_scheduler()
_execute_mm_encoder()
_gather_mm_embeddings()
_preprocess()
```

---

## 8. `_update_states()` 如何保存 mm_features

入口：`gpu_model_runner.py:1127`

新请求进入 Worker 时，会创建 `CachedRequestState`：

```text
req_id
prompt_token_ids
prompt_embeds
prompt_is_token_ids
mm_features
sampling_params / pooling_params
block_ids
num_computed_tokens
lora_request
```

位置：`gpu_model_runner.py:1224` 到 `gpu_model_runner.py:1237`

同时，如果模型使用 M-RoPE / XD-RoPE，会在新请求阶段预计算 positions：

```text
uses_mrope → _init_mrope_positions(req_state)
uses_xdrope_dim > 0 → _init_xdrope_positions(req_state)
```

位置：`gpu_model_runner.py:1248` 到 `gpu_model_runner.py:1254`

这一步说明：

```text
多模态位置信息不是 forward 时临时猜的，而是在请求进入 batch 状态时就基于 prompt_token_ids + mm_features 计算。
```

---

## 9. `_execute_mm_encoder()` 如何运行 tower / projector

入口：`gpu_model_runner.py:2892`

### 9.1 先从 SchedulerOutput 取本轮需要跑的 encoder input

```text
_batch_mm_inputs_from_scheduler()
  → 遍历 scheduler_output.scheduled_encoder_inputs
  → 根据 req_id 找到 req_state.mm_features[mm_input_id]
  → 收集 mm_hashes / mm_kwargs / mm_lora_refs
```

位置：`gpu_model_runner.py:2849` 到 `gpu_model_runner.py:2890`

这一步只处理本轮 scheduler 认为需要 encoder 的多模态 item。

### 9.2 `prompt_embeds` 是 passthrough modality

`prompt_embeds` 不需要 encoder，因为它已经在模型 embedding 空间中。

处理逻辑：

```text
如果 modality == "prompt_embeds"：
  直接把 tensor 写入 encoder_cache[mm_hash]
  从待编码列表中过滤掉
```

位置：`gpu_model_runner.py:2902` 到 `gpu_model_runner.py:2927`

这说明 vLLM 把“预计算 embedding”也统一成 encoder_cache + gather 的路径。

### 9.3 按 modality 分组 batch

核心逻辑：

```text
for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(...):
    batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
    sanity_check_mm_encoder_outputs(...)
    encoder_outputs.extend(batch_outputs)
```

位置：`gpu_model_runner.py:3013` 到 `gpu_model_runner.py:3091`

这一步调用的就是模型类实现的 `embed_multimodal()`。

### 9.4 encoder 输出写入 GPU 侧 cache

```python
for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
```

位置：`gpu_model_runner.py:3093` 到 `gpu_model_runner.py:3099`

cache key 是 `mm_feature.identifier`，通常来自多模态输入 hash。

### 9.5 LoRA / Encoder CUDA Graph / 顺序视频编码

`_execute_mm_encoder()` 还处理三类工程细节：

```text
1. tower / connector LoRA mapping；
2. encoder_cudagraph_manager；
3. multimodal pruning 或 sequential video encoding 时的视频逐个编码。
```

LoRA mapping 位置：`gpu_model_runner.py:2944` 到 `gpu_model_runner.py:3011`

Encoder CUDA Graph 位置：`gpu_model_runner.py:3074` 到 `gpu_model_runner.py:3087`

逐个视频编码位置：`gpu_model_runner.py:3033` 到 `gpu_model_runner.py:3062`

---

## 10. `_gather_mm_embeddings()` 如何把 encoder cache 切回本轮 token window

入口：`gpu_model_runner.py:3101`

它解决的问题是：

```text
encoder 输出可能是整张图 / 整段视频 / 整段音频的 embedding；
但本轮 prefill 可能只调度了 prompt 的一段 token window；
所以需要只取本轮 scheduled token 范围内对应的 multimodal embeddings。
```

主逻辑：

```text
for req_id in input_batch.req_ids:
  num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
  num_computed_tokens = req_state.num_computed_tokens
  找出当前 window 内的 mm_features
  根据 PlaceholderRange 计算 start_idx / end_idx
  从 encoder_cache[mm_hash] 取对应切片
  标记 is_mm_embed mask
```

位置：`gpu_model_runner.py:3120` 到 `gpu_model_runner.py:3173`

关键对象：

```text
mm_feature.mm_position.offset
mm_feature.mm_position.length
mm_feature.mm_position.is_embed
mm_feature.identifier
```

### 10.1 chunked prefill 为什么需要切片

如果一个图像 placeholder 覆盖 576 个 token，但本轮只执行其中一部分：

```text
num_computed_tokens = 128
num_scheduled_tokens = 64
```

那么 `_gather_mm_embeddings()` 只取这 64 个 token window 里对应的视觉 embedding。

这让多模态模型可以和 chunked prefill、KV cache、prefix cache 共存。

### 10.2 `is_mm_embed` 是最终 scatter mask

`is_mm_embed` 是 CPU pinned bool tensor，长度等于本轮 scheduled tokens。

```text
True  的位置：用 multimodal embeddings 覆盖；
False 的位置：保留普通 token embedding。
```

返回值：

```text
(mm_embeds, is_mm_embed)
```

位置：`gpu_model_runner.py:3105` 到 `gpu_model_runner.py:3200`

---

## 11. `_preprocess()` 如何合并到 inputs_embeds

入口：`gpu_model_runner.py:3430`

多模态路径的判断条件：

```text
self.supports_mm_inputs
and is_first_rank
and not is_encoder_decoder
```

位置：`gpu_model_runner.py:3451`

### 11.1 decoder-only 多模态模型

主流程：

```text
1. _execute_mm_encoder(scheduler_output)
2. _gather_mm_embeddings(scheduler_output)
3. model.embed_input_ids(input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed)
4. copy 到 self.inputs_embeds.gpu
5. input_ids, inputs_embeds = _prepare_mm_inputs(num_input_tokens)
6. model.forward(input_ids=input_ids or None, inputs_embeds=inputs_embeds, positions=...)
```

位置：`gpu_model_runner.py:3451` 到 `gpu_model_runner.py:3503`

这里有一个重要设计：

```text
多模态模型统一使用 inputs_embeds 路径，即使文本 token 也先变成 embedding。
```

对应注释位置：`gpu_model_runner.py:3460` 到 `gpu_model_runner.py:3462`

原因是：

```text
同一个 batch 里既有普通 text token embedding，也有 soft token / vision embedding；
用 inputs_embeds 可以把它们统一成一种输入形态。
```

### 11.2 raw input tokens 例外

`_prepare_mm_inputs()`：

```python
if self.model.requires_raw_input_tokens:
    input_ids = self.input_ids.gpu[:num_tokens]
else:
    input_ids = None

inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
```

位置：`gpu_model_runner.py:3419` 到 `gpu_model_runner.py:3428`

大多数多模态模型只传 `inputs_embeds`，但少数模型可能还需要 raw `input_ids`。

### 11.3 encoder-decoder 多模态模型

如果是 encoder-decoder 模型，多模态输入不是替换 decoder prompt，而是进入 encoder outputs：

```text
if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
    encoder_outputs = self._execute_mm_encoder(scheduler_output)
    model_kwargs.update({"encoder_outputs": encoder_outputs})
```

位置：`gpu_model_runner.py:3555` 到 `gpu_model_runner.py:3562`

这类模型的心智模型是：

```text
encoder 处理多模态输入；
decoder 根据 encoder_outputs 生成文本。
```

---

## 12. M-RoPE / XD-RoPE positions 如何接入

多模态模型不仅要合并 embeddings，还要处理 position ids。

### 12.1 GPUModelRunner 的 position buffer

初始化位置：`gpu_model_runner.py:768` 到 `gpu_model_runner.py:790`

```text
uses_mrope：创建 shape=(3, max_num_tokens + 1) 的 mrope_positions；
uses_xdrope_dim > 0：创建 shape=(xdrope_dim, max_num_tokens + 1) 的 xdrope_positions。
```

`_get_positions()` 会根据模型类型返回普通 positions、M-RoPE positions 或 XD-RoPE positions。

位置：`gpu_model_runner.py:982` 到 `gpu_model_runner.py:994`

### 12.2 新请求阶段预计算 prompt positions

M-RoPE：`gpu_model_runner.py:1590` 到 `gpu_model_runner.py:1620`

```text
req_state.mrope_positions, req_state.mrope_position_delta =
  model.get_mrope_input_positions(input_tokens, mrope_features)
```

XD-RoPE：`gpu_model_runner.py:1622` 到 `gpu_model_runner.py:1633`

```text
req_state.xdrope_positions =
  model.get_xdrope_input_positions(prompt_token_ids, mm_features)
```

### 12.3 每轮执行阶段切出 scheduled positions

M-RoPE：`gpu_model_runner.py:2654` 到 `gpu_model_runner.py:2701`

XD-RoPE：`gpu_model_runner.py:2703` 到 `gpu_model_runner.py:2745`

逻辑是：

```text
prefill prompt 部分：从 req_state 预计算 positions 中切片；
decode completion 部分：根据 context_len 在线生成后续 positions。
```

### 12.4 Qwen2-VL 的 M-RoPE 实现

Qwen2-VL 实现了 `SupportsMRoPE.get_mrope_input_positions()`。

位置：`qwen2_vl.py:1240` 到 `qwen2_vl.py:1277`

它会遍历 image / video 的 grid 信息：

```text
offset
llm_grid_t
llm_grid_h
llm_grid_w
t_factor
```

然后构造 3D position ids：

```text
text：三个维度相同；
image/video：按 temporal / height / width grid 展开；
video：可根据 second_per_grid_ts 和 tokens_per_second 调整时间维。
```

所以 Qwen2-VL forward 看到的 `positions` 不是一维，而是：

```text
(3, seq_len)
```

对应 forward 注释：`qwen2_vl.py:1695` 到 `qwen2_vl.py:1707`

---

## 13. 几个典型模型族对比

### 13.1 LLaVA：CLIP/SigLIP/Pixtral tower + MLP projector

结构：

```text
<image> placeholders
  → vision_tower(pixel_values)
  → multi_modal_projector
  → image_embeds
  → embed_input_ids scatter
  → language_model.model(inputs_embeds=...)
```

关键源码：

```text
LlavaForConditionalGeneration
  → _mark_tower_model("image")
  → vision_tower
  → multi_modal_projector
  → _process_image_input()
  → embed_multimodal()
  → forward()
```

位置：`llava.py:504` 到 `llava.py:756`

LLaVA 的 forward 注释特别说明：

```text
input_ids 已经提前插入足够多的 image placeholder token，
这样 positions 和 attention metadata 与视觉 embedding token 数一致。
```

位置：`llava.py:676` 到 `llava.py:701`

### 13.2 Qwen2-VL：视觉 transformer + merger + M-RoPE

结构：

```text
<|vision_start|><|image_pad|><|vision_end|>
  → visual(pixel_values, grid_thw)
  → split by grid_thw / spatial_merge_size
  → M-RoPE 3D positions
  → language_model.model(inputs_embeds=..., positions=(3, seq_len))
```

关键源码：

```text
Qwen2VLForConditionalGeneration
  → visual = Qwen2VisionTransformer
  → language_model = Qwen2ForCausalLM
  → _process_image_input()
  → _process_video_input()
  → get_mrope_input_positions()
  → embed_multimodal()
```

位置：`qwen2_vl.py:1180` 到 `qwen2_vl.py:1738`

Qwen2-VL 同时支持：

```text
image
video
image_embeds
video_embeds
mm_encoder_tp_mode="data"
encoder CUDA graph
M-RoPE
```

### 13.3 Qwen2-Audio：audio encoder + linear projector

结构：

```text
<|audio_bos|><|AUDIO|><|audio_eos|>
  → audio_tower(input_features, attention_mask)
  → multi_modal_projector
  → 根据 audio_output_lengths mask / split
  → language_model.model(inputs_embeds=...)
```

关键源码：

```text
Qwen2AudioForConditionalGeneration
  → audio_tower
  → multi_modal_projector
  → _process_audio_input()
  → embed_multimodal()
  → forward()
```

位置：`qwen2_audio.py:331` 到 `qwen2_audio.py:485`

音频模型常见特点：

```text
1. 原始输入通常先被 HF processor 转成 input_features；
2. 有 feature_attention_mask；
3. encoder 输出长度由音频长度决定；
4. projector 后要按 audio_output_lengths 拆回每个 audio item。
```

---

## 14. load_weights 和名称映射差异

多模态模型通常需要加载三类权重：

```text
language_model.*
vision_tower / visual / audio_tower.*
multi_modal_projector / merger / connector.*
```

不同 HF checkpoint 的命名前缀经常不一致，所以模型类常用 `WeightsMapper` 或自定义 `load_weights()`。

### 14.1 LLaVA

映射：

```text
model.language_model.       → language_model.model.
model.vision_tower.         → vision_tower.
model.multi_modal_projector.→ multi_modal_projector.
lm_head.                    → language_model.lm_head.
```

位置：`llava.py:517` 到 `llava.py:525`

加载：

```python
loader = AutoWeightsLoader(self)
return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

位置：`llava.py:728` 到 `llava.py:730`

### 14.2 Qwen2-VL

映射：

```text
model.language_model. → language_model.model.
model.visual.         → visual.
lm_head.              → language_model.lm_head.
model.                → language_model.model.
```

位置：`qwen2_vl.py:1188` 到 `qwen2_vl.py:1198`

加载：`qwen2_vl.py:1726` 到 `qwen2_vl.py:1728`

### 14.3 Qwen2-Audio

Qwen2-Audio 使用 `AutoWeightsLoader(self)` 直接加载。

位置：`qwen2_audio.py:483` 到 `qwen2_audio.py:485`

### 14.4 tower / connector / language model 映射

一些多模态模型还提供：

```python
def get_mm_mapping(self) -> MultiModelKeys:
```

用于说明哪些模块属于：

```text
language_model
connector
tower_model
```

LLaVA：`llava.py:732` 到 `llava.py:740`

Qwen2-VL：`qwen2_vl.py:1730` 到 `qwen2_vl.py:1738`

这个映射对多模态 LoRA、模块分组、权重管理很重要。

---

## 15. 多模态 LoRA 如何挂接

多模态 LoRA 不只可能作用在语言模型上，还可能作用在：

```text
tower model：vision/audio encoder；
connector：projector / merger；
language model：LLM backbone。
```

`GPUModelRunner._execute_mm_encoder()` 中，如果 LoRA manager 支持 tower / connector LoRA，会构建两类 mapping。

位置：`gpu_model_runner.py:2944` 到 `gpu_model_runner.py:3011`

关键逻辑：

```text
1. 根据 mm_lora_refs 找到每个多模态 item 属于哪个请求；
2. 从 input_batch.request_lora_mapping 获取 lora_id；
3. 用 get_num_mm_encoder_tokens() 计算 tower token 数；
4. 如果有 connector，再用 get_num_mm_connector_tokens() 计算 connector token 数；
5. 分别设置 LoRAMappingType.TOWER 和 LoRAMappingType.CONNECTOR。
```

因此支持多模态 LoRA 的模型通常需要实现：

```text
get_num_mm_encoder_tokens(num_image_tokens)
get_num_mm_connector_tokens(num_vision_tokens)
get_mm_mapping()
```

LLaVA 示例：`llava.py:742` 到 `llava.py:756`

---

## 16. Encoder cache 与 processor cache 的区别

多模态里有两类 cache，容易混淆。

### 16.1 processor cache

位置：`registry.py:268` 到 `registry.py:347`

作用：

```text
缓存 HF processor 处理后的 mm_kwargs / prompt updates，
减少 API/engine 侧重复图像预处理、token 替换、hash 计算等开销。
```

它发生在模型 encoder 前。

### 16.2 encoder cache

位置：`gpu_model_runner.py:535` 到 `gpu_model_runner.py:536`

```python
self.encoder_cache: dict[str, torch.Tensor] = {}
```

作用：

```text
缓存 GPU 侧 modality encoder / projector 的输出 embedding，
避免 decode 或 chunked prefill 后续 step 重复跑视觉/音频 encoder。
```

写入位置：`gpu_model_runner.py:3093` 到 `gpu_model_runner.py:3099`

读取位置：`gpu_model_runner.py:3153` 到 `gpu_model_runner.py:3162`

释放位置：`gpu_model_runner.py:1158` 到 `gpu_model_runner.py:1160`

两者区别：

```text
processor cache：缓存输入预处理结果，通常在 CPU / 进程间层；
encoder cache：缓存模型 encoder 输出，位于 GPU ModelRunner 侧。
```

---

## 17. 多模态和 Attention / KV Cache 的关系

多模态 embedding 合入 `inputs_embeds` 后，对 attention / KV cache 来说，它们就是普通序列 token。

### 17.1 placeholder 先扩展，KV cache 才能对齐

如果图像 encoder 输出 576 个 embedding，那么 prompt 中必须有 576 个可替换位置。这样：

```text
positions 长度一致；
slot_mapping 长度一致；
attention metadata 长度一致；
KV cache 写入位置一致。
```

LLaVA forward 注释强调了这一点。

位置：`llava.py:676` 到 `llava.py:701`

### 17.2 ModelRunner 先 prepare inputs，再 gather mm embeddings

`_preprocess()` 注释说明：

```text
_prepare_inputs may reorder the batch,
so we must gather multi modal outputs after that to ensure the correct order
```

位置：`gpu_model_runner.py:3447` 到 `gpu_model_runner.py:3448`

也就是说：

```text
batch 可能被 attention backend / persistent batch 优化重排，
所以多模态 embedding 必须在最终 batch 顺序确定后再 gather。
```

### 17.3 多模态不改变 decode token 语义

prefill 阶段：

```text
text token + multimodal placeholder embeddings 一起写入 KV cache。
```

decode 阶段：

```text
通常只输入上一步生成的新 token，不再重复输入图像/音频 embedding。
```

这也是为什么需要 encoder cache 和 scheduled_encoder_inputs。

---

## 18. 多模态 pruning / dynamic embedding 数

有些模型会在 encoder 后裁剪视觉 token，导致实际 embedding 数和原始 placeholder 数发生变化。

vLLM 用 `SupportsMultiModalPruning` 表示这类能力。

定义位置：`interfaces.py:413` 到 `interfaces.py:448`

它需要实现：

```text
recompute_mrope_positions(input_ids, multimodal_embeddings, mrope_positions, num_computed_tokens)
```

`_gather_mm_embeddings()` 中，如果开启 pruning 且使用 M-RoPE，会调用：

```text
self.model.recompute_mrope_positions(...)
```

位置：`gpu_model_runner.py:3175` 到 `gpu_model_runner.py:3188`

含义是：

```text
视觉 token 被 prune 后，原先按未裁剪序列计算的 M-RoPE positions 不再准确，
需要根据裁剪后的 multimodal embeddings 重算。
```

---

## 19. 多模态和 Pipeline Parallel 的边界

在 pipeline parallel 下，多模态 encoder 和 embedding 合并只发生在 first PP rank。

`_preprocess()` 中：

```text
if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
  执行多模态 encoder / gather / embed_input_ids
```

位置：`gpu_model_runner.py:3451` 到 `gpu_model_runner.py:3503`

非 first PP rank：

```text
不会重新处理 input_ids / inputs_embeds，
而是接收上一个 PP stage 发来的 IntermediateTensors。
```

位置：`gpu_model_runner.py:3547` 到 `gpu_model_runner.py:3553`

模型类 forward 也会处理这个情况：

```text
if intermediate_tensors is not None:
    inputs_embeds = None
```

LLaVA：`llava.py:713` 到 `llava.py:718`

Qwen2-VL：`qwen2_vl.py:1709` 到 `qwen2_vl.py:1717`

Qwen2-Audio：`qwen2_audio.py:469` 到 `qwen2_audio.py:474`

---

## 20. 开发一个新多模态模型要实现什么

如果要在 vLLM 中新增一个 decoder-only 多模态模型，一般需要补齐以下内容。

### 20.1 模型类

```text
1. 继承 nn.Module 和 SupportsMultiModal；
2. 构造 tower / connector / language_model；
3. 用 _mark_tower_model() 标记多模态 tower；
4. 用 _mark_language_model() 标记 LLM；
5. 实现 get_placeholder_str()；
6. 实现 embed_multimodal(**kwargs)；
7. forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs)；
8. compute_logits(hidden_states)；
9. load_weights() 和必要的 WeightsMapper；
10. 如支持 LoRA，提供 get_mm_mapping() 和 token count helper。
```

### 20.2 Processor

```text
1. 注册 MULTIMODAL_REGISTRY.register_processor；
2. 实现 ProcessingInfo，声明支持的 modality 和限制；
3. 实现 DummyInputsBuilder，用于 profile / memory planning；
4. 实现 _get_mm_fields_config()，把 HF processor 输出字段映射到 mm_kwargs；
5. 实现 _get_prompt_updates()，保证 placeholder 数量等于 encoder 输出 embedding 数；
6. 处理 image/video/audio embeds 等 passthrough 输入。
```

### 20.3 Position / special rope

如果模型需要特殊 position ids：

```text
M-RoPE：实现 SupportsMRoPE.get_mrope_input_positions()；
XD-RoPE：实现 SupportsXDRoPE.get_xdrope_input_positions()；
Pruning：实现 SupportsMultiModalPruning.recompute_mrope_positions()。
```

### 20.4 权重加载

需要确认：

```text
HF checkpoint 里的 language_model / tower / connector 前缀；
是否需要 mapper；
是否有 fused qkv / gate_up 等 packed modules；
mm_encoder_only 或 limit-mm-per-prompt=0 时缺失模块如何跳过。
```

---

## 21. 容易疑惑的点

### 21.1 多模态模型 forward 里为什么看不到 image 参数？

因为 decoder-only 多模态模型在 `GPUModelRunner._preprocess()` 阶段已经把 image/video/audio 转成 `inputs_embeds` 了。

模型 forward 通常只接收：

```text
input_ids
positions
intermediate_tensors
inputs_embeds
**kwargs
```

真正处理图像/音频的是 `embed_multimodal()`，不是主 forward。

### 21.2 placeholder token 是给 LLM 读的吗？

不是直接读原始 placeholder 的语义。

placeholder 的主要作用是：

```text
1. 在 token 序列中占位；
2. 让 attention / KV cache / positions 按真实长度构建；
3. 给 is_multimodal mask 提供替换位置。
```

随后这些位置的 token embedding 会被多模态 embedding 覆盖。

### 21.3 为什么需要 `mm_hash`？

`mm_hash` 是多模态 item 的稳定标识，用于：

```text
processor cache；
encoder cache；
跨 step 复用 encoder output；
Scheduler 指示 Worker 释放 cache。
```

### 21.4 为什么有 `scheduled_encoder_inputs`？

因为不是每个 step 都需要跑 encoder。

```text
prefill 首次遇到某个 image/audio：需要跑 encoder；
后续 chunk 或 decode：通常只从 encoder_cache 取；
请求结束：Scheduler 通过 free_encoder_mm_hashes 释放。
```

### 21.5 `is_embed` 是什么？

`PlaceholderRange.is_embed` 表示 placeholder 范围中哪些 token 位置真的要被多模态 embedding 替换。

这用于支持带 wrapper token 的模板，例如：

```text
<|vision_start|><|image_pad|>...<|vision_end|>
```

其中 start/end 可能不对应 encoder embedding。

### 21.6 为什么多模态模型通常用 `inputs_embeds` 而不是 `input_ids`？

因为图像/音频 embedding 不是词表 id，无法通过 embedding table lookup 得到。

统一成 `inputs_embeds` 后，LLM backbone 只需要处理 hidden-size 相同的连续向量序列。

### 21.7 多模态 encoder 输出数量不匹配会怎样？

`_merge_multimodal_embeddings()` 会在赋值失败时检查：

```text
num_actual_tokens != num_expected_tokens
```

然后抛出 placeholder 数量不匹配错误。

位置：`utils.py:501` 到 `utils.py:511`

### 21.8 图像 embedding 会每个 decode step 都重新算吗？

通常不会。

首次需要时 `_execute_mm_encoder()` 写入 `encoder_cache`，后续 `_gather_mm_embeddings()` 从 cache 取，直到 Scheduler 通过 `free_encoder_mm_hashes` 释放。

### 21.9 多模态模型和 pooling 模型冲突吗？

不冲突。

`GPUModelRunner.execute_model()` 在 forward 后判断：

```text
is_pooling_model → _pool()
否则 → compute_logits() → sample_tokens()
```

所以多模态 embedding 只是输入形态，输出可以是 generation，也可以是 pooling / embedding / rerank。

### 21.10 encoder-decoder 多模态模型是不是同一套替换逻辑？

不完全相同。

decoder-only 模型通常把多模态 embedding scatter 到 decoder prompt 的 `inputs_embeds`；encoder-decoder 模型则把 `_execute_mm_encoder()` 的结果作为 `encoder_outputs` 传给 decoder。

位置：`gpu_model_runner.py:3555` 到 `gpu_model_runner.py:3562`

---

## 22. 总结

多模态模型接入主链路可以压缩成：

```text
MultiModalRegistry / Processor
  → HF processor
  → placeholder update
  → MultiModalInput
  → Request.mm_features
  → SchedulerOutput.scheduled_encoder_inputs
  → GPUModelRunner._execute_mm_encoder()
  → model.embed_multimodal()
  → encoder_cache
  → GPUModelRunner._gather_mm_embeddings()
  → model.embed_input_ids(..., multimodal_embeddings, is_multimodal)
  → inputs_embeds
  → model.forward(inputs_embeds=..., positions=...)
  → language_model
  → logits / pooler
```

如果只记一句话：

```text
vLLM 的多模态模型不是让 LLM 直接读图片或音频，而是把非文本输入编码成和 token embedding 同维度的 soft tokens，再用 placeholder mask 精确覆盖到 inputs_embeds 中。
```

再压缩成最小心智模型：

```text
processor 负责占位；
Scheduler 负责安排 encoder；
ModelRunner 负责编码、缓存、合并；
模型类负责 tower/projector/backbone/权重映射；
最终 LLM 只看到融合后的 inputs_embeds。
```
