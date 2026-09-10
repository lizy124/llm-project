# 04. 多模态 placeholder 如何和 prompt token 对齐？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\inputs\engine.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\inputs.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\processing\processor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\renderers\hf.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\engine\input_processor.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\interfaces.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\models\qwen3_vl.py`

本问题关注：多模态输入不是简单追加到 prompt 后面，而是必须在文本 token 序列中占住模型语义上的位置。vLLM 用 `PlaceholderRange` 记录每个 media item 对应的 token span，再用 `MultiModalFeatureSpec` 把这个 span 和 processor / encoder 产出的 feature 绑定起来，最后在 ModelRunner / model embedding 阶段把多模态 embedding scatter 回这些位置。

---

## 1. 一句话回答

多模态 placeholder 的本质是：

```text
在 prompt_token_ids 中预留一段 token span，
用 PlaceholderRange(offset, length, is_embed) 记录这段 span，
再在 forward 前把该 span 中需要承载视觉 / 音频 / prompt_embeds 的位置替换成对应 embedding。
```

主链路是：

```text
chat template / renderer / HF processor
  → prompt 中出现 <image> / <|image_pad|> / <video> 等用户级占位符
  → MultiModalProcessor._get_prompt_updates()
  → PromptReplacement / PromptInsertion 把用户级占位符扩展成 feature 级 placeholder tokens
  → find_mm_placeholders() 找到每个 media item 的 token span
  → mm_input(prompt_token_ids, mm_kwargs, mm_hashes, mm_placeholders)
  → InputProcessor 按 offset 排序并生成 MultiModalFeatureSpec
  → Scheduler 按 encoder cache / budget 决定哪些 item 本轮编码
  → GPUModelRunner 执行 embed_multimodal() 或复用 encoder cache
  → _gather_mm_embeddings() 按本轮 token window 切片并生成 is_mm_embed mask
  → model.embed_input_ids(..., multimodal_embeddings, is_multimodal)
  → _merge_multimodal_embeddings() 覆盖 placeholder 位置的 text embedding
```

所以 placeholder 解决的不是“如何传图片”，而是“图片 / 音频 / 视频 feature 应该放在语言模型 token 序列的哪个位置”。

---

## 2. 先区分几个概念

### 2.1 用户级 placeholder

用户或 chat template 里看到的是用户级 placeholder，例如：

```text
<image>
<|image_pad|>
<|vision_start|><|video_pad|><|vision_end|>
```

这些 token 通常只是“这里有一个 media item”的标记，不一定等于最终送进 LLM 的 feature token 数。

例如一个 `<image>` 可能被扩展成几百个 image token；一个 video placeholder 可能被扩展成：

```text
timestamp tokens + vision_start_token + video feature tokens + vision_end_token
```

Qwen3-VL 的 `_get_prompt_updates()` 就是典型例子：

- image：把 `hf_processor.image_token` 替换成 `image_token_id * num_tokens`
- video：把 `<|vision_start|><|video_pad|><|vision_end|>` 替换成按帧、时间戳、vision 边界 token 组织出来的序列

位置：`qwen3_vl.py:1409` 到 `qwen3_vl.py:1496`

### 2.2 feature 级 placeholder tokens

processor 更新后的 `prompt_token_ids` 中，真正用于对齐的是 feature 级 placeholder tokens。

它们有两个特点：

```text
1. 数量要能和 encoder output 的 feature 数对上；
2. 位置要保留在 prompt_token_ids 中，因为 attention / position / KV cache 都仍然按统一 token 序列推进。
```

`BaseMultiModalProcessor.apply()` 的注释直接说明了这三步：

```text
1. 调 HF Processor 得到 token IDs 和 processor tensors；
2. 把 token IDs 中的相关片段更新成 placeholder tokens；
3. 从处理后的 token IDs 中抽取 placeholder token 的位置信息。
```

位置：`processor.py:1663` 到 `processor.py:1707`

### 2.3 PlaceholderRange

`PlaceholderRange` 是最终跨层传递的位置信息。

定义位置：`inputs.py:118`

关键字段：

```python
offset: int
length: int
is_embed: torch.Tensor | None = None
```

含义是：

```text
offset：placeholder span 在 prompt_token_ids 中的起点；
length：这个 span 覆盖多少个 token；
is_embed：可选 mask，表示 span 内哪些 token 真的要被多模态 embedding 替换。
```

最简单情况：

```text
PlaceholderRange(offset=2, length=336, is_embed=None)
```

表示从 token index 2 开始的 336 个 token 都是 image embedding 位置。

更复杂情况：

```text
PlaceholderRange(
  offset=2,
  length=5,
  is_embed=[False, True, False, True, True],
)
```

表示 span 长度是 5，但只有其中 3 个位置承载 encoder embedding，其他位置仍保留普通 text / special token embedding。

这对包含边界 token、时间戳 token、frame separator token 的模型很重要。

---

## 3. Processor 如何生成 placeholder range

### 3.1 模型给出 PromptUpdate

每个多模态模型的 processor 通过 `_get_prompt_updates()` 告诉 vLLM：

```text
对某个 modality，应该查找 prompt 里的什么 target，
以及把它插入或替换成什么 content。
```

抽象定义在：`processor.py:297`

两种更新模式：

```text
PromptInsertion：在 target 后插入 placeholder content；
PromptReplacement：用 placeholder content 替换 target。
```

位置：

- `PromptInsertion`：`processor.py:353`
- `PromptReplacement`：`processor.py:423`

例如 `PromptReplacement(modality="image", target="<image>", replacement=...)` 表示：

```text
把 prompt 里的一个 <image> 替换成这个 image item 对应的 feature placeholder 序列。
```

### 3.2 PromptUpdateDetails 允许“span 内部分位置才是 embedding”

`PromptUpdateDetails` 的字段是：

```python
full: _S
is_embed: Callable[[TokenizerLike | None, PromptSeq], torch.Tensor] | None = None
```

位置：`processor.py:205`

`full` 是完整替换内容；`is_embed` 是可选 mask。

如果 `is_embed=None`：

```text
full 中所有 token 都会被认为需要承载多模态 embedding。
```

如果提供 `is_embed`：

```text
full 中只有 mask=True 的 token 会被多模态 embedding 替换；
mask=False 的 token 仍按普通文本 token 走 embedding 层。
```

vLLM 提供了几个便利方法：

```text
PromptUpdateDetails.select_text(...)
PromptUpdateDetails.select_token_id(...)
PromptUpdateDetails.select_token_ids(...)
```

位置：`processor.py:229` 到 `processor.py:269`

这个设计解释了为什么 `PlaceholderRange.length` 不一定等于最终 encoder embedding 数。真正 embedding 数由 `PlaceholderRange.get_num_embeds()` 决定。

### 3.3 应用更新：先 token match，必要时退回 text match

`_apply_prompt_updates()` 会先尝试在 token ids 上应用更新：

```text
_apply_token_matches(token_ids, mm_prompt_updates)
```

如果没有全部匹配成功，会把 token ids decode 成文本，再按文本做更新，然后重新 tokenize：

```text
_apply_text_matches(_seq2text(...), mm_prompt_updates)
```

原因是：某些普通文本 target 可能因为 tokenizer 合并而无法在 token id 序列中直接找到。例如搜索 `foo`，但 prompt 里是 `food` 且 `food` 被 tokenizer 合成一个 token。

位置：`processor.py:1528` 到 `processor.py:1579`

### 3.4 查找 placeholder：_iter_placeholders()

更新完成后，`_find_mm_placeholders()` 会调用：

```python
find_mm_placeholders(new_token_ids, mm_prompt_updates, tokenizer)
```

位置：`processor.py:1074`

核心逻辑在 `_iter_placeholders()`：

```text
从 prompt_token_ids 左到右扫描；
按 modality 和 item_idx 顺序查找每个 item 的 replacement / insertion content；
找到后产出 PlaceholderFeaturesInfo；
每个 item 找到一次后 item_idx 前进；
匹配不允许重叠。
```

位置：`processor.py:865` 到 `processor.py:940`

`PlaceholderFeaturesInfo` 最后会转成 `PlaceholderRange`：

```python
def to_range(self) -> PlaceholderRange:
    return PlaceholderRange(
        offset=self.start_idx,
        length=self.length,
        is_embed=self.is_embed,
    )
```

位置：`processor.py:675` 到 `processor.py:694`

### 3.5 校验数量必须对齐

`_maybe_apply_prompt_updates()` 会校验三件事：

```text
1. mm_kwargs 中的 item 数量要等于输入 media item 数量；
2. prompt_updates 中的 item 数量要等于输入 media item 数量；
3. 找到的 mm_placeholders 数量要等于输入 media item 数量。
```

位置：`processor.py:1636` 到 `processor.py:1661`

如果 placeholder 缺失，常见错误信息会提示：

```text
Expected there to be N prompt updates corresponding to N image items...
This is likely because you forgot to include input placeholder tokens...
```

或：

```text
Expected there to be N prompt placeholders corresponding to N image items...
Make sure _call_hf_processor and _get_mm_fields_config are consistent...
```

这说明 vLLM 认为 media item 和 placeholder 是一一对应的：有几张图片，就必须能找到几个 image placeholder span。

---

## 4. mm_input 如何保存对齐结果

processor 最终返回的是 `MultiModalInput`。

定义位置：`engine.py:135`

关键字段：

```python
prompt_token_ids: list[int]
mm_kwargs: MultiModalKwargsOptionalItems
mm_hashes: MultiModalHashes
mm_placeholders: MultiModalPlaceholders
```

其中：

```text
prompt_token_ids：已经包含 feature 级 placeholder tokens；
mm_kwargs：每个 media item 经 processor 处理后的模型输入参数；
mm_hashes：每个 media item 的 hash，用于 processor / encoder cache；
mm_placeholders：每个 modality 下每个 item 的 PlaceholderRange。
```

`MultiModalPlaceholders` 的类型是：

```python
Mapping[str, Sequence[PlaceholderRange]]
```

位置：`engine.py:129`

所以数据结构大致是：

```text
mm_placeholders = {
  "image": [
    PlaceholderRange(offset=10, length=336),
    PlaceholderRange(offset=400, length=336),
  ],
  "video": [
    PlaceholderRange(offset=800, length=1200, is_embed=...),
  ],
}
```

注意：这里仍然按 modality 分组，不是执行层最终使用的扁平顺序。

---

## 5. InputProcessor 如何把 placeholder 绑定到 feature

Engine 侧的 `InputProcessor` 会把 `MultiModalInput` 转成 `EngineCoreRequest`。

位置：`input_processor.py:300` 到 `input_processor.py:385`

对多模态输入，它会取出：

```python
decoder_mm_inputs = decoder_inputs["mm_kwargs"]
decoder_mm_positions = decoder_inputs["mm_placeholders"]
decoder_mm_hashes = decoder_inputs["mm_hashes"]
```

位置：`input_processor.py:335` 到 `input_processor.py:338`

然后做一件关键的事：

```python
sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)
```

位置：`input_processor.py:352`

`argsort_mm_positions()` 会把按 modality 分组的 placeholders 打平，并按 `PlaceholderRange.offset` 升序排序。

位置：`utils.py:137` 到 `utils.py:157`

排序后，InputProcessor 逐个生成 `MultiModalFeatureSpec`：

```python
MultiModalFeatureSpec(
    data=decoder_mm_inputs[modality][idx],
    modality=modality,
    identifier=self._get_mm_identifier(base_mm_hash, lora_request),
    mm_position=decoder_mm_positions[modality][idx],
    mm_hash=base_mm_hash,
)
```

位置：`input_processor.py:354` 到 `input_processor.py:368`

这一步非常重要，因为后面的 scheduler 和 ModelRunner 不再主要看 `mm_placeholders` 字典，而是看请求上的：

```text
req.mm_features: list[MultiModalFeatureSpec]
```

每个 `MultiModalFeatureSpec` 同时携带：

```text
modality：image / audio / video / prompt_embeds；
data：processor 结果；
identifier：encoder cache key；
mm_position：这个 item 在 prompt_token_ids 中的 PlaceholderRange；
mm_hash：processor hash。
```

定义位置：`inputs.py:301` 到 `inputs.py:331`

---

## 6. 为什么要按 offset 排序

多模态输入可能混合多种 modality，例如：

```text
文本 A <image1> 文本 B <audio1> 文本 C <image2>
```

原始 `mm_placeholders` 是按 modality 分组的：

```text
image: [image1, image2]
audio: [audio1]
```

如果执行层直接按 modality 顺序处理，就会变成：

```text
image1, image2, audio1
```

这和 prompt 中的自然顺序不一致。

所以 `argsort_mm_positions()` 会按 `offset` 排序，得到：

```text
image1, audio1, image2
```

这保证：

```text
1. req.mm_features 按 prompt 中出现顺序排列；
2. Scheduler 按 token window 判断哪些 feature 落在本轮执行范围时可以二分 / 顺序处理；
3. _gather_mm_embeddings() 按请求 token span 拼接 embedding 时顺序稳定。
```

---

## 7. placeholder token 数如何确定

答案是：由模型的 `_get_prompt_updates()` 决定，通常依赖 processor output 的 feature shape。

以 Qwen3-VL image 为例：

```python
grid_thw = out_item["image_grid_thw"].data
num_tokens = int(grid_thw.prod()) // merge_length
return [hf_processor.image_token_id] * num_tokens
```

位置：`qwen3_vl.py:1426` 到 `qwen3_vl.py:1432`

这里 `num_tokens` 不是写死的，而是由 image processor 输出的 grid shape 决定。

video 更复杂：

```text
num_frames
每帧 tokens_per_frame
是否启用 video pruning
timestamp tokens
vision_start / vision_end token
video_token_id 是否是实际 embedding 位置
```

普通 video 路径用 `PromptUpdateDetails.select_token_id(..., video_token_id)` 只选择 video token 位置；启用 video pruning 时会先用 `from_seq()` 保留完整 span，再由模型 embedding 阶段细化实际替换位置。

位置：`qwen3_vl.py:1434` 到 `qwen3_vl.py:1559`

这解释了两个常见现象：

```text
1. 同一个模型下，不同尺寸图片可能产生不同数量的 placeholder tokens；
2. placeholder span 的 length 可能大于 encoder embedding 数，因为 span 内可能含有文本 / 边界 / 时间戳 token。
```

---

## 8. length、get_num_embeds()、encoder output shape 的关系

`PlaceholderRange.length` 表示 prompt 里的 token span 长度。

`PlaceholderRange.get_num_embeds()` 表示这个 span 里真正需要多模态 embedding 的位置数。

实现位置：`inputs.py:152` 到 `inputs.py:156`

逻辑是：

```text
如果 is_embed is None：get_num_embeds() = length；
如果 is_embed 不为空：get_num_embeds() = is_embed 中 True 的数量。
```

这个值会被多个地方使用。

### 8.1 输入校验

InputProcessor 会检查每个 item 的 embedding token 数是否超过 encoder cache 预分配大小：

```python
num_embeds = mm_position.get_num_embeds()
if num_embeds > self.mm_encoder_cache_size:
    raise ValueError(...)
```

位置：`input_processor.py:454` 到 `input_processor.py:467`

注意这里检查的是 `get_num_embeds()`，不是 `length`。

### 8.2 本轮 gather embedding

ModelRunner 在 `_gather_mm_embeddings()` 中按本轮执行窗口裁剪 placeholder span。

位置：`gpu_model_runner.py:3165` 到 `gpu_model_runner.py:3273`

核心变量：

```text
start_pos = pos_info.offset
num_encoder_tokens = pos_info.length
start_idx = max(num_computed_tokens - start_pos, 0)
end_idx = min(num_computed_tokens - start_pos + num_scheduled_tokens, num_encoder_tokens)
```

位置：`gpu_model_runner.py:3200` 到 `gpu_model_runner.py:3207`

如果 `is_embed` 存在，还会把 token span 内的局部范围映射成 encoder output 的局部范围：

```python
curr_embeds_start, curr_embeds_end = pos_info.get_embeds_indices_in_range(
    start_idx, end_idx
)
```

位置：`gpu_model_runner.py:3209` 到 `gpu_model_runner.py:3211`

这一步解决的是 chunked prefill 场景：

```text
一个 image placeholder span 可能跨多个调度 step；
每个 step 只执行其中一段 token；
ModelRunner 只能取本 step 需要的那段 encoder embedding。
```

---

## 9. ModelRunner 如何把 embedding 放回 placeholder 位置

### 9.1 先执行或复用多模态 encoder

`_execute_mm_encoder()` 会从本轮 `scheduled_encoder_inputs` 中取出需要编码的 media item。

位置：`gpu_model_runner.py:2956` 到 `gpu_model_runner.py:3163`

它先调用 `_batch_mm_inputs_from_scheduler()`：

```text
scheduled_encoder_inputs
  → req_state.mm_features[mm_input_id]
  → mm_hashes
  → mm_kwargs
  → mm_lora_refs
```

位置：`gpu_model_runner.py:2913` 到 `gpu_model_runner.py:2954`

然后按 modality / batch 组织调用：

```python
model.embed_multimodal(**mm_kwargs_batch)
```

位置：`gpu_model_runner.py:3119` 和 `gpu_model_runner.py:3150`

输出会写入：

```python
self.encoder_cache[mm_hash] = output
```

位置：`gpu_model_runner.py:3157` 到 `gpu_model_runner.py:3161`

### 9.2 再按本轮 token window gather

`_gather_mm_embeddings()` 会遍历当前 batch 中的请求：

```text
for req_id in self.input_batch.req_ids:
  num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
  num_computed_tokens = req_state.num_computed_tokens
  mm_features = req_state.mm_features
```

然后调用：

```python
get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_scheduled_tokens,
)
```

位置：`gpu_model_runner.py:3187` 到 `gpu_model_runner.py:3196`

只处理本轮 token window 覆盖到的 feature。

### 9.3 生成 is_mm_embed mask

`_gather_mm_embeddings()` 不只是返回 embedding，还会返回：

```python
is_mm_embed: torch.Tensor[bool]
```

它的长度是本轮调度 token 总数：

```python
is_mm_embed = torch.zeros(total_num_scheduled_tokens, dtype=torch.bool, device="cpu")
```

位置：`gpu_model_runner.py:3172` 到 `gpu_model_runner.py:3178`

当某个 placeholder span 落在本轮执行窗口内时，会把对应位置标成 True：

```python
is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = True
```

如果 `is_embed` 存在，则只 OR 上 mask=True 的位置：

```python
is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] |= is_embed
```

位置：`gpu_model_runner.py:3236` 到 `gpu_model_runner.py:3245`

这就是 placeholder range 到实际 embedding scatter mask 的转换。

### 9.4 embed_input_ids() 合并 text embedding 和 multimodal embedding

在 `_preprocess()` 中，多模态模型统一走 embeddings 输入：

```python
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

位置：`gpu_model_runner.py:3538` 到 `gpu_model_runner.py:3542`

`SupportsMultiModal.embed_input_ids()` 的语义是：

```text
先对 input_ids 做普通文本 embedding；
再把 multimodal_embeddings scatter 到 is_multimodal=True 的位置。
```

位置：`interfaces.py:380` 到 `interfaces.py:415`

真正覆盖位置的函数是 `_merge_multimodal_embeddings()`：

```python
inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
```

位置：`utils.py:526` 到 `utils.py:562`

如果实际 embedding 数和 mask 中 True 的数量不一致，会抛出：

```text
Attempted to assign ... multimodal tokens to ... placeholders
```

位置：`utils.py:545` 到 `utils.py:558`

这个错误通常表示：

```text
processor 生成的 placeholder 数量 / is_embed mask
和 model.embed_multimodal() 产出的 feature 数量不一致。
```

---

## 10. mm_req_doc_ranges 和 placeholder 的关系

`mm_req_doc_ranges` 不是多模态 embedding 对齐的主机制。

主机制是：

```text
PlaceholderRange → _gather_mm_embeddings() → is_mm_embed → _merge_multimodal_embeddings()
```

`mm_req_doc_ranges` 是 attention metadata 的补充信息，主要用于 mm prefix LM 类模型中，让 attention backend 知道某些视觉 document span 的双向范围。

构造位置：`gpu_model_runner.py:2352` 到 `gpu_model_runner.py:2410`

逻辑是：

```text
如果 self.is_mm_prefix_lm：
  遍历 req_state.mm_features；
  跳过 audio；
  对每个 mm_feature.mm_position 调 extract_embeds_range()；
  得到 image_doc_ranges；
  写入 CommonAttentionMetadata.mm_req_doc_ranges。
```

`extract_embeds_range()` 会基于 `PlaceholderRange.is_embed` 提取实际 embedding 区间。

位置：`inputs.py:179` 到 `inputs.py:202`

所以关系可以这样理解：

```text
placeholder range：决定 embedding 放到哪里；
mm_req_doc_ranges：在部分 attention backend / mm prefix LM 场景中，告诉 attention 哪些多模态 span 有特殊 attention 语义。
```

---

## 11. prompt_embeds 混合输入如何接入同一套机制

`prompt_embeds` 混合输入也复用了 `mm_placeholders` 机制。

HF renderer 中 `_apply_prompt_embeds_to_engine_input()` 会把 prompt embeds 当成一个特殊 modality：

```text
modality = "prompt_embeds"
```

它会向 `engine_input` 里追加：

```text
mm_kwargs["prompt_embeds"]
mm_hashes["prompt_embeds"]
mm_placeholders["prompt_embeds"]
```

位置：`hf.py:1278` 到 `hf.py:1329`

对应的 placeholder 是：

```python
PlaceholderRange(offset=start, length=length, is_embed=None)
```

位置：`hf.py:1315` 到 `hf.py:1319`

Worker 侧对 `prompt_embeds` 有特殊处理：它不需要再跑多模态 encoder，而是在 `_execute_mm_encoder()` 里直接把 tensor 放进 encoder cache：

```python
self.encoder_cache[mm_hashes[i]] = pe_tensor.to(self.device)
```

位置：`gpu_model_runner.py:2966` 到 `gpu_model_runner.py:2991`

之后 `_gather_mm_embeddings()` 和 `_merge_multimodal_embeddings()` 完全复用普通多模态路径。

所以 prompt_embeds 的对齐方式是：

```text
先在 token 序列中用 placeholder span 占位，
再把外部传入的 embedding tensor 当成 encoder output，
最后按 is_mm_embed mask scatter 到 inputs_embeds。
```

---

## 12. 多个 image / video 如何区分

区分靠三个层次共同保证。

### 12.1 processor 阶段按 item_idx 绑定

`_bind_and_group_updates()` 会把每个 `PromptUpdate` 按 modality 和 item_idx resolve：

```text
mm_prompt_updates[modality][item_idx]
```

位置：`processor.py:1042` 到 `processor.py:1053`

`_iter_placeholders()` 扫描 prompt 时，每找到一个 item 的 placeholder，就让该 modality 的 `item_idx` 前进。

位置：`processor.py:879` 到 `processor.py:931`

所以同一种 modality 下的多个 item 是按 prompt 中匹配顺序对应的。

### 12.2 engine 阶段按 offset 排序

InputProcessor 使用 `argsort_mm_positions()` 将所有 modality 的 item 按 prompt 位置排序。

位置：`input_processor.py:349` 到 `input_processor.py:368`

这保证跨 modality 的相对顺序也正确。

### 12.3 runtime 阶段用 mm_feature.identifier 对 encoder cache

每个 `MultiModalFeatureSpec` 有自己的 `identifier`：

```text
base mm hash + 可选 LoRA prefix
```

ModelRunner 编码后用这个 identifier 写入 / 读取 encoder cache。

位置：

- 生成 `MultiModalFeatureSpec.identifier`：`input_processor.py:361` 到 `input_processor.py:366`
- 从 encoder cache 读取：`gpu_model_runner.py:3217` 到 `gpu_model_runner.py:3228`

所以多个 image 即使 token 形式相同，也会通过 item 顺序、offset 和 hash / identifier 区分。

---

## 13. 和 attention / position / logits 的关系

placeholder token 是 prompt token 序列的一部分，因此它会影响：

```text
prompt length
position ids
chunked prefill window
KV cache slot mapping
attention metadata
logits index 的相对位置
max_model_len 校验
```

但 logits 通常不会直接在多模态 placeholder 上采样。生成类模型的 logits 关注的是本轮需要输出 / 采样的位置，而 placeholder 主要影响 prefill 中的上下文表示。

也就是说：

```text
多模态 placeholder 参与序列长度、position 和 attention，
但它们的 embedding 在 forward 前被替换成多模态 encoder output。
```

`_preprocess()` 中多模态路径会最终返回：

```text
input_ids：某些模型仍需要 raw input tokens 时保留；
inputs_embeds：真正喂给多模态 LLM 的 embedding 序列；
positions：仍按统一 token 序列准备。
```

位置：`gpu_model_runner.py:3468` 到 `gpu_model_runner.py:3553`

---

## 14. 常见错误如何定位

### 14.1 忘了在 prompt 中放 placeholder

表现：processor 校验失败，提示 prompt updates 或 prompt placeholders 数量不匹配。

重点看：

```text
输入 media item 数量
chat template 是否插入 <image> / <video>
模型 _get_prompt_updates() 的 target 是否能匹配 prompt
```

源码位置：`processor.py:1600` 到 `processor.py:1634`

### 14.2 placeholder span 数量和 encoder output 数量不一致

表现：merge 阶段报：

```text
Attempted to assign X multimodal tokens to Y placeholders
```

重点看：

```text
_get_prompt_updates() 计算的 replacement token 数；
PromptUpdateDetails.is_embed mask；
model.embed_multimodal() 输出 shape；
_get_mm_fields_config() 是否和 HF processor 输出一致。
```

源码位置：`utils.py:545` 到 `utils.py:558`

### 14.3 max_model_len 不够

因为 placeholder tokens 已经进入 `prompt_token_ids`，所以 prompt 长度会包含文本 token + 多模态 placeholder token。

InputProcessor 的错误提示会说明：

```text
For image inputs, the number of image tokens depends on the number of images,
and possibly their aspect ratios as well.
```

位置：`input_processor.py:387` 到 `input_processor.py:432`

### 14.4 encoder cache size 不够

每个 item 的 `get_num_embeds()` 不能超过 `mm_encoder_cache_size`。

位置：`input_processor.py:454` 到 `input_processor.py:467`

这类问题通常需要减少单个 media item 的 feature 数，或调整 `--limit-mm-per-prompt` 等启动配置。

---

## 15. 总结

可以把 placeholder alignment 理解成四张表之间的对齐：

```text
prompt_token_ids
  token index → token id

mm_placeholders
  modality / item_idx → PlaceholderRange(offset, length, is_embed)

mm_features
  prompt offset order → MultiModalFeatureSpec(data, identifier, mm_position)

is_mm_embed + mm_embeds
  scheduled token index → 是否用多模态 embedding 覆盖
```

最终 forward 前发生的是：

```text
普通 token：input_ids → text embedding
placeholder token：input_ids 先占位 → 对应位置被 multimodal embedding 覆盖
```

因此，多模态 prompt 对齐的核心不在“额外传了哪些 tensor”，而在：

```text
processor 生成的 feature 数、placeholder token span、encoder output shape、ModelRunner 本轮 token window
必须在同一个 token 坐标系下保持一致。
```
