# 13 multimodal 背诵文档

## 1. 专题定位

`multimodal` 讲的是 vLLM V1 如何把图片、音频、视频、预计算 embeddings 等非文本输入接入语言模型执行链路。

它不只是 processor，也不只是模型的 vision tower。

一句话：

```text
Multimodal 负责把非文本输入变成模型 forward 能消费的 embeddings，并把这件事接入调度、缓存和执行链路。
```

## 2. 最小心智模型

主链路：

```text
用户请求 prompt + multi_modal_data
  → entrypoints / renderer
  → MultiModalDataParser
  → model-specific MultiModalProcessor
  → MultiModalInput(prompt_token_ids, mm_kwargs, mm_placeholders, mm_hashes)
  → InputProcessor.process_inputs()
  → MultiModalFeatureSpec
  → EngineCoreRequest.mm_features
  → Request.mm_features
  → Scheduler.schedule()
  → Scheduler._try_schedule_encoder_inputs()
  → EncoderCacheManager
  → SchedulerOutput.scheduled_encoder_inputs
  → GPUModelRunner._execute_mm_encoder()
  → model.embed_multimodal(**mm_kwargs)
  → GPUModelRunner.encoder_cache[identifier]
  → GPUModelRunner._gather_mm_embeddings()
  → model.embed_input_ids(...)
  → inputs_embeds
  → model forward / logits / sampling
```

要背住：

```text
原始 media → processor outputs → MultiModalFeatureSpec → Scheduler encoder 调度 → ModelRunner encoder 执行 → inputs_embeds 拼接 → 普通 forward。
```

## 3. Multimodal 负责什么

它负责：

```text
判断模型是否支持多模态
解析 image / audio / video / embeds 原始数据
调用模型注册的 processor
生成 placeholder / mm_position / mm_hash / identifier
构造 MultiModalFeatureSpec
把 mm_features 放进 EngineCoreRequest / Request
为 Scheduler 提供 encoder budget 和 encoder cache size
让 Scheduler 决定本轮哪些 encoder input 必须可用
在 ModelRunner 侧执行 model.embed_multimodal()
缓存 encoder output
根据 placeholder 位置把 encoder output 拼回 inputs_embeds
汇入普通 decoder forward、logits、sampling、output 链路
```

## 4. Multimodal 不负责什么

它不负责：

```text
用户请求队列调度
decoder KV block 分配和抢占
attention backend 主链路执行
token sampling
detokenize
OpenAI response 包装
模型权重加载
ModelRunnerOutput 到 RequestOutput 转换
```

这些属于 Scheduler、KVCacheManager、Attention、Sampler、OutputProcessor、ModelLoader。

## 5. 为什么要拆多层

多模态链路横跨 CPU 输入解析、Scheduler 预算、GPU encoder 执行和模型特定接口。

拆层原因：

```text
Renderer / Processor：处理原始 media 和 tokenizer / HF processor。
InputProcessor：把 processor 结果标准化成 EngineCoreRequest。
Scheduler：决定本轮 encoder output 是否必须可用。
ModelRunner：真正执行 encoder 并拼 inputs_embeds。
Model class：实现 embed_multimodal 和 embed_input_ids。
```

一句话：

```text
CPU 侧解析、Scheduler 侧预算、GPU 侧 encoder、模型特定语义必须分开。
```

## 6. MultiModalConfig

`MultiModalConfig` 控制多模态能力和资源边界。

关键字段：

```text
language_model_only
limit_per_prompt
enable_mm_embeds
media_io_kwargs
mm_processor_kwargs
mm_processor_cache_gb
mm_processor_cache_type
mm_tensor_ipc
mm_encoder_only
mm_encoder_tp_mode
mm_encoder_attn_backend
mm_encoder_attn_dtype
skip_mm_profiling
video_pruning_rate
```

要区分：

```text
processor 侧配置：影响 media decode、HF processor、CPU tensor、processor cache。
encoder 侧配置：影响 multimodal encoder 执行图、TP、attention backend、dtype。
```

## 7. MultiModalRegistry

`MultiModalRegistry` 是模型类到多模态 processor 的注册中心。

它负责：

```text
判断模型是否支持多模态输入
注册模型对应 processor / info / dummy inputs
创建 processor
创建 processor cache
给 MultiModalBudget 提供模型相关最大 token 信息
```

一句话：

```text
模型能不能处理 image / audio / video，不靠通用层硬编码，而靠模型类注册 processor 和 processing info。
```

## 8. MultiModalDataParser

`MultiModalDataParser` 把用户传入的多模态数据标准化。

支持：

```text
image
audio
video
vision_chunk
embeddings / prompt_embeds
```

它把各种输入形态变成内部 data items。

例如：

```text
URL / base64 / PIL / ndarray / tensor
音频 array / tensor / sampling rate
视频 frames / metadata
```

## 9. Renderer / InputPreprocessor

Renderer 负责调用 processor。

主流程：

```text
Renderer._process_multimodal()
  → get_mm_processor()
  → processor.info.parse_mm_data(mm_data)
  → parse_mm_uuids(mm_uuids)
  → processor.apply(MMProcessorInputs)
  → MultiModalInput
```

`InputPreprocessor` 多数情况下只是薄封装，把工作转发给 renderer。

## 10. MultiModalInput 包含什么

Renderer / processor 产出的 `MultiModalInput` 通常包含：

```text
prompt_token_ids
mm_kwargs
mm_placeholders
mm_hashes
```

含义：

```text
prompt_token_ids：包含多模态 placeholder 的 token 序列。
mm_kwargs：模型 processor 输出，后续传给 embed_multimodal()。
mm_placeholders：每个多模态 item 在 prompt token 中的位置范围。
mm_hashes：processor / encoder 缓存复用所需 hash。
```

## 11. InputProcessor 构造 mm_features

InputProcessor 会把 `MultiModalInput` 整理成 `EngineCoreRequest.mm_features`。

步骤：

```text
1. 从 decoder_inputs 中取出 mm_kwargs、mm_placeholders、mm_hashes。
2. 使用 argsort_mm_positions() 按 placeholder 出现顺序排序。
3. 对每个多模态 item 构造 MultiModalFeatureSpec。
4. 写入 EngineCoreRequest.mm_features。
```

为什么排序：

```text
Scheduler 和 ModelRunner 后续都按 request.mm_features 的索引引用多模态 item。
```

## 12. MultiModalFeatureSpec

这是 V1 多模态链路最核心的数据结构。

它代表一个多模态 item。

字段：

```text
data：processor 输出，即 encoder 需要的 kwargs；缓存命中时可以为 None。
modality：image / audio / video / prompt_embeds 等。
identifier：encoder output cache key。
mm_position：placeholder 在 prompt token 序列中的范围。
mm_hash：processor output cache key。
```

一句话：

```text
MultiModalFeatureSpec 把一个多模态 item 的数据、位置和缓存身份绑定在一起。
```

## 13. placeholder

多模态 decoder-only VLM 通常在 prompt token 中放一段 placeholder。

作用：

```text
先在 token 序列中占住位置，后续用 encoder embeddings 替换对应 token embeddings。
```

例如图片可能对应很多 placeholder token。

## 14. mm_position

`mm_position` 描述某个多模态 item 对应的 placeholder 范围。

它表达：

```text
offset：placeholder 起始 token 位置。
length：placeholder token 区间长度。
get_num_embeds()：对应多少 encoder embeddings。
get_embeds_indices_in_range()：把本轮 token window 映射到 embedding slice。
```

重要点：

```text
placeholder token 数量和 encoder embedding 数量不一定一一对应。
```

动态分辨率、视频、M-RoPE、prompt_embeds 场景都依赖精确映射。

## 15. mm_hash 和 identifier

### mm_hash

偏 processor/cache 层。

用途：

```text
processor output 复用
多进程 IPC 是否重传 tensor
基础 cache key
```

### identifier

偏 encoder output cache 层。

用途：

```text
GPUModelRunner.encoder_cache 的 key
Scheduler EncoderCacheManager 的逻辑 key
```

区别：

```text
mm_hash：processor output 复用 key。
identifier：encoder output 复用 key。
```

tower connector LoRA 场景下，identifier 可能带 LoRA 名称。

## 16. Scheduler 为什么要调度 encoder input

Scheduler 不跑 encoder。

但它必须保证：

```text
decoder forward 触达某个多模态 placeholder 前，对应 encoder output 已经可用。
```

因此它要判断：

```text
本轮 token window 是否覆盖某个 placeholder。
encoder output 是否已经 cache。
encoder compute budget 是否足够。
encoder cache 是否有空间。
是否允许切分多模态输入。
远端 EC connector 是否有缓存。
```

## 17. token window 如何决定 encoder 调度

Scheduler 看本轮 decoder token 范围：

```text
[num_computed_tokens, num_computed_tokens + num_new_tokens)
```

如果这个范围覆盖某个 `mm_feature.mm_position`：

```text
就必须保证该 mm_feature 对应 encoder output 已经可用。
```

如果预算或 cache 不够，Scheduler 会缩短本轮 `num_new_tokens`，最多算到 placeholder 前。

## 18. SchedulerOutput 多模态字段

多模态相关字段：

```text
scheduled_encoder_inputs: dict[req_id, list[int]]
free_encoder_mm_hashes: list[str]
```

含义：

```text
scheduled_encoder_inputs：某个 req_id 的哪些 mm_features 索引需要本轮跑 encoder。
free_encoder_mm_hashes：Scheduler 逻辑 cache 已驱逐，Worker 应删除物理 tensor。
```

注意：

```text
scheduled_encoder_inputs 存的是 mm_features 的索引，不是 hash。
```

## 19. processor cache 与 encoder cache

多模态有两类 cache。

### processor cache

缓存：

```text
Renderer / MultiModalProcessor 处理后的 processor outputs / mm_kwargs。
```

目的：

```text
避免重复 decode 图片、处理音频、抽帧、调用 HF processor、构造 CPU tensor。
```

### encoder cache

缓存：

```text
model.embed_multimodal() 的输出 tensor。
```

目的：

```text
避免重复运行视觉塔 / 音频塔 / 多模态 encoder。
```

## 20. EncoderCacheManager 与 GPUModelRunner.encoder_cache

### EncoderCacheManager

Scheduler 侧逻辑账本。

它记录：

```text
哪些 encoder output 已逻辑缓存
每个 cached output 被哪些 request 引用
哪些 output 没有引用、可释放
cache 空间是否足够
需要通知 worker 删除哪些物理 tensor
```

它不存 GPU tensor。

### GPUModelRunner.encoder_cache

Worker 侧物理 cache。

它存：

```text
model.embed_multimodal() 的输出 tensor
```

边界：

```text
EncoderCacheManager 决定哪些 output 应该存在。
GPUModelRunner.encoder_cache 真正保存 output tensor。
```

## 21. ModelRunner 多模态执行链路

`GPUModelRunner` 中的关键阶段：

```text
_update_states()
  → 保存 mm_features，删除 free_encoder_mm_hashes 对应物理 cache。

_batch_mm_inputs_from_scheduler()
  → 从 scheduled_encoder_inputs 取本轮要跑的 mm_features。

_execute_mm_encoder()
  → 调 model.embed_multimodal() 或直接处理 prompt_embeds。
  → 写入 encoder_cache[identifier]。

_gather_mm_embeddings()
  → 根据当前 token window 和 mm_position 从 encoder_cache 取 slice。

_preprocess()
  → model.embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
  → 生成 inputs_embeds。
```

## 22. _execute_mm_encoder

它是真正运行多模态 encoder 的入口。

处理情况：

```text
prompt_embeds：输入已经是 embedding，直接放入 encoder_cache。
普通 image / audio / video：按 modality 分组和 batch。
encoder CUDA graph：如果可用，走 cudagraph manager。
普通路径：调用 model.embed_multimodal(**mm_kwargs_batch)。
输出检查：确认 encoder output 数量和 item 数量一致。
写入 self.encoder_cache[identifier]。
需要时保存到 EC connector。
```

一句话：

```text
_execute_mm_encoder 把 processor 输出变成 GPU encoder embeddings。
```

## 23. _gather_mm_embeddings

执行完 encoder 后，还不能直接 forward。

因为：

```text
本轮 batch 可能只计算 prompt 一部分。
_prepare_inputs 可能重排 batch。
每个 request 当前 token window 不同。
```

`_gather_mm_embeddings()` 会：

```text
按当前 input_batch.req_ids 遍历 batch
根据 num_computed_tokens 和 num_scheduled_tokens 找本轮 token window
找 window 覆盖的 mm_features
根据 mm_position 计算 encoder output slice
从 encoder_cache[identifier] 取 tensor
生成 mm_embeds
生成 is_mm_embed mask
```

## 24. _preprocess 中如何拼 inputs_embeds

decoder-only VLM 路径：

```text
_execute_mm_encoder(scheduler_output)
  → _gather_mm_embeddings(scheduler_output)
  → model.embed_input_ids(
        input_ids,
        multimodal_embeddings=mm_embeds,
        is_multimodal=is_mm_embed,
    )
  → inputs_embeds
  → _model_forward(inputs_embeds=inputs_embeds)
```

关键转换：

```text
prompt token ids + placeholder + encoder output
  → inputs_embeds
```

之后模型 forward 不再关心用户原始图片或音频。

## 25. decoder-only 与 encoder-decoder

### decoder-only VLM

```text
prompt 中有 image/audio/video placeholder
  → 执行 multimodal encoder
  → gather embeddings
  → embed_input_ids 替换 placeholder 对应位置
  → decoder forward 吃 inputs_embeds
```

### encoder-decoder

```text
执行 encoder
  → 得到 encoder_outputs
  → 作为 model_kwargs 传给 decoder
```

一句话：

```text
Decoder-only VLM 把 encoder output 拼进 inputs_embeds；encoder-decoder 把 encoder output 作为 encoder_outputs。
```

## 26. prompt_embeds / embedding-only 路径

`prompt_embeds` 的特点：

```text
输入已经在 embedding space 中，不需要再执行 image/audio/video encoder。
```

但它仍复用多模态链路：

```text
MultiModalFeatureSpec
SchedulerOutput.scheduled_encoder_inputs
GPUModelRunner.encoder_cache
_gather_mm_embeddings()
model.embed_input_ids()
```

区别只是：

```text
_execute_mm_encoder 对 prompt_embeds 直接把 tensor 放进 encoder_cache。
```

## 27. 多模态和 chunked prefill

多模态 placeholder 可能很长，尤其图片和视频。

chunked prefill 下，prompt 分多轮执行。

Scheduler 必须保证：

```text
如果本轮 token window 覆盖某个多模态 placeholder，encoder output 必须在 forward 前可用。
```

如果不满足：

```text
本轮只算到 placeholder 前。
后续轮次有 budget / cache 后再继续。
```

这就是多模态和 chunked prefill 强耦合的原因。

## 28. 多模态和 LoRA

默认 language model LoRA：

```text
多模态 encoder output 拼成 inputs_embeds 后，language model 部分按普通 LoRA mapping 执行。
```

tower / connector LoRA：

```text
如果 enable_tower_connector_lora=True，同一 media 在不同 LoRA 下 encoder output 可能不同。
identifier 需要包含 LoRA 名称，避免错误复用 encoder cache。
```

## 29. 多模态模型接口

多模态 model class 通常需要提供：

```text
embed_multimodal(**mm_kwargs)
embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)
get_num_mm_encoder_tokens(...)
get_num_mm_connector_tokens(...)
get_mm_mapping()
```

通用层不理解每个模型的视觉塔、音频塔、projector 细节。

它只约定接口。

## 30. 常见易混点

### mm_hash 和 identifier 不一样

```text
mm_hash：processor output 复用 key。
identifier：encoder output cache key。
```

### EncoderCacheManager 不存 tensor

```text
它是 Scheduler 侧逻辑账本；GPU tensor 存在 ModelRunner.encoder_cache。
```

### scheduled_encoder_inputs 存的是索引

```text
它是 request.mm_features 的索引，不是 hash 或 tensor。
```

### 多模态不是绕过普通 forward

```text
多模态最终还是变成 inputs_embeds 或 encoder_outputs，然后走普通 forward / logits / sampling。
```

## 31. 与其他专题的关系

```text
engine：InputProcessor 如何构造 EngineCoreRequest.mm_features。
scheduler：如何调度 encoder input 和 encoder cache。
executor_worker_model_runner：ModelRunner 在 preprocess 中执行 encoder 和拼 embeddings。
model_architectures：多模态模型类要实现 embed_multimodal / embed_input_ids。
lora_and_adapters：tower connector LoRA 会影响 encoder cache key。
parallelism：mm encoder TP mode 和 encoder attention backend。
compilation_and_cuda_graph：multimodal encoder 可有独立 compile / cudagraph。
sampling_and_output：多模态 forward 后仍进入普通 logits / sampling / output。
```

## 32. 背诵总结

背这一段：

```text
vLLM V1 的多模态链路把 image、audio、video 或 prompt_embeds 从原始输入转换成模型 forward 可消费的 embeddings。Renderer 和 MultiModalProcessor 先把原始 media 转成 MultiModalInput，InputProcessor 再按 placeholder 位置构造 MultiModalFeatureSpec 并放入 EngineCoreRequest.mm_features。Scheduler 不跑 encoder，但会根据本轮 token window、encoder budget 和 encoder cache 判断哪些 mm_features 的 encoder output 必须可用，并写入 SchedulerOutput.scheduled_encoder_inputs。ModelRunner 根据这些索引执行 model.embed_multimodal，把输出存入 encoder_cache，再按 mm_position gather 对应 slice，通过 model.embed_input_ids 把文本 token embedding 和多模态 embedding 合成 inputs_embeds，最后进入普通 model forward、logits、sampling 和输出链路。
```
