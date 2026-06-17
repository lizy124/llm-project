# 03 MultiModalConfig 与媒体解析

本篇梳理 vLLM 多模态配置、输入数量限制、外部媒体加载和原始多模态数据解析。它们共同决定“一个请求允许带多少图片/音频/视频，以及这些外部对象如何变成 processor 可消费的内部 item”。

## 1. `MultiModalConfig` 的职责

定义：`code/vllm/vllm/config/multimodal.py:74`。

核心字段/接口：

- `limit_per_prompt`：`code/vllm/vllm/config/multimodal.py:80`
- `interleave_mm_strings`：`code/vllm/vllm/config/multimodal.py:180`
- `video_pruning_rate`：`code/vllm/vllm/config/multimodal.py:190`
- `_validate_limit_per_prompt()`：`code/vllm/vllm/config/multimodal.py:201`
- `get_limit_per_prompt()`：`code/vllm/vllm/config/multimodal.py:310`
- `merge_mm_processor_kwargs()`：`code/vllm/vllm/config/multimodal.py:326`

可以把它理解成多模态总策略配置，负责：

1. 每个 prompt 每种 modality 的数量上限；
2. 多模态字符串占位是否允许交错插入；
3. 视频 pruning 策略；
4. processor kwargs 合并；
5. 多模态 encoder backend/fp8/shm 等配置校验。

## 2. `limit_mm_per_prompt` / `limit_per_prompt`

`limit_mm_per_prompt` 在配置对象中体现为 `limit_per_prompt`。

它控制每个 prompt 最多允许多少个多模态 item，例如：

```text
image: 1
video: 1
audio: 2
```

读取接口：`code/vllm/vllm/config/multimodal.py:310`。

processing context 会把用户配置和模型支持上限合并：

- `allowed_mm_limits()`：`code/vllm/vllm/multimodal/processing/context.py:392`
- `validate_num_items()`：`code/vllm/vllm/multimodal/processing/context.py:409`

核心规则是：

```text
allowed_limit = min(user_limit, model_supported_limit)
```

如果请求超过上限，会在 processor 前报错，避免 HF processor 收到不合法输入。

## 3. `language_model_only` 对多模态的影响

当配置为 language-model-only 模式时，`get_limit_per_prompt()` 会让多模态限制返回 0，实际禁用多模态输入。

相关位置：`code/vllm/vllm/config/multimodal.py:310`。

这类模式适合只想使用语言模型部分、不启用 vision/audio tower 的场景。

## 4. media connector

外部媒体加载由 `MediaConnector` 负责。

定义：`code/vllm/vllm/multimodal/media/connector.py:75`。

关键函数：

- `merge_media_io_kwargs()`：`code/vllm/vllm/multimodal/media/connector.py:51`
- `load_from_url()`：`code/vllm/vllm/multimodal/media/connector.py:286`
- `fetch_audio()`：`code/vllm/vllm/multimodal/media/connector.py:371`
- `fetch_image()`：`code/vllm/vllm/multimodal/media/connector.py:401`
- `fetch_video()`：`code/vllm/vllm/multimodal/media/connector.py:451`

职责：

- 读取 URL / data URL / 本地路径 / bytes；
- 根据 modality 选择 image/audio/video IO；
- 处理重定向和缓存；
- 校验本地路径、远程域名等访问限制；
- 输出 parser/processor 可接受的媒体对象。

它是外部世界和 vLLM 内部多模态链路的边界层。

## 5. `MultiModalDataParser`

parser 文件：`code/vllm/vllm/multimodal/parse.py`。

关键结构：

- `MultiModalDataItems`：`code/vllm/vllm/multimodal/parse.py:419`
- `MultiModalDataParser`：`code/vllm/vllm/multimodal/parse.py:489`
- `_parse_audio_data()`：`code/vllm/vllm/multimodal/parse.py:566`
- `_parse_image_data()`：`code/vllm/vllm/multimodal/parse.py:605`
- `_parse_video_data()`：`code/vllm/vllm/multimodal/parse.py:626`
- `parse_mm_data()`：`code/vllm/vllm/multimodal/parse.py:692`
- `parse_mm_uuids()`：`code/vllm/vllm/multimodal/parse.py:714`

parser 的职责不是做模型特征提取，而是统一输入形态。

输入可能是：

```text
image → PIL image / URL / bytes / embedding
audio → waveform / URL / bytes / feature
video → frames / URL / bytes / metadata
```

输出是规范化的 `MultiModalDataItems`。

## 6. 原始媒体输入与 embedding-only 输入

parser 会区分：

1. 需要经过 HF processor / tower 的原始媒体；
2. 已经预计算好的 embeddings。

这个区分后续会影响：

- 是否需要执行 `embed_multimodal()`；
- 是否需要走 tower encoder；
- encoder budget 是否算作 tower modality；
- GPU worker 是否直接把 embedding 放入 encoder cache。

在 budget 中，embedding-only modality 虽然不进 tower，但仍可能占 encoder cache 空间。

相关预算逻辑：`code/vllm/vllm/multimodal/encoder_budget.py:115`。

## 7. processing context

核心文件：`code/vllm/vllm/multimodal/processing/context.py`。

关键类/方法：

- `InputProcessingContext`：`code/vllm/vllm/multimodal/processing/context.py:90`
- `get_mm_config()`：`code/vllm/vllm/multimodal/processing/context.py:155`
- `get_hf_processor()`：`code/vllm/vllm/multimodal/processing/context.py:179`
- `call_hf_processor()`：`code/vllm/vllm/multimodal/processing/context.py:244`
- `BaseProcessingInfo`：`code/vllm/vllm/multimodal/processing/context.py:296`
- `get_data_parser()`：`code/vllm/vllm/multimodal/processing/context.py:353`
- `allowed_mm_limits()`：`code/vllm/vllm/multimodal/processing/context.py:392`
- `validate_num_items()`：`code/vllm/vllm/multimodal/processing/context.py:409`
- `parse_mm_data()`：`code/vllm/vllm/multimodal/processing/context.py:430`

可以把 context 理解成 processor 的运行环境：它知道 tokenizer、HF processor、模型配置、多模态限制和 parser。

## 8. dummy inputs

dummy inputs builder 定义：`code/vllm/vllm/multimodal/processing/dummy_inputs.py:28`。

关键函数：

- `get_dummy_processor_inputs()`：`code/vllm/vllm/multimodal/processing/dummy_inputs.py:67`
- `_get_dummy_audios()`：`code/vllm/vllm/multimodal/processing/dummy_inputs.py:94`
- `_get_dummy_images()`：`code/vllm/vllm/multimodal/processing/dummy_inputs.py:115`
- `_get_dummy_videos()`：`code/vllm/vllm/multimodal/processing/dummy_inputs.py:147`

用途：

- profile；
- capacity estimation；
- 推导单个多模态 item 的最大 token/embedding 数；
- 构造最坏情况输入。

它不是在线推理主路径，但会影响 encoder budget、memory profile 和最大容量估计。

## 9. 视频 pruning / EVS

视频相关限制和 token 裁剪不只在 processor 中，还涉及 EVS。

相关文件：`code/vllm/vllm/multimodal/evs.py`。

关键函数：

- `compute_retained_tokens_count`：`code/vllm/vllm/multimodal/evs.py:16`
- `compute_retention_mask`：`code/vllm/vllm/multimodal/evs.py:38`
- `recompute_mrope_positions`：`code/vllm/vllm/multimodal/evs.py:154`

作用：

- 裁剪视频 token；
- 保留首帧 token；
- 根据跨帧相似度裁剪；
- pruning 后重算 M-RoPE 位置，避免位置编码错位。

## 10. 常见错误场景

### 10.1 超过每 prompt 多模态数量限制

错误通常来自：`code/vllm/vllm/multimodal/processing/context.py:409`。

排查：

```text
MultiModalConfig.limit_per_prompt
模型支持的 supported_mm_limits
请求中各 modality item 数量
language_model_only 是否为 true
```

### 10.2 media URL 读取失败

排查：

```text
MediaConnector.fetch_image / fetch_audio / fetch_video
远程域名是否被允许
本地路径是否允许访问
data URL 格式是否正确
媒体格式是否被对应 IO 支持
```

### 10.3 video metadata 不匹配

排查：

```text
_parse_video_data
video 输入是否有 processor 需要的 metadata
帧数、fps、duration、尺寸是否符合模型 processor 要求
```

### 10.4 processor kwargs 不生效

排查：

```text
请求级 mm_processor_kwargs
MultiModalConfig 中的 mm_processor_kwargs
merge_mm_processor_kwargs()
具体 HF processor 是否实际消费该字段
```

## 11. 一句话总结

`MultiModalConfig` 决定多模态输入允许什么、允许多少、processor 参数如何合并；`MediaConnector` 负责把外部媒体取回来；`MultiModalDataParser` 负责把原始输入规范化成 item；`InputProcessingContext` 则把配置、tokenizer、HF processor、parser 和数量校验组织成 processor 可运行的上下文。
