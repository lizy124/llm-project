# vLLM 多模态 Q&A

## 1. vLLM 多模态主链路是什么？

```text
OpenAI Chat / Python Prompt
  ↓
multi_modal_data
  ↓
MultiModalDataParser
  ↓
BaseMultiModalProcessor
  ↓
mm_kwargs + mm_placeholders + mm_hashes
  ↓
InputProcessor → MultiModalFeatureSpec
  ↓
Request.mm_features
  ↓
Scheduler 调度 encoder input
  ↓
GPU worker 执行 embed_multimodal
  ↓
embed_input_ids 合并多模态 embedding
```

核心入口：

- `code/vllm/vllm/multimodal/processing/processor.py:1663`
- `code/vllm/vllm/v1/engine/input_processor.py:242`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1279`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`
- `code/vllm/vllm/model_executor/models/vision.py:95`

## 2. OpenAI Chat 里的图片在哪里变成 `multi_modal_data`？

链路：

```text
OpenAIServingChat
  ↓
render_chat(request)
  ↓
HF renderer render_messages()
  ↓
parse_chat_messages()
  ↓
prompt["multi_modal_data"] = mm_data
prompt["multi_modal_uuids"] = mm_uuids
```

关键位置：

- `code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:108`
- `code/vllm/vllm/renderers/hf.py:899`
- `code/vllm/vllm/entrypoints/chat_utils.py:1863`
- `code/vllm/vllm/renderers/hf.py:985`

## 3. Python API 如何传多模态数据？

Python prompt schema 支持：

- `multi_modal_data`
- `multi_modal_uuids`
- `mm_processor_kwargs`

定义位置：

- `code/vllm/vllm/inputs/llm.py:45`
- `code/vllm/vllm/inputs/llm.py:70`
- `code/vllm/vllm/inputs/llm.py:76`
- `code/vllm/vllm/inputs/llm.py:84`

## 4. `MultiModalConfig` 管什么？

定义：`code/vllm/vllm/config/multimodal.py:74`。

它主要管理：

- `limit_per_prompt` / `limit_mm_per_prompt`；
- `interleave_mm_strings`；
- `video_pruning_rate`；
- 多模态 processor kwargs 合并；
- mm encoder backend/fp8/shm 等校验。

关键接口：

- `get_limit_per_prompt()`：`code/vllm/vllm/config/multimodal.py:310`
- `merge_mm_processor_kwargs()`：`code/vllm/vllm/config/multimodal.py:326`

## 5. `limit_mm_per_prompt` 在哪里生效？

processing context 中生效：

- `allowed_mm_limits()`：`code/vllm/vllm/multimodal/processing/context.py:392`
- `validate_num_items()`：`code/vllm/vllm/multimodal/processing/context.py:409`

规则是：

```text
实际允许数量 = min(用户配置数量, 模型支持数量)
```

测试：`code/vllm/tests/multimodal/test_processing.py:902`。

## 6. 外部 URL 图片/视频/音频在哪里读取？

由 `MediaConnector` 处理。

定义：`code/vllm/vllm/multimodal/media/connector.py:75`。

关键函数：

- `load_from_url()`：`code/vllm/vllm/multimodal/media/connector.py:286`
- `fetch_audio()`：`code/vllm/vllm/multimodal/media/connector.py:371`
- `fetch_image()`：`code/vllm/vllm/multimodal/media/connector.py:401`
- `fetch_video()`：`code/vllm/vllm/multimodal/media/connector.py:451`

## 7. `MultiModalDataParser` 做什么？

它把原始 `MultiModalDataDict` 规范化为 `MultiModalDataItems`。

位置：

- `MultiModalDataParser`：`code/vllm/vllm/multimodal/parse.py:489`
- `parse_mm_data()`：`code/vllm/vllm/multimodal/parse.py:692`
- `parse_mm_uuids()`：`code/vllm/vllm/multimodal/parse.py:714`

它不做模型特征提取，只做输入形态标准化。

## 8. `BaseMultiModalProcessor` 做什么？

定义：`code/vllm/vllm/multimodal/processing/processor.py:972`。

职责：

1. 调 HF processor 生成 tensor kwargs；
2. 处理 processor cache；
3. 应用 prompt insertion/replacement；
4. 找到 placeholder token 区间；
5. 输出 `prompt_token_ids + mm_kwargs + mm_hashes + mm_placeholders`。

入口：`code/vllm/vllm/multimodal/processing/processor.py:1663`。

## 9. placeholder 是什么？

`PlaceholderRange` 记录某个多模态 item 在 prompt token 序列中的区间。

定义：`code/vllm/vllm/multimodal/inputs.py:118`。

字段：

- `offset`
- `length`
- `is_embed`

scheduler 和 GPU worker 都依赖它把 encoder output 对齐到 token window。

## 10. prompt replacement 和 insertion 是什么？

定义在：

- `PromptUpdate`：`code/vllm/vllm/multimodal/processing/processor.py:298`
- `PromptInsertion`：`code/vllm/vllm/multimodal/processing/processor.py:354`
- `PromptReplacement`：`code/vllm/vllm/multimodal/processing/processor.py:423`

作用：根据模型规则修改 prompt，例如插入 `<image>` token、把用户占位文本替换成模型需要的多 token 模板。

## 11. 为什么 processor 要优先 token-space 匹配？

因为最终需要准确的 token offset/length。

相关函数：

- `iter_token_matches()`：`code/vllm/vllm/multimodal/processing/processor.py:619`
- `replace_token_matches()`：`code/vllm/vllm/multimodal/processing/processor.py:648`
- `apply_token_matches()`：`code/vllm/vllm/multimodal/processing/processor.py:831`
- `apply_text_matches()`：`code/vllm/vllm/multimodal/processing/processor.py:848`

## 12. `Request.mm_features` 从哪里来？

V1 `InputProcessor` 会把：

```text
mm_kwargs + mm_placeholders + mm_hashes
```

转成 `MultiModalFeatureSpec` 列表。

关键位置：

- `code/vllm/vllm/v1/engine/input_processor.py:332`
- `code/vllm/vllm/multimodal/inputs.py:301`
- `code/vllm/vllm/v1/request.py:59`
- `code/vllm/vllm/v1/request.py:197`

## 13. `MultiModalFeatureSpec` 里有哪些重要字段？

定义：`code/vllm/vllm/multimodal/inputs.py:301`。

字段：

- `data`：单个 item 的 processor kwargs；
- `modality`：image/audio/video；
- `identifier`：encoder cache key；
- `mm_position`：placeholder 位置；
- `mm_hash`：processor cache/content hash。

## 14. scheduler 如何决定何时执行 encoder？

核心函数：`code/vllm/vllm/v1/core/sched/scheduler.py:1279`。

逻辑：

```text
当前 token window 覆盖某个 placeholder
  ↓
如果 encoder cache 命中 → 复用
否则如果 budget/cache capacity 足够 → 调度 encoder
否则 → 收缩本 step token 数，停在 placeholder 前
```

## 15. encoder budget 怎么算？

入口：

- `MultiModalBudget`：`code/vllm/vllm/multimodal/encoder_budget.py:44`
- `compute_mm_encoder_budget()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:269`

核心思想：budget 至少要容纳单个最大多模态 item，最终可用预算受 compute budget 和 cache size 共同限制。

## 16. scheduler 侧 encoder cache 和 GPU encoder cache 有什么区别？

| 缓存 | 内容 | 位置 |
|---|---|---|
| scheduler encoder cache | key、token 数、引用、容量 | `code/vllm/vllm/v1/core/encoder_cache_manager.py:17` |
| GPU encoder cache | 真实 encoder output tensor | `code/vllm/vllm/v1/worker/gpu/mm/encoder_cache.py:8` |

scheduler 不持有 tensor，GPU cache 不做全局 admission control。

## 17. GPU worker 如何执行多模态 encoder？

链路：

```text
scheduled_encoder_inputs
  ↓
_batch_mm_inputs_from_scheduler()
  ↓
_execute_mm_encoder()
  ↓
model.embed_multimodal(**mm_kwargs_batch)
```

关键位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2846`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`

## 18. 多模态 embedding 如何合入语言模型？

GPU worker 先 gather 当前 token window 需要的多模态 embeddings：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3100`

然后调用：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3447`

模型默认 `embed_input_ids()` 会把多模态 embeddings 覆写到 placeholder token embedding 位置：

- `code/vllm/vllm/model_executor/models/vision.py:374`
- `code/vllm/vllm/model_executor/models/utils.py:479`

## 19. 多模态模型必须实现什么？

核心接口：`code/vllm/vllm/model_executor/models/vision.py:95`。

常见方法：

- `embed_multimodal(...)`
- `get_language_model()`
- `embed_input_ids(...)`
- `get_num_mm_encoder_tokens(...)`
- `get_num_mm_connector_tokens(...)`

## 20. Whisper 为什么是例外？

Whisper 是 encoder-decoder 音频模型。它的音频 encoder output 不是覆写 decoder token embedding，而是作为 cross-attention 条件给 decoder。

关键位置：

- `code/vllm/vllm/model_executor/models/whisper.py:598`
- `code/vllm/vllm/model_executor/models/whisper.py:990`
- `code/vllm/vllm/model_executor/models/whisper.py:998`

## 21. `mm_hash` 和 `identifier` 有什么区别？

定义：`code/vllm/vllm/multimodal/inputs.py:301`。

- `mm_hash`：多模态内容/processor 输出身份，通常不带 LoRA 语义；
- `identifier`：encoder output 身份，可能带 LoRA 语义。

processor cache 通常优先用 `mm_hash`，encoder cache 用 `identifier`。

## 22. processor cache 和 encoder cache 有什么区别？

| 缓存 | 缓存内容 | key | 容量单位 |
|---|---|---|---|
| processor/receiver cache | `MultiModalKwargsItem`、processor 输出 | `mm_hash` 优先 | bytes |
| encoder cache | encoder output embeddings | `identifier` | encoder tokens/embeds |

相关文件：

- `code/vllm/vllm/multimodal/cache.py:98`
- `code/vllm/vllm/multimodal/cache.py:589`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py:17`

## 23. 常见“图片传了但模型没用上”怎么查？

按顺序：

```text
1. renderer 是否生成 multi_modal_data
2. MediaConnector 是否加载成功
3. MultiModalDataParser 是否生成 item
4. BaseMultiModalProcessor 是否生成 mm_kwargs/placeholders
5. InputProcessor 是否生成 mm_features
6. scheduler 是否调度 encoder input
7. GPU worker 是否执行 embed_multimodal
8. embed_input_ids 是否 merge embedding
```

## 24. placeholder 长度不匹配怎么查？

检查：

```text
processor 生成的 PlaceholderRange.length
模型 get_num_mm_encoder_tokens / get_num_mm_connector_tokens
image/video grid_thw、merge size
audio feature_attention_mask
InputProcessor 校验
```

校验位置：`code/vllm/vllm/v1/engine/input_processor.py:434`。

## 25. 推荐阅读顺序是什么？

```text
1. inputs/llm.py
2. entrypoints/chat_utils.py / renderers/hf.py
3. config/multimodal.py
4. multimodal/parse.py
5. multimodal/processing/context.py
6. multimodal/processing/processor.py
7. multimodal/inputs.py
8. v1/engine/input_processor.py
9. v1/request.py
10. multimodal/encoder_budget.py
11. v1/core/encoder_cache_manager.py
12. v1/core/sched/scheduler.py
13. v1/worker/gpu_model_runner.py
14. model_executor/models/vision.py
15. 具体模型 qwen2_vl.py / llava.py / qwen2_audio.py / whisper.py
```

## 26. 一句话总结

vLLM 多模态不是“图片直接送模型”，而是先把媒体解析成 processor kwargs，并把 prompt 占位变成精确 token 区间；runtime 再根据这些区间调度 encoder、缓存输出，最后在 embedding 层把多模态结果覆写回语言模型输入。
