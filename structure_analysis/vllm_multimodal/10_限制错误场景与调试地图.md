# 10 限制、错误场景与调试地图

本篇汇总 vLLM 多模态链路中的限制、常见错误、测试覆盖和调试阅读顺序。多模态问题通常跨越入口、processor、placeholder、scheduler、worker、模型接口和缓存，不适合只看报错所在文件。

## 1. 数量限制

每 prompt 的模态数量限制来自：

- `MultiModalConfig.limit_per_prompt`：`code/vllm/vllm/config/multimodal.py:80`
- `get_limit_per_prompt()`：`code/vllm/vllm/config/multimodal.py:310`
- `allowed_mm_limits()`：`code/vllm/vllm/multimodal/processing/context.py:392`
- `validate_num_items()`：`code/vllm/vllm/multimodal/processing/context.py:409`

规则：

```text
实际允许数量 = min(用户配置数量, 模型支持数量)
```

超限通常会报类似：

```text
At most N image(s) may be provided in one prompt.
```

测试：`code/vllm/tests/multimodal/test_processing.py:902`。

## 2. `language_model_only`

当启用 language-model-only，多模态输入会被禁用。

相关实现：`code/vllm/vllm/config/multimodal.py:310`。

测试：`code/vllm/tests/config/test_multimodal_config.py:29`。

表现：多模态 limit 返回 0，请求中带 image/audio/video 会在处理阶段被拒绝。

## 3. UUID 校验错误

renderer 侧 UUID 校验：`code/vllm/vllm/renderers/base.py:617`。

常见错误：

1. `multi_modal_uuids[modality]` 长度与 `multi_modal_data[modality]` item 数量不一致；
2. 某 modality data 为空但 UUID 存在；
3. item 为空且 UUID 也为空；
4. cache/prefix caching 关闭时用户以为 UUID 会跨请求复用，但实际被 request-local 标识替代。

测试：`code/vllm/tests/renderers/test_process_multi_modal_uuids.py:38`。

当 processor cache 和 prefix caching 都关闭时，系统可能忽略用户 UUID，使用 `<mm_req_id>-<modality>-<index>` 生成 request-local identifier。

实现：`code/vllm/vllm/renderers/base.py:662`。

## 4. processor cache 与 renderer worker 限制

当：

```text
renderer_num_workers > 1
且 mm_processor_cache_gb > 0
```

会报错，因为 multimodal processor cache 不是线程安全的。

实现：`code/vllm/vllm/config/model.py:703`。

测试：`code/vllm/tests/test_config.py:1436`。

## 5. MultiModalConfig backend/fp8/shm 错误

相关校验：`code/vllm/vllm/config/multimodal.py:226`。

常见错误：

- `mm_encoder_attn_backend="XFORMERS"` 已移除；
- backend 字符串非法；
- `mm_encoder_fp8_scale_path` / `save_path` 只能和 `mm_encoder_attn_dtype="fp8"` 联用；
- scale path 文件不存在；
- save path 父目录不存在；
- 非 SHM 模式设置 `mm_shm_cache_max_object_size_mb`。

测试：`code/vllm/tests/config/test_multimodal_config.py:11`、`code/vllm/tests/config/test_multimodal_config.py:46`。

## 6. encoder budget 错误

关键实现：`code/vllm/vllm/v1/core/encoder_cache_manager.py:269`。

常见错误：

```text
单个多模态 item token 数 > max_num_batched_tokens
且 disable_chunked_mm_input=False
```

这种情况下 scheduler 无法在一个 step 中容纳单个 item，会直接报错。

排查：

```text
max_num_batched_tokens
max_num_encoder_input_tokens
encoder_cache_size
max_tokens_per_mm_item
disable_chunked_mm_input
模型 processor 的 dummy input 估算
```

## 7. placeholder 与 encoder 输出长度不匹配

V1 输入校验：`code/vllm/vllm/v1/engine/input_processor.py:434`。

常见原因：

- prompt 中 placeholder token 数不对；
- processor replacement/insertion 没生效；
- 模型 `get_num_mm_encoder_tokens` / `get_num_mm_connector_tokens` 返回不对；
- image/video `grid_thw` 与 merge size 不匹配；
- audio effective length 与 feature mask 不匹配；
- 用户直接传 token prompt，但 token ids 没有包含正确 placeholder。

排查文件：

```text
multimodal/processing/processor.py
multimodal/inputs.py
v1/engine/input_processor.py
model_executor/models/vision.py
具体模型文件 qwen2_vl.py / llava.py / qwen2_audio.py / ...
```

## 8. OpenAI Chat 图片没有进入模型

按顺序检查：

```text
1. entrypoints/openai/chat_completion/serving.py 是否进入 render_chat
2. entrypoints/chat_utils.py 是否识别 content item
3. renderer 是否写入 prompt["multi_modal_data"]
4. media connector 是否成功 fetch image/audio/video
5. MultiModalDataParser 是否生成 item
6. processor 是否生成 mm_kwargs 和 placeholder
7. V1 InputProcessor 是否生成 mm_features
8. scheduler 是否调度 encoder input
9. GPU worker 是否执行 embed_multimodal
```

关键文件：

- `code/vllm/vllm/entrypoints/chat_utils.py:1863`
- `code/vllm/vllm/renderers/hf.py:985`
- `code/vllm/vllm/multimodal/media/connector.py:75`
- `code/vllm/vllm/multimodal/parse.py:489`
- `code/vllm/vllm/multimodal/processing/processor.py:1663`

## 9. Scheduler 没有调度 encoder input

检查：

```text
Request.mm_features 是否为空
当前 token window 是否覆盖 placeholder
EncoderCacheManager 是否已命中 cache
encoder compute budget 是否不足
encoder cache capacity 是否不足
_try_schedule_encoder_inputs 是否收缩了 num_new_tokens
```

关键文件：

- `code/vllm/vllm/v1/request.py:59`
- `code/vllm/vllm/v1/core/sched/scheduler.py:1279`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py:94`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py:123`

## 10. GPU worker 没有 merge 多模态 embedding

检查：

```text
scheduled_encoder_inputs 是否传到 worker
req_state.mm_features 是否存在对应 mm_input_id
GPU encoder cache 是否保存 encoder output
_gather_mm_embeddings 是否拿到当前 window 需要的 embedding
is_multimodal mask true 数量是否等于 mm embedding 数量
embed_input_ids 是否被调用
```

关键文件：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2846`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3100`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:3447`
- `code/vllm/vllm/model_executor/models/utils.py:479`

## 11. 多模态 LoRA 下 cache 命中异常

检查：

```text
mm_hash 是否相同
identifier 是否带 LoRA 语义
processor cache 是否应该跨 LoRA 共享
audio/image tower LoRA 是否影响 encoder output
EncoderCacheManager 是否使用 identifier
receiver cache 是否优先使用 mm_hash
```

关键文件：

- `code/vllm/vllm/multimodal/inputs.py:301`
- `code/vllm/vllm/multimodal/cache.py:589`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py:109`
- `code/vllm/tests/multimodal/test_cache.py:517`

## 12. EVS / video pruning 问题

相关文件：`code/vllm/vllm/multimodal/evs.py`。

排查：

```text
video_pruning_rate
compute_retained_tokens_count
compute_retention_mask
recompute_mrope_positions
首帧 token 是否保留
pruning 后 M-RoPE 位置是否重算
```

关键位置：

- `code/vllm/vllm/multimodal/evs.py:16`
- `code/vllm/vllm/multimodal/evs.py:38`
- `code/vllm/vllm/multimodal/evs.py:154`

## 13. 推荐阅读顺序

### 13.1 入口到 processor

```text
1. entrypoints/chat_utils.py
2. renderers/hf.py
3. inputs/llm.py
4. inputs/preprocess.py
5. multimodal/parse.py
6. multimodal/processing/context.py
7. multimodal/processing/processor.py
```

### 13.2 processor 到 engine request

```text
1. multimodal/inputs.py
2. inputs/engine.py
3. v1/engine/input_processor.py
4. v1/engine/core.py
5. v1/request.py
```

### 13.3 scheduler 和缓存

```text
1. multimodal/encoder_budget.py
2. v1/core/encoder_cache_manager.py
3. v1/core/sched/scheduler.py
4. multimodal/cache.py
5. multimodal/hasher.py
```

### 13.4 GPU 执行和模型实现

```text
1. v1/worker/gpu_input_batch.py
2. v1/worker/gpu_model_runner.py
3. v1/worker/gpu/mm/encoder_runner.py
4. model_executor/models/vision.py
5. model_executor/models/utils.py
6. 具体模型 qwen2_vl.py / llava.py / qwen2_audio.py / whisper.py
```

## 14. 测试地图

| 主题 | 测试文件 |
|---|---|
| hasher | `code/vllm/tests/multimodal/test_hasher.py:19` |
| processor cache | `code/vllm/tests/multimodal/test_cache.py:198` |
| SHM cache | `code/vllm/tests/multimodal/test_cache.py:336` |
| LoRA cache 共享 | `code/vllm/tests/multimodal/test_cache.py:517` |
| limit_mm_per_prompt | `code/vllm/tests/multimodal/test_processing.py:902` |
| UUID 校验 | `code/vllm/tests/renderers/test_process_multi_modal_uuids.py:38` |
| MultiModalConfig | `code/vllm/tests/config/test_multimodal_config.py:11` |
| renderer/cache 并发限制 | `code/vllm/tests/test_config.py:1436` |
| encoder cache manager | `code/vllm/tests/v1/core/test_encoder_cache_manager.py:35` |
| encoder cudagraph budget | `code/vllm/tests/v1/cudagraph/test_encoder_cudagraph.py:124` |

## 15. 一句话总结

多模态问题要沿链路排查：入口是否挂上 `multi_modal_data`，processor 是否生成正确 `mm_kwargs/placeholders`，engine 是否生成 `mm_features`，scheduler 是否有预算并调度 encoder，worker 是否执行并缓存 encoder output，模型是否把 embeddings 正确覆写到 placeholder 位置。
