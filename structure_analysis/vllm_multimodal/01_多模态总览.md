# 01 多模态总览

本篇从整体上梳理 vLLM multimodal 的分层。vLLM 多模态不是单纯“把图片传给模型”，而是贯穿入口解析、prompt 改写、processor、engine request、scheduler、encoder cache、GPU worker、模型接口和 embedding 合并的一条完整链路。

## 1. 端到端主链路

```text
OpenAI Chat / Python Prompt
  ↓
multi_modal_data / multi_modal_uuids / mm_processor_kwargs
  ↓
InputPreprocessor / renderer
  ↓
MultiModalDataParser
  ↓
BaseMultiModalProcessor
  ├─ 调 HF processor 得到 tensor kwargs
  ├─ 生成 mm_kwargs / mm_hashes
  └─ 生成 mm_placeholders
  ↓
MultiModalInput
  ↓
V1 InputProcessor.process_inputs()
  ↓
MultiModalFeatureSpec 列表
  ↓
EngineCoreRequest.mm_features
  ↓
Request.mm_features
  ↓
Scheduler 根据 token window 调度 encoder inputs
  ↓
GPU worker 执行 embed_multimodal
  ↓
GPU encoder cache 保存 encoder outputs
  ↓
当前 step gather multimodal embeddings
  ↓
model.embed_input_ids(..., multimodal_embeddings=...)
  ↓
语言模型 forward
```

关键入口：

- OpenAI chat serving：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:108`
- chat message 解析：`code/vllm/vllm/entrypoints/chat_utils.py:1863`
- HF renderer：`code/vllm/vllm/renderers/hf.py:899`
- Python Prompt schema：`code/vllm/vllm/inputs/llm.py:45`
- 输入预处理：`code/vllm/vllm/inputs/preprocess.py:274`
- 多模态 parser：`code/vllm/vllm/multimodal/parse.py:489`
- 多模态 processor：`code/vllm/vllm/multimodal/processing/processor.py:972`
- V1 engine input processor：`code/vllm/vllm/v1/engine/input_processor.py:242`
- runtime request：`code/vllm/vllm/v1/request.py:59`
- scheduler：`code/vllm/vllm/v1/core/sched/scheduler.py:387`
- GPU worker/model runner：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2846`
- 多模态模型接口：`code/vllm/vllm/model_executor/models/vision.py:95`

## 2. 关键数据结构

| 数据结构 | 位置 | 职责 |
|---|---|---|
| `MultiModalDataDict` | `code/vllm/vllm/inputs/llm.py:45` | 用户/Python prompt 中携带的原始多模态数据字典。 |
| `MultiModalDataItems` | `code/vllm/vllm/multimodal/parse.py:419` | parser 后的规范化多模态 item 集合。 |
| `MultiModalKwargsItem` | `code/vllm/vllm/multimodal/inputs.py:854` | 单个多模态 item 经 processor 后的模型输入 kwargs。 |
| `MultiModalKwargsItems` | `code/vllm/vllm/multimodal/inputs.py:882` | 按 modality/item 组织的 processor 输出。 |
| `PlaceholderRange` | `code/vllm/vllm/multimodal/inputs.py:118` | 某个多模态 item 在 prompt token 序列中的占位区间。 |
| `MultiModalInput` | `code/vllm/vllm/inputs/engine.py:126` | engine 输入层承载 prompt token、mm kwargs、hash、placeholder 的结构。 |
| `MultiModalFeatureSpec` | `code/vllm/vllm/multimodal/inputs.py:301` | V1 runtime 中每个多模态 item 的统一描述。 |
| `Request.mm_features` | `code/vllm/vllm/v1/request.py:59` | scheduler/worker 后续处理多模态的中心字段。 |

## 3. 分层职责

### 3.1 入口层

入口层负责把外部协议转换成 vLLM prompt：

- OpenAI Chat message 中的 image/audio/video content；
- Python API 中的 `multi_modal_data`；
- 可选 `multi_modal_uuids`；
- 可选 `mm_processor_kwargs`。

OpenAI Chat 通过 `parse_chat_messages()` 和 renderer 把多模态数据挂到 prompt dict：

```text
prompt["multi_modal_data"] = mm_data
prompt["multi_modal_uuids"] = mm_uuids
```

相关位置：`code/vllm/vllm/renderers/hf.py:985`。

### 3.2 parser/media 层

这一层负责把外部对象变成 vLLM 内部能识别的 item：

- URL / data URL / bytes / file path；
- PIL image；
- audio；
- video；
- 已经预计算好的 embedding。

核心文件：

- `code/vllm/vllm/multimodal/media/connector.py:75`
- `code/vllm/vllm/multimodal/parse.py:489`

### 3.3 processor 层

processor 是多模态链路的中枢，负责两件事：

1. 调 HF processor，把原始媒体转成 tensor kwargs；
2. 修改/定位 prompt 中的多模态占位。

核心入口：`code/vllm/vllm/multimodal/processing/processor.py:1663`。

processor 输出不只是 tensor，还包括：

```text
prompt_token_ids
mm_kwargs
mm_hashes
mm_placeholders
```

### 3.4 engine/request 层

V1 `InputProcessor` 将 `MultiModalInput` 转成 runtime 能理解的 `MultiModalFeatureSpec`：

```text
mm_kwargs + mm_placeholders + mm_hashes
  ↓
MultiModalFeatureSpec(data, modality, identifier, mm_position, mm_hash)
```

相关位置：`code/vllm/vllm/v1/engine/input_processor.py:332`。

最终进入：`Request.mm_features`。

### 3.5 scheduler/cache 层

scheduler 不直接执行 image/audio/video encoder。它根据 `Request.mm_features` 判断：

- 当前 token window 是否走到了某个多模态 placeholder；
- 这个 encoder output 是否已缓存；
- 本 step 是否有 encoder compute budget；
- encoder cache 是否有容量；
- 如果不能调度，是否要收缩本 step 的 token 数。

核心函数：`code/vllm/vllm/v1/core/sched/scheduler.py:1279`。

encoder cache 记账器：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`。

### 3.6 GPU worker 层

worker 才真正执行多模态 encoder：

```text
scheduled_encoder_inputs
  ↓
_batch_mm_inputs_from_scheduler()
  ↓
_execute_mm_encoder()
  ↓
model.embed_multimodal(**mm_kwargs_batch)
  ↓
GPU encoder cache
  ↓
_gather_mm_embeddings()
  ↓
embed_input_ids(..., multimodal_embeddings=...)
```

关键位置：

- batch mm inputs：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2846`
- 执行 encoder：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2889`
- gather embeddings：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3100`
- 合入 input embedding：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3447`

### 3.7 模型接口层

模型类通过 `SupportsMultiModal` 等接口接入通用 runtime：

- `embed_multimodal(...)`：把 image/audio/video 输入变成 embeddings；
- `embed_input_ids(...)`：把 multimodal embeddings 覆写到 token embedding 序列；
- `get_language_model()`：返回语言模型主体；
- `get_num_mm_encoder_tokens(...)` / `get_num_mm_connector_tokens(...)`：给预算和 placeholder 长度使用。

接口位置：`code/vllm/vllm/model_executor/models/vision.py:95`。

## 4. 两类缓存必须区分

vLLM 多模态至少有两层缓存：

| 缓存 | key | 缓存内容 | 容量单位 | 所在层 |
|---|---|---|---|---|
| processor / receiver cache | `mm_hash` 或 `mm_hash or identifier` | 预处理后的 `MultiModalKwargsItem` / IPC 对象 | bytes | frontend/core 输入处理 |
| encoder cache | `identifier` | 多模态 encoder 输出 embeddings | encoder tokens / embeds | scheduler/GPU worker |

相关文件：

- processor cache：`code/vllm/vllm/multimodal/cache.py:98`
- receiver cache：`code/vllm/vllm/multimodal/cache.py:589`
- encoder cache manager：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`
- GPU encoder cache：`code/vllm/vllm/v1/worker/gpu/mm/encoder_cache.py:8`

## 5. 多模态融合的核心思想

vLLM 的 decoder-only 多模态主路径不是把图片 tensor 直接塞进 attention，而是：

1. 在 prompt token 序列中保留 placeholder token 区间；
2. encoder 输出多模态 embeddings；
3. 在 embedding 层把 placeholder 位置替换为多模态 embeddings；
4. 替换后的 embedding 序列进入语言模型。

公共合并逻辑：`code/vllm/vllm/model_executor/models/utils.py:479`。

默认 `embed_input_ids()`：`code/vllm/vllm/model_executor/models/vision.py:374`。

Whisper 等 encoder-decoder 音频模型是例外：多模态 encoder 输出作为 cross-attention 条件，而不是覆写 decoder token embedding。

## 6. 一句话总结

vLLM 多模态的本质是“prompt 占位坐标 + processor tensor kwargs + encoder output cache + embedding 层覆写”的组合系统；它把不同外部模态统一成 `Request.mm_features`，再由 scheduler 和 GPU worker 围绕这些 feature 进行按需 encoder 计算与复用。
