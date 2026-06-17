# 05 Engine、Request 与 mm_features

本篇梳理多模态 processor 输出如何进入 V1 engine，并最终成为 runtime `Request.mm_features`。这是从“输入预处理”进入“调度与执行”的边界。

## 1. 总体链路

```text
MultiModalInput
  ├─ prompt_token_ids
  ├─ mm_kwargs
  ├─ mm_placeholders
  └─ mm_hashes
        ↓
InputProcessor.process_inputs()
        ↓
MultiModalFeatureSpec 列表
        ↓
EngineCoreRequest.mm_features
        ↓
EngineCore.preprocess_add_request()
        ↓
Request.from_engine_core_request()
        ↓
Request.mm_features
```

关键位置：

- `InputProcessor` 初始化：`code/vllm/vllm/v1/engine/input_processor.py:36`
- `process_inputs()`：`code/vllm/vllm/v1/engine/input_processor.py:242`
- 构造 `MultiModalFeatureSpec`：`code/vllm/vllm/v1/engine/input_processor.py:332`
- `EngineCoreRequest`：`code/vllm/vllm/v1/engine/__init__.py:86`
- `EngineCore.preprocess_add_request()`：`code/vllm/vllm/v1/engine/core.py:853`
- `Request.from_engine_core_request()`：`code/vllm/vllm/v1/request.py:197`
- `Request.mm_features` 保存：`code/vllm/vllm/v1/request.py:59`

## 2. `InputProcessor` 初始化

`InputProcessor` 在初始化时会准备多模态相关预算和限制。

位置：`code/vllm/vllm/v1/engine/input_processor.py:36`。

它需要知道：

- 模型配置；
- 多模态配置；
- processor/registry；
- encoder cache size；
- 多模态 item 最大长度；
- prompt 侧合法性约束。

因此它不是只做简单 tokenization，而是 engine 侧多模态输入合法性与 runtime feature 构造的入口。

## 3. `process_inputs()` 处理多模态输入

主入口：`code/vllm/vllm/v1/engine/input_processor.py:242`。

处理 decoder 多模态输入时，它会读取：

```text
decoder_inputs["mm_kwargs"]
decoder_inputs["mm_placeholders"]
decoder_inputs["mm_hashes"]
```

然后按 placeholder 位置排序，构造每个多模态 item 的 runtime 描述。

排序函数：`code/vllm/vllm/multimodal/utils.py:137`。

## 4. `MultiModalFeatureSpec`

定义：`code/vllm/vllm/multimodal/inputs.py:301`。

它包含：

```text
data: MultiModalKwargsItem
modality: image/audio/video/...
identifier: encoder cache key
mm_position: PlaceholderRange
mm_hash: processor/cache hash
```

关键字段语义：

| 字段 | 说明 |
|---|---|
| `data` | 单个多模态 item 的 processor 输出 kwargs。 |
| `modality` | 模态名，如 image/audio/video。 |
| `identifier` | encoder output cache key，可带 LoRA 语义。 |
| `mm_position` | 在 prompt token 序列中的 placeholder 区间。 |
| `mm_hash` | processor cache / 内容 hash，通常不带 LoRA 语义。 |

构造位置：`code/vllm/vllm/v1/engine/input_processor.py:332`。

## 5. `identifier` 与 `mm_hash`

这两个字段必须区分：

- `mm_hash` 更偏向“多模态内容/processor 输出是否相同”；
- `identifier` 更偏向“encoder 输出是否可复用”。

当 LoRA 影响多模态 tower/connector 时，同一张图的 processor 输出可以共享，但 encoder output 不一定能共享。因此 `identifier` 可能带 LoRA 前缀，而 `mm_hash` 不带。

定义位置：`code/vllm/vllm/multimodal/inputs.py:301`。

receiver cache 也体现了这个区别：`code/vllm/vllm/multimodal/cache.py:589`。

## 6. placeholder 合法性校验

V1 engine 会校验多模态 placeholder 和 encoder 输出长度关系。

位置：`code/vllm/vllm/v1/engine/input_processor.py:434`。

常见校验：

- placeholder 长度是否和 encoder/connector token 数匹配；
- 单个多模态 item 是否超过 encoder cache 可容纳上限；
- prompt token 序列中多模态范围是否有效；
- encoder-decoder 输入和 decoder-only 输入是否符合模型结构。

这一步很重要，因为如果 placeholder 区间错误，scheduler 和 GPU worker 后续会在错误位置 gather/merge embeddings。

## 7. `EngineCoreRequest`

`InputProcessor.process_inputs()` 返回 `EngineCoreRequest`。

结构位置：`code/vllm/vllm/v1/engine/__init__.py:86`。

它包含：

- request id；
- prompt token ids；
- sampling params；
- mm_features；
- prompt embeds；
- encoder-decoder 相关输入；
- LoRA / priority / tracing 等。

多模态输入在这里已经从“processor 输出字段”变成了统一的 `mm_features`。

## 8. `EngineCore` 到 `Request`

转入 runtime request 的位置：`code/vllm/vllm/v1/engine/core.py:853`。

流程：

```text
EngineCore.preprocess_add_request(engine_core_request)
  ↓
如果有 mm_receiver_cache，先更新 request.mm_features 中 data
  ↓
Request.from_engine_core_request(...)
```

`Request.from_engine_core_request()`：`code/vllm/vllm/v1/request.py:197`。

它基本把 `EngineCoreRequest.mm_features` 原样传给 `Request`。

## 9. `Request.mm_features`

`Request` 保存：

```python
self.mm_features = mm_features or []
```

位置：`code/vllm/vllm/v1/request.py:59`。

后续 scheduler 和 worker 都围绕它工作。

辅助接口：

- `num_encoder_inputs`：`code/vllm/vllm/v1/request.py:258`
- `has_encoder_inputs`：`code/vllm/vllm/v1/request.py:285`
- `get_num_encoder_embeds`：`code/vllm/vllm/v1/request.py:285`

这些接口用于 scheduler 判断请求是否有 encoder 输入、需要多少 encoder token/embed。

## 10. `mm_receiver_cache` 的位置

在 P0/P1 或前后端分离场景中，processor 输出可能通过 receiver cache 复用或共享内存传递。

`EngineCore.preprocess_add_request()` 会在构造 `Request` 前处理 `mm_receiver_cache`。

位置：`code/vllm/vllm/v1/engine/core.py:853`。

相关缓存实现：

- receiver cache：`code/vllm/vllm/multimodal/cache.py:589`
- SHM receiver cache：`code/vllm/vllm/multimodal/cache.py:678`

## 11. 为什么要按位置排序

processor 输出的多模态 item 可能按 modality 分组，但 runtime 调度需要按 prompt token offset 处理。

例如 prompt：

```text
text <image1> text <audio1> text <image2>
```

scheduler 关心的是 token window 何时遇到 `<image1>`、`<audio1>`、`<image2>`，所以必须按 placeholder offset 排序。

排序函数：`code/vllm/vllm/multimodal/utils.py:137`。

## 12. 一句话总结

V1 `InputProcessor` 把 processor 输出的 `mm_kwargs + mm_placeholders + mm_hashes` 重新组织成按 prompt 位置排序的 `MultiModalFeatureSpec`，并挂入 `Request.mm_features`；从此以后，scheduler 和 worker 不再关心原始 prompt 字段，而是围绕 `mm_features` 进行多模态调度、缓存和执行。
