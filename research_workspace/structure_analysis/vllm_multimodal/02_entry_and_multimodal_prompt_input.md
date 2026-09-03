# 02 入口与 Prompt 多模态输入

本篇梳理多模态数据如何从 OpenAI Chat 或 Python API 进入 vLLM prompt。入口层的目标不是直接运行 encoder，而是把外部协议中的图片、音频、视频等内容转换成统一的 prompt 字段：`multi_modal_data`、`multi_modal_uuids`、`mm_processor_kwargs`。

## 1. OpenAI Chat 输入链路

OpenAI 兼容 chat serving 类：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:108`。

请求处理过程中会进入 `render_chat(request)`，调用位置：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:233`。

HF renderer 的 message 渲染入口：`code/vllm/vllm/renderers/hf.py:885`。

简化链路：

```text
OpenAI /v1/chat/completions request
  ↓
OpenAIServingChat
  ↓
render_chat(request)
  ↓
HF renderer render_messages()
  ↓
parse_chat_messages()
  ↓
conversation + mm_data + mm_uuids
  ↓
prompt["multi_modal_data"] = mm_data
prompt["multi_modal_uuids"] = mm_uuids
```

关键位置：

- `render_messages()` 调用 `parse_chat_messages()`：`code/vllm/vllm/renderers/hf.py:899`
- `parse_chat_messages()` 定义：`code/vllm/vllm/entrypoints/chat_utils.py:1863`
- 写入 prompt multi_modal_data：`code/vllm/vllm/renderers/hf.py:985`
- 写入 prompt multi_modal_uuids：`code/vllm/vllm/renderers/hf.py:987`

## 2. `parse_chat_messages()` 的职责

`parse_chat_messages()` 负责把 OpenAI 风格 messages 转成 vLLM 内部 conversation，同时收集多模态项。

它做三件事：

1. 解析文本消息，形成 chat template 所需 conversation；
2. 用 `MultiModalItemTracker` 收集 image/audio/video 等多模态项；
3. 返回 `mm_data` 与 `mm_uuids`。

可以理解为：

```text
messages = [
  {role: user, content: [
    {type: text, text: ...},
    {type: image_url, image_url: ...},
  ]}
]
  ↓
conversation: 给 chat template 的文本/占位内容
mm_data: 真正的多模态数据
mm_uuids: 可选用户指定或系统生成的多模态身份
```

入口位置：`code/vllm/vllm/entrypoints/chat_utils.py:1863`。

## 3. renderer 层把多模态挂到 prompt

renderer 不是直接调用模型，而是构造 prompt dict。

在 HF renderer 中：

```text
prompt["multi_modal_data"] = mm_data
prompt["multi_modal_uuids"] = mm_uuids
```

相关位置：`code/vllm/vllm/renderers/hf.py:985`。

这一步的意义是：后续 `InputPreprocessor` 不需要知道原始请求来自 OpenAI Chat、Responses API 还是其他入口，只要看到 prompt 中的 `multi_modal_data` 就进入统一多模态预处理。

## 4. Python Prompt 输入 schema

Python API 的 prompt schema 在 `code/vllm/vllm/inputs/llm.py`。

关键定义：

- `MultiModalDataBuiltins`：`code/vllm/vllm/inputs/llm.py:29`
- `MultiModalDataDict`：`code/vllm/vllm/inputs/llm.py:45`
- `_PromptOptions.multi_modal_data`：`code/vllm/vllm/inputs/llm.py:70`
- `_PromptOptions.mm_processor_kwargs`：`code/vllm/vllm/inputs/llm.py:76`
- `_PromptOptions.multi_modal_uuids`：`code/vllm/vllm/inputs/llm.py:84`
- `DataPrompt`：`code/vllm/vllm/inputs/llm.py:223`

这说明 Python API 中无论是 text prompt、tokens prompt 还是 data prompt，都可以携带多模态字段。

典型结构可以理解为：

```python
{
  "prompt": "<image> describe this",
  "multi_modal_data": {
    "image": image_object,
  },
  "multi_modal_uuids": {
    "image": ["optional-stable-id"],
  },
  "mm_processor_kwargs": {
    "some_processor_arg": "...",
  },
}
```

## 5. `multi_modal_data` 的语义

`multi_modal_data` 是按 modality 组织的原始输入字典。常见 key：

- `image`
- `audio`
- `video`
- `image_embeds`
- 其他模型/processor 支持的 embedding-only 输入

它可以包含：

- 单个 item；
- item list；
- URL / data URL；
- bytes；
- PIL image；
- numpy / torch tensor；
- 已经预处理过的 embedding。

真正支持哪些形态，由 `MultiModalDataParser`、media connector 和具体模型 processor 决定。

## 6. `multi_modal_uuids` 的语义

`multi_modal_uuids` 是用户可选传入的多模态身份标识。

作用：

- 避免对大图片/音频重复计算内容 hash；
- 让不同请求中的同一媒体可复用 processor cache 或 prefix/encoder 相关缓存；
- 在一些 cache 关闭场景下可被系统替换为 request-local 标识。

UUID 解析入口：`code/vllm/vllm/multimodal/parse.py:714`。

renderer 侧 UUID 校验：`code/vllm/vllm/renderers/base.py:617`。

需要注意：UUID 数量必须和对应 modality 的 item 数量匹配，否则会报错。

## 7. `mm_processor_kwargs` 的语义

`mm_processor_kwargs` 是传给多模态 processor / HF processor 的请求级额外参数。

它会和配置层 `MultiModalConfig` 中的 processor kwargs 合并。

合并接口：`code/vllm/vllm/config/multimodal.py:326`。

用途包括：

- 控制图像 resize/crop；
- 控制视频帧采样；
- 控制音频特征处理；
- 给模型特定 processor 传附加参数。

## 8. 统一预处理入口

所有 prompt 最终会进入 `InputPreprocessor.preprocess()`。

入口：`code/vllm/vllm/inputs/preprocess.py:274`。

如果 prompt 带有 `multi_modal_data`，会进入 `_process_multimodal()`：`code/vllm/vllm/inputs/preprocess.py:90`。

文本和 tokens 两种路径也会在需要时下沉到多模态处理：

- `_process_tokens()`：`code/vllm/vllm/inputs/preprocess.py:133`
- `_process_text()`：`code/vllm/vllm/inputs/preprocess.py:161`

这意味着多模态预处理不限定输入必须是纯文本 prompt；token prompt 也可以带多模态数据，只是 placeholder 匹配方式会不同。

## 9. `MultiModalInput` 的目标形态

engine 输入结构在 `code/vllm/vllm/inputs/engine.py`：

- `MultiModalInput`：`code/vllm/vllm/inputs/engine.py:126`
- `mm_input` 字段：`code/vllm/vllm/inputs/engine.py:151`
- encoder-decoder 多模态输入：`code/vllm/vllm/inputs/engine.py:176`

processor 最终要构造的不是“原始图片”，而是：

```text
prompt_token_ids
mm_kwargs
mm_hashes
mm_placeholders
```

后续 V1 engine 会把这组信息转换成 `MultiModalFeatureSpec`。

## 10. 入口层和 processor 层的边界

入口层只负责把外部协议变成统一 prompt 字段，通常不做：

- 图像 patchify；
- 音频 feature extraction；
- 视频 frame tensor 构造；
- placeholder 长度最终确定；
- encoder cache 分配。

这些都在 parser/processor/engine/runtime 层完成。

## 11. 常见问题定位

### 11.1 OpenAI 请求里的图片没有进入模型

检查：

```text
OpenAI content type 是否被 parse_chat_messages 识别
renderer 是否写入 prompt["multi_modal_data"]
URL/data URL 是否可被 MediaConnector 加载
limit_mm_per_prompt 是否允许该 modality
processor 是否支持该模型的 image 输入
```

### 11.2 Python API 传了 image 但报缺 placeholder

检查：

```text
prompt 文本中是否有模型要求的 image placeholder
processor 是否会自动插入 placeholder
chat template 是否已经插入占位
PromptReplacement/PromptInsertion 是否匹配成功
```

### 11.3 UUID 数量报错

检查：

```text
multi_modal_uuids[modality] 的长度
multi_modal_data[modality] 的 item 数量
是否存在 None/空 item
processor cache / prefix caching 是否关闭导致 UUID 被忽略
```

相关校验：`code/vllm/vllm/renderers/base.py:617`。

## 12. 一句话总结

入口层把 OpenAI/Python 的多模态输入统一挂到 prompt 上，形成 `multi_modal_data`、`multi_modal_uuids`、`mm_processor_kwargs`；真正的媒体读取、tensor 化、prompt 占位更新和 runtime feature 构造在后续 parser、processor 和 V1 engine 中完成。
