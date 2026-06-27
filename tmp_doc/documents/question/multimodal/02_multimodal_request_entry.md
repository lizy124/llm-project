# 02. 多模态用户输入如何进入 vLLM？

源码位置：

- `code/vllm/vllm/entrypoints/`
- `code/vllm/vllm/entrypoints/openai/`
- `code/vllm/vllm/renderers/`
- `code/vllm/vllm/inputs/`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/engine/__init__.py`

本问题关注：用户通过 OpenAI-compatible API、offline LLM 或 EngineInput 提供 image / audio / video / prompt_embeds 后，这些输入如何被 renderer / InputProcessor 转成 `EngineCoreRequest.mm_features` 和 prompt token 结构。

---

## 1. 一句话回答占位

```text
外部多模态请求先被 entrypoints / renderer 解析成 EngineInput，
再由 InputProcessor 转成 EngineCoreRequest，
其中多模态数据进入 mm_features，文本与 placeholder 进入 prompt_token_ids。
```

---

## 2. 输入来源占位

```text
OpenAI Chat Completions：
  messages content parts 里包含 image_url / input_audio / video 等。

OpenAI Responses / Embeddings：
  可能携带多模态内容或 embedding / pooling 输入。

Offline LLM：
  PromptType / EngineInput 中直接携带 multi_modal_data。

Prompt embeds：
  外部直接传入 prompt_embeds，绕过部分 tokenizer / processor 路径。
```

---

## 3. EngineInput 到 EngineCoreRequest 占位

后续补充：

```text
EngineInput
  → renderer / input processor
  → prompt_token_ids
  → mm_features
  → mm_placeholders
  → EngineCoreRequest
```

需要重点说明：

```text
- prompt 文本如何被 tokenize；
- placeholder token 如何插入；
- mm data 如何保留为 feature spec；
- request.external_req_id 如何保留用户 request id；
- prompt_embeds 与 prompt_is_token_ids 如何表达混合输入。
```

---

## 4. 后续待补源码证据

占位：补充 OpenAI chat rendering、offline PromptType、InputProcessor.process_inputs、EngineCoreRequest 字段定义。
