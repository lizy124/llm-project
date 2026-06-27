# 04. 多模态 placeholder 如何和 prompt token 对齐？

源码位置：

- `code/vllm/vllm/multimodal/inputs.py`
- `code/vllm/vllm/multimodal/processing.py`
- `code/vllm/vllm/renderers/`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/model_executor/models/`

本问题关注：多模态输入不是简单附加在 prompt 后面，而是需要在文本 token 序列中占位。placeholder token / placeholder range 如何表达多模态内容在 prompt 中的位置，是理解多模态 forward 的关键。

---

## 1. 一句话回答占位

```text
placeholder 用 token span 记录多模态内容在 prompt 中的位置，
processor 生成的 feature 会在 ModelRunner / model forward 阶段替换或合并到这些位置对应的 embedding。
```

---

## 2. placeholder 解决什么问题占位

```text
- 文本 token 序列需要保留 image/audio/video 的位置；
- 模型需要知道哪些 token 是普通文本，哪些 token 对应多模态 embedding；
- attention / position / logits 仍然以统一 token 序列为基础；
- 多个 media item 需要分别对应不同 token span。
```

---

## 3. 主链路占位

```text
chat template / prompt rendering
  → 插入 image/audio/video placeholder 文本或 token
  → tokenizer 得到 prompt_token_ids
  → placeholder ranges 记录位置
  → MultiModalFeatureSpec 绑定 feature 与位置
  → ModelRunner 合并 embeddings
```

---

## 4. 需要解释的问题占位

```text
- placeholder token 数如何确定？
- image token span 和 feature shape 如何对齐？
- prompt 中多个 image 如何区分？
- placeholder 与 mm_req_doc_ranges / attention metadata 是否有关？
- prompt_embeds 混合输入如何表达 token / embed 位置？
```

---

## 5. 后续待补源码证据

占位：补充 placeholder range、mm_placeholders、模型 merge multimodal embeddings 的源码位置。
