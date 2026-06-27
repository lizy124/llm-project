# 10. 多模态模型架构如何接入？

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/multimodal/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/model_executor/layers/rotary_embedding.py`

本问题关注：vision-language / audio-language 等多模态模型如何把非文本特征接入语言模型 backbone。

---

## 1. 一句话回答

多模态模型通常通过 modality encoder 和 projector 把图像 / 音频等特征转换成可与 token embedding 合并的 hidden states，再进入语言模型 backbone。

---

## 2. 典型结构占位

```text
text tokens
  → token embeddings

image / audio inputs
  → modality encoder
  → projector
  → multimodal embeddings

merge
  → inputs_embeds
  → language model forward
```

---

## 3. 需要梳理的问题

```text
- placeholder tokens 如何标记多模态位置；
- mm processor 如何生成 mm features；
- encoder cache 如何复用；
- projector 如何接入 model class；
- inputs_embeds 如何替代 input_ids；
- M-RoPE / XD-RoPE positions；
- multimodal 模型的 load_weights 差异。
```

---

## 4. 和 ModelRunner 的关系

```text
ModelRunner._preprocess()
  → 执行或读取 multimodal encoder 输出
  → gather multimodal embeddings
  → merge 到 inputs_embeds
  → model.forward(inputs_embeds=...)
```

---

## 5. 一句话总结

```text
多模态模型架构的核心，是把非文本 encoder 输出折叠进语言模型可消费的 inputs_embeds。
```
