# 09. 多模态模型类需要提供什么接口？

源码位置：

- `code/vllm/vllm/model_executor/models/registry.py`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/multimodal/registry.py`
- `code/vllm/vllm/config/model.py`

本问题关注：vLLM 如何识别一个模型是多模态模型，多模态模型类需要提供哪些接口，ModelRunner 如何通过这些接口执行 encoder、获取 embeddings、合并 embeddings 并继续 decoder forward。

---

## 1. 一句话回答占位

```text
多模态模型类除了普通 forward / compute_logits，还需要声明支持的 modality，并提供把 processor feature 转成 embedding、再合并到文本 embedding 的接口。
```

---

## 2. 需要关注的模型能力占位

```text
- 是否是 multimodal model；
- 支持哪些 modality：image / audio / video；
- 是否需要 encoder；
- placeholder token 数如何计算；
- 是否支持 pooling / embedding；
- 是否支持 LoRA / quantization；
- 是否支持 V1 runner；
- 是否需要特殊 processor / mapper。
```

---

## 3. 常见接口占位

后续补充具体源码名称：

```text
- get_language_model；
- get_multimodal_embeddings；
- get_input_embeddings；
- merge_multimodal_embeddings；
- get_mm_mapping / get_supported_mm_limits；
- load_weights；
- forward；
- compute_logits；
- pooler。
```

---

## 4. model registry 关系占位

```text
HF config.architectures
  → ModelRegistry.resolve_model_cls()
  → 判断是否 multimodal
  → 初始化模型类
  → 注册 multimodal processor / mapper
  → ModelRunner 通过统一接口调用
```

---

## 5. 后续待补源码证据

占位：补充 registry 判断多模态能力、典型模型类如 LLaVA / QwenVL / Phi-3-Vision 等接口实现。
