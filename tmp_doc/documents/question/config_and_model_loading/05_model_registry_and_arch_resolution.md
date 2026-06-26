# 05. model registry 如何解析模型类？

源码位置：

- `code/vllm/vllm/model_executor/models/registry.py`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`

本问题关注：vLLM 如何根据 Hugging Face config 中的 `architectures`，找到对应的 vLLM 模型实现类，并判断该模型支持 generation、pooling、embedding、多模态等能力。

---

## 1. 一句话回答占位

占位：后续补充 model registry 是“HF architecture → vLLM model class”的解析层。

```text
HF config.architectures
  → ModelRegistry.resolve_model_cls()
  → vLLM model class
  → initialize_model()
  → model instance
```

---

## 2. registry 需要回答什么占位

```text
- 这个 architecture 在 vLLM 中有没有实现？
- 对应哪个 Python class？
- 是文本生成模型、embedding 模型、pooling 模型，还是多模态模型？
- 是否支持跨任务 runner？
- 是否需要 trust_remote_code 或特殊包装？
- 是否支持当前 vLLM V1 路径？
```

---

## 3. 模型类和执行接口占位

后续补充：

```text
- vLLM 模型类通常如何定义 forward；
- 如何暴露 compute_logits；
- pooling 模型如何暴露 pooler；
- 多模态模型如何接收 inputs_embeds / multimodal kwargs；
- 模型类如何声明 packed modules / LoRA 支持 / quantization method。
```

---

## 4. 后续待补源码证据

占位：补充 `_ModelRegistry`、`resolve_model_cls()`、模型能力判断函数、典型模型类示例。
