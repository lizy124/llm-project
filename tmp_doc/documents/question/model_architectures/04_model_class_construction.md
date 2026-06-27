# 04. 一个 vLLM model class 是如何构造的？

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/layers/`
- `code/vllm/vllm/config/model.py`

本问题关注：vLLM model class 如何根据 config 构造 embedding、layers、norm、lm_head 或 pooler。

---

## 1. 一句话回答

一个 vLLM model class 通常把模型拆成可复用组件：embedding、若干 decoder / encoder layers、final norm、lm_head 或 pooler。

---

## 2. 典型结构占位

```text
ModelForCausalLM
  → model backbone
      → embeddings
      → layers[0..N]
          → attention
          → mlp / moe
          → norms
      → final norm
  → lm_head
  → logits processor / sampler 接口
```

Pooling model：

```text
ModelForPooling
  → backbone
  → pooler
```

Multimodal model：

```text
Vision tower / encoder
  → projector
  → language model backbone
```

---

## 3. 构造时要注入的配置

```text
- hidden size；
- num layers；
- num heads / num kv heads；
- intermediate size；
- rope config；
- sliding window；
- quantization config；
- prefix / cache config；
- parallel config；
- LoRA / adapter capability。
```

---

## 4. 一句话总结

```text
model class construction 是把 HF config 翻译成 vLLM layer graph 的过程。
```
