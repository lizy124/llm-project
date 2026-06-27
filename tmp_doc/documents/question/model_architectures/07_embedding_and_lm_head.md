# 07. Embedding、LM head 和 logits processor 如何接入？

源码位置：

- `code/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`
- `code/vllm/vllm/model_executor/layers/logits_processor.py`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：输入 embedding、输出 lm_head、tie weights 和 logits 计算如何在模型架构中接入。

---

## 1. 一句话回答

Embedding 和 LM head 是模型架构与 sampling/output 链路的接口：前者把 token ids 变成 hidden states，后者把 hidden states 变成 vocab logits。

---

## 2. input embedding 占位

```text
input_ids
  → vocab parallel embedding
  → hidden states
```

需要补充：

```text
- vocab parallel 切分；
- padding token；
- tied embeddings；
- inputs_embeds / prompt_embeds / multimodal embeddings 的覆盖；
- LoRA / prompt adapter 对 embedding 的影响。
```

---

## 3. LM head 占位

```text
hidden_states[logits_indices]
  → lm_head
  → logits processor
  → logits
  → sampler
```

需要补充：

```text
- tie word embeddings；
- vocab parallel logits gather；
- logits soft cap；
- final logits dtype；
- quantized lm_head。
```

---

## 4. 一句话总结

```text
Embedding 是模型输入口，LM head 是生成式模型接入 sampler 的输出口。
```
