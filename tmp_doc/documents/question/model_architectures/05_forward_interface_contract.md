# 05. ModelRunner 对 model forward 有什么接口约定？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/forward_context.py`

本问题关注：不同模型 class 如何遵守统一 forward 接口，使 ModelRunner 可以用同一条执行链路调用。

---

## 1. 一句话回答

ModelRunner 不希望理解每个模型的内部结构，因此 model class 必须提供统一的 forward 约定。

```text
input_ids / inputs_embeds
positions
intermediate_tensors
**model_kwargs
  → hidden_states / intermediate_tensors / pooler output
```

---

## 2. forward 输入占位

```text
- input_ids；
- inputs_embeds；
- positions；
- intermediate_tensors；
- encoder_outputs；
- multimodal kwargs；
- token_type_ids；
- lora / prompt adapter context；
- forward context 中的 attention metadata。
```

---

## 3. forward 输出占位

```text
普通 generation：
  hidden states。

PP 非最后 rank：
  intermediate_tensors。

pooling：
  hidden states 后接 pooler。

encoder-decoder：
  可能携带 encoder outputs。
```

---

## 4. 额外接口占位

```text
- compute_logits；
- load_weights；
- get_input_embeddings；
- make_empty_intermediate_tensors；
- pooler；
- supports_lora / supports_multimodal 等能力标记。
```

---

## 5. 一句话总结

```text
forward interface contract 让 ModelRunner 可以把不同模型当作同一类执行对象调用。
```
