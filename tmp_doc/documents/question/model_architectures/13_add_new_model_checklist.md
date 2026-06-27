# 13. 新增一个模型架构需要实现什么？

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/models/registry.py`
- `code/vllm/vllm/model_executor/models/interfaces.py`
- `code/vllm/vllm/model_executor/model_loader/`
- `code/vllm/vllm/tests/`

本问题关注：向 vLLM 新增一个模型架构时，需要实现哪些文件、接口和验证。

---

## 1. 一句话回答

新增模型架构的核心是：实现一个符合 vLLM forward / load_weights 约定的 model class，并在 registry 中把 HF architecture 映射到它。

---

## 2. 实现清单占位

```text
1. 确认 HF config architecture name。
2. 在 registry 注册 model class。
3. 实现 model backbone / layer / attention / MLP。
4. 接入 vLLM Attention layer。
5. 实现 forward 接口。
6. 实现 compute_logits 或 pooler。
7. 实现 load_weights。
8. 处理 TP / PP / quant / LoRA hooks。
9. 处理 RoPE / sliding window / MLA / MoE 等特殊结构。
10. 添加测试和最小推理验证。
```

---

## 3. forward 检查占位

```text
- input_ids / inputs_embeds；
- positions；
- intermediate_tensors；
- attention metadata 从 forward_context 获取；
- PP 下返回 intermediate tensors；
- generation 下返回 hidden states；
- pooling 下可接 pooler。
```

---

## 4. weight loading 检查占位

```text
- checkpoint name mapping；
- fused QKV / gate_up；
- TP shard；
- tied weights；
- quantized weights；
- missing / unexpected weights；
- safetensors shard。
```

---

## 5. 测试占位

```text
- model load；
- one-step forward；
- generate 对齐 HF 或参考输出；
- TP / PP；
- quantization；
- LoRA；
- long context；
- pooling / multimodal 特殊任务。
```

---

## 6. 一句话总结

```text
新增模型不是只写 forward，还要接入 registry、load_weights、vLLM layers、并行、量化和测试。
```
