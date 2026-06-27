# 12. Quantization、LoRA、Parallelism 如何 hook 到模型架构？

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/lora/`
- `code/vllm/vllm/distributed/`

本问题关注：模型架构如何为量化、LoRA 和并行机制提供扩展点。

---

## 1. 一句话回答

模型架构不是独立实现所有功能，而是在构造 layer 和加载权重时把 quantization、LoRA、TP/PP/EP 等 hook 接入进来。

---

## 2. quantization hook 占位

```text
- layer 创建时传 quant_config；
- quant_method.create_weights；
- weight_loader 加载 qweight / scales；
- forward 调 quant_method.apply；
- MoE / attention / lm_head 特殊支持。
```

---

## 3. LoRA hook 占位

```text
- target modules 是否支持 LoRA；
- fused layer 如何拆分 LoRA target；
- LoRA wrapper 如何替换 layer；
- embedding / lm_head LoRA；
- active LoRA mapping 如何作用到 forward。
```

---

## 4. parallelism hook 占位

```text
- ColumnParallelLinear / RowParallelLinear；
- QKVParallelLinear；
- vocab parallel embedding；
- pipeline parallel layer partition；
- expert parallel MoE；
- intermediate tensors。
```

---

## 5. 一句话总结

```text
模型架构层要把模型结构和 vLLM 的量化、LoRA、并行执行能力接起来。
```
