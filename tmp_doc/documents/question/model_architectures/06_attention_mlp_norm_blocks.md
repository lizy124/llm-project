# 06. Attention / MLP / Norm blocks 如何在模型中复用？

源码位置：

- `code/vllm/vllm/model_executor/layers/attention/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/layernorm.py`
- `code/vllm/vllm/model_executor/layers/rotary_embedding.py`
- `code/vllm/vllm/model_executor/models/`

本问题关注：不同模型如何复用 vLLM 的基础 layer 组件。

---

## 1. 一句话回答

vLLM model class 通常不是从零实现所有算子，而是组合复用 Attention、Linear、MLP、Norm、RoPE 等基础组件。

---

## 2. Attention block 占位

```text
- num_heads / num_kv_heads；
- head_size；
- qkv projection；
- rotary embedding；
- Attention layer；
- output projection；
- sliding window / MLA / cross attention 特例。
```

---

## 3. MLP block 占位

```text
- gate_proj；
- up_proj；
- down_proj；
- fused gate_up_proj；
- activation function；
- quant_method。
```

---

## 4. Norm / position 占位

```text
- RMSNorm；
- LayerNorm；
- pre-norm / post-norm；
- RoPE；
- ALiBi；
- M-RoPE / XD-RoPE；
- sliding window position。
```

---

## 5. 一句话总结

```text
模型架构差异大多体现在如何组合这些基础 blocks，而不是每个模型重写所有底层算子。
```
