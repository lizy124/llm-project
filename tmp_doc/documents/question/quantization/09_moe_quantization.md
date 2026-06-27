# 09. MoE / Fused MoE 量化如何处理？

源码位置：

- `code/vllm/vllm/model_executor/layers/fused_moe/`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/models/`

本问题关注：Mixture-of-Experts 模型中 expert 权重如何量化，以及 fused MoE kernel 如何消费量化参数。

---

## 1. 一句话回答

MoE 量化比普通 Linear 更复杂，因为它不仅有低 bit expert 权重，还要处理 routing、top-k experts、expert parallel、fused kernel 和 expert 权重布局。

---

## 2. MoE 量化对象占位

```text
- gate / router 是否量化；
- expert up / gate / down projection 权重；
- fused expert weight；
- per-expert scale / zero point；
- group size；
- expert parallel 下的 expert 切分。
```

---

## 3. fused MoE kernel 占位

```text
hidden states
  → router logits
  → top-k experts
  → expert weight quantized GEMM
  → combine expert outputs
```

需要补充：

```text
- 哪些量化方法支持 fused MoE；
- 哪些需要 fallback；
- expert 权重如何 pack；
- scales 是否按 expert 维度组织；
- EP / TP 下如何切分。
```

---

## 4. 容易混淆点占位

```text
1. 普通 Linear 支持某量化方式，不代表 fused MoE 也支持。
2. MoE 权重量化还要考虑 expert 维度。
3. routing 本身通常仍是高精度或单独路径。
4. expert parallel 会影响量化权重加载和通信。
```

---

## 5. 一句话总结

```text
MoE 量化的核心，是让低 bit expert 权重、routing 和 fused MoE kernel 的布局完全对齐。
```
