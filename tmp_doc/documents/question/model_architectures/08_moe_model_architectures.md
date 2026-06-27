# 08. MoE 模型架构如何组织？

源码位置：

- `code/vllm/vllm/model_executor/layers/fused_moe/`
- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/distributed/`

本问题关注：MoE 模型中 router、experts、fused MoE、shared experts 和 expert parallel 如何接入 vLLM 模型架构。

---

## 1. 一句话回答

MoE 模型把普通 MLP 替换成 router + 多个 experts 的组合，vLLM 通常通过 fused MoE layer 和 expert parallel 来高效执行。

---

## 2. MoE block 占位

```text
hidden states
  → router / gate
  → top-k experts
  → expert MLP
  → combine outputs
```

---

## 3. 需要梳理的问题

```text
- router logits 如何计算；
- top-k expert 选择；
- shared experts；
- expert weights 如何组织；
- fused MoE kernel；
- expert parallel；
- quantized MoE；
- DeepSeek / Mixtral 等模型差异。
```

---

## 4. 和普通 MLP 的区别

```text
普通 MLP：
  每个 token 走同一套 FFN。

MoE：
  不同 token 路由到不同 experts，
  batch 内会产生 token-expert 分组和通信。
```

---

## 5. 一句话总结

```text
MoE 模型架构的核心，是把 token 路由、expert 权重和 fused kernel 组织成 vLLM 可执行的 layer。
```
