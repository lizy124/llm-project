# 06. MoE fused 算子如何执行 expert routing？

源码位置：

- `vllm/vllm/model_executor/layers/fused_moe/`
- `vllm/vllm/model_executor/layers/quantization/`
- `vllm/csrc/moe/`

这个问题关注：MoE 的 routing、top-k、token dispatch、grouped GEMM、expert combine、quantization、expert parallel 等环节如何通过 fused kernel 或专用 backend 高效执行。

---

## 1. 一句话回答

Fused MoE kernel 负责把 token 按 router 结果分发给 expert，批量执行 expert MLP，再把结果合并回原 token 顺序。

最小链路是：

```text
hidden states
  → router logits / top-k
  → token dispatch / expert grouping
  → grouped GEMM / fused expert compute
  → combine / reduce
  → output hidden states
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. MoE layer forward 的主流程；
2. top-k routing 和 expert id / weight 如何进入 kernel；
3. token 重排、padding、grouped GEMM 如何组织；
4. fused MoE 与 unfused fallback 的差异；
5. MoE quantization 如何改变 weight 和 kernel；
6. expert parallel / tensor parallel 如何影响 dispatch 和 combine；
7. 常见性能瓶颈和 debug 方法。
```

---

## 3. 需要串起来的主线

```text
MoE model layer
  → router
  → fused_moe wrapper
  → backend kernel
  → combine output
```

---

## 4. 后续补充重点

```text
- expert map / top-k ids / top-k weights；
- grouped GEMM layout；
- EP 通信边界；
- quantized MoE；
- token imbalance 与性能。
```
