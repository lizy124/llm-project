# 07. RMSNorm、activation、RoPE 等基础算子在哪里用？

源码位置：

- `vllm/vllm/model_executor/layers/layernorm.py`
- `vllm/vllm/model_executor/layers/activation.py`
- `vllm/vllm/model_executor/layers/rotary_embedding.py`
- `vllm/vllm/_custom_ops.py`
- `vllm/csrc/`

这个问题关注：RMSNorm、LayerNorm、SiluMul、GeluAndMul、RoPE、M-RoPE 等基础算子如何支撑 Transformer block 的常见计算，并如何影响模型 forward 性能。

---

## 1. 一句话回答

基础算子负责 Transformer block 中高频但相对固定的张量变换，是 attention 和 MLP 之外的重要性能组成部分。

最小链路是：

```text
model block
  → norm / activation / rope layer
  → op wrapper
  → fused CUDA / Triton / torch fallback
  → transformed tensor
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. RMSNorm / LayerNorm 在不同模型结构中的位置；
2. SiluMul / GeluAndMul 如何服务 gated MLP；
3. RoPE / M-RoPE 如何处理 positions；
4. fused activation 为什么比拆分 torch op 更高效；
5. dtype、hidden size、contiguous layout 对 kernel 的影响；
6. fallback 和数值稳定性问题如何排查。
```

---

## 3. 需要串起来的主线

```text
model architecture block
  → norm / rope / activation module
  → custom op or torch fallback
  → next attention / MLP layer
```

---

## 4. 后续补充重点

```text
- RMSNorm residual fused path；
- gated activation fused path；
- RoPE cache / positions / scaling；
- multimodal M-RoPE；
- CUDA Graph capture 对临时 tensor 的要求。
```
