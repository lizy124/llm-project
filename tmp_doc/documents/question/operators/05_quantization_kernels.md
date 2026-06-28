# 05. 量化算子如何参与权重加载和 forward？

源码位置：

- `vllm/vllm/model_executor/layers/quantization/`
- `vllm/vllm/model_executor/layers/linear.py`
- `vllm/csrc/quantization/`

这个问题关注：GPTQ、AWQ、FP8、INT8、Marlin、CUTLASS、scaled mm 等量化算子如何替代普通 dense linear，并在 forward 时使用量化权重、scale、zero point、group metadata。

---

## 1. 一句话回答

量化算子把普通矩阵乘替换为带量化权重和缩放信息的专用 kernel，用更低显存和更高吞吐完成 linear / MoE 等计算。

最小链路是：

```text
LoadConfig / QuantConfig
  → quantization method
  → quantized weight loading
  → quantized linear layer
  → backend-specific matmul kernel
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. quantization config 如何选择 layer method；
2. 权重加载时如何保存 packed weight / scales / zero points；
3. quantized linear forward 如何调用 kernel；
4. GPTQ / AWQ / FP8 / INT8 / Marlin / CUTLASS 路径差异；
5. activation quantization 和 weight-only quantization 的差异；
6. LoRA / TP / MoE 与量化算子的组合限制；
7. 精度、性能和 fallback 排查。
```

---

## 3. 需要串起来的主线

```text
model config
  → quant config
  → layer replacement
  → weight loading
  → quantized matmul op
  → output activation
```

---

## 4. 后续补充重点

```text
- scale / zero point / group size；
- packed weight layout；
- dtype 与 hardware capability；
- Marlin / CUTLASS / Triton / torch fallback；
- quantized KV cache 与 attention 交互。
```
