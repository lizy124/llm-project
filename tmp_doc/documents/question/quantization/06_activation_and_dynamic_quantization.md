# 06. Activation / Dynamic quantization 如何参与 forward？

源码位置：

- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/_custom_ops.py`

本问题关注：activation quantization、dynamic quantization、per-token scaling 等运行时量化如何参与 forward。

---

## 1. 一句话回答

Activation / dynamic quantization 不是只改变权重存储，而是在 forward 时根据输入 activation 动态计算 scale 或转换 dtype，再调用对应 kernel。

---

## 2. 与 weight-only 的区别

```text
weight-only：
  权重离线或加载时量化，activation 通常保持高精度。

activation quant：
  输入 activation 也会被量化，可能需要 runtime scale。

dynamic quant：
  scale 可能每个 batch、每个 token 或每个 channel 动态计算。
```

---

## 3. 需要梳理的问题

```text
- activation scale 在哪里计算；
- scale 是 per-token、per-tensor 还是 per-channel；
- 是否需要 smooth quant；
- output 是否需要 dequant；
- dynamic quant 和 CUDA graph 是否兼容；
- dynamic quant 对延迟的额外开销；
- 哪些 kernel 支持 activation quant。
```

---

## 4. forward 占位

```text
activation
  → calculate scale
  → quantize activation
  → quantized GEMM
  → dequant / output scale
  → residual / layer norm / next layer
```

---

## 5. 一句话总结

```text
Activation quantization 把量化从“加载时问题”变成“每次 forward 的运行时问题”。
```
