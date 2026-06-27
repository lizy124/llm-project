# 05. Weight-only 量化如何工作？

源码位置：

- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/fused_moe/`

本问题关注：GPTQ、AWQ、Marlin、INT4、INT8、FP8 等主要权重量化路径如何接入 vLLM。

---

## 1. 一句话回答

Weight-only 量化主要压缩权重，forward 时使用低 bit 权重和 scale / zero point 恢复近似计算，activation 通常仍以 fp16/bf16/fp32 等格式输入。

---

## 2. 常见方式占位

```text
GPTQ：
  离线量化权重，通常包含 qweight / scales / qzeros / g_idx。

AWQ：
  activation-aware 权重量化，常有 group scales。

Marlin：
  面向特定 GPU kernel 的 packed INT4/INT8 权重格式。

FP8 weight：
  使用 FP8 保存权重和 scale。

bitsandbytes：
  可能走独立加载和 kernel 路径。
```

---

## 3. 需要梳理的问题

```text
- 权重是离线量化还是加载时量化；
- scale 是 per-tensor、per-channel 还是 per-group；
- zero point 是否存在；
- group size 如何限制 hidden size；
- 是否需要 packed layout；
- 是否支持 MoE；
- 是否支持 tensor parallel 切分；
- 是否支持 CUDA graph / torch.compile。
```

---

## 4. kernel 关系占位

```text
不同 weight-only 量化格式经常绑定特定 kernel：

- 通用 torch fallback；
- custom CUDA op；
- Marlin kernel；
- Cutlass / Triton kernel；
- fused MoE kernel。
```

---

## 5. 一句话总结

```text
Weight-only 量化的难点不只是权重变小，而是低 bit 权重格式必须和加载器、TP 切分和 kernel 布局匹配。
```
