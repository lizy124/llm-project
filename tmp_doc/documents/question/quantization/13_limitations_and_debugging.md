# 13. 量化有哪些限制和调试入口？

源码位置：

- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/model_loader/`
- `code/vllm/vllm/platforms/`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/config.py`

本问题关注：量化常见不支持场景、错误信息、fallback 和调试路径。

---

## 1. 一句话回答

量化问题通常不是单点 bug，而是配置、checkpoint 格式、kernel 能力、shape 约束和平台支持之间不匹配。

---

## 2. 常见限制占位

```text
- 当前 GPU 架构不支持某 kernel；
- hidden size / head size / group size 不满足要求；
- checkpoint 缺少 scale / zero point；
- quantization method 和 load format 不匹配；
- MoE / LoRA / PP / TP 组合不支持；
- attention backend 不支持当前 KV cache dtype；
- CUDA graph / torch.compile 不支持某动态路径；
- structured output / spec decode 与某些 logits processor 不兼容。
```

---

## 3. 调试入口占位

```text
1. 打印 VllmConfig / ModelConfig / CacheConfig。
2. 确认 quantization method。
3. 确认 checkpoint 文件里有哪些 tensor。
4. 检查 layer.quant_method 类型。
5. 检查 weight_loader 是否命中对应参数。
6. 检查 attention backend 选择日志。
7. 检查 kernel fallback / warning。
8. 用小 batch 对比 FP16 输出。
```

---

## 4. 常见报错分类占位

```text
配置错误：
  unknown quantization method / unsupported dtype。

加载错误：
  missing qweight / scales / qzeros。

shape 错误：
  hidden size 不可被 group size 或 TP size 整除。

kernel 错误：
  no supported kernel / illegal memory access / unsupported arch。

精度问题：
  输出异常、NaN、logits 分布漂移。
```

---

## 5. 一句话总结

```text
调试量化问题时，要沿着“配置 → checkpoint → layer 参数 → kernel → runtime 输出”逐层定位。
```
