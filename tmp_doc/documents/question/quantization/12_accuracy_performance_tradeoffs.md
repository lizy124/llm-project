# 12. 量化的精度、显存和性能如何取舍？

源码位置：

- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/platforms/`

本问题关注：不同量化方式对显存、吞吐、延迟、精度和数值稳定性的影响。

---

## 1. 一句话回答

量化不是一定更快，也不是一定精度足够；它是在显存、带宽、kernel 支持、数值误差和运行时开销之间做取舍。

---

## 2. 显存收益占位

```text
权重量化：
  降低模型参数显存。

KV cache 量化：
  降低长上下文 / 大 batch 下 KV cache 显存。

activation quantization：
  可能降低中间激活或 GEMM 输入带宽，但也可能增加 scale 计算开销。
```

---

## 3. 性能收益占位

```text
可能更快：
  权重带宽瓶颈、低 bit kernel 高效、batch 足够大。

可能不快：
  kernel fallback、dynamic scale 开销、dequant 开销、shape 不适配、batch 太小。
```

---

## 4. 精度风险占位

```text
- 权重误差；
- activation clipping；
- KV cache 累积误差；
- long context 下 attention 精度；
- logits 分布改变；
- structured output / sampling 边界 token 受影响；
- MoE routing 敏感性。
```

---

## 5. 评估建议占位

```text
- 对比 FP16/BF16 baseline；
- 检查困惑度 / benchmark；
- 检查长上下文输出；
- 检查 structured output 合法率；
- 检查 throughput、TTFT、TPOT；
- 检查显存峰值和 KV cache capacity。
```

---

## 6. 一句话总结

```text
量化的价值取决于瓶颈在哪里：权重显存、KV cache 显存、内存带宽还是 kernel 吞吐。
```
