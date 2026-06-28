# 09. 不同算子 backend 如何选择和回退？

源码位置：

- `vllm/vllm/platforms/`
- `vllm/vllm/attention/backends/`
- `vllm/vllm/model_executor/layers/quantization/`
- `vllm/vllm/model_executor/layers/fused_moe/`

这个问题关注：vLLM 如何根据平台、硬件能力、dtype、shape、配置、已安装依赖选择 CUDA extension、Triton、FlashAttention、FlashInfer、CUTLASS 或 torch fallback。

---

## 1. 一句话回答

backend selection 决定“同一个上层算子最后由哪个具体实现执行”。

可以先记成：

```text
operator request
  → capability / config / dtype / shape check
  → preferred backend
  → fallback backend
  → error or output
```

---

## 2. 本文占位目标

后续补全文档时，本章需要展开：

```text
1. attention backend selection 的判断维度；
2. quantization backend selection 的判断维度；
3. MoE backend selection 的判断维度；
4. CUDA / ROCm / CPU / XPU 等平台差异；
5. dtype、head size、block size、group size、batch shape 如何触发 fallback；
6. 如何从日志或 profiler 确认实际 backend；
7. fallback 是正常兼容还是性能问题。
```

---

## 3. 需要串起来的主线

```text
config / platform / tensor shape
  → backend capability check
  → selected implementation
  → kernel execution or fallback
```

---

## 4. 后续补充重点

```text
- current_platform；
- env flags；
- optional dependency import；
- unsupported dtype / shape；
- fallback warning 和性能诊断。
```
