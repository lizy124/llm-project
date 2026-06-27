# 08. LoRA 如何与量化 base model 共存？

源码位置：

- `code/vllm/vllm/lora/`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：量化 base model 与 LoRA adapter 同时启用时，forward 和权重管理如何处理。

---

## 1. 一句话回答

LoRA + 量化通常表示 base model 权重是量化的，而 LoRA adapter 作为额外低秩权重在 forward 时叠加到 base output。

```text
quantized base output
  + LoRA delta
  → final output
```

---

## 2. 需要梳理的问题

```text
- quantized Linear 是否支持 LoRA；
- LoRA delta 用什么 dtype；
- base output 和 LoRA output 如何对齐 dtype；
- LoRA 权重是否也支持量化；
- AWQ / GPTQ / FP8 / Marlin 等量化方式与 LoRA 的兼容性；
- CUDA graph / compile 是否支持动态 LoRA；
- 性能开销如何估计。
```

---

## 3. forward 占位

```text
x
  → quantized base linear
  → base_output

x
  → LoRA A/B
  → lora_delta

base_output + lora_delta
  → next layer
```

---

## 4. 容易混淆点占位

```text
1. base model 量化不代表 LoRA adapter 也量化。
2. 某量化方式支持普通 forward，不代表支持 LoRA。
3. LoRA rank 和 adapter 数量会额外占显存。
4. dtype 转换可能影响性能和精度。
```

---

## 5. 一句话总结

```text
LoRA 与量化共存的核心，是让低 bit base layer 和高效 LoRA delta 在同一次 forward 中正确相加。
```
