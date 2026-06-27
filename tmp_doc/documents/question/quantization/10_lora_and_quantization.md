# 10. LoRA 和量化如何共存？

源码位置：

- `code/vllm/vllm/lora/`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/quantization/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：量化 base model 与 LoRA adapter 如何共存，以及哪些组合有限制。

---

## 1. 一句话回答

LoRA + 量化通常表示：base model 权重是量化的，但 LoRA adapter 权重以单独形式加载，并在 forward 中与 base output 合并。

---

## 2. 需要梳理的问题

```text
- 量化 Linear 是否支持 LoRA 注入；
- LoRA 权重是否量化；
- LoRA output 和 quantized base output 如何合并；
- LoRA rank / dtype 是否受限制；
- 多 LoRA / LoRA hot-swap 与量化 kernel 是否兼容；
- GPU memory profile 如何计算量化 base + LoRA cache。
```

---

## 3. 主链路占位

```text
load quantized base model
  → load / register LoRA adapter
  → request 指定 LoRA
  → ModelRunner 激活当前 batch LoRA
  → quantized base linear forward
  → LoRA delta forward
  → merge output
```

---

## 4. 容易混淆点占位

```text
1. base model 量化不等于 LoRA adapter 也量化。
2. 某些量化 kernel 不支持动态 LoRA。
3. LoRA 与量化共存可能影响 CUDA graph / compile。
4. TP 下 LoRA 权重也要和 base layer 切分对齐。
```

---

## 5. 一句话总结

```text
LoRA + 量化的关键，是让低 bit base layer 和可切换 adapter 在同一个 forward 中正确叠加。
```
