# 10. 哪些场景会导致 cudagraph fallback？

源码位置：

- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/lora/`
- `code/vllm/vllm/model_executor/layers/quantization/`

本问题关注：哪些功能或输入形态不适合 CUDA graph / compile，以及 fallback 如何保持正确性。

---

## 1. 一句话回答

只要本轮执行无法满足固定 shape、稳定地址、稳定控制流或 backend capture 支持，就需要 fallback。

---

## 2. 常见 fallback 场景占位

```text
- batch shape 未命中 capture size；
- prefill token 数过大或过动态；
- multimodal 输入动态；
- LoRA mapping 动态且不支持 capture；
- 某些量化 kernel 不支持 graph；
- attention backend metadata 不支持 capture；
- spec decode logits layout 动态；
- pooling 输出形态动态；
- PP / DP rank shape 不一致；
- debug / profile 模式强制 eager。
```

---

## 3. fallback 要保证什么

```text
- 输出语义一致；
- KV cache 写入一致；
- attention metadata 正确；
- sampler / output 不受影响；
- stats 能记录为什么 miss；
- 不应该因为 graph miss 中断请求。
```

---

## 4. 一句话总结

```text
cudagraph 是优化路径，eager fallback 是正确性兜底。
```
