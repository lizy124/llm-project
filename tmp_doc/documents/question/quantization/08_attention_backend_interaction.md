# 08. 量化如何影响 attention backend？

源码位置：

- `code/vllm/vllm/v1/attention/backend.py`
- `code/vllm/vllm/v1/attention/selector.py`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py`
- `code/vllm/vllm/v1/attention/backends/flashinfer.py`
- `code/vllm/vllm/v1/attention/backends/triton_attn.py`
- `code/vllm/vllm/platforms/cuda.py`

本问题关注：KV cache dtype、head size、MLA、backend 能力和量化格式如何共同决定 attention backend 选择。

---

## 1. 一句话回答

attention backend 选择不只看硬件，也要看量化格式是否被 backend 支持。

```text
同样是 attention：

FP16 KV cache 可能走 FlashAttention；
FP8 / INT8 / NVFP4 KV cache 可能要求 FlashInfer / Triton / 特定 backend；
MLA + 量化又可能走 FlashMLA / Triton MLA / FlashInfer MLA。
```

---

## 2. backend 能力占位

需要梳理：

```text
- backend 支持哪些 kv_cache_dtype；
- 是否支持 per-token-head scale；
- 是否支持 FP8 / NVFP4；
- 是否支持 quantized MLA；
- 是否支持 sliding window + quantized KV；
- 是否支持 cascade attention + quantized KV。
```

---

## 3. selector 占位

```text
get_attn_backend()
  → AttentionSelectorConfig
  → current_platform 选择 backend
  → backend ability check
  → fallback 或报错
```

---

## 4. 与 KV cache layout 的关系

```text
量化格式可能要求特定 KV cache layout：

- NHD / HND；
- packed pages；
- page padding；
- scale tensor 附加存储；
- backend-specific stride order。
```

---

## 5. 一句话总结

```text
KV cache 量化会把 attention backend 选择从“哪个最快”变成“哪个既支持该量化格式又能跑得快”。
```
