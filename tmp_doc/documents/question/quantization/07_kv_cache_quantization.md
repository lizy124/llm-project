# 07. KV cache quantization 如何工作？

源码位置：

- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `code/vllm/vllm/config.py`

本问题关注：KV cache dtype、KVQuantMode、scale、layout 和 attention backend 的关系。

---

## 1. 一句话回答

KV cache quantization 压缩的是历史 K/V cache，不是模型权重。

```text
weight quantization：
  影响 Linear / MoE 权重和 matmul。

KV cache quantization：
  影响 attention 读取历史 K/V 的存储格式、scale 和 backend kernel。
```

---

## 2. KVQuantMode 占位

后续补充：

```text
- none；
- FP8 per tensor；
- INT8 per token/head；
- FP8 per token/head；
- NVFP4；
- 其他 backend 特定模式。
```

---

## 3. 需要梳理的问题

```text
- kv_cache_dtype 从哪里配置；
- KV cache scale 如何保存；
- scale 是 per-layer、per-tensor、per-token 还是 per-head；
- attention backend 如何声明支持哪些 KV quant；
- KV cache layout 是否随 quant mode 改变；
- paged KV cache 与量化格式如何结合；
- prefix cache / external KV transfer 是否要保存量化格式。
```

---

## 4. attention 链路占位

```text
Attention.forward()
  → 写入 K/V 到 KV cache
  → 可能 quantize K/V
  → backend 读取量化 KV cache
  → 使用 scale dequant 或 fused compute
```

---

## 5. 一句话总结

```text
KV cache quantization 是 attention 子系统的内存优化，它和权重量化是两条不同但会互相限制的路径。
```
