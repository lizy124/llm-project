# 14. MHA / MQA / GQA：head 结构如何影响 attention 和 KV cache？

源码位置：

- `code/vllm/vllm/model_executor/models/`
- `code/vllm/vllm/model_executor/layers/attention/`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/attention/`
- `code/vllm/vllm/config/model.py`

本文用于梳理 MHA、MQA、GQA 的结构差异，以及这些差异如何影响 KV head 数、KV cache shape、attention backend 参数。

---

## 1. 本文要回答的问题

```text
MHA / MQA / GQA 的区别是什么？
num_heads 和 num_kv_heads 如何影响 KV cache？
GQA 对 paged KV cache shape 有什么影响？
backend forward 如何接收 head / kv head 参数？
不同模型如何在 vLLM 中声明这些结构？
Tensor Parallel 下 head / kv head 如何切分？
```

---

## 2. 核心结论占位

占位：后续补充 MHA / MQA / GQA 与 vLLM attention 参数的映射。

```text
MHA / MQA / GQA 是模型 attention head 结构的差异，核心影响是 query heads 和 KV heads 的数量关系；它会改变 KV cache shape、backend 参数和 TP 切分方式，但不是 backend 类型本身。
```

---

## 3. 待梳理源码点

```text
num_attention_heads
num_key_value_heads
num_kv_heads
AttentionSpec
KV cache shape
model attention layer config
backend metadata
TP head partition
```

---

## 4. 对比占位

```text
MHA：
  Q heads 和 K/V heads 通常一一对应。

MQA：
  多个 Q heads 共享一组 K/V heads。

GQA：
  多组 Q heads 共享较少数量的 K/V heads，是 MHA 和 MQA 之间的折中。
```

---

## 5. 后续重点占位

```text
- HF config 到 vLLM model config 的字段映射；
- KV cache tensor 中 num_kv_heads 的来源；
- backend 对 GQA 的支持条件；
- TP 下 num_heads / num_kv_heads 的本地 rank 切分。
```
