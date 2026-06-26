# 18. HMA 与 KV cache layout：它和 Attention 有什么关系？

源码位置：

- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/block_pool.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`

本文用于梳理 HMA / hybrid KV cache manager / KV cache group / layout 与 attention 的关系，说明 HMA 不是 attention 类型本身，而是影响 attention KV cache 管理和 connector 兼容性的内存组织机制。

---

## 1. 本文要回答的问题

```text
HMA 是什么？
它和 hybrid KV cache manager 有什么关系？
KV cache groups 如何影响 attention metadata？
connector 为什么要声明 supports HMA？
HMA 如何影响 KV cache layout、block table 和 attention backend？
HMA 与 Mamba / hybrid attention 模型有什么关系？
```

---

## 2. 核心结论占位

占位：后续补充 HMA 与 KV cache / attention 的关系。

```text
HMA 更接近 KV cache 内存管理和分组机制，而不是 attention 算法；它通过 KV cache groups、block 分配和 connector 兼容性影响 attention metadata 与 KV cache layout。
```

---

## 3. 待梳理源码点

```text
KV cache groups
hybrid KV cache manager
disable_hybrid_kv_cache_manager
SupportsHMA
request_finished_all_groups()
AttentionSpec groups
cross-layer / group layout
connector supports_hma_config
```

---

## 4. 主链路占位

```text
VllmConfig / KVCacheConfig
  → KV cache groups
  → KVCacheManager / BlockPool allocation
  → GPUModelRunner metadata builders per group
  → AttentionMetadata per group
  → backend KV cache read / write
  → connector HMA-compatible save / load
```

---

## 5. 后续重点占位

```text
- HMA 和普通单 KV cache group 的区别；
- 多 KV cache group 如何影响 block ids / slot mapping；
- connector request_finished_all_groups() 为什么必要；
- HMA 与 cross-layer KV cache layout 不是同一概念；
- HMA 与 attention backend 支持边界如何对齐。
```
