# 13. PagedAttention：vLLM 的分页 KV cache 注意力机制

源码位置：

- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/block_pool.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/attention/`
- `code/vllm/vllm/v1/kv_cache_interface.py`

本文用于梳理 PagedAttention 的核心思想：KV cache 分块、block table、slot mapping、非连续 KV 存储和 attention backend 如何读取 paged KV cache。

---

## 1. 本文要回答的问题

```text
PagedAttention 解决什么问题？
block table 和操作系统 page table 的类比是什么？
Scheduler / KVCacheManager 如何分配 paged KV blocks？
ModelRunner 如何生成 slot mapping？
attention backend 如何按 block table 读取历史 KV？
PagedAttention 和 prefix cache / preemption / continuous batching 有什么关系？
PagedAttention 和 FlashAttention 是什么关系？
```

---

## 2. 核心结论占位

占位：后续补充 PagedAttention 的完整链路。

```text
PagedAttention 是 vLLM 通过固定大小 KV blocks 管理请求历史 KV 的机制，它让请求不需要连续显存，也让 continuous batching、prefix cache、preemption 和高效复用成为可能。
```

---

## 3. 待梳理源码点

```text
KVCacheManager
BlockPool
KVCacheBlocks
block table
slot mapping
InputBatch
AttentionMetadata
paged KV backend kernel
```

---

## 4. 主链路占位

```text
Request tokens
  → Scheduler / KVCacheManager allocate blocks
  → block table
  → GPUModelRunner slot mapping
  → AttentionMetadata
  → backend paged attention kernel
  → paged KV cache read / write
```

---

## 5. 后续重点占位

```text
- prompt prefill 如何写入多个 blocks；
- decode token 如何追加到已有 block；
- prefix cache 命中时如何复用 blocks；
- preemption 时 block 如何释放或重分配；
- external KV load 如何把远端 KV 注入 paged blocks。
```
