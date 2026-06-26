# 08. KV cache 如何受并行策略影响？

源码位置：

- `vllm/vllm/v1/core/kv_cache_manager.py`
- `vllm/vllm/v1/worker/kv_cache_model_runner_mixin.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/`
- `vllm/vllm/distributed/parallel_state.py`

本问题关注：KV cache 在 TP / PP / DP / EP / CP 下如何放置和访问，block table / slot mapping 是否随并行维度变化，KV cache shape 和 layout 如何由 backend 与并行配置共同决定，以及 KV connector / external KV transfer 如何适配并行执行。

---

## 1. 一句话回答

KV cache 是并行体系和 attention backend 的交汇点：

```text
Scheduler / KV manager 分配逻辑块；
ModelRunner 生成 block_table / slot_mapping；
每个 rank 按自身并行身份持有或访问本地 KV cache；
attention backend 根据 metadata 在本地或分片 KV cache 上执行。
```

---

## 2. 本文要回答的问题

```text
TP 下 KV heads 是否按 rank 分片？
PP 下哪些 stage 持有 KV cache？
DP 下每个 replica 的 KV cache 是否独立？
EP 是否影响 KV cache？
CP / DCP 是否切分 KV context？
block_table / slot_mapping 在并行下如何解释？
KV cache layout 为什么和 backend 相关？
KV connector / external KV transfer 如何处理 rank 归属？
```

---

## 3. 分并行策略占位

```text
TP：
  attention heads / KV heads 可能按 TP rank 分配；每个 rank 持有自己负责 head 的 KV cache。

PP：
  只有包含 attention layers 的 pipeline stage 需要对应层的 KV cache。

DP：
  不同 replica 通常持有独立 KV cache 状态。

EP：
  MoE expert 并行本身不直接改变 attention KV cache，但会和同一模型中的 TP / DP / PP 共存。

CP / DCP / PCP：
  可能改变每个 rank 可见的 context / KV 长度，需要 attention metadata 协同。
```

---

## 4. 待梳理源码点

```text
KV cache spec 生成
get_kv_cache_shape
get_kv_cache_stride_order
bind_kv_cache
block_table_tensor
slot_mapping
KV cache group / attention group
KV cache 初始化与 parallel_config
prefix cache 与并行 rank 的关系
KV connector load / save 与 rank 映射
external KV transfer 对 DP / TP / PP 的要求
```

---

## 5. 和已有文档的关系

```text
../kv_cache_transfer/：
  已经梳理 KV manager、block pool、prefix cache、external KV transfer。

本篇关注：
  这些 KV cache 机制放到 TP / PP / DP / EP / CP 并行体系后如何解释。

../attention/：
  attention metadata 中的 block_table / slot_mapping 是 KV cache 与 backend 的接口。
```
