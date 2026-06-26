# 07. Context Parallel / DCP / PCP 如何切分上下文？

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/flash_attn.py`
- `vllm/vllm/v1/attention/backends/flashinfer.py`
- `vllm/vllm/v1/attention/ops/`

本问题关注：Context Parallel、Decode Context Parallel、Prefill Context Parallel 如何把长上下文或 attention 计算拆给多个 rank，为什么 attention partial output 不能简单相加，LSE 在 merge 中起什么作用，以及哪些 attention backend 支持这些路径。

---

## 1. 一句话回答

Context Parallel 是 attention / context 维度的并行：

```text
长上下文或 KV context 被多个 rank 分担；
每个 rank 计算局部 attention state；
最终通过 softmax LSE 语义正确地合并 partial output。
```

一句话记忆：

```text
CP 是“把长上下文 attention 拆给多个 rank 合作算”。
```

---

## 2. 本文要回答的问题

```text
CP / DCP / PCP 分别是什么？
它们切的是 query、KV context，还是 prefill/decode 阶段？
为什么 attention merge 需要 LSE？
backend 为什么要声明 can_return_lse_for_decode？
FlashAttention DCP 路径如何拆 context attention 和 query attention？
DCP 和 cascade attention / sliding window / ALiBi 有哪些冲突？
```

---

## 3. 最小主链路占位

```text
query / KV context
  → 按 CP group 切分 context 或 query
  → 每个 rank 调 attention backend 计算 partial output + partial LSE
  → 跨 rank 通信收集 partial state
  → merge_attn_states / LSE merge
  → 得到完整 attention output
```

---

## 4. 通信原语占位

```text
all-gather：
  收集 query 或 partial states。

all-to-all：
  某些 DCP 通信 backend 可用 a2a 路径。

reduce / custom merge：
  attention output 需要按 softmax LSE 合并，不是普通 sum。
```

---

## 5. 待梳理源码点

```text
decode_context_parallel_size
prefill context parallel 相关配置
get_dcp_group / get_pcp_group
AttentionImplBase 中 dcp_world_size / pcp_world_size
can_return_lse_for_decode
FlashAttentionImpl._forward_with_dcp
merge_attn_states
cp_lse_ag_out_rs
dcp_a2a_lse_reduce
attention backend 对 DCP / PCP 的支持声明
```

---

## 6. 和 Attention / KV cache 的关系

```text
CP 直接改变 attention 的计算方式；
不是所有 backend 都支持 partial state 和 LSE；
KV cache 的本地可见长度、block table、seq_lens metadata 可能随 CP 变化；
DCP 通常比 TP / PP 更贴近 attention backend 内部实现。
```
