# 07. Context Parallel / DCP / PCP 如何切分上下文？

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/v1/worker/cp_utils.py`
- `vllm/vllm/v1/worker/gpu/cp_utils.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu/block_table.py`
- `vllm/vllm/v1/worker/attn_utils.py`
- `vllm/vllm/v1/core/kv_cache_utils.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/flash_attn.py`
- `vllm/vllm/v1/attention/backends/flashinfer.py`
- `vllm/vllm/v1/attention/ops/common.py`
- `vllm/vllm/v1/attention/ops/merge_attn_states.py`
- `vllm/vllm/v1/attention/ops/dcp_alltoall.py`

本问题关注：Context Parallel、Decode Context Parallel、Prefill Context Parallel 如何把长上下文或 attention 计算拆给多个 rank，DCP 如何把 KV cache 按上下文维度分片，为什么 attention partial output 不能简单相加，LSE 在 merge 中起什么作用，以及哪些 attention backend 支持这些路径。

---

## 1. 一句话回答

Context Parallel 是 attention / context 维度的并行：

```text
把同一个请求的上下文 KV 按 CP 规则映射到多个 rank；
每个 rank 主要读写自己负责的本地历史 KV 分片，并参与对应 attention 计算；
attention backend 计算局部 attention output + 局部 softmax LSE；
最后用 LSE 语义正确地合并 partial attention states。
```

在 vLLM v1 里最主要的落地形态是 DCP：

```text
DCP = Decode Context Parallel
  → decode 阶段复用 TP group 内的 GPU，
    把 KV context 切到多个 DCP rank 上。
```

一句话记忆：

```text
CP 是“把长上下文 attention 的 KV 维度拆给多个 rank 合作算”。
```

---

## 2. 本文要回答的问题

```text
CP / DCP / PCP 分别是什么？
DCP 切的是 query、KV context，还是 prefill/decode 阶段？
DCP 为什么复用 TP group，而不增加 world size？
cp_kv_cache_interleave_size 如何影响 KV cache slot？
为什么 attention merge 需要 LSE？
backend 为什么要声明 can_return_lse_for_decode / supports_pcp？
FlashAttention DCP 路径如何拆 context attention 和 query attention？
DCP 和 KV cache block table / slot mapping / seq_lens metadata 有什么关系？
DCP 和 hybrid KV cache / cascade attention / speculative decoding 有哪些限制？
```

---

## 3. 先给结论：CP 在 vLLM 里切的是什么

Context Parallel 不是 TP / PP / DP 的替代品，它切的是 **同一个请求的上下文维度**。

可以先用 decode 阶段理解：

```text
一个请求已有很长 KV cache：
  K/V positions: 0 ... N-1

DCP size = 4 时：
  rank 0 保存/计算其中一部分 KV positions
  rank 1 保存/计算其中一部分 KV positions
  rank 2 保存/计算其中一部分 KV positions
  rank 3 保存/计算其中一部分 KV positions

每个 rank 对完整 query 的一部分 context 做 attention，
再把 partial attention state 合并成完整 output。
```

关键点：

```text
1. KV context 被切分；
2. query 可能需要 all-gather 到各个 DCP rank；
3. 每个 rank 的 attention output 只是局部 softmax 结果；
4. 局部 output 必须用 LSE 重新归一化后合并。
```

---

## 4. CP / DCP / PCP 的区别

| 名称 | 全称 | 主要阶段 | vLLM 里的状态 |
|---|---|---|---|
| CP | Context Parallel | 泛指 context 维度并行 | 抽象概念，配置和 kernel 里体现为 cp size / cp rank |
| DCP | Decode Context Parallel | decode | 主要落地路径，FlashAttention / FlashInfer 等 backend 支持 |
| PCP | Prefill Context Parallel | prefill | 配置和能力检查已存在，backend 需要显式 `supports_pcp` |

配置字段在 `ParallelConfig` 中：

```text
prefill_context_parallel_size
decode_context_parallel_size
cp_kv_cache_interleave_size
dcp_comm_backend
```

它们的关系是：

```text
total_cp_world_size = pcp_world_size * dcp_world_size

total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
```

`cp_kv_cache_interleave_size` 定义 token 如何在 total CP ranks 之间交错存储。

---

## 5. 配置层：ParallelConfig 如何定义 DCP / PCP

`ParallelConfig` 中的关键字段：

```text
prefill_context_parallel_size: int = 1
decode_context_parallel_size: int = 1
dcp_comm_backend: "ag_rs" | "a2a"
cp_kv_cache_interleave_size: int = 1
```

### 5.1 decode_context_parallel_size

注释里说得很直接：

```text
DCP 不改变 world size；
它复用 TP group 的 GPU；
因此 tp_size 必须能被 dcp_size 整除。
```

校验逻辑：

```text
tensor_parallel_size % decode_context_parallel_size == 0
```

如果不满足，直接报错。

### 5.2 dcp_comm_backend

DCP 当前有两类通信 backend：

```text
ag_rs：AllGather + ReduceScatter，默认路径；
a2a：All-to-All exchange partial output + LSE，再用 Triton kernel combine。
```

如果设置：

```text
dcp_comm_backend = "a2a"
```

则要求：

```text
decode_context_parallel_size > 1
```

### 5.3 cp_kv_cache_interleave_size

它定义 KV cache 在 CP rank 之间的交错粒度：

```text
interleave_size = 1：
  token i 存在 rank i % total_cp_world_size

interleave_size = block_size：
  更接近 block 级切分；
  前一个 rank 的 block 填满后再轮到下一个 rank。
```

约束：

```text
block_size >= cp_kv_cache_interleave_size
block_size % cp_kv_cache_interleave_size == 0
```

---

## 6. 进程组：DCP / PCP group 如何建立

`parallel_state.py` 中有两个全局 group：

```text
_DCP → get_dcp_group()
_PCP → get_pcp_group()
```

初始化 model parallel group 时，rank layout 包含这些维度：

```text
ExternalDP x DP x PP x PCP x TP
```

DCP group 的构建方式：

```text
在现有 rank mesh 上按 decode_context_parallel_size reshape 成组；
DCP 复用 TP 相关 rank，但不额外增加 world size。
```

源码注释强调：

```text
DCP 不改变 world size；
只是复用 TP group 的 GPU。
```

PCP group 的构建方式：

```text
在 rank tensor 中 transpose PCP 和 TP 维度，
再按 prefill_context_parallel_size reshape 成 group。
```

因此：

```text
TP 决定权重/头维度如何切；
DCP/PCP 决定 context KV 如何在已有 ranks 内切。
```

---

## 7. Scheduler 层如何感知 CP

`Scheduler` 初始化时会读取：

```text
self.dcp_world_size = parallel_config.decode_context_parallel_size
self.pcp_world_size = parallel_config.prefill_context_parallel_size
```

这些值会继续传给 KV cache manager / metrics 等组件。

Scheduler 层的核心职责不是做 attention merge，而是保证：

```text
1. block size / prefix cache 对齐规则与 CP 一致；
2. 不兼容功能提前禁用或报错；
3. KV cache allocation 不产生 backend 无法解释的 block table。
```

例如，一些功能会因为 CP 被禁用：

```text
enable_return_routed_experts 不能与 context parallelism 同时使用。
```

---

## 8. KV cache block size 与 CP 的关系

`resolve_kv_cache_block_sizes()` 是 CP 和 KV cache 对齐关系的关键入口。

单 KV cache group 时：

```text
scheduler_block_size = cache_config.block_size * dcp * pcp
hash_block_size = scheduler_block_size
```

这表示：

```text
Scheduler 以“完整 CP group 覆盖的一组 token”为对齐单位。
```

多 KV cache groups 时，如果启用了 CP：

```text
如果 dcp != 1 或 pcp != 1：
  抛错：Hybrid KV cache groups with multiple block sizes do not support context parallelism
```

因此当前限制是：

```text
Context Parallel 不支持多 block size 的 hybrid KV cache groups。
```

这是因为多 group / 多 block size 本身就需要复杂的 block table 对齐，再叠加 CP 的 context 切分会让 scheduler block alignment、prefix cache hash、slot mapping 都变复杂。

---

## 9. ModelRunner 初始化时如何接入 DCP

GPU ModelRunner 初始化时会保存：

```text
self.dcp_size = parallel_config.decode_context_parallel_size
self.use_dcp = self.dcp_size > 1
self.dcp_rank = get_dcp_group().rank_in_group if use_dcp else 0
self.cp_interleave = parallel_config.cp_kv_cache_interleave_size
```

初始化 KV cache 时，DCP 会影响每个请求需要的 block table 长度：

```text
max_num_blocks = ceil(max_model_len / (block_size * dcp_size))
```

含义是：

```text
同一个全局上下文被 dcp_size 个 rank 分摊；
每个 rank 只需要存约 1 / dcp_size 的 KV blocks。
```

随后 `BlockTables` 会带着 CP 参数初始化：

```text
BlockTables(
  block_sizes=...,
  kernel_block_sizes=...,
  cp_size=dcp_size,
  cp_rank=dcp_rank,
  cp_interleave=cp_kv_cache_interleave_size,
)
```

所以：

```text
DCP 在 Worker 侧首先体现为 CP-aware block table / slot mapping。
```

---

## 10. CP-aware slot mapping 如何工作

`BlockTables.compute_slot_mappings()` 会把 CP 参数传入 Triton kernel：

```text
CP_SIZE
cp_rank
CP_INTERLEAVE
```

普通 slot mapping 公式是：

```text
block_index = position // block_size
block_offset = position % block_size
block_number = block_table[request, block_index]
slot_id = block_number * block_size + block_offset
```

DCP / CP 下变成：

```text
block_index = position // (block_size * CP_SIZE)
block_offset = position % (block_size * CP_SIZE)
```

然后判断这个 position 是否属于当前 rank：

```text
is_local = block_offset // CP_INTERLEAVE % CP_SIZE == cp_rank
```

如果属于当前 rank：

```text
rounds = block_offset // (CP_INTERLEAVE * CP_SIZE)
remainder = block_offset % CP_INTERLEAVE
local_offset = rounds * CP_INTERLEAVE + remainder
slot_id = block_number * block_size + local_offset
```

如果不属于当前 rank：

```text
slot_id = PAD_SLOT_ID
```

这就是 CP 的 KV 写入核心：

```text
全局 position → 当前 rank 的本地 KV slot；
非本 rank 负责的 token 不写 KV cache。
```

---

## 11. 用例子理解 cp_kv_cache_interleave_size

假设：

```text
CP_SIZE = 2
block_size = 8
```

### 11.1 interleave_size = 1

token 级轮转：

```text
position 0 → rank 0
position 1 → rank 1
position 2 → rank 0
position 3 → rank 1
...
```

rank 0 本地 offset：

```text
position 0 → local offset 0
position 2 → local offset 1
position 4 → local offset 2
```

### 11.2 interleave_size = 4

每 4 个 token 轮转：

```text
position 0-3   → rank 0
position 4-7   → rank 1
position 8-11  → rank 0
position 12-15 → rank 1
```

这种方式更接近块状布局，可能减少某些场景的碎片或通信复杂度。

---

## 12. dcp_local_seq_lens 是什么

DCP 下 attention backend 不能只看全局 `seq_lens`。

原因：

```text
seq_lens 是请求的全局上下文长度；
但每个 DCP rank 只持有其中一部分 KV。
```

因此 Worker 会构造：

```text
dcp_local_seq_lens
```

它表示：

```text
每个 request 在当前 DCP rank 上实际可见的本地 KV 长度。
```

源码位置：`code/vllm/vllm/v1/worker/gpu/cp_utils.py:8`

`prepare_dcp_local_seq_lens()` 的核心公式：

```text
rounds = seq_lens // (dcp_size * cp_interleave)
remainder = seq_lens % (dcp_size * cp_interleave)

local_remainder = clamp(
  remainder - dcp_rank * cp_interleave,
  0,
  cp_interleave,
)

local_seq_lens = rounds * cp_interleave + local_remainder
```

这和 slot mapping 使用的是同一套 interleave 规则。

---

## 13. Attention metadata 中的 CP 字段

源码位置：`code/vllm/vllm/v1/attention/backend.py:395`

`CommonAttentionMetadata` 是 attention backend 的公共 metadata。

与 CP / DCP 相关的字段包括：

```text
query_start_loc
seq_lens
max_seq_len
block_table_tensor
slot_mapping
dcp_local_seq_lens
dcp_local_seq_lens_cpu
positions
```

其中：

```text
block_table_tensor：告诉 backend 当前 request 使用哪些本地 blocks；
slot_mapping：告诉 KV cache update 当前 token 写到本 rank 哪个 slot；
dcp_local_seq_lens：告诉 backend 当前 rank 本地 context 长度；
seq_lens：仍保留全局语义，用于 batch / query 组织。
```

所以 CP 不是只影响通信，也会影响 attention metadata 的基础语义。

---

## 14. attention backend 的 CP 能力检查

源码位置：`code/vllm/vllm/v1/worker/cp_utils.py:22`

`check_attention_cp_compatibility()` 会逐层检查 attention impl。

如果：

```text
pcp_size * dcp_size > 1
```

则进入 CP 兼容性检查。

### 14.1 DCP 要求

DCP 要求：

```text
layer_impl.need_to_return_lse_for_decode == True
```

如果不满足，会报错：

```text
Decode Context Parallelism requires attention implementations to return the softmax LSE during decode
```

### 14.2 PCP 要求

PCP 要求：

```text
layer_impl.supports_pcp == True
```

如果不满足，会报错。

### 14.3 speculative decoding + interleave 限制

如果启用了 speculative decoding，且：

```text
cp_kv_cache_interleave_size > 1
```

还要求：

```text
supports_mtp_with_cp_non_trivial_interleave_size == True
```

这说明 CP 不只是通信配置，还必须得到 attention backend / speculative decode path 的联合支持。

---

## 15. 为什么 DCP merge 需要 LSE

attention 输出是：

```text
softmax(QK^T) V
```

如果把 KV context 分成两段：

```text
KV = [KV_a, KV_b]
```

每个 rank 分别计算：

```text
out_a = softmax(scores_a) V_a
out_b = softmax(scores_b) V_b
```

不能直接：

```text
out = out_a + out_b
```

因为 `out_a` 和 `out_b` 各自的 softmax 分母不同。

正确合并需要知道每段的 log-sum-exp：

```text
lse_a = log(sum(exp(scores_a)))
lse_b = log(sum(exp(scores_b)))
lse = log(exp(lse_a) + exp(lse_b))
```

然后按权重合并：

```text
out = exp(lse_a - lse) * out_a
    + exp(lse_b - lse) * out_b
```

这就是为什么 attention backend 必须能返回 LSE。

---

## 16. merge_attn_states 的语义

`merge_attn_states()` 的职责是：

```text
把 prefix/context partial output 和 suffix/query partial output
按 LSE 重新归一化后合并。
```

函数参数体现了这一点：

```text
prefix_output
prefix_lse
suffix_output
suffix_lse
```

文档注释里也说明：

```text
使用 log-sum-exp rescaling method 合并 partial attention outputs。
```

这类合并常见于两种场景：

```text
1. DCP：不同 rank 分别算不同 KV context 的 partial attention；
2. prefix/suffix attention：prefix cache 与本轮新 token 分开算后合并。
```

---

## 17. FlashAttention DCP 路径总览

FlashAttention backend 中：

```text
FlashAttentionImpl.can_return_lse_for_decode = True
```

初始化时会根据配置选择 DCP combine 函数：

```text
如果 dcp_comm_backend == "a2a":
  dcp_combine = dcp_a2a_lse_reduce
否则：
  dcp_combine = cp_lse_ag_out_rs
```

forward 时，如果：

```text
self.dcp_world_size > 1
```

就进入：

```python
_forward_with_dcp(...)
```

否则走普通 FlashAttention varlen 路径。

---

## 18. FlashAttention _forward_with_dcp 做了什么

DCP FlashAttention 路径可以拆成三段：

```text
1. context attention：
   当前 rank 用本地 KV cache 计算所有 DCP ranks 的 query 对本地 context 的 attention。

2. DCP combine：
   跨 DCP ranks 用 LSE 合并 context attention partial states。

3. query attention：
   对本轮 query/new token 内部再算一段 attention，
   最后与 context attention 用 merge_attn_states 合并。
```

### 18.1 all-gather query

首先：

```python
query_across_dcp = get_dcp_group().all_gather(query, dim=1)
```

含义：

```text
每个 rank 需要看到所有 DCP rank 的 query，
才能用自己本地 KV context 计算对应 partial attention。
```

### 18.2 对本地 KV context 做 attention

然后调用 `flash_attn_varlen_func()`：

```text
q = query_across_dcp
k = key_cache
v = value_cache
seqused_k = attn_metadata.dcp_context_kv_lens
max_seqlen_k = attn_metadata.max_dcp_context_kv_len
block_table = attn_metadata.block_table
return_softmax_lse = True
causal = False
```

这里 `causal=False` 的原因是：

```text
这段只算“历史 context KV”；
query 对历史 context 全部可见，因果关系已经由 context/query 拆分处理。
```

### 18.3 DCP combine

context attention 得到：

```text
context_attn_out
context_lse
```

然后调用：

```text
dcp_combine(context_attn_out, context_lse, get_dcp_group(), return_lse=True)
```

默认 `cp_lse_ag_out_rs` 做的是：

```text
all_gather LSE
→ correct_attn_out 按全局 LSE 修正 output
→ reduce_scatter output
→ 返回当前 rank 负责的 heads / lse
```

如果使用 `a2a` backend，则走 all-to-all + Triton reduce 路径。

### 18.4 query attention

随后用本 rank 的 `query/key/value` 再算本轮 query token 之间的 attention：

```text
q = query
k = key
v = value
cu_seqlens_q = query_start_loc
cu_seqlens_k = query_start_loc
causal = attn_metadata.causal
return_softmax_lse = True
```

这段处理的是：

```text
本轮新 token 之间的 causal attention。
```

### 18.5 合并 context 和 query 两段 attention

最后：

```python
merge_attn_states(
    output,
    context_attn_out_cor,
    context_lse_cor,
    query_attn_out,
    query_lse,
)
```

这一步把：

```text
历史 context attention
+
本轮 query attention
```

用 LSE 语义合并成最终 output。

---

## 19. DCP 中为什么既有 context attention 又有 query attention

decode / extend 场景里，一个 token 的可见 KV 可以拆成：

```text
历史 KV cache：已经写入 cache 的 context；
本轮新 K/V：当前 forward 里刚产生的 key/value。
```

DCP 的 KV cache 只覆盖历史 context 的本地分片；但当前 forward 的 new K/V 还在本 rank 的 query batch 里。

因此 FlashAttention DCP 路径拆成：

```text
context_attn：query attends to distributed KV cache；
query_attn：query attends to current-step K/V；
merge：用 LSE 合并两段 attention state。
```

这和普通 attention 一次性看完整 KV 的结果等价，但更适合分布式 KV cache。

---

## 20. cp_lse_ag_out_rs 的通信语义

默认 DCP combine 是：

```text
cp_lse_ag_out_rs
```

名字可以拆开看：

```text
cp：context parallel
lse：需要 softmax LSE
ag：all-gather
out：attention output
rs：reduce-scatter
```

核心流程：

```text
1. all-gather 每个 rank 的 partial LSE；
2. 用所有 LSE 计算全局 softmax 归一化；
3. 修正本 rank partial output；
4. reduce-scatter 得到当前 rank 的最终 output 分片。
```

这不是普通 all-reduce，因为 attention output 的合并权重依赖 LSE。

---

## 21. dcp_a2a_lse_reduce 的位置

`dcp_a2a_lse_reduce()` 是 DCP 的另一个通信后端。

配置：

```text
dcp_comm_backend = "a2a"
```

会让 backend 使用：

```text
dcp_a2a_lse_reduce
```

它的目标是减少某些模型上的 NCCL 调用次数，尤其是 MLA 场景：

```text
All-to-All exchange partial outputs + LSE
→ Triton kernel combine
```

所以：

```text
ag_rs 是默认通用路径；
a2a 是更专门的通信优化路径。
```

---

## 22. PCP 当前怎么体现

PCP 字段已经在配置、进程组、能力检查中存在：

```text
prefill_context_parallel_size
get_pcp_group()
supports_pcp
```

但相比 DCP，PCP 的运行态路径更依赖具体 backend 是否实现。

统一兼容性检查要求：

```text
pcp_size > 1 时，所有 attention impl 必须 supports_pcp=True。
```

也就是说：

```text
PCP 不是只改一个配置就能启用；
必须 attention backend 明确支持 prefill context parallel。
```

从代码结构上看，PCP 与 DCP 共享：

```text
cp_kv_cache_interleave_size
total_cp_world_size
total_cp_rank
context-parallel group 概念
```

但 DCP 在 FlashAttention `_forward_with_dcp()` 中有更直接的完整链路。

---

## 23. DCP 与 KV cache update 的关系

attention layer 的 KV cache update 通常会调用类似：

```text
reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, ...)
```

这里的 `slot_mapping` 已经经过 CP-aware kernel 处理：

```text
属于当前 rank 的 token → 有效 slot id；
不属于当前 rank 的 token → PAD_SLOT_ID。
```

因此 KV cache update 不需要重新理解 DCP 的切分规则。

它只要按 slot mapping 写入即可：

```text
slot_mapping 决定写入；
PAD_SLOT_ID 屏蔽非本 rank token。
```

---

## 24. DCP 与 block table 的关系

DCP 下 block table 仍然是 request → block ids 的映射，但 block 的含义变成：

```text
一个 block 表示当前 rank 上的一段本地 KV storage。
```

全局 position 到 block table 的索引使用：

```text
position // (block_size * CP_SIZE)
```

这意味着：

```text
多个 CP ranks 共同覆盖一个全局 block span；
每个 rank 在这个 span 内只保存自己的 interleaved tokens。
```

所以 DCP 不只是 attention backend 的通信优化，它从 block table 层就改变了 global position → local slot 的映射。

---

## 25. DCP 与 prefix cache 的关系

在单 KV group 下：

```text
scheduler_block_size = block_size * dcp * pcp
```

这意味着 prefix cache / num_computed_tokens 的对齐粒度必须覆盖完整 CP group。

原因：

```text
如果只按单 rank 的本地 block 对齐，
不同 CP rank 可能对同一个 global prefix 是否完整有不同判断，
导致 slot mapping / prefix cache hit 不一致。
```

因此 Scheduler 用更大的 `scheduler_block_size` 作为全局安全边界。

---

## 26. DCP 与 hybrid KV cache 的限制

当前源码明确限制：

```text
Hybrid KV cache groups with multiple block sizes do not support context parallelism
```

也就是：

```text
多 KV cache group / 多 block size 的 HMA 场景，
不能再叠加 DCP / PCP。
```

根本原因是两个机制都会影响 block 对齐：

```text
HMA：不同 attention/cache group 有不同 block/table 语义；
CP：同一 group 内 global context 又被多个 ranks 分片。
```

两者叠加需要更复杂的 prefix cache、block hash、slot mapping 对齐逻辑。

---

## 27. DCP 与 cascade attention

FlashAttention 普通路径中，如果没有 DCP，会根据 metadata 选择：

```text
普通 varlen attention
或
cascade_attention
```

DCP 路径在 `not use_cascade` 分支里优先处理：

```text
if self.dcp_world_size > 1:
    _forward_with_dcp(...)
```

这说明 DCP 和 cascade attention 不是简单叠加的两个独立开关。

从能力上理解：

```text
cascade attention 已经把 prefix / suffix attention 拆开；
DCP 也要拆 context attention / query attention 并做 LSE merge；
二者同时启用会让 metadata 与 merge 语义更复杂。
```

因此很多 backend 会对 DCP、cascade、sliding window、ALiBi、varlen 等组合做额外限制或走不同 builder 路径。

---

## 28. DCP 与 sliding window / ALiBi

FlashAttention DCP 路径仍会把这些参数传给 `flash_attn_varlen_func()`：

```text
alibi_slopes
window_size
softcap
```

但要注意：

```text
DCP context attention 使用 causal=False；
query attention 使用 attn_metadata.causal；
sliding window 需要在这两段 attention 中保持等价语义。
```

所以 backend 支持 DCP 并不只是“能通信”，还要保证：

```text
局部分片 attention + LSE merge 后，
和原始完整 attention 的 mask / window / ALiBi 语义一致。
```

这也是为什么 DCP 支持需要 backend 显式声明。

---

## 29. DCP 与 speculative decoding / MTP

`check_attention_cp_compatibility()` 中有一条特殊检查：

```text
如果 speculative_config 不为空，且 cp_kv_cache_interleave_size > 1，
则要求 supports_mtp_with_cp_non_trivial_interleave_size。
```

原因是 speculative decoding / MTP 会让一次 forward 中出现多个 draft token，这些 token 的 KV 写入和验收逻辑本来就复杂。

如果再叠加非 1 的 CP interleave：

```text
draft token → global position → local CP slot → accepted/rejected 状态
```

需要 backend / metadata 明确支持，否则很容易写错 KV 或合并错 logits。

---

## 30. AttentionImplBase 中的 CP 能力字段

attention backend 通过这些字段声明能力：

```text
can_return_lse_for_decode
need_to_return_lse_for_decode
supports_pcp
supports_mtp_with_cp_non_trivial_interleave_size
dcp_world_size
dcp_rank
pcp_world_size
pcp_rank
```

含义分别是：

```text
can_return_lse_for_decode：backend 理论上能返回 decode LSE；
need_to_return_lse_for_decode：当前配置下必须返回 LSE；
supports_pcp：是否支持 prefill context parallel；
supports_mtp_with_cp_non_trivial_interleave_size：是否支持 MTP + CP 非平凡 interleave；
dcp_world_size / dcp_rank：当前 DCP group 信息；
pcp_world_size / pcp_rank：当前 PCP group 信息。
```

DCP 运行时真正依赖的是：

```text
backend 能拿到 partial output + partial LSE，
并且能用 CP group 正确通信合并。
```

---

## 31. Forward context 如何把 CP metadata 传给模型层

ModelRunner 准备好：

```text
attn_metadata
slot_mappings_by_layer
```

后，会通过 forward context 注入模型 forward。

模型里的 attention layer 不需要层层传参，它通过当前 forward context 拿到：

```text
当前 layer 的 AttentionMetadata；
当前 layer 的 slot_mapping。
```

DCP 下这些对象已经包含：

```text
CP-aware slot mapping；
DCP local seq lens；
DCP backend 需要的 block table / seq lens 信息。
```

---

## 32. 完整主链路

把 DCP 从配置到 backend 串起来：

```text
启动配置
  → ParallelConfig.decode_context_parallel_size
  → 校验 tp_size % dcp_size == 0
  → 初始化 get_dcp_group()

KV cache / Scheduler
  → resolve_kv_cache_block_sizes()
      → scheduler_block_size = block_size * dcp * pcp
  → Scheduler 保存 dcp_world_size / pcp_world_size
  → KVCacheManager 按 CP 对齐后的 block size 分配 blocks

Worker / ModelRunner
  → self.dcp_size / self.dcp_rank / self.cp_interleave
  → initialize_kv_cache()
      → max_num_blocks 按 block_size * dcp_size 缩小
      → BlockTables(cp_size, cp_rank, cp_interleave)
  → prepare_dcp_local_seq_lens()
  → prepare_attn()
      → gather_block_tables()
      → compute_slot_mappings()
  → build_attn_metadata()
      → CommonAttentionMetadata.dcp_local_seq_lens
      → block_table_tensor / slot_mapping

Attention backend
  → check_attention_cp_compatibility()
  → FlashAttentionImpl.forward()
  → _forward_with_dcp()
      → all_gather query
      → local context attention returns output + LSE
      → cp_lse_ag_out_rs / dcp_a2a_lse_reduce
      → query attention returns output + LSE
      → merge_attn_states()
  → final attention output
```

---

## 33. 和 TP / PP / DP 的区别

| 并行方式 | 切分对象 | 典型目的 | 是否改变 request 的 context 语义 |
|---|---|---|---|
| TP | 权重 / heads / hidden dim | 单层计算并行 | 不直接切 context |
| PP | layer pipeline | 模型层流水 | 不直接切 context |
| DP | 请求副本 / batch | 吞吐扩展 | 不切单请求 context |
| DCP | decode KV context | 长上下文 decode 降低单 rank KV 压力 | 切同一请求的 KV context |
| PCP | prefill context | 长 prefill attention 并行 | 切同一请求 prefill context |

DCP 的特殊点是：

```text
它复用 TP group 的 GPU，不增加 world size；
但会改变 KV cache 在这些 ranks 上的布局。
```

---

## 34. 容易混淆的点

### 34.1 DCP 是不是 tensor parallel？

不是。

TP 切权重 / heads；DCP 切 context KV。

但 DCP 复用 TP group 中的 GPU，所以配置上要求 `tp_size % dcp_size == 0`。

### 34.2 DCP 是不是只改通信？

不是。

DCP 同时影响：

```text
KV cache block 数；
slot mapping；
dcp_local_seq_lens；
attention metadata；
attention backend merge。
```

### 34.3 partial attention output 为什么不能直接 all-reduce？

因为每个 partial output 的 softmax 分母不同。

必须用 LSE 重新归一化后再合并。

### 34.4 can_return_lse_for_decode 和 need_to_return_lse_for_decode 是一回事吗？

不是。

```text
can_return_lse_for_decode：backend 有这个能力；
need_to_return_lse_for_decode：当前配置确实需要它返回 LSE。
```

DCP 会让后者成为必须条件。

### 34.5 DCP 会不会减少每个 rank 的 KV cache？

会。

ModelRunner 计算 max blocks 时使用：

```text
max_model_len / (block_size * dcp_size)
```

也就是每个 DCP rank 只保存上下文的一部分。

### 34.6 PCP 是否等同于 DCP？

不是。

PCP 面向 prefill 阶段，必须 backend 显式 `supports_pcp`。DCP 面向 decode 阶段，目前在 FlashAttention / FlashInfer 路径中更完整。

### 34.7 DCP 可以和所有 KV cache layout 一起用吗？

不能。

多 KV cache group / 多 block size 的 hybrid KV cache 当前不支持 context parallelism。

---

## 35. 最终可以记成一张表

| 阶段 | 关键函数 / 类 | 核心产物 | 作用 |
|---|---|---|---|
| 配置定义 | `ParallelConfig` | `decode_context_parallel_size`、`prefill_context_parallel_size` | 打开 DCP / PCP |
| 配置校验 | `ParallelConfig` post init | `tp_size % dcp_size == 0` | 确保 DCP 可复用 TP group |
| 进程组 | `get_dcp_group()`、`get_pcp_group()` | DCP / PCP group | 提供 CP 通信域 |
| KV 对齐 | `resolve_kv_cache_block_sizes()` | `scheduler_block_size = block_size * dcp * pcp` | 保证 prefix cache / scheduler 对齐 |
| Worker 初始化 | `GPUModelRunner.initialize_kv_cache()` | `BlockTables(cp_size, cp_rank, cp_interleave)` | 初始化 CP-aware block table |
| slot 映射 | `BlockTables.compute_slot_mappings()` | CP-aware `slot_mapping` | global position → local KV slot |
| 本地长度 | `prepare_dcp_local_seq_lens()` | `dcp_local_seq_lens` | 计算当前 DCP rank 本地 KV 长度 |
| metadata | `CommonAttentionMetadata` | block table / slot mapping / local seq lens | 传给 backend |
| 能力检查 | `check_attention_cp_compatibility()` | LSE / PCP / MTP 支持检查 | 防止 backend 不兼容 |
| DCP backend | `FlashAttentionImpl._forward_with_dcp()` | partial output + LSE | 执行 DCP attention |
| 合并 | `cp_lse_ag_out_rs()` / `dcp_a2a_lse_reduce()` / `merge_attn_states()` | final output | LSE 语义正确合并 |

---

## 36. 总结

Context Parallel 在 vLLM 中可以压缩成下面这条链路：

```text
ParallelConfig
  → DCP / PCP process group
  → KV cache block size 对齐
  → CP-aware block table
  → CP-aware slot mapping
  → dcp_local_seq_lens
  → CommonAttentionMetadata
  → backend 返回 partial output + LSE
  → CP group 通信合并
  → merge_attn_states 得到最终 attention output
```

如果只记住三句话：

```text
1. DCP 切的是 decode 阶段的 KV context，不是简单切 batch。
2. DCP 从 KV cache slot mapping 开始就改变 global position 到 local slot 的映射。
3. DCP attention output 必须依赖 LSE 合并，不能直接 sum / all-reduce。
```

因此，Context Parallel 和 attention 的关系非常直接：

```text
它把 attention 的 KV context 拆到多个 rank，
再要求 backend 用 softmax LSE 语义把各 rank 的 partial attention state 合回完整结果。
```
