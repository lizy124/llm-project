# 09. Attention backend 如何感知并行？

源码位置：

- `vllm/vllm/config/model.py`
- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/vllm/v1/attention/selector.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/flash_attn.py`
- `vllm/vllm/v1/attention/backends/flashinfer.py`
- `vllm/vllm/v1/attention/backends/mla/`
- `vllm/vllm/v1/attention/ops/common.py`
- `vllm/vllm/v1/attention/ops/dcp_alltoall.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/utils.py`

本问题关注：attention backend 如何受到 TP / PP / DP / EP / DCP / PCP 的影响，`num_heads / num_kv_heads` 如何按 TP 分布，GQA / MQA / MLA 在并行下如何执行，DCP 为什么要求 backend 返回 LSE，以及 FlashAttention / FlashInfer / Triton / FlashMLA 等 backend 对并行特性的支持边界。

---

## 1. 一句话回答

Attention 是 vLLM 并行体系中最敏感的执行点之一，因为它同时接触：

```text
TP：
  改变本 rank 的 Q heads / KV heads 数量。

PP：
  决定当前 rank 是否持有某些 attention layers 和它们的 KV cache。

DCP / PCP：
  改变一个 attention kernel 看到的上下文范围，并要求跨 rank 合并 partial attention。

DP：
  每个 replica 有独立 batch / KV cache / attention metadata，但需要在外层同步执行状态。

EP：
  不直接改 dense attention，但 MoE 模型中 attention 与 expert parallel 层交替出现，静态 forward context 会同时保存 attention 和 MoE 层。
```

如果只记一句话：

```text
Attention backend 感知并行的入口不是一个单点，而是 head 数量、KV cache spec、attention metadata、forward context 和 CP/DCP merge 共同作用的结果。
```

---

## 2. 本文要回答的问题

```text
TP 下 num_heads / num_kv_heads 如何变化？
GQA / MQA 的 KV heads 如何分配到 TP rank？
attention backend 选择时会看哪些并行相关能力？
attention metadata 中哪些字段和并行有关？
DCP 为什么需要 LSE？
FlashAttention 的 DCP 路径如何拆分和合并？
MLA backend 如何处理 prefill / decode / DCP？
PP 下 attention layer 的 KV cache 归属如何处理？
DP 下 attention state 是否 replica-local？
EP 与 dense attention 的边界在哪里？
```

---

## 3. 最小主链路

可以把 attention 与并行的关系压缩成下面这条线：

```text
ParallelConfig / ModelConfig
  → verify_with_parallel_config()
      校验 TP heads、DCP 与 GQA/MQA 约束、PP 支持性

  → model layer 构造 Attention / MLAAttention
      使用 model_config.get_num_attention_heads(parallel_config)
      使用 model_config.get_num_kv_heads(parallel_config)
      选择 attention backend

  → ModelRunner 初始化 KV cache
      get_kv_cache_spec() 记录本 rank num_kv_heads / head_size / block_size
      bind_kv_cache() 把本 rank KV cache 绑定到 attention layer

  → GPUModelRunner._build_attention_metadata()
      构造 CommonAttentionMetadata
      填入 block_table / slot_mapping / seq_lens / dcp_local_seq_lens
      交给 backend MetadataBuilder

  → set_forward_context(...)
      把 attn_metadata、slot_mapping、static_forward_context 放进 ForwardContext

  → Attention.forward() / MLAAttention.forward()
      reshape 本 rank Q/K/V heads
      用 slot_mapping 写本 rank KV cache
      调 backend impl.forward / forward_mha / forward_mqa
      必要时通过 DCP group 合并 partial attention 输出
```

---

## 4. TP 如何影响 attention heads

### 4.1 校验：总 attention heads 必须能整除 TP

`ModelConfig.verify_with_parallel_config()` 会先校验：

```text
total_num_attention_heads % tensor_parallel_size == 0
```

位置：`vllm/vllm/config/model.py:1159`

否则每个 TP rank 无法拿到相同数量的 Q heads。

### 4.2 本 rank num_heads

本 rank 的 attention heads 数量由：

```python
num_heads = total_num_attention_heads // tensor_parallel_size
```

位置：`vllm/vllm/config/model.py:1272`

这意味着模型定义里传给 `Attention(...)` 的 `num_heads` 已经是本 rank 局部值，而不是全局值。

### 4.3 本 rank num_kv_heads

普通 attention 的 KV heads 计算是：

```python
return max(1, total_num_kv_heads // tensor_parallel_size)
```

位置：`vllm/vllm/config/model.py:1259`

这个 `max(1, ...)` 很关键：

```text
如果 total KV heads 小于 TP size，KV heads 会在多个 TP ranks 上复制，
保证每个 rank 至少有 1 个本地 KV head。
```

这正是 MQA / GQA 和 TP 组合时最容易混淆的地方。

### 4.4 MLA 的 num_kv_heads 固定为 1

如果模型使用 MLA：

```python
if self.use_mla:
    return 1
```

位置：`vllm/vllm/config/model.py:1261`

`MLAAttention` 里也明确：

```python
self.num_kv_heads = 1
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:368`

原因是 MLA decode 路径在计算形态上更接近 MQA：KV cache 存的是 compressed latent KV，而不是每个 query head 一份传统 K/V。

---

## 5. MHA / GQA / MQA 在 TP 下怎么理解

### 5.1 MHA

MHA 通常有：

```text
total_num_attention_heads == total_num_kv_heads
```

TP 后每个 rank：

```text
local_q_heads  = total_q_heads / TP
local_kv_heads = total_kv_heads / TP
```

每个 rank 负责一组 Q/K/V heads，本地 attention 输出后再由后续 projection / TP 通信完成聚合。

### 5.2 GQA

GQA 通常有：

```text
total_q_heads > total_kv_heads
num_queries_per_kv = total_q_heads / total_kv_heads
```

TP 后可能出现两种情况：

```text
1. total_kv_heads >= TP：
   每个 rank 拿 total_kv_heads / TP 个 KV heads。

2. total_kv_heads < TP：
   每个 rank 至少拿 1 个 KV head，KV heads 在 TP ranks 上复制。
```

`FlashAttentionImpl` 中会用本地值计算：

```python
self.num_queries_per_kv = self.num_heads // self.num_kv_heads
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:663`

注意这里的 `num_heads / num_kv_heads` 都是本 rank 局部值。

### 5.3 MQA

MQA 可以看作：

```text
total_num_kv_heads = 1
```

如果 TP > 1，本地 KV heads 仍通过 `max(1, ...)` 保持为 1。也就是说多个 TP ranks 会各自持有可用的 KV head 副本或对应本地 KV cache 视图。

这避免了某些 TP rank 没有 KV head 可算的问题。

---

## 6. Attention 初始化时如何选择 backend

### 6.1 Attention.__init__ 的选择入口

普通 attention 在初始化时调用：

```python
self.attn_backend = get_attn_backend(
    head_size,
    dtype,
    kv_cache_dtype,
    use_mla=False,
    has_sink=self.has_sink,
    use_mm_prefix=self.use_mm_prefix,
    use_per_head_quant_scales=use_per_head_quant_scales,
    attn_type=attn_type,
)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:318`

MLA 则调用：

```python
self.attn_backend = get_attn_backend(
    self.head_size,
    dtype,
    kv_cache_dtype,
    use_mla=True,
    use_sparse=use_sparse,
    num_heads=self.num_heads,
)
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:387`

### 6.2 selector 会把能力需求打包成 AttentionSelectorConfig

`get_attn_backend()` 会构造：

```text
head_size
dtype
kv_cache_dtype
block_size
use_mla
has_sink
use_sparse
use_mm_prefix
use_per_head_quant_scales
attn_type
use_non_causal
use_batch_invariant
use_kv_connector
```

位置：`vllm/vllm/v1/attention/selector.py:20`

这些不全是“并行配置”，但很多会和并行一起决定 backend 能不能用：

```text
TP 改变本地 head 数 / kv head 数；
DCP 要求 backend 能返回 LSE 或有专门 merge 路径；
batch invariance 会限制某些 backend 与 prefix cache 的组合；
KV connector 要求 backend 支持 KV transfer 场景。
```

### 6.3 backend 基类定义能力边界

`AttentionBackend` 提供一组能力声明：

```text
supports_head_size
supports_dtype
supports_kv_cache_dtype
supports_block_size
supports_mm_prefix
supports_non_causal
supports_batch_invariance
supports_kv_connector
supports_attn_type
supports_per_head_quant_scales
get_required_kv_cache_layout
```

位置：`vllm/vllm/v1/attention/backend.py:55`

backend 不是在 forward 时才发现不支持，而是在选择和初始化阶段尽量过滤。

---

## 7. KV cache spec 如何体现 TP

`Attention.get_kv_cache_spec()` 会把本 rank 的 `num_kv_heads` 写入 KV cache spec。

普通 full attention：

```python
FullAttentionSpec(
    block_size=...,
    num_kv_heads=self.num_kv_heads,
    head_size=self.head_size,
    head_size_v=self.head_size_v,
    dtype=...,
)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:581`

Sliding window attention 类似，只是返回 `SlidingWindowSpec`。

MLA 返回：

```python
MLAAttentionSpec(
    block_size=...,
    num_kv_heads=1,
    head_size=self.head_size,
    dtype=...,
)
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:985`

这说明：

```text
KV cache 的物理形状是 per-rank 的，已经包含 TP 后的本地 KV heads。
```

---

## 8. forward context 如何把 metadata 交给 attention

### 8.1 Attention layer 被登记进 static_forward_context

`Attention.__init__()` 会把自己放到：

```python
compilation_config.static_forward_context[prefix] = self
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:410`

MLA 也一样：

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:472`

这使得后续 attention custom op 能通过 `layer_name` 找回 layer 对象。

### 8.2 KV cache 绑定

Worker 初始化 KV cache 后，会调用 `bind_kv_cache()`：

```text
kv_caches[layer_name] → forward_context[layer_name].kv_cache
```

位置：`vllm/vllm/v1/worker/utils.py:462`

它说明：

```text
每个 attention layer 的 kv_cache 是运行时绑定的，不是在 layer __init__ 时就有真实 cache tensor。
```

### 8.3 forward 时按 layer_name 取上下文

`get_attention_context(layer_name)` 从 forward context 里取：

```text
attn_metadata
attn_layer
kv_cache
layer_slot_mapping
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:670`

所以 attention forward 的隐式输入是：

```text
layer_name
  → static_forward_context[layer_name]
  → per-layer attn metadata
  → per-layer KV cache
  → per-layer slot mapping
```

---

## 9. CommonAttentionMetadata 中哪些字段和并行有关

`CommonAttentionMetadata` 是所有 backend builder 的公共输入。

位置：`vllm/vllm/v1/attention/backend.py:393`

和并行最相关的字段包括：

```text
block_table_tensor：
  当前 KV cache group 的 block table。TP/PP 后是本 rank 本地 KV cache 的索引视图。

slot_mapping：
  本轮 token 写本 rank KV cache 的 slot。

seq_lens / max_seq_len：
  每个请求全局逻辑序列长度。CP/DCP backend 会再转换成本地 context 长度。

query_start_loc / max_query_len：
  varlen batch 边界，影响 backend 是否能走 decode / prefill / cudagraph 路径。

dcp_local_seq_lens：
  DCP 下当前 rank 的本地 context 长度。

positions：
  token 逻辑位置，影响 paged KV slot、RoPE、某些稀疏 / MLA backend metadata。

encoder_seq_lens：
  encoder-decoder / cross attention 场景的 encoder 侧长度。
```

ModelRunner 在 DCP 下会额外计算：

```python
self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(...)
cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2357`

---

## 10. AttentionImplBase 如何自动感知 DCP / PCP

所有 attention impl 都继承 `AttentionImplBase`。

位置：`vllm/vllm/v1/attention/backend.py:734`

它在 `__new__()` 里尝试读取：

```python
self.dcp_world_size = get_dcp_group().world_size
self.dcp_rank = get_dcp_group().rank_in_group
self.pcp_world_size = get_pcp_group().world_size
self.pcp_rank = get_pcp_group().rank_in_group
```

位置：`vllm/vllm/v1/attention/backend.py:780`

并计算：

```text
total_cp_world_size = pcp_world_size * dcp_world_size
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
```

如果 group 还没初始化，则默认：

```text
dcp_world_size = 1, dcp_rank = 0
pcp_world_size = 1, pcp_rank = 0
```

这就是 attention backend “感知 CP/DCP”的最底层入口。

---

## 11. DCP 为什么需要 LSE

DCP 的核心问题是：一个 query 对完整 context 的 attention 被拆到多个 rank 上计算。

每个 rank 只能看到一部分 KV context，得到的是：

```text
partial_output_i = softmax(Q K_i^T) V_i
partial_lse_i    = logsumexp(Q K_i^T)
```

最终要得到对完整 K 的：

```text
softmax(Q [K_0, K_1, ...]^T) [V_0, V_1, ...]
```

不能简单把 `partial_output_i` 相加，因为每个 partial softmax 的归一化分母不同。

因此需要 LSE：

```text
用每个 rank 的 lse 恢复全局 softmax 分母，
再按正确权重合并 partial outputs。
```

`AttentionImplBase` 中的标记是：

```python
need_to_return_lse_for_decode = self.dcp_world_size > 1

随后兼容性检查会要求 backend 具备返回 decode LSE 的能力。
```

位置：`vllm/vllm/v1/attention/backend.py:803`

`ops/common.py` 中 `correct_attn_out()` 正是在做这件事：

```text
all-gather 各 rank lse
  → 计算 final lse
  → 用 exp(local_lse - final_lse) 修正 local output
  → reduce_scatter / all_reduce 合并 output
```

位置：`vllm/vllm/v1/attention/ops/common.py:110`

---

## 12. FlashAttention 的 DCP 路径

### 12.1 FlashAttention 支持返回 decode LSE

`FlashAttentionImpl` 声明：

```python
can_return_lse_for_decode = True
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:626`

这使它可以参与 DCP 合并。

### 12.2 MetadataBuilder 先构造 DCP 本地 context 长度

`FlashAttentionMetadataBuilder` 在初始化时读取 DCP group：

```python
self.dcp_world_size = get_dcp_group().world_size
self.dcp_rank = get_dcp_group().rank_in_group
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:353`

在 `build()` 中，如果 `dcp_world_size > 1`，它会：

```text
1. query_lens = query_start_loc 差分；
2. context_kv_lens = seq_lens - query_lens；
3. local_context_kv_lens = get_dcp_local_seq_lens(...);
4. max_dcp_context_kv_len 按 dcp_world_size 和 interleave size 取上界；
5. scheduler_metadata 用 local context length 生成。
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:498`

### 12.3 forward 时分两段计算

FlashAttention 普通路径中，如果 DCP 开启：

```python
if self.dcp_world_size > 1:
    self._forward_with_dcp(...)
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:810`

`_forward_with_dcp()` 做两段 attention：

```text
1. context attention：
   query across DCP all-gather，访问本 rank 的历史 KV context，返回 context_out + context_lse。

2. query attention：
   当前 rank 本轮 query 内部 causal attention，返回 query_out + query_lse。

3. merge_attn_states：
   用 LSE 把 context attention 和 query attention 合并。
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:962`

其中 context attention 会调用：

```python
query_across_dcp = get_dcp_group().all_gather(query, dim=1)
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:983`

然后通过：

```python
self.dcp_combine(..., get_dcp_group(), return_lse=True)
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:1018`

把跨 DCP rank 的 partial context 输出合并。

---

## 13. cp_lse_ag_out_rs 和 dcp_a2a_lse_reduce 的区别

FlashAttention / MLA 都会在 DCP 下选择合并函数。

默认路径：

```text
cp_lse_ag_out_rs
```

含义大致是：

```text
all-gather LSE
  → 修正 local output
  → reduce-scatter output head 维度
```

位置：`vllm/vllm/v1/attention/ops/common.py:212`

如果配置：

```text
parallel_config.dcp_comm_backend == "a2a"
```

则可能使用：

```text
dcp_a2a_lse_reduce
```

相关选择在 FlashAttention：

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:689`

MLA 中也有相同判断：

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:493`

两者目标相同：

```text
用 LSE 正确合并跨 context-parallel rank 的 partial attention states。
```

---

## 14. MLA attention 如何感知并行

### 14.1 MLA 初始化时本地 heads 已经是 TP 后的值

`MLAAttention` 接收的 `num_heads` 是本 rank local heads，并在初始化 backend 时传入：

```python
num_heads=self.num_heads
num_kv_heads=1
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:446`

### 14.2 MLA decode 是 MQA-style

文件顶部注释明确：

```text
For decode the attention "simulates" multi-head attention,
while the compute is similar to multi-query attention.
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:26`

MLA KV cache 存的是：

```text
kv_c：compressed KV latent
k_pe：position embedding 部分
```

而不是传统每个 KV head 一份 K/V。

### 14.3 MLA metadata 会拆 decode 和 prefill

`MLACommonMetadata` 额外记录：

```text
num_decodes
num_decode_tokens
num_prefills
prefill metadata
decode metadata
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:1274`

`MLACommonMetadataBuilder.build()` 会调用：

```python
split_decodes_and_prefills(...)
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:1623`

这使 MLA 可以：

```text
decode 走 forward_mqa；
prefill / chunked prefill 走 forward_mha 或专门 prefill backend。
```

### 14.4 MLA DCP 路径

MLA decode 中如果 `dcp_world_size > 1`：

```python
mqa_q = torch.cat(mqa_q, dim=-1)
mqa_q = get_dcp_group().all_gather(mqa_q, dim=1)
attn_out, lse = self.impl.forward_mqa(...)
attn_out = cp_lse_ag_out_rs(...) 或 dcp_a2a_lse_reduce(...)
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:787`

这和 FlashAttention 的核心思想一致：

```text
DCP 下先扩大 query heads / 可见上下文，再用 LSE 做正确合并。
```

### 14.5 MLA DCP 对 FP8 KV cache 的限制

MLA 当前代码中有断言：

```python
assert not fp8_attention, "DCP not support fp8 kvcache now."
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:787`

所以 MLA + DCP + FP8 KV cache 不是无条件可用。

---

## 15. PCP 如何影响 attention

PCP 是 prefill context parallel。

`AttentionImplBase` 会记录：

```text
pcp_world_size
pcp_rank
total_cp_world_size = pcp_world_size * dcp_world_size
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
```

位置：`vllm/vllm/v1/attention/backend.py:792`

`ParallelConfig.cp_kv_cache_interleave_size` 的注释说明，DCP 和 PCP 会共同决定 KV cache token 在 total CP ranks 上的 interleave 方式：

```text
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
total_cp_world_size = pcp_world_size * dcp_world_size
```

位置：`vllm/vllm/config/parallel.py:359`

因此 PCP 不是简单“多一个 group”，它会影响：

```text
- 本地 KV cache token 分布；
- prefill attention 的 context 切分；
- backend 是否支持 PCP；
- CP interleave size 是否与 block size / kernel 对齐。
```

---

## 16. PP 下 attention layer 和 KV cache 归属

Pipeline parallel 会把 transformer layers 切到不同 PP ranks。

`ModelConfig.get_layers_start_end_indices()` 根据：

```text
pp_rank = (rank // (tensor_parallel_size * prefill_context_parallel_size)) % pipeline_parallel_size
```

计算当前 rank 负责的 layer 范围。

位置：`vllm/vllm/config/model.py:1282`

这意味着：

```text
一个 PP rank 只构造 / 持有自己 stage 内的 attention layers；
它只绑定这些 layers 的 KV cache；
其他 stage 的 attention 不在当前 rank forward。
```

PP 执行链路中：

```text
非 first PP rank：接收上个 stage 的 intermediate tensors；
当前 stage：执行本 stage attention / MLP；
非 last PP rank：发送 intermediate tensors 给下个 stage；
last PP rank：才进入 logits / sampling。
```

所以 PP 对 attention 的影响不是改变 head 数，而是改变：

```text
当前 rank 有哪些 attention layer，以及 KV cache 属于哪些 layer。
```

---

## 17. DP 下 attention state 是否共享

DP 下每个 replica 有自己的：

```text
request batch
InputBatch
KV cache blocks
block table
slot mapping
attention metadata
```

也就是说 attention 的运行态是 replica-local 的。

DP group 不会让两个 replica 共享同一个 KV cache。

DP 主要影响：

```text
- 多个 replica 是否还有未完成请求的状态同步；
- pause / stop / engine progress 同步；
- MoE expert parallel 中 DP 与 TP 共同组成 EP group；
- 某些 profile / 0-token / external launcher 场景下的协调。
```

因此：

```text
DP 不直接改变单个 attention backend 的 Q/K/V 形状，
但会改变系统中有多少份独立 attention batch 在并行执行。
```

---

## 18. EP 和 attention 的关系

Expert parallel 主要作用于 MoE MLP 层，不直接改变 dense attention 的数学形式。

但它和 attention 有几个交叉点：

```text
1. MoE 模型中 attention 和 expert layers 交替出现；
2. static_forward_context 同时保存 Attention 和 FusedMoE 等运行时需要访问的 layer；
3. DP + EP 场景下 profile run / padding / zero fill 要保证 DP group 内 expert 输出一致；
4. sequence parallel MoE 可能要求 attention / projection 后的 hidden states 处于合适的 shard / replicate 状态。
```

例如 MLA profile run 中有注释：

```text
The zero fill is required when used with DP + EP
to ensure all ranks within a DP group compute the same expert outputs.
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:657`

所以 EP 不直接改 attention kernel，但会影响 attention 周边 tensor 在 MoE 前后的并行布局约束。

---

## 19. backend 能力边界如何看

### 19.1 backend class 负责声明“能不能用”

例如 FlashAttention backend 声明：

```text
supports_batch_invariance = True
supports_non_causal = True
supports_attn_type 支持 decoder / encoder / encoder_only / encoder_decoder
get_supported_kernel_block_sizes 要求 block size 是 16 的倍数
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:68`

### 19.2 impl class 负责声明“能不能返回 LSE / 支持 CP”

`AttentionImplBase` 有：

```text
can_return_lse_for_decode
supports_pcp
supports_mtp_with_cp_non_trivial_interleave_size
supports_quant_query_input
```

位置：`vllm/vllm/v1/attention/backend.py:734`

FlashAttention 设置：

```python
can_return_lse_for_decode = True
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:626`

MLA backend 则要看具体实现类是否支持 `forward_mqa` 返回 LSE、是否支持 sparse、是否支持 varlen query。

### 19.3 metadata builder 负责声明 cudagraph / reorder 能力

`AttentionMetadataBuilder` 有：

```text
_cudagraph_support
reorder_batch_threshold
supports_update_block_table
```

位置：`vllm/vllm/v1/attention/backend.py:565`

DCP 下如果 backend 不支持 varlen DCP，builder 会把 batch reorder threshold 限制成 decode-only：

```python
if decode_context_parallel_size > 1 and not supports_dcp_with_varlen:
    self.reorder_batch_threshold = 1
```

位置：`vllm/vllm/v1/attention/backend.py:625`

---

## 20. 和 paged KV cache 的关系

Attention 并行不是只影响 heads，也影响 KV cache layout。

普通 attention forward：

```text
query: [num_tokens, local_num_heads, head_size]
key:   [num_tokens, local_num_kv_heads, head_size]
value: [num_tokens, local_num_kv_heads, head_size_v]
kv_cache: per-rank cache tensor
```

KV 写入依赖：

```text
slot_mapping：本轮 token 写本 rank 哪个 KV slot
```

KV 读取依赖：

```text
block_table + seq_lens：历史 KV blocks 在哪里、每个请求读多长
```

DCP / PCP 下还要考虑：

```text
某个 position 的 KV 是否属于当前 CP rank；
本地 context length 是多少；
跨 rank partial attention 如何合并。
```

所以 paged KV cache 是 attention backend 和并行 group 之间的重要桥梁。

---

## 21. 与 CUDA graph / batch invariance 的关系

Attention metadata builder 会声明 cudagraph 支持等级：

```text
ALWAYS
UNIFORM_BATCH
UNIFORM_SINGLE_TOKEN_DECODE
NEVER
```

位置：`vllm/vllm/v1/attention/backend.py:548`

FlashAttention builder 对 FA3 / XPU 设置更强支持，对 FA2 则更保守。

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:294`

batch invariance 也会影响 backend：

```text
VLLM_BATCH_INVARIANT 开启时，selector 会把 use_batch_invariant 传入 backend 选择；
某些 backend + prefix cache 组合会被禁用或报错。
```

普通 Attention 中有示例：

```text
FLASHINFER / TRITON_MLA + prefix caching + batch invariance 时禁用 prefix cache。
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:341`

---

## 22. 容易混淆的点

### 22.1 Attention 的 num_heads 是全局值吗？

不是。

进入 `Attention` / `MLAAttention` 的通常已经是 TP 后的本 rank local heads。

### 22.2 num_kv_heads 小于 TP size 时某些 rank 没 KV 吗？

不是。

`get_num_kv_heads()` 使用 `max(1, total_num_kv_heads // TP)`，保证每个 rank 至少有 1 个 KV head。

### 22.3 DCP 是不是等同于 TP？

不是。

DCP 复用 TP group 内的 GPU，但它切的是 context / attention 计算，不是模型权重矩阵本身。

### 22.4 DCP 输出能不能直接 all_reduce？

通常不能。

attention 的 softmax 分母跨 context shard 不同，需要 LSE 修正后才能合并。

### 22.5 PP 会改变 attention kernel 吗？

通常不会。

PP 改变的是当前 rank 持有哪些 layers 和 KV cache；attention kernel 的本地 head / KV 形状主要由 TP / CP 决定。

### 22.6 DP 会共享 KV cache 吗？

不会。

DP replica 的 attention batch / KV cache / metadata 是各自独立的。

### 22.7 EP 会改变 dense attention 的 Q/K/V 吗？

不会。

EP 主要作用在 MoE expert 层，但会影响 attention 前后 hidden states 的并行布局和 DP/EP 一致性要求。

---

## 23. 最终可以记成一张表

| 并行维度 | 对 attention 的直接影响 | 关键源码 |
|---|---|---|
| TP | 切分 Q heads / KV heads，决定本 rank `num_heads / num_kv_heads` | `config/model.py:1259`、`attention.py:304` |
| PP | 决定当前 rank 持有哪些 attention layers / KV cache | `config/model.py:1282`、`worker/utils.py:462` |
| DCP | 切分 decode context，要求 LSE merge partial attention | `backend.py:780`、`flash_attn.py:962`、`ops/common.py:110` |
| PCP | 切分 prefill context，和 DCP 共同组成 total CP rank | `backend.py:792`、`config/parallel.py:359` |
| DP | replica-local attention state，外层同步执行状态 | `config/parallel.py:688` |
| EP | 不改 dense attention kernel，但影响 MoE 交替层和 hidden layout | `parallel_state.py:1201`、`mla_attention.py:657` |
| paged KV | 本 rank KV cache 读写依赖 block table / slot mapping | `gpu_model_runner.py:2216`、`attention.py:713` |

---

## 24. 一句话总结

Attention backend 感知并行的完整链路是：

```text
ModelConfig / ParallelConfig
  → 计算本 rank heads 与 KV heads
  → Attention / MLAAttention 选择 backend
  → KV cache spec 记录本 rank KV 形状
  → ModelRunner 构造 per-layer metadata
  → AttentionImplBase 读取 DCP / PCP group
  → backend forward 用本 rank Q/K/V + paged KV cache 执行
  → DCP / PCP 场景用 LSE 正确合并 partial attention
```

如果只记最后一句：

```text
TP 决定 attention 的 head 形状，PP 决定 layer 归属，DCP/PCP 决定 context 切分与 LSE 合并，DP/EP 主要影响 attention 周边的执行一致性和 hidden state 布局。
```
