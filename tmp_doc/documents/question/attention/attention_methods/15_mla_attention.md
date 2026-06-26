# 15. MLA Attention：Multi-Head Latent Attention 如何接入 vLLM？

源码位置：

- `vllm/vllm/model_executor/layers/mla.py`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/selector.py`
- `vllm/vllm/v1/attention/backends/registry.py`
- `vllm/vllm/v1/attention/backends/mla/`
- `vllm/vllm/v1/attention/backends/mla/prefill/`
- `vllm/vllm/v1/attention/ops/flashmla.py`
- `vllm/vllm/v1/kv_cache_interface.py`
- `vllm/vllm/model_executor/models/deepseek_v2.py`

本文关注：MLA（Multi-Head Latent Attention）在 vLLM 中不是普通 MHA / GQA 的一个小分支，而是同时改变了模型层投影方式、KV cache 表示、attention metadata、prefill / decode kernel 路径和 backend 选择逻辑的一整套 attention 接入。

---

## 1. 一句话回答

MLA 的核心变化是：KV cache 不再保存每个 head 展开的 `K / V`，而是保存每个 token 的压缩 latent KV 表示 `kv_c` 和 RoPE 位置部分 `k_pe`。

在 vLLM 中，主链路可以概括为：

```text
DeepSeek / MLA 模型层
  → MultiHeadLatentAttentionWrapper
  → 生成 q、kv_c_normed、k_pe
  → MLAAttention
  → 写入 MLA KV cache：concat(kv_c_normed, k_pe)
  → MLACommonMetadataBuilder 构造 prefill / decode metadata
  → prefill 走 compute-friendly MHA-style 路径
  → decode 走 data-movement-friendly MQA-style 路径
  → backend.forward_mqa / prefill_backend.run_prefill_*
  → v_up_proj + o_proj
```

所以可以先记住一句话：

```text
普通 attention cache 的是展开后的 K/V；
MLA cache 的是压缩 latent KV + RoPE K；
vLLM 再按 prefill / decode 场景选择“展开计算”还是“压缩计算”。
```

---

## 2. 本文要回答的问题

```text
1. MLA 和普通 MHA / GQA 的结构差异是什么？
2. MLA 的 KV cache 到底保存什么？
3. MultiHeadLatentAttentionWrapper 和 MLAAttention 各自负责什么？
4. MLACommonMetadata / MLACommonMetadataBuilder 是什么？
5. MLA prefill 和 decode 为什么走不同路径？
6. FlashMLA / TritonMLA / FlashInferMLA 等 backend 如何接入？
7. DeepSeek V2 / V3 类模型在 vLLM 中如何走 MLA 路径？
8. MLA 对 KV connector、slot mapping、paged cache 有什么特殊要求？
9. Sparse MLA / DeepSeek V3.2 / FP8 cache 有哪些额外变化？
```

---

## 3. MLA 和普通 MHA / GQA 的根本差异

普通 MHA / GQA 的 KV cache 通常保存的是：

```text
K cache: [num_blocks, block_size, num_kv_heads, head_size]
V cache: [num_blocks, block_size, num_kv_heads, v_head_size]
```

也就是说，attention kernel decode 时直接从 cache 中取出已经展开好的 K/V。

MLA 的思路不同。以 DeepSeek V2 / V3 风格为例，模型先把 hidden states 投影到低秩 latent 空间：

```text
q_c      = hidden_states → q latent
kv_c     = hidden_states → compressed KV latent
k_pe     = hidden_states → RoPE K position part
q_nope   = q_c → per-head no-RoPE Q
q_pe     = q_c → per-head RoPE Q
k_nope   = kv_c → per-head no-RoPE K
v        = kv_c → per-head V
```

区别在于：

```text
普通 MHA / GQA：
  cache K/V 展开结果，decode 直接读 K/V。

MLA：
  cache kv_c + k_pe，
  prefill 时可以临时展开成 K/V 做 compute-friendly attention，
  decode 时尽量不展开全部 K/V，而是把 Q 投到 latent 空间做 MQA-style attention。
```

源码顶部也直接解释了两种 MLA 计算方式：

- `forward_mha`：compute-friendly，适合 prefill；见 `vllm/vllm/model_executor/layers/attention/mla_attention.py:66`
- `forward_mqa`：data-movement-friendly，适合 decode；见 `vllm/vllm/model_executor/layers/attention/mla_attention.py:94`

---

## 4. MLA 中几个维度的含义

`mla_attention.py` 文件顶部给了一组非常关键的符号，可以把它们记成下面这张表。

| 符号 / 字段 | 含义 | DeepSeek V3 常见值 | 作用 |
|---|---|---:|---|
| `N` | attention heads 数 | 128 | Q 的 head 数 |
| `Lq` / `q_lora_rank` | Q latent rank | 1536 | Q 低秩压缩维度 |
| `Lkv` / `kv_lora_rank` | KV latent rank | 512 | KV cache 的主要压缩维度 |
| `P` / `qk_nope_head_dim` | no-RoPE Q/K 维度 | 128 | 不做 RoPE 的 QK 部分 |
| `R` / `qk_rope_head_dim` | RoPE Q/K 维度 | 64 | 位置编码部分 |
| `V` / `v_head_dim` | V head 维度 | 128 | attention 输出每 head 维度 |
| `head_size` | MLA cache 语义 head size | `Lkv + R` | cache 中每 token 保存的总维度 |

在 `MLAAttention.__init__()` 中，MLA 的 cache head size 被定义为：

```text
head_size = kv_lora_rank + qk_rope_head_dim
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:364`

这说明 MLA 的 cache 维度不是 `qk_nope_head_dim + qk_rope_head_dim`，也不是普通 V head dim，而是：

```text
压缩 KV latent 维度 + RoPE K 维度
```

---

## 5. 模型层如何接入 MLA

以 DeepSeek V2 / V3 路径为例，模型层入口在：

```text
vllm/vllm/model_executor/models/deepseek_v2.py
```

### 5.1 DecoderLayer 如何决定走 MLA

`DeepseekV2DecoderLayer` 会先判断是否是普通 MHA：

```text
如果是旧 deepseek 或 qk_nope_head_dim / qk_rope_head_dim 为 0：
  → DeepseekAttention / DeepseekV2Attention

否则如果 model_config.use_mla 为 true：
  → DeepseekV2MLAAttention
```

对应代码：`vllm/vllm/model_executor/models/deepseek_v2.py:1121` 到 `vllm/vllm/model_executor/models/deepseek_v2.py:1138`

所以 `use_mla` 是 DeepSeek 类模型进入 MLA 路径的关键开关。

### 5.2 DeepseekV2MLAAttention 构造哪些模块

`DeepseekV2MLAAttention` 会构造 MLA 需要的投影模块：

```text
fused_qkv_a_proj / kv_a_proj_with_mqa
q_a_layernorm
q_b_proj / q_proj
kv_a_layernorm
kv_b_proj
o_proj
rotary_emb
indexer  # sparse MLA 场景
```

这些模块被打包成 `MLAModules`，再交给 `MultiHeadLatentAttentionWrapper`：

```text
DeepseekV2MLAAttention
  → MLAModules
  → MultiHeadLatentAttentionWrapper
```

对应代码：

- `MLAModules` 定义：`vllm/vllm/model_executor/layers/mla.py:13`
- DeepSeek 组装 `MLAModules`：`vllm/vllm/model_executor/models/deepseek_v2.py:1051`
- 创建 `MultiHeadLatentAttentionWrapper`：`vllm/vllm/model_executor/models/deepseek_v2.py:1071`

### 5.3 MultiHeadLatentAttentionWrapper 做什么

`MultiHeadLatentAttentionWrapper` 是模型结构层，负责把 hidden states 变成 MLA attention op 能消费的三类输入：

```text
q            # [tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
kv_c_normed  # [tokens, kv_lora_rank]
k_pe         # [tokens, 1, qk_rope_head_dim]
```

它的核心步骤是：

```text
1. 从 hidden_states 生成 q latent 和 kv latent。
2. 对 q latent 做 q_a_layernorm，再通过 q_b_proj 展开成 per-head Q。
3. 从 kv latent 中拆出 kv_c 和 k_pe。
4. 对 kv_c 做 kv_a_layernorm，得到 kv_c_normed。
5. 对 q 的 RoPE 部分和 k_pe 应用 rotary embedding。
6. 如果是 sparse MLA，调用 indexer 生成 top-k token 索引。
7. 调用 MLAAttention(q, kv_c_normed, k_pe)。
8. 最后通过 o_proj 投影回 hidden size。
```

对应代码：`vllm/vllm/model_executor/layers/mla.py:120` 到 `vllm/vllm/model_executor/layers/mla.py:182`

可以把它理解为：

```text
MultiHeadLatentAttentionWrapper 负责“模型结构侧的 MLA 投影”；
MLAAttention 负责“cache 更新 + attention kernel 调用”。
```

---

## 6. MLAAttention 是 MLA 执行层的核心

`MLAAttention` 定义在：

```text
vllm/vllm/model_executor/layers/attention/mla_attention.py:322
```

它的职责和普通 `Attention` 很像，但输入和 cache 语义完全不同。

### 6.1 初始化阶段：选择 MLA backend

`MLAAttention.__init__()` 如果没有显式传 `attn_backend`，会调用：

```python
get_attn_backend(
    self.head_size,
    dtype,
    kv_cache_dtype,
    use_mla=True,
    use_sparse=use_sparse,
    num_heads=self.num_heads,
)
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:387` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:394`

这里最关键的是：

```text
use_mla=True
```

它会让 attention selector 只选择 `is_mla() == True` 的 backend。

### 6.2 初始化阶段：创建 backend impl

选出 backend 后，`MLAAttention` 会创建 backend 对应的 impl：

```text
impl_cls = attn_backend.get_impl_cls()
self.impl = impl_cls(... MLA args ...)
```

传入的 MLA 专属参数包括：

```text
q_lora_rank
kv_lora_rank
qk_nope_head_dim
qk_rope_head_dim
qk_head_dim
v_head_dim
kv_b_proj
indexer
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:446` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:468`

### 6.3 初始化阶段：创建 prefill backend

MLA 的 decode backend 和 prefill backend 是分开的。

`MLAAttention` 会额外选择 MLA prefill backend：

```text
prefill_backend_cls = get_mla_prefill_backend(vllm_config)
self.prefill_backend = prefill_backend_cls(...)
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:478` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:487`

这意味着：

```text
FlashMLA / TritonMLA / FlashInferMLA 等主要控制 decode-style forward_mqa；
MLA prefill 则通过 mla/prefill/ 下的 backend 单独选择。
```

---

## 7. MLA 的 KV cache 保存什么

### 7.1 普通 attention 的 cache 是 K + V

普通 full attention 的 page size 通常是：

```text
block_size * num_kv_heads * (head_size + head_size_v) * dtype_size
```

对应 `FullAttentionSpec.real_page_size_bytes`：`vllm/vllm/v1/kv_cache_interface.py:308`

### 7.2 MLA 的 cache 是单向量

`MLAAttention.get_kv_cache_spec()` 返回的是 `MLAAttentionSpec`：

```text
MLAAttentionSpec(
  block_size=cache_config.block_size,
  num_kv_heads=1,
  head_size=kv_lora_rank + qk_rope_head_dim,
  dtype=kv_cache_dtype,
  cache_dtype_str=cache_config.cache_dtype,
)
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:985` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:995`

`MLACommonBackend.get_kv_cache_shape()` 也明确返回：

```text
(num_blocks, block_size, head_size)
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:1201` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:1208`

也就是说 MLA cache 的形状可以理解为：

```text
[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
```

而不是普通 attention 的：

```text
[num_blocks, block_size, num_kv_heads, K/V head dim]
```

### 7.3 写 cache 时做 concat_and_cache_mla

`MLAAttention.forward()` 会先从 forward context 中拿到：

```text
attn_metadata
slot_mapping
self.kv_cache
```

然后调用：

```text
self.impl.do_kv_cache_update(
  kv_c_normed,
  k_pe,
  kv_cache,
  slot_mapping,
  kv_cache_dtype,
  k_scale,
)
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:553` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:579`

真正写 cache 的默认实现在 `MLAAttentionImpl.do_kv_cache_update()`：

```text
ops.concat_and_cache_mla(
  kv_c_normed,
  k_pe.squeeze(1),
  kv_cache,
  slot_mapping.flatten(),
  kv_cache_dtype,
  scale,
)
```

对应代码：`vllm/vllm/v1/attention/backend.py:1043` 到 `vllm/vllm/v1/attention/backend.py:1063`

这一步说明 cache 中每个 slot 实际写入的是：

```text
concat(kv_c_normed, k_pe)
```

### 7.4 MLA cache 和 slot mapping 的关系

MLA 仍然使用 vLLM 的 paged KV cache 和 slot mapping：

```text
Scheduler 分配 block
  → ModelRunner 构造 block_table / slot_mapping
  → MLAAttention 根据 slot_mapping 写当前 token 的 MLA cache
  → backend 根据 block_table / seq_lens 读取历史 MLA cache
```

差异不在 block 管理本身，而在每个 slot 保存的数据内容。

普通 attention slot 保存：

```text
K + V
```

MLA slot 保存：

```text
kv_c_normed + k_pe
```

---

## 8. MLACommonMetadata 是什么

`MLACommonMetadata` 定义在：

```text
vllm/vllm/model_executor/layers/attention/mla_attention.py:1275
```

它是 MLA backend 的公共 metadata，和普通 attention metadata 相比，多了 MLA 特有的 prefill / decode 拆分信息。

核心字段包括：

```text
num_reqs
max_query_len
max_seq_len
num_actual_tokens
query_start_loc
slot_mapping
num_decodes
num_decode_tokens
num_prefills
prefill
decode
head_dim
```

其中最重要的是：

```text
num_decodes / num_decode_tokens / num_prefills
```

因为 MLA 的同一个 batch 会被拆成：

```text
decode tokens：走 forward_mqa
prefill tokens：走 forward_mha
```

### 8.1 prefill metadata

`MLACommonPrefillMetadata` 定义在：

```text
vllm/vllm/model_executor/layers/attention/mla_attention.py:1230
```

它保存 prefill 路径需要的信息：

```text
block_table
query_start_loc
max_query_len
chunked_context
q_data_type
output_dtype
prefill_backend
```

其中 `chunked_context` 是 MLA 相比普通 attention 很重要的扩展，用于处理 chunked prefill 中已有 context 过长的问题。

### 8.2 decode metadata

`MLACommonDecodeMetadata` 定义在：

```text
vllm/vllm/model_executor/layers/attention/mla_attention.py:1264
```

它保存 decode 路径需要的信息：

```text
block_table
seq_lens
dcp_tot_seq_lens
```

FlashMLA 会在它的子类 `FlashMLADecodeMetadata` 中额外加入：

```text
scheduler_metadata
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashmla.py:108`

---

## 9. MLACommonMetadataBuilder 如何构造 metadata

`MLACommonMetadataBuilder` 定义在：

```text
vllm/vllm/model_executor/layers/attention/mla_attention.py:1385
```

它接收 ModelRunner 构造出来的 `CommonAttentionMetadata`，再把它翻译成 MLA 专用 metadata。

主流程是：

```text
CommonAttentionMetadata
  → split_decodes_and_prefills()
  → 构造 prefill metadata
  → 构造 decode metadata
  → 返回 MLACommonMetadata
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:1600` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:1878`

### 9.1 为什么要 split decodes and prefills

MLA 对 prefill 和 decode 使用不同策略：

```text
prefill：Sq / Skv 比较大，展开 K/V 后做 MHA 更合适；
decode：Sq / Skv 很小，读大量展开 K/V 的代价高，所以用 latent cache 做 MQA-style attention。
```

因此 builder 会调用：

```text
split_decodes_and_prefills(...)
```

把 batch 拆成：

```text
num_decodes
num_prefills
num_decode_tokens
num_prefill_tokens
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:1623` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:1632`

### 9.2 decode_threshold 和 query_len_support

不同 MLA decode backend 支持的 query length 不同。

`QueryLenSupport` 有三档：

```text
SINGLE_ONLY：只支持 query_len = 1
UNIFORM：支持 batch 内统一 query_len
VARLEN：支持 batch 内可变 query_len
```

定义位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:1130`

builder 会用 `reorder_batch_threshold` 来决定哪些请求归入 decode 路径。比如 FlashMLA：

```text
query_len_support = UNIFORM
reorder_batch_threshold = 128
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashmla.py:118` 到 `vllm/vllm/v1/attention/backends/mla/flashmla.py:122`

这就是为什么部分小 prefill / spec decode query 也可能走 decode-style 路径。

### 9.3 chunked prefill metadata

当 prefill 请求前面已经有 context 时，MLA 如果把所有历史 `kv_c` 都一次性展开为 per-head K/V，显存会很大。

所以 builder 会为 context 构造 chunked metadata：

```text
context_lens
max_context_chunk
chunk_starts / chunk_ends
cu_seq_lens
token_to_seq
workspace
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:1657` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:1833`

这套 metadata 的作用是：

```text
把已有 context 按 chunk 从 MLA cache 中 gather 出来，
每次只展开一块 context 的 K/V，
计算 attention 后用 merge_attn_states 合并结果。
```

---

## 10. MLA forward 的主链路

`MLAAttention.forward_impl()` 是 MLA attention 真正执行 prefill / decode 的地方。

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:609`

主流程可以压缩成：

```text
MLAAttention.forward_impl(q, kv_c_normed, k_pe, kv_cache, attn_metadata)
  → 去掉 CUDA graph padding，只保留 num_actual_tokens
  → 判断 FP8 KV cache
  → 根据 metadata 拆出 num_mqa_tokens / num_mha_tokens
  → prefill tokens 调 impl.forward_mha(...)
  → decode tokens 调 impl.forward_mqa(...)
  → decode 结果再做 v_up_proj
  → 返回 output
```

### 10.1 prefill：forward_mha

prefill 走 `MLACommonImpl.forward_mha()`：

```text
q
kv_c_normed
k_pe
kv_cache
attn_metadata
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:2275`

它会先把当前新 token 的 compressed KV 展开：

```text
kv_nope = kv_b_proj(kv_c_normed)
k_nope, v = split(kv_nope)
k = concat(k_nope, k_pe)
```

然后调用 prefill backend：

```text
prefill_backend.run_prefill_new_tokens(q, k, v, ...)
```

如果当前 prefill 还有历史 context，则继续：

```text
1. 从 kv_cache 中按 block_table gather 历史 MLA cache。
2. 拆出 kv_c_normed 和 k_pe。
3. 对历史 kv_c_normed 做 kv_b_proj，展开成 k_nope / v。
4. 调 run_prefill_context_chunk。
5. 用 merge_attn_states 把历史 context attention 和当前 token attention 合并。
```

对应代码：

- 当前 token prefill：`vllm/vllm/model_executor/layers/attention/mla_attention.py:2302` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:2323`
- context chunk prefill：`vllm/vllm/model_executor/layers/attention/mla_attention.py:2060` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:2165`

### 10.2 decode：forward_mqa

decode 走 `backend impl.forward_mqa()`，它是每个 MLA decode backend 必须实现的方法。

在调用 backend 前，公共逻辑会先把 decode Q 做一次变换：

```text
q = [q_nope, q_pe]
q_nope: [B, N, P]
q_pe:   [B, N, R]

mqa_ql_nope = q_nope × W_UK
mqa_q = concat(mqa_ql_nope, q_pe)
```

这里 `mqa_ql_nope` 的维度是 latent KV 空间，也就是 `kv_lora_rank`。

然后 backend 的 `forward_mqa()` 在 cache 上做 MQA-style attention：

```text
Q: concat(q projected to latent, q_pe)
K: concat(kv_c, k_pe) from cache
V: kv_c from cache
```

得到的输出还是 latent 空间：

```text
attn_out: [B, N, kv_lora_rank]
```

最后公共逻辑调用：

```text
_v_up_proj(attn_out)
```

把 latent 输出映射成：

```text
[B, N, v_head_dim]
```

对应代码：

- decode Q 转换：`vllm/vllm/model_executor/layers/attention/mla_attention.py:724` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:797`
- V up projection：`vllm/vllm/model_executor/layers/attention/mla_attention.py:816` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:817`
- `_v_up_proj` 实现：`vllm/vllm/model_executor/layers/attention/mla_attention.py:997`

---

## 11. FlashMLA backend 如何接入

FlashMLA 的入口在：

```text
vllm/vllm/v1/attention/backends/mla/flashmla.py
vllm/vllm/v1/attention/ops/flashmla.py
```

### 11.1 FlashMLABackend 的能力声明

`FlashMLABackend` 继承 `MLACommonBackend`，声明：

```text
get_name() = FLASHMLA
block size = 64
supported dtype = fp16 / bf16
supported kv cache dtype = auto / fp16 / bf16 / fp8 / fp8_e4m3
supports compute capability = SM90 / SM100
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashmla.py:47` 到 `vllm/vllm/v1/attention/backends/mla/flashmla.py:105`

同时它会通过 `supports_combination()` 检查底层 FlashMLA 扩展是否可用：

```text
is_flashmla_dense_supported()
is_flashmla_sparse_supported()
```

FlashMLA 扩展可用性检查在：`vllm/vllm/v1/attention/ops/flashmla.py:33` 到 `vllm/vllm/v1/attention/ops/flashmla.py:78`

### 11.2 FlashMLA metadata

FlashMLA 的 decode metadata 比通用 MLA 多一个字段：

```text
scheduler_metadata: FlashMLASchedMeta
```

构造时会调用：

```text
get_mla_metadata(seq_lens_device, num_q_tokens_per_head_k, 1, ...)
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashmla.py:161` 到 `vllm/vllm/v1/attention/backends/mla/flashmla.py:210`

这里的 `1` 表示 decode path 按 MQA 方式处理，KV head 数等价为 1。

### 11.3 FlashMLA decode forward

FlashMLA 的 decode 实现在：

```text
FlashMLAImpl.forward_mqa()
```

位置：`vllm/vllm/v1/attention/backends/mla/flashmla.py:263`

核心调用是：

```text
flash_mla_with_kvcache(
  q,
  k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),
  block_table,
  cache_seqlens,
  head_dim_v=kv_lora_rank,
  tile_scheduler_metadata,
  softmax_scale,
  causal=True,
)
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashmla.py:329` 到 `vllm/vllm/v1/attention/backends/mla/flashmla.py:339`

如果是 FP8 KV cache，则调用：

```text
flash_mla_with_kvcache_fp8(...)
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashmla.py:314` 到 `vllm/vllm/v1/attention/backends/mla/flashmla.py:327`

所以 FlashMLA backend 本质上接管的是：

```text
MLA decode 的 MQA-style kernel
```

而不是整个 `MultiHeadLatentAttentionWrapper`。

---

## 12. TritonMLA backend 如何接入

Triton MLA 的入口在：

```text
vllm/vllm/v1/attention/backends/mla/triton_mla.py
```

### 12.1 TritonMLABackend 的特点

它声明：

```text
get_name() = TRITON_MLA
block size = multiple of 16
支持 fp16 / bf16
支持 fp8 KV cache
supports_batch_invariance = True
```

对应代码：`vllm/vllm/v1/attention/backends/mla/triton_mla.py:36` 到 `vllm/vllm/v1/attention/backends/mla/triton_mla.py:87`

### 12.2 TritonMLA decode forward

`TritonMLAImpl.forward_mqa()` 会调用通用 Triton decode attention kernel：

```text
decode_attention_fwd(..., is_mla=True)
```

对应代码：`vllm/vllm/v1/attention/backends/mla/triton_mla.py:144` 到 `vllm/vllm/v1/attention/backends/mla/triton_mla.py:223`

其中 `is_mla=True` 告诉 kernel：

```text
KV cache 不是普通 K/V 双 cache，
而是单个 concat(kv_c, k_pe) cache。
```

TritonMLA 还会根据 `max_seq_len` 和 SM 数决定 `num_kv_splits`，用于 decode 阶段分片归约。

---

## 13. FlashInferMLA 和 prefill backend 的关系

FlashInferMLA 的入口在：

```text
vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py
```

它和 FlashMLA / TritonMLA 一样，是 MLA decode backend：

```text
get_name() = FLASHINFER_MLA
get_impl_cls() = FlashInferMLAImpl
get_builder_cls() = FlashInferMLAMetadataBuilder
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:38` 到 `vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:107`

FlashInferMLA 有一个额外点：

```text
get_required_kv_cache_layout() = HND
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:103` 到 `vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:105`

attention selector 会读取 backend 的 required layout，并设置 KV cache layout：

```text
backend.get_required_kv_cache_layout()
  → set_kv_cache_layout(required_layout)
```

对应代码：`vllm/vllm/v1/attention/selector.py:132` 到 `vllm/vllm/v1/attention/selector.py:142`

这说明部分 MLA backend 不只要求特定 kernel，也要求特定 cache layout。

---

## 14. MLA prefill backend 如何选择

MLA prefill backend 的选择逻辑在：

```text
vllm/vllm/v1/attention/backends/mla/prefill/selector.py
```

主入口是：

```text
get_mla_prefill_backend(vllm_config)
```

对应代码：`vllm/vllm/v1/attention/backends/mla/prefill/selector.py:72`

选择逻辑是：

```text
如果用户显式指定 attention_config.mla_prefill_backend：
  → 校验后使用用户指定 backend

否则自动按设备能力选择：
  Blackwell / SM100：FLASH_ATTN → TRTLLM_RAGGED → FLASHINFER → TOKENSPEED_MLA
  Hopper / 其他：FLASH_ATTN
```

对应代码：`vllm/vllm/v1/attention/backends/mla/prefill/selector.py:48` 到 `vllm/vllm/v1/attention/backends/mla/prefill/selector.py:69`

prefill backend 的抽象接口在：

```text
vllm/vllm/v1/attention/backends/mla/prefill/base.py
```

必须实现：

```text
run_prefill_new_tokens(...)
run_prefill_context_chunk(...)
```

对应代码：`vllm/vllm/v1/attention/backends/mla/prefill/base.py:129` 到 `vllm/vllm/v1/attention/backends/mla/prefill/base.py:149`

所以可以把 MLA backend 分成两层：

```text
attention backend：负责 decode / metadata builder / KV cache shape
prefill backend：负责 prefill MHA-style kernel
```

---

## 15. attention selector 如何保证选到 MLA backend

attention backend 抽象类默认：

```text
is_mla() = False
```

对应代码：`vllm/vllm/v1/attention/backend.py:237`

`MLACommonBackend` 重写为：

```text
is_mla() = True
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:1221` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:1227`

在 backend 校验阶段，如果请求 `use_mla=True`，但 backend 不是 MLA backend，会报：

```text
MLA not supported
```

对应代码：`vllm/vllm/v1/attention/backend.py:338` 到 `vllm/vllm/v1/attention/backend.py:342`

backend registry 中注册了多种 MLA backend：

```text
FLASHINFER_MLA
TOKENSPEED_MLA
FLASHINFER_MLA_SPARSE
TRITON_MLA
CUTLASS_MLA
FLASHMLA
FLASHMLA_SPARSE
FLASH_ATTN_MLA
ROCM_AITER_MLA
ROCM_AITER_TRITON_MLA
XPU_MLA_SPARSE
```

对应代码：`vllm/vllm/v1/attention/backends/registry.py:53` 到 `vllm/vllm/v1/attention/backends/registry.py:93`

因此 MLA 的 backend 选择链路是：

```text
MLAAttention(use_mla=True, use_sparse=...)
  → get_attn_backend()
  → current_platform.get_attn_backend_cls(...)
  → backend.validate_configuration()
  → 只允许 is_mla() 匹配的 backend
```

---

## 16. Sparse MLA / DeepSeek V3.2 的特殊路径

Sparse MLA 是在 MLA 基础上增加“只关注 top-k token”的稀疏检索机制。

### 16.1 indexer 在模型层生成 top-k

在 DeepSeek V3.2 风格配置中，如果存在 `index_topk`，`DeepseekV2MLAAttention` 会认为这是 V3.2 / sparse MLA：

```text
self.is_v32 = hasattr(config, "index_topk")
```

对应代码：`vllm/vllm/model_executor/models/deepseek_v2.py:999`

它会创建 `Indexer`，并通过 `MLAModules` 传入 `MultiHeadLatentAttentionWrapper`：

```text
indexer
indexer_rotary_emb
is_sparse=True
topk_indices_buffer
```

对应代码：`vllm/vllm/model_executor/models/deepseek_v2.py:1029` 到 `vllm/vllm/model_executor/models/deepseek_v2.py:1069`

`MultiHeadLatentAttentionWrapper.forward()` 中，如果满足：

```text
self.indexer and self.is_sparse and not self.skip_topk
```

就会调用：

```text
self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
```

对应代码：`vllm/vllm/model_executor/layers/mla.py:169` 到 `vllm/vllm/model_executor/layers/mla.py:170`

### 16.2 Sparse MLA backend

FlashMLA Sparse 在：

```text
vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py
```

它声明：

```text
get_name() = FLASHMLA_SPARSE
is_mla() = True
is_sparse() = True
supported head_size = 576
supported kv cache dtype = auto / bfloat16 / fp8_ds_mla / fp8
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:90` 到 `vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:127`

Sparse MLA 的 metadata 不再直接复用 `MLACommonMetadata`，而是有自己的 `FlashMLASparseMetadata`，包含：

```text
block_table
req_id_per_token
topk_tokens
FP8 kernel metadata
separate prefill / decode metadata
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:147`

### 16.3 fp8_ds_mla cache 格式

FlashMLA Sparse 对 FP8 cache 有特殊格式。

文件注释中写明：

DeepSeek V3.2 每 token 656 bytes：

```text
512 bytes：FP8 NoPE latent
16 bytes：4 个 float32 scale
128 bytes：BF16 RoPE part
```

DeepSeek V4 每 token 584 bytes：

```text
448 bytes：FP8 NoPE latent
128 bytes：BF16 RoPE part
8 bytes：ue8m0 scales + padding
```

对应代码：`vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:65` 到 `vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:87`

这也是为什么 `MLAAttention.__init__()` 中，如果 backend 是 `FLASHMLA_SPARSE` 且启用 FP8 KV cache，会把 cache dtype 自动改成：

```text
fp8_ds_mla
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:396` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:410`

---

## 17. MLA 和 KV cache spec 的关系

`MLAAttentionSpec` 定义在：

```text
vllm/vllm/v1/kv_cache_interface.py:366
```

它继承自 `FullAttentionSpec`，但覆盖了 MLA 相关行为。

### 17.1 page size 计算

普通 attention page size 需要 K + V。

MLA page size 默认是：

```text
storage_block_size * num_kv_heads * head_size * dtype_size
```

对应代码：`vllm/vllm/v1/kv_cache_interface.py:393` 到 `vllm/vllm/v1/kv_cache_interface.py:398`

其中：

```text
num_kv_heads = 1
head_size = kv_lora_rank + qk_rope_head_dim
```

也就是每 token 只保存一个 latent KV 向量。

### 17.2 compress_ratio

DeepSeek V4 等路径可能设置 `compress_ratio`，`MLAAttentionSpec.storage_block_size` 会变成：

```text
block_size // compress_ratio
```

对应代码：`vllm/vllm/v1/kv_cache_interface.py:379` 到 `vllm/vllm/v1/kv_cache_interface.py:381`

这说明某些 MLA cache 的物理存储 token 数可能和逻辑 block size 不完全一致。

### 17.3 fp8_ds_mla 特殊 page size

如果 `cache_dtype_str == "fp8_ds_mla"`，`MLAAttentionSpec.real_page_size_bytes` 会走特殊字节数：

```text
DeepSeek V4：storage_block_size * 584
DeepSeek V3.2：block_size * 656
```

对应代码：`vllm/vllm/v1/kv_cache_interface.py:384` 到 `vllm/vllm/v1/kv_cache_interface.py:392`

这说明 MLA cache spec 不只是“shape”，还要精确描述 backend 特定的物理 cache 格式。

---

## 18. MLA 对 KV connector / paged cache / prefix cache 的影响

### 18.1 paged cache 仍然保留

MLA 没有绕过 vLLM 的 paged KV cache 体系。它仍然使用：

```text
block_table
slot_mapping
seq_lens
query_start_loc
```

区别是 page 中每个 token 保存的内容变成：

```text
concat(kv_c_normed, k_pe)
```

### 18.2 KV connector 仍通过 attention layer hook 接入

`unified_mla_attention_with_output` 上有：

```python
@maybe_transfer_kv_layer
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:1065` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:1067`

这说明 MLA attention 仍然走 vLLM attention layer 的 KV transfer hook，但传输的数据语义是 MLA cache，而不是普通 K/V cache。

### 18.3 KV sharing 不支持

`MLACommonImpl.__init__()` 中，如果传入 `kv_sharing_target_layer_name`，会直接报错：

```text
KV sharing is not supported for MLA
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:1991` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:1993`

所以 MLA 当前不支持普通 attention 中某些 KV sharing 优化。

### 18.4 prefix caching 有 backend 限制

`MLAAttention.__init__()` 中对部分组合有保护逻辑：

```text
如果 enable_prefix_caching 且 VLLM_BATCH_INVARIANT，
并且 backend 是 TRITON_MLA / FLASHINFER，
则禁用 prefix caching。
```

对应代码：`vllm/vllm/model_executor/layers/attention/mla_attention.py:426` 到 `vllm/vllm/model_executor/layers/attention/mla_attention.py:440`

这说明 MLA 的 prefix caching 不是完全不可用，但会受到 backend 和 batch invariance 约束。

---

## 19. 和 ModelRunner attention metadata 链路的关系

从执行层看，MLA 不是自己重新发明一套 batch 准备流程，而是接入已有 ModelRunner attention metadata 链路。

整体关系是：

```text
GPUModelRunner._prepare_inputs()
  → input_ids / positions / query_start_loc / seq_lens
  → block_table / slot_mapping

GPUModelRunner._build_attention_metadata()
  → CommonAttentionMetadata
  → MLACommonMetadataBuilder.build(...)
  → MLACommonMetadata

set_forward_context(...)
  → MLAAttention.forward()
  → 从 forward_context 读取 attn_metadata / slot_mapping
```

因此 MLA 的特殊性主要发生在：

```text
1. Attention layer 如何解释 q/k/v 输入；
2. KV cache slot 中保存什么；
3. AttentionMetadataBuilder 如何拆 prefill / decode；
4. backend 如何执行 forward_mha / forward_mqa。
```

而 request 调度、block 分配、slot mapping 的上层语义仍和普通 paged attention 保持一致。

---

## 20. 整体主链路图

可以把 MLA 从模型层到 kernel 的链路画成：

```text
DeepseekV2DecoderLayer
  → 判断 model_config.use_mla
  → DeepseekV2MLAAttention
  → MultiHeadLatentAttentionWrapper
      → fused_qkv_a_proj / q_proj
      → q_a_layernorm / q_b_proj
      → kv_a_layernorm
      → rotary_emb(q_pe, k_pe)
      → optional indexer for sparse MLA
      → MLAAttention(q, kv_c_normed, k_pe)

MLAAttention.forward
  → 从 forward_context 取 attn_metadata / slot_mapping
  → concat_and_cache_mla(kv_c_normed, k_pe, slot_mapping)
  → forward_impl(...)

forward_impl
  → 根据 MLACommonMetadata 拆 decode / prefill
  → prefill tokens:
      → forward_mha
      → kv_b_proj 展开 k_nope / v
      → prefill_backend.run_prefill_new_tokens
      → optional chunked context gather + merge
  → decode tokens:
      → q_nope × W_UK 得到 latent query
      → backend.forward_mqa
      → FlashMLA / TritonMLA / FlashInferMLA kernel
      → _v_up_proj

输出
  → o_proj
  → decoder layer 后续 MLP / residual
```

---

## 21. 最容易混淆的几个点

### 21.1 MLA 的 `head_size` 不是普通 attention 的 head size

在 MLA cache 里：

```text
head_size = kv_lora_rank + qk_rope_head_dim
```

它表示 cache 中每 token 保存的 latent KV + RoPE K 总宽度。

### 21.2 `kv_b_proj` 不在 cache 写入时执行

cache 写入的是：

```text
kv_c_normed + k_pe
```

不是 `kv_b_proj(kv_c_normed)` 后的展开 K/V。

`kv_b_proj` 主要在：

```text
prefill 展开 K/V
或 decode 前后投影 W_UK / W_UV 权重准备
```

中使用。

### 21.3 prefill 和 decode 不是同一个 kernel

MLA prefill 通常走 compute-friendly MHA-style：

```text
展开 k_nope / v，再做多头 attention
```

MLA decode 走 data-movement-friendly MQA-style：

```text
Q 投到 latent KV 空间，直接读压缩 cache
```

### 21.4 FlashMLA 主要是 decode backend

`FlashMLABackend` 的 `forward_mqa()` 调用 FlashMLA kernel 处理 decode。

prefill backend 是单独通过：

```text
get_mla_prefill_backend(vllm_config)
```

选择的。

### 21.5 Sparse MLA 不只是换 kernel

Sparse MLA 还需要：

```text
indexer
topk_indices_buffer
特殊 metadata
可能的 fp8_ds_mla cache layout
```

因此它不仅是 `use_sparse=True`，还会改变模型层和 cache 格式。

---

## 22. 最终可以记成一张表

| 层级 | 关键类 / 函数 | 主要职责 |
|---|---|---|
| 模型层选择 | `DeepseekV2DecoderLayer` | 根据 `use_mla` 选择 MLA attention 类 |
| MLA 模型包装 | `MultiHeadLatentAttentionWrapper` | 从 hidden states 生成 `q / kv_c_normed / k_pe`，处理 RoPE / sparse indexer / o_proj |
| MLA attention layer | `MLAAttention` | 选择 backend，写 MLA KV cache，调 prefill / decode 实现 |
| KV cache spec | `MLAAttentionSpec` | 描述 MLA cache 的 page size、head size、特殊 FP8 layout |
| KV 写入 | `do_kv_cache_update` / `concat_and_cache_mla` | 按 slot mapping 写入 `concat(kv_c_normed, k_pe)` |
| Metadata | `MLACommonMetadata` | 保存 MLA batch 的 prefill / decode 拆分信息 |
| Metadata builder | `MLACommonMetadataBuilder` | 从 `CommonAttentionMetadata` 构造 MLA metadata、chunked context metadata |
| Prefill 公共实现 | `MLACommonImpl.forward_mha` | 展开当前 token / context K/V，调用 prefill backend |
| Decode 公共实现 | `MLAAttention.forward_impl` | 构造 latent query，调用 backend `forward_mqa`，做 V up projection |
| FlashMLA decode | `FlashMLAImpl.forward_mqa` | 调用 `flash_mla_with_kvcache` / FP8 版本 |
| TritonMLA decode | `TritonMLAImpl.forward_mqa` | 调用 `decode_attention_fwd(..., is_mla=True)` |
| Prefill backend | `MLAPrefillBackend` | 提供 `run_prefill_new_tokens` / `run_prefill_context_chunk` |
| Sparse MLA | `FlashMLASparseBackend` / `Indexer` | top-k 稀疏 attention、特殊 FP8 cache 格式 |

---

## 23. 一句话总结

MLA 在 vLLM 中可以理解为一条“压缩 KV cache + 双路径 attention”的实现链路：

```text
模型层把 hidden states 投影成 q、kv_c、k_pe；
KV cache 只保存 concat(kv_c_normed, k_pe)；
prefill 为了计算效率临时展开 K/V；
decode 为了减少内存搬运直接在 latent cache 上做 MQA-style attention；
不同硬件和配置下由 FlashMLA / TritonMLA / FlashInferMLA / Sparse MLA 等 backend 接管 kernel。
```

如果只记住一件事，就是：

```text
MLA 的关键不是“attention 公式换了名字”，而是 KV cache 表示方式变了；
vLLM 的所有 metadata、backend、slot mapping、prefill / decode 分流，都是围绕这个 cache 表示变化展开的。
```
