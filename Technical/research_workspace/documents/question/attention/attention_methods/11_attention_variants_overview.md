# 11. Attention variants 总览：各种 attention 名词如何分类？

源码位置：

- `vllm/vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/vllm/model_executor/layers/attention/cross_attention.py`
- `vllm/vllm/model_executor/layers/attention/chunked_local_attention.py`
- `vllm/vllm/model_executor/layers/attention/prefill_prefix_lm_attention.py`
- `vllm/vllm/model_executor/layers/attention/static_sink_attention.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/selector.py`
- `vllm/vllm/v1/attention/backends/registry.py`
- `vllm/vllm/v1/attention/backends/flash_attn.py`
- `vllm/vllm/v1/attention/backends/flashinfer.py`
- `vllm/vllm/v1/attention/backends/triton_attn.py`
- `vllm/vllm/v1/attention/backends/flex_attention.py`
- `vllm/vllm/v1/attention/backends/utils.py`
- `vllm/vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/vllm/v1/attention/backends/mla/prefill/registry.py`
- `vllm/vllm/v1/attention/backends/mla/prefill/selector.py`
- `vllm/vllm/v1/kv_cache_interface.py`
- `vllm/vllm/v1/worker/gpu/attn_utils.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/platforms/cuda.py`
- `vllm/vllm/platforms/rocm.py`
- `vllm/vllm/config/attention.py`

本文用于给 FlashAttention、PagedAttention、MHA / MQA / GQA、MLA、Sliding Window、FlashInfer、FlashMLA、Triton、FlexAttention、cascade attention、chunked prefill、HMA / hybrid KV cache 等名词建立分类框架，避免把模型结构、KV cache 管理、kernel backend 和调度优化混在一起。

本文中的源码路径按 vLLM 源码仓库根目录写作，例如 `vllm/...`。

---

## 0. 梳理规划

本篇是 `attention_methods` 的总览文档，目标不是把每个 backend 的每行实现都展开，而是先建立一个分类坐标系：

```text
同样都被叫作 attention，实际可能属于完全不同的层：

- 模型结构层：MHA / MQA / GQA / MLA
- mask 语义层：causal / non-causal / sliding window / local / PrefixLM
- KV cache 管理层：PagedAttention / block table / slot mapping / KV cache groups / HMA
- kernel backend 层：FlashAttention / FlashInfer / FlashMLA / Triton / FlexAttention / torch SDPA
- 调度优化层：prefill / decode / chunked prefill / cascade attention / prefix cache / spec decode
- 分布式和内存协作层：KV connector / disaggregated prefill-decode / external KV / cross-layer KV cache
```

要回答的问题分成 8 组：

```text
1. 各种 attention 名词分别属于哪一层概念？
2. FlashAttention 和 PagedAttention 是一类东西吗？
3. MHA / MQA / GQA 和 backend 有什么关系？
4. MLA 是模型结构、KV layout，还是 backend？
5. Sliding window / local attention 属于算法语义还是 backend？
6. FlashInfer / FlashMLA / Triton / FlexAttention 和 FlashAttention family 如何区分？
7. HMA / hybrid KV cache 算 attention 类型吗？它如何影响 attention？
8. vLLM 中这些概念如何落到源码路径？
```

阅读顺序建议：

```text
11_attention_variants_overview.md
  → 12_flash_attention_family.md
  → 13_paged_attention.md
  → 14_mha_mqa_gqa.md
  → 15_mla_attention.md
  → 16_sliding_window_and_local_attention.md
  → 17_flashinfer_flashmla_triton_backends.md
  → 18_hma_and_kv_cache_layout.md
```

---

## 1. 一句话回答

“attention 类型”不是单一维度。

最容易混淆的是：

```text
MHA / MQA / GQA / MLA：
  描述模型 Q/K/V head 和 KV 表示方式。

PagedAttention：
  描述 KV cache 如何分页存储、如何通过 block table 访问。

FlashAttention / FlashInfer / Triton / FlexAttention / FlashMLA：
  描述底层 attention kernel 或 backend 实现。

Sliding window / local / causal / PrefixLM：
  描述 token 可见性和 mask 语义。

chunked prefill / cascade attention / prefix cache / spec decode：
  描述调度、分块、复用或混合 batch 的优化方式。
```

所以阅读 vLLM attention 代码时，第一步不是问“这是什么 attention”，而是先判断它属于哪一层：

```text
它改变的是 Q/K/V 形状？
还是改变 KV cache 布局？
还是换了 kernel？
还是改变 mask？
还是改变一轮 batch 怎么拆、怎么复用、怎么调度？
```

---

## 2. 总体分类框架

### 2.1 一张总表

| 名词 | 所属层次 | 它回答的问题 | 在 vLLM 中主要落点 |
|---|---|---|---|
| MHA | 模型结构 | 每个 query head 是否有独立 K/V head | Attention 初始化参数、模型层投影形状 |
| MQA | 模型结构 | 多个 query head 是否共享一组 K/V | `num_heads` / `num_kv_heads` / KV cache shape |
| GQA | 模型结构 | query heads 分组共享 K/V | 同上 |
| MLA | 模型结构 + KV 表示 + backend 特化 | 是否用 latent KV 表示并拆分 MHA / MQA 路径 | `MLAAttention`、`MLAAttentionImpl`、MLA backend |
| PagedAttention | KV cache 管理 | 历史 KV 如何分页、如何按 block table 读取 | KV cache spec、block table、slot mapping、backend paged kernel |
| FlashAttention | kernel backend | 用哪种高效 attention kernel 计算 | `flash_attn.py` |
| FlashInfer | kernel backend | 用 FlashInfer wrapper 做 prefill / decode | `flashinfer.py` |
| FlashMLA | MLA kernel backend | MLA decode / sparse MLA 的高性能 kernel | backend registry 中 MLA 项 |
| Triton attention | kernel backend | 用 vLLM 自有 Triton unified kernel | `triton_attn.py`、`triton_unified_attention.py` |
| FlexAttention | kernel backend + mask 表达 | 用 PyTorch FlexAttention 和 block mask 表达可见性 | `flex_attention.py` |
| Sliding window | mask 语义 + cache spec | 每个 token 只能看固定窗口内历史 token | `SlidingWindowSpec`、backend `window_size` |
| Local attention | mask 语义 + batch 变换 | 将长序列拆成局部可见 virtual batches | `chunked_local_attention.py` |
| PrefixLM / non-causal prefill | mask 语义 | prefill 阶段部分或全部 token 双向可见 | `prefill_prefix_lm_attention.py` |
| Static sink attention | mask / cache 变体 | 固定 sink tokens 始终可见 | `static_sink_attention.py` |
| Chunked prefill | 调度优化 | 长 prompt 分多轮或分 chunk 计算 | scheduler + metadata + MLA chunked context |
| Cascade attention | 调度 / kernel 优化 | 对共同 prefix 和 suffix 分开算再 merge | FlashAttention / FlashInfer metadata |
| Prefix cache | KV 复用优化 | 已有前缀 KV 是否可复用 | Scheduler / KVCacheManager / block table |
| Spec decode metadata | 混合执行优化 | draft / target token 的 logits 和 attention 如何对齐 | ModelRunner + attention metadata |
| HMA / hybrid KV cache | KV cache 分组和布局 | 不同层或不同 attention spec 如何共享 / 分组 cache | `kv_cache_interface.py`、`attn_utils.py` |
| KV connector | 分布式 KV 协作 | KV 如何在远端加载、保存、迁移 | `distributed/kv_transfer`、ModelRunner mixin |

### 2.2 最核心的判断方法

遇到一个 attention 名词，可以按下面顺序判断：

```text
1. 它是不是改变模型数学结构？
   是 → MHA / MQA / GQA / MLA 这一层。

2. 它是不是改变 token 可见范围？
   是 → causal / non-causal / sliding window / local / PrefixLM 这一层。

3. 它是不是改变 KV cache 如何存储和索引？
   是 → PagedAttention / block table / slot mapping / HMA / KV cache group 这一层。

4. 它是不是换了底层计算库或 kernel？
   是 → FlashAttention / FlashInfer / Triton / FlexAttention / FlashMLA 这一层。

5. 它是不是改变一轮请求怎么拆、怎么复用、怎么合并？
   是 → chunked prefill / cascade / prefix cache / spec decode 这一层。
```

---

## 3. vLLM attention 的整体执行链路

### 3.1 从 ModelRunner 到 attention layer

在 V1 执行链路里，attention 不是模型层自己临时拼参数，而是 ModelRunner 先把本轮 batch 状态整理成 metadata，再放进 forward context，模型层执行时从 context 中取出来。

主链路可以压缩成：

```text
SchedulerOutput
  → GPUModelRunner._update_states()
  → GPUModelRunner._prepare_inputs()
      → input_ids / positions / query_start_loc / seq_lens
      → block_table.commit_block_table()
      → compute_slot_mapping()
  → GPUModelRunner._build_attention_metadata()
      → CommonAttentionMetadata
      → per-backend AttentionMetadata
  → set_forward_context(...)
  → model forward
  → Attention.forward()
      → 从 forward context 取 attn_metadata / kv_cache
      → 写 KV cache
      → 调 backend impl forward
```

相关源码位置：

- `vllm/vllm/v1/worker/gpu_model_runner.py:1930`：`_prepare_inputs()` 准备本轮 token 级输入。
- `vllm/vllm/v1/worker/gpu_model_runner.py:2254`：`_build_attention_metadata()` 构造 attention metadata。
- `vllm/vllm/v1/worker/gpu/attn_utils.py:454`：`build_attn_metadata()` 按 group / builder 构造 metadata。
- `vllm/vllm/model_executor/layers/attention/attention.py:485`：`Attention.forward()` 是普通 attention 层入口。
- `vllm/vllm/model_executor/layers/attention/attention.py:726`：forward 中从上下文拿 metadata / cache 的主逻辑附近。

### 3.2 Attention 层做什么

普通 `Attention` 层的职责不是实现所有 kernel，而是做一层统一封装：

```text
1. 初始化时选择 backend；
2. 根据 backend 决定 KV cache shape / layout；
3. forward 时读取当前 layer 的 metadata；
4. 把本轮 K/V 写入 KV cache；
5. 调用 backend 的 AttentionImpl；
6. 返回 attention output。
```

关键位置：

- `vllm/vllm/model_executor/layers/attention/attention.py:221`：`Attention` 类定义附近。
- `vllm/vllm/model_executor/layers/attention/attention.py:349`：根据参数选择 backend 的逻辑附近。
- `vllm/vllm/model_executor/layers/attention/attention.py:616`：生成 KV cache spec 的逻辑附近。
- `vllm/vllm/model_executor/layers/attention/attention.py:769`：统一 KV cache update 路径附近。
- `vllm/vllm/model_executor/layers/attention/attention.py:813`：统一 attention op 调用路径附近。

可以把 `Attention` 理解成：

```text
模型层里的 attention facade。

它不等于 FlashAttention，也不等于 PagedAttention；
它负责把模型结构参数、KV cache、metadata 和 backend 串起来。
```

### 3.3 AttentionBackend / AttentionImpl / MetadataBuilder 三件套

vLLM V1 中 attention backend 抽象主要分成三部分。

第一层是 `AttentionBackend`：

```text
描述一个 backend 的能力和配套类：

- backend 名称；
- impl class；
- metadata builder class；
- KV cache shape；
- 支持的 dtype / head size / block size / attention type；
- 是否支持 sliding window / sink / non-causal / encoder-decoder 等能力。
```

位置：`vllm/vllm/v1/attention/backend.py:55`

第二层是 `AttentionImpl`：

```text
真正执行普通 attention forward 的接口。
```

位置：`vllm/vllm/v1/attention/backend.py:860`

第三层是 `AttentionMetadataBuilder`：

```text
把 CommonAttentionMetadata 转成 backend 自己要的 metadata。
```

位置：`vllm/vllm/v1/attention/backend.py:600`

对于 MLA，还有单独的 `MLAAttentionImpl`：

```text
forward_mha：prefill / compute-friendly 路径。
forward_mqa：decode / memory-friendly 路径。
```

位置：`vllm/vllm/v1/attention/backend.py:943`

---

## 4. Attention type：decoder、encoder、cross attention 是另一条维度

vLLM 还会区分 attention type：

```text
DECODER
ENCODER
ENCODER_ONLY
ENCODER_DECODER
```

位置：`vllm/vllm/v1/attention/backend.py:32`

这组概念回答的是：

```text
当前 attention 是自回归 decoder self-attention？
还是 encoder attention？
还是 encoder-decoder cross attention？
```

它和 backend 不是同一个维度。

例如：

```text
FlashAttention backend 可以服务 decoder attention；
也可以在支持时服务 encoder-only 或 encoder-decoder attention。

CrossAttention 是 attention 语义 / 模型结构层的包装；
底层仍然可能复用某个 backend。
```

相关实现：

- `vllm/vllm/model_executor/layers/attention/cross_attention.py:73`：`CrossAttention` 包装底层 backend。
- `vllm/vllm/model_executor/layers/attention/cross_attention.py:123`：cross attention KV cache / slot mapping 相关逻辑附近。
- `vllm/vllm/v1/attention/backends/flash_attn.py:68`：FlashAttention backend 支持 attention type 的声明附近。

---

## 5. Backend 是如何注册和选择的

### 5.1 backend registry

vLLM 把 backend 集中注册在 `AttentionBackendEnum` 中。

位置：`vllm/vllm/v1/attention/backends/registry.py:34`

常见 backend 可以粗分为：

```text
普通 attention backend：
  FLASH_ATTN
  FLASH_ATTN_DIFFKV
  TRITON_ATTN
  TRITON_ATTN_DIFFKV
  FLASHINFER
  FLEX_ATTENTION
  CPU_ATTN
  TURBOQUANT

MLA backend：
  FLASHINFER_MLA
  TOKENSPEED_MLA
  TRITON_MLA
  CUTLASS_MLA
  FLASHMLA
  FLASH_ATTN_MLA

Sparse MLA / model-specific backend：
  FLASHINFER_MLA_SPARSE
  FLASHMLA_SPARSE
  DeepSeek V4 sparse MLA variants
  MiniMax M3 sparse backend
```

这只是常见 attention backend 的分类示例，完整枚举以 `AttentionBackendEnum` 为准；当前源码还包含 ROCm / XPU / model-specific / no-attention / custom 等后端，例如 `ROCM_AITER_MLA_SPARSE`、`ROCM_AITER_TRITON_MLA`、`XPU_MLA_SPARSE`、`NO_ATTENTION`、`CUSTOM`。

这里要注意：

```text
FlashAttention / FlashInfer / Triton 是 backend 名；
MLA 是模型结构和 KV 表示；
FlashMLA 是 MLA 场景下的特化 backend。
```

### 5.2 selector 入口

普通 attention backend 选择入口是 `get_attn_backend()`。

位置：`vllm/vllm/v1/attention/selector.py:21`

它会构造 `AttentionSelectorConfig`，再交给当前 platform 决定具体 backend。

位置：

- `vllm/vllm/v1/attention/selector.py:54`
- `vllm/vllm/v1/attention/selector.py:90`
- `vllm/vllm/v1/attention/selector.py:106`

大致可以理解成：

```text
Attention 层给出需求：
  head_size / dtype / kv_cache_dtype / block_size / is_mla / attention_type / sliding_window / sink 等

selector 构造配置：
  AttentionSelectorConfig

platform 按硬件和已安装库选择：
  CUDA / ROCm / CPU / XPU / TPU / custom platform
```

### 5.3 CUDA 上的选择逻辑

CUDA 平台会按硬件能力、是否 MLA、KV cache dtype、backend 可用性等选择。

位置：`vllm/vllm/platforms/cuda.py:82`

粗略优先级可以记成：

```text
非 Blackwell + 非 MLA：
  FlashAttention → FlashInfer → Triton → FlexAttention → TurboQuant

Blackwell + 非 MLA：
  FlashInfer → FlashAttention → Triton → FlexAttention → TurboQuant

MLA：
  根据 Blackwell、稀疏 MLA、已安装库、配置项选择
  FlashInfer MLA / TokenSpeed MLA / Cutlass MLA / FlashAttention MLA / FlashMLA / Triton MLA 等。
```

相关位置：

- `vllm/vllm/platforms/cuda.py:92`
- `vllm/vllm/platforms/cuda.py:117`
- `vllm/vllm/platforms/cuda.py:130`
- `vllm/vllm/platforms/cuda.py:147`

### 5.4 配置覆盖

`AttentionConfig` 提供显式配置入口。

位置：`vllm/vllm/config/attention.py:16`

常见字段包括：

```text
backend：
  显式指定 attention backend；auto / None 表示自动选择。

flash_attn_version：
  强制 FlashAttention 版本。

use_prefill_decode_attention：
  使用独立 prefill / decode kernel，而不是 unified Triton kernel。

mla_prefill_backend：
  指定 MLA prefill backend。

use_prefill_query_quantization：
  prefill query 量化。

use_non_causal：
  非因果 attention。

FlexAttention block 配置：
  控制 FlexAttention 的 block mask / block size 等行为。
```

位置：

- `vllm/vllm/config/attention.py:19`
- `vllm/vllm/config/attention.py:22`
- `vllm/vllm/config/attention.py:26`
- `vllm/vllm/config/attention.py:45`
- `vllm/vllm/config/attention.py:49`
- `vllm/vllm/config/attention.py:59`

---

## 6. MHA / MQA / GQA：模型结构层

### 6.1 它们回答的问题

MHA / MQA / GQA 主要回答：

```text
Q heads 和 K/V heads 是什么对应关系？
```

可以这样记：

```text
MHA：
  每个 query head 有自己对应的 K/V head。

MQA：
  多个 query head 共享同一组 K/V head。

GQA：
  query heads 分组，每组共享一组 K/V head。
```

这组概念影响：

```text
- Q/K/V projection 的形状；
- num_heads 和 num_kv_heads；
- KV cache 每层要存多少 K/V head；
- backend forward 时如何把 query head 映射到 KV head。
```

但它不等于 backend。

也就是说：

```text
同一个 GQA 模型，可以用 FlashAttention backend；
也可以用 Triton backend；
也可以在某些设备上用 FlashInfer backend。
```

### 6.2 在 vLLM 中的落点

普通 `Attention` 初始化时会接收 head 相关参数，并用这些参数选择 backend、创建 KV cache spec。

关键位置：

- `vllm/vllm/model_executor/layers/attention/attention.py:221`：`Attention` 类入口。
- `vllm/vllm/model_executor/layers/attention/attention.py:196`：head 数、head size 等初始化参数附近。
- `vllm/vllm/model_executor/layers/attention/attention.py:616`：根据 layer 配置生成 KV cache spec。

在 backend 层，普通 `AttentionImpl` 看到的是已经整理好的：

```text
query: [num_tokens, num_heads, head_size]
key:   [num_tokens, num_kv_heads, head_size]
value: [num_tokens, num_kv_heads, head_size]
```

具体形状会因 backend 和 layout 有差异，但逻辑上就是：

```text
模型结构决定 Q/K/V 形状；
backend 决定如何高效计算。
```

---

## 7. MLA：同时跨模型结构、KV 表示和 backend 特化

### 7.1 为什么 MLA 容易混淆

MLA 不像 MHA / GQA 那样只是“head 数关系”的简单变化。

在 vLLM 中，MLA 通常同时牵涉三层：

```text
1. 模型结构：
   使用 latent KV 表示，而不是直接缓存完整 K/V。

2. KV cache layout：
   KV cache 中保存的是压缩后的 latent 表示和 RoPE 相关部分。

3. backend 特化：
   decode / prefill 可能分别走不同 kernel。
```

所以 MLA 既不是单纯 backend，也不是单纯 KV cache 管理方式。

### 7.2 MLAAttention 是独立入口

普通 attention 走 `Attention`。

MLA 走 `MLAAttention`。

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:322`

MLA 初始化时会选择：

```text
- MLA decode backend；
- MLA prefill backend；
- MLA KV cache spec / metadata builder；
- 是否启用 chunked prefill context。
```

相关位置：

- `vllm/vllm/model_executor/layers/attention/mla_attention.py:387`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py:446`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py:478`

### 7.3 MLA 的 prefill / decode 分流

`MLAAttentionImpl` 明确拆出两个 forward：

```text
forward_mha：
  prefill 路径，更偏 compute-friendly。

forward_mqa：
  decode 路径，更偏 memory-friendly。
```

位置：

- `vllm/vllm/v1/attention/backend.py:943`
- `vllm/vllm/v1/attention/backend.py:924`
- `vllm/vllm/v1/attention/backend.py:939`

`MLACommonMetadata` 中会记录：

```text
num_decodes
num_decode_tokens
num_prefills
prefill metadata
decode metadata
```

位置：

- `vllm/vllm/model_executor/layers/attention/mla_attention.py:1275`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py:1298`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py:1307`

forward 时大致是：

```text
prefill tokens → forward_mha()
decode tokens  → forward_mqa()
```

位置：

- `vllm/vllm/model_executor/layers/attention/mla_attention.py:681`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py:688`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py:713`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py:724`

### 7.4 MLA prefill backend

MLA prefill backend 还有单独 registry。

位置：`vllm/vllm/v1/attention/backends/mla/prefill/registry.py:34`

常见项包括：

```text
FLASH_ATTN
FLASHINFER
TRTLLM_RAGGED
TOKENSPEED_MLA
```

选择器位置：`vllm/vllm/v1/attention/backends/mla/prefill/selector.py:48`

这说明：

```text
MLA backend 不是一个单点。

它可能同时包含：
- MLA decode backend；
- MLA prefill backend；
- chunked context 处理；
- 稀疏 MLA 特化 backend。
```

---

## 8. PagedAttention：KV cache 管理和访问方式

### 8.1 PagedAttention 不是 FlashAttention 的同类概念

FlashAttention 回答的是：

```text
attention score / softmax / value 聚合如何高效计算？
```

PagedAttention 回答的是：

```text
历史 KV cache 如何分页保存？
一个请求的 token 如何映射到一组非连续 KV blocks？
attention kernel 如何根据 block table 找到历史 K/V？
```

所以二者不是同一层。

在 vLLM 里常见组合是：

```text
FlashAttention backend + paged KV cache
Triton backend + paged KV cache
FlashInfer backend + paged KV cache
```

### 8.2 block table 和 slot mapping

Paged KV cache 里最关键的是两个映射：

```text
block table：
  request → block ids

slot mapping：
  当前 token → KV cache 中的具体写入 slot
```

ModelRunner 先根据 Scheduler 分配的 block ids 更新 `InputBatch.block_table`，再在 `_prepare_inputs()` 中计算 slot mapping，最后放入 attention metadata。

相关位置：

- `vllm/vllm/v1/worker/gpu_model_runner.py:2118`：`compute_slot_mapping()` 调用附近。
- `vllm/vllm/v1/worker/gpu_model_runner.py:2330`：`CommonAttentionMetadata` 填充 block table / slot mapping 附近。
- `vllm/vllm/v1/attention/backend.py:395`：`CommonAttentionMetadata` 定义附近。

可以这样记：

```text
block table 决定“读哪些历史 blocks”；
slot mapping 决定“当前 token 写到哪个 KV slot”。
```

### 8.3 KV cache spec 类型

vLLM 不只一种 KV cache spec。

位置：`vllm/vllm/v1/kv_cache_interface.py:86`

常见类型包括：

```text
FullAttentionSpec
MLAAttentionSpec
SlidingWindowSpec
SlidingWindowMLASpec
ChunkedLocalAttentionSpec
SinkFullAttentionSpec
EncoderOnlyAttentionSpec
CrossAttentionSpec
```

相关位置：

- `vllm/vllm/v1/kv_cache_interface.py:204`：MLA spec 附近。
- `vllm/vllm/v1/kv_cache_interface.py:366`：sliding window 相关 spec 附近。
- `vllm/vllm/v1/kv_cache_interface.py:477`：`SlidingWindowSpec` 附近。
- `vllm/vllm/v1/kv_cache_interface.py:549`：sink attention spec 附近。
- `vllm/vllm/v1/kv_cache_interface.py:676`：encoder / cross attention spec 附近。

这说明：

```text
attention variant 最终经常会落成不同的 KV cache spec。

例如：
- 普通 decoder attention → FullAttentionSpec
- sliding window attention → SlidingWindowSpec
- MLA → MLAAttentionSpec
- sink attention → SinkFullAttentionSpec
- cross attention → CrossAttentionSpec
```

### 8.4 HMA / hybrid KV cache 在哪一层

HMA / hybrid KV cache 不应该理解成一种新的 attention 数学形式。

它更接近：

```text
KV cache group / layout / allocation 组织方式。
```

它影响的是：

```text
- 哪些 layer 共用一个 KV cache group；
- 不同 attention spec 如何分组；
- raw KV tensor 如何 reshape 成 backend 要的布局；
- metadata builder 如何按 group 构造；
- 是否存在不同 layer / group 的 hybrid attention layout。
```

相关位置：

- `vllm/vllm/v1/worker/gpu/attn_utils.py:74`：初始化 attention backend / metadata builders。
- `vllm/vllm/v1/worker/gpu/attn_utils.py:166`：KV cache tensor reshape / layout 处理附近。
- `vllm/vllm/v1/worker/gpu/attn_utils.py:250`：packed backing / page padding / stride 相关逻辑附近。
- `vllm/vllm/v1/worker/gpu/attn_utils.py:416`：hybrid attention layout 相关逻辑附近。

---

## 9. FlashAttention / FlashInfer / Triton / FlexAttention：backend 层

### 9.1 backend 层回答的问题

backend 层回答：

```text
给定 Q/K/V、KV cache、block table、slot mapping、mask 语义，
到底调用哪套 kernel 算 attention？
```

它通常不决定：

```text
- 模型是不是 MQA / GQA；
- Scheduler 是否做 prefix cache；
- 请求是否被 chunked prefill；
- KV blocks 怎样分配。
```

但 backend 必须支持这些上层语义传下来的参数。

### 9.2 FlashAttention backend

FlashAttention backend 的实现位置：

- `vllm/vllm/v1/attention/backends/flash_attn.py`

它的特点：

```text
- 使用 FlashAttention varlen / paged 相关 kernel；
- 支持 decoder、encoder、encoder-only、encoder-decoder 等 attention type；
- 支持 non-causal；
- 支持 sliding window 时转换成 window_size；
- 可在部分场景启用 cascade attention；
- 支持 sink / PrefixLM / multimodal PrefixLM mask 等能力取决于 backend 版本。
```

关键位置：

- `vllm/vllm/v1/attention/backends/flash_attn.py:68`：backend 能力声明附近。
- `vllm/vllm/v1/attention/backends/flash_attn.py:651`：sliding window 转换逻辑附近。
- `vllm/vllm/v1/attention/backends/flash_attn.py:701`：forward 主路径附近。
- `vllm/vllm/v1/attention/backends/flash_attn.py:1490`：cascade attention 启发式判断附近。
- `vllm/vllm/v1/attention/backends/flash_attn.py:1568`：cascade attention 执行路径附近。

### 9.3 FlashInfer backend

FlashInfer backend 的实现位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py`

它的特点：

```text
- 显式维护 prefill wrapper 和 decode wrapper；
- metadata 中明确记录 num_decodes / num_prefills；
- 可以走 FlashInfer native path；
- 也可能走 TRTLLM prefill / decode path；
- 支持 FP8、NVFP4 等 KV cache dtype / quant mode；
- 某些设备和 dtype 下要求特定 KV layout。
```

关键位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:519`：`FlashInferMetadata` 字段附近。
- `vllm/vllm/v1/attention/backends/flashinfer.py:558`：prefill / decode wrapper 初始化附近。
- `vllm/vllm/v1/attention/backends/flashinfer.py:937`：metadata build 中拆分 prefill / decode 附近。
- `vllm/vllm/v1/attention/backends/flashinfer.py:1436`：forward 主路径附近。
- `vllm/vllm/v1/attention/backends/flashinfer.py:1587`：prefill / decode 分流附近。

可以记成：

```text
FlashInfer 更强调 prefill / decode wrapper 的显式管理；
Triton unified attention 更强调用统一 metadata 表达 mixed batch。
```

### 9.4 Triton backend

Triton backend 的实现位置：

- `vllm/vllm/v1/attention/backends/triton_attn.py`
- `vllm/vllm/v1/attention/ops/triton_unified_attention.py`

它的特点：

```text
- 使用 vLLM 自有 Triton unified attention kernel；
- metadata 不一定显式保存 prefill / decode 数量；
- 通过 query_start_loc、seq_lens、block_table、slot_mapping 统一表达 mixed batch；
- 支持 sliding window、non-causal、sink、mm prefix、KV quant mode 等；
- KV cache update 和 attention forward 可以分离。
```

关键位置：

- `vllm/vllm/v1/attention/backends/triton_attn.py:58`：metadata 定义附近。
- `vllm/vllm/v1/attention/backends/triton_attn.py:248`：backend 能力声明附近。
- `vllm/vllm/v1/attention/backends/triton_attn.py:530`：forward 主路径附近。
- `vllm/vllm/v1/attention/backends/triton_attn.py:642`：调用 `unified_attention()` 附近。
- `vllm/vllm/v1/attention/ops/triton_unified_attention.py:179`：Triton unified kernel 入口附近。

### 9.5 FlexAttention backend

FlexAttention backend 的实现位置：

- `vllm/vllm/v1/attention/backends/flex_attention.py`

它的特点：

```text
- 基于 PyTorch FlexAttention；
- 通过 block mask / mask mod 表达可见性；
- 支持 decoder 和 encoder-only；
- 支持 non-causal、batch invariance、mm prefix；
- 可表达 causal、bidirectional、paged KV、sliding window、block sparsity hint 等。
```

关键位置：

- `vllm/vllm/v1/attention/backends/flex_attention.py:85`：backend 能力声明附近。
- `vllm/vllm/v1/attention/backends/flex_attention.py:362`：block mask / mask mod 构造附近。
- `vllm/vllm/v1/attention/backends/flex_attention.py:454`：forward 路径附近。

---

## 10. Sliding window / local attention：mask 语义层

### 10.1 sliding window 改变的是可见范围

Sliding window attention 的核心语义是：

```text
每个 query token 只能看左侧固定窗口内的历史 token，
而不是看完整历史上下文。
```

它首先是 mask 语义，不是某个特定 backend。

同样的 sliding window 语义，可以落到：

```text
- KV cache spec：SlidingWindowSpec；
- FlashAttention 参数：window_size；
- Triton unified kernel 参数：window_left / sliding_window；
- FlashInfer metadata：window_left 一致性约束；
- FlexAttention block mask。
```

### 10.2 在 Attention 层的落点

普通 `Attention` 初始化支持 sliding window。

位置：

- `vllm/vllm/model_executor/layers/attention/attention.py:215`
- `vllm/vllm/model_executor/layers/attention/attention.py:228`
- `vllm/vllm/model_executor/layers/attention/attention.py:308`

生成 KV cache spec 时，如果是 sliding window，会返回 `SlidingWindowSpec`。

位置：

- `vllm/vllm/model_executor/layers/attention/attention.py:594`
- `vllm/vllm/model_executor/layers/attention/attention.py:598`
- `vllm/vllm/v1/kv_cache_interface.py:477`

这意味着：

```text
sliding window 不只影响 kernel mask，
也影响 KV cache 最大可保留 token 数和 block 分配策略。
```

### 10.3 在 backend 的落点

FlashAttention：

- `vllm/vllm/v1/attention/backends/flash_attn.py:651`
- `vllm/vllm/v1/attention/backends/flash_attn.py:825`

Triton：

- `vllm/vllm/v1/attention/backends/triton_attn.py:460`
- `vllm/vllm/v1/attention/backends/triton_attn.py:655`

FlashInfer：

- `vllm/vllm/v1/attention/backends/utils.py:118`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1018`

### 10.4 chunked local attention

Local attention 可以通过“把请求拆成局部 virtual batches”来表达。

实现位置：

- `vllm/vllm/model_executor/layers/attention/chunked_local_attention.py:30`
- `vllm/vllm/v1/attention/backends/utils.py:199`

它的关键思路是：

```text
不是发明一个完全新的 attention backend，
而是包装已有 backend 的 metadata builder，
把原 batch 改造成多个局部可见的 virtual batches。
```

---

## 11. Prefill / decode / mixed batch：执行阶段层

### 11.1 prefill 和 decode 不是模型结构

prefill / decode 描述的是请求处于哪个执行阶段：

```text
prefill：
  prompt 或 extend 阶段，一次可能计算多个 query tokens。

decode：
  自回归生成阶段，通常每个请求每轮计算 1 个新 token。

mixed batch：
  同一轮 batch 里既有 decode，也有 prefill / extend。
```

它们不是 attention 数学结构，而是执行阶段。

### 11.2 通用拆分工具

vLLM 有工具函数将 batch 拆成 decode / prefill 段。

位置：

- `vllm/vllm/v1/attention/backends/utils.py:538`：`split_decodes_and_prefills()`。
- `vllm/vllm/v1/attention/backends/utils.py:637`：`reorder_batch_to_split_decodes_and_prefills()`。

这个函数假设 batch 已按类似顺序组织：

```text
decode
  → short extend
  → long extend
  → prefill
```

再根据 query length 判断边界。

### 11.3 不同 backend 的处理方式不同

FlashInfer 显式区分：

```text
metadata 里记录 num_decodes / num_prefills，
forward 时分别调用 prefill wrapper 和 decode wrapper。
```

位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:519`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1587`

Triton unified attention 则更像：

```text
用 query_start_loc / seq_lens / max_query_len / block table / slot mapping
统一表达 mixed batch，kernel 内部处理不同形态。
```

位置：

- `vllm/vllm/v1/attention/backends/triton_attn.py:58`
- `vllm/vllm/v1/attention/ops/triton_unified_attention.py:179`

MLA 则显式走：

```text
prefill → forward_mha
decode  → forward_mqa
```

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:681`

---

## 12. Chunked prefill：调度优化，不是新 attention 类型

### 12.1 chunked prefill 的问题背景

长 prompt 一次性 prefill 可能导致：

```text
- token 数过多；
- 激活和临时 workspace 过大；
- decode 请求等待时间过长；
- MLA 场景下 up-project 后内存压力很大。
```

chunked prefill 的思路是：

```text
把一个长 prompt / long context 拆成多个 chunk，
分多轮或分多段计算，
再在 attention state 层合并。
```

### 12.2 普通调度层面的 chunked prefill

从执行层看，chunked prefill 最终体现为：

```text
- num_computed_tokens 不是 0；
- 本轮只 schedule 一部分 prompt tokens；
- positions 从已计算 token 后继续；
- block table 中已有 blocks 被继续使用；
- attention metadata 标记当前请求处于 prefill / extend。
```

这部分和 `ModelRunner._prepare_inputs()`、`_build_attention_metadata()` 强相关。

位置：

- `vllm/vllm/v1/worker/gpu_model_runner.py:1930`
- `vllm/vllm/v1/worker/gpu_model_runner.py:2254`

### 12.3 MLA chunked context

MLA 文件顶部对 chunked prefill 有专门设计说明。

位置：`vllm/vllm/model_executor/layers/attention/mla_attention.py:14`

MLA 场景下，chunked context 的核心是：

```text
1. 对 new tokens 先算一部分 attention；
2. 对已有 context 按 chunk gather / dequant / up-project；
3. 每个 chunk 调 prefill backend；
4. 用 merge_attn_states 合并 attention state。
```

关键位置：

- `vllm/vllm/model_executor/layers/attention/mla_attention.py:1235`：chunked context metadata 附近。
- `vllm/vllm/model_executor/layers/attention/mla_attention.py:2060`：`_compute_prefill_context()` 附近。
- `vllm/vllm/model_executor/layers/attention/mla_attention.py:2275`：`forward_mha()` 中 chunked context 合并路径附近。

所以：

```text
chunked prefill 不是一种 attention 公式，
而是一种把长上下文分段计算并保持结果等价的执行优化。
```

---

## 13. Cascade attention：共同 prefix 复用的 kernel / metadata 优化

### 13.1 它解决什么问题

当一个 batch 中多个请求共享较长 prefix 时，可以把 attention 分成：

```text
shared prefix 部分：
  多个请求共同使用。

suffix 部分：
  每个请求自己的后缀。
```

cascade attention 的思路是：

```text
先算 shared prefix 的非因果 / prefix attention state，
再算 suffix 的 causal attention state，
最后 merge_attn_states。
```

它属于：

```text
调度信息 + backend kernel 支持 + attention state 合并
```

不是模型结构层的 attention 类型。

### 13.2 FlashAttention 中的 cascade

FlashAttention metadata 中有 cascade 相关字段：

```text
use_cascade
common_prefix_len
prefix / suffix KV lens
```

位置：

- `vllm/vllm/v1/attention/backends/flash_attn.py:254`
- `vllm/vllm/v1/attention/backends/flash_attn.py:489`
- `vllm/vllm/v1/attention/backends/flash_attn.py:528`

是否启用 cascade 有启发式判断，例如：

```text
- common prefix 太短不启用；
- ALiBi、sliding window、local attention 不支持；
- 请求数太少不启用；
- DCP 场景不启用；
- 根据 FlashDecoding 与 cascade 的粗略性能模型决定。
```

位置：

- `vllm/vllm/v1/attention/backends/flash_attn.py:1490`
- `vllm/vllm/v1/attention/backends/flash_attn.py:1189`
- `vllm/vllm/v1/attention/backends/flash_attn.py:1195`
- `vllm/vllm/v1/attention/backends/flash_attn.py:1203`

执行路径：

- `vllm/vllm/v1/attention/backends/flash_attn.py:1568`
- `vllm/vllm/v1/attention/backends/flash_attn.py:1289`
- `vllm/vllm/v1/attention/backends/flash_attn.py:1317`

### 13.3 FlashInfer 中的 cascade

FlashInfer metadata 中保留了 cascade wrapper / metadata 路径，但当前 `FlashInferMetadataBuilder.use_cascade_attention()` 明确返回 `False`，因此常规调度不会自动选择 FlashInfer cascade；不要把它理解成和 FlashAttention cascade 同等启用的路径。

位置：

- `vllm/vllm/v1/attention/backends/flashinfer.py:547`
- `vllm/vllm/v1/attention/backends/flashinfer.py:873`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1112`
- `vllm/vllm/v1/attention/backends/flashinfer.py:1325`

---

## 14. Prefix cache / spec decode：调度状态如何影响 attention

### 14.1 prefix cache

prefix cache 不是 backend，也不是 attention 公式。

它回答的是：

```text
这个请求的前缀是否已经有可复用 KV blocks？
```

从 Worker / attention 角度看，prefix cache 的结果表现为：

```text
- num_computed_tokens 增大；
- block table 中已经有可复用 blocks；
- 本轮 positions 从已计算位置后继续；
- 当前 token 只需要写入新 slot；
- attention backend 通过 block table 读取已有历史 KV。
```

相关位置：

- `vllm/vllm/v1/worker/gpu_model_runner.py:1920`：positions 根据 `num_computed_tokens` 计算附近。
- `vllm/vllm/v1/worker/gpu_model_runner.py:2330`：metadata 携带 block table / slot mapping 附近。

所以：

```text
prefix cache 命中发生在调度和 KV cache manager 层，
attention backend 只消费最后形成的 block table / seq_lens / positions。
```

### 14.2 spec decode metadata

Spec decode 会让一个请求在一轮里出现 draft token、target token、bonus token 等位置关系。

它影响 attention 的方式通常不是“换一个 attention backend”，而是：

```text
- 本轮 token 数和 logits_indices 更复杂；
- metadata 要能描述哪些 token 是 draft / accepted / bonus；
- slot mapping 和 seq_lens 需要与 speculative 状态对齐；
- sample 阶段还要根据 target logits 做接受 / 拒绝。
```

相关位置：

- `vllm/vllm/v1/worker/gpu_model_runner.py:1930`：`_prepare_inputs()` 构造 spec decode metadata 的主入口附近。
- `vllm/vllm/v1/worker/gpu_model_runner.py:2254`：attention metadata 构造会接入 spec decode 相关公共 metadata。

---

## 15. PrefixLM / non-causal / static sink：mask 语义变体

### 15.1 PrefixLM / prefill prefix LM attention

PrefixLM 的核心是：

```text
decoder attention 仍然保留 KV cache，
但 prefill 阶段某些范围内可以 non-causal / bidirectional 可见。
```

实现位置：

- `vllm/vllm/model_executor/layers/attention/prefill_prefix_lm_attention.py:17`
- `vllm/vllm/model_executor/layers/attention/prefill_prefix_lm_attention.py:73`

它会影响：

```text
- KV cache spec 中的 non_causal 标记；
- chunked prefill / prefix caching 等是否兼容；
- backend 的 causal 参数或 mask 表达。
```

### 15.2 Static sink attention

Static sink attention 的核心是：

```text
给每层 attention 增加固定 sink tokens，
这些 tokens 在后续 attention 中始终可见。
```

实现位置：

- `vllm/vllm/model_executor/layers/attention/static_sink_attention.py:40`
- `vllm/vllm/model_executor/layers/attention/static_sink_attention.py:80`
- `vllm/vllm/model_executor/layers/attention/static_sink_attention.py:196`

它会影响：

```text
- seq_lens；
- block_table；
- sink K/V 写入 KV cache 的时机；
- backend 是否支持 sink。
```

---

## 16. KV cache quantization 和 layout：不是 attention 类型，但会限制 backend

### 16.1 KVQuantMode

KV cache dtype / quant mode 会显著影响 backend 选择。

位置：`vllm/vllm/v1/kv_cache_interface.py:33`

常见模式包括：

```text
none
FP8 per tensor
INT8 per token/head
FP8 per token/head
NVFP4
```

相关位置：

- `vllm/vllm/v1/kv_cache_interface.py:40`
- `vllm/vllm/v1/kv_cache_interface.py:60`
- `vllm/vllm/v1/kv_cache_interface.py:77`

这类信息不是 attention 语义，但会影响：

```text
- 某个 backend 是否可用；
- KV cache 的实际 layout；
- 是否需要 scale tensors；
- kernel 是否支持 per-token-head scaling；
- Blackwell / Hopper 上是否使用特定布局。
```

### 16.2 KV cache layout

不同 backend 可能要求不同 KV cache layout，例如 NHD / HND、packed backing、page padding、stride order 等。

相关位置：

- `vllm/vllm/v1/worker/gpu/attn_utils.py:166`
- `vllm/vllm/v1/worker/gpu/attn_utils.py:194`
- `vllm/vllm/v1/worker/gpu/attn_utils.py:303`
- `vllm/vllm/v1/worker/gpu/attn_utils.py:312`

可以记成：

```text
layout 是 backend 与 KV cache tensor 之间的契约。

它不改变 attention 数学语义，
但会决定同一份 KV cache 在物理内存中如何排列。
```

---

## 17. Cross attention / encoder-only attention：模型类型维度

### 17.1 CrossAttention

Cross attention 用于 encoder-decoder 模型。

实现位置：`vllm/vllm/model_executor/layers/attention/cross_attention.py:73`

它的特点：

```text
- query 来自 decoder；
- key / value 来自 encoder outputs 或 encoder cache；
- 通常 causal=False；
- 需要 encoder 侧的 slot mapping / KV cache 处理；
- 底层仍然可以复用某个 attention backend。
```

关键位置：

- `vllm/vllm/model_executor/layers/attention/cross_attention.py:81`
- `vllm/vllm/model_executor/layers/attention/cross_attention.py:123`
- `vllm/vllm/model_executor/layers/attention/cross_attention.py:189`

### 17.2 Encoder-only attention

Encoder-only attention 常见于 embedding / rerank / encoder 模型。

它和 decoder attention 的区别是：

```text
- 不一定需要自回归 causal mask；
- 不一定需要像生成模型一样长期维护 decoder KV cache；
- metadata / KV cache spec 可能走 EncoderOnlyAttentionSpec；
- backend 必须声明自己支持对应 attention type。
```

相关位置：

- `vllm/vllm/v1/attention/backend.py:40`
- `vllm/vllm/v1/kv_cache_interface.py:669`

---

## 18. 分布式 KV / KV connector：不是 attention 类型，但会穿过 attention 链路

KV connector、disaggregated prefill-decode、external KV 等概念属于分布式 KV 协作层。

它们回答：

```text
KV cache 是否来自本地？
是否需要远端加载？
prefill 和 decode 是否在不同 worker / 实例上完成？
KV 保存和接收什么时候完成？
```

从 attention 角度看，它最终仍然要落成：

```text
- 当前 worker 上可访问的 KV cache tensor；
- block table；
- slot mapping；
- seq_lens；
- metadata 中的 KV 布局描述。
```

所以它不是新的 attention 公式，也不是某个 kernel backend。

它影响的是：

```text
forward 前后 KV 是否已经 ready；
0-token step 是否仍需推进 KV transfer；
Scheduler 和 Worker 如何同步 finished_sending / finished_recving；
哪些 blocks 被认为可用或失效。
```

---

## 19. 几个容易混淆的问题

### 19.1 FlashAttention 和 PagedAttention 是一类东西吗？

不是。

```text
FlashAttention：
  kernel backend / attention 计算算法实现。

PagedAttention：
  KV cache 分页存储和访问方式。
```

它们可以组合使用。

### 19.2 MHA / MQA / GQA 和 FlashAttention 是什么关系？

```text
MHA / MQA / GQA 决定 Q/K/V head 结构；
FlashAttention 决定如何高效计算 attention。
```

一个 GQA 模型可以使用 FlashAttention，也可以使用 Triton 或 FlashInfer。

### 19.3 MLA 是 backend 吗？

不完全是。

```text
MLA 首先是一种模型结构和 KV 表示；
在 vLLM 中它又有专门的 MLAAttention、MLA KV cache spec、MLA metadata、MLA backend。
```

所以不要把 MLA 和 FlashMLA 混为一谈：

```text
MLA：模型结构 / KV 表示。
FlashMLA：MLA 场景下的一个 backend / kernel 实现。
```

### 19.4 Sliding window 是 backend 吗？

不是。

```text
Sliding window 是 mask / 可见性语义；
backend 只是负责支持这种语义。
```

### 19.5 Chunked prefill 是 attention 类型吗？

不是。

```text
Chunked prefill 是长 prompt / long context 的执行和调度优化。
它改变一轮算多少 token、metadata 如何描述上下文，
不改变 attention 的基本数学定义。
```

### 19.6 Cascade attention 是 prefix cache 吗？

不是同一个概念。

```text
prefix cache：
  跨请求 / 跨轮复用已经算好的 KV blocks。

cascade attention：
  在同一轮 attention 计算中利用共同 prefix，把 prefix 和 suffix 分开算再合并 state。
```

二者都和 prefix 有关，但层次不同。

### 19.7 HMA / hybrid KV cache 是 attention 类型吗？

不是。

它更接近 KV cache group / layout / allocation 组织方式，会影响 attention metadata 和 cache tensor layout，但不是新的 attention 公式。

---

## 20. 源码路径速查表

| 想看什么 | 入口文件 | 重点 |
|---|---|---|
| 普通 attention 层 | `model_executor/layers/attention/attention.py` | `Attention` 初始化、KV cache spec、forward、backend dispatch |
| MLA attention 层 | `model_executor/layers/attention/mla_attention.py` | `MLAAttention`、prefill/decode 分流、chunked context |
| backend 抽象 | `v1/attention/backend.py` | `AttentionBackend`、`AttentionImpl`、`MLAAttentionImpl`、`CommonAttentionMetadata` |
| backend registry | `v1/attention/backends/registry.py` | 所有 attention backend 枚举 |
| backend selector | `v1/attention/selector.py` | 构造 selector config，交给 platform 选择 |
| CUDA backend 优先级 | `platforms/cuda.py` | FlashAttention / FlashInfer / Triton / MLA backend 自动选择 |
| ROCm backend 优先级 | `platforms/rocm.py` | AITER / Triton / ROCm attention 选择 |
| FlashAttention backend | `v1/attention/backends/flash_attn.py` | FA backend、sliding window、cascade attention |
| FlashInfer backend | `v1/attention/backends/flashinfer.py` | prefill / decode wrapper、TRTLLM path、FlashInfer metadata |
| Triton backend | `v1/attention/backends/triton_attn.py` | vLLM Triton unified attention backend |
| Triton kernel | `v1/attention/ops/triton_unified_attention.py` | unified attention kernel 参数和 mask / quant 支持 |
| FlexAttention backend | `v1/attention/backends/flex_attention.py` | PyTorch FlexAttention、block mask / mask mod |
| MLA prefill backend | `v1/attention/backends/mla/prefill/registry.py` | MLA prefill backend 注册 |
| MLA prefill selector | `v1/attention/backends/mla/prefill/selector.py` | MLA prefill backend 选择 |
| KV cache spec | `v1/kv_cache_interface.py` | Full / MLA / SlidingWindow / Sink / Cross attention specs |
| attention metadata build | `v1/worker/gpu/attn_utils.py` | backend 初始化、KV cache reshape、metadata builder |
| ModelRunner attention metadata | `v1/worker/gpu_model_runner.py` | `_prepare_inputs()`、`_build_attention_metadata()` |
| Cross attention | `model_executor/layers/attention/cross_attention.py` | encoder-decoder cross attention 包装 |
| PrefixLM attention | `model_executor/layers/attention/prefill_prefix_lm_attention.py` | prefill non-causal / PrefixLM 语义 |
| Static sink attention | `model_executor/layers/attention/static_sink_attention.py` | sink tokens 注入和 metadata 调整 |
| Chunked local attention | `model_executor/layers/attention/chunked_local_attention.py` | local attention virtual batches |
| backend 工具 | `v1/attention/backends/utils.py` | prefill/decode 拆分、local virtual batches、window 处理 |
| attention 配置 | `config/attention.py` | backend override、FA version、MLA prefill backend、non-causal 等 |

---

## 21. 后续专题如何展开

```text
12_flash_attention_family.md：
  梳理 FlashAttention v1-v4、FlashAttention backend、FlashAttention MLA、cascade attention 的关系。

13_paged_attention.md：
  梳理 vLLM paged KV cache、block table、slot mapping、KV cache spec、prefix cache 与 attention backend 的关系。

14_mha_mqa_gqa.md：
  梳理 MHA / MQA / GQA 如何影响 Q/K/V shape、num_kv_heads、KV cache size 和 backend 参数。

15_mla_attention.md：
  梳理 DeepSeek 类 MLA 的 latent KV 表示、MLAAttention、forward_mha / forward_mqa、FlashMLA / Triton MLA / FlashInfer MLA。

16_sliding_window_and_local_attention.md：
  梳理 sliding window、local attention、PrefixLM、non-causal、static sink 等 mask 语义。

17_flashinfer_flashmla_triton_backends.md：
  对比 FlashInfer、FlashMLA、Triton、FlexAttention、FlashAttention backend 的适用场景和 metadata 差异。

18_hma_and_kv_cache_layout.md：
  梳理 HMA / hybrid KV cache groups、KV cache layout、packed backing、page padding、stride order 和 attention metadata 的关系。
```

---

## 22. 一句话总结

vLLM 里的 attention 相关名词要按层次理解：

```text
MHA / MQA / GQA / MLA
  → 模型结构和 KV 表示。

PagedAttention / block table / slot mapping / HMA
  → KV cache 管理和访问方式。

FlashAttention / FlashInfer / FlashMLA / Triton / FlexAttention
  → kernel backend。

causal / non-causal / sliding window / local / PrefixLM / sink
  → mask 和可见性语义。

prefill / decode / chunked prefill / cascade / prefix cache / spec decode
  → 执行阶段、调度复用和 metadata 优化。
```

如果只记住一句话，就是：

```text
attention variants 不是一棵单层分类树，而是一组会叠加的维度；
一次真实的 vLLM attention forward，通常同时包含模型结构、KV cache spec、mask 语义、backend kernel 和调度 metadata 五层信息。
```
