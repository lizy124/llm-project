# 03. AttentionMetadataBuilder 如何构造 metadata？

源码位置：

- `code/vllm/vllm/v1/attention/backend.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/utils.py`
- `code/vllm/vllm/v1/worker/ubatch_utils.py`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `code/vllm/vllm/model_executor/layers/attention/cross_attention.py`
- `code/vllm/vllm/model_executor/layers/attention/encoder_only_attention.py`
- `code/vllm/vllm/model_executor/layers/attention/chunked_local_attention.py`
- `code/vllm/vllm/model_executor/layers/attention/static_sink_attention.py`

本文用于梳理 `AttentionMetadataBuilder` 如何把 `ModelRunner` 的 batch 状态、KV cache block table、slot mapping、prefill / decode 形态、CUDA graph / ubatching / cascade / spec decode 等执行条件，翻译成具体 attention backend 可以直接消费的 metadata。

---

## 1. 本文要回答的问题

```text
AttentionMetadataBuilder 是什么？
它由谁创建、何时创建？
CommonAttentionMetadata 和 backend-specific metadata 是什么关系？
GPUModelRunner._build_attention_metadata() 给 builder 传了哪些输入？
metadata 为什么要按 KV cache group / attention group 构造？
prefill / decode / mixed batch / spec decode 如何影响 metadata？
不同 backend 的 metadata builder 有什么差异？
ubatching、CUDA graph、cascade attention、DCP 会如何改变构造路径？
metadata 最终如何进入 attention forward？
```

---

## 2. 一句话回答

`AttentionMetadataBuilder` 是 `GPUModelRunner` 和具体 `AttentionBackend` 之间的适配层。

它接收 `CommonAttentionMetadata` 这份跨 backend 的公共 batch 描述，再把它翻译成 FlashAttention、FlashInfer、Triton、FlexAttention、MLA、CPU、Mamba、GDN 等 backend 自己的 `AttentionMetadata`。

最小链路是：

```text
GPUModelRunner._prepare_inputs()
  → query_start_loc / seq_lens / positions / num_scheduled_tokens

GPUModelRunner._get_slot_mappings()
  → slot_mapping

GPUModelRunner._build_attention_metadata()
  → CommonAttentionMetadata
  → AttentionMetadataBuilder.build(...)
  → backend-specific AttentionMetadata

set_forward_context(...)
  → Attention.forward()
  → get_attention_context(layer_name)
  → backend impl.forward(...)
```

如果只记住一句话：

```text
CommonAttentionMetadata 说明“这一批 token 和 KV cache 长什么样”，AttentionMetadataBuilder 负责把这份说明改写成“某个 kernel 具体怎么跑”的参数结构。
```

---

## 3. 核心对象关系

### 3.1 AttentionBackend

`AttentionBackend` 是 backend 家族的抽象类。

位置：`code/vllm/vllm/v1/attention/backend.py:56`

它最关键的职责包括：

```text
get_impl_cls()
  返回真正执行 attention forward 的 impl class。

get_builder_cls()
  返回本 backend 对应的 AttentionMetadataBuilder class。

get_kv_cache_shape()
  返回本 backend 期望的 KV cache tensor 形状。

validate_configuration(...)
  声明并校验当前 backend 是否支持某组模型 / dtype / KV dtype / attention type / capability。
```

所以选择一个 backend，不只是选择一个 kernel，而是选择一整套协议：

```text
backend class
  → impl class
  → metadata builder class
  → KV cache shape / layout
  → CUDA graph support
  → forward 中 KV cache update 与 attention kernel 的组合方式
```

### 3.2 AttentionMetadata

`AttentionMetadata` 是 backend-specific metadata 的基类。

位置：`code/vllm/vllm/v1/attention/backend.py:387`

它本身只定义统一类型边界，真正字段在具体 backend 中扩展，例如：

```text
FlashAttentionMetadata
FlashInferMetadata
TritonAttentionMetadata
FlexAttentionMetadata
CPUAttentionMetadata
MLACommonMetadata
GDNAttentionMetadata
BaseMambaAttentionMetadata
```

这些对象最终会被 attention impl 的 `forward()` 消费。

### 3.3 CommonAttentionMetadata

`CommonAttentionMetadata` 是 builder 的统一输入。

位置：`code/vllm/vllm/v1/attention/backend.py:395`

它保存的是所有 backend 都可能需要的 batch 级信息，例如：

```text
query_start_loc
seq_lens
num_reqs
num_actual_tokens
max_query_len
max_seq_len
block_table_tensor
slot_mapping
causal
positions
encoder_seq_lens
dcp_local_seq_lens
is_prefilling
mm_req_doc_ranges
rswa_prefix_lens
```

它不是最终 kernel 参数，而是公共中间表示。

可以这样理解：

```text
InputBatch / SchedulerOutput / block table / slot mapping
  → GPUModelRunner 归一化
  → CommonAttentionMetadata
  → backend builder 翻译
  → backend-specific AttentionMetadata
```

### 3.4 AttentionMetadataBuilder

`AttentionMetadataBuilder` 是本文主角。

位置：`code/vllm/vllm/v1/attention/backend.py:600`

关键接口包括：

```text
build(common_prefix_len, common_attn_metadata, fast_build=False)
  正常构造 backend metadata。

build_for_cudagraph_capture(common_attn_metadata)
  CUDA graph capture 时构造形状更固定、分支更保守的 metadata。

build_for_drafting(common_attn_metadata, draft_index)
  speculative decoding / drafting 场景使用。

update_block_table(metadata, blk_table, slot_mapping)
  如果 metadata 结构可以复用，只更新 block table / slot mapping。

use_cascade_attention(...)
  判断当前 batch 是否值得启用 cascade attention。
```

此外 builder 还会声明：

```text
_cudagraph_support
reorder_batch_threshold
supports_update_block_table
```

这些字段会反过来影响 `GPUModelRunner` 的 batch 执行形态。

### 3.5 AttentionGroup

`AttentionGroup` 定义在 worker utils 中。

位置：`code/vllm/vllm/v1/worker/utils.py:243`

它把同一 KV cache group 内满足以下条件的 layer 聚在一起：

```text
backend 相同；
KVCacheSpec 相同；
num_heads_q 相同。
```

这样同一组 layer 可以共享同一份 metadata builder 和同一份 metadata。

原因是这些 layer 对 attention kernel 来说形状协议一致，没必要每层重复构造 metadata。`initialize_attn_backend()` 实际用 backend 的 `full_cls_name()`、per-layer `KVCacheSpec` 和 `num_heads_q` 组成去重 key，避免动态包装 backend 类对象不稳定导致误分组。

---

## 4. AttentionMetadataBuilder 何时创建

builder 不是每轮 forward 才创建，而是在 KV cache 初始化阶段创建。

主链路如下：

```text
Worker.initialize_from_config(...)
  → GPUModelRunner.initialize_kv_cache(kv_cache_config)
  → initialize_attn_backend(kv_cache_config)
  → initialize_metadata_builders(kv_cache_config, kernel_block_sizes)
  → AttentionGroup.create_metadata_builders(...)
  → backend.get_builder_cls()(kv_cache_spec, layer_names, vllm_config, device)
```

### 4.1 initialize_attn_backend() 先按 backend 分组

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6854`

`initialize_attn_backend()` 会遍历 KV cache group 中的 attention layer，对每个 layer 调用：

```text
layer.get_attn_backend()
```

然后按下面这个 key 聚合：

```text
AttentionGroupKey(
  attn_backend,
  kv_cache_spec,
  num_heads_q,
)
```

聚合结果保存在：

```text
self.attn_groups[kv_cache_group_id][attn_group_id]
```

这说明 `attn_groups` 是二维结构：

```text
KV cache group
  → attention group
      → layer_names
      → backend
      → metadata_builders
```

### 4.2 initialize_metadata_builders() 创建 builder

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6961`

每个 `AttentionGroup` 会调用：

```text
AttentionGroup.create_metadata_builders(...)
```

位置：`code/vllm/vllm/v1/worker/utils.py:255`

内部逻辑是：

```text
builder_cls = backend.get_builder_cls()
builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
```

如果启用 ubatching，一个 attention group 可能会持有多个 builder，因为不同 ubatch 需要独立 builder 实例来避免复用临时 buffer 时互相覆盖。

### 4.3 builder 影响 batch reorder 阈值

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7052`

`GPUModelRunner.calculate_reorder_batch_threshold()` 会读取所有 builder 的：

```text
reorder_batch_threshold
```

然后取最保守值，用于决定 batch 是否要按 decode / prefill 形态重排。

这说明 builder 不只是“被动接收 batch”，它也能声明自己喜欢什么 batch 形态。

---

## 5. 每轮 forward 时 metadata 如何构造

每轮 `execute_model()` 中，attention metadata 的核心入口是：

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2254`

```text
GPUModelRunner._build_attention_metadata(...)
```

它发生在：

```text
_may_reorder_batch()
  → _prepare_inputs()
  → _compute_cascade_attn_prefix_lens()（可选）
  → _determine_batch_execution_and_padding()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(...)
  → _model_forward()
```

其中 `_may_reorder_batch()` 会在 `_prepare_inputs()` 前根据 `reorder_batch_threshold` 重排 `InputBatch`，保证后续 `query_start_loc` 和 block table 行顺序已经与 backend 期望的 decode / prefill 分区一致。

### 5.1 _build_attention_metadata() 的输入来自哪里

它的输入大致来自四类上游：

```text
1. _prepare_inputs()
   query_start_loc、seq_lens、positions、num_scheduled_tokens、logits_indices、spec decode 信息。

2. _compute_cascade_attn_prefix_lens() / _determine_batch_execution_and_padding()
   cascade_attn_prefix_lens、num_tokens_padded、num_reqs_padded、cudagraph mode、ubatch slices。

3. _get_slot_mappings()
   每个 KV cache group 的 slot_mapping，并为 `forward_context.slot_mapping` 生成按 layer name 索引的视图。

4. SchedulerOutput / InputBatch
   block_table、num_common_prefix_blocks、encoder inputs、request 状态。
```

所以 `_build_attention_metadata()` 并不是从零计算 batch，它是在前面几个阶段已经准备好的状态上做“attention 视角的翻译”。

### 5.2 先构造 CommonAttentionMetadata base

runner 会先构造一份基础的 `CommonAttentionMetadata`，可以称为 `cm_base`。

核心字段包括：

```text
query_start_loc
query_start_loc_cpu
seq_lens
seq_lens_cpu_upper_bound
max_seq_len
num_reqs
num_actual_tokens
max_query_len
block_table_tensor
slot_mapping
causal
positions
is_prefilling
encoder_seq_lens
dcp_local_seq_lens
mm_req_doc_ranges
rswa_prefix_lens
```

当启用 KV sharing fast prefill 且存在 logits indices 时，`cm_base` 还会额外填充：

```text
logits_indices_padded
num_logits_indices
```

这份 `cm_base` 是公共模板，后续会按 KV cache group 替换其中的 group-specific 字段。

### 5.3 再按 KV cache group 替换 block table / slot mapping

不同 KV cache group 可能有不同的 block table 和 slot mapping。

所以 runner 会按 group 循环：

```text
for kv_cache_group_id in kv_cache_groups:
    cm = shallow copy of cm_base
    cm.block_table_tensor = 当前 group 的 block table
    cm.slot_mapping = 当前 group 的 slot_mapping
    cm.encoder_seq_lens = 当前 group 的 encoder seq lens（如有）
```

这里使用浅拷贝的原因是：

```text
大部分公共字段相同；
只有 block table、slot mapping、encoder seq lens 等 group-specific 字段不同。
```

### 5.4 再按 attention group 调 builder

每个 KV cache group 内可能有多个 attention group。

runner 会继续循环：

```text
for attn_group in self.attn_groups[kv_cache_group_id]:
    builder = attn_group.metadata_builders[...]
    attn_metadata_i = builder.build(common_prefix_len, cm, ...)
    for layer_name in attn_group.layer_names:
        attn_metadata[layer_name] = attn_metadata_i
```

结果是：

```text
同一个 attention group 内的多层共享同一份 attn_metadata_i；
forward context 中仍然按 layer_name 建索引，方便 Attention.forward() 取用。
```

### 5.5 CUDA graph capture 走专用 build

如果当前是 CUDA graph capture 阶段，runner 不走普通 `build()`，而是调用：

```text
builder.build_for_cudagraph_capture(cm)
```

原因是 capture 阶段需要更固定的形状和更少的动态行为，例如：

```text
padding 到 capture size；
避免某些 CPU/GPU sync；
避免依赖运行时变化的 request 分布；
预先构造 block mask 或 scheduler metadata。
```

不同 backend 对 CUDA graph 的支持程度由 `AttentionCGSupport` 描述。

### 5.6 支持 update_block_table 的 builder 可以复用 metadata

有些 backend 的 metadata 结构只和 batch 形状有关，block table / slot mapping 可以单独替换。

这类 builder 会声明：

```text
supports_update_block_table = True
```

runner 在 hybrid KV cache group 场景下可以复用已有 metadata，只调用：

```text
builder.update_block_table(metadata, blk_table, slot_mapping)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2435` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:2508`

这样可以减少重复构造开销。

FlashAttention、部分 Mamba-like builder 都有类似能力；但如果 backend metadata 中包含强绑定 block table 的 plan / mask / wrapper，就通常不能简单复用。

### 5.7 ubatching 会切分 metadata

如果启用 ubatching，runner 会先有一份全量 common metadata，再通过 ubatch 工具切成多个子 metadata。

相关位置：

- `code/vllm/vllm/v1/worker/ubatch_utils.py:134`
- `code/vllm/vllm/v1/worker/ubatch_utils.py:251`

核心动作包括：

```text
切 query_start_loc；
切 seq_lens；
切 block_table_tensor；
切 slot_mapping；
修正每个 ubatch 内部的起始 offset；
为每个 ubatch 使用对应 builder 构造 metadata。
```

ubatching 的关键点是：

```text
metadata 不是简单按 token 切片；
它必须保持 request 边界、query_start_loc 前缀和、block table 行号、slot_mapping token offset 一致。
```

---

## 6. CommonAttentionMetadata 字段详解

`CommonAttentionMetadata` 可以按语义分成几组。

### 6.1 query 布局字段

```text
query_start_loc
query_start_loc_cpu
max_query_len
num_actual_tokens
```

`query_start_loc` 是 packed query tensor 的 request 边界。

例如本轮三个请求分别调度：

```text
[2, 5, 3]
```

则：

```text
query_start_loc = [0, 2, 7, 10]
```

它表示：

```text
request 0 的 query token 在 [0, 2)
request 1 的 query token 在 [2, 7)
request 2 的 query token 在 [7, 10)
```

`max_query_len` 决定本轮是否更像 decode：

```text
max_query_len = 1
  通常是 decode-only batch。

max_query_len > 1
  可能是 prefill、chunked prefill、spec decode 或 mixed batch。
```

`num_actual_tokens` 名称有历史包袱：它在 `CommonAttentionMetadata` 中由 runner 传入 `num_tokens_padded`，普通路径下等于真实 token 数；FULL CUDA graph / padding 路径下则可能等于 padded token 数。需要未 padding 的视图时，应结合调用处的 `num_tokens` / `num_reqs`，或使用 `CommonAttentionMetadata.unpadded(...)`。

### 6.2 sequence 布局字段

```text
seq_lens
seq_lens_cpu_upper_bound
max_seq_len
num_reqs
```

`seq_lens` 表示每个 request 当前可见的总序列长度，通常等于：

```text
已计算 tokens + 本轮 query tokens
```

但在 spec decode、chunked prefill、DCP、cross attention、encoder-only attention 等场景下，语义会被 wrapper 或 builder 进一步调整。

`max_seq_len` 常用于 kernel 选择、workspace 分配、mask 构造。

`seq_lens_cpu_upper_bound` 的作用是给 CPU 侧一个安全上界，避免某些路径为了拿真实最大长度触发 GPU 到 CPU 的同步。

### 6.3 KV cache 布局字段

```text
block_table_tensor
slot_mapping
```

`block_table_tensor` 是 request 到 KV cache blocks 的映射。

它回答：

```text
某个 request 当前用了哪些 KV blocks？
```

`slot_mapping` 是当前 query token 到实际 KV cache slot 的映射。

它回答：

```text
本轮第 i 个 token 的 K/V 应该写入哪个 slot？
```

二者关系可以理解为：

```text
block_table：请求级、block 粒度、二维行表；
slot_mapping：token 级、slot 粒度、扁平映射。
```

backend builder 一般都需要这两个字段，但不同 backend 对字段形态要求不同。

例如：

```text
FlashAttention 直接把 block table / slot mapping 交给 paged KV kernel；
FlashInfer 会进一步构造 paged_kv_indptr / paged_kv_indices / last_page_len；
FlexAttention 会利用 block table 建 physical-to-logical 映射和 block mask；
CrossAttention wrapper 会用 encoder block table 重算 slot mapping。
```

### 6.4 causal 和 mask 相关字段

```text
causal
mm_req_doc_ranges
positions
```

`causal` 可以是 bool，也可能是动态 tensor。

普通 decoder attention：

```text
causal = True
```

encoder-only / cross attention / non-causal：

```text
causal = False
```

多模态 prefix LM 或 PrefixLM 场景可能需要 `mm_req_doc_ranges` 来描述哪些文档区间可以做 bidirectional / full attention。

`positions` 对普通 attention 可能只是辅助字段，但对 sparse、indexer、encoder-only、mask builder、prefix range 等路径会变得重要。

### 6.5 encoder / cross attention 字段

```text
encoder_seq_lens
encoder_seq_lens_cpu
```

encoder-decoder / cross attention 中，decoder query 要 attend 到 encoder KV。

这时 wrapper builder 会把：

```text
seq_lens 替换为 encoder_seq_lens；
causal 改为 False；
slot_mapping 改成 encoder KV cache 的 slot mapping。
```

也就是说 cross attention 复用同一套 builder 协议，但在进入底层 builder 前改写 common metadata。

### 6.6 DCP 字段

```text
dcp_local_seq_lens
dcp_local_seq_lens_cpu
```

DCP 是 decode context parallelism / context parallel 相关路径。

它表示当前 rank 本地可见的 context 长度。

不同 backend 会用它构造不同结构：

```text
FlashAttention：当前在 builder 内按 context_kv_lens 重新计算 dcp_context_kv_lens 和 max_dcp_context_kv_len；
FlashInfer：调整 seq lens 并选择 DCP prefill wrapper / plan；
MLA：decode 侧直接使用 dcp_local_seq_lens，chunked prefill 侧构造 local chunk context metadata；
Triton / Flex：按各自支持程度消费或忽略。
```

注意：runner 会把 `dcp_local_seq_lens` 放进 common metadata，但并非每个 backend 都直接读取该字段；例如 FlashAttention builder 会基于 `seq_lens - query_lens` 自行计算本地 context KV 长度。

### 6.7 KV sharing / logits index 字段

```text
logits_indices_padded
num_logits_indices
```

这组字段主要服务 KV sharing fast prefill。

它们不是普通 attention 必需字段，而是 fast prefill wrapper 会先用它们把 common metadata 改写成只覆盖 logits token 的视图，再把 `logits_indices_padded` / `num_logits_indices` 附加到返回的 backend metadata 上，用于 fast prefill 的特殊执行路径。

### 6.8 request 状态字段

```text
is_prefilling
```

`is_prefilling` 表示 request 是否仍处于 prefill 阶段。

它对普通 FlashAttention 可能不是核心字段，但对 Mamba、GDN、spec decode、batch reorder、prefill/decode 分类等路径很重要。

---

## 7. _build_attention_metadata() 主流程展开

可以把 `_build_attention_metadata()` 理解成下面这条线：

```text
输入：
  num_tokens / num_tokens_padded
  num_reqs / num_reqs_padded
  max_query_len
  slot_mappings
  logits_indices
  use_spec_decode
  num_scheduled_tokens
  cascade_attn_prefix_lens
  ubatch_slices
  for_cudagraph_capture

处理：
  1. 计算公共 batch 字段
  2. 构造 CommonAttentionMetadata base
  3. 补充 DCP / encoder / mm_prefix / KV sharing 字段
  4. 按 KV cache group 替换 block table、slot mapping 和 encoder_seq_lens
  5. 按 attention group 调用 builder
  6. 可选在 CUDA graph capture 中调用 build_for_cudagraph_capture
  7. 可选复用 cached metadata 并 update_block_table
  8. 可选按 ubatch 切分 metadata

输出：
  attn_metadata: dict[layer_name, AttentionMetadata]
  spec_decode_common_attn_metadata: CommonAttentionMetadata | None
```

### 7.1 为什么输出是 layer_name 到 metadata 的 dict

模型 forward 时，attention layer 是按 `layer_name` 从 forward context 里取 metadata 的。

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:726`

```text
get_attention_context(layer_name)
```

因此 `_build_attention_metadata()` 最终要构造：

```text
attn_metadata[layer_name] = metadata_for_this_layer
```

`attn_metadata` 在普通执行路径下是 `dict[layer_name, metadata]`。只有在 speculative decoding 等需要同时保留多份 attention context 的路径中，forward context 才可能持有 `list[dict[layer_name, metadata]]`；而 ubatching 本身主要通过 `ubatch_slices` 和切分后的 common metadata 驱动逐个 ubatch 执行，不把这个 list 形态当作它的默认外部语义。

但为了减少构造开销，同一 attention group 的多个 layer 可以指向同一个 metadata 对象。

### 7.2 为什么按 KV cache group 构造

KV cache group 可能不同：

```text
block size 不同；
KV cache spec 不同；
layer 集合不同；
encoder-only / cross / decoder / hybrid model 的 cache 组织不同；
某些 layer 可能共享 KV cache，某些不共享。
```

所以 block table、slot mapping 和 encoder_seq_lens 必须先按 group 取出。

### 7.3 为什么还要按 attention group 构造

同一个 KV cache group 中也可能存在多个 backend 或多种 head 数。

例如 hybrid 模型中可能同时有：

```text
普通 decoder attention；
MLA attention；
Mamba / SSM-like layer；
cross attention；
encoder-only attention。
```

这些 layer 的 metadata 结构不同，所以还要按 attention group 再分发到不同 builder。

### 7.4 spec decode 为什么会返回 common metadata

`_build_attention_metadata()` 除了返回 `attn_metadata`，还可能返回：

```text
spec_decode_common_attn_metadata
```

这是因为 speculative drafter / proposer 有时需要复用或改写目标模型本轮的 common attention layout，必要时还会按 KV cache group 提供 block table / slot mapping，或在 padding 后通过 `unpadded(...)` 恢复真实 request/token 视图。

也就是说 spec decode 不只影响 sampler 的 logits index，也会影响 attention metadata 中 query length、accepted token、draft token 边界等信息。

---

## 8. Prefill / Decode / Mixed batch 对 builder 的影响

prefill 和 decode 的差异不是一个单独字段，而是由多个 metadata 字段共同表达。

### 8.1 Decode-only batch

典型 decode-only batch：

```text
每个 request 本轮 query_len = 1；
max_query_len = 1；
query_start_loc = [0, 1, 2, 3, ...]；
seq_lens 表示每个请求已有上下文 + 当前 token；
slot_mapping 表示每个新 token 写入哪个 KV slot。
```

backend 可能据此走 decode kernel。

例如 FlashInfer、MLA、GDN、Mamba 等 builder 会显式把 decode tokens 拆出来；是否要求 decode query 长度统一，还会受 `reorder_batch_threshold` 和 backend 的 spec-as-decode 能力影响。

### 8.2 Prefill batch

典型 prefill batch：

```text
某个 request 本轮 query_len > 1；
max_query_len > 1；
query_start_loc 中 span 可能很长；
seq_lens 包含本轮 prompt 末尾的总长度；
slot_mapping 覆盖多个 prompt token 的 KV 写入位置。
```

backend 可能走 varlen prefill kernel、chunked prefill kernel、ragged prefill wrapper 或 MHA-style path。

### 8.3 Chunked prefill

chunked prefill 的特点是：

```text
request 还没完成完整 prompt；
本轮只计算 prompt 的一段；
seq_lens 需要表达“已有 context + 当前 chunk”；
block table 已经包含当前 chunk 会写入的 KV blocks；
query_start_loc 只描述本轮 chunk 的 query tokens。
```

MLA、FlashInfer、FlashAttention 等 backend 会根据自身 kernel 支持做不同处理。

### 8.4 Mixed batch

mixed batch 指同一轮既有 decode，又有 prefill / extend。

builder 常见处理方式有三类：

```text
1. 不显式拆分
   FlashAttention 风格，依赖 varlen / paged KV 参数统一表达。

2. 显式拆成 prefill 和 decode
   FlashInfer、MLA 风格，metadata 中有 prefill / decode 子对象。

3. 先重排 batch 再拆分
   通过 reorder_batch_threshold 和 split_decodes_and_prefills() 让 decode / prefill 区域更适合 kernel。
```

### 8.5 batch reorder

相关工具：

- `code/vllm/vllm/v1/attention/backends/utils.py:564`
- `code/vllm/vllm/v1/attention/backends/utils.py:663`

`reorder_batch_to_split_decodes_and_prefills(...)` 会把 batch 组织成：

```text
decode
short_extend
long_extend
pure_prefill
```

`split_decodes_and_prefills(...)` 则在 builder 内部常用，用来把已经重排后的 common metadata 按 query lens、decode threshold、uniform 要求拆成不同区域；必要时还会结合 `is_prefilling` 把 short extend 归到 prefill。

这个机制的核心目的：

```text
让 backend 更容易选择高效 kernel，避免 mixed batch 让所有请求都走低效通用路径。
```

---

## 9. Backend-by-backend：builder 做了什么

### 9.1 FlashAttentionMetadataBuilder

文件：`code/vllm/vllm/v1/attention/backends/flash_attn.py`

关键对象：

```text
FlashAttentionMetadata
FlashAttentionMetadataBuilder
FlashAttentionImpl
```

FlashAttention builder 的主要工作是把 common metadata 转成 FlashAttention varlen / paged KV kernel 所需参数。

典型字段包括：

```text
query_start_loc
seq_lens
block_table
slot_mapping
max_query_len
max_seq_len
causal
```

它的特点：

```text
1. metadata 结构接近 FlashAttention kernel 参数；
2. 支持 supports_update_block_table，可以复用 metadata 后更新 block table / slot mapping；
3. 支持 cascade attention；
4. 支持 DCP context KV lens；
5. 支持 mm_prefix range tensor；
6. FA3 路径可能构造 AOT scheduler_metadata；
7. CUDA graph 支持程度取决于 FA 版本和平台；
8. full CUDA graph 下会把 scheduler_metadata 复制到预分配 buffer，并通过 max_num_splits 约束中间 buffer 上界。
```

#### Cascade attention

FlashAttention 是当前最典型支持 cascade attention 的 backend；其 `use_cascade_attention()` 会要求 common prefix 至少约 256 token、请求数不少于 8，且不启用 ALiBi、sliding window、local attention 或 DCP。

runner 先根据 common prefix blocks 计算：

```text
common_prefix_len
```

然后调用：

```text
builder.use_cascade_attention(...)
```

如果启用，FlashAttention metadata 会把 attention 拆成：

```text
shared prefix attention
suffix / per-request attention
```

相关字段可能包括：

```text
use_cascade
common_prefix_len
cu_prefix_query_lens
prefix_kv_lens
suffix_kv_lens
```

这样多个请求共享的 prefix KV 可以用更高效的方式处理。

#### DCP

DCP 下 FlashAttention builder 会构造：

```text
dcp_context_kv_lens
max_dcp_context_kv_len
```

用于描述当前 rank 本地 context KV 的可见范围。

### 9.2 TritonAttentionMetadataBuilder

文件：`code/vllm/vllm/v1/attention/backends/triton_attn.py`

关键对象：

```text
TritonAttentionMetadata
TritonAttentionMetadataBuilder
TritonAttentionBackend
```

Triton builder 除了通用字段，还要管理 Triton kernel 的 scratch / segment buffer。

典型字段包括：

```text
query_start_loc
seq_lens
block_table
slot_mapping
softmax_segm_output
softmax_segm_max
softmax_segm_expsum
seq_threshold_3D
num_par_softmax_segments
```

它的特点：

```text
1. 比 FlashAttention 更显式地管理 kernel 临时 buffer；
2. 可支持 decoder / encoder / encoder_only / encoder_decoder 等 attention type；
3. 支持 non-causal、mm_prefix、sink、ALiBi sqrt、batch invariance 等特殊能力；
4. backend 的 forward_includes_kv_cache_update=False，KV cache update 可以和 attention forward 分开；
5. CUDA graph capture 时会先普通 build，再把 `seq_lens` 填成 1，避免 full graph capture 因 `max_model_len` 过大而过慢；
6. backend 类声明 `use_cascade_attention()` 返回 False，metadata dataclass 虽有 cascade 字段，常规路径不会主动启用 Triton cascade。
```

Triton metadata 不像 FlashInfer 那样显式持有 `prefill` / `decode` 子 metadata，更多是统一 paged attention 参数加 Triton kernel workspace。这类 builder 的重点不是只打包 varlen 参数，还要帮自有 Triton kernel 准备执行计划和 workspace。

### 9.3 FlashInferMetadataBuilder

文件：`code/vllm/vllm/v1/attention/backends/flashinfer.py`

关键对象：

```text
FlashInferMetadata
FlashInferMetadataBuilder
FIPrefill / FIDecode
TRTLLMPrefill / TRTLLMDecode
```

FlashInfer builder 的核心特点是显式拆分 prefill 和 decode。

它会计算：

```text
num_decodes
num_decode_tokens
num_prefills
num_prefill_tokens
```

输出 metadata 中常见结构：

```text
prefill: FIPrefill | TRTLLMPrefill | None
decode: FIDecode | TRTLLMDecode | None
cascade_wrapper
```

FlashInfer native path 会构造 paged KV 三元组：

```text
paged_kv_indptr
paged_kv_indices
paged_kv_last_page_len
```

这些字段把 vLLM 的 block table 转成 FlashInfer wrapper 更熟悉的 CSR-like paged KV 表示。

FlashInfer builder 的特点：

```text
1. 不只是打包参数，还会做 wrapper planning；
2. 同一个 batch 内 prefill 和 decode 可能分别选择 FlashInfer native 或 TRT-LLM kernel；
3. TRTLLM path 会尽量使用 GPU tensor，减少 CPU sync；
4. 默认 `reorder_batch_threshold` 为 1；如果 TRTLLM decode 可用，会通过 `_init_reorder_batch_threshold(..., supports_spec_as_decode=True)` 把阈值提升到 spec decode 可作为 uniform decode 处理的 query 长度；
5. 代码中保留 cascade wrapper / metadata 路径：若 `common_prefix_len > 0` 会构造 `cascade_wrapper`，把 shared prefix 和 suffix paged KV metadata 交给 `MultiLevelCascadeAttentionWrapper.plan(...)`；但当前 `use_cascade_attention()` 明确返回 `False`，常规调度不会自动启用 FlashInfer cascade。
```

### 9.4 FlexAttentionMetadataBuilder

文件：`code/vllm/vllm/v1/attention/backends/flex_attention.py`

关键对象：

```text
FlexAttentionMetadata
FlexAttentionMetadataBuilder
FlexAttentionMetadata.build_block_mask()
```

FlexAttention builder 的重点是 mask，而不是传统 varlen 参数。

它需要构造：

```text
BlockMask
mask_mod
physical_to_logical
decode_offset
num_blocks_per_seq
persistent_kv_indices
persistent_doc_ids
```

它解决的问题是：

```text
paged KV cache 是物理 block 布局；
attention mask 通常按逻辑 token 坐标定义；
FlexAttention 需要把物理 block 映射回逻辑 token 区间，再应用 causal / sliding window / prefix LM 等 mask。
```

支持的 mask 语义包括：

```text
causal mask
bidirectional / non-causal mask
sliding window / local attention mask
prefix LM / multimodal prefix full attention mask
```

CUDA graph 下，FlexAttention builder 会尽量预构建 block mask，避免 capture 中出现不 graph-safe 的动态 mask 构造；`build_for_cudagraph_capture()` 还会用 `seq_lens_cpu_upper_bound` 重新收紧 `max_seq_len`，避免用 `max_model_len` 触发不必要的 torch.compile 重编译。

### 9.5 CPUAttentionMetadataBuilder

文件：`code/vllm/vllm/v1/attention/backends/cpu_attn.py`

关键对象：

```text
CPUAttentionMetadata
CPUAttentionMetadataBuilder
```

CPU builder 的重点是 host 侧 scheduler metadata。

它会调用 CPU attention op 辅助函数构造调度信息，例如：

```text
ops.cpu_attn_get_scheduler_metadata(...)
```

特点：

```text
1. 支持 dynamic causal；
2. 强制使用特定 KV cache layout，例如 HND；
3. encoder-only path 会重建 encoder block table；
4. encoder-only path 会重新计算 slot mapping；
5. 可能分配 encoder_cache_tensor。
```

CPUAttention builder 比 GPU backend 更偏向“调度描述”，因为 CPU kernel 的并行方式和内存访问方式不同。

### 9.6 MLACommonMetadataBuilder

文件：`code/vllm/vllm/model_executor/layers/attention/mla_attention.py`

关键对象：

```text
MLACommonPrefillMetadata
MLACommonDecodeMetadata
MLACommonMetadata
MLACommonMetadataBuilder
```

MLA builder 是独立体系，因为 MLA 的 KV cache 表示、prefill 路径和 decode 路径都不同于普通 MHA / GQA。

它会显式拆成：

```text
prefill: MLACommonPrefillMetadata | None
decode: MLACommonDecodeMetadata | None
```

prefill 侧可能包含 chunked context metadata：

```text
cu_seq_lens
starts
seq_tot
max_seq_lens
workspace
token_to_seq
```

decode 侧则更接近 MQA-style 读取 compressed KV cache，并在 DCP 下会用 `dcp_local_seq_lens` 替换本地 `seq_lens`，同时保留 `dcp_tot_seq_lens` 供后续合并。

MLA builder 的特点：

```text
1. prefill 和 decode 是两条明显不同路径；
2. chunked prefill 需要额外 workspace；
3. DCP 下要构造 local chunk seq lens 和 all-rank context lens；
4. 会调用 prefill backend 的 prepare_metadata(prefill_metadata)；
5. query_len_support 会影响 spec decode 下的 reorder_batch_threshold；
6. full CUDA graph 通常只支持 decode-only。
```

MLA 派生 backend 包括：

```text
code/vllm/vllm/v1/attention/backends/mla/flashmla.py
code/vllm/vllm/v1/attention/backends/mla/flashattn_mla.py
code/vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py
code/vllm/vllm/v1/attention/backends/mla/triton_mla.py
code/vllm/vllm/v1/attention/backends/mla/cutlass_mla.py
```

稀疏 MLA 还有：

```text
flashmla_sparse.py
flashinfer_mla_sparse.py
sparse_swa.py
indexer.py
```

### 9.7 Mamba / GDN metadata builder

这些不是传统 attention，但复用了 attention metadata builder 管线。

原因是它们同样需要：

```text
batch request 边界；
state / cache slot 映射；
prefill / decode / spec decode 分类；
CUDA graph / ubatch 协议；
forward context 注入。
```

#### Mamba

文件：`code/vllm/vllm/v1/attention/backends/mamba_attn.py`

关键对象：

```text
BaseMambaAttentionMetadata
BaseMambaAttentionMetadataBuilder
```

Mamba metadata 关注：

```text
prefill / decode 数量；
state indices；
initial state；
prefix caching block index；
causal conv1d metadata；
spec decode accepted tokens。
```

Mamba builder 默认支持 `supports_update_block_table`，但启用 speculative decoding 后会关闭该复用能力，因为 spec decode metadata 中的 state index 与 accepted-token 状态更强绑定。

#### GDN

文件：`code/vllm/vllm/v1/attention/backends/gdn_attn.py`

关键对象：

```text
GDNAttentionMetadata
GDNAttentionMetadataBuilder
```

GDN builder 会显式区分：

```text
normal decode
spec decode
prefill
```

spec decode 存在时，它可能把 non-spec decode 重新分类为 prefill，以统一 kernel 路径。

典型字段包括：

```text
spec_query_start_loc
non_spec_query_start_loc
spec_state_indices_tensor
non_spec_state_indices_tensor
spec_token_indx
non_spec_token_indx
FLA chunk metadata
causal conv1d metadata
```

测试入口：

```text
code/vllm/tests/v1/attention/test_gdn_metadata_builder.py
```

---

## 10. Wrapper 型 backend builder

有些 backend 不是完全独立实现，而是动态包装已有 backend，并改写 builder 行为。

### 10.1 Cross attention

文件：`code/vllm/vllm/model_executor/layers/attention/cross_attention.py`

核心行为：

```text
causal = False；
使用 encoder_seq_lens 替换 seq_lens；
根据 encoder KV block table 重新计算 cross-attention slot_mapping；
调用底层 builder.build(...)；
覆盖返回 metadata 中的 slot_mapping。
```

它说明 cross attention 的 builder 不是从头写一套 kernel metadata，而是在底层 backend metadata 前修改 common metadata 语义。

### 10.2 Encoder-only attention

文件：`code/vllm/vllm/model_executor/layers/attention/encoder_only_attention.py`

核心行为：

```text
causal = False；
复用底层 builder。
```

encoder-only 的复杂逻辑更多在 block table / slot mapping / CPU path 中，而 wrapper 本身较薄。

### 10.3 Chunked local attention

文件：`code/vllm/vllm/model_executor/layers/attention/chunked_local_attention.py`

核心行为：

```text
调用 make_local_attention_virtual_batches(...) 改写 common metadata；
为 metadata 附加 make_virtual_batches_block_table；
禁用 CUDA Graph。
```

它通过“虚拟 batch”表达 local attention，而不是要求底层 backend 原生理解所有 chunk 逻辑。

### 10.4 Static sink attention

文件：`code/vllm/vllm/model_executor/layers/attention/static_sink_attention.py`

核心行为：

```text
给 seq_lens 加上 sink length；
构造带 sink blocks 的 block_table_with_sink；
调用底层 builder。
```

这类 wrapper 把 StreamingLLM / attention sink 语义转成底层 paged KV 能理解的 block table 变化。

### 10.5 KV sharing fast prefill

文件：`code/vllm/vllm/v1/attention/backends/utils.py`

核心行为：

```text
包装底层 builder；
如果不是纯 decode，则先按 logits_indices 生成只覆盖 logits token 的 common metadata；
调用底层 builder；
再附加 logits_indices_padded 和 num_logits_indices。
```

它说明 builder 还可以作为功能扩展点，把非 kernel 原生字段挂到 metadata 上。

---

## 11. CUDA Graph 对 metadata builder 的影响

CUDA graph 要求执行图中 tensor shape 和控制流尽量固定。

因此 builder 需要声明：

```text
AttentionCGSupport
```

位置：`code/vllm/vllm/v1/attention/backend.py:583`

常见支持级别可以理解为：

```text
NEVER
  不支持 CUDA graph。

UNIFORM_SINGLE_TOKEN_DECODE
  只支持 uniform decode，通常每个 request 一个 token。

UNIFORM_BATCH
  支持更宽松的 uniform batch。

ALWAYS
  更强的 graph 支持。
```

`GPUModelRunner` 会收集所有 builder 的支持等级，取最保守结果来决定最终 CUDA graph mode。

### 11.1 build_for_cudagraph_capture()

capture 阶段调用：

```text
builder.build_for_cudagraph_capture(common_attn_metadata)
```

这个方法存在的原因是：

```text
普通 build 可能包含动态 plan、动态 mask、CPU/GPU sync 或按真实 batch 分支；
捕获 graph 时需要更稳定的 metadata。
```

不同 backend 处理方式不同：

```text
FlashAttention：按 FA2 / FA3 / 平台能力决定支持级别；
Triton：可能构造简化 seq_lens，避免 capture 过慢；
FlexAttention：预构建 block mask；
MLA：full graph 多数只支持 decode-only；
Mamba / GDN：通常对 decode-only 支持更友好。
```

### 11.2 padding 对 metadata 的影响

CUDA graph 下，runner 可能把：

```text
num_tokens
num_reqs
```

padding 到 capture size。

runner 调用处因此会同时维护真实尺寸和 padded 尺寸：

```text
num_tokens / num_reqs：真实 token / request 数；
batch_desc.num_tokens / num_reqs_padded：执行图看到的 padded token / request 数；
CommonAttentionMetadata.num_actual_tokens：传给 builder 的执行 token 数，FULL CUDA graph 下可能是 padded token 数；
slot_mapping padded 区域通常填 -1；
block table padded rows 会填 `NULL_BLOCK_ID`（当前为 0，保留给 padding）。
```

builder 必须保证 backend kernel 不会把 padded token 当作真实请求处理；spec drafter 若需要真实视图，会在返回前通过 `CommonAttentionMetadata.unpadded(...)` 裁掉 padding。

---

## 12. Cascade attention 对 builder 的影响

cascade attention 解决的是多个请求共享 prefix KV 时的重复计算问题。

runner 相关函数：

```text
GPUModelRunner._compute_cascade_attn_prefix_lens(...)
GPUModelRunner._compute_cascade_attn_prefix_len(...)
```

大致流程：

```text
SchedulerOutput.num_common_prefix_blocks
  → runner 计算 common_prefix_len
  → builder.use_cascade_attention(...) 判断是否启用
  → builder.build(common_prefix_len, common_attn_metadata)
  → backend metadata 中携带 cascade 所需字段
```

是否启用不只看 prefix 长度，还要看：

```text
backend 是否支持；
batch 是否适合；
当前是否 CUDA graph；
模型是否禁用 cascade；
query / KV 形态是否满足 backend 要求。
```

不同 backend 差异：

```text
FlashAttention：典型支持；
FlashInfer：代码中有 cascade wrapper 路径，但当前 `use_cascade_attention()` 返回 `False`，常规调度不会自动启用；
FlexAttention：通常不启用；
Triton：dataclass 可能有字段，但普通 builder 不主动启用；
MLA / Mamba / GDN：按各自实现能力处理。
```

---

## 13. Spec decode 对 builder 的影响

Speculative decoding 会让“本轮 query tokens”不再只是普通 token。

它可能包含：

```text
target token；
draft tokens；
bonus token；
accepted tokens；
rejected tokens 对应的后处理关系。
```

对 attention metadata 的影响包括：

```text
query length 可能大于 1，即使这是 decode 阶段；
某些 backend 可以把 spec decode 当成 decode 处理；
某些 backend 需要把 spec decode 和 non-spec decode 分开；
Mamba / GDN 这类 stateful layer 还要知道 accepted token 和 state index。
```

典型差异：

```text
FlashInfer：可以通过 spec-as-decode 能力调整 reorder threshold；
MLA：query_len_support 会影响 spec decode 是否仍走 decode path；
GDN：显式拆 spec / non-spec，并可能重分类；
Mamba：metadata 中维护 accepted tokens、state indices 等。
```

所以 spec decode metadata 不等于 sampler metadata。

```text
SpecDecodeMetadata
  主要服务 logits / sampler / rejection。

AttentionMetadata / CommonAttentionMetadata
  主要服务本轮 token 如何 attend、读写哪些 KV / state。
```

两者在 `execute_model_state` 中都会被保存，后续 `sample_tokens()` 可能继续使用。

---

## 14. DCP / context parallel 对 builder 的影响

DCP 会让单个 request 的 context 分布在多个 rank 或局部上下文中。

common metadata 中的字段是：

```text
dcp_local_seq_lens
dcp_local_seq_lens_cpu
```

builder 需要把它翻译成 backend 能理解的结构。

不同 backend 做法：

```text
FlashAttention
  基于 context_kv_lens 重新计算 dcp_context_kv_lens / max_dcp_context_kv_len。

FlashInfer
  调整 paged KV lens，并选择支持 DCP 的 wrapper / plan。

MLA
  decode 侧使用 dcp_local_seq_lens，chunked prefill 侧构造 local chunk seq lens 和 chunked context metadata。

Triton / Flex
  按具体支持能力使用 common metadata 或走 fallback。
```

DCP 下一个重要原则是：

```text
尽量避免为了构造 metadata 触发 GPU → CPU sync。
```

因此很多 builder 会优先使用已有 CPU side upper bound 或 GPU tensor plan。

---

## 15. metadata 如何进入 forward

metadata 构造完成后，不会作为普通参数一层层传给每个 module，而是放进 forward context。

位置：`code/vllm/vllm/forward_context.py:260`

```text
set_forward_context(...)
```

其中包括：

```text
attn_metadata
slot_mapping
dp_metadata
cudagraph_runtime_mode
batch_descriptor
ubatch_slices
```

attention layer 执行时再取：

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:726`

```text
get_attention_context(layer_name)
```

最终调用：

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:813`

```text
unified_attention_with_output(...)
  → self.impl.forward(..., attn_metadata, ...)
```

这条链路说明：

```text
builder 构造的是 forward 的隐式参数；
forward context 是 metadata 从 ModelRunner 进入模型层的通道；
backend impl 才是 metadata 的最终消费者。
```

---

## 16. 一个完整例子：普通 FlashAttention decode

假设当前是普通 decoder-only 模型，backend 选中 FlashAttention，本轮是三个请求 decode：

```text
num_scheduled_tokens = [1, 1, 1]
```

执行链路：

```text
1. _prepare_inputs()
   query_start_loc = [0, 1, 2, 3]
   positions = 每个请求的新 token position
   seq_lens = 每个请求已有长度 + 1

2. _get_slot_mappings()
   根据 block_table 和 positions 算出三个 token 的 KV slot。

3. _build_attention_metadata()
   构造 CommonAttentionMetadata。

4. FlashAttentionMetadataBuilder.build(...)
   生成 FlashAttentionMetadata：
     query_start_loc
     seq_lens
     block_table
     slot_mapping
     max_query_len = 1
     max_seq_len

5. set_forward_context(...)
   按 layer_name 注入 metadata。

6. Attention.forward()
   从 context 取 metadata 和 kv_cache。

7. FlashAttentionImpl.forward(...)
   写入当前 token K/V，并对历史 KV 做 paged attention。
```

这个例子里 builder 基本是“公共 batch 描述 → FlashAttention kernel 参数”的翻译器。

---

## 17. 一个完整例子：FlashInfer mixed batch

假设 batch 中同时存在：

```text
request A：decode，query_len = 1
request B：chunked prefill，query_len = 128
request C：decode，query_len = 1
```

FlashInfer builder 的处理更复杂：

```text
1. 根据 query_start_loc / query_lens 拆分 decode 和 prefill。
2. 必要时要求 runner 预先 reorder batch，让 decode / prefill 连续。
3. 构造 decode metadata：FIDecode 或 TRTLLMDecode。
4. 构造 prefill metadata：FIPrefill 或 TRTLLMPrefill。
5. 从 block_table 构造 paged_kv_indptr / paged_kv_indices / paged_kv_last_page_len。
6. 返回 FlashInferMetadata(prefill=..., decode=...)。
```

所以 FlashInfer builder 的心智模型是：

```text
先分类，再 plan wrapper，最后把每类 token 映射到 FlashInfer / TRT-LLM kernel 所需的 paged KV 表示。
```

---

## 18. 一个完整例子：MLA prefill + decode

MLA 的 metadata 不只是标准 attention metadata 的变体。

假设 batch 同时有 prefill 和 decode：

```text
prefill tokens：需要 MHA-style 计算；
decode tokens：需要 MQA-style 读取 compressed KV。
```

MLA builder 会构造：

```text
MLACommonMetadata(
  prefill=MLACommonPrefillMetadata(...),
  decode=MLACommonDecodeMetadata(...),
)
```

后续 `MLAAttention.forward_impl()` 会根据 metadata 中的 token 分布分别调用：

```text
impl.forward_mha(...)
impl.forward_mqa(...)
```

这就是 MLA builder 与标准 builder 的最大差异：

```text
标准 attention builder 多数是在给一个 attention kernel 准备参数；
MLA builder 是在组织两套计算路径，并协调 prefill backend、decode backend、chunked context workspace 和 compressed KV cache。
```

---

## 19. 容易疑惑的点

### 19.1 CommonAttentionMetadata 是最终 metadata 吗？

不是。

```text
CommonAttentionMetadata 是公共输入；
AttentionMetadata 是 backend-specific 输出。
```

真正传给 backend impl 的是具体 backend 的 metadata，例如 `FlashAttentionMetadata`、`FlashInferMetadata`、`MLACommonMetadata`。

### 19.2 builder 是每层一个吗？

不是严格每层一个。

```text
同一 AttentionGroup 的多层共享 builder 和 metadata；
ubatching 开启时，同一 group 可能有多个 builder 实例。
```

最终 forward context 仍按 layer name 保存，是为了 attention layer 取用方便。

### 19.3 backend selection 和 metadata build 是同一件事吗？

不是。

```text
backend selection：模型初始化 / layer 初始化阶段选择 backend class；
metadata build：每轮 execute_model 根据当前 batch 构造 backend metadata。
```

backend 通常固定，metadata 每轮变化。

### 19.4 block table 和 slot mapping 谁负责？

```text
Scheduler / KVCacheManager：决定逻辑 block 分配；
InputBatch / block table：保存请求到 block 的映射；
_get_slot_mappings()：把 positions + block table 展开成 token 级 slot_mapping；
_build_attention_metadata()：把 block table / slot_mapping 放进 common metadata；
builder：翻译成 backend-specific 表示。
```

### 19.5 为什么有些 builder 要 reorder batch？

因为很多高性能 kernel 对 batch 形态有偏好。

```text
decode 连续放在一起，可以走 decode kernel；
prefill 连续放在一起，可以走 prefill kernel；
short extend 和 long extend 分开，可以避免小 query 被大 query 拖慢。
```

`reorder_batch_threshold` 是 builder 向 runner 表达这种偏好的方式。

### 19.6 为什么 CUDA graph 需要专门 build？

因为普通 metadata build 可能包含动态 shape、动态 mask、动态 wrapper plan 或 CPU/GPU 同步。

CUDA graph capture 需要更稳定的结构，所以 builder 可以实现：

```text
build_for_cudagraph_capture(...)
```

### 19.7 为什么 Mamba / GDN 也在 attention metadata builder 体系里？

因为它们虽然不是传统 attention，但在执行层面同样需要：

```text
request 边界；
state cache 映射；
prefill / decode 分类；
spec decode accepted token；
CUDA graph / ubatch 支持；
forward context 注入。
```

复用 builder 管线可以让 ModelRunner 用统一方式调度这些 layer。

---

## 20. 调试入口

如果要调试 attention metadata，建议按这条顺序下断点：

```text
1. GPUModelRunner._prepare_inputs()
   看 query_start_loc、seq_lens、positions、num_scheduled_tokens。

2. GPUModelRunner._get_slot_mappings()
   看 slot_mapping 是否和 block_table / positions 对得上。

3. GPUModelRunner._build_attention_metadata()
   看 CommonAttentionMetadata base 和 per-group 替换。

4. AttentionGroup.create_metadata_builders()
   看每个 group 绑定了哪个 backend builder。

5. 具体 backend 的 builder.build()
   看 common metadata 如何变成 backend metadata。

6. set_forward_context(...)
   看 metadata 是否按 layer_name 注入。

7. get_attention_context(layer_name)
   看 attention layer 取到的 metadata 是否正确。

8. backend impl.forward(...)
   看 kernel 实际消费的字段。
```

常用测试入口：

```text
code/vllm/tests/v1/attention/utils.py
  create_common_attn_metadata(...)

code/vllm/tests/v1/attention/test_attention_backends.py
  多 backend 通用 attention 测试。

code/vllm/tests/v1/attention/test_gdn_metadata_builder.py
  GDN metadata builder 分类测试。
```

---

## 21. 最终可以记成一张表

| 阶段 | 主要函数 / 类 | 核心输入 | 核心输出 | 作用 |
|---|---|---|---|---|
| backend 选择 | `get_attn_backend()` / `AttentionBackend` | head size、dtype、KV dtype、attention type、平台能力 | backend class | 决定 impl / builder / KV cache 协议 |
| backend 分组 | `initialize_attn_backend()` | layer、KVCacheSpec、backend | `AttentionGroup` | 找出可共享 metadata 的 layer 组 |
| builder 创建 | `initialize_metadata_builders()` | `AttentionGroup`、config、device、kernel_block_size | `AttentionMetadataBuilder` | 为每个 group 创建 metadata 构造器；ubatching 时每个 group 多个实例 |
| 输入准备 | `_prepare_inputs()` | `SchedulerOutput`、`InputBatch` | query_start_loc、seq_lens、positions | 生成 token / request 边界信息 |
| slot 映射 | `_prepare_inputs()` / `_get_slot_mappings()` | block table、positions、padding 信息 | slot_mapping | 先由 block table kernel 生成 token 到 KV slot 的映射，再按 group / layer 取视图 |
| 公共 metadata | `_build_attention_metadata()` | batch 字段、block table、slot mapping、encoder_seq_lens | `CommonAttentionMetadata` | 形成跨 backend 的公共 batch 描述 |
| backend metadata | `builder.build()` / `build_for_cudagraph_capture()` | `CommonAttentionMetadata`、common_prefix_len | backend-specific `AttentionMetadata` | 生成 kernel / wrapper 可消费的 metadata |
| context 注入 | `set_forward_context()` | `attn_metadata`、slot_mapping | `ForwardContext` | 让模型层能按 layer name 取 metadata |
| attention 执行 | `Attention.forward()` / `unified_attention_with_output()` / impl | QKV、KV cache、metadata | attention output | 调用具体 backend kernel |

---

## 22. 总结

`AttentionMetadataBuilder` 可以压缩成下面这条线：

```text
SchedulerOutput / InputBatch
  → _prepare_inputs()
  → query_start_loc / seq_lens / positions
  → _get_slot_mappings()
  → block_table / slot_mapping
  → _build_attention_metadata()
  → CommonAttentionMetadata
  → backend.get_builder_cls().build(...) / build_for_cudagraph_capture(...)
  → backend-specific AttentionMetadata
  → set_forward_context(...)
  → Attention.forward()
  → backend impl.forward(...)
```

它的核心价值是把复杂的执行状态分层：

```text
ModelRunner 只负责构造公共 batch 描述；
builder 负责理解 backend 的 kernel 协议；
impl 负责真正执行 attention；
forward context 负责把 metadata 从执行层传进模型层。
```

如果只记住最小心智模型：

```text
CommonAttentionMetadata 是“公共语言”，AttentionMetadataBuilder 是“翻译器”，backend-specific AttentionMetadata 是“kernel 方言”。
```