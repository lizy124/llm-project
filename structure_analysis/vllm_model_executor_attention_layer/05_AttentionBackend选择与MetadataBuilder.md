# 05 AttentionBackend 选择与 MetadataBuilder

本篇梳理 V1 attention 抽象层：`AttentionBackend`、`AttentionImpl`、`AttentionMetadataBuilder`、`CommonAttentionMetadata`，以及 backend 选择流程。

## 1. 相关文件

```text
vllm/v1/attention/backend.py
vllm/v1/attention/selector.py
vllm/v1/attention/backends/registry.py
vllm/v1/attention/backends/*
vllm/v1/attention/ops/*
```

关键锚点：

- `AttentionBackend`：`code/vllm/vllm/v1/attention/backend.py:55`
- `CommonAttentionMetadata`：`code/vllm/vllm/v1/attention/backend.py:362`
- `AttentionMetadataBuilder`：`code/vllm/vllm/v1/attention/backend.py:533`
- `AttentionImplBase`：`code/vllm/vllm/v1/attention/backend.py:702`
- `AttentionImpl`：`code/vllm/vllm/v1/attention/backend.py:780`
- `get_attn_backend()`：`code/vllm/vllm/v1/attention/selector.py:54`

## 2. 四层抽象关系

```text
Attention 层
  ↓ 选择/持有
AttentionBackend
  ↓ 创建
AttentionImpl
  ↑ 使用
AttentionMetadataBuilder
  ↑ 输入
CommonAttentionMetadata
```

职责拆分：

| 对象 | 职责 |
|---|---|
| `AttentionBackend` | 描述 backend 能力、KV cache shape/layout、支持哪些配置 |
| `AttentionImpl` | 单层 attention 的实际实现对象，执行 forward |
| `AttentionMetadataBuilder` | 把公共 batch metadata 转成 backend-specific metadata |
| `CommonAttentionMetadata` | GPUModelRunner 构造的公共 attention metadata |
| `Attention` | 模型层 nn.Module，持有 backend/impl 并调用它们 |

## 3. AttentionBackend

`AttentionBackend` 是 backend 家族抽象。

它需要声明：

### 3.1 backend identity

```text
get_name()
get_impl_cls()
get_builder_cls()
```

- `get_name()` 返回 backend 名称；
- `get_impl_cls()` 返回具体 `AttentionImpl` 类；
- `get_builder_cls()` 返回 metadata builder 类。

### 3.2 KV cache shape/layout

```text
get_kv_cache_shape()
get_kv_cache_block_dim()
get_kv_cache_stride_order()
get_required_kv_cache_layout()
```

这些方法决定 worker 如何分配和 reshape KV cache tensor。

### 3.3 能力检查

```text
supports_head_size()
supports_dtype()
supports_kv_cache_dtype()
supports_block_size()
supports_attn_type()
supports_compute_capability()
supports_combination()
validate_configuration()
```

backend 选择时会用这些能力过滤不兼容实现。

### 3.4 特性声明

```text
is_mla()
supports_sink()
supports_alibi_sqrt()
supports_mm_prefix()
is_sparse()
supports_per_head_quant_scales()
supports_non_causal()
supports_batch_invariance()
supports_kv_connector()
is_ssm()
```

这些决定 backend 是否可用于特殊模型结构或特殊推理场景。

## 4. AttentionSelectorConfig

`selector.py` 中有 `AttentionSelectorConfig`，用于封装 backend 选择条件。

它通常包含：

- head size；
- dtype；
- kv cache dtype；
- block size；
- 是否 MLA；
- 是否需要 sink；
- 是否需要 sliding window；
- attention type；
- 是否使用 mm prefix；
- 是否需要 per-head scales；
- 是否 sparse；
- kv sharing target；
- 当前平台能力。

## 5. get_attn_backend 选择流程

入口：`code/vllm/vllm/v1/attention/selector.py:54`。

简化流程：

```text
Attention.__init__
  ↓
get_attn_backend(...)
  ↓
构造 AttentionSelectorConfig
  ↓
_cached_get_attn_backend(...)
  ↓
current_platform.get_attn_backend_cls(...)
  ↓
backend.validate_configuration(...)
  ↓
返回 AttentionBackend class
```

`current_platform.get_attn_backend_cls` 会根据平台选择 CUDA/ROCm/CPU/XPU 对应 backend。

## 6. backend registry

`vllm/v1/attention/backends/registry.py` 维护 backend 名称到类的映射。

常见 backend 类型包括：

- FlashAttention；
- FlashInfer；
- Triton；
- ROCm attention；
- CPU attention；
- MLA attention；
- sparse attention；
- no attention；
- platform-specific optimized backend。

具体可用 backend 取决于当前环境、安装依赖和模型配置。

## 7. CommonAttentionMetadata

`CommonAttentionMetadata` 是 GPUModelRunner 生成的公共格式，位置：`code/vllm/vllm/v1/attention/backend.py:362`。

它包含运行时 batch 信息，例如：

- `query_start_loc`；
- `seq_lens`；
- `block_table_tensor`；
- `slot_mapping`；
- `positions`；
- `num_computed_tokens`；
- `is_prefilling`；
- `max_query_len`；
- `max_seq_len`；
- multimodal prefix ranges；
- DCP/PCP local seq lens；
- common prefix/cascade attention 信息。

它是 backend-specific metadata 的输入。

## 8. AttentionMetadataBuilder

`AttentionMetadataBuilder` 定义在 `code/vllm/vllm/v1/attention/backend.py:533`。

职责：

1. 接收 `CommonAttentionMetadata`；
2. 根据 backend 需求构造具体 metadata；
3. 处理 prefill/decode 差异；
4. 处理 CUDA graph capture metadata；
5. 更新 block table；
6. 判断是否使用 cascade attention；
7. 为 speculative drafting 构造 metadata。

主要方法：

| 方法 | 作用 |
|---|---|
| `build()` | 构造当前 batch 的 backend metadata |
| `update_block_table()` | 更新 block table |
| `build_for_cudagraph_capture()` | CUDA graph capture 用 metadata |
| `build_for_drafting()` | speculative drafting 用 metadata |
| `use_cascade_attention()` | 判断是否启用 cascade attention |
| `get_cudagraph_support()` | 声明 CUDA graph 支持程度 |

## 9. AttentionImpl

`AttentionImplBase` 在 `code/vllm/vllm/v1/attention/backend.py:702`。

`AttentionImpl` 在 `code/vllm/vllm/v1/attention/backend.py:780`。

它是实际执行 attention 的对象。

主要职责：

- 保存单层 attention 参数；
- 执行 `forward()`；
- 处理 KV cache quant mode；
- 支持 fused output quant；
- 支持 fused RoPE + KV cache update；
- 对 MLA 实现 MHA/MQA 特化路径；
- 调用底层 ops/custom kernels。

## 10. MLA AttentionImpl

文件中还有：

- `MLAAttentionImpl`：`code/vllm/vllm/v1/attention/backend.py:863`
- `SparseMLAAttentionImpl`：`code/vllm/vllm/v1/attention/backend.py:954`

它们为 MLA/sparse MLA 模型提供不同 forward 接口：

- `forward_mha()`；
- `forward_mqa()`；
- `do_kv_cache_update()`。

这类实现常见于 DeepSeek 系列等特殊 attention 架构。

## 11. backend 选择影响什么

backend 一旦选定，会影响：

1. KV cache tensor shape；
2. KV cache layout/stride；
3. metadata builder 类型；
4. attention forward kernel；
5. 是否 forward 内更新 KV cache；
6. 是否支持 CUDA graph；
7. 是否支持 prefix/multimodal/sink/non-causal；
8. 是否支持 KV quant；
9. 是否支持特定 head size/block size。

因此 backend 选择错误可能导致：

- 初始化失败；
- KV cache shape 不匹配；
- kernel 报错；
- 输出错误；
- 性能明显下降。

## 12. backend 与 GPUModelRunner 的关系

GPUModelRunner 在初始化 KV cache 和构造 attention metadata 时，会使用 backend 提供的信息：

```text
backend.get_kv_cache_shape
backend.get_kv_cache_stride_order
backend.get_builder_cls
metadata_builder.build
```

所以 backend 不是只在 `Attention.forward` 才生效，而是从 KV cache 分配阶段就已经影响运行时布局。

## 13. 一句话总结

`AttentionBackend` 描述“这套 attention 实现能做什么、需要什么布局”，`AttentionMetadataBuilder` 把 batch runtime 信息转换成它需要的 metadata，`AttentionImpl` 执行实际 attention，而 `selector.py` 负责根据模型、平台和配置选择正确 backend。
