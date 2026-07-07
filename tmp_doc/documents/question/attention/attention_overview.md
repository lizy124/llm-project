# vLLM V1 Attention 子系统逻辑梳理

源码位置：

- `code/vllm/vllm/v1/attention/`
- `code/vllm/vllm/model_executor/layers/attention/`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `code/vllm/vllm/v1/worker/gpu/block_table.py`
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/block_pool.py`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/config/attention.py`
- `code/vllm/vllm/platforms/cuda.py`
- `code/vllm/vllm/compilation/`

本文用于总览 vLLM V1 Attention 子系统，重点梳理 attention 在执行链路中的位置：backend 如何选择，metadata 如何构造，prefill / decode 如何区分，slot mapping / block table 如何连接 paged KV cache，attention forward 如何读写 KV，以及 cascade attention、KV connector hook、CUDA graph / compile 如何挂接。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的写法，本文按“先定角色，再走主链路，再拆关键阶段，最后总结接口和数据结构”的方式组织。

要回答的问题分成 10 组：

```text
1. Attention 子系统在 vLLM V1 中是哪一层？负责什么，不负责什么？
2. AttentionBackend / AttentionMetadataBuilder / AttentionImplBase 各自负责什么？
3. vLLM 如何选择 FlashAttention / FlashInfer / FlashMLA / Triton 等 backend？
4. GPUModelRunner 如何构造 attention metadata？
5. prefill / decode / chunked prefill / mixed batch / spec decode 的 metadata 有什么差异？
6. slot mapping / block table 如何把 request tokens 映射到 paged KV cache？
7. attention forward 如何读写 KV cache？
8. cascade attention / prefix cache 命中如何影响 attention？
9. KV connector hook 如何挂到 attention layer 边界？
10. CUDA graph / torch.compile 对 attention 路径有什么约束？
```

阅读顺序建议：

```text
attention_overview.md
  → 01_attention_role.md
  → 02_backend_selection.md
  → 03_attention_metadata_builder.md
  → 04_prefill_decode_metadata.md
  → 05_slot_mapping_and_block_table.md
  → 06_attention_forward_flow.md
  → 07_kv_cache_layout_and_backend.md
  → 08_cascade_attention.md
  → 09_attention_and_kv_connector_hooks.md
  → 10_cuda_graph_compile_interaction.md
```

如果要专门理解各种 attention 名词、算法家族和 backend 家族，再读：

```text
attention_methods/11_attention_variants_overview.md
  → attention_methods/12_flash_attention_family.md
  → attention_methods/13_paged_attention.md
  → attention_methods/14_mha_mqa_gqa.md
  → attention_methods/15_mla_attention.md
  → attention_methods/16_sliding_window_and_local_attention.md
  → attention_methods/17_flashinfer_flashmla_triton_backends.md
  → attention_methods/18_hma_and_kv_cache_layout.md
```

---

## 1. 一句话总览

vLLM V1 的 attention 子系统不是单独一段 kernel 调用，而是一条从“调度结果”到“模型层执行”的翻译链：

```text
SchedulerOutput
  → GPUModelRunner._update_states()
  → GPUModelRunner._prepare_inputs()
  → block table / slot mapping
  → GPUModelRunner._build_attention_metadata()
  → set_forward_context(attn_metadata, slot_mapping, ...)
  → model attention layers
  → AttentionBackend / AttentionImplBase
  → paged KV cache read / write
  → hidden states / logits / sampling
```

一句话记忆：

```text
Scheduler 决定本轮跑哪些 token；
ModelRunner 把这些 token 翻译成 attention metadata；
Attention layer / backend 根据 metadata 读写 paged KV cache 并执行 attention kernel。
```

attention 子系统横跨三类代码：

```text
1. 抽象和 backend：
   code/vllm/vllm/v1/attention/

2. 模型层入口：
   code/vllm/vllm/model_executor/layers/attention/

3. 每轮执行时的 metadata 构造：
   code/vllm/vllm/v1/worker/gpu_model_runner.py
   code/vllm/vllm/v1/worker/gpu_input_batch.py
   code/vllm/vllm/v1/worker/gpu/block_table.py
```

所以理解 attention 不能只看 `flash_attn.py` 或某个 kernel，而要一起看：

```text
backend 选择
  → KV cache shape / layout
  → InputBatch block table
  → slot mapping
  → CommonAttentionMetadata
  → backend-specific metadata
  → ForwardContext
  → Attention.forward()
  → AttentionImpl.forward()
```

---

## 2. Attention 子系统负责什么，不负责什么

### 2.1 它负责什么

Attention 子系统负责把“当前 batch 的 token、位置、KV 布局”变成具体 backend 可执行的 attention 调用。

核心职责包括：

```text
- 根据平台、模型结构、dtype、KV dtype、block size 等条件选择 backend；
- 定义每种 backend 的 KV cache shape / stride / layout；
- 把 block table / slot mapping / seq lens / query_start_loc 组装成 metadata；
- 在模型 forward 期间让每一层 attention 能取到自己的 metadata、KV cache 和 slot mapping；
- 将当前 token 的 K/V 写入 paged KV cache；
- 从 block table 指向的历史 KV blocks 中读取上下文；
- 根据 prefill / decode / mixed batch / spec decode / cascade attention 选择不同 kernel 路径；
- 配合 KV connector 做逐层 KV load / save；
- 配合 CUDA graph / torch.compile 处理固定形状、opaque op 和 graph capture。
```

### 2.2 它不负责什么

Attention 子系统不负责调度、采样和资源账本本身。

```text
不负责：
- 决定哪些 request 本轮被调度；
- 决定 token budget；
- 分配、抢占、释放 KV block 的账本；
- 决定 prefix cache 命中；
- 决定外部 KV 什么时候可用；
- 计算 logits 后采样 token；
- 处理最终 RequestOutput。
```

这些由 Scheduler、KVCacheManager、ModelRunner sampling 阶段和 EngineCore 负责。

边界可以记成：

```text
Scheduler / KVCacheManager：
  管“哪些 token 能跑、哪些 KV block 可用”。

ModelRunner / Attention metadata：
  管“这些 token 和 KV block 如何喂给 attention”。

AttentionBackend / AttentionImpl：
  管“用哪个 kernel 真正算 attention”。
```

---

## 3. 核心角色

### 3.1 `AttentionBackend`

源码位置：`code/vllm/vllm/v1/attention/backend.py:55`

`AttentionBackend` 是“后端能力描述 + 工厂入口 + KV cache 布局定义”。

它负责：

```text
- 声明 backend 名称；
- 返回 impl class；
- 返回 metadata builder class；
- 定义 KV cache shape；
- 定义 KV cache stride order / block dim；
- 声明支持哪些 head size、dtype、kv cache dtype、block size；
- 声明是否支持 MLA、sparse、sink、non-causal、KV connector、batch invariant 等；
- 在 backend 选择时做 validate_configuration()。
```

典型接口包括：

```text
get_name()
get_impl_cls()
get_builder_cls()
get_kv_cache_shape()
get_kv_cache_stride_order()
get_kv_cache_block_dim()
validate_configuration()
```

一句话：

```text
AttentionBackend 说明“这个 backend 能不能用、KV cache 长什么样、最终谁来执行”。
```

### 3.2 `CommonAttentionMetadata`

源码位置：`code/vllm/vllm/v1/attention/backend.py:393`

`CommonAttentionMetadata` 是所有 backend 共享的 batch 级 attention 描述。

它携带：

```text
query_start_loc：每个 request 的 query token 起止边界；
seq_lens：每个 request 当前序列长度；
num_reqs：当前 request 数；
num_actual_tokens：metadata 当前携带的 token 数；普通路径等于真实 token 数，但 FULL CUDA graph / padding 路径可能包含 padded token；
max_query_len：本轮单 request 最大 query 长度；
max_seq_len：当前最大上下文长度；
block_table_tensor：request → KV blocks；
slot_mapping：token → KV cache slot；
causal：是否 causal；
positions：token 位置；
is_prefilling：哪些 request 处于 prefill/chunked prefill；
encoder_seq_lens / dcp_local_seq_lens / mm_req_doc_ranges：特殊模型路径需要的辅助字段。
```

它不是最终 kernel 参数，而是 backend-specific builder 的输入。

一句话：

```text
CommonAttentionMetadata 是“所有 backend 都能看懂的公共 batch 描述”。
```

### 3.3 `AttentionMetadataBuilder`

源码位置：`code/vllm/vllm/v1/attention/backend.py:565`

`AttentionMetadataBuilder` 负责把 `CommonAttentionMetadata` 转换成某个 backend 私有的 metadata。

它负责：

```text
- 接收 CommonAttentionMetadata；
- 结合 KVCacheSpec / layer_names / vLLM config / device；
- 构造 FlashAttentionMetadata、FlashInfer metadata、Triton metadata、MLA metadata 等；
- 处理 cascade attention 的 prefix/suffix metadata；
- 处理 CUDA graph capture 所需的固定 shape metadata；
- 必要时通过 update_block_table() 复用已有 metadata，只替换 block table / slot mapping；
- 声明该 backend 对 CUDA graph 的支持级别。
```

典型接口：

```text
build(common_prefix_len, common_attn_metadata)
build_for_cudagraph_capture(...)
build_for_drafting(...)
update_block_table(...)
use_cascade_attention(...)
get_cudagraph_support()
```

一句话：

```text
AttentionMetadataBuilder 负责“把公共 metadata 翻译成某个 backend 能直接消费的格式”。
```

### 3.4 `AttentionImplBase` / `AttentionImpl`

源码位置：

- `code/vllm/vllm/v1/attention/backend.py:734`
- `code/vllm/vllm/v1/attention/backend.py:812`

`AttentionImplBase` 是具体 attention 执行实现的基类，保存通用属性：

```text
num_heads
head_size
scale
num_kv_heads
alibi_slopes
sliding_window
kv_cache_dtype
DCP / PCP world size 和 rank
是否需要返回 decode LSE
是否支持 quant query input
```

`AttentionImpl` 定义普通 attention 的 forward 接口。

对 MLA，则使用 `MLAAttentionImpl`，拆成：

```text
forward_mha()：prefill 侧 MHA 路径；
forward_mqa()：decode 侧 MQA 路径；
do_kv_cache_update()：MLA KV cache 写入。
```

一句话：

```text
AttentionImplBase / AttentionImpl 负责“真正调 kernel 算 attention”。
```

### 3.5 `Attention layer`

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:318`

模型里的 attention layer 是后端实现和模型 forward 的连接点。

它负责：

```text
- 在初始化时选择 backend；
- 创建 backend impl；
- 注册到 `compilation_config.static_forward_context`，forward 时经 `ForwardContext.no_compile_layers` 访问；
- forward 时从 ForwardContext 取当前层的 metadata / KV cache / slot mapping；
- 必要时先写 KV cache；
- 调用 backend impl forward；
- 对 torch.compile 暴露 unified attention opaque op。
```

一句话：

```text
Attention layer 是“模型层里调用 attention backend 的统一入口”。
```

### 3.6 `ForwardContext`

源码位置：`code/vllm/vllm/forward_context.py:128`

`ForwardContext` 是模型 forward 生命周期内的隐式参数区。

它保存：

```text
attn_metadata：按 layer name 组织的 attention metadata；
slot_mapping：按 layer name 组织的 slot mapping；
no_compile_layers：forward 期间可直接访问的静态 layer 对象映射，来源于 compilation_config.static_forward_context；
cudagraph_runtime_mode：当前是 FULL / PIECEWISE / NONE；
batch_descriptor：CUDA graph dispatch 需要的 batch 描述；
ubatch_slices：microbatch 切片；
dp_metadata：DP 场景的 token 信息。
```

`GPUModelRunner.execute_model()` 在模型 forward 前调用 `set_forward_context()` 注入这些信息。

一句话：

```text
ForwardContext 让模型内部 attention layer 不用显式传参，也能拿到本轮 metadata 和 KV cache。
```

---

## 4. Backend 如何选择

### 4.1 选择入口在 Attention layer 初始化

模型构建 attention layer 时，如果没有显式传入 `attn_backend`，会调用 `get_attn_backend()`。

源码位置：

- `code/vllm/vllm/model_executor/layers/attention/attention.py:317`
- `code/vllm/vllm/v1/attention/selector.py:54`

选择时会传入：

```text
head_size
num_heads
num_kv_heads
dtype
kv_cache_dtype
block_size
is_attention_free
is_hybrid
is_mla
use_sparse
has_sink
use_mm_prefix
attention_type
use_kv_connector
```

也就是说 backend 选择不是只看平台，而是同时看模型结构、KV cache 配置和运行功能。

### 4.2 selector 层做配置归一

`AttentionSelectorConfig` 负责把这些条件归一成一个可缓存的配置对象。

源码位置：`code/vllm/vllm/v1/attention/selector.py:21`

然后 `_cached_get_attn_backend()` 会：

```text
1. 读取 vllm_config.attention_config.backend；
2. 如果用户指定 backend，则优先使用指定 backend；
3. 否则委托 current_platform.get_attn_backend_cls() 自动选择；
4. 通过 AttentionBackendEnum lazy import 后端类；
5. 如果 backend 要求特定 KV cache layout，则更新全局 KV cache layout。
```

### 4.3 registry 维护 backend 枚举

源码位置：`code/vllm/vllm/v1/attention/backends/registry.py:34`

`AttentionBackendEnum` 维护 backend 名称到 class path 的映射，例如：

```text
FLASH_ATTN
FLASHINFER
TRITON_ATTN
FLEX_ATTENTION
FLASHMLA
FLASH_ATTN_MLA
CUTLASS_MLA
FLASHINFER_MLA
TOKENSPEED_MLA
ROCM_AITER_MLA
ROCM_AITER_TRITON_MLA
ROCM_AITER_UNIFIED_ATTN
CPU_ATTN
ROCM_ATTN
TURBOQUANT
CUSTOM
```

registry 还支持自定义 backend 注册。

### 4.4 CUDA 平台的自动优先级

源码位置：`code/vllm/vllm/platforms/cuda.py:82`

CUDA 平台会按优先级尝试 backend。

非 MLA 常见优先级可以理解为：

```text
Blackwell / compute capability major 10：
  FlashInfer → FlashAttention → Triton → FlexAttention → TurboQuant

其他 CUDA：
  FlashAttention → FlashInfer → Triton → FlexAttention → TurboQuant
```

MLA backend 会根据 device capability、head count、KV dtype、sparse 需求等走另一套优先级。

最终逻辑是：

```text
如果用户指定 backend：
  validate_configuration() 通过就用，否则报错。

如果 auto：
  按平台优先级逐个 validate_configuration()；
  选择第一个满足条件的 backend。
```

### 4.5 FlashAttention backend 示例

源码位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py`

`FlashAttentionBackend` 说明了典型 backend 需要定义什么：

```text
get_name()：FLASH_ATTN
get_impl_cls()：FlashAttentionImpl
get_builder_cls()：FlashAttentionMetadataBuilder
get_kv_cache_shape()：(num_blocks, 2, block_size, num_kv_heads, head_size)
get_kv_cache_stride_order()：根据 NHD / HND layout 决定 stride
validate_head_size() / validate_dtype() / validate_kv_cache_dtype() / validate_block_size()
```

对应关系是：

```text
FlashAttentionBackend
  → FlashAttentionMetadataBuilder
  → FlashAttentionMetadata
  → Attention.forward() 先调用 do_kv_cache_update()
  → FlashAttentionImpl.forward()
  → flash_attn_varlen_func / cascade_attention
```

---

## 5. GPUModelRunner 如何构造 attention metadata

### 5.1 初始化阶段：按 KV cache group 创建 builders

`GPUModelRunner` 初始化 KV cache 后，会根据 layer 与 KV cache group 的关系创建 attention groups。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6766`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6810`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6846`

核心过程：

```text
1. 从 KV cache group 的 layer names 找到 attention layer；
2. 读取每层 attention layer 的 backend；
3. 按 backend class + KVCacheSpec + num_heads_q 聚合成 AttentionGroup；
4. 每个 AttentionGroup 创建 AttentionMetadataBuilder；
5. 汇总所有 builder 的 CUDA graph support；
6. 解析最终 cudagraph mode。
```

这说明 metadata builder 不是每一层都单独创建，而是按 group 复用。

原因是：

```text
同一个 KV cache group 内，多层 attention 往往共享 block table、slot mapping 和 metadata 结构；
复用 builder / metadata 可以减少构造开销。
```

### 5.2 execute_model 阶段主链路

源码入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4047`

构造 attention metadata 所处位置：

```text
GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
  → _compute_cascade_attn_prefix_lens()
  → _determine_batch_execution_and_padding()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context()
  → _model_forward()
```

其中 attention 相关最关键的是三步：

```text
_prepare_inputs()
  准备 query_start_loc / positions / seq_lens，并触发 block table 计算 slot mapping。

_get_slot_mappings()
  生成 by-kv-cache-group 和 by-layer 两种 slot mapping。

_build_attention_metadata()
  构造 CommonAttentionMetadata，并交给各 backend builder 生成最终 metadata。
```

### 5.3 `_prepare_inputs()` 先把 request 状态变成 token 级张量

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1889`

它会做：

```text
1. commit block table；
2. 根据每个 request 的 num_scheduled_tokens 生成 req_indices；
3. 计算 query_start_loc；
4. 用 num_computed_tokens + query_pos 计算 positions；
5. 更新 seq_lens / optimistic seq_lens；
6. 计算 discard_request_mask；
7. 调用 block_table.compute_slot_mapping()；
8. 根据普通 decode 或 spec decode 生成 logits_indices / SpecDecodeMetadata。
```

可以把它理解成：

```text
把“请求级状态”压平成“本轮 token 级输入”。
```

### 5.4 `_get_slot_mappings()` 同时服务 metadata 和 forward context

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3963`

它返回两种 mapping：

```text
slot_mappings_by_gid：
  dict[int, Tensor]
  按 KV cache group 组织，交给 attention metadata builder。

slot_mappings_by_layer：
  dict[str, Tensor] 或 microbatch list
  按 layer name 组织，交给 ForwardContext / attention layer。
```

为什么要两份？

```text
metadata 构造按 KV cache group 更自然；
模型 forward 时按 layer name 查 metadata / KV cache 更自然。
```

padding token 的 slot mapping 会填 `-1`，让 KV cache update 跳过无效位置。

### 5.5 `_build_attention_metadata()` 是核心翻译层

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2216`

它的主要流程：

```text
1. 计算 padding 后的 token/request 数；
2. 取每个 KV cache group 的 block_table_tensor；
3. padding request 的 block table 填 NULL_BLOCK_ID；
4. 构造 CommonAttentionMetadata；
5. 补充 DCP / encoder / multimodal / KV sharing fast prefill 相关字段；
6. 遍历 KV cache group；
7. 遍历 group 内 AttentionGroup；
8. 调用 builder.build() / build_for_cudagraph_capture() / update_block_table()；
9. 生成 per-layer attn_metadata；
10. spec decode 场景保存 drafter 需要的 common metadata；
11. microbatch 场景切分 metadata；
12. padding 场景下给 drafter 使用 unpadded metadata。
```

最终产出：

```text
attn_metadata：
  dict[layer_name, backend_specific_metadata]

spec_decode_common_attn_metadata：
  drafter / speculative decoding 需要的 CommonAttentionMetadata
```

---

## 6. prefill / decode / mixed batch / spec decode 如何体现在 metadata 中

### 6.1 vLLM V1 不是用单个 mode 标记所有情况

vLLM V1 的 attention metadata 通常不是简单写一个字段：

```text
mode = prefill 或 decode
```

而是通过这些字段共同表达：

```text
num_scheduled_tokens
query_start_loc
max_query_len
seq_lens
num_computed_tokens
positions
is_prefilling
logits_indices
discard_request_mask
scheduled_spec_decode_tokens
```

因此 prefill、decode、chunked prefill、mixed batch 可能出现在同一个 batch 中。

### 6.2 decode

普通 decode 的典型特征：

```text
每个 request 本轮通常调度 1 个 token；
query_start_loc 差分基本都是 1；
max_query_len = 1；
seq_lens = num_computed_tokens + 1；
logits_indices = 每个 request 当前 token 的位置；
slot mapping 指向新 token 应写入的 KV slot。
```

如果启用 speculative decoding，则每个 request 可能是：

```text
1 个真实 token + N 个 draft token
```

此时 uniform decode query length 变为：

```text
1 + num_spec_tokens
```

### 6.3 prefill

prefill 的典型特征：

```text
某个 request 本轮调度多个 prompt token；
max_query_len 可能大于 1；
positions 从 num_computed_tokens 开始连续增长；
slot mapping 会覆盖这一段 prompt token 写入 KV cache 的位置；
通常只需要最后位置的 logits 用于采样。
```

如果请求 prompt logprobs，则可能需要额外位置的 logits，但这属于 logits / sampling 阶段的需求。

### 6.4 chunked prefill

chunked prefill 是“长 prompt 被拆成多个 step 计算”。

它的特征：

```text
num_computed_tokens < num_prompt_tokens；
本轮只处理 prompt 的一个 chunk；
positions 从已计算位置继续；
seq_lens 只推进到当前 chunk 末尾；
非最后 chunk 产生的 sampled token 可能会被 discard_request_mask 忽略。
```

所以 chunked prefill 不是一种独立 backend，而是通过 `num_computed_tokens`、`num_scheduled_tokens`、`discard_request_mask` 和 logits selection 共同表达。

### 6.5 mixed batch

vLLM serving 中常见 mixed batch：

```text
request A：decode 1 token
request B：prefill 128 tokens
request C：chunked prefill 64 tokens
request D：spec decode 1 + N tokens
```

这些 request 会被压平成一个 token batch。

关键边界靠 `query_start_loc` 表达：

```text
query_start_loc = [0, len(A), len(A)+len(B), ...]
```

attention backend 通过 `query_start_loc`、`seq_lens`、`block_table`、`slot_mapping` 分清每个 request。

### 6.6 spec decode

spec decode 会影响两类 metadata：

```text
1. attention metadata：
   本轮 query 可能包含多个 draft token，seq_lens / positions / slot_mapping 都要覆盖这些位置。

2. sampling metadata：
   需要知道哪些 logits 对应 draft token 验证，哪些对应 bonus token。
```

`_prepare_inputs()` 会构造 `SpecDecodeMetadata`，包括：

```text
draft_token_ids
num_draft_tokens
cu_num_draft_tokens
cu_num_sampled_tokens
target_logits_indices
bonus_logits_indices
logits_indices
```

`_build_attention_metadata()` 还会给 drafter 保存一份 `spec_decode_common_attn_metadata`。

---

## 7. block table / slot mapping 如何连接 paged KV cache

### 7.1 三层结构

Worker / ModelRunner 侧的 KV cache 使用可以分成三层：

```text
1. 底层 KV cache tensor
   Worker 初始化时分配的 GPU KV cache 物理内存。

2. InputBatch.block_table
   request index → block ids。

3. slot mapping / attention metadata
   token → 具体 KV cache slot。
```

主链路：

```text
Scheduler 分配 block ids
  → Worker 把 block ids 写入 InputBatch.block_table
  → _prepare_inputs() 计算 positions / query_start_loc
  → block_table.compute_slot_mapping()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → AttentionImpl 使用 block table 读历史 KV，用 slot mapping 写当前 KV
```

### 7.2 block table 是请求级映射

`block table` 记录每个 request 拥有哪些 KV blocks。

语义：

```text
request index
  → logical token block index
  → physical KV block id
```

它用于告诉 attention kernel：

```text
当前 request 的历史上下文 KV 分布在哪些 physical blocks。
```

### 7.3 slot mapping 是 token 级映射

`slot mapping` 记录本轮每个 token 的 K/V 要写到 KV cache 的哪个 slot。

语义：

```text
当前 batch 第 i 个 token
  → request index
  → token position
  → block id + block 内 offset
  → KV cache slot
```

它用于告诉 KV update：

```text
当前算出来的 key/value 应该写入哪个位置。
```

### 7.4 两者不是一回事

```text
block table：
  request → block ids
  偏历史上下文读取。

slot mapping：
  token → KV cache slot
  偏当前 token 写入。
```

Attention backend 同时需要两者：

```text
读历史 KV：看 block table + seq_lens。
写当前 KV：看 slot mapping。
```

---

## 8. Attention forward 如何读写 KV cache

### 8.1 forward 外层先设置 context

`GPUModelRunner.execute_model()` 在 `_model_forward()` 前设置：

```text
set_forward_context(
  attn_metadata,
  vllm_config,
  num_tokens,
  num_tokens_across_dp,
  cudagraph_runtime_mode,
  batch_descriptor,
  ubatch_slices,
  slot_mapping,
  skip_compiled,
)
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4305`

这样模型内部 attention layer 可以通过当前 context 拿到 metadata。

### 8.2 Attention layer 从 context 取当前层信息

模型执行到某层 attention 时，会通过 layer name 取：

```text
当前层 backend-specific attn_metadata；
当前层 Attention module；
当前层 KV cache tensor；
当前层 slot mapping。
```

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:670`

### 8.3 写 KV cache

对普通 attention，当前 token 的 key/value 需要写入 paged KV cache。

写入依据是：

```text
key tensor / value tensor
  + kv_cache tensor
  + slot_mapping
  + kv_cache_dtype / scale
```

多数 V1 attention backend（如 FlashAttention、FlashInfer、Triton、Flex、CPU、ROCm 等）声明 `forward_includes_kv_cache_update=False`，因此 `Attention.forward()` 会先通过 `unified_kv_cache_update()` 写 KV cache，再调用 backend forward。如果某个 backend 声明 `forward_includes_kv_cache_update=True`，KV 写入才会留在 impl forward 内部完成。

FlashAttention 的 `do_kv_cache_update()` 会调用 `reshape_and_cache_flash()`，按照 `slot_mapping` 写 KV cache。

源码位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:927`

### 8.4 读 KV cache

attention kernel 读取历史 KV 时依赖：

```text
kv_cache tensor
block_table
seq_lens
query_start_loc
max_seq_len
max_query_len
```

FlashAttention 非 cascade 路径会把这些 metadata 传给 varlen flash attention kernel。

对于 paged KV cache，kernel 不是读取一段连续 `[0:seq_len]` 内存，而是通过 block table 找到每个逻辑 block 对应的物理 block。

### 8.5 输出 hidden states

backend impl forward 输出 attention 后的 hidden states，然后模型继续执行 MLP / 后续层。

最后一个 PP rank 会再基于 `logits_indices` 计算 logits：

```text
hidden_states[logits_indices]
  → model.compute_logits()
  → sample_tokens()
```

所以 attention 的输出不是最终 token，而是模型 hidden states 的一部分。

---

## 9. KV cache layout 和 backend 的关系

### 9.1 backend 决定 KV cache shape

不同 backend 对 KV cache 的形状和 stride 有不同要求。

例如 FlashAttention 常见 KV cache shape：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

这里的 `2` 表示 key / value。

MLA、FlashInfer、Triton、CPU、ROCm 等 backend 可能要求不同 layout。

因此 `AttentionBackend` 必须提供：

```text
get_kv_cache_shape()
get_kv_cache_stride_order()
get_kv_cache_block_dim()
```

### 9.2 layout 会影响 Worker 初始化 KV cache

Worker 初始化 KV cache 时，不是创建一个完全 backend 无关的 tensor，而是根据各层 backend / KVCacheSpec 决定 shape 和 layout。

这也是为什么 backend 选择发生在模型 attention layer 初始化阶段，并会参与后续 KV cache 初始化。

### 9.3 uniform KV cache / KV connector 对 layout 有额外要求

如果启用 KV connector，并且 connector 希望跨层使用 uniform KV cache，则要求：

```text
- 启用 KV transfer group；
- connector 偏好 cross-layer blocks；
- 只有一个 KV cache group，且其中只有一个 attention group；
- KV cache spec 是 AttentionSpec；
- backend 的 get_kv_cache_stride_order(include_num_layers_dimension=True) 能表达额外 layer 维度；
- 返回的 stride order 不是 identity 布局，能支持跨层连续组织。
```

源码位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:115`

这说明 KV cache layout 不只是 kernel 层细节，也会影响远端 KV 传输能力。

---

## 10. cascade attention / prefix cache 如何影响 attention

### 10.1 prefix cache 命中先发生在 Scheduler / KVCacheManager

prefix cache 命中不是 attention backend 自己发现的。

大致链路：

```text
KVCacheManager 发现 prefix blocks 可复用
  → Scheduler 更新 num_computed_tokens / block_ids
  → SchedulerOutput 携带这些状态给 Worker
  → Worker 更新 InputBatch
  → attention metadata 中体现为 block table + num_computed_tokens + positions
```

Worker / attention 侧只消费结果。

### 10.2 cascade attention 利用公共 prefix

cascade attention 的目标是减少多个 request 共享 prefix 时的重复 attention 计算。

基本思路：

```text
多个 request 共享 common prefix blocks
  → prefix 部分单独算一段 attention state
  → suffix 部分正常 causal attention
  → merge prefix/suffix attention states
```

是否启用由 backend builder 判断，不是所有 backend / 所有场景都可用。

### 10.3 common_prefix_len 的来源

`SchedulerOutput` 中有 `num_common_prefix_blocks`。

`GPUModelRunner` 会根据：

```text
num_common_prefix_blocks * block_size
min(num_computed_tokens)
sliding window / local attention 限制
backend builder 是否支持 cascade
```

计算每个 KV cache group 的 `common_prefix_len`。

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2519`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2557`

这里得到的是 cascade attention 可用的 common prefix length，不一定等于请求真实共享 prefix。它还会被 `min(num_computed_tokens)` 截断，并按 block size 向下取整；如果 ubatching 已启用、builder 判断当前 batch 不适合 cascade，或 backend 不支持该路径，最终会退化为 0。

### 10.4 cascade 会影响 metadata 和 CUDA graph

cascade attention 会让 backend metadata 中多出 prefix/suffix 相关字段，例如：

```text
use_cascade
common_prefix_len
cu_prefix_query_lens
prefix_kv_lens
suffix_kv_lens
prefix_scheduler_metadata
```

FlashAttention builder 会构造这些字段，FlashAttention impl 会调用 `cascade_attention()`。

源码位置：

- `code/vllm/vllm/v1/attention/backends/flash_attn.py:489`
- `code/vllm/vllm/v1/attention/backends/flash_attn.py:898`

因为 cascade 路径形状和普通路径差异较大，runtime 中通常会禁用 full CUDA graph。

---

## 11. KV connector hook 如何挂到 attention layer 边界

### 11.1 ModelRunner 侧包住整个 forward

KV connector 在 ModelRunner 侧的入口是 `KVConnectorModelRunnerMixin`。

源码位置：`code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`

forward 前后大致流程：

```text
maybe_get_kv_connector_output()
  → bind_connector_metadata(scheduler_output.kv_connector_metadata)
  → start_load_kv(get_forward_context())
  → model forward
  → wait_for_save()
  → get_finished(finished_req_ids)
  → get invalid block ids / stats / events
  → clear_connector_metadata()
```

在 `GPUModelRunner.execute_model()` 中，KV connector context 和 `set_forward_context()` 一起包住 `_model_forward()`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4305`

### 11.2 Attention layer 侧做逐层 wait/save

源码位置：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:15`

`maybe_transfer_kv_layer` 装饰 attention custom op。

每层 attention 执行时：

```text
attention layer 前：
  connector.wait_for_layer_load(layer_name)

attention layer 后：
  connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

这样 KV connector 可以做到逐层 KV load / save，而不是只能在整个模型 forward 前后做粗粒度操作。

### 11.3 0-token step 也可能需要 connector

如果本轮没有 token 需要 forward，但外部 KV 传输仍要推进，可能走：

```text
kv_connector_no_forward()
```

这说明：

```text
KV connector 的生命周期不完全等同于模型 forward 生命周期；
有些 step 只是在推进 KV transfer 状态。
```

---

## 12. CUDA graph / torch.compile 对 attention 路径的约束

### 12.1 backend builder 声明 CUDA graph 支持级别

源码位置：`code/vllm/vllm/v1/attention/backend.py:548`

`AttentionCGSupport` 常见级别：

```text
ALWAYS：
  该 backend 对 CUDA graph 支持较完整。

UNIFORM_BATCH：
  需要 batch 结构比较统一。

UNIFORM_SINGLE_TOKEN_DECODE：
  只适合单 token decode 这类固定形状。

NEVER：
  不支持 CUDA graph。
```

每个 `AttentionMetadataBuilder` 通过 `get_cudagraph_support()` 暴露自己的能力。

### 12.2 ModelRunner 汇总所有 builder 的限制

`GPUModelRunner` 会遍历所有 attention metadata builder，取最保守的支持级别，再解析最终 cudagraph mode 和 capture sizes。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6880`

这意味着：

```text
只要某个 attention backend 不支持某种 graph 模式，整个模型执行就要降级到兼容模式。
```

### 12.3 runtime 根据 batch descriptor 选择 FULL / PIECEWISE / NONE

`_determine_batch_execution_and_padding()` 会判断：

```text
- 当前是否 uniform decode；
- token 数是否命中 capture size；
- 是否有 encoder input；
- 是否有 cascade attention；
- 是否需要 DP padding；
- 是否需要 microbatch；
- LoRA / compile config 是否允许 graph replay。
```

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3813`

最终通过 `CudagraphDispatcher` 选择：

```text
FULL
PIECEWISE
NONE
```

### 12.4 CUDA graph capture 需要专门构造 metadata

capture / warmup 时会跑 dummy input。

注意点：

```text
- dummy run 也要构造 attention metadata；
- slot mapping 通常填 -1，避免真实写 KV cache；
- capture 时调用 build_for_cudagraph_capture()；
- max_seq_len 可能用 max_model_len，避免 kernel 选择在 replay 时不一致。
```

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5660`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5869`

### 12.5 torch.compile 通过 opaque attention op 隔离复杂逻辑

源码位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:504`

Attention layer 通常会把 attention 包成：

```text
torch.ops.vllm.unified_attention_with_output
```

这样做的意义是：

```text
- 对 torch.compile 来说，attention 是一个 opaque custom op；
- 编译器不需要理解复杂的 paged KV / block table / backend 分支；
- attention 内部仍可按当前 backend 调 FlashAttention / FlashInfer / Triton 等 kernel；
- KV connector hook 也可以包在 custom op 边界上。
```

---

## 13. 主链路完整展开

把 attention 放回一次 `execute_model()`，完整关系是：

```text
EngineCore.step()
  → Scheduler.schedule()
      → 决定 scheduled requests
      → 分配 / 复用 KV blocks
      → 输出 num_common_prefix_blocks / kv_connector_metadata

  → Executor.execute_model()
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
      → _update_states()
          → 更新 InputBatch
          → 写入 request block_ids

      → _prepare_inputs()
          → commit block table
          → 计算 req_indices / positions
          → 计算 query_start_loc / seq_lens
          → compute_slot_mapping()
          → 计算 logits_indices / spec metadata

      → _compute_cascade_attn_prefix_lens()
          → 根据 num_common_prefix_blocks 计算 common_prefix_len

      → _determine_batch_execution_and_padding()
          → 决定 cudagraph mode / padding / ubatch

      → _get_slot_mappings()
          → slot_mappings_by_gid
          → slot_mappings_by_layer

      → _build_attention_metadata()
          → CommonAttentionMetadata
          → backend-specific AttentionMetadata
          → per-layer attn_metadata

      → _preprocess()
          → input_ids / inputs_embeds / positions / model_kwargs

      → set_forward_context(attn_metadata, slot_mapping, batch_descriptor, ...)
      → maybe_get_kv_connector_output()
      → _model_forward()
          → model layers
          → Attention.forward()
          → unified_attention_with_output()
          → get_attention_context(layer_name)
          → KV connector wait_for_layer_load()
          → KV cache update by slot_mapping
          → backend impl forward reads block_table / kv_cache
          → KV connector save_kv_layer()

      → hidden_states
      → compute_logits(hidden_states[logits_indices])
      → sample_tokens()
      → ModelRunnerOutput
  → Scheduler.update_from_output()
```

这条链路里，attention 的核心输入可以压缩成：

```text
query / key / value tensor
  + KV cache tensor
  + block table
  + slot mapping
  + seq_lens
  + query_start_loc
  + backend-specific metadata
```

---

## 14. 关键数据结构关系

### 14.1 `AttentionBackend`

后端能力与 KV cache layout 的声明者。

```text
backend name
impl class
metadata builder class
KV cache shape / stride / block dim
capability validation
```

### 14.2 `AttentionSelectorConfig`

backend 选择时的归一化输入。

```text
head_size / dtype / kv_cache_dtype / block_size
is_mla / use_sparse / has_sink / use_mm_prefix
attention_type / use_kv_connector
```

### 14.3 `AttentionGroup`

ModelRunner 初始化阶段把多个 layer 聚合成的 attention metadata 构造单元。

```text
backend class + KVCacheSpec + num_heads_q + layer_names
```

### 14.4 `CommonAttentionMetadata`

公共 batch 描述。

```text
query_start_loc
seq_lens
max_query_len
max_seq_len
block_table_tensor
slot_mapping
positions
is_prefilling
```

### 14.5 backend-specific `AttentionMetadata`

具体 backend 的运行时 metadata。

例如 FlashAttentionMetadata 会额外保存：

```text
scheduler_metadata
prefix_scheduler_metadata
use_cascade
common_prefix_len
prefix_kv_lens / suffix_kv_lens
mm_prefix_range_tensor
```

### 14.6 `ForwardContext`

forward 生命周期内的隐式参数区。

```text
per-layer attn_metadata
per-layer slot_mapping
no_compile_layers / static_forward_context 中注册的静态对象
cudagraph_runtime_mode
batch_descriptor
ubatch_slices
```

### 14.7 `InputBatch.block_table`

请求级 KV block 映射。

```text
req_index → block ids
```

### 14.8 `slot_mapping`

token 级 KV slot 映射。

```text
token_index → kv cache slot
```

---

## 15. 和已有专题的关系

Attention 子系统和以下专题强相关：

```text
../executor_worker_model_runner/06_prepare_inputs_and_attention_metadata.md
  解释 ModelRunner 如何准备 input_ids / positions / attention metadata。

../executor_worker_model_runner/07_model_forward_and_logits.md
  解释 set_forward_context、_model_forward、hidden_states / logits 的位置。

../executor_worker_model_runner/09_worker_kv_cache_interaction.md
  解释 Worker / ModelRunner 如何使用 block table / slot mapping / KV connector。

../kv_cache_transfer/07_worker_kv_connector_flow.md
  解释 KV connector 如何在 Worker / ModelRunner 中 load/save KV。

../scheduler/05_prefix_and_external_kv_hits.md
  解释 prefix cache / external KV 命中如何先在 Scheduler 侧体现。

../scheduler/06_kv_block_allocation_and_preemption.md
  解释 KV block 分配、抢占和释放的资源账本。
```

可以按下面方式串起来：

```text
Scheduler 文档：
  解释 block 为什么这样分配。

executor_worker_model_runner 文档：
  解释 SchedulerOutput 如何进入 Worker 并变成 forward。

attention 文档：
  解释 forward 内 attention metadata / KV cache / backend kernel 如何工作。

kv_cache_transfer 文档：
  解释外部 KV load/save 如何插入 attention 层边界。
```

---

## 16. 容易混淆的点

### 16.1 AttentionBackend 是 kernel 吗？

不是。

`AttentionBackend` 是能力描述和工厂入口；真正执行的是 `AttentionImpl` 及其调用的底层 kernel。

### 16.2 CommonAttentionMetadata 是最终 kernel 参数吗？

不是。

它是公共描述，最终还要经 `AttentionMetadataBuilder` 转成 backend-specific metadata。

### 16.3 block table 和 slot mapping 是一回事吗？

不是。

```text
block table：request → block ids
slot mapping：token → KV cache slot
```

### 16.4 prefill / decode 是两个完全不同 batch 吗？

不是。

vLLM 可以 mixed batch，prefill、decode、chunked prefill、spec decode 通过 `query_start_loc`、`seq_lens`、`positions`、`is_prefilling` 等字段共同表达。

### 16.5 prefix cache 是 attention backend 自己发现的吗？

不是。

prefix cache 命中主要由 Scheduler / KVCacheManager 决定；attention 侧只消费已经分配好的 block table 和 num_computed_tokens。

### 16.6 KV connector 是 forward 后统一处理的吗？

不是。

它既在 ModelRunner forward 外层有生命周期，也在每层 attention 前后有 hook。

### 16.7 CUDA graph 只影响模型外层吗？

不是。

CUDA graph 会影响：

```text
attention metadata shape
padding token / request
slot mapping 是否填 -1
builder 是否走 build_for_cudagraph_capture()
backend 是否支持 graph replay
custom op / compile 边界
```

---

## 17. 推荐阅读路线

### 17.1 快速建立全局印象

```text
attention_overview.md
  → 01_attention_role.md
  → 02_backend_selection.md
  → 06_attention_forward_flow.md
```

### 17.2 按执行链路完整阅读

```text
attention_overview.md
  → 03_attention_metadata_builder.md
  → 04_prefill_decode_metadata.md
  → 05_slot_mapping_and_block_table.md
  → 06_attention_forward_flow.md
  → 07_kv_cache_layout_and_backend.md
```

### 17.3 深入 prefix / cascade / external KV

```text
08_cascade_attention.md
  → 09_attention_and_kv_connector_hooks.md
  → ../scheduler/05_prefix_and_external_kv_hits.md
  → ../kv_cache_transfer/07_worker_kv_connector_flow.md
```

### 17.4 深入 backend 和算法家族

```text
attention_methods/12_flash_attention_family.md
  → attention_methods/13_paged_attention.md
  → attention_methods/15_mla_attention.md
  → attention_methods/17_flashinfer_flashmla_triton_backends.md
  → attention_methods/18_hma_and_kv_cache_layout.md
```

### 17.5 深入 CUDA graph / compile

```text
10_cuda_graph_compile_interaction.md
  → ../executor_worker_model_runner/10_executor_worker_lifecycle.md
  → code/vllm/vllm/v1/cudagraph_dispatcher.py
  → code/vllm/vllm/compilation/
```

---

## 18. 文档定位

```text
attention_overview.md：
  总览主文档，适合快速建立 attention 子系统全局图。

01-10：
  按问题拆开的专题文档，适合逐段精读 attention 主链路源码。

attention_methods/11-18：
  专门解释各种 attention 名词、结构、backend、KV layout 和优化策略。
```

---

## 19. 一句话总结

vLLM V1 attention 子系统的本质，是把 Scheduler 已经决定好的“本轮请求和 KV blocks”，翻译成 attention backend 可执行的 `block table + slot mapping + metadata + KV cache layout`，再通过 ForwardContext 注入模型 forward，让每层 attention backend 正确读写 paged KV cache。

最核心的主线是：

```text
SchedulerOutput
  → InputBatch / block table
  → positions / query_start_loc / seq_lens
  → slot mapping
  → CommonAttentionMetadata
  → backend-specific AttentionMetadata
  → ForwardContext
  → Attention.forward()
  → AttentionImpl.forward()
  → paged KV cache read/write
  → hidden states
```

如果只记住一句话：

```text
block table 告诉 attention “历史 KV 在哪些 block”，slot mapping 告诉 attention “当前 KV 写到哪里”，metadata builder 告诉 backend “这一批 token 应该怎么被 kernel 看见”。
```
