# 01. Attention 子系统是什么层？

源码位置：

- `code/vllm/vllm/v1/attention/backend.py`
- `code/vllm/vllm/v1/attention/selector.py`
- `code/vllm/vllm/v1/attention/backends/`
- `code/vllm/vllm/model_executor/layers/attention/attention.py`
- `code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py`
- `code/vllm/vllm/model_executor/layers/attention_layer_base.py`
- `code/vllm/vllm/forward_context.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/kv_cache_interface.py`

本文用于梳理 Attention 子系统在 vLLM V1 中的职责边界：它处在 ModelRunner 和底层 kernel 之间，既不是 Scheduler，也不是单纯的 FlashAttention kernel，而是一组负责 **backend 选择、KV cache layout 约定、attention metadata 构造、attention layer forward、paged KV cache 读写、CUDA graph / compile 适配、KV connector hook** 的执行子系统。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录的文档风格，本篇按“先定角色，再走主链路，再拆关键组件，最后总结边界”的方式梳理 Attention 子系统。

要回答的问题分成 10 组：

```text
1. Attention 子系统处在 vLLM V1 哪一层？
2. Attention layer、AttentionBackend、AttentionMetadataBuilder、AttentionImplBase 分别是什么？
3. Attention 子系统和 Scheduler / KVCacheManager / ModelRunner 的边界是什么？
4. Attention 子系统如何参与 KV cache spec / KV cache layout？
5. backend selection 发生在哪里？由哪些条件决定？
6. Attention metadata 从哪里来？为什么 backend 不直接读 InputBatch？
7. Attention.forward() 真正做了什么？
8. paged KV cache update 和 attention kernel 调用如何串起来？
9. KV connector hook、CUDA graph、torch.compile 如何挂在 attention 层？
10. 各种 attention 名词在本篇里应该如何定位？
```

阅读顺序建议：

```text
attention_overview.md
  → 01_attention_role.md
  → 02_backend_selection.md
  → 03_attention_metadata_builder.md
  → 05_slot_mapping_and_block_table.md
  → 06_attention_forward_flow.md
  → 07_kv_cache_layout_and_backend.md
  → 09_attention_and_kv_connector_hooks.md
```

本篇重点讲 Attention 子系统的“定位和边界”，不会把每个 backend 的 kernel 细节展开。各种 attention 名词和 backend 家族后续放在：

```text
attention_methods/11_attention_variants_overview.md
attention_methods/12_flash_attention_family.md
attention_methods/13_paged_attention.md
attention_methods/14_mha_mqa_gqa.md
attention_methods/15_mla_attention.md
attention_methods/17_flashinfer_flashmla_triton_backends.md
```

---

## 1. 一句话回答

Attention 子系统是 vLLM V1 执行层中连接 **ModelRunner batch 状态**、**paged KV cache**、**模型 attention layer** 和 **底层 attention backend / kernel** 的中间层。

它负责：

```text
1. 根据模型结构、平台能力、dtype、KV cache dtype、block size、MLA / sliding window / KV connector 等条件选择 attention backend；
2. 由 Attention layer 报告每层需要的 KVCacheSpec；
3. 由 backend 定义 KV cache tensor shape / stride / layout；
4. 在 ModelRunner 侧按 KV cache group / attention group 创建 metadata builders；
5. 把 query_start_loc、seq_lens、block_table、slot_mapping、positions、cascade prefix 等公共信息构造成 backend metadata；
6. 在模型 forward 时通过 ForwardContext 把 attention metadata 和 slot mapping 暴露给每个 attention layer；
7. 在 Attention.forward() 中 reshape Q/K/V，必要时更新 KV cache，再调用 backend impl forward；
8. 支持 prefill、decode、mixed batch、spec decode、sliding window、cascade attention、MLA、encoder-only / cross attention 等不同 attention 语义；
9. 在 attention layer 边界触发 KV connector 的 wait_for_layer_load() / save_kv_layer()；
10. 配合 CUDA graph / torch.compile，把 attention 调用包装成统一 op 或直接调用 backend。
```

它不负责：

```text
1. 不负责接收用户请求；
2. 不负责维护 waiting / running 队列；
3. 不负责决定本轮调度哪些请求；
4. 不负责分配逻辑 KV blocks；
5. 不负责 prefix cache 命中决策；
6. 不负责采样 token；
7. 不负责构造 RequestOutput；
8. 不负责跨 Worker RPC。
```

一句话压缩：

```text
Scheduler 决定“哪些 token 要跑”，ModelRunner 整理“这些 token 怎么喂模型”，Attention 子系统决定“这些 token 如何在当前 backend 和 paged KV cache 上做 attention”。
```

---

## 2. Attention 子系统在主链路中的位置

从完整执行链路看，Attention 子系统处在 `GPUModelRunner.execute_model()` 内部。

上游主线是：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
```

进入 ModelRunner 后，attention 相关链路是：

```text
GPUModelRunner._update_states()
  → GPUModelRunner._prepare_inputs()
  → GPUModelRunner._get_slot_mappings()
  → GPUModelRunner._build_attention_metadata()
  → GPUModelRunner._preprocess()
  → set_forward_context(attn_metadata, slot_mapping, ...)
  → GPUModelRunner._model_forward()
  → model attention layer
  → Attention.forward()
  → unified_kv_cache_update()  # 某些 backend
  → unified_attention_with_output()
  → AttentionImpl.forward()
```

对应源码入口：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4244`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4255`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`
- `code/vllm/vllm/model_executor/layers/attention/attention.py:438`

所以 Attention 子系统不是一个独立的外层服务，而是：

```text
ModelRunner 执行模型 forward 时动态使用的一组机制。
```

---

## 3. Attention 子系统的四层对象

Attention 子系统里最容易混淆的是几个名字：

```text
Attention layer
AttentionBackend
AttentionMetadataBuilder
AttentionImplBase / AttentionImpl
```

它们不是同一层。

### 3.1 Attention layer：模型里的 attention 模块

`Attention` 定义在：

```text
code/vllm/vllm/model_executor/layers/attention/attention.py:178
```

源码注释直接说明它做三件事：

```python
"""
Attention layer.

This class takes query, key, and value tensors as input.
The class does the following:

1. Store the input key and value tensors in the KV cache.
2. Perform (multi-head/multi-query/grouped-query) attention.
3. Return the output tensor.
"""
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:178`

它是模型结构的一部分，每个 transformer block 通常会持有一个 attention layer。

它保存：

```text
- num_heads
- num_kv_heads
- head_size
- head_size_v
- sliding_window
- kv_cache_dtype
- attn_backend
- impl
- kv_cache
- layer_name
- kv_sharing_target_layer_name
```

注意：`Attention` 自己不决定本轮 batch 有哪些请求，也不构造 `SchedulerOutput`。它只在模型 forward 到这一层时，使用当前 `ForwardContext` 中的 metadata 执行当前层 attention。

### 3.2 AttentionBackend：backend 能力和工厂

`AttentionBackend` 定义在：

```text
code/vllm/vllm/v1/attention/backend.py:55
```

它是抽象类，负责声明：

```text
- backend 名字；
- 使用哪个 AttentionImpl；
- 使用哪个 AttentionMetadataBuilder；
- KV cache tensor shape；
- KV cache stride order / layout；
- 支持哪些 dtype、KV cache dtype、head size、block size；
- 是否支持 MLA、sink、sparse、mm prefix、non-causal、batch invariance、KV connector；
- 是否支持某种 AttentionType；
- 是否要求特定 KV cache layout。
```

关键抽象方法：

```python
get_name()
get_impl_cls()
get_builder_cls()
get_kv_cache_shape(...)
```

位置：`code/vllm/vllm/v1/attention/backend.py:72` 到 `code/vllm/vllm/v1/attention/backend.py:96`

所以 backend 不是“某个 attention 算法名”的泛称，而是 vLLM 内部的执行后端契约：

```text
backend = metadata builder + impl + KV cache layout + capability checks。
```

### 3.3 AttentionMetadataBuilder：把公共 batch 信息翻译成 backend metadata

`AttentionMetadataBuilder` 定义在：

```text
code/vllm/vllm/v1/attention/backend.py:533
```

它的核心方法是：

```python
def build(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    fast_build: bool = False,
) -> M:
```

位置：`code/vllm/vllm/v1/attention/backend.py:599`

它负责把 `CommonAttentionMetadata` 变成具体 backend 的 metadata。

为什么需要 builder？

```text
因为不同 backend 要的 metadata 不一样：
FlashInfer、FlashAttention、Triton、CPU、MLA、FlexAttention 等 backend 对 seq_lens、page table、block table、workspace、kernel 参数的组织方式都可能不同。
```

ModelRunner 不应该直接知道每个 backend 的私有字段，所以它只准备公共信息，然后交给 builder。

### 3.4 AttentionImplBase / AttentionImpl：真正执行 backend forward

`AttentionImplBase` 定义在：

```text
code/vllm/vllm/v1/attention/backend.py:702
```

标准 attention impl 是：

```python
class AttentionImpl(AttentionImplBase[T], Generic[T]):
```

位置：`code/vllm/vllm/v1/attention/backend.py:780`

核心 forward 抽象是：

```python
def forward(
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output,
    ...
) -> torch.Tensor:
```

位置：`code/vllm/vllm/v1/attention/backend.py:806`

MLA 则有单独抽象：

```text
MLAAttentionImpl.forward_mha()
MLAAttentionImpl.forward_mqa()
```

位置：`code/vllm/vllm/v1/attention/backend.py:863`

所以执行分层是：

```text
Attention layer：模型模块外壳
AttentionBackend：选择 impl / builder / layout 的 backend 类
AttentionMetadataBuilder：构造运行时 metadata
AttentionImpl：真正调用 backend kernel / op 的实现
```

---

## 4. Attention 子系统和 ModelRunner 的边界

ModelRunner 负责“把本轮 SchedulerOutput 翻译成模型 forward 所需信息”。

Attention 子系统负责“用这些信息执行 attention”。

### 4.1 ModelRunner 负责准备公共输入

`GPUModelRunner.execute_model()` 中，会先准备 slot mapping：

```python
slot_mappings_by_group, slot_mappings = self._get_slot_mappings(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4244`

然后构造 attention metadata：

```python
attn_metadata, spec_decode_common_attn_metadata = (
    self._build_attention_metadata(...)
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4255`

最后用 forward context 包住模型 forward：

```python
set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    ubatch_slices=ubatch_slices_padded,
    slot_mapping=slot_mappings,
    skip_compiled=has_encoder_input,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`

也就是说，ModelRunner 准备：

```text
- input_ids / inputs_embeds / positions；
- query_start_loc；
- seq_lens；
- block_table_tensor；
- slot_mapping；
- cascade prefix lengths；
- cudagraph / ubatch / padding 信息；
- backend-specific attention metadata。
```

### 4.2 Attention layer 负责消费 forward context

`Attention.forward()` 的注释说明：

```text
Attention metadata is set using a context manager in the model runner's execute_model method.
It is accessed via forward context using get_forward_context().attn_metadata.
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:452`

所以 attention layer 的运行时参数不是全靠函数显式传入，而是通过：

```text
ForwardContext
```

获取。

边界一句话：

```text
ModelRunner 准备 attention 需要看的“地图”，Attention layer 按这张地图真正读写 KV 并执行 kernel。
```

---

## 5. Attention 子系统和 KVCacheManager / BlockPool 的边界

这是理解 vLLM attention 的关键。

### 5.1 KVCacheManager / BlockPool 管逻辑 block

Scheduler 侧的 `KVCacheManager` 和 `BlockPool` 负责：

```text
- 哪些请求持有哪些 KV blocks；
- prefix cache 命中了哪些 blocks；
- 本轮要分配哪些新 blocks；
- preemption 时释放或重分配哪些 blocks；
- 请求结束时释放 blocks。
```

这些逻辑不在 attention backend 中。

### 5.2 Attention 子系统管如何用这些 blocks 做 attention

Attention 子系统看到的是 ModelRunner 整理后的：

```text
- block_table_tensor；
- slot_mapping；
- seq_lens；
- query_start_loc；
- KV cache tensor；
- backend metadata。
```

`CommonAttentionMetadata` 中直接保存：

```python
block_table_tensor: torch.Tensor
slot_mapping: torch.Tensor
```

位置：`code/vllm/vllm/v1/attention/backend.py:387`

它们是逻辑 block 分配进入 attention backend 的桥。

### 5.3 Attention layer 报告自己需要什么 KV cache

每个 `Attention` layer 可以通过：

```python
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:567`

报告它需要的 KV cache spec。

普通 full attention 返回：

```text
FullAttentionSpec
```

sliding window 返回：

```text
SlidingWindowSpec
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:573`

ModelRunner 会收集这些 spec：

```python
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7459`

然后 EngineCore / KV cache manager 才能知道每层 KV cache 需要多少显存、如何分组。

因此边界是：

```text
Attention layer 声明“我需要什么形状的 KV cache”；
KVCacheManager 管“blocks 怎么分配”；
ModelRunner 管“这些 blocks 怎么变成 slot mapping”；
Attention backend 管“如何按这些 metadata 读写 KV cache”。
```

---

## 6. Attention backend 如何被选择

backend selection 的入口是：

```python
def get_attn_backend(...)
```

位置：`code/vllm/vllm/v1/attention/selector.py:54`

它会构造 `AttentionSelectorConfig`：

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

位置：`code/vllm/vllm/v1/attention/selector.py:21`

然后调用平台层：

```python
attention_cls = current_platform.get_attn_backend_cls(
    backend,
    attn_selector_config=attn_selector_config,
    num_heads=num_heads,
)
```

位置：`code/vllm/vllm/v1/attention/selector.py:121`

这说明 backend 选择不仅取决于模型，还取决于：

```text
- 当前平台；
- 用户指定的 backend；
- dtype / KV cache dtype；
- head_size；
- block_size；
- 是否 MLA；
- 是否 sparse；
- 是否 mm prefix；
- 是否 non-causal；
- 是否启用 KV connector；
- attention type。
```

Attention layer 初始化时会调用这个 selector：

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

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:304`

backend 选好后，Attention layer 立刻创建 impl：

```python
impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(...)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:373`

所以 backend selection 发生在模型 attention layer 初始化阶段，而 metadata builder 创建发生在 ModelRunner 初始化 KV cache 阶段。

---

## 7. AttentionBackend 的能力检查是什么

`AttentionBackend` 不只是返回 impl，它还定义了大量 capability check。

核心字段和方法包括：

```text
supported_dtypes
supported_kv_cache_dtypes
forward_includes_kv_cache_update
get_supported_kernel_block_sizes()
supports_head_size()
supports_dtype()
supports_kv_cache_dtype()
supports_block_size()
is_mla()
supports_sink()
supports_mm_prefix()
is_sparse()
supports_non_causal()
supports_batch_invariance()
supports_kv_connector()
supports_attn_type()
supports_compute_capability()
get_required_kv_cache_layout()
```

位置：`code/vllm/vllm/v1/attention/backend.py:55` 到 `code/vllm/vllm/v1/attention/backend.py:351`

它们的作用是：

```text
在模型真正跑之前，把“这个 backend 能不能跑当前模型 / 当前配置 / 当前平台”判断清楚。
```

例如：

```text
- head_size 不支持 → 不能选该 backend；
- KV cache dtype 不支持 → 不能选该 backend；
- attn_type 不是 decoder 而 backend 不支持 → 不能选；
- 启用了 KV connector 但 backend 不支持 → 不能选；
- backend 要求特定 KV cache layout → selector 会设置 layout。
```

`get_attn_backend()` 中还会处理 required layout：

```python
required_layout = backend.get_required_kv_cache_layout()
if required_layout is not None:
    set_kv_cache_layout(required_layout)
```

位置：`code/vllm/vllm/v1/attention/selector.py:132`

这说明：

```text
backend selection 会反向影响 KV cache layout。
```

---

## 8. Attention 子系统如何参与 KV cache 初始化

KV cache 初始化入口在 ModelRunner：

```python
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig, ...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7303`

其中 attention 相关步骤是：

```text
1. may_add_encoder_only_layers_to_kv_cache_config()
2. maybe_add_kv_sharing_layers_to_kv_cache_groups()
3. initialize_attn_backend(kv_cache_config)
4. prepare_kernel_block_sizes(kv_cache_config, self.attn_groups)
5. initialize_metadata_builders(kv_cache_config, kernel_block_sizes)
6. initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7317` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7340`

### 8.1 initialize_attn_backend() 创建 attention groups

入口：

```python
def initialize_attn_backend(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6736`

它会遍历 KV cache groups，按这些维度去重：

```text
- attention backend class；
- KV cache spec；
- num_heads_q。
```

对应 key：

```python
class AttentionGroupKey(NamedTuple):
    attn_backend: type[AttentionBackend]
    kv_cache_spec: KVCacheSpec
    num_heads_q: int
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6746`

为什么要分 attention group？

```text
同一个 KV cache group 内，不同层可能使用相同 backend / spec / head 配置；
这些层可以共享一份 attention metadata builder 和 metadata。
```

### 8.2 initialize_metadata_builders() 创建 builders

入口：

```python
def initialize_metadata_builders(self, kv_cache_config, kernel_block_sizes)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6843`

它会对每个 KV cache group、每个 attention group 调：

```python
attn_group.create_metadata_builders(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6849`

如果启用了 ubatching，会为每个 ubatch 创建 builder：

```text
num_metadata_builders = num_ubatches
```

否则只创建 1 个。

### 8.3 Attention backend 影响 CUDA graph mode

`initialize_attn_backend()` 里会先检查所有 backend 的 cudagraph 支持：

```python
self._check_and_update_cudagraph_mode(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6830`

它会看每个 backend builder 的：

```python
builder_cls.get_cudagraph_support(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6898`

所以 Attention 子系统还会影响：

```text
这一轮或这个模型能不能用 FULL / PIECEWISE / NONE CUDA graph。
```

---

## 9. Attention metadata 是什么

`CommonAttentionMetadata` 定义在：

```text
code/vllm/vllm/v1/attention/backend.py:361
```

它是“跨 layer / backend 的公共 batch 描述”。

核心字段包括：

```text
query_start_loc
query_start_loc_cpu
seq_lens
num_reqs
num_actual_tokens
max_query_len
max_seq_len
block_table_tensor
slot_mapping
causal
logits_indices_padded
num_logits_indices
encoder_seq_lens
encoder_seq_lens_cpu
dcp_local_seq_lens
positions
is_prefilling
seq_lens_cpu_upper_bound
mm_req_doc_ranges
```

位置：`code/vllm/vllm/v1/attention/backend.py:361` 到 `code/vllm/vllm/v1/attention/backend.py:425`

它表达的是：

```text
当前 batch 中每个 request 的 query 在哪里、上下文有多长、KV blocks 在哪里、本轮 token 写到哪些 slots、是否 causal、是否 prefill、是否有 encoder / multimodal / DCP 等特殊信息。
```

为什么 backend 不直接读 `InputBatch`？

```text
InputBatch 是 ModelRunner 的 worker-local 持久状态；
backend 只应该看到本轮 forward 需要的、已经整理好的、形状稳定的 metadata；
不同 backend 有不同私有结构，所以需要 builder 做转换。
```

边界可以记成：

```text
InputBatch = worker 侧状态账本；
CommonAttentionMetadata = 本轮 attention 公共运行参数；
backend metadata = 某个 kernel 可直接消费的私有参数。
```

---

## 10. _build_attention_metadata() 如何把公共信息交给 backend

`GPUModelRunner._build_attention_metadata()` 是 ModelRunner 和 AttentionBackend 的桥。

入口：

```python
def _build_attention_metadata(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2208`

它的核心步骤是：

```text
1. 如果没有 KV cache groups，说明是 attention-free model，直接返回空 metadata；
2. 根据 num_tokens / num_reqs / padding 情况确定 padded 形状；
3. 为每个 KV cache group 取 block table；
4. 从 slot_mappings 中取对应 group 的 slot mapping；
5. 构造 CommonAttentionMetadata；
6. 处理 DCP、mm prefix、logits_indices、kv sharing fast prefill；
7. 对每个 KV cache group / attention group 调 builder.build()；
8. 把生成的 metadata 绑定到 group 内每个 layer_name；
9. 如果有 ubatching，则生成 list[dict[layer_name, metadata]]；
10. 如果有 spec decode，则返回 drafter 需要的 common metadata。
```

其中构造 `CommonAttentionMetadata` 的位置是：

```python
cm_base = CommonAttentionMetadata(
    query_start_loc=...,
    seq_lens=...,
    block_table_tensor=...,
    slot_mapping=...,
    positions=...,
    is_prefilling=...,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2330`

调用 builder 的位置是：

```python
attn_metadata_i = builder.build(
    common_prefix_len=cascade_attn_prefix_len,
    common_attn_metadata=common_attn_metadata,
    **extra_attn_metadata_args,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2431`

最终把同一 attention group 的 metadata 赋给每个 layer：

```python
for layer_name in attn_group.layer_names:
    attn_metadata_dict[layer_name] = attn_metadata_i
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2446`

所以：

```text
Attention metadata 的最终形态是 layer_name → backend-specific metadata。
```

---

## 11. ForwardContext 的作用

`ForwardContext` 定义在：

```text
code/vllm/vllm/forward_context.py:128
```

它保存：

```text
no_compile_layers
attn_metadata
slot_mapping
dp_metadata
cudagraph_runtime_mode
batch_descriptor
ubatch_slices
skip_compiled
all_moe_layers
additional_kwargs
```

位置：`code/vllm/vllm/forward_context.py:128` 到 `code/vllm/vllm/forward_context.py:180`

`set_forward_context()` 是一个 context manager：

```python
def set_forward_context(attn_metadata, vllm_config, ...):
```

位置：`code/vllm/vllm/forward_context.py:249`

它的注释说明：

```text
stores the current forward context, can be attention metadata, etc.
Here we can inject common logic for every model forward pass.
```

位置：`code/vllm/vllm/forward_context.py:261`

为什么需要 ForwardContext？

```text
模型层的 forward 签名通常只传 hidden states / positions / intermediate_tensors 等显式参数；
但 attention backend 还需要本轮动态 metadata、slot mapping、cudagraph mode、ubatch slices；
这些信息如果层层手传，会污染每个模型实现；
所以 vLLM 用 ForwardContext 做 forward 作用域内的隐式参数区。
```

一句话：

```text
ForwardContext 是 Attention 子系统在模型 forward 内部取到 runtime metadata 的桥。
```

---

## 12. Attention.forward() 真正做什么

`Attention.forward()` 入口：

```python
def forward(self, query, key, value, output_shape=None) -> torch.Tensor:
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:438`

它的核心步骤是：

```text
1. 如果需要，计算 KV scales；
2. 如果 backend 支持并启用了 query quant，先量化 query；
3. 分配 output tensor；
4. reshape query / key / value 成 [tokens, heads, head_dim]；
5. 如果 backend 不在 forward 内更新 KV cache，则先调用 unified_kv_cache_update()；
6. 调用 unified_attention_with_output()；
7. 返回 output.view(-1, hidden_size)。
```

关键代码：

```python
query = query.view(-1, self.num_heads, self.head_size)
output = output.view(-1, self.num_heads, self.head_size_v)
key = key.view(-1, self.num_kv_heads, self.head_size)
value = value.view(-1, self.num_kv_heads, self.head_size_v)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:481`

KV cache update 分支：

```python
if not self.attn_backend.forward_includes_kv_cache_update:
    kv_cache_dummy_dep = unified_kv_cache_update(...)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:491`

attention 调用：

```python
unified_attention_with_output(query, key, value, output, self.layer_name, ...)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:502`

所以 `Attention.forward()` 不是直接写一段 softmax 逻辑，而是：

```text
把 Q/K/V 整形好，保证 KV cache update 和 attention op 的顺序，再转交统一 attention op / backend impl。
```

---

## 13. unified_attention_with_output 如何进入 backend impl

`unified_attention_with_output()` 定义在：

```text
code/vllm/vllm/model_executor/layers/attention/attention.py:734
```

它被两个装饰器包住：

```python
@eager_break_during_capture
@maybe_transfer_kv_layer
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:734`

函数内部会：

```python
attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
self.impl.forward(
    self,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=output,
    ...
)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:750`

`get_attention_context()` 会从 ForwardContext 中取：

```text
- 当前 layer 的 attention metadata；
- 当前 layer 实例；
- 当前 layer 的 kv_cache tensor；
- 当前 layer 的 slot_mapping。
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:649`

这就是 attention backend impl 获取 runtime 信息的关键路径：

```text
ForwardContext
  → get_attention_context(layer_name)
  → attn_metadata / kv_cache / slot_mapping
  → impl.forward()
```

---

## 14. KV cache update 在哪里发生

attention 的一个关键副作用是写 KV cache。

`AttentionBackend` 有一个字段：

```python
forward_includes_kv_cache_update: bool = True
```

位置：`code/vllm/vllm/v1/attention/backend.py:65`

它表示：

```text
这个 backend 的 attention forward 是否已经包含 KV cache update。
```

如果不包含，`Attention.forward()` 会显式调用：

```python
unified_kv_cache_update(key, value, self.layer_name)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:499`

`unified_kv_cache_update()` 内部会取当前 layer 的 kv_cache 和 slot_mapping：

```python
_, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
attn_layer.impl.do_kv_cache_update(
    attn_layer,
    key,
    value,
    kv_cache,
    layer_slot_mapping,
)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:692`

为什么要有 `kv_cache_dummy_dep`？

源码注释解释：

```text
Returns a dummy that is passed to unified_attention to signal a side effect and the data dependency between them to ensure torch.compile preserves ordering.
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:697`

也就是说：

```text
为了让 torch.compile 不重排 KV cache update 和 attention read，vLLM 显式制造了一个依赖。
```

这说明 Attention 子系统不仅是 kernel 调用，还要处理：

```text
KV cache 写入副作用和编译图执行顺序。
```

---

## 15. Attention 和 KV Connector hook 的关系

KV connector hook 发生在 attention layer 边界。

装饰器在：

```text
code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:15
```

逻辑是：

```python
connector.wait_for_layer_load(layer_name)
result = func(*args, **kwargs)
connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

位置：`code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:50` 到 `code/vllm/vllm/model_executor/layers/attention/kv_transfer_utils.py:57`

为什么挂在 attention layer，而不是 ModelRunner 外层？

```text
1. attention layer entry 是读取该层历史 KV 前的最后安全点；
2. attention layer exit 是该层新 KV 写入后的第一个可保存点；
3. 这里能拿到 layer_name、kv_cache tensor、attn_metadata；
4. async layer-by-layer load/save 可以和模型 forward overlap。
```

所以 Attention 子系统是 KV transfer 真正读写 KV tensor 的挂点。

这部分在 `../kv_cache_transfer/07_worker_kv_connector_flow.md` 和 `09_attention_and_kv_connector_hooks.md` 中会继续展开。

---

## 16. Attention 和 CUDA graph / torch.compile 的关系

Attention 子系统会影响 CUDA graph / compile。

### 16.1 backend builder 声明 cudagraph 支持等级

`AttentionMetadataBuilder` 有：

```python
_cudagraph_support: AttentionCGSupport = AttentionCGSupport.NEVER
```

位置：`code/vllm/vllm/v1/attention/backend.py:533`

支持等级包括：

```text
ALWAYS
UNIFORM_BATCH
UNIFORM_SINGLE_TOKEN_DECODE
NEVER
```

位置：`code/vllm/vllm/v1/attention/backend.py:516`

ModelRunner 初始化 attention backend 时会取所有 backend 的最小支持级别，决定最终 cudagraph mode。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6877`

### 16.2 Attention op 对 torch.compile 是 opaque custom op 或 direct call

`Attention.__init__()` 中：

```python
self.use_direct_call = not current_platform.opaque_attention_op()
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:394`

源码注释说明：

```text
For cuda-alike and cpu platforms, we control how torch.compile works by registering the attention as one giant opaque custom op.
For other platforms, we directly call them and let torch.compile handle them.
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:390`

这意味着：

```text
Attention 子系统是 torch.compile 图边界设计的重要部分。
```

### 16.3 KV cache update 顺序也要照顾 compile

`unified_kv_cache_update()` 返回 dummy dependency，就是为了让 compile 保持 KV update 与 attention 的顺序。

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:697`

所以 attention 角色里必须包含：

```text
执行正确性 + 编译图约束。
```

---

## 17. Attention 子系统支持哪些 attention type

`AttentionType` 定义在：

```text
code/vllm/vllm/v1/attention/backend.py:32
```

包括：

```text
DECODER：decoder self-attention；
ENCODER：encoder attention for encoder-decoder；
ENCODER_ONLY：encoder-only attention；
ENCODER_DECODER：decoder query 对 encoder K/V 的 cross attention。
```

AttentionBackend 默认只支持 decoder：

```python
def supports_attn_type(cls, attn_type: str) -> bool:
    return attn_type == AttentionType.DECODER
```

位置：`code/vllm/vllm/v1/attention/backend.py:248`

这说明：

```text
Attention 子系统不是只服务 decoder-only LLM，抽象上也覆盖 encoder-only 和 encoder-decoder attention；但具体 backend 是否支持由 capability check 决定。
```

---

## 18. Attention 子系统如何看待 MHA / MQA / GQA / MLA / PagedAttention

本篇先只定位，不展开细节。

### 18.1 MHA / MQA / GQA

这些是模型结构里的 head 组织方式。

在 `Attention` layer 中对应：

```text
num_heads
num_kv_heads
head_size
head_size_v
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:290`

它们影响：

```text
- Q heads 数量；
- K/V heads 数量；
- KV cache shape；
- backend forward 参数；
- Tensor Parallel 切分。
```

但它们不是 backend 名字。

### 18.2 MLA

MLA 是模型结构和 KV 表示方式的特殊路径。

`AttentionBackend.is_mla()` 默认返回 False：

```python
def is_mla(cls) -> bool:
    return False
```

位置：`code/vllm/vllm/v1/attention/backend.py:204`

MLA backend 会覆盖该能力，并走 `MLAAttentionImpl` 相关接口。

### 18.3 PagedAttention

PagedAttention 是 KV cache 管理和访问方式，不是某个单独 layer 类。

它在代码里体现为：

```text
KVCacheManager / BlockPool 分配 blocks；
InputBatch 维护 block_table；
ModelRunner 构造 slot_mapping；
AttentionMetadata 携带 block_table_tensor / slot_mapping；
backend kernel 按 paged KV cache 访问。
```

### 18.4 FlashAttention / FlashInfer / Triton

这些是 backend / kernel 路径。

它们通常落在：

```text
code/vllm/vllm/v1/attention/backends/
```

并通过 `AttentionBackend` 抽象接入。

---

## 19. Attention 子系统和 Scheduler 的边界

Scheduler 负责：

```text
- 维护 waiting / running；
- token budget；
- prefill / decode / chunked prefill 调度；
- KV block 分配；
- prefix cache 命中；
- num_common_prefix_blocks；
- SchedulerOutput。
```

Attention 子系统负责：

```text
- 根据 SchedulerOutput 间接产生的 num_scheduled_tokens / block_table / slot_mapping / seq_lens 执行 attention；
- 不参与调度决策；
- 不修改 Scheduler 的请求状态；
- 不直接分配或释放 KV blocks。
```

边界一句话：

```text
Scheduler 决定 attention 要服务哪些 token；Attention 子系统决定这些 token 如何在 backend 上做注意力计算。
```

---

## 20. Attention 子系统和 ModelRunner 的边界

ModelRunner 负责：

```text
- 消费 SchedulerOutput；
- 维护 InputBatch；
- 准备 input_ids / positions；
- 计算 slot mappings；
- 调用 _build_attention_metadata()；
- 设置 ForwardContext；
- 调用模型 forward。
```

Attention 子系统负责：

```text
- 定义 backend；
- 定义 metadata builder；
- 定义 KV cache shape；
- 构造 backend-specific metadata；
- 在 attention layer 中读 ForwardContext；
- 调用 backend impl；
- 更新 / 读取 KV cache。
```

边界一句话：

```text
ModelRunner 负责“把 batch 准备好”，Attention 子系统负责“把 attention 算出来”。
```

---

## 21. Attention 子系统和模型实现的边界

模型实现通常负责：

```text
- Q / K / V projection；
- RoPE / ALiBi 等位置编码；
- 调用 Attention.forward(query, key, value)；
- 后续 output projection / MLP / residual。
```

Attention 子系统负责：

```text
- 对 Q/K/V 做统一 reshape；
- KV cache update；
- 依据 metadata 运行 attention；
- 返回 attention output。
```

也就是说，模型层会把 Q/K/V 交给 `Attention.forward()`，但不自己处理 paged KV cache 的底层读写细节。

---

## 22. 一个完整例子：普通 decoder attention

假设一个普通 decoder-only generation 请求被调度。

链路如下：

```text
1. Scheduler 分配 KV blocks，生成 SchedulerOutput。

2. ModelRunner._update_states()
   → 把 block ids 写进 InputBatch 的 block table。

3. ModelRunner._prepare_inputs()
   → 准备 input_ids / positions / query_start_loc / seq_lens。

4. ModelRunner._get_slot_mappings()
   → 从 block table 得到每个 token 的 KV slot。

5. ModelRunner._build_attention_metadata()
   → 构造 CommonAttentionMetadata；
   → 调 backend builder.build()；
   → 得到 layer_name → attn_metadata。

6. set_forward_context()
   → 把 attn_metadata 和 slot_mapping 放进 ForwardContext。

7. 模型 block forward
   → 做 Q/K/V projection；
   → 调 Attention.forward(query, key, value)。

8. Attention.forward()
   → reshape Q/K/V；
   → 如果 backend 不含 KV update，先 unified_kv_cache_update()；
   → unified_attention_with_output()。

9. unified_attention_with_output()
   → get_attention_context(layer_name)；
   → 取 attn_metadata / kv_cache；
   → 调 self.impl.forward(...)

10. backend impl
    → 调 FlashAttention / FlashInfer / Triton / CPU 等具体 kernel；
    → 输出 attention result。
```

关键点：

```text
attention backend 从不直接问 Scheduler “这个 request 有哪些 block”；
它只消费 ModelRunner 已经准备好的 block table / slot mapping / metadata。
```

---

## 23. 一个完整例子：KV connector load / save 场景

如果启用了 KV connector，Attention 子系统还负责在 attention layer 边界触发 load / save hook。

链路是：

```text
1. SchedulerOutput.kv_connector_metadata 进入 ModelRunner。
2. ModelRunner.maybe_get_kv_connector_output() 绑定 connector metadata。
3. start_load_kv(forward_context) 发起外部 KV load。
4. Attention layer 调 unified_attention_with_output()。
5. maybe_transfer_kv_layer 装饰器先执行 wait_for_layer_load(layer_name)。
6. attention forward 读取已经 load 完的 KV。
7. attention forward 后执行 save_kv_layer(layer_name, kv_cache, attn_metadata)。
8. forward context 退出时 wait_for_save()，收集 KVConnectorOutput。
```

这说明：

```text
KV connector 的真实数据传输点不是 Scheduler，也不是单纯 ModelRunner 外层，而是 attention layer 的 KV cache 边界。
```

---

## 24. 容易疑惑的点

### 24.1 AttentionBackend 是不是 FlashAttention？

不是。

```text
FlashAttention 可以是某个 backend 或 backend 内部 kernel；
AttentionBackend 是 vLLM 的抽象接口，它还可以代表 FlashInfer、FlashMLA、Triton、CPU、FlexAttention 等路径。
```

### 24.2 PagedAttention 是不是一个 backend？

不完全是。

```text
PagedAttention 更像 vLLM 的 KV cache 管理和访问机制；
它通过 block table / slot mapping / paged KV cache 体现在多个 backend 的执行路径里。
```

### 24.3 Attention 子系统是否分配 KV blocks？

不分配。

```text
KV blocks 的分配在 Scheduler / KVCacheManager；
Attention 子系统只按已分配的 block table / slot mapping 读写 KV cache。
```

### 24.4 Attention metadata 是不是一个统一结构？

不是只有一个统一结构。

```text
CommonAttentionMetadata 是公共输入；
不同 backend builder 会把它转换成不同的 backend-specific metadata。
```

### 24.5 Attention.forward() 是否直接调用具体 kernel？

中间还有一层统一 op。

```text
Attention.forward()
  → unified_attention_with_output()
  → get_attention_context()
  → self.impl.forward()
  → backend kernel
```

### 24.6 MHA / GQA / MLA 和 FlashInfer / FlashAttention 是同一维度吗？

不是。

```text
MHA / MQA / GQA / MLA 偏模型结构；
FlashAttention / FlashInfer / Triton 偏 backend / kernel；
PagedAttention / HMA 偏 KV cache 管理和 layout。
```

---

## 25. 从“回答问题”的角度总结

如果问：

```text
Attention 子系统在 vLLM V1 里是什么层？
```

可以回答：

```text
Attention 子系统是 ModelRunner 内部执行模型 forward 时使用的计算和 metadata 子系统。它接收 ModelRunner 根据 SchedulerOutput、InputBatch、block table、slot mapping 构造出的 attention metadata，通过 ForwardContext 暴露给每个模型 attention layer。每个 Attention layer 根据模型配置选择 AttentionBackend，并持有具体 AttentionImpl。forward 时，Attention.forward() 会 reshape Q/K/V，必要时根据 slot mapping 更新 paged KV cache，然后通过 unified_attention_with_output() 取出当前层 metadata、KV cache tensor，并调用 backend impl 执行具体 attention kernel。它不负责调度和 block 分配，但负责把已经分配好的 paged KV cache 用正确的 backend 算起来。
```

如果问：

```text
Attention 子系统和 Scheduler / ModelRunner / KVCacheManager 怎么分工？
```

可以回答：

```text
Scheduler 决定本轮哪些请求跑、跑多少 token，并通过 KVCacheManager 分配逻辑 KV blocks。ModelRunner 把这些调度结果同步到 InputBatch，生成 block table、slot mapping、seq_lens、query_start_loc 和 attention metadata。Attention 子系统消费这些 metadata，在每个模型 attention layer 中读写本地 paged KV cache，并调用具体 backend kernel 完成 attention 计算。
```

---

## 26. 最关键流程图

```text
初始化阶段
  ├─ 模型构造 Attention layer
  │    ├─ get_attn_backend(...)
  │    ├─ self.attn_backend
  │    ├─ self.impl = attn_backend.get_impl_cls()(...)
  │    └─ compilation_config.static_forward_context[layer_name] = self
  │
  ├─ ModelRunner.get_kv_cache_spec()
  │    └─ Attention.get_kv_cache_spec()
  │         └─ FullAttentionSpec / SlidingWindowSpec / MLA spec / etc.
  │
  └─ ModelRunner.initialize_kv_cache()
       ├─ initialize_attn_backend()
       │    └─ attention groups by backend / spec / num_heads_q
       ├─ prepare_kernel_block_sizes()
       ├─ initialize_metadata_builders()
       └─ initialize_kv_cache_tensors()

执行阶段
  ├─ SchedulerOutput
  ├─ GPUModelRunner.execute_model()
  │    ├─ _update_states()
  │    ├─ _prepare_inputs()
  │    ├─ _get_slot_mappings()
  │    ├─ _build_attention_metadata()
  │    │    ├─ CommonAttentionMetadata
  │    │    ├─ builder.build(...)
  │    │    └─ layer_name → backend-specific metadata
  │    ├─ _preprocess()
  │    └─ set_forward_context(attn_metadata, slot_mapping, ...)
  │
  └─ model forward
       └─ each Attention layer
            ├─ Attention.forward(query, key, value)
            ├─ reshape Q/K/V
            ├─ optional unified_kv_cache_update()
            ├─ unified_attention_with_output()
            │    ├─ maybe_transfer_kv_layer
            │    │    ├─ wait_for_layer_load(layer_name)
            │    │    └─ save_kv_layer(layer_name, kv_cache, attn_metadata)
            │    ├─ get_attention_context(layer_name)
            │    └─ impl.forward(layer, q, k, v, kv_cache, attn_metadata, output)
            └─ attention output
```

---

## 27. 最关键对象关系

```text
Attention
  模型里的 attention layer 外壳，持有 backend、impl、kv_cache、layer_name。

AttentionBackend
  backend 抽象，定义 impl、metadata builder、KV cache shape、layout 和 capability checks。

AttentionMetadataBuilder
  把 CommonAttentionMetadata 转成 backend-specific metadata。

CommonAttentionMetadata
  本轮 batch 的公共 attention 运行参数，包括 query_start_loc、seq_lens、block_table_tensor、slot_mapping 等。

AttentionMetadata
  backend-specific metadata 的基类。

AttentionImpl / MLAAttentionImpl
  真正执行 attention forward 的 backend 实现。

ForwardContext
  模型 forward 期间的隐式上下文，保存 attn_metadata、slot_mapping、cudagraph mode、ubatch 信息。

KVCacheSpec / AttentionSpec
  Attention layer 向系统声明的 KV cache 需求。

KV cache tensor
  Worker / ModelRunner 侧真实物理 KV cache，attention backend 读写它。

slot_mapping
  本轮 token 到物理 KV slot 的映射。

block_table_tensor
  request 到 KV blocks 的映射，paged attention 读取历史 KV 的核心索引。
```

---

## 28. 和后续专题的关系

本篇回答的是 Attention 子系统的总定位。

后续专题继续拆：

```text
02_backend_selection.md
  详细解释 get_attn_backend()、platform selector、backend capability checks。

03_attention_metadata_builder.md
  详细解释 CommonAttentionMetadata 和 backend-specific metadata 构造。

04_prefill_decode_metadata.md
  详细解释 prefill / decode / mixed batch / spec decode 的 metadata 差异。

05_slot_mapping_and_block_table.md
  详细解释 block table、slot mapping、paged KV cache 地址映射。

06_attention_forward_flow.md
  详细解释 Attention.forward() 到 backend impl forward 的调用链。

07_kv_cache_layout_and_backend.md
  详细解释 KV cache shape / stride / kernel block size / backend layout。

08_cascade_attention.md
  详细解释 num_common_prefix_blocks 和 cascade attention。

09_attention_and_kv_connector_hooks.md
  详细解释 wait_for_layer_load() / save_kv_layer() hook。

10_cuda_graph_compile_interaction.md
  详细解释 attention 与 CUDA graph / torch.compile 的关系。
```

最终最小心智模型：

```text
Attention 子系统 = backend 选择 + KV cache spec/layout + attention metadata builder + ForwardContext + Attention.forward + backend impl + paged KV cache read/write。
```
