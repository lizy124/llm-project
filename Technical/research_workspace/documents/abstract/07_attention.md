# 07 attention 背诵文档

## 1. 专题定位

`attention` 讲的是 vLLM V1 中 attention 子系统如何把调度结果、KV cache、模型层和底层 backend 串起来。

它不是只讲 FlashAttention，也不是只讲某个 kernel。

一句话：

```text
Attention 子系统负责把当前 batch 的 token、位置、block table、slot mapping 和 KV cache layout 翻译成具体 backend 可执行的 attention 调用。
```

## 2. 最小心智模型

主链路是：

```text
SchedulerOutput
  → GPUModelRunner._update_states()
  → GPUModelRunner._prepare_inputs()
  → block table / slot mapping
  → GPUModelRunner._build_attention_metadata()
  → set_forward_context(attn_metadata, slot_mapping, ...)
  → model attention layers
  → AttentionBackend / AttentionImpl
  → paged KV cache read / write
  → hidden states
```

要背住：

```text
Scheduler 决定本轮跑哪些 token；ModelRunner 把这些 token 翻译成 attention metadata；Attention backend 根据 metadata 读写 paged KV cache 并执行 kernel。
```

## 3. attention 子系统负责什么

它负责：

```text
1. 根据平台、dtype、head size、KV dtype、block size 等选择 backend。
2. 定义 backend 的 KV cache shape / stride / layout。
3. 把 block table / slot mapping / seq_lens / query_start_loc 组装成 metadata。
4. 在模型 forward 期间让每层 attention 拿到自己的 metadata 和 KV cache。
5. 把当前 token 的 key/value 写入 paged KV cache。
6. 从 block table 指向的历史 blocks 中读取上下文 KV。
7. 区分 prefill / decode / mixed batch / spec decode / cascade attention。
8. 配合 KV connector 做逐层 load / save。
9. 配合 CUDA graph / torch.compile 处理固定 shape 和 capture。
```

## 4. attention 子系统不负责什么

它不负责：

```text
决定哪些 request 被调度
token budget
KV block 分配和抢占
prefix cache 命中决策
外部 KV 什么时候可用
采样 token
构造 RequestOutput
```

这些属于：

```text
Scheduler
KVCacheManager
ModelRunner sampling
OutputProcessor
```

边界：

```text
Scheduler / KVCacheManager 管哪些 token 能跑、哪些 block 可用。
ModelRunner 管这些 token 和 block 如何喂给 attention。
AttentionBackend / AttentionImpl 管用哪个 kernel 算 attention。
```

## 5. 核心角色总览

```text
AttentionBackend：说明 backend 能不能用、KV cache 长什么样、impl 和 builder 是谁。
CommonAttentionMetadata：所有 backend 共享的 batch 级 attention 描述。
AttentionMetadataBuilder：把公共 metadata 转成 backend-specific metadata。
AttentionImplBase / AttentionImpl：真正调用 kernel 算 attention。
Attention layer：模型层中调用 backend 的统一入口。
ForwardContext：forward 期间保存 metadata / slot mapping / cudagraph mode 的隐式上下文。
```

## 6. AttentionBackend

`AttentionBackend` 是后端能力描述和工厂入口。

它负责：

```text
声明 backend 名称
返回 impl class
返回 metadata builder class
定义 KV cache shape
定义 KV cache stride order
声明支持哪些 dtype / head size / block size / KV dtype
声明是否支持 MLA / sparse / sink / non-causal / KV connector / CUDA graph
validate_configuration()
```

一句话：

```text
AttentionBackend 说明这个 backend 能不能用、KV cache 怎么布局、最终谁来执行。
```

常见 backend：

```text
FlashAttention
FlashInfer
Triton Attention
FlexAttention
FlashMLA
CUTLASS MLA
CPU / ROCm / TurboQuant / custom backend
```

## 7. CommonAttentionMetadata

`CommonAttentionMetadata` 是所有 backend 共享的 batch 描述。

它包含：

```text
query_start_loc：每个 request 的 query token 起止边界。
seq_lens：每个 request 当前序列长度。
num_reqs：当前 request 数。
num_actual_tokens：真实 token 数，padding 场景可能小于 metadata token 数。
max_query_len：单 request 最大 query 长度。
max_seq_len：最大上下文长度。
block_table_tensor：request → KV blocks。
slot_mapping：token → KV cache slot。
causal：是否 causal。
positions：token 位置。
is_prefilling：哪些 request 处于 prefill。
encoder_seq_lens / dcp_local_seq_lens / mm_req_doc_ranges：特殊路径字段。
```

一句话：

```text
CommonAttentionMetadata 是所有 backend 都能看懂的公共 batch attention 描述。
```

## 8. AttentionMetadataBuilder

Builder 负责把 `CommonAttentionMetadata` 转成某个 backend 私有的 metadata。

它做：

```text
接收 CommonAttentionMetadata
结合 KVCacheSpec / layer_names / config / device
构造 FlashAttentionMetadata / FlashInfer metadata / Triton metadata / MLA metadata
处理 cascade attention
处理 CUDA graph capture metadata
必要时 update_block_table 复用旧 metadata
声明 CUDA graph 支持级别
```

一句话：

```text
AttentionMetadataBuilder 是公共 metadata 到 backend metadata 的翻译器。
```

## 9. AttentionImpl

`AttentionImpl` 是真正执行 attention 的实现。

它保存：

```text
num_heads
num_kv_heads
head_size
scale
sliding_window
kv_cache_dtype
alibi_slopes
DCP / PCP world size 和 rank
```

普通 attention 调：

```text
AttentionImpl.forward()
```

MLA 可能拆成：

```text
forward_mha()
forward_mqa()
do_kv_cache_update()
```

一句话：

```text
AttentionImpl 负责真正调 backend kernel 算 attention。
```

## 10. Attention layer

模型里的 attention layer 是模型 forward 和 backend 的连接点。

它负责：

```text
初始化时选择 backend
创建 backend impl
注册到 static_forward_context
forward 时从 ForwardContext 取当前层 metadata / KV cache / slot mapping
必要时先写 KV cache
调用 backend impl forward
对 torch.compile 暴露统一 attention opaque op
```

一句话：

```text
Attention layer 是模型层里调用 attention backend 的统一入口。
```

## 11. ForwardContext

`ForwardContext` 是模型 forward 生命周期内的隐式参数区。

它保存：

```text
attn_metadata：按 layer name 组织的 attention metadata。
slot_mapping：按 layer name 组织的 slot mapping。
no_compile_layers：forward 期间可访问的静态 layer 对象。
cudagraph_runtime_mode：当前 FULL / PIECEWISE / NONE。
batch_descriptor：CUDA graph dispatch key。
ubatch_slices：microbatch 切片。
dp_metadata：data parallel token 信息。
```

ModelRunner 在 forward 前调用：

```text
set_forward_context(...)
```

这样 attention layer 不需要显式传一堆参数，也能拿到本轮 metadata。

## 12. backend 如何选择

选择入口在 Attention layer 初始化时：

```text
get_attn_backend()
```

输入条件包括：

```text
head_size
dtype
kv_cache_dtype
block_size
use_mla
has_sink
use_sparse
use_mm_prefix
attn_type
use_non_causal
use_batch_invariant
use_kv_connector
num_heads
```

选择逻辑：

```text
1. 如果用户指定 backend，优先使用指定 backend。
2. validate_configuration() 通过就用。
3. 如果 auto，由 current_platform 按优先级尝试 backend。
4. 选择第一个满足条件的 backend。
5. 必要时更新全局 KV cache layout。
```

一句话：

```text
backend 选择同时看平台、模型结构、dtype、KV cache 配置和运行功能。
```

## 13. CUDA 平台 backend 优先级

非 MLA 常见理解：

```text
Blackwell / compute capability 10：
  FlashInfer → FlashAttention → Triton → FlexAttention → TurboQuant

其他 CUDA：
  FlashAttention → FlashInfer → Triton → FlexAttention → TurboQuant
```

MLA backend 有另一套优先级，会根据：

```text
device capability
head count
KV dtype
sparse 需求
```

选择。

## 14. ModelRunner 如何构造 attention metadata

运行时主链路：

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

attention 关键三步：

```text
_prepare_inputs：准备 query_start_loc / positions / seq_lens。
_get_slot_mappings：生成按 group 和按 layer 的 slot mapping。
_build_attention_metadata：构造 CommonAttentionMetadata 和 backend-specific metadata。
```

## 15. _prepare_inputs 的作用

`_prepare_inputs()` 把 request 状态变成 token 级张量。

它做：

```text
commit block table
根据 num_scheduled_tokens 生成 req_indices
计算 query_start_loc
计算 positions
更新 seq_lens
生成 discard_request_mask
计算 block_table.compute_slot_mapping()
生成 logits_indices
准备 SpecDecodeMetadata
```

一句话：

```text
_prepare_inputs 把请求级状态压平成本轮 token 级输入。
```

## 16. slot mapping 和 block table

### block table

block table 是请求级映射：

```text
request index
  → logical token block index
  → physical KV block id
```

用于读历史 KV。

### slot mapping

slot mapping 是 token 级映射：

```text
当前 batch 第 i 个 token
  → request index
  → token position
  → block id + block 内 offset
  → KV cache slot
```

用于写当前 token 的 K/V。

要背住：

```text
block table 偏历史读取；slot mapping 偏当前写入。attention backend 两者都需要。
```

## 17. _get_slot_mappings 为什么有两种

`_get_slot_mappings()` 返回：

```text
slot_mappings_by_gid：按 KV cache group 组织，交给 metadata builder。
slot_mappings_by_layer：按 layer name 组织，交给 ForwardContext / attention layer。
```

原因：

```text
metadata 构造按 KV group 更自然。
model forward 按 layer name 查 metadata 更自然。
```

padding token 的 slot mapping 会是：

```text
-1
```

表示不写入真实 KV cache。

## 18. _build_attention_metadata 的作用

它是核心翻译层。

流程：

```text
1. 计算 padding 后 token / request 数。
2. 取每个 KV cache group 的 block_table_tensor。
3. padding request 的 block table 填 NULL_BLOCK_ID。
4. 构造 CommonAttentionMetadata。
5. 补充 DCP / encoder / multimodal / KV sharing 字段。
6. 遍历 KV cache group。
7. 遍历 group 内 AttentionGroup。
8. 调 builder.build() / build_for_cudagraph_capture() / update_block_table()。
9. 生成 per-layer attn_metadata。
10. 保存 spec decode / drafter 需要的 common metadata。
11. 处理 microbatch metadata。
```

最终输出：

```text
layer_name → backend_specific_attention_metadata
```

## 19. prefill / decode / mixed batch 如何表达

vLLM V1 通常不是简单写：

```text
mode = prefill / decode
```

而是通过字段共同表达：

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

所以同一个 batch 里可以混合：

```text
decode 1 token
prefill 128 tokens
chunked prefill 64 tokens
spec decode 1 + N tokens
```

## 20. decode 特征

普通 decode：

```text
每个 request 通常调度 1 个 token。
query_start_loc 差分基本是 1。
max_query_len = 1。
seq_lens = num_computed_tokens + 1。
slot mapping 指向新 token 写入位置。
logits_indices 选择当前 token。
```

spec decode 下：

```text
每个 request 可能是 1 个真实 token + N 个 draft token。
```

## 21. prefill 特征

prefill：

```text
某个 request 本轮调度多个 prompt token。
max_query_len 可能大于 1。
positions 从 num_computed_tokens 开始连续增长。
slot mapping 覆盖这一段 prompt token。
通常只需要最后位置 logits 用于采样。
```

chunked prefill：

```text
长 prompt 被拆成多个 step。
num_computed_tokens < prompt length。
本轮只处理一个 chunk。
非最后 chunk 可能不输出 token。
```

## 22. attention forward 如何读写 KV cache

forward 前：

```text
ModelRunner set_forward_context(attn_metadata, slot_mapping, ...)
```

模型执行到某层 attention：

```text
Attention layer 从 ForwardContext 取当前层 metadata、KV cache、slot mapping。
```

写 KV cache：

```text
当前 key/value
  + kv_cache tensor
  + slot_mapping
  + kv_cache_dtype / scale
  → 写入 paged KV cache
```

读 KV cache：

```text
kv_cache tensor
block_table
seq_lens
query_start_loc
max_seq_len
max_query_len
  → attention kernel 读取历史上下文
```

输出：

```text
attention hidden states
  → 后续 MLP / layers
  → final hidden states
  → logits / sampling
```

## 23. KV cache layout 和 backend

不同 backend 要求不同 KV cache shape。

例如 FlashAttention 常见：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

其中 `2` 是 key/value。

backend 必须定义：

```text
get_kv_cache_shape()
get_kv_cache_stride_order()
get_kv_cache_block_dim()
```

重要点：

```text
Worker 初始化 KV cache 时，会受 attention backend 的 layout 要求影响。
```

## 24. cascade attention

cascade attention 通常和 prefix cache / common prefix 有关。

核心想法：

```text
多个请求共享一段公共 prefix 时，可以把 prefix attention 和 suffix attention 拆开处理，减少重复计算。
```

它会影响：

```text
common prefix length
attention metadata
backend 是否支持 cascade
CUDA graph 是否可用
```

不是所有 backend / batch 形态都能走 cascade。

## 25. KV connector hook

KV connector 与 attention 的边界在每层 attention 附近。

典型时序：

```text
attention layer entry：
  wait_for_layer_load(layer_name)

attention forward：
  backend 读写本地 KV cache

attention layer exit：
  save_kv_layer(layer_name, kv_cache, attn_metadata)
```

这样可以：

```text
逐层等待远端 KV load。
逐层把本地 KV save 到外部 KVPool。
```

## 26. CUDA graph / compile 对 attention 的约束

CUDA graph 要求固定 shape 和稳定执行图。

attention metadata 必须支持：

```text
padded num_tokens
padded num_reqs
padding slot_mapping = -1
padding block table = NULL_BLOCK_ID
backend-specific metadata 可 capture / replay
```

backend 需要声明 CUDA graph 支持级别：

```text
ALWAYS
UNIFORM_BATCH
UNIFORM_SINGLE_TOKEN_DECODE
NEVER
```

不支持时会降级到：

```text
PIECEWISE 或 NONE
```

## 27. 常见易混点

### block table 和 slot mapping 不是一回事

```text
block table：request → blocks，用于读历史 KV。
slot mapping：token → slot，用于写当前 KV。
```

### AttentionBackend 不等于 AttentionImpl

```text
Backend 描述能力和布局；Impl 真正执行 kernel。
```

### metadata 不是单一 prefill/decode flag

```text
prefill、decode、mixed、spec decode 由 query_start_loc、seq_lens、positions、slot_mapping 等共同表达。
```

### attention 输出不是 token

```text
attention 输出 hidden states；token 由后续 logits 和 sampler 产生。
```

## 28. 与其他专题的关系

```text
scheduler：分配 KV blocks，生成 SchedulerOutput。
executor_worker_model_runner：ModelRunner 构造 attention metadata。
operators：attention backend 最终落到 kernel。
parallelism：TP / CP / DCP 改变 heads、context 和通信。
kv_cache_transfer：KV connector 在 attention layer 边界 load / save。
compilation_and_cuda_graph：attention metadata 必须适配 capture / replay。
quantization：KV cache dtype / FP8 KV / quant attention backend 会影响 backend 选择。
```

## 29. 背诵总结

背这一段：

```text
vLLM V1 的 attention 子系统是一条从 SchedulerOutput 到 backend kernel 的翻译链。Scheduler 决定本轮 token 和 KV block；ModelRunner 在 _prepare_inputs 中生成 positions、query_start_loc、seq_lens 和 slot mapping，在 _build_attention_metadata 中构造 CommonAttentionMetadata 和每层 backend metadata；forward 前通过 ForwardContext 注入 metadata 和 slot_mapping；模型中的 Attention layer 从 context 取当前层信息，先按 slot_mapping 写当前 token 的 K/V，再按 block table 和 seq_lens 读取历史 KV，由 AttentionImpl 调 backend kernel 计算 hidden states。block table 管历史读取，slot mapping 管当前写入，backend 决定 KV cache layout 和 kernel 路径。
```
