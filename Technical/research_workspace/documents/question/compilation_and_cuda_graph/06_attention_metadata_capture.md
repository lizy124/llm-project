# 06. Attention metadata 在 CUDA graph capture 下有什么特殊处理？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backend.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu\attn_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\flash_attn.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\triton_attn.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\flashinfer.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\attention\backends\mla\`
- `D:\lzy\project\kv_pool\code\vllm\docs\design\cuda_graphs.md`

本问题关注：`GPUModelRunner` 在 CUDA Graph capture / replay 场景下如何构造 attention metadata；哪些字段需要按 padded shape 固定；哪些字段可以每轮更新；为什么 full cudagraph 对 attention metadata 的要求比 piecewise cudagraph 更高；不同 attention backend 如何通过 `build_for_cudagraph_capture()` 和 `AttentionCGSupport` 表达自己的能力边界。

---

## 1. 一句话回答

CUDA Graph replay 要求 kernel launch 形态、输入 tensor 地址和关键 metadata shape 稳定；对 attention 来说，这不只是 `input_ids / positions` 固定，还包括 `query_start_loc / seq_lens / block_table / slot_mapping / max_query_len / max_seq_len / backend wrapper / scheduler metadata` 等 attention metadata 必须在 capture 和 replay 间保持同一种结构。

主链路是：

```text
GPUModelRunner.execute_model()
  → _determine_batch_execution_and_padding()
      → 决定 cudagraph_mode 和 BatchDescriptor
      → FULL 时得到 num_tokens_padded / num_reqs_padded
  → _get_slot_mappings()
      → 按 padded shape 准备 slot_mapping
      → padding token slot 填 -1
  → _build_attention_metadata()
      → 构造 CommonAttentionMetadata
      → capture 时设置 for_cudagraph_capture=True
      → 每个 AttentionMetadataBuilder 生成 backend metadata
  → set_forward_context(attn_metadata, cudagraph_runtime_mode, batch_descriptor, ...)
  → _model_forward()
      → attention layer 从 ForwardContext 取 metadata
      → replay 已 capture 的 full / piecewise graph
```

所以可以把核心结论压缩成：

```text
CUDAGraphDispatcher 决定要不要固定 shape；
_build_attention_metadata() 把固定后的 shape 翻译成 attention metadata；
AttentionMetadataBuilder 决定某个 backend 如何在 capture/replay 下保持 metadata 可复用。
```

---

## 2. 先给结论：哪些字段最关键

`CommonAttentionMetadata` 是所有 backend metadata 的公共输入，定义在 `backend.py`。

位置：`code/vllm/vllm/v1/attention/backend.py:361`

关键字段可以分成六类。

### 2.1 batch 边界字段

```text
query_start_loc
query_start_loc_cpu
num_reqs
num_actual_tokens
max_query_len
max_seq_len
```

含义：

- `query_start_loc`：每个 request 的 query token 在本轮 token buffer 里的起止位置
- `num_reqs`：本轮 request 数；full cudagraph 下可能是 padded request 数
- `num_actual_tokens`：名字叫 actual，但在 full cudagraph 下可能是 padded token 数
- `max_query_len`：区分 decode / spec decode / prefill 的关键字段
- `max_seq_len`：很多 backend 用它选 kernel 或 plan workspace

### 2.2 KV cache 索引字段

```text
block_table_tensor
slot_mapping
seq_lens
```

含义：

- `block_table_tensor`：request → KV block id 映射
- `slot_mapping`：token → KV cache slot 映射
- `seq_lens`：每个 request 当前上下文长度

在 cudagraph padding 下：

```text
padding request 的 block_table 要填 NULL_BLOCK_ID；
padding token 的 slot_mapping 要填 -1；
seq_lens / query_start_loc 的切片长度要和 padded request 数一致。
```

### 2.3 控制 attention 行为的字段

```text
causal
is_prefilling
mm_req_doc_ranges
positions
```

含义：

- `causal`：是否因果 attention，PrefixLM / mm prefix 可能局部非因果
- `is_prefilling`：Mamba / GDN 等 backend 区分 prefill 和 decode
- `mm_req_doc_ranges`：多模态 PrefixLM bidirectional range
- `positions`：某些 backend 会预计算 position-dependent metadata

### 2.4 encoder / cross attention 字段

```text
encoder_seq_lens
encoder_seq_lens_cpu
```

只在 cross attention / encoder-decoder 相关 KV cache group 中使用。

capture 时 encoder length 不能随便为空，因为 backend 可能用它确定 `max_seqlen_k` 或 plan 形态。

### 2.5 DCP 字段

```text
dcp_local_seq_lens
dcp_local_seq_lens_cpu
```

Decode Context Parallelism 下，每个 rank 看到本地 shard 的 context length。capture/replay 时它也必须按 padded request 数切片并拷到 GPU。

### 2.6 spec decode / fast prefill 辅助字段

```text
logits_indices_padded
num_logits_indices
seq_lens_cpu_upper_bound
```

这些字段不一定每个 backend 都用，但会影响 spec decode、KV sharing fast prefill 或 async spec decode 下 metadata 的安全性。

---

## 3. attention metadata 为什么会影响 CUDA Graph

CUDA Graph capture 记录的是一段 GPU work graph。

对 attention 来说，graph 是否能 replay 取决于：

```text
1. kernel launch grid 是否稳定；
2. 参与 kernel 的 tensor shape 是否稳定；
3. wrapper / plan 对象是否复用同一类执行路径；
4. kernel 内部读取的 metadata tensor 地址是否稳定；
5. prefill / decode / cascade / DCP / encoder 分支是否和 capture 时一致。
```

attention metadata 正好决定这些东西。

### 3.1 max_query_len 决定 kernel routine

`max_query_len` 是最关键的 shape 语义之一。

普通 decode：

```text
max_query_len = 1
```

spec decode 验证步：

```text
max_query_len = 1 + num_speculative_tokens
```

mixed prefill-decode：

```text
max_query_len = 本轮最长 prefill chunk 长度
```

`docs/design/cuda_graphs.md` 明确说明，full cudagraph capture 时会通过设置 `max_query_len` 区分 uniform decode 和 non-uniform mixed batch。

位置：`code/vllm/docs/design/cuda_graphs.md:145`

### 3.2 block_table 和 slot_mapping 决定 KV cache 写读位置

attention forward 通常会：

```text
先 reshape/cache 当前 token 的 K/V
再根据 block_table 读历史 KV
```

如果 full cudagraph capture 录的是 padded shape，replay 时 padding token 也会进入固定 shape 的 kernel launch。

因此：

```text
padding token 的 slot_mapping 必须是 -1，避免写入真实 KV cache；
padding request 的 block_table 必须是 NULL_BLOCK_ID，避免读到旧 request 的 block。
```

### 3.3 metadata Python 对象结构也要稳定

不同 backend metadata 可能包含：

```text
FlashAttentionMetadata
TritonAttentionMetadata
FlashInferMetadata
MLACommonMetadata
MambaAttentionMetadata
```

这些 dataclass 里可能又包含：

```text
wrapper 对象
scheduler_metadata tensor
workspace buffer
prefix/suffix metadata
cascade metadata
```

如果 capture 时 metadata 有某个字段，replay 时换成另一种结构，就可能导致：

```text
走不同 Python 分支；
调用不同 kernel；
kernel launch 参数数量或 shape 变化；
graph replay 不再安全。
```

---

## 4. _build_attention_metadata() 的整体职责

入口：`GPUModelRunner._build_attention_metadata()`。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2208`

它不是某一个 backend 的逻辑，而是统一做四层转换：

```text
1. 从 ModelRunner 的持久状态取出本轮公共 batch 信息；
2. 构造 CommonAttentionMetadata；
3. 按 KV cache group / attention group / ubatch 派发给 builder；
4. 返回 per-layer attn_metadata，放入 ForwardContext。
```

整体伪代码：

```text
_build_attention_metadata(...):
  num_tokens_padded = num_tokens_padded or num_tokens
  num_reqs_padded = num_reqs_padded or num_reqs

  max_seq_len = max_model_len if for_cudagraph_capture else max(seq_lens)

  block_table_gid_0 = _get_block_table(0)
  slot_mapping_gid_0 = slot_mappings[0]

  cm_base = CommonAttentionMetadata(
      query_start_loc = self.query_start_loc.gpu[:num_reqs_padded + 1],
      seq_lens = self.seq_lens[:num_reqs_padded],
      num_reqs = num_reqs_padded,
      num_actual_tokens = num_tokens_padded,
      max_query_len = max_query_len,
      max_seq_len = max_seq_len,
      block_table_tensor = block_table_gid_0,
      slot_mapping = slot_mapping_gid_0,
      positions = self.positions[:num_tokens_padded],
      ...
  )

  for each kv_cache_group:
      cm = shallow copy of cm_base
      cm.encoder_seq_lens = _get_encoder_seq_lens(...)
      cm.block_table_tensor = group block table
      cm.slot_mapping = group slot mapping

      for each attention group:
          if for_cudagraph_capture:
              builder.build_for_cudagraph_capture(cm)
          elif cached and supports_update_block_table:
              builder.update_block_table(cached, cm.block_table, cm.slot_mapping)
          else:
              builder.build(common_prefix_len, cm, ...)
```

---

## 5. capture 路径和普通路径的核心差异

### 5.1 capture 时 max_seq_len 用 max_model_len

`_build_attention_metadata()` 中：

```text
if for_cudagraph_capture:
    max_seq_len = self.max_model_len
else:
    max_seq_len = max(self.optimistic_seq_lens_cpu[:num_reqs])
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2239`

这样做是为了让某些 backend 在 capture 时看到足够大的 `max_seq_len`，尤其是 sliding window 模型或会按 `max_seq_len` 选择 kernel 的 backend。

注意：这并不表示真实 replay 时所有 request 的 `seq_lens` 都等于 `max_model_len`；它只是让 metadata 中的上界足够覆盖目标执行路径。

### 5.2 capture 时调用 build_for_cudagraph_capture()

核心分支：

```text
if for_cudagraph_capture:
    attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
else:
    attn_metadata_i = builder.build(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2417`

`AttentionMetadataBuilder` 的默认实现是：

```text
build_for_cudagraph_capture(cm):
    return build(common_prefix_len=0, common_attn_metadata=cm)
```

位置：`code/vllm/vllm/v1/attention/backend.py:634`

所以：

```text
没有重写的 backend，capture metadata 和普通 build 基本一致；
重写的 backend，可以专门调整 capture 下的 seq_lens / wrapper / workspace / scheduler metadata。
```

### 5.3 capture 时禁用 cascade common_prefix

默认 `build_for_cudagraph_capture()` 调用 `build(common_prefix_len=0)`。

这意味着 capture metadata 不走 cascade attention 路径。

原因是 cascade attention 依赖本轮请求的 common prefix 结构，属于动态分支；真实 dispatch 里如果使用 cascade attention，也会禁用 `FULL`，退到 `PIECEWISE` 或 `NONE`。

### 5.4 普通路径可以复用 metadata 并更新 block table

非 capture 路径中，`_build_attention_metadata()` 有一个缓存：

```text
cached_attn_metadata[(kv_cache_spec, builder_type)]
```

如果另一个 KV cache group 使用相同 builder 和 spec，并且 builder 支持：

```text
supports_update_block_table = True
```

就可以：

```text
builder.update_block_table(cached_metadata, new_block_table, new_slot_mapping)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2421`

这和 cudagraph capture 是两个层面的优化：

```text
capture 关注 graph replay 的固定 shape；
update_block_table 关注同一轮内多个 group metadata 构建的复用。
```

---

## 6. FULL cudagraph 下为什么要 padded attention metadata

在 `execute_model()` 中：

```text
pad_attn = cudagraph_mode == CUDAGraphMode.FULL
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4196`

如果本轮 dispatch 到 `FULL`，attention 也在完整 graph 内，因此 attention metadata 必须按 captured shape 准备。

### 6.1 slot_mapping 使用 padded token 数

`_get_slot_mappings()` 在 full graph 或 separate KV update 场景下会按 padded token 数构造：

```text
num_tokens_padded if pad_attn or has_separate_kv_update else num_tokens_unpadded
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4244`

然后：

```text
slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4008`

### 6.2 block_table 使用 padded request 数

`_get_block_table()` 内部：

```text
blk_table_tensor = block_table.get_device_tensor(num_reqs_padded)
blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2249`

这保证 padding request 不会继承旧 request 的 block id。

### 6.3 CommonAttentionMetadata 使用 padded 长度

`cm_base` 中：

```text
query_start_loc = self.query_start_loc.gpu[:num_reqs_padded + 1]
seq_lens = self.seq_lens[:num_reqs_padded]
num_reqs = num_reqs_padded
num_actual_tokens = num_tokens_padded
positions = self.positions[:num_tokens_padded]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2330`

注意 `num_actual_tokens` 在 full cudagraph 下实际表达的是“metadata 要暴露给 kernel 的 token 数”，可能已经包含 padding。

---

## 7. _dummy_run() 中 capture metadata 如何被创建

CUDA Graph capture 主要由 `_dummy_run()` 触发。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5657`

### 7.1 dummy run 先构造可控 batch shape

`_dummy_run()` 会根据参数构造：

```text
num_scheduled_tokens_list
num_reqs
max_query_len
uniform_decode
```

其中：

```text
uniform_decode=True  → max_query_len = uniform_decode_query_len
uniform_decode=False → max_query_len = num_tokens
```

这一步直接影响 attention backend capture 到 decode routine 还是 mixed/prefill routine。

### 7.2 dummy run 也走 dispatch 和 padding

```text
_determine_batch_execution_and_padding(
  num_tokens=num_tokens_unpadded,
  num_reqs=num_reqs,
  max_num_scheduled_tokens=max_query_len,
  force_uniform_decode=uniform_decode,
  force_has_lora=...,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5762`

这样 capture 的 `BatchDescriptor` 与真实 dispatch 的 key 保持一致。

### 7.3 FULL capture 会强制构造 attention metadata

`_dummy_run()` 中：

```text
if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
    _build_attention_metadata(..., for_cudagraph_capture=is_graph_capturing)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5831`

含义是：

```text
FULL graph capture 必须包含 attention metadata；
PIECEWISE graph capture 通常可以不 capture attention；
warmup 即使 mode=NONE，也可以 force_attention 预热 attention backend。
```

### 7.4 padding slot 在 dummy run 中也要安全

dummy run 没有真实 KV slot，所以会：

```text
slot_mappings_by_group.values(): fill_(-1)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5821`

这保证 capture / warmup 不会把 dummy token 写入真实 KV cache。

---

## 8. AttentionCGSupport：backend 如何声明能力

`AttentionMetadataBuilder` 用 `_cudagraph_support` 声明 full cudagraph 支持级别。

位置：`code/vllm/vllm/v1/attention/backend.py:516`

枚举含义：

```text
ALWAYS
  支持 mixed prefill-decode 和 uniform decode 的 full cudagraph。

UNIFORM_BATCH
  只支持所有 request query length 一致的 batch。
  普通 decode 和 spec decode 验证步通常属于这个场景。

UNIFORM_SINGLE_TOKEN_DECODE
  只支持 query_len == 1 的 decode。

NEVER
  不支持 full cudagraph attention。
```

`init_attn_backend()` 会遍历所有 attention group，取最小能力：

```text
min_cg_support = min(all builder.get_cudagraph_support(...))
```

位置：`code/vllm/vllm/v1/worker/gpu/attn_utils.py:118`

然后 `GPUModelRunner._check_and_update_cudagraph_mode()` 会据此修正最终 `cudagraph_mode`。

这解释了为什么某些模型虽然配置了 `FULL`，最终也可能退到：

```text
FULL_AND_PIECEWISE
FULL_DECODE_ONLY
PIECEWISE
NONE
```

---

## 9. backend build_for_cudagraph_capture() 的典型差异

不同 backend 对 capture metadata 的处理差异很大。下面只抓关键设计点。

### 9.1 TritonAttention

`TritonAttentionMetadataBuilder`：

```text
_cudagraph_support = ALWAYS
```

位置：`code/vllm/vllm/v1/attention/backends/triton_attn.py:98`

它重写了 `build_for_cudagraph_capture()`：

```text
attn_metadata = self.build(0, common_attn_metadata)
attn_metadata.seq_lens.fill_(1)
return attn_metadata
```

位置：`code/vllm/vllm/v1/attention/backends/triton_attn.py:173`

注释说明：

```text
full graph capture 时如果 seq_lens 设置为 max_model_len，capture 会非常慢；
因此 capture metadata 中把 seq_lens 填成 1。
```

这体现了一个重要原则：

```text
capture metadata 不一定等于真实请求 metadata；
它只要能录到正确 kernel 形态，并让 replay 时用稳定 tensor 地址/shape 即可。
```

### 9.2 FlashAttention

`FlashAttentionMetadataBuilder` 支持 `update_block_table`：

```text
supports_update_block_table = True
```

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:318`

它会根据 FA 版本声明 cudagraph 能力：

```text
FA3 / XPU → 更强的 full cudagraph 支持
FA2      → 通常限制到 UNIFORM_BATCH
```

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:294`

FlashAttention metadata 中比较特殊的是 `scheduler_metadata`：

```text
scheduler_metadata
prefix_scheduler_metadata
max_num_splits
```

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:265`

在 FA3 + full cudagraph 下，builder 会预分配固定大小的 `self.scheduler_metadata`，build 时把本轮 scheduler metadata copy 进去，并把剩余部分置 0。

位置：`code/vllm/vllm/v1/attention/backends/flash_attn.py:562`

目的：

```text
保持 full graph 下 scheduler metadata 的地址/容量稳定；
避免无效 scheduler slot 被 kernel 线程块读取导致覆盖输出。
```

### 9.3 FlashInfer

`FlashInferMetadataBuilder` 的特殊点是 wrapper / plan 对象。

它在 full cudagraph decode 启用时会为不同 batch size 准备独立 decode wrapper：

```text
_decode_wrappers_cudagraph: dict[int, BatchDecodeWithPagedKVCacheWrapper]
```

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:597`

构造 cudagraph decode wrapper 时会传：

```text
use_cuda_graph=True
paged_kv_indptr_buffer=...
paged_kv_indices_buffer=...
paged_kv_last_page_len_buffer=...
```

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:808`

这说明 FlashInfer 的 CUDA Graph 兼容性不只依赖 tensor shape，还依赖 FlashInfer wrapper 内部 plan 是否使用固定 buffer。

在 `build()` 中，decode path 会根据是否使用 cudagraph，选择 cudagraph wrapper 或普通 wrapper，并调用 `fast_plan_decode()` 准备 decode plan。

位置：`code/vllm/vllm/v1/attention/backends/flashinfer.py:1243`

### 9.4 MLA backend

MLA 系列 backend 多数声明：

```text
UNIFORM_BATCH
```

例如：

- `FlashInferMLAMetadataBuilder`：`code/vllm/vllm/v1/attention/backends/mla/flashinfer_mla.py:33`
- `FlashMLAMetadataBuilder`：`code/vllm/vllm/v1/attention/backends/mla/flashmla.py:110`
- `TritonMLAMetadataBuilder`：`code/vllm/vllm/v1/attention/backends/mla/triton_mla.py:32`

含义是：

```text
MLA full cudagraph 通常只覆盖 uniform decode / spec decode 验证这类 query length 一致场景；
mixed prefill-decode 要退到 PIECEWISE 或 NONE。
```

某些 MLA backend 只支持：

```text
UNIFORM_SINGLE_TOKEN_DECODE
```

例如 Cutlass MLA。

位置：`code/vllm/vllm/v1/attention/backends/mla/cutlass_mla.py:32`

### 9.5 Mamba / GDN / Linear 类 backend

这类 backend 不一定是传统 attention，但仍通过同一套 attention metadata builder 接口参与 dispatch。

例如 Mamba 声明：

```text
_cudagraph_support = UNIFORM_BATCH
```

位置：`code/vllm/vllm/v1/attention/backends/mamba_attn.py:81`

`_build_attention_metadata()` 还会在 spec decode 下给 Mamba2 / GDN builder 传额外字段：

```text
num_accepted_tokens
num_decode_draft_tokens_cpu
prev_last_scheduled_idx
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2398`

这些字段也要按 `num_reqs_padded` 切片，否则 full graph 下 shape 不一致。

---

## 10. block_table / slot_mapping 的可更新性

### 10.1 为什么要 update_block_table()

在 hybrid KV cache 或多个 KV cache group 场景中，不同 group 的 metadata 可能完全相同，只是：

```text
block_table_tensor 不同
slot_mapping 不同
```

如果每个 group 都重新 build backend metadata，成本较高。

所以 builder 可以声明：

```text
supports_update_block_table = True
```

并实现：

```text
update_block_table(metadata, blk_table, slot_mapping)
```

接口位置：`code/vllm/vllm/v1/attention/backend.py:619`

### 10.2 它和 cudagraph replay 的关系

`update_block_table()` 不是直接做 graph replay，而是 metadata 构建层的复用。

但它体现了 cudagraph 兼容 metadata 的一个关键约束：

```text
metadata 对象结构可以复用；
动态变化的部分尽量收敛为 tensor 内容更新；
不要每轮创建完全不同的 Python 对象结构和 kernel 分支。
```

### 10.3 FlashAttention 的例子

FlashAttention 支持 `supports_update_block_table=True`。

因此同一轮多个 KV group 如果只差 block table，就可以复用 metadata，再替换：

```text
block_table
slot_mapping
```

这对 hybrid model / KV sharing 场景很重要。

---

## 11. spec decode 下为什么要 unpadded metadata

`_build_attention_metadata()` 返回两个对象：

```text
attn_metadata
spec_decode_common_attn_metadata
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2222`

`attn_metadata` 给主模型 forward 使用；`spec_decode_common_attn_metadata` 会给 drafter / proposer 使用。

如果 full cudagraph 导致 metadata 被 padding：

```text
num_reqs != num_reqs_padded
或
num_tokens != num_tokens_padded
```

则会：

```text
spec_decode_common_attn_metadata = spec_decode_common_attn_metadata.unpadded(
    num_tokens, num_reqs
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2499`

原因注释也写得很直接：

```text
drafter 目前仍只使用 piecewise cudagraph，并且会直接修改 attention metadata；
因此它不想使用 padded attention metadata。
```

这说明：

```text
主模型 full graph 需要 padded metadata；
draft/spec 路径可能仍需要真实 unpadded metadata；
两者不能混用。
```

---

## 12. encoder / cross attention 的 capture 处理

`_build_attention_metadata()` 会按 KV cache group 调用：

```text
_get_encoder_seq_lens(..., for_cudagraph_capture=for_cudagraph_capture)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2455`

如果是 `CrossAttentionSpec`，encoder seq lens 会进入 common metadata。

capture 时有专门处理：

```text
During CUDA graph capture, we need to use realistic encoder lengths
so that max_seqlen_k is captured with the correct value.
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1873`

但真实 runtime dispatch 中，encoder-decoder 首轮有 encoder output 时会禁用 `FULL`，并且 `set_forward_context(skip_compiled=has_encoder_input)`。

这两层一起保证：

```text
可预测的 decoder step 可以走 cudagraph；
带 encoder input/output 的动态首轮不强行走 full graph。
```

---

## 13. ubatch / DBO 下的 metadata 切分

如果启用 microbatching / DBO，`_build_attention_metadata()` 的输出不是一个 dict，而是：

```text
list[dict[layer_name, AttentionMetadata]]
```

初始化：

```text
attn_metadata = [dict() for _ in range(len(ubatch_slices))]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2235`

每个 ubatch 会调用：

```text
split_attn_metadata(ubatch_slices, cm)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2491`

对 full cudagraph 来说，ubatch 也必须和 padded shape 对齐，所以 `execute_model()` 会区分：

```text
ubatch_slices
ubatch_slices_padded
```

并在 `pad_attn=True` 时使用 padded slices。

---

## 14. cascade attention 为什么和 FULL cudagraph 冲突

cascade attention 会把 attention 拆成：

```text
common prefix kernel
suffix kernel
```

其 metadata 包含：

```text
common_prefix_len
cu_prefix_query_lens
prefix_kv_lens
suffix_kv_lens
prefix_scheduler_metadata
```

这些字段取决于本轮请求之间共享 prefix 的长度。

`_compute_cascade_attn_prefix_lens()` 会按当前 batch 动态计算 common prefix。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2511`

因此真实 dispatch 时，如果使用 cascade attention：

```text
invalid_modes={CUDAGraphMode.FULL}
```

也就是说：

```text
cascade attention 可以和 PIECEWISE 或 eager 共存；
但不进入 full cudagraph capture/replay。
```

---

## 15. ForwardContext 中 metadata 的作用

`_build_attention_metadata()` 生成的结果不会直接作为参数传给每个 attention layer，而是放入 `ForwardContext`：

```text
set_forward_context(
  attn_metadata,
  vllm_config,
  cudagraph_runtime_mode=cudagraph_mode,
  batch_descriptor=batch_desc,
  slot_mapping=slot_mappings,
  ubatch_slices=ubatch_slices_padded,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`

`ForwardContext` 中的关键字段：

```text
attn_metadata
slot_mapping
cudagraph_runtime_mode
batch_descriptor
ubatch_slices
skip_compiled
```

位置：`code/vllm/vllm/forward_context.py:128`

attention layer 在 forward 时从 context 中按 layer name 取自己的 metadata。

这有两个好处：

```text
1. ModelRunner 不需要把 metadata 显式塞进每层 forward 参数；
2. CUDAGraphWrapper 和 attention backend 看到的是同一个 runtime context，dispatch 决策一致。
```

---

## 16. capture / replay 时哪些内容固定，哪些内容可变

可以按“对象结构、tensor shape、tensor 内容”三层理解。

### 16.1 必须固定的内容

```text
metadata dataclass 类型
关键 tensor shape
kernel routine 分支
wrapper / plan 类型
batch_descriptor 对应的 padded num_tokens / num_reqs
max_query_len 对应的 decode/prefill routine
```

例如：

```text
capture 是 uniform decode，replay 不能突然变成 mixed prefill-decode；
capture 是 FULL metadata，replay 不能换成 cascade metadata；
FlashInfer cudagraph decode wrapper 不能换成普通 decode wrapper。
```

### 16.2 可以更新的内容

```text
block_table 内容
slot_mapping 内容
seq_lens 内容
query_start_loc 内容
scheduler_metadata 内容
paged_kv_indices 内容
paged_kv_last_page_len 内容
```

前提是：

```text
tensor 地址和 shape 稳定；
backend 允许用同一 metadata 对象/同一 wrapper 表达这些变化；
padding 区域被安全填充。
```

### 16.3 需要特殊防护的内容

```text
padding request 的 block_table
padding token 的 slot_mapping
FA scheduler_metadata 未使用区域
FlashInfer paged_kv buffers
async spec decode 下 CPU/GPU seq_lens 权威来源
DCP local seq_lens
```

这些字段如果处理不好，常见错误不是“无法 capture”，而是：

```text
replay 读到旧 block；
padding token 写入真实 KV；
attention kernel 走错 routine；
无效 scheduler slot 覆盖输出；
CPU/GPU metadata 不一致。
```

---

## 17. 常见场景串起来看

### 17.1 普通 decode + FULL cudagraph

```text
uniform_decode=True
max_query_len=1
num_tokens_padded = captured size
num_reqs_padded = captured request count
pad_attn=True
```

metadata 特点：

```text
query_start_loc / seq_lens 按 padded req 数切片；
slot_mapping 按 padded token 数切片，padding 部分为 -1；
block_table padding rows 为 NULL_BLOCK_ID；
backend build 出 decode routine 对应 metadata；
ForwardContext 中 cudagraph_runtime_mode=FULL。
```

### 17.2 spec decode 验证步 + FULL cudagraph

```text
uniform_decode=True
max_query_len=1 + num_speculative_tokens
```

metadata 特点：

```text
主模型可以用 padded metadata；
drafter/proposer 可能需要 unpadded spec_decode_common_attn_metadata；
backend 必须至少支持 UNIFORM_BATCH。
```

### 17.3 mixed prefill-decode + FULL_AND_PIECEWISE

```text
uniform_decode=False
max_query_len=最长 prefill chunk
```

典型 dispatch：

```text
FULL decode key 不匹配；
PIECEWISE key 命中；
attention 不进入 full graph；
metadata 按真实 token/requires 构造，或只为 eager attention 构造。
```

### 17.4 cascade attention

```text
common_prefix_len > 0
```

metadata 特点：

```text
会生成 prefix/suffix 相关字段；
FULL 被禁用；
不会作为 full cudagraph capture 的 metadata 形态。
```

### 17.5 encoder-decoder 首轮

```text
has_encoder_input=True
has_encoder_output=True
```

metadata 特点：

```text
cross attention group 需要 encoder_seq_lens；
FULL 被禁用；
skip_compiled=True；
后续纯 decoder step 才可能进入 cudagraph。
```

---

## 18. 最小伪代码

### 18.1 execute_model 中的 attention metadata

```text
cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = \
    _determine_batch_execution_and_padding(...)

pad_attn = cudagraph_mode == CUDAGraphMode.FULL

slot_mappings_by_group, slot_mappings = _get_slot_mappings(
    num_tokens_padded = num_tokens_padded if pad_attn else num_tokens_unpadded,
    num_reqs_padded = num_reqs_padded if pad_attn else num_reqs,
    num_tokens_unpadded = num_tokens_unpadded,
)

attn_metadata, spec_common = _build_attention_metadata(
    num_tokens = num_tokens_unpadded,
    num_tokens_padded = num_tokens_padded if pad_attn else None,
    num_reqs = num_reqs,
    num_reqs_padded = num_reqs_padded if pad_attn else None,
    max_query_len = max_num_scheduled_tokens,
    slot_mappings = slot_mappings_by_group,
)

with set_forward_context(attn_metadata, cudagraph_runtime_mode=cudagraph_mode, ...):
    self.model(...)
```

### 18.2 capture 中的 attention metadata

```text
_dummy_run(..., cudagraph_runtime_mode=FULL, is_graph_capturing=True):
  max_query_len = uniform_decode_query_len if uniform_decode else num_tokens
  cudagraph_mode, batch_desc = _determine_batch_execution_and_padding(...)

  attn_metadata = _build_attention_metadata(
      num_tokens = num_tokens_unpadded,
      num_tokens_padded = batch_desc.num_tokens,
      num_reqs = num_reqs,
      num_reqs_padded = batch_desc.num_reqs,
      max_query_len = max_query_len,
      for_cudagraph_capture = True,
  )

  with set_forward_context(..., cudagraph_runtime_mode=FULL, batch_descriptor=batch_desc):
      self.model(...)
```

### 18.3 builder 分支

```text
if for_cudagraph_capture:
    metadata = builder.build_for_cudagraph_capture(cm)
elif can_reuse_metadata_and_update_block_table:
    metadata = builder.update_block_table(cached, cm.block_table, cm.slot_mapping)
else:
    metadata = builder.build(common_prefix_len, cm, ...)
```

---

## 19. 一句话总结

```text
attention metadata 是 full CUDA Graph 能否安全 replay 的核心约束：ModelRunner 必须把真实动态 batch 规整成固定 padded metadata，backend builder 必须把这些字段转换成稳定的 kernel plan / wrapper / tensor 结构；凡是 metadata 形态无法固定的场景，就会退到 PIECEWISE 或 eager，而不是强行 replay。
```
