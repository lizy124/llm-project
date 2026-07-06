# 06. ModelRunner 如何准备输入和 attention metadata？

源码位置：

- `code/vllm/vllm\v1\worker\gpu_model_runner.py`
- `code/vllm/vllm\v1\worker\gpu_input_batch.py`
- `code/vllm/vllm\v1\worker\gpu\attn_utils.py`
- `code/vllm/vllm\v1\worker\gpu\block_table.py`
- `code/vllm/vllm\forward_context.py`
- `code/vllm/vllm\v1\attention\backend.py`

本问题关注：`SchedulerOutput` 进入 `GPUModelRunner.execute_model()` 之后，`ModelRunner` 如何把请求级状态变成本轮 forward 需要的张量和 metadata，具体包括 `input_ids / inputs_embeds / positions / slot_mapping / block_table / attention metadata / spec decode metadata / multimodal metadata` 等；这些数据分别从哪里来、在哪一层组装、哪些字段只为某些模型或 attention backend 服务。

---

## 1. 一句话回答

`ModelRunner` 的输入准备不是“一次性造出一个大 batch”，而是分成三步：

```text
1. _update_states()
   → 把 SchedulerOutput 同步到 Worker 侧持久 batch

2. _prepare_inputs()
   → 把 batch 里的请求状态整理成本轮 forward 的 input_ids / positions / logits_indices / spec_decode_metadata

3. _build_attention_metadata()
   → 依据 block table、slot mapping、seq_lens、positions、ubatch / cudagraph / spec decode 等信息
     组装 attention backend 所需的 per-layer metadata
```

如果再往前看一步，`_preprocess()` 还会负责把多模态输入、encoder-decoder 输入、prompt_embeds、M-RoPE / XD-RoPE 位置等特殊路径折叠进统一的 forward 输入里。

所以可以把主链路概括为：

```text
SchedulerOutput
  → _update_states()
  → _prepare_inputs()
  → _determine_batch_execution_and_padding()
  → _get_slot_mappings()
  → _build_attention_metadata()
  → _preprocess()
  → set_forward_context(...)
  → _model_forward()
```

---

## 2. 先给结论：这几类数据分别在哪里准备

### 2.1 请求级状态先落到 InputBatch

`GPUModelRunner` 不是每轮都重建全部状态，而是维护一份持久的 `InputBatch`。`_update_states()` 会把 `SchedulerOutput` 里的新增、继续、结束、抢占、重排等变化同步进去。

`InputBatch` 里最关键的几类信息是：

- `req_id_to_index` / `_req_ids`：当前 batch 中有哪些请求，以及它们在 batch 里的行号
- `token_ids_cpu_tensor`：每个请求的 token 序列缓存
- `num_prompt_tokens` / `num_computed_tokens` / `num_tokens_no_spec`：请求推进到哪里了
- `block_table`：每个请求映射到哪些 KV blocks
- `sampling_metadata`：采样相关参数
- `spec_token_ids`：spec decode 相关 token 占位
- `pooling_params` / `pooling_states`：池化模型状态

对应实现见 `gpu_input_batch.py:91` 开始，以及 `CachedRequestState` 在 `gpu_input_batch.py:34`。

### 2.2 _prepare_inputs() 负责把持久状态压成本轮输入

`_prepare_inputs()` 主要产出两样东西：

```text
logits_indices
spec_decode_metadata
```

同时它还会把这些关键张量准备好：

- `input_ids`
- `inputs_embeds`
- `positions`
- `query_start_loc`
- `seq_lens`
- `num_computed_tokens`
- `num_accepted_tokens`
- `discard_request_mask`
- `req_indices`
- `num_scheduled_tokens`
- `slot_mapping`

其中真正给模型 forward 用的 `input_ids / inputs_embeds / positions` 是在 `_preprocess()` 里选最终形态；而 `logits_indices` 则决定哪些位置要拿去做 logits / sampling。

### 2.3 _build_attention_metadata() 负责把 batch 状态翻译成 attention backend 可消费的 metadata

attention backend 不直接读 `InputBatch`，它们读的是 `CommonAttentionMetadata` 和由具体 `AttentionMetadataBuilder` 构造的 per-layer metadata。

`_build_attention_metadata()` 会把这些公共输入整理出来：

- `query_start_loc`
- `seq_lens`
- `max_seq_len`
- `max_query_len`
- `block_table_tensor`
- `slot_mapping`
- `positions`
- `dcp_local_seq_lens`
- `encoder_seq_lens`
- `mm_req_doc_ranges`
- `is_prefilling`

然后按 KV cache group、attention group、ubatch、spec decode 等维度调用 builder。

### 2.4 _preprocess() 负责把特殊输入路径合并成最终 forward 输入

`_preprocess()` 处理的是“模型真正 forward 前最后一公里”的问题：

- 文本模型：直接用 `input_ids`
- 多模态模型：把 token ids 和 multimodal embeddings 合成 `inputs_embeds`
- prompt_embeds：把部分位置保留为外部 embedding
- encoder-decoder：补上 `encoder_outputs`
- M-RoPE / XD-RoPE：选择对应的 `positions`
- PP 非首 rank：把 `intermediate_tensors` 同步到当前 rank

---

## 3. 整体流程图

可以把输入准备理解成下面这条线：

```text
SchedulerOutput
  → _update_states()
      → 维护 self.requests
      → 维护 self.input_batch
      → 更新 block_table / token_ids / sampling / pooling / spec 状态

  → _prepare_inputs()
      → 计算 req_indices / positions / query_start_loc
      → 从 InputBatch 取 token ids
      → 计算 logits_indices
      → 计算 spec_decode_metadata
      → 计算 seq_lens / discard mask / num_accepted_tokens

  → _determine_batch_execution_and_padding()
      → 决定 cudagraph / padding / DP 对齐 / batch_descriptor

  → _get_slot_mappings()
      → 从 block_table 生成 slot_mapping

  → _build_attention_metadata()
      → 构造 CommonAttentionMetadata
      → 交给每个 attention builder 生成 per-layer metadata

  → _preprocess()
      → 合并 multimodal / prompt_embeds / encoder-decoder / PP 输入
      → 最终得到 input_ids / inputs_embeds / positions / model_kwargs

  → set_forward_context(...)
      → 把 attention metadata 和 slot mapping 放进 forward context

  → _model_forward()
```

---

## 4. _update_states() 先把“本轮要执行什么”同步到持久 batch

这一层不负责真正构造 forward 输入，但它决定了后面准备什么。

位置：`gpu_model_runner.py:1127`

### 4.1 清理结束请求

`finished_req_ids` 会同时从：

- `self.requests`
- `self.input_batch`

中移除。

这样后续 `_prepare_inputs()` 看到的 batch 就只剩当前活跃请求。

### 4.2 补齐新请求

`scheduled_new_reqs` 会被转换成 `CachedRequestState` 并加入 `self.requests`，再在后面加入 `InputBatch`。

新请求里最关键的来源包括：

- `prompt_token_ids` / `prompt_embeds`
- `mm_features`
- `sampling_params`
- `block_ids`
- `num_computed_tokens`
- `lora_request`
- `pooling_params`

### 4.3 更新现有请求

对 `scheduled_cached_reqs`，`_update_states()` 会更新：

- `num_computed_tokens`
- `output_token_ids`
- `block_ids`
- spec decode 相关的 `prev_num_draft_len`
- async scheduling 下的 draft / accepted token 状态

### 4.4 先更新，再准备输入

这一步很重要，因为 `_prepare_inputs()` 读取的全是 `self.input_batch` 里的当前态。

也就是说：

```text
InputBatch 不是 SchedulerOutput 的原始副本，
而是 SchedulerOutput 经过 _update_states() 归一后的执行态。
```

---

## 5. _prepare_inputs() 如何生成本轮 token 级输入

位置：`gpu_model_runner.py:1889`

它的职责可以概括成一句话：

```text
把“每个请求当前已经走到哪里、这轮要走多少 token”
翻译成模型可直接消费的 token / position / sampling 索引张量。
```

### 5.1 先把 block table 提前拷到 GPU

```python
self.input_batch.block_table.commit_block_table(num_reqs)
```

这是一个小优化：先开始异步拷贝 block table，再继续做 CPU 侧的索引计算，让数据传输和 CPU 计算重叠。

### 5.2 构造 request 维度展开

如果本轮每个请求调度 token 数是：

```text
[2, 5, 3]
```

则：

```text
req_indices = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
```

`query_pos` 则给每个 token 一个在本请求内部的局部位置：

```text
[0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
```

这两个数组拼起来，就能推出每个 token 的全局位置。

### 5.3 计算 positions

```python
positions_np = num_computed_tokens_cpu[req_indices] + query_pos
```

它表达的是：

```text
当前 token 在各自请求中的绝对位置 = 已经算过的 token 数 + 本轮内偏移
```

如果是 M-RoPE 或 XD-RoPE，则这里还会额外计算专用 positions：

- `_calc_mrope_positions()`
- `_calc_xdrope_positions()`

### 5.4 从 token 缓存里取出 input_ids

`InputBatch.token_ids_cpu_tensor` 是二维缓存：

```text
[req_index, token_pos]
```

`_prepare_inputs()` 会把 `(req_indices, positions)` 映射成扁平索引，然后用 `torch.index_select` 取出：

```python
self.input_ids.cpu[:total_num_scheduled_tokens]
```

如果启用了 prompt_embeds，还会同步准备：

- `self.is_token_ids`
- `self.inputs_embeds`

### 5.5 计算 query_start_loc

`query_start_loc` 是 attention 里很常见的前缀和边界数组：

```text
[0, 2, 7, 10]
```

表示三条请求分别占用 2、5、3 个 token。

它会在后续 attention metadata 中被当作 query span 的边界。

### 5.6 计算 optimistic seq_lens

```python
optimistic_seq_lens = num_computed_tokens + num_scheduled_tokens
```

这个值是“乐观长度”：默认假设 draft token 都会被接受，或者当前轮的 token 都会成为有效序列长度。

它后面会用于：

- `max_seq_len` 的估计
- `discard_request_mask`
- attention metadata 构造

### 5.7 计算 discard mask

chunked prefill 或 spec decode 场景下，有些请求虽然参与了 batch，但不应该把当前轮采样结果当成最终输出。

这时 `discard_request_mask` 会标记这些请求，后面 sampling 结果会被丢掉或修正。

### 5.8 计算 spec decode metadata

如果 `scheduled_spec_decode_tokens` 不为空，`_prepare_inputs()` 会进入 spec decode 分支，构造 `SpecDecodeMetadata`，包括：

- `draft_token_ids`
- `num_draft_tokens`
- `cu_num_draft_tokens`
- `cu_num_sampled_tokens`
- `target_logits_indices`
- `bonus_logits_indices`
- `logits_indices`

这个 metadata 的作用是：

```text
告诉 sampler 哪些 logits 位置对应 draft token，
哪些位置对应 bonus token，
以及最终应该采样哪些位置。
```

### 5.9 产出 logits_indices

如果没有 spec decode：

```text
logits_indices = 每个请求最后一个 token 的位置
```

如果有 spec decode：

```text
logits_indices = draft + bonus token 对应的一组位置
```

后续 `compute_logits()` 和 `_sample()` 都依赖它。

### 5.10 LoRA hot-swap

如果启用了 LoRA，`_prepare_inputs()` 末尾会调用 `set_active_loras()`，把当前 batch 的 LoRA 激活状态同步好，避免 forward 时走错适配器。

---

## 6. block table 和 slot mapping 是怎么来的

`block_table` 是“请求 → KV blocks”的逻辑映射；`slot_mapping` 是“token 位置 → KV cache slot”的实际展开映射。

### 6.1 block table 在 InputBatch 里

`MultiGroupBlockTable` 维护多个 KV cache group 的 block 表。

位置：`gpu_block_table.py:223`（同目录为 `worker/block_table.py` 的旧实现；当前执行链路使用 `gpu/block_table.py`）

核心职责：

- `append_row()` / `add_row()`：更新每个请求的 block ids
- `commit_block_table()`：把 CPU 侧 block table 提交到 GPU
- `compute_slot_mapping()`：根据 query positions 计算 slot mapping

### 6.2 slot mapping 的语义

slot mapping 解决的是这个问题：

```text
第 i 个 token 在 KV cache 里对应哪个实际 slot？
```

对 paged attention 来说，这通常由：

```text
block_id + block 内 offset
```

共同决定。

### 6.3 _get_slot_mappings() 会为每个 KV cache group 构造 slot mapping

位置：`gpu_model_runner.py:3960`

它返回两套视图：

- `slot_mappings_by_gid`
- `slot_mappings_by_layer`

构造细节：

- encoder-only attention：slot mapping 直接置零或走特殊路径
- 其他 attention：从 `self.input_batch.block_table[kv_cache_gid]` 里取 `slot_mapping.gpu`
- padded 区域填 `-1`，给 CUDA graph 兼容使用

### 6.4 为什么还要区分 by-gid 和 by-layer

因为一个 KV cache group 里可能包含多个 attention layer，它们共享同一个 metadata；而 forward context 里则更方便按 layer name 访问。

所以内部通常是：

```text
gid 级 slot mapping → layer 级映射
```

---

## 7. _build_attention_metadata() 如何组装 attention backend 所需信息

位置：`gpu_model_runner.py:2208`

这是整条链路里最容易混淆的部分，因为它连接了：

- 请求级状态
- KV cache 布局
- attention backend
- cudagraph / ubatching
- spec decode / cascade attn / multimodal / encoder-decoder

### 7.1 先构造 CommonAttentionMetadata

`attn_utils.build_attn_metadata()` 定义了公共结构：

```python
CommonAttentionMetadata(
    query_start_loc,
    query_start_loc_cpu,
    seq_lens,
    seq_lens_cpu_upper_bound,
    max_seq_len,
    num_reqs,
    num_actual_tokens,
    max_query_len,
    block_table_tensor,
    slot_mapping,
    causal,
    dcp_local_seq_lens,
    positions,
    ...
)
```

位置：`attn_utils.py:384`

它可以理解为：

```text
所有 attention backend 都会用到的公共 batch 描述。
```

### 7.2 per-layer metadata 由 builder 构造

`AttentionMetadataBuilder` 是抽象基类，每种 backend 都会实现自己的 builder。

它通常接收：

- `CommonAttentionMetadata`
- 以及少量 backend 特定参数

然后返回某种 `AttentionMetadata` 实例。

### 7.3 按 KV cache group / attention group 构造

`_build_attention_metadata()` 会先按 KV cache group 循环，再按组内 attention group 循环。

这样做的原因是：

- 不同 group 可能有不同 block size / cache spec
- 不同 backend 可能对 metadata 形状要求不同
- 有些 layer 共享 metadata，有些不共享

### 7.4 builder 缓存与 update_block_table

如果 builder 支持 `supports_update_block_table`，那么不同 KV cache group 之间相同结构的 metadata 可以复用，只更新 block table，减少构造开销。

这也是为什么 `_build_attention_metadata()` 里会维护 `cached_attn_metadata`。

### 7.5 CUDA graph capture 路径

如果是 cudagraph capture，builder 会走 `build_for_cudagraph_capture()`。

此时常常需要：

- 更保守的 `max_seq_len`
- 固定形状的 padding
- 避免依赖动态分支

### 7.6 传给 attention backend 的常见字段

典型字段包括：

- `query_start_loc`
- `seq_lens`
- `max_seq_len`
- `max_query_len`
- `block_table_tensor`
- `slot_mapping`
- `positions`
- `causal`
- `is_prefilling`
- `dcp_local_seq_lens`
- `encoder_seq_lens`
- `mm_req_doc_ranges`

这些字段最终会决定：

```text
attention kernel 该看哪些 token、
从哪些 KV slot 取值、
哪些位置属于 prefill / decode、
哪些位置要做 padding 或局部可见性限制。
```

---

## 8. _preprocess() 如何把特殊输入路径合并成最终 forward 输入

位置：`gpu_model_runner.py:3426`

这一层是“模型输入整形层”。它决定 forward 最终拿到的是 `input_ids` 还是 `inputs_embeds`，以及要不要附带 `encoder_outputs`、`mm kwargs`、`intermediate_tensors`。

### 8.1 文本模型

最简单的路径：

```text
input_ids = self.input_ids.gpu[:num_input_tokens]
inputs_embeds = None
```

### 8.2 多模态模型

如果模型支持 multimodal inputs：

1. 先通过 `maybe_get_ec_connector_output()` 处理 encoder cache / connector
2. 再执行 `_execute_mm_encoder()`
3. 再通过 `_gather_mm_embeddings()` 把多模态 embedding 取出来
4. 最后把 token ids 和 multimodal embeddings 合成 `inputs_embeds`

这里有个关键原则：

```text
多模态输入不是单独一条通路，而是被折叠进统一的 inputs_embeds 流。
```

### 8.3 prompt_embeds

如果某些 prompt token 直接由外部 embedding 提供，那么 `_preprocess()` 会只对 token-id 位置做 embedding lookup，并保留 prompt_embeds 位置的已有值。

这也是为什么 `_prepare_inputs()` 里会维护 `is_token_ids`。

### 8.4 encoder-decoder

对 encoder-decoder 模型，`_preprocess()` 会额外调用 `_execute_mm_encoder()` 并把 `encoder_outputs` 塞进 `model_kwargs`。

### 8.5 M-RoPE / XD-RoPE

如果模型使用旋转位置编码的变体：

- `positions = self.mrope_positions.gpu[:, :num_input_tokens]`
- `positions = self.xdrope_positions.gpu[:, :num_input_tokens]`

否则直接用普通 `positions`。

### 8.6 PP / intermediate_tensors

非首个 pipeline stage 不会自己构造完整 token 输入，而是接收 `intermediate_tensors` 并同步 gather。

所以 `_preprocess()` 最终返回的不是单一 input，而是一组统一接口：

```text
input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_connector_output
```

---

## 9. forward context 里为什么还要放 attention metadata 和 slot mapping

`set_forward_context()` 在 `forward_context.py` 里定义。

它把这些信息放进一个线程/作用域级上下文中：

- `attn_metadata`
- `slot_mapping`
- `batch_descriptor`
- `ubatch_slices`
- `cudagraph_runtime_mode`
- `dp_metadata`
- `skip_compiled`

这样模型层在执行 attention op 时，不需要把这些参数层层手传。

可以把它理解为：

```text
forward_context 是模型 forward 的隐式参数区。
```

这也是为什么 `_build_attention_metadata()` 不能只返回一个对象就结束：它最终要进入 forward context，供 attention layer / backend 在执行时取用。

---

## 10. 几个最容易混淆的概念

### 10.1 input_ids 和 positions 不是一回事

- `input_ids`：token 本身的 ID
- `positions`：token 在序列中的位置

两者常常一起喂给模型，但职责完全不同。

### 10.2 block table 和 slot mapping 不是一回事

- `block_table`：请求当前持有哪些 KV blocks
- `slot_mapping`：当前 token 的绝对位置对应哪个 slot

前者偏“索引关系”，后者偏“展开后的位置映射”。

### 10.3 CommonAttentionMetadata 不是最终 kernel 参数

它只是公共输入容器。最终 attention backend 还会通过 builder 变成更具体的 per-layer metadata。

### 10.4 spec decode metadata 不等于 sampling metadata

- `SamplingMetadata`：采样参数
- `SpecDecodeMetadata`：spec decode 过程中 logits / draft / bonus token 的位置关系

两者会同时存在，但用途不同。

### 10.5 multimodal encoder 输入不直接替代所有 token 输入

多模态模型通常仍然需要 token ids、positions、slot mapping，只是部分位置的表示会被 embedding 或 encoder 输出覆盖。

---

## 11. 最终可以记成一张表

| 阶段 | 主要函数 | 核心产物 | 作用 |
|---|---|---|---|
| 同步请求状态 | `_update_states()` | `self.requests`、`self.input_batch` | 把 SchedulerOutput 变成 Worker 侧持久态 |
| 生成 token 输入 | `_prepare_inputs()` | `input_ids`、`positions`、`logits_indices`、`spec_decode_metadata` | 把 batch 压缩成本轮 token 级输入 |
| 决定执行形态 | `_determine_batch_execution_and_padding()` | `batch_descriptor`、`cudagraph_mode` | 决定是否 padding / ubatch / DP 对齐 |
| 生成 slot 映射 | `_get_slot_mappings()` | `slot_mappings_by_gid`、`slot_mappings_by_layer` | 把 block table 变成 slot 映射 |
| 生成 attention metadata | `_build_attention_metadata()` | `attn_metadata`、`spec_decode_common_attn_metadata` | 供 attention backend 使用 |
| 整理最终 forward 输入 | `_preprocess()` | `input_ids`、`inputs_embeds`、`positions`、`model_kwargs` | 合并多模态 / prompt_embeds / encoder-decoder / PP |
| 注入 forward 上下文 | `set_forward_context()` | `ForwardContext` | 让模型内部 attention 层取到 metadata |

---

## 12. 一句话总结

`ModelRunner` 的输入准备本质上是一个“从请求状态到 forward 张量和 attention 元数据的逐层翻译过程”：

```text
SchedulerOutput
  → 持久 batch 状态
  → token 级输入
  → slot mapping / block table
  → attention metadata
  → multimodal / spec decode / encoder-decoder 整形
  → forward context
  → 模型执行
```

如果只记住一句话，就是：

```text
InputBatch 负责“记住状态”，
_prepare_inputs 负责“算出这轮喂什么”，
_build_attention_metadata 负责“告诉 attention 怎么看”，
_preprocess 负责“把特殊输入折成模型能吃的形状”。
```
