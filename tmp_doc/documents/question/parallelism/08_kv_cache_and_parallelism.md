# 08. KV cache 如何受并行策略影响？

源码位置：

- `vllm/vllm/config/model.py`
- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/model_executor/layers/attention/attention.py`
- `vllm/vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/vllm/v1/kv_cache_interface.py`
- `vllm/vllm/v1/core/kv_cache_utils.py`
- `vllm/vllm/v1/core/kv_cache_manager.py`
- `vllm/vllm/v1/worker/gpu_input_batch.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/block_table.py`
- `vllm/vllm/v1/worker/utils.py`
- `vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/flash_attn.py`
- `vllm/vllm/v1/attention/ops/common.py`

本问题关注：KV cache 在 TP / PP / DP / EP / CP 下如何放置和访问，`block table / slot mapping` 是否随并行维度变化，KV cache shape 和 layout 如何由 attention backend 与并行配置共同决定，以及 KV connector / external KV transfer 如何适配并行执行。

---

## 1. 一句话回答

KV cache 不是一个全局共享的大缓存，而是 **每个执行 rank 根据自己的并行身份持有本地 KV cache**。

可以先压缩成这条链路：

```text
ModelConfig / ParallelConfig
  → 计算本 rank 的 attention heads / KV heads / layer 范围 / CP rank
  → Attention.get_kv_cache_spec()
      生成本 rank、本 layer 的 KV cache spec
  → KVCacheManager / KVCacheConfig
      分配逻辑 blocks 和物理 cache tensor
  → ModelRunner / InputBatch
      维护 block table，计算 slot mapping
  → Attention metadata / ForwardContext
      把 block table、slot mapping、seq_lens、dcp_local_seq_lens 传给 backend
  → Attention backend
      在本 rank KV cache 上写入本地 K/V，读取本地或 CP 分片 K/V
      必要时通过 TP/CP/PP/DP 周边通信完成整体执行
```

一句话记忆：

```text
TP 决定本 rank 有多少 KV heads，PP 决定本 rank 有哪些 layers，DP 决定有多少份独立 KV cache replica，CP/DCP/PCP 决定同一请求的 KV context 是否按 position 分散到多个 ranks。
```

---

## 2. 本文要回答的问题

```text
TP 下 KV heads 是否按 rank 分片？
GQA / MQA 下 total KV heads 小于 TP size 怎么办？
PP 下哪些 stage 持有 KV cache？
DP 下不同 replica 的 KV cache 是否共享？
EP 会不会改变 attention KV cache？
DCP / PCP 如何改变 KV block、slot mapping 和本地 seq_lens？
block table / slot mapping 在并行下分别表示什么？
KV cache shape 为什么由 attention backend 决定？
KV cache layout 和 get_kv_cache_stride_order 有什么关系？
KV connector / external KV transfer 如何适配并行 rank？
```

---

## 3. 最小主链路

KV cache 与并行策略的关系可以分成六层：

```text
1. 配置层
   ParallelConfig 定义 TP / PP / DP / EP / DCP / PCP。

2. 模型层
   ModelConfig 根据 parallel_config 计算本 rank num_heads / num_kv_heads / layer 范围。

3. KV spec 层
   Attention / MLAAttention 为本 rank 的 attention layer 生成 KVCacheSpec。

4. KV 分配层
   KVCacheManager 按 request 分配逻辑 blocks；Worker 按 KVCacheConfig 分配物理 tensor。

5. batch 映射层
   InputBatch.block_table 记录 request → block ids；slot_mapping 记录 token → local KV slot。

6. backend 执行层
   attention backend 根据 metadata 在本 rank KV cache 上更新和读取 KV。
```

对应到源码主路径：

```text
ModelConfig.get_num_kv_heads()
ModelConfig.get_layers_start_end_indices()
  → Attention.get_kv_cache_spec()
  → KVCacheSpec.page_size_bytes / max_memory_usage_bytes()
  → GPUWorker.initialize_from_config()
  → GPUModelRunner.initialize_kv_cache()
  → _reshape_kv_cache_tensors()
  → bind_kv_cache()
  → InputBatch.block_table.append_block_ids()
  → BlockTables.compute_slot_mappings()
  → CommonAttentionMetadata(block_table_tensor, slot_mapping, dcp_local_seq_lens)
  → Attention.forward() / backend impl.forward()
```

---

## 4. KV cache 的几层含义先分清

### 4.1 KVCacheSpec：描述一层 cache 应该长什么样

`KVCacheSpec` 是 per-layer 的 cache 规格。

常见 attention spec 包括：

```text
FullAttentionSpec
SlidingWindowSpec
MLAAttentionSpec
SlidingWindowMLASpec
ChunkedLocalAttentionSpec
```

普通 attention 的核心字段是：

```text
block_size
num_kv_heads
head_size
head_size_v
dtype
kv_quant_mode
page_size_padded
indexes_kv_by_block_stride
```

位置：`vllm/vllm/v1/kv_cache_interface.py:95`、`vllm/vllm/v1/kv_cache_interface.py:159`、`vllm/vllm/v1/kv_cache_interface.py:205`

它表达的是：

```text
一个 KV block 有多少 token；
每个 token 有多少本地 KV heads；
每个 head 的维度是多少；
K/V 是否量化；
一个 page / block 要占多少 bytes。
```

### 4.2 KVCacheConfig：描述本 worker 实际有哪些 cache group

KV cache 不一定每层一组。vLLM 会把 cache spec 相同或可合并的 layers 放进 KV cache groups。

可以理解为：

```text
KVCacheSpec：单层需要什么 cache。
KVCacheGroupSpec：一组 layers 共享同一种 cache 规格和 metadata 构造逻辑。
KVCacheConfig：当前 worker/rank 的全部 KV cache group 配置。
```

多 KV cache group 常见于：

```text
- sliding window + full attention 混合；
- MLA / full attention 混合；
- Mamba / attention hybrid；
- cross-layer KV sharing；
- encoder-decoder / cross attention 特殊路径。
```

### 4.3 物理 KV cache tensor：真正的 GPU memory

Worker 初始化时会拿到已经规划好的 KV cache tensor，然后 reshape 成 backend 需要的形状。

关键路径在 GPUModelRunner 中很清楚：

```text
_raw tensor
  → 根据 kv_cache_spec.page_size_bytes 算 num_blocks
  → attn_backend.get_kv_cache_shape(...)
  → attn_backend.get_kv_cache_stride_order()
  → _reshape_attention_kv_cache(...)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:7072`

这说明：

```text
KV cache 的逻辑 block/page 由 KVCacheSpec 决定；
KV cache 的张量 shape / stride 由 attention backend 决定；
最终 attention layer 通过 bind_kv_cache 拿到自己的 tensor view。
```

### 4.4 block table：请求到 blocks 的映射

`block table` 是请求级映射：

```text
request row → block ids
```

在新 GPU block table 实现里：

```text
BlockTables.block_tables[group_id][req_index, block_index] = block_id
```

位置：`vllm/vllm/v1/worker/block_table.py:17`

它不直接告诉某个 token 写哪个 slot，而是告诉：

```text
某个 request 在某个 KV cache group 下拥有哪些 blocks。
```

### 4.5 slot mapping：token 到本地 KV slot 的映射

`slot mapping` 是 token 级映射：

```text
token position → local KV cache slot id
```

普通情况下公式可以理解为：

```text
block_index  = position // block_size
block_offset = position % block_size
block_number = block_table[request, block_index]
slot_id      = block_number * block_size + block_offset
```

位置：`vllm/vllm/v1/worker/block_table.py:283`

attention update 时通常只消费 slot mapping，不再重新理解 request 的 block 表。

---

## 5. TP 如何影响 KV cache

### 5.1 TP 先改变本 rank 的 attention heads

`ModelConfig.get_num_attention_heads()` 返回的是本 rank 的 local heads：

```text
local_num_heads = total_num_attention_heads // tensor_parallel_size
```

位置：`vllm/vllm/config/model.py:1272`

这意味着传给 `Attention(...)` 的 `num_heads` 通常已经不是全局值。

### 5.2 TP 改变本 rank 的 KV heads

普通 attention 的 KV heads 由：

```python
return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

决定。

位置：`vllm/vllm/config/model.py:1259`

关键点是 `max(1, ...)`：

```text
如果 total KV heads >= TP：
  每个 rank 持有 total_num_kv_heads / TP 个 KV heads。

如果 total KV heads < TP：
  每个 rank 至少持有 1 个 KV head。
  KV heads 会在多个 TP ranks 上复制或形成本地可用视图。
```

这对 MQA / GQA 很重要。

### 5.3 MLA 的 KV heads 固定为 1

如果模型使用 MLA：

```text
get_num_kv_heads() 直接返回 1。
```

位置：`vllm/vllm/config/model.py:1261`

因为 MLA decode 路径更接近 MQA：KV cache 存的是 compressed latent KV，而不是传统每个 query head 对应一组 K/V。

### 5.4 TP 后 KVCacheSpec 记录的是本 rank local KV heads

`Attention.get_kv_cache_spec()` 会把 `self.num_kv_heads` 写入 spec：

```text
FullAttentionSpec(
  block_size=...,
  num_kv_heads=self.num_kv_heads,
  head_size=self.head_size,
  head_size_v=self.head_size_v,
  dtype=...,
)
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:581`

所以：

```text
KV cache 的物理形状是 per-rank 的。
TP 已经体现在 num_kv_heads 上。
```

### 5.5 TP 下每个 rank 写自己的 KV cache

attention forward 中会把 K/V reshape 成本地形状：

```text
key   → [num_tokens, local_num_kv_heads, head_size]
value → [num_tokens, local_num_kv_heads, head_size_v]
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:495`

然后通过当前 layer 的 `kv_cache` 和 `slot_mapping` 更新本 rank KV cache。

因此 TP 下不是所有 rank 写同一份 KV cache，而是：

```text
每个 TP rank 持有自己本地 heads 对应的 KV cache。
```

### 5.6 TP 不改变 block id 的逻辑语义

TP 改变的是 head 维度，不是 request 的 block 分配语义。

也就是说：

```text
block_id = 第几个 KV block
```

在 TP rank 上仍然表示本 rank 本地 KV cache tensor 中的 block。

但同一个 request 在不同 TP ranks 上会有各自本地的 KV cache 内容：

```text
rank 0 block 10：保存 rank 0 负责的 KV heads；
rank 1 block 10：保存 rank 1 负责的 KV heads；
...
```

block id 可以相同，但物理 tensor 是 rank-local 的。

---

## 6. GQA / MQA 下 TP 与 KV cache 的特殊点

### 6.1 MHA

MHA 通常满足：

```text
total_num_attention_heads == total_num_kv_heads
```

TP 后：

```text
local_q_heads  = total_q_heads / TP
local_kv_heads = total_kv_heads / TP
```

每个 rank 的 KV cache 保存自己那组 KV heads。

### 6.2 GQA

GQA 通常满足：

```text
total_q_heads > total_kv_heads
```

TP 后需要看 `total_kv_heads` 和 `TP` 的大小关系：

```text
如果 total_kv_heads >= TP：
  KV heads 按 TP rank 分片。

如果 total_kv_heads < TP：
  每个 TP rank 至少有 1 个 KV head，KV heads 会被复制到多个 rank 的本地 cache 视图。
```

backend 内部常用本地值计算：

```text
num_queries_per_kv = local_num_heads // local_num_kv_heads
```

### 6.3 MQA

MQA 可以看作：

```text
total_num_kv_heads = 1
```

如果 `TP > 1`，`max(1, total_num_kv_heads // TP)` 仍然让每个 rank 有 1 个本地 KV head。

否则某些 rank 会没有 KV head，attention kernel 无法正常执行。

### 6.4 容易误解的点

```text
误解：TP=8 且 total_kv_heads=1 时，只有一个 rank 持有 KV cache。

实际：每个 TP rank 至少有一个 local KV head 视图，KV cache 是 rank-local 的。
```

---

## 7. PP 如何影响 KV cache

### 7.1 PP 切的是 layers，不是 heads

Pipeline parallel 根据当前 rank 计算负责的 layer 范围：

```text
pp_rank = (rank // tensor_parallel_size) % pipeline_parallel_size
start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
```

位置：`vllm/vllm/config/model.py:1282`

所以 PP 对 KV cache 的直接影响是：

```text
当前 PP rank 只构造 / 持有自己 stage 内 attention layers 的 KV cache。
```

### 7.2 每个 PP stage 只绑定自己的 KV cache

`bind_kv_cache()` 会把已分配的 `kv_caches[layer_name]` 绑定到 forward context 里的 attention layer：

```text
forward_context[layer_name].kv_cache = kv_cache
```

位置：`vllm/vllm/v1/worker/utils.py:462`

这意味着：

```text
如果某个 layer 不在当前 PP stage，当前 rank 不会持有它的 attention layer，也不会绑定它的 KV cache。
```

### 7.3 PP 下请求会穿过多个 stage，但 KV cache 不跨 stage 合并

PP forward 大致是：

```text
first PP rank：接收 token / embeddings，执行前几层；
middle PP rank：接收 intermediate tensors，执行中间层；
last PP rank：执行末尾层，并负责 logits / sampling。
```

KV cache 的归属仍然是：

```text
stage 0 的 layers → stage 0 ranks 的 KV cache；
stage 1 的 layers → stage 1 ranks 的 KV cache；
...
```

不会出现一个 PP rank 直接读取另一个 PP rank 某层 KV cache 的常规路径。

### 7.4 PP 对 block table 的影响

从 request 语义看，同一个请求在各 PP stage 都需要对应的 KV blocks。

但从物理视角看：

```text
每个 PP stage 的 block table 指向当前 stage 本地 layers 的 KV cache。
```

因此 PP 下要注意：

```text
block id 是当前 rank / 当前 stage 本地 KV cache 的 block id；
不是跨 PP stages 的全局物理内存地址。
```

### 7.5 PP 与 speculative decoding

源码里 speculative draft model 当前通常放在 last PP rank：

```text
if speculative_config and get_pp_group().is_last_rank:
  initialize drafter / rejection sampler
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:547`

这说明 PP 不只是影响 KV cache layer 归属，也影响：

```text
哪些 rank 执行 logits / sampling / speculative decode 相关逻辑。
```

但 attention KV cache 本身仍按 layer 所在 PP stage 分配。

---

## 8. DP 如何影响 KV cache

### 8.1 DP replica 的 KV cache 是独立的

Data parallel 扩展的是 replica / batch 维度。

每个 DP replica 有自己的：

```text
running requests
InputBatch
KVCacheManager
BlockPool
KV cache tensors
block table
slot mapping
attention metadata
```

因此：

```text
DP 不共享同一份 KV cache。
```

### 8.2 DP 不改变单个 attention backend 的 KV shape

对单个 replica 内的某个 rank：

```text
local_num_heads / local_num_kv_heads 仍由 TP 决定；
layer 范围仍由 PP 决定；
context 切分仍由 DCP / PCP 决定。
```

DP 主要增加的是：

```text
系统中有多少份独立的 KV cache 副本在并行服务不同请求。
```

### 8.3 DP 外层需要同步执行状态，但不是同步 KV tensor

DP 可能需要同步：

```text
- 哪些 replica 仍有未完成请求；
- pause / stop / engine progress；
- profile / padding / 0-token step；
- 外部 launcher 或外部负载均衡下的调度协调。
```

但常规推理中不会把一个 replica 的 KV cache tensor 直接同步给另一个 replica。

### 8.4 DP 与 KV connector 的关系

如果启用了 external KV transfer，DP replica-local 的原则仍然存在，但 connector 需要知道：

```text
哪个 request；
哪些 block ids；
属于哪个 worker / rank；
load / save 是否完成；
是否有 invalid block ids；
```

Worker 侧会在 forward 前后调用 KV connector，并把 `kv_connector_worker_meta` 等信息带回 Scheduler。

位置：`vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:78`

所以：

```text
DP 本身不共享 KV cache；
KV connector 是额外的跨实例 / 跨进程 KV 传输机制，需要显式 metadata 协调。
```

---

## 9. EP 如何影响 KV cache

### 9.1 EP 主要作用在 MoE expert 层

Expert parallel 切分的是 MoE experts，不是 dense attention 的 KV cache。

因此 EP 对 attention KV cache 的直接影响通常是：

```text
没有直接改变 KV cache shape；
没有直接改变 block table / slot mapping 公式；
没有让 attention KV heads 按 expert rank 切分。
```

### 9.2 EP 与 attention 的交叉点在 hidden states 和执行一致性

MoE 模型中 attention 和 expert layers 交替出现：

```text
attention → MoE MLP → attention → MoE MLP ...
```

EP 会影响 attention 前后的 hidden states 如何 dispatch / combine，但 attention KV cache 仍由 attention layer 自己管理。

`parallel_state.py` 中 EP group 提供 expert dispatch / combine 相关通信能力。

位置：`vllm/vllm/distributed/parallel_state.py:1201`

### 9.3 DP + EP 场景需要更关注一致性

在 DP + EP 下，系统可能需要保证同一个 DP group 内 expert 输出、padding、profile run 等行为一致。

但这属于 MoE / hidden state 执行约束，不是 KV cache 分配公式变化。

可以总结为：

```text
EP 不直接改 KV cache；
EP 改的是 attention 周边 MoE 层的 expert dispatch / combine 和 hidden state 布局。
```

---

## 10. CP / DCP / PCP 如何影响 KV cache

### 10.1 CP 切的是同一请求的 context KV

Context Parallel 的核心是：

```text
把同一个 request 的 KV context 按 position 分散到多个 CP ranks。
```

这和 TP / PP / DP 都不一样：

```text
TP：切 heads / hidden / weights；
PP：切 layers；
DP：切 replica / requests；
CP：切同一个请求的 context positions。
```

### 10.2 DCP 复用 TP group，不增加 world size

`ParallelConfig.decode_context_parallel_size` 的注释说明：

```text
DCP does not change world size;
it reuses GPUs of TP group;
tp_size needs to be divisible by dcp_size.
```

位置：`vllm/vllm/config/parallel.py:339`

校验逻辑：

```text
tensor_parallel_size % decode_context_parallel_size == 0
```

位置：`vllm/vllm/config/parallel.py:498`

所以 DCP 和 TP 的关系是：

```text
TP 决定已有 rank 如何切权重 / heads；
DCP 复用 TP 相关 rank，在现有 rank mesh 上组织 context-parallel 通信域。
```

### 10.3 PCP 与 DCP 共同组成 total CP rank

`cp_kv_cache_interleave_size` 的注释定义：

```text
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
total_cp_world_size = pcp_world_size * dcp_world_size
```

位置：`vllm/vllm/config/parallel.py:359`

因此 CP-aware KV cache 布局看的是 total CP rank，而不是只看 DCP rank。

### 10.4 CP 会减少每个 rank 最坏 KV memory

`FullAttentionSpec.max_memory_usage_bytes()` 中：

```text
if dcp_world_size * pcp_world_size > 1:
  max_model_len = ceil(max_model_len / (dcp_world_size * pcp_world_size))
```

位置：`vllm/vllm/v1/kv_cache_interface.py:236`

含义是：

```text
同一个请求的上下文 KV 被多个 CP ranks 分摊；
每个 rank 只需要保存约 1 / total_cp_world_size 的 KV tokens。
```

### 10.5 SlidingWindowSpec 当前限制 DCP

`SlidingWindowSpec.max_memory_usage_bytes()` 中有断言：

```text
DCP not support sliding window.
```

位置：`vllm/vllm/v1/kv_cache_interface.py:528`

这说明 CP 不是一个能和所有 KV cache 类型无条件叠加的开关。

---

## 11. CP-aware slot mapping 如何改变写入位置

### 11.1 普通 slot mapping

普通情况下：

```text
block_index  = position // block_size
block_offset = position % block_size
slot_id      = block_number * block_size + block_offset
```

位置：`vllm/vllm/v1/worker/block_table.py:283`

### 11.2 CP 下 slot mapping 会判断 token 属不属于当前 rank

CP 下公式变成：

```text
block_index  = position // (block_size * CP_SIZE)
block_offset = position %  (block_size * CP_SIZE)

is_local = block_offset // CP_INTERLEAVE % CP_SIZE == cp_rank

rounds       = block_offset // (CP_INTERLEAVE * CP_SIZE)
remainder    = block_offset % CP_INTERLEAVE
local_offset = rounds * CP_INTERLEAVE + remainder
slot_id      = block_number * block_size + local_offset
```

如果不是当前 rank 负责的 token：

```text
slot_id = PAD_SLOT_ID
```

位置：`vllm/vllm/v1/worker/block_table.py:289`

这就是 CP 对 KV cache 写入最直接的影响：

```text
global position 不再总是写当前 rank；
只有属于当前 CP rank 的 positions 才写本地 KV cache。
```

### 11.3 cp_kv_cache_interleave_size 的直观理解

假设：

```text
CP_SIZE = 2
block_size = 8
```

如果：

```text
CP_INTERLEAVE = 1
```

则 token 级轮转：

```text
position 0 → rank 0
position 1 → rank 1
position 2 → rank 0
position 3 → rank 1
```

如果：

```text
CP_INTERLEAVE = 4
```

则每 4 个 token 轮转：

```text
position 0-3   → rank 0
position 4-7   → rank 1
position 8-11  → rank 0
position 12-15 → rank 1
```

所以 `cp_kv_cache_interleave_size` 控制的是：

```text
global positions 在 CP ranks 间交错存储的粒度。
```

---

## 12. DCP local seq lens 为什么需要单独计算

### 12.1 全局 seq_lens 不等于本 rank 可见 KV 长度

在 DCP 下：

```text
seq_lens：请求的全局上下文长度；
dcp_local_seq_lens：当前 DCP rank 本地实际保存 / 可见的上下文长度。
```

Worker 在构造 attention metadata 时会计算：

```text
get_dcp_local_seq_lens(
  optimistic_seq_lens,
  dcp_world_size,
  dcp_rank,
  cp_kv_cache_interleave_size,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2357`

然后写入：

```text
cm_base.dcp_local_seq_lens
cm_base.dcp_local_seq_lens_cpu
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2367`

### 12.2 attention backend 需要它做本地 context attention

DCP backend 不能只拿全局 `seq_lens`，否则会以为本 rank 拥有完整 KV context。

它需要知道：

```text
当前 rank 的本地 KV context 有多长；
block table 中哪些 blocks 对应本地 context；
本地 partial attention 应该覆盖哪些 positions。
```

因此 `CommonAttentionMetadata` 专门包含：

```text
dcp_local_seq_lens
dcp_local_seq_lens_cpu
```

位置：`vllm/vllm/v1/attention/backend.py:432`

---

## 13. block table 在 CP 下的语义变化

CP 下 block table 仍然是：

```text
request → block ids
```

但 block 的含义变成：

```text
当前 CP rank 本地 KV storage 中的 block。
```

global position 到 block table index 的计算使用：

```text
position // (block_size * CP_SIZE)
```

这表示：

```text
多个 CP ranks 共同覆盖一个全局 block span；
每个 rank 在这个 span 里只保存自己负责的 interleaved positions。
```

所以：

```text
CP 不只是 attention backend 的通信优化；
它从 slot mapping / block table 解释层就改变了 global position → local slot 的映射。
```

---

## 14. KV cache shape 为什么和 backend 相关

### 14.1 backend 必须定义 get_kv_cache_shape

`AttentionBackend` 抽象接口要求 backend 提供：

```text
get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)
```

位置：`vllm/vllm/v1/attention/backend.py:87`

原因是不同 backend 的 KV cache layout 不一样。

例如常见逻辑可能是：

```text
普通 paged attention：
  [2, num_blocks, block_size, num_kv_heads, head_size]

MLA：
  [num_blocks, block_size, compressed_kv_dim]

某些 backend：
  num_blocks 维度前置，或 heads 维度前置，或 K/V 分开。
```

具体 shape 不能由 Scheduler 推断，必须由 backend 提供。

### 14.2 stride order 决定物理内存布局

backend 还可以提供：

```text
get_kv_cache_stride_order()
```

位置：`vllm/vllm/v1/attention/backend.py:119`

它描述的是：

```text
逻辑 shape 的各维度在物理内存中的排列顺序。
```

例如逻辑 shape 是：

```text
[2, num_blocks, block_size, num_heads, head_size]
```

如果 stride order 返回：

```text
(1, 3, 0, 2, 4)
```

则物理布局更接近：

```text
[num_blocks, num_heads, 2, block_size, head_size]
```

这会影响：

```text
- block stride；
- connector 是否能按 blocks-first 传输；
- cross-layer uniform KV layout；
- backend 是否能 tolerate non-contiguous block dim。
```

### 14.3 indexes_kv_by_block_stride

`AttentionBackend.indexes_kv_by_block_stride()` 会根据 stride order 判断 backend 是否按 runtime block stride 读取 KV pages。

位置：`vllm/vllm/v1/attention/backend.py:204`

如果 backend 支持 blocks-first 或 layered stride order，它更容易和：

```text
- page size padding；
- packed backing tensor；
- KV connector 的 block 传输；
- cross-layer KV cache sharing；
```

组合。

### 14.4 KV cache tensor reshape 入口

GPUModelRunner 的 `_reshape_kv_cache_tensors()` 会：

```text
1. 根据 raw tensor 字节数算 num_blocks；
2. 根据 kv_cache_spec.block_size / kernel_block_size 算 kernel_num_blocks；
3. 调 attn_backend.get_kv_cache_shape；
4. 调 attn_backend.get_kv_cache_stride_order；
5. 构造最终 per-layer kv_cache tensor view。
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:7072`

所以：

```text
KV cache layout 是 attention backend 和 KVCacheSpec 共同决定的，不是固定格式。
```

---

## 15. attention metadata 如何承接 KV cache 与并行

`CommonAttentionMetadata` 是 backend builder 的公共输入。

和 KV cache / 并行最相关的字段包括：

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
encoder_seq_lens
dcp_local_seq_lens
positions
is_prefilling
seq_lens_cpu_upper_bound
```

位置：`vllm/vllm/v1/attention/backend.py:393`

可以按职责理解：

```text
block_table_tensor：
  request → local blocks。

slot_mapping：
  current tokens → local KV slots。

seq_lens / max_seq_len：
  全局逻辑上下文长度。

dcp_local_seq_lens：
  DCP 下当前 rank 的本地上下文长度。

positions：
  global token positions，用于 RoPE、slot mapping、某些 backend metadata。

query_start_loc：
  batch 内每个 request 的 query span 边界。
```

注意：

```text
attention backend 不直接读 Scheduler 的 KVCacheManager；
它读的是 ModelRunner 构造好的 block table / slot mapping / metadata。
```

---

## 16. KV cache group / attention group 与并行的关系

### 16.1 为什么要分 group

不同 attention layers 可能有不同 KV cache spec 或 backend metadata 需求。

因此 vLLM 会形成：

```text
KV cache group：
  KV cache 分配和 block table 的单位。

attention group：
  attention metadata builder 和 backend 构造的单位。
```

### 16.2 多 group 下 block table / slot mapping 是按 group 组织的

`BlockTables` 内部维护：

```text
num_kv_cache_groups x [max_num_reqs, max_num_blocks]
```

以及：

```text
slot_mappings[num_kv_cache_groups, max_num_batched_tokens]
```

位置：`vllm/vllm/v1/worker/block_table.py:47`、`vllm/vllm/v1/worker/block_table.py:73`

这说明：

```text
同一个 request 在不同 KV cache groups 下可能有不同 block table；
同一个 token 在不同 group 下也可能映射到不同 slot。
```

### 16.3 CP + hybrid KV cache 有限制

Context Parallel 会改变 block 对齐和 slot mapping。

如果再叠加多 group / 多 block size hybrid KV cache，prefix cache hash、scheduler block size、slot mapping 对齐都会更复杂。

相关限制在 `resolve_kv_cache_block_sizes()` 和 CP 文档中体现：

```text
Hybrid KV cache groups with multiple block sizes do not support context parallelism。
```

位置：`vllm/vllm/v1/core/kv_cache_utils.py`

---

## 17. Prefix cache 在并行下怎么理解

### 17.1 Prefix cache 的 block hash 是逻辑层面的

Prefix cache 命中发生在 Scheduler / KVCacheManager 层。

它判断的是：

```text
某个请求前缀对应的 blocks 是否已经存在并可复用。
```

Worker / ModelRunner 只消费结果：

```text
num_computed_tokens
block_ids
new_block_ids_to_zero
```

### 17.2 TP 下 prefix cache 复用的是各 rank 本地 KV heads

TP rank 的 KV cache 是本地 heads 视图。

因此 prefix cache 命中可以理解为：

```text
同一个逻辑 prefix 在每个 TP rank 上复用对应本地 KV heads 的 blocks。
```

### 17.3 PP 下 prefix cache 复用各 stage 的 layers

PP stage 只持有自己 layers 的 KV cache。

因此同一请求的 prefix cache 在物理上分布在不同 PP stages：

```text
stage 0 复用 stage 0 layers 的 prefix KV；
stage 1 复用 stage 1 layers 的 prefix KV；
...
```

### 17.4 DCP 下 prefix cache 对齐粒度必须覆盖完整 CP group

启用 CP 时，Scheduler 使用更大的 scheduler block size：

```text
scheduler_block_size = cache_config.block_size * dcp * pcp
```

原因是：

```text
prefix cache 的逻辑 block 必须覆盖所有 CP ranks 的 interleaved positions；
否则不同 CP ranks 可能对同一个 prefix 是否完整产生不同判断。
```

---

## 18. KV connector / external KV transfer 如何适配并行

### 18.1 KV connector 包在 forward 上下文周围

Worker 侧 mixin 会在 forward 前：

```text
kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
kv_connector.start_load_kv(get_forward_context())
```

forward 后：

```text
kv_connector.wait_for_save()
kv_connector.get_finished(...)
kv_connector.get_block_ids_with_load_errors()
kv_connector.get_kv_connector_stats()
kv_connector.get_kv_connector_kv_cache_events()
kv_connector.build_connector_worker_meta()
kv_connector.clear_connector_metadata()
```

位置：`vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:78`

所以 KV connector 不是 attention forward 后的旁路补丁，而是：

```text
嵌在 ModelRunner 执行上下文里的 KV load/save 生命周期。
```

### 18.2 KV connector 需要 backend 支持

`AttentionBackend` 提供：

```text
supports_kv_connector()
```

默认返回 `True`，但某些平台 / backend 可能因为 KV layout 不兼容而不支持。

位置：`vllm/vllm/v1/attention/backend.py:276`

例如 ROCm 平台注释提到：

```text
ROCM_ATTN uses (2, num_blocks, ...) KV cache layout
which is incompatible with KV connectors that require blocks-first layout.
```

位置：`vllm/vllm/platforms/rocm.py`

### 18.3 external KV transfer 必须理解 rank-local KV cache

因为 KV cache 是 rank-local 的，external KV transfer 不能只说：

```text
request X 的 block 10
```

还必须隐含或显式处理：

```text
- TP rank：这份 block 对应哪些 local KV heads；
- PP rank：这份 block 对应哪些 layers；
- DP replica：这份 block 属于哪个 replica / worker；
- CP rank：这份 block 对应哪些 interleaved positions；
- KV cache group：这份 block 属于哪个 group；
- backend layout：这份 block 在 tensor 中如何排列。
```

### 18.4 connector shape 也依赖 backend

KV connector mixin 中也会调用 backend 的 KV cache shape / stride order 来构造传输视图。

位置：`vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:207`

这说明：

```text
external KV transfer 传输的不是抽象 token，而是 backend layout 下的 KV block bytes / tensor view。
```

### 18.5 0-token step 也可能推进 KV connector

如果本轮没有可执行 token，但有 KV transfer 需要推进，会走：

```text
kv_connector_no_forward()
```

位置：`vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:36`

因此：

```text
没有 forward 不等于完全不碰 KV 状态；
KV connector 仍可能 load / save / finalize。
```

---

## 19. 并行策略对 block table / slot mapping 的影响汇总

### 19.1 TP

```text
block table：
  逻辑上仍是 request → block ids。

slot mapping：
  仍是 token → local slot。

变化点：
  local slot 对应本 TP rank 的 local KV heads。
```

### 19.2 PP

```text
block table：
  当前 PP stage 本地 layers 的 request → block ids。

slot mapping：
  当前 stage 本地 attention layers 使用。

变化点：
  当前 rank 只持有本 stage layers 的 KV cache。
```

### 19.3 DP

```text
block table：
  每个 DP replica 独立维护。

slot mapping：
  每个 DP replica 独立计算。

变化点：
  不同 replica 不共享 KV cache。
```

### 19.4 EP

```text
block table：
  不因 EP 本身改变。

slot mapping：
  不因 EP 本身改变。

变化点：
  EP 影响 MoE hidden state dispatch / combine，不直接影响 dense attention KV cache。
```

### 19.5 CP / DCP / PCP

```text
block table：
  block 表示当前 CP rank 本地 storage 中的 block。

slot mapping：
  global position → local slot；非本 rank token → PAD_SLOT_ID。

变化点：
  同一个 request 的 context KV 被多个 CP ranks 分摊。
```

---

## 20. 并行策略对 KV cache shape 的影响汇总

### 20.1 TP 改变 num_kv_heads

```text
KV cache shape 中的 num_kv_heads 是 local_num_kv_heads。
```

例如普通 attention 可理解为：

```text
[num_blocks, block_size, local_num_kv_heads, head_size]
```

或 backend 自定义的等价布局。

### 20.2 PP 改变 layer 数量

```text
当前 rank 只为本 PP stage 的 layers 分配 KV cache。
```

它不直接改变单层 KV cache 的 head shape。

### 20.3 DP 增加 replica 数量

```text
DP 不改变单个 replica 内 KV cache shape；
它增加 KV cache 副本数量。
```

### 20.4 EP 不改变 attention KV cache shape

```text
EP 不直接改变 dense attention KV shape。
```

### 20.5 CP 减少本地可保存的 context tokens

```text
max_model_len_local ≈ ceil(max_model_len / total_cp_world_size)
```

但单个 local block 的 shape 仍由 backend 和 KVCacheSpec 决定。

### 20.6 backend 决定最终 tensor layout

最终 shape 入口是：

```text
attn_backend.get_kv_cache_shape(...)
attn_backend.get_kv_cache_stride_order(...)
```

因此即使并行参数相同，不同 backend 的物理 KV cache layout 也可能不同。

---

## 21. 从一次 forward 看 KV cache 与并行

可以把一次执行串成下面这条线：

```text
Scheduler
  → 为请求分配 / 复用 KV blocks
  → 输出 block_ids / num_computed_tokens / new_block_ids_to_zero / kv_connector_metadata

Worker / ModelRunner
  → _update_states()
      把 block_ids 写入 InputBatch / BlockTables
  → _prepare_inputs()
      计算 positions / query_start_loc / seq_lens
  → BlockTables.compute_slot_mappings()
      普通：position → block_id + offset
      CP：position → 当前 CP rank local slot 或 PAD_SLOT_ID
  → _build_attention_metadata()
      构造 block_table_tensor / slot_mapping / dcp_local_seq_lens
  → set_forward_context(...)
      把 metadata 和 per-layer slot mapping 放进 forward context

Attention layer
  → query/key/value reshape 成 local heads
  → 用 slot_mapping 更新本 rank KV cache
  → 用 block_table + seq_lens 读取历史 KV
  → DCP 时只读本地 context KV，并用 LSE 合并 partial outputs
```

这条链路里：

```text
Scheduler 管逻辑资源账本；
Worker 管本 rank 物理 cache 和映射；
backend 管具体 KV layout 与 attention kernel。
```

---

## 22. 与 attention backend 的关系

### 22.1 backend 选择会受 KV cache dtype / block size / connector 影响

`AttentionBackend.validate_configuration()` 会检查：

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

位置：`vllm/vllm/v1/attention/backend.py:308`

其中和 KV cache / 并行关系较强的是：

```text
TP：改变 local heads / KV heads；
DCP：要求 backend 能返回 decode LSE；
PCP：要求 backend 声明 supports_pcp；
KV connector：要求 layout / backend 支持外部 KV 传输；
block size：影响 paged attention kernel 支持性。
```

### 22.2 DCP 要求 backend 返回 LSE

`AttentionImplBase` 中：

```text
need_to_return_lse_for_decode = dcp_world_size > 1 and can_return_lse_for_decode
```

位置：`vllm/vllm/v1/attention/backend.py:780`

DCP 下 partial attention output 不能直接相加，必须用 LSE 修正。

### 22.3 DCP 的 merge 不是普通 all-reduce

DCP 每个 rank 计算的是：

```text
partial_out_i = softmax(Q K_i^T) V_i
partial_lse_i = logsumexp(Q K_i^T)
```

最终需要：

```text
用所有 partial_lse_i 计算全局 softmax 分母，
再加权合并 partial_out_i。
```

相关合并路径包括：

```text
cp_lse_ag_out_rs
dcp_a2a_lse_reduce
merge_attn_states
```

位置：`vllm/vllm/v1/attention/ops/common.py`

---

## 23. KV cache 与 CUDA graph / padding 的关系

CUDA graph 要求很多 tensor shape 稳定，因此 KV cache 映射也要支持 padding。

`BlockTables.compute_slot_mappings()` 的 Triton kernel 会在最后一个 program 中把 padding 区域填成：

```text
PAD_SLOT_ID
```

位置：`vllm/vllm/v1/worker/block_table.py:261`

这很重要：

```text
padding token 不应该写入真实 KV cache；
旧 step 残留的 slot id 不能污染当前 cudagraph 执行。
```

DCP 下非本 rank token 也会映射成 `PAD_SLOT_ID`。

因此 `PAD_SLOT_ID` 同时承担两类语义：

```text
1. padding / 无效 token；
2. CP 下不属于当前 rank 的 token。
```

---

## 24. 和 Scheduler / KVCacheManager 的边界

### 24.1 Scheduler / KVCacheManager 负责

```text
- 判断请求是否能分配 KV blocks；
- prefix cache lookup / block hash；
- allocate_slots / free / evict；
- 维护 block ref_cnt；
- 处理 preemption / recompute；
- 生成 block_ids / new_block_ids_to_zero；
- 生成 KV connector metadata；
- 在 CP 下使用正确 scheduler_block_size / hash_block_size。
```

### 24.2 Worker / ModelRunner 负责

```text
- 分配或接收本 rank 物理 KV cache tensor；
- 将 block_ids 写入 InputBatch / BlockTables；
- zero 新分配的 physical blocks；
- 计算 positions / slot mapping；
- 构造 CommonAttentionMetadata；
- 将 KV cache 绑定到 attention layers；
- 驱动 KV connector load/save/finalize。
```

### 24.3 Attention backend 负责

```text
- 定义 KV cache shape / stride order；
- 更新本 rank KV cache；
- 根据 block table / slot mapping / seq_lens 读取 KV；
- DCP / PCP 下执行 partial attention 和 LSE merge；
- 声明自己支持哪些 dtype / block size / connector / CP 能力。
```

### 24.4 一句话边界

```text
Scheduler 管“应该用哪些 blocks”，ModelRunner 管“这些 blocks 在本 rank 怎么映射成 slots”，backend 管“这些 slots 的 KV tensor 怎么读写”。
```

---

## 25. 容易混淆的点

### 25.1 KV cache 是所有 ranks 共享的吗？

不是。

```text
KV cache 是 rank-local 的物理 tensor。
```

不同并行维度只是在逻辑上让这些本地 KV cache 共同组成完整执行。

### 25.2 TP 下 block id 相同是否表示同一块物理内存？

不是。

```text
不同 TP ranks 上的 block id 只是各自本地 KV cache tensor 的索引。
```

它们保存的是不同 local KV heads 或复制后的 local KV head 视图。

### 25.3 PP rank 会持有所有 layers 的 KV cache 吗？

不会。

```text
PP rank 只持有自己 stage 内 attention layers 的 KV cache。
```

### 25.4 DP replicas 会共享 prefix cache 的物理 blocks 吗？

常规推理中不会。

```text
DP replica 的 KV cache / block table / slot mapping 是独立的。
```

如果要跨实例复用，需要 external KV transfer / connector 显式介入。

### 25.5 EP 会改变 attention KV heads 吗？

不会。

```text
EP 切 MoE experts，不切 dense attention KV heads。
```

attention KV heads 主要由 TP / model config 决定。

### 25.6 DCP 是不是只改 attention 通信？

不是。

DCP 同时改变：

```text
- 每 rank 最大 KV memory；
- global position 到 local slot 的映射；
- block table 的解释；
- dcp_local_seq_lens；
- attention backend partial output + LSE merge。
```

### 25.7 block table 和 slot mapping 是一回事吗？

不是。

```text
block table：request → block ids
slot mapping：token → local KV slot
```

### 25.8 KV cache shape 是不是固定 `[2, num_blocks, block_size, heads, dim]`？

不是。

```text
最终 shape 由 attention backend 的 get_kv_cache_shape() 决定。
```

MLA、FlashInfer、ROCm、量化 KV cache、blocks-first layout 都可能不同。

### 25.9 CP 下非本 rank token 会写 KV cache 吗？

不会。

```text
slot mapping 会把非本 rank token 映射为 PAD_SLOT_ID。
```

### 25.10 KV connector 是否只需要 request id？

不够。

KV connector 还必须处理：

```text
layer / group / block id / rank / backend layout / CP interleave / load-save 状态。
```

---

## 26. 最终可以记成一张表

| 并行维度 | 对 KV cache 的直接影响 | 对 block table / slot mapping 的影响 | 关键源码 |
|---|---|---|---|
| TP | 改变本 rank `num_heads / num_kv_heads`，KV cache 保存 local KV heads | block id 仍是本地 block；slot 对应本 rank local heads | `config/model.py:1259`、`attention.py:581` |
| PP | 当前 rank 只持有本 stage layers 的 KV cache | block table 指向本 stage 本地 layers 的 blocks | `config/model.py:1282`、`worker/utils.py:462` |
| DP | 每个 replica 有独立 KV cache 副本 | 每个 replica 独立维护 block table / slot mapping | `config/parallel.py:516` |
| EP | 不直接改变 dense attention KV cache | 不直接改变 block table / slot mapping | `parallel_state.py:1201` |
| DCP | 同一请求 context KV 按 decode CP ranks 分片 | global position → local slot；非本 rank → `PAD_SLOT_ID` | `config/parallel.py:339`、`gpu/block_table.py:289` |
| PCP | prefill context parallel，与 DCP 共同组成 total CP rank | 使用同一套 CP interleave 语义 | `config/parallel.py:359`、`attention/backend.py:780` |
| backend layout | 决定 KV cache tensor shape / stride | connector 和 kernel 必须按 backend layout 解释 KV blocks | `attention/backend.py:87`、`gpu_model_runner.py:7072` |
| KV connector | 跨实例 load/save KV blocks | 需要理解 rank-local block、group、layout、finished 状态 | `kv_connector_model_runner_mixin.py:78` |

---

## 27. 一句话总结

KV cache 与并行策略的完整关系可以记成：

```text
TP：切 heads，所以 KV cache 存 local KV heads。
PP：切 layers，所以 KV cache 只存在于对应 pipeline stage。
DP：切 replica，所以每个 replica 有独立 KV cache。
EP：切 experts，不直接切 attention KV cache。
CP/DCP/PCP：切 context positions，所以 slot mapping 会把 global position 映射成本 rank local slot。
backend：决定 KV cache tensor 的最终 shape / stride / kernel 读写方式。
```

如果只保留一句话：

```text
KV cache 在 vLLM 中始终以 rank-local 物理 tensor 存在；并行策略决定“这个 rank 应该保存哪些 heads、哪些 layers、哪些 context positions”，block table 和 slot mapping 则把 Scheduler 的逻辑 blocks 翻译成 attention backend 能读写的本地 KV slots。
```
