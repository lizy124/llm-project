# 13. PagedAttention：vLLM 的分页 KV cache 注意力机制

源码位置：

- `vllm/vllm/v1/core/kv_cache_manager.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/block_pool.py`
- `vllm/vllm/v1/core/kv_cache_coordinator.py`
- `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/vllm/v1/kv_cache_interface.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/v1/worker/gpu/block_table.py`
- `vllm/vllm/v1/attention/backend.py`
- `vllm/vllm/v1/attention/backends/flash_attn.py`
- `vllm/vllm/v1/attention/ops/paged_attn.py`
- `vllm/vllm/model_executor/layers/attention/attention.py`

本文用于梳理 PagedAttention 的核心思想：KV cache 分块、block table、slot mapping、非连续 KV 存储，以及 attention backend 如何读取 paged KV cache。

注意：`code/vllm/docs/design/paged_attention.md` 明确标注为历史文档，不再描述当前 vLLM 代码。本文只把它作为概念背景；当前实现以 V1 的 KV cache manager、block table、attention backend 和各 backend kernel 路径为准。

---

## 1. 本文要回答的问题

```text
PagedAttention 解决什么问题？
block table 和操作系统 page table 的类比是什么？
Scheduler / KVCacheManager 如何分配 paged KV blocks？
ModelRunner 如何生成 block table 和 slot mapping？
attention backend 如何按 block table 读取历史 KV？
PagedAttention 和 prefix cache / preemption / continuous batching 有什么关系？
PagedAttention 和 FlashAttention 是什么关系？
```

---

## 2. 一句话回答

PagedAttention 是 vLLM 用“固定大小 KV blocks + block table + slot mapping”管理请求历史 KV 的机制。

它把每个请求的逻辑 token 序列切成一页页 KV block：

```text
逻辑序列位置 position
  → logical block index = position // block_size
  → block table 查到 physical block id
  → block offset = position % block_size
  → slot id = physical block id * block_size + block offset
```

这样请求的 KV cache 不需要在 GPU 上连续存放。请求可以一边生成一边追加 block，也可以复用 prefix block、释放尾部 block、被抢占后重新分配，continuous batching 里不同长度请求也能稳定混跑。

如果只记一句话：

```text
PagedAttention 负责“把每个请求连续的逻辑上下文，映射到 GPU 上不连续的 KV cache blocks”。
```

---

## 3. 为什么需要 PagedAttention

普通 KV cache 最直接的做法是：每个请求预留一段连续显存，里面按 token 顺序保存 K/V。

这会带来几个问题：

```text
1. 请求长度不可预知，连续预留容易浪费；
2. 不同请求结束时间不同，连续区域容易碎片化；
3. prefix cache 想复用一段已有 KV 时，很难让两个请求共享连续片段；
4. continuous batching 中请求频繁加入 / 结束 / 抢占，连续 KV 布局维护成本高；
5. chunked prefill、spec decode、external KV load 都要求 KV 状态可以按块追加、复用、迁移。
```

PagedAttention 的做法是把 KV cache 变成“页式内存”：

```text
物理 KV cache：
  block 0 | block 1 | block 2 | block 3 | ...

请求 A 的逻辑 blocks：
  [block 7, block 2, block 9]

请求 B 的逻辑 blocks：
  [block 4, block 2, block 11]
```

注意两个请求甚至可以共享同一个 prefix block。只要 block table 描述清楚，attention kernel 就能按逻辑顺序读到正确 KV。

---

## 4. OS page table 类比

PagedAttention 的名字来自操作系统分页内存的类比。

| OS 虚拟内存 | vLLM PagedAttention |
|---|---|
| virtual address | token position |
| page size | KV block size |
| virtual page number | logical block index |
| page table | block table |
| physical frame | physical KV block |
| page offset | block offset |
| physical address | KV cache slot |
| frame ref count | KVCacheBlock.ref_cnt / prefix block 共享 |

对应关系可以写成：

```text
position = 37, block_size = 16

logical_block_idx = 37 // 16 = 2
block_offset      = 37 % 16  = 5
physical_block_id = block_table[req_idx][2]
slot_id           = physical_block_id * 16 + 5
```

区别是：

```text
OS page table 由硬件 MMU / 内核参与地址翻译；
PagedAttention 的 block table 是显式传给 attention kernel 的张量，
kernel 自己根据 block table 去 KV cache 中加载 K/V。
```

---

## 5. PagedAttention 的核心对象

### 5.1 KVCacheSpec：定义“一页”有多大

`KVCacheSpec` 是每类 KV cache 的规格基类。

位置：`vllm/vllm/v1/kv_cache_interface.py:95`

其中最关键的是：

```text
block_size：一个 KV block 容纳多少 token；
page_size_bytes：这个 block 对应多少字节显存；
storage_block_size：某些 MLA / 压缩 KV cache 的物理存储粒度。
```

对于普通 attention，`AttentionSpec.page_size_bytes` 大致由这些因素决定：

```text
2 * block_size * num_kv_heads * head_size * dtype_size
```

这里的 `2` 对应 K 和 V。

位置：`vllm/vllm/v1/kv_cache_interface.py:159`

### 5.2 KVCacheConfig：描述整个模型有哪些 KV cache group

`KVCacheConfig` 记录：

```text
- num_blocks：全局 KV block 数；
- kv_cache_tensors：Worker 应该如何创建底层 KV cache tensor；
- kv_cache_groups：哪些 layer 共享一组 block table / cache spec。
```

位置：`vllm/vllm/v1/kv_cache_interface.py:879`

对普通 decoder-only 模型，通常只有一个 KV cache group。对 hybrid 模型、MLA、Mamba、cross-attention 等，可能会有多个 group。

### 5.3 KVCacheBlock：物理 block 的账本对象

`BlockPool` 内部维护所有 `KVCacheBlock`。

位置：`vllm/vllm/v1/core/block_pool.py:144`

每个 block 至少承担三类状态：

```text
1. block_id：物理 KV block 编号；
2. ref_cnt：有多少请求或缓存状态引用它；
3. block_hash：如果进入 prefix cache，用 hash 表支持命中查询。
```

### 5.4 KVCacheBlocks：KVCacheManager 对 Scheduler 暴露的分配结果

`KVCacheBlocks` 是 `KVCacheManager` 和 `Scheduler` 之间的接口对象。

位置：`vllm/vllm/v1/core/kv_cache_manager.py:25`

它的结构是：

```text
blocks[group_id][block_idx] = KVCacheBlock
```

之所以 group 在外层，是因为未来不同 KV cache group 可以有不同 block size / block 数量。

它会通过 `get_block_ids()` 转换成 Worker 更容易消费的：

```text
tuple[list[int], ...]
```

也就是：

```text
每个 KV cache group 一组 block ids。
```

---

## 6. 主链路总览

PagedAttention 横跨 Scheduler、Worker / ModelRunner 和 attention backend。

完整链路是：

```text
Request tokens
  → Scheduler.schedule()
  → KVCacheManager.get_computed_blocks() 查 prefix cache
  → KVCacheManager.allocate_slots() 分配 / 复用 KV blocks
  → SchedulerOutput 携带 block ids / num_computed_tokens / num_common_prefix_blocks
  → GPUModelRunner._update_states() 同步到 InputBatch
  → InputBatch.block_table 写入 request → block ids
  → GPUModelRunner._prepare_inputs() 计算 positions / seq_lens / slot mapping
  → GPUModelRunner._get_slot_mappings() 取出 gid / layer 两种 slot mapping
  → GPUModelRunner._build_attention_metadata() 生成 CommonAttentionMetadata / backend metadata
  → set_forward_context(...)
  → Attention.forward()
  → attention backend 写入当前 K/V，并按 block table 读取历史 K/V
```

压缩成一句话：

```text
Scheduler 负责“分配哪些 blocks”，ModelRunner 负责“把 blocks 翻译成 metadata”，attention backend 负责“按 metadata 读写 KV”。
```

---

## 7. Scheduler / KVCacheManager 如何分配 blocks

### 7.1 prefix cache 命中先查 computed blocks

Scheduler 在调度 waiting 请求时，会先尝试 prefix cache：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:711`

`get_computed_blocks()` 的职责是：

```text
根据 request.block_hashes 查找最长可复用 prefix blocks，
返回命中的 KVCacheBlocks 和对应 computed token 数。
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:202`

有个重要细节：如果 prompt 全部命中，也通常要至少重算最后一个 token 来拿 logits，所以最大命中长度会限制到 `prompt_length - 1`。

### 7.2 allocate_slots() 是分配主入口

位置：`vllm/vllm/v1/core/kv_cache_manager.py:244`

它处理的不是简单的“给 N 个新 token 分配 N 个 slot”，而是这一段布局：

```text
-----------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
-----------------------------------------------------------------
                                  |      < to be allocated >       |
-----------------------------------------------------------------
                      |        < to be cached roughly >            |
-----------------------------------------------------------------
```

含义是：

```text
comp：请求之前已经计算过的 token；
new_comp：本轮 prefix cache 新命中的 token；
ext_comp：外部 KV connector 已经计算、但不在本地 prefix cache 的 token；
new：本轮要真正 forward 的 token；
lookahead：spec decode / EAGLE 等需要预留的未来 token。
```

### 7.3 分配分三步

`allocate_slots()` 的注释把流程拆成三阶段：

```text
1. 释放不再需要的旧 blocks，并检查 free blocks 是否足够；
2. 处理 prefix tokens，包括本地 prefix cache 和 external KV；
3. 为本轮真正要计算的 new tokens / lookahead tokens 分配 blocks。
```

位置：`vllm/vllm/v1/core/kv_cache_manager.py:328`

其中 `remove_skipped_blocks()` 会先释放 sliding window / local attention 场景下 attention 已经不会看的 blocks。

### 7.4 BlockPool 负责真正拿 free blocks

位置：`vllm/vllm/v1/core/block_pool.py:542`

`BlockPool.get_new_blocks()` 从 free queue 里取 block，并递增 ref count。

如果开启 prefix caching，取到的 free block 可能还带着旧 hash，此时需要先从 prefix cache hash 表中驱逐。

### 7.5 分配失败会触发 preemption

Scheduler 调度 running 请求时，如果 `allocate_slots()` 返回 `None`，说明 KV blocks 不够。

位置：`vllm/vllm/v1/core/sched/scheduler.py:523`

这时 Scheduler 会选择一个 running request 抢占：

```text
allocate_slots() 失败
  → 选择优先级最低或队尾请求
  → _preempt_request()
  → 释放该请求 KV blocks
  → request.status = PREEMPTED
  → request.num_computed_tokens = 0
  → 后续重新进入 waiting / resumed 流程
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1107`

所以 PagedAttention 和 preemption 的关系是：

```text
因为 KV cache 是 block 级分配 / 释放，抢占一个请求时可以直接释放它持有的 blocks，
不需要搬移其他请求的连续 KV 内存。
```

---

## 8. BlockPool 如何支持 prefix cache

### 8.1 full block 才是主要 prefix cache 单位

`BlockPool.cache_full_blocks()` 会把已经填满、且可以作为 prefix 的 blocks 加入 hash 表。

位置：`vllm/vllm/v1/core/block_pool.py:226`

核心映射是：

```text
BlockHashWithGroupId → KVCacheBlock 或 {block_id: KVCacheBlock}
```

位置：`vllm/vllm/v1/core/block_pool.py:34`

### 8.2 命中时 touch block

`BlockPool.touch()` 会增加 block 的 `ref_cnt`。

位置：`vllm/vllm/v1/core/block_pool.py:597`

如果命中的 block 当前在 free queue 里，它会被移出 free queue，避免被分配给其他请求覆盖。

也就是说：

```text
prefix cache 命中不是复制 KV，
而是多个请求通过 ref_cnt 共享同一个 physical KV block。
```

### 8.3 释放时不一定立刻删除缓存

`BlockPool.free_blocks()` 会减少 ref count。

位置：`vllm/vllm/v1/core/block_pool.py:614`

当 ref count 变成 0：

```text
- 没有 hash 的 block：放回 free queue 头部，优先复用；
- 有 hash 的 block：放到 free queue 尾部，作为可被驱逐的 prefix cache block。
```

这就是 prefix cache 能在请求结束后继续保留的原因。

### 8.4 为什么不对相同内容 blocks 去重

`BlockHashToBlockMap` 注释里有个关键设计点：当前不会对相同 hash 的 blocks 做去重。

位置：`vllm/vllm/v1/core/block_pool.py:48`

原因是：

```text
vLLM 希望已经分配出去的 block ids 不改变，
从而让 block tables 可以 append-only 地增长，
避免因为去重而重写已有请求的 block table。
```

---

## 9. Worker 侧 block table 是什么

Scheduler 只负责资源账本；真正 forward 前，Worker 需要把 block ids 变成 GPU 上 attention backend 能读的 block table。

当前 GPU 路径的 block table 实现在：

`vllm/vllm/v1/worker/gpu/block_table.py:17`

### 9.1 BlockTables 维护多组 block table

`BlockTables` 内部维护：

```text
block_tables[group_id]：StagedWriteTensor，CPU staged write + GPU tensor；
input_block_tables[group_id]：forward 实际使用的 block table；
slot_mappings[group_id]：每个 scheduled token 对应的 KV slot。
```

位置：`vllm/vllm/v1/worker/gpu/block_table.py:47`

### 9.2 append_block_ids() 写入请求行

位置：`vllm/vllm/v1/worker/gpu/block_table.py:107`

它把 Scheduler 分配好的 block ids 写入某个 request index 对应的行。

如果 KV cache 的 framework block size 和 kernel block size 不同，还会把一个 framework block 展开成多个 kernel block：

```text
framework block id = b
blocks_per_kv_block = 2
kernel block ids = [b * 2, b * 2 + 1]
```

### 9.3 commit / gather 是为了 forward 使用

`GPUModelRunner._prepare_inputs()` 一开始就会调用：

```python
self.input_batch.block_table.commit_block_table(num_reqs)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:1906`

目的不是立刻 attention，而是提前把 CPU staged writes 提交到 GPU，和后续 CPU 侧 input 准备重叠。

后面 `_build_attention_metadata()` 中 `_get_block_table()` 会拿到每个 group 的 device block table：

```python
blk_table = self.input_batch.block_table[kv_cache_gid]
blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2257`

### 9.4 block table 的形状

典型形状是：

```text
[num_reqs, max_num_blocks_per_req]
```

例如：

```text
request 0: [7, 2, 9, 0, 0, ...]
request 1: [4, 8, 0, 0, 0, ...]
request 2: [5, 6, 3, 1, 0, ...]
```

在当前 `BlockPool` 初始化中，第一个 block 会被取作 `null_block` 并标记为占位，用于 sliding/local 等场景；但不要把普通 block table 中出现的 0 一律解释为无效 block，具体语义取决于该位置是否确实填入了 null block。

---

## 10. slot mapping 是什么

block table 是“请求 → blocks”的映射；slot mapping 是“本轮 token → KV cache slot”的映射。

### 10.1 slot mapping 的计算入口

`_prepare_inputs()` 在算完 positions / seq_lens 后调用：

```python
self.input_batch.block_table.compute_slot_mapping(
    num_reqs,
    self.query_start_loc.gpu[: num_reqs + 1],
    self.positions[:total_num_scheduled_tokens],
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2126`

底层 kernel 在：

`vllm/vllm/v1/worker/gpu/block_table.py:240`

### 10.2 slot mapping 的公式

普通无 context parallel 的情况是：

```text
block_indices = positions // block_size
block_offsets = positions % block_size
block_numbers = block_table[req_idx][block_indices]
slot_ids      = block_numbers * block_size + block_offsets
```

对应实现位置：`vllm/vllm/v1/worker/gpu/block_table.py:283`

举例：

```text
block_size = 16
request 0 block_table = [7, 2, 9]
position = 18

block_index = 18 // 16 = 1
block_offset = 18 % 16 = 2
physical_block_id = block_table[0][1] = 2
slot_id = 2 * 16 + 2 = 34
```

这个 `slot_id` 就是当前 token 的 K/V 要写入的物理位置。

### 10.3 padding slot 使用 -1

为了 CUDA graph / padding，未使用 token 区间会填 `PAD_SLOT_ID`，通常是 `-1`。

位置：`vllm/vllm/v1/worker/gpu/block_table.py:261`

`GPUModelRunner._get_slot_mappings()` 也会对 padded 区域填 `-1`。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4009`

### 10.4 context parallel 下 slot 可能不是本 rank 本地

当 `CP_SIZE > 1` 时，slot mapping kernel 会判断某个 token 的 KV 是否属于当前 CP rank。

如果不是本 rank，就写成 `PAD_ID`。

位置：`vllm/vllm/v1/worker/gpu/block_table.py:292`

所以更严格地说：

```text
slot mapping = token position 到“当前 rank 本地 KV cache slot”的映射。
```

---

## 11. ModelRunner 如何把 block table / slot mapping 放进 attention metadata

### 11.1 _get_slot_mappings() 生成两种视图

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3963`

它返回：

```text
slot_mappings_by_gid：
  kv_cache_group_id → slot_mapping
  给 _build_attention_metadata() 使用。

slot_mappings_by_layer：
  layer_name → slot_mapping
  放进 forward context，给 Attention layer / KV cache update 使用。
```

为什么要两种？

```text
attention metadata 构造按 KV cache group 组织；
模型 forward 内部按 layer name 取 attention 层上下文。
```

### 11.2 _build_attention_metadata() 先构造 CommonAttentionMetadata

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2216`

核心公共字段包括：

```python
CommonAttentionMetadata(
    query_start_loc=...,
    seq_lens=...,
    max_query_len=...,
    max_seq_len=...,
    block_table_tensor=...,
    slot_mapping=...,
    positions=...,
    is_prefilling=...,
)
```

对应定义在：`vllm/vllm/v1/attention/backend.py:361`

这些字段分别解决：

```text
query_start_loc：每个请求本轮 query token 的边界；
seq_lens：每个请求当前总上下文长度；
max_query_len：本轮单请求最大 query 长度；
max_seq_len：本轮最大上下文长度；
block_table_tensor：请求逻辑 blocks 到物理 blocks 的映射；
slot_mapping：本轮 token 写入 KV cache 的 slot；
positions：token 的逻辑位置；
is_prefilling：请求是否仍在 prefill 阶段。
```

### 11.3 per-layer metadata 由 backend builder 构造

`AttentionMetadataBuilder` 是抽象基类。

位置：`vllm/vllm/v1/attention/backend.py:533`

每种 attention backend 会把 `CommonAttentionMetadata` 转成自己的 metadata。

例如 FlashAttention builder 会生成 `FlashAttentionMetadata`：

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:236`

它保留了：

```text
query_start_loc
seq_lens
block_table
slot_mapping
max_query_len
max_seq_len
cascade attention 相关字段
scheduler_metadata
```

### 11.4 多 KV cache group 下只替换 block table / slot mapping

`_build_attention_metadata()` 会按 KV cache group 循环。

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2460`

通常不同 group 共享大部分 batch 形状信息，只是：

```text
block_table_tensor 不同；
slot_mapping 不同；
encoder_seq_lens 可能不同；
kv_cache_spec / builder 可能不同。
```

如果 builder 支持 `update_block_table()`，可以复用 metadata，只更新 block table 和 slot mapping。

FlashAttention builder 就支持：

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:611`

---

## 12. Attention.forward() 如何拿到 paged KV 信息

### 12.1 forward context 是隐式参数区

`GPUModelRunner.execute_model()` 在模型 forward 前调用：

```python
set_forward_context(
    attn_metadata,
    self.vllm_config,
    ...,
    slot_mapping=slot_mappings,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4306`

这意味着模型层的 `Attention.forward()` 不需要显式传入 block table / slot mapping，它们都在 forward context 里。

### 12.2 Attention layer 自己持有 kv_cache

`Attention.forward()` 注释里说明：

```text
KV cache 存储在 Attention class 内部，通过 self.kv_cache 访问；
attention metadata 通过 forward context 访问。
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:452`

实际执行时，统一入口会从 forward context 中按 `layer_name` 取：

```text
attn_metadata
attn_layer
kv_cache
layer_slot_mapping
```

位置：`vllm/vllm/model_executor/layers/attention/attention.py:675`

### 12.3 KV cache update 也依赖 slot mapping

对于 FlashAttention backend：

```text
forward_includes_kv_cache_update = False
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:96`

因此 `Attention.forward()` 会先调用统一 KV cache update，把当前 token 的 K/V scatter 到 paged cache 中。

入口：`vllm/vllm/model_executor/layers/attention/attention.py:713`

FlashAttention 的具体实现：

```python
reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    self.kv_cache_dtype,
    layer._k_scale,
    layer._v_scale,
)
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:927`

所以写入路径是：

```text
当前 forward 算出的 key/value
  → layer_slot_mapping
  → reshape_and_cache_flash / reshape_and_cache
  → physical KV cache slots
```

---

## 13. attention backend 如何按 block table 读取历史 KV

### 13.1 FlashAttention backend 的读路径

FlashAttention forward 中会从 metadata 取：

```text
cu_seqlens_q = attn_metadata.query_start_loc
seqused_k    = attn_metadata.seq_lens
block_table  = attn_metadata.block_table
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:792`

然后调用：

```python
flash_attn_varlen_func(
    q=query[:num_actual_tokens],
    k=key_cache,
    v=value_cache,
    ...,
    seqused_k=seqused_k,
    block_table=block_table,
)
```

位置：`vllm/vllm/v1/attention/backends/flash_attn.py:870`

注意这里传给 FlashAttention 的 `k/v` 不是本轮 key/value，而是 paged KV cache。

`block_table` 告诉 kernel：

```text
每个 sequence 的第 j 个逻辑 block 对应 key_cache / value_cache 的哪个 physical block。
```

### 13.2 Triton paged decode 的读路径

`chunked_prefill_paged_decode()` 也直接接收：

```text
key_cache
value_cache
block_table
query_start_loc
seq_lens
max_seq_len
```

位置：`vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py:266`

当进入 paged attention kernel 时，会传：

```python
block_tables_ptr=processed_block_table,
seq_lens_ptr=seq_lens,
block_table_stride=processed_block_table.stride(0),
```

位置：`vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py:442`

### 13.3 paged_attn.py 是简化 helper，不是所有 backend 的统一入口

`PagedAttention.write_to_paged_cache()` 做的事情很直接：

```python
ops.reshape_and_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping.flatten(),
    kv_cache_dtype,
    k_scale,
    v_scale,
)
```

位置：`vllm/vllm/v1/attention/ops/paged_attn.py:31`

它体现了 PagedAttention 的核心：

```text
写入靠 slot_mapping；
读取靠 block_table + seq_lens。
```

但在 V1 主路径中，FlashAttention / Triton / FlashInfer 等通常由各自 backend 的 `do_kv_cache_update()` 调用对应 cache update op，例如 `reshape_and_cache_flash` 或 Triton cache op；不要把 `paged_attn.py` 理解为所有 backend 的统一入口。

---

## 14. prefill 和 decode 中 PagedAttention 分别做什么

### 14.1 prompt prefill

prefill 阶段一个请求本轮可能调度多个 token。

```text
num_scheduled_tokens[req] > 1
max_query_len > 1
```

ModelRunner 会为这些 token 计算连续 positions：

```python
positions_np = num_computed_tokens_cpu[req_indices] + query_pos
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:1920`

然后 slot mapping 会把这些 positions 分散写入多个 physical blocks。

例如：

```text
block_size = 16
本轮 prefill positions = 14, 15, 16, 17, 18
block_table = [7, 2]

position 14 → block 7 offset 14
position 15 → block 7 offset 15
position 16 → block 2 offset 0
position 17 → block 2 offset 1
position 18 → block 2 offset 2
```

也就是说，一个 prefill chunk 可以跨 block 写入。

### 14.2 decode

decode 阶段通常每个请求本轮只调度一个新 token。

```text
num_scheduled_tokens[req] = 1
max_query_len = 1
```

slot mapping 只需要给这个新 token 找到追加位置：

```text
position = 当前 seq_len
slot = block_table[position // block_size] * block_size + position % block_size
```

但读取时 attention 仍然要看完整历史上下文：

```text
seq_lens 告诉 kernel 历史长度；
block_table 告诉 kernel 历史 blocks 在哪里。
```

### 14.3 chunked prefill

chunked prefill 本质上是：

```text
一个长 prompt 被拆成多轮 prefill chunk，
每轮只写入本 chunk 对应的 positions，
block table 则持续记录已经分配的全部 blocks。
```

如果本 chunk 还没到 prompt 末尾，`discard_request_mask` 会让采样结果被丢弃。

这和 PagedAttention 的关系是：

```text
Paged KV cache 允许 prompt 分多轮、跨 blocks 地逐步写入，而不要求一次性连续计算完整 prompt。
```

---

## 15. PagedAttention 和 continuous batching

continuous batching 的特点是每轮 batch 都可能变化：

```text
- 新请求进入；
- 老请求继续 decode；
- 某些请求结束；
- 某些请求被抢占；
- 不同请求本轮 query length 不同。
```

如果 KV cache 要求每个请求连续存储，就很难维护。

PagedAttention 下，batch 变化只需要更新几类小结构：

```text
1. Scheduler 侧请求持有哪些 KVCacheBlock；
2. Worker 侧 InputBatch.block_table 的 row；
3. 本轮 token 的 slot_mapping；
4. attention metadata 的 query_start_loc / seq_lens。
```

物理 KV cache 中的 blocks 不需要因为 batch 重排而搬移。

这也是为什么 `InputBatch` 可以 condense / move row，而 KV block 本身不动。

---

## 16. PagedAttention 和 prefix cache

prefix cache 可以理解成 PagedAttention 的自然扩展。

### 16.1 命中时发生什么

```text
请求 B 的 prompt prefix 和请求 A 一样
  → Scheduler 通过 block hashes 找到 A 已经缓存的 blocks
  → BlockPool.touch() 增加这些 blocks 的 ref_cnt
  → B 的 block table 前几项直接指向这些 physical blocks
  → B 从命中长度之后继续计算
```

Worker / ModelRunner 不需要知道“为什么这些 block ids 是复用来的”。它只消费：

```text
block_ids
num_computed_tokens
```

### 16.2 prefix cache 对 _prepare_inputs() 的影响

`num_computed_tokens` 变大后，positions 会从命中 prefix 后继续：

```text
positions = num_computed_tokens + query_pos
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:1920`

也就是：

```text
命中的 prefix 不再重新写 KV；
新 token 的 slot mapping 从 prefix 之后开始；
attention 读取时仍能通过 block table 看见 prefix blocks。
```

### 16.3 common prefix blocks 和 cascade attention

Scheduler 每轮还会计算 running requests 的公共 prefix block 数：

```python
num_common_prefix_blocks = (
    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
)
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1001`

它会进入 `SchedulerOutput.num_common_prefix_blocks`。

位置：`vllm/vllm/v1/core/sched/scheduler.py:1063`

ModelRunner 用它计算 cascade attention prefix length：

```text
common_prefix_len = num_common_prefix_blocks * block_size
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:2583`

这是一种利用 batch 内公共 prefix 的优化，和 prefix cache 一样依赖“多个请求共享 blocks”这个基础能力。

---

## 17. PagedAttention 和 preemption

preemption 的核心是释放 KV blocks，而不是搬移 KV 数据。

当 KV blocks 不够时：

```text
Scheduler 选择一个 request
  → _free_request_blocks(request)
  → KVCacheManager.free(request)
  → coordinator / BlockPool 降低 ref_cnt 或放回 free queue
  → request.status = PREEMPTED
  → request.num_computed_tokens = 0
```

位置：`vllm/vllm/v1/core/sched/scheduler.py:1107`

如果被释放的 blocks 仍有 prefix cache hash，可能不会立刻清除内容，而是变成可驱逐缓存。

这和 PagedAttention 的关系是：

```text
请求的 KV 状态由“block id 列表”描述，
抢占时释放这些 blocks 即可；
其他请求的 block table 仍然指向自己的 blocks，不受影响。
```

---

## 18. PagedAttention 和 external KV / KV connector

`allocate_slots()` 支持 `num_external_computed_tokens` 和 `delay_cache_blocks`。

位置：`vllm/vllm/v1/core/kv_cache_manager.py:244`

这说明 PagedAttention 不只服务本地 prefix cache，也服务远端 KV：

```text
外部系统已经算过一段 KV
  → Scheduler 为这些 external computed tokens 准备本地 blocks / 状态
  → KV connector 异步 load KV 到这些 blocks
  → 请求恢复后，block table 指向已加载的 blocks
  → attention backend 按普通 paged KV 读取
```

当本轮没有 token forward，但仍需要推进 KV transfer 时，ModelRunner 会走：

```text
kv_connector_no_forward(...)
```

对应调用位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4099`

因此 external KV 的最终落点仍然是：

```text
paged KV blocks + block table。
```

---

## 19. PagedAttention 和 FlashAttention 是什么关系

这两个概念经常被混淆。

### 19.1 PagedAttention 是 KV cache 管理 / 寻址机制

它回答的是：

```text
历史 K/V 存在哪里？
一个请求的第 j 个逻辑 block 对应哪个 physical block？
当前 token 的 K/V 应该写入哪个 slot？
```

核心数据是：

```text
KV blocks
block table
slot mapping
seq_lens
```

### 19.2 FlashAttention 是 attention 计算 kernel / 算法

它回答的是：

```text
给定 Q 和历史 K/V，如何高效计算 softmax(QK^T)V？
如何做 varlen batch？
如何降低 HBM 读写？
如何用 FA2 / FA3 / FA4 / AOT scheduler / CUDA graph？
```

核心是计算，不是分配。

### 19.3 在 vLLM 中二者可以组合

FlashAttention backend 会：

```text
1. 用 slot_mapping 把本轮 K/V 写入 paged KV cache；
2. 调 flash_attn_varlen_func 时传入 block_table；
3. 由 FlashAttention kernel 按 paged KV layout 读取历史 K/V。
```

所以更准确的说法是：

```text
PagedAttention 提供 paged KV cache 布局；
FlashAttention 可以作为读取这种布局并执行 attention 的 backend。
```

不是“有了 FlashAttention 就不需要 PagedAttention”，也不是“PagedAttention 是 FlashAttention 的替代品”。

---

## 20. 多 KV cache group / hybrid 模型下的 PagedAttention

vLLM V1 里 PagedAttention 不只面向普通 full attention。

`KVCacheSpecKind` 包括：

```text
FULL_ATTENTION
MLA_ATTENTION
SLIDING_WINDOW
SLIDING_WINDOW_MLA
MAMBA
CHUNKED_LOCAL_ATTENTION
SINK_FULL_ATTENTION
ENCODER_ONLY_ATTENTION
CROSS_ATTENTION
```

位置：`vllm/vllm/v1/kv_cache_interface.py:82`

因此实际运行中可能有多个 KV cache group：

```text
group 0：普通 attention layers
group 1：sliding window attention layers
group 2：Mamba / SSM state
group 3：cross-attention cache
```

ModelRunner 侧处理方式是：

```text
- block table 按 group 存；
- slot mapping 按 group 算；
- attention metadata 按 group / attention group 构造；
- forward context 再映射到 layer name。
```

这就是 `_get_slot_mappings()` 同时返回 by-gid 和 by-layer 的原因。

---

## 21. PagedAttention 中几个容易混淆的概念

### 21.1 block table 和 slot mapping 不是一回事

```text
block table：请求级，描述 request 的第 j 个逻辑 block 在哪个 physical block。
slot mapping：token 级，描述本轮第 i 个 token 的 K/V 写入哪个 physical slot。
```

### 21.2 block id 和 slot id 不是一回事

```text
block id：KV block 编号；
slot id：block 内具体 token slot 的扁平编号。
```

普通情况下：

```text
slot_id = block_id * block_size + block_offset
```

### 21.3 seq_lens 和 query_start_loc 不是一回事

```text
seq_lens：每个请求当前总上下文长度；
query_start_loc：本轮 query tokens 在扁平 batch 张量中的边界。
```

例如：

```text
请求 A 历史 100，本轮 1 token：seq_len=101
请求 B 历史 20，本轮 5 tokens：seq_len=25
query_start_loc=[0, 1, 6]
```

### 21.4 prefix cache 命中不等于 Worker 重新发现 prefix

prefix cache 的命中和 block 复用主要发生在 Scheduler / KVCacheManager。

Worker 只看到：

```text
num_computed_tokens 已经增加；
block_table 前缀已经有可用 block ids。
```

### 21.5 PagedAttention 不等于某一个 kernel

vLLM 里有 `PagedAttention.write_to_paged_cache()`，也有 FlashAttention、Triton、ROCm、自定义 MLA 等 backend。

PagedAttention 更像一套统一 KV cache layout 和 metadata 协议，而不是单一 CUDA kernel。

---

## 22. 最终可以记成一张表

| 阶段 | 主要位置 | 核心产物 | 作用 |
|---|---|---|---|
| KV 规格定义 | `kv_cache_interface.py` | `KVCacheSpec`、`KVCacheConfig` | 定义 block size、page size、KV cache group |
| 物理 block 管理 | `block_pool.py` | `KVCacheBlock`、free queue、hash map | 分配 / 释放 / prefix cache / ref count |
| Scheduler 分配 | `kv_cache_manager.py`、`scheduler.py` | `KVCacheBlocks`、block ids、`num_computed_tokens` | 决定请求持有哪些 blocks |
| Worker 状态同步 | `gpu_model_runner.py::_update_states()` | `InputBatch`、request state | 把 SchedulerOutput 落到 Worker 侧持久 batch |
| block table 更新 | `worker/gpu/block_table.py` | `block_tables[group]` | request row → physical block ids |
| slot mapping 计算 | `worker/gpu/block_table.py` | `slot_mappings[group]` | token position → physical KV slot |
| attention metadata | `attention/backend.py`、各 backend builder | `CommonAttentionMetadata`、backend metadata | 给 attention kernel 描述 batch 和 KV layout |
| KV 写入 | `attention.py`、`flash_attn.py`、`paged_attn.py` | reshape/cache op | 当前 K/V scatter 到 paged cache |
| KV 读取 | `flash_attn.py`、paged decode ops | block table + seq_lens | kernel 按逻辑顺序读取历史 KV |

---

## 23. 一句话总结

PagedAttention 的本质是：

```text
把每个请求连续增长的逻辑上下文，拆成固定大小的 KV blocks，
再用 block table 把逻辑 blocks 映射到 GPU 上不连续的 physical blocks，
并用 slot mapping 把本轮 token 的 K/V 写入正确 slot。
```

它让 vLLM 可以高效支持：

```text
continuous batching
prefix cache
chunked prefill
spec decode lookahead
preemption
sliding window / hybrid KV cache
external KV load / save
FlashAttention 等多种 backend
```

最核心的主线是：

```text
KVCacheManager 分配 blocks
  → InputBatch.block_table 记录 request → blocks
  → _prepare_inputs() 生成 positions / slot mapping
  → _build_attention_metadata() 生成 block table + slot mapping metadata
  → Attention.forward() 写入当前 KV
  → backend 按 block table 读取历史 KV
```

如果只记最后一句：

```text
block table 负责“读历史 KV 时去哪找”，slot mapping 负责“写当前 KV 时写到哪”。
```
