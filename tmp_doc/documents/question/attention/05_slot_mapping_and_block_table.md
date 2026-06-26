# 05. slot mapping 和 block table 如何工作？

源码位置：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/block_table.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/block_table.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/backend.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/backends/utils.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/model_executor/layers/attention/attention.py`

本问题关注：vLLM V1 中 request 的逻辑 token 序列如何通过 `block table` 和 `slot mapping` 映射到 paged KV cache 的物理位置；Scheduler、KVCacheManager、InputBatch / ModelRunner、attention metadata builder、attention backend 分别维护哪一部分；以及 prefix cache、chunked prefill、spec decode、context parallelism、CUDA graph padding、KV connector、多 KV cache group、hybrid block 等路径如何影响这套映射。

---

## 1. 一句话回答

`block table` 和 `slot mapping` 是 vLLM paged KV cache 的两级地址翻译：

```text
block table：request 维度的“逻辑 block index → 物理 KV block id”映射；
slot mapping：本轮 token 维度的“token position → 物理 KV slot id”映射。
```

最常见的地址计算可以理解为：

```text
position
  → logical_block_index = position // block_size
  → block_id = block_table[request_row, logical_block_index]
  → block_offset = position % block_size
  → slot_id = block_id * block_size + block_offset
```

所以：

```text
block table 解决“这个请求当前有哪些 KV blocks”；
slot mapping 解决“本轮每个 token 的 K/V 写到哪个 KV slot”；
attention metadata 把 block table / slot mapping 交给 backend；
attention forward 用 slot mapping 写当前 K/V，用 block table 读历史 K/V。
```

如果只记住一句话：

```text
block table 是页表，slot mapping 是本轮 token 的展开后物理地址。
```

---

## 2. 先给最小主链路

在 classic `GPUModelRunner` 路径中，主链路是：

```text
Scheduler.schedule()
  → KVCacheManager.allocate_slots()
  → KVCacheBlocks.get_block_ids()
  → SchedulerOutput.scheduled_new_reqs / scheduled_cached_reqs
  → GPUModelRunner._update_states()
  → InputBatch.block_table.add_row() / append_row()
  → GPUModelRunner._prepare_inputs()
      → commit_block_table(num_reqs)
      → positions / query_start_loc / seq_lens
      → compute_slot_mapping(num_reqs, query_start_loc, positions)
  → GPUModelRunner._get_slot_mappings()
      → slot_mappings_by_gid
      → slot_mappings_by_layer
  → GPUModelRunner._build_attention_metadata()
      → CommonAttentionMetadata(block_table_tensor, slot_mapping, ...)
      → AttentionMetadataBuilder.build(...)
  → set_forward_context(..., slot_mapping=slot_mappings_by_layer)
  → Attention.forward()
  → unified_kv_cache_update() / unified_attention_with_output()
  → backend impl.forward(...)
```

关键位置：

- `KVCacheManager.allocate_slots()`：`code/vllm/vllm/v1/core/kv_cache_manager.py:244`
- `_prepare_inputs()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1889`
- `_build_attention_metadata()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2208`
- `_get_slot_mappings()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3960`
- `get_attention_context()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:649`
- `unified_kv_cache_update()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:692`
- `unified_attention_with_output()`：`code/vllm/vllm/model_executor/layers/attention/attention.py:734`

这里有两个输出视图：

```text
slot_mappings_by_gid：给 _build_attention_metadata() 使用，按 KV cache group 查；
slot_mappings_by_layer：给 forward context 使用，按 layer_name 查。
```

---

## 3. 需要先分清四个概念

### 3.1 position：token 的逻辑序列位置

`position` 是 token 在单个 request 序列中的绝对位置。

在 `_prepare_inputs()` 中：

```text
positions = num_computed_tokens[req_indices] + query_pos
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1920`

例如一个请求已经计算 128 个 token，本轮调度 3 个 token：

```text
query_pos = [0, 1, 2]
positions = [128, 129, 130]
```

它后续同时用于：

```text
1. 从 InputBatch.token_ids_cpu_tensor 取 input_ids；
2. 计算普通 RoPE / M-RoPE / XD-RoPE 位置；
3. 计算 slot_mapping；
4. 进入 CommonAttentionMetadata，供部分 backend 构造 mask / sparse metadata。
```

### 3.2 block：KV cache 的分页单位

`block` 是 KV cache 的页。

`KVCacheSpec.block_size` 表示一个 KV block 容纳多少 token。

位置：`code/vllm/vllm/v1/kv_cache_interface.py:95`

例如 `block_size = 16`：

```text
positions 0..15    → logical block 0
positions 16..31   → logical block 1
positions 32..47   → logical block 2
```

### 3.3 block_id：物理 KV block 编号

`block_id` 是 block pool 中的物理 block 编号。

`BlockPool` 初始化时会创建：

```text
self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
```

位置：`code/vllm/vllm/v1/core/block_pool.py:162`

注意：`BlockPool` 会保留一个 `null_block`，它的 `block_id = 0`，用于 padding / null block。

位置：`code/vllm/vllm/v1/core/block_pool.py:176`

因此很多路径会把 padding block table 行填成 `NULL_BLOCK_ID = 0`。

位置：`code/vllm/vllm/v1/attention/backends/utils.py:45`

### 3.4 slot：KV cache 中 token 级物理位置

slot 是 block 内部展开后的 token 位置：

```text
slot_id = block_id * block_size + block_offset
```

attention backend 会用 slot id 把当前 token 的 K/V 写入 KV cache。

常见约定：

```text
PAD_SLOT_ID = -1
```

位置：`code/vllm/vllm/v1/attention/backends/utils.py:45`

含义是：

```text
这个 token 对当前 rank / 当前 padded 区域不是有效 KV 写入位置，KV update 应跳过。
```

---

## 4. block table 和 slot mapping 的职责边界

可以把职责切成四层：

| 层级 | 负责什么 | 不负责什么 |
|---|---|---|
| Scheduler / KVCacheManager | 分配、复用、释放物理 KV blocks | 不计算 token 级 slot mapping |
| SchedulerOutput | 携带每个 request 的 block ids / new block ids / scheduled token 数 | 不携带 GPU slot tensor |
| InputBatch / BlockTable | 保存 request row 到 block ids 的映射，并按 positions 计算 slot mapping | 不决定调度策略 |
| Attention backend | 消费 block table / slot mapping 执行 KV read/write 和 attention | 不分配 blocks |

这条边界非常重要：

```text
Scheduler 管资源；
Worker / ModelRunner 管执行态；
BlockTable 管地址翻译；
Attention backend 管 kernel 如何解释这些地址。
```

换句话说：

```text
Scheduler 决定“给你哪些页”；
ModelRunner 决定“这一轮哪些 token 要写”；
slot mapping 决定“这些 token 写到页内哪个位置”；
backend 决定“怎么用这些地址跑 kernel”。
```

---

## 5. Scheduler 负责分配 blocks，不负责算 slot mapping

Scheduler 在 `schedule()` 中决定本轮每个 request 要执行多少 token，并通过 `KVCacheManager.allocate_slots()` 分配 KV blocks。

运行中请求路径：

```text
new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:521`

等待请求 / 新请求路径也会调用 `allocate_slots()`，并把请求当前完整 blocks 放入 `req_to_new_blocks`。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:873`

SchedulerOutput 构造时：

```text
NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1024`

对于 cached / resumed request，会通过 `_make_cached_request_data()` 生成增量数据。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1031`

SchedulerOutput 最终包含：

```text
scheduled_new_reqs
scheduled_cached_reqs
num_scheduled_tokens
total_num_scheduled_tokens
scheduled_spec_decode_tokens
num_common_prefix_blocks
new_block_ids_to_zero
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1057`

核心点：

```text
SchedulerOutput 只携带 block ids 和本轮 token 数；
slot mapping 是 Worker / ModelRunner 根据 block table + positions 现场算出来的。
```

---

## 6. SchedulerOutput 中 block ids 的形态

`NewRequestData` 中的字段是：

```text
block_ids: tuple[list[int], ...]
```

位置：`code/vllm/vllm/v1/core/sched/output.py:31`

`CachedRequestData` 中的字段是：

```text
new_block_ids: list[tuple[list[int], ...] | None]
```

位置：`code/vllm/vllm/v1/core/sched/output.py:111`

这里的外层 tuple 是 KV cache group：

```text
block_ids[group_id] = [physical_block_id_0, physical_block_id_1, ...]
```

这意味着：

```text
一个 request 在不同 KV cache group 中可以有不同 block list；
attention metadata 构造时必须按 group 取对应 block table / slot mapping。
```

`CachedRequestData.resumed_req_ids` 还影响 Worker 如何解释 `new_block_ids`：

```text
普通 cached request：new_block_ids 追加到已有 block ids；
resumed request：new_block_ids 替换旧 block ids。
```

位置：`code/vllm/vllm/v1/core/sched/output.py:112`

---

## 7. KVCacheManager 分配的是 KVCacheBlocks

`KVCacheManager.allocate_slots()` 是 Scheduler 和 KV cache 分配器之间的核心接口。

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:244`

它处理的内容不只是“新分配几个 block”，还包括：

```text
prefix cache hit blocks
external KV connector computed tokens
需要新计算的 tokens
lookahead / spec decode slots
sliding window / skipped blocks
watermark / reserved blocks
encoder-decoder cross-attention blocks
```

函数注释里的布局很关键：

```text
--------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
--------------------------------------------------------------------
                                      |       to be allocated       |
                                                | to be computed |
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:290`

含义：

```text
comp：request 已经计算过的 token；
new_comp：本轮新命中的 vLLM prefix cache token；
ext_comp：connector 外部 KV 已经有的 token；
new：本轮要实际计算的新 token，包括未验证 draft token；
lookahead：spec decode / EAGLE 等预留 token。
```

真正要分配 slots 的范围大致是：

```text
ext_comp + new + lookahead
```

但真正要计算的是：

```text
new
```

这解释了一个容易混淆的点：

```text
某些 token 不需要本地 forward 计算，但仍需要本地 KV slot 和 block table entry，
因为后续 attention 要从本地 KV cache 统一读取它们。
```

---

## 8. KVCacheBlocks 是跨 KV cache group 的结果

`KVCacheBlocks` 定义：

```text
blocks: tuple[Sequence[KVCacheBlock], ...]
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:25`

语义是：

```text
blocks[group_id][block_index] = KVCacheBlock
```

它提供 `get_block_ids()`：

```text
return tuple([blk.block_id for blk in group] for group in self.blocks)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:69`

返回形态：

```text
tuple[list[int], ...]
```

外层 tuple 对应 KV cache group，内层 list 是该 group 的 physical block ids。

所以 SchedulerOutput 里的 block ids 已经是物理 block id，而不是逻辑 block index。

逻辑 block index 是后面用 position 除以 block size 得到的。

---

## 9. prefix cache 如何影响 block table

prefix cache 命中时，Scheduler 侧会先找到已计算 blocks：

```text
KVCacheManager.get_computed_blocks(request)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:202`

如果 prefix caching 关闭，或者请求不允许读取 prefix cache，则返回空 blocks。

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:214`

如果命中，它会返回：

```text
computed_blocks
num_new_computed_tokens
```

随后 `allocate_slots()` 会把这些 computed blocks 和新分配 blocks 一起纳入请求当前 blocks。

效果是：

```text
block table 中的 block 不一定都是本轮新写入的；
它可能包含 prefix cache 命中的历史 block、connector load 的 block、新分配待写 block。
```

注意一个细节：

```text
当所有 prompt token 都命中 cache 时，也通常要重算最后一个 token 来得到 logits。
```

源码里通过：

```text
max_cache_hit_length = request.num_tokens - 1
```

实现。

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:221`

所以 prefix cache 命中并不等于本轮完全没有 token 进入 slot mapping。

---

## 10. Worker 侧持久保存 block table

在 classic runner 中，Worker 侧通过 `InputBatch` 持久保存当前执行 batch。

`InputBatch` 定义位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:91`

它内部创建：

```text
self.block_table = MultiGroupBlockTable(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:170`

这里用的是：

```text
vllm.v1.worker.block_table.MultiGroupBlockTable
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:30`

这份 block table 是 Worker 侧执行态的一部分，不是 Scheduler 的数据结构。

它和 `InputBatch.req_id_to_index` 必须严格同步：

```text
request_id → req_index → block_table row
```

如果 req_index 变了，block table row 也必须移动，否则 slot mapping 会从错误的 row 读 block ids。

---

## 11. 新请求：add_row

新请求进入 Worker 时会先构造 `CachedRequestState`，其中包含 SchedulerOutput 发来的完整 `block_ids`。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1224`

加入 `InputBatch` 时：

```text
self.block_table.add_row(request.block_ids, req_index)
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:378`

`add_row` 的语义是：

```text
这是一个新 row，从头设置该 request 的 block table。
```

在底层 `BlockTable.add_row()` 中：

```text
num_blocks_per_row[row_idx] = 0
append_row(block_ids, row_idx)
```

位置：`code/vllm/vllm/v1/worker/block_table.py:120`

---

## 12. 继续请求：append_row

对已经在 Worker 缓存中的 request，SchedulerOutput 只发送新 block ids。

`GPUModelRunner._update_states()` 中：

```text
new_block_ids = req_data.new_block_ids[i]
resumed_from_preemption = req_id in req_data.resumed_req_ids
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1284`

普通继续请求：

```text
for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
    block_ids.extend(new_ids)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1370`

如果请求当前还在 `InputBatch`，还会同步更新持久 block table：

```text
self.input_batch.block_table.append_row(new_block_ids, req_index)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1400`

`append_row` 的语义是：

```text
这个 request 已经有 block table row，只追加本轮新增 block ids。
```

---

## 13. preemption resume：不能 append，要 replace

如果请求从 preemption 恢复：

```text
resumed_from_preemption = True
```

Worker 会：

```text
req_state.block_ids = new_block_ids
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1376`

并且源码断言：

```text
assert req_index is None
```

含义是：

```text
preempted request 恢复时不应该还在当前 InputBatch 里；
恢复后要作为重新加入的请求 add_request / add_row。
```

原因：

```text
preemption 会释放旧 blocks；
恢复后 KVCacheManager 可能重新分配一批新 blocks；
旧 block table row 已经无效，不能继续 append。
```

Scheduler 侧 preempt 会：

```text
_free_request_blocks(request)
request.status = PREEMPTED
request.num_computed_tokens = 0
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1105`

---

## 14. remove / condense / swap 必须同步 block table row

`InputBatch.remove_request()` 会清理 block table row：

```text
self.block_table.clear_row(req_index)
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:510`

batch 重排时会交换 row：

```text
self.block_table.swap_row(i1, i2)
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:566`

condense 时会移动 row：

```text
self.block_table.move_row(last_req_index, empty_index)
```

位置：`code/vllm/vllm/v1/worker/gpu_input_batch.py:683`

这点非常关键：

```text
InputBatch 的 req_index 是 block table 的行号；
任何 batch 行移动，都必须同步移动 token_ids、num_computed_tokens、sampling 状态和 block table row。
```

否则后面：

```text
slot_mapping = f(block_table[req_index], positions)
```

就会读错 request 的 blocks，导致 K/V 写到错误 cache 位置。

---

## 15. MultiGroupBlockTable 和 BlockTable

classic runner 使用：

```text
MultiGroupBlockTable
  → BlockTable for group 0
  → BlockTable for group 1
  → ...
```

位置：`code/vllm/vllm/v1/worker/block_table.py:223`

每个 `BlockTable` 内部维护：

```text
block_table：[max_num_reqs, max_num_blocks_per_req]
num_blocks_per_row：[max_num_reqs]
slot_mapping：[max_num_batched_tokens]
```

位置：`code/vllm/vllm/v1/worker/block_table.py:70`

一行 block table 可以理解为：

```text
block_table[req_index] = [block_id_0, block_id_1, block_id_2, ...]
```

`num_blocks_per_row[req_index]` 表示该 row 当前有效 block 数。

`slot_mapping` 则是本轮 token 级输出，不是 request 级持久状态。

---

## 16. hybrid block：allocation block size 和 kernel block size 可以不同

`BlockTable.__init__()` 会比较：

```text
block_size：KV manager / cache allocation block size
kernel_block_size：attention kernel 使用的 block size
```

位置：`code/vllm/vllm/v1/worker/block_table.py:18`

如果二者相同：

```text
blocks_per_kv_block = 1
use_hybrid_blocks = False
```

位置：`code/vllm/vllm/v1/worker/block_table.py:47`

如果不同，会进入 hybrid block 模式：

```text
allocation block size = 32
kernel block size = 16
blocks_per_kv_block = 2
physical KV block 0 → kernel block [0, 1]
physical KV block 1 → kernel block [2, 3]
```

对应函数：

```text
map_to_kernel_blocks(...)
```

位置：`code/vllm/vllm/v1/worker/block_table.py:174`

这解释了为什么：

```text
Scheduler / KVCacheManager 分配的是 allocation block id；
最终给 attention kernel 的 block table 可能是展开后的 kernel block id。
```

当前 modular GPU block table 也有类似逻辑：

```text
blocks_per_kv_block = block_size // kernel_block_size
block_ids = [b * bpk + k for b in block_ids for k in range(bpk)]
```

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:43`

---

## 17. _prepare_inputs() 何时计算 slot mapping

classic runner 的 `_prepare_inputs()` 中，slot mapping 计算发生在 token positions 准备之后。

### 17.1 先提交 block table 到 GPU

```text
self.input_batch.block_table.commit_block_table(num_reqs)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1906`

这一步是优化：

```text
先启动 block table CPU → GPU 拷贝；
随后 CPU 侧继续计算 req_indices / positions / query_start_loc；
让数据传输和 CPU 计算重叠。
```

### 17.2 计算 req_indices / query_pos / positions

例如本轮每个请求调度 token 数是：

```text
[2, 5, 3]
```

则：

```text
req_indices = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
query_pos   = [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
positions   = num_computed_tokens[req_indices] + query_pos
```

位置：

- `req_indices`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1910`
- `positions_np`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1920`

### 17.3 准备 query_start_loc

`query_start_loc` 是本轮 packed query 的 request 边界。

例如 `[2, 5, 3]`：

```text
query_start_loc = [0, 2, 7, 10]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2001`

padding 区域会填成非递减值，避免 FlashAttention 等 kernel 不接受非法边界。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2004`

### 17.4 准备 seq_lens

GPU 上最终会计算：

```text
seq_lens = num_computed_tokens + num_scheduled_tokens
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2113`

它表示每个 request 当前 attention 可见的总长度。

### 17.5 计算 slot mapping

最后调用：

```text
self.input_batch.block_table.compute_slot_mapping(
    num_reqs,
    self.query_start_loc.gpu[: num_reqs + 1],
    self.positions[:total_num_scheduled_tokens],
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2118`

输入是：

```text
num_reqs：本轮 batch 请求数；
query_start_loc：每个 request 在扁平 token batch 中的起止边界；
positions：本轮每个 token 的逻辑序列位置。
```

输出写入每个 KV cache group 对应的 `BlockTable.slot_mapping.gpu`。

---

## 18. classic slot mapping kernel 的核心逻辑

Triton kernel 定义在 `worker/block_table.py`。

位置：`code/vllm/vllm/v1/worker/block_table.py:325`

每个 request 一段 token：

```text
start_idx = query_start_loc[req_idx]
end_idx   = query_start_loc[req_idx + 1]
```

位置：`code/vllm/vllm/v1/worker/block_table.py:354`

对每个 token：

```text
pos = positions[token_index]
block_index = pos // virtual_block_size
block_number = block_table[req_idx, block_index]
virtual_block_offset = pos - block_index * virtual_block_size
local_block_offset = ...
slot_id = block_number * block_size + local_block_offset
```

位置：`code/vllm/vllm/v1/worker/block_table.py:357`

普通无 CP 情况下：

```text
virtual_block_size = block_size
local_block_offset = pos % block_size
slot_id = block_number * block_size + local_block_offset
```

这就是最直观的 slot mapping。

---

## 19. Context Parallelism 如何改变 slot mapping

如果启用 PCP / DCP，总 CP world size 会影响 slot 计算。

classic kernel 中：

```text
virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
```

位置：`code/vllm/vllm/v1/worker/block_table.py:357`

它会判断某个 token 是否属于当前 rank：

```text
is_local = (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
```

位置：`code/vllm/vllm/v1/worker/block_table.py:369`

不是本 rank 负责的 token，会写成：

```text
PAD_SLOT_ID
```

位置：`code/vllm/vllm/v1/worker/block_table.py:378`

所以 CP 下：

```text
逻辑 position 仍然是全局序列位置；
block table row 仍然描述 request 的 KV blocks；
但当前 rank 只负责其中一部分 KV slot；
不属于当前 rank 的 token 在本 rank slot_mapping 中是 PAD_SLOT_ID。
```

modular GPU kernel 中逻辑类似：

```text
block_indices = positions // (block_size * CP_SIZE)
block_offsets = positions % (block_size * CP_SIZE)
is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
local_offsets = rounds * CP_INTERLEAVE + remainder
slot_ids = block_numbers * block_size + local_offsets
slot_ids = tl.where(is_local, slot_ids, PAD_ID)
```

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:283`

---

## 20. padding slot 为什么是 -1

slot mapping kernel 最后会专门把 unused token 区域填成 `PAD_SLOT_ID`。

classic kernel：

```text
if req_idx == tl.num_programs(0) - 1:
    store PAD_ID for [num_tokens, max_num_tokens)
```

位置：`code/vllm/vllm/v1/worker/block_table.py:343`

modular GPU kernel：

```text
if batch_idx == tl.num_programs(1) - 1:
    actual_num_tokens = query_start_loc[batch_idx]
    store PAD_ID for [actual_num_tokens, max_num_tokens)
```

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:261`

目的：

```text
CUDA graph / padded batch 下，真实 token 后面会有 padded token；
这些 padded token 不能误写 KV cache；
因此 slot_mapping padded 区域必须是 -1 / PAD_SLOT_ID。
```

`_get_slot_mappings()` 也会再次保证 padded token 区域为 `-1`：

```text
slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4006`

---

## 21. block table padding 为什么用 NULL_BLOCK_ID

CUDA graph 要求形状稳定，所以 request 维度也可能 padding。

`_build_attention_metadata()` 中，block table padded rows 会填成：

```text
NULL_BLOCK_ID
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2262`

`NULL_BLOCK_ID = 0`。

位置：`code/vllm/vllm/v1/attention/backends/utils.py:46`

原因：

```text
block 0 是 null block / padding block；
padded request 不应该读到随机历史 KV；
填 NULL_BLOCK_ID 可以让 padding 行安全。
```

和 slot mapping 的区别：

```text
padded token：slot_mapping = -1，表示不写 KV；
padded request row：block_table = 0，表示读 null block。
```

---

## 22. _get_slot_mappings() 生成两种视图

slot mapping 计算完后，classic runner 会调用：

```text
GPUModelRunner._get_slot_mappings(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3960`

### 22.1 by group：给 metadata builder

```text
slot_mappings_by_gid = {
  kv_cache_group_id: slot_mapping_tensor,
  ...
}
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4012`

它会传给 `_build_attention_metadata()`：

```text
slot_mappings=slot_mappings_by_group
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4255`

### 22.2 by layer：给 forward context

```text
slot_mappings_by_layer = {
  layer_name: slot_mapping_tensor,
  ...
}
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4017`

它会进入：

```text
set_forward_context(..., slot_mapping=slot_mappings)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`

attention layer 后续按 `layer_name` 取自己的 slot mapping。

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:649`

### 22.3 ubatch 情况

如果启用 ubatching，`_get_slot_mappings()` 会按 `ubatch.token_slice` 切片，返回：

```text
list[dict[layer_name, sliced_slot_mapping]]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4023`

这样每个 ubatch 的 forward context 只看到自己的 token slice。

---

## 23. block table / slot mapping 如何进入 CommonAttentionMetadata

`_build_attention_metadata()` 会构造 `CommonAttentionMetadata`。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2208`

核心字段：

```text
CommonAttentionMetadata(
    query_start_loc=...,
    seq_lens=...,
    num_reqs=...,
    num_actual_tokens=...,
    max_query_len=...,
    max_seq_len=...,
    block_table_tensor=block_table_gid_0,
    slot_mapping=slot_mapping_gid_0,
    causal=True,
    is_prefilling=...,
    positions=...,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2330`

对于多个 KV cache group：

```text
kv_cache_gid == 0：使用 cm_base 默认 block_table / slot_mapping；
kv_cache_gid > 0 ：浅拷贝 cm_base 后替换该 group 的 block_table / slot_mapping。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2451`

之后每个 attention group 调用 builder：

```text
builder.build(common_prefix_len=..., common_attn_metadata=cm)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2431`

同一个 attention group 内的多个 layer 会共享同一份 backend-specific metadata，但 forward context 仍按 layer name 建索引。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2446`

---

## 24. block_table_tensor 用于读历史 KV

`block_table_tensor` 回答：

```text
这个 request 的历史上下文分布在哪些物理 KV blocks。
```

backend builder 会把它翻译成自己 kernel 需要的表示。

常见方式：

```text
FlashAttention：直接使用 paged KV block table；
FlashInfer：转成 paged_kv_indptr / paged_kv_indices / paged_kv_last_page_len；
FlexAttention：用 block table 构造 physical-to-logical mapping 和 block mask；
CPU backend：保存 block_table / slot_mapping 并构造 CPU scheduler metadata。
```

这就是为什么 prefix cache 命中的 blocks 也必须进入 block table：

```text
虽然本轮不重算这些 token 的 K/V，
但 attention 仍然要通过 block table 读到它们。
```

---

## 25. slot_mapping 用于写当前 K/V

标准 attention forward 中，会通过 forward context 取出：

```text
attn_metadata
attn_layer
kv_cache
layer_slot_mapping
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:649`

如果 backend 的 KV update 不包含在 attention forward 内，则会显式调用：

```text
unified_kv_cache_update(key, value, layer_name)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:692`

内部会调用：

```text
attn_layer.impl.do_kv_cache_update(..., layer_slot_mapping)
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:707`

如果 backend 自己在 forward 内完成 KV cache update，则 `slot_mapping` 通常会作为 backend metadata 的一部分或 impl.forward 的上下文参数被消费。

核心点：

```text
block table 偏读历史 KV；
slot mapping 偏写当前 K/V。
```

---

## 26. 当前代码里还有一套 modular GPU BlockTables

除了 classic `gpu_model_runner.py + gpu_input_batch.py + worker/block_table.py`，当前代码库还有 modular GPU runner：

```text
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/input_batch.py
vllm/v1/worker/gpu/block_table.py
```

其中 `gpu/model_runner.py` 会创建：

```text
self.block_tables = BlockTables(...)
```

位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:445`

这套 `BlockTables` 的设计更偏 GPU-native / staged write：

```text
BlockTables
  block_tables[group]：持久 block table staging tensor
  input_block_tables[group]：本轮 forward 使用的 gathered block table
  slot_mappings[group, token]：本轮每个 group 的 slot mapping
  num_blocks[group, req]：每个 request 当前有效 block 数
```

位置：

- `block_tables`：`code/vllm/vllm/v1/worker/gpu/block_table.py:47`
- `num_blocks`：`code/vllm/vllm/v1/worker/gpu/block_table.py:56`
- `input_block_tables`：`code/vllm/vllm/v1/worker/gpu/block_table.py:67`
- `slot_mappings`：`code/vllm/vllm/v1/worker/gpu/block_table.py:73`

这套实现和 classic 实现语义一致，但接口不同。

---

## 27. modular GPU runner 的 block table 更新路径

新请求加入时：

```text
self.block_tables.append_block_ids(
    req_index, new_req_data.block_ids, overwrite=True
)
```

位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:797`

继续请求如果有新 block ids：

```text
self.block_tables.append_block_ids(
    req_index, req_new_block_ids, overwrite=False
)
```

位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:828`

`append_block_ids()` 内部会：

```text
start = num_blocks[group, req_index]  # overwrite=False
start = 0                             # overwrite=True
写入 staged block table
更新 num_blocks[group, req_index]
```

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:107`

如果 `blocks_per_kv_block > 1`，会把 allocation block ids 展开成 kernel block ids。

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:117`

然后通过：

```text
apply_staged_writes()
```

把 staged writes 应用到 GPU block table。

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:122`

---

## 28. modular GPU runner 如何 gather block tables

modular runner 的 `_build_attention_inputs()` 中：

```text
block_tables = self.block_tables.gather_block_tables(
    input_batch.idx_mapping,
    num_reqs_padded=input_batch.num_reqs_after_padding,
)
```

位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1015`

这里有一个重要差异：

```text
持久 block table 的 row 是 req_state_idx；
本轮 forward 的 batch row 可能经过重排 / padding；
idx_mapping 用来把本轮 batch row 映射回持久 request state row。
```

`_gather_block_tables_kernel` 会：

```text
for each group_id, batch_idx:
  if batch_idx >= num_reqs:
      zero out padded row
  else:
      req_idx = idx_mapping[batch_idx]
      copy src_block_table[group_id][req_idx] → input_block_table[group_id][batch_idx]
```

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:199`

这和 classic runner 的直接 `block_table[req_index]` 不同：

```text
classic：InputBatch row 本身就是 block table row；
modular：input batch row 通过 idx_mapping 映射到持久 req state row。
```

但对 attention backend 来说，最终看到的仍然是：

```text
本轮 batch row → block ids
```

---

## 29. modular GPU runner 如何计算 slot mappings

modular runner 中：

```text
slot_mappings = self.block_tables.compute_slot_mappings(
    input_batch.idx_mapping,
    input_batch.query_start_loc,
    input_batch.positions,
    input_batch.num_tokens_after_padding,
)
```

位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1021`

`BlockTables.compute_slot_mappings()` 启动 kernel：

```text
_compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](...)
```

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:160`

kernel 的维度是：

```text
program_id(0)：KV cache group id
program_id(1)：batch row / padding sentinel
```

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:256`

核心计算：

```text
req_state_idx = idx_mapping[batch_idx]
positions = pos[token_offsets]
block_indices = positions // (block_size * CP_SIZE)
block_offsets = positions % (block_size * CP_SIZE)
block_numbers = block_table[group][req_state_idx, block_indices]
slot_ids = block_numbers * block_size + block_offsets       # CP_SIZE == 1
```

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:276`

输出形态：

```text
slot_mappings: [num_kv_cache_groups, num_tokens_padded]
```

位置：`code/vllm/vllm/v1/worker/gpu/block_table.py:73`

所以 modular 路径和 classic 路径的差异主要是工程组织：

```text
classic：每个 group 一个 BlockTable 对象，每个对象一个 slot_mapping；
modular：一个 BlockTables 对象统一管理所有 group 的 block table 和 slot_mappings。
```

地址翻译语义不变。

---

## 30. 两套实现对照

| 维度 | classic runner | modular GPU runner |
|---|---|---|
| runner 文件 | `worker/gpu_model_runner.py` | `worker/gpu/model_runner.py` |
| InputBatch 文件 | `worker/gpu_input_batch.py` | `worker/gpu/input_batch.py` |
| block table 文件 | `worker/block_table.py` | `worker/gpu/block_table.py` |
| block table 对象 | `MultiGroupBlockTable` | `BlockTables` |
| 持久 row | `InputBatch.req_index` | request state index |
| 本轮 row 映射 | row 直接对应 request | 通过 `idx_mapping` gather |
| block table 提交 | `commit_block_table(num_reqs)` | `apply_staged_writes()` + `gather_block_tables()` |
| slot mapping 输出 | 每个 `BlockTable.slot_mapping.gpu` | `slot_mappings[group, token]` |
| padding token | kernel / `_get_slot_mappings()` 填 `-1` | kernel 填 `PAD_SLOT_ID` |

共同点：

```text
1. Scheduler 仍只分配 blocks；
2. Worker 仍维护 request → block ids；
3. positions 仍是 token 逻辑序列位置；
4. slot mapping 仍是 position + block table 的展开结果；
5. attention backend 仍通过 block table 读历史 KV，通过 slot mapping 写当前 K/V。
```

---

## 31. prefill / decode 下 slot mapping 的差异

`slot_mapping` 的公式不区分 prefill / decode，但输入的 `positions` 和 `query_start_loc` 不同。

### 31.1 纯 prefill

例如两个请求第一次 prefill：

```text
num_computed_tokens = [0, 0]
num_scheduled_tokens = [4, 3]
query_start_loc = [0, 4, 7]
positions = [0, 1, 2, 3, 0, 1, 2]
```

slot mapping 会为 7 个 prompt token 生成 7 个写入位置。

特点：

```text
一次写多个 KV slots；
可能跨多个 logical blocks；
block table 需要覆盖当前 prompt chunk 使用到的 blocks。
```

### 31.2 纯 decode

例如三个请求 decode：

```text
num_computed_tokens = [128, 256, 300]
num_scheduled_tokens = [1, 1, 1]
query_start_loc = [0, 1, 2, 3]
positions = [128, 256, 300]
```

slot mapping 每个 request 通常只有一个新 slot。

特点：

```text
slot_mapping 写当前新 token；
block table 读完整历史上下文；
很多 backend 可以走 decode kernel 或 CUDA graph decode path。
```

### 31.3 chunked prefill

例如 prompt 长度 100，已算 64，本轮继续算 16：

```text
num_computed_tokens = 64
num_scheduled_tokens = 16
positions = [64, 65, ..., 79]
seq_lens = 80
```

slot mapping 只覆盖当前 chunk。

关键点：

```text
chunked prefill 仍然写 KV cache；
只是中间 chunk 的 sampled token 会被 discard_request_mask 丢弃；
block table 随请求推进持续追加或复用 blocks。
```

`discard_request_mask` 位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2029`

---

## 32. spec decode / lookahead 如何影响 block table 和 slot mapping

Spec decode 会让一次调度包含：

```text
target token + draft tokens + bonus token / lookahead slots
```

Scheduler 会通过 `num_lookahead_tokens` 给 KV manager 预留额外 slots。

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:250`

`allocate_slots()` 注释中明确：

```text
new includes unverified draft tokens
lookahead = num_lookahead_tokens
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:320`

影响：

```text
1. block table 可能包含为 draft / lookahead 分配的 blocks；
2. slot mapping 可以为乐观执行的 draft token 计算写入位置；
3. 最终只有 accepted tokens 推进请求状态；
4. rejected draft 对应的 token 进度会在 scheduler update / output update 阶段修正。
```

在 `_prepare_inputs()` 中，如果有 spec decode：

```text
num_draft_tokens[req_idx] = len(draft_token_ids)
num_decode_draft_tokens[req_idx] = draft_len  # 仅对已完成 prompt 的 decode request
spec_decode_metadata = _calc_spec_decode_metadata(...)
logits_indices = spec_decode_metadata.logits_indices
num_sampled_tokens = num_draft_tokens + 1
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2163`

注意：

```text
scheduled_spec_decode_tokens 不直接进入普通 attention kernel；
它先影响 input_ids / positions / logits_indices / spec_decode_metadata，
再间接影响 attention metadata 和 slot mapping。
```

---

## 33. KV connector / external KV load 如何影响映射

KV connector 可能让某些 tokens 的 KV 来自远端，而不是本地 forward 计算。

Scheduler 中会计算：

```text
num_external_computed_tokens
load_kv_async
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:721`

如果需要远端 load：

```text
load_kv_async = True
num_new_tokens = 0
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:781`

但仍然会调用：

```text
KVCacheManager.allocate_slots(..., num_external_computed_tokens=...)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:873`

这意味着：

```text
远端 KV 最终也要落到本地 KV cache blocks；
这些 block ids 也会进入 block table；
后续 attention 统一通过 block table 读取，不关心 KV 是本地算的还是远端 load 的。
```

如果 async load 尚未完成，请求会进入：

```text
WAITING_FOR_REMOTE_KVS
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:916`

KV load 失败时，Scheduler 会根据 invalid blocks 调整 computed token 数，触发重算或失败处理。

---

## 34. sliding window / skipped blocks 对 block table 的影响

`allocate_slots()` 在分配新 blocks 前会先调用：

```text
remove_skipped_blocks(request.request_id, total_computed_tokens)
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:394`

注释说明：

```text
Free the blocks that are skipped during the attention computation
(e.g., tokens outside the sliding window).
```

位置：`code/vllm/vllm/v1/core/kv_cache_manager.py:394`

这意味着：

```text
对 sliding window attention，过早位置的 KV block 可能被移除或替换成 null block；
block table 不一定保留从 position 0 开始的所有历史 blocks；
backend / cache coordinator 会保证可见窗口内的 block table 语义正确。
```

文档层面可以这样理解：

```text
block table 表示“当前 attention 还需要、且本地持有的 blocks”；
不是请求从出生以来所有 blocks 的永久日志。
```

---

## 35. CUDA graph padding 的完整图景

CUDA graph 需要稳定 shape，因此会产生两类 padding：

```text
request 维度 padding：num_reqs → num_reqs_padded
token 维度 padding：num_tokens → num_tokens_padded
```

在 `_build_attention_metadata()` 中：

```text
num_tokens_padded = num_tokens_padded or num_tokens
num_reqs_padded = num_reqs_padded or num_reqs
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2231`

对应安全填充值：

```text
padded request rows：block_table 填 NULL_BLOCK_ID / 0；
padded token entries：slot_mapping 填 -1 / PAD_SLOT_ID；
padded is_prefilling rows：填 False；
query_start_loc padding：保持非递减。
```

位置：

- block table padding：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2262`
- slot mapping padding：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4006`
- is_prefilling padding：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2297`
- query_start_loc padding：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2004`

目的：

```text
padding token 不写 KV cache；
padding request 不读随机 block；
CUDA graph replay 时不会因为旧数据污染 attention。
```

---

## 36. Encoder-only / cross-attention 的特殊性

不是所有 attention 都使用普通 decoder paged KV cache。

在 `_get_slot_mappings()` 中，如果是 `EncoderOnlyAttentionSpec`：

```text
slot_mapping = torch.zeros((num_tokens_padded,), dtype=torch.int64, device=self.device)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3996`

在 `_build_attention_metadata()` 中，如果是 `EncoderOnlyAttentionSpec`：

```text
block_table_tensor = torch.zeros((num_reqs_padded, 1), dtype=torch.int32, device=self.device)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2252`

含义：

```text
普通 decoder attention 强依赖 paged block table；
encoder-only / 特殊 attention 类型可能使用占位 block table / slot mapping；
具体语义由 wrapper backend 或具体 backend 解释。
```

cross-attention wrapper 通常会把 common metadata 改写为 encoder KV 视角，例如：

```text
causal = False
seq_lens 使用 encoder_seq_lens
slot_mapping 使用 encoder KV cache 对应的 mapping
```

所以不要把 decoder self-attention 的 block table 语义机械套到所有 attention 类型。

---

## 37. cascade attention 和 num_common_prefix_blocks

SchedulerOutput 中有：

```text
num_common_prefix_blocks: list[int]
```

位置：`code/vllm/vllm/v1/core/sched/output.py:205`

Scheduler 会通过 KVCacheManager 计算：

```text
num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1001`

`GPUModelRunner` 再把它转换成 builder 使用的：

```text
cascade_attn_prefix_lens[kv_cache_group_id][attn_group_idx]
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2511`

然后传给：

```text
builder.build(common_prefix_len=cascade_attn_prefix_len, ...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2431`

cascade attention 不改变 slot mapping 的基本公式。

它改变的是 backend metadata 的执行计划：

```text
共享 prefix KV 可以被单独处理；
每个 request 的 suffix / current query 仍然依赖 block table 和 slot mapping。
```

---

## 38. 一个具体地址翻译例子

假设：

```text
block_size = 16
request A 的 block_table row = [10, 25, 7]
本轮 positions = [30, 31, 32]
```

计算：

```text
position 30:
  logical_block_index = 30 // 16 = 1
  block_id = block_table[A, 1] = 25
  block_offset = 30 % 16 = 14
  slot_id = 25 * 16 + 14 = 414

position 31:
  logical_block_index = 1
  block_id = 25
  block_offset = 15
  slot_id = 415

position 32:
  logical_block_index = 2
  block_id = 7
  block_offset = 0
  slot_id = 112
```

所以本轮 slot mapping 是：

```text
[414, 415, 112]
```

如果这是 prefill 跨 block 边界，slot mapping 会自然跨到下一个 block；如果是 decode token，它通常只映射到当前最后一个 block 的某个 offset。

---

## 39. 一个 prefix cache 例子

假设：

```text
block_size = 16
prompt 长度 = 80
prefix cache 命中前 64 tokens
本轮需要重新计算 position 64..79
```

那么：

```text
prefix cache hit blocks 可能是 logical blocks 0..3
新计算 tokens 使用 logical block 4
block table row 可能类似：[cached_b0, cached_b1, cached_b2, cached_b3, new_b4]
slot_mapping 只覆盖本轮 positions 64..79
```

关键点：

```text
命中的 blocks 进入 block table；
但命中的 tokens 不进入本轮 slot_mapping；
slot_mapping 只描述本轮 scheduled tokens。
```

---

## 40. 一个 chunked prefill 例子

假设：

```text
block_size = 16
prompt 长度 = 100
第一轮调度 64 tokens
第二轮调度 32 tokens
```

第一轮：

```text
num_computed_tokens = 0
positions = 0..63
slot_mapping 写 logical blocks 0..3
```

第二轮：

```text
num_computed_tokens = 64
positions = 64..95
slot_mapping 写 logical blocks 4..5
```

此时：

```text
block table row 已经包含第一轮 blocks；
如果第二轮需要新 blocks，SchedulerOutput 会发送 new_block_ids；
Worker append_row 后，slot mapping 才能把 positions 64..95 映射到正确 physical blocks。
```

如果第二轮还不是最后 prefill chunk，sampled token 会被 `discard_request_mask` 丢弃，但 KV cache 写入仍然有效。

---

## 41. 一个 spec decode 例子

假设某 decode request 已经完成 100 tokens，本轮验证 4 个 draft tokens：

```text
num_computed_tokens = 100
num_scheduled_tokens = 5  # target + 4 draft
positions = [100, 101, 102, 103, 104]
```

slot mapping 会给 5 个 token 都计算潜在写入位置。

如果 draft 全部接受：

```text
请求进度可以推进到 105；
这些 slot 都成为有效 KV。
```

如果只接受 2 个 draft：

```text
请求最终逻辑进度会修正；
后续 scheduler / KV manager 会按接受数量处理；
被拒绝 draft 对应的乐观状态不会作为最终输出继续推进。
```

所以：

```text
slot mapping 可以为乐观执行分配物理位置；
请求最终状态由 accepted tokens 决定。
```

---

## 42. 容易疑惑的点

### 42.1 block table 是每轮重新构造的吗？

不是完全重新构造。

classic runner 中，`InputBatch` 持久保存每个 active request 的 block table row：

```text
新请求 add_row；
继续请求 append_row；
移除 / condense / swap 时同步更新 row。
```

modular runner 中，`BlockTables` 持久保存 request state row，本轮通过 `idx_mapping` gather 出 forward 使用的 `input_block_tables`。

### 42.2 slot mapping 是持久状态吗？

不是请求级持久状态。

slot mapping 是每轮根据当前：

```text
positions + query_start_loc + block table
```

计算出来的 token 级映射。它只描述本轮 scheduled tokens。

### 42.3 block id 和 slot id 是一回事吗？

不是。

```text
block_id：KV cache 页号；
slot_id：KV cache token 位置号。
slot_id = block_id * block_size + offset。
```

### 42.4 为什么 block table 里可能有 prefix cache blocks？

因为 prefix cache 命中的 blocks 是当前 request 的历史上下文。

即使不重算，也必须通过 block table 让 attention 读到它们。

### 42.5 为什么 padded slot 要填 -1？

因为 padded token 不应该写 KV cache。

`-1` / `PAD_SLOT_ID` 是告诉 KV update kernel：

```text
跳过这个 token。
```

### 42.6 为什么 padded block table row 要填 0？

因为 `0` 是 `NULL_BLOCK_ID` / null block。

padded request 不应该读到随机 block，填 null block 可以让 padded row 安全。

### 42.7 为什么有 slot_mappings_by_gid 和 slot_mappings_by_layer 两份？

因为 metadata builder 按 KV cache group 工作，而 forward context / attention layer 按 layer name 查找。

它们是同一批 tensor 的不同索引视图：

```text
group 视图：给 metadata builder；
layer 视图：给 Attention.forward / KV update。
```

### 42.8 CP 下为什么有些 slot 是 PAD_SLOT_ID？

Context parallelism 下，一个 token 的 KV 只属于某个 CP rank。

对其他 rank 来说，它不是本地 KV slot，所以 slot mapping 写 `PAD_SLOT_ID`。

### 42.9 kernel_block_size 和 block_size 为什么会不同？

KV manager 的 allocation block size 是内存管理单位；attention kernel 可能只支持某些 block size。

hybrid block 会把一个 allocation block 展开成多个 kernel block。

### 42.10 slot mapping 是否决定 attention 能看多长历史？

不完全是。

```text
slot_mapping：本轮 token 写到哪里；
seq_lens / query_start_loc：本轮 query 能看多长上下文；
block_table：这些上下文分布在哪些 blocks；
mask / backend metadata：哪些 token 可见。
```

所以 attention 可见性不是只由 slot mapping 决定。

### 42.11 block table row 的顺序是否等于 request 到达顺序？

不保证。

row 顺序由当前 `InputBatch` / model runner 的 batch 排列决定，可能因为：

```text
请求结束；
condense；
backend reorder；
ubatching；
CUDA graph padding；
modular runner idx_mapping。
```

发生变化。

### 42.12 block table 中的 block id 是否一定连续？

不一定。

物理 block 来自 block pool，可能因为复用、prefix cache、preemption、sliding window、connector load 等原因不连续。

连续的是逻辑 block index，不是物理 block id。

---

## 43. 调试入口

如果要调试 slot mapping / block table，建议按这条顺序看：

```text
1. Scheduler.schedule()
   看 num_scheduled_tokens、req_to_new_blocks、scheduled_new_reqs、scheduled_cached_reqs。

2. KVCacheManager.allocate_slots()
   看 prefix cache / external KV / new / lookahead 分别分配了哪些 blocks。

3. SchedulerOutput
   看 NewRequestData.block_ids 和 CachedRequestData.new_block_ids。

4. GPUModelRunner._update_states()
   看 req_state.block_ids 是否 append / replace 正确。

5. InputBatch.add_request() / block_table.append_row()
   看 request row 到 block ids 是否写入正确。

6. GPUModelRunner._prepare_inputs()
   看 req_indices、query_start_loc、positions、seq_lens。

7. BlockTable.compute_slot_mapping()
   看 slot_mapping 是否和 positions / block_table 对齐。

8. GPUModelRunner._get_slot_mappings()
   看 by_gid / by_layer 是否切到正确 group / layer。

9. GPUModelRunner._build_attention_metadata()
   看 CommonAttentionMetadata 中 block_table_tensor / slot_mapping。

10. get_attention_context(layer_name)
    看 attention layer 取到的 metadata、kv_cache、layer_slot_mapping。

11. backend impl.forward() / do_kv_cache_update()
    看 backend 如何消费 block table / slot mapping。
```

常看位置：

```text
code/vllm/vllm/v1/core/sched/scheduler.py:521
code/vllm/vllm/v1/core/kv_cache_manager.py:244
code/vllm/vllm/v1/worker/gpu_model_runner.py:1127
code/vllm/vllm/v1/worker/gpu_model_runner.py:1889
code/vllm/vllm/v1/worker/gpu_model_runner.py:2208
code/vllm/vllm/v1/worker/gpu_model_runner.py:3960
code/vllm/vllm/v1/worker/block_table.py:325
code/vllm/vllm/v1/worker/gpu/block_table.py:239
code/vllm/vllm/model_executor/layers/attention/attention.py:649
```

---

## 44. 最终可以记成一张表

| 阶段 | 主要函数 / 类 | 核心产物 | 作用 |
|---|---|---|---|
| KV block 分配 | `KVCacheManager.allocate_slots()` | `KVCacheBlocks` | 为 request 分配 / 复用物理 KV blocks |
| 调度输出 | `SchedulerOutput` | `block_ids` / `new_block_ids` | 把 block ids 从 Scheduler 传给 Worker |
| 持久状态更新 | `GPUModelRunner._update_states()` | `CachedRequestState.block_ids` | 更新 Worker 侧请求状态 |
| batch block table | `InputBatch.block_table` | `MultiGroupBlockTable` | 保存 request row → block ids |
| GPU 提交 | `commit_block_table()` | GPU block table tensor | 把 block table 拷到 GPU |
| token 位置计算 | `_prepare_inputs()` | `positions` / `query_start_loc` | 得到本轮 token 的逻辑位置和边界 |
| slot 计算 | `compute_slot_mapping()` | `slot_mapping` | 把本轮 token 映射到 KV slot |
| metadata 构造 | `_build_attention_metadata()` | `CommonAttentionMetadata` | 把 block table / slot mapping 交给 backend builder |
| forward 上下文 | `set_forward_context()` | `slot_mapping by layer` | attention layer 按 layer 取 slot mapping |
| attention 执行 | backend impl | KV read/write + attention output | 用 block table 读历史 KV，用 slot mapping 写当前 KV |

modular GPU runner 对应表：

| 阶段 | 主要函数 / 类 | 核心产物 | 作用 |
|---|---|---|---|
| 持久 block table | `BlockTables.append_block_ids()` | staged block ids | 更新 request state row 的 block ids |
| 写入提交 | `BlockTables.apply_staged_writes()` | GPU block table | 应用 staged writes |
| 本轮 gather | `BlockTables.gather_block_tables()` | `input_block_tables` | 通过 `idx_mapping` 生成本轮 batch block table |
| slot 计算 | `BlockTables.compute_slot_mappings()` | `[group, token] slot_mappings` | 对所有 KV cache group 计算 token 到 slot 的映射 |
| attention 输入 | `_build_attention_inputs()` | block_tables / slot_mappings | 给 metadata builder 和 forward context 使用 |

---

## 45. 总结

slot mapping 和 block table 的完整链路可以压缩成：

```text
Scheduler 决定本轮跑哪些 token
  → KVCacheManager 分配 / 复用物理 KV blocks
  → SchedulerOutput 携带 block ids
  → Worker / InputBatch 持久保存 request 的 block table row
  → _prepare_inputs 计算 positions / query_start_loc / seq_lens
  → BlockTable kernel 计算 token 级 slot_mapping
  → _build_attention_metadata 把 block_table + slot_mapping 给 backend
  → set_forward_context 把 layer 级 slot_mapping 给 Attention.forward
  → backend 用 slot_mapping 写 K/V，用 block_table 读历史 K/V
```

如果只记一句话：

```text
block table 是 request 到 KV page 的映射，slot mapping 是本轮 token 到 KV slot 的映射；前者让 attention 找到历史上下文，后者让当前 token 的 K/V 写到正确位置。
```

再压缩成最小心智模型：

```text
Scheduler 分 block；
InputBatch 记 block table；
_prepare_inputs 算 position；
BlockTable 算 slot；
metadata builder 组织给 backend；
attention kernel 按这些地址读写 KV cache。
```
