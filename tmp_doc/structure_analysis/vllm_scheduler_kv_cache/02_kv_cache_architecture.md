# 02. vLLM V1 KV Cache 架构

## 1. 核心文件

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:26`：`KVCacheBlocks`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:110`：`KVCacheManager`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:60`：`KVCacheCoordinator`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:368`：`KVCacheCoordinatorNoPrefixCache`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:418`：`UnitaryKVCacheCoordinator`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:505`：`HybridKVCacheCoordinator`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:32`：`SingleTypeKVCacheManager`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:538`：`FullAttentionManager`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:599`：`SlidingWindowManager`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:806`：`ChunkedLocalAttentionManager`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:956`：`MambaManager`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:1234`：`CrossAttentionManager`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:34`：`BlockHashToBlockMap`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:130`：`BlockPool`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_utils.py:116`：`KVCacheBlock`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_utils.py:165`：`FreeKVCacheBlockQueue`

## 2. 分层结构

V1 的 KV Cache 管理可以分成四层：

```text
Scheduler
  -> KVCacheManager
      -> KVCacheCoordinator
          -> SingleTypeKVCacheManager per KV cache group/type
              -> BlockPool
                  -> KVCacheBlock / FreeKVCacheBlockQueue / BlockHashToBlockMap
```

### 2.1 Scheduler 层

Scheduler 只关心：

- 请求能否调度。
- 请求已有多少 computed tokens。
- 新增多少 tokens 要执行。
- 是否命中 prefix cache。
- 是否需要外部 KV connector。
- 分配出来的 block ids 是什么。

Scheduler 不直接维护 free list，也不直接驱逐 block。

### 2.2 KVCacheManager 层

`KVCacheManager` 是 scheduler 的统一入口。

它隐藏了 coordinator、single type manager、block pool 的内部复杂度，并向 scheduler 暴露这些方法：

- `get_computed_blocks(request)`：查 prefix cache。
- `allocate_slots(...)`：为请求分配 slots/blocks。
- `free(request)`：释放请求 blocks。
- `get_blocks(request_id)` / `get_block_ids(request_id)`：拿请求当前 block。
- `cache_blocks(request, num_computed_tokens)`：提交可缓存 block。
- `reset_prefix_cache()`：重置 prefix cache。
- `take_events()`：取 KV cache events。

### 2.3 KVCacheCoordinator 层

`KVCacheCoordinator` 负责协调多个 KV cache group。

一个模型可能同时有：

- full attention group
- sliding window group
- chunked local attention group
- MLA group
- Mamba state group
- cross attention group

这些 group 的 block size、缓存行为、窗口规则、是否保留全部历史都可能不同。Coordinator 把这些 group 统一包装给 `KVCacheManager`。

### 2.4 SingleTypeKVCacheManager 层

每个 `SingleTypeKVCacheManager` 管理一种 cache spec。

它负责：

- 每个 request 的 block 列表：`req_to_blocks`
- 已缓存 block 数：`num_cached_block`
- 新分配 block ids：`new_block_ids`
- 不同 attention 类型的 prefix hit、跳过窗口、回收逻辑

不同派生类处理不同语义：

- `FullAttentionManager`：完整保留上下文。
- `SlidingWindowManager`：只保留滑动窗口内需要的 blocks，窗口外可替换成 null block 或释放。
- `ChunkedLocalAttentionManager`：局部块注意力。
- `MambaManager`：状态空间模型 cache，语义不同于普通 KV block。
- `CrossAttentionManager`：encoder-decoder cross attention 的静态分配。

### 2.5 BlockPool 层

`BlockPool` 是所有实际 block 元数据的池。

它负责：

- 创建 `KVCacheBlock` 列表。
- 维护 free block 双向链表。
- 维护 prefix cache hash -> block 映射。
- 分配、释放、touch、evict。
- 维护 null block。
- 发出 KV cache events。

## 3. KVCacheBlocks：Scheduler 和 Manager 的接口对象

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:26`

`KVCacheBlocks` 是 scheduler 和 KV manager 之间的结果容器。

字段：

```text
blocks: tuple[Sequence[KVCacheBlock], ...]
```

外层 tuple 对应 KV cache group；内层 sequence 是该 group 下的 block 列表。

为什么不是“token block 作为外层”？因为未来不同 group 可能拥有不同 block_size。当前大多数情况下 group 数和 block 数对齐，但代码已经为更复杂场景保留扩展。

关键方法：

- `get_block_ids()`：转成 worker 能使用的 block id 列表。
- `get_unhashed_block_ids()`：找还没有 hash 的 block。
- `get_unhashed_block_ids_all_groups()`：多 group 版本。
- `new_empty()`：创建同 group 数的空 blocks。
- `__add__()`：合并两组 blocks。

`KVCacheManager` 内部预构造 `empty_kv_cache_blocks`，避免频繁创建空对象造成 GC 压力，位置在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:172`。

## 4. KVCacheManager

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:110`

### 4.1 初始化

`__init__` 接收：

- `kv_cache_config`
- `max_model_len`
- `scheduler_block_size`
- `hash_block_size`
- `max_num_batched_tokens`
- `enable_caching`
- `use_eagle`
- `log_stats`
- `enable_kv_cache_events`
- `dcp_world_size`
- `pcp_world_size`
- `watermark`

核心初始化动作：

1. 创建 coordinator：`get_kv_cache_coordinator(...)`。
2. 记录 `num_kv_cache_groups`。
3. 暴露 coordinator 的 `block_pool`。
4. 计算 watermark blocks。
5. 准备 KV cache event metadata。
6. 准备 `empty_kv_cache_blocks`。

### 4.2 usage

`usage` 在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:181`。

它直接返回：

```text
block_pool.get_usage()
```

即 block pool 视角下的使用率。

### 4.3 get_computed_blocks()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:202`

职责：查询本地 prefix cache 命中。

流程：

1. 如果 prefix caching 禁用，或请求要求跳过读取 prefix cache，则返回空。
2. 设置 `max_cache_hit_length = request.num_tokens - 1`。
   - 即使全部 prompt 命中，也必须重新计算最后一个 token 以获取 logits。
3. 调 coordinator 的 `find_longest_cache_hit(request.block_hashes, max_cache_hit_length)`。
4. 记录 prefix cache stats。
5. 返回 `KVCacheBlocks` 与命中 token 数。

关键语义：prefix cache 命中必须是完整 block；尾部不完整 block 不会直接当 computed block 复用。

### 4.4 allocate_slots()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:244`

这是 KV 分配的核心方法。

参数语义：

- `request`：目标请求。
- `num_new_tokens`：本轮新增计算 token 数。
- `num_new_computed_tokens`：新命中的本地 prefix cache token 数。
- `new_computed_blocks`：对应本地命中的 blocks。
- `num_lookahead_tokens`：spec decode 需要额外预留的 tokens。
- `num_external_computed_tokens`：connector 外部命中的 tokens。
- `delay_cache_blocks`：是否延迟 cache blocks。P/D remote KV load 常用。
- `num_encoder_tokens`：cross-attention encoder tokens。
- `full_sequence_must_fit`：admission gate，要求整段序列能容纳。
- `reserved_blocks`：为其他 in-flight prefills 预留的 free blocks。
- `has_scheduled_reqs`：是否已有调度请求，用于决定 watermark 是否生效。

源码中的 layout 注释在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:290`，核心区间：

```text
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
```

含义：

- `comp`：请求之前已经 computed 的 tokens。
- `new_comp`：本轮新命中的本地 prefix cache tokens。
- `ext_comp`：外部 connector 已经缓存的 tokens。
- `new`：本轮要实际计算的 tokens。
- `lookahead`：spec decode 预留 tokens。

分配三阶段：

1. 释放 `comp` 中不再需要的 blocks，并检查 free blocks 是否足够。
2. 处理 prefix tokens：local computed + external computed。
3. 为新计算 tokens 和 lookahead tokens 分配新 blocks。

### 4.5 allocate_slots() 内部关键步骤

1. 计算 `num_local_computed_tokens`：

```text
request.num_computed_tokens + num_new_computed_tokens
```

2. 计算 `total_computed_tokens`：

```text
num_local_computed_tokens + num_external_computed_tokens
```

3. 根据请求状态决定 watermark：

- waiting/preempted 请求且本轮已有 scheduled req 时生效。
- running 请求不套 waiting admission watermark。

4. 如果 `full_sequence_must_fit`，先做整段容量检查。

5. 计算 `num_tokens_need_slot`：

```text
min(total_computed_tokens + num_new_tokens + num_lookahead_tokens, max_model_len)
```

6. 调 coordinator 的 `remove_skipped_blocks()`。
   - sliding window 或 local attention 会释放窗口外 block。

7. 调 coordinator 的 `get_num_blocks_to_allocate()`。

8. 检查：

```text
required_blocks = num_blocks_to_allocate + watermark_blocks
required_blocks <= block_pool.get_num_free_blocks() - reserved_blocks
```

9. 如存在新 computed blocks 或 external computed tokens，调用：

```text
coordinator.allocate_new_computed_blocks(...)
```

10. 调用：

```text
coordinator.allocate_new_blocks(...)
```

11. 如果启用 caching 且未 `delay_cache_blocks`，调用：

```text
coordinator.cache_blocks(request, num_tokens_to_cache)
```

12. 返回 `KVCacheBlocks`。

## 5. KVCacheCoordinator

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:60`

### 5.1 初始化

核心字段：

- `self.kv_cache_config`
- `self.max_model_len`
- `self.enable_caching`
- `self.scheduler_block_size`
- `self.block_pool`
- `self.eagle_group_ids`
- `self.single_type_managers`
- `self.retention_interval`

`self.block_pool` 在 coordinator 层创建：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:90`。

每个 KV cache group 创建对应 single type manager：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:106`。

### 5.2 关键方法

- `get_num_blocks_to_allocate()`：汇总每个 group 需要分配的 block 数。
- `allocate_new_computed_blocks()`：两阶段处理本地命中与外部 computed blocks。
- `allocate_new_blocks()`：为新 token 分配真实 blocks。
- `cache_blocks()`：提交 prefix cache。
- `free()`：释放请求。
- `pop_blocks_for_free()`：deferred free 时先弹出 bookkeeping。
- `get_num_common_prefix_blocks()`：计算 running 请求共享的 prefix blocks。
- `find_longest_cache_hit()`：查询最长 prefix cache 命中。
- `new_step_starts()`：每步调度开始时刷新状态。

### 5.3 Coordinator 类型

- `KVCacheCoordinatorNoPrefixCache`
  - prefix cache 禁用时使用。
- `UnitaryKVCacheCoordinator`
  - 单一类型/单一 group 简化路径。
- `HybridKVCacheCoordinator`
  - 多种 cache group 混合，如 full attention + mamba + sliding window。

Hybrid 模式下，命中可能按 group 分别计算，并需要取满足所有 group 的共同可用位置。

## 6. SingleTypeKVCacheManager

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:32`

### 6.1 核心字段

- `self.scheduler_block_size`
- `self.block_size`
- `self.dcp_world_size`
- `self.pcp_world_size`
- `self.kv_cache_spec`
- `self.block_pool`
- `self.enable_caching`
- `self.new_block_ids`
- `self.req_to_blocks`
- `self.num_cached_block`
- `self.kv_cache_group_id`
- `self._null_block`
- `self.use_eagle`

### 6.2 req_to_blocks

`req_to_blocks: defaultdict[str, list[KVCacheBlock]]`

这是每个请求持有哪些 block 的核心索引。

注意：

- block list 中可能包含真实 block。
- 对 sliding window 等类型，窗口外跳过区域可能被 null block 占位。
- request 完成或抢占时，需要根据该列表释放或弹出 blocks。

### 6.3 num_cached_block

`num_cached_block: dict[str, int]`

记录某个 request 已经被缓存的 block 数。

用途：

- 避免重复缓存同一 block。
- 区分 running 请求和首次调度请求。
- prefix hit 后，将 computed blocks 计入 cached 区间。

### 6.4 get_num_blocks_to_allocate()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:101`

核心逻辑：

1. `num_required_blocks = cdiv(num_tokens, block_size)`。
2. 如果是 recycling-aware specs 且 admission cap 开启，限制每请求最大占用。
3. 获取当前请求已有 blocks。
4. 如果是 running 请求，没有新 prefix hit，直接计算差值。
5. 计算 skipped tokens 与 skipped blocks。
6. 计算还需要新增多少 block。
7. 如果 prefix hit blocks 当前在 free queue 中可被驱逐，需要把它们也计入 capacity check，因为 touch 后它们不再 free。

### 6.5 add_local_computed_blocks()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:182`

作用：把本地 prefix cache 命中的 blocks 加到请求上。

关键步骤：

1. 计算 skipped tokens。
2. 对被滑窗跳过的 blocks 做截断。
3. 如果开启 caching，`block_pool.touch(new_computed_blocks)`，避免命中的 block 被 eviction。
4. 在 req blocks 前面补 null blocks 表示 skipped 区域。
5. 把命中的 computed blocks 加入 req blocks。
6. 设置 `num_cached_block[request_id]`。

### 6.6 allocate_external_computed_blocks()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:234`

用于 KV connector 场景：外部已经 computed，但本地还需要 block 来接收这些 KV。

必须在所有 group 的 local computed blocks 已经 touch 后运行，避免一个 group 的 external allocation 驱逐另一个 group 尚未 touch 的 local cache hit blocks。

### 6.7 allocate_new_blocks()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:277`

为请求实际新增 block。新增 block ids 会进入 `new_block_ids`，scheduler 后续通过 `take_new_block_ids()` 传给 worker 清零。

### 6.8 cache_blocks()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:316`

用于把已经完整计算的 blocks 变成 prefix cache 可复用 blocks。

它会基于 request 的 `block_hashes` 给 full block 写入 hash，并插入 `BlockPool.cached_block_hash_to_block`。

### 6.9 remove_skipped_blocks()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:477`

用于滑窗、局部注意力等场景：computed tokens 前面已经不会再被 attention 访问的 blocks 可以释放，并用 null block 占位，保持位置语义。

## 7. BlockPool

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:130`

### 7.1 初始化字段

- `self.num_gpu_blocks`
- `self.enable_caching`
- `self.hash_block_size`
- `self.blocks`
- `self.free_block_queue`
- `self.cached_block_hash_to_block`
- `self.null_block`
- `self.enable_kv_cache_events`
- `self.kv_event_queue`
- `self.metrics_collector`

`self.blocks` 是所有 block 元数据：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:162`。

`self.free_block_queue` 是 free list：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:168`。

`self.null_block` 是从 free queue 中取出的特殊 block：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:176`。

### 7.2 null block

null block 用于占位，尤其是 sliding window/跳过 block 场景。

特点：

- `block_id` 来自实际 pool，但被标记为 `is_null = True`。
- 不参与正常缓存/释放。
- ref count 不按普通 block 维护。

### 7.3 BlockHashToBlockMap

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:34`

用途：prefix cache 的 hash -> block 索引。

内部结构：

```text
dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
```

为什么 value 可能是单 block 或 dict：

- 大多数 hash 只对应一个 block，用单对象减少内存和 GC。
- 如果出现多个相同 hash 的 block，再升级成 dict。

它没有去重已缓存 blocks。原因是 block table 要保持 append-only 语义，不能因为发现相同内容就替换已分配 block id。

### 7.4 get_cached_block()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:184`

输入：

- `block_hash`
- `kv_cache_group_ids`

它会对每个 group id 生成 `BlockHashWithGroupId`，分别查找。如果任意 group miss，则整体 miss。

### 7.5 cache_full_blocks()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:211`

职责：把 request 中已经完整计算的 blocks 写入 prefix cache。

它会：

1. 读取 request 预先计算好的 block hashes。
2. 给 block 设置 hash metadata。
3. 插入 `cached_block_hash_to_block`。
4. 可能产生 `BlockStored` KV cache event。

## 8. KVCacheBlock 与 FreeKVCacheBlockQueue

### 8.1 KVCacheBlock

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_utils.py:116`

字段：

- `block_id`
- `ref_cnt`
- `_block_hash`
- `prev_free_block`
- `next_free_block`
- `is_null`

语义：

- `ref_cnt > 0` 表示正在被请求引用。
- `ref_cnt == 0` 且有 hash 的 block 可以作为 prefix cache eviction candidate。
- `_block_hash` 只能在 block full 且 cached 后设置。
- `reset_hash()` 在 eviction 时清除 hash。

### 8.2 FreeKVCacheBlockQueue

定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_utils.py:165`

为什么不用 Python `deque`：

- 需要 O(1) 删除链表中间的 block。
- eviction/touch 会把某些 cached blocks 从 free queue 中移除。
- 为减少 GC，直接操作 block 上的 `prev_free_block` / `next_free_block`。

队列顺序语义：

1. 初始按 block id 排序。
2. block 被释放后按 eviction 顺序放回。
3. LRU block 在队头。
4. 同一时间释放的 block 中，tail block 更靠前，优先驱逐。

## 9. Block Hash 与 Prefix Cache

相关定义：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_utils.py:40`

- `BlockHash = NewType("BlockHash", bytes)`
- `BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)`

### 9.1 为什么 hash 要带 group id

同样 token 内容在不同 KV cache group 中可能对应不同 layout、不同 attention 语义、不同 sliding window 规则，所以不能跨 group 复用。

`make_block_hash_with_group_id()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_utils.py:56`。

它把 block hash 与 group id 打包成 bytes：

```text
block_hash + group_id.to_bytes(4, "big")
```

对应解包函数：

- `get_block_hash()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_utils.py:68`
- `get_group_id()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_utils.py:73`

### 9.2 Request 中的 block_hashes

`Request` 初始化时会调用 `update_block_hashes()`，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:184`。

`update_block_hashes()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:237`。

当生成新 token 时，`append_output_token_ids()` 也会更新 block hashes，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/request.py:224`。

### 9.3 Prefix cache 生命周期

```text
Request 创建 / token 追加
  -> update_block_hashes()
  -> Scheduler 调度 waiting 请求
  -> KVCacheManager.get_computed_blocks()
  -> Coordinator.find_longest_cache_hit()
  -> BlockPool.get_cached_block()
  -> 命中 blocks 被 touch 并加入 req_to_blocks
  -> 新计算完成的 full blocks 通过 cache_blocks() 插入 prefix cache
```

## 10. Watermark 与 admission

`KVCacheManager` 中的 `watermark_blocks` 在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:160` 附近计算。

它的作用：

- waiting/preempted 请求入场时保留一些 free blocks。
- 避免系统刚接纳新请求就频繁抢占。

在 `allocate_slots()` 中，只有满足以下条件才应用 watermark：

- `has_scheduled_reqs` 为 True。
- request status 是 `WAITING` 或 `PREEMPTED`。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:363`。

## 11. Eviction 与 Free

### 11.1 free request

Scheduler 调用：

```text
kv_cache_manager.free(request)
```

`KVCacheManager.free()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:460`，内部委托给 coordinator。

### 11.2 free 的反向顺序

释放 blocks 时通常按反向顺序归还，使 tail blocks 更早被 eviction。这与 `FreeKVCacheBlockQueue` 的注释一致：tail block 代表更长 hash chain 的后部，优先驱逐对 prefix 命中影响较小。

### 11.3 evict_blocks()

`KVCacheManager.evict_blocks()` 在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:496`。

它允许按 block id 从 prefix cache 中驱逐 blocks，底层调用 `block_pool.evict_blocks(block_ids)`。

## 12. 初始化与容量计算

EngineCore 初始化 KV cache 的核心在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:240`。

流程：

1. `register_all_kvcache_specs(vllm_config)`。
2. `model_executor.get_kv_cache_specs()` 获取模型各层 KV cache spec。
3. 如果存在非因果 attention，禁用 chunked prefill 和 prefix caching。
4. `model_executor.determine_available_memory()` profile 可用显存。
5. `get_kv_cache_configs(...)` 计算 worker 侧 KV cache config。
6. `generate_scheduler_kv_cache_config(...)` 生成 scheduler 侧统一配置。
7. 写回 `cache_config.num_gpu_blocks`、`block_size`、`kv_cache_size_tokens`、`kv_cache_max_concurrency`。
8. `model_executor.initialize_from_config(kv_cache_configs)` 真正初始化 worker 侧 KV cache tensor 并 warmup。

关键位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:294` 到 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/engine/core.py:321`。

## 13. Scheduler block size、hash block size、实际 block size

几个概念需要区分：

- `scheduler_block_size`
  - 调度粒度，通常是各 group block size 的 LCM。
- `hash_block_size`
  - prefix cache hash 的粒度。
- `kv_cache_spec.block_size`
  - 某个 group 的实际 KV cache block size。
- `storage_block_size`
  - 某些压缩/特殊 KV spec 中存储用 block size 可能不同。
- `kernel_block_size`
  - worker attention backend 支持的 kernel block size。

Coordinator 初始化时要求：

```text
scheduler_block_size % hash_block_size == 0
scheduler_block_size % each_group.block_size == 0
```

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:82`。

worker 侧 `attn_utils.py` 会进一步根据 backend 支持确定 kernel block size。

## 14. 一句话总结

KV Cache 管理层的核心是：用 `BlockPool` 管理物理 block，用 `SingleTypeKVCacheManager` 管理不同 attention/cache 类型的请求级 block 列表，用 `KVCacheCoordinator` 统一多 group 行为，用 `KVCacheManager` 给 scheduler 提供简单的查询、分配、释放接口。
