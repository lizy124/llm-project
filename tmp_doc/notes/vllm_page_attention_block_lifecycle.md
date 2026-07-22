# vLLM V1 Page Attention / Block 生命周期说明

本文按代码路径整理 `vllm` 里 page attention（paged attention）相关的 block 生命周期：**什么时候申请 block、谁申请、申请后怎么传给 kernel、kernel 如何读写、最后如何释放**。

这里主要看 V1 GPU 主路径，默认 attention backend 可以是 FlashAttention 或 Triton，但它们都遵守同一套调度与 block table 约定。

## 1. 总体结论

一条请求从进入到退出，关键链路是：

1. 请求进入 `Scheduler`。
2. `Scheduler.schedule()` 根据本轮要执行的 token 数，调用 `KVCacheManager.allocate_slots()` 申请 block。
3. `KVCacheManager` 再下沉到 `KVCacheCoordinator`、`SingleTypeKVCacheManager`、`BlockPool`，真正从 free queue 里拿出 GPU block。
4. 调度结果里会带上 `block_ids`，worker 侧把它们写入 `InputBatch.block_table`。
5. `_prepare_inputs()` 先把 block table 复制到 GPU，再计算 `slot_mapping`。
6. attention kernel 用 `block_table` 读取历史 KV，用 `slot_mapping` 把当前 token 写进 KV cache。
7. 请求结束、被抢占、或被外部 finish 时，scheduler 释放 block；如果存在并发 in-flight 写，则会延迟释放。

## 2. 什么时候需要申请 block

申请 block 的时机在 `Scheduler.schedule()` 里。只要某个 request 本轮要推进计算，就会计算 `num_new_tokens`，然后走：

- running requests：`Scheduler.schedule()` 中先处理 `self.running`
- waiting requests：随后处理 `self.waiting` / `self.skipped_waiting`

对应代码：`Scheduler.schedule()` 在遇到请求后调用：

- `self.kv_cache_manager.allocate_slots(...)`

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:569`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:942`

### 谁来申请

严格说是 **Scheduler 发起申请**，但真正分配发生在这条链上：

- `Scheduler.schedule()`
- `KVCacheManager.allocate_slots()`
- `KVCacheCoordinator.allocate_new_blocks()` / `allocate_new_computed_blocks()`
- `SingleTypeKVCacheManager.allocate_new_blocks()` / `cache_blocks()`
- `BlockPool.get_new_blocks()`

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:269`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:238`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:321`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:647`

## 3. 申请时到底做了什么

### 3.1 Scheduler 先做 admission

`Scheduler.schedule()` 会先算：

- 这个 request 本轮要新增多少 token
- 是否有 prefix cache 命中
- 是否需要 lookahead/speculative token
- 是否要处理 sliding window / Mamba 对齐 / encoder cache

然后才进入 block 申请。

如果分配失败，Scheduler 会按策略抢占一个低优先级 running request，释放它的 block，再重试。

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:569`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:583`

### 3.2 KVCacheManager 负责“算需要多少 block”

`KVCacheManager.allocate_slots()` 不是简单取块，它先做三件事：

1. 先把不再需要的 skipped blocks 释放掉。
2. 计算本 request 当前到底需要多少物理 block。
3. 检查 free pool 够不够，不够就返回 `None`。

它内部会调用 coordinator 的：

- `remove_skipped_blocks()`
- `get_num_blocks_to_allocate()`
- `allocate_new_computed_blocks()`
- `allocate_new_blocks()`

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:269`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:420`

### 3.3 Coordinator 负责按 KV group 分配

`KVCacheCoordinator` 负责把一个 request 的 block 申请拆到每个 KV cache group：

- 单一 attention group 走 `UnitaryKVCacheCoordinator`
- 混合 attention group 走 `HybridKVCacheCoordinator`

它会把每组的 manager 串起来做一致的分配和缓存。

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:432`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:519`

### 3.4 SingleTypeKVCacheManager 真正维护 request->blocks

每个 group 对应一个 `SingleTypeKVCacheManager` 子类，负责：

- `req_to_blocks[request_id]`：这个 request 持有哪些块
- `num_cached_block[request_id]`：哪些块已经进入 prefix cache
- partial hit / CoW 复制逻辑
- `free()` 时把块返还给 block pool

关键点：

- `get_new_blocks()` 从 `BlockPool` 取出真正的新 block
- `touch()` 会增加引用计数，避免 prefix cache 命中块被回收
- `free_blocks()` 释放时按 reverse 顺序进 free queue，保证尾块先回收

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:94`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:321`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:647`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:719`

### 3.5 完整调用链路（"下沉"过程）

从 `KVCacheManager.allocate_slots()` 到真正从 free queue 拿出 GPU block 的完整调用链：

```
KVCacheManager.allocate_slots(request, num_new_tokens, num_lookahead_tokens)
  │
  ├─① coordinator.get_num_blocks_to_allocate(...)     // 算需要多少 block
  │    │  num_tokens = total_computed_tokens + num_new_tokens + num_lookahead_tokens
  │    │  num_required_blocks = cdiv(num_tokens, block_size)
  │    │  return max(num_required_blocks - len(req_blocks), 0)
  │    │
  │    └─ for each SingleTypeKVCacheManager:
  │         manager.get_num_blocks_to_allocate()
  │
  ├─② coordinator.remove_skipped_blocks(...)           // 清理滑出滑动窗口的 block
  │
  ├─③ coordinator.allocate_new_computed_blocks(...)    // 处理 prefix cache 命中的 block
  │    └─ 将命中的 block 加入 req_to_blocks（不分配新块，只建立引用）
  │
  ├─④ coordinator.allocate_new_blocks(...)             // 真正分配新 block
  │    └─ for each SingleTypeKVCacheManager:
  │         manager.allocate_new_blocks()
  │           │
  │           ├─ num_new_blocks = cdiv(num_tokens, block_size) - len(req_blocks)
  │           │
  │           └─ self.block_pool.get_new_blocks(num_new_blocks)  ← 从 free queue 拿
  │                │
  │                └─ self.free_block_queue.popleft_n(num_blocks)  ← 出队
  │                     for each block:
  │                       block.ref_cnt += 1                      ← 标记引用计数
  │
  └─⑤ coordinator.cache_blocks(request, num_tokens_to_cache)  // 将 block 计入 prefix cache
```

**关键点：**

- **KVCacheManager**：调度入口，协调整个分配流程，先算量、再清理、再分配
- **KVCacheCoordinator**：遍历所有 `single_type_managers`（每个对应一个 KV cache group，如 layer 分层），对每个 group 分别操作
- **SingleTypeKVCacheManager**：维护 `req_to_blocks` 映射表（每个请求持有哪些 block），计算差值后调用 `block_pool`
- **BlockPool**：block 资源池，`get_new_blocks()` 从 `free_block_queue`（双端队列）左侧弹出空闲 block，标记 `ref_cnt += 1`

**为什么需要多个 `SingleTypeKVCacheManager`？**

`SingleTypeKVCacheManager` 管理的不是"一个 block 池"，而是"一种 block 生命周期策略"。不同 attention 类型的 block 该什么时候分配、什么时候释放、什么时候被 cache 命中，规则都不一样，必须各自独立。例如 `FullAttentionManager` 永不释放 block，`SlidingWindowManager` 按窗口滑动释放，`ChunkedLocalAttentionManager` 按 chunk 边界释放，`CrossAttentionManager` 管理 encoder KV cache——每种策略对应一个独立的 manager 实例，各自维护自己的 `req_to_blocks` 和 `block_pool`。

**`num_new_tokens` 与 `request.num_computed_tokens` 的分工：**

- `num_new_tokens`：本轮调度要新增的 token 数（scheduler 的调度决策），驱动"总共需要多少 block"
- `request.num_computed_tokens`：已计算过的 token 数（绝对位置），用于计算总 token 数的起点和滑动窗口清理
- 已有多少 block 来自 `self.req_to_blocks`（manager 内部维护），不是从 request 字段直接算出的
- 新增 block 数 = `cdiv(总token数, block_size) - len(req_blocks)`，本质由 `num_new_tokens` 驱动

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:269`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:420`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:130`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_coordinator.py:238`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:135`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/single_type_kv_cache_manager.py:321`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:647`

## 4. block 申请完之后怎么使用

Scheduler 在 `SchedulerOutput` 里把新分配的 block 以 `block_ids` 形式带出去：

- 新请求走 `NewRequestData.block_ids`
- 运行中的请求走 `CachedRequestData.new_block_ids`

worker 侧收到后会：

1. 更新 `CachedRequestState.block_ids`
2. 把 block_ids 写入 `InputBatch.block_table`
3. 把新增的 token id 写进 CPU batch
4. 最后在 `_prepare_inputs()` 里把 block table 和 slot mapping 送到 GPU

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:1084`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_model_runner.py:1411`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_model_runner.py:1441`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_model_runner.py:1947`

## 5. block table 和 slot mapping 是什么关系

这两张表是 page attention 的核心：

- `block_table`：每个 request 的第几个逻辑 block，对应哪个物理 GPU block id
- `slot_mapping`：每个 token 最终落到哪个 KV cache slot

### 5.1 block table

`BlockTable` 里保存的是按 request 排列的 block id 行表。

如果 kernel 的 block size 和调度 block size 一样，就直接一对一映射；如果不一样，`BlockTable` 会把一个调度块拆成多个 kernel block。

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/block_table.py:24`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/block_table.py:58`

### 5.2 slot mapping

`compute_slot_mapping()` 会根据：

- request 的 `query_start_loc`
- 每个 token 的 `positions`
- `block_table`

计算出每个 token 的物理 KV slot。

核心公式在 Triton kernel 里很直白：

- `block_numbers = block_table[row, position // block_size]`
- `slot_ids = block_numbers * block_size + position % block_size`

对非本地 CP/DCP token，会写成 `PAD_SLOT_ID`。

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/block_table.py:153`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/block_table.py:348`

## 6. worker 侧什么时候把 block table 交给 GPU

在 `_prepare_inputs()` 里，worker 先：

1. `commit_block_table(num_reqs)`，把 CPU 侧 block table 拷到 GPU
2. 计算每个 token 的 `positions`
3. 调 `compute_slot_mapping()` 生成 slot mapping
4. 再构建 attention metadata

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_model_runner.py:1947`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu_model_runner.py:2167`

## 7. kernel 层具体怎么操作

这里分成两类：**写 KV cache** 和 **读 KV cache**。

### 7.1 写 KV cache：slot_mapping scatter

当模型 forward 产生新的 K/V 后，backend 会调用写入路径：

- FlashAttention：`reshape_and_cache_flash(...)`
- Triton：`triton_reshape_and_cache_flash(...)`

它们都接收 `slot_mapping`，然后把每个 token 的 K/V scatter 到对应的物理 slot 中。

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/backends/flash_attn.py:1067`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/backends/triton_attn.py:787`

### 7.2 读 KV cache：block_table paging

attention 前向时，backend 会把 `block_table` 传进 kernel：

- FlashAttention：`flash_attn_varlen_func(..., block_table=block_table, ...)`
- Triton：`unified_attention(..., block_table=block_table, ...)`

kernel 侧按 token 的 block index 去查 block_table，定位物理 KV block，再根据 block 内偏移读 K/V。

#### FlashAttention

FlashAttention backend 里，`FlashAttentionMetadata` 持有：

- `block_table`
- `slot_mapping`
- `seq_lens`
- `query_start_loc`

forward 时直接把 `block_table` 传给 `flash_attn_varlen_func()`。

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/backends/flash_attn.py:228`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/backends/flash_attn.py:1010`
  - 这里 `flash_attn_varlen_func` 的参数里包含 `block_table`

#### Triton

Triton 的 unified attention kernel 更直观：

- `physical_block_idx = tl.load(block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE)`
- 再用 `seq_offset % BLOCK_SIZE` 算 block 内偏移
- 然后拼出 K/V 的真实地址

如果启用 tensor descriptor 路径，逻辑一样，只是 load 方式更适合硬件优化。

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:420`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:424`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:468`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/attention/ops/triton_unified_attention.py:474`

## 8. 什么时候释放 block

释放有两种场景：

### 8.1 request 正常结束或被外部 finish

`Scheduler.finish_requests()` 会把 request 标记成 finished，然后走 `_free_request()`。

`_free_request()` 会：

1. 处理 KV connector / EC connector 的收尾
2. 释放 encoder cache
3. 如果没有延迟释放需求，直接 `_free_blocks(request)`

`_free_blocks()` 里最终调用 `KVCacheManager.free(request)`。

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2093`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2156`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:492`

### 8.2 preempt / async overlap 时延迟释放

如果开启了 `defer_block_free`，scheduler 不会立刻把 block 还回 pool，而是：

- `pop_blocks_for_free()` 先把 request->blocks 的 bookkeeping 拿掉
- 这些 block 进入 `deferred_frees`
- 等 `processed_step_seq` 追上 fence 后，再由 `_drain_deferred_frees()` 真正释放

这是为了避免“前一个 step 还在写 block，后一个 step 已经重新分配同一块”这种竞态。

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2197`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/scheduler.py:2223`

### 8.3 block pool 里真正怎么回收

`BlockPool.free_blocks()` 会：

- 对每个 block `ref_cnt -= 1`
- `ref_cnt == 0` 的块重新放回 free queue
- 有 hash 的块放在后面，没 hash 的块更优先回收

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:719`

## 9. prefix cache 命中时的补充

如果 request 命中 prefix cache：

- coordinator 会找到已经存在的 `KVCacheBlock`
- `touch()` 会把命中的块引用计数加一
- `cache_blocks()` 会把新计算满的 block 记入 hash table

所以 page attention 的“block”不是每次都新建的；很多时候它是复用已有物理 block。

参考：
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/kv_cache_manager.py:207`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:702`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/block_pool.py:225`

## 10. 一句话版流程

可以把整条链压缩成一句话：

> Scheduler 决定这一步要算哪些 token，然后通过 KVCacheManager/Coordinator/BlockPool 拿到物理 block；worker 把这些 block id 写入 block_table，再生成 slot_mapping；attention kernel 用 block_table 读历史 KV、用 slot_mapping 写当前 KV；请求结束时由 scheduler 统一把 block 还回 block pool，必要时延迟释放。

