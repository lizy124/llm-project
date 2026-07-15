# 02. BlockPool 和 KV block 生命周期如何工作？

源码位置：

- `vllm/vllm/vllm/v1/core/block_pool.py`
- `vllm/vllm/vllm/v1/core/kv_cache_manager.py`
- `vllm/vllm/vllm/v1/core/kv_cache_coordinator.py`
- `vllm/vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/vllm/vllm/v1/core/kv_cache_utils.py`
- `vllm/vllm/vllm/v1/core/sched/scheduler.py`

本问题关注：`BlockPool` 到底管理什么；KV block 如何从空闲池进入请求；prefix cache 命中时 block 如何复用；full block 什么时候写入 prefix cache；请求结束、抢占、KV transfer、deferred free 会如何影响 block 释放；以及 Scheduler、KVCacheManager、coordinator、BlockPool、Worker 之间的职责边界。

---

## 1. 一句话回答

`BlockPool` 是 **Scheduler 侧的 KV block 元数据资源池**。

它管理的是：

```text
KVCacheBlock 元数据：
  block_id
  ref_cnt
  _block_hash / block_hash
  prev_free_block / next_free_block
  is_null
```

它不直接管理的是：

```text
GPU 上真实的 KV cache tensor 内容。
```

真实 KV cache tensor 在 Worker / ModelRunner 侧分配和读写；Scheduler 侧只通过 `block_id` 建账，Worker 侧再把这些 `block_id` 转成 block table / slot mapping 供 attention backend 使用。

最小心智模型是：

```text
BlockPool 管“哪些 block id 被谁占用、能否复用、能否驱逐”；
Worker 管“这些 block id 对应的 GPU KV cache 槽位里写了什么”。
```

---

## 2. 总体链路

KV block 生命周期可以压缩成：

```text
Engine 初始化
  → 创建 KVCacheManager
  → 创建 KVCacheCoordinator
  → 创建 BlockPool(num_gpu_blocks)
  → 初始化 KVCacheBlock[0..N-1]
  → 取出 block 0 作为 null_block

请求进入 Scheduler
  → get_computed_blocks() 查 prefix cache
  → allocate_slots() 判断 block 是否够
  → touch prefix-hit blocks
  → get_new_blocks() 分配缺失 blocks
  → cache_blocks() 把 full blocks 放入 prefix cache
  → SchedulerOutput 携带 block_ids
  → Worker / ModelRunner 使用 block table + slot mapping

请求结束 / 抢占 / 滑窗淘汰
  → free(request) 或 pop_blocks_for_free()
  → BlockPool.free_blocks()
  → ref_cnt--
  → ref_cnt 为 0 的 block 回到 free_block_queue
  → 下次 get_new_blocks() 可重新分配或驱逐其 prefix cache hash
```

这里要特别注意：

```text
进入 free_block_queue 不一定等于 block_hash 立刻被删除。
```

开启 prefix caching 时，一个 ref_cnt=0 的 cached block 可以留在 free queue 里作为“可驱逐缓存”。如果后续请求命中它，会被 `touch()` 从 free queue 移走并 ref_cnt++；如果后续要分配新 block，则 `get_new_blocks()` 会从 free queue 弹出它并 `_maybe_evict_cached_block()`，这时才真正从 prefix cache hash map 删除。

---

## 3. 关键对象关系

### 3.1 KVCacheBlock

定义在：`vllm/vllm/v1/core/kv_cache_utils.py:118`

一个 `KVCacheBlock` 是 Scheduler 侧的 block 元数据：

```text
block_id        物理 KV cache block 编号
ref_cnt         当前被多少请求引用
_block_hash / block_hash  full block 被 prefix cache 记录后的 hash key
prev_free_block / next_free_block  free_block_queue 双向链表指针
is_null         是否是 null block
```

它不是一块 GPU tensor，也不保存 K/V 数据本身。

### 3.2 BlockPool

定义在：`vllm/vllm/v1/core/block_pool.py:143`

`BlockPool` 保存：

```text
blocks                         所有 KVCacheBlock 元数据
free_block_queue               可分配 / 可驱逐 block 队列
cached_block_hash_to_block      prefix cache hash → block 映射
null_block                     占位 block
enable_kv_cache_events          是否产出 KV cache events
metrics_collector               block residency / eviction 统计
```

它提供的核心操作是：

```text
get_cached_block()
cache_full_blocks()
get_new_blocks()
touch()
free_blocks()
evict_blocks()
reset_prefix_cache()
```

### 3.3 KVCacheManager

定义在：`vllm/vllm/v1/core/kv_cache_manager.py:114`

`KVCacheManager` 是 Scheduler 调用的门面。Scheduler 通常不直接操作 `BlockPool`，而是通过它调用：

```text
get_computed_blocks()
allocate_slots()
free()
pop_blocks_for_free()
cache_blocks()
get_blocks()
get_block_ids()
take_new_block_ids()
```

它的重点职责是：

```text
把请求级调度语义转成 block 分配 / 缓存 / 释放操作。
```

### 3.4 KVCacheCoordinator

定义在：`vllm/vllm/v1/core/kv_cache_coordinator.py:60`

coordinator 负责多 KV cache group 的编排：

```text
- 单一 KV group：UnitaryKVCacheCoordinator
- 多类型 / hybrid KV group：HybridKVCacheCoordinator
- prefix caching 关闭：KVCacheCoordinatorNoPrefixCache
```

它内部持有多个 `SingleTypeKVCacheManager`，每个 manager 负责一种 KV cache spec / group 的请求账本。

### 3.5 SingleTypeKVCacheManager

定义在：`vllm/vllm/v1/core/single_type_kv_cache_manager.py:32`

它负责每个 group 内的请求到 blocks 映射：

```text
req_to_blocks[request_id]      请求当前持有的 KVCacheBlock 列表
num_cached_block[request_id]   该请求已有多少 full block 进入 prefix cache
new_block_ids                  本轮新分配、需要 Worker zero 的 block ids
```

它也处理不同 attention 类型的差异：

```text
FullAttentionManager
SlidingWindowManager
ChunkedLocalAttentionManager
MambaManager
CrossAttentionManager
```

---

## 4. BlockPool 初始化发生了什么

入口在 `BlockPool.__init__()`：`block_pool.py:149`

初始化过程是：

```text
1. 创建 num_gpu_blocks 个 KVCacheBlock；
2. 用所有 block 初始化 FreeKVCacheBlockQueue；
3. 创建 cached_block_hash_to_block；
4. 从 free queue 弹出第一个 block 作为 null_block；
5. 标记 null_block.is_null = True；
6. 初始化 event queue / metrics collector。
```

核心代码对应：

```text
self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
self.null_block = self.free_block_queue.popleft()
self.null_block.is_null = True
```

位置：`block_pool.py:161` 到 `block_pool.py:177`

这意味着：

```text
block_id=0 通常会成为 null_block；
它从 free queue 中移除；
它不参与正常分配；
usage 统计要减掉这个 null block。
```

`BlockPool.get_usage()` 里也明确减掉 1：`block_pool.py:504` 到 `block_pool.py:515`

---

## 5. null_block 是什么

`null_block` 是一个占位 block，用来表示：

```text
这个位置在 block table 结构上存在，
但当前 attention 不需要真实 KV block。
```

常见场景：

```text
- sliding window 左侧已经滑出窗口的 blocks；
- chunked local attention 当前 chunk 之外的 blocks；
- Mamba align 模式下被跳过或已经不需要的 state block；
- prefix cache hit 返回时，为了保持 block index 对齐而填充的空洞。
```

它有几个特殊点：

```text
1. is_null=True；
2. ref_cnt 不按普通 block 维护；
3. 不应该被 free；
4. 不应该进入 prefix cache hash map；
5. Worker 侧看到它，本质上是“这个 block table 位置不用真实 block”。
```

所以文档里看到 `null_block` 时，不要理解成“真的分配了一个可用 KV block”，它只是一个结构占位符。

---

## 6. free_block_queue 是什么

`FreeKVCacheBlockQueue` 定义在：`kv_cache_utils.py:165`

它是一个双向链表，不是 Python `deque`。原因是：

```text
prefix cache 命中时，如果 block.ref_cnt == 0，
这个 block 正在 free queue 中等待被驱逐，
此时 touch() 需要 O(1) 从队列中间移除它。
```

队列支持：

```text
popleft() / popleft_n()    从头部拿可分配 block
remove(block)              O(1) 移除某个命中的 cached block
append() / append_n()      放到尾部
prepend_n()                放到头部
```

队列语义是：

```text
队头：更早被驱逐 / 更优先复用为新 block
队尾：更晚被驱逐 / 更适合保留为 prefix cache
```

释放 block 时，`BlockPool.free_blocks()` 会区分：

```text
无 hash block：prepend 到队头，优先被重新分配；
有 hash block：append 到队尾，尽量保留 prefix cache。
```

位置：`block_pool.py:419` 到 `block_pool.py:440`

这就是为什么 prefix cache 可以在请求结束后继续保留：

```text
ref_cnt=0 + block_hash 不为空 + 位于 free queue 尾部
  → 它已经没人持有，但仍可作为 prefix cache 命中来源。
```

---

## 7. prefix cache hash map 如何工作

`BlockHashToBlockMap` 定义在：`block_pool.py:34`

它维护：

```text
BlockHashWithGroupId → KVCacheBlock 或 {block_id → KVCacheBlock}
```

为什么 value 可能是 dict？

```text
因为 vLLM 当前不会对相同内容的 cached block 做去重。
如果两个不同 block_id 的内容 hash 相同，它们都可能挂在同一个 hash key 下。
```

源码注释解释了原因：

```text
为了保证已分配 block ids 不变，block table 可以 append-only，
不会因为发现重复内容就把请求的 block id 替换掉。
```

位置：`block_pool.py:48` 到 `block_pool.py:55`

### 7.1 hash key 为什么包含 group id

`make_block_hash_with_group_id()` 定义在：`kv_cache_utils.py:56`

它把：

```text
block_hash + kv_cache_group_id
```

打包成 `BlockHashWithGroupId`。

原因是 hybrid KV cache 下，不同 KV group 可能有不同 block table / attention 类型。即使 token 内容相同，也必须区分它属于哪个 group。

### 7.2 get_cached_block()

`BlockPool.get_cached_block()`：`block_pool.py:184`

它输入：

```text
block_hash
kv_cache_group_ids
```

输出：

```text
每个 group 对应的 cached block 列表；
只要任意 group miss，就返回 None。
```

所以一次 prefix cache hit 必须满足：

```text
这个 token block 的 hash 在需要的所有 KV cache group 中都能找到。
```

---

## 8. 请求什么时候获得 block

请求获得 block 的主入口是：

```python
KVCacheManager.allocate_slots(...)
```

位置：`kv_cache_manager.py:244`

Scheduler 在两类路径调用它。

### 8.1 RUNNING 请求继续执行

在 `Scheduler.schedule()` 的 RUNNING 阶段：`scheduler.py:429` 起

关键流程：

```text
1. 计算 num_new_tokens；
2. 调用 kv_cache_manager.allocate_slots(request, num_new_tokens, lookahead)；
3. 如果返回 None，说明 block 不够；
4. Scheduler 选择一个 running request 抢占；
5. 被抢占请求释放 blocks；
6. 重试 allocate_slots()。
```

对应位置：`scheduler.py:521` 到 `scheduler.py:571`

这说明 RUNNING 请求不是一次性拿满所有 blocks，而是随着 `num_computed_tokens` 推进逐步补 block。

### 8.2 WAITING 请求首次进入运行

在 WAITING 阶段，Scheduler 先查本地 prefix cache 和外部 KV：

```text
get_computed_blocks(request)
connector.get_num_new_matched_tokens(...)
```

然后调用：

```text
allocate_slots(
  request,
  num_new_tokens,
  num_new_computed_tokens,
  new_computed_blocks,
  num_external_computed_tokens,
  delay_cache_blocks,
  ...
)
```

位置：`scheduler.py:671` 到 `scheduler.py:885`

这条路径会同时处理：

```text
- 本地 prefix cache hit；
- 远端 KV hit；
- async KV load；
- chunked prefill；
- encoder cache；
- speculative lookahead blocks。
```

---

## 9. allocate_slots() 分成哪几步

`KVCacheManager.allocate_slots()` 的源码注释已经给出了三阶段模型：`kv_cache_manager.py:328` 到 `kv_cache_manager.py:335`

可以理解为：

```text
1. 释放当前 attention 不再需要的 skipped blocks，检查容量；
2. 处理 prefix computed tokens：
   - 本地 prefix cache hit blocks；
   - 外部 KV computed tokens；
   - sliding window / chunked local 的 null padding；
3. 给本轮真正要计算的新 token 和 lookahead token 分配 blocks。
```

### 9.1 容量检查

它先计算：

```text
num_tokens_need_slot = total_computed_tokens + num_new_tokens + num_lookahead_tokens
num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(...)
available_blocks = block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
```

如果：

```text
required_blocks > available_blocks
```

就返回 `None`，Scheduler 会决定是否抢占其他请求。

位置：`kv_cache_manager.py:404` 到 `kv_cache_manager.py:420`

### 9.2 先 remove_skipped_blocks()

在真正分配前，会调用：

```python
self.coordinator.remove_skipped_blocks(request.request_id, total_computed_tokens)
```

位置：`kv_cache_manager.py:394` 到 `kv_cache_manager.py:402`

原因是：

```text
对于 sliding window / chunked local / Mamba，
有些旧 block 已经不再参与 attention，
可以先释放，减少后续新 block 分配时的驱逐压力。
```

Full attention 默认不会 skip token，因此通常不会在这里释放。

### 9.3 处理 prefix cache hit blocks

如果存在 `new_computed_blocks` 或外部 computed tokens，会调用：

```python
coordinator.allocate_new_computed_blocks(...)
```

位置：`kv_cache_manager.py:422` 到 `kv_cache_manager.py:433`

本地 prefix cache hit 的 block 会走：

```text
SingleTypeKVCacheManager.add_local_computed_blocks()
  → block_pool.touch(new_computed_blocks)
  → req_to_blocks[request_id].extend(...)
  → num_cached_block[request_id] = len(req_blocks)
```

位置：`single_type_kv_cache_manager.py:182` 到 `single_type_kv_cache_manager.py:233`

`touch()` 很关键：

```text
如果 cached block ref_cnt == 0，它在 free queue 中；
touch() 会把它从 free queue 移除；
然后 ref_cnt++，表示新请求引用了它。
```

位置：`block_pool.py:402` 到 `block_pool.py:417`

### 9.4 处理外部 KV computed tokens

外部 KV 命中时，本地还没有真实 GPU block 内容，需要先为这些 token 分配本地 blocks，等 Worker 侧 connector load 写入。

对应：

```text
SingleTypeKVCacheManager.allocate_external_computed_blocks()
  → block_pool.get_new_blocks(...)
  → req_to_blocks.extend(allocated_blocks)
  → new_block_ids 记录需要 zero 的 attention blocks
```

位置：`single_type_kv_cache_manager.py:234` 到 `single_type_kv_cache_manager.py:276`

如果是 async KV load，`delay_cache_blocks=True`，`allocate_slots()` 会先返回，不立刻把这些 blocks 放入 prefix cache。

位置：`kv_cache_manager.py:442` 到 `kv_cache_manager.py:445`

### 9.5 分配本轮新 token blocks

真正新 token / lookahead token 需要的 block 通过：

```python
coordinator.allocate_new_blocks(...)
```

位置：`kv_cache_manager.py:435` 到 `kv_cache_manager.py:440`

对于普通 manager，最后会到：

```text
SingleTypeKVCacheManager.allocate_new_blocks()
  → 计算 num_required_blocks
  → block_pool.get_new_blocks(num_new_blocks)
  → req_to_blocks[request_id].extend(new_blocks)
```

位置：`single_type_kv_cache_manager.py:277` 到 `single_type_kv_cache_manager.py:308`

---

## 10. get_new_blocks() 做了什么

`BlockPool.get_new_blocks()`：`block_pool.py:333`

它的语义是：

```text
从 free_block_queue 头部弹出 n 个 block，
必要时驱逐它们原来的 prefix cache hash，
然后 ref_cnt 从 0 变成 1。
```

核心步骤：

```text
1. 检查 free block 数量是否足够；
2. free_block_queue.popleft_n(num_blocks)；
3. 如果开启 prefix caching，对弹出的每个 block 调用 _maybe_evict_cached_block()；
4. assert block.ref_cnt == 0；
5. block.ref_cnt += 1；
6. 记录 metrics。
```

位置：`block_pool.py:344` 到 `block_pool.py:363`

### 10.1 为什么分配时才 evict hash

因为 ref_cnt=0 的 cached block 仍可能被 prefix cache 命中。

只有当它真的被拿来给新 token 写入时，旧内容才会被覆盖，因此必须：

```text
删除 cached_block_hash_to_block 中的映射；
block.reset_hash()；
必要时发 BlockRemoved event。
```

这就是 `_maybe_evict_cached_block()` 的职责。

位置：`block_pool.py:365` 到 `block_pool.py:400`

### 10.2 block_id 为什么稳定

`get_new_blocks()` 返回的 block 会追加到请求的 `req_to_blocks`。vLLM 不会因为发现相同 hash 的 block 就替换请求已有 block_id。

这是为了让 Worker 侧 block table 保持稳定，避免已经下发的 block id 被重写。

---

## 11. prefix cache 命中如何复用 block

prefix cache 查询入口：

```python
KVCacheManager.get_computed_blocks(request)
```

位置：`kv_cache_manager.py:202`

它会：

```text
1. 如果 prefix caching 关闭或请求跳过读取，直接返回空；
2. 设置 max_cache_hit_length = request.num_tokens - 1；
3. 调 coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)；
4. 返回命中的 blocks 和命中的 token 数。
```

为什么最多命中 `request.num_tokens - 1`？

```text
即使 prompt 全部命中，也要重算最后一个 token 来获得 logits。
```

位置：`kv_cache_manager.py:221` 到 `kv_cache_manager.py:227`

### 11.1 Full attention 的命中

`FullAttentionManager.find_longest_cache_hit()` 从左到右查：

```text
block_hash[0]
block_hash[1]
block_hash[2]
...
```

只要某个 block miss，后续必然不能作为连续 prefix 命中，因此停止。

位置：`single_type_kv_cache_manager.py:538` 到 `single_type_kv_cache_manager.py:586`

### 11.2 Sliding window 的命中

`SlidingWindowManager.find_longest_cache_hit()` 从右到左查，允许前面的块用 `null_block` 占位。

原因是 sliding window 只需要窗口内连续 block 命中，窗口左侧的历史 token 已经不参与 attention。

位置：`single_type_kv_cache_manager.py:599` 到 `single_type_kv_cache_manager.py:702`

### 11.3 Hybrid KV cache 的命中

`HybridKVCacheCoordinator.find_longest_cache_hit()` 会让不同 attention group 共同收敛到一个可用 hit length：

```text
Full attention 可能给出一个长度；
Sliding window / Mamba 可能只能接受更短长度；
如果某个 group 缩短 hit length，就重新校验；
直到所有 group 对同一个 hit length 达成一致。
```

位置：`kv_cache_coordinator.py:621` 到 `kv_cache_coordinator.py:731`

所以 hybrid 场景下，prefix cache hit 不是简单看一个 hash 链，而是多个 KV group 的共同结果。

---

## 12. block 什么时候进入 prefix cache

block 进入 prefix cache 的入口是：

```python
KVCacheManager.cache_blocks(request, num_computed_tokens)
```

或者在 `allocate_slots()` 内部自动调用：

```python
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

位置：`kv_cache_manager.py:447` 到 `kv_cache_manager.py:456`

### 12.1 只 cache full blocks

`SingleTypeKVCacheManager.cache_blocks()` 计算：

```text
num_full_blocks = num_tokens // block_size
```

只有完整 block 会进入 prefix cache。

位置：`single_type_kv_cache_manager.py:334` 到 `single_type_kv_cache_manager.py:359`

### 12.2 只 cache 已 finalized 的 token

`allocate_slots()` 里会限制：

```text
num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)
```

位置：`kv_cache_manager.py:447` 到 `kv_cache_manager.py:455`

这是为了排除还可能被拒绝的 draft tokens。spec decode 的 lookahead / unverified draft token 不应该提前进入 prefix cache。

### 12.3 cache_full_blocks() 具体做什么

`BlockPool.cache_full_blocks()`：`block_pool.py:211`

它会：

```text
1. 找出新变成 full 且尚未 cached 的 blocks；
2. 根据 request.block_hashes 取对应 block hash；
3. 加上 kv_cache_group_id 构成 BlockHashWithGroupId；
4. 写入 blk.block_hash；
5. 插入 cached_block_hash_to_block；
6. 可选地产生 BlockStored event。
```

位置：`block_pool.py:245` 到 `block_pool.py:331`

注意：

```text
缓存的是 block 元数据和 hash 映射；
真实 KV tensor 内容已经由 Worker 在 forward 中写入。
```

---

## 13. block 什么时候释放

释放有三类入口。

### 13.1 请求正常结束

`Scheduler.update_from_output()` 中如果请求 stopped，会调用：

```text
_handle_stopped_request()
_free_request()
```

位置：`scheduler.py:1655` 到 `scheduler.py:1663`（调用点），`_handle_stopped_request()` 定义在 `scheduler.py:1830`

`_free_request()` 做：

```text
1. connector request_finished；
2. free encoder cache；
3. 记录 finished_req_ids；
4. 如果不需要 delay_free_blocks，则 _free_blocks(request)。
```

位置：`scheduler.py:2046` 到 `scheduler.py:2063`

最终：

```text
_free_blocks()
  → _free_request_blocks()
  → kv_cache_manager.free(request)
  → coordinator.free(request_id)
  → SingleTypeKVCacheManager.free(request_id)
  → block_pool.free_blocks(reversed(req_blocks))
```

对应位置：

```text
scheduler.py:2065
scheduler.py:2077
kv_cache_manager.py:460
kv_cache_coordinator.py:284
single_type_kv_cache_manager.py:399
block_pool.py:419
```

### 13.2 请求被抢占

当 `allocate_slots()` 返回 `None`，Scheduler 会抢占一个 running request：

```python
self._preempt_request(preempted_req, scheduled_timestamp)
```

位置：`scheduler.py:534` 到 `scheduler.py:564`

`_preempt_request()` 会：

```text
1. 释放 request 的 KV blocks；
2. 释放 encoder cache；
3. 从 inflight prefill 集合移除；
4. request.status = PREEMPTED；
5. request.num_computed_tokens = 0；
6. 清空 spec_token_ids；
7. num_preemptions++；
8. 放回 waiting 队列头部。
```

位置：`scheduler.py:1105` 到 `scheduler.py:1127`

重点是：

```text
抢占会释放请求当前持有的 block 引用，
但 full cached blocks 的 hash 可以保留，
因此请求恢复时可能通过 prefix cache 重新命中。
```

### 13.3 sliding window / local attention 中途释放

在 `allocate_slots()` 开始时，会调用 `remove_skipped_blocks()`。

对于 sliding window：

```text
旧 token 已经滑出 attention window；
对应 block 可以释放；
req_to_blocks 中的位置替换成 null_block。
```

位置：`single_type_kv_cache_manager.py:477` 到 `single_type_kv_cache_manager.py:518`

这样一个长请求不一定一直持有从 token 0 开始的所有 physical blocks。

---

## 14. free_blocks() 的精确语义

`BlockPool.free_blocks()`：`block_pool.py:419`

它不是“无条件清空 block”，而是：

```text
1. 对每个 block ref_cnt--；
2. 如果 ref_cnt 仍 > 0，说明还有其他请求复用它，不进 free queue；
3. 如果 ref_cnt == 0 且不是 null block：
   - block_hash is None：加入 blocks_without_hash；
   - block_hash 不为空：加入 blocks_with_hash；
4. 无 hash blocks prepend 到 free queue 头；
5. 有 hash blocks append 到 free queue 尾。
```

这产生三个重要结论：

```text
1. free 只是减少引用，不一定释放到底；
2. prefix cache 复用的 block 会被多个请求共享，只有所有请求都释放后 ref_cnt 才归零；
3. ref_cnt=0 的 cached block 仍然可以被 prefix cache 命中，直到它被重新分配时才 evict。
```

---

## 15. ref_cnt 如何变化

### 15.1 新分配

```text
get_new_blocks()
  → block.ref_cnt 从 0 变成 1
```

位置：`block_pool.py:333` 到 `block_pool.py:363`

### 15.2 prefix cache 命中

```text
add_local_computed_blocks()
  → block_pool.touch(new_computed_blocks)
  → 如果 ref_cnt == 0，从 free queue 移除
  → ref_cnt++
```

位置：`single_type_kv_cache_manager.py:182` 到 `single_type_kv_cache_manager.py:233`，`block_pool.py:402` 到 `block_pool.py:417`

### 15.3 请求释放

```text
free_blocks()
  → block.ref_cnt--
  → ref_cnt == 0 时回到 free queue
```

位置：`block_pool.py:419` 到 `block_pool.py:440`

### 15.4 reset hash 不影响 ref_cnt

`_maybe_evict_cached_block()` 只负责从 prefix cache hash map 中移除映射并 `reset_hash()`，它不改变引用计数。

位置：`block_pool.py:365` 到 `block_pool.py:400`

---

## 16. block_hash 如何变化

一个 block 的 `block_hash` 生命周期通常是：

```text
新建 / 新分配：block_hash = None
  → block 被写满并 cache_full_blocks()
  → block_hash = BlockHashWithGroupId(...)
  → 请求释放后 ref_cnt 可能变 0，但 block_hash 保留
  → 如果被 prefix cache 命中，touch() 后继续复用
  → 如果被 get_new_blocks() 重新分配，_maybe_evict_cached_block()
  → block_hash reset 为 None
```

所以：

```text
block_hash 存在表示“这个 block 的旧内容可被 prefix cache 查找”；
ref_cnt 表示“这个 block 当前是否被请求持有”；
两者是两个维度。
```

典型状态组合：

```text
ref_cnt > 0, block_hash is None
  新分配、正在写、尚未 full/cache。

ref_cnt > 0, block_hash exists
  已 full/cache，且当前仍被一个或多个请求使用。

ref_cnt == 0, block_hash exists
  没有请求持有，但仍作为 prefix cache 驱逐候选保留。

ref_cnt == 0, block_hash is None
  普通空闲 block，优先被重新分配。
```

---

## 17. SchedulerOutput 里 block 如何传给 Worker

Scheduler 在构造输出时，会把 block ids 放进：

```text
scheduled_new_reqs
scheduled_cached_reqs
```

新请求：

```python
NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
```

位置：`scheduler.py:1015` 到 `scheduler.py:1029`

已在 batch 中的请求：

```python
CachedRequestData(..., new_block_ids=...)
```

位置：`scheduler.py:1219` 到 `scheduler.py:1277`

也就是说，Worker 不接收 `KVCacheBlock` 对象，而是接收：

```text
tuple[list[int], ...]
```

其中：

```text
外层 tuple：KV cache group
内层 list：该 group 的 block ids
```

这也是 `KVCacheBlocks.get_block_ids()` 的作用。

位置：`kv_cache_manager.py:57` 到 `kv_cache_manager.py:84`

---

## 18. new_block_ids_to_zero 是什么

有些 KV cache block 新分配后需要 Worker 侧清零，Scheduler 会把这些 block id 放入：

```text
SchedulerOutput.new_block_ids_to_zero
```

Scheduler 构造它的位置：

```python
new_block_ids_to_zero = (
    (self.kv_cache_manager.take_new_block_ids() or None)
    if self.needs_kv_cache_zeroing
    else None
)
```

位置：`scheduler.py:1044` 到 `scheduler.py:1048`

`KVCacheManager.take_new_block_ids()` 会收集各 single-type manager 的 `new_block_ids`。

位置：`kv_cache_manager.py:605` 到 `kv_cache_manager.py:610`

普通 attention manager 在新分配 FullAttention / TQFullAttention / MLA blocks 时记录这些 ids：

```text
single_type_kv_cache_manager.py:302 到 single_type_kv_cache_manager.py:307
```

也就是说，`new_block_ids_to_zero` 是面向需要 zero 的 attention KV blocks 的列表，不是所有 KV cache spec 分配都会进入这个列表。

Worker 侧再根据 `scheduler_output.new_block_ids_to_zero` 真正清 GPU KV cache。原因是：

```text
Scheduler 只知道 block id；
清零是设备侧内存操作，必须由 Worker / ModelRunner 执行。
```

---

## 19. KV transfer 对生命周期的影响

KV connector 会让 block 生命周期多出几种特殊状态。

### 19.1 外部 KV 命中但本地 block 还没内容

当 connector 发现远端 KV 命中时：

```text
num_external_computed_tokens > 0
```

Scheduler 会为这些 token 分配本地 blocks，但真实内容由 Worker 侧 connector load 进来。

如果是 async load：

```text
request.status = WAITING_FOR_REMOTE_KVS
request.num_computed_tokens = num_computed_tokens
```

位置：`scheduler.py:916` 到 `scheduler.py:937`

此时 block 已经分配，但请求还不进入 RUNNING。

### 19.2 delay_cache_blocks

async load 时 `allocate_slots(..., delay_cache_blocks=True)`。

这意味着：

```text
这些 blocks 先绑定到请求，
但不会立即 cache_full_blocks()，
因为 Worker 还没把远端 KV 真正加载完成。
```

位置：`kv_cache_manager.py:442` 到 `kv_cache_manager.py:445`

等 Worker 报告 `finished_recving` 后，Scheduler 调用：

```text
_update_waiting_for_remote_kv()
  → kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
```

位置：`scheduler.py:2350` 到 `scheduler.py:2383`

### 19.3 finished_sending 后再释放

P/D 或 KV save 场景下，请求结束并不一定马上释放 blocks，因为 connector 可能还要把 KV 发出去。

`_connector_finished()` 会返回：

```text
connector_delay_free_blocks
kv_xfer_params
```

位置：`scheduler.py:2299` 到 `scheduler.py:2328`

如果需要延迟释放，`_free_request()` 不会立即 `_free_blocks()`。

等 Worker connector 输出：

```text
finished_sending
```

Scheduler 再调用：

```text
_update_from_kv_xfer_finished()
  → _free_blocks(request)
```

位置：`scheduler.py:2417` 到 `scheduler.py:2445`

---

## 20. deferred free 是什么

`defer_block_free` 是另一种延迟释放机制，主要用于：

```text
KV consumer + max_concurrent_batches > 1 的 overlapping batches 场景。
```

`async scheduling`、pipeline parallel 等能力可能造成多批次重叠，但当前源码里的启用条件是 `multiple_inflight_batches = max_concurrent_batches > 1` 且 `kv_transfer_config.is_kv_consumer`。

Scheduler 初始化时，如果：

```text
max_concurrent_batches > 1
且 kv_transfer_config.is_kv_consumer
```

则：

```python
self.defer_block_free = True
```

位置：`scheduler.py:145` 到 `scheduler.py:151`

### 20.1 为什么需要 deferred free

注释说明了核心原因：

```text
一个 step 可能还在写某个请求的 KV blocks；
如果 Scheduler 提前释放并重新分配这些 blocks，
consumer KV connector 可能把远端 KV load 到同一批 blocks，
与尚未完成的写发生乱序。
```

所以不能只看 Scheduler 逻辑上请求是否结束，还要等对应 in-flight GPU step 的输出被处理。

### 20.2 deferred free 如何实现

Scheduler 维护：

```text
sched_step_seq
processed_step_seq
deferred_frees: deque[(fence_seq, blocks)]
```

位置：`scheduler.py:292` 到 `scheduler.py:299`

调度非空 step 时：

```text
sched_step_seq += 1
```

位置：`scheduler.py:1091` 到 `scheduler.py:1094`

`_update_after_schedule()` 会把当前 step 序号记录到 request：

```text
request.last_sched_seq = self.sched_step_seq
```

位置：`scheduler.py:1142` 到 `scheduler.py:1144`

释放请求时，如果 last scheduled step 还没 processed：

```text
pop_blocks_for_free(request)
  → 只从 req_to_blocks 删除账本
  → 不立刻 block_pool.free_blocks()
  → blocks 放入 deferred_frees
```

位置：`scheduler.py:2077` 到 `scheduler.py:2091`

等 `update_from_output()` 处理完对应 step：

```text
processed_step_seq += 1
_drain_deferred_frees()
```

位置：`scheduler.py:1477` 到 `scheduler.py:1482`

真正归还 block：

```text
kv_cache_manager.block_pool.free_blocks(reversed(blocks))
```

位置：`scheduler.py:2092` 到 `scheduler.py:2105`

---

## 21. 抢占、释放、deferred free 的区别

这三个概念容易混。

### 21.1 抢占 preemption

```text
目的：腾出 KV blocks 给更优先 / 更早的请求。
发生点：schedule() 中 allocate_slots() 失败时。
结果：请求变 PREEMPTED，num_computed_tokens 归 0，回 waiting 队列。
```

### 21.2 普通释放 free

```text
目的：请求完成或中止后释放它持有的 block 引用。
发生点：update_from_output() / finish_requests()。
结果：ref_cnt--，ref_cnt=0 的 block 回 free queue。
```

### 21.3 deferred free

```text
目的：避免仍在进行的 GPU 写和新分配 / KV load 乱序。
发生点：请求逻辑上结束，但 last_sched_seq 尚未 processed。
结果：先从请求账本移除，等 fence 通过后再 block_pool.free_blocks()。
```

---

## 22. prefix cache reset 做了什么

入口：

```python
Scheduler.reset_prefix_cache(...)
```

位置：`scheduler.py:2143`

最终调用：

```python
BlockPool.reset_prefix_cache()
```

位置：`block_pool.py:461`

它只在除了 null block 之外没有 block 被使用时才能成功：

```python
num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
if num_used_blocks != 1:
    return False
```

位置：`block_pool.py:470` 到 `block_pool.py:477`

成功后会：

```text
1. 清空 cached_block_hash_to_block；
2. 对所有 blocks reset_hash()；
3. reset metrics；
4. 可选地产生 AllBlocksCleared event。
```

位置：`block_pool.py:479` 到 `block_pool.py:494`

如果调用方传入 `reset_running_requests=True`，Scheduler 会先把所有 running 请求抢占回 waiting，再 reset prefix cache。

位置：`scheduler.py:2153` 到 `scheduler.py:2179`

---

## 23. KV cache events 和 metrics

BlockPool 可以产生几类事件：

```text
BlockStored      full block 进入 prefix cache
BlockRemoved     cached block 被 evict
AllBlocksCleared prefix cache reset
```

相关位置：

```text
BlockStored：block_pool.py:315 到 block_pool.py:331
BlockRemoved：block_pool.py:392 到 block_pool.py:399
AllBlocksCleared：block_pool.py:491 到 block_pool.py:492
```

`KVCacheManager.take_events()` 会给 `BlockStored` 补充 KV cache spec kind / sliding window 等语义信息。

位置：`kv_cache_manager.py:554` 到 `kv_cache_manager.py:578`

Scheduler 在 `update_from_output()` 中收集并发布事件：

```text
kv_cache_manager.take_events()
connector.take_events()
kv_event_publisher.publish(KVEventBatch(...))
```

位置：`scheduler.py:1752` 到 `scheduler.py:1767`

metrics 方面，BlockPool 在这些动作上通知 collector：

```text
on_block_allocated
on_block_accessed
on_block_evicted
reset
```

这些只影响观测，不改变 block 生命周期本身。

---

## 24. 与 Worker 侧物理 KV cache 的边界

### 24.1 Scheduler / BlockPool 负责

```text
- 判断请求是否有足够 block 可运行；
- 分配 block ids；
- 维护 request_id → KVCacheBlock 列表；
- 维护 prefix cache hash → block；
- 维护 ref_cnt；
- 处理抢占、释放、deferred free；
- 把 block ids 放进 SchedulerOutput。
```

### 24.2 Worker / ModelRunner 负责

```text
- 初始化真实 GPU KV cache tensor；
- 接收 SchedulerOutput 中的 block ids；
- 更新 InputBatch.block_table；
- 计算 slot mapping；
- 在 forward / attention backend 中读写 KV；
- 根据 new_block_ids_to_zero 清零新 block；
- 执行 KV connector load/save/finalize。
```

### 24.3 一句话边界

```text
Scheduler 管 KV block 账本；Worker 管 KV block 内容。
```

---

## 25. 一个请求的完整生命周期示例

### 25.1 首次 prefill，无 prefix cache 命中

```text
WAITING request
  → get_computed_blocks() 返回 0 hit
  → allocate_slots(num_new_tokens)
  → get_new_blocks(k)
  → ref_cnt=1, block_hash=None
  → req_to_blocks[req_id] = [blocks...]
  → SchedulerOutput.scheduled_new_reqs 携带 block_ids
  → Worker forward 写入 KV tensor
  → cache_blocks() 把 full blocks 写入 prefix cache
  → block_hash 设置完成
```

### 25.2 另一个请求命中 prefix cache

```text
new WAITING request with same prefix
  → get_computed_blocks() 查到 cached blocks
  → add_local_computed_blocks()
  → block_pool.touch(hit_blocks)
  → ref_cnt++
  → req_to_blocks[req_id] 引用同一批 KVCacheBlock
  → 只为未命中的 suffix 分配新 blocks
```

### 25.3 第一个请求结束

```text
request finished
  → free_blocks(reversed(blocks))
  → shared cached blocks ref_cnt--
  → 如果第二个请求还在用，ref_cnt 仍 > 0，不进 free queue
  → 只有完全没人引用时才进入 free queue
```

### 25.4 所有引用结束后

```text
ref_cnt 变 0
  → 如果 block_hash exists，append 到 free queue 尾部
  → 它仍可被 prefix cache 命中
  → 如果之后被 get_new_blocks() 弹出
  → _maybe_evict_cached_block()
  → 删除 hash 映射
  → reset_hash()
  → 作为新 block 分配给请求
```

---

## 26. 容易疑惑的点

### 26.1 BlockPool 管的是逻辑 block 还是物理 tensor？

都不是完整意义上的“物理 tensor”。

它管的是与物理 KV cache slot 对齐的 `KVCacheBlock` 元数据。真实 tensor 在 Worker 侧，BlockPool 只持有 `block_id` 和调度账本。

### 26.2 free block 是否一定没有缓存内容？

不是。

```text
ref_cnt=0 的 cached block 可以在 free_block_queue 中，
同时仍然保留 block_hash，等待被 prefix cache 命中或被驱逐。
```

### 26.3 block 进入 prefix cache 是否意味着请求结束？

不是。

运行中的请求只要产生 full block，就可以把它 cache 起来。这个 block 仍可能 `ref_cnt > 0`，同时已经可被其他请求 prefix cache 命中。

### 26.4 prefix cache hit 是否会复制 KV？

不会。

命中时只是多个请求引用同一个 `KVCacheBlock`，通过 `ref_cnt` 防止被驱逐。Worker 侧会用相同 block id 读取已存在的 KV。

### 26.5 为什么请求释放时要 reversed(blocks)？

为了让尾部 block 更早进入驱逐候选。

`free_blocks()` 要求传入的 blocks 按 eviction priority 排序；请求释放时反向释放，使序列尾部 block 优先被驱逐。

位置：`single_type_kv_cache_manager.py:406` 到 `single_type_kv_cache_manager.py:407`

### 26.6 为什么 cached block 不立即从 hash map 删除？

因为它可能继续作为 prefix cache 被新请求复用。只有当它被重新分配用于写新内容时，旧 hash 才必须删除。

### 26.7 block_hash 相同会不会合并 block？

不会。

相同 hash 可以对应多个 block_id，`BlockHashToBlockMap` 会用 dict 保存多个 block。这是为了保持已分配 block table 稳定。

### 26.8 async KV load 为什么要 delay cache？

因为本地 block 已分配，但远端 KV 内容还没真正写入 GPU cache。提前放入 prefix cache 会让其他请求命中未就绪内容。

---

## 27. 最小心智模型

如果只记主线，可以记：

```text
BlockPool 是 KV block 的 Scheduler 侧账本：
它用 free queue 管可分配 / 可驱逐 block，
用 ref_cnt 管共享引用，
用 block_hash map 管 prefix cache，
用 null_block 表示结构占位。
```

完整链路再压缩成一句话：

```text
请求通过 KVCacheManager.allocate_slots() 获得 block ids，Worker 用这些 ids 写真实 KV；full block 通过 hash 进入 prefix cache；请求释放时只减少引用，cached block 可在 free queue 中继续等待复用，直到被重新分配时才真正 evict。
```