# Q003：为什么 vLLM 服务启动时会占用大部分 HBM？

完成度：可定位

## 问题

在实际启动 vLLM 服务时，如果设置 `--gpu-memory-utilization 0.9`，启动过程中 HBM 往往会被占据大部分。这些显存具体被谁占用了？什么时候占用的？这和 PagedAttention 中“动态按需分配 KVCacheBlock”是否矛盾？

## 一句话结论

`gpu_memory_utilization=0.9` 不是把 90% 显存直接分给每个请求，而是先给 vLLM 定义一个总显存预算；启动时 vLLM 会扣除权重、profiling 前向峰值、非 torch 分配和 CUDA Graph 估算后，把剩余预算一次性预分配成 GPU 侧 KV Cache Tensor，后续请求只是从这个已分配的大池子里按需分配逻辑 `KVCacheBlock`。

## L1：概念边界

### 它是什么

这个问题问的是 vLLM 启动阶段的 GPU 显存规划和实际分配行为。`--gpu-memory-utilization` 控制的是 vLLM 允许使用的 GPU 总显存预算，KV Cache 只是其中一部分。PagedAttention 的“按需分配”发生在逻辑 block 管理层：请求运行时按需拿 `KVCacheBlock` / `block_id`，但 GPU 上承载 KV 数据的大 Tensor 通常在服务启动阶段就已经预分配完成。

### 它解决什么问题

vLLM 这样设计是为了把运行期的 KV Cache 分配变成固定池内的 block 分配，避免每个请求到来或每生成一段 token 就临时 `cudaMalloc`。启动时先占住 KV Cache pool，可以降低运行期显存碎片、分配开销和 OOM 不确定性。请求执行时，Scheduler 和 BlockPool 只需要分配 CPU 侧 block 元数据和 block id，再通过 block table / slot mapping 在已经存在的 GPU Tensor 中寻址。

### 它不负责什么

`gpu_memory_utilization` 不负责决定单个请求最多能生成多少 token，也不直接等价于 KV Cache 的最终大小。KV Cache 的最终大小还要扣除模型权重、profile 期间测出来的峰值激活、非 torch 显存、CUDA Graph 估算，以及可能的用户显式 `kv_cache_memory_bytes`。`KVCacheBlock` 也不是 GPU 上的一块真实 Tensor，它主要是 CPU 侧 block 元数据。

### 和相邻模块的边界

- `CacheConfig`：保存用户传入的 `gpu_memory_utilization`、`kv_cache_memory_bytes`、`block_size` 等上层配置。
- `Worker.determine_available_memory()`：负责通过 profiling 计算可用于 KV Cache 的显存预算。
- `get_kv_cache_configs()`：负责把 KV Cache 规格和可用显存转换成 `KVCacheConfig`，包括 `num_blocks` 和 `kv_cache_tensors`。
- `GPUModelRunner.initialize_kv_cache()`：负责根据 `KVCacheConfig` 真正分配 GPU 侧 KV Cache Tensor。
- `BlockPool` / `KVCacheManager`：运行期只管理逻辑 KV blocks，不重新为每个请求申请大块 GPU Tensor。

## L2：端到端链路

### 输入

- 用户 CLI / API 配置：`--gpu-memory-utilization`、`--kv-cache-memory-bytes`、`--block-size`、`--kv-cache-dtype` 等。
- 模型结构：每层 attention / Mamba / MLA 等模块暴露的 `KVCacheSpec`。
- 设备状态：启动时 GPU 总显存、空闲显存、NCCL / CUDA runtime 初始化后的显存状态。
- profiling 结果：模型 dummy forward 的 torch 峰值、非 torch 增量、权重显存和 CUDA Graph 估算。

### 输出

- `available_kv_cache_memory_bytes`：worker 侧估算出的 KV Cache 可用显存。
- `KVCacheConfig`：包含 `num_blocks`、`kv_cache_tensors`、`kv_cache_groups`。
- GPU 侧真实 KV Cache Tensor：通过 `torch.zeros(..., device=self.device)` 在 HBM 中分配。
- CPU 侧运行期管理结构：`BlockPool`、`KVCacheBlock`、`FreeKVCacheBlockQueue`、`KVCacheManager`。

### 主链路

```text
CLI 参数
  -> EngineArgs.create_engine_config()
  -> VllmConfig.cache_config
  -> Worker.init_device()
  -> requested_memory = total_memory * gpu_memory_utilization
  -> Worker.load_model()
  -> 模型权重占用 HBM
  -> EngineCore._initialize_kv_caches()
  -> model_executor.get_kv_cache_specs()
  -> model_executor.determine_available_memory()
  -> get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)
  -> KVCacheConfig(num_blocks, kv_cache_tensors, kv_cache_groups)
  -> model_executor.initialize_from_config(kv_cache_configs)
  -> Worker.initialize_from_config(kv_cache_config)
  -> GPUModelRunner.initialize_kv_cache(kv_cache_config)
  -> torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
  -> HBM 中的 KV Cache pool 被一次性占用
```

### GPU Tensor 与 FreeKVCacheBlockQueue 的关系

`GPUModelRunner._allocate_kv_cache_tensors()` 分配的是 GPU 侧真实 HBM 数据池；`BlockPool` / `FreeKVCacheBlockQueue` 管理的是 CPU 侧 `KVCacheBlock` 元数据队列。二者不是包含关系，队列里没有 Tensor、Tensor slice、data pointer 或 Tensor view。它们通过同一个 `block_id` 命名空间关联：`KVCacheBlock(block_id=17)` 对应 GPU KV Cache Tensor 中第 17 个物理 block 区间。

```text
GPU 侧
KV Cache Tensor
  -> 第 0 个物理 block 区间
  -> 第 1 个物理 block 区间
  -> ...
  -> 第 17 个物理 block 区间

CPU 侧
BlockPool.blocks
  -> KVCacheBlock(block_id=0)
  -> KVCacheBlock(block_id=1)
  -> ...
  -> KVCacheBlock(block_id=17)

FreeKVCacheBlockQueue
  -> 当前空闲 / 可驱逐的 KVCacheBlock 链表
```

因此，初始化阶段真正占 HBM 的是 GPU KV Cache Tensor；运行期分配和释放 block 时，主要改变的是 CPU 侧 `KVCacheBlock.ref_cnt`、free queue 链表、request 到 block 的映射，以及 GPU 侧 block table / slot mapping。单个 block 的分配/释放通常不会释放或重新申请 GPU Tensor。

### 分配和释放主链路

运行期分配 block 时：

```text
Scheduler
  -> KVCacheManager.allocate_slots()
  -> SingleTypeKVCacheManager.get_num_blocks_to_allocate()
  -> SingleTypeKVCacheManager.allocate_new_blocks()
  -> BlockPool.get_new_blocks()
  -> FreeKVCacheBlockQueue.popleft_n()
  -> KVCacheBlock.ref_cnt += 1
  -> req_to_blocks[request_id].extend(new_blocks)
  -> SchedulerOutput 携带 new_block_ids
  -> GPUModelRunner._update_states()
  -> InputBatch.block_table.append_row(new_block_ids, req_index)
  -> BlockTables.append_block_ids()
  -> attention kernel 通过 block_table / slot_mapping 在已分配 Tensor 中读写 KV
```

运行期释放 block 时：

```text
请求完成 / abort / preempt / 滑动窗口丢弃旧 block
  -> KVCacheManager.free() 或 remove_skipped_blocks()
  -> SingleTypeKVCacheManager.free()
  -> req_to_blocks.pop(request_id)
  -> BlockPool.free_blocks(reversed(req_blocks))
  -> KVCacheBlock.ref_cnt -= 1
  -> ref_cnt == 0 的 block append/prepend 回 FreeKVCacheBlockQueue
  -> GPU KV Cache Tensor allocation 不变
```

如果开启 prefix caching，释放回 free queue 的 block 可能仍然保留 `block_hash` 和 GPU Tensor 中的旧 KV 数据。它此时是“可驱逐的缓存候选”：如果后续命中 prefix cache，`BlockPool.touch()` 会把它从 free queue 移除并增加 `ref_cnt`；如果后续被新请求拿去覆盖，`BlockPool._maybe_evict_cached_block()` 会先从 hash map 删除旧缓存索引并 reset hash。

### FreeKVCacheBlockQueue 的顺序语义

`FreeKVCacheBlockQueue` 这个数据结构本身是“从队首分配，默认回收到队尾”：`popleft()` / `popleft_n()` 从 `fake_free_list_head.next_free_block` 摘 block，`append()` / `append_n()` 把 block 接到 `fake_free_list_tail` 前面。因此，队首是最先被重新分配的位置，也就是最先被覆盖 / 驱逐的位置。

但从 vLLM 整体策略看，不能简单写成“所有回收都回队尾”。`BlockPool.free_blocks()` 默认 `prepend=False`，普通 request 结束时确实会把释放出的 block append 到队尾；但它也支持 `prepend=True`，用于把某些无缓存价值的 block 放到队首优先复用。典型例子是 `remove_skipped_blocks()`：SWA / chunked local attention 回收窗口外 block 时，会把带 `block_hash` 的 cached blocks 默认 append，让它们作为 best-effort prefix cache 留久一点；而没有 `block_hash` 的 uncached scratch blocks 会 `prepend=True` 放到队首，让下次分配优先覆盖它们。

```text
队首
  -> 最先被 popleft_n() 分配
  -> 最容易被覆盖 / 驱逐

队尾
  -> 默认释放 append 到这里
  -> 更晚被重新分配
  -> cached block 有更多机会被 prefix cache 命中复用
```

这说明 free queue 不只是普通 FIFO，它还承担了 LRU-like eviction order。源码注释明确说，队首是 least recently used block；如果同一批 block 的 last accessed time 相同，则 hash token 更多的 block，也就是请求尾部 block，会排得更靠前。因此，队列顺序的目标不是“清空旧数据”，而是在资源不足必须覆盖时，尽量优先覆盖复用价值低的 block，尽量保留仍有 `block_hash` 的 cached block，提高 prefix cache 的复用机会。

### 状态变化对象

- `CacheConfig`：保存用户配置，并在 KV cache 初始化后记录 `num_gpu_blocks`。
- `Worker`：记录 `init_snapshot`、`requested_memory`、`available_kv_cache_memory_bytes`、权重和 profiling 相关显存状态。
- `KVCacheConfig`：记录最终 `num_blocks`、每个 KV tensor 的字节大小和共享层。
- `GPUModelRunner`：持有真实 GPU KV Cache Tensor，并把 raw tensor reshape 成 attention backend 需要的布局。
- `BlockPool`：运行期管理 CPU 侧 `KVCacheBlock`，它的 block id 指向预分配 KV Cache Tensor 中的物理 block。

### 真实计算对象

真实占用 HBM 的大头是模型权重和 GPU 侧 KV Cache Tensor。`KVCacheBlock`、`BlockPool`、`FreeKVCacheBlockQueue` 等是 CPU 侧调度和元数据对象，不是 HBM 大头。请求运行时 attention kernel 通过 `block_table` 和 `slot_mapping` 在已分配的 KV Cache Tensor 中读写 K/V 数据。

## L3：源码对象

### 关键类 / 函数

- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/engine/arg_utils.py:1759`：`EngineArgs.create_engine_config()` 创建 `CacheConfig`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/engine/arg_utils.py:1760`：把用户传入的 `block_size` 写入 `CacheConfig`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/engine/arg_utils.py:1761`：把 `gpu_memory_utilization` 写入 `CacheConfig`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/engine/arg_utils.py:1762`：把 `kv_cache_memory_bytes` 写入 `CacheConfig`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/utils.py:405`：`request_memory()` 计算 vLLM 的请求显存预算。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/utils.py:410`：`requested_memory = ceil(total_memory * gpu_memory_utilization)`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:313`：worker 初始化设备后记录 `MemorySnapshot`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:314`：worker 调用 `request_memory()` 得到 `requested_memory`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/engine/core.py:235`：`EngineCore._initialize_kv_caches()` 是 KV cache 初始化主入口。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/engine/core.py:243`：从 executor/worker 获取每层 `KVCacheSpec`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/engine/core.py:257`：通过 profiling 获取可用于 KV Cache 的显存。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/engine/core.py:268`：调用 `get_kv_cache_configs()` 生成 `KVCacheConfig`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/engine/core.py:279`：生成 scheduler 侧使用的 `KVCacheConfig`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/engine/core.py:290`：把 worker 侧 `kv_cache_configs` 交给 executor 初始化。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:372`：`determine_available_memory()` 负责 profile 模型峰值显存并计算 KV cache 可用空间。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:384`：如果设置了 `kv_cache_memory_bytes`，它会直接作为 KV Cache 预算。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:404`：没有显式 KV Cache 大小时，执行 dummy forward 做 profiling。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:431`：`non_kv_cache_memory` 包含非 torch 增量、torch 峰值增量和权重显存。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:461`：计算 `available_kv_cache_memory_bytes`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:1956`：`get_kv_cache_configs()` 生成每个 worker 的 KV cache 配置。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:2066`：调用 `get_kv_cache_config_from_groups()` 生成单 worker 配置。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:2072`：多 worker 场景取最小 `num_blocks`，保证所有 worker 对齐。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:1247`：`get_kv_cache_config_from_groups()` 根据 group 和可用显存生成 `KVCacheConfig`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:1328`：返回 `KVCacheConfig(num_blocks, kv_cache_tensors, kv_cache_groups)`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:562`：worker 侧 `initialize_from_config()` 开始根据 `KVCacheConfig` 分配 KV cache。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:568`：把 `kv_cache_config.num_blocks` 写回本地 `cache_config`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:578`：调用 `model_runner.initialize_kv_cache(kv_cache_config)`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_model_runner.py:6999`：`_allocate_kv_cache_tensors()` 按 `KVCacheConfig` 分配 KV cache buffer。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_model_runner.py:7013`：遍历 `kv_cache_config.kv_cache_tensors`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_model_runner.py:7014`：`torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)` 真正占用 HBM。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_model_runner.py:7040`：`_reshape_kv_cache_tensors()` 把 raw int8 buffer reshape 成 backend 需要的 KV cache 视图。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:149`：`BlockPool` 用 `num_gpu_blocks` 初始化 CPU 侧 block 资源池。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:162`：`BlockPool` 创建 `KVCacheBlock(idx)` 元数据列表。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:168`：`FreeKVCacheBlockQueue` 用这些 `KVCacheBlock` 组成空闲 / 可驱逐队列。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:333`：`BlockPool.get_new_blocks()` 是运行期申请逻辑 block 的入口。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:347`：`get_new_blocks()` 通过 `free_block_queue.popleft_n()` 从队列取 block。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:350`：开启 prefix caching 时，重新分配 cached block 前会检查是否需要驱逐旧 hash。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:354`：新分配 block 的 `ref_cnt` 增加为 1。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:365`：`_maybe_evict_cached_block()` 清理被覆盖 block 的 prefix cache 索引。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:402`：`BlockPool.touch()` 在 prefix cache 命中时增加 block 引用计数。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:413`：prefix cache 命中的空闲 block 会先从 free queue 移除，避免被驱逐。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:419`：`BlockPool.free_blocks()` 是释放逻辑 block 的入口。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:433`：释放时对每个 block 执行 `ref_cnt -= 1`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:435`：只有 `ref_cnt == 0` 且不是 null block 的对象会回到 free queue。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:441`：默认释放路径把 block append 回 free queue。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:116`：`KVCacheBlock` 只保存 `block_id`、`ref_cnt`、hash 和链表指针等 CPU 元数据。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:174`：`FreeKVCacheBlockQueue` 注释说明初始按 block id 排序，分配再释放后按 eviction order 排列。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:176`：源码注释说明队首是 least recently used block。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:178`：源码注释说明同一 last accessed time 下，hash tokens 更多的尾部 block 排在更前。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:217`：`FreeKVCacheBlockQueue.popleft()` 从队首弹出一个 block。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:254`：`FreeKVCacheBlockQueue.popleft_n()` 从双向链表头部摘下多个 block。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:307`：`FreeKVCacheBlockQueue.append()` 把单个 block 放回队尾。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:330`：`FreeKVCacheBlockQueue.prepend_n()` 支持把 blocks 放到队首。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:351`：`FreeKVCacheBlockQueue.append_n()` 支持把 blocks 放到队尾。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:259`：`allocate_new_blocks()` 为某个 request 申请新的逻辑 block。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:276`：`req_to_blocks[request_id]` 保存请求持有的 block 元数据列表。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:282`：新分配 blocks 被追加到请求的 block 列表。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:363`：`SingleTypeKVCacheManager.free()` 释放某个 request 持有的 blocks。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:371`：释放时从 `req_to_blocks` 移除该 request 的 block 列表。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:377`：释放路径调用 `block_pool.free_blocks()`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:448`：`remove_skipped_blocks()` 用于 SWA / chunked local 等场景提前释放窗口外 block。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:479`：窗口外 block 被拆成 uncached scratch blocks 和 cached blocks 两类。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:497`：源码注释说明 `prepend=True` 让 uncached scratch blocks 成为下一批分配候选，而 cached blocks 作为 best-effort prefix cache 留在后面。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:500`：cached blocks 走默认 `free_blocks()`，即 append 到队尾。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:501`：uncached scratch blocks 走 `prepend=True`，被放到队首优先复用。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_model_runner.py:1368`：worker 更新 request 侧 block ids。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_model_runner.py:1401`：worker 把新 block ids 写入 input batch 的 block table。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu/block_table.py:96`：`BlockTables.append_block_ids()` 把 block ids 写进 GPU 侧 block table。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu/block_table.py:267`：slot mapping kernel 通过 token position 计算 logical block index。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu/block_table.py:269`：slot mapping kernel 从 block table 取 physical block id。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu/block_table.py:273`：slot mapping kernel 计算最终写入 KV Cache Tensor 的 slot id。

### 关键字段

- `CacheConfig.gpu_memory_utilization`：用户配置的 vLLM GPU 显存使用比例。
- `CacheConfig.kv_cache_memory_bytes`：用户手动指定 KV Cache 大小时，会绕过基于 `gpu_memory_utilization` 的 KV cache 预算计算。
- `Worker.init_snapshot.total_memory`：设备总显存，用于计算 `requested_memory`。
- `Worker.init_snapshot.free_memory`：启动时空闲显存，用于校验是否满足 `requested_memory`。
- `Worker.requested_memory`：`total_memory * gpu_memory_utilization`，表示 vLLM 希望控制在内的总显存预算。
- `profile_result.weights_memory`：模型权重占用。
- `profile_result.torch_peak_increase`：profile forward 期间 torch allocated peak 增量。
- `profile_result.non_torch_increase`：非 torch 分配增量，例如 CUDA/NCCL/runtime 等。
- `Worker.cudagraph_memory_estimate`：CUDA Graph 相关显存估算。
- `Worker.available_kv_cache_memory_bytes`：最终可用于 KV Cache 的显存预算。
- `KVCacheConfig.num_blocks`：KV cache pool 中可用的 block 数。
- `KVCacheConfig.kv_cache_tensors`：描述每个 GPU KV cache raw tensor 的大小和共享层。
- `KVCacheTensor.size`：某个 raw KV cache buffer 需要分配的字节数。
- `KVCacheTensor.shared_by`：共享该 raw KV cache buffer 的 layer names。
- `KVCacheBlock.block_id`：CPU 元数据中的物理 block 编号，与 GPU KV Cache Tensor 中的 block 区间按约定对应。
- `KVCacheBlock.ref_cnt`：引用计数，决定 block 是否仍被请求持有，是否能回到 free queue。
- `KVCacheBlock._block_hash`：prefix cache 索引 key，表示这个 block 的数据是否可被相同 prefix 复用。
- `KVCacheBlock.prev_free_block` / `next_free_block`：`FreeKVCacheBlockQueue` 双向链表指针。
- `FreeKVCacheBlockQueue.num_free_blocks`：当前 free queue 中可分配 / 可驱逐 block 数。
- `SingleTypeKVCacheManager.req_to_blocks`：request id 到 `KVCacheBlock` 列表的映射，是运行期请求持有 block 的 CPU 侧状态。

### 状态改变方法

- `EngineArgs.create_engine_config()`：把 CLI 参数转换成 `VllmConfig.cache_config`。
- `Worker.init_device()`：初始化 CUDA/distributed 环境、记录 memory snapshot、计算 `requested_memory`。
- `Worker.load_model()`：加载模型权重，先占用一部分 HBM。
- `Worker.determine_available_memory()`：通过 dummy forward profiling 计算 KV cache 可用显存。
- `get_kv_cache_configs()`：把模型 KV cache specs 和显存预算转换成 `KVCacheConfig`。
- `Worker.initialize_from_config()`：接收 `KVCacheConfig` 并触发 worker 侧 KV cache 初始化。
- `GPUModelRunner._allocate_kv_cache_tensors()`：真正分配 GPU KV Cache Tensor，占用 HBM。
- `GPUModelRunner._reshape_kv_cache_tensors()`：把 raw tensor 解释成具体 backend 的 KV cache 形状。
- `KVCacheManager.allocate_slots()`：运行期为请求分配逻辑 slots / blocks，但不再为每个请求申请独立的大 GPU Tensor。
- `SingleTypeKVCacheManager.allocate_new_blocks()`：计算某个 request 还需要多少 block，并把新 block 追加到 `req_to_blocks[request_id]`。
- `BlockPool.get_new_blocks()`：运行期从 CPU 侧 free block queue 取出逻辑 block，并把 `ref_cnt` 增加为 1。
- `FreeKVCacheBlockQueue.popleft_n()`：从 free queue 头部摘下 N 个 `KVCacheBlock`，更新双向链表和 `num_free_blocks`。
- `BlockTables.append_block_ids()`：worker 侧把分配出的 block ids 写入 GPU 侧 block table，让 kernel 能从逻辑 block 找到物理 block。
- `BlockPool.free_blocks()`：释放 request 持有的 blocks，减少 `ref_cnt`，归零后放回 free queue。
- `SingleTypeKVCacheManager.free()`：请求完成 / abort / preempt 时从 `req_to_blocks` 删除请求的 block 列表并释放。
- `SingleTypeKVCacheManager.remove_skipped_blocks()`：SWA / chunked local attention 场景下提前释放窗口外 block，并用 null block 填充逻辑位置。
- `BlockPool.touch()`：prefix cache 命中时增加 block 引用计数；如果 block 在 free queue 中，会先从队列移除以避免被驱逐。
- `BlockPool._maybe_evict_cached_block()`：cached block 被重新分配覆盖前，从 prefix cache hash map 删除旧索引并 reset hash。

### 关键配置

- `--gpu-memory-utilization`：控制 vLLM 的总显存预算，不直接等于 KV Cache 大小。
- `--kv-cache-memory-bytes`：手动指定 KV Cache 显存预算；设置后不再按 `gpu_memory_utilization` 自动计算 KV Cache 大小。
- `--block-size`：影响每个 KV block 容纳的 token 数，也影响 `num_blocks` 和 block table 规划。
- `--kv-cache-dtype`：影响 `KVCacheSpec.page_size_bytes`，从而影响同样显存下能切出的 block 数。
- `--max-model-len`：影响单请求最大 KV cache 需求和可支持并发估算。
- `--max-num-batched-tokens`：影响 profiling、SWA / chunked local attention 的最大 KV cache 估算。
- `--num-gpu-blocks-override`：直接覆盖自动计算出的 `num_blocks`，常用于实验和调试。
- CUDA Graph 相关配置：影响是否额外估算和预留 CUDA Graph 显存。

### 计划对象与结果对象

- 计划对象：`CacheConfig`、`KVCacheSpec`、`KVCacheGroupSpec`、`KVCacheTensor`、`KVCacheConfig`。
- 状态对象：`Worker`、`GPUModelRunner`、`BlockPool`、`KVCacheManager`、`InputBatch`。
- 结果对象：GPU 侧真实 KV Cache Tensor、`available_kv_cache_memory_bytes`、`num_blocks`、启动后的 HBM 占用。

## L4：取舍、性能与排查

### 为什么这样设计

vLLM 的目标是高吞吐 serving，而不是让服务启动后看起来显存占用很低。预分配 KV Cache pool 可以把运行期不可控的显存申请转化成可控的 block id 分配。这样 Scheduler 在每轮调度时只需要判断 free block 数量，而不需要关心 GPU allocator 是否还能找到连续空间。

PagedAttention 的价值也正是在这个前提下体现：HBM 中先有一个大 KV Cache pool，运行期请求持有的是逻辑 block 序列，这些逻辑 block 可以映射到 pool 中不连续的物理 block。这样既避免了传统 per-request 连续大块预留，也避免了运行期频繁申请释放 GPU 显存。

### 优化了什么

- **吞吐**：更多请求可以共享同一个固定 KV cache pool，减少因碎片和静态预留导致的浪费。
- **运行期稳定性**：请求到来时主要是分配 block id，不频繁触发 GPU allocator。
- **调度可预测性**：Scheduler 能用 `num_blocks` 精确判断是否还能接纳新 token / 请求。
- **碎片控制**：避免 per-request 连续显存分配导致外部碎片。
- **prefix cache / 共享**：多个请求可以通过 `ref_cnt` 和 block hash 共享已计算 block。

### 牺牲了什么

- **启动显存占用高**：服务刚启动，即使没有请求，也会占用大量 HBM。
- **启动时间更长**：需要 load model、profile、生成 KVCacheConfig、分配大 Tensor、warmup / capture CUDA Graph。
- **弹性较弱**：KV Cache pool 大小在启动后基本固定，不能自然让给其他进程。
- **低并发浪费明显**：如果实际请求量很低，预分配的大 KV cache pool 可能长期空闲。
- **排查复杂**：用户看到 HBM 被占满时，需要区分权重、KV cache、CUDA graph、profiling peak、非 torch 分配。

### 什么情况下收益不明显

- 单请求、低并发、短上下文场景：大 KV cache pool 的吞吐收益不明显，显存预占反而显得浪费。
- 显存紧张且有其他进程共用 GPU：`gpu_memory_utilization` 设置过高容易启动失败或挤压其他服务。
- 输出很短的 workload：KV Cache 需求小，预分配大量 blocks 的收益有限。
- 模型权重已经接近显存上限：可用于 KV Cache 的空间很小，PagedAttention 只能管理有限资源，不能凭空创造显存。

### 常见问题与排查路径

1. 现象：服务刚启动、没有请求，`nvidia-smi` 就显示 HBM 几乎被占满。
   - 可能原因：vLLM 已经按 `gpu_memory_utilization` 预分配了 KV Cache Tensor，同时权重和 CUDA Graph 也占用显存。
   - 排查对象：启动日志中的 `Available KV cache memory`、`GPU KV cache size`、`Maximum concurrency`、`gpu_memory_utilization`。
   - 验证方法：降低 `--gpu-memory-utilization` 或设置较小的 `--kv-cache-memory-bytes`，观察启动后 HBM 占用是否同步下降。

2. 现象：设置 `gpu_memory_utilization=0.9` 后仍然 OOM。
   - 可能原因：其他进程占用显存、profiling 低估、CUDA Graph 实际占用高于估算、模型权重或 `max_num_batched_tokens` 过大。
   - 排查对象：`init_snapshot.free_memory`、`requested_memory`、`available_kv_cache_memory_bytes`、CUDA Graph 日志。
   - 验证方法：先降低到 0.8 或关闭/调整 CUDA Graph，再逐步提高；同时确认启动前 GPU 上没有其他进程。

3. 现象：想让 vLLM 只占固定大小的 KV Cache，而不是按比例占满。
   - 可能原因：当前使用的是 `gpu_memory_utilization` 自动预算模式。
   - 排查对象：`kv_cache_memory_bytes` 是否设置。
   - 验证方法：使用 `--kv-cache-memory-bytes` 手动指定 KV Cache 大小；源码中该配置会直接返回为 KV Cache 预算。

4. 现象：请求运行时显存没有随着请求数明显增长。
   - 可能原因：KV Cache Tensor 已在启动阶段分配，运行时只是分配逻辑 block。
   - 排查对象：`BlockPool` free block 数、`num_blocks`、请求的 block ids，而不是 `nvidia-smi` 总占用。
   - 验证方法：观察 vLLM metrics 中 KV cache usage / block usage，而不是只看进程总 HBM。

5. 现象：并发上不去，但 HBM 看起来已经被占满。
   - 可能原因：KV cache block 数不足、`max_model_len` 太大、block size / dtype / attention 类型导致每请求 block 需求高。
   - 排查对象：`KVCacheConfig.num_blocks`、`page_size_bytes`、`max_model_len`、`Maximum concurrency` 日志。
   - 验证方法：降低 `max_model_len`、减少 `max_num_batched_tokens`、使用更小 KV cache dtype 或更小模型做对照。

6. 现象：误以为 `FreeKVCacheBlockQueue` 里保存的是具体 Tensor 或 Tensor 元数据。
   - 可能原因：把启动时预分配的 GPU KV Cache Tensor 和运行期 CPU 侧 block 元数据混成一层理解。
   - 排查对象：`KVCacheBlock` 字段、`BlockPool.blocks`、`FreeKVCacheBlockQueue` 链表、`GPUModelRunner._allocate_kv_cache_tensors()`。
   - 验证方法：查看 `KVCacheBlock` 定义，它只包含 `block_id`、`ref_cnt`、hash 和链表指针；再查看 `torch.zeros(kv_cache_tensor.size, ...)`，真实 Tensor 由 model runner 持有。

7. 现象：释放 request 后 HBM 总占用没有下降。
   - 可能原因：释放 block 只是 `ref_cnt -= 1` 并把元数据放回 free queue，GPU KV Cache Tensor 的 allocation 不会按 request 释放。
   - 排查对象：`BlockPool.free_blocks()`、`FreeKVCacheBlockQueue.num_free_blocks`、KV cache usage metrics。
   - 验证方法：观察 free block 数是否增加，而不是期待 `nvidia-smi` 中进程显存立即下降。

8. 现象：误以为 free queue 是简单 FIFO，所有回收 block 都进入队尾。
   - 可能原因：只看到了 `append_n()` 默认路径，没有看到 `prepend=True` 特例和队列的 eviction order 语义。
   - 排查对象：`FreeKVCacheBlockQueue` 注释、`BlockPool.free_blocks(prepend=False)`、`remove_skipped_blocks()` 中 cached / uncached blocks 的不同释放路径。
   - 验证方法：确认普通 request 结束走默认 append；再确认 SWA / chunked local 回收窗口外 uncached scratch blocks 时走 `prepend=True`，让无缓存价值的 block 优先被覆盖。

### Benchmark 设计

- 指标：启动后 HBM 占用、`available_kv_cache_memory_bytes`、`num_blocks`、最大并发、TTFT、TPOT、throughput、OOM 率、KV cache usage。
- 变量：`gpu_memory_utilization`、`kv_cache_memory_bytes`、`max_model_len`、`block_size`、`kv_cache_dtype`、`max_num_batched_tokens`、是否启用 CUDA Graph。
- 对照组：
  - `gpu_memory_utilization=0.7/0.8/0.9/0.95`。
  - 自动 KV cache 预算 vs 手动 `kv_cache_memory_bytes`。
  - 短上下文 workload vs 长上下文 workload。
  - 低并发 vs 高并发。
- 预期现象：
  - `gpu_memory_utilization` 越高，启动后 HBM 占用通常越高，`num_blocks` 越多。
  - 手动降低 `kv_cache_memory_bytes` 会降低启动后 HBM 占用，但最大并发和可承载上下文下降。
  - 高并发/长上下文更能体现预分配 KV Cache pool 的吞吐收益。
  - 低并发/短输出场景下，大量预分配 KV blocks 可能长期空闲。

## 源码证据

- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/engine/arg_utils.py:1759`：CLI 参数在 `EngineArgs.create_engine_config()` 中形成 `CacheConfig`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/engine/arg_utils.py:1761`：`gpu_memory_utilization` 被写入 `CacheConfig`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/utils.py:410`：`requested_memory` 按 `total_memory * gpu_memory_utilization` 计算。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:313`：worker 初始化设备后记录启动时显存快照。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:372`：`determine_available_memory()` 负责计算可用于 KV cache 的显存。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:384`：设置 `kv_cache_memory_bytes` 时会直接使用该值作为 KV Cache 预算。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:431`：`non_kv_cache_memory` 包含权重、torch peak 和非 torch 增量。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:461`：`available_kv_cache_memory_bytes` 由 requested memory 扣除非 KV 和 CUDA Graph 估算得到。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/engine/core.py:268`：`get_kv_cache_configs()` 使用 `vllm_config`、`kv_cache_specs`、`available_gpu_memory` 生成 KV cache 配置。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:1328`：`KVCacheConfig` 包含 `num_blocks`、`kv_cache_tensors`、`kv_cache_groups`。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_worker.py:578`：worker 根据 `KVCacheConfig` 初始化 model runner 的 KV cache。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu_model_runner.py:7014`：`torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)` 是 GPU KV Cache Tensor 真正占用 HBM 的位置。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:162`：`BlockPool` 创建的是 CPU 侧 `KVCacheBlock` 元数据列表。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:168`：`FreeKVCacheBlockQueue` 管理的是这些 `KVCacheBlock` 对象，不是 Tensor。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:333`：运行期从 free block pool 获取逻辑 block。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/block_pool.py:419`：释放 block 时只改变引用计数和 free queue 状态。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:176`：`FreeKVCacheBlockQueue` 注释说明队首是 LRU block，即优先驱逐 / 覆盖的位置。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:217`：`popleft()` 从队首分配。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:307`：`append()` 把 block 回收到队尾。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/kv_cache_utils.py:330`：`prepend_n()` 支持把 block 放到队首，作为默认 append 的例外能力。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:497`：SWA / chunked local 回收窗口外 block 时，注释说明 uncached scratch blocks 会优先复用，cached blocks 作为 best-effort prefix cache 保留。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:500`：cached blocks 走默认释放路径。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/core/single_type_kv_cache_manager.py:501`：uncached scratch blocks 通过 `prepend=True` 放到队首。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu/block_table.py:269`：GPU 侧通过 block table 从逻辑 block 找到物理 block id。
- `D:/lzy/project/kv_pool/tmp_file/vllm/vllm/v1/worker/gpu/block_table.py:273`：GPU 侧通过 physical block id 和 offset 计算 KV Cache Tensor 中的 slot。

## 容易混淆点

- 混淆点 1：`gpu_memory_utilization=0.9` 不等于 KV Cache 占 90% HBM。它是 vLLM 总预算，KV Cache 是扣除权重和运行时开销后的剩余部分。
- 混淆点 2：PagedAttention 的“按需分配”不是运行时按请求不断 `cudaMalloc`，而是在预分配 KV Cache Tensor 中按需分配逻辑 block。
- 混淆点 3：`KVCacheBlock` 不是 GPU Tensor，它是 CPU 侧元数据；真正占 HBM 的是 `kv_cache_tensors` 对应的 `torch.zeros` 分配。
- 混淆点 4：启动后 HBM 高占用不代表已经有请求占满 KV Cache，只表示 vLLM 把 KV Cache pool 预留好了。
- 混淆点 5：`nvidia-smi` 只能看到进程总显存，不能直接告诉你 KV Cache usage；运行期是否真的用满要看 block usage / KV cache metrics。
- 混淆点 6：`FreeKVCacheBlockQueue` 不是 GPU Tensor 队列，也不是 Tensor 元数据队列；它保存的是 CPU 侧 `KVCacheBlock` 对象。
- 混淆点 7：`KVCacheBlock(block_id=N)` 和 GPU KV Cache Tensor 第 N 个物理 block 区间有关联，但这个关联是通过 `block_id`、`block_table`、`slot_mapping` 和 kernel 寻址实现的，不是通过 `KVCacheBlock` 保存 Tensor 指针实现的。
- 混淆点 8：释放 block 不会释放 GPU Tensor allocation，只会让该 `block_id` 回到可复用 / 可驱逐状态；所以 request 结束后 HBM 总占用通常不会下降。
- 混淆点 9：prefix cache 命中不会复制 GPU Tensor 数据，而是复用已有 `block_id`，通过 `touch()` 增加引用计数并把可驱逐 block 从 free queue 中移除。
- 混淆点 10：`FreeKVCacheBlockQueue` 的基础操作是从队首分配、默认回收到队尾，但完整策略不是所有 block 都回队尾；`BlockPool.free_blocks(prepend=True)` 可以把无缓存价值的 scratch blocks 放到队首优先复用。
- 混淆点 11：队首不是“最值得保留的位置”，而是最容易被重新分配 / 覆盖的位置；队尾 block 更晚被拿走，因此带 `block_hash` 的 cached block 留在队尾附近更有机会被 prefix cache 命中。
- 混淆点 12：这个队列顺序服务于 best-effort prefix cache 复用：优先覆盖无 hash 的 uncached block，尽量延后覆盖有 hash 的 cached block，但并不保证 cached block 永远不被覆盖。

## 我还不确定的点

- 当前文档基于 `D:/lzy/project/kv_pool/tmp_file/vllm` 的 `releases/v0.23.0` 分支分析；如果后续切到 vLLM 0.25.0 或更新版本，CUDA Graph 估算、KV cache zeroing、block table 或 V2 runner 细节可能变化，需要重新校对源码。
- 不同平台后端（CUDA、ROCm、XPU、CPU）在 memory profiling 和 graph memory 估算上存在差异，本文重点覆盖 CUDA/GPU worker 的主路径。
