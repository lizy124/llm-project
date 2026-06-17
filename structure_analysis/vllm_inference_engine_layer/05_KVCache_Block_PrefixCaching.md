# 05 KV Cache、Block 与 Prefix Caching

本篇详细梳理 vLLM 推理引擎层中的 KV Cache、block 管理、prefix caching、block table 和 KV connector/offload。

## 1. KV Cache 为什么是 vLLM 核心

自回归 LLM 推理中，每生成一个 token 都需要历史 token 的 K/V。直接重复计算历史 attention 成本极高，所以推理系统会缓存每层 attention 的 K/V。

vLLM 的核心创新之一是把 KV cache 拆成固定大小的 block/page，用类似虚拟内存分页的方式管理。

这带来几个能力：

- 多请求共享显存池；
- 动态分配和回收 KV block；
- 支持 prefix caching；
- 支持 preemption 后恢复；
- 支持 chunked prefill；
- 支持 paged attention kernel；
- 支持 KV transfer/offload/disaggregated prefill。

## 2. 相关路径

```text
vllm/config/cache.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/block_pool.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/kv_cache_interface.py
vllm/v1/worker/block_table.py
vllm/v1/worker/gpu/block_table.py
vllm/v1/worker/gpu/attn_utils.py
vllm/v1/worker/gpu_model_runner.py
vllm/distributed/kv_transfer/
vllm/v1/kv_offload/
csrc/*cache*
```

## 3. KV Cache 配置入口

KV cache 的用户配置主要在 `vllm/config/cache.py`。

常见关键参数：

| 参数 | 作用 |
|---|---|
| `block_size` | 物理 KV cache block 存放多少 token |
| `hash_block_size` | prefix cache hash 的 token 粒度 |
| `cache_dtype` | KV cache dtype |
| `enable_prefix_caching` | 是否启用 prefix caching |
| `prefix_caching_hash_algo` | prefix hash 算法 |
| `kv_cache_memory_bytes` | 手动指定 KV cache 显存大小 |
| `gpu_memory_utilization` | 自动 profile 时显存利用率 |
| `num_gpu_blocks_override` | 覆盖 GPU block 数 |
| `kv_offloading_size` | KV offload 相关大小 |
| `mamba_cache_mode` | Mamba/hybrid cache 模式 |

这些参数最终会影响：

- worker 侧分配多少 KV tensor；
- scheduler 侧认为有多少可用 block；
- attention backend 使用什么 cache layout；
- prefix caching 的命中粒度；
- 是否允许 offload/transfer。

## 4. KV Cache 初始化链路

初始化链路从 `EngineCore._initialize_kv_caches()` 开始，位置是 `code/vllm/vllm/v1/engine/core.py:240`。

```text
EngineCore._initialize_kv_caches
  ↓
register_all_kvcache_specs(vllm_config)
  ↓
Executor.get_kv_cache_specs()
  ↓
Worker.get_kv_cache_spec()
  ↓
GPUModelRunner.get_kv_cache_spec()
  ↓
model_executor Attention 层提供 KVCacheSpec
  ↓
Executor.determine_available_memory()
  ↓
Worker.determine_available_memory()
  ↓
GPUModelRunner.profile_run()
  ↓
get_kv_cache_configs(...)
  ↓
generate_scheduler_kv_cache_config(...)
  ↓
Executor.initialize_from_config(kv_cache_configs)
  ↓
Worker.initialize_from_config(kv_cache_config)
  ↓
GPUModelRunner.initialize_kv_cache(kv_cache_config)
```

### worker 侧显存 profile

`Worker.determine_available_memory()` 定义在 `code/vllm/vllm/v1/worker/gpu_worker.py:371`。

它会：

1. 如果用户手动指定 `kv_cache_memory_bytes`，仍然跑一次 profile/warmup，但返回用户指定大小；
2. 否则执行 dummy forward；
3. 统计非 KV cache 显存、权重显存、activation peak、CUDA graph 预估显存；
4. 计算剩余可用于 KV cache 的显存。

这就是 vLLM 启动时要 profile 的原因。

## 5. KVCacheManager

`KVCacheManager` 定义在 `code/vllm/vllm/v1/core/kv_cache_manager.py:110`。

它是 scheduler 侧的 KV block 管理入口。

### 主要职责

| 方法 | 作用 |
|---|---|
| `get_computed_blocks()` | 查找 prefix cache 命中的 blocks |
| `allocate_slots()` | 为请求新 token 分配 KV slots/blocks |
| `free()` | 释放请求 KV blocks |
| `remove_skipped_blocks()` | 移除跳过/不再需要的 blocks |
| `pop_blocks_for_free()` | 取出待释放 blocks |
| `evict_blocks()` | 主动驱逐 blocks |
| `reset_prefix_cache()` | 重置 prefix cache |
| `get_num_common_prefix_blocks()` | 获取共同前缀 block 数，用于 cascade attention |
| `cache_blocks()` | 将 computed blocks 写入 prefix cache 索引 |
| `take_new_block_ids()` | 取本步新分配 block ids |
| `new_step_starts()` | 新 step 开始时重置临时状态 |

### KVCacheBlocks

`KVCacheBlocks` 定义在 `code/vllm/vllm/v1/core/kv_cache_manager.py:25`。

它是 `KVCacheManager` 和 `Scheduler` 之间的接口对象，用于隐藏内部 block 管理细节。

它的结构是：

```text
blocks: tuple[Sequence[KVCacheBlock], ...]
```

外层 tuple 对应 KV cache group，内层是该 group 的 block 序列。

为什么要按 group 组织？

因为 hybrid KV cache 可能存在多个 cache group，例如普通 attention、sliding window、Mamba state 等，它们不一定永远共享完全相同的 block 语义。

## 6. allocate_slots 的核心逻辑

`allocate_slots()` 从 `code/vllm/vllm/v1/core/kv_cache_manager.py:244` 开始。

它是 KV 管理最重要的方法。

输入包括：

- request；
- 本步要新算的 token 数；
- prefix cache 命中的 token 数；
- 外部 connector 命中的 token 数；
- spec decode lookahead token 数；
- encoder token 数；
- 是否要求 full sequence 必须 fit；
- reserved blocks；
- 是否已有请求被调度。

注释里给出一张非常重要的布局图，含义如下：

```text
| comp | new_comp | ext_comp | new | lookahead |

comp      = 已经计算过的 token
new_comp  = 本地 prefix cache 新命中的 token
ext_comp  = connector 外部命中的 token
new       = 本步要计算的新 token
lookahead = spec decode 额外预留 token
```

allocate 分三阶段：

1. 释放 `comp` 中不再需要的 blocks，并检查是否有足够 free blocks；
2. 处理 prefix token：本地命中、外部命中、sliding window 外 block 释放；
3. 为本步要计算的新 token 和 lookahead token 分配新 blocks。

返回值：

- 成功：返回新分配的 `KVCacheBlocks`；
- 失败：返回 `None`，Scheduler 会考虑 preemption 或跳过请求。

## 7. BlockPool

`BlockPool` 定义在 `code/vllm/vllm/v1/core/block_pool.py:130`。

它是底层 block 池，负责实际维护 free/cached/evictable blocks。

主要方法：

| 方法 | 作用 |
|---|---|
| `get_cached_block()` | 按 hash 查找 cached block |
| `cache_full_blocks()` | 把完整 blocks 放入 prefix cache 索引 |
| `get_new_blocks()` | 分配新 blocks |
| `_maybe_evict_cached_block()` | 必要时驱逐 cached block |
| `touch()` | 更新 blocks 使用状态，避免被驱逐 |
| `free_blocks()` | 释放 blocks |
| `evict_blocks()` | 按 block id 驱逐 |
| `reset_prefix_cache()` | 清理 prefix cache |
| `get_num_free_blocks()` | 获取 free block 数 |
| `get_usage()` | 获取使用率 |
| `take_events()` | 取 KV cache event |

### BlockHashToBlockMap

`BlockHashToBlockMap` 在 `code/vllm/vllm/v1/core/block_pool.py:34`。

它维护：

```text
BlockHashWithGroupId -> KVCacheBlock
```

这是 prefix cache 的索引基础。

## 8. Prefix Caching

Prefix caching 的目标：多个请求共享相同前缀时，复用已经计算好的 KV block。

### 基本流程

```text
请求进入 InputProcessor/EngineCore
  ↓
为 request 计算 block hashes
  ↓
Scheduler 调度 waiting request
  ↓
KVCacheManager.get_computed_blocks(request)
  ↓
BlockPool 按 hash 查找 cached block
  ↓
命中部分变成 computed tokens
  ↓
Scheduler 只调度未命中 token
```

### 注意点

1. prefix cache 只能复用完整 block；
2. 即使全部 prompt 命中，通常也要重算最后一个 token 来获得 logits；
3. non-causal attention 会禁用 prefix caching；
4. sliding window 会影响哪些 blocks 仍需保留；
5. prompt logprobs、pooling 等场景可能跳过读取 prefix cache；
6. KV connector 可以提供外部命中 token，与本地 prefix cache 组合。

## 9. hash_block_size 与 block_size

- `block_size`：物理 KV cache block 大小；
- `hash_block_size`：prefix cache hash 粒度。

大多数情况下二者一致；但 hash 粒度也可以更细，用于更细粒度的 prefix cache 匹配。

EngineCore 会调用：

```text
resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
```

生成 scheduler 使用的 block size 和 hash block size。

## 10. Block table：scheduler block 到 kernel block 的桥

Scheduler 侧只关心 request 分配了哪些 block ids；worker/kernel 侧需要知道每个 token 对应 KV cache 的哪个 slot。

这中间靠 block table 和 slot mapping。

相关路径：

```text
vllm/v1/worker/block_table.py
vllm/v1/worker/gpu/block_table.py
vllm/v1/worker/gpu_model_runner.py
```

在 `GPUModelRunner.execute_model()` 中，会调用：

```text
slot_mappings_by_group, slot_mappings = self._get_slot_mappings(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4244`。

slot mapping 的作用：

```text
本 step 中第 i 个 token
  -> 属于哪个 request
  -> 是该 request 的第几个 token
  -> 对应哪个 KV block
  -> block 内 offset 是多少
  -> 最终 KV cache tensor 写入/读取位置
```

attention kernel 正是依靠 block table/slot mapping 找到 KV cache 页。

## 11. KV cache tensor 分配

worker 侧分配 KV cache tensor 的入口：

```text
Worker.initialize_from_config()
  -> GPUModelRunner.initialize_kv_cache()
```

代码位置：

- `Worker.initialize_from_config()`：`code/vllm/vllm/v1/worker/gpu_worker.py:562`
- `GPUModelRunner.initialize_kv_cache()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7303`

`GPUModelRunner` 内还有：

- `_allocate_kv_cache_tensors()`：分配 tensor；
- `_reshape_kv_cache_tensors()`：按 backend/layout reshape；
- `_update_hybrid_attention_mamba_layout()`：处理 hybrid attention/mamba 布局；
- `initialize_attn_backend()`：初始化 attention backend；
- `initialize_metadata_builders()`：初始化 attention metadata builder。

## 12. KV Connector / Transfer / Offload

相关路径：

```text
vllm/distributed/kv_transfer/
vllm/distributed/kv_transfer/kv_connector/
vllm/v1/kv_offload/
```

KV connector 主要用于：

- disaggregated prefill/decode；
- remote KV load/save；
- KV offload；
- 外部 KV cache 系统对接。

Scheduler 侧 connector：

```text
KVConnectorFactory.create_connector(..., role=KVConnectorRole.SCHEDULER)
```

Worker 侧 connector：

```text
ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
```

worker 初始化位置：`code/vllm/vllm/v1/worker/gpu_worker.py:575`。

### connector 在调度中的影响

waiting request 调度时，Scheduler 不只查本地 prefix cache，还可能问 connector：

```text
connector.get_num_new_matched_tokens(...)
```

如果外部 KV 可用，Scheduler 可以把这些 token 当成 external computed tokens，不必本地重新算。

## 13. KV cache events

如果启用 KV cache events，BlockPool/KVCacheManager 会产生事件，例如 block stored/free/evict。

用途：

- 观测 cache 行为；
- 外部系统同步；
- 调试 prefix cache 命中和驱逐。

相关方法：

- `KVCacheManager.take_events()`：`code/vllm/vllm/v1/core/kv_cache_manager.py:554`
- `BlockPool.take_events()`：`code/vllm/vllm/v1/core/block_pool.py:517`

## 14. KV cache 的生命周期

```text
启动阶段：
  model layer 声明 KVCacheSpec
  worker profile 可用显存
  EngineCore 生成 KVCacheConfig
  worker 分配 KV cache tensor

请求进入：
  request 计算 block hash
  scheduler 放入 waiting

调度阶段：
  get_computed_blocks 查 prefix cache
  allocate_slots 分配新 blocks
  SchedulerOutput 带 block ids 发给 worker

执行阶段：
  GPUModelRunner 构造 block table / slot mapping
  Attention 写入新 K/V，读取历史 K/V

完成阶段：
  scheduler.update_from_output 更新状态
  完成/取消时 free request blocks
  完整 blocks 可能进入 prefix cache 索引
  空闲或可驱逐 blocks 回到 block pool
```

## 15. 常见调试问题定位

| 问题 | 优先看 |
|---|---|
| 启动时 KV cache 太小/OOM | `gpu_worker.py:determine_available_memory`, `config/cache.py` |
| prefix cache 不命中 | `kv_cache_manager.get_computed_blocks`, `block_pool.get_cached_block`, request block hashes |
| 请求被 preempt | `scheduler.schedule`, `kv_cache_manager.allocate_slots` 返回 None |
| block 泄漏/显存不释放 | `scheduler.finish_requests`, `_free_request_blocks`, `BlockPool.free_blocks` |
| disaggregated prefill KV 传输异常 | `scheduler.connector`, `distributed/kv_transfer`, worker `ensure_kv_transfer_initialized` |
| attention 读写 KV 位置异常 | `GPUModelRunner._get_slot_mappings`, block table, attention metadata |

## 16. 一句话总结

KV Cache 管理层把“请求历史 token 的 K/V”抽象成可分配、可复用、可驱逐、可转移的 block；Scheduler 决定 block 的逻辑分配，GPUModelRunner 把 block 转成 kernel 可用的 slot mapping，Attention kernel 根据这些映射读写实际 KV cache tensor。
