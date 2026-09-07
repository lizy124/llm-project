# 01. Ascend Store 总览：源码地图与一次请求的完整生命周期

源码基线：

- vLLM Ascend：`0a97c475ab120ab2e182a358f5b1306eeddc7a8f`
- vLLM：`ba07e4a48fc951300d97eb506217dd530583dea3`

`ascend_store` 不是一个简单的 KV backend 适配器。它同时包含 vLLM connector 适配、Scheduler 侧命中与状态管理、跨侧 metadata、NPU cache 注册与地址展开、异步传输线程、layerwise buffer 复用以及多种外部 backend。

当前主体源码超过 8,000 行，不能在一篇文章中同时解释清楚。本篇建立全局地图和一次请求的完整生命周期，具体实现拆到 `02` 至 `07`。

---

## 1. 一句话定位

`ascend_store` 把外部 KV 命中转换成 vLLM 可调度的 token 范围，再把 Scheduler 分配的本地 block 转换成 Worker 可执行的 key/address/size 任务，最后通过异步线程和 backend 完成 load/save。

```text
Scheduler 控制面
  AscendStoreConnector
    -> KVPoolScheduler
    -> AscendStoreCoordinator
    -> AscendConnectorMetadata

Worker 数据面
  AscendStoreConnector
    -> KVPoolWorker
    -> LayerBatchBuilder / KVTransferThread
    -> Backend
    -> NPU KV cache
```

Scheduler 侧决定"哪些 token 可以从外部恢复、需要哪些本地 block"；Worker 侧决定"这些 block 对应哪些实际地址、怎样传输以及何时完成"。

---

## 2. 源码地图

| 文件 | 主要职责 |
| --- | --- |
| `ascend_store_connector.py` | 对接 vLLM connector 生命周期，按 role 转发到 Scheduler 或 Worker |
| `pool_scheduler.py` | 外部命中查询、request tracker、分配后状态、connector metadata、结束与延迟释放 |
| `coordinator.py` | hybrid KV cache group 的命中交集、可达 mask、Eagle/SWA/Mamba 语义适配 |
| `metadata.py` | key 格式、token 到地址转换、request/transfer metadata 和 worker metadata |
| `layerwise_cache_layout.py` | layerwise 物理层编号、共享 buffer、预取和复用关系 |
| `pool_worker.py` | backend 初始化、NPU cache 注册、load/save 任务构造、同步与完成汇总 |
| `kv_transfer.py` | 批量地址构造、线程公共协议、普通和 layerwise 发送/接收线程 |
| `attention_fence.py` | attention 与复用 buffer 之间的 NPU event 顺序 |
| `backend/` | Mooncake、MemCache、YuanRong 的存储协议适配 |

几个最大的文件分别承担不同层次的复杂度，不能仅按一次调用链顺序混在一起阅读：

```text
pool_scheduler.py  请求状态与调度语义
metadata.py        描述语言与地址模型
pool_worker.py     设备资源与任务编排
kv_transfer.py     异步执行与后端 I/O
```

---

## 3. 一次请求的完整生命周期

### 3.1 第一次调度：命中查询发生在哪里

请求进入 waiting 队列后，vLLM Scheduler 的调度逻辑先通过 `KVCacheManager.get_computed_blocks()` 查询本地 prefix cache。命中的 block 会减少本轮需要真正计算的 token 数。

如果配置了 Ascend connector，Scheduler 再调用 connector 的 `get_num_new_matched_tokens()`。在 `AscendStoreConnector` 中，该调用进入 `KVPoolScheduler`，由其生成外部 key、查询命中并按 transfer granularity 对齐：

```text
Scheduler.schedule()
  -> KVCacheManager.get_computed_blocks()
  -> connector.get_num_new_matched_tokens()
  -> KVPoolScheduler 查询外部 pool
  -> 合并 local hit + external hit
```

两个语义要点：

- **外部命中只在首轮查**：vLLM Scheduler 仅对 `num_computed_tokens == 0` 的请求询问 connector；先取本地 HBM 连续 prefix（通过 `_get_local_prefix_cache_hit()` 封装，处理 connector 的 `supports_divergent_local_hybrid_hits` 差异），再问外部。
- **传入的是 block 对齐后的本地命中数**：让"更长的外部命中"接管不足一个 block 的尾巴，避免 CoW 竞争。

这里的 external hit 仍然只是"可恢复 token 范围"，并不代表 NPU block 已经装载完成。

### 3.2 为什么还要 allocate_slots()

Scheduler 必须先为命中的外部 KV 分配本地 block，Worker 才知道外部数据应该写到哪里。因此命中查询后仍会回到：

```text
KVCacheManager.allocate_slots()
  -> 生成本地 KVCacheBlocks
  -> connector.update_state_after_alloc()
```

`update_state_after_alloc()` 是把外部 token 范围和本地 block ids 连接起来的关键入口。它的准确含义是"提交分配结果"，不是"开始传输"。

### 3.3 SchedulerOutput 如何形成 Worker 的 load 计划

本轮 block 分配完成后，vLLM 调用 connector 的 `build_connector_meta()`。`KVPoolScheduler` 根据 request tracker、loading request、preempted request 和本轮 block ranges 构造 `AscendConnectorMetadata`。metadata 通常携带：

```text
- request id；
- 要 load/save 的范围；
- 本地 block ids；
- 外部 key/hash 或 layer/group 信息；
- loading、preempted、delayed-free 请求集合。
```

vLLM 把 connector metadata 放入 `SchedulerOutput.kv_connector_metadata`，随本轮调度结果交给 Executor 和 Worker。**Scheduler 不传输 KV 数据本身，只传输目标和计划。**

### 3.4 Worker 如何开始 load

模型初始化后，Worker 侧 connector 先执行 `register_kv_caches()`，建立 layer name -> KV cache tensor -> physical layer/group -> block address/stride 的映射。没有这张映射，metadata 里的 block id 和外部 key 无法转换成实际 NPU 地址。

Worker 收到 SchedulerOutput 后，`start_load_kv()` 读取 metadata、准备 load task、生成 key/address/size，提交接收线程（或同步直调 backend）。非 layerwise 路径以 request/批次组织；layerwise 路径进一步按当前 layer 组织任务。

在 layerwise 模式下，模型 runner 在每个相关 layer 的 attention 前调用 `wait_for_layer_load()`；Worker 等待当前 layer event，并检查接收线程异常。新版本中 `wait_for_layer_load()` 之后还会执行 Mamba 状态拷贝（通过 `prepare_mamba_state_copy()/finish_mamba_state_copy()`）。非 layerwise 模式可能在更粗的阶段等待 load 完成，但仍需要满足 NPU stream/event 的可见性条件。**不要把"线程任务已入队"当成"attention 已可读"。**

### 3.5 forward 期间如何 save 新 KV

- **layerwise save**：模型 runner 在 layer 边界调用 `save_kv_layer()`，把当前层的新 KV 地址交给发送线程。load/save 可以和后续 layer forward 重叠，但每层都需要自己的 event 和失败检查。
- **非 layerwise save**：由 `wait_for_save()` 统一处理，Worker 记录 NPU event 确保 source KV 不再被当前 forward 使用，再让发送线程读取地址并调用 backend.put。
- **consumer 角色**：`kv_role` 是 `kv_consumer` 且 `consumer_is_to_put` 关闭时，connector 跳过 save——consumer 只获取数据，不发布新 KV。

### 3.6 完成状态如何回到 Scheduler

`KVPoolWorker.get_finished()` 从发送/接收线程汇总 `done_recving`、`done_sending`、load error block ids 和 KV cache events，并依据 `preempted_req_ids`、`delayed_free_req_ids` 和 loading request 集合过滤结果，避免把过期或仍被引用的任务误报告。

connector 把结果包装成 vLLM `KVConnectorOutput` 后，Scheduler 侧执行：

```text
finished_recving
  -> 外部 KV 已经可在本地 block 中使用
  -> 请求可以离开 WAITING_FOR_REMOTE_KVS 或继续运行

finished_sending
  -> 外部保存完成
  -> 延迟释放的 block 才可能真正回收

invalid block ids
  -> 回退 num_computed_tokens
  -> 触发 recompute 或请求失败
```

最终仍由 vLLM Scheduler 决定请求是继续 forward、重新计算、被抢占还是结束。

### 3.7 请求结束时发生什么

请求结束并不等于所有 KV transfer 都结束。Scheduler 通过 connector 的 `request_finished()` 通知 pool：

```text
request bookkeeping 可以清理
  -> 但 pending send/load 可能仍存在
  -> delayed_free 保留相关 block 约束
  -> finished_sending 后再解除外部写入占用
  -> 最终由 KVCacheManager/BlockPool 回收本地 block
```

如果请求被抢占，metadata 中的 preempted request 集合还会影响 Worker 清理哪些完成结果，防止旧请求的异步回调污染新一轮调度。

注意三个事件不是同一件事：**传输任务完成、NPU tensor 对 attention 可见、请求 block 可以释放**，分别由线程、设备同步和 Scheduler 生命周期处理。

---

## 4. 运行模式矩阵

### 4.1 非 layerwise

普通路径以 request/chunk 为主，key 表示完整 KV 对象；同步 load 可以直接在 Worker 调用 backend，异步 load/save 则由普通发送/接收线程执行。

### 4.2 key-based layerwise

启用 `use_layerwise` 后，key 进一步包含 `layer_id`。每层有独立 load/save event，模型 forward 在 layer 边界等待或提交任务。

### 4.3 MemCache layerwise（GVA 路径）

当 `use_layerwise=True` 且 backend 为 MemCache 时，进入 GVA 路径：外部对象先分配 GVA，Worker 再批量执行 GVA 与本地 HBM 地址之间的 copy。共享物理 buffer 时还要处理上一 owner layer 的 save、attention fence 和外部 slot release。layerwise key 的生成由 backend 的 `layerwise_protocol` 提供，不再在 scheduler/worker 中内联。

### 4.4 Hybrid cache group

Full Attention、SWA、Mamba 等 group 可能具有不同 block size 和可达范围。命中不能只看单一 group；Scheduler 和 Worker 都必须保留按 group 组织的 block ids、mask、地址与 cache family。

---

## 5. 子篇导航

- [02：挂载、契约与控制面](02_connector_and_control_plane.md)——插件机制、connector 方法全景、KVPoolScheduler 的命中查询与状态管理、coordinator 的 hybrid 语义。
- [03：Metadata 与 Layout](03_metadata_and_layout.md)——Key/Tracker/ReqMeta 如何共同描述一项传输，layerwise 物理布局。
- [04：Worker Pipeline](04_worker_pipeline.md)——NPU cache 注册、block id 展开为地址、普通/layerwise/GVA/partial/TP mismatch 分流。
- [05：存储模型、传输线程与 Backend](05_transfer_backend_storage.md)——两层存储与两套地址、六类传输线程、三种 backend 能力差异。
- [06：并发、同步与配置](06_concurrency_and_config.md)——三层同步语义、attention fence、delayed_free、配置组合如何改变路径。
- [07：查询路径设计决策](07_lookup_path_design.md)——ZMQ/直连分叉、减法/合并模型、buffer 复用、两条正交的轴。

---

## 6. 阅读路线

### 6.1 第一遍：严格顺序

建议按 `02 -> 03 -> 04 -> 05`。控制面先定义"为什么传"，metadata 再定义"传什么"，Worker 解释"地址从哪里来"，最后才进入线程和 backend 的"怎样传"；随后用 `06` 收拢并发不变量，用 `07` 理解查询设计的深层原因。

### 6.2 定位问题时：按症状跳转

```text
命中 token 不对              -> 02、07
key、block range、group 不对 -> 03
地址、stride、layer 映射不对 -> 04
卡住、失败、完成未回传       -> 05、06
```

### 6.3 跟读源码的最短路线

```text
1. vllm/v1/core/sched/scheduler.py
   找 waiting 调度、allocate_slots、update_from_output

2. vllm/v1/core/kv_cache_manager.py + block_pool.py
   理解 local block 与 prefix cache 账本

3. vllm/distributed/kv_transfer/kv_connector/v1/base.py
   确认 Scheduler/Worker connector 契约

4. ascend_store/ascend_store_connector.py
   看接口如何转发到 scheduler/worker 对象

5. ascend_store/pool_scheduler.py
   跟 get_num_new_matched_tokens、update_state_after_alloc、build_connector_meta

6. ascend_store/coordinator.py + metadata.py
   理解 hybrid 命中 mask、RequestTracker 和 ReqMeta

7. ascend_store/pool_worker.py
   跟 register_kv_caches、start_load_kv、save_kv_layer、get_finished

8. ascend_store/kv_transfer.py + backend/
   最后深入线程、批量地址、backend put/get 和完成状态
```

阅读每个方法时，固定问四个问题：

```text
输入是 request、block、metadata 还是 tensor？
输出是 token 数、任务、地址还是完成状态？
执行发生在 Scheduler、Worker 还是后台线程？
失败后由谁决定重算、释放或报错？
```

---

## 7. 本文结论

```text
1. external hit 先参与 Scheduler 的 token/block 决策，之后才进入 Worker 的真实 load。
2. SchedulerOutput 传递的是 connector metadata，Worker 注册的 cache 映射负责把它落到 NPU 地址。
3. layerwise 把 load/save 拆到 layer 边界，非 layerwise 按更粗粒度批次推进。
4. Worker 只报告完成和失败，是否继续、重算、释放由 Scheduler 和 KVCacheManager 决定。
5. request finished、send finished 和 block free 是三个不同事件，必须沿异步链分别追踪。
```
