# 02. ascend_store 源码总览

源码基线：

- vLLM Ascend：`d85e6714a09bef4d9de6b8c05e9425183d46ba23`
- vLLM：`58d3918e3ea0a544ffedadad2ba84559e9c51d8f`

`ascend_store` 不是一个简单的 KV backend 适配器。它同时包含 vLLM connector 适配、Scheduler 侧命中与状态管理、跨侧 metadata、NPU cache 注册与地址展开、异步传输线程、layerwise buffer 复用以及多种外部 backend。

当前主体源码超过 8,000 行，不能在一篇文章中同时解释清楚。本篇只建立全局地图，具体实现拆到 `02_1` 至 `02_4`。

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

Scheduler 侧决定“哪些 token 可以从外部恢复、需要哪些本地 block”；Worker 侧决定“这些 block 对应哪些实际地址、怎样传输以及何时完成”。

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

## 3. 一次请求的最小主链

### 3.1 Scheduler 阶段

```text
get_num_new_matched_tokens()
  -> 生成 PoolKey / GVA key
  -> backend exists / batch_get_key_info
  -> coordinator 合并各 cache group 的可达命中
  -> 按 cache_transfer_granularity 向下对齐

KVCacheManager.allocate_slots()
  -> update_state_after_alloc()
  -> RequestTracker 保存本地 block 与外部命中关系

build_connector_meta()
  -> ReqMeta
  -> AscendConnectorMetadata
  -> SchedulerOutput
```

### 3.2 Worker 阶段

```text
register_kv_caches()
  -> layer/group/tensor
  -> base address/block length/stride
  -> backend register_buffer
  -> 启动对应传输线程

start_load_kv() / save_kv_layer() / wait_for_save()
  -> ReqMeta
  -> LayerTransferTask / LayerLoadTask
  -> key/address/size 或 GVA/address/size
  -> backend get/put 或 batch_copy
```

### 3.3 完成回传

```text
transfer thread finished/failed
  -> KVPoolWorker.get_finished()
  -> KVConnectorOutput
  -> KVPoolScheduler.update_connector_output()
  -> 请求继续、回退、延迟释放或清理
```

注意三个事件不是同一件事：传输任务完成、NPU tensor 对 attention 可见、请求 block 可以释放，分别由线程、设备同步和 Scheduler 生命周期处理。

---

## 4. 运行模式矩阵

### 4.1 非 layerwise

普通路径以 request/chunk 为主，key 表示完整 KV 对象；同步 load 可以直接在 Worker 调用 backend，异步 load/save 则由普通发送/接收线程执行。

### 4.2 key-based layerwise

启用 `use_layerwise` 后，key 进一步包含 `layer_id`。每层有独立 load/save event，模型 forward 在 layer 边界等待或提交任务。

### 4.3 MemCache GVA layerwise

当 `use_layerwise=True` 且 backend 为 MemCache 时，会进入 GVA 路径：外部对象先分配 GVA，Worker 再批量执行 GVA 与本地 HBM 地址之间的 copy。共享物理 buffer 时还要处理上一 owner layer 的 save、attention fence 和外部 slot release。

### 4.4 Hybrid cache group

Full Attention、SWA、Mamba 等 group 可能具有不同 block size 和可达范围。命中不能只看单一 group；Scheduler 和 Worker 都必须保留按 group 组织的 block ids、mask、地址与 cache family。

---

## 5. 子篇导航

### [02_1：Connector、Scheduler 与 Coordinator](02_1_ascend_store_control_plane.md)

回答控制面如何把外部命中变成可调度 token，以及新请求、运行中请求、抢占、异步 load 和延迟释放如何进入状态机。

### [02_2：Key、Metadata 与 Cache Layout](02_2_ascend_store_metadata_and_layout.md)

回答 `PoolKey`、`ChunkedTokenDatabase`、`RequestTracker`、`ReqMeta` 和 layerwise/hybrid layout 如何共同描述一项传输。

### [02_3：KVPoolWorker 如何生成设备任务](02_3_ascend_store_worker_pipeline.md)

回答真实 NPU cache 如何注册，block id 如何展开成地址，以及普通、layerwise、GVA、partial block、TP mismatch 路径如何分流。

### [02_4：传输线程与 Backend](02_4_ascend_store_transfer_and_backend.md)

回答六类传输线程如何组织任务，backend 接口如何映射到 Mooncake、MemCache、YuanRong，以及同步、异常和完成状态如何闭环。

---

## 6. 推荐阅读顺序

第一次阅读建议严格按 `02_1 -> 02_2 -> 02_3 -> 02_4`。控制面先定义“为什么传”，metadata 再定义“传什么”，Worker 解释“地址从哪里来”，最后才进入线程和 backend 的“怎样传”。

如果正在定位问题，可以按症状跳转：

```text
命中 token 不对              -> 02_1
key、block range、group 不对 -> 02_2
地址、stride、layer 映射不对 -> 02_3
卡住、失败、完成未回传       -> 02_4 和 05
```
