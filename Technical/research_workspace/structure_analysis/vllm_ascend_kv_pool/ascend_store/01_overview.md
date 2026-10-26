# 01. AscendStore：源码地图与请求闭环

基线与源码链接见 [00](../00_source_baseline.md)。本篇建立完整请求模型，具体机制分别见 02–09。

## 1. 主线与三个数据面

AscendStore 把“外部可恢复前缀”转换为 vLLM 的 token/block 分配，再把本地 block 转成 backend key、地址、长度和依赖。控制流与数据流在 metadata 处连接。

```text
Scheduler:
  Request -> local prefix -> external lookup -> allocation
          -> LoadSpec / RequestTracker -> ReqMeta -> SchedulerOutput

Worker:
  已注册的 KV tensor + ReqMeta
          -> key / local address / remote offset / GVA
          -> load -> forward -> save -> publish / cleanup
          -> transfer results / worker metadata / stats / KV events
```

| 路径 | 对象形态 | 执行形态 |
|---|---|---|
| 普通 object | chunk 的多段 value | `Backend.get/put` |
| MemCache GVA layerwise | 包含相关层数据的对象空间 | 每层 `GVA + offset` copy，最后 write finish |
| Mooncake block-key layerwise | block/group/stage 对象 | session + 每层 range copy + commit/end |

逐层执行不等于逐层独立 key。Mooncake 的现行专用协议与旧 KeyLayer 路径不同。

## 2. 文件职责

| 文件 | 主要问题 |
|---|---|
| `ascend_store_connector.py` | vLLM 接口如何转给 scheduler/worker；何时准备每轮状态 |
| `pool_scheduler.py` | 查多少 token、分配后如何记账、每轮发哪些 ReqMeta |
| `coordinator.py` | Full/SWA/Mamba/压缩等 group 的共同可达范围 |
| `metadata.py` | 身份、粒度、token range、block ids、快照与任务描述 |
| `layerwise_cache_layout.py` | 哪些层共用物理 buffer、谁是上一 owner |
| `pool_worker.py` | tensor 注册、地址展开、三类数据面编排、同步与结果 |
| `kv_transfer.py` | 队列、批量地址、range/GVA copy、失败、保存 fence |
| `backend/` | exists/readable、对象生命周期、SDK 返回值归一化 |
| `attention_fence.py` | 设备 attention 边界及 transfer window callbacks |
| `dspark_prefix_cache.py` | draft/auxiliary prefix 与 target 命中的联合条件 |
| `metrics.py` | 采样、聚合、Prometheus 指标 |

## 3. 一次普通请求

### 3.1 Waiting：先查本地，再查外部

vLLM `Scheduler.schedule()` 在 `request.num_computed_tokens == 0` 的分支查询本地 prefix，再调用 connector。这里包括首次进入，也可能包括抢占后计数被重置的请求，不能理解为一个 request id 一生只查一次。

本地结果经 `_get_local_prefix_cache_hit()` 和 `block_aligned_local` 处理。connector 接收的是对齐后的 token 数，不是 Worker tensor 的地址或逐 block 内容。

AscendStore 依据数据面选择：
- 非 layerwise：ZMQ `LookupKeyClient` → rank 0 Worker 的 `lookup_scheduler`。
- GVA layerwise：Scheduler 的 `_get_layerwise_hit_tokens`。
- block-key layerwise：`_get_block_key_layerwise_hit_tokens`。
- 其他 layerwise：普通逐层 key 的 exists 查询。

查询只承诺某个恢复范围；还没有把外部数据放进本地 tensor。

### 3.2 分配：外部命中仍需本地 block

Scheduler 按外部贡献的 token 范围调用 `KVCacheManager.allocate_slots()`。`update_state_after_alloc()` 把已分配的 `KVCacheBlocks` 与 `LoadSpec` 关联；异步 bulk load 请求被加入 `_loading_req_ids`。

概念上，外部新增贡献为 `max(0, hit_tokens - local_tokens)`，但还要保留末 token、Eagle 重算区和私有 state 的边界。layerwise 即使新增贡献为 0，也可能需要 load，不能删掉对应 metadata。

### 3.3 下行：metadata 是一轮计划

`build_connector_meta()` 处理 new/cached/preempted/async-load 请求，建立 `ReqMeta` 列表。`AscendConnectorMetadata` 还携带：
- `preempted_req_ids`：旧任务状态失效。
- `loading_req_ids`：异步接收生命周期。
- `released_req_ids`：本轮可能覆盖这些请求已释放的 block，Worker 先保护旧保存。

这是可序列化的计划。NPU pointer 来自 Worker 自己的 cache 注册。

### 3.4 Load：runner 的执行时机不能省略

上游 V1 mixin 根据 `SchedulerOutput.has_sync_kv_loads` 决定是否在 forward 前 `start_load_kv()`；无需前置的 load 可在 forward launch 后发起，避免占用当前计算关键路径。

因此 AscendStore layerwise **在 `bind_connector_metadata()` 内调用 `prepare_layerwise_step()`**，提前准备 tasks 和 `current_layer`。其 `start_load_kv()` 不再重置 layerwise 状态，避免 target forward 结束后、MTP 开始前错误地回到第 0 层。

非 layerwise：
- 同步 get 写入当前 forward 要用的 KV。
- 异步 get 由 recv thread 执行，请求等待远端 KV，完成后再调度。

layerwise：每层 `wait_for_layer_load()` 在使用对应 tensor 前等待其 load event，再执行需要的 Mamba align copy。

### 3.5 Save：保存增量与发布

非 layerwise 在 `wait_for_save()` 记录 NPU event，然后把请求列表交给 `add_save_batch()`。**执行体只提交后台 batch，不表示已把本轮全部 KV 写完。**

layerwise 通过逐层 hooks / attention callbacks 提交任务，普通层与 recurrent state 的保存时点不同；最后一层默认 drain 待保存任务。GVA 最终 `batch_write_finish`，block-key 最终 commit，才使对象完整可读。

保存受 `can_save`、角色、已存前缀、水位线、group mask 和 decode 配置影响，不是只在请求结束时才发生。

### 3.6 上行：完成、失败、事件各有含义

```text
get_transfer_results()
  finished_recving  异步 bulk load 可以离开等待态
  failed_recving    以 request 为单位报告恢复失败
  finished_sending  当前普通 AscendStore 返回空集合

get_block_ids_with_load_errors()
  单 block 恢复失败信息

build_connector_worker_meta()
  Mamba 特殊 sending event 完成计数

get_kv_connector_kv_cache_events()
  对外存储的 KV 事件

get_kv_connector_stats()
  性能统计
```

Scheduler 按其失败策略处理，不能把“缓存未命中”“已报告加载失败”“后台线程异常”当成相同结果。

## 4. 请求结束或抢占：保护的是覆盖时刻

当前 `request_finished()` / `request_finished_all_groups()` 都返回 `(False, None)`，允许上游正常释放 block。下一轮 metadata 包含 released ids，Worker `handle_preemptions()` 调用 `handle_released_saves()`，仅等待这些请求排队中的保存，随后才能让新 forward 覆盖相同地址。

另外，非 layerwise Mamba 非 speculative blocks 可能被 `touch_sending_mamba_blocks()` 临时增加引用；所有预期 Worker 的 event 完成后再 free。这是专门的 block 引用协议，不能与 request delayed-free 混成一条链。

## 5. 用一组不变量读代码

1. lookup 命中范围、实际 load 范围和 save skip 水位不是同一个数。
2. Scheduler block id 只有与 Worker 的 group/stride 表结合后才成为地址。
3. `size > 0`、存在、可读、copy 成功、发布成功必须分别判断。
4. request 生命周期结束不证明后台已停止读其地址。
5. 一个 layer event 不表示整个 request、整个对象或所有 rank 已完成。

继续阅读：[02 控制面](02_connector_and_control_plane.md) → [03 Metadata](03_metadata_and_layout.md) → [04 Worker](04_worker_pipeline.md) → [05 协议](05_transfer_backend_storage.md)。
