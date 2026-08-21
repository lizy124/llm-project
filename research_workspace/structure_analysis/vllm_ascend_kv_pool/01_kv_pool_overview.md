# 01. kv_pool 在 vLLM Ascend 中处于什么位置？

源码基线：

- vLLM Ascend：`d85e6714a09bef4d9de6b8c05e9425183d46ba23`
- vLLM：`58d3918e3ea0a544ffedadad2ba84559e9c51d8f`
- 对应关系：`vllm-ascend/.github/vllm-main-verified.commit`

源码位置：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_connector/`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/worker/`

本文关注的问题是：`kv_pool` 到底属于 vLLM 的哪一层？它和 Scheduler、Worker、KV Connector、GPU/NPU KV cache 以及外部 backend 分别是什么关系？如果把这些边界弄清楚，后续阅读 `ascend_store` 的 load/save 才不会把“block 管理”“tensor 搬运”和“外部存储”混成一件事。

---

## 0. 本文要回答的问题

```text
1. kv_pool 是 vLLM KV cache 管理器，还是一个外部 KV transfer 实现？
2. Scheduler 侧的 connector 和 Worker 侧的 connector 分别负责什么？
3. ascend_store、kv_offload、recompute_cpu_offload、simple_cpu_offload 有什么关系？
4. kv_pool 是否直接管理 GPU/NPU KV cache block？
5. backend、传输线程和 attention fence 分别处在哪一层？
6. 从 vLLM 的请求调度到 kv_pool，中间经过哪些关键接口？
```

本文只建立角色和边界，不展开一次具体请求的完整 save/load 时序；完整时序放在后续专题中。

---

## 1. 一句话回答

`kv_pool` 是 vLLM KV Connector 在 Ascend 后端上的一组具体实现：它接收 vLLM 已经决定好的 KV transfer 任务，负责查询、保存、加载、CPU/NPU 搬运和异步完成通知；它不是 Scheduler 的 KV block allocator，也不是 attention backend 本身。

可以先记住下面这条边界：

```text
Scheduler 决定“这次请求需要多少 token、哪些本地 block 可用”；
KV Connector 把这个决定转换成 load/save metadata；
kv_pool 决定“数据如何查找、传输、落在哪里、何时完成”；
Worker / ModelRunner 把结果写入或读出真正的 NPU KV cache tensor。
```

---

## 2. kv_pool 在整体架构中的位置

vLLM V1 的调用边界可以先压缩成下面这条链：

```text
Request
  -> Scheduler
  -> KVCacheManager / BlockPool
  -> KV Connector 的 scheduler-side 接口
  -> SchedulerOutput.kv_connector_metadata
  -> Executor / Worker / ModelRunner
  -> KV Connector 的 worker-side 接口
  -> NPU KV cache tensor
```

`kv_pool` 插入的是 KV Connector 的实现位置：

```text
vLLM Scheduler
  -> KVConnectorBase_V1 接口
  -> AscendStoreConnector / OffloadConnector /
     RecomputeCPUOffloadConnector
  -> scheduler、worker、backend 或本地内存
```

因此，`kv_pool` 不是一个单独的“池对象”，而是围绕 KV Connector 接口组织的一组实现。不同实现可以拥有完全不同的保存介质，但都要遵守 vLLM connector 的 scheduler-side / worker-side 生命周期。

---

## 3. vLLM 原生层负责什么

### 3.1 Scheduler 和 KVCacheManager

vLLM 的 Scheduler 负责本轮选择哪些请求、推进多少 token、是否抢占以及是否允许请求继续执行。`KVCacheManager` 和 `BlockPool` 负责本地 KV block 的账本：

```text
- 查询本地 prefix cache 命中；
- 为请求分配或回收本地 KV block；
- 维护 request 到 block 的映射；
- 把 block ids 放入 SchedulerOutput；
- 处理 block 的引用计数、缓存和释放。
```

这些代码位于 vLLM 的 `v1/core` 和 `v1/core/sched`，不在 Ascend 的 `kv_pool` 目录中。

### 3.2 KV Connector 接口

vLLM 在 `distributed/kv_transfer/kv_connector/v1/` 定义 connector 的生命周期接口。Scheduler 侧通常关心：

```text
get_num_new_matched_tokens()
update_state_after_alloc()
build_connector_meta()
request_finished()
```

Worker 侧通常关心：

```text
register_kv_caches()
bind_connector_metadata()
start_load_kv()
wait_for_layer_load()
save_kv_layer()
wait_for_save()
get_finished()
```

这些接口是 vLLM 和 Ascend 实现之间的契约。`kv_pool` 的各个 connector 不应改变 Scheduler 的 block 分配语义，而是利用这些接口登记外部命中、生成传输 metadata，并报告完成或失败。

---

## 4. Ascend kv_pool 的五条实现路径

### 4.1 `ascend_store`

位置：`kv_pool/ascend_store/`

这是最完整的外部 KV Store 路径。它包含：

```text
AscendStoreConnector       vLLM connector 适配层
Coordinator                多任务/多 worker 协调
PoolScheduler              任务排队和调度
PoolWorker                 实际准备和执行传输
KVTransfer                 传输线程、请求状态和 layer/block 批处理
metadata/layout            key、shape、层布局和传输描述
backend/                   memcache、mooncake、yuanrong 等后端
```

它可以向外部 backend 查询 key、申请存储、写入或读回 KV，再由 Worker 侧写入本地 NPU KV cache。

### 4.2 `kv_offload`

位置：`kv_pool/kv_offload/`

这是本地内存层级之间的搬运路径。`native` 和 `simple` 两套实现主要面对 CPU/NPU buffer、copy operation、worker 调用和同步，不以外部 KV backend 为核心。

它解决的是：

```text
NPU KV tensor <-> CPU buffer
```

而不是：

```text
key <-> 外部 KV Store object
```

### 4.3 `recompute_cpu_offload`

位置：`kv_pool/recompute_cpu_offload/`

这条路径不把完整 KV 当作唯一恢复材料，而是保存重计算所需的 metadata，并在 Worker 侧通过 manager/worker 组织恢复。它的“命中”不等于可以直接把 KV tensor 读回来，后续可能仍需计算。

### 4.4 `simple_cpu_offload`

当前代码结构中，simple CPU offload 的 connector 实现在 `kv_offload/simple/` 下，由 `SimpleCPUOffloadConnector` 和相应 worker/copy backend 组成。它是本地 CPU offload 的简化实现，不应和 `ascend_store` 的外部 backend 混用概念。

---

## 5. kv_pool 是否直接管理 NPU KV block？

答案是：通常不直接管理 Scheduler 意义上的 block 生命周期，但会使用 block 对应的地址、长度和 tensor view 来完成传输。

需要区分两种 block：

```text
vLLM KV block
  - 由 KVCacheManager / BlockPool 管理
  - 有 block id、引用计数、prefix cache 语义
  - Scheduler 通过它决定本地 cache 布局

kv_pool transfer block
  - 由 metadata、key、地址和长度描述
  - 用于 backend put/get 或本地 copy
  - 关注传输范围、layer layout 和 buffer 生命周期
```

两者在 connector metadata 和 Worker 的 tensor view 处汇合，但所有权不同。`ascend_store` 中的 `LayerBatchBuilder`、`KVTransferThread` 和 backend `put/get` 负责“怎么传”；vLLM `BlockPool` 负责“这个 block 在 Scheduler 账本里是否可用”。

---

## 6. backend、传输线程和 attention fence 的层次

### 6.1 backend

`ascend_store/backend/base.py` 定义 backend 抽象，主要提供：

```text
exists / batch_is_exist
batch_get_key_info / batch_alloc
batch_add_lease / batch_remove_lease
batch_write_finish
put / get
```

它只应该关注外部存储协议和地址/长度，不应该决定 vLLM request 的调度状态。

### 6.2 传输线程

`ascend_store/kv_transfer.py` 中的传输线程把请求拆成 layer/block 批次，准备地址数组，调用 backend 或设备拷贝，并记录 finished、failed、stored 等状态。发送和接收方向有各自的线程实现。

它位于 connector 与 backend 之间，是异步数据面的核心；但它不会替代 Scheduler 维护 request 的运行状态。

### 6.3 attention fence

`ascend_store/attention_fence.py` 中的 `AttentionComputeStartGate` 负责在 attention 开始使用数据前建立同步条件。它解决的是：异步 load 已经提交，但 NPU tensor 是否真的可读。

所以它属于“设备数据可见性”边界，而不是外部 backend 的一致性协议。

---

## 7. 最小主链

从 vLLM 请求视角，可以先记住这条最小路径：

```text
Scheduler
  -> connector.get_num_new_matched_tokens()
  -> KVCacheManager.allocate_slots()
  -> connector.update_state_after_alloc()
  -> connector.build_connector_meta()
  -> SchedulerOutput

Worker / ModelRunner
  -> bind_connector_metadata()
  -> start_load_kv() / wait_for_layer_load()
  -> forward 使用本地 NPU KV cache
  -> save_kv_layer() / wait_for_save()
  -> get_finished()

Scheduler
  -> update_connector_output()
  -> 推进请求、处理失败或释放资源
```

在这条链中，`kv_pool` 具体实现的是 connector 两侧的行为和中间的数据面；请求调度、block 账本和模型 forward 仍由 vLLM 原生层负责。

---

## 8. 本文结论

```text
1. kv_pool 是 Ascend 对 vLLM KV Connector 的实现集合，不是新的 Scheduler。
2. KVCacheManager / BlockPool 管理本地 block 账本，kv_pool 管理传输和外部/CPU 存储。
3. connector 把 Scheduler 的决定转换成 metadata，再由 Worker 执行真正的 tensor load/save。
4. ascend_store 是外部 KV Store 路径，kv_offload 是本地 CPU/NPU 路径，recompute_cpu_offload 是不保存完整 KV 的恢复路径。
5. backend、传输线程和 attention fence 分别对应外部存储、异步数据面和设备可见性同步。
```

后续阅读建议：先看 `02_ascend_store_core.md`，沿 `AscendStoreConnector -> Coordinator/PoolScheduler -> PoolWorker -> KVTransfer/backend` 继续展开；再分别阅读 offload 和重计算路径。
