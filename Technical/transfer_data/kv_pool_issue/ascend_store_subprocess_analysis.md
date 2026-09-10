# AscendStore KV 传输子进程重构 - 详细分析

## 1. 背景与题目理解

原题要求将 AscendStore 的 worker 本地 daemon 传输线程替换为独立传输子进程加 IPC。期望结果是 key 构建、buffer 调度和后端 I/O 提交不再与 worker 进程共享 GIL。

要求覆盖当前所有传输模式：

- 非 layerwise 传输。
- 基于 key 的 layerwise 传输。
- GVA layerwise 传输。

验收要求不只是一个进程管理原型，而是要求：

- 三种模式功能等价。
- greedy 和 non-greedy 输出无回归。
- 传输子进程异常不能拖垮 worker，并且必须能重启或降级到现有线程模式。
- 改造前后性能数据必须包含吞吐、延迟、key 构建 CPU 占比和 IPC 开销。
- 测试必须覆盖子进程启停、异常恢复、跨进程 buffer 共享和 IPC 断连。

这意味着最难的部分不是可选开放问题。跨进程 buffer 共享和 NPU stream/event 协同属于验收面的一部分。

## 2. 当前架构

### 2.1 Connector 入口

`AscendStoreConnector` 是暴露给 scheduler 和 worker 角色的顶层 connector。它读取 KV transfer 配置，判断是否启用 layerwise，判断是否适用 GVA layerwise，并把 worker 侧操作委托给 `KVPoolWorker`。

代码依据：

- `AscendStoreConnector` 只在 layerwise 开启且 backend 为 memcache 时选择 GVA layerwise：`vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:89`。
- worker 调用会委托给 `KVPoolWorker`：register、start load、wait layer load、save layer、wait save、get finished 和 KV event 收集：`ascend_store_connector.py:206`。

这一层不是主要 GIL 压力来源，但它是按配置选择 transfer engine 模式的合适入口。

### 2.2 Worker 协调

`KVPoolWorker` 拥有当前传输生命周期。cache 注册期间，它会计算 KV cache base address、block length、stride、group metadata，初始化并注册后端 buffer，然后启动 transfer 线程。

代码依据：

- `register_kv_caches()` 计算 base address 和 buffer length，把它们注册到后端，然后启动 transfer 线程：`pool_worker.py:758`、`pool_worker.py:864`。
- `_start_kv_transfer_threads()` 最多创建一个 send 线程和一个 recv 线程，并根据模式选择具体线程类：`pool_worker.py:460`。

当前模式矩阵：

- 非 layerwise：
  - 可 save 的角色使用 `KVCacheStoreSendingThread`。
  - `load_async` 开启时，异步 load 使用 `KVCacheStoreRecvingThread`。
  - 同步 load 当前直接在 `KVPoolWorker.start_load_kv()` 中执行，没有和 worker 进程隔离。

- Key layerwise：
  - save 路径使用 `KVCacheStoreKeyLayerSendingThread`。
  - load 路径使用 `KVCacheStoreKeyLayerRecvingThread`。

- GVA layerwise：
  - save 路径使用 `KVCacheStoreLayerSendingThread`。
  - load 路径使用 `KVCacheStoreLayerRecvingThread`。

worker 还拥有 layerwise 状态：当前 layer index、下一层 prefetch layer、load/save task list、load/save finished event、attention-start gate 协调，以及 NPU save event。

### 2.3 传输线程

`KVTransferThread` 是 transfer worker 的公共基类。它是一个 daemon `threading.Thread`，持有进程本地状态：

- `request_queue: queue.Queue`
- `stored_requests`
- `finished_requests`
- `kv_events`
- `_fatal_error`

代码依据：

- 通用线程循环会设置 device context、标记 ready、从 `queue.Queue` 拉取请求、处理请求，并把 fatal exception 保存到 `_fatal_error`：`kv_transfer.py:311`、`kv_transfer.py:496`。
- `raise_if_failed()` 把线程中保存的异常传播给 worker：`kv_transfer.py:375`。

这可以捕获 transfer 线程中的 Python 异常，但不能把 native backend crash、backend 状态损坏或 Python CPU 压力从 worker 进程中隔离出去。

### 2.4 元数据对象

现有 metadata 层包含逻辑传输信息，但当前对象本身不能直接作为 IPC payload。

代码依据：

- `ReqMeta` 包含 request metadata，也混有 `torch.npu.Event`、numpy array、GVA array、partial GVA 状态、生成过的 key、group runtime metadata 等运行态字段：`metadata.py:857`。
- `LayerBlockRange` 引用完整 `ReqMeta`；`LayerTransferTask` 包含这些 range 的列表、可选 `SharedBlockData`、write-finish key 和 cached token processing 数据：`metadata.py:1137`、`metadata.py:1158`。
- `LayerLoadTask` 包含 `AttentionComputeStartGate`，这是进程本地同步状态：`metadata.py:1173`。

因此子进程设计必须定义字段级 IPC payload schema。直接发送 `ReqMeta`、`LayerTransferTask` 或 `LayerLoadTask` 不是有效的设计边界。

## 3. 当前热路径

### 3.1 Key 构建

key 构建发生在 Python-heavy 循环中：

- `PoolKey.to_string()` 把 model、rank、group、cache family 和 block hash 序列化成 key string。
- `LayerPoolKey.to_string()` 在 base key 上增加 layer identity。
- `ChunkedTokenDatabase` 缓存 prefix 并构造 token-key string。
- `process_token_key_strings_with_block_ids()` 直接为 block id 生成 key string。

这些路径会跨 request、group、block 和 layer 循环执行。

### 3.2 Address 和 Size 构建

非 layerwise send 和 receive 路径会在调用 backend 前构建 key、address、size 和 block id 列表。

代码依据：

- 非 layerwise save 会在 `m_store.put()` 前构建 mask、key、addr、size 和 event：`kv_transfer.py:717`。
- 异步非 layerwise load 会在 `m_store.get()` 前构建 key/address/size/block-id 列表：`kv_transfer.py:923`。
- 同步非 layerwise load 直接在 `KVPoolWorker.start_load_kv()` 中构建同样的列表：`pool_worker.py:924`。

GVA layerwise 路径会在 `batch_copy` 前构建 GVA、address 和 size 的 numpy array。

代码依据：

- GVA save 拼接 GVA/address/size array 并调用 `_batch_copy_with_limits(..., direction=0)`：`kv_transfer.py:1388`。
- GVA load 拼接 GVA/address/size array 并调用 `_batch_copy_with_limits(..., direction=1)`：`kv_transfer.py:1545`。

### 3.3 后端提交

transfer 代码调用的后端操作包括：

- `exists`
- `put`
- `get`
- `batch_alloc`
- `batch_copy`
- `batch_write_finish`
- `batch_remove_lease`

即使 native backend 操作内部释放 GIL，请求准备和提交编排仍然由 Python 驱动。

### 3.4 Layerwise 同步

layerwise 模式尤其敏感，因为 transfer 与模型执行交错进行。

代码依据：

- `save_kv_layer()` 在提交 save work 前记录 per-layer NPU event：`pool_worker.py:1736`。
- Key layerwise save 在 backend `put` 前同步 `sync_save_events[layer_id]`：`kv_transfer.py:1186`。
- GVA load 会等待前一个 save layer，同步 save NPU event，然后才复用 HBM buffer 数据：`kv_transfer.py:1553`。
- Layerwise load 可能会等待 `AttentionComputeStartGate` 后才提交 H2D work：`pool_worker.py:1684`、`kv_transfer.py:1589`。

子进程设计必须保持这些 event 语义。CPU 侧 IPC ack 不会自动等价于 NPU stream/event ordering。

## 4. 主要问题

### 4.1 Worker GIL 竞争

当前 transfer 线程和 worker 进程共享 Python interpreter 和 GIL。transfer 侧 key 构建、list 构建、numpy 编排和 backend 提交准备会延迟 worker 侧 Python 工作。

线程设计有助于阻塞 I/O 并发，但不能把 CPU-bound Python 调度移出 worker 进程。

### 4.2 失败域过大

transfer 异常会被捕获到 `_fatal_error`，但 transfer engine 仍然运行在 worker 进程内。native backend crash、deadlock 和 backend 状态损坏仍可能影响 worker。

子进程可以隔离普通 Python 异常和 child process exit。它未必能完全隔离 fatal NPU runtime 或 device-level failure。设计必须清楚说明这个边界。

### 4.3 同步机制都是进程本地的

当前实现依赖：

- `queue.Queue`
- `threading.Event`
- `torch.npu.Event`
- `AttentionComputeStartGate`
- 内存中的 request counter 和 completion set

只有 control state 可以直接用 IPC 表达。NPU event 和 stream ordering 需要明确跨进程设计，不能只用 event id 表示。

### 4.4 GVA Layerwise 存在后端特定的跨进程阻塞点

实现风险最高的是 memcache 的 GVA layerwise。

代码依据：

- GVA layerwise 在 `use_layerwise` 为 true 且 backend 为 memcache 时启用：`ascend_store_connector.py:89`。
- scheduler 明确说明 GVA allocation 被移动到 worker 侧，是因为 memcache 要求 `batch_alloc` 和 `batch_copy` 在同一进程运行；`batch_copy` 查询的 `gvaBlobTracker` 是 per-process：`pool_scheduler.py:705`。
- worker 会跟踪已经分配的 GVA，因为 `batch_alloc` 是非幂等的：`pool_worker.py:335`。

如果子进程被期望执行 GVA `batch_copy`，而 GVA allocation/tracking 仍然是进程本地状态，这会和目标架构冲突。GVA 能被认为支持 subprocess mode 之前，原型必须证明以下至少一种方案：

- child 进程可以同时执行 `batch_alloc` 和 `batch_copy`，并为该 worker 拥有完整的 per-process GVA tracker。
- backend 提供受支持的跨进程 GVA tracker state 或 descriptor export/import 能力。
- GVA mode 在 backend 支持前保留在线程 fallback，并且把该限制作为对原始验收范围的已知限制明确记录。

没有这个证明，GVA layerwise subprocess 支持不只是有风险，而是按当前代码描述对 memcache 很可能不可行。

### 4.5 职责混杂

`KVPoolWorker` 当前拥有太多执行职责：

- 启动 mode-specific transfer 线程。
- 拥有 layerwise task list 和 event。
- 直接向线程 queue 提交请求。
- 等待 queue 完成。
- 检查线程失败。
- 使用 scheduler metadata 过滤 finished request。
- 收集 KV event 和 completed worker metadata。

重构后，`KVPoolWorker` 应该保留 connector-facing coordinator 的职责，但边界不能过度抽象。它必须保留当前状态机行为。

### 4.6 Fallback 还不是一等路径

现有线程实现可以成为 fallback，但当前它还不是可互换 engine。引入 transfer engine abstraction 仍是正确第一步，但接口必须包含足够上下文来保持当前语义。

例子：

- 当前 `get_finished()` 会处理 `preempted_req_ids`、`delayed_free_req_ids` 和 `loading_req_ids`：`pool_worker.py:2082`。
- 只接收 `finished_req_ids` 的 engine method 不够。

## 5. 按模式分析

### 5.1 非 Layerwise Save

当前行为：

- worker 收集 stored request。
- worker 为 save batch 记录一个 NPU event。
- 每个 `ReqMeta` 接收 `current_event`。
- send 线程构建 store mask、skip range、key、addr 和 size。
- send 线程在 `put` 前等待 NPU event。
- finished state 和 KV event 记录在线程本地状态中。

代码依据：

- event 创建和记录发生在 `wait_for_save()`：`pool_worker.py:1757`。
- send 线程在 `m_store.put()` 前同步 `current_event`：`kv_transfer.py:888`。

子进程影响：

- child 不能接收 raw `torch.npu.Event` 对象。
- 如果 worker 在发送 IPC 前同步，正确性更简单，但 GIL/latency 收益会缩小。
- 如果 child 负责同步，设计需要受支持的跨进程 NPU event handle 或等价 stream fence。
- KV event generation 必须结构化返回给 worker。

主要风险：

- 如果只移动 key 构建，却把 event synchronization 和 address 构建留在 worker，无法解决预期的大部分 GIL 压力。

### 5.2 非 Layerwise Load

当前行为：

- 如果 `load_async` 为 false，load preparation 和 backend `get` 直接在 worker 中执行。
- 如果 `load_async` 为 true，recv 线程构建 load key、address、size，按 TP rank rotate，调用 backend `get`，并记录 failed block id。

代码依据：

- 同步 load 路径在 `KVPoolWorker.start_load_kv()` 中：`pool_worker.py:924`。
- 异步 load 路径在 `KVCacheStoreRecvingThread._handle_request()` 中：`kv_transfer.py:923`。

子进程影响：

- 设计应覆盖 async 和 synchronous load 两种行为。否则同步非 layerwise load 仍留在 worker 进程，仍然共享 GIL。
- failed-block reporting 必须通过 IPC 显式返回，并按同样的 hybrid/single-group 行为合并到 `_invalid_block_ids`。

主要风险：

- 如果 child 只报告 request-level failure，而不是精确 block id，failed block recording 和 partial miss handling 会改变 scheduler-visible 行为。

### 5.3 Key Layerwise Save

当前行为：

- worker 构建 per-layer `LayerTransferTask` list。
- save 线程构建或复用 cached token processing 结果。
- 它生成 layer key，构建 layer address 和 size，等待 per-layer NPU event，调用 backend `put`，递减 stored request counter，并标记 per-layer save finished event。

代码依据：

- Cached process-token data 存在 `LayerTransferTask.cached_process_tokens`：`metadata.py:1167`。
- save event synchronization 在 backend `put` 前发生：`kv_transfer.py:1186`。
- layer save completion 通过 `layer_save_finished_events[layer_id].set()` 表示：`kv_transfer.py:1194`。

子进程影响：

- task payload 不能嵌入 `LayerBlockRange.request` 这种完整 `ReqMeta` 对象。
- stored request counter 应由 worker 或 engine 明确拥有，所有权必须清晰。
- per-layer completion 必须通过 IPC 返回，并保持当前 final layer 阻塞行为。

主要风险：

- 过早或过晚报告 layer completion 都会改变 forward-layer 对 KV 数据的可见性。

### 5.4 Key Layerwise Load

当前行为：

- recv 线程在需要时等待前一个 save layer。
- 它可能等待 attention compute start。
- 它构建 per-layer key 和 address/size list。
- 它调用 backend `get` 并标记 layer load completion。

代码依据：

- save-layer wait 使用 `layer_save_finished_events`，等待后会 clear：`kv_transfer.py:1236`。
- attention-start gate wait 发生在 load I/O 前：`kv_transfer.py:1250`。
- layer load completion 通过 `layer_load_finished_events[layer_id].set()` 表示：`kv_transfer.py:1300`。

子进程影响：

- worker-facing `wait_for_layer_load()` 当前按 `current_layer` 前进，而不是按 connector 传入的 `layer_name` 参数。
- IPC 必须建模 `current_layer`、prefetch、wait-for-save dependency 和 attention-start gating。

主要风险：

- 如果 child 观察不到相同 gating 行为，forward execution 可能使用缺失或陈旧 KV 数据。

### 5.5 GVA Layerwise Save

当前行为：

- worker 准备 GVA allocation 和 shared block data。
- save 线程构建 GVA/address/size array。
- 它等待 per-layer NPU save event。
- 它调用 `_batch_copy_with_limits(..., direction=0)`。
- 它为最终 publication key 调用 `batch_write_finish`。

代码依据：

- GVA save 使用 direction 0 的 `batch_copy`，随后执行 `batch_write_finish`：`kv_transfer.py:1417`。
- worker 因为 `batch_alloc` 非幂等而跟踪 allocated GVA：`pool_worker.py:335`。

子进程影响：

- 对 memcache 而言，GVA allocation 和 GVA copy 很可能必须在同一进程内。
- 如果 child 拥有 GVA allocation，worker 必须停止拥有 `_allocated_gvas`，或只能安全地镜像它。
- crash 后不能盲目 replay `batch_write_finish`。

主要风险：

- 后端 per-process GVA state 使该模式成为风险最高且可能被阻塞的模式。

### 5.6 GVA Layerwise Load

当前行为：

- recv 线程等待 save dependency。
- 对复用 layer，它会同步 save NPU event 后才复用本地 HBM buffer。
- 它可能等待 attention compute start。
- 它构建 GVA/address/size array。
- 它提交 `_batch_copy_with_limits(..., direction=1)`。
- 在最终 layer，它用 `batch_remove_lease` 释放 lease。

代码依据：

- save dependency wait 和 NPU event synchronization 在 load copy 前发生：`kv_transfer.py:1553`。
- final-layer lease release 在成功 final-layer load 后发生：`kv_transfer.py:1633`。

子进程影响：

- lease ownership 必须明确。
- load 过程中 crash 可能泄漏 lease。
- child 在 batch copy 后、lease release 前死亡，需要 cleanup/reconcile policy。

主要风险：

- cleanup 错误会泄漏 lease；event timing 错误会过早复用 HBM buffer。

## 6. 跨进程数据分类

### 6.1 直接 IPC Payload

小型 control data 可以直接发送：

- Request id。
- Layer id 和 physical layer id。
- Group id 和 group 内 layer index。
- Boolean flag。
- Mode name。
- Backend configuration。
- Error code。
- Completion status。
- Finished request id。

### 6.2 字段级序列化 Metadata

中等规模 metadata 可以序列化，但必须有边界并测量：

- Request id。
- Token length：`token_len_chunk`、`save_start_token`、`save_end_token`、`target_token_len`。
- `can_save`、`is_last_chunk`。
- load 所需的 `LoadSpec` 字段。
- Block hash。
- Block ids by group。
- KV cache group id 和 cache family name。
- Partial block index。
- 需要时的 GVA key 和 lease key。

设计不应发送包含 runtime-only 字段的完整 Python 对象。

### 6.3 Shared Memory 或 Backend-Owned Handle

大型和重复数据应尽量避免通过 IPC 复制：

- KV cache tensor。
- Backend-registered memory region。
- 大型 block-id numpy array。
- GVA array。
- 可复用的 address/size array。

但是 shared memory 本身不足以保证 device/backend memory 正确性。backend 必须支持具体的跨进程 register/import 语义。

### 6.4 不能 Raw 跨进程的运行态字段

以下字段不能原样发送：

- `threading.Event`。
- `queue.Queue`。
- `torch.npu.Event` reference。
- `AttentionComputeStartGate` instance。
- Backend object instance。
- 完整 `ReqMeta` object reference。
- 带 embedded request object 的完整 `LayerBlockRange`。
- Worker-owned mutable task list。

它们必须用 task id、event/fence descriptor、completion message 或 child-local reconstructed state 表示。

## 7. 失败与恢复分析

### 7.1 可恢复失败

可恢复失败包括：

- 命令开始前的 IPC command timeout。
- 非非幂等 backend 操作期间的 child Python exception。
- child idle 时退出。
- startup failure 且允许 thread fallback。
- 任何 transfer operation 前检测到 child registration unsupported。

期望行为：

- 标记 engine unhealthy。
- 保守地 fail 或 cancel 当前 transfer task。
- 只在 command state 安全时 restart。
- 如果支持，重新 register buffer。
- 当 restart 不安全或耗尽时 downgrade 到 thread mode。

### 7.2 Fatal 或不可 Replay 失败

fatal/non-replayable failure 包括：

- backend 无法在 subprocess 中初始化。
- backend 无法在 child 中 import/register worker KV buffer。
- GVA tracker state 是进程本地且无法安全移动到 child。
- child 在 `batch_copy`、`batch_write_finish` 或 lease release 期间死亡。
- 重复 crash 超过 retry limit。
- IPC protocol version mismatch。
- NPU event/fence 无法跨进程正确表示。

期望行为：

- 为当前 worker 禁用 subprocess mode。
- 正确性仍可保障时，未来请求 fallback 到 thread mode。
- fail 当前 transfer，而不是盲目 replay 非幂等 work。
- 输出清晰 warning，包含 mode、command、request id、layer id 和 fallback decision。

### 7.3 Worker 存活性

worker 不应依赖 child process liveness 来维持 Python 进程存活。它应把 transfer 视为具有以下状态的 engine：

- Running。
- Restarting。
- Downgraded。
- Disabled for current mode。

但不能承诺每一种 native/device-level failure 下 worker 都能存活。设计应区分 child-process failure isolation 和 NPU runtime fatal error isolation。

## 8. 精度与正确性风险

如果 KV load/save ordering 改变、failed load 处理不同，或 layerwise completion 过早上报，greedy 和 non-greedy 输出精度都可能变化。

正确性敏感点：

- save/load I/O 前的 NPU event synchronization。
- Per-layer load completion timing。
- `consumer_is_to_put` 或 GVA layer reuse 场景下的 save-before-load ordering。
- Attention-start gating。
- TP-rank key rotation。
- Failed block recording。
- Final-layer lease release。
- Key string format 和 layer id inclusion。
- Cached prefix/token behavior。
- GVA allocation 和 write-finish publication ordering。

子进程重构不能改变 key format、block selection、load mask、store mask、backend read/write order 或 scheduler-visible completion behavior，除非现有实现本身已经允许该异步行为。

## 9. 现有测试覆盖缺口

现有测试不足以满足原始验收标准。

缺失覆盖：

- 子进程启动和 graceful shutdown。
- Child process crash detection。
- Child exit 后 restart。
- Downgrade 到 thread mode。
- IPC disconnect handling。
- Cross-process buffer registration 成功和失败。
- Cross-process NPU event/fence behavior，或 unsupported 时 fallback。
- Per-mode request equivalence。
- Layerwise event ordering。
- GVA allocation/copy/write-finish/lease semantics。
- Failed-block propagation。

测试应尽可能 mock NPU 和 backend 行为，但 GVA/memcache cross-process behavior 至少需要定向 integration 或硬件原型，因为 blocker 来自 backend/runtime 语义。

## 10. 性能分析要求

对比必须隔离 subprocess mode 是否真正改善目标瓶颈。

必需指标：

- End-to-end throughput。
- P50/P95/P99 latency。
- Worker process Python CPU share。
- Transfer child Python CPU share。
- Key-building CPU share。
- Address/size construction time。
- IPC round-trip latency。
- IPC payload size。
- Serialization/deserialization time。
- Backend put/get/batch-copy time。
- Restart/fallback overhead。

必需对比：

- 现有 in-process thread mode。
- 相同 workload 下的 subprocess mode。
- Crash/restart 场景下的 subprocess mode。
- Fallback 场景下的 subprocess mode。
- GVA subprocess feasibility result，或 unsupported 时 fallback cost 测量。

有效原型应在 key construction、address/size construction、IPC serialization/deserialization 和 backend submission 周围加 counter，而不只是测总 request latency。

## 11. 推荐重构边界

最干净的边界仍然是 worker-facing transfer engine interface：

- `ThreadTransferEngine`：包装现有实现，并保留为 reference behavior。
- `SubprocessTransferEngine`：管理 child process、IPC、command state 和 fallback。

但接口必须匹配现有语义，而不是简化过的理想 API。它必须包含：

- 带 KV cache memory region 和 backend capability result 的 registration context。
- 不含进程本地字段的 request metadata payload。
- Layer current-index 和 dependency information。
- 当前由 `AscendConnectorMetadata` 提供的 finished-request filtering input。
- Failed-block reporting。
- KV event 和 completed-event reporting。
- Health/fallback status。

## 12. 实现前推荐重新分析

实现全部三种模式前，应先做两个 feasibility spike。

### 12.1 跨进程 Buffer 与 Backend Registration Spike

目标：

- 证明每个 backend 是否能在 child process 中 register/import worker KV cache memory。
- 对 memcache/GVA，证明 `batch_alloc` 和 `batch_copy` 是否都能在 child 中针对 worker-owned KV buffer 正确运行。

决策结果：

- Supported：继续为该 backend/mode 实现 subprocess executor。
- Unsupported but safely detectable：该 backend/mode 强制 thread fallback。
- Ambiguous：默认保持 subprocess disabled。

### 12.2 NPU Event/Fence Spike

目标：

- 证明 `torch.npu.Event` 和 layerwise event 语义的安全跨进程替代方案。

决策结果：

- Child 可以等待 cross-process event/fence：保持当前 async 行为。
- Worker 必须在 IPC 前同步：正确性可行，但性能收益缩小，需要测量。
- 没有安全表示：需要该 event 的模式强制 thread fallback。

## 13. 结论

子进程重构方向与 GIL 和隔离目标一致，但正确方案不能只处理进程启动和 IPC。

当前代码给出的关键设计约束是：

- Metadata object 不能直接作为 IPC payload。
- NPU event 和 attention-gate 语义对正确性很关键。
- GVA layerwise with memcache 在当前代码中有明确的 per-process tracker 约束。
- Restart/fallback 必须尊重非幂等 backend operation。
- Transfer engine interface 必须保持当前 worker 状态机行为。

最重要的设计规则仍然是：worker 提交高层 transfer intent，transfer subprocess 拥有 key construction、address/size scheduling 和 backend I/O submission。但对于 GVA/memcache 和 NPU stream/event ordering，这条规则只有在 backend 和 runtime feasibility 被证明后才成立。在此之前，这些模式必须由 capability detection 控制，并 fallback 到现有 thread engine。