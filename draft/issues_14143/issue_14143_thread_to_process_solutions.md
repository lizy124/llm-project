# Issue #14143：AscendStore 传输线程改为子进程的解决方案分析

> 本文基于 `issue_14143_thread_to_process_analysis.md`，结合当前 `vllm-ascend` 和 `vllm` 工作树中的实现边界，给出三种可实施路线。重点不是把 `threading.Thread` 机械替换为 `multiprocessing.Process`，而是分别解决 GIL 竞争、NPU 内存访问、layerwise 同步、故障传播和性能验证问题。

## 1. 结论先行

Issue #14143 指出的结构性问题成立：`KVTransferThread` 在模型 worker 进程内执行了 key/token/block 元数据计算，和模型主线程共享 Python 解释器；线程异常还可能让后续任务、`wait_for_save()` 或队列等待处于不完整状态。

但现有代码不能直接改成子进程，原因也很明确：

- `KVPoolWorker.register_kv_caches()` 当前通过 `data_ptr()` 注册 NPU buffer，裸地址不能跨进程复用。
- `AttentionComputeStartGate` 依赖 `torch.npu.Event`、stream 和 device 上下文，不能由普通 `multiprocessing.Event` 等价替换。
- `LayerLoadTask`、request、`queue.Queue`、完成集合和 backend 对象都是进程内对象，不是稳定的 IPC 协议。
- 子进程退出后，主进程必须主动唤醒 `wait_for_layer_load()` / `wait_for_save()` 的等待者，并决定重放、重算、降级还是终止 engine。

因此建议采用下面的决策顺序：

1. **所有路线都先做方案一中的可靠性和 profiling 基础**，否则没有可比较的基线。
2. **中期优先实施方案二（混合进程）**：把 CPU 元数据规划和 lookup 放到子进程，NPU buffer、NPU event 和真正的 MemCache 读写仍由主进程拥有。这条路线不依赖未经确认的 NPU IPC API，是当前最稳妥的进程化方案。
3. **只有在 Ascend runtime、torch_npu、Mooncake/MemCache 明确支持跨进程 buffer/event handle 后，才实施方案三（完整传输进程）**。
4. 如果 profiling 表明 Python 准备工作占比很低，方案一可能已经足够，不能为了“进程化”而引入 IPC 复制和同步开销。

## 2. 共同设计约束

### 2.1 必须拆开控制面和数据面

控制面传递 request、layer、block、key 和状态；数据面传递大批量 key/address 描述或 NPU buffer。不能把完整的 `LayerLoadTask`、request 对象、tensor 或 backend 实例直接 pickle 到子进程。

建议定义稳定的、版本化的命令和结果结构。实际传输可以使用 `multiprocessing.connection`、Unix domain socket、Linux pipe 或 Windows named pipe；大量数组应放入共享内存或固定大小 ring buffer，消息中只传 offset/length。

```text
TransferCommand {
  protocol_version: int,
  command_id: uint64,          # 每次提交唯一
  request_id: string,           # 对应 vLLM 请求或批次
  operation: load | save | lookup | cancel | shutdown,
  mode: normal | key_layerwise | gva_layerwise,
  layer_id: int | null,
  group_id: int | null,
  block_ranges: [(start, count)],
  key_ref: SharedArrayRef | InlineKeyIds,
  buffer_ref: BufferRef | null,
  dependency: DependencyToken | null,
  deadline_ns: int,
  attempt: int,
  idempotency_key: string,
}
```

返回消息至少应包含：

```text
TransferResult {
  protocol_version: int,
  command_id: uint64,
  request_id: string,
  status: SUCCESS | PARTIAL | RETRYABLE_ERROR | FATAL_ERROR | CANCELLED,
  completed_layers: [int],
  completed_blocks: [int],
  failed_blocks: [int],
  backend_error_code: int | null,
  retryable: bool,
  error_message: string | null,
  worker_generation: uint64,
}
```

`command_id` 用于匹配响应，`idempotency_key` 用于重试去重，`worker_generation` 用于丢弃旧进程重启前迟到的响应。结果不能只返回一个布尔值，否则无法区分“没有命中”“部分 block 失败”“子进程已经死亡”和“后端不可重试错误”。

### 2.2 用显式状态机替代隐式队列状态

每个 load/save 命令都应遵循如下状态：

```text
NEW
  -> SUBMITTED
  -> RUNNING
  -> COMMITTING
  -> SUCCEEDED

RUNNING/COMMITTING -> RETRYABLE_ERROR -> REPLAYING -> RUNNING
NEW/SUBMITTED/RUNNING -> CANCELLED
任意状态 -> LOST（worker 心跳消失或进程退出）
LOST -> REPLAYING / RECOMPUTE / ENGINE_ERROR
```

状态应由 supervisor 持有，而不是只存在传输线程的 `finished_requests` 集合中。这样即使 worker 崩溃，主进程仍能遍历未完成命令并唤醒等待者。save 必须先写临时 key 或带 generation 的 key，只有全部 block 成功后才提交可见标记，避免半写入被后续 lookup 当成完整命中。

### 2.3 所有等待都必须有超时和唤醒路径

`queue.join()`、layer event 和 `wait_for_save()` 都不能无限等待。建议统一使用：

- 每个命令的 deadline 和全局 watchdog；
- `Condition.wait(timeout)` 或带超时的 future；
- worker death、cancel、timeout 时广播唤醒；
- 主线程看到 `LOST` 后转换为 typed error，而不是继续等待本地 event；
- 退出时先停止接收新任务，再 cancel/replay 未完成任务，最后 drain 并关闭 IPC。

### 2.4 明确 NPU buffer 的所有权

三种路线的内存所有权不同，必须在设计评审中写死：

| 路线 | NPU KV cache 所有者 | 子进程是否接触 NPU buffer | 是否需要厂商 IPC handle |
|---|---|---|---|
| 方案一 | 主进程 | 否，仍由主进程线程读写 | 否 |
| 方案二 | 主进程 | 否，子进程只处理纯 CPU 描述 | 否 |
| 方案三 | 主进程创建，子进程导入并持有映射 | 是 | 是，必须验证 |

任何方案都不得通过 IPC 传递 `data_ptr()` 整数并假定子进程可以访问。Linux 和 Windows 均优先使用显式 `spawn`；子进程内重新初始化 device、通信库和 backend，不继承已运行的 NPU runtime、ZMQ context、锁或线程。

## 3. 方案一：保留线程，增加 Supervisor 并原生化 CPU 元数据

### 3.1 适用场景和目标

这是最低风险路线。它不把传输器搬到独立进程，而是先修复当前线程模型的故障语义，再将最耗时的 Python 元数据工作批量化、缓存化或下沉到原生实现。适用于：

- profiling 证明 `m_store.get()` / `put()` 和 NPU wait 占主要耗时，GIL 占比不高；
- 当前环境没有可靠的 NPU 跨进程 buffer API；
- 首要目标是消除永久等待、异常不可见和线程状态残留；
- 希望先获得一个可以作为方案二、三 A/B 基线的稳定版本。

它不能提供完整的 Python 崩溃隔离，但可以把结构性风险降到可观察、可超时、可恢复的范围。

### 3.2 建议架构

```text
KVPoolWorker
  -> TransferSupervisor（主进程内）
      -> bounded command queue
      -> KVTransferThread send/recv
      -> heartbeat / deadline / completion registry
      -> typed error / cancel / fallback
```

`TransferSupervisor` 是新的生命周期边界，不直接承担 key 计算。它负责：

1. 创建和停止 send/recv 线程；
2. 为每个命令分配 `command_id`、deadline 和状态；
3. 监视线程 ready、heartbeat、fatal error 和队列长度；
4. 统一广播失败，唤醒所有 `wait_for_layer_load()` / `wait_for_save()`；
5. 在关闭时取消新任务、标记未完成任务、清理本地状态；
6. 根据配置选择 `thread`、`disable` 或请求失败策略。

### 3.3 代码改动

建议新增以下模块，名称可以按仓库约定调整：

- `ascend_store/transfer_protocol.py`：命令、结果、错误码和状态机；
- `ascend_store/transfer_supervisor.py`：线程生命周期、heartbeat、deadline 和等待者唤醒；
- `ascend_store/transfer_errors.py`：`TransferTimeoutError`、`TransferWorkerError`、`PartialTransferError` 等 typed exception。

修改现有模块：

- `kv_transfer.py`：在 `KVTransferThread.run()` 外层增加统一 `try/finally`，保证当前任务最终执行 `task_done()`；异常时向 supervisor 报告，而不是只设置 `_fatal_error` 后返回；
- `pool_worker.py`：`_start_kv_transfer_threads()` 返回 supervisor；`start_load_kv()`、`wait_for_layer_load()`、`save_kv_layer()`、`wait_for_save()` 通过 supervisor 查询状态；
- `metadata.py`：把可复用的 token/block 解析结果变为不可变的 `TransferPlan`，避免同一请求在发送和接收路径重复计算；
- `memcache_backend.py`：增加批量 descriptor 接口、明确 backend error code 和 partial write 状态；
- `ascend_store_connector.py`：将 supervisor 错误转换成 vLLM connector 可识别的 load/save failure。

### 3.4 关键实现细节

#### 统一任务完成保证

线程从队列取得任务后，应使用如下语义：

```text
task = queue.get()
try:
    supervisor.mark_running(task)
    result = handle(task)
    supervisor.complete(task, result)
except Cancelled:
    supervisor.cancel(task)
except Exception as exc:
    supervisor.fail(task, classify(exc))
finally:
    queue.task_done()
```

`task_done()` 只能执行一次；任务完成集合、失败集合和 waiter 唤醒也必须放在同一个 finally-aware 的生命周期中。这样 backend 异常发生在 key 生成、地址切分或结果 bookkeeping 任意阶段时，都不会留下无法归零的 unfinished task。

#### 限制队列和施加背压

当前无限制地向 `queue.Queue` 投递任务会放大故障时的积压。应设置 `max_inflight`，提交方达到上限时等待带超时的 condition。超时后返回 `TransferBackpressureError`，由 connector 选择跳过 save 或进行 recompute，而不是继续堆积 Python 对象。

#### 原生化和缓存化元数据

优先优化以下热点，而不是先引入进程复制：

- 缓存 `PoolKey.to_string()` 的稳定前缀和 layer/group 组合；
- 将连续 block range 从逐 block Python 循环改成批量 descriptor；
- 一次性生成地址、stride、size 数组，减少重复 list append 和对象创建；
- 将 `ChunkedTokenDatabase.process_tokens()` 的纯数组运算改为 NumPy/C++/torch 算子（需评估 NPU/CPU 复制开销）；
- 将 `m_store.exists()` 合并为批量 lookup，减少 Python 到原生库的调用次数。

### 3.5 故障策略

- **load backend 失败**：返回 `RETRYABLE_ERROR` 或 `MISS`，标记对应 block 无效，沿用现有 recompute 路径；
- **save backend 失败**：本地模型结果仍视为有效，但撤销本次写入 lease，清理临时 key；
- **线程异常**：supervisor 立即设置全局 failure generation，唤醒所有 waiter，拒绝新命令；
- **超时**：不调用无限期 `join()`，先 cancel，再等待线程在 deadline 内退出；无法退出时记录 engine-level error；
- **恢复**：当前线程实现可以重新创建 send/recv 线程，但重启前必须清空旧队列并给旧命令返回 `LOST`，不能让旧线程和新线程同时写同一个 key。

### 3.6 优点、缺点和性能预期

优点：改动面最小，不触碰 NPU buffer/event 的跨进程语义；可立即修复永久等待和错误不可见；能建立可靠 profiling 基线。

缺点：Python 线程仍与模型主线程共享 GIL，无法获得完整崩溃隔离；若 key/address 准备确实占主要 CPU 时间，收益有限。

性能判断必须以 profiling 为准。建议把“CPU 元数据时间占比”和“模型 forward 重叠程度”作为是否继续进程化的闸门，而不是把进程化本身当成性能指标。

### 3.7 验收条件

- 注入 key 生成、地址切分、backend、NPU wait 各类异常后，所有 waiter 都能在 deadline 内返回；
- 任意任务失败后，`unfinished_tasks` 最终归零或被明确取消，不再出现无期限 `queue.join()`；
- 线程重启不会重复提交旧命令，save 不会产生可见的半写入 key；
- 与改造前相比，正常请求的 TTFT、load/save p95 不回退；
- 所有非 layerwise 和 layerwise 单测、异常测试通过。

## 4. 方案二：混合进程——子进程负责 CPU 规划/lookup，主进程保留 NPU 读写

### 4.1 适用场景和推荐理由

这是当前最推荐的进程化路线。它把真正受 GIL 影响的纯 Python 规划工作隔离出去，但不要求子进程直接解引用 NPU 地址，也不要求跨进程导入 `torch.npu.Event`。

适用条件：

- profiling 证明 key/token/block 遍历、key 字符串构造或 lookup 组织占据明显 CPU 时间；
- 需要把元数据异常与模型 worker 隔离；
- Ascend/MemCache 尚未提供可验证的跨进程 NPU buffer handle；
- 可以接受主进程仍保留一个轻量 NPU I/O 线程或原生异步调用。

核心原则是：**子进程只处理可序列化、不可变、纯 CPU 的输入；主进程仍拥有 NPU tensor、stream、event 和 `MemCacheBackend` 的设备访问权。**

### 4.2 架构

```text
模型 worker 主进程
  -> KVPoolWorker
      -> TransferSupervisor
          -> PlannerClient  -- IPC --> PlannerProcess
          -> NPUTransferThread（主进程内）
      -> NPU KV cache / torch.npu.Event / MemCacheBackend

PlannerProcess
  -> ChunkedTokenDatabase（纯 CPU 副本）
  -> PoolKey / key ids 生成
  -> lookup key 批处理
  -> 返回 TransferPlan，不接触 NPU 指针和 event
```

此方案中 send/recv 逻辑需要拆成两段：

1. planner 生成 `TransferPlan`：key、block 映射、layer/group、地址索引和依赖 token；
2. 主进程 NPU transfer 执行 plan：把 plan 中的逻辑 buffer slot 映射到本地 tensor 地址，调用 `MemCacheBackend.get()` / `put()`。

### 4.3 输入和返回数据设计

Planner 输入不应携带 request 对象或 tensor，而应是快照：

```text
PlanCommand {
  request_id,
  command_id,
  operation: load | save | lookup,
  mode,
  tokens_ref,             # 共享内存中的只读 token 数组
  block_ids_ref,
  block_ranges_ref,
  layer_group_config_hash,
  block_size,
  tp_rank, dcp_rank, pp_rank,
  cache_generation,
  deadline_ns,
}
```

返回值只描述逻辑位置，不传裸 NPU 地址：

```text
TransferPlan {
  command_id,
  request_id,
  key_ref,
  block_indices_ref,
  layer_descriptors_ref,
  logical_buffer_slots_ref,
  lookup_result_ref,
  plan_hash,
}
```

主进程持有 `logical_buffer_slot -> tensor, offset, stride, size` 的 registry。这样 planner 即使被重启，也不能访问已经失效的设备地址；主进程还可以校验 `plan_hash`、slot 范围和 layer/group 数量，拒绝越界或过期计划。

大数组应使用共享内存 descriptor，而不是每次 pickle 完整 list。共享内存只存 CPU 数据，并带 owner、generation、refcount 和释放状态。命令完成或超时后由 supervisor 统一回收，避免 planner 崩溃留下无法释放的 segment。

### 4.4 启动和初始化流程

1. `KVPoolWorker` 创建 supervisor 和 `spawn` 出来的 planner process；
2. 子进程只接收可序列化配置：block size、rank、KV group 描述、key 前缀、协议版本；
3. planner 初始化 CPU `ChunkedTokenDatabase`，发送 `READY(generation)`；
4. 主进程注册 NPU cache，建立逻辑 slot registry；该 registry 不发送给 planner；
5. supervisor 开始接受 load/save 命令。

不应让 planner 进程初始化 NPU、torch.npu、MemCache client 或 ZMQ client。这样可以避免 `spawn` 后设备上下文重复初始化，也避免 planner 故障影响 NPU cache 本身。

### 4.5 load 流程

```text
ModelRunner.start_load_kv()
  -> KVPoolWorker 快照 request/token/block 元数据
  -> PlannerClient 提交 PlanCommand
  -> PlannerProcess 生成 key 和逻辑 block plan
  -> 返回 TransferPlan
  -> 主进程校验 plan generation/hash
  -> 主进程根据 layer gate 安排 NPUTransferThread
  -> MemCacheBackend.get(local tensor descriptors, key descriptors)
  -> 主进程记录 block 命中/失败
  -> wait_for_layer_load() 返回 typed result
```

普通模式可以一次生成整个请求的 plan；key-layerwise 和 GVA-layerwise 可以按 layer 生成小 plan。layerwise 的 NPU event 等待必须仍在主进程完成：主进程先根据 `AttentionComputeStartGate` 判断允许执行，再把相应 layer plan 交给 NPUTransferThread。

### 4.6 save 流程

```text
KV connector.save_kv()
  -> 主进程生成不可变元数据快照
  -> PlannerProcess 生成 key/lookup/commit plan
  -> 主进程等待当前 layer 的 attention/NPU event 边界
  -> MemCacheBackend.put(local tensor descriptors, key descriptors)
  -> 主进程根据 backend result 提交完整 key 或清理临时 key
  -> wait_for_save() 返回
```

planner 只能生成 key，不能宣布 save 成功。save 的成功条件必须由主进程根据 MemCache 返回码和所有 block 完成情况决定，避免 planner 正常退出但 NPU 写入失败时错误地将 key 标记为命中。

### 4.7 layerwise 同步设计

方案二不跨进程传递 `torch.npu.Event`。建议使用两层 token：

- **设备依赖**：由主进程中的 `AttentionComputeStartGate` 和 `sync_save_events` 保证；
- **规划依赖**：planner 返回的 `dependency_token` 只表示上一层的计划已生成，不代表 NPU copy 已完成。

每层的正确顺序为：

```text
上一层 NPU save 完成
  -> 主进程记录 layer_save_finished
  -> 提交下一层 planner 命令
  -> planner 返回 plan
  -> 主进程等待 attention gate
  -> 执行当前层 NPU load/save
  -> 设置当前层完成事件
```

这种顺序会牺牲一部分 planner 提前计算空间，但保留了 NPU stream 语义，适合作为第一版。后续可在不跨进程传 event 的前提下提前生成多个 layer plan，再由主进程按 gate 逐层执行。

### 4.8 planner 故障、超时和降级

- planner 进程退出：supervisor 将当前 generation 标记失效，所有未完成 plan 返回 `LOST`；
- load plan 丢失：主进程可以直接走本地 key 生成或跳过远端 KV 并 recompute；
- save plan 丢失：保留模型本地结果，放弃远端写入，不影响当前 token 生成；
- planner 超时：只取消 CPU 计划，不触碰正在执行的 NPU 操作；
- 连续重启：达到上限后自动切换到方案一线程模式，或按配置禁用 AscendStore；
- 旧 planner 的迟到响应：通过 `worker_generation` 和 `cache_generation` 丢弃，不能重新提交旧 plan。

该降级路径是方案二的主要价值：即使 planner 崩溃，NPU cache 和模型 forward 仍在主进程，恢复成本明显低于完整传输进程。

### 4.9 代码改动范围

- 新增 `transfer_protocol.py`、`planner_process.py`、`planner_client.py`、`transfer_supervisor.py`；
- 将 `metadata.py` 中 `ChunkedTokenDatabase`、`PoolKey` 的纯 CPU 部分整理为无进程状态的可构造组件；
- `pool_worker.py` 增加 logical buffer registry、planner 生命周期和本地 NPU transfer adapter；
- `kv_transfer.py` 拆出“计划生成”和“backend 执行”，保留原线程实现作为 fallback；
- `attention_fence.py` 保持 API 不变，所有 NPU event 操作仍由主进程执行；
- `memcache_backend.py` 增加接受逻辑 descriptor、批量 key 和明确 partial result 的接口；
- `ascend_store_connector.py` 将 planner failure、backend failure、KV miss 分别映射为 connector 状态；
- 不需要修改 vLLM 上游的 `execute_model()` 控制流，只需确保 `wait_for_layer_load()` 和 `wait_for_save()` 可接收 typed error 和 timeout。

### 4.10 优点、缺点和适用边界

优点：

- 不需要跨进程导出 NPU 内存和 NPU event，规避当前最大技术不确定性；
- Python key/token 计算与模型 worker 解耦，能降低 GIL 竞争；
- planner 崩溃不会直接破坏 NPU backend 和 cache；
- 可渐进迁移，普通模式先行，layerwise 后行；
- 可与方案一共用 supervisor、协议、超时和测试。

缺点：

- 主进程仍有一个 NPU transfer thread，不能隔离所有原生库故障；
- plan 序列化、共享内存管理和主进程 descriptor 映射会增加复杂度；
- 如果 `MemCacheBackend` 的 Python 包装本身是主要 GIL 热点，收益可能不完整；
- layerwise 需要主进程协调 event，跨进程并行度低于理想状态。

### 4.11 验收条件

- planner 与主进程可独立重启，旧 generation 的响应不会污染新请求；
- 共享内存 plan 无泄漏，异常退出后 supervisor 能回收或重新创建；
- 完整命中、部分命中、全 miss 与线程模式结果一致；
- layerwise 每层 attention fence、stagger 顺序和最终 KV 正确性一致；
- planner CPU 时间从主 worker 中移出，主 worker GIL/runnable 时间下降；
- 在相同 workload 下，IPC 计划开销小于节省的 Python 准备时间，否则自动回退方案一。

## 5. 方案三：完整传输子进程——子进程导入 NPU buffer 并拥有 backend

### 5.1 适用场景和前置条件

这是隔离最完整、改动最大的路线。`TransferProcess` 在独立进程内初始化 device、MemCache backend，并通过厂商提供的 IPC handle/GVA 访问主进程创建的 KV cache。主进程只负责调度、协议、状态和 engine 集成。

在进入开发前，必须由 Ascend runtime、torch_npu、Mooncake/MemCache 维护者共同确认以下 API 和语义：

1. device memory 是否可以导出 opaque IPC handle，子进程能否在同一 device/rank 导入；
2. handle 是否支持 `spawn`，是否要求 exporter/importer 在特定 runtime context；
3. buffer 的引用计数、关闭顺序和进程异常退出后的回收机制；
4. `torch.npu.Event` 或等价 device event 是否支持 IPC；
5. MemCache `register_buffer()` 是否接受跨进程 GVA/handle，而不是只接受当前进程的裸地址；
6. 多卡、TP/DCP/PP rank、容器 namespace 和 device reset 时的行为。

如果上述任何一项没有可运行原型，方案三只能停留在 mock backend，不能承载真实 KV load/save。

### 5.2 架构

```text
主进程
  -> TransferSupervisor
      -> TransferProcessClient -- control IPC --> TransferProcess
      -> KVPoolWorker / EngineCore
  -> 创建 KV cache，导出 BufferHandle
  -> 创建或导出 EventHandle（若采用 event IPC）

TransferProcess（spawn）
  -> set_device(rank)
  -> import BufferHandle
  -> 初始化 MemCacheBackend
  -> register imported buffers
  -> 执行 key 解析、backend get/put、NPU event wait
  -> 返回 TransferResult
```

完整方案中，子进程应拥有自己的 backend client 和 event wait 逻辑，主进程不再通过 `KVTransferThread` 直接调用 `m_store.get()` / `put()`。但 request、scheduler 和 engine 状态仍归主进程所有，子进程不能反向修改 vLLM 对象。

### 5.3 启动、buffer 注册和关闭顺序

推荐顺序如下：

1. 主进程完成 NPU cache allocation，但不把裸 `data_ptr()` 发给子进程；
2. 通过厂商 API 为每个 KV group、layer 或连续 region 导出 `BufferHandle`，包含 `device_id`、逻辑 slot、byte size、stride、layout version 和 exporter generation；
3. `spawn` 子进程；
4. 子进程设置 rank/device，初始化 NPU runtime 和 MemCache client；
5. 子进程导入 handle，校验 device、size、stride、layout hash；
6. 子进程调用后端跨进程注册接口，返回 `BUFFER_READY`；
7. supervisor 发送 `START_ACCEPTING`，之后才允许 load/save；
8. 关闭时先停止新命令，再等待/取消 in-flight 命令，子进程注销 backend buffer，最后导出端释放 handle 和 cache。

禁止在已初始化 NPU、ZMQ 或 MemCache 后 `fork`。子进程只能通过 `spawn` 从可序列化配置启动；所有 runtime/client/stream 都在子进程重新建立。

### 5.4 两种 event 同步模式

#### 模式 A：主进程显式同步后发送 token（第一版推荐）

主进程继续持有 `AttentionComputeStartGate` 和 `torch.npu.Event`。当 event 达到可安全传输的状态后，主进程发送 `DEPENDENCY_SATISFIED`，子进程只等待这个控制 token。

```text
主进程 record_attention_compute_start()
  -> 主进程按现有 stream 语义 synchronize/wait
  -> 发送 layer dependency token
  -> 子进程执行该 layer 的 get/put
```

优点是实现和调试简单，语义最接近现有代码；缺点是主进程可能因为显式同步失去部分 overlap。

#### 模式 B：导出 NPU event IPC handle（优化版）

主进程导出 event handle，子进程导入后在正确 device/stream 上等待。必须验证 event handle 生命周期和 device reset 语义。任何导入失败都应回退模式 A 或方案二，不能静默改成普通 Python event。

### 5.5 普通模式的 load/save 流程

普通 load：

```text
start_load_kv()
  -> 主进程生成版本化 TransferCommand
  -> command IPC 发送 key/block/buffer handle 引用
  -> 子进程校验 handle 和 generation
  -> MemCacheBackend.get()
  -> 子进程返回 hit/miss/failed block
  -> 主进程更新 request 状态并允许 forward
```

普通 save：

```text
save_kv()
  -> 主进程发送带 idempotency_key 的 save command
  -> 子进程执行 put 到临时 generation
  -> 所有 block 成功后提交可见标记
  -> 返回 SAVE_COMMITTED
  -> 主进程 wait_for_save() 完成
```

save 重试必须幂等。命令重放时，子进程应根据 `idempotency_key` 检查已有临时写入，避免重复写入造成 lease 泄漏或同一 key 的版本冲突。

### 5.6 layerwise 和 GVA-layerwise 流程

每个 layer command 应包含：`layer_id`、`group_id`、逻辑 buffer handle、上一层 dependency token、当前层 event handle/token、stagger slot 和 layout version。子进程完成当前层后返回 `LAYER_LOAD_DONE` 或 `LAYER_SAVE_DONE`，主进程才推进对应事件数组。

必须保留以下不变量：

- layer `n` 的 load 不能覆盖仍被 attention 使用的 layer `n-1` cache；
- layer `n` 的 save 只有在 attention 对应 stream 已到达 fence 后才能开始；
- 跨 group stagger 的 slot 只能被一个 in-flight command 所有；
- 进程重启后所有未确认 layer command 都按 `LOST` 处理，不能假定设备 copy 已完成；
- `completed_layers` 必须由子进程实际 backend 结果产生，不能由命令发送方推断。

建议第一版只迁移普通 non-layerwise，再迁移 key-layerwise，最后处理 GVA-layerwise 和 event IPC。完整路线如果一次性覆盖三种模式，调试面会同时包含协议、NPU handle、事件顺序和 stagger slot，风险不可控。

### 5.7 Supervisor、心跳和重启

完整传输进程必须有独立 supervisor 逻辑，不能依靠 `Process.is_alive()` 的偶尔查询：

- `HELLO/READY`：确认协议版本、device、rank、buffer generation；
- `HEARTBEAT`：周期性报告当前 command、队列深度、backend 状态和最后完成时间；
- `COMMAND_ACK`：确认命令已被子进程接收；
- `RESULT`：报告最终状态；
- `FATAL`：携带错误码、command_id 和是否可重试；
- `SHUTDOWN_ACK`：确认 backend、event 和 handle 已关闭。

watchdog 判定进程死亡或失联后：

1. 停止发送新命令；
2. 将未完成命令从 `RUNNING` 转为 `LOST`；
3. 广播唤醒所有 layer/load/save waiter；
4. 让 connector 选择 recompute、请求失败或 engine shutdown；
5. 关闭旧 IPC endpoint，销毁或标记旧 buffer generation；
6. 按重启次数和退避策略 spawn 新进程；
7. 新进程重新导入 buffer handle、重新注册 backend；
8. 只重放明确幂等且尚未提交的命令。

不要在进程崩溃后无条件重放 save。若无法确认远端是否已经完成写入，应查询 generation/commit marker；无法确认时宁可丢弃该次远端 save，也不能发布可能不完整的 key。

### 5.8 代码改动范围

- 新增 `transfer_process.py`、`transfer_supervisor.py`、`transfer_protocol.py`、`buffer_registry.py` 和 `transfer_errors.py`；
- `pool_worker.py`：管理 process client、buffer handle export/import、generation 和 fallback；
- `memcache_backend.py`：增加 `register_buffer_handle()`、handle 注销、partial/commit API；
- `kv_transfer.py`：把现有派生类改造成 process worker 可调用的纯函数/命令 handler，删除对进程内 queue/event 的隐式依赖；
- `metadata.py`：所有任务转换为协议 descriptor，禁止携带 request、tensor、event 等不可序列化引用；
- `attention_fence.py`：提供“显式同步 token”和“event handle”两种后端，默认显式 token；
- `ascend_store_connector.py`：把进程 `LOST`、timeout、partial save 和 retryable backend error 转换为 vLLM 可处理的状态；
- `vllm/v1/worker/kv_connector_model_runner_mixin.py`：原则上不改执行顺序，只要求 connector 的 wait 方法不再无限等待；必要时增加 typed exception 到 engine 的映射；
- `vllm/v1/engine/core.py`：只有当产品策略要求进程连续重启失败后终止 engine 时才需要增加明确的 engine-level error 处理。

### 5.9 优点、缺点和性能预期

优点：

- Python 计算、backend client 和部分原生故障与模型 worker 隔离；
- 理论上可以独立调节传输进程 CPU 亲和性、队列和重启；
- 主 worker 的 GIL 竞争最少，适合 CPU 元数据和 backend 包装都较重的 workload；
- 可实现真正的传输进程心跳、超时和独立扩缩容。

缺点：

- 最大依赖是 NPU memory/event IPC API，当前代码没有这层抽象；
- handle、device context、stream、event、backend socket 的生命周期非常容易形成泄漏或 use-after-close；
- IPC 序列化和同步可能抵消 GIL 收益；
- 进程重启后的 NPU cache 一致性、partial write 和 lease 恢复需要大量硬件测试；
- 任何 rank/device 绑定错误都可能表现为低层 runtime 错误，排查成本高。

只有在方案二的 CPU profiling 已经证明仍有明显瓶颈，并且 NPU IPC 原型通过完整正确性测试后，才值得选择方案三。

## 6. 三种方案对比

| 维度 | 方案一：线程增强 | 方案二：混合进程（推荐） | 方案三：完整传输进程 |
|---|---|---|---|
| 主要解决 | 等待/异常可靠性、部分 GIL | Python 元数据 GIL、元数据故障隔离 | 最大化 GIL 和故障隔离 |
| NPU buffer | 主进程裸地址 | 主进程裸地址 | 需要 IPC handle/GVA |
| NPU event | 主进程原样使用 | 主进程原样使用 | 显式 token 或 event IPC |
| 改动规模 | 小 | 中 | 大 |
| 对厂商 API 依赖 | 低 | 低 | 很高 |
| layerwise 风险 | 低 | 中低 | 高 |
| 可回退性 | 高 | 高，可回退方案一 | 中，需要处理 handle generation |
| 故障隔离 | 线程级 + supervisor | planner 进程级，NPU 仍在主进程 | 完整传输进程级 |
| 预期性能 | 取决于原生化效果 | 若 metadata 较重，通常最有希望 | 上限最高，但 IPC 开销也最高 |
| 推荐阶段 | 必做基础 | 首个生产候选 | 实验性/后续阶段 |

决策规则：

- `metadata_cpu_time / total_transfer_time` 很低：选方案一；
- 该比例较高但没有 NPU IPC：选方案二；
- 该比例较高、需要完整崩溃隔离且 NPU IPC 验证通过：再选方案三。

## 7. 推荐实施路线

### 阶段 0：基线、协议和可靠性（所有方案必做）

1. 在现有线程模式中增加分段计时：key 生成、token/block 处理、地址准备、backend call、NPU wait、队列等待、模型 forward。
2. 增加 command/request/layer/block 维度的结构化日志和 trace id。
3. 统一 `task_done()`、异常分类、取消、timeout 和 waiter 唤醒。
4. 引入 `TransferResult` 和 typed error，即使底层仍是线程。
5. 建立正常、异常、超时、取消、重启/fallback 的测试矩阵。

阶段 0 的完成标准是：任何注入故障都不会无限卡住模型 worker；同时得到可用于判断 GIL 占比的真实数据。

### 阶段 1：实现方案一作为稳定 fallback

将 `TransferSupervisor` 和统一协议接入现有 send/recv thread，默认配置仍是线程模式。这样后续方案二、三可以复用状态机、错误码、配置和监控，不需要再重写 connector 等待逻辑。

建议配置项（名称可按项目规范调整）：

```text
ASCEND_STORE_TRANSFER_MODE=thread
ASCEND_STORE_MAX_INFLIGHT=<bounded value>
ASCEND_STORE_TRANSFER_TIMEOUT_MS=<deadline>
ASCEND_STORE_FALLBACK_MODE=thread|disable|fail
```

### 阶段 2：实现方案二并只迁移普通模式

1. 把 `ChunkedTokenDatabase` 和 `PoolKey` 纯 CPU 逻辑封装成 planner worker。
2. 使用 shared-memory descriptor 传 token/block 数组，先支持 normal non-layerwise。
3. 主进程根据 plan 调用现有 `MemCacheBackend`，保持 NPU buffer/event 不变。
4. 做线程 vs 混合进程 A/B，确认命中结果、KV 数值和端到端延迟一致。
5. planner 进程异常时自动回退方案一。

只有普通模式稳定后，再按 key-layerwise、GVA-layerwise 的顺序迁移。每种模式都要有单独开关，不能以一个全局 experimental flag 同时打开所有路径。

### 阶段 3：验证方案三的 NPU IPC 原型

先不要接入 vLLM 全流程，单独建立最小硬件原型：

1. 主进程分配一块 KV tensor，导出 handle；
2. spawn 子进程并导入 handle；
3. 子进程写入固定 pattern，主进程读取并校验；
4. 主进程记录 event，子进程等待后读写，验证 stream 顺序；
5. 强制子进程异常退出，验证 handle、buffer 和 device reset 后能恢复；
6. 在多卡、TP/DCP rank 和多个 layer 下重复。

原型必须通过上述测试，且不存在未释放 handle、随机数据或死锁，才开始实现方案三的真实 load/save。

### 阶段 4：生产灰度和回滚

- 默认仍为方案一或方案二，方案三只允许显式实验开关；
- 记录 process generation、重启次数、plan IPC bytes、NPU copy time、命中率和 fallback 次数；
- 先在单卡、normal mode 灰度，再扩展 TP/DCP 和 layerwise；
- 任一 correctness 失败立即关闭实验模式并回退，不允许自动切换到一个未验证的 event 语义；
- 连续重启超过阈值时停止远端 KV，而不是不断创建新进程。

## 8. 测试和验收矩阵

### 8.1 单元测试

- 协议版本、字段缺失、未知 operation、超大 block range、非法 buffer slot；
- command/result 配对、重复 result、迟到 result、旧 generation result；
- 状态机转换和非法转换；
- `task_done()` 恰好一次、异常路径 waiter 唤醒、队列取消；
- retryable/fatal/partial/backend miss 的错误分类；
- save idempotency key 和 commit marker；
- planner 进程启动失败、协议不兼容、共享内存释放。

### 8.2 mock 进程集成测试

- planner/transfer process 正常启动和 READY handshake；
- 处理一个或多个并发 request，验证 backpressure；
- 在 key 生成、lookup、backend、结果发送四个阶段分别注入异常；
- 进程被 kill、心跳超时、IPC 断开，确认所有等待者在 timeout 内返回；
- 旧进程结果迟到、新进程 generation 已切换时，不污染新请求；
- fallback 到线程后，未完成 load/save 的语义符合策略。

### 8.3 Ascend NPU 正确性测试

- 完整命中、首/中间/尾部 block miss；
- 多 KV cache group、不同 block size、hybrid/mamba；
- TP、PP、DCP、MTP/speculative layer；
- non-layerwise、key-layerwise、GVA-layerwise；
- attention fence 前后写入顺序、stagger slot 复用和跨 group 顺序；
- load 失败 recompute、save 失败 partial key 清理；
- process 正常退出、异常退出、重启后重新注册 buffer；
- 长时间压力运行，观察 NPU memory、IPC segment、handle 和 backend connection 是否增长。

### 8.4 性能验收

固定模型、NPU 型号、卡数、并行配置、prompt 长度、block 数和 KV 命中率，至少比较：

- worker CPU 使用率、GIL contention、planner/transfer process CPU 使用率；
- key/address preparation、IPC serialization、backend call、NPU wait 的分项耗时；
- load/save 吞吐和 p50/p95/p99；
- TTFT、inter-token latency、模型 forward overlap、NPU 利用率；
- 进程重启、fallback 和部分 block 失败的恢复时间。

建议把性能门槛设为“相对于线程基线”的可审查目标，而不是绝对数值。例如：正常路径 TTFT p95 不允许出现不可解释的回退；若声称解决 GIL，应同时证明主 worker Python CPU 时间下降，并且下降幅度大于 IPC/复制新增开销。具体百分比应根据实际 workload 和发布目标在阶段 0 后确定。

## 9. 主要风险和对应措施

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 把裸 `data_ptr()` 当跨进程指针 | 子进程非法访问、NPU runtime 崩溃 | 方案二不跨进程传 buffer；方案三必须使用并验证 opaque handle |
| 用普通 IPC Event 替代 NPU Event | cache 被提前覆盖、layerwise 顺序错误 | 方案二保留主进程 event；方案三第一版使用主进程显式同步 token |
| 直接 pickle 富任务对象 | 序列化失败、对象过期、复制开销大 | 只传版本化 immutable descriptor 和 shared-memory ref |
| worker 死亡后 waiter 不醒 | `execute_model()` 永久阻塞 | supervisor heartbeat、deadline、广播唤醒和 typed error |
| save 重试产生重复/半写 key | 远端 KV 错误命中 | 临时 generation、commit marker、idempotency key |
| 进程重启仍使用旧 handle | use-after-close 或随机失败 | buffer/worker generation 校验，重启后完整重新注册 |
| IPC 复制抵消 GIL 收益 | TTFT/吞吐回退 | bounded queue、共享内存、批量 descriptor、A/B profiling |
| 一次迁移所有 layerwise 模式 | 调试面过大 | normal -> key-layerwise -> GVA-layerwise 分步灰度 |
| fallback 状态不一致 | 新旧线程/进程重复写入 | supervisor 统一所有权，旧 generation 命令全部失效 |

## 10. 最终推荐

推荐采用“**方案一打底，方案二落地，方案三预研**”的组合：

1. 先把 supervisor、协议、超时、异常传播和 `task_done()` 语义接入当前线程；这一步无论最终是否进程化都必须做。
2. 以方案二作为首个真实进程方案，把 `ChunkedTokenDatabase`、key 生成和 lookup 规划迁移到 planner process，NPU cache、`AttentionComputeStartGate` 和 `MemCacheBackend` 保留在主进程。
3. 只有当 profiling 证明方案二仍受主进程 NPU backend 包装限制，并且硬件原型确认 buffer/event IPC 生命周期可靠时，才推进方案三。

这个选择同时满足三个目标：先解决已经被源码证实的等待和故障风险；在不依赖未知 NPU IPC 的情况下获得主要的 GIL 隔离收益；为未来完整传输进程保留协议、状态机、监控和 fallback 基础。

在没有真实 Ascend NPU profiling 和跨进程 buffer/event 原型之前，不应把 Issue 的验收标准写成“所有线程改为子进程”或“性能必然提升”。更准确的验收表述是：**传输任务具备明确的进程/线程后端边界，故障可被检测和恢复，layerwise 与 NPU 内存语义保持正确，并通过目标 workload 的端到端数据证明选择的后端确实有收益。**

