# Issue #14143: AscendStore 传输线程改为子进程的代码分析

> 本文记录对 `vllm-ascend#14143` 的源码分析，重点回答两个问题：当前 Issue 描述的问题是否真实存在，以及把传输线程改成子进程时真正需要解决哪些边界。文档引用使用“文件路径 + 类/函数/方法名 + 职责/流程”，不使用行号。

## 1. 分析范围和版本

分析基于本地两个源码仓库的当前工作树：

| 仓库 | 本地路径 | 分析时版本 |
|---|---|---|
| vLLM Ascend | `E:\lizy\code\vllm-project\vllm-ascend` | `182b75ea84ff13ece27b57391331ef8935410aea` (`main`，相对 `origin/main` ahead 136) |
| vLLM | `E:\lizy\code\vllm-project\vllm` | `cdc4824a21eaa986d4d1fee90a7e6465c9f706e6` |

重点查看的文件和职责如下：

| 文件 | 关键符号 | 职责 |
|---|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` | `KVTransferThread` 及其发送/接收派生类 | 后台 KV load/save、key 生成、地址构造、后端读写和完成状态维护 |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` | `KVPoolWorker._start_kv_transfer_threads()` | 按配置创建普通或 layerwise 传输线程，并建立事件和任务队列 |
| 同上 | `register_kv_caches()` | 从 NPU tensor 提取裸地址、长度、stride，向外部 store 注册 buffer |
| 同上 | `start_load_kv()`、`wait_for_layer_load()`、`save_kv_layer()`、`wait_for_save()` | 将 vLLM 一次执行中的 load/save 任务提交到后台传输器并等待完成 |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py` | `PoolKey.to_string()`、`ChunkedTokenDatabase.process_tokens()`、`LayerLoadTask` | 生成远端 key、把 token/block 映射为 KV 地址区间、承载 layerwise 任务元数据 |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` | `_generate_store_query_keys()`、`_get_store_lookup_hit_tokens()`、`LookupKeyClient.lookup()` | scheduler 侧查询远端 KV 命中并通过 ZMQ IPC 与 worker 通信 |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py` | `AttentionComputeStartGate` | 让 layerwise 传输等待 attention 计算流到达指定 NPU event |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py` | `register_buffer()`、`get()`、`put()` | 将地址数组交给 Mooncake/MemCache 后端执行 NPU 内存读写 |
| `vllm/v1/worker/kv_connector_model_runner_mixin.py` | `start_load_kv()`、`wait_for_save()` 调用链 | 把 KV connector 生命周期包在一次 `execute_model()` 内 |
| `vllm/v1/executor/uniproc_executor.py` | `UniProcExecutor.execute_model()` | UniProc 场景下在同一 worker 进程直接执行模型方法 |
| `vllm/v1/engine/core.py` | `EngineCore.step()` 及 `execute_model()` future 回收 | scheduler 调度、worker 执行、KV connector 回收的上层闭环 |

## 2. Issue 要解决的问题

Issue 的核心诉求是：AscendStore 的 KV 传输工作当前放在 Python 线程中，线程内包含 key 构造、token/block 遍历、列表构造、地址计算以及外部 store 调用；希望改成独立子进程，通过进程间通信（IPC）承载任务，以降低 GIL 竞争并获得更好的故障隔离。

这个诉求包含两个不同层面，不能混成一个结论：

1. **结构性问题**：传输线程确实执行了相当多的 Python 计算，并与模型主线程共享解释器，因此存在 GIL 竞争和线程故障影响主流程的风险。
2. **性能结论**：仅凭静态代码不能证明 GIL 已经是主要瓶颈，也不能证明换成进程后一定提升吞吐或 TTFT。需要在真实 NPU workload 上 profiling。

## 3. 结论摘要

### 3.1 问题是否存在

**存在，但应准确表述为“结构性风险已被源码证实，性能损失尚未被实测量化”。**

- 当前传输器是进程内 daemon 线程，不是独立 worker 进程。
- 线程执行路径包含 Python 密集的 key 和地址准备工作，GIL 竞争是合理风险。
- 线程异常只会终止该线程；部分等待方仍可能等待队列完成，存在永久等待或请求停滞的风险。
- 现有 layerwise 机制强依赖进程内对象、`threading.Event`、`torch.npu.Event` 和 NPU tensor 地址，不能直接把 `Thread` 替换成 `Process`。
- 跨进程复用 NPU KV buffer 需要 NPU IPC handle、GVA/共享内存机制或厂商后端提供的跨进程注册能力；裸 `data_ptr()` 不能作为通用跨进程指针。

### 3.2 对“改成子进程”的判断

**方向可行，但不是局部重构。** 最小可行方案必须同时设计：任务协议、NPU 内存所有权、事件同步、错误传播、进程重启/fallback，以及退出时的队列 drain。建议先把“Python 元数据准备”和“真正的 NPU/MemCache 传输”解耦，再决定哪些部分放入子进程。

### 3.3 当前证据的边界

本分析没有在 Ascend NPU 环境执行吞吐、延迟、CPU/GIL 或 TTFT profiling。因此不能写成“已证明线程导致性能下降”，只能写成：**代码结构已经形成可解释的 GIL 和故障隔离风险，需用 profiling 判断其在目标 workload 中的占比。**

## 4. 当前实现架构

### 4.1 线程基类的生命周期

`vllm_ascend/.../ascend_store/kv_transfer.py` 中的 `KVTransferThread` 继承 `threading.Thread`，以 daemon 线程启动。它在构造时持有：

- `Backend` 实例 `m_store`；
- `ChunkedTokenDatabase`，负责 token/block 到 key 和地址区间的映射；
- block size、TP/DCP rank 等并行配置；
- `queue.Queue` 类型的 `request_queue`；
- `stored_requests`、`finished_requests`、`kv_events` 等进程内状态；
- `threading.Lock` 和 `threading.Event` 等同步对象。

`KVTransferThread.run()` 的主要流程是：

```text
线程启动
  -> m_store.set_device()
  -> ready_event.set()
  -> request_queue.get()
  -> 派生类 _handle_request(request)
  -> 派生类 task_done()/更新完成集合
  -> 继续处理下一个请求
```

`add_request()` 只负责把 Python 对象放入队列；`raise_if_failed()` 在主流程需要时检查 `_fatal_error`。

### 4.2 三种传输模式

`KVPoolWorker._start_kv_transfer_threads()` 根据配置选择以下模式：

| 模式 | 发送端 | 接收端 | 主要用途 |
|---|---|---|---|
| 非 layerwise | `KVCacheStoreSendingThread` | `KVCacheStoreRecvingThread` | 按完整 KV block 批量保存/加载 |
| key-layerwise | `KVCacheStoreKeyLayerSendingThread` | `KVCacheStoreKeyLayerRecvingThread` | 按 layer 顺序生成 key，并用 layer event 串联 load/save |
| GVA-layerwise | `KVCacheStoreLayerSendingThread` | `KVCacheStoreLayerRecvingThread` | 按 GVA/批量地址描述执行 layerwise 传输，支持 stagger 和跨 group 构造 |

每个 `KVPoolWorker` 通常最多启动一个 send 线程和一个 recv 线程；layerwise 还会为每个 layer 准备事件数组，但不是每层启动独立 OS 线程。

### 4.3 一次请求的主流程

普通模式可概括为：

```text
vLLM ModelRunner
  -> KV connector.start_load_kv()
  -> KVPoolWorker.start_load_kv()
  -> 构造 Layer/Transfer task
  -> recv thread 从 request_queue 取任务
  -> ChunkedTokenDatabase 生成 key、addr、size
  -> MemCacheBackend.get()
  -> finished_requests 更新
  -> ModelRunner 执行 attention/forward
  -> KV connector.save_kv()
  -> send thread 执行 lookup + MemCacheBackend.put()
  -> wait_for_save() 等待队列和完成状态
```

`vllm/v1/worker/kv_connector_model_runner_mixin.py` 将 `start_load_kv()`、模型 forward 和 `wait_for_save()` 放进同一次 `execute_model()` 生命周期，因此传输线程的阻塞、异常或状态残留会直接影响 worker 执行闭环。

## 5. GIL 竞争是否真实存在

### 5.1 传输线程不是纯 I/O 包装

在 `KVCacheStoreSendingThread._handle_request()`、`KVCacheStoreRecvingThread._handle_request()` 以及两组 layerwise 派生类中，线程需要执行：

- 遍历每个请求的 block range 和 token chunk；
- 调用 `ChunkedTokenDatabase.process_tokens()` 或带 block id 的变体；
- 构造 `PoolKey`，执行 `PoolKey.to_string()`；
- 生成多层 key 列表、地址列表和 size 列表；
- 按 TP/DCP/PP、group、layer 过滤和切片；
- 计算 `data_ptr()` 派生的地址、stride 和分块传输范围；
- 在 layerwise 模式中维护每层的 task、request id 和完成事件。

这些步骤大部分是 Python 对象操作。即使底层 `m_store.get()`/`put()` 最终进入 C/C++ 并可能释放 GIL，调用前后的准备和结果处理仍在 Python 解释器中执行。模型 worker 主线程同时运行 scheduler/connector bookkeeping、模型调用包装和 Python 调度逻辑，因而存在争用。

### 5.2 哪些部分未必受 GIL 限制

不能把所有传输耗时都归因于 GIL：

- `torch.npu.Event.synchronize()`、NPU kernel、DMA 和 MemCache batch copy 主要在设备或原生库中执行；
- `m_store.exists()`、`get()`、`put()` 的底层实现可能释放 GIL；
- 当 key 数量很少或模型 forward 远重于传输准备时，线程的 GIL 占比可能很低。

因此验收时应分别测量：Python key/address preparation、backend call、NPU wait、队列等待、模型 forward 和端到端 TTFT，而不是只比较总耗时。

## 6. 故障隔离和潜在等待问题

### 6.1 当前线程故障行为

`KVTransferThread.run()` 捕获异常后执行以下逻辑：

1. 把异常保存到 `_fatal_error`；
2. 记录日志；
3. 直接 `return`，线程结束；
4. 后续队列任务不再处理。

`tests/ut/distributed/ascend_store/test_kv_transfer.py` 中的 `test_fatal_error_stops_before_next_queued_task` 明确验证了这一语义：第一个任务失败后线程停止，第二个任务仍留在队列中，`raise_if_failed()` 才能在调用方看到错误。

### 6.2 `queue.join()` 的风险

正常路径由各派生类在处理结束时调用 `request_queue.task_done()`；`KVPoolWorker.wait_for_save()` 会等待发送队列完成。在任务处理过程中发生异常时，基类 `run()` 的异常分支并不统一补偿当前任务的 `task_done()`。如果异常发生在某个派生类完成 bookkeeping 之前，队列的 unfinished task 计数可能无法归零，调用 `request_queue.join()` 的线程就可能永久等待。

这不是说每一种异常都必然死锁：有些派生类有自己的 `try/finally`，可以完成 `task_done()`；但基类没有提供统一的 finally 保障，故障语义依赖具体派生类实现。改进前应先为每类任务建立“异常、取消、重试、task_done”测试矩阵。

### 6.3 子进程能解决什么，不能自动解决什么

子进程可以把 Python 崩溃、解释器级异常和部分原生库故障隔离出去，并允许 supervisor 通过 exit code/heartbeat 发现 worker 死亡。但它不会自动解决：

- 主进程如何取消已经发出的 NPU 操作；
- 未完成任务如何重放或标记失败；
- `wait_for_save()` 如何从进程死亡中及时返回；
- worker 重启后如何重新初始化 device、store 和 buffer；
- request id、finished set、lease 和外部 KV 一致性如何恢复。

## 7. NPU buffer 跨进程的核心难点

### 7.1 当前注册方式依赖进程内裸地址

`KVPoolWorker.register_kv_caches()` 对每个 NPU tensor 使用 `cache.data_ptr()`，同时计算 block length、stride、region length，然后调用 `MemCacheBackend.register_buffer(ptrs, lengths)`。`MemCacheBackend` 再把这些整数地址交给底层 store 的 `register_buffer()`。

这套接口的隐含前提是：注册和使用地址的代码运行在同一进程、同一 NPU runtime 上下文中。把整数 `data_ptr()` 通过 `multiprocessing.Queue` 或 socket 发送给子进程，并不能保证子进程能合法访问同一设备内存；在多数运行时中，它只是子进程地址空间里的无效整数。

### 7.2 子进程方案必须明确内存所有权

可选设计至少有三类：

1. **子进程创建并持有 KV cache**：模型 worker 不再直接拥有同一块 cache，改动最大，可能影响 attention kernel 的输入布局。
2. **NPU IPC handle/GVA 导出导入**：主进程导出设备内存句柄，子进程导入后得到可用映射；需要确认 Ascend runtime、torch_npu 和 Mooncake/MemCache 的 API 是否支持，并处理句柄生命周期。
3. **后端自身的跨进程 buffer 注册**：由厂商 store 接受共享内存/GVA 描述，主进程只发送 opaque handle；需要后端明确保证跨进程、跨 device context 和 fork/spawn 语义。

在没有上述机制之前，不能把 `register_buffer(ptrs, lengths)` 原样搬到子进程。

### 7.3 fork/spawn 不能作为默认答案

NPU runtime、ZMQ context、MemCache client 和线程锁都不适合在已初始化后无条件 `fork`。Windows 本地开发环境本身通常走 `spawn`，而 Linux 部署也应优先显式 `spawn` 并在子进程内重新初始化 device/store。需要把“创建进程前允许继承的对象”限制为可序列化配置和 opaque handles，不能继承已运行的 `torch.npu.Event`、backend socket 或工作线程。

## 8. layerwise 模式的同步边界

### 8.1 进程内事件链

`KVPoolWorker._start_kv_transfer_threads()` 在 layerwise 模式创建：

- `get_event`；
- 每层一个 `layer_load_finished_events`；
- 每层一个 `layer_save_finished_events`；
- 每层一个 `torch.npu.Event` 类型的 `sync_save_events`。

`KVCacheStoreKeyLayerRecvingThread` 和 `KVCacheStoreLayerRecvingThread` 会等待上一层 save event、attention start gate 和当前层 load 条件；发送线程在 layer save 完成后设置对应事件。

### 8.2 attention fence 不是普通 Python 通知

`attention_fence.py` 中的 `AttentionComputeStartGate` 先由 `record()` 在当前 NPU stream 上记录 `torch.npu.Event`，再由传输线程的 `wait()` 调用 event synchronize。`vllm_ascend/attention/attention_v1.py` 的 attention forward 在提交 attention 操作前调用 `record_attention_compute_start()`。

这意味着 layerwise 顺序依赖“同一个 NPU event 对象 + 正确的 device/stream 上下文”，不只是一个跨线程布尔变量。若改成子进程，需要设计：

- 主进程如何把 event 完成状态传给子进程；
- 是否由主进程同步后再发送“允许传输”的消息；
- 或者是否导出 NPU event IPC handle，让子进程等待同一事件；
- 进程退出/重启时如何清理未完成 event 和未释放 slot。

简单替换为 `multiprocessing.Event` 会丢失 NPU stream 语义，可能提前 load、覆盖仍在使用的 cache，或让传输失去 stagger 设计。

## 9. 任务对象和 IPC 协议

### 9.1 当前任务是富 Python 对象

`metadata.py` 中的 `LayerLoadTask` 包含 `transfer_tasks`、`layer_id` 和可选的 `AttentionComputeStartGate`；transfer task 还带 request、block range、block ids、缓存的 `process_tokens` 结果等引用。普通模式任务也会携带请求对象、key/hash 列表和地址元数据。

这些对象适合进程内队列，不适合直接作为稳定的跨进程协议。尤其是：

- request 对象可能继续被 scheduler 修改；
- `AttentionComputeStartGate` 和 `torch.npu.Event` 不可直接 pickle；
- `torch.Tensor`、NPU storage 和 backend 对象不能依赖默认 pickle 传递；
- 把完整 key/address 列表复制到 IPC 会增加序列化和内存带宽成本，可能抵消 GIL 收益。

### 9.2 建议的消息边界

应把任务拆成可版本化的纯数据描述，例如：

```text
TransferCommand {
  request_id,
  operation: load | save | lookup,
  layer_id,
  group_id,
  block_ranges,
  block_hashes 或已生成的 key ids,
  cache buffer handle,
  addr/size descriptor,
  dependency token,
  retry policy,
}
```

返回消息至少要有 `request_id`、状态码、完成层、失败 block、可重试标志和后端错误码。协议应有版本字段，避免主进程和 worker 进程使用不兼容的 task schema。

### 9.3 Python 计算放在哪里

更稳妥的分阶段拆分是：

1. 先在主进程保留 scheduler/request 访问和 key metadata 生成；
2. 把已解析的、不可变的地址描述和 opaque buffer handle 发给传输进程；
3. 子进程只负责 backend I/O、NPU event wait 和结果回报；
4. 在 profiling 证明 key/address preparation 占比足够高后，再把可序列化的 `ChunkedTokenDatabase` 计算迁移到子进程。

这样可以先验证故障隔离和 backend 独立性，避免一次性重写所有 metadata 逻辑。

## 10. vLLM 执行边界和故障传播

`vllm/v1/worker/kv_connector_model_runner_mixin.py` 将 KV connector 的 load、forward 和 save finalize 组织在 `execute_model()` 中；`vllm/v1/executor/uniproc_executor.py` 在 UniProc 模式直接调用 worker 方法；`vllm/v1/engine/core.py` 通过 future 管理非阻塞的模型执行并在 step 中回收结果。

因此子进程方案的错误传播必须跨越三层：

```text
transfer process failure
  -> IPC supervisor detects exit/timeout
  -> KV connector converts to typed load/save failure
  -> ModelRunner/EngineCore chooses recompute, request failure or engine shutdown
```

不能只让子进程打印 traceback。若 `wait_for_layer_load()` 或 `wait_for_save()` 仍只等待本地 event/queue，它们看不到子进程死亡，主 worker 可能一直卡住。

建议定义明确的故障策略：

- load 失败：标记 block 无效并按现有 `kv_load_failure_policy` recompute；
- save 失败：保留本地计算结果，但清理 lease/partial write，不能把半写入 key 当作完整命中；
- 进程死亡：停止接收新任务、唤醒所有等待者、回收旧 IPC endpoint，再按策略重启或降级；
- 连续重启失败：让 EngineCore 得到可诊断错误，而不是无期限等待。

## 11. 当前实现已经做过的保护和仍然缺的能力

### 已有能力

- `KVTransferThread.raise_if_failed()` 可由调用方显式检查异步线程异常。
- 发送/接收线程分别维护完成 request 集合，worker 可以延迟释放 block。
- layerwise 线程通过每层 event 保证一定的保存/加载顺序。
- `MemCacheBackend.get()`/`put()` 会记录底层非零错误码，并返回失败结果。
- 单测覆盖了线程异常后停止处理后续任务的行为。

### 仍然缺失

- 没有真正的子进程崩溃隔离。
- 没有自动重启、心跳、请求超时和队列取消协议。
- 没有线程模式与进程模式之间的配置化 fallback。
- 没有证明所有异常路径都会执行 `task_done()` 并唤醒 `join()` 等待者。
- 没有 NPU buffer/event 的跨进程句柄实现。
- 没有针对进程重启后的外部 KV 一致性、lease 和 partial write 的恢复测试。

## 12. 验收标准逐项判断

| Issue 关注点 | 当前代码能否确认 | 判断 |
|---|---|---|
| 传输使用 Python 线程 | 能 | `KVTransferThread` 继承 `threading.Thread`，由 `KVPoolWorker._start_kv_transfer_threads()` 启动 |
| 线程存在 GIL 竞争机会 | 能确认结构性风险 | key、token、list、地址元数据均在 Python 线程中处理；占比需 profiling |
| 线程故障会影响请求完成 | 能确认风险 | `_fatal_error` 后线程退出，未统一保证队列计数和等待者唤醒 |
| 改为子进程可直接复用当前任务 | 不能 | 任务包含 request 引用、Python event 和 NPU event，需新 IPC schema |
| 改为子进程可直接复用当前 buffer 地址 | 不能 | 当前是裸 `data_ptr()` 注册，需 IPC handle/GVA/后端跨进程 API |
| layerwise 顺序可用普通 IPC Event 替代 | 不能 | 依赖 NPU stream event 和 attention 边界，需保留设备同步语义 |
| 改进后一定有性能收益 | 不能 | 尚无 NPU profiling，且 IPC 序列化/复制可能引入新开销 |
| Issue 的方向值得推进 | 可以 | GIL 和故障隔离目标合理，但应分阶段实现并先建立基线 |

## 13. 推荐实施方案

### 阶段 0：先建立基线和故障测试

在现有线程模式增加 profiling 和测试：

- 记录 key generation、address preparation、backend call、NPU wait、队列等待的耗时；
- 测量不同 prompt 长度、block 数量、layerwise/non-layerwise、TP/DCP 配置下的 CPU 时间和 GIL contention；
- 为每个派生线程增加异常时 `task_done()`、完成集合和等待者唤醒测试；
- 测试 backend 返回错误、线程异常、请求取消和 `wait_for_save()` 超时。

### 阶段 1：引入 supervisor 和可观测的独立 worker

先不迁移 NPU buffer：

- 增加 `TransferSupervisor`，统一管理 worker 状态、heartbeat、command id 和 timeout；
- 保留现有线程作为默认实现；
- 引入实验性的 process backend，只处理纯 CPU lookup/key metadata 或 mock backend；
- 明确进程死亡后的 typed error 和 recompute/fallback 行为。

### 阶段 2：验证 NPU IPC 资源

与 Ascend runtime、torch_npu、Mooncake/MemCache 维护者确认并实现：

- device memory IPC handle/GVA 导出和导入；
- NPU event IPC 或主进程显式同步协议；
- buffer、event、stream、device context 的生命周期和关闭顺序；
- spawn 模式下的初始化和多卡 rank 绑定。

没有这一步，process backend 只能作为 CPU/mock 实验，不能承载真实 KV load/save。

### 阶段 3：迁移真实传输路径

按风险从低到高迁移：

1. 普通 non-layerwise recv/save；
2. 普通模式 lookup 和 key preparation；
3. key-layerwise；
4. GVA-layerwise、attention fence 和 staggered transfer。

每一步都保留线程 fallback，并使用相同的 request/result schema 做 A/B 对照。

## 14. 必须补充的 profiling 和 e2e 验证

### 14.1 CPU/GIL profiling

至少记录：

- `ChunkedTokenDatabase.process_tokens()`、`process_token_key_strings_with_block_ids()` 的 CPU 时间；
- `PoolKey.to_string()` 和 key list materialization 的累计时间；
- `prepare_value_layer()`、地址切片和 batch split 的时间；
- `m_store.exists/get/put` 的 Python wall time 与 native/device time；
- 模型 forward 同期的线程 runnable/blocked 时间和 GIL contention。

### 14.2 NPU/端到端 profiling

固定模型、NPU 型号、卡数、TP/PP/DCP、prompt 长度和 KV 命中率，比较：

- 线程模式 vs 进程模式；
- non-layerwise vs key-layerwise vs GVA-layerwise；
- load/save 吞吐、p50/p95/p99 延迟、TTFT、GPU/NPU 利用率；
- IPC 序列化、复制和 event wait 的新增开销；
- 进程重启、backend error、部分 block 失败时的恢复时间。

### 14.3 正确性矩阵

至少覆盖：

- 完整命中、首 block miss、中间 block miss、尾部 miss；
- 多 KV cache group、hybrid/mamba、不同 block size；
- TP/PP/DCP、MTP/speculative layer；
- layerwise 每层 save/load 顺序和 attention fence；
- load 失败后的 recompute，以及 save 失败后的 partial key 清理；
- 进程正常退出、异常退出、超时、重启和 fallback。

## 15. 最终判断

Issue #14143 指出的现象在当前代码中**确实存在**：AscendStore KV 传输由进程内 Python daemon 线程承担，线程执行了非少量 Python 元数据工作，且线程失败后的等待/恢复边界不完整。把这件事描述为“存在 GIL 竞争和故障隔离不足的结构性问题”是有源码依据的。

但“因此必须把所有线程改成子进程，并且一定提升性能”目前还不能成立。真实方案必须解决 NPU buffer 的跨进程访问、NPU event/stream 同步、任务对象的 IPC 协议、错误传播、重启和 fallback。最合理的推进方式是先做 profiling 和 supervisor/故障语义，再验证 NPU IPC 资源，最后分模式迁移并保留线程 fallback。

在没有 NPU profiling 和跨进程 buffer/event 原型之前，建议将 Issue 状态定为：**问题确认，方向可行，实施方案和性能收益待验证；不建议直接进行 Thread 到 Process 的机械替换。**

