# [Arch/Perf] 传输路径改 IPC：消除 GIL 瓶颈，实现真并行传输

> 编号：kv-35 | 维度：Arch/Perf | 严重程度：高 | 建议优先级：**P0（重点）**
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-12 | 回收：2026-09-30
> 依据：[kv_pool_线程存取与GIL分析.md](file:///D:/lzy/project/kv_pool/llm-project/draft/kv_pool_线程存取与GIL分析.md) 第三、四章

## 背景

`ascend_store` 的 save/load 全部走进程内 daemon 线程（`kv_transfer.py` 中的 `KVCacheStore*Sending/RecvingThread`），主线程与 send/recv 线程之间用 `queue.Queue` + `threading.Event` 协作。该架构的根本限制：

- **每进程仅 1 个 send 线程 + 1 个 recv 线程**（`pool_worker.py:328-329`，`_start_kv_transfer_threads` 二选一），无法横向扩展
- **GIL 卡住 Python 层热路径**：`PoolKey.to_string` 拼接、`_handle_stored_request` 的 `for index, start in enumerate(starts)` 循环构建 addrs/sizes（`kv_transfer.py:846-855`）、key 生成器循环均在 Python 层，无法与 forward 计算真正并行
- **C 扩展是否释放 GIL 本层无证据**：mooncake `batch_put_from_multi_buffers` / memcache `batch_copy` / yuanrong `get` 等是否 `Py_BEGIN_ALLOW_THREADS` 需查 C 源码确认；即便释放，Python 层 key 构建仍串行
- **设计目标是"I/O 与 forward 重叠"，但实际被 GIL 拖累成"key 构建与 forward 串行"**

分析文档原结论"改 IPC 不划算"基于现状代价评估，但本 issue 明确**接受改造代价、推进 IPC 化**，把上述代价作为设计中必须攻克的难点而非放弃理由。理由：长期看，单进程多线程架构无法支撑多 rank 并行传输、无法独立崩溃隔离、无法与 NPU stream/event 跨进程协作的新特性（如独立 prefetch 进程、热重启不丢 KV）。

## 任务

将 `ascend_store` 的传输路径由"进程内多线程 + queue.Queue"改造为"独立传输子进程 + IPC"，使 key 构建、buffer 调度、I/O 提交彻底脱离 worker 主进程的 GIL。分三阶段：

### 阶段 1：设计与可行性验证（必须先过关）
1. **NPU KV cache buffer 跨进程共享方案选型**。当前 `register_buffer` 注册的是设备内存 `data_ptr`（`pool_worker.py:793` 取 `cache.data_ptr()`，`pool_worker.py:864` 调 `register_buffer`）。需评估：
   - Ascend 平台跨进程 NPU 句柄传递 API（对标 CUDA `ipcMemHandle`）
   - 若平台不支持，fallback 为 worker 进程内 mmap CPU staging buffer + 子进程 H2D（需量化中转开销）
2. **mooncake/memcache store 在子进程的初始化策略**。子进程需重新 `store.setup()`（`mooncake_backend.py:124/137`）+ `register_buffer`，评估：
   - 是否复用 worker 进程的 metadata server 注册（避免双份内存池贡献）
   - lease / GVA 分配是否可在子进程独立持有
3. **IPC 通道选型**。参考已有 `LookupKeyServer/Client` 的 ZMQ `ipc://`（`pool_scheduler.py:1199`），但本场景需双向 + 高频 + 流式，评估：
   - ZMQ DEALER/ROUTER vs Unix domain socket vs 共享内存 ring buffer
   - keys（字符串列表）、addrs（`list[list[int]]`）、sizes 的序列化方案（pickle / msgpack / Arrow IPC）
   - 完成事件反向通知（对标当前 `threading.Event`）
4. **输出**：设计文档（含方案对比、选型理由、风险清单、回退预案），评审通过后再进阶段 2

### 阶段 2：原型与基线对比
1. 选定最小路径（建议：非 layerwise save，mooncake 后端）做单子进程原型
2. 量化对比（必须附数据）：
   - 端到端 prefill/decode 吞吐与延迟
   - key 构建阶段 CPU 占比（profile 验证 GIL 释放效果）
   - IPC 序列化往返开销 vs 当前 `queue.Queue.put` 微秒级
3. 若原型证明收益 ≥ 10% 或解锁了新能力（如多 rank 并行传输），进阶段 3；否则回退并产出 follow-up 报告

### 阶段 3：全路径落地
1. 覆盖三种传输模式：非 layerwise / key layerwise / GVA layerwise
2. 子进程生命周期管理：随 worker 启停、异常重启、资源回收
3. 跨进程 NPU Event 协作（layerwise 的 `AttentionComputeStartGate`，`attention_fence.py:27`，需评估跨进程 event 语义）
4. 与现有机制对齐：`attention_fence`、mamba 块延迟释放（`touch_sending_mamba_blocks`）、TP mismatch strided I/O、partial block、KV events 聚合

## 验收标准

### 1. 功能正确性
- 三种传输模式（非 layerwise / key layerwise / GVA layerwise）功能与改造前一致
- greedy / non-greedy 输出一致（精度不退化）
- 子进程异常不拖垮 worker 主进程，且能被重启或降级回进程内线程模式

### 2. 性能收益
- 附改造前后对比数据：吞吐 / 延迟 / key 构建 CPU 占比 / IPC 开销
- 若未达收益阈值，产出 follow-up issue 说明原因并回链本 issue

### 3. 回归保护
- 现有 `tests/ut/distributed/ascend_store/` 单测全绿
- 新增 IPC 路径单测：子进程启停、异常恢复、跨进程 buffer 共享、IPC 通道断连
- 新增跨进程 NPU buffer 共享的正确性测试（多 rank 场景）

### 4. 交付件
- 设计文档（阶段 1）
- 原型对比报告（阶段 2）
- PR + 设计说明（改动动机 / 方案 / 风险 / 回退路径，阶段 3）
- 单测覆盖

## 证据

- 传输线程类与启动点：[kv_transfer.py:496-1643](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L496-L1643)、[pool_worker.py:453](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L453)
- 每进程仅 1 send + 1 recv：[pool_worker.py:328-329](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L328-L329)
- GIL 受影响的 Python 层热路径：
  - key 生成：[metadata.py:114](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L114)
  - addrs/sizes 循环：[kv_transfer.py:846-855](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L846-L855)
- buffer 注册（跨进程共享难点）：[pool_worker.py:793](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L793)、[pool_worker.py:864](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L864)
- store 初始化（子进程重置代价）：[mooncake_backend.py:124](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L124)
- 已有 IPC 先例（ZMQ，但目的不同）：[pool_scheduler.py:1199](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1199)
- 跨进程 event 协作难点：[attention_fence.py:27](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py#L27)

## 重点关注

- **NPU 跨进程内存共享是硬骨头**：若 Ascend 平台无对标 CUDA `ipcMemHandle` 的稳定 API，需在阶段 1 明确 fallback 并量化开销，不可默默降级
- **mooncake/memcache store 子进程重初始化**：避免双份内存池贡献、避免 metadata server 注册冲突
- **IPC 序列化开销可能吃掉 GIL 释放收益**：keys/addrs/sizes 每次都要过管道，需选高效序列化方案；若数据量大可考虑共享内存 + 仅传描述符
- **layerwise 的 NPU Event 跨进程语义**：`AttentionComputeStartGate` 当前依赖同进程 NPU event，跨进程需重新设计门控
- **子进程崩溃隔离 vs KV 一致性**：子进程崩溃时未完成的 save/load 必须有明确的失败传播路径（与 kv-25/kv-26 联动）
- **降级回退路径**：保留进程内线程模式作为 fallback，子进程启动失败或平台不支持时自动降级，保证可用性

## 与其他 issue 的关系

- **前置/协同**：kv-24（S5 配置 schema）—— IPC 相关配置需纳入 `KVPoolConfig`
- **协同**：kv-25/kv-26/kv-27（正确性 P0 三连）—— 子进程异常清理与失败传播复用其机制
- **可拆分**：阶段 1 设计文档可独立交付；阶段 3 的三种模式可拆为子 PR
- **不冲突**：kv-02（key 向量化）、kv-09（消除 .tolist()）—— 这些 Python 层优化在 IPC 化后依然有益（减少子进程内 CPU 占用）

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数 + TP/CP/PP 配置）
- 关联任务池：#9079
- 验收人：@赵鹏博
