# [Reliability] 建立 transfer thread 终止式失败协议

> 编号：kv-25 | 维度：Reliability | 严重程度：高 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

`KVTransferThread.run` 从 queue 取出任务后，若 `_handle_request` 抛出未捕获异常，会记录 `_fatal_error` 并终止线程。当前通用路径没有完整处理初始化失败、当前任务 queue accounting、未处理任务取消、per-request 状态以及各子类资源清理。

普通 sending thread 已在自身 `try/finally` 中处理常规 `task_done`，layerwise wait 也会周期性 `raise_if_failed`；因此不能简单假设所有异常都会永久卡住，也不能通过 set 所有 event 后继续消费任务来掩盖 fatal。提交 `7201c97a6` 已明确要求 fatal 后停止处理后续任务。

静态代码能够确认 startup 的无界 ready wait 和“缺少统一终止协议”；它不能替代逐子类故障注入来断言每条路径都会遗留相同的 counter、event 或 lease。

## 实现前故障矩阵

1. 对 startup、普通 send、async recv、layerwise key 和 GVA thread 分别在取任务前、取任务后、backend 调用中和 event/lease 更新后注入异常。
2. 每个注入点记录 ready/fatal 状态、`unfinished_tasks`、finished/failed request 状态、等待方、event、counter 和 lease；据此确认需要 cleanup 的真实资源，不重复清理已有 `finally` 覆盖的路径。
3. 先用单测/集成测试复现，再修改状态机。无法在本地 mock 的 NPU event 或 MemCache lease cleanup，需在目标环境补验证，并在交付说明中标明未验证项。

## 任务

1. 定义统一的终止式失败状态：首次 fatal 原因可查询，线程停止消费新任务，后续 `add_request` 被拒绝或由 owner 重建完整 thread/backend 状态。
2. 将 `set_device` 等初始化纳入异常捕获；无论成功失败都唤醒 creator，并为 ready wait 增加有界等待和异常传播。
3. 对故障矩阵中实际进入 queue accounting 的当前任务保证 exactly-once `task_done`；线程终止时取消或排空未处理 queue 项，修正 `unfinished_tasks` 和相关 per-request counters。
4. 各子类只对其实际拥有的资源实现幂等 cleanup：释放本 step 已取得的 GVA lease，清理 transfer state，并唤醒可能等待的 layer/get event。
5. event 只表示等待结束；等待方被唤醒后必须再次 `raise_if_failed`，不能把 event set 解释为传输成功。
6. async non-layerwise `get_finished` 检查 thread fatal，并把失败传播到 model runner/engine，不能留下 `WAITING_FOR_REMOTE_KVS` 请求。

## 验收标准

### 1. 功能正确性
- startup failure 在有限时间内返回 creator，不永久等待 ready event
- fatal 后不处理第二个排队任务，保持现有停止语义
- 当前项与未处理项的 queue accounting 最终归零，`queue.join` 不等待永远不会处理的项
- layerwise key/GVA、普通 send 和 async recv 的等待方均能获知失败，而不是被标记成功
- GVA lease 和 per-request counters 在异常路径幂等清理
- fatal 后新增任务被明确拒绝

### 2. 回归保护
- 保留 `test_fatal_error_stops_before_next_queued_task`
- 新增 startup failure、queue drain/cancel、每类 layer event、lease cleanup 和 async finished polling 测试
- 现有单测全绿

### 3. 交付件
- PR + 线程失败状态机说明 + 资源所有权说明 + 单测

## 证据

- thread lifecycle：`kv_transfer.py:311-399, 496-525`
- 普通 sending thread：`kv_transfer.py:670-715`
- layerwise key/GVA threads：`kv_transfer.py:1042-1652`
- worker wait/poll：`pool_worker.py:456-578, 1701-1771, 2078-2107`
- 关键历史：`7201c97a6`

## 重点关注

- cleanup 自身必须幂等且不能覆盖原始 fatal 原因
- 不恢复“异常后继续处理下一任务”的旧行为
- kv-26 的 per-key put 失败通常不是 thread fatal；只有系统性 backend 异常才进入本协议

## 环境约定
- vllm-ascend：审核基线 `d5e9816065ede613327d93908f87fee9f5c47128` 或提交时最新 main
- 硬件：Ascend NPU（注明型号、卡数和 transfer mode）
- 关联任务池：#9079
- 验收人：@赵鹏博
