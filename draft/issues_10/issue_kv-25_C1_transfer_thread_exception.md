# [Correctness] 传输线程异常路径资源清理，规避死锁/泄漏

> 编号：kv-25 | 维度：Correctness | 严重程度：高 | 建议优先级：**P0**
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

`KVTransferThread.run()` 捕获异常后直接 `return` 并设置 `_fatal_error`，但**没有调用 `_handle_request_exception`、没有 `task_done()`、也没有释放 lease / 设置 event**。这会导致三类连锁故障：

- `task_done()` 未调用 → `request_queue.join()` 永久阻塞（如 [pool_worker.py:1761](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1761) 的 `wait_for_save`）
- `layer_load_finished_events` 永不 set → `wait_for_layer_load` 等待方死锁（[pool_worker.py:1691-1707](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1691-L1707)）
- lease 未释放 → memcache 资源泄漏

触发概率低，但**触发即卡死**，是生产级风险。

## 任务

修复 `KVTransferThread.run()` 异常分支，确保：
1. 调用 `self._handle_request_exception(request_data)`（基类当前为空实现，子类如 send/recv 线程已实现正常路径版本）
2. 尝试 set 所有相关完成 event，让等待方不被永久挂起
3. 记录失败状态供上层查询（而非只记日志后 return）
4. 保证 `task_done()` 在异常路径也被调用，避免 `queue.join()` 死锁

## 验收标准

### 1. 功能正确性
- 注入异常（mock backend 抛错 / 网络断开）后，等待方能在有限时间内被唤醒，不永久阻塞
- `request_queue.join()` 不再因异常分支挂死
- lease / event 资源在异常路径被正确清理

### 2. 回归保护
- 现有 `tests/ut/distributed/ascend_store/` 单测全绿
- 新增针对异常路径的单测（注入异常验证 event set + task_done 调用 + 状态可查）

### 3. 交付件
- PR + 设计说明（异常清理策略）
- 单测覆盖异常分支

## 证据

- 异常分支直接 return：[kv_transfer.py:510-518](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L510-L518)
- `_handle_request_exception` 基类空实现：[kv_transfer.py:523-525](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L523-L525)
- 子类仅正常路径调用：[kv_transfer.py:670-677](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L670-L677)

## 重点关注

- 不同子类（send/recv/GVA）的 event 集合不同，清理逻辑需按子类差异化处理
- 异常清理本身不能再抛异常（需 try/except 兜底）
- 与 kv-26（backend put 失败传播）联动：put 失败应转化为可被本机制处理的异常

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
