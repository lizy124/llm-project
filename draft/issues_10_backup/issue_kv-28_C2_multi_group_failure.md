# [Correctness] 多 group 加载失败错误传播策略

> 编号：kv-28 | 维度：Correctness | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

layerwise 多 group 加载失败时，worker 侧释放 lease 并抛 `RuntimeError`，但 recv 线程 `_handle_request` 在多 group 场景**只记日志跳过**，不更新 invalid blocks。注释明确说"多 group 不能安全回退到 per-block recomputation"，意味着失败处理不完整——失败后状态不一致，可能继续用残缺数据。

## 任务

为多 group 加载失败定义明确的恢复策略：
1. 评估可行策略：整请求失败重算 / 降级到 per-block recomputation（需验证安全性）/ 标记整组 invalid
2. recv 线程多 group 失败分支不再"只记日志"，按选定策略传播
3. 失败状态可被 scheduler 观测并触发重算

## 验收标准

### 1. 功能正确性
- 多 group 加载失败后，状态一致（不残留半成品数据被当作有效缓存）
- 失败可被上层观测并触发既定恢复策略
- 单 group 失败路径行为不变（不引入回归）

### 2. 回归保护
- 现有单测全绿
- 新增多 group 失败场景单测

### 3. 交付件
- PR + 设计说明（恢复策略选型理由）
- 单测覆盖多 group 失败

## 证据

- worker 抛异常：[pool_worker.py:1509-1534](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1509-L1534)
- recv 单 group 更新 invalid、多 group 只记日志：[kv_transfer.py:923-1039](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L923-L1039)

## 重点关注

- 多 group 场景目前较少，但 DSV4 hybrid 会增多，紧迫度上升
- 与 kv-25（线程异常清理）联动

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
