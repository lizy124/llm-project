# [Perf] ZMQ lookup 只发后缀 hashes

> 编号：kv-17 | 维度：Perf | 严重程度：中高 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

scheduler 在调用 `LookupKeyClient.lookup()` 时，把**完整的 `request.block_hashes`** 传给 client，再由 worker 侧用 `hbm_hit_tokens` 跳过前缀。对于长 prompt，这会放大 IPC payload 和 msgpack 编码成本。

## 任务

只发送待查询后缀的 hashes，或发送 offset + sliced hashes，减少 IPC 和编码开销。

## 验收标准

### 1. 功能正确性
- 后缀切片后命中结果与发完整 hashes 一致
- 边界（hbm_hit_tokens = 0 / = 全长）处理正确
- 现有单测全绿

### 2. 性能验证
- 长 prompt（16K/64K）下 IPC payload 大小下降
- lookup 延迟下降（msgpack 编解码成本）

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- scheduler 调用：[pool_scheduler.py:565-572](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L565-L572)
- client 编码：[pool_scheduler.py:1158-1178](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1158-L1178)
- worker 侧按 `hbm_hit_tokens` 跳过前缀：[pool_worker.py:2273-2288](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2273-L2288)

## 重点关注

- offset 语义需 scheduler 与 worker 双向对齐（hbm_hit_tokens 的来源）
- 与 kv-15（ZMQ 批合并）协同：payload 减小 + 批合并双重收益
- 长 prompt / 高频 lookup 场景更明显

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
