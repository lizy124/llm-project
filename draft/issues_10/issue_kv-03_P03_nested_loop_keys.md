# [Perf] `_generate_store_query_keys` 5 层嵌套循环优化

> 编号：kv-03 | 维度：Perf | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

scheduler 侧生成查询 key 时有 5 层嵌套 for 循环（block_hash × pcp × dcp × head_or_tp × pp），每个 block 都构造 `KeyMetadata` + `PoolKey` 对象，再逐层 `to_string`。命中检查频率受 PCP/DCP 影响（通常为 1，但 head_or_tp / pp 维度仍累乘）。

## 任务

1. `KeyMetadata` 对象预构造并复用（同 group 的 metadata 不变）
2. 用 numpy 批量生成 key 字符串数组，避免逐对象构造
3. `split_layers` 逐层构造 `LayerPoolKey` 列表改成直接批量生成字符串

## 验收标准

### 1. 功能正确性
- 生成的 key 集合与改动前完全一致（集合相等）
- 现有单测全绿

### 2. 性能验证
- 该函数耗时下降（给出 profile 对比）
- 多 PP / 多 head_or_tp 组合下的收益

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- [pool_scheduler.py:247-283](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L247-L283)

## 重点关注

- 与 kv-02（key 向量化）协同，可共用批量生成工具
- PCP/DCP 通常为 1，收益主要来自 head_or_tp / pp 维度

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
