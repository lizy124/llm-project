# [Refactor] `to_string` 重复实现统一

> 编号：kv-22 | 维度：Refactor | 严重程度：低 | 建议优先级：P3
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

`PoolKey.to_string()` 和 `LayerPoolKey.to_string()` 是两段高度相似的 f-string，但差异不止 `@layer_id`：`LayerPoolKey.to_string()` 相对 `PoolKey.to_string()` **多了 `@layer_id` 且省略了 `@pp_rank`**（layerwise 下 PP 维度由 layer 顺序保证）。直接"复用父类前缀 + 追加 layer_id"行不通——父类前缀含 pp_rank。`__hash__` 同样受影响（`PoolKey.__hash__` 含 pp_rank，`LayerPoolKey.__hash__` 不含）。

## 任务

抽公共 key builder 时显式处理 pp_rank 的有/无（PoolKey 含、LayerPoolKey 不含），不能简单继承前缀；或整体重构为统一 key builder（配合 kv-02）。

## 验收标准

### 1. 功能正确性
- 改动后两类 to_string 输出与改动前逐字符一致
- `__hash__` / `__eq__` 行为不变
- 现有单测全绿

### 2. 交付件
- PR + 设计说明 + 单测（含 hash 一致性）

## 证据

- [config_data.py:114-124](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L114-L124)
- [config_data.py:161-171](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L161-L171)

## 重点关注

- 会增加 kv-02 向量化改造的维护成本，建议与 kv-02 协同
- pp_rank 有/无是核心差异点，hash 也受影响

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
