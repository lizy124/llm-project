# [Perf] GVA 分配循环对象构造批量化

> 编号：kv-05 | 维度：Perf | 严重程度：低 | 建议优先级：P3
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

GVA 分配循环里反复构造 key、查询 `_allocated_gvas`、append 列表。仅 GVA layerwise + memcache 场景触发，规模有限。

## 任务

批量化构造 key，用 numpy 数组替代 Python list append，减少对象构造。

## 验收标准

### 1. 功能正确性
- 分配结果与改动前一致
- 现有单测全绿

### 2. 性能验证
- 该循环耗时下降（profile 对比）

### 3. 交付件
- PR + 设计说明 + 单测

## 证据

- [pool_worker.py:1264-1293](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1264-L1293)

## 重点关注

- 与 kv-08（GVA 元数据 RPC 合并）可协同：合并后本循环规模进一步下降
- 严重程度低，可作为 kv-08 的附带优化

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
