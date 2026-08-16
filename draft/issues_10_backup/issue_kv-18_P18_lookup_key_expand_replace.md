# [Perf] lookup key 展开避免字符串 replace

> 编号：kv-18 | 维度：Perf | 严重程度：中 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

`_expand_lookup_keys_by_rank()` 对每个 key 逐 rank 做两次字符串 replace，属于 lookup 热路径上的额外 Python 字符串复制。TP/PP 较大时更明显。

## 任务

按目标 rank 直接构造 key prefix，或缓存 `(group_id, pp_rank, head_or_tp_rank, cache_family)` 的前缀，避免"先生成后 replace"。

## 验收标准

### 1. 功能正确性
- 展开后的 key 与 replace 方式逐字符一致
- 现有单测全绿

### 2. 性能验证
- lookup key 展开耗时下降（profile 对比）
- TP/PP 较大时收益

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- [pool_worker.py:2222-2229](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2222-L2229)
- [pool_worker.py:2211-2220](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2211-L2220)

## 重点关注

- 与 kv-02（key 向量化）协同：可共用前缀缓存工具
- 前缀缓存需注意 invalidation（config 变化时）

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
