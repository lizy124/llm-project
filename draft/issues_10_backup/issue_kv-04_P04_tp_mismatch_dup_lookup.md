# [Perf] TP mismatch 路径重复 lookup 消除

> 编号：kv-04 | 维度：Perf | 严重程度：中 | 建议优先级：P3
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

写侧 TP mismatch 路径先 `lookup()` 查已有缓存、过滤 missing keys 后再 `put`，存在重复查询。读侧 `_build_tp_mismatch_keys_and_addrs` 也是双重循环。仅 TP mismatch 场景触发。

## 任务

1. 评估 lookup 是否可缓存（同一请求的 key 状态在短时间内稳定）
2. 考虑 put 时直接覆盖（mooncake 支持），省掉前置 lookup
3. 读侧双重循环合并为单次

## 验收标准

### 1. 功能正确性
- TP mismatch 场景输出与改动前一致
- 现有单测全绿

### 2. 性能验证
- TP mismatch 场景下 lookup/put 调用次数下降
- 端到端延迟对比

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- 写侧先 lookup 再 put：[pool_worker.py:2005-2035](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2005-L2035)
- 读侧双重循环：[pool_worker.py:1936-1967](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1936-L1967)
- 非 mismatch 对比：[kv_transfer.py:717-818](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L717-L818)

## 重点关注

- "直接覆盖"语义需确认 backend 行为一致（mooncake / memcache）
- 仅 TP mismatch 场景，优先级低于通用路径

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
