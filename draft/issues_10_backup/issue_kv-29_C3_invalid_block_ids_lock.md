# [Correctness] `_invalid_block_ids` 锁保护范围审计

> 编号：kv-29 | 维度：Correctness | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

`_invalid_block_ids` 有锁保护，但读取它的路径（如 `get_block_ids_with_load_errors`）是否都在锁内未核实。若存在锁外读取，可能与 recv 线程的并发写入产生竞态，导致返回不一致的 invalid 集合（漏报或重复）。

## 任务

1. 审计所有 `_invalid_block_ids` 访问点（读 / 写），确认是否都在锁内
2. 锁外的读取改为锁内读取或返回快照副本
3. 必要时把锁的粒度 / 类型文档化

## 验收标准

### 1. 功能正确性
- 所有 `_invalid_block_ids` 读写均在锁保护内（或通过不可变快照返回）
- 并发场景下 `get_block_ids_with_load_errors` 返回一致结果

### 2. 回归保护
- 现有单测全绿
- 新增并发读写单测（多线程同时 load 失败 + 查询）

### 3. 交付件
- PR + 审计清单（每个访问点的锁状态）
- 单测覆盖并发场景

## 证据

- 锁定义：[pool_worker.py:145-152](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L145-L152)
- 传给 recv 线程：[pool_worker.py:570-571](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L570-L571)

## 重点关注

- `get_block_ids_with_load_errors` 是否被计算流热路径调用（若是，锁竞争需评估）
- 考虑用 `copy` 快照降低锁持有时间

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
