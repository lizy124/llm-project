# [Ext] 高风险路径测试覆盖补强

> 编号：kv-33 | 维度：Ext | 严重程度：中高 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

`kv_pool` 不是"没有测试"，而是**已有测试**（`tests/ut/distributed/ascend_store/` 下有 `test_config_data.py` / `test_pool_worker.py` / `test_pool_scheduler.py` / `test_kv_transfer.py` / `test_coordinator.py` 等），但**对高风险路径的覆盖仍不均衡**。从源码复核看，异常路径、GVA 批量元数据、multi-group 失败恢复、异步 load/save 状态机仍需补强。

## 任务

优先补"会卡死、会吞错、会错标完成"的高风险路径测试，而不是只补纯逻辑函数的 happy path：
1. `KVTransferThread.run()` 异常分支与资源清理（配合 kv-25）
2. GVA 元数据批量路径（配合 kv-08）
3. ZMQ lookup 的超时/异常/重连路径（配合 kv-27）
4. layerwise / non-layerwise 的同步等待与失败传播（配合 kv-13/kv-14/kv-28）

## 验收标准

### 1. 功能正确性
- 新增测试不改变生产行为
- 现有测试全绿

### 2. 覆盖指标
- 上述高风险路径有针对性测试（异常注入 / 并发 / 失败传播）
- 覆盖率报告显示相关路径覆盖提升

### 3. 交付件
- PR + 新增测试文件 + 覆盖率对比

## 证据

- `tests/ut/distributed/ascend_store/` 下已存在相关单测文件
- 待补强路径：
  - `kv_transfer.py:496-518`（线程异常分支）
  - `pool_worker.py:1359-1534`（GVA 元数据批量）
  - `pool_scheduler.py:1146-1178`、`ascend_store_connector.py:293-340`（ZMQ lookup）
  - `pool_worker.py:1670-1763`（同步等待与失败传播）

## 重点关注

- **是 kv-07 / kv-08 / kv-20 / kv-21 等重构的前置**——没有测试网重构风险极高
- 建议作为第一批落地项，为后续重构铺安全网
- 优先级 P1

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
