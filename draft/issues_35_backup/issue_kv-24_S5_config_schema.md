# [Refactor] 配置项集中 schema（KVPoolConfig）

> 编号：kv-24 | 维度：Refactor | 严重程度：高 | 建议优先级：**P0**
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

15 处 `extra_config.get` 散落在 3 个文件（pool_worker 9 处、pool_scheduler 5 处、config_data 1 处），无集中定义。新增配置项时无法发现已有哪些，易冲突或重复。这是后续重构（S1 拆分、S2 初始化去重）的前置基础。

## 任务

定义 `KVPoolConfig` dataclass 集中所有配置项（含默认值、类型、文档），scheduler/worker 都从它读取。这也解决 kv-21 的重复初始化问题。

## 验收标准

### 1. 功能正确性
- 所有配置项默认值与改动前一致
- 改动后读取到的配置值与改动前一致
- 现有单测全绿

### 2. 代码质量
- 配置项单一来源，带类型注解 + docstring
- 新增配置项只改一处
- 散落的 `extra_config.get` 全部收敛到 KVPoolConfig

### 3. 交付件
- PR + 设计说明 + 配置项清单 + 单测

## 证据

- 散落统计见概览表：pool_worker.py 9 处、pool_scheduler.py 5 处、config_data.py 1 处
- 典型散落点：[pool_worker.py:145-177](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L145-L177)、[pool_scheduler.py:93-105](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L93-L105)

## 重点关注

- 是 kv-21（初始化去重）的前置
- timeout / prefetch 等可调参数收敛后，kv-06 / kv-15 / kv-27 的配置读取统一
- 注意 backward compat：外部传入的 extra_config 结构不变

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
