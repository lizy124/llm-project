# [Refactor] scheduler/worker 初始化逻辑去重

> 编号：kv-21 | 维度：Refactor | 严重程度：高 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

`use_mla` 判断、`num_kv_head` 计算、`put_step` 计算、backend 加载、layerwise layout 构建在 scheduler 和 worker 两边各写一遍。一致性风险高——一边改了另一边没改就会漂移（已有 MLA 分析中就因为两边逻辑需对照确认）。

## 任务

抽取 `KVPoolConfig` 共享配置类，scheduler 和 worker 都从它读取，避免逻辑漂移。

## 验收标准

### 1. 功能正确性
- 改动后 scheduler 与 worker 的初始化结果（`use_mla`/`num_kv_head`/`put_step`/backend_map/layerwise_layout）与改动前一致
- 现有单测全绿

### 2. 代码质量
- 初始化逻辑单一来源（single source of truth）
- 新增配置项只改一处

### 3. 交付件
- PR + 设计说明 + 单测

## 证据

- worker：[pool_worker.py:188-208](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L188-L208) `_init_key_head_config`
- scheduler：[pool_scheduler.py:183-193](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L183-L193) 内联在 `__init__`
- 两边都独立做 `infer_tp_mismatch_info`、backend_map 查找、layerwise layout 构建

## 重点关注

- **前置依赖 kv-24（配置集中 schema）**：KVPoolConfig 是去重的基础
- 与 kv-20（巨文件拆分）协同

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
