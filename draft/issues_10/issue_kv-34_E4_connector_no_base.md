# [Ext] Connector 公共基类评估

> 编号：kv-34 | 维度：Ext | 严重程度：低 | 建议优先级：P3
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

4 个 connector（ascend_store / recompute_cpu_offload / simple_cpu_offload / ucm_connector）各自实现 `KVConnectorBase_V1`，但无 kv_pool 层面的公共基类抽取公共逻辑。

## 任务

评估是否有可抽取的公共逻辑（如配置解析、event 聚合）。若各 connector 差异大，可不强行抽取，但应文档化各 connector 的定位和边界。

## 验收标准

### 1. 交付件
- 评估报告：各 connector 职责对比 + 是否抽取的结论
- 若抽取：PR + 公共基类 + 单测
- 若不抽取：文档化各 connector 定位与边界

## 证据

- 4 个 connector 入口（ascend_store / recompute_cpu_offload / simple_cpu_offload / ucm_connector）无共同父类

## 重点关注

- 4 个 connector 差异较大，强行抽取可能过度设计
- 若评估结论是不抽取，本 issue 以文档交付收尾

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
