# [Ext] backend_map 支持外部注册

> 编号：kv-32 | 维度：Ext | 严重程度：低 | 建议优先级：P3
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

`backend_map` 字典硬编码三个后端（mooncake / memcache / yuanrong），新增后端需改源码。影响第三方扩展。

## 任务

支持 entry_point 或注册函数机制，让外部包能注册新后端。

## 验收标准

### 1. 功能正确性
- 改动后三内置后端行为不变
- 外部注册的后端可被正确加载
- 现有单测全绿

### 2. 交付件
- PR + 设计说明 + 单测（含外部注册示例）

## 证据

- [backend/__init__.py:17-30](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py#L17-L30)

## 重点关注

- 目前后端较少，紧迫度低
- 与 kv-31（Backend 抽象拆分）协同：拆分后注册机制更清晰

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
