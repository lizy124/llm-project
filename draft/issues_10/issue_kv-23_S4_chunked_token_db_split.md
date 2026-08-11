# [Refactor] `ChunkedTokenDatabase` 职责拆分

> 编号：kv-23 | 维度：Refactor | 严重程度：中 | 建议优先级：P3
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

`ChunkedTokenDatabase` 承担了 token 分块、key prefix 缓存、buffer/stride 管理、layer cache 准备等多职责。

## 任务

拆分为：
- `TokenChunker`（token → chunk 映射）
- `KeyBuilder`（chunk → key）
- `BufferLayout`（key → addr/size）

## 验收标准

### 1. 功能正确性
- 拆分后行为完全不变
- 现有单测全绿

### 2. 交付件
- PR + 设计说明 + 单测

## 证据

- [config_data.py:255-467](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L255-L467)

## 重点关注

- 与 kv-20（config_data 拆分）协同：本 issue 是 config_data 拆分的细化子项
- 与 kv-02（key 向量化）协同：KeyBuilder 拆出后向量化更聚焦

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
