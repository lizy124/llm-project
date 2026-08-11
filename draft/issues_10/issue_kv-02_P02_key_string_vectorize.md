# [Perf] Key 字符串生成向量化/下沉

> 编号：kv-02 | 维度：Perf | 严重程度：高 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

`PoolKey.to_string()` / `LayerPoolKey.to_string()` 用 f-string 逐块拼接，是 Python 层热路径主要开销（GIL 文档已确认）。现有 `_key_prefix_cache` 只覆盖部分路径。主调用点 `process_token_key_strings_with_block_ids` 应批量化。

## 任务

1. 参考 `LayerBatchBuilder` 已有的 numpy 向量化模式，把 key 生成整体批量化
2. 评估把 key 生成下沉到 C++ 扩展（配合 kv-22 to_string 统一与 kv-20 config_data 拆分）
3. `process_token_key_strings_with_block_ids` 改为批量生成字符串数组

## 验收标准

### 1. 功能正确性
- 生成的 key 字符串与改动前逐字符一致
- 现有单测全绿

### 2. 性能验证
- Python 层 key 生成耗时下降（给出 profile 对比）
- 长 prompt（32K token）下的端到端收益

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- `PoolKey.to_string`：[config_data.py:114-124](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L114-L124)
- `LayerPoolKey.to_string`：[config_data.py:161-171](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L161-L171)
- 前缀缓存（部分覆盖）：[config_data.py:284-306](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L284-L306)
- 参考向量化模式：[kv_transfer.py:101](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L101)
- 主调用点：[config_data.py:574-600](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L574-L600)

## 重点关注

- 与 kv-20（config_data 拆分 → keys.py）协同：先拆 keys.py 再做向量化
- 与 kv-22（to_string 统一）协同：注意 pp_rank 有/无差异
- C++ 下沉需评估跨团队成本

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
