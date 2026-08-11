# [Perf] `_handle_stored_request` 双重建+三遍遍历

> 编号：kv-12 | 维度：Perf | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

非-layerwise 写侧对同一 block 数据遍历三遍：①append 构造 5 个列表（starts/ends/keys/block_hashes/key_block_ids）；②`lookup` 后用 5 个列表推导按 `missing_indices` 过滤重建 5 个新列表、旧列表丢弃；③再遍历一次构造 `addrs`/`sizes`。合计 10 次部分拷贝。每 forward 每请求每 group 触发。

## 任务

一次遍历同时完成 lookup 过滤和 addrs/sizes 构造，用 numpy mask 一次性索引，消除中间重建。

## 验收标准

### 1. 功能正确性
- 构造的 keys/addrs/sizes 与改动前一致
- 现有单测全绿

### 2. 性能验证
- 该函数拷贝次数下降（profile 对比）
- 每 forward 每请求每 group 的开销下降

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- [kv_transfer.py:766-890](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L766-L890)

## 重点关注

- 与 kv-07（非-layerwise I/O 合并）协同：合并后遍历对象减少，本优化收益叠加
- numpy mask 方案需注意 missing_indices 的稀疏性

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
