# [Perf] `block_hash_to_str` 重复转换 3 次

> 编号：kv-10 | 维度：Perf | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

`_alloc_gvas_for_save`（save 侧）里，同一个 `group_block_hashes[block_idx]` 在 candidate_keys 列表推导、while 循环、for 循环里被 `block_hash_to_str`（`.hex()`）转换 3 次，产生三个内容相同的临时 str。load 侧 key 构造模式相同但只转换 1 次，无此冗余。

## 任务

在 `_alloc_gvas_for_save` 入口处一次性把所需范围的 `group_block_hashes` 转成 `list[str]`，后续循环复用。

## 验收标准

### 1. 功能正确性
- 生成的 key 与改动前一致
- 现有单测全绿

### 2. 性能验证
- save 侧 hex 转换次数从 3× 降到 1×（profile 对比）
- 长 prompt（block 数大）下收益

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- save 侧 3 次转换：[pool_worker.py:1243-1268](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1243-L1268)
- load 侧仅 1 次（对比）：[pool_worker.py:1422-1425](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1422-L1425)

## 重点关注

- 与 kv-08（GVA 元数据 RPC 合并）可协同：合并重构时顺手消除

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
