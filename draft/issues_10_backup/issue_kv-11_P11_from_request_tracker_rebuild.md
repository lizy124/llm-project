# [Perf] `from_request_tracker` 增量 buffer 替代全量重建

> 编号：kv-11 | 维度：Perf | 严重程度：中高 | 建议优先级：P1/P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

scheduler 每 step 对每个 request 调 `from_request_tracker`，把**全量** `allocated_block_ids` 用 `np.asarray` 重新拷贝成新 ndarray。长 prompt（32k token / 16 block_size = 2000 blocks）下，每步每请求 O(N) 拷贝，scheduler 热路径显著。

## 任务

在 `RequestTracker` 中维护增量 numpy buffer：
- 方案 A：预分配 np 数组，`update()` 追加 block 时往预分配数组追写
- 方案 B：缓存上次转换的 (长度, 数组)，只追增量

## 验收标准

### 1. 功能正确性
- 增量 buffer 内容与全量重建逐元素一致
- block 被释放 / 回收时增量 buffer 正确同步
- 现有单测全绿

### 2. 性能验证
- scheduler 每 step 的 numpy 拷贝开销下降（profile 对比）
- 长 prompt（32K/64K）下端到端延迟收益

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- [config_data.py:1093-1100](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L1093-L1100)

```python
block_ids_np=np.asarray(tracker.allocated_block_ids, dtype=np.int64),
block_ids_by_group_np=[np.asarray(ids, dtype=np.int64) for ids in tracker.allocated_block_ids_by_group]
```

## 重点关注

- block 释放（free）时增量 buffer 的收缩语义需明确
- 多 group 的 `allocated_block_ids_by_group` 需分别维护增量
- 与 kv-20（config_data 拆分 → request_meta.py）协同

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
