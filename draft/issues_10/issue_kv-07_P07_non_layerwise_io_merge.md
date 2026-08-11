# [Perf] 非-layerwise I/O 跨 request 合并【高价值】

> 编号：kv-07 | 维度：Perf | 严重程度：高 | 建议优先级：P1
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

非-layerwise 模式下，**每个 request 单独调一次 `put`（+ 一次 `lookup`）和一次 `get`**，跨 request 完全不合并。一批 N 个 request、G 个 group → N×G 次 put + N×G 次 exists（写侧）/ N 次 get（读侧）。后端 API 原生支持扁平 list，本可一次完成。

对比：layerwise GVA 路径已正确合并（`np.concatenate` + 一次 `batch_copy` + `batch_write_finish`），**非-layerwise 路径没有对齐这个合并模式**。

## 任务

1. **写侧**：跨 request 累积 keys/addrs/sizes，一次 `lookup` + 一次 `put`（参考 GVA 路径 `batch_write_finish` 合并模式）
2. **读侧**：跨 request 拼接 key/addr/size，一次 `get`；`_circular_shift` 在合并后的总 list 上做
3. **异步读侧**：改为 drain 队列里所有就绪 request 一次性处理

## 验收标准

### 1. 功能正确性
- 合并后各 request 拿到的数据与逐个调用一致
- `_circular_shift` / `store_mask` / `skip` per-request 差异正确处理
- 现有单测全绿

### 2. 性能验证
- 一批 request（4/8/16）下，put/get 调用次数从 N×G 降到 1~2
- 端到端延迟对比曲线
- 同批 request 共享 `current_event`，合并后 `synchronize()` 只需一次

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- 写侧 per-(request,group) lookup：[kv_transfer.py:809](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L809)
- 写侧 per-(request,group) put：[kv_transfer.py:890](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L890)
- 读侧同步 per-request get：[pool_worker.py:971](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L971)
- 读侧异步 per-request get：[kv_transfer.py:997](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L997)
- 入队逐 request：[pool_worker.py:1749-1762](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1749-L1762)
- GVA 合并参考：[kv_transfer.py:1419-1429](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1419-L1429)、[kv_transfer.py:1444-1452](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1444-L1452)

## 重点关注

- 合并前提：key 命名空间全局唯一（含 group_id + block_hash + rank），跨 request 合并无冲突
- `store_mask`/`skip` per-request 不同只影响"选哪些 block"，不影响合并后的调用
- 与 kv-08（GVA 元数据 RPC 合并）模式相同，可同步推进
- 前置依赖 kv-33（补测试），重构需测试网兜底

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
