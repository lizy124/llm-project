# [Perf] layerwise 最后一层 save 同步等待推迟

> 编号：kv-14 | 维度：Perf | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

layerwise 模式下，`save_kv_layer` 在 `current_layer == num_layers - 1` 时，`while not layer_save_finished_events[num_layers-1].wait(timeout=10)` 阻塞计算线程，直到最后一层 save（L2G）提交完成。前面所有层异步，唯独最后一层同步等待。每步 forward 末尾固定阻塞一次。

## 任务

把等待推迟到 `get_finished` 或下一 step 开始时（类似 non-layerwise 的 delayed free），而不是在 `save_kv_layer` 里硬等。

## 验收标准

### 1. 功能正确性
- 最后一层 save 完成语义不变
- 不引入跨 step 的 buffer 竞态
- 现有单测全绿

### 2. 性能验证
- 计算线程在最后一层不再阻塞（profiler 时间线）
- 端到端延迟下降

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- [pool_worker.py:1731-1740](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1731-L1740)

## 重点关注

- 推迟等待需保证下一 step 开始前最后一层 save 已完成（否则 prefill 数据未就绪）
- 与 kv-13（非-layerwise wait_for_save 异步化）同主题，模式可对齐

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
