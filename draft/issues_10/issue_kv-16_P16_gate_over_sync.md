# [Perf] gate 对非复用预取层过度同步

> 编号：kv-16 | 维度：Perf | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

layerwise 预取层（`layer_id != current_layer`）的 load task 一律携带 `attention_start_gate`，recv 线程在 `_handle_request` 里 `gate.wait()` → `event.synchronize()` 阻塞直到计算流到达 attention 边界。但对**非 buffer 复用**的预取层（`prefetch_layer_map` 无该层），不存在共享 buffer 数据竞争，gate 是多余的——load 可以立即开始。

**影响**：单 recv 线程被 gate 阻塞会卡住后续所有层的 load（recv 线程串行处理队列）。

## 任务

按"是否有 reuse_source"决定是否加 gate：非复用层不加 gate，立即 load；复用层保留 gate 保证 buffer 数据安全。

## 验收标准

### 1. 功能正确性
- 复用层 gate 行为不变（buffer 数据安全）
- 非复用层 load 立即开始，无数据竞争
- 现有单测全绿

### 2. 性能验证
- recv 线程被 gate 阻塞时间下降（profiler 时间线）
- 预取层就绪更早，attention 前等待减少

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- gate 附着条件：[pool_worker.py:1670-1672](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1670-L1672)
- gate wait：[kv_transfer.py:1597-1599](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1597-L1599)
- gate 实现（硬阻塞）：[memcache_comm_fence.py:53-61](file:///D:/lzy/project/kv_pool/code/vllm-ascend/memcache_comm_fence.py#L53-L61)

## 重点关注

- GVA 复用场景 gate 是正确性需要，不能去掉
- 与 kv-06（prefetch 默认值）联动：预取层增多时 gate 精细化收益更大

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
