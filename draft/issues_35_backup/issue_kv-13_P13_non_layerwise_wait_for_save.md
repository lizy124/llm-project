# [Perf] 非-layerwise `wait_for_save` 异步化

> 编号：kv-13 | 维度：Perf | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

非-layerwise 模式下，`wait_for_save` 提交所有 save task 后调 `send_thread.request_queue.join()`，**阻塞计算线程直到 send 线程处理完所有 save**——"计算流阻塞等 I/O"的典型点。每步 forward 后的主要阻塞点。

```python
if current_event is not None:
    send_thread.request_queue.join()   # 阻塞直到所有 save task 处理完
```

## 任务

改为异步：不 join，在后续 step 的 `get_finished` 里检查 save 是否完成（delayed free 机制已部分支持，但 `wait_for_save` 仍硬 join）。

## 验收标准

### 1. 功能正确性
- save 完成语义不变（数据最终被持久化 / 可被后续 get 命中）
- 不引入资源越界 / buffer 复用竞态
- 现有单测全绿

### 2. 性能验证
- 计算线程在 `wait_for_save` 处不再阻塞（profiler 时间线）
- 端到端吞吐提升

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- [pool_worker.py:1761](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1761)

## 重点关注

- 异步化需保证 save 未完成时 buffer 不被覆盖（delayed free / 双缓冲）
- 与 kv-14（最后一层 save 同步等待）同主题
- 与 kv-07（I/O 合并）协同：合并后 save 数减少，join 阻塞也减轻

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
