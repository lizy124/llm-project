# [Perf] ZMQ Lookup RPC 超时与批合并

> 编号：kv-15 | 维度：Perf | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-10-31

## 背景

非-layerwise 模式下，scheduler 对每个新请求调 `LookupKeyClient.lookup()`，同步 ZMQ REQ-REP，**无 timeout、无批合并**——同一 step 多个新请求各自串行 RPC。worker rank0 的 `LookupKeyServer` 单线程处理。新请求数多时累积延迟显著。

## 任务

1. 加 `socket.poll(timeout)` 超时保护
2. 同 step 多请求的 block_hashes 合并成一次批量 RPC
3. 评估用 DEALER / 多 REQ 并发

## 验收标准

### 1. 功能正确性
- 批合并后各请求拿到的命中结果与逐个 RPC 一致
- 超时降级语义明确（与 kv-27 协同）
- 现有单测全绿

### 2. 性能验证
- 同 step N 个新请求的 RPC 次数从 N 降到 1
- 新请求数多时的延迟对比

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- recv 无 timeout：[pool_scheduler.py:1175](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1175)
- per-request 调用：[pool_scheduler.py:556](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L556)

## 重点关注

- 超时保护与 kv-27（正确性失效保护）重叠：建议本 issue 做性能批合并，kv-27 做失效保护，实现可合并但 issue 分开追踪
- 与 kv-17（只发后缀 hashes）协同：payload 减小 + 批合并双重收益
- timeout 值纳入 kv-24 KVPoolConfig

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
