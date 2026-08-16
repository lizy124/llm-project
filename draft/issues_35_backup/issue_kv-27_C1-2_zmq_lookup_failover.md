# [Correctness] ZMQ lookup server/client 失效保护，规避永久阻塞

> 编号：kv-27 | 维度：Correctness | 严重程度：高 | 建议优先级：**P0**
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

非-layerwise 模式下，scheduler 通过 ZMQ REQ-REP 向 worker rank0 查询 KV 命中。当前 `LookupKeyServer` 主循环**无异常保护**，`LookupKeyClient.recv()` **无 timeout**。一旦 server 异常退出或报文损坏，client 会永久阻塞在 `recv()`，整条 lookup 链路卡死，生产级风险。

## 任务

为 ZMQ lookup 链路加失效保护：
1. **server 侧**：主循环加 try/except，异常时返回错误响应而非退出循环
2. **client 侧**：`recv()` 改用 `socket.poll(timeout)` + 超时重试 / 降级（返回"未命中"或抛可恢复异常）
3. 必要时支持 socket 重建（连接断开后可恢复）
4. 报文校验：损坏报文不致 client 挂死

## 验收标准

### 1. 功能正确性
- server 抛异常时不再退出循环，client 收到错误响应并按降级处理
- client 在 server 无响应时能在 timeout 内脱困（不永久阻塞）
- socket 断开后可重建并恢复后续 lookup

### 2. 回归保护
- 现有单测全绿
- 新增单测：server 异常 / client 超时 / 报文损坏 三类场景

### 3. 交付件
- PR + 设计说明（超时策略 + 降级语义）
- 单测覆盖三类失效场景

## 证据

- server 循环无保护：[ascend_store_connector.py:312-334](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L312-L334)
- client recv 无 timeout：[pool_scheduler.py:1175-1178](file:///D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L1175-L1178)

## 重点关注

- 与 kv-15（ZMQ lookup 超时+批合并，性能向）有重叠：本 issue 聚焦正确性失效保护，kv-15 聚焦性能批合并，可合并实现但 issue 分开追踪
- 降级语义需明确：超时是返回"全未命中"还是抛异常触发上层重算？建议优先返回未命中 + 告警，避免阻塞计算流
- timeout 值需可配置（纳入 kv-24 KVPoolConfig）

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
