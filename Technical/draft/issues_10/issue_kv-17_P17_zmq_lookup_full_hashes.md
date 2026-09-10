# [Perf] 为 ZMQ lookup 设计 offset-aware 后缀 hash 协议

> 编号：kv-17 | 维度：Perf | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

Lookup client 当前会把完整 `request.block_hashes` 逐个转为十六进制字符串，经 msgpack 编码后通过 ZMQ 发送。这一代码事实已确认；对于长 prompt 且本地 HBM 已命中大段前缀的请求，完整 payload 可能产生可避免的编码和 IPC 开销，但目前没有数据证明它是 lookup 或 TTFT 的显著瓶颈。

但 worker 不能直接把 `block_hashes[k:]` 当作新列表从 token 0 处理：hybrid 与 non-hybrid 路径的 HBM 前缀语义不同，chunk/grouped-hash 计算依赖绝对 token 位置，各 group 的 effective block size 也可能不同。

## 开发前置验证（Phase 0）

1. 不改协议，先记录 16K/64K prompt、HBM 前缀命中率 0%/50%/接近 100% 时的 hash 数、wire bytes、`.hex()`/msgpack 编解码耗时、ZMQ 往返延迟和端到端 TTFT。
2. 根据现有完整 hash 列表离线计算 suffix payload 的理论上限收益，明确协议版本化、兼容和额外分支成本。
3. 预先定义 Go/No-Go 阈值。若 payload 编解码与 IPC 占比很低，或减少 payload 对 TTFT 没有可测收益，可以提交验证报告后保持现协议，不要求为完成任务引入新 wire protocol。

## 任务

1. Phase 0 达到 Go 条件后，定义版本化 lookup request，显式携带 `hash_start_token` 或等价绝对 hash offset，以及 suffix hashes。
2. worker 在 non-hybrid 和 hybrid 路径中都按绝对 token 位置构造 grouped hashes、cache-family key 和 lookup mask。
3. 已确认的 HBM 前缀作为命中基线参与最终 hit 计算，保持 multi-group、compress、coordinator mask、TP/PP key 展开和 partial block 语义。
4. 保持旧报文兼容，或提供明确的协议版本协商/拒绝策略。
5. 只有实际 payload 和端到端数据证明收益后才默认启用后缀协议。

## 验收标准

### 1. 功能正确性
- 新旧协议返回的 remote hit tokens 逐项一致
- 覆盖 non-hybrid、hybrid、多个 cache family 和 multi-group
- 覆盖 group block size 大于 hash block size，以及 offset 为 0、非对齐位置和全长
- malformed offset、hash 数量和 response 必须被拒绝，不能静默计算错误命中
- 现有单测全绿

### 2. 性能验证
- 报告 16K/64K prompt 下 payload bytes、`.hex()`/msgpack 编解码耗时和 lookup 延迟
- 分别覆盖 HBM 前缀命中率 0%、50%、接近 100% 的场景
- 提供端到端 TTFT 数据，不只报告报文大小

### 3. 交付件
- Phase 0 验证报告；达到 Go 条件时再交付 PR + wire protocol 说明 + 兼容策略 + 性能数据 + 单测

## 证据

- scheduler lookup：`pool_scheduler.py:565-575`
- client 编码与发送：`pool_scheduler.py:1168-1187`
- server/worker 入口：`ascend_store_connector.py:312-325`、`pool_worker.py:2245-2330, 2339-2390`
- token chunk/grouped hash：`metadata.py:468-524, 641-671`

## 重点关注

- 不允许用一个简单的 `hbm_hit_tokens // block_size` 下标覆盖所有 group
- 本任务聚焦 payload 协议，不与 kv-27 的 REQ/REP 故障恢复混合
- 优先级保持 P2，是否升级由长上下文实测决定

## 环境约定
- vllm-ascend：审核基线 `d5e9816065ede613327d93908f87fee9f5c47128` 或提交时最新 main
- 硬件：Ascend NPU（注明型号、卡数和缓存配置）
- 关联任务池：#9079
- 验收人：@赵鹏博
