# [Perf] 设计 offset-aware ZMQ lookup 协议，避免发送已确认的 HBM 前缀 hashes

> 编号：kv-17 | 维度：Perf | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

scheduler 当前把完整 `request.block_hashes` 转成 bytes 列表，经 msgpack/ZMQ 发给 worker。对于 hash block 较小、prompt 很长且本地 HBM 已确认大段前缀命中的请求，这会增加编码和 IPC payload。

不能直接发送 `block_hashes[k:]`：worker 的 chunk/key 构造以 hash 列表从 token 0 开始为前提；hybrid coordinator 和普通 non-hybrid fallback 对 `hbm_hit_tokens` 的处理也不同。协议必须携带绝对 offset，并让 worker 按绝对 token 位置恢复 grouped hash、cache-family 和返回 hit 长度语义。

## 任务

1. 定义带版本的 lookup request，至少包含 `hash_start_token`（或等价绝对 hash index）、suffix hashes、token_len 和 HBM hit baseline；保留旧报文兼容或明确协议升级策略。
2. scheduler 仅在起点满足 hash block/group effective block 对齐要求时裁剪；无法安全裁剪时退回完整报文。
3. worker 的 non-hybrid 和 hybrid 路径都按绝对 token 位置构造 grouped keys/masks，并把 HBM 前缀作为命中基线，不能把 suffix 当成 token 0。
4. 保持 multi-group、compress/cache-family、partial block、TP/PP expansion 和 coordinator mask 语义。
5. 合并 kv-18 的有效部分：重做协议和 key 构造时，可按 `(group, role, family, pp_rank, head_or_tp_rank)` 直接生成或缓存 immutable prefix，避免当前字段切片替换产生的中间字符串；该局部优化必须用微基准验证，不单独定级。

## 验收标准

### 1. 功能正确性
- 新旧协议在相同请求上的 hit token 结果逐项一致
- 覆盖 non-hybrid/hybrid、multi-group、不同 hash/group block size 和多个 cache family
- 覆盖 HBM hit 为 0、非对齐、对齐、全长，以及 suffix 为空
- TP/PP all-rank 展开的 key 与当前实现逐字符一致
- malformed/version-mismatch 报文按 kv-27 的错误协议处理，不得永久阻塞
- 现有单测全绿

### 2. 性能验证
- 报告 16K/64K prompt、不同 HBM hit ratio 下的 payload bytes 和 msgpack 编解码耗时
- 报告 worker key 构造、all-rank expansion 和端到端 lookup latency
- 数据不足以证明收益时不得默认启用裁剪协议

### 3. 交付件
- PR + 设计说明 + 性能数据 + 单测

## 证据

- scheduler/client framing：`pool_scheduler.py:565-575, 1168-1187`
- server framing：`ascend_store_connector.py:312-325`
- worker lookup 与 all-rank key expansion：`pool_worker.py:2219-2243, 2245-2330, 2339-2390`
- token chunk absolute-position 语义：`metadata.py:468-524, 641-671`

## 重点关注

- offset 语义必须由 scheduler、wire protocol 和 worker 共同定义
- 与 kv-27 的 timeout、typed response 和 socket recovery 协同，但不能阻塞正确性修复
- 普通 non-hybrid 路径当前不使用 `hbm_hit_tokens`，必须单独设计
- 长 prompt / 高频 lookup 场景更明显
- 本任务已吸收原 kv-18 的 all-rank key prefix 构造优化

## 环境约定
- vllm-ascend：最新 main
- 硬件：Ascend NPU（注明型号 + 卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
