# 07. 查询路径：ZMQ、直连、可达性与本地命中

基线与源码入口见 [00](../00_source_baseline.md)。主要符号为 `KVPoolScheduler.get_num_new_matched_tokens`、`KVPoolWorker.lookup_scheduler`、`AscendStoreCoordinator.find_reachable_hit_tokens`。

## 1. 现行分流

| 模式 | 调用链 | 本地命中是否并入查询 |
|---|---|---|
| 非 layerwise、coordinator 可用 | ZMQ→Worker `_lookup_with_coordinator` | 传入 `hbm_hit_tokens`，合并本地覆盖 |
| 非 layerwise fallback | ZMQ→Worker 连续/断续位置计算 | 不消费本地水位构造并集 |
| GVA layerwise、hybrid | Scheduler `_lookup_layerwise_with_coordinator` | 查询 callback 提供远端可读块，不传 HBM 水位 |
| GVA layerwise、简单布局 | Scheduler `_lookup_layerwise_contiguous` | 外部连续前缀，逐 group 取 min |
| Mooncake block-key hybrid | Scheduler coordinator | group 可达性 + all-rank/stage readable |
| Mooncake block-key 简单布局 | Scheduler `_lookup_block_key_contiguous` | 外部连续前缀 |
| 其他 layerwise | Scheduler `_get_store_lookup_hit_tokens` | 普通 layer keys 存在性前缀 |

因此旧的“layerwise 一律只查存在性、不用 coordinator”已不成立。判断模式至少要看 layerwise、data plane 和 hybrid。

## 2. ZMQ 到底传什么

非 layerwise 的 `LookupKeyServer` 仅在 Worker role、全局 rank 0 启动。Scheduler 懒建 `LookupKeyClient`，请求含：
- token length。
- group ids。
- 本地 `hbm_hit_tokens`。
- block hashes。

Worker 调用 `lookup_scheduler()`，返回一个 **4 字节 big-endian 命中长度整数**。它不传 KV 数据、不传真实 tensor pointer，也不返回逐 block 位图。

server handler 失败记录日志并回 0；Worker lookup 内部也有异常保守回 0 的路径。故 lookup 的零命中既可能是正常 miss，也可能是查询异常，需要结合日志；这与传输线程 fatal 异常在同步点抛出不同。

client/server 有显式 close；server poll 循环允许停机，不能遗留一直占用 socket 的服务线程。

## 3. 为什么当前普通查询放在 Worker

**代码事实**：
1. Scheduler 即使在非 layerwise 模式也创建 `store_scheduler`。
2. 两侧拿到 group/cache 配置，key 的逻辑身份可以各自推导。
3. 普通 chunk 枚举、连续/断续 hit 计算、跨 rank/group 组合当前组织在 Worker。
4. ZMQ 只回聚合整数，因此当前跨块合并在应答侧完成。

由此可得的解释是：**现有代码把普通命中计算放在 Worker，Scheduler 通过 RPC 复用这套实现。** 这不是“Scheduler 永远无法得到信息”的架构定理。layerwise hybrid 已在 Scheduler 构造 coordinator，本身就是反例。

把这段逻辑搬到 Scheduler 在信息上并非不可能，但需迁移相同的 key、mask、rank 和命中语义。本文没有据此宣称某次历史 PR 的设计动机，也不把性能收益作为未经测量的事实。

## 4. 普通 coordinator 的本地∪远端模型

`_lookup_with_coordinator()`：
- 仅接受适用的完整 group 集合。
- 按 group page 将本地水位向下对齐，查询其后的远端 keys。
- callback 返回远端存在的 chunk hashes。
- `find_reachable_hit_tokens(..., hbm_hit_tokens=...)` 补入本地已可用前缀，求共同可恢复位置。

这里的并集语义是“本地已可用或能从池恢复”，不是“本地有，所以池也有”。

推导例，假设全是普通连续页：
```text
本地: b0 b1 b2 b3 b4
池中: b0 b1 b2       b5 b6 b7 b8 b9 b10
并集: b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10
```
外部自身连续前缀只有 3 页，但合并后可到 11 页。只取 `max(local_prefix, remote_prefix)` 会遗漏 b5–b10 的可恢复价值。

fallback 没有这类本地并集，按 group 是否 align-state 选连续/断续算法，再求共同命中位置。纯 attention 中间缺页必须截断；某些状态 group 本就稀疏，不能机械要求每页存在。

## 5. Layerwise hybrid：存在性必须服从可达性

Scheduler 也有 coordinator。`_lookup_layerwise_with_coordinator()` 根据候选 token 范围生成 lookup masks，只查可达块，然后用通用 `find_reachable_hit_tokens()` 求 hit。

每个逻辑块的 key 集合必须覆盖协议要求的：
```text
group / page identity
  × head 或有效 TP ranks
  × PP stages
  × DSpark draft companion（启用时）
```

`_query_layerwise_block_hits()` 将 `batch_is_readable` 的结果折叠为 block hit：
- 返回长度必须等于 key 数。
- 每项必须是真正 bool。
- 同一 block 的所有必需 key 都可读才算 hit。

对 SWA/压缩/状态 group，要查的是该恢复位置所需的稀疏可达对象，不是从 b0 开始每页都存在。否则会把正确保存的稀疏状态永久判 miss。

## 6. 简单 layerwise 前缀与短期缓存

无 coordinator 的 GVA 简单路径从 block 0 走外部连续前缀，各 group 命中换成 token 后取 min。Mooncake 简单 block-key 路径也检查从头的 committed block prefix。

GVA contiguous 路径的 `_lw_block_hit_cache` 当前：
- miss TTL 30 秒。
- hit TTL 0.5 秒。
- 最大 200,000 项，超限清理。
- 物理 reuse 或 DSpark 联合前缀场景绕过跨请求缓存，每次重查远端可读性。

这些是该具体分支的实现，不能扩展成“所有 layerwise 查询都缓存”。短 hit TTL 只缩小淘汰竞态窗口，**不提供 lease 或强一致性保证**；真正 get 失败仍必须正确处理。缓存 miss 过期前可能少报新保存的前缀，代价是额外计算。

## 7. 查询起点与实际 load 起点是不同问题

| 问题 | 决定什么 |
|---|---|
| 查询从哪开始 | 如何计算外部恢复范围 |
| load 从哪开始 | 哪些本地内容必须写回 |
| 保存从哪开始 | 已有远端对象与本轮增量水位 |

对 GVA：
- 无 reuse 或独立层，load 通常可跳过本地 cached prefix。
- 共享 buffer 层，历史数据已可能被其他层覆盖，load 必须恢复其需要的历史范围。
- 即使外部新增 allocation 为 0，仍可能需要 layerwise load metadata。

对 block-key：
- 根据 load session 的 committed keys 和 slots 构造 load ranges。
- chunked prefill 后续 step 可使用之前 commit 的对象。
- 不套用 GVA 的共享 slot 公式；当前 Mooncake protocol 不启用该物理 reuse。

## 8. 复用的容量收益与流量代价

物理 reuse 后，一个 slot 只保留最后 owner 的内容，prefix hash 账本不能证明所有共享层仍驻留。必须依赖外部完整对象、partial snapshot 与轮转恢复。

推导上，decode 多 step 也可能不断产生历史 KV load/save。实际流量取决于 layer ownership、mask、完整/partial 水位和已存在对象过滤，不能把“每层每步全量保存”写成所有模式的固定行为。

非 layerwise 默认也不是“等请求结束才存”；它在相应 step 的 `wait_for_save()` 提交增量 batch。复用与保存时机必须分别读执行体。

## 9. 用查询日志定位问题

按下面顺序记录同一 request：
1. prompt/token 数、local watermark、hash/page/granularity。
2. 实际 data plane、hybrid/coordinator、consumer load 开关。
3. Scheduler/Worker 生成的真实 key sample、PP namespace、group id。
4. raw exists/readable 返回、mask、group hit、最终水位。
5. Eagle/最后 token 修正后的 allocation，以及 `LoadSpec.can_load`。
6. 实际 load keys/slots、失效记录与最终 Scheduler 失败处理。

症状与测试映射见 [09](09_observability_testing.md)；组合场景见 [08](08_composition_dspark_kvpp.md)。
