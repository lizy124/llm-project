# kv_pool 优化点 → 社区 Issue 草稿索引

> 来源：原始 35 项候选及源码审核记录；历史草稿可从 Git 历史追溯
> 参考格式：vllm-ascend issue #13745 / #13746 / #13747（测试任务型 issue）
> 目标仓库：`vllm-project/vllm-ascend`，代码路径 `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store`
> 审核基线：main @ `d5e9816065ede613327d93908f87fee9f5c47128`
> 验收人：@赵鹏博
> 关联任务池：[#9079 [Contribution] vLLM-Ascend 外部开发者任务池](https://github.com/vllm-project/vllm-ascend/issues/9079)
> 发布日期：2026-08-11
> 说明：原有 35 个候选 issue；经源码审核、任务合并和二次优先级筛选，最终形成以下 10 个发布候选。历史筛选记录见 [check.md](check.md)，当前代码严格复核见 [CODE_AUDIT.md](CODE_AUDIT.md)，开发价值与实施顺序见 [VALUE_RANKING.md](VALUE_RANKING.md)，当前优先发布稿见 [TOP3.md](TOP3.md)。

## 最终筛选规则

- 只保留能由当前源码和调用链证明的问题；性能收益仍需硬件数据验证。
- P0/P1 优先覆盖可能导致永久等待、残缺 KV 进入 forward 或 transfer 状态失控的问题。
- 将可随主任务完成的局部优化并入对应 issue：kv-10 → kv-08，kv-12 → kv-07，kv-21 的 rank-invariant 派生部分 → kv-24。
- 审核认为“可修改后发布”但本轮未入选的 kv-29，仅有预防性低优先级价值；不占用最终 10 个名额。
- kv-35 不再预设 IPC 解法；只有 profile 证明 Python/GIL 是关键瓶颈后，才另立 benchmark/RFC。

## 通用约定

- **代码基线**：提交前需在最新 main 复核证据位置和行为
- **硬件**：Ascend NPU（提交时注明型号、卡数、TP/CP 和 backend）
- **交付件**：PR + 设计说明 + 对应单测
- **性能任务**：先完成 Phase 0 基线和 Go/No-Go 判断；必须附调用次数、局部耗时和端到端 TTFT/吞吐数据，不预设收益。数据不支持改动时，以验证报告和 No-Go 结论收口，不强行修改热路径
- **正确性任务**：必须注入失败并验证状态传播、资源清理及其他请求隔离
- **外部契约**：MemCache lease、backend 返回码、ZMQ timeout 等无法由本仓库静态证明的语义，必须注明实际版本并通过官方契约或隔离实验确认
- **事实边界**：区分“代码可证明的调用/控制流”“机器上观察到的行为”和“待验证假设”，不得把可能收益或未来扩展风险写成已发生故障
- **回归红线**：现有单测全绿；greedy / non-greedy 输出一致；等待路径无永久挂起

## 候选清单

### 正确性与可靠性（4 个）

| 编号 | 文件 | 审核后任务边界 | 优先级 |
|------|------|----------------|--------|
| kv-28 | [hybrid KV load failure](issue_kv-28_C2_multi_group_failure.md) | hybrid KV load 失败不得使用残缺 KV 进入 forward，并建立 request-level 失败传播 | P0 |
| kv-27 | [ZMQ lookup recovery](issue_kv-27_C1-2_zmq_lookup_failover.md) | 完整处理 REQ/REP timeout、error reply、协议校验和 socket 重建 | P0 |
| kv-25 | [transfer thread fatal protocol](issue_kv-25_C1_transfer_thread_exception.md) | 建立 transfer thread 终止式失败协议及幂等资源清理 | P1 |
| kv-26 | [Backend.put result contract](issue_kv-26_C1-1_backend_put_failure.md) | 先核对三种 backend 实际返回语义，再统一 AVAILABLE/FAILED/UNKNOWN 结果与 KV event | P2 |

### 性能（4 个）

| 编号 | 文件 | 审核后任务边界 | 优先级 |
|------|------|----------------|--------|
| kv-01 | [MLA TP read dedup PoC](issue_kv-01_P01_MLA_read_dedup.md) | 先验证重复读取成本，再决定是否实现 non-layerwise leader get + TP broadcast PoC | P2 |
| kv-07 | [non-layerwise backend batching](issue_kv-07_P07_non_layerwise_io_merge.md) | 先验证各路径 batch 机会，再按路径批处理并保留逐 request/block 语义 | P2 |
| kv-08 | [GVA metadata RPC aggregation](issue_kv-08_P08_gva_meta_rpc_merge.md) | 先验证 RPC 占比和 MemCache lease 语义，再聚合安全的 metadata 操作 | P2 |
| kv-17 | [offset-aware lookup protocol](issue_kv-17_P17_zmq_lookup_full_hashes.md) | 先验证 payload/TTFT 占比，再决定是否引入 offset-aware suffix 协议 | P2 |

### 配置与接口（2 个）

| 编号 | 文件 | 审核后任务边界 | 优先级 |
|------|------|----------------|--------|
| kv-24 | [AscendStore typed config parser](issue_kv-24_S5_config_schema.md) | 在复用现有 layerwise parser/helper 的基础上统一 connector-owned extra config | P2 |
| kv-31 | [backend capability model](issue_kv-31_E1_backend_abstraction_split.md) | 清理 Backend/GVA 类型边界；定位为未来扩展保护，不声称当前已有可达错误配置 | P3 |

## 风险修复排期

> 本节按当前故障紧迫度安排实施，不等同于任务对池化能力的战略价值；池化价值排序见 [VALUE_RANKING.md](VALUE_RANKING.md)。

1. `kv-28`：阻止残缺 KV 进入 forward
2. `kv-27`：消除 lookup 永久阻塞
3. `kv-25`：建立 thread fatal 与清理协议
4. `kv-26`：统一 save 失败结果和事件语义
5. `kv-24`：集中配置解析，为 timeout 等参数提供稳定入口
6. `kv-01`：MLA non-layerwise 去重 PoC
7. `kv-07`：non-layerwise backend batching
8. `kv-08`：GVA metadata RPC aggregation
9. `kv-17`：offset-aware lookup payload
10. `kv-31`：backend capability 类型边界
