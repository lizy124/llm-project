# [Refactor] AscendStore extra config typed parser

> 编号：kv-24 | 维度：Refactor | 严重程度：中 | 建议优先级：P2
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

AscendStore 自有配置缺少覆盖全部 connector-owned 字段、并由各角色共享的统一解析入口。当前 `extra_config.get` 与 `get_from_extra_config` 分布在 connector、scheduler、worker、metadata、layerwise layout 和 lookup 路径；上游 `KVTransferConfig` 只提供 `dict[str, Any]`，没有为 AscendStore 字段提供统一的类型转换、范围校验和错误信息。

仓库并非完全没有 typed parsing：`layerwise_cache_layout.py` 已有 `_parse_int_config`、范围检查和 `get_gva_layerwise_config`，后者已处理 direct connector 与 `MultiConnector.connectors` 两种来源。部分历史配置允许字符串整数，部分默认值依赖 layerwise/non-layerwise 上下文。因此本任务应复用或迁移现有局部逻辑，而不是从零重复实现；也不要求消灭全部字典读取。

当前 `discard_partial_chunks` 的两个读取分支默认值都为 `True`。它需要纳入回归矩阵，但不能被描述成目前已存在模式间默认值不一致。

## 实现前基线清单

1. 列出当前 main 的 connector-owned 字段、读取角色、默认值、允许类型、范围、兼容别名和是否仍需把原始值传给 backend。
2. 用参数化测试固定 direct connector、MultiConnector、layerwise/non-layerwise 和缺省配置的当前行为；发现文档与代码不一致时先在 PR 中说明选择，不静默改变兼容语义。
3. 明确哪些字段属于 AscendStore，哪些属于 backend/未来扩展。无法确认所有权的字段保留在原始 extra dict，不为了 schema 完整度擅自拒绝。

## 任务

1. 建立 AscendStore connector-owned extra config 的统一 typed parser，集中字段名、兼容默认值、类型 coercion、范围约束和错误信息。
2. 复用或收敛现有 `get_gva_layerwise_config`，使 direct connector 与 MultiConnector 共用同一提取入口；scheduler、worker 和 layout 初始化读取同一 immutable parsed config。
3. 保留原始 extra dict 或明确的 unknown-field 策略，避免截断 backend/未来版本仍需消费的扩展键。
4. 保持上下文相关默认值的现有行为，如 layerwise reuse layout 和 prefetch；`discard_partial_chunks` 当前两分支均默认 `True`，重构后必须保持，除非另有独立行为变更依据。
5. 合并 kv-21 中可独立验证的部分：可在第二阶段提取 rank-invariant key layout 派生值；runtime rank、backend、thread/event 和 request 状态仍由各角色初始化。

## 验收标准

### 1. 功能正确性
- 缺省配置和现有文档示例的行为与改动前一致
- 兼容字符串整数，例如 `lookup_rpc_port: "0"`，并对非法类型/范围给出字段级错误
- direct connector 与 MultiConnector 解析结果一致
- layerwise/non-layerwise 的上下文默认值保持原语义
- deprecated `mooncake_rpc_port` 等兼容项有明确迁移与告警策略
- 现有单测全绿

### 2. 代码质量
- connector-owned 字段具有单一 schema 和类型定义
- scheduler/worker 不再各自重复做同一字段的 coercion 和校验
- 未知字段不会被无意丢弃
- 若实现 rank-invariant 派生 helper，需用同一 VllmConfig 参数化验证 scheduler/worker 结果一致

### 3. 交付件
- PR + 配置字段清单 + 兼容策略 + 参数化单测

## 证据

- worker：`pool_worker.py:145-180, 263-281, 428`
- scheduler：`pool_scheduler.py:93-140, 169, 1194-1209`
- connector：`ascend_store_connector.py:79-95`
- layerwise layout：`layerwise_cache_layout.py:70-154`
- 上游：vLLM `config/kv_transfer.py:23-72, 120-121`

## 重点关注

- schema 解决输入解析，不自动等价于 scheduler/worker 全部初始化去重
- 不删除或平行重写现有 layerwise typed parser；先决定其迁移/复用边界
- 不以“所有 `.get` 消失”作为验收指标
- kv-27 的 timeout 可接入本 parser，但 kv-27 的可靠性修复不应被本任务阻塞

## 环境约定
- vllm-ascend：审核基线 `d5e9816065ede613327d93908f87fee9f5c47128` 或提交时最新 main
- 硬件：Ascend NPU（注明型号和卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
