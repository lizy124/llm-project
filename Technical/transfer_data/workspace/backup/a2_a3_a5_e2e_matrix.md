# tests/e2e A2 / A3 / A5 平台覆盖对照表

## 一、术语说明

代码中 `kv_transfer` 下有两个独立子目录，对应不同职责：

| 目录 | 职责 | 涉及 connector |
|---|---|---|
| `kv_p2p/` | PD 分离：prefill → decode 跨节点 KV 直传 | MooncakeConnectorV1、MooncakeLayerwiseConnector、MooncakeHybridConnector |
| `kv_pool/` | 池化：KV cache 单节点存储/复用 | AscendStoreConnector、SimpleCPUOffloadConnector、RecomputeCPUOffloadConnector、LMCacheAscendConnector、UCMConnector |

这是两个独立维度，不应混在一起讨论。

## 二、覆盖总表

| 维度 | A2 | A3 | A5 |
|---|---|---|---|
| 图模式（PR 级 pytest） | 有 | 有 | 无 |
| PD 分离（kv_p2p，nightly YAML） | 无 | 有 | 无 |
| 池化（kv_pool，nightly YAML/pytest） | 无 | 有（仅 AscendStoreConnector） | 无 |

## 三、A2 平台

### 3.1 图模式 — 有

PR 级 pytest 中明确标 `hardware="A2"` 的用例：

- `tests/e2e/pull_request/one_card/test_qwen3_0_6b.py`
- `tests/e2e/pull_request/one_card/test_batch_invariant.py`（4 处）
- `tests/e2e/pull_request/one_card/lora/test_qwen3_multi_loras.py`

### 3.2 PD 分离 — 无

nightly 中 A2 命名的多节点 YAML 均为纯分布式部署，不含 `disaggregated_prefill` 字段：

- `GLM5_1-W8A8-A2-dual-nodes.yaml`
- `Qwen3-235B-A22B-A2.yaml`
- `Kimi-K2_5-W4A8-A2-dual-nodes.yaml`

### 3.3 池化 — 无

A2 平台没有任何池化（kv_pool）connector 的 e2e 测试。

## 四、A3 平台

### 4.1 图模式 — 有（部分 skip）

PR 级 pytest 中明确标 `hardware="A3"` 的用例：

- `tests/e2e/pull_request/four_card/test_deepseek_v4.py`（2 处，有效）
- `tests/e2e/pull_request/two_card/test_qwen3_30b_a3b.py`（整体 skip）

### 4.2 PD 分离 — 有

nightly 中带 `disaggregated_prefill` + `ASCEND_A3_ENABLE` 的 YAML：

- `DeepSeek-V3_2-W8A8-EP.yaml`（4 节点）
- `GLM5_1-W8A8-EP.yaml`（internal_dp，2 节点）
- `GLM5_1-W8A8-EP-external.yaml`（external_dp，2 节点）

全部使用 `MooncakeConnectorV1`（kv_p2p）。

### 4.3 池化 — 有（仅 AscendStoreConnector）

nightly 中 `a3.multi_card` 调度的池化测试：

- `test_qwen3_30b_acc.py`（`AscendStoreConnector`，单节点 KV pool 复用）

其余 4 个池化 connector（SimpleCPUOffload、RecomputeCPUOffload、LMCache、UCM）在 nightly 无覆盖。

## 五、A5 平台

### 5.1 图模式 — 无

无 `hardware="A5"` 的 pytest 用例。

### 5.2 PD 分离 — 无

无 A5 平台的 PD 分离 YAML。

### 5.3 池化 — 无

无 A5 平台的池化测试。

## 六、补充说明

- 文件名中的 `A22B`、`A3B` 是模型名片段，不代表平台归属。
- `pull_request/one_card/pooling/*` 是 embedding/classification 的 pooling runner，不是 kv_pool 池化。
- 多节点 YAML 不等于 PD 分离，必须看 `disaggregated_prefill` 字段。