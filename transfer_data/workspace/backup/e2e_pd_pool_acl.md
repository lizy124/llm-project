# vllm-ascend e2e：PD 分离与池化

## 一、代码层面的区分

`vllm_ascend/distributed/kv_transfer/` 下两个独立子目录，对应不同职责：

| 目录 | 职责 | 源文件 | 注册名 |
|---|---|---|---|
| `kv_p2p/` | PD 分离：prefill → decode 跨节点 KV 直传 | `mooncake_connector.py`、`mooncake_layerwise_connector.py`、`mooncake_hybrid_connector.py` | `MooncakeConnectorV1`、`MooncakeLayerwiseConnector`、`MooncakeHybridConnector` |
| `kv_pool/` | 池化：KV cache 单节点存储/复用 | `ascend_store/`、`lmcache_ascend_connector.py`、`ucm_connector.py`、`simple_cpu_offload/`、`recompute_cpu_offload/` | `AscendStoreConnector`、`LMCacheAscendConnector`、`UCMConnector`、`SimpleCPUOffloadConnector`、`RecomputeCPUOffloadConnector` |

两者是独立模块，没有"PD 分离 vs 池化"的对比关系。

## 二、PD 分离（kv_p2p）e2e 覆盖

### 2.1 3 个 connector

| Connector | 实现 | 特点 |
|---|---|---|
| `MooncakeConnectorV1` | `mooncake_connector.py` | 基准：按 request 粒度传输 |
| `MooncakeLayerwiseConnector` | `mooncake_layerwise_connector.py` | 按 layer 粒度逐层传输 |
| `MooncakeHybridConnector` | `mooncake_hybrid_connector.py` | `use_hybrid=True`，处理 MLA/Full Attention 混合 block size |

### 2.2 Nightly 覆盖

| Connector | Nightly YAML 数 | 模型示例 |
|---|---|---|
| `MooncakeConnectorV1` | 10 | Qwen3-235B 系列、DeepSeek-V3.2/R1、GLM-5.1 |
| `MooncakeLayerwiseConnector` | 0 | — |
| `MooncakeHybridConnector` | 0 | — |

Nightly 的 PD 分离覆盖集中在 V1，Layerwise 和 Hybrid 缺覆盖。

### 2.3 判定方式

YAML 中同时出现以下字段即为 PD 分离测试：

- `disaggregated_prefill: enabled: true`
- `--kv-transfer-config` 中指定 `kv_connector` 为上述 3 种之一

## 三、池化（kv_pool）e2e 覆盖

### 3.1 5 个 connector

| Connector | 用途 | 外部依赖 |
|---|---|---|
| `AscendStoreConnector` | 基于 Mooncake 后端的 KV pool 存储/复用 | Mooncake 服务 |
| `SimpleCPUOffloadConnector` | NPU 适配的 CPU KV offload | 无 |
| `RecomputeCPUOffloadConnector` | 重计算 + CPU offload | 无 |
| `LMCacheAscendConnector` | LM Cache 集成 | `lmcache_ascend` 库 |
| `UCMConnector` | 统一缓存管理 | `ucm` 库 + UCM 服务 |

### 3.2 Nightly 覆盖

| Connector | 测试 | 状态 |
|---|---|---|
| `AscendStoreConnector` | `test_qwen3_30b_acc.py`（`a3.multi_card`） | 已覆盖 |
| `SimpleCPUOffloadConnector` | `test_simple_cpu_offload.py`（仅 PR 级别） | 缺 nightly |
| `RecomputeCPUOffloadConnector` | `test_recompute_cpu_offload.py`（仅 UT） | 缺 e2e |
| `LMCacheAscendConnector` | 无 | 缺（依赖外部库） |
| `UCMConnector` | 无 | 缺（依赖外部服务） |

### 3.3 判定方式

池化 connector 通过 `--kv-transfer-config` 的 `kv_connector` 字段指定，与是否 PD 分离无关：

- `AscendStoreConnector`：单节点 `kv_role: kv_both`，需 MooncakeLauncher
- `SimpleCPUOffloadConnector`：单节点 `kv_role: kv_both`，无外部依赖
- `RecomputeCPUOffloadConnector`：单节点，无外部依赖

## 四、常见误解

1. **"多节点 = PD 分离"**：错。多节点 YAML 可能只是纯分布式部署（如 `DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml`），不含 `disaggregated_prefill`。
2. **"PD 分离 = 池化"**：错。PD 分离是 kv_p2p 直传，池化是 kv_pool 存储/复用。
3. **"`pull_request/one_card/pooling/*` 是池化"**：错。那是 embedding/classification 的 pooling runner。
4. **"模型名里的 A22B/A3B 是平台"**：错。那是模型参数量标记。