# 池化与 PD 分离：用例覆盖分析与缺口

> **概念区分参考：** [pd_vs_pool_concept.md](backup/pd_vs_pool_concept.md) / 官方设计文档 [KV_Cache_Pool_Guide.md](../../../code/vllm-ascend/docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md) / [disaggregated_prefill.md](../../../code/vllm-ascend/docs/source/developer_guide/Design_Documents/disaggregated_prefill.md)

## 核心区别

| 维度 | 池化（kv_pool） | PD 分离（disaggregated prefill） |
|---|---|---|
| **本质** | KV cache **存储**：存到池子 → 按 key 查找 → 加载复用 | KV cache **传输**：prefill 算完直接 P2P 传 decode，用完即丢 |
| **数据生命周期** | 持久化在池中（DRAM/SSD），跨请求复用 | 一次性传输，不持久化 |
| **目录** | `kv_transfer/kv_pool/` | `kv_transfer/kv_p2p/` |
| **核心 Connector** | `AscendStoreConnector`（后端起 Mooncake/Memcache/Yuanrong） | `MooncakeConnectorV1`（pull）/ `MooncakeLayerwiseConnector`（push） |
| **是否可共存** | 可以！通过 `MultiConnector` 同时启用池化+PD 分离：池化负责 prefix cache 复用，PD 分离负责 P2P 传输 |

> **池化有两种部署模式：**
> 1. **PD-Mixed**（`kv_both`）：单实例自己存自己取，池子作为共享 prefix cache
> 2. **PD 分离 + 池化**（`kv_producer`/`kv_consumer`）：prefill 存入池子，decode 从池子加载，通过 `MultiConnector` 组合 `MooncakeConnectorV1` + `AscendStoreConnector`

---

## 一、池化（kv_pool）用例覆盖分析

`vllm_ascend/distributed/kv_transfer/__init__.py` 注册了 5 个池化 connector：

| Connector | 注册名 | 后端 | 部署模式 | 外部依赖 |
|---|---|---|---|---|
| `AscendStoreConnector` | `MooncakeConnectorStoreV1` / `AscendStoreConnector` | Mooncake、Memcache、Yuanrong | `kv_both`、`kv_producer`/`kv_consumer` | Mooncake 服务 / Memcache 服务 |
| `SimpleCPUOffloadConnector` | `SimpleCPUOffloadConnector` | CPU DRAM | `kv_both` | 无 |
| `RecomputeCPUOffloadConnector` | `RecomputeCPUOffloadConnector` | CPU DRAM | `kv_both` | 无 |
| `LMCacheAscendConnector` | `LMCacheAscendConnector` | LMCache 后端 | `kv_both` | `lmcache_ascend` 库 |
| `UCMConnector` | `UCMConnector` | UCM | `kv_both` | `ucm` 库 + UCM 服务 |

> **`AscendStoreConnector` 还支持 Layerwise 模式**（`use_layerwise: true`），以逐层方式 save/load KV cache，减少首 token 延迟。当前仅支持 Memcache 后端。详见 [layerwise_kv_pool.md](../../../code/vllm-ascend/docs/source/user_guide/feature_guide/layerwise_kv_pool.md)。

### 1.1 pull_request 测试

| Connector | 测试文件 | 平台 | 状态 |
|---|---|---|---|
| `SimpleCPUOffloadConnector` | `tests/e2e/pull_request/one_card/test_simple_cpu_offload.py` | A2 | **已覆盖** |
| `AscendStoreConnector` | — | — | **无** |
| `RecomputeCPUOffloadConnector` | `tests/ut/kv_offload/test_recompute_cpu_offload.py` | — | **仅 UT，无 e2e** |
| `LMCacheAscendConnector` | — | — | **无** |
| `UCMConnector` | — | — | **无** |

**缺失：**
- `AscendStoreConnector` 在 PR 级别无任何 e2e 测试
- `RecomputeCPUOffloadConnector` 仅有 UT，缺少 e2e 测试
- `LMCacheAscendConnector` 和 `UCMConnector` 完全无测试

### 1.2 nightly 测试

| Connector | 测试文件 | 调度方式 | 平台 | 状态 |
|---|---|---|---|---|
| `AscendStoreConnector` | `tests/e2e/weekly/single_node/models/test_qwen3_30b_acc.py` | `nightly_config.yaml` → `a3.multi_card` | A3 | **已覆盖**（Qwen3-30B 精度测试，非专项池化测试） |
| `SimpleCPUOffloadConnector` | — | — | — | **无** |
| `RecomputeCPUOffloadConnector` | — | — | — | **无** |
| `LMCacheAscendConnector` | — | — | — | **无** |
| `UCMConnector` | — | — | — | **无** |

**缺失：**
- `SimpleCPUOffloadConnector`：PR 已有 e2e 测试，但未纳入 nightly 看护，建议直接复用 `test_simple_cpu_offload.py`
- `RecomputeCPUOffloadConnector`：需先补齐 e2e 测试，再纳入 nightly
- `LMCacheAscendConnector` 和 `UCMConnector`：依赖外部库/服务，暂不补充

### 1.3 weekly 测试

| Connector | 测试文件 | 调度方式 | 平台 | 状态 |
|---|---|---|---|---|
| `AscendStoreConnector` | — | — | — | **无** |
| `SimpleCPUOffloadConnector` | — | — | — | **无** |
| `RecomputeCPUOffloadConnector` | — | — | — | **无** |
| `LMCacheAscendConnector` | — | — | — | **无** |
| `UCMConnector` | — | — | — | **无** |

**缺失：** 所有池化 connector 在 weekly 中均无覆盖。`AscendStoreConnector` 仅在 nightly 中有非专项覆盖（Qwen3-30B 精度测试）。

### 1.4 池化缺口汇总

| 优先级 | Connector | 当前状态 | 建议动作 |
|---|---|---|---|
| **P0** | `SimpleCPUOffloadConnector` | PR 有 e2e，nightly 无 | 在 `nightly_config.yaml` 新增条目，复用已有测试 |
| **P1** | `RecomputeCPUOffloadConnector` | 仅 UT | 参考 `test_simple_cpu_offload.py` 新增 e2e 测试，再纳入 nightly |
| **P2** | `LMCacheAscendConnector` | 无任何测试 | 需确认 `lmcache_ascend` 环境就绪后再评估 |
| **P2** | `UCMConnector` | 无任何测试 | 需 UCM 服务端环境就绪后再评估 |
| — | `AscendStoreConnector` | nightly 有非专项覆盖 | 暂无独立池化测试，当前随 Qwen3-30B 精度测试附带覆盖 |

---

## 二、PD 分离（disaggregated prefill）用例覆盖分析

`vllm_ascend/distributed/kv_transfer/kv_p2p/` 下共 3 个 connector：

| Connector | 传输模式 | 工作原理 |
|---|---|---|
| `MooncakeConnectorV1` | **Pull**（D 节点拉取） | Proxy 路由请求到 P 节点做完 prefill → D 节点主动从 P 节点拉取 KV cache |
| `MooncakeLayerwiseConnector` | **Push**（P 节点推送） | P 节点逐层算完 KV 后立即推送给 D 节点，D 节点逐层接收后开始 decode |
| `MooncakeHybridConnector` | Pull + 混合 | 继承 V1，处理 MLA/Full Attention 混合模型（如 DeepSeek-V4） |

> **PD 分离架构：** 全局 Proxy 接收外部请求，将 prefill 转发到 P 节点，decode 转发到 D 节点，KV cache 通过 P2P 在 P/D 节点间交换。详见 [disaggregated_prefill.md](../../../code/vllm-ascend/docs/source/developer_guide/Design_Documents/disaggregated_prefill.md)。

### 2.1 pull_request 测试

| Connector | 测试文件 | 测试场景 | 平台 | 状态 |
|---|---|---|---|---|
| `MooncakeConnectorV1` | `tests/e2e/pull_request/four_card/test_deepseek_v3_2_w8a8_pruning.py` | 1P1D PD 分离，DeepSeek-V3.2-W8A8-Pruning，TP=2 | A3 | **已覆盖** |
| `MooncakeLayerwiseConnector` | — | — | — | **无** |
| `MooncakeHybridConnector` | — | — | — | **无** |

**缺失：**
- `MooncakeLayerwiseConnector`：PR 级别无测试，可以考虑用更小的模型（如 GLM-4.7）做 1P1D 的 layerwise 测试
- `MooncakeHybridConnector`：PR 级别无测试，但该 connector 仅用于 DeepSeek-V4，模型过大不适合 PR 级别

### 2.2 nightly 测试

nightly 中 PD 分离的用例全部使用 `MooncakeConnectorV1`，共 **6 个**（均在 A3 平台）：

| 调度 YAML | 调度分组 | 节点数 | 模型 | 平台 |
|---|---|---|---|---|
| `DeepSeek-V3_2-W8A8-EP.yaml` | `multi_node` | 4 | DeepSeek-V3.2-W8A8 | A3 |
| `DeepSeek-R1-W8A8-longseq.yaml` | `double_node` | 2 | DeepSeek-R1-W8A8 | A3 |
| `Qwen3-235B-disagg-pd.yaml` | `double_node` | 2 | Qwen3-235B-A22B | A3 |
| `Qwen3-VL-235B-disagg-pd.yaml` | `double_node` | 2 | Qwen3-VL-235B-A22B | A3 |
| `Qwen3-235B-W8A8-EPLB.yaml` | `double_node` | 2 | Qwen3-235B-A22B-W8A8 | A3 |
| `Qwen3-235B-W8A8-longseq.yaml` | `double_node` | 2 | Qwen3-235B-A22B-W8A8 | A3 |

| Connector | 用例数 | 状态 |
|---|---|---|
| `MooncakeConnectorV1` | 6 | **已覆盖** |
| `MooncakeLayerwiseConnector` | 0 | **无** |
| `MooncakeHybridConnector` | 0 | **无** |

**缺失：**
- `MooncakeLayerwiseConnector`：nightly 无覆盖，建议用 GLM-4.7 等较轻模型做 2 节点 layerwise 测试（参考 weekly 的 `GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml`）
- `MooncakeHybridConnector`：nightly 无覆盖，但 DeepSeek-V4 模型过大，不适合 nightly 频率

### 2.3 weekly 测试

weekly 中 PD 分离用例覆盖最全面，共 **20 个**（均在 A3 平台）：

#### 2.3.1 external_dp（17 个）

| 调度 YAML | Connector | 节点数 | 模型 |
|---|---|---|---|
| `DeepSeek_V3.1T_MTP1_PD.yaml` | `MooncakeConnectorV1` | 4 | DeepSeek-V3.1T |
| `DeepSeek_V3.1T_MTP1_128K_1K_PD.yaml` | `MooncakeConnectorV1` | 4 | DeepSeek-V3.1T (128K) |
| `DeepSeek_V3.1T_MTP1_3_5K_1_5K_PD.yaml` | `MooncakeConnectorV1` | 4 | DeepSeek-V3.1T (3.5K) |
| `DeepSeek_V3.1T_MTP3_PD.yaml` | `MooncakeConnectorV1` | 4 | DeepSeek-V3.1T (MTP3) |
| `DeepSeek_V3.2T_MTP2_PD.yaml` | `MooncakeConnectorV1` | 4 | DeepSeek-V3.2T (MTP2) |
| `DeepSeek_V3.2T_MTP3_PD.yaml` | `MooncakeConnectorV1` | 4 | DeepSeek-V3.2T (MTP3) |
| `GLM_5_1_PD_in32k_bs16-0.yaml` | `MooncakeConnectorV1` | 4 | GLM-5.1 (32K) |
| `GLM_5_1_PD_in32k_bs20-90.yaml` | `MooncakeConnectorV1` | 4 | GLM-5.1 (32K) |
| `GLM_5_1_PD_in64k_bs16-90.yaml` | `MooncakeConnectorV1` | 4 | GLM-5.1 (64K) |
| `Kimi-K2.5-W4A8-16k-1k-TPOT50.yaml` | `MooncakeConnectorV1` | 4 | Kimi-K2.5 (16K) |
| `Kimi-K2.5-W4A8-128k-1k-TPOT50.yaml` | `MooncakeConnectorV1` | 4 | Kimi-K2.5 (128K) |
| `MiniMax-PD-in32k-bs4-1.yaml` | `MooncakeConnectorV1` | 3 | MiniMax (32K) |
| `QWEN3_235B_PD.yaml` | `MooncakeConnectorV1` | 3 | Qwen3-235B |
| `QWEN3_235B_PD_3_5K_1_5k.yaml` | `MooncakeConnectorV1` | 3 | Qwen3-235B (3.5K) |
| `Qwen-3.5-397B-A17B-W8A8-PD.yaml` | `MooncakeConnectorV1` | 3 | Qwen3.5-397B |
| `DeepSeek_V3.1T_layerwise_PD.yaml` | **`MooncakeLayerwiseConnector`** | 4 | DeepSeek-V3.1T |
| `DeepSeek-V4-flash-w8a8-PD.yaml` | **`MooncakeHybridConnector`** | 2 | DeepSeek-V4-Flash |

#### 2.3.2 internal_dp（3 个）

| 调度 YAML | Connector | 节点数 | 模型 |
|---|---|---|---|
| `DeepSeek-V3_2-W8A8-EP_weekly.yaml` | `MooncakeConnectorV1` | 2 | DeepSeek-V3.2-W8A8 |
| `DeepSeek-V3.yaml` | `MooncakeConnectorV1` | 2 | DeepSeek-V3 |
| `GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml` | **`MooncakeLayerwiseConnector`** | 2 | GLM-4.7-W8A8C8 |

#### 2.3.3 按 Connector 汇总

| Connector | external_dp | internal_dp | 合计 | 状态 |
|---|---|---|---|---|
| `MooncakeConnectorV1` | 15 | 2 | **17** | **已覆盖** |
| `MooncakeLayerwiseConnector` | 1 | 1 | **2** | **已覆盖** |
| `MooncakeHybridConnector` | 1 | 0 | **1** | **已覆盖** |

weekly 的 PD 分离覆盖已非常全面，三个 connector 均有覆盖。

### 2.4 PD 分离缺口汇总

| 优先级 | Connector | pull_request | nightly | weekly | 建议动作 |
|---|---|---|---|---|---|
| **P0** | `MooncakeLayerwiseConnector` | 无 | 无 | 有（2 个） | 在 nightly 补充 2 节点 GLM-4.7 layerwise 测试（参考 weekly 已有用例） |
| **P1** | `MooncakeLayerwiseConnector` | 无 | — | — | 评估是否在 PR 级别加入轻量 layerwise 测试 |
| **P2** | `MooncakeHybridConnector` | 无 | 无 | 有（1 个） | DeepSeek-V4 模型过大，nightly 暂不补充；PR 级别也不适合 |
| — | `MooncakeConnectorV1` | 有（1 个） | 有（6 个） | 有（17 个） | 覆盖充分，无需补充 |

---

## 三、总结

### 3.1 池化覆盖总览

| Connector | PR | Nightly | Weekly | 整体评估 |
|---|---|---|---|---|
| `AscendStoreConnector` | — | ✓（非专项） | ✗ | nightly 有覆盖但非专项池化测试 |
| `SimpleCPUOffloadConnector` | ✓ | ✗ | ✗ | **需补充 nightly** |
| `RecomputeCPUOffloadConnector` | 仅 UT | ✗ | ✗ | **需补充 e2e + nightly** |
| `LMCacheAscendConnector` | ✗ | ✗ | ✗ | 待环境就绪 |
| `UCMConnector` | ✗ | ✗ | ✗ | 待环境就绪 |

### 3.2 PD 分离覆盖总览

| Connector | PR | Nightly | Weekly | 整体评估 |
|---|---|---|---|---|
| `MooncakeConnectorV1` | ✓ | ✓ (6) | ✓ (17) | 覆盖充分 |
| `MooncakeLayerwiseConnector` | ✗ | ✗ | ✓ (2) | **需补充 nightly** |
| `MooncakeHybridConnector` | ✗ | ✗ | ✓ (1) | 模型过大，暂不补充 |

### 3.3 优先行动项

| 序号 | 动作 | 优先级 | 涉及 Connector | 改动范围 |
|---|---|---|---|---|
| 1 | nightly 补充 `SimpleCPUOffloadConnector` | **P0** | 池化 | `nightly_config.yaml` 新增 1 条目（复用已有测试） |
| 2 | nightly 补充 `MooncakeLayerwiseConnector` | **P0** | PD 分离 | `nightly_config.yaml` 新增 1 条目（参考 weekly 已有 YAML） |
| 3 | 新增 `RecomputeCPUOffloadConnector` e2e 测试 | **P1** | 池化 | 新增测试文件 + nightly_config 注册 |
| 4 | 评估 `LMCacheAscendConnector` / `UCMConnector` | **P2** | 池化 | 待环境就绪后评估 |
