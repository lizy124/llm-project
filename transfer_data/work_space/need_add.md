# 待补充用例

> 详细覆盖率分析见 [current_issues.md](current_issues.md)，概念区分见 [pd_vs_pool_concept.md](back_up/pd_vs_pool_concept.md)。

---

## 一、池化（kv_pool）待补充用例

### 1.1 P0：`SimpleCPUOffloadConnector` 纳入 nightly

**当前状态：** PR 已有完整 e2e 测试（`tests/e2e/pull_request/one_card/test_simple_cpu_offload.py`），但未纳入 nightly 看护。

**资源需求：**

| 资源 | 需求 |
|---|---|
| 平台 | A2（单卡） |
| 模型 | `Qwen/Qwen3-0.6B`（极小模型，约 1.2GB） |
| 额外依赖 | 无 |

**加入方式：** 在 `.github/workflows/configs/nightly_config.yaml` 的 `a2.single_node.test_config` 新增一条 pytest 条目：

```yaml
- name: test_simple_cpu_offload
  os: linux-aarch64-a2b3-1
  tests: tests/e2e/pull_request/one_card/test_simple_cpu_offload.py
```

**测试内容：**
- `test_simple_cpu_offload_accuracy`：冷跑填充 GPU cache → CPU offload → 重置 GPU prefix cache → 从 CPU 重新加载 KV → 验证输出一致
- `test_simple_cpu_offload_no_crash_on_repeat`：多次短请求，覆盖 eager/lazy offload 两种路径，验证无崩溃

**为什么简单：** 测试已写好，不需要新模型，不需要新代码，只需一条配置。

---

### 1.2 P1：`RecomputeCPUOffloadConnector` 新增 e2e 测试

**当前状态：** 仅有 UT（`tests/ut/kv_offload/test_recompute_cpu_offload.py`），无 e2e 测试。

**为什么不能直接加：** 根据官方文档，该 connector 有硬性约束：

1. **只能在 PD 分离的 Decode 节点使用**（`kv_role="kv_consumer"`），不支持 PD-Mixed
2. **必须启用 `RecomputeScheduler`**（`--additional-config '{"scheduler_config":{"recompute_scheduler_enable":true}}'`）
3. **必须通过 `MultiConnector` 组合**（`MooncakeConnectorV1` + `RecomputeCPUOffloadConnector`）

**测试方案：**

```
2 节点 PD 分离环境
├── P 节点（kv_producer）：MooncakeConnectorV1
└── D 节点（kv_consumer）：MultiConnector
    ├── MooncakeConnectorV1（收 P 的 KV cache）
    └── RecomputeCPUOffloadConnector（D 节点 HBM 不够时卸到 CPU，恢复时加载回来）
```

**测试逻辑：**
1. 构造大量并发请求，使 D 节点 HBM 不足，触发 `RecomputeScheduler` 抢占
2. 被抢占请求的 KV cache 通过 `RecomputeCPUOffloadConnector` 保存到 CPU DRAM
3. 请求恢复调度时，从 CPU 加载回 HBM
4. 验证恢复后的输出正确（与不抢占的基线对比）

**需要做的：**
1. 新增 e2e 测试文件（参考 `test_deepseek_v3_2_w8a8_pruning.py` 的 PD 分离测试框架）
2. 新增 PD 分离 YAML 配置文件（P 节点用 `MooncakeConnectorV1`，D 节点用 `MultiConnector`）
3. 纳入 `nightly_config.yaml`（A3 平台，2 节点）

---

### 1.3 P2：`LMCacheAscendConnector` / `UCMConnector`

| Connector | 阻塞原因 | 预计动作 |
|---|---|---|
| `LMCacheAscendConnector` | 依赖 `lmcache_ascend` 外部库，环境未就绪 | 环境就绪后新增 e2e 测试 + 纳入 nightly |
| `UCMConnector` | 依赖 `ucm` 库 + UCM 服务端，环境未就绪 | UCM 服务端就绪后新增 e2e 测试 + 纳入 nightly |

---

## 二、PD 分离（disaggregated prefill）待补充用例

### 2.1 P0：`MooncakeLayerwiseConnector` 纳入 nightly

**当前状态：**
- weekly 已有 2 个用例：`GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml`（2 节点）、`DeepSeek-V3.1T-MTP1-Mooncake-Layerwise.yaml`（4 节点）
- nightly **无** layerwise 用例

**建议：** 将 GLM-4.7 的 layerwise 用例纳入 nightly。理由：
- 模型较轻（GLM-4.7），2 节点即可
- 和 nightly 已有的 `multi-node-GLM-5.1-w8a8-A3` 属于同一系列，模型已在 A3 环境就绪
- 适合每日看护 layerwise（Push 模式）路径

**加入方式：** 在 `nightly_config.yaml` 的 `a3.double_node.test_config` 新增一条：

```yaml
- name: multi-node-glm-4.7-mooncake-layerwise
  config_file_path: GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml
  size: 2
```

> **注意：** 对应的 YAML 文件需确认已存在于 `tests/e2e/nightly/multi_node/internal_dp/config/` 目录下。如果 weekly 的 YAML 在 weekly 目录，需要拷贝或建立软链接。

---

### 2.2 P1：`MooncakeLayerwiseConnector` 纳入 PR 级别测试

**当前状态：** PR 级别无任何 layerwise 测试。

**测试方案：** 参考现有 `test_deepseek_v3_2_w8a8_pruning.py`（1P1D V1 测试），新增 1P1D 的 layerwise 测试：

- 使用较小模型（如 GLM-4.7），1P1D 部署
- 功能验证：发请求 → 验证正常返回
- 不走精度基准（PR 级别不需要）

---

### 2.3 P1：`MooncakeHybridConnector` 纳入 nightly

**当前状态：** weekly 已有 1 个用例（`DeepSeek-V4-Flash-W8A8-Mooncake-Hybrid.yaml`，2 节点），nightly 无。

**阻塞点：** `MooncakeHybridConnector` 目前仅用于 DeepSeek-V4 系列。V4 模型较大，2 节点是否稳定、是否适合 nightly 需确认。

**建议：** 评估 nightly 已有的 `DeepSeek-V4-Flash-W8A8-A3`（单节点）在多节点 hybrid PD 分离场景下的稳定性，再决定是否纳入。

---

## 三、汇总

| 优先级 | 模块 | Connector | 动作 | 资源 | 阻塞 |
|---|---|---|---|---|---|
| **P0** | 池化 | `SimpleCPUOffloadConnector` | 加一条 nightly YAML 配置 | A2 单卡 | 无 |
| **P0** | PD 分离 | `MooncakeLayerwiseConnector` | 加一条 nightly YAML 配置（GLM-4.7） | A3 2 节点 | 确认 YAML 文件位置 |
| **P1** | 池化 | `RecomputeCPUOffloadConnector` | 新增 e2e 测试 + 纳入 nightly | A3 2 节点 PD 分离 | 需先写 e2e 测试 |
| **P1** | PD 分离 | `MooncakeLayerwiseConnector` | 新增 PR 级别 1P1D 测试 | A3 4 卡 | 需先写 e2e 测试 |
| **P1** | PD 分离 | `MooncakeHybridConnector` | 评估后纳入 nightly | A3 2 节点 | 需确认 V4 多节点稳定性 |
| **P2** | 池化 | `LMCacheAscendConnector` | 等环境就绪 | — | `lmcache_ascend` 库 |
| **P2** | 池化 | `UCMConnector` | 等环境就绪 | — | UCM 服务端 |

---

## 四、遗留问题：A5 平台用例缺失

当前 PD 分离、池化、图模式的所有用例均在 **A2 / A3** 平台，**无 A5 平台的用例**。

| 模块 | A2 | A3 | A5 |
|---|---|---|---|
| PD 分离 | 有 | 有 | **无** |
| 池化 | 有 | 有 | **无** |
| 图模式 | 有 | 有 | **无** |

**阻塞点：无任何 A5 参考用例。** 整个 `nightly_config.yaml` 和 `weekly_config.yaml` 中不存在任何 A5 条目，无法参考以下信息：

- A5 的 runner 标签（A2 为 `linux-aarch64-a2b3-*`，A3 为 `linux-aarch64-nightly-a3-*`，A5 未知）
- A5 的镜像配置
- A5 的调度方式（单节点/多节点、节点规格）
- A5 的 YAML 配置样例

> **待 A5 平台环境就绪且至少有一个特性的 A5 用例落地后，再补充 PD 分离、池化、图模式的 A5 用例。**