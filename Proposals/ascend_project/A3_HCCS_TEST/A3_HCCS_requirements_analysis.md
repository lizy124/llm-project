A3_HCCS_requirements_analysis.md
# A3_HCCS 双机池化验证 — 需求分析与承接（原始需求 + 分析）

> **文档定位**：澄清 `A3_HCCS/` 这套验证工作（PD 分离 + KV Cache 池化，DS V4 Flash，200K 上下文）的来龙去脉——**原始需求从哪来、拆成了哪些子项、本目录承接了哪一部分、验证目标如何落到现有验证方案**。
> **对下关系**：本文是 `A3_HCCS/README.md`、`A3_HCCS/128G/verification_plan.md`、`A3_HCCS/64G/verification_plan.md` 的上游输入。
> **归口负责人**：A3 HCCS 场景由 lizy124 承接。

---

## 0. 结论先行（Exec Summary）

1. 需求来源是**一体机规划**：一体机典型机型（FM8 单机 / FM16 双机）须支持 KV Cache 池化，**A3 HCCS 背靠背双机互联是其中一条明确待验证场景**，由 lizy124 承接。
2. 单个需求文档把「机型清单 + 组网形态分工 + 统一验证目标」三块放在一处，但**验证目标（200K / 128K+1K / DSV4 Flash / 90% 重复率 / 80%DDR+20%SSD / 5~8x）对所有机型和所有组网形态共享**，不单属于 A3 HCCS。
3. `A3_HCCS/` 目录把它**承接为两个阶段**：先 128G（优先，已 Phase 1 两轮 PASS），后 64G（等机器）；Phase 2 双机 HCCS 跨机传输是 A3 场景相对单机基线的核心增量。
4. **验证依赖**：SSD 与 layerwise 特性依赖上游特性进度；其中 memcache SSD 已知问题已解决、收益待实测——这是 Phase 3/4 的开放项。

---

## 1. 原始需求记录（原文）

> 粘贴自需求文档（单一需求 SR20260821262965）。正文按原文保留，含「需求描述」与「需求展开（组网分工 + 共享验证目标）」两块，仅做结构化分段。

### 1.1 原始需求（单一需求，编号 SR20260821262965）

> 说明：以下两块同属**一个需求**（SR20260821262965）。第一块为「需求描述」正文（机型清单）；第二块是同一需求下的**展开说明**——组网形态分工 + 共享验证目标，并非独立的第二个需求。

**编号**：SR20260821262965
**创建**：kezhan 00930096（VisionIT），2026-08-21 18:56:42

**【需求来源】** 一体机规划

**【需求背景】**
一体机典型机型陆续上市，包括跨 RoCE 互联、跨 UB 互联（双机背靠背）需要支持池化能力

**【需求描述】**
一体机典型机型需要支持池化，机型包括

- FM8 单机：A3 560T 128G、A3 560T 64G、A5 650（PR）、650E（DT）、150（单机 4 卡，已支持）
- FM16 双机：A3 560T 128G、A3 560T 64G、A5 650（PR）、650E（DT）背靠背 HCCS 互联（A3/A5 各挑一个）

**【需求分类】**
- 需求类别：vLLM--Ascend / RP3 / 池化 / 一体机
- 归属：一体机规划需求

**【需求展开：组网形态分工】**
> 同一需求的展开段，非独立需求；对 A5 各形态组网做拆解与分工，其中 A3 HCCS 归 lizy124。

根据 A5 各形态涉及的组网形态（**2 个后端都需要验证**，需要叠加池化与 pp/layerwise/ssd 特性），总结场景如下：

| # | 组网形态 | 状态 | 传输方案 | 责任 |
|---|---------|------|----------|------|
| 1 | UB | 已支持 | HIXL / Memfabric | wanglei |
| 2 | UBOE | 已支持 | HIXL / Memfabric | tanyueyue |
| 3 | UBG | 确认中 | — | liushiyu |
| 4 | DEVICE ROCE | 理论支持，待验证，等物料 | — | A5 Roce baoguoqiang |
| 5 | **A3 HCCS** | 待验证 | — | **lizy124** |

**共享验证目标（对本任务，A3 HCCS）**：
- max-model-len = **200K** 上下文长度
- 序列长度：**128K 输入 / 1K 输出**
- 模型：**DS V4 Flash**
- 多前缀，前缀重复率 **90%**
- 池化基本能力正常，关闭 HBM PrefixCache 命中，启动池化 DDR/SSD 可正常加载卸载 KVCache
- PrefixCache 命中率达成 **80% DRAM / 20% SSD**（chenbo 那边的数据集构造工具）
- Prefill 吞吐提升 **5~8 倍**（memcache 的 SSD 已知问题已经解决，具体收益暂不确定？）
- 池化 + ssd 开启，命中率占比要满足要求，吞吐收益要获得（**对照组：不开池化+ssd、也不开 hbm、完全重算**）
- 验证依赖上面 **SSD 和 layerwise 特性**

---

## 2. 需求分析

### 2.1 需求解读与拆解

| 维度 | 内容 |
|------|------|
| 一句话 | 一体机典型机型要能靠 KV 外部池（DDR/SSD）在长上下文（200K）下提升吞吐，A3 HCCS 双机是其中一条待验证的背靠背组网形态 |
| 本质诉求 | 超额上下文 + 高命中复用：关闭 HBM 前缀缓存，把 KV 溢出到 DDR/SSD 并高效命中复用，换取端到端 Prefill 吞吐提升 |
| 必须满足 | 池化基本能力正常（存/取/去重）、命中率 80D/20S、吞吐 5~8x（vs 完全重算） |
| 隐含约束 | 模型固定 DSV4 Flash、上下文 200K、90% 前缀重复率、层叠 pp/layerwise/SSD 需与池化共存 |
| 依赖项 | SSD tiering 特性（memcache 已知问题已解决，**收益待实测**）、layerwise 特性（DSV4 为已知硬限，见下） |

### 2.2 组网形态分工与本目录承接

- 该需求对五种组网形态做了明确责任分工，**A3 HCCS 归 lizy124**——即本目录落地范围。
- 五种形态共享同一套验证目标，可复用同一套验证方法与口径（Phase 1 单机 → Phase 2 双机 → Phase 3 特性叠加 → Phase 4 性能验收）。

### 2.3 验证目标的落点映射（→ verification_plan）

| 原始需求目标 | 承接落点 | 现状 |
|---|---|---|
| 200K max-model-len | 128G plan §5.4 / §1.4（内存预算） | Phase 1 用 131072 口径；200K 属 Phase 4 待采 |
| 128K 输入 / 1K 输出 | 全篇测试口径 | Phase 1 用 131070 前缀 + 1 token 输出 |
| DS V4 Flash | 环境（135 权重已就位） | ✅ |
| 多前缀 / 90% 重复率 | 测试方法 §6 + Phase 4 数据集构造 | Phase 1 4×131070 + 32 并发 |
| 关闭 HBM PrefixCache | P0：`prefix_cache_*` 恒 0 | ✅ Phase 1 已证 |
| DDR/SSD 加载卸载正常 | Phase 3 SSD 分级 + Phase 4 | 未开始 |
| 80% DRAM / 20% SSD | Phase 4 命中率验收 | 未开始 |
| Prefill 5~8x（vs 完全重算） | §7（冷灌即完全重算基线） | ✅ ~11x 实测（超额） |
| 叠加 pp / layerwise / SSD | Phase 3 特性叠加 | pp/SSD 未开始；**layerwise 对 DSV4 豁免（W1）** |

### 2.4 与需求相关的关键说明

- **命中率基线的口径**：Phase 1 的 96.88% 是单机 131072 口径、单前缀复用的实测；需求目标的 80D/20S 是双侧 D+S 合计 ~90% 的复合口径，两者不对等，Phase 4 需按需求口径重新构造数据（chenbo 数据集工具）。
- **吞吐对比基组**：需求要求对照组为「不开池化 + 不开 SSD、也不开 HBM、完全重算」。Phase 1 的冷灌首算（约 19.5s）即等价于这个完全重算基组，命中后约 1.7s，故加速约 11 倍。
- **layerwise 豁免（W1）**：DSV4 有 5 个 cache spec，`build_layerwise_reuse_layout` 仅支持每层 2 spec，`use_layerwise=true` 必崩——原始需求要求叠加 layerwise，但这是已知功能硬限（#12853）、非本任务缺陷，A3 场景按 non-layerwise 纯池验证，matrix #9/#10 标 EXPECTED_FAIL。
- **开放项**：memcache SSD 已知问题已修复，但**收益具体提升值仍待 Phase 3/4 实测**；这是需求文本中自带的未知数，需实测回填。

---

## 3. 当前状态与下一步

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 1 | 单机基线（mooncake + memcache 三形态） | ✅ 两轮 PASS（2026-09-09） |
| Phase 2 | 双机 HCCS 跨机 KV 传输 | 阻塞：等第二台 128G 机器 |
| Phase 3 | 特性叠加（SSD / pp / 全栈） | 待开始 |
| Phase 4 | 200K / 命中率 80/20 / 吞吐 5~8x 验收 | 待开始 |

**已确认解决项**：
- 机型口径：128G = 每卡 128GB / 整机 1024GB（已确认）；64G = 16 die × 32G / 整机 512GB（已确认）。
- 池化基本能力 + HBM 关闭 + 完全重算基线：Phase 1 全部 P0 通过。

**下一步动作**（按依赖顺序）：

**A. 立即可做（单机 135，不依赖第二台机器）**
1. 澄清「2 个后端」口径：与需求方确认 HIXL/Memfabric 与落地采用的 mooncake/memcache 的对应关系，避免验收口径对不上。
2. Phase 3 之 SSD 分级（单机即可）：mooncake.json 加 SSD 路径/容量、memcache 用 mmc-local.conf 启用 SSD medium 层；构造超 DRAM 池容量数据集，验证淘汰到 SSD、SSD 命中率占比 ~20%，并**回填 memcache SSD 收益实测值**（需求开放问号）。
3. Phase 3 之 pp 叠加（单机验证）：vllm 加 `--pipeline-parallel-size 2`（pp_size × tp_size ≤ 16 die），验证 pp + 池化可共存。

**B. 第二台 128G 机器到位后**
4. 申请第二台 128G 机器 → 解 Phase 2。
5. Phase 2 双机 HCCS 核心：双机 PD 分离 + 跨机 KV 传输（mooncake、memcache 各一遍，plan §5.2）。
6. Phase 3 全栈：pp + SSD 逐步叠加（验证矩阵 #4/#8 组合）。

**C. Phase 4 性能验收（正式收口）**
7. 用 chenbo 数据集工具构造 200K / 128K 输入+1K 输出 / 90% 重复率数据。
8. 采集 80% DRAM + 20% SSD 命中率、Prefill 5~8x（对照组 = 完全重算），正式验收通过。

**D. 64G 机型后续**
9. 64G（16 die × 32G / 512GB）机器到位后，确认并落地 64G plan，复用 128G 方法与结果；注意其 512GB 预算比 128G 紧，200K 可能需降 max_len 或分阶段。

---

## 4. 相关文档

| 文件 | 位置 | 作用 |
|---|---|---|
| A3_HCCS 总索引 | A3_HCCS/README.md | 概述 / 机型口径 / 公共技术栈 / 进展 |
| 128G 验证方案 | A3_HCCS/128G/verification_plan.md | 128G 全量方案（本文 Goal 承接处） |
| 64G 验证方案 | A3_HCCS/64G/verification_plan.md | 64G 骨架（口径确认后细化） |
| 对外总览 | A3_HCCS/128G/verification_full_report.md | 自包含对外介绍（方案 + Phase 1 细节） |
| 需求分析 | A3_HCCS/A3_HCCS_requirements_analysis.md | 本文档 |
| 上游专项需求 | Layerwise-Pooling-Optimization/01_requirements_analysis.md | SR20260820223202 池化性能专项（可观测性配套） |