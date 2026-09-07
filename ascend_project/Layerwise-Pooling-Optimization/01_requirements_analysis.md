# 【RP3】【高性能】【vLLM】池化性能专项优化，Layerwise传输加速实现吞吐提升 — 需求分析

> 文档编号：SR20260820223202
> 创建人：彭晓 00541980（VisionIT）
> 创建时间：2026-08-20 11:50:16
> 分析时间：2026-08-25
> 需求类别：高性能 / RP3
> 关联需求：AR20260820031213【vLLM-Ascend】【RP3】【易用性】LLM推理监控平台对接vLLM优化（可观测性配套需求，分析文档同目录）

---

## 一、原始需求记录

### 1.1 需求价值

- 在同样的 KVCache 缓存命中率下，KVCache 缓存命中 DDR、SSD，性能相比缓存命中 HBM，性能提升。
- 在缓存命中 DDR、SSD 的情况下，系统的 MFU 持平或优于 H200。

### 1.2 应用场景

Agentic 业务场景命中率普遍 >90%，序列长度较长，平均序列长度到 200K，需要通过 DDR 和 SSD 扩充 KV Cache 存储容量，同时保证性能下降在业务接受范围内。

### 1.3 需求描述

1. 通过 MemCache/MoonCake 池化方案管理 HBM/DRAM/SSD 多级存储，同样 KV Cache 缓存命中率下，部分 KV Cache 溢出到 DRAM 或者 SSD 存储，端到端吞吐提升。
2. 关键技术：
   - 支持计算和 KV Cache 分层并发加载，掩盖 KV 传输耗时。
   - 支持叠加 MTP、DCP、LayerSplit 等特性。
   - 极光监控大盘支持观测未掩盖 KVC 池化（not overlapped）读取时间的 metric，KVC 延迟释放 req 排队的情况 metric。

### 1.4 验收标准

| # | 要素 | 内容 |
|---|------|------|
| 1 | 模型 | DeepSeekV4 Flash |
| 2 | 负载 | 平均 128K 输入，平均 90% KVCache 命中，最优并发条件 |
| 3 | 对比 | DRAM 池化 vs HBM PrefixCache |
| 4 | 目标 | DRAM 池化相比 HBM PrefixCache 性能劣化不超过 5% |
| 5 | 性能指标 | Prefill TPS |
| 6 | 性能基线 | HBM PrefixCache，多前缀、HBM 可缓存满前缀 |

---

## 二、需求分析

### 2.1 需求定性（一句话）

**用 DRAM/SSD 容量换 HBM 容量不足的工程权衡**：Agentic 场景（200K 序列、>90% 命中）下 KV Cache 总量远超 HBM，池化扩容维持高命中率；代价是读带宽下降，用 Layerwise 计算与传输 overlap 把暴露时间压进验收预算（Prefill TPS 劣化 ≤5%）。

### 2.2 原始需求的内部矛盾（须向需求方澄清）

"命中 DDR/SSD 比命中 HBM 性能提升"与验收标准不自洽，存在两种合法解读，对应完全不同的实验设计：

| 解读 | 对比组 | 结论性质 |
|---|---|---|
| A（价值叙事） | 池化扩容 vs 不池化（HBM 容量不足 → 命中率跌/重算） | 池化"提升"成立，端到端吞吐对比 |
| B（验收标准） | DRAM 池化 vs 理想 HBM（满前缀可缓存） | 池化必然劣化，"≤5%"是代价上限 |

验收标准选了 B——用不可能赢的理想基线给池化代价封顶。这本身合理（防止劣化失控），但"性能提升"的表述在验收口径下不成立，对外对齐时必须统一口径。**建议：验收按 B 执行，价值论证按 A 补一组对照（同并发下限制 HBM 前缀容量 vs 池化扩容）。**

### 2.3 P0 前置风险：DSV4 Flash × Layerwise 兼容性

已确认的技术约束：**DSV4 MLA 架构每层有多个 cache spec，与 layerwise reuse layout 不兼容**（layerwise_cache_layout.py 仅支持 1 main + 1 indexer.k_cache）。而验收模型就是 DSV4 Flash、核心手段就是 layerwise。

两条出路：
1. 扩展 layerwise layout 支持 DSV4 多 cache spec——工作量未评估，可能是本需求最大隐性成本
2. 验收改用非 layerwise reuse 的 overlap 路径——需确认该路径是否存在且性能够

**此问题不解决，验收标准技术上不可达，列为 P0 确认项。**

### 2.4 核心技术拆解

1. **多级存储管理**：Mooncake/MemCache 池化管理 HBM/DRAM/SSD，KV Cache 按策略溢出。
2. **Layerwise 传输加速（核心）**：第 N 层 KV 传输时计算第 N-1 层，用计算时间掩盖传输时间。注意"not overlapped ≈ 0"不现实——首层传输无前置计算可掩盖、尾层计算后无后续传输，层间必然有气泡。目标是**最小化暴露**而非归零，5% 预算应理解为"允许部分暴露"。
3. **特性叠加**：池化 × MTP × DCP × LayerSplit 组合兼容，验证矩阵不小。
4. **监控指标**（与 AR20260820031213 交汇）：not overlapped 池化读取时间（区分总耗时与未掩盖耗时）、KV Cache 延迟释放 req 排队数。

### 2.5 验收标准逐条审查

| 要素 | 原文 | 问题 | 修正建议 |
|---|---|---|---|
| 模型 | DeepSeekV4 Flash | 与 layerwise 存在兼容性矛盾（见 2.3） | P0 确认兼容路径 |
| 负载 | 128K 输入、90% 命中 | "平均 90%"定义模糊——按 token 还是按请求？ | 明确：多前缀共享数据集，按 token 计命中率；命中部分须从 DRAM 池化读取，排除本地 HBM 命中污染 |
| 并发 | 最优并发条件 | 未定义 | 两组各自扫并发取峰值 Prefill TPS 点对比，并发点须记录 |
| 指标 | Prefill TPS | 遗漏 TTFT——池化读取直接拉长首 token 延迟，用户可感知 | 增加 TTFT 劣化约束（建议同样 ≤5% 或单独给预算） |
| 目标 | 劣化 ≤5% | 合理，作为代价上限 | 保留 |
| 基线 | HBM 可缓存满前缀 | 理想基线，实际部署达不到；测出的是"代价上限"而非"实际代价" | 保留（作严格上限），报告中注明基线性质 |
| （价值） | MFU 持平 H200 | 无测量方法、无 H200 环境，不可验收 | 降级为方向性目标，不列验收项 |
| （范围） | DDR **和 SSD** | 验收只写 DRAM；SSD 带宽更低，≤5% 大概率达不到 | 向需求方确认 SSD 是否本期范围 |

**Prefill TPS 作为主指标成立**：KV 加载只发生在 prefill 阶段，layerwise 收益集中于此；decode 阶段基本不受池化读取影响。

### 2.6 实验设计前置条件

项目已验证"vllm 默认 enable-prefix-caching 导致本地命中，阻碍跨 worker KV transfer"。本需求基线是 HBM PrefixCache、实验组是 DRAM 池化，**两组的 prefix caching 行为必须显式控制**——池化组若发生意外本地命中，对比数据全部作废。这是实验设计的前置条件，不是测试细节。

### 2.7 与配套需求及当前工作的关系

- **AR20260820031213（可观测性）**：本需求做优化本体，AR 需求做度量工具。验收"劣化 ≤5%"需要 not overlapped metric 证明掩盖效果，两需求进度耦合。
- **DSV4 既有问题**：`cache_transfer_granularity=4096` 导致短 prompt `num_tokens_to_save=0` 跳过 save，修复是前置条件。
- **930 专项时间线**：9/12–9/18 "AscendStore/Layerwise 是否入 nightly" 出结论，9/19–9/24 做性能基线——本需求验收测试即性能基线工作的一部分。
- **51 服务器环境**：8×A3 每卡 5G DRAM 池化现成环境可直接复用。

---

## 三、风险与挑战点（按优先级）

| 级别 | 风险 |
|---|---|
| P0 | DSV4 MLA 与 layerwise reuse layout 不兼容，验收路径未确认 |
| P0 | 验收口径矛盾（提升叙事 vs 劣化基线）未与需求方统一 |
| P1 | ≤5% 是硬指标，overlap 不充分即不达标；首/尾层气泡无法消除 |
| P1 | SSD 范围未确认，若纳入本期则 ≤5% 大概率不可达 |
| P1 | not overlapped metric 依赖 AR 需求交付，进度耦合 |
| P2 | 特性叠加矩阵（池化 × MTP × DCP × LayerSplit）验证工作量可能超预期 |
| P2 | 对照组 prefix caching 污染风险，需实验设计阶段显式控制 |

---

## 四、一句话总结

高性能/RP3 需求：Agentic 长序列（200K、命中 >90%）下用池化把 KV Cache 溢出到 DRAM/SSD 扩容，靠 Layerwise overlap 掩盖搬运耗时；验收锚定 DSV4 Flash、128K 输入、90% 命中，DRAM 池化 Prefill TPS 相比理想 HBM 基线劣化 ≤5%。**当前最大不确定性是 DSV4 与 layerwise 的兼容性，须最先确认。**