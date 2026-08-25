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

**高性能需求，本质是"容量换性能"的权衡工程**：长序列场景下 HBM 装不下 KV Cache，用 Mooncake/MemCache 池化把 KV 溢出到 DRAM/SSD 扩容；代价是读取带宽下降，用 **Layerwise 计算与传输并发（overlap）** 把这个代价掩盖掉，目标是 DRAM 池化相比理想 HBM 只劣化 ≤5%。

### 2.2 需求价值解读（注意表述里的真实含义）

原文"同样命中率下，命中 DDR/SSD 性能相比命中 HBM **性能提升**"需要正确理解——DDR/SSD 带宽必然低于 HBM，单次命中读取不可能更快。真实逻辑是**系统级对比**：

- **不池化**：HBM 容量不足 → 前缀装不下 → 命中率下降/触发重算 → 性能差
- **池化**：DRAM/SSD 扩容 → 维持 90% 高命中 → 端到端吞吐提升

所以"提升"的参照系是**容量不足的场景**，而不是 HBM 本身。验收标准里说的"劣化不超过 5%"才是对 HBM 的真实关系——池化是**用可控的劣化换取容量**。

"系统的 MFU 持平或优于 H200"则是与竞品（NVIDIA H200，1077GB HBM3e）的宏观对标目标。

### 2.3 应用场景画像

| 要素 | 数值 |
|---|---|
| 业务形态 | Agentic（多轮工具调用，前缀高度共享） |
| 命中率 | 普遍 >90% |
| 序列长度 | 平均 200K（验收用 128K 输入） |
| 核心矛盾 | HBM 容量 vs KV Cache 总量 |

### 2.4 核心技术拆解（四块）

#### （1）多级存储管理
Mooncake/MemCache 池化管理 HBM/DRAM/SSD 三级存储，KV Cache 按策略溢出。

#### （2）Layerwise 传输加速（本需求核心）
- 传统（非 layerwise）：整个请求 KV 加载完才能开始算 → 传输时间全额暴露
- Layerwise：按层粒度流水线——第 N 层 KV 在传输时，计算在算第 N-1 层 → **用计算时间掩盖传输时间**
- 理想状态传输完全 overlapped，`not overlapped` 时间 ≈ 0

#### （3）特性叠加
要求与 MTP（多 token 预测/投机解码）、DCP、LayerSplit 等并行特性**可叠加不冲突**——组合兼容性要求，组合矩阵不小。

#### （4）极光监控大盘指标（与 AR20260820031213 交汇）
- `not overlapped` 的 KV 池化读取时间 metric → 即 AR 需求验收标准 2"池化 KV Cache 加载耗时（请求粒度）"的深化版，还要**区分总耗时和未被掩盖的耗时**
- KV Cache 延迟释放 req 排队 metric → **就是 AR 需求验收标准 3**（延迟释放请求数量）

### 2.5 验收标准深度解读

| 要素 | 内容 | 关键点 |
|---|---|---|
| 模型 | DeepSeekV4 Flash | 项目历史上踩过 DSV4 KV pool 的坑（见 2.7 节） |
| 负载 | 128K 平均输入，90% 命中 | 需要专门构造多前缀共享数据集 |
| 对比 | DRAM 池化 vs HBM PrefixCache | 最优并发条件下各自调到最优 |
| 指标 | **Prefill TPS** | 注意：验收只看 Prefill TPS，不是端到端吞吐（虽然价值描述里提了端到端） |
| 目标 | 劣化 ≤5% | 拿"理想 HBM"作上限基线，定义池化代价上限 |
| 基线定义 | 多前缀、**HBM 可缓存满前缀** | 基线是理想情况：所有前缀都能装进 HBM。刻意设置的严格基线 |

**为什么验收指标是 Prefill TPS**：KV Cache 加载发生在 prefill 阶段（命中前缀的请求要先把 KV 从 DRAM 拉回来再继续算），Layerwise overlap 的收益直接体现在 prefill 吞吐上；decode 阶段基本不受池化读取影响。

**注意 SSD 路径**：需求价值提了 DDR 和 SSD，但验收标准只写了 DRAM 池化——SSD 路径本阶段可能不验收（带宽更低，劣化更难控制在 5% 内）。

### 2.6 与第一个需求（AR20260820031213）的关系：配套需求

```
SR20260820223202（本需求）          AR20260820031213（第一个需求）
高性能：Layerwise overlap 优化   ←──  可观测性：指标埋点 + 极光平台对接
         │                                    │
         └────────── 指标交汇 ────────────────┘
   not overlapped 读取时间  ←→  池化KV加载耗时（请求粒度）
   延迟释放 req 排队        ←→  KV Cache 延迟释放请求数
```

- 本需求做**优化本体**，AR 需求做**度量工具**
- 本需求验证"劣化 ≤5%"是否达标，恰恰需要 AR 需求的指标来证明（not overlapped 时间小说明掩盖效果好）
- "极光监控大盘"就是 AR 需求里说的"监控平台"

### 2.7 与当前 PD/池化验证工作的直接关联

结合项目记忆和 PD_POOLING_TEST_STRENGTHENING_930.md 专项文档：

1. **Layerwise 模式已在验证范围内**：项目记录过"non-layerwise mode saves after request completion"（非 layerwise 模式请求完成后才 save），说明 mooncake transfer 存在 layerwise / 非 layerwise 两种模式——本需求的"计算与 KV 分层并发加载"就是 layerwise 模式，当前团队是直接验证方。
2. **DSV4 的坑会直接复现**：`cache_transfer_granularity=4096` 太大导致短 prompt `num_tokens_to_save=0` 跳过 save——DSV4 Flash 是本需求验收模型，此问题的修复是前置条件。
3. **对照实验的关键控制变量**：项目验证过"vllm 默认 enable-prefix-caching 导致本地命中，阻碍跨 worker KV transfer"——本需求基线恰恰是 HBM PrefixCache，实验组是 DRAM 池化，**两组的 prefix caching 行为必须严格控制**，否则测出来的对比不干净（比如池化组意外发生本地命中，数据全废）。
4. **930 专项时间线里的落点**：专项 9/12–9/18 有"AscendStore/Layerwise 是否入 nightly 有结论"，9/19–9/24 做性能基线——本需求的验收测试（DSV4 Flash、128K、90% 命中、最优并发）就是性能基线工作的一部分。
5. **51 服务器环境**：8×A3 每卡 5G DRAM 池化配置的现成环境可以直接复用。

---

## 三、风险与挑战点

- **≤5% 劣化是硬指标**：layerwise overlap 实现不充分（not overlapped 时间长）直接不达标，且是在"最优并发"下测，负载构造要精确。
- **特性叠加矩阵**：池化 × MTP × DCP × LayerSplit 的兼容性验证工作量可能超预期。
- **指标依赖**：not overlapped metric 依赖 AR20260820031213 需求交付，两个需求有进度耦合。
- **SSD 路径**：验收标准未覆盖，本阶段可能不验收，但需求价值里有预期，需确认范围边界。

---

## 四、一句话总结

高性能/RP3 需求：面向 Agentic 长序列（200K、命中 >90%）场景，用 Mooncake 池化把 KV Cache 从 HBM 溢出到 DRAM/SSD 扩容，核心手段是 Layerwise 计算与传输并发加载掩盖搬运耗时；验收锚点为 DeepSeekV4 Flash、128K 输入、90% 命中下，DRAM 池化的 Prefill TPS 相比理想 HBM PrefixCache 基线劣化不超过 5%。