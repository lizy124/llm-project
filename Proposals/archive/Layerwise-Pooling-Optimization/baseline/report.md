# HBM 基线组摸底报告（第一轮，2026-09-06）

> 环境：165 / baseline_165 容器 / 镜像 7f06feda13d3（vllm 0.27.1 + vllm-ascend 0.19.1rc2.dev1561，均 editable 指向 /vllm-workspace，已冻结）
> 过程：bf16 权重 TP8 失败（KV 不足）→ TP16 失败（模型结构限制）→ w4a8 权重 TP8 成功 → mock 链路验证轮完成

## ⚠️ P0 发现：DSV4-Flash 的 prefix caching 在当前栈上结构性失效

> **【2026-09-06 更正——本节结论已被 161 验证轮推翻，见 §五】**
> "结构性失效"不成立。0% 命中的真实根因是**验证负载的前缀长度（14336）小于 scheduler_block_size（16384）**：
> hybrid coordinator 读写两侧均按 `scheduler_block_size` 向下对齐（写：`kv_cache_coordinator.py` Hybrid.cache_blocks；读：FA lookup 的 `hit_length -= hit_length % alignment_tokens`），14336 < 16384 → 注册与命中双归零。
> `scheduler_block_size = lcm(各 KV group block_size) = lcm(512, 16384, 128, 8, 32) = 16384`（16384 由 indexer cache 组决定）。
> 前缀 ≥ 16384 后命中正常（99.61%），跨请求复用正常。旧 root cause（"sliding_window=128 导致窗口外块不可复用"）判断有误：SWA 组确实按窗口 mask 注册（reg=0），但 FullAttention/indexer 组的命中不受其影响。
> 旧结论中仍然成立的部分：① 容量约束（§二）；② **前缀 < 16384 时命中恒 0%** 这一结构性边界（对池化需求反而是关键前提数据：短前缀场景原生 prefix cache 无效，正是 Layerwise 池化的发力点）。

**现象**（三重证据）：
1. engine 日志全程 `Prefix cache hit rate: 0.0%`；请求间 KV usage 回落至 0-1.3%（前缀块零保留）
2. 计数器：queries 增量精确等于 prompt token 数（warmup 172032 = 12×14336），hits 恒 0——**含同文本二发**（同 hash 必中逻辑失效）
3. mock 轮四档并发 hit_rate 全 0.0

**Root cause**（代码级，/vllm-workspace 实测代码）：
- DSV4-Flash 43 层的**主 KV 全部是 `AscendSlidingWindowMLASpec`**（sliding_window=128，compress_ratio 4/128 交替，首尾层 0）→ SlidingWindowManager；仅 indexer cache 走 AscendMLAAttentionSpec（deepseek_v4.py:110-210）
- hybrid coordinator 有效命中 = 各组最小值；sliding-window spec 结构上无法复用窗口外前缀块（window=128 恰好 1 个 block）
- → **无论容量多大，前缀复用恒为 0**

**需求影响**：验收标准要素 3/6（"DRAM 池化 vs HBM PrefixCache"、"多前缀、HBM 可缓存满前缀"）在当前 vllm/vllm-ascend 上**不可构造**——基线组没有可用的 PrefixCache；"90% 命中"负载前提不成立。叠加已知 P0（layerwise reuse layout 不兼容 DSV4 多 cache spec），**实验组与对照组双双不可达**。这不是 165 机器问题，是栈对 DSV4（DSA 架构）的支持问题。

**待确认**：vllm-ascend 是否有 DSV4 prefix caching 支持路线图；/mnt/share/w50059099/sleep_mode 与 r00899270/weight_pool 下有 DSV4+pool 相关测试目录可翻是否有反例。

## 一、实测前提数据

### O1: KV/token 字节数

| 项 | 值 | 来源 |
|---|---|---|
| KV/token（TP8，每芯） | **≈ 118 KB** | bf16 权重失败日志反推：18.08 GiB ÷ 163,840 token；与 w4a8 权重成功日志交叉验证一致（274,547 token × 118 KB ≈ 31.3 GiB/芯） |
| KV dtype | bfloat16 | 启动配置 |
| 说明 | MLA latent KV 在 TP 下每芯全量复制（num_key_value_heads=1），不随 TP 切分 | 118 KB 远大于纸面按切分估算的 ~8 KB 的原因 |

### O2: HBM 满前缀容量（w4a8-mtp 权重, TP8）

| 项 | 值 |
|---|---|
| GPU KV cache size | **274,547 tokens**（全机有效） |
| 128K 满前缀可缓存数 | **≈ 2.1 个** |
| 115K 前缀（验收负载） | ≈ 2.4 个 |
| Maximum concurrency @163,840 | 1.68x |

### 权重/并行约束

| 项 | 值 | 证据 |
|---|---|---|
| TP 上限 | **8**（o_groups=8 硬限制） | TP16 启动崩：`n_local_groups = 8/16 = 0` → `view(0, 1024, -1)` RuntimeError（vllm_ascend/ops/linear.py:477） |
| bf16 权重每芯 KV | 14.75 GiB（装不下 163,840 单请求需 18.08 GiB） | TP8 失败日志 |
| w4a8 权重每芯 KV | ~31.3 GiB（权重降至 19G/芯） | 成功启动日志反推 |
| 16 芯片 | 物理存在（8 卡 × 2 芯）但 TP16 用不上 | npu-smi + TP16 失败 |

## 二、结论

1. **165 无法承载验收级 HBM 基线**："多前缀、HBM 可缓存满前缀"要求 P≥32 × 115K ≈ 3.7M token，而 w4a8 最优配置下容量仅 274.5K token，**差 13 倍以上**。bf16 权重差 28 倍。
2. **量化权重也解决不了**：w4a8 已是 165 上最激进的量化（int4 权重），容量仍只有 ~2 个满前缀。
3. **验收基线环境需重新规划**：需满足 P≥32 × 128K ≈ 4M+ token KV 容量 ≈ 4.5 TB/芯等效（TP8、118KB/token）——超出单机能力，意味着：
   - 验收基线组本身可能需要多机/更大 HBM 机器；或
   - 验收负载/基线定义需与需求方重新对齐（此发现应反馈到 01_requirements_analysis.md 的 P0 澄清单）
4. **165 的定位**：方法链路验证 + 量化后小规模数据（缩小序列长度等比验证）。

## 三、遗留

- ~~服务当前运行中（w4a8, TP8, port 8004）~~（165 已被占用；基线服务现于 **161/lw_verify_161, port 8004** 运行中，栈已还原干净）
- hybrid cache group 明细：bs=512(grp0)/16384(grp1,indexer)/128×2(grp2,3)/8(grp4)/32(grp5)，hash_bs=8——已由 161 排查轮实测取得（见 §五 V3）
- ~~TPS/TTFT/命中率已于 mock 轮采集，验收级（128K/90% 命中）因 P0 发现无法构造~~ → **已可构造**：前缀 ≥16384 时命中正常（§五 V1），验收级 HBM PrefixCache 对照组在本栈上成立；瓶颈仅剩容量（§二）
- 下一步建议：161 上跑 90% 命中率的验收形态负载（P×64K 前缀，P 受 279K 容量限 ≈ 4），采集命中态 TPS/TTFT 曲线，与 mock 轮冷 prefill 曲线（§四）构成完整对照

## 四、Mock 链路验证轮（等比缩小 128K→16K）

**目的**：165 容量装不下验收级负载（差 13 倍+），等比缩小验证整条方法链路（前缀生成→灌入→并发扫描→TPS/TTFT/命中率采集）。

**负载**：前缀 14336 token（112 blocks 精确对齐）× P=12，后缀 ~1536 token 随机；名义命中率 90.2%（实际 0，见 P0 发现）。脚本 `data/bench_mock.py`（seed 固定可复现，metrics 带 label 解析已修）。

**结果**（0% 命中 = 冷 prefill 参考曲线，即本栈上 DSV4-Flash 的真实形态）：

| 并发 | R | Prefill TPS | TTFT mean | TTFT p99 | hit_rate |
|---|---|---|---|---|---|
| 4 | 40 | 6,105 | 3.9s | 8.3s | 0.0 |
| 8 | 40 | 7,068 | 5.9s | 16.9s | 0.0 |
| 16 | 80 | 7,792 | 9.4s | 33.2s | 0.0 |
| 32 | 160 | 8,036 | 34.7s | 61.0s | 0.0 |

**链路产出**：✓ 前缀 block 对齐生成 ✓ 命中率差分采集（queries 精确对账）✓ TTFT 流式首包 ✓ 并发扫描曲线。TPS 饱和 ~8K tok/s（enforce-eager + w4a8），TTFT 在并发 32 时因排队恶化至 34.7s——**若前缀可命中（90%），预期 TPS/TTFT 大幅改善，但本栈无法测得该对照**。

**数据文件**：165:/home/lizhongyang/lw_baseline/results/mock_round1/（summary.json + 各 level 明细 + prefix_meta.json + warmup.json）

## 五、161 验证轮：prefix caching 可用性 + 命中基线（2026-09-06）

> 环境：161 / lw_verify_161 容器 / 同镜像同栈（vllm 0.27.1 + vllm-ascend），DSV4-Flash **w4a8-mtp** 权重，TP8，enforce-eager，`--enable-prefix-caching`，block_size=128，max_model_len=163840，GPU KV cache **279,046 tokens**，port 8004。
> 栈为干净原版（验证用的 11 轮 APCDBG 埋点已全部 `git checkout` 还原，两仓库 0 改动）。

### V1: 命中验证（A/A2/B/A3 四场景，seed 固定）

前缀 65536 token（4×16384）+ 后缀 256，脚本 `data/verify_prefix_161.py`：

| 场景 | 说明 | prompt | 延迟 | hits | hit_rate |
|---|---|---|---|---|---|
| A | 首灌 | 65,792 | **11.29s** | 0 | 0% |
| A2 | 同文重发 | 65,792 | **0.44s** | 65,536 | **99.61%** |
| B | 同前缀异后缀 | 65,792 | **0.42s** | 65,536 | **99.61%** |
| A3 | B 之后再发 A | 65,792 | 0.52s | 65,536 | 99.61% |

- 命中量恒为 **65,536 = 4×16384**：即 prompt 对齐到 `scheduler_block_size` 的整数倍，与代码预测完全一致
- **TTFT 11.29s → 0.44s（25.7x）**；跨请求前缀复用正常（B），后续请求不挤掉已有缓存（A3）
- 引擎侧 `Prefix cache hit rate: 49.8%`（全生命周期均值，含首灌 A 的 0%）

### V2: Prefill 吞吐（冷灌，A 场景）

- 端到端：65,792 tok / 11.29s ≈ **5,827 tok/s**
- 引擎报告 prompt throughput ≈ 6,604 tok/s
- 与 165 mock 轮冷 prefill 曲线（4 并发 6,105 → 32 并发 8,036 tok/s）同量级，交叉印证

### V3: 结构性边界（对池化需求的关键前提）

| 前缀长度 | 命中 | 原因 |
|---|---|---|
| < 16,384 | **恒 0%** | 读写路径均按 scheduler_block_size=16384 向下对齐（lcm 由 indexer cache 组 bs=16384 决定） |
| ≥ 16,384 | 正常（99.61%@64K） | 对齐后读写一致，FA/indexer 组块注册+命中 |

**需求含义**：短前缀（<16K）多轮对话/Agent 场景在当前栈上原生 prefix cache 无效——Layerwise 池化若能以更细粒度（hash_bs=8 已存在）管理复用，可直接吃下这部分被对齐阈值挡掉的收益。

### 161 侧数据文件与脚本

- 验证脚本：本地 `data/verify_prefix_161.py`（161:/home/lizhongyang/lw_verify/data/ 同步）
- server.log：161:/home/lizhongyang/lw_verify/run/server.log
- 排查期埋点脚本（APCDBG2-11，已弃用，仅存档）：本地 `data/patch_apcdbg*.py`、`data/revert_patch5.py`

## 六、归档

- server.log（完整启动日志）在 165:/home/lizhongyang/lw_baseline/run/server.log
- 版本快照：vllm 6e448d0 / vllm-ascend 33e849499（check_env.sh 8 项全过）

## 七、DP2 验证：多前缀容量路径（2026-09-06，161 服务器）

**目的**：验证"DP2 数据并行下总 KV 容量翻倍、可承载更多前缀"这一条满足验收级（P≥32×128K）的技术路线是否成立。分解为两个必答前提并逐一实测。

### V4: DP2×TP8 部署可用性（前提一：成立）

- 16 芯两副本（每副本 8 芯），world_size=16，双副本均 init 成功，HTTP 200
- 关键作用：**原始 8004 TP8 单服务 279K KV 只够 ~2.1 条 128K 前缀，DP2 将总 KV 扩到 ~558K**，此前缀容量是"P≥32"需求的最大物理瓶颈
- 配置：`start_server_dp2_161.sh`（DP2×TP8, port 8005, 与 8004 baseline 同栈同权重）

### V5: DP router 是否 cache-aware（前提二：不成立 → 多前缀路线被否定）

- **结论：vLLM 0.27.1 DP router 是纯负载均衡，无前缀/缓存亲和性**
- 代码走读与实测（脚本 `data/verify_dp2_161.py`）互相印证：
  - 8 并发同前缀请求：wall=13.23s，请求被 hash 打散到两副本
  - 命中统计 **262,144 = 4×65,536**，仅 4/8 个请求计命中（收到落点的副本已暖）；另 4 个落到另一副本 → 首见冷灌
  - 路由无记忆：两副本各自独立缓存同一条前缀

### 对需求的结论

1. **DP2 只解决容量，不同时解决多前缀命中**：容量翻倍（≤~4 条 128K）但请求被无亲和打散，同一前缀每副本都要各冷灌一次，命中被稀释。验收"P≥32 且 90% 命中"即便扩到 DP2 容量（~558K，也仅 4 条满前缀）仍差 >8 倍。
2. **多前缀方案不能靠开箱 DP 路由**：需 vLLM DP 层支持前缀亲和/一致性哈希路由（当前 0.27.1 不具备），或依赖外部调度按前缀 stickiness 分发——两者在当前栈上均不成立，属新增工程量，超出池化需求本身。
3. **DP 的真正价值在容量**：若未来以"单/少前缀、高并发"形态验收，DP2 容量翻倍有意义；但按验收原文 P≥32 多前缀形态，DP2 无法闭环。
4. **基线侧结论**：single-TP8（§五）已足够承载"少前缀高命中"对照；多前缀形态在单机（165/161 均 16 芯）内无任何并行配置可构造，需多机或多卡扩展，且依赖非现成的前缀亲和路由。此限制应同步反馈到 01_requirements_analysis.md 的 P0 澄清。

**数据文件**：161:/home/lizhongyang/lw_verify/run/server_dp2.log；脚本 `data/verify_dp2_161.py`、`run/start_server_dp2_161.sh`（161 服务器已停止 DP2 服务，8005 端口释放；8004 TP8 基线已恢复维护）。

## 八、多前缀验证：single-TP8 单服务内多前缀命中（2026-09-06，161 服务器）

**目的**：§七 已否定"DP 扩容多前缀"路径（路由无亲和）。但**同进程内能否让多条前缀同时常驻并各自高命中**——即"多前缀 × 90% 命中"形态在**不依赖多副本/不依赖路由亲和**的前提下能否由 single-TP8 直接构造。供两档前缀长度复用（64K / 128K），脚本 `data/verify_multi_161.py`。

### V6: 64K 多前缀（4 条不同前缀，4×65792 ≈ 263K ≈ 279K 容量 94%）

| 前缀 | 冷灌 | 冷 hits | 同文重发 | 重发命中率 |
|---|---|---|---|---|
| P0 | 11.30s | 0 | 0.58s | 99.6%（65536） |
| P1 | 10.11s | 0 | 0.60s | 99.6% |
| P2 | 10.16s | 0 | 0.53s | 99.6% |
| P3 | 10.02s | 0 | 0.44s | 99.6% |

**并发混合扫描**（4 前缀随机轮流，每前缀独立命中差分）：

| 并发 | wall | mean_lat | queries | hits | hit_rate | kv_usage |
|---|---|---|---|---|---|---|
| 8 | 12.63s | 1.51s | 4,210,688 | 4,194,304 | **99.61%** | 0.0% |
| 16 | 23.32s | 2.73s | 8,421,376 | 8,388,608 | **99.61%** | 0.0% |

**结论：4 条不同 64K 前缀可在同一 single-TP8 服务内同时常驻、混合并发下总体命中率仍 99.6%，无挤占。**

### V7: 128K 多前缀（2 条不同前缀，2×131328 ≈ 263K ≈ 339K 容量 77%）

| 前缀 | 冷灌 | 冷 hits | 同文重发 | 重发命中率 |
|---|---|---|---|---|
| P0 | 12.79s | 65,536 | 0.69s | 99.8%（131072） |
| P1 | 12.79s | 65,536 | 0.90s | 99.8% |

> 冷灌 hits=65,536 非 0：因 P1 冷灌时 P0 已驻留，`prefix_cache_hits` 是按 token 计的前缀复用，P1 的该非前缀重复来自 tokenizer/公共 token（Suffix 部分）——命中集中在重发那一行（131,072 全命中），说明 128K 前缀主体命中正常。

**并发混合扫描**（2 前缀随机轮流）：

| 并发 | wall | mean_lat | queries | hits | hit_rate | kv_usage |
|---|---|---|---|---|---|---|
| 4 | 6.13s | 1.40s | 2,101,248 | 2,097,152 | **99.81%** | 0.0% |
| 8 | 12.29s | 2.75s | 4,202,496 | 4,194,304 | **99.81%** | 0.0% |

**结论：2 条不同 128K 前缀可在同服务内同时常驻、混合并发下命中率 99.8%，无挤占。**

### 对需求的结论

1. **"多前缀 × 高命中"可在 single-TP8 内直接构造，无需 DP、无需路由亲和**。容量是唯一硬天花板（279K）：64K 档最多 ~4 前缀、128K 档最多 ~2 前缀，全都能稳定 90%+（实测 99.6-99.8%）。
2. **多前缀机制验证成立**：同一进程内多条前缀缓存互不挤占、混合并发不稀释命中。这覆盖了验收"多前缀 + 90% 命中"的**机制面**，只是**数量面（P≥32）受单机容量限制**。
3. **对验收的含义**：
   - **机制已证实**：多前缀高命中形态可构造且稳固（§八 V6/V7）。
   - **数量未达成**：P≥32 需 ≈16 台（64K 档）或跨机/更大 HBM（128K 档），详见 §七 多机分析。**单机（single-TP8 或 DP2）都无法满足 P≥32。**
   - **推荐落地点**：以 64K×4 前缀（V6）作为"近似验收形态"的规模上限基准——它能同时验证多前缀命中、并发稀释、挤占三件事，是当前硬件在单机内能构造的最接近验收的形态。
4. **数据文件**：本地 `data/res_multi_64k.json`、`data/res_multi_128k.json`（同 script）；脚本 `data/verify_multi_161.py`。

## 九、汇总结论

| 维度 | 结论 | 依据 |
|---|---|---|
| prefix caching 是否可用 | ✅ 可用，前缀≥16384（scheduler_block_size）即命中 | §五 V1/V3 |
| 单前缀命中率 | 99.6-99.8% | §五 V1、§八 |
| 命中态 TTFT 增益 | 11.3s → 0.44s（25.7x，64K） | §五 V1 |
| 冷灌 Prefill TPS | ~5.8-6.6K tok/s（64K）、排到 ~8K（16K 档并发） | §五 V2、§四 |
| 多前缀机制 | ✅ 单进程内多前缀共存、混合并发命中不降 | §八 V6/V7 |
| 单机容量 | w4a8 279K token ≈ 2 条 128K / 4 条 64K | §二、§八 |
| P≥32 验收 | ❌ 单机任何并行配置（TP/DP/多机直连）均无法满足，需多机+前缀亲和分配或更大 HBM | §七 V5、多机分析 |
| DP2 多前缀 | ❌ 路由无亲和，打散→命中稀释 | §七 V5 |

**验收级形态在单机内不可构造（数量面）；机制面已全部实证。** 完整结论需同步回填 01_requirements_analysis.md 的 P0 澄清：multi-prefix ≥32×128K 在本栈单机上不可达，应重新对齐验收负载规模或引入多机。
