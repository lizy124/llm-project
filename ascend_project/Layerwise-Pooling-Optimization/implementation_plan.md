# SR20260820223202 Layerwise 池化性能优化 — 实施计划

> 配套文档：设计方案见同目录 [design_proposal.md](design_proposal.md)，需求解读见 [requirements_analysis.md](requirements_analysis.md)
> 代码基线：vllm-ascend upstream/main @ `ff998aad1`，开发分支 `layerwise_pooling`（已创建）
> 计划编制：2026-08-26；执行假设：1–2 名研发 + 51 服务器（8×A3，每卡 5G DRAM 池化配置）可用
> 时间锚点：对齐 PD/池化验证强化 930 专项（9/12–9/18 nightly 结论窗口、9/19–9/24 性能基线窗口）

---

## 一、阶段总览

| 阶段 | 时间 | 目标 | 出口标准 |
|---|---|---|---|
| Phase 0 可行性与环境 | 8/26–8/29 | 裁决两大不确定性（hybrid × layerwise、带宽-计算匹配），环境就绪 | R1/R2 均有明确结论 |
| Phase 1 基线与负载 | 9/1–9/5 | 建立双组基线，量化初始 gap 并分解来源 | 基线 TPS 可复现，gap 分解完成 |
| Phase 2 优化迭代 | 9/8–9/11 | 参数矩阵调优 + not overlapped 指标落地，gap 收敛 | 验收配置锁定，比值 ≥0.95 或残余 gap 定位清楚 |
| Phase 3 叠加与 nightly | 9/12–9/18 | 特性叠加验证，产出入 nightly 结论（对齐 930 窗口） | MTP/DCP 验证记录齐备，nightly 有结论 |
| Phase 4 验收测试 | 9/19–9/24 | 正式对照实验与验收报告（对齐 930 性能基线窗口） | 判定结论 + 证据三件套 |
| Phase 5 长稳与收尾 | 9/25–9/30 | 长稳数据、遗留归档、文档定版 | 长稳 ≥48h，遗留清单有去向 |

```
8/26      8/29  9/1       9/5  9/8      9/11  9/12            9/18  9/19          9/24  9/25      9/30
├─ Phase 0 ──┤ ├─ Phase 1 ──┤ ├─ Phase 2 ─┤ ├──── Phase 3 ─────┤ ├─── Phase 4 ───┤ ├─ Phase 5 ─┤
  可行性裁决      基线+gap       优化收敛        叠加+nightly结论     验收实验          长稳+收口
                              ▲                                  ▲
                    930专项锚点：9/12–9/18 nightly 结论   930专项锚点：9/19–9/24 性能基线
```

Phase 0 若裁决出"DSV4 多 cache family 需适配"（风险 R1），插入 Phase 0.5（预估 1–2 周），Phase 1–2 顺延，第一时间同步需求方调整验收窗口预期——这是本计划唯一预设的插队分支。

---

## 二、分支与代码集成策略

| 分支 | 用途 | 状态 |
|---|---|---|
| `layerwise_pooling` | 本需求主开发分支，基于 upstream/main @ ff998aad1 | 已创建，与 upstream 同步 |
| `kv_metrics_observability` | 指标埋点（7 提交，基于旧 main fa668c649），未合入 upstream | 待整合 |

整合步骤：

1. 将 `kv_metrics_observability` 的 7 个提交 rebase 到 `layerwise_pooling`（改动集中在 metrics.py 新文件与 pool_worker/pool_scheduler 埋点，与 8/21 以来 upstream 变更冲突面预计在 pool_worker.py，逐个解决）。
2. rebase 后跑 `tests/ut/distributed/ascend_store` UT 全量回归，确认指标埋点在最新基线行为不变。
3. not overlapped 指标（设计方案 7.2）作为独立提交叠加，与埋点基础分开评审。
4. 上游 PR 策略：指标 PR（埋点 + not overlapped）与优化类 PR（参数、nightly 用例）分线推进，指标 PR 优先提交，避免评审阻塞本地验收。
5. 验收 tag：Phase 4 正式实验前在 layerwise_pooling 上打 tag，两组对照实验锁定同一 commit。

---

## 三、分阶段任务

### Phase 0 可行性与环境（8/26–8/29）

| 编号 | 任务 | 产出 | 验收口径 | 估计 |
|---|---|---|---|---|
| P0-T1 | 指标分支整合：rebase kv_metrics_observability 到 layerwise_pooling 并回归 | 可用指标的开发分支 | ascend_store UT 全过；冒烟中 4 个指标可见 | 1d |
| P0-T2 | 环境就绪：51 服务器 memcache_hybrid / memfabric_hybrid 1.2 安装核对、hugepages（200000）、DSV4 Flash 权重就位 | 环境检查记录 | kv_pool_runtime 冒烟跑通 | 1d |
| P0-T3 | DSV4 Flash 结构确认：层数、compress_ratios、cache families、block sizes、cache_transfer_granularity 推导值、每 token KV 字节、index_topk | 结构参数表（进实验台账） | 数值齐备，前缀数 N 上界可计算 | 0.5d |
| P0-T4 | **R1 裁决**：DSV4 Flash × layerwise 冒烟，prompt 长度对齐 granularity | 冒烟记录：跑通 / 拦截点 + 适配工作量评估 | save/load 跑通且池内 key 数 >0；或拦截点定位与评估结论 | 1d |
| P0-T5 | **R2 裁决**：带宽-计算匹配表（每层 KV 字节 / DRAM 有效带宽 / 每层计算时间 / W_i 与 C_i 比值） | 一页匹配表 | 覆盖 128K 输入场景，给出可完全掩盖 / 需并发摊薄 / 物理不可达三选一结论 | 1d |
| P0-T6 | Phase 0 出口评审：汇总 R1/R2 结论，确认 Phase 1 计划或触发 Phase 0.5 | 评审纪要 | 两大风险均有裁决与去向 |

### Phase 0.5（条件触发：R1 需适配，预估 1–2 周）

| 编号 | 任务 | 产出 | 验收口径 |
|---|---|---|---|
| P0.5-T1 | 多 cache family 的 layerwise IO 打通：接收线程按 group 分派、hybrid group block 区间映射（复用发送线程 group_builders 基础） | 适配代码 + UT | DSV4 Flash 冒烟跑通（P0-T4 口径） |
| P0.5-T2 | 适配期间同步需求方：验收窗口重估 | 沟通记录 | 书面确认新的 Phase 4 时间 |

### Phase 1 基线建立与负载构造（9/1–9/5）

| 编号 | 任务 | 产出 | 验收口径 | 估计 |
|---|---|---|---|---|
| P1-T1 | 多前缀数据集构造：前缀长度对齐 granularity、设计命中 90%、N 取 HBM 预算与 DRAM 池容量约束的较小值 | 生成脚本 + 数据集 | 脚本参数化可重生成；实测命中率与设计值偏差 <2 个百分点 | 1d |
| P1-T2 | 压测与度量管道：max_tokens=1 的 Prefill TPS、并发扫描（8/16/32/64）、≥3 轮统计、Prometheus 指标抓取 | bench 脚本 + 报告模板 | 单命令产出 TPS/轮间波动/指标快照 | 1d |
| P1-T3 | HBM PrefixCache 基线组：预热后并发扫描 | 基线数据包 | 命中率符合设计；最优并发点确定；轮间波动 <2% | 1d |
| P1-T4 | DRAM 池化实验组初测（默认参数）：灌池 → 压测 | 初始 gap 数据 | TPS 比值、not overlapped、逐层等待分布三项齐备 | 1d |
| P1-T5 | gap 分解：首层暴露 / 带宽不足层 / lookup 开销 / 调度开销占比排序 | gap 分解报告 | 残余劣化来源有量化排序，形成 Phase 2 优化清单 | 0.5d |

### Phase 2 优化迭代（9/8–9/11）

| 编号 | 任务 | 产出 | 验收口径 | 估计 |
|---|---|---|---|---|
| P2-T1 | not overlapped 指标实现（设计方案 7.2：wait_for_layer_load 净等待计时 + overlap_ratio） | 代码 + UT | 两个新指标正确上报，UT 覆盖计时逻辑 | 1d |
| P2-T2 | 参数矩阵调优：prefetch（1/2/4/8）、h2d_stagger_us、transfer batching、shared_buffers、independent_layers | 配置矩阵实验台账 | 每项独立 A/B ≥3 轮；锁定验收配置组 | 2d |
| P2-T3 | lookup 路径 profile 与 Go/No-Go（kv-08 RPC 聚合、kv-17 payload 缩减） | profile 报告 + 决策 | lookup 占调度时间 >5% 才立项，否则记录后不做 | 1d |
| P2-T4 | 阶段复测：验收配置下 gap 收敛验证 | 收敛数据 | TPS 比值 ≥0.95；未达则残余 gap 定位到层 | 0.5d |
| P2-T5 | nightly 用例准备：DSV4 Flash layerwise DRAM 池化用例接入 kv_pool_runtime 框架 | nightly 用例 + 配置 | 本地触发跑通，供 Phase 3 结论窗口使用 | 1d |

### Phase 3 特性叠加与 nightly 结论（9/12–9/18）

| 编号 | 任务 | 产出 | 验收口径 | 估计 |
|---|---|---|---|---|
| P3-T1 | 池化 × MTP 验证（设计方案 6.1 三口径：复用正确性 / 接受率 / TPS 波动） | 验证记录 | 三口径全过或问题单归档 | 1d |
| P3-T2 | 池化 × DCP 验证：P/D 分离下 prefill save → DCP decoder load 的 key 一致性与正确性 | 验证记录 | 命中与输出正确；不一致则出问题单与立项建议 | 1.5d |
| P3-T3 | LayerSplit 范围确认（与需求方），确认后按 6.1/6.2 格式补验证 | 范围说明或验证记录 | 书面确认结论 | 0.5d |
| P3-T4 | nightly 结论（对齐 930 专项 9/12–9/18 窗口） | 入/不入 nightly 的结论文档 | 用例连续 3 天绿，或给出不入的原因与后续计划 | 1d |
| P3-T5 | 指标 PR 推进上游（埋点 + not overlapped） | upstream PR | PR 提交、CI 过；评审推进不阻塞本地验收 | 0.5d |

### Phase 4 验收测试（9/19–9/24）

| 编号 | 任务 | 产出 | 验收口径 | 估计 |
|---|---|---|---|---|
| P4-T1 | 正式对照实验：验收 tag 固定 commit、双组最优并发、≥3 轮、控制变量清单逐项留痕 | 验收原始数据 | 控制变量清单（设计方案 8.4）全项核对通过 | 2d |
| P4-T2 | 极光大盘指标呈现 | 大盘截图 / 链接 | not overlapped、load duration、delayed release、load_keys 可读且语义正确 | 0.5d |
| P4-T3 | SSD 路径预研（不设指标） | 预研数据或可行性说明 | DRAM→SSD 分层读带宽一组数据；介质不可用则如实记录 | 1d |
| P4-T4 | 验收报告：TPS 比值判定 + 证据三件套（比值 / not overlapped / 逐层等待分布） | 验收报告 | 判定结论明确，原始数据可追溯 | 1d |

### Phase 5 长稳与收尾（9/25–9/30）

| 编号 | 任务 | 产出 | 验收口径 | 估计 |
|---|---|---|---|---|
| P5-T1 | 长稳运行（验收配置） | 长稳数据 | ≥48h 无 fatal、指标平稳、TPS 无衰减趋势；排期允许则延至 7 天 | 3d（并行） |
| P5-T2 | 遗留归档：kv-01 layerwise 扩展、SSD 正式支持、CP 集成、MTP trailing block、LayerSplit 后续 | 遗留清单 | 每项有去向（issue / 后续需求 / 关闭） | 0.5d |
| P5-T3 | 文档收口：设计方案、实施计划、验收报告与最终状态一致 | 定版文档 | 三份文档状态同步 | 0.5d |

---

## 四、里程碑

| 里程碑 | 日期 | 内容 | 对齐 |
|---|---|---|---|
| M1 | 8/29 | Phase 0 出口：R1（hybrid × layerwise）与 R2（带宽匹配）双裁决 | — |
| M2 | 9/5 | 双组基线 + gap 分解完成 | — |
| M3 | 9/11 | 优化收敛，验收配置锁定，nightly 用例就绪 | — |
| M4 | 9/18 | 叠加验证齐备，nightly 结论产出 | 930 专项 9/12–9/18 窗口 |
| M5 | 9/24 | 验收判定与报告 | 930 专项 9/19–9/24 性能基线窗口 |
| M6 | 9/30 | 长稳与收口 | 930 专项收口 |

---

## 五、风险登记册

设计层面风险（R1–R6、O1）的定义与应对见设计方案第九节，此处只跟踪进度视角的触发与响应。

| 风险 | 进度触发点 | 响应动作 |
|---|---|---|
| R1 DSV4 hybrid × layerwise 需适配 | P0-T4 冒烟 raise | 启动 Phase 0.5；M2–M5 顺延 1–2 周；8/29 前同步需求方 |
| R2 DRAM 带宽物理不足 | P0-T5 匹配表大面积 W_i > C_i | 调整策略为"并发摊薄 + 深预取"，同步预期；若仍不可达，产出物理边界报告作为验收结论的一部分 |
| R3 指标分支 rebase 冲突超预期 | P0-T1 超 1 人日 | 降级为只 cherry-pick metrics.py + 最小埋点集，not overlapped 直接在新基线重写 |
| R4 特性叠加问题（DCP key 不一致等） | P3-T2 验证失败 | 叠加验证不阻塞主线（M5 验收不依赖叠加结论）；问题单立项走后续 |
| R6 51 服务器排期冲突 | P0-T2 环境锁定失败 | 冒烟/UT 转 165 服务器；正式验收实验必须 51 服务器，提前一周锁排期 |
| 时间线挤压 | 任一里程碑延期 >2 天 | 优先级保 M5（验收）；P4-T3 SSD 预研、P3-T3 LayerSplit 为首先裁剪项 |

---

## 六、验收判定标准

最终判定（Phase 4 出口，对应需求验收标准表全部要素）：

1. **模型**：DeepSeekV4 Flash（结构参数已在 P0-T3 固化）。
2. **负载**：平均 128K 输入、实测命中率与设计 90% 偏差 <2 个百分点、双组各自最优并发。
3. **对比**：DRAM 池化（memcache backend + use_layerwise）vs HBM PrefixCache（多前缀、满前缀驻留 HBM），同 commit、同镜像、同数据集。
4. **指标**：Prefill TPS（max_tokens=1 口径，引擎计数器交叉校验，≥3 轮均值）。
5. **判定**：`TPS_pool / TPS_hbm ≥ 0.95`。
6. **机理佐证**：not overlapsed P90 趋近 0 且 overlap_ratio 高（具体阈值以 Phase 2 收敛后的实测分布定标，写入验收报告）。
7. **配套交付**：极光大盘两指标可读；特性叠加验证记录；nightly 结论；长稳 ≥48h。

---

## 七、执行现状快照（2026-08-26）

- 开发分支 `layerwise_pooling` 已创建并与 upstream/main（ff998aad1）同步；
- 指标分支 `kv_metrics_observability`（7 提交）待整合，P0-T1 为首个动作；
- 51 服务器环境、DSV4 Flash 权重可用性待 P0-T2 核实；
- DSV4 多 cache family × layerwise 的裁决（P0-T4）是本周最关键产出。
