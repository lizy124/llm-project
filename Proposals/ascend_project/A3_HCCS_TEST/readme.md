# A3 HCCS 双机池化验证（总索引）

一体机规划需求：FM16 双机 A3 560T HCCS 背靠背互联，PD 分离 + KV Cache 池化，DS V4 Flash 模型，200K 上下文，PrefixCache 命中率 80% DRAM + 20% SSD，Prefill 吞吐提升 5\~8 倍。

负责人：lizy124 | 交付：主干分支 RP3 版本

## 机型拆分（2026-09-08）

按 HBM 容量 SKU 拆分为两个独立验证流，**128G 优先执行**：

| 目录             | 机型                               | 状态       | 说明                                    |
| -------------- | -------------------------------- | -------- | ------------------------------------- |
| [128G/](128G/) | A3 560T 128G（每卡 128GB，整机 1024GB） | **优先执行** | 当前手上有 128G 机器；135 已确认可用作 Phase 1 单机基线 |
| [64G/](64G/)   | A3 560T 64G                      | 待启动      | 机型口径待确认 + 机器未到位                       |

## 机型口径速查（避免混淆）

A3 是一卡双 die 封装（npu-smi 可见 16 chip），两种常见计数口径：

| 口径                    | 128G SKU           | 说明                   |
| --------------------- | ------------------ | -------------------- |
| 按卡（华为官方/需求命名）         | 8 × 128GB = 1024GB | "A3 560T 128G" 的命名口径 |
| 按 die（vllm-ascend 文档） | 16 × 64GB = 1024GB | 同一台机器的另一种数法          |

注意：vllm-ascend 文档里的 "64GB × 16" 指的就是 1024GB 整机，不是 64G SKU。

## 公共技术栈

| 项    | 值                                                                         |
| ---- | ------------------------------------------------------------------------- |
| 互联   | HCCS（背靠背双机直连，A3 官方推荐分支 ASCEND\_ENABLE\_USE\_FABRIC\_MEM=1 + device\_sdma） |
| 池化后端 | mooncake + memcache（两个均需验证）                                               |
| 特性叠加 | pp + SSD + layerwise（layerwise 对 DSV4 为已知硬限豁免）                            |
| 模型   | DeepSeek V4 Flash（135: /data/combinded\_nfs/DeepSeek-V4-Flash-w8a8-mtp）   |
| 验收数据 | 200K 上下文，128K 输入 / 1K 输出，多前缀，前缀重复率 90%                                    |

## 当前进展（2026-09-09：Phase 1 单机基线 ✅ 完成）

**Phase 1（135 单机，131070-token 前缀口径）三形态全部验证通过**，性能数据齐备：

| 后端形态 | 冷灌 TTFT（完全重算） | 命中 TTFT | 命中率 | 加速比 | 32 并发聚合 TPS |
| -------------------- | ----------------- | ------- | ----- | ---- | ------------- |
| memcache co-located（device_sdma） | 19.95s | 1.70s | 96.88% | ~11.7x | 132.6K |
| memcache standalone | 19.88s | 1.72s | 96.88% | ~11.6x | 135.4K |
| mooncake | 19.47s（warm0） | 1.78s | 96.88% | ~10.9x | 126.2K |

> 存/取/去重三维证据链 + HBM PrefixCache 关闭（计数器恒 0）均留证；mooncake A 首灌 137.4s 为 store 惰性初始化一次性成本。
> 详细报告：test_pool/01（co-located）、01b（standalone）、01c（mooncake）；进度与证据索引见 [128G/stage_summary.md](128G/stage_summary.md)，结果表见 [128G/verification_plan.md](128G/verification_plan.md) §5.1。
> **Phase 1 Round 1 成套报告**：[128G/phase1_round1_report.md](128G/phase1_round1_report.md) + evidence_round1/（12 份原始证据）。
> **Phase 1 Round 2 复验 PASS**：[128G/phase1_round2_report.md](128G/phase1_round2_report.md) + evidence_round2/（三形态清池重启重跑 14 份证据；命中率 96.88% 六次一致、KV 条目 9455 复现、TTFT 偏差 <1.5%）—— **Phase 1 关闭**。
> 下一阻塞：Phase 2 双机 HCCS（等第二台 128G 机器）；200K 口径数据属 Phase 4 待采集。

## 引用知识库

- D:\project\agent_project\pool_verify\ — 池化验证知识库（mooncake/memcache 拉起、测试方法、判定标准、误区、已知限制）

- D:\project\agent_project\env_install\ — 环境安装知识库

- D:\project\agent_project\AGENTS.md — 远程服务器操作协议