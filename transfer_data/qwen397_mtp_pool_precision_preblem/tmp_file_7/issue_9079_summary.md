# vLLM-Ascend Issue 9079 详情汇总

来源 API: https://api.github.com/repos/vllm-project/vllm-ascend/issues/9079
评论 API: https://api.github.com/repos/vllm-project/vllm-ascend/issues/9079/comments?per_page=100
页面链接: https://github.com/vllm-project/vllm-ascend/issues/9079
汇总日期: 2026-07-27

## Issue 元数据

- Issue 编号: #9079
- 标题: [Contribution] vLLM-Ascend 外部开发者任务池
- 状态: open
- 创建者: chenchuw886
- 作者关联: CONTRIBUTOR
- 标签: guide
- 创建时间: 2026-05-12T02:36:09Z
- 更新时间: 2026-07-26T04:17:16Z
- 关闭时间: null
- assignees: 无
- milestone: 无
- 评论数: 65
- sub_issues_summary.total: 0
- sub_issues_summary.completed: 0
- issue_dependencies_summary.blocked_by: 0
- issue_dependencies_summary.blocking: 0

## 正文说明

Issue 正文说明 vLLM-Ascend 社区开放 33 项外部开发者任务，覆盖核心架构重构、多模态与长序列功能适配、主流模型适配、特性验证及 CI 稳定性提升等方向。

参与方式原文要点:

1. 在本 issue 下留言，例如 `认领任务#1`
2. 优先联系任务对应的对接人，解答具体细节问题
3. 认领状态仅保留三类: `待认领`、`已认领`、`已关闭`
4. PR 需关联本 Issue，并尽量提供复现步骤、修复说明和回归验证

更新说明原文要点:

- 每月刷新一次任务池，新增近期任务，移除已完成或不再适合外部认领的任务
- 任务被明确认领后，将状态改为 `已认领`
- 任务已关闭后，将状态改为 `已关闭`
- 无明确认领证据时，状态保持 `待认领`

## 正文任务表状态汇总

说明: 本节只统计 issue 正文任务表中 `认领状态` 字段的原文，不做“已接收”“待验收”等状态映射。

- 正文声明任务总数: 33
- 表格任务行数: 33
- `认领状态` 字段包含 `已认领` 的任务行数: 26
- `认领状态` 字段为空的任务行数: 7
- `认领状态` 字段显示 `待认领` 的任务行数: 0
- `认领状态` 字段显示 `已关闭` 的任务行数: 0
- 状态为空的任务 ID: 8, 9, 10, 13, 15, 16, 17

## 正文任务清单

| 任务ID | Issue 链接 | 任务名称 | 对接人 | 正文认领状态 |
| --- | --- | --- | --- | --- |
| 1 | https://github.com/vllm-project/vllm-ascend/issues/10648 | vllm主社区多模态能力加强，视频场景evs能力压缩 | @jyoung6652 | 已认领 @csw7777 |
| 2 | https://github.com/vllm-project/vllm-ascend/issues/10649 | FlashComm1 环境变量迁移到 AscendConfig | @jyoung6652 | 已认领 @sydbll |
| 3 | https://github.com/vllm-project/vllm-ascend/issues/10650 | MoE通信方式选择代码重构 | @jyoung6652 | 已认领 @Mango03111 |
| 4 | https://github.com/vllm-project/vllm-ascend/issues/10651 | Mooncake KV Connector 调度侧/Worker 侧解耦重构 | @jyoung6652 | 已认领 @jo-pillar |
| 5 | https://github.com/vllm-project/vllm-ascend/issues/10652 | PD 分离非对等 TP 配置简化 | @jyoung6652 | 已认领 @duanzhaol |
| 6 | https://github.com/vllm-project/vllm-ascend/issues/10653 | MooncakeConnector handshake 端口动态分配 | @jyoung6652 | 已认领 @Mango03111 |
| 7 | https://github.com/vllm-project/vllm-ascend/issues/10654 | PD 分离特性叠加场景看护 | @jyoung6652 | 已认领 @KyuuJuu260 |
| 8 | https://github.com/vllm-project/vllm-ascend/issues/10655 | Mooncake 池化场景高可用模式验证 | @jyoung6652 | 空 |
| 9 | https://github.com/vllm-project/vllm-ascend/issues/10656 | Memcache 池化后端高可用模式验证 | @jyoung6652 | 空 |
| 10 | https://github.com/vllm-project/vllm-ascend/issues/10657 | A3 25.5 HDK + CANN 9.0 fabric_mem 池化能力验证 | @jyoung6652 | 空 |
| 11 | https://github.com/vllm-project/vllm-ascend/issues/10658 | Mooncake Store standalone 池化模式验证 | @jyoung6652 | 已认领 @jo-pillar |
| 12 | https://github.com/vllm-project/vllm-ascend/issues/10659 | AscendStore 池化模块 UT 测试补充 | @jyoung6652 | 已认领 @Mango03111 |
| 13 | https://github.com/vllm-project/vllm-ascend/issues/10660 | 长序列场景下pcp-dcp功能支持 | @jyoung6652 | 空 |
| 14 | https://github.com/vllm-project/vllm-ascend/issues/10661 | 投机解码 | @jyoung6652 | 已认领 @Gin-Only |
| 15 | https://github.com/vllm-project/vllm-ascend/issues/10662 | IQuest-Coder-V1-40B 代码大模型适配 | @jyoung6652 | 空 |
| 16 | https://github.com/vllm-project/vllm-ascend/issues/10663 | Gemma4 E2B/E4B轻量级图片、语音、文本多模态大模型适配 | @jyoung6652 | 空 |
| 17 | https://github.com/vllm-project/vllm-ascend/issues/10664 | 310P线上CI稳定性构建 | @jyoung6652 | 空 |
| 18 | https://github.com/vllm-project/vllm-ascend/issues/10665 | triton算子使用策略重构 | @jyoung6652 | 已认领 @treason258 |
| 19 | https://github.com/vllm-project/vllm-ascend/issues/10666 | 310P支持compressed_tensors_config.py | @jyoung6652 | 已认领 @treason258; 已认领 @jo-pillar |
| 20 | https://github.com/vllm-project/vllm-ascend/issues/10667 | 310P DeepSeek-OCR2 多模态OCR模型适配优化 | @jyoung6652 | 已认领 @treason258 |
| 21 | https://github.com/vllm-project/vllm-ascend/issues/10668 | 310P DeepSeek-R1蒸馏系列模型适配 | @jyoung6652 | 已认领 @330800awesome; 已认领 @treason258 |
| 22 | https://github.com/vllm-project/vllm-ascend/issues/10669 | 【社区模型】适配qwq-32b | @jyoung6652 | 已认领 @WinterSun-ysws |
| 23 | https://github.com/vllm-project/vllm-ascend/issues/10670 | 【社区模型】适配baichuan-7b | @jyoung6652 | 已认领 @WinterSun-ysws |
| 24 | https://github.com/vllm-project/vllm-ascend/issues/10671 | 【社区模型】适配DS-distill-LLaMA-70B | @jyoung6652 | 已认领 @WinterSun-ysws |
| 25 | https://github.com/vllm-project/vllm-ascend/issues/10672 | 【社区模型】适配DS-distill-Qwen-32B | @jyoung6652 | 已认领 @mygitljf |
| 26 | https://github.com/vllm-project/vllm-ascend/issues/10673 | 【社区模型】适配internlm2_20b_chat | @jyoung6652 | 已认领 @WinterSun-ysws |
| 27 | https://github.com/vllm-project/vllm-ascend/issues/10674 | 【社区模型】适配Llama-3.2-3B-Instruct | @jyoung6652 | 已认领 @mygitljf |
| 28 | https://github.com/vllm-project/vllm-ascend/issues/10675 | 【社区模型】适配MiniCPM-2B | @jyoung6652 | 已认领 @zhangkx-777 |
| 29 | https://github.com/vllm-project/vllm-ascend/issues/10676 | 【社区模型】适配MiniCPM3-4B | @jyoung6652 | 已认领 @zhangkx-777 |
| 30 | https://github.com/vllm-project/vllm-ascend/issues/10677 | 【社区模型】适配Phi-4-mini | @jyoung6652 | 已认领 @zhangkx-777 |
| 31 | https://github.com/vllm-project/vllm-ascend/issues/10678 | bge-reranker-v2-m3 离线推理功能测试 | @jyoung6652 | 已认领 @gygdh-001 |
| 32 | https://github.com/vllm-project/vllm-ascend/issues/10679 | e5-mistral-7b-instruct 离线推理功能测试 | @jyoung6652 | 已认领 @EheinWang |
| 33 | https://github.com/vllm-project/vllm-ascend/issues/10680 | Qwen2.5-1.5B-apeach 离线推理功能测试 | @jyoung6652 | 已认领 @zhengzhi132 |

## 任务详情摘要

### 架构、通信与池化任务

- 任务 1: vllm 主社区多模态能力加强，视频场景 evs 能力压缩。验收关注 qwen3.5 模型、evs 框架打通、收益、使用说明、用例、RFC 和特性指导。
- 任务 2: FlashComm1 环境变量迁移到 AscendConfig。验收关注 additional-config、环境变量兼容或废弃策略、Worker 子进程 Config 上下文、分布式 e2e、enable_sp、MoE/SP 和 FlashComm 联动。
- 任务 3: MoE 通信方式选择代码重构。验收关注性能不劣化和 developer guide 中的 MoE 通信算子选择策略说明。
- 任务 4: Mooncake KV Connector 调度侧/Worker 侧解耦重构。验收关注对齐主仓 NIXL connector 新架构、Scheduler/Worker 职责拆分、普通 Mooncake 与 layerwise Mooncake 传输能力、UT/e2e 和 lint。
- 任务 5: PD 分离非对等 TP 配置简化。验收关注移除冗余 kv_connector_extra_config、从现有配置推导 DP/TP、非对等 TP 场景、rank 映射、KV block 分发、兼容或迁移提示。
- 任务 6: MooncakeConnector handshake 端口动态分配。验收关注动态端口选择、多 rank 多实例不冲突、P/D 端口同步、KV cache 传输和异常清理。
- 任务 7: PD 分离特性叠加场景看护。验收关注 PD 与 MTP、prefix cache、chunked prefill、PCP、DCP 的组合 UT，合法组合和互斥组合报错。
- 任务 8: Mooncake 池化场景高可用模式验证。正文状态为空；验收关注 etcd、多 Mooncake Master、HA 模式、leader 切换、metadata、P/D 握手和 KV cache 传输。
- 任务 9: Memcache 池化后端高可用模式验证。正文状态为空；验收关注 K8S、Memcache HA 部署、vllm-ascend 对接、读写命中、Pod 重启和服务切换。
- 任务 10: A3 25.5 HDK + CANN 9.0 fabric_mem 池化能力验证。正文状态为空；验收关注 A3 + 25.5 HDK + CANN 9.0 环境、ASCEND_ENABLE_USE_FABRIC_MEM=1、池化读写传输命中、与 26.0 HDK 对比、异常场景和验证报告。
- 任务 11: Mooncake Store standalone 池化模式验证。验收关注 standalone store service 启动、vllm-ascend client 对接、缓存读写命中、数据传输、服务异常和部署验证说明。
- 任务 12: AscendStore 池化模块 UT 测试补充。验收关注 config_data、kv_transfer、pool_scheduler、pool_worker、ascend_store_connector、backend 六层 UT，smart-ut 和覆盖报告。
- 任务 13: 长序列场景下 pcp-dcp 功能支持。正文状态为空；验收关注 pcp_utils.py 可快速适配新 attention 后端，后端之间解耦。
- 任务 14: 投机解码。验收关注 vllm 与 vllm-ascend 关于投机解码代码基本统一。

### 310P、CI 与模型适配任务

- 任务 15: IQuest-Coder-V1-40B 代码大模型适配。正文状态为空；验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 16: Gemma4 E2B/E4B 轻量级图片、语音、文本多模态大模型适配。正文状态为空；验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 17: 310P 线上 CI 稳定性构建。正文状态为空；验收关注 CI 看护范围刷新、线上构建不同组件解耦、main2main 保证 310P 无异常。
- 任务 18: triton 算子使用策略重构。验收关注 310P 推理链路与 Triton 算子开关完全解耦。
- 任务 19: 310P 支持 compressed_tensors_config.py。验收关注 LLM-Compressor 压缩的 qwen3-32B-W8A8 模型，精度无问题。
- 任务 20: 310P DeepSeek-OCR2 多模态 OCR 模型适配优化。验收关注 e2e 模型配置、模型支持清单、OCR 适配部署教程和 SKILL.md。
- 任务 21: 310P DeepSeek-R1 蒸馏系列模型适配。验收关注 e2e 模型配置、模型支持清单、模型部署教程和 SKILL.md。
- 任务 22: 适配 qwq-32b。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 23: 适配 baichuan-7b。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 24: 适配 DS-distill-LLaMA-70B。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 25: 适配 DS-distill-Qwen-32B。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 26: 适配 internlm2_20b_chat。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 27: 适配 Llama-3.2-3B-Instruct。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 28: 适配 MiniCPM-2B。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 29: 适配 MiniCPM3-4B。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 30: 适配 Phi-4-mini。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 31: bge-reranker-v2-m3 离线推理功能测试。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 32: e5-mistral-7b-instruct 离线推理功能测试。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。
- 任务 33: Qwen2.5-1.5B-apeach 离线推理功能测试。验收关注 e2e 模型配置、模型适配教程文档和 SKILL.md。

## 评论时间线汇总

说明: 本节按评论 API 返回的 65 条评论顺序汇总，记录认领、确认、问题反馈、PR/SKILL 提交等信息。这里不覆盖正文表格状态，只作为评论侧证据。

1. 2026-05-12 @nayihz: 提到 issue #2649 是必现问题，PR #9046 尝试修复，请求 review。
2. 2026-06-08 @zhengzhi132: 申请认领任务 #33，并询问如何联系对接人。
3. 2026-06-09 @jyoung6652: 回复 @zhengzhi132，任务 33 可基于仓库已有模型适配，提交 PR 或留言提问。
4. 2026-06-12 @Gin-Only: 申请认领任务 #14。
5. 2026-06-12 @zhengzhi132: 提交 Task #33 的 SKILL.md 内容，关联 PR #10409，总结 Qwen2.5-1.5B-apeach 适配策略、交付文件和最佳实践。
6. 2026-06-17 @Gin-Only: 再次申请任务 #14，并询问是否对比 vllm-ascend 单独实现的投机解码和 vllm 原生实现并查漏补缺。
7. 2026-06-18 @mygitljf: 申请认领任务 #27，适配 Llama-3.2-3B-Instruct。
8. 2026-06-18 @mygitljf: 申请认领任务 #25，适配 DS-distill-Qwen-32B。
9. 2026-06-18 @jyoung6652: 回复 @Gin-Only，确认方向，但要求检查最新 spec decode 代码。
10. 2026-06-18 @jyoung6652: 回复 @mygitljf，可参考模型文档模板进行任务适配。
11. 2026-06-22 @duanzhaol: 申请认领任务 #5。
12. 2026-06-23 @330800awesome: 申请认领 issue #10668，即任务 21。
13. 2026-06-25 @WinterSun-ysws: 申请认领任务 #22、#23、#24、#26。
14. 2026-06-25 @zhangkx-777: 申请认领任务 #28、#29、#30。
15. 2026-06-25 @KyuuJuu260: 申请认领“任务 7”，同时贴出 issue #10659 链接。
16. 2026-06-25 @jyoung6652: 回复 @duanzhaol，确认收到。
17. 2026-06-25 @jyoung6652: 回复 @330800awesome，确认 OK。
18. 2026-06-25 @jyoung6652: 回复 @WinterSun-ysws，确认 OK。
19. 2026-06-25 @jyoung6652: 回复 @zhangkx-777，确认 OK。
20. 2026-06-25 @jyoung6652: 回复 @KyuuJuu260，确认 OK。
21. 2026-06-26 @gygdh-001: 申请认领任务 #31。
22. 2026-06-26 @treason258: 申请认领 issue #10663、#10662、#10665、#10669。
23. 2026-06-26 @EheinWang: 申请认领任务 #32。
24. 2026-06-26 @jo-pillar: 申请认领 issue #10649、#10662、#10678、#10679、#10680。
25. 2026-06-26 @jo-pillar: 申请认领 issue #10652。
26. 2026-06-26 @Mango03111: 申请认领任务 #3/#10650、#6/#10653、#12/#10659。
27. 2026-06-27 @sydbll: 申请认领任务 #2。
28. 2026-06-27 @xyfsl123: 申请认领任务 #3、#6、#12、#13。
29. 2026-06-27 @sankalok: 申请认领 Task #31 / issue #10678。
30. 2026-06-27 @csw7777: 申请认领任务 #1 / issue #10648。
31. 2026-06-28 @xiaohongshu528: 申请认领 issue #10654。
32. 2026-06-28 @xiaohongshu528: 申请认领 issue #10655。
33. 2026-06-29 @zhangkx-777: 反馈任务表 issue 链接与任务名称列没有对齐，并询问 Phi-4-mini 和 MiniCPM-2B 的具体模型选择。
34. 2026-06-29 @jyoung6652: 回复 @gygdh-001，确认收到任务 #31。
35. 2026-06-29 @jyoung6652: 回复 @treason258，可做 no.15、no.16、no.18，no.22 已被认领。
36. 2026-06-29 @jyoung6652: 回复 @sydbll，确认收到。
37. 2026-06-29 @jyoung6652: 回复 @jo-pillar，可获得 no.2 任务，其他任务已被认领。
38. 2026-06-29 @jyoung6652: 回复 @Mango03111，可开始 no.3、no.6、no.12。
39. 2026-06-29 @jyoung6652: 回复 @sydbll，no.2 已被认领。
40. 2026-06-29 @jyoung6652: 回复 @xyfsl123，获得 no.13，其他任务已被认领。
41. 2026-06-29 @jyoung6652: 回复 @sankalok，no.31 已被认领，后续会推出更多任务。
42. 2026-06-29 @jyoung6652: 回复 @csw7777，确认收到任务 #1。
43. 2026-06-29 @jyoung6652: 回复 @xiaohongshu528，no.7 已被认领，可开始 no.8。
44. 2026-06-29 @jyoung6652: 回复 @zhangkx-777，确认 Phi-4-mini-instruct 和 MiniCPM-2B-dpo-bf16 可以。
45. 2026-06-29 @zhangkx-777: 申请认领任务 #15、#16。
46. 2026-06-29 @duanzl1999-sys: 申请认领任务 #13。
47. 2026-06-29 @ccsuzzh: 申请认领任务 17。
48. 2026-06-30 @sydbll: 申请认领任务 #8、#10。
49. 2026-06-30 @WinterSun-ysws: 反馈任务清单存在 issue 链接匹配问题，称所领取的 #22、#23、#24、#26 均存在该问题，并附图。
50. 2026-06-30 @xyfsl123: 反馈 #13 也存在 issue 链接错位问题。
51. 2026-06-30 @330800awesome: 提到暂时可以通过 issue id - 1 来链接任务。
52. 2026-06-30 @chenchuw886: 回复 Issue 链接问题已修复。
53. 2026-06-30 @jo-pillar: 反馈 no.2 task 被误登记给别人，并附图。
54. 2026-06-30 @xiaohongshu528: 说明之前认领 NO.7/NO.8，即 #10654/#10655；若 NO.7 已被认领，则保留 NO.8 / #10655。
55. 2026-06-30 @xiaohongshu528: 再次确认将开始 no.8。
56. 2026-06-30 @xyfsl123: 反馈 #13 已认领，但正文认领状态未修改。
57. 2026-06-30 @sanyu6: 申请认领 #10663、#10662。
58. 2026-07-02 @gygdh-001: 请求检查任务 #31 的结果，关联 PR #11275。
59. 2026-07-02 @kenpaul877: 申请认领任务 #8、#9，即 issue #10655、#10656，并说明有集群环境可验证。
60. 2026-07-05 @TreamTik: 申请认领任务 #15、#16，即 issue #10662、#10663。
61. 2026-07-05 @Agoni-02: 申请认领任务 #17，即 issue #10664。
62. 2026-07-07 @WanhaoZhang: 说明有验证环境，申请认领任务 #8 / issue #10655。
63. 2026-07-10 @Kurumi5210: 说明有验证环境，申请认领任务 13 / issue #10660。
64. 2026-07-10 @Kurumi5210: 说明有验证环境，申请认领任务 15 / issue #10662 和任务 16 / issue #10663。
65. 2026-07-26 @ZhaXionghui: 申请认领任务 #10663。

## 评论侧可见的补充信息

- 正文任务表中状态为空的任务 8、9、10、13、15、16、17，在评论中均能看到后续认领或申请认领痕迹。
- 评论中存在多处“任务编号”和“issue 链接”错位反馈，维护者 @chenchuw886 于 2026-06-30 回复“Issue链接问题已修复”。
- 评论中存在多处重复申请或申请后被告知已被认领的情况，例如任务 2、3、6、12、13、15、16、22、31 等。
- 评论中出现结果提交或待检查信息: 任务 #33 关联 PR #10409 和 SKILL.md；任务 #31 关联 PR #11275。
- 正文表格状态与评论侧认领痕迹并不完全一致，后续统计时应明确采用“正文表格状态”还是“评论证据状态”。
