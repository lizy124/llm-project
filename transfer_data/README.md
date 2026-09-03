# Transfer Data

这里保存 KV Pool/vLLM Ascend 的验证、重构和运行环境资料。它是工作资料区，不是长期知识库。

## 子目录

- `playbook/`：通用方法手册（agent 必读）——`create_env.md` 环境搭建指南 + `verify_guide.md` 池化 E2E 验证指南（判据设计/虚假通过防范/双轮夹逼等）+ `run_dir/` 场景运行手册（按场景一篇：环境变量/拉起/请求/判定，与具体 PR 解耦）。
- `refactor/backend/`：backend 重构验证脚本和报告（含 PR #13354 单测/安装脚本）。
- `refactor/kvpool/`：KV Pool 重构、PR 验证和模型场景脚本（含 PR #13160 多模型场景、#14465 合并评审）。
- `refactor/layerwise/`：layerwise 传输路径通用化重构，part1（PR #15367）已合入 upstream，全部材料归档在 `archive/`（需求终稿与多轮审核、PR 记录、E2E 证据与复测脚本；part2 后续工作以此为基线）。
- `refactor/hash_key/`：hash key 重构状态与方案。
- `refactor/weight_transfer/`：weight transfer 设计、时间线和验证记录（含 PR #13049）。
- `kv_pool_issue/`：KV Pool 专题设计与分析。
- `workspace/`：仍在整理的 E2E、问题清单和概念材料。

> 边界说明：`refactor/layerwise/` 承载 layerwise **落地重构的完整记录**（审核、开发、E2E 证据）；项目初期的**提案/规划**（需求分析、设计、计划和测试方案）在 `ascend_project/Layerwise-Pooling-Optimization/`。二者一个是"方案"，一个是"落地记录"，互斥存放、不相往来。

## 保留规则

- 可复现步骤、配置和结论保留；运行生成的日志、PID、`results/` 和 `runs/` 目录不提交。
- 结论稳定后，链接到研究笔记或知识库，并在这里保留必要的验证证据。
- `workspace/` 只用于工作区命名，不再使用旧的 `work_space/` 或 `back_up/` 拼写。
- 短期笔记不单独建目录；有结论的内容直接归档到 `workspace/`、`draft/` 或研究文档。
