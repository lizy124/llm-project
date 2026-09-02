# Transfer Data

这里保存 KV Pool/vLLM Ascend 的验证、重构和运行环境资料。它是工作资料区，不是长期知识库。

## 子目录

- `playbook/`：通用方法手册（agent 必读）——`create_env.md` 环境搭建指南 + `verify_guide.md` 池化 E2E 验证指南（判据设计/虚假通过防范/双轮夹逼等）+ `run_dir/` 场景运行手册（按场景一篇：环境变量/拉起/请求/判定，与具体 PR 解耦）。
- `refactor/backend/`：backend 重构验证脚本和报告。
- `refactor/kvpool/`：KV Pool 重构、PR 验证和模型场景脚本。
- `refactor/weight_transfer/`：weight transfer 设计、时间线和验证记录。
- `kv_pool_issue/`：KV Pool 专题设计与分析。
- `workspace/`：仍在整理的 E2E、问题清单和概念材料。

## 保留规则

- 可复现步骤、配置和结论保留；运行生成的日志、PID、`results/` 和 `runs/` 目录不提交。
- 结论稳定后，链接到研究笔记或知识库，并在这里保留必要的验证证据。
- `workspace/` 只用于工作区命名，不再使用旧的 `work_space/` 或 `back_up/` 拼写。
- 短期笔记不单独建目录；有结论的内容直接归档到 `workspace/`、`draft/` 或研究文档。
