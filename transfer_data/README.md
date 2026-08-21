# Transfer Data

这里保存 KV Pool/vLLM Ascend 的验证、重构和运行环境资料。它是工作资料区，不是长期知识库。

## 子目录

- `envs/`：环境变量清单及生成脚本。
- `refactor/backend/`：backend 重构验证脚本和报告。
- `refactor/kvpool/`：KV Pool 重构、PR 验证和模型场景脚本。
- `refactor/weight_transfer/`：weight transfer 设计、时间线和验证记录。
- `kv_pool_issue/`：KV Pool 专题设计与分析。
- `workspace/`：仍在整理的 E2E、问题清单和概念材料。
- `daily_notes/`：按日期记录的短期工作笔记。

## 保留规则

- 可复现步骤、配置和结论保留；运行生成的日志、PID、`results/` 和 `runs/` 目录不提交。
- 结论稳定后，链接到研究笔记或知识库，并在这里保留必要的验证证据。
- `workspace/` 只用于工作区命名，不再使用旧的 `work_space/` 或 `back_up/` 拼写。
