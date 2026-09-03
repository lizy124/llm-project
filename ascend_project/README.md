# Ascend Project

这里存放 Ascend/vLLM-Ascend 相关**项目提案与开发规划**（需求分析、设计、实施计划、测试方案、PR 描述与走读）。定位是"方案/规划"区，只放稳定可评审的提议文档，不放运行日志与验证证据。

## 子目录

- `Layerwise-Pooling-Optimization/`：KV 池化 Layerwise 传输加速专项——需求分析、设计提案、实施/测试计划（`requirements_analysis.md` / `design_proposal.md` / `implementation_plan.md` / `dev_plan.md` / `test_plan.md`）。
- `vLLM-Observability/`：vLLM 推理监控平台对接专项——需求分析、PR 描述/走读/审查验证报告，以及 `scripts_165/` 服务器环境探查与回归脚本。

## 边界说明（与其它分区的关系）

- **本区 vs `transfer_data/`**：本区放**方案/规划**（需求、设计、计划）；`transfer_data/` 放**落地重构的完整记录**（`refactor/layerwise/archive/` 为 layerwise 实际的审核、开发、E2E 证据与脚本）。方案 → 落地，两个区互斥存放：
  - Layerwise 专项的**提案/规划** → 本区 `Layerwise-Pooling-Optimization/`
  - Layerwise **落地重构记录** → `transfer_data/refactor/layerwise/archive/`
  - vLLM 可观测性专项的**落地脚本/回归记录** → `transfer_data/` 相关 refactor 子目录
- **本区 vs `llm_knowledge_base/` / `research_workspace/`**：本区是"立项与计划"，知识库/研究区是沉淀后的"长期知识"；规划结论稳定后按需链接过去，不复制整篇。