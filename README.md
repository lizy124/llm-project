# LLM Project Notes

这是围绕 LLM、vLLM、KV Cache/KV Pool 和 vLLM Ascend 的个人研究与验证资料库。
目录整体分为**两大部分 + 两个独立区**：技术类文档（第一部分）、项目方案/规划区（第二部分），以及独立的 `assets`（非文档资产）与 `utils`（环境工具）。

## 目录分组图

### 第一部分：技术类文档 — `Technical/`

| 目录 | 内容 |
|------|------|
| `Technical/llm_knowledge_base` | 长期知识沉淀 |
| `Technical/research_workspace` | 研究问答 + 源码结构分析 |
| `Technical/transfer_data` | 技术记录与证据文档（含少量方案边角，见下） |
| `Technical/draft` | 阶段性技术草稿（根因分析、验证报告等） |
| `Technical/skills` | 技能定义（原 basic_skills） |

### 第二部分：项目方案/规划区 — `Proposals/`

| 目录 | 内容 |
|------|------|
| `Proposals/ascend_project` | 各专项的需求分析、设计提案、实施/测试计划、PR 描述与走读 |

> `Technical/transfer_data` 内仍保留少量方案类文档（requirements/design/plan/review），未做二次拆分；如需阅读方案类材料，可同时参考两区。

### 第三区：非文档资产 — `assets/`

脚本、测试数据、E2E 证据、图片、diff 等所有非 md 文件，按来源顶层目录镜像存放：

| 目录 | 来源 |
|------|------|
| `assets/ascend_project` | ascend_project 下的脚本/数据/基线（baseline、scripts_165、bgq base、pr15602.diff 等） |
| `assets/research` | research_workspace 的 figure（svg 图） |
| `assets/transfer_data` | transfer_data 下的测试脚本、E2E evidence、pr_13160 运行脚本、weight_transfer 验证产物 |
| `assets/draft` | draft 下的 diff 与测试脚本 |

### 独立区：环境工具 — `utils/`

通用工作环境备忘与脚本（clash 代理、docker、codex/go 安装、远程执行、Signed-off-by、net/proxy 问题），与具体项目无关，不归入上述两区。

## 边界与归档

- 稳定后的草稿应归档到 `Technical/llm_knowledge_base` 或 `Technical/research_workspace`，避免同一结论长期存放在多个目录。
- 方案 → 落地互斥：方案/规划放 `Proposals/`，落地记录与技术证据放 `Technical/`（+ `assets/`）。
- 所有非 md 资产统一放 `assets/`，文档内引用 assets 使用相对路径 `../assets/...`。