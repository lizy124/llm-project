# LLM Project Notes

这是围绕 LLM、vLLM、KV Cache/KV Pool 和 vLLM Ascend 的个人研究与验证资料库。
目录整体分为**两大部分 + 若干独立区**：技术类文档（第一部分）、项目方案/规划区（第二部分），以及项目执行手册 `Templates/`、非文档资产 `assets/`、环境工具 `utils/`。

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

### 项目执行手册 — `Templates/`

"新 PR / 新专项"从头走到尾的可复用方法论（需求分析 → 方案设计 → 环境搭建 → 池化拉起 → 测试设计 → 结果判定）：

| 目录 | 内容 |
|------|------|
| `Templates/01_requirements` | 需求分析方法论 |
| `Templates/02_design` | 设计提案/实施/测试计划方法论 |
| `Templates/03_env_setup` | 环境搭建指南（建容器/配代理/装包） |
| `Templates/04_pool_setup` | 池化后端**完整可跑方案**（memcache 共置/standalone、mooncake 单机） |
| `Templates/05_vllm_launch` | 池化 vllm 启动（参数/READY/失败速查） |
| `Templates/06_test_method` | 测试设计 + 按场景运行手册 |
| `Templates/07_pass_criteria` | E2E 判定标准、误区清单、已知硬限 |
| `Templates/_common` | IRON RULE、版本配对、端口规划等公共约束 |

> 详细索引见 [Templates/README.md](Templates/README.md)。

### 独立区：非文档资产 — `assets/`

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