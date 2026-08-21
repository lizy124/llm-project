# LLM Project Notes

这是围绕 LLM、vLLM、KV Cache/KV Pool 和 vLLM Ascend 的个人研究与验证资料库。
目录中既有长期知识，也有阶段性分析和实验记录；资料的“状态”由所在目录表达。

## 从哪里开始

- [LLM Knowledge Base](llm_knowledge_base/README.md)：已经整理、可长期复用的知识条目。
- [Research Workspace](research_workspace/README.md)：系统性阅读笔记、架构分析、学习路线和图示。
- [Transfer Data](transfer_data/README.md)：验证记录、脚本、环境清单和 PR/refactor 工作资料。
- [Draft](draft/README.md)：尚未沉淀为正式文档的 review、issue 分析和交接材料。
- [Utils](utils/)：安装、环境和本地工具说明。
- [Basic Skills](basic_skills/)：供自动化工具使用的工作规范。

## 资料生命周期

1. 新问题先放入 `llm_knowledge_base/99_inbox/`，确认主题后再归档到对应分类。
2. 正在研究或需要反复修改的内容放入 `research_workspace/`。
3. 一次性验证、运行记录和环境快照放入 `transfer_data/`；可复用结论应回写知识库或研究笔记。
4. 临时 review 和交接内容放入 `draft/`。完成后转为正式文档，过时内容直接删除或在提交说明中说明原因。

## 命名与格式

- 目录使用小写 `snake_case`，README 统一命名为 `README.md`。
- 稳定文档使用小写 `kebab-case`；按顺序阅读的系列文档使用两位数字前缀（如 `01_...`）。
- 脚本使用动作或序号开头的清晰名称；运行产物（日志、PID、results/runs 目录）不提交。
- Markdown 使用 UTF-8、一个一级标题、相对链接和 LF 换行；不要把机器绝对路径作为唯一入口。

## 维护检查

提交前至少执行：

```text
git diff --check
git status --short
```

本目录是资料库，不是 vLLM/vllm-ascend 源码仓库。文档中的代码路径用于定位上下文，实际源码位于同级的 `../code/`。
