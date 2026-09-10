# AGENTS.md — llm-project 工作规则（agent 必读）

> 本文件位于仓库根目录，供 AI agent 与本仓库协作时**自动读取并必须遵守**。
> 规则将逐步补充（编号递增），技术实施约束见 [Templates/_common/common_constraints.md](./Templates/_common/common_constraints.md)。

---

## 0. 机器/安全硬约束（最高优先级，无例外）

（待补充：本机安全与系统级约束，如禁止自动重启 / 自动更新等）

## 1. 仓库结构规则

### 规则 #1：内容必须有结构层次，不可随意放置单独文件

`llm-project/` 的一切内容（文档、脚本、数据）都必须放在有明确语义的目录层级中，**禁止在仓库根目录或任意目录下随意放置游离的单独文件**。新增内容前先确认归属分区，找不到合适位置时先与用户对齐，而不是随手新建文件/目录。

## 2. 仓库导航

（待补充：目录分组图与各分区用途，供 agent 先看再动手）

## 3. 仓库操作纪律

### 3.1 提交（commit）

（待补充）

### 3.2 分支与推送

（待补充）

### 3.3 目录维护

（待补充：归档、git mv、不重复存放、索引同步等）

## 4. agent 工作规则

### 4.1 远程命令传输（多层 shell 引号规避）

**根因**：命令穿越 PowerShell → ssh → 远程 bash → docker exec 每层重新解析引号，嵌套转义层层叠加、错误静默难排查。

**两种合规传输**：

| 方案 | 适用 | 用法 |
|------|------|------|
| **base64（go.ps1 双重 base64）** | 带引号命令（简单） | `powershell -File _config/go.ps1 map_XX '<命令>'` |
| **.sh 脚本推送** | 复杂/多步流程（强制） | 本地写 .sh → scp → `ssh "bash /tmp/xxx.sh"` |

**禁用**（ssh 内联）：嵌套引号、`$( )` 命令替换、awk 引号（`'{}'` 与外层冲突）、手写多层 `ssh "docker exec ... bash -c '...'"`（改用 go.ps1）。

**决策**：无引号简单命令→直接 ssh；单层引号→go.ps1；多步/含 awk-sed/长 prompt→.sh 脚本推送（**长 prompt 绝不硬编码在命令行**）。

**配套**：PowerShell 解析问题（如 `|` 管道）也可用 base64 规避（实测）。

### 4.2 其他

（待补充：先读后改、只改该改的、发问、验证等）

## 5. 技术纪律索引

池化验证相关的技术约束（版本配对表、端口规划、共享服务器纪律、IRON RULE、经验回写规则）统一放在 [Templates/_common/common_constraints.md](./Templates/_common/common_constraints.md)。