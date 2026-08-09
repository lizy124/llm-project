# Codex 本地安装记录

本文记录本机 Windows 环境下按 `llm-project/utils/tmp.md` 安装 Codex CLI 的实际过程。

## 环境

- 系统：Windows 11
- 工作目录：`D:\lzy\project\kv_pool`
- Node.js：`v24.15.0`
- npm：`11.12.1`
- Codex CLI：`0.147.0`

## 1. 检查 Node.js 和 npm

先确认 Node.js 与 npm 已安装：

```bash
node --version
npm --version
```

本机输出：

```text
v24.15.0
11.12.1
```

Codex 文档要求 Node.js 18 或更高版本，本机已满足。

## 2. 检查 Codex 是否已安装

```bash
codex -V
```

最初本机输出：

```text
codex: command not found
```

说明当时尚未安装 Codex CLI。

## 3. 安装 Codex CLI

使用 npm 全局安装官方 Codex CLI：

```bash
npm install -g @openai/codex --registry=https://registry.npmjs.org/ --loglevel=info
```

安装包版本：

```text
@openai/codex 0.147.0
```

本机安装时 Windows x64 包下载耗时较长，最终安装成功。

## 4. 创建 Codex 配置目录

Codex 配置目录位于：

```text
C:\Users\lzysh\.codex
```

如需手动创建，可在 PowerShell 中运行：

```powershell
New-Item -Path "$env:USERPROFILE\.codex" -ItemType Directory -Force
```

## 5. 写入认证配置

文件路径：

```text
C:\Users\lzysh\.codex\auth.json
```

内容格式如下，实际使用时将 `<YOUR_UNIVIBE_API_KEY>` 替换为 UniVibe API Key：

```json
{
  "OPENAI_API_KEY": "<YOUR_UNIVIBE_API_KEY>"
}
```

注意：不要把真实 API Key 提交到仓库或公开文档中。

## 6. 写入 Codex 配置

文件路径：

```text
C:\Users\lzysh\.codex\config.toml
```

内容：

```toml
model_provider = "univibe"
model = "gpt-5.6-sol"
model_reasoning_effort = "high"
disable_response_storage = true
preferred_auth_method = "apikey"

[model_providers.univibe]
name = "univibe"
base_url = "https://api.univibe.cc/openai"
wire_api = "responses"
```

说明：用户提供的 `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` 是 Anthropic 接口环境变量；本次按 Codex 文档配置的是 OpenAI 兼容接口，因此 Codex 使用 `OPENAI_API_KEY` 和 `https://api.univibe.cc/openai`。

## 7. 验证安装

检查 Codex 版本：

```bash
codex -V
```

本机输出：

```text
codex-cli 0.147.0
```

查看帮助：

```bash
codex --help
```

确认命令已可用。

## 8. 运行 doctor 检查

```bash
codex doctor
```

本机结果：大部分检查通过，但连通性检查返回：

```text
reachability provider base URL route returned 404 - verify the configured API prefix
```

这个 404 来自 Codex doctor 对 provider base URL 的探测路由。实际模型调用测试是成功的，因此当前判断为第三方 OpenAI 兼容网关没有实现 doctor 探测的那条路由，不影响正常使用。

## 9. 实际调用测试

由于当前目录 `D:\lzy\project\kv_pool` 不是 git 仓库，直接运行 `codex exec` 会提示：

```text
Not inside a trusted directory and --skip-git-repo-check was not specified.
```

需要加上 `--skip-git-repo-check`：

```bash
codex exec --skip-git-repo-check "Reply with exactly: codex-ok"
```

本机成功返回：

```text
codex-ok
```

说明 Codex CLI 已安装成功，且 UniVibe 配置可实际调用模型。

## 10. 日常使用方式

进入交互模式：

```bash
codex
```

一次性执行任务：

```bash
codex exec --skip-git-repo-check "解释一下这个项目结构"
```

在 git 仓库内使用时，通常不需要 `--skip-git-repo-check`。如果当前目录不是 git 仓库，并且只是临时使用 Codex，可以继续加该参数。

## 11. 后续更新

更新 Codex CLI：

```bash
npm install -g @openai/codex
```

或使用 Codex 自带更新命令：

```bash
codex update
```
