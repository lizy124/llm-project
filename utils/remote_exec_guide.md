# 远程执行指南 — 与服务器交流的引号铁律(agent 必读)

> 适用:本机(PowerShell 5)ssh 到服务器(如 165 = root@192.168.13.165),
> 以及再深入 `docker exec <容器> bash -c` 的场景。
> 配套工具:同目录 `go.ps1`(双层 base64 执行器)。

## 问题根源

命令要穿过 **PowerShell 5 → ssh → 远端 bash → docker exec bash -c** 共 3~4 层
shell,每层都有自己的引号解析规则。任何一层转义错一次,命令就被截断或变形,
且往往**不报错、静默执行错**,极难排查。

实测事故(2026-09-01):`grep -E \"^(Version|...)\"` 的反斜杠转义被 PowerShell
原样传给 ssh,远端 bash 又把 `\"` 解析变形,最终 `Version` 被当成独立命令执行。

## 方法一:base64 编码传输(首选,go.ps1 已内置)

原理:base64 字符集只有 `A-Za-z0-9+/=`,**不含任何引号和特殊字符**,天然对
多层 shell 免疫。

go.ps1 的实现是**双层 base64**:

```
用户命令 --base64--> 内层串 --包进 "echo 内层 | base64 -d | bash" --base64--> 外层串
本机只执行:ssh root@host "echo <外层串> | base64 -d | bash"
```

传输链路上出现的只有 base64 字符,任何一层 shell 都无可解析之物。

### 使用方式(调用即自动编码,命令原样写引号随意用)

```powershell
# 整条远端命令作为一个 PowerShell 字符串传入。
# 推荐外层用 PS 单引号(字面量,防 $ 被 PowerShell 提前展开);
# 命令内部的单引号在 PS 单引号串里写两遍('' 转义)。
.\go.ps1 'docker exec cxy_cann9.1.0 bash -c "pip show vllm | grep -E \"^(Version|Location)\""'
.\go.ps1 'npu-smi info | grep -c 0000:'
.\go.ps1 -Server root@192.168.13.165 'ls /data'
```

- 默认 Server = `root@192.168.13.165`,可用 `-Server` 覆盖
- 命令里的引号、`$()`、管道、awk 全部原样到达远端 bash,无需任何手工转义

## 方法二:脚本化 push 执行(复杂流程必用)

把命令写成 `.sh` 文件 → scp 推到远端 → `ssh "bash /tmp/xxx.sh"` 执行。

- 引号只在脚本文件内出现,**不经过任何传输层**,写什么就是什么
- 适用于:多步骤流程、awk/sed 复杂文本处理、长 prompt 拼接
  (不要硬编码在命令行里)
- 项目惯例:`probe_server.sh` 就是这么跑的;测试脚本统一 push 到远端
- 落盘约定:本地写 `D:\lzy\project\kv_pool\tmp\xxx.sh` →
  scp 到 `/home/lizhongyang/tmp/` → `ssh root@192.168.13.165 "bash /home/lizhongyang/tmp/xxx.sh"`
- 注意:Windows 写出的 .sh 若带 CRLF 行尾,远端 bash 可能报
  `\r: command not found`——脚本内首行加 `sed -i 's/\r$//' $0` 或保存时选 LF

## 禁止事项(硬规矩)

| 禁止 | 原因 |
|---|---|
| ssh 内联命令里用**嵌套引号** | 多层解析必翻车 |
| ssh 内联命令里用 `$( )` 命令替换 | 被 PowerShell 或中间 shell 提前展开 |
| ssh 内联命令里用 **awk 引号** | awk 的 `'{}'` 与外层引号冲突 |
| 手写 `ssh ... "docker exec ... bash -c '...'"` 多层引号 | 用 go.ps1 代替,它已自动 base64 |

## 判断标准

- **单条简单命令**(无引号无 `$()`)→ 可直接 ssh 内联
  (如 `ssh root@192.168.13.165 "docker ps"`)
- **命令里有引号但单层** → 走 go.ps1(自动 base64)
- **多步骤 / 复杂文本处理 / 长 prompt** → 写 .sh push 上去跑
