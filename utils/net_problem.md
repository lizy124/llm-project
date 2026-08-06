# 服务器 GitHub 网络问题处理流程

## 背景

使用 Win10 笔记本通过 VS Code Remote SSH 连接远程服务器，Claude Code 也运行在远程服务器上。笔记本本地访问 GitHub、`git clone` 正常，但服务器直接访问 GitHub 不稳定，例如：

- `github.com:443` 连接超时
- `git fetch` / `git ls-remote` 超时
- `api.github.com` 偶尔可通，但 `github.com` Web/Git HTTPS 不稳定

原因是：VS Code 远程连接只负责登录服务器，服务器访问 GitHub 时默认走服务器自己的网络出口，不会自动走 Win10 笔记本的网络或 Clash 代理。

最终方案：使用 SSH 反向隧道，把 Win10 本机 Clash 代理端口映射到服务器本机 `127.0.0.1:7897`，然后让服务器 Git 永久使用这个代理。

## 一、Win10 笔记本操作

### 1. 确认 Clash/Mihomo 代理端口

当前环境使用：

```text
Clash/Mihomo mixed port: 7897
系统代理地址: 127.0.0.1:7897
```

### 2. 开启 Clash 局域网连接

在 Clash 设置里确认：

```text
局域网连接: 开启
端口设置: 7897
```

如果有这些选项，也建议设置为：

```text
Allow LAN: enabled
Bind Address: 0.0.0.0
Mixed Port: 7897
```

### 3. 放行 Windows 防火墙端口

用管理员 PowerShell 执行：

```powershell
New-NetFirewallRule -DisplayName "Allow Clash Mihomo 7897" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 7897
```

### 4. 确认 Clash 正在监听所有地址

PowerShell 执行：

```powershell
netstat -ano | findstr :7897
```

期望看到：

```text
TCP    0.0.0.0:7897      0.0.0.0:0      LISTENING
TCP    [::]:7897         [::]:0         LISTENING
```

只要有 `0.0.0.0:7897 LISTENING`，说明 Clash 已允许非本机访问。

### 5. 建立 SSH 反向隧道

在 Win10 PowerShell 新开一个窗口，执行：

```powershell
ssh -N -R 127.0.0.1:7897:127.0.0.1:7897 root@192.168.13.165
```

说明：

- 第一个 `127.0.0.1:7897`：服务器上的监听地址和端口
- 第二个 `127.0.0.1:7897`：Win10 本机 Clash 代理地址和端口
- `root@192.168.13.165`：远程服务器 SSH 登录地址

这条命令会一直占用当前 PowerShell 窗口，正常情况下不会输出内容。不要关闭这个窗口。

关闭该窗口后，服务器上的 `127.0.0.1:7897` 代理也会失效。

## 二、远程服务器操作

### 1. 验证反向隧道端口是否可用

在服务器执行：

```bash
python3 - <<'PY'
import socket, time
try:
    t = time.time()
    s = socket.create_connection(('127.0.0.1', 7897), timeout=5)
    dt = (time.time() - t) * 1000
    s.close()
    print(f'127.0.0.1:7897 TCP OK {dt:.0f}ms')
except Exception as e:
    print(f'127.0.0.1:7897 TCP FAIL {type(e).__name__}: {e}')
PY
```

期望输出类似：

```text
127.0.0.1:7897 TCP OK 2ms
```

### 2. 验证 curl 通过代理访问 GitHub

```bash
curl -x http://127.0.0.1:7897 -I -L --connect-timeout 8 --max-time 30 https://github.com/
```

期望看到：

```text
HTTP/1.1 200 Connection established
HTTP/2 200
```

### 3. 验证 Git 通过代理访问 GitHub

```bash
git -c http.proxy=http://127.0.0.1:7897 \
    -c https.proxy=http://127.0.0.1:7897 \
    ls-remote https://github.com/vllm-project/vllm-ascend refs/heads/main
```

期望看到类似：

```text
65e16558933a9dfad1a7645a124f91252168ca44    refs/heads/main
```

### 4. 永久设置 Git 全局代理

验证成功后执行：

```bash
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

确认配置：

```bash
git config --global --get-regexp 'http.*proxy|https.*proxy'
```

期望输出：

```text
http.proxy http://127.0.0.1:7897
https.proxy http://127.0.0.1:7897
```

当前服务器已完成该永久设置。

## 三、日常使用方式

每次需要服务器访问 GitHub 前：

1. Win10 打开 Clash，并确认当前节点可用
2. Win10 保持 Clash 端口为 `7897`
3. Win10 PowerShell 执行并保持窗口不关闭：

```powershell
ssh -N -R 127.0.0.1:7897:127.0.0.1:7897 root@192.168.13.165
```

4. 服务器上正常使用 Git：

```bash
git fetch
git pull
git clone https://github.com/owner/repo.git
git ls-remote https://github.com/vllm-project/vllm-ascend refs/heads/main
```

因为服务器 Git 已永久配置代理，所以不需要每次手动加 `-c http.proxy=...`。

## 四、常见问题

### 1. Git 报错连接 `127.0.0.1:7897` 失败

通常是 Win10 的 SSH 反向隧道窗口没开，或已经断开。

重新在 Win10 PowerShell 执行：

```powershell
ssh -N -R 127.0.0.1:7897:127.0.0.1:7897 root@192.168.13.165
```

然后在服务器验证：

```bash
curl -x http://127.0.0.1:7897 -I https://github.com/
```

### 2. PowerShell 隧道命令卡住没有输出

这是正常现象。`ssh -N -R ...` 的作用就是保持隧道，不执行远程命令。

只要窗口不退出，隧道就在。

### 3. Clash 端口不是 7897

如果 Clash 端口变化，例如改成 `7890`，则所有命令里的 `7897` 都要同步替换成新端口：

```powershell
ssh -N -R 127.0.0.1:7890:127.0.0.1:7890 root@192.168.13.165
```

服务器 Git 也要重新设置：

```bash
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

### 4. 想取消 Git 永久代理

如果不再使用 SSH 反向隧道，可以在服务器执行：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### 5. 为什么不直接让服务器连 Win10 的局域网 IP

已测试服务器连接以下地址均超时：

```text
100.125.59.169:7897
192.168.207.68:7897
```

说明服务器到 Win10 的这些地址之间存在路由隔离或防火墙路径问题。

SSH 反向隧道由 Win10 主动连接服务器，不依赖服务器能直接访问 Win10，因此更稳定。

## 五、本次验证结果

服务器通过 SSH 反向隧道访问 GitHub 成功：

```text
127.0.0.1:7897 TCP OK 2ms
HTTP/1.1 200 Connection established
HTTP/2 200
```

Git 验证成功：

```text
65e16558933a9dfad1a7645a124f91252168ca44    refs/heads/main
```
