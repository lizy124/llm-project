04_verification_env_and_launch.md
# 04 — 验证：环境搭建、服务拉起与测试配置

> 归档骨架。复制到 `Proposals/<专项>/<任务>/archive/04_verification_env_and_launch.md` 后按提示填写；执行期间按任务试行规范维护五个主题工作目录。复制后核对相对链接。
> 可引用 `03_verification_scripts.md` 的脚本编号，不重复贴脚本全文。
> 写法参考：[03_env_setup](../../03_env_setup/env_setup_guide.md) / [05_vllm_launch](../../05_vllm_launch/vllm_pool_launch.md)

## 1. 环境

| 项 | 值 |
|---|---|
| 服务器 / 容器 |  |
| 镜像 / 版本 |  |
| 模型 / 权重路径 |  |
| vllm / vllm-ascend 版本 |  |
| 工作目录 | <服务器上路径> |

## 2. 环境搭建

- **代理 / 依赖安装**：<命令，或指向 S1>
- **版本核对**：<命令与期望输出>

## 3. 服务拉起

### 3.1 池后端

| 项 | 值 |
|---|---|
| backend | memcache / mooncake |
| 端口 |  |
| 配置文件 |  |

- **拉起命令**：<或指向 S2>

### 3.2 vLLM

- **关键参数**（含**命中所必需的参数**）：
  ```bash
  vllm serve <...>
  ```
- **READY 判据**：<health 探活 / ready 文件 / status.txt>

## 4. 测试配置与命令

- **负载**：<数据集 / prompt 长度 / 并发 / 输出长度>
- **命中判据（正计数形态）**：<要 grep 的日志形态，如 `hit tokens > 0` / `need to load > 0` / counters 增长>
- **测试命令**：<或指向 S3>

## 5. 前置检查

- **NPU 空闲**：`npu-smi info`
- **端口占用**：`ss -ltnp`
- 指向 `05_verification_process_and_results.md`（过程与结果）