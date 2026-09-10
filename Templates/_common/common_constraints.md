# 公共约束（跨专项复用）

> 本章汇总项目执行手册的公共约束：IRON RULE、版本配对表、端口规划、共享服务器纪律。各章引用本章，不重复存放。

---

## 1. IRON RULE（最高优先级，无例外）

1. **任何材料不得上传公网**（云存储/网盘/公共服务一律禁止）
2. **任何情况下禁止执行 `git push`**（内网 remote 也不行；`git pull`/`git fetch` 允许）
3. 允许：与内网服务器（51/112/135/165 等）的 scp/rsync 等非 git 传输

## 2. vllm ↔ vllm-ascend 版本配对表

vllm 与 vllm-ascend **强耦合**，配错版本 = 启动即崩。目标版本由验证任务决定，**vllm 版本必须先于 vllm-ascend 确定**。

| vllm-ascend 分支 | 配对文件 | 用途 |
|---|---|---|
| `.github/vllm-release-tag.commit` | 如 `v0.27.1` | release 配对（**常用**） |
| `.github/vllm-main-verified.commit` | commit hash | 跟 vllm main 时用 |
| `Dockerfile.a3` 的 `ARG VLLM_TAG=` | 如 `v0.27.1` | 与上两者一致，可交叉验证 |

## 3. 端口规划（硬规则）

| 用途 | 端口 | 原因 |
|------|------|------|
| mooncake master RPC | **50088** | 50051 被宿主机 OceanStor DTMA 占用（51 和 112 都踩过），**禁止用 50051** |
| mooncake metrics (HTTP) | **9008** | 本库脚本 `--metrics_port` 覆盖值；官方 mooncake 默认 **9003**，换机按实际启动参数对齐，勿照抄 |
| vllm API | 8004 | 与 master/metrics 错开 |
| memcache MetaService | 5000/6000/8000 | meta_service / config_store / metrics；宿主机占用时改 5001/6001/8001（见 04_backend_startup/memcache_startup.md §3.10） |

## 4. 共享服务器纪律

| 约束 | 说明 |
|------|------|
| NPU 占用 | 先 `npu-smi info` 查空闲，只申明空闲 chip（HBM ~3GB 为空闲，~60GB 为被占） |
| 端口冲突 | 启动前做端口预检（ss/netstat 确认端口空闲） |
| 容器命名 | 格式 `<用途>_<编号>`，SSH 端口标记与编号一致（如 `refactor_8203` ↔ 8203） |
| 代理设置 | 服务器不能直连公网，必须走 CCW 代理（squid, 端口 3128），代理 IP 按服务器网段选择（90/141/80 网段各有代理 IP，新网段需确认后回写） |
| 代理生效 | 代理只在当前 shell 生效，**每个新 ssh/docker exec 会话都要重新 export** |
| git 配置 | 容器内 `git config --global http.sslVerify false`（代理 MITM 证书，必须关校验）；git 不单独配 proxy，走环境变量 |
| pip 源 | 容器 pip 默认已配华为云源（无需代理也可尝试，实测走代理更稳） |

## 5. 代码位置（硬规矩）

- 仓库必须 clone 到 **`/home/lizhongyang/` 下**，不要乱放
- 首选 `/home/lizhongyang/map_XX/`（与本地 agent_project/map_XX 镜像同构）；`/home/lizhongyang/code/` 是历史位置也可
- remote：vllm 用官方 `https://github.com/vllm-project/vllm`；vllm-ascend 用个人 fork `https://github.com/lizy124/vllm-ascend`（PR 分支在 fork 上）
- 注意：只允许 pull/fetch，**禁止 push**（IRON RULE）

## 6. 经验沉淀规则

1. **新 lesson 必须回写**：任何服务器上实操得到的新经验/新坑/新判定标准/新硬限，完成后回写 `Templates/` 对应章节。
2. **服务器特定信息不放这里**：某台服务器的端口占用/权重路径/容器名等，放 `Technical/` 对应专项。
3. **只放通用方法论**：换个服务器、换个 PR 依然成立的规则才入库。
4. **修改同步检查**：`Templates/_common/common_constraints.md` 的公共约束（IRON RULE、端口）与其他区文档保持一致。