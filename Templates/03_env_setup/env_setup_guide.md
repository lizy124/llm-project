# env_install — vllm-ascend 环境安装完整指南（通用层，agent 必读）

> 本文档由 `env_install/` 目录下的 README + 0~6 号文件整合而成，**单一入口、可直接照做**。
> 源文件全部保留，本文档是"通用规则层"的合并视图。
>
> - 适用：**任何服务器**（51 / 112 / 未来新机）从零搭建 vllm-ascend 验证环境
> - 具体服务器实例（选了哪个镜像、容器名、实测结果）→ 对应 `map_XX/PREP.md` 与 `map_XX/README.md`
> - 流程总入口：[../AGENTS.md](../AGENTS.md) §2 七阶段（本指南 = Step4 建容器 + Step5 部署代码 的细化）
> - 版本配对表：见 [../AGENTS.md](../AGENTS.md) §2 Step5

---

## 目录

| 序号 | 阶段 | 文件（源） | 一句话 |
|------|------|-----------|--------|
| 0 | 总览 | README.md | 用途、分层、操作顺序、依赖 |
| 1 | Step 0 读参考教程 | 0_tutorials.md | 装之前先读用户指定的参考文档 |
| 2 | Step 1 选镜像 | 1_image_selection.md | day-build + 日期新 + Ubuntu24.04 + 跟随主流 |
| 3 | Step 2 建容器 | 2_container_creation.md | 标准 docker run 模板，照抄，别自作主张 |
| 4 | Step 3 通网络 | 3_network.md | CCW 代理按网段选，git/pip 都靠它 |
| 5 | Step 4 查版本卸载 | 4_version_check_uninstall.md | 查两包版本→决定重装→先卸载干净 |
| 6 | Step 5 装 vllm | 5_vllm_install.md | checkout 目标 tag → editable |
| 7 | Step 6 装 vllm-ascend | 6_vllm_ascend_install.md | checkout 目标分支 → editable |
| 8 | 安装后端到端验证 | 汇总 | 一次跑完的验收清单 |

---

## 0. 总览

### 0.1 目录定位（分层心智模型）

| 层 | 位置 | 内容 |
|----|------|------|
| **通用规则层** | `env_install/`（本文档） | 跨服务器复用的方法与步骤，按序号排好 |
| **服务器实例层** | `map_XX/PREP.md` / `map_XX/README.md` | 该服务器选了哪个镜像、容器名、实测结果、已知约束 |

### 0.2 操作顺序（从零到可跑 vllm）

```
Step 0  读教程     → 参考文档          （装之前先读用户指定的参考文档）
Step 1  选镜像     → 镜像选择规则      （day-build + 日期新 + Ubuntu24.04 + 跟随主流）
Step 2  建容器     → 容器创建模板      （标准 docker run，照抄）
Step 3  通网络     → 网络用法          （CCW 代理按网段选，git/pip 都靠它）
Step 4  查版本卸载 → 版本检查与卸载    （查两包版本→决定重装→先卸载干净）
Step 5  装 vllm   → vllm 安装          （checkout 目标 tag → editable）
Step 6  装 vllm-ascend → vllm-ascend 安装（checkout 目标分支 → editable）
```

### 0.3 顺序即依赖

- 没容器装不了包（Step2→4/5/6）
- 没网络 fetch 不了代码（Step3→5/6）
- 不查版本就不知道要不要重装、不卸干净就会新旧混装（Step4→5/6）
- **vllm 版本必须先于 vllm-ascend 确定**（配对关系见 [../AGENTS.md](../AGENTS.md) §2 Step5）

### 0.4 通用已知事实

- vllm 与 vllm-ascend **强耦合**，配错版本 = 启动即崩（配对表见 AGENTS.md §2 Step5）
- 容器预装的 vllm / vllm-ascend 版本**大概率不是目标版本**，需 checkout 源码 + editable 安装覆盖
- 历史坑：setuptools-rust 卡 vllm editable 安装（mock 方案：map_51/archive/）
- **经验沉淀规则：每次实操得到的新经验/新坑，必须同步写回 `env_install/` 对应文件**（不分服务器）

---

## 1. Step 0 — 读参考教程

### 1.1 教程清单

| # | 教程 | 位置 | 适用场景 | 优先级 |
|---|------|------|----------|--------|
| 1 | **vllm-ascend 仓库的 Dockerfile.a3**（及 Dockerfile/Dockerfile.a5 等） | 每个 vllm-ascend 代码库根目录，如 `map_XX/vllm-ascend/Dockerfile.a3` | vllm / vllm-ascend 安装命令与顺序的**权威参考** | ★★★ 最高 |
| 2 | vllm-ascend 版本配对文件 | `.github/vllm-release-tag.commit` / `.github/vllm-main-verified.commit` | 确定 vllm 目标版本 | ★★★ |
| 3 | vllm-ascend 官方安装文档 | https://docs.vllm.ai/projects/ascend/en/latest/ （getting_started/installation） | 背景 | ★ |

### 1.2 Dockerfile.a3 的安装序列（摘要）

```
① vllm:   VLLM_TARGET_DEVICE="empty" pip install -e .[audio] --extra-index-url https://download.pytorch.org/whl/cpu/
② vllm:   pip uninstall -y triton
③ ascend: source ascend-toolkit/set_env.sh + source nnal/atb/set_env.sh
④ ascend: PIP_EXTRA_INDEX_URL=华为云ascend源 + VLLM_BATCH_INVARIANT=1 + SOC_VERSION=ascend910_9391
          pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu/
⑤ ascend: pip uninstall -y triton triton-ascend → pip install triton-ascend==3.2.2（华为云ascend源）
⑥ ascend: pip install concurrent-log-handler
```

> 细节展开见本指南 Step 5 / Step 6。

---

## 2. Step 1 — 选镜像

### 2.1 规则（按优先级）

1. **选 `vllm-ascend:dev-<版本>.day<YYYYMMDD>-800I-A3-py311-Ubuntu24.04-lts-aarch64` 这类 day-build 镜像**
2. **日期尽量新**
3. **尽可能选 `Ubuntu24.04-lts-aarch64`**（OS 底座，偏好而非硬性）
4. **参考该服务器上大多数容器在用的类型**——别人都在用的就是经过实战检验的，跟随选择

### 2.2 操作步骤

```bash
# 1) 盘点镜像
docker images --format '{{.Repository}}:{{.Tag}}|{{.Size}}|{{.CreatedSince}}'
# 2) 对照运行中容器，统计主流镜像类型
docker ps --format '{{.Names}}|{{.Image}}'
```

在满足"日期尽量新 + Ubuntu24.04"的候选里，优先选**多数容器在用**的那款。

### 2.3 注意

- 镜像自带的 vllm/vllm-ascend 会被 clone 的源码 + editable 安装覆盖，
  **选镜像本质是选基础系统层**（CANN / torch / py3.11 / 工具链 / Ascend 适配），不是选 vllm 版本
- 若服务器上没有 day-build 镜像，退而求其次选 `quay.io/ascend/vllm-ascend` 官方镜像，同样遵循"多数容器在用什么就选什么"
- 服务器具体选了哪个镜像 → 记入该服务器 `map_XX/PREP.md`（Step 4 行）和 `map_XX/README.md`

### 2.4 实例参考

- 51（2026-08-25）：`dev-26.2.0.day20260819`（最新 day-build + 3 个容器在用）→ [../map_51/README.md](../map_51/README.md)
- 112（2026-08-25）：主流为 `dev-26.2.0.day20260821-openEuler24.03`，同系有 `dev-26.1.0.cann9.1.0.day20260730-...Ubuntu24.04` → 建 112 容器时按规则权衡，见 [../map_112/PREP.md](../map_112/PREP.md)

---

## 3. Step 2 — 建容器

### 3.1 标准 docker run 模板（用户惯例，A3/NPU 服务器）

```bash
docker run -dit -u root \
 -p 0.0.0.0:<SSH端口>:22 \                # 按容器名编号（如 refactor_8203 用 8203）；--net=host 下仅作标记
 --name <容器名> \
 -e ASCEND_RUNTIME_OPTIONS=NODRV \         # 关键：容器内无驱动，用宿主机挂载驱动
 --privileged=true \                       # 关键：特权模式，NPU 设备直通必需
 -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware \
 -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver \
 -v /home:/home \
 -v /data:/data \
 -v /tmp:/tmp \
 -v /mnt:/mnt \
 -v /usr/local/sbin/:/usr/local/sbin \
 -v /etc/hccn.conf:/etc/hccn.conf \
 -v /etc/ascend_install.info:/etc/ascend_install.info \
 -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
 --shm-size=100g \
 --net=host \
 --cap-add=SYS_PTRACE \                    # 调试用（strace/gdb）
 --security-opt seccomp=unconfined \       # 关闭 seccomp 限制
 -w /home \
 <镜像ID或名> \
 /bin/bash
```

### 3.2 参数作用表

| 参数 | 作用 | 缺了会怎样 |
|------|------|-----------|
| `--privileged=true` | 特权模式 | npu-smi 报 `dcmi module initialize failed -8005`，NPU 不可见（**实测踩过**） |
| `-e ASCEND_RUNTIME_OPTIONS=NODRV` | 声明容器内无驱动，走宿主机挂载 | torch_npu 运行时可能找不到驱动 |
| `--cap-add=SYS_PTRACE` | 进程调试权限 | 无法 strace/gdb 排障 |
| `--security-opt seccomp=unconfined` | 解除系统调用限制 | 部分诊断工具行为异常 |
| `--net=host` | 宿主网络 | vllm/mooncake 端口直接用宿主的，无需映射（注意端口冲突检查） |
| `--shm-size=100g` | 共享内存 | 多卡推理共享内存不足 |
| `-u root` | root 用户 | 权限问题 |

### 3.3 建容器前必查

1. **先抄本地惯例**（每台机器挂载细节可能不同，本模板为基准）：

```bash
docker inspect <现有正常容器> --format '{{json .HostConfig.Binds}} {{.HostConfig.Privileged}}'
```

2. **NPU 占用**：`npu-smi info` —— 共享服务器只申明空闲 chip（HBM ~3GB 为空闲，~60GB 为被占）
3. **端口冲突**：50051 / 8004 / 9008 / 50088 谁被占（host 网络下端口直接用宿主的）

### 3.4 建容器后首查

```bash
npu-smi info | grep -c 0000:            # chip 可见数（应为预期数量）
python3 --version; which python         # python 环境
pip list | grep -iE "vllm|torch|mooncake"  # 预装包（对照目标版本）
ls /usr/local/Ascend/                   # CANN 目录
```

### 3.5 容器命名与端口惯例

- 格式 `<用途>_<编号>`，SSH 端口标记与编号一致（如 `refactor_8203` ↔ 8203）
- 建容器脚本放 `map_XX/start/create_<容器名>.sh`（幂等：已存在则删了重建，容器无状态、数据都在挂载卷上）

### 3.6 实例参考

- 51 `kv_metrics_51`（PR14912 验证）：[../map_51/start/create_kv_metrics_51.sh](../map_51/start/create_kv_metrics_51.sh)

---

## 4. Step 3 — 通网络

### 4.1 核心结论

**服务器不能直连公网，必须走 CCW 代理（squid, 端口 3128），代理 IP 按服务器网段选择。**
假象提醒：curl 直接测公网会失败（DNS 解析到 IPv6 而 IPv6 不通），**不代表没网**，走代理即通。

### 4.2 代理选择规则（代理 IP 的网段与服务器网段匹配）

| 服务器网段 | CCW 代理（端口 3128） | 适用示例 |
|-----------|----------------------|----------|
| 90.x.x.x | `90.255.140.5:3128` | 90 网段机器 |
| **141.x.x.x** | **`141.2.174.135:3128`** | **51（141.61.81.51）** |
| **80.x.x.x** | **`80.254.14.6:3128`** | **112（80.5.17.112）** |

> 新服务器网段不在表内：先向用户确认该网段的 CCW 代理 IP（或宿主机上已有容器的代理配置），
> 确认后务必把新网段补写回本表（经验沉淀规则）。

### 4.3 代理设置（标准用法）

```bash
# 51 服务器（141 网段）
CCW_HOST_IP=141.2.174.135
export http_proxy="http://${CCW_HOST_IP}:3128"
export https_proxy=${http_proxy}
export ftp_proxy=${http_proxy}

# git 配套设置（一次性，容器内）
git config --global http.sslVerify false      # 代理 MITM 证书，必须关校验
git config --global --unset http.proxy        # git 不单独配 proxy，走环境变量
git config --global --unset https.proxy
```

```bash
# 112 服务器（80 网段）
CCW_HOST_IP=80.254.14.6
export http_proxy="http://${CCW_HOST_IP}:3128"
export https_proxy=${http_proxy}
export ftp_proxy=${http_proxy}
```

### 4.4 pip 源

容器 pip 默认已配华为云源（无需代理也可尝试，实测走代理更稳）：
- index-url: `https://repo.huaweicloud.com/repository/pypi/simple`
- extra-index-url: `https://mirrors.huaweicloud.com/ascend/repos/pypi`
- trusted-host: repo.huaweicloud.com mirrors.huaweicloud.com mirrors.aliyun.com

### 4.5 git clone 来源（已实证 remote）

- vllm: `https://github.com/vllm-project/vllm`
- vllm-ascend: `https://github.com/lizy124/vllm-ascend`（个人 fork，PR 分支在此）
- 注意：只允许 pull/fetch，**禁止 push**（IRON RULE）

### 4.6 下载大文件（权重/依赖包）

<!-- 待用户说明 -->

### 4.7 已知坑

- **curl 直测公网失败是假象**：DNS 解析出 IPv6 地址（如 baidu → 2408:...），IPv6 路由不通；
  判断网络是否可用应走代理测：`curl -x http://141.2.174.135:3128 -sI https://github.com`
- 代理只在当前 shell 生效，**每个新 ssh/docker exec 会话都要重新 export**
  （脚本里用 curl/git 前先 export，或写入 ~/.bashrc）

---

## 5. Step 4 — 检查版本 / 决定重装 / 卸载干净

> 位置：建容器（Step2）、通网络（Step3）之后，装包（Step5/Step6）之前
> 原则：**先查两包版本 → 对照目标 → 不匹配才重装 → 重装前必须卸载干净**

### 5.1 检查预装版本（vllm 和 vllm-ascend 一起查）

```bash
pip show vllm        | grep -E '^(Version|Location|Editable)'
pip show vllm-ascend | grep -E '^(Version|Location|Editable)'
python3 -c "import vllm; print(vllm.__version__, vllm.__file__)"
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)"
```

关注三点：
- **Version** 是否等于目标版本
- **Location** 是 site-packages（wheel 预装）还是源码目录（editable 安装）
- **import 解析路径**是否指向预期位置

### 5.2 对照目标版本，决定是否重装

目标版本由验证任务决定（vllm↔vllm-ascend 配对表见 [../AGENTS.md](../AGENTS.md) §2 Step5）。

**经验：容器预装版本大概率不符**——day-build 镜像带的是构建当日的 main 快照，不是你要的 tag/分支。

| 判定 | 动作 |
|---|---|
| 两包都符合目标 | 跳过重装，直接做 Step5/6 的安装后验证 |
| 任一不符 | 重装（两包强耦合，通常**一起卸、一起装**） |

### 5.3 卸载干净（重装前必做）

```bash
pip uninstall -y vllm vllm-ascend
```

pip uninstall 能删 wheel 装的包体，但 **editable 安装的残留不会自动清**，必须手动复查：

```bash
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
ls $SP | grep -iE "vllm|ascend"    # 应为空（或只剩 torch_npu 等无关包）
```

重点残留物（有则手动 rm -rf）：

| 残留物 | 来源 |
|---|---|
| `vllm*.dist-info` / `vllm_ascend*.dist-info` | pip 元数据 |
| `__editable__.vllm*.pth` / `__editable__*.py` | editable 安装的 finder |
| `vllm/` / `vllm_ascend/` 目录 | wheel 解压残留 |
| 源码目录里的 `vllm.egg-info` / `vllm_ascend.egg-info` | 挂载卷上的构建残留，**容器重建也不消失** |

**卸载干净的判定标准：**

```bash
python3 -c "import vllm" 2>&1 | grep -q "No module" && echo OK-vllm || echo 残留-vllm
python3 -c "import vllm_ascend" 2>&1 | grep -q "No module" && echo OK-ascend || echo 残留-ascend
```

### 5.4 坑

- **editable 残留不清** → `pip install -e .` 后 import 仍解析到旧路径：明明 checkout 了新代码、版本号也对，行为却是旧的，极难排查
- 容器删了重建可重置 site-packages，但 `/home` 挂载卷上源码目录里的 `egg-info` 还在 → **重建容器后也要查源码目录残留**
- 镜像预装可能带 `+empty` 后缀版本（如 `0.26.0+empty`），这是 wheel 占位包，照常卸载即可

### 5.5 实证记录

- 51 kv_metrics_51（2026-08-25）：预装 `vllm 0.26.0+empty`（wheel）+ `vllm-ascend 0.19.1rc2.dev1350`（wheel），
  对照目标 v0.27.1 + kv_metrics_observability → 两包均不符 → 卸载后 editable 重装
- 卸载结果：`pip uninstall` 两包干净（site-packages 无残留、无 `__editable__` 残留），
  **但两个源码目录都有 `egg-info` 残留**（/home/lizhongyang/code/vllm/vllm.egg-info、
  vllm-ascend/vllm_ascend.egg-info——PR14465 时代 editable 安装留下的，挂载卷上存活至今）→ `rm -rf` 清掉。
  证实"egg-info 在挂载卷上、容器重建也不消失"这条坑真实存在。脚本：map_51/test/uninstall_vllm_clean.sh

---

## 6. Step 5 — 装 vllm（必须先于 vllm-ascend）

> 前置：Step 4 已查版本并卸载干净；目标 vllm 版本已由 vllm-ascend 分支配对确定

### 6.1 权威参考

每个 vllm-ascend 代码库都有 `Dockerfile.a3`（及类似 Dockerfile），**vllm 和 vllm-ascend 的安装方式以它为准，不自由发挥**。
安装顺序：**vllm 先装 → vllm-ascend 后装**（顺序重要）。

### 6.2 目标版本怎么定（由 vllm-ascend 分支决定）

vllm 版本不独立选择，**根据 vllm-ascend 的 commit 去对应**。vllm-ascend 仓库自带配对文件：

| 文件 | 内容 | 用途 |
|---|---|---|
| `.github/vllm-release-tag.commit` | 如 `v0.27.1` | release 配对（**常用**） |
| `.github/vllm-main-verified.commit` | commit hash | 跟 vllm main 时用 |
| `Dockerfile.a3` 的 `ARG VLLM_TAG=` | 如 `v0.27.1` | 与上两者一致，可交叉验证 |

### 6.3 代码位置（硬规矩）

- 仓库必须 clone 到 **`/home/lizhongyang/` 下**，不要乱放
- 首选 `/home/lizhongyang/map_XX/`（与本地 agent_project/map_XX 镜像同构）；`/home/lizhongyang/code/` 是历史位置也可
- remote：vllm 用官方 `https://github.com/vllm-project/vllm`；vllm-ascend 用个人 fork `https://github.com/lizy124/vllm-ascend`（PR 分支在 fork 上）

### 6.4 安装命令

#### 方案 A（推荐）：Dockerfile.a3 原文

```bash
cd /home/lizhongyang/map_XX/vllm
git checkout <目标tag>          # 由 6.2 节配对文件确定

# 关键：VLLM_TARGET_DEVICE="empty" 跳过 CUDA 编译（Ascend 上无 NVCC）
VLLM_TARGET_DEVICE="empty" python3 -m pip install -e .[audio] \
    --extra-index-url https://download.pytorch.org/whl/cpu/

# vllm 会带上 x86 的 triton，Ascend 上不可用，装完立即卸载
python3 -m pip uninstall -y triton
```

#### 方案 B（降级）：容器已锁定 torch/torch_npu 配对时

真实 51 容器因动不了 torch/torch_npu 配对，才用 `--no-deps`；照搬方案 A 的 pytorch extra-index 会撞代理自签证书 SSL 失败（PR14465 实证）。

```bash
VLLM_TARGET_DEVICE="empty" python3 -m pip install -e .[audio] --no-deps
python3 -m pip uninstall -y triton
```

> **选择条件：默认先按方案 A（Dockerfile.a3 原文）；只有容器已有锁定的 torch/torch_npu 配对、且照 A 撞 SSL/版本冲突时才降级方案 B。**
> 另外：`setuptools_rust` 缺失时直接 `pip install setuptools-rust==1.9.0` 装真包（走代理即可，勿 mock）。

### 6.5 已知坑

- 忘了 `VLLM_TARGET_DEVICE="empty"` → 尝试编译 CUDA kernel 失败/卡死
- 忘了卸 triton → 与后续 triton-ascend 冲突（vllm 装完即卸 triton）
- `[audio]` extra 别丢（Dockerfile 原样如此）
- **setuptools_rust 缺失** → day-build 镜像没带；**直接装真包，勿 mock**（mock 不全会在 metadata 阶段 import 崩）
- **`--no-deps` 是"容器实测适配"flag，不是 Dockerfile.a3 自带的**（见 6.4 方案选择）

### 6.6 安装后验证

```bash
pip show vllm | grep -E 'Version|Editable'   # 版本正确 + Editable 指向源码
python3 -c "import vllm; print(vllm.__version__)"
pip list | grep -iE "^triton"                # 应无 triton（triton-ascend 由 Step6 装）
```

### 6.7 实证记录

- 51（2026-08-25）：kv_metrics_observability (cf1296559) 配对 v0.27.1，三处证据一致
  （release-tag.commit / main-verified.commit / Dockerfile.a3）

---

## 7. Step 6 — 装 vllm-ascend（必须在 vllm 之后）

> 前置：Step 5 已装好配对 vllm 并卸掉 triton；本包分支多为**个人 fork** 的 PR 分支

### 7.1 权威参考

安装命令照 Dockerfile.a3，**先 source CANN 环境，再装**。顺序：vllm（Step5）→ vllm-ascend（本步）。

### 7.2 分支来源

- vllm-ascend 多为个人 fork 分支（PR 验证场景）：`https://github.com/lizy124/vllm-ascend`
- 先切到目标分支，**再从该分支的配对文件反查 vllm 版本**（见 Step 5 §6.2）

### 7.3 安装命令（照 Dockerfile.a3，含实测适配）

```bash
cd /home/lizhongyang/map_XX/vllm-ascend
git checkout <目标分支>                     # 如 kv_metrics_observability

# 1) CANN 环境（两个 set_env.sh 都要 source）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 2) 安装（含编译自定义 kernel，耗时较长）
export PIP_EXTRA_INDEX_URL="https://mirrors.huaweicloud.com/ascend/repos/pypi"
export VLLM_BATCH_INVARIANT=1
export SOC_VERSION=ascend910_9391            # 800I A3；Dockerfile.a3 同款（其他芯片按实际改）
python3 -m pip install -v -e . --no-deps --no-build-isolation

# 3) triton 清理与替换（顺序不能乱）
python3 -m pip uninstall -y triton triton-ascend
python3 -m pip install triton-ascend==3.2.2 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
python3 -m pip install concurrent-log-handler
```

> **实测适配说明**：用户惯例用 `pip install -v -e .`，配 `--no-deps --no-build-isolation` 即可。
> **别照抄 Dockerfile.a3 的 `--extra-index-url https://download.pytorch.org/whl/cpu/`**——走代理时 pip 拉 build 依赖会撞自签证书 SSL 崩（本步实测 v0.1 失败的根本原因）。

### 7.4 已知坑

- 安装命令会编译自定义 kernel（build.sh --ops= 一串算子 + cmake/ninja -j1280 target package），**安静 + 耗时是正常的，别误判卡死**
- **确认 vllm 版本要从 vllm-ascend 分支的配对文件反查**，不是随便挑。且**实际符号名要以分支代码为准**：
  - 例（kv_metrics_observability 分支）：vllm side 是 `KVConnectorStats`（无 `Base`），包内是 `AscendStoreKVConnectorStats(KVConnectorStats)`
  - 报错信息里的抽象称呼（如 `KVConnectorStatsBase`）**不是真实可 import 符号**——校验文案别照抄
- **不 source set_env.sh 直接装** → 找不到 CANN 工具链，编译 kernel 失败
- triton/triton-ascend 替换顺序乱 → 旧 triton-ascend 残留导致运行时异常
- `SOC_VERSION` 不对 → 编译产物与芯片不匹配（800I A3 用 `ascend910_9391`）
- 历史坑：ChunkedTokenDatabase 签名 bug（a72196356 缺 use_hybrid，上游已修）

### 7.5 安装后验证

```bash
pip show vllm | grep -E 'Version|Editable'
pip show vllm-ascend | grep -E '^Version'
pip list | grep -iE "^triton"        # 稳定版配合：triton 3.5.0 + triton_ascend 3.2.2
python3 -c "
import vllm, vllm_ascend
print(vllm.__version__)              # 0.27.1
"
```

> **分支真实 import 校验**（按目标分支代码改写，勿抄审查报告符号；示例为 kv_metrics_observability 分支）：
> ```bash
> python3 -c "
> from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
> from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metrics import AscendStoreKVConnectorStats
> import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector
> print('ascend_store 全链路 OK')
> "
> ```

### 7.6 实证记录

- 51 kv_metrics_51（2026-08-25）：kv_metrics_observability (cf1296559) ← fork lizy124/vllm-ascend

---

## 8. 安装后端到端验证清单（一次跑完）

装完两包后，按顺序核对以下全部通过才算环境就绪：

| # | 检查项 | 命令 | 期望 |
|---|--------|------|------|
| 1 | vllm 版本 + editable | `pip show vllm \| grep -E 'Version\|Editable'` | 版本=目标，Editable 指向源码 |
| 2 | vllm-ascend 版本 | `pip show vllm-ascend \| grep -E '^Version'` | =目标分支版本 |
| 3 | triton 无残留 | `pip list \| grep -iE "^triton"` | 只剩 triton 3.5.0 + triton_ascend 3.2.2 |
| 4 | import 两包 | `python3 -c "import vllm, vllm_ascend; print(vllm.__version__)"` | 无报错，版本正确 |
| 5 | 分支真实符号 | 见 §7.5（按目标分支改写） | 全链路 OK |
| 6 | NPU 可见 | `npu-smi info \| grep -c 0000:` | =预期 chip 数 |

---

## 9. 通用已知事实与坑清单（跨服务器）

| 坑 / 事实 | 规避 / 说明 |
|---|---|
| vllm ↔ vllm-ascend 版本不配对 | 启动即崩；先查配对表（AGENTS.md §2 Step5），失败先查版本 |
| 容器预装版本大概率不符 | Step 4 必查，不符则重装 |
| 50051 端口被占（51/112 都踩过，OceanStor DTMA） | mooncake 一律用 50088 |
| 共享服务器 NPU 被他人占用 | 先 npu-smi 查 HBM，只申明空闲 chip |
| `--privileged` 缺失 | npu-smi 报 `-8005`，NPU 不可见 |
| setuptools-rust 卡 vllm editable 安装 | 装真包 `setuptools-rust==1.9.0`，勿 mock |
| editable 残留不清 | import 解析到旧路径，行为新旧混杂；卸载后查 site-packages + 源码目录 egg-info |
| 代理只在当前 shell 生效 | 每个新 ssh/docker exec 会话重新 export |
| curl 直测公网失败是假象（IPv6） | 走代理测 `curl -x http://<proxy>:3128 -sI https://github.com` |
| pkill 杀不死 mooncake | 显式 PID `kill -9` |

---

## 10. IRON RULE（最高优先级，无例外）

1. **任何材料不得上传公网**（云存储/网盘/公共服务一律禁止）
2. **任何情况下禁止执行 `git push`**（内网 remote 也不行；`git pull`/`git fetch` 允许）
3. 允许：与内网服务器（51/112）的 scp/rsync 等非 git 传输

---

## 11. 装完之后（不属于本目录，指向对应位置）

- 启动 mooncake / vllm → `map_XX/start/`（参考 51 的现有脚本）
- 验证任务 → `map_XX/test/` + 结论归档 `map_XX/record_final/`

---

## 12. 各源文件状态与本文档对照

| 源文件 | 状态 | 对应章节 |
|--------|------|----------|
| README.md | 框架 ✅ | §0 / §9 / §10 / §11 |
| 0_tutorials.md | ✅ 用户指定第一参考（2026-08-25） | §1 |
| 1_image_selection.md | ✅ | §2 |
| 2_container_creation.md | ✅ | §3 |
| 3_network.md | ✅ | §4 |
| 4_version_check_uninstall.md | ✅ 流程已实证 | §5 |
| 5_vllm_install.md | ✅ 流程已实证 | §6 |
| 6_vllm_ascend_install.md | ✅ 流程已实证 | §7 |