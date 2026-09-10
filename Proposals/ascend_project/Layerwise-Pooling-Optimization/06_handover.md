# 交接文档 — Layerwise 池化性能验证（165/pool_165）

> 更新时间：2026-09-07（晚第三次更新：栈就绪+启动战役全记录+protocol 修复+165 重启中）
> 用途：新会话接续。包含：目标、已完成事实、启动失败全复盘（7 次尝试逐一定位）、当前状态（**165 重启中**）、重启后 runbook、测量方案、文件清单、操作教训。

---

## 0. 当前目标与状态（先读这段）

**目标**：测出 memcache layerwise 当前的性能劣化%（DRAM 池化命中态 vs HBM PrefixCache 基线，指标 Prefill TPS / TTFT）。验收口径见 [01_requirements_analysis.md](01_requirements_analysis.md) §1.4：DSV4-Flash、128K、90% 命中、劣化 ≤5%。

**路径（09-07 晚定型）**：不换服务器。pool_165 上已建 `layerwise_dsv4_v2` 栈 = pr15854 分支（main@0905 + #15854 全部 4 提交）+ cherry-pick #15442（多 main spec 适配）。UT 363 全绿。**09-07 下午用户已与同事协调，16 卡归我们独占使用**（zhuyixiang、vllm_lmt 的任务已被 kill）。

**当前状态（接手第一件事看这里）**：
1. **165 服务器已执行 `reboot`（09-07 ~15:10）**，原因是 attempt 7 挂在 `HalMemCreate 128GB 失败 ret:6`（VMM 碎片/内存不连续，用户判断重启可清）。
2. 重启后按 §3.1 runbook 恢复：起容器 → 起服务 → 轮询 → 冒烟 → 测量。
3. **已攻克的所有关卡**（详见 §1.5）：5-spec 布局 ✓（PR#15442 生效实证）、host_rdma→device_sdma protocol 修复 ✓、僵尸清理 ✓。
4. **唯一剩余关卡**：128GB DRAM 池 `HalMemCreate ret:6`。重启后若复发 → 把 `dram.size` 从 128GB 降到 64GB（见 §3.1 步骤 6 的分析：16 worker × 128GB = 2TB > 宿主 2013GB，若为 per-rank 分配则必超卖）。

---

## 1. 已完成的事实（数据均已落盘）

### 1.1 HBM 基线（对照组，完成，但注意形态差异）

环境：165 容器 `pool_165`，DSV4-Flash-**w4a8**-mtp，**TP8**，`--enable-prefix-caching`，端口 8004，旧栈 vllm-ascend 0.19.1rc2.dev1561。

| 场景 | 命中率 | 命中态 TTFT（单请求重发） | 并发 mean_lat |
|---|---|---|---|
| 64K×4 前缀 | 99.61%（并发8/16 混合） | 0.42~0.52s | c8: 1.35s / c16: 2.67s |
| 128K×2 前缀 | 99.81%（并发4/8） | 0.70~0.75s | c4: 1.42s / c8: 2.79s |

数据：本地 [baseline/data/hbm_baseline_64k.json](baseline/data/hbm_baseline_64k.json)、[baseline/data/hbm_baseline_128k.json](baseline/data/hbm_baseline_128k.json)；远端 165 容器内 `/home/lizhongyang/pool165/data/` 同款。

⚠️ **形态差异警告**：此基线是 w4a8+TP8+无 MTP，与池化组目标形态（w8a8+TP4×DP4+MTP3+1M）不同。**口径B 的严格对照需要在同形态下跑一次 HBM prefix-caching**（改 [start_hbm_baseline_165.sh](baseline/run/start_hbm_baseline.sh) 的权重/TP/DP/MTP 对齐即可），或向同事索要同形态 TPS/TTFT。

### 1.2 DSV4 + mooncake 非-layerwise：通路验证成功（但非需求路线）

`backend=mooncake`（无 use_layerwise）启动正常，单前缀 64K+256 二发：冷 31.78s（queries 65792 / hits 0）→ 同前缀重发 0.68s（hits 65536，前缀 100%）。DSV4 的 DRAM 池化 put/get/命中链路成立，但整块传输无逐层 overlap，仅作兜底。

### 1.3 旧栈 5-spec 崩溃的根因与 PR 修复（历史背景，已解决）

旧栈（main@0818 快照）启动即崩：`Physical layer 2 with multiple cache specs must have exactly one main spec...got [5 个 spec]`。校验点 `layerwise_cache_layout.py build_layerwise_reuse_layout()`（旧版 217-224 行）。

**修复 = PR #15442**（作者 lsy0214 即同事本人）：多 main spec 层选 `.attn` 为 main，其余非 indexer spec 进 `extra_main_specs`，indexer 可选，单 spec 层 fast path，全 indexer 层仍 raise。带完整 UT。这正是 06 文档此前评估的"中大型改造"——已有人做完且公开。
**PR #15854**（tyy0829，Closed 未合入）= 表格"修复内存"：layerwise 传输只存 reachable blocks，单条 128K 请求池内存 45.6GB→1.4GB（解释多前缀 0.75%→87.50%：32 前缀×45.6GB 撑爆池，修复后放得下）。

### 1.4 同事测试数据（DSV4+layerwise 已跑通的铁证，2026-09-05）

> 完整整理见 [07_colleague_test_data.md](07_colleague_test_data.md)（含完整启动脚本）。要点：

| 轮次 | 形态 | 实际命中率 |
|---|---|---|
| 第1次 | 128K×32 前缀、128 请求、并发 32 | 0.75% |
| 修复内存后 | 同上 | **87.50%**（=4096 粒度截断 114688/131072） |
| 第3次 | 16K×1 前缀 | 76.78%（=12288/16000，含 MTP trim 修复） |

- 环境：A3、16 卡 PD混部、TP4×DP4、w8a8-mtp、block 32、MTP(3)、`--no-enable-prefix-caching`、AscendStoreConnector + memcache + `use_layerwise: true` + `layerwise_prefetch_layers: 3`
- 同事栈 = main@0827 + #15442 + #15854（两 PR 均未进 upstream）
- #15442 自测（8x910B3、host_shm）：layerwise OFF 146.70 / ON 53.76 / ON+prefetch 55.37 tok/s——ON 劣化 63%、prefetch +3%，距 ≤5% 目标很远（方向性参考）
- **TPS/TTFT 数据仍缺**——正式测量是本任务核心目标

### 1.5 栈构建 + 启动战役全记录（09-07，本会话核心产出）

#### (a) 栈：`layerwise_dsv4_v2`（就绪，UT 363 全绿）

- 构建：pool_165 内 `git fetch origin pull/15854/head:pr15854` → 以 pr15854（main@~0905 + #15854 全部 4 提交）为基线 → `git cherry-pick 923be9030`（#15442 唯一提交）→ 零冲突。
- 本地 delta：main 新测试 `test_ambiguous_multi_spec_layer_is_rejected`（#15367 引入）与新语义冲突，已改写为 `test_multi_main_spec_layer_selects_attn_as_main`。
- 分支位于 `/vllm-workspace/vllm-ascend`（editable 安装即时生效）。`layerwise_dsv4`（main@0827+15442）留作精确复刻备选。
- 构建脚本：`D:\lzy\project\kv_pool\tmp\build_stack_v2.sh`。

#### (b) 七次启动尝试：每次死点与根因（证据链完整）

| # | 时间 | 配置 | 死点 | 根因 | 判定 |
|---|---|---|---|---|---|
| 1 | 11:57 | 16卡 | worker init：`aclrtMallocHostWithCfg 207001` 锁页 OOM | zhuyixiang 12:01:12 起了 16 卡实例，加载风暴挤爆驱动锁页池 | 环境冲突 |
| 2 | 12:14 | 16卡 | HBM 检查 `Free 2.13/61.27 GiB` | zhuyixiang 满载 60G/卡（device 8-15） | 环境冲突 |
| 3 | 12:56 | 8卡 1M | KV 容量：1M 需 17.07G > 可用 7.85G | 8 卡物理不够 | 容量 |
| 4 | 13:01 | 8卡 128K | KV 容量：128K 需 11.48G > 7.85G | 同上。**发现：单请求固定开销 ≈10.7G（compressor/indexer state，与长度无关）；8 卡数学上不可行，这就是同事用 16 卡的原因** | 容量 |
| 5 | 14:20 | 16卡 | memcache `assert res == 0` ← `HYBM LoadExtendLibrary for HOST RDMA failed: -6` ← `Unable to open library [libhcom.so]` | **165 无 libhcom.so（宿主 CANN 只有 libhcom*m*.so 双 m），host_rdma 协议必需它**。同事平台镜像有此库 | **已修复** |
| 6 | 14:48 | 16卡 | HBM 检查 `Free 32.81GB` | 自己 attempt5 的僵尸 worker（setproctitle 改名 `VLLM::Worker_DP*`，`pkill -f "vllm serve"` 杀不到）+ vllm_lmt 的 Qwen3-8B profiler 抢卡 | **已修复** |
| 7 | 14:59 | 16卡 干净 | `HalMemCreate failed ret:6 size:137438953472`（128GB DRAM 池，1GB 页失败→2MB 页重试 196s 后失败） | VMM 碎片/内存不连续（用户经验：重启可清）；**备选解释：16 worker 各自分配 128GB = 2TB > 宿主 2013GB 总量（1473G 可用），若为 per-rank 物理分配则必超卖** | **待重启验证** |

#### (c) attempt 7 的关键里程碑（重启后应原样复现）

1. 15:03:35 主权重加载 22.6s（页缓存热）
2. 15:04:18 **16 个 worker 全部打印 `layerwise config: num_layers=44 num_groups=6 physical_layer_to_group_layers_sample={2: [(0,0),(2,1),(4,0)], ...}`（pool_worker.py:443）——5-spec layerwise 布局构建成功，PR#15442 在真实 DSV4-Flash 上生效，旧栈的 P0 崩溃已根除**
3. 15:04:38 起 DRAM 池映射（device_sdma，MMC/HYBM 错误已消失，protocol 修复生效）
4. 15:07:55 挂在 HalMemCreate 128GB

#### (d) protocol 修复详情（已落盘，重启后仍在）

- 文件（容器内）：`/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-local.conf`
- 改动：`ock.mmc.local_service.protocol = host_rdma` → `device_sdma`（备份：`mmc-local.conf.bak_host_rdma`）
- 依据：conf 注释原文 "host_rdma, host_urma and host_tcp require hcom library"；本机 s2 场景（DSV2-Lite+memcache+layerwise，09-01 跑通）用的就是 device_sdma；`memcache_backend.py` 有 `_is_device_sdma()` 一等公民分支
- 注意：A3 环境所有协议需 1GB 对齐（128GB 满足）；`dram.size=128GB` 是 09-07 为多前缀场景从 1GB 调大的

---

## 2. 环境现状（165 服务器）

- 登录：`ssh root@192.168.13.165`，密码 `!Q2w3e4r`；远程执行工具 `& D:\lzy\project\llm-project\utils\go.ps1 '<命令>'`
- **服务器重启中（09-07 ~15:10 执行 reboot）**。ssh 恢复后 docker 容器未必自启，需 `docker start pool_165`
- 容器：`pool_165`；栈 `/vllm-workspace/vllm-ascend` 分支 `layerwise_dsv4_v2`（重启不影响磁盘内容）
- 权重：`/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp`（与同事同款，页缓存热时 22s 加载完）
- MetaService：重启后**不在**，需在 pool_165 内拉起（conf 已是 device_sdma + 128GB）
- 其他容器（zhuyixiang 的、vllm_lmt 的）09-07 已被我们 kill 过任务；**16 卡独占是用户协调来的，注意别人可能再起任务抢卡**（NPU 上出现非本容器进程时，先 `cat /proc/<pid>/cgroup` 确认归属再动手，只杀自己的）
- watcher 自动脚本（watch_16c_v2.sh）已停止且锁已清；当前策略是手动重启验证，不需要 watcher

## 3. 下一步

### 3.1 重启后 runbook（按序执行，每步脚本都已就位）

1. **等 ssh 通**（`go.ps1 'uptime'`，reboot 后约 2-5 分钟）。
2. **起容器**：`docker ps -a | grep pool_165` 确认，`docker start pool_165` 若未跑。
3. **确认 NPU 干净**：宿主 `bash /home/lizhongyang/tmp/check_now.sh`（各卡 HBM 应回落 ~3GB 基线、进程表空）。
4. **拉 MetaService**：容器内执行 `cd /home/lizhongyang/map_165/run && nohup python3 -c "from memcache_hybrid import MetaService; MetaService.main()" > metaservice.log 2>&1 &`，`pgrep -f MetaService` 确认。
5. **启动 16 卡服务**：容器内 `bash /home/lizhongyang/tmp/launch_16c_v2.sh`（自动清僵尸含改名 worker + 清 /dev/shm/psm_* + 检查 MetaService + 备份旧日志 + 启动，端口 8100）。
6. **轮询**：容器内 `bash /home/lizhongyang/tmp/poll_layerwise.sh`，关注标记依次出现：`Loading weights took` → `layerwise config: num_layers=44` →（DRAM 池）→ `Application startup complete`。**若再次挂在 `HalMemCreate ... 137438953472 failed ret:6`：把 mmc-local.conf 的 `dram.size` 降到 `64GB`（16×64=1TB < 1473G 可用，口径A 池需求仅 ~45GB 量级，64G 绰绰有余），重启 MetaService + 服务重试**。若 64GB 仍挂再降 32GB 并重新评估 per-rank/per-node 分配语义（看 s2 的 pool_metrics.txt 或 memcache 文档）。
7. **冒烟**：容器内 `bash /home/lizhongyang/tmp/smoke_layerwise.sh`（发 2 个相同请求，看延迟差 + 日志 layerwise/lookup/hit 痕迹）。
8. 启动总时长预期 ~4-5 分钟（页缓存热）；每 60-90s 轮询一次即可。

### 3.2 测量方案（服务起来后，两个口径均要）

- **口径A（与同事对齐，复现命中态）**：128K×32 前缀、128 请求、并发 32，复现 87.5% 命中，采集 TPS/TTFT。脚本可基于 [baseline/data/verify_multi_161.py](baseline/data/verify_multi_161.py) 改造（端口 8100、模型名 dsv4、32×128K 前缀、并发 32）。
- **口径B（劣化测量，验收口径）**：单前缀 128K×1，池化组 vs **同形态 HBM 组**（w8a8+TP4×DP4+MTP3+1M+prefix-caching——需要补跑一次，见 §1.1 形态差异警告），命中率相当（87.50% vs 85.45%）下 TPS/TTFT 差即纯池化读取代价，**≤5% 验收锚点**。
- 指标：冷灌/命中态 Prefill TPS、命中态 TTFT；劣化% = (基线 − 池化) / 基线。对照 #15442 自测 OFF 146.70 / ON 53.76 / ON+prefetch 55.37 tok/s 验证量级。
- 数据回填 [baseline/report.md](baseline/report.md)、[01_requirements_analysis.md](01_requirements_analysis.md) §2.3 P0（改为"适配已由 PR#15442/#15854 解决，劣化% 实测见 X"）、本文档。

### 3.3 备选项（仅当 165 反复失败）

- 降 dram.size 仍失败 → 换服务器/51 现成环境（栈构建脚本 build_stack_v2.sh 可复用）。
- 向同事索要同形态 TPS/TTFT 及 MTP 采纳率（口径B 的补充对照）。

## 4. 关键文件清单

### 4.1 本地（D:\lzy\project\kv_pool\tmp\，scp 到 165 用）

| 文件 | 用途 |
|---|---|
| `build_stack_v2.sh` | 栈构建（pr15854 基线 + 15442 pick），换服务器可复用 |
| `start_dsv4_layerwise.sh` | **16 卡启动脚本（同事配置 165 适配版，容器内 /home/lizhongyang/tmp/ 已有）** |
| `launch_16c_v2.sh` | 清僵尸+MetaService 检查+启动 一键脚本（容器内） |
| `poll_layerwise.sh` | 启动轮询（容器内，盯 dsv4_layerwise_v2_8100.log） |
| `smoke_layerwise.sh` | 冒烟：2 个相同请求验证 layerwise save/load（容器内） |
| `check_now.sh` | NPU 状态盘点（宿主） |
| `kill_zombie2.sh` / `killall_launch.sh` / `kill_lmt.sh` | 僵尸/抢占清理（宿主，cgroup 归属校验版） |
| `watch_16c_v2.sh` | 120s 自动等卡+启动+冒烟 watcher（当前停用） |
| `first_fatal*.sh` / `root_cause*.sh` / `diag_*.sh` | 本次排查用的日志分析脚本（可复用模式） |

### 4.2 165 远端

- 容器 pool_165 `/home/lizhongyang/tmp/`：上述容器侧脚本 + `start_dsv4_layerwise_8c.sh`（8 卡变体，**DSV4 不可用**，固定开销 10.7G 装不下，留作小模型参考）
- 服务日志：容器 `/home/lizhongyang/map_165/run/dsv4_layerwise_v2_8100.log`（每轮自动备份 `_prev.log` / `_attempt1.log`）；attempt7 的成功里程碑和 HalMemCreate 失败现场都在最新一份里
- HBM 基线资产：`/home/lizhongyang/pool165/`（脚本+数据），对照组继续有效
- s2 参考（本机 memcache layerwise 唯一成功先例）：`/home/lizhongyang/map_165/run/s2_memcache_layerwise/`（DSV2-Lite、device_sdma）

### 4.3 文档

| 文件 | 内容 |
|---|---|
| [07_colleague_test_data.md](07_colleague_test_data.md) | 同事测试数据整理 + 完整启动脚本 + 栈成分/PR 分析 + 命中率闭环 |
| [01_requirements_analysis.md](01_requirements_analysis.md) | 需求与验收口径（§1.4、§2.3 P0、§2.6） |
| [baseline/](baseline/) | HBM 基线脚本与数据（w4a8 TP8 形态） |

## 5. 操作教训（避免重复踩坑，9-16 为本会话新增）

1. **PowerShell ssh 引号嵌套必坏**：复杂命令一律本地写 .sh → `scp` → `docker exec pool_165 bash <脚本>`。内联 `awk -F'/'` 里的单引号必炸。
2. DSV4 权重 151G：页缓存热时 22s，冷时数分钟；就绪轮询给足时间。
3. 杀服务双确认：`pgrep` + `npu-smi info`。
4. memcache 需先拉 MetaService（5000/6000）；**先确认 protocol 与本机库匹配**（host_rdma 需 libhcom.so，165 没有，用 device_sdma）。
5. 池化组必须 `--no-enable-prefix-caching`。
6. `verify_multi.py` 依赖 `PYTHONHASHSEED=0`。
7. `docker exec bash -c "...$b..."` 的 `$b` 宿主展开成空——带变量一律写 .sh。
8. "同事能跑我不能跑"先比栈再比配置；`git ls-remote origin main` 直接比远端，勿信本地 remote-tracking 引用；cherry-pick 从 PR 分支算基线。
9. **`pkill -f "vllm serve"` 杀不到 worker**：vllm worker 用 setproctitle 改名成 `VLLM::Worker_DP*`/`VLLM::EngineCore`，pkill pattern 匹配不上。清理要加 `pkill -9 -f "VLLM::"`，或宿主侧按 npu-smi 进程表 + `/proc/<pid>/cgroup` 归属精确 kill。**崩溃后必查 npu-smi 残留，僵尸占卡会导致下轮启动 HBM 检查失败（attempt 6 教训）**。
10. **`kill $PIDS` 前必须校验 pid 是纯数字**：npu-smi 输出里混着 PCIe 地址 `0000:xx`，正则抓出来 `kill 0000` = kill 进程组 = 自杀（本次 takeover 脚本翻车原因）。
11. **共享服务器启动前先 `npu-smi info` 盘点占用者**：`/proc/<pid>/cgroup` + `docker ps --no-trunc` 定位容器归属；只杀自己的，别人的去协调。
12. **启动失败的错误要分层看**：APIServer 的 `RuntimeError: Engine core initialization failed` 是包装；Gloo `Connection was likely closed` 是次生；TBE `main process disappeared` 是尸体。真正的第一现场用 `grep -nE "ERROR" log | head` + 时间戳最早的那条 C++/Worker 错误。
13. **DSV4 单请求固定 KV 开销 ≈10.7G**（compressor/indexer state，与长度无关；由 1M→17.07G / 128K→11.48G 两点拟合）：8 卡（TP4×DP2，每卡 7.85G KV）连一条请求都装不下，DSV4 池化验证必须 16 卡。
14. **memcache DRAM 池分配失败的排查顺序**：protocol 库依赖（libhcom.so）→ /dev/shm psm_* 残留 → VMM 碎片（reboot）→ per-rank 容量超卖（16×dram.size vs 宿主 RAM，降 dram.size）。
15. **watcher 类后台脚本要加单实例锁 + trap 清锁**（.watch_16c.lock 模式），且探测函数要区分"自己启动的进程"与"别人的"（cgroup 归属）。
16. **vllm_lmt/zhuyixiang 等同事任务随时可能回来抢卡**：每次启动前 check_now.sh，NPU 上有别人的进程先协调再动手。

## 6. 一句话状态

栈 `layerwise_dsv4_v2`（main@0905 + PR#15442 + #15854，UT 363 绿）已就绪且 **5-spec layerwise 布局在 attempt 7 实证通过**（16 worker 全部 `layerwise config: num_layers=44 num_groups=6`）；host_rdma→device_sdma 已修、僵尸清理已修；**165 已 reboot（清 HalMemCreate ret:6 的 VMM 碎片），重启后按 §3.1 runbook 恢复，若 128GB 池分配再挂就降 dram.size=64GB**；服务起来后做口径A（128K×32 复现 87.5%）和口径B（128K×1 劣化%，需补同形态 HBM 基线）——劣化% 数据是唯一核心目标。
