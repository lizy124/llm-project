# PR #17624 池化 Nightly 服务器侧测试指南

| 项目 | 内容 |
|---|---|
| 适用对象 | A3 / A5 服务器上的执行 agent，及 PR 侧操作人 |
| PR | `vllm-project/vllm-ascend#17624`（分支 `pool_nightly` @ fork `lizy124/vllm-ascend`） |
| 提交 | `bf0278b90`（PD / A5 / MTP3 三载体）+ `e37e47e33`（A3 读路径哨兵，gate FAIL 时的 revert 对象），base `main@31608b3f5`，+110/-7，4 个 yaml |
| 设计依据 | 同目录 `pool_nightly_proposal.md`（本指南是其执行视图；语义冲突时以 proposal 为准并回报） |
| 配套件 | 同目录 `r9_gate.sh`（R9 gate 脚本）、`static_check.py`（静态校验） |
| 版本配对 | vLLM main @ `ced6857afa0ea7b2e3f0846a62e1394e90f15607`（PR 已 pin），vllm-ascend = PR 分支 |

---

## 1. 背景

KV 池化（AscendStoreConnector，memcache / mooncake 两种池 backend，layerwise 开关）目前没有任何 nightly 覆盖——池化回归要等到人工 weekly 才会暴露。本 PR 把池化配置叠进 DeepSeek-V4-Flash 家族已有的 4 个 nightly 服务，让既有考题顺带看护池化。

硬约束：不新增 yaml、不新增 test_case、CI 注册与 runner 脚本零改动。所有池化配置一律叠加在现有服务上。

四个载体及其看护分工：

| 载体 | nightly 入口 | 服务器 | 池化叠加 | 看护内容 |
|---|---|---|---|---|
| S1 | `DeepSeek-V4-flash-w8a8-PD-prefix` | a3 双节点 1p1d（a3-800i-16 ×2，multi_node runner） | memcache 池（5000/6000）+ 两端 MooncakeHybrid → MultiConnector（PD 传输保留在 connectors[0]，engine_id 保留；AscendStore producer layerwise=true / consumer false） | 写路径精度（AIME2025 90/10）+ 开销水位（TPOT50 15569.48/0.95，新增） |
| S2 case #1 | `DeepSeek_V4_Flash_A5` | a5 单机 8 卡（single_node runner） | mooncake 池（master 50088 / metrics 50089）+ AscendStore kv_both layerwise=true；仅 case #1，benchmarks 内联解耦 | 写路径精度（GPQA 87.4/5）+ 开销水位（gsm8k 83.50/0.95，新增） |
| S4 | `acceptace_rate_deepseekv4-flash-w8a8-mtp` | a3 单机 TP8（a3-800i-8，single_node runner） | memcache 池 `_kv_pool` anchor（5000/6000，dram 10GB）+ AscendStore kv_both layerwise=false，挂在共享 `_server_cmd`；open/close 两 case 同挂 | MTP3×池交互（接受率 0.925/0.05 + perf 90/0.97，×2 case） |
| S3 | `DeepSeek-V4-Flash-W8A8-A3` | a3 单机 16 卡 DP4×TP4（a3-800i-16，single_node runner） | memcache 池（5000/6000）+ AscendStore kv_both layerwise=true + prefix-caching 显式关闭（池成为唯一命中路径） | 读路径哨兵：GPQA 86.36/5 保留 + perf 换 prefix90 共享前缀负载（占位 baseline 1 / threshold 0.1） |

S3 是全家族唯一的读路径覆盖：其 perf 负载带 ~90% 共享前缀，批量发压的第二波起命中池并回读 KV；其余三个载体的负载无共享前缀，只走查询路径、不走命中路径。

合并前有一道硬门槛：S3 关闭 prefix-caching 后，上游（vLLM v0.30.0 配对）是否仍为请求生成 `block_hashes` 无法静态核实，而池 lookup 两条路径都直接消费它。若不生成，失败形态是哑失效——池从未被读、考题照常全绿。因此必须先在 a3 上跑 R9 gate 拿到正向命中信号，PASS 才允许合并、才允许触发 S3 的 nightly。

PR 当前状态：CI 全绿（DCO / ci-gate / pre-commit / main），mergeable clean，无评审意见。

## 2. 测试项目

| 编号 | 项目 | 执行位置 | 性质 | 判定依据 |
|---|---|---|---|---|
| G0 | 前置预检（NPU / 端口 / 版本配对） | A3、A5 | 门槛 | §4 阶段 0 |
| G1 | R9 gate：prefix-off 下 block_hashes 生成 + 池命中 | A3（a3-800i-16 pod） | 合并前硬门槛 | 请求 #2 日志 `kvpool hit tokens: >0` |
| G2 | S1 PD nightly | CI（a3-800i-16 ×2） | 观察 + 基线 | AIME ≥80；TPOT50 ≥14791.0 |
| G3 | S2#1 A5 nightly | CI（a5） | 观察 + 基线 | GPQA 82.4–92.4；gsm8k 首跑记录 |
| G4 | S4 MTP3 nightly | CI（a3-800i-8） | 观察 + 基线 | 接受率 0.875–0.975；perf ≥87.3 |
| G5 | S3 A3 nightly（读路径哨兵） | CI（a3-800i-16） | 观察 + 哨兵 | GPQA 81.36–91.36；perf 正吞吐；波次 2+ 命中 >0 |
| G6 | 池服务生命周期 | 各机 | 观察 | 每 case 池服务干净起停、无残留 |
| G7 | 哑失效监控 | 各机 | 观察 | 有池化但 `kvpool hit tokens` 恒 0 且无报错 = 异常 |

执行顺序：G0 → G1 →（PASS）→ G2–G6 随 nightly 触发 → G7 贯穿。G1 FAIL 则停止后续，走 §5.3 分支。

## 3. 纪律与红线

- 禁止 `git push`（IRON RULE，内网 remote 也不行）；`git pull` / `git fetch` 允许。revert / 校准提交由 PR 侧操作人推送，服务器 agent 只报告不推。
- 任何材料不得上传公网（云存储 / 网盘 / 公共服务）。
- 不改线上 nightly yaml、CI 注册、runner 脚本——本 PR 的全部改动已在分支上，服务器侧只做验证，不做配置修改。
- 共享服务器纪律：先 `npu-smi info` 查空闲（HBM ~3GB 为空闲，~60GB 为被占），只申明空闲 chip；启动前做端口预检。
- 服务器不能直连公网，需要时走 CCW 代理（squid 3128）；代理只在当前 shell 生效，每个新 ssh / docker exec 会话都要重新 export。
- 遇到本指南未覆盖的异常：报告并停，不要临场发挥回退方案（回退菜单见 §5.3，决策权在操作人）。

## 4. 怎么测

### 阶段 0：前置预检

A3 与 A5 各自执行：

```bash
# 1. NPU 空闲确认（只申明空闲 chip）
npu-smi info

# 2. 端口预检（有输出 = 冲突，先解决再继续）
#    A3: memcache 5000/6000（R9 gate 另需 API 端口 8000）
#    A5: mooncake master 50088 / metrics 50089
ss -ltn | grep -E ':(5000|6000|8000|50088|50089) '

# 3. 版本配对确认（vllm 与 vllm-ascend 强耦合，配错 = 启动即崩）
pip show vllm vllm-ascend | grep -E 'Name|Version'
# vllm-ascend 侧应为 PR 分支内容（含本 PR 的 4 个 yaml 改动）
```

端口规划参照（`common_constraints.md` §3）：mooncake master 固定 50088（50051 被宿主机 OceanStor DTMA 占用，禁用）；memcache MetaService 5000/6000，宿主机占用时改 5001/6001 并回报。

### 阶段 1：R9 gate（A3，合并前硬门槛）

执行位置：a3-800i-16 nightly pod（NPU 侧，装有 `memcache_hybrid` + vllm CLI 的环境）。脚本 `r9_gate.sh` 与本指南同目录，自包含：起池 → 起服务 → 发压 → 判据 → 清理。

```powershell
# 本地（Windows）推送脚本——.sh 脚本推送是合规传输方式
scp <本目录>\r9_gate.sh root@<a3服务器>:/tmp/r9_gate.sh
```

```bash
# 服务器侧：若落点是宿主机、执行环境是容器，先拷入容器
docker cp /tmp/r9_gate.sh <容器名>:/tmp/r9_gate.sh
docker exec <容器名> bash /tmp/r9_gate.sh
# 若 ssh 落点直接是 pod：bash /tmp/r9_gate.sh
```

脚本行为：起 memcache MetaService（5000/6000，S4 `_kv_pool` 同款配置）→ 起 S3 同形态服务（Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp，DP1×TP4，block 128，MTP1，1M ctx；相对 nightly S3 仅 DP4→DP1，保证两条请求落同一 engine）→ 串行发两条共享 ~4.2k token 前缀的请求 → 判定。

时间与预算：服务启动预算 40 分钟（1M ctx TP4 编译慢，`READY_TIMEOUT=2400`），全程约 45–60 分钟。

可调环境量：

| 变量 | 用途 |
|---|---|
| `R9_KEEP=1` | 失败时保留现场（服务与日志留在 `/tmp/r9_gate`） |
| `R9_DEBUG=1` | DEBUG 级日志，输出逐块 hit_check |
| `MODEL=<本地路径>` | pod 禁网，`Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp` 名字解析失败时指向本地模型路径 |
| `PREFIX_CHARS` / `READY_TIMEOUT` | 前缀长度（默认 17000 字符 ≈4.2k token）/ 就绪超时 |

判定（脚本自动输出，agent 原样采集回报）：

- `[R9-GATE] PASS — … hit tokens=N`（N>0，期望 ≈4000，即共享前缀对齐块数）→ 进入阶段 2
- `[R9-GATE] FAIL — 请求#2 hit tokens=0 且无报错` → 哑失效成立，停止后续，报告并附判定行与日志尾；处置由操作人决策（revert `e37e47e33` 或 §5.3 R9 回退菜单）
- 其他 FAIL（端口占用 / MetaService 未就绪 / 服务提前退出 / 请求失败）→ 属环境问题，按报错提示处理后重跑，不计入 R9 结论

### 阶段 2：触发 nightly（PR 侧，R9 PASS 之后）

触发前在对应共享 runner 上复核 5000/6000/50088/50089 空闲，然后在 PR #17624 评论区逐条发：

```
/nightly DeepSeek-V4-flash-w8a8-PD-prefix
/nightly DeepSeek_V4_Flash_A5
/nightly acceptace_rate_deepseekv4-flash-w8a8-mtp
/nightly DeepSeek-V4-Flash-W8A8-A3
```

nightly 执行由 CI runner 承担（多机入口走 multi_node runner，单机入口走 single_node runner），服务器 agent 不手工跑 pytest 复现——手工复现缺少 runner 的模型/数据集缓存与调度语境，结果不可比。

时长预期：S1 的 TPOT50 压测 4096 条，耗时最长（已知代价，见 proposal R3）；A5 GPQA `max_out_len` 131072，同属长时项。四个入口互相独立，可并行触发。

### 阶段 3：运行期观察与采集（A3 / A5 agent）

每个 case 生命周期内采集三类证据：

1. 池命中日志（核心观测点）。INFO 级逐请求日志，签名固定：

```
Reqid: <id>, Total tokens <n>, kvpool hit tokens: <n>, need to load: <n>
```

```bash
grep -E 'kvpool hit tokens' <服务日志>
```

预期分布：
- S1 / S2#1 / S4：负载无共享前缀，`kvpool hit tokens` 应恒为 0——这里 0 是正常的（查询路径被走到、命中路径本就不该触发）
- S3：第一波（冷池）为 0，第二波起应 >0（perf 写满共享前缀后的池读收益）

2. 池服务生命周期。每 case 起停各一次：
- memcache：MetaService 日志无异常退出、无 OOM；case 结束后 5000/6000 释放（`ss -ltn` 复核）
- mooncake（A5）：mooncake-master.log 正常启动；关注 `global_segment_size=1GB` 是否出现容量不足告警（proposal R5，首跑看日志按需调）

3. 考题结果。aisbench 输出的每门 benchmark 实测值（精度分 / 吞吐），逐门采集，供阶段 4 回填与评估。

### 阶段 4：校准回填与复验

首跑绿后，把记录式占位与待校准基线以实测值回填（由 PR 侧提交校准 commit，服务器 agent 提供实测数字）：

| 条目 | 当前值 | 回填动作 |
|---|---|---|
| S2#1 perf gsm8k | 83.50/0.95 | 首跑实测核对，偏差大则回填实测 |
| S4 perf | 90/0.97 | 实测回填 |
| S3 perf | baseline 1 / threshold 0.1（占位） | 回填实测吞吐，threshold 恢复 0.97 |

校准 commit 推送后重新触发对应 `/nightly` 复验，全绿方可合并。

## 5. 怎么评估

### 5.1 逐项通过判据

判卷语义：accuracy 为 |实测 − baseline| ≤ threshold；performance 为 实测吞吐 ≥ threshold × baseline。

| 项 | 判据 | 数值展开 |
|---|---|---|
| G1 R9 gate | 请求 #2 `kvpool hit tokens: N`，N > 0 | 期望 ≈4000 |
| S1 AIME2025 | \|x − 90\| ≤ 10 | 80–100 |
| S1 TPOT50 | x ≥ 0.95 × 15569.48 | ≥14791.0 |
| S2#1 GPQA | \|x − 87.4\| ≤ 5 | 82.4–92.4 |
| S2#1 gsm8k | x ≥ 0.95 × 83.50 | ≥79.33（首跑以记录为主，池化开销致红则走回填） |
| S4 接受率 ×2 | \|x − 0.925\| ≤ 0.05 | 0.875–0.975 |
| S4 perf ×2 | x ≥ 0.97 × 90 | ≥87.3（首跑回填） |
| S3 GPQA | \|x − 86.36\| ≤ 5 | 81.36–91.36 |
| S3 perf | x ≥ 0.1 × 1 | 正吞吐即过（占位语义，首跑记录） |
| S3 命中观测 | perf 波次 2+ `kvpool hit tokens` > 0 | 观测项，非判卷项，但恒 0 触发 §5.2 |
| 池生命周期 | 每 case 干净起停，端口释放 | 各机 |

S1 TPOT50 偏离 15569.48 超 5% 时的怀疑顺序（proposal Q1 观察协议）：① prefix×写池交互（可切 `--no-enable-prefix-caching` 复跑验证）→ ② MultiConnector 子条目 engine_id → ③ 硬件环境差异。

### 5.2 哑失效规则

S3 的考题全绿不等于读路径活着。若 S3 日志中 `kvpool hit tokens` 恒为 0 且服务无任何报错，说明池从未被读（大概率是 prefix-off 下 block_hashes 未生成），读路径覆盖实际为零——此时即使三门考题全绿也判 FAIL，处置同 R9 FAIL。这条规则是 S3 哨兵语义的一部分，评估时与判卷结果并列看。

### 5.3 失败处置路由

回退动作均需操作人确认后由 PR 侧执行，服务器 agent 负责采集证据与复现环境。

| 失败形态 | 第一怀疑点 | 回退 / 缓解 |
|---|---|---|
| R9 gate FAIL（hit=0 无报错） | prefix-off 下 block_hashes 未生成 | revert `e37e47e33`；或回退菜单 ① 保 prefix + 超量负载制造 HBM 淘汰 ② 退回浅叠（只保写路径） |
| S1 PD 握手 / 路由异常 | MultiConnector 子条目 engine_id（上游是否逐子条目转发未证） | 删除子条目 engine_id、对齐 weekly 先例 |
| S1 TPOT50 偏差 >5% | 见 §5.1 怀疑顺序 | 逐级排除后按实测重标 |
| S2#1 起服务失败 | mooncake + kv_both + layerwise + fp8 KV 无先例叠加 | 阶梯回退：① layerwise 改 false → ② backend 改 memcache → ③ fp8 解档（case #1 内联 server_cmd，`--kv-cache-dtype` 改 auto）。故障只波及 case #1 |
| S2#1 池容量告警 | `global_segment_size=1GB` 不足 | 看 mooncake-master.log 调大 |
| S4 接受率掉线 | 池化变量引入（nightly 内无非池化对照） | 归因动作：本地移池对照跑一次（去掉 kv_pool + kv-transfer-config 即恢复纯哨兵），对比定位 |
| S3 考题红但命中正常 | 按普通精度/性能回归排查 | 与池化无关则走常规流程 |

### 5.4 合并条件

全部满足才可合并：

1. R9 gate PASS
2. 四个 nightly 入口判卷全绿（按 §5.1）
3. S3 `kvpool hit tokens` 波次 2+ >0（哑失效规则不触发）
4. 池服务生命周期干净（G6）
5. 占位基线已回填实测并复验绿（阶段 4）

合并后：非池化参考水位的消失是方案已确认的代价（proposal R7），无需另行处理。

## 6. 报告格式

服务器 agent 每轮执行后按此结构回报，数字与日志行原样粘贴，不做解读加工：

```
## 执行报告 <日期> <服务器>
### G1 R9 gate
- 结论：PASS / FAIL / 环境问题
- 判定行：<kvpool hit tokens 原始日志行>
- 异常（如有）：<报错摘录>

### G2–G5 nightly（按入口分节）
- 入口：<nightly 名>
- 判卷：<每门 benchmark 实测值 vs 判据，过/红>
- 命中日志：<grep 结果摘录，标注波次>
- 池生命周期：起停是否正常、端口是否释放

### 实测值汇总（供回填）
- S2#1 gsm8k：<值>
- S4 perf：<值> ×2
- S3 perf：<值>

### 需要决策的事项
- <如有，列出并停在此处>
```

## 7. 附录

### 7.1 grep 签名速查

| 签名 | 含义 | 级别 |
|---|---|---|
| `kvpool hit tokens: <n>, need to load: <n>` | 逐请求池命中（判据信号，`pool_scheduler.py:715-721`） | INFO |
| `hit_check: req=... no participating groups` | block_key layerwise 路径签名（S3 不走此路径，仅 R9_DEBUG 排障参考） | DEBUG |

### 7.2 端口速查

| 用途 | 端口 | 备注 |
|---|---|---|
| memcache MetaService / config store | 5000 / 6000 | 占用时改 5001/6001 并回报 |
| mooncake master RPC | 50088 | 50051 被宿主机 DTMA 占用，禁用 |
| mooncake metrics | 50089 | nightly/template 语境值（共享验证服务器语境纪律值为 9008，两者语境不同均正确，勿互改） |
| R9 gate vLLM API | 8000 | 脚本默认，可经 `API_PORT` 改 |

### 7.3 文件速查

| 用途 | 路径 |
|---|---|
| 四个改动 yaml | `tests/e2e/nightly/multi_node/external_dp/config/DeepSeek-V4-flash-w8a8-PD-prefix.yaml`；`tests/e2e/nightly/single_node/models/configs/` 下 `DeepSeek_V4_Flash_A5.yaml`、`acceptace_rate_deepseekv4-flash-w8a8-mtp.yaml`、`DeepSeek-V4-Flash-W8A8-A3.yaml` |
| 池 lookup / 命中日志 | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` |
| kv_pool 配置解析 | `tests/e2e/common/kv_pool/config.py` |
| 单机 / 多机 runner | `tests/e2e/nightly/single_node/models/scripts/test_single_node.py`；`tests/e2e/nightly/multi_node/external_dp/`（runner 按 yaml 序串行跑同 case 的 benchmarks） |
| weekly 先例 | `tests/e2e/weekly/multi_node/external_dp/config/DSv4-Flash-Memcache-PD-layerwise.yaml`；`tests/e2e/weekly/single_node/configs/Qwen3.5-397B-Memcache-perf.yaml` |

### 7.4 配置地雷

`kv_load_failure_policy` 保持默认 `fail`（响亮失败）是有意为之：池读失败必须暴露，静默 recompute 会掩盖读路径 bug。template.md 示例中的 `kv_load_failure_policy: "recompute"` 不可照抄——hybrid 模型（DSV4-Flash）配 recompute 会被 `platform.py` 断言直接拒绝，服务起不来。各载体 yaml 省略该字段即默认 fail，即正确语义。
