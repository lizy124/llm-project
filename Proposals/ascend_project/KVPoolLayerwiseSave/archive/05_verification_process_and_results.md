# 05 — 验证过程、证据与结论归档

归档日期：2026-10-24，时区 Asia/Shanghai。覆盖已有 10-22～24 材料；最新历史采样截至 10-24 10:24:53。本次只做本地迁移、哈希核对、文本取证和 TSV 重算，没有重新运行推理、压测或 NPU 单测。

## 1. 测试与证据矩阵

| 编号 | 场景、身份与工具 | 对应标准 | 结果与限制 |
|---|---|---|---|
| V1 | run A，129/130 b113，既有日志离线分析 | A1/A6 | 事件超时形态明确，设备侧根因未闭环 |
| V2 | run B，基线 06b685f346，既有 dump/异常 | A1/A2 | 首发 GVA 覆盖断言明确，随后通信级联 |
| V3 | run C，带诊断的 16:59 复现 | A2 | 绝对 `[0,224)` 对 63 项数组，次步 127，机制证据成立 |
| V4 | 止血 v1 两机部署 | A3 | 功能失败：1832 次 UNPROVABLE，正常保存停摆 |
| V5 | 止血 v3，P e69705578 / D 8879bb3c5 | A3 | 报告记载正常保存恢复、真实 WINDOW 拦截、零 GVA fatal；非全模式验收 |
| V6 | 19302-token 同 prompt 两请求 | A3 | 历史报告输出哈希相同，一次命中 16384；原响应 JSON 未取得 |
| V7 | 10-23 031621 aisbench 循环，19 记录 | A5 | 聚合零 GVA fatal；含 9 超时、180 已知请求失败及其他证据限制 |
| V8 | 10-24 d15a81d92 加探针，mock 和真机采样 | A4 | 支持 V2 标记修复；非纯 PR head，缺完整 worker/NPU/HCCL 验收 |
| V9 | 10-24 b118 synchronize | A6 | copy event 报 107020，多流超时；P0 plog 未采集 |
| V10 | 本次本地归档 | 材料可追溯 | 文件迁移 SHA256、脚本全文、链接和统计自检；不计作模型复验 |

对应需求标准见 [01 §5](01_requirements_and_design.md)，版本与方案差异见 [02](02_development_and_pr.md)，工具、命令及环境见 [03](03_verification_scripts.md)、[04](04_verification_env_and_launch.md)。

## 2. 关键过程与原始片段

### 2.1 首先分型

10-22 run A 约 01:38～01:42 发生设备事件等待，发送线程等待保存事件，设备报 234s 超时。run B 的首个软件异常在 09:05:07 由 DP2 的 `build_shared` 抛出 `[0,224)` 覆盖断言；其他组随后才通信/event 超时。不能仅凭尾部线程名或 WorkerProc 退出归为同根因。

归档保留带源路径、源 SHA256 和原始文本的 [run A 片段](evidence/20261022_runA/key_excerpt.txt)、[run B 片段](evidence/20261022_runB/key_excerpt.txt)。详细日志在外置留存区。

### 2.2 run C 把矛盾落实到数组

[原始片段](evidence/20261022_runC/key_excerpt.txt)记录：

```text
16:59:35 builder_group=0 is_save=True range=[0,224)
gva_start=0 gva_end=224 len(block_gvas_np)=63 len(block_ids_np)=63
RuntimeError: ReqMeta GVA metadata does not cover requested block range [0, 224) with offset 0
```

这是实际日志字段摘录。更完整的冻结数组逐值比对由历史分析完成：首步等于 63 项增量，次步追加为 127。run B 的“128=前值64+本步64”被后续逐值复核更正，不能继续用可变列表反推不存在的历史状态。

### 2.3 窗口对齐冻结与止血部署

多轮评审发现准入快照可为空、对象字段混用、窗口连续性及逐组语义未闭合，冻结对齐实现。止血 v1 又实际触发全量 UNPROVABLE 误杀；v3 将 new 分支的完整表直接判 ZERO 后，正常保存恢复，目标窗口被跳过。历史报告先记载真实窗口拦截 23 次，后续长稳是另一统计窗口。

归档保留 [WINDOW 原始片段](evidence/20261023_stop_validation/key_excerpt.txt)。1832 次误杀、10 例状态单测、19302-token 输出哈希一致等来自历史部署/修复报告；未获得相应原始响应和全部单测 stdout，不能把报告摘录冒充完整原始结果。

### 2.4 MRV2 根因修复

恢复先经 new 建完整表，旧 marker 未清使下一步增量误入 preempted，覆盖 tracker。V2 new 分支 discard 修复该生命周期。历史 mock 输出为恢复保存 `[0,32)`、随后 `[32,48)`，而非继承旧游标 16；已提交 keys 会被过滤。

历史 10-24 分析记录 MRV1 8 例通过、旧基线 V2 8 例失败，以及本地变体 mock 的完整表保留；这些是报告层证据。本次未重跑，纯 PR head 工件和真实 worker 边界覆盖不足。历史分析来源见 [02 的追溯入口](02_development_and_pr.md#4-源码快照与历史材料)。

## 3. 长稳原始数据及本次复算

实际窗口：2026-10-23 03:16:21～14:48:24，约 11h32m。计划是 20h，原目录名是 test_10h_logs，三者含义不同。

小体积原件完整复制在 [TSV](evidence/20261023_soak/soak_summary_20261023_031621.tsv)、[心跳](evidence/20261023_soak/soak_heartbeat_20261023_031621.txt)、[AISBench CSV](evidence/20261023_soak/aisbench_result.csv)。CSV 含更早预压测记录，不逐行假设与 19 用例一一对应。

| 轮次 | 用例数 | WINDOW | UNPROVABLE | GVA fatal |
|---|---:|---:|---:|---:|
| r1 | 7 | 393 | 0 | 0 |
| r2 | 7 | 403 | 0 | 0 |
| r3 | 5 | 262 | 0 | 0 |
| 合计 | 19 | 1058 | 0 | 0 |

| 有效性项目 | 本次 TSV 复算 |
|---|---:|
| exit=0 / exit=124 | 10 / 9 |
| 已知 Failed Requests | 180（14+121+45） |
| fail=NA | 2 条，不能按 0 处理 |
| 外部计数负增量 | 1 条，r3c5 |

复算结果完整保留在 [summary.json](evidence/20261023_soak/summary.json)。使用 03 第二部分的只读 PowerShell 脚本，逐行计 exit，分别对 skip_window_d、skip_unprov_d、fatal_d 求和；fail=NA 单列，其他 fail 求和。TSV 源 SHA256 为 `5C988BADDA96B4D6A00A4C9CCF3A550D2DF1086BDC895CE118F6244DD7FA14CF`。

解释边界：WINDOW 是按保存步计，不能当作独立请求数；聚合计数含外部客户端；r3c5 指标负增量，exit=0 也不能证明主测试完整有效。09:52 后的 P 原始日志及 D 日志未完整留存，soak 主日志为 0 字节。由此只能支持“留存窗口持续拦截目标形态，未记录 GVA fatal”，不能写成 20h 全部通过。

## 4. 根因修复采样与设备故障分开

| 本地修复变体采样 | GVA fatal | probe | 其他记录 |
|---|---:|---:|---|
| 10-24 09:57:49 | 0 | 4 | preempt=48，hit=16384 累计558 |
| 10-24 10:24:53 | 0 | 4 | 极端 soak，hitlines=12791 |

这些数值来自历史分析快照，包含外部流量；一次性探针4条不是4次恢复，也不能替代逐请求链路。d15a81d92 加探针的采样不得和止血 v3 的 11h32m 累加成同一版本的长稳结果。

b118 的 [设备异常原文](evidence/20261024_synchronize/key_excerpt.txt)在 03:53:51 进入 `async_utils.py → self.copy_event.synchronize()`，随后 `aclrtSynchronizeEvent` 报107020，多条流234s event超时。[采集错误原件](evidence/20261024_synchronize/collect_errors.txt)明确 P0 plog 缺失。此位置不等于最早 layerwise 线程位置；相同错误码不足以证明与 run A 同根因。

## 5. 证据分层与外置位置

详细日志、本地源码和压缩包存于 [data 任务留存区](../../../../../data/task_docs_retained/20261024/KVPoolLayerwiseSave/README.md)：

| 类别 | 迁移文件数 | 原始字节数 | 目录 |
|---|---:|---:|---|
| 详细日志及其原有伴随材料 | 316 | 407210231 | raw_logs，按 run A/B/C、止血、soak、v4_pro、synchronize 分开 |
| 原压缩包 | 2 | 82176282 | packages，不解压、不改写 |
| 源码快照 | 8 | 751419 | source_snapshots |
| 合计 | 326 | 490137932 | 约467.43 MiB，另有小体积迁移前文本备份 |

保留两组已确认的重复原件：run B 的 output.log 与输出文本（各54336084字节），09:52 P快照的两个副本（各40904377字节）。哈希相同，不重复计作独立证据；未删除任何副本。

[迁移清单](evidence/20261024_local_archive/migration_manifest.json)逐文件登记原位置、现位置、原大小和SHA256；[证据来源](evidence/20261024_local_archive/evidence_sources.json)登记小原件和摘录来源。摘录明确跳过超长dump、截断上限及 StreamReader 行号语义，不能当成完整日志。

## 6. 结论与未完成项

已经完成首发故障分型、累计表契约定位、止血误杀修正、留存长稳统计及 V2 生命周期代码缺陷定位。本阶段验证结论为部分通过。

仍需固定纯修复head，补历史目标请求入口链路、恢复步metadata、同一步抢占恢复、async-load、worker fence、snapshot/partial以及多卡NPU/HCCL；设备event问题单独取证。单个精度样本、mock通过、零GVA计数均不替代这些验收。

本次归档自检结果见 [local_archive_validation.json](evidence/20261024_local_archive/local_archive_validation.json)。归档完成仅表示材料整理、迁移及本地校验完成，不宣称后续补测已经执行。

## 7. 2026-10-26 文档精简

按 agent 部分按需收录规则，03保留配置、启动与测试的20项全文，原6项agent脚本改为必要说明与链接追溯；原配置/测试正文、实际脚本、原始数据及验证结论未改。修订范围、迁移前文档快照和本地核对见 [修订记录](evidence/20261026_agent_section_revision/README.md)。本次没有重跑模型或现场测试。
