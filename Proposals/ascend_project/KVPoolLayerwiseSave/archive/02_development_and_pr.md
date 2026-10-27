# 02 — 开发、版本与 PR 归档

日期：2026-10-24。以下 PR 与部署身份来自留存快照，本次没有查询在线状态、提交或推送代码。不能据此判断当前 PR 已合并或当前服务器仍运行某个版本。

## 1. 版本身份

| 对象 | 留存身份 | 使用边界 |
|---|---|---|
| 早期现场基线 | `releases/v0.30.0 @ 06b685f346ad525027d03cffe80d58a491a13081` | run A/B 的基线；run C 另有诊断插桩 |
| 本地 main 比较对象 | `91db8b306` | 与现场 `kv_transfer.py` 曾有 39 hunk、+221/−112，不能直接覆盖 |
| 现场已有隔离修复 | #19090 `b411aca25`，对应 main #19101 `2581a69bc` | 本次分析确认 GVA 断言位于该线程隔离链路之外 |
| 诊断修订 | `dev_test @ 346f74fda` | 修订生命周期日志与字段来源；不等于正式根因修复 |
| 冻结实现 | `de0375537`；附加打点 `d2c9a5267` | 窗口对齐实现及打点被冻结；不得当作已验收版本 |
| 止血 v1 本地部署 | P `2c181701d` / D `4558bb668` | 10-23 首次部署，随后触发全量 UNPROVABLE 误杀 |
| 错误尝试 | `8489d7afa` | 取用不存在的 Request.block_ids，记录已回滚 |
| 止血 v3 本地部署 | P `e69705578` / D `8879bb3c5` | new 分支直接判 ZERO，供 10-23 短期及长稳记录使用 |
| #19533 后续报告 head | `883c8031e`（更早快照为 `544121ff`） | 报告记载与早期部署说明有时间差，不能混用 |
| MRV2 本地变体 | `d15a81d92` 加一次性探针 | 10-24 分析及采样对象，非未经修改的 PR head |
| vLLM 精确配对 SHA / 镜像 digest | 未完整留存 | 不宣称任意环境可完全重建 |

## 2. PR 与改动面

| PR | 目标与关键文件 | 状态以本地快照为限 |
|---|---|---|
| [#19517](https://github.com/vllm-project/vllm-ascend/pull/19517) | 窗口 base 对齐，涉及 scheduler、metadata、worker、transfer | 概览记录 base 为 releases/v0.30.0；相关早期实现被冻结 |
| [#19533](https://github.com/vllm-project/vllm-ascend/pull/19533) | 调度侧三态与 metadata 保存闸门；worker 断言保留 | base 为 releases/v0.30.0；有止血部署、误杀修正和验证记录 |
| [#19567](https://github.com/vllm-project/vllm-ascend/pull/19567) | V2 `_process_new_request` 清理 `_preempted_req_ids` | 精确 PR head/目标分支未在本次单独固定，现场基线与本地变体身份见上表 |

### 2.1 诊断而非行为修复

早期诊断在 `build_shared` 覆盖失败时分段记录 range、group、数组来源及长度，后续增加 tracker 创建、更新和准入探针。已有诊断补丁见 [插桩 patch](../02_development_and_pr/patches/kv_transfer_gva_coverage_instrumentation.patch)，它不是最终 MRV2 修复补丁。

### 2.2 止血的实现与误杀修正

`pool_scheduler.py` 将表状态附到 tracker；`metadata.py` 在生成保存元数据前，对 WINDOW/UNPROVABLE 设置 skip_save。正常 ZERO 路径继续旧保存逻辑，load 请求保留其必要元数据。

v1 的参照表与 new 表不匹配，导致误杀。v3 改在 new 分支直接信任 `NewRequestData.block_ids`，并刷新参照；preempted 定义法拦截、running 状态继承保留。本地 UNTRACKED（None）语义与早期上游版本存在差异，复用前必须核对。

保留的 [kvpool_stop.diff](../02_development_and_pr/patches/stop_v1/kvpool_stop.diff) 与 [pr19533.diff](../02_development_and_pr/patches/stop_v1/pr19533.diff) 是历史快照，不能宣称它们就是 v3 最终补丁或 #19567 的纯 PR diff。最终纯 head 工件不足属于复现缺口。

### 2.3 MRV2 根因修复

关键行为示意如下，不冒充已固定的完整 PR patch：

```diff
 def _process_new_request(...):
+    if self.vllm_config.use_v2_model_runner:
+        self._preempted_req_ids.discard(request.req_id)
```

`build_connector_meta` 先登记 preempted IDs，再处理 new，最后处理 cached。V2 恢复从 new 完成后消费标记，下一步 cached 不会误入 preempted handler，而是走 running 追加路径。MRV1 保持原有行为。

恢复后的新 tracker 保存游标起始为 0；mock 记录恢复范围 `[0,32)`、随后 `[32,48)`。已提交 keys 由 `_pool_keys_missing` 过滤，GVA 数组按完整表长度填充，未传输位置为 0，valid mask 过滤。本归档不把“恢复完整表”误写成“直接继承旧保存游标”，也不把 metadata 游标增长当成写入成功回执。

## 3. 取证、评审与实现边界

run C 冻结数组首步 63 项等于当步增量，次步 127 项等于前表加下一增量；现场 V2 堆栈与具体缺陷路径吻合。但历史恢复入口 marker 和覆盖前 tracker 未完整留存。

需继续覆盖 released 请求和在途保存的隔离、snapshot/partial、async-load、同一步抢占恢复、多卡 NPU/HCCL。一次性 `logger.info_once` 4 条只表示进程内分支命中，不表示 4 次恢复。

## 4. 源码快照与历史材料

完整源码快照已迁到本地 [source_snapshots](../../../../../data/task_docs_retained/20261024/KVPoolLayerwiseSave/source_snapshots/)，包括 `_fix_impl`、`_probe_impl` 和 `_stop_impl/pristine` 共 8 文件；它们保留原版本语义，不代表最终 MRV2 head。

多轮评审、错误尝试和部署说明归入 [开发历史目录](../02_development_and_pr/history/check_and_review/)。精确迁移前路径、大小与 SHA256 见 [迁移清单](evidence/20261024_local_archive/migration_manifest.json)。归档核心结论可独立阅读，历史材料仅用于追溯。

## 5. 编译、部署与回退边界

相关修复为 Python，原记录使用 py_compile 验证；运行服务需重启已加载模块的 EngineCore/worker 才生效。不能仅凭 editable 安装或文件变化判断旧进程已加载新代码。

本次只进行本地归档和脚本静态整理，没有执行打补丁、编译 NPU 算子、重启、清理服务器或推送。原有打补丁脚本保留为历史辅助工具，不能未经版本核对用于当前共享环境。服务准备与停止见 [04](04_verification_env_and_launch.md)，验证事实见 [05](05_verification_process_and_results.md)。

## 6. 历史辅助工具

2026-10-26 按 agent 部分按需收录规则，将原03的辅助脚本全文改为链接追溯：

- [原 A1：patch_new_branch_zero.py](../03_verification_scripts/agent/patch_new_branch_zero.py)是止血v3的一次性源码补丁，用于 new 分支直接判 ZERO 并刷新参照表；要求旧文本恰好匹配一次，不作为任意当前源码的通用修复入口。验证应以已固定的源码版本为起点。
- [原 A2：add_probe_readmit_bugfix_mrv2.sh](../03_verification_scripts/agent/add_probe_readmit_bugfix_mrv2.sh)是历史探针注入工具，其观测位置、日志语义及还原源码的影响见 [03 诊断探针](03_verification_scripts.md#agent-probe)。
- 原 A3/A4 的健康检查要求见 [03 健康检查](03_verification_scripts.md#agent-health)。原 A5/A6 的材料迁移、选证工具见 [文档修订记录](evidence/20261026_agent_section_revision/README.md)，不参与产品源码修复或功能验证。

原脚本未删除、未修改；原身份和SHA256见 [历史辅助脚本清单](evidence/20261024_local_archive/historical_agent_scripts.json)。
