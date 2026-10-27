# 05 过程与结果 — 2026-10-13 阶段性归档

**本次没有形成新的有效A/B回退率结论。**计划每组3轮，实际完成1个有效池化轮。第一轮无池化在冷启动阶段被用户要求中止，未开始预热或正式AISBench；其余4轮未执行。环境已让出，验证暂停。

## 有效结果

| 轮次 | Prefill TPS | TTFT avg / P90 | 正式成功/失败 | 四DP预热queries | HBM / external hit |
|---|---:|---|---|---|---|
| 01_pool_1 | 6531.3315 token/s | 20080.9 / 23227.2 ms | 196 / 0 | 每个118046 | 87.45% / 0.00% |

实际平均输入131155.0561 token（包含请求模板），输出1 token；并发49。原始P结果时间2026-10-13 22:05:24，保留服务器时间标签；不能与客户端运行编号秒级等同。冻结数据前后SHA256匹配。定向预热真实P/D各请求均200、D流正常结束且输出1 token，四DP均覆盖；无profiler。

主证据：[逐轮报告](evidence/20261013_pinned_revalidation_214027_retry1/01_pool_1/README.md)、[原始CSV](evidence/20261013_pinned_revalidation_214027_retry1/01_pool_1/raw/aisbench_result.csv)、[原始日志](evidence/20261013_pinned_revalidation_214027_retry1/01_pool_1/raw/aisbench.log)、[预热响应](evidence/20261013_pinned_revalidation_214027_retry1/01_pool_1/raw/pinned_warmup.json)、[指标和有效性](evidence/20261013_pinned_revalidation_214027_retry1/01_pool_1/derived/result.json)、[缓存指标](evidence/20261013_pinned_revalidation_214027_retry1/01_pool_1/derived/cache.json)、[代码身份](evidence/20261013_pinned_revalidation_214027_retry1/01_pool_1/inputs)。服务日志以当前轮启动的.out为准；辅助采证p.log/d.log可能保留前一场景内容，不参与结果判定。

只有1个池化样本，无有效无池化样本，不计算组间中位数回退、离散度或置信界限。交接旧配对池化6153.3989、无池化6802.8062 TPS（9.55%回退）仅作历史参考，未与本次定向预热轮混算；不能用旧基线与本次6531.3315拼接结论。

## 异常和排除样本

| 记录 | 实际事件 | 统计处理 |
|---|---|---|
| [20261013_revalidation_211740](evidence/20261013_revalidation_211740/README.md) | 在包目录跑pip版本采证，platform.py遮蔽标准库，未测试 | 排除，无性能样本 |
| [20261013_revalidation_211740_retry1](evidence/20261013_revalidation_211740_retry1/README.md) | 快照假定代理在scripts根目录，实际在外部路径，未测试 | 排除，无性能样本 |
| [20261013_revalidation_211740_retry2](evidence/20261013_revalidation_211740_retry2/README.md) | 池化6154.7955 TPS、196/196成功，但预热2/1/0/1，DP2缺失 | 无效，排除 |
| [20261013_pinned_revalidation_214027](evidence/20261013_pinned_revalidation_214027/README.md) | P DP1 TransferEngine绑定20700端口遇Address already in use，未发请求 | 排除，无性能样本 |
| [当前02_nopool_1](evidence/20261013_pinned_revalidation_214027_retry1/02_nopool_1/README.md) | 按用户要求在冷启动期间停止 | 未执行正式测试 |

异常证据全部留存。端口绑定失败不能直接归因为池化吞吐回退；定向预热修复后只得到上述1个有效轮，尚不足确认性能结论。首次AISBench属性恢复报不支持权限复制，后来改为内容/链接恢复并核对成功。

## 证据完整性与复算

每个运行inputs留镜像/挂载、组件版本、脚本配置tar快照、commit/分支、runtime.diff、package_SHA256与冻结数据哈希；本次工具源码留初始快照或03正文。原模型及大冻结数据不复制到归档，原位置及哈希足以在现有内网起点取得。未记录镜像构建过程，是从空环境重建的缺口。

完整内网原件：/mnt/share/lizhongyang/dts_zyd_ab/<run_id>/。各evidence_manifest.json记录文件位置、大小、SHA256和是否完整纳入；大日志仅保存首尾片段时明确标注。归档evidence保留原小体积输出，不只链接到可能继续变化的活动目录。原始CSV含先前追加行，取本轮最后一行且用同轮原始日志对照，不能把旧行计入本轮。

复算首轮有效性：python3 analyze_revalidation.py evidence/20261013_pinned_revalidation_214027_retry1/01_pool_1。六轮汇总模式需未来凑齐各组三个有效轮后使用，此次未执行完整汇总。报告的--partial模式只生成有效逐轮事实，不生成回退结论。

## 结束现场

用户因环境被他人使用要求停止。两机lizy_1009测试P/D、proxy、Meta/standalone均已停，无残留测试进程；master、工作区干净；AISBench链接与内容恢复并校验通过。保留既有容器、冻结数据、pyarrow25.0.1及辅助脚本，未清理其他容器。证据：[停止原因](evidence/20261013_pinned_revalidation_214027_retry1/stopped_by_user.txt)、[退出和恢复状态](evidence/20261013_pinned_revalidation_214027_retry1/final_status.txt)、[135进程](evidence/20261013_pinned_revalidation_214027_retry1/raw/final_processes_135.txt)、[133进程](evidence/20261013_pinned_revalidation_214027_retry1/raw/final_processes_133.txt)、[最终分支](evidence/20261013_pinned_revalidation_214027_retry1/final/inputs)。exit=130表示按用户要求中止，P_stop/D_stop/P_master/D_master/AISBench_restore均0。

后续待办：环境可用后，新建运行编号、两组按同一冻结数据及定向四DP预热重新交替运行，每组至少3个有效轮，报告中位数与离散度再判断是否严格超过5%。
