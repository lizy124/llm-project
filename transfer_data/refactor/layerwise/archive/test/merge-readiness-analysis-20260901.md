# PR #15367 合入就绪性论证报告

- 分析日期:2026-09-01(晚)
- 分析对象:vllm-project/vllm-ascend PR #15367,head `63be9e03b`(rebase 到 upstream/main `72a988f9d` 后,12 commits,+523/-120,16 文件)
- 问题:现有验证证据是否足以支撑合入
- **结论:证据链已实质充分,合入的技术前提全部满足;剩余缺口均为流程性而非验证性,建议立即走完收尾动作后合入**

---

## 1. 分析框架

本项目建立了六层验证体系(见 §7 成熟度评估),判据分两类:

- **验证性判据(本报告核心)**:改动是否行为正确、零回归、被审阅。此报告论证"够不够"。
- **流程性判据**:合并队列的硬门槛(CI 全绿、reviewer 请求清空、文档 CI)。这些不证明正确性,只放行。

以下按验证性四层逐一评估,再讨论残余风险。

## 2. UT 层:通过,但属"行为保持"型证据

| 事实 | 说明 |
|---|---|
| 规模 | 314 passed(含返工轮新增 7 个断言/场景),服务器 refactor_165 全量 `tests/ut/distributed/ascend_store/` 实跑,两份 venv 指纹入档 |
| 性质 | 返工 PR 的 UT 是**行为保持(behavior-preserving)**断言:key 格式逐字节快照(`test_key_string_snapshot_identical_to_pre_move`)、三方一致性(`test_protocol_functions_store_overrides_and_registry_marker_agree`)、真值表 6 场景、not-found 打点和外部 waiter 单测等。它证明"重构前后行为不变",不证明"行为本身正确" |
| 评价 | 对重构类 PR 是恰当的证据类型。局限:纯 UT 无法证明 gva 传输、PD 链路在真实 NPU 上的非零激活——这正是 e2e 层的职责,见 §3 |

## 3. e2e 层:双轮互证,证据充分且超出常规要求

### 3.1 两轮验证构成"行为不变"的强互证

| 轮次 | 代码版本 | 结果 | 关系 |
|---|---|---|---|
| 首轮 `e2e-report-20260901.md` | `2a239d18a`(返工前,7 commits) | 三场景全 PASS | 基线 |
| 复测轮 `e2e-report-20260901-rebase.md` | `63be9e03b`(返工 4 commits + mypy 修复 + rebase 后,12 commits) | 三场景全 PASS | 与基线逐维对比一致 |

两轮之间恰好夹着全部"风险最高"的变更(检视返工的 5 个设计决策 + rebase 带入 upstream #15364 的 KV Pool 改动)。两轮的同构 PASS 证明:返工和 rebase 均未引入行为漂移。这是比"单轮 PASS"更强的论证形态。

### 3.2 三场景覆盖矩阵与验收标准达成

| 场景 | 覆盖的代码路径 | 验收标准 | 实测(复测轮) | 判定 |
|---|---|---|---|---|
| S1 MultiConnector PD | **P1 新增面**:AscendMultiConnector、waiter 下沉、gva assert 删除后的线程入口、双 connector 顺序 | 成功率 100%;无 AttributeError | 5/5(含 2 条 GSM8K-lite 真实问题);P/D 零 AttributeError;AscendMultiConnector ×5;D 侧 27 层 LayerMetadata 完整 | PASS |
| S2 memcache layerwise | **协议迁移的核心面**:make_* 函数、registry opt-in、hit_check/load_gvas 打点、三维证据链 | hit_tokens>0;valid_gvas>0 | hit_tokens=3328;valid_gvas=26、lease_fail=0;MetaService alloc=28/stored=28/query=130 | PASS |
| S3 mooncake 非 layerwise | **零回归面**:通用路径在加 gate 后是否对非 layerwise 后端透明 | 与 main 基线一致;无 layerwise 标记 | 无 load_gvas/hit_check 任何行;master 三维 939.5MB/112 keys/4 clients | PASS |

关键论证点:

- **S2 的 `hit_tokens=3328>0` + `valid_gvas=26>0`** 直接证伪了重构最隐蔽的失败模式——"key 格式在迁移中静默改变,导致 hit_check 永远 miss 但无报错"。key 快照 UT(§2)与 e2e hit 实测(§3)构成双重锚定。
- **S3 的 layerwise 标记缺席**是"通用层中性化"最直接的零回归证据:非 layerwise 后端看不到任何返工痕迹,等价于 gate 对 mooncake 完全透明。
- **S1 的 27 层 LayerMetadata 完整到达 P 侧**覆盖了返工中被质疑最深的 assert 删除路径(线程后端无关化)——该路径此前由 `isinstance` assert 保护,删除后由 e2e 的 PD 链路完整性替代论证。

### 3.3 数值浮动已排除缺陷解释

两轮间 hit_tokens 3456→3328、load_gvas 行数 28→16 的差异,报告中已论证为滑动窗口命中块数/请求分布的时序浮动(判据均为 ">0" 而非定值),且 completion 长度、elapsed 为观察项非判据。此解释与 S3 的 external hits=3328(与 S2 相同,因同一前缀)自洽,不构成疑点。

## 4. CI 层:全量绿,跨硬件矩阵覆盖

`gh pr checks 15367`(run 33496963848,head `63be9e03b`):

- **核心门禁**:ci-gate ✅、pre-commit ✅、main ✅、Recommend tests from coverage ✅、select-tests ✅
- **跨硬件 e2e 矩阵 18 项全 PASS**:310p-1、310p-4、a2-1×5、a3-16、a3-2×2、a3-4×2、a3-8、a3-800i-2×2、a3-800i-4×2、cpu-0、e2e-upstream_pr(py3.12,910b)。最长 1h13m,无超时失败
- **历史 mypy 问题已闭环**:第一轮 mypy 3.10 报 6 错(`backend_map` 布尔标记使条目 join 成 object)→ 显式注解修复 → rebase 后 head 的 CI 中 mypy 已含于 pre-commit/main 等绿项中

唯一非绿项:readthedocs vllm-ascend / vllm-ascend-cn 两个 docs 构建 fail(见 §6.1)。

## 5. Review 层:实质性意见已全部闭环

- **Pz1116(该模块作者/主审)**:7 条行内意见(2026-09-01 06:48)→ 返工 4 commits 逐条对应 → **APPROVED(2026-09-01 08:52)**。返工后又发生 rebase + mypy 修复 + e2e 复测,均为 behavior-preserving 或验证性动作,未引入新的待审设计决策。
- **gemini-code-assist**:2 条建议(assert 改 TypeError、waiter 设置顺序)→ 分别由"删 assert"和"waiter 顺序加固"消解,已在检视回复中逐条说明。
- **requested_reviewers 剩余**:LCAIZJ(1 人,未响应)。见 §6.2。

## 6. 剩余缺口分析(均为流程性,非验证性)

### 6.1 readthedocs 构建 fail — 需确认与 PR 无关后忽略

两个 docs 项目(build 34329670/34329671)构建失败。初步判断:PR 未触碰 docs/、mkdocs 配置或 docstring 结构,失败大概率为 docs 侧基础设施问题(main 上同期 PR 常见)。**建议动作**:打开 build 日志确认失败原因与 PR 无关(若是依赖拉取/主题问题),在 PR 留言说明即可,不阻塞合入。若意外与 PR 相关,修复成本也限于文档层。

### 6.2 LCAIZJ review 请求未清空 — 需决定等 or 走

Pz1116 已 approve,但 requested_reviewers 里 LCAIZJ 未响应。两个选项:

1. 若 LCAIZJ 是例行 owner 审查,等待或 ping;Pz1116 的 approve 已覆盖该模块的专业判断。
2. 若合并策略允许,维护者可在 CI 绿 + 已有 approve 下直接入队。

无论哪条,这不是验证缺口——没有证据表明 LCAIZJ 掌握 Pz1116 未覆盖的验证维度。

### 6.3(信息项)rebase 后 Pz1116 的 approve 技术上仍有效

rebase 属 force-push,但内容等价(无冲突),GitHub 不自动撤销 approve。若项目要求 rebase 后重新 ack,一条"rebased onto latest main, no conflicts, e2e re-verified"的简短留言即可,非实质工作。

## 7. 验证体系成熟度评估

| 层 | 工具 | 本 PR 状态 | 对"支撑合入"的贡献 |
|---|---|---|---|
| L1 UT | pytest + Mock,314 用例 | ✅ 全过(行为保持型) | 证明重构前后等价 |
| L2 组件/集成 | CI 推荐测试 + select-tests | ✅ 全过 | 证明与周边模块兼容 |
| L3 系统 e2e | 三场景 8 卡拓扑,双轮 | ✅ 双 PASS 互证 | 证明真实路径非零激活 + 零回归 |
| L4 静态检查 | pre-commit、main、mypy(修复后) | ✅ 全绿 | 风格/类型合规 |
| L5 跨硬件矩阵 | 310p/a2/a3/a3-800i/cpu ×18 | ✅ 全过 | 排除硬件特有风险 |
| L6 Review | Pz1116 7 条 + gemini 2 条 | ✅ 全部闭环 + APPROVED | 设计决策被模块作者认可 |

对照社区常规 bar:**多数 vllm-ascend PR 的合入证据 = CI 绿 + 1 个 approve + 至多一轮 e2e**。本 PR 超出该 bar 的部分:双轮 e2e 互证、三场景验收标准量化(>0 判据)、行为保持型 UT 快照、rebase 后复测。证据充分度处于"高于常规"区间。

## 8. 结论与建议

**验证充分性:充分。** 改动面(prod +240/-74,test +283/-46)的每一条高风险路径——协议迁移、gate 下沉、assert 删除、通用层中性化——都有 UT 或 e2e 的直接锚定证据,且返工与 rebase 两个高风险动作被双轮 e2e 夹逼验证为零行为漂移。未发现任何"已知未覆盖"的验证维度。

**合入动作建议(按序):**

1. 查 readthedocs 两个 build 日志,确认与 PR 无关后在 PR 留言说明(5 分钟);
2. 在 PR 留言:rebase 说明 + 复测结论(链接 `e2e-report-20260901-rebase.md` 要点),@Pz1116 可选择性 re-ack;
3. ping LCAIZJ 或请维护者基于 Pz1116 approve + CI 绿直接入队;
4. 入队前确认 merge queue 对 requested_reviewers 的硬性要求(部分配置会阻塞)。

完成 1–2 后即可推进,3–4 取决于仓库合并策略,不需要补充任何代码或测试。
