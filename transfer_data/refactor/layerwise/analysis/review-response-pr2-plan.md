# 第一次审核独立分析 与 二 PR 执行方案（v2）

> 输入：[review-round-1.md](review-round-1.md)（外部审核意见）
> 处理原则：逐条独立核实，不轻信——本次核实中既证实了审核的多数断言，也**证伪了其中一处结论**，并发现审核自身的一处事实失真
> 决策前提（用户裁定）：**提 2 个 PR**；**不等待 #12854，抢先合入，由后续 PR rebase**——"直管做正确的重构"
> 本文取代 [p0_modification_plan.md](p0_modification_plan.md) §4 与 [p1_p2_modification_plan.md](p1_p2_modification_plan.md) §4 的 PR 拆分章节；两份方案的问题级修改细节（怎么改）仍然有效

---

## 1. 审核意见逐条独立核实

| # | 审核断言 | 我的独立验证方法 | 判定 |
|---|---|---|---|
| 1 | 【重要】PR3 与 #12854 冲突被低估：#12854 重做 `_alloc_gvas_for_save`/`_prepare_load_gvas` 同区域，两方案存在设计竞争（LayerwiseTransferPreparer vs GVASession） | 抓取 PR #12854 commits 页：存在 commit "Centralize GVA resolution, lease ownership, vectorized transfer-array construction, and multi-group layerwise scheduling in dedicated components... roll back failed leases"——与 PR3 迁移区域确实重叠 | **事实属实，采纳**。但应对策略不采纳（见 §3：用户裁定抢先合入，且有独立于审核的工程理由） |
| 2 | 【中】"平移式迁移"名不副实：实际引入 ≥5 新类型 + 2 工厂，是重新封装 | 属实——GVAKeyFactory/GVASession/GVAHitChecker/GVALayerwiseCapable/GVALayerwiseThreadContext 确为新类型 | **采纳**。PR 描述按"行为保持的重新封装（re-encapsulation）"表述：搬运部分逐字符一致（可机械校验），新增挂接代码 ~150 行需人工审查。不再声称"纯移动" |
| 3 | 【小】P0 §2.3 表格事实错误：mooncake/yuanrong 中不存在 5 个 NotImplementedError 存根 | grep 验证：`NotImplementedError` 仅存在于 base.py:36-48（另 mooncake_backend.py:68 为无关的协议不支持报错）；子类仅**继承** base 存根 | **属实，采纳**。已修正 p0 方案 §2.3 与目录树：两后端**零改动**，删存根动作只发生在 base.py |
| 4 | 【小】PR4×#14697 协调缺口：#14697 计划删除 `init_store`，`on_worker_ready` 等价性论证前提可能消失 | 拉取 #14697.diff 验证：**确实删除 `init_store`**（审核事实正确）；**但结论错误**——LIFE 设计 hook 的是 `ensure_initialized`（memcache_backend.py:62-72），#14654 未触碰该方法（`batch_alloc`/`put` 仍在调用它），pending-buffer 补注册机制完整保留 | **半真：事实采纳、结论证伪**。等价性前提不消失。另审核漏了两点：① #14697 作者是用户本人，协调成本为零；② #14697 **保留**了 `use_gva_layerwise` 双处定义（diff 可见仅重排该行），即 P0-1 在 #14697 之后仍然必要。已顺带修正 p1/p2 方案 §2.2 措辞（`init_store` → `ensure_initialized`） |
| 5 | 【微】精度瑕疵：P0-1 标题"重复定义 4 处"夸大（实际 2 定义 + 1 判断 + 1 引用）等 | P0-1 标题已精确化（"散布 4 处（2 处定义 + 1 处判断 + 1 处消费引用）"） | **部分采纳**。同时反向指出：审核附表声称我方文档记"connector 文件 283 行"——grep 三份文档零命中，该行系审核自设靶（我方从未记录 connector 行数）。审核自身亦有精度问题 |
| 6 | 【微】PR6 部分超需求 4 范围：`check_all_layers_exists` 迁 metadata 非 backend 收敛 | 与我方 P1/P2 §0 修正 2 的裁定一致（gate 为 `use_layerwise`，key-mode 三后端通用） | **采纳并加码**：从执行方案中**砍掉**（不进任何 PR，不作为"顺带清理"）。docstring 中性化保留（那是后端痕迹清理，在需求 4 范围内） |
| 附 | #14465 提交信息声称该属性 "stored but never read"（加强 P0-1 动机） | commit 页仅返回 diff，提交信息未取到；浅克隆无历史 | **未核实**，不影响任何决策（仅动机叙述，PR 描述中不引用该细节） |

**净结论**：审核的核实工作是扎实的（9 项"核实为真"断言我抽查了其中关键 5 项全部属实），问题清单 6 项中 **4 项采纳、1 项半采纳（事实对结论错）、1 项部分采纳**。但审核的执行建议（等待 #12854）与用户策略相反，且其在 #14697 问题上的影响评估有误——均已在本文修正。

---

## 2. 二 PR 执行方案

### PR-A：协议收敛 + 接口分层 + main 回归修复

**主题**：GVA 协议与后端能力判定收敛进 backend（新增 `backend/gva_protocol.py`）

**内容**（细节引用原方案，不重抄）：

| 模块 | 原方案章节 | 内容 |
|---|---|---|
| CAP | p0 §1 | backend/__init__.py：capabilities 表 + `use_gva_layerwise()` + `backend_supports()`；4 处调用点收敛；**ascend_store_connector.py:199 回归修复（main 活回归）** |
| IFACE | p0 §2 | base.py：5 个 memcache 存根移入 `GVALayerwiseCapable` 抽象接口；`MemcacheBackend(Backend, GVALayerwiseCapable)`；mooncake/yuanrong 零改动；一致性 UT |
| KEY | p0 §3 前半 | `GVAKeyFactory`（full_key/partial_key/hit_check_keys）+ **字节级快照 UT 基线（前置保护）** |
| PROTO | p0 §3 后半 | `GVASession`（alloc_gvas_for_save/prepare_load_gvas/_refresh_allocated_gvas）+ `GVAHitChecker`（hit_tokens）+ 4 个常量迁移 + `get_partial_block_index` → metadata.py |
| LIFE | p1/p2 §2 | `on_worker_ready()` hook（~15 行），删除 pool_worker:816-817 特判；**含 `_lazy_init` gate（二轮审核问题 1：compress+sdma 组合的惰性初始化契约不被击穿）** |

**规模**：~150 行新增生产代码 + ~500 行协议平移 + ~600-700 行 UT ≈ **1300-1500 行**（视 git 统计口径）

**出口验证**：UT 全量回归；mooncake 非 layerwise 冒烟；**MultiConnector PD 冒烟（回归修复生效点，PR-A 的独立合入价值所在）**；memcache layerwise TP=4 长前缀冒烟；**compress 模型路径（lazy_init 下 `on_worker_ready` no-op + 首传自愈，UT 覆盖或冒烟）**

### PR-B：传输线程收敛

**主题**：GVA 传输线程与批量拷贝收敛进 backend（新增 `backend/gva_threads.py`）。**依赖 PR-A**。

**内容**：

| 模块 | 原方案章节 | 内容 |
|---|---|---|
| THRD | p1/p2 §1 | `LayerBatchBuilder` + `_split_transfer_packets`/`_batch_copy_with_limits` + GVA 收/发线程平移；`batch_copy` 接口化（封堵 `m_store.store.batch_copy` 直捅，GVALayerwiseCapable 第 6 个方法）；ctx dataclass + 模块级工厂；`_build_group_layer_builders` 逻辑入工厂 |
| TRACE | p1/p2 §3.2 | docstring 中性化 3 处（attention_fence.py:31/:65、layerwise_cache_layout.py:71） |

**明确砍掉**（审核问题 6 采纳结果）：`check_all_layers_exists` 迁 metadata——非 backend 相关，属需求 3 范畴，不做（避免范围蔓延质疑）。

**规模**：~120 行新增 + ~700 行平移 + ~150 行 UT 改动 ≈ **950-1000 行**

**出口验证**：UT 全量；memcache layerwise TP=4 冒烟（传输路径全变，PR-B 核心出口）；三条 grep 断言（GVA 类名不出现在通用层 / `.store.` 直捅清零 / `== "memcache"` 仅注册表命中）

---

## 3. 抢先合入策略（对 #12854）

**用户裁定：不等 #12854，抢先合入，让 #12854 rebase。** 独立于审核意见，该策略的工程正当性：

1. **顺序正确性**：PR-A/PR-B 是**行为保持**的重构；#12854 是**语义变更**的重做（per-request 租约引用计数、失败回滚、fatal 错误停机等行为变化）。经典顺序就是行为保持先行、语义变更后行——后者的 rebase 面向新基座是机械工作，反向则是对既有语义的重解释。
2. **任务正当性**：需求 4 是立项任务；#12854 是个人 in-flight PR。收敛方向与 #12854 的 "Centralize GVA resolution, lease ownership" commit **同向**——我们的 GVASession 落地后，其 LayerwiseTransferPreparer 恰好可以直接组合它，rebase 后代码量预期更小。
3. **独立合入价值**：PR-A 携带 main 活回归修复（MultiConnector AttributeError，#14697 不触碰 connector，已核实），有不受 #12854 影响的合入理由。
4. **反向等待的代价**：若等 #12854 先合，其 Preparer 成为事实基座，需求 4 的收敛需对齐其新结构，等于被一个 open PR 无限期绑架。

**诚实的代价**（接受）：
- ader47 的 #12854 rebase 负担重（同区 ~500 行移动）；PR-B 与其 kv_transfer.py pipeline 重构冲突面更大
- 可能引来 maintainer 的合入顺序仲裁；若最终裁决 #12854 优先，我们 rebase——方向正确性不依赖合入顺序

**缓解措施**（把抢先合入的摩擦降到最低）：
- PR-A/PR-B 描述中附**新旧方法映射表**（下方），主动降低对方 rebase 成本
- PR 描述 @ #12854 作者，说明收敛与其目标同向、GVASession 可被其 Preparer 直接组合

**新旧方法映射表**（附于 PR 描述，供 #12854 rebase）：

| main 位置 | PR-A 后位置 |
|---|---|
| pool_worker._make_layerwise_gva_key（:1113-1123） | backend/gva_protocol.py `GVAKeyFactory.full_key` |
| pool_worker._make_layerwise_partial_key（:1125-1133） | `GVAKeyFactory.partial_key` |
| pool_worker._get_partial_block_index（:1097-1111） | metadata.py `get_partial_block_index` |
| pool_worker._refresh_allocated_gvas（:1134-1149） | `GVASession._refresh_allocated_gvas` |
| pool_worker._alloc_gvas_for_save（:1151-1313） | `GVASession.alloc_gvas_for_save` |
| pool_worker._prepare_load_gvas（:1315-1515） | `GVASession.prepare_load_gvas` |
| pool_scheduler._make_layerwise_gva_keys_for_hit_check（:306-316） | `GVAKeyFactory.hit_check_keys` |
| pool_scheduler._get_layerwise_gva_hit_tokens（:319-387） | `GVAHitChecker.hit_tokens` |

---

## 4. 执行时序（含 #14697 的自排序）

```
① #14697 先行合入（用户本人 PR，已在 review 流程中）
   —— 已核实其 diff 恰好重排了 _init_kv_transfer_config 周边（use_gva_layerwise 行保留仅重排），
      与 PR-A 的 CAP 区域相邻；先落地可消除一次自冲突
② PR-A 开工（唯一前置 = #14697；与 #12854 无关，抢先策略）
③ PR-B 在 PR-A 合入后开工
```

#14697 落地后 PR-A 的适配点（已核实 diff，均为单行级）：`backend_name` 取值一行化（仍保留 `use_gva_layerwise` 双处定义，P0-1 仍必要）、`_init_backend` 签名简化、`init_store` 删除（不影响 LIFE，见 §1 问题 4）。

---

## 5. PR-A 规模与验收标准 11 的处理

需求分析 §5 验收标准 11 要求 PR ≤ 1000 行。PR-A 预计 1300-1500 行，**超限论证**：

- 构成上 ~80% 为机械平移 + UT 保护——而 UT 覆盖本身就是验收标准 1/3 的硬要求，无法压缩
- 人工审查面收敛于 ~150 行新代码（capabilities 表 + 接口签名 + hook + 工厂挂接）；平移部分按 p0 方案既定纪律做新旧代码块机械 diff 校验
- PR 描述中明确该构成比例，请求 reviewer 按审查面而非行数评估

**若 maintainer 坚持硬约束的回退切点**：KEY 独立成前置小 PR（~350 行，纯 key 构造统一 + 快照 UT），PR-A 剩 ~1100 行。**默认不切——按用户裁定保持 2 PR。** PR-B（~950-1000 行）压线通过。

---

## 6. 风险表（合并保留 + 抢先策略新增）

| 风险 | 应对 |
|---|---|
| #12854 作者 rebase 负担重 / maintainer 顺序仲裁 | §3 映射表 + @ 作者沟通；接受最坏情形（我方 rebase），方向正确性不受影响 |
| #14697 与 PR-A 的 CAP 区域相邻冲突 | §4 自排序：#14697 先行，PR-A 基于其上 |
| ready_event 时序（send 线程 ready 后才建 recv 的 save_failure_checker） | 工厂调用顺序逐行保持现结构；PR-B 冒烟覆盖 save 失败传导路径 |
| `_batch_copy_with_limits` 提取后 key-mode 线程误引用 | 已核实仅 GVA 路径调用（kv_transfer.py:1363/:1560）；提取后 grep 复核 |
| on_worker_ready 顺序变化破坏 memcache buffer 注册 | §1 问题 4 的等价性论证（ensure_initialized 内补注册）+ 专项 UT |
| ctx dataclass 15+ 参数遗漏 | 工厂签名与现线程 __init__ 一一对应 + mypy + UT 参数映射断言 |
| 平移过程顺手重构引入行为漂移 | 字节级快照 UT 前置（KEY）；平移纪律"逐字符一致"；新旧代码块机械 diff |

---

## 7. 本轮文档修订记录

| 文档 | 修订 |
|---|---|
| requirements_analysis.md | P0-1 标题精确化（2 定义 + 1 判断 + 1 引用） |
| p0_modification_plan.md | §2.3 与目录树勘误：mooncake/yuanrong **零改动**（存根在 base.py，审核问题 3） |
| p1_p2_modification_plan.md | §2.2 措辞修正：等价性依赖 `ensure_initialized`（:62-72）而非 `init_store`（#14697 已删后者，不影响论证——审核问题 4 的证伪结论） |
| 本文档 | 新增；取代两份方案的 §4 PR 拆分章节（6 PR → 2 PR）；确立抢先合入策略 |
