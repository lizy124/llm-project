# 第四次审核回应与二 PR 执行方案（v5 · 执行基准 · 文本轮收口）

> 输入：[review-round-4.md](四轮审核)（四轮审核）
> 处理原则：延续"逐条独立核实、不轻信"——本轮核实结果：四轮三项发现**全部成立**，其中问题 1（scheduler 前提空集）证伪了 v4 自己的"本轮新发现"，交叉审核的证伪权与出错权再次并存
> 文档地位：本文档取代 [第三次review-response-pr2-plan.md](第三次review-response-pr2-plan.md)（v4，含其 §4.1 时序弹性段）成为唯一执行基准，并作为**文字阶段的收口文档**——四轮审核的收口建议（不再多轮文字往返）经评估采纳。此前所有文档保持原样不动（审计链：需求分析 → p0/p1p2 → v2 → 一/二/三/四轮审核 → v3/v4 → 本文）

---

## 0. v4 勘误清单（原文不动，以本文为准）

| v4 位置 | 问题 | 修正 |
|---|---|---|
| §1.1 "本轮新发现：scheduler 侧 hit check 同样依赖该语义" | **前提为空集**（四轮问题 1，本轮独立验证成立）：scheduler 经 `create_scheduler_client` 构造，不传 `lazy_init` → 构造器默认 `False` → `_lazy_init` 恒 False → `:58-60` 构造即 eager。scheduler 侧 lazy 未初始化态**不可能存在** | 撤回该"新发现"；正确表述见 §1.1 |
| §2.2 UT 第 6 条（scheduler 侧 :348 覆盖） | 对不存在的状态做覆盖（无害但无意义） | 删除；GVAHitChecker 常规 UT 自然覆盖 :348 路径（其 store 恒 eager，无 lazy 分支可达） |
| §1.1 "PD 分离场景 P 节点即此形态" | D/P 颠倒（四轮问题 2）：纯 consumer 是 **D 节点**（decode 只 load）；P 节点对新请求必然产生可 save 新块、`new_keys` 非空、store 自愈 | 改为 D 节点 |
| §1.1 传导链行号 `:1394/:1476/:1500/:1608` | 漂移 ±1~10（四轮问题 3） | 校准为 `:1397/:1466/:1502/:1609`（本轮 grep 实测）；另 v4 "连 warning 都没有"精确化为"无 warning 级及以上日志，:1490 有 debug 级 `valid_gvas=0` 可观测点" |

**不受影响**：严格版 LIFE 设计（gate + GVASession self-ensure）、二 PR 结构、UT 第 4/5 条、超限论证、映射表——四轮 §三全部核验通过，零变更。

---

## 1. 四轮审核逐条独立核实

| # | 四轮断言 | 我的独立验证 | 判定 |
|---|---|---|---|
| 1 | 【中】v4 "scheduler 侧依赖"前提为空集 | 读取 memcache_backend.py:40-60：构造器 `lazy_init: bool = False`（默认）、`self._lazy_init = lazy_init and _is_device_sdma()`、`if not self._lazy_init: self.store = self._setup_store(); self._store_initialized = True`（构造即 eager）；:100-105 `create_scheduler_client` 返回 `cls(parallel_config, local_rank=0, init_bm=False)` **不传 lazy_init**；pool_scheduler.py:172 确认经此构造。**传导：scheduler 侧 `_lazy_init` 恒 False → `batch_get_key_info` 的 lazy 短路分支（:142-151）永不可达**。附带验证 mooncake 侧 `create_scheduler_client`（mooncake_backend.py:169-171）同样不传 lazy_init——两后端 scheduler 客户端均恒 eager，v4 的"若 scheduler 侧 store 亦处 lazy 态"对任何后端都不成立 | **属实，采纳撤回**。四轮自身注记（首参实为 parallel_config、"语义示意"）亦诚实准确 |
| 2 | 【微】P/D 颠倒 | 概念复核成立：PD 分离中 D（decode）只 load 是纯 consumer 的实证形态；P（prefill）对新请求必产生新块。v4 写 P 节点系笔误 | **采纳**：改 D 节点 |
| 3 | 【微】行号漂移 | grep 实测：`for ki...zip` = :1397、`if invalid_block_ids:` = :1466、`full_gvas = [0]*full_len` = :1502、`_prepare_load_gvas(requests)` = :1609 / `_alloc_gvas_for_save(requests)` = :1610、debug 日志 = :1490——**四轮给出的行号全部精确**，v4 引用漂移确认 | **采纳**：行号以其为准 |
| 4 | （其 §1.2 精度备注）:1489-1499 有 debug 级日志可观测 valid_gvas=0 | grep 确认 :1490 `"load_gvas: ... valid_gvas=%d ..."`，空转后仍执行且显示 valid_gvas=0——生产默认级别不可见，但 debug 可观测 | **采纳**：v4 "连 warning 都没有"修正为"无 warning 级及以上；debug 级可观测" |
| 5 | （其 §1.3）GVA 路径除 :816 特判外无其他 ensure 点（batch_copy 直捅 kv_transfer.py:468 / batch_write_finish :166-172 无 ensure） | 前轮已读验证一致：`batch_alloc`（:153-156）是唯一自愈点；`get`（:174-183）lazy 分支仅日志报错 | **属实**（维持既有结论） |

**净判定**：四轮对 v4 的三项修正全部成立——包括对我 v4 "本轮新发现"的证伪。四轮总账表（每轮证伪方引入新错误、严重度递减）与我方核对一致，是清晰的收敛信号。

---

## 2. #14697 后合情形的完整分析（回应用户裁定方向）

v4 §4.1 已确立"非硬前置、两序皆可"。本轮按"**PR-A 基于当前 main 开工，#14697 挂起后合 rebase**"展开为可执行方案（所有事实基于本轮对 main 与 #14697.diff 的比对）。

### 2.1 PR-A 基于当前 main（无 #14697）的适配

当前 main 的两处 `use_gva_layerwise` 定义（PR-A 的 CAP 替换点）：

```python
# pool_worker._init_kv_transfer_config（main 现状）
self.backend = extra_config.get("backend", "mooncake")
self.backend_name = self.backend.lower()
self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"   # ← CAP 替换

# pool_scheduler.__init__（main 现状）
backend_name = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")
self.backend_name = backend_name.lower()                                          # ← main 有此属性
self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"   # ← CAP 替换
```

PR-A 在此基线上的 CAP 调用点引用 `self.backend_name`（两侧属性在当前 main 均存在）。设计无任何变化——CAP/IFACE/KEY/PROTO/LIFE 五模块在有无 #14697 的两个基线上等价适用（v4 §4.1 已论证：`use_gva_layerwise` 定义行两基线逐字符一致；LIFE 挂 `ensure_initialized`，#14697 不触碰）。

### 2.2 #14697 后合时的 rebase 账单（逐 hunk）

| #14697 hunk | 与 PR-A 合并后 main 的关系 | rebase 动作 |
|---|---|---|
| worker `_init_kv_transfer_config`（backend_name 一行化，use_gva_layerwise 行为**上下文**） | CAP 已替换该上下文行 | 冲突 → 保 #14697 的 backend_name 简化，丢上下文适配。**分钟级** |
| worker `__init__`（删 dp_rank）/ `_init_parallelism_info`（use_mla 简化）/ `_init_backend`（签名简化） | PR-A 不触碰这些函数 | 自动合入 |
| worker `:816` eager 特判区域 | PR-A 的 LIFE 已删除该特判（换 `on_worker_ready`） | #14697 本就不碰该区域 → 无交集 |
| scheduler `__init__`（backend_name 局部变量化 + **修改 use_gva_layerwise 行**：`self.backend_name` → 局部 `backend_name`） | CAP 已替换 use_gva_layerwise 行，且其调用点引用 `self.backend_name` | **唯一真实适配点**（见 2.3） |
| scheduler `:274` rename（`batch_is_exist`→`exists`） | PR-A 不动该行（hit-check 在 :306-387） | 自动合入 |
| memcache_backend 删 `init_store`（:107-114） | PR-A 加 `on_worker_ready`/继承/`GVASession` self-ensure 在不同区域 | 自动合入 |
| test_pool_scheduler 1 行 mock 改名 | PR-A 的 hit-check 用例改造在别的用例 | 自动/小冲突 |

**净额**：2 处强制冲突 + 1 处单行适配，其余全部自动——分钟级。且方向最优："小 PR（~50 行）rebase 到大 PR 之后"远便宜于反向。

### 2.3 唯一真实适配点：scheduler 侧局部变量

#14697 会把 scheduler 的 `self.backend_name` 属性降为局部变量 `backend_name`。若 PR-A 的 CAP 调用点（已合入 main）写的是 `use_gva_layerwise(self.backend_name, self.use_layerwise)`，#14697 rebase 删掉属性赋值后该行将 `AttributeError`——**但不会静默发生**：#14697 的 scheduler hunk 本身修改 `use_gva_layerwise` 行（属性→局部变量），与 CAP 的替换**必然产生强制冲突**，rebase 者被冲突引导到此行，改为 `use_gva_layerwise(backend_name, self.use_layerwise)` 即可。

消除该适配点的可选优化（PR-A 描述中提示 rebase 者即可，不改设计）：CAP 调用点不引用 `self.backend_name` 属性，直接 `use_gva_layerwise(<config 读出的 backend 名>, self.use_layerwise)`——则 #14697 后合零适配。**默认不做**（当前 main 属性存在，引用属性是最小改动；为未来 rebase 预优化属过度设计）。

### 2.4 两序决策规则（终版）

```
#14697 review 顺利先合  → PR-A 按 v3 §2.4 适配（调用点引用局部变量 backend_name）
#14697 卡住/想抢先      → PR-A 直接基于当前 main 开工（本文 §2.1），
                          #14697 后合按 §2.2 账单 rebase（2 冲突 + 1 适配，分钟级）
唯一避免                → 两 PR 并行流转互追 rebase（等 #14697 有明确落地窗口再开工 PR-A，或 PR-A 开工后 #14697 暂缓 rebase 直到 PR-A 合入）
```

抢先合入 PR-A 的额外收益：connector :199 回归修复更早落地、GVA 协议基座更早成型——与对 #12854 的抢先策略同构（行为保持重构先行）。

---

## 3. 文本阶段收口（采纳四轮建议）

四轮观察成立：四轮总账呈"每轮证伪方引入新错误、严重度单调递减（冲突低估 → 功能破坏 → 细节证伪 → 空前提）"的收敛形态，剩余修正已全部文字级。**本文档为文字阶段终稿**：

- 四轮必改清单 3 项已全部落入 §0（无遗留）
- 后续不再产出新的回应文档；若出现第五轮审核且仅有文字级发现，直接并入 PR-A 描述处理
- 交叉审核精力转移至 **PR-A/PR-B 描述的 self-review**（四轮指定的下一个有真实发现空间的位置）——尤其两处高风险面：①静默失效探针（UT 第 5 条非零 gva 端到端断言）是否真的进入出口验证；②新旧方法映射表与最终 diff 的逐项一致性

## 4. 执行就绪（终态）

```
PR-A（LIFE 严格版 + §2.2 五条 UT[第 6 条已删] + MultiConnector 回归修复）
  ├─ 基线：当前 main（#14697 未合亦可开工，§2.1）
  └─ 出口：UT 全量 / mooncake 非 layerwise 冒烟 / MultiConnector PD 冒烟 / memcache layerwise TP=4 长前缀冒烟 / 三条 grep 断言
→ PR-B（gva_threads 线程收敛，依赖 PR-A）
  └─ 出口含 UT 第 5 条（静默失效唯一直接探针，PR-B 后置合理：load 路径端到端在传输线程收敛后才完整）

UT 清单（终版五条）：① lazy gate 不触发 ② GVASession self-ensure ③ exists 短路契约固化
                     ④ batch_get_key_info 空返回契约 golden ⑤ load 非 0 gva 端到端断言（PR-B）
```

方案文档链至此收敛：**需求分析 → p0/p1p2 方案 → v2 → 四轮审核 → v5（本文，唯一执行基准）**。无遗留未决项，PR-A 可直接开工。
