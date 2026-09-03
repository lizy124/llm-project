# 第三次审核回应与二 PR 执行方案（v4 · 执行基准）

> 输入：[review-round-3.md](review-round-3.md)（三轮审核）
> 处理原则：延续"逐条独立核实、不轻信"——本轮证实三轮的核心发现（最小方案 load 路径功能破坏），同时**证伪其一处细节推演**（assert 崩溃不可达），并将破坏定性修正为比三轮所述更隐蔽的形态
> 文档地位：本文档取代 [第二次review-response-pr2-plan.md](第二次review-response-pr2-plan.md)（v3）成为唯一执行基准。按用户指示，此前所有文档**保持原样不动**（审计链完整：需求分析 → p0/p1p2 方案 → v2 → 一/二/三轮审核 → v3 → 本文）。v3 被三轮证实瑕疵之处以本文档 §0 勘误为准

---

## 0. v3 勘误清单（原文不动，以本文为准）

| v3 位置 | 问题 | 修正 |
|---|---|---|
| §2.1 末段 "两案均满足必改清单 #1" | **危险低估**（三轮问题 1 主论点，本轮证实）：最小方案的 load 路径是功能破坏而非时机偏移 | §1.1 / §2.1：最小方案**不满足**必改清单 #1，仅作设计对照，禁止作为实现选项 |
| §2.1 最小方案括注 "batch_alloc/put 内部自愈终态一致" | 只验了 save 路径，漏了 load 路径全链无自愈 | §1.1 事实链：load 路径经 `batch_get_key_info` 空返回 → GVA 向量静默全 0 |
| §1 末尾我方对二轮的批注 a（"非 layerwise 口径略窄"） | 无效扩张（三轮问题 2a，本轮复核成立） | 撤回：二轮 "memcache + 非 layerwise + compress" 即精确集合 |
| §1 末尾我方对二轮的批注 b（"认账主体实为 v2 回应"） | 指认错误对象（三轮问题 2b，本轮复核成立） | 撤回：认账方是二轮审核作者（在其文档中确认 v2 反指控成立）；v2 是反指控提出方，不是认账方 |

**不受影响的部分**：v3 §2.1 推荐方案（gate + GVASession self-ensure，即"严格版"）经三轮 1.2 逐环核验全部通过——**设计本身维持不变**，本次仅修正对最小方案的定性、撤回两处误批注、补充 UT 覆盖。

---

## 1. 三轮审核逐条独立核实

### 1.1 问题 1（最小方案 load 路径功能破坏）——主论点采纳，一处细节证伪，定性加重

本轮对代码事实链做了完整独立验证（引用行号均为当前 main 快照）：

**调用顺序**（pool_worker.py `process_layer_data`）：`:1608 _prepare_load_gvas` **先于** `:1610 _alloc_gvas_for_save` ✓——load 先行，save 的自愈救不了已经拿空结果的 load。

**backend 方法 lazy 行为**（memcache_backend.py，本轮逐一核实）：

| 方法 | lazy 未初始化时 | 自愈性 |
|---|---|---|
| `exists`（:133-140） | 短路返回全 0 | 优雅降级 |
| `batch_get_key_info`（:142-151） | 短路返回 `[]` | **无自愈** |
| `batch_alloc`（:153-156） | 首行 `ensure_initialized()` | **唯一自愈点**（save 路径） |
| `batch_add_lease`（:158-160） | 无 lazy 分支，`assert self.store is not None` | 伪威胁（见下） |
| `batch_remove_lease`（:162-164） | 同上直接 assert | 伪威胁（同下） |
| `get`（:175-183） | 日志报错并 return | 静默失败 |

**load 路径完整传导链**（最小方案下 GVA + lazy、store 未初始化、请求含已缓存前缀）：

```
:1393  key_infos = self.m_store.batch_get_key_info(keys)   → 短路返回 []
:1394  for ki, key, block_idx in zip(key_infos, keys, ...)  → zip 空转，循环体零次执行
        ⇒ gvas=[] / valid_gva_indices=[] / valid_keys=[] / invalid_block_ids=[]
:1417  if valid_keys:                                       → False，跳过 batch_add_lease
:1476  if invalid_block_ids:                                → False，跳过 batch_remove_lease
:1500  full_gvas = [0] * full_len；gvas[:normal_gva_count] 为空切片
        ⇒ request.load_block_gvas_np = 全 0 向量，无任何告警（循环体没跑，连 warning 都没有）
```

**对三轮细节的证伪**：三轮称 "`batch_add_lease` 更会在 `assert` 上崩溃"——**不可达**。`if valid_keys:`（:1417）保护在 key_info 空返回的传导下使 add_lease 调用被跳过；`batch_remove_lease`（:1480）更在 `if invalid_block_ids:`（多 group 分支）与 `if leased_keys_to_release:` 双重保护下不可达。空结果链下两个 assert 均不会触发。

**定性加重（比三轮所述更危险）**：破坏形态不是崩溃而是**静默失效**——全 0 的 `load_block_gvas_np` 不触发任何现有断言/异常/warning，仅表现为 KV 缓存命中率归零、传输日志无 load 发生。崩溃会被 CI 与冒烟立即抓住；静默退化恰恰**穿透** PR-A 的全部出口验证（UT 回归绿、冒烟跑通、无报错）。三轮"出生产事故"的预警方向正确，事故形态需要修正——这也解释了为何该缺陷连续四份材料（需求分析、p0/p1p2、二轮、v3）均漏检。

**"纯 consumer 节点"论点证实**：`:1234 if new_keys:` 保护使无新块可 save 的节点不调用 `batch_alloc`——store 永不初始化，load 永久失效 ✓（PD 分离场景 P 节点即此形态）。

**本轮新发现（三轮亦未展开）**：scheduler 侧 hit check 同样依赖该语义——pool_scheduler.py:348 `key_infos = self.store_scheduler.batch_get_key_info(all_keys)`。若 scheduler 侧 store 亦处 lazy 未初始化态，hit check 拿空 → 全 miss → 请求根本不进入 worker 的 load 路径（表现为"缓存永不命中"）。严格版方案（GVASession self-ensure）+ v3 §2.2 的 UT 需覆盖此面（见 §2.2 补充）。

**结论**：三轮问题 1 主论点采纳——v3 "两案均满足必改清单 #1" 是危险低估，最小方案下 load 是**功能性静默失效**。但"assert 崩溃"细节证伪，且静默形态比崩溃更值得警惕。

### 1.2 问题 2（v3 两处批注本身有误）——全部采纳撤回

- **批注 a 复核成立**：`use_gva_layerwise = use_layerwise ∧ backend=="memcache"` ⟹ 对 memcache，非 GVA ≡ 非 layerwise；key-mode layerwise 仅存在于 mooncake/yuanrong，其 `on_worker_ready` 继承默认 no-op，不在 LIFE 影响范围。二轮口径即精确集合，我方"精确化"实为引入无关分支的无效扩张。
- **批注 b 复核成立**：反指控由 v2 提出、由二轮审核作者在其文档中确认成立——认账方是二轮审核作者；v2 是被审对象兼反指控方，无认账动作。我方指认"认账主体是 v2 回应"把提出方当成了认账方。

### 1.3 三轮 1.2 节（LIFE 升级核验）——接受其全部结论

其对本轮 v3 严格版的六项逐环验证（#12854 commit 80d3bc6 引用逐字属实、non-lazy 构造器幂等、pending-buffer 对称、GVASession 构造点顺序等）与我所读代码一致，无异议。补充本轮独立验证的部分：`:1608/:1610` 顺序、六个 backend 方法的 lazy 行为、两条保护性分支（:1417/:1476）、`:1500` 全 0 向量构造——见 §1.1。

---

## 2. 修正后的设计（仅列 delta，其余沿用 v3 §2 与原方案）

### 2.1 LIFE 段落重写（三轮必改 #1）

**推荐方案（严格版，不变）**：`on_worker_ready` 加 `_lazy_init` gate + `GVASession.__init__` 首行 `store.ensure_initialized()`。四情形等价表、依据（#12854 commit 80d3bc6 证明 GVA+lazy eager 是既定意图）均沿用 v3 §2.1。

**最小方案定性段落（重写）**：

> 最小方案（仅 `on_worker_ready` 加 gate、无 GVASession self-ensure）**不满足必改清单 #1**：load 路径 `batch_get_key_info` 空返回 → zip 空转 → `load_block_gvas_np` 静默全 0 → KV load 功能性失效；纯 consumer 节点（`if new_keys:` 保护下无 save 分配）store 永不初始化、load 永久失效；且全 0 向量不触发任何异常/断言/告警，可穿透全部出口验证。此方案仅作设计对照保留，**禁止作为实现选项**。（三轮所述 "batch_add_lease assert 崩溃" 不可达——:1417 `if valid_keys:` 保护；真实的破坏形态是静默失效，比崩溃更难检出。）

### 2.2 UT 覆盖扩充（v3 §2.2 基础上追加）

在 v3 三条 UT（lazy gate 不触发 / GVASession self-ensure / exists 短路契约固化）之上追加：

4. **`batch_get_key_info` 空返回传导回归测试**（golden 测试）：`_lazy_init=True` 且未初始化时 `batch_get_key_info` 返回 `[]`、`batch_alloc` 后返回真实结果——把"空返回是降级信号而非错误信号"的契约固化为 UT，防后续 PR 把它当 bug "修复"或当正常态沿用
5. **load 路径非空 keys 的端到端断言**（PR-B 后置）：构造含已缓存前缀的请求，断言 `load_block_gvas_np` 中命中块的 gva > 0——此断言在最小方案下会失败（全 0），是"静默失效"的唯一直接探针
6. scheduler 侧（pool_scheduler.py:348 路径）hit check 的 key_info 语义随 PR-A 的 GVAHitChecker UT 一并覆盖

### 2.3 其余修正

- v3 §1 末尾两处对二轮的批注：撤回（§0 勘误 3/4 项），后续引用二轮口径时以其原文"memcache + 非 layerwise + compress"为准
- v3 §2.2/§2.3/§2.4、§3 二 PR 结构、§5 超限论证与回退切点、新旧方法映射表：**全部不变**（三轮 §三已核验通过）
- **时序弹性（追加核实，见 §4.1）**：经拉取 #14697 完整 diff 逐区域比对，#14697 与 PR-A/PR-B **无硬依赖、冲突面小且全机械**——非"唯一前置"，两序皆可

---

## 3. 三轮必改清单回应映射（三轮 §四 → 本文档）

| 三轮必改项 | 回应 |
|---|---|
| 1.【必改】§2.1 "两案均满足必改清单 #1" 改为 "最小方案不满足，仅作对照" | §2.1：采纳重写；定性从三轮的"load 功能破坏 + assert 崩溃"修正为"load 功能破坏 + 静默失效（assert 不可达）" |
| 2.【建议】删除 §1 末尾两处对二轮的误批注 | §0 勘误 3/4 项 + §1.2：采纳撤回 |

---

## 4. 净结论

三轮审核的核心发现（问题 1）经本轮独立验证**成立且需加重**：最小方案的功能破坏形态是静默的全 0 GVA 向量——不崩溃、不报错、不告警，可穿透全部出口验证，这正是它连续逃过四份材料检出的原因。三轮自身的细节推演（assert 崩溃）被本轮证伪，但该证伪不减轻反而加重主论点（静默失效比崩溃更危险）。问题 2 两处批注的撤回无争议。

**三轮审核对 v3 的其余核验（§三全表、1.2 六项、必改清单映射）全部通过，执行方案结构零变更。** 交叉审核三轮战绩更新：

| 轮次 | 首检出的缺口 |
|---|---|
| 一轮 | PR3 × #12854 冲突低估 |
| 二轮 | LIFE lazy_init 击穿（一行修复有盲区） |
| 三轮 | 最小方案定性的危险低估（load 路径） |
| 本轮回应 | 三轮细节证伪（assert 不可达）+ 静默失效定性 + scheduler 侧 :348 依赖面 |

每轮均有对方未见的新发现，且全部落在"可修正"范围——二 PR 框架本身三轮零动摇。延续机制至 PR-A/PR-B 描述的 self-review。

**执行就绪（终态）**：`PR-A（LIFE 严格版 + §2.2 六条 UT + MultiConnector 回归修复）→ PR-B`。方案文档链至此收敛于本文档，无遗留未决项。

### 4.1 #14697 时序弹性（追加核实：非硬前置）

拉取 #14697 完整 diff（4 文件）与 PR-A/PR-B 计划区域逐块比对：

**#14697 实际改动**：pool_worker（`__init__` 删 dp_rank / `_init_backend` 签名简化 / `_init_kv_transfer_config` backend_name 一行化，**`use_gva_layerwise` 行保留不动** / `_init_parallelism_info` use_mla 简化）；pool_scheduler（`__init__` backend_name 局部变量化 + 同一行保留 / `:274` `batch_is_exist`→`exists`）；memcache_backend（删 `init_store`）；test_pool_scheduler（1 行 mock 改名）。

**与 PR-A 冲突面（全部机械可解）**：

| 区域 | #14697 | PR-A | 冲突性质 |
|---|---|---|---|
| pool_worker `_init_kv_transfer_config` | 重排 backend_name 行 | CAP 删除 `use_gva_layerwise` 行换派生调用 | 文本冲突，drop hunk 即解 |
| pool_scheduler `__init__` | backend_name 局部变量化 | CAP 替换同行 | 文本冲突，drop hunk 即解 |
| pool_scheduler `:274` rename | `batch_is_exist`→`exists` | 不动该行（hit-check 区在 :306-387） | 无冲突 |
| memcache_backend | 删 `init_store`（:107） | 加 `on_worker_ready` + 继承（不同区域） | 无冲突 |
| test_pool_scheduler | 1 行 mock 改名 | hit-check 用例改造（不同用例） | 小冲突 |
| `_init_backend` 签名 / dp_rank / use_mla | 简化 | 不动 | 无冲突 |

**与 PR-B**：#14697 不触碰 kv_transfer.py / gva_threads.py / attention_fence / layerwise_cache_layout——**零冲突**。

**无硬依赖**：LIFE 挂 `ensure_initialized`（main 已有，#14697 不动）；`use_gva_layerwise` 定义行在两基线上逐字符一致（PR-A 的 hunk 等价适用）；init_store 删与否不影响 PR-A 设计。

**两序决策规则**：#14697 review 顺利 → 先合（现行计划，PR-A 一次适配点已文档化于 v3 §2.4）；review 卡住 → **不必等**，PR-A 直接基于当前 main 开工，#14697 后合 rebase（其两个 use_gva_layerwise hunks 被 CAP 吸收 drop，分钟级）。"小 PR 后合 rebase" 便宜于 "大 PR 后合 rebase"（#14697 ~50 行 vs PR-A ~1400 行），且 PR-A 先行可更早落地 connector 回归修复。唯一避免：两 PR 并行流转互追 rebase。
