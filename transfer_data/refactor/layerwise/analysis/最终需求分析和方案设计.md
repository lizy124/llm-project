# 最终需求分析与方案设计：AscendStore Layerwise 后端收敛

> 文档地位：**唯一执行基准**。综合需求分析、p0/p1p2 方案、v2-v5 回应、五轮交叉审核的终稿——此前所有文档（需求分析 → p0/p1p2 方案 → v2 → 一至四轮审核 → v3/v4/v5）保持原样不动，作为完整审计链；本链中被后续轮次修正的表述以本文档为准
> 代码基线：`code/vllm-ascend` `refactor_layerwise` 分支（2026-08-29，与 fork main 完全同步），行号均为实测
> 需求：kv_pool 四项重构之第 4 项——"AscendStore Layerwise 代码重构，将后端强相关的逻辑收敛到 backend 文件中"（前三项已落地：UT 精简 #14465、目录重构 #13354、死代码清理主体完成）

---

## 1. 需求与范围

### 1.1 核心目标

layerwise 传输路径中后端强相关的逻辑（当前即 memcache 独有的 GVA 机制）散落在通用层（pool_worker / pool_scheduler / kv_transfer），收敛到 backend 目录，实现框架与后端解耦——此后新增或维护后端不需要动框架代码。

### 1.2 术语

- **后端强相关逻辑**：仅对特定 backend 生效/有意义的代码路径。当前 layerwise 场景下指 memcache 独有的 GVA（Global Virtual Address）机制。
- **GVA layerwise**：`use_layerwise=True` 且 `backend=memcache`（`use_gva_layerwise`）。现有后端：mooncake（默认）、memcache、yuanrong；layerwise 传输加速仅支持 memcache。

### 1.3 现有目录结构（需求 2 已落地）

```
kv_pool/
├── ascend_store/
│   ├── backend/
│   │   ├── base.py            # 39 行，通用+memcache 混合接口
│   │   ├── memcache_backend.py # 206 行
│   │   ├── mooncake_backend.py # 362 行
│   │   └── yuanrong_backend.py # 146 行
│   ├── pool_worker.py          # 2266 行，GVA 协议最大散布点
│   ├── pool_scheduler.py       # 969 行
│   ├── kv_transfer.py          # 1615 行（物理行），6 个传输线程类
│   ├── metadata.py             # 1075 行
│   ├── ascend_store_connector.py # 336 行
│   ├── layerwise_cache_layout.py # 357 行
│   └── coordinator.py / attention_fence.py
├── ucm_connector/              # 需求 2 新开，与 ascend_store 平级
└── ...
```

---

## 2. 现状问题（五轮审核确认的最终版）

### P0-1 `use_gva_layerwise` 派生语义散布 4 处

2 处定义 + 1 处判断 + 1 处消费引用：

| 位置 | 内容 |
|---|---|
| pool_worker.py:157 | `self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"` |
| pool_scheduler.py:163 | 同式独立定义 |
| layerwise_cache_layout.py:98-100 | `backend == "memcache" and use_layerwise` 判断 |
| ascend_store_connector.py:199 | 消费引用（**main 活回归：定义已被 #14465 误删**） |

**回归事故（已证实）**：#14465 死代码清理时以"stored but never read"为由删除 connector 侧定义（提交信息点名 use_compress 等字段、etc. 隐含），但 :199 仍在读——MultiConnector（PD 分离组合）初始化即 `AttributeError`。触发链：`AscendMultiConnector.__init__` → `_configure_layerwise_reuse_completion`（ascend_multi_connector.py:52）→ `set_external_slot_release_waiter` → :199。修复 commit `412b157` 挂在 open PR #12711，**main 仍带回归**。此事故即 P0-1 的直接代价：派生语义多点复制，删一处即崩。

### P0-2 GVA 协议实现全部位于通用层

| 逻辑 | 位置 | 规模 |
|---|---|---|
| GVA key 构造（save/load） | pool_worker.py:1113-1133 | `model[@group_id]@hash@rank`、`model@partial@req@...@rank` |
| GVA key 构造（hit-check） | pool_scheduler.py:306-317 | 同格式第二份实现（all-rank 展开） |
| save 侧 GVA 分配 | pool_worker.py:1151-1313 | `batch_alloc`、已分配跳过、partial 块、`_allocated_gvas` 本地表 |
| load 侧 GVA + 读租约 | pool_worker.py:1315-1515 | `batch_get_key_info`、`batch_add_lease`（10 次 × 1ms 部分租约重试）、invalid block 上报 |
| GVA 本地缓存失效 | pool_worker.py:1134-1149 | `batch_is_exist` 驱动淘汰 |
| hit-check 判定语义 | pool_scheduler.py:319-387 | `ki.size()>0` 逐 group 收集、全局 min |

### P0-3 base.py 混入 memcache 专属接口

5 个 memcache 专属方法（`batch_get_key_info` / `batch_alloc` / `batch_add_lease` / `batch_remove_lease` / `batch_write_finish`，base.py:35-48）为普通 NotImplementedError 方法（非 @abstractmethod），通用基类承载单一后端协议。mooncake/yuanrong **子类零改动**（存根仅在 base.py；mooncake_backend.py:68 的 NotImplementedError 为无关协议报错）。

### P1-4 传输线程组合与选择逻辑

kv_transfer.py 6 个线程类（非 layerwise / key-layerwise / GVA-layerwise × 收/发），选择逻辑在 pool_worker.py:463-584 三重 if/else 嵌套；GVA 专属的 `LayerBatchBuilder`（:39-303）与 `_batch_copy_with_limits`（:426-476）也在通用文件。

**封装破坏**：GVA 线程经 `_batch_copy_with_limits` 直捅 backend 内部——kv_transfer.py:468 `self.m_store.store.batch_copy(...)`（绕过 put/get 接口直达内部属性）。put/get 仅被 key 模式线程以 key 语义使用（:863/:970/:1155/:1256）。

### P1-5 backend 生命周期硬编码

pool_worker.py:816-817：`if self.use_gva_layerwise: self.m_store.ensure_initialized()`——memcache 惰性初始化时序由通用层特判驱动。

### P2-6 零散后端痕迹

attention_fence.py:31/:65、layerwise_cache_layout.py:71 docstring 的 "MemCache" 字样（通用层）。`check_all_layers_exists`（:2386-2396）经裁定**不迁移**——gate 为 `use_layerwise`（:2121/:2320），服务 key-mode 三后端通用路径，非 backend 相关，属需求 3 范畴。

### 明确不动项

layerwise 调度语义（eagle trim、`kvpool_store_skip_tokens` 等）、reuse layout / offload 布局计算、非 layerwise 三后端数据面、coordinator.py / attention_fence.py 机制、ReqMeta 字段结构（`kv_transfer.py` 消费方零改动）。

---

## 3. 总体设计

### 3.1 目标结构

```
ascend_store/
├── backend/
│   ├── __init__.py        # [改] backend_map + capabilities 表 + use_gva_layerwise() 派生函数
│   ├── base.py            # [瘦身] 仅通用数据面；新增 GVALayerwiseCapable 抽象接口
│   ├── gva_protocol.py    # [新增·PR-A] GVA 常量 / GVAKeyFactory / GVASession / GVAHitChecker
│   ├── gva_threads.py     # [新增·PR-B] LayerBatchBuilder / GVA 收发线程 / ctx 工厂
│   ├── memcache_backend.py # [改] 继承 GVALayerwiseCapable + on_worker_ready + batch_copy
│   ├── mooncake_backend.py # [零改动]
│   └── yuanrong_backend.py # [零改动]
├── metadata.py            # [改] 新增 get_partial_block_index()
├── pool_worker.py         # [瘦身 ~420 行] GVA 协议方法删除，委托 self._gva_session
├── pool_scheduler.py      # [瘦身 ~85 行] hit-check 委托 self._gva_hit_checker
└── ascend_store_connector.py # [改] 回归修复（use_gva_layerwise 恢复）
```

### 3.2 模块清单

**PR-A：协议收敛 + 接口分层 + main 回归修复**（新增 backend/gva_protocol.py）

| 模块 | 内容 |
|---|---|
| CAP | backend/__init__.py 加 capabilities 表 + `backend_supports()` + `use_gva_layerwise()` 单点派生；4 处调用点收敛；**connector :199 回归修复** |
| IFACE | base.py 瘦身为纯通用数据面；5 个 memcache 接口提取为 `GVALayerwiseCapable`（ABC）；`MemcacheBackend(Backend, GVALayerwiseCapable)` 双继承；一致性 UT 防能力表与实现漂移 |
| KEY | `GVAKeyFactory`（full_key / partial_key / hit_check_keys）统一两份 key 构造；**字节级快照 UT 前置** |
| PROTO | `GVASession`（alloc_gvas_for_save / prepare_load_gvas / _refresh_allocated_gvas）+ `GVAHitChecker`（hit_tokens）+ 4 个常量迁移；`get_partial_block_index` → metadata.py 模块函数（GVA 与 key-mode 通用路径共用，不进 backend） |
| LIFE | `on_worker_ready()` hook + `_lazy_init` gate + GVASession self-ensure（见 §3.3） |

**PR-B：传输线程收敛**（新增 backend/gva_threads.py，依赖 PR-A）

| 模块 | 内容 |
|---|---|
| THRD | `LayerBatchBuilder` + `_split_transfer_packets`/`_batch_copy_with_limits` + GVA 收/发线程整类平移；**`batch_copy` 接口化**（GVALayerwiseCapable 第 6 个方法，封堵 `m_store.store.batch_copy` 直捅）；`GVALayerwiseThreadContext` dataclass + 模块级工厂（group_builders 构造逻辑入工厂）；`_build_group_layer_builders`（pool_worker.py:444-459）随迁 |
| TRACE | docstring 中性化 3 处 |

依赖方向无环：`kv_transfer.py（通用基类）← import ← backend/gva_threads.py ← import ← pool_worker.py`。

### 3.3 LIFE 设计（严格行为保持版——五轮审核演化的最终形态）

这是五轮交叉审核改动最大、也是最重要的设计点。背景：memcache 构造器 `lazy_init = lazy_init and _is_device_sdma()`，`lazy_init` 由 pool_worker.py:324 传入 `self.use_compress`（仅 DSV4 压缩模型）；lazy 模式下 `exists`/`batch_get_key_info` 在未初始化时短路返回全 0 / 空列表。

```python
# base.py — Backend 默认（所有后端，mooncake/yuanrong 继承 no-op）
def on_worker_ready(self) -> None:
    """kv caches 注册完成后、传输线程启动前调用。需要 eager 初始化的后端覆写。"""

# memcache_backend.py
def on_worker_ready(self) -> None:
    # lazy_init（compress + device_sdma）刻意延迟：exists/batch_get_key_info
    # 依赖"未初始化即全 miss"短路，无条件 eager 会击穿。
    if self._lazy_init:
        return
    self.ensure_initialized()

# backend/gva_protocol.py — GVASession.__init__ 首行
store.ensure_initialized()
```

pool_worker.py:816-817 特判删除，改为无条件 `self.m_store.on_worker_ready()`（在 register_buffer 之后、_start_kv_transfer_threads 之前）。

**四情形等价表**：

| 情形 | 现 main | 修正后 | 判定 |
|---|---|---|---|
| mooncake / yuanrong | 无 eager | 继承 no-op | 零变化 |
| memcache 非 lazy（含 GVA 非 compress） | 构造器已初始化 | `ensure_initialized` 幂等返回 | 零变化 |
| memcache + lazy + 非 GVA | :816 gate 不触发，首次传输自愈 | gate 同样不触发，短路契约完整保留 | 零变化 |
| memcache + lazy + GVA | :816 **eager**（既定意图，见 #12854 commit 80d3bc6 "Initialize the lazy MemCache store when layerwise GVA buffers are registered so the first batch_alloc call cannot observe an empty backend"） | `GVASession.__init__` 显式 ensure，fail-fast 保持 | 零变化 |

等价性依据：pending-buffer 对称设计（`register_buffer` set pending + `_register_buffers_if_needed` 立即 flush，memcache_backend.py:119-130）使两种构造顺序收敛同一终态；`ensure_initialized`（:62-73）自身末尾补注册，不依赖被 #14697 删除的 `init_store`。

**为什么最小方案（仅 gate、无 GVASession self-ensure）被禁止**：GVA + lazy 组合下 store 永不初始化 → load 路径 `batch_get_key_info` 空返回 → `zip` 空转（:1397）→ `if valid_keys:`（:1417）跳过租约 → `load_block_gvas_np` 全 0 向量（:1502）——**静默功能失效**，不崩溃、无 warning（仅 :1490 debug 级 `valid_gvas=0` 可观测），可穿透 UT 回归绿与冒烟；纯 consumer（PD 分离 **D 节点**，`if new_keys:` 保护下无 save 分配）store 永不初始化、load 永久失效。scheduler 侧不受影响（`create_scheduler_client` 恒 eager 构造，lazy 短路分支不可达——两后端均不传 lazy_init）。

### 3.4 关键平移映射（供 PR 描述与 #12854 rebase）

| main 位置 | 新位置 |
|---|---|
| pool_worker._make_layerwise_gva_key（:1113-1123） | gva_protocol.py `GVAKeyFactory.full_key` |
| pool_worker._make_layerwise_partial_key（:1125-1133） | `GVAKeyFactory.partial_key` |
| pool_worker._get_partial_block_index（:1097-1111） | metadata.py `get_partial_block_index` |
| pool_worker._refresh_allocated_gvas（:1134-1149） | `GVASession._refresh_allocated_gvas` |
| pool_worker._alloc_gvas_for_save（:1151-1313） | `GVASession.alloc_gvas_for_save` |
| pool_worker._prepare_load_gvas（:1315-1515） | `GVASession.prepare_load_gvas` |
| pool_scheduler._make_layerwise_gva_keys_for_hit_check（:306-317） | `GVAKeyFactory.hit_check_keys` |
| pool_scheduler._get_layerwise_gva_hit_tokens（:319-387） | `GVAHitChecker.hit_tokens` |
| pool_worker.py:87-93 常量 ×4 | gva_protocol.py（逐字） |
| kv_transfer.LayerBatchBuilder（:39-303） | gva_threads.py（PR-B） |
| kv_transfer._split_transfer_packets/_batch_copy_with_limits（:393/:426） | gva_threads.py（PR-B） |
| kv_transfer.KVCacheStoreLayerSending/RecvingThread（:1271/:1405） | gva_threads.py（PR-B） |

平移纪律：**逐字符一致、禁止顺手重构**（f-string、日志文案一律原样）；key 构造输出以字节级快照 UT 锁定；ReqMeta 字段写入不变；`process_layer_data` 内 prepare_load（:1609）→ alloc_save（:1610）顺序不变；hit_tokens 内 `_get_or_create_request_tracker` 移至调用点（行为等价）。

挂接要点（PROTO）：`GVASession` 构造点放 `_init_layerwise_config`（:348-442）末尾（`_init_backend` 先于其执行，m_store 已就绪）；调用点 `process_layer_data` 委托 `_gva_session.prepare_load_gvas(requests, on_invalid_blocks=self._report_invalid_blocks)` + `alloc_gvas_for_save(requests)`；方法内 `if not self.use_gva_layerwise: return` gate 删除（构造点已 gate），`kv_role == "kv_consumer"` 与 `tp_rank % put_step` gate 保留（协议语义）。GVAHitChecker 构造需 `head_or_tp_rank=0, head_or_tp_ranks=tp_size // put_step`（hit-check 不使用单 rank）。

**定位表述（诚实性）**：此为"行为保持的重新封装"，非纯移动——新类型 5 个（GVAKeyFactory/GVASession/GVAHitChecker/GVALayerwiseCapable/GVALayerwiseThreadContext）+ 工厂函数 2 个，真正新增生产代码 ~150 行（PR-A）+ ~120 行（PR-B），其余为平移与 UT。PR 描述按此口径，reviewer 以新旧代码块机械 diff + ~150 行人工审的方式验收。

---

## 4. PR 拆分与执行时序

### 4.1 二 PR 结构

| PR | 内容 | 规模 | 主题 |
|---|---|---|---|
| **PR-A** | CAP + IFACE + KEY + LIFE + PROTO | ~1465 行（~150 新增 + ~500 平移 + ~600-700 UT + ~650 删除另计） | GVA 协议收敛 → gva_protocol.py |
| **PR-B** | THRD + TRACE | ~880 行（~120 新增 + ~700 平移 + ~150 UT） | 线程收敛 → gva_threads.py |

超 1000 行约束的处理（原验收标准 11）：PR-A 以构成论证（~80% 平移 + UT 保护，人工审查面收敛于 ~150 行）请求 reviewer 按审查面评估；回退切点为 KEY 独立成前置小 PR（~350 行）。

### 4.2 执行时序（含 #14697 弹性）

```
#14697 已合入 → PR-A 按 v3 §2.4 适配（scheduler 侧 CAP 引用局部变量 backend_name）
#14697 未合   → PR-A 直接基于当前 main 开工（两处 use_gva_layerwise 定义行
               逐字符一致，设计等价适用；#14697 后合 rebase 账单：
               2 处强制冲突 + 1 处单行适配，分钟级，方向最优）
唯一避免     → 两 PR 并行流转互追 rebase
→ PR-B（依赖 PR-A 的 GVALayerwiseCapable）
```

#14697 rebase 账单（逐 hunk 核实）：worker `_init_kv_transfer_config`（上下文冲突，drop hunk 即解）、scheduler `__init__`（`use_gva_layerwise` 行被 #14697 改为局部变量引用——与 CAP 替换强制冲突，rebase 者被引导至该行改为局部变量即可）；其余 hunk（dp_rank/use_mla/`_init_backend` 签名/:274 rename/init_store 删除/test mock 改名）自动合入或无交集。

### 4.3 对 open PR 的策略

| PR | 状态 | 策略 |
|---|---|---|
| #14697（backend 路径简化，本人） | open | 非硬前置，两序皆可（§4.2） |
| #12711（Qwen3.5 layerwise + 412b157 回归修复） | open | PR-A 含等效回归修复，谁先合谁生效，后到者 rebase 去重 |
| **#12854**（layerwise transfer rework，ader47） | open（与 main 冲突） | **抢先合入，由其 rebase**。正当性：①行为保持重构先于语义变更重做，后者 rebase 面向新基座是机械工作；②PR-A 携带 main 活回归修复，有独立合入价值；③反向等待等于被 open PR 无限期绑架。**不做** Preparer/GVASession 组合或取代的预期承诺（职责重叠的两种封装，结局不可预期）。缓解：PR 描述附 §3.4 映射表 + @ 作者说明收敛方向同向 |

---

## 5. UT 计划（终版五条 + 模块级）

| # | UT | 防护目标 | PR |
|---|---|---|---|
| 1 | `_lazy_init=True` 下 `on_worker_ready` 不触发初始化（mock `_setup_store` 断言未调用） | LIFE gate | A |
| 2 | `GVASession.__init__` 触发 `ensure_initialized` | GVA+lazy eager 前置条件 | A |
| 3 | `exists` lazy 未初始化时全 miss 短路回归 | 短路契约固化，防未来 PR 击穿 | A |
| 4 | `batch_get_key_info` 空返回契约 golden（lazy 未初始化返回 []，初始化后返回真实结果） | 防把降级信号当 bug "修复"或当正常态沿用 | A |
| 5 | **load 路径非零 gva 端到端断言**（含已缓存前缀的请求，断言命中块 gva > 0） | **静默失效唯一直接探针** | B |

模块级：test_backend.py（capabilities 双向一致性：`issubclass(cls, GVALayerwiseCapable) == ("gva_layerwise" in capabilities)`；`use_gva_layerwise()` 派生真值表）；test_gva_protocol.py 新建（key 格式字节级快照；GVASession alloc 跳过/缓存命中/partial 分配失败；GVAHitChecker 全组命中取 min/MTP safe extent）；test_ascend_store_connector.py（MultiConnector `use_gva_layerwise` 属性存在性回归，#14465 场景复现）；test_metadata.py（`get_partial_block_index` 用例迁入）；test_pool_worker/test_pool_scheduler 相关用例迁移适配（test_mtp_gva_prepare_uses_safe_extent_not_store_skip_extent :1166、test_evicted_allocated_gva_is_reallocated :1423、test_layerwise_gva_hit_tokens :800、test_layerwise_mtp_hit_uses_safe_load_extent :139）。

---

## 6. 验收标准

**功能**：
1. UT 全量回归（tests/ut/distributed/ascend_store）
2. mooncake 非 layerwise 冒烟：现有 e2e 无回归
3. memcache layerwise 冒烟：TP=4 + layerwise + 长前缀 load 复测通过
4. **MultiConnector PD 冒烟**（P 4×TP + D consumer，proxy，GSM8K prefix-cache，请求成功率 100%）——回归修复生效点，必须实测而非仅 UT
5. main 回归（#14465 引入）消除，且回归用例固化（UT 模块级末条）

**结构**：
6. `use_gva_layerwise` 语义定义仅 1 处；`grep '== "memcache"'` 业务判断在 backend 目录外零残留（仅 backend/__init__.py 注册表）
7. PR-A 后 pool_worker/pool_scheduler 无 GVA 协议方法；PR-B 后 kv_transfer.py 无 GVA 线程类、`.store.` 直捅清零
8. base.py 通用接口与 GVA 扩展接口分离（GVALayerwiseCapable + 一致性 UT）

**兼容性**：
9. **存储 key 格式不变**（快照 UT 锁定，已部署集群数据升级后仍可命中）
10. 配置接口不变（`use_layerwise` / `backend` / `layerwise_max_transfer_blocks` 等 extra_config 项及默认值）
11. ReqMeta 字段结构不变，`kv_transfer.py` 消费方零改动

**工程**：
12. Conventional Commits + Signed-off-by；ruff / mypy（3.10-3.12）通过
13. PR-A 超限按构成论证（§4.1），回退切点 KEY 独立

---

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| #12854 双向大冲突（PR-B 同区 ~700 行移动） | 抢先合入 + §3.4 映射表降 rebase 成本 + @ 作者沟通；接受最坏情形（我方 rebase），方向正确性不依赖合入顺序 |
| #14697 与 PR-A CAP 区域相邻 | §4.2 两序机制：先合则适配局部变量，后合则分钟级 rebase |
| **lazy_init 短路契约被后续 PR 无意击穿** | UT 3/4 固化为回归测试（这是连续四份材料漏检的教训） |
| **静默失效穿透出口验证** | UT 5 唯一直接探针（PR-B）；debug 日志 `valid_gvas=0`（:1490）为人工排查点 |
| key 格式漂移（线上数据 miss） | 快照 UT 前置（PR-A KEY 模块先于 PROTO 合入）+ 逐字符平移纪律 |
| GVASession 构造点初始化顺序（grouped_block_size / group_block_len / page_size_bytes 就绪性） | 构造点放 `_init_layerwise_config` 末尾 + 构造 UT 覆盖 |
| ready_event 时序（send 线程 ready 后才建 recv 的 save_failure_checker） | 工厂调用顺序逐行保持现结构（:478-544 的 start/wait 顺序）；PR-B 冒烟覆盖 save 失败传导 |
| `_batch_copy_with_limits` 提取后 key-mode 线程误引用 | 已核实仅 GVA 路径调用（:1363/:1560）；提取后 grep 复核 |
| on_worker_ready 顺序变化破坏 memcache buffer 注册 | §3.3 四情形等价表 + UT 1/2 专项覆盖 |
| ctx dataclass 15+ 参数遗漏 | 工厂签名与现线程 `__init__` 一一对应 + mypy + UT 参数映射断言 |
| 平移过程顺手重构引入行为漂移 | 快照 UT 前置 + "逐字符一致"纪律 + 新旧代码块机械 diff |
| open PR 交织（rebase 顺序） | §4.2/§4.3 策略矩阵 |

---

## 8. 审计链与五轮交叉审核总账

```
需求分析 → p0/p1p2 方案 → v2 → 一轮审核 → v3 → 二轮审核 → v4 → 三轮审核
→ v5 → 四轮审核 → 本文档（终稿）
```

| 轮次 | 首检出的缺口 | 该轮引入的错误（次轮修正） |
|---|---|---|
| 一轮 | PR3 × #12854 冲突低估；6 PR 过度碎片化 | 283 行归因错误 |
| 二轮 | LIFE lazy_init 击穿（一行修复有盲区） | assert 崩溃推演短路 |
| 三轮 | 最小方案危险低估（load 路径功能破坏） | —（被 v4 证伪上项） |
| 四轮 | scheduler 前提空集证伪 + P/D 颠倒 + 行号校准 | v4 §4.1 两处表述（被 v5 取代） |
| 五轮 | v4 §4.1 表述与 #14697.diff 不符 | — |

**净结论**：发现严重度单调递减（冲突低估 → 功能破坏 → 细节证伪 → 空前提 → 勘误完备性）——收敛信号明确，文字阶段收口。本文档之后不再产出文字往返；交叉审核精力转移至 **PR-A/PR-B 描述的 self-review**，重点两处：① UT 5（静默失效探针）是否真实进入出口验证；② §3.4 映射表与最终 diff 的逐项一致性。

**执行就绪（终态）**：PR-A 可直接开工（#14697 两序皆可，含 LIFE 严格版 + 五条 UT + MultiConnector 回归修复）→ PR-B。无遗留未决项。
