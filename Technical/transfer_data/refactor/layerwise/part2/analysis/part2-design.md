# Layerwise 逻辑收敛 part2 — 方案设计

> 依据：[requirements_analysis.md](requirements_analysis.md)（实测基线与 R1-R4 需求）
> 前序资产：`archive/develop/PR-B record.md`（#15307，gva_threads.py 891 行 + test_gva_threads.py 568 行）、`archive/develop/PR-15367 record.md` §7.1（协议对象两件套设计——**按返工后形态重新推导，不照搬**）
> 红线：PR ≤ 1000 行（生产+UT）；key 快照逐字节不变；零行为变与行为变更严格分批

---

## 0. 总体结构：两个 PR，风险递进

| 批次 | 内容 | 性质 | 依赖 |
|---|---|---|---|
| **part2a** | GVA 传输三块物理搬迁（LayerBatchBuilder + 两个 Layer 线程 → backend 侧）+ UT 迁移 | **零行为变**（逐字符平移） | 无 |
| **part2b** | 协议对象收编运行期编排（worker 3 方法 + 8 分支、scheduler 2 分支）+ base.py 5 桩删除 | **行为变更**（结构重组，语义保持） | part2a 合入 |

顺序理由：part2a 是纯物理移动，验证手段简单（树级对比、UT、e2e），先走可把 kv_transfer.py 清场；part2b 的协议对象需要干净的 backend 侧才能落位。若 part2a 尺寸超红线，备用拆分见 §3。

## 1. part2a：GVA 传输四块搬迁

### 1.1 目标形态

```
backend/
├── __init__.py          # registry 不变（get_layerwise_protocol 已就位）
├── base.py              # 不动（5 桩留待 part2b）
├── memcache_backend.py  # 不动（协议函数已就位）
└── <GVA_THREADS>.py     # [新] LayerBatchBuilder + 两线程 + 上下文/工厂
```

依赖方向：`<GVA_THREADS>.py` → import `kv_transfer.KVTransferThread`（具体→通用，无环，同 #15307 B-5 论证）；`pool_worker` → import `<GVA_THREADS>` 的工厂；`kv_transfer.py` 恢复纯通用（8 类 → 5 类）。

### 1.2 决策 D1：模块归宿与命名（**待 reviewer 确认项**）

reviewer 意见 #1（"gva_protocol.py 应移到 memcache backend 里"）针对的是协议函数；线程类的归宿没有直接表态，但精神一致：memcache 专属代码在 memcache 的地盘。三个选项：

| 选项 | 形态 | 论证 |
|---|---|---|
| A | `backend/gva_threads.py`（#15307 原案名） | 文件级隔离、import 图干净；但 backend/ 根下出现 memcache 专属文件，与"registry 字段 + memcache_backend 承载"的 part1 格局不完全对称 |
| B（推荐） | `backend/memcache_gva_threads.py` | 文件名显式标记 memcache 归属；registry 的 `layerwise_protocol` 解析仍指向 memcache_backend 模块（协议函数面），线程模块由 memcache 协议装配路径引入 |
| C | 全部塞进 `memcache_backend.py` | 最彻底，但 290 → 1100+ 行单文件，不可维护 |

B 的理由：memcache 的 layerwise 协议物理上需要两个文件（协议函数 + 传输线程），文件名前缀声明同一归属；通用层的解析入口（`get_layerwise_protocol`）不必感知线程模块——worker 侧 import 由搬迁后的装配代码持有（见 1.3）。若 reviewer 偏好 A，改一行文件名即可，无结构差异。

### 1.3 commit 序列（零行为变）

| # | commit | 内容 | 验证 |
|---|---|---|---|
| a1 | 纯新增 | `<GVA_THREADS>.py`：LayerBatchBuilder + `_LayerTransferThreadBase`（接收通用基类 `KVTransferThread` 的线程骨架，持有 `_split_transfer_packets`/`_batch_copy_with_limits`——GVA 包切分与限流拷贝从通用基类下沉于此，#15307 B-2 设计）+ `KVCacheStoreLayerSendingThread`/`RecvingThread` + `LayerwiseThreadContext` + 工厂×3（`build_group_layer_builders`/`create_layer_sending_thread`/`create_layer_recving_thread`，中性名） | main 零变化；新增模块自带 UT 副本 |
| a2 | 切换+删除 | pool_worker `_start_kv_transfer_threads` layerwise 分支改走 ctx+工厂；kv_transfer.py 删四块（:39-304/:393-475/:1271-1407/:1408-1621，通用基类 `KVTransferThread` 瘦身回线程骨架）；`.store.batch_copy` 直捅（:468）随两方法下沉消失——backend 侧 `m_store.batch_copy` 直转；test_kv_transfer.py 删 GVA 用例 | 全量 UT + grep（kv_transfer 无 `LayerBatchBuilder`/`Layer*Thread`/`_batch_copy_with_limits` 残留） |
| a3 | 收尾 | docstring 中性化、工厂参数映射断言、lint/mypy | ruff + mypy（CI 同款） |

a1 的"纯新增"使中间树可编译可测；a2 的删除量（-483 生产 + -330 UT）与新增在 a1，单 commit 面可控。

**行为保持论证手段**（沿用 #15307 已验证的方法）：a1 的新类与 kv_transfer 原类做 AST 方法体逐字符对比（断言形态差异白名单化：#15307 时代的 `assert isinstance(m_store, Backend)`/`_gva_store` property 均因断言已删而不需要——5 个 batch 方法现在全在 Backend ABC 上，mypy 直接过，这是返工带来的简化，实测为准）。

### 1.4 #15307 资产回收评估

| 资产 | 状态 | 回收方式 |
|---|---|---|
| gva_threads.py 891 行结构（类切分/中间基类/ctx/工厂） | 结构有效 | 骨架照搬；宿主名按 D1 |
| 逐字符平移的类体 | 大部分有效 | 需 rebase 到 main 实态：#15367 后行号已漂移、断言已删、`use_gva_layerwise`→`use_layerwise_transfer` 改名波及、#12711 的 GDN 扩展触碰 pool_worker 装配区 |
| `_DualSpecStore` mypy 修复、`spec=MemcacheBackend` fixture | **作废** | 服务于已删除的 isinstance 断言；现 fixture 是普通 MagicMock |
| test_gva_threads.py 568 行 | 用例集有效 | 同样需按 main 实态重对（mock 面从 session/hit_checker 部分回退为直接 mock） |

结论：**结构级回收、代码级重对**。回收率预估：设计/验证方法 100%，代码体 ~70%（剩余为适配工时，非重写）。

### 1.5 尺寸预估与备用拆分

平移量：生产 +483/-483、UT +330/-330，另有 ctx/工厂新增 ~120 行与工厂参数测试 ~80 行。若 rename/copy 检测不理想导致 GitHub 计数超 1000：**按消费者分层拆**——a1'（LayerBatchBuilder 单独先行，纯函数集合，可独立测试）+ a2'（线程两块）。Builder 先行有独立价值（无状态、零线程 fixture），拆分不产生怪异中间态。

## 2. part2b：协议对象收编（方案要点，实施前再细化）

### 2.1 协议对象形态（决策 D3）

part1 档案 §7.1 件二的方向仍然成立，但宿主按返工后事实修正：

- **协议对象与线程同宿主**（`memcache_gva_threads.py` 或按 D1 结果），不回到独立 gva_protocol.py——reviewer #1 已否决该形态
- 对象面按消费者需要起（part1 返工确立的原则）：worker 侧 `prepare_save(requests)` / `prepare_load(requests)` / `refresh_allocation(keys)`；scheduler 侧 `probe_hits(...)`
- worker 构造期 `self.layerwise_protocol`（模块）保持，协议对象由 worker 装配并传入线程 ctx——与 part2a 的 `LayerwiseThreadContext` 天然衔接

### 2.2 运行期分支的下沉边界

8 处 worker 分支不追求全消（"分支变对象存在性"只对 GVA 编排分支成立）；下沉判据：**分支体是 GVA 语义 → 进协议对象；分支体是通用 layerwise 流程（如线程启动、waiter 接线）→ 保留在 worker，条件仍用 `use_layerwise_transfer`**。逐分支归类在 part2b 实施文档中列表论证。

### 2.3 base.py 5 桩删除

随协议对象收编 GVA store 调用后删除（`batch_get_key_info`/`batch_alloc`/`batch_add_lease`/`batch_remove_lease`/`batch_write_finish` 退出 Backend ABC，MemcacheBackend 保留实现供协议对象调用）。这是行为变更 commit，单独成 commit，排在协议对象全量切换验证之后。

### 2.4 part2b 的 commit 序列原则

行为变更按"先立管道（对象构造+并存）→ 切换（分支改走对象）→ 收尾（删桩）"三段，每段独立可验证；语义保持靠 e2e 判据（hit_tokens/valid_gvas）而非纯 UT（UT 对行为变更只有锚定价值）。

## 3. 验证策略（两批共用）

| 层 | 手段 | 判据 |
|---|---|---|
| L1 UT | 本地 shim（`run_ascend_store_ut.py`）→ 165 全量 `tests/ut/distributed/ascend_store/` | 全 passed（既有 coordinator 2 例 stub 失败按惯例排除，CI 有真实 vllm） |
| L2 行为保持 | part2a：AST 逐字符对比（a1/a2 间）+ 中间 commit 树级 diff 审查；part2b：语义等价论证表 | key 快照测试逐字节不变（part1 已合入的锚点，零触碰） |
| L3 e2e | 165 三场景，复用 `archive/test/scripts/`（rerun_e2e.sh 编排） | S1 无 AttributeError + 成功率 100%；S2 `hit_tokens>0` + `valid_gvas>0`；S3 无 layerwise 标记 + 三维证据链 |
| L4 静态 | ruff 0.14.0（CI 同版本）+ mypy 3.10/3.11/3.12 | 全绿 |
| L5 度量 | R4 终态 grep：通用层类/方法/分支级 gva 归零（L4 白名单：metadata 字段等，逐项登记） | 归零清单入 PR 描述 |

## 4. 风险与开放问题

1. **D1 命名**待 reviewer 确认（A/B 两选项一行之差，不阻塞开工——先按 B 实施纯新增 commit）。
2. **同区 open PR 竞争（实测 2026-09-03，开工前必须重查）**：
   - **#12854**（layerwise transfer rework，OPEN）：同区巨型改动——kv_transfer +243/−403、pool_worker +392/−684、scheduler +87/−131，并新增通用层文件 `layerwise_transfer.py`（513 行）与 `layerwise_cache_layout.py`（+316，ADDED 表明其分叉早于该文件进 main，基线已老）。它若激活，Layer 线程区会被 rework 替换，part2 需基于新形态重排。处置：开工前确认其活跃度（updatedAt 停更）与作者意图，必要时在 PR/issue 上与维护者对齐"结构收敛 vs 调度重写"的先后；策略沿用 #15307 时代结论（收敛 PR 抢先合入、rework rebase）但需重新确认。
   - **Pz1116 性能四连发**（#14772 value preparation 向量化 / #14773 load receivers 并行化 / #15242 lookup+save 开销削减 / #15243 lookup 移到 scheduler）：全部触碰 kv_transfer/worker 编排区，且 #14773 极可能直接改 `KVCacheStoreLayerRecvingThread`。这些是 reviewer 本人的 PR，大概率先于 part2 合入——**part2 基线必须取它们的合入后 main，行号与本方案全部重测**。
   - #15255/#15442/#15507（layerwise 布局/key 区）与 #14763（waiter 修复）与 part2a 主区交叠小，rebase 级冲突。
3. **#12711 的 GDN 路径**：hybrid 模型显式 wait/save 走通用 layerwise 流程，e2e S2 场景需确认不因搬迁破坏（UT 层 test_layerwise 相关用例覆盖）。
4. 165 服务器状态（HBM 残留/容器基线）在实施前按惯例预检（archive 教训已入 memory）。
5. **旧 PR 清理前置**：#15277/#15307（本作者，均 OPEN）已被 #15367 取代，开工 part2a 前关闭并留言指向，避免 reviewer 困惑（两 PR 的验证清单已被复用，关闭无损失）。
