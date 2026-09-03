# PR-B 执行计划（施工版）

> 输入：[最终需求分析和方案设计.md](../最终需求分析和方案设计.md)（唯一执行基准）§3.2 THRD/TRACE、§3.4 映射表、§5 UT 5、§6 验收标准 7、§7 风险表
> 施工仓：`d:\lzy\project\kv_pool\code\vllm-ascend`，分支 `refactor_layerwise_B`（= `refactor_layerwise` @ a0bb5dcf2 = PR #15277 当前 HEAD，工作区干净）
> 前置：PR-A 已提交 upstream（#15277，Open）。本分支基于其上开工；**若 PR-A review 产生修订，先在本分支 rebase 对应修订再继续**（避免两 PR 互追）
> 验证：165 服务器另行执行（听用户指示），本计划只含代码施工与本地 UT
> 开工前代码实况锚定完成（本 session 逐区域重读），行号均为 **PR-A 合入后实测值**（与终稿 §3.4 的 main 行号不同，本表为准）

---

## 0. 施工决策点（终稿未展开的 5 个实现细节，按代码实况裁定）

| # | 问题 | 裁定 |
|---|---|---|
| B-1 | `batch_copy` 接口化后 `_batch_copy_with_limits` 内 `self.m_store` 类型为 `Backend`（无 batch_copy），mypy 报错 | 调用前 `assert isinstance(self.m_store, GVALayerwiseCapable)` 收窄。运行时新增 fail-fast 断言（非 GVA 后端误挂 GVA 线程立即 AttributeError→AssertionError，性质同 PR-A GVASession self-ensure），PR 描述说明 |
| B-2 | `_split_transfer_packets`（:393-425）与 `_batch_copy_with_limits`（:426-478）现挂在通用基类 `KVTransferThread` 上，用 `self.num_addrs_per_block`（:325 通用基类构造）与 `self.m_store` | gva_threads.py 内建中间基类 `_GVALayerTransferThreadBase(KVTransferThread)`：持有 `max_transfer_blocks`/`max_transfer_bytes`，两方法下沉其中（逐字符平移，仅 :453 assert 与 :468 直捅按 B-1/C1 改）。两 GVA 线程类改继承它。不做模块函数化（需传一长串状态参数，反而增加映射面） |
| B-3 | ready_event 构造归属：终稿要求"工厂调用顺序逐行保持现结构（start/wait 顺序）" | ready_event 仍由 pool_worker 构造、传入工厂；`start()` + `ready_event.wait()` 留在 pool_worker。工厂只负责组装线程对象（终稿"模块级工厂"最小语义） |
| B-4 | `group_builders` 现状：send（:499）与 recv（:539）各调一次 `_build_group_layer_builders()`，两线程持**独立** builder 实例（`_ensure_buf` 有内部 buffer 状态，共享有线程安全隐患） | 保持两次独立构建。`build_group_layer_builders()` 工厂函数被 send/recv 各调一次，builder 实例隔离语义不变（禁止"优化"为共享） |
| B-5 | C2"纯新增"模式下新旧类并存（~700 行副本），import 方向需无环 | 依赖方向：`kv_transfer.py（KVTransferThread）← import ← backend/gva_threads.py ← import ← pool_worker.py`。kv_transfer.py 不 import gva_threads（C3 删旧类前旧类仍在 kv_transfer 内自洽）。已核实无环 |

## 1. 平移对象锚定（PR-A 后实测行号）

### 1.1 kv_transfer.py（现 1615 行）→ backend/gva_threads.py

| main 位置（现实测） | 内容 | 规模 | 新位置 |
|---|---|---|---|
| :39-303 | `LayerBatchBuilder`（含 `_ensure_buf`/`_dedupe_transfer_blocks`/`_build_transfer_arrays`/`_require_request_arrays`/`build_shared`/`build_addrs`/`build`） | ~265 行 | gva_threads.py 整类平移 |
| :393-425 | `_split_transfer_packets` | ~33 行 | `_GVALayerTransferThreadBase` 下沉（B-2） |
| :426-478 | `_batch_copy_with_limits`（:453/:468 直捅） | ~53 行 | 同上；C1 先接口化、C2 平移 |
| :1271-1404 | `KVCacheStoreLayerSendingThread`（含 `build_shared_data` :1321） | ~134 行 | gva_threads.py 整类平移 |
| :1405-1600 | `KVCacheStoreLayerRecvingThread`（含 `build_shared_data` :1459、`_get_h2d_stagger_delay_us`/`_stagger_h2d_submit`） | ~196 行 | gva_threads.py 整类平移 |

**留在 kv_transfer.py 不动**（已逐一核实非 GVA 专属）：`_circular_shift`（:33，key-mode：pool_worker :931-934 与 KeyLayerRecving :1253-1255 使用）、`KVTransferThread` 通用基类（:305-597 其余部分）、`KVCacheStoreSendingThread`（:598）/`KVCacheStoreRecvingThread`（:868）/`KVCacheStoreKeyLayerSendingThread`（:1015）/`KVCacheStoreKeyLayerRecvingThread`（:1169）、`record_failed_blocks`（:1601，仅 key-mode recv :972/:988 与 pool_worker :945/:960/:1548/:1552 调用，GVA 线程零调用）。

### 1.2 pool_worker.py 引用点（C3 切换面）

| 位置（现实测） | 内容 | C3 动作 |
|---|---|---|
| :44-51 | import（`KVCacheStoreLayerRecvingThread`/`KVCacheStoreLayerSendingThread`/`LayerBatchBuilder` 等） | GVA 三符号改从 `backend.gva_threads` import；key-mode 符号（`_circular_shift`/`record_failed_blocks`/`KVCacheStoreKeyLayer*` 等）保持 |
| :454-468 | `_build_group_layer_builders` | 方法删除，逻辑平移为 gva_threads 模块级 `build_group_layer_builders()` 工厂 |
| :483-502 | GVA send 线程构造 + `start()` + `ready_event_sending.wait()` | 换 `create_gva_sending_thread(ctx, ready_event_sending)`；start/wait 原地保留（B-3） |
| :521-544 | GVA recv 线程构造（含 `save_failure_checker=self.kv_send_thread.raise_if_failed if ... else None`） | 换 `create_gva_recving_thread(ctx, ready_event, save_failure_checker=...)`；**send-ready 后才建 recv 的顺序逐行保持**（终稿风险表） |
| :1113-1149 | `_build_shared_save_data` GVA 分支（isinstance `KVCacheStoreLayerSendingThread`） | 仅 import 源变化，逻辑零改动 |
| :1170 | `_build_shared_load_data`（isinstance `KVCacheStoreLayerRecvingThread`） | 同上 |
| :1628 | `isinstance(send_thread, (KVCacheStoreSendingThread, KVCacheStoreLayerSendingThread))` | 同上（类名不变，import 源变化） |

### 1.3 ctx 参数映射表（防 15+ 参数遗漏，终稿风险表指定手段）

线程构造参数 → `GVALayerwiseThreadContext`（公共 12）+ 工厂入参（独有）：

| 参数 | send(:1272-1288) | recv(:1406-1427) | 归属 |
|---|---|---|---|
| m_store / token_database / block_size / tp_rank / tp_size / dcp_size / page_size_bytes / num_layers / layer_save_finished_events / sync_save_events / max_transfer_blocks / max_transfer_bytes | ✓ | ✓ | **ctx ×12** |
| ready_event | ✓ | ✓ | 工厂入参（B-3） |
| group_builders | ✓ | ✓ | 工厂内调 `build_group_layer_builders()`（B-4） |
| get_event / layer_load_finished_events / h2d_stagger_us / external_slot_release_waiter | — | ✓ | **recv 工厂独有 ×4** |
| save_failure_checker | — | ✓ | recv 工厂入参（保持 `send 线程 raise_if_failed` 闭包注入） |

ctx 构造点：`_start_kv_transfer_threads` 的 layerwise 分支内（`use_layerwise` 判定后）。所需属性就绪性已核实：`token_database`（__init__）、`h2d_stagger_us`/`layerwise_max_transfer_blocks`/`layerwise_max_transfer_bytes`（:174-176）、`external_slot_release_waiter`（:470-471，connector 在线程启动前调用 setter——现 main :540 已在读，语义等价）。

## 2. Commit 序列

### C1 — batch_copy 接口化（封堵直捅，独立小 commit）
```
refactor(kv_pool): expose batch_copy on GVALayerwiseCapable to seal direct store access
```
- `backend/base.py`：`GVALayerwiseCapable` +第 6 个 @abstractmethod `batch_copy(gvas, addrs, sizes, direction) -> int`
- `backend/memcache_backend.py`：实现 `batch_copy`——`assert self.store is not None; return self.store.batch_copy(...)` 直转发，**不加 ensure、不加 lazy 检查**（现 :468 直捅即无检查，行为保持；GVA 线程仅在 on_worker_ready / GVASession ensure 之后运行）
- `kv_transfer.py` `_batch_copy_with_limits`：删 :453 assert，:468 `self.m_store.store.batch_copy(...)` → `self.m_store.batch_copy(...)`（前置 B-1 isinstance 收窄）
- UT：test_backend.py GVALayerwiseCapable 一致性断言自动覆盖新方法（双向一致性已存在）；补 MemcacheBackend.batch_copy 转发断言（FakeStore 捕获参数）；此 commit 后 `.store.` 直捅全仓清零（grep 断言）

### C2 — gva_threads.py 纯新增（main 行为零变化）
```
refactor(kv_pool): add GVA layerwise transfer thread module with thread context factory
```
- 新增 `backend/gva_threads.py`：
  - `_GVALayerTransferThreadBase(KVTransferThread)`（B-2）：`_split_transfer_packets`（:393-425 逐字符）+ `_batch_copy_with_limits`（C1 后版本逐字符）+ `max_transfer_blocks`/`max_transfer_bytes` 构造
  - `LayerBatchBuilder`（:39-303 逐字符整类）
  - `KVCacheStoreLayerSendingThread` / `KVCacheStoreLayerRecvingThread`（整类平移，继承中间基类；`__init__` 签名不变——ctx 由工厂拆开传入，线程类保持显式参数以兼容既有 UT 直接构造）
  - `GVALayerwiseThreadContext` dataclass（§1.3 十二字段）
  - 工厂 ×3：`build_group_layer_builders(token_database, page_size_bytes, num_layers, num_kv_cache_groups, group_num_layers, group_block_len)`（pool_worker :454-468 逻辑平移）；`create_gva_sending_thread(ctx, ready_event)`；`create_gva_recving_thread(ctx, ready_event, save_failure_checker)`
- 新增 `tests/ut/distributed/ascend_store/test_gva_threads.py`：从 test_kv_transfer.py 迁移 GVA 用例副本——`TestLayerBatchBuilderOffsets`(:93)、`TestGVALayerTransferFailures`(:220)、`TestGVALayerReceivingTaskOwnership`(:284)、`TestLayerBatchBuilder`(:664)，及 `FakeStore`(:50)/`FakeTokenDatabase`(:69) 中 GVA 依赖部分（key-mode 用例继续用 test_kv_transfer 内原 helper，不共享文件，避免迁移期耦合）
- 此 commit 未接线（pool_worker 未引用），main 行为零变化；grep 断言 `from ...backend.gva_threads import` 仅测试文件引用

### C3 — 切换 + 删除（唯一大删改 commit）
```
refactor(kv_pool): delegate worker GVA transfer threads to gva_threads module
```
- `pool_worker.py`（§1.2 全表）：import 切换、`_build_group_layer_builders` 删除、线程构造点换工厂（start/wait 顺序原地保留）、isinstance 三处仅换 import 源
- `kv_transfer.py`：删 §1.1 五项（~700 行；`LayerBatchBuilder` import 从 metadata 链中清理时保留 key-mode 仍用的符号）、删 `_split_transfer_packets`/`_batch_copy_with_limits`（已随基类下沉）
- `test_kv_transfer.py`：删 GVA 用例四类（C2 已建副本）与 GVA 专用 helper
- 此 commit 后 grep 断言（终稿验收标准 7）：`kv_transfer.py` 无 `KVCacheStoreLayer*Thread`/`LayerBatchBuilder`/`_batch_copy_with_limits`；通用层（backend 目录外）`.store.` 直捅零残留；`LayerBatchBuilder` 定义仅 backend/gva_threads.py

### C4 — TRACE + UT 5 + lint 收尾
```
refactor(kv_pool): neutralize backend names in generic docstrings and add load path probe
```
- TRACE 三处 docstring 中性化（终稿 §3.2）：
  - attention_fence.py:31 `MemCache worker threads` → 中性表述（如 `backend worker threads`）
  - attention_fence.py:65 `per-layer gate for MemCache work` → 中性表述
  - layerwise_cache_layout.py:75 `extra config for the MemCache GVA layerwise path` → `for the GVA layerwise path`（backend 由 capabilities 表声明）
  - 仅 docstring 措辞，零代码变化
- **UT 5（静默失效唯一直接探针，终稿 §5）**：`test_gva_threads.py::test_load_path_end_to_end_nonzero_gva`——FakeStore `batch_get_key_info` 返回含有效 gva 的 key info → `GVASession.prepare_load_gvas`（复用 test_gva_protocol 的 session 构造）→ 断言 `request.load_block_gvas` 命中块 > 0 → recv 线程 `build_shared_data`/`LayerBatchBuilder.build` → 断言 `_batch_copy_with_limits` 收到的 gvas 非零（mock `batch_copy` 捕获参数）。覆盖链：协议 → 布局 → 传输；任一环节退化（lazy 空返回 / 租约跳过 / gva 全 0）即红
- UT 参数映射断言（终稿风险表）：工厂构造的线程属性与显式参数一一相等（send 14 项 / recv 19 项逐项 assertEqual）
- ruff / mypy 全绿（如需单独 style commit 则并入此处）

## 3. 文件清单

| 文件 | 操作 | 规模预估 |
|---|---|---|
| `backend/gva_threads.py` | **新增** | ~850 行（~700 平移 + 中间基类/ctx/工厂 ~100 新增 + docstring） |
| `backend/base.py` | 改 | GVALayerwiseCapable +`batch_copy` ~6 行 |
| `backend/memcache_backend.py` | 改 | +`batch_copy` ~5 行 |
| `kv_transfer.py` | 改 | 1615 → ~915 行（删 ~700） |
| `pool_worker.py` | 改 | 删 `_build_group_layer_builders`，构造点换工厂，import 调整（净 ~-25 行） |
| `attention_fence.py` / `layerwise_cache_layout.py` | 改 | docstring 3 处 |
| `tests/ut/distributed/ascend_store/test_gva_threads.py` | **新增** | ~450 行（迁移用例 + UT 5 + 参数映射断言） |
| `tests/ut/distributed/ascend_store/test_kv_transfer.py` | 改 | 删 GVA 用例与专用 helper（~-250 行） |
| `tests/ut/distributed/ascend_store/test_backend.py` | 改 | +batch_copy 转发断言（一致性断言既有自动覆盖） |

规模口径（对齐终稿 §4.1"~880 行 = ~120 新增 + ~700 平移 + ~150 UT"）：本计划 UT 侧超预算（迁移副本 + 探针 ~450 行），新增生产代码 ~100 行与终稿一致；PR 描述按实际构成论证。

## 4. 验证出口（PR-B 合入门槛，165 服务器执行，听指示）

1. `tests/ut/distributed/ascend_store` 全量绿（含 UT 5；本地用 `python d:\lzy\project\kv_pool\run_ascend_store_ut.py ... --noconftest -q -p no:cacheprovider`，test_layerwise_cache_layout 仍归 165）
2. **memcache layerwise 冒烟：TP=4 + 长前缀 load 复测**（线程平移正确性：save 失败传导、h2d stagger、layer 事件时序）
3. mooncake 非 layerwise 冒烟无回归（key-mode 四线程类未触碰的旁证）
4. grep 断言三条（终稿验收标准 7）：kv_transfer.py 无 GVA 线程类与 `LayerBatchBuilder`；通用层 `.store.` 直捅清零；`LayerBatchBuilder` 定义仅在 backend/
5. ruff / mypy（3.10-3.12）
6. UT 5 在 165 真环境跑通一次（mock 探针 + 真链路冒烟双重确认）

## 5. 平移纪律（终稿 §3.4 重申 + PR-B 特有）

逐字符一致、禁止顺手重构（f-string、日志文案、debug 级 `split_gvas=...` 一律原样）；`_start_kv_transfer_threads` 内 start→wait→start→wait 顺序逐行保持（:484-560 现结构）；`save_failure_checker` 注入点保持 send-ready 之后（终稿风险表"ready_event 时序"）；group_builders 两线程独立构建（B-4）；`batch_copy` 转发不加 ensure/lazy 检查（现直捅语义）；ReqMeta / LayerTransferTask 字段结构零改动。

## 6. 与 PR-A / open PR 的关系

- 本分支基于 PR-A HEAD（a0bb5dcf2）；PR-A 若因 review 修订，先 rebase 本分支再继续施工（避免终稿 §4.2"两 PR 并行流转互追 rebase"的禁区）
- #12854（layerwise transfer rework）与 PR-B 同区 ~700 行移动冲突（终稿 §7 首行风险）：维持"PR-A 抢先合入，#12854 rebase"策略；PR-B 描述附 §3.4 映射表 + @ 作者说明，PR-B 自身合入前若 #12854 已进 main，按映射表逐块人工 rebase（禁止 mindless rebase，逐行核对语义）
- #14697 与 PR-B 无交集（线程区域不触碰）
