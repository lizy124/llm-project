# PR-B 记录

## 状态：已提交 upstream（#15307，Open），CI mypy 修复已推送

- PR：https://github.com/vllm-project/vllm-ascend/pull/15307
- 分支：`refactor_layerwise_B` → `vllm-project:main`（基于 `refactor_layerwise` @ a0bb5dcf2 = PR #15277 HEAD）
- 规模：5 commits / 11 files changed / +1526 −1081（生产代码新增 gva_threads.py 891 行、删 kv_transfer.py 689 行，与终稿 §4.1 口径一致；UT 侧超预算，已在 PR 描述按实际构成论证）
- PR 描述：同目录 `PR-B description.md`（精简版，已用于 #15307）

## Commit 列表

| Commit | 阶段 | 内容 |
|---|---|---|
| d425b1913 | C1 | `batch_copy` 接口化：GVALayerwiseCapable 第 6 个抽象方法 + MemcacheBackend 直转发（无 ensure/lazy），封堵全仓最后一处 `.store.` 直捅 |
| 452e88d86 | C2 | gva_threads.py 纯新增（879 行）+ test_gva_threads.py 测试副本（568 行），未接线，main 行为零变化 |
| 116f7fe45 | C3 | 切换 + 删除：pool_worker 委托工厂、kv_transfer.py 删 689 行 GVA 专属代码、test_kv_transfer.py 删 328 行用例 |
| de85a2968 | C4 | TRACE 三处 docstring 中性化 + UT 5 探针 + 工厂参数映射断言 + lint 收尾 |
| b38798e68 | CI fix | mypy 4 错误修复（详见下节） |

## 各 Commit 具体产出

- **C1**（d425b1913）
  - backend/base.py：`GVALayerwiseCapable` 新增 `batch_copy(gvas, addrs, sizes, direction) -> int` @abstractmethod
  - backend/memcache_backend.py：`assert self.store is not None; return self.store.batch_copy(...)` 直转发，不加 ensure/lazy（原直捅即无检查；GVA 线程仅在 on_worker_ready / GVASession ensure 之后运行）
  - kv_transfer.py `_batch_copy_with_limits`：`self.m_store.store.batch_copy(...)` → `self.m_store.batch_copy(...)`，前置 isinstance 收窄（B-1：非 GVA 后端误挂 GVA 线程时 fail-fast）
  - test_backend.py：+MemcacheBackend.batch_copy 转发断言（FakeStore 捕获参数）；一致性断言既有双向检查自动覆盖新方法

- **C2**（452e88d86，纯新增）
  - backend/gva_threads.py（879 行）：
    - `LayerBatchBuilder`（:64，整类逐字符平移，含 `_ensure_buf`/`_dedupe_transfer_blocks`/`_build_transfer_arrays`/`build_shared`/`build_addrs`/`build`）
    - `_GVALayerTransferThreadBase(KVTransferThread)`（:330，B-2 中间基类）：持有 `max_transfer_blocks`/`max_transfer_bytes`，下沉 `_split_transfer_packets` + `_batch_copy_with_limits`（逐字符，仅 :453 assert 与直捅按 B-1/C1 改）
    - `KVCacheStoreLayerSendingThread`（:448）/ `KVCacheStoreLayerRecvingThread`（:582）：整类平移改继承中间基类，`__init__` 签名不变（兼容既有 UT 直接构造）
    - `GVALayerwiseThreadContext` dataclass（:779，公共 12 字段）
    - 工厂 ×3：`build_group_layer_builders(ctx)`（:804，pool_worker `_build_group_layer_builders` 逻辑平移）、`create_gva_sending_thread(ctx, ready_event)`（:828）、`create_gva_recving_thread(ctx, ready_event, ..., save_failure_checker=...)`（:850）
    - 依赖方向无环：kv_transfer.py（基类）← gva_threads.py ← pool_worker.py（B-5）
  - test_gva_threads.py（568 行）：从 test_kv_transfer.py 迁移四类 GVA 用例副本 + GVA 专用 helper（key-mode 用例不动）
  - AST 校验方法体逐字符一致

- **C3**（116f7fe45，切换 + 删除）
  - pool_worker.py：
    - import 切换（:37-42）：GVA 三符号改从 `backend.gva_threads` import；key-mode 符号保持
    - `_build_group_layer_builders` 方法删除（逻辑已入工厂）
    - `_start_kv_transfer_threads` layerwise 分支：ctx 构造（:471）→ `create_gva_sending_thread`（:495）→ `create_gva_recving_thread`（:518）；`start()`/`ready_event.wait()` 原地保留（B-3）；send-ready 后才建 recv 的顺序逐行保持；`save_failure_checker` 注入点保持 send-ready 之后
    - isinstance 三处（_build_shared_save_data / _build_shared_load_data / :1628）仅换 import 源，类名不变
  - kv_transfer.py：1615 → 926 行，删五项（LayerBatchBuilder / _split_transfer_packets / _batch_copy_with_limits / 两线程类）；key-mode 四线程类、`_circular_shift`、`record_failed_blocks`、KVTransferThread 通用基类全部保留
  - test_kv_transfer.py：删 GVA 用例四类与 GVA 专用 helper（-328 行）
  - group_builders 保持 send/recv 各构建一次、实例隔离（B-4，禁止"优化"为共享——`_ensure_buf` 有内部 buffer 状态）

- **C4**（de85a2968）
  - TRACE 三处 docstring 中性化：attention_fence.py 两处（`MemCache worker threads`/`per-layer gate for MemCache work`）+ layerwise_cache_layout.py 一处（`for the MemCache GVA layerwise path` → `for the GVA layerwise path`），仅措辞零代码变化
  - UT 5 探针 `test_load_path_end_to_end_nonzero_gva`：FakeStore `batch_get_key_info` 返回有效 gva → GVASession.prepare_load_gvas → 断言 load_block_gvas 非零 → recv 线程 `_handle_request` → mock batch_copy 捕获参数断言 gvas 全非零。覆盖协议→布局→传输链，任一环节退化（lazy 空返回/租约跳过/gva 全 0）即红
  - 工厂参数映射断言（TestFactoryParameterMapping ×3）：send/recv 全参数与线程属性一一相等 + send/recv builders 独立实例断言（B-4 不变量）
  - ruff / mypy 修复收尾

## CI mypy 修复（b38798e68，2026-08-29）

#15307 首轮 CI 的 mypy job 报 4 错。根因：PR-A 把 `batch_write_finish`/`batch_remove_lease` 等从 `Backend` 挪进 `GVALayerwiseCapable` 后，原直捅 `.store.`（`store: Any`，mypy 不可见）被接口化，类型错位才暴露——本地因缺真实 vllm（import-not-found 掩盖）未拦住，CI 用完整 vllm 源码作类型源才报出。

- `gva_threads.py:348` arg-type：`_GVALayerTransferThreadBase.__init__` 传 `GVALayerwiseCapable` 给 `KVTransferThread.__init__`（形参 `Backend`）。修复：super() 前 `assert isinstance(m_store, Backend)`（运行时恒真：MemcacheBackend 双继承；性质同 B-1 fail-fast）
- `gva_threads.py:557/:758` attr-defined：`self.m_store`（Backend）上调用能力接口方法。修复：新增 `_gva_store` property（isinstance 收窄返回 `GVALayerwiseCapable`），三个调用点（batch_write_finish / batch_remove_lease / batch_copy）改走 property；`_batch_copy_with_limits` 循环外取局部变量，断言次数与原版一致
- `pool_scheduler.py:405` union-attr：`_gva_hit_checker`（`GVAHitChecker | None`）在 `use_gva_layerwise` 分支内访问。修复：分支内 `assert ... is not None`（构造条件 `if self.use_gva_layerwise` 与分支条件同一 invariant）
- 测试适配：`MagicMock(spec=(A, B))` 元组 spec 实测 isinstance 全 False（不可用），改测试内定义 `_DualSpecStore(Backend, GVALayerwiseCapable)` 双继承类作 spec
- 本地复现方法：`git archive HEAD` 导出未修复版到临时目录，`mypy --follow-imports skip --check-untyped-defs --ignore-missing-imports`（4 文件：base/gva_threads/kv_transfer/pool_scheduler）精确复现 CI 同 4 错；修复后同命令 0 错误。注意：4 文件必须同时列出，缺任何一个则该依赖变 Any、对应错误不报

修复后验证：全量 UT 314 passed / 2 failed（既有 coordinator stub 失败）、ruff check/format 通过、mypy 模拟 CI 0 错误。

## 验证结论

- 全量 ascend_store UT：**314 passed / 2 failed / 123 subtests passed**（1.51s）
  - test_coordinator 两例为 PR-A 起点 6953f2669 即失败的本地 stub 问题（`find_longest_cache_hit` 返回形状与真实 vllm 不符），CI/165 有真实 vllm 可过，非本 PR 引入（与 PR-A 相同结论）
  - 相比 PR-A 的 309 passed 净增 5：UT 5 探针 1 + 参数映射 3 + batch_copy 转发 1（迁移用例在 C2 建副本、C3 删原版，净 0）
- grep 断言三条全过（终稿验收标准 7）：
  1. kv_transfer.py 无 `KVCacheStoreLayer*Thread` / `LayerBatchBuilder` / `_batch_copy_with_limits` / `_split_transfer_packets`
  2. backend 目录外 `.store.` 直捅清零
  3. `class LayerBatchBuilder` 定义仅 backend/gva_threads.py；attention_fence.py / layerwise_cache_layout.py 无 "MemCache" 残留
- ruff check + format：通过（28 files already formatted）
- mypy（gva_threads.py / kv_transfer.py / pool_worker.py）：仅 15 个 import-not-found（本机无真实 torch/vllm，与基线一致），零类型错误
- test_layerwise_cache_layout.py 需真实 torch，本机跳过，归 CI / 165 验证
- 本地运行方式：`python d:\lzy\project\kv_pool\run_ascend_store_ut.py tests/ut/distributed/ascend_store/ --ignore=tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py --noconftest -q -p no:cacheprovider`（cwd = vllm-ascend 仓库根；shim 依赖 os.getcwd 补 stub 包父属性绑定）

## 与 PR-A / open PR 的关系

- 本分支基于 PR-A HEAD（a0bb5dcf2）；PR-A 若因 review 修订，先 rebase 本分支再推送
- #12854（layerwise transfer rework）与 PR-B 同区 ~690 行移动冲突：维持"PR-A 抢先合入，#12854 rebase"策略；PR-B 描述已附映射注释，若 #12854 先进 main 则按终稿 §3.4 映射表逐块人工 rebase（禁止 mindless rebase）
- #14697 与 PR-B 无交集（线程区域不触碰）

## 待办

1. 跟踪 #15307 CI（mypy 修复 b38798e68 已推送，等待重跑结果）与 reviewer 意见
2. 165 服务器验证（执行计划 §4，听指示）：memcache layerwise 冒烟 TP=4 + 长前缀 load 复测（save 失败传导 / h2d stagger / layer 事件时序）、mooncake 非 layerwise 冒烟、UT 5 真环境跑通
