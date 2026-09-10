# PR-A 记录

## 状态：已提交 upstream，等待 review

- PR：https://github.com/vllm-project/vllm-ascend/pull/15277（2026-08-29 提交，Open）
- 标题：[Refactor] (kv_pool): converge layerwise GVA protocol into backend/gva_protocol
- 分支：`refactor_layerwise` → `vllm-project:main`；4 commits / 15 files changed
- Reviewer：LCAIZJ（code owner，已 request review）；label: `module:tests`
- PR 描述已完成（含 #14465 回归修复动机、测试结论、vLLM base commit ba07e4a）

## Commit 列表（已全部推送）

| Commit | 阶段 | 内容 |
|---|---|---|
| f636716 | C1 | backend capability registry + 单一 `use_gva_layerwise()` 派生 + connector 回归修复（#14465） |
| 45badca | C2 | `GVALayerwiseCapable` 接口抽取 + `on_worker_ready()` 生命周期钩子 |
| e788a10 | C3 | `gva_protocol.py` 纯新增（GVAKeyFactory / GVASession / GVAHitChecker） |
| a0bb5dc | C4 | worker / scheduler 委托切换，删除内联实现 |
| b89884b | C5 | mypy 修复：`batch_write_finish`/`batch_remove_lease` 调用点 `assert isinstance(m_store, GVALayerwiseCapable)`；`_gva_hit_checker` assert non-None；UT mock 改 `_DualSpecStore`（Backend+GVALayerwiseCapable 双 spec） |

## C5 mypy 修复（2026-08-30，rebase upstream/main 后 CI 报 3 错）

- 错误：kv_transfer.py:1380/1581（Backend 无 capability 方法）+ pool_scheduler.py:405（GVAHitChecker | None）
- 修法与 PR-B 的 de736a01 一致（assert narrowing）；GVA 线程仅运行于 memcache（实现该接口），assert 运行时也成立
- UT mock `MagicMock(spec=_DualSpecStore)` 过 isinstance 断言；test_pool_worker 的 `object.__new__` 用法不走调用点，不受影响
- 验证：UT 309 passed / 2 failed（coordinator 两例为 PR-A 起点即有的本地 stub 问题）；ruff 0.14.0 check+format 过；mypy narrowing 模式经本地最小复现实验验证（仓库 mypy.ini）
- refactor_layerwise_B 已 rebase 到含 C5 的 A 上（3 处冲突均按 B 侧意图解决：import 单行、mock 单 spec、类删除）；B 最终 diff 与 rebase 前字节级一致（C5 被 B 自身 commit 序列完全消化），UT 314 passed

## C4 具体产出（已验证）

- pool_worker.py ：
  - 删除 6 个 GVA 方法 + 4 个常量（LAYERWISE_READ_LEASE_TTL_MS / MEMCACHE_UNMATCHED_STATE / PARTIAL_LEASE_RETRY_COUNT / PARTIAL_LEASE_RETRY_INTERVAL_S，均已在 gva_protocol）
  - `_init_layerwise_config` 末尾构造 `self._gva_session`（`_init_backend` 先执行，m_store 就绪，与终稿 §3.3 一致）；`register_kv_caches` 内 `bind_layout`
  - `process_layer_data` 委托 `prepare_load_gvas` → `alloc_gvas_for_save`（顺序不变，session-None gate）
  - :816 `if use_gva_layerwise: ensure_initialized()` → 无条件 `on_worker_ready()`（四情形等价表已论证）
  - `_report_invalid_blocks` 作为 `on_invalid_blocks` 回调注入；`get_block_ids_with_load_errors` 语义不变
  - `import numpy` 随方法迁移移除
- pool_scheduler.py ：
  - 删除 `_make_layerwise_gva_keys_for_hit_check` / `_get_layerwise_gva_hit_tokens`
  - `__init__` 构造 `GVAHitChecker`；hit 检查委托 `hit_tokens`，`_get_or_create_request_tracker` 前置到委托点（行为等价）
- UT 迁移：test_pool_worker / test_pool_scheduler mock 面全部切换到 session / hit_checker；含 4 个既有用例适配
  - test_mtp_gva_prepare 改为构造前传 use_eagle（session 构造时捕获；生产中 use_eagle 在 `_init_kv_transfer_config` 赋值，早于 session 构造，语义等价）
  - test_layerwise_multi_group_layout_includes_mtp 裸对象补 GVASession 构造所需属性

## 验证结论

- 全量 ascend_store UT：309 passed / 2 failed（test_coordinator 两例在 PR-A 起点 6953f2669 即失败——本地 stub 的 `find_longest_cache_hit` 返回形状与真实 vllm 不符，CI/165 有真实 vllm 可过，非本 PR 引入；已在 PR 描述中注明）
- grep 断言：6 个旧方法名全仓零残留
- ruff check + format：通过
- test_layerwise_cache_layout.py 需真实 torch，本机跳过，归 CI / 165 验证
- 本地运行方式：`python d:\lzy\project\kv_pool\run_ascend_store_ut.py tests/ut/distributed/ascend_store/ --noconftest -q -p no:cacheprovider`（shim 补 stub 包父属性绑定；CI 不受影响）

## 待办

1. 跟踪 PR CI（10 checks）与 reviewer（LCAIZJ）/ Gemini Code Assist 意见，按需修订
2. 165 服务器验证（如合入前被要求）
3. PR-B（gva_threads.py 线程收敛，依赖 PR-A 合入后再开工，避免冲突）
