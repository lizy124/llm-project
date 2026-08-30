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
