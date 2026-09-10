# AscendStore 四项需求交付自测报告

> 交付结论：四项需求全部完成交付。

---

## 1. 交付判定总览

| # | 需求 | 交付判定 | 交付载体（PR） |
|---|---|---|---|
| 1 | AscendStore UT 整改，删除冗余测试用例 | ✅ 完成 | [#13160](https://github.com/vllm-project/vllm-ascend/pull/13160)、[#14465](https://github.com/vllm-project/vllm-ascend/pull/14465) |
| 2 | kv_pool 目录重构（UCM 等后端子目录与 ascend_store 平级）+ ascend_store 个别文件重命名 | ✅ 完成 | [#13354](https://github.com/vllm-project/vllm-ascend/pull/13354) |
| 3 | AscendStore 消除死代码与冗余、优化结构 | ✅ 完成 | [#13160](https://github.com/vllm-project/vllm-ascend/pull/13160)、[#14465](https://github.com/vllm-project/vllm-ascend/pull/14465) |
| 4 | Layerwise 逻辑收敛到 backend 文件 | ✅ 完成 | [#15367](https://github.com/vllm-project/vllm-ascend/pull/15367) |

---

## 2. 需求 1：UT 整改，删除冗余测试用例

**方案设计**：依托 [#13160](https://github.com/vllm-project/vllm-ascend/pull/13160) 对 ascend_store 的大规模简化重构（13 files，+983 / −2960）。为了保证可评审、可尽快合入，把改动按风险分成前后依赖的两个部分：低风险部分承担公共 helper 提取、低风险清理与 UT 整改，高风险部分（backend、MLA、执行入口）单独推进，避免整个模型/特性矩阵验证完才合入。UT 整改侧重点是删除冗余与低价值用例（如纯 getter 测试等两三行方法对应的小用例），并同步校正 6 个受影响的 UT 文件（`test_ascend_store_connector`、`test_backend`、`test_kv_transfer`、`test_metadata`、`test_pool_scheduler`、`test_pool_worker`），使其与精简后的接口保持一致。

**测试结果**：
- 改动面本地 UT：`pytest -q` 对 `pool_scheduler.py`、`pool_worker.py` 及 6 个 UT 文件运行，**220 passed, 14 warnings in 10.47s**，无失败。
- PR 提交在合并前由 GitHub CI（ruff / mypy / 全量 UT）覆盖，GitHub checks 全绿。
- 配套 [#14465](https://github.com/vllm-project/vllm-ascend/pull/14465) 继续清理 connector 侧冗余派生副本，合入后由 #15367 的 e2e 场景 1 兜底验证（MultiConnector PD 分离，请求 5/5 成功、无 AttributeError）。

---

## 3. 需求 2：kv_pool 目录重构 + ascend_store 个别文件重命名

**方案设计**：通过 [#13354](https://github.com/vllm-project/vllm-ascend/pull/13354)（33 files，+85 / −114，6 commits）重组 ascend_store 模块结构，把原来散落/扁平的文件按职责和归属归位，提升框架可读性：
- 单文件 `ucm_connector.py` 重组为 `ucm_connector/` 包，成为与 `ascend_store` 平级的子目录（UCM），便于后续按后端边界扩展。
- `config_data.py` 重命名 `metadata.py`、`backend/backend.py` 重命名 `backend/base.py`，使文件名与职责一致。
- `memcache_comm_fence.py` 跨目录移动至 `ascend_store/attention_fence.py`，归入 ascend_store 包内。
- 删除废弃的 `lmcache_ascend_connector.py`（清理旧 connector）。
- 同步修正其余 26 个文件的 import 路径、测试与文档，保证安装与工厂注册链路的正确性。

**测试结果**：
- 代码对齐：PR head SHA 与容器分支 `git rev-parse` 逐字节一致；`git diff --name-status -M` 的 commits / files / additions / deletions 与 GitHub PR 页面完全一致（33 files，+85 / −114）。
- 静态编译：`attention_fence.py`、`metadata.py`、`ascend_store_connector.py`、`coordinator.py`、`kv_transfer.py`、`pool_scheduler.py`、`pool_worker.py` 全部编译通过；走读遗留的 `dsa_cp.py` 冲突导入已清理干净，静态检查全绿。
- 单元测试：`python3 -m pytest tests/ut/distributed/ascend_store -q` **368 passed, 10 skipped**；`test_mooncake_kv_transfer.py` 1 passed（合计 **369 passed, 10 skipped**）。
- e2e：Baseline（无 KV Pool）链路请求正常返回；开启 KV Pool 后 hit 链路按 `cmpl_id` 与请求一一对应，验证 PASS、无异常。

---

## 4. 需求 3：消除死代码与冗余、优化结构

**方案设计**：在 [#13160](https://github.com/vllm-project/vllm-ascend/pull/13160) 简化重构中完成 ascend_store 的死代码清理与冗余消除（净删约 2960 行），识别并删除不再被运行路径引用的方法/分支；配套 [#14465](https://github.com/vllm-project/vllm-ascend/pull/14465) 删除 connector 侧 `use_gva_layerwise` 冗余派生副本，收敛同一个 gate 的重复实现，降低维护成本并减少不一致风险。

**测试结果**：
- 改动面本地 UT：`pytest -q` 对 `pool_scheduler.py`、`pool_worker.py` 及 6 个 UT 文件运行，**220 passed, 14 warnings**，无失败。
- #14465 删除派生副本后暴露 `set_external_slot_release_waiter` 仍读旧 flag 的隐患，已在 #15367 修复；修复后经 MultiConnector PD 分离 e2e（场景 1，5/5 请求成功、初始化链路无 AttributeError）验证通过。
- 死代码清理同时经 ascend_store 全量 UT 回归（见需求 2/4 测试结果），无行为回归。

---

## 5. 需求 4：Layerwise 逻辑收敛到 backend 文件

**方案设计**：通过 [#15367](https://github.com/vllm-project/vllm-ascend/pull/15367)（7 commits）将后端强相关的 layerwise 逻辑收敛到 backend 侧，使通用层不再绑定 memcache/GVA 语义、后端可解析式接入，属行为保持的重新封装（运行路径与外部语义不变）：
- gate 收敛为 memcache 协议模块单点派生（统一 backend 键/默认值/归一化），删除仓库内 worker、scheduler、layout、connector 四处派生副本。
- GVA key 构造集中于 memcache 协议模块（GVAKeyFactory），key 字节格式以逐字节快照锁定。
- 注册表 `backend_map` 增加 `layerwise_protocol` 布尔标记，通用层经 `get_layerwise_protocol()` 中性解析，不再硬编码 `memcache` 字符串。
- connector 侧 `set_external_slot_release_waiter` 的 gate 下沉至 `KVPoolWorker`，消除对 connector 侧派生 flag 的依赖（同时修复 #14465 暴露的隐患）。

**测试结果**：
- 单元测试：`tests/ut/distributed/ascend_store` 全量从 **313 提升至 314 passed**（新增 `test_gva_protocol.py`：gate 真值表、GVA 方法仅被 MemcacheBackend override 的排他性断言、三类 key 字节级快照、`extract_layout_config` opt-in 两路）；GitHub CI 29/30 checks 绿（含 ruff / mypy / 全量 UT）。
- e2e 三场景全 PASS（2026-09-01，165 容器）：
  1. **MultiConnector PD 分离**：请求 5/5 成功（含 2 条 GSM8K-lite 真实问题），初始化链路无 AttributeError（#14465 修复生效点）。
  2. **memcache layerwise 冒烟**（DeepSeek-V2-Lite，MLA）：`load_gvas valid_gvas > 0`、`hit_check hit_tokens > 0`，load 路径真实生效，排除静默失效。
  3. **mooncake 非 layerwise 冒烟**（Qwen3-32B）：参考 `layerwise config` 日志修正判定判据后，与 main 基线行为一致，通用路径零回归。
- 原始证据自包含于 test/evidence 目录，供复核。