# PR #13354 验证报告

- **PR**: [vllm-project/vllm-ascend #13354](https://github.com/vllm-project/vllm-ascend/pull/13354)
- **标题**: [Refactor] reorganize ascend store kv pool modules
- **作者**: lizy124
- **base**: vllm-project:main ← lizy124:refector_backend
- **验证日期**: 2026-08-11
- **验证环境**: 192.168.13.165 / 容器 `refactor_811`
- **Run 目录**: `llm-project/transfer_data/refactor/backend/test/runs/20260811_090221_env/`
- **验证人**: 自动化脚本 + 人工核对日志

> 本报告所有数字均来自本次 run 目录的日志原文，未经脑补或推断。每个关键数据都附有证据出处（日志文件名 + 行内容）。

---

## 1. 代码对齐验证

### 1.1 PR head SHA 核对

| 来源 | HEAD SHA |
|------|----------|
| GitHub PR #13354 head | `ef5a4613dbcdce37b0b8e071e1b3a16648708625` |
| 容器内 `pr-13354` 分支 `git rev-parse` | `ef5a4613dbcdce37b0b8e071e1b3a16648708625` |

**结论**：容器内 `/vllm-workspace/vllm-ascend` 当前 `pr-13354` 分支 HEAD 与 GitHub PR head commit 逐字节一致。验证所用代码 = PR #13354 代码。

### 1.2 改动量对齐

| 项 | GitHub PR 页面 | `git diff 86db2ed32..ef5a4613d --shortstat` | 对齐 |
|----|---------------|---------------------------------------------|------|
| commits | 6 | 6 | ✅ |
| files changed | 33 | 33 | ✅ |
| additions | +85 | +85 | ✅ |
| deletions | -114 | -114 | ✅ |

**结论**：改动量 +85/-114、33 文件、6 commits，与 GitHub PR 页面完全一致。

---

## 2. 文件改动明细

通过 `git diff --name-status -M 86db2ed32..ef5a4613d` 得到完整文件级改动。关键的 rename / delete / add：

| 操作 | 源路径 | 目标路径 | 相似度 | 说明 |
|------|--------|----------|--------|------|
| **R100** | `vllm_ascend/memcache_comm_fence.py` | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py` | 100% | 跨目录移动到 ascend_store 包内 |
| **R099** | `ascend_store/config_data.py` | `ascend_store/metadata.py` | 99% | 重命名 + 微调 |
| **R100** | `ascend_store/backend/backend.py` | `ascend_store/backend/base.py` | 100% | 重命名 |
| **R100** | `ascend_store/ucm_connector.py` | `ascend_store/ucm_connector/connector.py` | 100% | 单文件 → 包结构 |
| **D** | `ascend_store/lmcache_ascend_connector.py` | — | — | 旧 lmcache connector 删除 |
| **A** | `ascend_store/ucm_connector/__init__.py` | — | — | 新建包初始化 |
| **A** | `ascend_store/__init__.py` | — | — | 包初始化 |

其余 26 个文件为 import 路径修正、测试同步更新、文档更新等。

> ⚠️ 说明：`lmcache_ascend_connector.py`（删除）和 `ucm_connector.py → ucm_connector/connector.py`（重命名）是**两件独立的事**。前者是清理废弃的 lmcache connector，后者是把单文件 `ucm_connector.py` 重组为 `ucm_connector/` 包。不能用 `--stat` 截断输出把它们读成一个 rename。

---

## 3. 走读问题验证

走读文档（`pr_13354_code_walkthrough.md`）中指出的 `dsa_cp.py` 导入冲突问题：

| 检查项 | 命令 | 结果 |
|--------|------|------|
| 残留 `all_gather_async` 导入 | `grep all_gather_async dsa_cp.py` | ✅ 无（返回 NO） |
| `record_attention_compute_start` 导入路径 | `grep import ... dsa_cp.py` | ✅ `from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start` |

**结论**：PR 作者通过后续 commit (`ebdc99620`) 已删除冲突导入，`dsa_cp.py` 导入区干净，指向新的 `ascend_store.attention_fence` 路径。

---

## 4. 静态检查 + 单元测试

脚本：`01_check_import_ut.sh`，日志：`check_import_ut.log`

### 4.1 Import 检查（10 模块）

```
OK import vllm
OK import vllm_ascend
OK import vllm_ascend.distributed.kv_transfer
OK import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence
OK import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata
OK import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker
OK import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler
OK import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer
OK import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector
OK import vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base
```

✅ 10/10 全部导入成功。

### 4.2 py_compile（7 文件）

```
OK py_compile
```

✅ `attention_fence.py` / `metadata.py` / `ascend_store_connector.py` / `coordinator.py` / `kv_transfer.py` / `pool_scheduler.py` / `pool_worker.py` 全部编译通过。

### 4.3 pytest

| 套件 | 结果 |
|------|------|
| `tests/ut/distributed/ascend_store` | ✅ **368 passed, 10 skipped**（2.67s） |
| `tests/ut/distributed/mooncake/test_mooncake_kv_transfer.py` | ✅ **1 passed**（0.04s） |

**结论**：静态检查 + UT 全绿。

---

## 5. Baseline E2E（无 KV Pool）

脚本：`03_start_server_baseline.sh` + `05_send_requests.py --case baseline --repeat 2`
模型：`/mnt/weight/Qwen3-0.6B`，端口 8100，单卡

### 5.1 请求结果（来自 `summary_baseline.json` + `requests.jsonl`）

| 请求 | status | elapsed_sec | prompt_tokens | completion_tokens |
|------|--------|-------------|---------------|-------------------|
| 1 | 200 | 1.893 | 1143 | 64 |
| 2 | 200 | 1.883 | 1143 | 64 |

### 5.2 文本一致性

两次请求输出文本完全一致（重复输出 `" AscendStore KV Pool should reuse a long shared prefix across requests. ..."`）。

**结论**：Baseline 链路正常，无 KV Pool 时请求正常返回。

---

## 6. KV Pool E2E（mooncake backend, kv_both）

脚本：`02_start_mooncake_master.sh` + `10_start_server_kvpool_custom.sh` + `05_send_requests.py --case kvpool_both --repeat 2` + `09_send_stream_requests.py --case stream_kvpool --repeat 4`

环境：`KV_ROLE=kv_both`, `KV_BACKEND=mooncake`, `MOONCAKE_MASTER_ADDRESS=127.0.0.1:50088`

### 6.1 请求结果总览

共 6 个请求（2 普通 + 4 流式），全部 status=200。

| # | 类型 | status | elapsed_sec | prompt_tokens | text_len |
|---|------|--------|-------------|---------------|----------|
| 1 | 普通 | 200 | 2.004 | 1143 | — |
| 2 | 普通 | 200 | 1.888 | 1143 | — |
| 3 | 流式 | 200 | 1.909 | 560 | 382 |
| 4 | 流式 | 200 | 1.895 | 560 | 382 |
| 5 | 流式 | 200 | 2.047 | 560 | 382 |
| 6 | 流式 | 200 | 2.038 | 560 | 382 |

### 6.2 KV Pool hit 逐请求对应（关键）

日志 `server_kvpool_custom.log` 中：
- `POST /v1/completions` 共 **6 条**
- `pool_scheduler.py:586` hit 日志共 **4 条**

通过 cmpl_id 将请求与 hit 日志一一对应：

| 请求 | cmpl_id（来自 requests.jsonl） | hit 日志 cmpl_id | Total tokens | kvpool hit tokens | need to load | 说明 |
|------|-------------------------------|------------------|--------------|-------------------|--------------|------|
| 普通1 | `cmpl-a7b0e61c3fb34cad` | — | 1143 | — | — | **miss**：pool 空，无 hit 日志，put KV |
| 普通2 | `cmpl-bf10718c06004bee` | `cmpl-bf10718c06004bee-0-894ceffc` | 1143 | **1024** | 1024 | 命中普通1 |
| 流式1 | (无 cmpl_id 返回) | — | 560 | — | — | **miss**：不同 prompt，pool 空，put KV |
| 流式2 | — | `cmpl-8f6b0c4d34d30b16-0-9b3bca85` | 560 | **512** | 512 | 命中流式1 |
| 流式3 | — | `cmpl-aed5cbfc2f9e4608-0-944bd16f` | 560 | **512** | 512 | 命中 |
| 流式4 | — | `cmpl-9e01764707b0bcbc-0-93600afb` | 560 | **512** | 512 | 命中 |

**hit 日志缺失原因已查清**：普通1 和流式1 是各自 prompt 系列的第一个请求，KV Pool 中没有对应缓存，属于 miss → put 流程，因此没有 `pool_scheduler.py:586` hit 日志。这是预期行为，不是 bug。

### 6.3 hit 日志原文（4 条）

```
[09:10:07] [pool_scheduler.py:586] Reqid: cmpl-bf10718c06004bee-0-894ceffc, Total tokens 1143, kvpool hit tokens: 1024, need to load: 1024
[09:10:07] [pool_scheduler.py:607] KV pool load spec created req=cmpl-bf10718c06004bee-0-894ceffc vllm_cached=0 kvpool_cached=1024 need_to_allocate=1024 load_async=False use_layerwise=False
[09:10:11] [pool_scheduler.py:586] Reqid: cmpl-8f6b0c4d34d30b16-0-9b3bca85, Total tokens 560, kvpool hit tokens: 512, need to load: 512
[09:10:13] [pool_scheduler.py:586] Reqid: cmpl-aed5cbfc2f9e4608-0-944bd16f, Total tokens 560, kvpool hit tokens: 512, need to load: 512
[09:10:15] [pool_scheduler.py:586] Reqid: cmpl-9e01764707b0bcbc-0-93600afb, Total tokens 560, kvpool hit tokens: 512, need to load: 512
```

### 6.4 External prefix cache hit rate = 36.0% 口径核实

日志原文（`09:10:10` 时间点快照）：

```
[09:10:10] [loggers.py:310] Engine 000: Avg prompt throughput: 164.5 tokens/s, Avg generation throughput: 14.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 36.0%
```

计算验证（该时间点已处理 3 个请求：普通1 + 普通2 + 流式1）：

```
hit_tokens / cumulative_prompt_tokens
= 1024 / (1143 + 1143 + 560)
= 1024 / 2846
= 35.98%
≈ 36.0% ✅
```

**口径确认**：`External prefix cache hit rate` = `累计 hit_tokens / 累计 prompt_tokens`，是日志时间点的快照值，不是最终值。最终值（6 请求）= 2560 / 4526 ≈ 56.5%，但日志中未打印最终快照。

### 6.5 流式请求文本一致性

4 个流式请求输出文本完全一致：

```
" The fence relocation must not change stream behavior. The fence relocation must not change stream behavior. ..."（重复 7.5 次，text_len=382）
```

**结论**：
- KV Pool 链路正常，`AscendStoreConnector` 正常工作
- hit 行为符合预期（首个请求 miss + put，后续请求命中）
- hit rate 36.0% 口径已核实
- 流式输出文本一致

---

## 7. 异常检查

`log_extract.txt` 中扫描关键字：

| 关键字 | 出现次数 |
|--------|----------|
| `ImportError` | 0 |
| `ModuleNotFoundError` | 0 |
| `Traceback` | 0 |

**结论**：无异常。

---

## 8. 验证结论

| 维度 | 项 | 结论 |
|------|----|------|
| 代码对齐 | HEAD SHA | ✅ `ef5a4613d` 与 GitHub PR 一致 |
| 改动量 | +85/-114, 33 files, 6 commits | ✅ 与 GitHub PR 一致 |
| 文件改动 | rename/delete/add 分类正确 | ✅ 已用 `--name-status -M` 核实 |
| 走读问题 | dsa_cp.py 导入冲突 | ✅ 已修复 |
| 静态检查 | 10 模块 import + 7 文件 py_compile | ✅ 全绿 |
| 单元测试 | ascend_store + mooncake | ✅ 369 passed, 10 skipped |
| Baseline E2E | 2 请求 | ✅ 全 200 |
| KV Pool E2E | 2 普通请求 | ✅ 全 200，hit 行为符合预期 |
| 流式 E2E | 4 请求 | ✅ 全 200，文本一致 |
| hit rate | 36.0% | ✅ 口径已核实（1024/2846） |
| 异常检查 | ImportError/Traceback | ✅ 无 |

### 最终结论

**PR #13354 验证通过。**

本次重构（ascend store kv pool modules 重组）在以下方面均验证 OK：
1. 代码与 GitHub PR 完全对齐
2. 走读文档指出的导入冲突问题已修复
3. 静态检查 + UT 全绿
4. Baseline / KV Pool / 流式 E2E 全绿
5. KV Pool 命中行为符合预期，hit rate 口径已核实
6. 无任何 ImportError / ModuleNotFoundError / Traceback

---

## 9. 附录：验证产物清单

Run 目录：`test/runs/20260811_090221_env/`

| 文件 | 用途 |
|------|------|
| `env.txt` | 环境快照（git 状态、Python 包、NPU 信息） |
| `check_import_ut.log` | Import 检查 + py_compile + pytest 日志 |
| `server_baseline.log` | Baseline 服务器日志 |
| `server_kvpool_custom.log` | KV Pool 服务器日志（含 hit 日志） |
| `mooncake_master.log` | Mooncake master 日志 |
| `mooncake.json` | Mooncake 配置 |
| `requests.jsonl` | 普通请求记录（含 cmpl_id、status、usage） |
| `summary.json` | 普通请求汇总 |
| `summary_baseline.json` | Baseline 请求汇总 |
| `stream_requests.jsonl` | 流式请求记录 |
| `stream_summary.json` | 流式请求汇总 |
| `log_extract.txt` | 关键日志抽取（hit 日志、异常扫描） |

---

## 10. 附录：验证脚本

本次验证使用的脚本（位于 `test/` 目录）：

| 脚本 | 用途 |
|------|------|
| `00_env_snapshot.sh` | 环境快照 |
| `01_check_import_ut.sh` | Import + py_compile + pytest |
| `02_start_mooncake_master.sh` | 启动 Mooncake master |
| `03_start_server_baseline.sh` | 启动 Baseline 服务器 |
| `05_send_requests.py` | 发送普通请求 |
| `06_grep_logs.sh` | 抽取关键日志 |
| `07_stop_run.sh` | 停止所有进程 |
| `09_send_stream_requests.py` | 发送流式请求 |
| `10_start_server_kvpool_custom.sh` | 启动 KV Pool 服务器 |
| `run_baseline_v2.sh` | Baseline 编排脚本 |
| `run_kvpool_v2.sh` | KV Pool + 流式编排脚本 |

---

*报告生成时间：2026-08-11*
*验证 run 目录：`test/runs/20260811_090221_env/`*
