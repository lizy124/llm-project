# PR 13049 验证计划

## 1. 验证对象

PR:

```text
https://github.com/vllm-project/vllm-ascend/pull/13049
```

当前本地验证仓库：

```text
/vllm-workspace/vllm-ascend
branch: pr-13049-new
HEAD: fb9c84529 style: apply ruff fixes to test_packed_tensor
base: origin/main
```

PR 元数据（GitHub 显示）：

```text
7 commits, 8 files changed, +825 / -161
title: [Test] Extract weight transfer HTTP/e2e helpers and add unit tests
```

实际改动文件（8 个）：

```text
examples/rl/rlhf_http_hccl.py
examples/rl/rlhf_http_npu_ipc.py
examples/rl/weight_transfer_http_utils.py
tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py
tests/e2e/pull_request/weight_transfer_utils.py
tests/ut/distributed/weight_transfer/test_http_utils.py
tests/ut/distributed/weight_transfer/test_packed_tensor.py
```

PR 范围经过一次重大调整：评审反馈 weight_transfer 本身已足够简洁，不需要加抽象层重构。因此早期提交引入的 `compat.py` / `registry.py` / `test_compat.py` / `test_registry.py` 已在 commit `cffe2a455` 回退，`vllm_ascend/distributed/weight_transfer/__init__.py` 恢复原版（33 行，直接调用 `WeightTransferEngineFactory.register_engine()`）。

最终 PR 只保留两类改动：

1. **HTTP/e2e helper 抽取**：把两个 example 脚本和两个 e2e 测试中重复的 HTTP 调用和 `BackgroundPost` 抽到公共 utils。
2. **UT 补充**：新增 `test_http_utils.py`（3 个 UT）和 `test_packed_tensor.py`（18 个 UT，CPU-only）。

## 2. Commit 历史

```text
fb9c84529 style: apply ruff fixes to test_packed_tensor
7ba2ad8a3 test(weight-transfer): add UT for packed_tensor module
cffe2a455 revert(weight-transfer): drop compat/registry abstraction layer
a2e996889 fix(weight-transfer): gate aliases by platform and fix example imports
472dcc07d style: apply ruff fixes to weight transfer changes
a4cf967ce fix(weight-transfer): detect stateful trainer capability
0aa63ffb4 refactor(weight-transfer): centralize registration and HTTP helpers
```

历史说明：commit 1-4 引入 compat/registry 抽象层，commit 5 回退，commit 6-7 补 UT。最终 diff 干净（无 compat/registry 残留），但历史有"加了又删"的 5 个中间 commit。若评审要求干净历史，合并前可 squash 成单个 commit。

## 3. 风险边界

本 PR 不改变 weight transfer 的核心行为：

- HCCL worker transport 不变。
- NPU IPC worker transport 不变。
- packed tensor wire contract 不变。
- HTTP/Ray/callable payload schema 不变。
- `__init__.py` 注册逻辑不变（原版 `vllm_version_is("0.26.0")` 判断保留）。

风险点集中在 helper 抽取后的行为等价性：

1. **HTTP helper**：URL 拼接、timeout、status 检查、异常传播是否与原 inline 实现一致。
2. **e2e helper**：`BackgroundPost` 的线程语义、异常捕获是否与原 inline 实现一致。
3. **example 导入**：按文档 `python rlhf_http_*.py` 运行是否能正常导入。
4. **`client` 变量覆盖**：`rlhf_http_npu_ipc.py` 中 OpenAI client 是否被 `HTTPVLLMWeightSyncClient` 覆盖。

## 4. review.md 三条评审意见处置

| # | 评审意见 | 处置 | 状态 |
|---|---|---|---|
| 1 | `nccl`/`ipc` alias 无条件覆盖 upstream registry | 回退 compat/registry 后自动消失（原版 `__init__.py` 不注册 alias） | ✅ 自动解决 |
| 2 | 示例按文档 `python rlhf_http_*.py` 运行导入失败 | 保留的 helper 用同目录导入 `from weight_transfer_http_utils import ...` | ✅ 已修复 |
| 3 | `rlhf_http_npu_ipc.py` 中 `client` 被 `HTTPVLLMWeightSyncClient` 覆盖 | 保留的 example 改名为 `weight_sync_client` | ✅ 已修复 |

## 5. 第一层：本地静态检查

```bash
git status --short --branch
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
git diff --check origin/main...HEAD
python -m compileall -q \
  examples/rl/rlhf_http_hccl.py \
  examples/rl/rlhf_http_npu_ipc.py \
  examples/rl/weight_transfer_http_utils.py \
  tests/e2e/pull_request/weight_transfer_utils.py \
  tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py \
  tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py \
  tests/ut/distributed/weight_transfer/test_http_utils.py \
  tests/ut/distributed/weight_transfer/test_packed_tensor.py
```

预期：

- 分支为 `pr-13049-new`，HEAD 为 `fb9c84529`。
- `origin/main..HEAD` 包含 7 个 commit。
- diff 统计为 `8 files changed, 825 insertions(+), 161 deletions(-)`。
- 文件范围只包含上文列出的 8 个文件。
- `git diff --check` 无 whitespace error。
- compileall 通过。

## 6. 第二层：纯 Python UT

本 PR 新增两组 UT，全部可在 CPU 环境运行，无需 NPU 硬件。

```bash
python -m pytest -q \
  tests/ut/distributed/weight_transfer/test_http_utils.py \
  tests/ut/distributed/weight_transfer/test_packed_tensor.py
```

### 6.1 test_http_utils.py（3 个 UT）

| 测试 | 覆盖点 |
|---|---|
| `test_post_weight_transfer_endpoint_*` | URL 拼接、payload 传递、timeout、response 处理 |
| `test_start_weight_update_*` | `/start_weight_update` endpoint 调用 |
| `test_get_world_size_*` | `/get_world_size` 返回值解析 |

### 6.2 test_packed_tensor.py（18 个 UT）

通过 stub `torch.npu.Stream` / `group.broadcast` / `rebuild_npu_tensor` 实现 CPU-only 测试。

| 函数 | UT 数 | 覆盖点 |
|---|---|---|
| `packed_broadcast_producer` | 6 | 单 tensor、多 tensor、超 buffer 切分、空迭代器、num_buffers 轮转、src 传递 |
| `packed_broadcast_consumer` | 6 | 单 tensor、多 tensor、空迭代器、src 传递、多 dtype |
| `packed_npu_ipc_producer` | 5 | 单 chunk、空迭代器、超 buffer 切分、单 tensor 超 buffer 报错、dtype 名称提取 |
| `packed_npu_ipc_consumer` | 5 | roundtrip、UUID 不匹配报错、device_index 覆盖、clone 独立 storage、content_size 截断 |

### 6.3 完整 weight_transfer UT 目录回归

```bash
python -m pytest -q tests/ut/distributed/weight_transfer
```

确认回退 compat/registry 后，原有 `test_npu_ipc_engine.py`（4 个 UT）仍能通过。

## 7. 第三层：人工 diff 审查

按文件组逐项确认 helper 抽取未引入回归：

| 文件组 | 审查重点 |
|---|---|
| `weight_transfer_http_utils.py` / `test_http_utils.py` | URL 拼接、timeout、status 检查、异常传播，不编码 backend payload |
| `rlhf_http_hccl.py` / `rlhf_http_npu_ipc.py` | 只复用 helper，不改变 HCCL/NPU IPC payload 和 lifecycle 顺序；同目录导入正确；`client` / `weight_sync_client` 命名正确 |
| `weight_transfer_utils.py` / e2e tests | `BackgroundPost` 线程语义、异常捕获与原 inline 实现一致；e2e 断言不变 |
| `test_packed_tensor.py` | mock 范围只覆盖 NPU 原语，不掩盖被测逻辑；断言验证数据正确性 |

## 8. 第四层：Ascend e2e 回归

本 PR 没有改 HCCL/NPU IPC data-plane，但 helper 抽取后建议跑一遍核心路径，确认 HTTP 调用语义未破坏。

```bash
python -m pytest -q tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
python -m pytest -q tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py
```

环境前置：

- 模型路径覆盖为本地 `/mnt/weight/Qwen3-0.6B`（避免 HuggingFace 外网访问）。
- NPU 8 卡 Ascend910 在线。
- `vllm` 0.26.0、`torch_npu`、`requests`、`pytest` 已安装。

预期：

- NPU IPC one-card：baseline → init → pause → start → update → finish → resume → 更新后 completions 全流程通过。
- HCCL two-card：server/trainer rendezvous → 310 tensors packed broadcast → 全流程通过。

## 9. 验收结论模板

完成验证后记录：

```text
PR: #13049
branch: pr-13049-new
HEAD: fb9c84529

static checks:
  git diff --check: <passed/failed>
  compileall: <passed/failed>

unit tests:
  test_http_utils.py: <passed/failed + count>
  test_packed_tensor.py: <passed/failed + count>
  weight_transfer UT full: <passed/failed + count>

Ascend e2e:
  npu_ipc one-card: <passed/failed/skipped + reason>
  hccl two-card: <passed/failed/skipped + reason>

review.md disposition:
  #1 alias override: auto-resolved by revert
  #2 example import: fixed by same-directory import
  #3 client shadowing: fixed by weight_sync_client rename

final decision:
  <pass/fail/blocker>
```

## 10. 不再需要的验证项（相对旧计划）

旧版计划中的以下层级已不再适用，因为 compat/registry 抽象层已回退：

- ❌ 第四层：import 和 lazy-load smoke test（compat/registry 已不存在）
- ❌ 版本矩阵验证（v0.25 / v0.26 / v0.27 / main 四套 venv）
- ❌ `test_compat.py` / `test_registry.py` UT
- ❌ trainer API capability 探测验证

验证范围从旧版的 6 层缩减为 4 层，复杂度显著降低。
