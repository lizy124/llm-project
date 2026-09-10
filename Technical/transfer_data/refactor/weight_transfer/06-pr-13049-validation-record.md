# PR 13049 验证记录

## 1. 验证对象

PR:

```text
https://github.com/vllm-project/vllm-ascend/pull/13049
```

本地验证仓库：

```text
/vllm-workspace/vllm-ascend
branch: weight_transfer_refactor
HEAD: c312a05dd fix(weight-transfer): resolve test_packed_tensor failures on NPU env
base: origin/main
```

PR 元数据（GitHub 显示与本地一致）：

```text
8 commits, 8 files changed, +825 / -161
title: [Test] Extract weight transfer HTTP/e2e helpers and add unit tests
```

涉及文件（8 个）：

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

历史说明：PR 早期版本曾引入 `compat.py` / `registry.py` / `test_compat.py` / `test_registry.py` 抽象层（commit `0aa63ffb4` / `a4cf967ce` / `472dcc07d` / `a2e996889`），评审反馈"weight_transfer 本身已足够简洁，不需要加抽象层重构"后，在 commit `cffe2a455` 回退，并在 commit `7ba2ad8a3` / `fb9c84529` 转向补 UT 方向。

## 2. 验证环境

服务器：`192.168.13.165`（`ipb21b04a5.dynamic.kabel-deutschland.de`）
容器：`refactor_810`

环境能力（已确认）：

- `pytest` / `requests` / `ray` / `torch_npu` 均已安装
- `vllm` 0.26.0
- `torch_npu` 2.10.0.post2
- `npu-smi` 可用，8 x Ascend910
- NPU 驱动/CANN 可用，支持 e2e 测试

## 3. 执行结果

### 3.1 PR 元数据对齐

```text
PR 显示: 8 commits, 8 files, +825 / -161
本地实际: 8 commits, 8 files, +825 / -161
HEAD: c312a05dd (both)
状态: 对齐
```

本地 8 个 commit（从旧到新）：

```text
0aa63ffb4 refactor(weight-transfer): centralize registration and HTTP helpers
a4cf967ce fix(weight-transfer): detect stateful trainer capability
472dcc07d style: apply ruff fixes to weight transfer changes
a2e996889 fix(weight-transfer): gate aliases by platform and fix example imports
cffe2a455 revert(weight-transfer): drop compat/registry abstraction layer
7ba2ad8a3 test(weight-transfer): add UT for packed_tensor module
fb9c84529 style: apply ruff fixes to test_packed_tensor
c312a05dd fix(weight-transfer): resolve test_packed_tensor failures on NPU env
```

### 3.2 静态检查

已通过（CI pre-commit hooks）：

```text
ruff check: Passed (after auto-fix)
ruff format: Passed
codespell: Passed
typos: Passed
clang-format: Passed
markdownlint: Passed
gitleaks: Passed
```

### 3.3 UT

已通过（容器 `refactor_810`，vllm 0.26.0 + torch_npu 2.10.0.post2）：

```text
tests/ut/distributed/weight_transfer/test_http_utils.py: 3 passed
tests/ut/distributed/weight_transfer/test_packed_tensor.py: 21 passed
weight_transfer UT full: 30 passed, 0 failed
```

UT 修复内容（commit `c312a05dd`）：

- `_stub_torch_npu` fixture 添加 `stream=lambda s: s` 上下文管理器
- `ipc_handle` args tuple 扩展到长度 7+ 防止 `list_args[6]` 越界
- consumer 测试添加 `.cpu()` 避免跨设备 `torch.equal`
- `overwrites_device_index` 在 `with patcher:` 内部获取 `call_args`

### 3.4 人工 diff 审查

已通过。8 个文件逐一审查结论：

```text
1. examples/rl/weight_transfer_http_utils.py (新增 +84)
   - 提取 HTTP helper：post_weight_transfer_endpoint/pause/resume/init/start/update/finish/get_world_size
   - URL 拼接用 rstrip/lstrip 处理斜杠，timeout 常量化（CONTROL/UPDATE/WORLD_SIZE）
   - 无逻辑变更，纯抽取

2. tests/e2e/pull_request/weight_transfer_utils.py (新增 +74)
   - 提取 e2e 共享 post/log/BackgroundPost
   - BackgroundPost 用 daemon 线程 + raise_if_failed 模式，异常不丢失
   - 无逻辑变更，纯抽取

3. examples/rl/rlhf_http_hccl.py (-75/+27)
   - 移除本地 HTTP 函数，改用 weight_transfer_http_utils
   - init_weight_transfer_engine/update_weights 保留业务参数组装，转发给 helper
   - 无逻辑变更，纯重构

4. examples/rl/rlhf_http_npu_ipc.py (-46/+11)
   - 移除本地 HTTP 函数，改用 helper
   - 修复 client 变量覆盖：weight_sync_client vs OpenAI client（评审意见 #3）
   - 无其他逻辑变更

5. tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py (-7/+1)
   - 移除本地 _post，改用 weight_transfer_utils.post
   - 无逻辑变更

6. tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py (-46/+11)
   - 移除本地 _post/_log/_BackgroundPost，改用 helper
   - 无逻辑变更

7. tests/ut/distributed/weight_transfer/test_http_utils.py (新增 +53)
   - 3 个 UT：URL 规范化、可选 payload、返回值解析
   - mock requests.post/get，断言调用参数

8. tests/ut/distributed/weight_transfer/test_packed_tensor.py (新增 +559)
   - 21 个 UT：packed_broadcast_producer/consumer、packed_npu_ipc_producer/consumer
   - stub torch.npu.Stream/current_stream/stream/synchronize
   - stub reduce_tensor/rebuild_npu_tensor
   - 覆盖正常流程、错误处理、边界条件
```

审查结论：helper 抽取干净，无功能回归；变量覆盖问题已修复；所有 diff 均为纯重构+测试补充。

### 3.5 Ascend e2e

待执行：

```text
npu_ipc one-card: <pending>
hccl two-card: <pending>
```

## 4. review.md 评审意见复核

### 4.1 `nccl` / `ipc` alias 无条件覆盖 upstream registry

结论：评审意见在旧 PR 版本成立，但当前 PR 已回退 compat/registry，问题自动消失。

证据：

- 当前 `vllm_ascend/distributed/weight_transfer/__init__.py` 已恢复原版（33 行）。
- 原版只调用 `WeightTransferEngineFactory.register_engine("hccl", ...)` 和 `register_engine("npu_ipc", ...)`，不注册 `nccl` / `ipc` alias。
- `compat.py` / `registry.py` 已删除。

### 4.2 示例按文档 `python rlhf_http_*.py` 运行导入失败

结论：评审意见成立，当前 PR 已修复。

证据：

- `examples/rl/rlhf_http_hccl.py` 使用 `from weight_transfer_http_utils import ...`（同目录导入）。
- `examples/rl/rlhf_http_npu_ipc.py` 使用同样导入方式。
- 按 `python rlhf_http_*.py` 运行时，脚本目录在 `sys.path[0]`，同目录导入可解析。

### 4.3 NPU IPC 示例覆盖 OpenAI client

结论：评审意见成立，当前 PR 已修复。

证据：

- `examples/rl/rlhf_http_npu_ipc.py` 中 `HTTPVLLMWeightSyncClient` 实例命名为 `weight_sync_client`，不再覆盖 OpenAI `client`。
- 后续 `generate_completions(client, MODEL_NAME, prompts)` 使用的是原 OpenAI client。

## 5. 结论

待所有验证执行完成后填写。

## 6. 本轮命令摘要

```text
ssh root@192.168.13.165
docker exec refactor_810 bash
cd /vllm-workspace/vllm-ascend
git checkout pr-13049
git fetch origin pull/13049/head:pr-13049-new
git checkout pr-13049-new
git log --oneline origin/main..HEAD
git diff --stat origin/main..HEAD
```
