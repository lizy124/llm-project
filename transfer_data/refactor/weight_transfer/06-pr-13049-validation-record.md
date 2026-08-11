# PR 13049 验证记录

## 1. 验证对象

PR:

```text
https://github.com/vllm-project/vllm-ascend/pull/13049
```

本地验证仓库：

```text
/vllm-workspace/vllm-ascend
branch: pr-13049
HEAD: 59ce856701afe9609023f2aa8e7f7c871dec5d0c style: apply ruff fixes to weight transfer changes
base: origin/main 6fadabbfb5e18c60aa328845b3145d91a8d2b955
```

本地 `origin/main` 更新后，本次 PR 当前真实 diff 范围为 12 个文件，统计为：

```text
12 files changed, 537 insertions(+), 184 deletions(-)
```

涉及文件：

```text
examples/rl/rlhf_http_hccl.py
examples/rl/rlhf_http_npu_ipc.py
examples/rl/weight_transfer_http_utils.py
tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py
tests/e2e/pull_request/weight_transfer_utils.py
tests/ut/distributed/weight_transfer/test_compat.py
tests/ut/distributed/weight_transfer/test_http_utils.py
tests/ut/distributed/weight_transfer/test_registry.py
vllm_ascend/distributed/weight_transfer/__init__.py
vllm_ascend/distributed/weight_transfer/compat.py
vllm_ascend/distributed/weight_transfer/registry.py
```

说明：更新 `origin/main` 前曾用旧本地基线误算出 71 文件/12 提交，并误报非 weight_transfer 的 whitespace 问题；执行 `git fetch origin main` 后，基线变为 `6fadabbfb`，PR 口径恢复为 GitHub 页面一致的 12 文件/3 提交，`git diff --check` 通过。

## 2. 验证环境

服务器能力满足本次验证：

- `pytest` / `requests` / `ray` / `torch_npu` 已安装。
- `vllm` 可导入，版本为 `0.26.0`。
- `npu-smi` 可用。
- 8 张 Ascend910 在线。
- `/mnt/weight/Qwen3-0.6B` 存在本地模型文件，可用于离线 e2e。

本次实际使用的 NPU：

- one-card NPU IPC：测试代码默认物理卡 0。
- two-card HCCL：server 使用物理卡 0，trainer 使用逻辑设备 1。

## 3. 执行结果

### 3.1 静态检查

```text
git status --short --branch: passed, ## pr-13049
git log --oneline origin/main..HEAD: 3 commits
  59ce85670 style: apply ruff fixes to weight transfer changes
  22bff546e fix(weight-transfer): detect stateful trainer capability
  0582a44dd refactor(weight-transfer): centralize registration and HTTP helpers
git diff --shortstat origin/main...HEAD: 12 files changed, 537 insertions(+), 184 deletions(-)
git diff --name-only origin/main...HEAD: matches GitHub PR current 12-file scope
git diff --check origin/main...HEAD: passed
python -m compileall -q ...: passed
```

### 3.2 UT

```text
python -m pytest -q tests/ut/distributed/weight_transfer/test_compat.py tests/ut/distributed/weight_transfer/test_registry.py tests/ut/distributed/weight_transfer/test_http_utils.py
  -> 10 passed, 14 warnings

python -m pytest -q tests/ut/distributed/weight_transfer
  -> 16 passed, 14 warnings
```

覆盖确认：

- factory 不存在时走 legacy。
- factory 存在但 trainer `_registry` 无 `ipc` 时走 legacy。
- trainer `_registry` 有 `ipc` 时启用 stateful。
- registry 注册 `hccl` / `npu_ipc` native backend。
- 当前 PR 的 UT 明确期望 `nccl` / `ipc` alias 也被写入 Ascend lazy loader。
- HTTP helper 的 URL 拼接、timeout、status 检查和 background POST 异常传播均由 UT 覆盖。

### 3.3 import / lazy-load smoke

```text
vllm: 0.26.0
factory: <class 'vllm.distributed.weight_transfer.factory.WeightTransferTrainerFactory'>
supports_stateful: False
uses_legacy: True
ray False
torch_npu True
```

解释：本机 vLLM 0.26 环境符合计划预期，factory 类存在但 upstream trainer registry 没有 `ipc`，因此 `supports_stateful=False` / `uses_legacy=True`。

`torch_npu True` 来自 vLLM 平台插件激活 Ascend 平台，不代表 `compat.py` / `registry.py` 自身顶层直接 import HCCL 或 NPU IPC engine。源码层面 `registry.py` 保存的是 lazy loader，engine 模块在 factory 创建 engine 时才 import。

### 3.4 版本矩阵

本次只在当前本机 vLLM 0.26.0 环境实跑：

```text
v0.26.0: legacy, supports_stateful=False, uses_legacy=True
```

未额外创建独立 venv 覆盖 v0.25.0 / v0.27.0 / main。对应能力场景由 `test_compat.py` 和 `test_registry.py` 的 mock UT 覆盖，但这不等价于真实多版本环境验证。

### 3.5 one-card NPU IPC e2e

标准 pytest 命令：

```text
python -m pytest -q tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
  -> failed before weight-transfer path
```

失败原因：测试默认 server model 是 `Qwen/Qwen3-0.6B`，当前服务器访问 HuggingFace 失败，server 在加载 config 阶段退出：

```text
OSError: Can't load the configuration of 'Qwen/Qwen3-0.6B'
```

随后用本地模型路径覆盖测试常量后执行同一测试函数：

```text
MODEL_NAME = /mnt/weight/Qwen3-0.6B
backend = ipc
result = passed
```

确认路径：server 启动、`NPUIPCWeightTransferEngine` 创建、baseline completions、`init_weight_transfer_engine`、`pause`、`start_weight_update`、`update_weights`、`finish_weight_update`、`resume`、更新后 completions 均完成。

### 3.6 two-card HCCL e2e

用本地模型路径覆盖测试常量后执行同一测试函数：

```text
MODEL_NAME = /mnt/weight/Qwen3-0.6B
backend = hccl
server device = physical NPU 0
trainer device = logical npu:1
result = passed
```

确认路径：server 启动、`HCCLWeightTransferEngine` 创建、baseline completions、HCCL rendezvous、`init_weight_transfer_engine` background POST、`pause`、`start_weight_update`、310 tensors packed broadcast、`update_weights` background POST、`finish_weight_update`、`resume`、更新后 completions 均完成。

## 4. review.md 评审意见复核

### 4.1 `nccl` / `ipc` alias 无条件覆盖 upstream registry

结论：评审意见成立，当前状态建议修改。

证据：

- `vllm_ascend/distributed/weight_transfer/registry.py` 中 `include_upstream_aliases` 默认是 `True`。
- 同文件会把 `nccl -> hccl`、`ipc -> npu_ipc` 直接写进 `WeightTransferEngineFactory._registry`。
- `vllm_ascend/distributed/weight_transfer/__init__.py` 的 `register_engine()` 调用没有传参，因此走默认 alias 覆盖。
- `vllm_ascend/__init__.py` 的 `register_connector()` 会调用 `register_engine()`。
- `setup.py` 的 `vllm.general_plugins` 注册了 `ascend_kv_connector = vllm_ascend:register_connector`。

实测 probe：

```text
before: {'nccl': 'function', 'ipc': 'function', 'hccl': None, 'npu_ipc': None}
after : {'nccl': 'function', 'ipc': 'function', 'hccl': 'function', 'npu_ipc': 'function'}
changed: {'nccl': True, 'ipc': True, 'hccl': True, 'npu_ipc': True}
```

这说明当前注册确实会覆盖 upstream 已有 `nccl` / `ipc` entry。PR 的 UT 也把 alias 覆盖作为预期行为固定下来。因此如果合入目标是不影响非 Ascend/CUDA 进程或其他插件同名 backend，这一项需要改；最低限度应做平台判断、显式开关，或不默认注册 alias，并同步更新示例/测试为 native backend 名称。

### 4.2 示例按文档 `python rlhf_http_*.py` 直接运行会导入失败

结论：评审意见成立，当前状态建议修改。

证据：

- `examples/rl/rlhf_http_hccl.py` 使用 `from examples.rl.weight_transfer_http_utils import ...`。
- `examples/rl/rlhf_http_npu_ipc.py` 使用同样导入方式。
- 仓库中没有 `examples/__init__.py` 和 `examples/rl/__init__.py`。
- 文档注释写的是在脚本目录直接执行：`python rlhf_http_hccl.py` / `python rlhf_http_npu_ipc.py`。

复现结果：模拟脚本目录作为 `sys.path[0]` 时：

```text
ModuleNotFoundError: No module named 'examples'
```

建议改成本目录导入，例如 `from weight_transfer_http_utils import ...`，或者把文档改成并测试 `python -m examples.rl.rlhf_http_hccl` / `python -m examples.rl.rlhf_http_npu_ipc`。从当前文档口径看，代码需要改。

### 4.3 NPU IPC 示例覆盖 OpenAI client

结论：评审意见成立，建议顺手修。

`examples/rl/rlhf_http_npu_ipc.py` 先创建 OpenAI client 用于 completions，非 v0.26 分支中又把同名变量覆盖成 `HTTPVLLMWeightSyncClient`，随后调用：

```text
outputs_updated = generate_completions(client, MODEL_NAME, prompts)
```

但当前 vLLM 的 `HTTPVLLMWeightSyncClient` 没有 `completions` 属性，因此在 stateful trainer 分支会失败。应改用不同变量名，例如 `weight_sync_client`，保留原 OpenAI client 给后续 completions 使用。

## 5. 结论

本次 PR 13049 的 weight_transfer 核心验证在当前 vLLM 0.26.0 + Ascend 环境下通过：静态检查、compileall、UT、one-card NPU IPC 本地模型 e2e、two-card HCCL 本地模型 e2e 均通过。

但 `review.md` 的前两条评审意见仍然是真的，并且会影响合入质量：

- alias 覆盖属于当前 PR 明确引入/固定的行为，存在非 Ascend 或其他插件场景下覆盖 upstream backend 的风险。
- 示例导入方式和文档运行方式不一致，按文档直接运行会失败。

第三条 NPU IPC 示例覆盖 `client` 也是真问题，虽然它只影响非 v0.26 的 stateful trainer 分支。

最终建议：当前不建议按“完全无修改”合入；至少先处理前两条评审意见，第三条建议一并修掉。验证结果本身说明 weight-transfer 主路径可用，但 review 暴露的是注册边界和示例入口问题。

## 6. 本轮命令摘要

```text
git fetch origin main
git status --short --branch
git log --oneline origin/main..HEAD
git diff --shortstat origin/main...HEAD
git diff --name-only origin/main...HEAD
git diff --check origin/main...HEAD
python -m compileall -q vllm_ascend/distributed/weight_transfer examples/rl/rlhf_http_hccl.py examples/rl/rlhf_http_npu_ipc.py examples/rl/weight_transfer_http_utils.py tests/e2e/pull_request/weight_transfer_utils.py tests/ut/distributed/weight_transfer
python -m pytest -q tests/ut/distributed/weight_transfer/test_compat.py tests/ut/distributed/weight_transfer/test_registry.py tests/ut/distributed/weight_transfer/test_http_utils.py
python -m pytest -q tests/ut/distributed/weight_transfer
python -m pytest -q tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
python local-model override for one-card NPU IPC e2e
python local-model override for two-card HCCL e2e
```
