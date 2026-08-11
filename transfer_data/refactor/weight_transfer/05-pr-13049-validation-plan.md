# PR 13049 验证计划

## 1. 验证对象

PR:

```text
https://github.com/vllm-project/vllm-ascend/pull/13049
```

当前本地验证仓库：

```text
/vllm-workspace/vllm-ascend
branch: pr-13049
HEAD: c5ed02f00 fix(weight-transfer): detect stateful trainer capability
```

本 PR 的实际改动范围是一个完整的 weight_transfer helper/registry/e2e helper 回合：

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
vllm_ascend/patch/platform/patch_weight_transfer_engine.py
```

PR diff 统计为 `13 files changed, 540 insertions(+), 234 deletions(-)`。它的核心目标不只是修正 stateful trainer API 的能力探测，还把 registry、HTTP helper、e2e helper 和 patch 入口一起收敛：不能只判断 upstream 是否存在 `WeightTransferTrainerFactory` 类，还必须确认 upstream trainer registry 已经注册 `ipc` backend。vLLM v0.26 可能有 factory 类但没有 stateful trainer backend，此时仍应走 legacy static trainer API。

## 2. 风险边界

这次 PR 主要重构的是 registration、HTTP helper、e2e helper 和 trainer capability 探测，因此验证重点是“helper 结构重排后行为不变”。

本 PR 不应改变：

- HCCL worker transport。
- NPU IPC worker transport。
- packed tensor wire contract。
- HTTP/Ray/callable payload schema。
- `nccl -> hccl`、`ipc -> npu_ipc` 的 worker-side alias 行为。
- import lazy-load 边界。
- e2e 用例的参数语义和断言。

重点风险有两个：

1. trainer API 误判。
2. helper 抽取后 HTTP/e2e payload 或 endpoint 参数发生回归。

```text
v0.25: 无 WeightTransferTrainerFactory -> legacy
v0.26: 有 WeightTransferTrainerFactory，但无 ipc trainer backend -> legacy
v0.27: 有 WeightTransferTrainerFactory，且有 ipc trainer backend -> stateful
main:  有 WeightTransferTrainerFactory，且有 ipc/nccl trainer backend -> stateful
```

如果 v0.26 被误判为 stateful，Ascend 注册 `npu_ipc` trainer engine 后可能暴露一个 upstream 无法驱动的调用路径。

## 3. 第一层：本地静态检查

在 `/vllm-workspace/vllm-ascend` 执行：

```bash
git status --short --branch
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
git diff --check origin/main...HEAD
python -m compileall -q \
  vllm_ascend/distributed/weight_transfer \
  vllm_ascend/patch/platform/patch_weight_transfer_engine.py \
  examples/rl/rlhf_http_hccl.py \
  examples/rl/rlhf_http_npu_ipc.py \
  examples/rl/weight_transfer_http_utils.py \
  tests/e2e/pull_request/weight_transfer_utils.py \
  tests/ut/distributed/weight_transfer
```

预期结果：

- 当前分支是 `pr-13049`。
- `origin/main..HEAD` 包含 `84b09f3e2` 和 `c5ed02f00` 两个 PR 提交。
- diff 统计为 `13 files changed, 540 insertions(+), 234 deletions(-)`。
- 文件范围只包含上文列出的 13 个文件。
- `git diff --check` 无 whitespace error。
- compileall 通过。

## 4. 第二层：纯 Python UT

优先执行 PR 新增的三组 UT：

```bash
python -m pytest -q \
  tests/ut/distributed/weight_transfer/test_compat.py \
  tests/ut/distributed/weight_transfer/test_registry.py \
  tests/ut/distributed/weight_transfer/test_http_utils.py
```

然后执行完整 weight_transfer UT 目录：

```bash
python -m pytest -q tests/ut/distributed/weight_transfer
```

必须覆盖以下断言：

- `get_trainer_factory()` 返回 `None` 时，`uses_legacy_trainer_api()` 为 `True`。
- trainer factory 存在但 `_registry` 为空时，`supports_stateful_trainer_api()` 为 `False`。
- trainer factory 的 `_registry` 包含 `ipc` 时，`supports_stateful_trainer_api()` 为 `True`。
- registry 注册 `hccl`、`npu_ipc` native backend。
- registry 将 upstream `nccl`、`ipc` alias 替换为 Ascend lazy loader。
- registry 注册函数只在 stateful trainer API 可用时注册 Ascend `npu_ipc` trainer engine。
- 重复注册保持幂等。
- HTTP helper 正确拼接 URL、传递 timeout、检查 HTTP status，并传播 background POST 异常。
- e2e helper 不改变 backend-specific payload 构造。

## 5. 第三层：人工 diff 审查

按职责逐组审查 13 个文件：

| 文件组 | 审查重点 |
|---|---|
| `compat.py` / `test_compat.py` | trainer capability 使用 registry entry 判断，而不是只看 factory 类存在 |
| `registry.py` / `test_registry.py` | native backend、alias replacement、trainer registry、幂等和 lazy loader |
| `__init__.py` / `patch_weight_transfer_engine.py` | 两个入口都只触发统一 registry，不再分散写 upstream registry |
| `weight_transfer_http_utils.py` / `test_http_utils.py` | HTTP URL、timeout、status 检查、异常传播，不编码 backend payload |
| `rlhf_http_hccl.py` / `rlhf_http_npu_ipc.py` | 示例只复用 helper，不改变 HCCL/NPU IPC payload 和 lifecycle 顺序 |
| `weight_transfer_utils.py` / e2e tests | e2e 公共请求 helper 不吞异常，不改变原测试语义 |

审查时不要只看新增 helper 是否“更干净”，要逐项确认旧示例和旧 e2e 中的 backend-specific 参数仍由调用方持有。

## 6. 第四层：import 和 lazy-load smoke test

确认兼容模块和注册模块不会提前 import transport 依赖：

```bash
python - <<'PY'
import sys

from vllm_ascend.distributed.weight_transfer import compat
from vllm_ascend.distributed.weight_transfer.registry import (
    register_ascend_weight_transfer_engines,
)

register_ascend_weight_transfer_engines()

print("factory:", compat.get_trainer_factory())
print("supports_stateful:", compat.supports_stateful_trainer_api())
print("uses_legacy:", compat.uses_legacy_trainer_api())

for name in ["ray", "torch_npu"]:
    print(name, name in sys.modules)
PY
```

预期结果：

- 脚本可以正常退出。
- `ray` 不应因 registry/compat 顶层路径被提前 import。
- `torch_npu` 不应因 registry/compat 顶层路径被提前 import。
- `supports_stateful` 的结果应与当前安装的 upstream vLLM trainer registry 一致。

## 7. 第四层：版本矩阵验证

这个 PR 的关键价值需要跨 upstream vLLM 版本确认。推荐使用独立 venv 或临时工作区，不要在主验证目录里直接来回覆盖当前环境。

| upstream vLLM | 期望结果 | 必测点 |
|---|---|---|
| `releases/v0.25.0` | legacy | factory 不存在时不报错 |
| `releases/v0.26.0` | legacy | factory 存在但无 `ipc` 时不启用 stateful |
| `releases/v0.27.0` | stateful | `ipc` 注册后启用 stateful |
| `main` | stateful | `ipc`/`nccl` 注册后启用 stateful |

每个环境执行：

```bash
python -m pytest -q tests/ut/distributed/weight_transfer/test_compat.py
python - <<'PY'
from vllm_ascend.distributed.weight_transfer import compat
print("factory:", compat.get_trainer_factory())
print("supports_stateful:", compat.supports_stateful_trainer_api())
print("uses_legacy:", compat.uses_legacy_trainer_api())
PY
```

判定标准：

- v0.25/v0.26 输出 `supports_stateful: False` 和 `uses_legacy: True`。
- v0.27/main 输出 `supports_stateful: True` 和 `uses_legacy: False`。
- 任一版本 import `compat.py` 不应因为不存在 newer trainer API 而失败。

## 8. 第五层：Ascend e2e 回归

本 PR 没有改 HCCL/NPU IPC data-plane，因此 e2e 不是最小合入门槛，但在完整 Ascend 环境中建议跑一遍核心路径，确认 trainer API 分流没有破坏现有用例。

```bash
python -m pytest -q tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py
python -m pytest -q tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py
```

预期结果：

- NPU IPC one-card weight transfer 通过。
- HCCL two-card weight transfer 通过。
- 如果环境缺少 NPU、CANN、torch_npu、pytest 或 requests，应记录为环境阻塞，而不是 PR 行为失败。

## 9. 遗留项

当前已经完成的验证覆盖了 helper、UT 和两条 e2e 主路径，但还有一项额外降风险验证没有做：

- `examples/rl/rlhf_http_hccl.py` 还没有作为 CLI 脚本单独跑一次。
- `examples/rl/rlhf_http_npu_ipc.py` 还没有作为 CLI 脚本单独跑一次。

后续补做时建议：

- 使用本地 `/mnt/weight/Qwen3-0.6B`，避免依赖外网。
- 保持离线环境或模型本地缓存可用。
- 把两条脚本验证作为 post-merge 或下一轮收尾 smoke。

这项目前不是合入阻塞项，但它仍然是值得补齐的最后一层入口验证。

## 10. Review 检查清单

合入前至少确认：

- `supports_stateful_trainer_api()` 使用 upstream `ipc` trainer registry entry 作为 capability signal。
- v0.26 的“factory 存在但 backend 未注册”场景已有 UT。
- `uses_legacy_trainer_api()` 只是 `supports_stateful_trainer_api()` 的反向结果，不引入第二套判断。
- `compat.py` 不顶层 import Ray、torch_npu、HCCL 或 NPU IPC engine。
- `registry.py` 对 trainer engine 的注册仍由 `supports_stateful_trainer_api()` 保护。
- 没有顺带修改 transport loop、packed payload 或 HTTP schema。

## 11. 验收结论模板

完成验证后记录：

```text
PR: #13049
branch: pr-13049
HEAD: c5ed02f00

static checks:
  git diff --check: <passed/failed>
  compileall: <passed/failed>

unit tests:
  test_compat.py: <passed/failed>
  weight_transfer UT: <passed/failed/skipped + reason>

version matrix:
  v0.25.0: <legacy/stateful/result>
  v0.26.0: <legacy/stateful/result>
  v0.27.0: <legacy/stateful/result>
  main: <legacy/stateful/result>

Ascend e2e:
  npu_ipc one-card: <passed/failed/skipped + reason>
  hccl two-card: <passed/failed/skipped + reason>

final decision:
  <pass/fail/blocker>
```
