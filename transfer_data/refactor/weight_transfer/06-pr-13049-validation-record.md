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
HEAD: c5ed02f00 fix(weight-transfer): detect stateful trainer capability
base: origin/main ac19e1e647785be51d22a87f336ba03c02357e18
```

这次 PR 的真实 diff 范围为 13 个文件，统计为：

```text
13 files changed, 540 insertions(+), 234 deletions(-)
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
vllm_ascend/patch/platform/patch_weight_transfer_engine.py
```

## 2. 验证环境

服务器能力满足本次验证：

- `pytest` / `requests` / `ray` / `torch_npu` 已安装。
- `vllm` 可导入，版本为 `0.26.0+empty`。
- `npu-smi` 可用。
- 8 张 Ascend910 在线。
- `/mnt/weight/Qwen3-0.6B` 存在本地模型文件，可用于离线 e2e。

本次实际使用的 NPU：

- one-card NPU IPC：物理卡 6
- two-card HCCL：物理卡 6 作为 server，物理卡 7 作为 trainer

## 3. 执行结果

### 3.1 静态检查

```text
git status --short --branch: passed
git log --oneline origin/main..HEAD: 2 commits
git diff --shortstat origin/main...HEAD: 13 files changed, 540 insertions(+), 234 deletions(-)
git diff --name-only origin/main...HEAD: matches PR file list
git diff --check origin/main...HEAD: passed
python -m compileall -q ...: passed
```

### 3.2 UT

```text
python -m pytest -q tests/ut/distributed/weight_transfer/test_compat.py tests/ut/distributed/weight_transfer/test_registry.py tests/ut/distributed/weight_transfer/test_http_utils.py
  -> 10 passed

python -m pytest -q tests/ut/distributed/weight_transfer
  -> 16 passed
```

### 3.3 one-card NPU IPC e2e

```text
result: passed
model: /mnt/weight/Qwen3-0.6B
physical card: 6
```

说明：

- 起初尝试在线拉 `Qwen/Qwen3-0.6B`，但服务器网络不可达。
- 随后改用本地模型路径 `/mnt/weight/Qwen3-0.6B`，验证通过。

### 3.4 two-card HCCL e2e

```text
result: passed
model: /mnt/weight/Qwen3-0.6B
server physical card: 6
trainer logical device: 1 under ASCEND_RT_VISIBLE_DEVICES=6,7
```

说明：

- 起初把 trainer 设成逻辑设备 `7`，在当前 `ASCEND_RT_VISIBLE_DEVICES=6,7` 下无效。
- 重跑时把 trainer 改成逻辑设备 `1`，对应可见集里的第二张卡，验证通过。

## 4. 结论

本次 PR 13049 在当前服务器上验证通过。

可确认的点：

- `supports_stateful_trainer_api()` 的能力探测逻辑可通过 UT 覆盖。
- `registry` / HTTP helper / e2e helper 的收敛没有破坏现有 weight transfer 路径。
- one-card NPU IPC 和 two-card HCCL 的主路径都能在本机验证。

遗留项：

- `examples/rl/rlhf_http_hccl.py` 还没有单独做 CLI smoke。
- `examples/rl/rlhf_http_npu_ipc.py` 还没有单独做 CLI smoke。

这两项不影响当前合入判断，但适合作为后续补充验证。

## 5. 详细日志和过程目录

正式过程目录保留在：

```text
/home/lizhongyang/refactor/llm-project/transfer_data/refactor/weight_transfer/pr-13049-validation/
```

其中包括：

- `README.md`
- `run.sh`
- `env.txt`
- `summary.json`
- `log_extract.txt`
- `static_checks.log`
- `unit_tests.log`
- `e2e_one_card_npu_ipc.log`
- `e2e_one_card_npu_ipc_offline.log`
- `e2e_one_card_npu_ipc_custom.log`
- `e2e_two_card_hccl_custom.log`
- `e2e_two_card_hccl_custom_retry.log`
- `npu_device_probe.log`

`backend/runs/20260810_094500_env/` 只作为最初参考格式和原始临时日志来源，不作为正式记录位置。

