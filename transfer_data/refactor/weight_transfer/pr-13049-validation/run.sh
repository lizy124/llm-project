#!/usr/bin/env bash
set -euo pipefail

ROOT=/vllm-workspace/vllm-ascend
RUN_DIR=$(cd "$(dirname "$0")" && pwd)

cd "$ROOT"

# 1. Static checks
python -m compileall -q \
  vllm_ascend/distributed/weight_transfer \
  vllm_ascend/patch/platform/patch_weight_transfer_engine.py \
  examples/rl/rlhf_http_hccl.py \
  examples/rl/rlhf_http_npu_ipc.py \
  examples/rl/weight_transfer_http_utils.py \
  tests/e2e/pull_request/weight_transfer_utils.py \
  tests/ut/distributed/weight_transfer

# 2. Unit tests
python -m pytest -q \
  tests/ut/distributed/weight_transfer/test_compat.py \
  tests/ut/distributed/weight_transfer/test_registry.py \
  tests/ut/distributed/weight_transfer/test_http_utils.py
python -m pytest -q tests/ut/distributed/weight_transfer

# 3. One-card NPU IPC e2e
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
WEIGHT_TRANSFER_TEST_MODEL=/mnt/weight/Qwen3-0.6B \
ASCEND_RT_VISIBLE_DEVICES=6,7 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
python - <<'PY'
import traceback
from tests.e2e.pull_request.one_card import test_npu_ipc_weight_transfer as t

MODEL_PATH = "/mnt/weight/Qwen3-0.6B"
CARD_ID = 6

t.MODEL_NAME = MODEL_PATH
t.INFERENCE_DEVICE_INDEX = CARD_ID

try:
    t.test_npu_ipc_weight_transfer_updates_server_weights()
except Exception:
    traceback.print_exc()
    raise
PY

# 4. Two-card HCCL e2e
ASCEND_RT_VISIBLE_DEVICES=6,7 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
python - <<'PY'
import traceback
from tests.e2e.pull_request.two_card import test_hccl_weight_transfer as t

MODEL_PATH = "/mnt/weight/Qwen3-0.6B"
SERVER_CARD = 6
TRAINER_DEVICE_INDEX = 1

t.MODEL_NAME = MODEL_PATH
t.INFERENCE_WORLD_SIZE = 1
t.TRAINER_DEVICE_INDEX = TRAINER_DEVICE_INDEX

OrigRemoteOpenAIServer = t.RemoteOpenAIServer

class PatchedRemoteOpenAIServer(OrigRemoteOpenAIServer):
    def __init__(self, *args, **kwargs):
        env_dict = kwargs.get('env_dict')
        if env_dict is None:
            env_dict = {}
            kwargs['env_dict'] = env_dict
        env_dict['ASCEND_RT_VISIBLE_DEVICES'] = str(SERVER_CARD)
        super().__init__(*args, **kwargs)


t.RemoteOpenAIServer = PatchedRemoteOpenAIServer

try:
    t.test_hccl_weight_transfer_updates_server_weights()
except Exception:
    traceback.print_exc()
    raise
PY
