#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="$(require_run_dir "${1:-}")"
MODEL="${MODEL_PATH:-/mnt/weight/Qwen3-0.6B}"
PORT="${PORT:-8100}"
DEVICE="${ASCEND_RT_VISIBLE_DEVICES:-0}"
LOG="${RUN_DIR}/server_baseline.log"

export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"
export PYTHONPATH="${PYTHONPATH:-}:/vllm-workspace/vllm"
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:${LD_LIBRARY_PATH:-}"

cd /vllm-workspace/vllm-ascend

python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --trust-remote-code \
  --enforce-eager \
  --no-enable-prefix-caching \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --data-parallel-size "${DP_SIZE:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --block-size "${BLOCK_SIZE:-128}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  > "${LOG}" 2>&1 &

PID=$!
write_pid "${RUN_DIR}" server_baseline "${PID}"

echo "RUN_DIR=${RUN_DIR}"
echo "SERVER_PID=${PID}"
echo "SERVER_LOG=${LOG}"
echo "MODEL=${MODEL}"
echo "PORT=${PORT}"
