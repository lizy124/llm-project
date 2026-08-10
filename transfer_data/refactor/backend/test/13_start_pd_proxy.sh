#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="$(require_run_dir "${1:-}")"
HOST="${PROXY_HOST:-127.0.0.1}"
PORT="${PROXY_PORT:-9000}"
PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
DECODE_PORT="${DECODE_PORT:-8200}"
LOG="${RUN_DIR}/pd_proxy.log"

cd /vllm-workspace/vllm-ascend

python3 examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --prefiller-hosts "${PREFILL_HOST}" \
  --prefiller-ports "${PREFILL_PORT}" \
  --decoder-hosts "${DECODE_HOST}" \
  --decoder-ports "${DECODE_PORT}" \
  > "${LOG}" 2>&1 &

PID=$!
write_pid "${RUN_DIR}" pd_proxy "${PID}"

{
  echo "PROXY_HOST=${HOST}"
  echo "PROXY_PORT=${PORT}"
  echo "PREFILL_HOST=${PREFILL_HOST}"
  echo "PREFILL_PORT=${PREFILL_PORT}"
  echo "DECODE_HOST=${DECODE_HOST}"
  echo "DECODE_PORT=${DECODE_PORT}"
} > "${RUN_DIR}/pd_proxy_env.txt"

echo "RUN_DIR=${RUN_DIR}"
echo "PROXY_PID=${PID}"
echo "PROXY_LOG=${LOG}"
echo "PROXY_PORT=${PORT}"
