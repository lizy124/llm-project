#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="$(require_run_dir "${1:-}")"
PORT="${MOONCAKE_MASTER_PORT:-50088}"
LOG="${RUN_DIR}/mooncake_master.log"

if ! command -v mooncake_master >/dev/null 2>&1; then
  echo "mooncake_master not found" | tee "${LOG}"
  exit 1
fi

mooncake_master --port "${PORT}" \
  --eviction_high_watermark_ratio "${MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO:-0.9}" \
  --eviction_ratio "${MOONCAKE_EVICTION_RATIO:-0.1}" \
  --default_kv_lease_ttl "${MOONCAKE_DEFAULT_KV_LEASE_TTL:-11000}" \
  > "${LOG}" 2>&1 &

PID=$!
write_pid "${RUN_DIR}" mooncake_master "${PID}"

echo "RUN_DIR=${RUN_DIR}"
echo "MOONCAKE_MASTER_PID=${PID}"
echo "MOONCAKE_MASTER_LOG=${LOG}"
