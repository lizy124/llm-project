#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="$(require_run_dir "${1:-}")"
OUT="${RUN_DIR}/log_extract.txt"

{
  echo "# Extracted KV Pool related logs"
  echo
  for log in "${RUN_DIR}"/*.log; do
    [[ -f "${log}" ]] || continue
    echo "## ${log}"
    grep -Ein "AscendStore|Mooncake|KV Pool|kv pool|lookup|put|get|load|save|hit|miss|failure|recompute|error|exception|traceback|finished|free" "${log}" || true
    echo
  done
} > "${OUT}"

echo "RUN_DIR=${RUN_DIR}"
echo "LOG_EXTRACT=${OUT}"
