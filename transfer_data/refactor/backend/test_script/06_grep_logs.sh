#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="$(require_run_dir "${1:-}")"
OUT="${RUN_DIR}/log_extract.txt"

{
  printf '# Extracted backend/KV Pool related logs\n\n'
  for log in "${RUN_DIR}"/*.log; do
    [[ -f "${log}" ]] || continue
    printf '## %s\n' "${log}"
    grep -Ein "AscendStore|Mooncake|KV Pool|kv pool|backend|mooncake_backend|memcache_backend|yuanrong_backend|store_utils|lookup|put|get|load|save|hit|miss|failure|recompute|error|exception|traceback|finished|free" "${log}" || true
    printf '\n'
  done
} > "${OUT}"

printf 'RUN_DIR=%s\n' "${RUN_DIR}"
printf 'LOG_EXTRACT=%s\n' "${OUT}"
