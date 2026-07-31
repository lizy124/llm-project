#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"

make_run_dir() {
  local case_name="${1:-manual}"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local run_dir="${RUNS_DIR}/${ts}_${case_name}"
  mkdir -p "${run_dir}"
  printf '%s\n' "${run_dir}"
}

require_run_dir() {
  local run_dir="${1:-}"
  if [[ -z "${run_dir}" ]]; then
    echo "usage: $0 <run_dir>" >&2
    exit 2
  fi
  mkdir -p "${run_dir}"
  printf '%s\n' "${run_dir}"
}

write_pid() {
  local run_dir="$1"
  local name="$2"
  local pid="$3"
  printf '%s\n' "${pid}" > "${run_dir}/${name}.pid"
}
