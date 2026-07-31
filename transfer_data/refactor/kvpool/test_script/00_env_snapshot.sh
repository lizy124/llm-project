#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(make_run_dir env)"
else
  RUN_DIR="$(require_run_dir "${RUN_DIR}")"
fi

OUT="${RUN_DIR}/env.txt"
{
  echo "# Time"
  date -Is
  echo

  echo "# Repositories"
  echo "## vllm-ascend"
  git -C /vllm-workspace/vllm-ascend status --short --branch || true
  git -C /vllm-workspace/vllm-ascend rev-parse HEAD || true
  echo
  echo "## vllm"
  git -C /vllm-workspace/vllm status --short --branch || true
  git -C /vllm-workspace/vllm rev-parse HEAD || true
  echo
  echo "## verified vllm commit from vllm-ascend"
  if [[ -f /vllm-workspace/vllm-ascend/.github/vllm-main-verified.commit ]]; then
    cat /vllm-workspace/vllm-ascend/.github/vllm-main-verified.commit
  fi
  echo

  echo "# Python packages"
  python3 - <<'PY'
import importlib.util
import sys
print('python:', sys.executable)
for name in ['vllm', 'vllm_ascend']:
    spec = importlib.util.find_spec(name)
    print(f'{name}: {spec.origin if spec else "NOT_FOUND"}')
PY
  python3 -m pip show vllm vllm-ascend 2>/dev/null || true
  echo

  echo "# NPU"
  npu-smi info || true
  echo

  echo "# Important env"
  env | sort | grep -E '^(ASCEND|HCCL|MOONCAKE|MMC|PYTHONHASHSEED|PYTHONPATH|LD_LIBRARY_PATH|ACL_)' || true
} > "${OUT}" 2>&1

echo "RUN_DIR=${RUN_DIR}"
echo "ENV_SNAPSHOT=${OUT}"
