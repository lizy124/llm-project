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
  printf '# Time\n'
  date -Is
  printf '\n'

  printf '# Repositories\n'
  printf '## vllm-ascend\n'
  git -C /vllm-workspace/vllm-ascend status --short --branch || true
  git -C /vllm-workspace/vllm-ascend rev-parse HEAD || true
  git -C /vllm-workspace/vllm-ascend rev-parse --abbrev-ref HEAD || true
  printf '\n'
  printf '## vllm\n'
  git -C /vllm-workspace/vllm status --short --branch || true
  git -C /vllm-workspace/vllm rev-parse HEAD || true
  git -C /vllm-workspace/vllm rev-parse --abbrev-ref HEAD || true
  printf '\n'
  printf '## verified vllm commit from vllm-ascend\n'
  python3 - <<'PY'
from pathlib import Path
path = Path('/vllm-workspace/vllm-ascend/.github/vllm-main-verified.commit')
print(path.read_text(encoding='utf-8').strip() if path.exists() else 'MISSING')
PY
  printf '\n'

  printf '# Python packages\n'
  python3 - <<'PY'
import importlib.util
import sys
print('python:', sys.executable)
for name in ['vllm', 'vllm_ascend', 'mooncake', 'memcache_hybrid', 'lmcache_ascend', 'ucm']:
    spec = importlib.util.find_spec(name)
    print(f'{name}: {spec.origin if spec else "NOT_FOUND"}')
PY
  python3 -m pip show vllm vllm-ascend mooncake memcache-hybrid lmcache-ascend ucm 2>/dev/null || true
  printf '\n'

  printf '# NPU\n'
  npu-smi info || true
  printf '\n'

  printf '# Important env\n'
  python3 - <<'PY'
import os
prefixes = ('ASCEND', 'HCCL', 'MOONCAKE', 'MMC', 'PYTHONHASHSEED', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'ACL_', 'DS_')
for key in sorted(os.environ):
    if key.startswith(prefixes):
        print(f'{key}={os.environ[key]}')
PY
} > "${OUT}" 2>&1

printf 'RUN_DIR=%s\n' "${RUN_DIR}"
printf 'ENV_SNAPSHOT=%s\n' "${OUT}"
