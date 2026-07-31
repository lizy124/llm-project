#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(make_run_dir check)"
else
  RUN_DIR="$(require_run_dir "${RUN_DIR}")"
fi

LOG="${RUN_DIR}/check_import_ut.log"
cd /vllm-workspace/vllm-ascend

{
  echo "# Import check"
  python3 - <<'PY'
import importlib
modules = [
    'vllm',
    'vllm_ascend',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker',
]
for module in modules:
    importlib.import_module(module)
    print(f'OK import {module}')
PY
  echo

  echo "# py_compile"
  python3 -m py_compile \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py
  echo "OK py_compile"
  echo

  echo "# pytest"
  if python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('pytest') else 1)
PY
  then
    python3 -m pytest tests/ut/distributed/ascend_store -q
  else
    echo "SKIP pytest: pytest not installed"
  fi
} > "${LOG}" 2>&1

echo "RUN_DIR=${RUN_DIR}"
echo "CHECK_LOG=${LOG}"
