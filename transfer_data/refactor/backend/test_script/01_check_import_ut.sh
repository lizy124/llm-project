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
  printf '# Required import check\n'
  python3 - <<'PY'
import importlib
required = [
    'vllm',
    'vllm_ascend',
    'vllm_ascend.distributed.kv_transfer',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.ascend_store_connector',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.config_data',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.coordinator',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.kv_transfer',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.pool_scheduler',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.store_utils.pool_worker',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_backend.mooncake_backend',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.memcache_backend.memcache_backend',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.yuanrong_backend.yuanrong_backend',
]
optional = [
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ucm_connector.ucm_connector',
    'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.lmcache_ascend_connector.lmcache_ascend_connector',
]
failed = []
for module in required:
    try:
        importlib.import_module(module)
        print(f'OK required import {module}')
    except Exception as exc:
        failed.append((module, repr(exc)))
        print(f'FAIL required import {module}: {exc!r}')
for module in optional:
    try:
        importlib.import_module(module)
        print(f'OK optional import {module}')
    except Exception as exc:
        print(f'SKIP/FAIL optional import {module}: {exc!r}')
if failed:
    raise SystemExit('required imports failed: ' + repr(failed))
PY
  printf '\n# KVConnector registry path check\n'
  python3 - <<'PY'
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm_ascend.distributed.kv_transfer import register_connector
register_connector()
for name in ['AscendStoreConnector', 'MooncakeConnectorStoreV1', 'UCMConnector', 'LMCacheAscendConnector']:
    print(f'{name}: {KVConnectorFactory._registry.get(name)}')
PY
  printf '\n# py_compile\n'
  python3 -m py_compile \
    vllm_ascend/distributed/kv_transfer/__init__.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/store_utils/ascend_store_connector.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/store_utils/config_data.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/store_utils/coordinator.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/store_utils/kv_transfer.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/store_utils/pool_scheduler.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/store_utils/pool_worker.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mooncake_backend/mooncake_backend.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/memcache_backend/memcache_backend.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/yuanrong_backend/yuanrong_backend.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ucm_connector/ucm_connector.py \
    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/lmcache_ascend_connector/lmcache_ascend_connector.py
  printf 'OK py_compile\n'
  printf '\n# pytest targeted\n'
  if python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('pytest') else 1)
PY
  then
    python3 -m pytest \
      tests/ut/distributed/ascend_store \
      tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py \
      -q
  else
    printf 'SKIP pytest: pytest not installed\n'
  fi
} > "${LOG}" 2>&1

printf 'RUN_DIR=%s\n' "${RUN_DIR}"
printf 'CHECK_LOG=%s\n' "${LOG}"
