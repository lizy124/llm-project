#!/bin/bash
cd /root/kv_metrics_ut
PYTHONPATH=/root/kv_metrics_ut python -m pytest \
  "tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVTransferTpMismatchDispatch::test_recving_dispatches_to_worker_when_tp_mismatch" \
  "tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerHelpers::test_wait_for_layer_load_fallback_waits_for_reuse" \
  -v 2>&1 | grep -vE "DeprecationWarning|warnings.warn|Docs:|warnings summary|torch/jit" | tail -60
