#!/bin/bash
# Run UT for issue 14148 changes (block_hash_to_str dedup)
# Usage: bash run_14148_ut.sh [filter]
set -o pipefail
cd /tmp/va_14148
export PYTHONPATH=/tmp/vllm_14148
FILTER="${1:-tests/ut/distributed/ascend_store/test_pool_worker.py}"
python3 -m pytest "$FILTER" -q 2>&1 | tail -40