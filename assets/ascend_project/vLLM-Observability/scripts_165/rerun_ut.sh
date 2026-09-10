#!/bin/bash
# Re-pull the branch and re-run the full ascend_store UT suite.
set -e
WORK=/root/kv_metrics_ut
cd "$WORK"
git fetch origin kv_metrics_observability 2>&1 | tail -1
git reset --hard origin/kv_metrics_observability >/dev/null
git log --oneline -1
PYTHONPATH="$WORK" python -m pytest tests/ut/distributed/ascend_store/ 2>&1 | tail -4
