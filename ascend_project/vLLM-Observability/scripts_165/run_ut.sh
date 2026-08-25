#!/bin/bash
# Clone the PR1 branch into an isolated dir and run the ascend_store UTs.
set -e
WORK=/root/kv_metrics_ut
rm -rf "$WORK"
git clone --depth 1 -b kv_metrics_observability https://github.com/lizy124/vllm-ascend.git "$WORK" 2>&1 | tail -2
cd "$WORK"
git log --oneline -1
echo "=== UT: test_metrics.py ==="
PYTHONPATH="$WORK" python -m pytest tests/ut/distributed/ascend_store/test_metrics.py -v 2>&1 | tail -35
echo "=== UT: full ascend_store regression ==="
PYTHONPATH="$WORK" python -m pytest tests/ut/distributed/ascend_store/ 2>&1 | tail -8
