#!/bin/bash
exec > /tmp/vllm_ascend_install.log 2>&1
set -x
echo "=== START vllm-ascend install $(date) ==="
cd /vllm-workspace/vllm-ascend || exit 1
pip install -e . 2>&1
echo "=== END vllm-ascend install rc=$? $(date) ==="
