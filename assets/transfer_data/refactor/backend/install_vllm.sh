#!/bin/bash
exec > /tmp/vllm_install.log 2>&1
set -x
echo "=== START vllm install $(date) ==="
cd /vllm-workspace/vllm || exit 1
pip install -e . 2>&1
echo "=== END vllm install rc=$? $(date) ==="
