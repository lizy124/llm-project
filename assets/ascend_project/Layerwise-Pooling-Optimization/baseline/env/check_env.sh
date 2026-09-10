#!/bin/bash
# check_env.sh — baseline_165 预装环境核查（playbook §8 清单 + DSV4 支持）
set -uo pipefail

echo "=== [1] vllm workspace git ==="
cd /vllm-workspace/vllm && git log --oneline -1 && git status --short | head -5

echo "=== [2] vllm-ascend workspace git ==="
cd /vllm-workspace/vllm-ascend && git log --oneline -1 && git status --short | head -5

echo "=== [3] pip 两包版本/Editable ==="
pip show vllm 2>/dev/null | grep -E "^(Version|Location|Editable)"
pip show vllm-ascend 2>/dev/null | grep -E "^(Version|Location|Editable)"

echo "=== [4] triton 状态 ==="
pip list 2>/dev/null | grep -iE "^triton"

echo "=== [5] DSV4 arch 支持 ==="
python3 - <<'EOF'
from vllm.model_executor.models.registry import ModelRegistry
archs = [a for a in ModelRegistry.get_supported_archs() if "V4" in a or "V2" in a]
print("V4/V2 archs:", archs)
EOF

echo "=== [6] benchmark 工具 ==="
ls /vllm-workspace/vllm/benchmarks/benchmark_serving.py 2>/dev/null || echo "MISSING benchmark_serving.py"

echo "=== [7] NPU 可见 ==="
npu-smi info | grep -c 0000:

echo "=== [8] 配对文件（vllm-ascend 期望的 vllm 版本）==="
cat /vllm-workspace/vllm-ascend/.github/vllm-release-tag.commit 2>/dev/null
cat /vllm-workspace/vllm-ascend/.github/vllm-main-verified.commit 2>/dev/null

echo "=== DONE ==="
