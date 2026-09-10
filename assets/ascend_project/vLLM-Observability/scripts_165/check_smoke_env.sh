#!/bin/bash
echo "=== vllm metrics framework present? ==="
python3 - <<'PYEOF'
import vllm.distributed.kv_transfer.kv_connector.v1.metrics as m
import inspect
print("metrics.py OK:", m.__file__)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
for hook in ("get_kv_connector_stats", "build_kv_connector_stats", "build_prom_metrics"):
    print(hook, hasattr(KVConnectorBase_V1, hook))
PYEOF
echo "=== PYTHONPATH override check ==="
PYTHONPATH=/root/kv_metrics_ut python3 -c "import vllm_ascend; print(vllm_ascend.__file__)" 2>/dev/null | tail -1
echo "=== NPU usage ==="
npu-smi info 2>/dev/null | head -25
echo "=== models available ==="
ls /root/.cache/huggingface 2>/dev/null || true
ls /home/hucong/vllm-ascend-workspace/test_log 2>/dev/null | head -5 || true
