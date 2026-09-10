#!/bin/bash
# recon_s1.sh — 场景1 前置侦察(容器内执行)
echo "===== s2 env.txt ====="
cat /home/lizhongyang/map_165/run/s2_memcache_layerwise/env.txt 2>/dev/null
echo
echo "===== which python3 (default) ====="
which python3
echo "===== login shell python ====="
bash -lc 'which python3' 2>/dev/null
echo "===== vllm_ascend location ====="
python3 - <<'PY'
try:
    import vllm_ascend, os
    print("vllm_ascend:", os.path.dirname(vllm_ascend.__file__))
except Exception as e:
    print("ERR vllm_ascend:", e)
try:
    import vllm, os
    print("vllm:", os.path.dirname(vllm.__file__), getattr(vllm, "__version__", "?"))
except Exception as e:
    print("ERR vllm:", e)
PY
echo "===== pip proxy deps ====="
pip list 2>/dev/null | grep -iE 'fastapi|uvicorn|httpx|msgspec'
echo "===== repo candidates ====="
ls -d /home/lizhongyang/* /workspace/* /vllm* 2>/dev/null
find / -maxdepth 3 -name "load_balance_proxy_layerwise_server_example.py" 2>/dev/null | head -3
