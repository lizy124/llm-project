#!/bin/bash
# Environment probe inside the 165 container.
echo "=== python ==="
which python python3
python3 --version
echo "=== packages ==="
pip list 2>/dev/null | grep -iE 'vllm|torch|pytest|prometheus|msgpack|zmq|numpy|regex'
echo "=== installed vllm_ascend location ==="
python3 -c "import vllm_ascend; print(vllm_ascend.__file__)" 2>&1
echo "=== vllm version ==="
python3 -c "import vllm; print(vllm.__version__)" 2>&1
echo "=== github reachability ==="
curl -s -o /dev/null -w '%{http_code}\n' --connect-timeout 8 https://github.com
echo "=== disk space ==="
df -h /root /tmp | tail -3
