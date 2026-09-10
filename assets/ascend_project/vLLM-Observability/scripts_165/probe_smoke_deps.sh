#!/bin/bash
echo "=== small models on host ==="
for d in /root/models /data/models /home/hucong/models /vllm-workspace/models /root/.cache/modelscope /workspace/models; do
  [ -d "$d" ] && echo "--- $d ---" && ls "$d" 2>/dev/null | head -20
done
echo "=== huggingface cache dirs ==="
find /root/.cache/huggingface -maxdepth 2 -type d -name "models--*" 2>/dev/null | head -20
echo "=== mooncake binaries ==="
which mooncake_master mooncakectl 2>/dev/null || echo "mooncake_master not in PATH"
ls /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/ 2>/dev/null | grep -i mooncake || echo "no mooncake in ascend-toolkit"
echo "=== memcache_hybrid ==="
python3 -c "import memcache_hybrid; print('memcache_hybrid OK', memcache_hybrid.__file__)" 2>&1 | tail -1
echo "=== free NPUs (full list) ==="
npu-smi info 2>/dev/null | grep -E "^\| [0-9]+" | head -10
echo "=== processes using npu ==="
ps -ef 2>/dev/null | grep -E "vllm serve|vllm.entrypoints" | grep -v grep | head -5 || echo "no vllm server running"
