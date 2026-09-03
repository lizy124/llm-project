#!/bin/bash
# clean_npu.sh — 清理容器内残留 vllm 进程并查 NPU 占用
ps aux | grep -E "vllm|from multiprocessing|EngineCore|Worker_TP|proxy_layerwise" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 5
npu-smi info | grep -E "MiB|No processes" | head -20
echo "---"
ps aux | grep -E "vllm|multiprocessing" | grep -v grep || echo NO_VLLM_LEFT
