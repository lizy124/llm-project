#!/bin/bash
# start_hbm_baseline.sh — HBM 基线: single-TP8 + 原生 prefix-caching (对照组), 在 pool_165 内执行
# 权重=/mnt/weight/DeepSeek-V4-Flash-w4a8-mtp, 端口 8004, 栈=vllm.ascend rc2.dev1561 (=§八 基线)
set -euo pipefail
BASE=/home/lizhongyang/pool165/run
LOG=$BASE/hbm_baseline_server.log
MODEL=/mnt/weight/DeepSeek-V4-Flash-w4a8-mtp
PORT=8004
mkdir -p "$BASE"

if [ -f "$BASE/hbm_baseline_server.pid" ] && kill -0 "$(cat $BASE/hbm_baseline_server.pid)" 2>/dev/null; then
  echo "FAIL: already running pid=$(cat $BASE/hbm_baseline_server.pid)"; exit 1
fi

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=2400
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 --port $PORT \
  --served-model-name dsv4-flash \
  --trust-remote-code --enforce-eager \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --max-num-seqs 128 \
  --max-model-len 163840 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --block-size 128 \
  > "$LOG" 2>&1 &

echo $! > "$BASE/hbm_baseline_server.pid"
echo "HBM baseline starting: pid=$(cat $BASE/hbm_baseline_server.pid), log=$LOG"