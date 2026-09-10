#!/bin/bash
# start_server_161.sh — 161 鉴别实验服务（与 165 基线同配置: w4a8-mtp, TP8, prefix-caching ON）
set -euo pipefail

BASE=/home/lizhongyang/lw_verify
RUN=$BASE/run
LOG=$RUN/server.log

MODEL=/mnt/weight/DeepSeek-V4-Flash-w4a8-mtp
PORT=8004
mkdir -p "$RUN"

if [ -f "$RUN/server.pid" ] && kill -0 "$(cat $RUN/server.pid)" 2>/dev/null; then
  echo "FAIL: server already running (pid $(cat $RUN/server.pid))"; exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

python3 -m vllm.entrypoints.openai.api_server \
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

echo $! > "$RUN/server.pid"
echo "server starting: pid $(cat $RUN/server.pid), log $LOG"
