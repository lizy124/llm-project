#!/bin/bash
# start_server_dp2_161.sh — 16 芯 DP2×TP8 基线服务 (每副本 8 芯)
# 前提: 现有 lw_verify_161 容器已挂载全卡; ASCEND_RT_VISIBLE_DEVICES 控制每副本芯
BASE=/home/lizhongyang/lw_verify
RUN=$BASE/run
LOG=$RUN/server_dp2.log
MODEL=/mnt/weight/DeepSeek-V4-Flash-w4a8-mtp
PORT=8005
mkdir -p "$RUN"

if [ -f "$RUN/server_dp2.pid" ] && kill -0 "$(cat $RUN/server_dp2.pid)" 2>/dev/null; then
  echo "FAIL: dp2 server already running (pid $(cat $RUN/server_dp2.pid))"; exit 1
fi

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
  --data-parallel-size 2 \
  --enable-prefix-caching \
  --max-num-seqs 128 \
  --max-model-len 163840 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.9 \
  --block-size 128 \
  > "$LOG" 2>&1 &

echo $! > "$RUN/server_dp2.pid"
echo "dp2 server starting: pid $(cat $RUN/server_dp2.pid), log $LOG"