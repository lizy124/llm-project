#!/bin/bash
# start_server.sh — HBM 基线组 vllm 服务拉起（DSV4 Flash, TP8, prefix-caching ON）
# 参考: playbook/run_dir 惯例（pid 文件 + 日志重定向 + 显式 PID 停服）
# 用法: bash start_server.sh   （容器内执行;重复执行前先 stop_server.sh）
set -euo pipefail

BASE=/home/lizhongyang/lw_baseline
RUN=$BASE/run
LOG=$RUN/server.log

MODEL=/mnt/weight/DeepSeek-V4-Flash-w4a8-mtp
PORT=8004
mkdir -p "$RUN"

# ---- 前置检查 ----
if [ -f "$RUN/server.pid" ] && kill -0 "$(cat $RUN/server.pid)" 2>/dev/null; then
  echo "FAIL: server already running (pid $(cat $RUN/server.pid)); run stop_server.sh first"
  exit 1
fi
npu-smi info | awk '/0000:/{getline; print "HBM check: " $9}' | head -8 || true

# ---- 环境变量（对齐 playbook/run_dir/memcache-layerwise.md §1，裁剪池化相关项）----
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
# TP=8 加载 280G 权重慢,给足就绪窗口
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# ---- 启动（基线组: 标准特性,无 kv-transfer-config, 无池化/MTP/spec）----
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
echo "poll: bash $RUN/poll_ready.sh"
