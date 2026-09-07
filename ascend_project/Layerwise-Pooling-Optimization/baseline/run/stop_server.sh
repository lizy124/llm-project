#!/bin/bash
# stop_server.sh — 停 vllm 服务（playbook/run_dir/common-prerequisites.md §5 模板）
# 模式: 显式 PID → 等待 → kill -9 → 清 multiprocessing 残余（残留 worker 占 HBM）
set -uo pipefail

RUN=/home/lizhongyang/lw_baseline/run
PID=$(cat $RUN/server.pid 2>/dev/null || echo "")

if [ -z "$PID" ]; then
  echo "no server.pid, nothing to stop"
else
  if kill -0 "$PID" 2>/dev/null; then
    echo "stopping pid $PID ..."
    kill "$PID"
    for i in $(seq 1 10); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 2
    done
    kill -9 "$PID" 2>/dev/null || true
  else
    echo "pid $PID already dead"
  fi
  rm -f "$RUN/server.pid"
fi

# vllm 崩溃残留 worker 占 NPU HBM,必清
pkill -9 -f "from multiprocessing" 2>/dev/null || true
sleep 2
echo "---- residual NPU processes ----"
npu-smi info | awk '/0000:/{getline; print "HBM: " $9}' | head -8
echo "stopped"
