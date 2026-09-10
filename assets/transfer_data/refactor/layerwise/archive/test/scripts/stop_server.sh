#!/bin/bash
# stop_server.sh — 停掉指定场景目录里的 vllm 服务(容器内执行)
# 用法:bash stop_server.sh <result_dir>
DIR=$1
if [ -f "$DIR/server.pid" ]; then
  PID=$(cat "$DIR/server.pid")
  kill "$PID" 2>/dev/null
  for i in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 2
  done
  kill -9 "$PID" 2>/dev/null
  # 清理残余 worker
  pkill -9 -f "from multiprocessing" 2>/dev/null
  sleep 3
  echo "stopped pid=$PID"
else
  echo "no pid file in $DIR"
fi
