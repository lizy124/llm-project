#!/bin/bash
# stop_s1.sh — 停 S1 全部:prefill/decode/proxy(容器内执行)
D=/home/lizhongyang/map_165/run/s1_pd_multiconn
for name in proxy decode prefill; do
  if [ -f "$D/$name.pid" ]; then
    PID=$(cat "$D/$name.pid")
    kill "$PID" 2>/dev/null
    for i in 1 2 3 4 5; do
      kill -0 "$PID" 2>/dev/null || break
      sleep 2
    done
    kill -9 "$PID" 2>/dev/null
    echo "stopped $name pid=$PID"
  fi
done
pkill -9 -f "from multiprocessing" 2>/dev/null
sleep 3
ps aux | grep -E "vllm.entrypoints|proxy_layerwise" | grep -v grep || echo ALL_STOPPED
