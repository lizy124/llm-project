#!/bin/bash
# poll_ready.sh — 轮询服务就绪
set -u
PORT=${1:-8004}
for i in $(seq 1 120); do
  if curl -s --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200; then
    echo "READY after ${i}0s: 8004 http 200"; exit 0
  fi
  sleep 10
done
echo "TIMEOUT waiting 8004"; exit 1