#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test/runs/20260811_090221_env"
TEST_DIR="/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test"
MODEL="/mnt/weight/Qwen3-0.6B"
PORT=8100

echo "=== Start baseline server ==="
cd "$TEST_DIR"
MODEL_PATH="$MODEL" PORT="$PORT" bash 03_start_server_baseline.sh "$RUN_DIR"

echo "=== Waiting for server ==="
ready=0
for i in $(seq 1 60); do
  if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "Server ready after ${i} attempts"
    ready=1
    break
  fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "ERROR: Server not ready"
  tail -50 "${RUN_DIR}/server_baseline.log"
  exit 1
fi

echo "=== Send requests ==="
python3 05_send_requests.py --run-dir "$RUN_DIR" --case baseline --model "$MODEL" --port "$PORT" --repeat 2

echo "=== Grep logs ==="
bash 06_grep_logs.sh "$RUN_DIR"

echo "=== Stop server ==="
bash 07_stop_run.sh "$RUN_DIR"

echo "BASELINE_DONE"
