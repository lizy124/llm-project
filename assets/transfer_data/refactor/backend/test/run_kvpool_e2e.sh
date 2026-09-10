#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test/runs/20260811_082723_env"
MODEL="/mnt/weight/Qwen3-0.6B"
PORT=8100
TEST_DIR="/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test"

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
  echo "ERROR: Server not ready after 60 attempts"
  echo "=== Server log tail ==="
  tail -50 "${RUN_DIR}/server_kvpool_custom.log"
  exit 1
fi

echo "=== Send requests ==="
cd "$TEST_DIR"
python3 05_send_requests.py --run-dir "$RUN_DIR" --case kvpool_both --model "$MODEL" --port "$PORT" --repeat 2

echo "=== Send stream requests ==="
python3 09_send_stream_requests.py --run-dir "$RUN_DIR" --case stream_kvpool --model "$MODEL" --port "$PORT" --repeat 4

echo "=== Grep logs ==="
bash 06_grep_logs.sh "$RUN_DIR"

echo "=== Stop run ==="
bash 07_stop_run.sh "$RUN_DIR"

echo "KVPOOL_E2E_DONE"
