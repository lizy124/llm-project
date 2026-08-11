#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test/runs/20260811_090221_env"
TEST_DIR="/home/lizhongyang/refactor/llm-project/transfer_data/refactor/backend/test"
MODEL="/mnt/weight/Qwen3-0.6B"
PORT=8100

cd "$TEST_DIR"

echo "=== Start mooncake_master ==="
bash 02_start_mooncake_master.sh "$RUN_DIR"
sleep 3

echo "=== Start vllm server (kv_both) ==="
MODEL_PATH="$MODEL" PORT="$PORT" ASCEND_RT_VISIBLE_DEVICES=0 \
  MOONCAKE_MASTER_ADDRESS=127.0.0.1:50088 \
  KV_ROLE=kv_both KV_BACKEND=mooncake \
  bash 10_start_server_kvpool_custom.sh "$RUN_DIR"

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
  tail -80 "${RUN_DIR}/server_kvpool_custom.log"
  exit 1
fi

echo "=== Send normal requests ==="
python3 05_send_requests.py --run-dir "$RUN_DIR" --case kvpool_both --model "$MODEL" --port "$PORT" --repeat 2

echo "=== Send stream requests ==="
python3 09_send_stream_requests.py --run-dir "$RUN_DIR" --case stream_kvpool --model "$MODEL" --port "$PORT" --repeat 4

echo "=== Grep logs ==="
bash 06_grep_logs.sh "$RUN_DIR"

echo "=== Stop run ==="
bash 07_stop_run.sh "$RUN_DIR"

echo "KVPOOL_DONE"
