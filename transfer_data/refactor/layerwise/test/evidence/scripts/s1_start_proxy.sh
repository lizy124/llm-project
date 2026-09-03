#!/usr/bin/env bash
# s1_start_proxy.sh — 场景1 layerwise proxy(/v1/metaserver;--host 禁 0.0.0.0)
set -Eeuo pipefail

BASE=/home/lizhongyang/map_165
RESULT_DIR="$BASE/run/s1_pd_multiconn"
PROXY_PORT="${PROXY_PORT:-9000}"
PROXY_SCRIPT=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py

LOG_FILE="$RESULT_DIR/proxy.log"
PID_FILE="$RESULT_DIR/proxy.pid"
STATUS_FILE="$RESULT_DIR/proxy_status.txt"

mkdir -p "$RESULT_DIR"

fail() { echo "FAIL: $*" | tee -a "$STATUS_FILE" >&2; exit 1; }
blocked() { echo "BLOCKED: $*" | tee -a "$STATUS_FILE" >&2; exit 2; }

[ -f "$PROXY_SCRIPT" ] || blocked "proxy script not found: $PROXY_SCRIPT"
curl -s --max-time 5 "http://127.0.0.1:8100/v1/models" | grep -q '"data"' || blocked "prefill not ready"
curl -s --max-time 5 "http://127.0.0.1:8200/v1/models" | grep -q '"data"' || blocked "decode not ready"

rm -f "$LOG_FILE" "$PID_FILE"
echo "RUNNING: starting proxy" | tee "$STATUS_FILE"
nohup python3 "$PROXY_SCRIPT" \
  --host 127.0.0.1 \
  --port "$PROXY_PORT" \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 \
  --decoder-ports 8200 \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

for i in $(seq 1 15); do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    tail -50 "$LOG_FILE" >&2 || true
    fail "proxy exited during startup"
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$PROXY_PORT/healthcheck" | grep -q .; then
    curl -s --max-time 5 "http://127.0.0.1:$PROXY_PORT/healthcheck" | tee -a "$STATUS_FILE"
    echo ""
    echo "PASS: proxy ready at http://127.0.0.1:$PROXY_PORT" | tee -a "$STATUS_FILE"
    exit 0
  fi
  sleep 2
done
tail -50 "$LOG_FILE" >&2 || true
fail "proxy not ready"
