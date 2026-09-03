#!/usr/bin/env bash
# s3_start.sh — 场景3:Qwen3-32B + mooncake 非 layerwise 零回归(容器内执行)
set -Eeuo pipefail

BASE=/home/lizhongyang/map_165
RESULT_DIR="$BASE/run/s3_mooncake_non_layerwise"
MODEL_PATH="${MODEL_PATH:-/mnt/weight/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b-mooncake}"
SERVER_PORT="${SERVER_PORT:-8006}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=32768
MAX_NUM_BATCHED_TOKENS=16384
MAX_NUM_SEQS=20
READINESS_TIMEOUT_S=1800

LOG_FILE="$RESULT_DIR/server.log"
PID_FILE="$RESULT_DIR/server.pid"
ENV_FILE="$RESULT_DIR/env.txt"
STATUS_FILE="$RESULT_DIR/status.txt"

mkdir -p "$RESULT_DIR"

fail() { echo "FAIL: $*" | tee -a "$STATUS_FILE" >&2; exit 1; }
blocked() { echo "BLOCKED: $*" | tee -a "$STATUS_FILE" >&2; exit 2; }

[ -f "$MODEL_PATH/config.json" ] || blocked "model not found: $MODEL_PATH"
[ -f "$BASE/test/mooncake.json" ] || blocked "mooncake.json not found"

python3 - "$SERVER_PORT" <<'PY' || blocked "port in use"
import socket, sys
s = socket.socket()
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
finally:
    s.close()
PY

export ASCEND_RT_VISIBLE_DEVICES
export MOONCAKE_CONFIG_PATH="$BASE/test/mooncake.json"
export MOONCAKE_MASTER=127.0.0.1:50088
export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=$READINESS_TIMEOUT_S
export ACL_OP_INIT_MODE=1
export ASCEND_ENABLE_USE_FABRIC_MEM=1

KV_CONFIG='{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "lookup_rpc_port": "0"
  }
}'

{
  echo "date=$(date -Is)"
  echo "scenario=s3_mooncake_non_layerwise model=$MODEL_PATH tp=$TENSOR_PARALLEL_SIZE chips=$ASCEND_RT_VISIBLE_DEVICES"
  git -C /vllm-workspace/vllm-ascend rev-parse HEAD
  git -C /vllm-workspace/vllm rev-parse HEAD
  pip list 2>/dev/null | grep -iE '^(vllm|vllm_ascend|torch|torch_npu|mooncake)'
} > "$ENV_FILE" 2>&1

rm -f "$LOG_FILE" "$PID_FILE"
echo "RUNNING: starting service" | tee "$STATUS_FILE"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$SERVER_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --enforce-eager \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization 0.9 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --kv-transfer-config "$KV_CONFIG" \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

DEADLINE=$((SECONDS + READINESS_TIMEOUT_S))
while (( SECONDS < DEADLINE )); do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    tail -100 "$LOG_FILE" >&2 || true
    fail "server exited before readiness"
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$SERVER_PORT/v1/models" | grep -q '"data"'; then
    echo "PASS: service ready at http://127.0.0.1:$SERVER_PORT" | tee -a "$STATUS_FILE"
    exit 0
  fi
  sleep 10
done
tail -100 "$LOG_FILE" >&2 || true
fail "service not ready within ${READINESS_TIMEOUT_S}s"
