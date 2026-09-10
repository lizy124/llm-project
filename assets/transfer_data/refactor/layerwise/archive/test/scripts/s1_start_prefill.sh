#!/usr/bin/env bash
# s1_start_prefill.sh — 场景1 P 侧:DSV2-Lite TP=4,MultiConnector(MooncakeLayerwise+AscendStore/memcache layerwise)
set -Eeuo pipefail

BASE=/home/lizhongyang/map_165
RESULT_DIR="$BASE/run/s1_pd_multiconn"
MODEL_PATH="${MODEL_PATH:-/mnt/weight/DeepSeek-V2-Lite-Chat}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv2-lite-pd}"
SERVER_PORT="${SERVER_PORT:-8100}"
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=32768
MAX_NUM_BATCHED_TOKENS=16384
MAX_NUM_SEQS=20
READINESS_TIMEOUT_S=1800
MMC_LOCAL_CONFIG_PATH=/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-local.conf

LOG_FILE="$RESULT_DIR/prefill.log"
PID_FILE="$RESULT_DIR/prefill.pid"
STATUS_FILE="$RESULT_DIR/prefill_status.txt"

mkdir -p "$RESULT_DIR"

fail() { echo "FAIL: $*" | tee -a "$STATUS_FILE" >&2; exit 1; }
blocked() { echo "BLOCKED: $*" | tee -a "$STATUS_FILE" >&2; exit 2; }

[ -f "$MODEL_PATH/config.json" ] || blocked "model not found: $MODEL_PATH"
[ -f "$MMC_LOCAL_CONFIG_PATH" ] || blocked "mmc-local.conf not found"
[ -f "$BASE/test/mooncake.json" ] || blocked "mooncake.json not found"

python3 - "$SERVER_PORT" <<'PY' || blocked "port $SERVER_PORT in use"
import socket, sys
s = socket.socket()
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
finally:
    s.close()
PY

export ASCEND_RT_VISIBLE_DEVICES
export MMC_LOCAL_CONFIG_PATH
export MOONCAKE_CONFIG_PATH="$BASE/test/mooncake.json"
export MOONCAKE_MASTER=127.0.0.1:50088
export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=$READINESS_TIMEOUT_S
export VLLM_LOGGING_LEVEL=DEBUG
export ACL_OP_INIT_MODE=1
export ASCEND_ENABLE_USE_FABRIC_MEM=1

# 回归核心路径:AscendMultiConnector.__init__ → _configure_layerwise_reuse_completion
# → AscendStoreConnector.set_external_slot_release_waiter(#14465 在此 AttributeError)
KV_CONFIG='{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "engine_id": "0",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeLayerwiseConnector",
        "kv_role": "kv_producer",
        "kv_port": 30000,
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {"dp_size": 1, "tp_size": 4},
          "decode": {"dp_size": 1, "tp_size": 4}
        }
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
          "backend": "memcache",
          "lookup_rpc_port": "0",
          "use_layerwise": true
        }
      }
    ]
  }
}'

{
  echo "date=$(date -Is)"
  echo "scenario=s1_pd_multiconn role=prefill model=$MODEL_PATH tp=$TENSOR_PARALLEL_SIZE chips=$ASCEND_RT_VISIBLE_DEVICES"
  echo "kv_config=$KV_CONFIG"
} > "$RESULT_DIR/prefill_env.txt" 2>&1

rm -f "$LOG_FILE" "$PID_FILE"
echo "RUNNING: starting prefill" | tee "$STATUS_FILE"
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
    fail "prefill exited before readiness"
  fi
  if curl -s --max-time 5 "http://127.0.0.1:$SERVER_PORT/v1/models" | grep -q '"data"'; then
    echo "PASS: prefill ready at http://127.0.0.1:$SERVER_PORT" | tee -a "$STATUS_FILE"
    exit 0
  fi
  sleep 10
done
tail -100 "$LOG_FILE" >&2 || true
fail "prefill not ready within ${READINESS_TIMEOUT_S}s"
