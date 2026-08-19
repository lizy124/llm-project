#!/usr/bin/env bash
set -Eeuo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$SCENARIO_DIR/results}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/$(date +%Y%m%d_%H%M%S)}"
LATEST_LINK="$RESULT_ROOT/latest"

MODEL_PATH="${MODEL_PATH:-/mnt/share/modelscope/hub/models/UploadWeight}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4-flash-kvpool}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8004}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
READINESS_TIMEOUT_S="${READINESS_TIMEOUT_S:-3600}"
HARDWARE_SERIES="${HARDWARE_SERIES:-A3}"

KV_BACKEND="memcache"
MMC_LOCAL_CONFIG_PATH="${MMC_LOCAL_CONFIG_PATH:-/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-local.conf}"

ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ATB_ENV="${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"

LOG_FILE="$RESULT_DIR/server.log"
PID_FILE="$RESULT_DIR/server.pid"
ENV_FILE="$RESULT_DIR/env.txt"
CMD_FILE="$RESULT_DIR/command.sh"
STATUS_FILE="$RESULT_DIR/status.txt"

mkdir -p "$RESULT_DIR"
rm -f "$LATEST_LINK"
ln -s "$RESULT_DIR" "$LATEST_LINK"

fail_blocked() {
  echo "BLOCKED: $*" | tee -a "$STATUS_FILE" >&2
  exit 2
}

fail_failed() {
  echo "FAIL: $*" | tee -a "$STATUS_FILE" >&2
  exit 1
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || fail_blocked "$label not found: $path"
}

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || fail_blocked "$label not found: $path"
}

port_free() {
  python - "$SERVER_PORT" <<'PY'
import socket
import sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

count_devices() {
  python - "$ASCEND_RT_VISIBLE_DEVICES" <<'PY'
import sys
items = [x.strip() for x in sys.argv[1].split(',') if x.strip()]
print(len(items))
PY
}

require_dir "$MODEL_PATH" "model directory"
require_file "$MODEL_PATH/config.json" "model config"
require_file "$ASCEND_TOOLKIT_ENV" "Ascend toolkit env"
require_file "$ATB_ENV" "ATB env"
require_file "$MMC_LOCAL_CONFIG_PATH" "MMC local config"
command -v python >/dev/null 2>&1 || fail_blocked "python is not available"
command -v npu-smi >/dev/null 2>&1 || fail_blocked "npu-smi is not available"
port_free || fail_blocked "server port is already in use: $SERVER_PORT"

DEVICE_COUNT="$(count_devices)"
if [[ "$DEVICE_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]]; then
  fail_blocked "ASCEND_RT_VISIBLE_DEVICES has $DEVICE_COUNT devices but TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
fi

set +u
source "$ASCEND_TOOLKIT_ENV"
source "$ATB_ENV"
set -u

export ASCEND_RT_VISIBLE_DEVICES
export MMC_LOCAL_CONFIG_PATH
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$READINESS_TIMEOUT_S}"
export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"
export ASCEND_ENABLE_USE_FABRIC_MEM="${ASCEND_ENABLE_USE_FABRIC_MEM:-1}"

KV_CONFIG='{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "lookup_rpc_port": "0",
    "use_layerwise": true
  }
}'

CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "$MODEL_PATH"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --enforce-eager
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --block-size 128
  --enable-expert-parallel
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --no-enable-prefix-caching
  --quantization ascend
  --kv-transfer-config "$KV_CONFIG"
)

{
  echo "date=$(date -Is)"
  echo "scenario=$SCENARIO_DIR"
  echo "repo=/vllm-workspace/vllm-ascend"
  git -C /vllm-workspace/vllm-ascend status --short --branch || true
  git -C /vllm-workspace/vllm-ascend rev-parse HEAD || true
  python - <<'PY'
import importlib.metadata as md
for name in ("vllm", "vllm-ascend"):
    try:
        dist = md.distribution(name)
        print(f"{name}={dist.version} location={dist.locate_file('')}")
    except Exception as exc:
        print(f"{name}=unavailable error={exc}")
PY
  echo "MODEL_PATH=$MODEL_PATH"
  echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
  echo "SERVER_HOST=$SERVER_HOST"
  echo "SERVER_PORT=$SERVER_PORT"
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
  echo "DATA_PARALLEL_SIZE=$DATA_PARALLEL_SIZE"
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
  echo "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
  echo "MAX_NUM_SEQS=$MAX_NUM_SEQS"
  echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
  echo "KV_BACKEND=$KV_BACKEND"
  echo "USE_LAYERWISE=true"
  echo "MMC_LOCAL_CONFIG_PATH=$MMC_LOCAL_CONFIG_PATH"
  echo "HARDWARE_SERIES=$HARDWARE_SERIES"
  npu-smi info || true
} > "$ENV_FILE" 2>&1

printf '%q ' "${CMD[@]}" > "$CMD_FILE"
printf '\n' >> "$CMD_FILE"

rm -f "$LOG_FILE" "$PID_FILE"
echo "RUNNING: starting service" | tee "$STATUS_FILE"
"${CMD[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

BASE_URL="http://127.0.0.1:$SERVER_PORT"
DEADLINE=$((SECONDS + READINESS_TIMEOUT_S))
while (( SECONDS < DEADLINE )); do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    tail -200 "$LOG_FILE" >&2 || true
    if grep -Eiq 'Configuration loading failed|Store initialization failed|memcache_backend.py.*assert res == 0|Initialize mooncake failed|metadata_server|master_server|Connection refused|meta_service|config_store|MetaService|ConfigStore' "$LOG_FILE"; then
      fail_blocked "server exited before readiness because KV backend service/configuration was unavailable"
    fi
    fail_failed "server process exited before readiness"
  fi
  if python - "$BASE_URL" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request
url = sys.argv[1] + "/v1/models"
with urllib.request.urlopen(url, timeout=5) as resp:
    data = json.loads(resp.read().decode())
if "data" not in data:
    raise SystemExit(1)
PY
  then
    echo "PASS: service ready at $BASE_URL" | tee -a "$STATUS_FILE"
    echo "result_dir=$RESULT_DIR"
    exit 0
  fi
  sleep 10
done

tail -200 "$LOG_FILE" >&2 || true
fail_failed "service did not become ready within ${READINESS_TIMEOUT_S}s"
