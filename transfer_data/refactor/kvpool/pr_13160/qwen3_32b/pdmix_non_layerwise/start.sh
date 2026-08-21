#!/usr/bin/env bash
set -Eeuo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$SCENARIO_DIR/results}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/$(date +%Y%m%d_%H%M%S)}"
LATEST_LINK="$RESULT_ROOT/latest"

MODEL_PATH="${MODEL_PATH:-/mnt/share/modelscope/hub/models/Qwen/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b-kvpool}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8004}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
READINESS_TIMEOUT_S="${READINESS_TIMEOUT_S:-1800}"
HARDWARE_SERIES="${HARDWARE_SERIES:-A3}"
KV_BACKEND="${KV_BACKEND:-memcache}"
MOONCAKE_MASTER="${MOONCAKE_MASTER:-127.0.0.1:50088}"
MOONCAKE_TE_META_DATA_SERVER="${MOONCAKE_TE_META_DATA_SERVER:-P2PHANDSHAKE}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-ascend}"
MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-}"
MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-1GB}"
MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-1GB}"

ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ATB_ENV="${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
MEMCACHE_ENV="${MEMCACHE_ENV:-/usr/local/memcache_hybrid/set_env.sh}"
MEMFABRIC_ENV="${MEMFABRIC_ENV:-/usr/local/memfabric_hybrid/set_env.sh}"
MMC_LOCAL_CONFIG_PATH="${MMC_LOCAL_CONFIG_PATH:-/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-local.conf}"

MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-$RESULT_DIR/mooncake.json}"
LOG_FILE="$RESULT_DIR/server.log"
PID_FILE="$RESULT_DIR/server.pid"
ENV_FILE="$RESULT_DIR/env.txt"
CMD_FILE="$RESULT_DIR/command.sh"
STATUS_FILE="$RESULT_DIR/status.txt"

mkdir -p "$RESULT_DIR"
rm -f "$LATEST_LINK"
ln -s "$RESULT_DIR" "$LATEST_LINK"

if [[ "$KV_BACKEND" == "mooncake" && ! -f "$MOONCAKE_CONFIG_PATH" ]]; then
  python - "$MOONCAKE_CONFIG_PATH" "$MOONCAKE_TE_META_DATA_SERVER" "$MOONCAKE_PROTOCOL" "$MOONCAKE_DEVICE" "$MOONCAKE_MASTER" "$MOONCAKE_GLOBAL_SEGMENT_SIZE" "$MOONCAKE_LOCAL_BUFFER_SIZE" <<'PY'
import json
import sys
path, metadata_server, protocol, device_name, master, global_segment, local_buffer = sys.argv[1:]
config = {
    "metadata_server": metadata_server,
    "protocol": protocol,
    "device_name": device_name,
    "master_server_address": master,
    "global_segment_size": global_segment,
    "local_buffer_size": local_buffer,
    "preferred_segment": False,
    "prefer_alloc_in_same_node": True,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
PY
fi

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
if [[ "$KV_BACKEND" == "memcache" ]]; then
  require_file "$MMC_LOCAL_CONFIG_PATH" "MMC local config"
fi
if [[ "$KV_BACKEND" != "memcache" && "$KV_BACKEND" != "mooncake" ]]; then
  fail_blocked "invalid KV_BACKEND=$KV_BACKEND, expected memcache or mooncake"
fi
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
if [[ -f "$MEMFABRIC_ENV" ]]; then
  source "$MEMFABRIC_ENV"
else
  echo "WARN: MemFabric env script not found, continue with Python package environment: $MEMFABRIC_ENV" | tee -a "$STATUS_FILE"
fi
if [[ -f "$MEMCACHE_ENV" ]]; then
  source "$MEMCACHE_ENV"
else
  echo "WARN: Memcache env script not found, continue with Python package environment: $MEMCACHE_ENV" | tee -a "$STATUS_FILE"
fi
set -u

export ASCEND_RT_VISIBLE_DEVICES
export MMC_LOCAL_CONFIG_PATH
export MOONCAKE_CONFIG_PATH
export MOONCAKE_MASTER
export MOONCAKE_TE_META_DATA_SERVER
export MOONCAKE_PROTOCOL
export MOONCAKE_DEVICE
export MOONCAKE_GLOBAL_SEGMENT_SIZE
export MOONCAKE_LOCAL_BUFFER_SIZE
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$READINESS_TIMEOUT_S}"

if [[ "$HARDWARE_SERIES" == "A3" ]]; then
  export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"
  export ASCEND_ENABLE_USE_FABRIC_MEM="${ASCEND_ENABLE_USE_FABRIC_MEM:-1}"
elif [[ "$HARDWARE_SERIES" == "A2" ]]; then
  :
else
  fail_blocked "invalid HARDWARE_SERIES=$HARDWARE_SERIES, expected A2 or A3"
fi

if [[ "$KV_BACKEND" == "mooncake" ]]; then
  KV_CONFIG='{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
      "backend": "mooncake",
      "lookup_rpc_port": "1"
    }
  }'
else
  KV_CONFIG='{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "backend": "memcache",
      "lookup_rpc_port": "0"
    }
  }'
fi

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
  echo "MMC_LOCAL_CONFIG_PATH=$MMC_LOCAL_CONFIG_PATH"
  echo "MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH"
  echo "MOONCAKE_MASTER=$MOONCAKE_MASTER"
  echo "MOONCAKE_TE_META_DATA_SERVER=$MOONCAKE_TE_META_DATA_SERVER"
  echo "MOONCAKE_PROTOCOL=$MOONCAKE_PROTOCOL"
  echo "MOONCAKE_DEVICE=$MOONCAKE_DEVICE"
  echo "MOONCAKE_GLOBAL_SEGMENT_SIZE=$MOONCAKE_GLOBAL_SEGMENT_SIZE"
  echo "MOONCAKE_LOCAL_BUFFER_SIZE=$MOONCAKE_LOCAL_BUFFER_SIZE"
  echo "MEMCACHE_ENV=$MEMCACHE_ENV"
  echo "MEMFABRIC_ENV=$MEMFABRIC_ENV"
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
