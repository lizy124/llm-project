#!/usr/bin/env bash
set -Eeuo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$SCENARIO_DIR/results}"
RESULT_DIR="${RESULT_DIR:-$RESULT_ROOT/$(date +%Y%m%d_%H%M%S)}"
LATEST_LINK="$RESULT_ROOT/latest"

MODEL_PATH="${MODEL_PATH:-/mnt/share/modelscope/hub/models/Qwen/Qwen3.5-397B-A17B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-397b-kvpool}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8005}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-16}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
READINESS_TIMEOUT_S="${READINESS_TIMEOUT_S:-3600}"
HARDWARE_SERIES="${HARDWARE_SERIES:-A3}"
KV_BACKEND="${KV_BACKEND:-mooncake}"
KV_LOAD_FAILURE_POLICY="${KV_LOAD_FAILURE_POLICY:-}"
DTYPE="${DTYPE:-bfloat16}"
SEED="${SEED:-1024}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
ENABLE_HYBRID_KV_CACHE_MANAGER="${ENABLE_HYBRID_KV_CACHE_MANAGER:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
QUANTIZATION="${QUANTIZATION:-}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 1, \"enforce_eager\": true}}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-{\"enable_cpu_binding\": true, \"enable_fused_mc2\": 1}}"
OFFICIAL_BF16_REQUIRED_DEVICES="${OFFICIAL_BF16_REQUIRED_DEVICES:-32}"
OFFICIAL_RESOURCE_CHECK="${OFFICIAL_RESOURCE_CHECK:-warn}"

MOONCAKE_MASTER="${MOONCAKE_MASTER:-127.0.0.1:50088}"
MOONCAKE_TE_META_DATA_SERVER="${MOONCAKE_TE_META_DATA_SERVER:-P2PHANDSHAKE}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-ascend}"
MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-}"
MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-1GB}"
MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-1GB}"

ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ATB_ENV="${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"

MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-$RESULT_DIR/mooncake.json}"
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

write_mooncake_config() {
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
}

build_kv_config() {
  python - "$KV_BACKEND" "$KV_LOAD_FAILURE_POLICY" <<'PY'
import json
import sys
backend, policy = sys.argv[1:]
config = {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": backend,
        "lookup_rpc_port": "1" if backend == "mooncake" else "0",
    },
}
if policy:
    config["kv_load_failure_policy"] = policy
print(json.dumps(config, indent=2))
PY
}

require_dir "$MODEL_PATH" "model directory"
require_file "$MODEL_PATH/config.json" "model config"
require_file "$ASCEND_TOOLKIT_ENV" "Ascend toolkit env"
require_file "$ATB_ENV" "ATB env"
if [[ "$KV_BACKEND" != "mooncake" && "$KV_BACKEND" != "memcache" ]]; then
  fail_blocked "invalid KV_BACKEND=$KV_BACKEND, expected mooncake or memcache"
fi
if [[ -n "$KV_LOAD_FAILURE_POLICY" && "$KV_LOAD_FAILURE_POLICY" != "fail" && "$KV_LOAD_FAILURE_POLICY" != "recompute" ]]; then
  fail_blocked "invalid KV_LOAD_FAILURE_POLICY=$KV_LOAD_FAILURE_POLICY, expected empty, fail, or recompute"
fi
command -v python >/dev/null 2>&1 || fail_blocked "python is not available"
command -v npu-smi >/dev/null 2>&1 || fail_blocked "npu-smi is not available"
port_free || fail_blocked "server port is already in use: $SERVER_PORT"

DEVICE_COUNT="$(count_devices)"
if [[ "$DEVICE_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]]; then
  fail_blocked "ASCEND_RT_VISIBLE_DEVICES has $DEVICE_COUNT devices but TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
fi
if [[ -z "$QUANTIZATION" && "$DTYPE" == "bfloat16" && "$DEVICE_COUNT" -lt "$OFFICIAL_BF16_REQUIRED_DEVICES" ]]; then
  msg="official Qwen3.5-397B BF16 doc requires about $OFFICIAL_BF16_REQUIRED_DEVICES A3/A2 cards, visible devices=$DEVICE_COUNT"
  if [[ "$OFFICIAL_RESOURCE_CHECK" == "strict" ]]; then
    fail_blocked "$msg"
  fi
  echo "WARN: $msg; continuing because OFFICIAL_RESOURCE_CHECK=$OFFICIAL_RESOURCE_CHECK" | tee -a "$STATUS_FILE"
fi

if [[ "$KV_BACKEND" == "mooncake" && ! -f "$MOONCAKE_CONFIG_PATH" ]]; then
  write_mooncake_config
fi

set +u
source "$ASCEND_TOOLKIT_ENV"
source "$ATB_ENV"
set -u

export ASCEND_RT_VISIBLE_DEVICES
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$READINESS_TIMEOUT_S}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export VLLM_TORCH_PROFILER_WITH_STACK="${VLLM_TORCH_PROFILER_WITH_STACK:-0}"

if [[ "$HARDWARE_SERIES" == "A3" ]]; then
  export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"
  export ASCEND_ENABLE_USE_FABRIC_MEM="${ASCEND_ENABLE_USE_FABRIC_MEM:-1}"
elif [[ "$HARDWARE_SERIES" == "A2" ]]; then
  export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
else
  fail_blocked "invalid HARDWARE_SERIES=$HARDWARE_SERIES, expected A2 or A3"
fi

if [[ "$KV_BACKEND" == "mooncake" ]]; then
  export ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-10000}"
  export ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-10000}"
  export HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT:-17}"
fi

KV_CONFIG="$(build_kv_config)"

CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "$MODEL_PATH"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --seed "$SEED"
  --dtype "$DTYPE"
  --data-parallel-size "$DATA_PARALLEL_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-transfer-config "$KV_CONFIG"
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  CMD+=(--enforce-eager)
fi
if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  CMD+=(--enable-expert-parallel)
fi
if [[ "$ENABLE_HYBRID_KV_CACHE_MANAGER" == "1" ]]; then
  CMD+=(--no-disable-hybrid-kv-cache-manager)
fi
if [[ -n "$QUANTIZATION" ]]; then
  CMD+=(--quantization "$QUANTIZATION")
fi
if [[ -n "$SPECULATIVE_CONFIG" ]]; then
  CMD+=(--speculative-config "$SPECULATIVE_CONFIG")
fi
if [[ -n "$COMPILATION_CONFIG" ]]; then
  CMD+=(--compilation-config "$COMPILATION_CONFIG")
fi
if [[ -n "$ADDITIONAL_CONFIG" ]]; then
  CMD+=(--additional-config "$ADDITIONAL_CONFIG")
fi

{
  echo "date=$(date -Is)"
  echo "scenario=$SCENARIO_DIR"
  echo "repo=/vllm-workspace/vllm-ascend"
  git -C /vllm-workspace/vllm-ascend status --short --branch || true
  git -C /vllm-workspace/vllm-ascend rev-parse HEAD || true
  python - <<'PY'
import importlib.metadata as md
for name in ("vllm", "vllm-ascend", "mooncake-transfer-engine-npu"):
    try:
        dist = md.distribution(name)
        print(f"{name}={dist.version} location={dist.locate_file('')}")
    except Exception as exc:
        print(f"{name}=unavailable error={exc}")
PY
  python - "$MODEL_PATH/config.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
text = cfg.get("text_config") or {}
print(f"model_architectures={cfg.get('architectures')}")
print(f"model_type={cfg.get('model_type')}")
print(f"text_dtype={text.get('dtype')}")
print(f"text_layers={text.get('num_hidden_layers')}")
print(f"text_max_position_embeddings={text.get('max_position_embeddings')}")
print(f"text_num_experts={text.get('num_experts')}")
print(f"text_num_experts_per_tok={text.get('num_experts_per_tok')}")
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
  echo "DTYPE=$DTYPE"
  echo "QUANTIZATION=$QUANTIZATION"
  echo "KV_BACKEND=$KV_BACKEND"
  echo "KV_LOAD_FAILURE_POLICY=$KV_LOAD_FAILURE_POLICY"
  echo "MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH"
  echo "MOONCAKE_MASTER=$MOONCAKE_MASTER"
  echo "MOONCAKE_TE_META_DATA_SERVER=$MOONCAKE_TE_META_DATA_SERVER"
  echo "MOONCAKE_PROTOCOL=$MOONCAKE_PROTOCOL"
  echo "MOONCAKE_DEVICE=$MOONCAKE_DEVICE"
  echo "MOONCAKE_GLOBAL_SEGMENT_SIZE=$MOONCAKE_GLOBAL_SEGMENT_SIZE"
  echo "MOONCAKE_LOCAL_BUFFER_SIZE=$MOONCAKE_LOCAL_BUFFER_SIZE"
  echo "HARDWARE_SERIES=$HARDWARE_SERIES"
  echo "ENABLE_EXPERT_PARALLEL=$ENABLE_EXPERT_PARALLEL"
  echo "ENABLE_HYBRID_KV_CACHE_MANAGER=$ENABLE_HYBRID_KV_CACHE_MANAGER"
  echo "ENFORCE_EAGER=$ENFORCE_EAGER"
  echo "SPECULATIVE_CONFIG=$SPECULATIVE_CONFIG"
  echo "COMPILATION_CONFIG=$COMPILATION_CONFIG"
  echo "ADDITIONAL_CONFIG=$ADDITIONAL_CONFIG"
  echo "OFFICIAL_BF16_REQUIRED_DEVICES=$OFFICIAL_BF16_REQUIRED_DEVICES"
  echo "OFFICIAL_RESOURCE_CHECK=$OFFICIAL_RESOURCE_CHECK"
  npu-smi info || true
} > "$ENV_FILE" 2>&1

printf '%q ' "${CMD[@]}" > "$CMD_FILE"
printf '\n' >> "$CMD_FILE"

rm -f "$LOG_FILE" "$PID_FILE"
echo "RUNNING: starting Qwen3.5-397B-A17B PDMix non-layerwise KV Pool service" | tee -a "$STATUS_FILE"
"${CMD[@]}" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

BASE_URL="http://127.0.0.1:$SERVER_PORT"
DEADLINE=$((SECONDS + READINESS_TIMEOUT_S))
while (( SECONDS < DEADLINE )); do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    tail -200 "$LOG_FILE" >&2 || true
    if grep -Eiq 'Configuration loading failed|Store initialization failed|Initialize mooncake failed|metadata_server|master_server|Connection refused|Mooncake|mooncake.*(failed|error)|OutOfMemoryError|out of memory|OOM|HBM|Insufficient|Memory_Allocation_Failure|ACL.*(memory|alloc)' "$LOG_FILE"; then
      fail_blocked "server exited before readiness because KV backend or 397B BF16 resource requirements were unavailable"
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
if grep -Eiq 'OutOfMemoryError|out of memory|OOM|HBM|Insufficient|Memory_Allocation_Failure|ACL.*(memory|alloc)' "$LOG_FILE"; then
  fail_blocked "service did not become ready because available NPU memory/resources are insufficient for Qwen3.5-397B BF16"
fi
fail_failed "service did not become ready within ${READINESS_TIMEOUT_S}s"
