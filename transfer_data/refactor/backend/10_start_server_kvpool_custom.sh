#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="$(require_run_dir "${1:-}")"
MODEL="${MODEL_PATH:-/mnt/weight/Qwen3-0.6B}"
PORT="${PORT:-8100}"
DEVICE="${ASCEND_RT_VISIBLE_DEVICES:-0}"
LOG="${RUN_DIR}/server_kvpool_custom.log"
MOONCAKE_CONFIG="${MOONCAKE_CONFIG_PATH:-${RUN_DIR}/mooncake.json}"
MASTER_ADDR="${MOONCAKE_MASTER_ADDRESS:-127.0.0.1:50088}"
export KV_CONNECTOR="${KV_CONNECTOR:-AscendStoreConnector}"
export KV_ROLE="${KV_ROLE:-kv_both}"
export KV_LOAD_FAILURE_POLICY="${KV_LOAD_FAILURE_POLICY:-recompute}"
export KV_BACKEND="${KV_BACKEND:-mooncake}"
export LOOKUP_RPC_PORT="${LOOKUP_RPC_PORT:-1}"
export LOAD_ASYNC="${LOAD_ASYNC:-}"
export CONSUMER_IS_TO_PUT="${CONSUMER_IS_TO_PUT:-}"
export CONSUMER_IS_TO_LOAD="${CONSUMER_IS_TO_LOAD:-}"
export USE_LAYERWISE="${USE_LAYERWISE:-}"
export SAVE_DECODE_CACHE="${SAVE_DECODE_CACHE:-}"
export PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-}"
export DECODE_TP_SIZE="${DECODE_TP_SIZE:-}"
export EXTRA_CONFIG_JSON="${EXTRA_CONFIG_JSON:-}"

if [[ ! -f "${MOONCAKE_CONFIG}" ]]; then
  cat > "${MOONCAKE_CONFIG}" <<JSON
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "${MASTER_ADDR}",
  "global_segment_size": "${MOONCAKE_GLOBAL_SEGMENT_SIZE:-1GB}",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true
}
JSON
fi

export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"
export ASCEND_ENABLE_USE_FABRIC_MEM="${ASCEND_ENABLE_USE_FABRIC_MEM:-1}"
export HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT:-17}"
export ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-10000}"
export ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-10000}"
export PYTHONPATH="${PYTHONPATH:-}:/vllm-workspace/vllm"
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:${LD_LIBRARY_PATH:-}"

KV_TRANSFER_CONFIG="$(${PYTHON:-python3} - <<'PY'
import json
import os

extra = {
    "lookup_rpc_port": os.environ["LOOKUP_RPC_PORT"],
    "backend": os.environ["KV_BACKEND"],
}

bool_fields = {
    "LOAD_ASYNC": "load_async",
    "CONSUMER_IS_TO_PUT": "consumer_is_to_put",
    "CONSUMER_IS_TO_LOAD": "consumer_is_to_load",
    "USE_LAYERWISE": "use_layerwise",
    "SAVE_DECODE_CACHE": "save_decode_cache",
}
for env_name, key in bool_fields.items():
    value = os.environ.get(env_name, "")
    if value:
        extra[key] = value.lower() in {"1", "true", "yes", "on"}

int_fields = {
    "PREFILL_TP_SIZE": "prefill_tp_size",
    "DECODE_TP_SIZE": "decode_tp_size",
}
for env_name, key in int_fields.items():
    value = os.environ.get(env_name, "")
    if value:
        extra[key] = int(value)

extra_json = os.environ.get("EXTRA_CONFIG_JSON", "")
if extra_json:
    extra.update(json.loads(extra_json))

config = {
    "kv_connector": os.environ["KV_CONNECTOR"],
    "kv_role": os.environ["KV_ROLE"],
    "kv_load_failure_policy": os.environ["KV_LOAD_FAILURE_POLICY"],
    "kv_connector_extra_config": extra,
}
print(json.dumps(config, separators=(",", ":")))
PY
)"

cd /vllm-workspace/vllm-ascend

python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --trust-remote-code \
  --enforce-eager \
  --no-enable-prefix-caching \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --data-parallel-size "${DP_SIZE:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --block-size "${BLOCK_SIZE:-128}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  > "${LOG}" 2>&1 &

PID=$!
write_pid "${RUN_DIR}" server_kvpool_custom "${PID}"
printf '%s\n' "${KV_TRANSFER_CONFIG}" > "${RUN_DIR}/kv_transfer_config.json"

{
  echo "MODEL=${MODEL}"
  echo "PORT=${PORT}"
  echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
  echo "MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH}"
  echo "MOONCAKE_MASTER_ADDRESS=${MASTER_ADDR}"
  echo "KV_CONNECTOR=${KV_CONNECTOR}"
  echo "KV_ROLE=${KV_ROLE}"
  echo "KV_LOAD_FAILURE_POLICY=${KV_LOAD_FAILURE_POLICY}"
  echo "KV_BACKEND=${KV_BACKEND}"
  echo "LOOKUP_RPC_PORT=${LOOKUP_RPC_PORT}"
  echo "LOAD_ASYNC=${LOAD_ASYNC}"
  echo "CONSUMER_IS_TO_PUT=${CONSUMER_IS_TO_PUT}"
  echo "CONSUMER_IS_TO_LOAD=${CONSUMER_IS_TO_LOAD}"
  echo "USE_LAYERWISE=${USE_LAYERWISE}"
  echo "SAVE_DECODE_CACHE=${SAVE_DECODE_CACHE}"
  echo "PREFILL_TP_SIZE=${PREFILL_TP_SIZE}"
  echo "DECODE_TP_SIZE=${DECODE_TP_SIZE}"
  echo "EXTRA_CONFIG_JSON=${EXTRA_CONFIG_JSON}"
  echo "KV_TRANSFER_CONFIG=${KV_TRANSFER_CONFIG}"
} > "${RUN_DIR}/kvpool_custom_env.txt"

echo "RUN_DIR=${RUN_DIR}"
echo "SERVER_PID=${PID}"
echo "SERVER_LOG=${LOG}"
echo "MODEL=${MODEL}"
echo "PORT=${PORT}"
echo "MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH}"
echo "KV_ROLE=${KV_ROLE}"
echo "KV_BACKEND=${KV_BACKEND}"
