#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="$(require_run_dir "${1:-}")"
MODEL="${MODEL_PATH:-/mnt/weight/Qwen3-8B-W8A8}"
PORT="${PORT:-8200}"
DEVICE="${ASCEND_RT_VISIBLE_DEVICES:-5}"
LOG="${RUN_DIR}/server_pd_decode.log"
MOONCAKE_CONFIG="${MOONCAKE_CONFIG_PATH:-${RUN_DIR}/mooncake.json}"
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG}"
MASTER_ADDR="${MOONCAKE_MASTER_ADDRESS:-127.0.0.1:50090}"
KV_PORT="${KV_PORT:-20002}"
export KV_CONNECTOR="MultiConnector"
export KV_ROLE="kv_consumer"
export KV_LOAD_FAILURE_POLICY="recompute"
export LOOKUP_RPC_PORT="${LOOKUP_RPC_PORT:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"
export ASCEND_ENABLE_USE_FABRIC_MEM="${ASCEND_ENABLE_USE_FABRIC_MEM:-1}"
export HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT:-17}"
export ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-10000}"
export ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-10000}"
export PYTHONPATH="${PYTHONPATH:-}:/vllm-workspace/vllm"
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:${LD_LIBRARY_PATH:-}"

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

KV_TRANSFER_CONFIG="$(${PYTHON:-python3} - <<'PY'
import json
import os

config = {
    "kv_connector": os.environ["KV_CONNECTOR"],
    "kv_role": os.environ["KV_ROLE"],
    "kv_load_failure_policy": os.environ["KV_LOAD_FAILURE_POLICY"],
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": os.environ["KV_ROLE"],
                "kv_port": os.environ["KV_PORT"],
                "kv_connector_extra_config": {
                    "prefill": {"dp_size": 1, "tp_size": 1},
                    "decode": {"dp_size": 1, "tp_size": 1},
                },
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": os.environ["KV_ROLE"],
                "kv_connector_extra_config": {
                    "lookup_rpc_port": os.environ["LOOKUP_RPC_PORT"],
                    "backend": "mooncake",
                },
            },
        ]
    },
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
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.94}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  > "${LOG}" 2>&1 &

PID=$!
write_pid "${RUN_DIR}" server_pd_decode "${PID}"
printf '%s\n' "${KV_TRANSFER_CONFIG}" > "${RUN_DIR}/kv_transfer_config_pd_decode.json"

{
  echo "MODEL=${MODEL}"
  echo "PORT=${PORT}"
  echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
  echo "MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH}"
  echo "MOONCAKE_MASTER_ADDRESS=${MASTER_ADDR}"
  echo "KV_CONNECTOR=${KV_CONNECTOR}"
  echo "KV_ROLE=${KV_ROLE}"
  echo "KV_LOAD_FAILURE_POLICY=${KV_LOAD_FAILURE_POLICY}"
  echo "LOOKUP_RPC_PORT=${LOOKUP_RPC_PORT}"
  echo "KV_PORT=${KV_PORT}"
  echo "KV_TRANSFER_CONFIG=${KV_TRANSFER_CONFIG}"
} > "${RUN_DIR}/pd_decode_env.txt"

echo "RUN_DIR=${RUN_DIR}"
echo "SERVER_PID=${PID}"
echo "SERVER_LOG=${LOG}"
echo "MODEL=${MODEL}"
echo "PORT=${PORT}"
echo "KV_ROLE=${KV_ROLE}"
echo "KV_PORT=${KV_PORT}"
