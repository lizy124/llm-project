#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="$(require_run_dir "${1:-}")"
MODEL="${MODEL_PATH:-/mnt/weight/Qwen3-0.6B}"
PORT="${PORT:-8100}"
DEVICE="${ASCEND_RT_VISIBLE_DEVICES:-0}"
LOG="${RUN_DIR}/server_kvpool_mixed.log"
MOONCAKE_CONFIG="${MOONCAKE_CONFIG_PATH:-${RUN_DIR}/mooncake.json}"
MASTER_ADDR="${MOONCAKE_MASTER_ADDRESS:-127.0.0.1:50088}"

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

KV_TRANSFER_CONFIG='{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "lookup_rpc_port": "1",
    "backend": "mooncake"
  }
}'

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
write_pid "${RUN_DIR}" server_kvpool_mixed "${PID}"
printf '%s\n' "${KV_TRANSFER_CONFIG}" > "${RUN_DIR}/kv_transfer_config.json"

echo "RUN_DIR=${RUN_DIR}"
echo "SERVER_PID=${PID}"
echo "SERVER_LOG=${LOG}"
echo "MODEL=${MODEL}"
echo "PORT=${PORT}"
echo "MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH}"
