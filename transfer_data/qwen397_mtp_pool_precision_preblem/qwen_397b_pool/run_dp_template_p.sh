#!/bin/bash
set -euo pipefail

# ============================================================
# Arguments from launch_online_dp.py
#   $1: visible_devices
#   $2: vllm_engine_port
#   $3: dp_size
#   $4: dp_rank
#   $5: dp_address
#   $6: dp_rpc_port
#   $7: tp_size
# ============================================================

if [ "$#" -ne 7 ]; then
  echo "Usage: $0 <visible_devices> <port> <dp_size> <dp_rank> <dp_address> <dp_rpc_port> <tp_size>"
  echo "Example: $0 0,1 8000 4 0 90.90.97.27 12321 2"
  exit 1
fi
export VLLM_STARTUP_TIMEOUT=6000
VISIBLE_DEVICES="$1"
VLLM_ENGINE_PORT="$2"
DP_SIZE="$3"
DP_RANK="$4"
DP_ADDRESS="$5"
DP_RPC_PORT="$6"
TP_SIZE="$7"

# ============================================================
# Basic config
# ============================================================

LOCAL_IP="${LOCAL_IP:-90.90.97.27}"
NIC_NAME="${NIC_NAME:-enp194s0f0}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"

# P side: kv_producer
KV_ROLE="kv_producer"
KV_PORT="${KV_PORT:-55010}"

PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-${DP_SIZE}}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-${TP_SIZE}}"
DECODE_DP_SIZE="${DECODE_DP_SIZE:-${DP_SIZE}}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-${TP_SIZE}}"

# ============================================================
# Environment
# ============================================================

unset ftp_proxy FTP_PROXY
unset https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY

export IP_ADDRESS="${LOCAL_IP}"
export NETWORK_CARD_NAME="${NIC_NAME}"

export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"

export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"

export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE=1

export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"
export PYTHONHASHSEED=0

MOONCAKE_PY_PATH="${MOONCAKE_PY_PATH:-/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake}"
if [ -d "${MOONCAKE_PY_PATH}" ]; then
  export LD_LIBRARY_PATH="${MOONCAKE_PY_PATH}:${LD_LIBRARY_PATH:-}"
fi
export MOONCAKE_CONFIG_PATH="./mooncake.json"

JEMALLOC_PATH="${JEMALLOC_PATH:-/usr/lib/aarch64-linux-gnu/libjemalloc.so.2}"
if [ -f "${JEMALLOC_PATH}" ]; then
  export LD_PRELOAD="${JEMALLOC_PATH}:${LD_PRELOAD:-}"
fi

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

export DYNAMIC_EPLB="true"

# System tuning
if [ -w /proc/sys/vm/swappiness ]; then
  sysctl -w vm.swappiness=0 || true
fi
if [ -w /proc/sys/kernel/numa_balancing ]; then
  sysctl -w kernel.numa_balancing=0 || true
fi
if [ -w /proc/sys/kernel/sched_migration_cost_ns ]; then
  sysctl -w kernel.sched_migration_cost_ns=50000 || true
fi
if ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1; then
  echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null || true
fi

echo "============================================================"
echo "Starting vLLM Prefill (P) DP rank"
echo "  ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "  VLLM_HOST=${VLLM_HOST}"
echo "  VLLM_ENGINE_PORT=${VLLM_ENGINE_PORT}"
echo "  DP_SIZE=${DP_SIZE}"
echo "  DP_RANK=${DP_RANK}"
echo "  DP_ADDRESS=${DP_ADDRESS}"
echo "  DP_RPC_PORT=${DP_RPC_PORT}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  KV_ROLE=${KV_ROLE}"
echo "  KV_PORT=${KV_PORT}"
echo "============================================================"

# ============================================================
# Start vLLM — Prefill
# ============================================================

exec vllm serve /data/weights/Qwen3.5-397B-A17B-w4a8-org \
  --quantization ascend \
  --served-model-name qwen35 \
  --allowed-local-media-path / \
  --trust-remote-code \
  --data-parallel-size "${DP_SIZE}" \
  --data-parallel-rank "${DP_RANK}" \
  --data-parallel-address "${DP_ADDRESS}" \
  --data-parallel-rpc-port "${DP_RPC_PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --enable-expert-parallel \
  --host "${VLLM_HOST}" \
  --port "${VLLM_ENGINE_PORT}" \
  --max-num-seqs "${MAX_NUM_SEQS:-10}" \
  --max-model-len "${MAX_MODEL_LEN:-110000}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --seed "${SEED:-1024}" \
  --no-enable-prefix-caching \
  --async-scheduling \
  --enforce-eager \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --additional-config '{"enable_cpu_binding": true, "eplb_config":{"dynamic_eplb":true,"expert_heat_collection_interval":50,"algorithm_execution_interval":20, "eplb_policy_type":2, "num_redundant_experts":16}}' \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./profiling", "torch_profiler_with_stack": false}' \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode data \
  --distributed-executor-backend mp \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config \
  '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_producer",
      "kv_load_failure_policy": "fail",
      "kv_connector_extra_config": {
          "connectors": [
              {
                  "kv_connector": "MooncakeConnectorV1",
                  "kv_buffer_device": "npu",
                  "kv_role": "kv_producer",
                  "kv_port": "61001", 
                  "kv_connector_extra_config": {
                      "prefill": {
                          "dp_size": 2,
                          "tp_size": 4
                      },
                      "decode": {
                          "dp_size": 2,
                          "tp_size": 4
                      }
                  }
              },
              {
                  "kv_connector": "AscendStoreConnector",
                  "kv_role": "kv_producer",
                  "kv_connector_extra_config": {
                      "lookup_rpc_port":"0",
                      "backend": "mooncake"
                  }
              }
          ]
      }
  }'