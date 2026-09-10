#!/bin/bash
# start_dsv4_layerwise.sh — pool_165 内执行: DSV4-Flash layerwise 池化服务 (同事配置 165 适配版)
# 适配点: 移除已废弃 VLLM_ASCEND_APPLY_DSV4_PATCH(#10333); 去重 --model-loader-extra-config; 路径改 165
set -Eeuo pipefail

LOG=/home/lizhongyang/map_165/run/dsv4_layerwise_v2_8100.log

# ============ Configurable ============
TP_SIZE=8
DP_SIZE=1
MAX_MODEL_LEN=131072
MAX_NUM_BATCHED_TOKENS=10240
MAX_NUM_SEQS=64

# MemCache config paths
export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-local.conf

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/lib/
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}

# ---------- Environment Variables ----------
export PYTHONHASHSEED=0
export VLLM_USE_V1=1
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# ---------- Launch vLLM ----------
nohup vllm serve /mnt/weight/DeepSeek-V4-Flash-w8a8-mtp \
    --host 0.0.0.0 \
    --served-model-name dsv4 \
    --enable-expert-parallel \
    --no-disable-hybrid-kv-cache-manager \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --quantization ascend \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config '{
     "enable_multithread_load": true,
     "num_threads": 128
     }' \
    --no-enable-prefix-caching \
    --tensor-parallel-size "${TP_SIZE}" \
    --data-parallel-size "${DP_SIZE}" \
    --enforce-eager \
    --port 8100 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --block-size 32 \
    --gpu-memory-utilization 0.90 \
    --api-server-count 1 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex":true,
        "enable_static_kernel":false
        },
    "enable_cpu_binding": true,
    "enable_dsa_cp": true,
    "enable_flashcomm1": true}' \
    --kv-transfer-config '{
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "memcache",
                    "use_layerwise": false,
                     "layerwise_prefetch_layers":3
                }
            }' > "$LOG" 2>&1 &

echo "vllm serve started, pid $!, log: $LOG"
sleep 8
tail -8 "$LOG"
