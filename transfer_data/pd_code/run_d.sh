unset ftp_proxy
unset https_proxy
unset http_proxy

export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

#---------------------------------------打印日志---------------------------#
RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_ROOT=/home/zjs/script/logs_d2rh
LOG_DIR=${LOG_ROOT}/${RUN_ID}

mkdir -p "${LOG_DIR}"

export ASCEND_PROCESS_LOG_PATH="${LOG_DIR}"
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0

echo "ASCEND_PROCESS_LOG_PATH=${ASCEND_PROCESS_LOG_PATH}"

#---------------------------------------打印日志---------------------------#


export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
source /home/liziyu/CONFIG
export HCCL_EXEC_TIMEOUT=60
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=$IP_ADDRESS
export GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME
export TP_SOCKET_IFNAME=$NETWORK_CARD_NAME
export HCCL_SOCKET_IFNAME=$NETWORK_CARD_NAME
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1300
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_LLMDD_RPC_PORT=6657
export VLLM_TORCH_PROFILER_WITH_STACK=0
export TASK_QUEUE_ENABLE=1
export VLLM_LOGGING_LEVEL="info"
# export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"

#cp /home/zjs/script/connectors/mooncake_connector.py /usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
#cp /home/zjs/script/connectors/mooncake_d2rh_connector.py /usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_d2rh_connector.py

vllm serve /data/weights/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 30060 \
  --enable-expert-parallel \
  --no-disable-hybrid-kv-cache-manager \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --api-server-count 1 \
  --data-parallel-address 90.90.97.28 \
  --data-parallel-rpc-port 6884  \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --max-num-seqs 8 \
  --distributed-executor-backend mp \
  --served-model-name deepseek \
  --max-model-len 133120 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional_config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": true, "enable_shared_expert_dp":true}' \
  --speculative-config '{"num_speculative_tokens": 3,"method": "mtp","enforce_eager": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeHybridConnector",
  "kv_role": "kv_consumer",
  "kv_port": "36010",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 8,
                    "tp_size": 1
             }
      }
   }'
#   --compilation-config '{ "cudagraph_mode": "FULL_DECODE_ONLY"}' \
#    MooncakeConnectorV1
#    --enforce-eager \
#    --no-enable-prefix-caching \
#    --compilation-config '{"cudagraph_capture_sizes":[1, 4, 8, 16, 24, 32, 40, 48, 56, 64], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
#    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
# MooncakeHybridConnector