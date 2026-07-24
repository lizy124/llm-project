unset ftp_proxy
unset https_proxy
unset http_proxy


export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
source /home/liziyu/CONFIG
export HCCL_EXEC_TIMEOUT=60
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=$IP_ADDRESS
export GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME
export TP_SOCKET_IFNAME=$NETWORK_CARD_NAME
export HCCL_SOCKET_IFNAME=$NETWORK_CARD_NAME
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1500
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_LLMDD_RPC_PORT=6657
export VLLM_TORCH_PROFILER_WITH_STACK=0
export TASK_QUEUE_ENABLE=1
export VLLM_LOGGING_LEVEL="info"
# export HCCL_INTRA_ROCE_ENABLE=1
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"
# cp /home/liziyu/pull_mamba/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py /usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
# cp /home/liziyu/pull_mamba/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py /usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py
# cp /home/liziyu/pull_mamba/vllm-ascend/vllm_ascend/core/recompute_scheduler.py /usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/core/recompute_scheduler.py

vllm serve /data/weights/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 30050 \
  --enable-expert-parallel \
  --no-disable-hybrid-kv-cache-manager \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --api-server-count 1 \
  --data-parallel-address 90.90.97.28 \
  --data-parallel-rpc-port 6884 \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --distributed-executor-backend mp \
  --served-model-name deepseek \
  --max-model-len 133120 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --speculative-config '{"num_speculative_tokens": 3,"method": "mtp","enforce_eager": true}' \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeHybridConnector",
  "kv_role": "kv_producer",
  "kv_port": "36000",
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
#    MooncakeHybridConnector
#    MooncakeConnectorV1
#    --no-enable-prefix-caching \