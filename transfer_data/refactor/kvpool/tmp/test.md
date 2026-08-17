A3 DeepSeek-V2-Lite-Chat，使用0506分支代码测试精度和性能
PD混部，开启layerwise
rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export DISABLE_L2_CACHE=1;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
# export ASCEND_BUFFER_POOL=4:8;
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_NZ=1
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

# layerwise新增变量
export VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS=""
# export NUM_REUSE_LAYERS=27
# export NUM_LAYERS=27 
export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0"}}' 
        #> log_p_64_0506layerwise_pdmix_1.log 2>&1
启动测试：
ais_bench --models vllm_api_stream_chat --datasets gpqa_gen_0_shot_cot_chat_prompt --dump-eval-details


PD混部，关闭layerwise
rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export DISABLE_L2_CACHE=1;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
# export ASCEND_BUFFER_POOL=4:8;
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_NZ=1
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

# layerwise新增变量
# export VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS=""
# export NUM_REUSE_LAYERS=27
# export NUM_LAYERS=27 
export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"backend": "memcache","mooncake_rpc_port":"0"}}' > log_p_64_0506_closelayerwise_pdmix_1.log 2>&1

ais_bench --models vllm_api_stream_chat --datasets gpqa_gen_0_shot_cot_chat_prompt --dump-eval-details

PD混部，开启prefixcache，不开池化
rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export DISABLE_L2_CACHE=1;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
# export ASCEND_BUFFER_POOL=4:8;
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_NZ=1
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

# layerwise新增变量
# export VLLM_ASCEND_KV_POOL_LAYERWISE_INDEPENDENT_LAYERS=""
# export NUM_REUSE_LAYERS=27
# export NUM_LAYERS=27 
export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size 2 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' > log_0506_prefixcache_pdmix.log 2>&1

ais_bench --models vllm_api_stream_chat --datasets gpqa_gen_0_shot_cot_chat_prompt --dump-eval-details


PD分离，关闭layerwise
p节点：
rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export PYTHONHASHSEED=0
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

# layerwise新增变量
# export NUM_REUSE_LAYERS=4
# export NUM_LAYERS=27 
export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8005 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
		"kv_connector": "MultiConnector",
		"kv_role": "kv_producer",
		"kv_connector_extra_config": {
			"connectors": [
			{
					"kv_connector": "MooncakeConnectorV1",
					"kv_role": "kv_producer",
					"kv_buffer_device": "npu",
					"kv_rank": 0,
					"kv_port": "36001",
					"kv_connector_extra_config": {
						"use_ascend_direct": true,
						"prefill": {
							"dp_size": 1,
							"tp_size": 1
						},
						"decode": {
							"dp_size": 1,
							"tp_size": 1
						}
					}
				},{
					"kv_connector": "AscendStoreConnector",
					"kv_role": "kv_producer",
					"kv_connector_extra_config":{
						"backend": "memcache",
						"mooncake_rpc_port":"0",
						"use_layerwise": false
					}
				}
			]
		}
	}' > log_0506_closelayerwise_1p.log 2>&1
	

d节点：

rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export PYTHONHASHSEED=0
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

# layerwise新增变量
# export NUM_REUSE_LAYERS=4
# export NUM_LAYERS=27 
export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8006 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
		"kv_connector": "MultiConnector",
		"kv_role": "kv_consumer",
		"kv_connector_extra_config": {
			"connectors": [
			{
					"kv_connector": "MooncakeConnectorV1",
					"kv_role": "kv_consumer",
					"kv_buffer_device": "npu",
					"kv_rank": 1,
					"kv_port": "36021",
					"kv_connector_extra_config": {
						"use_ascend_direct": true,
						"prefill": {
							"dp_size": 1,
							"tp_size": 1
						},
						"decode": {
							"dp_size": 1,
							"tp_size": 1
						}
					}
				}
			]
		}
	}' > log_0506_closelayerwise_1d.log 2>&1

Start proxy_server(PD分离场景下用于负载均衡和控制kv 传输的特定脚本，

python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 80.5.17.112 \
    --port 8004 \
    --prefiller-hosts 80.5.17.112 \
    --prefiller-ports 8005 \
    --decoder-hosts 80.5.17.112 \
    --decoder-ports 8006

curl下请求：
curl -s http://80.5.17.112:8004/v1/completions -H "Content-Type: application/json" -d '{ "model": "ds2", "prompt": "Hello. I have a question. The president of the United States is", "max_completion_tokens": 200, "temperature":0.0 }'

PD分离，开启layerwise
p节点：
rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export PYTHONHASHSEED=0
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

# layerwise新增变量
# export NUM_REUSE_LAYERS=4
# export NUM_LAYERS=27 
export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8005 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
		"kv_connector": "MultiConnector",
		"kv_role": "kv_producer",
		"kv_connector_extra_config": {
			"use_layerwise": true,
			"connectors": [
			{
					"kv_connector": "MooncakeLayerwiseConnector",
					"kv_role": "kv_producer",
					"kv_buffer_device": "npu",
					"kv_rank": 0,
					"kv_port": "36001",
					"kv_connector_extra_config": {
						"use_ascend_direct": true,
						"prefill": {
							"dp_size": 1,
							"tp_size": 1
						},
						"decode": {
							"dp_size": 1,
							"tp_size": 1
						}
					}
				},{
					"kv_connector": "AscendStoreConnector",
					"kv_role": "kv_producer",
					"kv_connector_extra_config":{
						"backend": "memcache",
						"mooncake_rpc_port":"0",
						"use_layerwise": true
					}
				}
			]
		}
	}' > log_0506_layerwise_1p.log 2>&1
	
d节点：
rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export PYTHONHASHSEED=0
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

# layerwise新增变量
# export NUM_REUSE_LAYERS=4
# export NUM_LAYERS=27 
export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8006 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
		"kv_connector": "MultiConnector",
		"kv_role": "kv_consumer",
		"kv_connector_extra_config": {
				"use_layerwise": true,
			"connectors": [
			{
					"kv_connector": "MooncakeLayerwiseConnector",
					"kv_role": "kv_consumer",
					"kv_buffer_device": "npu",
					"kv_rank": 1,
					"kv_port": "36021",
					"kv_connector_extra_config": {
						"use_ascend_direct": true,
						"prefill": {
							"dp_size": 1,
							"tp_size": 1
						},
						"decode": {
							"dp_size": 1,
							"tp_size": 1
						}
					}
				}
			]
		}
	}' > log_0506_layerwise_1d.log 2>&1
Start proxy_server(PD分离场景下用于负载均衡和控制kv 传输的特定脚本
python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
    --host 80.5.17.112 \
    --port 8004 \
    --prefiller-hosts 80.5.17.112 \
    --prefiller-ports 8005 \
    --decoder-hosts 80.5.17.112 \
    --decoder-ports 8006

curl下请求：
curl -s http://80.5.17.112:8004/v1/completions -H "Content-Type: application/json" -d '{ "model": "ds2", "prompt": "Hello. I have a question. The president of the United States is", "max_completion_tokens": 200, "temperature":0.0 }'

PD分离，开启layerwise+offload
p节点：
rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export PYTHONHASHSEED=0
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

# offload新增变量
export VLLM_ASCEND_KV_POOL_LAYERWISE_PREFETCH_LAYERS=2
export VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS=2

export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8005 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 40 \
    --max-model-len 16384 \
    --max-num-batched-tokens 131072 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
		"kv_connector": "MultiConnector",
		"kv_role": "kv_producer",
		"kv_connector_extra_config": {
			"connectors": [
			{
					"kv_connector": "MooncakeLayerwiseConnector",
					"kv_role": "kv_producer",
					"kv_buffer_device": "npu",
					"kv_rank": 0,
					"kv_port": "36001",
					"kv_connector_extra_config": {
						"use_ascend_direct": true,
						"prefill": {
							"dp_size": 1,
							"tp_size": 1
						},
						"decode": {
							"dp_size": 1,
							"tp_size": 1
						}
					}
				},{
					"kv_connector": "AscendStoreConnector",
					"kv_role": "kv_producer",
					"kv_connector_extra_config":{
						"backend": "memcache",
						"mooncake_rpc_port":"0",
						"use_layerwise": true
					}
				}
			]
		}
	}' > log_0506_layerwise_offload_1p.log 2>&1
	
d节点：
rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export PYTHONHASHSEED=0
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

# offload新增变量
export VLLM_ASCEND_KV_POOL_LAYERWISE_PREFETCH_LAYERS=2
export VLLM_ASCEND_KV_POOL_LAYERWISE_NUM_SHARED_BUFFERS=2

export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8006 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 40 \
    --max-model-len 16384 \
    --max-num-batched-tokens 131072 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
		"kv_connector": "MultiConnector",
		"kv_role": "kv_consumer",
		"kv_connector_extra_config": {
			"connectors": [
			{
					"kv_connector": "MooncakeLayerwiseConnector",
					"kv_role": "kv_consumer",
					"kv_buffer_device": "npu",
					"kv_rank": 1,
					"kv_port": "36021",
					"kv_connector_extra_config": {
						"use_ascend_direct": true,
						"prefill": {
							"dp_size": 1,
							"tp_size": 1
						},
						"decode": {
							"dp_size": 1,
							"tp_size": 1
						}
					}
				}
			]
		}
	}' > log_0506_layerwise_offload_1d.log 2>&1

proxy：
python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py     --host 80.5.17.112     --port 8004     --prefiller-hosts 80.5.17.112     --prefiller-ports 8005     --decoder-hosts 80.5.17.112     --decoder-ports 8006

aisbench测试性能：
python3 aisbench_test.py --input_len 15000 --output_len 1 --data_num 512 --concurrency 60 --request_rate 0 --dataset_type prefix_cache --repeat_rate 0.9 --prefix_test --seed 1024

PD分离，开启layerwise，不offload
p节点：
rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export PYTHONHASHSEED=0
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf


export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8005 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 40 \
    --max-model-len 16384 \
    --max-num-batched-tokens 131072 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
		"kv_connector": "MultiConnector",
		"kv_role": "kv_producer",
		"kv_connector_extra_config": {
			"connectors": [
			{
					"kv_connector": "MooncakeLayerwiseConnector",
					"kv_role": "kv_producer",
					"kv_buffer_device": "npu",
					"kv_rank": 0,
					"kv_port": "36001",
					"kv_connector_extra_config": {
						"use_ascend_direct": true,
						"prefill": {
							"dp_size": 1,
							"tp_size": 1
						},
						"decode": {
							"dp_size": 1,
							"tp_size": 1
						}
					}
				},{
					"kv_connector": "AscendStoreConnector",
					"kv_role": "kv_producer",
					"kv_connector_extra_config":{
						"backend": "memcache",
						"mooncake_rpc_port":"0",
						"use_layerwise": true
					}
				}
			]
		}
	}' > log_0506_layerwise_nooffload_1p.log 2>&1
	
d节点：
rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=2,3
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export PYTHONHASHSEED=0
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

export VLLM_VERSION=0.19.0

vllm serve /home/weight/DeepSeek-V2-Lite-Chat \
    --host 0.0.0.0 \
    --port 8006 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name ds2 \
    --enable-expert-parallel \
    --max-num-seqs 40 \
    --max-model-len 16384 \
    --max-num-batched-tokens 131072 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
		"kv_connector": "MultiConnector",
		"kv_role": "kv_consumer",
		"kv_connector_extra_config": {
			"connectors": [
			{
					"kv_connector": "MooncakeLayerwiseConnector",
					"kv_role": "kv_consumer",
					"kv_buffer_device": "npu",
					"kv_rank": 1,
					"kv_port": "36021",
					"kv_connector_extra_config": {
						"use_ascend_direct": true,
						"prefill": {
							"dp_size": 1,
							"tp_size": 1
						},
						"decode": {
							"dp_size": 1,
							"tp_size": 1
						}
					}
				}
			]
		}
	}' > log_0506_layerwise_nooffload_1d.log 2>&1

proxy：
python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py     --host 80.5.17.112     --port 8004     --prefiller-hosts 80.5.17.112     --prefiller-ports 8005     --decoder-hosts 80.5.17.112     --decoder-ports 8006
aisbench性能：
python3 aisbench_test.py --input_len 15000 --output_len 1 --data_num 512 --concurrency 60 --request_rate 0 --dataset_type prefix_cache --repeat_rate 0.9 --prefix_test --seed 1024
