# 03 验证脚本全文 — 阶段性归档

原归档范围仍为2026-10-13暂停时的实际选定版本；2026-10-19仅重排附录为固定三部分，文件代码块不变。当前工作目录的配置路径修订未复验，不能用当前文件冒充本归档实测版本。

## 第一部分：配置与拉起服务

归档全文就是本次选定快照；下表路径用于定位任务内材料，不能用工作目录后来修改的启动脚本替换归档实测版本。运行起点与操作顺序见 [04](04_verification_env_and_launch.md)。P=135、D=133，均指 `lizy_1009` 容器；`scripts` 指 `/home/lizhongyang/1011/scripts`。

| 文件及全文 | 任务内路径 | 用途 | 运行位置 | 输入/配置 | 输出 | 验证状态 |
|---|---|---|---|---|---|---|
| [mmc-meta.conf](#archive-0-0) | `03_verification_scripts/config/mmc-meta.conf` | MetaService 配置 | P 容器 | 135 的 5000/6000/8004 端口 | Meta/配置存储/指标端点配置 | 有效池化轮使用 |
| [mmc-local.conf](#archive-0-1) | `03_verification_scripts/config/mmc-local.conf` | 引擎侧 memcache 配置 | P、D 容器 | device_sdma、world_size=256、DRAM=0GB | 引擎连接外部池的配置 | 有效池化轮使用 |
| [mmc-local-standalone.conf](#archive-0-2) | `03_verification_scripts/config/mmc-local-standalone.conf` | standalone 池容量配置 | P、D 容器 | 同一 Meta、DRAM=20GB | standalone 池配置 | 有效池化轮使用 |
| [meta.sh](#archive-0-3) | `03_verification_scripts/launch/meta.sh` | 启动 MetaService | P 容器 scripts | CANN、同目录 mmc-meta.conf | Meta 服务及轮次启动日志 | 有效池化轮运行；工作目录路径修订未复验 |
| [standalone.sh](#archive-0-4) | `03_verification_scripts/launch/standalone.sh` | 启动池服务 | P、D 容器 scripts | 同目录 standalone 配置、设备 0 | 初始化成功日志、池服务 | 有效池化轮运行；工作目录路径修订未复验 |
| [memcache_standalone.py](#archive-0-5) | `03_verification_scripts/launch/memcache_standalone.py` | 初始化 DistributedObjectStore 并保持服务存活 | P、D 容器，由 standalone.sh 调用 | MMC_LOCAL_CONFIG_PATH | 初始化结果；失败时非零退出 | 有效池化轮运行 |
| [p.sh](#archive-0-6) | `03_verification_scripts/launch/p.sh` | 池化 Prefill，DP4/TP4 | P 容器 scripts | pr-18277、模型、MooncakeHybridConnector + AscendStoreConnector | 8910 服务、p.log、轮次 .out | 有效池化轮运行；工作目录路径修订未复验 |
| [d.sh](#archive-0-7) | `03_verification_scripts/launch/d.sh` | 池化 Decode，DP16/TP1 | D 容器 scripts | pr-18277、模型、两类 Connector | 8920 服务、d.log、轮次 .out | 有效池化轮运行；工作目录路径修订未复验 |
| [p_nopool.sh](#archive-0-8) | `03_verification_scripts/launch/p_nopool.sh` | 无池化 Prefill 基线 | P 容器 scripts | master、模型，仅 MooncakeHybridConnector | 8910 服务、p_nopool.log、轮次 .out | 启动阶段中止，未完成正式无池化测试；工作目录修订未复验 |
| [d_nopool.sh](#archive-0-9) | `03_verification_scripts/launch/d_nopool.sh` | 无池化 Decode 基线 | D 容器 scripts | master、模型，仅 MooncakeHybridConnector | 8920 服务、d_nopool.log、轮次 .out | 启动阶段中止，未完成正式无池化测试；工作目录修订未复验 |
| [proxy.sh](#archive-0-10) | `03_verification_scripts/launch/proxy.sh` | 启动原 P→D 代理 | P 容器 scripts | 原代理外部路径、P:8910、D:8920 | 135:8100、proxy.log、轮次 .out | 有效池化轮运行；工作目录路径修订未复验 |
| [load_balance_proxy_server_example.py](#archive-0-11) | `03_verification_scripts/launch/load_balance_proxy_server_example.py` | 转发请求和 P→D KV 交换 | P 容器；历史部署在 proxy.sh 指定的外部路径 | 后端地址、OpenAI 兼容请求 | 流式响应、kv_transfer_params 传递 | 有效池化轮正式请求使用 |

<a id="archive-0-0"></a>
### mmc-meta.conf

````ini
ock.mmc.meta_service_url = tcp://80.5.9.135:5000
ock.mmc.meta_service.config_store_url = tcp://80.5.9.135:6000
ock.mmc.meta_service.metrics_url = http://80.5.9.135:8004
ock.mmc.log_level = error
````


<a id="archive-0-1"></a>
### mmc-local.conf

````ini
ock.mmc.meta_service_url = tcp://80.5.9.135:5000
ock.mmc.local_service.config_store_url = tcp://80.5.9.135:6000
ock.mmc.log_level = error
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 0GB
ock.mmc.local_service.max.dram.size = 1024GB
````


<a id="archive-0-2"></a>
### mmc-local-standalone.conf

````ini
ock.mmc.meta_service_url = tcp://80.5.9.135:5000
ock.mmc.local_service.config_store_url = tcp://80.5.9.135:6000
ock.mmc.log_level = error
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 20GB
ock.mmc.local_service.max.dram.size = 1024GB
````


<a id="archive-0-3"></a>
### meta.sh

````bash
#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh
export MMC_META_CONFIG_PATH="$PWD/mmc-meta.conf"
python3 -u -c 'from memcache_hybrid import MetaService; MetaService.main()'
````


<a id="archive-0-4"></a>
### standalone.sh

````bash
#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MMC_LOCAL_CONFIG_PATH="$PWD/mmc-local-standalone.conf"
python3 -u memcache_standalone.py
````


<a id="archive-0-5"></a>
### memcache_standalone.py

````python
import time

from memcache_hybrid import DistributedObjectStore


if __name__ == "__main__":
    store = DistributedObjectStore()
    result = store.init(0)
    if result != 0:
        raise SystemExit(f"Failed to initialize memcache, res = {result}")
    print("Successfully initialized memcache.", flush=True)
    while True:
        time.sleep(1)
````


<a id="archive-0-6"></a>
### p.sh

````bash
#!/usr/bin/env bash
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh

export PYTHONHASHSEED=0
export MMC_LOCAL_CONFIG_PATH="$PWD/mmc-local.conf"

unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

nic_name=enp194s0f0
local_ip=80.5.9.135

export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve /mnt/share/weights/DeepSeek-V4-Flash-0731-w8a8 \
  --host 0.0.0.0 \
  --port 8910 \
  --data-parallel-size 4 \
  --data-parallel-address $local_ip \
  --data-parallel-rpc-port 12321 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name model \
  --max-model-len 1048576 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --no-disable-hybrid-kv-cache-manager \
  --model-loader-extra-config='{"enable_multithread_load":true,"num_threads":128}' \
  --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
  --trust-remote-code \
  --block-size 32 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --enforce-eager \
  --enable-prefix-caching \
  --attention_config.indexer_kv_dtype int8 \
  --additional-config '{"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_flashcomm1":true}' \
  --kv-transfer-config \
    '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"30000","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":16,"tp_size":1}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"memcache"}}]}}' 2>&1 | tee p.log
````


<a id="archive-0-7"></a>
### d.sh

````bash
#!/usr/bin/env bash
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh

export PYTHONHASHSEED=0
export MMC_LOCAL_CONFIG_PATH="$PWD/mmc-local.conf"

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

nic_name=enp194s0f0
local_ip=80.5.9.133

export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1800
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve /mnt/share/weights/DeepSeek-V4-Flash-0731-w8a8 \
    --host 0.0.0.0 \
    --port 8920 \
    --data-parallel-size 16 \
    --data-parallel-address $local_ip \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name model \
    --max-model-len 1048576 \
    --max-num-batched-tokens 400 \
    --max-num-seqs 32 \
    --async-scheduling \
    --block-size 32 \
    --enable-prefix-caching \
    --attention_config.indexer_kv_dtype int8 \
    --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config='{"enable_multithread_load":true,"num_threads":128}' \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"30200","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":16,"tp_size":1}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"memcache"}}]}}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":false}' 2>&1 | tee d.log
````


<a id="archive-0-8"></a>
### p_nopool.sh

````bash
#!/usr/bin/env bash
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh

export PYTHONHASHSEED=0

unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

nic_name=enp194s0f0
local_ip=80.5.9.135

export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve /mnt/share/weights/DeepSeek-V4-Flash-0731-w8a8 \
  --host 0.0.0.0 \
  --port 8910 \
  --data-parallel-size 4 \
  --data-parallel-address $local_ip \
  --data-parallel-rpc-port 12321 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name model \
  --max-model-len 1048576 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --no-disable-hybrid-kv-cache-manager \
  --model-loader-extra-config='{"enable_multithread_load":true,"num_threads":128}' \
  --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
  --trust-remote-code \
  --block-size 32 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --enforce-eager \
  --enable-prefix-caching \
  --attention_config.indexer_kv_dtype int8 \
  --additional-config '{"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_flashcomm1":true}' \
  --kv-transfer-config \
    '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"30000","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":16,"tp_size":1}}}]}}' 2>&1 | tee p_nopool.log
````


<a id="archive-0-9"></a>
### d_nopool.sh

````bash
#!/usr/bin/env bash
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh

export PYTHONHASHSEED=0

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

nic_name=enp194s0f0
local_ip=80.5.9.133

export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1800
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve /mnt/share/weights/DeepSeek-V4-Flash-0731-w8a8 \
    --host 0.0.0.0 \
    --port 8920 \
    --data-parallel-size 16 \
    --data-parallel-address $local_ip \
    --data-parallel-rpc-port 12321 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name model \
    --max-model-len 1048576 \
    --max-num-batched-tokens 400 \
    --max-num-seqs 32 \
    --async-scheduling \
    --block-size 32 \
    --enable-prefix-caching \
    --attention_config.indexer_kv_dtype int8 \
    --no-disable-hybrid-kv-cache-manager \
    --model-loader-extra-config='{"enable_multithread_load":true,"num_threads":128}' \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"30200","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":16,"tp_size":1}}}]}}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":false}' 2>&1 | tee d_nopool.log
````


<a id="archive-0-10"></a>
### proxy.sh

````bash
#!/usr/bin/env bash
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 /home/w00985460/script/vllm/load_balance_proxy_server_example.py \
    --prefiller-hosts 80.5.9.135 --prefiller-ports 8910 \
    --decoder-hosts 80.5.9.133 --decoder-ports 8920 \
    --host 80.5.9.135 --port 8100 2>&1 | tee proxy.log
````


<a id="archive-0-11"></a>
### launch/load_balance_proxy_server_example.py

````python
# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 9000 \
#     --prefiller-hosts 141.61.73.112 141.61.73.112 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 141.61.73.112 141.61.73.112 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Step 5: Add or Remove Prefiller or Decoder Instances (Optional)
# ---------------------------------------------------------------
# You can add or remove prefiller or decoder instances after the proxy is started.
# For example, add 2 prefiller instances:
#
#   curl -X POST http://localhost:9000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["141.61.73.112:8102", "141.61.73.112:8103"]
#         }'
#
# or remove 1 decoder instance:
#
#   curl -X POST http://localhost:9000/instances/remove \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "decode",
#           "instances": "141.61.73.112:8201"
#         }'
#
# This will return a JSON object with the adding or removing info
# and the current prefiller and decoder instances.
#
# When adding instances, if the instances are not started,
# the proxy will wait and try until the instances to be started
# or exceeding the number of attempts
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import functools
import heapq
import ipaddress
import json
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Add uvloop for faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"


TAINT_PRIORITY = 1e15


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        try:
            ip = ipaddress.ip_address(self.host)
            if isinstance(ip, ipaddress.IPv6Address):
                self.url = f"http://[{host}]:{port}/v1"
        except Exception:
            pass
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0  # Number of active requests
        self.aborted_requests = set()  # Track aborted requests
        # Removed individual server lock - will use global locks instead

    def __eq__(self, other):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("141.61.73.112", "0.0.0.0")
        other_host = other.host.replace("localhost", "0.0.0.0").replace("141.61.73.112", "0.0.0.0")
        return self_host == other_host and str(self.port) == str(other.port)

    def __hash__(self):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("141.61.73.112", "0.0.0.0")
        return hash((self_host, str(self.port)))

    def __repr__(self):
        return f"{self.host}:{self.port}"


class ProxyState:
    def __init__(self, prefiller_instances, decoder_instances):
        self.request_num = 0
        self.tainted_prefillers: list[ServerState] = []
        self.tainted_decoders: list[ServerState] = []
        self.node_listener = NodeListener(self)

        self.prefillers: list[ServerState] = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        # Removed selection locks - no longer needed for synchronous methods

        # Initialize priority queues for efficient server selection
        # Each entry is (priority_score, server_index, server_reference)
        # Lower priority score = higher priority (less loaded)
        self.prefiller_heap = [(0.0, i, server) for i, server in enumerate(self.prefillers)]
        self.decoder_heap = [(0.0, i, server) for i, server in enumerate(self.decoders)]
        heapq.heapify(self.prefiller_heap)
        heapq.heapify(self.decoder_heap)

    def _update_prefiller_priority(self, server_idx: int):
        """Update the priority of a prefiller server in the heap."""
        server = self.prefillers[server_idx]
        # Priority based on active_tokens and active_kv_cache
        priority = server.active_tokens + server.active_kv_cache * 0.3
        # Remove old entry and add new one
        self.prefiller_heap = [(p, i, s) for p, i, s in self.prefiller_heap if i != server_idx]
        heapq.heappush(self.prefiller_heap, (priority, server_idx, server))

    def _update_decoder_priority(self, server_idx: int):
        """Update the priority of a decoder server in the heap."""
        server = self.decoders[server_idx]
        priority = server.active_tokens
        # Remove old entry and add new one
        self.decoder_heap = [(p, i, s) for p, i, s in self.decoder_heap if i != server_idx]
        heapq.heappush(self.decoder_heap, (priority, server_idx, server))

    def abort_prefiller_request(self, server_idx: int, request_id):  # Changed to synchronous
        """
        Mark a request as aborted. This will helps to release kv cache in
        prefiller node.
        """
        # No lock needed - atomic operation
        if server_idx >= len(self.prefillers):
            return
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int):  # Changed to synchronous
        """
        Get the set of aborted requests and clear it.
        This is used to release kv cache in prefiller node.
        """
        # No lock needed - atomic operation
        if server_idx >= len(self.prefillers):
            return set()
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_prefiller(self, token_count):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.prefiller_heap:
            raise RuntimeError("No prefiller servers available")

        priority, chosen, server = heapq.heappop(self.prefiller_heap)

        # Update the chosen server atomically
        self.prefillers[chosen].active_tokens += token_count
        self.prefillers[chosen].active_kv_cache += token_count

        # Update priority and re-add to heap
        self._update_prefiller_priority(chosen)

        return chosen

    def release_prefiller(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.prefillers):
            return
        self.prefillers[idx].active_tokens -= token_count
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.prefillers):
            return
        if self.prefillers[idx].active_kv_cache > 0:
            self.prefillers[idx].active_kv_cache -= token_count
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def select_decoder(self, token_count):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.decoder_heap:
            raise RuntimeError("No decoder servers available")

        priority, chosen, server = heapq.heappop(self.decoder_heap)

        # Update the chosen server atomically
        self.decoders[chosen].active_tokens += token_count

        # Update priority and re-add to heap
        self._update_decoder_priority(chosen)

        return chosen

    def release_decoder(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.decoders):
            return
        self.decoders[idx].active_tokens -= token_count
        # Update priority queue after releasing
        self._update_decoder_priority(idx)

    # Omni_infer's calculate_input_scores function
    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        input_score = length_score * 0.0345 + 120.0745
        return input_score

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length

    async def add_instances(self, instance_type: str, instances: list[ServerState]) -> tuple[list[str], list[str]]:
        added_nodes, waiting_nodes = [], []
        for server in instances:
            is_valid = await self.node_listener.check_instance_status(server.client)
            if is_valid and instance_type == InstanceType.PREFILL:
                self.add_prefillers([server])
                added_nodes.append(str(server))
            elif is_valid and instance_type == InstanceType.DECODE:
                self.add_decoders([server])
                added_nodes.append(str(server))
            else:
                node = str(server)
                self.node_listener.waiting_nodes[node] = (instance_type, server, 0)
                waiting_nodes.append(node)
        return added_nodes, waiting_nodes

    def add_prefillers(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_prefillers:
                self.tainted_prefillers.remove(server)
                self.prefiller_heap = [
                    (0, idx, server) if srv == server else (priority, idx, srv)
                    for priority, idx, srv in self.prefiller_heap
                ]
                heapq.heapify(self.prefiller_heap)
            elif server not in self.prefillers:
                self.prefillers.append(server)
                # prefiller_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]
                heapq.heappush(self.prefiller_heap, (0, len(self.prefillers) - 1, server))
        self.print_status(f"Add prefiller instances: {instances}.")

    def add_decoders(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_decoders:
                self.tainted_decoders.remove(server)
                self.decoder_heap = [
                    (0, idx, server) if srv == server else (priority, idx, srv)
                    for priority, idx, srv in self.decoder_heap
                ]
                heapq.heapify(self.decoder_heap)
            elif server not in self.decoders:
                self.decoders.append(server)
                # decoder_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]
                heapq.heappush(self.decoder_heap, (0, len(self.decoders) - 1, server))
        self.print_status(f"Add decoder instances: {instances}.")

    def remove_prefillers(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        if self.request_num > 0:
            logger.warning(f"Start to taint prefill instances {instances}.")
            self._taint_prefillers(instances)
            return True

        instances_to_remove = set(instances)
        self.prefillers = [server for server in self.prefillers if server not in instances_to_remove]
        prefiller_heap_copy = self.prefiller_heap.copy()
        prefiller_heap_copy.sort(key=lambda x: x[1])  # sorted by key: prefiller_idx
        prefiller_heap = []
        idx = 0
        for priority, _, server in prefiller_heap_copy:
            if server not in instances_to_remove:
                prefiller_heap.append((priority, idx, server))
                idx += 1

        # prefiller_heap: [(priority_0, 0, server_0), (priority_1, 1, server_1)] -> [(priority_1, 0, server_1)]
        self.prefiller_heap = prefiller_heap
        heapq.heapify(self.prefiller_heap)
        self.print_status(f"Remove prefiller instances: {instances}.")
        return False

    def remove_decoders(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        if self.request_num > 0:
            logger.warning(f"Start to taint decode instances {instances}.")
            self._taint_decoders(instances)
            return True

        instances_to_remove = set(instances)
        self.decoders = [server for server in self.decoders if server not in instances_to_remove]
        decoder_heap_copy = self.decoder_heap.copy()
        decoder_heap_copy.sort(key=lambda x: x[1])  # sorted by key: decoder_idx
        decoder_heap = []
        idx = 0
        for priority, _, server in decoder_heap_copy:
            if server not in instances_to_remove:
                decoder_heap.append((priority, idx, server))
                idx += 1

        # decoder_heap: [(priority_0, 0, server_0), (priority_1, 1, server_1)] -> [(priority_1, 0, server_1)]
        self.decoder_heap = decoder_heap
        heapq.heapify(self.decoder_heap)
        self.print_status(f"Remove decoder instances: {instances}.")
        return False

    def _taint_prefillers(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.prefillers:
            if server in instances_to_taint and server not in self.tainted_prefillers:
                self.tainted_prefillers.append(server)

        self.prefiller_heap = [
            (TAINT_PRIORITY, idx, srv) if srv in instances_to_taint else (priority, idx, srv)
            for priority, idx, srv in self.prefiller_heap
        ]
        heapq.heapify(self.prefiller_heap)

    def _taint_decoders(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.decoders:
            if server in instances_to_taint and server not in self.tainted_decoders:
                self.tainted_decoders.append(server)

        self.decoder_heap = [
            (TAINT_PRIORITY, idx, srv) if srv in instances_to_taint else (priority, idx, srv)
            for priority, idx, srv in self.decoder_heap
        ]
        heapq.heapify(self.decoder_heap)

    def print_status(self, msg: str) -> None:
        status = {
            "prefill_instances": [str(server) for server in self.prefillers],
            "decode_instances": [str(server) for server in self.decoders],
        }
        print(f"{msg} Status: {status}")


proxy_state = None


class NodeListener:
    def __init__(self, proxy):
        self.proxy_state = proxy
        self.waiting_nodes: dict[str, tuple[str, Any, int]] = {}
        self.listening_thread = threading.Thread(target=self._node_listener, daemon=True)
        self.listening_thread.start()

    def _node_listener(self) -> None:
        while True:
            for node, (instance_type, server, check_times) in list(self.waiting_nodes.items()):
                is_valid = asyncio.run(self.check_instance_status(server.client))
                print(f"Checking instance {node}...")
                check_times += 1
                if is_valid:
                    if instance_type == InstanceType.PREFILL:
                        self.proxy_state.add_prefillers([server])
                    else:
                        self.proxy_state.add_decoders([server])
                    self.waiting_nodes.pop(node)
                elif check_times == global_args.max_waiting_retries:
                    print(f"Instance {node} was not added to the proxy.")
                    self.waiting_nodes.pop(node)
                else:
                    self.waiting_nodes[node] = (instance_type, server, check_times)

            if self.proxy_state.tainted_prefillers and not self.proxy_state.request_num:
                need_waiting = self.proxy_state.remove_prefillers(self.proxy_state.tainted_prefillers)
                if not need_waiting:
                    self.proxy_state.tainted_prefillers.clear()

            if self.proxy_state.tainted_decoders and not self.proxy_state.request_num:
                need_waiting = self.proxy_state.remove_decoders(self.proxy_state.tainted_decoders)
                if not need_waiting:
                    self.proxy_state.tainted_decoders.clear()
            time.sleep(global_args.waiting_retry_interval)

    @staticmethod
    async def check_instance_status(client: httpx.AsyncClient) -> bool:
        endpoint = "/models"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            response = await client.get(endpoint, headers=headers)
            response.raise_for_status()
            return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    parser.add_argument(
        "--max-waiting-retries", type=int, default=3, help="Maximum number of retries for waiting nodes to be started"
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting nodes to be started",
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.prefiller_instances, global_args.decoder_instances)
    print(f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients.")
    yield
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()


async def listen_for_disconnect(request: Request) -> None:
    """Return if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_id: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    aborted_requests = proxy_state.acquire_aborted_prefiller_requests(prefiller_id)
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": list(aborted_requests),
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    req_data["min_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
                raise last_exc


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return  # Success, exit after streaming
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise e
        except Exception as e:
            # If any chunk has been sent, do not retry, just log and drop
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error(f"Streaming to client interrupted after response started: {str(e)}")
                return
            else:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                else:
                    logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                    raise e


async def _handle_select_instance(api: str, req_data: Any, request_length: int):
    prefiller_score = proxy_state.calculate_prefill_scores(request_length)
    logger.debug(f"Request length: {request_length}, Prefiller score: {prefiller_score}")
    request_id = await proxy_state.next_req_id()
    # Select prefiller
    prefiller_idx = proxy_state.select_prefiller(prefiller_score)
    prefiller = proxy_state.prefillers[prefiller_idx]
    # Send request to prefiller
    response = await send_request_to_service(
        prefiller.client,
        prefiller_idx,
        api,
        req_data,
        request_id,
        max_retries=global_args.max_retries,
        base_delay=global_args.retry_delay,
    )
    proxy_state.release_prefiller(prefiller_idx, prefiller_score)
    response_json = response.json()
    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params
    # Select decoder
    decoder_score = proxy_state.calculate_decode_scores(request_length)
    logger.debug("Decoder score: %f", decoder_score)
    # Use the prefiller's kv_transfer_params to select decoder
    decoder_idx = proxy_state.select_decoder(decoder_score)
    decoder = proxy_state.decoders[decoder_idx]
    logger.debug("Using %s %s", prefiller.url, decoder.url)
    return InstanceInfo(
        request_id=request_id,
        prefiller_idx=prefiller_idx,
        prefiller_score=prefiller_score,
        prefiller=prefiller,
        decoder=decoder,
        decoder_idx=decoder_idx,
        decoder_score=decoder_score,
    )


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_idx: int
    prefiller_score: float
    prefiller: ServerState
    decoder_idx: int
    decoder_score: float
    decoder: ServerState


async def _handle_completions(api: str, request: Request):
    try:
        proxy_state.request_num += 1
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        instance_info = await _handle_select_instance(api, req_data, request_length)
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        # refer to vLLM sampling_params: max_token default value
        origin_max_tokens = req_data.get("max_tokens", 16)

        async def generate_stream():
            nonlocal instance_info
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0
            # Only one await per chunk, minimal logic in loop
            try:
                while retry:
                    retry = False
                    async for chunk in stream_service_response_with_retry(
                        instance_info.decoder.client,
                        api,
                        req_data,
                        request_id=instance_info.request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    ):
                        if not released_kv and chunk:
                            proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                            released_kv = True
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            logger.debug(f"Skipping chunk: {chunk}")
                            yield chunk
                            continue
                        if not chunk_str:
                            continue
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: ") :]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            # if chunk is [done], skip it.
                            logger.debug(f"Skipping chunk: {chunk_str}")
                            yield chunk
                            continue
                        choices = chunk_json.get("choices", [])
                        if not choices:
                            yield chunk
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        content = delta.get("content") or message.get("content") or choice.get("text") or ""
                        generated_token += content

                        stop_reason = choice.get("stop_reason")
                        usage = chunk_json.get("usage", {})
                        completion_tokens = (
                            (completion_tokens + 1)
                            if stream_flag
                            else (completion_tokens + usage.get("completion_tokens"))
                        )
                        if stop_reason == "recomputed":
                            retry = True
                            retry_count += 1
                            if chat_flag:
                                messages[0]["content"] = origin_prompt + generated_token
                            else:
                                req_data["prompt"] = origin_prompt + generated_token
                            req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                            tmp_request_length = len(json.dumps(req_data).encode("utf-8"))
                            instance_info = await _handle_select_instance(api, req_data, tmp_request_length)
                            break
                        if retry_count > 0 and not stream_flag:
                            if chat_flag:
                                choice["message"]["content"] = generated_token
                            else:
                                choice["text"] = generated_token
                            chunk = json.dumps(chunk_json).encode("utf-8")
                        yield chunk
            except Exception as e:
                logger.error(
                    f"Error during streaming from decoder {instance_info.decoder.url}: {str(e)} "
                    f"the aborted request {instance_info.request_id} will be routing to the target "
                    "prefiller when new request is ready to dispatch to it"
                )
                proxy_state.abort_prefiller_request(instance_info.prefiller_idx, instance_info.request_id)
                proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)

            # After streaming done, release tokens
            proxy_state.release_decoder(instance_info.decoder_idx, instance_info.decoder_score)

        # Determine the correct media type based on stream flag
        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        proxy_state.request_num -= 1


async def _handle_adjust_instances(adjust_mode: str, request: Request):
    try:
        req_data = await request.json()
        instance_type = req_data.get("type", "")
        instances = req_data.get("instances", [])
        if isinstance(instances, str):
            instances = [instances]
        instances = trans_instances(instances)
        all_msg = f"{adjust_mode} {instance_type} instances: {[str(server) for server in instances]}."

        if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
            return {
                "error": f"Instance type {instance_type} is not supported. "
                f"Only support '{InstanceType.PREFILL}' and '{InstanceType.DECODE}'."
            }

        if adjust_mode == "add":
            added_nodes, waiting_nodes = await proxy_state.add_instances(instance_type, instances)
            if waiting_nodes:
                all_msg = (
                    f"{adjust_mode} {instance_type} instances: {added_nodes}. "
                    f"Instances {waiting_nodes} are waiting to be added."
                )
        elif adjust_mode == "remove":
            if instance_type == InstanceType.PREFILL:
                need_waiting = proxy_state.remove_prefillers(instances)
            else:
                need_waiting = proxy_state.remove_decoders(instances)

            if need_waiting:
                all_msg = f"Instances {instances} are isolated and waiting to be removed."
        return {
            "message": all_msg,
            "current_prefill_instances": [str(prefiller) for prefiller in proxy_state.prefillers],
            "current_decode_instances": [str(decoder) for decoder in proxy_state.decoders],
        }
    except Exception as e:
        logger.error(f"Failed to {adjust_mode} instances: {e}")
        raise e


def trans_instances(instances: list[str]) -> list[ServerState]:
    server_list = []
    for instance in instances:
        h, p = instance.split(":")
        server_list.append(ServerState(h, int(p)))
    return server_list


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders),
    }


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await _handle_adjust_instances("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await _handle_adjust_instances("remove", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
````


## 第二部分：测试相关

任务内路径均相对于任务根目录。下列模块按原快照完整保留；不是因为由 agent 编写就归入辅助工具。真正支撑有效结果的入口为 `run_pinned.sh`，普通预热入口不能替代四 DP 定向预热。

| 文件及全文 | 任务内路径 | 用途 | 运行位置 | 输入/配置 | 输出 | 验证状态 |
|---|---|---|---|---|---|---|
| [aisbench/config.py](#archive-1-0) | `03_verification_scripts/tests/aisbench/config.py` | 工具路径、模型和服务配置 | P 容器 aisbench 目录 | 模型、135:8100、P metrics 地址 | 各测试模块使用的配置 | 有效池化轮使用 |
| [aisbench/run.sh](#archive-1-1) | `03_verification_scripts/tests/aisbench/run.sh` | 原 AISBench 固定负载入口 | P 容器 scripts/aisbench | 131072 输入、1 输出、49 并发、196 条、repeat_rate=0.9、DP4 | 测试日志、CSV、性能输出 | 普通预热轮已运行但 DP 覆盖不足，排除；不作有效轮入口 |
| [aisbench_test.py](#archive-1-2) | `03_verification_scripts/tests/aisbench/aisbench_test.py` | 组织预热、正式请求与指标收集 | P 容器 scripts/aisbench | run.sh/run_pinned.sh 参数、config.py、冻结数据 | aisbench.log、aisbench_result.csv、outputs | 有效池化轮经定向预热适配器调用正式路径 |
| [cal_prefix_hit_rate.py](#archive-1-3) | `03_verification_scripts/tests/aisbench/cal_prefix_hit_rate.py` | 获取并解析 HBM/外部缓存计数 | P 容器，由测试模块导入 | P /metrics、计数类型 | 各 DP 查询与命中计数 | 有效池化轮采样使用 |
| [data_picker.py](#archive-1-4) | `03_verification_scripts/tests/aisbench/data_picker.py` | 从 GSM8K 选择数据并记录已选编号 | P 容器，数据生成模块导入 | JSONL、picked_ids、prefix_flag | 选中记录、编号记录文件 | 数据准备模块；本次复验使用冻结数据，未独立重跑验收 |
| [default_api.py](#archive-1-5) | `03_verification_scripts/tests/aisbench/default_api.py` | AISBench 模型请求配置模板 | P 容器，由测试入口替换占位符 | 模型、地址、并发、速率、输出长度 | 生成的 API 配置 | 有效池化轮测试链使用 |
| [generate_dataset.py](#archive-1-6) | `03_verification_scripts/tests/aisbench/generate_dataset.py` | 生成普通/前缀数据集 | P 容器 scripts/aisbench | tokenizer、GSM8K、长度/数量/重复率/seed | dataset 下的 JSONL | 本次复用冻结数据并核对哈希，不能把重新生成数据视为同一输入 |
| [save_file.py](#archive-1-7) | `03_verification_scripts/tests/aisbench/save_file.py` | 提取性能指标并保存日志/CSV | P 容器，由测试模块导入 | AISBench 输出和场景参数 | aisbench_result.csv、aisbench.log | 有效池化轮正式结果写入；预热不写入吞吐 CSV |
| [analyze_revalidation.py](#archive-1-8) | `03_verification_scripts/tests/analyze_revalidation.py` | 逐轮有效性检查及完整六轮统计 | 135 宿主机；历史部署 /tmp/dts_zyd_analyze_revalidation.py | 轮次目录；或 --summary 与运行根目录 | derived/result.json；完整模式 summary.json | 有效池化轮已分析；六轮汇总未执行 |
| [pin_prefix_warmup.py](#archive-1-9) | `03_verification_scripts/tests/pin_prefix_warmup.py` | 按请求头对 P DP0～3 定向预热并完成 P→D 交换 | P 容器，由适配器导入 | 4 条相同冻结前缀、dp=4、DTS_PIN_PREFIX_RESULT | pinned_warmup.json，真实 P/D 响应与 usage | 有效池化轮运行，四 DP 覆盖已核对 |
| [render_revalidation_report.py](#archive-1-10) | `03_verification_scripts/tests/render_revalidation_report.py` | 根据已校验结果生成轮次/运行报告 | 可读取运行目录的 Python 环境；历史具体调用位置未单独记录 | 运行根目录，--partial 或完整模式 | README.md、derived/cache.json、summary.json | 阶段性报告已留存；完整六轮结论模式未执行 |
| [run_pinned.sh](#archive-1-11) | `03_verification_scripts/tests/run_pinned.sh` | 本次有效轮正式测试入口 | P 容器；切到 scripts/aisbench 执行 | 固定负载参数、DTS_PIN_PREFIX_RESULT | 预热响应、正式日志与 CSV | 有效池化轮实际运行；无池化正式轮未执行 |
| [run_pinned_aisbench.py](#archive-1-12) | `03_verification_scripts/tests/run_pinned_aisbench.py` | 在内存中替换预热步骤，保留原正式测试路径 | P 容器 revalidation_tools，由 run_pinned.sh 调用 | 原 aisbench_test.py；目标代码块必须各匹配一次 | 定向预热及原正式测试执行结果，不改磁盘原代码 | 有效池化轮实际运行 |

输入冻结哈希、有效性证据及执行命令见 [04](04_verification_env_and_launch.md) 与 [05](05_verification_process_and_results.md)。这些状态仅描述 2026-10-13 归档范围；本次文档整理未执行远程服务或测试。

<a id="archive-1-0"></a>
### aisbench/config.py

````python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_PATH = str(ROOT / 'dataset')
OUTPUT_DIR = str(ROOT / 'outputs')
WORK_PATH = '/usr/local/python3.11.10/lib/python3.11/site-packages'
MODEL_NAME = 'model'
MODEL_PATH = '/mnt/share/weights/DeepSeek-V4-Flash-0731-w8a8'
HOST_IP = '80.5.9.135'
HOST_PORT = '8100'
POD_INFO = ['80.5.9.135:8910']
URL = ''
API_KEY = ''
DEFAULT_PERFORMANCE_TEST = 'default_perf'
````


<a id="archive-1-1"></a>
### aisbench/run.sh

````bash
#!/usr/bin/env bash
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python3 -u aisbench_test.py \
    --input_len 131072 --output_len 1 \
    --concurrency 49 --data_num 196 \
    --dataset_type prefix_cache --repeat_rate 0.9 --prefix_num 1 \
    --prefix_test --dp 4 --npu_num 32 \
    --request_rate 0 --seed 1
````


<a id="archive-1-2"></a>
### aisbench/aisbench_test.py

````python
import os, errno
import sys
import argparse
import re
import logging
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import *
from generate_dataset import *
from save_file import get_data, save_csv, save_log
from cal_prefix_hit_rate import *
logging.getLogger().setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=3500, help="input token length")
    parser.add_argument("--output_len", type=str, default="1500", help="output token length")
    parser.add_argument("--data_num", type=int, default=8192, help="dataset number")
    parser.add_argument("--concurrency", type=str, default="2048", help="max concurrency")
    parser.add_argument("--request_rate", type=str, default="0", help="request rate")
    parser.add_argument("--test_type", type=str, default="stream", help="text or stream")
    parser.add_argument("--dataset", type=str, default="none", help="dataset path")
    parser.add_argument("--repeat", type=int, default=1, help="number of test repeat times")
    parser.add_argument("--enable_think", action='store_true', default=False, help="enable thinking for ds v3.1")
    parser.add_argument("--test_accuracy", action='store_true', default=False, help="test accuracy")
    parser.add_argument("--npu_num", type=int, default=1, help="npu numbers")
    parser.add_argument("--dataset_type", type=str, default="normal", help="normal or prefix_cache")
    parser.add_argument("--prefix_num", type=int, default=1, help="prefix numbers")
    parser.add_argument("--repeat_rate", type=str, default="0", help="dataset repeat rate")
    parser.add_argument("--prefix_test", action='store_true', default=False, help="test prefix dataset firstly")
    parser.add_argument("--seed", type=int, default=1, help="dataset random seed")
    parser.add_argument("--dp", type=int, default=1, help="dp size")
    parser.add_argument("--length_mean", type=int, default=None, help="gaussian mean for variable length")
    parser.add_argument("--length_std", type=float, default=None, help="gaussian std for variable length")
    parser.add_argument("--length_min", type=int, default=None, help="min length for uniform range or gaussian clip")
    parser.add_argument("--length_max", type=int, default=None, help="max length for uniform range or gaussian clip")
    return parser.parse_args()

def symlink_force(target, link_name):
    logging.info(f"make symlink: {link_name} ==> {target}")
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def create_gsm8k_dataset(dataset_type, input_len, data_num, model_path, dataset_path, prefix_num, repeat_rate, seed,
                         length_mean=None, length_std=None, length_min=None, length_max=None):
    if not os.path.exists(dataset_path):
        logging.warning(f"dataset work path {dataset_path} not exist. creating it automatically.")
        os.makedirs(dataset_path, exist_ok=True)
        logging.info(f"dataset work path {dataset_path} created.")

    base_name = os.path.basename(os.path.normpath(model_path))
    if dataset_type == "prefix_cache":
        _plen = int(input_len * repeat_rate)
        _pp = os.path.join(dataset_path, f'prefix-GSM8K-in{_plen}-num{dp*prefix_num}-{base_name}.jsonl')
        _dp = os.path.join(dataset_path, f'GSM8K-in{input_len}-num{data_num}-{base_name}-repeatRate{repeat_rate}.jsonl')
        if os.path.exists(_pp) and os.path.exists(_dp):
            logging.info(f"[复用] 复用已存在数据集，跳过生成: {_dp}")
            return _pp, _dp
        prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,1,dp,repeat_rate,seed,prefix_num,
                                                                             length_mean, length_std, length_min, length_max)
        logging.info("[完成] 数据集已生成：")
        logging.info(f"  - 公共前缀：{prefix_jsonl_path}  (行数={dp*prefix_num})")
        logging.info(f"  - 数据集：  {dataset_jsonl_path} (行数={data_num})")
        logging.info("[信息] 配置：")
        logging.info(f"  tokens(单条长度)={input_len}, prefix_ratio(前缀重复率)={repeat_rate}")
        if length_mean is not None and length_std is not None:
            logging.info(f"  length_dist=gaussian(mean={length_mean}, std={length_std}, min={length_min}, max={length_max})")
        elif length_min is not None and length_max is not None:
            logging.info(f"  length_dist=uniform_int([{length_min}, {length_max}])")
        else:
            logging.info("  length_dist=fixed")
    else:
        dataset_name = "GSM8K-in" + str(input_len) + "-num" + str(data_num) + "-" + base_name + ".jsonl"
        logging.info(f"dataset_name: {dataset_name}")
        dataset_jsonl_path = os.path.join(dataset_path, dataset_name)
        prefix_jsonl_path = ""
        # 判断数据集是否存在
        if not os.path.exists(dataset_jsonl_path):
            logging.warning(f"Dataset {dataset_name} is not exist. Start create dataset")
            # create_data(input_len, data_num, model_path, dataset_path)
            prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,0,dp,0,seed,prefix_num,
                                                                                 length_mean, length_std, length_min, length_max)
            logging.info(f"Dataset {dataset_name} created.")
        else:
            logging.info(f"Dataset {dataset_name} exist.")
    return prefix_jsonl_path, dataset_jsonl_path

def generate_aisbench_command(DEFAULT_PERFORMANCE_TEST):
    if test_accuracy:
        ais_bench_cmd = f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --work-dir {OUTPUT_DIR} --dump-eval-details"
    else:
        ais_bench_cmd = f"ais_bench --models vllm_api_chat_temp --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --summarizer {DEFAULT_PERFORMANCE_TEST} --work-dir {OUTPUT_DIR} --debug --num-warmups 0 2>&1 | tee -a aisbench.log | tee -a aisbench_all.log"
    return ais_bench_cmd

def generate_test_dataset(src_file, dst_dir):
    dst_file = os.path.join(dst_dir, "test.jsonl")
    logging.info(f"src_file: {src_file}")
    logging.info(f"dst_file: {dst_file}")
    # 使用软连接
    symlink_force(src_file, dst_file)
    return

def save_result(request_rate, npu_num, repeat_rate):
    aisbench_log_dir = "aisbench.log"
    filename = "aisbench_result.csv"
    ans, log_dir=get_data(aisbench_log_dir,request_rate,npu_num,repeat_rate)
    save_log(aisbench_log_dir, log_dir)
    save_csv(ans, filename)

def modify_aisbench_api(concurrency, output_len, request_rate_param=None):
    if request_rate_param is None:
        request_rate_param = request_rate
    file_default = open("default_api.py", 'r+')
    file_temp = open("temp_api.py", "w+")
    logging.info("Api config file:")
    for ss in file_default.readlines():
        tt = re.sub("model_path_for_replace", MODEL_PATH, ss)
        tt = re.sub("model_name_for_replace", MODEL_NAME, tt)
        tt = re.sub("api_key_for_replace", API_KEY, tt)
        tt = re.sub("rr_for_replace", request_rate_param, tt)
        tt = re.sub("test_type_for_replace", api_test_type, tt)
        tt = re.sub("test_abbr_for_replace", api_test_abbr, tt)
        tt = re.sub("ip_for_replace", HOST_IP, tt)
        tt = re.sub("port_for_replace", HOST_PORT, tt)
        tt = re.sub("url_for_replace", URL, tt)
        tt = re.sub("outputlen_for_replace", output_len, tt)
        tt = re.sub("concurrency_for_replace", concurrency, tt)
        if test_accuracy:
            generation_kwargs = "temperature=0.6,\n\t\t\ttop_p = 0.95"
        else:
            generation_kwargs = "temperature=0,\n\t\t\tignore_eos=True"
        if enable_think:
            generation_kwargs = generation_kwargs + ",\n\t\t\tchat_template_kwargs={\"enable_thinking\": True}"
        tt = re.sub("generation_kwargs_for_replace", generation_kwargs.expandtabs(4), tt)
        print(tt, end='')
        file_temp.write(tt)
    file_default.close()
    file_temp.close()
    symlink_force(
        os.path.join(os.getcwd(), "temp_api.py"),
        os.path.join(WORK_PATH, "ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py")
    )

def _fetch_pod_metrics(pod):
    if pod.startswith('['):
        # IPv6 格式: [ip]:port
        ip = pod.split(']:')[0][1:]
        port = pod.split(']:')[1]
    elif pod.count(':') > 1:
        # IPv6 格式无方括号: ip:port
        ip, port = pod.rsplit(':', 1)
    else:
        # IPv4 格式: ip:port
        ip, port = pod.split(":")
    query_token, query_external = get_prefix_queries_total(ip, port)
    hit_token, hit_external = get_prefix_hits_total(ip, port)
    return pod, query_token, query_external, hit_token, hit_external

def get_pod_metrics_info(pod_info):
    query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = {}, {}, {}, {}
    with ThreadPoolExecutor(max_workers=len(pod_info)) as executor:
        futures = [executor.submit(_fetch_pod_metrics, pod) for pod in pod_info]
        for future in as_completed(futures):
            pod, q_n, q_e, h_n, h_e = future.result()
            query_tokens[pod] = q_n
            query_tokens_external[pod] = q_e
            hit_tokens[pod] = h_n
            hit_tokens_external[pod] = h_e
    return query_tokens, query_tokens_external, hit_tokens, hit_tokens_external

def cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external, query_tokens_new,
                        query_tokens_external_new, hit_tokens_new, hit_tokens_external_new):
    if not query_tokens:
        logging.warning("No prefix cache metrics found.")
        return

    # 输出格式
    line_width = 118
    scope_width, hbm_rate_width, hbm_detail_width, external_rate_width = 35, 19, 21, 21
    headers = ["scope", "hbm_hit_rate", "hbm(hit/query)", "external_hit_rate", "external(hit/query)"]
    header = (f"{headers[0]:<{scope_width}}{headers[1]:<{hbm_rate_width}}"
              f"{headers[2]:<{hbm_detail_width}}{headers[3]:<{external_rate_width}}{headers[4]}")

    def format_rate(hits, queries):
        return "0.00%" if queries == 0 else format(hits / queries, ".2%")

    def build_row(scope, hbm_hits, hbm_queries, external_hits, external_queries):
        return (f"{scope:<{scope_width}}{format_rate(hbm_hits, hbm_queries):<{hbm_rate_width}}"
                f"{f'{hbm_hits}/{hbm_queries}':<{hbm_detail_width}}"
                f"{format_rate(external_hits, external_queries):<{external_rate_width}}"
                f"{external_hits}/{external_queries}")

    total_hbm_queries = total_hbm_hits = total_external_queries = total_external_hits = 0
    pod_sections, skipped_pods = [], []

    for pod, engines in sorted(query_tokens.items()):
        # 八类指标必须全部存在，且 engine_id 完全一致；
        # 否则无法可靠计算"测试后累计值 - 测试前累计值"，跳过该 POD
        metrics = [query_tokens.get(pod), query_tokens_external.get(pod), hit_tokens.get(pod),
                   hit_tokens_external.get(pod), query_tokens_new.get(pod), query_tokens_external_new.get(pod),
                   hit_tokens_new.get(pod), hit_tokens_external_new.get(pod)]

        if not all(metrics):
            skipped_pods.append(pod)
            continue

        engine_ids = set(engines)
        if any(set(metric) != engine_ids for metric in metrics[1:]):
            skipped_pods.append(pod)
            continue

        # 按 engine_id 升序遍历，为每个 engine 单独生成一行
        engine_rows = []
        for engine_id in sorted(engine_ids):
            engine_scope = f"{pod}/engine{engine_id}"
            engine_hbm_queries = query_tokens_new[pod][engine_id] - query_tokens[pod][engine_id]
            engine_hbm_hits = hit_tokens_new[pod][engine_id] - hit_tokens[pod][engine_id]
            engine_external_queries = query_tokens_external_new[pod][engine_id] \
                - query_tokens_external[pod][engine_id]
            engine_external_hits = hit_tokens_external_new[pod][engine_id] \
                - hit_tokens_external[pod][engine_id]
            engine_rows.append(build_row(engine_scope, engine_hbm_hits, engine_hbm_queries,
                                         engine_external_hits, engine_external_queries))

        # 合并当前 POD 下所有 engine_id 的指标增量，用于 ALL_PODS 汇总
        pod_hbm_queries = sum(query_tokens_new[pod][engine_id] - query_tokens[pod][engine_id]
                              for engine_id in engine_ids)
        pod_hbm_hits = sum(hit_tokens_new[pod][engine_id] - hit_tokens[pod][engine_id]
                           for engine_id in engine_ids)
        pod_external_queries = sum(query_tokens_external_new[pod][engine_id]
                                   - query_tokens_external[pod][engine_id] for engine_id in engine_ids)
        pod_external_hits = sum(hit_tokens_external_new[pod][engine_id]
                                - hit_tokens_external[pod][engine_id] for engine_id in engine_ids)

        # 每个 POD 按 engine 升序输出，POD 之间用分隔线隔开
        pod_sections.append(engine_rows)
        total_hbm_queries += pod_hbm_queries
        total_hbm_hits += pod_hbm_hits
        total_external_queries += pod_external_queries
        total_external_hits += pod_external_hits

    if not pod_sections:
        logging.warning(f"No valid prefix cache metrics found. Skipped PODs: {', '.join(skipped_pods)}")
        return

    all_pods_row = build_row("ALL_PODS", total_hbm_hits, total_hbm_queries,
                             total_external_hits, total_external_queries)
    skipped_pods_line = f"Skipped PODs: {', '.join(skipped_pods)}" if skipped_pods else None

    # 控制台只输出所有有效 POD 的汇总
    console_lines = ["=" * line_width, "Prefix cache hit summary", "=" * line_width, header,
                     "-" * line_width, all_pods_row, "=" * line_width]

    # 日志输出每个 POD 下所有 engine 的详细信息，POD 之间用分隔线隔开
    log_lines = ["=" * line_width, "Prefix cache hit detail", "=" * line_width, header,
                 "-" * line_width]
    for idx, section in enumerate(pod_sections):
        if idx > 0:
            log_lines.append("-" * line_width)
        log_lines.extend(section)
    log_lines.extend(["-" * line_width, all_pods_row, "=" * line_width])

    if skipped_pods_line:
        console_lines.append(skipped_pods_line)
        log_lines.append(skipped_pods_line)

    print("\n".join(console_lines))

    try:
        with open("aisbench.log", "a", encoding="utf-8") as log_file:
            log_file.write("\n" + "\n".join(log_lines) + "\n")
        with open("aisbench_all.log", "a", encoding="utf-8") as all_log_file:
            all_log_file.write("\n" + "\n".join(log_lines) + "\n")
    except OSError as error:
        logging.error(f"Failed to write prefix cache information to log files: {error}")

def setup_logging():
    """配置 Python logging：同时输出到控制台、aisbench.log 和 aisbench_all.log"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清除已有 handlers（避免重复），但保留会输出到控制台的 StreamHandler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 写入 aisbench.log（本次运行日志）
    file_handler = logging.FileHandler("aisbench.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 写入 aisbench_all.log（所有运行累积日志）
    all_file_handler = logging.FileHandler("aisbench_all.log", mode="a", encoding="utf-8")
    all_file_handler.setFormatter(formatter)
    logger.addHandler(all_file_handler)

if __name__ == '__main__':
    args = parse_arguments()
    input_len = args.input_len
    output_len = args.output_len
    data_num = args.data_num
    concurrency = args.concurrency
    request_rate = args.request_rate
    test_type = args.test_type
    dataset_path_input = args.dataset
    test_times = args.repeat
    enable_think = args.enable_think
    test_accuracy = args.test_accuracy
    npu_num = args.npu_num
    prefix_num = args.prefix_num
    repeat_rate = parse_prefix_ratio(args.repeat_rate)
    prefix_test = args.prefix_test
    dataset_type = args.dataset_type
    seed = args.seed
    dp = args.dp
    length_mean = args.length_mean
    length_std = args.length_std
    length_min = args.length_min
    length_max = args.length_max

    # 初始化日志文件：清空 aisbench.log（新运行覆盖），aisbench_all.log 添加分隔符（续写）
    with open("aisbench.log", "w", encoding="utf-8") as f:
        f.write("")
    with open("aisbench_all.log", "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*50}\n")
        f.write(f"New run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")

    # 配置日志输出到控制台 + aisbench.log + aisbench_all.log
    setup_logging()
    logging.info("aisbench.log has been cleared for this run.")
    logging.info("aisbench_all.log separator added for this run.")

    # 变长参数校验
    if (length_mean is None) ^ (length_std is None):
        raise ValueError("length_mean 和 length_std 必须同时提供或同时不提供")
    if (length_min is None) ^ (length_max is None):
        raise ValueError("length_min 和 length_max 必须同时提供或同时不提供")
    if length_mean is not None and length_mean < 1:
        raise ValueError("length_mean 必须 >= 1")
    if length_std is not None and length_std < 0:
        raise ValueError("length_std 必须 >= 0")
    if length_min is not None and length_min < 1:
        raise ValueError("length_min 必须 >= 1")
    if length_max is not None and length_max < 1:
        raise ValueError("length_max 必须 >= 1")

    logging.info(f"input token length: {input_len}")
    logging.info(f"output token length: {output_len}")
    logging.info(f"number of dataset: {data_num}")
    logging.info(f"concurrency: {concurrency}")
    logging.info(f"request rate: {request_rate}")
    logging.info(f"test type: {test_type}")
    logging.info(f"test_times: {test_times}")
    logging.info(f"v3.1 enable_think: {enable_think}")
    logging.info(f"accuracy test: {test_accuracy}")
    logging.info(f"npu numbers: {npu_num}")
    logging.info(f"prefix numbers: {prefix_num}")
    logging.info(f"dataset repeat rate: {repeat_rate}")
    logging.info(f"test prefix dataset: {prefix_test}")
    logging.info(f"dataset type: {dataset_type}")
    logging.info(f"seed: {seed}")
    logging.info(f"dp size: {dp}")
    logging.info(f"length_mean: {length_mean}")
    logging.info(f"length_std: {length_std}")
    logging.info(f"length_min: {length_min}")
    logging.info(f"length_max: {length_max}")

    # 区分流式和非流式
    if test_type == "text":
        api_test_type = "VLLMCustomAPIChat"
        api_test_abbr = "vllm-api-general-chat"
    elif test_type == "stream":
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = "vllm-api-stream-chat"
    else:
        api_test_type = "VLLMCustomAPIChatStream"
        api_test_abbr = "vllm-api-stream-chat"

    if dataset_path_input == "none":
        src_file_prefix, src_file_data = create_gsm8k_dataset(
            dataset_type, input_len, data_num, MODEL_PATH, DATASET_PATH,
            prefix_num, repeat_rate, seed,
            length_mean, length_std, length_min, length_max)

    else:
        # 指定数据集路径逻辑
        if not os.path.exists(dataset_path_input):
            logging.error(f"Dataset {dataset_path_input} is not exist.")
            sys.exit(1)
        src_file_data = dataset_path_input
        src_file_prefix = ""

    dst_dir = os.path.join(WORK_PATH, "ais_bench/datasets/gsm8k")

    # 判断 aisbench 的 gsm8k 文件夹是否存在在
    if not os.path.exists(dst_dir):
        logging.info("dataset work path not exist. creating.")
        os.makedirs(dst_dir)
        logging.info("dataset work path created.")
    # 判断 aisbench 的 gsm8k 文件夹是否存在在 train.jsonl 文件
    train_dataset = os.path.join(dst_dir, "train.jsonl")
    if not os.path.exists(train_dataset):
        logging.info("train dataset not exist. creating.")
        file = open(train_dataset, 'w')
        file.close()
        logging.info("train dataset created.")    
    
    # 生成 aisbench 命令
    ais_bench_cmd = generate_aisbench_command(DEFAULT_PERFORMANCE_TEST)
    logging.info(f"test start, use command: {ais_bench_cmd}")
    
    # 执行命令    
    if dataset_type == "prefix_cache":
        # 前缀数据集测试
        if prefix_test:
            if not POD_INFO:
                pod_info = [HOST_IP+":"+HOST_PORT]
            else:
                pod_info = POD_INFO
            logging.info(f"pod_info: {pod_info}")
            
            logging.info(f"[开始] 前缀数据集测试")
            # 前缀测试: request rate 固定为 0
            modify_aisbench_api(str(dp), "1", request_rate_param="0")
            dst_file = generate_test_dataset(src_file_prefix, dst_dir)

            # 命中率计算
            query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)

            subprocess.run(ais_bench_cmd, shell=True, check=True)
            logging.info(f"[完成] 前缀数据集测试完成，结果保存在aisbench_result.csv")

            query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)
            cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external,query_tokens_new,
                                query_tokens_external_new, hit_tokens_new, hit_tokens_external_new)
            
            # 保存前缀测试结果
            save_result(0, npu_num, repeat_rate)

            logging.info(f"[开始] 全量数据集测试")
            # 命中率计算
            query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)
            
            modify_aisbench_api(concurrency,str(output_len))
            dst_file = generate_test_dataset(src_file_data, dst_dir)
            # 执行测试命令
            subprocess.run(ais_bench_cmd, shell=True, check=True)
            logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv")
            
            query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)
            cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external, query_tokens_new,
                                query_tokens_external_new, hit_tokens_new, hit_tokens_external_new)
            
        else:
            logging.info(f"[开始] 全量数据集测试")
            modify_aisbench_api(concurrency,str(output_len))
            dst_file = generate_test_dataset(src_file_data, dst_dir)
            subprocess.run(ais_bench_cmd, shell=True, check=True)
            logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv")

    else:
        logging.info(f"[开始] 全量数据集测试")
        modify_aisbench_api(concurrency,str(output_len))
        dst_file = generate_test_dataset(src_file_data, dst_dir)
        if test_times > 1:
            for test_time in range(test_times):
                logging.info(f"Execution rounds: {test_time + 1}")
                subprocess.run(ais_bench_cmd, shell=True, check=True)
        else:
            subprocess.run(ais_bench_cmd, shell=True, check=True)
        logging.info(f"[完成] 全量数据集测试完成，结果保存在aisbench_result.csv")

    
    # 保存结果
    save_result(request_rate, npu_num, repeat_rate)
````


<a id="archive-1-3"></a>
### aisbench/cal_prefix_hit_rate.py

````python
import logging
import re
import requests

logging.getLogger().setLevel(logging.INFO)


def _fetch_metrics(ip_address, port):
    """获取 /metrics 接口文本"""
    url = f"http://{ip_address}:{port}/metrics"
    resp = requests.get(url, proxies={"http": None, "https": None}, timeout=30)
    resp.raise_for_status()
    return resp.text


def _get_prefix_cache_metrics(ip_address, port, metric_type):
    """
    通用的前缀缓存指标获取函数

    Args:
        ip_address: IP 地址
        port: 端口号
        metric_type: 指标类型，'queries' 或 'hits'

    Returns:
        tuple: (normal_stats, external_stats) 两个字典，key 为 engine_id(int)，value 为指标值
    """
    try:
        text = _fetch_metrics(ip_address, port)
        lines = [l for l in text.split('\n')
                 if 'model_name' in l and f'prefix_cache_{metric_type}_total' in l]
        normal_stats = {}
        external_stats = {}

        for line in lines:
            # 提取engine值
            engine_match = re.search(r'engine="(\d+)"', line)
            if not engine_match:
                continue

            engine = engine_match.group(1)

            # 提取最后一个数字
            parts = line.split()
            if len(parts) < 2:
                continue

            value_str = parts[-1]
            try:
                value = float(value_str)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                continue

            # 根据指标名称分类
            if f'external_prefix_cache_{metric_type}_total' in line:
                external_stats[int(engine)] = value
            elif f'vllm:prefix_cache_{metric_type}_total' in line:
                normal_stats[int(engine)] = value

        logging.info(f"{metric_type}: normal={normal_stats}, external={external_stats}")
        return normal_stats, external_stats
    except Exception as e:
        logging.error(f"获取{metric_type}指标失败: {e}")
        return {}, {}


def get_prefix_queries_total(ip_address, port):
    """
    获取查询token总数,返回{engine:tokens}
    """
    return _get_prefix_cache_metrics(ip_address, port, 'queries')


def get_prefix_hits_total(ip_address, port):
    """
    获取命中token总数,返回{engine:tokens}
    """
    return _get_prefix_cache_metrics(ip_address, port, 'hits')
````


<a id="archive-1-4"></a>
### aisbench/data_picker.py

````python
import json
import random
import os


class DataPicker:
    def __init__(self, jsonl_file, record_file="picked_ids.txt", prefix_flag=0):
        """
        初始化挑选器

        Args:
            jsonl_file: GSM8K数据集文件路径
            record_file: 记录已挑选数据编号的文件路径
        """
        self.jsonl_file = jsonl_file
        self.record_file = record_file
        self.total_lines = self._count_lines()
        self.picked_ids = self._load_picked_ids()
        self.prefix_flag = prefix_flag

    def _count_lines(self):
        """统计jsonl文件总行数"""
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _load_picked_ids(self):
        """加载已挑选的编号记录"""
        if os.path.exists(self.record_file):
            with open(self.record_file, 'r', encoding='utf-8') as f:
                return set(int(line.strip()) for line in f if line.strip())
        return set()

    def _save_picked_ids(self):
        """保存已挑选的编号记录"""
        with open(self.record_file, 'w', encoding='utf-8') as f:
            for pid in sorted(self.picked_ids):
                f.write(f"{pid}\n")

    def get_unpicked_ids(self):
        """获取所有未被挑选过的编号"""
        all_ids = set(range(self.total_lines))
        return list(all_ids - self.picked_ids)

    def pick_one(self):
        """
        随机挑选一条数据

        Returns:
            挑选的数据内容，如果没有未挑选的数据则返回None
        """

        # 获取可选的编号列表
        if self.prefix_flag == 1:
            # 不重复模式：从未挑选的编号中选择
            available_ids = self.get_unpicked_ids()
            if not available_ids:
                return None
            # 随机选择一个未挑选的编号
            selected_id = random.choice(available_ids)
            # 记录已挑选的编号
            self.picked_ids.add(selected_id)
            self._save_picked_ids()
        else:
            # 可重复模式：从所有编号中随机选择，不记录编号
            selected_id = random.randint(0, self.total_lines - 1)

        # 读取对应编号的数据
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == selected_id:
                    data = json.loads(line)
                    break

        return data['question']

    # def reset(self):
    #     """重置记录（清空已挑选记录）"""
    #     self.picked_ids.clear()
    #     self._save_picked_ids()
````


<a id="archive-1-5"></a>
### aisbench/default_api.py

````python
from ais_bench.benchmark.models import test_type_for_replace

models = [
    dict(
        attr="service",
        type=test_type_for_replace,
        abbr='test_abbr_for_replace',
        path="model_path_for_replace",
        model="model_name_for_replace",
        api_key="api_key_for_replace",
        request_rate=rr_for_replace,
        retry=2,
        host_ip="ip_for_replace",
        host_port=port_for_replace,
        url="url_for_replace",
        max_out_len=outputlen_for_replace,
        batch_size=concurrency_for_replace,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True,
        )
    )
]
````


<a id="archive-1-6"></a>
### aisbench/generate_dataset.py

````python
import logging
from transformers import AutoTokenizer
from data_picker import *
import os
import random
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dataset(tokenizer_path: str, input_len: int, number: int, prefix_flag):

    logging.info(f"加载tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    output_samples = []
    attempts = 0
    max_attempts = number * 10  # 防止意外无限循环

    pbar = tqdm(total=number, desc="Generating dataset", unit="row") if tqdm else None
    while len(output_samples) < number and attempts < max_attempts:
        attempts += 1
        # 随机选择一条文本
        picker = DataPicker("./GSM8K.jsonl", "./picked_ids.txt", prefix_flag)
        raw_text = picker.pick_one()
        if raw_text is None:
            break

        # tokenize
        tokens = tokenizer.encode(raw_text, add_special_tokens=False)

        if len(tokens) == 0:
            # logging.info(f"生成数据集失败，请清空pick ids")
            break

        # 根据需求调整长度：重复或截断
        if len(tokens) >= input_len:
            # 截断到 input_len
            adjusted_tokens = tokens[:input_len]
        else:
            # 重复整个序列直到达到 input_len
            repeat_times = (input_len + len(tokens) - 1) // len(tokens)
            repeated_tokens = tokens * repeat_times
            adjusted_tokens = repeated_tokens[:input_len]

        # 解码回文本
        adjusted_text = tokenizer.decode(adjusted_tokens, skip_special_tokens=True)
        final_len = len(tokenizer.encode(adjusted_text, add_special_tokens=False))
        if final_len != input_len:
            corrected_tokens = tokenizer.encode(adjusted_text, add_special_tokens=False)
            if len(corrected_tokens) >= input_len:
                corrected_tokens = corrected_tokens[:input_len]
            else:
                corrected_tokens = (corrected_tokens * ((input_len // len(corrected_tokens)) + 1))[:input_len]
            adjusted_text = tokenizer.decode(corrected_tokens, skip_special_tokens=True)


        output_samples.append(adjusted_text)
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    if len(output_samples) < number:
        return None

    return output_samples


def generate_unique_tokens(tokenizer_path, seed, n, number):
    """
    根据模型 tokenizer 和随机种子，生成 n 个不相同的 token，共 number 行数据

    Args:
        model_name_or_tokenizer: 模型名称或已加载的 tokenizer 对象
        seed: 随机种子
        n: 每行需要生成的 token 数量
        number: 需要生成的数据行数

    Returns:
        list: 包含 number 行数据的列表
    """
    # 设置随机种子
    random.seed(seed)
    torch.manual_seed(seed)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 获取词表大小
    vocab_size = len(tokenizer)

    if n > vocab_size:
        raise ValueError(f"每行请求的 token 数量 {n} 超过词表大小 {vocab_size}")

    all_lines = []
    pbar = tqdm(total=number, desc="Generating unique tokens", unit="row") if tqdm else None

    for line_idx in range(number):
        if pbar:
            pbar.update(1)
        # 为每一行使用不同的种子，确保行间数据不重复
        line_seed = seed + line_idx
        random.seed(line_seed)
        torch.manual_seed(line_seed)

        unique_tokens = []
        seen_tokens = set()
        max_attempts = n * 10  # 防止无限循环
        attempts = 0

        while len(unique_tokens) < n and attempts < max_attempts:
            # 随机生成 token ID
            token_id = random.randint(0, vocab_size - 1)

            # 检查是否重复
            if token_id in seen_tokens:
                attempts += 1
                continue

            # 转换为文本
            try:
                token_text = tokenizer.decode([token_id])

                # 可选：跳过特殊 token 或空 token
                # if token_text.strip() or token_text:  # 保留非空 token
                unique_tokens.append(token_text)
                seen_tokens.add(token_id)
            except:
                pass

            attempts += 1

        if len(unique_tokens) < n:
            print(f"警告：第 {line_idx + 1} 行只生成了 {len(unique_tokens)} 个唯一 token")
        # combined_text = ''.join(unique_tokens)
        all_lines.append(''.join(unique_tokens))

    if pbar:
        pbar.close()
    return all_lines

def write_data(path, dataset, num):
    if num is not None:
        if len(dataset) < num:
            # 重复数据
            repeats = num // len(dataset)
            remainder = num % len(dataset)
            dataset = dataset * repeats + dataset[:remainder]
        else:
            # 截取数据
            dataset = dataset[:num]
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps({"question": item, "answer": "none"}, ensure_ascii=False))
            f.write("\n")

def sample_target_length(rng, fixed_length, length_mean=None, length_std=None, length_min=None, length_max=None):
    """从高斯或均匀分布中采样目标长度"""
    fixed_length = max(1, int(fixed_length))
    has_gauss = (length_mean is not None) and (length_std is not None)
    has_range = (length_min is not None) and (length_max is not None)

    lo = 1 if length_min is None else max(1, int(length_min))
    hi = None if length_max is None else max(1, int(length_max))
    if hi is not None and lo > hi:
        lo, hi = hi, lo

    if has_gauss:
        mu = max(1, int(length_mean))
        sigma = max(0.0, float(length_std))
        val = mu if sigma == 0 else int(round(rng.gauss(mu, sigma)))
        if hi is not None:
            val = min(val, hi)
        val = max(lo, val)
        return max(1, val)

    if has_range:
        return rng.randint(lo, hi)

    return fixed_length


def _build_length_tag(input_len, length_mean, length_std, length_min, length_max):
    if (length_mean is not None) and (length_std is not None):
        tag = f"G{int(length_mean)}_{str(length_std).replace('.', 'd')}"
        if (length_min is not None) and (length_max is not None):
            tag += f"_C{int(length_min)}_{int(length_max)}"
        return tag
    if (length_min is not None) and (length_max is not None):
        return f"U{int(length_min)}_{int(length_max)}"
    return f"L{int(input_len)}"


def _truncate_or_pad_text(tokenizer, text, target_len):
    """将文本的 token 长度调整到 target_len（截断或重复填充）"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) >= target_len:
        tokens = tokens[:target_len]
    else:
        repeat_times = (target_len + len(tokens) - 1) // len(tokens)
        tokens = (tokens * repeat_times)[:target_len]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def create_multi_prefix_dataset(tokenizer_path: str, input_len: int, number: int, save_path, prefix_flag, dp, repeat_rate, seed, prefix_num,
                                length_mean=None, length_std=None, length_min=None, length_max=None):
    base_name = os.path.basename(os.path.normpath(tokenizer_path))
    use_variable_length = (
        (length_mean is not None and length_std is not None)
        or (length_min is not None and length_max is not None)
    )

    # ========== 普通数据集（无前缀）==========
    if prefix_flag == 0:
        if use_variable_length:
            rng = random.Random(seed)
            real_lens = [max(1, int(sample_target_length(rng, input_len, length_mean, length_std, length_min, length_max))) for _ in range(number)]
            max_len = max(real_lens)
            # 先生成统一最大长度的文本池，再逐条截断
            long_texts = create_dataset(tokenizer_path, max_len, number, 0)
            if long_texts is None:
                logging.error("生成数据集失败，请清空picked ids")
                exit(0)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            dataset = []
            pbar = tqdm(total=number, desc="Truncating to variable lengths", unit="row") if tqdm else None
            for i, rl in enumerate(real_lens):
                adjusted = _truncate_or_pad_text(tokenizer, long_texts[i], rl)
                dataset.append(adjusted)
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()
            length_tag = _build_length_tag(input_len, length_mean, length_std, length_min, length_max)
            dataset_path = os.path.join(save_path, f'GSM8K-{length_tag}-num{number}-{base_name}.jsonl')
            write_data(dataset_path, dataset, number)
            return "", dataset_path
        else:
            dataset = create_dataset(tokenizer_path, input_len, number, 0)
            dataset_path = os.path.join(save_path, f'GSM8K-in{input_len}-num{number}-{base_name}.jsonl')
            write_data(dataset_path, dataset, number)
            return "", dataset_path

    # ========== 前缀数据集 ==========
    if use_variable_length:
        return _create_prefix_dataset_variable(tokenizer_path, input_len, number, save_path, base_name, dp, repeat_rate, seed, prefix_num,
                                               length_mean, length_std, length_min, length_max)

    # -------- 定长前缀数据集（原有逻辑）--------
    prefix_len = int(input_len * repeat_rate)
    prefix_data = []
    prefix_data = create_dataset(tokenizer_path, prefix_len, prefix_num, 1)
    if prefix_data == None and repeat_rate > 0:
        logging.error(f"生成数据集失败，请清空picked ids")
        exit(0)

    prefix_dataset = []
    for i in range(prefix_num):
        for j in range(dp):
            prefix_dataset.append(prefix_data[i])

    prefix_path = os.path.join(save_path, f'prefix-GSM8K-in{prefix_len}-num{dp*prefix_num}-{base_name}.jsonl')
    write_data(prefix_path, prefix_dataset, dp*prefix_num)
    if repeat_rate >= 1:
        dataset_path = os.path.join(save_path, f'GSM8K-in{prefix_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
        write_data(dataset_path, prefix_dataset, number)
        return prefix_path, dataset_path

    # 前缀后插入3个随机token
    uniq_token_set = generate_unique_tokens(tokenizer_path, seed, 3, number)
    # 后缀数据
    suffix_len = int(input_len - prefix_len - 3)
    suffix_dataset = create_dataset(tokenizer_path, suffix_len, number, 0)

    # 拼接完整数据集
    dataset = []
    data_len = 0
    pbar = tqdm(total=number, desc="Stitching dataset", unit="row") if tqdm else None
    while data_len < number:
        single_data = prefix_data[data_len % prefix_num] + uniq_token_set[data_len] + suffix_dataset[data_len]
        dataset.append(single_data)
        data_len += 1
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    dataset_path = os.path.join(save_path, f'GSM8K-in{input_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
    write_data(dataset_path, dataset, number)

    return prefix_path, dataset_path


def _create_prefix_dataset_variable(tokenizer_path, input_len, number, save_path, base_name, dp, repeat_rate, seed, prefix_num,
                                    length_mean, length_std, length_min, length_max):
    """变长前缀数据集生成：预采样长度 → 前缀池 → 逐条截断前缀/后缀并拼接"""
    rng = random.Random(seed)

    # 预采样每条数据的实际长度和公共前缀长度
    real_lens = [max(1, int(sample_target_length(rng, input_len, length_mean, length_std, length_min, length_max))) for _ in range(number)]
    common_lens = [max(0, min(rl, int(round(rl * repeat_rate)))) for rl in real_lens]
    max_common_len = max(common_lens) if common_lens else 0

    # 生成前缀池（统一最大公共长度，后续逐条截断）
    prefix_data = []
    if max_common_len > 0:
        prefix_data = create_dataset(tokenizer_path, max_common_len, prefix_num, 1)
        if prefix_data is None:
            logging.error("生成数据集失败，请清空picked ids")
            exit(0)
    else:
        prefix_data = [""] * prefix_num

    # 写前缀文件
    prefix_dataset = []
    for i in range(prefix_num):
        for j in range(dp):
            prefix_dataset.append(prefix_data[i])
    prefix_path = os.path.join(save_path, f'prefix-GSM8K-in{max_common_len}-num{dp*prefix_num}-{base_name}.jsonl')
    write_data(prefix_path, prefix_dataset, dp*prefix_num)
    if repeat_rate >= 1:
        dataset_path = os.path.join(save_path, f'GSM8K-in{max_common_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
        write_data(dataset_path, prefix_dataset, number)
        return prefix_path, dataset_path

    # 生成唯一 token 集合（每组3个不同token）
    uniq_token_set = generate_unique_tokens(tokenizer_path, seed, 3, number)

    # 后缀池：生成最大后缀长度，逐条截断
    max_suffix_len = max(rl - cl - 3 for rl, cl in zip(real_lens, common_lens))
    if max_suffix_len < 1:
        max_suffix_len = 1
    suffix_pool = create_dataset(tokenizer_path, max_suffix_len, number, 0)
    if suffix_pool is None:
        logging.error("生成后缀数据集失败，请清空picked ids")
        exit(0)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 逐条拼接
    dataset = []
    pbar = tqdm(total=number, desc="Stitching dataset (variable)", unit="row") if tqdm else None
    for idx in range(number):
        rl = real_lens[idx]
        cl = common_lens[idx]
        suffix_len_needed = max(0, rl - cl - 3)

        # 前缀截断
        prefix_text = prefix_data[idx % prefix_num]
        if cl > 0 and prefix_text:
            prefix_text = _truncate_or_pad_text(tokenizer, prefix_text, cl)
        else:
            prefix_text = ""

        # 后缀截断
        suffix_text = suffix_pool[idx]
        if suffix_len_needed > 0 and suffix_text:
            suffix_text = _truncate_or_pad_text(tokenizer, suffix_text, suffix_len_needed)
        else:
            suffix_text = ""

        single_data = prefix_text + uniq_token_set[idx] + suffix_text
        dataset.append(single_data)

        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()

    length_tag = _build_length_tag(input_len, length_mean, length_std, length_min, length_max)
    dataset_path = os.path.join(save_path, f'GSM8K-{length_tag}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl')
    write_data(dataset_path, dataset, number)

    logging.info(f"  max_common_len={max_common_len}, max_suffix_len={max_suffix_len}")
    logging.info(f"  avg_hit_ratio={sum(c / r for c, r in zip(common_lens, real_lens)) / len(real_lens):.2%}")

    return prefix_path, dataset_path

def parse_prefix_ratio(r: str) -> float:
    """
    "50%" -> 0.5, "0.5" -> 0.5, "0.500" -> 0.5
    """
    r = str(r).strip()
    if r.endswith("%"):
        v = float(r[:-1]) / 100.0
    else:
        v = float(r)
    if not (0.0 <= v <= 1.0):
        raise ValueError("prefix-ratio 必须在 [0,1] 区间或百分数 [0%,100%]")
    return v
````


<a id="archive-1-7"></a>
### aisbench/save_file.py

````python
import os
import re
import logging
import shutil
import traceback
import pandas as pd
from datetime import datetime
from config import MODEL_PATH
logging.getLogger().setLevel(logging.INFO)

DEFAULT_METRICS = {
    "current_time": None,
    "input_len": 99999,
    "output_len": 99999,
    "total_req": 99999,
    "max_cc": 99999,
    "cc": 99999,
    "rr": 0,
    "repeat_rate": 0,
    "TTFT avg": 99999,
    "TTFT P90": 99999,
    "TPOT avg": 99999,
    "TPOT SLO_P90": 99999,
    "E2E_time": 99999,
    "output_throughput": 99999,
    "single_output_throughput": 99999,
    "E2E_throughput": 9999,
    "single_E2E_throughput": 9999,
    "qps": 9999,
    "qpm": 9999,
    "input_token_throughput": 9999,
    "prefill_token_throughput": 9999,
    "E2EL avg": 9999,
    "E2EL P90": 9999
}

_METRIC_RULES = [
    ("E2EL",                      r'(\d+\.\d+)', 5, 1.0,    None),
    ("TTFT",                      r'(\d+\.\d+)', 5, 1.0,    None),
    ("TPOT",                      r'(\d+\.\d+)', 5, 1.0,    None),
    ("Benchmark Duration",        r'(\d+\.\d+)', 0, 1/1000, None),
    ("Concurrency",               r'(\d+\.\d+)', 0, 1.0,    None),
    ("Output Token Throughput",   r'(\d+\.\d+)', 0, 1.0,    lambda v, n: {
        "output_throughput": v, "single_output_throughput": v / n}),
    ("Input Token Throughput",    r'(\d+\.\d+)', 0, 1.0,    None),
    ("Total Token Throughput",    r'(\d+\.\d+)', 0, 1.0,    lambda v, n: {
        "E2E_throughput": v, "single_E2E_throughput": v / n}),
    ("InputTokens",               r'(\d+\.?\d*)', 0, 1.0,   None),
    ("OutputTokens",              r'(\d+\.?\d*)', 0, 1.0,   None),
    ("Total Requests",            r'(\d+\.?\d*)', 0, 1.0,   None),
    ("Request Throughput",        r'(\d+\.\d+)', 0, 1.0,    lambda v, n: {
        "qps": v, "qpm": v * 60}),
    ("Prefill Token Throughput",  r'(\d+\.\d+)', 0, 1.0,    None),
]


def _extract_value(line, pattern, index, scale):
    matches = re.findall(pattern, line)
    if not matches or index >= len(matches):
        return None
    try:
        return float(matches[index]) * scale
    except (ValueError, TypeError):
        return None


def _get_field_name(keyword):
    name_map = {
        "E2EL": ["E2EL P90", "E2EL avg"],
        "TTFT": ["TTFT P90", "TTFT avg"],
        "TPOT": ["TPOT SLO_P90", "TPOT avg"],
        "Benchmark Duration": ["E2E_time"],
        "Concurrency": ["cc"],
        "Max Concurrency": ["max_cc"],
        "Output Token Throughput": ["output_throughput"],
        "Input Token Throughput": ["input_token_throughput"],
        "Total Token Throughput": ["E2E_throughput"],
        "InputTokens": ["input_len"],
        "OutputTokens": ["output_len"],
        "Total Requests": ["total_req"],
        "Request Throughput": ["qps"],
        "Prefill Token Throughput": ["prefill_token_throughput"],
    }
    return name_map.get(keyword, [])


def get_data(aisbench_log, req_rate, npu_num, repeat_rate):
    log_dir = ""
    metrics = dict(DEFAULT_METRICS)
    try:
        with open(aisbench_log, 'r') as f_streaming:
            txt = f_streaming.readlines()
            for line in txt:
                if "Current exp folder" in line:
                    matches = re.search(r"Current exp folder:\s*(.+)$", line)
                    if matches:
                        log_dir = matches.group(1).strip()
                if "Max Concurrency" in line:
                    matches = re.findall(r"[\w']+", line)
                    if matches:
                        metrics["max_cc"] = matches[-1]
                    continue
                for keyword, pattern, index, scale, extra_fn in _METRIC_RULES:
                    if keyword in line:
                        raw_value = _extract_value(line, pattern, index, scale)
                        if raw_value is not None:
                            if extra_fn:
                                metrics.update(extra_fn(raw_value, npu_num))
                            else:
                                if keyword in ("E2EL", "TTFT", "TPOT"):
                                    p90_key = f"{keyword} {'SLO_' if keyword == 'TPOT' else ''}P90"
                                    avg_key = f"{keyword} avg"
                                    if index == 5:
                                        metrics[p90_key] = raw_value
                                    avg_value = _extract_value(line, pattern, 0, scale)
                                    if avg_value is not None:
                                        metrics[avg_key] = avg_value
                                else:
                                    field_names = _get_field_name(keyword)
                                    if field_names:
                                        metrics[field_names[0]] = raw_value
                        break
        metrics["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["rr"] = req_rate
        metrics["repeat_rate"] = repeat_rate
    except Exception as e:
        logging.warning(traceback.format_exc())
    logging.info(metrics)
    return metrics, log_dir


def save_log(aisbench_log, log_dir):
    # 保存到 aisbench 的 Current exp folder 路径
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        shutil.copy2(aisbench_log, log_dir)
        logging.info(f"aisbench.log copied to: {log_dir}")
    # aisbench.log 本身就在当前运行 python 命令的目录（无需额外复制）
    # aisbench_all.log 也保存在当前目录（由脚本直接写入）
    logging.info(f"aisbench.log saved at current directory: {os.getcwd()}")


def save_csv(ans, filename):
    file_exists = os.path.exists(filename)
    # 从 config.py 的 MODEL_PATH 中取最后一段路径作为 model_name，作为 CSV 第一列
    model_name = MODEL_PATH.split("/")[-1]
    ans = {"model_name": model_name, **ans}
    df_new = pd.DataFrame([ans])
    try:
        if file_exists:
            df_existing = pd.read_csv(filename)
            logging.info("文件已存在，读取现有数据")
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            df_updated = df_updated.reindex(columns=df_existing.columns.union(df_new.columns, sort=False))
            # 确保 model_name 始终是第一列
            df_updated = df_updated.reindex(
                columns=["model_name"] + [c for c in df_updated.columns if c != "model_name"])
            df_updated.to_csv(filename, index=False)
            logging.info("成功追加新行")
        else:
            df_new.to_csv(filename, index=False)
            logging.info("创建新文件并写入数据")
    except Exception as e:
        logging.error(f"操作失败: {e}")
````


<a id="archive-1-8"></a>
### tests/analyze_revalidation.py

````python
"""Validate frozen-data A/B runs and derive medians from original CSV/logs.

Input: round directory containing raw/aisbench.log, aisbench_result.csv,
       dataset_before.sha256 and dataset_after.sha256.
Output: derived/result.json. Nonzero exit means the round is invalid.
TPOT=99999 is expected for one output token, and does not invalidate Prefill TPS.
"""
import ast
import csv
import json
import re
import statistics
import sys
from pathlib import Path

EXPECTED = {
    '932522b95ec848cfbf6718e480b2925768d498d008ae20e719ab1e1121958614',
    'ad4bf6bbe17b5bf3423858045bcbb4267eae963fb37b4bbc282cbe8782f2b59b',
}


def validate(root):
    raw = root / 'raw'
    log = (raw / 'aisbench.log').read_text(errors='replace')
    rows = list(csv.DictReader((raw / 'aisbench_result.csv').open()))
    row = rows[-1]
    errors = []
    for name in ('dataset_before.sha256', 'dataset_after.sha256'):
        hashes = {line.split()[0] for line in (raw / name).read_text().splitlines()}
        if hashes != EXPECTED:
            errors.append(f'{name}: dataset hashes differ')
    success = [int(n) for n in re.findall(r'Success Requests\s*│\s*total\s*│\s*(\d+)', log)]
    failed = [int(n) for n in re.findall(r'Failed Requests\s*│\s*total\s*│\s*(\d+)', log)]
    pinned_path = raw / 'pinned_warmup.json'
    pinned = json.loads(pinned_path.read_text()) if pinned_path.exists() else None
    if pinned is not None:
        if pinned['success'] != 4 or pinned['failed'] != 0 or [r['rank'] for r in pinned['requests']] != [0, 1, 2, 3]:
            errors.append('Pinned warmup missing a DP success')
        if any(r['P_http_status'] != 200 or r['D_http_status'] != 200 or not r['stream_done'] or r['D_usage']['completion_tokens'] != 1 for r in pinned['requests']):
            errors.append('Pinned warmup response invalid')
        if success != [196] or failed != [0]:
            errors.append(f'formal success={success}, failed={failed}, expected [196]/[0]')
    elif success != [4, 196] or failed != [0, 0]:
        errors.append(f'success={success}, failed={failed}, expected [4,196]/[0,0]')
    warmup = re.search(r'\[完成\] 前缀数据集测试.*?queries: normal=(\{[^\n]+?\}), external=', log, re.S)
    queries = ast.literal_eval(warmup.group(1)) if warmup else {}
    if set(queries) != {0, 1, 2, 3} or len(set(queries.values())) != 1 or not all(queries.values()):
        errors.append(f'unequal or missing DP warmup: {queries}')
    if '[复用]' not in log:
        errors.append('dataset reuse marker missing')
    if '[完成] 全量数据集测试完成' not in log:
        errors.append('formal completion marker missing')
    tps = float(row['prefill_token_throughput'])
    if not (0 < tps < 99999) or float(row['total_req']) != 196:
        errors.append('missing throughput or request count')
    if float(row['output_len']) != 1 or int(row['max_cc']) != 49:
        errors.append('unexpected output length or concurrency')
    result = {
        'round': root.name, 'valid': not errors, 'errors': errors,
        'time_beijing': row['current_time'], 'prefill_tps': tps,
        'ttft_avg_ms': float(row['TTFT avg']), 'ttft_p90_ms': float(row['TTFT P90']),
        'input_len_mean': float(row['input_len']), 'output_len_mean': float(row['output_len']),
        'success_requests': success, 'failed_requests': failed, 'warmup_queries': queries,
        'warmup_mode': 'pinned' if pinned is not None else 'automatic_proxy_routing',
        'dataset_sha256': sorted(EXPECTED),
    }
    derived = root / 'derived'
    derived.mkdir(exist_ok=True)
    (derived / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return result


def summarize(root):
    groups = {'pool': [], 'nopool': []}
    rounds = []
    for path in sorted(root.glob('*/derived/result.json')):
        result = json.loads(path.read_text())
        rounds.append(result)
        if result['valid']:
            group = 'nopool' if '_nopool_' in result['round'] else 'pool'
            groups[group].append(result['prefill_tps'])
    summary = {'rounds': rounds, 'groups': {}}
    for group, values in groups.items():
        if values:
            summary['groups'][group] = {
                'n': len(values), 'values': values, 'median': statistics.median(values),
                'min': min(values), 'max': max(values), 'range': max(values)-min(values),
                'relative_range_pct': (max(values)-min(values))/statistics.median(values)*100,
            }
    if all(len(v) >= 3 for v in groups.values()):
        base = summary['groups']['nopool']['median']
        pooled = summary['groups']['pool']['median']
        summary['regression_pct'] = (base-pooled)/base*100
        summary['strictly_gt_5_pct'] = summary['regression_pct'] > 5
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == '--summary':
        summarize(Path(sys.argv[2]))
    else:
        raise SystemExit(0 if validate(Path(sys.argv[1]))['valid'] else 2)
````


<a id="archive-1-9"></a>
### tests/pin_prefix_warmup.py

````python
"""Warm every P DP once using frozen prefixes and vLLM's supported rank header.

Matches the original proxy's P->D exchange; only warmup P routing is pinned.
The formal benchmark continues through the unmodified proxy/AISBench path.
Input: four identical frozen prefix rows. Output: real P/D status and usage per DP.
No retries: any missing response invalidates the round.
"""
import concurrent.futures
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import httpx


def warmup(dataset, dp):
    rows = [json.loads(line) for line in Path(dataset).read_text().splitlines()]
    if len(rows) != dp or dp != 4:
        raise ValueError(f'Expected four frozen prefix rows, found {len(rows)}')
    if len({row['question'] for row in rows}) != 1:
        raise ValueError('This scenario requires identical prefix rows across DP domains')

    def request(rank):
        started = time.monotonic()
        rid = 'dts-pin-prefix-' + str(uuid.uuid4())
        common = {
            'model': 'model', 'messages': [{'role': 'user', 'content': rows[rank]['question']}],
            'temperature': 0, 'ignore_eos': True, 'max_tokens': 1,
        }
        prefill = dict(common, stream=False, min_tokens=1, kv_transfer_params={
            'do_remote_decode': True, 'do_remote_prefill': False,
            'remote_engine_id': None, 'remote_block_ids': None,
            'remote_host': None, 'remote_port': None, 'aborted_request': [],
        })
        with httpx.Client(timeout=600, trust_env=False) as client:
            response = client.post('http://80.5.9.135:8910/v1/chat/completions', json=prefill,
                                   headers={'X-data-parallel-rank': str(rank), 'X-Request-Id': rid})
            response.raise_for_status()
            pjson = response.json()
            transfer = pjson.get('kv_transfer_params')
            if not transfer:
                raise ValueError(f'P DP{rank}: missing PD transfer parameters')
            decode = dict(common, stream=True, stream_options={'include_usage': True}, kv_transfer_params=transfer)
            chunks = 0
            usage = {}
            done = False
            with client.stream('POST', 'http://80.5.9.133:8920/v1/chat/completions', json=decode,
                               headers={'X-Request-Id': rid}) as decoded:
                decoded.raise_for_status()
                for line in decoded.iter_lines():
                    if not line.startswith('data: '):
                        continue
                    event = line[6:]
                    if event == '[DONE]':
                        done = True
                        continue
                    event = json.loads(event)
                    if 'error' in event:
                        raise ValueError(event['error'])
                    chunks += 1
                    if event.get('usage'):
                        usage = event['usage']
                if not done or usage.get('completion_tokens') != 1:
                    raise ValueError(f'D DP warmup did not complete one token: rank={rank}, done={done}, usage={usage}')
                return {'rank': rank, 'request_id': rid, 'P_http_status': response.status_code,
                        'D_http_status': decoded.status_code, 'stream_done': done, 'stream_chunks': chunks,
                        'P_usage': pjson.get('usage'), 'D_usage': usage,
                        'elapsed_seconds': time.monotonic()-started,
                        'prefix_text_sha256': hashlib.sha256(rows[rank]['question'].encode()).hexdigest()}

    result = {'mode': 'pinned_P_DP_0_to_3_with_original_PD_exchange', 'requests': [], 'errors': []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=dp) as executor:
        futures = {executor.submit(request, rank): rank for rank in range(dp)}
        for future in concurrent.futures.as_completed(futures):
            try:
                result['requests'].append(future.result())
            except Exception as exc:
                result['errors'].append({'rank': futures[future], 'error': str(exc)})
    result['requests'].sort(key=lambda row: row['rank'])
    result['success'] = len(result['requests'])
    result['failed'] = len(result['errors'])
    output = Path(os.environ['DTS_PIN_PREFIX_RESULT'])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    if result['errors'] or result['success'] != dp:
        raise RuntimeError(f'Pinned prefix warmup failed: {result["errors"]}')
    print(f'PINNED WARMUP SUCCESS: {dp}/{dp}; evidence={output}', flush=True)
````


<a id="archive-1-10"></a>
### tests/render_revalidation_report.py

````python
"""Render the six-round report from validated original evidence, after cleanup."""
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
partial = '--partial' in sys.argv[2:]
summary = ({'rounds': [row for path in sorted(root.glob('*/derived/result.json'))
                      if (row := json.loads(path.read_text()))['valid']]}
           if partial else json.loads((root / 'summary.json').read_text()))
rounds = summary['rounds']
status = (root / 'final_status.txt').read_text() if (root / 'final_status.txt').exists() else 'Still running'
if not partial and (len(rounds) != 6 or not all(row['valid'] for row in rounds)):
    raise SystemExit('Six valid rounds required before rendering a final conclusion')
if not partial and (root / 'exit_code.txt').read_text().strip() != '0':
    raise SystemExit('Final cleanup not successful; record the failure before finalizing')
table = ['| 轮次 | 组 | 原始服务器时间 | Prefill TPS | TTFT avg (ms) | 正式成功/失败 | 预热每DP queries | HBM / external hit |',
         '|---|---|---|---:|---:|---|---:|---|']
for row in rounds:
    directory = root / row['round']
    raw = directory / 'raw'
    log = (raw / 'aisbench.log').read_text(errors='replace')
    hits = re.findall(r'^ALL_PODS\s+(\S+)\s+(\d+)/(\d+)\s+(\S+)\s+(\d+)/(\d+)', log, re.M)
    cache = dict(zip(['hbm_rate', 'hbm_hits', 'hbm_queries', 'external_rate', 'external_hits', 'external_queries'], hits[-1])) if hits else {}
    (directory / 'derived/cache.json').write_text(json.dumps(cache, indent=2)+'\n')
    group = '无池化' if '_nopool_' in row['round'] else '池化 #18277'
    queries = list(row['warmup_queries'].values())[0]
    rate = f'{cache.get("hbm_rate","未记录")} / {cache.get("external_rate","未记录")}'
    table.append(f'| {row["round"]} | {group} | {row["time_beijing"]} | {row["prefill_tps"]:.4f} | {row["ttft_avg_ms"]:.1f} | 196 / 0 | {queries} | {rate} |')
    commits = []
    for node in ('135', '133'):
        inputs = directory / 'inputs' / node
        commits.append(f'- {node}: `{(inputs/"branch.txt").read_text().strip()}` / `{(inputs/"commit.txt").read_text().strip()}`。')
    body = f'''# {row['round']} — {group}

状态：有效。正式请求196/196成功、失败0；定向预热4/4，四DP queries均为{queries}；测试前后冻结数据SHA256一致。

原始P服务器结果时间：{row['time_beijing']}（服务器北京时间标签；两机时钟有偏差，按轮次对齐）。

Prefill TPS：**{row['prefill_tps']:.4f} token/s**；TTFT avg：{row['ttft_avg_ms']:.1f} ms；TTFT P90：{row['ttft_p90_ms']:.1f} ms。平均输入{row['input_len_mean']} token、输出1 token、并发49，TPOT不可计算占位不参与判断。HBM/external hit：{rate}。

实际代码：

{chr(10).join(commits)}

实际执行：每轮停P/D清HBM；池化额外重启Meta/standalone；启动本组P/D，health通过后启动原proxy。135容器执行 `DTS_PIN_PREFIX_RESULT={raw}/pinned_warmup.json bash /home/lizhongyang/1011/revalidation_tools/run_pinned.sh`。预热用冻结前缀与官方DP请求头，复刻原P→D交换，随后执行未改动的正式AISBench路径。性能测试未开启profiler。

证据：[正式原始日志](raw/aisbench.log)、[原始CSV](raw/aisbench_result.csv)、[定向预热真实响应](raw/pinned_warmup.json)、[轮前哈希](raw/dataset_before.sha256)、[轮后哈希](raw/dataset_after.sha256)、[有效性及指标](derived/result.json)、[缓存命中](derived/cache.json)。完整服务日志如超过5MiB，保留在内网服务器；本地保存首尾片段，精确路径/大小/SHA256见上一级evidence_manifest.json。

输入追溯：本轮inputs保存两机commit、分支、全体Git跟踪文件SHA256和runtime.diff；父目录inputs保存两机脚本/配置tar快照、组件版本、镜像/挂载与定向预热工具源码。每轮只取正式阶段CSV最后一行，warmup不进入吞吐统计。
'''
    (directory / 'README.md').write_text(body)

if partial:
    print(f'Rendered {len(rounds)} validated round reports; no final A/B conclusion')
    raise SystemExit(0)

pool = summary['groups']['pool']
base = summary['groups']['nopool']
regression = summary['regression_pct']
verdict = '严格超过5%，按本任务验收标准判定池化吞吐回退超过阈值' if summary['strictly_gt_5_pct'] else '未严格超过5%，本次中位数未触发回退阈值'
pairs = []
for index in range(0, 6, 2):
    pooled = rounds[index]['prefill_tps']
    baseline = rounds[index+1]['prefill_tps']
    pairs.append((baseline-pooled)/baseline*100)
summary['paired_regression_pct'] = pairs
(root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
(root / 'derived').mkdir(exist_ok=True)
with (root / 'derived/round_metrics.csv').open('w', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=['round','time_beijing','prefill_tps','ttft_avg_ms','ttft_p90_ms','input_len_mean','output_len_mean'])
    writer.writeheader()
    writer.writerows({key:row[key] for key in writer.fieldnames} for row in rounds)
body = f'''# A3 池化 #18277 vs 无池化：冻结数据六轮复验

状态：六轮验证及清理完成。结论：**回退率{regression:.2f}%，{verdict}**。

无池化中位数 **{base['median']:.4f} TPS**；池化中位数 **{pool['median']:.4f} TPS**。计算 `(无池化中位数-池化中位数)/无池化中位数×100%`，严格大于5%才判超阈值。各3个有效轮，未混入此前失败启动或预热不完整的轮次。

## 实验身份与计划

用户于2026-10-13明确授权直接使用两机。P=80.5.9.135（node-97-36），D=80.5.9.133（vllm-dev-a3-133），各16个A3芯片；容器均lizy_1009；镜像day20261012，完整标识 `sha256:723f75865d785bcafdc0bd7be6999695ffa5777d4cfd24486cfbd0aa742e0b0e`。池化为pr-18277的4文件runtime补丁，无池化为镜像master。两机安装包跟踪文件逐轮SHA256完全一致；精确commit与差异见各轮inputs。

负载：输入131072、输出1、并发49、196条，repeat_rate0.9，prefix_test、dp4、npu_num32、request_rate0、seed1。正式冻结数据SHA256 `932522b95ec848cfbf6718e480b2925768d498d008ae20e719ab1e1121958614`；前缀 `ad4bf6bbe17b5bf3423858045bcbb4267eae963fb37b4bbc282cbe8782f2b59b`。每轮前后核对，未重新生成数据。

顺序：池化1→无池化1→池化2→无池化2→池化3→无池化3。每轮完整重启P/D清HBM，池化额外重启Meta/两机standalone清池；health就绪后测试，不开profiler。只统计正式196条阶段，预热剔除。

## 预热方案及偏差

原普通代理预热首轮出现DP分配2/1/0/1，正式196条虽成功仍排除该轮（6154.7955 TPS）。修正后使用vLLM已支持的 `X-data-parallel-rank` 请求头，给P DP0～3各发一条相同冻结前缀；完全复刻原代理的P→D KV交换。原aisbench_test.py仅在内存替换预热子过程、跳过预热CSV保存；正式请求路径、代理、数据、采样参数不变。每轮定向预热真实P/D响应4/4，四DP指标增量相等。

先前两次输入采证启动失败及上述无效轮均独立留存，不计入统计。AISBench安装目录属性恢复曾受挂载限制，修正为内容/链接恢复并校验。P/D日志时钟有偏差，保留原始服务器时间并按轮次对齐，所有正式指标由同一P客户端计算。

## 原始结果与统计

{chr(10).join(table)}

| 组 | n | 中位数 TPS | min | max | 极差 | 极差/中位数 |
|---|---:|---:|---:|---:|---:|---:|
| 池化 #18277 | 3 | {pool['median']:.4f} | {pool['min']:.4f} | {pool['max']:.4f} | {pool['range']:.4f} | {pool['relative_range_pct']:.2f}% |
| 无池化 | 3 | {base['median']:.4f} | {base['min']:.4f} | {base['max']:.4f} | {base['range']:.4f} | {base['relative_range_pct']:.2f}% |

相邻三对回退率（诊断，不替代组中位数主口径）：{', '.join(f'{value:.2f}%' for value in pairs)}。

## 结论范围

结果适用于上述镜像/模型、A3两机1P1D和128K输入/输出1/并发49负载，以及池化#18277组合相对镜像无池化组合。三轮中位数与离散度是本次观测，不代表已建立总体统计置信界限。不据此定位某个函数的成本，未比较#18471/#18472、其他输出长度或profiler结果。外部命中指标见上表；external hit为0时结论覆盖池化路径和写入成本，未验证外部读取收益。

## 证据、复算与现场恢复

完整原件内网位置：`{root}`（135及133共享/mnt/share）。本地小体积原始证据直接保存，大服务日志保留SHA256、大小、精确位置和首尾片段，见evidence_manifest.json；片段不冒充全量日志。脚本/配置完整快照在inputs，组件版本与镜像/挂载亦在inputs。汇总[summary.json](summary.json)与[round_metrics.csv](derived/round_metrics.csv)可复算；每轮README链接对应原始CSV/日志与预热证据。

复算：`python3 analyze_revalidation.py <round_dir>`；所有轮完成后 `python3 analyze_revalidation.py --summary <run_dir>`，再 `python3 render_revalidation_report.py <run_dir>`。上述脚本实际版本随inputs留存。

结束状态：测试P/D、proxy、Meta/standalone均停止；两机回到master；AISBench配置/数据链接和原temp_api.py恢复且内容校验通过。保留既有容器、冻结数据、pyarrow25.0.1与测试辅助脚本。清理原始状态：

```text
{status.strip()}
```
'''
(root / 'README.md').write_text(body)
print(f'Rendered six-round report: regression={regression:.4f}%, {verdict}')
````


<a id="archive-1-11"></a>
### tests/run_pinned.sh

````bash
#!/usr/bin/env bash
# Same formal input/parameters as aisbench/run.sh; deterministic four-DP warmup.
set -o pipefail
cd /home/lizhongyang/1011/scripts/aisbench || exit 1
source /usr/local/Ascend/cann/set_env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python3 -u /home/lizhongyang/1011/revalidation_tools/run_pinned_aisbench.py \
    --input_len 131072 --output_len 1 \
    --concurrency 49 --data_num 196 \
    --dataset_type prefix_cache --repeat_rate 0.9 --prefix_num 1 \
    --prefix_test --dp 4 --npu_num 32 \
    --request_rate 0 --seed 1
````


<a id="archive-1-12"></a>
### tests/run_pinned_aisbench.py

````python
"""Execute the unchanged formal AISBench code with a deterministic warmup adapter.

The installed/task original aisbench_test.py is read and compiled in memory.
Only its prefix subprocess is replaced, and the unused prefix CSV save is skipped.
Each expected source block must match once; otherwise abort before any requests.
"""
import os
import sys
from pathlib import Path

source_path = Path('/home/lizhongyang/1011/scripts/aisbench/aisbench_test.py')
source = source_path.read_text()
old = '''            subprocess.run(ais_bench_cmd, shell=True, check=True)
            logging.info(f"[完成] 前缀数据集测试完成，结果保存在aisbench_result.csv")'''
new = '''            from pin_prefix_warmup import warmup as pinned_prefix_warmup
            pinned_prefix_warmup(src_file_prefix, dp)
            logging.info("[完成] 前缀数据集测试完成，逐DP定向预热响应保存在pinned_warmup.json")'''
skip = '            save_result(0, npu_num, repeat_rate)'
if source.count(old) != 1 or source.count(skip) != 1:
    raise SystemExit('AISBench source differs from the reviewed warmup adapter target')
source = source.replace(old, new).replace(skip, '            logging.info("Pinned warmup excluded from formal CSV")')
os.chdir(source_path.parent)
sys.path.insert(0, str(source_path.parent))
exec(compile(source, str(source_path), 'exec'), {'__name__': '__main__', '__file__': str(source_path)})
````




## 第三部分：agent 专用（必要操作说明与历史追溯）

2026-10-27 按新规范将原 4 个辅助脚本全文改为用途、操作条件和历史引用。它们负责远程编排、宿主机操作、停止和证据整理，可由以下明确说明重建，不作为本归档的正式测试入口。配置、启动及第二部分正式测试全文均保持原样。需要按历史实现追溯时使用下列原件，不将未跑完的自动化流程标成已验证。

| 历史工具 | 用途 | 运行位置 | 输入/配置 | 输出 | 验证状态与追溯 |
|---|---|---|---|---|---|
| [package_revalidation.py](#archive-2-0) | 整理原始证据及打包 | 135，可读取共享运行目录的 Python 环境 | 运行根目录；5 MiB 全量纳入上限 | evidence_manifest.json、日志片段、旁置 evidence.tar.gz、缓存摘要 | 清单和证据已留存；进行中快照等分支未逐项验证；[保留工具](../03_verification_scripts/agent/package_revalidation.py) |
| [revalidate_ab.sh](#archive-2-1) | 交替 A/B、逐轮采证及退出恢复 | 135 宿主机，需免密 SSH 到 133 | run_id、两机 lizy_1009、/tmp 中 host/stop/analyze 工具 | 逐轮 inputs/raw/derived、编排日志、final_status.txt | 仅完成 01_pool_1；02_nopool_1 启动中止，六轮未完成；[实测快照](evidence/20261013_pinned_revalidation_214027_retry1/inputs/revalidate_ab.sh) |
| [revalidate_host.sh](#archive-2-2) | 两机快照、切版本、启动、停止及采证 | 135、133 宿主机，仅操作目标容器 | action、输出目录、节点标识及入口/分支 | 代码/配置身份、轮次启动日志、进程和 NPU 记录 | 已用于有效池化轮及退出清理，不能推定全部分支已验证；[实测快照](evidence/20261013_pinned_revalidation_214027_retry1/inputs/revalidate_host.sh) |
| [revalidate_stop.py](#archive-2-3) | 停止目标服务并检查残留 | 两机 lizy_1009 容器内部 | 待停止服务的进程身份 | 信号发送记录、残留检查及退出状态 | 最终停止状态 P_stop/D_stop 均 0；[实测快照](evidence/20261013_pinned_revalidation_214027_retry1/inputs/revalidate_stop.py) |

<a id="archive-2-0"></a>
### 证据整理与打包

输入为 `/mnt/share/lizhongyang/dts_zyd_ab/<run_id>`。保留原小体积输出，逐文件记录相对路径、服务器路径、大小、SHA256 和是否完整纳入。历史实现以 5 MiB 为全量纳入上限；超过上限只携带首 32768 字节及尾 65536 字节片段，完整原件留内网，片段不能冒充完整日志。不复制数据集，不跟随符号链接，排除 AISBench 安装备份、__pycache__ 和旧清单；进行中快照不读取仍在追加且未形成完成状态的服务 .out 日志。

缓存摘要取各轮 `raw/aisbench.log` 最后一个 `ALL_PODS` 行，按 HBM rate/hits/queries、external rate/hits/queries 保存为 `derived/cache.json`；需要精确复算时使用第二部分已保留的统计实现，并与原始日志核对。AISBench 日志标出的性能输出仅从 `/home/lizhongyang/1011/scripts/aisbench/outputs/` 收集符合体积限制的普通文件。清单写入后将选定文件打成运行目录旁的证据包。

<a id="archive-2-1"></a>
### A/B 编排与有效性条件

计划顺序为 `01_pool_1 → 02_nopool_1 → 03_pool_2 → 04_nopool_2 → 05_pool_3 → 06_nopool_3`；每轮独立留存 inputs、raw、derived。开始前记录镜像/挂载、组件版本、两机代码及配置快照，并备份 AISBench 被替换的文件、链接与原不存在状态。

- 每轮先停止本套 P/D 清 HBM；池化轮额外重启 Meta 和两机 standalone 清池态。只操作两机目标容器，确认设备健康、16 芯片资源已释放且约定端口无残留监听，不能停止其他使用者服务来满足条件。
- 池化切 `pr-18277`，无池化切 `master`；核对两机 Git 跟踪文件哈希一致，并与 [02](02_development_and_pr.md) 的选定代码身份对应。两组分别调用第一部分的原启动入口，保持同一冻结数据、负载和原代理，均启用 HBM 前缀缓存，不开启 profiler。
- 池化先启动 Meta、确认 5000 监听，再启动两机 standalone 并确认初始化成功；之后启动 P/D。两端 `/health` 成功、原代理 8100 监听才开始测试。历史等待上限为 Meta 120 秒、standalone 300 秒、单端 health 1800 秒；启动最多尝试 3 次，失败先保存当次原始记录，再停本套服务、等待 15 秒重试。引擎致命启动错误立即判该次启动无效，重试不能掩盖失败记录。
- 正式入口为第二部分 `run_pinned.sh`；通过 `DTS_PIN_PREFIX_RESULT` 指定当前轮预热响应路径。必须核对冻结前缀和正式数据的前后 SHA256、四 DP 的真实预热响应及查询增量、正式 196/196 成功和有效吞吐。预热不进入正式 CSV；只取同轮正式阶段 CSV 最后一行，并与原始日志交叉核对。退出码 0 本身不是轮次有效的充分条件。
- 逐轮调用第二部分分析器，失败轮保留原因并排除。仅在两组各有 3 个有效轮后计算组中位数、min/max/极差及回退率。历史实际只有 1 个有效池化轮，不得把旧基线拼入本轮或生成新的 A/B 回退结论。

退出或中断时先采集当前轮证据，再停止两机目标服务、切回 master，恢复 AISBench 原内容/链接/原不存在状态并核对；不强制复制文件系统不支持的属性。记录清理各步返回值及最终进程、分支状态；历史 TERM/INT 退出为 130，清理失败需另报，不能冒充测试通过。后续复验使用新 run_id 和完整清态起点。

<a id="archive-2-2"></a>
### 两机操作与采证边界

宿主机操作参数为 action、输出目录、节点标识，切分支或启动时另给分支/入口。只进入 `lizy_1009`；容器内 scripts 为 `/home/lizhongyang/1011/scripts`，被测安装包为 `/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend`。快照保留 Git 状态/分支/commit、master 与 pr-18277 差异、Git 跟踪文件哈希、配置/脚本、外部代理来源及哈希、组件版本和容器 inspect；版本采证在中立目录执行，避免 platform.py 遮蔽标准库。

后台启动使用 `docker exec ... bash -c` 调用第一部分正式入口，轮次启动输出单独保存为 `<节点>_<入口名>.out`。采证记录测试后 NPU、进程和相关服务日志；辅助 p.log/d.log 可能带旧内容，以本轮 .out 为服务启动证据。停服后检查 16 芯片健康及显存状态、8910/8920/8100/5000/6000/8004/12321 端口；历史脚本还检查每芯片显存不超过 5000 MiB，该阈值仅为现场清态检查，不能作为吞吐通过标准。

<a id="archive-2-3"></a>
### 停止与恢复

先确认目标容器及本轮服务进程身份，只向已确认的 P/D、代理、Meta、standalone 及其子进程发送 TERM；历史实现等待 8 秒后对仍符合其匹配规则的进程发送 KILL，再等待 3 秒，最后检查是否仍有匹配进程。不得把历史脚本对 VLLM 或 multiprocessing 的宽泛匹配直接用于存在其他任务的容器；若复用，需重新确认归属，按本轮记录的 PID 与身份操作。这里只说明必要边界，不在本次文档整理中执行任何停止操作。

历史退出与恢复事实见 [05](05_verification_process_and_results.md) 及 [final_status.txt](evidence/20261013_pinned_revalidation_214027_retry1/final_status.txt)。本次删去的是正文重复全文，原工具和运行证据未删改；修订前文档、4 个工具副本及哈希记录见 [修订留存](../../../../../data/task_docs_retained/20261027/dts_zyd/document_revisions/agent_section_simplification/revision.md)。
