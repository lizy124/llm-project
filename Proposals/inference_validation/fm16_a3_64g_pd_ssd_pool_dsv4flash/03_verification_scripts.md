

# 03 完整配置与脚本

本章集中保存服务配置、启动与池化指标脚本全文；AISBench 只保留必要配置和一条测试命令，其工具实现独立整理。代码块前的 `file:` 标记表示相对于运行根目录的保存路径；保存时使用 UTF-8 和 LF。运行根目录建议为 `/home/lizhongyang/inference_validation/fm16_a3_64g_pd_ssd_pool_dsv4flash`，设置 `export TASK_ROOT=该路径` 后按 04 执行。

需要展开文件时，先在本地 PowerShell 切到当前电脑的 `llm-project` 仓库根目录，再执行下面的文档导出工具。它只从本章提取带 `file:` 标记的完整文件，写到新的临时目录；目标已存在则停止，不覆盖原文件，也不会修改 `ascend_project`。导出后仍须按 02、04 准备代理实现、统一的 AISBench 工具包及其数据等外部依赖。

```powershell
$DocumentationRoot = (Get-Location).Path
$Document = Join-Path $DocumentationRoot 'Proposals/inference_validation/fm16_a3_64g_pd_ssd_pool_dsv4flash/03_verification_scripts.md'
if (-not (Test-Path -LiteralPath $Document -PathType Leaf)) { throw '请先切到 llm-project 仓库根目录。' }
$ExportRoot = Join-Path $env:TEMP 'fm16_a3_pd_ssd_runtime'
if (Test-Path -LiteralPath $ExportRoot) { throw '导出目录已存在，请使用新的专属目录。' }
$ExportRoot = [IO.Path]::GetFullPath($ExportRoot)
$text = [IO.File]::ReadAllText($Document, [Text.Encoding]::UTF8)
$pattern = '(?ms)^<!-- file: (?<path>[^\r\n]+) -->\r?\n```[a-z0-9]*\r?\n(?<body>.*?)^```[ \t]*\r?$'
$files = [regex]::Matches($text, $pattern)
if ($files.Count -eq 0) { throw '未找到文件代码块。' }
$encoding = [Text.UTF8Encoding]::new($false)
foreach ($file in $files) {
    $relative = $file.Groups['path'].Value
    $destination = [IO.Path]::GetFullPath((Join-Path $ExportRoot $relative))
    if (-not $destination.StartsWith($ExportRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw '导出路径越界。'
    }
    if (Test-Path -LiteralPath $destination) { throw '存在重复或已有文件，停止导出。' }
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination)) | Out-Null
    $body = $file.Groups['body'].Value.Replace("`r`n", "`n").TrimEnd("`r", "`n") + "`n"
    [IO.File]::WriteAllText($destination, $body, $encoding)
}
Write-Output ('已导出 {0} 个文件到 {1}' -f $files.Count, $ExportRoot)
```

## 1. 配置与拉起服务

| 文件 | 执行位置与用途 | 输入/输出 | 状态 |
| --- | --- | --- | --- |
| `config/mmc-meta-ssd.conf` | P 容器，Meta 元数据、连接和淘汰水位 | :5000/:6000/:8004 | 来源快照，本次未重跑 |
| `config/mmc-local-standalone-ssd.conf` | P/D 容器各一份，贡献 DRAM/SSD | 5GB/节点、专用 loop 设备 | 来源快照，本次未重跑 |
| `config/mmc-local.conf` | P/D worker，只接入池 | DRAM=0GB | 来源快照，本次未重跑 |
| `launch/meta_ssd.sh` | P 容器前台，单个 MetaService | SSD meta 配置、服务日志 | 来源快照，本次未重跑 |
| `launch/standalone_ssd.sh`、`memcache_standalone.py` | P/D 容器各一个 standalone | SSD local 配置、初始化日志 | 来源快照，本次未重跑 |
| `launch/p_rank.sh`、`d_rank.sh` | 各自容器，每次一个 DP rank | 可见设备、API 端口、DP rank；stdout 由启动者接收 | 来源快照，本次未重跑 |
| `launch/proxy.sh` | P 容器，代理入口 | 依赖同目录代理 Python 原件，端口 8080 | 启动脚本完整，外部实现取得要求见 02 |

下面所有路径都以 `03_verification_scripts/` 开始。来源脚本的默认 TASK_ROOT 是旧任务目录；04 显式覆盖该变量，使本次使用独立路径。固定的 IP、网卡、拓扑和模型参数仍需对应现场，不能只修改 launcher 的参数。

### config/mmc-meta-ssd.conf

<!-- file: 03_verification_scripts/config/mmc-meta-ssd.conf -->
```ini
ock.mmc.meta_service_url = tcp://80.48.29.133:5000
ock.mmc.meta_service.config_store_url = tcp://80.48.29.133:6000
ock.mmc.meta_service.metrics_url = http://80.48.29.133:8004
ock.mmc.log_level = error
ock.mmc.evict_threshold_high = 80
ock.mmc.evict_threshold_low = 70
ock.mmc.meta_service.storage.metadata_mode = aggregated
```

### config/mmc-local-standalone-ssd.conf

<!-- file: 03_verification_scripts/config/mmc-local-standalone-ssd.conf -->
```ini
ock.mmc.meta_service_url = tcp://80.48.29.133:5000
ock.mmc.local_service.config_store_url = tcp://80.48.29.133:6000
ock.mmc.log_level = error
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 5GB
ock.mmc.local_service.max.dram.size = 1024GB
ock.mmc.local_service.storage.enabled = true
ubsio.disk.path = /dev/loop0
ubsio.mem.size_in_gb = 10
ubsio.standalone.device_count = 1
ubsio.standalone.force_new_disk = true
```

### config/mmc-local.conf

<!-- file: 03_verification_scripts/config/mmc-local.conf -->
```ini
ock.mmc.meta_service_url = tcp://80.48.29.133:5000
ock.mmc.local_service.config_store_url = tcp://80.48.29.133:6000
ock.mmc.log_level = error
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 0GB
ock.mmc.local_service.max.dram.size = 1024GB
```

### launch/meta_ssd.sh

<!-- file: 03_verification_scripts/launch/meta_ssd.sh -->
```bash
#!/usr/bin/env bash
# MetaService（SSD 三级池化）—— memcache 池元数据服务
# 运行位置：P 机 80.48.29.133 的容器内（wl_1011），日志 -> <task>/ssd_round/logs/meta_ssd.out
TASK_ROOT=${TASK_ROOT:-/home/lizhongyang/pool_64g_a3_profiling}
source /usr/local/Ascend/cann/set_env.sh
export MMC_META_CONFIG_PATH="$TASK_ROOT/03_verification_scripts/config/mmc-meta-ssd.conf"
exec python3 -u -c 'from memcache_hybrid import MetaService; MetaService.main()' >> "$TASK_ROOT/ssd_round/logs/meta_ssd.out" 2>&1
```

### launch/standalone_ssd.sh

<!-- file: 03_verification_scripts/launch/standalone_ssd.sh -->
```bash
#!/usr/bin/env bash
# standalone 池服务（SSD 三级池化，device_sdma + UBS IO SSD）—— 每机一个
# 运行位置：133/134 的容器内，日志 -> <task>/ssd_round/logs/standalone_ssd.out
TASK_ROOT=${TASK_ROOT:-/home/lizhongyang/pool_64g_a3_profiling}
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MMC_LOCAL_CONFIG_PATH="$TASK_ROOT/03_verification_scripts/config/mmc-local-standalone-ssd.conf"
exec python3 -u "$TASK_ROOT/03_verification_scripts/launch/memcache_standalone.py" >> "$TASK_ROOT/ssd_round/logs/standalone_ssd.out" 2>&1
```

### launch/memcache_standalone.py

<!-- file: 03_verification_scripts/launch/memcache_standalone.py -->
```python
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
```

### launch/p_rank.sh

<!-- file: 03_verification_scripts/launch/p_rank.sh -->
```bash
#!/usr/bin/env bash
# P 侧单 rank 启动（池化组：MooncakeHybrid 做 PD 传输 + AscendStore(memcache) 外部池）
# 运行位置：P 机 80.48.29.133 的容器内
# 参数约定见 vllm-ascend/examples/external_online_dp：$1=设备 $2=端口 $3=DP大小 $4=DP rank
set -e

TASK_ROOT=${TASK_ROOT:-/home/lizhongyang/pool_64g_a3_profiling}
VISIBLE=$1
PORT=$2
DP_RANK=$4

source /usr/local/Ascend/cann/set_env.sh
unset https_proxy
unset http_proxy

export LOCAL_IP=80.48.29.133
export NIC_NAME=enp48s3u1u1
export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME

export MMC_LOCAL_CONFIG_PATH="$TASK_ROOT/03_verification_scripts/config/mmc-local.conf"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/python3.11.10/lib/python3.11/site-packages/mooncake:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:/usr/local/python3.11.10/lib:/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_TRANSFER_TIMEOUT=1800
export HCCL_IF_BASE_PORT=51000
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
unset ASCEND_LOCAL_COMM_RES_AUTO_HCCS
export ASCEND_SUPER_POD_ID=2
export ASCEND_SERVER_INDEX=0
export ASCEND_NET_INSTANCE_ID=2
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export HCCL_ALGO=level0:fullmesh
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V1=1
export ASCEND_A3_ENABLE=1
export ACL_OP_INIT_MODE=1
export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE

unset VLLM_PP_LAYER_PARTITION
unset HCCL_INTRA_PCIE_ENABLE
unset HCCL_INTRA_ROCE_ENABLE
unset ASCEND_BUFFER_POOL
export ASCEND_ENABLE_USE_FABRIC_MEM=1
export PATH=/usr/local/bin:/usr/local/Ascend/driver/tools:$PATH
export VLLM_HOST_IP=$LOCAL_IP

# 每个 Prefill rank 独立 Mooncake handshake 端口
KV_PORT=$((30000 + DP_RANK))

exec vllm serve /mnt/share/weight/DeepSeek-V4-Flash-DSpark-w4a8-scope-ci \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-parallel-size 2 \
  --data-parallel-rank "$DP_RANK" \
  --data-parallel-address 80.48.29.133 \
  --data-parallel-rpc-port 12345 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --seed 1024 \
  --served-model-name ds \
  --async-scheduling \
  --max-model-len 134144 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.96 \
  --quantization ascend \
  --enforce-eager \
  --block-size 32 \
  --speculative-config '{"num_speculative_tokens": 5,"method": "dspark","enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --tokenizer-mode deepseek_v4 \
  --attention_config.indexer_kv_dtype int8 \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config "{
      \"kv_connector\": \"MultiConnector\",
      \"kv_role\": \"kv_producer\",
      \"engine_id\": \"0\",
      \"kv_connector_extra_config\": {
          \"connectors\": [
              {
                  \"kv_connector\": \"MooncakeHybridConnector\",
                  \"kv_role\": \"kv_producer\",
                  \"kv_port\": \"$KV_PORT\",
                  \"kv_connector_extra_config\": {
                      \"prefill\": {\"dp_size\": 2, \"pp_size\": 1, \"tp_size\": 8},
                      \"decode\": {\"dp_size\": 8, \"tp_size\": 2}
                  }
              },
              {
                  \"kv_connector\": \"AscendStoreConnector\",
                  \"kv_role\": \"kv_producer\",
                  \"kv_connector_extra_config\": {
                      \"lookup_rpc_port\": \"0\",
                      \"backend\": \"memcache\"
                  }
              }
          ]
      }
  }" \
  --additional-config '{
      "multistream_overlap_shared_expert": true,
      "enable_dsa_cp": true,
      "enable_sparse_sfa_c8": false,
      "enable_sparse_li_c8": false,
      "ascend_compilation_config": {"fuse_muls_add": true}
  }'
```

### launch/d_rank.sh

<!-- file: 03_verification_scripts/launch/d_rank.sh -->
```bash
#!/usr/bin/env bash
# D 侧单 rank 启动（池化组：MooncakeHybrid 做 PD 传输 + AscendStore(memcache) 外部池）
# 运行位置：D 机 80.48.29.134 的容器内
# 参数约定见 vllm-ascend/examples/external_online_dp：$1=设备 $2=端口 $3=DP大小 $4=DP rank
set -e

TASK_ROOT=${TASK_ROOT:-/home/lizhongyang/pool_64g_a3_profiling}
VISIBLE=$1
PORT=$2
DP_RANK=$4

source /usr/local/Ascend/cann/set_env.sh
unset https_proxy
unset http_proxy

export LOCAL_IP=80.48.29.134
export NIC_NAME=enp48s3u1u1
export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME

export MMC_LOCAL_CONFIG_PATH="$TASK_ROOT/03_verification_scripts/config/mmc-local.conf"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/python3.11.10/lib/python3.11/site-packages/mooncake:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:/usr/local/python3.11.10/lib:/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_TRANSFER_TIMEOUT=1800
export HCCL_IF_BASE_PORT=51000
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
unset ASCEND_LOCAL_COMM_RES_AUTO_HCCS
export ASCEND_SUPER_POD_ID=2
export ASCEND_SERVER_INDEX=1
export ASCEND_NET_INSTANCE_ID=2
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'
export HCCL_ALGO=level0:fullmesh
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_USE_V1=1
export ASCEND_A3_ENABLE=1
export ACL_OP_INIT_MODE=1
export ASCEND_RT_VISIBLE_DEVICES=$VISIBLE

unset HCCL_INTRA_PCIE_ENABLE
unset HCCL_INTRA_ROCE_ENABLE
unset ASCEND_BUFFER_POOL
export ASCEND_ENABLE_USE_FABRIC_MEM=1
export PATH=/usr/local/bin:/usr/local/Ascend/driver/tools:$PATH
export VLLM_HOST_IP=$LOCAL_IP

exec vllm serve /mnt/share/weight/DeepSeek-V4-Flash-DSpark-w4a8-scope-ci \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-parallel-size 8 \
  --data-parallel-rank "$DP_RANK" \
  --data-parallel-address 80.48.29.134 \
  --data-parallel-rpc-port 13345 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name ds \
  --async-scheduling \
  --max-model-len 134144 \
  --max-num-batched-tokens 192 \
  --trust-remote-code \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.95 \
  --quantization ascend \
  --block-size 32 \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  --attention_config.indexer_kv_dtype int8 \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"num_speculative_tokens": 5, "method": "dspark", "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{
      "kv_connector": "MultiConnector",
      "kv_role": "kv_consumer",
      "engine_id": "1",
      "kv_connector_extra_config": {
          "connectors": [
              {
                  "kv_connector": "MooncakeHybridConnector",
                  "kv_role": "kv_consumer",
                  "kv_port": "30200",
                  "kv_connector_extra_config": {
                      "prefill": {"dp_size": 2, "pp_size": 1, "tp_size": 8},
                      "decode": {"dp_size": 8, "tp_size": 2}
                  }
              },
              {
                  "kv_connector": "AscendStoreConnector",
                  "kv_role": "kv_consumer",
                  "kv_connector_extra_config": {
                      "lookup_rpc_port": "0",
                      "backend": "memcache"
                  }
              }
          ]
      }
  }' \
  --additional-config '{
      "multistream_overlap_shared_expert": true,
      "enable_dsa_cp": true,
      "enable_sparse_sfa_c8": false,
      "enable_sparse_li_c8": false,
      "ascend_compilation_config": {"fuse_muls_add": true}
  }'
```

### launch/proxy.sh

<!-- file: 03_verification_scripts/launch/proxy.sh -->
```bash
#!/usr/bin/env bash
# PD proxy（运行于 P 机 80.48.29.133 的容器内）：对外 8080 -> P 133:8000,8001 / D 134:8101..8108
# 依赖同目录 load_balance_proxy_server_example.py（来源：容器 /home/w00985460/script/vllm/）
set -o pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

exec python3 "$(dirname "$0")/load_balance_proxy_server_example.py" \
  --host 0.0.0.0 \
  --port 8080 \
  --prefiller-hosts 80.48.29.133 80.48.29.133 \
  --prefiller-ports 8000 8001 \
  --decoder-hosts 80.48.29.134 80.48.29.134 80.48.29.134 80.48.29.134 80.48.29.134 80.48.29.134 80.48.29.134 80.48.29.134 \
  --decoder-ports 8101 8102 8103 8104 8105 8106 8107 8108
```

## 2. 测试相关

### AISBench：必要配置与单条测试命令

AISBench 及前缀工具单独整理。本章只保留本场景的客户端配置和调用命令，不收录主工具、数据生成、数据选取、统计实现、许可证或运行包装脚本全文。执行前在独立工具目录中准备统一的 AISBench 工具包、依赖和数据，并将以下配置保存为该工具目录的 config.py。

### tests/aisbench/config.py

<!-- file: 03_verification_scripts/tests/aisbench/config.py -->
```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_PATH = str(ROOT / 'dataset')
OUTPUT_DIR = str(ROOT / 'outputs')
# AISBench 安装根（测试人员 133 上的 ais_bench editable 源，可写；工具会替换其
# vllm_api_chat_temp.py 与 datasets/gsm8k/test.jsonl 两个链接）
WORK_PATH = '/mnt/share/l30041859/space/aisbench/benchmark'
MODEL_NAME = 'ds'
MODEL_PATH = '/mnt/share/weight/DeepSeek-V4-Flash-DSpark-w4a8-scope-ci'
HOST_IP = '80.48.29.133'
HOST_PORT = '8080'
POD_INFO = ['80.48.29.133:8000']
URL = ''
API_KEY = ''
DEFAULT_PERFORMANCE_TEST = 'default_perf'
```

在 P 容器内切到已准备好的 `$AISBENCH_TOOL_DIR`，使用 `$TEST_PY` 执行这一条命令；变量准备见 04。默认是 128K/1 的 SSD 工作集，DRAM 小样本只调整 CONCURRENCY、SEED、PREFIX_NUM、DATA_NUM，命令本身相同。

```bash
"$TEST_PY" -u aisbench_test.py \
  --input_len "${INPUT_LEN:-131072}" --output_len "${OUTPUT_LEN:-1}" \
  --concurrency "${CONCURRENCY:-32}" --data_num "${DATA_NUM:-64}" \
  --dataset_type prefix_cache --repeat_rate 0.9 --prefix_num "${PREFIX_NUM:-32}" \
  --prefix_test --dp 2 --npu_num 16 --request_rate 0 --seed "${SEED:-201}"
```

模型名、tokenizer 路径、代理地址、WORK_PATH 和 POD_INFO 以本配置为准。POD_INFO 原配置只列 P:8000，两个 P API 的完整指标检查按 05 执行。工具目录与其可写安装位置需独立，避免旧数据或其他任务影响。

### 池化指标查看

以下是池化专项采证工具，不属于 AISBench 内部实现。来源查看器只打印累计值，缺指标时可能显示为零，不能单独作为判据；严格差值工具见其后。

### tests/read_tier_hits.sh

<!-- file: 03_verification_scripts/tests/read_tier_hits.sh -->
```bash
#!/usr/bin/env bash
# 读取 MetaService 指标，解出 DRAM/SSD 命中占比与 SSD 容量，用于 80/20 判读
# 运行位置：P 机 80.48.29.133 宿主机（MetaService 在容器内，metrics 端口 8004）
# 用法：bash read_tier_hits.sh            # 读当前累计值
# MetaService 监听绑在 metrics_url 的 IP 上（非 0.0.0.0），默认用配置里的 IP
META=${META:-http://80.48.29.133:8004/metrics}
TXT=$(curl -fsS --max-time 5 "$META" 2>/dev/null) || { echo "META_METRICS_UNREACHABLE $META"; exit 1; }

awk '
  /^memcache_get_hits_dram_total /{d=$2}
  /^memcache_get_hits_ssd_total /{s=$2}
  /^memcache_ubs_io_disk_capacity_bytes/{cap+=$2}
  /^memcache_ubs_io_disk_status/{st=st" "$2}
  END{
    tot=d+s;
    printf "dram_hits=%d  ssd_hits=%d  total=%d\n", d, s, tot;
    if (tot>0) printf "dram_share=%.2f%%  ssd_share=%.2f%%\n", 100*d/tot, 100*s/tot;
    else       printf "dram_share=n/a  ssd_share=n/a (total=0)\n";
    printf "ubs_io_disk_capacity_bytes=%d  (~%d GiB)  disk_status=[%s]\n", cap, cap/1073741824, st;
  }
' <<<"$TXT"
```

### tests/tier_delta.py（本次补充的离线差值工具）

用途：读取前后两个原始 Meta Prometheus 快照，只计算 DRAM/SSD 命中计数差，不发送请求、不修改服务、不保存成绩。依赖 Python 3 标准库；用法为 `python3 tier_delta.py before.prom after.prom`。缺指标或计数器异常时退出码为 2；零增量表示本窗口未观察到该层命中。仍须人工确认窗口内未重启服务且没有其他客户端。

这是为弥补来源累计值查看器缺项处理而新增的辅助工具，整理时只进行可用的本地检查，尚未在目标环境实测。

<!-- file: 03_verification_scripts/tests/tier_delta.py -->
```python
#!/usr/bin/env python3
"""Compare two raw MetaService metric snapshots; never infer missing metrics as zero."""
import argparse
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

NAMES = ("memcache_get_hits_dram_total", "memcache_get_hits_ssd_total")
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+\S+)?$")


def read_metrics(path):
    found = {name: {} for name in NAMES}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.fullmatch(line)
        if not match or match[1] not in found:
            continue
        name, labels, token = match[1], match[2] or "", match[3]
        value = Decimal(token)
        if not value.is_finite() or value < 0:
            raise ValueError(f"{path}: invalid counter {name}{labels}={token}")
        if labels in found[name]:
            raise ValueError(f"{path}: duplicate series {name}{labels}")
        found[name][labels] = value
    for name, series in found.items():
        if not series:
            raise ValueError(f"{path}: missing metric {name}; do not interpret as zero")
    return found


def compare(before, after):
    deltas = {}
    for name in NAMES:
        if before[name].keys() != after[name].keys():
            raise ValueError(f"series changed for {name}; check registration/restart")
        changes = []
        for labels, old_value in before[name].items():
            change = after[name][labels] - old_value
            if change < 0:
                raise ValueError(f"counter decreased for {name}{labels}; invalid window")
            changes.append(change)
        deltas[name] = sum(changes, Decimal(0))
    return deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    try:
        deltas = compare(read_metrics(args.before), read_metrics(args.after))
    except (OSError, ValueError, InvalidOperation) as exc:
        print(f"INVALID_WINDOW: {exc}", file=sys.stderr)
        return 2
    for name, value in deltas.items():
        print(f"{name}: delta={value}")
    total = sum(deltas.values(), Decimal(0))
    if total > 0:
        ssd = deltas[NAMES[1]]
        print(f"ssd_read_share_in_window={100 * ssd / total:.2f}%")
    else:
        print("NO_TIER_HIT_OBSERVED: no tier read was observed in this window")
    print("Interpret only with successful requests, no restart, and an isolated workload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 3. 通用 DP 启动工具：引用 vllm-ascend examples

本场景直接引用 `vllm-ascend` 仓库中的通用 DP 工具，文件位置按仓库根目录计算：

| 用途 | vllm-ascend 仓库内路径 |
| --- | --- |
| DP launcher | `examples/external_online_dp/launch_online_dp.py` |
| 配套使用说明 | `examples/external_online_dp/README.md` |
| 模板和位置参数约定 | `examples/external_online_dp/run_dp_template.sh` |

不在本文复制 launcher 实现，也不假设 `vllm-ascend` 与 `llm-project` 在本机的相对位置。取得对应版本的代码仓后，以实际仓库根目录定位这些文件。

本次核对的工具仓库 commit 为 `06b685f346ad525027d03cffe80d58a491a13081`。这是通用工具的引用版本，不代表来源容器中的 vLLM/vllm-ascend 软件版本。远端从已准备好的代码仓读取相应脚本，不修改仓库示例；仓库路径由 `VLLM_ASCEND_REPO` 指定，04 示例使用 `/home/lizhongyang/code/vllm-ascend`。

### 3.1 launcher 参数

| 参数 | 含义 | P 配置 | D 配置 |
| --- | --- | --- | --- |
| `--dp-size` | 当前 P 或 D 服务的全局 DP 数，各自独立计算 | `2` | `8` |
| `--tp-size` | 每个 DP rank 使用的 die 数 | `8` | `2` |
| `--dp-size-local` | 本节点启动多少个 DP rank | `2` | `8` |
| `--dp-rank-start` | 本节点起始 DP rank | `0` | `0` |
| `--dp-address` | 当前 P 或 D 服务的 DP 主地址 | `80.48.29.133` | `80.48.29.134` |
| `--dp-rpc-port` | 当前 DP 组的 RPC 端口 | `12345` | `13345` |
| `--vllm-start-port` | 本节点第一个 API 端口；后续逐 rank 递增 | `8000` | `8101` |

P/D 各在本机使用 16 个 die。分配公式为：本机第 i 个 rank 使用 `i*tp_size` 到 `(i+1)*tp_size-1`，API 端口为起始端口加 i。它不负责跨节点执行命令，两台容器分别启动。

### 3.2 本场景的启动模板

该版本 launcher 固定调用**当前工作目录**的 `./run_dp_template.sh`，没有 `--template` 参数。按 04 在运行目录建立 P/D 各自的子目录，并将本场景的 `launch/p_rank.sh` 或 `launch/d_rank.sh` 链接为该目录的 `run_dp_template.sh`。这样直接复用仓库 launcher，并保留本场景的模型、PD 和池化配置。

| 位置参数 | launcher 传入的内容 | 本场景单 rank 脚本的处理 |
| --- | --- | --- |
| `$1` | 可见设备列表 | 设置 `ASCEND_RT_VISIBLE_DEVICES`。 |
| `$2` | 当前 API 端口 | 传给 vLLM `--port`。 |
| `$3` | 全局 DP size | 来源 P/D 脚本仍分别显式固定为 2/8，必须与 launcher 参数一致。 |
| `$4` | 当前 DP rank | 传给 `--data-parallel-rank`，P 还用它计算 KV 端口。 |
| `$5` | DP 主地址 | 来源脚本显式固定为本机 P/D 地址，必须与参数表一致。 |
| `$6` | DP RPC 端口 | 来源脚本显式固定为 12345/13345，必须与参数表一致。 |
| `$7` | TP size | 来源脚本显式固定为 8/2，必须与参数表一致。 |

因此改变拓扑时，必须联动核对 P/D 单 rank 脚本和两端 connector 的 prefill/decode 配置，不能只修改 launcher 命令。仓库通用模板仅用于理解参数接口，不直接代替本场景的完整 P/D 启动参数。

### 3.3 日志和执行状态

该仓库版本没有 `--log-prefix`、`--follow-logs`、`--no-follow-logs` 或 `--stagger-sec`，不能沿用旧任务定制版的这些参数。04 使用外层 `nohup` 重定向，将本机各 rank 输出汇集到一个 launcher 日志；它连续启动本机各 rank，不提供旧版的逐 rank 文件和错峰选项。更换为仓库 launcher 后的流程尚未在目标机器复验。

它只等待子进程退出，不检查所有 API 是否就绪，也不汇总失败为可靠的整体退出状态。按 04 检查全部 P/D 端口，停止时核对 launcher、rank 和 worker 的进程树。