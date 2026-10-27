# 03 — 验证脚本与配置全文归档

归档日期：2026-10-24。按用途区分配置/服务、测试、agent辅助；附录保存选定版本的全文，不执行其中的服务、清理、补丁或远程动作。所有实际验证范围以 [05](05_verification_process_and_results.md) 为准。

## 1. 清单与状态

正文清单逐项指向文末附录。SHA256以原脚本文件字节计算；附录只按Markdown换行规范嵌入，检查时统一CRLF/LF再逐字比较。每项源文件仍可从任务03目录直接取得。

### 配置与拉起服务

| 编号 | 文件 | 用途与状态 | 全文 |
|---|---|---|---|
| C1 | [mmc-meta.conf](../03_verification_scripts/config/repro_133_135/mmc-meta.conf) | 133/135 Meta 配置；原文保留，未证明该环境已运行。 | [C1](#src-c1) |
| C2 | [mmc-local.conf](../03_verification_scripts/config/repro_133_135/mmc-local.conf) | 133/135 engine proxy 配置（DRAM=0GB）；原文保留。 | [C2](#src-c2) |
| C3 | [mmc-local-standalone.conf](../03_verification_scripts/config/repro_133_135/mmc-local-standalone.conf) | 133/135 standalone store 配置（20GB）；原文保留。 | [C3](#src-c3) |
| C4 | [aisbench_config.py](../03_verification_scripts/config/repro_133_135/aisbench_config.py) | AISBench 工具 config.py 候选；需复制到工具根，工具精确版本未固定。 | [C4](#src-c4) |
| C5 | [smoke_req.json](../03_verification_scripts/config/smoke_req.json) | 129/130 历史冒烟请求 JSON；复用前核对模型名。 | [C5](#src-c5) |
| L1 | [meta.sh](../03_verification_scripts/launch/repro_133_135/meta.sh) | 135 Meta，整理版；环境变量后单条命令，未重新 NPU 验证。 | [L1](#src-l1) |
| L2 | [standalone.sh](../03_verification_scripts/launch/repro_133_135/standalone.sh) | 133、135 各一个 standalone store，整理版；未重新 NPU 验证。 | [L2](#src-l2) |
| L3 | [memcache_standalone.py](../03_verification_scripts/launch/repro_133_135/memcache_standalone.py) | standalone 的 Python 主体，原文保留，需 memcache_hybrid。 | [L3](#src-l3) |
| L4 | [p.sh](../03_verification_scripts/launch/repro_133_135/p.sh) | 133 P：DP4×TP4，整理版；不同于129的DP4×PCP4。 | [L4](#src-l4) |
| L5 | [d.sh](../03_verification_scripts/launch/repro_133_135/d.sh) | 135 D：DP16×TP1，整理版；未重新 NPU 验证。 | [L5](#src-l5) |
| L6 | [proxy.sh](../03_verification_scripts/launch/repro_133_135/proxy.sh) | 133代理，整理版；依赖未随任务取得的外部代理Python文件。 | [L6](#src-l6) |

### 测试相关

| 编号 | 文件 | 用途与状态 | 全文 |
|---|---|---|---|
| T1 | [soak_aisbench_prefix_loop.sh](../03_verification_scripts/tests/soak_aisbench_prefix_loop.sh) | 129/130 10-23实际循环测试的留存脚本；含专用目录数据/输出清理，本次未运行。 | [T1](#src-t1) |
| T2 | [soak_mrv2_109k_10h.sh](../03_verification_scripts/tests/soak_mrv2_109k_10h.sh) | 10-24定向soak留存版本；不等同所有历史采样时刻的参数，本次未运行。 | [T2](#src-t2) |
| T3 | [precision_check.sh](../03_verification_scripts/tests/precision_check.sh) | 历史精度两请求脚本；报告记载运行，原响应JSON缺失，本次未运行。 | [T3](#src-t3) |
| T4 | [run_case_1m.sh](../03_verification_scripts/tests/repro_133_135/run_case_1m.sh) | 133/135 1M复现候选；需放入AISBench工具目录，未留存成功运行证据。 | [T4](#src-t4) |
| T5 | [run_case_1m_c1024.sh](../03_verification_scripts/tests/repro_133_135/run_case_1m_c1024.sh) | 133/135并发1024候选；需放入AISBench工具目录，未留存成功运行证据。 | [T5](#src-t5) |
| T6 | [run_case_c6.sh](../03_verification_scripts/tests/repro_133_135/run_case_c6.sh) | 133/135 c6候选；注释中的一分钟是历史推断，非本次验证。 | [T6](#src-t6) |
| T7 | [smoke_test.sh](../03_verification_scripts/tests/repro_133_135/smoke_test.sh) | 133:8100冒烟候选；curl输出不等于完整功能判定。 | [T7](#src-t7) |
| T8 | [pref_test.sh](../03_verification_scripts/tests/history/pref_test.sh) | 历史九用例原文；10-23循环只选前七条，不代表九条全部运行。 | [T8](#src-t8) |
| T9 | [analyze_soak_summary.ps1](../03_verification_scripts/tests/analyze_soak_summary.ps1) | 只读TSV统计；本次已在Windows PowerShell执行并核对summary.json。 | [T9](#src-t9) |

### agent 专用（按需辅助说明）

仅保留必要的[诊断探针说明](#agent-probe)和[健康检查说明](#agent-health)，原 A1～A6 不再作为全文附录。一次性补丁工具见 [02 的历史辅助工具](02_development_and_pr.md#6-历史辅助工具)，材料迁移与选证工具见 [本次文档修订记录](evidence/20261026_agent_section_revision/README.md)。原脚本保留，通过链接追溯。

## 2. 调用与依赖

- E1（129/130）平台启动器全文缺失；04给出已留存调用记录，不伪造可运行启动脚本。E1参数节录见本文件的记录附录。
- E2（133/135）配置C1～C4、服务L1～L6为另一组复现材料，不能当作E1的同环境对照。五个shell服务入口仅保留环境设置与一条启动命令，取消脚本内cd/tee；原版本及SHA256在data的original_documents中。整理版未重新NPU复验。
- 运行L1/L2/L4/L5前，由调用者设置KVPOOL_CONFIG_DIR；L2还需要KVPOOL_LAUNCH_DIR。目录创建、复制和日志重定向放在04前置步骤。
- T1/T2在129容器的/home/aisbench_prefix运行，参数SOAK_HOURS、CASE_TIMEOUT，固定P日志位置；包含重建数据/清理输出，使用前确认专用目录。本次未运行。
- T4～T6以自身目录寻找aisbench_test.py，因此将选定入口复制到已安装的工具根再执行。工具config.py取C4，工具精确版本未随任务留存。
- T3比较两次响应，不自动证明第二次命中池；需结合服务侧hit/need-to-load证据，且缺失choices时不可仅凭两个空值相等判PASS。
- T9为本次实际复算入口，既不访问网络也不接触图片：

```powershell
& .\03_verification_scripts\tests\analyze_soak_summary.ps1 -Path .\archive\evidence\20261023_soak\soak_summary_20261023_031621.tsv
```

历史T1示例为SOAK_HOURS=20、CASE_TIMEOUT=3600，但实际留存时长11h32m。T2默认10h是脚本配置，不是已完成10h的证据。

## 3. 未选入正式入口的历史材料

judge_gva_coverage.sh仍为历史诊断脚本，原文存在grep -F/-E组合及基于早期假设的提示，不能当作当前权威判定；fix_pr19533_full_block_ids_source.sh对应一次已回滚的错误取字段尝试，不作为正式修复入口。其余容器创建、安装和一次性缩进工具仍在agent历史目录，不因留存就宣称已验证。

归档03保留配置、启动与测试部分的20项全文；agent 部分按需简述，不再嵌入原 A1～A6 的脚本。冻结代码、完整第三方源码、大日志与压缩包均不放入本附录。正式全文与历史辅助脚本分别由对应清单追溯。

## 4. 归档版本核对与恢复

[script_manifest.json](evidence/20261024_local_archive/script_manifest.json)只登记20项正式全文的编号、源路径、大小与SHA256。[historical_agent_scripts.json](evidence/20261024_local_archive/historical_agent_scripts.json)保留原 A1～A6 的身份与哈希，明确标为链接追溯，不再指向已移除的全文锚点。补丁及归档工具的使用边界见 02 和本次修订记录。

本次未改变服务器，因此没有现场回退动作。后续修改正文或脚本后需重新核对附录，并将实际运行版本另存轮次。

## 附录：选定文本全文

<a id="part-config-launch"></a>

## 配置与拉起服务

<a id="src-c1"></a>

### C1 — mmc-meta.conf

源文件：[03_verification_scripts/config/repro_133_135/mmc-meta.conf](../03_verification_scripts/config/repro_133_135/mmc-meta.conf)。状态：133/135 Meta 配置；原文保留，未证明该环境已运行。

SHA256：`3D12691DAB88CE7504A3C393AA692925C4764FCCB45B55F4C229466F980A0CC9`；195 字节。

```ini
ock.mmc.meta_service_url = tcp://80.5.9.135:5000
ock.mmc.meta_service.config_store_url = tcp://80.5.9.135:6000
ock.mmc.meta_service.metrics_url = http://80.5.9.135:8004
ock.mmc.log_level = error
```

<a id="src-c2"></a>

### C2 — mmc-local.conf

源文件：[03_verification_scripts/config/repro_133_135/mmc-local.conf](../03_verification_scripts/config/repro_133_135/mmc-local.conf)。状态：133/135 engine proxy 配置（DRAM=0GB）；原文保留。

SHA256：`47AA7C5E0AF75194302664790DFBC19957F6FE326B4CDC2ED267D578A7A9EB9F`；305 字节。

```ini
ock.mmc.meta_service_url = tcp://80.5.9.135:5000
ock.mmc.local_service.config_store_url = tcp://80.5.9.135:6000
ock.mmc.log_level = error
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 0GB
ock.mmc.local_service.max.dram.size = 1024GB
```

<a id="src-c3"></a>

### C3 — mmc-local-standalone.conf

源文件：[03_verification_scripts/config/repro_133_135/mmc-local-standalone.conf](../03_verification_scripts/config/repro_133_135/mmc-local-standalone.conf)。状态：133/135 standalone store 配置（20GB）；原文保留。

SHA256：`6D0D238E875BB385BCB0865F1611FC8892651D047FABE8CE37D68C63AE61A11D`；306 字节。

```ini
ock.mmc.meta_service_url = tcp://80.5.9.135:5000
ock.mmc.local_service.config_store_url = tcp://80.5.9.135:6000
ock.mmc.log_level = error
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 20GB
ock.mmc.local_service.max.dram.size = 1024GB
```

<a id="src-c4"></a>

### C4 — aisbench_config.py

源文件：[03_verification_scripts/config/repro_133_135/aisbench_config.py](../03_verification_scripts/config/repro_133_135/aisbench_config.py)。状态：AISBench 工具 config.py 候选；需复制到工具根，工具精确版本未固定。

SHA256：`6767E83F549FD0456EB782E16047887A153FD99116669D74FA3E8589ECE17CD7`；409 字节。

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_PATH = str(ROOT / 'dataset')
OUTPUT_DIR = str(ROOT / 'outputs')
WORK_PATH = '/vllm-workspace/vllm-ascend/benchmark'
MODEL_NAME = 'model'
MODEL_PATH = '/mnt/share/weights/DeepSeek-V4-Flash-0731-w8a8'
HOST_IP = '80.5.9.133'
HOST_PORT = '8100'
POD_INFO = ['80.5.9.133:8910']
URL = ''
API_KEY = ''
DEFAULT_PERFORMANCE_TEST = 'default_perf'
```

<a id="src-c5"></a>

### C5 — smoke_req.json

源文件：[03_verification_scripts/config/smoke_req.json](../03_verification_scripts/config/smoke_req.json)。状态：129/130 历史冒烟请求 JSON；复用前核对模型名。

SHA256：`745E247B4A1F7381569E7D22D998D27BA230DBE820055E9EA7318C0EBAE7F8BE`；81 字节。

```json
{"model":"dsv4","messages":[{"role":"user","content":"say hi"}],"max_tokens":10}
```

<a id="src-l1"></a>

### L1 — meta.sh

源文件：[03_verification_scripts/launch/repro_133_135/meta.sh](../03_verification_scripts/launch/repro_133_135/meta.sh)。状态：135 Meta，整理版；环境变量后单条命令，未重新 NPU 验证。

SHA256：`A518FB3C27E9134584DB734DA0B9C723892E9BBD8F82A391B4339AF560CA1A00`；359 字节。

```bash
#!/usr/bin/env bash
# Archive layout adaptation; not revalidated on NPU. Set KVPOOL_CONFIG_DIR/KVPOOL_LAUNCH_DIR as needed.
# Meta 服务（80.5.9.135）
source /usr/local/Ascend/cann/set_env.sh
export MMC_META_CONFIG_PATH="${KVPOOL_CONFIG_DIR:?Set KVPOOL_CONFIG_DIR}/mmc-meta.conf"
python3 -u -c 'from memcache_hybrid import MetaService; MetaService.main()'
```

<a id="src-l2"></a>

### L2 — standalone.sh

源文件：[03_verification_scripts/launch/repro_133_135/standalone.sh](../03_verification_scripts/launch/repro_133_135/standalone.sh)。状态：133、135 各一个 standalone store，整理版；未重新 NPU 验证。

SHA256：`FC93C373B10AD859487CF24BA65DAFCC6BA20EB42C3E4014F5A72BD8BF9EE6DC`；435 字节。

```bash
#!/usr/bin/env bash
# Archive layout adaptation; not revalidated on NPU. Set KVPOOL_CONFIG_DIR/KVPOOL_LAUNCH_DIR as needed.
# 本地池服务（133、135 各一个，各 20GB）
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MMC_LOCAL_CONFIG_PATH="${KVPOOL_CONFIG_DIR:?Set KVPOOL_CONFIG_DIR}/mmc-local-standalone.conf"
python3 -u "${KVPOOL_LAUNCH_DIR:?Set KVPOOL_LAUNCH_DIR}/memcache_standalone.py"
```

<a id="src-l3"></a>

### L3 — memcache_standalone.py

源文件：[03_verification_scripts/launch/repro_133_135/memcache_standalone.py](../03_verification_scripts/launch/repro_133_135/memcache_standalone.py)。状态：standalone 的 Python 主体，原文保留，需 memcache_hybrid。

SHA256：`23400DB0FA7EC69BA005728E07183C28F64CAF0813DCD55920207E5DE593C7B4`；350 字节。

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

<a id="src-l4"></a>

### L4 — p.sh

源文件：[03_verification_scripts/launch/repro_133_135/p.sh](../03_verification_scripts/launch/repro_133_135/p.sh)。状态：133 P：DP4×TP4，整理版；不同于129的DP4×PCP4。

SHA256：`36045D77173D64F140292E378A87C8957FC3A3134E90149068A86403CE26F023`；2784 字节。

```bash
#!/usr/bin/env bash
# Archive layout adaptation; not revalidated on NPU. Set KVPOOL_CONFIG_DIR/KVPOOL_LAUNCH_DIR as needed.
# P 侧（80.5.9.133）prefill / kv_producer，4 DP x 4 TP，layerwise 存池开启
source /usr/local/Ascend/cann/set_env.sh

export PYTHONHASHSEED=0
export MMC_LOCAL_CONFIG_PATH="${KVPOOL_CONFIG_DIR:?Set KVPOOL_CONFIG_DIR}/mmc-local.conf"

unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

nic_name=enp194s0f0
local_ip=80.5.9.133

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
  --max-num-seqs 32 \
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
    '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"30000","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":16,"tp_size":1}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"memcache","use_layerwise":true,"layerwise_prefetch_layers":3}}]}}' 2>&1
```

<a id="src-l5"></a>

### L5 — d.sh

源文件：[03_verification_scripts/launch/repro_133_135/d.sh](../03_verification_scripts/launch/repro_133_135/d.sh)。状态：135 D：DP16×TP1，整理版；未重新 NPU 验证。

SHA256：`CFF941E8AD621CE3A45152F45B9D876D42A58450BDC2CF17322C3918AA7FB2C6`；2909 字节。

```bash
#!/usr/bin/env bash
# Archive layout adaptation; not revalidated on NPU. Set KVPOOL_CONFIG_DIR/KVPOOL_LAUNCH_DIR as needed.
# D 侧（80.5.9.135）decode / kv_consumer，16 DP x 1 TP
source /usr/local/Ascend/cann/set_env.sh

export PYTHONHASHSEED=0
export MMC_LOCAL_CONFIG_PATH="${KVPOOL_CONFIG_DIR:?Set KVPOOL_CONFIG_DIR}/mmc-local.conf"

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
# 16 DP 权重加载 + torch.compile/图捕获远超默认 600s，必须放大否则前端等不到 engine core 就绪
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

nic_name=enp194s0f0
local_ip=80.5.9.135

export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1800
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
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
    --kv-transfer-config \
    '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"30200","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":16,"tp_size":1}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"memcache","use_layerwise":false}}]}}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":false}' 2>&1
```

<a id="src-l6"></a>

### L6 — proxy.sh

源文件：[03_verification_scripts/launch/repro_133_135/proxy.sh](../03_verification_scripts/launch/repro_133_135/proxy.sh)。状态：133代理，整理版；依赖未随任务取得的外部代理Python文件。

SHA256：`559A74F9F7C8FA06893D5FB1ACBEA20B9933D45365CD50CDDA5B4F903EA30815`；510 字节。

```bash
#!/usr/bin/env bash
# Archive layout adaptation; not revalidated on NPU. Set KVPOOL_CONFIG_DIR/KVPOOL_LAUNCH_DIR as needed.
# 负载均衡代理（80.5.9.133:8100 -> prefill 133:8910 / decode 135:8920）
source /usr/local/Ascend/cann/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 /home/w00985460/script/vllm/load_balance_proxy_server_example.py \
    --prefiller-hosts 80.5.9.133 --prefiller-ports 8910 \
    --decoder-hosts 80.5.9.135 --decoder-ports 8920 \
    --host 80.5.9.133 --port 8100 2>&1
```

<a id="part-tests"></a>

## 测试相关

<a id="src-t1"></a>

### T1 — soak_aisbench_prefix_loop.sh

源文件：[03_verification_scripts/tests/soak_aisbench_prefix_loop.sh](../03_verification_scripts/tests/soak_aisbench_prefix_loop.sh)。状态：129/130 10-23实际循环测试的留存脚本；含专用目录数据/输出清理，本次未运行。

SHA256：`F03E4103E655911034860C33CB34F8746742599B9C4B3AC435276DEF24397535`；5620 字节。

```bash
#!/usr/bin/env bash
# aisbench prefix_cache 长稳压测循环（在 P 容器内运行）
#
#   SOAK_HOURS    总时长小时数（默认 20）
#   CASE_TIMEOUT  单用例超时秒数（默认 3600）
#
# 每用例：重建数据集（数据集名不含 seed/repeat/prefix，会串味）→ 采指标基线 → 跑 →
#         采增量 → 记 TSV 一行 → 清 outputs（单次 1M 用例约 2.7G）
# 服务不可达时等待恢复（最多 30 分钟/用例），不重启服务。
#
# 用例集说明：pref_test.sh 的前 7 条。第 8/9 条（output_len 512000 / 1024000）是
# 长生成用例，单条就可能超过 7 小时、会吃光时长预算，且与池化无关，故未纳入；
# 需要时把对应行加回 CASES 即可。
set -u

TOOL=/home/aisbench_prefix
DS=$TOOL/dataset
P0_LOG=/tmp/inference/09750b07-1a8e-41f3-b3b1-c772992cb403/p0_0/output.log
PORTS="8000 8001 8002 8003"

HOURS=${SOAK_HOURS:-20}
CASE_TIMEOUT=${CASE_TIMEOUT:-3600}
TS=$(date +%Y%m%d_%H%M%S)
MAIN=$TOOL/soak_${TS}.log
SUM=$TOOL/soak_summary_${TS}.tsv
HEART=$TOOL/soak_heartbeat_${TS}.txt

cd "$TOOL" || exit 1
exec >>"$MAIN" 2>&1

CASES=(
  "--input_len 128 --output_len 128 --data_num 40960 --concurrency 4096 --request_rate 0"
  "--input_len 512 --output_len 512 --data_num 16384 --concurrency 4096 --request_rate 0"
  "--input_len 1024000 --output_len 1024 --data_num 640 --concurrency 128 --request_rate 0 --dataset_type prefix_cache --repeat_rate 99% --seed 200 --prefix_num 1 --prefix_test --dp 4"
  "--input_len 1024000 --output_len 1024 --data_num 2048 --concurrency 1024 --request_rate 0 --dataset_type prefix_cache --repeat_rate 99% --seed 200 --prefix_num 1 --prefix_test --dp 4"
  "--input_len 1024000 --output_len 1024 --data_num 512 --concurrency 128 --request_rate 0 --dataset_type prefix_cache --repeat_rate 0% --seed 200 --prefix_num 1 --prefix_test --dp 4"
  "--input_len 1024000 --output_len 1024 --data_num 640 --concurrency 128 --request_rate 0 --dataset_type prefix_cache --repeat_rate 90% --seed 200 --prefix_num 16 --prefix_test --dp 4"
  "--input_len 1024000 --output_len 1024 --data_num 640 --concurrency 128 --request_rate 0 --dataset_type prefix_cache --repeat_rate 90% --seed 200 --prefix_num 32 --prefix_test --dp 4"
)

END=$(( $(date +%s) + HOURS * 3600 ))
printf 'round\tcase\texit\tdur_s\tfail\ttotal_req\text_hit\text_query\text_rate\tlocal_hit\tlocal_query\tskip_d\tskip_window_d\tskip_unprov_d\tfatal_d\thit_events_d\tat\n' >"$SUM"
echo "$(date) SOAK START hours=$HOURS case_timeout=$CASE_TIMEOUT end_epoch=$END" >>"$HEART"

metric_sum() {
  local k=$1 p v tot=0
  for p in $PORTS; do
    v=$(curl -s -m 5 "http://127.0.0.1:$p/metrics" | awk -v k="$k" '{n=$1; sub(/\{.*/,"",n); if(n==k) s+=$NF} END{printf "%.0f", s+0}')
    tot=$((tot + ${v:-0}))
  done
  echo "$tot"
}

wait_healthy() {
  local n=0
  while [ "$n" -lt 30 ]; do
    [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = "200" ] && return 0
    sleep 60
    n=$((n + 1))
  done
  return 1
}

round=0
while [ "$(date +%s)" -lt "$END" ]; do
  round=$((round + 1))
  idx=0
  for c in "${CASES[@]}"; do
    idx=$((idx + 1))
    [ "$(date +%s)" -ge "$END" ] && break
    read -r -a a <<<"$c"

    s0=$(grep -c -F 'layerwise save skipped' "$P0_LOG")
    w0=$(grep -c -F 'mode=WINDOW' "$P0_LOG")
    u0=$(grep -c -F 'mode=UNPROVABLE' "$P0_LOG")
    f0=$(grep -c -F 'does not cover requested block' "$P0_LOG")
    h0=$(grep -c -F 'kvpool hit tokens' "$P0_LOG")
    eh0=$(metric_sum vllm:external_prefix_cache_hits_total)
    eq0=$(metric_sum vllm:external_prefix_cache_queries_total)
    lh0=$(metric_sum vllm:prefix_cache_hits_total)
    lq0=$(metric_sum vllm:prefix_cache_queries_total)

    rm -f "$DS"/*.jsonl "$TOOL/picked_ids.txt"

    if ! wait_healthy; then
      printf '%s\t%s\tSVC_DOWN\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t%s\n' \
        "$round" "$idx" "$(date +%H:%M:%S)" >>"$SUM"
      echo "$(date) round=$round case=$idx SERVICE NOT HEALTHY after wait" >>"$HEART"
      continue
    fi

    L=$TOOL/case_${TS}_r${round}_c${idx}.log
    t0=$(date +%s)
    timeout "$CASE_TIMEOUT" python3 aisbench_test.py "${a[@]}" >"$L" 2>&1
    ec=$?
    dur=$(( $(date +%s) - t0 ))

    s1=$(grep -c -F 'layerwise save skipped' "$P0_LOG")
    w1=$(grep -c -F 'mode=WINDOW' "$P0_LOG")
    u1=$(grep -c -F 'mode=UNPROVABLE' "$P0_LOG")
    f1=$(grep -c -F 'does not cover requested block' "$P0_LOG")
    h1=$(grep -c -F 'kvpool hit tokens' "$P0_LOG")
    eh1=$(metric_sum vllm:external_prefix_cache_hits_total)
    eq1=$(metric_sum vllm:external_prefix_cache_queries_total)
    lh1=$(metric_sum vllm:prefix_cache_hits_total)
    lq1=$(metric_sum vllm:prefix_cache_queries_total)

    fail=$(grep -o -E 'Failed Requests[^0-9]*[0-9]+' "$L" | tail -n 1 | grep -o -E '[0-9]+' | tail -n 1)
    treq=$(tail -n 1 "$TOOL/aisbench_result.csv" | cut -d, -f5)
    rate=$(awk -v h=$((eh1 - eh0)) -v q=$((eq1 - eq0)) 'BEGIN{if(q>0) printf "%.2f%%", 100*h/q; else printf "NA"}')

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$round" "$idx" "$ec" "$dur" "${fail:-NA}" "${treq:-NA}" \
      "$((eh1 - eh0))" "$((eq1 - eq0))" "$rate" \
      "$((lh1 - lh0))" "$((lq1 - lq0))" \
      "$((s1 - s0))" "$((w1 - w0))" "$((u1 - u0))" "$((f1 - f0))" "$((h1 - h0))" \
      "$(date +%H:%M:%S)" >>"$SUM"

    echo "$(date) round=$round case=$idx exit=$ec dur=${dur}s fail=${fail:-NA} skip_d=$((s1 - s0))(w$((w1 - w0))/u$((u1 - u0))) fatal_d=$((f1 - f0)) ext_rate=$rate" >>"$HEART"

    rm -rf "$TOOL/outputs/default"/*
  done
done

echo "$(date) SOAK END rounds=$round" >>"$HEART"
```

<a id="src-t2"></a>

### T2 — soak_mrv2_109k_10h.sh

源文件：[03_verification_scripts/tests/soak_mrv2_109k_10h.sh](../03_verification_scripts/tests/soak_mrv2_109k_10h.sh)。状态：10-24定向soak留存版本；不等同所有历史采样时刻的参数，本次未运行。

SHA256：`1FD42173C2D53DC5052613697DE703970D2B6999AE2FFFC440E3BADC2C77892D`；4971 字节。

```bash
#!/usr/bin/env bash
# 10h 定向 soak：以 109k 长上下文为主，穿插 1M / 短请求，循环压测。
# 目的：逼出 RuntimeError: ReqMeta GVA metadata does not cover requested block range
#
#   SOAK_HOURS    总时长小时数（默认 10）
#   CASE_TIMEOUT  单用例超时秒数（默认 3600）
#
# 每用例：重建数据集 → 采指标基线 → 跑 → 采增量 → 记 TSV → 清 outputs。
# 服务不可达时等待恢复（最多 30 分钟/用例），不重启服务。
set -u

TOOL=/home/aisbench_prefix
DS=$TOOL/dataset
P0_LOG=/tmp/p_restart_b113.log
PORTS="8000 8001 8002 8003"

HOURS=${SOAK_HOURS:-10}
CASE_TIMEOUT=${CASE_TIMEOUT:-3600}
TS=$(date +%Y%m%d_%H%M%S)
MAIN=$TOOL/soak_109k_${TS}.log
SUM=$TOOL/soak_109k_summary_${TS}.tsv
HEART=$TOOL/soak_109k_heartbeat_${TS}.txt

cd "$TOOL" || exit 1
exec >>"$MAIN" 2>&1

# 极端组合：先 1M 高 KV 压力源，再 109k 低等待目标 + 短请求填充
CASES=(
  "--input_len 1024000 --output_len 1024 --data_num 640 --concurrency 192 --request_rate 0 --dataset_type prefix_cache --repeat_rate 99% --seed 200 --prefix_num 1 --prefix_test --dp 4"
  "--input_len 109000 --output_len 1024 --data_num 4096 --concurrency 64 --request_rate 0 --dataset_type prefix_cache --repeat_rate 99% --seed 200 --prefix_num 1 --prefix_test --dp 4"
  "--input_len 109000 --output_len 1024 --data_num 2048 --concurrency 32 --request_rate 0 --dataset_type prefix_cache --repeat_rate 99% --seed 200 --prefix_num 1 --prefix_test --dp 4"
  "--input_len 128 --output_len 128 --data_num 40960 --concurrency 4096 --request_rate 0"
)

END=$(( $(date +%s) + HOURS * 3600 ))
printf 'round\tcase\texit\tdur_s\tfail\ttotal_req\text_hit\text_query\text_rate\tlocal_hit\tlocal_query\tskip_d\tskip_window_d\tskip_unprov_d\tfatal_d\thit_events_d\tat\n' >"$SUM"
echo "$(date) SOAK START hours=$HOURS case_timeout=$CASE_TIMEOUT end_epoch=$END" >>"$HEART"

metric_sum() {
  local k=$1 p v tot=0
  for p in $PORTS; do
    v=$(curl -s -m 5 "http://127.0.0.1:$p/metrics" | awk -v k="$k" '{n=$1; sub(/\{.*/,"",n); if(n==k) s+=$NF} END{printf "%.0f", s+0}')
    tot=$((tot + ${v:-0}))
  done
  echo "$tot"
}

wait_healthy() {
  local n=0
  while [ "$n" -lt 30 ]; do
    [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = "200" ] && return 0
    sleep 60
    n=$((n + 1))
  done
  return 1
}

round=0
while [ "$(date +%s)" -lt "$END" ]; do
  round=$((round + 1))
  idx=0
  for c in "${CASES[@]}"; do
    idx=$((idx + 1))
    [ "$(date +%s)" -ge "$END" ] && break
    read -r -a a <<<"$c"

    s0=$(grep -c -F 'layerwise save skipped' "$P0_LOG")
    w0=$(grep -c -F 'mode=WINDOW' "$P0_LOG")
    u0=$(grep -c -F 'mode=UNPROVABLE' "$P0_LOG")
    f0=$(grep -c -F 'does not cover requested block' "$P0_LOG")
    h0=$(grep -c -F 'kvpool hit tokens' "$P0_LOG")
    eh0=$(metric_sum vllm:external_prefix_cache_hits_total)
    eq0=$(metric_sum vllm:external_prefix_cache_queries_total)
    lh0=$(metric_sum vllm:prefix_cache_hits_total)
    lq0=$(metric_sum vllm:prefix_cache_queries_total)

    rm -f "$DS"/*.jsonl "$TOOL/picked_ids.txt"

    if ! wait_healthy; then
      printf '%s\t%s\tSVC_DOWN\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t%s\n' \
        "$round" "$idx" "$(date +%H:%M:%S)" >>"$SUM"
      echo "$(date) round=$round case=$idx SERVICE NOT HEALTHY after wait" >>"$HEART"
      continue
    fi

    L=$TOOL/case_109k_${TS}_r${round}_c${idx}.log
    t0=$(date +%s)
    timeout "$CASE_TIMEOUT" python3 aisbench_test.py "${a[@]}" >"$L" 2>&1
    ec=$?
    dur=$(( $(date +%s) - t0 ))

    s1=$(grep -c -F 'layerwise save skipped' "$P0_LOG")
    w1=$(grep -c -F 'mode=WINDOW' "$P0_LOG")
    u1=$(grep -c -F 'mode=UNPROVABLE' "$P0_LOG")
    f1=$(grep -c -F 'does not cover requested block' "$P0_LOG")
    h1=$(grep -c -F 'kvpool hit tokens' "$P0_LOG")
    eh1=$(metric_sum vllm:external_prefix_cache_hits_total)
    eq1=$(metric_sum vllm:external_prefix_cache_queries_total)
    lh1=$(metric_sum vllm:prefix_cache_hits_total)
    lq1=$(metric_sum vllm:prefix_cache_queries_total)

    fail=$(grep -o -E 'Failed Requests[^0-9]*[0-9]+' "$L" | tail -n 1 | grep -o -E '[0-9]+' | tail -n 1)
    treq=$(tail -n 1 "$TOOL/aisbench_result.csv" | cut -d, -f5)
    rate=$(awk -v h=$((eh1 - eh0)) -v q=$((eq1 - eq0)) 'BEGIN{if(q>0) printf "%.2f%%", 100*h/q; else printf "NA"}')

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$round" "$idx" "$ec" "$dur" "${fail:-NA}" "${treq:-NA}" \
      "$((eh1 - eh0))" "$((eq1 - eq0))" "$rate" \
      "$((lh1 - lh0))" "$((lq1 - lq0))" \
      "$((s1 - s0))" "$((w1 - w0))" "$((u1 - u0))" "$((f1 - f0))" "$((h1 - h0))" \
      "$(date +%H:%M:%S)" >>"$SUM"

    echo "$(date) round=$round case=$idx exit=$ec dur=${dur}s fail=${fail:-NA} skip_d=$((s1 - s0))(w$((w1 - w0))/u$((u1 - u0))) fatal_d=$((f1 - f0)) ext_rate=$rate" >>"$HEART"

    rm -rf "$TOOL/outputs/default"/*
  done
done

echo "$(date) SOAK END rounds=$round" >>"$HEART"
```

<a id="src-t3"></a>

### T3 — precision_check.sh

源文件：[03_verification_scripts/tests/precision_check.sh](../03_verification_scripts/tests/precision_check.sh)。状态：历史精度两请求脚本；报告记载运行，原响应JSON缺失，本次未运行。

SHA256：`C95DF57261E077A62A302122DFB957CDEAF1EAA53DBD3A7A6ED535D326198DAE`；2035 字节。

```bash
#!/bin/bash
# 精度验证：同一 prompt 连发两次，第二次应命中池，比对输出是否逐字一致
set -u
WORK=/tmp/precision_check_$(date +%H%M%S)
mkdir -p "$WORK"

python3 - "$WORK/body.json" <<'PYEOF'
import json, sys
sentence = "The quick brown fox jumps over the lazy dog near the river bank at dawn. "
shared = sentence * 1200          # ~16k token 共享前缀，保证跨 4096 粒度
question = "\n\nQuestion: What is 27 multiplied by 43? Answer with only the final integer."
body = {
    "model": "dsv4",
    "messages": [{"role": "user", "content": shared + question}],
    "max_tokens": 300,
    "temperature": 0.0,
}
json.dump(body, open(sys.argv[1], "w"))
PYEOF

URL="http://80.48.29.129:1999/v1/chat/completions"
echo "[$(date +%H:%M:%S)] sending request #1 ..."
curl -s --max-time 600 -X POST "$URL" -H 'Content-Type: application/json' -d @"$WORK/body.json" > "$WORK/resp1.json"
echo "[$(date +%H:%M:%S)] sending request #2 (should hit pool) ..."
curl -s --max-time 600 -X POST "$URL" -H 'Content-Type: application/json' -d @"$WORK/body.json" > "$WORK/resp2.json"

python3 - "$WORK" <<'PYEOF'
import json, sys, hashlib
work = sys.argv[1]
r1 = json.load(open(f"{work}/resp1.json"))
r2 = json.load(open(f"{work}/resp2.json"))
def pick(r):
    c = (r.get("choices") or [{}])[0].get("message", {})
    return (c.get("content"), c.get("reasoning"))
t1, t2 = pick(r1), pick(r2)
print("id1:", r1.get("id"))
print("id2:", r2.get("id"))
print("usage1:", r1.get("usage"))
print("usage2:", r2.get("usage"))
h1 = [hashlib.sha1((x or "").encode()).hexdigest()[:12] for x in t1]
h2 = [hashlib.sha1((x or "").encode()).hexdigest()[:12] for x in t2]
print("content_hash1:", h1[0], "reasoning_hash1:", h1[1])
print("content_hash2:", h2[0], "reasoning_hash2:", h2[1])
print("IDENTICAL" if t1 == t2 else "DIFFERENT")
print("answer1 (last 120 chars of reasoning):", (t1[1] or "")[-120:].replace(chr(10), " "))
print("answer2 (last 120 chars of reasoning):", (t2[1] or "")[-120:].replace(chr(10), " "))
PYEOF
echo "WORK=$WORK"
```

<a id="src-t4"></a>

### T4 — run_case_1m.sh

源文件：[03_verification_scripts/tests/repro_133_135/run_case_1m.sh](../03_verification_scripts/tests/repro_133_135/run_case_1m.sh)。状态：133/135 1M复现候选；需放入AISBench工具目录，未留存成功运行证据。

SHA256：`CE1E49DCE2CE0647FD1882AD389AEECBE86274F0985C7C2AB554B596FE8E187D`；546 字节。

```bash
#!/usr/bin/env bash
# 1M 输入前缀用例：目标是触发 P 侧抢占，从而暴露窗口块表存池 fatal
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python3 -u aisbench_test.py \
    --input_len 1000000 --output_len 1024 \
    --concurrency 128 --data_num 640 \
    --dataset_type prefix_cache --repeat_rate 0.99 --prefix_num 1 \
    --prefix_test --dp 4 --npu_num 16 \
    --request_rate 0 --seed 1
```

<a id="src-t5"></a>

### T5 — run_case_1m_c1024.sh

源文件：[03_verification_scripts/tests/repro_133_135/run_case_1m_c1024.sh](../03_verification_scripts/tests/repro_133_135/run_case_1m_c1024.sh)。状态：133/135并发1024候选；需放入AISBench工具目录，未留存成功运行证据。

SHA256：`0C7122E57B03AFB94487A1C7B9801F192D2167A0B1ABC0D5555D8B624D717BAD`；575 字节。

```bash
#!/usr/bin/env bash
# 1M 输入 + 1024 并发：高并发下 P 侧 KV 预算被打满，触发抢占重建窗口块表，暴露存池 fatal
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python3 -u aisbench_test.py \
    --input_len 1000000 --output_len 1024 \
    --concurrency 1024 --data_num 2048 \
    --dataset_type prefix_cache --repeat_rate 0.99 --prefix_num 1 \
    --prefix_test --dp 4 --npu_num 16 \
    --request_rate 0 --seed 1
```

<a id="src-t6"></a>

### T6 — run_case_c6.sh

源文件：[03_verification_scripts/tests/repro_133_135/run_case_c6.sh](../03_verification_scripts/tests/repro_133_135/run_case_c6.sh)。状态：133/135 c6候选；注释中的一分钟是历史推断，非本次验证。

SHA256：`D2D89DAD16BF1A91CAA890F0F27979272A68476DDC8EAA411936F1CAECB5BA94`；632 字节。

```bash
#!/usr/bin/env bash
# c6 复现用例：1M 输入 / 并发 128 / 90% 重复 / 16 前缀
# 历史统计：该组合抢占密度最高（单条 314 次），开测后 1 分钟内即可复现 GVA fatal
set -o pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
source /usr/local/Ascend/cann/set_env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python3 -u aisbench_test.py \
    --input_len 1000000 --output_len 1024 \
    --concurrency 128 --data_num 640 \
    --dataset_type prefix_cache --repeat_rate 0.9 --prefix_num 16 \
    --prefix_test --dp 4 --npu_num 16 \
    --request_rate 0 --seed 1
```

<a id="src-t7"></a>

### T7 — smoke_test.sh

源文件：[03_verification_scripts/tests/repro_133_135/smoke_test.sh](../03_verification_scripts/tests/repro_133_135/smoke_test.sh)。状态：133:8100冒烟候选；curl输出不等于完整功能判定。

SHA256：`6F0D315471AEBB0E3FB7F9BD2A8643110748C9E5C21F09DB81ACC1D5497F6707`；332 字节。

```bash
#!/usr/bin/env bash
# 冒烟：经 proxy 发一条短请求，确认 PD 链路与池化路径正常
curl -s -m 180 -X POST http://80.5.9.133:8100/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","prompt":"Hello, please introduce yourself briefly.","max_tokens":16,"temperature":0}' | head -c 800
echo
```

<a id="src-t8"></a>

### T8 — pref_test.sh

源文件：[03_verification_scripts/tests/history/pref_test.sh](../03_verification_scripts/tests/history/pref_test.sh)。状态：历史九用例原文；10-23循环只选前七条，不代表九条全部运行。

SHA256：`4F5B5519267B46BC73ECA80E2AA20FAAD6F2BC21341DD1AB2D41DF431315906F`；1582 字节。

```bash
python3 aisbench_test.py --input_len 128 --output_len 128 --data_num 40960 --concurrency 4096 --request_rate 0

python3 aisbench_test.py --input_len 512 --output_len 512 --data_num 16384 --concurrency 4096 --request_rate 0

python3 aisbench_test.py --input_len 1024000 --output_len 1024 --data_num 640 --concurrency 128 --request_rate 0 --dataset_type prefix_cache --repeat_rate 99% --seed 200 --prefix_num 1 --prefix_test --dp 4

python3 aisbench_test.py --input_len 1024000 --output_len 1024 --data_num 2048 --concurrency 1024 --request_rate 0 --dataset_type prefix_cache --repeat_rate 99% --seed 200 --prefix_num 1 --prefix_test --dp 4

python3 aisbench_test.py --input_len 1024000 --output_len 1024 --data_num 512 --concurrency 128 --request_rate 0 --dataset_type prefix_cache --repeat_rate 0% --seed 200 --prefix_num 1 --prefix_test --dp 4

python3 aisbench_test.py --input_len 1024000 --output_len 1024 --data_num 640 --concurrency 128 --request_rate 0 --dataset_type prefix_cache --repeat_rate 90% --seed 200 --prefix_num 16 --prefix_test --dp 4

python3 aisbench_test.py --input_len 1024000 --output_len 1024 --data_num 640 --concurrency 128 --request_rate 0 --dataset_type prefix_cache --repeat_rate 90% --seed 200 --prefix_num 32 --prefix_test --dp 4

python3 aisbench_test.py --input_len 32786 --output_len 512000 --data_num 256 --concurrency 128 --request_rate 0 --dataset_type prefix_cache --repeat_rate 90% --seed 200 --prefix_num 1 --prefix_test --dp 4

python3 aisbench_test.py --input_len 1024 --output_len 1024000 --data_num 128 --concurrency 32 --request_rate 0 
```

<a id="src-t9"></a>

### T9 — analyze_soak_summary.ps1

源文件：[03_verification_scripts/tests/analyze_soak_summary.ps1](../03_verification_scripts/tests/analyze_soak_summary.ps1)。状态：只读TSV统计；本次已在Windows PowerShell执行并核对summary.json。

SHA256：`84AC3D5EDF72EF742CD2E13D92942D573B59BB68675A2745E3B50E17EF6387C0`；1688 字节。

```powershell
# Read-only replay of archived aisbench TSV counters. No server or image access.
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$Path)
$ErrorActionPreference = 'Stop'
$rows = @(Import-Csv -LiteralPath $Path -Delimiter "`t")
if (-not $rows.Count) { throw 'Empty TSV' }
foreach ($column in @('round','case','exit','fail','skip_window_d','skip_unprov_d','fatal_d','ext_hit','ext_query')) {
    if ($rows[0].PSObject.Properties.Name -notcontains $column) { throw "Missing TSV column: $column" }
}
$rounds = @($rows | Group-Object round | ForEach-Object {
    [ordered]@{ round = $_.Name; cases = $_.Count
        window = ($_.Group | Measure-Object skip_window_d -Sum).Sum
        unprovable = ($_.Group | Measure-Object skip_unprov_d -Sum).Sum
        gva_fatal = ($_.Group | Measure-Object fatal_d -Sum).Sum }
})
[ordered]@{
    source_sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    cases = $rows.Count
    exit_0 = @($rows | Where-Object exit -eq '0').Count
    timeout_124 = @($rows | Where-Object exit -eq '124').Count
    window = ($rows | Measure-Object skip_window_d -Sum).Sum
    unprovable = ($rows | Measure-Object skip_unprov_d -Sum).Sum
    gva_fatal = ($rows | Measure-Object fatal_d -Sum).Sum
    known_failed_requests = (($rows | Where-Object fail -ne 'NA') | Measure-Object fail -Sum).Sum
    unknown_failure_cases = @($rows | Where-Object fail -eq 'NA').Count
    negative_external_counter_cases = @($rows | Where-Object { [double]$_.ext_hit -lt 0 -or [double]$_.ext_query -lt 0 }).Count
    rounds = $rounds
    scope = 'Aggregate service counters, includes external clients; exit=0 is not a PASS verdict.'
} | ConvertTo-Json -Depth 5
```

<a id="part-agent"></a>

## agent 专用（按需辅助说明）

<a id="agent-probe"></a>

### 诊断探针

在 V2 的 `_process_new_request` 清除 `_preempted_req_ids` 的分支，原工具插入 `logger.info_once("this if branch is ok")`，用于确认该进程执行了对应分支。常量消息每进程只打印一次；P 侧最多4条不能解释为4次抢占恢复，更不能替代同一请求的完整生命周期证据。

[历史探针工具原文](../03_verification_scripts/agent/add_probe_readmit_bugfix_mrv2.sh)继续留存。该工具会先 `git checkout --` 还原目标源码文件，可能丢弃其未提交修改；这里只说明观测位置和语义，不将它作为当前可直接执行的验证入口。

<a id="agent-health"></a>

### 健康检查

| 对象 | 检查范围 | 就绪条件 | 原工具重试方式 |
|---|---|---|---|
| P | `80.48.29.129:8000～8003/health` | 同一轮4个接口全部返回200 | 最多32轮，每轮后等待15秒；单次HTTP请求超时5秒 |
| D | `80.48.29.130:8000～8015/health` | 同一轮16个接口全部返回200 | 最多40轮，每轮后等待15秒；单次HTTP请求超时5秒 |

按上述要求轮询即可：全部就绪返回成功，重试耗尽则记录未就绪接口并停止后续测试，不自动重启服务。总耗时包含HTTP检查时间，不能将轮询间隔之和当成严格总超时。

原 [P 健康检查脚本](../03_verification_scripts/agent/wait_p_health.sh)和 [D 健康检查脚本](../03_verification_scripts/agent/wait_d_health.sh)作为历史参考保留。健康检查只证明接口可访问，还需 04 所述的推理冒烟及目标功能命中检查。

<a id="legacy-e1-parameters"></a>

## E1 平台调用与关键参数记录（不作为可执行shell脚本）

来源为原04的日志节录，未取得平台启动器及完整注入配置；只固定已知条件，不补造缺失字段。

```text
P: python launch_online_dp.py --pcp-size 4 --dcp-size 1 --dp-size 4 --tp-size 1 --pp-size 1 --dp-size-local 4 --dp-rank-start 0 --dp-address 80.48.29.129 --dp-rpc-port 12321 --vllm-start-port 8000
D: python launch_online_dp.py --dp-size 16 --tp-size 1 --pp-size 1 --dp-size-local 16 --dp-rank-start 0 --dp-address 80.48.29.130 --dp-rpc-port 12321 --vllm-start-port 8000

model=/mnt/share/weights/DeepSeek-V4-Flash-0731-w8a8/
tokenizer_mode=deepseek_v4
trust_remote_code=True
quantization=ascend
max_model_len=1048576
block_size=32
max_num_batched_tokens=8192
max_num_seqs=16
enforce_eager=True
enable_expert_parallel=True
gpu_memory_utilization=0.9
enable_prefix_caching=True
enable_chunked_prefill=True
kv_cache_dtype=auto
speculative_config={'method':'dspark','num_speculative_tokens':5,'enforce_eager':True}
kv_transfer_config=MultiConnector[connectors=[
  {'kv_connector':'MooncakeHybridConnector','kv_role':'kv_producer','kv_port':'30000',
   'kv_connector_extra_config':{'prefill':{'dp_size':4,'tp_size':1,'pcp_size':4},'decode':{'dp_size':16,'tp_size':1}}},
  {'kv_connector':'AscendStoreConnector','kv_role':'kv_producer',
   'kv_connector_extra_config':{'lookup_rpc_port':'0','backend':'memcache','use_layerwise':True,'layerwise_prefetch_layers':3}}]]
run A additional_config={'enable_cpu_binding':True,'enable_fused_mc2':0}
run B additional_config={'enable_cpu_binding':True,'enable_fused_mc2':1}
D additional_config={'ascend_compilation_config':{'enable_npugraph_ex':True,'enable_static_kernel':False},
                    'enable_cpu_binding':True,'multistream_overlap_shared_expert':True,'recompute_scheduler_enable':True}
```

