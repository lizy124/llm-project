verification_full_report.md
# A3 560T 128G HCCS 池化验证总览（方案 + Phase 1 完整细节）

> **本文定位**：`A3_HCCS/128G/` 验证工作的对外总览，关键结论与数据内联于本文，随包独立分发，无需回看原始文档。
> **当前状态**（2026-09-10）：Phase 1 单机基线（mooncake + memcache 双后端三形态）两轮验证 **PASS**；Phase 2 双机 HCCS 阻塞（等第二台 128G 机器）。

---

## 0. 执行摘要（Executive Summary）

**背景**：一体机要在双机 A3 560T 128G（共 1024GB HBM）上做 PD 分离，用外部 KV 池承载 DS V4 Flash 的 200K 上下文，避免每次请求都重新计算前缀。

**做的事情**：验证「不把 KV 放 HBM，而是放进 DDR/SSD 外部池」。Phase 1 先在单机（135）上打通两条池化后端——mooncake 与 memcache，其中 memcache 又验证了两种部署形态（内嵌 co-located / 独立进程 standalone），共三形态。

**结论（PASS）**：
- 池化功能正常，且**关闭 HBM 前缀缓存、全部命中走外部池**（P0 证据完整）。
- 冷灌首算 **~19.5s**，命中后 **~1.7s**，Prefill 提速 **~11 倍**（目标 5~8x，超额）。
- 命中率 **96.88%**，32 并发下三形态均 **32/32 成功**，聚合 TPS 126K~135K。
- 三形态两轮独立复现，数据完全一致（无后端回归）。

**卡点**：Phase 2 双机 HCCS 跨机 KV 传输需第二台 128G 机器到位；SSD / pp / 200K 全长验收为 Phase 3/4 待做。

---

## 1. 背景与定位

### 1.1 机型口径（已确认）

| 项 | 值 | 说明 |
|---|---|---|
| SKU 命名 | A3 560T 128G | 560T = 560 TFLOPS @FP16，128G = 每卡 HBM |
| 卡数 | 8 卡/机（双 die 封装，npu-smi 可见 16 chip） | 每机 FM8 |
| 每 die HBM | 64GB | 135 实测：16 chip × 64G |
| 整机 HBM 总量 | 1024GB（8 × 128GB） | 华为官方规格页口径 |
| 双机形态 | FM16 = 2 × FM8，HCCS 背靠背直连 | PD 分离部署 |
| 参考机器 | **135（80.5.9.135，已确认 128G 版）** | Phase 1 单机在其上执行 |

### 1.2 需求来源

一体机规划需求：FM16 双机 A3 560T 128G HCCS 背靠背互联，PD 分离形态，**DS V4 Flash 模型，200K 上下文**。负责人：李忠洋。交付目标：RP3 版本，主干分支。

### 1.3 技术栈

| 项 | 值 |
|---|---|
| 互联 | HCCS（背靠背双机直连） |
| 池化后端 | mooncake + memcache（两个均需验证） |
| 特性叠加 | pp + SSD（layerwise 见豁免条款 W1） |
| 模型 | DeepSeek V4 Flash（w8a8-mtp） |
| 度量口径 | 200K 上下文，128K 输入 / 1K 输出，前缀重复率 90% |

---

## 2. 验证目标与验收标准

### 2.1 核心目标（双机全量口径）

| 指标 | 目标值 |
|---|---|
| PrefixCache 命中率 | 80% DRAM + 20% SSD，总命中率 ~90%（匹配前缀重复率 90%） |
| Prefill 吞吐提升 | 5~8 倍（vs 无池化完全重算） |
| 上下文长度 | 200K（128K 输入 + 1K 输出） |
| 多前缀并发 | 支持 |
| HBM PrefixCache | 关闭（命中全部来自池化 DDR/SSD） |

### 2.2 验收项（P0 优先级）

| 优先级 | 标准 | 验证方式 |
|---|---|---|
| P0 | 池化基本功能正常（存/取/去重三维证据链） | metrics 计数器 |
| P0 | HBM PrefixCache 关闭，池化正常 | `vllm:prefix_cache_*` 恒 0 |
| P0 | 无池化完全重算基线对照 | 冷灌 0% 命中即基线 |
| P1 | 命中率 80% DRAM + 20% SSD | 命中率统计 |
| P1 | Prefill 吞吐提升 5~8 倍 | TTFT 对比基线 |
| P1 | 200K 上下文正常工作 | 长序列推理正确 |
| P2 | 双机 HCCS 跨机 KV 传输正常 | 跨机命中率 |
| P2 | PD 分离形态正常 | P/D 角色验证 |

### 2.3 已知豁免

| 编号 | 项 | 原因 |
|---|---|---|
| W1 | layerwise + DSV4 不覆盖 | DSV4 有 5 个 cache spec（attn + compressor.state_cache + indexer.k_cache + indexer.compressor.state_cache + swa_cache），`build_layerwise_reuse_layout` 仅支持每层 2 spec（1 main + 1 indexer.k_cache），`use_layerwise=true` 必崩 `ValueError`。已知硬限（#12853），非本任务缺陷 |
| W2 | 单节点失败路径不可构造 | pool_verify 知识库已知限制 |

### 2.4 128G 内存预算（对比 64G 的核心优势）

| 项 | 估算 |
|---|---|
| DSV4-Flash W8A8 权重 | ~230GB（TP8 分摊后每 die ~15GB） |
| 200K 上下文 KV（单请求） | 按实测校准 |
| 池化 buffer | 每卡 DRAM 5GB（mooncake global segment；A3 fabric mem 下 local_buffer_size 不生效） |
| 结论 | 128G（1024GB 总量）预算充裕，可全量跑 200K；64G 版可能需降 max_len 或分段——这是两机型方案分开写的核心原因 |

---

## 3. 总体方案：验证矩阵与阶段划分

### 3.1 后端 × 特性矩阵

| 序号 | 后端 | layerwise | SSD | pp | 预期 | 说明 |
|---|---|---|---|---|---|---|
| 1 | mooncake | false | false | false | PASS | 核心路径基线 |
| 2 | mooncake | false | true | false | PASS | 加 SSD 分级 |
| 3 | mooncake | false | false | true | PASS | 加 pp |
| 4 | mooncake | false | true | true | PASS | 全栈（无 layerwise） |
| 5 | memcache | false | false | false | PASS | 核心路径基线 |
| 6 | memcache | false | true | false | PASS | 加 SSD 分级 |
| 7 | memcache | false | false | true | PASS | 加 pp |
| 8 | memcache | false | true | true | PASS | 全栈（无 layerwise） |
| 9 | memcache | true | false | false | EXPECTED_FAIL | DSV4 multi-spec 硬限（W1） |
| 10 | memcache | true | true | false | EXPECTED_FAIL | 同根因 |

### 3.2 四阶段推进

- **Phase 1**：单机基线（135 先行）→ ✅ 已完成（两轮 PASS）。
- **Phase 2**：双机 HCCS 核心（PD 分离）→ 阻塞：等第二台 128G 机器。
- **Phase 3**：特性叠加（SSD / pp / 全栈）→ 待开始。
- **Phase 4**：性能验收（200K / 命中率 / 吞吐）→ 待开始。

单机基线与双机核心代码路径一致，仅传输拓扑不同，故 Phase 1 可先行。

### 3.3 双机验证拓扑（Phase 2 目标形态）

```
机A (FM8 128G)  <--HCCS背靠背-->  机B (FM8 128G)
  P-worker (Prefill)                D-worker (Decode)
  mooncake/memcache client          mooncake/memcache client
  DRAM: 5G/card + SSD               读 KV（跨机传输）
  mooncake master（可选）+ MetaService（单节点）
```

逻辑流：请求 → Router → 机A Prefill 计算 KV → 存机A DRAM（异步淘汰到 SSD）→ 机B Decode 从机A DRAM/SSD 加载 KV。

### 3.4 端口规划（全阶段统一）

| 用途 | 端口 | 说明 |
|---|---|---|
| mooncake master RPC | 50088 | 避开 50051（宿主机 OceanStor DTMA 占用，51/112/135 均踩过） |
| mooncake metrics | 9008 | 脚本覆盖值，官方默认 9003 |
| MetaService | 5001 / 6001 / 8001 | RPC / config_store / metrics；默认 5000/6000/8000 被宿主机占用，必改 |
| vllm API | 8004 | 与端口池错开 |

---

## 4. Phase 1 验证环境快照（135 实测口径）

| 项 | 值 |
|---|---|
| 硬件 | A3 560T 128G（8 卡 16 die，整机 1024GB HBM） |
| 服务器/容器 | 80.5.9.135 / pr15367_135（Ubuntu 24.04 aarch64，Python 3.11） |
| vllm / vllm-ascend | 0.27.1 / 0.1.dev4937+g03e0e41e1 |
| mooncake | mooncake-transfer-engine-npu 0.3.11.post1 |
| memcache | memcache_hybrid 1.2.0（pip）+ memfabric 1.2.0（pip）+ 1.2.1（/usr/local 编译版） |
| 模型 | DeepSeek-V4-Flash-w8a8-mtp（/data/combinded_nfs/DeepSeek-V4-Flash-w8a8-mtp） |
| vllm 形态 | TP8 × DP1，max_model_len 131072，max_num_seqs 64，block_size 32，`--no-enable-prefix-caching`，`--enforce-eager`，`--async-scheduling`，`--gpu-memory-utilization 0.90`，`--quantization ascend`，`--tokenizer-mode deepseek_v4`，`--enable-expert-parallel` |
| KV 连接器 | AscendStoreConnector，kv_role=kv_both，use_layerwise=false |
| 关键环境变量 | ASCEND_ENABLE_USE_FABRIC_MEM=1（A3 HCCS 必须），ACL_OP_INIT_MODE=1，PYTHONHASHSEED=0，VLLM_USE_V1=1 |
| 测试口径 | 131070-token 前缀 + 1 输出 token；bench 脚本 bench_prefix_135.py（SEED=20260907）；多前缀 = 4×131070 灌入 + 32 并发 |

**三形态部署差异**：

| 形态 | 池进程 | vllm 侧配置 | 池容量 |
|---|---|---|---|
| memcache co-located | vllm 内嵌 LocalService（每 rank 贡献） | MMC_LOCAL_CONFIG_PATH=包内 mmc-local.conf（device_sdma, 10GB） | 每 rank 10GB |
| memcache standalone | 独立 standalone_local_service.py（先起，贡献池） | dram.size=0GB 只接入 | rank-N-dram 10GB（独立进程名下） |
| mooncake | mooncake_master（常驻，50088/9008） | MOONCAKE_CONFIG_PATH 指向 mooncake.json | global_segment 10GB |

> Phase 1 用 131072 max_len 口径（Phase 4 扩到 200K）；环境从 09-07 的 host_shm 口径（vllm 6e448d0 / vllm-ascend b5a5099ba）演进为 device_sdma 口径，性能同量级非回归。

---

## 5. 三种部署形态：架构与配置细节

三形态共用同一 vllm `AscendStoreConnector` 框架，仅 backend 与部署方式不同。

### 5.1 memcache co-located（共置模式）

**分层**：hugepages（内核参数，一次性）→ MetaService（常驻单例，vllm 重启不动它）→ vllm 接入（每次启动）。

**hugepages（硬前置，缺失即崩）**：

```bash
echo 200000 > /proc/sys/vm/nr_hugepages   # 200000 页 × 2MB = 400G
grep HugePages_Total /proc/meminfo        # 期望 200000
```

**mmc-meta.conf（端口必改，默认 5000/6000/8000 被宿主机占用）**：

```ini
ock.mmc.meta_service_url = tcp://127.0.0.1:5001
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:6001
ock.mmc.meta_service.metrics_url = http://127.0.0.1:8001
ock.mmc.log_level = info
```

**mmc-local.conf（vllm 侧读）**：

```ini
ock.mmc.local_service.protocol = device_sdma     # A3 正路；host_shm 是旁路回退
ock.mmc.local_service.dram.size = 10GB           # 每 rank 上限，1GB 对齐；惰性分配
ock.mmc.meta_service_url = tcp://127.0.0.1:5001
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6001
ock.mmc.local_service.world_size = 256
ock.mmc.log_level = info
```

**MetaService 启动（官方写法已验证可用）**：

```bash
export MMC_META_CONFIG_PATH=<site-packages>/memcache_hybrid/config/mmc-meta.conf
nohup python3 -c "from memcache_hybrid import MetaService; MetaService.main()" \
  > /home/lizhongyang/map_135/memcache_logs/meta_sdma.log 2>&1 &
# 探活：5001/6001/8001 全 OPEN + curl :8001/metrics 有 Prometheus 文本
# 日志可见 "Loaded 4 config items"
```

**版本匹配铁律（device_sdma 双轨共存）**：
- site-packages 的 memfabric 必须 **1.2.0**（`pip install memfabric_hybrid==1.2.0 --force-reinstall`）——memcache 1.2.0 的 `_pymmc.so` 只兼容它；被 .run 升到 1.2.1 则 MetaService 报 `_pymmc import error`（pybind11 类型未注册）。
- /usr/local 编译版 memfabric 必须 **1.2.1**（含 SDMA copy 扩展 `libmf_hybm_copy_extend.so`）——vllm 启动前 `source /usr/local/memfabric_hybrid/set_env.sh` 使 LD_LIBRARY_PATH 优先加载。

**vllm 启动（完整可抄）**：

```bash
export MMC_LOCAL_CONFIG_PATH=<site-packages>/memcache_hybrid/config/mmc-local.conf
export LD_LIBRARY_PATH=<Location>/memcache_hybrid/lib:${LD_LIBRARY_PATH}
source /usr/local/memfabric_hybrid/set_env.sh
export PYTHONHASHSEED=0 VLLM_USE_V1=1 ACL_OP_INIT_MODE=1

python -m vllm.entrypoints.openai.api_server \
  --model /data/combinded_nfs/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 --port 8004 --served-model-name dsv4 --trust-remote-code \
  --enable-expert-parallel --tokenizer-mode deepseek_v4 --quantization ascend \
  --no-enable-prefix-caching --tensor-parallel-size 8 --data-parallel-size 1 \
  --enforce-eager --max-model-len 131072 --max-num-batched-tokens 10240 \
  --max-num-seqs 64 --block-size 32 --gpu-memory-utilization 0.90 \
  --kv-transfer-config '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": { "backend": "memcache", "lookup_rpc_port": "0", "use_layerwise": false }
  }'
```

- `lookup_rpc_port="0"` = 关 RPC lookup（单实例本机查找）；mooncake 用 "1"。
- `use_layerwise=false` 是 V4 唯一可跑形态（true 会崩，见 §8 坑 7）。
- `kv_buffer_size` 不用配——Ascend 强制重置为 1e9，池大小只看 mmc-local.conf。

### 5.2 memcache standalone（分离部署模式）

**动机**（官方 §3.7，仅 A3 HCCS 支持）：co-located 时 vllm 先加载权重后初始化 connector，池只能从剩余内存分，可能分不够大；standalone 在 vllm 启动**前**用独立进程先占内存留足池，vllm 以 `dram.size=0GB` 只接入。注意：与 PD 分离不是一回事——拆的是“MemCache 进程 vs vllm 进程”。

**两份 mmc-local.conf（连接/协议一致，只差 DRAM 贡献）**：

```ini
## mmc-local-standalone.conf（独立进程用，贡献池）
ock.mmc.local_service.dram.size = 10GB
ock.mmc.local_service.max.dram.size = 1024GB

## mmc-local-vllm.conf（vllm 用，不贡献）
ock.mmc.local_service.dram.size = 0GB
ock.mmc.local_service.max.dram.size = 1024GB
# 其余：protocol=device_sdma、meta 5001/6001、world_size=256 两份相同
```

**standalone 进程（必须先设卡）**：

```python
# standalone_local_service.py 核心逻辑（参考 memcache 源码 example/benchmark 多进程 worker）
import acl
acl.init()
acl.rt.set_device(0)                  # device_sdma 池在设备侧，必须先设卡
from memcache_hybrid import DistributedObjectStore
store = DistributedObjectStore()
res = store.init(0, init_bm=True)     # 读 MMC_LOCAL_CONFIG_PATH
# res == 0 后 keep-alive
```

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh        # 1.2.1 编译版
export MMC_LOCAL_CONFIG_PATH=/home/lizhongyang/map_135/mmc-local-standalone.conf
nohup python3 -u standalone_local_service.py > standalone_ls.log 2>&1 &
# 池注册是 eager（init 即注册，无需等请求）：
# curl :8001/metrics → memcache_segment_capacity_bytes{segment="rank-1-dram"} 10737418240
```

部署顺序：MetaService → standalone 进程 → 等池注册 → vllm（`MMC_LOCAL_CONFIG_PATH` 指向 vllm conf，其余启动参数与 co-located 完全一致）。

### 5.3 mooncake

**mooncake.json（`MOONCAKE_CONFIG_PATH` 指定）**：

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "master_server_address": "127.0.0.1:50088",
  "global_segment_size": "10GB",
  "local_buffer_size": "5GB",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true
}
```

> global_segment_size 官方默认 5GB，135 为与 memcache 10GB 对齐实测用 10GB；A3 fabric mem 下 local_buffer_size 不生效。

**master 启动（二进制直启；官方 `python -m` 写法在 0.3.11 实测不通）**：

```bash
nohup /usr/local/python3.11.10/bin/mooncake_master \
  --rpc_port 50088 --metrics_port 9008 --logtostderr=true \
  > /home/lizhongyang/map_135/mooncake_logs/master.log 2>&1 &
```

**探活四查（全过才算就绪）**：`role=leader`、`state=serving`、`service_ready=true`、TCP 50088/9008 连通（启动 10s 内从 standby 自动转 leader）。

**vllm 侧差异（仅 kv-transfer-config）**：

```json
{ "kv_connector": "AscendStoreConnector", "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": { "backend": "mooncake", "lookup_rpc_port": "1", "use_layerwise": false } }
```

客户端数 = DP×TP = 8，探活看 `master_active_clients=8` 全注册后再压测。mooncake **不需要** hugepages 和 memcache/memfabric set_env。

---

## 6. 测试方法（bench 口径）

1. **预热**：2 个单请求。
2. **单前缀**：同一 131070-token prompt 发 3 次——A（首灌冷）、A2/A3（同文重发命中）。
3. **多前缀**：4 个不同 131070-token 前缀灌入 + 32 并发同前缀集（每请求 1 输出 token）。
4. **池指标快照**：每次 bench 前后抓 master/MetaService metrics（存证据）。
5. **P0 补全**：HBM 计数器恒 0 证明 + 去重重发实验。

判读：`external_prefix_cache_queries>0` = 链路活；`hits>0` = 真命中；`prefix_cache_queries=0` = HBM 关闭生效。脚本 bench_prefix_135.py（SEED=20260907 保证同文可复现）。

---

## 7. Phase 1 验证结果

> **一句话结论**：三形态（mooncake / memcache co-located / memcache standalone）在命中率、TTFT、吞吐上行为一致，池化功能正常、无后端回归，两轮独立复现确认结果可信。
> **怎么读下面的表**：核心对比看「冷灌（首算，无缓存）vs 命中（复用缓存）」。冷灌都约 19.5s，命中后都降到约 1.7s，差约 11 倍——这就是 KV 池复用的价值；三行数字接近，说明三个后端没有本质差异。

### 7.1 Round 1 性能总表（2026-09-08~09，131070-token 前缀口径）

> **关于命中率 96.88% 的通俗解释**：前缀长 131070 token，但 KV 池按固定粒度 4096 token 切块存取，需要**完整整块**才算命中。尾部不足一整块的那段（131070 ÷ 4096 余 4094）无法参与命中，相当于最多命中 131072−4096 = 126976，即
>
>    命中率上限 = 126976 / 131070 ≈ 96.88%
>
> 三形态都恰好打到这个理论上限，说明它们是“该命中的全命中了”，而非 3.12% 的真失败。若请求长度正好是 4096 的整数倍，命中率可到 100%。

| 后端形态 | 冷灌 TTFT | 命中 TTFT (A2/A3) | 命中率 | 加速比 | 32 并发聚合 TPS | 多前缀 TTFT mean/p99 |
|---|---|---|---|---|---|---|
| memcache co-located (device_sdma) | 19.95s | 1.70s / 1.70s | 96.88% | ~11.7x | 132.6K | 17.63s / 30.70s |
| memcache standalone | 19.88s | 1.72s / 1.69s | 96.88% | ~11.6x | 135.4K | 17.26s / 30.03s |
| mooncake | 19.47s (warm0) | 1.78s / 1.77s | 96.88% | ~10.9x | 126.2K | 18.47s / 32.24s |

> mooncake A 首灌 137.37s / 954 TPS（0% 命中）——首次 KV 保存触发 store 惰性初始化（vllm 日志 12:05:13 `MooncakeBackend.exists called before store initialization`）+ 10GB segment 分配，一次性成本；稳态冷灌以 multi warm0=19.47s 为准（三形态 warm0 均 19.4~19.5s）。
> 单前缀命中态 prefill TPS：co-located ≈77K / standalone 76.4K~77.7K / mooncake 73.7K~74.2K。

### 7.2 Round 2 全量复验（2026-09-09 13:00~13:55，清池重启重跑）

复验方式：每形态**杀进程重启 + 池清零**（非原地重复），数据独立采集。

| 指标 | mooncake | memcache co-located | memcache standalone |
|---|---|---|---|
| 命中率 | 96.88% | 96.88% | 96.88% |
| A2/A3 命中 TTFT | 1.76s / 1.76s | 1.71s / 1.69s | 1.72s / 1.70s |
| 稳态冷灌 (warm0) | 19.42s | 19.45s | 19.40s |
| multi 32 并发 | 32/32，129.2K TPS，p99 31.50s | 32/32，135.0K TPS，p99 30.12s | 32/32，135.1K TPS，p99 30.12s |
| 首灌（重启后） | 119.79s（惰性初始化） | 104.35s*（池段分配+首次落池） | 21.94s（+10% 正常抖动） |
| KV 条目 (single→multi) | 1891 → 9455 keys | 1891 → 9455 alloc | 1891 → 9455 alloc |
| 池指标 | allocated 1.34GB→6.71GB，clients 8 | LS 池 10GB | rank-0-dram 10GB |

\* co-located Round 2 首灌 104.35s：清池重启后首请求含 LocalService 池段分配 + 首次 KV 落池（Round 1 的 19.95s 是池已热的口径），稳态以 warm0 为准。

**Round 1 vs Round 2 关键对照**：命中率 96.88% **六次测量完全一致**（三形态×两轮）；KV 条目数 9455 **完全复现**；mooncake keys/allocated/clients 数值逐项复现；命中 TTFT 偏差 <1.5%；multi TPS ±2.4%。**结论：Round 1 全部可复现，Round 2 PASS，Phase 1 关闭。**

### 7.3 存 / 取 / 去重三维证据链（P0）

| 维度 | memcache（co-located/standalone） | mooncake |
|---|---|---|
| 存 | rank-N-dram 池注册 10GB；首灌后 KV 落池 | master_allocated_bytes 0→1.34GB(single)→6.71GB(multi)；key_count 0→1891→9455 |
| 取 | external_prefix_cache_hits：A2/A3 = 126976/131070 = 96.88% | 同左；external 累计 5.50M queries / 4.70M hits |
| 去重 | （无重复键生成） | 重发同前缀：key_count 9455→9455 不变、allocated 6713538560 不变、命中仍 96.88%（lat 1.75s） |
| 健康度 | — | PutStart 2695 = PutEnd 2695，失败 0、淘汰 0；active_clients 8 |

### 7.4 HBM PrefixCache 关闭证明（P0）

```
vllm:prefix_cache_queries_total 0.0              # HBM 前缀缓存从未查询
vllm:prefix_cache_hits_total    0.0              # HBM 命中恒 0
vllm:external_prefix_cache_queries_total 5.50494e+06   # 全部查询走 KV 池
vllm:external_prefix_cache_hits_total    4.698112e+06 # 全部命中来自 KV 池
```

### 7.5 三形态一致性判定

- 命中率：三形态均 96.88%（= 126976/131070，DSV4 granularity=4096 边界损失 3.12%，理论值稳定复现）。
- 命中 TTFT：1.69~1.78s（±5%）；冷灌 19.4~19.95s（±2.5%）。
- 结论：单机基线三形态行为一致，池化功能正常，mooncake 后端无回归（vs memcache 基线）。

### 7.6 验收标准对照（Phase 1 单机口径映射）

| 计划目标（双机口径） | Round 1/2 单机映射 | 达成 |
|---|---|---|
| 命中率 ~90%（90% 重复率） | 单前缀同文重发（理论 96.9% = 1-4096/131072） | ✅ 96.88% 实测 |
| Prefill 提升 5~8x | 冷灌 vs 命中 TTFT | ✅ ~11x（超额） |
| 多前缀并发 | 4×131070 + 32 并发 | ✅ 32/32 |
| 200K 上下文 | 131072（Phase 4 扩 200K） | ✅ 本轮口径 |
| 双机 HCCS / SSD / pp | Phase 2/3 范围 | N/A（未开始） |

**Phase 1 总结论：PASS**——P0 三项全满足（三维证据链、HBM 恒 0、冷灌即完全重算基线），两轮独立验证可复。

---

## 8. 坑与经验教训全集（135/165 实测）

| # | 坑 | 现象 | 处置/结论 |
|---|---|---|---|
| 1 | mooncake 首灌 137s（Round 2 复现 119.8s） | 首请求延迟虚高 | store 惰性初始化 + 10GB segment 分配，一次性成本；延迟敏感场景可发预热请求 |
| 2 | memcache 大池 80GB 起不来 | `HalMemCreate ret:6`（1GB page 失败）→ 回退 2MB page 卡死 | 降 10GB/rank（135 实测可用）；A3 强制 1GB 对齐，失败逐级下调（165 128G 同类） |
| 3 | MetaService 端口冲突 | 默认 5000/6000/8000 被宿主机占用，启动即挂 | mmc-meta.conf 改 5001/6001/8001；官方写法 `MetaService.main()` 会读该文件（曾误判其不读配置而绕路手工设字段——真因是端口没改） |
| 4 | mooncake master 端口冲突 | 50051 被宿主机 OceanStor DTMA 占用 | 固定 50088/9008 |
| 5 | mooncake master 官方启动写法不通 | `python -m mooncake_transfer_engine.mooncake_master --master_port/--config` 失败 | 0.3.11 用二进制直启 + `--rpc_port/--metrics_port` |
| 6 | memfabric 版本双轨 | site-packages 1.2.1 → `_pymmc import error`（pybind11 类型未注册） | pip 回退 1.2.0 喂 MetaService；/usr/local 1.2.1 编译版（含 libmf_hybm_copy_extend.so）喂 vllm SDMA copy；缺扩展时回退 host_shm |
| 7 | V4 + use_layerwise=true 崩 | `ValueError: multiple cache specs must have exactly one main spec`（V4 有 5 spec） | V4 只跑 non-layerwise 纯池；已知硬限（#12853），矩阵 #9/#10 EXPECTED_FAIL |
| 8 | 缺 hugepages | 传输层崩/不可用 | memcache 硬前置 200000 页；mooncake 不需要 |
| 9 | vllm 残留进程杀不掉 | setproctitle 改名，普通 pkill 失效 | `pkill -9 -f "VLLM::"` / `-f "vllm serve"` / `-f "from multiprocessing"` + 清 /dev/shm/psm_* |
| 10 | kv_buffer_size 改了没用 | Ascend 强制重置 1e9 | 池大小只看 mmc-local.conf dram.size / mooncake.json segment |
| 11 | standalone 独立进程不设卡 | device_sdma 池分配失败/异常 | `acl.init()` + `acl.rt.set_device(0)` 后再 `store.init`（参考 memcache example/benchmark） |
| 12 | vllm 误配 standalone conf | vllm 也贡献 DRAM，两进程抢池 | 两份配置分开：standalone 进程 10GB conf，vllm 0GB conf |
| 13 | go.ps1 内联命令含 `\|` 会裂 | 远程命令误执行 | 一律用 .sh 脚本推送执行（跨 PowerShell→ssh→bash→docker 多层引号转义） |
| 14 | LocalService 段注册时机 | co-located 首请求才注册（空载查 metrics 段容量 0 不算故障）；standalone eager 注册 | 以 metrics 实际值为准；standalone 段名 rank-N 编号由 MetaService 分配，不影响使用 |

---

## 9. 当前进度与后续计划

### 9.1 进度

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 环境准备 | 135 可用；第二台 128G 待申请 |
| 1 | 单机基线 | ✅ 完成（两轮 PASS，2026-09-09 关闭） |
| 2 | 双机 HCCS 核心 | 阻塞（等机器） |
| 3 | 特性叠加（SSD/pp/全栈） | 待开始 |
| 4 | 性能验收（200K/命中率/吞吐） | 待开始 |

### 9.2 Phase 2 要点（双机到位后）

- 机A P-worker + 机B D-worker（kv_role 均 kv_both 起步），master/MetaService 单节点部署。
- 双机一致性检查：DSV4 权重 md5、vllm/vllm-ascend git log -1 对齐、端口 50088/9008/5001/6001/8001/8004 双机各查、hugepages/mmc conf/set_env 双机各配。
- 验证跨机传输：机A master 指标增长 + 机B D-worker external hit rate > 0。

### 9.3 Phase 3/4 要点

- **SSD 分级**：构造超 DRAM 容量数据集 → 验证淘汰到 SSD → SSD 命中占比 ~20% → 对比 DRAM vs SSD 命中 TTFT。mooncake 改 mooncake.json，memcache 改 mmc-local.conf SSD 层（>=1.2.0 已知问题已修复）。
- **pp**：`--pipeline-parallel-size 2`，注意 pp×tp ≤ 总卡数、分组与物理拓扑匹配。
- **200K 验收**：陈波数据集工具构造（10 前缀×128K×重复 9 次 + 10 新前缀，重复率 90%）；对照实验四组——A 无池完全重算（135 实测 131070 token ≈19.9s）/ B 池化 DRAM 命中（实测 1.7s，~11.6x）/ C SSD 命中（待测）/ D 混合加权。目标 DRAM 80% + SSD 20%、总 ~90%、提升 5~8x。

### 9.4 风险清单

| 风险 | 应对 |
|---|---|
| HCCS 双机链路不稳定 | 重试机制 + master 监控 |
| SSD 命中率不达标 | 需实测，必要时调 tiering 策略 |
| 200K 超池容量 | 128G 预算充裕；仍超则查 kv_buffer_size、分段验证 |
| 双机权重/版本不一致 | md5 + git log 对齐 |
| pp+池化+SSD 叠加兼容 | 先分步后叠加 |
| 第二台 128G 未到位 | Phase 2 阻塞，提前申请资源 |

---

## 10. 证据与脚本索引

### 10.1 证据文件（本目录下）

**evidence_round1/**（12 份）：

| 文件 | 内容 |
|---|---|
| bench_sdma_single.log / bench_sdma_multi.log | memcache co-located 单/多前缀原始数据 |
| bench_standalone_single.log / bench_standalone_multi.log | memcache standalone 单/多前缀原始数据 |
| standalone_ls_20260909_105828.log | standalone LocalService 启动日志（rank-1-dram 10GB 注册） |
| bench_mooncake_single.log / bench_mooncake_multi.log | mooncake 单/多前缀原始数据 |
| bench_mooncake_master_{before,after_single,after_multi}.txt | mooncake master 指标快照（存证据） |
| phase1_gaps_evidence.txt | HBM-0 + 去重补全证据 |
| status.txt | 阶段状态快照（stage/PID/commit/next） |

**evidence_round2/**（14 份，三子目录）：mooncake/（single.log、multi.log、master_{before,after_single,after_multi}.txt）、memcache_colocated/（single.log、multi.log、meta_after_{single,multi}.txt）、memcache_standalone/（同左，LS rank-0-dram 10GB）。

### 10.2 服务器侧原始数据（135，保留不删）

- Phase 1 第一轮：`/home/lizhongyang/map_135/results/latest/bench_{sdma,standalone,mooncake}_{single,multi}.log` + master/meta 快照 + `phase1_gaps_evidence.txt` + `status.txt`
- Round 2：`/home/lizhongyang/map_135/run/bench_mooncake_r2_20260909_131113/`、`bench_memcache_r2_colocated_20260909_132613/`、`bench_memcache_r2_standalone_20260909_134121/`
- vllm/mooncake/memcache 日志：`run/v4_mooncake_131072.log`、`mooncake_logs/master.log`、`memcache_logs/*.log`

### 10.3 脚本清单（map_135/start/）

| 类别 | 脚本 |
|---|---|
| memcache co-located | setup_meta_sdma_135_v2.sh、switch_mmc_sdma_135.sh、start_v4_nolw_135.sh |
| memcache standalone | mmc-local-standalone.conf / mmc-local-vllm.conf、standalone_local_service.py、setup_standalone_135.sh、restart_standalone_ls_135.sh、start_v4_standalone_135.sh、smoke_standalone_135.sh、run_bench_standalone_135.sh |
| mooncake | start_mooncake_135.sh、start_v4_mooncake_135.sh、probe_mooncake_135.sh、poll_vllm_ready_135.sh |
| 基准 | bench_prefix_135.py、run_bench_mooncake_round2_135.sh、run_bench_memcache_r2_135.sh、restart_mooncake_round2_135.sh、switch_memcache_colocated_r2_135.sh、switch_standalone_r2_135.sh |
| P0 补全 | verify_phase1_gaps_135.sh |

---

## 附录 A：关键环境变量全集

```bash
# A3 HCCS 分支（必须）
export ASCEND_ENABLE_USE_FABRIC_MEM=1
export ACL_OP_INIT_MODE=1

# 通用
export PYTHONHASHSEED=0
export VLLM_USE_V1=1

# memcache 特有
export MMC_LOCAL_CONFIG_PATH=<path>/mmc-local.conf
export LD_LIBRARY_PATH=<site-packages>/memcache_hybrid/lib:${LD_LIBRARY_PATH}
source /usr/local/memfabric_hybrid/set_env.sh       # 1.2.1 编译版（SDMA copy）
source /usr/local/memcache_hybrid/set_env.sh

# mooncake 特有
export MOONCAKE_CONFIG_PATH=/home/lizhongyang/map_135/run/mooncake.json
```

## 附录 B：验证清单（部署后逐项确认）

- [ ] hugepages 已配置：`nr_hugepages=200000`（memcache 前置）
- [ ] memfabric 双版本就绪：site-packages 1.2.0（MetaService）+ /usr/local 1.2.1（vllm SDMA）
- [ ] mmc-meta.conf 端口已改为 5001/6001/8001（避开宿主机 5000/6000/8000）
- [ ] mmc-local.conf `protocol=device_sdma`、`dram.size=10GB`（standalone：进程 10GB + vllm 0GB 两份）
- [ ] mooncake master：50088/9008，四查全过（leader/serving/service_ready/TCP）且 `active_clients=8`
- [ ] vllm `--no-enable-prefix-caching` 生效（`prefix_cache_queries_total=0`，命中全走 external）
- [ ] 池指标快照：存（allocated/keys 增长）/ 取（hit rate>0）/ 去重（重发 keys 不变）