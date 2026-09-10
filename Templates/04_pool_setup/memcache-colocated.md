01_memcache_kv_pool_setup.md
# Memcache KV 池：配置与拉起手册（Ascend 800I-A3）

> 适用对象：在 Ascend 800I-A3 环境（openEuler 宿主机 + Ubuntu 容器）上，从零拉起 **memcache_hybrid MetaService + vllm KV 池** 的完整操作指引。
> 参考实现：135（80.5.9.135）pr15367_135 容器 / 165（192.168.13.165）pool_165 容器。
> 环境前提：NPU 已驱动、镜像已装 `memcache_hybrid`（pip 版）、vllm + vllm-ascend 已装。

---

## 0. 整体分层（先看这个，别搞混）

```
┌─────────────────────────────────────────┐
│  vllm 进程（每个 rank 一个）              │  ← 每次启动 vllm 时接入（步骤 3）
│  AscendStoreConnector + KVTransferConfig  │
└──────────────┬──────────────────────────┘
               │ host_shm / device_sdma 传输
┌──────────────┴──────────────────────────┐
│  memcache_hybrid MetaService（常驻单例）  │  ← 一次拉起（步骤 2），vllm 反复重启不动它
│  管理 KV 池元数据 + 数据段分配            │
└─────────────────────────────────────────┘
               │ 前置：hugepages（步骤 1）
```

| 层 | 拉起频率 | 生命周期 | 配置文件 |
|---|---|---|---|
| hugepages | 一次性（系统重启前） | 内核参数 | `/proc/sys/vm/nr_hugepages` |
| MetaService | 一次拉起 | 常驻，vllm 重启不动它 | `mmc-meta.conf` + `mmc-local.conf` |
| vllm KV 池接入 | 每次启动 vllm | 随 vllm 进程 | 启动参数 + 环境变量 |

---

## 1. 前置：hugepages（硬条件）

**为什么**：device transfer（SDMA/host_shm）需要大页内存。缺了会直接启动失败或传输不可用（165 实测缺大页必崩）。

```bash
# 检查当前
grep HugePages_Total /proc/meminfo

# 设置 200000 页 × 2MB = 400G（135/165 通用口径）
echo 200000 > /proc/sys/vm/nr_hugepages

# 确认生效
grep -E 'HugePages_Total|HugePages_Free' /proc/meminfo
```

> 期望：`HugePages_Total: 200000`。若设置后达不到，说明系统内存不够，先 abort 排查。

**注意**：privileged 容器可写 `/proc/sys/vm/nr_hugepages`；非 privileged 需在宿主机执行。

---

## 2. MetaService：一次拉起，常驻

### 2.1 找配置文件

```bash
# 容器内找 memcache_hybrid 安装路径
pip show memcache_hybrid | grep '^Location:'
# 输出示例: Location: /usr/local/python3.11.10/lib/python3.11/site-packages
# 配置文件在: <Location>/memcache_hybrid/config/mmc-{meta,local}.conf
```

### 2.2 配置 mmc-local.conf（关键容量配置）

```bash
CONF=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/mmc-local.conf
```

需要确认的项（135 实测口径）：

```ini
# 传输协议：device_sdma 为主（A3 正路，device→device SDMA 直拷）
# host_shm 是旁路（host 内存中转），仅在 device_sdma 环境缺库时临时用
ock.mmc.local_service.protocol = device_sdma

# 池子大小：每个 rank 的上限（A3 必须对齐 1GB）
# 建议 10GB/rank；A3 有对齐约束，若 HalMemCreate 失败（165 实测 128G 会崩 ret:6）则逐级下调
ock.mmc.local_service.dram.size = 10GB

# MetaService 地址（单例，本机）
ock.mmc.meta_service_url = tcp://127.0.0.1:5000
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6000

# 日志级别
ock.mmc.log_level = info

# 最大 rank 数（world_size，默认 256 够用）
ock.mmc.local_service.world_size = 256
```

**池子大小理解**：
- `dram.size` = 每个 rank 的池段上限，TP8 × 10GB → 理论合计 80GB
- 是**上限不是预分配**，按需惰性分配（实测 4 rank 只分配 ~314MB 就能全命中）
- A3 强制 1GB 对齐，建议 10GB；若 HalMemCreate 失败则下调（165 实测 128G 崩 ret:6）

### 2.3 配置 mmc-meta.conf（一般不动）

```ini
ock.mmc.meta_service_url = tcp://127.0.0.1:5000
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:6000
ock.mmc.meta_service.metrics_url = http://127.0.0.1:8000
```

### 2.4 拉起 MetaService（官方写法已验证：export MMC_META_CONFIG_PATH + MetaService.main()）

**2026-09-08 135 实测修正**：官方写法**完全可用且推荐**。`MetaService.main()` 会读 `MMC_META_CONFIG_PATH` 指向的 mmc-meta.conf（日志可见 `Loaded 4 config items`）。

```bash
export MMC_META_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/mmc-meta.conf

nohup python3 -c "from memcache_hybrid import MetaService; MetaService.main()" \
  > /home/lizhongyang/map_135/memcache_logs/meta_sdma.log 2>&1 &
echo $! > /home/lizhongyang/map_135/memcache_logs/meta.pid
```

> **坑（历史误判警示）**：早期曾以为 `MetaService.main()` 不读配置文件、改绕路用 `MetaConfig()` 手工设字段（`mc.meta_service_url=...`）。实测证明 **main() 会读 mmc-meta.conf**；当初失败真正原因是 **mmc-meta.conf 里没改成独立端口**（写的还是默认 5000/6000/8000），宿主机已占用 → 启动即挂。**正确做法 = 改 mmc-meta.conf 端口（5001/6001/8001）+ 官方写法**。手工设字段方式也能用（等价于改文件），但不是必须。
> mmc-meta.conf 内容（改端口后）：
> ```ini
> ock.mmc.meta_service_url = tcp://127.0.0.1:5001
> ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:6001
> ock.mmc.meta_service.metrics_url = http://127.0.0.1:8001
> ock.mmc.log_level = info
> ```
> 完整脚本（官方写法 + device_sdma）：`map_135/start/setup_meta_sdma_135_v2.sh`（可改用官方写法）。

### 2.5 探活（必须，否则后续全白搭）

```bash
# 5001/6001/8001 应全 OPEN（按 2.4 实际配置的端口）
python3 -c "
import socket
for p in (5001,6001,8001):
    s=socket.socket(); s.settimeout(1.5)
    try: s.connect(('127.0.0.1',p)); print(p,'OPEN'); s.close()
    except: print(p,'CLOSED')
"

# metrics 必须有合法 Prometheus 指标
curl -s http://127.0.0.1:8001/metrics | head -5
# 期望看到 memcache_total_capacity_bytes 等；LocalService 未注册前段容量为 0 属正常
# 注意：LocalService 段（rank-N-dram）是 vllm 首个推理请求时才注册，空载查不到不算故障
```

> 135 完整拉起脚本：[map_135/start/setup_meta_sdma_135_v2.sh](file:///d:/project/agent_project/map_135/start/setup_meta_sdma_135_v2.sh)

---

## 3. vllm 侧接入（每次启动 vllm）

### 3.1 环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# device_sdma 必须：加载 1.2.1 编译版 memfabric（含 libmf_hybm_copy_extend.so）
# 该 set_env 把 1.2.1 的 lib64 放到 LD_LIBRARY_PATH 最前，vllm 的 memcache 后端即用 1.2.1 core
source /usr/local/memfabric_hybrid/set_env.sh

# memcache 配置路径（vllm 侧读 local：protocol=device_sdma + dram.size + meta 端口）
export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/mmc-local.conf

# memcache 动态库路径
export LD_LIBRARY_PATH=<Location>/memcache_hybrid/lib:${LD_LIBRARY_PATH}

# vllm 必需
export PYTHONHASHSEED=0
export VLLM_USE_V1=1
export ACL_OP_INIT_MODE=1
```

> **版本匹配铁律（2026-09-08 135 实测）**：
> - `memcache_hybrid` 用 pip 1.2.0；其 `_pymmc.so` 只兼容 **site-packages 的 memfabric 1.2.0**。若 site-packages 的 memfabric 被 .run 装成 1.2.1，MetaService 会报 `_pymmc import error`（pybind11 类型未注册）。**修复：`pip install memfabric_hybrid==1.2.0 --force-reinstall` 回退。**
> - device_sdma 的 SDMA copy 扩展库 `libmf_hybm_copy_extend.so` 在 **1.2.1 编译版**（`/usr/local/memfabric_hybrid/latest/aarch64-linux/lib64`），vllm 进程靠 `source /usr/local/memfabric_hybrid/set_env.sh`（LD_LIBRARY_PATH 优先）加载。
> - 即：**site-packages memfabric 1.2.0（喂 MetaService）+ /usr/local memfabric 1.2.1（喂 vllm 的 SDMA copy）** 双轨共存，已验证全链路通。

### 3.2 启动 vllm（完整可抄版，以 DeepSeek-V4-Flash-w8a8-mtp 为例，135 实测）

```bash
# ---- 环境变量（device_sdma 必须 source memfabric 1.2.1 set_env）----
export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/mmc-local.conf
export PYTHONHASHSEED=0
export VLLM_USE_V1=1
export ACL_OP_INIT_MODE=1
source /usr/local/memfabric_hybrid/set_env.sh   # 加载 libmf_hybm_copy_extend.so（1.2.1 编译版）

# ---- 启动命令 ----
nohup python -m vllm.entrypoints.openai.api_server \
  --model /data/combinded_nfs/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8004 \
  --served-model-name dsv4 \
  --trust-remote-code \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --quantization ascend \
  --no-enable-prefix-caching \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enforce-eager \
  --max-model-len 131072 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 64 \
  --block-size 32 \
  --gpu-memory-utilization 0.90 \
  --kv-transfer-config '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "backend": "memcache",
      "lookup_rpc_port": "0",
      "use_layerwise": false
    }
  }' > v4_nolw_131072_sdma.log 2>&1 &
```

> 换其他模型时：删掉 V4 特有参数（`--tokenizer-mode deepseek_v4`、`--quantization ascend` 视模型而定），只留通用参数 + `--kv-transfer-config`。`--served-model-name` 用于指定推理时的模型名。
> 完整脚本：`map_135/start/start_v4_nolw_135.sh`

关键参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `kv_connector` | `AscendStoreConnector` | 固定，KV 池连接器 |
| `kv_role` | `kv_both` | 单实例：既生产又消费（PD 分离时 P=kv_producer / D=kv_consumer） |
| `backend` | `memcache` | 用 memcache 后端（对应 MetaService） |
| `lookup_rpc_port` | `"0"` | 关 RPC lookup（单实例本机查找） |
| `use_layerwise` | `false` | **关键**：`false` = 纯 KV 池（135 唯一可跑形态）；`true` = layerwise（V4 5-spec 会崩，见 §6） |

> vllm 侧 `kv_buffer_size` 不用配——Ascend 强制重置为 1e9（NCCL 后端残留），池子大小只看 mmc-local.conf 的 `dram.size`。

---

## 4. 验证 KV 池生效（别跳过）

服务起来后，必须确认 KV 池真在走（不是走了本地 prefix cache）：

```bash
# 1. 冒烟请求
curl -s http://127.0.0.1:8004/v1/completions -d '{"model":"dsv4","prompt":"hello","max_tokens":16}' | head -c 200

# 2. 查 metrics：external_prefix_cache_queries 应该有计数
curl -s http://127.0.0.1:8004/metrics | grep external_prefix_cache_queries
# 期望: vllm:external_prefix_cache_queries_total > 0

# 3. 前缀复用实验（快速鉴别）
#    同一 prompt 连发 2 次，第 2 次 external_prefix_cache_hits 应 > 0
curl -s http://127.0.0.1:8004/metrics | grep external_prefix_cache_hits
```

判读：
- `queries` 有计数 = KV 池链路活着（请求在查池）
- `hits` 有计数 = 真的命中了（KV 落池+复用成功）
- `queries=0` = 没走池（检查 `--no-enable-prefix-caching` 是否误开、`backend` 是否对）
- `hits=0` 但 `queries>0` = 池活着但没复用（正常，随机 prompt 本来就无共享前缀）

---

## 5. 常用操作

### 5.1 重启 MetaService（改配置后）

```bash
kill -9 $(cat /home/lizhongyang/map_135/memcache_logs/meta.pid) 2>/dev/null
sleep 1
# 重新执行 2.4 拉起 + 2.5 探活
```

### 5.2 调大池子（建议 10GB）

```bash
# 1. 改 mmc-local.conf
sed -i 's/^\(ock.mmc.local_service.dram.size\) *= *.*/\1 = 10GB/' $CONF_LOCAL

# 2. 重启 MetaService（5.1）

# 3. 启动 vllm 前观察 HalMemCreate 是否成功
#    失败会报 ret:6，逐级下调（8GB→4GB→1GB）
```

### 5.3 清理 vllm 残留（重启服务前）

```bash
# vllm 进程被 setproctitle 改名，普通 pkill 杀不掉
pkill -9 -f "vllm serve"
pkill -9 -f "VLLM::"
pkill -9 -f "from multiprocessing"
sleep 5
rm -f /dev/shm/psm_* 2>/dev/null
```

---

## 6. 已知限制与坑（135/165 实测）

| 坑 | 现象 | 解法 |
|---|---|---|
| **V4 + use_layerwise=true 会崩** | `ValueError: multiple cache specs must have exactly one main spec`（V4 有 5 个 spec：attn+compressor+indexer×2+swa） | 135 上 V4 只能用 `use_layerwise: false`（纯 KV 池）；layerwise 是已知功能限制（#12853），非配置问题 |
| **_pymmc import error（pybind11 类型未注册）** | `from _pymmc import ...` 报 `arg(): could not convert default argument...` | site-packages 的 memfabric 被 .run 升到 1.2.1，与 memcache 1.2.0 的 _pymmc.so 不兼容。**`pip install memfabric_hybrid==1.2.0 --force-reinstall` 回退**（2026-09-08 实测） |
| **MetaService 起来但绑默认端口 5000/6000/8000（与宿主机冲突）** | `MetaService.main()` 启动后 metrics 在 8000 而非 8001，或报 "Failed to start HTTP Service" | **mmc-meta.conf 里没改端口**（main() 会读它，但读到的还是默认值）。改 mmc-meta.conf 为独立端口（5001/6001/8001）即可；`MetaConfig()` 手工设字段等价但非必须（2026-09-08 实测） |
| **缺 hugepages** | 传输层直接不可用 / 启动崩 | 步骤 1 必须做，缺了即 abort |
| **dram.size 128G 崩** | `HalMemCreate ret:6` | A3 强制 1GB 对齐，建议 10GB，失败则逐级下调 |
| **vllm 重启后 MetaService 残留** | 下次启动 KV 池状态混乱 | 改池配置才重启 MetaService；日常 vllm 重启不动它 |
| **kv_buffer_size 改了没用** | Ascend 强制重置 1e9 | 池子大小只看 mmc-local.conf dram.size，别在 vllm 参数里找 |

---

## 7. 135 实测参考配置（device_sdma 版，2026-09-08 验证）

```bash
# ==== hugepages ====
echo 200000 > /proc/sys/vm/nr_hugepages

# ==== mmc-local.conf（vllm 侧读）====
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 10GB
ock.mmc.meta_service_url = tcp://127.0.0.1:5001
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6001

# ==== 版本匹配：site-packages memfabric 回退 1.2.0（喂 MetaService）====
pip install memfabric_hybrid==1.2.0 --force-reinstall

# ==== MetaService 拉起（Python 直接设字段，独立端口）====
export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/mmc-local.conf
nohup python3 -u -c "from memcache_hybrid import MetaService, MetaConfig; mc=MetaConfig(); mc.meta_service_url='tcp://127.0.0.1:5001'; mc.config_store_url='tcp://127.0.0.1:6001'; mc.metrics_url='http://127.0.0.1:8001'; mc.ha_enable=False; mc.log_level='info'; MetaService.setup(mc); MetaService.main()" > meta_sdma.log 2>&1 &

# ==== vllm 启动前（device_sdma 需要 1.2.1 copy_extend）====
source /usr/local/memfabric_hybrid/set_env.sh   # LD_LIBRARY_PATH 优先 1.2.1 lib64
```

### 7.1 device_sdma 实测基线（135, V4-Flash-w8a8-mtp, TP8, 131072, non-layerwise, 2026-09-08）

| 场景 | 命中率 | prefill TPS | TTFT mean / p99 | host_shm 对比 |
|---|---|---|---|---|
| 单前缀（A2/A3 同文命中） | 96.88% | ≈77K | 1.70s / — | host_shm ≈70K |
| 多前缀（4 前缀灌入 + 32 并发） | 96.88% | 聚合 ≈132.6K | 17.63s / 30.70s | host_shm ≈114K |

> device_sdma 与 host_shm 命中率一致（96.88%），吞吐更高（单前缀 +10%，多前缀 +16%），TTFT 相当。首灌（无缓存）prefill ≈6.5K TPS、≈20s/131K tokens。

# ==== vllm 启动（TP8×DP1, V4-Flash-w8a8-mtp, max_len 131072）====
python -m vllm.entrypoints.openai.api_server \
  --model /data/combinded_nfs/DeepSeek-V4-Flash-w8a8-mtp \
  --port 8004 --tensor-parallel-size 8 --max-model-len 131072 \
  --enable-expert-parallel --tokenizer-mode deepseek_v4 --quantization ascend \
  --no-enable-prefix-caching --block-size 32 --gpu-memory-utilization 0.9 \
  --kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"0","use_layerwise":false}}'

# ==== 验证 ====
curl -s http://127.0.0.1:8004/metrics | grep external_prefix_cache
```

实测结果（2026-09-08 device_sdma）：单前缀命中 96.88%、命中态 TPS ≈77K/TTFT 1.70s；32 并发聚合 TPS 132.6K、TTFT mean 17.63s/p99 30.70s（对比 host_shm 的 114.3K，见 §7.1）。

---

## 8. 相关文档

- 135 完整拉起脚本（device_sdma + 独立端口）：[map_135/start/setup_meta_sdma_135_v2.sh](file:///d:/project/agent_project/map_135/start/setup_meta_sdma_135_v2.sh)
- 135 协议切换脚本（device_sdma 停旧服务/改配置/重启）：[map_135/start/switch_mmc_sdma_135.sh](file:///d:/project/agent_project/map_135/start/switch_mmc_sdma_135.sh)
- 135 vllm 启动脚本（non-layerwise + device_sdma）：[map_135/start/start_v4_nolw_135.sh](file:///d:/project/agent_project/map_135/start/start_v4_nolw_135.sh)
- KV 池验证方法论：`D:\project\.trae\skills\kvpool-verification`（技能内 §3 Memcache 验证）
- 165 基线方法：[Layerwise-Pooling-Optimization/pre_optimization_baseline/01_method.md](file:///d:/project/llm-project/ascend_project/Layerwise-Pooling-Optimization/pre_optimization_baseline/01_method.md)