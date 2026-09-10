
# Memcache KV 池 standalone（分离部署）模式：配置与验证（Ascend 800I-A3）

> 适用对象：在 Ascend 800I-A3 上，把 **memcache LocalService 从 vllm 进程里剥离成独立进程**（贡献池），vllm 以 `dram.size=0GB` 接入已有池的部署模式。
> 与 [01_memcache_kv_pool_setup.md](01_memcache_kv_pool_setup.md) 的关系：01 = **co-located（共置）**——vllm 内嵌 LocalService 每 rank 贡献 DRAM；本文 = **standalone（分离）**——独立进程先起、预留池，vllm 后起只接入。
> 实测环境：135（80.5.9.135）pr15367_135 容器，2026-09-09 验证通过。

---

## 0. 为什么需要 standalone（官方 §3.7 依据）

**co-located 的痛**：vllm 先加载模型权重、后初始化 KV connector，MemCache 只能从**剩余可用内存**里分池，可能分不到足够大的池。

**standalone 的解**：在 vllm 启动**前**，每节点独立起一个 MemCache LocalService 进程，先占内存、把池留够；vllm 进程里的 memcache 只连接池、不贡献 DRAM。

- 官方限制：**仅 A3 HCCS 场景支持**。
- 与 PD 分离（prefill/decode 拆进程）**不是一回事**——这里拆的是"MemCache 进程 vs vllm 进程"，不是"P 进程 vs D 进程"。

---

## 1. 架构分层（与 01 对比）

```
co-located（01）                          standalone（本文）
┌─────────────────────────┐              ┌──────────────────────────────┐
│ vllm (rank 0..7)         │              │ standalone LocalService 进程 │ ← 先起，贡献池 dram.size=10GB
│   └ 内嵌 LocalService    │              │   DistributedObjectStore()   │
│      每 rank 贡献 10GB   │              └──────────────┬───────────────┘
└────────────┬────────────┘                             │ 池（rank-1-dram 10GB）
             │ host_shm / device_sdma                   ▼
┌────────────┴────────────┐              ┌──────────────────────────────┐
│ MetaService（常驻单例）  │◄─────────────│ vllm (TP8, dram.size=0GB)      │ ← 后起，只接入不贡献
└─────────────────────────┘              │   8 rank 全连同一个池          │
                                         └──────────────┬───────────────┘
                                                        │
                                         ┌──────────────┴──────────────┐
                                         │ MetaService（常驻单例）        │
                                         └─────────────────────────────┘
```

| 项 | co-located（01） | standalone（本文） |
|---|---|---|
| 池贡献者 | vllm 每个 rank（rank-N-dram，各 10GB） | **独立进程**（实测注册为 rank-1-dram，10GB） |
| vllm 侧配置 | dram.size = 10GB | **dram.size = 0GB** + max.dram.size = 1024GB |
| 池注册时机 | 首个推理请求时才注册（懒） | **init 即注册（eager）**，metrics 立即可见容量 |
| 部署顺序 | MetaService → vllm | MetaService → standalone 进程 → 等初始化 → vllm |
| 适用 | 通用 | 仅 A3 HCCS；需给池留大内存时 |

---

## 2. 配置：两份 mmc-local conf

连接/协议完全一致，只差 DRAM 贡献：

```ini
## mmc-local-standalone.conf —— 独立 LocalService 进程用（贡献池）
ock.mmc.meta_service_url = tcp://127.0.0.1:5001
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6001
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 10GB          # 135 实测值（见 §5 坑：80GB 不可用）
ock.mmc.local_service.max.dram.size = 1024GB
ock.mmc.local_service.world_size = 256
ock.mmc.log_level = info
```

```ini
## mmc-local-vllm.conf —— vllm 进程用（不贡献 DRAM，只接入）
ock.mmc.meta_service_url = tcp://127.0.0.1:5001
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6001
ock.mmc.local_service.protocol = device_sdma
ock.mmc.local_service.dram.size = 0GB           # 关键：0 = 不贡献
ock.mmc.local_service.max.dram.size = 1024GB
ock.mmc.local_service.world_size = 256
ock.mmc.log_level = info
```

> 注意对齐约束：A3 下 dram.size 必须 1GB 对齐；"HBM 与 DRAM 不能同时为 0"的报错约束，实测 0GB DRAM + max 1024GB 可正常 init（官方 §3.7 原样配置）。

---

## 3. 拉起步骤（135 实测，2026-09-09）

### 3.1 前置：hugepages（硬条件，同 01 §1）

```bash
grep HugePages_Total /proc/meminfo      # 期望 200000（135/165 通用口径）
echo 200000 > /proc/sys/vm/nr_hugepages # 缺失即 abort
```

### 3.2 MetaService（一次拉起，官方写法）

```bash
export MMC_META_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/mmc-meta.conf
nohup python3 -u -c "from memcache_hybrid import MetaService; MetaService.main()" \
  > /home/lizhongyang/map_135/memcache_logs/meta_standalone.log 2>&1 &
```

端口探活 5001/6001/8001 全 OPEN（mmc-meta.conf 需已是独立端口，见 01 §2.4）。

### 3.3 standalone LocalService 进程（本文核心，独立于 vllm）

进程内容（等价于 memcache 源码 example/benchmark 的多进程 worker 模式：acl 设卡 + store.init）：

```python
# standalone_local_service.py（135 完整版见 map_135/start/）
import acl
acl.init()
acl.rt.set_device(0)                    # 必须：device_sdma 池在设备侧，需先设卡
from memcache_hybrid import DistributedObjectStore
store = DistributedObjectStore()
res = store.init(0, init_bm=True)       # 读 MMC_LOCAL_CONFIG_PATH（standalone conf）
# res == 0 后 keep-alive（while True: sleep）
```

启动（device_sdma 必须 source memfabric 1.2.1 set_env，同 01 §3.1 铁律）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export MMC_LOCAL_CONFIG_PATH=/home/lizhongyang/map_135/mmc-local-standalone.conf
export LD_LIBRARY_PATH=<site-packages>/memcache_hybrid/lib:${LD_LIBRARY_PATH}
nohup python3 -u /home/lizhongyang/map_135/standalone_local_service.py \
  > /home/lizhongyang/map_135/memcache_logs/standalone_ls.log 2>&1 &
```

**验证池注册（eager，无需等请求）**：

```bash
curl -s http://127.0.0.1:8001/metrics | grep -E "segment_capacity|total_capacity"
# 期望（135 实测）：
#   memcache_segment_capacity_bytes{segment="rank-1-dram"} 10737418240   # 10GB
#   memcache_total_capacity_bytes{medium="dram"}            10737418240
```

### 3.4 vllm 接入（TP8 × DP1，V4-Flash 非 layerwise，device_sdma）

与 01 §3.2 相同，仅两处不同：
- `MMC_LOCAL_CONFIG_PATH` → `mmc-local-vllm.conf`（dram.size=0GB）
- 其余启动参数完全一致（`--kv-transfer-config` backend=memcache、use_layerwise=false、`--no-enable-prefix-caching`）

135 完整启动脚本：`map_135/start/start_v4_standalone_135.sh`。

---

## 4. 验证结果（135，V4-Flash-w8a8-mtp，TP8×DP1，max_len 131072，non-layerwise，device_sdma，2026-09-09）

### 4.1 冒烟

- `/v1/models` 正常；小请求 `external_prefix_cache_queries_total` 从 0 → 1（KV 池查询链路活）
- 池容量在 **standalone 进程**名下：rank-1-dram 10GB，vllm 侧 0 DRAM

### 4.2 单前缀（131070 token 同文 ×3，首灌 + 2 次命中）

| 请求 | prompt | lat | ext_q | ext_h | 命中率 | prefill TPS |
|---|---|---|---|---|---|---|
| A（首灌） | 131070 | 19.88s | 131070 | 0 | 0% | 6591.8 |
| A2（同文） | 131070 | 1.72s | 131070 | 126976 | **96.88%** | 76360 |
| A3（同文） | 131070 | 1.69s | 131070 | 126976 | **96.88%** | 77717.7 |

### 4.3 多前缀（4×131070 灌入 + 32 并发同前缀集，1 token 输出）

| 项 | 值 |
|---|---|
| 预热 | 4 前缀共 76.6s（每条约 19s） |
| 并发结果 | **32/32 成功** |
| 命中率 | **96.88%**（ext_h 4063232 / ext_q 4194240） |
| 并发耗时 | 31.0s |
| TTFT mean / p99 | 17.26s / 30.03s |
| 聚合 prefill TPS | **135.4K** |

### 4.4 与 co-located（01 §7.1 device_sdma 基线）对比

| 指标 | standalone | co-located（01） | 结论 |
|---|---|---|---|
| 单前缀命中率 | 96.88% | 96.88% | 一致 |
| 单前缀命中态 TPS | ≈76-78K | ≈77K | 一致 |
| 单前缀 TTFT | ~1.7s | 1.70s | 一致 |
| 多前缀命中率 | 96.88% | 96.88% | 一致 |
| 多前缀聚合 TPS | 135.4K | ≈132.6K | 一致（±2%） |
| 多前缀 TTFT mean / p99 | 17.26s / 30.03s | 17.63s / 30.70s | 一致 |

**结论：standalone 模式下 KV 池存取/复用行为与 co-located 完全一致**——命中率、TPS、TTFT 无回归；差异仅在"池由独立进程贡献、vllm 以 0 DRAM 接入"这一部署形态。

---

## 5. 坑与教训（135 实测）

| 坑 | 现象 | 解法 |
|---|---|---|
| **大池 80GB 起不来** | `HalMemCreate ret:6`（1GB page 失败）→ 回退 2MB page 后**卡死无输出** | 降级 10GB（135 实测可用；与 165 128G ret:6 同类——池子按需下调，A3 有对齐/内存约束） |
| **独立进程必须设卡** | 不起 acl / 不 set_device，device_sdma 池分配失败或行为异常 | 拷贝 memcache 源码 example/benchmark 多进程 worker 模式：`acl.init()` + `acl.rt.set_device(0)` 后再 `store.init` |
| **vllm 侧误配成 standalone conf** | vllm 自己也贡献 DRAM，两个进程抢池 | vllm 必须用 `mmc-local-vllm.conf`（dram.size=0GB），standalone 进程用 standalone conf，两份连接/协议一致 |
| standalone 段注册时机 | init 即 eager 注册（与 co-located 首次请求才注册不同） | 正常现象；用 8001 metrics 的 `segment_capacity` 直接验证，不用等请求 |
| 段名是 rank-1-dram | 独立进程 init 后段显示 `rank-1-dram`（非 rank-0） | 编号是 MetaService 侧分配，不影响使用，以 metrics 实际值为准 |

---

## 6. 脚本清单（map_135/start/）

| 脚本 | 用途 |
|---|---|
| `mmc-local-standalone.conf` / `mmc-local-vllm.conf` | 两份配置（3.3/3.4） |
| `standalone_local_service.py` | 独立 LocalService 进程（3.3） |
| `setup_standalone_135.sh` | 一键：清残留 → MetaService → standalone 进程 → 端口/池探活 |
| `restart_standalone_ls_135.sh` | 仅重启 standalone 进程（MetaService 不动，改池大小后重试） |
| `start_v4_standalone_135.sh` | vllm 启动（0GB 接入） |
| `wait_v4_standalone_ready_135.sh` | 就绪轮询 |
| `smoke_standalone_135.sh` | 冒烟 + 计数检查 |
| `run_bench_standalone_135.sh` | 前缀基线（single/multi）→ results/latest/ |

---

## 7. 相关文档

- co-located 完整手册：[01_memcache_kv_pool_setup.md](01_memcache_kv_pool_setup.md)
- 官方依据：vllm-ascend kv_pool.md §3.7 Separated Deployment of MemCache and vLLM（仅 A3 HCCS）
- 独立进程模式参考：memcache 源码 `example/benchmark/`（mutil_process.py / bench_start.sh 多进程 LocalService）
- 135 证据：`results/latest/bench_standalone_{single,multi}.log` + `memcache_logs/standalone_ls_*.log`