## 3. memcache 后端拉起规范

> 适用：KV backend 选 memcache 时。**memcache 与 layerwise 验证强相关**——PR14465 第 4 组
> （memcache layerwise）是 DSV4 NSA 硬限根因分析最详的一组，layerwise 验证首选 memcache 路径。
> 参考脚本：`map_51/pr14465_dsv4_kvpool/archive/layerwise_memcache_verify/`
> （start_layerwise_memcache.sh / update_mmc_conf.sh / probe_layerwise_memcache_env.sh）。

### 3.1 与 mooncake 的架构差异（决定拉起方式不同）

| 项 | mooncake | memcache |
|----|----------|----------|
| 独立后端进程 | 有（master，50088/9008，见第 2 章） | **有**（MetaService，等价 mooncake master，**需单独拉起 + 探活 5000/6000/8000**，见 §3.1.1） |
| 配置文件 | mooncake.json（脚本生成） | 容器内 site-packages 的 `mmc-local.conf`（**需手工改**，见 §3.2）+ `mmc-meta.conf`（MetaService 用，见 §3.1.1） |
| 环境注入 | MOONCAKE_* 变量 | `MMC_LOCAL_CONFIG_PATH` + source memcache/memfabric 的 set_env.sh |
| 指标观测 | `curl :9008/metrics`（allocated/key_count） | **有**：`curl :8000/metrics`（Prometheus，`memcache_allocated_bytes`/`memcache_alloc_requests_total` 等，**2026-08-31 实测确认可用**，见 §3.6） |
| lookup_rpc_port | `"1"` | `"0"` |

共同点：`--no-enable-prefix-caching`、长 prompt 门槛、存/取/去重判定逻辑（pool_scheduler/pool_worker 核心代码后端无关）。

#### 3.1.1 MetaService 独立启动（必须，v2 修订：memcache 不是"无 master 进程"）

官方（kv_pool §3.4）明确：**MetaService 是独立进程，只需在单个节点拉起**，等价 mooncake master。
不拉起时 vllm init 连 config_store(6000) 报 `errno:111`（51 腿3 实测根因，曾因 1b 旧版"无独立进程"误述导致该步被忽略）。

**① 启动命令（官方写法，优先）**：

```bash
export MMC_META_CONFIG_PATH={INSTALL_PATH}/memcache_hybrid/config/mmc-meta.conf
python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

- `{INSTALL_PATH}` = `pip show memcache_hybrid` 的 Location
- **51 实测变体**：start_mmc_meta.sh / leg3 ensure_mmc_meta 均**未导出 MMC_META_CONFIG_PATH** 也能跑通（依赖安装路径默认配置）。可用，但新服务器建议用官方显式写法，避免装到非默认路径时读错 conf
- 启动脚本应幂等：vllm 拉起前若 5000/6000 已监听则复用（leg3 ensure_mmc_meta 模式），日志/PID 落盘

**② mmc-meta.conf 关键项**（与 mmc-local.conf 的一致性约束，kv_pool §3.3 Key Focuses）：

| 项 | 要求 |
|----|------|
| `ock.mmc.meta_service_url` | P/D 节点必须配**同一** MetaService 端点 |
| `ock.mmc.local_service.config_store_url` | 必须等于 mmc-meta.conf 的 `ock.mmc.meta_service.config_store_url`；两文件不一致 → init 失败 |
| `ock.mmc.meta_service.metrics_url` | `http://xx:8000`（官方配置项；**2026-08-31 实测确认可用**，`curl :8000/metrics` 返回 Prometheus 指标，见 §3.6） |
| `ock.mmc.local_service.world_size` | LocalService 上限（示例 256），含未来扩容 |

**③ 探活**：确认 5000（meta_service）/ 6000（config_store）/ 8000（metrics）TCP 可连通，并 HTTP 确认 `curl http://127.0.0.1:8000/metrics` 返回 Prometheus 文本，参考 start_mmc_meta.sh：

```python
python - <<'PY'
import socket
for p in (5000,6000,8000):
    s=socket.socket(); s.settimeout(1.5)
    try: s.connect(('127.0.0.1',p)); print(f'{p}: OPEN'); s.close()
    except Exception: print(f'{p}: CLOSED')
PY
```

### 3.2 mmc-local.conf 配置（启动前必做）

配置路径（py 版本不同会变，先 `find /usr/local -maxdepth 8 -name mmc-local.conf` 定位）：
```
/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/mmc-local.conf
```

必改两项：

| 配置项 | 值 | 原因 |
|--------|----|------|
| `ock.mmc.local_service.protocol` | **`device_sdma`**（51 A3 实测跑通；官方默认 host_rdma，**必须改**） | 官方（kv_pool.md:552）protocol 取值矩阵：`device_sdma`=SDMA over device，**推荐 A3**（HCCS 可用时）；**`device_rdma`**=RDMA over device，A2 与 A3（有 device RoCE 时）均支持、**推荐 A2**（跨机/RDMA 场景用这个）；Ascend 950 另设 `device_urma`（UB）/`device_uboe`（UBOE）；其余协议见 memcache 官方 mmc-local.conf。**选型 = 按机器互连（HCCS vs RoCE）二选一**，别把 A3 的 device_sdma 照抄到 A2/RoCE 机器 |
| `ock.mmc.local_service.dram.size` | **1GB 的整数倍**（51 实测 1GB 跑通） | device_sdma 模式要求对齐；⚠️ 官方另有口径：**A3 开 HCCS 时可设 0GB**（kv_pool §3.3 Key Focuses，fabric mem 场景）——两口径并存，实际取值按环境记入 env.txt |

操作纪律（update_mmc_conf.sh 模式）：
1. **先备份** `mmc-local.conf.bak.<timestamp>`
2. sed 替换后 grep 回显确认
3. 幂等（已是 device_sdma 不重复改）

### 3.3 环境变量（memcache 特有）

```bash
export MMC_LOCAL_CONFIG_PATH=<上节的 conf 绝对路径>   # 必须
export LD_LIBRARY_PATH={INSTALL_PATH}/memcache_hybrid/lib:${LD_LIBRARY_PATH}  # 官方示例显式 export（kv_pool §3.5）
# set_env.sh（存在则 source，缺了只 WARN 不阻断）：
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
```

性能相关（start_layerwise_memcache.sh 实测配置，**注意硬件分支**，按需取舍）：
```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_INTRA_ROCE_ENABLE=1      # 对应 protocol=device_rdma 的 A2 分支（官方 required）；A3-HCCS 官方只要 FABRIC_MEM 一组（见下行），51 在 A3 上也设了（无害但勿照抄为 A3 必须）
export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
# A3-HCCS 官方分支（kv_pool §5.1）：ACL_OP_INIT_MODE=1 + ASCEND_ENABLE_USE_FABRIC_MEM=1（措辞 Recommended）
```

### 3.4 kv-transfer-config（backend 切换）

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lookup_rpc_port": "0",
    "backend": "memcache",
    "use_layerwise": true,
    "layerwise_prefetch_layers": 3
  }
}
```

与 mooncake 的差异就两处：`backend=memcache`、`lookup_rpc_port="0"`。
layerwise 验证时 `use_layerwise=true` + `layerwise_prefetch_layers`（预取层数，实测 3）。

**端口命名注明**：官方内部自相矛盾——kv_pool.md 用 `lookup_rpc_port`，layerwise_kv_pool.md 用
`mooncake_rpc_port`。代码（pool_scheduler.py:978-984）两者都收但**推荐 `lookup_rpc_port`**，且对
`mooncake_rpc_port` 打 deprecation 警告（"will be removed in the future"）。**一律用 `lookup_rpc_port`**；
读到官方 layerwise 文档的 `mooncake_rpc_port` 时知道它是将废弃别名。

**layerwise 参数面（M4）**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `use_layerwise` | false | 仅 memcache backend 支持 |
| `layerwise_prefetch_layers` | 1 | 预取层数，提高传输/计算重叠，典型 1–4（实测 3） |
| `layerwise_max_transfer_blocks` | 0（不限） | 单批 transfer 最大块数，防单大层垄断总线 |
| `layerwise_max_transfer_bytes` | 0（不限） | 单批 transfer 最大字节数 |
| `h2d_stagger_us` | 0 | 多 TP rank H2D 拷贝错峰（如 TP8 设 100），缓解总线争用 |
| `discard_partial_chunks` | ⚠️ 见下 | 是否丢弃不完整 chunk 边界 |

⚠️ **`discard_partial_chunks` 文档-代码分歧**：官方 layerwise 文档称"layerwise 默认 false（保留部分层）"，
但本分支代码 pool_scheduler.py:136-137（`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`）
**无条件默认 True**，全 ascend_store 目录未见 layerwise 分支改写。→ 验证"layerwise 保留部分层"语义时
**必须显式设 `discard_partial_chunks: false`**，不能依赖文档默认值。

### 3.5 layerwise 场景的额外启动参数（DSV4 实测集）

第 4 组实际使用的 layerwise 启动参数（相对 non-layerwise 的增量）：

```
--enable-expert-parallel
--disable-hybrid-kv-cache-manager      # ⚠️ 见下方警告
--tokenizer-mode deepseek_v4
--tool-call-parser deepseek_v4 --enable-auto-tool-choice
--reasoning-parser deepseek_v4
--quantization ascend
--block-size 128
--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}'
--async-scheduling
--additional-config '{"ascend_compilation_config": {...}, "enable_cpu_binding": true, ...}'
```

**⚠️ 关键警告**：`--disable-hybrid-kv-cache-manager` **不能**绕过 DSV4 NSA multi-spec 硬限
（它是 vllm 主干的 KV group 调度层开关，与 vllm-ascend 的 `build_layerwise_reuse_layout`
校验层独立，实测仍抛 `Physical layer 2 with multiple cache specs...`）。
加这个参数的原因是 DSV4 hybrid attention 本身的调度需要，不是 layerwise 逃生门。
硬限详情见 §8.1。

另注意 `--enforce-eager` 与 `--compilation-config cudagraph_mode` 语义冲突
（enforce-eager 会覆盖 cudagraph 配置），二选一。

### 3.6 指标观测差异（判定方法必须调整）

memcache **没有** mooncake 的 master_* 指标，但 **MetaService 自带 Prometheus `/metrics` 端点（:8000）**，
2026-08-31 实测确认可用。三维证据链的观测方式：

| 维度 | mooncake | memcache |
|------|----------|----------|
| 存 | `master_allocated_bytes`/`key_count` 增长 | `curl :8000/metrics` 的 `memcache_alloc_requests_total` / `memcache_batch_alloc_requests_total`（counter，看增量）；`memcache_allocated_bytes` 是瞬时 gauge（pool 释放后可为 0，看趋势不看瞬间） |
| 取 | `External prefix cache hit rate` | 同左（vllm 侧指标，后端无关，**依然可用**） |
| 去重 | 重发后 key_count 恒定 | `memcache_alloc_requests_total` 增量观察 + server.log 侧间接 |

**C2 实测裁决（2026-08-31，51 宿主机）**：`curl http://127.0.0.1:8000/metrics` 返回合法 Prometheus 文本
（root `/` 404，仅 `/metrics` 有效），实际输出示例：

```
memcache_total_capacity_bytes{medium="hbm"} 0
memcache_total_capacity_bytes{medium="dram"} 0
memcache_allocated_bytes{medium="hbm"} 0
memcache_allocated_bytes{medium="dram"} 0
memcache_alloc_requests_total 696
memcache_alloc_successes_total 696
memcache_alloc_failures_total 0
memcache_batch_alloc_requests_total 12
memcache_batch_alloc_successes_total 12
```

→ 旧版 1b 断言"memcache 无等价 HTTP 端点"**被实测推翻（C2 方向为"可用"）**；
**存维可用 :8000/metrics 做独立证据源**，不再只能靠"代码路径等价性"论证。

PR14465 第 3 组（memcache non-layerwise）的判定方法：
`use_layerwise=false` 时与已 PASS 的第 1 组**共享同一代码路径**（pool_scheduler/pool_worker 统一实现，
仅 backend 存储实现不同）→ 服务就绪 + 推理正常 + External hit rate 即可判等效 PASS。
**不能**把 mooncake 的 allocated_bytes 标准生搬到 memcache。

### 3.7 启动前探针（probe_layerwise_memcache_env.sh 模式）

memcache 启动失败多为环境缺件，启动前按序核查：

0. **hugepages（M2，layerwise/memcache device transfer 硬性前置，缺了必须 abort 不只是 WARN）**：
   ```bash
   echo 200000 > /proc/sys/vm/nr_hugepages   # 官方 layerwise Prerequisites；持久化见 /etc/sysctl.conf
   grep -E 'HugePages_Total|HugePages_Free' /proc/meminfo   # 确认已分配
   ```
   112 PLAN P11 教训：缺 hugepages 时 memcache device transfer 不可用 → store 层 `batch_copy` 返回 -1、
   请求 0/20（51 腿3 失败主因假设）。**探测脚本中缺失即 FAIL/abort**，不是 WARN。
1. `pip show memcache_hybrid`（包存在 + Location）
2. `find /usr/local -maxdepth 8 -name mmc-local.conf`（配置在）+ protocol 字段按机器分支正确（A3/HCCS→`device_sdma`；A2/device RoCE→`device_rdma`，见 §3.2）
3. `find /usr/local -maxdepth 6 -name set_env.sh | grep -E "memcache|memfabric"`（环境脚本在）
4. **MetaService 存活**（§3.1.1 ③，5000/6000/8000 探活 + `curl :8000/metrics` HTTP 确认；缺进程时 vllm init 连 6000 报 errno:111）
5. `grep -rn "backend.*memcache" $VA/vllm_ascend/`（该代码版本支持 memcache backend）
6. `grep -rn "use_layerwise\|layerwise_prefetch_layers" $VA/vllm_ascend/`（layerwise 支持面）
7. 权重 config.json 关键字段（architectures / o_groups / quantization_config）
8. 端口与 NPU 现状（ss + npu-smi）

### 3.8 失败模式速查（memcache 特有）

| 现象 | 含义 |
|------|------|
| `Store initialization failed` / `memcache_backend.py assert res == 0` | mmc-local.conf 配错（protocol 未按机器分支设置：A3→`device_sdma` / A2→`device_rdma`，或 dram 未对齐）或 set_env 未 source |
| init 连 127.0.0.1:6000 报 `errno:111` | **MetaService 未启动**（§3.1.1），vllm 拉起前先拉起并探活 |
| `Physical layer N with multiple cache specs...` | DSV4 NSA 硬限（见 §3.5 警告），判 EXPECTED_FAIL 非 memcache 问题 |
| `Layerwise ... save batch_copy failed with return code -1` | store 层 transfer 失败，主因假设 = **环境缺件**（hugepages 未设 → device transfer 不可用）；先查 §3.7 探针 0 项 |
| 启动即退 `Configuration loading failed` | MMC_LOCAL_CONFIG_PATH 未导出或路径错（py 版本目录变化） |

---

## 3.9 版本匹配铁律（2026-09-08 135 实测）

- `memcache_hybrid` 用 pip 1.2.0；其 `_pymmc.so` 只兼容 **site-packages 的 memfabric 1.2.0**。若 site-packages 的 memfabric 被 .run 装成 1.2.1，MetaService 会报 `_pymmc import error`（pybind11 类型未注册）。**修复：`pip install memfabric_hybrid==1.2.0 --force-reinstall` 回退。**
- device_sdma 的 SDMA copy 扩展库 `libmf_hybm_copy_extend.so` 在 **1.2.1 编译版**（`/usr/local/memfabric_hybrid/latest/aarch64-linux/lib64`），vllm 进程靠 `source /usr/local/memfabric_hybrid/set_env.sh`（LD_LIBRARY_PATH 优先）加载。
- 即：**site-packages memfabric 1.2.0（喂 MetaService）+ /usr/local memfabric 1.2.1（喂 vllm 的 SDMA copy）** 双轨共存，已验证全链路通。

## 3.10 独立端口变体（宿主机 5000/6000/8000 被占用时）

**坑（历史误判警示）**：早期曾以为 `MetaService.main()` 不读配置文件、改绕路用 `MetaConfig()` 手工设字段（`mc.meta_service_url=...`）。实测证明 **main() 会读 mmc-meta.conf**；当初失败真正原因是 **mmc-meta.conf 里没改成独立端口**（写的还是默认 5000/6000/8000），宿主机已占用 → 启动即挂。**正确做法 = 改 mmc-meta.conf 端口（5001/6001/8001）+ 官方写法**。手工设字段方式也能用（等价于改文件），但不是必须。

mmc-meta.conf 内容（改端口后）：

```ini
ock.mmc.meta_service_url = tcp://127.0.0.1:5001
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:6001
ock.mmc.meta_service.metrics_url = http://127.0.0.1:8001
ock.mmc.log_level = info
```

## 3.11 device_sdma 实测基线参考（135, V4-Flash-w8a8-mtp, TP8, 131072, non-layerwise, 2026-09-08）

| 场景 | 命中率 | prefill TPS | TTFT mean / p99 | host_shm 对比 |
|---|---|---|---|---|
| 单前缀（A2/A3 同文命中） | 96.88% | ≈77K | 1.70s / — | host_shm ≈70K |
| 多前缀（4 前缀灌入 + 32 并发） | 96.88% | 聚合 ≈132.6K | 17.63s / 30.70s | host_shm ≈114K |

> device_sdma 与 host_shm 命中率一致（96.88%），吞吐更高（单前缀 +10%，多前缀 +16%），TTFT 相当。首灌（无缓存）prefill ≈6.5K TPS、≈20s/131K tokens。

---


