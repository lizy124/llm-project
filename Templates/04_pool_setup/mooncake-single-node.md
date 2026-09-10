# Mooncake KV 池单机基线：配置与验证（Ascend 800I-A3 128G）

> 适用对象：在 Ascend 800I-A3 **128G** 单机上，用 **mooncake-transfer-engine-npu** 作为 KV Cache 池后端（非 layerwise），验证 存/取/去重 + 命中率/TTFT/TPS 基线。
> 与 [01_memcache_kv_pool_setup.md](01_memcache_kv_pool_setup.md) / [01b_memcache_standalone.md](01b_memcache_standalone.md) 的关系：同一 vllm `AscendStoreConnector` 框架，仅 `kv_connector_extra_config.backend` 不同（mooncake vs memcache）。
> 实测环境：135（80.5.9.135）pr15367_135 容器，2026-09-09 验证通过。属 A3_HCCS 128G Phase 1.1（见 `A3_HCCS/128G/verification_plan.md` §5.1.1）。

---

## 1. 架构与端口

```
┌─────────────────────────────── 单机 (135, A3 560T 128G, 16 die) ───────────────────────────────┐
│                                                                                               │
│  ┌──────────────┐   RPC(50088)    ┌───────────────────────────┐                               │
│  │ mooncake_master│◄─────────────│ vllm (TP8 x DP1, 8004)      │                               │
│  │ (50088/9008)  │                │   AscendStoreConnector     │                               │
│  │ role=leader   │                │     └ MooncakeDistributedStore                             │
│  │ service_ready │                │          (读 mooncake.json, 每 worker 1 客户端)             │
│  └──────────────┘                └────────────┬────────────────┘                               │
│       ▲                                      │ fabric mem (ASCEND_ENABLE_USE_FABRIC_MEM=1)     │
│       └─────────── metrics(9008) ─────────────┘                                               │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

| 项 | 值 | 说明 |
|---|---|---|
| master RPC | **50088** | 避开宿主机 OceanStor DTMA 的 50051（51/112/135 均踩过） |
| master metrics | **9008** | 脚本覆盖值，官方默认 9003 |
| vllm API | 8004 | |
| 池容量 | global_segment_size **10GB**（135 实测，5GB 官方默认亦可） | A3 fabric mem 下 local_buffer_size 不生效 |
| 客户端数 | 8 = TP8 × DP1 | 探活用 master_active_clients |

---

## 2. 配置

### 2.1 mooncake.json（vllm 侧 store 配置，`MOONCAKE_CONFIG_PATH` 指定）

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

### 2.2 关键环境变量

```bash
export MOONCAKE_CONFIG_PATH=/home/lizhongyang/map_135/run/mooncake.json
export PYTHONHASHSEED=0
export VLLM_USE_V1=1
export ACL_OP_INIT_MODE=1
export ASCEND_ENABLE_USE_FABRIC_MEM=1        # A3 HCCS 必须
export ASCEND_RT_VISIBLE_DEVICES=0..7
```

### 2.3 kv-transfer-config

```json
{ "kv_connector": "AscendStoreConnector", "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": { "backend": "mooncake", "lookup_rpc_port": "1", "use_layerwise": false } }
```

> 与 memcache 差异仅：`backend=mooncake`、`lookup_rpc_port="1"`（memcache 用 "0"）。

---

## 3. 启动

### 3.1 mooncake master（二进制直启，日志/PID 落盘）

```bash
nohup /usr/local/python3.11.10/bin/mooncake_master \
  --rpc_port 50088 --metrics_port 9008 --logtostderr=true \
  > /home/lizhongyang/map_135/mooncake_logs/master.log 2>&1 &
echo $! > /home/lizhongyang/map_135/mooncake_logs/master.pid
```

> ⚠️ 官方文档写法 `python -m mooncake_transfer_engine.mooncake_master --master_port --config` 在 0.3.11 实测用**二进制 + `--rpc_port/--metrics_port`** 才正确。

### 3.2 探活四查（全过才认为 master 就绪）

- `role=leader`、`state=serving`、`service_ready=true`、TCP 50088/9008 连通
- 启动后 10s 内从 standby/starting 自动转 leader/serving

### 3.3 vllm（P+D 同机，kv_both）

启动参数与 memcache 完全一致（TP8/DP1/131072/`--no-enable-prefix-caching`/`--enforce-eager` 等），仅 kv-transfer-config 不同。脚本：`map_135/start/start_v4_mooncake_135.sh`。

---

## 4. 验证结果（135，2026-09-09）

基准口径：131070-token 前缀、输出 1 token、`--no-enable-prefix-caching`（HBM 前缀缓存关闭）。

### 4.1 单前缀（存 → 取 → 再取）

| 场景 | 说明 | TTFT | 命中率 | Prefill TPS |
|---|---|---|---|---|
| A | 首灌（冷，含池存储） | 137.37s | 0% | 954 |
| A2 | 同前缀重发（命中） | **1.78s** | **96.88%** | 73694 |
| A3 | 再重发（命中） | **1.77s** | 96.88% | 74170 |

### 4.2 多前缀 + 并发（4 × 131070 灌入 + 32 并发）

- 灌入：4 个前缀 **76.6s**（warm0=19.47s = 稳态冷灌，≈memcache 19.88s）
- 并发：**32/32 ok**，TTFT mean 18.47s / p99 32.24s，聚合 prefill **126.2K TPS**

### 4.3 存 / 取 / 去重三维证据链

| 维度 | 证据 |
|---|---|
| 存 | master_allocated_bytes 0 → 1.34GB(single) → 6.71GB(multi)；key_count 0 → 1891 → 9455；active_clients **8** |
| 取 | vllm external_prefix_cache hits：5.11M→4.32M / 5.50M→4.70M 累计，A2/A3 96.88% |
| 去重 | 重发同前缀后 **key_count 9455→9455 不变**、allocated 不变、命中 96.88%（无重复落盘） |
| 健康 | master PutStart 2695 = PutEnd 2695，**失败 0、淘汰 0** |

### 4.4 HBM PrefixCache 关闭（P0 要求）

- `vllm:prefix_cache_queries_total 0.0` / `vllm:prefix_cache_hits_total 0.0`（HBM 缓存从未查询）
- 全部命中来自 `vllm:external_prefix_cache_*`（KV 池）

### 4.5 与 memcache 对比（同口径同 bench）

| 指标 | mooncake | memcache | 判定 |
|---|---|---|---|
| 命中率 | 96.88% | 96.88% | 一致 |
| 命中态 TTFT | 1.78s / 1.77s | 1.72s / 1.69s | 同量级 |
| 稳态冷灌 | 19.47s | 19.88s | 一致 |
| 加速比（vs 冷灌） | ~10.9x | ~11.6x | 同量级 |
| 多前缀 32 并发聚合 TPS | 126.2K | 135.4K | 同量级 |

---

## 5. 坑与教训

1. **A 首灌 137.4s ≠ 性能问题**：首次 KV 保存触发 mooncake store **惰性初始化**（日志 `12:05:13 MooncakeBackend.exists called before store initialization`）+ 10GB global segment 分配，一次性成本；稳态冷灌 = 19.47s。若首灌时间敏感，可提前发一个冷请求预热 store。
2. **master 端口**：必须 50088（宿主机 50051 被 OceanStor DTMA 占用）；metrics 用 9008。
3. **mooncake 不需要 hugepages**（memcache 需要），也不需 source memfabric/memcache 的 set_env。
4. **global_segment_size**：官方默认 5GB；135 为与 memcache 10GB 池对齐、安全容纳 4×131070 前缀，实测用 10GB 无分配问题。
5. **客户端数探活**：`master_active_clients` 应 = DP×TP（135 为 8），全注册后再压测。
6. **内联命令带 `|` 的 grep 在 go.ps1 传输会裂**：一律用脚本文件或单 pattern。

---

## 6. 证据归档

| 文件 | 位置 |
|---|---|
| bench single / multi | `map_135/results/latest/bench_mooncake_{single,multi}.log` |
| master 指标（前后） | `map_135/results/latest/bench_mooncake_master_{before,after_single,after_multi}.txt` |
| P0 补全（HBM-0 + 去重） | `map_135/results/latest/phase1_gaps_evidence.txt` |
| status（stage/PID/next） | `map_135/results/latest/status.txt`（本地+服务器） |
| 服务器完整日志 | `/home/lizhongyang/map_135/run/v4_mooncake_131072.log`、`mooncake_logs/master.log` |
| 启动/探活/轮询脚本 | `map_135/start/start_mooncake_135.sh`、`start_v4_mooncake_135.sh`、`probe_mooncake_135.sh`、`poll_vllm_ready_135.sh`、`verify_phase1_gaps_135.sh` |

---

## 7. 结论

mooncake 单机非 layerwise 基线 **PASS**：存/取/去重三维证据链完整，命中率/TTFT/TPS 与 memcache 一致（96.88% 命中、~11x 加速、126K 聚合 TPS），HBM PrefixCache 关闭由计数器证明。阶段结论同步至 `A3_HCCS/128G/stage_summary.md`（Phase 1.1 ✅）。