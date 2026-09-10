## 2. mooncake master 拉起规范

> 适用：KV backend 选 mooncake 时（memcache 后端不需要 mooncake master，但有自己的 MetaService 进程，见第 3 章）。
> 参考脚本：`map_51/pr14465_dsv4_kvpool/start/`（历史位置 map_51/start/ 亦有）。

### 2.1 端口规划（硬规则）

| 用途 | 端口 | 原因 |
|------|------|------|
| master RPC | **50088** | 50051 被宿主机 OceanStor DTMA 占用（51 和 112 都踩过），**禁止用 50051** |
| metrics (HTTP) | **9008** | `curl http://127.0.0.1:9008/metrics`。⚠️ **9008 是本库脚本 `--metrics_port` 的覆盖值**，官方 mooncake 默认 **9003**（kv_pool.md §2.2.3）。换服务器时若按官方默认起 master，照抄 9008 必失败——以实际启动参数为准，二选一对齐 |
| vllm API | 8004 | 与 master/metrics 错开 |

启动前先做端口预检（ss/netstat 确认 50088/9008 空闲）。

### 2.2 mooncake.json 配置模板（实测有效）

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "master_server_address": "127.0.0.1:50088",
  "global_segment_size": "5GB",
  "local_buffer_size": "5GB",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true
}
```

要点：
- **每卡 DRAM 配 5GB** 用于 pooling（global segment 5GB）
- ⚠️ **`local_buffer_size` 在 A3 fabric mem 下不生效**：开 `ASCEND_ENABLE_USE_FABRIC_MEM=1` 时，代码（mooncake_backend.py fabric 分支）硬编码 `local_buffer_size=0` 传给 store，官方 FAQ 5.3.2.2 同口径。只有非 fabric mem（A2/ROCE）分支才传递该值。→ A3 每卡实际贡献 = `global_segment_size` 一项，51 的 "local 5GB" 从未生效
- `protocol=ascend`（NPU 场景），metadata 用 P2PHANDSHAKE
- `prefer_alloc_in_same_node=true`：单节点验证默认值。注意改成 false 也**不会**让数据落到远端（见 §8.3，单节点失败路径不可构造的原因）

### 2.3 启动规范（四条纪律）

1. **日志必须重定向**到 `/home/lizhongyang/map_XX/mooncake_logs/`（master.log），禁止依赖终端回显
2. **PID 必须落盘**（master.pid），kill 时用
3. 启动后**探活四查**（全部通过才算 master 就绪）：
   - `role=leader`
   - `state=serving`
   - `service_ready=true`
   - TCP 50088 可连通
4. 等待 vllm worker 注册：`master_active_clients` 应等于 **DP × TP**（如 4×4=16），全部注册完再发请求

探活参考：`probe_mooncake_v2.sh` 模式（role/state/service_ready/TCP 四查）+ master.log `Master Admin Metrics` 行。

**可选调优参数**（官方 kv_pool §2.2.2；51 用默认未传，需要时按官方语义加）：
- `--default_kv_lease_ttl <ms>`：KV 对象租约 TTL，**必须大于 `ASCEND_CONNECT_TIMEOUT` / `ASCEND_TRANSFER_TIMEOUT`**，否则 get 阶段可能 `LEASE_EXPIRED`
- `--eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1`：达到高水位后按比例淘汰
- `--client_ttl <s>`：客户端存活 TTL（默认 10）；`enable_cpu_binding` 场景下 Ping 线程可能被 CPU 迁移错过心跳 → 建议抬到 60–120

**故障关键字速查**（官方 FAQ 5.3.1，出现即定位方向）：
- `NO_AVAILABLE_HANDLE` / `BatchPut failed ... insufficient space` → 淘汰后剩余空间放不下一次 BatchPut：加大容量 / 提高 eviction 余量 / 减小 batch
- `lease_expired_before_data_transfer_completed` / `LEASE_EXPIRED` → get 传输未完成租约已过期：加大 `--default_kv_lease_ttl`
- 既非 put 也非 get 的传输层错误 → 大概率 HIXL (ascend_direct) 问题，收集 `/root/ascend/log/debug/plog`

### 2.4 停止规范

- **显式 PID `kill -9`**（读 master.pid），**禁止依赖 `pkill -f mooncake_master`**——51 上实测杀不死（PID 19863/19865 残留）
- 重启前确认端口已释放，否则新 master 起不来

### 2.5 memcache 后端差异（指向第 3 章）

memcache 后端**有独立后端进程 MetaService**（等价 mooncake master，需单独拉起 + 探活 5000/6000/8000，官方 kv_pool §3.4），并有**自己的 Prometheus `/metrics` 端点（:8000，2026-08-31 实测确认）**，但指标集与 mooncake 的 master_* 完全不同；配置体系、环境变量、layerwise 额外启动参数亦不同——完整规范见第 3 章（**layerwise 验证必读**）。

### 2.6 常用检查命令

```bash
# master 核心指标（存证据用）
curl -s http://127.0.0.1:9008/metrics | grep -E '^master_(allocated_bytes|key_count|active_clients) '

# master.log 状态行（含 batch 速率）
grep "Master Admin Metrics" /home/lizhongyang/map_XX/mooncake_logs/master.log | tail -3

# master 是否活着 + 端口
ss -lntp | grep 50088

# 故障关键字定位（官方 FAQ 5.3.1）
grep -E 'NO_AVAILABLE_HANDLE|LEASE_EXPIRED|BatchPut failed' /home/lizhongyang/map_XX/mooncake_logs/master.log | tail
```

---

