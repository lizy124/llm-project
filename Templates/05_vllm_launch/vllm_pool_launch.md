## 4. vllm 池化场景启动

> 池化验证的 vllm 启动与普通推理不同：必须禁用本地 prefix caching、配 kv-transfer-config、
> 等待时间远超常规。参考实现：`map_51/pr14465_dsv4_kvpool/start/dsv4_mooncake_non_layerwise_start.sh`
> （通用 starter，KV_BACKEND × USE_LAYERWISE 环境变量切换）。

### 4.1 关键启动参数

```
--no-enable-prefix-caching          # 必须！否则本地命中抑制 KV pool 存取（见 §4.1.1）
--kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both",
   "kv_load_failure_policy":"recompute",
   "kv_connector_extra_config":{"backend":"mooncake","lookup_rpc_port":"1","use_layerwise":false}}'
```

`kv_connector_extra_config` 三要素：

| 键 | mooncake | memcache |
|----|----------|----------|
| `backend` | `mooncake` | `memcache` |
| `lookup_rpc_port` | `"1"` | `"0"` |
| `use_layerwise` | 按验证矩阵 | 按验证矩阵（layerwise 可加 `layerwise_prefetch_layers: 3`） |

⚠️ **`kv_load_failure_policy=recompute` 有模型限制（M5）**：官方 kv_pool §1 明确 **hybrid attention
模型不支持 recompute**（DeepSeekV4、Qwen 3.5 等），且 vllm 默认值是 `fail`。→ hybrid 模型必须用默认
`fail`（或显式设）；Qwen3（GQA 非 hybrid）、DSV2-Lite 等才可用 recompute。DSV4 腿注意此约束。

其他常规参数：`--enforce-eager --trust-remote-code`、TP/DP 按空闲 chip 数、
`--max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 20 --gpu-memory-utilization 0.9`。

#### 4.1.1 为什么必须 --no-enable-prefix-caching

开启 prefix caching 时，重复前缀本地命中，connector 认为无需外部存取 → KV pool 指标全 0，
**误判为池化未生效**。验证日志中 `Prefix cache hit rate: 0.0%` 是禁用生效的确认（本身也是一条证据）。

### 4.2 关键环境变量

```
PYTHONHASHSEED=0                 # 必须！官方 required：跨节点统一 hash 生成（kv_pool §1；memcache/layerwise 同样需要，layerwise Prerequisites 显式列出）
VLLM_USE_V1=1                    # 强制 v1 引擎
VLLM_ENGINE_READY_TIMEOUT_S=2400 # 大权重加载 20-40 分钟
HCCL_BUFFSIZE=1024
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# A3 专用：
ACL_OP_INIT_MODE=1
ASCEND_ENABLE_USE_FABRIC_MEM=1
```

- `ASCEND_RT_VISIBLE_DEVICES` 用**逗号分隔**（`0,1,2,...`），不能用 `0..15` 范围写法
- mooncake 后端：导出 `MOONCAKE_CONFIG_PATH`（指向 mooncake.json）、`MOONCAKE_MASTER=127.0.0.1:50088`
- memcache 后端：导出 `MMC_LOCAL_CONFIG_PATH`、source memcache/memfabric 的 set_env.sh、
  mmc-local.conf 需提前改 protocol（详见第 3 章；layerwise 场景的额外启动参数也在 §3.5）

### 4.3 启动脚本模板要点（照抄 dsv4 starter 模式）

每个验证场景的启动脚本应包含（该模式在 4 组矩阵验证中稳定使用）：

1. **结果目录**：`results/<YYYYMMDD_HHMMSS>/` + `latest` 软链，内含：
   - `server.log`（完整日志，重定向）
   - `server.pid`、`status.txt`（RUNNING → PASS/FAIL/BLOCKED）
   - `env.txt`（**版本快照**：vllm/vllm-ascend pip version + location、git commit、npu-smi、全部关键环境变量——失败回溯的第一现场）
   - `command.sh`（实际命令行 printf %q 记录）
   - `mooncake.json`（mooncake 后端时，脚本生成）
2. **前置检查**：模型 config.json 存在、toolkit/ATB env 存在、端口空闲、可见设备数 ≥ TP
3. **启动**：后台 + 日志重定向 + PID 落盘
4. **READY 轮询**：每 10s 查 `/v1/models`，超时 2400s；进程死掉时 tail 日志并按关键字分类
   （`layerwise_cache_layout` → BLOCKED「layerwise 不支持该模型」而非 FAIL，见 §8.1）

### 4.4 启动失败速查（池化特有）

| 现象 | 含义 |
|------|------|
| `RuntimeError: Worker failed ... layerwise_cache_layout` | 模型触发 multi-spec 硬限，见 §8.1 |
| `Initialize mooncake failed` / `Connection refused` | master 未起 / 端口配错（查 50088） |
| 主 vllm 活着但 8004 不监听、Worker 卡 do_poll | EngineCore 已崩但未发终止信号 → 手动 kill 主 PID + `pkill -9 -f VLLM::`，看 server.log 找崩溃堆栈 |
| `Failed to infer device type` | NPUPlatform 插件加载失败，查 plugin 加载阶段 |
| KV cache spec 校验阶段 HBM 只到 baseline（~18GB/die）就停 | shard loading 未完成即被 spec 校验挡下 |

### 4.5 就绪后第一件事

`/v1/models` 通了 ≠ 池化就绪。确认：
1. master.log 出现 `role=leader, state=serving, service_ready=true`
2. `master_active_clients` = DP×TP（全部 worker 注册完成）
3. server.log 中 N 个 `AscendStoreConnector` 创建成功（N = DP×TP，DSV4 4×4=8 connector/4DP 实测）

然后才进入测试（见第 5 章）。

---

