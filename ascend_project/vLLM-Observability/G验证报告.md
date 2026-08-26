# PR14912 kv_metrics_observability 冒烟验证报告

- 日期：2026-08-26
- 服务器：141.61.81.51（map_51）
- 容器：kv_metrics_51
- 分支：vllm-ascend `kv_metrics_observability`（commit gcf1296559）/ vllm v0.27.1
- 需求来源：AR20260820031213_vLLM监控平台对接需求分析.md

---

## 一、验证目标

单容器冒烟测试，验证 AscendStore KV pool 的 **4 个监控指标** 在真实请求负载下是否实际上报：

| # | 指标 | 类型 |
|---|------|------|
| 1 | `vllm:kv_pool_load_duration_seconds` | Histogram（label `path`） |
| 2 | `vllm:kv_pool_load_keys_total` | Counter（label `path`） |
| 3 | `vllm:kv_pool_load_failed_keys_total` | Counter（label `path`） |
| 4 | `vllm:kv_pool_delayed_release_requests` | Gauge |

---

## 二、环境与配置

- **NPU 卡**：使用后段空闲卡 `ASCEND_RT_VISIBLE_DEVICES=12,13,14,15`（TP=4，避开前段被占用卡）。
- **模型**：`/mnt/weight/qwen3-32b-pdmix`（TP 4 卡，共 8.51GB/卡 权重）。
- **KV backend**：mooncake，master `127.0.0.1:50088`，metrics `:9008`，`use_layerwise=false`。
- **vllm 关键参数**：`--no-enable-prefix-caching`（禁用本地前缀缓存，强制走 KV pool），`kv_load_failure_policy=recompute`。
- **已应用代码修复**：`ascend_store_connector.py#L305` 由直接访问 `self.connector_scheduler` 改为
  `getattr(self, "connector_scheduler", None)`，修复 worker 角色下 KV 指标收集的 AttributeError。

---

## 三、验证过程

### 3.1 服务启动

mooncake master 常驻（role=leader, state=serving, service_ready=true，端口 50088/9008 OPEN），
vllm server 以 setsid 方式 detached 启动，服务就绪（`PASS: service ready at http://127.0.0.1:8004`）。

### 3.2 请求构造

关键发现：`cache_transfer_granularity = hash_block_size = 128`。短 prompt（<128 tokens）会因
`num_tokens_to_save=0` 触发 `skip_save=True`，不产生任何 KV 存取。因此改用 **5042 token 的长 prompt**：

1. 首次长请求 → KV save 入库（等待 8s 落盘）；
2. 相同长 prompt 再次请求（2、3 次）→ 触发 KV pool load；
3. 短请求 → 触发 delayed-release 路径。

---

## 四、验证结果（4 指标实际上报）

从 `http://127.0.0.1:8004/metrics` 抓取的最终快照：

```
# HELP vllm:kv_pool_load_duration_seconds Histogram of per-request KV cache load duration from the KV pool.
# TYPE vllm:kv_pool_load_duration_seconds histogram
vllm:kv_pool_load_duration_seconds_count{engine="0",model_name="qwen3-32b-kvpool",path="sync"} 8.0
vllm:kv_pool_load_duration_seconds_sum{engine="0",model_name="qwen3-32b-kvpool",path="sync"} 0.156048

vllm:kv_pool_load_keys_total{engine="0",model_name="qwen3-32b-kvpool",path="sync"} 312.0

vllm:kv_pool_load_failed_keys_total{engine="0",model_name="qwen3-32b-kvpool",path="sync"} 0.0

vllm:kv_pool_delayed_release_requests{engine="0",model_name="qwen3-32b-kvpool"} 4.0
```

服务端聚合日志佐证（metrics.py#L103）：

```
KV Transfer metrics: load_count=8, load_avg_ms=19.506, load_p90_ms=26.759,
                     load_keys=312, load_failed_keys=0, delayed_release_requests=3
```

mooncake 存储侧证据：

```
master_allocated_bytes   1409286144  (~1.4 GB 实际落池)
master_key_count         168
master_active_clients    4
External prefix cache hit rate: 62.9%
```

### 逐指标结论

| 指标 | 是否上报 | 数值 | 说明 |
|------|---------|------|------|
| load_duration_seconds | ✅ | count=8, sum=0.156s, avg≈19.5ms, p90≈26.8ms | 每次请求加载耗时，path=sync |
| load_keys_total | ✅ | 312 | 共加载 312 个 pool key |
| load_failed_keys_total | ✅ | 0 | 无失败 key（未触发失败观测路径） |
| delayed_release_requests | ✅ | 4.0 | 当前处于延迟释放窗口的请求数 |

---

## 五、遇到的问题与处理

1. **容器 Exited(255)**：vllm 进程作为 exec 会话子进程随会话退出。改为 detached 容器 +
   `setsid nohup` 启动，进程脱离会话存活。
2. **AttributeError: 'AscendStoreConnector' object has no attribute 'connector_scheduler'**：
   worker 侧无 `connector_scheduler` 属性。已在本地修复并同步，详见上文 2 节。
3. **短 prompt 无任何 KV 指标**：`cache_transfer_granularity=128`，短 prompt 触发 skip_save。
   改用 5042-token 长 prompt 后指标正常上报。**该行为符合预期**（granularity 阈值设计），
   对监控平台而言，属于"该指标在无跨机 KV 复用时不产生数据"的正常表现。

---

## 六、遗留观察项

- `load_failed_keys_total` 恒为 0：本次负载无失败 key，**失败观测路径未被真实覆盖**。
  如需验证失败路径，需构造 KV load 失败场景（如命中校验失败/backend get 失败）。
- 未验证 `use_layerwise=true` 路径（本冒烟仅覆盖默认 sync 路径）。layerwise 的指标
  上报是否正常建议另行验证。
- mooncake master 的 `master_put_start_requests_total` 在长请求前为 0，长请求后
  `master_allocated_bytes/key_count` 增加 —— 说明 KV 确实落池；put/exist 计数与
  AscendStore 内部调度路径的对应关系未逐一对账（指标本身正常）。

---

## 七、结论

**4 个 kv_pool 监控指标均已在真实请求负载下实际上报**，数值合理且与请求行为一致
（同前缀请求触发 load、完成后进入 delayed-release 窗口）。冒烟验证通过。

## 八、相关脚本

- 启动：`map_51/start/kv_metrics_smoke_start.sh`、`map_51/start/launch_smoke.sh`
- 请求/验证：`map_51/test/smoke_long_prompt.sh`、`map_51/test/monitor_smoke.sh` 等
- 结果目录：`map_51/start/results/latest/`