# PR1 开发计划：池化 KV Cache 指标埋点（AscendStore 形态）

> 需求编号：AR20260820031213（验收标准 2/3）
> 开发分支：`kv_metrics_observability`（基于 vllm-ascend main `fa668c649`）
> 配套 vllm 版本：`ba07e4a48f`（vllm-ascend `.github/vllm-main-verified.commit` 声明的验证提交）
> 文档时间：2026-08-25

---

## 一、范围与目标

PR1 交付**池化（AscendStore）形态**的两个指标，覆盖 mooncake / memcache（含 layerwise）两种 backend：

1. **池化 KV Cache 加载耗时**（请求粒度，Histogram）——验收标准 2
2. **KV Cache 延迟释放请求数**（Gauge）——验收标准 3（池化部分）

PD 分离形态（mooncake_connector / hybrid / layerwise_connector）留到 PR2。

**零上游改动**：所有代码在 vllm-ascend 侧，复用上游 vLLM 已有的 KV Connector 指标框架。

---

## 二、版本兼容性确认结论（已验证）

`ba07e4a48f` 下指标框架完整可用：

| 组件 | 位置（vllm 仓库） | 状态 |
|---|---|---|
| `KVConnectorStats` / `KVConnectorLogging` / `KVConnectorPromMetrics` / `KVConnectorProm` | `vllm/distributed/kv_transfer/kv_connector/v1/metrics.py` | 存在 |
| 三个 connector 钩子：`get_kv_connector_stats` / `build_kv_connector_stats` / `build_prom_metrics` | `.../kv_connector/v1/base.py`（L424/L664/L703） | 存在 |
| Worker 侧自动拉取点 | `vllm/v1/worker/kv_connector_model_runner_mixin.py:107`，每步调 `connector.get_kv_connector_stats()` 塞入 `KVConnectorOutput` | 自动触发，无需接线 |
| TP 聚合 | `vllm/distributed/kv_transfer/kv_connector/utils.py:122-131`（`KVOutputAggregator` 调 `stats.aggregate()`） | 自动 |
| Scheduler 侧拉取点 | `vllm/v1/core/sched/scheduler.py:2097-2107`，与 worker stats 合并 | 延迟释放 Gauge 的出口 |
| Prometheus 出口 | `vllm/v1/metrics/loggers.py:1145-1147`（`KVConnectorProm.observe`，带 engine_idx） | 自动 |
| 文本日志出口 | `loggers.py:221-222`（`KVConnectorLogging.observe` → "KV Transfer metrics: ..."） | 自动 |
| 树内参考实现 | `.../mooncake/store/metrics.py`：`MooncakeStoreConnectorStats` + `MooncakeStorePromMetrics` | **直接模板** |

**数据流**（实现只需关注前两段）：

```
[Worker 进程] 埋点 → KVPoolWorker.get_stats()
    → mixin 每步调用 get_kv_connector_stats() 取走
    → KVOutputAggregator 跨 TP rank 调 aggregate()
[Scheduler 进程] connector.get_kv_connector_stats()（scheduler 角色）
    → 与 worker stats 合并（scheduler.py:2107）
[Engine 进程] 重建 stats → KVConnectorLogging（日志）+ KVConnectorProm（/metrics）
```

---

## 三、指标定义

### 指标 1：池化 KV Cache 加载耗时

| 项 | 值 |
|---|---|
| 名称 | `vllm:kv_pool_load_duration_seconds`（Histogram） |
| 语义 | 单个请求从 KV 池加载 KV Cache 的墙钟耗时：首个 `backend.get()` 提交 → 该请求全部块加载完成 |
| 标签 | 框架自带 label（engine_idx 等），可选加 `path`（sync/async/layerwise） |
| 桶 | 参考 mooncake store：1ms~5s 对数桶 |
| 伴生指标 | `vllm:kv_pool_load_keys_total`（加载的 key/block 数，用于折算带宽）、`vllm:kv_pool_load_failed_keys_total`（加载失败，异常状态方向） |

### 指标 2：KV Cache 延迟释放请求数

| 项 | 值 |
|---|---|
| 名称 | `vllm:kv_cache_delayed_release_requests`（Gauge） |
| 语义 | 当前处于延迟释放窗口（请求已结束、KV block 留待异步 save 后释放）的请求数，即 `len(pool_scheduler._delayed_free_req_ids)` 的快照 |
| 出口 | scheduler 侧 stats（scheduler.py:2102 拉取） |

命名与需求分析 4.1 节推测名（`vllm:kv_transfer_load_duration_seconds`）的差异：`kv_pool_` 前缀区分"池化加载"与 PR2 的"PD 分离传输"，最终命名 review 时定。

---

## 四、开发步骤

### Step 1：新建 stats + prom metrics 模块

文件：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py`（新建）

仿照 mooncake store 的 `metrics.py`：

- `AscendStoreKVConnectorStats(KVConnectorStats)`：
  - `data: dict[str, list[dict]]`（**必须纯 primitive**，跨进程 msgpack 序列化）
  - worker 侧记录：`record_load(req_id, duration_seconds, num_keys, num_failed_keys, path)` 追加到 `data["load"]`
  - scheduler 侧记录：`data["delayed_release"]` 存最新快照值（ Gauge 语义：interval 内最后一次观测）
  - 实现 `reset` / `is_empty` / `aggregate`（列表拼接；delayed 快照取 max 或 last）/ `reduce`（输出 count/avg/p90、delayed_last）
- `AscendStorePromMetrics(KVConnectorPromMetrics)`：
  - 注册 Histogram（load_duration）、Counter（load_keys / failed_keys）、Gauge（delayed_release）
  - `observe(transfer_stats_data, engine_idx)`：遍历 records 逐条 observe；gauge 用 `set()` 最新值
  - 用 `create_metric_per_engine` 处理多 engine label（参考 HF3FS L1140）

### Step 2：worker 侧加载计时埋点

文件：`pool_worker.py`（修改）

三条加载路径都要埋（同一 `record_load` 接口，`path` 标签区分）：

| 路径 | 计时起点 | 完成点 |
|---|---|---|
| sync（默认） | `start_load_kv()` 中构造 key_list 前 | `self.m_store.get()` 返回后（约 L933） |
| async（`load_async=True`） | `kv_recv_thread.add_request()` 前 | `get_finished()` 中 `get_and_clear_finished_requests()` 返回的 `done_recving` 集合（约 L2051） |
| layerwise | `process_layer_data()` 首层提交 | 末层完成（`_cache_write_events` 全部置位 / 请求层加载完成判定处） |

要点：
- 计时用 `time.perf_counter()`（纯 CPU 操作，无 NPU `item()` 同步问题）
- 起止时间存在 per-request dict（`req_id → t0`），完成时结算并从 dict 移除，防泄漏
- stats 对象挂在 `KVPoolWorker` 上，`get_stats()` 返回并 `reset()`（框架每步取走）

### Step 3：scheduler 侧延迟释放快照

文件：`pool_scheduler.py`（修改）

- `KVPoolScheduler` 持有一个 stats 对象；在 `build_connector_meta()` 或 `get_stats()` 被调时记录 `len(self._delayed_free_req_ids)` 快照
- 注意：`_delayed_free_req_ids` 的 add/discard 在 `request_finished()`（L947/L974）和 `update_finished_sending()`（L989）等处，**不在这些热点逐点埋点**，只在被拉取时快照，避免热路径开销

### Step 4：connector 三个钩子

文件：`ascend_store_connector.py`（修改）

`AscendStoreConnector` 按 `role` 已分派 `connector_scheduler` / `connector_worker`（构造函数 L107-120），`get_kv_connector_stats` 同样按角色转发：

```python
def get_kv_connector_stats(self) -> KVConnectorStats | None:
    if self.connector_scheduler is not None:
        return self.connector_scheduler.get_stats()
    return self.connector_worker.get_stats()

@classmethod
def build_kv_connector_stats(cls, data=None):
    return AscendStoreKVConnectorStats(data=data) if data is not None else None

@classmethod
def build_prom_metrics(cls, vllm_config, metric_types, labelnames, per_engine_labelvalues):
    return AscendStorePromMetrics(...)
```

注意：worker 进程和 scheduler 进程各自实例化 connector，两路 stats 在 scheduler.py:2107 自动合并，`aggregate()` 要能合并 load 记录列表与 delayed 快照。

### Step 5：UT

文件：`tests/ut/distributed/ascend_store/test_kv_pool_metrics.py`（新建）

- Stats：record → aggregate（两 rank 模拟）→ reduce 的数值正确性；空 stats 的 `is_empty`
- 跨进程序列化：`data` 经 msgpack round-trip 后 `build_kv_connector_stats(data)` 重建无损
- PromMetrics：mock metric_types，observe 后断言 histogram/counter/gauge 收到正确值
- 遵循现有 UT 风格（PEP 8：类间 2 空行、方法间 1 空行）

### Step 6：服务器验证

按既有工作流（本地写脚本 → scp → ssh 执行，测试记录独立目录）：

1. **冒烟两形态**：memcache backend（layerwise 开/关）+ mooncake backend，`curl /metrics` 确认两个指标暴露
2. **指标行为验证**：
   - 加载耗时：prefix 命中场景有观测值、全 miss 场景无观测
   - 延迟释放：enable/disable prefix caching 对照，数值与 `_delayed_free_req_ids` 日志一致
3. **回归**：现有 `tests/ut/distributed/ascend_store/` 全量通过

---

## 五、文件清单

| 文件 | 动作 | 内容 |
|---|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py` | 新建 | Stats + PromMetrics 两个类 |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` | 修改 | 三条路径加载计时（+`get_stats()`） |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` | 修改 | 延迟释放快照（+`get_stats()`） |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py` | 修改 | 三个钩子 override |
| `tests/ut/distributed/ascend_store/test_kv_pool_metrics.py` | 新建 | UT |

预计生产代码约 300~400 行 + UT 约 200~300 行，满足 PR ≤1000 行约束。

**不改**：上游 vllm 仓库、`kv_p2p/`（PD 分离，PR2）、backend 实现（`backend/*.py`，计时在 worker 层完成，不动 backend）。

---

## 六、关键设计决策与风险

1. **Gauge 与 interval 聚合框架的适配**：stats 框架按日志周期聚合（observe→aggregate→reduce），Gauge 记"最新快照"而非累计值；Prometheus 侧 `set()`。若 interval 内 scheduler 多次快照，取 last（`aggregate` 语义明确即可）。
2. **序列化约束**：`SchedulerStats.kv_connector_stats` 是 `dict[str, Any]`，stats 对象跨进程只传 `data`——所有字段必须是 primitive/list/dict。
3. **layerwise 完成点判定**是本 PR 技术难点：请求级"全部层加载完成"的判定要结合 `wait_for_layer_load` / `_cache_write_events`，动手前先读透 `process_layer_data` 主循环。
4. **热路径零开销原则**：无加载请求时 `get_stats()` 返回空 stats，`is_empty()` 短路；不在 scheduler 调度循环里加额外同步操作。
5. **多 connector 兼容**：若用户配 MultiConnector 包装 AscendStore，钩子签名必须与 base.py 一致（mooncake store 已验证该模式可行）。

---

## 七、提交规范

- Commit：`git commit -s`，格式 `feat(kv_pool): add KV cache load duration and delayed release metrics`
- PR 标题：`[Feat][KVPool] Add pool KV cache load duration and delayed release metrics`
- PR 描述含：指标语义、测试命令与结果、冒烟记录
