# PR1 技术详解：池化 KV Cache 指标埋点（AscendStore 形态）

> 对应 PR：[vllm-project/vllm-ascend#14912](https://github.com/vllm-project/vllm-ascend/pull/14912)
> 分支：`kv_metrics_observability`

---

## 1. 背景与目标

### 1.1 问题

vllm-ascend 的池化 KV Cache（AscendStore）把 KV Cache 从 NPU HBM 溢出到 CPU DRAM / SSD 扩容。但在此之前，池化行为完全是个黑盒：

- 一个请求从池里加载 KV Cache 到底花了多久？没人知道。
- 加载走了哪条路径（同步 / 异步 / 分层）？没有区分。
- 加载失败了多少 key？只有零散的 error 日志。
- 当前有多少请求的 KV 块被"延迟释放窗口"扣着不放？无法观测。

没有这些指标，池化性能调优（比如 Layerwise 传输与计算并发掩盖搬运耗时）就只能靠日志猜。

### 1.2 目标

把 AscendStoreConnector 接入上游 vllm 的 KV Connector 指标框架，让池化行为通过标准的 `/metrics` 端点（Prometheus）和周期性日志可观测：

| 指标名 | 类型 | 标签 | 含义 |
|---|---|---|---|
| `vllm:kv_pool_load_duration_seconds` | Histogram | `path` | 单请求从池加载 KV Cache 的墙钟耗时 |
| `vllm:kv_pool_load_keys_total` | Counter | `path` | 加载的 key（块）总数 |
| `vllm:kv_pool_load_failed_keys_total` | Counter | `path` | 加载失败的 key 总数 |
| `vllm:kv_pool_delayed_release_requests` | Gauge | — | 当前处于延迟释放窗口的请求数 |

其中 `path` 标签取值 `sync` / `async` / `layerwise`，对应三条加载路径。

Histogram 桶定义（秒）：`[1ms, 5ms, 10ms, 25ms, 50ms, 75ms, 100ms, 200ms, 300ms, 500ms, 750ms, 1s, 2.5s, 5s]`，覆盖从快命中到慢加载（SSD 池）的完整量程。

---

## 2. 上游指标框架机制（地基）

在讲实现之前必须先讲清楚上游 vllm（commit `ba07e4a48`）提供的指标框架，因为我们的全部工作就是"按框架的规矩接入"。

### 2.1 四个核心类

框架位于 `vllm/distributed/kv_transfer/kv_connector/v1/metrics.py`：

```
KVConnectorStats          ← 数据容器（可序列化），子类实现 reset/aggregate/reduce/is_empty
KVConnectorLogging        ← 日志消费方：周期性聚合 + 打日志
KVConnectorPromMetrics    ← Prometheus 消费方：子类注册指标 + 实现 observe()
KVConnectorProm           ← 入口：通过 factory 拿到 connector 类，调 build_prom_metrics()
```

### 2.2 Connector 需要实现的三个钩子

定义在 `KVConnectorBase_V1`（`base.py`）：

```python
def get_kv_connector_stats(self) -> KVConnectorStats | None          # 实例方法：取走自上次调用以来的统计
@classmethod
def build_kv_connector_stats(cls, data) -> KVConnectorStats | None   # 类方法：从 dict 重建 Stats 对象
@classmethod
def build_prom_metrics(cls, vllm_config, metric_types, labelnames,   # 类方法：构造 Prometheus 指标集
                      per_engine_labelvalues) -> KVConnectorPromMetrics
```

为什么 `build_kv_connector_stats` 是类方法？因为 Stats 对象要从 worker 进程跨进程传到 API server 进程，传输时只带原始 dict（msgpack 可序列化），消费方只有 connector 类名（从 kv_transfer_config 解析），必须用类方法从 dict 重建对象。

### 2.3 数据流（从埋点到 /metrics）

```
┌─ EngineCore 进程（scheduler + worker）──────────────────────────┐
│                                                                  │
│  pool_worker (计时)          pool_scheduler (延迟释放快照)        │
│      │ get_stats()               │ get_stats()                   │
│      ▼                            ▼                               │
│  kv_connector_model_runner    scheduler 调度循环                  │
│  每个 step 调 get_kv_connector_stats()                            │
│      │                            │                               │
│      └──────────┬─────────────────┘                               │
│                 ▼ aggregate()                                      │
│        SchedulerStats.kv_connector_stats (纯 dict)                 │
└─────────────────│─────────────────────────────────────────────────┘
                  │ EngineCoreOutput → msgpack
                  ▼
┌─ 前台进程（API server）──────────────────────────────────────────┐
│  PrometheusStatLogger / StatLoggerManager                         │
│      │                                                            │
│      ├─► KVConnectorProm.observe(data)                            │
│      │       └─► build_prom_metrics() 造的 AscendStorePromMetrics │
│      │            .observe(data, engine_idx) → 4 个指标更新        │
│      │                                                            │
│      └─► KVConnectorLogging.observe(data)（聚合，日志用）          │
│           └─► build_kv_connector_stats(data) 重建 Stats           │
│                .aggregate() 累积 → .reduce() → "KV Transfer        │
│                 metrics: load_avg_ms=..., ..."                    │
└───────────────────────────────────────────────────────────────────┘
```

关键点：

1. **采集在 EngineCore 进程，消费在 API server 进程**，中间靠 `SchedulerStats.kv_connector_stats: dict` + msgpack 传输。这就是为什么 Stats 的 `data` 只能放 primitives（dict/list/int/float/str）。
2. **worker stats 和 scheduler stats 在 scheduler 调度循环里聚合**（`scheduler.py:2097-2111`）：worker 的 stats 随 `kv_connector_output` 先到，然后调 `self.connector.get_kv_connector_stats()` 拿 scheduler 侧的，两者 `aggregate()` 后放进 `SchedulerStats`。
3. **`get_kv_connector_stats()` 是"取走即清空"语义**（consume），每次调用返回自上次以来的增量。

---

## 3. 设计

### 3.1 两个指标的采集位置

**指标 A：加载耗时（Histogram + 2 个 Counter）**

在 worker 侧采集。AscendStore 有三条加载路径，需要分别计时：

| 路径 | 执行位置 | 特点 |
|---|---|---|
| `sync` | `pool_worker.start_load_kv()` 直接调 `m_store.get()` | 调度循环内同步阻塞 |
| `async` | `KVCacheStoreRecvingThread`（kv_transfer.py）后台线程 | 请求入队，线程异步加载，完成后 set event |
| `layerwise` | `process_layer_data()` 按层提交任务，逐层 wait | 传输与计算并发，最后一层完成时全部结束 |

三条路径统一用 `worker._record_load_started(req_id)` / `worker._record_load_finished(req_id, num_keys, num_failed_keys, path)` 这对接口埋点：
- `started`：记 `time.perf_counter()` 到 `_load_start_times[req_id]`。
- `finished`：pop 出开始时间，耗时 = now - start，连同 key 数、失败数、路径标签写入 `_kv_stats`。

为什么用 `req_id` 做 key 而不是传时间戳对象？因为 async 路径的 started 和 finished 分属两个执行流（worker 调度线程 start，recv 线程 finish），需要一个共享的、按请求寻址的暂存字典。

**指标 B：延迟释放请求数（Gauge）**

在 scheduler 侧采集。`pool_scheduler` 维护 `_delayed_free_req_ids: set[str]`——这些请求已算完，但 KV 块被扣住暂不归还（等待窗口确认，避免池端误判"还有人在用"而把块转走/删掉）。这个集合的大小就是"当前被延迟释放扣住的请求数"。

采集点放在 `build_connector_meta()` 末尾：这个方法每个调度步必然被调用，且此时 `_delayed_free_req_ids` 是本步的稳定状态。Gauge 是"最新值覆盖"语义，每步 set 一次即可。

### 3.2 数据结构设计

`AscendStoreKVConnectorStats.data` 的布局（全 primitives，可 msgpack）：

```python
{
    "load": [                                    # list，append-only，aggregate 时 extend 合并
        {
            "duration_seconds": 0.042,           # float
            "num_keys": 12,                      # int
            "num_failed_keys": 1,                # int
            "path": "async",                     # str: sync|async|layerwise
        },
        ...
    ],
    "delayed_release": {"num_requests": 3},      # dict，每步覆盖，聚合时"最新的赢"
}
```

设计要点：

- **`load` 是 list**：Prometheus Histogram 需要 observe 每一个原始样本（不能只 observe 均值，否则桶分布失真），所以必须逐条保存、逐条回放。
- **`delayed_release` 是单值 dict**：Gauge 只关心最新快照。多步聚合时"最后写入的赢"（谁后到谁覆盖），语义天然正确。
- **`aggregate` 的两条规则**：list 类型 extend 合并；非 list（gauge 快照）直接覆盖。一个方法同时处理两种语义。

### 3.3 reduce() 的作用（日志路径）

`reduce()` 把一个日志周期内累积的样本压成汇总值，供 `KVConnectorLogging.log()` 打印：

```python
{
    "load_count": 10,            # 样本数
    "load_avg_ms": 550.0,        # 平均耗时
    "load_p90_ms": 900.0,        # p90（nearest-rank 法）
    "load_keys": 20,             # key 总数
    "load_failed_keys": 10,      # 失败 key 总数
    "delayed_release_requests": 4,
}
```

日志输出形如：`KV Transfer metrics: load_count=10, load_avg_ms=550.0, load_p90_ms=900.0, ...`

p90 用 nearest-rank 法（排序后取第 `int(0.9 * n)` 个），实现只有 6 行，不引入 numpy 依赖。

---

## 4. 代码实现详解

### 4.1 新文件：`ascend_store/metrics.py`

整个 PR 的核心，两个类。

**`AscendStoreKVConnectorStats`**（数据容器）：

```python
@dataclass
class AscendStoreKVConnectorStats(KVConnectorStats):
    def __post_init__(self):
        if not self.data:
            self.reset()                    # 空参构造 → 全新容器

    def aggregate(self, other):             # list extend / gauge 覆盖
    def reduce(self) -> dict:               # 上面 3.3 的汇总逻辑
    def record_load(self, duration_seconds, num_keys, *, num_failed_keys=0, path="sync")
    def record_delayed_release(self, num_requests)
```

注意 `__post_init__` 的判断：`data=None`（新构造）时 reset；`data={...}`（msgpack 重建）时原样保留——重建路径依赖这一点。

**`AscendStorePromMetrics`**（Prometheus 注册 + 回放）：

```python
class AscendStorePromMetrics(KVConnectorPromMetrics):
    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        super().__init__(...)                          # 基类解析 gauge/counter/histogram 类
        metric_labelnames = labelnames + ["path"]      # 基础标签 [model_name, engine] + path
        self._histogram_load_duration = self._histogram_cls(
            name="vllm:kv_pool_load_duration_seconds", ...,
            buckets=[1e-3, 5e-3, ..., 5.0],
            labelnames=metric_labelnames)
        self._counter_load_keys = self._counter_cls(...)
        self._counter_load_failed_keys = self._counter_cls(...)
        self._gauge_delayed_release = self._gauge_cls(..., labelnames=labelnames)  # 无 path
        self._delayed_release_per_engine = create_metric_per_engine(...)
```

一个容易被忽略的细节：`metric_types` 的 key 是 **prometheus_client 的类型对象**（`{Gauge: g, Counter: c, Histogram: h}`），基类构造函数里做 `metric_types[Gauge]` 这样的查找。测试里要构造同构的 dict 才能 Drill 进去。

`observe()` 是回放器——把传输过来的 dict 逐条写进 Prometheus：

```python
def observe(self, transfer_stats_data, engine_idx=0):
    if not transfer_stats_data:
        return
    for record in transfer_stats_data.get("load", []):
        metrics = self._get_load_metrics(engine_idx, str(record["path"]))
        metrics["duration"].observe(float(record["duration_seconds"]))
        metrics["keys"].inc(int(record["num_keys"]))
        metrics["failed_keys"].inc(int(record["num_failed_keys"]))
    delayed_release = transfer_stats_data.get("delayed_release")
    if delayed_release is not None:
        gauge = self._delayed_release_per_engine.get(engine_idx)
        if gauge is not None:
            gauge.set(int(delayed_release["num_requests"]))
```

`_get_load_metrics` 带一层 `(engine_idx, path)` 的缓存，避免每条记录都 `labels(...()` 触发 prometheus_client 的 label 查找开销。

### 4.2 `pool_worker.py`：计时埋点

初始化（构造函数里）：

```python
self._kv_stats = AscendStoreKVConnectorStats()
self._kv_stats_lock = threading.Lock()      # worker stats 会被多个线程并发写
self._load_start_times: dict[str, float] = {}    # req_id → perf_counter 起点
self._layerwise_load_keys: dict[str, int] = {}   # layerwise 专用：req_id → 块数
```

埋点接口：

```python
def _record_load_started(self, req_id):
    self._load_start_times[req_id] = time.perf_counter()

def _record_load_finished(self, req_id, num_keys, num_failed_keys=0, path="sync"):
    start_time = self._load_start_times.pop(req_id, None)
    if start_time is None or num_keys <= 0:      # 防御：没 start 过 / 空 load 不记录
        return
    with self._kv_stats_lock:
        self._kv_stats.record_load(time.perf_counter() - start_time,
                                    num_keys, num_failed_keys=..., path=...)
```

**sync 路径**（`start_load_kv`）：

```python
self._record_load_started(request.req_id)     # ① 在检查 load_spec 之后、真正加载之前
...
ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)   # ② 阻塞加载
num_failed_keys = sum(1 for r in ret if r != 0) if ret is not None else len(key_list_c)
self._record_load_finished(request.req_id, len(key_list_c),
                            num_failed_keys=num_failed_keys, path="sync")   # ③
```

失败计数规则：`m_store.get` 返回逐 key 的返回码列表，非 0 即失败；返回 None（整体失败）则全部计失败。

**async 路径**：worker 只把请求塞进 `kv_recv_thread.add_request(request)`，加载在后台线程做。所以 started 在 worker 这边（同 sync 的 ①），finished 在 `kv_transfer.py` 的接收线程里（见 4.4）。

**layerwise 路径**（按层加载，传输与计算并发）：

```python
def _record_layerwise_load_started(self):
    # 每层的任务里包含各请求的 block_range，遍历累计每个请求的块数
    start_time = time.perf_counter()
    for layer_tasks in self.layer_load_tasks:
        for task in layer_tasks:
            for block_range in task.block_ranges:
                req_id = block_range.request.req_id
                num_blocks = (block_range.end_block - block_range.start_block) + (
                    1 if block_range.partial_block_index is not None else 0)
                if req_id in self._load_start_times:
                    self._layerwise_load_keys[req_id] += num_blocks
                else:
                    self._load_start_times[req_id] = start_time
                    self._layerwise_load_keys[req_id] = num_blocks

def _record_layerwise_load_finished(self):
    # 最后一层 load 被 wait 完成时调用：本步所有请求同时结束
    if not self._load_start_times:
        return
    end_time = time.perf_counter()
    for req_id, num_keys in self._layerwise_load_keys.items():
        start_time = self._load_start_times.pop(req_id, None)
        if start_time is None or num_keys <= 0:
            continue
        with self._kv_stats_lock:
            self._kv_stats.record_load(end_time - start_time, num_keys, path="layerwise")
    self._layerwise_load_keys.clear()
    self._load_start_times.clear()    # 排掉中途被抢占的残留，防止跨步泄漏
```

layerwise 的语义与 sync/async 不同：耗时度量的是"从第一步任务提交到最后一层加载完成"的端到端墙钟时间（这正是 Layerwise 并发掩盖要优化的量），key 数是该请求所有层传输的块数总和。

**取走接口**：

```python
def get_stats(self) -> AscendStoreKVConnectorStats | None:
    with self._kv_stats_lock:
        if self._kv_stats.is_empty():
            return None
        stats = self._kv_stats
        self._kv_stats = AscendStoreKVConnectorStats()   # 原子交换
        return stats
```

### 4.3 `pool_scheduler.py`：延迟释放快照

```python
# __init__ 里
self._kv_stats = AscendStoreKVConnectorStats()

# build_connector_meta() 的 return meta 之前
self._kv_stats.record_delayed_release(len(self._delayed_free_req_ids))

def get_stats(self) -> AscendStoreKVConnectorStats:
    stats = self._kv_stats
    self._kv_stats = AscendStoreKVConnectorStats()
    return stats
```

与 worker 的差异：scheduler 的 `get_stats` 不判空、不用锁——它只在调度循环（单线程）里被 `get_kv_connector_stats()` 调用，且 `build_connector_meta` 每步都写快照，`data` 里始终有 `delayed_release` 键。

### 4.4 `kv_transfer.py`：异步线程回传

`KVCacheStoreRecvingThread._handle_request()` 是 async 路径的实际加载执行者。加载完成后把结果回传给 worker 的埋点：

```python
def _record_load_finished(self, req_id, num_keys, num_failed_keys=0, path="async"):
    if self.worker is not None:          # 线程持有 worker 引用
        self.worker._record_load_finished(req_id, num_keys,
                                          num_failed_keys=..., path=path)
```

调用点覆盖 `_handle_request` 的所有出口：

- 无 load_spec → `_record_load_finished(req_id, 0)`（num_keys=0，started 会被 pop 掉但不记录，见 4.2 的防御）
- tp_mismatch 分支（TP 数与池端不一致，走专用降级加载）→ 加载完带上真实 num_keys / num_failed_keys
- 空 key_list → 同样记 0
- 正常路径 `m_store.get` 返回后 → 记 `len(key_list_c)` 和失败数

### 4.5 `ascend_store_connector.py`：三个钩子

```python
def get_kv_connector_stats(self):
    if self.connector_scheduler is not None:
        return self.connector_scheduler.get_stats()     # scheduler 侧：延迟释放
    assert self.connector_worker is not None
    return self.connector_worker.get_stats()            # worker 侧：加载耗时

@classmethod
def build_kv_connector_stats(cls, data=None):
    return AscendStoreKVConnectorStats(data=data or {})

@classmethod
def build_prom_metrics(cls, vllm_config, metric_types, labelnames, per_engine_labelvalues):
    return AscendStorePromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)
```

同一个 connector 类在 scheduler 和 worker 两种角色下各实例化一次（框架如此），`get_kv_connector_stats` 按当前持有的组件分流。上游 scheduler.py 的聚合逻辑（2.3 节）会先把 worker stats（随 kv_connector_output 来）和 scheduler stats merge 到一起再发出去。

---

## 5. 关键设计决策与权衡

**为什么复用上游框架，而不是直接在 worker 里注册 Prometheus？**
Prometheus 指标必须在 API server 进程暴露（/metrics 端点在那），而埋点数据在 EngineCore 进程产生。跨进程传输 + 指标注册这套基础设施，上游框架已经全部做好了（SchedulerStats 通道、msgpack 序列化、per-engine 标签管理、日志聚合）。自己另起一套等于重复造轮子还破坏一致性——其他 connector（NIXL / Mooncake / LMCache / HF3FS）全走这条路。

**为什么 `load` 存原始样本列表而不是边采边算分位数？**
两条消费路径需求不同：Prometheus Histogram 需要逐样本 observe（保桶分布），日志需要周期汇总（均值/p90）。存原始列表两边都能满足；且 `data` 是唯一跨进程载体，样本必须在传输前保真。

**为什么 Histogram 桶这样定？**
池化加载的目标场景是"DRAM/SSD 扩容 + Layerwise 掩盖"，理想情况单请求加载应落在几十 ms 内（RP3 验收锚点），异常情况（SSD 冷读、池端争用）可能到秒级。桶在 1ms–5s 之间对数式加密，低段细（区分 1/5/10/25/50ms 的性能差异），高段粗（只标记"严重劣化"）。

**layerwise 的 num_keys 为什么含 partial block？**
`kv_transfer.py` 的 `build_shared` 在准备每层传输 buffer 时，块数就是 `end - start + (1 if partial_block_index is not None else 0)`。指标必须和实际传输量对齐，否则 layerwise 路径的 keys_total 会系统性偏低。这是 review（gemini-code-assist）指出后修复的。

**空 load 为什么不记录？**
`num_keys <= 0` 意味着该请求实际没从池里加载任何东西（比如全部命中本地 prefix cache），记录它会给 duration histogram 注入一堆无意义的近零样本，拉低 avg、扭曲分位数。跳过的同时必须把 start_time pop 掉，否则字典泄漏。

**线程安全边界？**
`_kv_stats` 的写入来自三个执行流（worker 调度线程 sync、recv 线程 async、layerwise 提交流），所以用 `_kv_stats_lock` 保护。`_load_start_times` 没上锁：同一 req_id 的 started 一定 happens-before finished（async 路径是入队→线程处理，layerwise 是提交→最后层完成），单线程视角下无竞态；CPython dict 单操作原子，GIL 下最坏也只是读到旧值，且值语义（时间戳/计数）允许最终一致。

---

## 6. 测试

### 6.1 单元测试（`tests/ut/distributed/ascend_store/test_metrics.py`，22 个用例）

| 组 | 覆盖 |
|---|---|
| `TestAscendStoreKVConnectorStats`（9 个） | 空/重置、record_load 追加、gauge 覆盖、aggregate 合并 list + gauge 取最新、空 aggregate no-op、reduce 的 avg/p90/keys/failed、reduce 空、msgpack 重建 |
| `TestAscendStorePromMetrics`（5 个） | observe 写入正确标签组合、三 path 标签隔离、gauge set、空 data no-op、4 个指标名注册正确 |
| `TestKVPoolWorkerLoadTiming`（6 个） | sync 计时精度（sleep 0.01 断言 ≥0.01）、get_stats 取走即清空、未 start 的请求被忽略、零 keys 被忽略且不泄漏 start_time、layerwise 计时、**partial block 计数（2 full + 1 partial = 3）** |
| `TestKVPoolSchedulerDelayedRelease`（2 个） | build_connector_meta 快照正确、get_stats 重置 |

测试基础设施要点：

- **Fake Prometheus 原语**：`_FakeMetric` / `_FakeMetricChild` 鸭子类型模拟 `labels()` / `observe()` / `inc()` / `set()`，可以断言"哪个标签组合被 observe 了什么值"——真实 prometheus_client 做不到这么直接的断言。
- **`_mock_deps.py` 扩充**：本地/CI 无 NPU 环境，`vllm` 的 metrics 框架模块、numpy、regex、zmq、msgpack 全部 mock。其中 metrics 框架的 mock 是"半真实"的——`_MockKVConnectorStats` 用真 dataclass 实现，保证子类化关系与真框架一致，UT 才能测到真实继承链。

结果：`pytest tests/ut/distributed/ascend_store/` → **311 passed**（含本 PR 新增 22 个 + 修复的 2 个存量用例）。

### 6.2 集成冒烟（165 服务器，Ascend 910 容器）

真实 vllm 指标框架下 22/22 项检查通过，`/metrics` 实际输出：

```
vllm:kv_pool_load_duration_seconds_bucket{...,path="sync",le="0.05"} 1.0
vllm:kv_pool_load_duration_seconds_sum{...,path="layerwise"} 0.12
vllm:kv_pool_load_keys_total{...,path="sync"} 10.0
vllm:kv_pool_load_failed_keys_total{...,path="layerwise"} 1.0
vllm:kv_pool_delayed_release_requests{...} 5.0
```

### 6.3 Review 修复记录

gemini-code-assist 提出两条 high 意见，均确认成立并修复（提交 `cf1296559`）：

1. **layerwise key 数漏算 partial block** → `num_blocks` 补上 `partial_block_index is not None` 时 +1，与 `build_shared` 对齐；新增 UT `test_layerwise_partial_block_counted`。
2. **layerwise finish 缺 num_keys<=0 防御** → 补 `if start_time is None or num_keys <= 0: continue`，与 sync 路径的 `_record_load_finished` 行为一致。

---

## 7. 文件清单

| 文件 | 状态 | 内容 |
|---|---|---|
| `vllm_ascend/.../ascend_store/metrics.py` | 新增 | Stats 数据容器 + PromMetrics 注册/回放 |
| `vllm_ascend/.../ascend_store/pool_worker.py` | 修改 | 三路径计时埋点、get_stats、layerwise 记账 |
| `vllm_ascend/.../ascend_store/kv_transfer.py` | 修改 | 异步线程加载结果回传 |
| `vllm_ascend/.../ascend_store/pool_scheduler.py` | 修改 | 延迟释放快照 + get_stats |
| `vllm_ascend/.../ascend_store/ascend_store_connector.py` | 修改 | 三个框架钩子 |
| `tests/.../test_metrics.py` | 新增 | 22 个单元测试 |
| `tests/.../_mock_deps.py` | 修改 | 指标框架及三方依赖 mock |
| `tests/.../test_pool_worker.py` | 修改 | 修复 `__new__` 构造缺新属性 |
| `tests/.../test_kv_transfer.py` | 修改 | 修复 tp_mismatch mock 返回值 |

---

## 8. 总结

这个 PR 做的事情一句话概括：**把 AscendStore 的池化行为翻译成上游指标框架的语言**。

- **设计上**，识别出两类本质不同的遥测——加载耗时（逐样本、多路径标签、Histogram 语义）和延迟释放窗口（最新快照、Gauge 语义），用同一个 Stats 容器的两种 data 布局承载，`aggregate` 用"list 合并 / 标量覆盖"一条规则同时服务两者。
- **实现上**，改动最小化：不碰任何加载逻辑本身，只在三条路径的首尾插计时（`perf_counter` + req_id 寻址），scheduler 侧在既有的 `build_connector_meta` 末尾加一行快照。所有数据通过框架既有通道（SchedulerStats → msgpack → API server）流动，新增的只有两端的"翻译"——采集端的 Stats 类和消费端的 PromMetrics 类。
- **质量上**，22 个 UT 覆盖了聚合/回放/埋点/边界（空样本、未 start、零 keys、partial block），165 上真实框架集成验证 22/22，review 意见当日闭环。

后续可观测性建设的落点：这 4 个指标是"池化行为"的第一层。第二层可以加池端视角（store 侧命中/容量/驱逐），第三层做 Grafana 看板与告警规则。指标框架的接入模式（三个钩子 + 两种 data 布局）可以直接复用。
