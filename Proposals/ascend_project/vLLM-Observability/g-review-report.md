# PR 14912（kv_metrics_observability 分支）代码审查报告

> 审查对象：vllm-ascend PR #14912 `[Feature] Add KV pool load-duration and delayed-release metrics`
> 分支：`kv_metrics_observability`（vllm-ascend 本地工作副本）
> 审查基线：`fa668c649`（main）→ `cf1296559`（分支 tip，2 个提交，9 个文件，+902/-3 行）
> 需求依据：[AR20260820031213_vLLM监控平台对接需求分析.md](./AR20260820031213_vLLM监控平台对接需求分析.md)（主），关联 [SR20260820223202_Layerwise池化性能优化需求分析.md](./SR20260820223202_Layerwise池化性能优化需求分析.md)
> 审查时间：2026-08-25
> 审查结论（先行）：**作为 AR 需求验收标准 2 的交付基本成立；验收标准 3 仅覆盖池化、未覆盖 PD 分离；存在 1 个 layerwise 路径的指标语义缺陷和 1 个部署硬阻断（vLLM 版本耦合）**。详见第 6、7 章。
> 2026-08-26 增补：已在 51 服务器完成实测验证（vllm v0.27.1 + 分支 cf1296559 + 官方修复 8c28898dd），**sync / layerwise / async 三条加载路径全部实测贯通**，四指标实际上报，7.4 三个待验证假设全部闭合（A1 证实跨 rank 聚合放大），并新发现 1 个 worker 角色阻塞缺陷 **M5**（原审查遗漏，见第 5、8 章）；layerwise 的 M1 耗时污染、M4 失败键恒 0 均获实测强化（见 8.6）。

---

## 一、审查范围与方法

- 通读分支全量 diff（`git diff fa668c649..cf1296559`），精读 4 个源文件改动与 3 个测试文件
- 以 AR20260820031213 的验收标准逐条对照代码证据（文件/行号）
- 交叉验证项目记忆中的历史结论（prefix caching 本地命中、layerwise 模式行为等）对指标语义的影响
- 未能在本地验证的部分（上游 vLLM v0.27.1 metrics 框架的真实调用契约）单独列为"待验证假设"

### 改动文件清单

| 文件 | 改动 | 角色 |
|---|---|---|
| `vllm_ascend/.../ascend_store/metrics.py` | 新增 210 行 | 指标数据模型 + Prometheus 定义 |
| `vllm_ascend/.../ascend_store/pool_worker.py` | +111 | worker 侧加载耗时埋点（sync/async/layerwise） |
| `vllm_ascend/.../ascend_store/kv_transfer.py` | +23/-3 | async 接收线程的计时结束点 + tp_mismatch 返回值改造 |
| `vllm_ascend/.../ascend_store/pool_scheduler.py` | +14 | scheduler 侧延迟释放快照 |
| `vllm_ascend/.../ascend_store/ascend_store_connector.py` | +39 | 挂接上游 metrics 框架的三个入口 |
| `tests/ut/distributed/ascend_store/_mock_deps.py` | 新增 77 行 | UT 依赖 mock 基建 |
| `tests/ut/distributed/ascend_store/test_metrics.py` | 新增 428 行 | 21 个单测 |
| `tests/ut/distributed/ascend_store/test_kv_transfer.py` | +1 | 修复存量 UT（tp_mismatch 返回值） |
| `tests/ut/distributed/ascend_store/test_pool_worker.py` | +2 | 修复存量 UT（`__new__` 构造缺新属性） |

---

## 二、代码结构总览

### 2.1 数据流架构

```
┌─ Worker 进程（每 TP rank）────────────────────────────┐
│ pool_worker.KVPoolWorker                              │
│   _record_load_started(req_id)      ← start_load_kv   │
│   _record_load_finished(req, keys, path)              │
│     ├─ sync: 主线程 key 构建 + m_store.get 之后       │
│     ├─ async: kv_transfer 接收线程 m_store.get 之后   │
│     └─ layerwise: 最后一层 wait 返回之后              │
│   get_stats() → 交换并重置 AscendStoreKVConnectorStats│
└──────────────┬────────────────────────────────────────┘
               │ stats.data（全原始类型，msgpack 序列化）
               ▼ worker→scheduler→engine core→API server
┌─ Scheduler 进程 ──────────────────────────────────────┐
│ pool_scheduler.KVPoolScheduler                        │
│   build_connector_meta() 每步快照                      │
│   len(_delayed_free_req_ids) → record_delayed_release │
│   get_stats() → 交换并重置                             │
└──────────────┬────────────────────────────────────────┘
               │ aggregate()：load 列表拼接，gauge 后者胜
               ▼
┌─ API Server（Prometheus 暴露）────────────────────────┐
│ AscendStorePromMetrics.observe(data, engine_idx)      │
│   → 4 个 Prometheus 指标（/metrics 端点）             │
└───────────────────────────────────────────────────────┘
```

### 2.2 指标定义（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py#L118-L210``）

| 指标名 | 类型 | Label | 语义 |
|---|---|---|---|
| `vllm:kv_pool_load_duration_seconds` | Histogram | path（sync/async/layerwise）+ 引擎 label | 每请求 KV 池加载墙钟耗时；桶 1ms–5s 共 14 档 |
| `vllm:kv_pool_load_keys_total` | Counter | path | 每请求加载的池 key 数（累计） |
| `vllm:kv_pool_load_failed_keys_total` | Counter | path | 每请求加载失败的 key 数（累计） |
| `vllm:kv_pool_delayed_release_requests` | Gauge | 引擎 label | 当前处于延迟释放窗口的请求数 |

### 2.3 数据模型（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py#L44-L115``）

- `AscendStoreKVConnectorStats.data` 只含原始类型（dict/list/str/int/float），设计上可过 msgpack 跨进程——有单测 `test_rebuild_from_plain_data` 验证重建
- `aggregate()`：`load` 列表 extend 拼接；`delayed_release` 为 gauge 快照，**后者覆盖前者**（latest-wins）
- `reduce()`：产出 `load_count / load_avg_ms / load_p90_ms / load_keys / load_failed_keys / delayed_release_requests`，P90 用自实现 nearest-rank 法（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py#L34-L41``，边界计算正确：n=10 时取 index 8，n=100 时取 index 89）

### 2.4 框架挂接（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L300-L331``）

实现上游 `KVConnectorBase_V1` 的三个扩展点：
- `get_kv_connector_stats()`：scheduler 侧优先返回延迟释放快照，否则返回 worker 侧加载记录
- `build_kv_connector_stats(data=)`：反序列化入口
- `build_prom_metrics(...)`：Prometheus 指标注册入口

---

## 三、埋点实现细节（三条加载路径）

### 3.1 sync 路径（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L853-L960``）

- **起表**：`start_load_kv` 循环内、key 构建之前（L862）
- **止表**：`m_store.get()` 返回后（L947-955）
- 失败计数：`ret` 非 None 时 `sum(1 for r in ret if r != 0)`；`ret is None` 时全部计为失败
- 空 key 请求（`not key_list`）：调用 `_record_load_finished(req_id, 0)` 仅清掉 start time，不产生样本（`num_keys <= 0` 守卫）

### 3.2 async 路径（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L893-L1033``）

- **起表**：同 sync，worker 主线程 `start_load_kv` 里 `add_request` 之前
- **止表**：接收线程 `_handle_request` 中 `m_store.get` 之后，经 `_record_load_finished` 包装（默认 `path="async"`，透传给 worker）
- tp_mismatch 分支：`_load_kv_tp_mismatch` 返回值从 `None` 改为 `tuple[int, int]`（keys, failed_keys），兼容修复见 3.6
- 无 load_spec / 空 key：`_record_load_finished(req_id, 0)` 清 start time，不采样

### 3.3 layerwise 路径（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1010-L1051``）

- **起表**：`start_load_kv` 中 `process_layer_data()` 之后调用 `_record_layerwise_load_started()`——遍历 `layer_load_tasks` 全部层，按请求累计 block 数（含 partial block +1，对应 review 修复提交 `cf1296559`）
- **止表**：`wait_for_layer_load()` 检测到 `current_layer == num_layers - 1` 时（should_wait 真假两分支均覆盖），统一用同一 `end_time` 记录所有请求
- num_keys 语义：**跨层 block 传输数**（≈ blocks × layers），注释明确说明
- 结束后清空两个簿记 dict；被抢占请求在 `get_finished()` 中提前 pop（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2152-L2154``）

### 3.4 延迟释放 gauge（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L875-L889``）

- `build_connector_meta()` 末尾**每个调度步**快照 `len(self._delayed_free_req_ids)`
- `get_stats()` 交换并重置，最新快照在聚合时覆盖旧值
- 窗口清空时记录 0，gauge 能正确回落

### 3.5 线程安全

- `_load_start_times`：worker 主线程写、接收线程 pop，依赖 CPython GIL 下单条 dict 操作的原子性（代码注释已说明）；`_layerwise_load_keys` 仅主线程访问
- `_kv_stats`：`record_load` 与 `get_stats` 交换均在 `_kv_stats_lock` 内
- scheduler 侧 `_kv_stats`：调度器单线程访问，无需锁

### 3.6 review 修复提交（cf1296559）验证

1. **partial_block_index +1**：与 `kv_transfer.py` 中 `build_shared` 的 key 数口径对齐——`test_layerwise_partial_block_counted` 验证 2 全块 + 1 partial = 3 ✓
2. **跳过 zero-key 记录**：`_record_layerwise_load_finished` 中 `num_keys <= 0` continue，防止空 load 事件拉低直方图 ✓

两处修复均有针对性单测，修复质量良好。

---

## 四、需求符合性逐条核对（AR20260820031213）

| 需求条目 | 代码证据 | 判定 |
|---|---|---|
| **验收 1**：tokenizer 时间统计（已声明由其他需求跟踪） | PR 无相关改动 | ✅ 符合边界声明，未越界 |
| **验收 2**：池化 KV Cache 加载耗时指标（请求粒度） | per-request 记录（`record_load` 每请求一条）；三条路径全覆盖；Histogram（桶 1ms–5s）+ reduce 给 avg/P90/count | ✅ **基本满足**。扣分项：layerwise 路径语义缺陷（见 M1）、sync/async 实测范围与 docstring 不符（m1/m2） |
| **验收 3**：KV Cache 延迟释放状态的请求数量（**涉及 PD 分离和池化**） | Gauge 快照 `len(_delayed_free_req_ids)`，仅 AscendStore 池化 scheduler | ⚠️ **部分满足**：池化 ✅；PD 分离场景（main2main 等 PD 传输路径）**未实现**（M2） |
| 需求内容 1：性能/特性满足度/异常状态可度量 | 性能 ✅（加载耗时）；异常状态部分覆盖：`load_failed_keys_total` ✅，但**请求异常终止**、**cache block 分配失败**无指标 | ⚠️ 部分满足 |
| 需求内容 2：完成监控平台对接 vLLM 指标可视化 | Prometheus 标准暴露（经上游 KV connector metrics 框架）；极光大盘配置不在本 PR | ⚠️ 暴露层 ✅（端到端未验证）；大盘层待平台侧交付 |

### 对 SR20260820223202（第二个需求）的支撑

| SR 需求项 | 本 PR 支撑情况 |
|---|---|
| 极光大盘：**not overlapped** KV 池化读取时间 metric | ❌ **未交付**。layerwise 路径记录的是总 span（含被计算掩盖的部分，且被计算耗时污染，见 M1），无法区分"总加载时间 vs 未掩盖时间" |
| 极光大盘：KVC 延迟释放 req 排队 metric | ✅ 同一 gauge 复用 |
| 大盘本身 | 不在本 PR |

---

## 五、发现清单（按严重程度）

### Major

**M1：layerwise 路径的耗时测量被计算耗时污染，语义不成立**（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1741-L1763``）

结束时间取自 `wait_for_layer_load()` 返回时刻。该函数是前向计算推进到某层、需要该层 KV 时的**同步点**：
- 若加载慢于计算（未掩盖）：wait 阻塞到加载完成，测量值 ≈ 真实加载时间 ✅
- 若加载快于计算（完全掩盖）：事件早已 set，wait 立即返回，结束时间 = **计算到达最后一层的时刻**，测量值 ≈ 计算耗时，与加载无关 ❌

即 layerwise 直方图实际测量 `max(加载 span, 计算推进到最后一层的 span)`，是加载耗时的**上界**。后果：
1. 对 AR 验收 2：layerwise 路径的耗时数据在"掩盖效果好"时系统性偏高，不能用于评估加载本身
2. 对 SR 需求：**无法从该指标得到 not overlapped 时间**——正确做法是记录传输线程 set 事件的时间戳（加载真实完成时刻）而非 wait 返回时刻，两者之差才是"计算等待加载"（not overlapped）的时间
3. 建议修复：在 `layer_load_finished_events` 的 set 处（传输线程侧）记录每层完成时刻；或至少把"事件 set 时刻"存入 event 附加数据，`_record_layerwise_load_finished` 用它作为 end_time

**M2：验收标准 3 的 PD 分离场景缺失**

需求原文"（涉及PD分离和池化）"。`_delayed_free_req_ids` 是 AscendStore 池化调度器的概念，PD 分离（prefill/decode 分离实例间 KV 传输）走不同 connector/路径，本 PR 完全未埋点。若验收时被追问 PD 场景，当前无数据。需要明确：要么补 PD 路径埋点，要么与需求方确认 PD 场景由后续需求覆盖。

**M3：vLLM 版本硬耦合，51 容器当前环境直接阻断**（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py#L24-L31``）

```python
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (...)
from vllm.v1.metrics.utils import create_metric_per_engine
```

顶层无降级 import。已实测确认：
- 该分支声明配对 vLLM **v0.27.1**（`.github/vllm-release-tag.commit` = v0.27.1，`vllm-main-verified.commit` = `ba07e4a48`）
- 51 容器实际安装 `vllm 0.26.1rc1.dev517+g58d3918e3`，**整个 vllm 包 grep 不到 `kv_connector_stats`**——上游 KV connector metrics 框架在容器版本中不存在
- 后果：在该分支 + 当前容器环境，AscendStore connector import 即 `ModuleNotFoundError`，服务起不来

**M4：layerwise 路径无失败键统计**

`_record_layerwise_load_finished` 不传 `num_failed_keys`（恒 0），`kv_pool_load_failed_keys_total{path=layerwise}` 永远为 0。layerwise 加载失败观测缺口（invalid_block_ids 机制存在但未接入计数）。

**M5：`get_kv_connector_stats()` 在 worker 角色下必然 AttributeError，服务无法处理任何请求（2026-08-26 实测发现，静态审查遗漏）**（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L306``）

PR 原始代码（分支 tip `cf1296559`）：

```python
if self.connector_scheduler is not None:
    return self.connector_scheduler.get_stats()
```

而 `__init__` 中（L114-125）`connector_scheduler` **仅在 `role == KVConnectorRole.SCHEDULER` 时赋值**，worker 角色只赋 `connector_worker`。上游 vllm v0.27.1 在每个 TP rank 的 model runner 的 `execute_model` 收尾处调用该方法（``vllm/v1/worker/kv_connector_model_runner_mixin.py#L107``），因此 worker 角色实例**首个前向批次即崩溃**：

```
AttributeError: 'AscendStoreConnector' object has no attribute 'connector_scheduler'
```

实测序列：升级 vllm v0.27.1 后服务可启动（健康检查通过），发送第一个请求时 worker 崩溃——即 **PR 原始代码在任何实际部署形态下均不可用**（AscendStore 必然存在 worker 角色）。本报告第六章测试缺口 4（"无 get_kv_connector_stats 的角色分派测试"）恰好掩盖了该缺陷：UT 直接以指定角色构造 connector 后调用，未覆盖"worker 角色被框架调用"的真实路径。

临时修复（验证环境已应用，等价语义）：`if getattr(self, "connector_scheduler", None) is not None:`。上游修复建议改为与同文件其他 worker 方法一致的模式（如 L209 `getattr(self, "connector_worker", None)`），或在 `__init__` 两个分支各自显式置空另一侧属性。

### Minor

**m1：async 路径 duration 含排队等待**。起表在 `start_load_kv`（请求交给接收线程队列前），止表在接收线程 `m_store.get` 完成后——包含接收线程队列中排在前面请求的处理时间。作为"请求粒度端到端加载耗时"语义可接受，但与 metrics.py 模块 docstring"wall-clock duration of loading ... (m_store.get)"不符，文档需修正，看板解读需知晓。

**m2：sync 路径 duration 含 key 准备阶段**（`load_mask` / `process_token_key_strings_with_block_ids` / `prepare_value` / 循环移位），非纯 `m_store.get`。同上，文档与实测范围不符。

**m3：num_keys 语义跨路径不一致**。sync/async = 本 rank 的 key chunk 数；layerwise = 跨层 block 传输数（blocks × layers）。Counter 文档"pool keys loaded per request"对 layerwise 不准确，跨 path 聚合对比时口径不可比。

**m4：TP 多 rank 聚合行为未验证**。每个 TP rank 的 worker 各自记录 stats（每请求每 rank 一条）。上游框架若聚合所有 rank：`load_count` 放大 TP 倍；若只收 rank0：keys 只计 rank0 子集。本地无 v0.27.1 源码无法确认，**列为待验证假设 A1**。看板使用前必须确认，否则吞吐类聚合（如 keys/s）口径错误。

**m5：gauge 为最新快照语义**。轮询间隔内的瞬时峰值不可见（后值覆盖前值）。对"排队情况"观测可接受，但解读时需知是"最近一次调度步的窗口大小"而非窗口峰值。

**m6：两侧 get_stats 返回契约不一致**。worker 空时返回 `None`，scheduler 空时返回空 stats 对象。依赖上游框架对两者的处理一致（`is_empty()` 检查），未验证（待验证假设 A2）。

**m7：`observe()` 内用 `assert isinstance(record, dict)`**（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py#L201``），`python -O` 下被剥离。生产代码建议显式校验或 try/except。

### Note

- **n1**：21 个 UT 与 PR 描述数量一致（8 stats + 5 prom + 6 worker 计时 + 2 scheduler 快照），全部真实断言业务行为，非 mock 自嗨；`_mock_deps.py` 质量高（同时支持有/无真实 vllm 环境，注释详尽说明泄漏风险）。风险：mock 的 `KVConnectorPromMetrics` 按 dict 插入顺序解析 gauge/counter/histogram 类（``tests/ut/distributed/ascend_store/_mock_deps.py#L164-L174``），依赖上游构造顺序稳定——契约若变，测试静默通过错误映射（待验证假设 A3）
- **n2**：`reduce()` 只给 avg/P90，Prometheus histogram 提供全分布，互补合理
- **n3**：极端场景：layerwise 请求既未到最后一层也未被 preempt（如异常中断步），其 start time 残留至下一步 `_record_layerwise_load_started` 时被误认为"已开始"，duration 跨两步。概率低、影响为个别样本偏大
- **n4**：直方图桶上限 5s + Inf；128K 长序列慢后端可能溢出到 Inf 桶，可接受
- **n5**："请求粒度"体现在**采样粒度**（每请求一个观测值），非**可查询粒度**（histogram 聚合后无法定位单请求）。符合验收语义，但排障时仍需日志配合
- **n6**：预清理设计完善：preempted pop（``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2152-L2154``）+ layerwise 收尾 drain + 空请求不采样，簿记无泄漏路径（除 n3 极端场景）
- **n7**：指标命名 `vllm:kv_pool_*` 符合 vllm 前缀约定，未与现有 vllm 指标冲突（本地代码库范围内确认）

---

## 六、测试覆盖评估

| 测试组 | 数量 | 覆盖点 | 评价 |
|---|---|---|---|
| TestAscendStoreKVConnectorStats | 8 | 空/重置、记录追加、gauge 覆盖、聚合（列表拼接+latest-wins）、reduce 数学、msgpack 重建 | 数据模型行为全覆盖，含 P90 边界注释 |
| TestAscendStorePromMetrics | 5 | path label 维度、gauge set、空数据 no-op、指标命名 | 暴露层逻辑覆盖 |
| TestKVPoolWorkerLoadTiming | 6 | sync 计时、get_stats 重置、幽灵请求、零 key、layerwise 计时+簿记清理、partial block 计数 | 计时核心逻辑覆盖 |
| TestKVPoolSchedulerDelayedRelease | 2 | 快照记录、重置 | scheduler 侧基础覆盖 |

**缺口**（未测）：
1. **无 async 路径的端到端计时测试**（`KVCacheStoreRecvingThread._handle_request` 各分支的 `_record_load_finished` 调用仅有 tp_mismatch 分支的存量测试间接触达）
2. **无 preempted 清理测试**（L2152-2154 未覆盖）
3. **无多 rank / 聚合语义测试**（m4 相关）
4. **无 `ascend_store_connector.get_kv_connector_stats` 的角色分派测试**（scheduler 优先 / worker 兜底分支）
5. reduce() 的 P90 只有 n=10 一档边界，无 n=1、n=2 等退化输入

---

## 七、结论与建议

### 7.1 交付判定

| 维度 | 判定 |
|---|---|
| 作为 AR 验收 2（池化加载耗时，请求粒度）的交付 | **基本成立**——但 layerwise 路径数据因 M1 不可用于"加载耗时"结论，仅 sync/async 路径数据可信 |
| 作为 AR 验收 3（延迟释放请求数）的交付 | **部分成立**——池化 ✅，PD 分离 ❌（M2），验收前需澄清范围 |
| 作为 SR 需求 not overlapped 指标的交付 | **不成立**（M1 语义 + 缺专用指标） |
| 代码工程质量 | **良好**——数据模型干净、线程安全有考虑、UT 真实、review 修复到位；主要问题是 layerwise 计时点和文档语义 |

### 7.2 建议（按优先级）

1. **（阻塞实测）解决 M3**：51 容器 vllm 升级到 v0.27.1（或 main ≥ `ba07e4a48`）并 editable 重装，再起 kv_metrics_observability 分支验证。注意项目历史教训：vllm-ascend 与 vllm 版本强耦合曾引发 FusedMoERouter 类 API 不兼容——升级后先跑现有 DSV4/mooncake 冒烟再上指标验证
2. **（建议修 PR）修 M1**：layerwise 结束时刻改为传输线程 set 事件时刻（真实加载完成点）；同时可顺带产出 not overlapped 时间 = wait 返回时刻 − 事件 set 时刻，一并满足 SR 需求
3. **（建议修 PR）文档修正 m1/m2/m3**：metrics.py docstring 与三个 path 的真实测量范围、num_keys 口径对齐
4. **（验收前澄清）M2**：与需求方确认 PD 分离场景是否在本期验收范围，若在需补埋点
5. **（看板使用前）验证 A1（m4）**：TP>1 部署下确认 `load_count` 是否被放大，必要时在 record 侧加 rank 维度或在框架侧只收 rank0
6. **（低优）M4/m7/n3**：layerwise 失败键接入、assert 改显式校验、跨步残留防御

### 7.3 51 环境实测前置动作清单

1. 容器内 `/home/lizhongyang/code/vllm` 切到 `v0.27.1` tag（或 main ≥ `ba07e4a48`），editable 重装（复用 `reinstall_editable_8192.sh` 模式，注意 setuptools-rust mock 问题历史）
2. vllm-ascend 保持 kv_metrics_observability 分支（服务器已切换，与本地 `cf1296559` 一致）
3. 起服务后 `curl :8004/metrics | grep kv_pool` 确认 4 个指标出现
4. 跑多前缀长 prompt 负载（复用 `multi_growth_requests.sh` / `long_prompt_retest.sh`），观察：
   - `path=sync` 或 `async` 直方图随命中率变化
   - `delayed_release_requests` 随 prefix 复用窗口波动（对照 enable/disable prefix caching，参考项目记忆中"本地命中阻碍跨 worker transfer"的对照方法）
5. TP>1 场景专门确认 A1 聚合口径

### 7.4 待验证假设汇总（本地无法确认，需 v0.27.1 源码或实测）

| # | 假设 | 验证方式 | 验证结果（2026-08-26，见第八章） |
|---|---|---|---|
| A1 | 上游框架只从 rank0（或聚合全部 rank）收集 worker stats | 读 v0.27.1 `KVConnectorOutput`/stats 收集代码；或 TP2 实测 load_count 是否翻倍 | **已闭合：聚合全部 rank**。代码：`vllm/v1/worker/kv_connector_model_runner_mixin.py#L107` 每 rank 调用 → `vllm/v1/executor/multiproc_executor.py#L375-L378` 收全部 rank 响应 → `vllm/distributed/kv_transfer/kv_connector/utils.py#L120-L130` 逐 rank `aggregate()`。实测：TP=4 下 2 个 load 请求 count=8=2×4 |
| A2 | 框架对 `get_kv_connector_stats()` 返回 None 与空对象的处理一致 | 读框架调用点 | **已闭合：一致，均正确跳过**。worker 侧 None 由 `utils.py#L124` walrus 短路跳过；scheduler 侧空对象由 `vllm/v1/core/sched/scheduler.py#L1985-L1988` `is_empty()` 守卫跳过 |
| A3 | 框架构造 `metric_types` dict 的键序（gauge/counter/histogram）与 mock 假设一致 | 读 `build_prom_metrics` 调用方 | **已闭合：一致**。真实构造为 `{Gauge, Counter, Histogram}` 插入序（`vllm/distributed/kv_transfer/kv_connector/v1/metrics.py#L160-L164`），与 `_mock_deps.py#L164-L174` 假设吻合；运行时按键取值不依赖顺序 |

---

## 八、实证验证记录（2026-08-26，51 服务器）

> 本章为第一轮静态审查（2026-08-25）后的实测补充，用于给前七章的推断提供运行时证据。
> 环境按 7.3 前置动作清单搭建，验证过程与原始数据详见
> [PR14912_kv_metrics_冒烟验证报告.md](./PR14912_kv_metrics_冒烟验证报告.md)。
> **前七章结论均未被推翻**；本章给出状态更新（7.4 假设闭合、M3 阻断解除）与新发现 M5。

### 8.1 验证环境

| 项 | 值 |
|---|---|
| 服务器 / 容器 | 141.61.81.51 / `kv_metrics_51` |
| vllm | **v0.27.1 tag（`6e448d0ea9`）**，editable 安装（`git describe --tags` 确认） |
| vllm-ascend | `kv_metrics_observability` 分支（`cf1296559`）+ M5 临时补丁（第一轮 sync 见 8.4）；layerwise / async 两轮改用官方修复 `8c28898dd`（8.6） |
| NPU | 卡 12–15（TP=4，避开前段被占用卡） |
| 模型 | `/mnt/weight/qwen3-32b-pdmix` |
| KV 后端 | mooncake（master 127.0.0.1:50088，metrics :9008） |
| 关键参数 | `--no-enable-prefix-caching`、`kv_load_failure_policy=recompute`、`use_layerwise=false`（默认 sync 路径） |
| 负载 | 3 次 5042-token 同前缀长请求（1 save + 2 load）+ 短请求若干（触发 delayed-release） |

### 8.2 四指标实测快照（`curl :8004/metrics`）

| 指标 | 实测值 | 判定 |
|---|---|---|
| `vllm:kv_pool_load_duration_seconds{path=sync}` | count=8，sum=0.156048s；服务端日志 avg=19.506ms、p90=26.759ms | ✅ 上报 |
| `vllm:kv_pool_load_keys_total{path=sync}` | 312 | ✅ 上报 |
| `vllm:kv_pool_load_failed_keys_total{path=sync}` | 0（指标已注册） | ✅ 上报（失败路径未触发，见 8.6） |
| `vllm:kv_pool_delayed_release_requests` | 4.0（Prometheus 终值；日志周期值 3，不同时刻快照，符合 m5 gauge 语义） | ✅ 上报 |

**端到端链路实测贯通**（对应第四章"验收 2 暴露层"与需求内容 2 的"端到端未验证"缺口）：
worker 埋点 → per-rank `KVConnectorOutput` → `KVOutputAggregator` 跨 rank 聚合 → scheduler 再聚合 delayed_release 快照 → `SchedulerStats.kv_connector_stats` → API server Prometheus `/metrics`。

**数据自洽性核验（A1 实证）**：5042 tokens ÷ 128（`cache_transfer_granularity`）= 39.4，即每请求每 rank 39 个 key chunk；2 个 load 请求 × 4 rank × 39 = **312**，与 `load_keys_total` 完全吻合；count = 2 × 4 = **8**，与直方图计数完全吻合。mooncake 侧佐证：`master_allocated_bytes≈1.4GB`、`master_key_count=168`、外部前缀命中率 62.9%。

### 8.3 7.4 待验证假设闭合（代码 + 实测双证据）

**A1（= m4）：框架聚合全部 TP rank 的 worker stats，`load_count` 放大 TP 倍 —— 证实**

完整证据链（vllm v0.27.1，`6e448d0ea9`）：

1. ``vllm/v1/worker/kv_connector_model_runner_mixin.py#L107``：每个 TP rank 的 model runner 在 `execute_model` 收尾调用 `kv_connector.get_kv_connector_stats()`，各 rank 独立产出 stats；
2. ``vllm/v1/executor/multiproc_executor.py#L375-L378``：配置 `KVOutputAggregator` 时 `output_rank=None`，即从**全部 rank** 收集响应而非只取 rank0；
3. ``vllm/distributed/kv_transfer/kv_connector/utils.py#L120-L130``：注释即"Aggregate kv_connector_stats from all workers"，逐 rank `aggregate()`（首 rank 作 accumulator，后续 extend 拼接 load 列表、gauge latest-wins）；
4. ``vllm/v1/core/sched/scheduler.py#L1979-L1993``：worker 聚合结果再与 scheduler 侧 delayed_release 快照聚合；
5. ``vllm/v1/metrics/loggers.py#L1145-L1148``：Prometheus `observe()`。

实测：TP=4、2 个 load 请求 → count=8、keys=312（见 8.2 自洽推导）。

**看板影响（原 7.2 建议 5 的答案）**：`load_count` 与 `load_keys_total` 的速率类聚合（如 keys/s）含 TP 倍重复计数，看板侧需 ÷TP 或在 record 侧增加 rank 维度；avg/P90 不受影响（histogram 的 sum/count 自动正确）。

**A2：None 与空 stats 对象处理一致 —— 证实**

- worker 侧空时返回 `None`（``pool_worker.py#L1054-L1061``）：聚合侧 ``utils.py#L124`` 的 `elif kv_connector_stats := kv_output.kv_connector_stats` 短路跳过；
- scheduler 侧空时返回空对象（``pool_scheduler.py#L883-L887`` 无条件返回并重置）：消费侧 ``scheduler.py#L1985-L1988`` 的 `is not None and not is_empty()` 双守卫跳过。
- 两侧最终都不产生脏数据。m6 描述的"契约不一致"事实存在，但框架守卫已消化，**风险解除**。

**A3：mock 的 metric_types 键序假设与真实框架一致 —— 证实**

真实构造（``vllm/distributed/kv_transfer/kv_connector/v1/metrics.py#L160-L164``）：
`{Gauge: self._gauge_cls, Counter: self._counter_cls, Histogram: self._histogram_cls}`（插入序 Gauge→Counter→Histogram），与 ``_mock_deps.py#L164-L174`` 的假设及注释完全吻合。运行时子类按键取值（L123-125 `metric_types[Gauge]` 等）不依赖顺序，实测 4 个指标类型（histogram/counter/counter/gauge）均正确注册。

### 8.4 M5：实测发现的 worker 角色阻塞缺陷（原审查遗漏）

PR 原始代码 `cf1296559` 在 51 环境的真实表现序列：

1. vllm 升级 v0.27.1 后，服务**可启动**（`get_kv_connector_stats` 仅在 `execute_model` 收尾调用，启动路径不触发）；
2. 发送**第一个请求**，4 个 worker rank 全部抛 `AttributeError: 'AscendStoreConnector' object has no attribute 'connector_scheduler'`（详见第五章 M5 代码证据）；
3. 应用 `getattr` 补丁后，全部验证才得以进行——即**本章 8.2/8.3 的全部实测数据均采集自打了 M5 补丁的代码**，PR 原始代码无法完成本验证。

该缺陷同时说明：第一轮报告 7.2 建议 1（"升级 vllm 后即可验证"）过于乐观——版本阻断（M3）解除后还有代码自身阻断（M5）。**PR 合入前必须修复 M5**。

### 8.5 既有结论的实测复核状态

| 结论 | 复核状态 | 依据 |
|---|---|---|
| M1 layerwise 计时污染 | **维持且实测强化**（8.6 layerwise 轮）：avg≈230ms ≈11.5× sync 的 ≈20ms，全部观测挤在 `(0.2,0.3)s` 桶——表明测得的是"计算推进到最后一层的 span"而非纯加载耗时 | 代码+实测 |
| M2 PD 分离缺失 | **维持**（本次为单机池化验证，无 PD 新证据） | 静态证据 |
| M3 vllm 版本硬耦合 | **事实维持、阻断已解除**：0.26.1 下 import 失败（第一轮实测）；按 7.2 建议 1 升级 v0.27.1 后服务正常运行——建议 1 可行性已证实 | 双轮实测 |
| M4 layerwise 失败键恒 0 | **维持且实测确认**：`load_failed_keys_total{path=layerwise}=0`（8.6）——与代码不传 `num_failed_keys` 一致，失败观测缺口坐实 | 代码+实测 |
| M5 worker 角色崩溃 | **本轮新发现**（8.4） | 实测 |
| m1 async duration 含排队 | **维持且方向一致**（8.6 async 轮）：avg 24.8ms > sync 20ms，差 ≈5ms 小、短队列下排队污染可忽略 | 代码+实测 |
| m2 sync duration 含 key 准备 | **维持且量化**：起表在 key 构建前（L862）、止表在 `m_store.get` 后（L949-958）；实测 5042 tokens / 39 keys（每 rank）avg 19.506ms、p90 26.759ms——量级可控，但语义偏差（非纯 get 耗时）成立 | 代码+实测 |
| m3 num_keys 口径不一致 | 维持；实测佐证：sync/async 口径=本 rank chunk 数（39=floor(5042/128)）；**layerwise= blocks×layers（39×64=2496/事件）**，同比放大 64 倍，跨 path 直接对比严重失真 | 代码+实测 |
| m4 = A1 | **已闭合：放大 TP 倍**（8.3） | 代码+实测 |
| m5 gauge 最新快照 | **实测符合**（终值 4.0 与日志周期值 3 为不同时刻快照） | 实测 |
| m6 两侧返回契约不一致 | 事实维持，**风险解除**（A2 证实框架守卫已消化） | 代码 |
| m7 `python -O` 剥离 assert | 维持 | 静态证据 |
| 短 prompt 无指标（本轮新观察） | `cache_transfer_granularity=128`，<128 token 的 prompt 触发 `skip_save`，不产生任何 KV 存取——**符合阈值设计**，但看板侧需知晓"无跨机复用时该指标族无数据"属正常 | 实测 |

### 8.6 layerwise 与 async 路径实测（本轮补齐，官方修复 8c28898dd）

补齐 8.5 中"layerwise、async 实测未覆盖"的缺口。两轮均以**官方修复提交 `8c28898dd`**（属性置空修复，替代临时 getattr 补丁）为基线，与 8.2 sync 轮同模型、同 KV 后端、TP=4（卡 12–15）、同 5042-token 长 prompt。

| 轮次 | path | `load_duration` | `load_keys_total` | `load_failed_keys_total` | `delayed_release_requests` |
|---|---|---|---|---|---|
| 第一轮 | sync | count=8, sum≈0.160s, avg≈20ms | 312 | 0 | 3.0 |
| 第二轮 | layerwise | count=8, sum=1.845s, avg≈230ms（全体落 `(0.2,0.3)s` 桶） | **19968** | **0** | 0.0 |
| 第三轮 | async（6 并发同前缀请求） | count=24, sum≈0.595s, avg≈24.8ms, p90≈32.0ms | **936** | 0 | 1.0 |

**实测结论（对应前文缺陷）**：

1. **M1（layerwise 计时污染）——实测强化确认**：layerwise avg≈230ms ≈ **11.5 倍**于 sync 的 ≈20ms，且全部 8 个观测挤在 `(0.2,0.3)s` 一档。数值远超同负载 sync 路径纯 `m_store.get` 量级，佐证止表取 `wait_for_layer_load()` 返回时刻、实际反映"计算推进到最后一层的 span"（加载已被计算掩盖时的加载耗时上界，见第五章 M1）。8.5 由静态证据升格为代码+实测。
2. **M4（layerwise 无失败键统计）——实测确认恒 0**：`load_failed_keys_total{path=layerwise}=0`，与代码 `_record_layerwise_load_finished` 不传 `num_failed_keys` 一致（``pool_worker.py#L1046-L1047``）。
3. **m3（num_keys 跨路径口径）——layerwise 口径实证**：19968 ÷ 8 事件 = 2496/事件 = **39 blocks × 64 layers**（Qwen3-32B 共 64 层）。即 layerwise 的 `load_keys_total` 是"跨层 block 传输数"，与 sync/async 的"本 rank key chunk 数"（39）同比放大 64 倍。**跨 path 直接对比 keys 会严重失真**，看板如需对比须先换算口径。
4. **async 指标上报 ✅、m1 方向性一致**：6 并发同前缀请求全部走加载路径，count=24=**6×4**（再证 A1），keys=936=6×39×4 自洽。avg 24.8ms > sync 20ms，与 m1（async duration 含接收线程排队等待）方向一致；绝对差 ≈5ms 小，5042-token 短队列下排队污染可忽略。
5. **m4=A1 再证**：第三轮 count=24 = 6 并发 × 4 rank，聚合放大 4 倍在并发场景同样成立。
6. 观察：mooncake master 侧 `exist/put/get` 四类请求计数三轮均为 0，但 sync 轮 master 内存/命中率指标有值——推测该轮 KV 经 mooncake 内存路径命中、统计未落在被观测的 4 个字段上，属观测口径问题，不干扰本 PR 指标结论。

### 8.7 尚未覆盖的验证缺口（更新后）

1. **失败路径**：`load_failed_keys_total` 恒为 0（三条路径均未真实触发失败观测），需构造 backend get 失败场景（如强制 `tp_mismatch` 或 store 返回失败键）；
2. **prefix caching 对照**：7.3 第 4 条建议的 enable/disable 对照仅跑了 disable 单边（`--no-enable-prefix-caching`），enable 边对"本地命中抑制跨机 transfer"的影响未测；
3. **M5 修复的规范化**：layerwise/async 两轮已在官方 `8c28898dd` 下跑通，验证通过；但该提交需正式走 PR 评审合入流程，本报告见证的是其实际可运行性。

### 8.8 对第七章判定的影响汇总

- 7.1 交付判定中"验收 2 基本成立"：**增强**——sync/layerwise/async 三条路径端到端（worker→Prometheus）均实测贯通，"暴露层端到端未验证"的保留意见消除；但 layerwise 数据因 M1 仍不可用于"加载耗时"结论（见 8.6）；
- 7.2 建议 1（升级 vllm 解除 M3 阻断）：**已执行且证实可行**，但需追加前置——先修 M5（官方 `8c28898dd` 已给出）；
- 7.2 建议 5（看板使用前验证 A1）：**已完成**，答案为"放大 TP 倍"，看板聚合需 ÷TP；
- 7.3 前置动作清单：1-3、5 已执行；4 部分执行（长 prompt 负载 ✅，prefix caching 对照未做）；
- 新增合入门槛：**M5 必须修复**（已由官方提交 `8c28898dd` 修复，实测两条路径验证通过）。

---

## 附录：审查中形成的关键代码证据索引

| 证据 | 位置 |
|---|---|
| 指标定义与 observe | ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py#L118-L210`` |
| 数据模型/聚合/reduce | ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py#L44-L115`` |
| sync 计时起止 | ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L862`` |
| layerwise 起止与簿记 | ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1010-L1051`` |
| layerwise 止表点（M1 核心） | ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1741-L1763`` |
| async 止表点 | ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L1022-L1033`` |
| 延迟释放快照 | ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L875-L889`` |
| 框架挂接三入口 | ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L300-L331`` |
| preempted 清理 | ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L2152-L2154`` |
| 版本声明 | ``.github/vllm-release-tag.commit``（v0.27.1）、``Dockerfile.a3#L57`` |
| M5 崩溃点（第八章新增） | ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L306``（对比 L114-125 角色分派） |
| worker stats 每 rank 采集（A1） | ``vllm/v1/worker/kv_connector_model_runner_mixin.py#L107`` |
| 跨 rank 聚合（A1 核心） | ``vllm/distributed/kv_transfer/kv_connector/utils.py#L120-L130``、``vllm/v1/executor/multiproc_executor.py#L375-L378`` |
| scheduler 侧聚合与守卫（A2） | ``vllm/v1/core/sched/scheduler.py#L1979-L1993`` |
| metric_types 真实键序（A3） | ``vllm/distributed/kv_transfer/kv_connector/v1/metrics.py#L160-L164`` |
| Prometheus observe 入口 | ``vllm/v1/metrics/loggers.py#L1145-L1148`` |