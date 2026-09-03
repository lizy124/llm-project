# B 轮审查修复记录 + 后续需要的测试验证

对应 PR：vllm-ascend #14912（分支 `kv_metrics_observability`）
修复提交序列：`8c28898dd`（M5）→ `e29cf99b6`（ruff 格式化）→ `4ca18b752`（mypy 类型收窄）→ `1923c61c6`（mypy TypeVar）→ `089de2fc1`（C 轮：M1 止表点 + m7 + M4 文档化）

---

## 一、C 轮修复（2026-08-26，提交 `089de2fc1`）

依据：[g-review-report.md](g-review-report.md)（M1/m7 项）+ 本人对代码证据的复核。改前已逐条核实：

1. kv_transfer.py 中 layer_load_finished_events 的 set 共 4 处（L1280/L1523/L1544/L1608），全部在传输线程侧；
2. wait_for_layer_load()（pool_worker.py L1745-1764）在事件 wait 返回后才调 `_record_layerwise_load_finished()`——止表确实是"计算侧等待返回时刻"；
3. layerwise 批量拷贝失败路径是 `raise RuntimeError`（kv_transfer.py L1596），非 soft-fail——M4 不加计数、只文档化的依据。

### 1. M1 修复（Major）— layerwise 止表改为传输线程完成时刻

**问题**：layerwise 直方图止表取 `wait_for_layer_load()` 返回时刻。加载快于计算（掩盖良好）时，wait 立即返回，测量值 ≈ 计算耗时，指标变成 `max(加载span, 计算span)`，是加载耗时上界。51 实测 avg≈230ms（≈11.5× sync），全部挤在 (0.2,0.3)s 桶，坐实污染。

**修复**（pool_worker.py）：
- 新增 `_TimedLayerLoadEvent(threading.Event)` 子类：`set()` 时记录 `set_time = time.perf_counter()`。事件在**传输线程**侧 set，即该层真实加载完成时刻。
- `layer_load_finished_events` 创建处改用该子类（仅 pool_worker.py 一处；kv_transfer.py 的 4 处 set 点**零改动**，多态自动生效）。
- `_record_layerwise_load_finished()` 止表改为 `_latest_layer_load_finish_time()`：取所有层事件 set_time 的 **max**（请求的 KV 全部层完成才算加载完成；clear() 不清 set_time，保证早层被 clear 后时间戳仍可读；旧步残留的时间戳必然更小，max 自动忽略）。
- 防御：无可用时间戳或时间戳早于本步最早 start（理论上不会发生）时回退 `perf_counter()`。

**语义变化（需知晓）**：
- 修复后 layerwise duration = 任务提交 → 最后一层传输完成，**不再包含**"加载完成后计算继续跑"的尾巴；
- 仍**包含** prefetch 层等待 attention_start_gate 的时间（kv_transfer.py L1557-1559，传输前等计算开始信号）——这是加载对计算的依赖，属加载路径一部分，保留；
- not overlapped 时间（SR 需求）= wait 返回时刻 − 事件 set 时刻，本轮未产出新指标，可由日志推导，SR 后续单独做。

### 2. m7 修复（Minor）— observe() 显式校验

metrics.py `observe()` 中 `assert isinstance(record, dict)` 改为 `if not isinstance(record, dict): continue`——`python -O` 下 assert 被剥离会导致裸 AttributeError，显式跳过更安全。

### 3. M4 处置 — 定性为"语义正确"，文档化而非加计数

B 轮已存疑，本轮核实：layerwise 传输失败是 `raise RuntimeError`（kv_transfer.py L1596），线程直接死亡，不存在逐键 soft-fail 路径。因此 `load_failed_keys_total{path=layerwise}` 恒 0 是**语义正确**而非观测缺口。已在 metrics.py 两处 docstring 写明：该 counter 仅对 sync/async 有意义。

### 4. 新增 1 个 UT

`test_layerwise_duration_uses_event_set_time`：构造 2 层 timed events，层1 先 set（0.02s）、层0 后 set（0.05s），再模拟 0.2s 计算尾巴。断言 duration ∈ [0.05, 0.15)——同时验证"取事件时刻而非 wait 返回时刻"和"跨层取 max"两个行为。

### 5. 本轮验证结果（本地）

| 项 | 结果 |
|---|---|
| UT：test_metrics + test_ascend_store_connector | **42 passed**（41 旧 + 1 新） |
| ruff check / format（v0.14.0，与 CI 一致） | 全过 |
| mypy（--follow-imports skip --check-untyped-defs） | 改动文件无错误（仅本地缺 numpy/torch stub 的 import-not-found，CI 环境有） |

---

## 二、历史轮次修复摘要

### B 轮（`8c28898dd`）

1. **M5（Critical）**：`__init__` 统一前置声明 `connector_scheduler`/`connector_worker` 为 None，修复 worker 角色 `get_kv_connector_stats()` AttributeError（51 实测 4 rank 首请求全崩）。后追加 `4ca18b752`：5 个 worker 侧方法加 `assert ... is not None` 满足 mypy union-attr。
2. **m2/m3**：metrics.py docstring 写清三条 path 计时起止范围、num_keys 口径、gauge latest-snapshot 语义。
3. **回归 UT ×2**：worker 角色不崩 + scheduler 角色优先。
4. `1923c61c6`：mock TypeVar 名对齐（mypy [misc]）。

---

## 三、后续需要增加的测试验证（按优先级）

### P0 — 本轮 M1 修复必须在真实环境重验（指标语义变了）

1. **165 全量 UT 重跑**
   - `pytest tests/ut/distributed/ascend_store/ -v`
   - 预期：311+3 全过（新增 test_layerwise_duration_uses_event_set_time）

2. **layerwise 真实部署对比（M1 修复的核心验证）**
   - 与 51 第二轮同形态：TP=4、`use_layerwise=true`、同 5042-token 负载
   - **预期**：`load_duration_seconds{path=layerwise}` avg 从 ≈230ms 显著回落（修复前测的是计算 span；修复后是真实传输完成时刻）
   - 交叉验证：取若干请求，用 kv_transfer 日志的 "Layer load event set: layer N" 时间戳减去 step 开始时间，与 histogram 样本量级对照
   - 观察桶分布：不再全部挤在 (0.2,0.3)s 一档

3. **回归确认**：sync/async 路径指标不受影响（本轮未动这两条路径的埋点，但 pool_worker 改动需回归）

### P1 — 遗留缺口（与 B 轮相同，仍待做）

4. **failed_keys 非零场景**（sync/async）：先写后删部分 key 再加载，验证 `load_failed_keys_total` 真实计数（该 counter 从未被非零验证过）
5. **prefix caching enable 对照**：验证本地命中抑制跨机 transfer 时指标族无数据属正常
6. **M2 PD 分离验收**：与需求方确认 prefill/decode 场景是否本期范围

### P2 — 后续可选

7. **not overlapped 指标**（SR 需求）：在 timed event 基础上记录 wait 返回时刻，差值即"计算等待加载"时间。基建（_TimedLayerLoadEvent）已就绪，加一个差值字段即可
8. **n3 跨步残留防御**：极端场景个别样本偏大，概率低

---

## 四、审查报告各项处置状态（截至 C 轮）

| 项 | 处置 |
|---|---|
| M1 layerwise 计时污染 | ✅ **C 轮已修**（`089de2fc1`，事件 set 时刻止表），待 P0-2 实测确认 |
| M3 版本耦合 | 部署环境问题（51 已升 v0.27.1），非代码缺陷，关闭 |
| M4 layerwise 失败键恒 0 | ✅ 定性为语义正确（失败=raise 非 soft-fail，L1596 核实），docstring 已写明，关闭 |
| M5 worker 角色崩溃 | ✅ B 轮已修（`8c28898dd`+`4ca18b752`），51 实测通过 |
| m1 async 含排队 / m2 sync 含 key 准备 | docstring 已文档化，语义可接受，关闭 |
| m3 num_keys 口径 | docstring 已文档化，关闭 |
| m4 TP 聚合放大 | 已确认（count×TP 属预期），看板解读需 ÷TP，关闭 |
| m5 gauge 最新快照 | docstring 已写明，关闭 |
| m6 返回契约不一致 | 上游双重守卫消化，关闭 |
| m7 assert 剥离 | ✅ **C 轮已修**（显式 skip） |
| M2 PD 分离 | 待需求方澄清，非代码缺陷 |
| n3 跨步残留 | 低概率，P2 |
| 报告 UT 计数 21 | 实际 22（现 23），报告计数错误 |
