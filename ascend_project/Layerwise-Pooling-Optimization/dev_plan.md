# Layerwise 池化优化 — 第一轮开发计划（指标基座 + not overlapped）

> 分支：`layerwise_pooling`（基线 upstream/main @ `ff998aad1`）
> 上游文档：[implementation_plan.md](implementation_plan.md)（P0-T1、P2-T1）、[design_proposal.md](design_proposal.md)（第七节）
> 参考资产：`kv_metrics_observability` 分支 7 提交（PR #14912，已过 B/C 两轮审查）；vLLM-Observability 目录的验证记录
> 编制：2026-08-26

---

## 一、本轮范围与理由

本轮交付 SR 需求"极光监控大盘"两个指标缺口中的技术部分：

1. **D1 整合**：把 `kv_metrics_observability` 的 7 个提交（4 个池化指标 + vLLM 指标框架挂接 + B/C 轮审查修复）带到 `layerwise_pooling` 新基线。此前该分支基于旧 main（fa668c649），未合入 upstream，且其上遗留两项待办正好由本轮补齐——165 全量 UT 重跑（B 文档 P0-1）、layerwise 指标语义修复后的真实部署验证（P0-2）。
2. **D2 not overlapped 指标**：SR 需求原文"支持观测未掩盖 KVC 池化（not overlapped）读取时间的 metric"。这是验收"劣化 ≤5%"的机理证据核心，当前代码只有总耗时口径。基建 `_TimedLayerLoadEvent`（事件 set 时刻记录）已就绪，B 文档 P2-7 明确"加一个差值字段即可"。
3. **D3/D4 配套**：UT 补充与静态检查，push 到 origin fork 供 165 拉取。

不在本轮范围：DSV4 Flash 适配（R1 裁决，P0-T4，需 51 环境）、性能调优参数矩阵（Phase 2）、延迟释放指标（已在 7 提交中交付）。

---

## 二、D1 整合 kv_metrics_observability

### 2.1 内容

7 个提交按序 cherry-pick 到 `layerwise_pooling`：

| 提交 | 内容 |
|---|---|
| `fa6cff82b` | 主体：metrics.py（新文件 223 行）+ 5 个源文件挂接 + 21 UT |
| `cf1296559` | layerwise load keys 计数补 partial_block、零 key 跳过 |
| `8c28898dd` | M5 修复：worker 角色前置声明 connector 属性（51 实测 4 rank 崩溃问题） |
| `e29cf99b6` | ruff 格式化 |
| `4ca18b752` | mypy 类型收窄（assert is not None） |
| `1923c61c6` | mypy mock TypeVar 对齐 |
| `089de2fc1` | C 轮：layerwise 止表改传输线程完成时刻（M1）+ observe 显式校验（m7）+ M4 文档化 + 1 UT |

操作：`git cherry-pick fa668c649..kv_metrics_observability`

### 2.2 冲突预期与处置

合并基 fa668c649 到 ff998aad1 之间，动过 `ascend_store/` 或 `tests/ut/distributed/ascend_store/` 的 upstream 提交只有一个：`2ffcca0af`（Mooncake tenant ID），改动文件为 `backend/mooncake_backend.py` 与 `test_backend.py`——与指标分支的 10 个文件**零交集**，预期干净落地。若 pool_worker.py 等文件因上下文漂移出现小冲突，逐 hunk 保留指标逻辑，冲突处置原则：指标改动是纯增量（新增类、新增方法、新增调用点），不改动既有逻辑行，冲突多为相邻行偏移。

### 2.3 落地后验证

本地（Windows，CPU 可跑）：`py_compile` 全部改动文件；`pytest tests/ut/distributed/ascend_store/test_metrics.py tests/ut/distributed/ascend_store/test_ascend_store_connector.py`（预期 42 passed，与 B 文档本地口径一致）。
全量回归在 165 执行（见 [test_plan.md](test_plan.md) T2）。

---

## 三、D2 not overlapped 指标设计

### 3.1 语义定义

**not overlapped**（未掩盖读取时间）= 计算线程因等待 KV 到达而阻塞的净时长，请求粒度按层累加。

每层净等待：

```
T_enter_i = 计算线程进入第 i 层 wait 循环的时刻（wait() 调用前采样）
T_set_i   = 传输线程置位第 i 层完成事件的时刻（_TimedLayerLoadEvent.set_time，已有）
stall_i   = max(0, T_set_i − T_enter_i)
```

请求粒度：`not_overlapped = Σ stall_i`（一个 step 内全部层）。

与已有 duration（总加载耗时）的关系：

```
duration      = T_set_last − T_submit_first   （首个层任务提交 → 最后一层传输完成）
not_overlapped ≤ duration                     （计算线程串行等待，各层 stall 互不重叠且落在窗口内）
overlap_ratio = 1 − not_overlapped / duration ∈ [0, 1]
```

sync/async 路径不与计算并发，全暴露：`not_overlapped ≡ duration`，ratio ≈ 0；layerwise 理想状态 ratio → 1。三条 path 同图呈现即可直观读出 layerwise 的掩盖收益。

### 3.2 埋点改动（pool_worker.py）

1. `start_load_kv` layerwise 分支：新增 `self._step_not_overlapped_s = 0.0`（与 layer_load_tasks 重置同处）。
2. `wait_for_layer_load`：wait 循环前采 `wait_enter = time.perf_counter()`；循环退出后读该层事件 `set_time`，`self._step_not_overlapped_s += max(0.0, set_time − wait_enter)`。
3. `_record_layerwise_load_finished`：duration 记账时把 `_step_not_overlapped_s` 一并传入，本 step 每个请求同值落账（与 duration 的请求粒度口径对齐，两个直方图样本数一致）。

### 3.3 数据结构与指标（metrics.py）

`record_load()` 增加 `not_overlapped_seconds: float | None = None` 参数，None 时取 duration（sync/async 默认语义）。load record dict 增加 `not_overlapped_seconds` 字段。

`reduce()` 增加：`not_overlapped_avg_ms`、`not_overlapped_p90_ms`、`overlap_ratio_avg`。

`AscendStorePromMetrics` 新增两个指标：

| 指标 | 类型 | buckets | 说明 |
|---|---|---|---|
| `vllm:kv_pool_load_not_overlapped_seconds` | Histogram（label path） | 1ms…5s（同 duration 桶，加密 200ms/500ms 档） | 请求粒度未掩盖时间 |
| `vllm:kv_pool_load_overlap_ratio` | Histogram（label path） | [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0] | 掩盖率，大盘直接读数 |

服务端聚合日志行扩展：`not_overlapped_avg_ms / not_overlapped_p90_ms / overlap_ratio_avg`。

### 3.4 边界情况处置表

| 场景 | 处理 | 理由 |
|---|---|---|
| 事件早于等待就绪（T_set ≤ T_enter） | max(0,·) → 0 | 完全掩盖，无阻塞 |
| `should_wait=False`（该层无加载任务） | 不进入等待，贡献 0 | 无传输依赖 |
| prefetch 层（在 prefetch_layer_map 中但无任务） | 仍等待，计入 | 属传输相关的真实阻塞 |
| timeout=10 重试循环 | 用 set_time 而非 wait 返回值 | 循环次数不影响测量 |
| 跨步残留 set_time（clear() 保留旧值） | 旧值 < 新 T_enter，max(0,·) 归零 | 天然免疫，与 C 轮 max 取值设计一致 |
| 测试用普通 Event（无 set_time） | getattr 判 None 跳过累加 | duration 照记，not_overlapped 缺省 |
| 请求中途 preempted | 沿用 `_load_start_times` 清理路径 | step 级累加值随本步正常落账或整体丢弃 |
| num_keys=0（无有效加载） | 不落账 | 与现有 duration 口径一致 |

---

## 四、D3 UT 补充清单

新增到 `test_metrics.py`（沿用 `_mock_deps` 框架）：

| 用例 | 验证点 |
|---|---|
| `test_not_overlapped_accumulates_layer_stalls` | 构造 2 层 timed event，层 0 晚于 wait_enter 0.03s set、层 1 早于 wait_enter set，断言累加值 ∈ 预期区间（仅层 0 贡献） |
| `test_not_overlapped_zero_when_fully_overlapped` | 所有层事件先于等待就绪 → not_overlapped=0、ratio=1 |
| `test_sync_path_not_overlapped_equals_duration` | sync 落账记录 not_overlapped == duration（默认语义） |
| `test_overlap_ratio_bounds` | 构造部分掩盖样本，ratio ∈ (0,1)；sum(stall) ≤ duration 不变式 |
| `test_step_not_overlapped_reset_between_steps` | 两个 step 连续，第二步累加不含第一步残留 |
| `test_prom_metrics_observe_not_overlapped` | observe() 后两个新 Histogram 出现样本、label path 正确 |

`test_pool_worker.py` 补 1 个：`wait_for_layer_load` 计时路径在普通 Event（无 set_time）下不抛异常、duration 照常落账。

探索性用例（R1 预判，独立提交、可放弃）：构造 c1/c4/c128 多 cache family 的 hybrid `kv_cache_config` + `use_layerwise=True` 初始化 KVPoolWorker/KVPoolScheduler，观察是否 raise。参考 `test_pool_scheduler.py` 已有的 mamba group 测试构造方式。产出只用于预判 DSV4 冒烟（P0-T4）风险，不作为裁决依据。

---

## 五、D4 静态检查

与 CI 一致的口径（沿用 B 轮命令）：

```
ruff check <files> && ruff format --check <files>   # v0.14.0
mypy --follow-imports skip --check-untyped-defs <files>
```

mypy 本地缺 numpy/torch stub 的 import-not-found 属已知噪声，以改动文件无新错误为准。

---

## 六、D5 提交划分与分支策略

| 提交 | 内容 |
|---|---|
| 1（cherry-pick ×7） | 原样保留 7 个提交的独立历史，便于与 PR #14912 对账 |
| 2 | not overlapped 埋点（pool_worker.py） |
| 3 | 指标暴露（metrics.py）+ UT |
| 4（可选） | R1 预判探索性 UT |

全部提交带 `Signed-off-by: lizy124 <1950471827@qq.com>`（DCO 要求）。

完成后 push `layerwise_pooling` 到 origin（lizy124 fork）——165 的 UT 脚本从 github 拉分支（沿用 scripts_165/run_ut.sh 模式）。上游 PR 不在本轮：not overlapped 待 165 实测确认语义后再合入指标 PR（#14912 的后续提交）。

---

## 七、依赖与风险

| 项 | 说明 | 应对 |
|---|---|---|
| vLLM 版本耦合 | metrics 框架依赖 vLLM 的 `KVConnectorPromMetrics`（v0.27.x 引入）；165 容器 vllm 版本待核查 | 测试计划 T0 第一项核查；若容器版本旧，复用 refactor_812 或按上次流程升级 |
| layerwise 冒烟需 memcache backend | 165 未确认装过 memcache_hybrid/memfabric_hybrid | 测试计划 T0 核查 + 安装路径 + 降级路径已列 |
| cherry-pick 后 UT 数量口径 | B 文档预期 311+3，再加本轮新增约 7 个 | 以 165 实跑为准，记录精确数 |
| not_overlapped 首版语义争议 | step 级同值落账 vs step 级单样本 | 先按请求粒度同值落账（与 duration 对齐），实测后如有解读问题再调整 |

---

## 八、第一轮执行结果（2026-08-26）

本轮按用户裁定范围执行：代码开发 + UT（服务器）+ 静态检查；端到端冒烟（test_plan T3/T4）不在本轮。

**提交**（`layerwise_pooling`，已推 origin）：

- D1：cherry-pick 7 提交零冲突落地（b1f0c3e22…d293e3282）
- D2：`7d4c3e543` not overlapped 指标完整特性（pool_worker 埋点 + metrics 暴露 + 9 个 UT）+ `dcdb215c2` mypy 类型修复（`layer_load_finished_events` 可空收窄）
- D5：D2 实现为一整个提交而非计划中的两个——埋点与 record_load 新参数跨文件耦合，拆开会使中间提交不可运行（计划中提交 2 单独存在会 crash），单提交保证 bisect 安全

**静态检查**（本地，与 CI 同版本）：

- ruff 0.14.0 check/format：3 文件全过
- mypy `--follow-imports skip --check-untyped-defs`：真实错误清零（import-not-found 为本地缺 stub 噪声）

**UT 结果**：

| 环境 | 范围 | 结果 |
|---|---|---|
| 本地（Windows，mock 驱动脚本） | test_metrics.py 全部 32 用例 | 32 passed（含 9 个新用例）；全目录 305 用例与 D1 基线对照：4 fail + 62 error 完全一致，均为本地无 vllm/torch 的 mock 环境产物，非回归 |
| 165（镜像 7f06feda13d3 = cxy_cann9.1.0 同镜像，独立容器 + NPU 驱动挂载 + 反向隧道代理） | test_metrics.py | **32 passed** |
| 165（同上） | ascend_store 全目录 | **331 passed, 0 failed**（较 B 轮 314 增量来自 upstream 新增用例 + 本轮 9 个） |

**环境事实记录**（后续轮次复用）：

- 新基线（ff998aad1，含 upstream #14746 vllm 0821 同步）要求 vllm ≥ ba07e4a48：`vllm.v1.attention.ops.pcp` 在 pip 装的 vllm 0.27.1 里不存在，refactor_812（旧容器）跑 UT 直接 ImportError——**refactor_812 不再适用于新基线**
- 可用路径：镜像 7f06feda13d3 的 `/vllm-workspace/vllm` 源码树含 pcp，起独立容器（--network host + cxy 同款 NPU 设备/驱动挂载 + 本机 7897 反向隧道代理）即可跑通
- 165 上 git clone 需经代理（宿主机与新容器均无直连）；隧道命令：`ssh -N -R 127.0.0.1:7897:127.0.0.1:7897 root@192.168.13.165`
- 本地 Windows 可用 `local_run_metrics_ut.py`（本目录）跑 mock UT，注意 `_mock_deps` 在无 vllm 环境下不设父包属性，patch 类用例需驱动脚本补线（脚本已处理）

**遗留到下一轮**：test_plan T3/T4（layerwise 冒烟 + 指标语义实测，需 memcache 组件与 DeepSeek 系模型，环境核查见 test_plan T0）；D3 的探索性 UT（多 cache family × layerwise 预判）未做——待 DSV4 冒烟裁决时一并处理更有价值。
