# SR20260820223202 Layerwise 池化性能优化 — 设计方案

> 需求编号：SR20260820223202（需求解读见同目录 [requirements_analysis.md](requirements_analysis.md)）
> 配套需求：AR20260820031213（可观测性，指标在两需求间交汇）
> 代码基线：vllm-ascend upstream/main @ `ff998aad1`（2026-08-26），开发分支 `layerwise_pooling`
> 编写时间：2026-08-26

---

## 一、设计目标与交付边界

### 1.1 验收锚点

| 要素 | 内容 |
|---|---|
| 模型 | DeepSeekV4 Flash（下文简称 DSV4 Flash） |
| 负载 | 平均 128K 输入，90% KVCache 命中，最优并发 |
| 对比 | DRAM 池化 vs HBM PrefixCache（多前缀、HBM 可缓存满前缀） |
| 指标 | Prefill TPS |
| 目标 | DRAM 池化相比 HBM PrefixCache 基线劣化不超过 5% |

需求的业务本质是"容量换性能"：Agentic 场景 200K 长序列、命中 >90%，HBM 装不下全量 KV，用池化溢出到 DRAM 扩容，再用 Layerwise 计算与传输并发把读取代价掩盖掉。设计方案围绕"如何把 not overlapped 时间压到趋近于零"展开。

### 1.2 交付边界

1. 本期验收只覆盖 **DRAM 池化**路径。SSD 只做预研数据采集，不设验收指标（需求价值提及 SSD，验收标准未覆盖，范围边界已在需求分析中确认）。
2. 特性叠加（MTP/DCP/LayerSplit）的交付形态是**验证结论加必要的兼容性修复**，不是全组合矩阵产品化。
3. 极光监控大盘交付两个指标：not overlapsed 池化读取时间、KV Cache 延迟释放 req 排队规模。

### 1.3 非目标（明确排除）

| 排除项 | 原因 |
|---|---|
| CP 变体（mla_cp / sfa_cp / attention_cp）与 layerwise 集成 | 上游明确列为后续工作，DSV4 Flash 验收不依赖 CP |
| MLA read dedup（issues_10 编号 kv-01）扩展到 layerwise | 该优化一期边界限定 non-layerwise 同步读路径 |
| non-layerwise IO 批处理（kv-07） | 只影响非 layerwise 路径，与本验收指标无关 |
| TP mismatch（producer/consumer TP 不一致） | 代码当前直接 raise，验收配置两边 TP 一致即可 |
| Mooncake backend 的 layerwise 支持 | layerwise 仅支持 memcache backend（见 2.2），扩展后端属后续演进 |

---

## 二、现状基线（ff998aad1）

### 2.1 已具备的能力

| 能力 | 来源 | 说明 |
|---|---|---|
| AscendStore 池化框架 | PR #13354 重构 | scheduler / worker 分体架构，backend 抽象（mooncake / memcache / yuanrong） |
| Layerwise 逐层传输 | `use_layerwise` 配置 | GVA layerwise（memcache backend），逐层收发线程，`wait_for_layer_load` / `save_kv_layer` 接口 |
| MTP + sparse C8 叠加 | PR #12853 | layerwise prefill offload 已支持 MTP 与稀疏 C8 布局 |
| 跨层 KV buffer 复用 | PR #12852 | prefill 侧 NPU 显存占用下降，`prefetch_layer_map` 与预取联动 |
| Sparse KV offload | PR #13026 | 支持 `index_topk` 类稀疏布局（DSV4 走此路径） |
| SFA attention 集成 | PR #14046 等 | `mla_v1` / `sfa_v1` 已接入 layerwise 等待/保存调用链 |
| KV pool 指标 | `kv_metrics_observability` 分支（未合入 main） | 4 个 Prometheus 指标，含 layerwise load duration（详见第七节） |
| nightly 基础设施 | `tests/e2e/nightly/.../kv_pool_runtime.py` | 单节点 KV pool 服务启停已有封装 |

### 2.2 关键约束

1. **layerwise 只支持 memcache backend**：`use_gva_layerwise = use_layerwise and backend == "memcache"`（pool_worker.py:157）。需求文案中的"MemCache/MoonCake 池化方案"在实现上落到 MemCache（GVA）路径——DRAM 池由 memcache_hybrid + memfabric_hybrid 提供，Mooncake backend 用于非 layerwise 池化形态。
2. **TP mismatch × layerwise / × sparse 直接拒绝**（pool_worker.py:230-247）。验收配置 producer/consumer TP 一致，规避即可。
3. **MTP 叠加存在已知截断**：eagle + layerwise 下 `num_external_hit_tokens` 被截断，trailing block 以 dirty data 方式加载（pool_scheduler.py 内 TODO 注明后续支持）。
4. **hybrid KV cache 与 layerwise 的支持现状存在文档与代码的偏差**：用户文档（layerwise_kv_pool.md）声明"多 cache family 走 layerwise 会 NotImplementedError"，但当前代码的 layerwise 发送线程已携带 `group_builders` 多 group 通道（pool_worker.py:489），`LayerTransferTask` / `LayerMultiBlockReqMeta` 也带 `kv_cache_group_id`。DSV4 Flash（c1/c4/c128 混合 cache family）能否直接跑通 layerwise，需要 Phase 0 实测裁决。这是本设计最大的不确定性，应对方案见第五节。
5. **cache_transfer_granularity 由各组 block size 的 lcm 推导**（metadata.py `infer_cache_transfer_granularity`）。DSV4 的 c128 组一个 cache block 对应 16384 token，粒度会被推高到万级 token。影响两件事：负载构造的前缀长度必须对齐粒度（否则实测命中率达不到设计的 90%）；历史上 granularity=4096 曾导致短 prompt `num_tokens_to_save=0` 而跳过 save。

### 2.3 代码地图

| 模块 | 路径（vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/） | 职责 |
|---|---|---|
| Connector 适配层 | ascend_store_connector.py | 实现 KVConnectorBase_V1 + SupportsHMA，按 role 代理 scheduler/worker |
| 调度侧 | pool_scheduler.py | 外部池命中查询（get_num_new_matched_tokens）、LoadSpec 构建、cache_transfer_granularity |
| Worker 侧 | pool_worker.py | start_load_kv / wait_for_layer_load / save_kv_layer，layerwise 任务编排 |
| 传输线程 | kv_transfer.py | KVCacheStoreLayerSendingThread 等收发线程族、LayerBatchBuilder |
| Buffer 布局 | layerwise_cache_layout.py | 跨层共享物理 buffer、prefetch_layer_map、independent_layers |
| 元数据 | metadata.py | PoolKey / LayerPoolKey / KeyMetadata / ReqMeta / 粒度推导 |
| 后端 | backend/（mooncake / memcache / yuanrong） | 存储后端实现，layerwise 走 memcache |
| 指标 | metrics.py（仅 kv_metrics_observability 分支） | load duration / delayed release 指标 |

---

## 三、总体架构

```
┌────────────────────────── vLLM 实例（PD-Mixed 或 P/D 分离） ──────────────────────────┐
│                                                                                      │
│  vLLM Scheduler ──► KVPoolScheduler                    vLLM Worker ──► KVPoolWorker  │
│   请求调度/排队        命中查询(ZMQ lookup)              前向计算         layerwise 编排 │
│                      update_state_after_alloc                          │            │
│                      build_connector_meta ──────────► AscendConnectorMetadata        │
│                                                                 ▼                    │
│                                          layer_load_tasks[i] / layer_save_tasks[i]   │
│                                          KVCacheStoreLayer Recv/Send Thread(s)       │
│                                          （按层提交/按层等待/预取/attention gate）      │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │ register_buffer / get / put（按层）
                                           ▼
┌──────────────────────────── memcache_hybrid + memfabric_hybrid 多级存储 ──────────────┐
│   HBM 热缓冲（topk_buffer_size）  →  DRAM 池（dram_size_per_dp_GB，本期验收）          │
│                                    →  SSD 扩展（memfabric，本期仅预研）                │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

要点：

- 池化不替换 vLLM 调度，KVPoolScheduler 嵌入 vLLM 调度循环，只负责外部池的命中查询、加载规格与元数据构建；本地 block 分配仍由 vLLM 完成。
- 51 服务器（8×A3）已有"每卡 5G DRAM 池化"现成配置，直接复用；验收期按数据集规模复核池容量（见 8.2）。
- DRAM 池的读写通道由 memcache_hybrid 管理，NPU HBM buffer 通过 `register_buffer` 注册，传输线程直接按层搬运，避免框架层多余的拷贝。

---

## 四、Layerwise overlap 流水线设计

### 4.1 加载路径（命中请求的 prefill）

```
Scheduler 侧                          Worker 侧（每步 forward）
────────────                          ──────────────────────────────
get_num_new_matched_tokens  ─┐
  ZMQ lookup 查池命中数       │        start_load_kv
update_state_after_alloc     │          ├─ process_layer_data：按层拆出
build_connector_meta ────────┘          │   LayerLoadTask[i]（含 group/block 区间）
                                       │   next_layer_to_submit=0
                                       ▼
                            前向第 i 层 attention 前
                            wait_for_layer_load(i)
                              ├─ 提交就绪层负载到 recv 线程
                              ├─ 等待 layer_load_finished_events[i]
                              └─ 第 i+1..i+P 层传输与第 i 层计算并发
```

- **流水线关系**：第 i 层 KV 在 DRAM→HBM 传输时，计算在算第 i 层（已到货）与准备第 i+1 层；预取深度 P（`layerwise_prefetch_layers`）决定领先计算前沿多少层。
- **attention_compute_start_gate**：计算侧与传输侧的同步门，避免"计算已就绪但该层负载尚未提交"的竞态；每步 forward 开始时 reset。
- **首层暴露**：第 0 层 KV 到达前没有任何可算的东西，这段等待是 not overlapped 的主要来源之一，通过调度间隙提前提交（build_connector_meta 到 forward 启动之间）压低。

### 4.2 保存路径（producer / kv_both）

```
第 i 层 attention 计算完成
  └─ save_kv_layer(i)
       ├─ sync_save_events[i].record()      # NPU event：等 KV 真正写进 HBM buffer
       ├─ 提交 layer_save_tasks[i] 给 send 线程
       └─ send 线程：等 event → 读 HBM → put 到 DRAM 池 → set layer_save_finished_events[i]
下一层计算与第 i 层保存并发
```

- 保存与加载共用一套逐层事件机制，`wait_for_save` 只在请求收尾兜底最后一层。
- **跨层 buffer 复用**（#12852）：相同 cache spec 签名的层共享物理 buffer（`buffer_slots`），prefetch_layer_map 把复用关系与预取绑定。DSV4 Flash 若含非 AttentionSpec 的 state 层，复用布局只覆盖 attention 层，state 层走 independent buffer（见 5.4）。

### 4.3 劣化模型（5% 目标的工程含义）

单请求 prefill 时间（池化）与理想 HBM 的关系：

```
T_pool   ≈ T_first + Σ_i max(C_i, W_i) + T_lookup + T_sched_extra
T_hbm    ≈ Σ_i C_i                （HBM PrefixCache 命中，无搬运）

劣化比   ≈ (T_pool − T_hbm) / T_hbm
```

其中 C_i 为第 i 层计算时间，W_i 为第 i 层传输时间，T_first 为首层到货等待。

**达标条件推导**：

1. 若每层 W_i ≤ C_i 且预取深度 ≥1，则 Σ max(C_i, W_i) = Σ C_i，层间传输被完全掩盖；
2. 劣化只剩 T_first、lookup 开销、调度开销三项，目标即 `T_first + T_lookup + T_sched_extra ≤ 5% × Σ C_i`；
3. 若某些层 W_i > C_i（DRAM 带宽不足），超出部分按层累积进 not overlapped，此时靠提高并发摊薄（多个请求的传输交错填满带宽）或增大预取深度部分掩盖。

**可行性前置测量**（Phase 0 执行，先于全量 E2E）：

| 测量项 | 方法 | 用途 |
|---|---|---|
| 每层 KV 字节数 | DSV4 Flash cache spec × 128K token 推导，加日志实测 | 算 W_i 的分子 |
| DRAM 池有效读带宽 | 51 服务器微基准（大块顺序读，绕过缓存） | 算 W_i 的分母 |
| 每层 attention 计算时间 | 128K 输入 prefill profile | 算 C_i |
| 判定 | W_i/C_i 比值分布 | 比值 ≤1 则物理上可完全掩盖；>1 的层数与超出幅度决定并发/预取策略 |

这一步的产出是一页"带宽-计算匹配表"，若出现大面积 W_i > C_i，提前向需求方报告物理边界，避免在不可达目标上消耗迭代轮次。

### 4.4 调优参数矩阵

| 参数 | 默认 | 作用 | 调优方向 |
|---|---|---|---|
| `layerwise_prefetch_layers` | 1 | 预取领先计算前沿的层数 | 带宽余量小则加大（典型 1–4），代价是 host 侧 staging 内存 |
| `h2d_stagger_us` | 0 | TP 各 rank H2D 拷贝错峰 | TP=8 时设 100 量级，缓解总线争抢 |
| `layerwise_max_transfer_blocks` / `..._bytes` | 0（不限） | 单批传输上限 | 防止单层大块独占总线；也可让首层分批到货、提前开算 |
| `layerwise_num_shared_buffers` | num_layers | 跨层共享物理 buffer 数 | 从 2–4 起调，平衡 HBM 占用与复用收益 |
| `layerwise_independent_layers` | [0] | 不参与复用的独立层 | state 层 / 首 Layerwise 层按需指定 |
| `discard_partial_chunks` | false（layerwise） | 非完整 chunk 处置 | 保持 false 以保留部分层数据 |

每项参数在 Phase 2 做独立 A/B 测量，配置矩阵记录进实验台账，最终锁定一组"验收配置"随报告归档。

---

## 五、DSV4 Flash 适配设计

### 5.1 cache family 与分组

DSV4 按 `compress_ratios` 解析出 c1 / c4 / c128 等 cache family，映射到多个 kv cache group（`infer_group_cache_families`）。每个 group 有独立 block size、cache spec 与传输粒度，layerwise 任务按 `(group_id, layer_idx_in_group)` 二维组织（`LayerTransferTask` / `LayerMultiBlockReqMeta` 均携带两维标识）。

### 5.2 granularity 对齐（负载构造的硬约束）

`cache_transfer_granularity = lcm(lcm_block_size, 各组 block size)`。c128 组的存在会把粒度推到 16384 token 量级（block_size=128 时一个 cache block 对应 128×128 token）。

设计上的应对：

1. **数据集前缀长度对齐粒度**：128K 输入、目标 90% 命中 → 命中部分取 7×16384=114688 token（约 89.6%），剩余 131072−114688=13384 token 为新增后缀。若实测粒度不同，按同一公式重算，保证"设计的命中率"与"实测的命中率"一致。
2. **冒烟测试的 prompt 长度必须 ≥ 粒度**：历史教训——granularity 过大导致 `num_tokens_to_save=0`、请求被静默跳过 save，池化验证出现"配了池但池里没数据"的假象。冒烟阶段加一条断言：save 后池内 key 数 >0。

### 5.3 多 group layerwise IO 的裁决路径

文档与代码在"hybrid × layerwise"上不一致（见 2.2 第 4 条）。裁决路径：

1. Phase 0 直接用 DSV4 Flash 配 `use_layerwise: true` 跑冒烟；
2. 若跑通：文档滞后，记录现状，进入正常基线流程；
3. 若 raise NotImplementedError / ValueError：定位拦截点，评估打通多 group 逐层 IO 的改动量。代码侧已有基础（发送线程的 `group_builders`、元数据的 group 维度），主要缺口大概率在接收线程按 group 分派与 hybrid group 的 block 区间映射，预估 1–2 周量级；此情况下 Phase 1 顺延，第一时间同步需求方调整时间预期。

### 5.4 state 层与 buffer 复用

`build_layerwise_reuse_layout` 只接受 AttentionSpec；若 DSV4 Flash 含 mamba/state 层，这些层进 `independent_layers`，复用只在 attention 层之间生效。mamba 混合模型要求 `mamba_cache_mode='align'`（pool_scheduler.py:87 强制）。

---

## 六、特性叠加设计

### 6.1 池化 × MTP

已有能力：#12853 支持 MTP layerwise prefill offload，`kvpool_store_skip_tokens` 机制处理 draft 相关 token。

已知限制（验证时标注，不修）：eagle + layerwise 下外部命中数被截断到整数块，trailing block 以 dirty data 加载（pool_scheduler.py TODO）。验证点：

| 验证点 | 通过口径 |
|---|---|
| MTP 开启时池化 save/load 正常 | 池内 key 数 >0，命中请求 KV 正确复用（输出与关闭池化时 token 一致） |
| MTP 接受率不受池化影响 | 开/关池化两组 acceptance rate 差异 <1% |
| Prefill TPS 不因 MTP 叠加异常回退 | 叠加组相比纯池化组波动在噪声范围内（<2%，3 轮均值） |

### 6.2 池化 × DCP

关键交互点：**PoolKey 的 KeyMetadata 携带 pcp_rank / dcp_rank**。DCP 是 decode 阶段并行（PCP 不支持），而 layerwise 池化加载发生在 prefill 阶段，两者在阶段上正交；风险在 key 一致性——prefill 实例（dcp_rank=0）保存的 key，decode 实例（dcp_size>1，各 rank dcp_rank 不同）查询时若按各自 dcp_rank 构造 key，会 miss 或错分片。

验证点：P/D 分离部署下，prefiller（无 DCP）save → decoder（DCP=2/4）load，命中数与输出正确性；若 key 不一致，修复方向是池化 key 的 dcp 维度归一化（保存侧写 canonical key，加载侧按自身 dcp 分片重映射），该项按验证结果决定是否立项。

### 6.3 池化 × LayerSplit

LayerSplit 的具体语义与和池化的交互面，当前项目资料中无足够信息。列为开放问题，Phase 3 前与需求方确认范围；确认后按 6.1/6.2 的格式补验证点。若确认为层切分类并行（与 KV 传输路径交叠小），预期以功能组合验证为主。

---

## 七、可观测性设计

### 7.1 已有基础（kv_metrics_observability 分支，待整合）

| 指标 | 类型 | 口径 |
|---|---|---|
| `vllm:kv_pool_load_duration_seconds` | Histogram（label: path） | layerwise 路径 = 任务提交 → 最后一个层传输完成（传输线程完成时刻，计算侧晚返回不拉长样本） |
| `vllm:kv_pool_load_keys_total` | Counter | layerwise 按层块传输计数（约 blocks×layers） |
| `vllm:kv_pool_load_failed_keys_total` | Counter | layerwise 失败即传输线程 raise，预期恒为 0 |
| `vllm:kv_pool_delayed_release_requests` | Gauge | 调度侧延迟释放窗口内请求数（最新快照语义） |

该分支 7 个提交（fa6cff82b 起，含 mypy/ruff 修复与 load duration 口径修正），基于旧 main（fa668c649），整合方式见实施计划第二节。

### 7.2 新增：not overlapped 读取时间指标

需求要求区分"总加载耗时"与"未被掩盖的耗时"。已有 load duration 是总耗时口径，not overlapped 需要新埋点：

**定义**：not overlapped = 计算侧就绪但 KV 未到货的阻塞时间，即 `wait_for_layer_load` 内部从进入等待到事件置位的净等待时长，按请求聚合（对层求和）。它与总耗时的关系：

```
overlap_ratio = 1 − not_overlapped / load_duration（请求粒度）
理想状态：not_overlapped → 0，overlap_ratio → 1
```

**埋点位置**：pool_worker.py `wait_for_layer_load` 的等待循环（`layer_load_finished_events[i].wait()` 前后计时，扣除已就绪直接返回的层）。

**暴露形态**：

| 新指标 | 类型 | 说明 |
|---|---|---|
| `vllm:kv_pool_load_not_overlapped_seconds` | Histogram（label: path=layerwise） | 请求粒度，P50/P90/P99 |
| `vllm:kv_pool_load_overlap_ratio` | Summary/Histogram | 请求粒度重叠率，用于大盘直接读数 |

**验收联动**：劣化 ≤5% 的达标证据链 = Prefill TPS 比值（性能结论）+ not overlapped 趋近 0（机理证据）+ 逐层等待分布（定位残余瓶颈层）。三者在验收报告中同时呈现。

### 7.3 极光大盘对接

指标经 vLLM Prometheus 端点暴露，极光平台采集后出看板。看板要素：load duration 与 not overlapped 的双曲线（P90）、overlap_ratio、delayed_release_requests、load_keys 速率。平台侧对接属于配套需求 AR20260820031213 的交付，本需求负责指标本身正确上报与语义定义。

---

## 八、验收基准测试设计

### 8.1 对照组配置

| 配置项 | 基线组（HBM PrefixCache） | 实验组（DRAM 池化） |
|---|---|---|
| 模型 / TP | DSV4 Flash，TP=8（51 服务器 8×A3） | 同左 |
| 本地 prefix caching | 开启 | 关闭（`--no-enable-prefix-caching`，排除本地命中干扰池化行为） |
| KV transfer | 无 | AscendStoreConnector，`kv_role=kv_both`，`backend=memcache`，`use_layerwise=true` |
| KV 存放 | 全部前缀可驻留 HBM（数据集规模受此约束，见 8.2） | DRAM 池（每卡 5G 起步，按数据集规模复核） |
| max-model-len / chunked prefill / gpu-memory-utilization / seed | 两组完全一致 | 同左 |
| 预热 | 跑一轮使 HBM prefix cache 满 | 先灌池（发一轮请求完成 save），确认池内 key 数符合设计值再测 |

### 8.2 负载构造

- 多前缀共享数据集：N 个共享前缀（长度对齐 cache_transfer_granularity，见 5.2），请求 = 前缀 + 少量新增后缀，平均输入 128K，设计命中率 90%。
- N 的上界受两组约束取小：基线组 HBM KV 预算 ≥ N × 每前缀 KV 字节；实验组 DRAM 池容量 ≥ 同值。每 token KV 字节在 Phase 0 实测后确定 N。
- 池化组灌池阶段单独计时，不计入压测窗口。

### 8.3 度量口径

- **Prefill TPS** = 压测窗口内总输入 token / 压测时长，请求 `max_tokens=1` 排除 decode 干扰；同时用引擎计数器（prompt tokens counter）交叉校验。
- 并发扫描两组各自独立进行（如 8/16/32/64 并发），各自取最优并发下的 TPS 进入对比。
- 每组每配置 ≥3 轮取均值，报告附轮间波动（波动 >2% 的配置点重测）。
- 判定：`TPS_pool / TPS_hbm ≥ 0.95`，附 not overlapped 与逐层等待分布作为机理佐证。

### 8.4 控制变量清单

1. 两组同一代码 commit（layerwise_pooling 分支验收 tag）、同一容器镜像、同一模型权重与配置模板，仅差第 8.1 表所列项。
2. 池化组确认无本地 prefix cache 命中（日志/指标核对），基线组确认 prefix cache 命中率符合设计值。
3. 压测客户端与服务器同网段，客户端不成为瓶颈（压测机 CPU/带宽余量 >50%）。
4. 温度 0、seed 固定、输出长度固定（max_tokens=1）。
5. 每轮之间清理状态：基线组重启实例清空 prefix cache 后重新预热；实验组清池重新灌池。

---

## 九、风险与开放问题

| # | 风险/问题 | 影响 | 应对 |
|---|---|---|---|
| R1 | DSV4 多 cache family × layerwise 兼容性未裁决（文档与代码不一致） | 若需适配，Phase 1 顺延 1–2 周 | Phase 0 第一优先级冒烟；适配方案已有代码基础（group_builders） |
| R2 | DRAM 带宽不足以逐层掩盖（W_i > C_i） | 物理上无法达标，5% 劣化不可达 | Phase 0 带宽-计算匹配表前置裁决；并发摊薄 + 预取加深；仍不足则如实报告物理边界 |
| R3 | 指标分支未合入 main | not overlapped 埋点的基础缺失 | 尽早 rebase 整合进 layerwise_pooling，指标 PR 与优化 PR 分开评审 |
| R4 | 特性叠加矩阵工作量超预期（尤其 DCP key 一致性） | 挤占验收窗口 | 叠加验证排 Phase 3，与优化迭代解耦；DCP 问题按验证结果决定是否单独立项 |
| R5 | LayerSplit 语义未明 | 无法设计验证点 | Phase 3 前与需求方确认范围 |
| R6 | 51 服务器资源排期冲突 | 环境不可用 | 提前锁定排期；165 服务器作为冒烟/UT 备用环境 |
| O1 | SSD 路径预研深度（memfabric SSD 配置是否现成） | 仅影响预研数据完整性 | Phase 4 前确认 SSD 介质可用性，不可用则以 DRAM 池分层模拟 |

---

## 十、设计要点回顾

- 实现路径锁定 **AscendStoreConnector + memcache backend + use_layerwise**，DRAM 池由 memcache_hybrid/memfabric_hybrid 承载；
- 达标机理是**逐层传输时间 ≤ 逐层计算时间**的流水线掩盖，Phase 0 用带宽-计算匹配表前置判定可行性；
- DSV4 Flash 的两个适配关键是**多 cache family 的 layerwise IO 裁决**（R1）与**粒度对齐的负载构造**（5.2）；
- 指标在 kv_metrics_observability 分支基础上**新增 not overlapped 口径**，与 load duration 形成"总耗时/未掩盖耗时"对照；
- 验收是严格受控的对照实验，prefix caching 行为、预热流程、并发扫描、轮次统计均有明确口径。
