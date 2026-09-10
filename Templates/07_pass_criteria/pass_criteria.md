## 6. E2E 通过判定标准

> **什么样的结果才算 E2E PASS**。三维证据链缺一不可；判定前先扫第 7 章（误区清单）。
> 蓝本：PR14465 第 1 组验收表（map_51/pr14465_dsv4_kvpool/record_final/02）。

### 6.1 三维证据链（核心标准）

E2E PASS = 以下三维**同时**成立，且每维都有留档证据：

#### ① 存（Put）
| 证据 | 通过条件 |
|------|----------|
| `master_allocated_bytes` | 干净基线 0 → **> 0**（如 41.31 MB） |
| `master_key_count` | 0 → **> 0**（如 61；不同内容多轮可至 244） |
| master.log batch 指标 | `PutStart:(Req=.., Item=..)` 非零速率 |
| （⚠️ `master_put_start_requests_total` 恒 0 是**正常**的，见 §7.1） |

#### ② 取（Get/ExistKey）
| 证据 | 通过条件 |
|------|----------|
| `External prefix cache hit rate` | **> 0**（首跑 37-42%，复验 9.0%，量级取决于负载） |
| `Prefix cache hit rate`（本地） | **= 0.0%**（证明确实禁用了本地缓存，命中来自外部 pool） |
| master.log | `Get:(Req=..)` / `ExistKey:(Req=..)` 非零速率 |

#### ③ 去重（不重复写入）
| 证据 | 通过条件 |
|------|----------|
| 相同内容重发 ×N | `key_count` / `allocated_bytes` **恒定不增长** |

三维关系：② 依赖 ①（先存才能取）；③ 证明 key 语义正确（同内容同 key）。
只做 ① 不做 ②③ = 半程验证，不能判 PASS。

#### ④ memcache 后端的等价三维口径（PR15367 实测，2026-09-01）

mooncake 的 master_* 指标在 memcache 场景换为 MetaService 证据（:8000/metrics 拉取）+ vllm 侧指标：

| 维度 | mooncake 口径 | memcache 等价口径（实测形态） |
|------|--------------|------------------------------|
| 存 | `master_allocated_bytes` / `master_key_count` 增长 | MetaService `alloc_successes=N stored_keys=N`（另一进程的独立计数）+ worker 日志 `load_gvas: keys=N valid_gvas=N` |
| 取 | `External prefix cache hit rate > 0` | vllm `/metrics` 的 **`external_prefix_cache_hits_total`（counter，精确断言用这个而非滑动窗口 rate）** + scheduler 日志 `hit_check: hit_tokens=N` |
| 去重 | 重发后 key_count 恒定 | 重发后 `stored_keys` 不再增长（或按 key 语义判等） |

注意：memcache 1.2.0 无 put 类命名指标（memcache_*_alloc_* 是分配口径不是写入口径），存维以 MetaService `stored_keys` + `load_gvas` 日志双源交叉（见 §6.6 跨进程证人原则）。

### 6.2 服务健康（前置条件，不算 PASS 的一部分但必须确认）

| 项 | 标准 |
|----|------|
| master 状态 | `role=leader, state=serving, service_ready=true` |
| `master_active_clients` | = DP × TP（如 16），全部 worker 注册 |
| connector 创建 | server.log 中 N 个 `AscendStoreConnector`（N=DP×TP） |
| device type | `device_type=npu`（NPUPlatform 插件加载成功） |
| 推理本身 | 请求正常返回、内容正常（池化不能破坏推理） |

### 6.3 验收表模板（每个场景一张，随证据归档）

```markdown
| 维度 | 结果 | 证据 |
|------|------|------|
| 服务启动 | ✅/❌ | connector 数、master serving、active_clients |
| 本地 prefix cache 禁用 | ✅ | Prefix cache hit rate: 0.0% |
| KV pool 存入 | ✅ | allocated_bytes=xxx, key_count=xxx |
| KV pool 取 | ✅ | External prefix cache hit rate=xx% |
| 去重 | ✅ | 重发后 keys 恒定 |
| 推理正确性 | ✅ | 响应内容 spot check |
```

归档位置：`map_XX/<pr_task>/record_final/`（结论 + evidence/ 原始文本快照）。

### 6.4 FAIL / BLOCKED / EXPECTED_FAIL 的区分

| 判定 | 含义 | 典型例子 |
|------|------|----------|
| **PASS** | 三维证据链齐全 | 第 1/3 组 |
| **FAIL** | PR 代码缺陷导致 | 启动崩溃且根因在 PR 改动内 |
| **EXPECTED_FAIL / BLOCKED** | 已知环境/上游硬限，**非 PR 缺陷** | DSV4 NSA multi-spec 硬限（§8.1）、master 未配置 |

判定 EXPECTED_FAIL 必须给出根因代码位置 + 对照证据（同代码 non-layerwise PASS /
单 attention 模型 layerwise PASS），否则降级为 FAIL 处理。

### 6.5 干净基线原则

- 每组验证前**全清重启**（kill vllm + mooncake，counters 归零），否则增长无法归因
- 基线采样留档（请求前的 counters 快照）
- PR 分支代码更新后须**复验**：干净基线重跑核心组，对比关键值（如 keys=61 / 41.31MB 两次复验一致 → 行为等价结论）
- 版本快照（env.txt：commit + pip version + location）随每组留档，防"验证的到底是哪个 commit"争议

### 6.6 虚假通过防范（vacuous pass 防范，PR15367 沉淀）⚠️ 判据设计层，先于一切 PASS 判定

**问题**：e2e"通过"可能是假的——三类典型形态：**A 服务根本没起来**；**B 起来了但 KV 没入池**；**C 入池了但静默不命中**（无报错、全 miss，验证却"通过"）。C 在 key 格式重构/迁移类 PR 中最阴险：key 静默漂移 = 永远 miss + 零报错。

**总原则一句话**：判据全部是"正计数 + 跨进程证人"，静默失效的任何一类都会让对应计数归零、判据 FAIL，而不是侥幸通过；阴性对照排除"验证方法本身失效"。

**① 正计数判据（不是"无错误"判据）**

- PASS 条件必须写成**计数 > 0** 的形态（`hit_tokens=3328`、`valid_gvas=26`、`stored_keys=28`），不能写成"无 Traceback / 无 Segfault"
- 无错误判据对 C 类静默失效**天然失明**：全 miss 也是无错误

**② 跨进程证人原则：每类失效模式配一个独立于被测代码的证人**

| 失效模式 | 证人（被测代码之外） |
|---|---|
| A 服务没起 | 每请求带回真实 `prompt_tokens=N completion=M`——死服务给不出带 token 计数的 completion |
| B 没入池 | 存储后端**自己进程**的计数（mooncake `master_key_count` / memcache MetaService `stored_keys`）——worker 代码谎报存入不可能让服务端计数增长 |
| C 静默不命中 | vllm `/metrics` 的 `external_prefix_cache_hits_total`（框架聚合，与 scheduler 日志的 `hit_tokens` 跨层对拍） |

**③ 双向断言（证明测试对目标失效是敏感的，不是恒通过的）**

- 不仅断言"命中 > 0"，还要断言"miss 被真实记录"：MetaService `query_not_found=28`（首发请求 miss 数）
- 若 key 静默漂移：会计变成"恒 miss 无报错"→ `hit_tokens=0` → 判据 FAIL。**query_not_found 的存在证明 miss 通道也在被观测**，测试有区分度
- 类似：`lease_fail=0` 与 `valid_gvas>0` 并读，排除"租约失败导致假阳性"

**④ 跨层算术一致（多源对拍）**

同一物理量在不同会计路径应可对账：`load_gvas keys=26` × 128 token/块 = `hit_tokens=3328` = `/metrics external_prefix_cache_hits_total=3328.0`；`load_gvas` 与 `hit_check` 应是**同一请求 ID**（`req=chatcmpl-...`）。编造或读错文件很难做到三处算术自洽。

**⑤ 阴性对照（negative control）——排除验证方法本身失效**

- 设计一个"该指标必然为 0"的场景做对照组：如 mooncake 非 layerwise 场景（S3）之于 memcache layerwise 场景（S2）
- 用**同一组 grep/判定命令**跑两个场景：S2 `load_gvas:` 行数 >0 且 S3 = 0 行 → 证明 grep 模式有效、读的是本轮日志、能区分路径
- 若验证方法本身坏了（grep 错文件/模式恒不匹配/读旧缓存日志），S2 也应得 0。**S2>0 且 S3=0 的不对称才是方法有效的证明**——单看 S3=0 毫无意义（恒真），单看 S2>0 不能排除巧合

**⑥ 数值敏感性证明读的是活数据（不是死文件/常量）**

- 两轮验证的数值应**不同**（时序浮动本身是活数据证据）：hit_tokens 3456→3328、请求 ID 不同
- 日志行号漂移（`pool_scheduler.py:389`→`:388`）与两棵源码树的 rebase 偏移吻合——佐证两轮跑的各自声称的代码版本（配 §6.5 版本快照：容器内 `git rev-parse HEAD`）
- 若两轮输出完全一致，反而要怀疑读了归档的旧日志

### 6.7 重构类 PR 的双轮夹逼复验（behavior-preserving 验证）

重构（宣称行为不变）的 PR，"行为等价"不能靠单轮 PASS 推断，**必须双轮夹逼**：

- **轮 1（基线）**：重构前的 head 跑完整场景集，全部 PASS 才作为有效基线
- **轮 2（复测）**：重构后（含 rebase / 检视返工 / bugfix）的 head 再跑**同构**场景集
- 两轮同构 PASS = 重构未引入行为漂移的实证（不是推断）；中间夹着的所有高风险变更被夹逼验证
- 数值浮动按 §6.6⑥ 归因（滑动窗口命中块数、请求分布），判据是 >0 类正计数，非定值对拍
- 报告双轮并列（`e2e-report-*.md` ×2），互相引用；PR 描述 Test plan 写"re-verified on <head> + validated earlier on <基线> (two rounds, same results)"

### 6.8 判定流程图

```
请求发出且正常返回？
├─ 否 → 查启动/推理问题（§4.4），不算池化结论
└─ 是 → 存：allocated/key_count 增长？
        ├─ 否 → 查误区：skip_save（prompt 长度/对齐）？prefix-caching 没禁？
        └─ 是 → 取：External hit rate > 0？
                ├─ 否 → 查 master Get/ExistKey 速率、lookup ZMQ 链路
                └─ 是 → 去重：重发 keys 恒定？
                        ├─ 否 → key 生成语义问题，FAIL
                        └─ 是 → E2E PASS，归档三维证据
```

（判定前自检 §6.6：你的判据是正计数吗？有跨进程证人吗？阴性对照在哪？——三问答不上来的 PASS 是可疑的。）

---

## 7. 判定误区清单

> 判 PASS/FAIL 前必扫。来源：PR14465/PR14912 实测踩坑（详见各 record_final / HANDOVER）。

### 7.1 `master_put_start_requests_total = 0` ≠ 池化未生效 ⚠️ 最高频误区

- 该 Prometheus counter 只计**单条** Put API
- mooncake 后端走 **batch API**（`batch_put_from_multi_buffers`），其统计只在 master.log
  `Batch Requests` 字段以每秒速率呈现
- **正确判定**：看 `master_allocated_bytes` / `master_key_count` / master.log `PutStart:(Req=..)`

### 7.2 `External prefix cache hit rate` 非实时

- 是**最近 1000 请求的滑动窗口平均**，请求结束后不自更新
- 数值量级与负载强相关（首跑 37-42%，干净复验 9.0%，都算 PASS）——只判 > 0，不设阈值

### 7.3 短 prompt "全 0" 是 skip_save，不是 bug

- `num_tokens_to_save < chunk_boundary` → `skip_save=True` → 整请求不存入
- granularity 因模型而异（DSV4 4096 / PR14912 场景 128）
- 详见 §5.1；先算门槛再发请求

### 7.4 本地 prefix cache 不禁用 → 永远看不到池化效果

- 本地命中抑制 connector 外部存取
- 必须确认 server.log `Prefix cache hit rate: 0.0%`（这本身是一条证据）

### 7.5 Qwen3 对照实验的教训：别急着怪配置

- "DSV4 全 0"最初怀疑 prefix-caching/权重，Qwen3-8B 对照（相同配置能存取）一次排除三个假设
- **怀疑配置时先做控制变量对照**，再下结论

### 7.6 主 vllm 活着 ≠ 服务可用

- EngineCore 崩溃后 Worker 卡 do_poll、8004 不监听、status.txt 不生成，但主进程存活
- 判启动失败要看 server.log 崩溃堆栈 + 端口监听，不能只看进程

### 7.7 `pkill -f mooncake_master` 杀不死

- 51 实测 PID 残留（19863/19865）
- 一律显式 PID `kill -9`；重启前确认端口释放

### 7.8 指标聚合的 TP rank 放大（PR14912 A1）

- 框架聚合**全部 TP rank** 的 worker stats → `load_count` / `load_keys_total` 放大 TP 倍
- avg / P90 不受影响；看板速率类聚合需 ÷TP

### 7.9 跨 path 指标不可直接对比（PR14912 m3）

- layerwise 的 `load_keys` = 跨层 block 数（blocks × layers），比 sync 的 "rank chunk 数" 放大 64 倍
- sync / layerwise / async 三条路径的计数**口径不同**，各自纵向对比，不横向比

### 7.10 master 重启不影响已存数据（失败路径构造误区）

- mooncake 同节点池把 KV 存 worker 本地宿主段，master 重启只清元数据不清数据 → get 继续命中
- `prefer_alloc_in_same_node=false` 也一样（数据段仍落本地）→ **单节点无法构造 load 失败**（见 §8.3）

### 7.11 HBM 到 baseline 就停 = spec 校验阶段崩溃

- 加载中每 die ~18GB baseline，完成 ~50GB；停在中间说明被 KV cache spec 校验挡下（如 layerwise 硬限）
- 不是权重/磁盘问题

### 7.12 版本验证要用 pip location 而非记忆

- `pip show vllm vllm-ascend` 看 **Editable project location** 是否指向预期源码目录
- editable 残留（旧 .pth / dist-info / egg-info 在挂载卷上）会让"以为切了分支其实跑的旧代码"——卸载规范见 env_install/4

---

## 8. 已知硬限与口径限制

> 这些不是 PR 缺陷，是 vllm-ascend / 环境的已知边界。判定 EXPECTED_FAIL / 设计验证方案前必读。
> **新发现的硬限发现后必须回写本章**（与 env_install 回写规则一致）。

### 8.1 DSV4 NSA 多 cache spec × layerwise = 硬限（最重要）

**位置**：vllm-ascend `distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py`
（`build_layerwise_reuse_layout`，PR14465 时为 :215-228，后续版本行号可能漂移）

**规则**：每个物理层若有多个 cache spec，必须**恰好 1 个 main + 1 个 `.indexer.k_cache`**（SFA 模式），
否则 `raise ValueError`（有 UT 明确断言拒绝：`test_layerwise_cache_layout.py::test_ambiguous_multi_spec_layer_is_rejected`）。

**触发**：DSV4-Flash（NSA 架构）layer 2 起 self_attn 含 **5 个 cache spec**：

| spec | 分类 |
|------|------|
| `compressor.state_cache` | main |
| `indexer.k_cache` | indexer |
| `indexer.compressor.state_cache` | main |
| `swa_cache` | main |
| `attn` | main |

→ main=4 ≠ 1 → ValueError → EngineCore 崩（错误信息 `Physical layer 2 with multiple cache specs...`）。

**边界**：
- `--disable-hybrid-kv-cache-manager` **不能绕过**（hybrid_kv_cache_manager 是 vllm 主干的调度层，与 vllm-ascend 的 layerwise 布局层独立）
- non-layerwise **不受影响**（`cache_coordinator=None` 走直接路径，不调 `build_layerwise_reuse_layout`）——同权重 non-layerwise PASS 是标准对照证据
- **绕行方案（v2 修正，拆两层）**：
  - **验证 layerwise 传输**：必须用 **MLA/SFA 模型**（官方支持矩阵，示例 DeepSeek-V2-Lite）——full attention（`attention_v1`）**未集成 layerwise wait/save**，用它验证得到的是"布局不崩"而非"layerwise 传输工作"
  - **仅验证布局不崩 / 与 NSA 硬限解耦**：才用单 attention 模型（Qwen3 / Llama-3 / Qwen3-32B-pdmix 等），且结论只能写"**布局通过**"，不得写"layerwise 验证通过"

**代码实证（v2）**：`wait_for_kv_layer_from_connector` / `maybe_save_kv_layer_to_connector` 的调用面——
mla_v1(:1901/:2019)、sfa_v1(:1550/:1590/:1698)、dsa_v1(:1250/:1260/:1284)、dsa_cp(:1434/:1513) 均有调用；
**attention_v1 为 0 处**（官方 Supported Models 段亦明文，但其"CP 变体均未集成"表述滞后于代码——dsa_cp 已有调用，引用模型支持面以代码 grep 为准）。

**PR15307 Q2 最终裁决（2026-08-30 实测，E2E 腿 2）**：
- **结论：触发（维持硬限成立）**。DSV4 + layerwise 在 worker 层 `get_kv_cache_spec → _get_layerwise_kv_cache_memory_info → build_layerwise_reuse_layout`（vllm-ascend refactor_layerwise_B，commit `b3a141331`）抛出同一 `Physical layer 2 with multiple cache specs` ValueError，5 spec 集合与顺序与 PR14465 堆栈**逐项一致** → 证实 PR15307 GVA 线程收敛重构**未改动** spec 收集/分组逻辑，该硬限是**上游存量 feature gap，非 PR15307 缺陷**。
- **裁级**：DSV4 + layerwise = **EXPECTED_FAIL**（上游缺口），后续方向：升级 `build_layerwise_reuse_layout` 支持 NSA 多 main spec（需区分 attn/swa_cache/compressor.state_cache/indexer.compressor.state_cache 的 main 归属，或将 NSA 各子 cache 归并为一个聚合 spec）。
- 留档：`map_51/pr15307_gva_threads/evidence/leg2_dsv4_layerwise/`（error_stack.txt 完整调用链）。

**相关调用点**（评审 layerwise 改动时关注）：`pool_worker.py` / `pool_scheduler.py` / `worker.py` 各 1 处。

### 8.2 存入时机与 token 门槛（语义限制）

- non-layerwise 的 KV 存入发生在**请求完成后**（非流式中途）
- `num_tokens_to_save < chunk_boundary` 且无 partial block → `skip_save=True`（详见 §5.1）
- 这是**语义正确**的设计，不是丢失；验证用例必须跨过门槛

### 8.3 单节点无法构造 load 失败路径（PR14912 结论）

- mooncake 同节点池把 KV 存 **worker 本地宿主段**，master 重启只清元数据不清这段数据 → get 继续命中
- 改 `prefer_alloc_in_same_node=false` 无效：数据段仍落本地持久化段
- 触发真实 load 失败需要：**多节点部署**（远端节点数据丢失）或物理删 worker 本地数据段（共享服务器不可安全构造）
- **处置**：失败观测路径由单测覆盖（`_record_load_finished(..., num_failed_keys=1)` 断言），交付表述为"1 指标仅注册未真实触发，有单测保护"

### 8.4 指标口径差异（跨 path 不可比，PR14912）

| 口径 | 含义 | 注意 |
|------|------|------|
| sync `load_keys` | rank chunk 数 | 基准 |
| layerwise `load_keys` | blocks × layers | 放大 ~64 倍，**不可与 sync 直接比** |
| worker stats 聚合 | 全 TP rank 求和 | `load_count`/`load_keys_total` 放大 TP 倍，速率类看板需 ÷TP |
| `load_failed_keys_total` | 失败键 | 见 §8.3，单节点恒 0 属预期 |
| layerwise 计时 | 传输线程 set 事件时刻起算 | C 轮修复后语义（`_TimedLayerLoadEvent`） |

### 8.5 模型侧约束速查（51 服务器实测）

| 模型 | 池化验证适用性 |
|------|----------------|
| DeepSeek-V4-Flash(-DSpark-w4a8) | non-layerwise ✅ / layerwise ❌（§8.1 硬限）；granularity=4096，prompt ≥ 8192 |
| Qwen3-32B(-pdmix) | 对照/control 首选；PR14912 三路径全通（granularity=128 场景 prompt ≥ 5042） |
| Qwen3-8B | 标准注意力，快速对照实验用 |
| 单 attention 架构（Llama-3 类） | ⚠️ 仅"layerwise 布局验证 / 与 NSA 硬限解耦"，结论只能写布局通过（`attention_v1` 未集成 layerwise wait/save）；**layerwise 传输验证必须用 MLA/SFA 模型**（DeepSeek-V2-Lite 等，见 §8.1） |

### 8.6 环境侧硬约束（与 AGENTS.md §6 交叉）

| 约束 | 说明 |
|------|------|
| 50051 端口 | 宿主机 OceanStor DTMA 占用（51、112 均有），mooncake 一律 50088 |
| vllm ↔ vllm-ascend 版本配对 | 见 AGENTS.md §2 Step5 配对表（配错启动即崩） |
| 共享服务器 NPU | 先 npu-smi 查空闲，只申明空闲 chip |
| `master_put_start_requests_total` | 单条 API 计数，batch 后端恒 0，非故障（§7.1） |