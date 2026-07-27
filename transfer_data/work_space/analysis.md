# PD 分离与池化场景看护体系分析

## 一、需求理解

当前要解决的问题不是简单地补几个 e2e 用例，而是为 vllm-ascend 中两类 KV transfer 场景建立一套更完整、可持续的看护体系：

- **PD 分离**：`kv_transfer/kv_p2p/`，核心是 prefill 节点算完 KV cache 后，通过 P2P 直接传给 decode 节点。
- **池化**：`kv_transfer/kv_pool/`，核心是把 KV cache 存入池子，后续请求通过 key 查找并加载复用。

需要考虑的看护维度包括但不限于：

- 正确性
- 精度
- 性能
- 稳定性
- 资源释放
- 异常恢复
- 配置有效性
- CI 调度有效性
- 指标趋势与回归发现

因此，完整方案不应只局限在 UT 和 e2e，而应该形成从静态检查、单元测试、组件测试、PR e2e、nightly、weekly、长稳故障注入到指标看板的分层看护体系。

---

## 二、PD 分离与池化的看护目标差异

### 2.1 PD 分离需要看护什么

PD 分离的本质是：

> Prefill 节点计算 KV cache，Decode 节点通过 P2P 获取 KV cache，然后继续 decode。

核心风险集中在“传输”和“路由”。

| 维度 | 需要看护的问题 |
|---|---|
| KV 传输正确性 | Decode 节点拿到的 KV 是否完整、顺序正确、layer/block/request 是否匹配 |
| 请求路由正确性 | Proxy 是否正确把 prefill/decode 分发到对应节点 |
| 输出一致性 | PD 模式输出是否和非 PD 基线一致或在可接受范围内 |
| 性能收益 | TTFT 是否下降，P/D 是否真正并行，TPOT/吞吐是否劣化 |
| 资源释放 | Prefill 节点传完 KV 后是否正确释放 block，是否泄漏 |
| 多 connector 覆盖 | `MooncakeConnectorV1`、`MooncakeLayerwiseConnector`、`MooncakeHybridConnector` 都应有看护 |
| 异常恢复 | P/D 节点慢、连接失败、请求取消、超时释放是否正确 |

### 2.2 池化需要看护什么

池化的本质是：

> KV cache 先存入池子，后续请求按 key 查找，命中后加载回来复用。

核心风险集中在“存储、查找、加载、复用”。

| 维度 | 需要看护的问题 |
|---|---|
| Save/Load 正确性 | 存进去的 KV 再加载回来是否一致 |
| Prefix 命中正确性 | 相同 prefix 是否命中，不同 prefix 是否不会误命中 |
| 输出一致性 | 命中池化后的输出是否和冷启动计算一致 |
| 性能收益 | 命中后 TTFT 是否下降，吞吐是否提升 |
| 命中率 | 预期能命中的请求是否真的命中 |
| eviction/容量 | 池子满后是否正确淘汰，不误删、不脏读 |
| 后端依赖 | Mooncake Store / Memcache / CPU DRAM / LMCache / UCM 是否正常工作 |
| 多 connector 覆盖 | `AscendStoreConnector`、`SimpleCPUOffloadConnector`、`RecomputeCPUOffloadConnector`、`LMCacheAscendConnector`、`UCMConnector` 分别看护 |

---

## 三、推荐的整体看护分层

建议把看护体系拆成 7 层，而不是只依赖 UT/e2e。

| 层级 | 类型 | 主要作用 | 适合看护什么 |
|---|---|---|---|
| L0 | 静态配置检查 | 防止 YAML/CI 配错 | connector 名、kv_role、平台调度、文件路径存在 |
| L1 | UT | 单模块逻辑正确性 | metadata、block 映射、hash、scheduler、状态机 |
| L2 | 组件/集成测试 | 不拉完整大模型，但跑 connector 核心流程 | save/load、send/recv、mock backend、本地 loopback |
| L3 | 小模型 PR e2e | 快速防止主路径坏掉 | 小模型输出一致性、无崩溃、connector 生效 |
| L4 | nightly e2e | 每日真实服务看护 | A2/A3 平台、HTTP server、多节点、性能冒烟 |
| L5 | weekly benchmark | 更重的精度/性能基准 | 大模型、长序列、多节点、aisbench |
| L6 | 长稳/故障注入/线上观测 | 发现慢性问题 | 内存泄漏、连接抖动、命中率下降、P99 抖动 |

---

## 四、L0：静态配置检查

### 4.1 为什么需要

很多 e2e 覆盖缺口不是测试代码没有写，而是：

- YAML 写了但没有被 `nightly_config.yaml` 或 `weekly_config.yaml` 调度。
- 配置文件路径写错。
- connector 名拼错。
- 平台 section 放错。
- `kv_role` 和 connector 不匹配。
- PD 分离配置中 P/D 分组不完整。
- 池化 connector 配了但没有启动对应后端服务。

这类问题用完整 e2e 才发现成本很高，但用配置 lint 可以在 PR 阶段快速发现。

### 4.2 建议检查规则

| 规则 | 说明 |
|---|---|
| 调度文件引用的 YAML 必须存在 | 防止 nightly/weekly matrix 指向不存在的配置 |
| connector 名必须在注册表中 | 防止 `kv_connector` 拼写错误 |
| PD 分离必须有 P/D 配置 | internal_dp 检查 `disaggregated_prefill`，external_dp 检查 `routing.type` 和 groups |
| `kv_role` 必须合法 | `kv_producer`、`kv_consumer`、`kv_both` 按场景约束 |
| `MultiConnector` 子 connector 必须合法 | 防止组合结构错误 |
| `RecomputeCPUOffloadConnector` 必须配 recompute scheduler | 检查 `recompute_scheduler_enable: true` |
| A2/A3/310P section 不混用 | 防止 A3 YAML 被放入 A2 section，或反之 |
| layerwise/hybrid 配置必须匹配 connector | 防止 connector 和 extra config 不一致 |
| 外部依赖 connector 必须声明服务 | 如 AscendStore、LMCache、UCM 等 |

### 4.3 建议落地方式

可以新增一个轻量 pytest 或脚本，例如：

- `tests/e2e/config/test_kv_transfer_config_lint.py`
- 或 `.github/workflows/scripts/check_kv_transfer_configs.py`

放在 PR 级别跑，目标是快速失败、错误信息明确。

---

## 五、L1：UT 看护

UT 不应该只验证函数能运行，而应重点锁住 KV transfer 内部不变量。

### 5.1 PD 分离 UT 建议

| 模块 | UT 关注点 |
|---|---|
| connector metadata | request id、block id、layer id、slot mapping 是否正确 |
| P/D 角色状态机 | `kv_producer` / `kv_consumer` 的初始化、收发、结束状态是否正确 |
| block 生命周期 | send 后 delayed free、DONE_RECVING、timeout free 是否正确 |
| V1 pull 模式 | consumer 拉取 KV 时 src/dst/length list 是否正确 |
| layerwise push 模式 | 每层 send/recv 顺序是否正确，partial layer 是否安全 |
| hybrid 模式 | MLA / full attention 混合 block size 下 mapping 是否正确 |
| 异常路径 | recv 失败、重复 DONE、请求取消、超时是否安全 |
| 并发路径 | 多 request 并发时 request/block 映射不串扰 |

### 5.2 池化 UT 建议

| 模块 | UT 关注点 |
|---|---|
| block hash | 相同 prefix hash 一致，不同 prefix 不误命中 |
| lookup | 命中、部分命中、miss、stale entry 处理正确 |
| save/load metadata | KV 的 shape、dtype、layer、block 顺序正确 |
| eviction | 容量满、LRU/引用计数、被占用 block 不应被淘汰 |
| SimpleCPUOffload | offload 后释放 HBM，再 load 回来状态一致 |
| RecomputeCPUOffload | 抢占、重计算、load 回来状态一致 |
| AscendStore | 后端 key 生成、put/get、连接参数处理正确 |
| backend error | 后端不可用、超时、返回空数据时 fallback 或报错明确 |

### 5.3 UT 的价值

UT 的核心价值是低成本锁住内部不变量。e2e 发现失败后往往难以判断问题来自 routing、metadata、block mapping、后端还是调度，而 UT 可以在更细粒度上提前定位问题。

---

## 六、L2：组件/集成测试

这是 UT 和 e2e 之间非常关键但容易缺失的一层。

它不一定拉完整大模型，但应该真实跑 connector 核心链路。

### 6.1 PD 分离组件测试

目标：

> 用 mock KV tensor 或小型 fake worker，验证 producer → consumer 的 KV 传输链路。

可以覆盖：

- producer 注册 KV block。
- consumer 请求读取 KV。
- Mooncake transfer engine mock 或本地 loopback。
- KV tensor checksum 一致。
- 多 request 并发传输。
- 请求取消后资源释放。
- layerwise 每层 push 顺序。
- hybrid 混合 block size mapping。

建议断言：

- 传输前后 KV checksum 一致。
- request id 和 block id 没有串扰。
- 传输完成后 producer 资源释放。
- 失败路径不会遗留脏状态。

### 6.2 池化组件测试

目标：

> 用 mock KV tensor 验证 save → lookup → load。

可以覆盖：

- 第一次请求 save KV。
- 第二次请求 lookup 命中。
- load 回来的 KV checksum 一致。
- 部分 prefix 命中。
- 池满 eviction。
- backend timeout/failure。
- CPU offload 后 load。
- recompute/offload 恢复路径。

建议断言：

- 命中和 miss 符合预期。
- 不同 prefix 不误命中。
- load 回来的 KV 与 save 时一致。
- eviction 后不会读到过期 KV。
- backend 异常有明确 fallback 或错误。

### 6.3 为什么这一层重要

组件测试比 e2e 便宜很多，但比 UT 更接近真实链路。对于 PD 分离和池化这种涉及 KV 元数据、调度、传输、后端状态的能力，这一层非常有价值。

---

## 七、L3：PR e2e 看护

PR e2e 的目标应该是：

> 合入前快速发现主路径是否被直接破坏。

它不适合跑大模型、大多节点、长性能测试，应保持小、快、确定性强。

### 7.1 PD 分离 PR e2e

| Connector | PR 看护建议 | 原因 |
|---|---|---|
| `MooncakeConnectorV1` | 应保留 | 主路径，当前已有 1P1D PD 分离覆盖 |
| `MooncakeLayerwiseConnector` | 建议补一个轻量 1P1D | 当前 PR 无覆盖，layerwise 容易被改坏 |
| `MooncakeHybridConnector` | 不建议 PR | DeepSeek-V4 太重，更适合 weekly/nightly 评估 |

建议断言不只看返回非空，还可以增加：

- temperature=0 下输出与非 PD baseline 一致或在可接受范围内。
- 服务健康检查通过。
- 日志或 metrics 确认使用目标 connector。
- 日志中无 transfer error、timeout、异常 force free。
- P/D 角色都成功启动。
- 如果 metrics 可用，确认发生过 KV transfer。

### 7.2 池化 PR e2e

| Connector | PR 看护建议 | 原因 |
|---|---|---|
| `SimpleCPUOffloadConnector` | 已有，应保留并增强 | 无外部依赖，适合 PR |
| `AscendStoreConnector` | 可考虑轻量 PR | 依赖 Mooncake，取决于 CI 环境稳定性 |
| `RecomputeCPUOffloadConnector` | 可以先补专项 e2e，再决定是否 PR | 需要触发抢占/重计算，复杂度较高 |
| `LMCacheAscendConnector` | 不建议 PR | 外部依赖较重 |
| `UCMConnector` | 不建议 PR | 外部服务依赖较重 |

建议 PR 池化 e2e 覆盖：

- 第一次请求 cold run。
- 第二次请求相同 prefix，预期命中。
- warm 输出和 cold 输出一致。
- 重置或清空 HBM prefix cache 后，从池化后端加载。
- 日志或 metrics 确认 cache hit/load 发生。
- PR 中可以只做性能 sanity，不强制严格性能阈值。

---

## 八、L4：nightly e2e 看护

nightly 的目标是：

> 在真实 server、真实硬件、真实多节点环境中，每天确认主路径没坏。

### 8.1 PD 分离 nightly

当前状态：

- `MooncakeConnectorV1` nightly 已有覆盖。
- `MooncakeLayerwiseConnector` nightly 无覆盖。
- `MooncakeHybridConnector` nightly 无覆盖。
- weekly 中已有 layerwise 和 hybrid 覆盖。

建议：

| 优先级 | 动作 |
|---|---|
| P0 | 将 weekly 中较轻的 `GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml` 纳入 nightly |
| P1 | 保持 V1 的 2 节点/4 节点覆盖 |
| P2 | Hybrid 暂不强行进 nightly，除非 DeepSeek-V4 Flash 多节点稳定且资源允许 |

nightly PD 分离应至少记录：

- TTFT。
- TPOT。
- throughput。
- P/D 节点显存峰值。
- KV transfer latency。
- transfer bytes。
- transfer error count。
- request success rate。
- P99 latency。
- proxy routing error count。

### 8.2 池化 nightly

当前状态：

- `AscendStoreConnector` 有非专项覆盖，主要随 Qwen3-30B 精度测试附带覆盖。
- `SimpleCPUOffloadConnector` PR 已有 e2e，但 nightly 无。
- `RecomputeCPUOffloadConnector` 无 e2e。
- `LMCacheAscendConnector` / `UCMConnector` 无覆盖，且依赖外部环境。

建议：

| 优先级 | 动作 |
|---|---|
| P0 | 把 `tests/e2e/pull_request/one_card/test_simple_cpu_offload.py` 纳入 A2 nightly |
| P1 | 新增 `RecomputeCPUOffloadConnector` e2e 后纳入 nightly |
| P1 | 对 `AscendStoreConnector` 增加明确 cache hit 指标，不只依赖 Qwen3-30B 精度测试附带覆盖 |
| P2 | LMCache/UCM 等外部依赖就绪后再补 |

nightly 池化应至少记录：

- cache hit count。
- cache miss count。
- hit ratio。
- save latency。
- load latency。
- backend get/put latency。
- cold TTFT。
- warm TTFT。
- HBM 使用量变化。
- CPU DRAM 使用量。
- eviction count。
- backend error count。

---

## 九、L5：weekly benchmark 看护

weekly 适合跑重模型、长序列、多节点、大 benchmark。它不适合替代 PR/nightly，而是用来发现更深层的精度和性能退化。

### 9.1 精度看护分档

| 档位 | 方法 | 适用场景 |
|---|---|---|
| 基础输出一致性 | temperature=0，对比固定 prompt 输出 | PR / nightly |
| 小数据集 accuracy | 小规模 aisbench / lm_eval | nightly |
| 完整 benchmark | 大模型、大数据集、长序列 | weekly |

### 9.2 PD 分离精度 baseline

建议对比：

- 非 PD 单体模式输出。
- PD V1 输出。
- PD Layerwise 输出。
- PD Hybrid 输出。

关注：

- greedy 输出是否一致。
- accuracy 是否低于 baseline。
- 长序列场景是否退化。
- 多 batch/并发下是否稳定。
- layerwise push 模式是否引入顺序或 partial KV 问题。

### 9.3 池化精度 baseline

建议对比：

- cold run，不命中池化。
- warm run，命中池化。
- offload/load 后恢复。
- recompute 恢复后输出。
- eviction 后重新计算。

关注：

- warm 输出是否和 cold 一致。
- 命中 prefix 后后续 token 是否一致。
- 部分命中场景是否正确。
- eviction 后是否不会读到旧 KV。
- recompute/offload 恢复后输出是否正确。

### 9.4 PD 分离性能指标

| 指标 | 含义 |
|---|---|
| TTFT | PD 分离最关键指标，预期下降 |
| TPOT | decode 阶段是否受影响 |
| E2E latency | 用户整体延迟 |
| throughput | 并发吞吐 |
| P/D 利用率 | prefill/decode 是否真正解耦 |
| transfer latency | KV 传输耗时 |
| proxy overhead | proxy 是否成为瓶颈 |
| P99 latency | 高分位稳定性 |

### 9.5 池化性能指标

| 指标 | 含义 |
|---|---|
| cold TTFT | 第一次请求，不命中池化 |
| warm TTFT | 命中池化后的请求 |
| hit ratio | 池化是否真的生效 |
| save/load latency | 存取开销 |
| backend latency | Mooncake/Memcache/CPU/UCM 后端瓶颈 |
| HBM savings | 是否降低显存压力 |
| recompute/offload overhead | offload 是否得不偿失 |
| eviction count | 淘汰是否频繁 |

---

## 十、L6：长稳与故障注入

KV transfer 场景最容易出现的问题不一定是功能直接失败，而是慢性问题，例如：

- 跑久后显存或 CPU 内存泄漏。
- 请求取消后 block 没释放。
- 后端偶发超时导致脏状态。
- 高并发时 request/block mapping 错乱。
- P/D 节点速度不均导致队列堆积。
- P99 延迟抖动。
- cache hit ratio 随时间异常下降。

### 10.1 PD 分离故障场景

建议覆盖：

- Prefill 节点慢。
- Decode 节点慢。
- Proxy 短暂不可用。
- P/D 连接失败。
- request cancel。
- streaming 中断。
- KV transfer 超时。
- P 节点传完后 D 没确认。
- 多个请求并发乱序返回。
- P/D 节点资源不均衡。

### 10.2 池化故障场景

建议覆盖：

- backend get 超时。
- backend put 失败。
- cache entry 不存在。
- 部分 layer save 成功、部分失败。
- eviction 与 load 并发发生。
- CPU offload 内存不足。
- Memcache/Mooncake Store 重启。
- 重复 key 写入。
- 脏 cache / stale cache。
- 后端恢复后是否能继续服务。

### 10.3 调度建议

这类测试不建议放 PR，也不建议全部每天跑。更合适的是：

- weekly 跑一部分。
- 专项 pipeline 跑长稳。
- 重大改动前手工触发。
- 对 KV transfer 相关目录改动时自动触发增强测试。

---

## 十一、指标产物与趋势看板

测试 pass/fail 只能说明当前是否失败，不能很好发现性能逐步退化。PD 分离和池化都应该把关键指标保存成结构化产物，并接入趋势看板。

### 11.1 通用字段

建议每次 nightly/weekly 保存：

- model。
- hardware。
- connector。
- kv_role。
- test type。
- commit。
- vllm version。
- dtype。
- TP/DP/EP 配置。
- pass/fail。
- error message。
- serve command。
- environment。

### 11.2 PD 分离指标

建议保存：

- TTFT avg / P50 / P90 / P99。
- TPOT avg / P99。
- throughput。
- transfer latency。
- transfer bytes。
- transfer error count。
- P/D server success rate。
- proxy routing latency。
- P/D NPU memory peak。
- request cancel count。
- timeout count。

### 11.3 池化指标

建议保存：

- cold TTFT。
- warm TTFT。
- hit ratio。
- hit count / miss count。
- save latency。
- load latency。
- backend latency。
- eviction count。
- CPU DRAM usage。
- HBM usage delta。
- fallback count。
- backend error count。

### 11.4 阈值建议

可以分两类阈值：

| 类型 | 用途 |
|---|---|
| 硬阈值 | 精度不能低于 baseline，错误率不能超过上限 |
| 趋势阈值 | TTFT、TPOT、hit ratio、latency 相比最近 N 次不能退化过多 |

性能类指标建议优先用趋势阈值，因为不同硬件、不同负载下绝对值波动较大。

---

## 十二、当前覆盖缺口与建议动作

### 12.1 当前主要缺口

| 场景 | 当前状态 | 缺口 |
|---|---|---|
| PD V1 | PR/nightly/weekly 均有覆盖 | 覆盖较充分 |
| PD Layerwise | weekly 有，PR/nightly 无 | nightly 应补，PR 可评估补轻量用例 |
| PD Hybrid | weekly 有，PR/nightly 无 | 模型较重，暂不建议 PR，nightly 需资源评估 |
| AscendStore 池化 | nightly 有非专项覆盖 | 缺明确 cache hit/save/load 指标 |
| SimpleCPUOffload | PR 有，nightly 无 | 应纳入 nightly |
| RecomputeCPUOffload | 仅 UT，无 e2e | 应补 e2e，再纳入 nightly |
| LMCacheAscend | 无 | 等外部库环境就绪 |
| UCM | 无 | 等 UCM 服务端环境就绪 |
| A5 平台 | 无任何参考用例 | 等 A5 环境和 runner 信息明确 |

### 12.2 P0 建议

1. **nightly 纳入 `SimpleCPUOffloadConnector`**

复用已有 PR 测试：

```yaml
- name: test_simple_cpu_offload
  os: linux-aarch64-a2b3-1
  tests: tests/e2e/pull_request/one_card/test_simple_cpu_offload.py
```

价值：成本低，不需要新模型，不需要外部依赖，可以马上提升池化 nightly 看护。

2. **nightly 纳入 `MooncakeLayerwiseConnector`**

复用 weekly 中较轻的 GLM-4.7 layerwise 用例：

```yaml
- name: multi-node-glm-4.7-mooncake-layerwise
  config_file_path: GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml
  size: 2
```

前提：确认 YAML 是否已存在于 nightly config 目录；如果只在 weekly，需要复制或建立合适引用。

3. **新增配置 lint**

重点检查：

- nightly/weekly config 引用文件存在。
- connector 名合法。
- kv_role 合法。
- PD 分离 P/D 分组完整。
- Recompute 必须启用 scheduler。
- A2/A3/310P section 不混用。

### 12.3 P1 建议

1. **给 `AscendStoreConnector` 增加明确池化命中验证**

当前 Qwen3-30B 精度测试只是附带覆盖，不够专项。应增加：

- cold run。
- warm run。
- hit count/hit ratio。
- save/load latency。
- cold/warm TTFT 对比。
- 输出一致性。

2. **新增 `RecomputeCPUOffloadConnector` e2e**

建议场景：

```text
2 节点 PD 分离环境
├── P 节点：MooncakeConnectorV1，kv_producer
└── D 节点：MultiConnector
    ├── MooncakeConnectorV1，接收 P 的 KV
    └── RecomputeCPUOffloadConnector，处理 D 节点抢占/offload/load
```

必须配置：

- `kv_role="kv_consumer"`。
- `MultiConnector`。
- `recompute_scheduler_enable: true`。
- 构造足够压力触发抢占。

验证：

- 被抢占请求可以恢复。
- 输出与不抢占 baseline 一致或在可接受范围内。
- offload/load 日志或 metrics 确认发生。
- 无资源泄漏。

3. **为 PD 分离增加 transfer metrics**

至少记录：

- transfer latency。
- transfer bytes。
- transfer error count。
- timeout count。
- P/D memory peak。

4. **为池化增加 cache metrics**

至少记录：

- hit ratio。
- save/load latency。
- backend error。
- cold/warm TTFT。
- eviction count。

### 12.4 P2 建议

1. **评估 Hybrid Connector 是否进入 nightly**

前提：

- DeepSeek-V4 Flash 多节点稳定。
- A3 2 节点资源允许。
- nightly 时长可接受。

2. **LMCache / UCM 环境就绪后补充 e2e**

在外部库和服务稳定前，不建议强行纳入 PR/nightly。

3. **建设长稳与故障注入 pipeline**

适合 weekly 或手工触发，重点发现慢性问题和异常恢复问题。

---

## 十三、推荐看护矩阵

| 场景 | UT | 组件测试 | PR e2e | nightly | weekly | 故障/长稳 | 指标看板 |
|---|---|---|---|---|---|---|---|
| PD V1 | 必须 | 建议 | 已有 | 已有 | 已有 | 建议 | 必须 |
| PD Layerwise | 必须 | 建议 | 建议补 | 必须补 | 已有 | 建议 | 必须 |
| PD Hybrid | 必须 | 建议 | 可不做 | 评估 | 已有 | 建议 | 必须 |
| AscendStore 池化 | 必须 | 必须 | 视依赖 | 已有但需增强 | 建议 | 建议 | 必须 |
| SimpleCPUOffload | 必须 | 建议 | 已有 | 必须补 | 可选 | 可选 | 建议 |
| RecomputeCPUOffload | 必须 | 必须 | 评估 | 必须补 | 建议 | 建议 | 必须 |
| LMCache | 必须 | 必须 | 不建议 | 环境就绪后 | 环境就绪后 | 建议 | 必须 |
| UCM | 必须 | 必须 | 不建议 | 环境就绪后 | 环境就绪后 | 建议 | 必须 |

---

## 十四、最终结论

PD 分离和池化的看护重点不同：

- **PD 分离** 应重点看护：KV 传输正确性、P/D 路由、输出一致性、TTFT/吞吐、资源释放、传输异常恢复。
- **池化** 应重点看护：save/load/lookup 正确性、prefix 命中准确性、cold/warm 输出一致性、命中率、后端稳定性、cold/warm 性能收益、eviction 和容量行为。

完整看护体系不应只靠 e2e，而应形成：

> 静态配置检查 + UT + 组件集成测试 + PR 小模型 e2e + nightly 真实服务 + weekly benchmark + 故障长稳 + 指标趋势看板。

最优先的落地动作是：

1. 把 `SimpleCPUOffloadConnector` 纳入 A2 nightly。
2. 把 `MooncakeLayerwiseConnector` 的 GLM-4.7 2 节点用例纳入 A3 nightly。
3. 增加 KV transfer 配置 lint。
4. 给 `AscendStoreConnector` 增加明确 cache hit/save/load 指标。
5. 新增 `RecomputeCPUOffloadConnector` e2e，并在稳定后纳入 nightly。
