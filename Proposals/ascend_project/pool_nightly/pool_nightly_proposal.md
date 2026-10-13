# Nightly 池化看护方案 —— DeepSeek-V4-Flash（a3 & a5）

| 项目 | 内容 |
|---|---|
| 状态 | 最终载体分工已确定；PR #17624 配置已同步，性能基线待实跑回填 |
| 日期 | 2026-09-25（2026-09-30 rebase、2026-10-04 载体矩阵修订） |
| 仓库 | vllm-ascend（工作分支 `pool_nightly`，基于 main@c39b31aab） |

---

> **2026-10-04 最终决议**：S2 改为 A5 Memcache layerwise，S4 改为 A3 Mooncake non-layerwise。本文后续基于旧 backend 分工的 S2/S4 配置、风险和决策记录作为历史保留；实际落地以本页 §2.1 及 `s2.md`、`s3.md`、`s4.md` 的最终修订为准。

## 1. 目标与约束

**目标**：把 KV 池化特性（mooncake / memcache 两种池 backend + layerwise）纳入 nightly 看护，模型 DeepSeek-V4-Flash，平台覆盖 a3 & a5，看护内容 = 精度 + 性能。

**硬约束**：不新增用例、不新起服务、不新建 yaml、CI 注册不动、runner 脚本不动——池化配置一律**叠进 nightly 已有服务**，让已有/扩展的考题来看护。

**设计基线**：各载体除池化外的服务参数保持 nightly 现状。Memcache layerwise 由 S1/S2/S3 分别覆盖 A3 PD、A5 DP8×TP1、A3 DP4×TP4；Mooncake non-layerwise 由 S4 覆盖 A3 TP8×MTP3 的整块 KV 读写。

**看护深度定义**（"池化精度"的完整含义）：
- **写路径精度**：池化写池不得破坏本地 KV / PD 传输（由精度考题判卷）
- **读路径精度**：池命中回读的 KV 必须正确（需要带共享前缀的判分负载才能触发）
- **MTP3×池交互**：生产 MTP 档位为 3，draft KV 写池与 verify×池并发
- **开销水位**：池化写池开销下的吞吐下限

**实施状态**：四个载体均叠加到既有服务；S2/S3/S4 使用同一 prefix90 暖池方法看护读取性能，正式 baseline/threshold 在各自最终形态上重新测量。

---

## 2. 总体设计

### 2.1 四载体看护矩阵（最终方案门面）

| 场景 | 平台/拓扑 | 池 backend | layerwise | 准确的主要看护内容 |
|---|---|---|---|---|
| S1 | A3，PD 分离 | memcache | Producer 开启，Consumer 关闭 | MultiConnector、PD 传输、Prefill 逐层写池、AIME 精度 |
| S2 | A5，DP8×TP1 | memcache | 开启 | A5 `device_urma`、逐层写入/热池读取、GPQA 精度、性能 |
| S3 | A3，DP4×TP4 | memcache | 开启 | A3 基础逐层热池读取、GPQA 精度、稳定性能哨兵 |
| S4 | A3，TP8、MTP3 | mooncake | 关闭 | 整块 KV 写入/读取、MTP3 acceptance、verify 开关、性能 |

精度由 S1 的 AIME 与 S2/S3 的 GPQA 承担；S4 的 acceptance rate 是 MTP 行为指标，不等同于答案正确率。严格的池命中后答案精度仍需要带共享前缀的可判分数据。

### 2.2 改动清单

**改 4 个文件；新增 0 个 yaml、0 个 test_case；CI 注册与 runner 脚本 0 改动。**

| 文件 | 载体 | 分期 |
|---|---|---|
| `nightly/multi_node/external_dp/config/DeepSeek-V4-flash-w8a8-PD-prefix.yaml` | S1 | Phase 1 |
| `nightly/single_node/models/configs/DeepSeek_V4_Flash_A5.yaml` | S2 case #1 | Phase 1 |
| `nightly/single_node/models/configs/acceptace_rate_deepseekv4-flash-w8a8-mtp.yaml` | S4（两个 case） | Phase 1 |
| `nightly/single_node/models/configs/DeepSeek-V4-Flash-W8A8-A3.yaml` | S3 | **Phase 2a**（perf-only 读路径哨兵，前置 = R9 gate）/ **Phase 2b**（acc-prefix 判卷，前置 = Q6 造数） |

### 2.3 可行性确认（无需改任何脚本）

- 单机 runner 支持 **per-test-case** `kv_pool`：case 生命周期内自动拉起 memcache MetaService / mooncake_master 并注入配置（`kv_pool_runtime.py`）
- 多机 external_dp runner 支持 yaml 级 `kv_pool`，node-0 拉起池服务
- `AscendStoreConnector` 支持 `backend: memcache/mooncake`、`use_layerwise`、`kv_role: kv_both`（单机）/ `kv_producer/kv_consumer`（PD）

---

## 3. 载体改造详情

### 3.1 S1：a3 1p1d memcache 池化（Phase 1）

**现状**：MooncakeHybridConnector PD 传输，仅 AIME2025 精度（90/10），无池化、无性能考题。

**改造**：与 weekly `DSv4-Flash-Memcache-PD-layerwise.yaml`（**同模型、同拓扑**先例）对齐——

① yaml 顶层新增 kv_pool（node-0 自动起 MetaService）：

```yaml
kv_pool:
  type: memcache
  meta_service_port: 5000
  config_store_port: 6000
  config:
    meta:
      ock.mmc.log_level: error
    local:
      ock.mmc.log_level: error
      ock.mmc.local_service.world_size: 256
      ock.mmc.local_service.protocol: device_sdma
      ock.mmc.local_service.dram.size: 1GB
```

② 两端 kv-transfer-config 由单 MooncakeHybrid 改为 MultiConnector（原连接器参数原样保留在 connectors[0]）：

```json
{"kv_connector": "MultiConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"connectors": [
  {"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_producer", "kv_port": "30000", "engine_id": "0", "kv_connector_extra_config": {"prefill": {"dp_size": 4, "tp_size": 4}, "decode": {"dp_size": 16, "tp_size": 1}}},
  {"kv_connector": "AscendStoreConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"backend": "memcache", "use_layerwise": true}}
]}}
```

```json
{"kv_connector": "MultiConnector", "kv_role": "kv_consumer", "kv_connector_extra_config": {"connectors": [
  {"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_consumer", "kv_port": "30100", "engine_id": "1", "kv_connector_extra_config": {"prefill": {"dp_size": 4, "tp_size": 4}, "decode": {"dp_size": 16, "tp_size": 1}}},
  {"kv_connector": "AscendStoreConnector", "kv_role": "kv_consumer", "kv_connector_extra_config": {"backend": "memcache", "use_layerwise": false}}
]}}
```

③ benchmarks 追加性能考题（与 weekly 同形对标，基线来自同款服务形态）：

```yaml
  perf_TPOT50_3_5_1_5:
    case_type: performance
    dataset_path: vllm-ascend/GSM8K-in3500-bs4096-prefix0-ds
    request_conf: vllm_api_stream_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_str_perf
    num_prompts: 4096
    max_out_len: 1500
    batch_size: 1024
    request_rate: 12
    baseline: 15569.48
    threshold: 0.95
```

**考题与基线**：AIME2025（90/10）原样保留，同时看护池化写路径精度；TPOT50 新增。

**注意**：与 weekly 先例的两处保留差异（nightly 现状）：prefill 端 prefix-caching 默认开（先例为关，见 Q1）、MooncakeHybrid 带 engine_id。其余全部对齐。性能影响分析见 §4.3。

**2026-09-29 rebase 记录**：上游 main 前进引发 PR #17624 冲突，已 rebase 至 main@31608b3f5。冲突源两个：#17628（CI DP 调度改造，yaml 顶层加 `dp_load_balancing: external`）与 #17661（MRV2 revert，**prefill 拓扑 TP1+CP4 → TP4**、删 `VLLM_USE_V2_MODEL_RUNNER`、additional-config 加 `enable_dsa_cp`）。解决 = connectors[0] 内嵌 prefill `tp_size` 随上游基座更新为 4。**这使 nightly prefill 拓扑与 weekly 先例首次精确一致**（此前 nightly 为 TP1+CP4 ≠ 先例 TP4——首版 PR 按nightly原状内嵌 `tp_size: 1`，与 §3.1 JSON 按先例形态记的 4 存在文档-实现偏差，本次 rebase 归一）。两处保留差异复核不变；`enable_dsa_cp`/`dp_load_balancing` 两例皆有，不构成新差异；weekly TPOT50 基线 15569.48/0.95 复核无误。

**2026-09-30 rebase 记录**：上游 main 再度前进（31608b3f5 → c39b31aab）引发 PR #17624 dirty，冲突源两个：#17737（A5 全部 case 模型路径改 `deepseek-ai/DeepSeek-V4-Flash`）与 #15011（batch-sharded sampling，删各 case additional-config 的 `enable_reduce_sample`）。唯一真冲突在 A5 case #1 的 additional-config 行（S4 同改动区域不重叠，自动合并）。解决 = rebase 至 main@c39b31aab，以 upstream 新版 case #1（新模型路径 + 无 `enable_reduce_sample`）为底重放 kv_pool / kv-transfer-config / 内联 benchmarks 三层叠加。diff 体量复核不变（4 文件 +110/-7），静态校验与 merge-tree 全过。现行 commit：`8716c5ae7`（Phase 1 三载体）+ `2d4a53e4c`（S3 读路径哨兵，gate FAIL 的 revert 对象）。

### 3.2 S2 case #1：a5 单机 Mooncake 旧方案（历史，已由 `s2.md` 的 Memcache Layerwise 方案取代）

**现状**：6 个 test_case 共享 `_server_cmd` anchor；case #1（GPQA_Accuracy_Test_01，MTP=1，135k）自带 GPQA 精度考题，是唯一不破坏 04/05/06 rejection sampler 对比矩阵的池化载体。

**改造**（只动 case #1，3 处变更：kv_pool 块、server_cmd_extra 追加 kv-transfer-config、benchmarks 显式展开并补挂 perf）：

```yaml
  - name: "DeepSeekV4_Flash_A5_Single_GPQA_Accuracy_Test_01"
    model: "/root/.cache/modelscope/hub/models/models--deepseek-ai--DeepSeek-V4-Flash"
    envs:
      <<: *envs
    kv_pool:
      type: "mooncake"
      master_port: 50088
      metrics_port: 50089
      config:
        metadata_server: "P2PHANDSHAKE"
        protocol: "ascend"
        device_name: ""
        global_segment_size: "1GB"
        preferred_segment: false
        prefer_alloc_in_same_node: true
    server_cmd: *server_cmd
    server_cmd_extra:
      - "--max-model-len"
      - "135000"
      - "--speculative-config"
      - '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":"true"}'
      - "--additional-config"
      - '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,"enable_reduce_sample":false,"rejection_sampler_config":{"enable_block_verify":false,"enable_entropy_verify":false,"posterior_threshold":0.95,"posterior_alpha":0.4}}'
      - "--kv-transfer-config"
      - '{"kv_connector": "AscendStoreConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"backend": "mooncake", "lookup_rpc_port": "0", "use_layerwise": true}}'
    benchmarks:
      acc:
        case_type: accuracy
        dataset_path: /root/.cache/modelscope/hub/datasets/vllm-ascend/gpqa
        request_conf: vllm_api_general_chat
        dataset_conf: gpqa/gpqa_gen_0_shot_cot_chat_prompt
        max_out_len: 131072
        batch_size: 32
        baseline: 87.4
        threshold: 5
        temperature: 1.0
        ignore_eos: false
        top_p: 1.0
        thinking: True
      perf:
        case_type: performance
        dataset_path: /root/.cache/modelscope/hub/datasets/vllm-ascend/DeekSeek-V4-Flash_input_3500_lines_128
        request_conf: vllm_api_stream_chat
        dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_str_perf
        num_prompts: 128
        max_out_len: 1500
        batch_size: 32
        request_rate: 0
        baseline: 83.50106007067137
        threshold: 0.95
        temperature: 0
        ignore_eos: True
```

**考题与基线**：GPQA 87.4/5 沿用（写路径精度）；gsm8k 83.50/0.95 首跑校准（开销水位）。

**注意**：
- benchmarks 显式展开、不引用共享 anchor（`*benchmarks_per` 被 04/05/06 引用，池化基线需独立校准，必须解耦）
- 端口纪律：50088/50089 为 nightly/template.md 语境标准值；`common_constraints` 端口规划表的 9008 为共享验证服务器语境的纪律值（注明"换机勿照抄"）——**二者语境不同、均正确，勿互"纠正"**。首跑前预检 5000/6000/50088/50089 空闲（共享服务器端口纪律）；S1 与 S4 同用 memcache 5000/6000 但分属 multi_node / single_node 两个 runner 池，并行无冲突
- mooncake 池 + kv_both + layerwise=true 组合无 yaml 实跑先例（框架与 template.md 文档支持已确认），回退见 R2
- 已知代价（R7）：case #1 池化后，a5"非池化 BF16 精度"失去 nightly 看护
- 其余 5 个 case 一字不动；夜间服务次数不变

### 3.3 S4：a3 单机 Memcache 旧方案（历史，已由 `s4.md` 的 Mooncake Non-layerwise 方案取代）

**现状**：全 nightly 唯一真正测量+断言 MTP 接受率的用例——2 个 case（enable-open / enable-close）构成 verify 开关对比，均带 `spec_decode_acceptance`（0.925/0.05）+ perf（90/0.97）；TP8，MTP=3。

**改造**（生产 MTP 档位 = 3，"MTP3×池共存"为需求内看护；两 case 均加，池化为 open/close 对比的常量，对比依然成立，且同挂=池问题、单挂=verify 组合问题，具诊断价值）：

```yaml
_kv_pool: &kv_pool                     # 新增 anchor（贴文件既有风格）
  type: "memcache"
  meta_service_port: 5000
  config_store_port: 6000
  config:
    meta:
      ock.mmc.log_level: "info"
    local:
      ock.mmc.log_level: "info"
      ock.mmc.local_service.world_size: 256
      ock.mmc.local_service.protocol: "device_sdma"
      ock.mmc.local_service.dram.size: "10GB"

_server_cmd: &server_cmd              # 原有参数全部保留，末尾追加：
  # ...
  - "--kv-transfer-config"
  - '{"kv_connector": "AscendStoreConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"backend": "memcache", "lookup_rpc_port": "0", "use_layerwise": false}}'

test_cases:
  - name: "deepseekv4-flash-w8a8-enable-open"
    # ... 原有字段全部不变 ...
    kv_pool: *kv_pool
  - name: "deepseekv4-flash-w8a8-enable-close"
    # ... 原有字段全部不变 ...
    kv_pool: *kv_pool
```

**考题与基线**：接受率 0.925/0.05 原样保留（哨兵升级为"池化语境下的 MTP3 接受率"，预期沿用，首跑确认）；perf 90/0.97 首跑校准。

**注意**：
- backend=memcache + kv_both + MTP=3 与 weekly `Qwen3.5-397B-Memcache-perf` **结构同款**（该先例已实跑验证 MTP3+memcache+kv_both 共存）；S4 特有增量 = DSV4 模型（MLA/indexer）+ W8A8 + deepseek_mtp + TP8
- `use_layerwise=false`：贴 Qwen 先例、聚焦 MTP3×池交互（layerwise 已由 S1/S2#1/S3 承载；见 Q8）
- 覆盖定位：TP8 非生产并行形态（生产为 DP4×TP1），本载体覆盖 **layout 无关的 MTP3×池代码交互**；生产形态的"纯 DP + MTP3 + kv_both"精确组合仍无载体（由 S2#1 与 S1 decode 分段近似），**记入已知缺口**

### 3.4 S3：a3 单机读路径看护（Phase 2a = perf-only 哨兵，2b = 判卷考题）

**现状**：单 case 自包含（W8A8，DP4×TP4，prefix-caching 开），GPQA（86.36/5）+ GSM8K prefix0 性能（181.88/0.97）。

**硬前置（按 2a/2b 分期）**：读路径看护 = perf-only 哨兵（2a）+ 判卷考题（2b），两者前置不同——
- **2a（perf-only 读路径哨兵）**：前置仅 **R9 gate**（block_hashes 预验证，判据见 §5，脚本已随方案交付：`r9_gate.sh`）。**S3 yaml 已随 PR #17624 第二个 commit（2d4a53e4c）提交——gate 语义转为"合并前验证"：FAIL 则 revert 该 commit。**负载**不需要造数**——`GSM8K_prefix{0|90|99}_in{长度}_bs{批量}_{模型后缀}` 是现成家族。**数据集已定案：`GSM8K_prefix90_in65536_bs1000_qwen`**——选择依据不是 weekly 先例（397B weekly 在 a3-800**t**-16，与 S3 的 a3-800**i**-16 **不是同一机器池**，weekly 缓存证据不可迁移），而是**同池实证**：a3-800i-16 上注册的 `qwen_3_6_27b_w8a8_A3` 与 `Kimi-K2.5` nightly 正在用 prefix90 系数据集（in65536_bs1000_qwen ×2、in131072_bs1000_kimi 等），pod 缓存存在性有据（pod 禁网不能自下载，缓存实证是硬前提）。qwen tokenizer 校准在 DSV4 分词下 token 数漂移——perf 哨兵可接受（yaml 注释已记）。`_deepseek` 后缀 prefix90 变体从未被任何配置引用、无缓存证据，弃用；2b 补造 deepseek 判分变体时可替换。
- **2b（acc-prefix 判卷）**：前置 = Q6 数据集。生态缺口精确化：**不是"prefix 负载不存在"，而是"带共享前缀 + 可判分（jsonl 带判分字段 + 配套判分 dataset_conf）"不存在**——所有 prefix 类数据集均只配 perf conf 不判卷，所有精度数据集均无共享前缀。Q6 规格可能降级：若现成 prefix90 数据集的 jsonl 自带 ground truth，造数退化为"验证判卷性 + 补一个 dataset_conf"（待服务器侧看数据样例）。
- **顺序约束不变**：`acc-gpqa → perf → acc-prefix`（perf 写满共享前缀、acc-prefix 暖池判卷）——但顺序有效性只约束 2b（两门同源才需前缀 token 级一致）；2a 单独上线时顺序自然退化为 `acc-gpqa → perf`，无额外条件。

**改造**（3 处服务变更 + 考题三门化）：

服务变更：① case 加 kv_pool（memcache，同 S4 的 `_kv_pool` 结构，dram 10GB）；② `--enable-prefix-caching` 改 `--no-enable-prefix-caching`（让池成为唯一命中/跳算来源、保证读路径信号确定性；原"关 HBM 截流"表述已按 §4.2 校准——S3 走 gva 路径本就不受截流；**依赖 block_hashes 预验证**，见 R9）；③ server_cmd 追加 kv-transfer-config（同 S4 写法但 `use_layerwise: true`）。

考题改造（GPQA 保留 + 新增读路径判分 + perf 换前缀负载）：

```yaml
    benchmarks:
      acc-gpqa:                                   # 保留：写路径精度（86.36 沿用）
        case_type: accuracy
        dataset_path: vllm-ascend/gpqa
        request_conf: vllm_api_general_chat
        dataset_conf: gpqa/gpqa_gen_0_shot_cot_chat_prompt
        max_out_len: 65536
        batch_size: 32
        baseline: 86.36
        threshold: 5
        thinking: true
        reasoning_effort: low
      perf:                                       # 替换：原 prefix0 perf（181.88 作废）→ 带前缀负载【Phase 2a 交付】
        case_type: performance                    # 混合信号：首波冷池纯写池，后续波次池读收益混入（语义校准见下方注意项）
        dataset_path: vllm-ascend/GSM8K_prefix90_in65536_bs1000_qwen     # 定案：同池（a3-800i-16）缓存实证，见"硬前置"2a
        request_conf: vllm_api_stream_chat
        dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_str_perf
        num_prompts: 64
        max_out_len: 1024
        batch_size: 16
        request_rate: 0
        baseline: 1                               # 首跑前记录式占位，跑通后回填
        threshold: 0.1                            # 占位期阈值（与 §5 Phase 2a、PR 提交一致），回填实测后恢复 0.97
      acc-prefix:                                 # 新增：读路径精度判卷（核心增量）【Phase 2b 交付，前置 = Q6 数据集】
        case_type: accuracy                       # 暖池判卷 = perf 已写满共享前缀，全量池命中
        dataset_path: vllm-ascend/GSM8K_prefix90_in32768_bs1000_deepseek   # 暂定名，待 Q6（2b 不就绪则该条目不挂）
        request_conf: vllm_api_general_chat
        dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_chat_prompt
        max_out_len: 1024
        batch_size: 32
        baseline: 0                               # 首跑前记录式占位，跑通后回填
        threshold: 100
```

**注意**：
- **考题顺序即执行序**：单机 runner 对同一 case 的 benchmarks 按 yaml dict 序串行执行（`test_single_node.py` 按 dict 序取 keys → `aisbench.py:run_aisbench_cases` 串行 for，全链无重排点，已代码级核实）。顺序定为 acc-gpqa（写路径）→ perf → acc-prefix
- **perf 语义校准**：prefix90 负载下只有**首波并发请求**是冷池（纯写池开销），第二波起 gva 路径即查询命中、池读收益混入——聚合数字实为"池化对 prefix90 负载的**净效应**"，**勿按纯开销水位解读**；作为吞吐下限哨兵仍成立（基线回填后看趋势，R4 兜底）。**acc-prefix 判卷语义**：perf 已把共享前缀写满池，全量请求池命中 = 读路径判卷干净——这是顺序设计的核心收益，两门考题的语义分工在此解耦
- 顺序有效的前提（记入 Q6 造数规格）：perf 与 acc-prefix 两个 dataset_conf 渲染出的**共享前缀必须 token 级一致**，否则暖池退化为冷池、读路径覆盖打折
- 对线上零扰动：S3 属 Phase 2 未落线，线上 acc-gpqa/perf 两条目的相对位置不变，acc-prefix 为纯新增尾挂
- GPQA 保留（Q5）：写路径看护零成本保留，S3 成为全家族看护深度最完整载体（写+读+开销三门）
- 已知代价（R7）：非池化 W8A8 单机参考水位消失（86.36 由池化语境承接、181.88 作废），趋势断点
- 池容量：prefix90 × 32k 输入 × 64 prompts 共享结构下 dram 10GB 足够，首跑看 memcache 日志确认

---

## 4. 技术依据

### 4.1 weekly 先例对照

| 先例 | 用途 | 关键点 |
|---|---|---|
| `weekly/.../DSv4-Flash-Memcache-PD-layerwise.yaml` | S1 模板 | **同模型同拓扑**：1p1d + MultiConnector(MooncakeHybrid + AscendStore(memcache)) + producer layerwise=true / consumer false + TPOT50（15569.48） |
| `weekly/single_node/configs/Qwen3.5-397B-Memcache-perf.yaml` | S2#1 / S3 / S4 结构模板 | 单机 kv_pool + AscendStoreConnector kv_both；**MTP3 + memcache + kv_both 已实跑共存**；其 `baseline: 1` 是新基准首跑前的事实标准占位写法 |
| `nightly/.../external_dp/config/template.md` | mooncake 池写法 | MultiConnector 与 kv_pool(type=mooncake) 标准示例、注入环境变量说明 |

**配置地雷（kv_load_failure_policy）**：默认 `fail` 即本方案看护所需语义——池读失败响亮失败，不静默吞掉；对 S3 读路径判卷尤其重要（若 recompute 可用，加载失败被静默掩盖、acc-prefix 形同虚设）。**勿抄 template.md 示例中的 `kv_load_failure_policy: "recompute"`**：hybrid 模型（DSV4-Flash）配 recompute 被 `platform.py:1065-1067` 断言直接拒绝，服务起不来。各载体省略该字段 = 默认 fail = 正确。

### 4.2 池命中机制与读路径看护原理

池 lookup 的代码逻辑（`pool_scheduler.py`）**按路径分化**，读路径看护的条件须按载体实际路径取用：

**① 粒度门（查询前置门槛）**：非 layerwise 路径与 block_key layerwise 路径查询前先 `floor_to_cache_transfer_granularity(prompt_len)`，低于 `cache_transfer_granularity`（可缓存组块尺寸 LCM，`metadata.py:infer_cache_transfer_granularity`；本方案块配置下量级 128~512 token）**不发起池查询、直接返回 0 命中**（非 layerwise `:592-598`、block_key `:583-586`）。**gva layerwise 路径无此门**（`:588-590`，token_len 直取 prompt 长度进查询）——S3（memcache+layerwise）恰走无门路径，32k 输入下两种门均无关。

**② HBM 截流（仅非 layerwise 路径）**：`num_computed_tokens >= token_len`（本地 prefix cache 全命中）时跳过池查询（`:605`）。**gva layerwise 路径例外**：显式从 block 0 查池、无视本地 HBM 命中（`:460-461`，注释明示"池存逐层数据可能与本地 prefix cache 不匹配"），且 HBM 全命中而池也命中时仍 force load（`:661-664`）。

**③ block_hashes 依赖**：两条 lookup 路径均直接消费 `request.block_hashes`（代码级证据见下段）。

因此读路径看护成立需四条件：⓪ prompt 过粒度门（S3 的 gva 路径免）；① 池化 on；② 前缀不在 HBM（关 prefix-caching，待 block_hashes 验证，见 R9）；③ 负载带共享前缀且可判分（命中 → 回读错 KV → 答案错 → 掉分报警）。S3 即按此设计。

**S3 关 prefix-caching 的理由校准**：原表述"绕过 HBM 截流"仅对非 layerwise 载体成立——S3 走 gva 路径本就不受截流（且有 force load 兜底）。关 prefix-caching 的真实理由 = **信号确定性**：开着时重复请求的主路径是 HBM 块复用（本地跳算），池读命中与 force-load 兜底分支混叠、perf 归因失义；关掉后**池成为唯一命中/跳算来源**，回读 KV 直接决定答案对错、跳算收益直接决定耗时——"池回读必须正确"的看护语义才成立。block_hashes 生成问题（R9）为第二理由，不变。

**block_hashes 依赖的代码级证据（R9 依据）**：两条 lookup 路径都直接消费 `request.block_hashes`——非 layerwise 路径 `LookupKeyClient.lookup(token_len, request.block_hashes, ...)`，layerwise 路径 `request.block_hashes[query_start_block:num_blocks]` 切片；且 hash 为空时**静默返回 0 命中、不报错**（`if not block_hashes_to_query: return 0`）。即若上游在关 prefix-caching 时不生成 block_hashes（配对 vLLM v0.30.0 的版本行为，本机源码不可核），失败形态是**哑失效**——池从未被读、考题照常全绿——而非响亮报错。因此 R9 预验证判据必须锚定**正向命中信号**：全路径通用观测点为 INFO 级逐请求日志 `Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d`（命中与哑失效可区分：`kvpool hit tokens: >0` vs 恒为 0），判据见 §5 Phase 2 第 2 步。

### 4.3 S1 叠池化对 PD 分离性能的影响

**结论：PD 主链路（P→D 传输、decode 端 TPOT）基本不受影响；额外开销集中在 prefill 端"每层多一次写池"。**

| 环节 | 影响 | 机制 |
|---|---|---|
| Decode 端 | ≈0 | consumer layerwise=false，KV 仍走 MooncakeHybrid 收；AscendStore 每步仅元数据查询；preempt 恢复为正收益 |
| Prefill 端 | **主要开销点** | `AscendMultiConnector.save_kv_layer` 每层先调 AscendStore 写池（provider）再调 MooncakeHybrid 传 PD；layerwise 逐层异步写可与后续层计算 overlap，但高压负载下竞争 copy/网络带宽 |
| 收益面 | 当前负载下为 0 | TPOT50 数据集 prefix0、AIME2025 每题独立，均无 prefix 复用收益（S3 Phase 2 上线后收益面才有专门载体） |

TPOT50 基线 15569.48 来自 weekly **同款服务形态**（同形对标），池化开销已历史性包含在基线内，首跑预期偏差小。

---

## 5. 验证与基线校准计划

**Phase 1（S1 + S2 + S4）**

1. **提交前本地静态校验**（零号步骤，不占 CI 资源）：全部改动 yaml 过 `yaml.safe_load`；每条 kv-transfer-config 字符串过 `json.loads`——runner 无 schema 预检（`config.py` 运行时才解析，格式错误要到起服务才暴露）。重点对象：S1 两段单行大 JSON、S4 新增 anchor、S2#1 anchor 解耦展开
2. `pool_nightly` 分支提交 3 个文件修改
3. 提 PR 后手动触发（触发前预检 5000/6000/50088/50089 端口空闲，共享服务器端口纪律）：
   - `/nightly DeepSeek-V4-flash-w8a8-PD-prefix`
   - `/nightly DeepSeek_V4_Flash_A5`
   - `/nightly acceptace_rate_deepseekv4-flash-w8a8-mtp`
4. 首跑校准：S1 TPOT50 预期即过（同形对标，首跑观察协议见 §7 Q1 决策记录）；S2#1 GPQA 87.4 预期不动、gsm8k 83.5 按实测回填；S4 接受率 0.925 预期沿用、perf 90 按实测回填
5. 跑通后合入

**Phase 2a（S3 perf-only 读路径哨兵）——前置仅 R9 gate，不依赖造数**

1. **R9 gate（脚本已交付：`r9_gate.sh`，随本方案目录）**：a3 测试机起池化服务——memcache 池（5000/6000，S4 `_kv_pool` 同款配置）+ AscendStoreConnector（`use_layerwise: true`, `kv_both`）+ `--no-enable-prefix-caching`，**S3 per-rank 同形态**（Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp，TP4，block 128，MTP1，1M ctx；DP1 保证两条请求落在同一 engine、命中归因无歧义）——先发 1 条、再发 1 条**共享前缀**请求（~4k token 前缀，31 个 128-token 块）；判据 = 第 2 条请求 INFO 日志 **`kvpool hit tokens: >0`**（`pool_scheduler.py:715-721`，logger.info 默认可见；第 1 条同时验证写路径正常性）。**`kvpool hit tokens: 0` 且无报错 = 哑失效 = gate 不过**，S3 2a 不提 PR，按 R9 回退
2. S3 改造已随 PR #17624 提交（2d4a53e4c）：kv_pool（memcache 5000/6000，S4 anchor 同款）+ prefix-off + AscendStoreConnector（memcache，`use_layerwise: true`，gva 路径）+ perf 数据集换 `GSM8K_prefix90_in65536_bs1000_qwen` + 记录式占位（baseline 1 / threshold 0.1，先例 qwen_3_6_27b `perf_prefix_90`）。gate FAIL 则 revert 该 commit 再定；`/nightly DeepSeek-V4-Flash-W8A8-A3` 触发
3. 首跑标定：perf 以实测回填并恢复 threshold 0.97（替换占位 baseline 1/threshold 0.1）；GPQA 86.36 预期沿用；`kvpool hit tokens` 日志顺带确证命中发生（可观测项，期望 perf 波次 2+ >0）
4. 跑通后合入——**读路径"活着且不慢"哨兵上线**（lookup/load-back 崩溃或挂起即红；零命中哑失效由 hit 日志人肉/巡检观测）

**Phase 2b（S3 acc-prefix 判卷）——前置 = Q6 数据集，2a 合入后随时可插**

1. 数据集就绪（带前缀可判分 GSM8K 变体，规格见 Q6；若现成 prefix90 jsonl 自带判分字段则降级为"补 dataset_conf"）
2. yaml 追加 acc-prefix 条目（暖池判卷，顺序依赖 §3.4 已就位的 perf 前置写池），首跑回填实测
3. 跑通后合入——**读路径"回读正确"看护上线**（全家族唯一判卷载体）

**校准规则**：新基准首跑前用记录式占位（accuracy: `baseline: 0, threshold: 100`；performance: `baseline: 1`），跑通后回填实测值；已有基准（GPQA 87.4、AIME 90、接受率 0.925）预期沿用、首跑确认。

---

## 6. 风险与回退

| # | 风险 | 影响 | 回退/缓解 |
|---|---|---|---|
| R1 | S1 与 weekly 先例存在两处保留差异（prefill prefix 默认开、engine_id） | 首跑与基线小偏差；若 PD 握手/路由异常 | 预期可校准；prefix×memcache 交互问题则改显式关闭（Q1）。**PD 握手/路由异常的第一怀疑点 = 子条目 engine_id**：MooncakeHybrid 从 `vllm_config.kv_transfer_config.engine_id` 读取（`mooncake_hybrid_connector.py:1212`），MultiConnector 是否逐子条目重建转发未证；weekly 先例子条目无 engine_id 照常工作，回退 = 删除之、对齐先例 |
| R2 | S2#1 mooncake + kv_both + layerwise 无 yaml 先例；且 case #1 为 fp8 KV（`--kv-cache-dtype fp8`，随 `_server_cmd` anchor 共享）而 Qwen 先例为 BF16——mooncake + layerwise + fp8 KV 页为无先例叠加 | case #1 起服务失败 | 阶梯回退：① layerwise 改 false；② backend 改回 memcache；③ **fp8 解档**——case #1 内联完整 `server_cmd`（与 anchor 解耦），`--kv-cache-dtype` 改 auto（与 R6"池化差异收敛在自有字段"原则一致，把解耦提前到回退预案）。故障只波及 case #1 |
| R3 | S1 TPOT50 压测 4096 条，夜间耗时长 | nightly 时长增加 | 可减量（基线需重标），见 Q2 |
| R4 | 池化后基线漂移 | 首跑报警 | 预期内，按 §5 校准；影响机制见 §4.3 |
| R5 | S2#1 mooncake 池 `global_segment_size=1GB` 未验证 | 池容量不足 | 首跑看 mooncake-master.log 按需调（Q3） |
| R6 | S2 各 case / S4 各 case 共享 `_server_cmd` anchor | 未来他人改共享配置波及全部 case | 现状即如此；池化差异已收敛在自有字段（S2#1 的 kv_pool/extra/benchmarks；S4 有意改共享 anchor 本身） |
| R7 | **已知代价**：S2#1 池化后 a5"非池化 BF16 精度"失看护；S3 深叠后 a3 单机"非池化 W8A8"参考水位消失（性能参考+趋势断点） | 非池化配置回归无人报警 | 方案评审确认接受。救济菜单：① 复议"追加独立 case"（违硬约束，需特批）；**② 备选 C（评审拍板）：a5 case #6（Maximum_Context 1M）目前零 benchmarks、纯起停冒烟——可挂 GPQA 条目恢复非池化 BF16 精度看护，零新服务零新 case；代价 = a5 夜间 +1~3h，且 #6 形态为 MTP4+block_verify ≠ #1 的 MTP1，属"新建非池化参考"而非 87.4 的趋势延续** |
| R8 | **S3 2b 造数依赖**：带前缀**可判分**数据集不存在。查证已收窄缺口：`GSM8K_prefix90_in32768` 负载家族**已存在**且 a3 weekly 在用（qwen/glm 后缀，同规格同长度），缺的是 `_deepseek` 后缀 prefix90 变体（modelscope 是否已有待确认）+ 判卷 dataset_conf | 仅 2b 受阻；**2a 不受影响**（perf 哨兵只需负载家族现有变体） | 2a/2b 拆分后，2a 前置仅剩 R9 gate；2b 规格见 Q6（若现成 jsonl 自带判分字段，降级为"补 dataset_conf"）；2a 数据集已定案 `in65536_bs1000_qwen`（同池缓存实证，PR 2d4a53e4c） |
| R9 | **S3 关 prefix-caching 后池 lookup 的 block_hashes 生成待验证**（两条 lookup 路径均直接消费 `request.block_hashes`，代码级证据见 §4.2） | acc-prefix 命中不生效且**哑失效**（`kvpool hit tokens: 0`、无报错、考题假绿） | **Phase 2a 前置 gate，判据固定，脚本已交付 `r9_gate.sh`**（见 §5 Phase 2a 第 1 步）；S3 yaml 已在 PR #17624 中，gate FAIL 则 revert 2d4a53e4c；回退：① 保 prefix + 负载超量制造 HBM 淘汰；② 退回浅叠（只保写路径） |
| R10 | **S4 唯一接受率哨兵暴露于池化变量**；接受率在含池化的流量上测量断言（warmup→benchmarks→接受率断言，`test_single_node.py:295-324`），nightly 内无非池化对照；池化故障时哨兵阶段性失明 | MTP3 看护诊断价值下降 | Qwen 结构先例降低失败概率；open/close 双 case 交叉诊断；基线首跑校准；**可逆门**——接受率掉线的归因动作固化为：本地跑一次非池化对照（移除 kv_pool + kv-transfer-config 即恢复纯哨兵），诊断路径不临场设计 |

---

## 7. 待拍板事项

| # | 问题 | 备选 | 倾向 |
|---|---|---|---|
| Q1 | S1 prefill 端 prefix-caching 保持默认开还是对齐先例显式关闭？ | A. 保持现状 / B. `--no-enable-prefix-caching` | **A 定稿**（决策记录见表后 Q1 注） |
| Q2 | S1 TPOT50 `num_prompts=4096` 全量还是减量？ | A. 4096 / B. 减量重标 | A |
| Q3 | S2#1 mooncake 池 `global_segment_size`？ | A. 1GB 起步 / B. 更大 | A（首跑看日志调） |
| Q4 | 是否同步改 weekly 的 v4-flash 池化用例？ | A. 不动 / B. 一起改 | A |
| Q5 | S3 是否保留 GPQA 考题（写路径看护）？ | A. 保留（本方案采用）/ B. 去掉 | A（零成本保留） |
| Q6 | **造数规格**（S3 2b 硬前置；2a 已解耦不依赖本项）：基座（建议 GSM8K 判分体系）、前缀占比（建议 ≥80%，暂定名 prefix90）、输入长度（建议 in32768）、生产位置（modelscope vllm-ascend 数据集仓库）。**查证更新**：prefix90_in32768 家族已存在（qwen/glm 后缀），待服务器侧确认 ① modelscope 是否已有 `_deepseek` 后缀变体、② 现成 jsonl 是否自带判分字段（是则降级为补 conf） | 待数据集团队定（先做①②两项确认） | acc/perf 同源双用一个数据集；perf 与 acc-prefix 的 dataset_conf 渲染前缀须 token 级一致（暖池判卷成立前提） |
| Q7 | S3 池 backend 确认 memcache？ | A. memcache（采用）/ B. mooncake | A（Qwen 单机先例 + 与 S1 同 backend） |
| Q8 | S4 `use_layerwise`？ | A. false（采用，贴 Qwen 先例、聚焦 MTP3×池交互）/ B. true | A（MTP3×layerwise×kv_both 为复合无先例组合） |
| Q9 | （可选增强，默认不做）S1 同一 case 内追加"重发考题"条目，看护"HBM 前缀全命中 × PD 传输"交互？ | A. 本方案不做 / B. 追加重发条目 | A；如做须独立提案 + 前置验证（决策记录见表后 Q9 注） |

**Q1 决策记录（A 定稿）**：

- **反对 B 的决定性论据**：weekly 先例 **perf_TPOT50 与 acc_aime2025 两门都挂**。若 S1 选 B（显式关 prefill prefix），改造后 S1 与 weekly 用例几乎完全重合（同模型/拓扑/池化/考题，仅剩 engine_id 差异），nightly 槽位差异化归零；A 使 S1 独家看护"PD + 池化 + prefill prefix 开"这一 weekly 不覆盖的组合。且"PD-prefix"用例名所声称的形态含 prefill prefix 开（decode 端本就显式关），B 使名不副实。
- **A 对池化的增量看护，按确定性分层**：
  - **L1 组合共存（每晚 100% 触发）**：prefix cache 索引 + `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`（池感知参数，`pool_scheduler.py:109` / `pool_worker.py:835` / `patch_kv_cache_coordinator.py:138` 三处读取）+ hybrid KV manager + MooncakeHybrid PD 传输 + AscendStore layerwise 写池——五方协调每晚有真实流量，任一启动断言/调度不变量冲突即红。
  - **L2 增量写池分支（机会性）**：`_prepare_layerwise_put_session` 的 `save_start_token` 切片 + `load_spec.can_load` 跳写两分支（`pool_worker.py:2114-2122`）在 prefix OFF 时是死代码；A 下 TPOT50 高压若触发 preempt，恢复请求经 retention 4096 获部分 HBM 命中 → 尾部重算 → 该分支被真实执行。是否触发**首跑看日志确认**（`kvpool hit tokens` / save 日志），不可默认已覆盖。
  - **L3 看护边界（勿超卖）**：前缀复用收益与池命中率均不可断言——两门考题负载无共享前缀，收益路径零量化看护；L2 触发与否可观测不可看护。
- **首跑观察协议**：TPOT50 偏离 15569.48 超 5% 时，怀疑顺序 = ① prefix×写池交互（切 B 复跑验证）→ ② 子条目 engine_id（R1）→ ③ 硬件环境差异。

**Q9 决策记录（暂不做，归宿固化）**：

- **机制修正**：nightly 跨夜重跑无前缀可复用（case 生命周期起停服务与池服务，均为冷启动）；唯一实现 = 同一服务生命周期内、同 yaml 挂第二个同数据集条目（结构属"扩展考题"，不违反硬约束）。且对 layerwise 载体（S1 producer / S3 / S2#1），重发**会触发真实池查询**——gva 路径从 block 0 查池、无视 HBM 命中、池命中 force load（§4.2）——不是单纯 HBM 复用。真正的看护点是"prefill HBM 全命中跳算时，MooncakeHybrid 给 decode 端送什么"（现 nightly/weekly 均无载体；若传输侧只覆盖本轮计算的块，重发即输出乱码、AIME 暴跌红——真实的 bug 捕手）。
- **前置验证**：先在测试环境单发验证"prefill HBM 全命中 + PD"组合被支持——若该组合今天就断（挂起/断言），重发条目将永久红。
- **成本**：AIME 重发 ≈ S1 夜间时长近翻倍（S1 为多机最贵槽位之一）；perf 侧重发进入全命中 regime、15569.48 基线失效。
- **结论**：前提未验证 + 收益判卷信号弱（前缀占比 <1%）+ S1 分工已满（写路径 + 开销水位）→ 本方案不做；若做，作为与池化看护解耦的独立增强提案。

---

## 附录 A：现有 nightly v4-flash 用例盘点（改造前现状）

| 编号 | CI 注册名 | 平台/拓扑 | 配置文件 | 现状考题 |
|---|---|---|---|---|
| S1 | `DeepSeek-V4-flash-w8a8-PD-prefix`（`a3.double_node`） | a3 双节点 1p1d（每节点 16 卡，W8A8） | `nightly/multi_node/external_dp/config/DeepSeek-V4-flash-w8a8-PD-prefix.yaml` | 仅 AIME2025 精度 |
| S2 | `DeepSeek_V4_Flash_A5`（`a5.single_node`） | a5 单机 8 卡（BF16，DP8，KV fp8） | `nightly/single_node/models/configs/DeepSeek_V4_Flash_A5.yaml` | 6 个 test_case：GPQA 精度、refusal sampling 04/05/06（性能）、短序列、1M |
| S3 | `DeepSeek-V4-Flash-W8A8-A3` | a3 单机 16 卡（W8A8，DP4×TP4） | `nightly/single_node/models/configs/DeepSeek-V4-Flash-W8A8-A3.yaml` | GPQA（86.36）+ GSM8K prefix0 性能（181.88） |
| S4 | `acceptace_rate_deepseekv4-flash-w8a8-mtp` | a3 单机 8 卡（TP8，MTP=3） | `nightly/single_node/models/configs/acceptace_rate_deepseekv4-flash-w8a8-mtp.yaml` | MTP3 接受率（0.925）+ perf（90）×2 case |

> a5 无多机 nightly 入口（`a5.multi_node` 整段注释），故 a5 池化落在单机。a3 单机另有 `DeepSeek-V4-Flash-W8A8-A3-Breakable.yaml` 存在但未注册 CI，与本方案无关。

## 附录 B：关键文件路径

| 用途 | 路径 |
|---|---|
| 待改 4 个 yaml | 见 §2.2 改动清单 |
| nightly CI 注册（不动） | `.github/workflows/configs/nightly_config.yaml` |
| kv_pool 配置解析 | `tests/e2e/common/kv_pool/config.py` |
| 单机 runner（case 生命周期 + 池服务部署） | `tests/e2e/nightly/single_node/models/scripts/test_single_node.py` / `kv_pool_runtime.py` |
| 池 lookup HBM 截流逻辑 | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` / `pool_worker.py` |
| AscendStoreConnector / MultiConnector | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py` / `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`（勘误：MultiConnector **不在** kv_pool/ascend_store 目录下） |
| 多机模板文档 | `tests/e2e/nightly/multi_node/external_dp/config/template.md` |
| weekly 先例 | `tests/e2e/weekly/multi_node/external_dp/config/DSv4-Flash-Memcache-PD-layerwise.yaml` / `tests/e2e/weekly/single_node/configs/Qwen3.5-397B-Memcache-perf.yaml` |

## 附录 C：术语表

| 术语 | 含义 |
|---|---|
| test_case | 单机 yaml 里一条服务启动配置；一个 test_case = 独立一轮"起服务→跑考题→停服务"；其 `benchmarks` 可同时挂精度与性能多门考题 |
| 写路径 / 读路径 | 写路径 = prefill 把 KV 写入池；读路径 = 请求命中池并回读 KV |
| 粒度门 | 池查询前置门槛：prompt 长度 floor 到 `cache_transfer_granularity`（组块尺寸 LCM，量级 128~512 token）后若低于门槛即不发起池查询；非 layerwise 与 block_key layerwise 路径有此门，gva layerwise 路径无（§4.2①） |
| HBM 命中截流 | 池 lookup 的机制（**仅非 layerwise 路径**）：本地 prefix cache（HBM）命中优先，池只查 HBM 未覆盖部分；gva layerwise 路径从 block 0 查池、不受此截流且池命中时 force load（§4.2②） |
| kv_pool 配置节 | yaml 配置块（可挂在单个 test_case 上），框架自动拉起 mooncake master 或 memcache MetaService 池服务 |
| 同形对标 | 性能基线取自 weekly 同款服务形态（同模型/拓扑/池化配置挂同门考题），池化开销已含在基线内 |
| 记录式占位 | 新基准首跑前的基线写法（`baseline: 0/1`），跑通后以实测回填 |
