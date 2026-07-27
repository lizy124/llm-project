# nightly e2e 测试详情

## 一、与 pull_request 的关键区别

| 维度 | pull_request | nightly |
|---|---|---|
| **拉起方式** | 进程内 `vllm.LLM()` | 启动真实 HTTP server，OpenAI API 调用 |
| **测试入口** | 独立 `test_*.py` 文件 | YAML 驱动，统一入口 `test_single_node.py` / `test_multi_node.py` |
| **规模** | 1-4 卡 | 单节点（多卡）/ 多节点（跨机） |
| **模型** | 小模型（0.6B-30B） | 大模型（27B-397B，DeepSeek-V4, Kimi-K2.5 等） |
| **平台标记** | `@pytest.mark.e2e_coverage(hardware="A2/A3")` | YAML 中 `ASCEND_A3_ENABLE` 或文件名后缀 `-A2`/`-A3` |
| **基准测试** | 无 | 有（accurancy + performance benchmarks via aisbench） |

---

## 二、拉起机制：YAML 驱动 + HTTP Server 模式

nightly 不直接构造 `vllm.LLM`，而是通过 YAML 配置 → 启动 HTTP server → OpenAI API 调用。

### 2.1 单节点流程

```
YAML 配置 → test_single_node.py 解析 → subprocess 启动 vllm serve → RemoteOpenAIServer → OpenAI API 调用
```

```python
# single_node/models/scripts/test_single_node.py
configs = SingleNodeConfigLoader.from_yaml_cases()  # 加载所有 YAML

async def run_completion_test(config, server):
    client = server.get_async_client()               # OpenAI 客户端
    batch = await client.completions.create(          # 发 HTTP 请求
        model=config.model,
        prompt=config.prompts,
        **config.api_keyword_args,
    )
```

### 2.2 多节点流程

```
YAML 配置 → test_multi_node.py 解析 → 多机启动 vllm serve → ProxyLauncher → OpenAI API 调用
```

YAML 中定义了 `num_nodes`、`npu_per_node`、`disaggregated_prefill` 等，由 `multi_node_config.py` 解析后分发到各节点执行。

---

## 三、目录结构

```
nightly/
├── single_node/
│   ├── models/                     # ★ 模型级测试（YAML 驱动）
│   │   ├── configs/                #   30 个 YAML 配置文件
│   │   └── scripts/
│   │       ├── test_single_node.py #   统一测试入口
│   │       └── single_node_config.py # 配置解析器
│   └── ops/                        # ★ 算子级测试（直接 pytest）
│       ├── singlecard_ops/         #   单卡算子（含 triton/ 子目录）
│       ├── multicard_ops_a2/       #   多卡算子（A2 平台）
│       └── multicard_ops_a3/       #   多卡算子（A3 平台）
├── multi_node/
│   ├── internal_dp/                # 多节点测试（含 PD 分离和纯分布式部署）
│   │   ├── config/                 #   17 个 YAML 配置
│   │   └── scripts/
│   │       ├── test_multi_node.py
│   │       └── multi_node_config.py
│   ├── external_dp/                # 多节点测试（外部 routing 协调）
│   │   ├── config/                 #   1 个 YAML + 模板
│   │   └── scripts/
│   │       ├── test_external_dp.py
│   │       └── runtime.py
│   └── scripts/                    #   通用工具（run.sh, benchmark_results.py）
├── 310p/                           # 310P 平台
│   └── single_node/ops/singlecard_ops/
└── scripts/                        # AOP（自动运维）脚本
```

---

## 四、单节点模型测试（single_node/models）

### 4.1 YAML 配置结构

每个 YAML 包含一个或多个 `test_cases`，每个用例定义：

```yaml
# 以 DeepSeek-V3.2-W8A8.yaml 为例
test_cases:
  - name: "DeepSeek-V3.2-W8A8-TP8-DP2"
    model: "vllm-ascend/DeepSeek-V3.2-W8A8"
    envs:
      ASCEND_A3_ENABLE: "1"           # ← 平台标记
      VLLM_ASCEND_ENABLE_FLASHCOMM1: "1"
      ...
    server_cmd:                       # ← vllm serve 启动命令
      - "--tensor-parallel-size 8"
      - "--data-parallel-size 2"
      - "--enable-expert-parallel"
      - "--quantization ascend"
      ...
    test_content: ["completion"]      # ← 测试类型
    benchmarks:                       # ← 可选：精度/性能基准
      perf:
        case_type: performance
        ...
```

### 4.2 单节点 YAML 按平台分类

#### A2 平台（文件名后缀 `-A2`）

| YAML | 模型 | 特性 |
|---|---|---|
| `MiniMax-M2.5-w8a8-QuaRot-A2.yaml` | MiniMax-M2.5 | QuaRot 量化 |
| `Qwen3-32B-Int8-A2.yaml` | Qwen3-32B-Int8 | Int8 量化 |
| `Qwen3.5-27B-w8a8-A2.yaml` | Qwen3.5-27B | W8A8, MTP |
| `Qwen3.5-397B-A17B-w4a8-mtp-A2.yaml` | Qwen3.5-397B-A17B | W4A8, MTP |

#### A3 平台（文件名后缀 `-A3` 或 `ASCEND_A3_ENABLE`）

| YAML | 模型 | 特性 |
|---|---|---|
| `DeepSeek-V4-Flash-W8A8-A3.yaml` | DeepSeek-V4-Flash | W8A8, Flash |
| `MiniMax-M2.5-w8a8-QuaRot-A3.yaml` | MiniMax-M2.5 | QuaRot 量化 |
| `Qwen3.5-27B-w8a8-A3.yaml` | Qwen3.5-27B | W8A8, MTP |
| `Qwen3.5-122B-A10B-W8A8-A3.yaml` | Qwen3.5-122B-A10B | W8A8 |
| `Qwen3.5-397B-A17B-W8A8-mtp-A3.yaml` | Qwen3.5-397B-A17B | W8A8, MTP |
| `Qwen3.5-397B-A17B-w8a8-mtp-longseq-A3.yaml` | Qwen3.5-397B-A17B | W8A8, MTP, 长序列 |
| `DeepSeek-V3.2-W8A8.yaml` | DeepSeek-V3.2 | W8A8, EP, flashcomm1 (env: A3) |
| `DeepSeek-V3.2-W8A8-DCP.yaml` | DeepSeek-V3.2 | W8A8, DCP (env: A3) |
| `GLM-5.1-W8A8-PrefillMC2.yaml` | GLM-5.1 | W8A8, PrefillMC2 (env: A3) |

#### 无明确平台标记

| YAML | 模型 | 特性 |
|---|---|---|
| `DeepSeek-R1-0528-W8A8.yaml` | DeepSeek-R1-0528 | W8A8 |
| `Gemma4-31B-Dense.yaml` | Gemma4-31B | Dense |
| `Gemma4.yaml` | Gemma4 | MoE |
| `GLM-4.7.yaml` | GLM-4.7 | - |
| `Hy3-preview.yaml` | Hy3-preview | - |
| `Kimi-K2-Thinking.yaml` | Kimi-K2-Thinking | - |
| `Kimi-K2.5.yaml` | Kimi-K2.5 | - |
| `MTPX-DeepSeek-R1-0528-W8A8.yaml` | DeepSeek-R1-0528 | MTP |
| `Prefix-Cache-DeepSeek-R1-0528-W8A8.yaml` | DeepSeek-R1-0528 | Prefix Caching |
| `Prefix-Cache-Qwen3-32B-Int8.yaml` | Qwen3-32B-Int8 | Prefix Caching |
| `Qwen3-235B-A22B-W8A8.yaml` | Qwen3-235B-A22B | W8A8 |
| `Qwen3-30B-A3B-W4A8-llm-compressor.yaml` | Qwen3-30B-A3B | W4A8, llm-compressor |
| `Qwen3-30B-A3B-W8A8.yaml` | Qwen3-30B-A3B | W8A8 |
| `Qwen3-30B-QuaRot-eagle3.yaml` | Qwen3-30B | QuaRot, eagle3 |
| `Qwen3-32B-Int8.yaml` | Qwen3-32B | Int8 |
| `Qwen3-32B-QuaRot-eagle3.yaml` | Qwen3-32B | QuaRot, eagle3 |
| `Qwen3-VL-235B-A22B-Instruct-W8A8.yaml` | Qwen3-VL-235B-A22B | W8A8, VL |
| `Qwen3-VL-32B-Instruct-W8A8.yaml` | Qwen3-VL-32B | W8A8, VL |

### 4.3 单节点测试类型

每个 YAML 的 `test_content` 可选：

| test_content | 说明 |
|---|---|
| `completion` | 基础文本补全 |
| `chat_completion` | 对话补全 |
| `image` | 多模态图片请求 |
| `benchmark` | 精度/性能基准测试（aisbench） |

### 4.4 一个服务同时验证多个特性（不是一对一）

nightly 不是"一个服务测一个特性"，而是**一个服务 = 一个模型的全栈配置 = 所有相关特性同时开启**。

#### 4.4.1 代码证据：`_extract_features` 返回列表

```python
# test_single_node.py:224-268
def _extract_features(server_cmd, envs) -> list[str]:
    features: list[str] = []       # ← 列表，不是单个值

    # 从 --additional-config 提取
    if additional.get("enable_weight_nz_layout"):
        features.append("weight_nz_layout")
    if tc.get("enabled"):
        features.append("torchair_graph")
    if asc.get("enabled"):
        features.append("ascend_scheduler")

    # 从 --compilation-config 提取
    if compilation.get("cudagraph_mode"):
        features.append("aclgraph")

    # 从 --speculative-config 提取
    if speculative:
        features.append(speculative.get("method", "speculative"))

    # 从 --enable-expert-parallel 提取
    if "--enable-expert-parallel" in cmd_list:
        features.append("expert_parallel")

    # 从环境变量批量提取（6 个 env var → 6 个 feature）
    for env_key, feature_name in _FEATURE_ENVS.items():
        val = str(envs.get(env_key, "0"))
        if val not in ("0", "", "false", "False"):
            features.append(feature_name)       # ← 遍历追加

    return features  # ← 返回一个列表，可能包含多个特性
```

#### 4.4.2 代码证据：`_FEATURE_ENVS` 定义了 6 个特性

```python
# test_single_node.py:187-194
_FEATURE_ENVS = {
    "VLLM_ASCEND_ENABLE_FLASHCOMM": "flashcomm",
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": "flashcomm1",
    "VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "topk_optimize",
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "matmul_allreduce",
    "VLLM_ASCEND_ENABLE_MLAPO": "mlapo",
    "VLLM_ASCEND_ENABLE_FUSED_MC2": "fused_mc2",
}
```

#### 4.4.3 实例：一个 YAML 同时开启 8+ 个特性

以 `DeepSeek-V3.2-W8A8.yaml` 为例，一个 `test_case` 同时启用：

```yaml
envs:
  ASCEND_A3_ENABLE: "1"                          # → A3 平台
  VLLM_ASCEND_ENABLE_FLASHCOMM1: "1"             # → flashcomm1
  VLLM_ASCEND_ENABLE_MLAPO: "1"                  # → mlapo
server_cmd:
  - "--tensor-parallel-size 8"                   # → TP8
  - "--data-parallel-size 2"                     # → DP2
  - "--enable-expert-parallel"                   # → EP
  - "--quantization ascend"                      # → W8A8
  - "--speculative-config {...}"                 # → mtp (speculative)
  - "--additional-config {torchair_graph...}"    # → torchair_graph
  - "--compilation-config {cudagraph_mode...}"   # → aclgraph
```

**一个服务同时开启了 TP8 + DP2 + EP + W8A8 + flashcomm1 + mlapo + mtp + torchair_graph + aclgraph，共 9+ 个特性。**

### 4.5 验证工作的具体流程（核心）

验证工作分两步，由 `test_single_node` 函数统一编排：

```python
# test_single_node.py:410-436
@pytest.mark.parametrize("config", configs, ids=[config.name for config in configs])
async def test_single_node(config: SingleNodeConfig) -> None:
    # 标准 OpenAI 服务模式
    with RemoteOpenAIServer(
        model=config.model,
        vllm_serve_args=config.server_cmd,    # ← 所有特性都在这里启动
        env_dict=config.envs,
        auto_port=False,
    ) as server:
        await _dispatch_tests(config, server)  # ① 功能验证
        _run_benchmarks(config, config.server_port)  # ② 基准验证
```

#### 4.5.1 第一步：功能正确性验证（`_dispatch_tests`）

`_dispatch_tests` 读取 YAML 中的 `test_content` 列表，逐个分发到对应的 handler：

```python
# test_single_node.py:161-172
TEST_HANDLERS = {
    "completion": run_completion_test,           # → 发送 prompt，断言返回非空
    "image": run_image_test,                     # → 发送图片请求
    "chat_completion": run_chat_completion_test, # → 发送 chat 请求
    "check_rank0_process_count": run_check_rank0_process_count,  # → 检查进程数
}
```

**每个 handler 做的事情极其简单，只有一个断言：**

```python
# test_single_node.py:30-39
async def run_completion_test(config, server):
    client = server.get_async_client()
    batch = await client.completions.create(
        model=config.model,
        prompt=config.prompts,              # prompt 来自 YAML
        **config.api_keyword_args,          # 额外参数（max_tokens 等）
    )
    choices = batch.choices
    assert choices[0].text, "empty response"  # ← 唯一断言：返回非空就行
```

```python
# test_single_node.py:42-45 — 图片请求同理
async def run_image_test(config, server):
    from tools.send_mm_request import send_image_request
    send_image_request(config.model, server)   # 发图片请求，内部断言非空

# test_single_node.py:48-55 — chat 请求同理
async def run_chat_completion_test(config, server):
    from tools.send_request import send_v1_chat_completions
    send_v1_chat_completions(
        config.prompts[0], model=config.model,
        server=server, request_args=config.api_keyword_args,
    )
```

**关键结论：功能验证不做任何特性级别的验证。** 只验证"这个模型 + 这套配置能正常启动 → 能正常回复 → 返回非空"。具体的特性正确性由单元测试和 PR 测试保证，nightly 做的是端到端集成验证。

#### 4.5.2 第二步：基准验证（`_run_benchmarks`）

```python
# test_single_node.py:396-409
def _run_benchmarks(config, port):
    benchmark_keys = [k for k, v in config.benchmarks.items() if v]
    aisbench_cases = [config.benchmarks[k] for k in benchmark_keys]
    if not aisbench_cases:
        return

    result = run_aisbench_cases(                # 调 aisbench 跑基准
        model=config.model,
        port=port,
        aisbench_cases=aisbench_cases,
    )
    _save_benchmark_results_json(config, benchmark_keys, result)  # 保存结果 JSON

    if "benchmark_comparisons" in config.test_content:
        run_benchmark_comparisons(config, result)  # 多任务间对比断言
```

YAML 中的 `benchmarks` 字段定义基准条件：

```yaml
benchmarks:
  accuracy:
    case_type: accuracy
    dataset_path: "datasets/..."
    baseline: 0.95
    threshold: 0.01           # 精度不能低于 baseline - threshold
  perf:
    case_type: performance
    dataset_conf: "..."
    baseline: 1000
    threshold: 0.9            # 吞吐不能低于 baseline * threshold
```

每个 benchmark 结果与 baseline/threshold 比较，通过才 pass：

```python
# test_single_node.py:283-287
def _task_passed(case_config, result):
    case_type = case_config.get("case_type")
    baseline = case_config.get("baseline")
    threshold = case_config.get("threshold")

    if case_type == "accuracy" and isinstance(result, (int, float)):
        return abs(float(result) - float(baseline)) <= float(threshold)
    if case_type == "performance" and isinstance(result, list):
        throughput_val = float(throughput_str.replace("token/s", "").strip())
        return throughput_val >= float(threshold) * float(baseline)
```

最终结果保存为 JSON，包含模型名、硬件、dtype、特性列表、所有指标的 pass/fail：

```python
# test_single_node.py:360-392
output = {
    "model_name": config.model,
    "hardware": _extract_hardware(runner),
    "dtype": _extract_dtype(config),
    "feature": _extract_features(config.server_cmd, config.envs),  # ← 提取的特性列表
    "vllm_version": vllm.__version__,
    "tasks": tasks,          # 每个 benchmark 的指标和 pass/fail
    "serve_cmd": _build_serve_cmd(config),
    "environment": _filter_environment(config.envs),
    "pass_fail": "pass" if passed else "fail",
}
```

#### 4.5.3 验证流程总结

```
启动 vllm serve（所有特性一起开，一个服务）
    │
    ├─ ① 功能验证：发几个 OpenAI API 请求 → 断言返回非空
    │   （不验证任何具体特性的正确性，只验证"没崩、能回复"）
    │
    └─ ② 基准验证（可选）：跑 aisbench → 对比精度/吞吐基线
        （也不验证具体特性，只验证整体指标达标）
```

---

## 五、多节点测试（multi_node）

### 5.1 internal_dp

| YAML | 模型 | 平台 |
|---|---|---|
| `DeepSeek-V3_2-W8A8-EP.yaml` | DeepSeek-V3.2-W8A8 | A3 (env) |
| `GLM5_1-W8A8-EP.yaml` | GLM-5.1-W8A8 | A3 (env) |
| `Qwen3-235B-disagg-pd.yaml` | Qwen3-235B | 无标记 |
| `Qwen3-VL-235B-disagg-pd.yaml` | Qwen3-VL-235B | 无标记 |
| `Qwen3-235B-W8A8.yaml` | Qwen3-235B-W8A8 | 无标记 |
| `Qwen3-235B-W8A8-EPLB.yaml` | Qwen3-235B-W8A8 | 无标记 |
| `Qwen3-235B-W8A8-longseq.yaml` | Qwen3-235B-W8A8 | 无标记 |
| `DeepSeek-R1-W8A8-EPLB.yaml` | DeepSeek-R1-W8A8 | 无标记 |
| `DeepSeek-R1-W8A8-longseq.yaml` | DeepSeek-R1-W8A8 | 无标记 |
| `DeepSeek-V3.1-BF16.yaml` | DeepSeek-V3.1 | 无标记 |
| `Qwen3-235B-A22B.yaml` | Qwen3-235B-A22B | 无标记 |
| `DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml` | DeepSeek-V3.2 | A3 (env) |
| `GLM5_1-W8A8-A2-dual-nodes.yaml` | GLM-5.1 | A2 (文件名) |
| `GLM5_1-W8A8-A3-dual-nodes.yaml` | GLM-5.1 | A3 (文件名) |
| `GLM5_2-W8A8-A3-dual-nodes.yaml` | GLM-5.2 | A3 (文件名) |
| `Kimi-K2_5-W4A8-A2-dual-nodes.yaml` | Kimi-K2.5 | A2 (文件名) |
| `Qwen3-235B-A22B-A2.yaml` | Qwen3-235B-A22B | A2 (文件名) |

### 5.2 external_dp

| YAML | 模型 | 平台 |
|---|---|---|
| `GLM5_1-W8A8-EP-external.yaml` | GLM-5.1-W8A8 | A3 (env) |

**internal_dp 和 external_dp 的区别**：

| | internal_dp | external_dp |
|---|---|---|
| **配置字段** | `disaggregated_prefill: enabled: true` | `routing.type: "disaggregated_prefill"` |
| **节点管理** | 内部管理，prefiller/decoder 由 YAML 指定 | 外部管理，通过 IP 和端口协调 |
| **测试入口** | `test_multi_node.py` | `test_external_dp.py` |

---

## 六、算子测试（ops）

### 6.1 单卡算子（singlecard_ops）

| 子目录/文件 | 内容 |
|---|---|
| `triton/` | 30+ Triton 算子测试（rope, temperature, min_p, mrope, l2norm, bad_words, rejection_sample 等） |
| 根目录 | 20+ 算子测试（fused_moe, kv_quant_sparse_flash_attention, mla_preprocess, bgmv, gmm_swiglu 等） |

### 6.2 多卡算子（multicard_ops）

| 目录 | 测试内容 |
|---|---|
| `multicard_ops_a2/` | `test_matmul_allreduce_add_rmsnorm.py`（A2 平台） |
| `multicard_ops_a3/` | `test_dispatch_ffn_combine.py` / `bf16` / `w4a8`（A3 平台） |

---

## 七、310P 平台

| 文件 | 内容 |
|---|---|
| `test_chunk_fwd_o_310.py` | chunk forward |
| `test_chunk_gated_delta_rule_fwd_h_aclnn.py` | gated delta rule |
| `test_recurrent_gated_delta_rule_v310.py` | recurrent gated delta rule |

---

## 八、A2/A3 平台如何配置验证（CI 调度层）

### 8.1 核心机制：`nightly_config.yaml` 是单一真相源

YAML 配置文件只是在 `configs/` 目录下的"可用池子"，具体哪些在哪个平台跑，由 `nightly_config.yaml` 决定：

```yaml
# .github/workflows/configs/nightly_config.yaml
a2:                          # ← A2 专属
  single_node:
    test_config:
      - name: qwen3-32b-int8
        config_file_path: Qwen3-32B-Int8-A2.yaml
      - name: Qwen3.5-27B-w8a8-A2
        config_file_path: Qwen3.5-27B-w8a8-A2.yaml
      - name: gemma4
        config_file_path: Gemma4.yaml
      ...
  multi_node:
    test_config:
      - name: multi-node-qwen3-235b-dp
        config_file_path: Qwen3-235B-A22B-A2.yaml
      ...

a3:                          # ← A3 专属（完全独立）
  single_node:
    test_config:
      - name: deepseek-v3-2-w8a8
        config_file_path: DeepSeek-V3.2-W8A8.yaml
      - name: kimi-k2.5
        config_file_path: Kimi-K2.5.yaml
      ...
  multi_node:
    test_config:
      - name: multi-node-deepseek-v3.2-W8A8-EP
        config_file_path: DeepSeek-V3_2-W8A8-EP.yaml
      ...
```

### 8.2 两层过滤保证 A2/A3 完全隔离

#### 第一层：CI 工作流只读取自己平台的 section

```yaml
# schedule_nightly_test_a3.yaml:148-153
# A3 的 CI 只解析 a3.* 路径
MATRIX_OUTPUTS: '{
    "multi_node":"a3.multi_node.test_config",
    "double_node":"a3.double_node.test_config",
    "single_node":"a3.single_node.test_config",
    "multi_card":"a3.multi_card.test_config"
}'
```

```yaml
# schedule_nightly_test_a2.yaml 同理，只解析 a2.* 路径
```

`resolve_nightly_tests.py` 脚本根据 `MATRIX_OUTPUTS` 中的路径（如 `a3.single_node.test_config`）逐级解析 YAML，只提取对应平台下的测试列表。

#### 第二层：硬件 runner 隔离

| | A2 | A3 |
|---|---|---|
| **CI 工作流** | `schedule_nightly_test_a2.yaml` | `schedule_nightly_test_a3.yaml` |
| **runner 标签** | `linux-aarch64-a2b3-*` | `linux-aarch64-a3-*` / `linux-aarch64-nightly-a3-*` |
| **Docker 镜像** | `nightly-ci-*-a2` | `nightly-ci-*-a3` |
| **CANN 版本** | `9.0.1-910b` | 不同（A3 专用） |

### 8.3 A2 和 A3 各自跑什么（实际调度列表）

| 类别 | A2 | A3 |
|---|---|---|
| **单节点模型** | 6 个 YAML：gemma4, gemma4-31b, qwen3-32b-int8, qwen3.5-27b, qwen3.5-397b-w4a8, qwen3-vl-32b | 19 个 YAML：DeepSeek-V3.2/V4/V4-Flash/R1, Kimi-K2/K2-Thinking, GLM-4.7/5.1, MiniMax, Qwen3-235B, Qwen3.5-122B/397B, Qwen3-VL-235B 等 |
| **多节点** | 3 个 YAML：GLM-5.1, Kimi-K2.5, Qwen3-235B（全部是 `-dual-nodes`） | 11 个 YAML：DeepSeek-V3.2 EP, Qwen3-235B disagg/EPLB/longseq, DeepSeek-R1 longseq/EPLB, GLM-5.1/5.2, DeepSeek-V3.1 等 |
| **精度测试** | accuracy-group-1/3/4 + pr-accuracy-group-1/2（小模型精度） | 无独立精度组 |
| **多卡测试** | 无 | 8 个 YAML：qwen3-30b-acc, qwen3.5-27b, qwen3-32b-int8, QuaRot 等 |
| **算子测试** | multicard_ops_a2/ | multicard_ops_a3/ |

### 8.4 总结

**A2 和 A3 是完全独立的测试管线**，不存在"同一个 YAML 在两个平台都跑"的情况。需要在哪个平台跑，就在 `nightly_config.yaml` 的对应 platform section 下添加条目。`tests/e2e/nightly/*/configs/` 目录下的 YAML 只是可用配置的池子，具体哪些被调度、在哪调度，由 `nightly_config.yaml` 决定。

---

## 九、总结

| 维度 | 详情 |
|---|---|
| **拉起方式** | YAML 驱动 → `vllm serve` HTTP server → OpenAI API 调用 |
| **单节点入口** | `test_single_node.py`（统一入口，加载 30 个 YAML） |
| **多节点入口** | `test_multi_node.py`（internal_dp） / `test_external_dp.py`（external_dp） |
| **算子上入口** | 直接 pytest（与 pull_request 相同） |
| **平台标记** | YAML 中 `ASCEND_A3_ENABLE: "1"` 或文件名后缀 `-A2`/`-A3` |
| **基准测试** | 可选 accuracy/performance benchmarks（通过 aisbench） |