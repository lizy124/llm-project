# weekly e2e 测试详情

## 一、与 nightly 的关键区别

| 维度 | nightly | weekly |
|---|---|---|
| **频率** | 每天（cron `0 2 * * *`） | 每周日（cron `0 2 * * 0`） |
| **平台** | A2、A3 | A2、A3、**310P** |
| **A2 测试内容** | 单节点模型服务 | **仅精度测试**（accuracy-group），无模型服务 |
| **A3 测试内容** | 单节点 + 多节点模型服务 | 单节点 + 多节点 + 引擎功能测试 |
| **测试脚本复用** | 自有 test_single_node.py / test_multi_node.py | 复用 nightly 的测试脚本，仅指向不同 config 目录 |
| **并发限制** | A3 最多 5 个并行 | A3 multi_node 最多 2 个，double_node 最多 3 个，single_node 最多 7 个 |

---

## 二、目录结构

```
tests/e2e/weekly/
├── single_node/
│   ├── configs/                           # 17 个 YAML（单节点模型服务）
│   │   ├── DeepSeek-V3.2-W8A8_A3_weekly.yaml
│   │   ├── GLM-5.yaml
│   │   ├── GLM-5_1-W8A8_A3_weekly.yaml
│   │   ├── Kimi-K2.5.yaml
│   │   ├── Kimi-K2.5-32k-512.yaml
│   │   ├── MiniMax-M2.5-W8A8-A3.yaml
│   │   ├── MiniMax-M2.5-w8a8-QuaRot-A3.yaml
│   │   ├── Qwen2.5-VL-7B-Instruct-EPD.yaml      # EPD (Encoder-Prefill-Decoder) 分离
│   │   ├── Qwen3-8B-w8a8sc-310p.yaml             # 310P 平台
│   │   ├── Qwen3-14B-w8a8sc-310p.yaml            # 310P 平台
│   │   ├── Qwen3-32B-w8a8sc-310p.yaml            # 310P 平台
│   │   ├── Qwen3-32B.yaml                        # ★ 未调度
│   │   ├── Qwen3.5-27B-w8a8-A3.yaml
│   │   ├── Qwen3.5-122B-A10B-W8A8-A2.yaml        # ★ 未调度（A2）
│   │   ├── Qwen3.5-122B-A10B-W8A8-A3.yaml
│   │   ├── Qwen3.5-397B-A17B-W8A8-mtp-A3.yaml    # ★ 未调度
│   │   └── Qwen3.5-397B-A17B-W8A8-mtp-A3_weekly.yaml
│   ├── engine_func_test_robot/             # 引擎功能测试（pytest 驱动）
│   │   ├── conftest.py                     # 拉起 Qwen3-VL-30B-A3B，TP=2
│   │   ├── tests/
│   │   │   ├── test_temperature.py         # temperature 参数（0.0/0.7/2.0/非法值）
│   │   │   ├── test_top_p.py               # top_p 参数（0.1/0.9/1.0/非法值）
│   │   │   ├── test_top_k.py               # top_k 参数
│   │   │   ├── test_max_tokens.py          # max_tokens 参数
│   │   │   ├── test_max_completion_tokens.py
│   │   │   ├── test_stop.py                # stop 字符串
│   │   │   ├── test_frequency_penalty.py   # frequency_penalty 参数
│   │   │   ├── test_presence_penalty.py    # presence_penalty 参数
│   │   │   ├── test_n.py                   # n（返回候选数）
│   │   │   ├── test_logprobs.py            # logprobs 参数
│   │   │   ├── test_tool_choice.py         # tool_choice 参数
│   │   │   ├── test_think_tag.py           # 思考标签
│   │   │   ├── test_content.py             # 响应内容验证
│   │   │   ├── test_context_length.py      # 上下文长度
│   │   │   ├── test_chat_template_kwargs.py
│   │   │   ├── test_request_id.py          # 请求 ID
│   │   │   ├── test_role.py                # 角色参数
│   │   │   └── test_repetition_penalty.py  # repetition_penalty 参数
│   │   └── utility/
│   │       ├── http_client.py              # HTTP 客户端封装
│   │       ├── completion_request.py       # 请求构造
│   │       └── assertion.py                # 断言工具
│   └── models/
│       └── test_qwen3_30b_acc.py           # 模型精度测试
├── multi_node/
│   ├── external_dp/                        # 多节点测试（外部 routing 协调）
│   │   └── config/                         # 17 个 YAML（全部被调度）
│   │       ├── DeepSeek-V4-flash-w8a8-PD.yaml
│   │       ├── DeepSeek_V3.1T_MTP1_PD.yaml
│   │       ├── DeepSeek_V3.1T_MTP1_128K_1K_PD.yaml
│   │       ├── DeepSeek_V3.1T_MTP1_3_5K_1_5K_PD.yaml
│   │       ├── DeepSeek_V3.1T_MTP3_PD.yaml
│   │       ├── DeepSeek_V3.1T_layerwise_PD.yaml
│   │       ├── DeepSeek_V3.2T_MTP2_PD.yaml
│   │       ├── DeepSeek_V3.2T_MTP3_PD.yaml
│   │       ├── GLM_5_1_PD_in32k_bs16-0.yaml
│   │       ├── GLM_5_1_PD_in32k_bs20-90.yaml
│   │       ├── GLM_5_1_PD_in64k_bs16-90.yaml
│   │       ├── Kimi-K2.5-W4A8-16k-1k-TPOT50.yaml
│   │       ├── Kimi-K2.5-W4A8-128k-1k-TPOT50.yaml
│   │       ├── MiniMax-PD-in32k-bs4-1.yaml
│   │       ├── QWEN3_235B_PD.yaml
│   │       ├── QWEN3_235B_PD_3_5K_1_5k.yaml
│   │       └── Qwen-3.5-397B-A17B-W8A8-PD.yaml
│   └── internal_dp/                        # 多节点测试
│       └── config/                         # 3 个 YAML（全部被调度）
│           ├── DeepSeek-V3.yaml
│           ├── DeepSeek-V3_2-W8A8-EP_weekly.yaml
│           └── GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml
```

---

## 三、CI 调度机制

### 3.1 调度入口：3 个独立 workflow

| Workflow | 平台 | 触发时间 | Runner |
|---|---|---|---|
| `schedule_weekly_test_a3.yaml` | A3 | 每周日 10:00（北京时间） | `linux-aarch64-a3-*` |
| `schedule_weekly_test_a2.yaml` | A2 | 每周日 10:00（北京时间） | `linux-aarch64-a2b3-*` |
| `schedule_weekly_test_310p.yaml` | 310P | 每周日 10:00（北京时间） | `linux-aarch64-310p-*` |

### 3.2 测试矩阵来源：`weekly_config.yaml`

与 nightly 的 `nightly_config.yaml` 完全一致的机制，通过 `resolve_nightly_tests.py` 解析 YAML 生成 CI matrix：

```yaml
# .github/workflows/configs/weekly_config.yaml
a2:                              # A2 仅精度测试
  accuracy:
    nightly:
      - name: accuracy-group-2
        model_list:
          - ERNIE-4.5-21B-A3B-PT
          - Molmo-7B-D-0924
          - Llama-3.2-3B-Instruct

a3:                              # A3 完整测试
  multi_node:        # 17 个 external_dp 条目
  double_node:       # 3 个 internal_dp 条目
  single_node:       # 12 个条目（11 YAML + 1 pytest）

310p:                            # 310P 单节点
  single_node:       # 4 个条目（3 YAML + 1 pytest）
```

### 3.3 测试脚本复用

weekly 没有自己的测试入口脚本，**完全复用 nightly 的测试脚本**，仅通过 CI 参数 `config_base_path` 指向不同的 config 目录：

| 测试类型 | 复用脚本 | config_base_path |
|---|---|---|
| 单节点模型服务 | `tests/e2e/nightly/single_node/models/scripts/test_single_node.py` | `tests/e2e/weekly/single_node/configs/` |
| 多节点 external_dp | `tests/e2e/nightly/multi_node/scripts/run.sh` + 模板 | `tests/e2e/weekly/multi_node/external_dp/config` |
| 多节点 internal_dp | 同上 | `tests/e2e/weekly/multi_node/internal_dp/config` |
| 引擎功能测试 | 直接 pytest（不经过 nightly 脚本） | `tests/e2e/weekly/single_node/engine_func_test_robot` |
| 模型精度测试 | 直接 pytest | `tests/e2e/weekly/single_node/models/test_qwen3_30b_acc.py` |

---

## 四、测试分类详解

### 4.1 单节点模型服务（YAML 驱动，复用 nightly）

与 nightly 完全相同的机制：YAML 配置 → `test_single_node.py` 解析 → `vllm serve` → OpenAI API 调用 → 可选 benchmarks。

**被调度的 11 个 YAML（A3）：**

| YAML | 模型 | TP | 编译模式 | 基准测试 |
|---|---|---|---|---|
| DeepSeek-V3.2-W8A8_A3_weekly | DeepSeek-V3.2 | 8 | FULL_DECODE_ONLY | perf (4 个场景) |
| GLM-5.yaml | GLM-5-W4A8 | - | - | - |
| GLM-5_1-W8A8_A3_weekly | GLM-5.1-W8A8 | - | - | - |
| Kimi-K2.5.yaml | Kimi-K2.5 | - | - | - |
| Kimi-K2.5-32k-512.yaml | Kimi-K2.5 (32k) | - | - | - |
| MiniMax-M2.5-W8A8-A3 | MiniMax-M2.5 | - | - | - |
| MiniMax-M2.5-w8a8-QuaRot-A3 | MiniMax-M2.5 (QuaRot) | - | - | - |
| Qwen3.5-27B-w8a8-A3 | Qwen3.5-27B | - | - | - |
| Qwen3.5-122B-A10B-W8A8-A3 | Qwen3.5-122B | - | - | - |
| Qwen3.5-397B-A17B-W8A8-mtp-A3_weekly | Qwen3.5-397B (MTP) | 16 | - | perf (3 个场景) |
| Qwen2.5-VL-7B-Instruct-EPD | Qwen2.5-VL-7B (EPD) | 1 | - | - |

**被调度的 3 个 YAML（310P）：**

| YAML | 模型 |
|---|---|
| Qwen3-8B-w8a8sc-310p | Qwen3-8B (W8A8SC) |
| Qwen3-14B-w8a8sc-310p | Qwen3-14B (W8A8SC) |
| Qwen3-32B-w8a8sc-310p | Qwen3-32B (W8A8SC) |

**未调度的 3 个 YAML：**

| YAML | 原因 |
|---|---|
| Qwen3-32B.yaml | weekly_config.yaml 中未添加 |
| Qwen3.5-122B-A10B-W8A8-A2.yaml | A2 平台 weekly 仅跑精度，不跑模型服务 |
| Qwen3.5-397B-A17B-W8A8-mtp-A3.yaml | 被 `_weekly` 版本替代 |

### 4.2 引擎功能测试（pytest 驱动，不经过 nightly 脚本）

在 `conftest.py` 中拉起一个 Qwen3-VL-30B-A3B 的 vllm 服务（TP=2），然后逐一测试 18 个 API 参数的正确性。

**拉起方式（独立于 YAML 配置）：**

```python
# conftest.py
server_args = [
    "--served-model-name", "auto",
    "--max-model-len", "65536",
    "--tensor-parallel-size", "2",
    "--enable-expert-parallel",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
    ...
]

@pytest.fixture(scope="session")
def api_client(request):
    model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    with RemoteOpenAIServer(model, server_args, ...) as server:
        yield HTTPClient(base_url=server.url_root)
```

**测试的 18 个参数：**

| 参数 | 测试文件 | 测试内容 |
|---|---|---|
| temperature | `test_temperature.py` | 合法值 0.0/0.7/2.0、非法值、streaming、组合 top_p |
| top_p | `test_top_p.py` | 合法值 0.1/0.9/1.0、非法值、streaming、组合 |
| top_k | `test_top_k.py` | 边界值、非法值 |
| max_tokens | `test_max_tokens.py` | 边界值、超限 |
| max_completion_tokens | `test_max_completion_tokens.py` | 同上 |
| stop | `test_stop.py` | 单/多停止词、streaming |
| frequency_penalty | `test_frequency_penalty.py` | 合法范围、非法值 |
| presence_penalty | `test_presence_penalty.py` | 合法范围、非法值 |
| n | `test_n.py` | 多候选、streaming |
| logprobs | `test_logprobs.py` | logprobs/top_logprobs、streaming |
| tool_choice | `test_tool_choice.py` | auto/none/required/指定函数 |
| think_tag | `test_think_tag.py` | 思考标签输出 |
| content | `test_content.py` | 响应内容非空 |
| context_length | `test_context_length.py` | 上下文长度 |
| chat_template_kwargs | `test_chat_template_kwargs.py` | 模板参数 |
| request_id | `test_request_id.py` | 请求 ID 传递 |
| role | `test_role.py` | system/user/assistant |
| repetition_penalty | `test_repetition_penalty.py` | 合法范围、非法值 |

引擎功能测试在 A3 和 310P 平台上都会执行，共用同一套代码但使用不同的 runner。

### 4.3 模型精度测试（独立 pytest）

`test_qwen3_30b_acc.py` 是 weekly 独有的精度测试，与 nightly 的 YAML 驱动模式完全不同。

**特点：**

- 直接在测试代码中拉起 vllm 服务（不依赖 YAML）
- 同时启动 MooncakeLauncher
- 配置 speculative decoding（Eagle3 推测解码）
- 测试完成后自动运行 aisbench 精度基准测试

```python
# test_qwen3_30b_acc.py 关键配置
mooncake_json = {
    "local_hostname": "localhost",
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "global_segment_size": 30000000000,
}

kv_transfer_config = {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",                    # ★ 既是 producer 也是 consumer
    "kv_connector_extra_config": {
        "register_buffer": True,
        "use_layerwise": False,
        "mooncake_rpc_port": "0",
    },
}

server_args = [
    "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
    "--speculative-config", json.dumps(speculative_config),  # Eagle3
    "--kv-transfer-config", json.dumps(kv_transfer_config),
    ...
]

with (
    MooncakeLauncher(mooncake_port, mooncake_metrics_port),
    RemoteOpenAIServer(model, server_args, ...) as server,
):
    # 先跑 2 轮功能验证
    # 再跑 aisbench 精度基准测试
    run_aisbench_cases(model, port, aisbench_cases)
```

### 4.4 A2 精度测试

A2 的 weekly 测试与 A3 完全不同——**不跑模型服务，只跑精度测试**：

```yaml
# weekly_config.yaml
a2:
  accuracy:
    nightly:
      - name: accuracy-group-2
        os: linux-aarch64-a2b3-1
        model_list:
          - ERNIE-4.5-21B-A3B-PT
          - Molmo-7B-D-0924
          - Llama-3.2-3B-Instruct
```

### 4.5 多节点测试

weekly 的多节点测试全部以 external_dp 为主（17 个）。

**external_dp（17 个，全部被调度）：**

| 类别 | YAML 数量 | 代表 |
|---|---|---|
| DeepSeek V3.1T MTP 变体 | 5 | MTP1/3, 128K/3.5K, layerwise |
| DeepSeek V3.2T MTP 变体 | 2 | MTP2, MTP3 |
| DeepSeek V4 Flash | 1 | w8a8-PD |
| GLM 5.1 PD 变体 | 3 | in32k/in64k, bs16/bs20 |
| Kimi K2.5 EP | 2 | 16k/128k, TPOT50 |
| MiniMax PD | 1 | in32k-bs4 |
| Qwen3 235B PD | 2 | 标准/3.5K |
| Qwen3.5 397B PD | 1 | W8A8 |

**internal_dp（3 个，全部被调度）：**

| YAML | 节点数 | 特点 |
|---|---|---|
| DeepSeek-V3.yaml | 2 | 标准 DeepSeek V3 |
| DeepSeek-V3_2-W8A8-EP_weekly.yaml | 2 | DeepSeek V3.2 W8A8 EP |
| GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml | 2 | MooncakeLayerwise KV 传输 |

---

## 五、多节点 YAML 格式差异

weekly 的多节点 YAML 同时存在两种格式：

### 5.1 旧格式（`disaggregated_prefill` 字段，与 nightly 相同）

```yaml
# GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml
disaggregated_prefill:
  enabled: true
  prefiller_host_index: [0]
  decoder_host_index: [1]

deployment:
  - envs: ...
    server_cmd: > ...
  - envs: ...
    server_cmd: > ...
```

### 5.2 新格式（`routing` 字段，external_dp 专用）

```yaml
# DeepSeek_V3.1T_layerwise_PD.yaml
routing:
  type: "disaggregated_prefill"
  groups:
    prefiller: [0, 1]
    decoder: [2, 3]

config:
  - node_index: 0
    port_start: 7100
    dp_size: 2
    tp_size: 8
    ...
```

新格式更灵活，支持精确的端口分配、DP rank 指定等。

---

## 六、总结

| 维度 | 详情 |
|---|---|
| **频率** | 每周日 10:00（北京时间） |
| **CI 入口** | 3 个独立 workflow（A2/A3/310P） |
| **测试矩阵** | `weekly_config.yaml`，通过 `resolve_nightly_tests.py` 解析 |
| **测试脚本** | 复用 nightly 的 `test_single_node.py` / `run.sh`，仅指向不同 config 目录 |
| **单节点** | 11 个 YAML（A3）+ 3 个 YAML（310P），复用 nightly 机制 |
| **引擎功能测试** | 18 个参数级测试，独立 conftest 拉起 Qwen3-VL-30B，A3 + 310P 均跑 |
| **模型精度** | `test_qwen3_30b_acc.py`，含 Eagle3 推测解码 |
| **多节点** | 20 个 YAML（17 external_dp + 3 internal_dp），全部在 A3 |
| **A2 测试** | 仅精度测试（3 个模型），无模型服务 |
| **310P 测试** | 引擎功能测试 + 3 个 Qwen3 W8A8SC 单节点模型 |