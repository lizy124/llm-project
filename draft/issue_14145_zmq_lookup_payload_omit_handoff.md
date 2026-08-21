# Issue #14145 ZMQ Lookup Payload 裁剪验证交接文档

## 当前任务目标

验证 fork 分支中关于 issue #14145 的 4 个提交：

```text
https://github.com/ChenZhuo888/vllm-ascend/tree/test/zmq-lookup-payload-omit
```

目标是确认该 PR 是否真正实现：scheduler 已经知道 HBM prefix 命中时，ZMQ lookup 请求只发送 HBM prefix 之后的 suffix hash，并且结果与发送完整 hash 时一致。

需要特别区分三类验证：

1. 代码级 / 单测验证；
2. 协议级真实 ZMQ round-trip 和 payload benchmark；
3. 真实 vLLM Ascend 端到端验证。

目前第 1、2 类已经完成，第 3 类只完成了“基础 vLLM Ascend 最小推理能跑”，还没有完成真正覆盖 `AscendStoreConnector + lookup_hash_mode=suffix` 的端到端验证。

## 一、已完成事项

### 1. 源码分支确认

官方仓库 `vllm-project/vllm-ascend` 没有该测试分支。用户提供 fork 后，已确认分支存在：

```bash
git -c http.proxy=http://127.0.0.1:7897 \
    -c https.proxy=http://127.0.0.1:7897 \
    ls-remote --heads \
    https://github.com/ChenZhuo888/vllm-ascend.git \
    test/zmq-lookup-payload-omit
```

结果：

```text
79e8d276001ce4c69416c8f533f04cd36b8a4cff    refs/heads/test/zmq-lookup-payload-omit
```

本地验证工作树：

```text
/tmp/vllm-ascend-pr
```

克隆命令：

```bash
git -c http.proxy=http://127.0.0.1:7897 \
    -c https.proxy=http://127.0.0.1:7897 \
    clone --branch test/zmq-lookup-payload-omit \
    --single-branch \
    https://github.com/ChenZhuo888/vllm-ascend.git \
    /tmp/vllm-ascend-pr
```

目标 4 个提交已确认：

```text
79e8d276 type unified
59e61d05 test(ascend_store): add coverage for suffix lookup hash omission
7e56764e type unified
b8a9c3c2 feat(kv-pool): support suffix hash lookup with legacy fallback
```

### 2. 静态检查和单测

已执行：

```bash
python -m compileall -q /tmp/vllm-ascend-pr/vllm_ascend \
    /tmp/vllm-ascend-pr/tests/ut/distributed/ascend_store
```

结果：通过。

已执行：

```bash
git -C /tmp/vllm-ascend-pr diff --check HEAD~4..HEAD
```

结果：通过。

分文件测试结果：

```text
test_pool_scheduler.py: 78 passed
test_ascend_store_connector.py: 29 passed
test_pool_worker.py: 110 passed
test_coordinator.py: 13 passed
```

目录级测试结果：

```bash
python -m pytest -q tests/ut/distributed/ascend_store
```

```text
402 passed, 10 skipped, 14 warnings in 2.76s
```

结论：该 PR 在 `ascend_store` 单元测试范围内是通过的。

### 3. 关键代码路径复核

关键文件：

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py
```

已确认的关键实现：

1. `LookupHashMode` 定义了 `FULL` 和 `SUFFIX`。
2. scheduler 从 `kv_connector_extra_config["lookup_hash_mode"]` 读取配置，默认 `full`。
3. `LookupKeyClient.lookup()` 新增 mode frame。
4. `LookupKeyServer` 固定从 `all_frames[3]` 解析 `LookupHashMode`，从 `all_frames[4:]` 解析 hashes。
5. worker fallback 路径在 `SUFFIX` 模式下使用 `token_offset = hbm_hit_tokens`。
6. worker fallback 在 `SUFFIX + hbm_hit_tokens > 0` 时会先插入 HBM 边界，再判断 `group_hits` 是否为空。
7. coordinator 路径使用 `HBMCachedBlockHashList` 表示 HBM prefix marker，保留逻辑 block index。

### 4. 已推翻 strict_review.md 的一个 P1

`draft/issue_14145_zmq_lookup_payload_omit_strict_review.md` 中曾认为：

```text
fallback SUFFIX 路径会丢失 HBM 前缀命中，首个 suffix miss 返回 0
```

当前验证结论：该判断不成立。

依据：

1. 当前代码中 `group_hits.insert(0, hbm_hit_tokens)` 在 `if not group_hits: return 0` 之前。
2. worker 相关测试通过。
3. `ascend_store` 目录级测试通过。
4. 空 suffix smoke test 结果返回 `64`，保留了 HBM prefix。

### 5. 真实 ZMQ 协议 round-trip

已做真实 `LookupKeyServer + LookupKeyClient` round-trip，worker 使用 stub。

结果：

```text
result= 42
call= (64, ['aabb', 'ccdd'], [0, 1], False, 16, <LookupHashMode.SUFFIX: 'suffix'>)
```

结论：新协议同版本 round-trip 是可用的，`lookup_hash_mode=SUFFIX` 能传到 worker。

### 6. 旧协议兼容性验证

已构造旧格式 request：

```text
[token_len, kv_group_ids, hbm_hit_tokens, hashes]
```

发送给新 server 后，client `recv()` 超时：

```text
Again Resource temporarily unavailable
```

结论：当前 server 没有旧格式 fallback。新协议不是 wire-compatible 扩展，而是强制新增 frame 的协议变更。

需要继续关注：

1. old client -> new server；
2. new client -> old server；
3. malformed mode；
4. 缺 frame；
5. server 是否应该返回明确错误而不是让 client 等待超时。

### 7. 协议级性能 benchmark

新增脚本：

```text
/tmp/vllm-ascend-pr/benchmarks/scripts/benchmark_zmq_lookup_payload.py
```

脚本功能：

1. 计算 `FULL` / `SUFFIX` encoded multipart payload bytes；
2. 启动真实 `LookupKeyServer` 和 `LookupKeyClient`；
3. 测本机 ZMQ REQ/REP round-trip latency；
4. worker 使用 stub，不依赖 Ascend NPU 或真实 AscendStore。

脚本语法检查通过：

```bash
python -m compileall -q benchmarks/scripts/benchmark_zmq_lookup_payload.py
```

已跑出的数据：

#### 16K tokens，50% HBM hit

```text
tokens=16384 block_size=16 hbm_hit_tokens=8192 full_hashes=1024 suffix_hashes=512
  payload_bytes: full=67602 suffix=33812 saved=33790 saved_pct=49.98
  rtt_us_full: mean=260.56 p50=257.52 p95=274.82 p99=288.13
  rtt_us_suffix: mean=184.27 p50=182.67 p95=197.50 p99=212.68
```

#### 64K tokens，50% HBM hit

```text
tokens=65536 block_size=16 hbm_hit_tokens=32768 full_hashes=4096 suffix_hashes=2048
  payload_bytes: full=270354 suffix=135188 saved=135166 saved_pct=50.00
  rtt_us_full: mean=732.07 p50=727.70 p95=751.35 p99=785.48
  rtt_us_suffix: mean=437.25 p50=435.77 p95=451.45 p99=469.13
```

#### 64K tokens，75% HBM hit

```text
tokens=65536 block_size=16 hbm_hit_tokens=49152 full_hashes=4096 suffix_hashes=1024
  payload_bytes: full=270354 suffix=67604 saved=202750 saved_pct=74.99
  rtt_us_full: mean=766.89 p50=765.33 p95=793.26 p99=819.40
  rtt_us_suffix: mean=268.63 p50=264.81 p95=290.87 p99=328.01
```

结论：协议级数据证明 payload bytes 和本机 ZMQ RTT 都有明显下降，并且收益随 HBM prefix 比例增加而增加。

注意：这不是端到端 TTFT，不包含真实 AscendStore 查询，不包含模型 forward。

## 二、端到端验证当前进展

### 1. NPU 环境确认

环境中存在 Ascend 设备：

```bash
which npu-smi
npu-smi info
```

结果显示有 16 个 Ascend910 chip，且 `torch_npu` 可用：

```text
torch: FOUND
torch_npu: FOUND
vllm: FOUND
vllm_ascend: FOUND
npu available True
device count 16
```

### 2. 释放设备前的状态

最初 `npu-smi info` 显示所有 chip 的 HBM 基本被占满，且进程不在当前容器内可见。

后来用户处理后，设备释放。释放后状态：

```text
No running processes found in NPU 0..7
HBM usage roughly 2.8GB - 3.1GB / 65536MB per chip
```

### 3. 最小 vLLM Ascend 离线推理尝试

第一次尝试命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
PYTHONPATH=/tmp/vllm-ascend-pr:$PYTHONPATH \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(
    model='/mnt/weight/Qwen3-0.6B',
    trust_remote_code=True,
    enforce_eager=True,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.80,
)
outputs = llm.generate(['hello'], SamplingParams(max_tokens=4, temperature=0))
print('OUTPUT:', outputs[0].outputs[0].text)
PY
```

失败原因：

```text
RuntimeError: aclnnAddRmsNormBias or aclnnAddRmsNormBiasGetWorkspaceSize not in libopapi.so, or libopapi.sonot found.
```

判断：设备已释放后，失败点不是显存，而是当前 CANN/libopapi 自定义算子支持不完整。

### 4. 禁用 custom ops 后最小推理成功

通过设置：

```bash
VLLM_BATCH_INVARIANT=1
```

触发 `enable_custom_op()` 返回 false，避开缺失的 `npu_add_rms_norm_bias` custom op。

成功命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
VLLM_BATCH_INVARIANT=1 \
PYTHONPATH=/tmp/vllm-ascend-pr:$PYTHONPATH \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(
    model='/mnt/weight/Qwen3-0.6B',
    trust_remote_code=True,
    enforce_eager=True,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.80,
)
outputs = llm.generate(['hello'], SamplingParams(max_tokens=4, temperature=0))
print('OUTPUT:', outputs[0].outputs[0].text)
PY
```

成功输出：

```text
OUTPUT: Question = "Hello
```

结论：基础 vLLM Ascend 离线推理端到端已经跑通，但这还没有覆盖 `AscendStoreConnector`、`LookupKeyClient/Server`、`lookup_hash_mode=suffix`。

### 5. 已跑通的最小 AscendStoreConnector e2e

新增了一个临时探针脚本：

```text
/tmp/vllm-ascend-pr/benchmarks/scripts/e2e_ascend_store_suffix_lookup_probe.py
```

它会启动 `mooncake_master` 和 `vllm.entrypoints.openai.api_server`，并配置：

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "lookup_rpc_port": "<port>",
    "lookup_hash_mode": "suffix",
    "use_layerwise": false
  }
}
```

已验证：

1. server 能启动并通过 `/health`；
2. 连续两次同一个长 prompt 请求可以完成；
3. 服务日志里能看到 `kv_transfer_config` 已携带 `lookup_hash_mode='suffix'`；
4. `LookupKeyServer -> KVPoolWorker.lookup_scheduler()` 的真实路径被命中；
5. 目前日志里看到的 lookup 结果仍是 `hit_tokens=0`，还没拿到“HBM prefix 命中后只发送 suffix hashes”的正向证据。

这意味着最小 e2e 已经把 connector 和 suffix 模式接通，但还没有完成真正的命中型验证。

## 三、目前还没完成的端到端验证

尚未完成真正覆盖 PR 的端到端验证，即：

1. 启动带 `AscendStoreConnector` 的 vLLM 服务或离线推理；
2. 配置 `lookup_hash_mode=suffix`；
3. 让请求触发 prefix cache / HBM hit；
4. 触发 scheduler -> worker lookup server 的 ZMQ lookup；
5. 记录真实请求级 TTFT 或 lookup latency；
6. 对比 `lookup_hash_mode=full` 和 `lookup_hash_mode=suffix`。

换句话说：目前完成的是“基础推理端到端”和“lookup 协议级端到端”，但还没有完成“PR 功能端到端”。

## 四、下一步计划

### Step 1：找最小 AscendStoreConnector e2e 配置

优先从以下路径查现成配置：

```text
/tmp/vllm-ascend-pr/docs/source/user_guide/feature_guide/kv_pool.md
/tmp/vllm-ascend-pr/tests/e2e
/tmp/vllm-ascend-pr/examples
/home/lizhongyang/llm-project/transfer_data/refactor/kvpool
```

已经读过 `kv_pool.md`，其中主要是 Mooncake/Memcache/Yuanrong 后端部署说明。需要进一步确定当前环境中哪个后端可用。

待确认：

```bash
python - <<'PY'
import importlib.util
for name in ['mooncake', 'mooncake_transfer_engine', 'pymemcache']:
    print(name, importlib.util.find_spec(name))
PY
```

也需要确认是否存在 `mooncake_master`：

```bash
which mooncake_master || true
```

### Step 2：优先尝试最小本地 backend

如果 mooncake 可用，优先尝试单机单卡配置：

- model: `/mnt/weight/Qwen3-0.6B`
- device: `ASCEND_RT_VISIBLE_DEVICES=0`
- custom op fallback: `VLLM_BATCH_INVARIANT=1`
- max model len: 1024 或 2048
- block size: 128 或默认
- `lookup_rpc_port`: 独立端口，例如 99123
- `lookup_hash_mode`: 先跑 `full`，再跑 `suffix`

如果 mooncake 不可用，考虑 memcache/yuanrong 是否可用；否则只能记录后端依赖缺失。

### Step 3：启动服务并发请求

理想形态是启动 OpenAI API server，然后用重复 prefix 请求测 TTFT。

示例方向，具体配置还要按可用 backend 调整：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
VLLM_BATCH_INVARIANT=1 \
PYTHONPATH=/tmp/vllm-ascend-pr:$PYTHONPATH \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
python -m vllm.entrypoints.openai.api_server \
    --model /mnt/weight/Qwen3-0.6B \
    --port 18000 \
    --trust-remote-code \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --max-model-len 1024 \
    --max-num-batched-tokens 1024 \
    --kv-transfer-config '<待确定 JSON>'
```

然后用 curl 或 Python client 发两类请求：

1. warmup / populate prefix；
2. repeat prefix，触发 lookup。

### Step 4：确认 PR 路径真的被命中

需要在日志里观察或插桩确认：

1. `LookupKeyClient.lookup()` 被调用；
2. `lookup_hash_mode=SUFFIX`；
3. scheduler 发送 suffix hash 数量少于 full hash；
4. `LookupKeyServer` 收到 mode frame；
5. worker `lookup_scheduler()` 收到 `LookupHashMode.SUFFIX`；
6. 返回 token hit 数正确。

可选方法：

- 临时提高 logger level；
- 使用现有 debug log；
- 临时加统计脚本或 monkeypatch，不改生产代码；
- 对比 benchmark 脚本输出和服务日志。

### Step 5：端到端指标

至少记录：

| 模式 | prompt tokens | HBM hit tokens | TTFT | lookup payload bytes | lookup RTT / latency |
|---|---:|---:|---:|---:|---:|
| full | 待测 | 待测 | 待测 | 待测 | 待测 |
| suffix | 待测 | 待测 | 待测 | 待测 | 待测 |

如果 16K / 64K 在小模型或当前资源下不可行，先跑 1K / 4K 做功能端到端，再说明长上下文仍待硬件资源和后端环境支持。

## 五、需要修正已有验证报告的地方

已有报告：

```text
/home/lizhongyang/llm-project/draft/issue_14145_zmq_lookup_payload_omit_container_validation.md
```

需要后续更新：

1. 更醒目地写明“尚未完成 PR 功能端到端验证”。
2. 增加本交接文档中“基础 vLLM Ascend 离线推理已跑通”的记录。
3. 增加 custom op 缺失和 `VLLM_BATCH_INVARIANT=1` workaround。
4. 如果后续跑通 `AscendStoreConnector + lookup_hash_mode=suffix`，再补充最终 e2e 结果。

## 六、当前判断

当前可以明确说：

1. 这 4 个 commit 的单测和 `ascend_store` 目录级测试通过。
2. PR 的新 ZMQ 协议在真实 client/server round-trip 中可用。
3. 协议级 payload bytes 和本机 ZMQ RTT 有明确收益。
4. 基础 vLLM Ascend 离线推理在禁用 custom op 后可以跑通。
5. 最小 `AscendStoreConnector + lookup_hash_mode=suffix` 服务已经跑通，且真实 lookup 路径被命中。

当前不能说：

1. 已完成“HBM prefix 命中后只发送 suffix hashes”的正向功能验证。
2. 已证明 16K/64K 真实请求 TTFT 下降。
3. 已证明真实 AscendStore backend 查询路径收益。
4. 已解决新旧协议兼容问题。

下一位接手时，优先任务是：

> 在已释放 NPU 且 `VLLM_BATCH_INVARIANT=1` 的前提下，找到可用的 AscendStore backend，构造能够明确产生 HBM prefix 命中的请求序列，记录 `LookupKeyClient -> LookupKeyServer -> KVPoolWorker.lookup_scheduler()` 在 `SUFFIX` 模式下的实际 token/hashes 截断行为，并补上命中型 e2e 证据。
