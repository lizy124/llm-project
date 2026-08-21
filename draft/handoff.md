# ZMQ lookup suffix 验证交接

## 背景

当前在验证 vLLM-Ascend fork 分支里的 PR：

```text
https://github.com/ChenZhuo888/vllm-ascend/tree/test/zmq-lookup-payload-omit
```

目标是判断该 PR 是否可以合入、有没有功能问题、有没有性能收益。

这个 PR 的核心目标是：当 scheduler 已经知道请求有 HBM prefix 命中时，ZMQ lookup 请求不再发送完整 block hashes，而是只发送 HBM prefix 之后的 suffix hashes，从而减少 scheduler 到 worker lookup server 的 payload 和 RTT。

## 当前结论

目前倾向于：**可以合入**。

理由：

1. `ascend_store` 相关单测已经通过。
2. 新 ZMQ lookup 协议同版本 round-trip 已验证可用。
3. `lookup_hash_mode=suffix` 能传到 worker。
4. 协议级 payload bytes 和 ZMQ RTT 有明确收益。
5. 最小 `AscendStoreConnector + lookup_hash_mode=suffix` vLLM 服务已启动成功，真实 lookup 路径被命中。

但还有一个未完成的补强验证：

> 在真实或半真实 e2e 场景里证明 `hbm_hit_tokens > 0` 时，`suffix` 模式发送的 hashes 少于 `full`，且返回 hit tokens 一致。

这比直接看 TTFT 更重要，因为 TTFT 受模型 forward、NPU 负载、调度抖动、Mooncake put/get 等因素影响；而这个 PR 直接优化的是 lookup 请求 payload。

## 已完成验证

### 1. 分支和 commit

本地工作树：

```text
/tmp/vllm-ascend-pr
```

目标分支：

```text
test/zmq-lookup-payload-omit
```

目标提交：

```text
79e8d276 type unified
59e61d05 test(ascend_store): add coverage for suffix lookup hash omission
7e56764e type unified
b8a9c3c2 feat(kv-pool): support suffix hash lookup with legacy fallback
```

### 2. 单测和静态检查

已通过：

```bash
python -m compileall -q /tmp/vllm-ascend-pr/vllm_ascend \
    /tmp/vllm-ascend-pr/tests/ut/distributed/ascend_store
```

已通过：

```bash
git -C /tmp/vllm-ascend-pr diff --check HEAD~4..HEAD
```

目录级测试结果：

```bash
python -m pytest -q tests/ut/distributed/ascend_store
```

结果：

```text
402 passed, 10 skipped, 14 warnings
```

### 3. 协议 round-trip

已做真实 `LookupKeyServer + LookupKeyClient` round-trip，worker 使用 stub。

结果证明：

- 新协议同版本可用；
- `LookupHashMode.SUFFIX` 能从 client 传到 server，再传给 worker。

### 4. 协议级性能 benchmark

脚本：

```text
/tmp/vllm-ascend-pr/benchmarks/scripts/benchmark_zmq_lookup_payload.py
```

该脚本会：

1. 计算 `FULL` / `SUFFIX` encoded multipart payload bytes；
2. 启动真实 `LookupKeyServer` 和 `LookupKeyClient`；
3. 测本机 ZMQ REQ/REP round-trip latency；
4. worker 使用 stub，不依赖 Ascend NPU 或真实 AscendStore。

已得到的数据：

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

结论：协议级性能收益明确，HBM prefix 越多，payload 和 RTT 下降越明显。

### 5. 最小 vLLM Ascend 推理

环境里有 Ascend 设备，`torch_npu` 可用。

最初基础推理失败在 custom op：

```text
RuntimeError: aclnnAddRmsNormBias or aclnnAddRmsNormBiasGetWorkspaceSize not in libopapi.so
```

通过设置：

```bash
VLLM_BATCH_INVARIANT=1
```

避开缺失 custom op 后，`/mnt/weight/Qwen3-0.6B` 最小离线推理成功。

### 6. 最小 AscendStoreConnector e2e

新增临时探针脚本：

```text
/tmp/vllm-ascend-pr/benchmarks/scripts/e2e_ascend_store_suffix_lookup_probe.py
```

脚本会启动：

- `mooncake_master`
- `vllm.entrypoints.openai.api_server`

并配置：

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
2. 连续请求可以完成；
3. 服务日志里能看到 `lookup_hash_mode='suffix'`；
4. `LookupKeyServer -> KVPoolWorker.lookup_scheduler()` 真实路径被命中。

但当前日志里看到的 lookup 结果仍是：

```text
hit_tokens=0
```

所以它证明了 connector 和 suffix 模式接通，但还没有证明“HBM prefix 命中后只发 suffix hashes”。

## 关于兼容性风险的澄清

之前曾提到“协议变更 + 兼容性风险”，需要准确理解。

这里说的是 vLLM 内部 lookup ZMQ IPC 通道，不是对外 API，也不是 Mooncake 数据同步协议。

- `LookupKeyClient` 在 scheduler 侧；
- `LookupKeyServer` 在 worker 侧；
- 正常一次 vLLM 启动会一起拉起这些进程；
- 它们通常来自同一个 vLLM-Ascend 安装包或镜像。

所以正常部署里一般不会混跑旧 client / 新 server。只要完整重启相关 vLLM 进程，这个不是合入阻塞项。

真正需要注意的是：当前请求 frame 格式确实变了。

旧协议大致是：

```text
[token_len, kv_group_ids, hbm_hit_tokens, hashes]
```

新协议是：

```text
[token_len, kv_group_ids, hbm_hit_tokens, lookup_hash_mode, hashes]
```

响应格式没有变，仍然是一个 hit token 数。

因此更准确的结论是：

> 这是内部 lookup 请求格式变更。标准部署里 client/server 同版本一起启动，不应作为常规合入阻塞项；但不支持内部 lookup client/server 独立混版本运行。

相关说明文档：

```text
/home/lizhongyang/llm-project/draft/zmq_lookup_protocol_compatibility.md
```

这份文档里有些表述偏重“混版本风险”，后续最好按上面的澄清修一下，避免把内部 IPC 风险写成普通外部 RPC 兼容风险。

## 为什么还要补一轮实验

用户追问：“端到端性能收益为什么不验证？”

原因是：当前最小 e2e 没有稳定产生 `hbm_hit_tokens > 0`。

这个 PR 的收益只有在以下条件同时成立时才体现：

```text
请求有 HBM prefix 命中
+
lookup_hash_mode=suffix
+
scheduler 只发送 HBM prefix 后面的 suffix hashes
```

如果 `hbm_hit_tokens=0`，suffix 模式没有前缀可裁，TTFT 对比没有解释价值。

因此下一步不是直接看 TTFT，而是先做更严格、更直接的功能/性能实验。

## 下一步要做的严格实验

目标：证明在 `hbm_hit_tokens > 0` 时：

1. `FULL` 模式发送完整 hashes；
2. `SUFFIX` 模式发送更少 hashes；
3. 两者返回的 hit tokens 一致；
4. `SUFFIX` 的 payload bytes / lookup RTT 更低；
5. 如果条件稳定，再补 TTFT 对比。

### 推荐实验方案 A：半真实协议级严格验证

这是最稳、最快、最直接的方案。

基于现有：

```text
/tmp/vllm-ascend-pr/benchmarks/scripts/benchmark_zmq_lookup_payload.py
```

增强脚本，让 stub worker 不只是返回 `token_len`，而是记录：

- `lookup_hash_mode`
- `token_len`
- `hbm_hit_tokens`
- 收到的 `block_hashes` 数量
- 返回的 hit tokens

然后构造同一个 case：

```text
token_len = 65536
block_size = 16
hbm_hit_tokens = 32768
full_hashes = 4096
suffix_hashes = 2048
```

分别调用：

- `FULL`: 发送 4096 hashes，`hbm_hit_tokens=32768`
- `SUFFIX`: 发送 2048 hashes，`hbm_hit_tokens=32768`

要求结果：

```text
FULL received hashes = 4096
SUFFIX received hashes = 2048
FULL returned hit_tokens == SUFFIX returned hit_tokens
SUFFIX payload bytes < FULL payload bytes
SUFFIX RTT < FULL RTT
```

这个实验能直接证明 PR 的核心改动是否正确，不受真实模型和 NPU 抖动影响。

### 推荐实验方案 B：真实 vLLM e2e 命中验证

在方案 A 通过后，再尝试真实 vLLM 服务。

基于现有探针：

```text
/tmp/vllm-ascend-pr/benchmarks/scripts/e2e_ascend_store_suffix_lookup_probe.py
```

需要增强或调整：

1. 分别跑 `lookup_hash_mode=full` 和 `lookup_hash_mode=suffix` 两组；
2. 使用相同模型 `/mnt/weight/Qwen3-0.6B`；
3. 设置 `VLLM_BATCH_INVARIANT=1`；
4. 使用 `ASCEND_RT_VISIBLE_DEVICES=0`；
5. 构造能稳定产生 HBM prefix hit 的请求序列；
6. 从日志或 monkeypatch 里记录：
   - `num_computed_tokens`
   - `hbm_hit_tokens`
   - client 发送 hash 数
   - server 收到 hash 数
   - worker 返回 hit tokens
   - 请求 TTFT 或请求耗时

如果日志里仍然只有：

```text
hit_tokens=0
```

就不能声称真实 e2e 命中型收益已验证。

### 如何产生 HBM prefix hit

当前同 prompt 连续请求没有稳定得到 `hbm_hit_tokens > 0`。下一步可以尝试：

1. 启用或确认 vLLM prefix caching；
2. 使用完全相同 prompt 连续请求；
3. 使用同一长 prefix、不同 suffix 的请求；
4. 增大 prompt 长度，例如 2K / 4K；
5. 检查 scheduler 日志里的 `num_computed_tokens`；
6. 必要时 monkeypatch `LookupKeyClient.lookup()` 打印：
   - `len(block_hashes)`
   - `hbm_hit_tokens`
   - `lookup_hash_mode`

如果真实 HBM prefix hit 很难稳定构造，先完成方案 A，方案 B 记录为待补充。

## 推荐合入意见

如果只基于当前已有验证，建议 PR 评论这样写：

```text
建议合入。该 PR 的 ascend_store 单测、真实 ZMQ client/server round-trip、payload benchmark 和最小 AscendStoreConnector suffix e2e 均已通过。协议级结果显示，在 64K tokens / 50% HBM hit 下 payload 从 270354 bytes 降至 135188 bytes，RTT mean 从 732.07 us 降至 437.25 us；在 75% HBM hit 下收益更明显。

lookup ZMQ 是 vLLM 内部 IPC 通道，标准部署中 scheduler/worker 同版本一起启动，因此旧新协议混跑不应作为常规合入阻塞项。建议后续补充真实 hbm_hit_tokens > 0 场景下的 e2e 记录，确认 suffix 模式发送 hash 数少于 full 且返回 hit tokens 一致；真实 TTFT 收益可作为进一步性能报告补充。
```

## 需要更新或保留的文件

主要交接文档：

```text
/home/lizhongyang/llm-project/draft/issue_14145_zmq_lookup_payload_omit_handoff.md
```

兼容性说明文档：

```text
/home/lizhongyang/llm-project/draft/zmq_lookup_protocol_compatibility.md
```

当前用户要求的新交接文档：

```text
/home/lizhongyang/llm-project/draft/handoff.md
```

临时验证脚本：

```text
/tmp/vllm-ascend-pr/benchmarks/scripts/benchmark_zmq_lookup_payload.py
/tmp/vllm-ascend-pr/benchmarks/scripts/e2e_ascend_store_suffix_lookup_probe.py
```

注意：这些脚本在 `/tmp/vllm-ascend-pr`，如果容器或临时目录清理，需要从前面的文档重新创建或从 git 工作树恢复。
