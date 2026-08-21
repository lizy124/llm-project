# Issue #14145 ZMQ Lookup Payload 裁剪容器验证记录

## 结论摘要

验证对象是 fork 仓库：

```text
https://github.com/ChenZhuo888/vllm-ascend/tree/test/zmq-lookup-payload-omit
```

本次在容器中实际克隆并验证了该分支。分支 HEAD 为：

```text
79e8d276001ce4c69416c8f533f04cd36b8a4cff
```

目标 4 个提交确认为：

| 提交 | 说明 |
|---|---|
| `b8a9c3c2` | `feat(kv-pool): support suffix hash lookup with legacy fallback` |
| `7e56764e` | `type unified` |
| `59e61d05` | `test(ascend_store): add coverage for suffix lookup hash omission` |
| `79e8d276` | `type unified` |

关键结论：

1. 这 4 个提交的核心代码路径在当前容器中可以通过现有单元测试和静态检查。
2. `strict_review.md` 中的 P1-1 “fallback SUFFIX 首个 suffix miss 会丢失 HBM 前缀并返回 0”不成立，已被代码和测试结果推翻。
3. 新增 ZMQ mode frame 是真实协议变更，新协议同版本 round-trip 可用。
4. 当前实现没有旧协议兼容回退，旧格式 request 发给新 server 时不能得到正常响应。
5. 容器内协议级 benchmark 显示，SUFFIX 能显著减少 lookup payload bytes，并降低本机 ZMQ round-trip latency。
6. 本次尚未验证真实 Ascend NPU 端到端 TTFT，也没有测真实 AscendStore 查询开销。

## 一、源码获取与分支确认

官方 `vllm-project/vllm-ascend` 仓库中没有 `test/zmq-lookup-payload-omit` 分支。用户提供 fork 后，验证到该分支存在：

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

确认提交：

```text
79e8d276 type unified
59e61d05 test(ascend_store): add coverage for suffix lookup hash omission
7e56764e type unified
b8a9c3c2 feat(kv-pool): support suffix hash lookup with legacy fallback
6af9257e [Feature] Support dynamic speculative decoding with DFlash and Unified dynamic selection logic (#13819)
```

## 二、静态检查与单元测试结果

### 1. compileall

执行：

```bash
python -m compileall -q /tmp/vllm-ascend-pr/vllm_ascend \
    /tmp/vllm-ascend-pr/tests/ut/distributed/ascend_store
```

结果：通过，无错误输出。

### 2. diff 检查

执行：

```bash
git -C /tmp/vllm-ascend-pr diff --check HEAD~4..HEAD
```

结果：通过，无错误输出。

### 3. 分文件单测

执行并通过：

```bash
python -m pytest -q /tmp/vllm-ascend-pr/tests/ut/distributed/ascend_store/test_pool_scheduler.py
```

结果：

```text
78 passed, 14 warnings
```

执行并通过：

```bash
python -m pytest -q /tmp/vllm-ascend-pr/tests/ut/distributed/ascend_store/test_ascend_store_connector.py
```

结果：

```text
29 passed, 14 warnings
```

执行并通过：

```bash
python -m pytest -q /tmp/vllm-ascend-pr/tests/ut/distributed/ascend_store/test_pool_worker.py
```

结果：

```text
110 passed, 14 warnings
```

执行并通过：

```bash
python -m pytest -q /tmp/vllm-ascend-pr/tests/ut/distributed/ascend_store/test_coordinator.py
```

结果：

```text
13 passed, 14 warnings
```

### 4. ascend_store 目录级单测

执行：

```bash
python -m pytest -q tests/ut/distributed/ascend_store
```

结果：

```text
402 passed, 10 skipped, 14 warnings in 2.76s
```

结论：当前容器中，`ascend_store` 目录级单测整体通过。

## 三、关键代码路径复核

### 1. LookupHashMode

位置：

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py:257
```

定义：

```python
class LookupHashMode(str, Enum):
    FULL = "full"
    SUFFIX = "suffix"
```

语义：

- `FULL`：发送完整 block hash 列表。
- `SUFFIX`：只发送 HBM 前缀之后的 suffix block hash。

### 2. scheduler 配置读取

位置：

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:157
```

行为：

- 从 `kv_connector_extra_config["lookup_hash_mode"]` 读取配置。
- 默认值为 `full`。
- 非法值抛出 `ValueError`。

### 3. client ZMQ frame 布局

位置：

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:1191
```

新布局：

```text
frame 0: token_len
frame 1: kv_cache_group_ids
frame 2: hbm_hit_tokens
frame 3: lookup_hash_mode
frame 4+: hash list
```

关键点：`LookupKeyClient.lookup()` 无论 `FULL` 还是 `SUFFIX` 都发送 `lookup_hash_mode` frame。

### 4. server 解码行为

位置：

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:317
```

行为：

```python
token_len = int.from_bytes(all_frames[0], byteorder="big")
kv_group_ids = self.decoder.decode([all_frames[1]])
hbm_hit_tokens = int.from_bytes(all_frames[2], byteorder="big")
lookup_hash_mode = LookupHashMode(self.decoder.decode([all_frames[3]]))
hashes_str = self.decoder.decode(all_frames[4:])
```

关键点：server 固定按新协议解析 `frame[3]` 为 mode，没有旧格式 fallback。

### 5. worker fallback 路径

位置：

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:2370
```

关键行为：

- `SUFFIX` 模式下：
  - `lookup_token_len = token_len - hbm_hit_tokens`
  - `token_offset = hbm_hit_tokens`
- `_build_lookup_keys()` 生成 key 时使用 `token_offset`，保证 suffix hash 的 token 坐标仍然对应原始 prompt。
- 当 `SUFFIX` 且 `hbm_hit_tokens > 0` 时，会先把 `hbm_hit_tokens` 插入 `group_hits`，再判断是否为空。

关键代码位置：

```text
pool_worker.py:2467-2475
```

语义：

```python
if lookup_hash_mode is LookupHashMode.SUFFIX and hbm_hit_tokens:
    group_hits.insert(0, hbm_hit_tokens)

if not group_hits:
    return 0
```

结论：`SUFFIX + 首个 suffix miss` 不会丢失 HBM 前缀。

### 6. coordinator 路径

位置：

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:2248
```

行为：

- `FULL` 模式：用真实 HBM 前缀 hash 更新 external existence view。
- `SUFFIX` 模式：构造 `HBMCachedBlockHashList`，用逻辑 marker 表示 HBM 前缀，避免传输真实前缀 hash。

关键点：coordinator 路径保留原始 block index 语义，避免 suffix hash 被误当成从 token 0 开始。

## 四、争议点复核

### 1. strict_review.md 的 P1-1 不成立

`strict_review.md` 曾判断：

```text
fallback SUFFIX 路径会丢失 HBM 前缀命中，首个 suffix miss 返回 0
```

本次验证结论：该判断不成立。

依据：

1. 当前代码中，`group_hits.insert(0, hbm_hit_tokens)` 在 `if not group_hits` 之前执行。
2. `test_pool_worker.py` 中与 suffix / fallback / HBM 相关测试通过。
3. `test_pool_worker.py` 整体 110 passed。
4. `ascend_store` 整体测试 402 passed。
5. 单独 smoke test 验证空 suffix 时返回 HBM 前缀长度。

### 2. 协议兼容性风险成立

新增 ZMQ frame 是真实 wire protocol 变更。

新协议同版本 round-trip 可用，但没有旧协议兼容解析。

已验证：

- 新 `LookupKeyClient` -> 新 `LookupKeyServer` 可正常 round-trip。
- server 收到 `lookup_hash_mode=LookupHashMode.SUFFIX`。
- 旧格式 request 发给新 server 时没有正常响应，client `recv()` 超时。

结论：如果部署中存在旧 client / 新 server 或新 client / 旧 server 混用，需要额外处理协议兼容、版本协商或升级顺序。

### 3. 非对齐 hbm_hit_tokens 行为明确但偏硬

已验证：

- `SUFFIX` 要求 `hbm_hit_tokens % hash_block_size == 0`。
- 非对齐输入会触发 `AssertionError`。

示例输出：

```text
Remote connection failed in lookup. type=AssertionError, error=hbm_hit_tokens must be aligned to hash_block_size when using suffix hash lookup.
```

结论：行为明确，但这是运行时 assert，不是可恢复 fallback。如果生产环境可能出现非对齐输入，仍建议改成显式回退 `FULL` 或配置级拒绝。

## 五、真实 ZMQ round-trip 验证

本次做了最小真实 ZMQ 测试：

- 启动真实 `LookupKeyServer`
- 使用真实 `LookupKeyClient`
- 使用 `LookupHashMode.SUFFIX`
- worker 使用 stub，只返回固定结果

结果：

```text
result= 42
call= (64, ['aabb', 'ccdd'], [0, 1], False, 16, <LookupHashMode.SUFFIX: 'suffix'>)
```

结论：新协议同版本真实 ZMQ round-trip 可用，mode frame 能正确传到 worker。

旧格式 request 测试结果：

```text
Again Resource temporarily unavailable
```

含义：client 等待响应超时。说明新 server 不兼容旧 frame 布局。

## 六、性能数据脚本

新增脚本位置：

```text
/tmp/vllm-ascend-pr/benchmarks/scripts/benchmark_zmq_lookup_payload.py
```

脚本用途：

1. 测量 `FULL` 和 `SUFFIX` 的 encoded ZMQ multipart payload bytes。
2. 启动真实 `LookupKeyServer` 和 `LookupKeyClient`。
3. 测量本机 ZMQ REQ/REP round-trip latency。
4. worker 使用 stub，不依赖 Ascend NPU 或真实 AscendStore。

脚本不测：

- Ascend NPU TTFT；
- 真实 AscendStore 查询耗时；
- 模型 forward；
- 多机部署网络延迟。

### 1. smoke test

命令：

```bash
python benchmarks/scripts/benchmark_zmq_lookup_payload.py \
    --iterations 20 \
    --warmup 5 \
    --case 1024:512
```

结果：

```text
tokens=1024 block_size=16 hbm_hit_tokens=512 full_hashes=64 suffix_hashes=32
  payload_bytes: full=4242 suffix=2132 saved=2110 saved_pct=49.74
  rtt_us_full: mean=112.82 p50=111.89 p95=120.25 p99=121.77
  rtt_us_suffix: mean=104.78 p50=103.94 p95=114.14 p99=114.92
```

### 2. 16K prompt，50% HBM hit

命令：

```bash
python benchmarks/scripts/benchmark_zmq_lookup_payload.py \
    --iterations 1000 \
    --warmup 100 \
    --case 16384:8192
```

结果：

```text
tokens=16384 block_size=16 hbm_hit_tokens=8192 full_hashes=1024 suffix_hashes=512
  payload_bytes: full=67602 suffix=33812 saved=33790 saved_pct=49.98
  rtt_us_full: mean=260.56 p50=257.52 p95=274.82 p99=288.13
  rtt_us_suffix: mean=184.27 p50=182.67 p95=197.50 p99=212.68
```

结论：

- payload 减少约 49.98%。
- mean RTT 从 260.56 us 降到 184.27 us。
- p95 RTT 从 274.82 us 降到 197.50 us。

### 3. 64K prompt，50% HBM hit

命令：

```bash
python benchmarks/scripts/benchmark_zmq_lookup_payload.py \
    --iterations 1000 \
    --warmup 100 \
    --case 65536:32768
```

结果：

```text
tokens=65536 block_size=16 hbm_hit_tokens=32768 full_hashes=4096 suffix_hashes=2048
  payload_bytes: full=270354 suffix=135188 saved=135166 saved_pct=50.00
  rtt_us_full: mean=732.07 p50=727.70 p95=751.35 p99=785.48
  rtt_us_suffix: mean=437.25 p50=435.77 p95=451.45 p99=469.13
```

结论：

- payload 减少约 50.00%。
- mean RTT 从 732.07 us 降到 437.25 us。
- p95 RTT 从 751.35 us 降到 451.45 us。

### 4. 64K prompt，75% HBM hit

命令：

```bash
python benchmarks/scripts/benchmark_zmq_lookup_payload.py \
    --iterations 1000 \
    --warmup 100 \
    --case 65536:49152
```

结果：

```text
tokens=65536 block_size=16 hbm_hit_tokens=49152 full_hashes=4096 suffix_hashes=1024
  payload_bytes: full=270354 suffix=67604 saved=202750 saved_pct=74.99
  rtt_us_full: mean=766.89 p50=765.33 p95=793.26 p99=819.40
  rtt_us_suffix: mean=268.63 p50=264.81 p95=290.87 p99=328.01
```

结论：

- payload 减少约 74.99%。
- mean RTT 从 766.89 us 降到 268.63 us。
- p95 RTT 从 793.26 us 降到 290.87 us。
- HBM prefix 比例越高，SUFFIX 的 payload 和 ZMQ RTT 收益越明显。

## 七、性能结论

容器内协议级 benchmark 证明：

1. SUFFIX 模式能按 HBM prefix 比例近似线性减少 lookup payload bytes。
2. payload bytes 降低会反映到本机 ZMQ REQ/REP round-trip latency 上。
3. 16K / 64K 两个 issue 关心的长 prompt 场景下，SUFFIX 均有明确收益。
4. 该 benchmark 是协议层微基准，不等价于端到端 TTFT 或真实 AscendStore lookup latency。

## 八、仍未完成的验证

以下内容本次没有完成，不能据此声称 issue #14145 已完整验收：

1. 真实 Ascend NPU 环境下的端到端 TTFT。
2. 真实 AscendStore 后端查询耗时。
3. 多机 / 多进程部署中的网络和进程调度影响。
4. 新旧 client/server 双向兼容完整矩阵。
5. 滚动升级策略验证。
6. 配置文档、部署说明和 PR 设计文档是否补齐。
7. commit sign-off 和提交信息是否满足上游仓库合入规范。

## 九、最终判断

当前 4 个提交在容器内的代码正确性验证结果是正向的：

- 相关单测全过；
- 目录级 `ascend_store` 测试全过；
- 新协议真实 ZMQ round-trip 可用；
- `SUFFIX` 对 HBM prefix 的边界处理正确；
- 性能微基准显示 payload 和本机 ZMQ RTT 明显下降。

但从 issue 验收或生产合入角度，仍建议保留以下 blocking / follow-up 项：

1. 明确新旧协议兼容策略，或在文档中要求 scheduler/worker 同步升级。
2. 将旧格式 / malformed mode 的失败行为变成可观测错误，而不是让 client 等待超时。
3. 明确 `hbm_hit_tokens` 非对齐时的产品语义，优先考虑回退 `FULL` 或配置级拒绝。
4. 补真实 Ascend 硬件上的 16K/64K TTFT、lookup latency、AscendStore 查询耗时。
5. 补充用户配置文档和部署说明。

简短结论：

> 这 4 个 commit 的实现方向和核心功能在容器中已验证通过；严格审查中关于“首个 suffix miss 丢 HBM 前缀”的 P1 不成立。当前最主要的真实风险不是核心 suffix 命中语义，而是 ZMQ wire protocol 兼容性、非对齐输入处理方式，以及缺少真实 Ascend 端到端性能验收。
