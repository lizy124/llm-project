# Issue #14145 严格代码审查报告

## 结论

审查对象是分支 `test/zmq-lookup-payload-omit` 的最后 4 个提交：

- `b8a9c3c2` `feat(kv-pool): support suffix hash lookup with legacy fallback`
- `7e56764e` `type unified`
- `59e61d05` `test(ascend_store): add coverage for suffix lookup hash omission`
- `79e8d276` `type unified`

对应任务为 [vllm-ascend#14145](https://github.com/vllm-project/vllm-ascend/issues/14145)，目标是让 scheduler 到 worker lookup server 的 ZMQ 请求只携带 HBM 前缀之后的 hash。

当前结论：**REQUEST CHANGES，不建议合入**。

代码方向基本正确，`FULL`/`SUFFIX` 两种模式、scheduler 侧切片、worker 侧绝对 offset、coordinator 的 HBM marker 都已经接通；但存在一个确定的功能错误，一个协议兼容性回归，以及多个未满足 issue 验收标准的交付缺口。尤其是第一个问题会直接破坏“HBM 前缀命中但远端第一个 suffix miss”的核心边界场景。

## Findings

### P1-1：fallback SUFFIX 路径会丢失 HBM 前缀命中

位置：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:2458-2474`

问题：

1. SUFFIX 模式下，`hbm_hit_tokens` 表示已经在本地 HBM 命中的前缀。
2. fallback 路径查询远端 suffix 后，先调用 `find_all_continuous_hit_positions()` 或 `find_all_discontinuous_hit_positions()`。
3. 当第一个远端 suffix block 未命中时，`group_hits` 是空列表。
4. 代码在 `2473-2474` 立即 `return 0`，而 `2471` 的 `group_hits.insert(0, hbm_hit_tokens)` 尚未执行。

因此，在 `hbm_hit_tokens=32`、远端 suffix 结果为 `[0, 1]` 时，正确结果应为 `32`，当前结果是 `0`。这违反了“裁剪后命中结果与发完整 hashes 一致”，也会使 scheduler 重新计算本地已经拥有的 HBM 前缀。

该问题已有新增测试意图覆盖：

- `tests/ut/distributed/ascend_store/test_pool_worker.py:955-966` 期望首个远端 suffix miss 时返回 `32`。

但按当前控制流，该测试应失败。修复方式应当是先建立包含 HBM 边界的 `group_hits`，再判断是否为空；或者在 `group_hits` 为空且 `SUFFIX` 且 `hbm_hit_tokens > 0` 时把 `[hbm_hit_tokens]` 作为本组结果。多 group 路径也必须保持相同语义。

### P1-2：新增 ZMQ frame 破坏旧版本 client/server 互操作

位置：

- client：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:1203-1211`
- server：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:318-329`

新 client 无条件插入一个 `lookup_hash_mode` frame；新 server 无条件把 `all_frames[3]` 当作 mode 并把 `all_frames[4:]` 当作 hashes。

这意味着：

- 新 client -> 旧 server：旧 server 会把 `"full"`/`"suffix"` 当作第一段 hash payload，后续 hash 数量和内容错位，最终 lookup 错误或返回 0。
- 旧 client -> 新 server：新 server 会把旧协议中的 hash 列表 frame 当作 mode，`LookupHashMode(...)` 抛出 `ValueError`，server 线程可能直接退出。

提交说明中称 `FULL` 是“backward-compatible behavior”，但这只保持了 lookup 算法默认值，没有保持 ZMQ wire protocol 的兼容性。对于 scheduler/worker 分进程部署、滚动升级、不同镜像版本并存的场景，这是实际回归。

建议：

- 增加协议版本或 capability negotiation；或
- server 同时接受旧格式和新格式，根据 frame 数量/可解码内容识别旧请求；或
- 先保持旧 frame 布局，把 mode 放入兼容的可选 payload 中。

至少应增加旧 client -> 新 server、新 client -> 旧 server 的协议测试。

### P2-1：SUFFIX 的对齐前提会在运行时直接中止 scheduler，且没有配置文档

位置：

- scheduler：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:585-590`
- worker：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:2263-2272`、`2385-2390`

SUFFIX 模式要求 `hbm_hit_tokens % hash_block_size == 0`，否则使用 `assert`。这个前提在 hash 列表切片上是可以理解的，但当前行为有两个风险：

- `num_computed_tokens` 是运行时状态，不是静态配置；一旦上游返回非 hash 边界，scheduler 会抛 AssertionError，而不是回退到 FULL 或明确拒绝 SUFFIX 配置。
- 新增 `lookup_hash_mode` 没有加入配置文档、示例、合法值说明或部署兼容说明。

如果产品设计明确要求 SUFFIX 只支持 hash 边界，应在配置层校验并记录限制；更稳妥的实现是在非对齐请求上退回 FULL，确保打开优化开关不会改变正确性。

### P2-2：没有提交 issue 要求的性能验证和设计说明

issue #14145 的交付件明确要求：

- 设计说明；
- 16K/64K 长 prompt 的 IPC payload 大小对比；
- lookup 延迟对比；
- Ascend 型号、卡数、TP/CP/PP 配置；
- 单测结果。

这 4 个 commit 只修改代码和单测，没有设计文档、benchmark、payload 字节数、延迟数据或硬件测试记录。因此即使修复上述代码问题，也不能据此判定 issue 已完成验收。

### P2-3：新增测试没有覆盖真实 codec 和版本兼容性

新增 wire protocol 测试主要 mock `MsgpackEncoder`/`MsgpackDecoder`，例如：

- `tests/ut/distributed/ascend_store/test_pool_scheduler.py:694-733`
- `tests/ut/distributed/ascend_store/test_ascend_store_connector.py:86-130`

这些测试验证了 frame 顺序，但没有使用真实 msgpack 编解码器，也没有启动一对真实 REQ/REP socket。因此无法发现以下问题：

- 真实 multipart frame 的解码结果是否与 server 预期一致；
- 空 suffix、完整 HBM 命中时的实际 payload；
- 旧协议/新协议互通；
- malformed mode 对 server 线程生命周期的影响。

建议补充最小真实 codec 测试和一个 in-process ZMQ round-trip 测试。

### P3-1：提交规范不符合仓库约定

仓库 `AGENTS.md` 要求 commit 使用 `git commit -s` 并包含 `Signed-off-by`。这 4 个提交均没有 sign-off；其中 `7e56764e` 和 `79e8d276` 的提交信息只是 `type unified`，不符合 Conventional Commits，也降低了审查和回溯价值。

这不是运行时 bug，但在当前仓库规范下应在合入前整理提交历史。

## 正确性检查结果

以下部分的实现方向是合理的：

- `KVPoolScheduler` 在 SUFFIX 模式按 `num_computed_tokens // hash_block_size` 切掉 HBM 前缀 hash，见 `pool_scheduler.py:583-597`。
- `LookupKeyClient` 将 mode 传给 lookup server，server 再传给 `lookup_scheduler`。
- `HBMCachedBlockHashList` 用惰性 marker 保留绝对 hash index，避免重新构造真实前缀 hash，见 `coordinator.py:28-77`。
- coordinator 路径通过 `ExternalCachedBlockPool` 将 marker 视为每个 group 都已命中，设计上能正确组合 HBM 前缀和远端 suffix。
- `hbm_hit_tokens=0` 和 suffix 为空的路径在 coordinator/fallback 的大部分分支中有处理；但必须同时修复 P1-1，并用真实边界测试确认。
- 默认模式仍是 `LookupHashMode.FULL`，不会在同版本部署中改变默认 lookup 算法。

## 测试与验证

已执行：

- `git diff --check HEAD~4..HEAD`：通过。
- `python -m compileall -q vllm_ascend tests/ut/distributed/ascend_store`：通过；输出了仓库已有的 `worker.py:219` `return in finally` SyntaxWarning。
- 读取并核对 GitHub issue #14145 的正文和评论。
- 静态检查 4 个 commit 的全部生产代码和新增测试。

未能执行完整单测：当前环境缺少依赖，具体为 `pytest`、`numpy` 未安装；`unittest` 导入测试时也因此失败。故本报告没有把“全部单测绿”作为已验证事实。

## 建议的修复顺序

1. 修复 P1-1，确保 HBM 命中且首个 suffix miss 返回 `hbm_hit_tokens`，并增加单 group、多 group、连续/非连续三类测试。
2. 设计并实现 ZMQ 协议兼容策略，覆盖新旧 client/server 组合。
3. 明确非对齐 `hbm_hit_tokens` 的产品语义：FULL 回退或配置级拒绝，避免运行时 AssertionError。
4. 使用真实 msgpack 和 in-process ZMQ 补充端到端测试。
5. 在 Ascend NPU 上完成 16K/64K payload 与 latency benchmark，提交设计说明和硬件/并行配置。
6. 按仓库规范重写两个 `type unified` 提交并补齐 sign-off。

## 最终判定

这组提交已经完成了 issue #14145 的主要代码骨架，但目前不能视为完整、可靠的解法。P1-1 会导致核心边界场景错误，P1-2 会导致跨版本进程通信回归；同时 issue 要求的性能数据和设计交付件缺失。建议修复并补齐验证后重新审查。
