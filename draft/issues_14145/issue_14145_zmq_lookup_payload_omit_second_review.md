# Issue #14145 第二次严格代码审查报告

## 1. 审查范围与方法

审查对象是仓库 `vllm-ascend-zmq-lookup-payload-omit` 分支 `test/zmq-lookup-payload-omit` 的最后 4 个提交，基线为 `HEAD~4..HEAD`：

| 提交 | 主题 |
| --- | --- |
| `b8a9c3c2` | `feat(kv-pool): support suffix hash lookup with legacy fallback` |
| `7e56764e` | `type unified` |
| `59e61d05` | `test(ascend_store): add coverage for suffix lookup hash omission` |
| `79e8d276` | `type unified` |

检查了全部生产代码差异、相关 vLLM 上游实现、单测和 issue #14145 当前正文。现有第一次报告仅作为待证伪材料；本报告的结论以代码控制流和协议帧布局为依据。

Issue 的明确验收目标是：

1. 裁剪后命中结果与发送完整 hashes 一致；
2. `hbm_hit_tokens = 0` 和全长边界正确；
3. 现有单测全绿；
4. 16K/64K 长 prompt 的 IPC payload 和 lookup 延迟下降；
5. 提交设计说明、性能数据和测试结果。

## 2. 结论

**结论：REQUEST CHANGES，不建议当前 4 个提交直接合入。**

SUFFIX 的核心数据流已经基本接通，且第一次报告中的一个关键指控不成立：`lookup_scheduler()` 在 `pool_worker.py:2467-2474` 先把 `hbm_hit_tokens` 插入 `group_hits`，再判断空列表；所以“首个 suffix miss 会返回 0、丢失 HBM 前缀”不是当前代码的 bug，新增测试 `test_lookup_scheduler_suffix_fallback_first_remote_miss()` 与实际控制流一致。

但是，新增的 ZMQ wire protocol 是无版本协商的强制变更。旧 client/新 server 会让 server 线程异常退出并使 client 永久阻塞；新 client/旧 server 会把 mode 当成 hash payload，产生错误 lookup。该问题直接违反可部署的协议鲁棒性，也使提交说明中所谓的“FULL backward-compatible behavior”不成立。除此之外，issue 要求的性能/设计交付件没有进入这 4 个提交，真实 codec/真实 socket 的协议验证也缺失。

## 3. Findings

### P1-1：ZMQ 协议新增强制 frame，没有版本协商或旧格式解析；旧 client -> 新 server 会卡死

证据：

- 新 client 无条件编码并发送 mode frame：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:1191-1213`。
- 新 server 无条件把第 4 个 frame 当成 mode：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:315-330`。
- server 的 `process_request()` 没有异常保护；线程是 daemon thread：`ascend_store_connector.py:315-341`。
- client 在发送后同步等待 response：`pool_scheduler.py:1211-1214`。
- `make_zmq_socket()` 没有为 REQ/REP 设置 `RCVTIMEO`；上游 `vllm/vllm/utils/network_utils.py:284-342` 只建立并连接 socket。因此这里的 `recv()` 没有超时兜底。
- vLLM 的 `MsgpackDecoder.decode()` 对 frame sequence 只解码第一个 buffer：上游 `vllm/v1/serial_utils.py:340-348`。因此旧协议的 `all_frames[3]` 是完整 hash-list frame，而不是 mode frame。

旧协议发送布局是：

```text
[token_len, kv_group_ids, hbm_hit_tokens, encoded_hash_list]
```

新协议发送布局是：

```text
[token_len, kv_group_ids, hbm_hit_tokens, encoded_mode, encoded_hash_list]
```

在“旧 client -> 新 server”场景：

1. 新 server 在 `all_frames[3]` 读取旧的 encoded hash-list；
2. `LookupHashMode(self.decoder.decode([all_frames[3]]))` 尝试把类似 `['aabb', 'ccdd']` 转换为 `LookupHashMode`；
3. `LookupHashMode(...)` 抛出 `ValueError`/`TypeError`；
4. 异常逃出 `process_request()`，线程结束，没有 `socket.send()`；
5. 旧 client 已经进入 `recv()`，因此一直等待。

这不是单纯的“返回错误命中数”，而是滚动升级、scheduler/worker 镜像版本不一致或残留旧 worker 时的请求级永久阻塞。修复应采用协议版本/capability negotiation，或 server 同时兼容旧布局和新布局；至少要在 server 捕获 malformed/unknown mode 并返回明确错误，不能让请求线程死亡。

### P1-2：新 client -> 旧 server 会把 `full`/`suffix` 当成 hash 列表，破坏跨版本 lookup

证据：

- 新 client 在 `pool_scheduler.py:1200-1209` 先发送 mode，再发送 hash frame。
- 旧 server 的实现（`b8a9c3c2^`）在 `ascend_store_connector.py` 中直接执行 `hashes_str = self.decoder.decode(all_frames[3:])`，即把第 4 个 frame 当作 hash-list。
- 由于 `MsgpackDecoder.decode()` 只解码 sequence 的第一个 buffer，旧 server 实际得到的是字符串 `"full"` 或 `"suffix"`，而不是 `list[str]`。

随后旧 server 会将这个字符串传入旧的 `lookup_scheduler()`。下游代码期望的是 hash sequence；字符串会按字符迭代，导致查询 key 错位/错误，或者在 key 构造处触发异常并返回 0。无论是错误命中还是静默 miss，都违反“裁剪后命中结果与完整 hashes 一致”。

因此，`LookupHashMode.FULL` 只保持了同版本内的默认算法，不保持 wire protocol 兼容。提交说明中的“legacy fallback”并没有实现旧 client/server 的协议 fallback。需要增加四种组合的测试：

| client | server | 当前结果 |
| --- | --- | --- |
| old | old | 原协议工作 |
| new | new | 新协议工作 |
| old | new | mode 解码异常，server 线程退出，client 阻塞 |
| new | old | mode 被当成 hash-list，错误 lookup/返回 0 |

### P2-1：SUFFIX 的运行时对齐约束依赖外部 scheduler 不变量，失败时直接抛 AssertionError

证据：

- scheduler 在 `pool_scheduler.py:585-590` 使用 `assert num_computed_tokens % self.hash_block_size == 0`。
- worker coordinator/fallback 两条路径也在 `pool_worker.py:2263-2272` 和 `pool_worker.py:2385-2390` 使用 `assert`。
- 当前单测明确验证非对齐输入会抛 AssertionError：`tests/ut/distributed/ascend_store/test_pool_scheduler.py:202-215`。

在标准 vLLM 调度路径中，调用 connector 前通常会把本地命中对齐到物理 block；并且 `hash_block_size` 通常是 block size 的因子，所以该断言在正常配置下应当不触发。这也是为什么第一次报告把它描述为“必然运行时中止”并不严谨。

但它仍然是一个未封装的跨模块前提：

- `num_computed_tokens` 是运行时状态，可能受 chunked prefill、spec decode、混合 KV group、CP 缩放或第三方调度入口影响；
- scheduler 侧的断言在 `get_num_new_matched_tokens()` 外层，没有被转换为 connector 的可恢复错误；
- worker 侧的异常会被总 `try/except` 转成命中数 0，表现为静默关闭远端命中，而不是明确拒绝配置。

建议将该前提在初始化阶段校验，并在非对齐输入上选择明确语义：回退 FULL、向下取整，或返回可重试错误；不要依赖 Python `assert` 作为生产协议契约。若产品明确只支持 hash boundary，应在配置文档中写明并增加 CP/混合 group 的实际调用测试。

### P2-2：新模式没有用户文档、配置示例或部署兼容说明

证据：

- 配置只在 `pool_scheduler.py:157-166` 从 `kv_connector_extra_config` 读取字符串 `lookup_hash_mode`；
- 全仓库没有该配置的 README、用户指南、示例、合法值说明或版本兼容说明；
- worker 只从 RPC frame 接收 mode，没有校验 scheduler/worker 的 `hash_block_size`、CP/TP/PP 配置是否一致。

这使 SUFFIX 成为一个未文档化的隐藏配置。尤其该功能改变了进程间 wire layout，部署方无法知道必须同时升级 client 和 server，也无法知道非对齐输入的行为。至少应记录默认值、合法值、要求的 scheduler/worker 配置一致性、升级策略和回退方式。

### P2-3：issue 要求的性能验证和交付件缺失，无法据此证明目标达成

issue #14145 要求 16K/64K 长 prompt 的 payload 大小下降、lookup 延迟下降、Ascend 型号/卡数/TP/CP/PP 配置、设计说明和单测结果。

这 4 个提交只修改生产代码和单测，没有：

- 16K/64K 的 before/after payload 字节数；
- msgpack 编解码和 ZMQ round-trip 延迟；
- Ascend 型号、卡数、TP/CP/PP 配置；
- 设计文档或 PR 验证记录。

因此即使 P1 修复，也不能将 issue 判定为已完成验收。性能优化必须用真实 payload 和真实 NPU/部署配置测量，不能只凭 `request.block_hashes[hbm_cached_hashes:]` 的静态代码推断收益。

### P2-4：协议测试全部 mock 编解码器和 socket，无法验证实际帧兼容性

证据：

- scheduler wire test 在 `tests/ut/distributed/ascend_store/test_pool_scheduler.py:245-280` mock `MsgpackEncoder` 和 ZMQ socket；
- server test 在 `tests/ut/distributed/ascend_store/test_ascend_store_connector.py:86-131` mock `MsgpackDecoder`、socket 和 thread；
- 测试直接让 mock decoder 按预设顺序返回 `groups`、`suffix`、`hashes`，没有运行真实 `MsgpackEncoder/MsgpackDecoder`。

这些测试能证明“调用者按照预期顺序调用 mock”，不能发现：

- old/new frame 数量不同时的真实解码结果；
- `decode(all_frames[3:])` 只使用第一个 buffer 的行为；
- 空 suffix、单 frame、malformed mode；
- server 线程异常退出后 client `recv()` 永久等待。

应增加使用真实 codec 的单进程 round-trip 测试，并至少覆盖上述四种 client/server 版本组合；对于 server 异常，还应验证客户端收到错误或超时，而不是无限等待。

### P3-1：4 个提交均没有 sign-off；其中两个提交信息不符合仓库规范

证据：

- `git log --format='%G?'` 对 4 个提交均显示 `N`，提交正文没有 `Signed-off-by:`；
- `7e56764e` 和 `79e8d276` 的提交信息只有 `type unified`；
- 仓库 `AGENTS.md` 要求 Conventional Commits 和 `git commit -s`。

这不是运行时正确性问题，但当前提交历史不满足仓库的合入要求，也让两个“type unified”提交无法从提交信息判断修改意图。合入前应整理历史并补齐 sign-off。

## 4. 对第一次报告的逐项复核

### 已证伪：所谓“首个 suffix miss 返回 0”

第一次报告声称在 `pool_worker.py:2471-2474` 中，`group_hits` 为空时会在插入 HBM 边界前返回 0。当前代码实际顺序是：

```python
if lookup_hash_mode is LookupHashMode.SUFFIX and hbm_hit_tokens:
    group_hits.insert(0, hbm_hit_tokens)

if not group_hits:
    return 0
```

位置为 `pool_worker.py:2467-2474`。因此：

- 首个连续 suffix miss：`find_all_continuous_hit_positions()` 返回 `[]`，插入后为 `[hbm_hit_tokens]`，返回 HBM 边界；
- 首个非连续 suffix miss：同样先插入 HBM 边界；
- 空 suffix：`pool_worker.py:2420-2431` 特判并保留 HBM 边界；
- coordinator 路径：`HBMCachedBlockHashList` 在 `coordinator.py:28-77` 用 marker 表示 HBM prefix，`ExternalCachedBlockPool.get_cached_block()` 在 `coordinator.py:95-107` 将 marker 视为命中。

对应新增测试 `tests/ut/distributed/ascend_store/test_pool_worker.py:955-966` 的期望与当前实现一致。该项不应作为当前版本的 finding。

### 基本成立但需精确定义：协议兼容性风险

第一次报告指出新增 frame 破坏旧 client/server 互操作，这个方向成立；本次复核进一步确认了 server 线程无异常保护和 decoder 只读取第一个 sequence buffer，因此应将其升级为“旧 client -> 新 server 请求永久阻塞”的 P1，而不仅是一般的“返回错误”。

### 成立但应降级：非对齐 assert

第一次报告把非对齐输入描述为必然运行时错误，缺少对上游 scheduler 对齐不变量的核对。本次复核将它降为 P2：这是脆弱的接口契约和错误处理问题，但不是已证明的正常路径必现 bug。

## 5. 已验证的正确部分

以下路径的实现方向和当前单测是相互一致的：

1. scheduler 在 SUFFIX 模式按 `num_computed_tokens // hash_block_size` 切分 request hashes，见 `pool_scheduler.py:583-598`；
2. client 将 mode 作为显式 frame 传递，server 将其转成 `LookupHashMode` 后传给 worker，见 `pool_scheduler.py:1200-1209` 和 `ascend_store_connector.py:318-330`；
3. `HBMCachedBlockHashList` 保留逻辑上的绝对 hash index，见 `coordinator.py:35-77`；
4. 空 suffix 和 HBM-only fallback 有显式处理，见 `pool_worker.py:2420-2431`；
5. 新增测试覆盖了 `hbm_hit_tokens=0`、suffix slicing、coordinator marker、多 group fallback 和首个 suffix miss。

这些正确部分不能抵消 wire protocol 的 P1 风险和验收交付件缺失。

## 6. 验证记录与限制

已执行：

- `git diff --check HEAD~4..HEAD`：通过；
- `python -m compileall -q vllm_ascend tests/ut/distributed/ascend_store`：通过；
- 读取并核对 issue #14145 当前正文；
- 读取 vLLM 依赖提交 `58d3918e3ea0a544ffedadad2ba84559e9c51d8f` 的 `MsgpackDecoder`、`BlockHash`、hash granularity 和 cache manager 实现；
- 静态核对 4 个提交的全部生产代码和新增测试。

未能执行单测：当前环境缺少 `pytest` 和 `numpy`，直接运行 unittest 导入时失败。因此不能声称单测全绿；新增 mock 测试也不能替代真实 codec/ZMQ/NPU 验证。

## 7. 修复和重新验收顺序

1. 先设计协议版本/能力协商，或实现新 server 对旧 frame 的兼容解析；对未知 mode、缺 frame 和 decoder 异常返回可观测错误，禁止 server lookup 线程静默退出。
2. 增加真实 `MsgpackEncoder/MsgpackDecoder` + in-process ZMQ round-trip 测试，覆盖 old/new 四种组合、空 suffix、HBM-only、malformed mode。
3. 明确 SUFFIX 的对齐契约；将生产路径的 `assert` 替换为初始化校验或可恢复回退，并覆盖 CP/混合 KV group/SpecDecode 调用链。
4. 补充 `lookup_hash_mode` 文档、配置示例、默认值、合法值和升级兼容策略。
5. 在 issue 指定的 Ascend 硬件上完成 16K/64K payload bytes、编码时间、ZMQ round-trip 和 lookup latency 对比，并记录型号、卡数、TP/CP/PP。
6. 整理提交信息并补齐 `Signed-off-by`。

## 8. 最终判定

这 4 个提交已经实现了 SUFFIX lookup 的主要代码骨架，且 HBM 前缀与首个远端 miss 的核心边界在当前代码中处理正确。但新增 RPC frame 是不兼容且无容错的协议变更，旧 client -> 新 server 会导致请求永久等待；同时 issue 要求的真实性能、设计和端到端协议证据缺失。

**最终判定：REQUEST CHANGES。修复 P1-1/P1-2 并补齐真实协议与性能验证后，才具备重新审查条件。**
