# vLLM Ascend Weight Transfer 重构状态

这个目录记录 vllm-ascend `weight_transfer` 重构的背景、设计、历史 PR 和 upstream 演进。

推荐阅读顺序：

1. `README.md`：当前状态、已完成内容、后续 PR 拆分。
2. `01-weight-transfer-core-flow.md`：weight transfer 启动和同步时序。
3. `02-refactor-design.md`：重构边界、设计方案和验收标准。
4. `03-vllm-ascend-pr-history.md`：vllm-ascend 相关 PR 脉络。
5. `04-upstream-vllm-timeline.md`：upstream vLLM weight_transfer 演进。

## 当前结论

`weight_transfer_refactor` 分支已经包含完整重构视图，后续不建议继续扩大范围。下一步应先确认整体设计，再拆成 2-3 个较小 PR 上库。

当前分支已有 3 个主要 commit：

```text
cae87ad1 Refactor weight transfer common helpers
ef9630ee Share weight transfer HTTP example helpers
178fb6ff Complete weight transfer helper refactor
```

## 已完成范围

### 公共 control-plane helper

已抽出：

- `registry.py`：集中注册 `hccl`、`npu_ipc`，并维护 upstream-compatible alias `nccl`、`ipc`。
- `lifecycle.py`：封装 checkpoint-format layerwise reload 和 direct weight copy lifecycle。
- `device_mapping.py`：集中 NPU IPC host/device/chip UUID 生成逻辑。
- `examples/rl/weight_transfer_http_utils.py`：封装 RL 示例里的公共 HTTP endpoint 调用。

### E2E 测试 helper

已新增：

```text
tests/e2e/pull_request/weight_transfer_utils.py
```

职责：

- 启动或协调 weight transfer e2e 测试里的 vLLM server。
- 封装 pause/init/start/update/finish/resume 调用。
- 封装 before/after generation 行为断言。
- backend-specific 配置仍保留在各自测试中。

### Trainer-side send orchestration

已新增：

```text
vllm_ascend/distributed/weight_transfer/trainer_send.py
```

职责：

- 遍历 model parameters。
- 构造 names、dtype names、shapes 和 packed metadata。
- 编排 direct send、HTTP send、packed send。
- backend-specific 发送行为仍委托给 HCCL / NPU IPC engine。

### Packed tensor helper

当前只完成低风险公共 helper 抽取：

- packed tensor metadata。
- buffer size 计算。
- unpack 公共逻辑。

没有重写 HCCL / NPU IPC 的核心 transport loop。HCCL broadcast 和 NPU IPC handle 逻辑仍保留在各自路径中。

## 后续 PR 拆分建议

### PR 1: 公共 control-plane helper

范围：

- backend registry / alias registration
- lifecycle policy
- NPU IPC device mapping
- HTTP example helper
- 对应 UT

目的：建立共享 control-plane 边界，不触碰高风险 data-plane transfer 内部逻辑。

### PR 2: 测试和 E2E helper 清理

范围：

- e2e weight transfer helper
- e2e callsite cleanup
- HCCL 和 NPU IPC 测试共享 lifecycle/assertion helper

目的：减少测试 orchestration 重复，让不同 backend 的行为更容易比较。

### PR 3: 核心 data-plane 清理

范围：

- trainer-side send orchestration helper
- packed tensor common helper
- HCCL 和 NPU IPC backend-specific cleanup
- 聚焦的 UT / e2e 覆盖

目的：区分公共协议编排和 Ascend-specific transport 实现。

## 验证状态

当前已完成：

```text
tests/ut/distributed/weight_transfer: 38 passed
Python syntax check: passed for changed files
```

正式拆 PR / 上库前建议补充：

- Ascend/CANN 环境下的 worker UT。
- HCCL 和 NPU IPC e2e 测试。

当前本地限制：非 Ascend 环境中，部分 worker 测试会在 import 阶段因为缺少 `acl` 失败。只要失败没有发生在新重构逻辑内部，应视为环境限制，而不是本次重构行为失败。
