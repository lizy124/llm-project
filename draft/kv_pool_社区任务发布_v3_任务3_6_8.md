# vLLM-Ascend KV Pool 社区任务发布稿（任务 3、6、8）

> 目标仓库：`vllm-project/vllm-ascend`
> 代码范围：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store`
> 参考基线：`main`，提交 `d5e9816065ede613327d93908f87fee9f5c47128`（2026-08-15）
> 关联任务池：[#9079 [Contribution] vLLM-Ascend 外部开发者任务池](https://github.com/vllm-project/vllm-ascend/issues/9079)
> 验收人：@赵鹏博

## 通用要求

- 认领时以最新 `main` 为准。
- 现有单测全绿，并在 `tests/ut/distributed/ascend_store/` 补充对应测试。
- 硬件验证需注明 Ascend NPU 型号、卡数和 TP/CP/PP 配置。
- 交付 PR、设计或测量说明、测试结果及回退方案。

---

## 任务 3：[Docs/Correctness] 对齐 layerwise 支持范围与 prefetch 默认值

> 建议优先级：P1

### 背景

公开文档说明 layerwise 仅支持 Memcache，但代码仍可进入非-GVA key-layerwise 分支，支持范围在文档、配置校验和运行时行为之间没有完全对齐。

同时，文档将 `layerwise_prefetch_layers` 的默认值写为 1，而 Memcache/GVA 路径的运行时默认值为 `min(num_shared_buffers, 8)`。

相关位置：

- `docs/source/user_guide/feature_guide/layerwise_kv_pool.md:40-42,81-84,235-236`
- `pool_worker.py:145-155,412-428`
- `layerwise_cache_layout.py:128-130`

### 任务

统一 layerwise 的支持范围、配置校验和默认值说明：

1. 明确非-GVA key-layerwise 的支持状态。
2. 对齐配置校验、错误信息、用户文档和测试。
3. 使 `layerwise_prefetch_layers` 的文档默认值与运行时行为一致。
4. 如需修改运行时默认值，说明兼容性、buffer 占用和性能影响。

本任务聚焦支持范围和配置契约的一致性，非-GVA layerwise 性能调优不在任务范围内。

### 验收标准

- 用户可以从文档和配置反馈中明确判断所选 backend 是否支持 layerwise。
- 不支持的配置会被明确拒绝或标记，受支持的配置行为与文档一致。
- `layerwise_prefetch_layers` 的默认值说明与实际运行时值一致。
- 新增支持范围、配置校验和默认值测试。
- 现有 layerwise 支持场景无功能和精度回归。

### 交付件

- PR 和设计说明。
- 更新后的用户文档与配置说明。
- 对应单测。

---

## 任务 6：[Correctness] ZMQ lookup RPC 超时、异常与 socket 恢复

> 建议优先级：P0

### 背景

非-layerwise 模式下，scheduler 通过 ZMQ REQ/REP 向 worker 查询外部 KV 命中。当前 `LookupKeyClient.lookup()` 同步等待 `socket.recv()`，没有接收超时；`LookupKeyServer` 的请求循环也缺少请求级异常处理。

server 无响应、报文损坏或 lookup 异常时，scheduler 可能长时间阻塞。REQ socket 在请求失败后还需要恢复到可继续请求的状态。

相关位置：

- `pool_scheduler.py:1168-1187`
- `ascend_store_connector.py:312-333`

### 任务

为 ZMQ lookup 链路增加有界等待、错误处理和恢复能力：

1. 增加可配置的请求超时。
2. server 对解码、参数校验和 lookup 异常返回明确错误。
3. client 在超时、断连或 REQ 状态异常后重建 socket。
4. 明确失败后的降级语义和日志/指标。
5. 补充正常请求、超时、畸形报文、server 异常及恢复测试。

### 验收标准

- server 无响应时，client 在配置的时间上限内结束等待。
- 单个异常请求不会导致 lookup server 静默退出。
- socket 恢复后，下一次正常 lookup 可以成功完成。
- 降级后 scheduler 能够继续重算，或向上层返回约定的可恢复错误。
- 错误日志不记录完整 hash payload。
- 正常路径、完整/部分命中和多 KV group 语义无回归。

### 交付件

- PR 和超时/降级设计说明。
- 异常与恢复单测。
- 正常路径开销对比。

---

## 任务 8：[Perf/Investigation] layerwise attention gate 队列等待与调度优化

> 建议优先级：P3

### 背景

layerwise 预取任务可能在 recv thread 中等待 `attention_start_gate`。recv thread 顺序处理队列时，队头等待会同时推迟后续任务。

当前调度还包含共享 buffer owner 等待：`prefetch_layer_map -> wait_for_save_layer` 用于保护 buffer 复用，`attention_start_gate` 用于等待计算流到达 attention 边界。评估调度优化时需要分别记录这两类等待。

相关位置：

- `pool_worker.py:1676-1699`
- `kv_transfer.py:1250-1293,1589-1621`
- `attention_fence.py:27-61`

### 任务

分析 layerwise recv 队列中的 gate 等待和任务调度，确认是否存在可安全提前执行的任务：

1. 增加时间线或 profiler 标记，区分 gate wait、save-owner wait、queue wait 和 backend transfer。
2. 覆盖不同 prefetch 窗口、多请求、独立 layer 和 buffer reuse 场景。
3. 记录任务入队、依赖满足、开始传输和完成时间。
4. 根据测量结果评估 ready queue、按 gate 分组或其他调度方案。

### 验收标准

- 给出可复现的队列等待时间线和测试配置。
- 明确区分 attention boundary 与共享 buffer owner 依赖。
- 调度优化必须保留 buffer 复用、save event 和 attention boundary 的正确性。
- 报告 TTFT、吞吐、queue wait、gate wait 和 NPU 时间线。
- 若存在优化机会，提交实现、单测和性能对比；否则提交调查结论。

### 交付件

- profiler/时间线报告和复现方法。
- 调度方案评估。
- 如实施优化，交付 PR、单测和改动前后性能数据。
