# `0d85d6b7` Ascend Store MP 提交逻辑审核计划

## 1. 审核目标与边界

### 1.1 审核对象

- 仓库：`D:\lzy\project\kv_pool\code\vllm-ascend-mp`
- 分支：`test/ascendstore-mp`
- 固定提交：`0d85d6b7f67a1e79f0255a619bee82aaa7aa616e`
- 提交主题：`feat(ascend_store): add multiprocessing KV cache service`

该提交新增了多进程 KV Cache 服务、RPC 协议与客户端/服务端、Scheduler/Worker 注册和查找流程、生命周期租约管理，以及 `AscendStoreMPConnector` 集成和对应测试。

### 1.2 本轮明确不做的事情

- 不启动服务、不创建进程、不连接 ZeroMQ endpoint。
- 不运行单元测试、集成测试、静态检查、类型检查或构建命令。
- 不修改被审查代码，不因“看起来可疑”直接修复。
- 不把测试文件中已有的断言结果当作实际运行结果；测试仅作为接口意图和覆盖范围的参考。

### 1.3 本轮要回答的问题

1. 协议帧、身份、请求 ID、响应状态和错误编码是否始终匹配。
2. 并发访问、线程池、队列、Future、亲和性路由是否存在竞态、死锁、饥饿、乱序或丢请求。
3. 注册、续租、过期、恢复、注销和会话淘汰是否满足状态机不变量。
4. Scheduler/Worker 的 KV lookup 语义是否正确，尤其是 rank、数据并行、token 数和异步标志。
5. 连接器是否正确接入当前 vLLM API、配置、角色和关闭流程。
6. 异常、超时、断连、服务端终止和资源释放后是否能得到可预测行为。
7. 是否存在安全边界、输入校验、资源上限或兼容性方面的高风险问题。
8. 哪些结论必须留到服务器上运行后才能确认，以及应如何设计验证用例。

## 2. 审核原则

- 以该提交及其父提交的差异为主，避免把分支后续变化误判为本提交行为。
- 先建立端到端调用链，再逐个审查局部实现；不只按文件逐行阅读。
- 所有问题必须绑定到文件、行号、调用路径和可复现的触发条件。
- 优先审查可能造成数据错误、跨请求污染、死锁、服务不可用或静默降级的问题。
- 将“代码逻辑确定的问题”和“需要运行环境验证的风险”分开记录。
- 对每个结论给出不变量、反例或时序证据，而不是只描述代码风格。

## 3. 审核范围清单

### 3.1 生产代码分组

1. **入口与连接器**
   - `vllm_ascend/distributed/kv_transfer/__init__.py`
   - `.../ascend_store/ascend_store_mp_connector.py`
2. **低层 KV Cache 客户端/服务端**
   - `.../ascend_store/mp_kv_cache.py`
   - `.../ascend_store/mp_protocol.py`
3. **MP KV Cache 领域协议与服务编排**
   - `.../ascend_store/mp/kv_cache_protocol.py`
   - `.../ascend_store/mp/kv_cache_client.py`
   - `.../ascend_store/mp/kv_cache_server.py`
   - `.../ascend_store/mp/kv_cache_service.py`
   - `.../ascend_store/mp/registration.py`
   - `.../ascend_store/mp/lookup_worker.py`
4. **通用 RPC 层**
   - `.../ascend_store/mp/rpc/protocol.py`
   - `.../ascend_store/mp/rpc/client.py`
   - `.../ascend_store/mp/rpc/server.py`
   - `.../ascend_store/mp/rpc/executor.py`
   - `.../ascend_store/mp/rpc/error.py`
5. **服务生命周期层**
   - `.../ascend_store/mp/service/lifecycle.py`
   - `.../ascend_store/mp/service/error.py`

### 3.2 测试代码的使用方式

测试文件用于反推作者期望的协议、错误、并发和生命周期语义，并用于识别未覆盖的分支；本轮不执行测试。重点参考：

- `tests/ut/distributed/ascend_store/mp/test_rpc.py`
- `test_service_lifecycle.py`
- `test_registration.py`
- `test_registration_recovery.py`
- `test_kv_cache_protocol.py`
- `test_kv_cache_service.py`
- `test_lookup_integration.py`
- `test_service_lifecycle.py`
- `test_ascend_store_mp_connector.py`
- `test_mp_kv_cache.py`

## 4. 分阶段审核计划

### 阶段 A：基线、差异和依赖确认

**目标**：锁定审查对象，避免遗漏新增文件或误读外部接口。

**检查项**：

- 核对提交父节点、提交文件清单、生产代码/测试代码边界。
- 读取仓库中相关旧版 Ascend Store、KV connector 和 vLLM 基类实现，比较命名和生命周期约定。
- 建立外部 API 依赖表：vLLM connector 接口、`Request` 字段、`KVCacheConfig`、`KVPoolScheduler/Worker`、ZeroMQ socket 语义。
- 标记所有线程、进程、socket、executor、Future、队列、事件和回调的创建/关闭位置。

**产出**：提交范围清单、模块依赖图、外部接口假设列表，以及待确认的版本兼容风险。

### 阶段 B：端到端架构和状态流审查

**目标**：确认真实的数据路径和控制路径。

**检查流程**：

1. 从 `AscendStoreMPConnector` 构造开始，跟踪 scheduler/worker 注册。
2. 跟踪 scheduler lookup 请求从编码、RPC 路由、executor、service manager 到 worker 的完整路径。
3. 跟踪响应返回、Future 完成、客户端状态更新和调用方返回值。
4. 单独画出正常流程、服务端忙、超时、断连、服务端重启、重复注册、过期和关闭流程。

**重点不变量**：

- 一个请求只能完成一次，不能同时返回成功和异常。
- 请求 ID、客户端身份和会话身份不能跨请求或跨客户端复用到错误对象。
- Scheduler 只能访问自己的注册状态，Worker 只能被正确的 scheduler/DP 映射调用。
- 服务关闭后不得继续接受新请求，已接受请求的处理策略必须明确。

**产出**：调用时序图、状态转移表、跨模块不变量清单。

### 阶段 C：协议、序列化和输入校验

**目标**：发现跨进程通信中最容易导致协议错位或远程崩溃的问题。

**检查项**：

- 核对 `encode_*`/`decode_*` 的帧数、帧顺序、字段类型、字节序和可选字段。
- 检查方法名、响应状态、错误 payload、deadline 和请求 ID 的编码是否具有唯一性。
- 检查空 payload、额外 payload、截断 payload、非法 UTF-8、负数、超大整数、重复 block hash 的处理。
- 验证 registration fingerprint 是否覆盖所有影响服务行为的配置字段。
- 检查异常是否能被安全地编码、传输和在客户端恢复为正确的异常类型。
- 检查协议版本缺失时，客户端/服务端是否会静默互操作或产生错误解释。

**证据要求**：每一个协议结论都记录“发送帧样式 -> 接收解析 -> 业务使用”的完整链路。

**产出**：协议帧对照表、输入校验矩阵、协议错误清单。

### 阶段 D：RPC server/client 并发正确性

**目标**：审查请求调度和响应匹配的逻辑正确性，不依赖实际压力测试。

**检查项**：

- 检查 ROUTER/DEALER 身份帧、连接事件、monitor socket 和 poller 的使用是否一致。
- 检查 outbound queue、pending map、output queue 与通知 socket 的并发访问保护。
- 检查 Future 在成功、超时、断连、关闭和服务端异常时是否只会完成一次。
- 检查 timeout deadline 的计算、poll timeout、超时清理和迟到响应处理。
- 检查 max workers、max pending requests、队列满和 executor 拒绝任务时的行为。
- 检查 affinity key 是否保证同一 scheduler/worker 的状态操作顺序，同时避免一个热点 key 阻塞无关请求。
- 检查服务端停止、drain、abort、close 的竞态：新请求、已排队请求、运行中请求分别如何处理。
- 检查异常回调、通知 socket 写入、socket close 与 poller 唤醒之间的时序。

**必须构造的静态时序场景**：响应先于超时、超时先于响应、断连发生在发送后、关闭发生在入队后、任务完成发生在 executor 关闭期间、重复 close、线程从自身调用 join。

**产出**：并发时序审查表、锁/队列所有权表、请求终态表。

### 阶段 E：注册、租约和会话生命周期

**目标**：验证服务身份管理不会产生僵尸服务、错误替换或旧客户端越权。

**检查项**：

- 检查 scheduler/worker identity 的唯一性、session ID 生成和 fingerprint 比较。
- 分析首次注册、同 session 重试、不同 session 冲突、注册工厂失败和并发注册。
- 分析 renew、lease expiration、unregister、server close 的交错顺序。
- 检查过期服务的 owner-close 回调是否在正确的 owner lane 执行，失败后是否仍然清理状态。
- 检查 recoverable session 是否可能被错误恢复、永久保留或被新 session 绕过。
- 检查 retired/stale session 的拒绝规则是否覆盖所有操作，而不只是 renew/unregister。
- 检查 lease 线程启动、停止、自身退出、网络暂时不可用和重新注册逻辑。
- 验证 `count/items/is_registered` 等观察接口是否与实际状态一致。

**关键不变量**：

- 同一 identity 在任意时刻最多有一个有效服务和一个注册 flight。
- 旧 session 不能操作新 session 的服务。
- 注册失败不能留下不可注销的 recoverable/retired 状态。
- 过期和关闭必须最终释放服务资源，且不能重复 close。

**产出**：生命周期状态机、并发注册时序、会话安全检查表。

### 阶段 F：KV Cache 业务语义与 lookup 正确性

**目标**：确认返回的匹配 token 数和异步标志符合 vLLM 调度语义。

**检查项**：

- 核对 `num_computed_tokens`、token length、hash 列表、KV cache group IDs 和 layerwise 参数的含义。
- 检查 lookup 是否错误地使用了 scheduler 的 session、engine ID、DP rank 或 worker rank。
- 检查 scheduler 到 worker 的 rank 映射、TP/PP/DP 计算及边界条件。
- 检查 worker 缺失、scheduler 缺失、lookup store 未绑定、worker 过期时的返回策略。
- 检查同步/异步 lookup 的转换，尤其是 `matched_tokens=0`、部分命中和异常时的语义。
- 检查对 `Request` 的序列化是否只传输必要且稳定的字段，是否存在不可序列化或跨版本字段依赖。
- 检查服务端业务 handler 是否在错误线程/错误 owner 上执行，是否可能并发访问非线程安全的 KV store。

**产出**：lookup 输入输出契约、rank 映射表、业务异常到返回值的映射表。

### 阶段 G：连接器和 vLLM 集成兼容性

**目标**：审查新 connector 在真实 vLLM 版本中的可用性和关闭行为。

**检查项**：

- 核对 connector factory 注册名称、模块路径和类名。
- 核对 scheduler/worker 两种 role 下构造参数和 `kv_cache_config` 要求。
- 核对基类要求的方法签名、返回值和 metadata 类型。
- 检查 `kv_connector_extra_config["kv_cache_server_url"]` 的配置校验和错误提示。
- 检查 connector 初始化中注册失败时 client 是否完整关闭。
- 检查 `shutdown` 的幂等性、异常传播和服务端不可达时的行为。
- 对照同仓库现有 connector，识别是否漏掉 save/load、metadata、request lifecycle 或 rank 配置。

**产出**：vLLM API 兼容性表、配置契约、连接器生命周期问题清单。

### 阶段 H：资源、异常、可靠性和安全审查

**目标**：识别非正常条件下的服务不可用、泄漏和输入攻击面。

**检查项**：

- socket、ZMQ context、monitor socket、通知 socket、线程和 executor 的所有权及释放顺序。
- linger、context termination、线程 join 和进程退出时是否可能阻塞。
- 远程错误信息是否泄漏内部路径、配置或敏感数据，是否可导致日志注入。
- 无认证 ZeroMQ endpoint 的部署假设、任意客户端注册风险和 endpoint 暴露范围。
- payload 大小是否有限制，恶意或异常请求是否可耗尽内存、线程或队列。
- 资源异常、工厂异常、回调异常是否被吞掉、重复记录或错误地标记为成功。
- 日志级别和关键状态变化是否足以支持服务器侧故障定位。

**产出**：资源所有权表、异常路径表、部署安全风险列表。

### 阶段 I：测试规格反查与服务器验证准备

**目标**：在不运行测试的前提下，判断现有测试是否覆盖关键逻辑，并为后续服务器测试准备最小矩阵。

**检查项**：

- 将测试按协议、正常流程、并发、超时、断连、恢复、关闭和连接器集成分类。
- 对每个高风险不变量确认至少有一个测试意图；没有覆盖的标为 gap。
- 区分只验证 mock 行为的测试与真正跨进程/跨线程行为。
- 设计服务器侧测试顺序：协议 smoke、单客户端、并发客户端、重启恢复、租约过期、优雅关闭、异常终止、长时间稳定性。
- 明确每个服务器测试的观测量：返回值、异常类型、服务计数、线程/进程退出、socket 释放、重复请求结果。

**产出**：测试覆盖矩阵、未覆盖风险列表、服务器运行测试清单和预期结果。

## 5. 审核记录与问题分级

每个发现使用统一格式记录：

```text
[ID] [等级] 标题
位置：文件:行号
触发条件：
逻辑链路：
违反的不变量/预期：
影响：
证据：
建议验证：静态确认 / 服务器运行确认
修复方向：
```

分级标准：

- **P0**：可能导致数据错误、跨租户/跨 session 越权、进程无法退出或普遍性服务中断。
- **P1**：正常部署下高概率触发的请求丢失、死锁、错误 lookup、注册失效或资源泄漏。
- **P2**：边界条件、恢复流程、兼容性或可观测性缺陷，影响局部功能或运维。
- **P3**：低风险代码质量、文档、错误信息或非关键测试缺口。

结论必须同时标注：`已由静态逻辑确认`、`需要运行确认` 或 `目前证据不足`。

## 6. 审核执行顺序

1. 完成阶段 A，锁定提交范围和外部接口假设。
2. 完成阶段 B，形成端到端调用链和状态机。
3. 优先完成阶段 C、D、E，先处理协议、并发和生命周期高风险项。
4. 完成阶段 F、G，确认业务语义和 vLLM 集成契约。
5. 完成阶段 H，审查资源、异常和部署安全边界。
6. 完成阶段 I，反查测试缺口并生成服务器验证清单。
7. 按 P0 到 P3 复核所有发现，去除重复项，确认每项都有代码证据。

## 7. 预期审核交付物

- `review_report.md`：按严重级别排序的正式问题报告，包含文件/行号和逻辑证据。
- `finding_matrix.md`：问题、触发条件、影响、静态结论和服务器验证状态矩阵。
- `server_validation_plan.md`：交给服务器环境执行的测试场景、命令入口、观测指标和通过标准。
- `coverage_gaps.md`：现有测试未覆盖的逻辑分支及建议新增测试。

本计划本身不代表该提交已经通过审核；在完成上述阶段并处理高等级问题前，不应给出“可合入”结论。
