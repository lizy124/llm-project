# [Correctness] ZMQ lookup REQ/REP 超时、协议错误与 socket 恢复

> 编号：kv-27 | 维度：Correctness | 严重程度：高 | 建议优先级：P0
> 验收人：@赵鹏博 | 关联任务池：#9079 | 发布：2026-08-11 | 回收：2026-09-30

## 背景

Lookup client 使用同步 ZMQ REQ socket，发送后直接阻塞在 `recv`，当前没有 receive timeout。server daemon loop 也没有逐请求异常保护；decode、字段访问、worker lookup 或 response conversion 任一异常都可能终止 server，令 client 永久等待。

REQ/REP 具有严格状态机：REQ 超时后仍处于等待 reply 状态，REP 在 receive 后也必须完成一次 send。仅增加 `try/except` 或在原 socket 上直接 retry 会触发 EFSM，不能形成可靠恢复。

永久等待风险可由代码闭合证明；具体 timeout、重试次数和退避值则取决于部署环境的正常 lookup 延迟分布，不能凭经验写死。

## 实现前验证

1. 在当前固定的 vLLM/pyzmq 版本上用最小 REQ/REP 测试验证 timeout 后原 REQ socket 的 EFSM 行为、`linger=0` 重建和 REP receive 后异常的恢复方式。
2. 在目标机器记录正常 lookup 的 p50/p95/p99/max 延迟和启动抖动，按数据选择 timeout、重试次数和退避，并设置保守上限；超时值不要求依赖 kv-24 才能落地。
3. 用故障注入覆盖 server 不回复、延迟回复、坏 frames、handler 抛错、空/截断/超长 response 和连续重建，确认降级只损失 remote 增量命中。

## 任务

1. client 使用经前置验证确定的有界 timeout；超时或协议错误后以 `linger=0` 关闭并重建 REQ socket，再执行有限次数重试。
2. 定义版本化 success/error response，明确区分合法 `hit=0`、server 错误和协议损坏。
3. server 对 receive 后的请求异常发送 typed error reply，保持 REP 状态机可继续服务；socket/receive 本身失败时重建 server socket或明确终止 connector。
4. 校验 frame 数、字段长度和类型、group/hash 解码、hit 范围及 response 长度。
5. 重试耗尽后安全降级为 remote miss，并输出限频日志和指标；不得静默吞错。

## 验收标准

### 1. 功能正确性
- server handler 异常后 client 收到 error reply，server 可继续处理下一请求
- client timeout 后不会永久阻塞，下一次 lookup 可在重建 socket 后成功
- 坏报文、截断 response、非法 hit 与合法 hit=0 可明确区分
- 重试耗尽只放弃远端增量命中，不破坏已有本地 HBM hit
- socket 重建无 FD/context 泄漏

### 2. 回归保护
- 现有单测全绿
- 新增 server 异常、client timeout、坏报文、error reply、timeout 后恢复和重复重建测试

### 3. 交付件
- PR + REQ/REP 状态机说明 + timeout/降级策略 + 单测

## 证据

- client：`pool_scheduler.py:1156-1191`
- server：`ascend_store_connector.py:293-339`
- 上游 socket helper：vLLM `58d3918e3e` 的 `utils/network_utils.py:make_zmq_socket`

## 重点关注

- timeout 配置最终由 kv-24 parser 管理，但本任务不能依赖 kv-24 才能修复，可先提供带校验的默认值
- timeout 后禁止复用仍在 waiting-reply 状态的 REQ socket
- 本任务只处理可靠性；lookup batching 和 payload 缩减分别由独立任务评估

## 环境约定
- vllm-ascend：审核基线 `d5e9816065ede613327d93908f87fee9f5c47128` 或提交时最新 main
- 硬件：Ascend NPU（注明型号和卡数）
- 关联任务池：#9079
- 验收人：@赵鹏博
