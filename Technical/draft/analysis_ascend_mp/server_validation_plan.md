# Server Validation Plan

本文件只描述后续在服务器上的验证，不代表本轮已经运行过。所有场景应使用隔离环境、独立 endpoint 和可回收的临时 backend；不要在生产 endpoint 上发送恶意序列化 payload 或故障注入请求。

## 1. 验证前置条件

- 固定代码版本为 `0d85d6b7f67a1e79f0255a619bee82aaa7aa616e`，记录实际 vLLM commit、Python、pyzmq、cloudpickle 和 backend 版本。
- 使用 `tcp://127.0.0.1` 或受 ACL 保护的临时 endpoint；跨主机测试前先确认防火墙和监听地址。
- 为 server、client、scheduler、worker 和 backend 配置独立日志文件，至少记录 request id、identity、session id、service count、executor busy、close/recovery 和 request status。
- 准备能区分“缓存命中但内容未加载”的哨兵 KV：远端和本地使用不同的已知数值，并在 forward 前后读取目标 block。
- 准备可控的 fake scheduler/worker/store，用于 close failure、factory failure、binding failure 和 executor saturation 注入。

## 2. P1 业务验证

### 2.1 同步 external hit

场景：`load_async=False`，注册 rank-0 worker 和 scheduler，使 lookup 返回一个完整 block 的正命中。

观察：

- scheduler 返回的 external tokens、分配的 block ids；
- `update_state_after_alloc` 是否产生任何 load metadata；
- worker `start_load_kv`、attention load/save、`wait_for_save` 的实际调用；
- 本地 block 在 forward 前是否等于远端哨兵；
- 输出是否与“本地预填充相同 KV”基线一致。

通过标准：只有在代码确实把远端 KV 写入本地 buffer 且结果一致时才通过。若 connector 被定义为 lookup-only，则应在配置/初始化时拒绝该场景，而不是返回正 external hit。

### 2.2 异步 external hit

场景：`load_async=True`，lookup 返回正命中。

观察：`WAITING_FOR_REMOTE_KVS` 的进入和退出、每轮 `KVConnectorOutput.finished_recving`、`get_finished` 返回值、请求最终 token count。

通过标准：请求必须在有界时间内产生 `finished_recving` 并恢复调度；若不支持异步，应在初始化时明确拒绝。

### 2.3 save/producer

场景：producer 或 `kv_role=kv_both` 完成一个请求，随后用独立 lookup client 查询新 block。

观察：`save_kv_layer`/`wait_for_save`、`request_finished`、远端 store key/value 和下一请求命中。

通过标准：保存完成后远端数据可读；若设计不保存，配置必须显式限制为 lookup-only consumer。

## 3. 安全和身份边界验证

### 3.1 endpoint 边界

记录 server 实际 bind URL、是否监听 `*`、网络 ACL、endpoint 是否与不可信租户共享。只在隔离子进程执行无害序列化探针，验证服务是否在反序列化阶段执行 reduce；不得使用真实命令执行型 payload。

通过标准：不可信客户端无法连接，或服务端在认证前拒绝 payload；registration 不应直接执行任意对象反序列化。

### 3.2 identity replacement

先注册正常 scheduler/worker，再用不同 session 但相同 identity 的第二 client 注册；分别测试新 factory 成功和故意失败。

观察：旧服务是否继续可用、切换是否原子、失败后 scheduler/worker count、lookup 和 close 次数。

通过标准：未授权 client 不能替换；新 factory 失败时旧 service 和 lookup 必须保留。

## 4. RPC 并发、超时和关闭

- 多 client 并发提交不同 identity 的 lookup/renew/unregister，核对 response request id、method、status 一一匹配。
- 同一 affinity key 提交顺序敏感的 fake handler，确认同 key 串行、不同 key 不被不必要阻塞。
- 填满 scheduler/worker executor，确认新请求得到 BUSY，Future 只完成一次，已完成响应不会被 abort 响应覆盖。
- 在 handler 运行、排队、deadline 即将到期、断连和重连四个时刻分别调用 `request_stop`、`close`、`abort`。
- 对 malformed request 使用有限 timeout；确认服务端不会因为一个坏帧停止，也不会让无 deadline Future 无界悬挂。
- 注入 ZeroMQ send/recv/context error，检查所有 Future 是否最终完成；重点验证 `MPClient._process_outbound` 中 send 异常发生在 pending map 插入之前的路径。

通过标准：每个 request Future 只完成一次；正常 drain 不接收新请求；abort 的运行中 handler、排队请求和 executor 均按文档收敛；重复 close/abort 不崩溃。

## 5. registration、lease、recovery

- 首次注册、同 session 同 fingerprint 重试、同 session 不同 fingerprint、不同 session 替换、旧 session stale 操作。
- server 未启动时 client 注册，再启动 server，确认 lease loop 恢复；server 重启后确认 recovery 是否保留同 session。
- 让 scheduler 只 renew，worker 不 renew，确认 worker 过期而 scheduler lookup 返回 miss，不发生错误 fallback 到其他 DP/rank worker。
- 注入 owner close failure，确认 failed service 不进入 recoverable；恢复成功前不允许创建第二个实例。
- 循环大量 session replacement，记录 retired session 数量和进程内存，确认有界清理策略。
- 在 registration publish 后、lookup-store binding 期间制造 executor busy/close，确认失败 registration 回滚 service。

## 6. 输入、兼容性和资源上限

- 空/截断/非 UTF-8/错误整数宽度/负数/超大整数/重复 block hash/超大 hash list。
- 正常和超大 registration payload，记录反序列化前后的 RSS 和响应时间。
- 设置 `use_layerwise=True`、hybrid KV、多 KV group、TP/PP/DP rank 组合，记录 MP scheduler/worker 实际属性与 lookup key。
- 使用 prompt embeddings-only Request，确认是否被明确拒绝；不要把异常静默当作 cache miss。
- 高并发 client queue、服务器 executor pending、ZMQ HWM 逐级增加，记录 BUSY、timeout、RSS、线程数和 renew 成功率。

通过标准：输入错误返回明确协议错误；超限请求在 admission 阶段被拒绝；不支持的 layerwise/prompt embedding 配置显式报错；资源指标在配置上限内。

## 7. 证据留存

每个场景保留：代码版本、配置、启动日志、服务端日志、客户端日志、关键 request/response frame 摘要、service count、线程/进程退出码、异常类型和观察到的时间线。服务器结果只用于补充本报告的“需服务器确认”项，不得覆盖已经由静态代码直接确认的逻辑矛盾。
