# Coverage Gaps

本轮已阅读提交中的相关测试文件，但没有执行任何测试。测试只能作为作者意图和覆盖范围参考，不能作为运行通过证据。

## 1. 现有测试已表达的范围

| 领域 | 已看到的测试意图 | 仍缺少的关键证明 |
|---|---|---|
| RPC framing/Future | response matching、timeout、busy、drain、abort、affinity | 真实 transport error 下 Future 完成性、malformed frame 的无 deadline 行为 |
| registration | idempotent registration、fingerprint conflict、stale session、recovery、lease | 多线程 reconfigure/close race、publish 后 binding failure rollback |
| lifecycle | expiration、owner lane、recoverable、retired session | owner close 抛异常后的禁止恢复语义、retired set 长期增长 |
| lookup | scheduler/worker/DP/rank-0 映射、原有 lookup 逻辑复用 | 实际 KV tensor 内容、真实 backend key/layout、worker load/save |
| connector | registration、lookup delegation、empty metadata、shutdown | `update_state_after_alloc`、load/save、`get_finished`、`request_finished` 的真实 vLLM step |
| protocol | registration/lookup round trip、基本 malformed payload | payload bytes/frame limits、`num_computed_tokens` 关系校验、prompt embeddings-only |
| server shutdown | RPC abort/drain 和“abort 跳过 graceful service close”的意图 | 进程退出时 backend/socket/thread 回收、abort 后再次启动/恢复 |

## 2. 未覆盖或证据不足的逻辑分支

### G-001：external hit 与 KV 内容一致性

现有 connector 单测只检查 `KVCacheClient.lookup` delegation，且明确允许 `build_connector_meta` 返回空 metadata；没有测试 scheduler 返回 external hit 后 worker 是否真的写入 KV buffer。需要 F-001 的哨兵 KV 场景。

### G-002：异步等待闭环

没有覆盖 `load_async=True` 的真实 vLLM scheduler 状态转换，也没有测试 `WAITING_FOR_REMOTE_KVS -> finished_recving -> running`。目标 connector 没有 `get_finished` 覆盖，需服务器确认是否长期等待。

### G-003：save/producer 语义

没有覆盖 `save_kv_layer`、`wait_for_save`、`request_finished` 和远端 store 新写入。现有 `AscendStoreConnector` 的 save 逻辑不能证明 MP connector 具备同等行为。

### G-004：未认证 endpoint

没有认证/ACL/租户隔离测试，也没有安全边界文档。禁止在生产环境发送恶意 cloudpickle；应在隔离进程执行无害探针并记录 endpoint 实际暴露范围。

### G-005：identity replacement 与原子切换

现有测试覆盖“新 session 替换”和“重绑定 store”正常路径，但没有覆盖新 factory 失败、binding executor busy 或替换过程中的并发 lookup。需要验证旧 service 是否会在失败前被删除。

### G-006：owner close failure

现有 lifecycle 测试覆盖 close 成功和 maintenance failure，但没有模拟 `_owner_close_handler` 抛异常后同 session recovery。该分支直接决定 F-005 的资源风险。

### G-007：无界资源

没有 client queue 上限、multipart 总 bytes、block-hash frame 数、registration payload 大小或逐 identity 配额测试。需要服务器侧逐级负载并设置内存保护。

### G-008：layerwise 静默降级

没有断言 MP scheduler/worker 的 `use_layerwise` 与配置一致，也没有测试 layerwise key layout。现实现把它硬编码为 false，至少应新增“不支持时显式失败”的测试。

### G-009：prompt embeddings-only

vLLM `Request.prompt_token_ids` 类型允许 `None`，MP protocol 使用 `len(request.prompt_token_ids)`。由于现有 AscendStore 共用 scheduler 也依赖 token ids，尚不能把它归为该 commit 独立回归；需要产品范围确认和显式拒绝测试。

### G-010：MPClient send exception race

`_process_outbound` 在发送成功前把 Future 标记为 running，非 `zmq.Again` 异常会被外层 `_io_loop` 捕获，而当前 request 尚未进入 pending map。需要可控的 socket/context fault 注入，确认 Future 是否遗留未完成；当前没有足够证据把它列入正式 finding。

### G-011：仓库门禁与直接依赖

本轮只阅读了 checker 和配置，没有运行 pre-commit，也没有在最小环境安装依赖。应单独确认新 `cloudpickle` import 是否被 checker 拒绝、目标发行包是否提供 cloudpickle。

## 3. 建议新增测试的最小集合

1. 同步 external hit 的 sentinel KV load end-to-end。
2. 异步 external hit 的 bounded completion 和 `finished_recving`。
3. producer save 后独立 client lookup 命中。
4. registration replacement 的 factory/binding failure rollback。
5. owner close failure 后禁止 recovery，成功重试后才允许 recovery。
6. payload/frame/queue size limits 和 per-identity admission。
7. layerwise 配置显式 reject 或完整 transfer 的契约测试。
8. transport send exception、disconnect、late response 下所有 Future 的终态测试。
9. endpoint ACL/auth 和 registration payload 不执行任意对象代码的安全测试。
10. pre-commit forbidden-import 和最小依赖安装验证。

## 4. 不应当作为覆盖证据的内容

- 本提交中的 mock/fake scheduler 或 fake worker 只能证明参数和调用顺序，不能证明真实 backend 的 KV 内容。
- 测试中的 `process.terminate()` 只能结束测试进程，不能证明 `KVCacheServer.abort()` 释放了 service/backend 资源。
- 现有 connector 单测对 no-op hook 的断言不能证明 no-op 满足 vLLM transfer contract。
- 本文及其他审核文档均未执行代码、测试、静态检查或构建。
