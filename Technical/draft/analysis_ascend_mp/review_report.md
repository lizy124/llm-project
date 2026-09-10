# `0d85d6b7` 静态逻辑审核报告

## 1. 审核结论

审核对象是仓库 `D:\lzy\project\kv_pool\code\vllm-ascend-mp` 的分支 `test/ascendstore-mp`，固定提交 `0d85d6b7f67a1e79f0255a619bee82aaa7aa616e`。参考 vLLM 仓库为 `D:\lzy\project\kv_pool\code\vllm`，当时 HEAD 为 `58d3918e3ea0a544ffedadad2ba84559e9c51d8f`。

本轮只做了 Git 元数据和源代码阅读，没有启动服务、创建进程、连接 ZeroMQ、运行测试、静态检查或构建命令。报告中的“静态确认”表示结论可以由代码调用链直接推出；“需服务器确认”表示影响大小、部署边界或具体后端行为还需要在用户提供的服务器环境中验证。

最重要的结论是：MP connector 当前能够把服务端 lookup 命中返回给 vLLM 调度器，但 worker 侧所有 KV 加载、保存和完成通知钩子都是 no-op，MP 服务端也没有对应的 KV 数据传输 RPC。只要实际配置产生正的 external hit，调度器就会分配外部 KV 的块并继续执行，而没有代码把外部 KV 写入本地 vLLM KV buffer；开启异步加载时还会把请求置于 `WAITING_FOR_REMOTE_KVS`，但没有完成通知路径。这是静态已确认的 P1 逻辑缺陷，除非该 connector 被明确限定为“只 lookup、不承担 KV transfer”，并且配置层同时禁止产生 external hit。

此外还确认了未认证 endpoint 上的 `cloudpickle.loads`、无认证 identity 替换、注册绑定失败后的非事务状态、owner close 失败后仍进入 recoverable 状态、retired session 无界增长、无界请求/载荷资源使用、layerwise 配置静默降级，以及仓库明确禁止的 `cloudpickle` import。安全问题和资源问题均注明了部署触发条件，没有把条件性风险表述为所有部署必然发生。

## 2. 审核范围与调用链

本提交新增的生产代码集中在：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_mp_connector.py`
- `.../ascend_store/mp/kv_cache_protocol.py`
- `.../ascend_store/mp/kv_cache_client.py`
- `.../ascend_store/mp/kv_cache_server.py`
- `.../ascend_store/mp/kv_cache_service.py`
- `.../ascend_store/mp/lookup_worker.py`
- `.../ascend_store/mp/registration.py`
- `.../ascend_store/mp/rpc/{protocol,client,server,executor,error}.py`
- `.../ascend_store/mp/service/{lifecycle,error}.py`

已闭合的主要调用链如下：

```text
AscendStoreMPConnector
  -> KVCacheClient.register_scheduler/register_worker
  -> MPClient
  -> MPServer
  -> KVCacheServiceManager
  -> MPKVPoolScheduler / LookupKVPoolWorker
  -> KVPoolScheduler.get_num_new_matched_tokens
  -> worker.lookup_scheduler
```

与当前 vLLM 调度和 worker 代码对照后确认：

- vLLM scheduler 在 `vllm/v1/core/sched/scheduler.py:777-820` 调用 connector lookup，并把返回的 external tokens 纳入调度决策。
- vLLM scheduler 在 `vllm/v1/core/sched/scheduler.py:1005-1014` 调用 `update_state_after_alloc`。
- 异步返回时，scheduler 在 `vllm/v1/core/sched/scheduler.py:1031-1051` 将请求放入 `WAITING_FOR_REMOTE_KVS`。
- worker runner 在 `vllm/v1/worker/kv_connector_model_runner_mixin.py:85-105` 调用 `start_load_kv`、`wait_for_save` 和 `get_finished`。
- `vllm/distributed/kv_transfer/kv_connector/v1/base.py:465-524` 明确了 scheduler lookup/update 契约，`base.py:304-405` 明确了 worker load/save/finished 契约。

## 3. Findings

### [F-001] [P1] lookup 命中与实际 KV transfer 脱节

位置：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_mp_connector.py:72-103`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/kv_cache_server.py:67-77,134-142`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:521-627`
- 参考 `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py:777-820,1005-1051`
- 参考 `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\kv_connector_model_runner_mixin.py:85-105`

触发条件：配置 `AscendStoreMPConnector`，服务端 lookup 返回正的 external hit。配置 `load_async=True` 时还会触发异步分支。

逻辑链路：

1. connector 的 `get_num_new_matched_tokens` 将 `KVCacheClient.lookup` 结果原样返回（72-79）。
2. 服务端只暴露 register/unregister/renew/lookup 路由（67-77）；没有传输 KV tensor、block copy、load completion 或 save RPC。
3. vLLM scheduler 根据 external hit 分配 KV blocks，并调用 `update_state_after_alloc`（参考 scheduler:1005-1014）。
4. connector 的 `update_state_after_alloc`、`start_load_kv`、`wait_for_layer_load`、`save_kv_layer`、`wait_for_save` 全部直接返回（81-103），`build_connector_meta` 也只返回空 metadata（84-85）。
5. 该 connector 没有覆盖 `get_finished` 或 `request_finished`，因此使用基类的 `get_finished -> (None, None)` 和 `request_finished -> (False, None)`（参考 vLLM base:369-385,559-578）。
6. `KVPoolScheduler.get_num_new_matched_tokens` 在 `load_async` 下会返回 `True`（pool_scheduler:627）；vLLM scheduler 随后把请求放到 `WAITING_FOR_REMOTE_KVS`（scheduler:1031-1051），而 no-op connector 没有产生 `finished_recving`。

违反的不变量/预期：vLLM connector 契约要求 scheduler 报告的 external KV 必须能够在 worker 侧加载到本地 paged KV buffer；异步加载必须最终通过 `finished_recving` 推进请求。只实现 lookup 不能同时宣称可加载 external tokens。

影响：

- 同步路径：新分配的本地 KV block 没有外部数据写入，模型可能读取未初始化、旧内容或错误内容，导致输出错误。
- 异步路径：请求可能永久停留在 `WAITING_FOR_REMOTE_KVS`，因为没有完成通知。
- producer/save 场景：没有保存钩子，运行过程中产生的 KV 不会通过该 connector 写入远端 store，后续 lookup 命中也没有可靠来源。

证据：上述每一步都由提交中的具体方法实现和当前 vLLM 调用点直接连接，属于“已由静态逻辑确认”。“实际输出错误”与“异步请求最终表现”仍需服务器环境用可观测用例确认。

建议验证：服务器上使用包含已知数值的远端 KV，分别测试 `load_async=False/True`；观察本地 KV buffer、forward 输出、请求状态、`finished_recving` 和 `get_finished` 返回值。另用 producer 配置确认远端 store 是否出现新写入。

修复方向：二选一并在配置层明确化：

- 实现完整 worker transfer：metadata、load/save、完成/失败状态和 request lifecycle；或
- 将 MP connector 明确定位为 lookup-only，在产生 external hit、`load_async`、producer/save 或 layerwise 配置时直接拒绝，而不是返回可分配的 external tokens。

### [F-002] [P1，条件性安全风险] 未认证 endpoint 对 registration payload 执行 `cloudpickle.loads`

位置：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/kv_cache_protocol.py:8,52-80`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/kv_cache_server.py:43-77`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/rpc/server.py:106-155`

触发条件：非完全可信的客户端能够访问 `KVCacheServer` 的 ZeroMQ endpoint。服务端绑定地址可以是 TCP 或其他由调用方传入的地址，提交中没有认证、共享密钥、签名或客户端 allowlist。

逻辑链路：

1. 任意已连接的 DEALER 都可以发送 `REGISTER_SCHEDULER` 或 `REGISTER_WORKER`（server:67-77）。
2. 服务端先接收 registration payload，再在 `decode_registration` 中执行 `cloudpickle.loads(payloads[0])`（protocol:69-76）。
3. `isinstance` 类型检查发生在反序列化之后（protocol:78-80），不能阻止反序列化期间的构造代码执行。

违反的不变量/预期：跨信任边界的 RPC 不应直接反序列化未经认证的任意对象。

影响：在 endpoint 可被非信任客户端访问时，恶意 payload 可导致服务端进程任意代码执行；即使不利用代码执行，也可发送超大 payload 消耗内存。该风险取决于部署是否严格限制 endpoint，不应表述为所有本机 IPC 部署必然存在。

证据：反序列化调用、路由暴露和缺少认证均在代码中直接可见；未执行恶意 payload。

建议验证：只在隔离的临时服务器进程中验证一个无害的自定义 reduce 对象是否能在服务端触发标记，不要在生产服务器执行恶意样本；同时确认实际 bind 地址和网络 ACL。

修复方向：优先改为显式、版本化、长度受限的结构化 schema（例如 msgpack 基本类型）；若必须使用 pickle 类协议，应增加认证和签名，并在协议层设置 payload 大小上限。

### [F-003] [P1，条件性安全/可用性风险] identity 没有所有权证明，任意新 session 可替换活动服务

位置：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/service/lifecycle.py:109-123,292-314`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/kv_cache_server.py:98-112`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/registration.py:43-83`

触发条件：攻击者可以访问 endpoint，并知道目标 `engine_id` 与 `data_parallel_rank`（worker 还需知道 rank）。

逻辑链路：

1. registration 的 identity 只由 payload 中的 engine/rank 字段构成，没有凭据或旧 session proof（registration.py:43-83）。
2. `ServiceLifecycleManager.register` 发现同 identity 但不同 session 时，先将旧 session 加入 retired、删除 `_services`，再把旧 service 交给 `_create_and_publish`（lifecycle.py:114-123）。
3. `_create_and_publish` 先关闭旧 service，再调用新 factory；factory 失败时旧 service 已经被移除（lifecycle.py:301-311）。
4. server 对 register 只做对象类型和 identity/config 一致性检查，没有验证调用方是否为旧 owner（kv_cache_server.py:98-112）。

违反的不变量/预期：同一 identity 的活动服务应只能由其合法 owner 替换，不能由任何能连 endpoint 的客户端夺取或驱逐。

影响：可驱逐活动 scheduler/worker，使 lookup 变成 miss；恶意或不兼容的新 registration 还可能在 factory 失败后留下“旧服务已删、新服务未发布”的停机窗口。跨租户共享 endpoint 时属于越权和拒绝服务风险。

证据：替换顺序和缺少所有权校验是静态确定的；攻击者实际是否能构造一个可通过 factory 的完整 `VllmConfig` 需要服务器环境确认。

建议验证：隔离环境中注册一个正常 service，再用不同 session 但相同 identity 的第二客户端注册；记录旧 service close、scheduler/worker count、lookup 结果和 factory 失败后的状态。

修复方向：在认证层绑定 identity 与 owner；替换前要求旧 session 的授权凭据；新 service factory 成功并完成绑定后再原子切换，失败时保留旧 service。

### [F-004] [P2] scheduler 注册在 lookup-store 绑定失败后仍可能留下已发布 service

位置：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/kv_cache_service.py:98-107,212-218`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/kv_cache_server.py:98-104`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/rpc/executor.py:116-136`

触发条件：scheduler registration 已成功创建并发布 service，但 worker executor 在 `_schedule_lookup_store_binding` 的 `submit(...).result()` 处因容量不足或 executor 关闭而抛出异常；自定义 worker 的 `bind_lookup_store` 抛异常也会触发同类问题。

逻辑链路：

1. `register_scheduler` 先完成 `_schedulers.register(...)` 并发布 scheduler（service:98-105）。
2. 发布后才调用 `_schedule_lookup_store_binding`（service:106）；该函数通过 worker executor 非阻塞提交并等待（service:212-218）。
3. `KVCacheServer._handle_register_scheduler` 只把 `ServiceBusyError` 映射为 BUSY，没有对已发布 service 做回滚（server:98-104）。
4. 因此客户端收到 BUSY/ERROR、保持 `_registered=False`，但 server 仍持有 scheduler，直到租约过期或后续重试清理。

违反的不变量/预期：registration 请求失败时不应留下客户端无法感知和无法正常注销的活动 service。

影响：长期运行时可能累积孤儿 scheduler、占用 backend/store 资源，并使 `scheduler_count` 与客户端状态不一致；重试成功时问题可能被掩盖。

证据：发布和绑定顺序、executor 的容量拒绝路径均由代码直接确认；实际容量压力下的频率需服务器确认。

建议验证：服务器上将 worker executor 的 pending/running 容量填满，再发送 scheduler registration；检查响应、`scheduler_count`、lease 过期前的 `items()` 和后续 unregister 行为。

修复方向：把 factory、绑定和 publish 组成事务；绑定失败时显式 unregister/close 已创建 service；或先完成绑定再发布，并为 retry 使用幂等 rollback。

### [F-005] [P2] owner close 失败后仍把旧 session 标记为 recoverable

位置：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/service/lifecycle.py:208-224`
- `.../service/lifecycle.py:351-360`
- `.../service/lifecycle.py:388-396`

触发条件：lease expiration 或 manager close 通过 `owner_close_handler` 关闭 service 时，owner lane/executor 或 service `close()` 抛出普通 `Exception`。

逻辑链路：

1. `expire_leases` 先把 service 从 `_services` 移到 `_expiring`（208-220）。
2. `_close_on_owner_safely` 捕获 close 异常，只记录日志，不返回失败状态（388-396）。
3. `_finish_expiration` 无条件移除 `_expiring`；只要 session 尚未 retired，就写入 `_recoverable_sessions`（351-357）。
4. 同 session、同 fingerprint 的下一次 register 会复用 recoverable 身份并创建新 service，而旧对象可能仍持有 store/thread/socket 资源。

违反的不变量/预期：只有确认旧 owner service 已释放后，session 才能进入可恢复状态。

影响：资源泄漏、旧/新 service 并存、backend 对同一 identity 的重复占用；异常被日志吞掉后，客户端只看到可恢复成功。

证据：异常捕获和 recoverable 写入之间没有成功标志或重试状态，属于静态已确认；具体 backend close 是否“抛异常但已释放”需服务器确认。

建议验证：注入一个第一次 close 抛异常、第二次 close 可成功的 service；触发过期、同 session recovery，然后检查 close 次数、线程/socket、service count 和 lookup 来源。

修复方向：记录 close 成功/失败；失败时保留 expiring/failed 状态并重试，禁止同 session recovery，或在明确隔离旧资源后再允许恢复。

### [F-006] [P2] `retired_sessions` 按 session 无界增长

位置：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/service/lifecycle.py:75-76,289-290,351-360,255-269`

触发条件：长时间运行的 server 对同一个 identity 发生大量 session 替换、注销或过期；每次客户端重启都会生成新的 UUID session。

逻辑链路：

1. `_retired_sessions` 是 `identity -> set[str]`（75-76）。
2. `_retire_session_locked` 每次只追加 session，不做 TTL、数量上限或按 fingerprint 压缩（359-360）。
3. 只有整个 lifecycle manager `close()` 才清空该结构（255-269）。

违反的不变量/预期：历史 session 记录应有界，否则长寿命 server 的内存与 stale-session 检查成本随重启次数增长。

影响：重复重启/替换场景下内存持续增长，identity 数量较大时集合遍历和序列化/日志开销也会增加。该问题通常是 P2 运维风险，不代表一次注册就会明显泄漏。

证据：增长路径和唯一清理点由代码直接确认。

建议验证：服务器上循环同 identity 注册新 session，周期性记录每个 identity 的 retired set 大小和进程 RSS；不需要运行本轮审核，作为后续验证项。

修复方向：保留有界的最近 session、按过期时间清理、限制每个 identity 的历史数量，并保留当前 session 的 stale 拒绝语义。

### [F-007] [P2] wire payload 与 client outbound queue 没有资源上限

位置：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/rpc/client.py:70,136-158`
- `.../mp/rpc/protocol.py:88-104`
- `.../mp/kv_cache_protocol.py:129-161,268-281`
- `.../mp/rpc/server.py:521-538`

触发条件：可访问 endpoint 的客户端大量并发调用 `submit_request`，或发送包含大量/超大 block hash、registration bytes 的请求。

逻辑链路：

1. client 使用无 `maxsize` 的 `queue.Queue()`，每个 public request 都无阻塞放入（client:70,155）。
2. wire protocol 只校验 frame 类型和整数宽度，不限制 frame 数量、单帧长度或总 payload bytes（protocol:88-104）。
3. lookup 会按所有 block-hash frame 建 list，并传给 worker/store（kv_cache_protocol:143,156；service:197-198）。
4. server `recv_multipart()` 一次接收完整 multipart，未在接收前设置业务级大小/数量上限（server:521-538）。

违反的不变量/预期：跨进程服务应对请求队列、帧数量和总字节数设置有界 admission，避免异常请求耗尽内存或 worker。

影响：内存增长、CPU 解析和 backend lookup 放大，可能挤占正常 registration/renew/lookup；在未认证 TCP endpoint 上可形成拒绝服务。默认 vLLM 正常请求量是否足以触发需服务器压测确认。

证据：无界 queue、无 payload limit 和完整 multipart 接收路径均为静态事实。

建议验证：服务器上使用受控的逐级 payload/并发增长，记录 client queue、server RSS、executor busy/error、renew 成功率；设置进程级内存保护，避免影响其他服务。

修复方向：client queue 设 `maxsize` 并明确 BUSY；协议限制最大 frame 数、单帧和总 bytes；在反序列化前做大小检查；设置 ZMQ HWM/速率和每 identity 配额。

### [F-008] [P2] layerwise 配置被静默强制为非 layerwise

位置：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/lookup_worker.py:49-58,70-84`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_mp_connector.py:38-105`
- 对照 `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:76-94,210-253`

触发条件：用户在 `kv_connector_extra_config` 中启用 `use_layerwise`，或传入需要 layerwise KV layout 的 `KVCacheConfig`。

逻辑链路：

1. `MPKVPoolScheduler` 构造父类时硬编码 `use_layerwise=False`（lookup_worker:52-58）。
2. `LookupKVPoolWorker` 同样硬编码 `use_layerwise=False`（lookup_worker:73-84）。
3. MP connector 自身不读取或拒绝 `use_layerwise`，并且 transfer hooks 已是 no-op。
4. 现有 `AscendStoreConnector` 则显式读取该配置并把它传入 scheduler/worker；这表明 MP 分支不是仅仅换了 transport，而是静默改变了功能模式。

违反的不变量/预期：不支持的 cache layout/config 应在初始化时显式拒绝，不能让用户以为 layerwise 已启用而实际按普通 block lookup 运行。

影响：layerwise 数据可能被按错误 key/粒度查询，出现 false hit/miss；结合 F-001，若返回 external hit 还会进一步扩大 KV 内容错误或异步等待风险。

证据：硬编码和对照实现均已静态核实；具体模型/后端下的错误命中形态需服务器确认。

建议验证：分别用 `use_layerwise=True/False`、普通和 hybrid KV cache 配置注册，并记录实际 scheduler/worker 的 `use_layerwise`、lookup key 数量和 connector hook 行为。

修复方向：实现完整 layerwise transfer，或在 connector 初始化时检测并明确抛出“不支持 layerwise”；不要静默改成 false。

### [F-009] [P2] 新增 `cloudpickle` import 违反仓库显式 import 门禁，且目标项目未声明直接依赖

位置：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mp/kv_cache_protocol.py:8`
- `tools/check_forbidden_imports.py:35-46`
- `.pre-commit-config.yaml:113-118`
- 目标仓库 `requirements.txt`/`pyproject.toml` 未声明 `cloudpickle`；参考 vLLM 的 common requirements 才有该依赖。

触发条件：运行仓库既有的 pre-commit `check-forbidden-imports`，或在不带 vLLM 完整依赖的环境安装该项目。

逻辑链路：

1. 仓库 checker 将 `import pickle/cloudpickle` 列入禁止项，allowlist 只包含三个既有文件（checker:35-46）。
2. 本提交的新文件不在 allowlist，却直接 `import cloudpickle`（protocol:8）。
3. pre-commit 配置明确注册该 checker（pre-commit:113-118）。
4. 目标项目自己的 requirements/build-system 没有同步声明该直接运行时依赖。

违反的不变量/预期：提交应遵守仓库 import policy，并显式声明新增运行时依赖。

影响：在正常 CI/pre-commit 流程中可能直接阻塞提交；在最小环境中可能在 import 阶段失败。此项本轮没有运行 checker，结论来自规则文件和源码对照。

建议验证：后续服务器/CI 环境单独运行该仓库既有的 import policy 和最小依赖安装检查，记录失败位置；不要把本轮静态结论当作已运行结果。

修复方向：优先采用仓库允许的结构化 serializer；若确需 cloudpickle，显式评审并加入最小范围 allowlist、requirements/pyproject 依赖和安全边界说明。

## 4. 未列为 finding 的观察

- `KVCacheServer.abort()` 跳过 `KVCacheServiceManager.close()` 是实现和测试共同表达的强制终止语义（`tests/ut/distributed/ascend_store/mp/test_kv_cache.py:307-319`），本轮不把它单独定为缺陷；后续应确认进程退出模型能回收 backend 资源。
- RPC server 对无法解析 request header 的 malformed frame 会直接丢弃而不是返回 error（`rpc/server.py:521-538`）。当前有效 framing 但业务 payload 缺失仍会返回远端错误；对无 deadline 的恶意 malformed request 是否会造成长期悬挂，需要服务器验证，因此列入覆盖缺口而不是已确认 finding。
- `KVCacheClient._process_outbound` 在 `send_multipart` 抛出非 `zmq.Again` 时，当前 request 已从 queue 取出但尚未进入 pending map（`rpc/client.py:179-197`）；该 future 是否能在真实 ZeroMQ transport 错误中遗留为未完成，需要专门注入 transport fault 后确认，暂不提升为正式 finding。
- `Request.prompt_token_ids` 在 vLLM 类型上允许为 `None`，MP 协议直接调用 `len(request.prompt_token_ids)`（`kv_cache_protocol.py:139`）。现有 AscendStore connector 的共用 `KVPoolScheduler` 也依赖 token ids；本轮将它标为兼容性验证项，而不是只归因于该 commit 的独立回归。

## 5. 总体处置建议

在 F-001 得到明确产品定位和修复前，不应把该 connector 当作可用的 external-KV transfer connector 合入生产路径。F-002/F-003 要先确定 endpoint 是否跨信任边界；若是，必须在服务器测试前先收紧认证和反序列化边界。F-004/F-005 需要在服务器环境用 executor/store close fault 做故障注入。F-006/F-007/F-008/F-009 可作为合入前的工程和兼容性门禁，但其中 F-006/F-007 的实际阈值要由服务器数据补足。

本报告不作“可合入”结论；应完成修复或明确降级设计，并完成 `server_validation_plan.md` 中的服务器验证后再复核。
