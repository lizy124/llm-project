# Ascend Store MP 代码设计与实现详细解读

## 0. 文档定位

本文针对仓库：

    D:\lzy\project\kv_pool\code\vllm-ascend-mp

以及固定提交：

    0d85d6b7f67a1e79f0255a619bee82aaa7aa616e

进行代码级解读。该提交主题为：

    feat(ascend_store): add multiprocessing KV cache service

本文不是单纯的静态 review 清单，而是尝试回答：

1. 这套代码要解决什么问题？
2. 各模块分别负责什么？
3. 一个 Scheduler lookup 请求从哪里进入、如何编码、如何路由、最终访问哪个 KV store？
4. 注册、续租、过期、恢复、注销和关闭如何工作？
5. 线程、executor、Future、socket、service 和 backend 分别由谁拥有？
6. 设计中哪些地方清晰、稳健和可复用？
7. 哪些地方只是接口占位，哪些地方已经形成实际逻辑缺陷？
8. 后续应该如何测试和验证？

### 阅读方法和证据等级

本轮只进行源码、提交元数据、现有测试和 vLLM 对应接口的静态阅读，没有：

- 启动 KVCacheServer；
- 创建实际服务进程或连接 ZeroMQ endpoint；
- 执行单元测试、集成测试、pre-commit、类型检查或构建；
- 修改被审查仓库中的生产代码。

本文使用以下证据标签：

| 标签 | 含义 |
|---|---|
| 静态已确认 | 由源码调用链和状态转移可以直接推出，不依赖运行环境 |
| 需要运行确认 | 代码路径明确，但实际影响取决于 backend、网络、模型或部署参数 |
| 目前证据不足 | 已发现可疑路径，但还不能仅凭静态阅读断言为缺陷 |

本文写入的“好处”和“不完善之处”均基于固定提交；如果分支继续变化，应重新核对路径和行号。

---

## 1. 先给出结论：这套代码实际是什么

### 1.1 设计目标

这次提交想建立一个常驻的多进程 KV Cache 服务：

- vLLM 的 Scheduler 和 Worker 进程不直接共享 Python 对象；
- 它们通过 ZeroMQ 连接一个 KVCacheServer；
- Scheduler 注册自己的配置，并通过 RPC 查询远程 KV cache 命中数；
- Worker 注册自己的 rank 和配置，提供实际 lookup 所需的 store；
- Server 负责把 Scheduler、Worker、请求、租约和生命周期组织起来；
- 原有 KVPoolScheduler/KVPoolWorker 的 lookup 逻辑尽量复用，而不是在 MP 层重新实现一套 key 计算。

从模块划分上，这个目标是合理的：传输层、业务协议、服务编排、生命周期和 vLLM connector 被拆成了不同层。

### 1.2 当前实现的实际能力

当前提交真正闭合的是：

    Scheduler connector
      -> KVCacheClient
      -> MPClient / ZeroMQ
      -> MPServer
      -> KVCacheServiceManager
      -> MPKVPoolScheduler
      -> WorkerLookupBridge
      -> LookupKVPoolWorker
      -> backend store.exists()
      -> matched token count

也就是说，它能够完成“远程 KV 是否存在、连续命中了多少 token”的查询。

但 vLLM 的 KV connector 不仅需要返回命中数量，还需要在 Worker 侧把命中的远程 KV 写入本地 paged KV buffer。当前 MP connector 中以下 hook 都是空操作：

- update_state_after_alloc
- start_load_kv
- wait_for_layer_load
- save_kv_layer
- wait_for_save

当前 MP server 路由中也没有 KV tensor copy、block load、block save 或 transfer completion RPC。

因此，当前实现更准确的产品定位是：

    一个能完成 registration、lease 和 lookup 的 MP 服务骨架，
    但还不是完整的 external-KV transfer connector。

如果服务端 lookup 返回正命中，Scheduler 会把这些 token 当成 external KV 纳入调度；然而 Worker 侧没有相应的加载动作。这是整个提交最关键的设计断点。

### 1.3 最重要的结论

如果产品目标是完整的 KV transfer：

- F-001 是 P1 级逻辑缺陷；
- 同步命中可能读取未写入本地 buffer 的 block；
- 异步命中可能把请求放入 WAITING_FOR_REMOTE_KVS 后一直等不到完成通知；
- producer/save 路径不会把 KV 写回远端。

如果产品目标其实是 lookup-only：

- 当前接口不应该返回可供 vLLM 分配的正 external hit；
- 应在初始化或配置校验时明确拒绝 load_async、producer/save 和不支持的 cache layout；
- 不能让调用方以为它是完整 transfer connector。

---

## 2. 背景：vLLM KV connector 到底需要完成什么

### 2.1 Scheduler 侧契约

vLLM Scheduler 在调度请求时大致执行：

1. 计算本地已经命中的 token 数；
2. 调用 connector.get_num_new_matched_tokens(request, num_computed_tokens)；
3. 将 connector 返回的 external token 数纳入总 computed tokens；
4. 分配对应 KV blocks；
5. 调用 connector.update_state_after_alloc；
6. 如果 connector 返回异步标志，则把请求状态设置为 WAITING_FOR_REMOTE_KVS；
7. 等 Worker 报告 finished_recving 后，再把请求重新放回可调度状态。

关键返回值语义是：

    (matched_tokens, load_async)

其中：

- matched_tokens 表示除了本地 computed tokens 之外，可以从 external KV 加载的 token 数；
- load_async=True 表示这些 token 会在之后的 Worker step 中异步加载；
- 当 matched_tokens > 0 时，connector 必须能够实际提供这些 KV；
- 异步加载最终必须通过 finished_recving 让 Scheduler 继续推进。

参考代码：

- vllm/v1/core/sched/scheduler.py 中 connector lookup 和 update_state_after_alloc 逻辑；
- vllm/distributed/kv_transfer/kv_connector/v1/base.py 中 Scheduler connector 抽象接口。

### 2.2 Worker 侧契约

vLLM Worker runner 在一次 forward 周期中会调用：

1. start_load_kv；
2. attention 层内部的 wait_for_layer_load；
3. attention 执行过程中可能调用 save_kv_layer；
4. forward 结束时调用 wait_for_save；
5. 调用 get_finished，收集 finished_sending 和 finished_recving；
6. 把这些结果返回 Scheduler。

这些接口不是为了满足抽象类而存在，而是控制真正的 KV buffer 数据和请求生命周期。

### 2.3 当前提交与契约的差异

| vLLM 需要的能力 | 当前 MP connector |
|---|---|
| 查询 external hit 数量 | 已实现 |
| 在分配后的本地 block 中写入远程 KV | 未实现 |
| 异步 load 的完成通知 | 未实现 |
| 保存本地 KV 到远端 store | 未实现 |
| request_finished 生命周期处理 | 使用基类默认行为 |
| get_finished | 使用基类默认返回值 |
| layerwise transfer | MP lookup 侧强制为 false |

这张表是后面所有分析的基础。

---

## 3. 提交范围和包结构

本次提交新增约 34 个文件、约 8,000 行代码和测试。可以按五层理解。

### 3.1 vLLM 入口和 connector 层

| 文件 | 作用 |
|---|---|
| vllm_ascend/distributed/kv_transfer/__init__.py | 注册 AscendStoreMPConnector 名称和导入路径 |
| ascend_store/ascend_store_mp_connector.py | vLLM connector 适配层 |

### 3.2 MP 业务协议和服务编排层

| 文件 | 作用 |
|---|---|
| ascend_store/mp/kv_cache.py | 对外 facade，重新导出 client/server |
| ascend_store/mp/kv_cache_client.py | 带注册、续租、恢复语义的业务 client |
| ascend_store/mp/kv_cache_server.py | 把业务路由接到 MPServer |
| ascend_store/mp/kv_cache_service.py | 管理 Scheduler/Worker service，并执行 lookup |
| ascend_store/mp/kv_cache_protocol.py | registration、session、lookup payload 编解码 |
| ascend_store/mp/lookup_worker.py | 在服务进程中复用原有 pool lookup 逻辑 |
| ascend_store/mp/registration.py | identity、registration 数据模型和 factory 类型 |
| ascend_store/mp/kv_cache_error.py | 业务层错误类型和错误前缀 |

### 3.3 通用 RPC 层

| 文件 | 作用 |
|---|---|
| mp/rpc/protocol.py | 通用 request/response frame 编解码 |
| mp/rpc/client.py | DEALER client、I/O thread、Future 和超时 |
| mp/rpc/server.py | ROUTER server、route、响应队列和关闭状态机 |
| mp/rpc/executor.py | Inline、bounded pool 和 affinity executor |
| mp/rpc/error.py | 通用 RPC 异常 |

### 3.4 服务生命周期层

| 文件 | 作用 |
|---|---|
| mp/service/lifecycle.py | registration、lease、expiration、recovery、retired session |
| mp/service/error.py | registration conflict、busy、stale session |

### 3.5 同一提交中的另一套旧 MP 实现

以下文件位于 ascend_store 根目录，而不是 mp/ 子包：

| 文件 | 作用 |
|---|---|
| ascend_store/mp_protocol.py | 只有 PING/ECHO 两种 RequestType |
| ascend_store/mp_kv_cache.py | 旧的通用 KV cache ROUTER/DEALER client/server |

当前 AscendStoreMPConnector 导入的是：

    vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.KVCacheClient

因此实际生产主链使用的是 mp/ 下的新实现，而不是根目录的 AscendStoreKVCacheClient/AscendStoreKVCacheServer。旧实现仍有独立测试，说明它可能是演进过程中的前一代设计或兼容保留代码。

两套实现都处理 ZeroMQ、Future、超时和关闭，但：

- 方法名不同；
- 错误类型不同；
- request frame 不同；
- 新实现加入了 registration、identity、session、lease 和 affinity；
- 两套代码同时存在会增加维护和理解成本。

### 3.6 源码导航索引

以下行号以固定提交为准，便于后续逐段阅读：

| 文件 | 重点符号 | 位置 |
|---|---|---|
| ascend_store_mp_connector.py | URL 校验、构造、lookup、transfer hooks | 26-105 |
| mp/registration.py | identity、registration dataclass | 22-137 |
| mp/kv_cache_protocol.py | registration/session/lookup 编解码 | 52-289 |
| mp/kv_cache_client.py | 注册、lease loop、lookup、close | 41-300 |
| mp/kv_cache_server.py | route handler、run、close、abort | 40-193 |
| mp/kv_cache_service.py | factory、binding、lookup、lease | 34-263 |
| mp/lookup_worker.py | bridge、MPKVPoolScheduler、LookupKVPoolWorker | 21-114 |
| mp/rpc/protocol.py | 通用 frame codec | 11-145 |
| mp/rpc/client.py | I/O thread、Future、timeout、disconnect | 64-365 |
| mp/rpc/server.py | route、dispatch、drain、abort、response send | 43-612 |
| mp/rpc/executor.py | Inline、bounded、affinity executor | 16-193 |
| mp/service/lifecycle.py | register、renew、expiration、recovery、close | 38-403 |
| pool_scheduler.py | lookup 和 Scheduler state | 521-750 附近 |
| pool_worker.py | load/save、key 构造、lookup_scheduler | 871-900、1724-1760、2114-2400 附近 |

建议阅读顺序是：先看 connector，再看 kv_cache_client/server，再看 kv_cache_service 和 lookup_worker，最后进入 rpc 和 lifecycle；pool_scheduler/pool_worker 只需重点读上表中的 lookup、load/save 和 key 构造区域。


---

## 4. 总体架构

### 4.1 组件拓扑

当前真正的主路径：

~~~text
vLLM Scheduler process
    |
    | AscendStoreMPConnector.get_num_new_matched_tokens()
    v
KVCacheClient
    |
    | MPClient.submit/request()
    | DEALER socket, I/O thread, Future
    v
ZeroMQ endpoint
    |
    v
MPServer
    |
    | ROUTER I/O thread
    | decode request -> Route -> Executor
    v
KVCacheServer
    |
    v
KVCacheServiceManager
    |                         \
    | SchedulerLifecycle       WorkerLifecycle
    v                           v
MPKVPoolScheduler          LookupKVPoolWorker
    |                           |
    | _WorkerLookupBridge       | m_store.exists()
    +-------------> worker ----> backend store
~~~

这张图里有一个重要语义：

- MPKVPoolScheduler 不是在本地直接访问 store；
- 它通过 _WorkerLookupBridge 把 lookup 委托给注册的 Worker；
- KVCacheServiceManager 再使用 Worker executor 调用 rank-0 worker；
- Worker 的 m_store 通常被绑定为 Scheduler 创建的 store_scheduler。

因此“Scheduler 注册”和“Worker 注册”共同决定 lookup 是否可用。

### 4.2 Server 内部的三类执行单元

KVCacheServer 构造时创建：

| 执行单元 | 默认类型 | 作用 |
|---|---|---|
| scheduler_executor | AffinityExecutor | registration、unregistration、lookup 等 Scheduler 相关操作 |
| worker_executor | AffinityExecutor | Worker registration、unregistration 和实际 worker lookup |
| lease_executor | InlineExecutor | renew 只更新生命周期元数据，不应被业务 lookup 阻塞 |

KVCacheServer 将 route 映射为：

| 方法 | executor | affinity key |
|---|---|---|
| REGISTER_SCHEDULER | scheduler executor | SchedulerIdentity |
| UNREGISTER_SCHEDULER | scheduler executor | SchedulerIdentity |
| LOOKUP | scheduler executor | SchedulerIdentity |
| RENEW_SCHEDULER | inline executor | 无 |
| REGISTER_WORKER | worker executor | WorkerIdentity |
| UNREGISTER_WORKER | worker executor | WorkerIdentity |
| RENEW_WORKER | inline executor | 无 |

设计意图：

- 同一个 Scheduler identity 的状态操作按序执行；
- 同一个 Worker identity 的状态操作按序执行；
- 不同 DP rank 的 lookup 可以并行；
- renew 不应该排在耗时 lookup 后面。

### 4.3 主要对象的所有权

| 对象 | 创建者 | 主要使用者 | 关闭者 | 线程约束 |
|---|---|---|---|---|
| ZeroMQ ROUTER socket | MPServer 构造函数 | MPServer.run 线程 | MPServer 关闭流程 | 只允许 run 线程做 I/O |
| ZeroMQ DEALER socket | MPClient I/O thread | MPClient I/O thread | I/O thread finally | public thread 不直接操作 |
| MPClient outbound queue | MPClient | 多个 public caller 写入，I/O thread 读取 | close 时失败所有未完成请求 | 多生产者、单消费者 |
| MPClient pending map | MPClient | I/O thread | I/O thread | 只由 I/O thread 访问 |
| MPServer output queue | handlers/callback | run 线程搬到 backlog | server cleanup | callback 不直接操作 socket |
| scheduler executor | KVCacheServer | RPC server 和 service manager | MPServer cleanup | affinity worker 执行 |
| worker executor | KVCacheServer | RPC server 和 service manager | MPServer cleanup | affinity worker 执行 |
| Scheduler service | lifecycle manager | lookup、binding、close | owner lane close | 不应跨任意线程访问 |
| Worker service | lifecycle manager | worker lookup、binding、close | owner lane close | 绑定和 lookup 使用 worker lane |
| lease maintenance thread | lifecycle manager | 过期检查 | stop_maintenance | 不负责直接业务请求 |
| client lease thread | KVCacheClient | renew 和重新注册 | KVCacheClient.close | 不能 join 自己 |
| backend store | 原有 KVPoolScheduler/Worker | lookup 或实际 transfer | service close | 当前 MP lookup 侧只使用 exists |

这个表揭示了一个设计重点：RPC socket 的线程所有权做得比较清楚，但 service 和 backend 的关闭成功与否没有被完整地反映到生命周期状态中。

---

## 5. vLLM connector 入口解读

### 5.1 connector 注册

vllm_ascend/distributed/kv_transfer/__init__.py 的 register_connector() 中加入：

    "AscendStoreMPConnector"
      -> ascend_store.ascend_store_mp_connector
      -> AscendStoreMPConnector

这让 vLLM 可以通过 connector 名称动态导入该类，注册位置也和原有 AscendStoreConnector 保持一致。

### 5.2 server URL 校验

ascend_store_mp_connector.py 的 _get_kv_cache_server_url()：

1. 读取 vllm_config.kv_transfer_config；
2. 没有 kv_transfer_config 时抛 ValueError；
3. 从 kv_connector_extra_config 读取 kv_cache_server_url；
4. 要求它是非空字符串；
5. 返回 endpoint。

优点是失败较早，错误信息直接指向配置键。

当前没有做的校验包括：

- URL scheme 是否为允许的 tcp/ipc；
- endpoint 是否只绑定本机；
- 是否需要认证；
- 是否与当前 engine_id、租户或进程身份匹配。

### 5.3 connector 构造

构造流程：

1. 调用 KVConnectorBase_V1 初始化；
2. 创建 KVCacheClient；
3. 如果 role 是 SCHEDULER：
   - 要求 kv_cache_config 存在；
   - 读取第一个 KV cache group 的 page_size_bytes；
   - 调用 register_scheduler；
4. 否则调用 register_worker；
5. 任意注册异常都会关闭 client 后重新抛出。

如果 client 已经创建，而 registration 失败，不会直接丢掉 I/O thread 和 socket，这是一个明确优点。

但 role 判断采用“不是 SCHEDULER 就走 Worker”。如果未来 vLLM 增加第三种 role，当前实现可能误判为 Worker。更严格的做法是显式列出支持的 role，对未知 role 抛错。

### 5.4 connector 方法的实际行为

| 方法 | 当前行为 | 影响 |
|---|---|---|
| get_num_new_matched_tokens | 委托 client.lookup | 会让远程命中参与 Scheduler 调度 |
| update_state_after_alloc | return None | 不记录本次分配的远程 block |
| build_connector_meta | 返回空 metadata | Worker 没有 load/save 所需元数据 |
| start_load_kv | no-op | 不启动 KV load |
| wait_for_layer_load | no-op | 不等待任何 layer |
| save_kv_layer | no-op | 不保存 KV |
| wait_for_save | no-op | 不等待保存完成 |
| shutdown | close client | 只关闭 RPC client |

这里最容易造成误解：方法都存在，抽象接口也能实例化，但“接口存在”不等于“transfer 语义已经实现”。

---

## 6. Identity、Registration 和 Fingerprint

### 6.1 SchedulerIdentity

Scheduler identity 由两个字段组成：

| 字段 | 来源 |
|---|---|
| engine_id | vllm_config.kv_transfer_config.engine_id |
| data_parallel_rank | vllm_config.parallel_config.data_parallel_rank |

同一 engine 的不同 DP Scheduler 被视为不同 service。

### 6.2 WorkerIdentity

Worker identity 由三个字段组成：

| 字段 | 来源 |
|---|---|
| engine_id | vllm_config.kv_transfer_config.engine_id |
| rank | vllm_config.parallel_config.rank |
| data_parallel_rank | vllm_config.parallel_config.data_parallel_rank |

Worker rank 是全局 rank，而 lookup 协调约定固定使用 rank 0。

### 6.3 Registration 对象

SchedulerRegistration 包含：

- identity；
- 完整 VllmConfig；
- KVCacheConfig；
- page_size_bytes；
- session_id。

WorkerRegistration 包含：

- identity；
- 完整 VllmConfig；
- KVCacheConfig；
- session_id。

它们使用 frozen dataclass，能避免注册完成后被调用方随意修改，是合理的建模方式。

### 6.4 registration fingerprint

客户端将 registration 编码成 payload，服务端用：

    hashlib.sha256(serialized_registration).digest()

得到 fingerprint。

生命周期管理器用 identity + session_id + fingerprint 判断：

- 同 identity、同 session、同 fingerprint：幂等成功；
- 同 identity、同 session、不同 fingerprint：RegistrationConflictError；
- 同 identity、不同 session：进入替换流程。

把“会话身份”和“配置版本”分开是好的建模：

- session_id 识别某次 client 生命周期；
- fingerprint 识别该生命周期使用的配置；
- 同 session 重试不需要重新创建 service。

但 fingerprint 只是对收到的序列化 bytes 做 hash，不是所有权凭证。知道 identity 的任意 client 可以构造新 session，成为 F-003 的根源。

---

## 7. 业务协议详解

业务协议位于 mp/kv_cache_protocol.py，通用 frame 外壳位于 mp/rpc/protocol.py。两层分工是正确的：

- rpc/protocol.py 不理解 Scheduler、Worker、block hash；
- kv_cache_protocol.py 不负责 ZeroMQ socket、Future 和 executor。

### 7.1 通用 RPC request frame

经过 ROUTER/DEALER 传输后，服务端收到：

~~~text
[client_identity, request_id, method, deadline, payload_0, payload_1, ...]
~~~

decode_request() 解析成：

    (request_id, method, deadline_ns, payloads)

约束：

- 至少有 request_id、method、deadline 三个业务 frame；
- request_id 必须是 bytes；
- method 必须是 UTF-8 且非空；
- deadline 为空 frame 表示无 deadline；
- deadline 非空时，必须能解析为正整数；
- 每个业务 payload 必须是 bytes。

### 7.2 通用 RPC response frame

服务端发送：

~~~text
[client_identity, request_id, method, status, response_0, response_1, ...]
~~~

客户端从 DEALER 收到：

~~~text
[request_id, method, status, response_0, response_1, ...]
~~~

status 有四种：

| 状态 | client 异常或处理 |
|---|---|
| OK | Future set_result |
| ERROR | MPRemoteError |
| BUSY | MPServerBusyError |
| ABORTED | MPServerAbortedError |

客户端同时验证 request_id 和 method，避免迟到响应或错误响应交给错误的 Future。

### 7.3 registration request frame

Scheduler registration：

~~~text
[engine_id, data_parallel_rank, serialized_scheduler_registration]
~~~

Worker registration：

~~~text
[engine_id, rank, data_parallel_rank, serialized_worker_registration]
~~~

服务端先从普通 bytes frame 解出 identity，再对最后一个 payload 做 cloudpickle.loads，最后检查：

    registration.identity == header_identity

类型和 identity 一致性校验是好事，但反序列化发生在类型检查之前，因此不能解决不可信 payload 的执行风险。

### 7.4 session request frame

Scheduler session：

~~~text
[engine_id, data_parallel_rank, session_id]
~~~

Worker session：

~~~text
[engine_id, rank, data_parallel_rank, session_id]
~~~

这些 frame 用于 renew 和 unregister。

### 7.5 lookup request frame

完整 lookup frame：

~~~text
[
  engine_id,
  data_parallel_rank,
  request_id,
  prompt_token_count,
  num_tokens,
  num_computed_tokens,
  block_hash_0,
  block_hash_1,
  ...,
  session_id
]
~~~

服务端解析成：

    SchedulerIdentity
    session_id
    LookupRequestView
    num_computed_tokens

LookupRequestView 保存：

- vLLM request_id；
- prompt_token_ids 的长度；
- block_hashes；
- request.num_tokens。

这里没有传输真实 prompt token ids，只用 range(prompt_token_count) 保留长度。当前 lookup 逻辑只需要 block hash 和 token length，因此这能减少 payload；但也意味着 protocol 强依赖“长度足以表达 lookup 语义”这一假设。

### 7.6 lookup response frame

~~~text
[matched_tokens, async_flag]
~~~

其中：

- matched_tokens 是 8 字节大端非负整数；
- async_flag 为 b"\x00" 或 b"\x01"。

如果 matched_tokens == 0，当前 pool scheduler 会返回同步 false；如果存在需要加载的 token，则根据 load_async 和 use_layerwise 决定 async flag。

### 7.7 输入校验的优点和缺口

已有校验：

- identity 字符串非空；
- rank 非负整数；
- session_id 非空；
- 整数固定 8 字节；
- UTF-8 合法；
- block hash 非空；
- response status 必须属于枚举；
- payload 数量符合 method 预期。

缺少的边界：

- 单 frame 最大字节数；
- multipart 总字节数；
- block hash 数量上限；
- registration serialized payload 上限；
- client outbound queue 上限；
- 每个 identity 的并发配额；
- 协议版本号；
- 对重复 block hash 的明确处理。

---

## 8. MPClient 详细解读

文件：mp/rpc/client.py。

### 8.1 线程模型

MPClient 的设计注释明确规定：

- public request 方法可以被多个应用线程调用；
- outbound queue 是多生产者、单消费者；
- DEALER socket 和 pending map 只由 I/O thread 拥有；
- Future callback 在 I/O thread 同步执行，不能在 callback 中调用会阻塞 client 的方法。

ZeroMQ socket 不能被多个线程随意操作，当前实现把 socket 操作集中到 I/O thread，是正确方向。

### 8.2 submit_request

submit_request() 的流程：

1. 标准化 method；
2. 检查 client 是否 close；
3. 检查 transport_connected；
4. 生成递增 request_id；
5. 创建 Future；
6. 计算 deadline_ns；
7. encode_request；
8. 将 _OutboundRequest 放入 queue；
9. 向 notify socket 写一个字节，唤醒 I/O thread。

public caller 不直接 send socket，因此不会把 ZeroMQ 的线程限制泄露给上层调用方。

### 8.3 I/O loop

I/O thread 管理：

- DEALER socket；
- monitor socket；
- notify socket；
- poller；
- pending request deadline。

每一轮大致执行：

1. 计算最近 pending request 的超时时间；
2. poll DEALER、monitor、notify；
3. notify 可读时处理 outbound queue；
4. DEALER 可读时处理 response；
5. monitor 可读时更新 transport 状态；
6. 过期 pending request。

### 8.4 outbound 状态

_process_outbound()：

1. 从 queue 取出 request；
2. Future 设置为 running；
3. 如果 deadline 已到，Future 设置 timeout；
4. 使用 NOBLOCK send_multipart；
5. 如果 zmq.Again，Future 设置 MPServerBusyError；
6. send 成功后把 request 放入 pending map。

正常路径：

    queue -> send -> pending map -> response -> pop pending -> complete Future

需要关注的竞态：

- Future 在 send 成功前已经设置为 running；
- 如果 send_multipart 抛出非 zmq.Again 异常；
- 外层 I/O loop 会捕获异常并调用 _fail_pending；
- 当前 request 尚未进入 pending map；
- 如果它也已经不在 outbound queue 中，_fail_pending 可能找不到它。

这条路径目前没有真实 transport fault 证据，属于需要专门注入验证的观察。

### 8.5 inbound response

_process_inbound()：

1. 读取完整 multipart；
2. 取第一帧 request_id；
3. 从 pending map 删除对应请求；
4. 没有 pending 时丢弃迟到响应；
5. decode_response；
6. 检查 response method 与 pending method 一致；
7. 按 status 转成不同异常或成功结果；
8. 完成 Future。

这套逻辑可以处理乱序响应，因为匹配依据是 request_id 而不是发送顺序。

### 8.6 timeout、断连和 close

客户端支持：

- 单请求 timeout；
- transport monitor 断连；
- 断连时失败所有 pending Future；
- close 时失败 outbound 和 pending Future；
- I/O thread join；
- monitor、DEALER、notify socket、context 清理。

close() 是幂等的，且不会让 public caller 直接关闭 ZeroMQ socket。

主要缺口：

- outbound queue 没有 maxsize；
- payload 没有大小预算；
- transport send 的非 Again 异常可能触发未入 pending 的 Future 遗留；
- reconnect 只恢复 transport，不自动恢复业务 registration，业务层 lease loop 负责补注册。

---

## 9. MPServer 详细解读

文件：mp/rpc/server.py。

### 9.1 ROUTER 所有权

MPServer 的模块注释规定 ROUTER socket 只能由 run() 所在线程操作。

handler 不直接操作 socket，而是：

1. handler 在 executor 中运行；
2. 完成后把 response 放入 output_queue；
3. 通过 notify socket 唤醒 run 线程；
4. run 线程把 response 转到 response_backlog；
5. 只有 run 线程执行 send_multipart。

这是清晰的“业务线程产出、I/O 线程发送”设计。

### 9.2 route 和 dispatch

Route 包含：

- method；
- handler；
- executor；
- 可选 key_factory。

收到请求后：

1. 检查 frame 数量；
2. decode_request；
3. 创建 _AcceptedRequest；
4. 以 (client_identity, request_id) 作为 request key；
5. 检查 server 是否仍接受新请求；
6. 查找 Route；
7. 计算 affinity key；
8. 提交 executor；
9. 给 Future 添加完成回调。

_accepted_requests 是 server 端确保“一次请求只对应一个终态”的核心数据结构。

### 9.3 response backlog 和 backpressure

run() 每轮根据 response_backlog 选择：

- backlog 为空：监听 POLLIN，继续收请求；
- backlog 非空：监听 POLLOUT，优先发响应，暂时不收新请求。

这个设计避免响应长期积压时继续无限接收请求，是一个比较好的 backpressure 方向。

但它只限制了“响应 backlog 进入发送阶段”的增长，不等于完整资源上限：

- client outbound queue 仍无上限；
- server 仍可接收很大的 multipart；
- executor capacity 有限，但 payload 已经可能在接收时占用内存；
- block hash list 和 registration bytes 没有业务级限制。

### 9.4 server state machine

状态定义：

~~~text
READY -> RUNNING -> DRAINING -> DRAINED -> CLOSED
                       \
                        -> ABORTING -> ABORTED
~~~

| 状态 | 语义 |
|---|---|
| READY | 已 bind，但 run 尚未开始 |
| RUNNING | 正常接收和执行请求 |
| DRAINING | 不再接受新请求，等待已接受请求结束 |
| DRAINED | 请求处理和发送完成，可释放资源 |
| ABORTING | 强制终止，排队任务取消，运行任务不等待 |
| ABORTED | 强制关闭完成 |
| CLOSED | 优雅关闭完成 |

request_stop() 只有在所有已接受请求都有有限 deadline 时才允许进入 DRAINING。无 deadline 请求会让 graceful close 返回 false，调用方应选择 abort。

### 9.5 abort 与正常响应竞争

abort() 会为每个 accepted request 放入 ABORTED response，并让后续普通 response 在发送阶段被丢弃。

因此：

- 已排队任务可以被 cancel；
- 正在运行的 handler 不会被强行杀死；
- client 会收到 ABORTED；
- server 不等待运行中的业务代码结束。

KVCacheServer 层还明确让 abort 跳过 KVCacheServiceManager.close()，把 abort 解释为进程级强制退出语义。这个选择有其合理性，但 backend、线程和 socket 是否最终由进程退出回收，需要真实服务器模型验证。

---

## 10. Executor 和并发模型

文件：mp/rpc/executor.py。

### 10.1 InlineExecutor

InlineExecutor 在 submit() 的调用线程中立即执行：

- lease renew 适合使用它；
- 不会产生线程切换；
- Future 接口仍保持统一；
- 异常被捕获并写入 Future。

### 10.2 BoundedThreadPoolExecutor

它使用：

    BoundedSemaphore(max_workers + max_pending_tasks)

限制运行中和等待中的任务总数。

容量不足时：

- block=False：抛 MPServerBusyError；
- block=True：等待 capacity。

任务完成时释放 semaphore。该类适合不需要 affinity 的独立任务，但当前 KVCacheServer 的 Scheduler/Worker 主路由使用的是 AffinityExecutor。

### 10.3 AffinityExecutor

AffinityExecutor：

- 创建 max_workers 个固定线程；
- 每个线程有一个 queue；
- 通过 hash(key) % worker_count 选择线程；
- 相同 key 永远进入同一 queue；
- 同一 key 因为同一 queue 的 FIFO 而串行；
- 不同 key 如果落在不同 queue，可以并行；
- 总容量由 semaphore 限制。

当前业务使用：

- SchedulerIdentity 作为 Scheduler/lookup key；
- WorkerIdentity 作为 Worker key。

这能保护非线程安全的 scheduler/worker service，尤其是 store binding、lookup、unregister 等状态操作。

### 10.4 affinity 的局限

固定 hash 映射有两个性质：

1. 不同 identity 可能 hash 到同一线程，因此不同 identity 不保证一定并行；
2. 某个热点 identity 的长任务会阻塞分配到同一线程的其他 identity。

这不是实现错误，而是 affinity 串行化的明确 trade-off。服务器压力测试应观察热点 identity 是否造成不必要的排队。

---

## 11. KVCacheServer 和业务路由

文件：mp/kv_cache_server.py。

### 11.1 构造阶段

KVCacheServer：

1. 创建 scheduler_executor；
2. 创建 worker_executor；
3. 创建 lease_executor；
4. 创建 KVCacheServiceManager；
5. 为业务方法建立 Route；
6. 创建 MPServer 并绑定 endpoint。

默认每个 affinity executor：

    max_pending_requests = 64

注意，这个 64 是 executor 级容量，不是：

- 每个 client 的 queue 上限；
- 每个 identity 的配额；
- 每个 multipart 的 bytes 上限；
- 全服务的内存预算。

### 11.2 registration handlers

_handle_register_scheduler()：

1. decode identity 和 registration；
2. 调用 service.register_scheduler；
3. ServiceBusyError 映射为 MPServerBusyError；
4. 返回 ACK_RESPONSE。

Worker registration 同理。

当前 handler 对 registration payload 的反序列化发生在 decode_registration_request 内，且在认证前执行。

### 11.3 lookup handler

_handle_lookup()：

1. decode_lookup_request；
2. 调用 service.lookup(identity, session, request, num_computed_tokens)；
3. encode_lookup_response(matched_tokens, is_async)。

lookup 的业务返回不是 KV 内容，而是两个元信息：

- 命中的 token 数；
- 是否需要异步加载。

因此 KV 数据传输必须存在于别的路径；当前提交中没有这条路径。

### 11.4 run、close 和 abort

run()：

1. 启动 scheduler/worker lease maintenance；
2. 运行 MPServer；
3. 异常时 abort；
4. 正常退出时 close。

close()：

1. 如果已经 closed，直接成功；
2. 如果 abort 已请求，返回 false；
3. 请求 RPC server 进入 drain；
4. 停止 lease maintenance；
5. 等待 RPC drain；
6. 关闭 service manager；
7. 关闭 RPC server。

顺序意图是：

    先停止接收新请求
    -> 等待正在处理的请求完成
    -> 再关闭 scheduler/worker service
    -> 最后关闭 RPC transport

这是合理的资源顺序；但如果 service close 抛异常，生命周期 manager 当前只记录日志，未必能准确表达资源已经完全释放。

---

## 12. KVCacheServiceManager 详细解读

文件：mp/kv_cache_service.py。

### 12.1 默认 factory

默认 Scheduler factory 创建：

    MPKVPoolScheduler(registration, self._lookup_worker)

默认 Worker factory 创建：

    LookupKVPoolWorker(
        registration.vllm_config,
        kv_cache_config=registration.kv_cache_config,
        rank=registration.identity.rank,
    )

### 12.2 Scheduler 注册

register_scheduler()：

1. 校验 registration.identity 与 VllmConfig 推导出的 identity 一致；
2. 对 serialized registration 做 SHA-256；
3. 调用 SchedulerLifecycleManager.register；
4. 注册成功后调用 _schedule_lookup_store_binding；
5. 返回 scheduler service。

重要顺序：

    publish scheduler
    -> bind scheduler store to coordinator worker

如果 binding 在 publish 之后失败，服务端可能已经持有 scheduler，但 client 得到失败。这就是 F-004 的非事务状态。

### 12.3 Worker 注册

register_worker()：

1. 校验 Worker identity；
2. 计算 fingerprint；
3. 调用 WorkerLifecycleManager.register；
4. 调用 _bind_lookup_store；
5. 返回 worker service。

Worker 可以先注册，也可以后注册。绑定逻辑在 Scheduler 或 Worker 任一方到达时都可能执行，因此测试覆盖了两种注册顺序。

### 12.4 store binding

_bind_lookup_store()：

1. 根据 SchedulerIdentity 查 scheduler；
2. 根据 WorkerIdentity 查 worker；
3. 从 scheduler 取得 store_scheduler；
4. 从 worker 取得 bind_lookup_store；
5. 将 scheduler 的 store 绑定到 worker。

默认 LookupKVPoolWorker 初始使用 _MissingLookupStore：

    exists(keys) -> [0] * len(keys)

这让 Worker 在 store 尚未绑定时安全地返回 miss，而不是直接崩溃。

但这也可能掩盖绑定错误：绑定失败时 lookup 表现为 cache miss，调用方不一定知道服务没有准备好。

### 12.5 lookup worker 路由

_lookup_worker()：

1. 把 SchedulerIdentity 映射到：

       WorkerIdentity(engine_id, rank=0, data_parallel_rank=scheduler_dp)

2. 构造 callback；
3. 如果存在 worker executor，以 worker identity 作为 affinity key 提交；
4. 等待 Future.result()；
5. 执行 worker.lookup_scheduler。

rank 0 是固定的 lookup coordinator。其他 rank worker 可以注册，但 lookup 不会 fallback 到非 coordinator worker。

### 12.6 lookup 异常处理

LookupKVPoolWorker.lookup_scheduler 内部会调用 m_store.exists。原有 pool worker 逻辑在远端 store 异常时记录日志并返回 0。

因此当前错误策略是：

    store failure -> hit=0 -> Scheduler 走 cache miss/recompute

这是一种可用性优先的降级策略，避免远端缓存失败直接中断推理；但配置错误和服务未准备好也可能被静默表现为 miss。

---

## 13. lookup_worker：如何复用原有 KV lookup

文件：mp/lookup_worker.py。

### 13.1 _WorkerLookupBridge

它把 Scheduler service 需要的 client.lookup 接口转换为 WorkerLookupHandler：

~~~text
KVPoolScheduler.get_num_new_matched_tokens()
    -> self.client.lookup(...)
    -> _WorkerLookupBridge.lookup()
    -> KVCacheServiceManager._lookup_worker()
    -> registered rank-0 worker.lookup_scheduler()
~~~

Bridge 总是传：

    use_layerwise=False

这是 MP lookup 侧强制非 layerwise 的直接来源之一。

### 13.2 MPKVPoolScheduler

MPKVPoolScheduler 继承 KVPoolScheduler，构造父类时传入：

    use_layerwise=False

其余主要配置来自 SchedulerRegistration：

- vllm_config；
- kv_cache_config；
- page_size_bytes。

然后把 parent 的 client 替换成 _WorkerLookupBridge。

优点：

- 复用原有 token length、block hash、group、TP rank、连续命中判断；
- MP 层只负责把 exists 查询转成跨进程调用；
- 避免复制一份复杂的 lookup 算法。

缺点：

- hard-code false 静默关闭 layerwise；
- parent constructor 仍可能创建或初始化一部分 backend/client 状态；
- MP connector 的真正 load/save hook 没有同步复用 parent transfer 逻辑。

### 13.3 LookupKVPoolWorker

LookupKVPoolWorker 继承 KVPoolWorker，但做了三件特殊处理：

1. 保存 registered rank；
2. 不初始化真实 distributed backend；
3. 重写 _init_parallelism_info，使用注册 rank 计算 tp_rank、pp_rank 等。

_init_backend() 是 pass，说明它不会启动正常 Worker 侧 KV transfer backend。

它将 m_store 设置成：

- 构造时传入的外部 store；或
- _MissingLookupStore；
- 后续可通过 bind_lookup_store 绑定 Scheduler 的 store。

这清晰地表达了“当前 Worker 只做 lookup”的意图，但也证明当前 MP worker 不是完整 KV transfer worker。

---

## 14. 原有 KVPoolScheduler lookup 逻辑

MPKVPoolScheduler 复用 pool_scheduler.py 的 get_num_new_matched_tokens()。

### 14.1 输入

- Request；
- num_computed_tokens；
- request.prompt_token_ids 长度；
- request.block_hashes；
- KV cache group；
- 当前配置的 cache transfer granularity；
- hbm_hit_tokens。

### 14.2 主要判断

原有逻辑会先处理：

- kv_role 是否允许加载；
- retention interval；
- prompt 长度；
- discard partial chunks；
- cache transfer granularity；
- layerwise/GVA/hybrid 配置；
- 已计算 token 数是否已经覆盖可查范围。

非 layerwise 路径最终通过 LookupKeyClient 或 worker lookup 查询 store key 是否存在。

### 14.3 命中数到 external tokens

pool scheduler 得到 num_external_hit_tokens 后：

1. 可能根据 Eagle 或 layerwise 规则调整尾部；
2. 如果完整命中整个 request，会减去一个 token，以保留最后一个 token 的计算；
3. 减去 num_computed_tokens 得到需要分配的 external tokens；
4. 创建 LoadSpec；
5. 返回：

       (need_to_allocate, self.load_async and not self.use_layerwise)

这个返回值本身符合 vLLM Scheduler 的接口形式。

### 14.4 update_state_after_alloc 的原实现

原有 KVPoolScheduler.update_state_after_alloc 会：

- 记录 request 对应的 local block ids；
- 检查 load_specs；
- 校验 external token 数和 LoadSpec 的关系；
- 标记 can_load；
- 异步路径将 request id 放入 _loading_req_ids；
- 为后续 Worker load 准备状态。

但 AscendStoreMPConnector 没有调用这个 parent 方法，而是直接 no-op。因此 MP lookup 的命中状态在 Scheduler connector 层被截断。

---

## 15. 端到端流程一：启动和注册

### 15.1 推荐启动顺序

server 先 bind endpoint，然后 Scheduler/Worker client 连接并注册。客户端创建时并不要求 server 已经启动，真正 registration 会在 transport connected 后尝试。

### 15.2 Scheduler 注册时序

~~~text
Scheduler process
  |
  | AscendStoreMPConnector(...)
  | KVCacheClient.register_scheduler(...)
  | SchedulerRegistration.create(...)
  | encode_registration_request(...)
  v
MPClient.request(REGISTER_SCHEDULER)
  |
  v
MPServer._receive_request()
  |
  | decode_request()
  | Route -> scheduler executor
  v
KVCacheServer._handle_register_scheduler()
  |
  | decode_registration_request()
  | cloudpickle.loads()
  v
KVCacheServiceManager.register_scheduler()
  |
  | LifecycleManager.register()
  | factory -> MPKVPoolScheduler
  | publish scheduler
  | bind lookup store
  v
ACK_RESPONSE
  |
  v
KVCacheClient._registered = True
  |
  v
lease thread starts
~~~

### 15.3 Worker 注册时序

Worker 走相同的 RPC 外壳，但：

- identity 多一个 rank；
- 进入 worker executor；
- factory 创建 LookupKVPoolWorker；
- 绑定 Scheduler store；
- 后续 lookup 只选择 DP group 的 rank 0 worker。

### 15.4 注册顺序的两个分支

| 顺序 | 行为 |
|---|---|
| Worker -> Scheduler | Worker 先存在但没有 Scheduler store；Scheduler 注册后触发 binding |
| Scheduler -> Worker | Scheduler 注册后 binding 可能暂时找不到 Worker；Worker 注册后再次 binding |

测试覆盖了两种顺序，这是设计上的优点。

---

## 16. 端到端流程二：同步 lookup

当 load_async=False 且 store 命中时，理论流程：

~~~text
vLLM Scheduler
  |
  | get_num_new_matched_tokens(request, local_tokens)
  v
AscendStoreMPConnector
  |
  | KVCacheClient.lookup()
  | encode lookup frames
  v
MPServer -> scheduler affinity executor
  |
  | decode request
  | validate Scheduler session
  v
MPKVPoolScheduler.get_num_new_matched_tokens()
  |
  | _WorkerLookupBridge.lookup()
  v
worker executor -> rank-0 LookupKVPoolWorker.lookup_scheduler()
  |
  | build keys
  | store.exists(keys)
  v
matched_tokens
  |
  v
Scheduler allocates external KV blocks
  |
  v
AscendStoreMPConnector.update_state_after_alloc()  [currently no-op]
  |
  v
Worker forward                         [currently no load hook]
~~~

最后两步是断裂点：Scheduler 已经因为 matched_tokens 分配了外部 block，但 Worker 没有把远程数据写入这些 block。

因此“lookup response 正确”不能证明“模型实际使用了正确 KV”。

---

## 17. 端到端流程三：异步 lookup

当 load_async=True 且非 layerwise 时，pool scheduler 返回：

    (need_to_allocate, True)

vLLM Scheduler 随后：

1. 分配用于接收 KV 的 block；
2. 设置 request.status = WAITING_FOR_REMOTE_KVS；
3. 放入 skipped waiting；
4. 记录 num_computed_tokens；
5. 等 Worker 的 KVConnectorOutput.finished_recving。

理论时序：

~~~text
Scheduler lookup hit
    -> allocate destination blocks
    -> WAITING_FOR_REMOTE_KVS
    -> Worker start_load_kv()
    -> asynchronous copy into local KV
    -> get_finished()
    -> finished_recving contains request_id
    -> Scheduler promotes request to WAITING/RUNNING
~~~

当前实际时序：

~~~text
Scheduler lookup hit
    -> allocate destination blocks
    -> WAITING_FOR_REMOTE_KVS
    -> AscendStoreMPConnector.start_load_kv() returns
    -> get_finished() uses base default (None, None)
    -> no finished_recving
    -> request may remain waiting indefinitely
~~~

这是 F-001 的异步部分，静态逻辑已经闭合；最终是否表现为永久等待、超时或其他 fallback，需要服务器运行确认。

---

## 18. 端到端流程四：save / producer

完整 transfer connector 的 producer 流程应当是：

~~~text
Worker forward
    -> save_kv_layer(layer, kv_tensor, metadata)
    -> remote put/copy
    -> wait_for_save()
    -> request_finished()
    -> remote store can be queried by another consumer
~~~

当前 MP connector 的实际行为：

- save_kv_layer() 不做任何事；
- wait_for_save() 不做任何事；
- request_finished() 沿用基类默认 False；
- MP service 协议没有 PUT/SAVE/TRANSFER_COMPLETE；
- MP worker 的 _init_backend() 是 pass。

所以当前提交不能证明 producer 产生的 KV 会被写入远端 store。

如果设计上只允许 consumer lookup：

- 配置应显式限制 kv_role；
- 不应该接受 kv_producer 或 kv_both；
- 不应该让用户以为 lookup hit 后会形成完整闭环。

---

## 19. 生命周期状态机

这是第一类需要重点整理的额外内容：把散落在多个方法里的状态变化集中起来。

### 19.1 ServiceLifecycleManager 状态

对每个 identity，内部可能同时存在：

| 结构 | 说明 |
|---|---|
| _services | 当前 active service |
| _registering | 正在创建 service 的 registration flight |
| _expiring | 已从 active 删除、正在 owner close 的 service |
| _recoverable_sessions | 过期后允许同 session、同 fingerprint 恢复的记录 |
| _retired_sessions | 明确禁止再次使用的 session 集合 |

概念状态图：

~~~text
                         factory success
                   +------------------------+
                   |                        v
ABSENT -> REGISTERING -> ACTIVE --------> EXPIRING
              |           |                 |
              |           | unregister      | close succeeds
              |           v                 v
              |        RETIRED         RECOVERABLE
              |                             |
              | factory failure              | same session + same fp
              v                             v
           ABSENT <---------------------- ACTIVE

ACTIVE -- new session --> old session RETIRED
ACTIVE -- old session request --> STALE error
RECOVERABLE -- different session --> old session RETIRED
~~~

### 19.2 register() 的关键顺序

register() 在锁内执行：

1. 检查 manager 未关闭；
2. 检查 session 是否已经 retired；
3. 如果 identity 正在 expiring，返回 busy；
4. 如果 active entry 存在：
   - 同 session：校验 fingerprint，幂等返回；
   - 不同 session：旧 session 加入 retired，删除 active entry，保存 old_service；
5. 如果已有同 identity registration flight：
   - 同 session 且 fingerprint 相同：等待同一个 Future；
   - 不同 session：RegistrationConflictError；
6. 否则创建新的 _RegistrationFlight。

锁外执行：

1. 关闭 old_service；
2. 执行 factory；
3. 重新拿锁 publish；
4. 设置 flight.future 结果。

把耗时 factory 和 close 放在锁外，避免阻塞其他 identity，是好的并发设计。

但“先删除旧 active，再关闭旧 service，再创建新 service”的顺序意味着 replacement 不是原子切换。factory 失败时，旧 service 已经被移除。

### 19.3 renew()

renew()：

1. 校验 session 非空；
2. 检查 manager 未关闭；
3. 拒绝 retired session；
4. 找 active service；
5. session 不匹配则 stale；
6. 更新 last_seen。

renew 只更新生命周期元数据，不执行业务 lookup，因此可以放在 InlineExecutor，不被 Scheduler/Worker affinity lane 阻塞。

### 19.4 get_for_session() 和 find()

两者的区别：

| 方法 | 是否校验 session | 是否 renew lease | 用途 |
|---|---|---|---|
| find | 否 | 否 | manager 内部做 binding 或按 identity 查对象 |
| get_for_session | 是 | 是 | 外部 owner 请求访问自己的 service |

lookup 使用 get_for_session，因此一个有效 Scheduler lookup 会更新 Scheduler lease。

Worker lookup 使用 find，因为 Worker 是由经过 Scheduler session 校验的 server-side callback 选择出来的；这依赖内部调用链正确，不能把 find 暴露成任意外部操作。

### 19.5 expiration

expire_leases()：

1. 计算 stale_before = now - lease_timeout；
2. 在锁内找出 last_seen 过期的 service；
3. 从 _services 移到 _expiring；
4. 锁外通过 owner_close_handler 关闭；
5. 执行 _finish_expiration；
6. 如果 session 尚未 retired，写入 recoverable。

默认参数：

- service lease timeout：60 秒；
- maintenance check interval：5 秒；
- client lease renew interval：1 秒；
- client renew request timeout：1 秒。

### 19.6 recovery

同 session、同 fingerprint 的下一次 register 可以利用 _recoverable_sessions 重新创建 service。

这解决了：

- service 因 lease 过期被清理；
- client 仍然保留同一个 session；
- client 重新注册时不被旧 session 规则直接拒绝。

但当前 close handler 捕获异常后只记录日志，_finish_expiration 仍可能把 session 标记为 recoverable。于是旧 service 可能尚未真正释放，新 service 已经创建，形成 F-005。

### 19.7 retired session

新 session 替换旧 session 时，旧 session 加入 _retired_sessions。此后：

- renew 被拒绝；
- get_for_session 被拒绝；
- unregister 被拒绝；
- client 收到 STALE_SESSION_PREFIX 后标记 _superseded。

这是一种 session fencing 设计，可以防止旧 client 继续操作新 service。

缺点是每次替换都会追加 session，只有整个 lifecycle manager close 才清空，形成 F-006 的无界增长。

---

## 20. Client 业务生命周期

文件：mp/kv_cache_client.py。

### 20.1 client 本地状态

| 字段 | 作用 |
|---|---|
| _registration | 当前配置和已编码 payload |
| _session_id | client 创建时生成的 UUID hex |
| _registered | server 当前是否认为已注册 |
| _superseded | 当前 session 是否被新 session 替换 |
| _closed | client 是否关闭 |
| _lease_thread | renew/re-register 线程 |

### 20.2 _configure_registration()

它在 registration lock 内：

1. 拒绝 closed client；
2. 拒绝 superseded client；
3. 禁止一个 client 同时注册 Scheduler 和 Worker；
4. 保存 registration 和 payload；
5. 将 _registered 置 false。

然后：

1. 调用 _try_register；
2. 启动 lease loop；
3. 返回当前注册是否成功。

如果 server 尚未启动，_try_register 会返回 false，但 lease loop 后续可以重试。

### 20.3 lease loop

每秒执行一次：

- 如果尚未 registered：尝试 registration；
- 如果已 registered：发送 RENEW；
- timeout、busy、unavailable：标记 unregistered；
- ServiceNotRegistered：标记 unregistered 并尝试重新注册；
- StaleSession：标记 superseded，终止恢复；
- 其他远程错误重新抛出并由 loop 日志记录。

这里区分了：

- 临时不可用；
- 服务已经丢失；
- 当前 session 已被新 session 替换。

### 20.4 lookup()

lookup()：

1. 先检查 superseded；
2. 取得 SchedulerRegistration；
3. 编码 request；
4. 未 registered 时先尝试 registration；
5. registration 仍失败则返回 (0, False)；
6. RPC lookup；
7. BUSY、timeout、unavailable 映射为 cache miss；
8. ServiceNotRegistered 映射为 unregistered + miss；
9. StaleSession 映射为 superseded + 抛 ServiceSessionExpiredError；
10. decode_lookup_response。

这是一个“远程服务不可用时降级为 miss”的策略。优点是避免缓存服务故障直接拖垮推理；缺点是业务配置错误和服务未准备好也可能表现为 miss。

### 20.5 close()

close()：

1. 标记 closed；
2. 停止 lease thread；
3. 尝试 unregister；
4. 关闭 MPClient。

unregister 是 best effort，失败只写 debug 日志。这适合 client 退出时网络已经断开的场景，但服务端仍可能保留 service，必须依赖 lease expiration 清理。

---

## 21. 设计意图和实际行为对照

这是第二类需要重点整理的额外内容。

| 设计意图 | 代码路径 | 实际行为 | 判断 |
|---|---|---|---|
| MP connector 提供远程 KV cache | connector lookup + server lookup | 只返回命中数，不传输 tensor | 设计未闭环 |
| 异步 KV 加载 | load_async 返回 true | connector load hook 和 get_finished 没有实现 | P1 风险 |
| producer 保存远端 KV | vLLM save hooks | save hooks 全部 no-op，协议无 save route | 功能缺失 |
| 同一 identity 至多一个 service | lifecycle active map + replacement | 新 session 可替换旧 service | 需要 owner/auth 边界 |
| 新注册失败时不影响旧服务 | replacement 应是原子 | 旧 service 先删除再 factory | 非事务 |
| lease 过期释放资源 | expiring + owner close | close 异常被吞掉后仍 recoverable | 状态可能失真 |
| affinity 保证 service 串行 | AffinityExecutor | 同 key 串行，不同 key 可能并行 | 设计成立 |
| executor 有界 | semaphore capacity | 只约束任务数，不约束 payload/queue bytes | 资源保护不完整 |
| 支持 layerwise 配置 | 原有 Ascend Store 支持 | MP lookup hard-code false | 静默降级 |
| registration payload 可安全解析 | identity header + type check | cloudpickle 在认证和类型检查前执行 | 条件性安全风险 |
| 关闭先 drain 再释放 service | KVCacheServer.close | 正常 close 具备该顺序 | 设计较清晰 |
| abort 强制退出 | MPServer abort + service skip close | 依赖进程退出回收 backend | 需服务器确认 |

---

## 22. 资源所有权和关闭顺序详解

这是第三类需要重点整理的额外内容。

### 22.1 正常启动顺序

~~~text
1. KVCacheServer 创建 context、ROUTER、executors、service manager
2. MPServer bind endpoint
3. KVCacheServer.run() 启动 lease maintenance
4. MPServer.run() 进入 ROUTER poll loop
5. client 创建 context、notify socket、I/O thread
6. I/O thread 创建 DEALER、monitor socket、connect
7. client registration
8. server 创建 scheduler/worker service
~~~

### 22.2 正常关闭顺序

~~~text
KVCacheServer.close()
  -> request_stop()
  -> MPServer DRAINING
  -> 不再接收新请求
  -> 已接受请求处理并发送响应
  -> MPServer DRAINED
  -> stop lease maintenance
  -> KVCacheServiceManager.close()
      -> worker lifecycle close
      -> scheduler lifecycle close
      -> owner lane close
      -> service.close()
  -> MPServer.close()
      -> close ROUTER
      -> shutdown executors
      -> close notify socket
      -> terminate context
~~~

核心不变量：

    service 仍存活时，不能先销毁执行它的 RPC executor；
    RPC 不再有请求时，才能关闭 service/backend；
    service/backend 完成关闭后，才能释放 transport。

KVCacheServer.close() 的注释也明确指出，MPServer 在 service close 完成前仍拥有 route executors。

### 22.3 强制 abort 顺序

~~~text
KVCacheServer.abort()
  -> set abort_requested
  -> MPServer.abort()
      -> accepted request 入 ABORTED response
      -> cancel queued executor tasks
      -> 不等待 running handler
      -> MPServer ABORTED
  -> stop lease maintenance(wait=False)
  -> 进程退出模型负责最终回收 service/backend
~~~

正常 close 和 abort 的差异：

| 场景 | 是否等待 handler | 是否调用 service.close | 适用情况 |
|---|---:|---:|---|
| close/drain | 是，受 deadline 约束 | 是 | 正常退出 |
| abort | 否 | 当前 KVCacheServer 不走 graceful service close | 故障或进程终止 |

### 22.4 close failure 的问题

ServiceLifecycleManager._close_on_owner_safely() 捕获 owner close 异常，只记录日志。

随后：

    _finish_expiration()
      -> 删除 _expiring
      -> 写入 _recoverable_sessions

如果 backend.close() 失败意味着资源仍然存在，那么生命周期状态已经对外宣称“可以恢复”，但旧资源可能仍然占用 socket、thread 或 store。这里应该增加 close success/failure 状态，而不是把所有 close 返回都当作可恢复。

---

## 23. 当前代码做得好的地方

### 23.1 分层边界清晰

通用 RPC、业务协议、业务 service、生命周期和 vLLM connector 被拆开。这样做的好处：

- 可以单独测试 frame 编解码；
- 可以用 fake factory 测试生命周期；
- 可以替换 transport，而不重写 lookup 业务。

### 23.2 socket 线程所有权明确

MPClient 和 MPServer 都把 ZeroMQ socket 限制在 I/O thread，public API 通过 queue 和 notify socket 与 I/O thread 通信。这比让多个业务线程直接调用 send/recv 更容易证明正确。

### 23.3 request ID 和 method 双重匹配

client 不仅按 request_id 找 pending，还检查 response method，能捕获：

- 错误响应；
- 迟到响应；
- 服务端 method 串线。

### 23.4 Future 终态处理覆盖面较广

代码考虑了：

- 正常成功；
- handler 抛异常；
- server busy；
- server aborted；
- request timeout；
- transport disconnect；
- client close；
- late response。

多数路径都通过 Future.done() 或 pending map 删除避免重复完成。

### 23.5 affinity executor 解决 service 非线程安全问题

Scheduler 和 Worker 业务对象通常含有可变状态和 backend client。用 identity 固定到 executor lane，能保证同一对象的操作顺序，避免共享 ThreadPoolExecutor 导致并发访问。

### 23.6 lease renew 与业务 lane 分离

renew 只更新生命周期元数据，被安排到 InlineExecutor，不必排队在长时间 lookup 后。这是对“租约不能被业务工作饿死”的正确认识。

### 23.7 registration flight 合并重复 factory

相同 identity、相同 session、相同 fingerprint 的并发注册会共享 Future，而不是重复创建两个 service。不同 identity 仍可并行创建。

### 23.8 显式 session fencing

retired session 和 stale error 让旧 client 在被新 session 替换后不能继续 renew、lookup 或 unregister 新 service。这比只按 identity 接受请求更安全。

### 23.9 graceful drain/backpressure 有基础

MPServer 通过 request_stop、wait_for_drain、abort、response backlog 和 POLLOUT 监听，明确区分：

- 是否接收新请求；
- 是否等待已接受请求；
- 是否强制返回 ABORTED；
- 是否优先发送已有响应。

### 23.10 复用原有 lookup 算法

MPKVPoolScheduler 和 LookupKVPoolWorker 没有复制复杂的 token/block/rank lookup 算法，而是复用 pool_scheduler.py 和 pool_worker.py 的逻辑。这降低了算法分叉风险，使 MP 层聚焦于进程间编排。

### 23.11 测试意图丰富

测试覆盖了很多并发和生命周期意图：

- 同 key affinity 串行；
- 不同 key 并行；
- busy；
- timeout；
- late response；
- drain/abort；
- registration flight；
- stale session；
- lease recovery；
- 注册顺序；
- DP rank 隔离。

这些测试对理解作者想保证的契约很有价值。

---

## 24. 不完善的地方和风险分析

### F-001：lookup 命中与实际 KV transfer 脱节，P1

触发条件：

- MP connector lookup 返回 matched_tokens > 0；
- 或 load_async=True。

代码链：

1. connector.get_num_new_matched_tokens 委托 KVCacheClient.lookup；
2. server 只执行 store.exists；
3. Scheduler 分配 external KV blocks；
4. connector.update_state_after_alloc 是 no-op；
5. connector.start_load_kv、wait_for_layer_load、save_kv_layer、wait_for_save 都是 no-op；
6. get_finished 使用基类默认值；
7. server 没有 KV 数据传输 RPC。

影响：

- 同步路径可能读到未初始化、旧内容或错误内容；
- 异步路径可能永久处于 WAITING_FOR_REMOTE_KVS；
- producer 不会写远端 KV。

修复方向二选一：

1. 实现完整 transfer：metadata、destination blocks、load/save、错误 block、finished_recving、request lifecycle；
2. 明确声明 lookup-only，并禁止返回正 external hit、禁止 load_async 和 producer 配置。

状态：lookup 和 hook 断点静态已确认，实际输出和等待时长需要服务器确认。

### F-002：未认证 endpoint 对 registration payload 执行 cloudpickle.loads，P1 条件性安全风险

触发条件：

- 不可信客户端能连接 endpoint；
- registration payload 可被其控制。

代码链：

    register route
      -> decode_registration_request
      -> decode_registration
      -> cloudpickle.loads(payload)
      -> isinstance/type check

反序列化先于类型检查，因此类型检查不能防止反序列化执行任意构造逻辑。

影响：

- 潜在任意代码执行；
- 恶意大 payload 造成内存/CPU DoS；
- 跨租户 endpoint 下可被任意注册或攻击。

修复方向：

- 用显式、版本化、长度受限的 schema；
- 或在严格认证、签名、ACL 和 payload 上限下使用受控序列化；
- registration 不应直接接收不可信 Python 对象。

状态：静态风险已确认，是否可达取决于 endpoint 部署边界。

### F-003：identity 没有所有权证明，任意新 session 可替换活动 service，P1 条件性风险

触发条件：

- 攻击者能访问 endpoint；
- 知道 engine_id、DP rank，Worker 还需 rank。

代码链：

1. identity 由 payload 字段构成；
2. 没有 token、共享密钥或旧 session proof；
3. 新 session register 会 retire 旧 session；
4. 旧 service 先被删除/关闭；
5. 新 factory 失败时旧 service 已经不存在。

影响：

- 远程 lookup 变 miss；
- 活动 service 被驱逐；
- 跨租户 endpoint 下形成越权或拒绝服务；
- 结合 F-004 产生停机窗口。

修复方向：

- 认证层绑定 identity 和 owner；
- 替换必须提供旧 session 授权；
- 新 service factory 和 binding 成功后再原子切换；
- factory 失败保留旧 service。

### F-004：scheduler publish 后 binding 失败，注册失败但 server 残留 service，P2

代码顺序：

    _schedulers.register(...)
      -> publish scheduler
      -> _schedule_lookup_store_binding(...)
      -> executor.submit(...).result()

如果 worker executor 满、已关闭或 bind_lookup_store 抛异常：

- client 可能收到 BUSY/ERROR；
- client _registered 保持 false；
- server 仍然保留已发布 scheduler；
- 后续依赖 lease expiration 清理。

修复方向：

- factory、binding、publish 组成事务；
- 或先 binding 后 publish；
- 失败时显式 unregister/close；
- 回滚也必须幂等。

### F-005：owner close 失败后仍进入 recoverable，P2

代码顺序：

    expire -> _services 移到 _expiring
          -> owner close
          -> close exception 被记录
          -> _finish_expiration
          -> _recoverable_sessions

这把“关闭请求已发出”误当作“资源已释放”。同 session recovery 可能创建新 service，与旧 backend/thread/socket 并存。

修复方向：

- close 返回成功/失败状态；
- close 失败保留 failed/expiring 状态；
- 失败时重试；
- 未确认资源隔离前禁止 recovery。

### F-006：retired_sessions 无界增长，P2

_retired_sessions 是 identity 到 session set 的映射，每次 replacement/unregister/expiration 都追加 session，只有 manager.close() 清空。

长期运行时，如果 identity 数量大、client 频繁重启、session 不断替换，会导致：

- 内存持续增长；
- stale 检查和日志成本增长；
- 大量历史 session 没有运维可观测的清理策略。

修复方向：

- 保留最近 N 个 session；
- 按时间 TTL 清理；
- 记录 session 时间戳；
- 保留当前 stale 拒绝所需的最小信息。

### F-007：queue 和 wire payload 没有完整资源上限，P2

没有限制：

- MPClient outbound queue；
- 单个 frame bytes；
- multipart 总 bytes；
- block hash frame 数；
- registration serialized bytes；
- 每 identity 并发。

executor 的 64 pending 只是在请求进入 executor 后才生效，不能保护接收和序列化阶段。

修复方向：

- queue maxsize；
- 最大 frame 数、单 frame bytes、总 bytes；
- 反序列化前大小检查；
- ZMQ HWM；
- per-client/per-identity quota；
- BUSY 和超限错误码。

### F-008：layerwise 被静默强制为 false，P2

MPKVPoolScheduler、LookupKVPoolWorker 和 _WorkerLookupBridge 都把 use_layerwise 固定为 false。MP connector 自身也不读取或拒绝该配置。

影响：

- 用户配置 use_layerwise=True 但实际走普通 block lookup；
- 可能产生错误 key 粒度、false hit 或 false miss；
- 与 F-001 组合后可能把错误 external hit 交给不存在的 load 路径。

修复方向：

- 实现完整 layerwise transfer；
- 或初始化时显式抛“不支持 layerwise”；
- 禁止静默降级。

### F-009：cloudpickle import 违反仓库 import 门禁且未声明直接依赖，P2

当前提交在 kv_cache_protocol.py 直接 import cloudpickle，但仓库 checker 将 pickle/cloudpickle 列入禁止项，allowlist 不含新文件；目标项目自身 requirements/pyproject 也未明确声明直接依赖。

影响：

- pre-commit 可能直接失败；
- 最小环境 import 失败；
- 依赖由上游 vLLM 间接提供，部署不稳定。

修复方向：

- 优先采用仓库允许的结构化 serializer；
- 如果必须使用 cloudpickle，完成 allowlist、依赖声明和安全评审。

---

## 25. 其他值得关注但暂未列为正式 finding 的观察

### 25.1 malformed request header 直接丢弃

MPServer._receive_request() 对无法解析 request header 的 frame 直接记录日志并丢弃，不返回 error response。

这对随机坏帧可以避免 server 崩溃，但如果 client 带无 deadline 请求后 frame 被丢弃，Future 是否长期悬挂需要服务器验证。

### 25.2 MPClient send exception race

非 zmq.Again 的 send 异常可能发生在 pending map 插入前。需要可控 socket/context fault 注入，检查所有 Future 是否最终完成。

### 25.3 prompt_token_ids 为 None

MP protocol 直接执行 len(request.prompt_token_ids)。vLLM Request 类型允许某些 prompt embeddings-only 场景使用 None。由于原有 AscendStore 的共用 scheduler 也依赖 token ids，本轮不把它单独归因于 MP 提交，但产品范围必须明确。

### 25.4 两套 MP transport 实现并存

根目录 mp_kv_cache.py 与 mp/rpc/client.py 都实现了自己的 I/O thread、Future、monitor、timeout 和 close。它们目前由不同测试使用，实际主链只用新实现。

长期维护时应：

- 明确旧实现的保留目的；
- 统一错误、protocol 和 close 语义；
- 或删除未使用的重复实现。

### 25.5 lookup miss 的错误吞没

store.exists 异常通常被转换为 hit=0。对远端暂时不可用，这是合理的降级；但对配置不一致、权限错误、数据损坏或 protocol bug，静默变成 miss 会降低可观测性。

---

## 26. 现有测试能证明什么

提交中的测试数量较多，适合用来反推设计，但不能把测试文件中的断言当成本轮已经运行过的证据。

### 26.1 RPC 测试

test_rpc.py 覆盖意图包括：

- protocol round trip；
- invalid deadline；
- client/server round trip；
- request_stop；
- close/abort；
- handler exception；
- executor failure；
- running handler drain；
- queued request abort；
- client/server backpressure；
- request deadline；
- 多 client response；
- busy；
- out-of-order response；
- late response；
- Future cancellation；
- bounded executor；
- affinity 串行/并行；
- executor shutdown。

仍无法证明：

- 真实网络断开、context fault、send exception 下所有 Future 都完成；
- 大 payload 和 multipart 资源是否受限；
- malformed header 的无 deadline 行为。

### 26.2 registration/lifecycle 测试

覆盖：

- idempotent registration；
- fingerprint conflict；
- concurrent same registration；
- factory failure；
- new session replacement；
- old session fencing；
- stale session；
- recovery；
- lease renewal；
- service expiration；
- owner lane close；
- manager close。

仍缺少：

- owner close 抛异常后 recovery 禁止；
- publish 后 binding failure rollback；
- retired session 长时间增长；
- 未认证 client 的 identity replacement。

### 26.3 lookup 测试

覆盖：

- rank-0 coordinator；
- DP 隔离；
- worker/scheduler 注册顺序；
- unregister 后 miss；
- 原有 lookup 逻辑复用；
- TP rank 全部命中要求；
- store 未绑定时 miss；
- store failure 时 miss。

这些测试验证的是 lookup 数值和调用顺序，不验证真实 KV tensor 内容。

### 26.4 connector 测试

覆盖：

- Scheduler/Worker registration；
- Scheduler lookup delegation；
- Worker 调用 Scheduler lookup 的错误；
- 空 metadata；
- registration failure 时关闭 client；
- invalid URL。

没有覆盖：

- update_state_after_alloc；
- start_load_kv；
- save_kv_layer；
- wait_for_save；
- get_finished；
- request_finished；
- 真实 vLLM forward step。

### 26.5 旧 mp_kv_cache 测试

test_mp_kv_cache.py 针对根目录旧实现，覆盖：

- PING/ECHO；
- 多进程；
- heartbeat；
- disconnect；
- timeout；
- close；
- recovery callback。

它们不能直接证明新 mp/ RPC 层或 AscendStoreMPConnector 的行为。

---

## 27. 推荐的服务器验证矩阵

以下不是本轮已经执行的结果，而是后续应执行的验证顺序。

### 27.1 前置条件

- 固定上述 commit；
- 记录 vLLM、Python、pyzmq、cloudpickle、backend 版本；
- 使用 tcp://127.0.0.1 或 ACL 保护的临时 endpoint；
- server、client、scheduler、worker、backend 使用独立日志；
- 准备远端和本地数值不同的 sentinel KV；
- 准备 fake scheduler/worker/store 注入 close/factory/binding failure；
- 设置进程内存和时间上限，避免压力测试影响其他服务。

### 27.2 P1 功能验证

#### 同步 external hit

检查：

- scheduler 返回 external tokens；
- update_state_after_alloc 是否产生 load metadata；
- Worker 是否调用 start_load_kv；
- 目标本地 block 是否等于远端 sentinel；
- forward 输出是否与本地预填充基线一致。

通过条件：只有远端 KV 实际写入本地 buffer 且输出一致才通过。

#### 异步 external hit

检查：

- WAITING_FOR_REMOTE_KVS 是否进入；
- Worker 是否产生 finished_recving；
- get_finished 是否返回 request id；
- 请求是否在有界时间内恢复调度；
- load error 是否能回传 invalid block ids。

如果产品不支持异步，初始化时应直接拒绝，而不是返回 true。

#### save/producer

检查：

- save_kv_layer；
- wait_for_save；
- request_finished；
- 远端 store 是否出现新 key/value；
- 独立 client 是否能 lookup 命中。

### 27.3 安全和身份

在隔离临时进程中：

- 记录实际 bind URL，确认是否监听 *；
- 检查 ACL；
- 使用无害 reduce 探针验证反序列化边界；
- 不得向生产 endpoint 发送恶意 payload；
- 注册合法 service 后，用不同 session、相同 identity 尝试替换；
- factory 成功和 factory 失败都要测；
- 记录旧 service 是否仍可用、count 是否一致。

### 27.4 RPC 并发、超时和关闭

逐项验证：

- 多 client 不同 identity 并发；
- 同 affinity key 串行；
- 不同 key 并行；
- executor 满时 BUSY；
- request timeout 后 late response 被丢弃；
- handler 运行、排队、断连、重连时分别 close/abort；
- 每个 Future 只完成一次；
- malformed request 不拖垮 server；
- send/recv/context fault 后所有 Future 有终态。

### 27.5 registration、lease、recovery

验证：

- 首次注册；
- 同 session 同 fingerprint；
- 同 session 不同 fingerprint；
- 不同 session replacement；
- stale old session；
- server 先不启动、client 先注册、server 后启动；
- worker 不 renew、scheduler 继续 renew；
- owner close failure；
- 大量 session replacement 的 retired set 和 RSS；
- registration publish 后 binding failure。

### 27.6 输入和资源上限

测试：

- 空、截断、非法 UTF-8、负数、超大整数；
- 重复 block hash；
- 超大 hash list；
- 大 registration payload；
- 高并发 client queue；
- executor pending；
- ZMQ HWM；
- use_layerwise=True；
- hybrid KV、多 group、TP/PP/DP 组合；
- prompt embeddings-only。

记录：

- BUSY/timeout 比例；
- RSS；
- CPU；
- thread count；
- executor capacity；
- renew success rate；
- service count；
- response latency。

---

## 28. 建议修复和演进顺序

### 第一优先级：明确产品定位并解决 F-001

选择一个方向。

#### 方向 A：完整 transfer connector

需要新增或接通：

- Scheduler 侧 load metadata；
- Worker 侧 start_load_kv；
- 真实 KV block copy/get；
- layer load wait；
- save_kv_layer；
- wait_for_save；
- request_finished；
- get_finished；
- invalid block reporting；
- sync/async error propagation；
- producer/consumer role contract。

#### 方向 B：lookup-only connector

需要：

- 正 external hit 禁止进入 vLLM allocation；
- load_async 显式拒绝；
- kv_producer/kv_both 显式拒绝；
- use_layerwise 显式拒绝；
- connector 名称和配置文档明确写 lookup-only；
- lookup 只用于统计、预检查或其他不会承诺 KV 可用性的场景。

### 第二优先级：安全边界

- 在跨进程/跨主机部署前解决 F-002；
- 为 endpoint 增加认证/ACL；
- 不可信 registration 不使用 cloudpickle；
- registration payload 设置大小上限；
- F-003 的 identity replacement 必须有 owner proof。

### 第三优先级：registration 事务和 close 状态

- binding 成功后再 publish，或实现完整 rollback；
- close 失败进入 failed/expiring，而不是直接 recoverable；
- 禁止未隔离旧资源时创建同 identity 新 service；
- 给 close/recovery 增加资源状态观测。

### 第四优先级：资源和兼容性

- queue、frame、payload、identity 配额；
- retired session 有界清理；
- layerwise 显式 reject 或完整实现；
- cloudpickle import policy 和直接依赖；
- 统一新旧两套 MP transport 实现。

### 第五优先级：端到端测试

至少新增：

1. 同步 sentinel KV load；
2. 异步 bounded completion；
3. producer save 后独立 lookup；
4. replacement factory/binding rollback；
5. owner close failure recovery fence；
6. payload/frame/queue limits；
7. layerwise explicit reject；
8. transport send exception Future terminal state；
9. endpoint ACL/auth；
10. pre-commit forbidden-import 和最小依赖安装。

---

## 29. 最终理解框架

可以用四句话理解这次提交：

1. 它把原有 Ascend Store 的 lookup 能力包装进了一个带 ZeroMQ、注册、续租和生命周期管理的多进程服务。
2. 它在 RPC、affinity、Future、drain/abort、session fencing 等基础设施上做了不少有价值的工程设计。
3. 它目前闭合的是“找到远程 KV 命中数量”，没有闭合“把 KV 内容加载到本地 vLLM buffer”。
4. 因此，注册和 lookup 代码不能单独证明 AscendStoreMPConnector 已经是可生产使用的完整 KV transfer connector。

在完成 F-001 的产品定位和实现/降级设计、收紧 F-002/F-003 的安全边界、修复生命周期回滚问题并完成服务器验证之前，不应给出“可直接合入生产”的结论。

---

## 30. 相关文档

本目录中已有的配套材料：

- review_plan.md：审核范围和分阶段检查计划；
- review_report.md：正式静态 findings；
- finding_matrix.md：F-001 到 F-009 的矩阵；
- coverage_gaps.md：测试覆盖缺口；
- server_validation_plan.md：服务器验证方案。

本文与上述文档的关系：

- 本文负责解释“代码怎么设计、怎么运行、各模块如何协作”；
- review_report.md 负责集中记录“哪些地方构成问题”；
- coverage_gaps.md 负责记录“现有测试没有证明什么”；
- server_validation_plan.md 负责说明“如何在真实环境补足证据”。
