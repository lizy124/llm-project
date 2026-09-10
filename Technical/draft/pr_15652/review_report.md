# PR #15652（AscendStore 多进程 KV Cache Server）全面审核报告

- **审核基线**：仓库 `D:\project\code\vllm-ascend`，分支 `pr-15652`，HEAD `d30d52a1e23e1f6fcddfa2fbdd6821e4d7e74bad`（基点 `6c118c6da`，共 10 个 PR commit）
- **审核方式**：全量静态走读（RPC / service / kv_cache / pool / npu_ipc / 接入面 六层），父类逐方法对照（`KVPoolScheduler` / `KVPoolWorker` / 三 Backend / 六个传输线程类），跨模块时序与错误路径专项推演，全部结论经二次重读交叉复核
- **本报告取代**：同目录 `review_report.md`（针对旧快照 `0d85d6b7`，lookup-only 阶段，文件布局已不存在，其结论与当前代码不对应，对照关系见第 8 节）
- **结论分级**：〔已确认〕= 调用链可由代码直接推出；〔需运行确认〕= 影响幅度需在真实环境验证

---

## 1. 总体结论

**架构判断：这是一次质量显著高于旧快照的重写，整体设计是成立的，可以进入合入前的收敛阶段。**

当前实现已经从旧快照的"lookup-only + worker 钩子 no-op"演进为完整的多进程架构：

- vLLM 进程只保留**活状态**（Request 引用、BlockPool、NPU 事件录制、KV 事件聚合），跨进程只传**快照与命令**；
- KVCacheServer 进程拥有完整的 `KVPoolScheduler` / `KVPoolWorker` 业务语义（复用父类，适配层极薄），传输线程、backend、导入的 NPU 映射全部由服务端持有；
- 服务生命周期（注册/租约/过期/恢复/替换/关停）由一个严谨的状态机管理，客户端有对称的降级与恢复分类学；
- NPU 跨进程共享复用仓库既有 weight-transfer IPC 惯例（`ipc_args[6]` 设备索引改写、UUID 设备解析），事件保序设计有明确论证。

**主要问题集中在错误路径与边界配置**，共 11 项（2 高 / 4 中 / 5 低）。其中两项高危需要合入前处理：

| 编号 | 级别 | 一句话结论 |
|---|---|---|
| F-01 | 高 | MP layerwise 接收线程丢失 `save_failure_checker`，发送线程死亡时从"进程内快速失败"退化为"无限等待 + 引擎挂死 + 清理挂死" |
| F-02 | 高 | MP Worker 强制 `pcp_rank=dcp_rank=0` 且无 CP 准入门禁，PCP/DCP>1 部署下 pool key 跨 rank 冲突（静默数据污染） |
| F-03 | 中 | MultiConnector layerwise slot-release 协调在 MP 模式下静默缺失，无门禁 |
| F-04 | 中 | worker 亲和线程队头阻塞 + 默认 4 线程 < 常见 TP 规模 → lookup 5s 超时降级抖动 |
| F-05 | 中 | 四个无 deadline RPC 与服务端挂死叠加 = 引擎静默挂死（F-01 的放大器） |
| F-06 | 中 | vLLM 进程与 server 进程的 env 契约完全隐式（MOONCAKE_* 等） |
| F-07 | 低 | at-most-once 命令信道：响应丢失会丢 touch/free 命令、提前释放块 |
| F-08 | 低 | cloudpickle 信任边界依赖 bind-url 保持在 loopback，无强制 |
| F-09 | 低 | `retired_sessions` 无界增长（自旧版遗留） |
| F-10 | 低 | close 失败仍进入 recoverable，允许同 session 重建（旧 F-005 语义残留） |
| F-11 | 低 | wire payload 无大小/数量上限，无 per-identity 配额 |

---

## 2. 代码布局与调用链（审核范围）

新增生产代码（`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/` 下）：

```text
mp/
├── rpc/            protocol(framing) / server(admit·drain·abort 状态机) / client(IO线程·deadline) /
│                   executor(Inline/Bounded/Affinity) / error
├── service/        lifecycle(状态机) / error
└── kv_cache/
    ├── protocol.py     21 个 KVCacheMethod 的编解码（cloudpickle 字典帧）
    ├── registration.py 可序列化配置投影 + 身份 + 边界校验
    ├── scheduler_view.py  服务端 Request/Blocks/Output 投影
    ├── manager.py      业务编排 + scheduler↔worker lookup 桥
    ├── server.py       RPC 路由 + 线程归属 + 生命周期协调
    ├── client.py       注册/租约恢复循环 + 业务 RPC + 降级分类
    ├── cli.py          `vllm-ascend kv-cache-server` 进程入口与信号监管
    ├── npu_ipc.py      存储导出/导入 + interprocess 事件
    └── pool/           scheduler.py(MPKVPoolScheduler) / worker.py(MPKVPoolWorker+线程mixin)
        └── backend/    mooncake / memcache / yuanrong 三变体
ascend_store_mp_connector.py   vLLM 侧双角色 connector
```

核心闭环（均已走读闭合）：

```text
vLLM Scheduler ──(快照↑ 命令↓)──> MPKVPoolScheduler(父类 KVPoolScheduler)
     │                                   │ 非layerwise lookup 经 _WorkerLookupBridge
     │                                   ▼ (同进程 executor 桥接, 无二次 RPC)
vLLM Worker ──(metadata↓ IPC句柄↓)──> MPKVPoolWorker(父类 KVPoolWorker)
     │   NPU事件: vLLM侧record → IPC handle → 服务端 import 保序
     └── KV 数据: vLLM 进程持有真实显存, 服务端 import 映射后由 backend 直接 D2H/H2D
```

---

## 3. 分层走读结论

### 3.1 RPC 层（`mp/rpc/`）

- **framing**：method + frames 的显式协议，帧类型/整数宽度校验；payload 层统一 cloudpickle 字典帧（`mp/kv_cache/protocol.py:699-735`）。
- **MPServer 状态机**：admit/drain/abort 三态；`request_stop` 发起排空，`abort` 取消排队工作。与 `KVCacheServer.close()` 的协作次序正确：先停租约维护（不等待）→ 等 RPC 排空 → 停维护（等待）→ 在**路由 executor 存活期间**执行 service close（`mp/kv_cache/server.py:316-342`），保证 close 命令能投递到各身份的亲和线程。
- **MPClient**：独立 IO 线程 + deadline；`timeout_ms=None` 表示无期限等待（业务层有意使用，见 F-05）。
- **executor**：`AffinityExecutor` 以 `hash(key) % N` 把同身份任务固定到同一线程串行执行，容量 = workers+pending 有界（BoundedSemaphore），满则 Busy。这是"同服务状态无并发访问"这一核心安全性质的来源，设计正确。

### 3.2 service 层（`mp/service/lifecycle.py`）

状态机 `absent → registering → active → expiring → recoverable`，附 retired-session 拒绝集：

- **替换语义**：同 identity 新 session 注册时，旧 active 先 retire + close（锁外执行），factory 成功后才发布新服务；factory 失败回滚到 `restore_on_failure`（recoverable）而非丢失（`register`/`_create_and_publish`/`_fail_registration`，L141-260）。〔已确认〕
- **过期事务**：`expire_leases` 持 `_expiration_lock`，先在生命周期锁内移入 `expiring`（立即对会话请求不可见），锁外经 owner handler 在**该服务的亲和线程**上 close，完成后才暴露 recoverable（L368-390）。过期清理与业务执行不会并发触碰同一 service——这个不变量靠亲和线程串行化保证，成立。〔已确认〕
- **会话校验原子性**：`_get_and_renew_entry` 把"校验 session → 检查 active → 续租"合并在同一锁内，过期无法插队在校验与续租之间摘除服务（L277-296）。〔已确认〕
- 残留问题：F-09（retired 无界）、F-10（close 失败仍 recoverable）。

### 3.3 kv_cache 层

- **protocol**：21 个方法对称编解码；cloudpickle 用于富对象（RequestView、AscendConnectorMetadata、WorkerKVCacheSpec、registration dataclass）。信任模型 = 受信本地 RPC（见 F-08）。
- **registration**：配置投影只携带服务端消费的字段；边界校验严格——spec 模块白名单（仅 vllm / vllm_ascend 两处）、dataclass 检查、`_project_extra_value` 只放行纯值/纯容器（`registration.py:524-555`），防止不可序列化或不安全的配置跨进程。identity 由 config 推导并与显式 identity 比对（`manager.py:160-172`），防止身份与配置不一致。〔设计良好〕
- **manager**：业务方法全部经 `_require_*` → `get_for_session`（顺带续租）。`_lookup_worker` 把 scheduler 侧非 layerwise lookup 提交到 rank-0 Worker 的亲和线程并阻塞等待（L314-337），**刻意不续租**（"Scheduler 流量不能给空闲 Worker 续命"，L309-312 注释）——租约语义考虑周到。但引入 F-04 的队头阻塞。
- **client**：注册/租约恢复循环 + 降级分类学（详见 3.6 节错误路径）。
- **scheduler_view**：静态字段（prompt/block_hashes）注册时快照一次，动态字段（num_computed_tokens/block_ids/all_token_ids）每步增量刷新（`pool/scheduler.py:107-134`），与父类读取的演化状态一致；新请求未注册时保留 payload 对象让父类抛出**同样的**错误（L118-122 注释）——失败语义也被对齐。

### 3.4 pool 层（与父类逐方法对比，本 PR 最关键环节）

**MPKVPoolScheduler**（`mp/kv_cache/pool/scheduler.py`）：

- `client = _WorkerLookupBridge(...)` 替换进程内 `LookupKeyClient`（ZMQ REQ → 对端 engine 的 lookup RPC 线程）。语义对齐验证：进程内 lookup 最终调用 `worker.lookup_scheduler(token_len, hash_strs, group_ids, use_layerwise=False, hbm_hit_tokens)`；MP 桥经 manager 同样调 `lookup_scheduler`，hex 字符串一致（`manager.py:339-352` vs `pool_worker.py:2317-2426`）。use_layerwise 分支不走桥（走 `store_scheduler.batch_get_key_info`，与进程内相同）。〔已确认等价〕
- `_block_pool = _BlockPoolProxy()`：父类 mamba 记账的 touch/free 落在代理上，命令经 RPC 响应回传给 vLLM 侧真实 BlockPool 执行（`scheduler.py:33-69` + connector `build_connector_meta`/`update_connector_output`）。所有权拆分干净。
- `close()` 为空操作（Scheduler 服务无 backend/线程/导入内存），满足 manager 统一关闭契约。

**MPKVPoolWorker**（`mp/kv_cache/pool/worker.py`）：

- `_init_parallelism_info` 重写：不依赖 torch.distributed，tp/pp 由注册 rank 算术推导（正确：`rank % tp_size`、`(rank // tp_size) % pp_size`）；**但 pcp_rank/dcp_rank 硬编码 0**（L213-216）→ F-02。
- `_init_backend` 推迟到缓存注册（首次 IPC 映射确定 NPU 后再建 backend 并绑定同设备）——因为只有注册才知道设备，这个延迟激活设计是**必要且正确**的。
- `configure_kv_caches` 幂等（同 spec 重试直接返回），激活失败时撤销部分激活并保留失败映射引用供 close 释放（L196-247）——部分失败清理路径完整。
- `_ImportedNPUEvent.record()` 为 no-op、`synchronize()` 委托：保序理由充分（在服务端重录会丢失源流顺序，L57-68 注释）。
- `wait_for_save` 用导入事件替代进程内新建事件，`request_queue.join()` 阻塞语义与父类一致（L327-347 vs `pool_worker.py:1735-1753`）。
- 三 backend 变体：mooncake/memcache 带注册回滚与显式注销，yuanrong 显式拒绝无法安全释放的 pregistration 模式（`backend/yuanrong.py:17-27`）——backend 所有权边界处理认真。
- **传输线程 mixin**：`_MPTransferThreadMixin` 增加准入门（stopping 后拒收）、`_STOP_REQUEST` 有序停止标记、两阶段停止（先全部发停止标记再逐个 join，`worker.py:462-469`）。`run()` 的异常路径（不 task_done 直接 return）与父类 `KVTransferThread.run` 一致（`kv_transfer.py:487-507`），非回归。`_set_os_thread_name` 存在于父类（L482-485），mixin 调用安全。〔已确认〕
- **`_start_kv_transfer_threads` 重写引入两处真实差异**：未传 `save_failure_checker`、未传 `external_slot_release_waiter`（对照进程内 `pool_worker.py:475-540` 两项都传）→ F-01、F-03。

### 3.5 npu_ipc（`mp/kv_cache/npu_ipc.py`）

- 存储导出按 `(device, data_ptr)` 去重，每个 allocation 只导出一次，tensor 视图用 storage_index+offset 描述，**无进程本地地址跨线**（L141-198）。
- 导入侧校验完整：尺寸、单设备约束、逐 tensor 越界检查（`_validate_tensor_spec`，L316-335），失败时已导入映射直接丢弃交 GC。
- `ipc_args[6] = device_index` 的位置假设与已合入的 weight-transfer 引擎**逐字相同**（`npu_ipc_engine.py:214-218` 同款注释），版本耦合风险为既有惯例而非新增。
- UUID 设备解析（`_resolve_device`）解决两进程逻辑设备号不一致问题；`import_storage`/`import_npu_event` 中的 `torch.npu.set_device` 副作用在亲和线程上执行，torch 设备上下文为线程局部，多 Worker 并发导入互不干扰。〔已确认〕
- 事件生命周期契约成立：源事件在 RPC 期间存活，服务端在 RPC 处理中完成 `from_ipc_handle`（`save_kv_layer_from_event`），客户端 `finally` 中关闭源引用发生在导入完成之后（connector `save_kv_layer`/`wait_for_save` + `exported_event.close()`）。〔已确认〕

### 3.6 接入面

- **connector**（`ascend_store_mp_connector.py`）：双角色门禁清晰；sleep mode 显式拒绝；调度器侧保活 Request 引用并用 `_synced_token_len` 增量采集 `all_token_ids`（与 scheduler_view 的增量刷新严格对偶）；KV 事件本地聚合避免服务端往返（L188-201 注释）；降级兜底三件套——`start_load_kv` 失败把 pending 块全部标记 load-error（强制重算，L269-283）、`wait_for_save` 未被接受时 locally-finished 一次性放行 delayed-free（避免 vLLM 无限等待，L301-320）、`get_block_ids_with_load_errors` 在服务端不可报告时把**全部** pending 视为不安全（保守正确，L340-352）。关闭次序：先关 client（让服务端在源分配仍存活时释放导入映射）再放本地 export（L375-383）。
- **注册路由**（`kv_transfer/__init__.py:23-58`）：`VLLM_ASCEND_STORE_MULTIPROCESS=1` 时 `AscendStoreConnector`/`MooncakeConnectorStoreV1` 两个既有名字透明切换到 MP 类，显式名 `AscendStoreMPConnector` 恒为 MP。env 注释明确"This selects the connector; it does not start the server process"——**server 需操作员先行启动，无自动拉起**，这是部署前提。
- **cli**（`mp/kv_cache/cli.py`）：信号只记录意图（处理器安全），控制线程驱动生命周期；二次 Ctrl-C 强制 abort；退出码区分正常排空(0)/自动 abort(1)/强制(130)。SIGTERM 重复发送不升级（仅 SIGINT 升级），与 systemd 语义兼容。

### 3.7 错误路径与降级分类学（跨模块专项）

客户端降级分类**一致且正确**（`client.py:537-601`）：

| 异常 | 处理 | 语义 |
|---|---|---|
| `MPServerBusyError` | 仅当次调用失败 | 容量瞬态，不破坏会话 |
| timeout / transport 断 / `ServiceNotRegisteredError` | `_mark_unregistered` → 租约线程下个 tick 重注册 | 可恢复 |
| `ServiceSessionExpiredError`（stale session） | `_mark_superseded`，永久失效 | 会话已被替换，不可恢复 |

注册竞态闭环验证：lease 过期 → 服务端 expiring（close 在亲和线程排队）→ 客户端重注册撞 `ServiceBusyError` → `_try_register` 捕获返回 False → 1s 后重试 → 服务端 close 完成、同 session+指纹走 recoverable 恢复路径。**该链条完整闭合**〔已确认〕。前提是 close 能完成——这正是 F-01 打破的环节。

deadline 传递链：scheduler 方法与 get_finished 等 5s；注册 500ms / 缓存注册 5s / 注销 5s / 租约 1s；`wait_for_save`/`start_load_kv`/`wait_for_layer_load`/`save_kv_layer` 无 deadline（事件生命周期契约的有意选择，`client.py:415-418` docstring）→ F-05。

租约续期走 `InlineExecutor`（RPC IO 线程），不排业务队列——"不能让活跃服务因业务阻塞而显得过期"，理由成立（`server.py:283-291`）。

---

## 4. Findings（详细论证）

### F-01〔高〕MP layerwise 接收线程缺失 `save_failure_checker`：错误路径从快速失败退化为三重挂死

**位置**
- 缺失点：`mp/kv_cache/pool/worker.py` `_start_kv_transfer_threads` 中 `MPKVCacheStoreLayerRecvingThread(...)` 构造参数（约 L386-406，参数列表终于 `group_builders=...`，无 `save_failure_checker`）
- 进程内对照：`pool_worker.py` `_start_kv_transfer_threads`（约 L500-540）显式传入 `save_failure_checker=(self.kv_send_thread.raise_if_failed if ... else None)`
- 消费点：`kv_transfer.py:1481-1491`，接收线程 `wait_for_save` 等待循环：
  ```python
  while not self.layer_save_finished_events[wait_for_save].wait(timeout=10):
      if self.save_failure_checker is not None:
          self.save_failure_checker()      # MP 下为 None，永不触发
      logger.info("Layerwise %d save wait timed out, keep waiting before load", ...)
  ```

**证据链**〔已确认，静态可推〕
1. MP 发送线程 mixin 的 `run()` 在 `_handle_request` 异常时 `_record_fatal_error` 后直接 return（`worker.py:96-110`），**不设置** `layer_save_finished_events[layer]`。
2. 接收线程因 `save_failure_checker=None` 无法感知发送线程死亡，在 10s 超时循环中无限等待。
3. 上层 `wait_for_layer_load`（`pool_worker.py:1681-1704`）每 10s 调 `self.kv_recv_thread.raise_if_failed()`——但接收线程**没有失败**（活着且在等待），检查永不命中。
4. `wait_for_layer_load` RPC 无客户端 deadline（F-05）→ vLLM worker 步进循环永久阻塞 → **引擎静默挂死**。进程内对照：接收线程会经 `save_failure_checker()` 抛出 → `_record_fatal_error` → `wait_for_layer_load` 的 `raise_if_failed()` 命中 → 引擎得到显式异常（崩溃可见）。
5. 级联二：服务 close 时 `_stop_kv_transfer_threads` 对卡死接收线程 `stop()` join 挂起 → 亲和线程永久占用 → lease 过期清理挂起 → 该 identity 永远停留在 expiring → 客户端重注册无限 `ServiceBusyError` → **永久降级**（所有 KV 功能退化为 miss）。
6. 级联三：`KVCacheServer.close()` 排空等待挂起，只剩二次 Ctrl-C abort 逃生（优雅关闭丢失）。

**触发条件**：layerwise 模式下发送线程在"最后一次 `raise_if_failed` 检查之后、设置层保存事件之前"死亡（backend batch_copy 失败、mooncake 异常等真实故障窗口）。

**修复方向**：MP `_start_kv_transfer_threads` 补传 `save_failure_checker=self.kv_send_thread.raise_if_failed`（一行修复，恢复与进程内一致的快速失败语义）。

---

### F-02〔高〕`pcp_rank`/`dcp_rank` 强制为 0 且无 CP 准入门禁：pool key 跨 rank 冲突

**位置**
- `mp/kv_cache/pool/worker.py:213-216`：
  ```python
  self.pcp_rank = 0
  self.pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
  self.dcp_rank = 0
  self.dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
  ```
- key 组成：`metadata.py:115-125` `PoolKey.to_string()` 含 `@pcp{pcp_rank}@dcp{dcp_rank}`
- 进程内对照：`pool_worker.py:100-116` 从 `get_pcp_group()`/`get_decode_context_model_parallel_*` 取真实 rank；`_init_key_head_config`（约 L196-209）`my_key_index` 与 `_init_metadata`（约 L306-320）`KeyMetadata(..., self.pcp_rank, self.dcp_rank, ...)` 均使用真实值
- 无门禁佐证：全 `mp/` 目录 grep `pcp|dcp` 仅命中上述 4 行与 registration.py 的 size 字段（L213-219，仅携带 size 用于 block size 缩放），无任何 `pcp_size>1` 拒绝逻辑

**证据链**〔已确认（键冲突推导），污染幅度需运行确认〕
1. CP 部署下（pcp_size>1 或 dcp_size>1），两个 rank 仅 pcp_rank 不同、tp_rank 相同（如 rank 0 与 rank tp_size），`head_or_tp_rank = registered_rank % tp_size` 相同。
2. 同一请求在 CP rank 间共享 block_hashes（提示词相同），`cp_scale` 只放大 block 尺寸不区分 rank。
3. 两 rank 的 MPKVPoolWorker 均以 `pcp_rank=0` 生成 KeyMetadata → 对同一 chunk_hash 生成**完全相同的 pool key**，但持有**不同 token 分片**的数据 → 互相覆盖写、读取拿到错误分片 → **静默数据污染**。
4. 次生影响：与进程内 connector 的键空间也不兼容（进程内写 `@pcp1`，MP 读 `@pcp0`）——混布署（部分引擎 MP、部分进程内）无法共享池内容。
5. MP Worker 无法从全局 rank 算术推导 pcp_rank（取决于运行时 group 划分），代码作者选择置 0——**要么是刻意不支持 CP，要么是遗漏；两种情况都缺一个显式门禁**。

**修复方向**（二选一）：
- 在 `AscendStoreMPConnector.__init__` 或 registration 校验中拒绝 `prefill_context_parallel_size>1 or decode_context_parallel_size>1`（明确不支持）；或
- 把 pcp_rank/dcp_rank 纳入 WorkerRegistration（由 vLLM 进程从真实 group 取出随注册传递），并在 manager 侧校验一致性。

---

### F-03〔中〕MultiConnector layerwise slot-release 协调在 MP 模式静默缺失

**位置与证据链**〔已确认〕
1. MP 接收线程构造未传 `external_slot_release_waiter`（同 F-01 构造点）；进程内对照 `pool_worker.py:517-540` 传入。
2. `MPKVPoolWorker` 未重写 `set_external_slot_release_waiter`（父类方法会向已运行线程补交 waiter，`pool_worker.py:517-530`），且**不存在任何 RPC 让 vLLM 进程把这个 waiter 送到服务端**。
3. `AscendStoreMPConnector` 全文（383 行）无 `set_external_slot_release_waiter`/`supports_layerwise_buffer_reuse`/`wait_for_layer_reuse`。
4. `AscendMultiConnector` 侧用 `getattr(connector, "set_external_slot_release_waiter", None)` 防御式探测（`ascend_multi_connector.py:60-63`）→ MP 子连接器既不会成为 provider（选择条件要求 `connector_worker` 属性，L44-49），也永远不会成为 sink → `_external_slot_release_sink_configured` 恒 False，组合体回退到自行等待路径。

**影响**：`MultiConnector + AscendStoreMPConnector(layerwise)` 组合下，跨连接器的物理槽位复用协调静默退化为组合体轮询等待——功能缺失或性能退化，无任何报错。进程内 connector 完整支持该协调（纯转发器，`ascend_store_connector.py:201-212`）。

**修复方向**：同 F-02——要么补一条 waiter 注册 RPC（把 MultiConnector 的 waiter 经 vLLM 侧转发到服务端接收线程），要么在 MP connector 初始化时检测组合场景并显式拒绝/告警。

---

### F-04〔中〕worker 亲和线程队头阻塞；默认线程数（4）小于常见 TP 规模 → lookup 超时降级抖动

**证据链**〔已确认（结构），幅度需运行确认〕
1. 所有 Worker 业务 RPC（含阻塞性强的 `WAIT_FOR_SAVE`：`request_queue.join()` 等全部 put 完成；`START_LOAD_KV` 同步路径阻塞 `m_store.get`）按身份亲和串行（`server.py:96-137` 路由 + `executor.py` AffinityExecutor）。
2. scheduler 侧非 layerwise lookup 经 `_WorkerLookupBridge` → `manager._lookup_worker` → **提交到同一 Worker 亲和线程并阻塞**（`manager.py:314-337`）。发送线程在忙时，lookup 只能排队。
3. lookup 客户端 deadline 5s（`client.py:59`）→ 排队超时 → `_try_scheduler_rpc` 降级返回 (0, False)（cache miss）。
4. 进程内对照：lookup 由对端 engine 的 ZMQ lookup RPC 线程执行（`pool_scheduler.py:1017-1065` `LookupKeyClient` → 独立服务线程），**不**与传输线程串行——MP 改变了这一执行模型。
5. 容量叠加：`DEFAULT_WORKER_THREADS = 4`（`server.py:57`），TP8 部署 8 个 Worker 身份经 `hash(key) % 4` 两两共线程 → 某 Worker 的长 `wait_for_save` 直接阻塞另一个 Worker 的全部 RPC（含其 5s deadline 的 get_finished）。

**修复方向**：`--worker-threads` 文档要求 ≥ TP×engine 数（或按注册身份自动扩容）；中期可给 lookup 单独的只读通道（lookup_scheduler 只读 token_database/m_store.exists，进程内本就不与传输互斥）。

---

### F-05〔中〕四个无 deadline RPC：服务端任何挂死直接转化为引擎静默挂死

**位置**：`client.py:415-427`（wait_for_save，docstring 明示"Wait without a default deadline so the source Event outlives accepted Store work"）、L478-491（start_load_kv）、L493-500（wait_for_layer_load / save_kv_layer，经 `_worker_rpc` 直接上抛基础设施错误）。

**论证**：无 deadline 的动机成立——源事件必须存活到服务端 Store 完成、以及避免大传输被客户端误杀。代价是：服务端任何无限等待（F-01 即实例；未来任何同类缺陷）都会让 vLLM worker 步进循环永久阻塞，且无超时日志可循。F-01 修复后此风险大幅收窄，但结构性风险仍在。

**修复方向**：设置一个足够大的上界（如 10-30 min，取决于最大传输规模）+ 超时时输出明确诊断（正在执行的 method、服务线程状态）；或服务端为无 deadline 方法提供活性心跳。至少应在部署文档写明该行为。

---

### F-06〔中〕双进程环境变量契约完全隐式

**证据**〔已确认〕：以下配置在**服务端进程**的 env 中读取，而配置者通常只配置 vLLM 进程：
- `MooncakeBackend`：`MOONCAKE_MASTER`、`MOONCAKE_GLOBAL_SEGMENT_SIZE`、`MOONCAKE_CONFIG_PATH`、`ASCEND_ENABLE_USE_FABRIC_MEM`（`backend/mooncake_backend.py:72,302-303,326`）——MP 模式下 backend 在 KVCacheServer 内构造；
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`：`pool_scheduler.py`（约 L102-104）与 `pool_worker.py` `_build_cache_coordinator`（约 L565-569）在**服务端**构造服务时读取——server 进程 env 若与 vLLM 不同，调度语义静默分叉；
- 对照：`kv_connector_extra_config` 已随注册投影跨进程（正确做法），但 env 类配置没有等价机制，也无启动时一致性校验或告警。

**修复方向**：最小改动——cli 启动时打印关键 env 快照 + 文档列明"server 进程必须继承的 env 清单"；更完整的做法是把可投影的配置并入 registration 并在服务端覆盖 env 缺省。

---

### F-07〔低〕at-most-once 命令信道：响应丢失窗口丢命令 / 提前释放

**证据链**〔已确认（窗口存在），频率需运行确认〕
1. 服务端 handler 先落状态、后经 RPC 响应回传命令（touch/free 块 id、delay_free 决策）。客户端超时即 `_mark_unregistered` 抛弃结果（`client.py:575-601`），但服务端可能已完成执行。
2. `update_connector_output` 响应丢失：服务端已 pop `sending_blocks` 并产出 free ids → 丢失 → vLLM 侧 delayed-free 的 mamba 块**永不释放**（BlockPool 容量泄漏）。
3. `build_connector_meta` 响应丢失：本步 metadata 被丢弃（无 save/load），但服务端 `sending_events/sending_blocks` 已记账 → 无 completed_events 抵达 → 服务端小额泄漏。
4. `request_finished` 降级返回 (False, None)：vLLM 立即释放块，而服务端 Store 仍在飞行中读这些块（导入映射仍有效，块可被新请求复用）→ 池内容可能被复用数据污染（进程内同步调用不存在该窗口）。

**缓解现状**：这些 handler 都很快、5s 预算充足，正常负载下几乎不可达；但无任何对账/重放机制。

**修复方向**：free/touch 命令改为带序号的增量游标（客户端下次 RPC 重取未确认区间）；或至少对 `request_finished` 降级改为保守 delay（宁可漏放不可早放）并记录告警。

---

### F-08〔低/部署边界〕cloudpickle 信任边界依赖 loopback bind，无强制

**证据**：全部业务体与注册负载为 cloudpickle（`protocol.py:15,77-88,699-735`；`npu_ipc.py:101-118`）；门禁经 allowlist 显式放行（commit `d30d52a1e`，`tools/check_forbidden_imports.py` 新增两文件，理由"trusted local RPC，同 npu_ipc_engine 先例"）；默认 bind `tcp://127.0.0.1:5555`（`envs.py:81`）但 `--bind-url` 可任意覆盖。同 identity 未认证替换（旧 F-003）在 lifecycle 中依旧可行（`register` 允许新 session 替换 active，L155-160）——在受信本机假设下可接受，一旦 bind 到非 loopback，反序列化 RCE + 服务驱逐都暴露。

**修复方向**：cli 对非 loopback bind 打印显著警告（或要求显式 `--i-understand-nonlocal-bind`）；文档写明信任边界。

---

### F-09〔低〕`retired_sessions` 无界增长（旧版遗留，新代码未修复）

`lifecycle.py`：`_retire_session_locked` 只追加（每次客户端重启/替换 +1 个 UUID），唯一清空点是 manager `close()`。长寿命 server + 频繁重启场景内存缓慢增长。修复：有界近期窗口或按时间淘汰。

### F-10〔低〕close 失败仍进入 recoverable（旧 F-005 语义残留）

`expire_leases` → `_close_on_owner_safely` 吞掉 close 异常仅记日志（L388-396 区域）→ `_finish_expiration` 无条件转 recoverable → 同 session+指纹可重建新服务，而旧服务的线程/backend 资源可能未释放。MPKVPoolWorker.close 内部已有嵌套 try/finally 兜底，实际触发面小；但状态机本身不区分 close 成败。修复：close 失败时保持 failed/expiring 态并阻断同 session 恢复。

### F-11〔低〕wire payload 无大小/帧数上限，无 per-identity 配额

server `recv_multipart` 整包接收，protocol 只校验帧类型/整数宽度；executor 容量（workers+pending）有界可挡并发洪峰，但单请求超大 payload（如巨量 block hash）无上限。受信本机场景下风险低；非 loopback 时并入 F-08。

---

## 5. 设计观察（非缺陷，供权衡参考）

1. **metadata 三跳序列化**：`AscendConnectorMetadata` 每步走 server→(RPC)→scheduler connector→(vLLM 广播)→worker connector→(RPC)→server worker。正确性无虞，序列化开销可测（layerwise 下每层事件 spec 亦每层一跳）。
2. **每业务 RPC 顺带续租**（`get_for_session`）+ 专用 1s renew 双保险——活跃服务不会过期，空闲服务 2×timeout 内被回收，语义自洽。
3. **每层一个 interprocess event**：layerwise save 每层 `record_npu_event()` 新建导出（connector `save_kv_layer`），相比进程内复用 `sync_save_events` 列表开销更高，但换来源流保序的正确性——权衡合理。
4. **lookup 目标固定 rank 0**（`_LOOKUP_COORDINATOR_RANK`，`manager.py:354-359`）：与进程内 dp 维度 socket 语义对齐；若 rank 0 Worker 未注册（`find` 返回 None → 0 命中）属安全降级。

---

## 6. 已验证良好的方面（正面清单）

1. **lifecycle 状态机**：转换全在锁内、慢操作（factory/close）全在锁外、retired-session 拒绝陈旧调用、factory 失败回滚保留恢复路径、并发 factory 乱序完成时按发布顺序重插（`_publish_locked` 注释）——这是全 PR 质量最高的组件。
2. **降级分类学**在 client 全部 21 个方法上贯彻一致（busy/可恢复/永久三类），且 `_report_degradation` 去重避免日志风暴。
3. **注册边界校验**：spec 模块白名单 + 纯值投影 + identity/config 交叉验证，把不兼容配置挡在进程边界之外。
4. **snapshot+增量同步**的 scheduler_view 与 connector 侧 `_collect_token_id_increments` 严格对偶，服务端请求状态与进程内演化一致。
5. **BlockPool 所有权拆分**：真实池留在 vLLM，服务端记账、命令回传——避免了两进程操作同一池的并发地狱。
6. **npu_ipc** 与既有 weight-transfer 引擎同惯例（`ipc_args[6]`），UUID 设备解析、越界校验、单设备约束、事件源流保序论证完整。
7. **backend 所有权**：mooncake/memcache 注册回滚 + 显式注销 + 失败列表保留重试语义；yuanrong 拒绝不安全模式而非带病运行。
8. **两阶段线程停止** + 运行时激活的部分失败撤销（`_configure_kv_caches`）。
9. **cli 信号协调**：意图记录/控制线程/退出码分类/二次 Ctrl-C 逃生，信号处理不持锁。
10. **connector 降级兜底三件套**（load-error 全标 / locally-finished 放行 / 服务端不可报告时全 pending 视为不安全）+ 本地 KV 事件聚合 + 关闭次序（先 client 后 export）。

---

## 7. 建议处置

**合入前必须处理**
- F-01（一行修复，恢复快速失败语义）
- F-02（至少加 CP 门禁；键空间冲突是静默数据污染）

**合入前强烈建议**
- F-04（`--worker-threads` 与 TP 规模的关系写入文档/启动告警）
- F-06（env 契约文档 + 启动时打印关键 env）
- F-08（非 loopback bind 警告）

**可后续跟进**
- F-03、F-05、F-07、F-09、F-10、F-11

---

## 8. 与旧版报告（快照 `0d85d6b7`）的对照

| 旧编号 | 旧结论 | 现状 |
|---|---|---|
| 旧 F-001 | lookup-only、worker 钩子 no-op、external hit 与传输脱节（P1） | **已解决**：完整传输 RPC（START_LOAD_KV / WAIT_FOR_SAVE / SAVE_KV_LAYER / WAIT_FOR_LAYER_LOAD / GET_FINISHED / GET_BLOCK_IDS_WITH_LOAD_ERRORS...），MPKVPoolWorker 完整继承父类传输语义 |
| 旧 F-002 | 未认证 endpoint cloudpickle RCE | **部分解决**：门禁 allowlist 显式化信任模型 + 默认 loopback；非 loopback 风险保留 → 本报告 F-08 |
| 旧 F-003 | identity 无所有权证明可被替换 | **语义保留**（trusted-local 假设下接受）→ 并入 F-08 |
| 旧 F-004 | 注册后 lookup-store 绑定失败留孤儿服务 | **已解决**：新架构无独立绑定阶段，注册即发布，factory 失败有完整回滚 |
| 旧 F-005 | close 失败仍 recoverable | **语义残留** → 本报告 F-10 |
| 旧 F-006 | retired_sessions 无界 | **未修复** → 本报告 F-09 |
| 旧 F-007 | wire/queue 无界 | **部分解决**（executor 有界容量）→ 本报告 F-11 |
| 旧 F-008 | layerwise 被静默置 False | **已解决**：MP Scheduler/Worker 真实读取 use_layerwise，六种 layerwise 传输线程 MP 化（但引入新的 F-01） |
| 旧 F-009 | cloudpickle 违反 import 门禁 | **已解决**：commit `d30d52a1e` allowlist（新 F-08 记录其边界） |

---

## 9. 审核方法与局限

- 全部结论来自静态走读与调用链推演，未启动服务、未连接 ZMQ、未运行测试（用户明确本次不审查测试代码）。
- 每项 Finding 均经过"发现 → 重读上下文挑战 → 二次确认"流程；F-01/F-02/F-03 的父类对照证据（进程内传参 vs MP 传参）逐一复读核对。
- 标注〔需运行确认〕的项目建议在合入前按第 4 节各条"修复方向/验证建议"在服务器环境补充验证。