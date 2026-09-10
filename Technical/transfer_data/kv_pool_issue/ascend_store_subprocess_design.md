# AscendStore KV 传输子进程重构 - 详细设计

## 1. 设计目标

目标是将 AscendStore KV 传输执行从 worker 进程内 daemon 线程迁移到独立传输子进程，同时保持所有传输模式的现有行为。

子进程应拥有 Python-heavy 的传输热路径：

- Key 构建。
- Address 和 size list 构建。
- Layerwise task scheduling。
- 在 backend/runtime 能力允许时，提交后端 `put`、`get`、`batch_copy`、`batch_write_finish` 和 `batch_remove_lease`。

worker 进程应保留 connector-facing orchestration：

- 构建高层 request metadata。
- 维护 scheduler-facing request lifecycle state。
- 提交 transfer command。
- 等待 completion 或 failure。
- 处理 restart 或 fallback。

必需属性：

- 非 layerwise、key layerwise 和 GVA layerwise 模式功能等价。
- greedy 或 non-greedy generation 无精度回归。
- 传输子进程 crash 不能导致 worker Python 进程 crash。
- 必须支持 restart 或 downgrade 到进程内线程模式。
- 性能测量必须区分 GIL relief 和 IPC overhead。

设计必须把跨进程 KV buffer 共享和 NPU stream/event ordering 当作一等要求，因为它们属于原题和测试验收面。

## 2. 非目标

本次重构不应改变：

- KV key string format。
- Block hash semantics。
- Scheduler protocol。
- Backend storage semantics。
- Model execution behavior。
- Cache-hit 或 failed-block semantics。
- Layerwise visible ordering。
- GVA allocation、publication 或 lease semantics。

第一版不能删除现有进程内线程实现。它应成为 fallback engine 和等价性测试的 reference behavior。

## 3. 代码约束依据

### 3.1 现有对象不是 IPC Payload

`ReqMeta`、`LayerTransferTask` 和 `LayerLoadTask` 是进程本地执行对象，不是 wire object。

相关字段包括：

- `ReqMeta.current_event: torch.npu.Event | None`。
- 用于 block id、GVA data 和 load GVA data 的 numpy array。
- 已生成的 save/load key。
- Partial GVA state。
- `LayerBlockRange.request`，其中嵌入完整 `ReqMeta`。
- `LayerLoadTask.attention_start_gate`，这是进程本地同步对象。

因此 subprocess mode 必须使用专门的 payload dataclass，并进行字段级转换。

### 3.2 NPU Event 语义对正确性很关键

当前行为依赖 NPU event synchronization：

- 非 layerwise save 在 worker 中记录 `current_event`，send 线程在 backend `put` 前同步它。
- Layerwise save 在提交 layer save 前记录 `sync_save_events[layer_id]`。
- GVA load 会等待前一个 save-layer CPU completion，然后同步 `sync_save_events[wait_for_save]`，再复用 HBM buffer。
- Layerwise prefetch load 可能等待 `AttentionComputeStartGate`。

IPC completion message 不会自动等价于 NPU stream fence。设计必须为每种模式提供 fence strategy。

### 3.3 GVA/Memcache 存在硬性跨进程约束

GVA layerwise 当前会为 memcache backend 启用。当前 scheduler 代码说明 memcache 要求 `batch_alloc` 和 `batch_copy` 在同一进程执行，因为 `batch_copy` 查询的 `gvaBlobTracker` 是 per-process。

这意味着 GVA subprocess 支持受 backend capability gating 约束。如果 child 被设计为执行 GVA `batch_copy`，就必须由 child 拥有兼容的 GVA allocation/tracking，且这些状态对应同一个 worker buffer；或者 backend 暴露受支持的跨进程 import/export 语义。否则，盲目把 GVA `batch_copy` 移到 child process 不是有效设计。

## 4. 架构方案

在 AscendStore KV pool 实现下引入 transfer engine abstraction。

```
AscendStoreConnector
        |
        v
KVPoolWorker
        |
        v
TransferEngine interface
        |
        +-- ThreadTransferEngine
        |
        +-- SubprocessTransferEngine
                  |
                  v
          TransferProcessManager
                  |
                  v
          Transfer subprocess
                  |
                  v
          Mode-specific executor
```

### 4.1 TransferEngine Interface

phase 1 后，worker 不应直接管理 `KVTransferThread` 实例。它应调用 transfer engine，且方法要匹配现有 worker lifecycle 和 scheduler-facing 语义。

建议接口形态：

```python
class TransferEngine(Protocol):
    def register_kv_caches(self, context: RegisterContext) -> RegisterResult: ...
    def start(self) -> None: ...
    def shutdown(self) -> None: ...

    def start_load_kv(self, payload: StartLoadPayload) -> None: ...
    def wait_for_current_layer_load(self, payload: WaitCurrentLayerPayload) -> None: ...

    def save_current_layer(self, payload: SaveCurrentLayerPayload) -> None: ...
    def submit_non_layerwise_save(self, payload: SaveBatchPayload) -> None: ...
    def wait_for_non_layerwise_save(self, payload: WaitSavePayload) -> None: ...

    def get_finished(self, payload: GetFinishedPayload) -> FinishedPayload: ...
    def get_failed_blocks(self) -> set[int]: ...
    def take_kv_events(self) -> list[BlockStoredPayload]: ...
    def take_completed_events(self) -> dict[int, int]: ...
    def health(self) -> TransferEngineHealth: ...
    def raise_if_failed(self) -> None: ...
```

它与简化接口的重要区别：

- `get_finished()` 必须接收当前 `KVPoolWorker.get_finished()` 使用的同等上下文：`finished_req_ids`、`preempted_req_ids`、`delayed_free_req_ids` 和 `loading_req_ids`。
- `wait_for_current_layer_load()` 应建模现有 `current_layer` 状态机，而不是 connector 中未使用的 `layer_name` 参数。
- Failed block、KV event 和 completed event id 是 engine output，必须继续对 scheduler 可见。

### 4.2 ThreadTransferEngine

`ThreadTransferEngine` 包装现有行为：

- 创建相同的 mode-specific send/recv thread class。
- 内部继续使用 `queue.Queue`、`threading.Event` 和 `torch.npu.Event`。
- 向 `KVPoolWorker` 暴露 `TransferEngine` interface。

该 engine 必须存在，用于：

- Fallback。
- Equivalence testing。
- 在不改变默认行为的前提下分阶段迁移。

### 4.3 SubprocessTransferEngine

`SubprocessTransferEngine` 负责：

- Child process startup。
- IPC channel creation。
- Command serialization。
- Result handling。
- Heartbeat/liveness check。
- Backend 和 event-fence capability detection。
- Restart/fallback policy。
- Worker call 与 transfer command 之间的转换。

它不应直接实现 mode-specific transfer logic，而是把 command dispatch 给 child process 内部的 executor。

### 4.4 Transfer Subprocess

child process 初始化：

- Backend instance。
- Device context。
- Imported 或 child-registered KV buffer descriptor。
- Mode-specific executor。
- Child-local key/token cache。
- Request/result loop。

child process 不能 import 或调用 worker forward logic。它只应操作 transfer metadata 和 backend buffer。

## 5. 配置与能力门控

增加保守的配置开关。

建议 engine mode 值：

- `thread`：强制当前线程 engine。
- `subprocess`：对 eligible mode 要求使用 subprocess；unsupported 时根据 fallback policy 决定 raise 或 downgrade。
- `auto`：对支持的 backend/mode 组合尝试 subprocess，不支持则 fallback 到 thread mode。

建议 fallback policy 值：

- `disabled`：subprocess failure 直接向 worker 抛错。
- `restart`：在 retry limit 内 restart child process，不进行 thread fallback。
- `thread_fallback`：subprocess failure 后 downgrade 到 thread engine。
- `auto`：安全时 restart 一次，然后 downgrade。

建议限制：

- `max_restart_attempts`：默认 1。
- `heartbeat_interval_ms`。
- `ipc_timeout_ms`。
- `startup_timeout_ms`。

初始 rollout 应默认 `thread`。只有当某个 backend/mode 组合通过 capability check、equivalence test 和 performance test 后，才启用 `auto`。

### 5.1 Capability Matrix

必须按 backend 和 transfer mode 选择 subprocess mode。

所需能力：

| Mode | Required capability |
| --- | --- |
| Non-layerwise save | Child 能访问 KV buffer；child 能在 backend `put` 前遵守 save fence。 |
| Non-layerwise load | Child 能访问 KV buffer；failed-block reporting 能 round-trip。 |
| Key layerwise save | Child 能访问 KV buffer；child 能遵守 per-layer save fence。 |
| Key layerwise load | Child 能访问 KV buffer；child 能保持 save-before-load 和 attention-start gating。 |
| GVA layerwise save | Child 能拥有/import GVA allocation/tracker state；`batch_alloc`、`batch_copy` 和 `batch_write_finish` 在 child 中安全。 |
| GVA layerwise load | Child 能拥有/import GVA/lease state；`batch_copy` 和 final `batch_remove_lease` cleanup 在 child 中安全。 |

如果缺失某项能力，engine 必须为该 backend/mode 选择 thread mode，并输出一条清晰 warning。

## 6. IPC 协议

### 6.1 Transport

使用适合 Python control message 的本地 multiprocessing transport。

第一版原型可接受选项：

- `multiprocessing.Queue`，分别用于 command queue 和 result queue。
- `multiprocessing.Pipe`，用于简单双向 command/result exchange。
- 如果后续偏好 authenticated local socket，可使用 `multiprocessing.connection`。

第一版使用 command/result queue 最容易测试。除非已明确测量并接受为原型范围，否则大型 array 和 KV buffer 不能通过这些 queue 发送。

实现必须显式定义 process start method，并验证它与 Ascend/NPU runtime initialization 兼容。child entrypoint 应放在可 import 的 module 中，以支持 spawn-based startup。

### 6.2 Message Envelope

所有 IPC message 都应使用 versioned envelope。

```python
@dataclass
class TransferMessage:
    version: int
    message_id: int
    command: str
    payload: object
```

所有 response 必须包含原始 `message_id`。

```python
@dataclass
class TransferResult:
    version: int
    message_id: int
    status: str
    payload: object | None
    error: TransferError | None
```

Status values：

- `ok`
- `accepted`
- `failed_recoverable`
- `failed_non_replayable`
- `failed_fatal`
- `unsupported`
- `shutdown_ack`

### 6.3 Commands

#### HELLO

目的：

- 协商 protocol version 和 child runtime information。

结果：

- Protocol version、child pid、supported command set。

#### PROBE_CAPABILITIES

目的：

- 判断选定 backend/mode 是否能运行 subprocess mode。

Payload：

- Backend name 和 configuration。
- Transfer mode。
- Device identity。
- Required capability list。

结果：

- 每个 required item 的 capability result。
- 如 unsupported，返回原因。

#### REGISTER

目的：

- 在 subprocess 中初始化 backend 并 register/import KV cache buffer。

Payload：

- Backend name 和 configuration。
- Worker rank / TP rank / PP rank / PCP rank / DCP rank / device identity。
- Transfer mode。
- KV cache group metadata。
- Buffer descriptor 或 child-registration information。
- Layerwise layout information。

结果：

- Backend ready。
- Buffer registration/import result。
- 如果 registration 无法工作，返回 unsupported process sharing reason。

#### START

目的：

- Registration 后通知 child process 进入 ready state。

结果：

- Ready ack。

#### START_LOAD

目的：

- 为非 layerwise 或 layerwise mode 提交一个 load step。

Payload：

- 包含 request payload 和当前 layerwise scheduling context 的 `StartLoadPayload`。

结果：

- Accepted ack，不一定表示 load-complete。

#### WAIT_CURRENT_LAYER_LOAD

目的：

- 保持当前 worker-facing `wait_for_layer_load()` 语义。

Payload：

- Current layer index。
- Prefetch/dependency state。
- Wait-for-save layer if any。
- 需要时的 attention-start fence descriptor。

结果：

- Current layer load completed。
- Failed block ids if any。
- Layer failed 时返回 error。

#### SAVE_CURRENT_LAYER

目的：

- 提交一个 layer save batch，并保持当前 `save_kv_layer()` 行为。

Payload：

- Current layer index。
- Layer transfer task payload。
- Save fence descriptor。
- Final-layer behavior flag。

结果：

- 根据当前语义返回 accepted 或 completed。
- 如果 final-layer completion 发生，返回 finished request id。

#### SUBMIT_SAVE_BATCH

目的：

- 提交非 layerwise save request。

Payload：

- Stored request payload。
- Save fence descriptor。
- Request id。

结果：

- Accepted ack。

#### WAIT_SAVE_BATCH

目的：

- 保持当前 blocking non-layerwise save 行为。

结果：

- Save complete。
- 产生的 KV event。
- Completed event id。
- Finished request id。

#### GET_FINISHED

目的：

- 按当前 scheduler filtering 语义返回 finished sending/receiving request id。

Payload：

- Scheduler 已知 finished request id。
- Preempted request id。
- Delayed-free request id。
- Loading request id。
- Layerwise flag。
- Load-async flag。

结果：

- 与当前 tuple 等价的 saved/loaded completion payload。

#### GET_FAILED_BLOCKS

目的：

- 返回并清空 failed load block id。

结果：

- Failed block id set。

#### TAKE_EVENTS

目的：

- 返回并清空 KV cache event 和 completed worker event。

结果：

- Serialized KV event。
- Completed event id。

#### HEARTBEAT

目的：

- Liveness check。

结果：

- 当前 child status、active command id、transfer mode、backend status。

#### SHUTDOWN

目的：

- Graceful subprocess exit。

结果：

- Shutdown ack。

### 6.4 Error Payload

error 应结构化。

```python
@dataclass
class TransferError:
    error_type: str
    message: str
    command: str
    message_id: int
    request_id: str | None
    layer_id: int | None
    recoverable: bool
    replay_safe: bool
    traceback: str | None
```

worker log 应包含 command、request id、layer id、backend/mode 和 fallback decision。

## 7. Payload 设计

### 7.1 Request Payload

不要发送完整 `ReqMeta`，应使用字段级 payload。

```python
@dataclass
class ReqMetaPayload:
    req_id: str
    token_len_chunk: int
    save_start_token: int
    save_end_token: int
    target_token_len: int
    block_ids_by_group: list[list[int]]
    block_hashes: list[bytes | str]
    can_save: bool | None
    load_spec: LoadSpecPayload | None
    is_last_chunk: bool | None
    kv_cache_group_ids: list[int] | None
    kv_cache_families_by_group: list[str] | None
    skip_null_blocks_by_group: list[bool] | None
    num_prompt_tokens: int | None
    token_ids: list[int] | None
    original_block_size: list[int] | int | None
    event_id: int | None
    partial_block_index: int | None
    last_block_gva: int | None
    gva_block_offset: int
    load_gva_block_offset: int
    partial_save_gva_per_group: list[int]
    partial_load_gva_per_group: list[int]
```

默认不应把大型 numpy array 加入该 payload。需要时应使用 shared memory descriptor 或 child-local reconstruction。

### 7.2 LoadSpec Payload

```python
@dataclass
class LoadSpecPayload:
    can_load: bool
    vllm_cached_tokens: int
    kvpool_cached_tokens: int
    kvpool_store_skip_tokens: int | None
    token_len: int | None
```

### 7.3 Layer Transfer Payload

不要发送带 embedded request reference 的 `LayerBlockRange`。

```python
@dataclass
class LayerBlockRangePayload:
    request: ReqMetaPayload
    start_block: int
    end_block: int
    partial_block_index: int | None

@dataclass
class LayerTransferTaskPayload:
    layer_id: int
    block_ranges: list[LayerBlockRangePayload]
    group_id: int
    layer_idx_in_group: int
    write_finish_keys: list[str]
    shared_block_data: SharedBlockDataDescriptor | None
    cached_process_tokens_ref: str | None
```

subprocess mode 激活时，`cached_process_tokens` 应在 child process 内构建和缓存，而不是从 worker 序列化到 child。

### 7.4 Layer Load Payload

```python
@dataclass
class LayerLoadTaskPayload:
    wait_for_save_layer: int | None
    transfer_tasks: list[LayerTransferTaskPayload]
    layer_id: int
    attention_start_fence: FenceDescriptor | None
```

### 7.5 Runtime-Only Field 替代

| Runtime field | Replacement |
| --- | --- |
| `queue.Queue` | IPC command queue/result queue。 |
| `threading.Event` load/save completion | Message id 加 child result；必要时 worker 侧 condition/event。 |
| `torch.npu.Event` | `FenceDescriptor`，或没有 cross-process fence 时 worker pre-synchronization。 |
| `AttentionComputeStartGate` | 显式 gate command/fence descriptor，或 child submit 前由 worker 持有 gate。 |
| Backend instance | Child-local backend instance。 |
| Worker task list references | 每个 command 的 immutable payload snapshot。 |

## 8. 跨进程 Buffer 与 Fence 设计

### 8.1 KV Cache Buffers

设计必须避免通过 IPC 复制 KV cache tensor。

首选策略：

1. worker 按当前方式计算 KV cache memory region。
2. worker 为每个 region 创建 `BufferDescriptor`。
3. child 用自己的 backend import/register 这些 descriptor。
4. child 使用 descriptor 和 cache layout metadata 构建 address 并提交 I/O。

```python
@dataclass
class BufferDescriptor:
    descriptor_type: str
    base_addr: int | None
    length: int
    device_id: int
    handle: bytes | None
    group_id: int | None
    cache_role: str
```

如果 backend 只接受注册进程内有效的 raw virtual address，那么 raw `base_addr` 不是有效 subprocess descriptor。Capability probing 必须检测这种情况并强制 thread fallback。

### 8.2 NPU Fence Strategy

按 backend/mode 使用以下策略之一，优先级从高到低：

1. Cross-process NPU event/fence handle。
   - worker 记录 event，并把 `FenceDescriptor` 发送给 child。
   - child 在 backend I/O 前等待该 fence。
   - 这最接近现有异步行为。

2. Worker pre-synchronization before IPC。
   - worker 记录并同步 NPU event，然后发送 command。
   - child 可以安全提交 backend I/O。
   - 正确性更简单，但性能收益可能缩小，必须测量。

3. Thread fallback。
   - 如果既没有 cross-process fence，也无法接受 pre-sync 行为，则该模式保留在 `ThreadTransferEngine`。

```python
@dataclass
class FenceDescriptor:
    fence_type: str
    event_id: int | None
    handle: bytes | None
    already_synchronized: bool
    layer_id: int | None
```

### 8.3 GVA/Memcache Strategy

GVA subprocess 支持需要以下受支持设计之一：

1. Child-owned GVA path。
   - child 执行 `batch_alloc`，拥有 child-local GVA tracker state，执行 `batch_copy`，调用 `batch_write_finish`，并释放 lease。
   - worker 只发送 key、size、request/layer metadata 和 KV buffer descriptor。
   - subprocess mode 下 worker 不再拥有 `_allocated_gvas`，或只为 observability 安全镜像 child state。

2. Backend-supported tracker import/export。
   - worker 可以 export allocation/tracker descriptor。
   - child import 后可以安全调用 `batch_copy`。

3. Thread fallback。
   - 如果 memcache 的 `gvaBlobTracker` 严格是 per-process，且没有 child-owned full path，GVA layerwise 必须保留在线程 engine。

实现不能在前两种路径被原型和测试证明之前宣称支持 GVA subprocess。

## 9. 子进程内 Executor 设计

subprocess 通过 shared interface dispatch 到 mode-specific executor。

```python
class TransferExecutor(Protocol):
    def probe_capabilities(self, payload: CapabilityProbePayload) -> CapabilityResult: ...
    def register(self, payload: RegisterPayload) -> RegisterResult: ...
    def start_load(self, payload: StartLoadPayload) -> TransferAck: ...
    def wait_current_layer_load(self, payload: WaitCurrentLayerPayload) -> TransferResultPayload: ...
    def save_current_layer(self, payload: SaveCurrentLayerPayload) -> TransferAck | TransferResultPayload: ...
    def submit_save_batch(self, payload: SaveBatchPayload) -> TransferAck: ...
    def wait_save_batch(self, payload: WaitSavePayload) -> TransferResultPayload: ...
    def get_finished(self, payload: GetFinishedPayload) -> FinishedPayload: ...
    def get_failed_blocks(self) -> FailedBlocksPayload: ...
    def take_events(self) -> EventsPayload: ...
    def shutdown(self) -> None: ...
```

具体 executor：

- `NonLayerwiseTransferExecutor`。
- `KeyLayerwiseTransferExecutor`。
- `GVALayerwiseTransferExecutor`，仅在 GVA capability probe 通过后启用。

当前 transfer thread 逻辑应尽量抽取成可复用 helper function。不要在 child 中直接 subclass `threading.Thread` 逻辑；应抽取纯 transfer routine，用于 key/address/size 构建和 backend call。

## 10. 按模式设计

### 10.1 Non-Layerwise Executor

职责：

- 接收 save/load payload。
- 在 subprocess 中构建 key string。
- 在 subprocess 中构建 addr 和 size。
- 保持 TP-rank ordering/rotation。
- 保持 store mask、skip range 和 skip-null-block logic。
- 调用 backend `exists`、`put` 和 `get`。
- 返回 KV event、completed event id、finished request id 和 failed block id。

重要等价性要求：

- 保持 key string format。
- 保持 TP-rank ordering/rotation。
- 保持 store mask 和 skip range logic。
- 保持 failed block recording behavior。
- 覆盖 synchronous 和 async load path。如果只迁移 async load，同步非 layerwise load 仍是共享 GIL 的路径。

### 10.2 Key Layerwise Executor

职责：

- 在 subprocess 中维护 cached process token。
- 在 subprocess 中构建 layer-specific key。
- 保持 per-layer save/load ordering。
- 在 backend `put` 前遵守 save fence。
- 保持 `consumer_is_to_put` wait behavior。
- 返回 per-layer completion。

重要等价性要求：

- 保持 `LayerPoolKey` formatting。
- 保持 layer index mapping。
- 保持 `wait_for_layer_load()` 和 `save_kv_layer()` 当前可见行为。
- 保持 final-layer finished request behavior。

### 10.3 GVA Layerwise Executor

capability probe 通过后承担以下职责：

- 安全拥有或 import GVA allocation/tracker state。
- 在 subprocess 中构建 shared block data，或 import worker-built descriptor。
- 构建 GVA/address/size array。
- 遵守 save/load fence。
- 按正确 direction 调用 batch copy。
- save 时调用 `batch_write_finish`。
- final load layer 调用 `batch_remove_lease`。
- 处理 partial failure cleanup。

重要等价性要求：

- 保持 batch copy limits。
- 保持 H2D stagger behavior。
- 保持 lease release semantics。
- 保持 final-layer cleanup。
- 保持非幂等 `batch_alloc` 行为。

如果 capability probe 失败，不能启用该 executor。engine 必须 fallback 到 thread mode 并报告 unsupported reason。

## 11. Worker 侧生命周期

### 11.1 Startup

重构后 worker 流程：

1. `KVPoolWorker.register_kv_caches()` 按当前方式准备 local metadata 和 memory region。
2. worker 构建选定的 `TransferEngine`。
3. 如果配置要求，engine 启动 subprocess。
4. engine 发送 `HELLO` 和 `PROBE_CAPABILITIES`。
5. 如果支持，engine 发送 `REGISTER`，携带 backend、layout、buffer descriptor 和 mode configuration。
6. child 初始化 backend 并 import/register buffer。
7. child 发送 ready ack。
8. worker 继续正常运行。

如果 startup 失败：

- `thread` mode：按当前行为 raise。
- `subprocess` mode 且无 fallback：raise structured error。
- `auto` 或 fallback-enabled mode：log 一条 warning 并启动 `ThreadTransferEngine`。

### 11.2 Normal Operation

worker-facing method 尽量接近现有行为：

- `start_load_kv()` 构建 request payload 并提交 load command。
- `wait_for_layer_load()` 使用 current-layer state，并阻塞直到 engine 报告 completion。
- `save_kv_layer()` 记录或创建所需 fence state，并提交 layer save。
- `wait_for_save()` 提交非 layerwise save 并等待 completion。
- `get_finished()` 把 scheduler filtering context 传给 engine。
- `get_block_ids_with_load_errors()` 返回 child-reported failed block。
- KV event 从 engine output 收集。

worker 不应直接访问 child internal queue、child event 或 child executor state。

### 11.3 Shutdown

shutdown 必须幂等：

1. subprocess running 时发送 `SHUTDOWN`。
2. 在 timeout 前等待 `shutdown_ack`。
3. timeout 时 terminate child process。
4. 清理 local IPC handle。
5. 如果已经 downgraded，正常 shut down thread engine。

## 12. Crash Recovery 和 Fallback

### 12.1 Crash Detection

通过以下方式检测 child failure：

- Process exit code。
- IPC EOF / broken pipe。
- Heartbeat timeout。
- Command timeout。

manager 检测后应立即标记 engine unhealthy。

### 12.2 Command Replay Policy

必须按 replay safety 给 command 分类。

Replay-safe 或通常安全：

- `HELLO`。
- `PROBE_CAPABILITIES`。
- transfer 开始前的 `REGISTER`，前提是 backend registration 幂等。
- `HEARTBEAT`。
- Read-only metadata query。

除非 backend 证明，否则 replay-unsafe：

- backend `put` 可能已经开始后的 `SUBMIT_SAVE_BATCH`。
- backend `put`、`batch_copy` 或 `batch_write_finish` 可能已经开始后的 `SAVE_CURRENT_LAYER`。
- GVA `batch_alloc`，因为当前 worker state 把它视为非幂等。
- GVA `batch_write_finish`。
- Final-layer `batch_remove_lease`。

默认策略：

- 不盲目 replay 非幂等 write。
- commit state 未知时 fail 当前 in-flight transfer。
- 如果 fallback enabled，未来 transfer downgrade 到 thread mode。

### 12.3 Restart Policy

满足以下条件时允许 restart：

- Child idle 时死亡。
- Child 在 command accepted 前死亡。
- Active command 是 replay-safe。
- Backend/buffer registration 可以重复。
- Restart attempt 仍在配置 limit 内。

Restart 流程：

1. Stop 或 reap old process。
2. Start new process。
3. 重新执行 `HELLO`、`PROBE_CAPABILITIES` 和 `REGISTER`。
4. 把 subprocess-local cache 重建为空。
5. 恢复接收新 request。
6. 根据 command replay safety 决定 fail 或 retry in-flight request。

### 12.4 Downgrade Policy

以下情况 downgrade 到 thread mode：

- Subprocess startup 失败。
- Buffer sharing unsupported。
- 选定模式不支持 NPU fence semantics。
- GVA tracker/import support unsupported。
- Restart limit exceeded。
- IPC protocol failure fatal。
- Backend 无法在 child process 中初始化。

Downgrade 流程：

1. 为当前 backend/mode 标记 subprocess engine disabled。
2. Shut down child resources。
3. 用现有已注册 worker state 创建 `ThreadTransferEngine`。
4. 启动现有 thread-mode transfer worker。
5. 后续 request 继续走 thread mode。

downgrade 应输出一条清晰 warning，而不是每个 request 反复刷日志。

### 12.5 In-Flight Requests

in-flight request handling 必须保守：

- 如果 load in-flight 时 child 死亡，把 load 视为 failed，并尽可能记录 failed block。
- 如果 save in-flight 且幂等性不确定，不要假设它成功。
- 如果 backend 能确认 committed key，用 backend state 完成或跳过 retry。
- 否则 fail transfer task，并让后续 task 走 fallback。
- 对 GVA lease，只通过 documented backend-safe API 做 best-effort cleanup。

## 13. 精度保持

精度保持依赖等价的 transfer ordering，而不只是 backend call 成功。

Guardrails：

- 保持 key format 不变。
- 保持 block id 和 hash mapping 不变。
- 保持 TP-rank ordering 不变。
- 保持 layer load wait semantics 不变。
- 保持 attention-start gating 不变。
- 保持 NPU save/load fence semantics 不变。
- 保持 failed-load behavior 不变。
- 保持 final-layer lease release 不变。
- 保持 scheduler-visible finished request reporting 不变。

Regression test 应比较：

- Greedy generation。
- 固定 seed 的 non-greedy generation。
- Cache hit path。
- Cache miss / partial miss path。
- 三种 transfer mode。

## 14. 可观测性

增加指标和日志，用于回答 subprocess mode 是否有帮助以及是否发生 fallback。

建议 counter/timer：

- `kv_transfer_engine_mode`。
- `kv_transfer_backend_mode_supported`。
- `kv_transfer_subprocess_start_total`。
- `kv_transfer_subprocess_restart_total`。
- `kv_transfer_subprocess_fallback_total`。
- `kv_transfer_ipc_request_total`。
- `kv_transfer_ipc_error_total`。
- `kv_transfer_ipc_roundtrip_ms`。
- `kv_transfer_ipc_payload_bytes`。
- `kv_transfer_ipc_serialize_ms`。
- `kv_transfer_ipc_deserialize_ms`。
- `kv_transfer_key_build_ms`。
- `kv_transfer_addr_size_build_ms`。
- `kv_transfer_backend_put_ms`。
- `kv_transfer_backend_get_ms`。
- `kv_transfer_backend_batch_copy_ms`。
- `kv_transfer_fence_wait_ms`。

日志应包含：

- Engine selected。
- Subprocess pid。
- Backend registration success/failure。
- Capability probe results。
- Restart reason。
- Downgrade reason。
- 带 command id 和 request id 的 fatal command failure。

## 15. 测试计划

### 15.1 单元测试

为 `TransferProcessManager` 添加测试：

- 启动 subprocess 并收到 ready ack。
- 发送 shutdown 并 clean exit。
- 检测 child process exit。
- 检测 IPC disconnect。
- 把 child exception 转成 structured error。
- 对 replay-safe command 在配置 limit 内 restart。
- 对 non-replayable command 拒绝 replay。
- Restart limit 后 downgrade。

为 payload conversion 添加测试：

- `ReqMeta` 到 `ReqMetaPayload`。
- `LayerTransferTask` 到 `LayerTransferTaskPayload`。
- `LayerLoadTask` 到 `LayerLoadTaskPayload`。
- Runtime-only field 被排除。
- Required field 被保留。
- 大型 numpy array 使用 descriptor path，或 unsupported 时被拒绝。

为 `SubprocessTransferEngine` 添加测试：

- Register/start lifecycle。
- Submit load command。
- Submit save command。
- Wait save command。
- Wait current layer load command。
- 带 preempted/delayed/loading filtering 的 get finished command。
- Failed block propagation。
- KV event propagation。
- Startup failure 时 fallback 到 `ThreadTransferEngine`。
- Unsupported capability result 时 fallback。

用 mocked backend 添加 executor 测试：

- Non-layerwise executor 使用预期 keys/addrs/sizes 调用 backend `put/get`。
- Key layerwise executor 保持 layer key format。
- Capability probe 失败时禁用 GVA layerwise executor。
- Capability probe 通过时，GVA layerwise executor 调用 batch copy、write finish 和 final lease release。
- Backend exception 按配置返回 recoverable/fatal/non-replayable status。

### 15.2 集成测试

有 device/backend 测试基础设施时：

- 一个 request 走 non-layerwise save/load。
- 一个 request 走 key layerwise save/load。
- 仅在 capability probe 通过时，一个 request 走 GVA layerwise save/load。
- idle 时 kill transfer subprocess 并验证 restart。
- request 中 kill transfer subprocess 并验证 worker survives。
- 强制 buffer registration failure 并验证 thread fallback。
- 强制 IPC disconnect 并验证 structured failure/fallback。
- 使用真实 backend registration/import path 验证 cross-process buffer sharing。
- 验证 NPU fence strategy 保持 ordering。

### 15.3 精度测试

使用 deterministic workload：

- Greedy decode baseline vs subprocess mode。
- Non-greedy fixed-seed baseline vs subprocess mode。
- Partial KV load failure path。
- Multi-layer transfer path。
- Capability probe 通过时的 GVA path。

预期结果：

- Deterministic test 的 token output 与 baseline 匹配。
- Cache hit/miss reporting 与 baseline 匹配。
- Finished request reporting 与 baseline 匹配。

### 15.4 性能测试

在相同配置下做 before/after 对比：

- 现有 thread mode。
- Subprocess mode。
- 从 steady-state metric 中排除 forced fallback 的 subprocess mode。
- 如果使用 worker pre-sync fence strategy，也单独测量。
- 如果支持 cross-process fence strategy，也单独测量。

收集：

- Throughput。
- P50/P95/P99 latency。
- Worker CPU profile。
- Transfer process CPU profile。
- Key-building CPU share。
- Address/size construction time。
- IPC overhead。
- Backend operation time。
- Fence wait time。

## 16. 实施计划

### Phase 0: Feasibility Spikes

- 探测 child process 中 backend buffer sharing/import 支持。
- 探测 NPU event/fence strategy。
- 探测 memcache GVA child-owned allocation/copy path。
- 记录 supported backend/mode matrix。

退出标准：

- 明确 capability matrix。
- Unsupported mode 可以确定性 fallback。
- 未证明前不宣称 GVA subprocess support。

### Phase 1: 引入 TransferEngine 边界

- 添加 `TransferEngine` interface。
- 将当前 thread startup logic 移入 `ThreadTransferEngine`。
- 让 `KVPoolWorker` 调用 engine method，而不是直接拥有 mode-specific transfer thread。
- 保持当前行为。
- 添加测试证明 thread engine 保持现有行为。

该阶段风险较低，因为默认行为仍是 thread mode。

### Phase 2: 添加 IPC 类型和 Process Manager

- 添加 versioned command/result dataclass。
- 添加 payload conversion helper。
- 添加 `TransferProcessManager`。
- 添加 child process main loop。
- 添加用于 lifecycle test 的 mock executor。
- 测试 startup、shutdown、exception、timeout、disconnect、restart 和 non-replayable failure handling。

该阶段先证明 process isolation，再迁移真实 transfer logic。

### Phase 3: 使用 Mocked Backend 添加 Subprocess Engine

- 实现 `SubprocessTransferEngine`。
- 在 config 后面接入 `KVPoolWorker`。
- 支持 probe/register/start/shutdown 和简单 command。
- 添加 fallback 到 thread engine。
- 默认保持 subprocess mode off。

该阶段验证 worker integration。

### Phase 4: 迁移 Non-Layerwise Executor

- 将非 layerwise key/address/size construction 移入 subprocess executor。
- 覆盖 async load 和 synchronous load 行为。
- 尽可能复用现有 helper function。
- 保持 backend call 和 result handling。
- 添加与 thread engine 的 equivalence test。
- 添加 IPC payload size metric。

### Phase 5: 迁移 Key Layerwise Executor

- 将 layer key construction 移入 subprocess。
- 维护 subprocess-local cached process token。
- 保持 layer wait behavior 和 attention gating strategy。
- 添加 per-layer completion result。
- 添加 equivalence test。

### Phase 6: Capability 允许时迁移 GVA Layerwise Executor

- 仅当 Phase 0 证明 child-owned 或 imported GVA tracker support 后启用。
- 将 GVA/address/size construction 移入 subprocess。
- 实现 child-owned GVA allocation 或受支持 import path。
- 保持 batch copy、write finish 和 lease release behavior。
- 添加 partial failure cleanup behavior。
- 添加 equivalence test。

如果 capability 不可用，GVA 保持 thread fallback，并把它记录为相对原始范围的 known limitation。

### Phase 7: Performance Prototype 和 Report

- 运行 baseline thread mode。
- 对每个 supported mode/backend 运行 subprocess mode。
- 测量 key-building CPU share 和 IPC overhead。
- 测量 fence strategy overhead。
- 记录 regression 和 bottleneck。
- 决定 default enablement 前是否需要额外 shared-memory optimization。

### Phase 8: Rollout

- 从 opt-in subprocess mode 开始。
- 对通过测试的 backend/mode 组合启用 `auto` mode。
- 保留 thread fallback。
- 在 production stability 被证明前不要删除 thread engine。

## 17. 原型对比报告模板

原型报告应包含：

- Test environment。
- Model and workload。
- Backend。
- Transfer mode。
- Backend/mode capability result。
- Buffer sharing strategy。
- NPU fence strategy。
- Batch size and sequence length。
- Thread-mode throughput and latency。
- Subprocess-mode throughput and latency。
- Worker CPU share before/after。
- Transfer child CPU share。
- Key-building CPU share before/after。
- Address/size construction time before/after。
- IPC round-trip latency。
- IPC payload size distribution。
- Serialization/deserialization time。
- Backend operation timings。
- Fence wait timing。
- Crash/restart behavior。
- Fallback behavior。
- Known limitations。

Decision section：

- Keep subprocess disabled。
- Enable for one mode/backend。
- Enable auto mode with fallback。
- Require further shared-memory optimization。
- Require backend work before GVA can be supported。

## 18. 兼容性策略

兼容性应通过构造保证：

- Thread engine 保留为 reference。
- Subprocess mode 尽可能使用相同 metadata conversion 和 helper function。
- Key format 不改变。
- Backend call 保持现有 argument ordering。
- Worker-facing connector method 保持相同语义。
- Existing tests continue to pass。
- Unsupported capability result 选择 thread fallback，而不是部分 subprocess 行为。

任何行为变化都应显式隔离，并用测试覆盖。

## 19. 开放技术问题

这些问题必须在 Phase 0 或启用受影响模式前解决：

- 每个 backend 能否在 child process 中安全 import/register KV buffer？
- 现有 backend handle 是否 process-shareable，还是必须由 child independent register？
- `torch.npu.Event` 语义能否通过 cross-process fence handle 安全表示？
- 如果不能，worker pre-synchronization 在正确性和性能上是否可接受？
- memcache GVA 的 `batch_alloc` 和 `batch_copy` 能否都在 child 中针对 worker-owned KV buffer 运行？
- memcache 能否 export/import GVA tracker state，还是它严格 process-local？
- 哪些 metadata array 大到需要立即使用 shared memory？
- 子进程 crash 后哪些 transfer operation 可以安全 replay？
- Child death 后 partial GVA lease cleanup 应如何处理？

## 20. 推荐首个 PR 形态

首个 PR 应避免一次性改动所有 transfer logic。

推荐内容：

- 添加 `TransferEngine` abstraction。
- 添加包装当前逻辑的 `ThreadTransferEngine`。
- 添加 subprocess IPC dataclass 和 process manager。
- 添加 capability probe skeleton。
- 添加 disabled config 后面的 subprocess engine。
- 添加 lifecycle test。
- 添加 fallback test。
- 添加 non-replayable command classification test。
- 不改变默认行为。

这会创建后续按 mode 迁移所需的安全扩展点。

## 21. 最终设计总结

重构应把 AscendStore transfer 变成 pluggable engine system。现有线程实现成为 compatibility 和 fallback engine。新的 subprocess implementation 在 backend 和 runtime capability 允许时，通过 child process 和显式 IPC protocol 拥有 transfer execution。

worker 进程继续负责高层 orchestration 和 scheduler-visible lifecycle state。child process 拥有 GIL-heavy 工作：key construction、address/size scheduling、layerwise transfer execution 和 backend I/O submission。

设计不能把 GVA/memcache 跨进程行为或 NPU event ordering 当作次要实现细节。它们是核心正确性约束。Subprocess mode 只应为通过 capability probing、equivalence test 和 performance validation 的 backend/mode 组合启用。Unsupported 组合必须干净 fallback 到现有 thread engine。