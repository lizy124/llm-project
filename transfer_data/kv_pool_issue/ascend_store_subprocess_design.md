# AscendStore KV Transfer Subprocess Refactor - Detailed Design

## 1. Design Goals

The goal is to move AscendStore KV transfer execution from worker-process daemon threads into an independent transfer subprocess, while preserving existing behavior for all transfer modes.

The subprocess should own the Python-heavy transfer hot paths:

- Key construction.
- Address and size list construction.
- Layerwise task scheduling.
- Backend `put`, `get`, `batch_copy`, `batch_write_finish`, and `batch_remove_lease` submission when backend/runtime capabilities allow it.

The worker process should keep connector-facing orchestration:

- Build high-level request metadata.
- Maintain scheduler-facing request lifecycle state.
- Submit transfer commands.
- Wait for completion or failure.
- Handle restart or fallback.

Required properties:

- Functional equivalence for non-layerwise, key layerwise, and GVA layerwise modes.
- No precision regression for greedy or non-greedy generation.
- Transfer subprocess crash must not crash the worker Python process.
- Restart or downgrade to in-process thread mode must be supported.
- Performance measurement must distinguish GIL relief from IPC overhead.

The design must treat cross-process KV buffer sharing and NPU stream/event ordering as first-class requirements, because they are part of the original task and test acceptance surface.

## 2. Non-Goals

This refactor should not change:

- KV key string format.
- Block hash semantics.
- Scheduler protocol.
- Backend storage semantics.
- Model execution behavior.
- Cache-hit or failed-block semantics.
- Layerwise visible ordering.
- GVA allocation, publication, or lease semantics.

The existing in-process thread implementation must not be deleted in the first version. It becomes the fallback engine and the reference behavior for equivalence tests.

## 3. Code-Backed Constraints

### 3.1 Existing Objects Are Not IPC Payloads

`ReqMeta`, `LayerTransferTask`, and `LayerLoadTask` are process-local execution objects, not wire objects.

Relevant fields include:

- `ReqMeta.current_event: torch.npu.Event | None`.
- Numpy arrays for block ids, GVA data, and load GVA data.
- Generated save/load keys.
- Partial GVA state.
- `LayerBlockRange.request`, which embeds a full `ReqMeta`.
- `LayerLoadTask.attention_start_gate`, which is process-local synchronization.

Therefore subprocess mode must use dedicated payload dataclasses and field-level conversion.

### 3.2 NPU Event Semantics Are Correctness-Critical

Current behavior depends on NPU event synchronization:

- Non-layerwise save records `current_event` in the worker and synchronizes it in the send thread before backend `put`.
- Layerwise save records `sync_save_events[layer_id]` before submitting layer save.
- GVA load waits for previous save-layer CPU completion and then synchronizes `sync_save_events[wait_for_save]` before reusing HBM buffers.
- Layerwise prefetch load may wait for `AttentionComputeStartGate`.

IPC completion messages are not automatically equivalent to NPU stream fences. The design must provide a fence strategy per mode.

### 3.3 GVA/Memcache Has a Hard Cross-Process Constraint

GVA layerwise is currently selected for memcache backend. The current scheduler code states that memcache requires `batch_alloc` and `batch_copy` to run in the same process because the `gvaBlobTracker` consulted by `batch_copy` is per-process.

This means GVA subprocess support is gated by backend capability. A design that blindly moves GVA `batch_copy` to a child process is not valid unless the child also owns compatible GVA allocation/tracking for the same worker buffers, or the backend exposes supported cross-process import/export semantics.

## 4. Proposed Architecture

Introduce a transfer engine abstraction under the AscendStore KV pool implementation.

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

The worker should not directly manage `KVTransferThread` instances after phase 1. It should call a transfer engine with methods matching the existing worker lifecycle and scheduler-facing semantics.

Suggested interface shape:

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

Important differences from a simplified interface:

- `get_finished()` must receive the same context currently used by `KVPoolWorker.get_finished()`: `finished_req_ids`, `preempted_req_ids`, `delayed_free_req_ids`, and `loading_req_ids`.
- `wait_for_current_layer_load()` should model the existing `current_layer` state machine, not the connector's unused `layer_name` argument.
- Failed blocks, KV events, and completed event ids are engine outputs and must remain scheduler-visible.

### 4.2 ThreadTransferEngine

`ThreadTransferEngine` wraps the existing behavior:

- It creates the same mode-specific send/recv thread classes.
- It keeps using `queue.Queue`, `threading.Event`, and `torch.npu.Event` internally.
- It exposes the `TransferEngine` interface to `KVPoolWorker`.

This engine is required for:

- Fallback.
- Equivalence testing.
- Phased migration without changing default behavior.

### 4.3 SubprocessTransferEngine

`SubprocessTransferEngine` owns:

- Child process startup.
- IPC channel creation.
- Command serialization.
- Result handling.
- Heartbeat/liveness checks.
- Backend and event-fence capability detection.
- Restart/fallback policy.
- Conversion between worker calls and transfer commands.

It should not implement mode-specific transfer logic directly. It dispatches commands to an executor inside the child process.

### 4.4 Transfer Subprocess

The child process initializes:

- Backend instance.
- Device context.
- Imported or child-registered KV buffer descriptors.
- Mode-specific executor.
- Child-local key/token caches.
- Request/result loop.

The child process must not import or call worker forward logic. It should only operate on transfer metadata and backend buffers.

## 5. Configuration and Capability Gating

Add a conservative configuration switch.

Suggested engine mode values:

- `thread`: force current thread engine.
- `subprocess`: require subprocess for eligible modes; raise if unsupported unless fallback policy allows downgrade.
- `auto`: try subprocess for supported backend/mode combinations and fallback to thread mode on unsupported setup.

Suggested fallback policy values:

- `disabled`: subprocess failure raises to worker.
- `restart`: restart child process within retry limits, without thread fallback.
- `thread_fallback`: downgrade to thread engine after subprocess failure.
- `auto`: restart once when safe, then downgrade.

Suggested limits:

- `max_restart_attempts`: default 1.
- `heartbeat_interval_ms`.
- `ipc_timeout_ms`.
- `startup_timeout_ms`.

Initial rollout should default to `thread`. `auto` should be enabled only after capability checks and equivalence/performance tests pass for a backend/mode pair.

### 5.1 Capability Matrix

Subprocess mode must be selected per backend and transfer mode.

Required capabilities:

| Mode | Required capability |
| --- | --- |
| Non-layerwise save | Child can access KV buffers; child can honor save fence before backend `put`. |
| Non-layerwise load | Child can access KV buffers; failed-block reporting can round-trip. |
| Key layerwise save | Child can access KV buffers; child can honor per-layer save fence. |
| Key layerwise load | Child can access KV buffers; child can preserve save-before-load and attention-start gating. |
| GVA layerwise save | Child can own/import GVA allocation/tracker state; `batch_alloc`, `batch_copy`, and `batch_write_finish` are safe in child. |
| GVA layerwise load | Child can own/import GVA/lease state; `batch_copy` and final `batch_remove_lease` cleanup are safe in child. |

If a capability is missing, the engine must choose thread mode for that backend/mode and log one clear warning.

## 6. IPC Protocol

### 6.1 Transport

Use a local multiprocessing transport suitable for Python control messages.

Acceptable first prototype options:

- `multiprocessing.Queue` for separate command and result queues.
- `multiprocessing.Pipe` for simple bidirectional command/result exchange.
- `multiprocessing.connection` if authenticated local sockets are preferred later.

For the first implementation, command/result queues are easiest to test. Large arrays and KV buffers must not be sent through these queues unless explicitly measured and accepted for prototype scope.

The implementation must define the process start method explicitly and verify it with Ascend/NPU runtime initialization. A child entrypoint should live in an importable module so spawn-based startup can work.

### 6.2 Message Envelope

All IPC messages should use a versioned envelope.

```python
@dataclass
class TransferMessage:
    version: int
    message_id: int
    command: str
    payload: object
```

All responses must include the originating `message_id`.

```python
@dataclass
class TransferResult:
    version: int
    message_id: int
    status: str
    payload: object | None
    error: TransferError | None
```

Status values:

- `ok`
- `accepted`
- `failed_recoverable`
- `failed_non_replayable`
- `failed_fatal`
- `unsupported`
- `shutdown_ack`

### 6.3 Commands

#### HELLO

Purpose:

- Negotiate protocol version and child runtime information.

Result:

- Protocol version, child pid, supported command set.

#### PROBE_CAPABILITIES

Purpose:

- Determine whether the selected backend/mode can run in subprocess mode.

Payload:

- Backend name and configuration.
- Transfer mode.
- Device identity.
- Required capability list.

Result:

- Capability result per required item.
- Unsupported reason if any.

#### REGISTER

Purpose:

- Initialize backend and register/import KV cache buffers in the subprocess.

Payload:

- Backend name and configuration.
- Worker rank / TP rank / PP rank / PCP rank / DCP rank / device identity.
- Transfer mode.
- KV cache group metadata.
- Buffer descriptors or child-registration information.
- Layerwise layout information.

Result:

- Backend ready.
- Buffer registration/import result.
- Unsupported process sharing reason if registration cannot work.

#### START

Purpose:

- Tell child process to enter ready state after registration.

Result:

- Ready ack.

#### START_LOAD

Purpose:

- Submit a load step for non-layerwise or layerwise mode.

Payload:

- `StartLoadPayload` with request payloads and current layerwise scheduling context.

Result:

- Accepted ack, not necessarily load-complete.

#### WAIT_CURRENT_LAYER_LOAD

Purpose:

- Preserve current worker-facing `wait_for_layer_load()` semantics.

Payload:

- Current layer index.
- Prefetch/dependency state.
- Wait-for-save layer if any.
- Attention-start fence descriptor if required.

Result:

- Current layer load completed.
- Failed block ids if any.
- Error if layer failed.

#### SAVE_CURRENT_LAYER

Purpose:

- Submit one layer save batch and preserve current `save_kv_layer()` behavior.

Payload:

- Current layer index.
- Layer transfer task payloads.
- Save fence descriptor.
- Final-layer behavior flags.

Result:

- Accepted or completed according to current semantics.
- Finished request ids if final-layer completion occurred.

#### SUBMIT_SAVE_BATCH

Purpose:

- Submit non-layerwise save requests.

Payload:

- Stored request payloads.
- Save fence descriptor.
- Request ids.

Result:

- Accepted ack.

#### WAIT_SAVE_BATCH

Purpose:

- Preserve current blocking non-layerwise save behavior.

Result:

- Save complete.
- KV events produced.
- Completed event ids.
- Finished request ids.

#### GET_FINISHED

Purpose:

- Return finished sending/receiving request ids with current scheduler filtering semantics.

Payload:

- Finished request ids known by scheduler.
- Preempted request ids.
- Delayed-free request ids.
- Loading request ids.
- Layerwise flag.
- Load-async flag.

Result:

- Tuple-equivalent payload for saved and loaded completions.

#### GET_FAILED_BLOCKS

Purpose:

- Return and clear failed load block ids.

Result:

- Failed block id set.

#### TAKE_EVENTS

Purpose:

- Return and clear KV cache events and completed worker events.

Result:

- Serialized KV events.
- Completed event ids.

#### HEARTBEAT

Purpose:

- Liveness check.

Result:

- Current child status, active command id, transfer mode, backend status.

#### SHUTDOWN

Purpose:

- Graceful subprocess exit.

Result:

- Shutdown ack.

### 6.4 Error Payload

Errors should be structured.

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

Worker logs should include command, request id, layer id, backend/mode, and fallback decision.

## 7. Payload Design

### 7.1 Request Payload

Do not send full `ReqMeta`. Use a field-level payload.

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

Large numpy arrays should not be added to this payload by default. They should use shared memory descriptors or child-local reconstruction if needed.

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

Do not send `LayerBlockRange` with embedded request references.

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

`cached_process_tokens` should be built and cached inside the child process when subprocess mode is active, rather than serialized from worker to child.

### 7.4 Layer Load Payload

```python
@dataclass
class LayerLoadTaskPayload:
    wait_for_save_layer: int | None
    transfer_tasks: list[LayerTransferTaskPayload]
    layer_id: int
    attention_start_fence: FenceDescriptor | None
```

### 7.5 Runtime-Only Field Replacements

| Runtime field | Replacement |
| --- | --- |
| `queue.Queue` | IPC command queue/result queue. |
| `threading.Event` load/save completion | Message id plus child result; worker-side condition/event if needed. |
| `torch.npu.Event` | `FenceDescriptor`, or worker pre-synchronization when no cross-process fence exists. |
| `AttentionComputeStartGate` | Explicit gate command/fence descriptor, or worker-held gate before child submit. |
| Backend instance | Child-local backend instance. |
| Worker task list references | Immutable payload snapshot per command. |

## 8. Cross-Process Buffer and Fence Design

### 8.1 KV Cache Buffers

The design must avoid copying KV cache tensors through IPC.

Preferred strategy:

1. Worker computes KV cache memory regions as today.
2. Worker creates `BufferDescriptor` values for each region.
3. Child imports/registers those descriptors with its backend.
4. Child uses descriptors and cache layout metadata to build addresses and submit I/O.

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

If a backend accepts raw virtual addresses only inside the registering process, raw `base_addr` is not a valid subprocess descriptor. Capability probing must detect this and force thread fallback.

### 8.2 NPU Fence Strategy

Use one of the following strategies per backend/mode, in this order of preference:

1. Cross-process NPU event/fence handle.
   - Worker records an event and sends a `FenceDescriptor` to the child.
   - Child waits on the fence before backend I/O.
   - This best preserves current asynchronous behavior.

2. Worker pre-synchronization before IPC.
   - Worker records and synchronizes the NPU event before sending a command.
   - Child can safely submit backend I/O.
   - Correctness is simpler, but performance benefit may shrink and must be measured.

3. Thread fallback.
   - If neither cross-process fence nor acceptable pre-sync behavior is supported, keep that mode on `ThreadTransferEngine`.

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

GVA subprocess support requires one of these supported designs:

1. Child-owned GVA path.
   - Child performs `batch_alloc`, owns child-local GVA tracker state, performs `batch_copy`, calls `batch_write_finish`, and releases leases.
   - Worker sends only keys, sizes, request/layer metadata, and KV buffer descriptors.
   - Worker no longer owns `_allocated_gvas` for subprocess mode, or it mirrors child state only for observability.

2. Backend-supported tracker import/export.
   - Worker can export allocation/tracker descriptors.
   - Child imports them and can safely call `batch_copy`.

3. Thread fallback.
   - If memcache keeps `gvaBlobTracker` strictly per-process with no child-owned full path, GVA layerwise must remain on thread engine.

The implementation must not claim GVA subprocess support until one of the first two paths is proven by prototype and test.

## 9. Executor Design Inside Subprocess

The subprocess dispatches to a mode-specific executor behind a shared interface.

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

Concrete executors:

- `NonLayerwiseTransferExecutor`.
- `KeyLayerwiseTransferExecutor`.
- `GVALayerwiseTransferExecutor`, enabled only when GVA capability probe passes.

Current transfer thread logic should be extracted into reusable helper functions where possible. Avoid subclassing `threading.Thread` logic directly in the child; extract pure transfer routines for key/address/size construction and backend calls.

## 10. Mode-Specific Design

### 10.1 Non-Layerwise Executor

Responsibilities:

- Receive save/load payloads.
- Build key strings in subprocess.
- Build addrs and sizes in subprocess.
- Preserve TP-rank ordering/rotation.
- Preserve store mask, skip range, and skip-null-block logic.
- Call backend `exists`, `put`, and `get`.
- Return KV events, completed event ids, finished request ids, and failed block ids.

Important equivalence requirements:

- Preserve key string format.
- Preserve TP-rank ordering/rotation.
- Preserve store mask and skip range logic.
- Preserve failed block recording behavior.
- Cover synchronous and async load paths. If only async load is moved, synchronous non-layerwise load remains a GIL-sharing path.

### 10.2 Key Layerwise Executor

Responsibilities:

- Maintain cached process tokens in subprocess.
- Build layer-specific keys in subprocess.
- Preserve per-layer save/load ordering.
- Honor save fences before backend `put`.
- Preserve `consumer_is_to_put` wait behavior.
- Return per-layer completion.

Important equivalence requirements:

- Preserve `LayerPoolKey` formatting.
- Preserve layer index mapping.
- Preserve current visible behavior of `wait_for_layer_load()` and `save_kv_layer()`.
- Preserve final-layer finished request behavior.

### 10.3 GVA Layerwise Executor

Responsibilities when capability probe passes:

- Own or import GVA allocation/tracker state safely.
- Build shared block data in subprocess or import worker-built descriptors.
- Build GVA/address/size arrays.
- Honor save/load fences.
- Call batch copy with correct direction.
- Call `batch_write_finish` for save.
- Call `batch_remove_lease` on final load layer.
- Handle cleanup on partial failure.

Important equivalence requirements:

- Preserve batch copy limits.
- Preserve H2D stagger behavior.
- Preserve lease release semantics.
- Preserve final-layer cleanup.
- Preserve non-idempotent `batch_alloc` behavior.

If capability probe fails, this executor must not be enabled. The engine must fall back to thread mode and report the unsupported reason.

## 11. Worker-Side Lifecycle

### 11.1 Startup

Worker flow after refactor:

1. `KVPoolWorker.register_kv_caches()` prepares local metadata and memory regions as today.
2. Worker constructs selected `TransferEngine`.
3. Engine starts subprocess if configured.
4. Engine sends `HELLO` and `PROBE_CAPABILITIES`.
5. If supported, engine sends `REGISTER` with backend, layout, buffer descriptors, and mode configuration.
6. Child initializes backend and imports/registers buffers.
7. Child sends ready ack.
8. Worker continues normal operation.

If startup fails:

- In `thread` mode: raise as current behavior would.
- In `subprocess` mode with no fallback: raise structured error.
- In `auto` or fallback-enabled mode: log one warning and start `ThreadTransferEngine`.

### 11.2 Normal Operation

Worker-facing methods remain close to existing behavior:

- `start_load_kv()` builds request payloads and submits load command.
- `wait_for_layer_load()` uses current-layer state and blocks until the engine reports completion.
- `save_kv_layer()` records or creates required fence state and submits layer save.
- `wait_for_save()` submits non-layerwise save and waits for completion.
- `get_finished()` passes scheduler filtering context to the engine.
- `get_block_ids_with_load_errors()` returns child-reported failed blocks.
- KV events are collected from engine output.

The worker should not directly access child internal queues, child events, or child executor state.

### 11.3 Shutdown

Shutdown must be idempotent:

1. Send `SHUTDOWN` if subprocess is running.
2. Wait for `shutdown_ack` until timeout.
3. If timeout, terminate child process.
4. Clean local IPC handles.
5. If downgraded, shut down thread engine normally.

## 12. Crash Recovery and Fallback

### 12.1 Crash Detection

Detect child failure by:

- Process exit code.
- IPC EOF / broken pipe.
- Heartbeat timeout.
- Command timeout.

The manager should mark the engine unhealthy immediately after detection.

### 12.2 Command Replay Policy

Commands must be classified by replay safety.

Replay-safe or usually safe:

- `HELLO`.
- `PROBE_CAPABILITIES`.
- `REGISTER` before transfer begins, if backend registration is idempotent.
- `HEARTBEAT`.
- Read-only metadata queries.

Replay-unsafe unless backend confirms otherwise:

- `SUBMIT_SAVE_BATCH` after backend `put` may have started.
- `SAVE_CURRENT_LAYER` after backend `put`, `batch_copy`, or `batch_write_finish` may have started.
- GVA `batch_alloc` because current worker state treats it as non-idempotent.
- GVA `batch_write_finish`.
- Final-layer `batch_remove_lease`.

Default policy:

- Do not blindly replay non-idempotent writes.
- Fail the current in-flight transfer when its commit state is unknown.
- Downgrade to thread mode for future transfers if fallback is enabled.

### 12.3 Restart Policy

Restart is allowed when:

- Child died while idle.
- Child died before a command was accepted.
- The active command is replay-safe.
- Backend/buffer registration can be repeated.
- Restart attempts remain under configured limit.

Restart flow:

1. Stop or reap old process.
2. Start new process.
3. Re-run `HELLO`, `PROBE_CAPABILITIES`, and `REGISTER`.
4. Rebuild subprocess-local caches as empty.
5. Resume accepting new requests.
6. Fail or retry the in-flight request according to command replay safety.

### 12.4 Downgrade Policy

Downgrade to thread mode when:

- Subprocess startup fails.
- Buffer sharing is unsupported.
- NPU fence semantics are unsupported for the selected mode.
- GVA tracker/import support is unsupported.
- Restart limit is exceeded.
- IPC protocol failure is fatal.
- Backend cannot initialize in the child process.

Downgrade flow:

1. Mark subprocess engine disabled for the current backend/mode.
2. Shut down any child resources.
3. Create `ThreadTransferEngine` using existing registered worker state.
4. Start existing thread-mode transfer workers.
5. Continue future requests through thread mode.

Downgrade should emit one clear warning, not repeated noisy logs on every request.

### 12.5 In-Flight Requests

In-flight request handling must be conservative:

- If a load was in-flight and child died, treat load as failed and record failed blocks where possible.
- If a save was in-flight and idempotency is uncertain, do not assume it succeeded.
- If backend can confirm committed keys, use backend state to complete or skip retry.
- Otherwise fail the transfer task and continue fallback for future tasks.
- For GVA leases, run best-effort cleanup only through documented backend-safe APIs.

## 13. Precision Preservation

Precision preservation depends on equivalent transfer ordering, not merely successful backend calls.

Guardrails:

- Keep key format unchanged.
- Keep block id and hash mapping unchanged.
- Keep TP-rank ordering unchanged.
- Keep layer load wait semantics unchanged.
- Keep attention-start gating unchanged.
- Keep NPU save/load fence semantics unchanged.
- Keep failed-load behavior unchanged.
- Keep final-layer lease release unchanged.
- Keep scheduler-visible finished request reporting unchanged.

Regression tests should compare outputs for:

- Greedy generation.
- Non-greedy generation with fixed seed.
- Cache hit path.
- Cache miss / partial miss path.
- Each of the three transfer modes.

## 14. Observability

Add metrics and logs that answer whether subprocess mode is helping and whether fallback is happening.

Suggested counters/timers:

- `kv_transfer_engine_mode`.
- `kv_transfer_backend_mode_supported`.
- `kv_transfer_subprocess_start_total`.
- `kv_transfer_subprocess_restart_total`.
- `kv_transfer_subprocess_fallback_total`.
- `kv_transfer_ipc_request_total`.
- `kv_transfer_ipc_error_total`.
- `kv_transfer_ipc_roundtrip_ms`.
- `kv_transfer_ipc_payload_bytes`.
- `kv_transfer_ipc_serialize_ms`.
- `kv_transfer_ipc_deserialize_ms`.
- `kv_transfer_key_build_ms`.
- `kv_transfer_addr_size_build_ms`.
- `kv_transfer_backend_put_ms`.
- `kv_transfer_backend_get_ms`.
- `kv_transfer_backend_batch_copy_ms`.
- `kv_transfer_fence_wait_ms`.

Logs should include:

- Engine selected.
- Subprocess pid.
- Backend registration success/failure.
- Capability probe results.
- Restart reason.
- Downgrade reason.
- Fatal command failure with command id and request id.

## 15. Test Plan

### 15.1 Unit Tests

Add tests for `TransferProcessManager`:

- Starts subprocess and receives ready ack.
- Sends shutdown and exits cleanly.
- Detects child process exit.
- Detects IPC disconnect.
- Converts child exception into structured error.
- Restarts within configured limit for replay-safe command.
- Refuses replay for non-replayable command.
- Downgrades after restart limit.

Add tests for payload conversion:

- `ReqMeta` to `ReqMetaPayload`.
- `LayerTransferTask` to `LayerTransferTaskPayload`.
- `LayerLoadTask` to `LayerLoadTaskPayload`.
- Runtime-only fields are excluded.
- Required fields are preserved.
- Large numpy arrays use descriptor path or are rejected when unsupported.

Add tests for `SubprocessTransferEngine`:

- Register/start lifecycle.
- Submit load command.
- Submit save command.
- Wait save command.
- Wait current layer load command.
- Get finished command with preempted/delayed/loading filtering.
- Failed block propagation.
- KV event propagation.
- Fallback to `ThreadTransferEngine` on startup failure.
- Fallback on unsupported capability result.

Add executor tests with mocked backend:

- Non-layerwise executor calls backend `put/get` with expected keys/addrs/sizes.
- Key layerwise executor preserves layer key format.
- GVA layerwise executor is disabled when capability probe fails.
- GVA layerwise executor calls batch copy, write finish, and final lease release when capability probe passes.
- Backend exception returns recoverable/fatal/non-replayable status as configured.

### 15.2 Integration Tests

Where device/backend test infrastructure is available:

- Run one request through non-layerwise save/load.
- Run one request through key layerwise save/load.
- Run one request through GVA layerwise save/load only when capability probe passes.
- Kill transfer subprocess during idle and verify restart.
- Kill transfer subprocess during request and verify worker survives.
- Force buffer registration failure and verify thread fallback.
- Force IPC disconnect and verify structured failure/fallback.
- Verify cross-process buffer sharing with real backend registration/import path.
- Verify NPU fence strategy preserves ordering.

### 15.3 Precision Tests

Use deterministic workloads:

- Greedy decode baseline vs subprocess mode.
- Non-greedy fixed-seed baseline vs subprocess mode.
- Partial KV load failure path.
- Multi-layer transfer path.
- GVA path when capability probe passes.

Expected result:

- Token outputs match baseline for deterministic tests.
- Cache hit/miss reporting matches baseline.
- Finished request reporting matches baseline.

### 15.4 Performance Tests

Run before/after comparisons under the same configuration:

- Existing thread mode.
- Subprocess mode.
- Subprocess mode with forced fallback excluded from steady-state metrics.
- Worker pre-sync fence strategy, if used.
- Cross-process fence strategy, if supported.

Collect:

- Throughput.
- P50/P95/P99 latency.
- Worker CPU profile.
- Transfer process CPU profile.
- Key-building CPU share.
- Address/size construction time.
- IPC overhead.
- Backend operation time.
- Fence wait time.

## 16. Implementation Plan

### Phase 0: Feasibility Spikes

- Probe backend buffer sharing/import support in child process.
- Probe NPU event/fence strategy.
- Probe memcache GVA child-owned allocation/copy path.
- Document supported backend/mode matrix.

Exit criteria:

- Clear capability matrix.
- Unsupported modes fall back deterministically.
- No claim of GVA subprocess support unless proven.

### Phase 1: Introduce TransferEngine Boundary

- Add `TransferEngine` interface.
- Move current thread startup logic into `ThreadTransferEngine`.
- Make `KVPoolWorker` call engine methods instead of directly owning mode-specific transfer threads.
- Preserve current behavior.
- Add tests proving thread engine preserves existing behavior.

This phase should be low risk because default behavior remains thread mode.

### Phase 2: Add IPC Types and Process Manager

- Add versioned command/result dataclasses.
- Add payload conversion helpers.
- Add `TransferProcessManager`.
- Add child process main loop.
- Add mock executor for lifecycle tests.
- Test startup, shutdown, exception, timeout, disconnect, restart, and non-replayable failure handling.

This phase proves process isolation before moving real transfer logic.

### Phase 3: Add Subprocess Engine With Mocked Backend

- Implement `SubprocessTransferEngine`.
- Wire it into `KVPoolWorker` behind config.
- Support probe/register/start/shutdown and simple commands.
- Add fallback to thread engine.
- Keep subprocess mode off by default.

This phase validates worker integration.

### Phase 4: Migrate Non-Layerwise Executor

- Move non-layerwise key/address/size construction into subprocess executor.
- Cover both async load and synchronous load behavior.
- Reuse existing helper functions where possible.
- Preserve backend calls and result handling.
- Add equivalence tests against thread engine.
- Add IPC payload size metrics.

### Phase 5: Migrate Key Layerwise Executor

- Move layer key construction into subprocess.
- Maintain subprocess-local cached process tokens.
- Preserve layer wait behavior and attention gating strategy.
- Add per-layer completion results.
- Add equivalence tests.

### Phase 6: Migrate GVA Layerwise Executor When Capability Allows

- Enable only if Phase 0 proves child-owned or imported GVA tracker support.
- Move GVA/address/size construction into subprocess.
- Implement child-owned GVA allocation or supported import path.
- Preserve batch copy, write finish, and lease release behavior.
- Add cleanup behavior for partial failures.
- Add equivalence tests.

If capability is not available, keep GVA on thread fallback and document this as a known limitation against the original scope.

### Phase 7: Performance Prototype and Report

- Run baseline thread mode.
- Run subprocess mode for each supported mode/backend.
- Measure key-building CPU share and IPC overhead.
- Measure fence strategy overhead.
- Document regressions and bottlenecks.
- Decide whether additional shared-memory optimization is required before default enablement.

### Phase 8: Rollout

- Start with opt-in subprocess mode.
- Enable `auto` mode for backend/mode combinations that pass tests.
- Keep thread fallback available.
- Do not remove thread engine until production stability is proven.

## 17. Prototype Comparison Report Template

The prototype report should include:

- Test environment.
- Model and workload.
- Backend.
- Transfer mode.
- Backend/mode capability result.
- Buffer sharing strategy.
- NPU fence strategy.
- Batch size and sequence length.
- Thread-mode throughput and latency.
- Subprocess-mode throughput and latency.
- Worker CPU share before/after.
- Transfer child CPU share.
- Key-building CPU share before/after.
- Address/size construction time before/after.
- IPC round-trip latency.
- IPC payload size distribution.
- Serialization/deserialization time.
- Backend operation timings.
- Fence wait timing.
- Crash/restart behavior.
- Fallback behavior.
- Known limitations.

Decision section:

- Keep subprocess disabled.
- Enable for one mode/backend.
- Enable auto mode with fallback.
- Require further shared-memory optimization.
- Require backend work before GVA can be supported.

## 18. Compatibility Strategy

Compatibility should be maintained by construction:

- Thread engine remains the reference.
- Subprocess mode uses the same metadata conversion and helper functions where possible.
- Key format does not change.
- Backend calls preserve existing argument ordering.
- Worker-facing connector methods keep the same semantics.
- Existing tests continue to pass.
- Unsupported capability results select thread fallback rather than partial subprocess behavior.

Any behavior change should be explicitly isolated and covered by tests.

## 19. Open Technical Questions

These must be resolved during Phase 0 or before enabling subprocess for the affected mode:

- Can each backend import/register KV buffers from a child process safely?
- Are existing backend handles process-shareable, or must the child register independently?
- Can `torch.npu.Event` semantics be represented safely through cross-process fence handles?
- If not, is worker pre-synchronization acceptable for correctness and performance?
- Can memcache GVA `batch_alloc` and `batch_copy` both run in the child while operating on worker-owned KV buffers?
- Can memcache export/import the GVA tracker state, or is it strictly process-local?
- Which metadata arrays are large enough to require shared memory immediately?
- Which transfer operations are safe to replay after subprocess crash?
- How should partial GVA lease cleanup be handled after child death?

## 20. Recommended First PR Shape

The first PR should avoid changing all transfer logic at once.

Recommended contents:

- Add `TransferEngine` abstraction.
- Add `ThreadTransferEngine` wrapper around current logic.
- Add subprocess IPC dataclasses and process manager.
- Add capability probe skeleton.
- Add subprocess engine behind disabled config.
- Add lifecycle tests.
- Add fallback tests.
- Add non-replayable command classification tests.
- No default behavior change.

This creates the safe extension point needed for later mode-by-mode migration.

## 21. Final Design Summary

The refactor should turn AscendStore transfer into a pluggable engine system. The existing thread implementation becomes the compatibility and fallback engine. The new subprocess implementation owns transfer execution through a child process and explicit IPC protocol when backend and runtime capabilities allow it.

The worker process remains responsible for high-level orchestration and scheduler-visible lifecycle state. The child process owns GIL-heavy work: key construction, address/size scheduling, layerwise transfer execution, and backend I/O submission.

The design must not treat GVA/memcache cross-process behavior or NPU event ordering as minor implementation details. Those are core correctness constraints. Subprocess mode should be enabled only for backend/mode pairs that pass capability probing, equivalence tests, and performance validation. Unsupported pairs must fall back cleanly to the existing thread engine.