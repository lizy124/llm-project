# AscendStore KV Transfer Subprocess Refactor - Detailed Design

## 1. Design Goals

The goal is to move AscendStore KV transfer execution from worker-process threads into an independent transfer subprocess, while preserving existing behavior for all transfer modes.

The subprocess must own the Python-heavy transfer hot paths:

- Key construction.
- Address and size list construction.
- Layerwise task scheduling.
- Backend `put`, `get`, `batch_copy`, `batch_write_finish`, and `batch_remove_lease` submission.

The worker process should keep only connector-facing orchestration:

- Build high-level request metadata.
- Submit transfer commands.
- Wait for completion or failure.
- Handle restart or fallback.

Required properties:

- Functional equivalence for non-layerwise, key layerwise, and GVA layerwise modes.
- No precision regression for greedy or non-greedy generation.
- Transfer subprocess crash must not crash the worker.
- Restart or downgrade to in-process thread mode must be supported.
- Performance measurement must distinguish GIL relief from IPC overhead.

## 2. Non-Goals

This refactor should not change:

- KV key string format.
- Block hash semantics.
- Scheduler protocol.
- Backend storage semantics.
- Model execution behavior.
- Cache-hit or failed-block semantics.
- Layerwise visible ordering.

The existing in-process thread implementation should not be deleted in the first version. It should become the fallback engine and the reference behavior for equivalence tests.

## 3. Proposed Architecture

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

### 3.1 TransferEngine Interface

The worker should not directly manage `KVTransferThread` instances. It should call a transfer engine with methods matching the existing worker lifecycle.

Suggested interface:

```python
class TransferEngine(Protocol):
    def register_kv_caches(self, context: RegisterContext) -> None: ...
    def start(self) -> None: ...
    def shutdown(self) -> None: ...

    def submit_load(self, metadata: LoadCommandPayload) -> None: ...
    def wait_for_layer_load(self, layer_name: str) -> None: ...

    def submit_save_layer(self, payload: SaveLayerCommandPayload) -> None: ...
    def submit_save(self, payload: SaveCommandPayload) -> None: ...
    def wait_for_save(self) -> None: ...

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]: ...
    def raise_if_failed(self) -> None: ...
```

The exact names can follow local style, but the boundary should separate worker orchestration from transfer execution.

### 3.2 ThreadTransferEngine

`ThreadTransferEngine` wraps the existing behavior:

- It creates the same mode-specific send/recv thread classes.
- It keeps using `queue.Queue` and `threading.Event` internally.
- It exposes the new `TransferEngine` interface to `KVPoolWorker`.

This engine is required for fallback and for equivalence tests.

### 3.3 SubprocessTransferEngine

`SubprocessTransferEngine` owns:

- Child process startup.
- IPC channel creation.
- Command serialization.
- Result handling.
- Heartbeat/liveness checks.
- Restart/fallback policy.
- Conversion between worker calls and transfer commands.

It should not implement mode-specific transfer logic directly. It should dispatch commands to an executor inside the child process.

### 3.4 Transfer Subprocess

The child process should initialize:

- Backend instance.
- Device context.
- Buffer registrations.
- Mode-specific executor.
- Request/result loop.

The child process should not import or call worker forward logic. It should only operate on transfer metadata and backend buffers.

## 4. Configuration

Add a configuration switch with conservative defaults.

Suggested values:

- `thread`: force current thread engine.
- `subprocess`: force subprocess engine; raise or fallback depending on policy.
- `auto`: try subprocess and fallback to thread mode on unsupported setup.

Suggested fallback policy values:

- `disabled`: subprocess failure raises to worker.
- `restart`: restart child process within retry limits.
- `thread_fallback`: downgrade to thread engine after restart failure.
- `auto`: restart once, then downgrade.

Suggested limits:

- `max_restart_attempts`: default 1 or 2.
- `heartbeat_interval_ms`.
- `ipc_timeout_ms`.
- `startup_timeout_ms`.

Initial rollout should default to thread mode unless the project wants immediate opt-in testing. Production enablement should start with `auto` only after equivalence and performance tests pass.

## 5. IPC Protocol

### 5.1 Transport

Use a local multiprocessing transport suitable for Python control messages.

Acceptable first prototype options:

- `multiprocessing.Pipe` for simple bidirectional command/result exchange.
- `multiprocessing.Queue` for separate command and result queues.
- `multiprocessing.connection` if authenticated local sockets are preferred later.

For the first implementation, a command queue plus result queue is easiest to test and reason about. Large payloads should not be carried through this queue when avoidable.

### 5.2 Message Envelope

All IPC messages should use a versioned envelope.

```python
@dataclass
class TransferMessage:
    version: int
    message_id: int
    command: str
    payload: object
```

All responses should include the originating `message_id`.

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
- `failed_recoverable`
- `failed_fatal`
- `unsupported`
- `shutdown_ack`

### 5.3 Commands

#### REGISTER

Purpose:

- Initialize backend and register KV cache buffers in the subprocess.

Payload:

- Backend name and configuration.
- Worker rank / tp rank / device identity.
- Transfer mode.
- KV cache group metadata.
- Buffer descriptors or registration information.
- Layerwise layout information.

Result:

- Backend ready.
- Unsupported process sharing reason if registration cannot work.

#### START

Purpose:

- Tell child process to enter ready state after registration.

Result:

- Ready ack.

#### SUBMIT_LOAD

Purpose:

- Submit non-layerwise or layerwise load metadata.

Payload:

- Request metadata.
- Load specs.
- Layerwise shared task metadata when applicable.

Result:

- Accepted ack, not necessarily load-complete.

#### WAIT_LAYER_LOAD

Purpose:

- Preserve current worker-facing `wait_for_layer_load(layer_name)` semantics.

Payload:

- Layer name.
- Layer index.
- Request group context if needed.

Result:

- Layer load completed.
- Failed block ids if any.
- Error if layer failed.

#### SUBMIT_SAVE_LAYER

Purpose:

- Submit one layer save task batch.

Payload:

- Layer name.
- Layer id.
- Layer transfer tasks.
- Event dependency descriptors.

Result:

- Accepted or completed depending on existing semantics.

#### SUBMIT_SAVE

Purpose:

- Submit non-layerwise save requests.

Payload:

- Stored request metadata.
- Current event descriptor.
- Request ids.

Result:

- Accepted ack.

#### WAIT_SAVE

Purpose:

- Preserve current blocking save completion behavior.

Result:

- Save complete.
- KV events produced.
- Finished request ids.

#### GET_FINISHED

Purpose:

- Return finished sending/receiving request ids, matching current connector behavior.

Payload:

- Finished request ids known by scheduler/worker.

Result:

- Tuple-equivalent payload for saved and loaded completions.

#### HEARTBEAT

Purpose:

- Liveness check.

Result:

- Current child status and mode.

#### SHUTDOWN

Purpose:

- Graceful subprocess exit.

Result:

- Shutdown ack.

### 5.4 Error Payload

Errors should be structured.

```python
@dataclass
class TransferError:
    error_type: str
    message: str
    command: str
    request_id: str | None
    layer_id: int | None
    recoverable: bool
    traceback: str | None
```

Worker logs should include command, request id, layer id, and fallback decision.

## 6. Data Sharing Design

### 6.1 Control Metadata

Small metadata should be serialized directly through IPC:

- Request ids.
- Layer ids.
- Group ids.
- Mode flags.
- Backend mode.
- Small lists of ranges.
- Completion statuses.

### 6.2 Request Metadata

`ReqMeta`, `LayerTransferTask`, and `LayerLoadTask` should be converted into subprocess-safe payloads. Avoid sending Python objects that contain non-serializable runtime fields.

Suggested approach:

- Add serializer/converter helpers near the transfer engine boundary.
- Keep raw runtime-only fields inside worker.
- Send only the fields required by transfer execution.

Example payload groups:

- `ReqMetaPayload`
- `LayerTransferTaskPayload`
- `LayerLoadTaskPayload`
- `SharedBlockDataPayload`

### 6.3 Runtime-Only Fields

Some fields should not be sent as-is:

- `threading.Event`
- `queue.Queue`
- Raw `torch.npu.Event` object references.
- Backend object instances.
- Python object references that only make sense in the worker process.

They must be replaced by:

- Task ids.
- Event ids.
- Completion messages.
- Buffer descriptors.
- Backend registration descriptors.

### 6.4 KV Cache Buffers

The design must avoid copying KV cache tensors through IPC.

Preferred strategy:

1. Worker registers KV cache buffers with the backend as before.
2. Worker obtains process-shareable backend descriptors if supported.
3. Subprocess initializes backend and registers/imports those descriptors.
4. Subprocess uses descriptors to build addresses and submit I/O.

If backend descriptors are not process-shareable, fallback to thread mode for that backend/mode.

### 6.5 Numpy Arrays and GVA Metadata

For block ids and GVA arrays:

- Small arrays may be serialized initially for prototype simplicity.
- Large or repeated arrays should use shared memory or subprocess-local cached copies.
- Payloads should include content hashes or generation ids when caching is used.

GVA layerwise mode should prioritize shared descriptors because it can move significant metadata per layer.

## 7. Executor Design Inside Subprocess

The subprocess should dispatch to a mode-specific executor behind a shared interface.

```python
class TransferExecutor(Protocol):
    def register(self, payload: RegisterPayload) -> None: ...
    def submit_load(self, payload: LoadPayload) -> TransferAck: ...
    def wait_layer_load(self, payload: WaitLayerLoadPayload) -> TransferResultPayload: ...
    def submit_save_layer(self, payload: SaveLayerPayload) -> TransferAck: ...
    def submit_save(self, payload: SavePayload) -> TransferAck: ...
    def wait_save(self, payload: WaitSavePayload) -> TransferResultPayload: ...
    def get_finished(self, payload: GetFinishedPayload) -> FinishedPayload: ...
    def shutdown(self) -> None: ...
```

Concrete executors:

- `NonLayerwiseTransferExecutor`
- `KeyLayerwiseTransferExecutor`
- `GVALayerwiseTransferExecutor`

These can initially wrap or reuse logic extracted from the current transfer thread classes.

## 8. Mode-Specific Design

### 8.1 Non-Layerwise Executor

Responsibilities:

- Receive save/load payloads.
- Build key strings in subprocess.
- Build addrs and sizes in subprocess.
- Call backend `lookup`, `put`, and `get`.
- Return KV events and finished request ids.
- Return failed block ids for load failures.

Important equivalence requirements:

- Preserve key string format.
- Preserve TP-rank ordering/rotation.
- Preserve store mask and skip range logic.
- Preserve failed block recording behavior.

### 8.2 Key Layerwise Executor

Responsibilities:

- Maintain cached process tokens in subprocess.
- Build layer-specific keys in subprocess.
- Preserve per-layer save/load ordering.
- Call backend `put/get`.
- Return per-layer completion.

Important equivalence requirements:

- Preserve `LayerPoolKey` formatting.
- Preserve `consumer_is_to_put` wait behavior.
- Preserve layer index mapping.
- Preserve current visible behavior of `wait_for_layer_load` and `save_kv_layer`.

### 8.3 GVA Layerwise Executor

Responsibilities:

- Build shared block data in subprocess or import worker-built shared payloads.
- Build GVA/address/size arrays.
- Call batch copy with correct direction.
- Call `batch_write_finish` for save.
- Call `batch_remove_lease` on final load layer.
- Handle cleanup on partial failure.

Important equivalence requirements:

- Preserve batch copy limits.
- Preserve stagger behavior if currently enabled.
- Preserve lease release semantics.
- Preserve final-layer cleanup.

## 9. Worker-Side Lifecycle

### 9.1 Startup

Worker flow after refactor:

1. `KVPoolWorker.register_kv_caches()` prepares local metadata.
2. Worker constructs selected `TransferEngine`.
3. Engine starts subprocess if configured.
4. Engine sends `REGISTER`.
5. Child initializes backend and imports/registers buffers.
6. Child sends ready ack.
7. Worker continues normal operation.

If startup fails:

- In `thread` mode: raise as current behavior would.
- In `subprocess` mode with no fallback: raise structured error.
- In `auto` or fallback-enabled mode: log warning and start `ThreadTransferEngine`.

### 9.2 Normal Operation

Worker-facing methods remain close to existing behavior:

- `start_load_kv()` submits load command.
- `wait_for_layer_load()` blocks until the child reports layer completion.
- `save_kv_layer()` submits layer save and waits according to current semantics.
- `wait_for_save()` blocks until save completion.
- `get_finished()` reads child-produced completion sets.

The worker should not directly access child internal queues or events.

### 9.3 Shutdown

On connector shutdown or worker teardown:

1. Send `SHUTDOWN`.
2. Wait for `shutdown_ack` until timeout.
3. If timeout, terminate child process.
4. Clean local IPC handles.
5. If already downgraded, shut down thread engine normally.

Shutdown must be idempotent.

## 10. Crash Recovery and Fallback

### 10.1 Crash Detection

Detect child failure by:

- Process exit code.
- IPC EOF / broken pipe.
- Heartbeat timeout.
- Command timeout.

The manager should mark the engine as unhealthy immediately after detection.

### 10.2 Restart Policy

Restart is allowed when:

- Child died outside a non-idempotent critical section.
- Backend/buffer registration can be repeated.
- Restart attempts remain under configured limit.

Restart flow:

1. Stop or reap old process.
2. Start new process.
3. Re-send `REGISTER`.
4. Rebuild subprocess-local caches as empty.
5. Resume accepting new requests.
6. Fail or retry the in-flight request according to command idempotency.

Default policy should not blindly replay non-idempotent writes. For safety, fail the current in-flight transfer and let upper layers continue through fallback unless replay is explicitly safe.

### 10.3 Downgrade Policy

Downgrade to thread mode when:

- Subprocess startup fails.
- Buffer sharing is unsupported.
- Restart limit is exceeded.
- IPC protocol failure is fatal.
- Backend cannot initialize in the child process.

Downgrade flow:

1. Mark subprocess engine disabled.
2. Shut down any child resources.
3. Create `ThreadTransferEngine` using existing registered worker state.
4. Start existing thread-mode transfer workers.
5. Continue future requests through thread mode.

Downgrade should emit one clear warning, not repeated noisy logs on every request.

### 10.4 In-Flight Requests

In-flight request handling must be conservative:

- If a load was in-flight and child died, treat load as failed and record failed blocks where possible.
- If a save was in-flight and idempotency is uncertain, do not assume it succeeded.
- If backend can confirm committed keys, use backend state to complete or skip retry.
- Otherwise fail the transfer task and continue fallback for future tasks.

## 11. Precision Preservation

Precision preservation depends on making transfer behavior equivalent, not merely successful.

Guardrails:

- Keep key format unchanged.
- Keep block id and hash mapping unchanged.
- Keep TP-rank ordering unchanged.
- Keep layer load wait semantics unchanged.
- Keep failed-load behavior unchanged.
- Keep final-layer lease release unchanged.
- Keep scheduler-visible finished request reporting unchanged.

Regression tests should compare outputs for:

- Greedy generation.
- Non-greedy generation with fixed seed.
- Cache hit path.
- Cache miss / partial miss path.
- Each of the three transfer modes.

## 12. Observability

Add metrics and logs that answer whether subprocess mode is helping.

Suggested counters/timers:

- `kv_transfer_engine_mode`.
- `kv_transfer_subprocess_start_total`.
- `kv_transfer_subprocess_restart_total`.
- `kv_transfer_subprocess_fallback_total`.
- `kv_transfer_ipc_request_total`.
- `kv_transfer_ipc_error_total`.
- `kv_transfer_ipc_roundtrip_ms`.
- `kv_transfer_ipc_payload_bytes`.
- `kv_transfer_key_build_ms`.
- `kv_transfer_addr_size_build_ms`.
- `kv_transfer_backend_put_ms`.
- `kv_transfer_backend_get_ms`.
- `kv_transfer_backend_batch_copy_ms`.

Logs should include:

- Engine selected.
- Subprocess pid.
- Backend registration success/failure.
- Restart reason.
- Downgrade reason.
- Fatal command failure with command id and request id.

## 13. Test Plan

### 13.1 Unit Tests

Add tests for `TransferProcessManager`:

- Starts subprocess and receives ready ack.
- Sends shutdown and exits cleanly.
- Detects child process exit.
- Detects IPC disconnect.
- Converts child exception into structured error.
- Restarts within configured limit.
- Downgrades after restart limit.

Add tests for payload conversion:

- `ReqMeta` to IPC payload.
- `LayerTransferTask` to IPC payload.
- `LayerLoadTask` to IPC payload.
- Runtime-only fields are excluded.
- Required fields are preserved.

Add tests for `SubprocessTransferEngine`:

- Register/start lifecycle.
- Submit load command.
- Submit save command.
- Wait save command.
- Wait layer load command.
- Get finished command.
- Fallback to `ThreadTransferEngine` on startup failure.

Add executor tests with mocked backend:

- Non-layerwise executor calls backend `put/get` with expected keys/addrs/sizes.
- Key layerwise executor preserves layer key format.
- GVA layerwise executor calls batch copy and final lease release.
- Backend exception returns recoverable/fatal status as configured.

### 13.2 Integration Tests

Where device/backend test infrastructure is available:

- Run one request through non-layerwise save/load.
- Run one request through key layerwise save/load.
- Run one request through GVA layerwise save/load.
- Kill transfer subprocess during idle and verify restart.
- Kill transfer subprocess during request and verify worker survives.
- Force buffer registration failure and verify thread fallback.

### 13.3 Precision Tests

Use deterministic workloads:

- Greedy decode baseline vs subprocess mode.
- Non-greedy fixed-seed baseline vs subprocess mode.
- Partial KV load failure path.
- Multi-layer transfer path.

Expected result:

- Token outputs match baseline for deterministic tests.
- Cache hit/miss reporting matches baseline.

### 13.4 Performance Tests

Run before/after comparisons under the same configuration:

- Existing thread mode.
- Subprocess mode.
- Subprocess mode with forced fallback excluded from steady-state metrics.

Collect:

- Throughput.
- P50/P95/P99 latency.
- Worker CPU profile.
- Transfer process CPU profile.
- Key-building CPU share.
- IPC overhead.
- Backend operation time.

## 14. Implementation Plan

### Phase 1: Introduce TransferEngine Boundary

- Add `TransferEngine` interface.
- Move current thread startup logic into `ThreadTransferEngine`.
- Make `KVPoolWorker` call engine methods instead of directly owning mode-specific transfer threads.
- Keep behavior unchanged.
- Add tests proving thread engine preserves existing behavior.

This phase should be low risk because it only wraps current behavior.

### Phase 2: Add IPC Types and Process Manager

- Add versioned command/result dataclasses.
- Add payload conversion helpers.
- Add `TransferProcessManager`.
- Add child process main loop.
- Add mock executor for lifecycle tests.
- Test startup, shutdown, exception, timeout, disconnect.

This phase proves process isolation before moving real transfer logic.

### Phase 3: Add Subprocess Engine With Mocked Backend

- Implement `SubprocessTransferEngine`.
- Wire it into `KVPoolWorker` behind config.
- Support register/start/shutdown and simple commands.
- Add fallback to thread engine.
- Keep subprocess mode off by default.

This phase validates worker integration.

### Phase 4: Migrate Non-Layerwise Executor

- Move non-layerwise key/address/size construction into subprocess executor.
- Reuse existing helper functions where possible.
- Preserve backend calls and result handling.
- Add equivalence tests against thread engine.
- Add IPC payload size metrics.

### Phase 5: Migrate Key Layerwise Executor

- Move layer key construction into subprocess.
- Maintain subprocess-local cached process tokens.
- Preserve layer wait behavior.
- Add per-layer completion results.
- Add equivalence tests.

### Phase 6: Migrate GVA Layerwise Executor

- Move GVA/address/size construction into subprocess.
- Implement buffer descriptor registration or fallback detection.
- Preserve batch copy and lease release behavior.
- Add cleanup behavior for partial failures.
- Add equivalence tests.

### Phase 7: Performance Prototype and Report

- Run baseline thread mode.
- Run subprocess mode.
- Measure key-building CPU share and IPC overhead.
- Document regressions and bottlenecks.
- Decide whether additional shared-memory optimization is required before default enablement.

### Phase 8: Rollout

- Start with opt-in subprocess mode.
- Enable `auto` mode for selected backend/mode combinations that pass tests.
- Keep thread fallback available.
- Do not remove thread engine until production stability is proven.

## 15. Prototype Comparison Report Template

The prototype report should include:

- Test environment.
- Model and workload.
- Backend.
- Transfer mode.
- Batch size and sequence length.
- Thread-mode throughput and latency.
- Subprocess-mode throughput and latency.
- Worker CPU share before/after.
- Transfer child CPU share.
- Key-building CPU share before/after.
- IPC round-trip latency.
- IPC payload size distribution.
- Backend operation timings.
- Crash/restart behavior.
- Fallback behavior.
- Known limitations.

Decision section:

- Keep subprocess disabled.
- Enable for one mode/backend.
- Enable auto mode with fallback.
- Require further shared-memory optimization.

## 16. Compatibility Strategy

Compatibility should be maintained by construction:

- Thread engine remains the reference.
- Subprocess mode uses the same metadata and helper functions where possible.
- Key format does not change.
- Backend calls preserve existing argument ordering.
- Worker-facing connector methods keep the same semantics.
- Existing tests continue to pass.

Any behavior change should be explicitly isolated and covered by tests.

## 17. Open Technical Questions

These should be resolved during prototype:

- Can each backend import/register KV buffers from a child process safely?
- Are existing backend handles process-shareable, or must the child register independently?
- Can `torch.npu.Event` semantics be represented safely through task completion messages?
- Which metadata arrays are large enough to require shared memory immediately?
- Which transfer operations are safe to replay after subprocess crash?
- How should partial GVA lease cleanup be handled after child death?

## 18. Recommended First PR Shape

The first PR should avoid changing all transfer logic at once.

Recommended contents:

- Add `TransferEngine` abstraction.
- Add `ThreadTransferEngine` wrapper around current logic.
- Add subprocess IPC dataclasses and process manager.
- Add subprocess engine behind disabled config.
- Add lifecycle tests.
- Add fallback tests.
- No default behavior change.

This creates the safe extension point needed for later mode-by-mode migration.

## 19. Final Design Summary

The refactor should turn AscendStore transfer into a pluggable engine system. The existing thread implementation becomes the compatibility engine. The new subprocess implementation owns transfer execution through a child process and explicit IPC protocol.

The worker process remains responsible for high-level orchestration, but the child process owns the GIL-heavy work: key construction, address/size scheduling, layerwise transfer execution, and backend I/O submission.

The design must treat process isolation, fallback, and observability as first-class features. Without those, the refactor may reduce one bottleneck while introducing correctness and operability risk.
