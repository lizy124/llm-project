# AscendStore KV Transfer Subprocess Refactor - Detailed Analysis

## 1. Background

The current AscendStore KV transfer path runs transfer work inside the worker process by using daemon threads. The main worker and transfer threads coordinate through `queue.Queue`, `threading.Event`, and mode-specific state stored on `KVPoolWorker` and `KVTransferThread` instances.

The requested refactor is to replace the in-process multi-thread transfer model with an independent transfer subprocess plus IPC, so key construction, buffer scheduling, and backend I/O submission run outside the worker process and no longer share the worker process GIL.

The scope covers all existing transfer modes:

- Non-layerwise transfer.
- Key-based layerwise transfer.
- GVA layerwise transfer.

The acceptance requirements are:

- The three modes remain functionally equivalent.
- Greedy and non-greedy output precision is unchanged.
- A transfer subprocess crash must not take down the worker. It must be restartable or downgrade to in-process thread mode.
- Before/after performance comparison must include throughput, latency, key-building CPU share, and IPC overhead.
- Deliverables must include design doc, prototype comparison report, PR/design note, and unit tests covering subprocess start/stop, exception recovery, cross-process buffer sharing, and IPC disconnect.

## 2. Current Architecture

### 2.1 Connector Entry

`AscendStoreConnector` is the top-level connector exposed to scheduler and worker roles. It reads the KV transfer configuration, determines whether layerwise transfer is enabled, determines whether GVA layerwise applies, and delegates worker-side operations to `KVPoolWorker`.

Important behavior:

- Worker role constructs `KVPoolWorker`.
- Scheduler role handles lookup server behavior for non-layerwise mode.
- Worker-side calls such as cache registration, load start, layer load wait, layer save, and save wait are delegated into `KVPoolWorker`.
- `use_gva_layerwise` is selected when layerwise mode is enabled and the backend is memcache.

This layer is not the main source of GIL pressure, but it is the right entry point for selecting whether the transfer engine is thread-based or subprocess-based.

### 2.2 Worker Coordination

`KVPoolWorker` owns the current transfer lifecycle. After cache registration, it starts send and receive transfer threads according to the selected mode.

Current mode matrix:

- Non-layerwise:
  - Save path uses `KVCacheStoreSendingThread` when the role can save.
  - Async load path uses `KVCacheStoreRecvingThread` when `load_async` is enabled.

- Key layerwise:
  - Save path uses `KVCacheStoreKeyLayerSendingThread`.
  - Load path uses `KVCacheStoreKeyLayerRecvingThread`.

- GVA layerwise:
  - Save path uses `KVCacheStoreLayerSendingThread`.
  - Load path uses `KVCacheStoreLayerRecvingThread`.

The worker also owns layerwise synchronization data, including current layer counters, layer finished events, save finished events, and NPU events used to order save/load behavior.

### 2.3 Transfer Threads

`KVTransferThread` is the common base for the current transfer workers. It is a daemon `threading.Thread` with:

- `request_queue: queue.Queue`
- `stored_requests`
- `finished_requests`
- `kv_events`
- `_fatal_error`

Its run loop sets backend device context, marks itself ready, pulls requests from the queue, and delegates to mode-specific `_handle_request` or `_handle_stored_request` logic. Exceptions are caught and saved as `_fatal_error`, then surfaced to the worker via `raise_if_failed()`.

This is a process-local failure model. It is useful for propagating thread exceptions, but it does not isolate the worker from transfer-side Python CPU pressure or unexpected transfer-side process corruption.

### 2.4 Metadata Objects

The existing metadata layer already contains the logical payloads needed by the transfer engine:

- `ReqMeta` for request-level KV metadata.
- `LoadSpec` for load targeting.
- `PoolKey` and `LayerPoolKey` for key construction.
- `ChunkedTokenDatabase` for efficient token-key generation.
- `SharedBlockData` for layerwise shared block metadata.
- `LayerTransferTask` for layerwise save tasks.
- `LayerLoadTask` for layerwise load tasks.
- `LayerBatchReqMeta`, `LayerBlockRange`, and `LayerMultiBlockReqMeta` for GVA/layer batching.

These objects are currently passed by reference within the process. In a subprocess design, they must become either serializable IPC payloads or references to shared memory / backend-registered buffers.

## 3. Current Hot Paths

The GIL-sensitive hot paths are concentrated in transfer-side Python code that still runs inside the worker process.

### 3.1 Key Construction

Key construction happens in several forms:

- `PoolKey.to_string()` serializes model, rank, group, cache family, and block hash into a key string.
- `LayerPoolKey.to_string()` adds layer identity to the base key.
- `ChunkedTokenDatabase` caches prefixes and constructs token-key strings.
- `process_token_key_strings_with_block_ids` directly generates key strings for block ids.

These operations are Python-heavy and may run in loops for many blocks/layers/requests.

### 3.2 Address and Size Construction

The non-layerwise send path builds lists of keys, addresses, and sizes before calling backend `put`. The receive path builds load key/address/size lists before calling backend `get`.

The layerwise GVA path builds arrays for GVA, address, and size batches before calling batch copy APIs.

These list/array construction paths run in Python and can contend with worker forward code.

### 3.3 Backend Submission

The transfer classes call backend operations such as:

- `put`
- `get`
- `batch_copy`
- `batch_write_finish`
- `batch_remove_lease`

Even if the backend operation releases the GIL internally, the request preparation and submission orchestration are Python-driven.

### 3.4 Layerwise Synchronization

Layerwise modes add more Python scheduling:

- Per-layer task construction.
- Per-layer load/save finished event handling.
- Waiting on previous save layers.
- Recording NPU events.
- Lease release on final layer for GVA load.

This makes layerwise modes especially sensitive to coordination overhead.

## 4. Main Problems

### 4.1 Worker GIL Contention

The current transfer threads share the same Python interpreter and GIL as the worker process. This means transfer-side key construction and scheduling can block or delay worker-side Python execution, including forward-path orchestration.

The transfer thread design improves concurrency for blocking I/O, but it does not solve CPU-bound Python scheduling overhead.

### 4.2 Failure Domain Is Too Large

Transfer exceptions are currently captured in `_fatal_error`, but the transfer engine still lives inside the worker process. A severe error, deadlock, native backend crash, or corrupted backend state can still affect the worker process directly.

The requested behavior requires a transfer crash not to take down the worker. That requires process isolation.

### 4.3 Synchronization Is Process-Local

The current design depends on `queue.Queue`, `threading.Event`, and in-memory Python state. These mechanisms do not cross a process boundary.

A subprocess design needs explicit command/result messages, explicit task IDs, and explicit completion/error reporting.

### 4.4 Mixed Responsibilities

`KVPoolWorker` currently owns too much of the execution story:

- It starts mode-specific transfer threads.
- It owns layerwise events.
- It submits requests directly to thread queues.
- It waits for queue completion.
- It checks thread failures.
- It collects finished events.

After the refactor, `KVPoolWorker` should remain the connector-facing coordinator, while transfer execution belongs to a transfer engine abstraction.

### 4.5 Fallback Is Not Yet a First-Class Path

The existing thread mode can become the fallback path, but it is not currently structured as an interchangeable backend. A clean refactor should introduce a transfer engine interface so thread mode and subprocess mode can share the same worker-facing contract.

## 5. Mode-by-Mode Analysis

### 5.1 Non-Layerwise Save

Current behavior:

- Worker gathers stored requests.
- Worker records an NPU event for each request.
- Send thread receives the request.
- Send thread builds store masks, skip ranges, keys, addrs, and sizes.
- Send thread performs lookup and calls backend `put`.
- Finished state and KV events are recorded in thread-local state.

Subprocess impact:

- Request metadata must be serializable.
- Device event ordering must be represented without passing raw Python event objects by reference.
- Key/address/size construction should move to the subprocess.
- Finished events must be returned over IPC.

Primary risk:

- If large metadata arrays are copied over IPC for every request, IPC overhead can consume the expected GIL savings.

### 5.2 Non-Layerwise Load

Current behavior:

- Worker submits async receive requests.
- Receive thread builds load keys and destination addresses.
- Receive thread rotates by TP rank when needed.
- Backend `get` is called.
- Failed block ids are recorded.

Subprocess impact:

- Load key generation should happen in the subprocess.
- Failed block ids and request completion must be reported back explicitly.
- The worker must be able to continue if the subprocess fails and load transfer is downgraded or disabled.

Primary risk:

- Failed-block reporting and prefix cache behavior must stay equivalent or cache-hit behavior can regress.

### 5.3 Key Layerwise Save

Current behavior:

- Worker builds layer transfer tasks.
- Save thread builds cached process tokens.
- For each layer, it splits keys by layer and calls `key.to_string()`.
- It waits on synchronization events and calls backend `put`.
- It marks per-layer save finished events.

Subprocess impact:

- `LayerTransferTask` must cross IPC.
- `cached_process_tokens` should live in the transfer process to avoid repeated serialization overhead.
- Per-layer completion must be reported to worker.
- Save/load ordering must be encoded in the command protocol.

Primary risk:

- Layer ordering and saved-event semantics can change subtly, causing correctness or precision regressions.

### 5.4 Key Layerwise Load

Current behavior:

- Receive thread waits for previous save layer when consumer-is-to-put semantics require it.
- It builds per-layer keys from block hashes.
- It calls backend `get` and marks per-layer load completion.

Subprocess impact:

- Load scheduling should run in subprocess.
- Worker-facing `wait_for_layer_load(layer_name)` must remain synchronous from the caller perspective.
- IPC result must include per-layer status.

Primary risk:

- If wait behavior changes, forward execution may observe missing KV data.

### 5.5 GVA Layerwise Save

Current behavior:

- Save thread builds shared block data.
- It constructs GVA/address/size arrays.
- It submits `_batch_copy_with_limits(..., direction=0)`.
- It calls `batch_write_finish`.

Subprocess impact:

- Backend buffer registration and GVA metadata must be available in the child process.
- Large arrays should not be copied repeatedly if avoidable.
- `batch_write_finish` semantics must remain identical.

Primary risk:

- Backend memory registration may be process-specific. If so, the subprocess must either register its own handles or receive backend-supported transferable descriptors.

### 5.6 GVA Layerwise Load

Current behavior:

- Receive thread builds shared data.
- It waits on save-layer dependencies and NPU events when required.
- It submits `_batch_copy_with_limits(..., direction=1)`.
- On final layer, it releases leases with `batch_remove_lease`.

Subprocess impact:

- Lease ownership and release must be carefully preserved.
- IPC failure must not leak backend leases.
- Final-layer cleanup must run even when partial failures occur.

Primary risk:

- Crash during GVA load can leave leases or registered buffers behind unless cleanup is explicit.

## 6. Cross-Process Data Classification

### 6.1 Direct IPC Payloads

Small control data can be sent directly:

- Request ids.
- Layer ids.
- Group ids.
- Boolean flags.
- Mode names.
- Backend configuration.
- Error codes.
- Completion statuses.

### 6.2 Serialized Metadata

Moderate request metadata can be serialized if the payload size is bounded:

- `ReqMeta` fields needed for key generation.
- `LoadSpec`.
- Block hash lists.
- Layer block ranges.
- Write-finish keys.

This must be measured because metadata volume can grow with batch size and sequence length.

### 6.3 Shared Memory or Registered Handles

Large arrays and device/backend buffers should not be copied through IPC:

- KV cache tensors.
- Backend-registered memory.
- Block GVA arrays.
- Block id numpy arrays if large.
- Load/store destination address arrays when reusable.

These should be shared through backend-supported handles, shared memory descriptors, or subprocess-local registration after startup.

## 7. Failure and Recovery Analysis

### 7.1 Recoverable Failures

Recoverable failures include:

- IPC command timeout.
- Transfer subprocess Python exception.
- Temporary backend operation failure.
- Child process exits before accepting a request.

Expected behavior:

- Mark current transfer task failed.
- Restart subprocess if policy allows.
- Re-register buffers after restart.
- Retry only if operation is idempotent and safe.
- Otherwise downgrade to thread mode.

### 7.2 Fatal Failures

Fatal failures include:

- Backend initialization cannot succeed in subprocess.
- Shared buffer registration is unsupported across processes.
- Repeated subprocess crashes.
- IPC protocol version mismatch.
- Corrupted or inconsistent transfer state.

Expected behavior:

- Disable subprocess mode for the current worker.
- Fall back to existing in-process thread mode.
- Surface a clear warning with mode, request id, and reason.

### 7.3 Worker Survival

The worker should not depend on transfer subprocess liveness for process survival. It should treat transfer as an optional engine with three possible states:

- Running.
- Restarting.
- Downgraded.

Only unrecoverable failures that affect correctness after fallback should propagate as user-visible errors.

## 8. Precision and Correctness Risks

Greedy and non-greedy output precision can change if KV load/save ordering changes, if failed loads are handled differently, or if layerwise completion is reported too early.

Correctness-sensitive areas:

- Per-layer load completion timing.
- Save-before-load ordering when `consumer_is_to_put` applies.
- TP-rank key rotation.
- Failed block recording.
- Final-layer lease release.
- Key string format and layer id inclusion.
- Cached prefix/token behavior.

The subprocess refactor must not change key formats, block selection, load masks, store masks, or backend write/read order except where the existing code already allows asynchronous behavior.

## 9. Existing Test Coverage Gap

Existing tests cover connector delegation and some backend/metadata utilities, but the acceptance criteria require more behavior-level tests.

Missing coverage:

- Subprocess start and graceful shutdown.
- Subprocess crash detection.
- Restart after child exit.
- Downgrade to thread mode.
- IPC disconnect handling.
- Cross-process buffer registration failure.
- Per-mode request equivalence.
- Layerwise event ordering.
- Failed-block propagation.

Tests should mock NPU and backend behavior where necessary so they can run in unit test environments without real Ascend devices.

## 10. Performance Analysis Requirements

The comparison must isolate whether subprocess mode actually improves the intended bottleneck.

Required metrics:

- End-to-end throughput.
- P50/P95/P99 latency.
- Worker process Python CPU share.
- Transfer process Python CPU share.
- Key-building CPU share.
- IPC round-trip latency.
- IPC payload size.
- Backend put/get/batch-copy time.
- Restart/fallback overhead.

Required comparisons:

- Existing in-process thread mode.
- Subprocess mode with identical workload.
- Subprocess mode under crash/restart scenario.
- Subprocess mode under fallback scenario.

A useful prototype should include counters around key construction and IPC serialization/deserialization, not just total request latency.

## 11. Recommended Refactor Boundary

The cleanest boundary is to introduce a worker-facing transfer engine interface:

- Thread engine: wraps the existing implementation.
- Subprocess engine: manages child process and IPC.

`KVPoolWorker` should call the interface instead of directly owning mode-specific transfer threads.

This keeps risk contained:

- Existing thread behavior remains available as fallback.
- Subprocess mode can be gated by config.
- Tests can validate both engines against the same contract.
- The migration can be staged without rewriting every mode at once.

## 12. Conclusion

The current design already has clear mode-specific transfer logic, but it is coupled to in-process Python threads. The GIL-sensitive parts are exactly the parts the requested task identifies: key construction, buffer scheduling, and backend submission orchestration.

A correct solution must do more than spawn a subprocess. It must introduce an explicit transfer engine boundary, define an IPC command/result protocol, classify data by serialization strategy, preserve layerwise ordering semantics, and keep the current thread model as a robust fallback.

The most important design rule is: the worker should submit high-level transfer intent, while the transfer subprocess owns key construction, address/size scheduling, and backend I/O submission. If those hot paths stay in the worker process, the refactor will not solve the GIL problem.
