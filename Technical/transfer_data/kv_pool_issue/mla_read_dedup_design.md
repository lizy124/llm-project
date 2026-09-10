# MLA KV Pool Read Dedup Detailed Design

## Goal

Implement MLA read-side deduplication for the worker-side KV pool load path: only one rank in each shared MLA KV group reads data from the KV pool, then distributes the exact loaded bytes to peer TP ranks so every rank's local KV cache is byte-identical to the pre-dedup behavior.

## Non-Goals

- Do not change key format.
- Do not change write-side dedup behavior.
- Do not optimize non-MLA models.
- Do not change TP mismatch strided I/O in the first implementation.
- Do not change scheduler-side cache hit decisions.
- Do not rely on backend-specific APIs beyond the existing `get()` interface for the first implementation.

## High-Level Design

For MLA with TP>1, ranks inside the same `put_step` group share identical KV pool keys. The first rank in that group is the load owner. The owner performs the backend `get()` into its local KV cache. Then all ranks in the group participate in a byte-level broadcast:

1. All ranks build the same logical key list and their own local destination address list.
2. Owner rank calls `m_store.get(key_list, owner_addr_list, size_list)`.
3. Owner rank packs bytes from its destination address list into a contiguous `torch.uint8` staging tensor.
4. Peer ranks allocate a same-sized staging tensor.
5. Owner broadcasts the staging tensor to the group.
6. Every peer unpacks bytes from the staging tensor into its local destination addresses.
7. Owner broadcasts backend return states so peers update invalid-block metadata consistently.

The design intentionally broadcasts byte payloads instead of higher-level typed tensors. That preserves exact byte identity across dtypes, KV layouts, C8/NZ layouts, and multiple KV cache entries.

## Applicability Guard

Add a helper such as `_can_dedup_mla_read()` on `KVPoolWorker`:

```python
def _can_dedup_mla_read(self) -> bool:
    return (
        self.enable_mla_read_dedup
        and self.use_mla
        and self.put_step > 1
        and not self.tp_mismatch
        and not self.use_layerwise
        and not self.load_async
    )
```

Recommended initial config:

- Add `enable_mla_read_dedup` in `kv_connector_extra_config`.
- Default can be `True` only for the narrow synchronous non-layerwise path above, because the guard is conservative.
- If maintainers prefer risk isolation, default to `False` and document the enable flag.

Do not enable for:

- `tp_mismatch=True`: keep `_load_kv_tp_mismatch()` unchanged.
- `use_layerwise=True`: separate layerwise design needed.
- `load_async=True`: separate thread/collective-order validation needed.
- `use_mla=False`: non-MLA keys are not shared the same way.
- `put_step == 1`: no redundant rank in group.

## Rank Grouping

Use `put_step` groups, not the whole TP group.

For each TP rank:

```python
shared_group_index = self.tp_rank // self.put_step
shared_group_start = shared_group_index * self.put_step
shared_group_ranks = list(range(shared_group_start, shared_group_start + self.put_step))
owner_tp_rank = shared_group_start
is_owner = self.tp_rank == owner_tp_rank
```

For MLA, `num_kv_head=1`, so `put_step=tp_size` and owner is TP rank 0. For a more general case where `num_kv_head < tp_size`, this grouping also matches existing write-side key ownership.

The collective needs the actual torch distributed process group containing those global ranks. There are two possible implementations:

1. Preferred: create/cache dedicated process groups for each `put_step` group.
2. Simpler but less efficient: use the tensor parallel group and broadcast from the group owner while all TP ranks participate.

Preferred implementation should add a small cached helper:

```python
def _get_mla_read_dedup_group(self):
    # Build once per worker. Use global ranks from the TP group if available.
    # Cache by tuple(shared_group_ranks).
```

Important detail: `torch.distributed.broadcast(tensor, src=...)` expects global rank for `src` when a process group is supplied. The existing `pyhccl.py` comments also note this behavior.

If vLLM's `get_tp_group()` exposes `device_group` and rank mapping, use that. Otherwise use `torch.distributed.new_group(...)` with global ranks derived from `parallel_config.rank` and TP topology at worker initialization.

## Helper Refactor

First extract existing synchronous key/address preparation into a reusable helper without changing behavior:

```python
def _build_load_entries(
    self,
    request: ReqMeta,
    token_len: int,
    load_group_ids: list[int],
) -> tuple[list[str], list[list[int]], list[list[int]], list[int]]:
    ...
```

This helper should contain the current logic from `start_load_kv()`:

- `request.skip_null_blocks_by_group = self.group_uses_align_state`
- `load_masks = self.token_database.load_mask(request.block_hashes, token_len)`
- group filtering using `kv_cache_group_ids`
- `mask_num` calculation from `load_spec.vllm_cached_tokens`
- `skip_null_blocks`
- `chunk_filter`
- `process_token_key_strings_with_block_ids(...)`
- `prepare_value(...)`

Then keep the existing circular shift exactly where it is needed:

```python
key_list_c = _circular_shift(key_list, self.tp_rank % len(key_list))
addr_list_c = _circular_shift(addr_list, self.tp_rank % len(addr_list))
size_list_c = _circular_shift(size_list, self.tp_rank % len(size_list))
block_id_list_c = _circular_shift(block_id_list, self.tp_rank % len(block_id_list))
```

For dedup, all ranks in a `put_step` group should use the owner's shift, not each peer's shift, so the broadcast payload ordering is identical:

```python
shift = owner_tp_rank % len(key_list)
key_list_c = _circular_shift(key_list, shift)
addr_list_c = _circular_shift(addr_list, shift)
size_list_c = _circular_shift(size_list, shift)
block_id_list_c = _circular_shift(block_id_list, shift)
```

This preserves owner get ordering and gives every peer the same segment order for unpacking.

## Byte Packing and Unpacking

Need two low-level helpers:

```python
def _total_transfer_bytes(size_list: list[list[int]]) -> int:
    return sum(sum(sizes) for sizes in size_list)
```

```python
def _pack_from_addrs(
    self,
    addr_list: list[list[int]],
    size_list: list[list[int]],
) -> torch.Tensor:
    ...
```

```python
def _unpack_to_addrs(
    self,
    payload: torch.Tensor,
    addr_list: list[list[int]],
    size_list: list[list[int]],
) -> None:
    ...
```

Implementation options:

### Option A: Tensor view over raw memory

Create `torch.uint8` tensors that alias each raw address span, then copy to/from a contiguous staging tensor.

Pseudo-code shape:

```python
payload = torch.empty(total_bytes, dtype=torch.uint8, device=self.device)
offset = 0
for addrs, sizes in zip(addr_list, size_list):
    for addr, size in zip(addrs, sizes):
        span = make_uint8_tensor_from_address(addr, size, device=self.device)
        payload[offset:offset + size].copy_(span)
        offset += size
```

The inverse copies from `payload` back into each local span.

The missing piece is a safe local utility for creating a tensor from a raw device pointer. If the codebase already has one in a backend or NPU utility, reuse it. If not, this option may require a small custom utility and careful device lifetime handling.

### Option B: Reconstruct typed cache tensor slices

Instead of raw pointer aliasing, derive slices from registered KV cache tensors. This is safer with PyTorch but requires more layout-specific code:

- For each `prepare_value()` segment, determine which cache tensor and block/token/head/layer slice it maps to.
- Use tensor slicing to pack/unpack.

This duplicates logic already encoded in `ChunkedTokenDatabase.prepare_value()` and is riskier for hybrid/sparse/future layouts.

### Option C: Backend-assisted local copy

If Mooncake/Memcache exposes a local memory copy or raw pointer copy primitive underneath `store`, use it for pack/unpack. This keeps raw address semantics but adds backend coupling.

Recommended for first implementation: Option A if a safe pointer-to-tensor utility exists or can be added with minimal native dependency. Otherwise, use Option C behind a small abstraction. Avoid Option B unless maintainers reject raw pointer wrapping.

## Collective Protocol

For each request:

1. Build key/address/size/block lists on all ranks.
2. If no keys, return on all ranks.
3. Select owner shift and circular-shift all lists.
4. Compute `payload_nbytes` on all ranks. Assert or broadcast this size from owner if necessary.
5. Owner calls `m_store.get()`.
6. Owner creates payload from local destination spans.
7. All ranks call `dist.broadcast(payload, src=owner_global_rank, group=dedup_group)`.
8. Non-owner ranks unpack payload into their local destination spans.
9. Broadcast return-state tensor from owner to all ranks.
10. All ranks run the same invalid-block update logic using `block_id_list_c` and return states.

Return-state representation:

```python
# owner
if ret is None:
    ret_tensor = torch.ones(len(block_id_list_c), dtype=torch.int32, device=device)
else:
    ret_tensor = torch.tensor(ret, dtype=torch.int32, device=device)

# peers
ret_tensor = torch.empty(len(block_id_list_c), dtype=torch.int32, device=device)
dist.broadcast(ret_tensor, src=owner_global_rank, group=dedup_group)
ret = ret_tensor.cpu().tolist()
```

If `payload_nbytes == 0`, skip payload broadcast but still broadcast return states if owner performed a get. In practice no keys should imply no get.

## Failure Handling

Existing behavior:

- `ret is not None and any(r != 0 for r in ret)`: call `record_failed_blocks(block_id_list_c, ret)`.
- `ret is None`: treat all blocks as failed.
- Single KV group: update `_invalid_block_ids`.
- Hybrid/multi-group: log an error and avoid scheduler crash.

Dedup must preserve this logic. The owner is the only rank that observes backend `ret`; it must broadcast normalized return states to peers. Then every rank applies the same invalid-block update to local worker state.

If owner `m_store.get()` raises an exception instead of returning failure states, there are two choices:

1. Preserve current behavior and allow the owner exception to fail the step. This risks peer ranks hanging in broadcast.
2. Catch exceptions on owner, convert to all-failed return states, broadcast failure, and log the exception.

Recommended: catch on owner around `m_store.get()`, normalize to all-failed states, still enter broadcast, and log `exception type` and `req_id`. This prevents collective deadlock and matches existing `ret is None` fallback semantics better than crashing only one rank.

## Interaction With TP Mismatch

Do not apply MLA read dedup when `self.tp_mismatch` is true.

Reason:

- TP mismatch maps local rank cache heads to multiple effective sub-keys.
- `_load_kv_tp_mismatch()` builds strided address lists with `_build_strided_addrs()` and rewrites `head_or_tp_rank` per sub-key.
- The existing path rotates keys using the local `tp_rank`.
- Applying a simple `put_step` broadcast could copy an owner's local strided layout into peers that need different head slices.

Implementation guard:

```python
if self.tp_mismatch:
    self._load_kv_tp_mismatch(...)
    continue
```

This must remain before the dedup path.

## Interaction With Non-MLA

For non-MLA, `self.use_mla` is false and the existing path is unchanged. Models where `num_kv_head < tp_size` but not MLA also use `put_step`; the first PR should not optimize them unless their byte identity across ranks is proven. The guard should require `self.use_mla`.

## Interaction With Hybrid and Sparse Layouts

The synchronous path already supports multiple KV groups and alignment masks. MLA read dedup can technically work if packing uses raw destination addresses and sizes, because it is layout-agnostic. However, first implementation should be conservative:

- Allow single-group dense MLA first.
- If enabling multi-group MLA, keep the same helper-generated address/size lists and byte-pack all groups in order.
- Keep sparse/TP mismatch excluded unless tested.

Recommended initial guard can include:

```python
and not self.use_sparse
```

If there are MLA sparse models that must be supported, remove this only after bytewise tests cover them.

## Async Path Design

Async path lives in `KVCacheStoreRecvingThread._handle_request()`.

Two possible approaches:

1. Disable dedup when `load_async=True` in first PR.
2. Move the same helper and broadcast protocol into the recv thread.

Risks with option 2:

- All ranks must enqueue identical request sequences into recv threads.
- Collectives run outside the main forward thread.
- Concurrent HCCL collectives from model execution and recv thread can conflict if not ordered by a dedicated group/stream.
- Finished-request bookkeeping must wait until broadcast and unpack complete on all ranks.

Recommended staged plan:

- First PR: guard out async, log a one-time debug message if `enable_mla_read_dedup=True` but `load_async=True`.
- Second PR: add async support after validating collective order under representative serving load.

If same-PR async support is mandatory, inject a worker callback into `KVCacheStoreRecvingThread` so it reuses `KVPoolWorker._load_kv_mla_dedup(...)` rather than duplicating logic.

## Layerwise Path Design

Layerwise has two read mechanisms:

- Key-based layer recv thread uses `m_store.get()` around `kv_transfer.py:1293`.
- GVA layerwise load uses `batch_copy(G2L)` around `kv_transfer.py:1614`.

Layerwise dedup options:

1. Owner loads each layer, broadcasts that layer's bytes, peers unpack before `wait_for_layer_load()` returns.
2. Owner loads all requested layers into local cache, broadcasts per-layer or batched payloads.
3. Use GVA/shared memory semantics if backend supports direct cross-rank mapping.

Recommended staged plan:

- Do not implement layerwise in first PR unless required.
- If required, start with key-based layer recv thread because it resembles synchronous get.
- Keep GVA layerwise separate; `batch_copy(G2L)` already has a different memory model and lease handling.

## Instrumentation

Add lightweight counters and debug logs under the worker:

- `mla_read_dedup_enabled`
- `mla_read_dedup_owner_get_calls`
- `mla_read_dedup_peer_get_skips`
- `mla_read_dedup_payload_bytes`
- `mla_read_dedup_broadcast_us`
- `mla_read_dedup_pack_us`
- `mla_read_dedup_unpack_us`
- existing backend get call count before/after, preferably backend-side if available

Counters should be per process/rank and loggable at request granularity behind debug level. For performance validation, add aggregate logs every N requests or use existing profiling config if available.

## Unit Test Plan

Add focused unit tests near existing KV transfer tests, likely under `tests/ut/distributed/ascend_store/` or `tests/ut/distributed/kv_transfer/`.

### Test 1: guard behavior

Construct a minimal/fake worker and verify `_can_dedup_mla_read()` returns:

- true for `use_mla=True`, `put_step>1`, no mismatch, no layerwise, no async.
- false for non-MLA.
- false for TP=1 / `put_step=1`.
- false for `tp_mismatch=True`.
- false for `use_layerwise=True`.
- false for `load_async=True` in first PR.

### Test 2: list construction unchanged

Mock `ChunkedTokenDatabase` and a `ReqMeta` so `_build_load_entries()` returns the same key/address/size/block lists as the previous inline implementation.

### Test 3: owner shift consistency

For TP ranks in the same `put_step` group, verify dedup mode uses owner shift for all ranks. Existing non-dedup mode should still use local `tp_rank` shift.

### Test 4: return-state propagation

Mock owner backend `get()` to return partial failures, broadcast those states through a fake collective, and verify peers update the same invalid block IDs.

### Test 5: byte pack/unpack round trip

Allocate local tensors, build artificial address/size lists pointing at discontiguous spans, pack from one tensor, unpack into another, and assert bytewise equality for all spans.

### Test 6: no backend get on peers

With fake backend and fake collective, run owner and peer flows separately and assert:

- owner calls `get()` once.
- peer calls `get()` zero times.
- peer local buffer receives expected bytes.

## Integration Test Plan

Run multi-rank tests on Ascend hardware because torch/HCCL collectives and NPU memory pointer behavior cannot be fully validated in CPU-only unit tests.

Scenarios:

1. MLA model, TP=2, sync non-layerwise KV pool load.
2. MLA model, TP=4.
3. MLA model, TP=8.
4. Non-MLA model, TP=2 or TP=4, same KV pool path.
5. MLA with `tp_mismatch=True`, if deployment has a supported producer/consumer TP mismatch case.
6. MLA with flag disabled, verify old behavior.

Correctness method:

- Run baseline without dedup and dump/compare local KV cache byte ranges after `start_load_kv` and before attention consumes them.
- Run dedup with same request/block hashes and compare every rank's relevant KV cache spans bytewise to baseline.
- Verify generated outputs for representative prompts are unchanged.
- Verify invalid-block fallback by injecting backend get failure for selected keys.

## Performance Validation Plan

Measure both get call count and latency. Use the same model, prompts, block size, backend, and hardware for baseline and dedup.

Matrix:

- TP: 2, 4, 8.
- Sequence length: 16K, 64K.
- Batch shape: at least one single-request long-prefix case and one multi-request batch case.
- Backend: the target production KV pool backend, likely Mooncake or Memcache.
- Mode: baseline, dedup enabled.

Metrics:

- Backend `get()` call count per rank and total per request.
- Total bytes read from KV pool per request.
- Broadcast payload bytes.
- Pack/unpack latency.
- Broadcast latency.
- `start_load_kv` latency.
- End-to-end prefill/decode latency depending on where KV load is used.
- Throughput if serving benchmark is available.

Expected result:

- Total backend get calls decrease by approximately `1 - 1 / put_step`.
- TP=2: about 50% fewer get calls.
- TP=4: about 75% fewer get calls.
- TP=8: about 87.5% fewer get calls.
- End-to-end latency improves when removed backend overhead exceeds pack/broadcast/unpack cost.

## Rollout Plan

1. Add config flag and guards.
2. Extract synchronous load-entry builder with no behavior change.
3. Add byte pack/unpack helpers and unit tests.
4. Add dedup group helper.
5. Implement `_load_kv_mla_dedup()` for synchronous non-layerwise path.
6. Route `start_load_kv()` through dedup helper when guard passes.
7. Add unit tests for guard, list construction, owner-only get, failure propagation, and byte copy.
8. Run multi-rank correctness tests.
9. Run performance matrix.
10. Decide whether async/layerwise support is required for the same PR or a follow-up.

## Proposed Code Shape

### New worker fields

Initialize in `_init_kv_transfer_config()` or `_init_state_vars()`:

```python
self.enable_mla_read_dedup = bool(extra_config.get("enable_mla_read_dedup", True))
self._mla_read_dedup_group = None
self._mla_read_dedup_owner_global_rank = None
```

### New helpers

```python
def _can_dedup_mla_read(self) -> bool: ...

def _get_mla_read_dedup_group(self): ...

def _build_load_entries(self, request, token_len, load_group_ids): ...

def _normalize_get_ret(self, ret, count): ...

def _apply_load_failures(self, request, block_id_list, ret): ...

def _pack_from_addrs(self, addr_list, size_list): ...

def _unpack_to_addrs(self, payload, addr_list, size_list): ...

def _load_kv_mla_dedup(self, request, token_len, load_group_ids): ...
```

### `start_load_kv()` routing

Inside the existing per-request loop, after `token_len` is computed and before the regular inline get path:

```python
if self.tp_mismatch:
    ... existing mismatch path if applicable ...

if self._can_dedup_mla_read():
    self._load_kv_mla_dedup(request, token_len, load_group_ids)
    continue
```

Then keep the existing non-dedup path for all other cases.

## Open Questions

1. Does the target deployment use `load_async=True` or `use_layerwise=True` for MLA KV pool reads? If yes, first PR scope must include that path or explicitly document the limitation.
2. Is there an existing utility in vLLM/vLLM Ascend to create a `torch.Tensor` view from a raw NPU pointer? If not, should we add one or use a backend-local copy primitive?
3. Should the feature default to enabled for the conservative guard, or disabled for staged rollout?
4. Which backend is the primary target for performance data: Mooncake, Memcache, or both?
5. Are producer and consumer TP sizes always equal for MLA in the target benchmark, or must TP mismatch be tested as a non-regression only?

## Acceptance Mapping

### Functional correctness

- Bytewise identity: covered by pack/unpack unit tests plus multi-rank KV cache span comparison.
- TP mismatch: guard keeps existing `_load_kv_tp_mismatch()` path unchanged; add non-regression test.
- MLA/non-MLA: guard enables only MLA and leaves non-MLA path unchanged; add tests for both.

### Performance validation

- TP=2/4/8: collect backend get count and `start_load_kv`/E2E latency before and after.
- 16K/64K: run long-prefix benchmark with same block/cache settings.
- Report pack/broadcast/unpack overhead to explain cases with smaller gains.

### Deliverables

- PR with guarded implementation.
- Design document based on this file.
- Unit tests and multi-rank correctness validation.
- Performance report with tables for TP and sequence-length matrix.
