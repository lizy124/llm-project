# MLA KV Pool Read Dedup Analysis

## Background

In `vllm-ascend`, MLA KV cache data is shared across multiple TP ranks because MLA has one effective KV head. The write side already avoids redundant writes: `KVPoolWorker._init_key_head_config()` sets `num_kv_head = 1` for MLA, derives `put_step = tp_size // num_kv_head`, and save paths skip ranks whose `tp_rank % put_step != 0`.

The read side is not aligned with this policy. In the worker-side `start_load_kv` path, each TP rank still builds the same KV pool keys and calls backend `get()` independently, then copies identical bytes into that rank's local KV buffer. This creates redundant KV pool I/O proportional to TP size.

Relevant code:

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:129`: detects MLA with `model_config.use_mla`.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:195`: sets `num_kv_head = 1` for MLA.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:200`: computes `put_step`; for MLA with TP=8, `put_step=8`.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:288`: keys use `head_or_tp_rank`; all ranks inside a `put_step` group share the same key namespace.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1026`: layerwise save path skips non-owner ranks with `tp_rank % put_step != 0`.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:924`: synchronous read path builds `key_list`, `addr_list`, `size_list` per rank.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:980`: every rank calls `self.m_store.get(...)`.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:923`: async recv thread repeats the same pattern.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1978`: TP mismatch has a dedicated strided I/O path.

## Current Read Behavior

For a loadable request, the synchronous path performs the following steps on every TP rank:

1. Derive `token_len` from `load_spec.kvpool_cached_tokens`, with a granularity correction for the `token_len - 1` case.
2. Resolve target KV groups from `request.kv_cache_group_ids` or default to group 0.
3. Build load masks from block hashes.
4. For each group, walk token chunks and block IDs with `process_token_key_strings_with_block_ids(...)`.
5. For each chunk, build target local KV cache addresses with `prepare_value(...)`.
6. Rotate key/address/size/block lists by `tp_rank % len(key_list)` to distribute first-key pressure across ranks.
7. Call backend `get(keys, addrs, sizes)`.
8. Convert non-zero return states into invalid block IDs.

For MLA TP>1, ranks in the same `put_step` group produce equivalent keys. They target different local buffers, but the bytes fetched from the KV pool are the same. This means TP=2/4/8 causes roughly 2x/4x/8x backend get calls and remote reads for the same logical MLA KV payload.

## Why Write Dedup Does Not Automatically Fix Read

Write-side dedup only changes which rank publishes data to the pool. It does not change the read-side contract: every rank still needs its own local KV cache populated before attention reads it. Since the local buffers are rank-private, avoiding duplicate remote `get()` requires an intra-TP data distribution step after a single owner rank has loaded data.

The owner rank must copy bytes from remote KV pool into its local KV cache first, then broadcast/copy those exact bytes to peer ranks' local KV cache addresses. The backend interface only supports `get(keys, addrs, sizes)` and `put(keys, addrs, sizes)`, where `addrs` are local memory addresses registered with the backend. It does not expose a cross-rank remote-copy primitive. Therefore the dedup design needs to use torch/HCCL communication over tensors that alias or stage the loaded memory.

## Affected Paths

### Synchronous non-layerwise path

This is the most direct target. It is implemented in `KVPoolWorker.start_load_kv()` when `self.use_layerwise` is false and `self.load_async` is false. This path currently calls `m_store.get()` inline.

### Async non-layerwise path

When `self.load_async` is true, `KVPoolWorker.start_load_kv()` enqueues the request into `KVCacheStoreRecvingThread`, whose `_handle_request()` performs the same key/address/size construction and calls `m_store.get()`.

If production MLA KV pool uses `load_async`, this path must be updated in the same PR. Otherwise a feature flag can initially restrict dedup to synchronous reads, but that would not fully satisfy the broad worker-side read-path requirement.

### Layerwise path

Layerwise read uses either key-based `m_store.get()` per layer or GVA `batch_copy(G2L)`, depending on backend/config. It has separate task construction and thread classes. It also has existing write-side `put_step` skipping.

Layerwise read dedup is more complex because distribution would happen per layer and must fit the layer load/wait pipeline. It should be treated as a second phase unless the target deployment definitely uses layerwise MLA KV pool.

### TP mismatch path

TP mismatch uses `_load_kv_tp_mismatch()` and `_build_tp_mismatch_keys_and_addrs()` to split a local KV rank into effective sub-keys. It is explicitly disabled for layerwise and hybrid layouts. The acceptance criterion says TP mismatch must not break dedup logic. The safest design is to bypass MLA read dedup when `self.tp_mismatch` is true and preserve the existing strided get behavior. Any future optimization for TP mismatch should be designed separately because the address/key relationship differs.

## Functional Requirements

1. For MLA TP>1 without TP mismatch, exactly one rank per `put_step` group should fetch each shared KV payload from the KV pool.
2. All ranks in the same `put_step` group must end with local KV cache bytes identical to the previous per-rank get behavior.
3. Failure handling must remain equivalent: if owner `get()` fails for a block, all peer ranks must mark the same invalid block IDs or observe the same fallback behavior.
4. Non-MLA models must keep the existing read path.
5. MLA with TP=1 must keep the existing read path.
6. TP mismatch must keep the existing strided I/O path unless explicitly implemented and tested.
7. The optimization must not alter key generation, block masking, partial-load behavior, or `load_spec.token_len` semantics.

## Complexity Assessment

Overall complexity: medium-high.

The code change for synchronous read can be moderate if implemented as helper extraction plus a guarded owner-load-and-broadcast path. The complexity is in correctness and validation:

- Local KV cache addresses are represented as raw integer addresses plus sizes, not ordinary tensors.
- `addr_list` entries may contain multiple discontiguous memory spans.
- HCCL/PyTorch collectives operate on tensors, so the implementation needs either tensor views over local KV buffers or staging tensors plus explicit local copies.
- Failure states from a single owner get must be broadcast to peers before invalid-block fallback decisions.
- Async and layerwise variants are structurally separate.
- Performance validation requires multi-rank hardware runs and instrumentation of backend get counts.

## Main Risks

### Raw-address distribution

The backend read target is a list of raw addresses and sizes. If the implementation cannot safely create tensor views over those exact address ranges, it needs staging buffers and explicit memory copies. Staging is safer but adds local copy overhead.

### Non-contiguous segments

One logical key may map to multiple `(addr, size)` segments, especially across KV entries/layers or TP mismatch. Broadcast needs to preserve segment boundaries or pack/unpack bytes deterministically.

### Collective synchronization

All ranks in the `put_step` group must participate in the same collectives in the same order. Branches based on request metadata must be identical across ranks, or the system can deadlock. The guard condition must be purely topology/config based plus request-level metadata that is already synchronized.

### Async thread interaction

Running torch distributed collectives inside background recv threads can be risky if other model code uses the same process group concurrently. If supported in this codebase, it must use strict ordering and likely a dedicated subgroup. Otherwise async dedup should initially be disabled.

### Layerwise scheduling

Layerwise load may overlap layer computation. Adding group communication in that pipeline could reduce overlap or introduce waits. It needs separate performance measurement.

## Recommended Scope for First PR

Recommended first PR scope:

- Implement synchronous, non-layerwise MLA read dedup for `use_mla=True`, `put_step>1`, `tp_mismatch=False`, `use_layerwise=False`, and `load_async=False`.
- Add a feature flag in `kv_connector_extra_config`, default enabled only for this narrow safe scope or default disabled if maintainers prefer staged rollout.
- Keep async and layerwise paths unchanged but explicitly guarded, logged, and documented as future work unless product deployment requires them now.
- Preserve TP mismatch path unchanged.

If the acceptance scope requires async/layerwise in the same PR, expand the implementation after the synchronous helper is validated.

## Expected Performance Impact

For MLA and TP=N where `put_step=N`, backend get call count should drop from N calls per request per rank group to 1 call per request per rank group. Expected remote I/O call reduction:

- TP=2: about 50% fewer backend get calls.
- TP=4: about 75% fewer backend get calls.
- TP=8: about 87.5% fewer backend get calls.

End-to-end latency improvement depends on the ratio of KV pool get overhead versus local broadcast/copy overhead. The benefit should be most visible when:

- MLA payload is small enough that fixed get overhead dominates.
- TP is high.
- Long-context prefix loads create many get keys.
- Backend latency is non-trivial compared with HCCL intra-node bandwidth.

For very small loads, collective setup overhead may offset gains. For very large loads, HCCL broadcast bandwidth should usually beat repeated remote get, but staging/copy overhead must be measured.
