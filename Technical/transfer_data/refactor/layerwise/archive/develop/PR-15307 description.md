# PR #15307 描述(合并口径终版)

> 用途:直接粘贴到 https://github.com/vllm-project/vllm-ascend/pull/15307 的编辑框。
> 标题需同步修改:`[Refactor] (kv_pool): converge layerwise GVA protocol and transfer threads into backend`
> #15277 保持 open,不在此描述中催促关闭,仅声明包含关系(见 Supersedes 行)。

---

### What this PR does / why we need it?

Converges the memcache-specific layerwise GVA logic scattered across the
generic layers (pool_worker / pool_scheduler / kv_transfer) into the
backend package, so backend work no longer touches framework code.
Behavior-preserving re-encapsulation: key formats, log messages, event
ordering, and the prepare-load -> alloc-save ordering are kept
byte-for-byte.

- `backend/gva_protocol.py` (new): `GVAKeyFactory` / `GVASession` /
  `GVAHitChecker`, relocated from pool_worker / pool_scheduler
- `backend/gva_threads.py` (new): `LayerBatchBuilder` + GVA layerwise
  send/recv threads, relocated from kv_transfer.py; the worker wires them
  via `GVALayerwiseThreadContext` + factories, start/wait and
  send-before-recv ordering preserved
- `backend/__init__.py`: capability registry with a single
  `use_gva_layerwise()` derivation (4 duplicated derivations converged);
  also fixes a live regression from #14465 — the connector-side
  derivation was deleted as dead code while AscendMultiConnector still
  read it, crashing PD-disconnect init with AttributeError
- `backend/base.py`: pure generic data plane; GVA methods split into a
  `GVALayerwiseCapable` ABC (incl. `batch_copy`); no `.store.` access
  outside `backend/`
- `backend/memcache_backend.py`: dual inheritance + `on_worker_ready()`
  lifecycle hook (lazy_init all-miss short-circuit contract preserved)
- generic layers: GVA code removed and delegated (~1.4k lines deleted);
  kv_transfer.py keeps only the generic base + key-mode threads
- mooncake / yuanrong backends: zero changes

Supersedes #15277 (all 5 commits included; it can be closed once this is
approved — kept open for now).

Review note: ~1.35k UT lines, ~1.5k relocated production lines (deleted
1:1 from the generic files, verifiable by mechanical old-vs-new diff),
~250 genuinely new production lines.

### Does this PR introduce _any_ user-facing change?

NO.

### How was this patch tested?

- ascend_store UT: 314 passed, 2 failed (pre-existing `test_coordinator`
  stub failures, same as base)
- `test_gva_protocol.py`: key-format snapshots, lazy-init short-circuit
  contracts, hit-check group-min semantics, #14465 regression case
- `test_gva_threads.py`: load-path non-zero-GVA probe, factory
  parameter-mapping assertions
- ruff clean; mypy clean
- Server smoke tests (MultiConnector PD, memcache layerwise, mooncake
  non-layerwise): pending, results will be posted in comments
