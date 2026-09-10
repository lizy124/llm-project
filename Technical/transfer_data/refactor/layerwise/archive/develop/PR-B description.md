### What this PR does / why we need it?

Moves the GVA layerwise transfer threads out of the generic `kv_transfer.py` into `backend/gva_threads.py` (follow-up to #15277, which moved the GVA protocol). Pure code motion, no behavior change.

- `LayerBatchBuilder`, `KVCacheStoreLayerSendingThread`, `KVCacheStoreLayerRecvingThread` moved to `backend/gva_threads.py`
- `KVPoolWorker` constructs threads via `GVALayerwiseThreadContext` + factories; start/wait and send-before-recv ordering preserved
- `batch_copy` added to `GVALayerwiseCapable`; no `.store.` access outside `backend/`
- `kv_transfer.py` now holds only the generic base + key-mode threads

### Does this PR introduce _any_ user-facing change?

No.

### How was this patch tested?

- ascend_store UT: 314 passed, 2 failed (pre-existing `test_coordinator` stub failures, same as #15277 base)
- New `test_gva_threads.py`: migrated cases, load-path probe, factory parameter-mapping assertions
- ruff: clean
