# PR: KV pool load-duration and delayed-release metrics

- **branch**: `lizy124:kv_metrics_observability` → `vllm-project:main`
- **create**: https://github.com/vllm-project/vllm-ascend/compare/main...lizy124:vllm-ascend:kv_metrics_observability

---

## Title

[Observability] Add KV pool load-duration and delayed-release metrics for AscendStoreConnector

## What this PR does / why we need it

Hooks the AscendStoreConnector into the upstream KV connector metrics framework so pool behavior is observable via the standard `/metrics` endpoint.

Adds 4 Prometheus metrics:

- `vllm:kv_pool_load_duration_seconds` (Histogram, label: `path`=sync/async/layerwise) — per-request wall-clock duration of loading KV cache from the pool
- `vllm:kv_pool_load_keys_total` (Counter, label: `path`) — number of keys loaded per request
- `vllm:kv_pool_load_failed_keys_total` (Counter, label: `path`) — number of keys that failed to load
- `vllm:kv_pool_delayed_release_requests` (Gauge) — number of requests currently held in the delayed-release window

Histogram buckets: `[1ms, 5ms, 10ms, 25ms, 50ms, 75ms, 100ms, 200ms, 300ms, 500ms, 750ms, 1s, 2.5s, 5s]`

### Files changed

- `ascend_store/metrics.py` **(new)** — `AscendStoreKVConnectorStats` (collect/aggregate) + `AscendStorePromMetrics` (register & observe the 4 metrics above)
- `ascend_store/pool_worker.py` — wall-clock timing for sync/async/layerwise load paths; `get_stats()` returns and resets
- `ascend_store/kv_transfer.py` — async receive thread reports timing back to worker (incl. tp_mismatch path)
- `ascend_store/pool_scheduler.py` — snapshot `len(_delayed_free_req_ids)` in `build_connector_meta()`
- `ascend_store/ascend_store_connector.py` — implement `get_kv_connector_stats` / `build_kv_connector_stats` / `build_prom_metrics` hooks
- `tests/.../test_metrics.py` **(new)** — 21 unit tests
- `tests/.../_mock_deps.py` — metrics framework dependency mocks
- `tests/.../test_pool_worker.py` — fix `__new__` construction missing new attributes
- `tests/.../test_kv_transfer.py` — fix tp_mismatch mock return value

### Data flow

```
worker (load timing)  ─┐
                       ├─ get_kv_connector_stats() → aggregate → observe → /metrics
scheduler (delayed    ─┘
 release snapshot)
```

## Does this PR introduce any user-facing change?

Yes. Adds 4 Prometheus metrics available via the `/metrics` endpoint. No config change; no behavior change when metrics are not scraped.

## How was this patch tested?

**UT**: `pytest tests/ut/distributed/ascend_store/` → **311 passed**

**Integration smoke test** (Ascend 910 container, real vllm metrics framework): 22/22 checks passed. `/metrics` output contains all 4 metrics with correct path labels and values.

```
vllm:kv_pool_load_duration_seconds_bucket{...,path="sync",le="0.05"} 1.0
vllm:kv_pool_load_duration_seconds_sum{...,path="layerwise"} 0.12
vllm:kv_pool_load_keys_total{...,path="sync"} 10.0
vllm:kv_pool_load_failed_keys_total{...,path="layerwise"} 1.0
vllm:kv_pool_delayed_release_requests{...} 5.0
```
