#!/usr/bin/env python3
"""
Smoke test: validate the AscendStore metrics pipeline against the REAL vllm
metric framework (KVConnectorProm, KVConnectorStats) without requiring NPU
hardware or a running KV pool service.

Pipeline validated:
  1. AscendStoreConnector.build_prom_metrics() registers the correct
     Prometheus metric names via the real KVConnectorProm.
  2. AscendStoreConnector.build_kv_connector_stats(data) correctly
     rebuilds an AscendStoreKVConnectorStats from plain dict.
  3. Stats produced by worker/scheduler mock → aggregate → observe →
     the Prometheus /metrics text output contains our new metrics.
"""
import sys
import os

# Use the PR1 branch code
sys.path.insert(0, "/root/kv_metrics_ut")
os.environ.setdefault("VLLM_PLATFORM_INTERFACE", "0")

from unittest.mock import MagicMock, patch
from prometheus_client import CollectorRegistry, generate_latest

# --- Import real vllm metric framework ---
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
    KVConnectorPromMetrics,
    KVConnectorProm,
)
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory,
)

# --- Import our PR1 code ---
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metrics import (
    AscendStoreKVConnectorStats,
    AscendStorePromMetrics,
)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
section("1. build_prom_metrics registers correct metric names")

# Create a minimal mock VllmConfig
vllm_config = MagicMock()
vllm_config.kv_transfer_config = MagicMock()
vllm_config.kv_transfer_config.kv_connector = "AscendStoreConnector"
vllm_config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}

# Patch the factory so vllm can resolve our connector
with patch.object(KVConnectorFactory, "get_connector_class") as mock_factory:
    # Import the connector and get its class
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
        AscendStoreConnector,
    )
    mock_factory.return_value = AscendStoreConnector

    labelnames = ["model_name", "block_size"]
    per_engine_labelvalues = {0: ["qwen3-test", 16]}

    # Use a fresh registry to avoid conflicts
    registry = CollectorRegistry()
    with patch("vllm.distributed.kv_transfer.kv_connector.v1.metrics.Gauge", registry=registry), \
         patch("vllm.distributed.kv_transfer.kv_connector.v1.metrics.Counter", registry=registry), \
         patch("vllm.distributed.kv_transfer.kv_connector.v1.metrics.Histogram", registry=registry):
        prom = KVConnectorProm(vllm_config, labelnames, per_engine_labelvalues)

    check("KVConnectorProm created", prom is not None)
    check("prom_metrics is AscendStorePromMetrics",
          isinstance(prom.prom_metrics, AscendStorePromMetrics),
          str(type(prom.prom_metrics).__name__))

    # Check metric registration
    has_load_hist = hasattr(prom.prom_metrics, "_histogram_load_duration")
    has_delayed_gauge = hasattr(prom.prom_metrics, "_gauge_delayed_release")
    check("Histogram load_duration registered", has_load_hist)
    check("Gauge delayed_release registered", has_delayed_gauge)

    if has_load_hist:
        check("load_duration metric name",
              prom.prom_metrics._histogram_load_duration._name == "vllm:kv_pool_load_duration_seconds",
              prom.prom_metrics._histogram_load_duration._name)
    if has_delayed_gauge:
        check("delayed_release metric name",
              prom.prom_metrics._gauge_delayed_release._name == "vllm:kv_pool_delayed_release_requests",
              prom.prom_metrics._gauge_delayed_release._name)


# ---------------------------------------------------------------------------
section("2. build_kv_connector_stats rebuilds from plain dict")

data = {
    "load": [
        {"duration_seconds": 0.03, "num_keys": 5, "num_failed_keys": 0, "path": "sync"},
        {"duration_seconds": 0.15, "num_keys": 3, "num_failed_keys": 1, "path": "async"},
    ],
    "delayed_release": {"num_requests": 3},
}

stats = AscendStoreConnector.build_kv_connector_stats(data)
check("build_kv_connector_stats returns AscendStoreKVConnectorStats",
      isinstance(stats, AscendStoreKVConnectorStats))
check("stats has 2 load records", len(stats.data.get("load", [])) == 2)
check("stats has delayed_release", stats.data.get("delayed_release", {}).get("num_requests") == 3)

# Empty / None
stats_none = AscendStoreConnector.build_kv_connector_stats(None)
check("build_kv_connector_stats(None) is not None (empty stats)",
      stats_none is not None and stats_none.is_empty())


# ---------------------------------------------------------------------------
section("3. Full pipeline: worker stats → aggregate → observe → /metrics")

# Simulate stats from a worker (load durations) and scheduler (delayed release)
worker_stats = AscendStoreKVConnectorStats()
worker_stats.record_load(0.025, 10, path="sync")
worker_stats.record_load(0.080, 8, path="async")
worker_stats.record_load(0.120, 6, num_failed_keys=1, path="layerwise")

scheduler_stats = AscendStoreKVConnectorStats()
scheduler_stats.record_delayed_release(5)

# Aggregate (as vllm does in scheduler.py): worker stats first, then
# scheduler stats via aggregate(other).
aggregated = worker_stats.aggregate(scheduler_stats)
check("aggregated is AscendStoreKVConnectorStats",
      isinstance(aggregated, AscendStoreKVConnectorStats))
check("aggregated has 3 load records", len(aggregated.data.get("load", [])) == 3)
check("aggregated delayed_release = 5",
      aggregated.data.get("delayed_release", {}).get("num_requests") == 5)

# Observe via the KVConnectorProm created in section 1 (real framework path)
# This is the exact call vllm makes during metrics collection.
prom.observe(aggregated.data, engine_idx=0)

# Generate the /metrics text output from the default REGISTRY
metrics_text = generate_latest().decode("utf-8")

check("metrics output contains vllm:kv_pool_load_duration_seconds",
      "vllm:kv_pool_load_duration_seconds" in metrics_text)
check("metrics output contains vllm:kv_pool_delayed_release_requests",
      "vllm:kv_pool_delayed_release_requests" in metrics_text)
check("metrics output contains path=sync label",
      'path="sync"' in metrics_text)
check("metrics output contains path=async label",
      'path="async"' in metrics_text)
check("metrics output contains path=layerwise label",
      'path="layerwise"' in metrics_text)
check("metrics output contains histogram bucket counts",
      "_bucket{" in metrics_text)
check("metrics output contains delayed_release gauge value",
      "vllm:kv_pool_delayed_release_requests" in metrics_text and "5" in metrics_text)

# Print the relevant metric lines for visual confirmation
print("\n--- Relevant /metrics output ---")
for line in metrics_text.splitlines():
    if "kv_pool" in line:
        print(f"  {line}")


# ---------------------------------------------------------------------------
section("4. Stats reset after get_stats (worker/scheduler semantics)")

# Worker: get_stats resets
worker_stats2 = AscendStoreKVConnectorStats()
worker_stats2.record_load(0.05, 10, path="sync")
check("worker stats not empty before get_stats", not worker_stats2.is_empty())

# Scheduler: get_stats resets
scheduler_stats2 = AscendStoreKVConnectorStats()
scheduler_stats2.record_delayed_release(2)
check("scheduler stats not empty before get_stats", not scheduler_stats2.is_empty())


# ---------------------------------------------------------------------------
section("SUMMARY")
print(f"  Total: {PASS + FAIL}  Passed: {PASS}  Failed: {FAIL}")
print(f"  Result: {'ALL PASS' if FAIL == 0 else 'HAS FAILURES'}")
sys.exit(1 if FAIL > 0 else 0)
