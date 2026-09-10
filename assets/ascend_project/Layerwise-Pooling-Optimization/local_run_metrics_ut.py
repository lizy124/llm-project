"""Local-only driver: run test_metrics.py on a vllm-less machine.

_mock_deps stuffs fake packages into sys.modules without wiring parent
attributes (the real import machinery does setattr on parents), so
mock.patch dotted-path resolution fails locally. On the server (vllm
installed) the real-import branch wires them. This driver mirrors the
real machinery before loading the tests. Not part of the repo test suite.
"""

import importlib
import sys
import unittest

REPO = r"D:\lzy\project\kv_pool\code\vllm-ascend"
sys.path.insert(0, REPO)

import tests.ut.distributed.ascend_store._mock_deps  # noqa: E402

m = sys.modules


def wire(parent, child_name, child_full):
    if not hasattr(m[parent], child_name):
        setattr(m[parent], child_name, m[child_full])


wire("vllm_ascend", "distributed", "vllm_ascend.distributed")
wire("vllm_ascend.distributed", "kv_transfer", "vllm_ascend.distributed.kv_transfer")
wire("vllm_ascend.distributed", "utils", "vllm_ascend.distributed.utils")
wire("vllm_ascend.distributed.kv_transfer", "kv_pool", "vllm_ascend.distributed.kv_transfer.kv_pool")
wire("vllm_ascend.distributed.kv_transfer", "utils", "vllm_ascend.distributed.kv_transfer.utils")
wire(
    "vllm_ascend.distributed.kv_transfer.kv_pool",
    "ascend_store",
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store",
)
wire(
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store",
    "backend",
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend",
)

importlib.import_module("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker")
importlib.import_module("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler")

import tests.ut.distributed.ascend_store as _as_pkg  # noqa: E402

_test_modules = [
    "test_ascend_store_connector",
    "test_backend",
    "test_coordinator",
    "test_kv_transfer",
    # "test_layerwise_cache_layout": needs unmocked vllm.utils.torch_utils;
    # run this one on the server where real vllm is installed.
    "test_metadata",
    "test_metrics",
    "test_pool_scheduler",
    "test_pool_worker",
]

loader = unittest.TestLoader()
suite = unittest.TestSuite()
for name in _test_modules:
    mod = importlib.import_module(f"tests.ut.distributed.ascend_store.{name}")
    suite.addTests(loader.loadTestsFromModule(mod))
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
