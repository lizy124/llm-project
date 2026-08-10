Run git config --global --add safe.directory "$GITHUB_WORKSPACE"
Run '/home/runner/k8s/index.js'
(node:1690) [DEP0005] DeprecationWarning: Buffer() is deprecated due to security and usability issues. Please use the Buffer.alloc(), Buffer.allocUnsafe(), or Buffer.from() methods instead.
(Use `node --trace-deprecation ...` to show where the warning was created)
ruff check..................................................................Failed
- hook id: ruff-check
- files were modified by this hook
ruff format.................................................................Passed
codespell...................................................................Passed
typos.......................................................................Passed
clang-format................................................................Passed
markdownlint................................................................Passed
Lint GitHub Actions workflow files..........................................Passed
Gitleaks Secret Scan(Local Binary)..........................................Passed
Lint shell scripts..........................................................Passed
Lint PNG exports from excalidraw............................................Passed
Check for spaces in all filenames...........................................Passed
Enforce __init__.py in Python packages......................................Passed
Forbid init_logger(__name__) in vllm_ascend modules.........................Passed
Check for forbidden imports.................................................Passed
Check for boolean ops in with-statements....................................Passed
Check that new functions over 100 lines have comments.......................Passed
Check symbolic shapes in meta kernels.......................................Passed
Suggestion..................................................................Passed
- hook id: suggestion
- duration: 0s

To bypass pre-commit hooks, add --no-verify to git commit.

pre-commit hook(s) made changes.
If you are seeing this message in CI, reproduce locally with: `pre-commit run --all-files`.
To run `pre-commit` as part of git workflow, use `pre-commit install`.
All changes made by hooks:
diff --git a/tests/ut/distributed/mooncake/test_mooncake_kv_transfer.py b/tests/ut/distributed/mooncake/test_mooncake_kv_transfer.py
index d35931eed..994e2ea42 100644
--- a/tests/ut/distributed/mooncake/test_mooncake_kv_transfer.py
+++ b/tests/ut/distributed/mooncake/test_mooncake_kv_transfer.py
@@ -7,14 +7,14 @@ import torch
 if not hasattr(torch, "npu"):
     torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]
 
+from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
+    KVCacheStoreSendingThread,
+)
 from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
     ChunkedTokenDatabase,
     KeyMetadata,
     ReqMeta,
 )
-from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
-    KVCacheStoreSendingThread,
-)
 
 
 class _FakeStore:
diff --git a/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py b/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
index ffb259278..1b1b38e6e 100644
--- a/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
+++ b/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
@@ -27,6 +27,12 @@ from vllm.v1.serial_utils import MsgpackEncoder
 from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
     backend_map,
 )
+from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
+    build_layerwise_cache_layout,
+    build_layerwise_reuse_layout,
+    get_gva_layerwise_config,
+    get_layerwise_kv_cache_specs,
+)
 from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
     AscendConnectorMetadata,
     AscendStoreKVConnectorWorkerMetadata,
@@ -43,12 +49,6 @@ from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
     infer_tp_mismatch_info,
     normalize_block_ids_by_group,
 )
-from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
-    build_layerwise_cache_layout,
-    build_layerwise_reuse_layout,
-    get_gva_layerwise_config,
-    get_layerwise_kv_cache_specs,
-)
 
 
 class KVPoolScheduler:
Error: Error: failed to run script step:
  ✗ exit code: 1
  → your script exited with a non-zero code; please check your script for errors
------------------------------------------------------------
Last output:
  +    build_layerwise_reuse_layout,
  +    get_gva_layerwise_config,
  +    get_layerwise_kv_cache_specs,
  +)
   from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
       AscendConnectorMetadata,
       AscendStoreKVConnectorWorkerMetadata,
  @@ -43,12 +49,6 @@ from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
       infer_tp_mismatch_info,
       normalize_block_ids_by_group,
   )
  -from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
  -    build_layerwise_cache_layout,
  -    build_layerwise_reuse_layout,
  -    get_gva_layerwise_config,
  -    get_layerwise_kv_cache_specs,
  -)
   
   
   class KVPoolScheduler:
Error: Process completed with exit code 1.
Error: Executing the custom container implementation failed. Please contact your self hosted runner administrator.