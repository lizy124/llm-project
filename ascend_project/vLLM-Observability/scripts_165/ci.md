Run PYTHONPATH="$PYTHONPATH:$(pwd)/vllm-empty"
  PYTHONPATH="$PYTHONPATH:$(pwd)/vllm-empty"
  export PYTHONPATH
  git config --global --add safe.directory "$GITHUB_WORKSPACE"
  for python_version in "3.10" "3.11" "3.12"; do
    echo "============================"
    tools/mypy.sh 1 "$python_version"
    echo "============================"
  done
  shell: bash -el {0}
Run '/home/runner/k8s/index.js'
  shell: /home/runner/externals/node20/bin/node {0}
(node:2523) [DEP0005] DeprecationWarning: Buffer() is deprecated due to security and usability issues. Please use the Buffer.alloc(), Buffer.allocUnsafe(), or Buffer.from() methods instead.
(Use `node --trace-deprecation ...` to show where the warning was created)
============================
Running mypy for vllm_ascend on python version: 3.10
Error: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:217: error: Item "None" of "KVPoolWorker | None" has no attribute "set_external_slot_release_waiter"  [union-attr]
Error: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:246: error: Item "None" of "KVPoolWorker | None" has no attribute "wait_for_layer_load"  [union-attr]
Error: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:257: error: Item "None" of "KVPoolWorker | None" has no attribute "save_kv_layer"  [union-attr]
Error: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:267: error: Item "None" of "KVPoolWorker | None" has no attribute "wait_for_save"  [union-attr]
Error: vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:290: error: Item "None" of "KVPoolWorker | None" has no attribute "get_kv_events"  [union-attr]
Found 5 errors in 1 file (checked 499 source files)
Error: Error: failed to run script step: Error: command terminated with non-zero exit code: command terminated with exit code 1
Error: Process completed with exit code 1.
Error: Executing the custom container implementation failed. Please contact your self hosted runner administrator.