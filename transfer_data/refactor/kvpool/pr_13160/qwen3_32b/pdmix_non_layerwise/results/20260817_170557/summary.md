# Qwen3-32B PDMix Non-Layerwise KV Pool Run Summary

## Result

Status: BLOCKED

The service did not reach readiness, so `test.sh` was not executed.

## Attempted Command

The latest start attempt used:

```bash
cd /home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_32b/pdmix_non_layerwise
VLLM_BATCH_INVARIANT=1 bash start.sh
```

`VLLM_BATCH_INVARIANT=1` was added after the first attempt failed because the current CANN/libopapi did not provide `aclnnAddRmsNormBias` / `npu_add_rms_norm_bias`. With custom ops disabled, the model progressed past that issue and reached KV Pool backend initialization.

## Root Cause

The second attempt failed during AscendStore Memcache backend setup:

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py:96
assert res == 0
AssertionError
```

The active Memcache local config was:

```text
/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-local.conf
```

It points to:

```text
ock.mmc.meta_service_url = tcp://127.0.0.1:5000
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6000
```

Port checks after failure showed:

```text
127.0.0.1:5000 closed_or_unreachable ConnectionRefusedError
127.0.0.1:6000 closed_or_unreachable ConnectionRefusedError
```

No Memcache/MetaService/ConfigStore process was found by process scan.

## Interpretation

This is an environment BLOCKED result, not evidence that PR #13160 failed. Qwen3-32B model loading started successfully, and the PR code path reached `AscendStoreConnector` / `KVPoolWorker` / `MemcacheBackend` initialization. The run cannot validate KV Pool save/load until a reachable Memcache MetaService and ConfigStore are running and `MMC_LOCAL_CONFIG_PATH` points to their matching local config.

## Artifacts

- Server log: `server.log`
- Startup command: `command.sh`
- Environment snapshot: `env.txt`
- Status file: `status.txt`
