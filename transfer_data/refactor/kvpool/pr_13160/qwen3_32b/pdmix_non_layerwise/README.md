# Qwen3-32B PD-Mixed Non-Layerwise KV Pool Test

## Goal

Validate PR #13160 with Qwen3-32B in a PD-mixed deployment using AscendStore KV Pool with the Memcache backend and the normal non-layerwise save/load path.

This scenario verifies that a single vLLM service can act as both Prefill and Decode (`kv_role=kv_both`), store KV through `AscendStoreConnector`, and reuse KV for repeated-prefix requests without enabling layerwise KV Pool.

## References

- Scenario rules: `/home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/README.md`
- Qwen3 Dense official deployment doc: `/home/lizhongyang/docs/source/tutorials/models/Qwen3-Dense.md`
- KV Pool official deployment doc: `/home/lizhongyang/docs/source/user_guide/feature_guide/kv_pool.md`
- Reference-only external validation note: `/home/lizhongyang/llm-project/transfer_data/refactor/kvpool/tmp/test.md`

## Model

Default model path:

```text
/mnt/share/modelscope/hub/models/Qwen/Qwen3-32B
```

The local model directory has `config.json`, tokenizer files, weight files, and a weight index. Its `config.json` reports:

- architecture: `Qwen3ForCausalLM`
- model type: `qwen3`
- dtype: `bfloat16`
- layers: 64
- max positions: 40960

Official Qwen3 Dense documentation says Qwen3-32B BF16 can run on one Atlas A3/A2 inference node and commonly uses TP across 4 x 64GB cards for single-node deployment.

## Deployment Mode

- Topology: PD-mixed, one service handles both Prefill and Decode.
- KV connector: `AscendStoreConnector`.
- KV role: `kv_both`.
- Backend: `mooncake` or `memcache` through `KV_BACKEND`; the validated run used `mooncake`.
- Layerwise: disabled. The KV config does not set `use_layerwise: true`.
- Default devices: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`.
- Default TP: `4`.
- Default port: `8004`.
- Served model name: `qwen3-32b-kvpool`.

## Prerequisites

Before this scenario can produce a PASS result, the environment must satisfy all of the following:

- `/vllm-workspace/vllm-ascend` is checked out to PR #13160.
- `vllm` and `vllm-ascend` import from the editable source paths used by this container.
- The model path exists and contains a complete Hugging Face style model.
- At least 4 target NPUs are free enough for Qwen3-32B BF16 TP4.
- For `KV_BACKEND=mooncake`, Mooncake Python bindings and a reachable Mooncake master are available.
- For `KV_BACKEND=memcache`, Memcache and MemFabric runtime packages are installed and `MMC_LOCAL_CONFIG_PATH` points to a valid `mmc-local.conf` that can reach a running Memcache MetaService/ConfigStore.
- The selected service port is free.

During validation, `KV_BACKEND=mooncake` passed using the existing Mooncake master at `127.0.0.1:50088`. The `memcache-hybrid==1.1.4` and `memfabric-hybrid==1.1.4` Python packages were present, but their package-provided `mmc-local.conf` pointed to `127.0.0.1:5000` and `127.0.0.1:6000`, where no Memcache MetaService/ConfigStore was serving. Therefore the memcache backend remained BLOCKED in this container.

## How To Run

Start the service with the Mooncake backend validated in this environment:

```bash
cd /home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_32b/pdmix_non_layerwise
KV_BACKEND=mooncake VLLM_BATCH_INVARIANT=1 MOONCAKE_MASTER=127.0.0.1:50088 bash start.sh
```

`VLLM_BATCH_INVARIANT=1` disables custom ops in this environment and avoids the missing `npu_add_rms_norm_bias` operator path.

Run validation after the service is ready:

```bash
cd /home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_32b/pdmix_non_layerwise
bash test.sh
```

Useful overrides:

```bash
MODEL_PATH=/mnt/share/modelscope/hub/models/Qwen/Qwen3-32B \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
TENSOR_PARALLEL_SIZE=4 \
SERVER_PORT=8004 \
MMC_LOCAL_CONFIG_PATH=/path/to/mmc-local.conf \
bash start.sh
```

If Memcache env scripts are installed outside `/usr/local`, set them explicitly:

```bash
MEMCACHE_ENV=/path/to/memcache_hybrid/set_env.sh \
MEMFABRIC_ENV=/path/to/memfabric_hybrid/set_env.sh \
MMC_LOCAL_CONFIG_PATH=/path/to/mmc-local.conf \
bash start.sh
```

## Pass Criteria

PASS requires all of the following:

- `start.sh` starts the vLLM OpenAI server and records the PID, command, environment, git commit, and logs under `results/latest`.
- The service readiness endpoint responds successfully.
- The startup log confirms the AscendStore KV Pool backend path initialized.
- `test.sh` sends at least one smoke request and two repeated-prefix requests successfully.
- Repeated-prefix responses are non-empty and valid JSON responses from the served model.
- Logs contain KV Pool/AscendStore backend activity.
- Logs do not contain traceback, fatal Python exceptions, NPU errors, HCCL errors, or backend store initialization failures.

## Result States

- `PASS`: service startup and all required assertions pass.
- `FAIL`: service starts but request correctness, KV Pool activity, or log assertions fail.
- `BLOCKED`: missing model, occupied NPUs, missing Memcache/MemFabric setup, invalid Memcache config, unavailable port, or any other environment issue prevents judging PR behavior.
