# Qwen3.5-397B-A17B PD-Mixed Non-Layerwise KV Pool Test

## Goal

Validate PR #13160 with Qwen3.5-397B-A17B in a PD-mixed deployment using AscendStore KV Pool with the Mooncake backend and the normal non-layerwise save/load path.

This scenario verifies that a single vLLM service can act as both Prefill and Decode (`kv_role=kv_both`), store KV through `AscendStoreConnector`, and reuse KV for repeated-prefix requests without enabling layerwise KV Pool.

## References

- Scenario rules: `/home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/README.md`
- Qwen3.5-397B official deployment doc: `/home/lizhongyang/docs/source/tutorials/models/Qwen3.5-397B-A17B.md`
- KV Pool official deployment doc: `/home/lizhongyang/docs/source/user_guide/feature_guide/kv_pool.md`
- Mooncake PD-mixed example: `/home/lizhongyang/docs/source/user_guide/feature_guide/kv_pool.md`
- Validated Qwen3-32B reference: `/home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_32b/pdmix_non_layerwise`

## Model

Default model path:

```text
/mnt/share/modelscope/hub/models/Qwen/Qwen3.5-397B-A17B
```

The local model directory has `config.json`, tokenizer files, weight files, and a weight index. Its `config.json` reports:

- architecture: `Qwen3_5MoeForConditionalGeneration`
- model type: `qwen3_5_moe`
- text dtype: `bfloat16`
- text layers: 60
- max positions: 262144
- experts: 512 total, 10 experts per token

The official Qwen3.5-397B documentation states that the BF16 model requires 2 Atlas 800 A3 nodes with 16 x 64GB cards per node, or 4 Atlas 800 A2 nodes with 8 x 64GB cards per node. The W8A8 MTP variant can run on fewer nodes, but that quantized model is not present in the local model cache used by this scenario.

## Deployment Mode

- Topology: PD-mixed, one service handles both Prefill and Decode.
- KV connector: `AscendStoreConnector`.
- KV role: `kv_both`.
- Backend: `mooncake` by default.
- Layerwise: disabled. The KV config does not set `use_layerwise: true`.
- Default devices: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`.
- Default TP: `16`.
- Default port: `8005`.
- Served model name: `qwen3.5-397b-kvpool`.

Qwen3.5 is a hybrid attention model. The KV Pool official document says `kv_load_failure_policy=recompute` is not supported for hybrid attention models such as Qwen3.5, so this scenario uses the default fail behavior unless `KV_LOAD_FAILURE_POLICY` is explicitly overridden.

## Prerequisites

Before this scenario can produce a PASS result, the environment must satisfy all of the following:

- `/vllm-workspace/vllm-ascend` is checked out to PR #13160.
- `vllm` and `vllm-ascend` import from the editable source paths used by this container.
- The model path exists and contains a complete Hugging Face style model.
- Enough A3/A2 NPU resources are available for Qwen3.5-397B BF16. If only one 16-card A3 node is visible, startup is expected to be BLOCKED or fail with an HBM/OOM resource error.
- Mooncake Python bindings and a reachable Mooncake master are available. The Qwen3-32B reference passed with an existing Mooncake master at `127.0.0.1:50088`.
- The selected service port is free.

## How To Run

Start the service with the Mooncake backend:

```bash
cd /home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_5/pdmix_non_layerwise
KV_BACKEND=mooncake MOONCAKE_MASTER=127.0.0.1:50088 bash start.sh
```

Run validation after the service is ready:

```bash
cd /home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_5/pdmix_non_layerwise
bash test.sh
```

Useful overrides:

```bash
MODEL_PATH=/mnt/share/modelscope/hub/models/Qwen/Qwen3.5-397B-A17B \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
TENSOR_PARALLEL_SIZE=16 \
SERVER_PORT=8005 \
MAX_MODEL_LEN=16384 \
MAX_NUM_BATCHED_TOKENS=4096 \
MAX_NUM_SEQS=8 \
bash start.sh
```

If testing a quantized 397B variant later, set the quantization and model path explicitly:

```bash
MODEL_PATH=/path/to/Qwen3.5-397B-A17B-w8a8-mtp \
QUANTIZATION=ascend \
bash start.sh
```

## Pass Criteria

PASS requires all of the following:

- `start.sh` starts the vLLM OpenAI server and records the PID, command, environment, git commit, and logs under `results/latest`.
- The service readiness endpoint responds successfully.
- The startup log confirms the AscendStore KV Pool backend path initialized.
- `test.sh` sends at least one smoke request and two repeated-prefix requests successfully.
- Repeated-prefix responses are non-empty and valid JSON responses from the served model.
- Logs contain KV Pool/AscendStore/Mooncake backend activity.
- Logs do not contain traceback, fatal Python exceptions, NPU errors, HCCL errors, OOM/resource errors, or backend store initialization failures.

## Result States

- `PASS`: service startup and all required assertions pass.
- `FAIL`: service starts but request correctness, KV Pool activity, or log assertions fail.
- `BLOCKED`: model, NPU capacity, Mooncake master, port, runtime packages, or other environment requirements prevent judging PR behavior.
