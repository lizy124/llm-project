# GLM-5.2 W4A8 PD-Mixed Non-Layerwise KV Pool Test

## Goal

Validate PR #13160 with GLM-5.2 W4A8 in a PD-mixed deployment using AscendStore KV Pool with the Mooncake backend and the normal non-layerwise save/load path.

This scenario verifies that a single vLLM service can act as both Prefill and Decode (`kv_role=kv_both`), store KV through `AscendStoreConnector`, and reuse KV for repeated-prefix requests without enabling layerwise KV Pool.

## References

- Scenario rules: `/home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/README.md`
- GLM-5.2 official deployment doc: `/home/lizhongyang/docs/source/tutorials/models/GLM5.2.md`
- KV Pool official deployment doc: `/home/lizhongyang/docs/source/user_guide/feature_guide/kv_pool.md`
- Validated Qwen3-32B reference: `/home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_32b/pdmix_non_layerwise`

## Model

Default model path:

```text
/mnt/share/modelscope/hub/models/Eco-Tech/GLM-5_2-w4a8
```

The local model directory has `config.json`, tokenizer files, weight files, and a weight index. Its `config.json` reports:

- architecture: `GlmMoeDsaForCausalLM`
- model type: `glm_moe_dsa`
- dtype: `bfloat16`
- layers: 78
- max positions: 1048576
- routed experts: 256
- experts per token: 8

The official GLM-5.2 document shows the single-node A3 quantized deployment recipe for `GLM-5.2-w4a8c8`. This scenario uses the local `GLM-5_2-w4a8` path supplied for validation and keeps the same key serving settings: `--quantization ascend`, expert parallelism, safetensors prefetch, GLM tool/reasoning parsers, and DeepSeek-style MTP speculative decoding.

## Deployment Mode

- Topology: PD-mixed, one service handles both Prefill and Decode.
- KV connector: `AscendStoreConnector`.
- KV role: `kv_both`.
- Backend: `mooncake` by default.
- Layerwise: disabled. The KV config does not set `use_layerwise: true`.
- Default devices: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`.
- Default DP/TP: `DATA_PARALLEL_SIZE=2`, `TENSOR_PARALLEL_SIZE=8`.
- Default port: `8006`.
- Served model name: `glm-5.2-kvpool`.

## Prerequisites

Before this scenario can produce a PASS result, the environment must satisfy all of the following:

- `/vllm-workspace/vllm-ascend` is checked out to PR #13160.
- `vllm` and `vllm-ascend` import from the editable source paths used by this container.
- The model path exists and contains a complete Hugging Face style model.
- At least 16 target A3 NPUs are free enough for the default DP2 TP8 single-node quantized layout.
- Mooncake Python bindings and a reachable Mooncake master are available. The Qwen3-32B reference passed with an existing Mooncake master at `127.0.0.1:50088`.
- The selected service port is free.

## How To Run

Start the service with the Mooncake backend:

```bash
cd /home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/glm5_2/pdmix_non_layerwise
KV_BACKEND=mooncake MOONCAKE_MASTER=127.0.0.1:50088 bash start.sh
```

Run validation after the service is ready:

```bash
cd /home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/glm5_2/pdmix_non_layerwise
bash test.sh
```

Useful overrides:

```bash
MODEL_PATH=/mnt/share/modelscope/hub/models/Eco-Tech/GLM-5_2-w4a8 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
DATA_PARALLEL_SIZE=2 \
TENSOR_PARALLEL_SIZE=8 \
SERVER_PORT=8006 \
MAX_MODEL_LEN=135000 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_NUM_SEQS=12 \
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
