# Qwen3-32B PDMix Non-Layerwise KV Pool Run Summary

## Result

Status: PASS

The Qwen3-32B PDMix non-layerwise KV Pool scenario passed using the existing Mooncake master.

## Commands

Start service:

```bash
cd /home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_32b/pdmix_non_layerwise
KV_BACKEND=mooncake VLLM_BATCH_INVARIANT=1 MOONCAKE_MASTER=127.0.0.1:50088 bash start.sh
```

Run validation:

```bash
cd /home/lizhongyang/llm-project/transfer_data/refactor/kvpool/pr_13160/qwen3_32b/pdmix_non_layerwise
bash test.sh
```

## Environment

- Repo: `/vllm-workspace/vllm-ascend`
- Branch: `pr-13160`
- Commit: `c6f72551f1abddaa5f2025e88b7f9c164f70943d`
- Model: `/mnt/share/modelscope/hub/models/Qwen/Qwen3-32B`
- Served model name: `qwen3-32b-kvpool`
- Backend: `mooncake`
- Mooncake master: `127.0.0.1:50088`
- Mooncake config: `mooncake.json` in this result directory
- Devices: `0,1,2,3`
- TP: `4`
- Port: `8004`
- Custom op workaround: `VLLM_BATCH_INVARIANT=1`

`VLLM_BATCH_INVARIANT=1` is required in this environment because a direct attempt without it failed during model profile with missing `npu_add_rms_norm_bias` / `aclnnAddRmsNormBias` support in the current CANN/libopapi.

## Validation Output

```text
models endpoint ok
smoke request ok
first repeated-prefix request ok elapsed=5.52s
second repeated-prefix request ok elapsed=3.55s
usage1 {"prompt_tokens": 3458, "total_tokens": 3474, "completion_tokens": 16, "prompt_tokens_details": null}
usage2 {"prompt_tokens": 3458, "total_tokens": 3474, "completion_tokens": 16, "prompt_tokens_details": null}
PASS: Qwen3-32B PDMix non-layerwise KV Pool smoke and repeated-prefix validation passed
```

## Artifacts

- Server log: `server.log`
- Test log: `test.log`
- Startup command: `command.sh`
- Environment snapshot: `env.txt`
- Mooncake config: `mooncake.json`
- Smoke response: `smoke_response.json`
- Repeated-prefix responses: `prefix_response_1.json`, `prefix_response_2.json`
