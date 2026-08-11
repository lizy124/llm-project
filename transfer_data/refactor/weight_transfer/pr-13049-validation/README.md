# PR 13049 validation bundle

This directory records the actual validation process used for vllm-ascend PR 13049.

## What was verified

- static checks on the PR diff
- weight_transfer UTs
- one-card NPU IPC e2e
- two-card HCCL e2e

## Results

- static checks: passed
- UT: passed
- one-card NPU IPC: passed
- two-card HCCL: passed

## Notes

- The validation was run on the current Ascend server.
- The local model path `/mnt/weight/Qwen3-0.6B` was used for offline e2e.
- The e2e process was first attempted with an invalid trainer logical device and then rerun successfully.
- Example CLI smoke for `examples/rl/rlhf_http_hccl.py` and `examples/rl/rlhf_http_npu_ipc.py` remains a follow-up item.

## Files

- `run.sh`: the exact validation flow used during this session
- `env.txt`: environment snapshot for the validation session
- `summary.json`: structured result summary
- `log_extract.txt`: human-readable result extract
- `static_checks.log`: static-check command output
- `unit_tests.log`: unit-test command output
- `e2e_one_card_npu_ipc.log`: first NPU IPC attempt
- `e2e_one_card_npu_ipc_offline.log`: offline NPU IPC retry
- `e2e_one_card_npu_ipc_custom.log`: final NPU IPC run
- `e2e_two_card_hccl_custom.log`: first HCCL attempt
- `e2e_two_card_hccl_custom_retry.log`: final HCCL run
