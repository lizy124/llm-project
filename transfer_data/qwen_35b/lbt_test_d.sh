#!/bin/bash
# Launch D (Decode) side: 4 DP × 2 TP = 8 GPUs (8-15), ports 8004-8007
# Run this AFTER lbt_test_p.sh, in a separate terminal

python launch_online_dp.py \
  --dp-size 8 \
  --tp-size 2 \
  --dp-size-local 4 \
  --dp-rank-start 4 \
  --dp-address 141.61.81.162 \
  --dp-rpc-port 12321 \
  --vllm-start-port 8004 \
  --template run_dp_template_d.sh