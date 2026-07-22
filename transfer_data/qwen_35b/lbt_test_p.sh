#!/bin/bash
# Launch P (Prefill) side: 4 DP × 2 TP = 8 GPUs (0-7), ports 8000-8003
# Run this first, then run lbt_test_d.sh in another terminal

python launch_online_dp.py \
  --dp-size 8 \
  --tp-size 2 \
  --dp-size-local 4 \
  --dp-rank-start 0 \
  --dp-address 141.61.81.162 \
  --dp-rpc-port 12321 \
  --vllm-start-port 8000 \
  --template run_dp_template_p.sh