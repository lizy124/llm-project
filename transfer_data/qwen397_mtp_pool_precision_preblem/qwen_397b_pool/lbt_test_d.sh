#!/bin/bash
# D (Decode): NPU 8-15, 2 DP x 4 TP, ports 8004-8005, dp-rpc 12322

bash run_dp_template_d.sh 8,9,10,11 8004 2 0 90.90.97.27 12322 4 &
bash run_dp_template_d.sh 12,13,14,15 8005 2 1 90.90.97.27 12322 4 &

wait