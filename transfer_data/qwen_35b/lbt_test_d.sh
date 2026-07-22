#!/bin/bash
# D (Decode): NPU 8-15, 4 DP x 2 TP, ports 8004-8007, dp-rpc 12322

bash run_dp_template_d.sh 8,9 8004 4 0 141.61.81.162 12322 2 &
bash run_dp_template_d.sh 10,11 8005 4 1 141.61.81.162 12322 2 &
bash run_dp_template_d.sh 12,13 8006 4 2 141.61.81.162 12322 2 &
bash run_dp_template_d.sh 14,15 8007 4 3 141.61.81.162 12322 2 &

wait