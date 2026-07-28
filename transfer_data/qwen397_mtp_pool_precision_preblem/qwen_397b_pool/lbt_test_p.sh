#!/bin/bash
# P (Prefill): NPU 0-7, 2 DP x 4 TP, ports 8000-8001, dp-rpc 12321

bash run_dp_template_p.sh 0,1,2,3 8000 2 0 90.90.97.27 12321 4 &
bash run_dp_template_p.sh 4,5,6,7 8001 2 1 90.90.97.27 12321 4 &

wait