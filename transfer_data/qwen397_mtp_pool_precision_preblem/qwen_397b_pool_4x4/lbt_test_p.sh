#!/bin/bash
# P (Prefill): NPU 0-7, 4 DP x 2 TP, ports 8000-8003, dp-rpc 12321

bash run_dp_template_p.sh 0,1 8000 4 0 90.90.97.27 12321 2 &
bash run_dp_template_p.sh 2,3 8001 4 1 90.90.97.27 12321 2 &
bash run_dp_template_p.sh 4,5 8002 4 2 90.90.97.27 12321 2 &
bash run_dp_template_p.sh 6,7 8003 4 3 90.90.97.27 12321 2 &

wait