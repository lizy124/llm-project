#!/bin/bash
# ================================================================
# Qwen3.5-35B-A3B PD 分离部署 (16 GPUs, 单机 162)
# 
# 架构: 8P + 8D, 8 DP × 2 TP = 16 GPUs
#   P (Prefill):  GPUs 0-7,  dp_rank 0-3, ports 8000-8003
#   D (Decode):   GPUs 8-15, dp_rank 4-7, ports 8004-8007
#
# 启动步骤:
#   1. 先启动 P:  bash lbt_test_p.sh
#   2. 另开终端启动 D:  bash lbt_test_d.sh
#   3. 等 P 和 D 都启动好后, 启动 proxy:  bash proxy.sh
#   4. Proxy 在 1999 端口提供服务
# ================================================================

echo "Please run P and D in separate terminals:"
echo "  Terminal 1: bash lbt_test_p.sh"
echo "  Terminal 2: bash lbt_test_d.sh"
echo "  Then:       bash proxy.sh"