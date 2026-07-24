#!/bin/bash
# ================================================================
# Qwen3.5-35B-A3B PD 分离部署 (16 GPUs, 单机 162)
# 
# 架构: P 和 D 各自独立的 DP 组
#   P (Prefill):  4 DP × 2 TP =  8 NPUs (0-7),  dp_rank 0-3, ports 8000-8003, dp-rpc 12321
#   D (Decode):   4 DP × 2 TP =  8 NPUs (8-15), dp_rank 0-3, ports 8004-8007, dp-rpc 12322
#   P/D 之间通过 Mooncake 传输 KV cache，不共享 DP 组
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