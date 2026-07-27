#!/bin/bash
#
# vLLM Load Balance Proxy Script
# Task: vLLM_PD分离_1机16卡_qwen35_20260722
# Proxy Port: 1999
# Connector: MooncakeConnectorV1 (prefill-first)
#
# Single machine (162): 4 P instances (port 8000-8003) + 4 D instances (port 8004-8007)
#   P: NPUs 0-7  (dp_rank 0-3, dp-rpc 12321)
#   D: NPUs 8-15 (dp_rank 0-3, dp-rpc 12322)
#   P/D 各自独立 DP 组，通过 Mooncake 传输 KV cache
#

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

python3 load_balance_proxy_server_example.py \
  --port 1997 \
  --host 90.90.97.27 \
  --prefiller-hosts \
    90.90.97.27 \
    90.90.97.27 \
    90.90.97.27 \
    90.90.97.27 \
  --prefiller-ports \
    8000 8001 8002 8003 \
  --decoder-hosts \
    90.90.97.27 \
    90.90.97.27 \
    90.90.97.27 \
    90.90.97.27 \
  --decoder-ports \
    8004 8005 8006 8007