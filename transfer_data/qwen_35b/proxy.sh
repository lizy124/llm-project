#!/bin/bash
#
# vLLM Load Balance Proxy Script
# Task: vLLM_PD分离_1机16卡_qwen35_20260722
# Proxy Port: 1999
# Connector: MooncakeConnectorV1 (prefill-first)
#
# Single machine (162): 4 P instances (port 8000-8003) + 4 D instances (port 8004-8007)
#   P: GPUs 0-7  (dp_rank 0-3)
#   D: GPUs 8-15 (dp_rank 4-7)
#

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

python3 load_balance_proxy_server_example.py \
  --port 1999 \
  --host 141.61.81.162 \
  --prefiller-hosts \
    141.61.81.162 \
    141.61.81.162 \
    141.61.81.162 \
    141.61.81.162 \
  --prefiller-ports \
    8000 8001 8002 8003 \
  --decoder-hosts \
    141.61.81.162 \
    141.61.81.162 \
    141.61.81.162 \
    141.61.81.162 \
  --decoder-ports \
    8004 8005 8006 8007