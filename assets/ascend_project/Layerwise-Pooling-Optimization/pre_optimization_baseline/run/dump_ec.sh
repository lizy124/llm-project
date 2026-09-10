#!/bin/bash
EC=$(pgrep -f "VLLM::EngineCore")
echo "=== EngineCore pid=$EC ==="
timeout 20 py-spy dump --pid $EC 2>&1 | sed -n '1,30p'
