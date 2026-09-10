#!/bin/bash
echo "=== modelscope Qwen ==="
ls -la /root/.cache/modelscope/Qwen 2>/dev/null | head
echo "=== modelscope hub ==="
ls /root/.cache/modelscope/hub 2>/dev/null | head -10
echo "=== model sizes ==="
for d in /root/.cache/modelscope/Qwen/*/; do
  echo "--- $d ---"
  ls "$d" | head -8
  du -sh "$d" 2>/dev/null | tail -1
done
