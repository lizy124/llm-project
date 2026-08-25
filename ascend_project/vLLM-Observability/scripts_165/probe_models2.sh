#!/bin/bash
echo "=== modelscope hub/models ==="
find /root/.cache/modelscope/hub -maxdepth 3 -type d 2>/dev/null | head -20
echo "=== sizes ==="
for d in /root/.cache/modelscope/hub/models/*/; do
  for m in "$d"*/; do
    echo "--- $m ---"
    du -sh "$m" 2>/dev/null
  done
done
echo "=== network to modelscope/hf ==="
curl -s -o /dev/null -w 'modelscope:%{http_code}\n' --connect-timeout 6 https://modelscope.cn
curl -s -o /dev/null -w 'hf:%{http_code}\n' --connect-timeout 6 https://huggingface.co
