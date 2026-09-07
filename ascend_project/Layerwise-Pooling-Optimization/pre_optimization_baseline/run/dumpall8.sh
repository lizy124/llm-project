#!/bin/bash
for p in $(pgrep -f "VLLM::Worker"); do
  R=$(tr '\0' ' ' < /proc/$p/comm 2>/dev/null)
  S=$(timeout 20 py-spy dump --pid $p 2>&1 | grep -E "File \"" | sed -n '1p;5p;8p')
  echo "== $R pid=$p"
  echo "$S"
done
