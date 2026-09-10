#!/bin/bash
# verify_dp1.sh — 容器内: TP16 DP1 单端口验证
set -uo pipefail
L=/home/lizhongyang/map_165/run/dsv4_layerwise_v2_8100.log
OUT=/tmp/dp1; mkdir -p $OUT; rm -f $OUT/*

echo "=== 1. 等 APIServer READY ==="
for i in $(seq 1 200); do
  N=$(grep -c "API server: HTTP server started" $L 2>/dev/null)
  if [ "$N" -ge 1 ]; then echo "READY (~$((i*3))s)"; break; fi
  sleep 3
  if [ $i -eq 200 ]; then echo "NOT READY in 600s"; tail -15 $L; exit 1; fi
done

echo "=== 2. 等 EngineCore idle ==="
for r in $(seq 1 16); do
  sleep 45
  EC=$(pgrep -f "VLLM::EngineCore")
  ST=$(timeout 15 py-spy dump --pid $EC 2>&1)
  if echo "$ST" | grep -q "_process_input_queue"; then echo "IDLE (round=$r)"; break; fi
  echo "round=$r: not idle yet"
done

echo ""
echo "=== 3. 单请求 ==="
t0=$(date +%s)
curl -s -m 120 "http://127.0.0.1:8100/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"hi, dp1 test"}],"max_tokens":16,"temperature":0}' \
  > $OUT/one.json 2>&1
t1=$(date +%s)
if grep -q '"content"' $OUT/one.json 2>/dev/null; then
  echo "单请求 OK ($((t1-t0))s): $(grep -o 'finish_reason.*' $OUT/one.json | head -c 50)"
else
  echo "单请求 FAIL ($((t1-t0))s): $(head -c 150 $OUT/one.json | tr -d '\n')"
fi

echo ""
echo "=== 4. 32 并发 ==="
t0=$(date +%s)
for j in $(seq 1 32); do
  curl -s -m 120 "http://127.0.0.1:8100/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"dsv4","messages":[{"role":"user","content":"hi, concurrent"}],"max_tokens":16,"temperature":0}' \
    > $OUT/con_$j.json 2>&1 &
done
wait
t1=$(date +%s)
OK=0
for f in $OUT/con_*.json; do grep -q '"content"' "$f" 2>/dev/null && OK=$((OK+1)); done
echo "32 并发: OK=$OK/32 用时=$((t1-t0))s"
