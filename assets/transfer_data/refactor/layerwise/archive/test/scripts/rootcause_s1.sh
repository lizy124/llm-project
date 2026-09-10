#!/bin/bash
# rootcause_s1.sh — 抓 P 侧首个 worker 异常段
D=/home/lizhongyang/map_165/run/s1_pd_multiconn
FIRST=$(grep -n "Traceback (most recent call last)" $D/prefill.log | head -1 | cut -d: -f1)
echo "first traceback at line $FIRST"
if [ -n "$FIRST" ]; then
  START=$((FIRST > 10 ? FIRST - 10 : 1))
  END=$((FIRST + 60))
  sed -n "${START},${END}p" $D/prefill.log
fi
