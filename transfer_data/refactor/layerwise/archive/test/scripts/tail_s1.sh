#!/bin/bash
# tail_s1.sh — 查看 S1 各日志关键行(容器内执行)
D=/home/lizhongyang/map_165/run/s1_pd_multiconn
echo "===== prefill error/engine lines ====="
grep -inE 'error|traceback|attributeerror|engine core|started|ready' $D/prefill.log 2>/dev/null | grep -viE 'shm_broadcast|poller' | tail -12
echo "===== prefill last 3 ====="
tail -3 $D/prefill.log 2>/dev/null
echo "===== decode last 3 ====="
tail -3 $D/decode.log 2>/dev/null
echo "===== proxy last 3 ====="
tail -3 $D/proxy.log 2>/dev/null
echo "===== status files ====="
cat $D/*status.txt 2>/dev/null
