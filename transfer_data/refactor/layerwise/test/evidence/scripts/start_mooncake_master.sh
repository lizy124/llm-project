#!/bin/bash
# start_mooncake_master.sh — 容器内执行:mooncake master 拉起(四条纪律)
BASE=/home/lizhongyang/map_165
LOGD=$BASE/run/mooncake_logs
mkdir -p $LOGD

# 停旧(显式 PID,禁 pkill)
[ -f $LOGD/master.pid ] && kill -9 $(cat $LOGD/master.pid) 2>/dev/null
sleep 1

mooncake_master --rpc_port 50088 --metrics_port 9008 > $LOGD/master.log 2>&1 &
echo $! > $LOGD/master.pid
sleep 3
tail -5 $LOGD/master.log
curl -s --max-time 5 http://127.0.0.1:9008/metrics | grep -E 'role|state|service_ready' | head -5
