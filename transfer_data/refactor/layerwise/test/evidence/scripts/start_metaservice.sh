#!/bin/bash
# start_metaservice.sh — 容器内执行:拉起 MetaService 并探活
BASE=/home/lizhongyang/map_165
CONF=/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config

pkill -f 'MetaService' 2>/dev/null
sleep 1
rm -f $BASE/run/metaservice.log
export MMC_META_CONFIG_PATH=$CONF/mmc-meta.conf
nohup python3 -c 'from memcache_hybrid import MetaService; MetaService.main()' > $BASE/run/metaservice.log 2>&1 &
sleep 3
tail -5 $BASE/run/metaservice.log
ss -tln | grep -E ':5000 |:6000 ' && echo META_PORTS_OK || echo META_PORTS_MISSING
curl -s --max-time 5 http://127.0.0.1:8000/metrics | head -3
