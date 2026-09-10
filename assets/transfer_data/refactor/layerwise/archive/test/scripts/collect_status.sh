#!/bin/bash
# collect_status.sh — 汇总三场景状态文件
B=/home/lizhongyang/map_165/run
for d in s2_memcache_layerwise s3_mooncake_non_layerwise s1_pd_multiconn; do
  echo "########## $d ##########"
  for f in $B/$d/*status.txt; do
    echo "--- $f ---"
    cat "$f" 2>/dev/null
  done
  echo "--- test.log (trunc 200c) ---"
  cut -c1-200 $B/$d/test.log 2>/dev/null
  echo
done
