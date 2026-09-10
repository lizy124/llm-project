#!/bin/bash
# show_s1_result.sh — S1 判定全文 + put 指标名排查
D=/home/lizhongyang/map_165/run/s1_pd_multiconn
echo "===== test_status.txt ====="
cat $D/test_status.txt
echo "===== test.log (trunc 200c) ====="
cut -c1-200 $D/test.log
echo "===== metaservice put-ish metrics ====="
grep -iE "put" $D/metaservice_metrics.txt | grep -v "^#" | cut -c1-120 | head -20
