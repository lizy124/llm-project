#!/bin/bash
# archive.sh — 把 run/ 产物归档进 record_final/(容器内执行)
set -Eeuo pipefail
B=/home/lizhongyang/map_165
RF=$B/record_final

# --- s1 ---
D=$RF/s1_pd_multiconn_20260901
mkdir -p $D
cp $B/run/s1_pd_multiconn/*status.txt $D/ 2>/dev/null || true
cp $B/run/s1_pd_multiconn/*_env.txt $D/ 2>/dev/null || true
cp $B/run/s1_pd_multiconn/test.log $D/ 2>/dev/null || true
cp $B/run/s1_pd_multiconn/metaservice_metrics.txt $D/ 2>/dev/null || true
grep -E "AscendMultiConnector|AttributeError" $B/run/s1_pd_multiconn/prefill.log | cut -c1-300 > $D/prefill_multiconn_init.txt 2>/dev/null || true
grep -cE "load_gvas:" $B/run/s1_pd_multiconn/prefill.log > $D/prefill_load_gvas_count.txt 2>/dev/null || true

# --- s2 ---
D=$RF/s2_memcache_layerwise_20260901
mkdir -p $D
cp $B/run/s2_memcache_layerwise/*status.txt $D/ 2>/dev/null || true
cp $B/run/s2_memcache_layerwise/env.txt $D/ 2>/dev/null || true
cp $B/run/s2_memcache_layerwise/test.log $D/ 2>/dev/null || true
grep -E "hit_check:.*hit_tokens=[1-9]|load_gvas:.*valid_gvas=[1-9]" $B/run/s2_memcache_layerwise/server.log | tail -6 | cut -c1-300 > $D/layerwise_markers.txt 2>/dev/null || true

# --- s3 ---
D=$RF/s3_mooncake_non_layerwise_20260901
mkdir -p $D
cp $B/run/s3_mooncake_non_layerwise/*status.txt $D/ 2>/dev/null || true
cp $B/run/s3_mooncake_non_layerwise/env.txt $D/ 2>/dev/null || true
cp $B/run/s3_mooncake_non_layerwise/test.log $D/ 2>/dev/null || true
cp $B/run/s3_mooncake_non_layerwise/pool_metrics.txt $D/ 2>/dev/null || true
cp $B/run/s3_mooncake_non_layerwise/vllm_metrics.txt $D/ 2>/dev/null || true

ls -la $RF/*/
echo ARCHIVE_DONE
