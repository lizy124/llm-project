#!/bin/bash
# rerun_e2e.sh — 在 rebase 后 head (63be9e03b) 上重跑三场景 e2e(宿主机执行)
# 顺序:更新代码 -> clean -> pool_prep(MetaService) -> S2 -> S3(含 master) -> S1
set -uo pipefail
BASE=/home/lizhongyang/map_165
CTR=refactor_165
SHA=63be9e03b

echo "===== rerun_e2e start $(date -Is) ====="

if [ "${SKIP_UPDATE:-0}" != "1" ]; then
echo "===== [1/6] update repo to $SHA $(date -Is) ====="
docker exec $CTR bash -lc "
set -e
cd /vllm-workspace/vllm-ascend
if [ -f /tmp/rerun.bundle ]; then
  git fetch /tmp/rerun.bundle 'refs/heads/refactor_layerwise_part1:refs/heads/from-bundle'
  git reset --hard 63be9e03b
else
  git fetch fork refactor_layerwise_part1
  git reset --hard $SHA
fi
echo HEAD=\$(git rev-parse --short HEAD)
echo vllm=\$(git -C /vllm-workspace/vllm rev-parse --short HEAD)
pip install -e . --no-deps --no-build-isolation -q 2>&1 | tail -2
python3 -c 'import vllm_ascend; print(\"vllm_ascend import OK\")'
" || { echo "FATAL: repo update failed"; exit 1; }
else
echo "===== [1/6] skipped (SKIP_UPDATE=1) ====="
fi

echo "===== [2/6] clean leftovers $(date -Is) ====="
docker exec $CTR bash $BASE/test/clean_npu.sh || true

echo "===== [3/6] pool prep (MetaService) $(date -Is) ====="
bash $BASE/start/pool_prep.sh || { echo "FATAL: pool prep failed"; exit 1; }

echo "===== [4/6] S2 memcache layerwise $(date -Is) ====="
if docker exec $CTR bash $BASE/test/s2_start.sh; then
  if docker exec $CTR bash $BASE/test/s2_test.sh; then
    echo "S2_RESULT=PASS"
  else
    echo "S2_RESULT=FAIL(exit=$?)"
  fi
else
  echo "S2_RESULT=START_FAIL"
fi
docker exec $CTR bash $BASE/test/stop_server.sh $BASE/run/s2_memcache_layerwise || true
docker exec $CTR bash $BASE/test/clean_npu.sh || true

echo "===== [5/6] S3 mooncake non-layerwise $(date -Is) ====="
docker exec $CTR bash $BASE/start/start_mooncake_master.sh || { echo "FATAL: mooncake master failed"; exit 1; }
if docker exec $CTR bash $BASE/test/s3_start.sh; then
  if docker exec $CTR bash $BASE/test/s3_test.sh; then
    echo "S3_RESULT=PASS"
  else
    echo "S3_RESULT=FAIL(exit=$?)"
  fi
else
  echo "S3_RESULT=START_FAIL"
fi
docker exec $CTR bash $BASE/test/stop_server.sh $BASE/run/s3_mooncake_non_layerwise || true
docker exec $CTR bash $BASE/test/clean_npu.sh || true

echo "===== [6/6] S1 MultiConnector PD $(date -Is) ====="
if docker exec $CTR bash $BASE/test/s1_start_prefill.sh; then
  if docker exec $CTR bash $BASE/test/s1_start_decode.sh; then
    if docker exec $CTR bash $BASE/test/s1_start_proxy.sh; then
      if docker exec $CTR bash $BASE/test/s1_test.sh; then
        echo "S1_RESULT=PASS"
      else
        echo "S1_RESULT=FAIL(exit=$?)"
      fi
    else
      echo "S1_RESULT=PROXY_START_FAIL"
    fi
  else
    echo "S1_RESULT=DECODE_START_FAIL"
  fi
else
  echo "S1_RESULT=PREFILL_START_FAIL"
fi
docker exec $CTR bash $BASE/test/stop_s1.sh || true
docker exec $CTR bash $BASE/test/clean_npu.sh || true

echo "===== summary $(date -Is) ====="
cat $BASE/run/s2_memcache_layerwise/test_status.txt 2>/dev/null | tail -2
cat $BASE/run/s3_mooncake_non_layerwise/test_status.txt 2>/dev/null | tail -2
cat $BASE/run/s1_pd_multiconn/test_status.txt 2>/dev/null | tail -2
echo "===== rerun_e2e done $(date -Is) ====="
