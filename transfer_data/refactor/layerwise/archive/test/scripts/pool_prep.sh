#!/bin/bash
# pool_prep.sh — memcache 池化前置(宿主机执行,幂等)
# 1) hugepages 2) mmc-local.conf 改 device_sdma+dram 3) MetaService 拉起+探活
set -e
BASE=/home/lizhongyang/map_165
CONF=/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config

echo "===== 1. hugepages ====="
CUR=$(cat /proc/sys/vm/nr_hugepages)
if [ "$CUR" -lt 200000 ]; then
  sysctl -w vm.nr_hugepages=200000
fi
grep -E 'HugePages_Total|HugePages_Free' /proc/meminfo

echo "===== 2. mmc-local.conf(device_sdma + dram 16GB) ====="
docker exec refactor_165 bash -lc "
set -e
cp -n $CONF/mmc-local.conf $CONF/mmc-local.conf.bak 2>/dev/null || true
sed -i 's/^ock.mmc.local_service.protocol = .*/ock.mmc.local_service.protocol = device_sdma/' $CONF/mmc-local.conf
sed -i 's/^ock.mmc.local_service.dram.size = .*/ock.mmc.local_service.dram.size = 16GB/' $CONF/mmc-local.conf
grep -E 'protocol|dram.size' $CONF/mmc-local.conf
"

echo "===== 3. MetaService 拉起(脚本文件方式) ====="
docker exec refactor_165 bash $BASE/start/start_metaservice.sh

echo "===== 4. 探活确认 ====="
for i in 1 2 3 4 5; do
  ss -tln | grep -q ':5000 ' && ss -tln | grep -q ':6000 ' && break
  sleep 2
done
ss -tln | grep -E ':5000 |:6000 ' || { echo "FAIL: meta ports not listening"; exit 1; }
echo "POOL_PREP_OK"
