#!/bin/bash
# find_metaservice.sh — 在 cxy 容器内找 MetaService 启动方式
docker exec cxy_cann9.1.0 bash -lc '
echo "===== memcache_hybrid 包全部文件 ====="
P=/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid
find $P -maxdepth 2 | head -40
echo
echo "===== mmc-meta.conf 内容 ====="
cat $P/config/mmc-meta.conf 2>/dev/null | head -40
echo
echo "===== mmc-local.conf 内容 ====="
cat $P/config/mmc-local.conf 2>/dev/null | head -40
echo
echo "===== memfabric_hybrid 包找可执行 ====="
F=/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid
find $F -maxdepth 3 -type f \( -name "*meta*" -o -perm -111 \) 2>/dev/null | head -20
echo
echo "===== site-packages 顶层 metaservice 痕迹 ====="
ls /usr/local/python3.12.13/lib/python3.12/site-packages | grep -iE "meta|mmc|hybrid" | head
echo
echo "===== /usr/local 下 metaservice 二进制 ====="
find /usr/local -maxdepth 4 -iname "*metaservice*" -o -maxdepth 4 -iname "*meta_service*" 2>/dev/null | head
'
