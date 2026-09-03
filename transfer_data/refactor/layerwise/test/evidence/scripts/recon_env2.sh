#!/bin/bash
# recon_env2.sh — 内存/容器 env/MetaService 深度侦察
echo "===== 1. 宿主机内存与大页 ====="
free -g | head -2
grep -i hugepages /proc/meminfo
echo
echo "===== 2. cxy 容器 Env / WorkingDir / Cmd ====="
docker inspect cxy_cann9.1.0 --format 'WorkingDir: {{.Config.WorkingDir}}'
docker inspect cxy_cann9.1.0 --format 'Cmd: {{json .Config.Cmd}}'
docker inspect cxy_cann9.1.0 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -vE '^(PATH|LS_COLORS)' | head -25
echo
echo "===== 3. MetaService / mmc-local.conf 深度搜索(cxy 容器,可能耗时) ====="
docker exec cxy_cann9.1.0 bash -lc 'pip show memcache_hybrid 2>/dev/null | head -8; echo ---; python3 -c "import memcache_hybrid, os; print(os.path.dirname(memcache_hybrid.__file__))" 2>/dev/null'
echo
echo "===== 4. memcache_hybrid 包内可执行与配置 ====="
docker exec cxy_cann9.1.0 bash -lc 'P=$(python3 -c "import memcache_hybrid, os; print(os.path.dirname(memcache_hybrid.__file__))" 2>/dev/null); echo PKG=$P; find $P -maxdepth 3 \( -name "*.conf" -o -iname "*metaservice*" -o -name "*.sh" \) 2>/dev/null | head -15'
echo
echo "===== 5. 系统范围内 mmc-local.conf / metaservice ====="
docker exec cxy_cann9.1.0 bash -lc 'find /usr /opt /etc -maxdepth 5 \( -name "mmc*.conf" -o -iname "*metaservice*" \) 2>/dev/null | head -10'
