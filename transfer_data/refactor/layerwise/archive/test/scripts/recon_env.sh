#!/bin/bash
# recon_env.sh — 建容器前最后一轮侦察(宿主机执行)
echo "===== 1. 镜像 7f06feda13d3 的 tag ====="
docker images --format '{{.Repository}}:{{.Tag}}  {{.ID}}' | grep 7f06feda13d3
echo
echo "===== 2. cxy_cann9.1.0 挂载与主机配置 ====="
docker inspect cxy_cann9.1.0 --format 'Image: {{.Config.Image}}'
docker inspect cxy_cann9.1.0 --format 'NetworkMode: {{.HostConfig.NetworkMode}}  Runtime: {{.HostConfig.Runtime}}  Privileged: {{.HostConfig.Privileged}}'
docker inspect cxy_cann9.1.0 --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Mode}}){{println}}{{end}}'
docker inspect cxy_cann9.1.0 --format 'Devices: {{json .HostConfig.Devices}}'
docker inspect cxy_cann9.1.0 --format 'ShmSize: {{.HostConfig.ShmSize}}  Ulimits: {{json .HostConfig.Ulimits}}'
echo
echo "===== 3. cxy 容器内池化相关包 ====="
docker exec cxy_cann9.1.0 bash -lc 'pip list 2>/dev/null | grep -iE "memcache|memfabric|mooncake|vllm|torch" ; echo ---; cat /proc/sys/vm/nr_hugepages'
echo
echo "===== 4. cxy 容器内网络实测 ====="
docker exec cxy_cann9.1.0 bash -lc 'curl -sI --max-time 8 https://github.com 2>&1 | head -1; curl -sI --max-time 8 https://pypi.org/simple/ 2>&1 | head -1; pip config list 2>/dev/null'
echo
echo "===== 5. 宿主机 hugepages 与端口 ====="
cat /proc/sys/vm/nr_hugepages
ss -tlnp 2>/dev/null | grep -E ':50051|:8004|:9008|:50088|:5000|:6000|:8000' || echo "目标端口均空闲"
echo
echo "===== 6. MetaService / mmc 配置痕迹(cxy 容器) ====="
docker exec cxy_cann9.1.0 bash -lc 'ls /usr/local/Ascend/memcache_hybrid 2>/dev/null | head; find / -maxdepth 4 -name "mmc-local.conf" 2>/dev/null | head -5; which metaservice 2>/dev/null; ls /usr/local/bin 2>/dev/null | grep -i meta'
