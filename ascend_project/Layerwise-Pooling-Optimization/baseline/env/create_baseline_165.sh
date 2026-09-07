#!/bin/bash
# create_baseline_165.sh — Layerwise-Pooling HBM 基线组专用容器
# 镜像: 7f06feda13d3 (refactor_165 同款, DSV4 支持实证)
# 模板: playbook/create_env.md §3 + refactor_165 实测挂载
# 幂等: 已存在则删除重建 (容器无状态, 数据在挂载卷)
set -euo pipefail

NAME=baseline_165
IMAGE=7f06feda13d3

# 建前必查: NPU 占用 + 端口
echo "== pre-check =="
npu-smi info | grep -c 0000: || true
ss -tlnp 2>/dev/null | grep -E ':50051|:50088|:8004|:9008' && { echo "PORT CONFLICT, abort"; exit 1; } || true

docker rm -f ${NAME} 2>/dev/null || true

docker run -dit -u root \
  --name ${NAME} \
  -e ASCEND_RUNTIME_OPTIONS=NODRV \
  --privileged=true \
  -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/sbin/:/usr/local/sbin \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home \
  -v /data:/data \
  -v /tmp:/tmp \
  -v /mnt:/mnt \
  -v /root/.cache:/root/.cache \
  --shm-size=100g \
  --net=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -w /home \
  ${IMAGE} \
  /bin/bash

echo "== post-check =="
docker exec ${NAME} bash -c "npu-smi info | grep -c 0000:; python3 --version"
echo "OK: ${NAME} created"
