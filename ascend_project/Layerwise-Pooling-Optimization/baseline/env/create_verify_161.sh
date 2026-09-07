#!/bin/bash
# create_verify_161.sh — 161 上 DSV4 prefix caching 鉴别实验容器
# 镜像: daa8cb6e2479 (nightly-main-a3: vllm 0.27.1 + vllm-ascend 0.19.1rc2.dev1689, 与 165 栈最接近)
# 模板: create_baseline_165.sh (playbook/create_env.md §3 挂载模式)
set -euo pipefail

NAME=lw_verify_161
IMAGE=daa8cb6e2479
BASE=/home/lizhongyang/lw_verify   # 注意: 161 无 lizhongyang 用户, 用 root 属主目录

echo "== pre-check =="
ss -tlnp 2>/dev/null | grep -E ':8004|:8005' && { echo "PORT CONFLICT, abort"; exit 1; } || true

docker rm -f ${NAME} 2>/dev/null || true
mkdir -p ${BASE}/{run,data,results}

docker run -dit -u root \
  --name ${NAME} \
  --privileged=true \
  -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/sbin/:/usr/local/sbin \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home \
  -v /tmp:/tmp \
  -v /mnt:/mnt \
  --shm-size=100g \
  --net=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -w /home \
  ${IMAGE} \
  /bin/bash

echo "== post-check =="
docker exec ${NAME} bash -c "npu-smi info | grep -c 0000:; python3 --version; pip list 2>/dev/null | grep -iE '^vllm'"
echo "OK: ${NAME} created"
