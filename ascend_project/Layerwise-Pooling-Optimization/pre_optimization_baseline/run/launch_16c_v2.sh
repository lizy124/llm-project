#!/bin/bash
# launch_16c_v2.sh — pool_165 内: 彻底清僵尸 (含 setproctitle 改名的 worker) + 启动
set -uo pipefail
RUN=/home/lizhongyang/map_165/run
LOG="$RUN/dsv4_layerwise_v2_8100.log"

echo "=== 1. 彻底清理 (apiserver + engine + 改名 worker) ==="
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::" 2>/dev/null || true
pkill -9 -f "from multiprocessing" 2>/dev/null || true
sleep 5
LEFT=$(pgrep -fc "vllm serve|VLLM::" 2>/dev/null || echo 0)
echo "残留进程: $LEFT"

echo "=== 1.5 清 /dev/shm psm 残留 ==="
rm -f /dev/shm/psm_* 2>/dev/null

echo "=== 2. MetaService 存活 ==="
if pgrep -f "MetaService" >/dev/null; then
  echo "MetaService alive (pid $(pgrep -f MetaService | head -1))"
else
  echo "MetaService 不在, 重启 (device_sdma conf)..."
  cd "$RUN"
  nohup python3 -c "from memcache_hybrid import MetaService; MetaService.main()" > "$RUN/metaservice.log" 2>&1 &
  sleep 8
  pgrep -f "MetaService" >/dev/null && echo "restarted" || echo "MetaService FAIL"
fi

echo "=== 3. 备份上一轮日志 ==="
[ -f "$LOG" ] && cp "$LOG" "${LOG%.log}_prev.log" && echo "backed up"

echo "=== 4. 启动 16 卡 (TP4×DP4, 1M, device_sdma) ==="
nohup bash /home/lizhongyang/tmp/start_dsv4_layerwise.sh > "$RUN/launch_16c.outer.log" 2>&1 &
sleep 10
head -4 "$RUN/launch_16c.outer.log"
