#!/bin/bash
# probe_weights.sh — 165 模型权重与 e2e 数据盘点(宿主机侧)
echo "===== /mnt/weight ====="
ls -la /mnt/weight 2>/dev/null || echo "NO /mnt/weight"
echo
echo "===== /mnt 其他 ====="
ls /mnt 2>/dev/null
echo
echo "===== /data ====="
ls /data 2>/dev/null | head -20
echo
echo "===== /home 下的模型痕迹 ====="
ls /home 2>/dev/null
for d in /home/*/models /home/*/weight* /home/*/.cache/modelscope /home/*/.cache/huggingface; do
  [ -d "$d" ] && echo "-- $d" && ls "$d" 2>/dev/null | head -8
done
echo
echo "===== qwen3-32b-pdmix 明细(若存在) ====="
ls /mnt/weight/qwen3-32b-pdmix 2>/dev/null | head -15
du -sh /mnt/weight/qwen3-32b-pdmix 2>/dev/null
echo
echo "===== GSM8K 数据集痕迹 ====="
for p in /mnt /data /home/lizhongyang /home/hucong; do
  find "$p" -maxdepth 4 -iname "*gsm8k*" 2>/dev/null | head -5
done
echo
echo "===== 磁盘余量 ====="
df -h /mnt /data /home 2>/dev/null
