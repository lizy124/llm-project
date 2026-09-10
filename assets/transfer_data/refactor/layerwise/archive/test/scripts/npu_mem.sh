#!/bin/bash
# npu_mem.sh — 每卡 HBM 占用一览
npu-smi info -t memory -i 0 2>/dev/null | head -10
echo "==="
for i in 0 1 2 3 4 5 6 7; do
  USED=$(npu-smi info -t memory -i $i 2>/dev/null | grep -i "used" | head -1)
  echo "chip$i: $USED"
done
