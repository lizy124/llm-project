#!/bin/bash
# probe_weights2.sh — 候选模型与数据集完整性核查
echo "===== Qwen3-32B 内容 ====="
ls /mnt/weight/Qwen3-32B 2>/dev/null
find /mnt/weight/Qwen3-32B -maxdepth 2 -name "*.safetensors" 2>/dev/null | head -5
echo "safetensors 数量: $(find /mnt/weight/Qwen3-32B -name '*.safetensors' 2>/dev/null | wc -l)"
du -sh /mnt/weight/Qwen3-32B 2>/dev/null
echo
echo "===== Qwen3-32B config 关键字段 ====="
python3 - <<'PYEOF'
import json
p = "/mnt/weight/Qwen3-32B/config.json"
try:
    c = json.load(open(p))
    for k in ("model_type", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_size", "max_position_embeddings"):
        print(k, "=", c.get(k))
except Exception as e:
    print("config 读取失败:", e)
PYEOF
echo
echo "===== Qwen3-0.6B(小模型备选) ====="
ls /mnt/weight/Qwen3-0.6B 2>/dev/null | head -10
echo
echo "===== GSM8K 数据集明细 ====="
ls -la /mnt/share/w00804037/datasets/gsm8k 2>/dev/null | head -10
wc -l /mnt/share/c00814587/aisbench_auto_tools_prefix/GSM8K.jsonl 2>/dev/null
wc -l /mnt/share/c00814587/vllm-ascend_gsm8k-lite/gsm8k-lite.jsonl 2>/dev/null
head -c 300 /mnt/share/c00814587/vllm-ascend_gsm8k-lite/gsm8k-lite.jsonl 2>/dev/null
echo
echo
echo "===== /mnt/weight 顶层(前 30) ====="
ls /mnt/weight | head -30
echo
echo "===== 权限可读性抽查 ====="
head -c 100 /mnt/weight/Qwen3-32B/config.json >/dev/null 2>&1 && echo "Qwen3-32B 可读" || echo "Qwen3-32B 不可读"
head -c 100 /mnt/share/c00814587/vllm-ascend_gsm8k-lite/gsm8k-lite.jsonl >/dev/null 2>&1 && echo "gsm8k-lite 可读" || echo "gsm8k-lite 不可读"
