#!/usr/bin/env python3
"""Download a small Qwen3 model from modelscope for smoke testing."""
import os
os.environ.setdefault("VLLM_USE_V1", "1")
try:
    from modelscope import snapshot_download
except ImportError as e:
    print(f"modelscope not installed: {e}")
    raise
model_dir = snapshot_download(
    "Qwen/Qwen3-0.6B",
    cache_dir="/root/.cache/modelscope",
)
print("DOWNLOADED:", model_dir)
