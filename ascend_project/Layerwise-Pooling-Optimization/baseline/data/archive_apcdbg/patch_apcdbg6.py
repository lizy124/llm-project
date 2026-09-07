#!/usr/bin/env python3
"""patch_apcdbg6.py — 第六轮: Hybrid.cache_blocks 打印 scheduler_block_size (实锤根因)."""
import py_compile

KC = "/vllm-workspace/vllm/vllm/v1/core/kv_cache_coordinator.py"

def edit(path, old, new, label):
    with open(path) as f:
        src = f.read()
    if new in src:
        print(f"[{label}] already")
        return
    n = src.count(old)
    if n != 1:
        print(f"[{label}] anchor count={n}, skip")
        return
    with open(path, "w") as f:
        f.write(src.replace(old, new, 1))
    print(f"[{label}] patched")

# 在 Hybrid cache_blocks 的 for 循环前打印关键值
edit(
    KC,
    '        for manager in self.single_type_managers:\n            num_tokens_to_cache = aligned_num_computed_tokens',
    '        print(f"APCDBG6 HYBRID-CB req={request.request_id} n={num_computed_tokens} "\n              f"sbs={self.scheduler_block_size} aligned={aligned_num_computed_tokens} "\n              f"partial={self.enable_partial_hash_hits}", flush=True)\n        for manager in self.single_type_managers:\n            num_tokens_to_cache = aligned_num_computed_tokens',
    "1-hybrid-cb")

py_compile.compile(KC, doraise=True)
print("COMPILE-OK")
