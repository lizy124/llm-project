#!/usr/bin/env python3
"""patch_apcdbg7.py — 验证性修复: Hybrid.cache_blocks 对齐粒度 sbs->hash_block_size.

诊断目的: 证明 aligned=0 是唯一根因 (非正式修复).
"""
import py_compile

KC = "/vllm-workspace/vllm/vllm/v1/core/kv_cache_coordinator.py"

with open(KC) as f:
    src = f.read()

old = '''            aligned_num_computed_tokens = (
                num_computed_tokens
                // self.scheduler_block_size
                * self.scheduler_block_size
            )'''
new = '''            aligned_num_computed_tokens = (
                num_computed_tokens
                // self.hash_block_size
                * self.hash_block_size
            )'''

if "APCDBG7" in src:
    print("[1] already")
else:
    n = src.count(old)
    assert n == 1, f"anchor count={n}"
    src = src.replace(old, new, 1)
    # 打标记
    src = src.replace(
        '        print(f"APCDBG6 HYBRID-CB req={request.request_id} n={num_computed_tokens} "',
        '        print(f"APCDBG7 HYBRID-CB req={request.request_id} n={num_computed_tokens} hash_bs={self.hash_block_size} "', 1)
    with open(KC, "w") as f:
        f.write(src)
    print("[1] patched")

py_compile.compile(KC, doraise=True)
print("COMPILE-OK")
