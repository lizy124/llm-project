#!/usr/bin/env python3
"""patch_apcdbg11.py — 第十轮: FA lookup 入口打印 block_size/alignment/迭代上限."""
import py_compile

SM = "/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py"

with open(SM) as f:
    src = f.read()

if "APCDBG11" in src:
    print("already")
else:
    old = '''        # Fine-grained mode (alignment_tokens == hash_block_size <
        # block_size): resolve_block_hashes kept the raw hash-granularity
        # list so interior boundaries can be probed.
        fine_grained = (
            alignment_tokens < block_size and block_size % alignment_tokens == 0
        )'''
    new = '''        print(f"APCDBG11 FA-LOOKUP bs={block_size} hash_bs={block_pool.hash_block_size} "
              f"align={alignment_tokens} max_len={max_length} iters={max_length // block_size} "
              f"groups={kv_cache_group_ids}", flush=True)
        # Fine-grained mode (alignment_tokens == hash_block_size <
        # block_size): resolve_block_hashes kept the raw hash-granularity
        # list so interior boundaries can be probed.
        fine_grained = (
            alignment_tokens < block_size and block_size % alignment_tokens == 0
        )'''
    n = src.count(old)
    assert n == 1, f"count={n}"
    src = src.replace(old, new, 1)
    with open(SM, "w") as f:
        f.write(src)
    print("patched")

py_compile.compile(SM, doraise=True)
print("COMPILE-OK")
