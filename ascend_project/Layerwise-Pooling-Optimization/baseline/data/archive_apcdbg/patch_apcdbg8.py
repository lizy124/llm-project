#!/usr/bin/env python3
"""patch_apcdbg8.py — 第八轮: lookup 时 dump hash 表大小 + Phase1 首 miss 详情."""
import py_compile

PC = "/vllm-workspace/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py"
SM = "/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py"

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

# 1. ASC-FIND entry 加 hash 表大小
edit(
    PC,
    '        print(f"APCDBG2 ASC-FIND entry max_len={max_cache_hit_length} n_hashes={len(block_hashes)}", flush=True)',
    '        print(f"APCDBG8 ASC-FIND entry max_len={max_cache_hit_length} n_hashes={len(block_hashes)} "\n              f"pool_cached={len(self.block_pool.cached_block)}", flush=True)',
    "1-find-poolsize")

# 2. FA Phase1 首 miss 打印详情
edit(
    SM,
    '        for block_hash in itertools.islice(full_block_hashes, max_length // block_size):\n            cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)\n            if not cached_block:\n                break',
    '        for block_hash in itertools.islice(full_block_hashes, max_length // block_size):\n            cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)\n            if not cached_block:\n                print(f"APCDBG8 FA-MISS hash_in_table={block_hash in block_pool.cached_block} "\n                      f"table_n={len(block_pool.cached_block)} groups={kv_cache_group_ids} bs={block_size}", flush=True)\n                break',
    "2-fa-miss")

for p in [PC, SM]:
    py_compile.compile(p, doraise=True)
print("COMPILE-OK")
