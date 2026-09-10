#!/usr/bin/env python3
"""patch_apcdbg8b.py — 修正版: 用 cached_block_hash_to_block._cache + 局部 import."""
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

edit(
    PC,
    '        print(f"APCDBG8 ASC-FIND entry max_len={max_cache_hit_length} n_hashes={len(block_hashes)} "\n              f"pool_cached={len(self.block_pool.cached_block)}", flush=True)',
    '        print(f"APCDBG8 ASC-FIND entry max_len={max_cache_hit_length} n_hashes={len(block_hashes)} "\n              f"pool_cached={len(self.block_pool.cached_block_hash_to_block._cache)}", flush=True)',
    "1-find-poolsize")

edit(
    SM,
    '                print(f"APCDBG8 FA-MISS hash_in_table={block_hash in block_pool.cached_block} "\n                      f"table_n={len(block_pool.cached_block)} groups={kv_cache_group_ids} bs={block_size}", flush=True)',
    '                from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id as _mbh\n                _probe = block_pool.cached_block_hash_to_block.get_one_block(\n                    _mbh(block_hash, kv_cache_group_ids[0]))\n                print(f"APCDBG8 FA-MISS probe_hit={_probe is not None} "\n                      f"table_n={len(block_pool.cached_block_hash_to_block._cache)} groups={kv_cache_group_ids} bs={block_size}", flush=True)',
    "2-fa-miss")

py_compile.compile(PC, doraise=True)
py_compile.compile(SM, doraise=True)
print("COMPILE-OK")
