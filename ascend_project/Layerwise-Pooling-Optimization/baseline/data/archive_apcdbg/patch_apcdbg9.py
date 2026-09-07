#!/usr/bin/env python3
"""patch_apcdbg9.py — 第九轮: 注册统计 + free 时表大小."""
import py_compile
import re

BP = "/vllm-workspace/vllm/vllm/v1/core/block_pool.py"

with open(BP) as f:
    src = f.read()

if "APCDBG9" in src:
    print("already")
else:
    # 1. 循环前: 统计变量
    old1 = '''        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue'''
    new1 = '''        _n_null = sum(1 for b in new_full_blocks if b.is_null)
        _t0 = len(self.cached_block_hash_to_block._cache)
        _n_reg = 0
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue'''
    n = src.count(old1)
    assert n == 1, f"old1 count={n}"
    src = src.replace(old1, new1, 1)
    print("[1-loop-pre] patched")

    # 2. _insert_block_hash 后计数
    old2 = '''            self._insert_block_hash(
                block_hash_with_group_id,
                blk,
                num_tokens=num_hash_tokens,
            )'''
    new2 = '''            self._insert_block_hash(
                block_hash_with_group_id,
                blk,
                num_tokens=num_hash_tokens,
            )
            _n_reg += 1'''
    n = src.count(old2)
    assert n == 1, f"old2 count={n}"
    src = src.replace(old2, new2, 1)
    print("[2-count] patched")

    # 3. 循环后打印 (enable_kv_cache_events 块前)
    old3 = '''        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:'''
    new3 = '''        print(f"APCDBG9 REG grp={kv_cache_group_id} t0={_t0} reg={_n_reg} null={_n_null} "
              f"table_n={len(self.cached_block_hash_to_block._cache)}", flush=True)
        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:'''
    n = src.count(old3)
    assert n == 1, f"old3 count={n}"
    src = src.replace(old3, new3, 1)
    print("[3-post] patched")

    with open(BP, "w") as f:
        f.write(src)

# 4. BlockPool 里可能的清理函数入口都打点: 先找 def free / def reset_prefix_cache
with open(BP) as f:
    src = f.read()

for fname in ["free", "reset_prefix_cache"]:
    pat = f"    def {fname}("
    if f"APCDBG9 {fname.upper()}" in src:
        print(f"[4-{fname}] already")
        continue
    m = re.search(re.escape(pat) + r"[^\n]*\n", src)
    if m:
        sig_line = m.group(0)
        # 找签名结束 (可能多行参数)
        start = m.start()
        # 找到签名块的 ')' 行
        end = src.index(")", start)
        # 下一行
        nl = src.index("\n", end)
        anchor = src[start:nl+1]
        ins = anchor + f'        print(f"APCDBG9 {fname.upper()} table_n={{len(self.cached_block_hash_to_block._cache)}}", flush=True)\n'
        src = src.replace(anchor, ins, 1)
        print(f"[4-{fname}] patched")
    else:
        print(f"[4-{fname}] not found")

with open(BP, "w") as f:
    f.write(src)

py_compile.compile(BP, doraise=True)
print("COMPILE-OK")
