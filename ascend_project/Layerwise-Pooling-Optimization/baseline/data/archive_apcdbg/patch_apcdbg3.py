#!/usr/bin/env python3
"""patch_apcdbg3.py — 第三轮: hash 长度 + cache_full_blocks 注册明细 (全 print).

1. GET-COMPUTED 打印加 n_hashes
2. ASC-FIND 入口打印加 n_hashes
3. base cache_blocks 入口加 print (round-1 logger 行疑似被过滤)
4. BlockPool.cache_full_blocks 入口加 print (注册侧真身)
"""
import py_compile
import re

KM = "/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py"
PC = "/vllm-workspace/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py"
STM = "/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py"
BP = "/vllm-workspace/vllm/vllm/v1/core/block_pool.py"


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


# 1. GET-COMPUTED 加 n_hashes
edit(
    KM,
    'print(f"APCDBG2 GET-COMPUTED req={request.request_id} enable_caching={self.enable_caching} "\n              f"skip={request.skip_reading_prefix_cache}", flush=True)',
    'print(f"APCDBG2 GET-COMPUTED req={request.request_id} enable_caching={self.enable_caching} "\n              f"skip={request.skip_reading_prefix_cache} n_hashes={len(request.block_hashes)}", flush=True)',
    "1-get-computed")

# 2. ASC-FIND 入口加 n_hashes
edit(
    PC,
    'print(f"APCDBG2 ASC-FIND entry max_len={max_cache_hit_length}", flush=True)',
    'print(f"APCDBG2 ASC-FIND entry max_len={max_cache_hit_length} n_hashes={len(block_hashes)}", flush=True)',
    "2-asc-find")

# 3. base cache_blocks 入口 print
edit(
    STM,
    'logger.info("APCDBG CACHE mgr=%s req=%s num_tokens=%s", type(self).__name__, request.request_id, num_tokens)',
    'logger.info("APCDBG CACHE mgr=%s req=%s num_tokens=%s", type(self).__name__, request.request_id, num_tokens)\n        print(f"APCDBG3 CACHE mgr={type(self).__name__} req={request.request_id} num_tokens={num_tokens} n_hashes={len(request.block_hashes)}", flush=True)',
    "3-base-cache")

# 4. BlockPool.cache_full_blocks 入口 print
with open(BP) as f:
    src = f.read()
if "APCDBG3 FULL" not in src:
    m = re.search(r"(    def cache_full_blocks\(\n(?:.*\n)*?    \) -> None:\n)", src)
    if m:
        sig = m.group(1)
        ins = ('        print(f"APCDBG3 FULL grp={kv_cache_group_id} n_cached={num_cached_blocks} "\n'
               '              f"n_full={num_full_blocks} bs={block_size} "\n'
               '              f"mask={None if block_mask is None else (sum(block_mask), len(block_mask))}", flush=True)\n')
        src = src.replace(sig, sig + ins, 1)
        with open(BP, "w") as f:
            f.write(src)
        print("[4-pool-full] patched")
    else:
        print("[4-pool-full] signature not found")
else:
    print("[4-pool-full] already")

for p in [KM, PC, STM, BP]:
    py_compile.compile(p, doraise=True)
print("COMPILE-OK")
