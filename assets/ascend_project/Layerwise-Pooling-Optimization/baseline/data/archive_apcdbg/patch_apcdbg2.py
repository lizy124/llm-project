#!/usr/bin/env python3
"""patch_apcdbg2.py — 第二轮埋点: 埋在真实执行路径 (print 输出, 无 logger 依赖).

A. kv_cache_manager.get_computed_blocks 入口: enable_caching/skip 状态
B. kv_cache_manager allocate 侧门控: enable_caching/delay_cache_blocks
C. ascend patch_kv_cache_coordinator.find_longest_cache_hit: 入口 + 每 group 命中
"""
import py_compile
import sys

KM = "/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py"
PC = "/vllm-workspace/vllm-ascend/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py"


def insert_after(path, anchor, code, label):
    with open(path) as f:
        src = f.read()
    if code.split("(")[0] in src:
        print(f"[{label}] already patched")
        return
    n = src.count(anchor)
    if n != 1:
        print(f"[{label}] anchor count={n} (need 1), skip")
        return
    src = src.replace(anchor, anchor + code, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"[{label}] patched")


def insert_before(path, anchor, code, label):
    with open(path) as f:
        src = f.read()
    if code.split("(")[0] in src:
        print(f"[{label}] already patched")
        return
    n = src.count(anchor)
    if n != 1:
        print(f"[{label}] anchor count={n} (need 1), skip")
        return
    src = src.replace(anchor, code + anchor, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"[{label}] patched")


mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
if mode == "revert":
    for p in [KM, PC]:
        with open(p) as f:
            lines = f.readlines()
        out = [l for l in lines if "APCDBG2" not in l]
        with open(p, "w") as f:
            f.writelines(out)
        print(f"{p}: removed {len(lines)-len(out)} lines")
    sys.exit(0)

# A. get_computed_blocks 入口
insert_after(
    KM,
    "    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:",
    '\n        print(f"APCDBG2 GET-COMPUTED req={request.request_id} enable_caching={self.enable_caching} "\n'
    '              f"skip={request.skip_reading_prefix_cache}", flush=True)\n',
    "A-get-computed")

# B. allocate 门控前
insert_before(
    KM,
    "        if not self.enable_caching or delay_cache_blocks:",
    '        print(f"APCDBG2 ALLOC-GATE enable_caching={self.enable_caching} delay={delay_cache_blocks}", flush=True)\n',
    "B-alloc-gate")

# C1. ascend find 入口
insert_before(
    PC,
    "        num_groups = len(self.kv_cache_config.kv_cache_groups)\n        hit_length = max_cache_hit_length",
    '        print(f"APCDBG2 ASC-FIND entry max_len={max_cache_hit_length}", flush=True)\n',
    "C1-asc-find-entry")

# C2. 每 group 命中结果
insert_after(
    PC,
    "                hit_blocks, _new_hit_length = hit_result",
    '\n                print(f"APCDBG2 ASC-HIT spec={type(spec).__name__} hit_len={_new_hit_length} "\n'
    '                      f"max={_max_length} groups={group_ids}", flush=True)\n',
    "C2-asc-hit")

py_compile.compile(KM, doraise=True)
py_compile.compile(PC, doraise=True)
print("COMPILE-OK")
