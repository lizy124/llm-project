#!/usr/bin/env python3
"""patch_apcdbg.py — 给 vllm 打临时 APC 调试日志 (验证容器专用, 可回滚).

埋点:
  1. kv_cache_coordinator.py (hybrid find_longest_cache_hit): 每 group 命中长度
  2. single_type_kv_cache_manager.py (base cache_blocks): 每管理器缓存调用
  3. block_pool.py (cache_blocks): 每组注册明细 (块数/掩码)
用法: python3 patch_apcdbg.py apply|revert
"""
import re
import sys

COORD = "/vllm-workspace/vllm/vllm/v1/core/kv_cache_coordinator.py"
STM = "/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py"
POOL = "/vllm-workspace/vllm/vllm/v1/core/block_pool.py"

TAG = "APCDBG"


def patch(path, anchor, insert, label):
    with open(path) as f:
        src = f.read()
    if TAG in src:
        print(f"[{label}] already patched")
        return
    if anchor not in src:
        print(f"[{label}] ANCHOR NOT FOUND, skip")
        return
    src = src.replace(anchor, insert + anchor, 1)
    # 确保 logger 可用
    if "logger = logging.getLogger" not in src:
        if re.search(r"^import logging$", src, re.M) is None:
            src = "import logging\n" + src
        src = re.sub(r"^(import logging\n)", r"\1logger = logging.getLogger(__name__)\n", src, count=1, flags=re.M) if False else src
    with open(path, "w") as f:
        f.write(src)
    print(f"[{label}] patched")


def revert(path, label):
    with open(path) as f:
        lines = f.readlines()
    out = [l for l in lines if TAG not in l]
    if len(out) != len(lines):
        with open(path, "w") as f:
            f.writelines(out)
        print(f"[{label}] reverted ({len(lines)-len(out)} lines removed)")
    else:
        print(f"[{label}] nothing to revert")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
    if mode == "revert":
        for p, lb in [(COORD, "coord"), (STM, "stm"), (POOL, "pool")]:
            revert(p, lb)
        return

    # 1. hybrid 命中埋点: 插在 eagle 判定前 (该上下文唯一)
    anchor1 = """                if drop_eagle_block:
                    eagle_verified.add(idx)"""
    ins1 = (f'                logger.info("{TAG} HIT spec=%%s hit_len=%%s max_len=%%s", '
            f'type(spec).__name__, _new_hit_length, _max_length)\n')
    # logger.info 的 %% 防止后续 replace 语义干扰 — 直接写 %s
    ins1 = (f'                logger.info("{TAG} HIT spec=%s hit_len=%s max_len=%s", '
            f'type(spec).__name__, _new_hit_length, _max_length)\n')
    patch(COORD, anchor1, ins1, "coord-hit")

    # 2. 基类 cache_blocks 入口
    with open(STM) as f:
        stm_src = f.read()
    m = re.search(r"(    def cache_blocks\(\n(?:.*\n)*?    \) -> None:\n)", stm_src)
    if m and TAG not in stm_src:
        sig = m.group(1)
        ins2 = (f'        logger.info("{TAG} CACHE mgr=%s req=%s num_tokens=%s", '
                f'type(self).__name__, request.request_id, num_tokens)\n')
        stm_src = stm_src.replace(sig, sig + ins2, 1)
        if "logger = logging.getLogger" not in stm_src:
            stm_src = stm_src.replace("import logging\n", "import logging\nlogger = logging.getLogger(__name__)\n", 1)
        with open(STM, "w") as f:
            f.write(stm_src)
        print("[stm-cache] patched")
    else:
        print("[stm-cache] skip (no match or already)")

    # 3. BlockPool.cache_blocks 参数埋点
    anchor3 = """        if num_cached_blocks >= num_full_blocks:
            return"""
    ins3 = (f'        logger.info("{TAG} POOL grp=%s n_cached=%s n_full=%s bs=%s mask=%s", '
            f'kv_cache_group_id, num_cached_blocks, num_full_blocks, block_size, '
            f'None if block_mask is None else (sum(block_mask), len(block_mask)))\n')
    patch(POOL, anchor3, ins3, "pool-cache")


if __name__ == "__main__":
    main()
