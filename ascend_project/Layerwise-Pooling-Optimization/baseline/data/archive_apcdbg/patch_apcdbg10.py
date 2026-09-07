#!/usr/bin/env python3
"""patch_apcdbg10.py — 第十轮: BlockHashToBlockMap.pop 打印调用者栈 (抓清表真凶)."""
import py_compile

BP = "/vllm-workspace/vllm/vllm/v1/core/block_pool.py"

with open(BP) as f:
    src = f.read()

if "APCDBG10" in src:
    print("already")
else:
    old = '''    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)'''
    new = '''    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        import traceback as _tb
        _frames = _tb.extract_stack()[-3:-1]
        _callers = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}:{f.function}" for f in _frames)
        print(f"APCDBG10 POP table_n={len(self._cache)} callers={_callers}", flush=True)
        blocks = self._cache.pop(key, None)'''
    n = src.count(old)
    assert n == 1, f"count={n}"
    src = src.replace(old, new, 1)
    with open(BP, "w") as f:
        f.write(src)
    print("patched")

py_compile.compile(BP, doraise=True)
print("COMPILE-OK")
