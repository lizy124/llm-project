#!/usr/bin/env python3
"""patch_apcdbg5.py — 第五轮: coordinator 入口 + FA 入口 + 修复 wrapper 栈打印."""
import py_compile

KM = "/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py"
KC = "/vllm-workspace/vllm/vllm/v1/core/kv_cache_coordinator.py"
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

# 0. 修复 wrapper 栈打印 (上一轮 genexpr 崩溃)
edit(
    KM,
    '            _st = " <- ".join(f"{f.filename.split(chr(47))[-1]}:{f.lineno}:{f.function}" for f in _tb.extract_stack()[-5:-1])\n            print(f"APCDBG4 WRAPPER req={request.request_id} n={num_computed_tokens} {_st}", flush=True)',
    '            _st = " <- ".join(_tb.format_stack()[-3:-1]).replace("\\n", " | ")\n            print(f"APCDBG4 WRAPPER req={request.request_id} n={num_computed_tokens} {_st}", flush=True)',
    "0-fix-wrapper")

# 1. 567 调用点加 coordinator 类型
edit(
    KM,
    'print(f"APCDBG4 ALLOC567 n={num_tokens_to_cache} new={num_new_tokens} "\n              f"comp={request.num_computed_tokens} req={request.request_id}", flush=True)',
    'print(f"APCDBG4 ALLOC567 n={num_tokens_to_cache} new={num_new_tokens} "\n              f"comp={request.num_computed_tokens} coord={type(self.coordinator).__name__} req={request.request_id}", flush=True)',
    "1-567-coordtype")

# 2. coordinator.cache_blocks 入口
edit(
    KC,
    '        for manager in self.single_type_managers:\n            manager.cache_blocks(',
    '        print(f"APCDBG5 COORD cls={type(self).__name__} n={num_computed_tokens} mgrs={len(self.single_type_managers)} id={id(self)%100000}", flush=True)\n        for manager in self.single_type_managers:\n            manager.cache_blocks(',
    "2-coord-entry")

# 3. FullAttentionManager.cache_blocks 入口
edit(
    SM,
    '    ) -> None:\n        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)\n        hash_block_size = self.block_pool.hash_block_size',
    '    ) -> None:\n        print(f"APCDBG5 FA-ENTRY cls={type(self).__name__} n={num_tokens} id={id(self)%100000}", flush=True)\n        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)\n        hash_block_size = self.block_pool.hash_block_size',
    "3-fa-entry")

for p in [KM, KC, SM]:
    py_compile.compile(p, doraise=True)
print("COMPILE-OK")
