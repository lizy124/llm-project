#!/usr/bin/env python3
"""patch_apcdbg4.py — 第四轮: 调用者栈回溯 (定位 cache_blocks(0) 的真凶)."""
import py_compile

KM = "/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py"

with open(KM) as f:
    src = f.read()

# 1. 567 调用点: 打印输入值
a1 = "        self.coordinator.cache_blocks(request, num_tokens_to_cache)"
assert src.count(a1) == 1
if "APCDBG4 ALLOC567" not in src:
    src = src.replace(a1,
        '        print(f"APCDBG4 ALLOC567 n={num_tokens_to_cache} new={num_new_tokens} "\n'
        '              f"comp={request.num_computed_tokens} req={request.request_id}", flush=True)\n' + a1, 1)

# 2. 773 wrapper: 打印值 + 调用栈
a2 = "            self.coordinator.cache_blocks(request, num_computed_tokens)"
assert src.count(a2) == 1, f"count={src.count(a2)}"
if "APCDBG4 WRAPPER" not in src:
    src = src.replace(a2,
        '            import traceback as _tb\n'
        '            _st = " <- ".join(f"{f.filename.split(chr(47))[-1]}:{f.lineno}:{f.function}" for f in _tb.extract_stack()[-5:-1])\n'
        '            print(f"APCDBG4 WRAPPER req={request.request_id} n={num_computed_tokens} {_st}", flush=True)\n' + a2, 1)

with open(KM, "w") as f:
    f.write(src)
py_compile.compile(KM, doraise=True)
print("COMPILE-OK")
