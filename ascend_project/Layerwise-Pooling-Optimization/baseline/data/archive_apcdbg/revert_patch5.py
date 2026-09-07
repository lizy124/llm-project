#!/usr/bin/env python3
"""revert_patch5.py — 撤掉 APCDBG5 埋点, 简化 wrapper 打印 (排除启动失败嫌疑)."""
import py_compile

KM = "/vllm-workspace/vllm/vllm/v1/core/kv_cache_manager.py"
KC = "/vllm-workspace/vllm/vllm/v1/core/kv_cache_coordinator.py"
SM = "/vllm-workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py"

def edit(path, old, new, label):
    with open(path) as f:
        src = f.read()
    n = src.count(old)
    if n == 0:
        print(f"[{label}] anchor not found (maybe already reverted)")
        return
    if n != 1:
        print(f"[{label}] anchor count={n}, skip")
        return
    with open(path, "w") as f:
        f.write(src.replace(old, new, 1))
    print(f"[{label}] reverted")

# 1. wrapper: 去掉栈打印, 只留 n
edit(
    KM,
    '            import traceback as _tb\n            _st = " <- ".join(_tb.format_stack()[-3:-1]).replace("\\n", " | ")\n            print(f"APCDBG4 WRAPPER req={request.request_id} n={num_computed_tokens} {_st}", flush=True)',
    '            print(f"APCDBG4 WRAPPER req={request.request_id} n={num_computed_tokens}", flush=True)',
    "1-wrapper")

# 2. 去掉 567 的 coord 类型
edit(
    KM,
    ' f"comp={request.num_computed_tokens} coord={type(self.coordinator).__name__} req={request.request_id}", flush=True)',
    ' f"comp={request.num_computed_tokens} req={request.request_id}", flush=True)',
    "2-567")

# 3. 去掉 COORD 入口
edit(
    KC,
    '        print(f"APCDBG5 COORD cls={type(self).__name__} n={num_computed_tokens} mgrs={len(self.single_type_managers)} id={id(self)%100000}", flush=True)\n',
    '',
    "3-coord")

# 4. 去掉 FA 入口
edit(
    SM,
    '        print(f"APCDBG5 FA-ENTRY cls={type(self).__name__} n={num_tokens} id={id(self)%100000}", flush=True)\n',
    '',
    "4-fa")

for p in [KM, KC, SM]:
    py_compile.compile(p, doraise=True)
print("COMPILE-OK")
