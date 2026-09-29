"""Proposal §5 step-1 static check: yaml.safe_load + JSON args json.loads + real kv_pool parser."""
import importlib.util
import json
import sys

import yaml

REPO = r"C:\E_segment\project\code\vllm-ascend"
FILES = [
    REPO + r"\tests\e2e\nightly\multi_node\external_dp\config\DeepSeek-V4-flash-w8a8-PD-prefix.yaml",
    REPO + r"\tests\e2e\nightly\single_node\models\configs\DeepSeek_V4_Flash_A5.yaml",
    REPO + r"\tests\e2e\nightly\single_node\models\configs\acceptace_rate_deepseekv4-flash-w8a8-mtp.yaml",
    REPO + r"\tests\e2e\nightly\single_node\models\configs\DeepSeek-V4-Flash-W8A8-A3.yaml",
]

spec = importlib.util.spec_from_file_location(
    "kvpool_config", REPO + r"\tests\e2e\common\kv_pool\config.py")
kvpool_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kvpool_config)

json_fail = 0


def walk_json_strings(obj, path="$"):
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") and s.endswith("}"):
            yield path, s
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_json_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_json_strings(v, f"{path}[{i}]")


def count_kv_transfer(obj):
    n = 0
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            if v == "--kv-transfer-config":
                n += 1
            else:
                n += count_kv_transfer(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            n += count_kv_transfer(v)
    return n


def find_kv_pools(obj):
    pools = []
    if isinstance(obj, dict):
        if obj.get("kv_pool") is not None:
            pools.append(obj["kv_pool"])
        for v in obj.values():
            if isinstance(v, (dict, list)):
                pools.extend(find_kv_pools(v))
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                pools.extend(find_kv_pools(v))
    return pools


for f in FILES:
    name = f.split("\\")[-1]
    with open(f, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    print(f"== {name}: yaml.safe_load OK")
    for p, s in walk_json_strings(data):
        try:
            json.loads(s)
        except Exception as e:
            global_count = None
            print(f"   [FAIL] json.loads {p}: {e}")
            json_fail += 1
    print(f"   --kv-transfer-config count: {count_kv_transfer(data)}")
    for raw in find_kv_pools(data):
        cfg = kvpool_config.parse_kv_pool_config(raw)
        kvpool_config.validate_kv_pool_config(cfg)
        ports = {k: getattr(cfg, k) for k in ("master_port", "metrics_port",
                                              "meta_service_port", "config_store_port")
                 if hasattr(cfg, k)}
        print(f"   kv_pool parse+validate OK: type={cfg.type} ports={ports}")

print("RESULT:", "ALL OK" if json_fail == 0 else f"{json_fail} JSON FAILURES")
sys.exit(1 if json_fail else 0)
