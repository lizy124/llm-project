import pathlib
import re
import subprocess
from collections import defaultdict, Counter

ROOT = pathlib.Path(r"D:\lzy\project\kv_pool\code\vllm-ascend")
OUT = pathlib.Path(r"D:\lzy\project\kv_pool\llm-project\transfer_data\envs\vllm-ascend-docs-environment-variables.md")
files = subprocess.check_output(["git", "ls-files", "docs/source"], cwd=ROOT, text=True, encoding="utf-8").splitlines()

patterns = [
    ("export", re.compile(r"(?m)(?:^|[;&|]\s*)export\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("inline assignment", re.compile(r"(?m)(?:^|[;&|\s])([A-Z][A-Z0-9_]*)=(?:\"[^\n\"]*\"|'[^\n']*'|[^\s\\;]+)\s+(?=(?:python|python3|pytest|torchrun|vllm|docker|kubectl|make|bash|sh)\b)")),
    ("docker -e/--env", re.compile(r"(?:--env(?:=|\s+)|(?:^|\s)-e\s+)([A-Z][A-Z0-9_]*)")),
    ("shell variable reference", re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}|\$([A-Z][A-Z0-9_]*)")),
    ("Python environment API", re.compile(r"(?:os\.environ(?:\.get)?\s*\[?\s*|getenv\s*\(\s*)[\"']([A-Z][A-Z0-9_]*)[\"']")),
]

# Explicit mentions in prose/backticks that are clearly environment variables.
mention_rx = re.compile(r"(?i)(?:environment variable|env(?:ironment)? var(?:iable)?|环境变量)[^`\n]{0,100}`([A-Z][A-Z0-9_]*)`|`([A-Z][A-Z0-9_]*)`[^`\n]{0,100}(?:environment variable|env(?:ironment)? var(?:iable)?|环境变量)")
refs = defaultdict(lambda: defaultdict(list))

def line_no(text, pos):
    return text.count("\n", 0, pos) + 1

for rel in files:
    path = ROOT / rel
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    for mechanism, rx in patterns:
        for match in rx.finditer(text):
            var = next((group for group in match.groups() if group), None)
            if var:
                refs[var][mechanism].append(f"{rel}:{line_no(text, match.start())}")
    for match in mention_rx.finditer(text):
        var = match.group(1) or match.group(2)
        refs[var]["正文明确提及"].append(f"{rel}:{line_no(text, match.start())}")

def classify(var, mechanisms, rels):
    if var.startswith("VLLM_ASCEND_") or var in {"DYNAMIC_EPLB", "MSMONITOR_USE_DAEMON"}:
        return "vLLM Ascend 产品配置"
    if var.startswith(("ASCEND_", "HCCL_", "LCCL_", "ATB_", "PYTORCH_NPU_", "TASK_QUEUE_", "CPU_AFFINITY_")):
        return "Ascend/CANN/HCCL 与 NPU 运行时"
    if var in {"WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE", "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"}:
        return "分布式启动与网络"
    if var.startswith(("MOONCAKE_", "MMC_", "MEMFABRIC_", "YR_", "RFORK_")) or var in {"DATASYSTEM_CLIENT_LOG_DIR", "NETLOADER_CONFIG"}:
        return "KV Transfer 与外部存储"
    if var.startswith(("VLLM_", "HF_", "TORCH_", "PYTORCH_", "MODELSCOPE_", "TOKENIZERS_")):
        return "上游 vLLM/PyTorch/模型生态"
    if var in {"PATH", "PYTHONPATH", "HOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "OMP_NUM_THREADS", "OMP_PROC_BIND", "PYTHONHASHSEED", "TERM"}:
        return "系统与通用运行环境"
    if var in {"IMAGE", "DEVICE", "TAG", "IMAGE_TAG", "IMAGE_NAME", "SRC_WORKSPACE", "WORKSPACE", "MODEL", "MODEL_PATH", "SERVER_PORT", "MASTER_IP", "MASTER_IP_ADDRESS", "NETWORK_CARD_NAME", "IP_ADDRESS", "NAME", "SAVE_PATH", "CONFIG_YAML_PATH", "CONFIG_BASE_PATH", "PROFILING_SYMBOLS_PATH", "SERVICE_PROF_CONFIG_PATH", "path_to_store_profiling_files"}:
        return "文档示例辅助变量"
    if var.startswith(("GITHUB_", "GH_", "AWS_", "OBS_", "QUAY_", "PIP_", "UV_", "AIS_", "BENCHMARK_")):
        return "CI/构建/发布辅助变量"
    if any(part.startswith("tutorials/models") or part.startswith("developer_guide/evaluation") for part in rels):
        return "模型与评测示例变量"
    return "文档中的其他环境变量"

explicit_mechanisms = {"export", "inline assignment", "docker -e/--env", "Python environment API", "正文明确提及"}
records = []
for var in sorted(refs):
    mechanisms = sorted(refs[var])
    # A bare $VAR reference may be a normal shell variable. Keep references only
    # when another occurrence proves that the name has environment semantics.
    if not (set(mechanisms) & explicit_mechanisms):
        continue
    locations = sorted({loc for vals in refs[var].values() for loc in vals})
    rels = [loc.rsplit(":", 1)[0].replace("docs/source/", "") for loc in locations]
    records.append({"name": var, "mechanisms": mechanisms, "locations": locations, "rels": rels, "category": classify(var, mechanisms, rels)})

categories = ["vLLM Ascend 产品配置", "Ascend/CANN/HCCL 与 NPU 运行时", "分布式启动与网络", "KV Transfer 与外部存储", "上游 vLLM/PyTorch/模型生态", "系统与通用运行环境", "文档示例辅助变量", "模型与评测示例变量", "CI/构建/发布辅助变量", "文档中的其他环境变量"]
count = Counter(record["category"] for record in records)
mechanism_count = Counter(mech for record in records for mech in record["mechanisms"])
area_pairs = {(record["name"], rel.split("/", 1)[0]) for record in records for rel in record["rels"]}
area_count = Counter(area for _, area in area_pairs)
lines = [
    "# vLLM Ascend `docs/source` 环境变量清单",
    "",
    f"- 扫描范围：`D:/lzy/project/kv_pool/code/vllm-ascend/docs/source` 下的 Git 跟踪文件（共 {len(files)} 个）。",
    f"- 文档中发现的唯一环境变量名称：**{len(records)}**。",
    "- 纳入形式：`export`、命令行行内进程赋值、Docker `-e/--env`、Python 环境 API，以及明确写成“环境变量”的文档说明。`${VAR}`/`$VAR` 只有在同名变量已被上述方式确认时才作为补充证据。",
    "- 示例中的 `IMAGE`、`DEVICE`、`TAG` 等也会记录，但单独归入“文档示例辅助变量”，不表示它们是 vLLM Ascend 产品配置。",
    "",
    "## 分类统计",
    "",
    "| 分类 | 数量 | 占比 | 说明 |",
    "|---|---:|---:|---|",
]
for category in categories:
    lines.append(f"| {category} | {count[category]} | {count[category] / len(records) * 100:.1f}% | |")
lines.append(f"| **合计** | **{len(records)}** | **100.0%** | 唯一变量名称，分类互斥 |")
lines += ["", "## 使用形式统计", "", "同一变量可能有多种使用形式，因此本表不要求合计等于唯一变量总数。", "", "| 使用形式 | 变量数 |", "|---|---:|"]
for mechanism, n in mechanism_count.most_common():
    lines.append(f"| {mechanism} | {n} |")
lines += ["", "## 文档目录分布", "", "| 文档目录 | 变量出现次数（去重变量-目录组合） |", "|---|---:|"]
for area, n in area_count.most_common():
    lines.append(f"| `{area}` | {n} |")
lines += ["", "## 分类明细", "", "位置使用 `docs/source` 下的相对路径和行号。每个变量最多列出 6 个代表位置；完整命中可通过仓库搜索复核。"]
for category in categories:
    subset = [record for record in records if record["category"] == category]
    lines += ["", f"### {category}（{len(subset)}）", "", "| 变量 | 使用形式 | 文档位置（示例） |", "|---|---|---|"]
    for record in subset:
        lines.append(f"| `{record['name']}` | {', '.join(record['mechanisms'])} | {'; '.join(record['locations'][:6])} |")
lines += ["", "## 说明", "", "1. 本文只统计 `docs/source`，不包含仓库源码、测试、CI 或其他目录中的独立命中。", "2. 文档中的环境变量可能只是外部组件（CANN/HCCL/PyTorch/Hugging Face）的配置，仓库不一定读取它们。", "3. `${PWD}`、`${HOME}` 等系统变量在文档中被引用时也保留，因为它们确实参与了示例命令的环境传播；普通 Shell 局部变量若没有环境变量语义则不纳入。", "4. 变量实际作用、默认值和源码读取位置请结合主清单 `vllm-ascend-environment-variables.md` 一起查看。"]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(records)} variables)")
