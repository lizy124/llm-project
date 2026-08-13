import pathlib
import re
import subprocess
from collections import Counter, defaultdict

ROOT = pathlib.Path(r"D:\lzy\project\kv_pool\code\vllm-ascend")
OUT = pathlib.Path(r"D:\lzy\project\kv_pool\llm-project\transfer_data\envs\vllm-ascend-tutorial-models-environment-variables.md")
files = subprocess.check_output(["git", "ls-files", "docs/source/tutorials/models"], cwd=ROOT, text=True, encoding="utf-8").splitlines()

patterns = [
    ("export", re.compile(r"(?m)(?:^|[;&|]\s*)export\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("inline assignment", re.compile(r"(?m)(?:^|[;&|\s])([A-Z][A-Z0-9_]*)=(?:\"[^\n\"]*\"|'[^\n']*'|[^\s\\;]+)\s+(?=(?:python|python3|pytest|torchrun|vllm|docker|kubectl|make|bash|sh|pip)\b)")),
    ("docker -e/--env", re.compile(r"(?:--env(?:=|\s+)|(?:^|\s)-e\s+)([A-Z][A-Z0-9_]*)")),
    ("Python environment API", re.compile(r"(?:os\.environ(?:\.get)?\s*\[?\s*|getenv\s*\(\s*)[\"']([A-Z][A-Z0-9_]*)[\"']")),
]
mention_rx = re.compile(r"(?i)(?:environment variable|env(?:ironment)? var(?:iable)?|环境变量)[^`\n]{0,100}`([A-Z][A-Z0-9_]*)`|`([A-Z][A-Z0-9_]*)`[^`\n]{0,100}(?:environment variable|env(?:ironment)? var(?:iable)?|环境变量)")
refs = defaultdict(lambda: defaultdict(list))
documents = defaultdict(set)

for rel in files:
    path = ROOT / rel
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    for mechanism, rx in patterns:
        for match in rx.finditer(text):
            if mechanism == "inline assignment":
                line_start = text.rfind("\n", 0, match.start()) + 1
                if re.search(r"\bexport\s+$", text[line_start:match.start()]):
                    continue
            var = next((group for group in match.groups() if group), None)
            if var:
                location = f"{rel}:{text.count(chr(10), 0, match.start()) + 1}"
                refs[var][mechanism].append(location)
                documents[var].add(rel)
    for match in mention_rx.finditer(text):
        var = match.group(1) or match.group(2)
        location = f"{rel}:{text.count(chr(10), 0, match.start()) + 1}"
        refs[var]["正文明确提及"].append(location)
        documents[var].add(rel)

explicit = {"export", "inline assignment", "docker -e/--env", "Python environment API", "正文明确提及"}
records = []
for name in sorted(refs):
    mechanisms = sorted(refs[name])
    if not set(mechanisms) & explicit:
        continue
    records.append({"name": name, "mechanisms": mechanisms, "locations": sorted({x for xs in refs[name].values() for x in xs}), "documents": documents[name]})

def classify(name):
    if name.startswith("VLLM_ASCEND_") or name in {"DYNAMIC_EPLB", "MSMONITOR_USE_DAEMON"}:
        return "vLLM Ascend 产品配置"
    if name.startswith(("ASCEND_", "HCCL_", "LCCL_", "ATB_", "TASK_QUEUE_", "CPU_AFFINITY_")):
        return "Ascend/CANN/HCCL 与 NPU 运行时"
    if name in {"GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_IF_IP", "HCCL_SOCKET_IFNAME", "OMP_NUM_THREADS", "OMP_PROC_BIND"}:
        return "分布式通信与并行运行环境"
    if name.startswith(("MOONCAKE_", "MMC_", "MEMFABRIC_", "YR_")):
        return "KV Transfer 与外部存储"
    if name.startswith(("VLLM_", "PYTORCH_", "TORCH_", "HF_", "TOKENIZERS_", "USE_MODELSCOPE")):
        return "上游 vLLM/PyTorch/模型生态"
    if name in {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHASHSEED", "PYTHONPATH", "HF_HOME"}:
        return "系统与通用运行环境"
    if name in {"IMAGE", "NAME", "DEVICE", "MODEL_PATH", "MODEL", "TAG", "IMAGE_TAG", "IMAGE_NAME", "local_ip", "nic_name", "node0_ip", "local_IP", "server_ip", "server_port"}:
        return "文档示例辅助变量"
    if name.startswith(("HUNYUAN", "MINIMAX", "QWEN", "GLM", "DEEPSEEK", "GEMMA", "PADDLE", "MODEL")):
        return "模型/评测专用变量"
    return "其他文档环境变量"

for record in records:
    record["category"] = classify(record["name"])

categories = ["vLLM Ascend 产品配置", "Ascend/CANN/HCCL 与 NPU 运行时", "分布式通信与并行运行环境", "KV Transfer 与外部存储", "上游 vLLM/PyTorch/模型生态", "系统与通用运行环境", "文档示例辅助变量", "模型/评测专用变量", "其他文档环境变量"]
count = Counter(r["category"] for r in records)
mechanism_count = Counter(m for r in records for m in r["mechanisms"])
doc_count = Counter(doc for r in records for doc in r["documents"])
lines = [
    "# `docs/source/tutorials/models` 环境变量清单",
    "",
    f"- 扫描目录：`D:/lzy/project/kv_pool/code/vllm-ascend/docs/source/tutorials/models`",
    f"- 文档文件数：**{len(files)}**",
    f"- 明确环境变量语义的唯一名称：**{len(records)}**",
    "- 统计范围：模型教程中的 `export`、行内进程赋值、Docker `-e/--env`、Python 环境 API，以及正文明确说明为环境变量的名称。",
    "- `IMAGE`、`MODEL_PATH`、`TAG`、`nic_name` 等命令辅助变量会保留，但单独分类，不视为产品配置。",
    "",
    "## 分类统计",
    "",
    "| 分类 | 变量数 | 占比 |",
    "|---|---:|---:|",
]
for category in categories:
    lines.append(f"| {category} | {count[category]} | {count[category] / len(records) * 100:.1f}% |")
lines.append(f"| **合计** | **{len(records)}** | **100.0%** |")
lines += ["", "## 使用形式统计", "", "同一变量可出现多种使用形式，统计存在交集。", "", "| 使用形式 | 变量数 |", "|---|---:|"]
for mech, n in mechanism_count.most_common():
    lines.append(f"| {mech} | {n} |")
lines += ["", "## 覆盖最多的模型文档", "", "| 文档 | 环境变量数 |", "|---|---:|"]
for doc, n in doc_count.most_common(20):
    lines.append(f"| `{doc.replace('docs/source/tutorials/models/', '')}` | {n} |")
lines += ["", "## 分类明细", "", "位置使用 `docs/source/tutorials/models` 下的相对路径和行号；每个变量最多展示 8 个代表位置。"]
for category in categories:
    subset = sorted((r for r in records if r["category"] == category), key=lambda r: r["name"])
    lines += ["", f"### {category}（{len(subset)}）", "", "| 变量 | 使用形式 | 出现文档数 | 文档位置（示例） |", "|---|---|---:|---|"]
    for r in subset:
        locations = "; ".join(r["locations"][:8])
        lines.append(f"| `{r['name']}` | {', '.join(r['mechanisms'])} | {len(r['documents'])} | `{locations}` |")
lines += ["", "## 口径说明", "", "1. 本文只覆盖 `docs/source/tutorials/models`，不扩展到源码、测试或其他文档目录。", "2. 文档中的环境变量可能属于 vLLM Ascend、CANN/HCCL、PyTorch 或宿主系统；分类按主要用途组织。", "3. 普通 Shell 局部赋值（如 `nic_name=...`、`MODEL_NAME=...`）不会因为被命令引用就自动算作环境变量；只有 `export`、命令前缀赋值等明确进入子进程环境的名称才纳入。"]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(records)} variables)")
