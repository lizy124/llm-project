import pathlib
import re
from collections import Counter, defaultdict

ROOT = pathlib.Path(r"D:\lzy\project\kv_pool\code\vllm-ascend\docs\source")
OUT = pathlib.Path(r"D:\lzy\project\kv_pool\llm-project\transfer_data\envs\vllm-ascend-service-inference-environment-variables.md")

LAUNCH_RX = re.compile(
    r"vllm\s+serve|vllm\.entrypoints\.(?:openai\.)?api_server|"
    r"python(?:3)?\s+-m\s+vllm|torchrun\b|"
    r"(?:python|python3)\s+[^\n]*(?:server|serve|proxy|inference|infer|offline)[^\n]*\.py|"
    r"kubectl\s+(?:apply|create)|ray\s+start",
    re.I,
)
INFERENCE_RX = re.compile(
    r"curl\s+[^\n]*(?:/v1/|/generate|/invocations)|"
    r"vllm\s+bench\s+serve|benchmark_serving|"
    r"(?:python|python3)\s+[^\n]*(?:client|request|benchmark|inference|infer)[^\n]*\.py",
    re.I,
)
ENV_PATTERNS = [
    ("export", re.compile(r"(?m)^[ \t]*export\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|$)")),
    ("行内进程赋值", re.compile(r"(?m)^[ \t]*([A-Z][A-Z0-9_]*)=(?:\"[^\n\"]*\"|'[^\n']*'|[^\s\\;]+)\s+(?=(?:python|python3|pytest|torchrun|vllm|bash|sh|pip)\b)")),
    ("Docker -e/--env", re.compile(r"(?:--env(?:=|\s+)|(?:^|\s)-e\s+)([A-Z][A-Z0-9_]*)", re.M)),
    ("Python os.environ", re.compile(r"os\.environ(?:\.get)?\s*\[?\s*[\"']([A-Z][A-Z0-9_]*)[\"']")),
]

# These names construct examples but are not configuration consumed by the
# vLLM service/inference process itself.
AUXILIARY = {
    "IMAGE", "IMAGE_NAME", "IMAGE_TAG", "TAG", "NAME", "DEVICE", "MODEL", "MODEL_PATH",
    "MODEL_NAME", "SERVER_PORT", "MASTER_IP", "MASTER_IP_ADDRESS", "IP_ADDRESS",
    "NETWORK_CARD_NAME", "IFNAME", "WORKSPACE", "SRC_WORKSPACE", "SAVE_PATH",
    "IMAGE_PATH", "IMAGE_BASE64", "INPUT_LEN", "OUTPUT_LEN", "NUM_PROMPTS",
    "SERVICE_PROF_CONFIG_PATH", "PROFILING_SYMBOLS_PATH", "DEVICE0", "DEVICE1", "ENDPOINT",
    "SOC_VERSION",
}
STALE_DOC_VARS = {
    "VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": (
        "曾用于 v0.9.1 sampler TopK/TopP patch；提交 830332ebf（2025-07-09，Clean up v0.9.1 code）"
        "已从 envs.py、patch 和测试删除。当前 GLM4.x 文档为遗留配置。"
    ),
    "VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED": (
        "仅存在于提交 e3636c7eb（2025-08-05，明确标注 0.9.1 only）的兼容实现；该提交不是当前 main 的祖先，"
        "当前源码没有注册或读取该变量。large_scale_ep.md 为旧文档迁移残留。"
    ),
    "VLLM_DP_SIZE_LOCAL": (
        "与 VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED 一同由提交 e3636c7eb（2025-08-05，明确标注 0.9.1 only）"
        "加入，用于旧版 external DP 的本机 DP size；当前 main 的上游 vLLM 与 vLLM Ascend 源码均未注册或读取。"
        "large_scale_ep.md 中的设置属于旧兼容实现残留。"
    ),
}

# Runtime product settings documented in additional_config.md. These are
# included even when the migration-period environment variable is not exported
# next to a launch command.
RUNTIME_CONFIG_SUPPLEMENTS = {
    "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": (
        "源码 os.getenv",
        ["docs/source/user_guide/configuration/additional_config.md:25"],
        ["Additional Configuration > Environment Variable Migration > 推荐使用 enable_transpose_kv_cache_by_block"],
    ),
}


def split_sections(text):
    headings = [(m.start(), len(m.group(1)), m.group(2).strip()) for m in re.finditer(r"(?m)^(#{1,6})\s+(.+)$", text)]
    if not headings:
        return [("文档", 0, len(text), text)]
    result = []
    stack = []
    for idx, (start, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        result.append((" > ".join(item[1] for item in stack), start, end, text[start:end]))
    return result


def classify(name):
    if name in STALE_DOC_VARS:
        return "文档遗留/当前源码不支持"
    if name.startswith("VLLM_ASCEND_") or name in {"DYNAMIC_EPLB", "EXPERT_MAP_RECORD", "MSMONITOR_USE_DAEMON"}:
        return "vLLM Ascend 产品配置"
    if name.startswith(("ASCEND_", "HCCL_", "LCCL_", "ATB_", "TASK_QUEUE_", "CPU_AFFINITY_")):
        return "Ascend/CANN/HCCL 与 NPU 运行时"
    if name in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "OMP_NUM_THREADS", "OMP_PROC_BIND", "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"} or name.startswith("VLLM_DP_"):
        return "分布式启动与并行环境"
    if name.startswith(("MOONCAKE_", "MMC_", "MEMFABRIC_", "YR_")) or name in {"ASCEND_TRANSFER_TIMEOUT", "ASCEND_CONNECT_TIMEOUT", "ASCEND_ENABLE_USE_FABRIC_MEM", "ASCEND_GLOBAL_RESOURCE_CONFIG"}:
        return "KV Transfer、PD 分离与存储后端"
    if name.startswith(("VLLM_", "PYTORCH_", "TORCH_", "HF_", "TOKENIZERS_", "RAY_")):
        return "上游 vLLM/PyTorch/模型生态"
    if name in {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHASHSEED", "OMP_NUM_THREADS", "OMP_PROC_BIND"}:
        return "系统运行环境"
    return "其他服务运行变量"


SCENARIO_REQUIRED = {
    "HCCL_IF_IP", "HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", "MOONCAKE_CONFIG_PATH",
    "RFORK_CONFIG", "NETLOADER_CONFIG", "VLLM_HOST_IP", "VLLM_DP_MASTER_IP",
    "VLLM_DP_MASTER_PORT", "VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
    "VLLM_DP_SIZE_LOCAL",
}
SCENARIO_CONFIG = {
    "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_A3_ENABLE", "ASCEND_ENABLE_USE_FABRIC_MEM",
    "ASCEND_GLOBAL_RESOURCE_CONFIG", "PYTHONPATH", "LD_LIBRARY_PATH", "VLLM_USE_MODELSCOPE",
    "USE_MODELSCOPE_HUB", "TIKTOKEN_ENCODINGS_BASE", "NPU_MEMORY_FRACTION",
}


def necessity(name):
    if name in STALE_DOC_VARS:
        return "不应使用"
    if name in SCENARIO_REQUIRED:
        return "场景必需"
    if name in SCENARIO_CONFIG:
        return "场景配置"
    return "可选调优/调试"


refs = defaultdict(lambda: defaultdict(list))
scenarios = defaultdict(set)
excluded = defaultdict(list)
launch_sections = 0
launch_docs = set()
all_files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".rst", ".yml", ".yaml"}]

for path in all_files:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    rel = path.relative_to(ROOT).as_posix()
    sections = split_sections(text)
    # Markdown deployment flows often place environment exports and launch
    # commands in adjacent subsections. Also inspect the parent heading group.
    launch_indices = {i for i, (_, _, _, body) in enumerate(sections) if LAUNCH_RX.search(body) or INFERENCE_RX.search(body)}
    relevant = set(launch_indices)
    for i in launch_indices:
        relevant.update(j for j in range(max(0, i - 2), min(len(sections), i + 3)))
    for i in sorted(relevant):
        title, start, _, body = sections[i]
        # Nearby sections are accepted only when they contain explicit env
        # syntax; this prevents unrelated prose from entering the inventory.
        found_any = False
        for mechanism, rx in ENV_PATTERNS:
            for match in rx.finditer(body):
                name = match.group(1)
                absolute = start + match.start()
                line = text.count("\n", 0, absolute) + 1
                location = f"docs/source/{rel}:{line}"
                found_any = True
                if name in AUXILIARY:
                    excluded[name].append(location)
                    continue
                refs[name][mechanism].append(location)
                scenarios[name].add(title)
        if i in launch_indices:
            launch_sections += 1
            launch_docs.add(rel)

records = []
for name in sorted(refs):
    mechanisms = sorted(refs[name])
    locations = sorted({loc for values in refs[name].values() for loc in values})
    records.append({"name": name, "category": classify(name), "necessity": necessity(name), "mechanisms": mechanisms, "locations": locations, "scenarios": sorted(scenarios[name])})

for name, (mechanism, locations, scenario_titles) in RUNTIME_CONFIG_SUPPLEMENTS.items():
    if name not in refs:
        records.append({
            "name": name,
            "category": classify(name),
            "necessity": necessity(name),
            "mechanisms": [mechanism],
            "locations": locations,
            "scenarios": scenario_titles,
        })
records.sort(key=lambda record: record["name"])

categories = [
    "vLLM Ascend 产品配置", "Ascend/CANN/HCCL 与 NPU 运行时", "分布式启动与并行环境",
    "KV Transfer、PD 分离与存储后端", "上游 vLLM/PyTorch/模型生态", "系统运行环境", "其他服务运行变量",
    "文档遗留/当前源码不支持",
]
category_count = Counter(record["category"] for record in records)
mechanism_count = Counter(mechanism for record in records for mechanism in record["mechanisms"])
necessity_count = Counter(record["necessity"] for record in records)

lines = [
    "# vLLM 服务启动与推理环境变量（`docs/source`）",
    "",
    f"- 扫描目录：`D:/lzy/project/kv_pool/code/vllm-ascend/docs/source`",
    f"- 扫描文档文件：**{len(all_files)}**",
    f"- 识别到含服务启动或推理命令的文档：**{len(launch_docs)}**",
    f"- 启动/推理相关章节：**{launch_sections}**",
    f"- 服务启动、推理及运行时配置主表变量：**{len(records)}**",
    f"- 当前源码可用变量：**{len(records) - len(STALE_DOC_VARS)}**",
    f"- 文档遗留、当前源码不支持：**{len(STALE_DOC_VARS)}**",
    "",
    "> 本文不是 `docs/source` 环境变量全集。主表以 `vllm serve`、API server、服务脚本、分布式启动、离线推理或推理客户端启动链路为主，并补充 additional config 中直接影响服务运行时的迁移变量。镜像名、模型路径、端口占位等 Shell 辅助变量被排除。",
    "",
    "## 分类统计",
    "",
    "| 分类 | 数量 | 占比 |",
    "|---|---:|---:|",
]
for category in categories:
    count = category_count[category]
    lines.append(f"| {category} | {count} | {count / len(records) * 100:.1f}% |")
lines.append(f"| **合计** | **{len(records)}** | **100.0%** |")
lines += ["", "## 必要性统计", "", "| 必要性 | 数量 | 解释 |", "|---|---:|---|"]
for label, explanation in [
    ("场景必需", "在对应多节点、Ray、DP、Mooncake/RFork/Netloader 等场景中缺失会导致启动链路不完整。"),
    ("场景配置", "用于选择设备、模型来源、外部库路径或特定部署资源；是否需要取决于环境。"),
    ("可选调优/调试", "通常存在默认行为，主要改变性能、超时、内存、日志或特性开关。"),
    ("不应使用", "文档仍出现，但当前代码没有注册或消费；设置后不会启用文档描述的功能。"),
]:
    lines.append(f"| {label} | {necessity_count[label]} | {explanation} |")
lines += ["", "## 设置方式统计", "", "同一变量可能通过多种方式进入服务环境。", "", "| 设置方式 | 变量数 |", "|---|---:|"]
for mechanism, count in mechanism_count.most_common():
    lines.append(f"| {mechanism} | {count} |")
lines += [
    "", "## 源码与文档不一致项", "",
    "以下变量仅能证明“文档写过”，不能证明当前 vLLM Ascend 支持。判定依据是当前 `vllm_ascend/envs.py`、全源码读取点和 Git 历史。",
    "", "| 变量 | 当前状态 | 历史证据与结论 |", "|---|---|---|",
]
for name, detail in STALE_DOC_VARS.items():
    lines.append(f"| `{name}` | **文档遗留，当前源码不支持** | {detail} |")
lines += [
    "", "## 必要性说明", "",
    "- **基础必需**：启动方式或硬件拓扑明确要求；未配置可能无法发现设备或建立通信。",
    "- **场景必需**：仅在多节点、Ray、PD 分离、Mooncake/UCM 等特定部署中必需。",
    "- **可选调优/调试**：不配置通常仍可启动，但会改变性能、内存、日志、profiling 或特性路径。",
    "",
    "本文不机械地把每个 `export` 都标成“必需”。具体必要性应结合对应文档场景和启动命令判断。",
    "",
    "## 分类明细",
    "",
    "`相关场景` 来自变量附近的文档标题；每个变量最多展示 6 个场景和 8 个位置。",
]
for category in categories:
    subset = [record for record in records if record["category"] == category]
    lines += ["", f"### {category}（{len(subset)}）", "", "| 变量 | 必要性 | 设置方式 | 相关场景（示例） | 文档位置（示例） |", "|---|---|---|---|---|"]
    for record in subset:
        scenario = "; ".join(record["scenarios"][:6]).replace("|", "\\|")
        locations = "; ".join(record["locations"][:8])
        lines.append(f"| `{record['name']}` | {record['necessity']} | {', '.join(record['mechanisms'])} | {scenario} | `{locations}` |")

lines += ["", "## 已排除的启动脚本辅助变量", "", "这些名称出现在启动流程附近，但主要用于拼接镜像、模型路径、容器名、请求长度或端口，不是服务进程的配置接口。", "", "| 辅助变量 | 文档位置（示例） |", "|---|---|"]
for name in sorted(excluded):
    lines.append(f"| `{name}` | {'; '.join(sorted(set(excluded[name]))[:6])} |")
lines += [
    "", "## 口径与限制", "",
    "1. 以文档中的实际启动链路为准，不扩展收录外部 vLLM/CANN 文档中可能支持但本仓库文档未使用的变量。",
    "2. `export` 会影响后续子进程，因此即使它与 `vllm serve` 分处相邻代码块，只要属于同一部署流程也纳入。",
    "3. Docker `-e/--env`、Kubernetes/Ray 启动所注入的变量属于服务环境；Docker 镜像构建参数不属于本文范围。",
    "4. `curl` 通常不需要环境变量；只有推理客户端自身明确读取或继承的变量才纳入。",
    "5. 文档示例可能包含可选性能参数。是否必须设置应以对应硬件、模型和部署拓扑为准。",
]

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(records)} variables, {len(launch_docs)} docs)")
