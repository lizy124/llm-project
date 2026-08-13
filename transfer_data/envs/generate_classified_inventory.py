import pathlib
import re
from collections import Counter

BASE = pathlib.Path(__file__).parent
SOURCE = BASE / "vllm-ascend-environment-variables.md"
OUTPUT = BASE / "vllm-ascend-environment-variables-classified.md"


def read_inventory():
    text = SOURCE.read_text(encoding="utf-8")
    revision = re.search(r"仓库版本：`([^`]+)`", text).group(1)
    table = text.split("## 全量索引", 1)[1].split("## 解释与使用建议", 1)[0]
    records = []
    for line in table.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        records.append(
            {
                "name": cells[0].strip("`"),
                "old_category": cells[1],
                "areas": [item.strip() for item in cells[2].split(",")],
                "mechanisms": [item.strip() for item in cells[3].split(",")],
                "locations": cells[4],
            }
        )
    return revision, records


CATEGORIES = [
    ("01", "vLLM Ascend 产品配置", "仓库自身的功能开关、优化策略和兼容行为。部署用户应优先阅读本节。"),
    ("02", "Ascend、CANN 与通信运行时", "由 Ascend/CANN/HCCL/LCCL 等运行时解释的设备、通信、日志和调试配置。"),
    ("03", "分布式启动与进程拓扑", "由 torchrun、Ray、vLLM DP 或集群启动器注入的 rank、world size、地址和端口。"),
    ("04", "KV Transfer、存储与解耦后端", "Mooncake、MemFabric、Memcache、YuanRong、LMCache、RFork 等后端配置。"),
    ("05", "上游 vLLM、PyTorch 与模型生态", "由上游 vLLM、PyTorch、Hugging Face、ModelScope 或模型工具解释的配置。"),
    ("06", "构建、编译、安装与 Docker 构建", "wheel、CMake、自定义算子、编译器、依赖和 Docker build 参数。Docker ARG 会明确标注。"),
    ("07", "系统与通用运行环境", "操作系统、动态链接器、Python、OpenMP 和代理等通用进程环境。"),
    ("08", "测试、基准与模型用例", "只用于 UT/E2E、nightly、性能基准或特定模型用例的输入。"),
    ("09", "CI、发布、凭据与仓库自动化", "GitHub Actions、镜像/制品发布、缓存、翻译、PR bot 和流水线传值。"),
    ("10", "文档与示例命令辅助变量", "文档为了复用命令而声明的占位路径、镜像、设备、端口等，不是产品配置接口。"),
    ("11", "仓库工具与其他辅助变量", "独立工具、诊断脚本或无法归入以上领域的仓库内部环境输入。"),
]

PRODUCT_EXACT = {
    "DYNAMIC_EPLB", "EXPERT_MAP_RECORD", "MSMONITOR_USE_DAEMON",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM", "TRITON_ALL_BLOCKS_PARALLEL",
}
DIST_EXACT = {
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
    "MASTER_PORT", "LWS_LEADER_ADDRESS", "LWS_WORKER_INDEX", "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME", "VLLM_HOST_IP", "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
}
KV_EXACT = {
    "ASCEND_CONNECT_TIMEOUT", "ASCEND_ENABLE_USE_FABRIC_MEM", "ASCEND_GLOBAL_RESOURCE_CONFIG",
    "ASCEND_TRANSFER_TIMEOUT", "ASCEND_TOTAL_MEMORY_GB", "DATASYSTEM_CLIENT_LOG_DIR",
    "NETLOADER_CONFIG", "OPENLIBING_SECRET", "YR_CONFIG_PATH", "RFORK_CONFIG",
}
BUILD_EXACT = {
    "MAX_JOBS", "CMAKE_BUILD_TYPE", "COMPILE_CUSTOM_KERNELS", "CXX_COMPILER", "C_COMPILER",
    "SOC_VERSION", "VERBOSE", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH",
    "ASCEND_TOOLKIT_HOME", "ASCENDC_CMAKE_DIR", "CC", "CXX", "FETCHCONTENT_BASE_DIR",
}
SYSTEM_EXACT = {
    "PATH", "HOME", "SYSTEMROOT", "CONDA_EXE", "PYTHONPATH", "PYTHONHASHSEED",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "OMP_NUM_THREADS", "OMP_PROC_BIND", "HTTP_PROXY",
    "HTTPS_PROXY", "NO_PROXY", "TERM", "FORCE_COLOR", "TOKENIZERS_PARALLELISM",
}
DOC_EXACT = {
    "DEVICE", "IMAGE", "MODEL", "MODEL_PATH", "NAME", "SAVE_PATH", "SERVER_PORT",
    "WORKSPACE", "IP_ADDRESS", "MASTER_IP", "MASTER_IP_ADDRESS", "NETWORK_CARD_NAME",
    "PHYSICAL_DEVICES", "ENDPOINT", "DATASET_SOURCE", "SERVICE_PROF_CONFIG_PATH",
    "PROFILING_SYMBOLS_PATH", "MM_IMAGE_PATH",
}
TEST_PREFIXES = (
    "VLLM_TEST_", "MINIMAX_", "QWEN3_MRV2_", "WEIGHT_TRANSFER_TEST_", "BISECT_",
)
CI_PREFIXES = (
    "GITHUB_", "GH_", "AWS_", "OBS_", "QUAY_", "HITEST_", "HW_TOKEN", "HW_USERNAME",
    "AUROGON_", "RUNS_ON_", "UV_", "PRE_COMMIT_",
)
KV_PREFIXES = ("MOONCAKE_", "MMC_", "MEMFABRIC_", "YR_", "RFORK_")
UPSTREAM_PREFIXES = ("VLLM_", "PYTORCH_", "TORCH_", "HF_", "MODELSCOPE_", "TRITON_")


def classify(record):
    name = record["name"]
    areas = set(record["areas"])
    mechanisms = set(record["mechanisms"])
    # Build arguments and CI metadata must be classified before the VLLM_ASCEND_
    # prefix rule; e.g. VLLM_ASCEND_BRANCH is not a runtime product switch.
    if "Docker ARG" in mechanisms or "Docker ENV" in mechanisms:
        return "06"
    if name in {"VLLM_ASCEND_BRANCH"}:
        return "06"
    if name in {"VLLM_ASCEND_COMMIT", "VLLM_ASCEND_VERSION"}:
        return "09" if "CI/发布" in areas else "08"
    if name in {"VLLM_ASCEND_REF"}:
        return "08" if "测试" in areas else "11"
    if name.startswith("VLLM_ASCEND_") or name in PRODUCT_EXACT:
        return "01"
    if name in DIST_EXACT or name.startswith("VLLM_DP_"):
        return "03"
    if name in KV_EXACT or name.startswith(KV_PREFIXES):
        return "04"
    if name.startswith(("HCCL_", "LCCL_", "ASCEND_", "ACL_", "TASK_QUEUE_", "CPU_AFFINITY_")):
        if name in BUILD_EXACT:
            return "06"
        return "02"
    if name.startswith(UPSTREAM_PREFIXES):
        if name.startswith(TEST_PREFIXES):
            return "08"
        return "05"
    if name in BUILD_EXACT or "Docker ARG" in mechanisms or "Docker ENV" in mechanisms:
        return "06"
    if name in SYSTEM_EXACT:
        return "07"
    if name.startswith(TEST_PREFIXES) or (areas == {"测试"}):
        return "08"
    if name.startswith(CI_PREFIXES) or (areas == {"CI/发布"}):
        return "09"
    if name in DOC_EXACT or areas == {"文档/示例"}:
        return "10"
    if "构建/打包" in areas and not ({"运行时代码", "测试", "CI/发布"} & areas):
        return "06"
    if "测试" in areas and not ({"运行时代码", "文档/示例"} & areas):
        return "08"
    if "CI/发布" in areas and "运行时代码" not in areas:
        return "09"
    return "11"


def pct(count, total):
    return f"{count / total * 100:.1f}%"


revision, records = read_inventory()
for record in records:
    record["category"] = classify(record)

total = len(records)
category_count = Counter(record["category"] for record in records)
area_count = Counter(area for record in records for area in record["areas"])
mechanism_count = Counter(mechanism for record in records for mechanism in record["mechanisms"])
direct_code = sum("Python/C/C++ API" in record["mechanisms"] for record in records)
central_count = 19
sensitive = sum(
    any(token in record["name"] for token in ("TOKEN", "SECRET", "ACCESS_KEY", "API_KEY", "PASSWORD", "APPCODE"))
    for record in records
)

lines = [
    "# vLLM Ascend 环境变量分类视图",
    "",
    f"- 仓库版本：`{revision}`",
    f"- 唯一名称总数：**{total}**",
    f"- 产品中央配置：**{central_count}**",
    f"- 被 Python/C/C++ 直接访问：**{direct_code}**",
    f"- 名称疑似敏感（Token/Key/Secret/Password）：**{sensitive}**",
    "",
    "> 统计对象是“仓库中具有环境变量语义的唯一名称”，并不表示这些名称全部是 vLLM Ascend 的公开配置。总数包含运行时环境变量、CI/Kubernetes `env`、测试输入，以及单列标注的 Docker `ARG`。",
    "",
    "## 如何阅读",
    "",
    "- 部署和调优：优先看第 1 至第 5 类。",
    "- 编译和制作镜像：看第 6 类。",
    "- 排查宿主机或动态库问题：看第 7 类。",
    "- 开发者和仓库维护者：看第 8、9、11 类。",
    "- 第 10 类通常只是示例中的临时占位变量，不应当当作产品接口。",
    "",
    "每个变量只分配一个“主分类”，因此分类数量可以直接相加得到总数；范围和机制统计允许一个变量重复计入。",
    "",
    "## 主分类统计",
    "",
    "| 主分类 | 数量 | 占比 | 定位 |",
    "|---|---:|---:|---|",
]
for code, title, description in CATEGORIES:
    lines.append(f"| {code}. {title} | {category_count[code]} | {pct(category_count[code], total)} | {description} |")
lines.append(f"| **合计** | **{sum(category_count.values())}** | **100.0%** | 互斥分类 |")

lines += [
    "",
    "## 出现范围统计",
    "",
    "同一变量可在多个范围出现，因此本表合计会大于唯一名称总数。",
    "",
    "| 出现范围 | 唯一变量数 | 占总数比例 |",
    "|---|---:|---:|",
]
for area, count in sorted(area_count.items(), key=lambda item: (-item[1], item[0])):
    lines.append(f"| {area} | {count} | {pct(count, total)} |")

lines += [
    "",
    "## 使用机制统计",
    "",
    "机制之间存在交集。例如一个变量既可能由文档 `export`，也可能被 Python 直接读取。Docker `ARG` 是构建参数，不等于容器运行时环境变量。",
    "",
    "| 使用机制 | 唯一变量数 | 占总数比例 |",
    "|---|---:|---:|",
]
for mechanism, count in sorted(mechanism_count.items(), key=lambda item: (-item[1], item[0])):
    lines.append(f"| {mechanism} | {count} | {pct(count, total)} |")

lines += [
    "",
    "## 分类明细",
    "",
    "`范围` 表示变量在仓库中出现的区域；`机制` 表示仓库如何把它作为环境变量使用。代表位置均为仓库相对路径。",
]
for code, title, description in CATEGORIES:
    subset = sorted((record for record in records if record["category"] == code), key=lambda item: item["name"])
    lines += [
        "",
        f"### {code}. {title}（{len(subset)}）",
        "",
        description,
        "",
        "| 变量 | 范围 | 机制 | 代表位置 |",
        "|---|---|---|---|",
    ]
    for record in subset:
        first_location = record["locations"].split(";")[0].strip()
        lines.append(
            f"| `{record['name']}` | {', '.join(record['areas'])} | "
            f"{', '.join(record['mechanisms'])} | `{first_location}` |"
        )

lines += [
    "",
    "## 口径说明",
    "",
    "1. 主分类按变量的主要消费者和生命周期确定，而不是只按名称前缀。`WORLD_SIZE`、`RANK` 等归入分布式启动与进程拓扑。",
    "2. `VLLM_ASCEND_*` 归入产品配置；但 `MAX_JOBS`、`CMAKE_BUILD_TYPE` 等即使登记在 `vllm_ascend/envs.py`，仍按用途归入构建类。",
    "3. `ASCEND_*`、`HCCL_*` 通常由 CANN/HCCL 解释；仓库可能读取、设置或仅在部署文档中推荐。",
    "4. 只在 GitHub Actions `env` 中出现的名称归入 CI/发布类，不视为用户运行 vLLM Ascend 时需要配置的变量。",
    "5. Docker `ARG` 仅在镜像构建期存在；同时转成 `ENV`、被导出或被代码读取时，才兼具环境变量语义。",
    "6. 完整的多位置证据和中央 19 项默认值说明见 `vllm-ascend-environment-variables.md`。",
]

OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUTPUT} ({total} variables)")
