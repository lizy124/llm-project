import pathlib
import re
import subprocess
from collections import defaultdict

ROOT = pathlib.Path(r"D:\lzy\project\kv_pool\code\vllm-ascend")
OUT = pathlib.Path(r"D:\lzy\project\kv_pool\llm-project\transfer_data\envs\vllm-ascend-environment-variables.md")
paths = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8").splitlines()
patterns = [
    ("Python/C/C++ API", re.compile(r'''(?:os\.(?:getenv|environ\.(?:get|setdefault|pop))\s*\(\s*|os\.environ\s*\[\s*|monkeypatch\.(?:setenv|delenv)\s*\(\s*|(?:std::)?getenv\s*\(\s*)["']([A-Za-z_][A-Za-z0-9_]*)["']''')),
    ("Shell export", re.compile(r"(?m)(?:^|[;&|]\s*)export\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("Docker -e/--env", re.compile(r"(?:--env(?:=|\\s+)|(?:^|\\s)-e\\s+)([A-Z][A-Z0-9_]*)")),
]
refs = defaultdict(lambda: defaultdict(list))
areas = defaultdict(set)
for name in paths:
    p = ROOT / name
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    area = "运行时代码" if name.startswith("vllm_ascend/") else "构建/打包" if name.startswith("csrc/") or name in {"setup.py", "CMakeLists.txt"} else "CI/发布" if name.startswith(".github/") else "测试" if name.startswith("tests/") else "文档/示例" if name.startswith("docs/") or name.startswith("examples/") else "工具/辅助"
    for mechanism, rx in patterns:
        for match in rx.finditer(text):
            var = match.group(1)
            line = text.count("\n", 0, match.start()) + 1
            refs[var][mechanism].append(f"{name}:{line}")
            areas[var].add(area)
    if pathlib.PurePosixPath(name).name.startswith("Dockerfile"):
        for match in re.finditer(r"(?mi)^\s*(ENV|ARG)\s+([A-Za-z_][A-Za-z0-9_]*)", text):
            var = match.group(2)
            line = text.count("\n", 0, match.start()) + 1
            refs[var][f"Docker {match.group(1).upper()}"] .append(f"{name}:{line}")
            areas[var].add("构建/打包")
    if pathlib.PurePosixPath(name).suffix.lower() in {".yml", ".yaml"}:
        env_indents = []
        in_k8s_env = []
        for line_no, line_text in enumerate(text.splitlines(), 1):
            if not line_text.strip() or line_text.lstrip().startswith("#"):
                continue
            indent = len(line_text) - len(line_text.lstrip())
            while env_indents and indent <= env_indents[-1]:
                env_indents.pop()
            if re.match(r"^\s*env:\s*$", line_text):
                env_indents.append(indent)
                in_k8s_env.append(indent)
                continue
            if env_indents:
                match = re.match(r"^\s*([A-Z][A-Z0-9_]*):", line_text)
                if match:
                    var = match.group(1)
                    refs[var]["YAML env"].append(f"{name}:{line_no}")
                    areas[var].add(area)
                match = re.match(r"^\s*-?\s*name:\s*([A-Z][A-Z0-9_]*)\s*$", line_text)
                if match:
                    var = match.group(1)
                    refs[var]["Kubernetes env"].append(f"{name}:{line_no}")
                    areas[var].add(area)

def category(var):
    if var.startswith("VLLM_ASCEND_"): return "vLLM Ascend 专用"
    if var.startswith(("ASCEND_", "HCCL_", "LCCL_", "MOONCAKE_", "MMC_", "YR_", "MEMFABRIC_")): return "Ascend/通信/KV 组件"
    if var.startswith(("VLLM_", "TORCH_", "PYTORCH_", "TRITON_")): return "vLLM/PyTorch 上游"
    if var in {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "HOME", "CC", "CXX", "OMP_NUM_THREADS", "OMP_PROC_BIND", "LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE"}: return "系统/分布式通用"
    if var.startswith(("GITHUB_", "GH_", "AWS_", "OBS_", "QUAY_", "UV_", "PIP_")): return "CI/凭据/包管理"
    if var.startswith(("MINIMAX_", "QWEN3_", "RFORK_", "VLLM_TEST_", "WEIGHT_TRANSFER_TEST")): return "测试/模型专用"
    return "仓库工具/文档专用"

central = [
    ("MAX_JOBS", "未设置（使用全部 CPU 核）", "构建 wheel 时的最大并行编译线程数。"),
    ("CMAKE_BUILD_TYPE", "Release", "CMake 构建类型：Release、Debug 或 RelWithDebugInfo。"),
    ("COMPILE_CUSTOM_KERNELS", "1/True", "是否编译自定义算子；无 NPU 的 UT 环境才建议关闭。"),
    ("CXX_COMPILER", "未设置（系统默认）", "C++ 编译器路径/命令。"),
    ("C_COMPILER", "未设置（系统默认）", "C 编译器路径/命令。"),
    ("SOC_VERSION", "未设置（通过 npu-smi 探测）", "构建目标 Ascend SoC 型号。"),
    ("VERBOSE", "0/False", "是否输出详细编译日志。"),
    ("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest（调用方回退）", "CANN toolkit 根目录。"),
    ("HCCL_SO_PATH", "libhccl.so（调用方回退）", "pyHCCL 通信后端加载的 HCCL 动态库。"),
    ("VLLM_VERSION", "未设置", "源码安装/开发场景覆盖兼容检查所用的 vLLM 版本。"),
    ("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0/False", "启用 FlashComm1 张量并行通信优化；已废弃，改用 additional_config。"),
    ("MSMONITOR_USE_DAEMON", "0/False", "启用 msMonitor daemon 性能监控。"),
    ("VLLM_ASCEND_ENABLE_MLAPO", "1/True", "启用 DeepSeek W8A8 的 MLAPO 优化，会额外占用 NPU 内存。"),
    ("VLLM_ASCEND_ENABLE_NZ", "1", "权重 FRACTAL_NZ 转换策略：0 关闭、1 仅量化、2 尽可能启用。"),
    ("DYNAMIC_EPLB", "false", "动态专家并行负载均衡开关（按小写字符串读取）。"),
    ("VLLM_ASCEND_ENABLE_FUSED_MC2", "0", "允许使用 dispatch_ffn_combine/mega_moe 融合 MC2 路径。"),
    ("VLLM_ASCEND_BALANCE_SCHEDULING", "0/False", "均衡调度开关；已废弃，改用 additional_config。"),
    ("VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK", "1/True", "启用 transpose_kv_cache_by_block 融合算子。"),
    ("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", "未设置（自动探测）", "KV cache offload 的 aclrtMemcpyBatchAsync：1 强制开、0 强制关。"),
]
lines = ["# vLLM Ascend 环境变量清单", "", "- 仓库版本：`" + subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip() + "`", "- 扫描范围：Git 跟踪文件（Python/C/C++、Shell、CMake、Dockerfile、GitHub Actions、测试、示例、文档）。", "- 纳入规则：明确作为环境变量读取、写入、删除、`export`、Docker `ARG/ENV`、`-e/--env`、CI/Kubernetes `env` 的名称。CMake 函数变量和 Shell 普通局部变量不纳入。", "- `ARG` 是镜像构建参数，只有通过 `ENV` 或命令传递后才进入运行时环境；文档示例和测试变量不代表生产默认配置。", "", "## 统计", "", f"去重后共 **{len(refs)}** 个变量。范围列说明变量在仓库的实际出现区域，一个变量可同时出现在多个区域。", "", "## 中央运行时配置", "", "以下 19 个变量在 `vllm_ascend/envs.py` 的 `env_variables` 中集中定义并惰性读取：", "", "| 变量 | 默认值/解析 | 作用 |", "|---|---|---|"]
for var, default, purpose in central:
    lines.append(f"| `{var}` | {default} | {purpose} |")
lines += ["", "## 全量索引", "", "`位置` 使用 `仓库相对路径:行号`；为控制篇幅，每种发现机制最多列出 4 个代表位置。", "", "| 变量 | 类别 | 范围 | 发现机制 | 位置（示例） |", "|---|---|---|---|---|"]
for var in sorted(refs):
    mechanisms = sorted(refs[var])
    locations = []
    for mechanism in mechanisms:
        locations.extend(refs[var][mechanism][:4])
    lines.append(f"| `{var}` | {category(var)} | {', '.join(sorted(areas[var]))} | {', '.join(mechanisms)} | {'; '.join(locations)} |")
lines += ["", "## 解释与使用建议", "", "1. 产品运行时优先查看 `vllm_ascend/envs.py` 及 `vllm_ascend/` 中的直接读取点；直接读取但未集中登记的变量属于历史兼容或外部组件接口。", "2. `ASCEND_*`、`HCCL_*`、`MOONCAKE_*` 等多数由 CANN/HCCL/Mooncake 等外部组件解释，仓库通常只读取或透传。", "3. CI、发布、凭据和测试变量只在相应自动化流程中生效；其中 `*_TOKEN`、`*_KEY`、`*_SECRET` 可能包含敏感信息，文档不记录实际值。", "4. 变量生效时机取决于读取点：安装/编译期、Python import/初始化期、worker 启动期或请求运行期；修改环境后通常需要重新启动进程。", "5. 本清单不把上游 vLLM、PyTorch、CANN 的全部可用变量扩展纳入，只记录本仓库明确出现或传递的变量。"]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(refs)} variables)")
