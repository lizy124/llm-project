skill_feedback_20260907.md
# vllm-api-compat skill 优化建议（给 workspace 框架仓库）

> 来源：map_135 全量参数兼容测试实测（2026-09-07）
> 框架仓库：github.com/underfituu/vllm-ascend-workspace（本地镜像 D:\project\code\vllm-ascend-workspace）
> 本文件供对外提交（PR / 提给仓库）使用

---

## 背景

在 800I-A3（map_135，容器 pr15367_135）对 vllm-api-compat skill 执行 200 用例全量参数兼容测试，最终口径 **178 PASS / 22 FAIL（零 Ascend 特有回归）**。测试过程中框架存在 4 处效率/判定问题，累计造成约 **2~3 小时纯浪费等待**，且产生 **12 个无效 fix stub（全部误报）**。以下建议基于实测证据，供框架侧改进。

## 问题总览

| # | 问题 | 影响 | 建议方案 | 预计收益 |
|---|---|---|---|---|
| 1 | `wait_for_ready` 无进程早退，启动即崩用例干等满 timeout | LoRA 8 例纯烧 ~53 分钟（400s×8）；grpc/headless 卡满 900s | 传 proc + `poll()` 早退 | 崩用例判 FAIL 从 400s→<5s |
| 2 | baseline 依赖资产无 pre-flight 检查，缺失仍启动照跑 | LoRA 8 例（adapter 缺失）、MM 7 例（模型不匹配）白白烧时间 | 执行前静态检查 yaml 引用的路径，缺失 SKIP | 15 例从"干等 400s"→"0s 判定" |
| 3 | `is_ascend_error` 子串匹配过宽，CANN 兜底打印导致误报 | 12 个 fix stub 全部无效（实际是 pydantic 校验错误） | 排除确定性校验文案；匹配限定 Python 级 traceback | 消除误报，节省人工审查 |
| 4 | `--rerun-failed` 无条件重跑所有 FAIL，不区分根因 | 确定性失败重复烧时间；偶发失败可能被遗漏 | 按根因筛选（deterministic / unknown），默认只重跑 unknown | 重跑池收敛到真实偶发 |

---

## 建议 1：`wait_for_ready` 增加进程早退

**代码位置**：`scripts/_common.py:260` `def wait_for_ready(port, timeout=120, host)` 

**现状**：只轮询 `http://<host>:<port>/health`，不感知 serve 进程状态。vllm 启动即崩（如 pydantic 参数校验失败、ImportError）时端口永远不会 200，于是**干等满 timeout**。

**实测证据**：
- 全量跑 LoRAConfig 8 例（adapter 路径不存在 → vllm 启动即崩），每例干等 400s，合计 ≈53 分钟纯浪费；
- 补跑 grpc 用例（缺 `smg-grpc-servicer` pip 包）卡满 900s 仍不返回，需人工 kill。

**方案**：
```python
def wait_for_ready(port, timeout=120, host="127.0.0.1", proc=None):
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False  # 进程已退出，无需继续等待
        try:
            resp = urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except (URLError, OSError, ConnectionError):
            pass
        time.sleep(2)
    return False
```
备用方案：监听端口状态跳变（"曾绑定 → 已释放"）也可作为退出信号。

---

## 建议 2：baseline 依赖资产 pre-flight 检查（缺失直接 SKIP）

**代码位置**：`scripts/cli.py` 加载 params-yaml 后、启动用例前（约 L408 附近）。

**现状**：yaml 引用的 baseline 模型路径、`--lora-modules` adapter 路径、`--draft-model` 路径不做存在性检查。资产缺失时 vllm 启动失败 → 干等 400s → FAIL，notes 为 `LoRAAdapterNotFoundError` / HTTP 400 等。

**实测证据**：
- LoRAConfig 8 例：yaml 引用 `/root/.cache/modelscope/hub/models/vllm-ascend/qwen35-4b-text-only-sql-lora` 不存在 → 每例干等 400s；
- MultiModalConfig 7 例：baseline 为纯文本模型（Qwen3-0.6B），发 multimodal_image 请求必然 HTTP 400 → 干等 + 假 FAIL。

**方案**：执行前解析 yaml，提取 baseline 路径 + 各参数中的本地文件路径引用，`os.path.exists()` 静态检查：
- 缺失 → 直接写 `SKIP`，notes 记 `asset missing: <path>`，不进启动流程；
- 提供 `--asset-check-only` 预检模式（纯检查不跑用例），方便 CI 前先出资产清单。

---

## 建议 3：`is_ascend_error` 排除确定性校验文案（消除误报）

**代码位置**：`scripts/daily_check.py:38`（`ASCEND_ERROR_PATTERNS`）+ `scripts/daily_check.py:260`（`is_ascend_error`）。

**现状**：Ascend 失败判定 = notes 是否包含子串 `("NotImplementedError", "AscendError", "npu", "torch_npu", "cann")`。但 vllm 进程崩溃时，CANN runtime 会对任意未捕获异常**兜底打印** `[ERROR] ... ERR99999 UNKNOWN applicaiton exception`（该行含 `cann` 相关上下文），被截入 notes 后即命中子串 → 误判 Ascend 失败并生成 fix stub。

**实测证据**：本次 12 个 fix stub **全部无效**。真实根因均为 vLLM 自身 pydantic 校验错误（如 `Unable to use nsight profiling unless workers run with Ray`、`enable_expert_parallel must be True to use EPLB`），只是崩溃时 CANN runtime 的 ERR99999 行恰好被截获。

**方案**（建议组合实施）：
- **a) 排除法优先**：notes 含以下确定性校验文案 → 直接判**非** Ascend：
  `ValidationError`、`pydantic_core`、`must be`、`requires`、`only supported`、`must have`、`Value error`；
- **b) 收紧匹配**：Ascend 判定的正向匹配限定在 Python 级 traceback 帧（`vllm_ascend`、`torch_npu` 的 import/调用栈），而非任意一行含 `cann`；
- **c) 输出分离**：serve 进程 stderr 单独落盘，把 CANN runtime 打印与 vLLM 业务错误分开，从源头避免污染 notes。

---

## 建议 4：`--rerun-failed` 支持按根因筛选（只重跑"可能偶发"）

**代码位置**：`scripts/cli.py:409-434`（`--rerun-failed` 模式）。

**现状**：无条件提取 summary 中所有 `status==FAIL` 的用例重跑。不区分根因：
- 确定性失败（pydantic 校验、资产缺失）重跑 100% 还是 FAIL，纯烧时间；
- 偶发失败（资源竞争）若不重跑则漏转正。

本次无法依赖 `--rerun-failed`，只能人工归因后手动挑"疑似偶发"集合（报告中称为"选择性 rerun"，41 条中 34 条转 PASS）。

**方案**：加 `--rerun-mode {unknown,all}`（默认 `unknown`）：

```text
对每条 FAIL：
  1. 静态路径检查命中（资产缺失 / 缺 pip 包）      → deterministic，跳过
  2. notes 匹配确定性文案库（复用建议 3 的排除模式）
     + 进程早退信号（建议 1 提供）               → deterministic，跳过
  3. 其余（进入过 serving 阶段才挂 / 根因不明）    → unknown，进重跑池
```
- `--rerun-mode all` 保留为强制双次验证（用于扩充确定性文案库）；
- 可选 `--rerun-exclude-notes <regex>` 由用户显式排除。

### 4.1 判定"确定性失败"的规则（本次实测总结）

**确定性失败 = 具备以下任一硬证据，重跑必然同因失败：**

| 层级 | 判据 | 判定方式 | 示例（本次实测） |
|---|---|---|---|
| L1 静态预检 | 资产路径不存在 / 缺包 | `os.path.exists()` 检查 yaml 引用，无需启动 | LoRA adapter 缺失、`ImportError: requires pip install` |
| L2 解析期报错 | config 组装阶段抛 pydantic `ValidationError`，根因不依赖运行时资源，进程 <30s 内崩溃 | 确定性文案库 + 进程早退信号 | `must be True to use EPLB`、`only supported with enable_eplb=True`、`unless workers run with Ray`、draft 缺 `pard_token` |
| L3（非确定性，进重跑池） | vllm **成功进入 serving 后**才失败，根因依赖运行时资源 | 报错含 `Free memory` / `Broken pipe` / 握手超时等 + 进程曾正常存活 | HBM 竞争 14 例（串行重跑全转 PASS）、PROFILER `Broken pipe` 4 例 |

**核心判据**：根因是否发生在**权重加载/服务对外提供之前**、是否**依赖运行时资源**。解析期报错 → 必然；运行期资源类 → 可能偶发，必须重跑。

### 4.2 进阶方案：`--config-dry-run`（参数合法性免启动判定）

vLLM 的 config 校验发生在 `VllmConfig(**kwargs)` 组装阶段，权重加载在其后。框架可加 `--config-dry-run`：构造 config 对象**不加载权重、不绑端口**。
- 参数不合法 → 1~2s 内抛 pydantic 错误（确定性归因，零 serve 启动）；
- 参数合法 → 打印 config 概要后退出。

适用面：pydantic 校验类用例（本次 ParallelConfig 3 + ExceptionError 同名 3 + MOE 2 + DataParallelLB 2 = 10 例）可完全不启动服务即完成归因，最省时且最准确。

---

## 实现优先级建议

1. **P0 建议 3**（消除误报）：代码改动小（一个模式库 + 一行匹配逻辑），直接消除 12 个无效 stub 的误导；
2. **P0 建议 1**（进程早退）：改动小，立即消灭"干等 timeout"这一最大时间黑洞；
3. **P1 建议 4**（重跑筛选）：依赖建议 1/3 的产物（早退信号 + 文案库），顺带打通；
4. **P2 建议 2 + 4.2**（pre-flight / dry-run）：结构性优化，收益大但改动面广，可后续版本。

> 注：建议 1/3/4 可共用同一个"确定性失败文案模式库"（抽成共享函数），避免重复实现。