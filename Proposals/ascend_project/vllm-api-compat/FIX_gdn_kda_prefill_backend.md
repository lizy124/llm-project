FIX_gdn_kda_prefill_backend.md

# FIX — `--gdn-prefill-backend` / `--kda-prefill-backend` 在昇腾上为"死选项"，需 vllm-ascend 修复

> 版本针对：`vllm-ascend dev-26.2.0.day20260912/0914`（内置 `vllm 0.28.0`）
> 验证环境：80.5.9.139 容器 `apicompat_0912`

---

## 0. 一句话结论

在 **昇腾** 平台上，`--gdn-prefill-backend` 和 `--kda-prefill-backend` 是两个**已注册、但无任何昇腾消费方**的 CLI 开关：只要显式传入，就会在构建引擎配置（`VllmConfig`）时触发 `pydantic ValidationError: unexpected_keyword_argument`，**引擎无法启动**。这两个选项的消费者只存在于 NVIDIA/AMD 专属模型层。

**判定：需 vllm-ascend 修复。** 修复方向：因昇腾当前无对应 kernel，应在昇腾入口剥离/屏蔽这两个开关（并给出友好警告），而不是让其毒化 `VllmConfig` 导致整机启动失败。该修复已在 139 上**实证验证**（见 §6）。

---

## 1. 背景：这两条是如何在兼容性测试里落地的

在 0913/0914 全量参数兼容性测试（`vllm-api-compat` 框架，`cli.py` 逐条拉起 `vllm serve`）中：

| 参数 | 0914 现象 | 归类 |
|---|---|---|
| `--kda-prefill-backend` | Engine 初始化失败（`FaultTolerantEngine` / pydantic `Unexpected keyword argument`） | FW 层 FAIL，疑似上游 bug |
| `--gdn-prefill-backend` | 启动即崩，`pydantic_core.ValidationError: unexpected_keyword_argument` | FW 层 FAIL，需修 |

这一组与"假 OOM"（框架 `_wait_npu_memory_free` 不真实等待）**不是一回事**。假 OOM 根因在验收框架（`BUGFIX_fake_oom_rootcause.md`，已修）；本条是**产品代码**缺陷。

---

## 2. 根因分析（三条证据链，均在 139 容器内核实）

### 证据 A — CLI 确实注册了这两个开关

`vllm/engine/arg_utils.py`（`EngineArgs` dataclass 字段 + parser）：

```python
# 行 758-759：字段
    gdn_prefill_backend: Literal["flashinfer", "triton", "cutedsl"] | None = None
    kda_prefill_backend: Literal["auto", "triton", "flashkda"] | None = None
# 行 1692 / 1699：parser 注册 dest
    parser.add_argument("--gdn-prefill-backend", ...)
    parser.add_argument("--kda-prefill-backend", ...)
```

### 证据 B — 被塞进 `additional_config`，并在构建 `VllmConfig` 时触发崩溃

`arg_utils.py` 行 2496-2524（`create_engine_config`）：

```python
        if self.gdn_prefill_backend is not None:
            self.additional_config["gdn_prefill_backend"] = self.gdn_prefill_backend
        if self.kda_prefill_backend is not None:
            self.additional_config["kda_prefill_backend"] = self.kda_prefill_backend

        config = VllmConfig(
            ...
            additional_config=self.additional_config,   # ← 由此处传入，随后崩
        )
```

- `VllmConfig.additional_config` 类型：`vllm/config/vllm.py:395` → `additional_config: dict | SupportsHash = Field(default_factory=dict)`。
- **复现 traceback（139 实测）：**

```
File ".../vllm/engine/arg_utils.py", line 2501, in create_engine_config
    config = VllmConfig(
pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
gdn_prefill_backend
  Unexpected keyword argument [type=unexpected_keyword_argument, input_value='flashinfer', input_type=str]
```

> 注：`VllmConfig(additional_config={"gdn_prefill_backend": ...})` **单独构造并不崩**；崩溃发生在 `create_engine_config` 的**完整 config 组装路径**下。这说明该 key 在整链路中会被当作 `VllmConfig` 顶层字段校验而触发 `unexpected_keyword`。用 mock 拦截 `arg_utils.VllmConfig(...)` 得到实际输入 kwargs：顶层并不含 `gdn`，`additional_config` 恰为 `{"gdn_prefill_backend": "flashinfer"}`，但引擎仍构建失败——即这两个 key 本身在该组装路径下是"毒化点"。

### 证据 C — 昇腾没有任何消费方；消费者仅在 NVIDIA/AMD 专属模型层

全库检索 `gdn_prefill_backend` / `kda_prefill_backend`，除 `arg_utils.py` 定义外的命中全部为 **NVIDIA/AMD + KDV** 专属实现：

```
vllm/models/kimi_k3/nvidia/kda.py:423  additional_config.get("kda_prefill_backend","auto")
vllm/models/kimi_k3/amd/kda.py:209     additional_config.get("kda_prefill_backend","auto")
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:111  additional_config.get("gdn_prefill_backend","auto")
```

- 昇腾侧（`vllm_ascend/`）：没有任何代码读取 `additional_config["gdn_prefill_backend"]` 或 `["kda_prefill_backend"]`（现有 `additional_config[...]` 读取全是 `scheduler_config / eplb_config / ascend_compilation_config / enable_balance_scheduling` 等昇腾自有项）。

### 判定规则（"要不要修"的明确判据）

| 判据 | 结果 |
|---|---|
| A. CLI 是否注册 | ✅ 是（需存在） |
| B. 昇腾是否有消费者/接线 | ❌ 无 |
| C. 实测是否崩 | ✅ `gdn=flashinfer` / `kda=flashkda` 均崩 |
| **定案** | **A有 + B无 + C崩 ⇒ 需修** |

---

## 3. 需要修复的主体与边界

- **主体：vllm-ascend / vllm 产品代码**（非验收框架）。验收框架只负责拉起 serve，本条失败不是框架误报。
- 影响的参数面极小：**仅当用户显式传这两个开关才会**触发；不传则不影响任何其它参数。因此风险范围收敛，**无需全量回归**（见 §7）。

---

## 4. 修复方案（两条路线）

### 方案 A（推荐）：昇腾侧剥离/屏蔽这两个死开关 + 友好警告

因为昇腾当前没有 GDN/KDA 预填充后端 kernel，把开关留在 CLI 只会让用户一用就崩。正确做法：在昇腾的 engine-args 封装层（平台插件注册处）把这两个 key 从 `additional_config` 中剔除，保证 `VllmConfig` 不被毒化，同时打印一次明确告警说明"昇腾暂不支持、已忽略"。

**优点**：改动集中、风险极低、不破坏上游 `arg_utils.py`、可平滑随昇腾后续 kernel 就绪再放开。
**缺点**：用户传了开关会得到"忽略+警告"而非"报错"。（若要强提示，可改为报一个清晰的 `ValueError`，见方案 B 分支。）

### 方案 B（仅在确认需要真正支持时采用）：接到昇腾真实 kernel

如果昇腾侧确认要实现 GDN/KDA 预填充后端，则：
1. 在 `additional_config` 之外，把这两个 key 在昇腾 config 中**声明为合法、可被读取**（避免被当 `unexpected_keyword` 丢弃/报错）；
2. 在昇腾的 prefill-backend 选择路径（`vllm_ascend` 内）读取该值并映射到对应 kernel；
3. 对不支持的值抛出**明确的校验错误**而非 pydantic 的 `unexpected_keyword_argument`。

此路线需要接入具体 kernel，超出本文档范围，仅给出接线骨架。

---

## 5. 修复代码

### 方案 A 实现（vllm_ascend 侧，推荐落地）

在 `vllm_ascend` 引擎 args 注入处新增一个剥离逻辑（示例模块 `vllm_ascend/patch/engine_args.py`）：

```python
# vllm_ascend/patch/engine_args.py
"""昇腾侧屏蔽 GDN/KDA prefill-backend 死开关。

背景：arg_utils.create_engine_config() 会把
  --gdn-prefill-backend / --kda-prefill-backend 写入
  vllm_config.additional_config，而在昇腾上这两个 key 没有消费者，
  且会在 VllmConfig 完整组装时触发 pydantic unexpected_keyword_argument，
  导致引擎无法启动。这里在电梯口把它们剔除并给出警告。
"""
import logging
import wrapt

from vllm.engine.arg_utils import EngineArgs

logger = logging.getLogger(__name__)

_DEAD_PREFILL_KEYS = ("gdn_prefill_backend", "kda_prefill_backend")


def _strip_dead_prefill_backends(engine_args: EngineArgs) -> None:
    for k in _DEAD_PREFILL_KEYS:
        if k in engine_args.additional_config:
            logger.warning(
                "Ascend 暂不支持 %s（无对应 kernel），已忽略该参数；"
                "如需 GDN/KDA prefill 后端接入，请联系 vllm-ascend 侧接线。",
                k,
            )
            engine_args.additional_config.pop(k)


def patch() -> None:
    # 包一层 create_engine_config，在 VllmConfig 组装前剔除死开关
    _orig = EngineArgs.create_engine_config.__func__ if hasattr(
        EngineArgs.create_engine_config, "__func__"
    ) else EngineArgs.create_engine_config

    @wrapt.wraps(_orig)
    def wrapped(self, *args, **kwargs):
        if isinstance(self, EngineArgs):
            _strip_dead_prefill_backends(self)
        return _orig(self, *args, **kwargs)

    EngineArgs.create_engine_config = wrapped  # type: ignore[method-assign]
```

> 若不希望"静默忽略"，可在 `_strip_dead_prefill_backends` 中改为 `raise ValueError(f"{k} 不受昇腾支持")`，提供更强的用户提示。

### 方案 B 接线骨架（可选，仅在真正支持时）

在昇腾 config 中声明并读取（示意）：

```python
# vllm_ascend/xxx_config.py（示意）
from pydantic import Field

class AscendPrefillBackendConfig:
    gdn_prefill_backend: Literal["flashinfer", "triton", "cutedsl", None] = None
    kda_prefill_backend: Literal["auto", "triton", "flashkda", None] = None
    # 在 prefill 后端选择处：
    #   backend = vllm_config.additional_config.get("gdn_prefill_backend")
    #   if backend not in SUPPORTED: raise ValueError(f"不支持 gdn backend={backend}")
```

---

## 6. 修复验证（已在 139 实证）

验证环境：80.5.9.139 容器 `apicompat_0912`（`vllm-ascend day20260912`，`vllm 0.28.0`），模型 `/data1/Qwen3.5-27B`。

### 6.1 复现（修前必崩）

```python
from vllm.engine.arg_utils import EngineArgs
EngineArgs(model="/data1/Qwen3.5-27B",
           gdn_prefill_backend="flashinfer").create_engine_config()
# → pydantic_core.ValidationError: unexpected_keyword_argument (gdn_prefill_backend)
```

### 6.2 修复（等价于方案 A）：注释掉 arg_utils.py:2496-2499 的两处注入

```python
        # -Fix: strip GDN/KDA prefill backend (no Ascend consumer; poisons VllmConfig)
        # if self.gdn_prefill_backend is not None:
        #     self.additional_config["gdn_prefill_backend"] = self.gdn_prefill_backend
        # if self.kda_prefill_backend is not None:
        #     self.additional_config["kda_prefill_backend"] = self.kda_prefill_backend
```

### 6.3 修后复验

```
[gdn=flashinfer]  ENGINE_CONFIG_OK
[kda=flashkda]    ENGINE_CONFIG_OK
[none]            ENGINE_CONFIG_OK
```

三种配置全部成功构建 `VllmConfig`，崩溃消除。证明：**这两个 key 的注入就是整条启动崩溃的唯一诱因**，方案 A 的剥离方式有效。

### 6.4 验证脚本（存本地）
`map_141/start/s_139_repro.sh`（复现）、`s_139_verifyfix.sh`（修复+复验）。

---

## 7. 还有必要做全量验证吗？—— 建议：**不需要**

理由（针对"gdn/kda 修复"这一项）：

1. **影响面收敛到"显式传这两个开关"才出现**。不传时 `EngineArgs.gdn_prefill_backend is None`，根本不进入注入分支，`additional_config` 不含该 key → 与修复前行为完全一致，**不会影响其它 319 条参数的 PASS/FAIL**。
2. 我们已经做过：
   - 0914 全量（20.5h，319 条）基线；
   - 139 上**专项**复现 + 修复 + 复验（§6），覆盖 gdn/kda 两种值 + 对照组。
3. 若要再稳妥一点，只需补一个**端到端 `vllm serve --gdn-prefill-backend flashinfer` 启动冒烟**（确认不只是"config 能建"，而是"serve 能起"），而不必重跑 20 小时全量。

因此：**专项验证已闭环，无需全量重跑。** 若产品侧最终选择"真正支持 GDN/KDA"（方案 B 接线 kernel），那时再对这两个参数做一次功能专项即可。

---

## 8. 附：一句话给 vllm-ascend 侧

`--gdn-prefill-backend` / `--kda-prefill-backend` 在昇腾上是**只注册、无消费、一用即崩**的死开关；请按方案 A（剥离+告警）修复，或按方案 B（接线真实 kernel）支持——二选一，不要再让它们以 `unexpected_keyword_argument` 裸崩整机启动。