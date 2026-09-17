# FIX — `--mm-encoder-only` 在昇腾上启动必崩：vllm-ascend 漏移植上游 dummy-run 早退特判

> 版本针对：`vllm-ascend dev-26.2.0.day20260912`（内置 `vllm 0.28.0`）
> 验证环境：80.5.9.139 容器 `apicompat_0912`，模型 `/data1/Qwen3.5-27B`
> 定性：**vllm-ascend 移植缺口（产品代码缺陷），非上游 bug、非验收框架问题**
> 状态：**修复已实测闭环（0916）**——补丁后 encoder-only serve 正常 startup，基线无回归

---

## 0. 一句话结论

`--mm-encoder-only` 模式下，vLLM 只加载多模态编码器、语言模型（LM）权重留在 `meta` 设备；而昇腾版 `model_runner_v1.py` 的编译期 dummy run **仍会执行 LM 前向**，撞上昇腾自定义算子 `npu_gemma_rms_norm` 的 FakeTensor 设备传播校验（`npu:0` 输入 vs `meta` 参数）直接崩溃。**上游 `vllm/v1/worker/gpu_model_runner.py` 早在 3 处函数头部为 encoder-only 写了早退特判，vllm-ascend fork 时一处都没移植**。修复 = 把早退补回来（两个函数、共 +11 行），已在 139 真机验证有效且零回归。

---

## 1. 背景：这条 FAIL 从哪来

### 1.1 全量测试中的落点

0914 全量参数兼容性测试（319 条）中，`MultiModalConfig` 段的 `--mm-encoder-only` 记录为 FAIL：

```json
{
  "section": "MultiModalConfig",
  "param": "--mm-encoder-only",
  "test_value": true,
  "status": "FAIL",
  "notes": "health timeout: 26-09-15-11:07:26 (PID:1000307, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception ..."
}
```

当时报告中它与 `--mm-encoder-attn-dtype=fp8` 一并标注为"引擎启动异常，建议后续单独复核"（其余 FAIL 多为配置组合不合法的预期失败）。0916 在 139 逐条复核定案：

| 参数 | 复现结果 | 定性 |
|---|---|---|
| `--mm-encoder-only` | `TorchRuntimeError`（FakeTensor 设备传播）必崩 | **vllm-ascend 移植缺口，需修**（本文档） |
| `--mm-encoder-attn-dtype=fp8` | 上游 `ValueError`（fp8 仅支持 ROCm AITER / CUDA FlashInfer cuDNN） | 上游通用预期失败，无需修（见 §8 附注） |

对照组 `--no-mm-encoder-only`（默认 False 路径）在 0914 为 PASS——与"仅 encoder-only 分支有问题"的判定自洽。

### 1.2 这条与假 OOM、gdn/kda 的关系

- 与**假 OOM**（0913，`_common.py` 进程/显存生命周期缺陷）无关——本条在假 OOM 修复后的 0914 干净环境里依然稳定 FAIL，是产品代码问题。
- 与 **gdn/kda 死开关**（昇腾 `AscendConfig extra="forbid"` 拒绝注入 key）机制不同——本条崩在更后面的 worker 编译期，不是 config 校验期。

---

## 2. 复现（139 实测，补丁前必崩）

### 2.1 复现命令

```bash
vllm serve /data1/Qwen3.5-27B \
  --mm-encoder-only \
  --tensor-parallel-size 2 --max-model-len 256 --max-num-seqs 64 \
  --gpu-memory-utilization 0.55 --port 8556 --trust-remote-code
```

（容器内 `ASCEND_RT_VISIBLE_DEVICES=10,11`，对应宿主机物理 NPU5；启动前确认目标卡空闲 ≥55G——见 §6.3 环境管理说明。）

### 2.2 崩溃链路（完整 traceback 关键帧）

主进程最终报的是通用包装 `RuntimeError: Engine core initialization failed`，真实根因在 Worker traceback：

```
(Worker_TP0) File ".../vllm_ascend/worker/model_runner_v1.py", line 3958, in _dummy_run
(Worker_TP0)     outputs = self._model_forward(
(Worker_TP0) File ".../vllm_ascend/worker/model_runner_v1.py", line 3055, in _model_forward
(Worker_TP0)     hidden_states = run_model()
(Worker_TP0) File ".../vllm_ascend/compilation/acl_graph.py", line 181, in __call__
(Worker_TP0)     return self.runnable(*args, **kwargs)
(Worker_TP0) File ".../vllm/model_executor/models/qwen3_5.py", line 573, in forward
(Worker_TP0)     hidden_states = self.language_model.model(          # ← LM 前向被执行
(Worker_TP0) File ".../vllm/compilation/decorators.py", line 663, in __call__
(Worker_TP0)     self.aot_compiled_fn = self.aot_compile(*args, **kwargs)
(Worker_TP0)     ... torch FakeTensor dispatch ...
(Worker_TP0) File ".../torch/_subclasses/fake_tensor.py", line 987, in merge_devices
(Worker_TP0)     raise RuntimeError(
(Worker_TP0) torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
    call_function _C_ascend.npu_gemma_rms_norm(
        *(FakeTensor(..., device='npu:0', size=(s59, 5120), dtype=torch.bfloat16),
          Parameter(FakeTensor(..., device='meta', size=(5120,), dtype=torch.bfloat16, requires_grad=True)),
          1e-06), **{}):
    got RuntimeError('Unhandled FakeTensor Device Propagation for
    _C_ascend.npu_gemma_rms_norm.default, found two different devices npu:0, meta')
```

### 2.3 读懂这个报错

- **`npu:0` vs `meta` 的来源**：`--mm-encoder-only` 模式下 vLLM 只把多模态编码器落到 NPU，**LM 的参数不被物化，留在 `meta` 设备**（省显存，这是该模式的语义）。
- **为什么 dummy run 会碰 LM**：dummy run 是 vLLM 编译/AclGraph 捕获前的"空跑预热"，它构造假 batch 走一遍模型前向来触发编译。它跑的是 **LM 执行路径**（`qwen3_5.py:573` 的 `self.language_model.model(...)`），于是 LM 的 `meta` 参数与 `npu:0` 的激活值在同一个算子里相遇。
- **为什么 CUDA 不崩**：不是 CUDA 算子对 meta 参数更宽容，而是**上游干脆不让 encoder-only 模型跑这个 dummy run**（见 §3）。
- **崩溃点算子**：`npu_gemma_rms_norm` 是 torch_npu 的 C++ 算子（`torch_npu/csrc/aten/CustomFunctions.h`），由 `vllm_ascend/ops/layernorm.py:106` 经 `DeviceOperator.npu_gemma_rms_norm`（`vllm_ascend/device/device_op.py:719`）接入 Qwen3.5 的 Gemma-RMSNorm 层。FakeTensor 编译期设备传播逻辑发现一个节点里混了两个设备且无 meta 折叠规则 → 抛错。

---

## 3. 根因分析：上游有 3 处特判，昇腾 fork 移植了 0 处

### 3.1 上游 `vllm/v1/worker/gpu_model_runner.py`（v0.28.0）的三处早退

**① `_dummy_run` 头部（约行 5984）**：

```python
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])
```

**② `_dummy_sampler_run` 头部（约行 6361）**：

```python
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model no need to run sampler.
            return torch.tensor([])
```

**③ `_dummy_pooler_run` 头部（约行 6533）**：

```python
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model not need to run pooler.
            return torch.tensor([])
```

上游注释把意图写得非常直白：**"当前 dummy run 只覆盖 LM 执行（encoder-only 模式下 LM 在 meta 上），所以直接跳过；将来如需 encoder dummy run 再补"**。

### 3.2 昇腾版 `vllm_ascend/worker/model_runner_v1.py` 的现状

逐项核对（139 容器内实测 grep）：

| 上游函数 | vllm_ascend 对应 | 是否移植早退 |
|---|---|---|
| `_dummy_run`（早退①） | `_dummy_run`（行 3648） | ❌ 无 |
| `_dummy_sampler_run`（早退②） | `_dummy_sampler_run`（行 3993） | ❌ 无 |
| `_dummy_pooler_run`（早退③） | **函数不存在**（fork 时删掉了 pooler 路径） | —（无需处理） |
| 全文 `grep mm_encoder_only` | **0 处命中** | — |

（佐证：`vllm_ascend/` 全目录 `grep -rn mm_encoder_only` 也是 0 处；而上游 `vllm/` 侧除定义外有 `gpu_model_runner.py` 3 处 + `config/vllm.py:547`（`is_encoder_only` 类判断）等。）

### 3.3 判定规则

| 判据 | 结果 |
|---|---|
| A. CLI 是否注册 | ✅ 是（`MultiModalConfig.mm_encoder_only`，`vllm/config/multimodal.py:173`，默认 `False`） |
| B. 上游是否支持该模式 | ✅ 是（CUDA 上 encoder-only 可正常 serve，上游有完整特判） |
| C. 昇腾实测是否崩 | ✅ 必崩（§2 traceback） |
| D. 崩因是否昇腾代码缺失 | ✅ 是（fork 漏移植 3 处早退，D 与 C 因果链完整） |
| **定案** | **vllm-ascend 移植缺口，需修（补早退）；上游无 bug、验收框架无责** |

---

## 4. 修复方案

**方案（唯一正确解，且就是对齐上游）**：在 `vllm_ascend/worker/model_runner_v1.py` 的 `_dummy_run` 与 `_dummy_sampler_run`（昇腾版实际存在的两个函数）头部，补上与上游逐字等价的 encoder-only 早退。不改判定链、不碰模型代码、不碰 torch_npu。

**为什么不选其它路**：
- ❌ 给 `npu_gemma_rms_norm` 注册 meta 设备折叠规则：侵入 torch_npu/torch 层，为一个模式特性动全局编译器行为，风险远大于收益；
- ❌ 让 encoder-only 模式把 LM 权重也物化到 NPU：违背该模式"只跑编码器"的语义，白吃几十 GB 显存；
- ❌ 在 `qwen3_5.py` 模型层跳过 LM 前向：改上游模型代码，影响所有平台。

---

## 5. 修复代码（实际落地的 diff，共 +11 行）

文件：`vllm_ascend/worker/model_runner_v1.py`（修改前已备份为 `model_runner_v1.py.orig`）。

### 5.1 补丁①：`_dummy_run` 头部（+6 行）

插入位置：函数签名结束（`-> tuple[torch.Tensor, torch.Tensor]:`）之后、原首行注释 `# only support eager mode and piecewise graph now` 之前：

```diff
     ) -> tuple[torch.Tensor, torch.Tensor]:
+        mm_config = self.vllm_config.model_config.multimodal_config
+        if mm_config and mm_config.mm_encoder_only:
+            # The current dummy run only covers LM execution, so we can skip it.
+            # mm encoder dummy run may need to add in the future.
+            return torch.tensor([]), torch.tensor([])
+
         # only support eager mode and piecewise graph now
         assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_modes()
```

补丁后该函数头部实际内容（行 3665-3673）：

```python
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        # only support eager mode and piecewise graph now
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_mode()
```

### 5.2 补丁②：`_dummy_sampler_run` 头部（+5 行）

插入位置：签名结束后、原首行 `output = None` 之前：

```diff
     def _dummy_sampler_run(
         self,
         hidden_states: torch.Tensor,
     ) -> torch.Tensor:
+        mm_config = self.vllm_config.model_config.multimodal_config
+        if mm_config and mm_config.mm_encoder_only:
+            # MM Encoder only model no need to run sampler.
+            return torch.tensor([])
+
         output = None
```

### 5.3 实现细节与依据

- **返回值语义**：与上游完全一致——`_dummy_run` 返回空 tensor 元组 `(torch.tensor([]), torch.tensor([]))`（对应 `(hidden_states, num_scheduled_tokens)` 两个返回槽），`_dummy_sampler_run` 返回空 tensor。下游 `profile_run`/编译流程对 encoder-only 模式本就消费不到有效 logits，空 tensor 只需"存在且可迭代"。
- **访问器**：用 `self.vllm_config.model_config.multimodal_config`（与该文件内既有的 `self.vllm_config.model_config.*` 访问风格一致，如 585/3767/3956 行）。`multimodal_config` 可能为 `None`（纯文本模型），所以保留 `mm_config and ...` 的判空——这也是上游原写法。
- **`_dummy_pooler_run` 不补**：昇腾 fork 中该函数整体不存在（pooler 路径未移植），无从插入，自然免疫。
- **为什么 `torch.tensor([])` 不带 device**：上游即如此；空 tensor 不参与任何设备绑定的算子调用，无设备歧义。
- **改动面**：`git diff` 合计 +11 行、0 删改；`python -m py_compile` 通过。
- **补丁脚本**（139 上实际执行的应用器，锚点断言 + 幂等保护）：`agent_project/map_139/mmeo_patch/apply_patch.sh`。

---

## 6. 修复验证（0916，139 真机，全部闭环）

### 6.1 验证矩阵

| 验证项 | 命令/方法 | 补丁前 | 补丁后 |
|---|---|---|---|
| **目标场景**：encoder-only serve | `vllm serve /data1/Qwen3.5-27B --mm-encoder-only ... --port 8556` | Worker `TorchRuntimeError: Unhandled FakeTensor Device Propagation ... npu:0, meta` → `Engine core initialization failed` 必崩 | **`Started server process` → `Application startup complete`**；`GET /health` → **`200 OK`** ✅ |
| **回归①**：不带 flag 基线 serve | 同上去掉 `--mm-encoder-only`，`--port 8557` | （PASS 基线） | `Application startup complete`；`/health 200` ✅ |
| **回归②**：基线真实生成 | `POST /v1/completions {"prompt":"The capital of France is","max_tokens":8}` | `" Paris."` | **`" Paris."` 与补丁前逐字一致** ✅（`system_fingerprint: vllm-0.28.0-tp2-...`，TP2 正常） |
| **语法** | `python -m py_compile model_runner_v1.py` | — | OK ✅ |
| **影响面** | `mm_encoder_only` 默认 `False` | — | 不传该参数时**零行为差异**（早退条件恒假，等价于无补丁） ✅ |

### 6.2 为什么不需要全量回归

1. 补丁唯一分支条件是 `mm_config and mm_config.mm_encoder_only`；319 条全量参数中只有 `--mm-encoder-only`（test_value=True）1 条会走进早退，`--no-mm-encoder-only`（默认路径）及其余 317 条的执行路径与补丁前逐指令一致。
2. 已做目标场景 + 基线 startup + 基线真实生成三级验证（§6.1），覆盖了 dummy run 的两个消费方（编译 profile 与图捕获预热路径均经 `_dummy_run`）。
3. 与 gdn/kda 修复同理：影响面收敛到"显式传该开关"，风险不需要 20 小时全量来兜底。

### 6.3 验证过程中的环境管理事故与修正（如实记录）

首次基线回归时 serve 崩在 `_init_device`（显存不足），排查发现**是我自己的清理脚本缺陷**，与补丁无关：

- **缺陷**：`pkill -9 -f "vllm"` **大小写敏感**，只杀掉 cmdline 含小写 `vllm` 的 serve 主进程，漏杀进程名为 `VLLM::EngineCore` / `VLLM::Worker_TP0`（大写）的 worker——上一个 encoder-only serve 的 worker 残留、持续占着 NPU5 的 33G 显存，基线一启动就撞上。
- **修正**：
  1. 清理改用 `pkill -9 -i -f "vllm"` + `pkill -9 -f "VLLM::"`（大小写全覆盖）；
  2. **启动脚本加预检门槛**：启动前必须同时满足 ① 宿主机 `npu-smi info` 确认目标物理卡无进程；② 容器内 `torch.npu.mem_get_info` 确认目标逻辑卡 free ≥ 55G（CANN 显存回收有延迟，杀完进程轮询等待，不是 `sleep 3` 就够）；不满足直接拒绝启动；
  3. 修复后重跑基线，全部通过（§6.1 回归①②）。
- **教训已沉淀**（项目记忆 + `mmeo_patch/cleanup_verify.sh` / `launch_base_gate.sh` 可复用），与 0913 假 OOM 根因同族：进程清理与显存生命周期管理必须"真确认"，不能"假定已清理"。

---

## 7. 结论与建议

1. **`--mm-encoder-only` 已修好**：+11 行早退补丁在 139 真机实测闭环（encoder-only serve 正常 startup + 健康检查 200 + 基线生成零回归）。
2. **定性**：vllm-ascend 对上游 `gpu_model_runner.py` 的 fork 漏移植 3 处 `mm_encoder_only` 特判（昇腾侧实际存在其中 2 个函数），属移植缺口；上游无 bug。
3. **建议上游化**：可直接给 vllm-ascend 提 PR，标题建议 `[BugFix][Multimodal] Port upstream mm_encoder_only early-exit in dummy runs to fix startup crash`，diff 即本文 §5（改动极小、语义与上游逐字对齐、易评审）；单测可仿 PR #16661 的做法，构造 `mm_encoder_only=True` 的 VllmConfig 断言 `_dummy_run` 返回空 tensor 且不触碰模型前向。
4. **报告口径**：本条修复落地后，两轮全量（0913/0914，319 条）中确认的**昇腾侧产品缺陷共 2 项均已闭环**——gdn/kda 死开关（社区 PR #16661 承接）+ `--mm-encoder-only` 移植缺口（本补丁）；其余 FAIL 均为上游预期失败/负例设计/前置条件缺失，无未定性项。

---

## 8. 附注：同批复核的另一条 `--mm-encoder-attn-dtype=fp8`（无需修复）

为完整性记录同批复核结论：`fp8` 取值的报错来自**上游** `vllm/model_executor/layers/attention/mm_encoder_attention.py:426 _init_fp8_state` 的显式校验：

```
ValueError: mm_encoder_attn_dtype='fp8' requires either the ROCM_AITER_FA backend
on ROCm or the FlashInfer cuDNN backend on CUDA, got AttentionBackendEnum.TORCH_SDPA.
```

昇腾 ViT attention 走 TORCH_SDPA 后端、无 fp8 encoder kernel，上游给出的是**清晰报错而非裸崩**，归类同 `--use-replayssm`（平台限定支持），**无需修复**；可选优化是仿 PR #16661 做成"告警+回退 bf16"，非必须。

---

## 9. 附：材料清单

| 材料 | 路径 |
|---|---|
| 补丁应用脚本（锚点断言+幂等） | `agent_project/map_139/mmeo_patch/apply_patch.sh` |
| encoder-only 复现启动/轮询 | `agent_project/map_139/mmeo_patch/{inspect.sh, launch_mmeo.sh, poll_mmeo.sh}` |
| 根因深挖（FakeTensor traceback / 上游特判对照） | `agent_project/map_139/mmeo_patch/{inspect2.sh, mmeo_src.sh, mmeo_cmp.sh, mmeo_exc.sh, mmeo_deep.sh}` |
| 基线回归（带预检门槛） | `agent_project/map_139/mmeo_patch/{launch_base_gate.sh, poll_base.sh}` |
| 环境清理与显存回收确认 | `agent_project/map_139/mmeo_patch/{cleanup_verify.sh, final_cleanup.sh}` |
| serve 原始日志（容器内） | `/tmp/mmeo/serve_mmeo.log`（encoder-only）、`/tmp/mmeo/serve_base.log`（基线回归） |
| 修改前源码备份（容器内） | `vllm_ascend/worker/model_runner_v1.py.orig` |
| 0914 全量原始记录 | `0914_run/results_0914/server-api-log/results/MultiModalConfig/`、`results/summary.json`（mm-encoder-only FAIL 条目） |
| 总报告对应章节 | `OVERALL_REPORT_0913_0914.md` §0（TL;DR 缺陷清单）、§5.2、§7、§8.3 |