fix_9363_compat_report.md
# 134 服务器 Ascend910 9363 兼容性修复全记录

> 日期：2026-09-11
> 机器：80.48.29.134（Atlas 800I A3，64G HBM × 16 die）
> 状态：✅ 已修复并验证（Qwen3-0.6B vllm 冒烟通过）
> 修复人：agent（lizhongyang 会话）
> 相关文档：`agent_project/map_134/FIX_9363.md`（技术速查）、`map_134/README.md`（134 盘点）、`map_134/PREP.md`（七阶段进度）

---

## 一、背景：为什么会有这次修复

A3 HCCS 双机池化验证项目（本目录上级）规划：
- **Phase 1（当前）**：64G 单机基线 —— 在 134 上跑通 DSv4-Flash w8a8-mtp，TP8 单机推理
- **Phase 2**：跨机池化验证

134 是新接入的 64G 机型（之前 128G 验证在 133/135 系列完成），按新服务器 7 阶段流程接入：

| 阶段 | 内容 | 结果 |
|---|---|---|
| 1 连通性 | SSH 免密、host key | ✅ |
| 2 盘点 | NPU/驱动/权重/容器 | ✅ 16 chip 可见，驱动 26.2.rc1 |
| 3 登记 | servers.yaml + README/PREP | ✅ |
| 4 建容器 | 镜像从 135 移植，创建 a3heng64_134 | ✅ |
| 5 小模型冒烟 | Qwen3-0.6B 打通 vllm | ❌→✅ **本文档主题** |
| 6 KV 后端 | mooncake + memcache | 待做 |
| 7 基线验证 | DSv4 w8a8 TP8 | 待做（权重传输 84%） |

**阶段 5 首跑即失败**，报错：
```
RuntimeError: Unsupported soc version: Ascend910 9363
```

---

## 二、问题本质：芯片修订号与软件栈白名单的代际错配

### 2.1 soc version 是什么

Ascend 芯片对上层软件自报"身份证"——soc version（如 `Ascend910_9392`）。torch_npu 等
框架编译时内置一张"认识的 soc 白名单"，遇到表外型号直接拒绝初始化。

### 2.2 三台机器对比（关键事实）

| 机器 | 驱动 | CANN runtime | 芯片 soc | torch_npu 2.10.0.post4 认识吗 |
|---|---|---|---|---|
| 135 | 26.0.rc2 | 8.5 | 9392 | ✅（表内） |
| 133 | 26.1 | 9.x | 251（归一化枚举） | ✅（表内） |
| **134** | **26.2.rc1** | **9.2.0** | **9363** | ❌ **表外** |

`strings libtorch_npu.so` 确认白名单含：`9361/9362/9372/9381/9382/9391/9392`，**无 9363**。

### 2.3 为什么 134 的芯片是 9363

134 装的是最新一代驱动+CANN（26.2.rc1 / 9.2.0）。宿主 CANN 9.2 的 `ascend950_aicore_stl`
组件库里出现 9363 字符串 —— **9363 是 9362 的后继修订（新一代批次），只被 CANN 9.2 世代支持**。
而手头所有 vllm-ascend 镜像都是上一代（torch_npu 2.10.0.post4 + CANN 9.1）产物。

一句话：**134 是"最新硬件批次 + 新驱动"，手头镜像是"上一代软件栈"，白名单没长出新芯片。**

### 2.4 旁证：hyl（同机其他用户）的现场

134 上发现 `/home/hyl/` 环境在 9 月 4-7 日间做过 9363 适配验证：
- torch_npu **v2.7.1** 内部构建 wheel（华为云 OBS 内部源，非公共可下载）
- 配套 `/home/hyl/cannb050/`（CANN 9.2 安装包，b050=驱动 26.2.rc1.b050 配套）
- 其调试日志残留 `'cc': 'Ascend910_9363'` 的成功运行记录

**证明：CANN 9.2 世代 + 适配版 torch_npu 可以在 134 上跑**。当时结论偏向"等官方适配镜像"，
但本修复证明：**不等新镜像，用三层 patch 也能跑通**。

---

## 三、排查过程：穷尽"正规升级"路径（全部无货）

用户决策"不绕着走，优先修复 134"。先系统排查了所有能拿到 9363 支持的途径：

| # | 途径 | 结果 |
|---|---|---|
| 1 | 华为云公共 pip 源升 torch_npu | ❌ 源上最高只到 2.9.0 dev，无 2.10.x wheel（容器里 2.10.0.post4 是镜像内置） |
| 2 | 134 容器/宿主直接访问 pip 源 | ❌ 无外网（DNS 解析失败） |
| 3 | 135 代下 wheel | ❌ 135 也无外网（curl/pip 全超时） |
| 4 | 本地 Windows 访问华为云源 | ❌ 已确认无 2.10.x（网页版目录列表核实） |
| 5 | nightly-main-a3 最新镜像（133 上，9-8 构建） | ❌ 同样 torch_npu 2.10.0.post4 + CANN 9.1 |
| 6 | 134 本地 5 个镜像（day902/904/909/910/nightly） | ❌ 全部 2.10.0.post4，全报 9363 |
| 7 | hyl 的内部 OBS 源直连 | ❌ 不可达（内网限制） |
| 8 | hyl 的 HTTP 代理（80.253.95.93:6688） | ❌ 不通（仅对特定网段开放） |
| 9 | 宿主 CANN 9.2 自带 python 包 | ❌ site-packages 无 torch_npu |

**结论：正规渠道全部断供。** 剩两条路：a) patch 现有二进制；b) 放弃 134 改用 133。
用户拍板"先修复，实在不行才用 133"。

---

## 四、修复方案：三层 patch（核心）

### 思路

9363 与 9362 是**同家族相邻修订**（CANN 9.2 组件按 9362 一脉组织）。既然软件栈认识 9362，
就**让 9363 在各个校验点"冒名" 9362**，按 9362 路径初始化/编译/执行。指令集兼容性由
同家族假设保证 —— 最终 Qwen3 推理验证通过，DSv4 大模型待验（见第六节风险）。

### Patch 1：libtorch_npu.so 字符串表（解决 soc 识别）

**问题**：torch_npu 初始化时字符串匹配 soc 名，`9363` 不在表 → 拒绝初始化。

**做法**：so 内白名单字符串直接替换（同长度，仅 1 字节差异）：
```
Ascend910_9362\x00  →  Ascend910_9363\x00
```
```python
so = '/usr/local/python3.11.10/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so'
data = open(so,'rb').read()
open(so,'wb').write(data.replace(b'Ascend910_9362\x00', b'Ascend910_9363\x00'))
```

**效果**：`get_soc_version()` 返回 255（9362 槽位枚举），torch.npu 初始化、tensor 上卡成功。

**注意**：so 里恰好只有 1 处 `Ascend910_9362` 字符串，全替换无副作用。原文件已备份
`libtorch_npu.so.orig9362`（容器内）+ `patched_libtorch_npu.so`（宿主 `/home/lizhongyang/map_134/`）。

### Patch 2：容器挂载宿主 CANN 9.2 runtime（解决算子执行）

**问题**：soc 识别通过后，第一个算子 `aclnnMuls` 即失败（ERR00100）——
镜像内 CANN **9.1 算子库没有 9363 的 kernel**，只有宿主 CANN **9.2** 有。

**做法**：
1. 容器多挂一个只读卷（create_ct_134.sh 已加）：
   ```
   -v /usr/local/Ascend/cann-9.2.0:/usr/local/Ascend/cann-9.2.0:ro
   ```
2. 运行时 env 把 9.2 的 lib64 放到 LD_LIBRARY_PATH **最前**：
   ```bash
   export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64:${LD_LIBRARY_PATH}
   ```

**效果**：Muls/Sum/Linear/BatchMatmul 全部跑通。

**关键坑（勿踩）**：**只换 runtime 库，不换 ASCEND_TOOLKIT_HOME**。
- 曾把 `ASCEND_TOOLKIT_HOME` 也指向 9.2 → triton 编译器（bishengir，与镜像 CANN 9.1 配套）报错
- 正确姿势：**编译器用镜像内 9.1，运行时用宿主 9.2**（混合模式）。ASCCEND_OPP_PATH 等也不动

### Patch 3：triton get_arch 映射（解决 torch.compile 编译）

**问题**：vllm 启动时 torch.compile 触发 triton 编译，`bishengir-compile` 收到
`--target=Ascend910_9363` 直接退出（其 target 枚举到 9362/9392 为止）。

**做法**：patch triton 的 driver.py，在源头把 arch 名映射回 9362：
```python
# /usr/local/python3.11.10/lib/python3.11/site-packages/triton/backends/ascend/driver.py
def get_arch(self):
    arch = self._load_mod().get_arch()
    # [9363-adapt] 9363 无 bishengir target, 按 9362 编译(同系列指令兼容)
    if arch == "Ascend910_9363":
        arch = "Ascend910_9362"
    return arch
```
bishengir 的 target 是通过 `NPUUtils().get_arch()` 单点获取的，改这一处即覆盖所有传参路径。

**效果**：`--target=Ascend910_9362` 编译通过，vllm profile_run/引擎初始化成功。

### 三层 patch 的关系（缺一不可）

```
芯片(9363)
  │
  ├─ torch_npu 初始化 ── Patch1: so 白名单字符串 9362→9363 ──→ 识别成功(按9362)
  │
  ├─ aclnn 算子执行 ── Patch2: 挂宿主 CANN 9.2 runtime(有9363 kernel) ──→ 算子跑通
  │
  └─ triton/torch.compile ── Patch3: get_arch 9363→9362 ──→ 编译通过
```
- 只做 1：算子执行报 ERR00100
- 只做 1+2：vllm 引擎初始化时 triton 编译失败
- 三层齐做：全流程通

---

## 五、验证结果

### 5.1 算子级验证（容器内，混合 runtime）
```
[1] import OK
[2] SOC enum: 255          ← 9363 识别成功
[3] randn 上卡 OK: npu:0
[4] Muls+Sum 算子 OK
[5] Linear(matmul) OK
[6] BatchMatmul OK
ALL PASS
```

### 5.2 vllm 全流程冒烟（Qwen3-0.6B，TP=1，port 8004）
```
===== Qwen3-0.6B 冒烟 PASS =====
服务就绪于 19x5s（95 秒）
/health OK
推理输出（连贯中文）:
"，然后用一句话介绍你的研究目标，并且用一句话介绍你的研究方法..."
system_fingerprint: vllm-0.26.0-9db1375f
```

### 5.3 当前环境快照
- 容器：`a3heng64_134`（镜像 dev-26.2.0.day20260902-800I-A3-py311-Ubuntu24.04，含 9.2 挂载）
- 16 chip 全部可见，hccl 通信正常（world_size=1 验证过；TP8 未验）
- dsv4-Flash w8a8-mtp 权重传输中（280G，84%，约 40 分钟完成）

---

## 六、风险与边界（后续必读）

1. **指令兼容假设未在 DSv4 上验证**
   9363 按 9362 编译执行，基于同家族指令兼容假设。Qwen3（0.6B dense）已通过，
   **DSv4 w8a8-mtp（MoE + MTP + w8a8 量化）是更大考验**——若出现计算错误/算子失败，
   优先怀疑此处。验证方法：对比 133（标准 soc）同模型输出/loss 是否一致。
2. **so patch 是"槽位冒名"**：patch 后真 9362 机型反而不被识别。仅影响本容器内 torch_npu，
   其他容器/镜像不受影响。
3. **混合 runtime 的隐性风险**：编译器(9.1) 与运行时(9.2)跨代。若遇到诡异 segfault/精度问题，
   检查是否混载了 9.1/9.2 的库（`ldd` 检查 libascendcl 等应来自 9.2）。
4. **官方适配到位后回归标准态**：拿到 CANN 9.2 配套、白名单含 9363 的 torch_npu/vllm-ascend
   镜像后，移除三层 patch（恢复 .orig9362 / driver.py.orig，去 9.2 挂载），避免长期背着 patch。
5. **TP8 未验证**：冒烟只跑了 TP=1。DSv4 基线前建议先 Qwen3 TP8 快速验证 hccl 多卡路径。

---

## 七、容器重建 SOP（patch 重放，10 分钟内完成）

场景：容器误删/镜像更新/需要重建时，按此流程恢复到可用状态。

```bash
# 1. 重建容器（脚本已含 CANN 9.2 挂载）
bash /home/lizhongyang/map_134/scripts/create_ct_134.sh

# 2. 恢复 patched so（宿主备份 → 容器）
docker cp /home/lizhongyang/map_134/patched_libtorch_npu.so \
  a3heng64_134:/usr/local/python3.11.10/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so

# 3. 恢复 patched triton driver
docker cp /home/lizhongyang/map_134/patched_triton_driver.py \
  a3heng64_134:/usr/local/python3.11.10/lib/python3.11/site-packages/triton/backends/ascend/driver.py

# 4. 冒烟验证（脚本内已注入 9.2 runtime env）
docker exec a3heng64_134 bash /home/lizhongyang/map_134/scripts/smoke_qwen3_134.sh
```

**任何在 134 容器内跑 vllm/torch_npu 的脚本，开头必须带：**
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64:${LD_LIBRARY_PATH}
```
（smoke_qwen3_134.sh / 后续 DSv4 启动脚本已内置；新写脚本勿忘，否则 soc 识别通过但算子失败）

## 八、备选方案：133

133（80.48.29.133）soc 为标准 251 枚举（torch_npu 原生识别），环境即插即用：
- 已有 `kv_pool_layerwise` 容器（day20260904 镜像，torch_npu 2.10.0.post4 正常初始化）
- DSv4 w8a8-mtp 权重现成在 `/data1/`（免 280G 传输）
- 负载低（load ~6.5 vs 135 的 36）

**触发条件**：若 DSv4 在 134 上出现 9363 兼容性导致的计算错误/算子崩溃且无法快速定位，
立即切 133 做基线，134 修复回归为独立攻关任务（等官方适配镜像）。

---

## 附：时间线

| 时间 | 事件 |
|---|---|
| 09-11 14:xx | Qwen3 首冒烟失败：`Unsupported soc version: Ascend910 9363` |
| 09-11 15:0x | 定位白名单机制；测全部 6 个镜像同版本 torch_npu，全不支持 |
| 09-11 15:1x | 排查 pip 源/nightly/OBS/代理，全部无货；用户拍板"修复优先" |
| 09-11 15:2x | 发现 hyl 环境证明 9363 可跑；侦察 so 确认字符串表匹配机制 |
| 09-11 15:44 | Patch 1 落地：soc 识别通过，但 aclnnMuls 失败 |
| 09-11 15:47 | Patch 2 落地：挂 CANN 9.2 runtime，算子全通 |
| 09-11 15:49 | vllm 二次失败：bishengir 不认 9363 target |
| 09-11 15:53 | Patch 3 落地：triton get_arch 映射 |
| 09-11 15:54 | **Qwen3 冒烟 PASS**，方案固化（FIX_9363.md + 备份） |