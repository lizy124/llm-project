# 阶段总结：Layerwise 池化优化前基线摸底（165）

> 日期：2026-09-07
> 环境：pool_165（A3 × 16 卡 / 910B3 / python3.12.13 / editable /vllm-workspace）
> 栈：vllm upstream main（33e849499，fetch 无更新）**未含** PR #15442 / #15854
> 目标口径（对齐 [07_colleague_test_data.md](../07_colleague_test_data.md)）：TP4×DP4、16 卡、max_len 1M、MTP、并发 32、输入 131072

---

## 1. 基线要测什么（目录用途）

存"**未做自己 layerwise 优化**"时的性能摸底数据，作优化前对照：
- 关键指标：**TPS / TTFT / 命中率 / MTP 采纳率**（同事表格缺 TPS/TTFT，正是要补的）
- 负载口径：单前缀 1×131072 与 多前缀 32×131072，并发 32，输出 1
- 策略对照：layerwise 池化（优化前） vs HBM prefix cache 基线（`baseline/` 目录已有部分）

**约束**：基线数据必须与"优化后"同栈、同配置（TP/DP、MTP 开关、max_len、并发数），否则对比无效。

---

## 2. 已尝试配置与结果（含失败）

### 2.1 TP4×DP4 + 单 APIServer（`--api-server-count 1`，同事原样配置）
- 结果：**死锁**。请求全落 DP0（DP0 执行 MoE prefill），DP1-3 卡 `dummy_run` → `_sync_metadata_across_dp` 的跨 DP all_reduce。
- 根因：单 APIServer 只连主 EngineCore(DP0)；DP1-3 无请求不参与跨 DP 同步 → DP0 永远等不到 → 死锁。

### 2.2 TP4×DP4 + 4 APIServer（`--api-server-count 4`，SO_REUSEPORT 共享 8100）
- 结果：**死锁**（32 并发全超时）。4 个 `VLLM::APIServer_0..3` 均启动，但请求仍只落 DP0（内核 reuse_port 分发不均/时序）。
- 证据：DP0 real prefill，DP1/2 卡 `dummy_run` all_reduce。

### 2.3 TP4×DP4 + multi-port-external-lb（4 独立端口 8100-8103，各连各 DP）
- 修复过程：`--data-parallel-multi-port-external-lb` 需补 `--data-parallel-size-local 4`（否则 validate 报错）。
- 结果：**仍死锁**。每端口独立请求，DP0 进 `step_with_batch_queue`（真实请求），DP1-3 卡 `execute_dummy_batch`（warmup）；后续轮次 DP0 卡主模型 DSA CP `_switch_o_proj_to_full_weight`（tp_weight_switch.py:277 `wait_tp_weight_all_gather`），DP1-3 卡 MTP `_propose` 的 `_sync_metadata_across_dp` all_reduce。
- **根因（栈级）**：`_sync_metadata_across_dp`（model_runner_v1.py:752）在每次 execute_model 无条件做跨 DP all_reduce。DP 间 step 节奏只要错位（任一 DP 无请求 / 处理快慢不一）即互等死锁。**165 栈缺 PR #15854（reachable-store/MTP trim 修复）与 #15442（多 main spec 布局）**，属栈级缺陷，配置无法对齐解决。

### 2.4 TP4×DP4 + multi-port + 关闭 MTP
- 结果：**仍死锁**（主模型 `execute_model` → `_determine_batch_execution_and_padding` → `_sync_metadata_across_dp` 同样无条件跨 DP all_reduce）。证明与 MTP 无关，纯 DP 同步缺陷。

### 2.5 TP16×DP1（单 DP 规避跨 DP 同步）
- 结果：**启动崩溃**。`deepseek_v4/model.py:1293 load_weights` → `linear.py:473` `shape '[0, 1024, -1]' is invalid for input of size 2097152`（`n_local_groups=0`）。
- 根因：TP16 下 MoE/attention 分组切分为 0，模型切分 bug，TP16 不可行。

### 2.6 TP8×DP1 + max_len 1M
- 结果：**启动失败**。`ValueError: 1M seq 需 17.07 GiB KV cache/卡 > 可用 11.82 GiB（估算最大 187264）`。
- 处理：max_len 降至 131072（对齐同事输入长度）。

### 2.7 TP8×DP1 + max_len 131072（当前，进行中）
- 状态：**已重启**（pid 260313，20:11），等待就绪验证。
- 单 DP 无跨 DP all_reduce，规避 2.1-2.4 死锁；131072 ≤ 187264 满足 KV 内存。
- **注意**：此口径 ≠ 同事口径（TP8×DP1 vs TP4×DP4；关 MTP），仅用于**验证 layerwise KV pool 主链路 + 功能级命中率**，正式基线口径待定（见 §4）。

### 2.8 TP8×DP1 + 131072 + layerwise（use_layerwise: true，20:16）
- 结果：**MoE 通信死锁**。单请求 120s 超时。
- 现象：8 Worker 全卡 `_moe_forward_shared`（fused_moe/moe_runner.py:161 → torch._ops 原生通信）；EngineCore 卡 `acquire_read`（shm_broadcast.py:795）等 worker 响应；EngineCore 每 60s 报 `shm_broadcast.py:802 No available shared memory broadcast block`。
- 死锁环：worker 卡 MoE all_reduce（rank 间节奏错位）→ 结果块未读完 → EngineCore 写/读块超时 → EngineCore 无法推进。
- **根因方向**：layerwise 预取（`layerwise_prefetch_layers:3`，pool_worker.py load 路径 `should_wait` 主线程等异步 KV 加载）使 **rank 间到达 MoE 层的时间不一致 → all_reduce ring 不闭合 → 死锁**。

### 2.9 ✅ TP8×DP1 + 131072 + 纯 KV 池（use_layerwise: false，20:30，隔离实验）
- 结果：**跑通**。单请求 OK（13s）；32 并发 **OK=32/32（8s）**。
- **结论（关键）**：`use_layerwise: false`（memcache KV 池，无 layerwise 预取）下 MoE 通信完全正常 → **死锁根因确认为 layerwise 预取机制**，与 TP/DP 配置无关（DP4 死锁 + TP8 死锁同一根因：layerwise 预取使 rank 不同步）。

### 2.10 TP8×DP1 + 131072 + layerwise + prefetch_layers=1（20:39）
- 结果：**仍死锁**（单请求 120s 超时；32 并发 0/32）。
- 结论：**预取窗口大小（3→1）不是关键**。

### 2.11 TP8×DP1 + 131072 + layerwise + prefetch_layers=0（20:49）
- 结果：**启动失败**。`layerwise_cache_layout.py:147 ValueError: layerwise_prefetch_layers must be at least 1`（`_PREFETCH_LAYERS` 强制 ≥1）。
- **结论：layerwise 必然带 ≥1 层预取，而 prefetch=1/3 均死锁 → 本栈 layerwise 无可用调参组合（C2 升级为必然死锁）**。

### 2.12 死锁根因分析（layerwise 机制层面）
- 死锁环（TP 多卡）：rank A 先进 attention（加载 gate 通过）→ 卡在 TP attention/MoE all_reduce（rank B 未到）→ rank A 的该层 KV 保存无法进行 → rank B 的层加载依赖源层保存（`wait_for_save_layer`）永远等不到 → rank B 卡 `wait_for_layer_load` → 循环等待。
- 纯 memcache（非 layerwise）正常原因：KV 加载是**主模型 forward 前一次性同步批量 lookup**，不在层间让 rank 错位。
- 同事能跑 layerwise（TP4×DP4 + prefetch=3）的差异：**165 栈缺 PR #15442/#15854**（main spec 布局 / reachable-store 修复），大概率含 layerwise 同步修复。

---

## 3. 关键证据位置

- 启动脚本：`D:\lzy\project\kv_pool\tmp\start_dsv4_layerwise.sh`（当前 TP8×DP1 / 131072 / 关 MTP）
- 日志（容器内）：`/home/lizhongyang/map_165/run/dsv4_layerwise_v2_8100.log`（历次已备份）
- 验证脚本：`kv_pool/tmp/verify_dp1.sh`（单端口：1 请求 + 32 并发）
- 死锁栈证据：EngineCore `step_with_batch_queue` / `execute_dummy_batch`（core.py:701/929/2108/2124/2137）；Worker `_sync_metadata_across_dp`（model_runner_v1.py:752）；DSA `wait_tp_weight_all_gather`（tp_weight_switch.py:277）

---

## 4. 结论与后续

1. **165 栈 DP>1 必然死锁**（`_sync_metadata_across_dp` 无条件跨 DP all_reduce + 缺 PR #15442/#15854），与同事环境（upstream + 两 PR + 极光平台分发）无法通过配置对齐。
2. **正式基线两条路**：
   - **A（推荐）**：165 栈打 PR #15442 + #15854 补丁 → 恢复 TP4×DP4 + MTP + 1M 同事口径，采对齐基线。
   - **B（降级）**：接受 TP8×DP1 + 131072 + 关 MTP 变体，作为功能验证基线；与同事数据不可直接对比。
3. **待办**：① 当前 TP8×DP1 验证通过后，测单/多前缀命中率 + TPS/TTFT；② 确认是否打 PR 补丁恢复同事口径；③ 补 MTP 采纳率数据。
