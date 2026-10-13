# PR #17624：KV Pool Nightly 验证与问题跟踪

> 更新：2026-10-12（北京时间）。跨会话接续入口。记录整个 PR 的 S1–S4、CI 运行、PR 侧与环境侧责任；每次提交或重跑后更新状态和运行表。配置事实以 [PR 当前 diff](https://github.com/vllm-project/vllm-ascend/pull/17624/files) 对应的 commit 为准，旧版 [test_guide.md](./test_guide.md) 中的 S2/S4 后端及性能基线已过时。

## 当前结论

- [PR #17624](https://github.com/vllm-project/vllm-ascend/pull/17624) 于本次核查时为 OPEN，head `9cc0aa4ffa3a2d72d71c97403c4482f94283661a`，改动 4 个 DeepSeek nightly YAML。普通 PR checks 的通过或跳过不代表 4 个 nightly 场景已验证。
- **S4 已有效触发，但两个 case 均在 Mooncake/Fabric 初始化阶段失败**；尚无接受率、池命中或性能结论。**S1、S2、S3 尚无本次核查确认的有效专项 nightly 结果**。S1 曾在获得权限前触发，被 bot 拒绝；其余场景不能从 S4 的结果推断通过。
- 下一步优先并行核实 S4 runner 的 Fabric 能力与配额、准备 PR 侧的小容量诊断；另外为 S1/S2/S3 分别建立有效运行和证据。不要用旧指南的阈值替代当前 YAML 判卷。

## 四个场景的验证台账

| 场景与配置文件（`tests/e2e/cases/models/configs/DeepSeek/` 下） | 当前 PR 配置与目标 | 已确认的 CI 状态 | 下一个可判定结果 |
|---|---|---|---|
| S1 `DeepSeek-V4-flash-w8a8-PD-prefix.yaml` | 双节点 PD；Memcache（5000/6000）与 MooncakeHybrid 组成 MultiConnector；AIME 精度及新增 TPOT50 基线 15569.48 / 阈值 0.95。 | [2026-09-30 的 `/nightly` 请求](https://github.com/vllm-project/vllm-ascend/pull/17624#issuecomment-5903444236)被权限检查拒绝；这不是测试失败。未确认后续有效运行。 | 权限已具备后重跑，核对 PD 双端启动、池写入、AIME 和 TPOT50；若路由异常，检查 MultiConnector 配置与 runner 日志。 |
| S2 `DeepSeek_V4_Flash_A5.yaml` | A5 的 pooled case #1：Memcache `device_urma`、10GB、AscendStore `kv_both` layerwise；GPQA 87.4±5；warmup 后共享前缀 perf，目前 `baseline: 1` / `threshold: 0.1` 为占位。其他 case 不应被误记为已池化。 | 未确认有效专项 nightly 运行或池命中证据。 | 核对 A5 环境支持 `device_urma`、case #1 启动及池写回/命中；记录真实吞吐并回填可靠基线。 |
| S3 `DeepSeek-V4-Flash-W8A8-A3.yaml` | A3 16 卡；Memcache `device_sdma`、10GB；关闭 prefix caching，AscendStore `kv_both` layerwise；GPQA 及 warmup→共享前缀 perf，性能基线目前也是 `1` / `0.1` 占位。 | 未确认有效专项 nightly 运行；读路径正向命中未证实。 | 先取得 prefix-off 下 `kvpool hit tokens > 0` 或等价的真实 load-back 证据，再看精度、吞吐；即使考题绿但命中恒 0，也不能算读路径覆盖完成。 |
| S4 `acceptace_rate_deepseekv4-flash-w8a8-mtp.yaml` | A3 8 卡，open/close 两 case；Mooncake master 50088、metrics 50089，32GB segment，`ASCEND_ENABLE_USE_FABRIC_MEM=1`；MTP3 接受率基线 0.925±0.05，warmup→perf 的 `1` / `0.1` 为占位。 | [有效 run 37871717656](https://github.com/vllm-project/vllm-ascend/actions/runs/37871717656)：两个 case 均失败，详见下文。 | 排除 Fabric 初始化故障后，两 case 均跑完，核对接受率、真实池回读、性能，并回填性能基线。 |

PR 中 S1/S2/S3 使用 Memcache，S4 使用 Mooncake。S1 的 MooncakeHybrid 用于 PD KV 传输，不能据此把 S1 的**池后端**记成 Mooncake。S2 的 `device_urma` 和 S3 的 `device_sdma` 也不能混用。以上是 [PR 当前 diff](https://github.com/vllm-project/vllm-ascend/pull/17624/files) 的静态配置；它不能证明 runner 具有相应能力或运行已成功。

## 全 PR 责任划分

| 事项 | PR 作者要做的事 | 需要环境维护者确认或处理的事 |
|---|---|---|
| 4 个 YAML 的配置及验证语义 | 核对后端、connector、端口、容量、warmup→perf 顺序、case 覆盖与 CI 平台；取得实际池写入/回读证据。对占位性能基线在有效运行后回填，并确认判卷阈值。 | 提供各平台 runner 的可用资源、网络和服务生命周期信息；若测试平台调度或依赖服务有问题，协助定位。 |
| S1 双节点 PD | 验证两端 MultiConnector 的角色、engine_id、路由和 PD 传输；分析 AIME、TPOT50 回归。 | 核实双节点网络、MooncakeHybrid 与 Memcache 服务连通、模型/数据集资源。 |
| S2 A5、S3 A3 Memcache | 核对 `device_urma` / `device_sdma` 选择、10GB 配额、prefix-off 的命中语义；把只启动成功与真正池读命中分开。 | 核实对应机型、驱动、内存及 Memcache/URMA/SDMA 可用性和镜像依赖版本。 |
| S4 Mooncake/Fabric | `ASCEND_ENABLE_USE_FABRIC_MEM=1` 和每 rank `global_segment_size: 32GB` 是 PR 主动选择；评估容量，设计同 runner 小容量诊断，按结果修改 PR。 | 查明该 runner 的 Fabric share handle 支持情况，提供 Mooncake wheel、灵衢、驱动/HDK 版本及 `fabric_memory.max_capacity`、实际可用容量；环境能力不足时修复或给出可用配置。 |

故障归属应由同 commit、同 runner 的日志和隔离实验决定。配置不匹配由 PR 修，runner/镜像/驱动/配额不满足约定由环境侧处理；仅凭一次报错不把责任全归某一方。

## S4 已确认的故障证据

- 正确触发命令：`/nightly acceptace_rate_deepseekv4-flash-w8a8-mtp --a3-560t`；此前不带平台参数的请求虽通过权限检查，却没有调度实际 S4 测试。有效 [run](https://github.com/vllm-project/vllm-ascend/actions/runs/37871717656) / [job](https://github.com/vllm-project/vllm-ascend/actions/runs/37871717656/job/113632380845) 于 2026-10-09 结束，pytest `2 failed`。
- runner：`linux-aarch64-a3-800i-8-cn12-001-5dlkl-runner-st69n`；镜像 `nightly-ci-main-a3`，`skip_build_image=true`。CI checkout PR head `9cc0aa4`，实际 `CONFIG_BASE_PATH` 为 `tests/e2e/cases/models/configs/DeepSeek`，pytest 入口为 `tests/e2e/common/single_node/test_single_node.py`。
- 日志出现 `mooncake_master --port 50088` 和 `mooncake KV pool is ready`：master 并非完全没有配置或启动。输出可能缓冲，不能只据行序还原进程时序。
- worker 注册 `34359738368 B`（32 GiB）段时报 `Memory_Allocation_Failure(EL0004)`、`rtMemExportToShareableHandleV2 ... driver error:feature not support`、`Failed to register mem`，继而 Mooncake 初始化 `ret=-600`。后续 `127.0.0.1:50088 Connection refused` 不足以证明 master 从未启动，也不能单独定根因。
- 日志已知 CANN `9.2.0`、npu-smi `25.5.1`；**Mooncake wheel、灵衢/HDK 详细版本、Fabric 配额未知**。CANN 和 npu-smi 数字不能代替这些信息。[vLLM-Ascend KV Pool 文档](https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/feature_guide/kv_pool.html)列出 Fabric 模式前提与容量要求；目前无法断定唯一根因。

复核命令：`gh run view 37871717656 --repo vllm-project/vllm-ascend --job 113632380845 --log`。搜索 `mooncake KV pool is ready`、`Memory_Allocation_Failure`、`feature not support`、`34359738368`、`Initialize mooncake failed`。

## 待办与复验判据

1. **S4 环境核实**：将 job 链接、runner 名称、`feature not support`、32 GiB 申请量提供给环境维护者，索取能力、版本和配额结论。本文仅准备沟通内容，尚未代用户发送。
2. **S4 PR 侧隔离**：同 runner、同镜像尝试更小且按 1 GiB 对齐的 segment。若仍同样报 `feature not support`，优先查环境能力；若通过，重新确定负载所需容量，随后完整复验。小容量测试只是诊断，不能代替性能验收。
3. **S1/S2/S3 分别取得有效运行**：先确认各自 nightly 入口/平台参数，再记录 run/job、commit、runner、配置文件、服务启动、精度与吞吐。S2/S3/S4 的 warmup→perf 应有实际正向池命中证据；尤其 S3 不以考题全绿代替读路径验证。
4. **整体验收**：4 个场景的预期 case 均在同一最终 PR commit 上通过；没有未解释的池初始化或读写故障；性能占位基线已根据有效测量回填；记录与本地对照的模型、镜像、硬件、配置和负载差异。

## 运行记录（追加，不覆盖历史）

| 时间（北京时间） | 场景 / commit | 命令或运行 | 结果与后续 |
|---|---|---|---|
| 2026-09-30 | S1 / 当时提交 | [无权限的 `/nightly` 请求](https://github.com/vllm-project/vllm-ascend/pull/17624#issuecomment-5903444236) | Bot 拒绝；未产生有效 S1 结果。 |
| 2026-10-08 | S4 / `9cc0aa4` | [未带 `--a3-560t` 的请求](https://github.com/vllm-project/vllm-ascend/pull/17624#issuecomment-6056818460) | 权限检查通过，但没有调度实际 S4 测试。 |
| 2026-10-09 | S4 / `9cc0aa4` | [带平台参数的请求](https://github.com/vllm-project/vllm-ascend/pull/17624#issuecomment-6072587676) → [run 37871717656](https://github.com/vllm-project/vllm-ascend/actions/runs/37871717656) | open/close 两 case 失败；Fabric 内存注册失败，Mooncake 初始化 `ret=-600`。 |
| 待填 | S1/S2/S3/S4 / 待填 | run/job 链接、命令、runner | case 数、启动/池读写证据、精度、吞吐、结论。 |

## 本地与 CI 对照注意

本地 `C:\E_segment\project\code\vllm-ascend` 的 `pool_nightly` 分支存在未提交的 S4 YAML 修改：`tests/e2e/nightly/single_node/models/configs/acceptace_rate_deepseekv4-flash-w8a8-mtp.yaml` 从 Memcache 改为 Mooncake（50088/50089、32GB），但未设置 PR YAML 的 `ASCEND_ENABLE_USE_FABRIC_MEM=1`。它与 CI 实际读取的 `tests/e2e/cases/models/configs/DeepSeek/...` 路径和内容均不同，不能用来解释既有 CI 结果。以后本地/CI 比较时，对齐 commit、配置、镜像和负载后再比较结论。
