# vLLM Ascend KV Pool 源码知识体系

本目录按 **2026-10-19 本地代码快照**整理。Ascend 基线为 `2982425cc7cc00249feda1eaaab3b9bf40432c11`，vLLM 基线为 `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`，后者与 Ascend 的 main verified commit 一致。版本、核验范围和变化见 [源码基线](00_source_baseline.md)。

学习主线是：**调度为什么允许跳过计算 → 用什么 metadata 描述恢复范围 → 如何得到真实地址 → 后端何时发布对象 → 谁保证数据可读和资源可复用**。按文件逐个读容易遗漏这些跨文件约束，因此下面同时提供概念路线和问题路线。

## 1. 阅读地图

| 层次 | 文档 | 读完应掌握 |
|---|---|---|
| 版本与证据 | [00 源码基线与变更](00_source_baseline.md) | 当前版本、源码入口、旧结论纠正、维护方法 |
| 全局边界 | [01 KV Pool 整体定位](01_kv_pool_overview.md) | vLLM block 管理、connector、pool、CPU offload 的职责 |
| 请求闭环 | [Store 01 总览](ascend_store/01_overview.md) | 从 waiting 查询到 load、forward、save、结果回传 |
| 控制面 | [Store 02 挂载与控制面](ascend_store/02_connector_and_control_plane.md) | 插件注册、调度接口、tracker、hybrid coordinator |
| 描述层 | [Store 03 Metadata 与 Layout](ascend_store/03_metadata_and_layout.md) | key、token/block 粒度、快照、地址与 offset |
| 设备编排 | [Store 04 Worker Pipeline](ascend_store/04_worker_pipeline.md) | cache 注册、每轮准备、线程分流、Mamba copy |
| 后端协议 | [Store 05 传输与存储](ascend_store/05_transfer_backend_storage.md) | object / GVA / block-key 三种数据面及发布协议 |
| 资源安全 | [Store 06 并发与配置](ascend_store/06_concurrency_and_config.md) | released fence、event、session、lease、配置约束 |
| 查询推导 | [Store 07 查询路径](ascend_store/07_lookup_path_design.md) | ZMQ/直连、并集与可达性、查询与 load 起点 |
| 组合场景 | [Store 08 MultiConnector、DSpark 与 KVPP](ascend_store/08_composition_dspark_kvpp.md) | 多 connector 的资源协同、target/draft 联合恢复 |
| 证据与排障 | [Store 09 指标、测试与排障](ascend_store/09_observability_testing.md) | 指标覆盖边界、测试地图、症状到代码的定位路径 |
| 本地 offload | [Native / Simple offload](other_paths/kv_offload.md) | 上游 manager + Ascend canonical layout / DMA |
| 抢占保护 | [Preempt offload](other_paths/preempt_offload.md) | 抢占时真实 KV 的 D2H/H2D、CPU block 生命周期 |
| 外部委托 | [UCM 接入](other_paths/ucm_connector.md) | 包装层、reserved hook 转发、分析边界 |
| 复习 | [读码问题与术语](notes.md) | 常见误读、地址心智模型、复习清单 |

旧链接 [recompute_cpu_offload](other_paths/recompute_cpu_offload.md) 保留为迁移说明，当前源码入口是 `kv_offload/preempt_offload/`。

## 2. 先辨认自己正在看的模式

| 模式 | 外部身份/地址 | 查询位置 | Worker 主要执行方式 |
|---|---|---|---|
| 非 layerwise | `PoolKey`，一 key 聚合多个内存片段 | Scheduler 经 ZMQ 委托 rank 0 Worker | `Backend.get/put`，同步或后台 load |
| Mooncake layerwise | 一个 block/group/stage 对象，层数据是对象内 range | Scheduler 直连、检查 committed/readable | start session → 每层 range copy → commit/end |
| MemCache layerwise | 一个完整 stack 对象的 GVA，层数据是 GVA 内 offset | Scheduler 直连 `batch_is_readable` | alloc/lease → 分层 copy → write finish |
| 无上述协议的 layerwise 分支 | `LayerPoolKey`，每层独立对象 | Scheduler 直连 exists | KeyLayer 发送/接收线程 |
| Native / Simple CPU offload | CPU block 索引与 tensor view | 各自 connector scheduler | 本地 CPU ↔ NPU 搬运 |
| Preempt offload | 被抢占请求的 CPU blocks 与恢复事件 | `PreemptOffloadScheduler` | 抢占前保存，恢复时加载 |

**layerwise 是执行粒度，不保证“一层一个外部 key”；是否共享 HBM buffer 还要看 backend 的 reuse protocol。** Mooncake 当前 block-key 协议不启用 GVA 物理复用。

## 3. 按问题阅读

- 命中为零/过长：先看 07，再查 03 的 namespace 与粒度；PP 和 hybrid 尤其要检查跨进程 key 一致性。
- key 命中但结果不对：先看 04 的 group/stride/Mamba copy，再看 05 的 commit/readable 与 06 的覆盖顺序。
- 请求结束或抢占后出错：先看 06 的 `released_req_ids` fence；本地抢占 offload 则看 Preempt 专篇。
- layerwise 卡住：看 06 的依赖顺序和 09 的空任务、异常、session 清理测试。
- 显存减少但延迟上升：看 03 的共享 buffer、07 的重载范围，以及 09 的测量边界。
- 要修改代码：沿 02→03→04→05 确定影响层，再在 09 找行为测试；不要仅凭某个方法名推断完成语义。

## 4. 使用约定

文中 `ascend_store/...`、`kv_offload/...` 均相对 `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/`；`vllm/...` 相对上游仓库根目录。稳定引用采用“文件 + 符号”，可点击的源码入口集中在 00，避免代码更新后大批裸行号失效。

这里是源码分析，测试表记录已有覆盖，**不代表本次运行了 NPU、后端服务或模型精度测试**。实现中的注释、文档字符串若与执行体不同，以执行体为准；推导性的例子会明确标注。
