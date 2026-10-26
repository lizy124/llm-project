# 00. 源码基线、证据入口与更新记录

## 1. 本次核验的本地快照

核验日期：2026-10-19。分析对象是用户指定的两个本地仓库，未拉取远端，也未改变源码。

| 仓库 | HEAD | commit 日期 |
|---|---|---|
| `vllm-ascend` | `2982425cc7cc00249feda1eaaab3b9bf40432c11` | 2026-10-19 |
| `vllm` | `c4df37dfcc7f72ca41cd43801a468526ecf99a2b` | 2026-09-30 |

[main verified 文件](../../../../../vllm-ascend/.github/vllm-main-verified.commit) 等于本地 vLLM HEAD；[release tag 文件](../../../../../vllm-ascend/.github/vllm-release-tag.commit) 为 `v0.31.0`，它是另一种版本选择依据，不能把它写成本次 checkout 的版本。核验时两份源码工作树均无本地修改。

旧编号篇基线为 Ascend `0a97c475ab120ab2e182a358f5b1306eeddc7a8f` / vLLM `ba07e4a48fc951300d97eb506217dd530583dea3`，原 notes 还引用过更早的 `6e448d0ea9`。本次统一为上表快照。

## 2. 需要替换的旧认知

| 旧说明 | 当前执行体与正确读法 |
|---|---|
| 工厂在 `kv_connector/v1/factory.py` | 当前是 `kv_connector/factory.py`，V1 接口仍在 `v1/base.py` |
| `recompute_cpu_offload` 只保存 metadata，不保存 KV | 当前迁至 `kv_offload/preempt_offload`；Worker 分配 CPU KV tensor 并执行真实 D2H/H2D |
| coordinator 只在 Worker 存在，layerwise 会闲置 | Scheduler 也构造 coordinator；GVA hybrid 和 block-key hybrid 都可进入可达性查询 |
| Mooncake layerwise 是每层一个 key | 当前是 block-key session/range 协议；一对象聚合 stage-local 层，逐层写不同 offset |
| MemCache GVA 是“一层一对象” | 完整对象分配包含相关层数据的空间；逐层 copy 使用对象内 offset |
| layerwise 查询检查 `size > 0` 就足够 | 统一调用 backend 的 `batch_is_readable`；MemCache 还需处理 SSD 等介质 |
| AscendStore 靠 request delayed-free / `finished_sending` 回收 | `request_finished*` 返回 False；metadata 携带 `released_req_ids`，Worker 在覆盖前等待相关保存任务 |
| `wait_for_save` 方法名保证当前保存全部完成 | 非 layerwise 执行体只 `add_save_batch` 入队；真正覆盖保护见 `handle_released_saves` |
| consumer 关闭 `consumer_is_to_put` 就绝不保存 | `save_decode_cache` 也参与 `can_put`；decode-only 保存有独立限制 |
| group block size 一定乘 PCP×DCP | 当前 Scheduler 的 grouped size 乘 DCP；Worker 另有 PCP/KVPP 布局处理，不能套统一公式 |
| 外部池只有 DRAM | 当前 Mooncake 有可选 SSD offload；MemCache 对 SSD key 有专门恢复路径 |

这些是本次知识体系的修订点，不表示它们都由同一个提交引入。

## 3. vLLM 证据入口

| 文件 | 重点符号/问题 |
|---|---|
| [connector 工厂](../../../../../vllm/vllm/distributed/kv_transfer/kv_connector/factory.py) | `KVConnectorFactory`：懒加载、HMA 校验、实例化 |
| [V1 契约](../../../../../vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py) | `KVConnectorBase_V1`、`KVConnectorTransferResults`、`SupportsHMA` |
| [Scheduler](../../../../../vllm/vllm/v1/core/sched/scheduler.py) | `_get_local_prefix_cache_hit`、`schedule`、`_handle_failed_recving` |
| [SchedulerOutput](../../../../../vllm/vllm/v1/core/sched/output.py) | `kv_connector_metadata`、`has_sync_kv_loads` |
| [KVCacheManager](../../../../../vllm/vllm/v1/core/kv_cache_manager.py) | 本地 prefix 命中与 allocation |
| [BlockPool](../../../../../vllm/vllm/v1/core/block_pool.py) | 引用计数、touch、free 与 prefix cache |
| [V1 runner mixin](../../../../../vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py) | `_get_kv_connector_output`：同步 load 前置、其他 load 可后置 |
| [V2 runner connector](../../../../../vllm/vllm/v1/worker/gpu/kv_connector.py) | `pre_forward`、`post_forward`、`get_transfer_results` |
| [native offload](../../../../../vllm/vllm/v1/kv_offload) | CPU/tiering spec、manager、worker 协议 |

## 4. Ascend 证据入口

| 文件 | 重点符号/问题 |
|---|---|
| [注册入口](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/__init__.py) | `register_connector` 同时替换 connector 和 offloading spec |
| [AscendStoreConnector](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py) | role 分流、bind 时准备、transfer results、shutdown |
| [pool_scheduler](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py) | 查询、tracker、released ids、Mamba sending event |
| [coordinator](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py) | `find_reachable_hit_tokens`、`find_longest_cache_hit`、load/store/lookup mask |
| [metadata](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py) | key、粒度、tracker、ReqMeta、`LayerRangeReqMeta` |
| [pool_worker](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py) | 注册地址、准备任务、执行/等待、失败与资源清理 |
| [kv_transfer](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py) | `LayerBatchBuilder`、线程、range/GVA 执行、request 保存 fence |
| [backend 协议](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py) | readable、object、GVA、session/range 能力 |
| [Mooncake layerwise](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_layerwise.py) | key namespace、PP/context parallel 校验、session 构造 |
| [MemCache](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py) | GVA、SSD、lease、atomic publication |
| [session tracker](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/session_tracker.py) | 跨 chunk 的 put/get owner 与 retry/terminal 清理 |
| [layerwise layout](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py) | `get_layerwise_reuse_config`、reuse layout、物理 plan |
| [attention fence](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py) | gate、callback、`attention_transfer_window` |
| [DSpark prefix](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/dspark_prefix_cache.py) | target/draft 身份与 companion object |
| [metrics](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metrics.py) | 非 layerwise get 计时、统计聚合 |
| [MultiConnector](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py) | 子 connector metadata、slot release、DSpark 联动 |
| [PreemptOffload](../../../../../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/preempt_offload) | manager、CPU KV、抢占/恢复与完成事件 |

## 5. 以后如何保持文档与代码一致

1. 记录两边 HEAD、dirty 状态及 verified 配对；本地有修改时同时记录相关 diff，不能只标 commit。
2. 先比较 `kv_pool/`、注册入口、MultiConnector，再比较 vLLM base、Scheduler、runner 的契约。
3. 顺着一次请求核对 lookup→allocation→metadata→load→forward→save→release；尤其检查返回值和真正等待点。
4. 对每种 backend 分别确认 key 身份、对象粒度、可读判定、发布、失败、清理，不能从相同方法名类推。
5. 更新本页、README 的模式表以及 06/09 的配置、测试地图，最后检查相对链接与过期符号。

本次为静态源码和已有测试阅读，未执行后端/NPU 推理验证。文中“支持”指当前代码提供的路径与校验边界，实际运行仍依赖 SDK 和硬件环境。
