# 02. 插件挂载、Connector 契约与调度控制面

基线与源码入口见 [00](../00_source_baseline.md)。本篇按对象所有权和输入输出组织，不统计容易随版本失效的“connector 共几个方法”。

## 1. 从 entry point 到两个 role 实例

```text
安装包的 vllm.general_plugins entry point
  -> vllm_ascend:register_connector
  -> vllm_ascend.distributed.kv_transfer.register_connector
  -> KVConnectorFactory.register_connector

vLLM 创建 connector
  -> kv_connector/factory.py 查 registry、懒加载、验证 HMA
  -> AscendStoreConnector(..., role=SCHEDULER/WORKER, kv_cache_config)
  -> KVPoolScheduler 或 KVPoolWorker
```

vLLM 通用插件加载受 `VLLM_PLUGINS` 门控。注册保存可加载的实现，真正实例化才创建具体运行对象；注册不是 backend 初始化。

当前 Ascend 注册还包括：
- `MultiConnector` → `AscendMultiConnector`。
- `OffloadingConnector` → Ascend native wrapper。
- 上游 simple 模块可导入时替换 `SimpleCPUOffloadConnector`。
- `CPUOffloadingSpec` / `TieringOffloadingSpec` → NPU spec。
- `PreemptOffloadConnector`、`UCMConnector` 和各 P2P connector。
- `MooncakeConnectorStoreV1` 是 `AscendStoreConnector` 的旧名称入口，有弃用提示。

P2P Mooncake connector 与 AscendStore 内部 Mooncake backend 是两层不同组件，不能只凭 Mooncake 名称归到同一实现。

## 2. 契约分组

| 归属 | 关键接口 | 输入/输出与意义 |
|---|---|---|
| scheduler | `get_num_new_matched_tokens` | Request + 本地 token 数 → 外部新增贡献、是否异步 |
| scheduler | `update_state_after_alloc` | allocation + external tokens → 更新 load 状态 |
| scheduler | `build_connector_meta` | SchedulerOutput → 下行 metadata |
| scheduler | `request_finished*` | 结束通知 → 是否需要上游延迟 free；Store 当前 False |
| scheduler | `update_connector_output` | Worker 结果 → 清 loading、聚合 Mamba event |
| scheduler | `bind_gpu_block_pool` / `has_pending_push_work` | 接收 block pool；保证特殊未完成发送仍可推进 |
| worker | `register_kv_caches` | tensor/layout → 本进程地址表、backend 注册、线程 |
| worker | `handle_preemptions` | released ids → 覆盖前保存 fence |
| worker | `bind_connector_metadata` | layerwise 每轮准备；bulk 保存 metadata 引用 |
| worker | `start_load_kv` / `wait_for_layer_load` | bulk 启动或逐层等待 |
| worker | `save_kv_layer` / `wait_for_save` | layerwise 提交或 bulk batch 入队 |
| worker | `get_transfer_results` | sending/receiving/failed receiving 集合 |
| 双侧/辅助 | stats、KV events、worker metadata、`shutdown` | 观测、聚合与资源生命周期 |

`set_external_slot_release_waiter`、DSpark configure/restore/save 是 Ascend 内部协同接口。判断某个 hook 是否通用契约，应到当前 vLLM base 查定义并查调用方，不能从命名推测。

## 3. Scheduler 的坐标系

`KVPoolScheduler.__init__()` 统一以下信息：

| 维度 | 内容 |
|---|---|
| 模型/并行 | model、KV heads、TP/PP、PCP/DCP、kv_role |
| group | cacheable group、cache family、Mamba/SWA、private state |
| 粒度 | original/grouped/hash/LCM/transfer granularity |
| 模式 | `use_layerwise`、`layerwise_data_plane`、`load_async`、save/decode |
| 协议 | `layerwise_keys`、coordinator、scheduler backend client |

当前 Scheduler `grouped_block_size[group] = original_block_size[group] * dcp_size`；hash 粒度由 `prefix_match_unit` 或最小可缓存 block 推导后乘 DCP。仅 cacheable group 参与相关 LCM。PCP 和 KVPP 的 Worker 数据布局另行处理，不能把旧的统一乘法套回这里。

`store_scheduler` 无条件创建；非 layerwise 仍走 ZMQ，不能据此声称 Scheduler 没有后端访问能力。

## 4. 命中入口与返回值

`get_num_new_matched_tokens()` 先处理 consumer load 门控、请求长度和 granularity，再按数据面查询，详细算法见 [07](07_lookup_path_design.md)。

后处理顺序中有三件关键事：
1. 保留原始命中 `store_skip_tokens`，用于避免重复保存已有完整对象。
2. 根据 Eagle 最后 granularity block、最后一个 token、私有 state 约束修正可跳过计算的范围。
3. 计算新增 allocation；有 layerwise 恢复需求时，即使新增为零也保存 `LoadSpec`。

Eagle 尾部边界用 `(request.num_tokens - 1) // lcm_block_size * lcm_block_size` 推导。只有命中进入最后块时才裁剪，不能对所有中间块命中都减一块。

`LoadSpec` 中：
- `vllm_cached_tokens`：本地覆盖水位。
- `kvpool_cached_tokens`：允许恢复/跳过计算的目标水位。
- `kvpool_store_skip_tokens`：外部已有对象对应的保存跳过水位。
- `can_load`：allocation 后本轮是否执行 load。

返回 async 标志仅为 `load_async and not use_layerwise`；逐层 pipeline 不采用同一 waiting-remote 协议。

## 5. 分配后状态与跨轮次 tracker

`update_state_after_alloc()` 记录 `_unfinished_requests`；存在 LoadSpec 时检查外部 token 数与两水位差是否一致。bulk async load 加入 `_loading_req_ids`。

`RequestTracker` 跨 step 累积：
- 各 group 本地 block ids。
- token 长度与已安排保存水位。
- Mamba speculative/null block 更新。
- GVA/partial 关联。
- KV events 所需 token 信息；事件关闭时避免复制完整 token 序列。

allocator 决定 block ids；tracker 记录它们并生成计划，不重新分配 block。

## 6. 一轮 metadata 的四类来源

| 来源 | 方法 | 处理重点 |
|---|---|---|
| new | `_process_new_request` | 从 allocation 建立 tracker，合并 load |
| 抢占后重新分配 | `_process_preempted_cached_request` | 丢弃旧 block 账本，重建 |
| 持续运行 | `_process_running_cached_request` | 增量 block/token；decode 边界与保存开关 |
| 未参加当前 forward 的 async load | `_process_async_load_request` | 仍需执行接收任务 |

`build_connector_meta()` 先清 finished/preempted tracker，再把 finished ∪ preempted 写进 `released_req_ids`。decode 保存不能只检查本 tick 是否分配新 block：一个 block 可能早已分配，而在本 tick 才填满。

`ReqMeta.from_request_tracker()` 推进的是“已安排保存”的水位；发送线程另有成功保存 offset 与重试状态，不能把两者当成后端持久化确认。

## 7. Hybrid coordinator：Scheduler 与 Worker 都有

`AscendStoreCoordinator` 使用 `ExternalCachedBlockPool` 包装存在性，复用 vLLM 各类型 cache manager 的可达性/命中规则。它自身不是外部存储，也不分配真实设备 block。

```text
group hashes + query callback + lookup masks
  -> 查询当前候选范围真正需要的 group blocks
  -> find_longest_cache_hit / find_reachable_hit_tokens
  -> 所有 group 共同可恢复的 token 水位
```

mask 的分工：
- `lookup_mask`：候选恢复位置需要查哪些对象。
- `load_mask`：恢复该位置必须写回哪些状态。
- `store_mask`：哪些当前状态值得发布供未来恢复。

Full、SWA、Mamba/压缩状态并不要求同样的连续对象前缀。当前 layerwise hybrid 也使用 coordinator；仅对无 coordinator 的简单场景才能用逐 group 连续命中再取 min。私有 state 可能不发布为公共 prefix，而在本地重建。

## 8. 结束与特殊引用释放

当前 request 结束回调返回 `(False, None)`；普通保存以 Worker released fence 保护覆盖顺序，见 [06](06_concurrency_and_config.md)。

另外 `touch_sending_mamba_blocks()`：
1. 仅非 layerwise、hybrid、确需保存的请求触发。
2. 对非 null、非 speculative scratch 的 Mamba block 增加引用。
3. Worker 上报 `completed_events`，Scheduler 按 `_expected_worker_count` 聚合。
4. 所有预期 Worker 完成后，按尾部优先顺序 `free_blocks`。
5. `has_pending_push_work()` 反映这些 event，避免引擎过早停止推进。

speculative scratch 必须保持 allocator 所需的独占引用条件；把它也 touch 会破坏后续重定位。layerwise 不重复 bulk touch，避免引用泄漏。

下一篇：[03 Metadata 与 Layout](03_metadata_and_layout.md)。
