# 02. 挂载、契约与控制面

源码基线：

- vLLM Ascend：`0a97c475ab120ab2e182a358f5b1306eeddc7a8f`
- vLLM：`ba07e4a48fc951300d97eb506217dd530583dea3`

源码位置：

- vLLM 插件加载：`vllm/v1/engine/core.py`、`vllm/plugins/__init__.py`
- vLLM connector 工厂：`vllm/distributed/kv_transfer/kv_connector/v1/factory.py`
- vllm-ascend 注册入口：`vllm_ascend/__init__.py`（`register_connector`）
- Ascend connector：`ascend_store/ascend_store_connector.py`
- Scheduler 侧：`ascend_store/pool_scheduler.py`
- Coordinator：`ascend_store/coordinator.py`

本文覆盖控制面的全部内容：vllm-ascend 如何被 vLLM 挂载（插件机制）、`AscendStoreConnector` 承接的完整方法面（契约）、`KVPoolScheduler` 的命中查询与状态管理（控制链）、以及 hybrid 场景下 coordinator 为什么独立存在。行号均为核验时的准确位置。

---

## 1. 插件机制：vllm-ascend 如何被 vLLM 挂载

vLLM 主干代码不 import vllm-ascend，后者通过 entry point 把自己的 `register_connector` 送到 vLLM 的通用插件加载点上，由 vLLM 启动时主动调用。

`KVConnectorBase_V1`（vllm `kv_connector/v1/base.py`）是 vLLM 所有 KV 跨节点/cache 传输后端的通用抽象契约。vllm-ascend 仓库内有 8 个实现（AscendStore / UCM / RecomputeCPUOffload / SfaRemoteD2H / 三个 Mooncake / AscendMultiConnector），vLLM 上游还有 `SimpleConnectorBase_V1`、`BasicKVCacheConnector` 等。抽象的意义是让 Scheduler / ModelRunner 不认识具体后端，只调固定签名，换后端零改动。

`_registry` 是 `KVConnectorFactory` 上的类级 dict（factory.py L28）：key = 连接器名，value = 懒加载闭包（importlib 延迟 import 返回类）。它自己不长内容，键值 100% 由 `register_connector()` 事后填充；entry point 是"把注册函数送到 vllm 面前、让 vllm 主动调用它"的运输机制。

完整链路：

```text
① pip install vllm-ascend
    → setup.py: entry_points["vllm.general_plugins"]
      "ascend_kv_connector = vllm_ascend:register_connector"
      → 写入 vllm_ascend.dist-info/entry_points.txt（安装时写元数据，不执行）
② vllm 启动 → EngineCore.__init__ 调 load_general_plugins()        (v1/engine/core.py L117)
③ load_plugins_by_group("vllm.general_plugins")                    (plugins/__init__.py L36)
    entry_points(group=...) 枚举该组所有已装包插件        (L42)
    逐个 plugin.load() 拿到可调用对象                    (L68)
    VLLM_PLUGINS 门控（默认 None 全加载）                 (L56/L64)
④ load_general_plugins 对每个 func() —— 即调用 register_connector()   (L87-91)
    → KVConnectorFactory.register_connector(...)          (kv_transfer/__init__.py L45)
    → factory.py L30 把懒加载 loader 写进 _registry["AscendStoreConnector"]
⑤ 用户 --kv-connector AscendStoreConnector
    → get_connector_class 查 _registry → loader() 触发 import 拿类 → 实例化
```

关键点：

- vllm 扫 entry point 用 `importlib.metadata.entry_points(group=...)`；`plugin.load()` 才真正 import vllm-ascend。
- 不注册，vllm 就没有"按名字找实现"的入口，不实例化，也不调用。
- **依赖方向是单向的**：vllm-ascend → vllm。vLLM 是唯一驱动者。

### 1.1 撞名替换

`register_connector` 遇同名先 `_registry.pop(...)` 再重挂——Ascend 要顶替 vllm 上游已占用的名字（`MultiConnector` / `OffloadingConnector` / `SimpleCPUOffloadConnector`）得先弹掉，否则撞名 `raise ValueError`。`AscendStoreConnector` 是上游没有的新名字，直接注册。

这就是"换后端零改动"的另一面：Ascend 不仅要提供新实现，还要在三个通用名字上替换上游实现，让既有配置无缝落到 Ascend 版本。

### 1.2 factory 的校验行为

`create_connector`（factory.py L42-75）做 HMA 支持检查——HMA 已启用而连接器不支持时 `raise ValueError`（L54-60）。没有 isinstance 校验、没有 nop 回退；注册表查不到名字时 `get_connector_class` 直接 `raise ValueError`。

---

## 2. 调用链：从 schedule() 到 get_num_new_matched_tokens

```text
EngineCore/AsyncLLM 每步调 schedule()
  → Scheduler.schedule()
    → WAITING 请求循环
      → if request.num_computed_tokens == 0   （新请求/首轮才查）
        → connector._get_local_prefix_cache_hit()  本地 HBM prefix 命中
        → self.connector.get_num_new_matched_tokens(...)  ★vLLM 调 connector
          → AscendStoreConnector.get_num_new_matched_tokens()  薄转发
            → KVPoolScheduler.get_num_new_matched_tokens    真正外部 pool 命中查询
```

三个语义要点：

- **connector 诞生**：`Scheduler.__init__` 配了 `kv_transfer_config` 就 `create_connector(role=SCHEDULER)`；Ascend 侧 `__init__` 建 `connector_scheduler = KVPoolScheduler(...)`。
- **只在首轮查外部命中**：仅当 `num_computed_tokens == 0` 才询问 connector；先取本地 HBM 连续 prefix，再问外部。没配 `kv_transfer_config` 则 `self.connector` 为 None，走普通 prefix cache。新版本中 `_get_local_prefix_cache_hit()` 封装了本地命中查询，并处理 connector 支持 `supports_divergent_local_hybrid_hits` 时的差异路径。
- **传参语义**：传入的是按 block 对齐后的本地命中数（`block_aligned_local`，减去尾块），让"更长的外部命中"接管不足一个 block 的尾巴，避免 CoW 竞争。返回 `(ext_tokens, load_kv_async)`。这对应 `LoadSpec` 里 `vllm_cached_tokens` 与 `kvpool_cached_tokens` 分开。

---

## 3. AscendStoreConnector 方法全景

`__init__`：`role==SCHEDULER` 只建 `connector_scheduler`，否则只建 `connector_worker`；`role=worker 且 rank==0 且 !use_layerwise` 时额外起 `LookupKeyServer`（为什么，见 [07](07_lookup_path_design.md)）。Scheduler 侧拥有 `Request`、`KVCacheBlocks`、`SchedulerOutput` 和本地 block pool；Worker 侧拥有真实 tensor、NPU 地址、stream 和 backend client。connector 负责保持 vLLM 接口不变，并把调用转发给拥有相应资源的一侧。

新版本中 `__init__` 还初始化了 `_mamba_copy_bufs = None` 和 `requires_mamba_state_copy_after_layer_load = self.use_layerwise`，为 layerwise Mamba 状态拷贝做准备。

### 3.1 转给 KVPoolScheduler（scheduler 侧，6 个）

| 方法 | 作用 |
|---|---|
| `get_num_new_matched_tokens(request, n)` | 外部 pool 命中查询 |
| `update_state_after_alloc(request, blocks, n)` | 分配后写状态 |
| `build_connector_meta(scheduler_output)` | 打下行 metadata |
| `request_finished(request, block_ids)` | 单组请求结束/延迟释放判定 |
| `request_finished_all_groups(request, block_ids)` | 多 KV group 结束判定 |
| `bind_gpu_block_pool(gpu_block_pool)` | 给 scheduler 塞 BlockPool 用于 delayed free |

### 3.2 转给 KVPoolWorker（worker 侧，12 个）

| 方法 | 作用 |
|---|---|
| `register_kv_caches(kv_caches)` | 注册 HBM tensor、起传输线程 |
| `start_load_kv(forward_context)` | 读 metadata、开始加载 |
| `wait_for_layer_load(layer_name)` | layerwise 每层等待加载完，之后执行 Mamba 状态拷贝 |
| `save_kv_layer(layer_name, kv_layer, attn_metadata)` | layerwise 保存（含 `consumer_is_to_put` 判定） |
| `wait_for_save()` | 非 layerwise 批量保存（含角色判定） |
| `get_finished(finished_req_ids)` | 回传 `(done_sending, done_recving)` |
| `get_block_ids_with_load_errors()` | 回传失败 block 集合 |
| `get_kv_connector_kv_cache_events()` | 取 worker 的 `BlockStored` 事件包成 `AscendStoreKVEvents` |
| `build_connector_worker_meta()` | 上行 worker metadata |
| `set_external_slot_release_waiter(waiter)` | 外部 slot 释放回调，委托给 worker 判定 |
| `prepare_mamba_state_copy(copy_bufs)` | 准备 layerwise Mamba 状态拷贝 |
| `finish_mamba_state_copy()` | 完成 layerwise Mamba 状态拷贝 |

### 3.3 connector 自身处理（7 个）

| 方法 | 说明 |
|---|---|
| `requires_piecewise_for_cudagraph` (classmethod) | `use_layerwise` 时强制 PIECEWISE graph |
| `set_xfer_handshake_metadata_pp_aware` | 直接 `pass`，不走 P/D handshake |
| `update_connector_output(output)` | 半转半自：前半转 KVPoolScheduler + 后半做 KV event 聚合 |
| `take_events()` | 聚合多 worker 事件后吐给 prefix cache |
| `get_kv_connector_stats()` | 按存在性分流：有 scheduler 走 scheduler，否则 worker |
| `build_kv_connector_stats` (classmethod) | 构造 `AscendStoreKVConnectorStats` |
| `build_prom_metrics` (classmethod) | 构造 `AscendStorePromMetrics` |

### 3.4 契约外的扩展

`set_external_slot_release_waiter` **不是 vLLM 基类方法**，是 Ascend 内部扩展，仅被 `AscendMultiConnector` 调用。新版本中该方法的门控逻辑简化：不再检查 `use_gva_layerwise`，而是直接委托给 worker 层判定。读代码时遇到基类没有的方法，要先确认调用方是谁——这类方法只保证 Ascend 内部闭环，不参与 vLLM 生命周期。

---

## 4. KVPoolScheduler 初始化了哪些语义？

`KVPoolScheduler.__init__()` 不只是保存配置。它提前确定后续命中与 metadata 的基本坐标系：

```text
模型与并行维度
  model_name、TP、PP、PCP、DCP、kv_role

cache group 维度
  use_hybrid、group ids、cache family、Mamba/SWA group

粒度维度
  original_block_size
  grouped_block_size = original_block_size * PCP * DCP
  hash_block_size
  lcm_block_size
  cache_transfer_granularity

运行模式
  load_async、use_layerwise、use_layerwise_transfer
  consumer_is_to_load、consumer_is_to_put、save_decode_cache
```

其中最容易混淆的是三种 block size：

```text
hash_block_size
  request.block_hashes 的粒度

grouped_block_size[group]
  某 cache group 的有效 token block 粒度

cache_transfer_granularity
  所有参与 group 都能安全传输的公共粒度
```

`cache_transfer_granularity` 由 group block size 与 LCM 共同推导。外部命中最终必须向下对齐到这个粒度，否则 Scheduler 可能认为一段无法完整恢复的 KV 已经可用。

Scheduler 同时创建 backend 的 scheduler client。它只需要 lookup 能力，不注册 NPU buffer，也不执行 `put/get` 数据搬运（能力差异的完整分析见 [09 存储模型一节](05_transfer_backend_storage.md)）。

---

## 5. 外部命中查询有哪几条路径？

三条路径的完整设计决策分析（为什么分叉、减法/合并模型）见 [07](07_lookup_path_design.md)；本节只列控制面视角的分流。

### 5.1 普通 key 路径

`_generate_store_query_keys()` 从 block hash 展开完整存储 key。一个 hash 可能对应多个并行维度：

```text
block hash
  x PCP rank
  x DCP rank
  x head_or_tp_rank
  x PP rank（非 layerwise 查询时覆盖所有 PP）
```

layerwise 模式下的 key 生成现已委托给 `self.layerwise_protocol.make_hit_check_keys()`，由 backend 提供协议实现，不再在 scheduler 中内联展开。

`_get_store_lookup_hit_tokens()` 调用 backend `exists()`，并按 block 检查这一组 key 是否全部存在。命中必须是从查询起点开始的连续前缀：某一 block 出现明确 miss 后停止，异常状态则作为错误处理。

非 layerwise 模式可以从 `num_computed_tokens` 之后开始查；layerwise 模式必须从 block 0 查询，因为远端逐层对象与本地 prefix cache 的覆盖范围不一定一致。

### 5.2 MemCache layerwise 路径

`use_layerwise=True` 且 backend 为 MemCache 时，Scheduler 通过 `_make_layerwise_hit_check_keys()`（委托给 `self.layerwise_protocol.make_hit_check_keys()`）和 `batch_get_key_info()` 查询已分配的 GVA。每个 group、每个 block、每个有效 head/TP rank 都必须返回有效 size，当前 block 才算命中。不同 group 的命中长度最终取最小值：

```text
group 0 hit tokens = 4096
group 1 hit tokens = 3072
最终可恢复范围     = 3072
```

这是因为 forward 需要同时满足所有参与 cache group，而不是任选一个 group 命中即可。

### 5.3 LookupKeyServer/Client 旁路

非 layerwise 的 rank 0 Worker 可以启动 `LookupKeyServer`，Scheduler 侧 `LookupKeyClient` 通过 ZMQ 请求 Worker 执行 lookup。这条路径用于避免 Scheduler 直接依赖某些 Worker backend 初始化状态。

它仍然只返回命中长度或命中位置，不传输 tensor。真正 load 仍发生在 Worker 数据面。

---

## 6. Coordinator 为什么独立存在？

单一 Full Attention cache 的命中通常是连续 prefix，直接检查 block key 即可。Hybrid KV cache 中，不同 group 可能对应：

- Full Attention；
- Sliding Window Attention；
- Mamba/线性注意力状态；
- Eagle speculative cache；
- 不同 compress ratio 和 block size。

这些 group 的"某 token 范围是否可恢复"并不等价。`AscendStoreCoordinator` 把外部存在性伪装成 vLLM 可理解的 `BlockPool`，复用各 `SingleTypeKVCacheManager` 的可达性算法。

### 6.1 ExternalCachedBlockPool

`ExternalCachedBlockPool` 是一个 duck-typed block pool。它不管理真实 block，只把 `(group_id, block_hash)` 是否存在映射为"present block"或 cache miss。

这样 coordinator 可以调用 vLLM 已有的 `_find_longest_cache_hit()` 语义，而不在 AscendStore 中重新实现 Full/SWA/Mamba 的命中规则。

### 6.2 find_longest_cache_hit()

coordinator 先按有效 cache spec 对 group 分组，再分别调用对应 manager 的命中算法。多个 attention group 会迭代收敛到共同可恢复的 `hit_length`。

Eagle group 需要按规则丢弃或验证额外 speculative block；简单 hybrid 和一般 hybrid 的收敛过程也不同。最终返回：

```text
masks: 每个 cache group 哪些逻辑 block 可用
hit_length: 所有参与 group 共同支持的最长 token 范围
```

### 6.3 load/store/lookup mask

同一个 token 范围在不同操作下使用不同 mask：

```text
load_mask()   当前恢复范围中哪些 group block 应写入本地 cache
store_mask()  哪些状态对后续 token 仍可达，值得存到外部
lookup_mask() lookup 时哪些 chunk 必须参与存在性检查
```

例如 SWA 的旧 block 或 Mamba 不再可达的状态，不应仅因为 hash 存在就被当成有效恢复材料。

### 6.4 存在与使用是分离的

coordinator 的实例化条件是 `use_hybrid`——由**模型架构**决定，与传输模式 `use_layerwise` 无关，hybrid 架构配 layerwise 时 coordinator 实例照样创建；但使用门槛在 `if self.cache_coordinator is None or use_layerwise: return None`——layerwise 查询即使看到 coordinator 也不理它。**hybrid 架构决定 coordinator 存在，layerwise 传输模式决定它被闲置**，完整分析见 [07](07_lookup_path_design.md) 第 5 节。

---

## 7. get_num_new_matched_tokens() 的结果是什么？

这个接口返回的不是"外部一共存了多少 token"，而是"在当前请求状态和本地命中基础上，还能交给 vLLM allocation 的外部 token 数"。核心过程是：

```text
request.block_hashes
  -> 查询外部连续命中
  -> hybrid coordinator 求共同可达范围
  -> 处理 layerwise/Eagle/最后一个 token 等规则
  -> 按 cache_transfer_granularity 对齐
  -> 扣除 num_computed_tokens
  -> 记录 LoadSpec
```

`LoadSpec` 同时保存：

- `vllm_cached_tokens`：本地已经计算或命中的 token；
- `kvpool_cached_tokens`：外部 pool 支持恢复到的 token；
- `can_load`：Scheduler 是否允许本轮真正 load；
- `kvpool_store_skip_tokens`：保存时应跳过的已有远端前缀。

异步 load 模式下，接口还会告诉 vLLM 需要 allocation，但请求可能先进入等待远端 KV 的状态；layerwise 模式则不沿用完全相同的异步等待语义。

新版本中 Eagle 命中的处理逻辑得到改进：之前对 layerwise+Eagle 场景始终裁剪尾块（`max(num_computed_tokens, hit - lcm_block_size)`），现在改为仅当外部命中触及 prompt 的最后一个 granularity block 时才裁剪，避免在中间 block 边界上误伤有效的 mamba 状态快照。

---

## 8. update_state_after_alloc() 为什么关键？

命中查询时还没有目标 block 地址。只有 `KVCacheManager.allocate_slots()` 完成后，`update_state_after_alloc()` 才能同时看到：

```text
Request
外部命中 token 数
KVCacheBlocks（按 group 组织）
本地 num_computed_tokens
新请求、运行中或恢复后的状态
```

它把这些信息写入 Scheduler 侧状态：

- `_unfinished_requests` 保存 request 与已分配 block ids；
- `RequestTracker` 持续记录 token 长度、各 group block ids、已保存 token 和 GVA；`token_ids` 仅在 `enable_kv_events` 为 True 时拷贝（`#15364` 性能优化，避免大序列的冗余内存拷贝）；
- `_loading_req_ids` 标记仍在等待异步 load 的请求；
- `LoadSpec.can_load` 根据 allocation 结果决定；
- Mamba speculative block、SWA clip 和 layerwise GVA offset 在此后具备计算条件。

因此该方法的准确含义是"提交分配结果"，不是"开始传输"。

---

## 9. build_connector_meta() 如何处理不同请求？

`build_connector_meta()` 先处理 finished/preempted 集合，再构造本轮 `AscendConnectorMetadata`。请求按来源进入四条路径：

```text
scheduled_new_reqs
  -> _process_new_request()

曾被抢占、现在重新获得 block 的 cached request
  -> _process_preempted_cached_request()

持续运行并新增 token/block 的 cached request
  -> _process_running_cached_request()

没有参与本轮 forward、但外部 load 仍需执行的请求
  -> _process_async_load_request()
```

四条路径最终都尽量归一成 `ReqMeta`，但 tracker 的重建或增量更新方式不同：

- 新请求从当前 allocation 建立 tracker；
- 抢占恢复请求丢弃旧 block 账本后重建；
- 运行中请求追加 token ids 和新 block；
- 异步 load 请求以外部命中范围建立只 load 的 metadata。

`ReqMeta.from_request_tracker()` 还会推进 `num_saved_tokens`，确保已保存区间不会被每轮重复发布。

---

## 10. 请求结束、发送完成和释放

非 layerwise save 可能在 request 已结束后仍读取本地 block。`request_finished()` 因此可能返回 `delay_free_blocks=True`，并把 request id 放入 `_delayed_free_req_ids`。

```text
request finished
  -> 本地 block 仍被后台 send 使用
  -> 暂不归还 BlockPool
  -> Worker 报告 finished_sending
  -> update_finished_sending()
  -> 清除 delayed-free 状态
```

Hybrid HMA 路径通过 `request_finished_all_groups()` 处理各 group block；SWA block 还可能先裁剪。Layerwise 路径当前不使用同一 sending event 延迟释放协议，因此直接释放，buffer 复用安全由 layer event、NPU event 和 slot release 机制保证。

新版本中 layerwise 路径还额外跳过了 `_write_mamba_blocks_to_send_events`——layerwise 的 Mamba block 由 `KVCacheStoreSendingThread` 逐层释放，`build_connector_meta` 中再 bulk touch 会 double-reference 导致 block leak。

`update_connector_output()` 还会聚合各 Worker 的 completed event count。达到预期 Worker 数后，Scheduler 才从绑定的 `BlockPool` 释放对应 block，防止分布式 rank 尚未完成时提前复用。

---

## 11. 阅读控制面的检查表

定位命中或调度问题时，依次确认：

```text
1. 当前走普通 key、LookupKey 还是 GVA lookup？
2. hash_block_size、group block size、transfer granularity 是否一致？
3. 所有 TP/PP/PCP/DCP key 是否都命中？
4. hybrid coordinator 的共同 hit_length 和 mask 是什么？
5. LoadSpec 是否允许 load，allocation 是否成功？
6. RequestTracker 是新建、增量更新还是抢占后重建？
7. 请求结束后 block 是否仍处于 delayed-free？
```

---

## 12. 本文结论

```text
1. vllm-ascend 通过 entry point 反向注入注册表，vLLM 全程只认名字不认实现。
2. Ascend 替换了三个上游通用名字（MultiConnector/OffloadingConnector/SimpleCPUOffloadConnector）。
3. AscendStoreConnector 是 25 个方法的薄转发层，真正逻辑在 KVPoolScheduler/KVPoolWorker。
4. 外部命中查询只在首轮发生，且传入的是 block 对齐后的本地命中数。
5. coordinator 把外部存在性伪装成 BlockPool，复用 vLLM 可达性算法；存在由架构决定，使用由模式决定。
6. 新版本中 layerwise key 生成委托给 backend protocol，简化了 scheduler 的职责；Eagle 命中裁剪改为条件触发，避免误伤中间 block 边界处的 mamba 状态。
```

下一篇 [03](03_metadata_and_layout.md) 继续解释这些状态如何被编码成 key、request metadata、layer task 和地址布局。
