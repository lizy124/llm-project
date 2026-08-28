# AscendStore / KVPool 读码笔记

> 基于 vllm `commit 6e448d0ea9` + 本地 vllm-ascend 源码。行号均为核验时的准确位置。
> 重组说明：按主题组织（挂载 → 结构 → 查询 → 存储 → 核心问题 → 本地命中），原 Q&A 随记序号已并入对应章节。

---

## 一、插件机制：vllm-ascend 如何被 vllm 挂载

**结论**：`KVConnectorBase_V1`（vllm `kv_transfer/kv_connector/v1/base.py`）是 vLLM 所有 KV 跨节点/cache 传输后端的通用抽象契约，不止 AscendStoreConnector——vllm-ascend 仓库内有 8 个实现（AscendStore/UCM/RecomputeCPUOffload/SfaRemoteD2H/三个 Mooncake/AscendMultiConnector），vLLM 上游还有 SimpleConnectorBase_V1、BasicKVCacheConnector 等。抽象的意义是让 Scheduler/ModelRunner 不认识具体后端，只调固定签名，换后端零改动。

**依赖方向是单向的**：vllm-ascend → vllm。vllm 不知道 vllm-ascend 里有哪些类，是 vllm-ascend 主动往 vllm 的空工厂里"塞"自己的实现。要被 vllm 认可：① 运行时把名字注册进 `KVConnectorFactory`；② 继承 `KVConnectorBase_V1` 并满足签名。**vLLM 是唯一驱动者**，vllm-ascend 只负责"满足契约、登记注册、响应调用"。

### 1.1 注册链路：entry point 如何注入 `_registry`

`_registry` 是 `KVConnectorFactory` 上的类级 dict（factory.py L28）：key = 连接器名，value = 懒加载闭包（importlib 延迟 import 返回类）。它自己不长内容，键值 100% 由 `register_connector()` 事后填充；entry point 是"把注册函数送到 vllm 面前、让 vllm 主动调用它"的运输机制。完整链路：

```
① pip install vllm-ascend
    → setup.py:entry_points["vllm.general_plugins"]
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
- `register_connector` 遇同名先 `_registry.pop(...)` 再重挂——Ascend 要顶替 vllm 上游已占用的名字（`MultiConnector` / `OffloadingConnector` / `SimpleCPUOffloadConnector`）得先弹掉，否则撞名 `raise ValueError`。`AscendStoreConnector` 是上游没有的新名字，直接注册。
- 不注册，vllm 就没有"按名字找实现"的入口，不实例化，也不调用。

**factory 的校验行为**：`create_connector`（factory.py L42-75）做 HMA 支持检查——HMA 已启用而连接器不支持时 `raise ValueError`（L54-60）。**没有** isinstance 校验、没有 nop 回退；注册表查不到名字时 `get_connector_class` 直接 `raise ValueError`。

### 1.2 调用链：从 vLLM 的 schedule() 到 get_num_new_matched_tokens

```
EngineCore/AsyncLLM 每步调 schedule()
  → Scheduler.schedule()                                  scheduler.py L439
    → WAITING 请求循环                                     L687
      → if request.num_computed_tokens == 0               L745  （新请求/首轮才查）
        → kv_cache_manager.get_computed_blocks_for_connector()  L757 本地 HBM prefix 命中
        → self.connector.get_num_new_matched_tokens(...)  L778  ★vLLM 调 connector
          → AscendStoreConnector.get_num_new_matched_tokens()  ascend L138 薄转发
            → KVPoolScheduler.get_num_new_matched_tokens    真正外部 pool 命中查询
```

- **connector 诞生**：`Scheduler.__init__ L140` 配了 `kv_transfer_config` 就 `create_connector(role=SCHEDULER)`；Ascend 侧 `__init__ L114` 建 `connector_scheduler = KVPoolScheduler(...)`。
- **只在首轮查外部命中**：L745 仅当 `num_computed_tokens == 0`；先取本地 HBM 连续 prefix，再问外部。没配 `kv_transfer_config` 则 `self.connector` 为 None，走普通 prefix cache。
- **传参语义**：L769-781 传入的是按 block 对齐后的本地命中数（`block_aligned_local`，减去尾块），让"更长的外部命中"接管不足一个 block 的尾巴，避免 CoW 竞争。返回 `(ext_tokens, load_kv_async)`。这对应 `LoadSpec` 里 `vllm_cached_tokens` 与 `kvpool_cached_tokens` 分开。

---

## 二、AscendStoreConnector：薄转发层

`__init__`（L114-125）：`role==SCHEDULER` 只建 `connector_scheduler`，否则只建 `connector_worker`；`role=worker 且 rank==0 且 !use_layerwise` 时额外起 `LookupKeyServer`。真正实现全在 KVPoolScheduler / KVPoolWorker。

### 2.1 转给 KVPoolScheduler（scheduler 侧，6 个）

| 方法 | 作用 |
|---|---|
| `get_num_new_matched_tokens(request, n)` | 外部 pool 命中查询 |
| `update_state_after_alloc(request, blocks, n)` | 分配后写状态 |
| `build_connector_meta(scheduler_output)` | 打下行 metadata |
| `request_finished(request, block_ids)` | 单组请求结束/延迟释放判定 |
| `request_finished_all_groups(request, block_ids)` | 多 KV group 结束判定 |
| `bind_gpu_block_pool(gpu_block_pool)` | 给 scheduler 塞 BlockPool 用于 delayed free |

### 2.2 转给 KVPoolWorker（worker 侧，10 个）

| 方法 | 作用 |
|---|---|
| `register_kv_caches(kv_caches)` | 注册 HBM tensor、起传输线程 |
| `start_load_kv(forward_context)` | 读 metadata、开始加载 |
| `wait_for_layer_load(layer_name)` | layerwise 每层等待加载完 |
| `save_kv_layer(layer_name, kv_layer, attn_metadata)` | layerwise 保存（含 `consumer_is_to_put` 判定） |
| `wait_for_save()` | 非 layerwise 批量保存（含角色判定） |
| `get_finished(finished_req_ids)` | 回传 `(done_sending, done_recving)` |
| `get_block_ids_with_load_errors()` | 回传失败 block 集合 |
| `get_kv_connector_kv_cache_events()` | 取 worker 的 `BlockStored` 事件包成 `AscendStoreKVEvents` |
| `build_connector_worker_meta()` | 上行 worker metadata |
| `set_external_slot_release_waiter(waiter)` | gva_layerwise 外部 slot 释放回调（**非 vLLM 基类方法**，Ascend 内部扩展，仅被 `AscendMultiConnector` L51 调用） |

### 2.3 connector 自身处理（7 个）

| 方法 | 说明 |
|---|---|
| `requires_piecewise_for_cudagraph` (classmethod) | `use_layerwise` 时强制 PIECEWISE graph |
| `set_xfer_handshake_metadata_pp_aware` | 直接 `pass`，不走 P/D handshake |
| `update_connector_output(output)` | 半转半自：前半转 KVPoolScheduler + 后半做 KV event 聚合 |
| `take_events()` | 聚合多 worker 事件后吐给 prefix cache |
| `get_kv_connector_stats()` | 按存在性分流：有 scheduler 走 scheduler，否则 worker |
| `build_kv_connector_stats` (classmethod) | 构造 `AscendStoreKVConnectorStats` |
| `build_prom_metrics` (classmethod) | 构造 `AscendStorePromMetrics` |

---

## 三、查询路径总览：两条路径，分界是 use_layerwise

- **非 layerwise**：Scheduler 通过 ZMQ 把查询委托给 rank0 worker，由 worker 用 `m_store.exists()` 查并合并本地 HBM 命中，回传命中数（为什么，见第五部分）。
- **layerwise**：Scheduler 用自己的 backend 客户端 `store_scheduler.batch_is_exist / batch_get_key_info` 直连远程 store 查，不经 worker 中转。

### 3.1 Scheduler 侧（KVPoolScheduler）

入口 `get_num_new_matched_tokens`（pool_scheduler.py L444），职责 = 选路 + 短路 + 算 LoadSpec，不碰数据。

**三道短路**（L461/464-470/481-482）：
1. `kv_consumer` 且 `!consumer_is_to_load` → 返回 `(0, False)`
2. 设了 `retention_interval` 且 prompt 过短 → 返回 0
3. 非 layerwise 且 `token_len < cache_transfer_granularity` → 返回 0

**三支查询**（L472-498）：

| 模式 | Scheduler 做什么 |
|---|---|
| gva/layerwise(memcache) | `_get_layerwise_gva_hit_tokens` → `batch_get_key_info(all_keys)`，逐 block 判 `ki.size()>0`，所有 rank 的 key 全有效才算命中（L324） |
| layerwise(mooncake，非 gva) | `_get_store_lookup_hit_tokens(include_layers=True)` → `batch_is_exist(query_keys)`，展开到 layer key，`all(exists==1)` 命中、`any(exists==0)` 停（L268） |
| 非 layerwise | 懒建 `LookupKeyClient`，`client.lookup(token_len, block_hashes, group_ids, hbm_hit_tokens)` → **走 ZMQ 问 worker**（L491-498） |

查完统一：命中数回落（eagle 减 `lcm_block_size`、等于全长减 1）→ `need_to_allocate = hit - num_computed_tokens` → 写 `load_specs[req_id] = LoadSpec(...)` → 返回 `(need_to_allocate, load_async and not use_layerwise)`。

### 3.2 Worker 侧（只在非 layerwise 出现）

**LookupKeyServer 何时起**：connector L124-125——`role=worker 且 rank==0 且 !use_layerwise`。ZMQ REP socket 常驻线程（L349-373），每收一帧调 `pool_worker.lookup_scheduler(...)`，**结果打包 4 字节回发**（L369）。

**worker 查询 `lookup_scheduler`**（pool_worker.py L2390）：
1. 先试 `_lookup_with_coordinator(...)`（coordinator 路径，见第六部分 (a)）
2. fallback：逐 group `_build_lookup_keys` → `_expand_lookup_keys_by_rank` → `m_store.exists(multi_tp_keys)` 查 DRAM 池
3. 按 `group_uses_align_state` 选连续/断续命中位置计算，逐 group 取最小，`_max_intersection_hit_position` 求多 group 交集
4. 返回最终命中 token 数

### 3.3 "(gva/memcache)" 是什么

**layerwise 模式运行在 memcache 后端上，每个按层存的 KV 对象用 GVA（全局虚拟地址）寻址**。`pool_scheduler.py L168`：`self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"`。

- **memcache**：后端之一，用 `memcache_hybrid.DistributedObjectStore`（host 侧分布式对象存储，走 memfabric 交换网络）。
- **gva**：每个存进 store 的 KV 对象分配一个 GVA；`batch_get_key_info`（memcache_backend.py L142）返回的 key_info 带 GVA 和 size（scheduler 用 `ki.size() > 0` 判存的来源）。get/put 也靠 GVA 做 DMA。
- 为什么绑 layerwise：只有 layerwise 把 KV 拆成"每层一个对象"，一个对象一个 GVA；非 layerwise 是整段连续 KV 一个块存，不走单对象 GVA 寻址。

---

## 四、存储模型：HBM/DRAM 两层

AscendStore 有**两层**，别混：

### 4.1 第 1 层：本地 KV 缓存 buffer = NPU HBM

`register_kv_caches` 把 vLLM 分配在 NPU 上的 `kv_caches` tensor 登记给后端（pool_worker.py L762）：

```python
base_addr = cache.data_ptr()             # NPU HBM 地址
self.m_store.register_buffer(ptrs, lengths)   # 登记本地 HBM buffer（put/get 的源/目标）
```

`register_buffer` 登记的是"给后端一块可搬移的内存区"，**不是持久化池本身**。

### 4.2 第 2 层：外部存储池 = host DRAM

真正被查询的池在 host DRAM：mooncake = 每卡贡献的 DRAM 池；memcache = host 侧 `DistributedObjectStore`。证据是 `MmcDirect` 的方向（memcache_backend.py L34-38）：
- `COPY_L2G`（Local→Global）→ `put()`：**HBM → DRAM 池**（L214）
- `COPY_G2L`（Global→Local）→ `get()`：**DRAM 池 → HBM**（L186）

### 4.3 两套地址

| 登记/分配 | 是什么 | 落在哪 |
|---|---|---|
| `register_buffer(ptrs, lengths)` | 本地 kv_cache 缓冲 | **HBM** |
| `batch_alloc(keys, sizes)` → GVA（memcache_backend.py L153）；worker `_allocated_gvas` 缓存（pool_worker.py L359） | 池里的对象 | **DRAM** |

**结论**：`m_store` 不是"HBM 上的 map"——它是外部后端对象，`exists` 查的是 DRAM 池；worker 的 HBM 只是 put/get 的本地侧 buffer。"5G DRAM pooling" 指 mooncake 每卡贡献的 host DRAM 池，与 `register_buffer` 是两类东西。

**后端给 scheduler 的客户端**：memcache `create_scheduler_client`（L100-105）返回 `cls(parallel_config, local_rank=0, init_bm=False)`——注释原文 "The scheduler is a single metadata client... must not initialize memcache storage"；mooncake（L164-166）返回 `contribute_memory=False` 的 master 客户端。两者都是**只读元数据客户端**，能触到 DRAM 池但看不到 worker 的权威状态。

---

## 五、核心问题：什么决定了用不用 ZMQ

**问题**：非 layerwise 和 layerwise 的不同到底为什么决定了用不用 ZMQ？关键是"谁能直连 store"吗？

**回答（结论先行）**：**两种模式的查询回答的问题根本不同——layerwise 的查询是"存在性问题"（这些层对象在不在池里，纯元数据，scheduler 只读客户端即可作答）；非 layerwise 的查询是"能力性问题"（结合本地已有什么、granularity 对齐、group 交集，本 TP 组能以"不重算"的方式服务这个请求的前缀到多远的地方），其计算机器在现行布局下全在 worker。** ZMQ 的存在不是为了解决"够不够得着池"，而是把"需要本地状态参与的复杂计算"收敛到持有状态的 worker 上。

### 5.1 五个先钉死的代码事实

1. `store_scheduler` 在 pool_scheduler.py **L177 无条件创建**——非 layerwise 下 scheduler 也持有 store 客户端，"够不够得着池"不是原因；
2. scheduler 的 `store_scheduler` 与 worker 的 `m_store` 是同一个 Backend 类的实例、连同一个 store（mooncake `create_scheduler_client` 也是 `MooncakeDistributedStore()` 连同一 master，mooncake_backend.py L164-166），"有没有池句柄"不是原因；
3. key 格式两边完全一致（worker `_get_key_prefix`（metadata.py L328-336）与 scheduler `_generate_store_query_keys`（pool_scheduler.py L230-266）生成相同格式，且都展开到所有 rank），"key 不同"不是原因；
4. 查询路径分支（L472-498）**只看 `use_layerwise`**；复用（`layerwise_offload`）只影响 worker 的 `load_start_block`（L1147-1151），从头到尾不参与查询路径选择；
5. LookupKeyServer 只在 `not use_layerwise` 时启动（ascend_store_connector.py L124-125）。

```
                 "这个请求能命中多少？"
                          │
        ┌─────────────────┴─────────────────┐
   layerwise                          非 layerwise
   问："池里这些层对象存在吗？"      问："我这个 TP 组，能以'不重算'的方式
                                        服务这个请求的前缀到多远的地方？"
        │                                   │
   纯元数据问题，只依赖               结果依赖 worker 布局状态：
   ① block_hashes（请求自带）         ① 连续/断续命中 + granularity 对齐
   ② 层号 × rank 展开                  （find_all_*_hit_positions 全是 worker 方法）
   ③ all/any 前缀走查                 ② 多 group 交集（_max_intersection_hit_position）
        │                             ③ HMA 时合并本地 HBM 可用性
        │                                （worker 方法 _lookup_with_coordinator）
        │                             ④ ZMQ 协议只回传 4 字节整数，
        │                                per-block 位图无法回传 → 合并必在应答方
        ▼                                   ▼
   scheduler 拿只读客户端              ZMQ → rank0 worker 用 m_store.exists 查
   直连，省一跳 RPC                     + 在 worker 上算完命中位置再回传一个数
```

### 5.2 非层命中依赖五个 worker 布局独有状态（lookup_scheduler L2390-2479 拆解）

1. **`cache_coordinator` 只在 worker 实例化**（pool_worker.py L321-322，条件 `use_hybrid`）。`lookup_scheduler` 第一步调 worker 方法 `_lookup_with_coordinator`（L2407）——预填充本地块（L2327-2332）、查池（L2358 `m_store.exists`）、合并三件事都是这个**方法**做的（"with coordinator"指借它的 `lookup_mask()` 筛 chunk（L2320）+ 按它的 `lcm_block_size` 粒度记账 `exists` 集合 + 用它的 `find_longest_cache_hit()` 算多 group 最长连续命中（L2376）；I/O 和预填是方法自己的活）。coordinator 是纯计算组件（不碰 store、不持 per-request 状态）。**合并点落在 worker 的接口原因**：connector 接口只把本地命中作为聚合整数传下，per-block 的"本地状态 + 池查询结果"只有在实际查池的一方才同时存在——谁拿着完整位图谁做并集（见第六部分 (a)）。

   **存在条件与使用条件是分离的两条正交轴**：实例化条件是 `use_hybrid = uses_hybrid_kv_cache(scheduler_config, kv_cache_groups)`（L172/L600-602）——由**模型架构**决定（是否存在混合 KV cache group，如 mamba 混合 / eagle 多 group），与传输模式 `use_layerwise` 无关，hybrid 架构配 layerwise 时 coordinator 实例**照样创建**；而使用门槛在 L2309 `if self.cache_coordinator is None or use_layerwise: return None`——layerwise 查询即使看到 coordinator 也不理它，落到 fallback。一句话：**hybrid 架构决定 coordinator 存在，layerwise 传输模式决定它被闲置**。layerwise 闲置它的原因不是"分层了凑合不用"，而是它要解决的问题被消解：coordinator 的全部价值是解决多 group 布局不统一（block_size/对齐方式不同 → 升 LCM 粒度统一记账 → 跨 group 求交），layerwise 每 group 独立查存在性前缀、`hit_tokens = min(hits_per_group)` 即可（L337-383），没有并集合并、没有 granularity 对齐、没有交叉布局计算。另注：coordinator 在 worker 里不止服务 lookup——L322 挂给 `token_database` 后，save/load 路径经 metadata.py L345-356 调它的 `store_mask`/`load_mask`；查/存/载三个用途全部只被非 layerwise 消费。
2. **`token_database` 携带 worker 实际 rank + 真实 buffer 地址**（L303-319 + `set_group_buffers` L805-813）。**查询依赖它的"尺子和刀"，不依赖它的"地图"**——查池用的是它的切块机：`get_block_size`（L2326，把 `hbm_hit_tokens` 换算成块数）+ `process_token_key_strings`（L2344/L2187，一次产出每个 chunk 的 key、start/end 边界）；chunk 边界正是命中位置计算的原料（`find_all_*_hit_positions` 靠 start/end 把"哪些 key 存在"换算成"命中 N 个 token"，含 granularity 对齐）。而它持有的真实 buffer 地址（`group_kv_caches_base_addr` 等）只有传输路径消费（put/get 落点，kv_transfer.py L51/L762/L946），查询路径零引用。key 格式两边一致（scheduler `_generate_store_query_keys` 是独立复刻），所以这条证明的是**机器的摆放位置**（整套切块+命中计算共生在 worker 的 `lookup_scheduler` 入口下），不是信息的不可得性。一句话：**token_database 是"传输必需、查询搭车"——worker 为 put/get 落点必须造这台切块机（真强依赖），查询需要的只是它的纯配置推导（块大小/chunk 枚举/边界，无运行时状态），顺手用而不再另造。**
3. **`group_uses_align_state`**（L619-635 推导，L2446-2453 消费）：初始化时从 `kv_cache_config` 逐 group 推导——**该 group 内是否含 align 模式的 `MambaSpec`**（纯 attention / eagle group 为 False）。它在查询里选"位图 → 命中位置"的换算算法：True 走 `find_all_discontinuous_hit_positions`，False 走 `find_all_continuous_hit_positions`（L2505-2535，两函数唯一差别：miss 时后者 `break`、前者跳过继续扫）。**这是五条里语义依赖最实的一条——flag 选错，答案必错，两个方向都会错**：对 mamba group 用连续算法，第一个天然 null 块（align 模式布局中本就不落池）就截断，命中被错报到接近 0；对 attention group 用断续算法，中间挖洞还继续报命中，但 attention KV 逐 token 增量、缺一段前缀即整体不可用，报出的命中是虚的。即：**它定义了"命中"对该 group 的语义，没有它位图无法换算成正确的命中数**（coordinator 是"有它更高效"，token_database 是"机器在这顺手用"，这条是"答案对不对取决于它"）。最终 `_max_intersection_hit_position`（L2482-2491）对各 group 命中位置集求交集取 max，也以此语义正确性为前提。flag 本身仍是纯配置推导（scheduler 理论上可推），限定注记照旧适用。
4. **命中位置对齐 `cache_transfer_granularity`**（find_all_continuous_hit_positions L2522-2535）：块边界来自 worker 的 `token_database` 基于真实 buffer 排布算出。
5. **多 group 交集算法**（`_max_intersection_hit_position` L2482-2491）：每个 group 依赖各自的 block_size、align_state、buffer 布局。

**限定注记（布局独有，非信息独有）**：vLLM factory 对两种角色都传 kv_cache_config（factory.py L75），connector 在 SCHEDULER 角色下 `assert kv_cache_config is not None`（L115）并全量交给 KVPoolScheduler——据此 scheduler 持有 `use_hybrid`/`kv_cache_group_ids`/`grouped_block_size`/`lcm_block_size`/`cache_transfer_granularity`/`mamba_group_ids`/`use_eagle` 全套 group 结构信息（pool_scheduler.py L76-133）。所以五个状态里**没有一个**在查询路径上是 scheduler 真正拿不到的信息；scheduler 真正没有的只有两样，且都是布局性的：① coordinator **实例**（构造原料 scheduler 全有，但引擎只在 worker 实例化）；② token_database 的 buffer 地址（传输态，查询不消费）。因此非层走 ZMQ 的根因是**结构与历史**，"能力性问题由 worker 回答"描述的是 HMA 主路径的算法复杂度与现行布局，**不是信息上的必然**。

**加强证据（比 factory 传参更直接）**：scheduler 的直连 store 客户端 `store_scheduler` 在 `__init__` **无条件创建**（pool_scheduler.py L177，不看 use_layerwise）——即非 layerwise 模式下 scheduler 手里**一直有现成的直连客户端**，但非层查询分支（L488-498）对它视而不见，现场 new 一个 ZMQ `LookupKeyClient` 去问 worker。为什么不用直连：直连客户端只给原始 per-key 存在位，layerwise 位图到手即答案（前缀走查 + 每 group 取 min）；非层的位图只是原料，后面还有整条命中计算流水线（本地块预填/并集合并、align_state 选连续/断续、granularity 对齐、多 group 交集）——这套机器全在 worker，scheduler 侧一行没有，改走直连意味着把流水线移植到 scheduler（纯重构、零功能收益）。因此最终三层归因（按权重）：① 历史——ZMQ 是原始设计（#5719 在先），layerwise（#10077）进场时选直连，没人回头改造老路径；② 成本——改造是纯重构；③ 自然归属——coordinator 路径要合并本地 per-block 状态，持有 HBM 的 worker 本来就是这套计算的自然主人。连"scheduler 够不着池"都不成立——ZMQ 纯粹是历史路径 + 计算机器摆放位置使然。

**再追问一层：流水线驻留 worker 是被"本地命中"拴住的吗？——不是，③"自然归属"也站不住**。非层流水线拆开看，九分之八不碰本地状态：构 key（静态）、查池（远端）、align_state 选算法（静态）、granularity 对齐（静态）、多 group 交集（静态）——唯一真需要本地动态信息的是 coordinator 路径的本地块预填/并集合并（L2327-2332）。两个直接证据：**(i) fallback 路径零本地合并也在 worker**——L2417-2464 全程不消费 `hbm_hit_tokens`、不做任何本地合并，纯构 key + 查池 + 静态配置换算，若"本地查询"是流水线驻留 worker 的原因，这条路径就该允许 scheduler 直查，但它照样走 ZMQ；**(ii) scheduler 侧直查代码里有"非层直查"的死分支**——L278/L341 `query_start_block = 0 if use_layerwise else min(num_computed_tokens // block_size, ...)`，else 分支永远走不到，但它的存在说明曾有人设计过"scheduler 直查非层路径、用 num_computed_tokens 跳过本地已算块"的方案，只是没启用（连综合本地已算量都不构成障碍，草稿都打好了）。所以准确说法：**流水线在 worker 不是"本地查询把它拴在那"，而是"它跟 put/get 机器一起出生在 worker（#5719），后来没人搬"——本地合并只让"不搬"显得合理，不是让"搬不走"**；三层归因里③应降权，①历史的分量相应加重。

### 5.3 对照：layerwise 为什么五个状态都不需要

layerwise 路径（pool_scheduler.py L268-309）只做纯元数据查询 + all/any 前缀走查，不需要 coordinator、worker rank/buffer 状态、align_state。逐条消解：

| worker 布局独有状态 | 在非层服务于什么 | 为何 layerwise 不需要 |
|---|---|---|
| `cache_coordinator` | 并集命中的计算引擎（查池+合并+多 group 连续命中） | **减法模型**：查询从 block 0 起只问池自己的命中总数（L278/L341 `query_start_block = 0 if use_layerwise`），本地状态在消费端（`load_start_block`）用，不在查询端合并 |
| `token_database`（rank/buffer 地址） | key 按 rank 拓扑展开；get 落回真实 buffer | key 是层×rank 逻辑粒度（L322 `model@[group@]block_hash@rank`），由 block_hashes + 层号 × rank 展开即可构造，与 buffer 物理排布脱钩 |
| `group_uses_align_state` | 连续/断续命中位置算法（块物理相邻性） | 逐层对象非 0 即 1（L301-305/L369-372），没有"物理连续性"概念；连续前缀只是顺序遍历 |
| granularity 对齐 | 命中位置对齐真实 buffer 块边界（get 搬移落点） | 对象即传输粒度，天然对齐；查询只回"命中多少 token" |
| 多 group 交集 | 按各 group 布局交叉求交集 | 每 group 独立查存在性连续前缀，`hit_tokens = min(hits_per_group)`（L337-383），scheduler 自己就能算 |

**一句话收束**：layerwise 把"这个请求能命中多少"这个问题本身问小了——从"结合本地/排布/多组交叉，我能以'不重算'的方式服务这个请求的前缀到多远"（能力性）降成"这些逻辑层 key 在不在池里"（存在性，block_hashes 就够）。决定直连/ZMQ 的不是"worker 手里有什么"，而是 **layerwise 把查询降维成了 scheduler 够得着的存在性问题**。

### 5.4 诚实注记（重要限定，含 git 实证）

严格说，非 layerwise 的 fallback 路径（无 coordinator 时）做的是纯池存在性查询，scheduler 理论上也能直连做。它仍走 ZMQ 的原因有两层：

1. **共用入口**：fallback 与 coordinator 共用 `lookup_scheduler` 这一个 worker 方法（L2390），granularity 命中位置计算和 group 交集全是 worker 方法，没人单独把 fallback 搬到 scheduler 侧；
2. **历史顺序**（git 证据，`git log -S` 核验）：`LookupKeyServer`（ZMQ 查询链路）最早出现于 `295018ec0`（PR #5719，distributed 模块重构，先于 layerwise 存在）；`create_scheduler_client` / `batch_get_key_info`（scheduler 直连查询）最早出现于 `5e3907448`（PR #10077 "Support Layerwise KV Pooling"，后经 #11021 revert、#11444 以 memcache 后端重新合入）。即 **ZMQ 是非 layerwise 的原始设计，scheduler 直连客户端正是随 layerwise 特性一起引入的**。

最准确的表述：**结构上是"layerwise 新路径直连、非 layerwise 旧路径保持 ZMQ"（git 实证）；语义上非 layerwise 的主路径（HMA）在现行布局下只有 worker 能答（布局独有，非信息独有）。**

（补充：直连/ZMQ 的分界是 `use_layerwise` 而非后端——layerwise(mooncake) 也走 scheduler 直连；后端只决定 layerwise 内部走 gva(memcache) 还是 store_lookup(mooncake) 两种直连风格。）

### 5.5 终局结论：非层命中为什么至今走 ZMQ

**一句话定稿**：非层命中走 ZMQ 是历史遗留的现行实现，**不是技术必然**——没有任何信息或能力上的障碍阻止 scheduler 直连查询；它至今没改，只是因为没有收益驱动。

每一个可能的"必然性障碍"都被代码证据逐一排除：

| 曾经的可能障碍 | 排除证据 |
|---|---|
| scheduler 不知道 group 配置？ | factory 传全量 kv_cache_config（factory.py L75），五状态的推导原料 scheduler 全有 |
| scheduler 连不上池？ | `store_scheduler` 直连客户端在 `__init__` **无条件创建**（pool_scheduler.py L177），非层模式下就在手里，只是查询分支不用（L488-498） |
| 本地命中信息只有 worker 有？ | `hbm_hit_tokens` 本来就是 scheduler 传过去的（L497）；per-block 细节可从 block_hashes 重构（worker 恰好就是这么做的） |
| 命中计算离不开 worker 机器？ | fallback 路径零本地合并（L2417-2464）；L278/L341 死分支证明"非层直查 + 跳过本地块"的草稿都写过 |
| ZMQ 有性能必要性？ | ipc 一跳 + 4 字节应答（L369/L1040-1041），查询是低频调度路径非热路径；真正延迟大头 `m_store.exists` 的网络往返与走谁无关 |

**至今未改的稳态三条件**：① 路是现成的、没坏；② 改直连收益为零（省的只是 ipc 一跳）而风险为正（命中流水线两份实现的一致性风险——align_state 语义、granularity、交集算法任何一处漂移就是错命中）；③ 唯一"直连"的先例 layerwise（#10077）不是有人回头改老路，而是新需求从零写的新路径（新 key 格式、新存在性语义），没有搬迁成本可比。

**因果裁决**："依赖五个 worker 独有状态"回答的是"为什么 ZMQ 的应答方**有能力**算"（机器恰好长在那），回答不了"为什么**必须**是它"——五个状态证明了"机器在哪"，不能证明"必须在哪"。此前"能力性问题需要 worker 回答"的叙事，正是把"现行布局下恰好由 worker 回答"（事实）包装成了"只有 worker 能回答"（因果倒置）。可以因为历史原因不改，但不能说必须 ZMQ 和 worker 通信才能查询。

---

## 六、本地 HBM 命中的命运（并集 / 复用 / 减法）

三个子问题：(a) 非 layerwise 怎么用本地命中；(b) layerwise 为什么（有时）要无视本地缓存全量重载；(c) 为什么查询从 block 0 起。

### (a) 非 layerwise：并集命中模型

不是"把本地 HBM 当作池的一部分"（存储上本地块不在池里，仍由 scheduler 认领），而是**命中数按"本地 HBM 前缀块 ∪ 外部池块"的并集算最长连续前缀**。

**先澄清一个容易错的推断**：本地有某个块，**推不出**池对象存在（存池在请求结束后才发生、consumer 只读不写、池侧可淘汰、短请求可跳过保存）。worker 把本地块标记进 `exists` 用的不是"本地有⇒池有"，而是"本地有⇒**这块不需要池提供**"——集合语义是**可用性（无需重算即可获得）**，不是池内容。配套证据：load 路径 `mask_num = vllm_cached_tokens // group_block_size * group_size`（pool_worker.py L900）——本地部分连 load 任务都不生成，两类块各走各的。

**本地命中由谁查**：vLLM scheduler 的 `kv_cache_manager.get_computed_blocks_for_connector`（查完还要认领块，这只有 scheduler 能做）——**worker 完全没有重查本地**，本地结果是 scheduler 查好后作为参数传进来的。证据链五步：

1. vllm scheduler 查本地（scheduler.py L757）→ 得 `block_aligned_local`；
2. 作为参数传入（pool_scheduler.py L493-498）：`client.lookup(..., hbm_hit_tokens=num_computed_tokens)`；
3. worker 只做"标记"不查询（pool_worker.py L2327-2332）：把前 `hbm_hit_tokens // group_block_size` 个块直接写进 `exists` 集合（不查任何 store）；L2333 算 `lookup_start`，配合 metadata.py L517-518 `if start_idx < mask_num: continue`——本地前缀连外部 store 查询都跳过；
4. 在并集上算连续命中（L2376-2381）：`find_longest_cache_hit(block_hashes, token_len, ExternalCachedBlockPool(hash_block_size, exists), ...)`——类名易误导，它包的是"**可用块视图**"（本地信任块 + store 验证块），复用 vLLM 自己的命中查找算法，只换后端；
5. worker 返回并集总数，scheduler 换算增量（L513-516 + L550）：`need_to_allocate = 总数 - 本地`。LoadSpec 同时存 `vllm_cached_tokens`（本地部分）与 `kvpool_cached_tokens`（**并集总数**）。

**为什么必须合并、不能 scheduler 自己 `max(local, external)`**：连续性拼不出来。例：本地有 b0..b4，外部因淘汰剩 b0..b2 + b5..b10（b3-b4 是洞）→ 并集连续 11 块，b5..b10 从池里 load；若只回 external 自身连续命中 3 块，`ext(3) < local(5)` → `need_to_allocate = 0`，总命中停在 5 块，b5..b10 全部重算。**协议硬约束**：ZMQ 协议只回传一个 4 字节整数——server `result.to_bytes(4, "big")`（ascend_store_connector.py L369），client `int.from_bytes(resp, "big")`（pool_scheduler.py L1040-1041），per-block 外部存在位图根本无法回传。**谁拿着完整位图，谁就得做并集连续性计算**，合并必然落在应答方 worker。

**重要限定：并集合并是 coordinator 路径专属**。`lookup_scheduler` 先试 `_lookup_with_coordinator`，返回 None 时（无 coordinator / group 集合不完整，L2309-2312）fallback 到纯池查询：L2417-2464 对全部块直接 `m_store.exists`，`hbm_hit_tokens` 参数完全没用——**纯池命中，不含本地**。保守方向正确（少报只是多重算，多报会 load 失败）：

| 路径 | exists 集合的语义 | 本地块怎么处理 |
|---|---|---|
| coordinator 路径（HMA/hybrid 开） | "无需重算即可用"：本地前缀（信任，不查）∪ store 验证块 | 留 HBM，load 从 `vllm_cached_tokens` 后开始 |
| fallback 路径 | 纯池存在性 | 不并入，命中只算池里的 |

**分工**：vllm scheduler = 真查本地 + 认领块；worker = 不重查本地，标记本地结果 + 查外部池 + 算并集最长连续前缀；pool_scheduler = 总数−本地=增量。两个查询不冗余：一个回答"本地有哪些块且认领它们"，一个回答"本地前缀如何与外部连续衔接"。

### (b) layerwise：开了 buffer 复用才重载，本地数据物理上"没了"

**buffer 复用是什么**：把多个 transformer 层的 KV cache 映射到同一块物理 HBM buffer 轮流使用。由 extra_config 的 `layerwise_num_shared_buffers` 控制（layerwise_cache_layout.py L17）；默认不传 = `num_layers`（每层独占，即没开复用）。61 层模型配 8 个共享 buffer 时按 `range(slot, len(reused_layers), num_shared_buffers)` 切条带：slot 1 服务层 1/9/17/.../57，KV tensor 从 61 块并成 9 块（含层 0 独立 slot），显存约降为 1/6.8。物理落地是改写 vLLM 的 `KVCacheTensor.shared_by`（`apply_layerwise_kv_cache_plan` L279-351）。约束：同 slot 的层 cache spec 必须一致（L326-333），仅支持 attention 类 spec（L256-267）；`independent_layers` 默认 `[0]`。

分水岭在 pool_worker.py L1147-1151：

```python
load_start_block = (
    request.load_spec.vllm_cached_tokens // block_size
    if not self.layerwise_offload or layer_id in self.independent_layers
    else 0    # ← 从第 0 块重载，本地前缀不信任
)
```

**三种变体对照**：

| 变体 | HBM 长期 KV | prefix cache 本地命中 | 池的角色 |
|---|---|---|---|
| 非 layerwise | 全量常驻 | 有效（并集合并，见 (a)） | 跨实例副本 |
| layerwise 无复用（默认） | **全量常驻（驻留模型）** | **有效（load 从本地命中后开始）** | 跨实例副本 |
| layerwise 有复用 | 只有旋转窗口（流过模型） | 对共享层失效（名义命中、数据已覆写） | **唯一全量副本** |

**为什么开复用就必须重载**：
1. **合并 tensor**：层 9 前向时覆写层 1 的 slot，层 10 覆写层 2 的……前向跑完，每个 slot 只剩"最后写它的那层"的数据；
2. **hash 与数据脱节**：prefix cache 的 block hash 仍命中（这块当初全层算完过），但共享层数据已被覆写——这就是注释 "per-layer data that may not be in HBM" 的字面意思；
3. **轮转协议**：`prefetch_layer_map` 定义 层N → 层N-8（同 slot 前任）；加载层 N 的任务带 `wait_for_save_layer=层N-8`（L1725）——必须先等前任层的 KV 存进池才允许覆写该 slot；`save_kv_layer` 每层算完立刻异步存池（L1780-1784）。

**复用模式的隐藏代价：decode 每步也在轮转**。`wait_for_layer_load` 和 `save_kv_layer` 是每个 forward step 每层都被调的钩子，不只 prefill——decode step t 的每个共享层都要走"load 全部历史 KV → attention → save 全量回池"循环。代价 = 池读写流量按层×步放大，收益 = HBM KV 占用从 61 份降到 9 份。本质：**拿 HBM 容量换池容量**。

**为什么非 layerwise 配不上复用**：非层是驻留模型——KV 生命周期 = 请求生命周期，存池在请求结束后批量做（`wait_for_save` L1800-1818，不在关键路径）。复用在它下面 ① 物理不可能（attention 要读全部历史层 KV）；② prefix cache 体系崩溃（block hash 语义 = 所有层算完）；③ 轮转协议无意义；④ 收益不存在（其他请求的 KV 也需各自常驻）。复用的本质是"用轮转+池补偿驻留丢失"，是 layerwise 传输模型天然携带的配套机制。

### (c) 为什么查询一律从 block 0 起——减法模型 vs 合并模型

两处查询函数有同样注释（pool_scheduler.py L276-277 / L330-331）："In layerwise mode, always query from block 0 because the remote pool stores per-layer data that may not match local prefix cache"。四条理由：

1. **两个世界的独立性**：layerwise 池里一个块 hash 对应 61 个独立层对象，"本地已算过这块"对它们的存在性零信息量（别人存的、可能部分层、可能被淘汰），唯一可靠的判定方式就是逐 key 查池。
2. **返回数语义——减法模型 vs 合并模型（最核心）**：layerwise 直连查询返回的是**池自己的总命中**（`num_hit_blocks = query_start_block + num_queried_hit_blocks`，L308-309/L374），scheduler 侧做减法（L513-516）。coordinator 路径是合并模型：查+合一起返回总数。**合并要求合并点同时持有本地 per-block 状态和池 per-block 状态**——worker 有，scheduler 没有（connector 接口只给一个聚合整数）。所以 layerwise 直连只能选减法模型，被减数必须是池从 0 起的总数。
3. **复用场景下"跳过本地"直接是错的**：`layerwise_offload=True` 时本地块的共享层数据已被覆写，这些块恰恰必须从池重载。
4. **单值驱动不了 per-group 跳过**：layerwise 逐 group 算再取 min（L337-383），混合架构下各 group 本地命中可能发散，从 0 查天然 group-safe。

**补充：非 layerwise 也不是"默认跳过本地"**。fallback 路径（L2417-2464）调 `_build_lookup_keys` 没传 `mask_num`，同样从块 0 全查，`hbm_hit_tokens` 参数没用上。"跳过本地"是 coordinator（HMA）路径独有的优化（`lookup_start` + `mask_num`）；两处三元表达式（`0 if use_layerwise else min(...)`）的 else 分支接近死代码——这两个函数当前只在 layerwise 分支被调用。准确图景：**"合并 + 跳过本地"是 coordinator 独有的优化路径；其余所有路径（layerwise 直连、非 layerwise fallback）都从 0 查、走减法模型。**

### (d) 终极对照表（两条独立的轴）

容易晕的根源：**两条互不相干的轴被混在一起**——

```
轴1：传输模式（use_layerwise？）
     决定：① 查询路径（直连/ZMQ） ② 池的粒度（61个层对象 / 1个块聚合对象）
轴2：buffer 复用（layerwise_num_shared_buffers < num_layers？）
     决定：本地 HBM 的 KV 是"长期驻留"还是"旋转窗口（用完即走）"
     —— 只对 layerwise 有意义，非 layerwise 没有复用概念
```

| | 非 layerwise | layerwise 无复用（默认） | layerwise 有复用 |
|---|---|---|---|
| **谁查池** | ZMQ→worker | scheduler 直连 | scheduler 直连 |
| **池粒度** | 1 块 = 1 个聚合对象 | 1 块 = 61 个层对象 | 同左 |
| **查询起点** | coordinator:跳本地 / fallback:从0 | **从 0** | **从 0** |
| **本地 HBM** | 全量驻留 | **全量驻留** | 只有旋转窗口 |
| **本地命中用吗** | 用（并集合并） | 用（load 从本地命中后开始） | 不用（全量重载） |
| **池的角色** | 跨实例副本 | 跨实例副本 | **唯一全量副本** |

三个最容易混的点：
1. **"查询起点"和"本地信不信"是两件事**——从 0 查（减法模型）说的是怎么算命中数；信不信本地说的是 load 从哪开始。无复用 layerwise 从 0 查但信本地；有复用从 0 查且不信本地。
2. **"本地有"永远推不出"池有"**——本地的价值是"这块不需要从池拿"（可用性），不是"池里也有"（池内容）。
3. **复用和 layerwise 不是绑定的**——layerwise 默认也可以不复用（按层传输 + 本地驻留），复用只是可选加成，用 HBM 容量换池依赖。

**最终答案：为什么非 layerwise 走 ZMQ→worker，而 layerwise（无论复用与否）都直连？**（见第五部分，含 git 实证）

- 决定查询路径的**只有 `use_layerwise` 这一个变量**（`layerwise_offload` 在查询分支里根本不出现，复用只决定查询结果怎么被消费）；
- **layerwise 的查询是存在性问题**：keys 由 block_hashes + 层号 × rank 展开即可构造，答案非 0 即 1，scheduler 的只读元数据客户端自己就能查，直连省一跳 RPC；
- **非 layerwise 的查询是能力性问题**：其计算机器在现行布局下全部实现为 worker 方法，且 ZMQ 协议只回传 4 字节整数，合并计算必然落在应答方 worker（布局性原因 + 历史性原因，非信息必然）。

---

> 有待补充/确认：
> - vllm 上游 `SimpleConnectorBase_V1` 等实现的入口