# 08. 查询路径为什么这样设计：ZMQ、直连、减法与合并

源码基线：

- vLLM Ascend：`d85e6714a09bef4d9de6b8c05e9425183d46ba23`
- vLLM：`58d3918e3ea0a544ffedadad2ba84559e9c51d8f`

源码位置：

- `ascend_store/pool_scheduler.py`（查询入口与三支分流）
- `ascend_store/pool_worker.py`（`lookup_scheduler` 与 ZMQ 服务端）
- `ascend_store/ascend_store_connector.py`（`LookupKeyServer` 启动条件）
- `ascend_store/coordinator.py`（hybrid 命中合并）
- `ascend_store/layerwise_cache_layout.py`（buffer 复用）

本文回答的问题是读码时最容易困惑的一组"为什么"：为什么非 layerwise 的命中查询要绕道 ZMQ 让 worker 代查，而 layerwise 由 scheduler 直连？为什么 layerwise 查询一律从 block 0 开始、不利用本地 HBM 命中？buffer 复用开了之后本地数据为什么"不可信"？这些不是孤立的设计选择，而是同一组约束（问题语义、状态摆放、协议能力）推导出的结果。

本文源自读码笔记第五、六部分的深度分析，行号均为核验时的准确位置。

---

## 0. 本文要回答的问题

```text
1. 决定查询走 ZMQ 还是直连的变量到底是哪个？
2. 为什么"scheduler 能不能直连 store"不是真正的原因？
3. 两条路径回答的问题在语义上有什么本质区别？
4. layerwise 为什么从 block 0 查起？减法模型和合并模型差别在哪？
5. cache_coordinator 什么时候存在、什么时候被闲置？为什么？
6. buffer 复用如何让本地 HBM 命中失效？轮转协议是什么？
7. 传输模式和复用是几条独立的轴？三个变体各自的画像是什么？
```

---

## 1. 一句话回答

两种模式的查询回答的问题根本不同——layerwise 的查询是"存在性问题"（这些层对象在不在池里，纯元数据，scheduler 只读客户端即可作答）；非 layerwise 的查询是"能力性问题"（结合本地已有什么、granularity 对齐、group 交集，本 TP 组能以"不重算"的方式服务这个请求的前缀到多远），其计算机器在现行布局下全在 worker。ZMQ 的存在不是为了解决"够不够得着池"，而是把"需要本地状态参与的复杂计算"收敛到持有状态的 worker 上。

---

## 2. 先钉死的五个代码事实

在解释"为什么"之前，先排除几个似是而非的答案：

1. `store_scheduler` 在 pool_scheduler.py **L177 无条件创建**——非 layerwise 下 scheduler 也持有 store 客户端，"够不够得着池"不是原因；
2. scheduler 的 `store_scheduler` 与 worker 的 `m_store` 是同一个 Backend 类的实例、连同一个 store（mooncake `create_scheduler_client` 也是 `MooncakeDistributedStore()` 连同一 master，mooncake_backend.py L164-166），"有没有池句柄"不是原因；
3. key 格式两边完全一致（worker `_get_key_prefix`（metadata.py L328-336）与 scheduler `_generate_store_query_keys`（pool_scheduler.py L230-266）生成相同格式，且都展开到所有 rank），"key 不同"不是原因；
4. 查询路径分支（L472-498）**只看 `use_layerwise`**；复用（`layerwise_offload`）只影响 worker 的 `load_start_block`（L1147-1151），从头到尾不参与查询路径选择；
5. LookupKeyServer 只在 `not use_layerwise` 时启动（ascend_store_connector.py L124-125）。

---

## 3. 查询路径总览

- **非 layerwise**：Scheduler 通过 ZMQ 把查询委托给 rank0 worker，由 worker 用 `m_store.exists()` 查并合并本地 HBM 命中，回传命中数。
- **layerwise**：Scheduler 用自己的 backend 客户端 `store_scheduler.batch_is_exist / batch_get_key_info` 直连远程 store 查，不经 worker 中转。

```text
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

### 3.1 Scheduler 侧入口

`get_num_new_matched_tokens`（pool_scheduler.py L444）职责 = 选路 + 短路 + 算 LoadSpec，不碰数据。

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

1. 先试 `_lookup_with_coordinator(...)`（coordinator 路径，见第 5 节）
2. fallback：逐 group `_build_lookup_keys` → `_expand_lookup_keys_by_rank` → `m_store.exists(multi_tp_keys)` 查 DRAM 池
3. 按 `group_uses_align_state` 选连续/断续命中位置计算，逐 group 取最小，`_max_intersection_hit_position` 求多 group 交集
4. 返回最终命中 token 数

### 3.3 "(gva/memcache)" 是什么

**layerwise 模式运行在 memcache 后端上，每个按层存的 KV 对象用 GVA（全局虚拟地址）寻址**。`pool_scheduler.py L168`：`self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"`。

- **memcache**：后端之一，用 `memcache_hybrid.DistributedObjectStore`（host 侧分布式对象存储，走 memfabric 交换网络）。
- **gva**：每个存进 store 的 KV 对象分配一个 GVA；`batch_get_key_info`（memcache_backend.py L142）返回的 key_info 带 GVA 和 size（scheduler 用 `ki.size() > 0` 判存的来源）。get/put 也靠 GVA 做 DMA。
- 为什么绑 layerwise：只有 layerwise 把 KV 拆成"每层一个对象"，一个对象一个 GVA；非 layerwise 是整段连续 KV 一个块存，不走单对象 GVA 寻址。存储层次详见 [05](05_transfer_backend_storage.md) 第 1 节。

---

## 4. 非 layerwise 的合并点为什么必然在 worker

### 4.1 非 layerwise 命中依赖的 worker 布局状态

`lookup_scheduler`（L2390-2479）拆解出五个 worker 独有状态：

1. **`cache_coordinator` 只在 worker 实例化**（pool_worker.py L321-322，条件 `use_hybrid`）。
2. **`token_database` 携带 worker 实际 rank + 真实 buffer 地址**（L303-319 + `set_group_buffers` L805-813）。查询依赖它的"尺子和刀"（`get_block_size`、`process_token_key_strings` 的 chunk 切分），不依赖它的"地图"（真实 buffer 地址只有传输路径消费）。
3. **连续/断续命中位置计算**（`find_all_*_hit_positions`）全是 worker 方法。
4. **多 group 交集**（`_max_intersection_hit_position`）是 worker 方法。
5. **HMA 时合并本地 HBM 可用性**（`_lookup_with_coordinator`）是 worker 方法。

### 4.2 为什么合并不可能在 scheduler 做

**connector 接口只把本地命中作为聚合整数传下**（`block_aligned_local`），per-block 的"本地状态 + 池查询结果"只有在实际查池的一方才同时存在——谁拿着完整位图谁做并集。且 ZMQ 协议只回传 4 字节整数，per-block 位图无法回传，合并计算必然落在应答方 worker。

| 路径 | exists 集合的语义 | 本地块怎么处理 |
|---|---|---|
| coordinator 路径（HMA/hybrid 开） | "无需重算即可用"：本地前缀（信任，不查）∪ store 验证块 | 留 HBM，load 从 `vllm_cached_tokens` 后开始 |
| fallback 路径 | 纯池存在性 | 不并入，命中只算池里的 |

**分工**：vllm scheduler = 真查本地 + 认领块；worker = 不重查本地，标记本地结果 + 查外部池 + 算并集最长连续前缀；pool_scheduler = 总数−本地=增量。两个查询不冗余：一个回答"本地有哪些块且认领它们"，一个回答"本地前缀如何与外部连续衔接"。

---

## 5. cache_coordinator：存在与使用是两条正交的轴

**存在条件**：实例化条件是 `use_hybrid = uses_hybrid_kv_cache(scheduler_config, kv_cache_groups)`（L172/L600-602）——由**模型架构**决定（是否存在混合 KV cache group，如 mamba 混合 / eagle 多 group），与传输模式 `use_layerwise` 无关，hybrid 架构配 layerwise 时 coordinator 实例**照样创建**。

**使用条件**：门槛在 L2309 `if self.cache_coordinator is None or use_layerwise: return None`——layerwise 查询即使看到 coordinator 也不理它，落到 fallback。

一句话：**hybrid 架构决定 coordinator 存在，layerwise 传输模式决定它被闲置**。

layerwise 闲置它的原因不是"分层了凑合不用"，而是它要解决的问题被消解：coordinator 的全部价值是解决多 group 布局不统一（block_size/对齐方式不同 → 升 LCM 粒度统一记账 → 跨 group 求交），layerwise 每 group 独立查存在性前缀、`hit_tokens = min(hits_per_group)` 即可（L337-383），没有并集合并、没有 granularity 对齐、没有交叉布局计算。

另注：coordinator 在 worker 里不止服务 lookup——L322 挂给 `token_database` 后，save/load 路径经 metadata.py L345-356 调它的 `store_mask`/`load_mask`；查/存/载三个用途全部只被非 layerwise 消费。

---

## 6. buffer 复用：本地数据物理上"没了"

### 6.1 复用是什么

把多个 transformer 层的 KV cache 映射到同一块物理 HBM buffer 轮流使用。由 extra_config 的 `layerwise_num_shared_buffers` 控制（layerwise_cache_layout.py L17）；默认不传 = `num_layers`（每层独占，即没开复用）。61 层模型配 8 个共享 buffer 时按 `range(slot, len(reused_layers), num_shared_buffers)` 切条带：slot 1 服务层 1/9/17/.../57，KV tensor 从 61 块并成 9 块（含层 0 独立 slot），显存约降为 1/6.8。物理落地是改写 vLLM 的 `KVCacheTensor.shared_by`（`apply_layerwise_kv_cache_plan` L279-351）。约束：同 slot 的层 cache spec 必须一致（L326-333），仅支持 attention 类 spec（L256-267）；`independent_layers` 默认 `[0]`。

### 6.2 分水岭：load 从哪开始

pool_worker.py L1147-1151：

```python
load_start_block = (
    request.load_spec.vllm_cached_tokens // block_size
    if not self.layerwise_offload or layer_id in self.independent_layers
    else 0    # ← 从第 0 块重载，本地前缀不信任
)
```

### 6.3 为什么开复用就必须重载

1. **合并 tensor**：层 9 前向时覆写层 1 的 slot，层 10 覆写层 2 的……前向跑完，每个 slot 只剩"最后写它的那层"的数据；
2. **hash 与数据脱节**：prefix cache 的 block hash 仍命中（这块当初全层算完过），但共享层数据已被覆写——这就是注释 "per-layer data that may not be in HBM" 的字面意思；
3. **轮转协议**：`prefetch_layer_map` 定义 层N → 层N-8（同 slot 前任）；加载层 N 的任务带 `wait_for_save_layer=层N-8`（L1725）——必须先等前任层的 KV 存进池才允许覆写该 slot；`save_kv_layer` 每层算完立刻异步存池（L1780-1784）。

**复用模式的隐藏代价：decode 每步也在轮转**。`wait_for_layer_load` 和 `save_kv_layer` 是每个 forward step 每层都被调的钩子，不只 prefill——decode step t 的每个共享层都要走"load 全部历史 KV → attention → save 全量回池"循环。代价 = 池读写流量按层×步放大，收益 = HBM KV 占用从 61 份降到 9 份。本质：**拿 HBM 容量换池容量**。

**为什么非 layerwise 配不上复用**：非层是驻留模型——KV 生命周期 = 请求生命周期，存池在请求结束后批量做（`wait_for_save` L1800-1818，不在关键路径）。复用在它下面 ① 物理不可能（attention 要读全部历史层 KV）；② prefix cache 体系崩溃（block hash 语义 = 所有层算完）；③ 轮转协议无意义；④ 收益不存在（其他请求的 KV 也需各自常驻）。复用的本质是"用轮转+池补偿驻留丢失"，是 layerwise 传输模型天然携带的配套机制。

---

## 7. 为什么 layerwise 查询一律从 block 0 起

两处查询函数有同样注释（pool_scheduler.py L276-277 / L330-331）："In layerwise mode, always query from block 0 because the remote pool stores per-layer data that may not match local prefix cache"。四条理由：

1. **两个世界的独立性**：layerwise 池里一个块 hash 对应 61 个独立层对象，"本地已算过这块"对它们的存在性零信息量（别人存的、可能部分层、可能被淘汰），唯一可靠的判定方式就是逐 key 查池。
2. **返回数语义——减法模型 vs 合并模型（最核心）**：layerwise 直连查询返回的是**池自己的总命中**（`num_hit_blocks = query_start_block + num_queried_hit_blocks`，L308-309/L374），scheduler 侧做减法（L513-516）。coordinator 路径是合并模型：查+合一起返回总数。**合并要求合并点同时持有本地 per-block 状态和池 per-block 状态**——worker 有，scheduler 没有（connector 接口只给一个聚合整数）。所以 layerwise 直连只能选减法模型，被减数必须是池从 0 起的总数。
3. **复用场景下"跳过本地"直接是错的**：`layerwise_offload=True` 时本地块的共享层数据已被覆写，这些块恰恰必须从池重载。
4. **单值驱动不了 per-group 跳过**：layerwise 逐 group 算再取 min（L337-383），混合架构下各 group 本地命中可能发散，从 0 查天然 group-safe。

**补充：非 layerwise 也不是"默认跳过本地"**。fallback 路径（L2417-2464）调 `_build_lookup_keys` 没传 `mask_num`，同样从块 0 全查，`hbm_hit_tokens` 参数没用上。"跳过本地"是 coordinator（HMA）路径独有的优化（`lookup_start` + `mask_num`）。准确图景：**"合并 + 跳过本地"是 coordinator 独有的优化路径；其余所有路径（layerwise 直连、非 layerwise fallback）都从 0 查、走减法模型。**

---

## 8. 终极对照表：两条独立的轴

容易晕的根源：**两条互不相干的轴被混在一起**——

```text
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

---

## 9. 本文结论

```text
1. 决定查询路径的只有 use_layerwise 这一个变量；复用只决定查询结果怎么被消费。
2. layerwise 查存在性（纯元数据，scheduler 可答）；非 layerwise 查能力（依赖 worker 布局状态）。
3. ZMQ 是"把计算搬到状态旁边"的选择，不是连接性问题。
4. 合并模型要求合并点持有双方 per-block 位图，只有 worker 满足；scheduler 直连只能减法模型。
5. coordinator 存在与否由架构（hybrid）决定，用不用由传输模式（layerwise）决定。
6. 复用让 HBM 变成旋转窗口、池变成唯一全量副本，代价是 decode 每步的层×步池流量。
7. 传输模式和复用是两条正交的轴，三个变体画像见对照表。
```
