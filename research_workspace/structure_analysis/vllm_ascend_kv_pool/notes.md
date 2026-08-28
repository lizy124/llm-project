# AscendStore / KVPool 读码随记（Q&A）

> 随读随记，后续问题按序号继续追加；同一话题的问题归入同一条。

---

## 1. `KVConnectorBase_V1` 到底是谁用的？

**问题：**

vLLM 在 `kv_transfer/kv_connector/v1/base.py` 定义的这套接口，是给谁用的？是不是仅给 AscendStoreConnector？以及——vllm-ascend 本质上是不是依赖 vllm、必须遵循这套签名，否则 vllm 根本不认识它、也不会去调用它？

**回答（结论先行）：**

不是只给 AscendStoreConnector 用的。`KVConnectorBase_V1` 是 vLLM 所有 KV 跨节点/cache 传输后端的通用抽象契约。AscendStoreConnector 只是其中一个实现者。而且 vllm-ascend 确实**单向依赖** vllm，它必须以"满足契约 + 登记注册"的方式被 vLLM 识别和驱动，否则不遵守签名就真的不会被 vllm 调用。

下面分三块讲清楚。

### 1.1 这个接口是通用的，不止 AscendStoreConnector

同一套 `KVConnectorBase_V1` 下，至少有两层实现者：

**vllm-ascend 仓库内（grep 到 8 个实现）：**

```text
AscendStoreConnector               kv_pool/ascend_store/ascend_store_connector.py
UCMConnectorV1                     kv_pool/ucm_connector/connector.py
RecomputeCPUOffloadConnectorV1     kv_pool/recompute_cpu_offload/recompute_cpu_offload_connector.py
SfaRemoteD2HConnector              kv_p2p/sfa_pd_rd2h/connector.py
MooncakeLayerwiseConnector         kv_p2p/mooncake_layerwise_connector.py
MooncakeConnector (hybrid)         kv_p2p/mooncake_hybrid_connector.py
MooncakeConnector                  kv_p2p/mooncake_connector.py
AscendMultiConnector               distributed/kv_transfer/ascend_multi_connector.py
```

**vLLM 上游自己也有**：`SimpleConnectorBase_V1`（含 mock/dummy）、`BasicKVCacheConnector`、`SingleNodeConnector`、`PyramidKVConnector`、`MooncakeEVictionConnector` 等。

它们共享同一套基类、同一套生命周期回调。抽象的意义就是**让上层（Scheduler / ModelRunner）不认识具体后端**：上层永远只调固定那组签名，换成哪个后端零改动。

```text
vLLM Scheduler / ModelRunner
        │  只认 KVConnectorBase_V1 的抽象签名
        ▼
KVConnectorFactory.create_connector(role)
        │  按 kv_connector 配置名实例化
        ├── AscendStoreConnector
        ├── UCMConnectorV1
        ├── MooncakeConnector
        └── ...（其他后端）
```

### 1.2 依赖方向是单向的：vllm-ascend → vllm

`ascend_store_connector.py` 顶部大量 `from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1` 以及各种 `from vllm.xxx import ...`——都是 vllm-ascend 反向 import vllm。vllm 根本不知道 vllm-ascend 里有哪些类，是 vllm-ascend 主动往 vllm 的空工厂里"塞"自己的实现。

### 1.3 为什么"不遵守签名 vllm 就不认识、也不调用"

分两个层面：

**（1）注册层——vllm 通过注册表决定"认不认识你"。**

vllm-ascend 的 `distributed/kv_transfer/__init__.py` 里的 `register_connector()`：

```python
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
KVConnectorFactory.register_connector(
    "AscendStoreConnector",
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
    "AscendStoreConnector",
)
```

触发点：vllm 启动时扫 entry points，`setup.py` 里 `ascend_kv_connector = vllm_ascend:register_connector`，把它注册进 vllm 的 `KVConnectorFactory._registry`。用户 `--kv-connector AscendStoreConnector` 时，factory 才按名字查表实例化。

→ 不注册，vllm 就没有"按这个名字找实现"的入口，不实例化，也不调用。

**（2）调用层——真正的调用者是 vLLM，不是 vllm-ascend。**

即便注册了，vLLM 的 factory 在 `create_connector`（factory.py L42-75）里做的是 **HMA 支持检查**：若 HMA 已启用（`disable_hybrid_kv_cache_manager=False`）而连接器不支持 HMA，直接 `raise ValueError`（L54-60）。（注：factory **没有** `isinstance` 校验，也没有 nop 回退——注册表查不到名字时仅在 `get_connector_class` 里 `raise ValueError`。）随后在 vLLM 自己的 Scheduler / ModelRunner 生命周期里主动调用：

- Scheduler 进程：`get_num_new_matched_tokens → update_state_after_alloc → build_connector_meta → update_connector_output`
- Worker 进程：`register_kv_caches → start_load_kv → wait_for_layer_load / save_kv_layer / wait_for_save → get_finished`

→ 缺方法：`AttributeError` / `NotImplementedError`，推理循环崩；
→ 签名/返回语义不对：vLLM 按基类约定解析，行为错乱。

### 1.4 一句话总结

- vllm-ascend **被动**运行在 vllm 之上，是靠 **entry point + 工厂注册表**动态挂载的插件。
- 要被 vllm 认可为合法 KV connector，必须：① 运行时把名字注册进 `KVConnectorFactory`；② 继承 `KVConnectorBase_V1` 并被当作符合签名的对象来驱动。
- **vLLM 是唯一驱动者和调用者**；vllm-ascend 只负责"满足契约、登记注册、响应调用"。不遵守签名，要么拿不到调用，要么拿到也会在运行时报错。

### 1.5 注册机制细讲：entry point 如何注入 `_registry`

`_registry` 是 `KVConnectorFactory` 上的**类级 dict**（`factory.py L28`）：`dict[str, Callable[[], type[KVConnectorBase]]]`。key = 连接器名字，value = 一个"懒加载闭包"（`importlib` 延迟 import、返回类）。两张查表点：`get_connector_class_by_name`（L78，查表在 L91）和 `get_connector_class`（L96，查表在 L124-125），都是 `_registry[name]` 取 loader 再 `()` 触发加载。

`_registry` 自己不会长内容，键值 100% 由 `register_connector()` 事后填充；entry point 则是"把这个函数送到 vllm 面前、让 vllm 主动调用它"的运输机制。完整链路：

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
- **运行时反过来 import vllm**：vllm 扫描 entry point 是 `importlib.metadata.entry_points(group=...)` 枚举所有已安装包；`plugin.load()` 才真正 import vllm-ascend 并取 `register_connector` 函数。
- `register_connector` 内部遇到同名会先 `_registry.pop(...)` 再重挂——因为 `_registry` 是可覆盖的可变字典，Ascend 要顶替 vllm 上游已占用的名字（`MultiConnector` / `OffloadingConnector` / `SimpleCPUOffloadConnector`）得先弹掉，否则撞名 `raise ValueError`。`AscendStoreConnector` 是上游没有的新名字，直接注册即可。
- 与 `_registry` 关系：**entry point 让注册代码跑起来，`register_connector` 把名字写进 `_registry`**。没前者，注册函数永不被执行，`_registry` 里就没这个名字。

### 1.6 具体调用链：从 vLLM 的 schedule() 一步步到 get_num_new_matched_tokens

以 commit `6e448d0ea9` 的 vllm 源码逐行走（多态点正是 1.3 讲的"继承 + 注册 + 薄转发"）。

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

- **第 1 步 connector 诞生**：`Scheduler.__init__ L140` 配了 `kv_transfer_config` 就 `KVConnectorFactory.create_connector(role=SCHEDULER)` → 工厂查 `_registry` 返回 AscendStoreConnector 类 → 实例化。Ascend 侧 `__init__ L114`：`role==SCHEDULER` → `self.connector_scheduler = KVPoolScheduler(...)`。于是 `Scheduler.connector` 是"持 KVPoolScheduler 的 AscendStoreConnector"。
- **第 2 步进入 waiting 循环**：L439 先调度 running（L485），再到 L687 `while (waiting or skipped_waiting) and token_budget > 0`。
- **第 3 步只在首轮查外部命中**：L745 仅当 `num_computed_tokens == 0`，先取本地 HBM 连续 prefix `get_computed_blocks_for_connector`。
- **第 4 步真正调用**：L769-781，`ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(request, block_aligned_local)`。传的是**按 block 对齐后的本地命中数**（`block_aligned_local = num_new_local_computed_tokens - 尾块`），让"更长的外部命中"接管不足一个 block 的尾巴，避免 CoW 竞争。返回值：`None` → 无法判定，请求弹回 waiting；否则 `num_external_computed_tokens` 为外部还能命中多少。这正对应 `LoadSpec` 里 `vllm_cached_tokens` 与 `kvpool_cached_tokens` 分开。
- **第 5 步多态到达 Ascend**：vLLM 只认 `KVConnectorBase_V1` 签名，`self.connector.get_num_new_matched_tokens` 落到 ascend `L138-140` 一行 `assert self.connector_scheduler is not None; return ...KVPoolScheduler.get_num_new_matched_tokens(...)`，内部再分流到 `_get_store_lookup_hit_tokens` / `_get_layerwise_gva_hit_tokens` 调 backend `batch_is_exist` / `batch_get_key_info`。

**关键点**：不是每个请求每步都调，只有 `num_computed_tokens == 0`（首轮）走外部命中；必须配了 `kv_transfer_config` 使 `self.connector` 非 None，否则走普通 prefix cache。

---

## 2. `AscendStoreConnector` 实现了基类的哪些方法？各自转给谁？

**问题：**

`AscendStoreConnector` 实现了 `KVConnectorBase_V1` 的哪些方法？哪些转给 `KVPoolScheduler`、哪些转给 `KVPoolWorker`？

**回答：**

`AscendStoreConnector` 实现的基类方法分三类：**纯转发给 KVPoolScheduler（scheduler 侧）**、**纯转发给 KVPoolWorker（worker 侧）**、**connector 自身处理**。

前置：`__init__`（L114-125）里 `role==SCHEDULER` 只建 `connector_scheduler`，否则只建 `connector_worker`。所以每个方法开头的 `assert self.connector_scheduler is not None` / `connector_worker is not None` 就是确认"我这个 role 侧存在"。代码里也有两个大注释块 `Scheduler Side Methods`（L127）和 `Worker Side Methods`（L205）与之对应。

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
| `set_external_slot_release_waiter(waiter)` | gva_layerwise 外部 slot 释放回调（条件性，改 gva 才转发，否则 return False）。**注意：这不是 vLLM 基类方法**，是 Ascend 内部扩展，仅被 vllm-ascend 的 `AscendMultiConnector`（ascend_multi_connector.py L51）调用 |

### 2.3 connector 自身处理（不转发，7 个）

| 方法 | 归属 | 说明 |
|---|---|---|
| `requires_piecewise_for_cudagraph` (classmethod) | connector | `use_layerwise` 时强制 PIECEWISE graph |
| `set_xfer_handshake_metadata_pp_aware` | connector | 直接 `pass`，不走 P/D handshake |
| `update_connector_output(output)` | 半转半自 | 前半转 KVPoolScheduler + 后半做 KV event 聚合 |
| `take_events()` | connector | 聚合多 worker 事件后吐给 prefix cache |
| `get_kv_connector_stats()` | connector | 按存在性分流：有 scheduler 走 scheduler，否则 worker |
| `build_kv_connector_stats` (classmethod) | connector | 构造 `AscendStoreKVConnectorStats` |
| `build_prom_metrics` (classmethod) | connector | 构造 `AscendStorePromMetrics` |

### 2.4 小结

- `AscendStoreConnector` 基本是"薄转发层"：把 vLLM 调用的基类方法按 role 落到 `connector_scheduler` / `connector_worker`；真正的实现全在 `KVPoolScheduler` / `KVPoolWorker`。
- 真正由 connector 自己写的横切逻辑只有：event 聚合（`update_connector_output` + `take_events`）、piecewise 声明、P/D handshake 忽略、stats/metrics 构造。
- 对照 1.3 的调用层：vLLM 调这些方法时不知道具体是哪个角色在背后，正是"继承 + 注册 + 薄转发"的多态体现。

---

## 3. 外部 KV pool 命中如何被查询？Scheduler 侧和 Worker 侧各做什么？

**问题：**

外部 KV pool 的命中是怎么被查询的？Scheduler 侧和 Worker 侧各自负责什么？

**回答（结论先行）：**

有两**条路径**，分界是 `use_layerwise`（gva 时绑定 memcache）：

- **非 layerwise**：Scheduler 的 backend 客户端虽能触到 DRAM 池，但看不到 worker 持有的权威池状态，所以**通过 ZMQ 把查询委托给 rank0 worker**，由 worker 用 `m_store.exists()` 查并合并本地 HBM 命中，再 RPC 回命中数（真实原因详见条目 6）。
- **layerwise（gva/memcache）**：Scheduler 用自己持有的 backend 客户端 `store_scheduler.batch_is_exist / batch_get_key_info` **直连远程 store 查**，不经 worker 中转。

### 3.1 Scheduler 侧（KVPoolScheduler）

入口 `get_num_new_matched_tokens`（`pool_scheduler.py L444`），职责 = "选路 + 短路 + 算 LoadSpec"，不碰数据。

**三道短路**（L461/464-470/481-482）：
1. `kv_consumer` 且 `!consumer_is_to_load` → 返回 `(0, False)`
2. 设了 `retention_interval` 且 prompt 过短 → 返回 0
3. 非 layerwise 且 `token_len < cache_transfer_granularity` → 返回 0

**三支查询**（L472-498）：

| 模式 | Scheduler 做什么 |
|---|---|
| gva/layerwise(memcache) | `_get_layerwise_gva_hit_tokens` → `store_scheduler.batch_get_key_info(all_keys)`，逐 block 判 `ki.size()>0`，所有 rank 的 key 全有效才算命中（L324） |
| layerwise(mooncake，非 gva) | `_get_store_lookup_hit_tokens(include_layers=True)` → `store_scheduler.batch_is_exist(query_keys)`，`.to_string()` 展开到 layer key，`all(exists==1)` 命中、`any(exists==0)` 停（L268） |
| 非 layerwise | 懒建 `LookupKeyClient`，`client.lookup(token_len, block_hashes, group_ids, hbm_hit_tokens)` → **走 ZMQ 问 worker**（L491-498） |

查完统一：命中数回落（eagle 减 `lcm_block_size`、等于全长减 1）→ 算 `need_to_allocate = hit - num_computed_tokens` → 写 `load_specs[req_id] = LoadSpec(vllm_cached_tokens, kvpool_cached_tokens, can_load, ...)` → 返回 `(need_to_allocate, load_async and not use_layerwise)`。

### 3.2 Worker 侧（只在非 layerwise 出现）

非 layerwise 的命中计算依赖 worker 独有的权威状态（本地 HBM 居住度、group 对齐模式、buffer 布局等，见条目 6.2），所以 **ZMQ 路径在 worker 侧**跑。

**LookupKeyServer 何时起**：`connector.__init__` L124-125——`role=worker 且 rank==0 且 !use_layerwise`。bind 一个 ZMQ REP socket，跑常驻线程 `process_request`（L349-373），每收一帧调 `pool_worker.lookup_scheduler(...)`，把结果打包 4 字节回发。

**worker 实际查询 `lookup_scheduler`**（`pool_worker.py L2390`）：
1. 先试 `_lookup_with_coordinator(...)`（若配了 coordinator 插件则由它答）
2. 无则逐 group：`_build_lookup_keys` → `_expand_lookup_keys_by_rank` 展开多 TP rank → **`self.m_store.exists(multi_tp_keys)`** 查 DRAM 池（外部后端 store，见条目 3.3/5）
3. 按 `group_uses_align_state` 选**连续/断续**命中位置计算，逐 group 取最小，`_max_intersection_hit_position` 求多 group 交集
4. 返回最终命中 token 数

### 3.3 为什么分两条路（架构成因）

| 维度 | 非 layerwise | layerwise(gva) |
|---|---|---|
| 谁查 | worker（`m_store.exists`） | scheduler（`store_scheduler.batch_get_key_info/exist`） |
| 传输 | ZMQ RPC（scheduler↔rank0 worker） | 后端连接直达 store |
| 命中语义 | 连续 prefix 命中为主 | 每层全有才算 + 支持 layerwise 复用 |
| 查询端性质 | worker 是 DRAM 池的**权威代理**（持 alloc GVA / group 状态 / 连续命中计算） | scheduler 的后端客户端是**只读元数据客户端**，按层对象+GVA 寻址适合直接查（见条目 4/5） |

**小结**：非 layerwise 走 worker、layerwise 走直连，本质不是"能否触到 store"（两边都能触到 DRAM 池），而是**谁是权威代理**：worker 持有 DRAM 池在当前组的权威状态（`_allocated_gvas`、group 状态、连续/断续命中计算、本地 HBM 居住度），scheduler 的 backend 客户端只是只读元数据客户端，所以非 layerwise 收敛到 rank0 worker 用 `m_store` 查（详见条目 4.2 修正、条目 5）。Scheduler 职责 = 选路 + 短路 + 算 LoadSpec；Worker 职责 = 用后端 store 真正算命中。

---

## 4. 澄清"(gva/memcache)"，以及为什么 layerwise 可直连查询

**问题：**

1. 条目 3 里写的 "(gva/memcache)" 是什么意思？
2. 为什么 layerwise 能直连查、非 layerwise 却不能直连查？

**回答：**

### 4.1 "(gva/memcache)" 的含义

简写，展开是：**layerwise 模式运行在 memcache 后端上，每个按层存的 KV 对象用 GVA（Global Virtual Address，全局虚拟地址）寻址**。对应 `pool_scheduler.py L168`：

```python
self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"
```

- **memcache**：Ascend 的一个后端，用 `memcache_hybrid.DistributedObjectStore`（host 侧分布式对象存储，走 memfabric 交换网络），即 `MemcacheBackend`。
- **gva**：memcache/memfabric 的内存寻址模型——每个存进 store 的 KV 对象分配一个全局虚拟地址 GVA。查询走 `batch_get_key_info`（`memcache_backend.py L142`），返回的 `key_info` 带该对象的 GVA 和 size（这就是 scheduler 那支用 `ki.size() > 0` 判存的来源）。后续 get/put 也靠 GVA 做 DMA。
- 为什么绑 layerwise：只有 layerwise 把每个 KV 拆成"每层一个对象"分别存，一个对象对应一个 GVA；memcache 的"按对象寻址 + 取对象信息"天然支持按层命中。非 layerwise 是整段连续 KV 一个块存，不走单对象 GVA 寻址，也就进不了 gva 路径。

### 4.2 为什么 layerwise 能直连查、非 layerwise 不能

根本原因**不是"层不层"，而是存 KV 的 store 能否被 Scheduler 进程当独立客户端来查**。

**memcache** `create_scheduler_client`（L100-105）：
```python
return cls(parallel_config, local_rank=0, init_bm=False)  # scheduler 是纯元数据客户端
```
`init_bm=False`、不碰 NPU HBM，可直接经 memfabric 调 `batch_is_exist / batch_get_key_info`——因为 store 是 host 侧、跨进程地址可达的分布式对象存储。

**mooncake** `create_scheduler_client`（L164-166）：
```python
return cls(parallel_config, contribute_memory=False)  # 只查不贡 memory
```
一个 mooncake master 客户端，走 RDMA/网络查远端元数据，scheduler 也能直连。

**非 layerwise 为什么走 ZMQ（修正版）**：不是 scheduler 读不到某个 HBM。两侧其实都能触到同一个外部 DRAM 池。真正原因是**谁是权威代理**——worker 持有该 DRAM 池在当前组的权威状态（已分配的 GVA `_allocated_gvas`、每 group 的 alloc 记录、连续/断续命中计算所需的 `group_uses_align_state`、以及要参考的本地 HBM 居住度 `hbm_hit_tokens`），而这些 scheduler 的后端客户端看不到全貌；scheduler 的 backend 客户端被设计成**只读元数据客户端**（memcache 注释：`The scheduler is a single metadata client... must not initialize memcache storage`）。所以命中判定收敛到持有权威状态的 rank0 worker 上，scheduler 经 ZMQ 委托。**池本身是 host DRAM，不是 HBM**（详见条目 5 的两层模型）。

| 模式 | KV 持久化在哪 | scheduler 拿到的客户端 | 谁能查 |
|---|---|---|---|
| layerwise/gva(memcache) | host DRAM 分布式对象存储（memfabric），按层对象+GVA | `create_scheduler_client` 给独立元数据客户端（不碰 HBM） | scheduler 直连查 |
| layerwise(mooncake) | 远端 mooncake DRAM 池 | mooncake master 客户端（`contribute_memory=False`） | scheduler 直连查 |
| 非 layerwise | 外部后端 DRAM 池；worker 另持本地 HBM buffer（`.data_ptr()` 注册） | 只读元数据客户端，不看 worker 的权威状态 | worker 用 `m_store` 查 |

**一句话**：layerwise 的 store 是"host/网络地址可达的 DRAM 共享存储"，后端给 scheduler 一个独立可查的只读元数据客户端，按层对象+GVA 寻址适合直连；非 layerwise 的命中依赖 worker 持有的权威池状态（GVA/group/本地 HBM 居住度），scheduler 客户端看不到，所以只能委托持有 `m_store` 的 worker 查。池是在 DRAM 上，并非 HBM（详见条目 5）。

---

## 5. 池到底在 HBM 还是 DRAM——两层模型（修正混淆）

**问题：**

条目 3/4 里说"KV 存在 worker 本地 NPU 的 HBM pool 里 / m_store 是 HBM 上的 map"，可池不是建立在 DRAM 上吗？

**回答（结论）：**

之前那句说法不准确，已更正。AscendStore 有**两层**：

### 5.1 第 1 层：本地 KV 缓存 buffer = NPU HBM

`register_kv_caches` 把 vLLM 分配在 NPU 上的 `kv_caches` tensor 按物理地址登记进后端（`pool_worker.py L762`）：

```python
base_addr = cache.data_ptr()             # NPU HBM 地址
...
self.m_store.register_buffer(ptrs, lengths)   # 把这段 HBM buffer 登记给后端
```

这里的 `ptrs` 来自 HBM 张量 `.data_ptr()`，所以 **`register_buffer` 登记的是本地 HBM buffer（源/目标）**，是"给后端一块可搬移的内存区"，不是持久化池本身。

### 5.2 第 2 层：外部存储池 = host DRAM

真正的持久化/被查询的池是后端在 **host DRAM** 上的存储：mooncake = 每卡贡献的 DRAM 池；memcache = host 侧 `DistributedObjectStore`。

证据是 memcache `MmcDirect` 的方向（`memcache_backend.py L34-38`）：
- `COPY_L2G`（Local→Global）→ `put()`：**HBM → DRAM 池**（L214）
- `COPY_G2L`（Global→Local）→ `get()`：**DRAM 池 → HBM**（L186）

"Local" = HBM 本卡，"Global" = 走 fabric 的 host DRAM 对象存储。

### 5.3 两套地址，别混

| 登记/分配 | 是什么 | 落在哪 |
|---|---|---|
| `register_buffer(ptrs, lengths)`（`kv_caches.data_ptr()`） | 本地 kv_cache 缓冲 | **HBM** |
| `batch_alloc(keys, sizes)` → 返回 GVA（`memcache_backend.py L153`）；worker `_allocated_gvas` 缓存（`pool_worker.py L359`） | 池里的对象 | **DRAM** |

`put` 把 HBM 里的 KV 经 device-sdma 拷到 DRAM 池对象；`get` 反向。所以：
- **`m_store` 不是"HBM 上的 map"**——它是外部后端（mooncake/memcache）对象，索引/数据对应的是 DRAM 池；
- `m_store.exists` 查的是 **DRAM 池**；
- worker 的 HBM 只是被登记为 put/get 的本地侧 buffer。

### 5.4 对命中查询的影响（衔接条目 4.2）

修正后，"为什么非 layerwise 经 ZMQ、layerwise 直连"的理由是**权威代理 vs 只读元数据客户端**，而非"scheduler 读不到 HBM"：

- scheduler 的 backend 客户端（`create_scheduler_client`：memcache `local_rank=0, init_bm=False`；mooncake `contribute_memory=False`）是**只读元数据客户端**，能触到 DRAM 池但看不到 worker 持有的权威状态；
- worker 持有 DRAM 池在当前组的权威状态（`_allocated_gvas`、group alloc 记录、`group_uses_align_state`、本地 HBM 居住度 `hbm_hit_tokens`）；
- 所以非 layerwise 命中收敛到持有 `m_store` 的 rank0 worker；layerwise 因按层对象+GVA 寻址适合只读直连。

**提醒**：以上基于 `commit 6e448d0ea9` 的 vllm + 本地 vllm-ascend 源码。"5G DRAM pooling" 主要指 mooncake 每卡贡献的 host DRAM 池；AscendStore 的 `register_buffer` 与它是两类东西，前者是 HBM buffer 登记，后者才是 DRAM 池。

---

## 6. 到底什么决定了用不用 ZMQ——逐条推翻错误解释（有代码证据）

**问题：**

非 layerwise 和 layerwise 的不同到底为什么决定了用不用 ZMQ？关键是"谁能直连 store"吗？

**回答（结论先行）：**

先把常见错误解释逐条否定，再给真正原因。最新版本质（2026-08-27 复核 L177 后确认）：**两种模式的查询回答的问题根本不同——layerwise 的查询是"存在性问题"（这些层对象在不在池里，纯元数据，scheduler 只读客户端即可作答）；非 layerwise 的查询是"能力性问题"（结合本地已有什么、granularity 对齐、group 交集，本 TP 组能不重算地服务多远），答案由 worker 独有状态参与计算，必须 worker 回答。** ZMQ 的存在不是为了解决"够不够得着池"（scheduler 的 `store_scheduler` 在 L177 无条件创建，非 layerwise 也持有 store 客户端），而是把"需要本地状态参与的复杂计算"收敛到持有状态的 worker 上。

### 6.1 先否定三个错误解释

| 错误说法 | 代码证据 | 结论 |
|---|---|---|
| "scheduler 读不到 HBM / 池" | mooncake `create_scheduler_client` 也是 `MooncakeDistributedStore()` 连同一个 master（mooncake_backend.py L164-166），`exists()` 两边都调 `store.batch_is_exist()`（L186）；且 scheduler 的 `store_scheduler` 在 pool_scheduler.py **L177 是无条件创建的**（非 layerwise 模式也持有 store 客户端） | **错**。scheduler 能触到同一个 DRAM 池，"够不够得着池"从来不是原因 |
| "worker 有池句柄，scheduler 没有" | scheduler 的 `store_scheduler` 和 worker 的 `m_store` 都是同一个 Backend 类的实例，都连同一个 store | **错**。两边都有 store 连接 |
| "key 格式不同" | worker `_get_key_prefix`（metadata.py L328-336）与 scheduler `_generate_store_query_keys`（pool_scheduler.py L230-266）生成完全相同格式的 key，且都展开到所有 rank | **错**。key 不是分水岭 |

### 6.2 非层命中依赖五个 worker 独有状态（`lookup_scheduler` L2390-2479 拆解）

1. **`cache_coordinator` 只有 worker 有**（构造时创建，pool_worker.py L321-322）。`lookup_scheduler` 第一步就调 `_lookup_with_coordinator`（L2407），其中用本地 HBM 命中预填充外部命中集（L2327-2332）：`exists.update(本 worker HBM 里已有的块)`——把"本地 HBM prefix cache 的块"和"外部 DRAM 池的块"**合并计算**。scheduler 不知道自己 HBM 里有什么（它从未注册 buffer），做不了这件事。
2. **`token_database` 携带 worker 实际 rank + 真实 buffer 地址**（L303-319 + `set_group_buffers` L805-813）。scheduler 没有这个。
3. **`group_uses_align_state`**（L2446-2453）：由 worker 实际 KV cache group 配置决定（mamba 混合架构对齐方式不同），决定走"断续命中"还是"连续命中"算法。scheduler 没有这个数组。
4. **命中位置对齐 `cache_transfer_granularity`**（find_all_continuous_hit_positions L2522-2535）：`ends`/`starts` 块边界来自 worker 的 `token_database` 基于真实 buffer 排布算出。
5. **多 group 交集算法**（`_max_intersection_hit_position` L2482-2491）：要求每个 group 的命中位置列表，而每个 group 又依赖各自的 block_size、align_state、buffer 布局。

### 6.3 对照：layerwise 为什么 scheduler 能直查

layerwise 路径（pool_scheduler.py L268-309）只做：

```python
exists_states = self.store_scheduler.batch_is_exist(query_keys)   # 纯元数据查询
for block_keys in query_keys_by_block:
    if all(exists == 1 for exists in block_states):
        num_queried_hit_blocks += 1
    if any(exists == 0 for exists in block_states):
        break
```

不需要 `cache_coordinator`、不需要 worker rank/buffer 状态、不需要 `group_uses_align_state`——因为 layerwise"每层一个独立对象"，命中语义简单："这个 key 存在吗？"，答案非 0 即 1，scheduler 自己就能算连续前缀。

### 6.4 最终结论：两种模式查询回答的问题根本不同（存在性问题 vs 能力性问题）

三个先钉死的代码事实：

1. `store_scheduler` 在 pool_scheduler.py **L177 无条件创建**——非 layerwise 下 scheduler 也持有 store 客户端，"够不够得着池"不是原因；
2. 查询路径分支（L472-498）**只看 `use_layerwise`**；复用（`layerwise_offload`）只影响 worker 的 `load_start_block`（L1147-1151），**从头到尾不参与查询路径选择**；
3. LookupKeyServer 只在 `not use_layerwise` 时启动（ascend_store_connector.py L124-125）。

```
                 "这个请求能命中多少？"
                          │
        ┌─────────────────┴─────────────────┐
   layerwise                          非 layerwise
   问："池里这些层对象存在吗？"      问："我这个 TP 组能不重算地服务多远？"
        │                                   │
   纯元数据问题，只依赖               结果依赖 worker 独有的东西：
   ① block_hashes（请求自带）         ① keys 按 worker 真实 rank 拓扑构造
   ② 层号 × rank 展开                  （池对象就是 worker 用这些 key PUT 的，
   ③ all/any 前缀走查                    scheduler 没有 token_database）
        │                             ② 命中位置按 cache_transfer_granularity
        │                                对齐 + 连续/断续 + 多 group 交集
        │                                （find_all_*_hit_positions 全是 worker 方法）
        │                             ③ HMA 时还要合并本地 HBM 可用性
        │                                （cache_coordinator，worker 独有）
        │                             ④ 最终执行 get 的就是 worker 各 rank
        │                                （它是"能服务什么"的权威）
        ▼                                   ▼
   scheduler 拿只读客户端              ZMQ → rank0 worker 用 m_store.exists 查
   直连，省一跳 RPC                     + 在 worker 上算完命中位置再回传一个数
```

- **layerwise 的查询 = 存在性问题**（"这些 key 在不在池里"），任何人都能回答，scheduler 的只读客户端自己就能查，所以直连；
- **非 layerwise 的查询 = 能力性问题**（"结合本地已有什么、granularity 怎么对齐、多 group 怎么交集，我这个 TP 组能不重算地推进多远"），答案由 worker 独有状态参与计算，所以必须由 worker 回答，scheduler 只能 ZMQ 去问。

一句话：非 layerwise 的命中是"本地 HBM + 外部池"联合判定（coordinator 路径；fallback 为纯池判定，见 6.5(a) 限定），联合状态只有 worker 有；layerwise 的命中是纯外部池元数据查询，scheduler 直连就够。

**诚实注记（避免绝对化）**：严格说，非 layerwise 的 fallback 路径（无 coordinator 时）做的是纯池存在性查询，scheduler 理论上也能直连做。它仍走 ZMQ 的原因是：fallback 与 coordinator 共用 `lookup_scheduler` 这一个入口（都在 worker 上），granularity 命中位置计算和 group 交集全是 worker 方法；再加上 ZMQ 是非 layerwise 的原始设计，scheduler 直连客户端是后来为 layerwise 引入的——没人单独把 fallback 搬到 scheduler 侧。最准确的表述：**结构上是"layerwise 新路径直连、非 layerwise 旧路径保持 ZMQ"；语义上非 layerwise 的主路径（HMA）确实只有 worker 能答。**

（补充：直连/ZMQ 的分界是 `use_layerwise` 而非后端——layerwise(mooncake) 也走 scheduler 直连；后端只决定 layerwise 内部走 gva(memcache) 还是 store_lookup(mooncake) 两种直连风格。）

### 6.5 追问展开：本地 HBM 命中的两种命运（非 layerwise 并集参与 / layerwise 为何重载）

**问题 1：** 非 layerwise 是不是把本地 HBM 命中当池子一部分？本地 HBM 命中在 vLLM scheduler 里不是有专门负责查询的吗？worker 为什么还要管本地命中？

**问题 2：** pool_scheduler.py L526-528 注释说 layerwise 模式即便本地有缓存也要从池里重载。为什么？难道不能从本地吗？

**回答：**

两个问题其实是同一枚硬币的两面：**非 layerwise 把本地命中"并入"外部命中一起算（并集模型）；layerwise(开复用) 把本地前缀"整个作废"从池重载（覆写模型）。** 分开看。

#### (a) 非 layerwise：并集命中模型（已修正"镜像"错误表述）

不是"把本地 HBM 当作池的一部分"（存储上本地块不在池里，仍由 scheduler 认领），而是**命中数按"本地 HBM 前缀块 ∪ 外部池块"的并集算最长连续前缀**。

**先澄清一个容易错的推断**：本地有某个块，**推不出**池对象存在（存池在请求结束后才发生、consumer 实例只读不写、池侧可淘汰、短请求可跳过保存）。worker 把本地块标记进 `exists` 用的**不是**"本地有⇒池有"，而是"本地有⇒**这块不需要池提供**"——集合的语义是**可用性（无需重算即可获得）**，不是池内容：要么已在本地 HBM，要么经 store 验证在池里。配套证据：load 路径 `start_load_kv` 里 `mask_num = vllm_cached_tokens // group_block_size * group_size`（pool_worker.py L900）——本地部分连 load 任务都不生成，两类块各走各的（本地留 HBM 直接用、池里的才传输），合起来恰好等于命中数。

本地 HBM 命中确实由 vLLM scheduler 的 `kv_cache_manager.get_computed_blocks_for_connector` 专门查询（它查完还要认领块，这只有 scheduler 能做）——**worker 完全没有重查本地**，本地结果是由 scheduler 查好后作为参数一路传进来的。证据链五步：

1. vllm scheduler 查本地（scheduler.py L757）→ 得 `block_aligned_local`；
2. 作为参数传入（pool_scheduler.py L493-498）：`client.lookup(..., hbm_hit_tokens=num_computed_tokens)`；
3. worker 只做"标记"不查询（pool_worker.py L2327-2332）：把前 `hbm_hit_tokens // group_block_size` 个块直接写进 `exists` 集合（不查任何 store，语义="无需重算"）；同时 L2333 算 `lookup_start`，配合 metadata.py L517-518 的 `if start_idx < mask_num: continue`——**本地前缀连外部 store 查询都跳过**；
4. 在并集上算连续命中（pool_worker.py L2376-2381）：`find_longest_cache_hit(block_hashes, token_len, ExternalCachedBlockPool(hash_block_size, exists), ...)`——类名易误导，它包的是"**可用块视图**"（本地信任块 + store 验证块），不是"池内容视图"；**复用 vLLM 自己的命中查找算法**，只换后端；
5. worker 返回并集总数，scheduler 侧换算增量（pool_scheduler.py L513-516 + L550）：`need_to_allocate = 总数 - 本地`，按契约返回"beyond what is already computed"。LoadSpec（L533-538）同时存 `vllm_cached_tokens`（本地部分）与 `kvpool_cached_tokens`（**并集总数**）。

**为什么必须合并、不能 scheduler 自己 `max(local, external)`**：连续性拼不出来。例：本地有 b0..b4，外部因淘汰剩 b0..b2 + b5..b10（b3-b4 是洞）→ 并集连续 11 块，b5..b10 从池里 load；若只回 external 自身连续命中 3 块，`ext(3) < local(5)` → `need_to_allocate = 0`，总命中停在 5 块，b5..b10 全部重算。ZMQ 只回传一个数，per-block 外部存在位图只在 worker 查询那一刻存在——**谁拿着完整位图，谁就得做并集连续性计算**，这又回到条目 6 的结论。

**重要限定：并集合并是 coordinator 路径专属**。`lookup_scheduler` 主体（pool_worker.py L2407-2416）先试 `_lookup_with_coordinator`，其返回 None 时（无 cache_coordinator / group 集合不完整，L2309-2312）fallback 到纯池查询：L2417-2464 对全部块直接 `m_store.exists`，`hbm_hit_tokens` 参数完全没用——**纯池命中，不含本地**。保守方向正确（少报只是多重算，多报会 load 失败）：

| 路径 | exists 集合的语义 | 本地块怎么处理 |
|---|---|---|
| coordinator 路径（HMA/hybrid 开） | "无需重算即可用"的虚拟可用集：本地前缀（信任，不查）∪ store 验证块 | 留 HBM，load 从 `vllm_cached_tokens` 后开始 |
| fallback 路径 | 纯池存在性 | 不并入，命中只算池里的 |

**分工**：vllm scheduler 的 kv_cache_manager = 真查本地 + 认领块；worker = 不重查本地，只把本地结果标记进位图 + 查外部池 + 算并集最长连续前缀；pool_scheduler = 总数−本地=增量。两个查询不冗余：一个回答"本地有哪些块且认领它们"，一个回答"本地前缀如何与外部连续衔接"。

#### (b) layerwise：先纠正前提——不是所有 layerwise 都重载，**开了 buffer 复用（`layerwise_num_shared_buffers < num_layers`）才重载**

**buffer 复用是什么**：把多个 transformer 层的 KV cache 映射到同一块物理 HBM buffer 上轮流使用。由 extra_config 的 `layerwise_num_shared_buffers` 控制（layerwise_cache_layout.py L17）；默认不传 = `num_layers`（每层独占，即"没开复用"）。61 层模型配 8 个共享 buffer 时，`storage_indices` 按 `range(slot, len(reused_layers), num_shared_buffers)` 切条带：slot 1 服务层 1/9/17/25/33/41/49/57，slot 2 服务层 2/10/18/...，KV tensor 从 61 块并成 9 块（含层 0 独立 slot），显存约降为 1/6.8。物理落地是改写 vLLM 的 `KVCacheTensor.shared_by`（`apply_layerwise_kv_cache_plan` L279-351 把 61 个单层 tensor 合并成 9 个）。约束：同 slot 的层 cache spec 必须一致（L326-333），且仅支持 attention 类 spec（indexer 等会 raise，L256-267）；`independent_layers` 默认 `[0]`（层 0 独占，可配 `"all"` 或列表）。

分水岭在 pool_worker.py L1147-1151：

```python
load_start_block = (
    request.load_spec.vllm_cached_tokens // block_size
    if not self.layerwise_offload or layer_id in self.independent_layers
    else 0    # ← 从第 0 块重载，本地前缀不信任
)
```

- 没开复用（`layerwise_offload=False`，默认 `num_shared_buffers=num_layers`）：`load_start_block=本地命中块数`，本地照常用、照常跳过；
- 开了复用：共享层从 block 0 重载；只有 `independent_layers`（独占 buffer 的层）仍跳过本地前缀。

**追问：那 layerwise 本地到底存不存长期 KV cache？——两种变体，答案相反（修正"layerwise=流过模型"的过度概括）：**

| 变体 | HBM 长期 KV | prefix cache 本地命中 | 池的角色 |
|---|---|---|---|
| 非 layerwise | 全量常驻 | 有效（并集合并，见 (a)） | 跨实例副本 |
| layerwise 无复用（默认） | **全量常驻（驻留模型）** | **有效（跳过本地前缀）** | 跨实例副本 |
| layerwise 有复用 | 只有旋转窗口（流过模型） | 对共享层失效（名义命中、数据已覆写） | **唯一全量副本** |

无复用变体与"驻留模型"并不矛盾：每层独占 buffer → 块算完常驻 HBM 直到请求结束/淘汰，prefix cache 照常工作，与非 layerwise 的区别只在"传输按层走、存储按层分对象"。而 `force_layerwise_load` 注释里 "may not be in HBM" 的 **"may"**，区分的正是这两种情况——该 flag 对两种变体都置位，但无复用变体里 `_process_load_for_layer_batch` 的 `load_start_block >= full_blocks` 会 `continue`（L1171），实际不建 load 任务，属保守保护；复用变体才真正从 block 0 重载。

**复用模式的隐藏代价：decode 每步也在轮转。** `wait_for_layer_load`（每层设闸）和 `save_kv_layer`（每层算完即存）是**每个 forward step 每层**都被调的钩子，不只 prefill——decode step t 的每个共享层都要走完整的"load 全部历史 KV → attention → save 全量回池"循环（load 前因 slot 已被同条带后层覆写）。代价 = 池读写流量按层×步放大，收益 = HBM KV 占用从 61 份降到 9 份。

**为什么开复用就必须重载：本地数据物理上"没了"。**

1. **合并 tensor**：`apply_layerwise_kv_cache_plan`（layerwise_cache_layout.py L279-351）把多层写进同一个 `KVCacheTensor(shared_by=[层A,层B,...])`。如 61 层模型配 8 个共享 buffer：层 9 前向时**覆写**层 1 的 slot，层 10 覆写层 2 的……前向跑完，每个 slot 只剩"最后写它的那层"的数据。
2. **hash 与数据脱节**：vLLM prefix cache 的 block hash 仍命中（这块当初全层算完过），但共享层的数据已被后续层覆写——这就是注释 "per-layer data that may not be in HBM" 的字面意思。
3. **轮转协议**：`prefetch_layer_map` 定义 `层N → 层N-8`（同 slot 前任）。加载层 N 的任务带 `wait_for_save_layer=层N-8`（L1725）——**必须先等层 N-8 的 KV 存进池，才允许加载覆写该 slot**；`save_kv_layer` 每层算完立刻异步存池（L1780-1784），因为 HBM 不打算留住它。

**为什么非 layerwise 不需要/不能复用**：非 layerwise 是驻留模型——KV 的生命周期 = 请求的生命周期，块算完常驻 HBM 供全周期 attention 复用，存池在请求结束后批量做（`wait_for_save` L1800-1818，不在关键路径）。复用在它下面 ① 物理上不可能（attention 要读全部历史层 KV，叠层后不在 HBM）；② prefix cache 体系崩溃（block hash 语义 = 所有层算完，复用后数据被覆写）；③ 轮转协议无意义（不存在"层 N 等层 N-8 存完"的前向内时序约束）；④ 收益不存在（省下的 HBM 无法转化为更多常驻块，因为其他请求的 KV 也需各自常驻）。**复用的本质是"用轮转+池补偿驻留丢失"，这恰是 layerwise 传输模型天然携带的配套机制，所以只有 layerwise 配得上它。**

**本质是设计目的而非缺陷**：layerwise(buffer 复用) 拿 HBM 容量换池容量——

| 模式 | HBM 里有什么 | 池(DRAM)里有什么 |
|---|---|---|
| 非 layerwise / 无复用 | 每层全量 KV（block 常驻） | 同样数据的副本 |
| layerwise + 复用 | 只有**旋转窗口**（如 8 个 slot） | **唯一的全量每层 KV 归宿** |

vLLM"本地 prefix cache"隐含假设"一个 block 持有并保住所有层数据"；buffer 一轮转，这个假设物理破产——**不是"有本地缓存却不用"，而是本地缓存对应的共享层数据已被覆写、根本不剩**。所以只能逐层 just-in-time 从池里拉回（`wait_for_layer_load` 逐层设闸），`force_layerwise_load = use_layerwise and store_skip_tokens > 0` 为此存在：外部命中哪怕不超过本地，也要建 LoadSpec 走层加载。

**追问：layerwise 查询为什么一律从 block 0 起，而非 layerwise 跳过本地已算部分？——"减法模型" vs "合并模型"（含一个反转）**

两处查询函数有同样注释（pool_scheduler.py L276-277 / L330-331）："In layerwise mode, always query from block 0 because the remote pool stores per-layer data that may not match local prefix cache"。四条理由：

1. **两个世界的独立性**（承接 (a) 的修正）：layerwise 池里一个块 hash 对应 61 个独立层对象，"本地已算过这块"对它们的存在性零信息量（别人存的、可能部分层、可能被淘汰），唯一可靠的判定方式就是逐 key 查池。
2. **返回数的语义不同——减法模型 vs 合并模型（最核心）**。layerwise 直连查询完：`num_hit_blocks = query_start_block + num_queried_hit_blocks`（L308-309/L374），返回的是**池自己的总命中**；随后 scheduler 侧做减法 `need_to_allocate = num_external_hit_tokens - num_computed_tokens`（L513-516）。而 coordinator 路径是**合并模型**：把本地块塞进 `exists` 集合、查+合一起返回总数。**合并要求合并点同时持有"本地 per-block 状态"和"池 per-block 状态"**——worker 有（vLLM 传来的本地命中 + 自己查的池结果），scheduler 没有（connector 接口只给一个 `num_computed_tokens` 聚合整数）。所以 layerwise 直连只能选减法模型，被减数必须是池从 0 起的总数，查询自然从 0 开始。
3. **复用场景下"跳过本地"直接是错的**：`layerwise_offload=True` 时本地块的共享层数据已被覆写，这些块恰恰必须从池重载，池对块 0..local_hit 的回答是刚需。
4. **单值驱动不了 per-group 跳过**：layerwise 命中检查逐 group 算再取 min（L337-383 `hit_tokens = min(hits_per_group)`），混合架构下各 group 本地命中可能发散，拿一个聚合整数裁剪 per-group 查询起点不安全；从 0 查天然 group-safe。

**反转：非 layerwise 其实也不是"默认跳过本地"**。fallback 路径（pool_worker.py L2417-2464）调 `_build_lookup_keys(token_len, ...)` **没传 `mask_num`，同样从块 0 全查**，`hbm_hit_tokens` 参数根本没用上。即："跳过本地"是 **coordinator（HMA 开启）路径独有的优化**（`lookup_start` + `mask_num`）；两处三元表达式（`0 if use_layerwise else min(...)`）的 else 分支**接近死代码**——这两个函数当前只在 layerwise 分支被调用，else 只是防御性保留。准确图景：**"合并 + 跳过本地"是 coordinator 独有的优化路径；其余所有路径（layerwise 直连、非 layerwise fallback）都从 0 查、走减法模型。** layerwise 的特别之处不在"从 0 查"（fallback 也从 0 查），而在它的池是层粒度 + 减法模型是唯一可行选项。

#### (c) 终极对照表（两条独立的轴：传输模式 × buffer 复用）

讨论中容易晕的根源：**两条互不相干的轴被混在一起**——

```
轴1：传输模式（use_layerwise？）
     决定：① 查询路径（直连/ZMQ） ② 池的粒度（61个层对象 / 1个块聚合对象）
轴2：buffer 复用（layerwise_num_shared_buffers < num_layers？）
     决定：本地 HBM 的 KV 是"长期驻留"还是"旋转窗口（用完即走）"
     —— 注意：这条轴只对 layerwise 有意义，非 layerwise 根本没有复用概念
```

轴 1 回答"怎么查"，轴 2 回答"本地数据还在不在"，两者独立。

| | 非 layerwise | layerwise 无复用（默认） | layerwise 有复用 |
|---|---|---|---|
| **谁查池** | ZMQ→worker | scheduler 直连 | scheduler 直连 |
| **池粒度** | 1 块 = 1 个聚合对象 | 1 块 = 61 个层对象 | 同左 |
| **查询起点** | coordinator:跳本地 / fallback:从0 | **从 0** | **从 0** |
| **本地 HBM** | 全量驻留 | **全量驻留** | 只有旋转窗口 |
| **本地命中用吗** | 用（并集合并，见 (a)） | 用（load 从本地命中后开始） | 不用（全量重载） |
| **池的角色** | 跨实例副本 | 跨实例副本 | **唯一全量副本** |

三个最容易混的点：

1. **"查询起点"和"本地信不信"是两件事**——从 0 查（减法模型）说的是怎么算命中数；信不信本地说的是 load 从哪开始。无复用 layerwise 从 0 查但信本地；有复用从 0 查且不信本地。
2. **"本地有"永远推不出"池有"**——本地的价值只是"这块不需要从池拿"（可用性），不是"池里也有"（池内容）。见 (a) 的修正段。
3. **复用和 layerwise 不是绑定的**——layerwise 默认也可以不复用（此时是"按层传输 + 本地驻留"的普通形态），复用只是 layerwise 的可选加成，用 HBM 容量换池依赖。

**最终答案：为什么非 layerwise 必须 ZMQ→worker，而 layerwise（无论复用与否）都直连？**（详见 6.4）

- 决定查询路径的**只有 `use_layerwise` 这一个变量**（L472-498 的分支结构 + L177 的 `store_scheduler` 无条件创建证明"够不够得着池"不是原因；`layerwise_offload` 在查询分支里根本不出现，复用只决定查询结果怎么被消费——`load_start_block` 信不信本地）；
- **layerwise 的查询是存在性问题**："池里这些层对象在不在？"——keys 由请求自带的 block_hashes + 层号 × rank 展开即可构造，答案非 0 即 1，scheduler 的只读元数据客户端自己就能查，直连省一跳 RPC；
- **非 layerwise 的查询是能力性问题**："结合本地已有什么、granularity 怎么对齐、连续/断续怎么算、多 group 怎么交集，我这个 TP 组能不重算地服务多远？"——答案由 worker 独有状态参与计算（真实 rank 拓扑构造的 key、cache_transfer_granularity 命中位置、group 交集、HMA 下合并本地可用性），必须由 worker 回答，所以 ZMQ 委托。

**一句话收束**：

- 非 layerwise：本地命中是并集的一部分，scheduler 查好传入、worker 标记后与外部命中合并计算——**能从本地的从本地，外部只补齐衔接**；
- layerwise 无复用（默认）：仍是驻留模型，本地 KV 照常长存，本地命中照常跳过；
- layerwise 有复用：HBM 被设计成只留一个层窗口，旧层数据已被后续层覆写，**本地根本不剩**，只能从池逐层 just-in-time 重载（decode 每步同样轮转）。

---

> 有待补充/确认：
> - ~~vllm 上游 `factory.py` 的 `create_connector`（isinstance 校验 + nop 回退）具体实现~~ **已核实（2026-08-27）**：`create_connector` 无 isinstance 校验、无 nop 回退；实际做 HMA 支持检查（HMA 启用且不支持 → `raise ValueError`），查不到名字时 `get_connector_class` 直接 `raise ValueError`。已更正条目 1.3。
> - vllm 上游 `SimpleConnectorBase_V1` 等实现的入口