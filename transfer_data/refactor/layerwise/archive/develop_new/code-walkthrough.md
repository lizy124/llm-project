# PR #15367 代码走读 — layerwise 传输路径通用化 + 协议收编为后端 opt-in 能力

> 走读对象：`lizy124/vllm-ascend:refactor_layerwise_part1` 最终代码（head `63be9e03b`）。
> 代码位置：`D:\lzy\project\kv_pool\code\vllm-ascend\`（下文相对路径以此为根，全部位于
> `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`）。
> 本文按**需求 → 实现**组织：§0 总需求 → §1–§8 八个子需求（每个先讲要什么，再讲代码怎么满足）
> → §9 一个请求的完整生命周期 → §10 行为不变怎么保证。
> 相关：验证报告 `test/e2e-report-20260901*.md`、合入论证 `test/merge-readiness-analysis-20260901.md`、
> 脚本与证据 `test/evidence/`。

## 0. 总需求

**layerwise（逐层）KV 传输路径是通用能力——mooncake 后续的 layerwise 也会走同一条路。
但当前实现里，memcache 专属的知识（key 格式、GVA 寻址、后端类型判断）散落在通用层
（worker / scheduler / layout / 传输线程 / connector），后端想换/想加都动不了。**

本 PR 把通用层清空为"只认协议接口"，后端专属知识全部收进 memcache backend，同时：

- key 字符串是**线上 wire 协议**（已部署集群存着旧 key），一个字符都不能变；
- 现有部署行为完全不变（memcache layerwise 照常、mooncake 非 layerwise 照常）。

实现思路一句话：**通用层保留"传输流程"，后端交出"怎么寻址、怎么开门"——两者通过一个
registry 约定的协议接口解耦。**

```
connector(转发) ─ scheduler(hit check) ─ worker(存/取编排) ─ 传输线程(执行)
       └────────────── 全部通用层，只认协议接口 ──────────────┘
                              │ get_layerwise_protocol(backend)
                              ▼
                    backend/__init__.py（registry：谁能做 layerwise）
                              │ 仅 memcache 声明
                              ▼
                    memcache_backend.py（协议宿主：key 格式 + opt-in）
                              │ Backend ABC
                              ▼
                    kv_transfer.py 传输线程（只见 ABC，后端无关）
```

三个贯穿全文的不变量：

1. **通用层零后端知识**：通用层运行路径 grep 不到 memcache/GVA；
2. **单一事实源**：谁支持 layerwise 只由 registry 标记决定；
3. **key 格式单一宿主**：key 拼接只存在于 memcache_backend.py，snapshot 测试锁死。

下面按八个需求逐个讲。

---

## 1. 需求一：后端要能声明"我支持 layerwise"

**问题**：通用层需要一个统一的地方问"这个 backend 能不能做 layerwise"。老实现是硬编码
`isinstance(store, MemcacheBackend)`——每加一个后端都要改通用层，这正是要消除的耦合。

**实现**：[backend/__init__.py](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py#L23-L56)，一个标记 + 一个函数：

```python
backend_map: dict[str, dict[str, Any]] = {
    "mooncake": {"name": "MooncakeBackend", "path": "...mooncake_backend"},
    "memcache": {"name": "MemcacheBackend", "path": "...memcache_backend",
                 "layerwise_protocol": True},      # ← 唯一的 opt-in 声明
    "yuanrong": {...},
}

def get_layerwise_protocol(backend_name: str):
    normalized_name = backend_name.strip().lower()
    backend = backend_map.get(normalized_name, {})
    if not backend.get("layerwise_protocol"):
        return None
    import importlib
    return importlib.import_module(backend["path"])
```

要点：

- **返回的是后端模块本身**，不是独立协议模块。协议函数（§2）就住在后端模块里，resolver
  复用 registry 已有的 `path` 字段——没有第二个模块路径可以和实现漂移。
- **惰性 import**：mooncake 没有标记就永远不会在这里被 import（其模块顶层有重第三方依赖）。
- 未知 backend 名返回 `None` 而非抛错——能力查询和后端工厂是两件事，查不到就当"不支持"。

mooncake 将来做 layerwise 时只需要：registry 加一行 `"layerwise_protocol": True`，
在自己模块实现 §2 的四个同签名函数。通用层零改动。

（细节：`backend_map` 的显式注解 `dict[str, dict[str, Any]]` 是必需品——条目里布尔标记和
路径字符串混居，没有注解 mypy 会把各条目 join 成 `object`，6 处 `.get()`/索引报错。）

## 2. 需求二：key 格式要收进后端，且一个字符都不能变

**问题**：key 字符串（`model@hash@rank` 等）是 memcache 池的**线上 wire 协议**——已部署集群
的池子里存着旧 key，升级后错一个字符，hit 就静默变 miss（无任何报错）。同时 key 构造散在
通用层是后端知识泄漏。要求：key 知识只住 memcache 后端 + 格式逐字节兼容。

**实现**：[memcache_backend.py:42-123](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L42-L123)，模块级四个协议函数：

| 函数 | key 格式 | 用途 |
|---|---|---|
| `make_full_key` | 单组 `model@hash@rank`（兼容 #11585 老格式）；多组 `model@group@hash@rank` | 整块存/取 |
| `make_partial_key` | `model@partial@req@group@block@end_token@rank` | 尾块（非整块部分） |
| `make_hit_check_keys` | full key 按 rank 全展开的列表 | scheduler 命中检查 |
| `extract_layout_config` | —（返回值，见需求四） | 启动期 opt-in 判定 |

```python
def make_full_key(model_name, group_id, block_hash_hex, head_or_tp_rank, num_groups):
    if num_groups > 1:
        return f"{model_name}@{group_id}@{block_hash_hex}@{head_or_tp_rank}"
    else:
        return f"{model_name}@{block_hash_hex}@{head_or_tp_rank}"
```

要点：

- 函数体从原实现**逐字节迁移**，`test_backend.py` snapshot 测试锁死输出；
- 用**模块级函数**而非类（原 `GVAKeyFactory` 是无状态类，工厂无收益）：后端模块即协议宿主，
  `importlib.import_module` 之后直接 `getattr`，约定最简；
- 通用层调用方（§5/§6）只贡献**本地拓扑参数**（model 名、rank、group 数），格式知识零持有——
  这就是"后端专属知识"和"通用编排知识"的精确分界。

## 3. 需求三：系统要能决定走不走 layerwise 路径

**问题**：用户配置 `use_layerwise: true` 只是**意图**，能否成立取决于后端能力
（mooncake 目前不支持）。且历史上这个判断散在 connector（isinstance）——connector 的
flag 拷贝曾被 #14465 误删、#15291 恢复，同一类回归出过两次。要求：判断单源、且长在
数据面。

**实现**：gate 合取公式，在 worker 与 scheduler 各自构造时派生（
[pool_worker.py:161-162](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L161-L162)、
[pool_scheduler.py:168-169](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L168-L169)）：

```python
self.layerwise_protocol = get_layerwise_protocol(self.backend_name)
self.use_layerwise_transfer = self.use_layerwise and self.layerwise_protocol is not None
```

- `use_layerwise`：用户开关，connector 只透传原始值，**不派生任何判断**；
- `layerwise_protocol is not None`：后端能力，registry 决定（需求一）；
- 合取结果 `use_layerwise_transfer` 是**运行时唯一生效的 gate**。

用户开了但后端不支持 → 静默走非 layerwise 通用路径（等同没配），不抛错。真值表
`(memcache, T)→T / (memcache, F)→F / (mooncake, *)→F / (未知, *)→F`，有真值表测试锁死。

gate 在 worker 派生还有个结构收益：传输线程（§7）只在 gate 开时被构造启动——"线程在
不支持的后端上运行"这个状态**从构造上不可达**，所以线程里不需要任何 isinstance 防御。

## 4. 需求四：模型加载前要确定 KV 的物理排布

**问题**：layerwise 传输要求 KV cache 按层连续排布（传输以层为单位），这必须在 vLLM 构建
KV layout 时（模型加载前）就定下来。配置来源有三种：单 AscendStoreConnector、
MultiConnector 里的 AscendStore 子项（可能多个）、其他 connector（无关）。要求：在不知道
任何后端名字的前提下，找到"生效的那个 layerwise 配置"。

**实现**：[layerwise_cache_layout.py:85-113](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py#L85-L113)：

```python
for connector_config in connector_configs:          # 展平三种来源
    if connector_config.get("kv_connector") not in ("AscendStoreConnector", "MooncakeConnectorStoreV1"):
        continue
    extra_config = connector_config.get("kv_connector_extra_config") or {}
    protocol = get_layerwise_protocol(str(extra_config.get("backend", "mooncake")))
    if protocol is None:
        continue                                     # 后端不支持 → 跳过这个子项
    layerwise_config = protocol.extract_layout_config(extra_config)
    if layerwise_config is not None:
        return layerwise_config                      # 找到第一个生效的
return None
```

注意 `extract_layout_config` 是**协议函数**（住 memcache_backend，[行 53](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py#L53)）：
它内部判断 `use_layerwise` 开则返回 extra_config、否则 None。也就是说连"用户开没开 layerwise"
这个语义判断都由协议侧完成——通用层只做"有没有哪个 connector 配置成立"的遍历。
两步能力发现（registry 标记 → 协议确认）保证：backend 不支持或用户没开，layout 层拿到 None，
走普通排布。

## 5. 需求五：请求到来时要判断远端池里有多少可复用（读路径）

**问题**：scheduler 要在每个请求调度前算出"远端池能命中多少 token"（`num_computed_tokens`），
用来扣减 prefill 计算量。难点：① 池里的 key 带 rank 信息，要查全 rank；② layerwise 模式下
远端池和本地 prefix cache 状态可能不一致，不能沿用本地缓存的起点。

**实现**：[pool_scheduler.py:318-380](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L318-L380)，三步：

```python
def _make_layerwise_hit_check_keys(self, group_id, block_hash_hex):
    head_or_tp_ranks = self.tp_size // self.put_step
    return self.layerwise_protocol.make_hit_check_keys(   # 协议函数（需求二）
        self.model_name, group_id, block_hash_hex,
        head_or_tp_ranks, len(self.kv_cache_group_ids))
```

1. **永远从 block 0 查起**：`_get_layerwise_hit_tokens` 不信任本地 prefix cache 状态
   （远端池存的是逐层数据）；
2. **逐 block 全 rank 检查**：对每个 hash block 构造全 rank key 列表（MLA 时同 put_step 组
   的 rank 共写一份，展开数 = `tp_size // put_step`）→ `batch_is_exist` → **全 rank 存在才算
   hit**，遇到第一个 miss 即停止（前缀语义）；
3. `hit_tokens = num_hit_blocks × block_size` 回填 scheduler。

MLA/MHA 的 rank 维度对偶关系在 worker 侧推导（`_init_key_head_config`，[pool_worker.py:210-226](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L210-L226)）：
MLA 时 `num_kv_head=1`，`put_step = tp_size // num_kv_head` 个 rank 共写一份（key 只带组号）；
MHA 时每 rank 独立一份。scheduler 展开 key 的数量与 worker 写 key 的份数由同一公式保证一致
——**两边 key 都由同一协议函数生成，逐字节一致是跨进程命中成立的根基**。

## 6. 需求六：要把 KV 存进池子（写路径）

**问题**：producer 侧在每步 forward 后要把新产生的 KV 块存入池。难点：① 池有淘汰，本地
GVA 缓存会失效；② 同一前缀重复请求不能重存（幂等）；③ 池按 key 寻址但传输走 GVA 直拷，
要先分配好地址。

**实现**：[pool_worker.py](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py)，两层：

**编排层 `start_load_kv`（[行 831](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L831)）**：每步重建
`layer_save_tasks / layer_load_tasks`（按层的任务列表，引用传给传输线程）。每步换新列表而非
复用——防上一步迟来的 clear 吞掉本步新任务、留下脏 buffer。

**地址层 `_alloc_gvas_for_save`（[行 1172-1290](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1172-L1290)）**，五步：

1. 逐 group 构造 candidate_keys——`_make_layerwise_full_key` 薄封装协议函数（worker 只给
   model 名 / rank / group 数）；
2. **驱逐对账** `_refresh_allocated_gvas`：本地 `_allocated_gvas`（key→地址缓存）与池的
   `batch_is_exist` 对账，exists==0（被池淘汰）的剔除本地条目——池会 LRU，本地缓存必须跟；
3. **幂等去重**：从 `save_start_block` 起连续已在 `_allocated_gvas` 的块跳过——同一前缀
   二次请求不重存；
4. 新 key 调 `store.alloc` 拿 GVA，key→GVA 记入 `_allocated_gvas`；
5. 尾块（不满整块）另用 `make_partial_key` 走 partial 通道。

存入的执行在传输线程（§7）：等该层 NPU 写完（`sync_save_events[l]`）→ `store.batch_copy(GVA
列表)` → 写完成标记 key。

## 7. 需求七：传输线程要能被任何后端复用

**问题**：layerwise 传输的执行体（两个线程：LayerSendingThread / LayerRecvingThread）是
mooncake layerwise 将来也要用的通用资产。老实现线程入口有 `assert isinstance(store,
MemcacheBackend)`——mooncake 复用前还得先删这段。要求：线程对后端零认知。

**实现**：[kv_transfer.py](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py)，做减法做到底——线程对 store 的全部调用
（`alloc` / `batch_copy` / `batch_is_exist`…）走 `Backend` ABC，线程入口只有一段注释：

```python
# Layerwise threads only run when the worker-side
# use_layerwise_transfer gate is on; every store call below sits on
# the Backend ABC, so the thread stays backend-agnostic.
```

不需要任何运行时类型防御，因为"线程在不支持的后端上运行"经需求三的 gate 论证**不可达**
（gate 只对有协议的后端开，线程只在 gate 开时构造）。

配套的线程选择在 [pool_worker.py:490-552](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L490-L552) `_start_kv_transfer_threads()`，四象限：

| | can_save（producer/both） | 只收（consumer） |
|---|---|---|
| **use_layerwise_transfer** | `LayerSendingThread`（GVA 直传） | `LayerRecvingThread`（GVA 直收） |
| 非 layerwise | `KeyLayerSendingThread`（key→blob 存） | `KeyLayerRecvingThread` |

注意 gate 关但 `use_layerwise` 开时走 Key*Layer 线程——Key 线程同样逐层，只是传输粒度是
key→blob 而非 GVA 直读。这就是"mooncake 开 use_layerwise 也能跑（走通用 Key 路径）"的
结构保证，四种线程覆盖全部组合。

## 8. 需求八：MultiConnector 场景要能协调 slot 释放（且不再回归）

**问题**：MultiConnector 同时挂多个 connector 时，用一个 composite slot 管理跨 connector 的
KV 接收；AscendStore 的 layerwise 接收线程每收完一层要回调"释放 slot"，否则其它 connector
被卡住。历史上这个回调的 gate 判断放在 connector 侧、依赖 connector 里的一份 flag 拷贝，
#14465 删 worker flag 时连带删掉（MultiConnector 初始化即崩），#15291 补回。要求：这类回归
结构性不可能。

**实现**：三层接力，gate 只在 worker 判一次：

**connector 纯转发**（[ascend_store_connector.py:198-210](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L198-L210)）：

```python
def set_external_slot_release_waiter(self, waiter) -> bool:
    """Pure forwarder: the layerwise transfer gate is evaluated by the worker..."""
    if getattr(self, "connector_worker", None) is None:
        return False
    return self.connector_worker.set_external_slot_release_waiter(waiter)
```

connector 不派生任何 gate（docstring 记录了两次回归史）——flag 拷贝没了，删无可删。

**worker 判 gate + 顺序加固交接**（[pool_worker.py:465-481](../../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L465-L481)）：

```python
def set_external_slot_release_waiter(self, waiter) -> bool:
    if not self.use_layerwise_transfer:
        return False
    self.external_slot_release_waiter = waiter          # ① 先存引用
    if isinstance(self.kv_recv_thread, KVCacheStoreLayerRecvingThread):
        self.kv_recv_thread.external_slot_release_waiter = waiter  # ② 再交接给已运行线程
    return True
```

两步的次序是加固点：接收线程构造时**快照**该字段（`_start_kv_transfer_threads` 传
`external_slot_release_waiter=self.external_slot_release_waiter`），若先交接后存引用，
后构造的线程拿到 None。①② 都做则同时覆盖"先注册后启动"与"先启动后注册"两种时序。

**接收线程回调**（kv_transfer.py，每层收完）：

```python
if self.external_slot_release_waiter is not None:
    self.external_slot_release_waiter(layer_id)   # 通知 MultiConnector 释放 composite slot
```

这是"知识放唯一宿主"的第三次体现（key→backend、gate→worker、协议→registry）。

---

## 9. 一个请求的完整生命周期（把需求四到八串起来）

S2 e2e 场景（同前缀两连发，`hit_tokens=3328` 的产生过程）：

```
[启动期] 需求四:layout 层遍历 connector 配置 → registry 查 memcache 有标记
         → extract_layout_config 确认 use_layerwise 开 → KV 按层连续排布
         worker/scheduler 构造 → 需求三:gate = T ∧ 有协议 = use_layerwise_transfer
         → 需求七:启动 LayerSending/LayerRecvingThread(只依赖 Backend ABC)

请求 1(首发,存入)                          请求 2(同前缀,命中)
────────────────────────                    ─────────────────────────
scheduler: 需求五 hit check                   scheduler: 需求五
  全 rank keys batch_is_exist → 0 hit           num_hit_blocks=26
  → num_computed_tokens=0(全量 prefill)        → hit_tokens=3328, 扣减 prefill
worker: 每步重建 save/load 任务列表            worker: layer_load_tasks 入队
worker: 需求六 _alloc_gvas_for_save            LayerRecivingThread 消费:
  驱逐对账 → 幂等去重 → store.alloc             key 已在 → GVA 直读本地
  → key→GVA 入 _allocated_gvas                 layer_load_finished_events[l].set()
LayerSendingThread:                            需求八: waiter(layer) → MultiConnector
  sync_save_events[l] 等 NPU 写完                 释放 composite slot
  → store.batch_copy(GVA 列表)
```

请求 2 能命中，靠的是 scheduler 查的 key 和请求 1 worker 存的 key **由同一协议函数生成**
（需求二）——这是跨进程命中成立的根基。三个会计独立记账互相印证：scheduler `hit_tokens`、
MetaService `stored_keys`、vLLM `external_prefix_cache_hits_total`，算术一致（26×128=3328）。

## 10. 行为不变怎么保证（读代码时反向验证）

1. **key 逐字节**：四个 make_* 函数体原样迁移，snapshot 测试锁输出；
2. **gate 真值表**：四组合全覆盖测试；
3. **两处有意非等价**（不可达或更安全）：线程无类型断言（gate 论证不可达）、waiter 顺序加固
   （收窄一个旧时序从未触发的窗口）；
4. **测试索引**（tests/ut/distributed/ascend_store/，314 passed）：snapshot（wire 兼容）、
   三方一致性（协议函数 ⇔ registry 标记 ⇔ store override）、gate 真值表、waiter
   gate/handover、线程 fixture 不 `spec=MemcacheBackend`（锁后端无关）；
5. **e2e 双轮夹逼**：返工前后两个 head 三场景同构 PASS，数值敏感性（hit_tokens 3456→3328、
   日志行号漂移与版本吻合）证明两轮跑的是各自版本的活数据——详见 test/ 两份报告。
