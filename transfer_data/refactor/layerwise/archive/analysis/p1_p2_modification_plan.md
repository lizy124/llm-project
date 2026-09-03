# AscendStore Layerwise P1/P2 问题修改方案

> 依据：[requirements_analysis.md](requirements_analysis.md)（问题清单 §3）+ [p0_modification_plan.md](p0_modification_plan.md)（P0 方案，PR1-PR3）
> 范围：P1-4（传输线程收敛）、P1-5（backend 生命周期）、P2-6（零散后端痕迹）
> 原则：与 P0 方案一致——平移式迁移，协议代码逐行搬移不改语义；本方案另含 2 处对需求分析文档的**事实修正**（§0）

## 0. 事实修正（本次细读代码新核实，修正需求分析文档 2 处表述）

1. **修正 2.2 "put/get 同接口不同语义"**：GVA 线程（`KVCacheStoreLayerSendingThread`/`KVCacheStoreLayerRecvingThread`）**根本不走 `put/get`**——save/load 均经 `_batch_copy_with_limits`（kv_transfer.py:426）直达 `self.m_store.store.batch_copy(...)`（kv_transfer.py:468），**直捅 backend 内部属性**。`put/get` 仅被 key 模式线程（非 layerwise :863/:970、key-layerwise :1155/:1256）以 key 语义使用。
   - 影响：P0-3 的接口分层无需处理"语义分离"（不存在该问题）；P1-4 需新增 `batch_copy` 接口方法封堵直捅（见 §1.3）。
2. **修正 §3 P2-6 "check_all_layers_exists 与 memcache key 布局强耦合"**：其调用点（pool_worker.py:2121/:2320）的 gate 是 `use_layerwise`（非 `use_gva_layerwise`），服务 **key-mode layerwise** 路径（三后端通用；GVA 模式 hit-check 走 `batch_get_key_info`，P0 方案已迁 GVAHitChecker）。"每 block num_layers 个连续 key" 布局由通用 key 构造（`_make_sub_key_str` :1859 等）产生。**结论：不迁 backend**，归属 metadata.py（§3）。

另：kv_transfer.py 实际行数 >1601（需求分析 §2.1 记 1484 有误，类分布以本文 §1.2 表为准）。

## 1. P1-4：传输线程类与选择逻辑收敛（THRD）

### 1.1 现状核实

kv_transfer.py 类分布（grep 验证）：

| 行号 | 类/函数 | 模式 | 归属裁定 |
|---|---|---|---|
| :33 | `_circular_shift` | 通用工具（key-mode :1253、worker :918 使用） | **留** |
| :39-303 | `LayerBatchBuilder` | **仅 GVA 线程构造**（:1309/:1452、worker `_build_group_layer_builders` :451） | **迁** |
| :305-597 | `KVTransferThread` | 线程基类（含 `_split_transfer_packets` :393 / `_batch_copy_with_limits` :426，**仅 GVA 线程调用** :1363/:1560） | 基类**留**；两个 GVA 专用方法**迁** |
| :598/:868 | 非 layerwise 收/发线程 | 通用 | 留 |
| :1015/:1169 | key-layerwise 收/发线程 | 通用（key 语义，三后端可用） | 留 |
| :1271-1404 | `KVCacheStoreLayerSendingThread` | **GVA 专属**（`batch_write_finish` :1380、GVA 地址构造） | **迁** |
| :1405-1600 | `KVCacheStoreLayerRecvingThread` | **GVA 专属** | **迁** |
| :1601+ | `record_failed_blocks` | 通用（key-mode :972/:988、worker :932 使用） | 留 |

pool_worker 侧的 GVA 专属代码：`_build_group_layer_builders`（:444-459，仅被 :489/:529 两处 GVA 线程构造调用）。

### 1.2 设计：新建 backend/gva_threads.py

**依赖方向**（无环）：

```
kv_transfer.py（通用线程基类/工具）  ←─ import ─ backend/gva_threads.py（GVA 线程 + LayerBatchBuilder + 工厂）
        ↑                                         └─ import ─ metadata.py（LayerTransferTask/SharedBlockData/LayerBatchReqMeta）
   pool_worker.py ─ import ─ backend/gva_threads.py（工厂调用）
```

kv_transfer.py 不 import gva_threads → 无循环。`SharedBlockData`/`LayerBatchReqMeta` 已在 metadata.py（:1110/:1091），无需移动。

**迁入内容**（自 kv_transfer.py 平移）：

| 内容 | 原位置 | 迁移要点 |
|---|---|---|
| `LayerBatchBuilder` | :39-303 | 整类平移（`build_shared`/`build_addrs` 及 numpy buffer 复用逻辑） |
| `_split_transfer_packets` | :393-424 | 自 `KVTransferThread` 提取为模块函数或 GVA 线程基类方法（仅 GVA 使用） |
| `_batch_copy_with_limits` | :426-476 | 同上；**`self.m_store.store.batch_copy` → `self.m_store.batch_copy`**（见 1.3） |
| `KVCacheStoreLayerSendingThread` | :1271-1404 | 整类平移；基类改 `from ..kv_transfer import KVTransferThread` |
| `KVCacheStoreLayerRecvingThread` | :1405-1600 | 整类平移 |
| `_build_group_layer_builders` | pool_worker.py:444-459 | 迁入工厂函数内部（读取 worker 传入的 token_database 等 ctx 字段） |

**新增线程上下文与工厂**（gva_threads.py，替代 pool_worker 内联构造）：

```python
@dataclass
class GVALayerwiseThreadContext:
    """GVA 线程构造参数汇总——字段即现有两个线程 __init__ 的参数。"""
    token_database: ChunkedTokenDatabase
    block_size: int | list[int]
    tp_rank: int
    tp_size: int
    dcp_size: int
    page_size_bytes: int
    num_layers: int
    get_event: threading.Event
    layer_load_finished_events: list[threading.Event]
    layer_save_finished_events: list[threading.Event]
    sync_save_events: list[torch.npu.Event]
    max_transfer_blocks: int
    max_transfer_bytes: int
    h2d_stagger_us: int
    external_slot_release_waiter: Callable[[int], None] | None


def create_gva_sending_thread(store: MemcacheBackend, ctx: GVALayerwiseThreadContext,
                              ) -> KVCacheStoreLayerSendingThread:
    """group_builders 在此构造（原 pool_worker._build_group_layer_builders 逻辑）。"""


def create_gva_recving_thread(store: MemcacheBackend, ctx: GVALayerwiseThreadContext,
                              save_failure_checker: Callable[[], None] | None,
                              ) -> KVCacheStoreLayerRecvingThread:
    ...
```

不采用"工厂方法挂 Backend 接口"的方案：会迫使 base.py 反向 import 线程类型（与 gva_threads → base 构成循环），且 15+ 构造参数膨胀接口。模块级工厂 + ctx dataclass 与 P0 方案的 GVASession/GVAHitChecker 挂接风格一致。

### 1.3 封堵 backend 内部直捅（batch_copy 接口化）

`GVALayerwiseCapable`（P0 方案 PR1 建立）新增第 6 个抽象方法：

```python
# base.py — GVALayerwiseCapable
@abstractmethod
def batch_copy(self, gvas: list[int], addrs: list[list[int]],
               sizes: list[list[int]], direction: int) -> int:
    """GVA 批量拷贝。direction: 0=L2G(save) / 1=G2L(load)。"""

# memcache_backend.py
def batch_copy(self, gvas, addrs, sizes, direction) -> int:
    self.ensure_initialized()
    assert self.store is not None
    return self.store.batch_copy(gvas, addrs, sizes, direction)
```

迁移后的 `_batch_copy_with_limits` 内 `assert self.m_store.store is not None` 与 `self.m_store.store.batch_copy(...)`（原 :467-473）替换为 `self.m_store.batch_copy(...)`。**这是 P1-4 中唯一的语义修正点**（修封装破坏），其余纯平移。

### 1.4 pool_worker 选择逻辑改造

`_start_kv_transfer_threads`（:463-584）保留通用调度分支（`use_layerwise` / `can_save` / `load_async`——这是与后端无关的调度语义），仅 GVA 分支内联构造替换为工厂调用：

```python
# before：GVA 分支两段 ~55 行内联构造（:478-501 / :502-544）
# after
if self.use_layerwise:
    ...  # events 构造不变（通用）
    can_save = ...
    if self.use_gva_layerwise and can_save:
        ctx = self._build_gva_thread_context()          # worker 组装 ctx（~20 行，纯字段取值）
        assert isinstance(self.m_store, MemcacheBackend)
        self.kv_send_thread = create_gva_sending_thread(self.m_store, ctx)
        self.kv_send_thread.start()
        ctx_ready.wait()                                 # ready_event wait 语义保持
    elif can_save:
        ...  # key-layerwise 线程构造不变（通用）
    if self.use_gva_layerwise:
        self.kv_recv_thread = create_gva_recving_thread(
            self.m_store, ctx,
            save_failure_checker=self.kv_send_thread.raise_if_failed if self.kv_send_thread else None)
    else:
        ...  # key-layerwise recving 不变
```

ready_event 的创建/start/wait 时序逐行保持（发送线程 ready 后才构造接收线程的 `save_failure_checker` 引用——现顺序依赖此语义）。

### 1.5 收敛后验收

- `grep -n "KVCacheStoreLayer\(Sending\|Recving\)Thread\|LayerBatchBuilder" pool_worker.py kv_transfer.py` → 零命中（仅 backend/gva_threads.py）
- `grep -n "\.store\." kv_transfer.py gva_threads.py` → 零命中（batch_copy 已接口化；memcache_backend.py 内部自用除外）
- pool_worker 中 `use_gva_layerwise` 引用仅剩：`_init_kv_transfer_config`（P0-1 单点定义）、`_start_kv_transfer_threads`（2 处分支）、`_init_layerwise_config` 相关——全部为能力 flag 的合法消费点

## 2. P1-5：backend 生命周期收敛（LIFE）

### 2.1 现状

pool_worker.py:816-817：

```python
if self.use_gva_layerwise:
    self.m_store.ensure_initialized()   # memcache 惰性初始化的 fail-fast，由通用层特判驱动
self.m_store.register_buffer(ptrs, lengths)
self._start_kv_transfer_threads()
```

根因：memcache 需在 worker 启动阶段同步暴露初始化错误（fail-fast），但"何时初始化"的知识被写死在通用层。

### 2.2 方案：通用生命周期 hook

```python
# base.py — Backend 基类新增（默认空实现）
def on_worker_ready(self) -> None:
    """kv caches 注册完成后、传输线程启动前调用。需要 eager 初始化的后端覆写。"""

# memcache_backend.py
def on_worker_ready(self) -> None:
    # Fail fast: 在 worker 启动阶段暴露 store 初始化错误，而非首次传输时。
    # 此前注册的 buffers 由 ensure_initialized 内 _register_buffers_if_needed 补注册。
    self.ensure_initialized()
```

pool_worker.py 改造（删除特判，无条件调用）：

```python
# after
self.m_store.register_buffer(ptrs, lengths)
self.m_store.on_worker_ready()
self._start_kv_transfer_threads()
```

**语义等价论证**（reviewer 关注点）：原顺序 `ensure_initialized → register_buffer`（直接注册）；新顺序 `register_buffer`（进入 `_pending_buffers`）→ `ensure_initialized`（其内部 memcache_backend.py:62-72 末尾 `_register_buffers_if_needed` 补注册）。最终注册状态一致，fail-fast 时机一致（均在 worker 启动同步段）。mooncake/yuanrong 继承默认 no-op，行为零变化。注：依赖的是 `ensure_initialized` 而非 `init_store`——#14697 已删除后者（已核实其 diff），不影响本论证。

### 2.3 提交归属

建议**并入 P0 方案的 PR1**（同为 backend 接口/生命周期主题，仅 ~15 行：base +1 方法、memcache +5、worker -2/+1）；若 PR1 已提交流转中，则独立小 PR 提交。不与 P1-4 捆绑（改动区域不同，避免无谓耦合）。

## 3. P2-6：零散后端痕迹清理（TRACE）

### 3.1 check_all_layers_exists 归属修正（见 §0 修正 2）

**不迁 backend**。平移至 metadata.py 作为模块级函数（与 P0 方案 §3.6 `get_partial_block_index` 同一先例——key 布局知识归 metadata 层）：

```python
# metadata.py
def check_all_layers_exists(res: list[int], num_layers: int) -> list[int]:
    """key-mode layerwise 布局聚合：每 block 的 num_layers 个连续 key 全 1 才算命中。"""
    # 平移自 pool_worker.py:2386-2396，逐行不改

```

pool_worker 两个调用点（:2121/:2320）改 import 调用；删除实例方法。函数签名与算法零改动。

### 3.2 "MemCache" 文案残留清理

P0 方案 PR3（GVA 协议迁移）落地后，pool_worker 内 :1135-:1437 的 MemCache 文案随方法自然进入 gva_protocol.py——**那里保留是对的**（协议实现处描述准确）。通用层真正残留（本次 grep 全量核实）：

| 位置 | 内容 | 处理 |
|---|---|---|
| attention_fence.py:31 | docstring "MemCache worker threads wait for..." | 改 "GVA layerwise transfer threads"（该文件是通用 fence 机制） |
| attention_fence.py:65 | docstring "per-layer gate for MemCache work" | 同上中性化 |
| layerwise_cache_layout.py:71 | docstring "for the MemCache GVA layerwise path" | 改 "for the GVA layerwise path"（函数本身已由 P0-1 收敛能力判断） |

纯 docstring 修正，~6 行 diff，随 PR6 或搭车任一 PR。

### 3.3 明确不处理项

- `pool_worker._make_sub_key_str`（:1859）等 key-mode key 构造在通用层——**正确归属**（key-mode 为三后端通用路径），不动
- memcache_backend.py / gva_protocol.py / gva_threads.py 内的 MemCache 文案——协议实现处，保留
- 需求分析 §3 P2-6 所列 pool_worker.py:1318 注释——已随 P0 PR3 迁移，无需独立处理

## 4. PR 拆分与提交顺序（延续 P0 编号）

| PR | 内容 | 规模 | 依赖 | 冲突面 |
|---|---|---|---|---|
| PR4 | P1-5（LIFE）：`on_worker_ready` hook + pool_worker 特判删除（或并入 PR1，见 §2.3） | ~15 行 | PR1（若 base.py 结构已定） | 极小 |
| PR5 | P1-4（THRD）：gva_threads.py 迁移 + `batch_copy` 接口化 + ctx 工厂 | ~800 行（移动为主） | PR1（GVALayerwiseCapable）+ PR3（gva_protocol 同目录聚合）；**强烈建议等 #12854 走向明确** | **大（#12854 同区重构）** |
| PR6 | P2-6（TRACE）：check_all_layers_exists 迁 metadata + docstring 清理 | ~80 行 | PR3 之后（文案残留清单才成立） | 极小 |

推荐顺序：**PR4 → （等 #12854 落地/关闭）→ PR5 → PR6**。PR4/PR6 均可独立先行。

### 与 open PR 的协调（P1-4 专项）

| PR | 关系 | 处理 |
|---|---|---|
| **#12854**（layerwise transfer rework） | **最高风险**：大概率重写 kv_transfer.py 线程结构，与 PR5 的 700 行移动双向冲突 | 启动 PR5 前必须确认 #12854 状态：已合并 → 基于其结果 rebase 方案行号；仍 open → 与作者协调，优先将"GVA 线程下沉 backend"作为 #12854 的前置基座合入（同 P0 方案对 gva_protocol.py 的定位） |
| #14697（backend 路径简化） | batch_copy 接口新增于 GVALayerwiseCapable，与其正交 | rebase 单行适配 |
| #12711（Qwen3.5 layerwise） | 触碰 GVA 线程行为（safe extent） | 平移不改行为，rebase 即可 |

## 5. UT 计划

| 测试文件 | 改动 | 对应 PR |
|---|---|---|
| test_kv_transfer.py | GVA 线程 / LayerBatchBuilder 用例 import 改 `backend.gva_threads`；mock 目标 `m_store.store.batch_copy` → `m_store.batch_copy` | PR5 |
| test_gva_protocol.py（P0 方案新建） | 新增：`batch_copy` 透传单测（MemcacheBackend mock store 断言参数与 direction 传递）；ctx 工厂构造参数映射（group_builders 数量 = num groups） | PR5 |
| test_backend.py | `on_worker_ready` 默认 no-op 断言（mooncake/yuanrong 不抛错）；MemcacheBackend 覆写触发 `ensure_initialized`（mock 验证调用） | PR4 |
| test_metadata.py | `check_all_layers_exists` 用例迁入（含每 block num_layers 全 1 / 任一 0 两种边界） | PR6 |
| test_pool_worker.py | 删除 `_build_group_layer_builders` 相关用例（逻辑已入工厂，由 test_gva_protocol 覆盖） | PR5 |

## 6. 验证清单（映射需求分析 §5）

| 验收标准 | 覆盖方式 |
|---|---|
| 1. UT 全量回归 | 每 PR 出口 |
| 2. mooncake 非 layerwise 冒烟 | PR4/PR5 出口（线程构造路径重构，回归确认） |
| 3. memcache layerwise 冒烟（TP=4 长前缀） | **PR5 核心出口**（GVA 线程构造/batch_copy 路径全变） |
| 6. grep 无后端判断残留 | PR5 后 §1.5 三条 grep 断言；PR6 后 docstring 清零 |
| 7. GVA 逻辑入 backend | PR5 后 kv_transfer.py 无 GVA 类；`.store.` 直捅清零 |
| 8. base.py 分层 | PR4（生命周期 hook）+ PR5（batch_copy 入 GVALayerwiseCapable） |
| 11. PR ≤ 1000 行 | PR5 ~800 行（移动为主）压线，若 #12854 rebase 后超限则拆 sending/recving 两 PR |

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| **#12854 双向大冲突**（PR5 最大风险） | §4 协调策略：先确认走向再启动；必要时与作者共建基座 |
| ready_event 时序（send 线程 ready 后才建 recv 的 save_failure_checker） | 工厂调用顺序逐行保持现结构（:478-544 的 start/wait 顺序）；PR5 冒烟重点覆盖 save 失败传导路径 |
| `_batch_copy_with_limits` 自基类提取后 key-mode 线程误引用 | 提取后立即 grep 确认 :972/:988（key-mode）无 `_batch_copy_with_limits` 调用（已核实：仅 :1363/:1560 GVA 使用） |
| on_worker_ready 时序变化破坏 memcache buffer 注册 | §2.2 等价论证 + PR4 专项 UT（pending buffer 补注册路径） |
| ctx dataclass 字段遗漏（15+ 参数） | 工厂内构造签名与现线程 __init__ 一一对应，mypy 强类型检查遗漏；UT 构造参数映射断言 |
| 文档间事实修正滞后误导 reviewer | 本文档 §0 修正已同步回 requirements_analysis.md（2.2 与 P2-6 表述） |
