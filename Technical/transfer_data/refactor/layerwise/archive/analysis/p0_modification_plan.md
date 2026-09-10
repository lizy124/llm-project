# AscendStore Layerwise P0 问题修改方案

> 依据：[requirements_analysis.md](requirements_analysis.md)（行号均对应 2026-08-29 main 快照，本地 clone 验证）
> 范围：P0-1（use_gva_layerwise 单点定义）、P0-2（GVA 协议收敛 backend）、P0-3（base.py 接口分层）
> 原则：**平移式迁移**——协议代码逐行搬移不改语义；key 构造逐字符一致；ReqMeta 字段写入不变；`kv_transfer.py` 消费方零改动

## 0. 方案总览

三个 P0 拆为三个改动包，按依赖顺序提交三个 PR：

| PR | 改动包 | 覆盖问题 | 预估 diff |
|---|---|---|---|
| PR1 | CAP + IFACE | P0-1 + P0-3 | ~300 行 |
| PR2 | KEY | P0-2 第一阶段：key 构造统一 | ~350 行 |
| PR3 | PROTO | P0-2 第二阶段：GVA 生命周期迁移 | ~800 行 |

收敛后目标结构：

```
ascend_store/
├── backend/
│   ├── __init__.py        # [改] backend_map + capabilities + use_gva_layerwise() 派生函数
│   ├── base.py            # [瘦身] 仅通用数据面；新增 GVALayerwiseCapable 抽象接口
│   ├── gva_protocol.py    # [新增] GVA 常量 / GVAKeyFactory / GVASession / GVAHitChecker
│   ├── memcache_backend.py   # [改] 继承 GVALayerwiseCapable
│   ├── mooncake_backend.py   # [不动] 存根在 base.py 不在子类，接口移出后自然失去继承
│   └── yuanrong_backend.py   # [不动] 同上
├── metadata.py            # [改] 新增 get_partial_block_index()（自 pool_worker 平移）
├── pool_worker.py         # [瘦身 ~420 行] GVA 协议方法删除，改为委托 self._gva_session
└── pool_scheduler.py      # [瘦身 ~85 行] hit-check 委托 self._gva_hit_checker
```

---

## 1. P0-1：`use_gva_layerwise` 单点定义（CAP）

### 1.1 设计

能力声明与后端注册表同址（backend/`__init__.py` 的 `backend_map`），派生语义收敛为一个函数。调用方只引用函数，不再出现 `backend_name == "memcache"` 字符串比较（验收标准 6）。

### 1.2 修改 backend/`__init__.py`

```python
backend_map = {
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "...mooncake_backend",
        "capabilities": (),                # 新增
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "...memcache_backend",
        "capabilities": ("gva_layerwise",),  # 新增
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "...yuanrong_backend",
        "capabilities": (),                # 新增
    },
}


def backend_supports(backend_name: str, capability: str) -> bool:
    entry = backend_map.get(backend_name)
    return entry is not None and capability in entry["capabilities"]


def use_gva_layerwise(backend_name: str, use_layerwise: bool) -> bool:
    """唯一派生点：GVA layerwise 是否启用。"""
    return use_layerwise and backend_supports(backend_name, "gva_layerwise")
```

### 1.3 四处调用点替换

**① pool_worker.py:155-157**

```python
# before
self.backend_name = str(extra_config.get("backend", "mooncake")).lower()
self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"
# after
self.use_gva_layerwise = use_gva_layerwise(self.backend_name, self.use_layerwise)
```

属性名 `self.use_gva_layerwise` 保留——`:816`（ensure_initialized 特判）、`:463-584`（线程选择）等引用点不动，属于 P1/P2 范畴。

**② pool_scheduler.py:161-163**：同 ①，替换为 `use_gva_layerwise(self.backend_name, self.use_layerwise)`。

**③ layerwise_cache_layout.py:98-99**（`get_gva_layerwise_config` 内）

```python
# before
if backend == "memcache" and use_layerwise:
    return config
# after
if use_gva_layerwise(backend, use_layerwise):
    return config
```

该函数的其余部分（MultiConnector 嵌套遍历、connector 名单判断）不动——那是调度侧配置发现逻辑，与后端能力判定无关。

**④ ascend_store_connector.py:199**（main 回归修复，等效 412b157）

```python
# after（__init__ 中，restore 属性定义；ascend_store_connector 与 backend 同包）
from .backend import use_gva_layerwise

self.use_gva_layerwise = use_gva_layerwise(
    str(extra_config.get("backend", "mooncake")).lower(),
    self.use_layerwise,
)
```

此改动使 #14465 引入的 MultiConnector `AttributeError` 在 main 上修复（验收标准 5）。若 #12711（含 412b157）先合入，PR1 删除等效部分即可。

### 1.4 涉及文件

| 文件 | 动作 |
|---|---|
| backend/`__init__.py` | 加 capabilities 表 + 2 个函数（~25 行） |
| pool_worker.py:155-157 | 1 行替换 |
| pool_scheduler.py:161-163 | 1 行替换 |
| layerwise_cache_layout.py:98-99 | 1 行替换 |
| ascend_store_connector.py | ~3 行（回归修复） |
| tests/ut/.../test_backend.py | capabilities 断言（见 §5） |

---

## 2. P0-3：base.py 接口分层（IFACE）

### 2.1 设计

`Backend` 瘦身为纯通用数据面；5 个 memcache 专属接口提取为独立抽象接口 `GVALayerwiseCapable`；`MemcacheBackend` 双继承。不采用运行时 hasattr 探测——静态（mypy/类型）可判定是本问题要达成的边界。

### 2.2 修改 backend/base.py

```python
# before：Backend 内含 5 个 memcache 专属抽象方法（base.py:30-45，
#         batch_get_key_info / batch_alloc / batch_add_lease /
#         batch_remove_lease / batch_write_finish）

# after
class Backend(ABC):
    """通用数据面：所有后端必须实现。"""
    set_device / register_buffer / exists / batch_is_exist / put / get   # 原样保留


class GVALayerwiseCapable(ABC):
    """GVA layerwise 协议数据面原语。仅 memcache 实现。"""
    @abstractmethod
    def batch_get_key_info(self, keys: list[str]) -> list[Any]: ...
    @abstractmethod
    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]: ...
    @abstractmethod
    def batch_add_lease(self, keys: list[str], ttl_ms: int) -> list[int]: ...
    @abstractmethod
    def batch_remove_lease(self, keys: list[str]) -> None: ...
    @abstractmethod
    def batch_write_finish(self, keys: list[str]) -> None: ...
```

接口放 base.py（接口分层一目了然），实现归属 gva_protocol.py / memcache_backend.py。

### 2.3 修改三个后端实现

| 文件 | 动作 |
|---|---|
| memcache_backend.py | `class MemcacheBackend(Backend, GVALayerwiseCapable):`——5 个方法已有实现，仅加继承 |
| mooncake_backend.py | **零改动**（勘误：5 个存根位于 base.py:35-48 而非子类，子类仅继承；接口移入 GVALayerwiseCapable 后两后端自然失去继承） |
| yuanrong_backend.py | 零改动（同上） |

### 2.4 一致性校验（防能力表与实现漂移）

UT 断言（test_backend.py）：

```python
def test_capability_table_matches_interface():
    for name, entry in backend_map.items():
        cls = load(entry)  # 动态加载后端类
        has_iface = issubclass(cls, GVALayerwiseCapable)
        declared = "gva_layerwise" in entry["capabilities"]
        assert has_iface == declared  # 双向一致
```

### 2.5 类型收益

`pool_worker` 中 `self.m_store: Backend` 不再承载未定义方法调用（GVA 调用点在 PR3 后全部位于 gva_protocol.py，协议类持有 `store: MemcacheBackend` 强类型引用），mypy 可静态检查 `batch_alloc` 等调用合法性。

---

## 3. P0-2：GVA 协议收敛（PROTO，分 PR2/PR3 两阶段）

### 3.1 新文件 backend/gva_protocol.py

**常量迁移**（自 pool_worker.py:87-93，逐字平移；已确认仅 pool_worker 使用）：

```python
LAYERWISE_READ_LEASE_TTL_MS = 5 * 60 * 1000
MEMCACHE_UNMATCHED_STATE = -3101
PARTIAL_LEASE_RETRY_COUNT = 10
PARTIAL_LEASE_RETRY_INTERVAL_S = 0.001
```

**类一：GVAKeyFactory（PR2）**——统一两份 key 构造，消灭格式漂移：

```python
class GVAKeyFactory:
    """GVA key 构造唯一实现。输出必须与历史格式逐字节一致（PR #11585 兼容）。"""

    def __init__(self, model_name: str, num_kv_cache_groups: int,
                 head_or_tp_rank: int, head_or_tp_ranks: int = 1): ...

    def full_key(self, group_id: int, block_hash_hex: str) -> str:
        # 平移自 pool_worker._make_layerwise_gva_key (1113-1123)
        # 多组: model@group_id@hash@rank；单组: model@hash@rank

    def partial_key(self, req_id: str, group_id: int,
                    block_index: int, end_token: int) -> str:
        # 平移自 pool_worker._make_layerwise_partial_key (1125-1133)
        # model@partial@req_id@group_id@block_index@end_token@rank

    def hit_check_keys(self, group_id: int, block_hash_hex: str) -> list[str]:
        # 平移自 pool_scheduler._make_layerwise_gva_keys_for_hit_check (306-316)
        # all-rank 展开：[full_key 同格式, rank=h for h in range(head_or_tp_ranks)]
```

`head_or_tp_ranks = tp_size // put_step` 由 scheduler 传入（worker 侧默认 1，不使用该参数）。

**类二：GVASession（PR3）**——worker 侧生命周期（保存分配 + 加载租约）：

```python
class GVASession:
    """Worker 侧 GVA 协议：save 前分配 GVA、load 前获取 key_info + 读租约。"""

    def __init__(self, store: MemcacheBackend, key_factory: GVAKeyFactory,
                 kv_role: str, consumer_is_to_put: bool,
                 tp_rank: int, put_step: int,
                 num_kv_cache_groups: int, grouped_block_size: list[int],
                 group_block_len: dict, hash_block_size: int,
                 layerwise_offload: bool, page_size_bytes: int):
        self._store = store
        self._keys = key_factory
        ...  # ctx 参数原样平移为实例属性
        self._allocated_gvas: dict[str, int] = {}   # 自 pool_worker.py:346 平移

    def alloc_gvas_for_save(self, requests: list[ReqMeta]) -> None: ...
        # 平移自 pool_worker._alloc_gvas_for_save (1151-1313)

    def prepare_load_gvas(self, requests: list[ReqMeta],
                          on_invalid_blocks: Callable[[list[int]], None]) -> None: ...
        # 平移自 pool_worker._prepare_load_gvas (1315-1515)

    def _refresh_allocated_gvas(self, keys: list[str]) -> None: ...
        # 平移自 pool_worker._refresh_allocated_gvas (1134-1149)
```

**类三：GVAHitChecker（PR3）**——scheduler 侧命中判定：

```python
class GVAHitChecker:
    """Scheduler 侧 GVA 命中检查：all-rank key → key_info → 逐组取 min。"""

    def __init__(self, store: MemcacheBackend, key_factory: GVAKeyFactory,
                 grouped_block_size: list[int], hash_block_size: int): ...

    def hit_tokens(self, request: "Request",
                   token_len: int, num_computed_tokens: int) -> int: ...
        # 平移自 pool_scheduler._get_layerwise_gva_hit_tokens (319-387)
```

### 3.2 迁移映射表（旧 → 新）

| 旧位置 | 新位置 | 平移要点 |
|---|---|---|
| pool_worker.py:87-93 常量 ×4 | gva_protocol.py | 逐字 |
| pool_worker.py:1113-1123 `_make_layerwise_gva_key` | GVAKeyFactory.full_key | 逐字符 |
| pool_worker.py:1125-1133 `_make_layerwise_partial_key` | GVAKeyFactory.partial_key | 逐字符 |
| pool_scheduler.py:306-316 `_make_layerwise_gva_keys_for_hit_check` | GVAKeyFactory.hit_check_keys | 逐字符 |
| pool_worker.py:1134-1149 `_refresh_allocated_gvas` | GVASession._refresh_allocated_gvas | `self.m_store`→`self._store` |
| pool_worker.py:1151-1313 `_alloc_gvas_for_save` | GVASession.alloc_gvas_for_save | 见 3.3 |
| pool_worker.py:1315-1515 `_prepare_load_gvas` | GVASession.prepare_load_gvas | 见 3.4 |
| pool_scheduler.py:319-387 `_get_layerwise_gva_hit_tokens` | GVAHitChecker.hit_tokens | 见 3.5 |
| pool_worker.py:1098-1110 `_get_partial_block_index` | metadata.py `get_partial_block_index`（模块级函数） | 见 3.6 |

### 3.3 GVASession 挂接（pool_worker 侧）

**构造**（替换 pool_worker.py:346 `self._allocated_gvas: dict[str, int] = {}`；实际构造点放在 `__init__` 各 `_init_*` 全部完成后——建议 `_init_layerwise_config`（:348-442）末尾，需保证 `grouped_block_size` / `group_block_len` / `page_size_bytes` 已赋值）：

```python
self._gva_session: GVASession | None = None
if self.use_gva_layerwise:
    assert isinstance(self.m_store, MemcacheBackend)
    self._gva_session = GVASession(
        store=self.m_store,
        key_factory=GVAKeyFactory(self.model_name, self.num_kv_cache_groups,
                                   self.head_or_tp_rank),
        kv_role=self.kv_role, consumer_is_to_put=self.consumer_is_to_put,
        tp_rank=self.tp_rank, put_step=self.put_step,
        num_kv_cache_groups=self.num_kv_cache_groups,
        grouped_block_size=self.grouped_block_size,
        group_block_len=self.group_block_len,
        hash_block_size=self.hash_block_size,
        layerwise_offload=self.layerwise_offload,
        page_size_bytes=self.page_size_bytes,
    )
```

**调用点**（pool_worker.py:1609-1610，process_layer_data 内）：

```python
# before
self._prepare_load_gvas(requests)
self._alloc_gvas_for_save(requests)
# after（调用顺序不变：先 load 租约后 save 分配）
if self._gva_session is not None:
    self._gva_session.prepare_load_gvas(
        requests, on_invalid_blocks=self._report_invalid_blocks)
    self._gva_session.alloc_gvas_for_save(requests)
```

**worker 新增薄回调**（解耦 `_invalid_block_ids` 状态所有权）：

```python
def _report_invalid_blocks(self, block_ids: list[int]) -> None:
    with self._invalid_block_ids_lock:
        self._invalid_block_ids.update(block_ids)
```

**方法内 gate 处理**：原 `_alloc_gvas_for_save` 开头的 `if not self.use_gva_layerwise: return` **删除**（协议对象仅在 GVA 模式构造，调用点已 gate）；`kv_role == "kv_consumer"` 与 `tp_rank % put_step` 两个 gate 保留在协议方法内（协议语义的一部分）。`_prepare_load_gvas` 同理。

### 3.4 prepare_load_gvas 平移改造点

- `self.m_store` → `self._store`；`self._make_layerwise_gva_key/_make_layerwise_partial_key` → `self._keys.full_key/partial_key`
- `self._get_partial_block_index` → `get_partial_block_index`（from metadata import）
- invalid block 上报（原 1458-1465 区域的 `with self._invalid_block_ids_lock: self._invalid_block_ids.update(...)`）→ `on_invalid_blocks(invalid_block_ids)`
- 其余逐行平移：租约获取、`MEMCACHE_UNMATCHED_STATE` 部分租约重试（10 次 × 1ms）、multi-group 失败时 `batch_remove_lease` 回收 + RuntimeError、日志文案原样保留（P2 才清理 "MemCache" 字样）

### 3.5 GVAHitChecker 挂接（pool_scheduler 侧）

**构造**（pool_scheduler.py:163 之后）：

```python
self._gva_hit_checker: GVAHitChecker | None = None
if self.use_gva_layerwise:
    assert isinstance(self.store_scheduler, MemcacheBackend)
    self._gva_hit_checker = GVAHitChecker(
        store=self.store_scheduler,
        key_factory=GVAKeyFactory(
            self.model_name, len(self.kv_cache_group_ids),
            head_or_tp_rank=0,                       # hit-check 不使用单 rank
            head_or_tp_ranks=self.tp_size // self.put_step),
        grouped_block_size=self.grouped_block_size,
        hash_block_size=self.hash_block_size,
    )
```

**调用点**（pool_scheduler.py:467-469，get_num_new_matched_tokens 分支）：

```python
# before
if self.use_gva_layerwise:
    self._get_or_create_request_tracker(request.request_id)
    num_external_hit_tokens = self._get_layerwise_gva_hit_tokens(
        request, token_len, num_computed_tokens)
# after
if self._gva_hit_checker is not None:
    self._get_or_create_request_tracker(request.request_id)  # tracker 归 scheduler，留在调用点
    num_external_hit_tokens = self._gva_hit_checker.hit_tokens(
        request, token_len, num_computed_tokens)
```

`_get_or_create_request_tracker` 是 scheduler 的请求跟踪逻辑（通用），不迁协议——原实现只是"顺手"在 hit_tokens 里调了一次，移到调用点行为等价（每次 hit check 前确保 tracker 存在）。

**hit_tokens 方法内两处简化**（平移时明示，供 reviewer 确认）：

1. `query_start_block = 0 if self.use_layerwise else min(...)` → 固定 `0`。推理：HitChecker 仅在 `use_gva_layerwise=True` 时构造，而 `use_gva_layerwise ⇒ use_layerwise`，分支恒走 0。
2. `self.tp_size // self.put_step` → KeyFactory 构造时传入的 `head_or_tp_ranks`。

### 3.6 `_get_partial_block_index` 的归属裁定

**不进 backend**：它被 `_process_save_for_layer_batch`（:994）/ `_process_load_for_layer_batch`（:1060）调用——这两者是 layerwise **通用**任务构造路径（GVA 与 key 模式共用）。同时 GVA 协议（:1251/:1359）也需要它。

方案：移到 metadata.py 作为模块级函数 `get_partial_block_index(token_count, block_size, hash_count, enabled)`（与既有 `get_block_hashes` / `get_group_block_size` 同层），pool_worker 与 gva_protocol 双方 import，删除 pool_worker 的 staticmethod。纯函数零依赖，迁移无风险。

### 3.7 明确不迁移项（避免范围蔓延）

| 项 | 理由 |
|---|---|
| `LayerBatchBuilder`（kv_transfer.py:39） | P1-4 线程选择/地址构造，本次不动 |
| `ensure_initialized` 特判（pool_worker.py:816-817） | P1-5，保留 `self.use_gva_layerwise` 引用（P0-1 后已指向单点定义） |
| `check_all_layers_exists`（:2386） | P2-6，layerwise key 布局耦合，非 GVA 专属 |
| 日志/异常文案中的 "MemCache" | P2-6，文案清理随协议已迁移自然带入 gva_protocol.py，通用层残留留待 P2 |
| ReqMeta 字段结构 | 不动（写入字段名/时序完全不变，`kv_transfer.py` 消费方零改动） |

### 3.8 兼容性保障（验收标准 9/10）

1. **key 字节级快照 UT 前置**（PR2 交付）：迁移前先在现有代码上运行固定输入采集输出，将字面量固化进 `test_gva_protocol.py`：

```python
def test_gva_key_format_snapshot():
    kf = GVAKeyFactory(model_name="qwen", num_kv_cache_groups=2, head_or_tp_rank=3)
    assert kf.full_key(1, "a" * 16) == "qwen@1@aaaaaaaaaaaaaaaa@3"
    assert kf.partial_key("req-42", 0, 7, 128) == "qwen@partial@req-42@0@7@128@3"
    kf2 = GVAKeyFactory(model_name="qwen", num_kv_cache_groups=1,
                        head_or_tp_rank=0, head_or_tp_ranks=4)
    assert kf2.hit_check_keys(0, "b" * 16) == [f"qwen@{'b'*16}@{h}" for h in range(4)]
```

2. ReqMeta 写入语句（`save_keys` / `block_gvas_by_group_np` / `partial_save_gva_per_group` / `load_keys` / `load_block_gvas_np` / `load_gva_block_offset` 等）为平移，字段与赋值时序不变。
3. `process_layer_data` 内 prepare_load → alloc_save 顺序保持。
4. 配置项（`use_layerwise` / `backend` / `layerwise_max_transfer_blocks` 等）不新增不改名。

---

## 4. PR 拆分与提交顺序

### PR1：CAP + IFACE（P0-1 + P0-3）

- 内容：§1 全部 + §2 全部
- 依赖：无（可立即开工）
- 附带价值：修复 main 上 MultiConnector 回归（#14465 遗留），不依赖 #12711 合入
- 冲突面：与 #14697 正交（本 PR 不触碰 `exists`/`batch_is_exist` 的调用与签名）
- 验证：UT 全量 + 验收标准 4（MultiConnector 冒烟）

### PR2：KEY（P0-2 第一阶段）

- 内容：新建 gva_protocol.py（常量 + GVAKeyFactory）；pool_worker 的 `_make_layerwise_gva_key` / `_make_layerwise_partial_key`、pool_scheduler 的 `_make_layerwise_gva_keys_for_hit_check` 三个方法删除，全部调用点改为 `self._gva_keys.full_key(...)` / `partial_key(...)` / `hit_check_keys(...)`（worker/scheduler 各构造一个 `GVAKeyFactory` 实例 `self._gva_keys`）
- 依赖：PR1（backend 目录能力判定就位）
- 关键 UT：§3.8-1 key 快照（本 PR 必须交付，作为 PR3 的前置基线）
- 注：本 PR 结束时两份 key 构造已合并为一份，格式漂移风险即刻消除

### PR3：PROTO（P0-2 第二阶段）

- 内容：GVASession + GVAHitChecker 迁入 gva_protocol.py；`get_partial_block_index` 移 metadata.py；§3.3/3.4/3.5 挂接与回调；删除 pool_worker :1134-1515（`_refresh_allocated_gvas` / `_alloc_gvas_for_save` / `_prepare_load_gvas`）与 pool_scheduler :306-387（两个 GVA 方法）
- 依赖：PR2
- UT 迁移：`test_pool_worker.py` 的 `test_mtp_gva_prepare_uses_safe_extent_not_store_skip_extent`（:1166）、`test_evicted_allocated_gva_is_reallocated`（:1423）与 `test_pool_scheduler.py` 的 `test_layerwise_gva_hit_tokens`（:800）、`test_layerwise_mtp_hit_uses_safe_load_extent`（:139）改为对 GVASession / GVAHitChecker 直接测试（不再构造完整 worker/scheduler），移入 test_gva_protocol.py
- diff 预估：生产代码净移动 ~500 行 + 调用点改造 + UT 适配 ≈ 800 行，满足 ≤1000 行约束

### 与 open PR 的协调

| PR | 关系 | 处理 |
|---|---|---|
| #12711（412b157 修复 + Qwen3.5） | PR1 含等效修复 | 谁先合谁生效，后到者 rebase 去重 |
| #14697（backend 路径/`batch_is_exist`→`exists`） | `_refresh_allocated_gvas` 平移时用到 `batch_is_exist` | 平移保持现状；rebase 时按 #14697 落地结果适配单行 |
| #12854（layerwise transfer rework，动 kv_transfer.py） | 本方案不动 kv_transfer.py | 冲突面小；PR3 合入后 gva_protocol.py 成为 #12854 的稳定基座 |

---

## 5. UT 计划汇总

| 测试文件 | 新增/改动 | 对应 PR |
|---|---|---|
| test_backend.py | capabilities 双向一致性断言（§2.4）；`use_gva_layerwise()` 派生真值表（mooncake×layerwise=False、memcake×layerwise=True 等） | PR1 |
| test_ascend_store_connector.py | MultiConnector 场景 `use_gva_layerwise` 属性存在性回归（#14465 场景复现） | PR1 |
| test_layerwise_cache_layout.py | `get_gva_layerwise_config` 用例适配（行为不变，断言不改语义） | PR1 |
| test_gva_protocol.py（新建） | key 格式字节级快照（§3.8-1） | PR2 |
| test_gva_protocol.py | GVASession（alloc 跳过/缓存命中/partial 分配失败）+ GVAHitChecker（全组命中取 min、任一 rank miss 即断、MTP safe extent）——由 test_pool_worker/test_pool_scheduler 迁入适配 | PR3 |
| test_metadata.py | `get_partial_block_index` 用例迁入 | PR3 |

---

## 6. 验证清单（映射需求分析 §5 验收标准）

| 验收标准 | 覆盖方式 |
|---|---|
| 1. UT 全量回归 | 每 PR 出口（165 服务器既有流程） |
| 2. mooncake 非 layerwise 冒烟 | PR1/PR3 出口（base.py 存根删除 + 协议迁移不影响该路径，回归确认） |
| 3. memcache layerwise 冒烟（TP=4 长前缀） | PR3 出口 |
| 4. MultiConnector PD 冒烟 | **PR1 出口即测**（回归修复生效点）；PR3 复测 |
| 5. use_gva_layerwise 回归不存在 | PR1（test_ascend_store_connector 回归用例固化） |
| 6. 单点定义 / grep 无残留 | PR1 后 `grep -rn '== "memcache"'` 仅 backend/__init__.py 注册表命中 |
| 7. GVA 逻辑入 backend | PR3 后 pool_worker/pool_scheduler 无 `_gva`/`_alloc_gvas`/`_prepare_load_gvas`/`_make_layerwise_gva` 方法 |
| 8. base.py 分层 | PR1（GVALayerwiseCapable + 一致性 UT） |
| 9. key 格式不变 | PR2 快照 UT（字面量固化） |
| 10. 配置接口不变 | 全程无配置变更；PR1 真值表 UT 顺带固化默认值 |

---

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| GVASession 构造点初始化顺序（`grouped_block_size` / `group_block_len` / `page_size_bytes` 是否就绪） | 构造点放 `_init_layerwise_config` 末尾并在 PR3 中核对 `__init__` 各 `_init_*` 的赋值顺序；新增 GVASession 构造 UT 覆盖 |
| key 格式漂移 | PR2 快照 UT 前置 + 平移逐字符（禁止顺手重构 f-string） |
| mypy 对 key_info duck-typing（`ki.size()` / `ki.gva_list()`）报 Any | 保留现状 Any；如需收紧在 gva_protocol.py 内定义局部 Protocol（不扩散到本次范围外） |
| UT mock 路径变化（原 mock pool_worker 方法 → 现直连 store） | mock 对象不变（仍 mock MemcacheBackend 的 `batch_alloc` 等），构造 GVASession 直测更简单 |
| open PR rebase 顺序 | §4 协调表；PR1 可独立先行的设计正是为了不阻塞于 #12711/#14697 |
