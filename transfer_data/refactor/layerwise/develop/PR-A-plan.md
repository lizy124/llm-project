# PR-A 执行计划（施工版）

> 输入：[最终需求分析和方案设计.md](../最终需求分析和方案设计.md)（唯一执行基准）
> 施工仓：`d:\lzy\project\kv_pool\code\vllm-ascend`，分支 `refactor_layerwise`（= upstream/main = origin/main @ 6953f2669，工作区干净）
> 验证：165 服务器另行执行（听用户指示），本计划只含代码施工与本地 UT
> 开工前代码实况锚定完成（本 session 逐区域重读），与终稿行号全部吻合

---

## 0. 施工决策点（终稿未展开的 4 个实现细节，按代码实况裁定）

| # | 问题 | 裁定 |
|---|---|---|
| P-1 | `group_block_len`（pool_worker.py:726 声明、:684 填充）与 `page_size_bytes`（:790）在 `register_kv_caches` 内才有值，而 GVASession 构造点按终稿放 `_init_layerwise_config` 末尾（init 阶段） | **二段式构造**：`GVASession.__init__` 传静态参数（model_name / head_or_tp_rank / num_kv_cache_groups / grouped_block_size / hash_block_size / layerwise_offload / use_eagle / kv_role / consumer_is_to_put / tp_rank / put_step / invalid 回调），`register_kv_caches` 内 :790 之后调用 `self._gva_session.bind_layout(group_block_len, page_size_bytes)` 补动态参数。PR 描述中说明此偏离展开 |
| P-2 | LIFE 若在 C2 切换 :816 调用点而 GVASession 尚未存在，GVA+lazy 的 eager 在 commit 间丢失（坏窗口） | **:816 替换与 GVASession self-ensure 同 commit（C4）落地**，保证每个 commit 独立行为保持、bisect 友好 |
| P-3 | `_prepare_load_gvas` 直写 `self._invalid_block_ids`（含 `_invalid_block_ids_lock`） | worker 新增 `_report_invalid_blocks(ids: list[int])`（lock+update），session 经构造时传入的 `on_invalid_blocks` 回调上送；multi-group raise 逻辑留在 session 内（行为不变） |
| P-4 | CAP 第 3 处（layerwise_cache_layout.py:98-100 `get_gva_layerwise_config`）是对 extra_config dict 的检查，不是 backend_name 属性 | 只替换 `== "memcache"` 判断部分为 `backend_supports(str(extra_config.get("backend", "mooncake")).lower(), "gva_layerwise")`，`use_layerwise` 检查原样保留（语义：该 connector 配置 memcache 且开 layerwise 才返回其 extra_config） |

## 1. GVASession 参数就绪性锚定（P-1 依据，实测行号）

| 参数 | 赋值点 | 就绪时机 |
|---|---|---|
| `model_name` | :138 | __init__ |
| `tp_rank` / `put_step` / `head_or_tp_rank` | :136 / :206-210 | __init__（_init_kv_transfer_config 内） |
| `use_gva_layerwise` / `kv_role` / `consumer_is_to_put` | :150-155 | __init__ |
| `num_kv_cache_groups` / `grouped_block_size` / `hash_block_size` | :172-179 | __init__ |
| `layerwise_offload` | :415/:424/:429 | **`_init_layerwise_config` 内**（:348-442）→ 构造点放其末尾已就绪 |
| `use_eagle` | :164 | __init__ |
| `group_block_len` | :726 声明 / :684 填充 | **register_kv_caches** → bind_layout |
| `page_size_bytes` | :790 | **register_kv_caches** → bind_layout |
| `_invalid_block_ids` (+lock) | :151-153 | __init__（回调闭包引用即可） |

GVAHitChecker（scheduler 侧）参数：`model_name`（:212）、`tp_size`（:178）、`put_step`（:185-187）、`grouped_block_size` / `hash_block_size` / `kv_cache_group_ids` / `use_layerwise` / `block_size`（hash_block_size 相关）——全部 `__init__` 就绪，构造点放 scheduler `__init__` 末尾无障碍。

## 2. Commit 序列

### C1 — CAP 派生收敛 + main 回归修复
```
refactor(kv_pool): add backend capability registry with single use_gva_layerwise derivation
```
- `backend/__init__.py`：`_BACKEND_CAPABILITIES = {"mooncake": frozenset(), "memcache": frozenset({"gva_layerwise"}), "yuanrong": frozenset()}` + `backend_supports(backend_name: str, capability: str) -> bool` + `use_gva_layerwise(use_layerwise: bool, backend_name: str) -> bool`
- 4 处调用点收敛：
  - pool_worker.py:157 `self.use_gva_layerwise = use_gva_layerwise(self.use_layerwise, self.backend_name)`
  - pool_scheduler.py:163 同式
  - layerwise_cache_layout.py:98-100（按 P-4）
  - ascend_store_connector.py `__init__`（:104 附近）恢复 `self.use_gva_layerwise = use_gva_layerwise(self.use_layerwise, str(extra_config.get("backend", "mooncake")).lower())` —— **回归修复主体**（#14465 误删、:199 仍读、MultiConnector 初始化即 AttributeError）
- UT：test_backend.py capabilities 真值表 +（C2 后补 issubclass 双向一致性）；test_ascend_store_connector.py MultiConnector `use_gva_layerwise` 属性存在性回归（#14465 场景复现）

### C2 — IFACE 接口分层
```
refactor(kv_pool): extract GVALayerwiseCapable interface from base Backend
```
- `backend/base.py`：删 :34-47 五个 memcache 存根 → 新 `GVALayerwiseCapable(ABC)`（batch_get_key_info / batch_alloc / batch_add_lease / batch_remove_lease / batch_write_finish，@abstractmethod）；`Backend` 加 `on_worker_ready()` 默认空实现
- `backend/memcache_backend.py`：`class MemcacheBackend(Backend, GVALayerwiseCapable)` 双继承（5 个已有实现自然满足 ABC，零方法改动）；`on_worker_ready` 覆写（`if self._lazy_init: return` + `self.ensure_initialized()` + 终稿注释）
- UT：on_worker_ready 默认 no-op（mooncake/yuanrong 不抛错）；`_lazy_init=True` 下 no-op 断言（**UT 1**）；exists lazy 短路契约（**UT 3**，既有行为固化）

### C3 — GVA 协议模块（纯新增，main 行为零变化）
```
refactor(kv_pool): add GVA protocol module with key factory, session and hit checker
```
- 新增 `backend/gva_protocol.py`：
  - 常量 ×4（pool_worker.py:87-93 逐字：`LAYERWISE_READ_LEASE_TTL_MS` / `MEMCACHE_UNMATCHED_STATE` / `PARTIAL_LEASE_RETRY_COUNT` / `PARTIAL_LEASE_RETRY_INTERVAL_S`）
  - `GVAKeyFactory`：`full_key(model_name, group_id, block_hash_hex, head_or_tp_rank, num_groups)` / `partial_key(model_name, req_id, group_id, block_index, end_token, head_or_tp_rank)` / `hit_check_keys(model_name, group_id, block_hash_hex, num_ranks)` —— 统一 worker :1113-1133 与 scheduler :306-317 两份实现（单/多 group 两格式）
  - `GVASession`：`__init__` 首行 `store.ensure_initialized()`（LIFE self-ensure）+ `bind_layout` + `_refresh_allocated_gvas`（:1134-1149 平移）+ `alloc_gvas_for_save`（:1151-1313 平移，`_allocated_gvas` 内化）+ `prepare_load_gvas(requests, on_invalid_blocks)`（:1315-1515 平移）
  - `GVAHitChecker`：`hit_tokens(request, token_len, num_computed_tokens)`（pool_scheduler :319-387 平移，`_get_or_create_request_tracker` 移至调用点）
- `metadata.py`：+`get_partial_block_index(token_count, block_size, hash_count, enabled)`（pool_worker :1097-1111 平移，staticmethod → 模块函数）
- 新增 `tests/ut/distributed/ascend_store/test_gva_protocol.py`：key 格式字节级快照（full/partial/hit-check × 单/多 group）；alloc 跳过已存在 / 缓存命中 / partial 分配失败；hit checker 全组命中取 min / MTP safe extent；GVASession 构造触发 ensure（**UT 2**）；batch_get_key_info 空返回契约 golden（**UT 4**）

### C4 — 委托切换 + LIFE 落地（唯一大删改 commit）
```
refactor(kv_pool): delegate worker and scheduler GVA paths to GVASession
```
- `pool_worker.py`：
  - 删 :87-93 常量 ×4、:1097-1515 六个方法（`_get_partial_block_index` / `_make_layerwise_gva_key` / `_make_layerwise_partial_key` / `_refresh_allocated_gvas` / `_alloc_gvas_for_save` / `_prepare_load_gvas`，~420 行）
  - `_init_state_vars`：删 `_allocated_gvas`（内化于 session）；`_init_layerwise_config` 末尾构造 `self._gva_session = GVASession(...) if self.use_gva_layerwise else None`（含 on_invalid_blocks 回调）
  - `register_kv_caches`：:790 后 `bind_layout(group_block_len, page_size_bytes)`；**:816-817 `if self.use_gva_layerwise: self.m_store.ensure_initialized()` → 无条件 `self.m_store.on_worker_ready()`**（P-2：与 self-ensure 同 commit）
  - `process_layer_data`（:1609-1610）：`if self._gva_session is not None:` 委托 `prepare_load_gvas(requests, on_invalid_blocks=self._report_invalid_blocks)` + `alloc_gvas_for_save(requests)`（prepare→alloc 顺序不变）；新增 `_report_invalid_blocks`
  - 其余 `_make_layerwise_gva_key` 等调用点（若有散在）一并切换
- `pool_scheduler.py`：删 :306-387 两方法（~85 行）；`__init__` 末尾构造 `self._gva_hit_checker = GVAHitChecker(model_name=..., head_or_tp_rank=0, head_or_tp_ranks=self.tp_size // self.put_step, ...)`；:467-469 委托 `self._gva_hit_checker.hit_tokens(request, token_len, num_computed_tokens)`（`_get_or_create_request_tracker` 调用移至 :467 委托前——行为等价）
- 既有 UT 迁移适配：test_pool_worker :1166（test_mtp_gva_prepare_uses_safe_extent_not_store_skip_extent）/ :1423（test_evicted_allocated_gva_is_reallocated）、test_pool_scheduler :139（test_layerwise_mtp_hit_uses_safe_load_extent）/ :800（test_layerwise_gva_hit_tokens）——mock 面从 worker/scheduler 方法改为 session/hit_checker
- 此 commit 后 grep 断言：backend 目录外 `== "memcache"` 业务判断零残留；pool_worker / pool_scheduler 无 GVA 协议方法

### C5 — lint/typing 收尾（如需）
```
style: fix ruff/mypy findings
```
C1-C4 全绿则跳过。

## 3. 文件清单

| 文件 | 操作 | 规模预估 |
|---|---|---|
| `backend/gva_protocol.py` | **新增** | ~650 行（含 docstring/注释） |
| `backend/__init__.py` | 改 | 30 → ~65 行 |
| `backend/base.py` | 改 | 56 → ~80 行 |
| `backend/memcache_backend.py` | 改 | +双继承 + `on_worker_ready` ~15 行 |
| `metadata.py` | 改 | +~20 行 |
| `pool_worker.py` | 改 | 2266 → ~1850 行 |
| `pool_scheduler.py` | 改 | 969 → ~885 行 |
| `ascend_store_connector.py` | 改 | +2 行（回归修复） |
| `layerwise_cache_layout.py` | 改 | :98-100 一处替换 |
| `test_gva_protocol.py` | **新增** | ~350 行 |
| `test_backend.py` | 改 | capabilities 真值表 + 双向一致性 |
| `test_ascend_store_connector.py` | 改/查 | MultiConnector 属性存在性回归 |
| `test_pool_worker.py` / `test_pool_scheduler.py` | 改 | 4 个用例迁移适配 |

## 4. 验证出口（PR-A 合入门槛，165 服务器执行，听指示）

1. `tests/ut/distributed/ascend_store` 全量绿
2. mooncake 非 layerwise 冒烟无回归（IFACE 波及面）
3. memcache layerwise 冒烟：TP=4 + 长前缀 load 复测（PROTO 平移正确性）
4. **MultiConnector PD 冒烟**（P 4×TP + D consumer，proxy，GSM8K prefix-cache，成功率 100%）——回归修复生效实测
5. 三条 grep 断言（终稿验收标准 6/7）
6. ruff / mypy（3.10-3.12）

## 5. 平移纪律（终稿 §3.4 重申）

逐字符一致、禁止顺手重构（f-string、日志文案一律原样）；key 构造输出以字节级快照 UT 锁定；ReqMeta 字段写入不变；`process_layer_data` 内 prepare_load（:1609）→ alloc_save（:1610）顺序不变；`kv_role == "kv_consumer"` 与 `tp_rank % put_step` gate 保留（协议语义）。
