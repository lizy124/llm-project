# 止血代码：只跳过"表不是权威全表"的保存

> **一句话**：不改保存逻辑，只在**调度侧**判断"这张块表是否从请求的 block 0 开始"，
> 不是就**本步不保存**。这样既不会抛 `ReqMeta GVA metadata does not cover requested block range`，
> 也不会出现"把 `base+k` 块的 KV 写到 `k` 块 key 下"的**静默写错**。
>
> 本文自包含：语义 / 删减清单 / 逐处代码 / 完整补丁 / 已做验证 / 落地 / 回退 / 代价与风险。

> **⚠️ 状态更新（2026-10-23，必读）**：本文 §5 的 v1 补丁已于 2026-10-23 01:22 部署到
> 129/130 两机，随即**触发 §9 的"UNPROVABLE 红线"**——所有长请求从第一个 chunk 起全部
> 被判 `UNPROVABLE` 跳过，池化写入停摆（重启前累计 1832 次跳过，0 次误写）。
> 现场已修复（new 分支不再依赖登记表比对，直接判 `ZERO`），当前两机运行为 **v3**
> （commit `e69705578`/`8879bb3c5`）：**新进程跳过 0 次、fatal 0 次、池命中率恢复 44–46%**。
> 完整时间线、根因与修复代码见 **§12**；§2–§11 保留为 v1 原始方案的历史记录。

> **提 PR 的速用路径**：根因分析 → §5 抽补丁 → §7 方式 A（`git apply`）→ §11 的 commit message 与 PR 描述。
> 其余章节是背景与审查依据，可作 PR 描述里的引用。

---

## 根因分析（评审用：先读这一节，再读止血方案）

### A.1 现象

layerwise 保存路径在建共享块数据时抛异常：

```text
RuntimeError: ReqMeta GVA metadata does not cover requested block range [<start>, <end>) with x < 0 or ... > N
```

调用链：

```text
model_runner.execute_model
  → kv_connector.pre_forward
  → ascend_store_connector.bind_connector_metadata
  → pool_worker.prepare_layerwise_step
  → process_layer_data
  → _build_shared_save_data
  → kv_transfer.build_shared_data
  → LayerBatchBuilder.build_shared        ← raise 在这里
```

抛出点（`kv_transfer.py`）：

```python
gva_start = block_range.start_block - gva_block_offset
gva_end   = block_range.end_block   - gva_block_offset
if gva_start < 0 or gva_end > len(block_gvas_np):
    raise RuntimeError("ReqMeta GVA metadata does not cover requested block range ...")
```

**关键认识**：这条 `raise` **只是保护**。它只在"越界"时拦一下；
当 range 恰好落在数组长度内、但**数组的起点与 range 的起点不是同一个基准**时，
断言**不会触发**，程序会带着错位的键值关系继续跑完 —— 也就是**静默写错**。
所以这个 bug 的两种表现是：**要么崩，要么悄悄写脏数据**。

### A.2 理解根因的钥匙：三个不同的"块号"

同一个请求上，代码里同时存在三种"第几个块"的口径：

| 口径 | 定义 | 谁在用 |
|---|---|---|
| **绝对块号 `b`** | 从**请求的第 0 块**开始数（`b` 个块 = `b × bs` 个 token） | 保存范围的**计算** |
| **表内下标 `i`** | 从**当前块表的表头**开始数（`i = 0` 是表的第一项） | GVA / block_id **数组的索引** |
| **hash 数** | 已提交 hash 的块计数 | load 侧的夹紧 |

当块表**就是**请求的完整表时，`表头 == 请求 block 0`，于是 `b ≡ i`，三套口径"看起来"一致 ——
这正是绝大多数请求能正常跑的原因。
但一旦块表变成**后缀窗口**（表头对应绝对块号 `base_blk > 0`），就有：

```text
绝对块号 b  ==  表内下标 i  +  base_blk        （base_blk > 0 时二者不再相等）
```

**而 `base_blk` 在整个代码里从来没有被记录过。** 这就是根因的落点。

### A.3 根因：三处各自用不同口径，且没人知道表的起点

| 环节 | 位置 | 实际口径 | 后果 |
|---|---|---|---|
| ① 造 **range** | `pool_worker._process_save_for_layer_batch` | **绝对块号**（由 token 数换算：`save_start_block` / `save_end_block`，并被 pool hit 抬高起点） | range 的上限按"请求第 0 块"算，可能远超表的块数 |
| ② 开 **save 数组** | `pool_worker._alloc_gvas_for_save_for_request` | **表内下标**：`end_limit = min(save_end_block, len(block_ids_by_group))` | 数组只保证"不越界"，却把 range 的**绝对语义悄悄换成了表内语义** |
| ③ 造 **load** | `pool_worker._prepare_load_gvas_for_request` | **hash 数** | 与①②又都不一样 |

①②口径不同 ⇒ `build_shared` 的覆盖断言被触发；
①②"恰好都不越界" ⇒ 键与 GVA 之间**整体错位 `base_blk` 个块**。

错位的具体形态（设 `gva_block_offset == 0`）：

```text
期望：绝对 b 块的 KV   →  挂在 key(绝对 b) 下
实际：数组按表内下标切片 [start, end)
      ⇒ 取到的是绝对 (base_blk + start) … (base_blk + end) 的 KV
      ⇒ 与 key(绝对 start) … key(绝对 end) 按位置配对
      ⇒ key(b) 下写的是绝对 (b + base_blk) 块的 KV        ← 偏移 base_blk 块
```

后果：**后续命中这些 key 的请求会读到别的块的 KV**。因为 key 是内容寻址（hash），
错位的键值对在读取时**完全无法被识别为错误** —— 命中率、日志、metric 全都正常。

### A.4 断言为什么有时拦住、有时拦不住

| 条件 | 结果 |
|---|---|
| `end_block > len(表)` | **抛 `RuntimeError`**（现场两次都是这一类） |
| `end_block ≤ len(表)` 且 `base_blk == 0` | 正常（绝大多数请求） |
| `end_block ≤ len(表)` 且 `base_blk > 0` | **不抛，静默错位 `base_blk` 块** ← 真正危险的情形 |

现场两次越界的形态（表长 6 组 vs range）：

```text
形态 A：group0 表长 128        ，save range [0, 224)     → 224 > 128   崩
形态 B 步1：表长 [63,1,255,255,4094,1023]  ，range [0, 224)   → 224 > 63    崩
形态 B 步2：表长 [127,3,511,511,8188,2047] ，range [224, 288) → 288 > 127   崩
```

这两组数字说明：**range 的块号尺度远大于表的块数** ⇒ `len(表) ≪ 表头之后的绝对块号`
⇒ 这些请求的表**必然是后缀窗口**（`base_blk > 0`），而不是从请求第 0 块起的完整表。

### A.5 窗口是怎么来的

最典型的来源是**抢占（preemption）后的重新调度**。这一支路的语义在代码里是明确的
（见 `_resolve_preempted_base_state` 的注释）：

> 被抢占请求的块表 = **本步新分配的那些块**，其起点 = 包含 `num_computed_tokens` 的块边界。

因此：

```text
num_computed_tokens == 0  →  表头就是请求 block 0   →  base_blk = 0   →  安全
num_computed_tokens >  0  →  表头在请求中间          →  base_blk > 0   →  **窗口**
```

即"被抢占过的请求"天然就是窗口请求。这解释了为什么该 fatal 与被抢占状态高度相关。

### A.6 为什么"取全表"这件事在现有代码里做不到

修复/止血都需要一个参照物：**请求的权威全表**（从请求 block 0 起的那张表）。
但：

| 事实 | 证据（已在容器内实测） |
|---|---|
| `vllm.v1.request.Request` **没有 `block_ids`**（`dir()` 只暴露 `update_block_hashes`） | `hasattr(Request, "block_ids") == False` |
| 连接器唯一的记录是 `_unfinished_requests[req_id] = (request, local_block_ids)`，**只在 `update_state_after_alloc` 写一次** | `pool_scheduler.py:780-784`（pristine） |
| 而它**只在 `num_external_tokens > 0` 时才记录真实表**，否则是 `[[], [], …]`（空表） | 同上，`local_block_ids: list[list[int]] = [[] for _ in self.kv_cache_group_ids]` |

⇒ 也就是说：**"没命中池的请求"在连接器侧根本没有全表可查**。
这正是止血补丁必须**额外登记一张权威全表**（`_full_block_ids`，无条件、每次分配都刷新）的原因。

### A.7 结论（一句话）

> **保存范围按"绝对块号"构造，GVA 数组按"表内下标"索引，
> 而"表头对应哪个绝对块号"（`base_blk`）从未被记录 ——
> 于是当请求的块表是后缀窗口时，两者错位：越界就崩，不越界就静默写错。**

### A.8 为什么这一版只做"止血"而不改保存逻辑

1. **断言是保护，不是病因**。要真正修好，需要把 range 与数组统一到同一绝对基准
   （前导 0 补齐 + 逐组夹紧 + 组数自检），那是另一个 PR 的规模（4 文件、+331/−6）；
2. 那个修复版的**参照物本身有缺陷**：它用的"全表"是上面 A.6 的空表快照，
   无命中请求会被判"不可证明"（详见 §10-9）；
3. 所以先做**能立刻证明安全的动作**：把"表不是权威全表"的请求**排除出保存**
   —— 丢一条缓存条目可重算，写错一条会污染池。

**覆盖性前提与验证方法**见 §2.5；**止血的确切语义**见 §2；**逐处代码**见 §4。

---

## 0. 一句话摘要（可直接当 PR 描述首段）

layerwise 保存路径用**绝对块号**造 range，却用**请求自己的块表**给 GVA 数组定基；
当块表不是从请求 block 0 起的权威全表时（如被抢占后只剩后缀窗口），两者错位：

```text
RuntimeError: ReqMeta GVA metadata does not cover requested block range
```

更糟的是 range 恰好落在数组长度内时**断言不触发**，于是绝对 `base+k` 块的 KV
被写到绝对 `k` 块的 key 下 —— **静默污染池内容且难以察觉**。

本补丁**不改保存逻辑**，只在调度侧证明"这张表是不是权威全表"，
不是就**本步不保存**（`can_save=False` / 不产生 `ReqMeta`）。
代价是"有窗口的请求不写池"（丢缓存条目，可重算），换来**不崩、不写错**。

---

## 1. 定位：止血 ≠ 修复

| | 止血（本文） | 修复（`dev_fix_p0p1` / `de0375537`） |
|---|---|---|
| 目标 | **不崩、不写错** | 不崩、不写错，**且窗口请求也能正确写池** |
| 手段 | 判出"表起点非 0"⇒ **本步不保存** | 记录 per-group base，并把 range 与数组**统一到绝对基准**（前导 0 补齐）+ 逐组夹紧 + 组数自检 + 游标 |
| 改动面 | **2 文件、+121 / −0** | 4 文件、+331 / −6 |
| 代价 | 有窗口的请求**不写池**（丢缓存条目） | 无（设计目标） |
| 风险面 | 只新增一个判定；`ZERO` 路径**逐位不变** | 需要验证对齐保存的正确性 |

**为什么"不写"是可接受的止血**：不写只损失一条缓存条目，下次重算即可；
写错则把错误 GVA 挂到别的 key 上，**污染池内容且难以察觉**。两者代价不对称。

**基线**：`releases/v0.30.0` @ `06b685f346ad525027d03cffe80d58a491a13081`（pristine）。
止血补丁**直接可打在 pristine 上**（不需要先有诊断插桩，也不包含任何插桩）。

---

## 2. 止血的确切语义

### 2.1 per-group base 状态（只用精确证据，不做长度推断）

| 状态 | 含义 | 判定方式 |
|---|---|---|
| `ZERO` | 该组的表**就是**权威全表（从请求 block 0 起） | `list(ids) == list(full_ids)` **逐元素相等** |
| `WINDOW` | 该组表是权威全表的**后缀窗口** | `ids[0]` 出现在 `full_ids` 里（**不做长度推断**） |
| `UNPROVABLE` | 其余（无全表 / 组数不齐 / 空组 / 定位不到） | 兜底 |

`full` 的来源 = 新增的 `_full_block_ids[req_id]`，在 `update_state_after_alloc` **每次分配块时无条件刷新**（不再只在 `num_external_tokens > 0` 时记录）。

### 2.2 preempted 分支（定义法，不查表）

被抢占请求的表 = **本步新分配的块**，起点 = 包含 `num_computed_tokens` 的块边界：
`num_computed_tokens <= 0` ⇒ `ZERO`；否则 ⇒ `WINDOW`。

### 2.3 请求级判据（每步一次，放在 `ReqMeta.from_request_tracker`）

| `state` | `_STOP_BLEED=1`（默认） | `_STOP_BLEED=0` |
|---|---|---|
| 全组 `ZERO` | 原路（**与修复前逐位一致**） | 原路 |
| 含 `WINDOW` | `skip_save=True` → 本步不保存 | 原路（可能 fatal/写错） |
| 含 `UNPROVABLE` | `skip_save=True` | 原路 |
| `None`（未接线） | `skip_save=True`（**fail-closed**） | 原路 |

判定点插在 `if skip_save and load_spec is None: return None` **之前**，因此：
- 纯保存请求被跳 ⇒ 直接不产生 `ReqMeta`（与"本来就不需要保存"完全一致）；
- 带 load 的请求 ⇒ 仍产生 `ReqMeta(can_save=False)`，**load 不受影响**。

### 2.4 三处接线（谁会被判定）

| 分支 | 判定函数 | 说明 |
|---|---|---|
| `_process_new_request` | `_resolve_base_state_by_group(req_id, block_ids_by_group)` | 与 admission 记录的全表对照 |
| `_process_preempted_cached_request` | `_resolve_preempted_base_state(num_computed_tokens)` | 定义法，不依赖全表 |
| `_process_async_load_request` | `_resolve_base_state_by_group(request_id, block_ids)` | 同上 |
| `_process_running_cached_request` | **不判定** | 它复用前一步 tracker，状态随之继承（表仍从 block 0 起则保持 `ZERO`） |

### 2.5 覆盖性依据（**明确前提，评审必读**）

止血的覆盖范围 = **"块表不是权威全表"这一类**。推理链：

1. `ZERO` ⇒ 该组表与权威全表**逐元素相同** ⇒ 数组索引 == 绝对块号 ⇒
   该步 range 与数组**同基**（正是修复前的正常路径，因此行为逐位一致）；
2. 非 `ZERO`（`WINDOW` / `UNPROVABLE`）⇒ 索引与绝对块号可能错位 ⇒ **跳过**；
3. 现场两次越界的形态（表长 `63/127/128` vs range 上限 `224/288`）说明
   **range 的块号尺度远大于表的块数** —— 这类请求的表必然是"非权威全表"，会被判 `WINDOW`/`UNPROVABLE` 而跳过。

> ⚠️ **前提（尚未在真实负载实测）**：`ZERO` 的请求不会出现 `save_end_block > len(表)`。
> 若现场出现"表与全表逐元素相同、却仍然越界"，那根因**不是窗口**而是
> "range 或表容量的换算本身有错"，**本止血不覆盖**，需要另行定位。
>
> **怎么确认前提成立**（压测/复现后两条命令即可）：
> ```bash
> grep -c 'does not cover requested block range' output.log   # 期望 0（止血生效）
> grep -o 'mode=[A-Z]*' output.log | sort | uniq -c           # 跳过原因分布
> ```
> 只要 `does not cover` 归零、且跳过都记为 `WINDOW`/`UNPROVABLE`，前提即成立。

---

## 3. 从 `de0375537` 删减清单（逐条可核对）

### 3.1 `pool_scheduler.py`（修复 +64 → 止血 **+58**）

| 处置 | 内容 |
|---|---|
| **改写** | `_resolve_base_state_by_group`：**只返回 states**，删掉 `bases`（`i0 * bs`）计算 —— 止血不做对齐保存，base 没有消费者 |
| **改写** | `_resolve_preempted_base_by_group` → `_resolve_preempted_base_state`：删掉 per-group `bs` 与 bases，只返回状态 |
| **必改** | `self.num_kv_cache_groups` → `len(self.kv_cache_group_ids)`。**该属性在 pristine 中不存在**（全仓 grep 无命中），修复版这里会 `AttributeError`；详见 §10 |
| **新增** | `_full_block_ids` 字典（修复版没有：它直接用 `_unfinished_requests` 的 admission 快照，而那张"快照"在无命中时是空表） |
| **新增** | `finished_req_ids` / `preempted_req_ids` 两处 `_full_block_ids.pop(...)`（防字典无限增长） |
| 无删行 | 唯一被替换的字面量是两处 `local_block_ids = [[] ...]` 的**保留不动**（异步 load 仍用它），只在其后追加登记 |

### 3.2 `metadata.py`（修复 +144 → 止血 **+63**）

| 处置 | 内容 |
|---|---|
| **删** | 开关 `VLLM_ASCEND_KVPOOL_WINDOW_BASE`、`VLLM_ASCEND_KVPOOL_WINDOW_BASE_OBS` → 只留 `VLLM_ASCEND_KVPOOL_STOP_BLEED`（默认 `1`） |
| **删** | `_KVPOOL_UNSET` 哨兵、`_KVPOOL_PARTIAL_REFUSED`、`layerwise_offload=True` 时的 ERROR 拒绝逻辑（止血不启用对齐保存，无需拒绝） |
| **删** | `RequestTracker` 的 `save_window_base_by_group` / `save_window_start_token` / `save_window_pending_token` / `save_window_last_can_save`（只留 `save_window_base_state`） |
| **删** | `ReqMeta` 的 4 个新字段（`save_block_ids_by_group_np` / `save_block_gvas_by_group_np` / `save_window_base_by_group` / `save_window_base_state`）——止血版 worker 不用它们 |
| **删** | 5 分支判据（`OFF/ZERO/SKIP_UNPROVABLE/SKIP_P1_OFF/WINDOW_ALIGNED`）→ **1 条判据**（全 `ZERO` 才放行） |
| **删** | 窗口游标"延迟一步提交"、`save_start_token = max(游标, min_base)` —— `save_start_token=previous_saved_tokens` **原样保留** |
| **改** | `[KVPOOL-BASE]` 逐步观测 → **一行** `[KVPOOL-STOP]`，**只在跳过时**输出（WARNING） |

### 3.3 零改动文件

| 文件 | 修复版改动 | 止血版 |
|---|---|---|
| `kv_transfer.py` | +39 / −2（对齐数组 + 跳过 range） | **0** —— 覆盖断言 `raise` 原样保留 |
| `pool_worker.py` | +84 / −3（逐组夹紧 + 前导 0 补齐 + 组数自检） | **0** |

**结论**：止血版 **2 文件、+121 / −0、零删除、零行为改动**（除新增的"跳过"判定）。

---

## 4. 逐处具体代码

### 4.1 `pool_scheduler.py`

**① `__init__`：新增全表登记（锚点：`self._unfinished_requests ...` 之后）**

```python
        self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
        # v4-stop-bleed: request_id -> authoritative block table (based at the
        # request's block 0), recorded for every request, not only for pool
        # hits. Kept separate from _unfinished_requests so the historic
        # load-spec / async-load semantics stay untouched.
        self._full_block_ids: dict[str, list[list[int]]] = {}
        self._loading_req_ids: set[str] = set()
```

**② 新增两个判定函数（锚点：`update_state_after_alloc` 之前）**

```python
    def _resolve_base_state_by_group(self, req_id: str, ids_by_group) -> list[str] | None:
        """Per-group save-window state: "ZERO" | "WINDOW" | "UNPROVABLE".

        "ZERO" means the request's block table *is* the authoritative table,
        i.e. it starts at the request's block 0 -- the only state in which the
        legacy table-indexed save arrays are valid. Any other state means the
        table is a suffix window (or cannot be identified), so saving it with
        the legacy arrays would index it as if it started at block 0; such a
        request must not be saved (stop-bleed).
        """
        full = self._full_block_ids.get(req_id)
        if not full or ids_by_group is None or len(ids_by_group) != len(full):
            n = 0 if ids_by_group is None else len(ids_by_group)
            return ["UNPROVABLE"] * n
        states: list[str] = []
        for ids, full_ids in zip(ids_by_group, full):
            if list(ids) == list(full_ids):
                states.append("ZERO")
            elif ids and ids[0] in full_ids:
                states.append("WINDOW")
            else:
                states.append("UNPROVABLE")
        return states

    def _resolve_preempted_base_state(self, num_computed_tokens: int) -> list[str]:
        """Base state for the preempted branch, known by construction.

        A preempted cached request's table is exactly the blocks allocated in
        this step, so it starts at the block boundary that contains
        num_computed_tokens: "ZERO" only when that boundary is block 0.
        """
        if num_computed_tokens <= 0:
            return ["ZERO"] * len(self.kv_cache_group_ids)
        return ["WINDOW"] * len(self.kv_cache_group_ids)
```

**③ `update_state_after_alloc`：无条件刷新权威全表**

```python
        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        # v4-stop-bleed: record the authoritative block table unconditionally
        # (refreshed on every allocation). The save path needs it to prove that
        # a table starts at the request's block 0; with the historic guard
        # (num_external_tokens > 0) there was no table at all for requests that
        # missed the pool, which would leave them unprovable forever.
        self._full_block_ids[request.request_id] = (
            local_block_ids
            if num_external_tokens > 0
            else normalize_block_ids_by_group(blocks.get_block_ids())
        )
```

> 复用 `local_block_ids`（命中时二者字面等价）避免重复物化；`num_external_tokens` 仍照旧驱动下方 load-spec 逻辑，**未改动**。

**④ 三处 tracker 接线（各自 `previous_tracker = ...` 之后 + `block_sizes=...` 之后）**

```python
# _process_new_request
        _stop_state = self._resolve_base_state_by_group(request.req_id, block_ids_by_group)
# _process_preempted_cached_request
        _stop_state = self._resolve_preempted_base_state(request_real.num_computed_tokens)
# _process_async_load_request
        _stop_state = self._resolve_base_state_by_group(request_id, block_ids)
```
并在各自的 `RequestTracker(...)` 末尾追加：
```python
            save_window_base_state=_stop_state,
```

**⑤ 生命周期回收（`finished_req_ids` / `preempted_req_ids` 两个循环）**

```python
            self._full_block_ids.pop(finished_req_id, None)
...
            self._full_block_ids.pop(req_id, None)
```

### 4.2 `metadata.py`

**① `import os`（`import math` 之后）**

**② 模块级开关（`is_kv_save_role` 之后）**

```python
_STOP_BLEED = os.getenv("VLLM_ASCEND_KVPOOL_STOP_BLEED", "1") == "1"
```
（含一段说明注释，见 §5 补丁）

**③ `RequestTracker`：新增字段 + 构造参数 + 赋值**

```python
    # v4-stop-bleed: per-group base state, "ZERO" | "WINDOW" | "UNPROVABLE".
    # "ZERO" for every group is the only state in which the legacy
    # table-indexed save arrays are valid. None => not wired: not saveable.
    save_window_base_state: list[str] | None = None
```

**④ `ReqMeta.from_request_tracker`：唯一判据（插在 `if skip_save and load_spec is None:` 之前）**

```python
        if _STOP_BLEED and not skip_save:
            _base_state = getattr(tracker, "save_window_base_state", None)
            _stop_mode = "UNKNOWN"
            if _base_state:
                if all(s == "ZERO" for s in _base_state):
                    _stop_mode = "ZERO"
                elif any(s == "WINDOW" for s in _base_state):
                    _stop_mode = "WINDOW"
                else:
                    _stop_mode = "UNPROVABLE"
            if _stop_mode != "ZERO":
                skip_save = True
                logger.warning(
                    "[KVPOOL-STOP] layerwise save skipped mode=%s req=%s state=%s table=%s "
                    "save_end_token=%d target=%d",
                    _stop_mode,
                    tracker.req_id,
                    _base_state,
                    [len(x) for x in tracker.allocated_block_ids_by_group],
                    num_tokens_to_save,
                    target_token_len,
                )
```

> 放在早退之前 ⇒ 止血跳过与"本来就跳过"走**同一条路径**；
> `if not skip_save:` 里的 `tracker.num_saved_tokens = max(...)` 不会被推进 ⇒ 不会留下"已保存"的假进度。

### 4.3 未改动

`kv_transfer.py`、`pool_worker.py` **零改动**；探针（L1–L5 / `[KVPOOL]` 10 条 / `KVPFIX` 打点）**全部不包含**。

---

## 5. 完整补丁

**① 从本文抽出补丁并自证**（**推荐用命令抽，不要手工复制** —— diff 里空上下文行是以单个空格开头的行，手工复制极易被编辑器吃掉）

```bash
# 在本文所在目录执行；取第一个 ```diff 代码块
awk '/^```diff$/{f=1;next} /^```$/{if(f)exit} f' 止血代码.md > kvpool_stop.diff

wc -l   kvpool_stop.diff   # 期望 247
md5sum  kvpool_stop.diff   # 期望 9c771bbe00d9ab47d867ca931a493c69
head -3 kvpool_stop.diff   # 期望以 "diff --git a/...metadata.py" 开头
tail -1 kvpool_stop.diff   # 期望是 "         meta = AscendConnectorMetadata("
```

**② 自证数字**（与 §6 一致）

```bash
git diff --numstat        # 期望 63/0 (metadata) 与 58/0 (pool_scheduler)
git diff | grep -E '^-' | grep -v '^--- '   # 期望无输出（零删除）
git apply --check -R kvpool_stop.diff       # 期望 exit 0
```

```diff
diff --git a/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py b/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py
index 4f20d4fb3..f1e355535 100644
--- a/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py
+++ b/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py
@@ -1,6 +1,7 @@
 from __future__ import annotations

 import math
+import os
 from collections.abc import Callable, Iterable, Mapping, Sequence
 from dataclasses import dataclass, field
 from itertools import compress
@@ -22,6 +23,27 @@ def is_kv_save_role(kv_role: str, consumer_is_to_put: bool) -> bool:
     return kv_role in ("kv_producer", "kv_both") or consumer_is_to_put


+# ---------------------------------------------------------------------------
+# v4-stop-bleed
+#
+# Root cause of "ReqMeta GVA metadata does not cover requested block range":
+# the save block range is built from absolute block numbers while the per-group
+# GVA array is indexed from the request's table itself. The two only agree when
+# the table starts at the request's block 0. When it does not (the table is a
+# suffix window, e.g. after a preemption), the save either trips the coverage
+# assertion in build_shared, or -- when the range happens to fit -- stores the
+# KV of block (base + k) under the key of block k.
+#
+# Stop-bleed keeps the legacy save path untouched and simply refuses to save a
+# request whose table cannot be proven to be the authoritative one. Losing a
+# cache entry is recoverable; poisoning the pool is not.
+#
+#   VLLM_ASCEND_KVPOOL_STOP_BLEED=0 -> restore the historic behaviour exactly
+#                                     (which can trip the assertion again).
+# ---------------------------------------------------------------------------
+_STOP_BLEED = os.getenv("VLLM_ASCEND_KVPOOL_STOP_BLEED", "1") == "1"
+
+
 @dataclass(frozen=True)
 class TPMismatchInfo:
     enabled: bool
@@ -901,6 +923,11 @@ class RequestTracker:

     block_sizes: list[int] | None = None

+    # v4-stop-bleed: per-group base state, "ZERO" | "WINDOW" | "UNPROVABLE".
+    # "ZERO" for every group is the only state in which the legacy
+    # table-indexed save arrays are valid. None => not wired: not saveable.
+    save_window_base_state: list[str] | None = None
+
     def __init__(
         self,
         req_id: str,
@@ -919,6 +946,7 @@ class RequestTracker:
         last_layerwise_snapshot_end_token: int | None = None,
         last_layerwise_snapshot_block_index: int | None = None,
         last_layerwise_snapshot_start_block: int | None = None,
+        save_window_base_state: list[str] | None = None,
     ) -> None:
         self.req_id = req_id
         self.token_len = token_len
@@ -938,6 +966,7 @@ class RequestTracker:
         self.last_layerwise_snapshot_end_token = last_layerwise_snapshot_end_token
         self.last_layerwise_snapshot_block_index = last_layerwise_snapshot_block_index
         self.last_layerwise_snapshot_start_block = last_layerwise_snapshot_start_block
+        self.save_window_base_state = save_window_base_state

     @property
     def allocated_block_ids(self) -> list[int]:
@@ -1211,6 +1240,40 @@ class ReqMeta:
         # normally.
         if load_spec is not None and load_spec.can_load and not save_partial_block:
             skip_save = True
+
+        # ---------------- v4-stop-bleed predicate --------------------------
+        # The legacy save path indexes the per-group GVA arrays as if the
+        # request's table started at its block 0. That only holds when the
+        # table IS the authoritative table for every group (state "ZERO").
+        # Anything else -- a suffix window, or a table we cannot identify --
+        # would either trip the coverage assertion in build_shared or, when the
+        # range happens to fit, silently store the KV of block (base + k) under
+        # the key of block k. Skip the save instead: not saving loses a cache
+        # entry, mis-saving poisons the pool. Placed before the early return
+        # below so a stop-bleed skip behaves exactly like any other skip.
+        if _STOP_BLEED and not skip_save:
+            _base_state = getattr(tracker, "save_window_base_state", None)
+            _stop_mode = "UNKNOWN"
+            if _base_state:
+                if all(s == "ZERO" for s in _base_state):
+                    _stop_mode = "ZERO"
+                elif any(s == "WINDOW" for s in _base_state):
+                    _stop_mode = "WINDOW"
+                else:
+                    _stop_mode = "UNPROVABLE"
+            if _stop_mode != "ZERO":
+                skip_save = True
+                logger.warning(
+                    "[KVPOOL-STOP] layerwise save skipped mode=%s req=%s state=%s table=%s "
+                    "save_end_token=%d target=%d",
+                    _stop_mode,
+                    tracker.req_id,
+                    _base_state,
+                    [len(x) for x in tracker.allocated_block_ids_by_group],
+                    num_tokens_to_save,
+                    target_token_len,
+                )
+
         if skip_save and load_spec is None:
             return None

diff --git a/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py b/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
index 947c3d8b5..5fb67491b 100644
--- a/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
+++ b/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
@@ -157,6 +157,11 @@ class KVPoolScheduler:
             "discard_partial_chunks", True
         )
         self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
+        # v4-stop-bleed: request_id -> authoritative block table (based at the
+        # request's block 0), recorded for every request, not only for pool
+        # hits. Kept separate from _unfinished_requests so the historic
+        # load-spec / async-load semantics stay untouched.
+        self._full_block_ids: dict[str, list[list[int]]] = {}
         self._loading_req_ids: set[str] = set()

         self._block_pool: BlockPool | None = None
@@ -770,6 +775,41 @@ class KVPoolScheduler:

         return need_to_allocate, self.load_async and not self.use_layerwise

+    def _resolve_base_state_by_group(self, req_id: str, ids_by_group) -> list[str] | None:
+        """Per-group save-window state: "ZERO" | "WINDOW" | "UNPROVABLE".
+
+        "ZERO" means the request's block table *is* the authoritative table,
+        i.e. it starts at the request's block 0 -- the only state in which the
+        legacy table-indexed save arrays are valid. Any other state means the
+        table is a suffix window (or cannot be identified), so saving it with
+        the legacy arrays would index it as if it started at block 0; such a
+        request must not be saved (stop-bleed).
+        """
+        full = self._full_block_ids.get(req_id)
+        if not full or ids_by_group is None or len(ids_by_group) != len(full):
+            n = 0 if ids_by_group is None else len(ids_by_group)
+            return ["UNPROVABLE"] * n
+        states: list[str] = []
+        for ids, full_ids in zip(ids_by_group, full):
+            if list(ids) == list(full_ids):
+                states.append("ZERO")
+            elif ids and ids[0] in full_ids:
+                states.append("WINDOW")
+            else:
+                states.append("UNPROVABLE")
+        return states
+
+    def _resolve_preempted_base_state(self, num_computed_tokens: int) -> list[str]:
+        """Base state for the preempted branch, known by construction.
+
+        A preempted cached request's table is exactly the blocks allocated in
+        this step, so it starts at the block boundary that contains
+        num_computed_tokens: "ZERO" only when that boundary is block 0.
+        """
+        if num_computed_tokens <= 0:
+            return ["ZERO"] * len(self.kv_cache_group_ids)
+        return ["WINDOW"] * len(self.kv_cache_group_ids)
+
     def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
         """
         Update KVConnector state after temporary buffer alloc.
@@ -782,6 +822,16 @@ class KVPoolScheduler:
             local_block_ids = normalize_block_ids_by_group(blocks.get_block_ids())

         self._unfinished_requests[request.request_id] = (request, local_block_ids)
+        # v4-stop-bleed: record the authoritative block table unconditionally
+        # (refreshed on every allocation). The save path needs it to prove that
+        # a table starts at the request's block 0; with the historic guard
+        # (num_external_tokens > 0) there was no table at all for requests that
+        # missed the pool, which would leave them unprovable forever.
+        self._full_block_ids[request.request_id] = (
+            local_block_ids
+            if num_external_tokens > 0
+            else normalize_block_ids_by_group(blocks.get_block_ids())
+        )
         if request.request_id not in self.load_specs:
             # No KV tokens from external KV cache, return
             logger.debug(
@@ -875,6 +925,7 @@ class KVPoolScheduler:
         request_real = request_tuple[0]
         block_ids_by_group = normalize_block_ids_by_group(request.block_ids)
         previous_tracker = self._request_trackers.get(request.req_id)
+        _stop_state = self._resolve_base_state_by_group(request.req_id, block_ids_by_group)
         request_tracker = RequestTracker(
             req_id=request.req_id,
             token_len=num_tokens_to_compute,
@@ -886,6 +937,7 @@ class KVPoolScheduler:
             gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
             num_speculative_blocks_by_group=self.num_speculative_blocks_by_group,
             block_sizes=self.grouped_block_size,
+            save_window_base_state=_stop_state,
         )
         self._request_trackers[request.req_id] = request_tracker
         return self._build_req_meta(
@@ -916,6 +968,7 @@ class KVPoolScheduler:
         request_real = request_tuple[0]
         num_tokens_to_compute = request_real.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
         previous_tracker = self._request_trackers.get(req_id)
+        _stop_state = self._resolve_preempted_base_state(request_real.num_computed_tokens)
         request_tracker = RequestTracker(
             req_id=req_id,
             token_len=num_tokens_to_compute,
@@ -927,6 +980,7 @@ class KVPoolScheduler:
             gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
             num_speculative_blocks_by_group=self.num_speculative_blocks_by_group,
             block_sizes=self.grouped_block_size,
+            save_window_base_state=_stop_state,
         )
         self._request_trackers[req_id] = request_tracker
         return self._build_req_meta(
@@ -1006,6 +1060,7 @@ class KVPoolScheduler:
         ):
             num_tokens_to_compute = num_tokens_to_compute + 1
         previous_tracker = self._request_trackers.get(request_id)
+        _stop_state = self._resolve_base_state_by_group(request_id, block_ids)
         request_tracker = RequestTracker(
             req_id=request_id,
             token_len=num_tokens_to_compute,
@@ -1016,6 +1071,7 @@ class KVPoolScheduler:
             gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
             num_speculative_blocks_by_group=self.num_speculative_blocks_by_group,
             block_sizes=self.grouped_block_size,
+            save_window_base_state=_stop_state,
         )
         self._request_trackers[request_id] = request_tracker
         return ReqMeta.from_request_tracker(
@@ -1042,12 +1098,14 @@ class KVPoolScheduler:
         for finished_req_id in scheduler_output.finished_req_ids:
             self._request_trackers.pop(finished_req_id, None)
             self._unfinished_requests.pop(finished_req_id, None)
+            self._full_block_ids.pop(finished_req_id, None)
             self._preempted_req_ids.discard(finished_req_id)

         for req_id in scheduler_output.preempted_req_ids:
             self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
             self._request_trackers.pop(req_id, None)
             self._unfinished_requests.pop(req_id, None)
+            self._full_block_ids.pop(req_id, None)
             self._loading_req_ids.discard(req_id)

         meta = AscendConnectorMetadata(
```

---

## 6. 已做验证（全部在容器 `yxf_vllm_b113` 的 pristine worktree 上实测）

| 项 | 结果 |
|---|---|
| `python3 -m py_compile`（两文件，python3.12） | **`SYNTAX_OK`** |
| `git diff --numstat` vs `06b685f346` | `metadata.py` **63/0**、`pool_scheduler.py` **58/0** ⇒ **+121 / −0** |
| 删除行 | **0 行**（`git diff \| grep '^-'` 无输出） |
| `git apply --check -R kvpool_stop.diff` | **`REVERSE_OK`**（与 worktree 差异逐字一致） |
| **单元自测**（从源码 AST 抽出两函数，脱离 vllm 依赖执行） | **10/10 PASS，`UNIT_FAILURES=0`** |

单测覆盖的用例（实际输出）：

```text
PASS equal -> all ZERO -> ['ZERO', 'ZERO', 'ZERO']
PASS suffix window -> WINDOW -> ['WINDOW', 'ZERO', 'WINDOW']
PASS head not found -> UNPROVABLE -> ['UNPROVABLE', 'ZERO', 'WINDOW']
PASS len mismatch -> UNPROVABLE -> ['UNPROVABLE', 'UNPROVABLE']
PASS empty group -> UNPROVABLE -> ['UNPROVABLE', 'ZERO', 'ZERO']
PASS ids None -> [] -> []
PASS no snapshot -> UNPROVABLE -> ['UNPROVABLE', 'UNPROVABLE', 'UNPROVABLE']
PASS preempted nct=0 -> ['ZERO']*6
PASS preempted nct=1 -> ['WINDOW']*6
PASS preempted nct=4096 -> ['WINDOW']*6
UNIT_FAILURES=0
```

**未做**：端到端跑真实负载（§7 观测即是为此准备）。

---

## 7. 落地

**方式 A：`git apply`（推荐，直接对 `releases/v0.30.0`）**

```bash
git fetch origin releases/v0.30.0
git checkout -b <your-branch> origin/releases/v0.30.0

# 基线必须是 06b685f346ad525027d03cffe80d58a491a13081；
# 若上游已前进，先确认落点区域未被改动（补丁上下文取自这两处函数附近）。
git rev-parse HEAD

git apply --check kvpool_stop.diff          # 期望 exit 0
git apply          kvpool_stop.diff
git add vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/{pool_scheduler,metadata}.py
git commit -s -m "fix(kvpool): skip layerwise save when the block table is not authoritative"
git log --oneline -1 && git show --stat HEAD
```

> 提交信息与 PR 描述见 §11。`-s` 是 sign-off（DCO）；若仓库另有要求，按仓库规范执行。

**方式 B：在现网节点上直接生效**

> ⛔ **执行前必须确认该节点上没有正在跑的压测/服务** —— 这一步会改工作区文件并要求重启引擎。
> 仅"生成补丁、抽取、自证"（§5 ①②）是纯读操作，可随时代做。

```bash
cd /vllm-workspace/vllm-ascend
git checkout -b dev_stopbleed 06b685f346       # 从 pristine 起，不带诊断插桩
git apply /tmp/kvpool_stop.diff
python3 -m py_compile \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py && echo SYNTAX_OK
# 重启整个引擎（EngineCore 与 Worker 都会 import 这两个模块）
```

> **生效前提**：`pool_scheduler` / `metadata` 在 **EngineCore 与 Worker 两个进程**都会被 import，
> 因此必须**重启整个引擎**（只重启 worker 不够）；日志级别需 ≤ WARNING 才能看到 `[KVPOOL-STOP]`。

---

## 8. 回退

| 方式 | 操作 |
|---|---|
| 运行期（最快） | `VLLM_ASCEND_KVPOOL_STOP_BLEED=0` ⇒ **逐位回到修复前行为**（可能再次 fatal/写错） |
| 代码级 | `git revert <commit>` 或 `git checkout -- <两文件>` 后重启引擎 |
| 分支级 | `git checkout <原分支>` 后重启引擎 |

---

## 9. 代价与观测（现场要点）

止血**唯一**的行为变化是"某些步不保存"，因此观测只有一个抓手：

```bash
grep -c '\[KVPOOL-STOP\]' output.log                       # 被跳过的保存次数
grep -o 'mode=[A-Z]*' output.log | sort | uniq -c          # 跳过原因分布
grep -c 'does not cover requested block range' output.log  # 期望 0
grep 'does not cover' output.log | grep -c 'is_save=False' # 若 >0 ⇒ 触发在 load 侧（见 §10）
```

| 观测 | 含义 | 处置 |
|---|---|---|
| `does not cover ...` = 0 | 止血成功 | 继续观察池命中率是否可接受 |
| `KVPOOL-STOP` 次数很少、`mode=WINDOW` 为主 | 正常：确实只有少量窗口请求 | 结论：止血代价极小 |
| `KVPOOL-STOP` 大量且 `mode=UNPROVABLE` 为主 | **红线**：全表登记与请求表结构不匹配（长度/分组），导致大量请求被判不可保存。**2026-10-23 实际触发过（1832 次全量误杀）** | 取 2–3 条日志，比对 `table` 列与实际块表；最终处置见 §12.4（new 分支直接判 `ZERO`） |
| `mode=UNKNOWN` 出现 | 该 tracker 未接线（理论上只有占位 tracker 会走到） | 定位该 req 的创建路径 |

**性能**：每次 `update_state_after_alloc` 多一次 `blocks.get_block_ids()` 物化（O(块数)）；判定为 O(组数 × 表长)。相对单步推理开销可忽略。

---

## 10. 已知边界（如实说明）

| # | 项 | 说明 |
|---|---|---|
| 1 | **只覆盖 save 侧** | 覆盖断言在 save/load 两条取数路径上都会触发。止血只判 save；若 fatal 出现在 load（日志里 `is_save=False`），本补丁不覆盖 |
| 2 | **窗口请求不写池** | 这是止血的定义，不是缺陷；若要"窗口也写"，须上 `de0375537`（且先解决 V5，见下） |
| 3 | **fail-closed** | `state is None` 也判为不可保存（宁可少写，不可写错）。若某条创建路径漏接，表现为"该路径请求不保存"，不会崩 |
| 4 | 依赖 `update_state_after_alloc` 每次分配都被调用 | 该函数是唯一刷新点；若某步分配了块却没调用，那一步会判 `UNPROVABLE`（只跳不做错） |
| 5 | 未改 load 侧对称缺陷 | `_prepare_load_gvas_for_request` 仍按 hash 数夹紧（与止血范围无关） |
| 6 | 未改 `_build_key_major_shared` | `use_key_major_ranges=True` 的路径未纳入 |
| 7 | 端到端未验证 | 已做 `py_compile` + 10 例单测 + 反向 apply。**2026-10-23 已在两机实测（§12）：v1 触发 UNPROVABLE 全量误杀红线，v3 修复后验证通过** |
| 8 | **顺带发现（重要）** | 冻结的修复 `de0375537` 在 `_resolve_preempted_base_by_group` 里用了 `self.num_kv_cache_groups`，而 **pristine 全仓无此属性**（`grep` 无命中）⇒ 首次遇到 preempted 请求即 `AttributeError`。止血版本已改用 `len(self.kv_cache_group_ids)`。**修复版解冻前必须先修掉这一处** |
| 9 | 关系说明 | 止血**不能**替代修复：它只是"停止出血"。修复解冻前仍需闭环 `dev_fix_p0p1` 的 V1–V6（尤其 V5：无命中请求的 admission 快照为空表） |

---

## 11. 提 PR 文案

**commit message**

```
fix(kvpool): skip layerwise save when the block table is not authoritative

The layerwise save path builds its block range from absolute block numbers
while the per-group GVA arrays are indexed from the request's own block
table. The two only agree when that table is the authoritative table of the
request, i.e. it starts at the request's block 0. When it does not -- e.g.
after a preemption leaves only a suffix window -- the range and the array
disagree and the coverage assertion fires:

  RuntimeError: ReqMeta GVA metadata does not cover requested block range

Worse, when the range happens to fit the array length the assertion does not
fire at all, and the KV of absolute block (base + k) is stored under the key
of absolute block k, silently poisoning the pool.

Record, per request, the authoritative block table at allocation time and
derive a per-group state from exact evidence only (the table equals the
authoritative table, or its head block id is found in it; the preempted
branch derives it from num_computed_tokens). Only an all-"ZERO" table -- the
one case where the legacy indexing is valid -- is saved; anything else skips
the save for that step:

  - pool_scheduler: keep the authoritative block table per request
    (_full_block_ids, refreshed on every allocation and dropped with the
    request) and resolve the per-group state for the new / preempted /
    async-load branches.
  - metadata: a single predicate in ReqMeta.from_request_tracker turns a
    non-authoritative table into skip_save, placed before the existing early
    return so the skip follows the normal path.

The legacy save path itself is untouched and no array layout is changed:
requests whose table is authoritative keep the previous behaviour
byte-for-byte. Not saving loses one cache entry; mis-saving poisons the pool.
Set VLLM_ASCEND_KVPOOL_STOP_BLEED=0 to restore the historic behaviour.

Signed-off-by: <your name> <your email>
```

**PR 描述要点**

1. **症状与调用链**：`pre_forward → prepare_layerwise_step → process_layer_data → _build_shared_save_data → build_shared_data → build_shared` 抛 `ReqMeta GVA metadata does not cover requested block range`；
2. **为什么不能只加日志**：断言只在越界时保护；range 恰好放得下时**静默写错**（`base+k` 的 KV 挂到 `k` 的 key 下）——这条引用 §0；
3. **根因一句话**：range 用绝对块号、数组用表自身定基，两者只在"表 == 权威全表"时一致；
4. **本 PR 的取舍**：不改保存逻辑，只把"表非权威全表"的请求**本步设为不保存**；代价是这些请求暂不写池（丢缓存条目，可重算）；
5. **兼容性**：`ZERO` 路径与关闭开关时**逐位不变**；`VLLM_ASCEND_KVPOOL_STOP_BLEED` 默认 `1`；
6. **验证**：`py_compile` + 10 例单元自测 + `git apply --check -R`；现场观测用 `[KVPOOL-STOP]` 的 `mode` 分布与 `does not cover` 是否归零（§6、§9）；
7. **已知边界请评审确认**：只覆盖 save 侧（§10-1）、窗口请求暂不写池（§10-2）、`fail-closed`（§10-3）、覆盖性前提（§2.5）。

---

## 12. 现场部署与修复记录（2026-10-23，当前生效版本）

> 本节记录 v1 补丁上线后的**实测表现**、触发红线后的**根因分析**、一次错误修复尝试、
> 以及最终 **v3 修复**与验证数据。§2–§11 的 v1 方案在 §12.4 之前已被现场数据推翻其"登记表比对"前提。

### 12.1 时间线

| 时间 | 事件 | commit |
|---|---|---|
| 10-22 16:10 | GitHub PR #19533 创建（base `releases/v0.30.0`） | head `544121ff` |
| 10-23 01:15 | 两机容器打补丁（本地化版本，含单测调整、`UNTRACKED` 语义） | P `2c181701d` / D `4558bb668` |
| 10-23 01:22–01:24 | P（4 引擎）、D（16 引擎）重启，加载该版本 | — |
| 10-23 01:30 | **红线触发**：`mode=UNPROVABLE` 从第一个 chunk 起全量出现 | — |
| 10-23 01:59 | 错误修复尝试（见 §12.3） | `8489d7afa` |
| 10-23 02:03 | 回滚错误修复（`git reset`），两机回到 `2c181701d` | — |
| 10-23 02:1x | 正确修复：new 分支直接判 `ZERO`（§12.4） | P `e69705578` / D `8879bb3c5` |
| 10-23 02:23 / 02:31 | P、D 依次重启生效 | — |
| 10-23 02:28 / 02:37 | P 4/4、D 16/16 健康；验证通过（§12.5） | — |

### 12.2 v1 实测：UNPROVABLE 红线（§9 预言成真）

重启后（01:22 起）外部流量涌入，日志表现为：

```text
WARNING [metadata.py:1257] [KVPOOL] layerwise save skipped mode=UNPROVABLE
  req=chatcmpl-... state=['UNPROVABLE' × 6]
  table=[65, 3, 257, 257, 4097, 1025]  save_end_token=4096  target=8188
```

关键数据：

- 重启前旧进程（诊断版）期间：**0 次** `does not cover`；
- v1 生效期间：**1832 次** skip，**100% 为 `mode=UNPROVABLE`**，`WINDOW` 一次都没有；
- 被跳过的请求表长从 65 块开始、每步 +63/+64 正常增长，`save_end` 始终在表长范围内——
  这些是**本应判 `ZERO` 的完整表请求**，被误杀；
- 后果：所有长请求池化写入停摆，但不崩、不写错（止血语义本身没破）。

### 12.3 红线的两个根因

**根因一：`update_state_after_alloc` 的登记表在本环境不可用（v1 补丁的致命假设）**

v1 的登记逻辑是：

```python
self._full_block_ids[request.request_id] = (
    local_block_ids
    if num_external_tokens > 0
    else normalize_block_ids_by_group(blocks.get_block_ids())
)
```

实测：无命中请求走 else 分支时，钩子里 `blocks.get_block_ids()` 登记下来的参照与
`_process_new_request` 里 `NewRequestData.block_ids` 构成的表**首块对不上**，
逐组 `ids[0] in full_ids` 全部失败 → 六组全 `UNPROVABLE`。
登记表作为"独立权威参照"的前提**不成立**。

**根因二：v1 的"自比较"设计本就无法产生正信号（评审已预警）**

即便登记表可用，`NewRequestData.block_ids` 与 `get_blocks()` 同源于同一调度步的
`req_to_new_blocks[request_id]`，比对恒等、只会返回 `ZERO`——这个判定在 new 分支
永远不可能发现窗口。**真正有判定能力的只有 preempted 定义法**。

**错误修复尝试（`8489d7afa`，已回滚）**：把登记改成
`normalize_block_ids_by_group(request.block_ids)`——但 `update_state_after_alloc` 的
`request` 是 `vllm.v1.request.Request`，**没有 `block_ids` 属性**（N1 已实测），
该改动会 `AttributeError` 崩调度器，故在 02:03 被 `git reset` 回滚。

### 12.4 v3 修复（当前两机运行版本）

思路：承认"new 分支的表天生就是权威全表"（vLLM 块表恒从下标 0 起，前缀命中占头部下标），
**不再依赖登记表比对**；登记表降级为仅供 async-load 分支使用的参照。

```python
# _process_new_request（pool_scheduler.py，v3 生效代码）
block_ids_by_group = normalize_block_ids_by_group(request.block_ids)
previous_tracker = self._request_trackers.get(request.req_id)
# vLLM 的块表恒从下标 0 开始：NewRequestData.block_ids 就是本步全表，
# 旧保存索引方式对它必然成立。不再依赖准入钩子的登记表比对——
# 该比对在本环境会拿到空参照表，导致所有请求被判 UNPROVABLE。
_stop_state = ["ZERO"] * len(block_ids_by_group)
self._full_block_ids[request.req_id] = block_ids_by_group
```

其余全部保持 v1：

- preempted 分支：`_resolve_preempted_base_state`（`num_computed>0 → WINDOW`）不动，
  这是拦截 run C 窗口形态的唯一有效判据；
- running 分支：继承 tracker 状态（`update()` 只 append，基准不变）；
- async-load 分支：保留比对（`use_layerwise=true` 下不活跃）；
- `metadata.py` 闸门：不动（两机为 `UNTRACKED` 语义：`None` 不跳过，仅 `WINDOW/UNPROVABLE` 跳过）。

### 12.5 v3 验证数据（两机重启后）

| 指标 | 旧进程（v1，pid=5201xx/5262xx） | 新进程（v3，pid=127x） |
|---|---|---|
| `layerwise save skipped` | 1832 次 | **0 次** |
| `mode=UNPROVABLE` | 1832 次 | **0 次** |
| `GVA metadata does not cover` | 0 | **0** |
| External prefix cache hit rate（P） | 停写期持续走低 | **恢复 44–46%** |

旁证（池化读写闭环）：

- 读：新进程 `kvpool hit tokens` 出现大命中（`692817 → 688128`，命中 99.3%）；
  D 侧 External prefix cache hit rate 100%；
- 写：跳过归零 + P 侧外部命中率回升（外部命中只能来自池内数据，证明写入恢复）；
- 端到端：测试请求经 1999 代理正常返回。

### 12.6 v3 已知边界与待办

| # | 项 | 说明 |
|---|---|---|
| 1 | **窗口拦截未经实测** | 目前只有"正常请求恢复写入"的证据；`WINDOW` 分支（run C 形态）还没有被真实抢占流量触发验证过。需构造高压力+抢占场景复测 |
| 2 | new 分支 fail-open | v3 对 new 分支不再做独立校验（恒 `ZERO`）。若未来出现非抢占路径的窗口表，将漏判；建议加只读探针观测 `tbl_first != 全表首块` 的异常 |
| 3 | 与上游不同步 | 两机运行的是本地分支 `dev_pr19533`（v3），GitHub PR #19533（head `544121ff`）仍是 v1 语义。若要合入上游，需把 v3 改动（含 `UNTRACKED` 语义）同步到 PR 分支 |
| 4 | `UNTRACKED` vs `UNKNOWN` | 两机本地版对 `state=None` 不跳过（fail-open），PR 版跳过（fail-closed），两边语义需统一 |
| 5 | load 侧对称缺陷 | 同 §10-5，本补丁不覆盖 |