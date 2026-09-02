# Layerwise 协议返工方案（vllm-ascend PR #15367）

## 1. 背景与前因后果

### 1.1 问题起源

ascend_store 的 GVA layerwise 门禁表达式 `use_layerwise and backend == "memcache"` 曾散落在 worker / scheduler / layout / connector 多处各自派生。重复的派生已经造成过一次线上回归：

- PR #14465 将 connector 侧的副本当作死代码删除，但读者仍在，导致 MultiConnector（PD 分离）初始化崩溃；
- PR #15291 以 hotfix 恢复了该 flag，形成同一派生的第 5 份副本。

### 1.2 本 PR 的初衷与做法

PR #15367（分支 `refactor_layerwise_part1`，7 个 commit）目标是把上述派生收敛到单一入口：

- `backend/__init__.py` 的 `backend_map` 为 memcache 增加 `layerwise_protocol` 字段（指向新文件 `backend/gva_protocol.py` 的模块路径），并提供 `get_layerwise_protocol()` resolver；
- GVA key 构造集中到 `gva_protocol.py` 的 `GVAKeyFactory`（worker 的 full/partial key 与 scheduler 的 hit-check key 原本两处拼写，可漂移）；
- `kv_transfer.py` 两个 layerwise 线程入口新增 `assert isinstance(self.m_store, MemcacheBackend)`；
- connector 的 slot-release 门禁下沉到 `KVPoolWorker`，connector 变为纯转发，替换 #15291 的 hotfix。

### 1.3 检视意见

Pz1116 于 2026-09-01 给出 7 条行内意见（review 5074771275）：

| # | 位置 | 意见 |
|---|------|------|
| 1 | `backend/gva_protocol.py` | 这部分应移到 memcache backend 里，而不是在 backend 目录下新建文件 |
| 2 | `ascend_store_connector.py` `set_external_slot_release_waiter` | 应删掉注释中的 GVA 表述，GVA 是后端专属概念 |
| 3 | `kv_transfer.py` `LayerSendingThread._handle_request` | 不应在此断言 MemcacheBackend，mooncake layerwise 之后也会走类似路径 |
| 4 | `kv_transfer.py` `LayerRecvingThread._handle_request` | 同上 |
| 5 | `pool_scheduler.py` | `use_gva_layerwise` 仍然存在于这里 |
| 6 | `pool_worker.py` | `self.layerwise_protocol.GVAKeyFactory` 也是后端专属的 |
| 7 | （整体） | 通用层不应携带后端专属知识 |

另有 gemini-code-assist bot 两轮意见：`backend_supports` 名称归一化（已采纳，16bdb52f）；两处 `assert` 改 `TypeError`（未采纳，理由为模块内 assert 是既有惯例且 mypy 收窄等价）；`set_external_slot_release_waiter` 存在初始化顺序风险（waiter 晚于线程启动设置时，已运行线程持有的引用仍为 None，待处理）。

检视意见经沟通后，整体修改方向已获 Pz1116 确认。

## 2. 对检视意见的理解

核心：**GVA 与 layerwise 是两个正交维度，PR 的前提"GVA 是 memcache 专属协议，把它绑到 memcache 上"定得太强。**

- layerwise 是传输粒度/调度维度的通用模式，由 connector `extra_config` 的 `use_layerwise` 开启，概念上与后端无关；
- GVA 是 memcache backend 的寻址/传输机制，是后端内部实现细节，不是 layerwise 的同义词；
- mooncake 后续实现 layerwise 时，复用的是同一套通用框架（`kv_transfer.py` 的 layerwise 线程 + `base.py` 的五个 batch 接口），只是由 `MooncakeBackend` 实现自己的 store 方法和 key 方案。

推论：通用层（`pool_worker.py` / `pool_scheduler.py` / `layerwise_cache_layout.py` / `ascend_store_connector.py` / `kv_transfer.py`）不应出现任何 GVA / MemcacheBackend 字样——包括变量名、类型断言、注释与 docstring。后端专属的只剩两样东西：key 格式、opt-in 门禁语义，二者都应收在 backend 模块内部。

## 3. 代码库走读结论

基于本地仓库 `D:\lzy\project\kv_pool\code\vllm-ascend`（分支 `refactor_layerwise_part1`，头 commit `2a239d18a`）：

- `backend/base.py`：`batch_get_key_info` / `batch_alloc` / `batch_add_lease` / `batch_remove_lease` / `batch_write_finish` 五个接口声明在 `Backend` ABC 上，非抽象实现（基类桩）。
- `backend/memcache_backend.py`：唯一 override 这五个接口的后端。
- `backend/mooncake_backend.py`：无任何 layerwise/GVA 相关代码，**mooncake 目前没有 layerwise**——"mooncake 之后也会走这条路"指的是未来复用通用 layerwise 线程框架。
- `kv_transfer.py`：两个 layerwise 线程调用的 store 方法全部在上述基类 ABC 上，`m_store` 保持 `Backend` 类型即可通过静态检查；线程入口的 memcache 断言是本 PR 新增、非既有逻辑。
- `pool_worker.py:319-326` 与 `pool_scheduler.py:171-179`：worker（建 store）与 scheduler（`create_scheduler_client`）本就要通过 `backend_map["path"]` import 后端模块。因此 registry 的 `layerwise_protocol` 字段（第二条模块路径字符串）属于重复声明，会与 `path` 漂移。
- `backend/mooncake_backend.py:13`：模块顶层即 `from mooncake.store import ReplicateConfig`，import 该模块有第三方重依赖。resolver 不能为做能力检查而无条件 import 所有后端模块。
- `memcache_backend.py` 顶层仅依赖 torch/vllm 与 base（memcache 客户端在 `_setup_store` 内惰性 import），import 成本低。
- `pool_worker.py:1513/1538`（`kv_transfer.py` 同）：接收线程在用点读取 `self.external_slot_release_waiter`，线程启动后更新该属性即生效——gemini 的初始化顺序加固可以低成本落实。

## 4. 总体设计

### 4.1 设计原则

通用层只面对"layerwise 协议"这个中性抽象；协议由后端模块以**中性命名的模块级函数**承载；能力发现保持 registry 单一入口。

### 4.2 设计决策

**D1. 协议收进 memcache backend，模块级中性函数。**
`make_full_key` / `make_partial_key` / `make_hit_check_keys` / `extract_layout_config` 四个函数移入 `memcache_backend.py`；`GVAKeyFactory` 类名消失；`gva_protocol.py` 删除。key 字符串逐字节不变（线上 wire format，快照测试锁定）。

**D2. 能力发现：registry 字段改为布尔能力标记，resolver 经既有 `path` 返回模块本身。**
`backend_map` 的 memcache 条目将 `layerwise_protocol` 字段的值由模块路径字符串改为 `True`（语义："该后端模块暴露 layerwise 协议函数"）；`get_layerwise_protocol(name)` 在标记存在时经 `backend_map[name]["path"]` import 并返回模块，否则返回 None。理由：

- 消除与 `path` 重复的第二条路径字符串（防漂移）；
- mooncake / yuanrong 无标记，返回 None 时不产生任何 import（规避 `mooncake_backend` 顶层重依赖）；
- 被 import 的只有"声明支持 layerwise 的后端"，且该模块在 worker/scheduler 初始化路径上本就会被 import（第 3 节证据），无新增开销；
- 能力声明仍集中在 registry 一张表内，可评审、可用测试锁定一致性。

**D3. 通用层改名。** `use_gva_layerwise` → `use_layerwise_transfer`（= `use_layerwise and get_layerwise_protocol(...) is not None`），worker / scheduler / layout 统一；key 构造改走 `self.layerwise_protocol.make_*`。

**D4. 线程保持后端无关。** 删除 `kv_transfer.py` 两处 `assert isinstance(self.m_store, MemcacheBackend)` 及顶部 `MemcacheBackend` import，不加任何替代检查——线程启停由 worker 侧 `use_layerwise_transfer` 门禁保证，store 调用均在基类 ABC 上。gemini 的 assert→TypeError 意见随检查删除一并消解。

**D5. connector 中性化 + waiter 加固。** `ascend_store_connector.py` 与 `pool_worker.py` 中 `set_external_slot_release_waiter` 的 docstring 去除 GVA 表述；worker 侧设置 waiter 时若 `kv_recv_thread` 已启动，同步更新线程属性（采纳 gemini 的初始化顺序意见）。

### 4.3 范围边界

- 不为 mooncake 提前抽象：协议函数面只覆盖 memcache 当前真实需求，mooncake 接入时再扩展。
- `pool_worker.py` 中 `_allocated_gvas` / `_alloc_gvas_for_save` 等"GVA 地址"语义的命名本轮不动（属 store API 返回值的地址语义，超出本次检视范围），如 reviewer 要求可后续单独 PR 处理。

## 5. 检视意见闭环对照

| 意见 | 闭环方式 |
|------|----------|
| Pz1116 #1（协议移到 memcache backend） | D1 |
| Pz1116 #2（connector 注释去 GVA） | D5 |
| Pz1116 #3/#4（删线程 memcache 断言） | D4 |
| Pz1116 #5（scheduler 仍有 use_gva_layerwise） | D3 |
| Pz1116 #6（layerwise_protocol.GVAKeyFactory 后端专属） | D1 + D3 |
| gemini：assert → TypeError | D4（检查整体删除） |
| gemini：waiter 初始化顺序 | D5 |
| gemini：backend 名大小写归一化 | 已在此前 commit 采纳（resolver 内 `strip().lower()` 保留） |

## 6. 修改计划

每步一个 commit，保证各中间 commit 可编译、测试通过、行为不变。

**执行状态（2026-09-01）**：4 个 commit 已全部完成并推送（`2a239d18a..359876493`，fast-forward）；服务器 refactor_165 全量 UT **314 passed**（较 313 +1，为新增 waiter 交接测试）；服务器验证树已 reset 至 `359876493`，与推送树哈希逐字节一致（`133fcec2`）；本地 ruff 0.14.0 check+format 全绿。

### Commit 1：迁移协议实现（`f057d367c`）

- `memcache_backend.py`：新增四个协议函数（key 字符串逐字节不变）；
- `gva_protocol.py`：转为向后兼容别名（`GVAKeyFactory` 静态方法委托到 memcache 函数，`extract_layout_config` 再导出），消费方与测试零改动；
- 状态：工作区已应用（`memcache_backend.py` +72 行，`gva_protocol.py` -73/+16）。

### Commit 2：消费方切换与改名（`5293cad05`）

- `backend/__init__.py`：`layerwise_protocol` 字段值改为 `True`，resolver 改为经 `path` import 并返回模块（D2）；
- `pool_worker.py` / `pool_scheduler.py`：key 构造改走 `layerwise_protocol.make_*`，`use_gva_layerwise` → `use_layerwise_transfer`，删 `_layerwise_key_factory`；
- `layerwise_cache_layout.py`：调用形态不变（`protocol.extract_layout_config` 对模块天然成立）；
- 测试同步：`test_backend.py` resolver 测试改断言协议函数；`test_pool_worker.py` / `test_pool_scheduler.py` 跟随改名。

### Commit 3：删除别名模块（`5b43d7617`）

- 删 `backend/gva_protocol.py`；
- `test_gva_protocol.py` 删除，其三部分测试迁入 `test_backend.py` memcache 区块：
  - 排他性测试改为：模块暴露协议函数 ⇔ 类 override 五个 batch 接口 ⇔ registry 标记存在；
  - `extract_layout_config` opt-in 测试；
  - key 格式快照测试（逐字节不变）；
- `_mock_deps.py` 注释同步。

### Commit 4：线程后端无关化与清理（`359876493`）

- `kv_transfer.py`：删两处 assert、`MemcacheBackend` import 及相关注释（D4）；
- `ascend_store_connector.py`：docstring 中性化（D5）；
- `pool_worker.py`：`set_external_slot_release_waiter` docstring 中性化 + waiter 顺序加固（D5）；
- `test_kv_transfer.py`：线程 fixture 去掉 `spec=MemcacheBackend`，恢复普通 MagicMock；
- `test_ascend_store_connector.py`：注释措辞同步。

## 7. 验收标准

- `tests/ut/distributed/ascend_store/` 全量通过（在带 torch 的环境执行，与代码修改分开进行）——**已达成：refactor_165 内 314 passed**；
- 通用层源码 grep 不到 `GVA` / `MemcacheBackend`（`memcache_backend.py` 自身与测试的 memcache 区块除外）——**已达成**：`MemcacheBackend` 仅剩 registry 类名字符串与 memcache_backend.py 自身；`use_gva_layerwise` / `GVAKeyFactory` / `gva_protocol` 全仓零残留。`GVA` 地址语义命名（`_allocated_gvas`、`gva_block_offset`、docstring 等）按 §4.3 保留，如 reviewer 追问再单独 PR；
- key 格式快照断言逐字节不变——**已达成**：快照测试迁入 `test_backend.py` `TestLayerwiseKeyFormats` 并全绿；
- 每个 commit 独立可验证、行为不变——4 个 commit 逐级 git am 应用成功。

## 8. 风险与待办

- ~~本地 UT 环境：系统 Python 缺 torch，需用带完整依赖的环境执行第 7 节验证~~——已在 refactor_165 完成（git am 4 补丁 → pytest → reset 至推送头）；
- Commit 2 的 resolver 语义：仅"无标记"返回 None，backend 模块自身的 import 错误向上抛出，不吞为"无协议"（已按此实现）；
- D2 的字段形态（布尔标记 vs 指向模块本身的路径字符串）如 reviewer 有偏好，resolver 仅一行之差，不影响其余各步；
- 待办：`5c550766d`（mypy 修复：`backend_map` 显式注解 `dict[str, dict[str, Any]]`）已推送；观察 CI（mypy 3.10/3.11/3.12、test_layerwise_cache_layout.py 收集、a3-16 卡 flake）。PR 描述与检视回复已更新（2026-09-01）。
