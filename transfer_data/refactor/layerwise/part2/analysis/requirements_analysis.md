# Layerwise 逻辑收敛 part2 — 需求分析

> 基线：upstream/main（本地 fetch 2026-09-03，`#15367` 合入 commit `eb617eaf1`，其后 ascend_store 区域又进 2 个 PR，见 §3）
> 方法：全部结论锚定 upstream/main 实测（行号可复验：`git grep -in "gva" upstream/main -- <path>`）；`archive/` 中 part1 材料仅作交叉验证，不作为事实来源
> 前序：PR #15367（part1，2026-09-02 合入）已完成 gate 单点化、key 构造归位 memcache_backend、通用层中性改名

---

## 1. 本质需求（重新推导，不继承结论）

**layerwise 传输是通用能力，GVA 是 memcache 的内部寻址机制——两者正交。**

part1 之后的 main 恰好是这个论断的反面证据：通用层文件里 235 处 `gva` 命中，其中相当部分不是命名残留，而是**可执行逻辑**——GVA 地址算术、租约管理、传输编排实打实长在 pool_worker / kv_transfer 里。具体到三段硬逻辑：

1. **GVA 地址构建算术**（kv_transfer.py LayerBatchBuilder：`base_gvas[:, None] + rank_layer_offset + layer_inner_offsets[None, :]` 拼地址、zero/negative gva 检测跳过）
2. **GVA 生命周期编排**（pool_worker.py：alloc 租约、write-finish、load gva 刷新）
3. **GVA 传输线程**（kv_transfer.py 后两个 Layer 线程类，整类只服务 memcache 寻址协议）

后果与 part1 之前同构：mooncake 实现自己的 layerwise 时，要么绕过这批代码另起炉灶（复用率为零），要么被迫理解并维护 GVA 语义（泄漏固化）。**part2 的需求就是把这批逻辑物理归位到 memcache backend，使"mooncake 接入 layerwise = 实现 store 方法 + 协议函数 + 注册表标记"成为代码事实而非口头承诺。**

## 2. 现状实测：泄漏点全景

### 2.1 分类框架（按收敛手段分四级）

| 级 | 定义 | 判定 | 手段 |
|---|---|---|---|
| L1 物理宿主错误 | 逻辑正确但住错文件：GVA 专属代码在通用传输层/编排层 | 类/方法整体仅被 GVA 路径使用 | 搬迁（行为保持） |
| L2 契约泄漏 | GVA 方法桩挂在通用 Backend ABC 上 | 仅 MemcacheBackend override | 契约下沉 |
| L3 运行期分支 | 通用流程里的 `use_layerwise_transfer` 分支承载 GVA 编排 | 分支体内是 GVA 语义 | 协议对象化 |
| L4 命名残留 | 地址语义的字段/变量名含 gva | 纯数据字段，无逻辑 | **本轮不动**（见 §5） |

### 2.2 逐文件实测（upstream/main 行号）

**kv_transfer.py（1621 行，8 类）— L1 大头（四块）：**

| 位置 | 内容 | 行数 |
|---|---|---|
| `:39-304` | `LayerBatchBuilder`：GVA 地址数组构建（`_build_transfer_arrays`/`build_shared`/`build_addrs`/`build`，内部全为 base_gva 偏移算术） | 265 |
| `:393-475` | `KVTransferThread` **通用基类**上的 `_split_transfer_packets` / `_batch_copy_with_limits`：GVA 包切分 + 限流批拷贝（`split_gvas = gvas[entry_indices] + transfer_offsets`，含 `.store.batch_copy` 直捅）——仅被两个 GVA Layer 线程调用（:1366/:1566），key-mode 线程零消费 | ~83 |
| `:1271-1407` | `KVCacheStoreLayerSendingThread`：GVA layerwise 发送线程 | ~136 |
| `:1408-1621` | `KVCacheStoreLayerRecvingThread`：GVA layerwise 接收线程 | ~213 |

gva 命中分布（实测 86 处，区分大小写）：Builder 60 / 基类两方法 16 / Layer 线程 10。搬迁面合计 ~700 行。

通用层应保留的对照（证明"层"可分）：`KVTransferThread`(:305 通用基类，**瘦身后**只留线程骨架)、`KVCacheStoreSendingThread`(:598)/`RecvingThread`(:868)（key-mode）、`KeyLayerSendingThread`(:1015)/`KeyLayerRecvingThread`(:1169)（key-mode layerwise——注意：这是**通用 layerwise**，与 GVA 版并存，恰证明两维度正交）。

**pool_worker.py（2475 行）— L1 + L3：**

| 位置 | 内容 |
|---|---|
| `:1158` | `_refresh_allocated_gvas(keys)`：GVA 地址簿刷新 |
| `:1175` | `_alloc_gvas_for_save(requests)`：save 前 GVA 分配 + 租约 |
| `:1339` | `_prepare_load_gvas(requests)`：load 前 GVA 准备 + 租约 |
| `:162` | 构造期派生 `use_layerwise_transfer`（含 1 处） |
| `:381/:423/:472/:492/:530/:835/:1183/:1349` | 8 处运行期分支（thread 启动、save/load 编排、waiter 路径） |

**pool_scheduler.py（993 行）— L3 少量 + L4：** `:169` 派生、`:197/:476` 两处分支（hit-check 编排）；其余 7 处命中为 tracker 字段传递（L4）。

**backend/base.py（43 行）— L2：** 5 个 GVA 方法桩以 `NotImplementedError` 形态挂在通用契约：`batch_get_key_info` / `batch_alloc` / `batch_add_lease` / `batch_remove_lease` / `batch_write_finish`。全仓仅 MemcacheBackend override（part1 排他性测试已锁定）。

**metadata.py（1093 行）— L4（全部 46 处）：** `block_gvas` / `gva_block_offset` / `load_block_gvas_np` / `partial_save_gva_per_group` 等全是 ReqMeta/Tracker 数据字段。

**backend/ 目录 — 零命中**（part1 成果）：registry 为布尔标记（`"layerwise_protocol": True`）+ `get_layerwise_protocol()` 单一解析器；协议函数（`make_full_key` 等 4 个）宿主 memcache_backend.py（290 行）。

### 2.3 与 part1 档案 §7.5 的交叉验证

| 档案遗留项 | 实测 | 结论 |
|---|---|---|
| worker 内 8 处分支 | 实测 8 处（§2.2 所列行号） | 一致，仍是 part2b 范围 |
| GVA 线程类在 kv_transfer.py | 实测四块 ~700 行（含**通用基类上的两方法**，档案未单列——#15307 的 B-2 中间基类设计正为此） | 一致（面更大），part2a 范围 |
| 协议其余部分（租约、write-finish）散在 worker/线程 | 实测 3 个编排方法 | 一致，part2b 范围 |
| base.py 5 桩 | 实测仍在（NotImplementedError 形态，part1 返工从 ABC 撤回后的既有形状） | 一致，part2b 末段 |

**档案未记的新事实**：档案写作时 kv_transfer 的三块带着 `assert isinstance(m_store, MemcacheBackend)`（后被 #15367 返工删除）且宿主设计是 gva_protocol.py；现在断言已删、宿主是 memcache_backend.py、线程 fixture 已是普通 MagicMock。**#15307 的 gva_threads.py 891 行不能直接复用，需按新形态重对**（偏差见方案 §资产评估）。

## 3. main 演进动态（#15367 之后）

ascend_store 区域后续合入（截至本地 fetch）：

| PR | 内容 | 与 part2 的关系 |
|---|---|---|
| #12711（09-02） | hybrid GDN+full_attention 模型 layerwise 支持：GDN 层显式 wait/save，触碰 multi_connector/ascend_store_connector/pool_scheduler/pool_worker | **不触碰 kv_transfer.py**——搬迁目标区域干净；但 pool_worker +20 行意味着 8 处分支行号已漂移，方案以实测为准 |
| #15469（09-03） | KV pool QoS 控制：mooncake_backend +64、pool_scheduler/worker 少量 | backend 继续生长的信号；QoS 常量进了 backend/base.py 通用契约，说明通用契约还在被塞东西——base.py 瘦身诉求更实 |

另：`KVCacheStoreKeyLayer*Thread`（key-mode layerwise，#12711 同期活跃）与 GVA 版在同文件并存，是"layerwise 通用 vs GVA 后端专属"正交性最直接的代码证据。

## 4. 需求陈述

- **R1（物理归位）**：kv_transfer.py 中 GVA 专属三块（LayerBatchBuilder + 两个 Layer 线程）迁入 backend 侧；kv_transfer.py 保留的每个类必须对全部后端有意义。
- **R2（编排归位）**：pool_worker 的 3 个 GVA 编排方法与 8 处分支、pool_scheduler 的 2 处分支，收编为协议对象（构造期经 `get_layerwise_protocol()` 解析，运行期按对象存在性分支）。
- **R3（契约瘦身）**：base.py 的 5 个 GVA 桩退出通用 Backend ABC（随 R2 的协议对象承接，不在行为保持的批次做）。
- **R4（可验收度量）**：通用层（pool_worker/pool_scheduler/kv_transfer/layerwise_cache_layout/ascend_store_connector/backend/base.py/backend/__init__.py）中，GVA **逻辑**归零——即：类/方法/分支级 gva 命中为 0；L4 命名字段除外（§5）。

## 5. 非需求（明确排除，防蔓延）

1. **L4 命名不动**：`block_gvas`、`gva_block_offset`、`_allocated_gvas` 等是 store API 返回值的地址语义，part1 返工时已与 reviewer 确认保留（"如 reviewer 追问再单独 PR"）。改它们触碰 ReqMeta 序列化兼容面，收益/风险比不成立。
2. **不为 mooncake 提前抽象**：协议对象方法面只覆盖 memcache 当前真实需求；mooncake 接入时再扩展（part1 返工确立的边界）。
3. **key 字符串冻结**：`make_full_key` 等 4 个协议函数已合入 main，本轮零触碰；快照测试作为回归锚点复用。
4. **key-mode 线程不动**：`KVCacheStoreKeyLayer*Thread` 是通用 layerwise 路径，属框架本体。

## 6. 验收标准（需求层面）

1. R1-R3 逐项达成后，§2.2 的 L1/L2/L3 清单全部归零（L4 保留项白名单化记录在案）；
2. 每个中间 commit 行为不变（判据：ascend_store 全量 UT + 与 main 的行为等价论证）；
3. e2e 三场景（MultiConnector PD / memcache layerwise / mooncake 非 layerwise）与 main 基线一致——判据沿用 part1（`hit_tokens>0`、`valid_gvas>0`、无 AttributeError）；
4. PR 尺寸：生产+UT 合计不超 1000 行红线，超了就再分批。
