# AscendStore Layerwise 后端收敛 — 需求分析

> 分析对象：vllm-project/vllm-ascend main 分支（2026-08-29 快照，本地 clone 验证）
> 需求：kv_pool 四项重构需求之第 4 项 — "AscendStore Layerwise 代码重构，将后端强相关的逻辑收敛到 backend 文件中"
> 配套方案：[p0_modification_plan.md](p0_modification_plan.md)（P0 全部问题的具体修改方案）、[p1_p2_modification_plan.md](p1_p2_modification_plan.md)（P1/P2 问题的具体修改方案）

## 1. 需求背景

前序需求状态（截至今日）：

| # | 需求 | 状态 |
|---|------|------|
| 1 | UT 整改，删除冗余测试 | 已完成（PR #14465） |
| 2 | kv_pool 目录重构 | 已完成（PR #13354） |
| 3 | 消除死代码/冗余代码 | 主体完成（PR #14465；延续 PR #14697 open） |
| 4 | Layerwise 后端逻辑收敛到 backend | **未完成（本文档）** |

术语约定：
- **后端强相关逻辑**：仅对特定 backend 生效/有意义的代码路径。当前 layerwise 场景下指 memcache 独有的 GVA（Global Virtual Address）机制相关代码。
- **GVA layerwise**：`use_layerwise=True` 且 `backend=memcache` 时的传输路径（`use_gva_layerwise`）。
- 现有后端：mooncake（默认）、memcache、yuanrong。

## 2. 现状

### 2.1 代码规模（ascend_store 目录）

| 文件 | 行数 | 说明 |
|---|---|---|
| pool_worker.py | 2266 | 最大散布点，GVA 协议主体在此 |
| kv_transfer.py | ~1620 | 6 个传输线程类（3 种模式 × 收/发） |
| metadata.py | ~1120 | ReqMeta/LoadSpec 等 |
| pool_scheduler.py | 969 | 调度侧 lookup 与分支 |
| backend/mooncake_backend.py | 362 | |
| backend/memcache_backend.py | 206 | |
| backend/yuanrong_backend.py | 146 | |
| backend/base.py | 39 | 当前抽象边界，见 2.2 |

### 2.2 当前 backend 抽象边界（backend/base.py）

接口清单：
- 通用数据面：`set_device` / `register_buffer` / `exists` / `batch_is_exist` / `put` / `get`
- **memcache 专属接口**（默认 `raise NotImplementedError`）：`batch_get_key_info` / `batch_alloc` / `batch_add_lease` / `batch_remove_lease` / `batch_write_finish`

关键事实 — **GVA 线程绕过 `put/get` 直捅 backend 内部**：
- `put/get` 仅被 key 模式线程以 key 语义使用（mooncake：`batch_put_from_multi_buffers`；memcache：`batch_put_from_layers`/`batch_get_into_layers`）。
- GVA 版线程（KVCacheStoreLayerSendingThread/RecvingThread）经 `_batch_copy_with_limits`（kv_transfer.py:426）直达 `self.m_store.store.batch_copy(...)`（:468）——**backend 内部属性被通用层直接访问**，封装破坏。（修正：本节初版误记为"put/get 同接口不同语义"，细读代码后更正，详见 P1/P2 方案 §0）

### 2.3 GVA layerwise 数据流（当前实现）

```
scheduler get_num_new_matched_tokens
  └─ use_gva_layerwise 分支 (pool_scheduler.py:467)
      └─ _get_layerwise_gva_hit_tokens (pool_scheduler.py:319)
          └─ _make_layerwise_gva_keys_for_hit_check (pool_scheduler.py:304)  ← key 构造 #1
          └─ store_scheduler.batch_get_key_info + ki.size()>0 判定           ← hit 语义在通用层
worker start_load_kv → process_layer_data (pool_worker.py:1597)
  ├─ _prepare_load_gvas (pool_worker.py:1315)   ← batch_get_key_info + batch_add_lease 读租约协议
  ├─ _alloc_gvas_for_save (pool_worker.py:1151) ← batch_alloc + partial key + _allocated_gvas 缓存
  └─ _process_save/_load_for_layer_batch → LayerTransferTask
worker save_kv_layer → KVCacheStoreLayerSendingThread (kv_transfer.py:1271)
  └─ LayerBatchBuilder 构造 GVA 地址 → m_store.put(gvas, addrs, sizes)
worker wait_for_layer_load → KVCacheStoreLayerRecvingThread (kv_transfer.py:1405)
  └─ m_store.get(gvas, addrs, sizes)
```

## 3. 问题清单

### P0-1 `use_gva_layerwise` 派生逻辑散布 4 处（2 处定义 + 1 处判断 + 1 处消费引用）

同一布尔语义在 4 个文件独立计算：

| 位置 | 内容 |
|---|---|
| pool_worker.py:155-157 | `self.backend = extra_config.get(...)` + `self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"` |
| pool_scheduler.py:161-163 | 同上（独立解析 extra_config） |
| layerwise_cache_layout.py:98-99 | `get_gva_layerwise_config()` 内 `backend == "memcache" and use_layerwise` |
| ascend_store_connector.py:199 | 引用 `self.use_gva_layerwise`（见 P0-2） |

**已发生的生产事故**：PR #14465 精简 `AscendStoreConnector.__init__` 时删除了该属性定义，导致 MultiConnector（PD 分离 + MooncakeLayerwiseConnector + AscendStoreConnector）初始化即 `AttributeError: 'AscendStoreConnector' object has no attribute 'use_gva_layerwise'`。修复 commit `412b157`（+2 行）目前挂在 open PR #12711，**main 分支仍带此回归**。此类事故是逻辑不收敛的直接代价：同一派生语义多点复制，删一处即崩。

### P0-2 memcache GVA 协议实现全部位于通用层

以下逻辑仅 memcache 后端使用，但物理上位于 pool_worker / pool_scheduler：

| 逻辑 | 位置 | 规模 |
|---|---|---|
| GVA key 构造（save/load） | pool_worker.py:1113-1133 `_make_layerwise_gva_key` / `_make_layerwise_partial_key` | key 格式 `model[@group_id]@hash@rank`、`model@partial@req@...` |
| GVA key 构造（scheduler hit-check） | pool_scheduler.py:304-317 `_make_layerwise_gva_keys_for_hit_check` | 同格式第二份实现（all-rank 展开） |
| save 侧 GVA 分配 | pool_worker.py:1151-1313 `_alloc_gvas_for_save` | ~160 行：`batch_alloc`、已分配跳过、partial 块、`_allocated_gvas` 本地表 |
| load 侧 GVA + 租约 | pool_worker.py:1315-1515 `_prepare_load_gvas` | ~200 行：`batch_get_key_info`、`batch_add_lease`、invalid block 上报 |
| GVA 本地缓存失效 | pool_worker.py:1134-1149 `_refresh_allocated_gvas` | `batch_is_exist` 驱动的淘汰 |
| hit-check 判定语义 | pool_scheduler.py:319-387 `_get_layerwise_gva_hit_tokens` | `ki.size() > 0` 逐 group 聚合取 min |

后果：新后端要支持 layerwise 时无协议模板可复用；key 格式变更需同步修改 worker/scheduler 两份构造代码，存在格式漂移风险（key 格式兼容性见 §5）。

### P0-3 base.py 混入 memcache 专属接口

`batch_get_key_info` / `batch_alloc` / `batch_add_lease` / `batch_remove_lease` / `batch_write_finish` 为 memcache GVA 机制专用，mooncake / yuanrong 实现为 `raise NotImplementedError`（base.py:30-45）。通用基类承载单一后端专属协议，后端切换时接口边界不可判定（无法在类型层面区分"支持 GVA 的后端"）。

### P1-4 传输线程类组合与选择逻辑

kv_transfer.py 并存 6 个线程类：

| 模式 | Sending | Recving |
|---|---|---|
| 非 layerwise | KVCacheStoreSendingThread (598) | KVCacheStoreRecvingThread (868) |
| layerwise 非 GVA | KVCacheStoreKeyLayerSendingThread (1015) | KVCacheStoreKeyLayerRecvingThread (1169) |
| layerwise GVA | KVCacheStoreLayerSendingThread (1271) | KVCacheStoreLayerRecvingThread (1405) |

选择逻辑位于 pool_worker.py:463-584 `_start_kv_transfer_threads`（`use_gva_layerwise` / `use_layerwise` / `can_save` 三重 if/else 嵌套），后端能力决定线程组合的映射关系未收敛。GVA 专用的 `LayerBatchBuilder.build_shared/build_addrs`（GVA 地址构造）也在该通用文件。

### P1-5 backend 特定时序硬编码

pool_worker.py:816-817：`if self.use_gva_layerwise: self.m_store.ensure_initialized()` — memcache 惰性初始化时序由通用层判断驱动，而非 backend 自身生命周期管理。

### P2-6 零散后端痕迹

- pool_worker.py:1318 注释、多处日志/异常文案硬编码 "MemCache" 字样（存在于通用层）。
- pool_worker.py:2386 `check_all_layers_exists`：per-layer exists 结果聚合逻辑与 layerwise key 布局（每 chunk `num_layers` 个 key）耦合。**修正**（细读代码后更正，见 P1/P2 方案 §0）：调用点 gate 为 `use_layerwise`（非 `use_gva_layerwise`），服务 key-mode 通用路径（三后端可用），归属应为 metadata 层而非 backend。

## 4. 重构范围

### 收敛项（本需求范围）

1. `use_gva_layerwise`（backend 能力判定）单点定义，其余位置引用派生。
2. GVA key 构造（两份）、save 侧分配、load 侧租约、GVA 本地缓存、hit-check 语义 → 收敛至 backend 目录（memcache_backend 或其 layerwise 扩展）。
3. base.py 接口分层：通用数据面原语与 layerwise/GVA 扩展接口分离，能力可探测。
4. 线程选择逻辑由 backend 能力驱动，`_start_kv_transfer_threads` 分支简化。

### 明确不动项（边界）

- layerwise 调度语义（eagle trim、`kvpool_store_skip_tokens`、force_layerwise_load 等）— 与后端无关。
- layerwise reuse layout / offload 语义（layerwise_cache_layout.py 的布局计算部分）。
- 非 layerwise 三后端数据面（`exists` / `put` / `get`）。
- coordinator.py / attention_fence.py / metadata.py。
- #14697 已覆盖的 `batch_is_exist`→`exists` API 统一。

## 5. 验收标准

功能（全部通过）：
1. UT 全量回归（tests/ut/distributed/ascend_store，165 服务器）。
2. mooncake（非 layerwise）冒烟：现有 e2e 脚本通过，行为无回归。
3. memcache layerwise 冒烟：TP=4 + layerwise + 长前缀 load 复测通过（对齐既有验证基线）。
4. PD 分离 MultiConnector（MooncakeLayerwiseConnector + AscendStoreConnector）冒烟：参照 `412b157` 验证方法（P 节点 4xTP + D 节点 consumer，proxy，GSM8K prefix-cache，请求成功率 100%，外部命中率不低于基线）。
5. main 分支 `AscendStoreConnector` 缺失 `use_gva_layerwise` 的回归（#14465 引入）在收敛后不存在。

结构：
6. `use_gva_layerwise` 语义定义仅存在 1 处；grep `backend_name == "memcache"` 在 backend 目录外无业务判断残留。
7. §3 P0-2 表列逻辑迁入 backend 目录；pool_worker/pool_scheduler 不再持有 GVA 协议实现。
8. base.py 中 memcache 专属接口与通用接口分离（分层或能力探测，方式不限）。

兼容性：
9. **存储 key 格式不变**（`model[@group_id]@hash@rank`、partial key 格式）— 已部署集群中的数据在升级后仍可命中。重构 PR 中 key 构造函数如需移动，需保证输出逐字节一致（以现有 UT 输出快照为基准）。
10. 配置接口不变：`use_layerwise` / `backend` / `layerwise_max_transfer_blocks` 等 extra_config 项及默认值。

工程约束：
11. PR ≤ 1000 行（生产代码 + UT），超出需按 §4 收敛项拆分序列提交。
12. commit 遵循 Conventional Commits + Signed-off-by；ruff 0.14.0 / mypy（3.10-3.12）通过。

## 6. 风险与依赖

| 风险/依赖 | 说明 |
|---|---|
| open PR 交织 | #12711（含 `412b157` 修复 + Qwen3.5 layerwise）、#14697（backend 路径简化）、#12854（layerwise transfer rework）均触碰同区域文件。收敛 PR 的 base/rebase 顺序需先确认三者走向，避免 412b157 场景重演（先落 #12711 修复 main 回归，或收敛 PR 内含等效修复）。 |
| key 格式兼容 | 收敛过程移动 key 构造代码时格式漂移会导致线上数据 miss，验收 9 需专项覆盖。 |
| 误删回归先例 | #14465（死代码清理）删 `use_gva_layerwise` 崩 MultiConnector — 本次收敛动同一区域，验收 4 必须实测 MultiConnector 而非仅 UT。 |
| UT 环境 | memcache layerwise UT 依赖 memcache backend（mooncake 无法覆盖该路径）；服务器执行走既有 tmp → scp → ssh 流程。 |

## 7. 附录：代码位置索引

```
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/
├── backend/
│   ├── base.py                      # 39 行，抽象接口（含 memcache 专属 5 个）
│   ├── memcache_backend.py          # 206 行，GVA 数据面原语
│   ├── mooncake_backend.py          # 362 行
│   └── yuanrong_backend.py          # 146 行
├── pool_worker.py
│   ├── :155-157                     # use_gva_layerwise 定义（#1 份）
│   ├── :348-442                     # _init_layerwise_config（GVA reuse layout 构建 :376-418）
│   ├── :463-584                     # _start_kv_transfer_threads 线程选择
│   ├── :816-817                     # ensure_initialized 后端时序特判
│   ├── :1098-1147                   # partial index / GVA key 构造 / _refresh_allocated_gvas
│   ├── :1151-1313                   # _alloc_gvas_for_save
│   ├── :1315-1515                   # _prepare_load_gvas
│   ├── :1597-1616                   # process_layer_data（GVA 协议挂载点）
│   └── :2386-2396                   # check_all_layers_exists
├── pool_scheduler.py
│   ├── :161-163                     # use_gva_layerwise 定义（#2 份）
│   ├── :191-206                     # GVA layerwise layout 初始化分支
│   ├── :304-317                     # _make_layerwise_gva_keys_for_hit_check（key 构造 #2 份）
│   ├── :319-387                     # _get_layerwise_gva_hit_tokens
│   └── :467-470                     # get_num_new_matched_tokens 分支入口
├── ascend_store_connector.py
│   └── :199                         # use_gva_layerwise 引用（#14465 回归点，定义缺失于 main）
├── layerwise_cache_layout.py
│   └── :98-99                       # get_gva_layerwise_config 中 backend 判断（#3 份）
└── kv_transfer.py
    ├── :39                          # LayerBatchBuilder（GVA 地址构造）
    ├── :598/868                      # 非 layerwise 收/发线程
    ├── :1015/1169                    # layerwise 非 GVA 收/发线程
    └── :1271/1405                    # layerwise GVA 收/发线程
```

关联 PR：
- #13354 目录重构（已合并）/ #14465 UT 精简 + 死代码（已合并，引入 use_gva_layerwise 回归）
- #14697 backend 路径简化（open）/ #12711 Qwen3.5 layerwise + 回归修复 412b157（open）/ #12854 layerwise transfer rework（open）
