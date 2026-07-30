# KV Pool Hash Key 能力现状与重构方向

一句话结论：**当前 hash key 支持同模型同布局 prefix KV 复用，支持 PCP/DCP rank、TP/head rank、PP rank、KV group、cache role、cache family 的基础隔离；不支持跨 CP layout 复用，不支持统一 key schema，不支持强 model identity，不支持 schema version/layout signature，不支持默认 decode KV 自动沉淀复用；TP/KV head mismatch 主路径端到端不支持，只有内部函数存在。**

本文按 `D:\lzy\project\kv_pool\code\vllm-ascend` 当前源码验证，工程判定标准是：**主路径端到端可用才叫支持；只存在内部函数、单测片段、兼容分支或配置开关默认关闭，不按支持宣传。**

## 1. 当前 hash key 到底能表达什么

当前通用 KV pool key 由 `PoolKey.to_string()` 生成，核心维度包括：

```text
{model_name}
@pcp{pcp_rank}
@dcp{dcp_rank}
@head_or_tp_rank:{head_or_tp_rank}
@pp_rank:{pp_rank}
@group:{kv_cache_group_id}
@cache_role:{cache_role}
@cache_family:{cache_family}
@{chunk_hash}
```

这些字段带来的能力是：

| 维度 | 当前作用 | 工程判定 |
| --- | --- | --- |
| `model_name` | 区分模型 basename | 支持弱隔离，不支持强模型身份 |
| `pcp_rank` | 区分 prefill CP rank | 支持 rank 隔离，不支持跨 CP layout 复用 |
| `dcp_rank` | 区分 decode CP rank | 支持 rank 隔离，不支持跨 CP layout 复用 |
| `head_or_tp_rank` | 区分 KV head shard 或 TP rank | 支持同布局 head/TP rank 隔离；TP mismatch 主路径端到端不支持 |
| `pp_rank` | 区分 pipeline parallel rank | 通用 `PoolKey` 支持；`LayerPoolKey` 不支持 PP 字段 |
| `group` | 区分 KV cache group | 通用 `PoolKey` 和 `LayerPoolKey` 支持；GVA layerwise 只在多 group 格式中支持 group |
| `cache_role` | 区分 cache 类型，默认 `kv` | 通用 `PoolKey` 和 `LayerPoolKey` 支持；GVA layerwise 不支持 |
| `cache_family` | 区分 default/c2/mixed 等 cache family | 通用 `PoolKey` 和 `LayerPoolKey` 支持；GVA layerwise 不支持 |
| `chunk_hash` | 表示 token prefix 内容 | 支持 prefix hash 命中 |

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:73` `KeyMetadata`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:94` `PoolKey`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:114` `PoolKey.to_string()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:337` `ChunkedTokenDatabase._make_key_by_hash()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:638` `get_block_hashes()`

## 2. 当前支持的能力

### 2.1 支持同布局 prefix KV 复用

结论：**支持。**

只要 producer 和 consumer 的模型 basename、PCP/DCP rank、TP/head rank、PP rank、KV group、cache family、cache role 等布局一致，并且 token prefix hash 一致，就可以通过 hash key 命中并复用 KV。

普通非 layerwise 路径会通过 worker 侧 `lookup()` / `lookup_scheduler()` 检查连续命中；scheduler 侧 lookup 会展开所需 PP/TP rank key，要求对应 rank 的 key 都存在。layerwise 路径会拆成 layer key 并检查所有 layer key 是否存在。

能力判断：**支持同模型同布局 prefix KV 复用。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1883` `KVPoolWorker.lookup()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:2082` `KVPoolWorker.lookup_scheduler()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1974` `_expand_lookup_keys_by_rank()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:2115` scheduler lookup 展开 rank key
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:2185` `check_all_layers_exists()`

### 2.2 支持 PCP/DCP rank 隔离

结论：**支持 rank 隔离。**

当前 key 明确包含 `pcp_rank` 和 `dcp_rank`，scheduler 和 worker 也都会计算：

```python
cp_scale = pcp_size * dcp_size
grouped_block_size = original_block_size * cp_scale
hash_block_size = requested_hash_block_size_or_prefix_match_unit * cp_scale
```

这能保证不同 PCP/DCP rank 不会落到同一个 key namespace。

当前 `get_block_hashes()` 在 `group_block_size > hash_block_size` 时使用 `_LazyGroupedBlockHashList`，取组内最后一个 fine-grained chained hash 作为 larger block 的 hash；代码没有单独的 `_rehash_block_hash_group()` 重哈希函数。

能力判断：**支持 CP rank 隔离；不支持跨 CP layout 复用。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:114` key 包含 `pcp_rank` / `dcp_rank`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:111` scheduler 计算 `cp_scale`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:146` worker 计算 `cp_scale`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:638` `get_block_hashes()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:649` `_LazyGroupedBlockHashList`

### 2.3 通用 PoolKey 支持 PP rank、KV group、cache role、cache family 隔离

结论：**通用 `PoolKey` 支持；layerwise/GVA 不完整支持。**

通用 `PoolKey` 包含 `pp_rank`、`kv_cache_group_id`、`cache_role`、`cache_family`，因此普通路径能隔离这些 namespace。

能力判断：

- **普通通用 key 路径支持 PP/group/cache role/cache family 隔离。**
- **`LayerPoolKey` 不支持 PP rank 字段。**
- **GVA layerwise key 不支持 PCP/DCP/PP/cache role/cache family 字段。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:114` `PoolKey.to_string()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:161` `LayerPoolKey.to_string()` 缺少 `pp_rank`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1062` GVA layerwise key

### 2.4 支持多轮对话 prefix-only 复用

结论：**支持 prefix-only。**

当前 key 不包含 conversation id，也不包含 request id。只要下一轮请求的 prompt 包含完整历史，并且 token 序列前缀相同，就可以通过 prefix hash 命中已经保存过的 KV。

能力判断：**支持 token-prefix 级别的多轮复用；不支持默认 decode KV 自动沉淀复用。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:114` key 不包含 conversation id / request id
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:98` `save_decode_cache` 默认 `False`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:823` decode 保存分支

## 3. 当前不支持的能力

### 3.1 不支持跨 CP layout 复用

结论：**不支持。**

当前 key 里有 `pcp_rank` / `dcp_rank`，hash/block 粒度也乘了 `pcp_size * dcp_size`。这只能做隔离，不能做跨 layout 复用。

缺少的信息包括：

- producer 的 PCP/DCP size；
- consumer 的 PCP/DCP size；
- producer shard 到 consumer shard 的 token range mapping；
- CP 分片是连续切分还是交错切分；
- 跨 CP layout 时需要聚合、拆分还是重排 KV；
- grouped hash 是全局 prefix hash 还是 shard-local hash 的明确语义。

能力判断：**不支持跨 CP layout 复用。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:73` `KeyMetadata` 没有 CP size/layout 字段
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:114` key 只写 rank，不写 layout
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1974` lookup 只展开 rank key，不做 CP mapping

### 3.2 不支持统一 key schema

结论：**不支持。**

当前至少有三套 key：

1. 通用 `PoolKey`。
2. `LayerPoolKey`。
3. GVA layerwise key。

它们包含的维度不同：

| key 类型 | PCP/DCP | PP | group | cache_role | cache_family | layer | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 通用 `PoolKey` | 有 | 有 | 有 | 有 | 有 | 无 | 支持普通路径隔离 |
| `LayerPoolKey` | 有 | 无 | 有 | 有 | 有 | 有 | 不支持统一 PP 语义 |
| GVA layerwise key | 无 | 无 | 单 group 无；多 group 有 | 无 | 无 | 隐含 | 不支持统一 schema |

能力判断：**不支持统一 key schema。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:114` `PoolKey.to_string()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:161` `LayerPoolKey.to_string()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1062` `_make_layerwise_gva_key()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:307` `_make_layerwise_gva_keys_for_hit_check()`

### 3.3 不支持 TP/KV head mismatch 端到端主路径

结论：**不支持端到端主路径。**

代码里存在 TP mismatch 推断和 worker 内部 load/store 函数，但普通发送/接收线程没有传入 `worker=self`，所以线程主路径进不到 TP mismatch 分支。

发送线程只有在 `self.worker is not None and self.worker.tp_mismatch` 时才走 `_store_kv_tp_mismatch()`。接收线程也是只有 `worker` 存在时才走 `_load_kv_tp_mismatch()`。但是 `KVPoolWorker._start_kv_transfer_threads()` 创建普通 `KVCacheStoreSendingThread` / `KVCacheStoreRecvingThread` 时没有传 `worker=self`。

能力判断：**端到端不支持。内部函数存在不改变工程判定。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:36` `infer_tp_mismatch_info()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:52` TP mismatch 启用条件
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:202` worker 初始化 TP mismatch
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:495` 创建发送线程未传 `worker=self`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:512` 创建接收线程未传 `worker=self`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:668` 发送线程 TP mismatch 分支依赖 `worker`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py:921` 接收线程 TP mismatch 分支依赖 `worker`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1721` `_load_kv_tp_mismatch()`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1757` `_store_kv_tp_mismatch()`

### 3.4 不支持 TP mismatch + sparse / layerwise / hybrid

结论：**不支持。**

worker 初始化阶段已经明确报错：

- TP mismatch + sparse：`ValueError`
- TP mismatch + layerwise：`ValueError`
- TP mismatch + hybrid：`NotImplementedError`

并且 TP mismatch 内部地址构造只处理 group 0，store 也只取 `block_ids_by_group[0]`。

能力判断：**不支持 sparse、layerwise、hybrid、多 group TP mismatch。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:212` sparse 限制
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:219` layerwise 限制
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:225` hybrid 限制
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1675` TP mismatch strided I/O 写死 group 0
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1766` TP mismatch store 只取 group 0 block ids

### 3.5 不支持默认 decode KV 自动沉淀复用

结论：**默认不支持。**

当前默认 `save_decode_cache=False`。running cached request 如果已经进入 decoding，且没有开启 `save_decode_cache`，scheduler 直接返回 `None`，不会继续保存 decode 阶段新增 token 的 KV。

能力判断：**默认不支持 decode KV 自动沉淀复用。**

如果要把多轮对话完整 KV 复用作为能力，需要显式开启并验证 `save_decode_cache=True`，同时处理：

- `discard_partial_chunks=True` 默认行为；
- 最后一个 partial block 是否保存；
- request finished 时异步保存是否完成；
- 下一轮请求是否要等待上一轮保存完成；
- decode token hash 和 prompt token hash 的边界一致性。

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:98` `save_decode_cache` 默认值
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:132` `discard_partial_chunks` 默认值
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:823` running cached request 保存判断
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:827` decoding 且未开启 `save_decode_cache` 时返回 `None`

### 3.6 不支持强模型身份隔离

结论：**不支持。**

当前 `model_name` 来自模型路径 basename，不包含 revision、dtype、quantization、rope scaling、LoRA 等身份信息。

这会带来冲突风险：

- 不同路径但 basename 相同；
- basename 相同但权重不同；
- revision 不同；
- dtype / quantization 不同；
- rope scaling 不同；
- LoRA 不同。

能力判断：**不支持强 model identity。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:130` worker `model_name = model_config.model.split("/")[-1]`
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:279` metadata 使用 basename
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:194` scheduler 使用 basename

### 3.7 不支持 schema version

结论：**不支持。**

`PoolKey.to_string()`、`LayerPoolKey.to_string()`、GVA layerwise key 都没有 `v1/v2/schema` 字段。

能力判断：**不支持 schema version。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:114` `PoolKey.to_string()` 无 schema version
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:161` `LayerPoolKey.to_string()` 无 schema version
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1062` GVA key 无 schema version

### 3.8 不支持 layout signature

结论：**不支持。**

当前 key 表达的是 rank 和 cache namespace，不表达完整 layout。

缺少的信息包括：

- producer TP/CP/PP size；
- consumer TP/CP/PP size；
- num_kv_heads；
- head_dim；
- heads_per_rank；
- KV layout order；
- cache tensor layout version；
- compatible layout signature；
- producer 到 consumer 的 head/CP shard mapping。

能力判断：**不支持 layout signature。**

相关代码：

- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:73` `KeyMetadata` 没有 layout 字段
- `code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py:114` key 字符串没有 layout signature

## 4. 当前最核心的问题

### 4.1 key schema 分裂

当前至少三套 key 同时存在：

1. 通用 `PoolKey`。
2. `LayerPoolKey`。
3. GVA layerwise key。

核心问题：**不同路径对“同一个 chunk 是否可复用”的判断维度不一致。**

直接后果：

- 普通路径和 layerwise 路径 PP rank 语义不一致；
- GVA layerwise 路径缺少 PCP/DCP/PP/cache role/cache family；
- 未来增加 schema version 或 layout signature 时容易漏路径；
- lookup、save、TP mismatch 都在手写/改写 key 字符串。

### 4.2 key 表达 rank，不表达 layout

当前 key 能区分 rank，但不能表达布局兼容关系。

核心问题：**rank 字段适合隔离，不适合表达跨布局复用规则。**

所以当前只能安全支持 exact layout hit；compatible layout hit 需要显式 LayoutDescriptor 和 mapping。

### 4.3 没有 schema version

当前 key 没有版本号。

核心问题：**无法清晰做 v2 灰度、fallback、日志定位和兼容迁移。**

### 4.4 TP mismatch 主路径没接通

当前不是“完整支持 TP mismatch”，而是：

- 推断函数存在；
- worker 内部函数存在；
- 线程类支持 `worker` 参数；
- 但普通线程创建时没传 `worker=self`；
- 所以端到端主路径不支持。

核心问题：**先接通线程主路径并补端到端验证，再谈 TP mismatch 能力。**

## 5. hash key 可以重构哪些

### 5.1 统一 KeyBuilder

建议新增集中模块，例如：

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/key_schema.py
```

里面统一放：

- `KVPoolKeyBuilder`
- `KVPoolKeyFields`
- `KVLayoutDescriptor`
- `HeadShardDescriptor`
- `CPShardDescriptor`
- `parse_key()` 可选，用于调试和迁移

需要接入的现有位置：

- `PoolKey.to_string()`
- `LayerPoolKey.to_string()`
- `ChunkedTokenDatabase._get_key_prefix()`
- `_make_layerwise_gva_key()`
- `_make_layerwise_gva_keys_for_hit_check()`
- `_make_sub_key_str()` / `_replace_key_field()`

目标：**所有路径都通过同一个 KeyBuilder 生成 key，不再散落手写字符串。**

### 5.2 引入 `kvpool:v2` schema version

建议 v2 key 形态：

```text
kvpool:v2
@model:{model_identity}
@layout:{layout_signature}
@role:{cache_role}
@family:{cache_family}
@group:{kv_cache_group_id}
@pp:{pp_rank}
@cp:p{pcp_rank}-d{dcp_rank}
@head:{head_shard_id}
@layer:{layer_id_or_all}
@hash:{chunk_hash}
```

重构收益：

- 新旧 key 可并存；
- lookup 可以 v2 优先、v1 fallback；
- 日志可以直接判断 key 版本；
- 后续加字段不会破坏旧数据。

### 5.3 合并 `PoolKey` 和 `LayerPoolKey` 语义

建议：

- 非 layerwise：`layer=all`
- layerwise：`layer={layer_id}`
- 其他字段完全一致，包括 `pp_rank`

目标：**layerwise 只是通用 key 的一个维度，不是另一套 schema。**

### 5.4 GVA layerwise 改成 legacy 兼容模式

当前 GVA key 是兼容格式：

```text
{model_name}@{block_hash_hex}@{head_or_tp_rank}
{model_name}@{group_id}@{block_hash_hex}@{head_or_tp_rank}
```

建议：

1. v2 key 作为主路径；
2. GVA legacy key 只作为兼容模式；
3. 灰度期 save 可双写 v2 + legacy；
4. lookup v2 优先，legacy fallback；
5. 迁移完成后可以关闭 legacy。

### 5.5 强化 model identity

建议不要只用 basename。

优先级：

1. 支持 `kv_connector_extra_config["model_namespace"]` 显式配置；
2. 默认用模型路径、revision、dtype、quantization、rope scaling 生成 identity hash；
3. 如果启用 LoRA，把 LoRA id/name/revision 纳入 identity，或者明确禁止 LoRA KV pool 复用。

目标：**避免不同模型实例误用同一份 KV。**

### 5.6 引入 LayoutDescriptor

建议新增：

```python
@dataclass(frozen=True)
class KVLayoutDescriptor:
    schema_version: int
    model_identity: str
    tp_size: int
    pp_size: int
    pcp_size: int
    dcp_size: int
    num_kv_heads: int
    head_dim: int
    heads_per_rank: int
    kv_layout: str
    cache_group_id: int
    cache_family: str
    cache_role: str
    use_mla: bool
    use_sparse: bool
    use_hybrid: bool
```

它至少应该派生：

- `layout_signature`：完全同布局才直接 load；
- `compatible_signature`：判断是否可以重排复用；
- `head_shard_map`：TP/KV head mismatch 映射；
- `cp_shard_map`：CP layout 映射。

目标：**先判断 exact / compatible / incompatible，再决定 direct load、mapping load 或 fallback recompute。**

### 5.7 TP mismatch 改成显式 HeadShardMap

当前 TP mismatch 使用 `_replace_key_field()` 改写 `head_or_tp_rank`，把 local rank、effective rank、head shard 混在同一个字段里。

建议改成：

```python
@dataclass(frozen=True)
class HeadShardDescriptor:
    effective_tp_size: int
    head_shard_id: int
    head_start: int
    head_end: int
    local_tp_rank: int
    local_sub_index: int
```

然后由 KeyBuilder 根据 `head_shard_id` 生成 key，store/load 根据 `HeadShardMap` 生成 strided I/O。

目标：**不要通过字符串替换表达 head mapping。**

### 5.8 CP layout mapping 单独建模

如果目标是支持 PD 两侧不同 CP size 的复用，必须新增：

```python
@dataclass(frozen=True)
class CPShardDescriptor:
    pcp_size: int
    dcp_size: int
    pcp_rank: int
    dcp_rank: int
    layout: str
    token_offset: int | None
    token_stride: int | None
```

必须明确：

- CP 是按 token 连续切分还是交错切分；
- block hash 是全局 prefix hash 还是 shard-local hash；
- 跨 CP 复用是聚合完整 KV，还是只允许同 shard 范围复用；
- producer shard 到 consumer shard 的 token range 如何转换。

目标：**跨 CP layout 复用不能靠 rank 猜，必须靠 CP shard map。**

### 5.9 明确多轮对话 cache 模式

建议定义两个模式：

| 模式 | 行为 | 适用场景 |
| --- | --- | --- |
| `prefix_only` | 不保存 decode KV，只复用已经作为 prompt/prefill 保存过的 prefix | 简单稳定，适合作默认 |
| `save_decode` | decode 生成 token 的 KV 也写入 pool，下一轮可复用完整历史 | 能力更强，但需要处理异步保存和 partial chunk |

建议新增配置：

```text
kv_pool_conversation_cache = prefix_only | save_decode
kv_pool_save_partial_decode_chunk = true | false
kv_pool_wait_decode_save_on_finish = true | false
```

目标：**把多轮对话能力从隐式开关变成明确模式。**

## 6. 建议落地顺序

### 阶段 1：先修确定性缺口

1. 普通发送线程创建 `KVCacheStoreSendingThread` 时传入 `worker=self`。
2. 普通接收线程创建 `KVCacheStoreRecvingThread` 时传入 `worker=self`。
3. 接收线程同时传入 worker 侧 `_invalid_block_ids` 和 `_invalid_block_ids_lock`，避免 fallback 状态不一致。
4. 补 P TP=4 / D TP=8 端到端测试。
5. 补 P TP=8 / D TP=4 端到端测试。
6. 保持 TP mismatch + sparse/layerwise/hybrid 的报错测试，不宣传支持。

### 阶段 2：补 key 边界测试

1. 通用 `PoolKey` 字段完整性测试：PCP/DCP、head/TP、PP、group、cache role、cache family。
2. `LayerPoolKey` 与通用 `PoolKey` 字段差异测试，明确当前缺 PP rank。
3. GVA layerwise key 测试，明确其 legacy 格式不含 PCP/DCP/PP/cache role/cache family。
4. grouped hash 测试，确认当前是 chained terminal hash，不是独立 rehash。
5. 同 prefix 不同 group/cache family/PP rank 不误命中测试。

### 阶段 3：统一 KeyBuilder 和 schema version

1. 新增 `key_schema.py`。
2. 引入 `kvpool:v2`。
3. 通用 `PoolKey`、`LayerPoolKey`、GVA key 都通过 KeyBuilder 生成。
4. layerwise 改为 `layer={id}`，非 layerwise 改为 `layer=all`。
5. lookup 支持 v2 优先、v1 fallback。
6. 日志打印 structured key fields。

### 阶段 4：引入 LayoutDescriptor

1. save 时记录 producer layout。
2. lookup 时构造 consumer layout。
3. 判断 exact / compatible / incompatible。
4. TP/KV head mismatch 改用显式 `HeadShardMap`。
5. compatible hit 才允许 strided I/O 或重排。
6. incompatible 直接 fallback recompute。

### 阶段 5：再做 CP 跨布局和多轮对话增强

1. 定义 CP shard map。
2. 支持 producer CP shard 到 consumer CP shard 的转换。
3. 明确多轮对话默认模式是 `prefix_only`。
4. `save_decode` 模式单独工程化 partial chunk、异步保存完成和下一轮等待策略。

## 7. 最终判断

| 能力 | 当前状态 | 工程结论 |
| --- | --- | --- |
| 同模型同布局 prefix 复用 | 主路径 lookup/store 支持 | 支持 |
| PCP/DCP rank 隔离 | key 含 PCP/DCP rank，block/hash 乘 CP scale | 支持隔离 |
| 跨 CP layout 复用 | 无 CP size/layout/mapping | 不支持 |
| 通用 PP/group/cache role/cache family 隔离 | 通用 `PoolKey` 包含这些字段 | 支持 |
| layerwise 与通用 key 统一语义 | `LayerPoolKey` 缺 PP rank | 不支持 |
| GVA layerwise 统一 key 语义 | legacy key 缺 PCP/DCP/PP/cache role/cache family | 不支持 |
| 多轮对话 prefix-only 复用 | token prefix hash 命中 | 支持 |
| decode KV 自动沉淀复用 | 默认 `save_decode_cache=False` | 默认不支持 |
| TP/KV head mismatch 端到端 | 普通线程未传 `worker=self` | 不支持 |
| TP mismatch + sparse/layerwise/hybrid | 初始化直接报错或 NotImplemented | 不支持 |
| 强模型身份隔离 | 只用 basename | 不支持 |
| schema version | key 无版本字段 | 不支持 |
| layout signature | key 无 layout 字段 | 不支持 |

最终结论：**当前 hash key 支持的是 exact layout prefix KV reuse 和基础 namespace 隔离；不支持跨 layout 复用。hash key 重构重点不是继续零散加字段，而是统一 KeyBuilder、引入 schema version、强化 model identity、显式 LayoutDescriptor，并把 exact hit / compatible hit / incompatible 三类命中判定从 key 字符串拼接中拆出来。**
