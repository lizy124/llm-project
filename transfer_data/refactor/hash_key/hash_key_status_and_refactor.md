# KV Pool Hash Key 能力现状与重构方向

一句话结论：**当前 hash key 已经具备比较完整的基础隔离能力，能区分模型、PCP/DCP rank、TP/head rank、PP rank、KV group、cache role、cache family 和 token prefix hash；但它主要解决“同布局下不误用”和“有限 TP/head 重排”的问题，还不能认为完整支持跨 CP layout 复用、多轮 decode KV 自动复用，也缺少统一 schema、显式 layout signature 和可靠的 model identity。**

## 1. 当前 hash key 到底能表达什么

当前通用 KV pool key 不是简单的 `model@hash`，而是由 `PoolKey.to_string()` 生成，核心维度包括：

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

| 维度 | 当前作用 | 能力判断 |
| --- | --- | --- |
| `model_name` | 区分不同模型名 | 有基础隔离，但只用 basename 不够强 |
| `pcp_rank` | 区分 prefill CP rank | 支持 PCP rank 级隔离 |
| `dcp_rank` | 区分 decode CP rank | 支持 DCP rank 级隔离 |
| `head_or_tp_rank` | 区分 KV head shard 或 TP rank | 支持 head/TP rank 级隔离，并为 TP mismatch 提供 namespace |
| `pp_rank` | 区分 pipeline parallel rank | 支持 PP rank 级隔离 |
| `group` | 区分 KV cache group | 支持 hybrid/group 级隔离 |
| `cache_role` | 区分 cache 类型，默认 `kv` | 支持未来不同 cache role 隔离 |
| `cache_family` | 区分 default/c2/mixed 等 cache family | 支持不同 cache layout/family 隔离 |
| `chunk_hash` | 表示 token prefix 内容 | 支持 prefix hash 命中 |

相关代码：

- `config_data.py:78` `KeyMetadata`
- `config_data.py:99` `PoolKey`
- `config_data.py:118` `PoolKey.to_string()`
- `config_data.py:319` `ChunkedTokenDatabase._make_key_by_hash()`

## 2. 当前已经支持的能力

### 2.1 同布局 prefix KV 复用

这是当前最明确、最可靠的能力。

只要 producer 和 consumer 的模型、CP/TP/PP/group/cache family 等布局一致，并且 token prefix hash 一致，就可以通过 hash key 命中并复用 KV。

当前命中判定不是只查一个 key。普通非 layerwise 路径会通过 `LookupKeyClient.lookup()` 到 worker 侧查连续命中；layerwise 路径会在 scheduler 侧生成全 rank、全 layer key 后 `batch_is_exist()`。无论哪条路径，只有所需分片都存在，才认为该 chunk 命中。

能力判断：**支持，同布局场景是当前 hash key 的主能力。**

相关代码：

- `pool_scheduler.py:219` `_generate_store_query_keys()`
- `pool_scheduler.py:257` `_get_store_lookup_hit_tokens()`
- `pool_scheduler.py:527` `LookupKeyClient.lookup()`
- `pool_worker.py:1904` `_expand_lookup_keys_by_rank()`
- `pool_worker.py:1989` `lookup_scheduler()`
- `pool_worker.py:2135` `find_all_continuous_hit_positions()`

### 2.2 PCP/DCP rank 隔离

当前 key 明确包含 `pcp_rank` 和 `dcp_rank`，同时 scheduler 和 worker 都会计算：

```python
cp_scale = pcp_size * dcp_size
hash_block_size = requested_hash_block_size_or_prefix_match_unit * cp_scale
block_size/grouped_block_size = original_block_size * cp_scale
```

这说明当前设计已经意识到 CP 拆分会影响 hash/block 粒度。

当 group block size 大于 hash block size 时，当前代码不是简单取组内最后一个 chained hash，而是用 `_rehash_block_hash_group()` 对一组 block hash 做带 domain 和长度前缀的 SHA256 重哈希，生成新的 grouped hash。

能力判断：**支持 CP rank 隔离，能避免不同 PCP/DCP rank 的 KV 直接混用。**

但要注意：这只是隔离能力，不等于跨不同 CP size/layout 的重排复用能力。

相关代码：

- `pool_scheduler.py:100` `pcp_size/dcp_size`
- `pool_scheduler.py:108` `grouped_block_size/hash_block_size`
- `pool_worker.py:124` `pcp_size/dcp_size`
- `pool_worker.py:144` `grouped_block_size/hash_block_size`
- `config_data.py:589` `get_block_hashes()`
- `config_data.py:604` `_rehash_block_hash_group()`
- `tests/ut/distributed/ascend_store/test_config_data.py:180` grouped hash 单测

### 2.3 PP rank、KV group、cache family 隔离

通用 `PoolKey` 包含 `pp_rank`、`kv_cache_group_id`、`cache_role`、`cache_family`。

这意味着不同 PP stage、不同 KV cache group、不同 cache family 不会落到同一个 key namespace。

能力判断：**支持基础隔离。**

这对 hybrid KV cache、不同 cache family、未来扩展不同 cache role 都是必要基础。

### 2.4 多轮对话的 prefix-only 复用

当前 key 不包含 conversation id，也不包含 request id，主要依赖 token prefix 的 block hash。

因此，如果下一轮请求的 prompt 包含完整历史，并且 token 序列前缀相同，就可以通过 prefix hash 命中已保存的 KV。

能力判断：**支持 token-prefix 级别的多轮复用。**

但这不是 conversation 对象级复用，也不是默认的 decode KV 自动沉淀复用。

相关代码：

- `pool_scheduler.py:98` `save_decode_cache`
- `pool_scheduler.py:823` `_process_running_cached_request()` decode 保存判断

### 2.5 有限 TP/KV head mismatch 设计

当前代码里已经有 TP mismatch 的设计：

- 从 `prefill_tp_size` / `decode_tp_size` 推断 peer TP size。
- 使用 `effective_tp_size = max(local_tp_size, peer_tp_size)` 建立共同 key namespace。
- 当本地 rank 拥有更多 KV heads 时，把本地 heads 拆成多个 sub-key。
- 存取时走 strided I/O，只传输对应 head slice。

能力判断：**有设计、有内部函数和单测覆盖，但当前普通线程入口没有把 worker 接进去，不能认为端到端可用。**

主要原因是发送/接收线程里的 TP mismatch 分支依赖 `self.worker`，`KVCacheStoreSendingThread` / `KVCacheStoreRecvingThread` 构造函数也确实支持 `worker` 参数；但 `KVPoolWorker` 创建普通发送/接收线程时当前没有传入 `worker=self`，所以实际线程路径进不到 `_store_kv_tp_mismatch()` / `_load_kv_tp_mismatch()`。

相关代码：

- `config_data.py:40` `infer_tp_mismatch_info()`
- `pool_worker.py:196` TP mismatch 推断
- `pool_worker.py:489` 创建 `KVCacheStoreSendingThread` 未传 `worker`
- `pool_worker.py:506` 创建 `KVCacheStoreRecvingThread` 未传 `worker`
- `pool_worker.py:1638` `_build_tp_mismatch_keys_and_addrs()`
- `pool_worker.py:1661` `_load_kv_tp_mismatch()`
- `pool_worker.py:1697` `_store_kv_tp_mismatch()`
- `kv_transfer.py:638` sender 构造函数支持 `worker`
- `kv_transfer.py:691` sender TP mismatch 分支
- `kv_transfer.py:857` receiver 构造函数支持 `worker`
- `kv_transfer.py:883` receiver TP mismatch 分支
- `tests/ut/distributed/ascend_store/test_pool_worker.py:1475` TP mismatch 内部单测

## 3. 当前不能认为支持的能力

### 3.1 不能认为完整支持跨 CP layout 复用

当前 key 里有 `pcp_rank` / `dcp_rank`，hash/block 粒度也乘了 `pcp_size * dcp_size`。

这能解决“不同 CP rank 不要误用”的问题，但不能完整表达：

- producer 的 CP size 是多少；
- consumer 的 CP size 是多少；
- token range 如何从 producer shard 映射到 consumer shard；
- CP 分片是连续切分还是交错切分；
- 跨 CP layout 时是否需要聚合、拆分或重排 KV。

能力判断：**当前支持 CP 隔离，不支持完整跨 CP layout 重排复用。**

如果目标是 PD 两侧不同 CP size 也能复用，需要新增显式 CP layout/shard mapping，而不是只靠 key 里的 rank 字段。

### 3.2 不能认为 LayerPoolKey 和通用 PoolKey 语义统一

layerwise key 会在 key 里加入：

```text
@layer_id:{layer_id}
```

但当前 `LayerPoolKey.to_string()` 不包含 `pp_rank`，而通用 `PoolKey.to_string()` 包含 `pp_rank`。

能力判断：**layerwise key 有独立能力，但和通用 key schema 不完全一致。**

这会导致不同路径的命中语义不一致，后续扩展 layout signature 或 schema version 时容易漏字段。

相关代码：

- `config_data.py:145` `LayerPoolKey`
- `config_data.py:165` `LayerPoolKey.to_string()`

### 3.3 不能认为 GVA layerwise key 和通用 key 共用 namespace

memcache/GVA layerwise 路径还有一套兼容 PR #11585 的特殊 key。

单 group：

```text
{model_name}@{block_hash_hex}@{head_or_tp_rank}
```

多 group：

```text
{model_name}@{group_id}@{block_hash_hex}@{head_or_tp_rank}
```

这套 key 没有包含 PCP/DCP/PP/cache_role/cache_family。

能力判断：**GVA layerwise 有自己的兼容 key，但它不是通用 PoolKey schema 的一个变体。**

风险是同一套 KV pool 系统里存在多套 key 语义，后续做统一命中、灰度、迁移会很困难。

相关代码：

- `pool_worker.py:1049` `_make_layerwise_gva_key()`
- `pool_scheduler.py:300` `_make_layerwise_gva_keys_for_hit_check()`

### 3.4 默认不支持 decode KV 自动沉淀后的多轮完整复用

当前默认 `save_decode_cache=False`。

running cached request 如果已经进入 decoding，且没有开启 `save_decode_cache`，不会继续保存 decode 阶段新增 token 的 KV。

能力判断：**默认只支持 prompt/prefix 已保存部分的复用，不支持上一轮生成 token 的 KV 自动沉淀并被下一轮完整复用。**

如果要把多轮对话作为核心能力，需要明确开启并验证 `save_decode_cache=True`，同时处理：

- `discard_partial_chunks` 默认开启时，partial chunk 会被丢弃还是需要保留；
- 最后一个 block 如何处理；
- request finished 时异步保存是否完成；
- 下一轮请求是否需要等待上一轮保存完成。

相关代码：

- `pool_scheduler.py:95` `save_decode_cache` 默认值
- `pool_scheduler.py:124` `discard_partial_chunks` 默认值
- `pool_scheduler.py:803` chunked prefill / decode 保存分支

### 3.5 TP mismatch 不支持若干复杂布局

当前 TP mismatch 路径有限制：

- `infer_tp_mismatch_info()` 只有在非 MLA、非 hybrid、`num_kv_heads >= effective_tp_size` 且能整除时才启用。
- `KVPoolWorker` 初始化阶段如果发现 TP mismatch 叠加 sparse KV layout，会直接 `ValueError`。
- `KVPoolWorker` 初始化阶段如果发现 TP mismatch 叠加 layerwise，会直接 `ValueError`。
- `KVPoolWorker` 初始化阶段如果发现 TP mismatch 叠加 hybrid KV cache，会直接 `NotImplementedError`。
- 实际 `_load_kv_tp_mismatch()` / `_store_kv_tp_mismatch()` 只处理 group 0，不覆盖多 group/hybrid。

能力判断：**TP/KV head 重排只能算有限场景能力，不能覆盖 MLA、sparse、hybrid、layerwise。**

相关代码：

- `config_data.py:40` `infer_tp_mismatch_info()`
- `pool_worker.py:206` sparse/layerwise/hybrid 限制
- `pool_worker.py:1697` `_store_kv_tp_mismatch()` 使用 `block_ids_by_group[0]`
- `tests/ut/distributed/ascend_store/test_pool_worker.py:1536` sparse 限制单测
- `tests/ut/distributed/ascend_store/test_pool_worker.py:1554` layerwise 限制单测

### 3.6 model_name 不足以区分模型实例

当前 key 里的 `model_name` 通常来自 `model_config.model.split('/')[-1]` 或 basename。

这会带来冲突风险：

- 不同路径但 basename 相同；
- basename 相同但权重不同；
- revision、dtype、quantization、rope scaling 不同；
- LoRA 不同。

能力判断：**当前只具备弱 model namespace，不能作为强模型身份。**

如果 KV pool 在多模型、多版本、多 LoRA 场景使用，应该强化 model identity。

## 4. 当前最核心的问题

### 4.1 key schema 不统一

目前至少有三套 key：

1. 通用 `PoolKey`。
2. `LayerPoolKey`。
3. GVA layerwise key。

它们包含的维度不同：

| key 类型 | PCP/DCP | PP | group | cache_role | cache_family | layer | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 通用 `PoolKey` | 有 | 有 | 有 | 有 | 有 | 无 | 主路径 |
| `LayerPoolKey` | 有 | 无 | 有 | 有 | 有 | 有 | 缺 PP rank |
| GVA layerwise key | 无 | 无 | 部分有 | 无 | 无 | 隐含 | 兼容路径 |

核心问题：**不同路径对“同一个 chunk 是否可复用”的判断维度不一致。**

### 4.2 key 表达的是 rank，不是 layout

当前 key 里有 rank 字段，但没有显式 layout 描述。

缺少的信息包括：

- producer TP/CP/PP size；
- consumer TP/CP/PP size；
- num_kv_heads；
- heads_per_rank；
- KV layout order；
- cache tensor layout version；
- hash schema version。

核心问题：**rank 字段适合做隔离，不适合表达跨布局重排规则。**

### 4.3 缺少 schema version

当前 key 没有显式 schema version。

这会让后续升级困难：

- 新旧 key 如何并存不清晰；
- lookup 是 v2 优先还是 v1 fallback 不清晰；
- 出问题时日志不好判断 key 属于哪一代规则。

核心问题：**没有版本化 key，就很难做兼容迁移和灰度。**

### 4.4 TP mismatch 实现需要先接通再谈能力

TP mismatch 现在不是单纯“缺设计”，而是“有设计但当前普通发送/接收线程没有把 worker 接到实际线程路径”。

短期最应该先确认和修复：

- 创建 `KVCacheStoreSendingThread` 时传入 `worker=self`。
- 创建 `KVCacheStoreRecvingThread` 时传入 `worker=self`。
- 补 P TP=4 / D TP=8、P TP=8 / D TP=4 的正反向验证。

核心问题：**没有端到端验证前，TP mismatch 不能算已完整支持。**

## 5. 哪些地方可以重构

### 5.1 统一 KeyBuilder

建议新增集中模块，例如：

```text
ascend_store/key_schema.py
```

里面统一放：

- `KeyBuilder`
- `KVPoolKeyV2`
- `KVLayoutDescriptor`
- `HeadShardDescriptor`
- `parse_key()` 可选，用于调试和迁移

目标是让通用、layerwise、GVA、TP mismatch 都通过同一个 KeyBuilder 生成 key，不再各自手写字符串。

建议 v2 key 形态：

```text
kvpool:v2
@model:{model_identity}
@schema:{hash_schema_version}
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

- 所有路径 key 维度一致；
- layerwise 只是 `layer=N`，非 layerwise 是 `layer=all`；
- GVA 可以作为兼容 key 或 v2 key 的一种生成模式；
- 新旧 key 可以按 schema version 做 fallback。

### 5.2 引入 LayoutDescriptor

建议把布局从零散字段抽象成结构体：

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

它至少应该派生出三类信息：

- `layout_signature`：判断 producer/consumer 是否完全同布局；
- `compatible_signature`：判断是否可以重排复用；
- `head_shard_map` / `cp_shard_map`：描述如何从 producer 映射到 consumer。

重构收益：**从“key 里塞 rank”升级为“先判断布局兼容性，再决定 exact hit、compatible hit 或 fallback”。**

### 5.3 明确命中等级

建议把命中分成三类：

| 命中等级 | 含义 | 行为 |
| --- | --- | --- |
| Exact layout hit | producer/consumer layout 完全一致 | 直接按 key load |
| Compatible layout hit | token hash、model/cache schema 一致，head/TP/CP 可映射 | 按 mapping 重排或 strided I/O |
| Incompatible | hash 即使命中也不能安全复用 | fallback recompute |

重构收益：**把“查到 key”与“KV 可以安全复用”拆开。**

### 5.4 修复 TP/KV head mismatch 路径

短期优先级最高的是把现有路径接通：

- 创建 `KVCacheStoreSendingThread` 时传入 `worker=self`。
- 创建 `KVCacheStoreRecvingThread` 时传入 `worker=self`。

然后补验证：

- P TP=4 / D TP=8；
- P TP=8 / D TP=4；
- `num_kv_heads > TP`；
- `num_kv_heads < TP`，即 `put_step > 1`；
- local rank 拥有多个 effective head shard。

中期再把 `_make_sub_key_str()` 里的隐式 effective rank 规则改成显式 `HeadShardMap`。

### 5.5 增加 CP layout mapping

如果目标是支持 PD 两侧不同 CP size 的复用，需要新增 CP shard 描述：

```python
@dataclass(frozen=True)
class CPShardDescriptor:
    pcp_size: int
    dcp_size: int
    pcp_rank: int
    dcp_rank: int
    token_stride: int | None
    token_offset: int | None
    layout: str
```

同时必须明确：

- PCP/DCP 是按 token 连续切分还是交错切分；
- block hash 是全局 prefix hash 还是每 CP shard 的局部 hash；
- 跨 CP 复用是重新聚合完整 KV，还是只允许相同 shard 范围复用。

重构收益：**让跨 CP 复用从“猜 rank 关系”变成“按 token range 和 shard map 显式转换”。**

### 5.6 明确多轮对话 cache 模式

建议定义两个模式：

| 模式 | 行为 | 适用场景 |
| --- | --- | --- |
| `prefix_only` | 不保存 decode KV，只复用已经作为 prompt/prefill 保存过的 prefix | 简单稳定，适合作默认 |
| `save_decode` | decode 生成 token 的 KV 也写入 pool，下一轮可复用完整历史 | 更强能力，但需要处理异步保存和边界 chunk |

建议新增配置：

```text
kv_pool_conversation_cache = prefix_only | save_decode
kv_pool_save_partial_decode_chunk = true | false
kv_pool_wait_decode_save_on_finish = true | false
```

重构收益：**让多轮对话能力从隐式行为变成明确模式。**

### 5.7 强化 model identity

建议不要只用 basename 作为模型身份。

可选方案：

- 增加 `kv_connector_extra_config['model_namespace']` 显式配置；
- 使用模型路径、revision、dtype、quantization、rope scaling 生成 identity hash；
- 如果启用 LoRA，把 LoRA id/name/revision 纳入 identity，或者明确禁止 LoRA KV pool 复用。

重构收益：**避免不同模型实例误用同一份 KV。**

## 6. 建议落地顺序

### 阶段 1：先确认当前能力边界

1. 在已有 `PoolKey`、`LayerPoolKey`、grouped hash、TP mismatch 内部单测基础上，补齐 PCP/DCP、PP、group、cache_family、GVA key 和组合场景覆盖。
2. 补基础命中测试：同 TP/CP、同 prefix、不同 group/cache family 不误命中。
3. 增加线程入口测试，确认 TP mismatch 时普通发送/接收线程能拿到 `worker` 并进入 worker 分支。

### 阶段 2：修复 TP mismatch 接通问题

1. 给发送线程和接收线程传入 `worker=self`。
2. 验证 P TP=4 / D TP=8。
3. 验证 P TP=8 / D TP=4。
4. 保持并补充 MLA/sparse/hybrid/layerwise 下的 fallback 或报错测试。

### 阶段 3：统一 KeyBuilder 和 schema version

1. 新增 `key_schema.py`。
2. 引入 `kvpool:v2`。
3. 通用 PoolKey、LayerPoolKey、GVA key 都通过 KeyBuilder 生成。
4. lookup 支持 v2 优先、v1 fallback。
5. 日志打印 structured key fields。

### 阶段 4：引入 LayoutDescriptor

1. save 时记录 producer layout。
2. lookup 时构造 consumer layout。
3. 判断 exact / compatible / incompatible。
4. TP/KV head mismatch 改用显式 `HeadShardMap`。

### 阶段 5：再做 CP 跨布局和多轮对话增强

1. 定义 CP shard map。
2. 支持 producer CP shard 到 consumer CP shard 的转换。
3. 明确多轮对话默认是 `prefix_only` 还是 `save_decode`。
4. 补齐 decode save 的边界处理和完成状态。

## 7. 最终判断

当前 hash key 的能力可以这样概括：

| 能力 | 当前状态 | 结论 |
| --- | --- | --- |
| 同模型同布局 prefix 复用 | 已实现 | 支持 |
| PCP/DCP rank 隔离 | 已实现 | 支持隔离 |
| PP/group/cache family 隔离 | 已实现 | 支持隔离 |
| 多轮对话 prefix-only 复用 | 已实现 | 支持 |
| TP/KV head mismatch | 有设计、内部函数和单测，但普通线程未传 worker | 需要修接通和端到端验证 |
| 跨 CP layout 复用 | 只有 rank 隔离和 grouped hash，没有 layout mapping | 不完整支持 |
| layerwise/GVA 统一 key 语义 | 多套 key 并存 | 不支持统一语义 |
| decode KV 自动沉淀复用 | 默认关闭且边界未工程化 | 默认不支持 |
| 强模型身份隔离 | 只用 basename | 不充分 |
| schema version / layout signature | 缺失 | 需要重构 |

因此，重构重点不是简单“再往 key 里加字段”，而是：**统一 key schema，显式描述 layout，先判断布局兼容性，再决定 exact hit、compatible hit 或 fallback。**
