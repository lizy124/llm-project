# KV Pool Hash Key 现状与重构梳理

## 1. 范围

本梳理面向：

- `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store`
- 重点关注 KV pool 的 hash key 组成、命中判定、PD 不同 CP/TP/KV head 布局复用、多轮对话复用能力。

核心文件：

- `pool_scheduler.py`
- `pool_worker.py`
- `config_data.py`
- `kv_transfer.py`

## 2. 当前 hash key 体系

### 2.1 KeyMetadata / PoolKey

当前通用池化 key 由 `PoolKey.to_string()` 序列化，格式大致为：

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

对应字段含义：

| 字段 | 含义 | 当前作用 |
| --- | --- | --- |
| `model_name` | 模型名 basename | 防止不同模型 KV 混用 |
| `pcp_rank` | prefill context parallel rank | 区分 prefill CP 分片 |
| `dcp_rank` | decode context parallel rank | 区分 decode CP 分片 |
| `head_or_tp_rank` | KV head rank 或 TP rank | 区分 KV head/TP 分片 |
| `pp_rank` | pipeline parallel rank | 区分 PP 层分片 |
| `group` | KV cache group id | 区分 hybrid KV cache group |
| `cache_role` | cache 类型，默认 `kv` | 为未来 state/cache 类型隔离预留 |
| `cache_family` | cache family，如 `default`/`c2`/`mixed` | 区分压缩/混合缓存布局 |
| `chunk_hash` | vLLM block hash/chained hash | 表示 token prefix 内容 |

相关代码：

- `config_data.py:72` `KeyMetadata`
- `config_data.py:94` `PoolKey`
- `config_data.py:114` `PoolKey.to_string()`
- `config_data.py:283` `ChunkedTokenDatabase._get_key_prefix()`

### 2.2 LayerPoolKey

layerwise key 会在 `PoolKey` 基础上追加：

```text
@layer_id:{layer_id}
```

但 `LayerPoolKey.to_string()` 当前不包含 `pp_rank`，而 `PoolKey.to_string()` 包含 `pp_rank`。
这意味着 layerwise key 和非 layerwise key 的维度并不完全一致。

相关代码：

- `config_data.py:140` `LayerPoolKey`
- `config_data.py:161` `LayerPoolKey.to_string()`

### 2.3 GVA layerwise key

memcache/GVA layerwise 路径还有一套兼容 PR #11585 的特殊 key：

单 group：

```text
{model_name}@{block_hash_hex}@{head_or_tp_rank}
```

多 group：

```text
{model_name}@{group_id}@{block_hash_hex}@{head_or_tp_rank}
```

相关代码：

- `pool_worker.py:1062` `_make_layerwise_gva_key()`
- `pool_scheduler.py:307` `_make_layerwise_gva_keys_for_hit_check()`

这套 key 没有包含 PCP/DCP/PP/cache_family/cache_role，因此和通用 PoolKey 不是同一套 namespace。

## 3. chunk_hash / block 粒度现状

### 3.1 hash_block_size 与 CP scale

当前 scheduler 和 worker 都会计算：

```python
cp_scale = pcp_size * dcp_size
hash_block_size = requested_hash_block_size_or_prefix_match_unit * cp_scale
block_size/grouped_block_size = original_block_size * cp_scale
```

相关代码：

- `pool_scheduler.py:103` `pcp_size/dcp_size`
- `pool_scheduler.py:111` `grouped_block_size/hash_block_size`
- `pool_worker.py:145` `grouped_block_size/hash_block_size`

这说明当前 key 粒度已经意识到 CP 分片后的逻辑块大小，避免不同 CP 拆分下直接用相同 hash 粒度造成错配。

### 3.2 get_block_hashes

当 group block size 大于 hash block size 时，使用 `_LazyGroupedBlockHashList`，取组内最后一个 chained hash 代表更大的 chunk。

相关代码：

- `config_data.py:638` `get_block_hashes()`
- `config_data.py:649` `_LazyGroupedBlockHashList`

这依赖 vLLM block hash 是 chained hash：后一个 hash 能代表此前 prefix。

## 4. 命中判定现状

### 4.1 scheduler 侧 lookup

非 layerwise 常规路径：

1. scheduler 调用 `LookupKeyClient.lookup()`。
2. lookup RPC 把 `token_len`、`kv_cache_group_ids`、`hbm_hit_tokens`、`block_hashes` 发给 worker。
3. worker 的 `lookup_scheduler()` 生成本 rank 的 base key，然后展开所有 rank variant。
4. 对所有 PP rank 与 group TP rank 做 `exists()`。
5. 只有所有相关 rank 对应 chunk 都存在，才认为该 chunk 命中。

相关代码：

- `pool_scheduler.py:544` `LookupKeyClient.lookup()`
- `pool_worker.py:2082` `lookup_scheduler()`
- `pool_worker.py:1974` `_expand_lookup_keys_by_rank()`
- `pool_worker.py:2215` `find_all_continuous_hit_positions()`

### 4.2 hybrid / coordinator 路径

如果有 hybrid KV cache，会优先走 `AscendStoreCoordinator`，使用 group mask / retention mask / EAGLE 相关规则计算最终命中长度。

相关代码：

- `pool_worker.py:1888` `_lookup_with_coordinator()`
- `coordinator.py`

### 4.3 layerwise 路径

普通 layerwise 路径会检查每一层 key 是否存在，必须全部 layer 存在才算该 chunk 命中。

相关代码：

- `pool_scheduler.py:537` `_get_store_lookup_hit_tokens(... include_layers=True)`
- `pool_worker.py:2185` `check_all_layers_exists()`

GVA layerwise 路径则使用专门的 GVA key，要求所有 head/tp rank key 都能返回有效 GVA。

相关代码：

- `pool_scheduler.py:320` `_get_layerwise_gva_hit_tokens()`
- `pool_worker.py:1196` `_prepare_load_gvas()`

## 5. 当前能力判断

### 5.1 PD 不同 CP 情况

当前已经把 PCP/DCP 维度放进 key，并且 hash/block 粒度乘了 `pcp_size*dcp_size`。

已具备能力：

- 不同 PCP rank 不会混用。
- 不同 DCP rank 不会混用。
- lookup 时会按 PCP/DCP/head_or_tp/PP/group 维度生成全量 key。

当前风险：

- key 中保存的是 rank 维度，而不是一个显式的“布局签名”。
- 如果 prefill/decode 两侧 CP size 不同，仅靠 `pcp_rank/dcp_rank` 与乘 scale 的 block 粒度，不一定能表达“旧布局如何映射到新布局”。
- 现有代码更像是“隔离不同 CP 分片”，而不是“跨不同 CP 布局重排复用”。

判断：

- 支持 CP 维度隔离。
- 对 PD 不同 CP 的跨布局复用还不能认为完整支持，需要增加明确的 layout mapping 和数据重排语义。

### 5.2 不同 KV head / TP 重排

当前已有 TP mismatch 设计：

- 从 `prefill_tp_size` / `decode_tp_size` 推断 peer TP size。
- 用 `effective_tp_size = max(local_tp_size, peer_tp_size)` 建立共同 key namespace。
- 当 local rank 拥有更多 heads 时，把本地 KV heads 拆成多个 sub-key。
- 存取时走 strided I/O，只传输子 head slice。

相关代码：

- `config_data.py:36` `infer_tp_mismatch_info()`
- `pool_worker.py:1654` `_make_sub_key_str()`
- `pool_worker.py:1688` `_build_tp_mismatch_keys_and_addrs()`
- `pool_worker.py:1721` `_load_kv_tp_mismatch()`
- `pool_worker.py:1757` `_store_kv_tp_mismatch()`

限制：

- 不支持 MLA。
- 不支持 sparse KV layout。
- 不支持 layerwise。
- 不支持 hybrid KV cache。
- 要求 `num_kv_heads >= effective_tp_size` 且能整除。

重要问题：

- `KVCacheStoreSendingThread` 和 `KVCacheStoreRecvingThread` 的 TP mismatch 分支依赖 `self.worker`。
- 但 `pool_worker.py` 创建这两个线程时没有传入 `worker=self`。
- 因此当前实际路径大概率不会进入 `_store_kv_tp_mismatch()` / `_load_kv_tp_mismatch()`。

相关代码：

- `pool_worker.py:495` 创建 `KVCacheStoreSendingThread`
- `pool_worker.py:512` 创建 `KVCacheStoreRecvingThread`
- `kv_transfer.py:668` sender TP mismatch 分支
- `kv_transfer.py:912` receiver TP mismatch 分支

判断：

- 不同 KV head/TP 重排当前是“有设计、有部分实现”，但还不能认为完整可用。
- 至少需要接通 worker 引用，并补全正反向复用验证。

### 5.3 多轮对话

当前 key 不包含 conversation id，也不包含 request id。主要依赖 token prefix 的 block hash。

这意味着：

- 如果下一轮请求 prompt 中包含完整历史，且 token 序列前缀相同，则可以通过 prefix block hash 命中。
- 这属于 token-prefix 级复用，不是会话对象级复用。

decode 阶段保存：

- 默认 `save_decode_cache=False`。
- running cached request 如果已经进入 decoding，且未开启 `save_decode_cache`，不会继续保存 decode 增量 KV。

相关代码：

- `pool_scheduler.py:98` `save_decode_cache`
- `pool_scheduler.py:823` `_process_running_cached_request()` decode 保存判断

判断：

- 支持“多轮对话带完整历史 prompt 后的 prefix 复用”。
- 默认不支持“上一轮生成 token 的 KV 自动沉淀并被下一轮完整复用”。
- 如果要把多轮对话作为核心能力，应明确开启并验证 `save_decode_cache=True` 的行为，尤其是部分 chunk、边界 token、最后 block、request finished 后异步保存完成顺序。

## 6. 当前主要问题清单

### 6.1 key schema 不统一

目前至少存在三套 key：

1. 通用 PoolKey。
2. LayerPoolKey。
3. GVA layerwise key。

它们包含的维度不同：

- 通用 PoolKey 有 PCP/DCP/PP/group/cache_role/cache_family。
- LayerPoolKey 缺少 PP rank。
- GVA layerwise key 缺少 PCP/DCP/PP/cache_role/cache_family。

风险：

- 不同路径之间命中语义不一致。
- 未来增强 CP/head/layout 复用时容易漏维度。
- 兼容旧 key 与新 key 的策略不清晰。

### 6.2 key 里缺少明确 layout signature

当前通过多个字段间接表达布局：

- pcp/dcp rank
- head_or_tp_rank
- pp_rank
- group/cache_family

但没有显式记录：

- producer TP/CP/PP size
- consumer TP/CP/PP size
- num_kv_heads
- heads_per_rank
- KV layout order
- cache tensor layout version
- hash schema version

风险：

- “隔离”容易，“跨布局复用”困难。
- 同一个 key namespace 无法表达重排规则。
- key schema 变化时不易兼容和灰度。

### 6.3 TP mismatch 实现未完全接通

现有代码已经准备了 strided I/O，但线程创建未传 worker，导致分支可能无法生效。

此外 TP mismatch 当前不支持 hybrid/layerwise/sparse/MLA。

### 6.4 多轮对话复用边界不清晰

当前更像 prefix cache，而不是 conversation cache。

需要明确：

- 是否要缓存 decode token KV。
- 是否允许 partial last block。
- 何时认为一轮对话的生成结果已经安全写入 pool。
- 下一轮请求如何等待或感知上一轮保存完成。

### 6.5 model_name 可能不足以区分模型实例

当前使用 `model_config.model.split('/')[-1]` 或 basename。

风险：

- 不同路径但 basename 相同的模型会冲突。
- 相同 basename 但权重、量化方式、rope scaling、LoRA、revision 不同会冲突。

建议至少纳入 `model_identity` 或可配置 namespace。

## 7. 重构目标建议

### 7.1 建立统一 Key Schema

建议引入显式版本化 key：

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

原则：

- 所有路径都走同一个 KeyBuilder。
- 普通/layerwise/GVA 只是在字段上有 `layer=all` 或 `layer=N` 的差异。
- 不再手写字符串 replace。
- 保留旧 key fallback，用于兼容已有缓存。

### 7.2 引入 LayoutDescriptor

建议把布局从零散字段抽象为结构体：

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

再派生：

- `layout_signature`：用于判断是否完全同布局。
- `compatible_signature`：用于判断是否可重排复用。
- `head_shard_map`：用于 producer/consumer head 重排。

### 7.3 明确三类复用模式

建议分成三种命中等级：

1. **Exact layout hit**
   - producer/consumer layout 完全一致。
   - 直接按 key load。

2. **Compatible layout hit**
   - token hash 一致。
   - model/cache schema 一致。
   - head/TP/CP 可通过规则映射。
   - 需要按 mapping 做重排或 strided I/O。

3. **Incompatible**
   - hash 即使命中也不能复用。
   - 直接回退 recompute。

### 7.4 重构 KeyBuilder

建议新增集中模块，例如：

```text
ascend_store/key_schema.py
```

包含：

- `KeyBuilder`
- `KVPoolKeyV2`
- `KVLayoutDescriptor`
- `HeadShardDescriptor`
- `parse_key()` 可选，用于调试与迁移

调用方：

- `KVPoolScheduler._generate_store_query_keys()`
- `ChunkedTokenDatabase._get_key_prefix()`
- `ChunkedTokenDatabase._make_key_by_hash()`
- `KVPoolWorker._make_layerwise_gva_key()`
- TP mismatch 的 `_make_sub_key_str()` 应替换成结构化生成，不再字符串替换。

### 7.5 修复并完成 TP/KV head mismatch 路径

短期必须修复：

- 创建 `KVCacheStoreSendingThread` 时传入 `worker=self`。
- 创建 `KVCacheStoreRecvingThread` 时传入 `worker=self`。

然后验证：

- P TP=4 / D TP=8。
- P TP=8 / D TP=4。
- num_kv_heads 大于 TP。
- num_kv_heads 小于 TP，即 `put_step > 1`。
- local rank 拥有多个 effective head shard。

中期增强：

- 用 `HeadShardMap` 代替 `effective_rank = tp_rank * num_sub_keys + sub_idx` 的隐式规则。
- 支持 head 顺序变化，而不仅是 TP size 变化。
- 明确 MLA/hybrid/sparse 不支持时的错误信息和 fallback。

### 7.6 CP 不同布局复用

如果目标是支持 PD 两侧不同 CP size 的复用，建议不要只依赖 key rank 字段。

需要新增：

- CP shard descriptor。
- token range 到 CP shard 的映射。
- producer CP layout 到 consumer CP layout 的转换。
- lookup 时按 consumer 需要的 token range 去查询 producer 可提供的 shard。

建议数据结构：

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

需要明确的问题：

- PCP/DCP 是按 token 连续切分还是交错切分。
- block hash 是全局 prefix hash 还是每 CP shard 的局部 hash。
- 跨 CP 复用是重新聚合完整 KV，还是只允许相同 shard 范围复用。

### 7.7 多轮对话支持策略

建议定义两个模式：

#### 模式 A：prefix-only

- 不保存 decode KV。
- 下一轮 prompt 只有历史中已作为 prompt/prefill 计算过的部分能命中。
- 简单稳定。

#### 模式 B：conversation-continuation

- 开启 `save_decode_cache=True`。
- decode 生成 token 的 KV 也按 chunk 写入 pool。
- request finished 前必须保证异步保存完成或有状态可查询。
- 下一轮 prompt 包含上一轮 generated tokens 时，可以命中完整历史。

建议新增配置：

```text
kv_pool_conversation_cache = prefix_only | save_decode
kv_pool_save_partial_decode_chunk = true | false
kv_pool_wait_decode_save_on_finish = true | false
```

### 7.8 model identity 强化

建议 key 中不要只用 basename。

可选方案：

- `kv_connector_extra_config['model_namespace']` 显式配置。
- 模型路径 + revision + dtype + quantization + rope scaling hash。
- 如果启用 LoRA，纳入 lora id/name/revision，或者明确禁止 LoRA KV pool 复用。

## 8. 建议落地步骤

### 阶段 1：现状修复与测试补齐

1. 接通 TP mismatch worker 分支。
2. 增加 key 生成单测，覆盖：
   - PCP/DCP rank
   - PP rank
   - group/cache_family
   - layerwise
   - GVA layerwise
   - TP mismatch effective rank
3. 增加端到端/集成验证：
   - 相同 TP/CP 命中。
   - 不同 TP 非 hybrid 非 layerwise 命中。
   - decode cache 关闭/开启两种多轮对话行为。

### 阶段 2：KeyBuilder 统一

1. 新增 `key_schema.py`。
2. PoolKey/LayerPoolKey/GVA key 统一走 KeyBuilder。
3. 引入 `schema_version=v2`。
4. lookup 支持 v2 优先，v1 fallback。
5. 日志打印 structured key fields，便于排查。

### 阶段 3：LayoutDescriptor 与兼容性判断

1. 新增 `KVLayoutDescriptor`。
2. save 时把 producer layout 写入 key 或 side metadata。
3. lookup 时判断 exact/compatible/incompatible。
4. TP/KV head mismatch 用显式 `HeadShardMap`。

### 阶段 4：CP layout 跨布局复用

1. 明确 PCP/DCP token 分片语义。
2. 定义 CP shard map。
3. 支持查询 producer shard 并重组 consumer 所需 KV。
4. 对无法重组的布局直接 fallback recompute。

### 阶段 5：多轮对话工程化

1. 明确默认模式为 `prefix_only` 还是 `save_decode`。
2. 补齐 decode save 的边界处理。
3. finished 请求与异步保存完成之间建立可靠状态。
4. 增加多轮对话命中率与正确性测试。

## 9. 当前结论

当前 hash key 已经不是简单 `model@hash`，而是包含：

- model
- PCP rank
- DCP rank
- head/TP rank
- PP rank
- KV cache group
- cache role
- cache family
- chunk hash

因此基础隔离能力已经较强。

但如果目标是“key 泛化能力重构增强”，当前仍有明显不足：

1. key schema 不统一，通用/layerwise/GVA 三套 key 维度不一致。
2. 不同 KV head/TP 重排有设计但实际路径疑似未完全接通。
3. PD 不同 CP 更偏隔离，不是完整跨 CP layout 复用。
4. 多轮对话默认只支持 prefix hash 复用，不默认保存 decode KV。
5. 缺少显式 layout signature / schema version / model identity。

建议优先做：

1. 修复 TP mismatch worker 接通问题。
2. 统一 KeyBuilder。
3. 引入 layout descriptor。
4. 明确多轮对话是否启用 decode KV 保存。
