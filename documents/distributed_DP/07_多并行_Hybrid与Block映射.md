# 第 7 章：多并行、Hybrid KV Cache 与 block/rank 映射

P/D 分离最容易出错的地方不是单次 Mooncake 传输，而是“当前 D rank 到底应该从哪些 P rank 拉哪些 blocks”。本章专门解释 `MooncakeConnector` 中的映射逻辑。

## 7.1 需要处理的并行维度

| 维度 | P 侧 | D 侧 | 影响 |
|------|------|------|------|
| TP | `prefill.tp_size` | `decode.tp_size` / 当前 engine TP | 决定 KV head shard 如何组合 |
| PP | `prefill.pp_size` | 当前要求 `decode.pp_size == 1` | D 侧要按 P PP stage 拉不同层范围 |
| PCP/DCP | `remote_pcp_size` / `remote_dcp_size` | `pcp_size` / `dcp_size` | 决定 prompt blocks 在 context parallel 中如何分布 |
| DP | P/D 服务实例数 | P/D 服务实例数 | 通过 Proxy 和 data_parallel_rank 选择实例 |
| HMA/Hybrid | 多 KV group | 多 KV group | 每个 group 的 block 粒度和 state 语义不同 |

## 7.2 TP 拉取数量：tp_num_need_pulls

D rank 需要从几个 P TP rank 拉取，取决于 D rank 持有的 KV heads 是否比 P rank 多。

```python
if model_config.is_deepseek_mla:
    self.tp_num_need_pulls = 1
else:
    num_d_block_heads = max(1, num_key_value_heads // self.tp_size)
    num_p_block_heads = max(1, num_key_value_heads // self._prefill_tp_size)
    self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads
```

含义：

- MLA：KV head 逻辑特殊，通常视作 1。
- P TP 大于 D TP 时，每个 P rank 负责更少 KV heads，D rank 可能要拉多个 P rank 的 shard。
- 当前要求 `prefill_tp_size >= decode_tp_size`。

## 7.3 选择远端 P rank

简单情况下，`_get_remote_ranks_for_req()` 返回每个 D rank 对应的 P rank 列表。

### 7.3.1 P TP == D TP

```python
if prefill_tp_size == self._decode_tp_size:
    return [
        [tp + pp * prefill_tp_size for pp in range(self._prefill_pp_size)]
        for tp in range(prefill_tp_size)
    ]
```

例：`prefill_tp_size=2, prefill_pp_size=2`

| D TP rank | P ranks |
|-----------|---------|
| 0 | `[0, 2]` |
| 1 | `[1, 3]` |

D rank 0 从 P 的 PP0/PP1 中 TP0 拉，D rank 1 从 P 的 PP0/PP1 中 TP1 拉。

### 7.3.2 P TP > D TP

当 P TP 大于 D TP，代码会按请求 id 做稳定 hash：

```python
seed = string_to_int64_hash(req_id)
rand = random.Random(seed)
rand_group_index = rand.sample(range(num_groups), max(self._decode_tp_size // num_kv_head, 1))
```

目的：

- 在有冗余 KV head group 时，为不同请求稳定选择一组 P ranks。
- 同一个 `req_id` 每次选择结果一致。
- 避免所有请求固定打到同一组 P ranks。

## 7.4 PP 层范围

D 侧需要知道 P 的每个 PP rank 负责哪些层：

```python
def get_prefill_pp_indices(num_hidden_layers, pp_rank, pp_size, partition_list_str=None):
    if partition_list_str is None:
        return get_pp_indices(num_hidden_layers, pp_rank, pp_size)
    partitions = [int(layer) for layer in partition_list_str.split(",")]
    start_layer = sum(partitions[:pp_rank])
    end_layer = start_layer + partitions[pp_rank]
    return (start_layer, end_layer)
```

如果配置了：

```python
"prefill": {
    "pp_size": 2,
    "pp_layer_partition": "20,28"
}
```

那么：

| P PP rank | layer range |
|-----------|-------------|
| 0 | `[0, 20)` |
| 1 | `[20, 48)` |

D 侧 transfer 时只拉属于该 PP rank 的层。

## 7.5 PCP/DCP 映射

当 P 或 D 开启 context parallel 时，prompt blocks 不再简单地全部属于一个 rank。`_get_kv_split_metadata()` 会进入复杂分支：

```python
if meta.remote_pcp_size * meta.remote_dcp_size * self.pcp_size * self.dcp_size == 1:
    # 简单路径
else:
    # PCP/DCP 路径
```

复杂路径会做：

1. 校验 P/D context parallel 参数。
2. 构造 P 侧 KV head group 到 CP group 的映射。
3. 构造 D 侧 KV head group 到 CP group 的映射。
4. 计算当前 D handshake port 对应哪些 P ports。
5. 考虑 prefix cache：已经本地命中的 blocks 不再远端拉取。
6. 保证最后一个可能不满的 block 被放到最后一个 D shard。

关键变量：

| 变量 | 含义 |
|------|------|
| `remote_block_nums_all` | P 侧每个 CP rank 上剩余可拉 blocks 数 |
| `num_prefix_cached_blocks` | D 侧已通过 prefix cache 命中的 blocks 数 |
| `local_cp_rank` | 当前 D rank 在 D CP 组中的位置 |
| `remote_handshake_port_list` | 当前 D rank 要访问的 P 端口组合 |

## 7.6 Hybrid KV Cache group

当前 `MooncakeConnector` 支持 HMA/Hybrid KV Cache。初始化时会构建：

```python
self.kv_group2layeridx = self._build_kv_group2layeridx()
self._is_hma_required = not disable_hybrid_kv_cache_manager and any(
    not isinstance(g.kv_cache_spec, FullAttentionSpec)
    for g in kv_cache_config.kv_cache_groups
)
```

`kv_group2layeridx` 结构：

```python
{
    group_id: (
        serialized_group_spec,
        [layer_idx0, layer_idx1, ...]
    )
}
```

它会被放入 `MooncakeAgentMetadata`，让远端知道：

- 每个 group 是 `FullAttentionSpec`、`SlidingWindowSpec` 还是 `MambaSpec`
- 每个 group 对应哪些实际 layer index
- 对应 cache tensor 的 shape/dtype/block 信息

## 7.7 GroupTransferInfo

Scheduler 端为每个 group 计算：

```python
@dataclass(frozen=True)
class GroupTransferInfo:
    tokens_per_block: int
    blocks_per_window: int
    is_state_group: bool
```

构造逻辑：

```python
tokens_per_block = block_size * compress_ratio
blocks_per_window = cdiv(sliding_window, block_size) + 1 if sliding_window else 0
is_state_group = any(isinstance(spec, MambaSpec) for spec in specs)
```

用途：

- `tokens_per_block`：确定 prompt 需要传多少 attention blocks。
- `blocks_per_window`：SWA group 只传 window tail。
- `is_state_group`：Mamba/state group 不按普通 context block 截断。

## 7.8 GroupPull metadata

Hybrid 场景下，不同 group 可能需要不同的 rank 选择和 TP pulls。

```python
def _get_hybrid_remote_rank_group_pulls(self, req_id, prefill_tp_size):
    for group_id, (group_spec, layer_indices) in self.kv_group2layeridx.items():
        if group_spec["kv_cache_spec_type"] == "MambaSpec":
            num_group_pulls = prefill_tp_size // self.tp_size
            ...
        else:
            num_group_pulls = self._get_attention_group_num_need_pulls(group_spec, prefill_tp_size)
            chosen_rank_list = self._get_remote_rank(req_id, prefill_tp_size)
            ...
```

Mamba group 约束：

```python
assert prefill_tp_size % self.tp_size == 0
```

也就是 P 侧 TP 必须能被 D 侧 TP 整除。

Attention group 的 `num_group_pulls`：

```python
num_key_value_heads = self._get_attention_group_num_key_value_heads(group_spec)
num_d_block_heads = max(1, num_key_value_heads // self.tp_size)
num_p_block_heads = max(1, num_key_value_heads // prefill_tp_size)
return num_d_block_heads // num_p_block_heads
```

## 7.9 block_size_scale 与 kernel block 展开

P/D 的逻辑 block 和实际 KV tensor 第一维可能不一致。代码用 `block_size_scale` 展开：

```python
def expand_block_ids(block_ids, scale):
    return [bid * scale + offset for bid in block_ids for offset in range(scale)]

local_scale = self.block_size_scale[layer_indices[0]][0]
remote_scale = remote_block_size_scale[layer_indices[0]][0]
kernel_local_block_ids = expand_block_ids(local_group_block_ids, local_scale)
kernel_remote_block_ids = expand_block_ids(remote_group_block_ids, remote_scale)
```

这样当一个逻辑 block 对应多个 kernel blocks 时，传输仍按实际内存连续性计算。

## 7.10 prefix cache 差异修正

D 侧可能已经通过本地 prefix cache 命中了一部分 token，所以不需要从 P 拉完整 prompt。

```python
num_computed_tokens = req_meta.get("num_computed_tokens", 0)
remote_kernel_block_size = self.block_size // remote_scale
remote_kernel_token_size = remote_kernel_block_size * self.group_compress_ratios[group_idx]
remote_start_idx = num_computed_tokens // remote_kernel_token_size
kernel_remote_block_ids = kernel_remote_block_ids[remote_start_idx:]
```

作用：跳过远端已经由 D 本地 prefix cache 覆盖的 blocks，只拉剩余 KV。

## 7.11 连续 block 合并

为了减少 Mooncake transfer item 数，代码会合并连续 blocks：

```python
group_concurrent_contiguous(src, dst, src_block_stride, dst_block_stride, block_len)
```

只有当 source block id、destination block id 和实际 byte stride 都连续时才合并。

如果 remote/local block stride 不等于 `block_len`，会调用：

```python
split_if_not_byte_contiguous(...)
```

重新拆开不连续的区间，避免把非连续内存当成连续传输。

## 7.12 reformat 条件

传输结束后，可能需要 reformat：

| 条件 | 操作 |
|------|------|
| `num_group_pulls > 1` | 多个 P TP shard 拼接为 D rank 的 KV head 布局 |
| `enable_kv_nz=True` | 转换为 NPU NZ KV Cache 布局 |
| HMA/hybrid linear | `reformat_kv_cache_hybrid_linear_torch()` |

代码只在 `is_group_transfer_end=True` 的 group shard 后做 reformat，确保所有分片已经到齐。

## 7.13 Mamba/state group 特殊处理

Mamba state 不是按普通 context blocks 存储的 KV：

- pull 模式中只传最终 state block。
- conv / ssm 分别计算地址。
- TP ratio > 1 时按 linear key/value head 维度切片。

相关函数：

```python
_append_mamba_transfer_meta(...)
```

它会生成 conv 和 ssm 的 `src/dst/length` 元数据。

## 7.14 当前限制

从代码和文档可见的主要限制：

- `prefill_tp_size >= decode_tp_size`。
- `decode.pp_size == 1`。
- 标准 connector 中 `pp_size > 1` 和 `pcp_size > 1` 当前不能同时开启。
- Mamba group 要求 P/D TP 满足整除关系。
- 非对称 TP 下，P TP 必须能支持 D 侧所需 KV head 拼接。
- Layerwise push 和标准 pull 的 metadata/完成确认机制不同，不要混用。
