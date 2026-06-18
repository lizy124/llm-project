# 第 7 章：ChunkedTokenDatabase - 键管理与地址计算

[`ChunkedTokenDatabase`](../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py) 是整个池化系统的核心数据结构，负责将 token 序列映射为存储 key，以及根据 block ID 计算 NPU 内存地址。

## 7.1 核心数据结构

### 7.1.1 KeyMetadata

描述一个 KV Cache 分片的完整身份信息：

```python
@dataclass
class KeyMetadata:
    model_name: str          # 模型名称
    head_or_tp_rank: int     # TP rank（或 KV head rank）
    pcp_rank: int            # Prefill Context Parallel rank
    dcp_rank: int            # Decode Context Parallel rank
    pp_rank: int             # Pipeline Parallel rank
    kv_cache_group_id: int   # KV Cache 组 ID（混合 Attention 场景）
    cache_role: str = "kv"   # "kv" 或 "state"
    cache_family: str = "default"  # 压缩族（如 "c1", "c4", "c128"）
```

### 7.1.2 PoolKey

将 `KeyMetadata` 和一个 chunk 的 hash 组合成唯一的池 key：

```python
@dataclass(order=True)
class PoolKey:
    key_metadata: KeyMetadata
    chunk_hash: str          # 该 chunk 的 SHA-256 hash（hex 字符串）

    def to_string(self):
        return (
            f"{model_name}"
            f"@pcp{pcp_rank}@dcp{dcp_rank}"
            f"@head_or_tp_rank:{head_or_tp_rank}"
            f"@pp_rank:{pp_rank}"
            f"@group:{kv_cache_group_id}"
            f"@cache_role:{cache_role}"
            f"@cache_family:{cache_family}"
            f"@{chunk_hash}"
        )
```

**Key 示例**：
```
DeepSeek-V4@pcp0@dcp0@head_or_tp_rank:3@pp_rank:0@group:0@cache_role:kv@cache_family:c1@a1b2c3d4...
```

### 7.1.3 LayerPoolKey

继承自 `PoolKey`，增加了 `layer_id` 字段，用于 Layerwise 传输模式：

```python
@dataclass(order=True)
class LayerPoolKey(PoolKey):
    layer_id: int

    def to_string(self):
        return (
            f"{model_name}"
            f"@pcp{pcp_rank}@dcp{dcp_rank}"
            f"@head_or_tp_rank:{head_or_tp_rank}"
            f"@group:{kv_cache_group_id}"
            f"@cache_role:{cache_role}"
            f"@cache_family:{cache_family}"
            f"@layer_id:{layer_id}"
            f"@{chunk_hash}"
        )
```

## 7.2 ChunkedTokenDatabase 初始化

```python
class ChunkedTokenDatabase:
    def __init__(self, metadata, block_size, partitions, use_hybrid=False, hash_block_size=None):
        self.metadata = metadata           # list[KeyMetadata]，每个 KV 组对应一个
        self.block_size = block_size       # list[int]，每个组的 block_size
        self.partitions = partitions       # PP 分区配置（consumer 模式）
        self.use_hybrid = use_hybrid       # 是否混合 KV Cache
        self.hash_block_size = hash_block_size or self.block_size[0]
        
        # 运行时设置的 buffer 信息
        self.group_kv_caches_base_addr: dict[int, list[int]] = {}  # group_id → [base_addr, ...]
        self.group_block_len: dict[int, list[int]] = {}            # group_id → [block_len, ...]
        self.group_block_stride: dict[int, list[int]] = {}         # group_id → [block_stride, ...]
        self.group_cache_families: dict[str, dict[int, str]] = {}   # cache_role → {group_id: family}
        self.group_num_layers: dict[str, dict[int, int]] = {}       # cache_role → {group_id: num_layers}
```

## 7.3 process_tokens：Token → Key 映射

这是最重要的方法，将 token 序列切分为 chunk，并为每个 chunk 生成池存储 key。

```python
def process_tokens(
    self,
    token_len: int,
    block_hashes: BlockHashList | list[str],
    mask_num: int = 0,
    kv_cache_group_id: int = 0,
    cache_role: str = "kv",
    cache_family: str | None = None,
) -> Iterable[tuple[int, int, PoolKey]]:
    """返回 (start_token_idx, end_token_idx, PoolKey) 的迭代器"""
    
    # 1. 获取该组的 block_size（考虑压缩比）
    group_block_size = self.get_block_size(kv_cache_group_id)  # 如 128
    if cache_family is None:
        cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
    cache_family_ratio = max(infer_cache_family_ratio(cache_family), 1)  # 如 c4 → 4
    group_block_size *= cache_family_ratio  # 128 * 4 = 512
    
    # 2. 重新哈希 block_hashes（如果 hash_block_size < group_block_size）
    block_hashes = get_block_hashes(block_hashes, group_block_size, self.hash_block_size)
    
    # 3. 确保 hashes 是字符串格式
    if not isinstance(block_hashes[0], str):
        block_hashes = [h.hex() for h in block_hashes]
    
    # 4. 遍历每个 chunk，生成 key
    start_idx = 0
    for chunk_id, hash_val in enumerate(block_hashes):
        start_idx = chunk_id * group_block_size
        if start_idx >= token_len:
            break
        end_idx = min(start_idx + group_block_size, token_len)
        
        # 跳过已缓存的 token（mask_num 之前的部分）
        if start_idx < mask_num:
            continue
        
        # 反归一化到原始 block_size
        start_idx //= cache_family_ratio
        end_idx //= cache_family_ratio
        if end_idx <= start_idx:
            continue
        
        yield (start_idx, end_idx, self._make_key_by_hash(hash_val, kv_cache_group_id, cache_role, cache_family))
```

### 7.3.1 分块示例

假设 `block_size=128`，`cache_family="c1"`（ratio=1），`token_len=500`，`mask_num=0`：

| chunk_id | hash_val | start | end | key |
|----------|----------|-------|-----|-----|
| 0 | hash_0 | 0 | 128 | `...@hash_0` |
| 1 | hash_1 | 128 | 256 | `...@hash_1` |
| 2 | hash_2 | 256 | 384 | `...@hash_2` |
| 3 | hash_3 | 384 | 500 | `...@hash_3` |

**有已缓存 token 时**（`mask_num=200`）：

| chunk_id | start | end | 是否跳过 |
|----------|-------|-----|---------|
| 0 | 0 | 128 | 跳过（<200） |
| 1 | 128 | 256 | 跳过（128<200） |
| 2 | 256 | 384 | 不跳过 |
| 3 | 384 | 500 | 不跳过 |

### 7.3.2 压缩组的特殊处理

对于 DeepSeek V4 的 c4 压缩组（`cache_family="c4"`, `ratio=4`）：

- `group_block_size = 128 * 4 = 512`
- 4 个原始 block 的 hash 被合并为一个 group hash
- 最终 `start` 和 `end` 会除以 4，映射回原始的 128-token block 坐标

## 7.4 process_tokens_with_block_ids

在 `process_tokens` 基础上，额外返回 block_id：

```python
def process_tokens_with_block_ids(self, token_len, block_hashes, block_ids, mask_num=0, ...):
    for start_idx, end_idx, key in self.process_tokens(
        token_len, block_hashes, mask_num, kv_cache_group_id, cache_role, cache_family,
    ):
        # 计算对应的 block ID
        block_idx = start_idx // self.get_block_size(kv_cache_group_id)
        if block_idx >= len(block_ids):
            continue
        block_id = block_ids[block_idx]
        
        # 跳过 null blocks（block_id <= 0）
        if skip_null_blocks and block_id <= 0:
            continue
        
        yield start_idx, end_idx, key, block_id
```

## 7.5 prepare_value：Token 范围 → NPU 内存地址

将 token 的 start/end 范围和 block_id 映射为实际的 NPU 内存地址列表。

```python
def prepare_value(self, start, end, block_ids, kv_cache_group_id=0, cache_role="kv"):
    group_block_size = self.get_block_size(kv_cache_group_id)  # 如 128
    block_id = block_ids[start // group_block_size]
    group_addrs, group_block_len, group_block_stride = self._get_group_buffers(kv_cache_group_id)
    
    addr_list = []
    size_list = []
    length = len(group_block_len)  # 每层的 cache tensor 数量（K/V = 2）
    
    for index, base_addr in enumerate(group_addrs):
        block_len = group_block_len[index % length]
        block_stride = group_block_stride[index % length] if group_block_stride else block_len
        
        # ★ 核心公式：addr = base_addr + block_id * block_stride
        addr = base_addr + block_id * block_stride
        
        # 按比例计算实际大小
        size = int(block_len / group_block_size * (end - start))
        
        addr_list.append(addr)
        size_list.append(size)
    
    return addr_list, size_list, block_id
```

### 7.5.1 地址计算示意图

假设一个 KV Cache 组的 buffer 信息如下：

```
group_addrs = [k_base, v_base]        # K cache 和 V cache 的基地址
group_block_len = [k_block_len, v_block_len]  # 每个 block 的字节数
group_block_stride = [k_stride, v_stride]     # 相邻 block 的字节步长
```

对于 `block_id=3`, `start=0`, `end=128`, `group_block_size=128`：

```
k_addr = k_base + 3 * k_stride
k_size = k_block_len  (因为 end - start == 128 == todo_block_size)

v_addr = v_base + 3 * v_stride
v_size = v_block_len
```

## 7.6 prepare_value_layer：Layerwise 模式的地址计算

```python
def prepare_value_layer(self, start, end, block_ids, layer_id):
    group_block_size = self.get_block_size(0)
    block_id = block_ids[start // group_block_size]
    group_addrs, group_block_len, group_block_stride = self._get_group_buffers(0)
    
    addr_list = []
    size_list = []
    length = len(group_block_len)
    
    for i in range(length):
        block_stride = group_block_stride[i] if group_block_stride else group_block_len[i]
        # ★ layer_id * length 偏移到目标层
        addr = group_addrs[layer_id * length] + block_id * block_stride
        size = int(group_block_len[i] / group_block_size * (end - start))
        addr_list.append(addr)
        size_list.append(size)
    
    return addr_list, size_list, block_id
```

在 Layerwise 模式中，所有层的地址是连续存储的，`group_addrs` 包含所有层所有 cache 的地址。当前代码通过 `layer_id * length` 定位到具体层。

> 注意：当前实现中循环变量 `i` 只用于选择 `block_stride` / `group_block_len`，地址基址写作 `group_addrs[layer_id * length]`，而不是更直观的 `group_addrs[layer_id * length + i]`。这意味着同一层内多个 cache tensor 会共用同一个 base address。该行为与当前代码一致，但如果后续排查 Layerwise K/V 地址错位问题，应优先验证这里是否需要加上 `+ i`。

## 7.7 get_block_hashes：块的再哈希

当 `hash_block_size < group_block_size` 时，需要将多个小块的 hash 合并为一个大块的 hash：

```python
def get_block_hashes(block_hashes, group_block_size, hash_block_size):
    if group_block_size == hash_block_size:
        return block_hashes
    
    scale_factor = group_block_size // hash_block_size
    return [
        _rehash_block_hash_group(block_hashes[idx : idx + scale_factor])
        for idx in range(0, len(block_hashes) // scale_factor * scale_factor, scale_factor)
    ]

def _rehash_block_hash_group(block_hashes):
    hasher = hashlib.sha256()
    hasher.update(b"vllm-ascend-grouped-block-hash-v1\0")
    hasher.update(len(block_hashes).to_bytes(4, "big"))
    for block_hash in block_hashes:
        hash_bytes = bytes(block_hash)
        hasher.update(len(hash_bytes).to_bytes(4, "big"))
        hasher.update(hash_bytes)
    return BlockHash(hasher.digest())
```

## 7.8 get_cache_family_granularity：压缩组的传输粒度

```python
def get_cache_family_granularity(block_size: int, cache_family: str | None) -> int:
    return block_size * infer_cache_family_ratio(cache_family)

def infer_cache_family_ratio(cache_family: str | None) -> int:
    if not cache_family or not cache_family.startswith("c"):
        return 1
    ratio = cache_family[1:]
    return int(ratio) if ratio.isdigit() else 1
```

例如：
- `c1` → granularity = 128 × 1 = 128
- `c4` → granularity = 128 × 4 = 512
- `c128` → granularity = 128 × 128 = 16384

对于 DeepSeek V4 的混合组，最终 `cache_transfer_granularity = lcm(128, 128, 512, 16384) = 16384`。

## 7.9 数据流总结

```
Token 序列
    │
    ▼
process_tokens()
    │  将 token 按 block_size 分块
    │  为每个块生成 PoolKey
    │
    ▼
(start, end, PoolKey) 迭代器
    │
    ├──→ pool_worker.lookup_scheduler()
    │        │  按当前 rank / 当前 group 生成 key
    │        │  hybrid 场景先筛选 lookup gate group
    │        │  调用 Backend.exists()
    │        └→ 返回 hit 的 token 数量
    │
    ├──→ prepare_value(start, end, block_ids)
    │        │  block_id = block_ids[start // block_size]
    │        │  addr = base_addr + block_id * block_stride
    │        └→ 返回 addr_list, size_list
    │
    └──→ Backend.put(key_list, addr_list, size_list)
         或 Backend.get(key_list, addr_list, size_list)
```