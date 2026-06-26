# 18. HMA 与 KV cache layout：它和 Attention 有什么关系？

源码位置：

- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/config/scheduler.py`
- `code/vllm/vllm/v1/kv_cache_interface.py`
- `code/vllm/vllm/v1/core/kv_cache_utils.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/kv_cache_coordinator.py`
- `code/vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `code/vllm/vllm/v1/core/block_pool.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/block_table.py`
- `code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/factory.py`
- `code/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py`

本文用于梳理 HMA / hybrid KV cache manager / KV cache group / layout 与 attention 的关系，说明 HMA 不是 attention 类型本身，而是影响 attention KV cache 管理、block table、slot mapping、attention metadata 和 connector 兼容性的内存组织机制。

---

## 1. 本文要回答的问题

```text
HMA 是什么？
它和 hybrid KV cache manager 有什么关系？
KV cache groups 如何影响 attention metadata？
connector 为什么要声明 supports HMA？
HMA 如何影响 KV cache layout、block table 和 attention backend？
HMA 与 Mamba / hybrid attention 模型有什么关系？
HMA、packed layout、cross-layer layout 是不是一回事？
```

---

## 2. 一句话回答

HMA 可以理解为 vLLM v1 的 **hybrid KV cache manager / hybrid memory allocator**：

```text
它不是一种 attention 算法，
而是当模型里存在多种 KV cache 需求时，
把不同 layer 分成多个 KV cache group，
分别维护 block table / prefix cache / slot mapping，
再把这些 group 重新映射到每个 attention layer 的内存管理机制。
```

它和 attention 的关系是：

```text
attention layer 决定自己需要什么 KVCacheSpec
  → HMA 根据 KVCacheSpec 把 layer 分成 KV cache groups
  → Scheduler / KVCacheManager 按 group 分配 blocks
  → Worker / ModelRunner 按 group 构造 block table 和 slot mapping
  → Attention metadata 按 layer 拿到对应 group 的 block table / slot mapping
  → Attention backend 用这些 metadata 读写 KV cache
```

如果只记一句话：

```text
HMA 管的是“不同 attention/cache 类型如何共享和拆分 KV cache 资源”，不是“attention 怎么算”。
```

---

## 3. 为什么需要 HMA

普通 LLaMA 类模型通常所有层都是同一种 full attention，KV cache 规格一致：

```text
layer.0  full attention
layer.1  full attention
...
layer.N  full attention
```

这种情况下只需要一个 KV cache group：

```text
KV cache group 0：所有 attention layers
```

所有层共享同一套请求级 block table，attention backend 每层按自己的 KV tensor 读写即可。

但 hybrid 模型不是这样。例如：

```text
full attention + sliding window attention
full attention + chunked local attention
full attention + MLA / SWA MLA
attention + Mamba / SSM cache
encoder self attention + decoder cross attention
```

这些层的 KV cache 生命周期、可保留 token 范围、最大内存需求、block 语义可能不同。例如 sliding window attention 只需要窗口内 KV，而 full attention 需要完整前缀。如果仍然按 full attention 给所有层分配完整 KV，会浪费显存。

HMA 解决的问题就是：

```text
不同 KV cache 类型分别管理，
但对 Scheduler / Worker / Attention backend 仍提供统一执行接口。
```

---

## 4. HMA 不是 attention backend

attention backend 负责计算：

```text
Q, K, V → attention output
```

HMA 负责管理：

```text
哪些 layer 使用哪种 KV cache spec；
每个请求在每个 group 里有哪些 block；
prefix cache 命中如何跨 group 对齐；
哪些 block 可以释放 / 复用；
block table 和 slot mapping 如何按 group 传给 attention backend。
```

二者边界可以这样看：

| 层次 | 负责内容 | 典型对象 |
|---|---|---|
| attention 类型 | full / sliding window / MLA / local / cross 等计算语义 | `FullAttentionSpec`、`SlidingWindowSpec`、`MLAAttentionSpec` |
| HMA / KV manager | 按不同 KV 规格分组、分配 block、维护 prefix cache | `KVCacheGroupSpec`、`KVCacheManager`、`HybridKVCacheCoordinator` |
| Worker block layout | 把请求 block ids 变成 block table / slot mapping | `BlockTables` |
| attention metadata | 把 group 级 block table 映射到 layer 级 metadata | `CommonAttentionMetadata`、`AttentionMetadataBuilder` |
| attention backend | 按 metadata 执行 kernel | FlashAttention / FlashInfer / Triton / Mamba backend |

所以 HMA 不会改变某层是 sliding window attention 还是 full attention；它改变的是这层 KV cache 的 block 管理方式。

---

## 5. 关键数据结构

### 5.1 KVCacheSpec：每层声明自己的 KV cache 需求

`kv_cache_interface.py` 里定义了多种 KV cache spec：

```text
KVCacheSpec
  ├─ AttentionSpec
  │   ├─ FullAttentionSpec
  │   ├─ MLAAttentionSpec
  │   ├─ ChunkedLocalAttentionSpec
  │   ├─ SlidingWindowSpec
  │   ├─ SlidingWindowMLASpec
  │   ├─ EncoderOnlyAttentionSpec
  │   ├─ CrossAttentionSpec
  │   └─ SinkFullAttentionSpec
  ├─ MambaSpec
  └─ UniformTypeKVCacheSpecs
```

这些 spec 描述的是：

```text
block_size
page_size_bytes
num_kv_heads
head_size
dtype
sliding_window / attention_chunk_size
Mamba state shape
cross attention encoder cache size
是否按 block stride 索引 KV
```

其中 `AttentionSpec` 是 attention layer 的 KV cache 规格；`MambaSpec` 不是 attention，但它也走同一套 KV/cache 管理框架。

### 5.2 KVCacheGroupSpec：把多个 layer 归为一个 KV cache group

`KVCacheGroupSpec` 表示：

```text
一组共享同一套 block table 语义的 layer。
```

结构核心是：

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    is_eagle_group: bool = False
```

含义：

```text
layer_names：这个 group 里有哪些模型层；
kv_cache_spec：这个 group 用什么 KV cache 规格；
is_eagle_group：是否包含 EAGLE / MTP draft attention layers。
```

### 5.3 KVCacheConfig：最终给 Scheduler 和 Worker 的 KV 配置

`KVCacheConfig` 包含三类信息：

```python
@dataclass
class KVCacheConfig:
    num_blocks: int
    kv_cache_tensors: list[KVCacheTensor]
    kv_cache_groups: list[KVCacheGroupSpec]
```

可以理解为：

```text
num_blocks：每个 KV pool 可用 block 数；
kv_cache_tensors：Worker 初始化 KV cache tensor 时怎么分配物理内存；
kv_cache_groups：Scheduler / Worker 按哪些 group 管理 block table。
```

### 5.4 KVCacheBlocks：一次分配结果必须带 group 维度

`KVCacheManager.allocate_slots()` 返回的是 `KVCacheBlocks`，它的核心字段是：

```python
blocks: tuple[Sequence[KVCacheBlock], ...]
```

外层 tuple 是 KV cache group：

```text
blocks[0] = group 0 的 blocks
blocks[1] = group 1 的 blocks
...
```

`get_block_ids()` 返回：

```text
tuple[list[int], ...]
```

这就是为什么 HMA 场景下不能只传一个 `list[int]`。请求在不同 KV cache group 里可能有不同 block 列表。

---

## 6. HMA 的启动期主链路

HMA 在启动期主要做两件事：

```text
1. 判断是否启用 hybrid KV cache manager；
2. 根据每层 KVCacheSpec 生成 KVCacheConfig。
```

主链路可以概括为：

```text
VllmConfig post init
  → 决定 disable_hybrid_kv_cache_manager
  → 收集每层 KVCacheSpec
  → get_kv_cache_groups()
  → get_kv_cache_config_from_groups()
  → KVCacheConfig
  → Scheduler 初始化 KVCacheManager
  → Worker / ModelRunner 初始化 KV cache tensors 与 BlockTables
```

---

## 7. disable_hybrid_kv_cache_manager 的含义

配置字段在 `SchedulerConfig` 中：

```text
disable_hybrid_kv_cache_manager: bool | None = None
```

它有三种语义：

```text
None：用户没有明确要求，vLLM 自动判断；
False：用户显式启用 HMA，如果当前配置不支持则报错；
True：用户显式禁用 HMA。
```

`VllmConfig` 的 post init 里会根据平台、模型、connector 自动决定是否需要禁用 HMA。

典型自动禁用条件包括：

```text
- 当前平台不支持 hybrid KV cache；
- chunked local attention + EAGLE 当前不支持；
- chunked local attention 默认因延迟回退而关闭 HMA，除非设置环境变量允许；
- 配置了 KV connector，但 connector 不支持 HMA。
```

如果用户显式启用 HMA，但运行时发现必须禁用，则直接抛错。

---

## 8. 禁用 HMA 并不是“不用 KV cache”

当 `disable_hybrid_kv_cache_manager=True` 时，vLLM 会调用：

```python
unify_hybrid_kv_cache_specs(kv_cache_spec)
```

它尝试把 hybrid KV specs 规约成单一类型，例如：

```text
SlidingWindowSpec → FullAttentionSpec
ChunkedLocalAttentionSpec → FullAttentionSpec
SlidingWindowMLASpec → MLAAttentionSpec
```

这样 Scheduler 侧可以按单一 KV cache 类型管理 blocks。

但注意：

```text
禁用 HMA 后，计算层面的 sliding window / local attention 仍然可以按原 attention 逻辑计算；
变化的是 KV cache manager 不再按窗口释放/节省对应 KV blocks。
```

源码警告也说明了这一点：

```text
不启用节省 KV cache memory 的优化，
例如不再丢弃 sliding window 外的 KV cache；
但 sliding window 等层的 compute 仍然可以节省。
```

因此：

```text
disable HMA = KV 管理退化为更保守的 full-cache 管理，
不是 attention backend 全部变成 full attention。
```

---

## 9. get_kv_cache_groups：逻辑分组

`get_kv_cache_groups()` 的输入是：

```text
dict[layer_name, KVCacheSpec]
```

输出是：

```text
list[KVCacheGroupSpec]
```

核心分支：

```text
1. 如果 disable_hybrid_kv_cache_manager=True
   → 先 unify_hybrid_kv_cache_specs()

2. 如果是 attention-free 模型
   → 返回空 groups

3. 如果所有层 KV spec 完全一致
   → 生成一个 group

4. 如果所有层属于同一种 KV cache 类型，但 hidden size 等细节不同
   → 生成一个 UniformTypeKVCacheSpecs group

5. 如果是 DeepSeek V4 一类特殊混合形态
   → group_and_unify_kv_cache_specs()

6. 通用 hybrid 模型
   → unify_kv_cache_spec_page_size()
   → _get_kv_cache_groups_uniform_page_size()
```

这一步的关键点是：

```text
HMA 的 group 是按 KV cache spec / attention type / page size 组织的，
不是简单按 layer index 平均切分。
```

---

## 10. hybrid 模型如何分组

`_get_kv_cache_groups_uniform_page_size()` 的注释给了一个典型例子：

```text
模型有 10 个 full attention layers，20 个 sliding window layers。

可以认为它重复了：
  1 * full + 2 * sliding window
这个 pattern 10 次。

因此生成 3 个 KV cache groups：
  group 0：10 个 full layers
  group 1：10 个 sliding window layers
  group 2：10 个 sliding window layers
```

为什么不是 2 个 group？

```text
因为 group 的目标不是“按 attention 类型粗分”，
而是让每个 group 有相同数量的 layer，
从而可以用固定的共享 tensor / block table 结构映射回所有层。
```

这也是 HMA 的核心设计：

```text
把模型层模式拆成若干 KV cache groups，
每个 group 代表 pattern 中的一个位置，
每个 group 内包含多次重复出现的同类 layer。
```

---

## 11. get_kv_cache_config_from_groups：物理布局生成

`get_kv_cache_groups()` 解决的是逻辑分组；`get_kv_cache_config_from_groups()` 解决的是物理内存布局。

输入：

```text
kv_cache_groups
available_memory
```

输出：

```text
KVCacheConfig(num_blocks, kv_cache_tensors, kv_cache_groups)
```

主要分支：

### 11.1 attention-free

没有 KV cache group 时，返回：

```text
num_blocks = 1
kv_cache_tensors = []
```

这里仍保留一个 block 是因为 `BlockPool` 需要 null block。

### 11.2 单 group + UniformTypeKVCacheSpecs

当所有层是同一种类型，但每层 page size 可能不同，会按 layer 分别创建 `KVCacheTensor`。

```text
每个 layer 一个 tensor，大小按该 layer 的 page_size_bytes * num_blocks 计算。
```

### 11.3 packed layout

如果启用 packed HMA layout，或者特殊模型需要 packed layout，会走：

```python
_get_kv_cache_config_packed(...)
```

这会生成带 `offset` / `block_stride` 的 `KVCacheTensor`，用于更紧凑地打包不同 page size 的 layer。

### 11.4 通用 multi-group layout

通用 HMA 下，会让不同 group 中相同 slot index 的 layer 共享一个 tensor：

```text
例如 groups：
  group 0: full.0, full.1
  group 1: sw.0, sw.2
  group 2: sw.1, padding

group_size = 2

物理 tensor 0 shared_by:
  full.0, sw.0, sw.1

物理 tensor 1 shared_by:
  full.1, sw.2
```

注意：

```text
共享 tensor 不代表共享 block table。
不同 KV cache group 仍有独立 block table 和 slot mapping。
```

---

## 12. KVCacheManager 与 HybridKVCacheCoordinator

`KVCacheManager` 是 Scheduler 侧对外入口：

```text
get_computed_blocks()
allocate_slots()
free()
remove_skipped_blocks()
get_blocks()
get_block_ids()
cache_blocks()
take_events()
```

它内部不直接实现所有 hybrid 逻辑，而是持有：

```python
self.coordinator = get_kv_cache_coordinator(...)
```

根据 KV cache groups 数量选择：

```text
单 group → UnitaryKVCacheCoordinator
多 group → HybridKVCacheCoordinator
无 prefix cache → KVCacheCoordinatorNoPrefixCache
```

`HybridKVCacheCoordinator` 的职责是：

```text
协调多个 SingleTypeKVCacheManager，
让它们在 prefix cache hit、block allocation、block free、skipped block removal 上保持一致。
```

---

## 13. HybridKVCacheCoordinator 做了什么

### 13.1 验证和拆分 groups

初始化时会调用：

```python
verify_and_split_kv_cache_groups()
```

它会把 KV cache groups 按 spec 聚成 attention groups，方便批量处理 cache hit。

同时 full attention 会被排在前面：

```text
full attention 的从左到右 cache hit 扫描可以提供更紧的初始上界，
后续 sliding window / local / Mamba 类 group 再在这个上界内检查。
```

### 13.2 find_longest_cache_hit

HMA 场景下 prefix cache hit 不能只看一个 group。

原因：

```text
full attention group 可能命中很长；
sliding window group 可能因为窗口/保留策略只命中一部分；
最终请求可跳过的 token 必须对所有相关 group 都安全。
```

因此 `HybridKVCacheCoordinator.find_longest_cache_hit()` 用 fixed-point 方式计算：

```text
从候选 hit_length 开始；
让每种 attention/cache group 检查是否接受这个长度；
如果某个 group 缩短 hit_length，则重新检查；
直到所有 group 都接受。
```

### 13.3 cache_blocks

缓存 block 时也必须按 group 对齐：

```text
cache hit 长度按 scheduler_block_size 对齐；
SWA / local 类 group 可能只缓存某些窗口相关 block；
EAGLE group 可能额外处理 lookahead block。
```

### 13.4 remove_skipped_blocks

sliding window / local attention 的一个重要收益是：

```text
窗口外 blocks 可以从请求持有列表中移除，
并用 null block 占位或释放。
```

这正是 HMA 对 KV 显存节省的核心来源之一。

---

## 14. BlockPool：真正的 block 资源池

`BlockPool` 管理底层 `KVCacheBlock`：

```text
blocks：所有 block 对象；
free_block_queue：可分配 / 可驱逐 block 队列；
cached_block_hash_to_block：prefix cache 哈希索引；
null_block：特殊占位 block；
kv_event_queue：KV cache 事件。
```

几个关键行为：

```text
get_new_blocks()：从 free queue 分配新 block；
touch()：prefix cache 命中时增加 ref_cnt；
free_blocks()：释放请求持有的 block；
evict_blocks()：从 prefix cache hash 中驱逐；
cache_full_blocks()：把 full block 加入 prefix cache；
cache_partial_block()：注册 partial prefix cache entry。
```

HMA 并不绕开 BlockPool。HMA 只是让上层有多个 KV cache group / manager 视图，最终 block 的生命周期仍落到 BlockPool。

---

## 15. Scheduler 运行期主链路

HMA 在 Scheduler 中的主链路可以简化为：

```text
请求进入 scheduler
  → get_computed_blocks()
      → HybridKVCacheCoordinator.find_longest_cache_hit()
      → 返回各 group 的 cached blocks + computed tokens

  → allocate_slots()
      → coordinator.remove_skipped_blocks()
      → coordinator.get_num_blocks_to_allocate()
      → coordinator.allocate_new_computed_blocks()
      → coordinator.allocate_new_blocks()
      → coordinator.cache_blocks()
      → 返回 KVCacheBlocks

  → SchedulerOutput 携带 block_ids
      → block_ids 是 tuple[list[int], ...]
      → 外层按 KV cache group 区分
```

普通单 group 模型中：

```text
block_ids = ([1, 2, 3],)
```

HMA 模型中可能是：

```text
block_ids = (
  [10, 11, 12],    # group 0: full attention
  [20, 21],        # group 1: sliding window
  [30, 31],        # group 2: sliding window
)
```

这就是 HMA 对下游 Worker / connector 的直接影响。

---

## 16. Worker / ModelRunner 初始化如何感知 HMA

Worker 侧初始化 KV cache 时，`ModelRunner.initialize_kv_cache()` 会读取 `kv_cache_config.kv_cache_groups`。

它会为每个 group 收集：

```text
block_size
max_num_blocks_per_group
kernel_block_size
```

然后初始化：

```python
self.block_tables = BlockTables(
    block_sizes=block_sizes,
    max_num_reqs=self.max_num_reqs,
    max_num_batched_tokens=self.max_num_tokens,
    max_num_blocks_per_group=max_num_blocks_per_group,
    kernel_block_sizes=self.kernel_block_sizes,
    ...
)
```

因此：

```text
HMA group 数量直接决定 Worker 侧 block table 的第一维。
```

单 group：

```text
slot_mappings shape = [1, max_num_batched_tokens]
```

multi group：

```text
slot_mappings shape = [num_kv_cache_groups, max_num_batched_tokens]
```

---

## 17. BlockTables：HMA 到 attention metadata 的关键桥梁

`BlockTables` 是 Worker 侧多 group block table 的实现。

核心字段：

```text
block_tables：每个 KV cache group 一个 staged block table；
input_block_tables：forward 时使用的 block table；
slot_mappings：每个 KV cache group 一条 slot mapping；
num_blocks：每个 group / request 当前有多少 blocks；
block_sizes：manager block size；
kernel_block_sizes：attention backend 实际 kernel block size。
```

### 17.1 append_block_ids

Scheduler 输出的 `tuple[list[int], ...]` 会按 group 写入：

```text
for each group:
  new_block_ids[group] → block_tables[group][req_index]
```

如果 manager block size 和 kernel block size 不一致，会展开：

```text
blocks_per_kv_block = block_size // kernel_block_size

manager block 7
  → kernel block 7*bpk + 0
  → kernel block 7*bpk + 1
  → ...
```

这说明：

```text
Scheduler 管的是 manager block；
attention backend 可能需要更细的 kernel block。
```

### 17.2 gather_block_tables

真正 forward 前，会把当前 batch 的 request index 映射到输入 block table：

```text
request state index → batch row
```

这样 attention backend 看到的是本轮 batch 顺序下的 block table。

### 17.3 compute_slot_mappings

slot mapping 的公式本质是：

```text
position → block_index + block_offset
block_number = block_table[request, block_index]
slot_id = block_number * block_size + block_offset
```

HMA 下这个计算按 group 重复：

```text
slot_mapping[group, token_idx]
```

如果启用了 context parallel，还会根据 rank 过滤本 rank 不负责的 token slot，把它们置为 `PAD_SLOT_ID`。

---

## 18. block table 和 slot mapping 的区别

二者经常被混淆。

```text
block table：请求级结构
  request row → block ids

slot mapping：token 级结构
  token position → physical KV slot
```

举例：

```text
block_size = 16
request 0 的 block_table = [5, 9]

token position 0  → block 5 offset 0  → slot 80
token position 15 → block 5 offset 15 → slot 95
token position 16 → block 9 offset 0  → slot 144
```

HMA 下每个 group 都有自己的 block table 和 slot mapping：

```text
group 0 block table → group 0 slot mapping
group 1 block table → group 1 slot mapping
group 2 block table → group 2 slot mapping
```

---

## 19. Attention metadata 如何按 group 构造

Worker 侧在 `prepare_attn()` 中生成：

```text
block_tables：num_kv_cache_groups x [num_reqs, max_num_blocks]
slot_mappings：num_kv_cache_groups x [num_tokens]
```

随后 `build_attn_metadata()` 按 group 组装 metadata：

```text
for kv_cache_group_id in range(num_kv_cache_groups):
    block_table = block_tables[group_id]
    slot_mapping = slot_mappings[group_id]

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=...,
        seq_lens=...,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        positions=...,
        ...
    )

    for attention group in attn_groups[group_id]:
        metadata = builder.build(common_attn_metadata)
        for layer_name in attention group:
            attn_metadata[layer_name] = metadata
```

最终 attention layer 不是按 group id 直接取 metadata，而是按 layer name 取：

```text
layer_name → AttentionMetadata
```

但这个 metadata 内部已经绑定了对应 KV cache group 的：

```text
block_table_tensor
slot_mapping
seq_lens
positions
```

---

## 20. HMA 对 attention backend 的影响

attention backend 不需要知道“这是 HMA 模型”这个概念，但它会受到 HMA 产物影响：

```text
1. 当前 layer 对应哪套 block table；
2. 当前 token 的 slot_mapping 属于哪个 group；
3. KV cache tensor 的物理 shape / stride / layout；
4. block_size 与 kernel_block_size 是否一致；
5. 是否存在 sliding window / local / MLA / Mamba 等特定 metadata。
```

所以可以说：

```text
HMA 不改变 attention backend 的数学语义，
但改变 attention backend 收到的 KV cache 索引和 metadata 来源。
```

---

## 21. metadata builder 缓存：为什么可以复用

旧版 `gpu_model_runner.py` 里有一段注释说明：

```text
在 hybrid KV-cache groups 之间缓存 attention metadata build。
如果相同 metadata builder 和 KVCacheSpec 唯一变化只是 block table，
并且 builder 支持 update_block_table，
就可以复用 metadata，仅更新 block table。
```

含义是：

```text
HMA 会增加 group 数量；
如果每个 group 都完整重建 metadata，会增加开销；
对于结构相同、只换 block table 的情况，可以复用 builder 结果。
```

这进一步说明：

```text
HMA 对 attention metadata 的主要差异是 block table / slot mapping，
而不是每次都改变 backend 算法。
```

---

## 22. HMA 与 Mamba / SSM 模型

`MambaSpec` 也继承自 `KVCacheSpec`，但它不是 attention spec。

Mamba 的 cache 可能包含：

```text
conv state
SSM state
speculative blocks
mamba_cache_mode = none / align / all
```

它仍然通过 KV cache manager 管理，是因为执行层需要统一处理：

```text
请求状态；
block allocation；
prefix cache / skipped block；
zeroing；
worker cache tensor 初始化；
model runner metadata。
```

对于 attention + Mamba 的 hybrid SSM 模型，HMA 更重要：

```text
attention layers 和 Mamba layers 的 cache 规格不同，
很难通过 disable_hybrid_kv_cache_manager 统一成一种 FullAttentionSpec。
```

配置警告里也明确提到：

```text
hybrid SSM models（例如 Jamba、Bamba）require HMA，
如果因为 connector 不支持 HMA 而关闭，启动会失败。
```

因此：

```text
Mamba 不是 attention，
但 Mamba + attention 的混合 cache 管理是 HMA 的典型应用场景。
```

---

## 23. 为什么 connector 必须声明 SupportsHMA

KV connector 负责外部 KV cache 的 load / save / transfer。

普通单 group connector 在请求结束时只需要：

```python
request_finished(request, block_ids: list[int])
```

HMA 下请求结束时必须拿到所有 group：

```python
request_finished_all_groups(
    request,
    block_ids: tuple[list[int], ...],
)
```

`SupportsHMA` 就是这个能力声明。

核心原因：

```text
请求的 KV 不再是一条 block_ids；
而是 group 维度的多条 block_ids。

connector 如果只保存 group 0，
就会丢失 sliding window / Mamba / cross-attention 等其他 group 的 cache。
```

Scheduler 的 `_connector_finished()` 里有明确分支：

```text
如果 connector 不支持 HMA：
  assert len(kv_cache_groups) == 1
  调 request_finished(request, block_ids[0])

如果 connector 支持 HMA：
  调 request_finished_all_groups(request, block_ids)
```

因此：

```text
SupportsHMA 不是性能优化标志，
而是 connector 是否能正确理解多 KV cache group 生命周期的正确性边界。
```

---

## 24. connector HMA 自动检查

`KVConnectorFactory.supports_hma_config()` 会检查 connector class 是否支持 HMA。

规则：

```text
普通 connector：
  connector class 必须 subclass SupportsHMA

MultiConnector：
  所有 child connectors 都必须支持 HMA
```

创建 connector 时，如果 HMA enabled 但 connector 不支持，会报错：

```text
Connector xxx does not support HMA but HMA is enabled.
Please set --disable-hybrid-kv-cache-manager.
```

如果用户没有显式启用 HMA，而 connector 不支持，`VllmConfig` 会自动关闭 HMA，并给出警告。

---

## 25. Worker 侧 KV connector 与 HMA 的关系

Worker 侧 connector 生命周期由 `KVConnectorModelRunnerMixin` 包住 forward：

```text
maybe_get_kv_connector_output()
  → bind_connector_metadata()
  → start_load_kv(forward_context)
  → model forward / attention layers wait or save KV
  → wait_for_save()
  → get_finished()
  → get_block_ids_with_load_errors()
  → build_connector_worker_meta()
  → clear_connector_metadata()
```

这里的关键是：

```text
connector 的 load/save 发生在 forward context 内部；
forward context 里有 attention metadata；
attention metadata 里有 layer 对应的 block table 和 slot mapping。
```

HMA 下 connector 如果要按 layer / group 保存 KV，就必须能理解：

```text
layer_name 属于哪个 KV cache group；
这个 group 的 block table 是什么；
request 的 block_ids 在所有 group 中分别是什么；
异步 save 完成前哪些 group 的 blocks 不能释放。
```

---

## 26. HMA 与 cross-layer KV cache layout 不是一回事

源码中还有一个容易混淆的优化：

```python
KVConnectorModelRunnerMixin.use_uniform_kv_cache()
KVConnectorModelRunnerMixin.allocate_uniform_kv_caches()
```

它描述的是 **cross-layer / uniform KV cache layout**：

```text
所有 layer 的 KV cache 共享同一个底层 tensor；
对于同一个 block number，各层 KV 数据在物理上尽量连续；
这样 connector 可以一次传输一个 block 对应的所有 layers KV。
```

它启用需要三个条件：

```text
1. KV cache config 只有一个 group，且所有 layers page size 相同；
2. 配置了 KV connector，并且 connector prefer_cross_layer_blocks=True；
3. attention backend 的 KV layout 支持按 block stride 索引，
   也就是 num_blocks 是最外层物理维度。
```

注意第一条：

```text
cross-layer uniform layout 通常要求只有一个 group；
HMA 通常是多个 group。
```

所以二者不是同一个概念：

| 概念 | 解决什么问题 | 典型场景 |
|---|---|---|
| HMA / hybrid KV cache manager | 多种 KV cache spec 如何分组管理 | full + SWA、attention + Mamba |
| packed HMA layout | 多 group / 多 page size 如何更紧凑地打包物理 tensor | DeepSeek V4、显式 packed HMA |
| cross-layer uniform layout | 单 group 下同 block 的所有层 KV 如何连续，便于 connector 传输 | connector prefer cross-layer blocks |

---

## 27. HMA 与 packed layout

`KVCacheTensor` 有这些字段：

```python
size: int
shared_by: list[str]
offset: int = 0
block_stride: int = 0
```

普通 layout 下主要用：

```text
size
shared_by
```

packed layout 下还会用：

```text
offset
block_stride
```

含义：

```text
多个 layer / group 的 KV 可能被放进同一个更大的 packed tensor；
每个 layer 通过 offset 找到自己的区域；
block_stride 表示一个 packed block 的总跨度。
```

`_use_packed_kv_cache_groups()` 会在特定条件下启用 packed HMA KV cache，例如：

```text
DeepSeek V4 特殊布局；
或者设置 VLLM_USE_PACKED_HMA_KV_CACHE。
```

packed layout 是物理 tensor 布局优化；HMA 是逻辑 group 管理机制。packed layout 可以服务 HMA，但不等于 HMA。

---

## 28. HMA 与 attention backend KV layout

attention backend 还会定义自己的 KV tensor shape，例如：

```text
[num_blocks, block_size, num_kv_heads, head_size]
或
[2, num_blocks, block_size, num_kv_heads, head_size]
或 backend 自定义 stride order
```

HMA 不直接规定 backend 的内部 layout。它只保证：

```text
1. 每个 layer 有对应 KV cache tensor；
2. 每个 layer 有对应 attention metadata；
3. metadata 中的 block table / slot mapping 与该 layer 的 group 匹配。
```

如果 connector 对 layout 有要求，可以通过：

```python
KVConnectorBase_V1.get_required_kvcache_layout()
```

声明需要的 KV cache layout，例如 HND / NHD。

因此 layout 至少有三层含义：

```text
逻辑 layout：KV cache groups / block tables / slot mappings；
物理 tensor layout：KVCacheTensor.shared_by / offset / block_stride；
kernel layout：attention backend 的 KV cache shape / stride order。
```

HMA 主要影响第一层，也会间接影响第二层；第三层由 backend 决定。

---

## 29. request_finished_all_groups 为什么必要

请求结束时，Scheduler 会先做：

```text
remove_skipped_blocks(request_id, total_computed_tokens)
block_ids = kv_cache_manager.get_block_ids(request_id)
```

这一步确保：

```text
交给 connector 的 block table 已经去掉窗口外不需要的 blocks。
```

然后：

```text
非 HMA connector：只能处理 block_ids[0]
HMA connector：处理完整 block_ids tuple
```

如果 HMA connector 要异步保存 KV，它可以返回：

```text
True, kv_transfer_params
```

含义是：

```text
connector 接管这些 blocks 的释放；
Scheduler 暂时不能 free；
直到 worker connector 的 get_finished() 返回对应 request_id。
```

多 group 下这尤其重要：

```text
某个 request 的 group 0 save 完了，不代表 group 1 / group 2 save 完了；
connector 必须以 request + all groups 为单位维护完成状态。
```

---

## 30. HMA 与 prefix cache

prefix cache 在 HMA 下也必须带 group 维度。

`BlockPool` 的 prefix cache key 会带 group id：

```text
BlockHashWithGroupId
```

原因：

```text
相同 token prefix 在不同 KV cache group 中对应的 cache block 语义可能不同；
不能只用 token hash 区分。
```

例如：

```text
full attention group 缓存完整前缀；
sliding window group 可能只保留窗口相关 blocks；
Mamba group 的 cache shape 和 attention KV 完全不同。
```

所以 HMA 下 prefix cache 命中返回：

```text
computed_blocks: tuple[list[KVCacheBlock], ...]
num_computed_tokens: int
```

`num_computed_tokens` 是跨 group 对齐后的安全命中长度。

---

## 31. HMA 与 skipped blocks

sliding window / chunked local attention 的 KV cache 优势来自：

```text
旧 token 超出可见窗口后，不必继续保留它们的 KV block。
```

HMA 通过 coordinator 调用各 group manager 的：

```text
remove_skipped_blocks()
```

把不再需要的 blocks 移除或替换为 null block。

这对 attention 的影响是：

```text
attention backend 仍通过 block table 访问 KV；
如果某些历史位置已经不可见，对应 block 不会再出现在有效 block table 中；
slot mapping 只为本轮需要写入 / 读取的可见部分服务。
```

禁用 HMA 后，这类内存节省通常会退化，因为 sliding window 层在 KV manager 看起来可能被统一成 full attention cache。

---

## 32. HMA 与 CUDA graph / padding

Worker 侧 `BlockTables` 会为 CUDA graph 维护稳定形状：

```text
input_block_tables：forward 使用的持久 tensor；
slot_mappings：持久 tensor；
PAD_SLOT_ID：padding token 的 slot id。
```

`compute_slot_mappings()` 会把实际 token 之后的部分填成 `PAD_SLOT_ID`，避免 CUDA graph replay 时读到上一次 chunk 的旧 slot id。

HMA 下 padding 也按 group 进行：

```text
slot_mappings[group, padded_token] = PAD_SLOT_ID
```

这说明：

```text
HMA 不只是 Scheduler 侧概念，
它会影响 Worker 侧 CUDA graph 需要维护的 tensor 形状和 padding 逻辑。
```

---

## 33. HMA 的完整主链路

把所有组件连起来：

```text
模型加载 / 配置阶段
  → 每个 layer 生成 KVCacheSpec
  → get_kv_cache_groups()
      → 单 group 或多个 KV cache groups
  → get_kv_cache_config_from_groups()
      → num_blocks
      → kv_cache_tensors
      → kv_cache_groups

Scheduler 初始化
  → KVCacheManager(kv_cache_config)
  → get_kv_cache_coordinator()
      → UnitaryKVCacheCoordinator 或 HybridKVCacheCoordinator
  → BlockPool

Scheduler 每步调度
  → get_computed_blocks()
  → allocate_slots()
  → KVCacheBlocks
  → block_ids: tuple[list[int], ...]
  → SchedulerOutput

Worker / ModelRunner
  → initialize_kv_cache(kv_cache_config)
  → BlockTables(num_kv_cache_groups, ...)
  → append_block_ids()
  → apply_staged_writes()
  → prepare_attn()
      → gather_block_tables()
      → compute_slot_mappings()
  → build_attn_metadata()
      → CommonAttentionMetadata per group
      → AttentionMetadata per layer
  → set_forward_context()
  → attention backend forward

KV connector
  → scheduler side request_finished_all_groups()
  → worker side start_load_kv() / save_kv_layer() / wait_for_save()
  → get_finished()
```

---

## 34. 与单 group KV cache 的对比

| 问题 | 单 group | HMA / multi group |
|---|---|---|
| KV cache groups | 1 个 | 多个 |
| block_ids | `list[int]` 包在单元素 tuple 中 | `tuple[list[int], ...]` |
| block table | 一张请求表 | 每个 group 一张 |
| slot mapping | 一条 token→slot 映射 | 每个 group 一条 |
| prefix cache hit | 单 manager 判断 | 多 group fixed-point 对齐判断 |
| skipped blocks | 通常简单 | 各 group 按自身窗口/状态移除 |
| connector request_finished | `request_finished(request, block_ids[0])` | `request_finished_all_groups(request, block_ids)` |
| attention metadata | 所有层多半共享同一 block table | 每层拿自己 group 的 metadata |
| 显存效率 | 对 uniform full attention 足够 | 对 hybrid 模型显著更重要 |

---

## 35. 容易混淆的点

### 35.1 HMA 是不是一种 attention？

不是。

HMA 是 KV cache 管理机制。attention 类型仍由 layer / backend 决定。

### 35.2 HMA 会不会把 sliding window attention 变成 full attention？

不会。

启用 HMA 时，sliding window 作为自己的 `SlidingWindowSpec` group 管理。

禁用 HMA 时，KV manager 可能把 sliding window 的 KV spec 规约成 `FullAttentionSpec`，但计算层面的 sliding window attention 仍可以保留。

### 35.3 KV cache group 和 attention group 是一回事吗？

不是完全一回事。

```text
KV cache group：KV manager / block table 的分组；
attention group：worker/backend 里使用同一 backend/builder 的 layer 分组。
```

通常 attention groups 嵌套在 KV cache group 下：

```text
attn_groups[kv_cache_group_id][attention_group_id]
```

### 35.4 block table 和 KV cache tensor 是一回事吗？

不是。

```text
KV cache tensor：真正存 K/V 或 state 的物理内存；
block table：请求持有哪些 block id 的索引表。
```

### 35.5 block table 和 slot mapping 是一回事吗？

不是。

```text
block table：request → block ids；
slot mapping：token → KV slot id。
```

### 35.6 SupportsHMA 是不是表示 connector 更快？

不是。

它首先表示 connector 能正确处理多 KV cache group 的 request finish / block ownership 语义。

### 35.7 cross-layer layout 是不是 HMA？

不是。

cross-layer layout 是 connector 友好的单 group 物理布局优化；HMA 是多 KV cache spec 的逻辑分组与管理机制。

---

## 36. 最终可以记成一张表

| 阶段 | 关键函数 / 类 | HMA 相关产物 | 对 attention 的影响 |
|---|---|---|---|
| 配置检查 | `VllmConfig` post init | `disable_hybrid_kv_cache_manager` | 决定是否保留 hybrid KV 管理 |
| 每层规格 | `KVCacheSpec` 系列 | full / SWA / MLA / Mamba specs | layer 声明自己的 KV 需求 |
| 逻辑分组 | `get_kv_cache_groups()` | `KVCacheGroupSpec` | 决定哪些 layer 共享 block table 语义 |
| 物理布局 | `get_kv_cache_config_from_groups()` | `KVCacheTensor`、`num_blocks` | 决定 KV tensor 怎么分配 |
| 调度管理 | `KVCacheManager` | `KVCacheBlocks` | 为每个 group 分配 blocks |
| hybrid 协调 | `HybridKVCacheCoordinator` | multi-group cache hit / allocation | 对齐不同 attention/cache 类型的安全命中长度 |
| 资源池 | `BlockPool` | `KVCacheBlock` | 管理真实 block 生命周期 |
| Worker 索引 | `BlockTables` | block tables / slot mappings | 把 group block ids 翻译成 token slots |
| metadata | `build_attn_metadata()` | per-layer metadata | attention backend 读取对应 group 的 block table |
| connector | `SupportsHMA` | `request_finished_all_groups()` | 外部 KV transfer 正确保存所有 groups |

---

## 37. 总结

HMA 的完整作用可以压缩成：

```text
每层 KVCacheSpec
  → KV cache groups
  → HybridKVCacheCoordinator
  → KVCacheBlocks(tuple by group)
  → Worker BlockTables
  → per-group slot mappings
  → per-layer attention metadata
  → attention backend 读写对应 KV cache
  → connector 按 all groups 保存 / 释放
```

如果只记住三句话：

```text
1. HMA 不是 attention 类型，而是 hybrid KV cache 的分组和内存管理机制。
2. HMA 的直接外显形态是：block_ids / block table / slot mapping 都带 KV cache group 维度。
3. connector 支持 HMA 的关键不是“能传 KV”，而是能正确处理 request_finished_all_groups() 的多 group block 生命周期。
```

因此，HMA 与 attention 的关系不是“替代 attention backend”，而是：

```text
它把不同 attention/cache 类型产生的 KV cache 需求，
组织成 attention backend 可以稳定消费的 block table、slot mapping 和 metadata。
```
