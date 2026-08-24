# issue #14148 审核分析：save 侧 `block_hash_to_str` 冗余转换

- 关联 issue：[vllm-project/vllm-ascend#14148](https://github.com/vllm-project/vllm-ascend/issues/14148)
- 审核仓库：本地 `D:\lzy\project\kv_pool\code\vllm-ascend`（`vllm-ascend/` 最新 main，未做改动）
- 审核日期：2026-08-24
- 审核结论：**问题真实存在**

## 1. 问题摘要

issue #14148 描述：save 侧的 `_alloc_gvas_for_save` 中，同一个 `group_block_hashes[block_idx]` 在候选 key 的列表推导、while 循环、for 循环三处被 `block_hash_to_str`（`.hex()`）转换 3 次，产生 3 个内容相同的临时 str；而 load 侧的 key 构造模式相同，但只转换 1 次，无此冗余。

本次审核需要核实：该描述在本地代码中是否存在，并以代码为证据给出严格结论。

## 2. 相关函数与证据路径

| 侧别 | 函数 | 位置 |
| --- | --- | --- |
| save | `_alloc_gvas_for_save` | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1151` |
| load | `_prepare_load_gvas` | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1315` |
| 工具 | `block_hash_to_str` | `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L690` |
| 工具 | `get_block_hashes` / `_LazyGroupedBlockHashList` | `metadata.py#L657` / `metadata.py#L668` |

关键工具实现：

```python
def block_hash_to_str(block_hash: BlockHash | str) -> str:
    return block_hash if isinstance(block_hash, str) else block_hash.hex()
```

说明：当输入 hash 不是 `str` 类型时，每次调用都会执行 `bytes.hex()`，**新分配一个字符串**。这三处调用都以 `group_block_hashes[...]` 为源、经 `block_hash_to_str` 生成 key，因此重复转换会产生内容相同但不同的临时 str。本链路运行时 hash 恒为 bytes（证据链见第 5.3 节），`.hex()` 分支必然执行。

## 3. save 侧：3 处重复转换（证据）

`_alloc_gvas_for_save` 内三个独立循环/推导中分别调用了 `block_hash_to_str(group_block_hashes[...])`：

### 3.1 列表推导（构造 candidate_keys，用于 `_refresh_allocated_gvas`）

`pool_worker.py#L1199-L1209`

```python
candidate_keys = [
    self._make_layerwise_gva_key(
        group_id,
        block_hash_to_str(group_block_hashes[block_idx]),
    )
    for block_idx in range(
        save_start_block,
        min(save_end_block, len(group_block_hashes)),
    )
]
self._refresh_allocated_gvas(candidate_keys)
```

遍历区间：`[save_start_block, min(save_end_block, len(group_block_hashes)))`。每个 block 转换 1 次。

### 3.2 while 循环（逐块检测 `key in _allocated_gvas`）

`pool_worker.py#L1211-L1218`

```python
while save_start_block < save_end_block and save_start_block < len(group_block_hashes):
    key = self._make_layerwise_gva_key(
        group_id, block_hash_to_str(group_block_hashes[save_start_block])
    )
    if key in self._allocated_gvas:
        save_start_block += 1
    else:
        break
```

从同一 `save_start_block` 开始（与 3.1 相同起点），每轮迭代转换 1 次。注意 3.1 的 `_refresh_allocated_gvas` 会先剔除已被 MemCache 驱逐的本地 GVA 条目，while 循环是在刷新后的 `_allocated_gvas` 上判定成员关系：命中缓存块时继续推进，直到第一个未分配块才 break（该块本身已被本轮 while 转换过一次）。

### 3.3 for 循环（构造真实分配 key）

`pool_worker.py#L1223-L1231`

```python
for blk_idx in range(save_start_block, min(save_end_block, len(group_block_hashes))):
    key = self._make_layerwise_gva_key(group_id, block_hash_to_str(group_block_hashes[blk_idx]))
    cached = self._allocated_gvas.get(key)
    ...
```

遍历区间以 3.2 推进后的 `save_start_block` 为起点（若 while 未推进则与 3.1 相同）。每个 block 转换 1 次。

### 3.4 逐块转换次数精确分析

记 `s0` 为进入 while 前的 `save_start_block`，`s1` 为 while 结束后的 `save_start_block`，`e = min(save_end_block, len(group_block_hashes))`。关键事实：**for 循环（3.3）从 `s1` 起步**，因此被 while 跳过的块不会进入 for 循环。

| 块区间 | 3.1 candidate | 3.2 while | 3.3 for | 合计转换次数 |
| --- | --- | --- | --- | --- |
| `[s0, s1)`（命中 `_allocated_gvas` 被 while 跳过） | ✓ | ✓ | ✗ | 2 |
| `s1`（while break 的首个未缓存块，若 `s1 < e`） | ✓ | ✓ | ✓ | **3** |
| `(s1, e)`（for 处理的尾部块） | ✓ | ✗ | ✓ | 2 |

边界情形：while 因区间耗尽退出（`s1 == e`，全部块已缓存）时，所有块恰好转换 2 次，for 循环范围为空。

**结论**：issue 所说「转换 3 次」指三个代码位置各自独立执行一次转换（三处无结果复用）；按块精确计数为**每块 2~3 次**，仅 while 的 break 块达到 3 次。无论哪种口径，单次 save 流程内同一 hash 只需转换 1 次即可，冗余确实存在。此外每处转换后还会再调 `_make_layerwise_gva_key` 做 f-string 拼接（同样分配新 str），重复构造成本被进一步放大。

## 4. load 侧：仅 1 次转换（验证对比）

`_prepare_load_gvas` 中 key 只在单个列表推导构造一次：

`pool_worker.py#L1374-L1377`

```python
keys = [
    self._make_layerwise_gva_key(group_id, block_hash_to_str(group_block_hashes[i]))
    for i in range(load_start_block, full_blocks)
]
```

每个 `group_block_hashes[i]` 只经过一次 `block_hash_to_str`。构造出的 `keys` 列表随后被整体复用：`self.m_store.batch_get_key_info(keys)`（L1393）与 `zip(key_infos, keys, block_indices)`（L1397）均直接消费该列表，无二次转换。partial key 走 `_make_layerwise_partial_key`（由 req_id/block_index 等拼成，不含 block hash），不涉及 `block_hash_to_str`。这与 issue 中「load 侧仅转换 1 次」的描述一致。

## 5. 结论

1. **问题真实存在**。`_alloc_gvas_for_save` 中 [L1202](file:///d:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1202)、[L1213](file:///d:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1213)、[L1224](file:///d:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1224) 三处代码位置各自独立调用 `block_hash_to_str`，同一 hash 按块计被转换 2~3 次；load 侧 `_prepare_load_gvas` 仅调用一次。issue #14148 的「save 侧 3 处转换 vs load 侧 1 次」描述与代码吻合（精确口径见 3.4 节）。

2. **优化空间**：在同一 save 流程内为每个 hash 缓存一次转换结果并复用，可将每块 hex 转换由 2~3 次降到 1 次。多组（`num_kv_cache_groups > 1`，每组独立重复上述流程）或长 prompt（block 数多）时收益显著。

3. **前置条件（重要）**：只有当 `group_block_hashes[...]` 为非 `str` 类型（走 `.hex()` 分支）时，重复调用才产生额外字符串分配；若已是 `str`，`block_hash_to_str` 原样返回同一对象，仅剩函数调用开销。本链路运行时 hash 恒为 bytes，证据链如下：

   - vLLM 侧：`BlockHash = NewType("BlockHash", bytes)`（`vllm/v1/core/kv_cache_utils.py#L44`），且 `BlockHashList = list[BlockHash] | BlockHashListWithBlockSize`（同文件 L2302），两者的元素/getitem 返回值均为 bytes；
   - scheduler 侧：`request_real.block_hashes` 传入 `_build_req_meta` → `ReqMeta.from_request_tracker(block_hashes=...)`（[pool_scheduler.py#L666-L672](file:///d:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L666-L672)），`ReqMeta.block_hashes` 声明为 `list[BlockHash]`（[metadata.py#L848](file:///d:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L848)）；
   - worker 侧：`get_block_hashes` 要么原样返回该序列，要么包装为 `_LazyGroupedBlockHashList`，其 `__getitem__` 仅做 O(1) 索引（[metadata.py#L687](file:///d:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L687)），元素仍为 bytes；
   - 因此 `.hex()` 分支必然执行：每次调用把 32 字节 hash 转为 64 字符新 str。`block_hash_to_str` 签名中的 `str` 分支在本链路不会触发（`get_block_hashes` 的 `BlockHash | str` 标注不改变上述运行时类型）。

   另注：`_LazyGroupedBlockHashList.__getitem__` 为 O(1) 索引，重复取 `group_block_hashes[i]` 本身开销可忽略；冗余成本集中在 `.hex()` 的新 str 分配与 `_make_layerwise_gva_key` 的 f-string 拼接。

## 6. 补充备注

- 本次审核仅做只读分析，未对仓库做任何改动。
- 若推进修复，建议在同一位置统一装配一次 `block_hash_to_str` 结果供 3 段复用，并补充/执行现有 UT（`tests/ut/distributed/ascend_store/test_pool_worker.py`）验证 key 生成结果不变。