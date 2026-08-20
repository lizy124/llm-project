# ascend-store-refactor-1 rebase 到 upstream/main 的冲突分析

## 1. 分析范围与当前状态

本文记录将 ascend-store-refactor-1 rebase 到最新 upstream/main 时的冲突原因和人工分析结果。

本次只做了 Git 只读检查和三方合并模拟，没有启动 rebase，也没有解冲突或修改仓库工作区。

检查仓库：

    D:\lzy\project\kv_pool\code\vllm-ascend

截至本次分析：

| 项目 | 提交/状态 |
|---|---|
| 当前分支 | ascend-store-refactor-1 |
| 当前分支 HEAD | bf4ba195b1c1182f100e57f584414b757fb8d95b |
| 当前分支远端跟踪 | origin/ascend-store-refactor-1 |
| upstream/main | 3475387f4e8b5703892830e0762bd0294574796e |
| 共同基点 | 1a144d6c386a9879d3172b8ce236715655fab60f |
| 工作区 | clean |
| rebase 状态 | 未开始 |

相对共同基点：

- 当前分支有 5 个提交；
- upstream/main 有 41 个新提交；
- 当前分支的 5 个提交都直接建立在旧的 main 基点 1a144d6c3 上。

当前分支的 5 个提交为：

    816ad0022 refactor(kv_pool): extract metadata helpers to module-level functions
    4b923ec54 refactor(kv_pool): remove redundant state and simplify unit tests
    2ef35ee6c fix(kv_pool): preserve lookup and block-size fallback behavior
    8de08dad5 test(kv_pool): drop trivial tests to reduce PR size
    bf4ba195b fix(kv_pool): remove blank lines flagged by ruff-format

冲突定位使用了三方合并模拟：

    git merge-tree --name-only HEAD upstream/main

该命令只生成模拟合并结果，未写入当前工作区。模拟结果显示，内容级冲突集中在 5 个文件；其他当前分支改动 Git 可以自动合并，不能因此认为这些自动合并部分已经完成了语义审查。

## 2. 与我们发生冲突的 upstream PR

### 2.1 主要冲突：PR #13242

对应提交：

    749bb6c82d2aed0d54f7eb898f33ea5a8169ba2e
    [Refactor][Feature] Remove compress attention manager for DeepSeek V4 (#13242)

PR 链接：

    https://github.com/vllm-project/vllm-ascend/pull/13242

PR #13242 修改了本次 5 个冲突文件中的全部 5 个文件：

- tests/ut/distributed/ascend_store/test_metadata.py
- vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py
- vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py
- vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
- vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py

因此，PR #13242 是本次 rebase 的主要冲突来源。

### 2.2 次要重叠：PR #14046

对应提交：

    fc86fff2e585fbec9bb1f9724a030ebafd6e1d2e
    [Feature][KV Transfer] Support Mooncake Layerwise SFA Transfer with KV Cache layerwise Prefill offload (#14046)

PR 链接：

    https://github.com/vllm-project/vllm-ascend/pull/14046

PR #14046 只修改了冲突文件中的 pool_worker.py。本次与它的重叠主要是文件头 import 区：

- upstream 增加了 Callable 和 Any；
- 当前分支的 helper 抽取删除了 math、Any 等旧的 worker 内部 helper 依赖，并改变了 import 上下文。

PR #14046 还增加了 layerwise buffer reuse waiter 相关逻辑；这些新增逻辑大部分不与当前分支发生内容冲突，可以自动合并。它不是本次 block-size 语义冲突的根因。

### 2.3 其他 upstream 提交

逐文件检查 1a144d6c3..upstream/main 的路径历史后，5 个冲突文件的相关 upstream 提交只有：

| 文件 | upstream 相关提交 |
|---|---|
| test_metadata.py | #13242 |
| coordinator.py | #13242 |
| metadata.py | #13242 |
| pool_scheduler.py | #13242 |
| pool_worker.py | #14046、#13242 |

结论：不能把冲突笼统归因于 41 个 upstream 提交；真正需要人工决定的业务语义主要来自 #13242，pool_worker.py 另有 #14046 的 import/上下文重叠。

## 3. 根本语义冲突：压缩 KV 的 block-size 定义变了

### 3.1 当前分支采用的旧语义

当前分支在 816ad0022 中把 cache family/block-size 推导抽成了 metadata.py 的模块级 helper，并在后续提交中继续使用：

    infer_cache_family_ratio(cache_family)
    get_cache_family_granularity(block_size, cache_family)
    get_effective_group_block_size(...)
    infer_cache_transfer_granularity(...)

核心计算仍然是：

    effective_block_size = group_block_size * compress_ratio

这套设计把 group_block_size 看作 cache-domain/base block size，再通过 cache_family（例如 c4、c128）得到逻辑 key/chunk 的有效跨度。它同时要求 AscendStore 的 key、hash、scheduler 查询和 GVA save/load 路径使用这个 effective size。

### 3.2 PR #13242 采用的新语义

PR #13242 的设计说明明确区分两个域：

1. 逻辑 raw-token 域：scheduler、prefix hash、allocation、KV connector、AscendStore/Mooncake key 使用；
2. 物理 compressed-page 域：cache tensor shape、page bytes、kernel block size、DSA/indexer metadata、物理传输行使用。

该 PR 将压缩规格改为：

    C4:   block_size = physical_block_size * 4
    C128: block_size = physical_block_size * 128
          storage_block_size = physical_block_size

因此 upstream 的新假设是：KVCacheSpec.block_size 已经是逻辑 raw-token block size，调用方不应再次乘 compress_ratio。这正好与当前分支 helper 的设计方向相反。

### 3.3 为什么不能无脑选择一侧

如果所有冲突都选择当前分支：

- 可能对已经是逻辑 token size 的新 KVCacheSpec.block_size 再乘一次压缩比；
- C4/C128 的 hash 边界、命中长度、scheduler allocation 和 AscendStore key 可能扩大 4 倍或 128 倍；
- CompressAttentionManager 的旧路径可能与 upstream 新 manager registry 重复生效。

如果所有冲突都选择 upstream：

- 当前分支的 helper 抽取提交会被部分回退；
- 当前分支用于兼容旧调用方的 fallback 行为可能丢失；
- ChunkedTokenDatabase、layerwise key、GVA save/load、coordinator、scheduler 的定义可能不一致。

人工解冲突必须先确定每个调用点使用的是逻辑 token span 还是物理 page/transfer row，再保证所有调用方采用同一套坐标定义。

## 4. 逐文件冲突分析

以下行号分别来自当前分支 HEAD、upstream/main 或三方模拟合并输出。模拟合并中的冲突标记只存在于临时模拟结果，不存在于当前工作区文件。

### 4.1 tests/ut/distributed/ascend_store/test_metadata.py

#### 冲突位置

三方模拟冲突区间：约第 196--255 行。

#### 当前分支一侧

当前分支在该位置保留合并后的通用测试：

    def test_get_block_hashes(self):
        for hashes, expected in [
            (["a", "b", "c", "d"], ["b", "d"]),
            ([b"a", b"b", b"c", b"d"], [b"b", b"d"]),
        ]:
            with self.subTest(hashes=hashes):
                self.assertEqual(list(get_block_hashes(hashes, 32, 16)), expected)

该测试验证字符串 hash 和 bytes hash 在固定 group_block_size=32、hash_block_size=16 时选取终止 hash 的通用行为。

#### upstream 一侧

PR #13242 在相同测试区域增加了：

    def test_compressed_keys_use_logical_spans_and_values_use_physical_rows(self):
        ...

它使用 block_size=[512]、hash_block_size=128、cache_family="c4"，验证：

- key/chunk 的跨度是 (0, 512)、(512, 1024)，即逻辑 raw-token span；
- prepare_value() 返回的地址仍按物理 row 计算；
- block id 和物理 size 保持对应关系。

upstream 同时把通用测试拆成两个更明确的测试：

- test_get_block_hashes_selects_terminal_str_hashes
- test_get_block_hashes_selects_terminal_byte_hashes

#### 人工处理原则

这里不是二选一的业务代码冲突。应保留两边的测试意图：

1. 保留当前分支的通用 hash 行为覆盖，或按 upstream 的两个测试拆分形式移植；
2. 保留 #13242 的压缩逻辑跨度/物理值行测试；
3. 根据最终采用的 block-size 契约，检查 ChunkedTokenDatabase 的测试输入和期望是否仍然成立；
4. 不应为了消除冲突删除压缩语义测试。

### 4.2 coordinator.py

文件：

    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/coordinator.py

#### 冲突位置 A：group_effective_block_sizes

三方模拟冲突区间：约第 86--93 行。

当前分支：

    self.group_effective_block_sizes = [
        get_cache_family_granularity(block_size, family)
        for block_size, family in zip(group_block_sizes, group_cache_families, strict=True)
    ]

upstream #13242：

    self.group_effective_block_sizes = list(group_block_sizes)

两者的实际差异：

- 当前分支认为传入的 group_block_sizes 仍是 base/physical domain，需要根据 cache family 放大；
- upstream 认为传入的 group block size 已经是逻辑 raw-token domain，不应再放大。

#### 冲突位置 B：load_mask()

三方模拟冲突区间：约第 167--176 行。

当前分支对不可 reachability-mask 的 cache family 生成全 true mask，并按 effective block size 计算 chunk 数：

    return tuple(
        [True] * cdiv(token_len, self.group_effective_block_sizes[group_id])
        if not _uses_reachable_mask(self.group_cache_families[group_id])
        else mask
        for group_id, mask in enumerate(masks)
    )

upstream 直接返回 manager 生成的 mask：

    return masks

这不是单纯的格式差异。必须确认：

- find_longest_cache_hit() 返回的 block 数是否已经按照新逻辑 block size 对齐；
- 对 c4/c128 cache family，是否还需要在 coordinator 侧人为扩展 mask；
- ExternalCachedBlockPool 的 hash 存在性集合使用的 group id/hash 是否与 scheduler/worker 一致。

#### 冲突位置 C：压缩 manager 特殊处理

当前分支在 _verify_and_split_kv_cache_groups() 中保留了针对 compress_ratio > 1 的 CompressAttentionManager 兼容处理，并在 _get_manager_class() 中优先选择 Ascend 特殊 manager。

PR #13242 删除了这条特殊路径，改为通过 upstream 的 manager registry/FullAttentionManager 处理压缩规格。

这部分虽然在模拟输出中不一定单独显示为冲突标记，但属于同一文件、同一业务语义的必审区域。不能只解决标记行而保留已经失效的 manager 选择逻辑。

### 4.3 metadata.py

文件：

    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py

#### 冲突位置

三方模拟冲突区间：约第 183--254 行。

#### 当前分支新增/保留的 helper

当前分支包含：

    def infer_cache_family_ratio(cache_family):
        ...

    def get_cache_family_granularity(block_size, cache_family):
        return block_size * infer_cache_family_ratio(cache_family)

    def get_effective_group_block_size(...):
        ...

    def infer_cache_transfer_granularity(...):
        ...

2ef35ee6c 还专门保留了 group id 超出 block-size 列表时回退到第一个 group block size 的行为。这个 fallback 是已有 scheduler/worker 调用约定的一部分，解冲突时不能因为 upstream 版本没有相同 helper 就直接删除。

#### upstream 一侧

PR #13242 删除了：

- infer_cache_family_ratio()；
- get_cache_family_granularity()；
- 依赖它们的旧压缩跨度计算。

同时，upstream 修改了 ChunkedTokenDatabase._iter_token_chunks() 的逻辑：

- 使用 logical_block_size = self.get_block_size(kv_cache_group_id) 生成 grouped hashes；
- start/end key span 使用逻辑 token 坐标；
- prepare value 的地址和 size 仍使用物理 cache row。

#### 人工处理原则

这一文件是整个冲突解决的坐标定义中心。必须先决定最终约定：

1. ChunkedTokenDatabase.block_size 是逻辑 token block size，还是物理 page size；
2. 物理 row 的 stride/size 从哪里读取；
3. cache_family 是否只作为 key namespace 元数据，还是仍参与 token chunk 分组；
4. helper 是否需要保留为兼容层，但不能让调用方重复乘压缩比；
5. group id fallback（越界时使用 group 0）是否继续保留。

在未确认这些问题前，不应简单删除整个 helper 区域，也不应简单保留全部 helper。

### 4.4 pool_scheduler.py

文件：

    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py

#### 冲突位置 A：metadata import

三方模拟冲突区间：约第 45--51 行。

当前分支 import 了模块级 helper：

    get_effective_group_block_size
    get_group_cache_family
    infer_cache_transfer_granularity
    infer_group_block_sizes

upstream 没有这些 import，因为它仍在 scheduler 类内部实现对应方法。

这反映的是当前分支提交 816ad0022 的结构性重构与 upstream 未包含该重构之间的冲突，不是单独一行 import 的问题。

#### 冲突位置 B：类内 helper 方法

三方模拟冲突区间：约第 395--436 行。

当前分支删除了以下私有方法，把逻辑抽到了 metadata.py：

- _infer_group_families()
- _infer_group_block_sizes()
- _get_group_block_size()
- _get_group_family()
- _get_effective_group_block_size()
- _infer_cache_transfer_granularity()

upstream 仍保留这些方法，并且 #13242 修改了其中的 block-size 语义，例如 _get_effective_group_block_size() 直接返回 group block size，不再乘压缩比。

#### 人工处理原则

这里有两个独立决策，不能混为一谈：

1. 代码组织决策：最终保留模块级纯函数，还是保留 scheduler 私有方法；
2. 语义决策：effective block size 返回逻辑 token size 还是再次乘 ratio。

如果保留当前分支的模块级 helper，应同步检查 scheduler 所有调用点，确保：

- original_block_size、grouped_block_size、hash_block_size、lcm_block_size 的单位一致；
- cache_transfer_granularity 不会对新 logical block size 再乘一次 ratio；
- 命中长度 hits_per_group 与 ChunkedTokenDatabase 生成的 key span 一致。

### 4.5 pool_worker.py

文件：

    vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py

#### 冲突位置 A：import

三方模拟冲突区间：约第 6--11 行和第 74--82 行。

upstream #14046 需要：

    from collections.abc import Callable, Generator
    from typing import Any

当前分支经过 helper 抽取后是：

    from collections.abc import Generator

还需要逐项检查 Callable、Any 是否被 #14046 新增的 layerwise waiter 代码使用。不能因为当前分支原来删除了 import 就无脑选当前一侧。

#### 冲突位置 B：worker 内部 block-size 推导方法

三方模拟冲突区间：约第 610--632 行和第 651--683 行。

当前分支删除/抽取了：

- _infer_group_families()
- _infer_group_block_sizes()
- _get_group_block_size()
- _get_effective_group_block_size()
- _get_group_family()
- _infer_cache_transfer_granularity()

upstream 仍保留这些方法，并在 #13242 中将它们改为直接使用 group logical block size。

这与 scheduler 的同名冲突是成对的。只解决 scheduler 而不解决 worker，会导致 scheduler 和 worker 对同一 KV group 使用不同单位。

#### 冲突位置 C：GVA save 路径

三方模拟冲突区间：约第 1236--1242 行。

当前分支：

    cache_family = get_group_cache_family(self.kv_cache_group_families, group_id)
    ratio = max(infer_cache_family_ratio(cache_family), 1)
    effective_block_size = group_block_size * ratio

upstream #13242：

    effective_block_size = group_block_size

该值随后决定：

- grouped hash 的分组边界；
- save 起止 block；
- pool hit token 到 block 的换算；
- GVA key 的候选范围。

#### 冲突位置 D：GVA load 路径

三方模拟冲突区间：约第 1406--1412 行。

当前分支同样按 cache_family 乘压缩比；upstream 直接使用 group_block_size。该值随后影响：

- load 起始 block；
- cached full blocks；
- partial block index；
- load GVA/key 范围。

save 和 load 必须采用完全相同的 block-size 坐标，否则会出现写入 key 与读取 key 不一致，或 partial block 边界错误。

#### 冲突位置 E：非 GVA coordinator lookup

当前分支还在 _lookup_with_coordinator() 中通过 get_cache_family_granularity() 计算 effective block size。upstream #13242 删除了这类重复压缩比换算，改用 token database 的逻辑 group block size。

这处即使没有单独的冲突标记，也必须纳入人工审查，否则可能出现“文件已解决、但 lookup 仍按旧单位”的隐性 bug。

#### PR #14046 的独立逻辑

PR #14046 增加了 layerwise buffer reuse waiter：

- worker 增加 waiter 类型和 setter；
- save/load 线程接收 external slot release waiter；
- layerwise load 完成后通知物理 slot 可复用；
- 与 Mooncake/SFA layerwise transfer 的完成顺序相关。

这部分关注的是 layerwise 物理 buffer 复用的时序安全，不等同于 #13242 的 logical/physical block-size 迁移。解冲突时应保留 #14046 的 waiter 语义，同时独立处理 block-size 语义，不能用一个 PR 的代码整体覆盖另一个 PR。

## 5. 冲突文件与冲突 hunk 汇总

以下是三方模拟合并报告出的冲突区间汇总：

| 文件 | 模拟冲突区间 | 主要来源 | 类型 |
|---|---:|---|---|
| test_metadata.py | 196--255 | #13242 | 测试插入区域重叠，同时需保留两类测试意图 |
| coordinator.py | 86--93、167--176 | #13242 | effective block size、load mask、manager 语义 |
| metadata.py | 183--254 | #13242 + 当前分支 helper 抽取 | helper 删除/新增与坐标定义冲突 |
| pool_scheduler.py | 45--51、395--436 | #13242 + 816ad0022 抽取 | import 和类内 helper 结构/语义冲突 |
| pool_worker.py | 6--11、74--82、610--632、651--683、1236--1242、1406--1412 | #14046、#13242 + 816ad0022 抽取 | import、helper、GVA save/load block-size 冲突 |

注意：这些区间是模拟合并中的冲突区间，不代表当前工作区文件已经写入冲突标记。当前工作区仍 clean。

## 6. 建议的人工逐行解冲突顺序

后续真正开始 rebase 时，建议按以下顺序逐个提交、逐个文件检查。

### 第一步：先固定最终 block-size 契约

先明确采用 #13242 后的契约：

- scheduler/hash/key/cache-hit 使用 logical raw-token block size；
- tensor/page/physical transfer 使用 physical storage block size；
- 不允许同一条路径重复乘 compress_ratio。

如果业务上必须兼容旧调用方，应增加明确命名的转换函数，并在边界处转换，而不是在多个层次隐式乘 ratio。

### 第二步：从 metadata.py 开始统一单位

先人工确认：

- ChunkedTokenDatabase._iter_token_chunks() 的 start/end 是逻辑 token；
- prepare_value() 的 address/size 是物理 row；
- get_block_hashes() 的 group block size 与 key span 使用同一逻辑单位；
- group id fallback 是否保留；
- cache family 是否只保留在 key namespace。

### 第三步：同步 coordinator、scheduler、worker

按调用链检查：

    KVCacheSpec / KVCacheGroup
            ↓
    metadata block-size helpers
            ↓
    pool_scheduler 命中与 key 生成
            ↓
    coordinator mask / cache hit
            ↓
    pool_worker GVA save/load 与 lookup
            ↓
    ChunkedTokenDatabase / physical transfer

同一 group 在这条链路中的 token boundary 必须一致。

### 第四步：单独合入 #14046 的 waiter 逻辑

确认 Callable/Any import 和 waiter setter、线程参数、完成通知都保留；然后再次检查它们是否改变 layerwise load/save 的 block range。时序修复和 block-size 语义修复要分开验证。

### 第五步：最后处理测试

测试至少应覆盖：

1. 普通 string hash；
2. 普通 bytes hash；
3. C4/C128 logical key span；
4. physical row address/size；
5. coordinator load/lookup/store mask；
6. GVA save/load 的同一 key 边界；
7. group id fallback；
8. layerwise buffer reuse waiter 的完成顺序。

## 7. 当前结论

1. 本次主要冲突 PR 是 #13242。它改变了 DeepSeek V4 压缩 KV 的 logical/physical block-size 契约，并触及全部 5 个冲突文件。
2. pool_worker.py 还有 #14046 的 import/上下文重叠；#14046 的主要业务是 layerwise buffer reuse waiter，不是本次 block-size 冲突根因。
3. 当前分支的 816ad0022 helper 抽取与 upstream 未包含该抽取提交，是 scheduler、worker、metadata 出现“删除 vs 保留/修改”冲突的结构性原因。
4. 不能使用 checkout ours、checkout theirs 或“全部接受一侧”的方式解冲突。
5. 真正解冲突前必须先统一 logical token span 与 physical storage row 的单位边界，然后逐个调用点人工审查。
6. 当前只完成了分析文档整理；代码尚未 rebase、尚未解冲突、尚未改变工作区。

