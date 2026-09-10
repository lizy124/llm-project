# ascend-store-refactor-2 交接文档

## 1. 任务目标

将 `ascend-store-refactor-2` 分支**完全对齐到最新 `upstream/main`**，然后**手动叠加第二部分的增量改动**，得到一个干净的、直接基于新基座的分支。

核心原因：PR #14465（第一部分）的核心内容已经合入 `upstream/main`（commit `503b1e090`），`ascend-store-refactor-2` 当前的 6 个 commit 都是基于旧 main（`1a144d6c3`）的，需要在新基座上重新叠加第二部分的纯增量改动。

## 2. 仓库与分支

- **仓库**：`D:\lzy\project\kv_pool\code\vllm-ascend`
- **远程**：origin = lizy124/vllm-ascend，upstream = vllm-project/vllm-ascend

### 当前分支状态

| 分支 | HEAD | 说明 |
|------|------|------|
| `upstream/main` | `d85e6714` | 最新上游 main |
| `ascend-store-refactor` | `0d29b9c6` | 修复后的原 PR 基线，保留作等价性基准 |
| `ascend-store-refactor-1` | `c642c8d52` | 第一部分（PR #14465 对应），8 个 commit，基于合入后的 upstream/main（merge-base `e3e8ba1e8`） |
| `ascend-store-refactor-2` | `f0ce0a0d` | 第二部分，6 个 commit，基于旧 main（merge-base `1a144d6c3`） |

### 分支关系

```
upstream/main (d85e6714)
    |
    +-- PR #14465 已合入 (503b1e090) ← metadata helper / 死代码清理 / UT 重构
    |
    +-- ascend-store-refactor-1 (8 commits on top of new main, merge-base e3e8ba1e8)
    |       包含: metadata helper 抽取、死代码删除、UT 重构、fallback 修复、ruff 清理、trivial test 删除
    |
    +-- ascend-store-refactor-2 (6 commits on top of OLD main, merge-base 1a144d6c3, 需重基座)
            两分支 commit hash 完全不同（无共享 commit），仅 commit message 有部分相似
            第二部分独有 commit: 25329b86c / 5fb8aee19 / f0ce0a0de
```

**重要**：
- `ascend-store-refactor` 和 `ascend-store-refactor-2` 的最终 tree hash 完全一致（`a2e71346`），但 commit 历史不同。两者基于旧 main，需要重置。
- `git diff ascend-store-refactor-1..ascend-store-refactor-2 --stat` 会显示大量非 Python 文件变更（`.github/workflows/*`、`CODEOWNERS` 等），这是因为两个分支的 merge-base 不同（refactor-1 基于合入 #14465 后的新 main，refactor-2 基于旧 main），这些差异属于 upstream 本身的变化，**不属于第二部分改动范围**。实际只需关注 commit `25329b86c` 的 4 个 Python 文件改动。

## 3. 已合入 upstream/main 的关键变更

这些改动已经在新 `upstream/main` 中，**不需要再在新 refactor-2 中重复**：

### PR #13242（主要影响 block-size 语义）
- 提交：`749bb6c8` `[Refactor][Feature] Remove compress attention manager for DeepSeek V4 (#13242)`
- 影响文件：`test_metadata.py`、`coordinator.py`、`metadata.py`、`pool_scheduler.py`、`pool_worker.py`
- 核心变更：
  - **block-size 契约变更**：scheduler/hash/key/cache-hit 使用 **logical raw-token block size**，tensor/page/physical transfer 使用 **physical storage block size**
  - C4: `block_size = physical_block_size * 4`
  - C128: `block_size = physical_block_size * 128`, `storage_block_size = physical_block_size`
  - 不允许同一条路径重复乘 compress_ratio
  - 删除了 `infer_cache_family_ratio()`、`get_cache_family_granularity()` 旧压缩跨度计算
  - `ChunkedTokenDatabase._iter_token_chunks()` 改用 logical block size
  - 删除了针对 CompressAttentionManager 的特殊处理路径

### PR #14465（第一部分核心内容）
- 提交：`503b1e090` `[Refactor] (kv_pool): simplify metadata helpers and unit tests (#14465)`
- 这是第一部分的核心，已合入

### PR #14046（layerwise buffer reuse waiter）
- 影响文件：`pool_worker.py`
- 新增了 `Callable`、`Any` import，layerwise waiter 相关逻辑
- **在新 refactor-2 中必须保留，不能被第二部分改动覆盖**

### 其他 upstream 提交
- CI 优化（PR #14298 等）
- DeepSeek V4 测试临时禁用（PR #14688）

## 4. 第二部分独有改动（需重新应用）

`ascend-store-refactor-2` 上有 3 个关键 commit 构成第二部分的独有改动（按 commit message 筛选，因为两分支无共享 commit）：

```
25329b86c refactor(kv_pool): simplify backend and model-specific paths  ← 核心业务改动
5fb8aee19 test(kv_pool): drop trivial tests to reduce PR size            ← 清理性
f0ce0a0de fix(kv_pool): remove blank lines flagged by ruff-format       ← 清理性
```

其中 `5fb8aee19` 和 `f0ce0a0de` 是清理性的，`25329b86c` 是核心业务改动。注意：由于 refactor-2 基于旧 main，这 6 个 commit 中前 3 个（`816ad0022`、`4b923ec54`、`2ef35ee6c`）与 refactor-1 的前 3 个 commit message 相同但 hash 不同——这些是第一部分在旧基座上的版本，在新的 upstream/main 上已经通过 #14465 合入，**不需要再应用**。

### 4.1 核心业务改动（commit 25329b86c）

4 个文件，+18 / -42：

#### 文件 1：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py`
```python
# 删除 init_store 方法（8 行）
# 原代码:
def init_store(self, init_bm: bool = True):
    if self.store is not None:
        return
    self._init_bm = init_bm
    self.store = self._setup_store()
    self._store_initialized = True
    self._register_buffers_if_needed()
```

#### 文件 2：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
```python
# 变化 1: backend 查找简化
# 原: backend_path = backend.get("path"); backend_class_name = backend.get("name"); assert ...; importlib.import_module(backend_path); getattr(...)
# 新: importlib.import_module(backend["path"]); getattr(backend_module, backend["name"])

# 变化 2: backend_name 合并
# 原: backend_name = config.get("backend", "mooncake"); self.backend_name = backend_name.lower()
# 新: backend_name = config.get("backend", "mooncake").lower()

# 变化 3: MLA 判断简化
# 原: self.use_mla = False; if hasattr(...) and isinstance(...) and ...: self.use_mla = True
# 新: self.use_mla = getattr(model_config, "use_mla", False) is True

# 变化 4: exists 查询统一
# 原: exists_states = self.store_scheduler.batch_is_exist(query_keys)
# 新: exists_states = self.store_scheduler.exists(query_keys)

# 变化 5: backend 找不到时用 ValueError 替代 assert
```

#### 文件 3：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
```python
# 变化 1: _init_backend 签名简化（去掉 extra_config 参数）
# 原: self._init_backend(parallel_config, extra_config)
# 新: self._init_backend(parallel_config)

# 变化 2: _init_backend 内部简化
# - 删除 backend_kwargs dict，改用直接参数 lazy_init=self.use_compress
# - backend 查找从 backend.lower() 改为 self.backend_name
# - assert 改为 ValueError

# 变化 3: _init_parallelism_info 简化
# - 删除 local_rank = envs.LOCAL_RANK（已被 upstream 其他逻辑替代？）
# - MLA 判断简化同 scheduler

# 变化 4: backend_name 提取简化
# 原: self.backend = extra_config.get("backend", "mooncake"); self.backend_name = self.backend.lower()
# 新: self.backend_name = extra_config.get("backend", "mooncake").lower()
```

#### 文件 4：`tests/ut/distributed/ascend_store/test_pool_scheduler.py`
```python
# 变化: batch_is_exist → exists（统一查询接口）
# 原: scheduler.store_scheduler.batch_is_exist.return_value = exists
# 新: scheduler.store_scheduler.exists.return_value = exists
```

### 4.2 清理性改动

#### commit 5fb8aee19 — 删除 trivial test
- 删除 `test_pool_worker.py`、`test_pool_scheduler.py` 等中的 getter/lookup 简单测试
- 用于控制 PR 体积

#### commit f0ce0a0de — ruff 格式清理
- 删除空行，按 ruff-format 规则调整

## 5. 操作步骤

### 步骤 1：备份当前分支
```bash
cd D:\lzy\project\kv_pool\code\vllm-ascend
git checkout ascend-store-refactor-2
git branch ascend-store-refactor-2-pre-rebase-backup
git checkout main
```

### 步骤 2：重置到 upstream/main
```bash
git checkout ascend-store-refactor-2
git fetch upstream main
git reset --hard upstream/main
```

### 步骤 3：逐文件手动应用第二部分改动

按以下顺序操作，每完成一个文件做一次 `git diff` 确认：

#### 3a. `memcache_backend.py` — 删除 `init_store` 方法
- 在 `__init__` 和 `set_device` 之间删除 `init_store` 方法
- **检查**：upstream/main 上该文件的结构可能不同，需要找到正确位置

#### 3b. `pool_scheduler.py` — 4 处改动
1. backend 查找简化（去 assert、用 dict key 直接访问）
2. backend_name 合并
3. MLA 判断简化（`getattr(..., False) is True`）
4. `batch_is_exist` → `exists`
5. ValueError 替代 assert

#### 3c. `pool_worker.py` — 4 处改动
1. `_init_backend` 签名（去掉 `extra_config` 参数）
2. `_init_backend` 内部简化
3. `_init_parallelism_info` 简化
4. `backend_name` 提取简化

#### 3d. `test_pool_scheduler.py` — `batch_is_exist` → `exists`

### 步骤 4：提交改动
```bash
git add <4 个文件>
git commit -s -m "refactor(kv_pool): simplify backend and model-specific paths"
```

### 步骤 5：验证
- `git diff upstream/main` 确认只有预期的 4 个文件变更
- Python 语法检查：`python -m compileall vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`
- 确认 `pool_worker.py` 中 #14046 的 waiter 逻辑（Callable/Any import、waiter setter 等）未被删除
- 确认 `pool_scheduler.py` 和 `pool_worker.py` 的 block-size 语义仍保持 #13242 后的契约（logical token size）

## 6. 关键风险与注意事项

### 6.1 block-size 语义不可破坏
PR #13242 改变了压缩 KV 的 block-size 契约。第二部分改动中没有直接修改 block-size 计算，但涉及 `pool_scheduler.py` 和 `pool_worker.py`，必须确认：
- scheduler 和 worker 对同一 KV group 使用相同的 block-size 单位
- `exists()` 查询使用 logical token span
- GVA save/load 的 key 边界一致
- 不应出现重复乘 compress_ratio 的情况

### 6.2 #14046 waiter 逻辑不可丢失
`pool_worker.py` 在 upstream/main 中已包含 #14046 的 layerwise buffer reuse waiter 逻辑（import Callable/Any、waiter setter、线程参数等）。手动应用改动时**必须保留**这些逻辑，不能因为简化 backend 初始化而误删。

### 6.3 `init_store` 的调用点
删除 `MemcacheBackend.init_store` 前，检查 upstream/main 中是否还有调用该方法的地方。如果有，需要一并调整调用方（可能在 `pool_worker.py` 的 `_init_backend` 或其他位置）。

### 6.4 `batch_is_exist` → `exists` 的接口变更
确认 upstream/main 中 `store_scheduler` 的实际接口名。如果 #14465 合入时已经做了这个重命名，则这一步不需要再做。

### 6.5 `local_rank` 的来源
第二部分改动中删除了 `pool_worker.py` 的 `self.local_rank = envs.LOCAL_RANK`。在 upstream/main 中检查是否由其他逻辑（如基类、`_init_parallelism_info` 的其他部分）负责设置。

## 7. 参考文档

| 文档 | 路径 | 说明 |
|------|------|------|
| PR #13160 拆分记录 | `transfer_data/refactor/kvpool/pr_split.md` | 拆分背景和分支关系 |
| Rebase 冲突分析 | `transfer_data/refactor/kvpool/rebase_conflict_analysis_ascend_store_refactor_1.md` | PR #13242/#14046 冲突分析，block-size 契约 |

## 8. 预期结果

完成后，新的 `ascend-store-refactor-2` 应该：
- 基于最新 `upstream/main`（`d85e6714`）
- 只包含 4 个文件的增量改动（+18/-42）
- 与 `ascend-store-refactor-1` 保持 clean diff 关系
- 通过 `python -m compileall` 语法检查
- 保留 #13242 的 block-size 契约和 #14046 的 waiter 逻辑
