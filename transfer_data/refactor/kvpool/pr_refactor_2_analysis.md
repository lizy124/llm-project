# PR #14465 第二部分（ascend-store-refactor-2）改动分析

## 1. 文档目的

本文档分析 `ascend-store-refactor-2` 分支相对于 `ascend-store-refactor-1` 的改动逻辑，说明每一处改动为什么属于"需要模型和特性组合验证"的范畴，而不是可以随第一部分提前合入的低风险清理。

记录日期：2026-08-19。

## 2. 分支关系

```
main (1a144d6c3)
  └─ ascend-store-refactor-1 (bf4ba195b)  PR #14465，第一部分
       └─ ascend-store-refactor-2 (f0ce0a0de)  第二部分
  └─ ascend-store-refactor (0d29b9c62)  修复后的基线，与 refactor-1 + refactor-2 叠加等价
```

第二部分相对于第一部分只有一个核心 commit：

```
25329b86c refactor(kv_pool): simplify backend and model-specific paths
4 files changed, 18 insertions(+), 42 deletions(-)
```

后续的 `5fb8aee19` 和 `f0ce0a0de` 是与第一部分同步的 trivial 测试删除，不属于第二部分的独立逻辑。

## 3. 改动清单

| 文件 | 改动类型 | 行数 |
|------|---------|------|
| pool_scheduler.py | backend 初始化简化 + MLA 判定简化 + 方法调用对齐 | +6/-14 |
| pool_worker.py | backend 初始化简化 + MLA 判定简化 + 死代码删除 | +6/-18 |
| memcache_backend.py | 删除死代码 init_store 方法 | +0/-8 |
| test_pool_scheduler.py | 测试 mock 方法名对齐 | +1/-1 |

## 4. 逐项分析

### 4.1 backend 初始化简化（pool_scheduler.py:163-174）

**改动前**：
```python
backend_name = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")
self.backend_name = backend_name.lower()
self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"
backend = backend_map.get(self.backend_name)
if backend is None:
    raise ValueError(f"Unsupported KV pool backend: {backend_name}")
backend_path = backend.get("path")
backend_class_name = backend.get("name")
assert backend_path is not None and backend_class_name is not None
backend_module = importlib.import_module(backend_path)
backend_class = getattr(backend_module, backend_class_name)
```

**改动后**：
```python
backend_name = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake").lower()
self.use_gva_layerwise = self.use_layerwise and backend_name == "memcache"
backend = backend_map.get(backend_name)
if backend is None:
    raise ValueError(f"Unsupported KV pool backend: {backend_name}")
backend_module = importlib.import_module(backend["path"])
backend_class = getattr(backend_module, backend["name"])
```

**为什么需要验证**：
- `self.backend_name` 字段被删除。如果有外部代码读取 `scheduler.backend_name`，会触发 AttributeError。
- `backend_name` 现在是局部变量，不再存储在实例上。需要确认没有其他方法依赖该字段。
- 错误消息中引用的变量从 `backend_name`（未 lower 的原始值）变为 lower 后的值，需要确认错误行为一致。
- 涉及 backend 选择和 layerwise 激活路径，必须在不同 backend（mooncake/memcake）和 layerwise 开关组合下验证。

### 4.2 MLA 判定简化（pool_scheduler.py:180-182, pool_worker.py:120-122）

**改动前**：
```python
self.use_mla = False
if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
    self.use_mla = True
```

**改动后**：
```python
self.use_mla = getattr(model_config, "use_mla", False) is True
```

**为什么需要验证**：
- `use_mla` 直接影响 `num_kv_head` 的计算（MLA 模型设为1，否则读取实际头数）。
- `use_mla` 还影响 `get_group_tp_size` 的返回值（MLA 时返回1）。
- 必须在 MLA 模型（如 DSV4）和非 MLA 模型（如 Qwen3）上验证，确认 `getattr + is True` 与原 `hasattr + isinstance + truthy` 三重检查行为完全一致。
- 边界场景：`use_mla` 为非 bool 值（如字符串 "true"）时，`is True` 返回 False，而原代码的 `isinstance` 检查也会返回 False，行为一致。但需要在真实模型上确认。

### 4.3 backend 初始化简化（pool_worker.py:310-323）

**改动前**：
```python
def _init_backend(self, parallel_config, extra_config) -> None:
    backend = backend_map.get(self.backend.lower())
    assert backend is not None
    backend_path = backend.get("path")
    backend_name = backend.get("name")
    assert backend_path is not None and backend_name is not None
    backend_module = importlib.import_module(backend_path)
    real_backend = getattr(backend_module, backend_name)

    backend_kwargs = {}
    backend_kwargs["lazy_init"] = self.use_compress
    self.m_store = real_backend(
        parallel_config,
        **backend_kwargs,
    )
```

**改动后**：
```python
def _init_backend(self, parallel_config) -> None:
    backend = backend_map.get(self.backend_name)
    if backend is None:
        raise ValueError(f"Unsupported KV pool backend: {self.backend_name}")
    backend_module = importlib.import_module(backend["path"])
    real_backend = getattr(backend_module, backend["name"])

    self.m_store = real_backend(
        parallel_config,
        lazy_init=self.use_compress,
    )
```

**为什么需要验证**：
- `self.backend` 字段被删除，`_init_backend` 不再接收 `extra_config` 参数。需要确认所有调用点已更新。
- `assert` 改为 `raise ValueError`，错误处理语义变化（assert 在 -O 模式下会被跳过，ValueError 不会）。
- `backend_kwargs` dict 被消除，直接传 `lazy_init=self.use_compress`。需要确认 backend 构造函数的参数签名一致。
- 涉及 backend 实例化，必须在 mooncake 和 memcache 两个 backend 上验证 m_store 能正确创建。

### 4.4 死代码删除

#### 4.4.1 pool_worker.py:117 `self.dp_rank`

**改动**：删除 `self.dp_rank = parallel_config.data_parallel_rank`

**验证依据**：
- `self.dp_rank` 在 pool_worker.py 中设置后没有被读取（grep 确认）。
- pool_scheduler.py:1039 中的 `dp_rank` 是独立函数中的局部变量，不依赖 worker 的字段。
- 风险低，但仍需确认没有外部模块通过 `worker.dp_rank` 访问。

#### 4.4.2 pool_worker.py:154-155 `self.backend` / `self.backend_name`

**改动**：
```python
# 改动前
self.backend = extra_config.get("backend", "mooncake")
self.backend_name = self.backend.lower()

# 改动后
self.backend_name = extra_config.get("backend", "mooncake").lower()
```

**验证依据**：
- `self.backend`（原始值）在 `_init_backend` 中被 `self.backend.lower()` 重新计算，等同于 `self.backend_name`，属于冗余字段。
- 删除后 `_init_backend` 改为使用 `self.backend_name`，逻辑等价。

#### 4.4.3 memcache_backend.py:107-113 `init_store` 方法

**改动**：删除整个 `init_store` 方法

**验证依据**：
- grep 确认 `init_store` 在整个代码库中只在定义处出现，测试文件中仅有一处注释引用。
- MemcacheBackend 的初始化路径已改为在 `register_kv_caches` 中直接完成，不再需要显式调用 `init_store`。
- 必须在 memcache backend + layerwise 场景下验证 store 初始化正常。

### 4.5 方法调用对齐（pool_scheduler.py:285, test_pool_scheduler.py:588）

**改动前**：
```python
exists_states = self.store_scheduler.batch_is_exist(query_keys)
# 测试中
scheduler.store_scheduler.batch_is_exist.return_value = exists
```

**改动后**：
```python
exists_states = self.store_scheduler.exists(query_keys)
# 测试中
scheduler.store_scheduler.exists.return_value = exists
```

**为什么需要验证**：
- `base.py:32` 中 `batch_is_exist` 实现为 `return self.exists(keys)`，两者等价。
- 但 scheduler 的 `store_scheduler` 是 `create_scheduler_client` 返回的实例，需要确认该路径下 `exists` 方法可用。
- pool_worker 中已有3处调用 `self.m_store.exists()`，本次改动让 scheduler 对齐 worker 的调用方式。
- 必须在 lookup 路径上验证，确认 `exists` 方法在 scheduler client 上行为正确。

## 5. 为什么第二部分不能提前合入

### 5.1 改动位于执行关键路径

| 改动 | 位于路径 | 影响 |
|------|---------|------|
| backend 初始化 | worker/scheduler `__init__` | m_store 创建失败会导致整个推理无法启动 |
| MLA 判定 | `__init__` 中的 `use_mla` | 影响 num_kv_head 计算，错误会导致 KV cache 布局异常 |
| exists 调用 | `_get_store_lookup_hit_tokens` | 影响 prefix cache 命中判断，错误会导致性能退化或错误加载 |
| layerwise 激活 | `use_gva_layerwise` | 影响 memcache + layerwise 路径，错误会导致 offload 失效 |

### 5.2 需要的验证矩阵

| 场景 | 模型 | backend | layerwise | 验证点 |
|------|------|---------|-----------|--------|
| MLA + 非layerwise | DSV4 | mooncake | False | use_mla 判定、num_kv_head |
| MLA + layerwise | DSV4 | memcache | True | use_gva_layerwise 激活、store 初始化 |
| 非MLA + 非layerwise | Qwen3-32B | mooncake | False | use_mla=False、exists 调用 |
| 非MLA + layerwise | Qwen3-32B | memcache | True | layerwise 路径、init_store 删除后无影响 |
| 非 MLA + 非 layerwise | Qwen3-32B | memcache | False | backend 切换、lazy_init 传递 |
| compress 模型 | DSV4 | mooncake | False | lazy_init=True 传递 |

### 5.3 与第一部分的风险对比

| 维度 | 第一部分（refactor-1） | 第二部分（refactor-2） |
|------|----------------------|----------------------|
| 改动位置 | helper 提取、死代码清理 | backend 初始化、MLA 判定、执行入口 |
| 行为风险 | 无行为变化（已证明等价） | 改变执行路径上的调用方式 |
| 验证方式 | UT + 基础 smoke | 需要完整模型矩阵验证 |
| 合入依赖 | 可独立合入 | 必须等待模型验证完成 |

## 6. 等价性保证

第二部分完成后，`ascend-store-refactor-2` 与修复后的基线 `ascend-store-refactor` 完全等价：

```bash
$ git diff ascend-store-refactor..ascend-store-refactor-2 --stat
# 无输出，确认文件内容完全一致
```

这意味着第二部分的所有改动都是原 PR #13160 修复后基线的子集，没有引入基线之外的任何改动。

## 7. 结论

第二部分的改动虽然行数少（+18/-42），但每一处都位于执行关键路径上：

1. **backend 初始化**：改变 m_store 的创建方式和错误处理语义
2. **MLA 判定**：影响 KV cache 布局的核心参数
3. **方法调用对齐**：改变 scheduler 的 lookup 调用路径
4. **死代码删除**：移除 init_store 方法，改变 memcache 初始化流程

这些改动必须通过完整的模型矩阵验证（MLA/非MLA × mooncake/memcache × layerwise 开关）才能合入，不能像第一部分那样仅凭 UT 和基础 smoke 就提前合入。
