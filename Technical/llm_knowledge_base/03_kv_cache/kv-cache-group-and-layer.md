---
title: KV cache group 和 layer 是什么关系？为什么 hybrid 模型里 layerwise 需要区分 group？
category: kv-cache
tags:
  - kv-cache
  - hybrid-kv-cache
  - kv-cache-group
  - layerwise
  - vllm-ascend
related:
  - hybrid-kv-cache.md
  - ../02_vllm_architecture/kv-connector-interface-design.md
source:
  - D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
  - D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py
  - D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py
---

# KV cache group 和 layer 是什么关系？为什么 hybrid 模型里 layerwise 需要区分 group？

## 问题

在解释 `request × group × layer` 时，说 layerwise + hybrid KV cache 需要同时区分请求、KV cache group 和 layer。这里是不是意味着：有的模型一层里可能包含几种不同的 KV cache，因为这些 KV cache 不同，所以需要分组？

## 简要回答

这个理解方向是对的，但需要更精确一点：

> 不是“每一层一定包含几种 KV Cache”，而是 **在 hybrid KV cache 模型里，整个模型的 KV Cache 被分成多个 group；这些 group 可能对应不同层、不同 attention/cache 类型、不同 block size、不同压缩方式。某些层可能属于某个 group，某些复杂模型也可能同一层里存在多种 cache/state 形态。**

所以 `request × group × layer` 的意思是：

```text
同一个 request
    可能有多个 KV cache group
        每个 group 里又包含若干 layer 的 KV/state
```

KV cache group 的本质不是“第几层”，而是：

> 一批可以用同一种 KV cache 规则管理的 cache 单元。

也就是说：

```text
layer 是模型结构维度
group 是 KV cache 管理维度
```

这两个维度可能交叉，所以 layerwise 传输时必须知道当前 request、当前 layer、当前 group。

## 详细解答

### 1. 为什么要分 KV cache group？

因为不同 cache 不能用同一种规则处理。

不同 KV cache group 可能有不同的：

```text
1. block size
2. cache 类型
3. 保存 / 加载粒度
4. key 命名方式
5. 内存地址布局
6. 生命周期
7. 压缩比例
```

普通模型可能很简单：

```text
request
└── group 0: full attention KV
    ├── layer 0 KV
    ├── layer 1 KV
    ├── layer 2 KV
    └── ...
```

这时只有一个 group。

但 hybrid 模型可能是：

```text
request
├── group 0: full attention KV / c1
│   ├── layer 0 KV
│   ├── layer 1 KV
│   └── ...
├── group 1: compressed KV / c4
│   ├── layer 10 KV
│   ├── layer 11 KV
│   └── ...
├── group 2: compressed KV / c128
│   ├── layer 20 KV
│   ├── layer 21 KV
│   └── ...
└── group 3: mamba state / align state
    ├── layer 30 state
    ├── layer 31 state
    └── ...
```

这时就不是一个统一的 KV cache，而是多类缓存并存。

### 2. “一层包含几种 KV Cache”有没有可能？

有可能，但要看模型结构。

#### 情况一：不同层属于不同 group

例如：

```text
layer 0-9      -> group 0: full attention
layer 10-19    -> group 1: sliding window
layer 20-29    -> group 2: compressed attention
```

这种情况下，不是一层里有多个 KV Cache，而是 **不同层使用不同 cache 类型**。

这是 hybrid KV cache manager 很常见的组织方式。

#### 情况二：同一层内部可能有不同 cache/state

某些复杂模型里，同一层可能同时有：

```text
attention KV cache
compressed KV cache
mamba/state cache
auxiliary state
```

这时确实可以说：

> 同一层内部可能有多个不同 cache 形态，因此也需要分组管理。

但具体到 vLLM 的 `kv_cache_groups`，它更像是按照 **cache spec / layer_names / block size / cache 类型** 组织出来的 group，而不一定严格等于“每层有几种 KV”。

### 3. group 的本质是什么？

group 的本质不是“第几层”，而是：

> 一批可以用同一种 KV cache 规则管理的 cache 单元。

比如同一 group 里的层通常共享：

```text
相同 block size
相同 cache spec
相同存取逻辑
相同地址计算方式
相同 cache family
```

所以 group 是管理维度，不是模型层维度。

可以这样理解：

```text
layer 是模型结构维度
group 是 KV cache 管理维度
```

两者是交叉关系：

```text
一个 group 可以包含多层
一层也可能涉及多个 cache/state group
```

### 4. 为什么 layerwise 模式更复杂？

非 layerwise 模式是按 request/chunk 搬 KV：

```text
request A
├── group 0 的一批 block
├── group 1 的一批 block
└── group 2 的一批 block
```

只要对 group 循环处理即可。

layerwise 模式是按层搬 KV：

```text
layer 0 forward 前后
layer 1 forward 前后
layer 2 forward 前后
```

如果再有 hybrid group，就必须知道：

```text
当前 layer 对应哪些 group？
当前 group 在这一层有没有 KV？
这个 group 的 block size 是多少？
这个 group 的 cache family 是 c1/c4/c128 还是 default？
这个 group 的地址在哪里？
```

也就是说，layerwise + hybrid 需要处理二维映射：

```text
layer -> group -> cache blocks
```

或者：

```text
group -> layer_names -> cache blocks
```

当前代码主要只做了：

```text
layer -> group 0
```

所以当前实现不支持 `use_layerwise=True` + hybrid KV cache groups。

### 5. 用一个例子说明

假设模型有 4 层：

```text
layer 0: full attention KV
layer 1: full attention KV
layer 2: sliding window KV
layer 3: mamba state
```

那么 KV cache group 可能是：

```text
group 0: full attention
    layer 0
    layer 1

group 1: sliding window
    layer 2

group 2: mamba state
    layer 3
```

非 layerwise 保存时可以这样做：

```text
保存 request 的 group 0 block
保存 request 的 group 1 block
保存 request 的 group 2 block
```

layerwise 保存时必须这样做：

```text
处理 layer 0 时：
    保存 group 0 的 layer 0 KV

处理 layer 1 时：
    保存 group 0 的 layer 1 KV

处理 layer 2 时：
    保存 group 1 的 layer 2 KV

处理 layer 3 时：
    保存 group 2 的 layer 3 state
```

所以它不只是“每一层都遍历所有 group”，更准确是：

```text
每一层要知道它属于哪个 group，以及这个 group 如何保存/加载。
```

### 6. 和当前代码的关系

当前代码里，Scheduler 和 Worker 都明确禁止 layerwise + 多 KV group：

```python
if self.use_layerwise and len(self.kv_cache_group_ids) > 1:
    raise NotImplementedError("AscendStore layerwise mode does not yet support hybrid KV cache groups.")
```

以及 Worker 侧：

```python
if self.use_layerwise and self.num_kv_cache_groups > 1:
    raise NotImplementedError("AscendStore layerwise mode does not yet support hybrid KV cache groups.")
```

原因是当前 layerwise 路径主要按单 group 写。例如 `store_layer()` 中直接固定：

```python
group_id = 0
```

这意味着当前 layerwise 保存逻辑只处理 group 0，没有完整处理多个 group 的情况。

如果要支持 hybrid + layerwise，就必须把逻辑扩展成：

```text
request × group × layer
```

也就是：

- 每个 request 可能有多个 KV group；
- 每个 group 可能覆盖不同 layer；
- 每个 layer 保存/加载时，要知道自己对应哪个 group；
- 每个 group 使用自己的 block size、cache family、地址映射和 key 生成规则。

### 7. 回答原问题

“有的模型，一层就包含几种不同的 KV cache，由于不同，所以分组？”

可以这样回答：

是的，方向是对的。更准确地说：

> 有些模型的 KV/cache 结构不是统一的一种普通 attention KV，而是存在多种 cache 形态，比如 full attention KV、sliding window KV、compressed KV、Mamba/state cache 等。为了让这些不同形态按各自的 block size、地址布局、压缩比例和生命周期管理，vLLM 会把它们划分成不同 KV cache group。

这些 group 可能表现为：

```text
不同层属于不同 group
```

也可能表现为：

```text
同一层内部存在多个 cache/state，因此涉及多个 group
```

但核心不是“层”，而是：

```text
cache 管理规则不同，所以分组。
```

## 和当前项目的关系

这个问题直接关系到 `vllm-ascend` 中 AscendStore KV Pool 的 hybrid KV cache 和 layerwise 传输实现。

相关代码点：

- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`：Scheduler 侧判断 `use_layerwise` 和 `kv_cache_group_ids`，并在 layerwise + 多 group 时抛出 `NotImplementedError`。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`：Worker 侧也禁止 `use_layerwise` + 多 group，并且当前 `store_layer()` 主要按 `group_id = 0` 处理。
- `D:/lzy/project/kv_pool/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py`：定义了 `KeyMetadata`、`PoolKey`、`LayerPoolKey`、`LayerMultiBlockReqMeta` 等 key / metadata 结构，里面已经有 `kv_cache_group_id` 和 `cache_family` 等字段。

这个问题也解释了为什么当前代码要禁止 `use_layerwise=True` + hybrid KV cache groups：不是理论上不能做，而是当前实现还没有完整实现 group × layer 的组合逻辑。

## 相关问题

- [hybrid KV cache 是什么？它是一种 KV cache 吗？](hybrid-kv-cache.md)
- [DeepSeek V4 里的 c4 / c128 到底是什么意思？](deepseek-v4-c4-c128-kv-cache.md)
- [为什么 vLLM 要求一个连接器接口同时实现 Scheduler 和 Worker 方法？](../02_vllm_architecture/kv-connector-interface-design.md)
