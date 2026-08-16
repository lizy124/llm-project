# 非-GVA layerwise prefetch 严格复核

> 代码基线：`vllm-ascend` `main`，提交 `d5e9816065ede613327d93908f87fee9f5c47128`（2026-08-15）。
> 分析对象：`kv_pool_社区任务发布_10issues_v2.md` 中“任务 3：[Perf] 非-GVA layerwise prefetch 行为修正”。
> 结论级别：代码路径存在；原任务所称公开支持场景和确定性能问题均未成立，不能按原表述发布。

## 结论摘要

原任务不能作为确定的性能修复任务发布，原因不是“代码里完全没有非-GVA layerwise”，而是它混淆了“内部可达路径”和“公开支持路径”，并把正常的滑动窗口补充策略写成了 bug。

严格结论如下：

1. 当前公开文档明确写明 layerwise 只支持 `backend="memcache"`；Mooncake 和 YuanRong 不支持 `use_layerwise`。
2. worker 中 `use_gva_layerwise = use_layerwise and backend_name == "memcache"`。因此，所有公开支持的 layerwise 配置都走 GVA 路径。
3. 非-GVA key-layerwise 类和分支仍在代码中，手工组合 `use_layerwise=true` 与非 Memcache backend 时可以到达，但该组合超出公开支持合同，不能直接当成面向用户的已支持性能问题。
4. 非-GVA 分支默认 `layerwise_prefetch_layers=1` 是代码事实；GVA 分支默认值是 `min(num_shared_buffers, 8)`，也是代码事实。
5. `_submit_ready_layer_loads()` 在初始层提交 N 个有效任务、之后每推进一层补交 1 个任务，是标准滑动窗口模式。只要初始 N 大于 1，“后续只提交 1”可以维持窗口，不能据此认定窗口失效。
6. 默认 N=1 可能使非-GVA 路径重叠不足，但该路径当前不在公开支持范围内，而且没有 profiler 或端到端数据证明它是实际瓶颈。
7. 当前真正可确认的问题是文档与支持路径实现不一致：文档把 `layerwise_prefetch_layers` 默认值写为 1，而公开支持的 Memcache/GVA 实现默认最高为 8。

建议不要发布原标题。应改为：

```text
[Docs/Correctness] 对齐 layerwise 支持范围与 prefetch 默认值
```

## 代码与文档事实

### 1. 公开支持范围只有 Memcache

`docs/source/user_guide/feature_guide/layerwise_kv_pool.md` 明确说明：

- layerwise 当前要求 `backend: "memcache"`；
- 配置表再次写明 layerwise only supports Memcache；
- 限制章节明确 Mooncake 和 YuanRong 不支持 `use_layerwise`。

对应位置：文档第 40-42、81-84、235-236 行。

这构成当前用户可依赖的支持合同。内部存在类或分支，不等于该配置已被承诺支持。

### 2. 支持路径必然是 GVA

worker 初始化逻辑为：

```python
self.use_gva_layerwise = self.use_layerwise and self.backend_name == "memcache"
```

位置：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:153-155`。

所以公开文档允许的 layerwise 配置都会进入 GVA 路径。非-GVA key-layerwise 是内部可达代码，但当前没有公开支持承诺，也没有对应 backend 的提交窗口和端到端覆盖可用来证明它可生产使用。

### 3. 两条路径的默认值确实不同

非-GVA 分支读取：

```python
self.num_prefetch_layers = int(self._extra_config.get("layerwise_prefetch_layers", 1))
```

位置：`pool_worker.py:427-428`。

GVA 分支的默认值由布局计算：

```python
num_prefetch_layers = min(num_shared_buffers, 8)
```

位置：`layerwise_cache_layout.py:128-130`。

这不能证明非-GVA 有需要立即修复的性能 bug，但证明配置文档没有准确描述当前受支持实现的默认行为。

### 4. `submit_count=1` 是窗口补充，不是窗口重置

核心逻辑为：

```python
submit_count = self.num_prefetch_layers if self.current_layer == 0 else 1
```

位置：`pool_worker.py:1693-1699`。

例如 N=4 时，初始提交 0、1、2、3 层；计算前沿每推进一层再补交下一层。这正是维持固定深度窗口的常见方式。原任务中“后续提交 1 导致窗口无法维持”的因果是错误的。

中间没有实际 load task 的层会被跳过，`submitted_layers` 只统计真正提交的任务。这一细节需要单测保护，但同样不能仅凭代码形态推导窗口失效。

## 原任务逐项判定

| 原表述 | 判定 | 说明 |
|---|---|---|
| 非-GVA 默认 prefetch=1 | 代码事实 | 但属于当前未公开支持路径 |
| GVA 默认 prefetch=8 | 不够准确 | 准确值是 `min(num_shared_buffers, 8)` |
| 后续层提交 1 无法维持窗口 | 错误 | 这是滑动窗口的正常补充方式 |
| 默认非-GVA load 与 attention 串行 | 未证明 | 只有代码风险推断，没有 profiler/端到端证据 |
| 应直接提高非-GVA 默认值 | 无发布依据 | 必须先决定该路径是否受支持 |

## 可发布的替代任务

替代任务应先解决合同问题，而不是预设性能修复：

1. 明确非-GVA key-layerwise 的产品状态：正式支持、实验性保留或拒绝配置。
2. 若不支持，在配置校验、文档和测试中保持一致，避免用户误入未支持路径。
3. 若决定支持，必须先补齐 Mooncake/YuanRong 的功能、异常和端到端覆盖，再单独建立性能基线。
4. 修正文档中的 `layerwise_prefetch_layers` 默认值，使其准确反映 Memcache/GVA 的实际计算规则；或调整代码，使公开默认值确实为 1。二者必须择一并解释原因。
5. 为 N=1/2/4/8、空 task 层和窗口推进补充单测，验证不重复提交、不越界、不死锁。

## 最终判断

任务 3 原描述中的问题不能被认定为当前公开支持功能的真实性能 bug，不能按“修正非-GVA prefetch”直接发布。

准确说法是：

```text
非-GVA key-layerwise 代码路径存在，默认窗口为 1；但当前公开支持的 layerwise
仅有 Memcache/GVA。后续每层补交 1 个任务是正常滑动窗口策略，不构成 bug。
当前可确认并适合发布的问题，是 layerwise 支持范围、配置校验和 prefetch
默认值在文档与实现之间不一致。
```
