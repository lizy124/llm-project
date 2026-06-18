# 06 Scheduler、EncoderBudget 与 Encoder Cache

本篇梳理 V1 scheduler 如何围绕多模态 encoder 输入做调度。多模态请求进入 `Request.mm_features` 后，并不是立即无条件执行所有 image/audio/video encoder；scheduler 会根据当前 token window、encoder budget、encoder cache 命中情况决定本 step 需要计算哪些 encoder output。

## 1. 总体链路

```text
Request.mm_features
  ↓
MultiModalBudget 计算 encoder_compute_budget / encoder_cache_size
  ↓
Scheduler 初始化 EncoderCacheManager
  ↓
每轮 schedule()
  ↓
_try_schedule_encoder_inputs(...)
  ├─ 当前 token window 是否覆盖某个 placeholder
  ├─ encoder cache 是否命中
  ├─ 本 step compute budget 是否足够
  └─ encoder cache capacity 是否足够
  ↓
SchedulerOutput.scheduled_encoder_inputs
SchedulerOutput.free_encoder_mm_hashes
  ↓
worker 执行 encoder / 释放 GPU encoder cache
```

关键文件：

- `MultiModalBudget`：`code/vllm/vllm/multimodal/encoder_budget.py:44`
- budget 计算：`code/vllm/vllm/v1/core/encoder_cache_manager.py:269`
- scheduler 初始化：`code/vllm/vllm/v1/core/sched/scheduler.py:199`
- scheduler 主入口：`code/vllm/vllm/v1/core/sched/scheduler.py:387`
- encoder cache manager：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`
- encoder admission：`code/vllm/vllm/v1/core/sched/scheduler.py:1279`
- scheduler output：`code/vllm/vllm/v1/core/sched/scheduler.py:1057`

## 2. `MultiModalBudget`

定义：`code/vllm/vllm/multimodal/encoder_budget.py:44`。

它负责根据模型 processor、配置和 scheduler 参数推导多模态 encoder 的预算。

它需要知道：

- 每种 modality 单个 item 最多产生多少 encoder tokens/embeds；
- 每个 prompt 允许多少 item；
- 每个 batch 最多能承载多少 item；
- 哪些 modality 需要 tower encoder；
- 哪些 modality 是 embedding-only；
- `max_num_batched_tokens`；
- `max_model_len`；
- `disable_chunked_mm_input`。

## 3. 单个多模态 item 的最大 token 数

相关函数：`code/vllm/vllm/multimodal/encoder_budget.py:15`。

获取方式有两类：

```text
1. processor 直接提供 get_mm_max_tokens_per_item
2. 否则构造 dummy multimodal inputs，跑标准 processor 流程，从 placeholder/embedding 数推导
```

这说明 dummy inputs 不只是测试用，而是预算估算的一部分。

## 4. tower modality 与 embedding-only modality

`MultiModalBudget` 会区分：

| 类型 | 说明 |
|---|---|
| tower modality | 需要执行 image/audio/video tower encoder。 |
| embedding-only modality | 用户已传入 embedding，不需要 tower，但仍可能占 encoder cache 空间。 |

相关逻辑：`code/vllm/vllm/multimodal/encoder_budget.py:115`。

一个容易忽略的点是：embedding-only 虽然不消耗 tower 计算，但仍会参与 active modalities 和 cache 空间估算。

## 5. `compute_mm_encoder_budget()`

定义：`code/vllm/vllm/v1/core/encoder_cache_manager.py:269`。

核心逻辑：

```text
如果没有 active multimodal modality
  → encoder_compute_budget = 0
  → encoder_cache_size = 0

否则：
  max_tokens_per_mm_item = 所有 modality 中单 item 最大 token 数

如果 disable_chunked_mm_input=False 且 max_tokens_per_mm_item > max_num_batched_tokens
  → 报错

encoder_compute_budget = max(max_num_encoder_input_tokens, max_tokens_per_mm_item)
encoder_cache_size = max(encoder_cache_size, max_tokens_per_mm_item)
```

这里强制 budget 至少能容纳“单个最大多模态 item”，避免出现模型允许某张图/视频输入，但 scheduler 永远无法调度它的情况。

## 6. 最终可用 encoder budget

`MultiModalBudget.get_encoder_budget()` 返回：

```text
min(encoder_compute_budget, encoder_cache_size)
```

位置：`code/vllm/vllm/multimodal/encoder_budget.py:188`。

也就是说，最终受两个约束共同限制：

- 本 step 最多能算多少 encoder tokens；
- encoder cache 最多能存多少 encoder tokens。

## 7. 每 prompt / 每 batch item 上限

相关逻辑：`code/vllm/vllm/multimodal/encoder_budget.py:140`。

核心推导：

```text
max_items_per_prompt
  = max(1, min(mm_limit, max_model_len // max_tokens_per_item))

max_encoder_items_per_batch
  = encoder_budget // max_tokens_per_item

max_decoder_items_per_batch
  受 max_num_seqs / max_num_batched_tokens / chunked prefill 影响

max_items_per_batch
  = max(1, min(max_encoder_items_per_batch, max_decoder_items_per_batch))
```

这解释了为什么多模态 batch 容量不仅由图片数量上限决定，还由 token budget、模型长度、scheduler batch 参数共同决定。

## 8. Scheduler 初始化 encoder cache

Scheduler 初始化时会读取多模态预算：`code/vllm/vllm/v1/core/sched/scheduler.py:199`。

它会设置：

```text
self.max_num_encoder_input_tokens = mm_budget.encoder_compute_budget
self.encoder_cache_manager = EncoderCacheManager(...)
```

`EncoderCacheManager` 是 scheduler 侧 cache 记账器，管理 encoder output 的引用和容量，但不持有真实 GPU tensor。

## 9. `EncoderCacheManager`

定义：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`。

它的职责：

- 判断某个 `identifier` 是否已缓存；
- 记录 request 对 encoder input 的引用；
- 根据 encoder token 数管理容量；
- 在 request 前进、抢占、结束时释放引用；
- 产出可通知 worker 删除的 freed hash/key。

关键方法：

- `check_and_update_cache()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:94`
- `can_allocate()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:123`
- `allocate()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:184`
- `free_encoder_input()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:216`
- `free()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:243`
- `get_freed_mm_hashes()`：`code/vllm/vllm/v1/core/encoder_cache_manager.py:255`

## 10. scheduler 如何判断本 step 要算哪个 encoder input

核心函数：`code/vllm/vllm/v1/core/sched/scheduler.py:1279`。

逻辑可以概括为：

```text
_try_schedule_encoder_inputs(request, num_new_tokens, ...)
  ↓
看当前 token window 覆盖哪些 mm_features 的 placeholder
  ↓
对每个 covered mm_feature:
    如果 encoder cache 命中
      → 复用，不占新的 compute budget
    否则如果 can_allocate
      → 加入 scheduled_encoder_inputs，占用预算和 cache
    否则
      → 收缩 num_new_tokens，让本 step 停在该 placeholder 前
```

这里的关键认识是：多模态 placeholder 是 decoder token 流中的同步点。当 decoder 的 token window 推进到某个 image/audio/video placeholder 时，该 item 的 encoder output 必须已经可用，否则本 step 不能越过它。

## 11. running 请求与 waiting 请求

调度 running 请求时会处理多模态：`code/vllm/vllm/v1/core/sched/scheduler.py:480`。

调度 waiting 请求时也会处理：`code/vllm/vllm/v1/core/sched/scheduler.py:813`。

二者本质类似：都要判断当前计划推进的 token window 是否覆盖多模态 placeholder，只是请求处于不同队列。

## 12. SchedulerOutput

调度结果中包含：

```text
scheduled_encoder_inputs
free_encoder_mm_hashes
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1057`。

含义：

- `scheduled_encoder_inputs`：告诉 worker 本 step 哪些 request 的哪些多模态 item 需要 encoder 输出；
- `free_encoder_mm_hashes`：告诉 worker 哪些 GPU encoder cache 中的输出可以删除。

## 13. 释放 encoder cache

释放发生在多种场景：

- step 推进后，某个 encoder input 已被消费：`code/vllm/vllm/v1/core/sched/scheduler.py:1576`
- 调度后更新状态：`code/vllm/vllm/v1/core/sched/scheduler.py:1866`
- request 被抢占：`code/vllm/vllm/v1/core/sched/scheduler.py:1105`
- request 结束：`code/vllm/vllm/v1/core/sched/scheduler.py:1983`
- finish 路径：`code/vllm/vllm/v1/core/sched/scheduler.py:2053`

scheduler 释放的是引用和容量，worker 根据 freed keys 删除真实 GPU tensor。

## 14. scheduler 侧 cache 与 GPU 侧 cache 的区别

| 层 | 文件 | 保存内容 | 作用 |
|---|---|---|---|
| scheduler | `v1/core/encoder_cache_manager.py` | key、token 数、引用关系 | admission control、容量管理、释放通知。 |
| GPU worker | `v1/worker/gpu/mm/encoder_cache.py` | encoder output tensor | 供当前/后续 step gather embedding。 |

不要把两者混为一谈：scheduler 不持有 tensor，GPU cache 不决定全局调度策略。

## 15. 一句话总结

V1 scheduler 把多模态输入当作 token 序列中的同步点：当当前 step 走到某个 placeholder 时，必须先通过 encoder cache 命中或预算分配确保 encoder output 可用；否则 scheduler 会收缩本 step 的文本推进，等待未来 step 再执行该多模态 encoder。
