# 06. Scheduler 如何调度多模态 encoder input？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\multimodal\encoder_budget.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\encoder_cache_manager.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\scheduler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\core\sched\output.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\request.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`

本问题关注：多模态请求除了 decoder token budget，还可能需要 encoder compute budget 和 encoder cache budget。Scheduler 如何判断本轮哪些 image / video / audio / prompt_embeds 等 encoder input 必须处理，如何把它们写入 `SchedulerOutput.scheduled_encoder_inputs`，以及 encoder cache 命中、cache 容量不足、chunked prefill、外部 encoder cache 这些因素如何影响调度。

---

## 1. 一句话回答

Scheduler 调度多模态请求时，不只看本轮能跑多少 decoder token，还会检查本轮 token window 是否覆盖某个 multimodal placeholder，并同时满足：

```text
1. 这个 encoder input 没有命中本地 encoder cache；
2. 没有被本轮同 hash 的输入重复调度；
3. 有足够 encoder compute budget；
4. encoder cache 有足够空间保存输出；
5. 如果 disable_chunked_mm_input=True，则不能只覆盖 multimodal item 的一部分。
```

满足条件的 input index 会进入：

```python
SchedulerOutput.scheduled_encoder_inputs: dict[str, list[int]]
```

含义是：

```text
req_id -> 本轮需要执行 encoder 的 mm input 下标列表
```

然后 Worker / GPUModelRunner 会：

```text
scheduled_encoder_inputs
  → _execute_mm_encoder()
  → self.encoder_cache[mm_hash] = encoder_output
  → _gather_mm_embeddings()
  → 把 encoder output 拼回本轮 inputs_embeds
  → model forward
```

所以 `scheduled_encoder_inputs` 不是“请求有哪些多模态输入”，而是“本轮哪些多模态输入需要实际跑 encoder”。

---

## 2. 为什么需要 encoder budget

多模态请求里，文本 token 和 encoder input 的成本不是同一类资源。

文本侧调度主要受这些限制：

```text
max_num_batched_tokens
max_num_seqs
KV cache block
max_model_len
prefix cache / chunked prefill / spec decode
```

多模态 encoder 侧还要额外受这些限制：

```text
encoder_compute_budget：本轮最多执行多少 encoder embeddings / tokens
encoder_cache_size：GPU 侧最多缓存多少 encoder output
mm item 粒度：一个 image / video / audio 通常要整体跑 encoder
encoder cache 命中：命中后不需要重复执行 encoder
外部 encoder cache：可能只需要加载，不需要本地计算
```

如果只按 decoder token budget 调度，会出现两个问题：

```text
1. Scheduler 可能把大量图片 / 视频同时塞进一轮，导致 encoder compute 或显存峰值失控；
2. Scheduler 可能推进到某个 multimodal placeholder 区间，但对应 encoder output 还不存在，Worker 在 _gather_mm_embeddings() 时会 cache miss。
```

因此 vLLM V1 把 encoder input 调度放在 scheduler 里，和 token budget、KV block 分配一起做成一个一致的计划。

---

## 3. 最小主链路

整体链路是：

```text
EngineCoreRequest.mm_features
  → Request.mm_features
  → Scheduler.schedule()
      → token_budget = max_num_scheduled_tokens
      → encoder_compute_budget = max_num_encoder_input_tokens
      → _try_schedule_encoder_inputs()
      → scheduled_encoder_inputs[req_id] = [mm_input_id, ...]
      → encoder_cache_manager.allocate(request, mm_input_id)
  → SchedulerOutput.scheduled_encoder_inputs
  → Worker / GPUModelRunner._update_states()
      → 清理 free_encoder_mm_hashes
      → 缓存 scheduled_new_reqs 的 mm_features
  → GPUModelRunner._execute_mm_encoder()
      → model.embed_multimodal(...)
      → encoder_cache[mm_hash] = output
  → GPUModelRunner._gather_mm_embeddings()
      → 从 encoder_cache 按当前 token window 取 embedding
      → 生成 inputs_embeds / is_mm_embed
  → model forward
```

关键边界：

```text
Scheduler 只决定“哪些 encoder input 本轮要处理”；
Worker 才真正执行 encoder，并把输出放入 GPU 侧 encoder_cache。
```

---

## 4. Request 上 encoder input 如何表达

`Request` 定义在：`request.py:59`

多模态输入挂在：

```python
self.mm_features = mm_features or []
```

位置：`request.py:171` 到 `request.py:172`

相关辅助属性：

```python
@property
def num_encoder_inputs(self) -> int:
    return len(self.mm_features)

@property
def has_encoder_inputs(self) -> bool:
    return self.num_encoder_inputs > 0

def get_num_encoder_embeds(self, input_id: int) -> int:
    assert input_id < len(self.mm_features)
    return self.mm_features[input_id].mm_position.get_num_embeds()
```

位置：`request.py:274` 到 `request.py:302`

每个 `mm_feature` 至少会被 scheduler 使用这些信息：

| 信息 | 用途 |
|---|---|
| `mm_feature.identifier` | encoder cache key，也就是 `mm_hash` |
| `mm_feature.modality` | Worker 侧按 modality batch encoder input |
| `mm_feature.data` | Worker 侧实际喂给 encoder 的 multimodal kwargs |
| `mm_feature.mm_position.offset` | placeholder 在 prompt token 序列中的起点 |
| `mm_feature.mm_position.length` | placeholder 覆盖的 token 长度 |
| `mm_feature.mm_position.get_num_embeds()` | encoder output / cache 占用的 embedding 数 |
| `mm_feature.mm_position.get_embeds_indices_in_range()` | 当前 token window 对应 encoder output 的哪段 embedding |

因此 scheduler 并不是直接理解图片或视频内容，而是基于 `mm_features` 中的位置信息、hash 和 embedding 数进行预算与 cache 判断。

---

## 5. MultiModalBudget 如何计算预算

`MultiModalBudget` 定义在：`encoder_budget.py:44`

初始化时会做四件事：

```text
1. 创建 multimodal processor；
2. 根据 processor / dummy mm inputs 计算每种 modality 的最大 token 数；
3. 调用 compute_mm_encoder_budget() 得到 encoder_compute_budget 和 encoder_cache_size；
4. 计算每个 prompt / 每个 batch 最多允许多少个 mm item。
```

核心代码：

```python
encoder_compute_budget, encoder_cache_size = compute_mm_encoder_budget(
    scheduler_config,
    active_mm_max_toks_per_item,
)

self.encoder_compute_budget = encoder_compute_budget
self.encoder_cache_size = encoder_cache_size
```

位置：`encoder_budget.py:115` 到 `encoder_budget.py:123`

### 5.1 每个 item 最大 token 数

入口是：

```python
def get_mm_max_toks_per_item(...)
```

位置：`encoder_budget.py:15`

优先从 processor info 读取：

```python
max_tokens_per_item = processor.info.get_mm_max_tokens_per_item(...)
```

如果 processor 没直接提供，则构造 dummy multimodal inputs，再对 placeholder 的 `get_num_embeds()` 求和：

```python
return {
    modality: sum(item.get_num_embeds() for item in placeholders)
    for modality, placeholders in mm_inputs["mm_placeholders"].items()
}
```

位置：`encoder_budget.py:25` 到 `encoder_budget.py:41`

### 5.2 active / tower / embed-only modality

`MultiModalBudget` 会区分两类 modality：

```text
tower_modalities：需要经过多模态 encoder tower 的输入；
embed_only_modalities：enable_mm_embeds=True 且 limit 为 0，只传入预计算 embedding 的输入。
```

位置：`encoder_budget.py:66` 到 `encoder_budget.py:85`

当前实现还会把没有独立 placeholder token 的共享 modality 从 `active_mm_max_toks_per_item` 中过滤掉，例如部分模型的 audio/video placeholder 共享场景。

位置：`encoder_budget.py:100` 到 `encoder_budget.py:113`

注意：

```text
encoder budget / encoder cache size 使用过滤后的 active modalities 计算；
per-prompt / per-batch 的 mm item 限制只对 tower modalities 计算。
```

对应代码：

```python
# Encoder budget is computed from all active modalities
encoder_compute_budget, encoder_cache_size = compute_mm_encoder_budget(
    scheduler_config,
    active_mm_max_toks_per_item,
)

# Per-prompt/per-batch limits are only relevant for tower modalities
for modality, max_toks_per_item in tower_mm_max_toks_per_item.items():
    ...
```

位置：`encoder_budget.py:115` 到 `encoder_budget.py:134`

### 5.3 compute_mm_encoder_budget()

`compute_mm_encoder_budget()` 定义在：`encoder_cache_manager.py:269`

逻辑很直接：

```python
max_tokens_per_mm_item = max(mm_max_toks_per_item.values())

encoder_compute_budget = max(
    scheduler_config.max_num_encoder_input_tokens,
    max_tokens_per_mm_item,
)
encoder_cache_size = max(
    scheduler_config.encoder_cache_size,
    max_tokens_per_mm_item,
)
```

位置：`encoder_cache_manager.py:296` 到 `encoder_cache_manager.py:316`

也就是说：

```text
encoder_compute_budget 至少要容纳最大的单个 mm item；
encoder_cache_size 也至少要容纳最大的单个 mm item。
```

否则一个合法的单图 / 单视频请求永远无法被调度。

如果禁用 multimodal input chunking，且最大单 item token 数已经超过 `max_num_batched_tokens`，会直接报错：

```python
if (
    scheduler_config.disable_chunked_mm_input
    and max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
):
    raise ValueError(...)
```

位置：`encoder_cache_manager.py:298` 到 `encoder_cache_manager.py:307`

原因是：禁用 chunking 意味着一个 mm item 对应的 placeholder token 不能拆轮处理。如果单 item 比 batch token budget 还大，就无法形成合法调度。

### 5.4 get_encoder_budget()

`MultiModalBudget.get_encoder_budget()` 返回：

```python
return min(self.encoder_compute_budget, self.encoder_cache_size)
```

位置：`encoder_budget.py:188` 到 `encoder_budget.py:189`

这个值用于计算每批最多能放多少个 mm item：

```python
max_encoder_items_per_batch = encoder_budget // max_tokens_per_item
```

位置：`encoder_budget.py:148` 到 `encoder_budget.py:153`

但 Scheduler 每轮真正消耗的 compute budget 用的是 `mm_budget.encoder_compute_budget`，cache 是否够由 `EncoderCacheManager` 单独判断。

---

## 6. Scheduler 初始化 encoder 相关状态

Scheduler 初始化位置：`scheduler.py:204` 到 `scheduler.py:230`

核心代码：

```python
supports_mm_inputs = mm_registry.supports_multimodal_inputs(
    vllm_config.model_config
)
mm_budget = (
    MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
)

self.max_num_encoder_input_tokens = (
    mm_budget.encoder_compute_budget if mm_budget else 0
)
encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
self.encoder_cache_manager = (
    EncoderDecoderCacheManager(cache_size=encoder_cache_size)
    if self.is_encoder_decoder
    else EncoderCacheManager(cache_size=encoder_cache_size)
)
```

含义：

```text
- 非多模态模型：encoder budget = 0，encoder cache size = 0；
- 多模态 decoder-only 模型：使用 EncoderCacheManager；
- encoder-decoder 模型：使用临时的 EncoderDecoderCacheManager。
```

encoder-decoder 分支特殊，是因为当前实现把 text-only encoder-decoder 也暂时包装成 multimodal interface，并且 encoder output 还不是按普通 multimodal cache 复用。

---

## 7. 每轮 schedule() 中 encoder budget 的位置

`schedule()` 定义在：`scheduler.py:433`

每轮开始时：

```python
scheduled_encoder_inputs: dict[str, list[int]] = {}
encoder_compute_budget = self.max_num_encoder_input_tokens
```

位置：`scheduler.py:458` 到 `scheduler.py:460`

也就是说：

```text
encoder_compute_budget 是每个 scheduler step 重置的 compute 预算；
scheduled_encoder_inputs 是本轮输出给 Worker 的 encoder 执行计划。
```

同一轮里，Scheduler 会先调度 RUNNING 队列，再调度 WAITING 队列。两个分支都会在 request 有 encoder input 时调用同一个函数：

```python
self._try_schedule_encoder_inputs(
    request,
    num_computed_tokens,
    num_new_tokens,
    encoder_compute_budget,
    shift_computed_tokens=1 if self.use_eagle else 0,
)
```

RUNNING 分支位置：`scheduler.py:532` 到 `scheduler.py:544`

WAITING 分支位置：`scheduler.py:884` 到 `scheduler.py:897`

返回值是：

```python
encoder_inputs_to_schedule,
num_new_tokens,
new_encoder_compute_budget,
external_load_encoder_input,
```

其中：

| 返回值 | 含义 |
|---|---|
| `encoder_inputs_to_schedule` | 本轮需要本地执行 encoder 的 input index |
| `num_new_tokens` | 可能被 encoder 预算 / cache 限制回退后的 decoder token 数 |
| `new_encoder_compute_budget` | 扣除本轮 encoder input 后剩余的 compute budget |
| `external_load_encoder_input` | 命中外部 encoder cache，需要分配本地 cache 状态但不放入本地执行列表的 input index |

---

## 8. _try_schedule_encoder_inputs() 的核心逻辑

函数位置：`scheduler.py:1367`

源码注释已经概括了一个 encoder input 被调度的条件：

```text
- 它的 output token 范围和本轮计算 token 范围重叠；
- 它还没有在 encoder cache 中算好；
- 它不存在于远端 encoder cache；
- 有足够 encoder token budget；
- encoder cache 有空间保存它。
```

位置：`scheduler.py:1379` 到 `scheduler.py:1394`

### 8.1 先找本轮 token window 覆盖哪些 mm_features

```python
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
)
```

位置：`scheduler.py:1409` 到 `scheduler.py:1413`

窗口含义是：

```text
[num_computed_tokens, num_computed_tokens + num_new_tokens)
```

如果启用 EAGLE，会通过 `shift_computed_tokens=1` 把窗口右移一点，避免投机解码场景下遗漏即将需要的 encoder input。

对于 encoder-decoder 模型，所有 encoder input 都在开头处理，所以强制：

```python
if self.is_encoder_decoder:
    lo = 0
```

位置：`scheduler.py:1414` 到 `scheduler.py:1416`

### 8.2 本地 encoder cache 命中则跳过执行

decoder-only 多模态模型会先去重、再查本地 encoder cache：

```python
if item_identifier in mm_hashes_to_schedule:
    continue

if self.encoder_cache_manager.check_and_update_cache(request, i):
    continue
```

位置：`scheduler.py:1441` 到 `scheduler.py:1452`

含义：

```text
同一轮同一个 mm_hash 只调度一次；
如果 encoder output 已在本地 cache 中，只更新引用关系，不加入 scheduled_encoder_inputs。
```

因此 cache 命中的多模态输入不会触发 `_execute_mm_encoder()`，但后续 `_gather_mm_embeddings()` 仍会从 GPU 侧 `encoder_cache` 读取它。

### 8.3 disable_chunked_mm_input 会阻止半个 item 被调度

如果配置了：

```text
disable_chunked_mm_input=True
```

并且本轮 token window 只覆盖某个 multimodal item 的一部分，Scheduler 会把 `num_new_tokens` 回退到 item 开始前：

```python
if (
    self.scheduler_config.disable_chunked_mm_input
    and num_computed_tokens < start_pos
    and (num_computed_tokens + num_new_tokens) < (start_pos + num_encoder_tokens)
):
    num_new_tokens = max(
        0, start_pos - (num_computed_tokens + shift_computed_tokens)
    )
    break
```

位置：`scheduler.py:1454` 到 `scheduler.py:1469`

直观理解：

```text
如果这轮 token budget 不够覆盖完整图片 / 视频 placeholder，
并且配置禁止 mm input chunking，
那就先只跑到图片 / 视频前面，下一轮再尝试完整处理该 item。
```

### 8.4 encoder compute budget / cache 不足时回退 token 数

真正的预算判断在：

```python
self.encoder_cache_manager.can_allocate(
    request, i, encoder_compute_budget, num_embeds_to_schedule
)
```

位置：`scheduler.py:1470` 到 `scheduler.py:1472`

如果不能分配，Scheduler 不会硬塞这个 request，而是调整 `num_new_tokens`：

```python
if num_computed_tokens + shift_computed_tokens < start_pos:
    num_new_tokens = start_pos - (num_computed_tokens + shift_computed_tokens)
else:
    num_new_tokens = 0
break
```

位置：`scheduler.py:1473` 到 `scheduler.py:1489`

含义分两种：

```text
1. 当前还没走到这个 mm item：
   本轮可以先跑 item 之前的文本 token。

2. 由于 prefix cache 等原因，num_computed_tokens 已经越过了 item 起点，
   但 encoder output 不可用：
   本轮不能调度这个 request，num_new_tokens=0。
```

这就是为什么 `schedule()` 里会把 `num_new_tokens == 0` 解释为：

```text
encoder budget exhausted / encoder cache exhausted
```

RUNNING 分支位置：`scheduler.py:551` 到 `scheduler.py:567`

### 8.5 只在当前范围确实需要 embedding 时才调度

即使 token window 和 mm feature 重叠，也不一定需要 encoder embedding。代码会把 placeholder token range 映射到 embedding range：

```python
curr_embeds_start, curr_embeds_end = (
    mm_feature.mm_position.get_embeds_indices_in_range(
        start_idx_rel, end_idx_rel
    )
)

if curr_embeds_end - curr_embeds_start == 0:
    continue
```

位置：`scheduler.py:1491` 到 `scheduler.py:1505`

这处理的是某些 placeholder token 不对应真实 embedding 的情况。

### 8.6 外部 encoder cache 命中

如果配置了 ECConnector，并且外部 encoder cache 已有这个 item：

```python
if self.ec_connector is not None and self.ec_connector.has_cache_item(
    item_identifier
):
    mm_hashes_to_schedule.add(item_identifier)
    external_load_encoder_input.append(i)
    num_embeds_to_schedule += num_encoder_embeds
    continue
```

位置：`scheduler.py:1507` 到 `scheduler.py:1513`

这类 input 不会加入 `scheduled_encoder_inputs`，因为本轮不需要本地跑 encoder；但稍后还是会调用 `encoder_cache_manager.allocate()`，让 scheduler 的本地 cache 状态和外部加载后的 Worker 状态保持一致。

### 8.7 真正加入 scheduled_encoder_inputs

最终本地需要执行 encoder 的 input 会走到：

```python
num_embeds_to_schedule += num_encoder_embeds
encoder_compute_budget -= num_encoder_embeds
mm_hashes_to_schedule.add(item_identifier)
encoder_inputs_to_schedule.append(i)
```

位置：`scheduler.py:1515` 到 `scheduler.py:1518`

返回给 `schedule()` 后，RUNNING / WAITING 分支都会写入：

```python
scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
for i in encoder_inputs_to_schedule:
    self.encoder_cache_manager.allocate(request, i)
encoder_compute_budget = new_encoder_compute_budget
```

RUNNING 分支位置：`scheduler.py:648` 到 `scheduler.py:661`

WAITING 分支位置：`scheduler.py:1036` 到 `scheduler.py:1050`

---

## 9. EncoderCacheManager 的状态机

`EncoderCacheManager` 定义在：`encoder_cache_manager.py:17`

它管理的是 scheduler 侧的 encoder cache 元数据，不直接保存 GPU tensor。GPU tensor 保存在 Worker 的：

```python
self.encoder_cache: dict[str, torch.Tensor] = {}
```

位置：`gpu_model_runner.py:563` 到 `gpu_model_runner.py:564`

Scheduler 侧主要状态：

| 字段 | 含义 |
|---|---|
| `cache_size` | encoder cache 总容量，单位是 encoder embeddings |
| `num_free_slots` | 当前真实空闲容量 |
| `num_freeable_slots` | 当前可通过驱逐无引用项回收的容量 |
| `cached` | `mm_hash -> 引用它的 request_id 集合` |
| `request_cached_ids` | `request_id -> 该请求已持有 cache 引用的 input_id 集合` |
| `freeable` | 无请求引用、可被驱逐的 `mm_hash -> num_encoder_embeds` |
| `freed` | 本轮因驱逐产生、需要通知 Worker 删除 tensor 的 mm_hash |

### 9.1 check_and_update_cache()

位置：`encoder_cache_manager.py:94`

逻辑：

```text
1. 如果 mm_hash 不在 cached：返回 False；
2. 如果 mm_hash 在 cached 但当前没有请求引用：从 freeable 移除；
3. 把当前 request_id 加到引用集合；
4. 把 input_id 加到 request_cached_ids；
5. 返回 True。
```

它的作用是：

```text
命中本地 encoder cache 时，不重新执行 encoder，只更新引用关系。
```

### 9.2 can_allocate()

位置：`encoder_cache_manager.py:123`

它同时检查两类约束：

```text
1. 单个 input 的 embedding 数不能超过本轮剩余 encoder_compute_budget；
2. encoder cache 要能容纳这个 input，加上本轮已经准备调度但还没 allocate 的 input。
```

核心判断：

```python
num_embeds = request.get_num_encoder_embeds(input_id)

if num_embeds > encoder_compute_budget:
    return False

num_embeds += num_embeds_to_schedule

if num_embeds <= self.num_free_slots:
    return True

if num_embeds > self.num_freeable_slots:
    return False
```

位置：`encoder_cache_manager.py:158` 到 `encoder_cache_manager.py:172`

如果真实空闲不足，但可回收空间足够，会在 scheduler 侧先驱逐无引用项：

```python
while num_embeds > self.num_free_slots:
    mm_hash, num_free_embeds = self.freeable.popitem(last=False)
    del self.cached[mm_hash]
    self.freed.append(mm_hash)
    self.num_free_slots += num_free_embeds
```

位置：`encoder_cache_manager.py:174` 到 `encoder_cache_manager.py:182`

注意：

```text
这里还没有删除 Worker 上的 GPU tensor；
只是把 mm_hash 放进 freed，稍后通过 SchedulerOutput.free_encoder_mm_hashes 通知 Worker 清理。
```

### 9.3 allocate()

位置：`encoder_cache_manager.py:184`

在 scheduler 已经决定调度某个 input 后调用：

```python
self.cached[mm_hash].add(request_id)
self.request_cached_ids.setdefault(request_id, set()).add(input_id)
self.num_free_slots -= num_encoder_embeds
self.num_freeable_slots -= num_encoder_embeds
```

位置：`encoder_cache_manager.py:195` 到 `encoder_cache_manager.py:210`

它只是保留 cache 容量和引用关系，不负责实际 encoder 计算。

### 9.4 free() / free_encoder_input()

当请求完成、取消、abort 或调度失败需要回滚时，会释放请求对 encoder input 的引用。

```python
def free(self, request: Request) -> None:
    for input_id in list(self.get_cached_input_ids(request)):
        self.free_encoder_input(request, input_id)
```

位置：`encoder_cache_manager.py:243` 到 `encoder_cache_manager.py:253`

`free_encoder_input()` 会把 request 从引用集合移除。如果某个 `mm_hash` 没有任何请求引用，就进入 `freeable`，但不会马上物理删除：

```text
无引用 -> freeable；
需要腾空间时 -> can_allocate() 驱逐 -> freed；
SchedulerOutput.free_encoder_mm_hashes -> Worker 删除 GPU tensor。
```

---

## 10. SchedulerOutput 中 encoder 相关字段

`SchedulerOutput` 定义在：`output.py:182`

encoder 相关字段：

```python
# req_id -> encoder input indices that need processing.
scheduled_encoder_inputs: dict[str, list[int]]

# list of mm_hash strings associated with the encoder outputs to be
# freed from the encoder cache.
free_encoder_mm_hashes: list[str]
```

位置：`output.py:203` 到 `output.py:217`

构造位置：`scheduler.py:1142` 到 `scheduler.py:1160`

```python
scheduler_output = SchedulerOutput(
    ...
    scheduled_encoder_inputs=scheduled_encoder_inputs,
    ...
    free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
    ...
)
```

这里要注意两个字段的方向：

```text
scheduled_encoder_inputs：告诉 Worker 本轮要新增哪些 encoder output；
free_encoder_mm_hashes：告诉 Worker 哪些旧 encoder output 可以删除。
```

二者都由 Scheduler 侧的 `EncoderCacheManager` 决定，但真正 tensor 的新增 / 删除都发生在 Worker 侧。

---

## 11. Worker 如何消费 scheduled_encoder_inputs

### 11.1 先清理被驱逐的 encoder output

`GPUModelRunner._update_states()` 会先处理 `free_encoder_mm_hashes`：

```python
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

位置：`gpu_model_runner.py:1199` 到 `gpu_model_runner.py:1201`

也就是说，Scheduler 在 `can_allocate()` 里标记的 freed hash，会在下一次 Worker 更新状态时真正从 GPU 侧 `encoder_cache` 删除。

### 11.2 从 scheduled_encoder_inputs 收集 mm kwargs

`_batch_mm_inputs_from_scheduler()` 定义在：`gpu_model_runner.py:2913`

核心逻辑：

```python
scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
if not scheduled_encoder_inputs:
    return [], [], []

for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
    req_state = self.requests[req_id]
    for mm_input_id in encoder_input_ids:
        mm_feature = req_state.mm_features[mm_input_id]
        if mm_feature.data is None:
            continue

        mm_hashes.append(mm_feature.identifier)
        mm_kwargs.append((mm_feature.modality, mm_feature.data))
        mm_lora_refs.append((req_id, mm_feature.mm_position))
```

位置：`gpu_model_runner.py:2933` 到 `gpu_model_runner.py:2954`

所以 `scheduled_encoder_inputs` 只传 index，不传大对象。Worker 根据自己缓存的 `CachedRequestState.mm_features` 找回真正的 multimodal data。

### 11.3 执行 encoder 并写 GPU encoder_cache

`_execute_mm_encoder()` 定义在：`gpu_model_runner.py:2956`

普通多模态输入会按 modality batch 后调用：

```python
batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
```

位置：`gpu_model_runner.py:3119` 到 `gpu_model_runner.py:3150`

执行完成后写入 Worker 侧 cache：

```python
for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
    self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

位置：`gpu_model_runner.py:3157` 到 `gpu_model_runner.py:3161`

`prompt_embeds` 是特殊 passthrough modality，不跑 encoder，直接把 embedding tensor 放进 encoder cache：

```python
self.encoder_cache[mm_hashes[i]] = pe_tensor.to(self.device)
```

位置：`gpu_model_runner.py:2966` 到 `gpu_model_runner.py:2991`

### 11.4 从 encoder_cache 拼回主模型输入

`_gather_mm_embeddings()` 定义在：`gpu_model_runner.py:3165`

它遍历当前 InputBatch 中每个 request 的本轮 token window：

```python
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_scheduled_tokens,
)
```

位置：`gpu_model_runner.py:3191` 到 `gpu_model_runner.py:3196`

然后按 `mm_hash` 从 `encoder_cache` 取 output；如果缺失且不是 drafter look-ahead 刚好越过已处理边界的情况，会直接报错：

```python
encoder_output = self.encoder_cache.get(mm_hash, None)
if encoder_output is None:
    if start_pos >= req_state.num_computed_tokens + num_scheduled_tokens:
        continue
    raise RuntimeError(f"Encoder cache miss for {mm_hash}.")
```

位置：`gpu_model_runner.py:3217` 到 `gpu_model_runner.py:3228`

这里的错误检查说明 Scheduler 必须保证：

```text
只要本轮 token window 需要某段 multimodal embedding，
对应 encoder output 要么之前已在 Worker encoder_cache，
要么本轮已通过 scheduled_encoder_inputs 执行并写入 encoder_cache，
要么通过 ECConnector 加载进 encoder_cache。
```

最后它返回：

```python
return mm_embeds, is_mm_embed
```

位置：`gpu_model_runner.py:3273`

主模型输入构造处会把这些 embedding 替换进 token embedding：

```python
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

位置：`gpu_model_runner.py:3526` 到 `gpu_model_runner.py:3541`

---

## 12. RUNNING 和 WAITING 两条调度路径的差异

### 12.1 RUNNING 请求

RUNNING 请求已经在 `self.running` 中。Scheduler 会先给它算本轮 `num_new_tokens`，再尝试 schedule encoder input：

```text
request.num_tokens_with_spec + output_placeholders - request.num_computed_tokens
  → long_prefill_token_threshold
  → token_budget
  → max_model_len
  → _try_schedule_encoder_inputs()
```

位置：`scheduler.py:510` 到 `scheduler.py:544`

如果 encoder 预算或 cache 使 `num_new_tokens == 0`，当前请求本轮跳过，但 scheduler 会继续尝试后面的 running 请求：

```python
req_index += 1
continue
```

位置：`scheduler.py:551` 到 `scheduler.py:567`

这意味着 RUNNING 队列在 encoder 资源不足时不严格阻塞整个 step。

如果之后 KV block 分配导致抢占，且被抢占请求本轮已经占用了 encoder budget，Scheduler 会把预算加回来：

```python
preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req_id, None)
if preempted_encoder_inputs:
    num_embeds_to_restore = sum(
        preempted_req.get_num_encoder_embeds(i)
        for i in preempted_encoder_inputs
    )
    encoder_compute_budget += num_embeds_to_restore
```

位置：`scheduler.py:596` 到 `scheduler.py:607`

### 12.2 WAITING 请求

WAITING 请求需要同时考虑 prefix cache、KVConnector、chunked prefill、是否允许 prefill 延迟等。进入 encoder 调度前，Scheduler 已经得到：

```text
num_computed_tokens = local prefix cache 命中 + external KV 命中
num_new_tokens = request.num_tokens - num_computed_tokens
```

位置：`scheduler.py:723` 到 `scheduler.py:827`

然后调用 `_try_schedule_encoder_inputs()`：

```python
if request.has_encoder_inputs:
    (...)= self._try_schedule_encoder_inputs(...)
    if num_new_tokens == 0:
        break
```

位置：`scheduler.py:884` 到 `scheduler.py:900`

WAITING 分支里，如果当前队首请求因为 encoder budget / cache 无法推进到任何 token，通常会 `break`，停止继续接纳新的 waiting 请求。这样可以避免后来的新请求越过队首请求过多，保持等待队列的基本公平性。

如果 encoder 已经 touch 了 cache 状态，但后续 KV block 分配失败，会回滚 encoder cache 引用：

```python
if request.has_encoder_inputs:
    self.encoder_cache_manager.free(request)
break
```

位置：`scheduler.py:956` 到 `scheduler.py:963`

---

## 13. encoder input 与 prefill / decode 的关系

Scheduler 的注释明确说：V1 scheduler 没有独立的 prefill phase / decode phase，而是统一用：

```text
request.num_computed_tokens
request.num_tokens_with_spec
```

来计算本轮还要补多少 token。

位置：`scheduler.py:433` 到 `scheduler.py:444`

多模态 encoder input 也遵循这个统一模型：

```text
如果本轮 token window 覆盖了某个 mm placeholder 对应的 embedding 区间，
那这个 encoder input 必须已经可用。
```

因此：

```text
- prefill 阶段通常会触发 encoder input；
- chunked prefill 可能分多轮逐步走到不同 mm item；
- decode 阶段通常不再触发 encoder input，因为 decode token window 已经越过 prompt multimodal placeholder；
- 但 prefix cache / EAGLE / encoder-decoder 等特殊路径会影响窗口边界。
```

一个典型 chunked prefill 示例：

```text
prompt:
  text_0 ... text_99 [image placeholder: 100 tokens] text_200 ...

step 1:
  token window = [0, 80)
  没覆盖 image，不调度 encoder。

step 2:
  token window = [80, 160)
  覆盖 image placeholder，Scheduler 尝试调度 image encoder。
  如果 budget / cache 足够：scheduled_encoder_inputs[req_id] = [0]
  如果不够且当前还在 image 前：num_new_tokens 回退到 image 起点前。

step 3:
  image encoder output 已在 encoder_cache，后续覆盖 image 剩余区间时不再重复执行 encoder。
```

---

## 14. cache 命中时是否还需要执行 encoder

分三种情况：

| 情况 | 是否进入 scheduled_encoder_inputs | 是否执行 _execute_mm_encoder() | 说明 |
|---|---:|---:|---|
| 本地 encoder cache 命中 | 否 | 否 | `check_and_update_cache()` 更新引用即可 |
| 本轮同 hash 已经调度过 | 否 | 否 | 防止同一 step 内重复执行同一 item |
| 外部 encoder cache 命中 | 否 | 否 | 加入 `external_load_encoder_input`，分配本地 cache 状态，实际加载由 ECConnector 路径处理 |
| 未命中且预算 / cache 足够 | 是 | 是 | 加入 `scheduled_encoder_inputs` |
| 未命中但预算 / cache 不足 | 否 | 否 | 回退 `num_new_tokens` 或本轮不调度该请求 |

重点：

```text
scheduled_encoder_inputs 只表示“本轮本地执行 encoder”。
它不是 encoder cache 中所有可用 input 的列表。
```

---

## 15. encoder-decoder 模型的特殊处理

encoder-decoder 模型使用 `EncoderDecoderCacheManager`：`encoder_cache_manager.py:323`

它和普通 `EncoderCacheManager` 的差异是：

```text
- check_and_update_cache() 永远返回 False；
- get_cached_input_ids() 返回所有 input；
- allocate() 只扣 num_free_slots 并记录 allocated mm_hash；
- get_freed_mm_hashes() 用 to_free / allocated 模拟延迟释放。
```

位置：`encoder_cache_manager.py:323` 到 `encoder_cache_manager.py:381`

Scheduler 侧 `_try_schedule_encoder_inputs()` 对 encoder-decoder 还有特殊判断：

```python
if self.is_encoder_decoder and num_computed_tokens > 0:
    assert start_pos == 0
    continue
```

位置：`scheduler.py:1425` 到 `scheduler.py:1439`

含义：

```text
encoder-decoder 的 encoder input 应该在 decoder 开始前处理；
一旦已经计算过 decoder token，就认为 encoder input 已经处理过，本轮不再重复调度。
```

Worker 侧也单独处理：

```python
if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
    encoder_outputs = self._execute_mm_encoder(scheduler_output)
    model_kwargs.update({"encoder_outputs": encoder_outputs})
```

位置：`gpu_model_runner.py:3605` 到 `gpu_model_runner.py:3612`

与 decoder-only 多模态模型不同，encoder-decoder 不做 prompt replacement，而是把 `encoder_outputs` 直接传给 decoder。

---

## 16. 一轮调度的具体伪代码

可以把 encoder 相关部分简化为：

```python
def schedule():
    token_budget = max_num_scheduled_tokens
    encoder_compute_budget = max_num_encoder_input_tokens
    scheduled_encoder_inputs = {}

    for request in running_then_waiting:
        num_new_tokens = compute_decoder_tokens(request, token_budget)

        if request.has_encoder_inputs:
            (
                encoder_input_ids,
                num_new_tokens,
                encoder_compute_budget_after,
                external_load_input_ids,
            ) = _try_schedule_encoder_inputs(
                request,
                num_computed_tokens=request.num_computed_tokens,
                num_new_tokens=num_new_tokens,
                encoder_compute_budget=encoder_compute_budget,
            )

        if num_new_tokens == 0:
            skip_or_break()

        new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
        if new_blocks is None:
            rollback_encoder_cache_if_needed()
            skip_or_preempt_or_break()

        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens

        if encoder_input_ids:
            scheduled_encoder_inputs[request_id] = encoder_input_ids
            for i in encoder_input_ids:
                encoder_cache_manager.allocate(request, i)
            encoder_compute_budget = encoder_compute_budget_after

        for i in external_load_input_ids:
            encoder_cache_manager.allocate(request, i)

    return SchedulerOutput(
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        free_encoder_mm_hashes=encoder_cache_manager.get_freed_mm_hashes(),
    )
```

这个伪代码体现了两个关键顺序：

```text
1. 先试探 encoder input 是否可调度，再真正分配 KV block；
2. 只有请求最终被调度后，才把 encoder input 写入 scheduled_encoder_inputs 并 allocate encoder cache。
```

---

## 17. 最容易混淆的点

### 17.1 encoder_compute_budget 和 encoder_cache_size 不是同一个东西

```text
encoder_compute_budget：限制一轮能算多少 encoder embedding；
encoder_cache_size：限制可以缓存多少 encoder output。
```

`can_allocate()` 会同时看 compute budget 和 cache 空间，但 compute budget 每轮重置，cache 空间跨请求 / 跨 step 保持状态。

### 17.2 scheduled_encoder_inputs 不包含 cache 命中的输入

命中本地 cache 的 input 已经可用，不需要 Worker 再跑 encoder，所以不会进入 `scheduled_encoder_inputs`。

但 `_gather_mm_embeddings()` 仍会在本轮需要这些 embedding 时从 `self.encoder_cache` 读取。

### 17.3 Scheduler 侧 cache manager 不保存 tensor

Scheduler 侧只维护：

```text
hash、引用关系、容量、可驱逐集合、需要释放的 hash 列表。
```

Worker 侧才保存：

```python
self.encoder_cache[mm_hash] = torch.Tensor
```

### 17.4 cache 驱逐是延迟物理删除

`can_allocate()` 中驱逐 freeable 项时，只是：

```text
从 Scheduler 元数据删除；
把 mm_hash 放进 freed。
```

真正的 GPU tensor 删除发生在 Worker 收到：

```python
scheduler_output.free_encoder_mm_hashes
```

之后。

### 17.5 token budget 不够时可能先跑到 mm item 前

如果当前 step 不能处理某个 encoder input，Scheduler 不一定完全跳过 request。只要当前位置还没到 mm item 起点，它可以把 `num_new_tokens` 回退到 `start_pos` 前，先处理纯文本部分。

### 17.6 prefix cache 可能让问题更严格

如果 prefix cache 让 `num_computed_tokens` 已经越过了某个 multimodal placeholder，但 encoder output 又不在 cache 中，那么 Scheduler 不能只跑后续 token，因为 Worker 会在拼 embedding 时 cache miss。此时 `_try_schedule_encoder_inputs()` 会把 `num_new_tokens` 置 0。

---

## 18. 总结

多模态 encoder input 的调度可以理解成 Scheduler 对 token window 的额外校验：

```text
本轮要计算的 token 区间中，凡是需要 multimodal embedding 的地方，
Scheduler 必须保证对应 encoder output 已经存在，或本轮会被执行 / 加载。
```

实现上分三层：

```text
预算层：
  MultiModalBudget / compute_mm_encoder_budget
  计算每轮 encoder compute 上限和 encoder cache 容量。

调度层：
  Scheduler._try_schedule_encoder_inputs()
  根据 token window、cache 命中、ECConnector、compute budget、cache space 决定 input index。

执行层：
  SchedulerOutput.scheduled_encoder_inputs
    → GPUModelRunner._execute_mm_encoder()
    → GPUModelRunner._gather_mm_embeddings()
  把 encoder output 写入 GPU cache，并拼回模型输入。
```

最终，`scheduled_encoder_inputs` 是连接调度层和执行层的核心字段：

```text
它只记录“本轮需要本地执行 encoder 的 input index”，
不记录已经 cache 命中的 input，
也不直接携带图片 / 视频数据。
```
