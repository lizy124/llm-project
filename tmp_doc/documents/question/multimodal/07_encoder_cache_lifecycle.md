# 07. EncoderCacheManager 如何缓存和释放 encoder output？

源码位置：

- `code/vllm/vllm/v1/core/encoder_cache_manager.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/multimodal/encoder_budget.py`
- `code/vllm/vllm/v1/engine/core.py`

本问题关注：vLLM V1 多模态 / encoder-decoder 场景下，encoder output 如何被调度、缓存、复用、释放和失效。这里的 encoder cache 不是 decoder KV cache；它缓存的是图像、音频、视频、`prompt_embeds` 等 encoder 侧产物，并由 Scheduler 的逻辑账本和 Worker 的物理 tensor cache 共同完成生命周期闭环。

---

## 1. 一句话回答

`EncoderCacheManager` 是 Scheduler 侧的逻辑账本，负责判断某个 multimodal input 是否已经有 encoder output、是否还有容量、哪些 hash 可以淘汰；`GPUModelRunner.encoder_cache` 是 Worker 侧的物理缓存，真正保存 `mm_hash -> encoder_output tensor`。

主链路是：

```text
请求带着 mm_features 进入 Scheduler
  → Scheduler._try_schedule_encoder_inputs() 找出本轮需要 encoder 的 input_id
  → EncoderCacheManager 检查命中 / 容量 / 引用计数 / 可淘汰项
  → SchedulerOutput.scheduled_encoder_inputs 发送给 Worker
  → GPUModelRunner._execute_mm_encoder() 执行 multimodal encoder
  → encoder output 写入 GPUModelRunner.encoder_cache[mm_hash]
  → _gather_mm_embeddings() 从 cache 取 output 并拼回模型输入
  → Scheduler.update_from_output() 在请求推进后释放 request 引用
  → 后续分配空间不足时 EncoderCacheManager 淘汰 freeable hash
  → SchedulerOutput.free_encoder_mm_hashes 通知 Worker 删除物理 tensor
```

所以：

```text
Scheduler 管“该不该算、能不能放、何时可淘汰”；
Worker 管“真正算出 tensor、保存 tensor、消费 tensor、删除 tensor”。
```

---

## 2. 两个 cache：逻辑账本和物理存储

### 2.1 Scheduler 侧：`EncoderCacheManager`

入口：`code/vllm/vllm/v1/core/encoder_cache_manager.py:17`

它管理的是 cache 状态，不保存 tensor。

核心字段：

```text
cache_size:
  encoder cache 总容量，单位是 encoder embeddings 数量。

num_free_slots:
  当前还没有被占用的容量。

num_freeable_slots:
  当前可以通过淘汰无引用条目回收的容量。

cached:
  mm_hash -> set(request_id)
  表示某个 encoder output 当前被哪些请求引用。

request_cached_ids:
  request_id -> set(input_id)
  表示某个请求已经引用了哪些 encoder input cache。

freeable:
  OrderedDict[mm_hash, num_encoder_embeds]
  表示没有请求引用、但物理 tensor 还留在 worker cache 里的条目。

freed:
  list[mm_hash]
  表示本轮逻辑淘汰掉、需要通知 worker 删除物理 tensor 的 hash。
```

对应源码：`encoder_cache_manager.py:67` 到 `encoder_cache_manager.py:79`

这几个字段说明一个关键点：

```text
“释放 request 引用”不等于“删除 GPU tensor”；
只有当未来容量不足并触发淘汰时，才会把 mm_hash 放进 freed，随后通过 SchedulerOutput.free_encoder_mm_hashes 通知 Worker 删除 tensor。
```

### 2.2 Worker 侧：`GPUModelRunner.encoder_cache`

入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:533`

Worker 侧真正保存 encoder output：

```python
self.encoder_cache: dict[str, torch.Tensor] = {}
```

它的 key 是 multimodal item 的 `identifier`，也就是文档里常说的 `mm_hash`。

Worker 侧会在三个地方操作它：

```text
1. _execute_mm_encoder()：写入 encoder_cache[mm_hash] = output；
2. _gather_mm_embeddings()：读取 encoder_cache[mm_hash] 并拼回输入 embeddings；
3. _update_states()：根据 free_encoder_mm_hashes 删除 encoder_cache[mm_hash]。
```

对应源码：

- 写入：`gpu_model_runner.py:3092` 到 `gpu_model_runner.py:3096`
- 读取：`gpu_model_runner.py:3149` 到 `gpu_model_runner.py:3151`
- 删除：`gpu_model_runner.py:1158` 到 `gpu_model_runner.py:1160`

---

## 3. encoder cache 和 KV cache 的区别

可以先用这个表抓住区别：

```text
维度                 Encoder cache                         Decoder KV cache
管理对象             multimodal / encoder output            decoder self-attention KV blocks
逻辑 key             mm_feature.identifier / mm_hash         token block hash / request block
物理内容             encoder output tensor                   KV block tensor
Scheduler 职责       引用计数、容量、命中、淘汰 hash          KV block 分配、prefix cache、抢占、释放
Worker 职责          保存 / 读取 / 删除 encoder output        使用 block table / slot mapping 访问 KV
释放粒度             multimodal item                         KV block
复用场景             相同图片/音频/视频/embedding             相同 token prefix
消费方式             拼回 inputs_embeds 或传 encoder_outputs   attention backend 读 KV cache
```

最容易误解的是：

```text
EncoderCacheManager 的容量单位是 encoder embeddings 数量，
不是原始图片大小，也不是 placeholder token 中所有文本/分隔 token 的数量。
```

源码注释明确说明：它按 multimodal embeddings 粒度管理，不把 multimodal embeddings 之间的 break/text tokens 算入 cache size。

位置：`encoder_cache_manager.py:41` 到 `encoder_cache_manager.py:45`

---

## 4. cache 容量和预算从哪里来

Scheduler 初始化时会先判断模型是否支持 multimodal inputs。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:199` 到 `scheduler.py:225`

核心流程：

```text
supports_multimodal_inputs(model_config)
  → MultiModalBudget(vllm_config, mm_registry)
  → mm_budget.encoder_compute_budget
  → mm_budget.encoder_cache_size
  → EncoderCacheManager(cache_size=encoder_cache_size)
```

### 4.1 `MultiModalBudget`

入口：`code/vllm/vllm/multimodal/encoder_budget.py:44`

它会：

```text
1. 创建 multimodal processor；
2. 读取模型支持的 modality limits；
3. 区分 tower modalities 和 embedding-only modalities；
4. 估算每种 modality 单个 item 最多需要多少 placeholder / embeds；
5. 调用 compute_mm_encoder_budget() 得到 compute budget 和 cache size；
6. 推导每个 prompt / batch 最多能放多少 multimodal item。
```

关键点：

```text
embedding-only modality 不经过 encoder tower，
但仍然需要 encoder cache 空间，因为后续会走统一的 gather / splice 路径。
```

对应源码：`encoder_budget.py:72` 到 `encoder_budget.py:123`

### 4.2 `compute_mm_encoder_budget()`

入口：`code/vllm/vllm/v1/core/encoder_cache_manager.py:269`

它返回两个值：

```text
encoder_compute_budget:
  一轮最多允许计算多少 encoder input tokens / embeddings。

encoder_cache_size:
  encoder cache 总容量。
```

逻辑是：

```text
max_tokens_per_mm_item = max(mm_max_toks_per_item.values())
encoder_compute_budget = max(max_num_encoder_input_tokens, max_tokens_per_mm_item)
encoder_cache_size = max(encoder_cache_size, max_tokens_per_mm_item)
```

位置：`encoder_cache_manager.py:296` 到 `encoder_cache_manager.py:316`

这保证至少一个最大的 multimodal item 能被调度和缓存。

如果 `disable_chunked_mm_input=True`，并且单个 multimodal item 比 `max_num_batched_tokens` 还大，会直接报错，避免后续永远无法调度完整 item。

位置：`encoder_cache_manager.py:298` 到 `encoder_cache_manager.py:307`

---

## 5. Scheduler 如何决定本轮要不要跑 encoder

核心入口是：

```text
Scheduler.schedule()
  → _try_schedule_encoder_inputs(...)
```

`Scheduler.schedule()` 每轮会初始化：

```python
scheduled_encoder_inputs: dict[str, list[int]] = {}
encoder_compute_budget = self.max_num_encoder_input_tokens
```

位置：`scheduler.py:412` 到 `scheduler.py:414`

`scheduled_encoder_inputs` 的含义是：

```text
req_id -> 本轮需要 worker 处理的 encoder input_id 列表
```

它最终会进入 `SchedulerOutput.scheduled_encoder_inputs`。

字段定义：`code/vllm/vllm/v1/core/sched/output.py:201` 到 `output.py:204`

---

## 6. `_try_schedule_encoder_inputs()` 的核心逻辑

入口附近：`code/vllm/vllm/v1/core/sched/scheduler.py:1320`

它做的不是直接执行 encoder，而是回答四个问题：

```text
1. 当前 scheduled token window 会覆盖哪些 mm_features？
2. 这些 mm_features 是否已经在 encoder cache 中？
3. 如果没有命中，本轮 encoder compute budget 和 cache capacity 是否够？
4. 如果启用了 EC connector，是否可以从外部 encoder cache 加载，而不是本地计算？
```

返回值是：

```text
encoder_inputs_to_schedule:
  本轮需要本地执行 encoder 的 input_id。

num_new_tokens:
  可能被 encoder 约束回退后的本轮 token 数。

encoder_compute_budget:
  扣减后的 encoder 计算预算。

external_load_encoder_input:
  本轮不本地计算、但需要为外部加载的 encoder cache 分配逻辑引用的 input_id。
```

位置：`scheduler.py:1432` 到 `scheduler.py:1437`

### 6.1 先找当前 token window 覆盖的 multimodal items

```python
lo, hi = get_mm_features_in_window(...)
```

位置：`scheduler.py:1321` 到 `scheduler.py:1325`

也就是说，只有当前调度 token 范围会用到的 multimodal placeholder，才需要考虑 encoder output。

对 encoder-decoder 模型，encoder input 统一放在起点，所以 `lo = 0`。

位置：`scheduler.py:1326` 到 `scheduler.py:1328`

### 6.2 encoder-decoder 的跳过规则

如果是 encoder-decoder，并且 `num_computed_tokens > 0`，说明 encoder input 已经在最开始处理过，后续 decoder token 不再重复调度 encoder。

位置：`scheduler.py:1337` 到 `scheduler.py:1351`

这和普通 VLM 不同：

```text
普通 multimodal 模型：encoder output 会替换 prompt 中的 multimodal placeholder；
encoder-decoder 模型：encoder output 作为 encoder_outputs 传给 decoder。
```

### 6.3 同一步内去重

普通 multimodal 模型会先检查：

```python
if item_identifier in mm_hashes_to_schedule:
    continue
```

位置：`scheduler.py:1353` 到 `scheduler.py:1359`

这避免同一个 `mm_hash` 在同一 scheduler step 中被重复加入 `scheduled_encoder_inputs`。

### 6.4 cache 命中：只更新引用，不再调度 encoder

```python
if self.encoder_cache_manager.check_and_update_cache(request, i):
    continue
```

位置：`scheduler.py:1361` 到 `scheduler.py:1364`

`check_and_update_cache()` 的行为是：

```text
- 如果 mm_hash 不在 cached：返回 False；
- 如果 mm_hash 在 cached 但没有 request 引用：从 freeable 移除，减少 num_freeable_slots；
- 把当前 request_id 加入 cached[mm_hash]；
- 把 input_id 加入 request_cached_ids[request_id]；
- 返回 True。
```

对应源码：`encoder_cache_manager.py:94` 到 `encoder_cache_manager.py:121`

所以 cache 命中后：

```text
SchedulerOutput.scheduled_encoder_inputs 不会包含这个 input_id，
Worker 不会重新执行 encoder，
但 _gather_mm_embeddings() 后续仍会从 worker 的 encoder_cache 中读取这个 mm_hash。
```

### 6.5 禁止 chunked multimodal input 时，不能只调度半个 item

如果 `disable_chunked_mm_input=True`，并且当前 token budget 只够覆盖 multimodal item 的一部分，Scheduler 会把 `num_new_tokens` 回退到 item 开始前。

位置：`scheduler.py:1366` 到 `scheduler.py:1381`

这避免出现：

```text
decoder token 已经进入 multimodal placeholder，
但 encoder output 还没完整算出来。
```

### 6.6 容量检查：`can_allocate()`

如果 cache 未命中，就检查能否为这个 encoder output 预留 cache 容量：

```python
self.encoder_cache_manager.can_allocate(
    request, i, encoder_compute_budget, num_embeds_to_schedule
)
```

位置：`scheduler.py:1382` 到 `scheduler.py:1384`

`can_allocate()` 会检查两件事：

```text
1. 单个 item 的 embeds 数不能超过本轮剩余 encoder_compute_budget；
2. cache 的 free slots 或 freeable slots 必须足够。
```

位置：`encoder_cache_manager.py:123` 到 `encoder_cache_manager.py:182`

如果 free slots 不够，但 freeable slots 够，它会淘汰最旧的 freeable 条目：

```text
freeable.popitem(last=False)
  → del cached[mm_hash]
  → freed.append(mm_hash)
  → num_free_slots += num_free_embeds
```

位置：`encoder_cache_manager.py:174` 到 `encoder_cache_manager.py:182`

注意这里的淘汰仍是 Scheduler 侧逻辑淘汰：

```text
真正的 GPU tensor 删除，要等这个 mm_hash 被放进 SchedulerOutput.free_encoder_mm_hashes 后，由 Worker 执行 pop。
```

### 6.7 EC connector 外部命中

如果启用了 EC connector，并且外部 encoder cache 已有该 item：

```python
if self.ec_connector is not None and self.ec_connector.has_cache_item(item_identifier):
    external_load_encoder_input.append(i)
    ...
    continue
```

位置：`scheduler.py:1419` 到 `scheduler.py:1425`

这表示：

```text
本地 worker 不需要重新执行 encoder，
但 Scheduler 仍然要为这个 input_id 分配 encoder cache 逻辑引用，
因为 worker 后续会通过 EC connector 把远端 encoder output 加载进 encoder_cache。
```

---

## 7. Scheduler 何时真正 allocate encoder cache

`_try_schedule_encoder_inputs()` 只做可行性判断和返回 input_id；真正调用 `allocate()` 是在 KV slots 成功分配、请求确定被调度之后。

running 请求路径：`scheduler.py:599` 到 `scheduler.py:612`

waiting 请求路径：`scheduler.py:963` 到 `scheduler.py:977`

典型逻辑：

```text
if encoder_inputs_to_schedule:
    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
    for i in encoder_inputs_to_schedule:
        encoder_cache_manager.allocate(request, i)
        ec_connector.update_state_after_alloc(request, i)
```

`allocate()` 会：

```text
- 创建 cached[mm_hash] 的 request set；
- 把 request_id 加进去；
- 记录 request_cached_ids[request_id].add(input_id)；
- 扣减 num_free_slots；
- 扣减 num_freeable_slots。
```

对应源码：`encoder_cache_manager.py:184` 到 `encoder_cache_manager.py:210`

为什么不在 `_try_schedule_encoder_inputs()` 里立即 allocate？

```text
因为请求后面还可能因为 KV block 分配失败而无法调度。
只有 KV slots 也成功分配后，Scheduler 才能确认本轮真的会执行这个 encoder input。
```

如果 waiting 请求在 KV slot 分配失败后已经“touch”过 encoder cache，Scheduler 会调用：

```python
self.encoder_cache_manager.free(request)
```

位置：`scheduler.py:887` 到 `scheduler.py:894`

preempt running 请求时也会释放 encoder cache 引用：

```python
self.encoder_cache_manager.free(request)
```

位置：`scheduler.py:1114` 到 `scheduler.py:1116`

---

## 8. `SchedulerOutput` 如何表达 encoder cache 状态

`SchedulerOutput` 有两个和 encoder cache 生命周期直接相关的字段：

```python
scheduled_encoder_inputs: dict[str, list[int]]
free_encoder_mm_hashes: list[str]
```

字段定义：`code/vllm/vllm/v1/core/sched/output.py:201` 到 `output.py:215`

含义分别是：

```text
scheduled_encoder_inputs:
  告诉 Worker：这些 request 的这些 encoder input_id 本轮需要处理。

free_encoder_mm_hashes:
  告诉 Worker：这些 mm_hash 对应的 encoder output tensor 可以从 worker cache 删除。
```

Scheduler 构造 output 时：

```python
free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes()
```

位置：`scheduler.py:1057` 到 `scheduler.py:1072`

`get_freed_mm_hashes()` 会返回并清空 `freed`：

位置：`encoder_cache_manager.py:255` 到 `encoder_cache_manager.py:266`

所以 `freed` 是一次性的通知队列。

---

## 9. Worker 如何执行 encoder 并写入 cache

执行入口在 `GPUModelRunner.execute_model()`。

在进入 forward 前，会先更新 worker 侧请求状态：

```python
deferred_state_corrections_fn = self._update_states(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4080` 到 `gpu_model_runner.py:4087`

### 9.1 先删除 scheduler 通知释放的物理 cache

`_update_states()` 里会消费：

```python
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

位置：`gpu_model_runner.py:1158` 到 `gpu_model_runner.py:1160`

这就是 encoder cache 的物理释放点。

### 9.2 从 `scheduled_encoder_inputs` 组装 multimodal batch

入口：`gpu_model_runner.py:2850`

`_batch_mm_inputs_from_scheduler()` 会遍历：

```text
scheduler_output.scheduled_encoder_inputs
  → req_id
  → encoder_input_ids
  → self.requests[req_id].mm_features[mm_input_id]
```

然后产出：

```text
mm_hashes:
  每个 multimodal item 的 identifier。

mm_kwargs:
  传给 model.embed_multimodal() 的 modality + data。

mm_lora_refs:
  用于 multimodal LoRA 映射的 request 和 placeholder range。
```

位置：`gpu_model_runner.py:2866` 到 `gpu_model_runner.py:2887`

### 9.3 `_execute_mm_encoder()` 执行 encoder

入口：`gpu_model_runner.py:2889`

核心流程：

```text
1. 从 SchedulerOutput 取本轮要处理的 mm inputs；
2. 如果 modality 是 prompt_embeds，直接把 tensor 放入 encoder_cache；
3. 按 modality 分组和 batch；
4. 如有 multimodal LoRA，设置 tower / connector adapter mapping；
5. 调用 model.embed_multimodal(**mm_kwargs_batch)；
6. sanity_check_mm_encoder_outputs()；
7. 写入 self.encoder_cache[mm_hash] = output；
8. 如启用 EC connector，调用 maybe_save_ec_to_connector()。
```

prompt embeds 分支：`gpu_model_runner.py:2899` 到 `gpu_model_runner.py:2924`

真正 encoder 调用：`gpu_model_runner.py:3062` 到 `gpu_model_runner.py:3088`

写入 cache：`gpu_model_runner.py:3092` 到 `gpu_model_runner.py:3096`

### 9.4 cache 命中时为什么 Worker 不会重新 encode

如果 Scheduler 发现 `mm_hash` 已经缓存，`scheduled_encoder_inputs` 不会包含这个 input_id。

因此 `_batch_mm_inputs_from_scheduler()` 不会把它放进 `mm_kwargs`，`_execute_mm_encoder()` 不会重新计算。

但后续 `_gather_mm_embeddings()` 仍然会在当前 token window 中看到这个 `mm_feature`，并从 `self.encoder_cache[mm_hash]` 读取已有 output。

这就是 cache 复用的实际路径。

---

## 10. Worker 如何消费 encoder output

普通 multimodal 模型和 encoder-decoder 模型的消费方式不同。

### 10.1 普通 multimodal 模型：拼回 `inputs_embeds`

在 `_preprocess()` 中：

```python
self._execute_mm_encoder(scheduler_output)
mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
inputs_embeds_scheduled = self.model.embed_input_ids(
    input_ids,
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

位置：`gpu_model_runner.py:3447` 到 `gpu_model_runner.py:3495`

`_gather_mm_embeddings()` 会：

```text
1. 遍历当前 input_batch 中每个 req；
2. 找当前 scheduled token window 覆盖的 mm_features；
3. 根据 mm_position 计算当前 window 对应的 embeds 范围；
4. 用 mm_feature.identifier 从 encoder_cache 取 tensor；
5. 构造 mm_embeds 和 is_mm_embed mask；
6. 交给 model.embed_input_ids() 把 text token embedding 和 multimodal embedding 合并。
```

关键读取：

```python
encoder_output = self.encoder_cache.get(mm_hash, None)
assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."
```

位置：`gpu_model_runner.py:3149` 到 `gpu_model_runner.py:3151`

这个 assert 体现了 Scheduler 和 Worker 的契约：

```text
只要当前 token window 需要某个 mm_hash，
Scheduler 要么本轮安排它执行 encoder，
要么之前已经保证它在 worker encoder_cache 中。
```

### 10.2 encoder-decoder 模型：传入 `encoder_outputs`

encoder-decoder 分支不做 prompt replacement。

在 `_preprocess()` 末尾：

```python
if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
    encoder_outputs = self._execute_mm_encoder(scheduler_output)
    model_kwargs.update({"encoder_outputs": encoder_outputs})
```

位置：`gpu_model_runner.py:3552` 到 `gpu_model_runner.py:3559`

也就是说：

```text
普通 VLM：encoder output 变成 inputs_embeds 的一部分；
encoder-decoder：encoder output 作为 encoder_outputs 传给 decoder。
```

---

## 11. Scheduler 何时释放 encoder cache 引用

执行完成后，`Scheduler.update_from_output()` 会推进请求状态，并在合适时机释放 request 对 encoder input 的引用。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1576` 到 `scheduler.py:1578`

```python
if request.has_encoder_inputs:
    self._free_encoder_inputs(request)
```

### 11.1 `_free_encoder_inputs()` 的释放条件

入口：`scheduler.py:1866`

它先取当前 request 已经引用的 encoder input ids：

```python
cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(request)
```

位置：`scheduler.py:1866` 到 `scheduler.py:1872`

然后逐个判断是否可以释放引用。

encoder-decoder：

```text
只要 request.num_computed_tokens > 0，说明 encoder 已经完成，
Whisper 这类模型的 cross-attention KV 已经算好，可以释放 encoder input 引用。
```

位置：`scheduler.py:1880` 到 `scheduler.py:1884`

普通 multimodal：

```text
当 start_pos + num_tokens <= request.num_computed_tokens - request.num_output_placeholders，
说明 decoder 已经越过该 multimodal placeholder 区间，
并且 speculative / async placeholder 回退也不会再退回这个区间，
就可以释放该 request 对 encoder output 的引用。
```

位置：`scheduler.py:1885` 到 `scheduler.py:1893`

### 11.2 `free_encoder_input()` 做什么

入口：`encoder_cache_manager.py:216`

它会：

```text
1. 从 request_cached_ids[req_id] 移除 input_id；
2. 如果 request_cached_ids[req_id] 为空，删除这个 req_id；
3. 从 cached[mm_hash] 的 request set 中移除 req_id；
4. 如果 cached[mm_hash] 变成空 set：
   - freeable[mm_hash] = num_encoder_embeds
   - num_freeable_slots += num_encoder_embeds
```

位置：`encoder_cache_manager.py:226` 到 `encoder_cache_manager.py:241`

注意：

```text
这里没有把 mm_hash 加入 freed，
也没有要求 Worker 删除 tensor。
它只是把该条目变成 freeable，也就是“无人引用、可被未来淘汰”。
```

---

## 12. 结束、取消、抢占时如何释放

### 12.1 请求正常结束 / abort

当请求被标记 finished 后，Scheduler 会调用 `_free_request()`。

入口：`scheduler.py:2046`

里面会：

```python
self.encoder_cache_manager.free(request)
self.finished_req_ids.add(request_id)
```

位置：`scheduler.py:2046` 到 `scheduler.py:2055`

`free(request)` 会释放这个 request 持有的所有 encoder input 引用：

位置：`encoder_cache_manager.py:243` 到 `encoder_cache_manager.py:253`

### 12.2 请求被外部 finish / abort

`finish_requests()` 会把 request 状态设置为传入的 finished status，并调用 `_free_request()`。

位置：`scheduler.py:1983` 到 `scheduler.py:2044`

所以外部取消也会释放 encoder cache 引用。

### 12.3 请求被抢占

`_preempt_request()` 会：

```text
1. 释放 KV blocks；
2. encoder_cache_manager.free(request)；
3. 把 request 状态置为 PREEMPTED；
4. num_computed_tokens 重置为 0；
5. 放回 waiting 队列。
```

位置：`scheduler.py:1105` 到 `scheduler.py:1127`

这意味着被抢占的请求不会继续持有 encoder cache 引用。之后如果恢复调度，需要重新通过 cache 命中或重新 encoder 来建立引用。

---

## 13. 物理删除为什么是延迟的

完整释放链路是两阶段的：

```text
阶段一：释放 request 引用
  Scheduler.update_from_output()
    → _free_encoder_inputs() / _free_request() / _preempt_request()
    → EncoderCacheManager.free_encoder_input()
    → cached[mm_hash] 变空
    → freeable[mm_hash] = num_encoder_embeds

阶段二：容量不足时淘汰 freeable 条目
  EncoderCacheManager.can_allocate()
    → pop oldest freeable mm_hash
    → del cached[mm_hash]
    → freed.append(mm_hash)

阶段三：通知 worker 删除物理 tensor
  SchedulerOutput.free_encoder_mm_hashes = get_freed_mm_hashes()
    → GPUModelRunner._update_states()
    → self.encoder_cache.pop(mm_hash, None)
```

为什么要这样设计？

```text
1. 已经无人引用的 encoder output 仍然可能被后续请求复用；
2. 只有容量真的不足时才删除，能提高相同多模态输入的复用率；
3. Scheduler 不持有 GPU tensor，只能通过 SchedulerOutput 通知 Worker 删除。
```

淘汰顺序是 oldest freeable first，因为 `freeable` 是 `OrderedDict`，`popitem(last=False)` 会取最早进入 freeable 的条目。

位置：`encoder_cache_manager.py:78` 和 `encoder_cache_manager.py:177` 到 `encoder_cache_manager.py:181`

---

## 14. EC connector 如何接入 encoder cache 生命周期

EC connector 是 encoder cache 的外部传输 / 共享通道。

Scheduler 初始化时，如果配置了 `ec_transfer_config`，会创建 scheduler 侧 connector：

```python
self.ec_connector = ECConnectorFactory.create_connector(...)
```

位置：`scheduler.py:157` 到 `scheduler.py:161`

### 14.1 Scheduler 侧：判断是否可外部加载

waiting 请求路径中，Scheduler 会先检查外部 encoder cache 是否已经可用：

```python
self.ec_connector.ensure_cache_available(request, num_computed_tokens)
```

位置：`scheduler.py:750` 到 `scheduler.py:760`

如果外部 encoder cache 还没 ready，请求会被暂时跳过，重新放回 waiting/skipped 队列。

### 14.2 Scheduler 侧：分配后更新 connector 状态

当 encoder input 被本地调度或外部加载时，Scheduler 都会调用：

```python
self.ec_connector.update_state_after_alloc(request, i)
```

位置：

- running 路径：`scheduler.py:603` 到 `scheduler.py:612`
- waiting 路径：`scheduler.py:967` 到 `scheduler.py:977`

构造 `SchedulerOutput` 后，Scheduler 会生成 connector metadata：

```python
ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
scheduler_output.ec_connector_metadata = ec_meta
```

位置：`scheduler.py:1084` 到 `scheduler.py:1089`

### 14.3 Worker 侧：加载 / 保存 encoder cache

`ECConnectorModelRunnerMixin` 定义了 worker 侧生命周期。

入口：`code/vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py:24`

在 `_preprocess()` 中，普通 multimodal 分支会包住 encoder 执行和 gather：

```python
with self.maybe_get_ec_connector_output(
    scheduler_output,
    encoder_cache=self.encoder_cache,
) as ec_connector_output:
    self._execute_mm_encoder(scheduler_output)
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
```

位置：`gpu_model_runner.py:3447` 到 `gpu_model_runner.py:3455`

context manager 内部会：

```text
1. bind_connector_metadata(scheduler_output.ec_connector_metadata)；
2. 如果当前 worker 是 consumer，start_load_caches(encoder_cache, ...)；
3. 执行 encoder / gather；
4. finally 中读取 finished_sending / finished_recving；
5. clear_connector_metadata()。
```

位置：`ec_connector_model_runner_mixin.py:60` 到 `ec_connector_model_runner_mixin.py:78`

当本地执行 encoder 后，`_execute_mm_encoder()` 会调用：

```python
self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

位置：`gpu_model_runner.py:3092` 到 `gpu_model_runner.py:3096`

`maybe_save_ec_to_connector()` 最终调用：

```python
connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)
```

位置：`ec_connector_model_runner_mixin.py:27` 到 `ec_connector_model_runner_mixin.py:35`

### 14.4 非 consumer 的特殊路径

在 `execute_model()` 中，如果启用了 EC transfer，且当前不是 consumer：

```python
with self.maybe_get_ec_connector_output(...):
    self._execute_mm_encoder(scheduler_output)
    return make_empty_encoder_model_runner_output(scheduler_output)
```

位置：`gpu_model_runner.py:4088` 到 `gpu_model_runner.py:4094`

这表示该角色只负责执行 / 保存 encoder cache，不继续完整 decoder forward。

`make_empty_encoder_model_runner_output()` 会构造一个只有 request bookkeeping、没有真正采样输出的 `ModelRunnerOutput`。

位置：`code/vllm/vllm/v1/outputs.py:318` 到 `outputs.py:345`

---

## 15. encoder-decoder 模型的特殊 CacheManager

encoder-decoder 模型使用 `EncoderDecoderCacheManager`。

入口：`code/vllm/vllm/v1/core/encoder_cache_manager.py:323`

它和普通 `EncoderCacheManager` 的差异很大：

```text
- check_and_update_cache() 永远返回 False；
- 当前并不复用 encoder-decoder 的 encoder output；
- allocate() 只扣减 num_free_slots，并记录 allocated mm_hash；
- get_cached_input_ids() 返回所有 mm_features 的 input_id；
- get_freed_mm_hashes() 用 allocated / to_free 模拟“执行后释放”的状态迁移。
```

对应源码：`encoder_cache_manager.py:323` 到 `encoder_cache_manager.py:381`

源码注释说明：这是临时实现，encoder-decoder 当前主要用它做 scheduling purpose，后续可能和普通 encoder cache 收敛。

位置：`encoder_cache_manager.py:319` 到 `encoder_cache_manager.py:322`

所以文档里提到 encoder cache 时要区分：

```text
普通 multimodal：真的按 mm_hash 复用 encoder output；
encoder-decoder：目前主要通过类似接口约束调度和释放，不做同样的缓存复用。
```

---

## 16. reset_encoder_cache 如何失效旧缓存

当模型权重更新或调试接口要求清空 encoder cache 时，需要同时清两边：

```text
Scheduler 侧 EncoderCacheManager 逻辑状态；
Worker 侧 GPUModelRunner.encoder_cache 物理 tensor。
```

入口：`code/vllm/vllm/v1/engine/core.py:687`

`EngineCore.reset_encoder_cache()` 会：

```python
self.scheduler.reset_encoder_cache()
self.model_executor.reset_encoder_cache()
```

位置：`engine/core.py:702` 到 `engine/core.py:705`

如果还有请求在跑，会打 warning：

```text
Resetting the encoder cache when requests are in progress may lead to desynced internal caches.
```

位置：`engine/core.py:694` 到 `engine/core.py:700`

Scheduler 侧 reset：

```python
self.encoder_cache_manager.reset()
```

位置：`scheduler.py:2215` 到 `scheduler.py:2221`

Worker 侧 reset：

```python
self.encoder_cache.clear()
self.late_interaction_runner.clear()
```

位置：`gpu_model_runner.py:925` 到 `gpu_model_runner.py:932`

Executor 只是广播控制命令：

```python
self.collective_rpc("reset_encoder_cache")
```

位置：`code/vllm/vllm/v1/executor/abstract.py:314` 到 `abstract.py:316`

---

## 17. 一个完整例子：两个请求复用同一张图片

假设两个请求都带同一张图片，图片 hash 是 `H`。

### 17.1 第一个请求 R1

```text
Scheduler._try_schedule_encoder_inputs(R1)
  → check_and_update_cache(H) 返回 False
  → can_allocate(R1, input_id) 返回 True
  → scheduled_encoder_inputs[R1] = [input_id]
  → EncoderCacheManager.allocate(R1, input_id)

Worker._execute_mm_encoder()
  → model.embed_multimodal(image)
  → encoder_cache[H] = output

Worker._gather_mm_embeddings()
  → 从 encoder_cache[H] 取 output
  → 拼回 inputs_embeds
```

### 17.2 R1 走过图片 placeholder 后

```text
Scheduler.update_from_output(R1)
  → _free_encoder_inputs(R1)
  → free_encoder_input(R1, input_id)
  → cached[H] 变成空 set
  → freeable[H] = num_encoder_embeds
```

此时：

```text
Scheduler 认为 H 无人引用、可淘汰；
Worker 仍然保留 encoder_cache[H] tensor。
```

### 17.3 第二个请求 R2 使用同一张图片

```text
Scheduler._try_schedule_encoder_inputs(R2)
  → check_and_update_cache(H) 返回 True
  → H 从 freeable 移除
  → cached[H].add(R2)
  → scheduled_encoder_inputs 不包含 H

Worker._execute_mm_encoder()
  → 不重新计算 H

Worker._gather_mm_embeddings()
  → 直接读取 encoder_cache[H]
```

这就是 encoder cache 的复用收益。

### 17.4 后续空间不足时淘汰 H

如果 R2 也释放引用，H 再次进入 freeable。之后某个新请求需要空间：

```text
EncoderCacheManager.can_allocate(new_item)
  → free slots 不够
  → freeable slots 够
  → pop oldest freeable H
  → freed.append(H)

SchedulerOutput.free_encoder_mm_hashes = [H]

GPUModelRunner._update_states()
  → encoder_cache.pop(H, None)
```

这时 H 的物理 tensor 才真正从 Worker cache 中删除。

---

## 18. 常见问题

### 18.1 encoder cache key 是什么？

key 是：

```text
request.mm_features[input_id].identifier
```

在文档和代码注释里通常叫 `mm_hash`。

读取位置：`encoder_cache_manager.py:109`、`gpu_model_runner.py:2883`、`gpu_model_runner.py:3149`

### 18.2 一个 request 可以有多个 encoder input 吗？

可以。

`SchedulerOutput.scheduled_encoder_inputs` 是：

```text
req_id -> list[input_id]
```

例如一个请求有两张图片，本轮可能是：

```text
{"req-1": [0, 1]}
```

字段注释位置：`output.py:201` 到 `output.py:204`

### 18.3 cache 命中后 SchedulerOutput 如何表达？

命中的 input 不会出现在 `scheduled_encoder_inputs` 中。

它的表达方式是隐式的：

```text
Scheduler 已经通过 check_and_update_cache() 建立 request 引用；
Worker 后续 gather 时会从已有 encoder_cache[mm_hash] 读取。
```

如果 Worker 实际没有这个 hash，`_gather_mm_embeddings()` 会触发：

```python
assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."
```

位置：`gpu_model_runner.py:3149` 到 `gpu_model_runner.py:3151`

### 18.4 `free_encoder_mm_hashes` 表示什么？

它表示：

```text
Scheduler 已经逻辑淘汰、Worker 可以物理删除的 mm_hash 列表。
```

它不是“请求刚刚结束”的列表，也不是“刚刚释放引用”的列表。

只有当 freeable 条目被 `can_allocate()` 淘汰后，才会进入 `freed`，并在下一个 `SchedulerOutput` 中通过 `free_encoder_mm_hashes` 发给 Worker。

### 18.5 为什么 `num_freeable_slots` 初始等于 `cache_size`？

初始化时：

```python
self.num_free_slots = cache_size
self.num_freeable_slots = cache_size
```

位置：`encoder_cache_manager.py:67` 到 `encoder_cache_manager.py:70`

这里的 `num_freeable_slots` 表示“当前可用于满足分配的容量上限”，包括真实 free slots 和可回收 slots 的综合账本。分配时它会随 `num_free_slots` 一起扣减；释放 request 引用时，只增加 `num_freeable_slots`，不增加 `num_free_slots`，因为物理空间还没删。

### 18.6 `prompt_embeds` 为什么也进入 encoder cache？

`prompt_embeds` 已经是模型 embedding 空间中的 tensor，不需要 encoder tower。

但为了复用统一的 gather / splice 路径，Worker 会直接把它写入：

```python
self.encoder_cache[mm_hash] = pe_tensor.to(self.device)
```

位置：`gpu_model_runner.py:2899` 到 `gpu_model_runner.py:2915`

所以它也走 `mm_hash -> encoder output` 的缓存通道。

### 18.7 为什么释放时要考虑 `num_output_placeholders`？

普通 multimodal 释放条件里用了：

```text
request.num_computed_tokens - request.num_output_placeholders
```

位置：`scheduler.py:1885` 到 `scheduler.py:1893`

原因是 speculative decoding / async scheduling 可能有占位输出 token，后续 rejection 会回退 computed token 语义。只有扣掉 placeholders 后仍然越过 multimodal placeholder 区间，才确认不会再需要该 encoder output。

---

## 19. 生命周期总览图

完整生命周期可以压缩成：

```text
初始化
  → MultiModalBudget 计算 encoder_compute_budget / encoder_cache_size
  → Scheduler 创建 EncoderCacheManager
  → Worker 创建 encoder_cache dict

调度
  → schedule() 初始化 scheduled_encoder_inputs / encoder_compute_budget
  → _try_schedule_encoder_inputs() 扫当前 token window
  → check_and_update_cache() 命中则复用
  → can_allocate() 未命中则检查预算和容量
  → allocate() 建立 request 引用并扣容量
  → SchedulerOutput 携带 scheduled_encoder_inputs / free_encoder_mm_hashes

执行
  → GPUModelRunner._update_states() 删除 free_encoder_mm_hashes
  → _batch_mm_inputs_from_scheduler() 组装 mm inputs
  → _execute_mm_encoder() 执行或 passthrough prompt_embeds
  → encoder_cache[mm_hash] = output
  → _gather_mm_embeddings() 从 cache 读取
  → 普通 VLM 拼进 inputs_embeds
  → encoder-decoder 放进 model_kwargs["encoder_outputs"]

回收
  → Scheduler.update_from_output()
  → _free_encoder_inputs() 释放已经消费完的 input 引用
  → 请求 finish / abort / preempt 时 free(request)
  → 无引用条目进入 freeable
  → 后续 can_allocate() 容量不足时淘汰 freeable
  → freed mm_hash 进入下一次 SchedulerOutput.free_encoder_mm_hashes
  → Worker pop 物理 tensor

失效
  → EngineCore.reset_encoder_cache()
  → Scheduler.reset_encoder_cache()
  → Executor / Worker / ModelRunner reset_encoder_cache()
```

---

## 20. 一句话总结

Encoder cache 的生命周期不是“算完立即释放”，而是一个三段式协议：Scheduler 用 `EncoderCacheManager` 维护 `mm_hash` 的引用和容量账本，Worker 用 `encoder_cache` 保存真正的 tensor；请求消费完 encoder output 后只释放引用，等未来空间不足时才淘汰 freeable 条目，并通过 `SchedulerOutput.free_encoder_mm_hashes` 让 Worker 删除物理缓存。
