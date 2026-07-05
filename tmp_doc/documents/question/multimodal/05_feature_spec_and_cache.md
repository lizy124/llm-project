# 05. MultiModalFeatureSpec 和 MultiModalCache 如何组织？

源码位置：

- `code/vllm/vllm/multimodal/inputs.py`
- `code/vllm/vllm/multimodal/cache.py`
- `code/vllm/vllm/v1/engine/input_processor.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/encoder_cache_manager.py`
- `code/vllm/vllm/v1/worker/worker_base.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：多模态 feature 在 vLLM 中如何被描述、缓存和复用。尤其要区分 processor cache、`MultiModalFeatureSpec`、encoder cache、decoder KV cache 的边界。

---

## 1. 一句话回答

`MultiModalFeatureSpec` 是“单个多模态 item 的运行时描述”；`MultiModalCache` 是“processor 输出 / IPC 传输层缓存”的基础设施；`EncoderCacheManager` 和 `GPUModelRunner.encoder_cache` 才是“模型 encoder output 缓存”。

主链路可以压缩成：

```text
raw media / prompt_embeds
  → MultiModalProcessor 生成 mm_kwargs / mm_placeholders / mm_hashes
  → InputProcessor 展平成 list[MultiModalFeatureSpec]
  → Request.mm_features
  → processor cache 可能复用 processed tensor / 避免 IPC
  → Scheduler 根据 EncoderCacheManager 判断哪些 feature 要跑 encoder
  → SchedulerOutput.scheduled_encoder_inputs
  → Worker / ModelRunner 执行 encoder 或复用 GPU encoder_cache
  → _gather_mm_embeddings() 按 placeholder range 把 encoder output 拼回 token 序列
```

所以要记住三句话：

```text
MultiModalFeatureSpec 描述 feature，不拥有缓存策略；
MultiModalCache 缓存 processor 后的输入张量，不缓存模型 encoder output；
EncoderCacheManager / GPUModelRunner.encoder_cache 管理 encoder output，不等同于 decoder KV cache。
```

---

## 2. 四个容易混淆的对象

### 2.1 MultiModalFeatureSpec

位置：`inputs.py:301`

它表示一个已经过 processor 处理并绑定了 prompt 位置的多模态输入 item。

字段是：

```python
data: MultiModalKwargsItem | None
modality: str
identifier: str
mm_position: PlaceholderRange
mm_hash: str | None = None
```

位置：`inputs.py:301` 到 `inputs.py:332`

含义：

```text
data：
  单个多模态 item 的 processed kwargs，例如 pixel_values / image_grid_thw / audio feature。
  如果 processor cache 已命中且为了避免 IPC，可为 None 或被替换成地址 item。

modality：
  image / video / audio / prompt_embeds 等模态名。

identifier：
  encoder output cache 的 key。
  当前构造逻辑会基于原始 mm_hash，并在需要时加 LoRA 前缀。

mm_position：
  该 item 对应 prompt 中的 placeholder 位置和长度。

mm_hash：
  processor output cache 的 key。
  通常是不带 LoRA 前缀的原始多模态数据 hash，用来让 processor cache 跨 LoRA 复用。
```

最关键的区分是：

```text
mm_hash：processor cache key，偏输入数据本身；
identifier：encoder cache key，偏模型执行结果，可能受 LoRA 影响。
```

### 2.2 PlaceholderRange

位置：`inputs.py:118`

`PlaceholderRange` 记录一个多模态 item 在 token 序列里的占位区间：

```python
offset: int
length: int
is_embed: torch.Tensor | None = None
```

位置：`inputs.py:118` 到 `inputs.py:145`

它解决两个问题：

```text
1. 这个 image / audio / video 对应 prompt 中哪一段 token；
2. 如果不是每个 placeholder token 都需要替换成 embedding，哪些位置是真的 embedding 位置。
```

相关 helper：

```text
get_num_embeds()：
  返回这个 feature 最终会占用多少 encoder embedding。

get_embeds_indices_in_range(start_idx, end_idx)：
  在 chunked prefill 场景下，把当前 token window 映射到 encoder output slice。

extract_embeds_range()：
  抽出 prompt 中连续 embedding 区间，调试 / 对齐时有用。
```

位置：`inputs.py:152` 到 `inputs.py:202`

这就是后面 Scheduler 能按 encoder budget 调度、ModelRunner 能按窗口切片复用 encoder output 的基础。

### 2.3 MultiModalCache

位置：`cache.py:98`

`MultiModalCache` 这个类本身不是“某一种缓存实例”，而是一组用于多模态 cache value 的工具方法：

```text
get_leaf_size()
get_item_size()
get_item_complexity()
get_lru_cache()
```

位置：`cache.py:98` 到 `cache.py:168`

它主要负责：

```text
- 递归估算 tensor / nested structure 的大小；
- 用 byte capacity 构造 LRUCache；
- 给 processor sender / receiver cache 复用同一套容量统计逻辑。
```

真正的 processor cache 类型在它下面：

```text
MultiModalProcessorOnlyCache
MultiModalProcessorSenderCache
ShmObjectStoreSenderCache
MultiModalReceiverCache
ShmObjectStoreReceiverCache
```

这些 cache 缓存的是 processed multimodal kwargs，不是 encoder output。

### 2.4 EncoderCacheManager

位置：`encoder_cache_manager.py:17`

`EncoderCacheManager` 是 Scheduler 侧的 encoder cache 账本。它不直接持有 GPU tensor，而是记录：

```text
哪些 encoder output 已经缓存；
哪些 request 正在引用这些 output；
还有多少 encoder embedding cache slot；
哪些无引用 entry 可以被释放；
哪些 mm_hash / identifier 需要通知 worker 删除。
```

真实 GPU tensor 在 `GPUModelRunner.encoder_cache`：

```python
self.encoder_cache: dict[str, torch.Tensor] = {}
```

位置：`gpu_model_runner.py:533` 到 `gpu_model_runner.py:534`

因此 encoder cache 是两层：

```text
Scheduler / EncoderCacheManager：
  决策和资源账本。

Worker / GPUModelRunner.encoder_cache：
  真正保存 encoder output tensor。
```

---

## 3. MultiModalFeatureSpec 是如何生成的

入口在 V1 的 `InputProcessor`。

当 decoder input 是 multimodal 时：

```python
if decoder_inputs["type"] == "multimodal":
    decoder_mm_inputs = decoder_inputs["mm_kwargs"]
    decoder_mm_positions = decoder_inputs["mm_placeholders"]
    decoder_mm_hashes = decoder_inputs["mm_hashes"]
```

位置：`input_processor.py:335` 到 `input_processor.py:338`

随后 vLLM 会做三件事。

### 3.1 校验 mm_hashes

```python
if not all(isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)):
    raise ValueError(...)
```

位置：`input_processor.py:340` 到 `input_processor.py:347`

这说明 processor 必须为每个多模态 item 产出字符串 hash。

### 3.2 按 prompt 位置排序

```python
sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)
```

位置：`input_processor.py:349` 到 `input_processor.py:352`

原因是 `mm_kwargs` 原来按 modality 分组，例如：

```text
image: [img0, img1]
audio: [audio0]
```

但模型 forward 时需要按 prompt 中出现顺序处理 feature，例如：

```text
<image0> text <audio0> text <image1>
```

所以最终 `Request.mm_features` 是一个按 `mm_position.offset` 排好序的 flat list。

### 3.3 构造 MultiModalFeatureSpec

核心代码：

```python
MultiModalFeatureSpec(
    data=decoder_mm_inputs[modality][idx],
    modality=modality,
    identifier=self._get_mm_identifier(base_mm_hash, lora_request),
    mm_position=decoder_mm_positions[modality][idx],
    mm_hash=base_mm_hash,
)
```

位置：`input_processor.py:354` 到 `input_processor.py:367`

这里直接体现了两种 hash 的边界：

```text
base_mm_hash → mm_hash：
  原始 processor cache key。

_get_mm_identifier(base_mm_hash, lora_request) → identifier：
  encoder cache key，可能带 LoRA 信息。
```

生成后的 spec 会挂到 `EngineCoreRequest.mm_features`：

```python
EngineCoreRequest(..., mm_features=mm_features, ...)
```

位置：`input_processor.py:370` 到 `input_processor.py:385`

---

## 4. MultiModalFeatureSpec 在 Request 中的角色

`Request` 初始化时直接保存：

```python
self.mm_features = mm_features or []
```

位置：`request.py:156` 到 `request.py:158`

然后提供几个和 encoder 调度相关的 helper：

```python
@property
def num_encoder_inputs(self) -> int:
    return len(self.mm_features)

@property
def has_encoder_inputs(self) -> bool:
    return self.num_encoder_inputs > 0

def get_num_encoder_embeds(self, input_id: int) -> int:
    return self.mm_features[input_id].mm_position.get_num_embeds()
```

位置：`request.py:258` 到 `request.py:287`

这说明在 V1 Scheduler 视角里：

```text
一个 MultiModalFeatureSpec 就是一个 encoder input。
```

`input_id` 指的是 `request.mm_features` 的下标。例如：

```text
request.mm_features = [image0, audio0, image1]

scheduled_encoder_inputs = {
  req_id: [0, 2]
}

含义：
  本轮只处理 image0 和 image1 的 encoder output。
```

---

## 5. FeatureSpec 如何跨进程发送到 Worker

SchedulerOutput 里新请求数据定义为：

```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    ...
```

位置：`output.py:30` 到 `output.py:41`

从 request 构造 `NewRequestData` 时：

```python
mm_features=request.mm_features
```

位置：`output.py:46` 到 `output.py:65`

也就是说：

```text
Request.mm_features
  → SchedulerOutput.scheduled_new_reqs[i].mm_features
  → WorkerWrapperBase / Worker / GPUModelRunner.requests[req_id].mm_features
```

已经进入 Worker 的老请求不会每步重复发送完整 `mm_features`；它们缓存在 Worker 侧 request state 中。后续调度只通过 `scheduled_cached_reqs` 发送 diff。

---

## 6. Processor cache：缓存 processor 结果，不缓存 encoder output

### 6.1 P0 / P1 cache 设计

`BaseMultiModalCache` 的注释把多模态 processor cache 分成两端：

```text
P0: API / frontend process
P1: EngineCore / worker process
```

缓存协议是：

```text
P0: is_cached() x N → get_and_update()
P1:                    get_and_update()
```

位置：`cache.py:175` 到 `cache.py:198`

关键约束：

```text
P0 和 P1 的 cache eviction order 必须保持一致；
这样 P0 只要查自己的 cache，就能推断 P1 是否已经有同一个 item；
命中时可以少传 tensor，甚至只传 None 或 shared-memory address。
```

### 6.2 MultiModalProcessorOnlyCache

位置：`cache.py:326`

用于 IPC cache 禁用时的 P0 cache。

逻辑是：

```text
命中：返回 cached_item.item 和 prompt_updates；
未命中：把 item + prompt_updates 存入本地 LRU，然后返回输入。
```

位置：`cache.py:326` 到 `cache.py:364`

这类 cache 存的是：

```text
MultiModalProcessorCacheItem(
  item=MultiModalKwargsItem,
  prompt_updates=Sequence[ResolvedPromptUpdate],
)
```

位置：`cache.py:42` 到 `cache.py:60`

### 6.3 MultiModalProcessorSenderCache

位置：`cache.py:379`

用于 IPC caching 启用时的 P0 sender cache。

逻辑是：

```text
命中：返回 None + cached prompt_updates，避免重复 IPC 传 tensor；
未命中：只在 P0 存 metadata，返回原始 item 给 P1。
```

位置：`cache.py:379` 到 `cache.py:434`

这里 P0 不保存完整 tensor，而是保存 `item_size` 和 `prompt_updates`：

```text
MultiModalProcessorCacheItemMetadata
```

位置：`cache.py:62` 到 `cache.py:85`

原因是：

```text
P1 才真正需要 processed tensor；
P0 只需要维持和 P1 一致的 LRU eviction 顺序。
```

### 6.4 ShmObjectStoreSenderCache

位置：`cache.py:437`

这是 shared memory 版本的 sender cache。

逻辑是：

```text
命中：返回 address / monotonic_id 组成的 MultiModalKwargsItem；
未命中：把 processed item 写入共享内存，并返回 address item；
太大或缓存满：退回返回原始 mm_item。
```

位置：`cache.py:437` 到 `cache.py:581`

它返回的 address item 形态是：

```text
MultiModalKwargsItem({
  "address": MultiModalFieldElem(...),
  "monotonic_id": MultiModalFieldElem(...),
})
```

位置：`cache.py:567` 到 `cache.py:581`

### 6.5 MultiModalReceiverCache

位置：`cache.py:630`

P1 receiver cache 的普通版本：

```text
命中：用 P1 缓存里的 MultiModalKwargsItem 替换输入；
未命中：把输入 item 存入 P1 LRU，然后返回输入。
```

位置：`cache.py:630` 到 `cache.py:675`

`get_and_update_features()` 会直接修改 feature：

```python
feature.data = self.get_and_update_item(feature.data, cache_key)
```

位置：`cache.py:589` 到 `cache.py:608`

cache key 选择是：

```python
cache_key = feature.mm_hash or feature.identifier
```

位置：`cache.py:598` 到 `cache.py:607`

含义：

```text
优先用不带 LoRA 前缀的 mm_hash 共享 processor output；
如果老数据没有 mm_hash，再 fallback 到 identifier。
```

### 6.6 ShmObjectStoreReceiverCache

位置：`cache.py:678`

shared memory receiver cache 的逻辑更简单：

```text
如果 feature.data 是 address item：
  通过 address / monotonic_id 从共享内存取真实 processed item；
否则：
  原样返回 feature.data。
```

位置：`cache.py:712` 到 `cache.py:723`

它不是 LRUCache 版本的“存一份 tensor”，而是 shared memory reader。

---

## 7. WorkerWrapperBase 在哪里应用 processor cache

执行模型前，`WorkerWrapperBase.execute_model()` 会先调用 `_apply_mm_cache()`：

```python
def execute_model(self, scheduler_output):
    self._apply_mm_cache(scheduler_output)
    return self.worker.execute_model(scheduler_output)
```

位置：`worker_base.py:340` 到 `worker_base.py:345`

`_apply_mm_cache()` 只处理本轮新请求：

```python
for req_data in scheduler_output.scheduled_new_reqs:
    req_data.mm_features = mm_cache.get_and_update_features(req_data.mm_features)
```

位置：`worker_base.py:330` 到 `worker_base.py:338`

这点很重要：

```text
processor cache 发生在真实 Worker 之前；
它更新的是 NewRequestData.mm_features[*].data；
它只解决 processed input 的复用 / IPC，和 encoder output 是否已缓存不是同一件事。
```

命中后的 `feature.data` 可能是：

```text
- P1 cache 中真实 MultiModalKwargsItem；
- shared memory 里取回的 MultiModalKwargsItem；
- None 或 address item 经 receiver cache 替换后的结果。
```

进入 `GPUModelRunner` 后，`req_state.mm_features` 已经是 receiver cache 处理后的版本。

---

## 8. Encoder cache：Scheduler 侧资源账本

`EncoderCacheManager` 的缓存粒度是单个 `MultiModalFeatureSpec`，也就是 `request.mm_features[input_id]`。

### 8.1 核心状态

位置：`encoder_cache_manager.py:67` 到 `encoder_cache_manager.py:80`

核心字段：

```text
cache_size：
  encoder cache 总容量，单位是 encoder embeddings 数量。

num_free_slots：
  当前完全空闲的 slot 数。

num_freeable_slots：
  当前可通过淘汰无引用 entry 回收的 slot 数。

cached: dict[str, set[str]]：
  mm_hash / identifier → 正在引用该 encoder output 的 request ids。

request_cached_ids: dict[str, set[int]]：
  request id → 该 request 哪些 input_id 已经占用 encoder cache。

freeable: OrderedDict[str, int]：
  无 request 引用、可在未来分配时淘汰的 encoder output。

freed: list[str]：
  最近真正淘汰、需要通知 Worker 删除 GPU tensor 的 key。
```

源码注释里说 cache key 是 `mm_hash`，但当前实现实际取的是：

```python
mm_hash = request.mm_features[input_id].identifier
```

位置：`encoder_cache_manager.py:109`、`encoder_cache_manager.py:195`、`encoder_cache_manager.py:227`

所以在多模态 + LoRA 场景下要按当前代码理解：

```text
encoder cache key = feature.identifier
processor cache key = feature.mm_hash or feature.identifier
```

### 8.2 cache 命中：check_and_update_cache

位置：`encoder_cache_manager.py:94`

逻辑：

```text
1. 取 request.mm_features[input_id].identifier；
2. 如果不在 cached，返回 False；
3. 如果已缓存但当前无 request 引用，从 freeable 移除，并减少 num_freeable_slots；
4. 把 request_id 加到 cached[key]；
5. 把 input_id 加到 request_cached_ids[request_id]；
6. 返回 True。
```

位置：`encoder_cache_manager.py:94` 到 `encoder_cache_manager.py:121`

这一步不是“读取 GPU tensor”，只是 Scheduler 侧确认：

```text
这个 encoder output 已存在，后续本轮不用再 schedule encoder compute。
```

### 8.3 cache 分配：can_allocate + allocate

`can_allocate()` 先判断预算：

```text
num_embeds = request.get_num_encoder_embeds(input_id)

如果 num_embeds > encoder_compute_budget：
  本轮不能调度。

如果 num_embeds + 已调度 embeds <= num_free_slots：
  可以分配。

否则如果可以从 freeable 回收：
  按 oldest-first 淘汰 freeable entry，并把 key 放入 freed。

否则：
  不能分配。
```

位置：`encoder_cache_manager.py:123` 到 `encoder_cache_manager.py:182`

`allocate()` 才真正更新账本：

```text
1. 确保 cached[key] 存在；
2. cached[key].add(request_id)；
3. request_cached_ids[request_id].add(input_id)；
4. num_free_slots -= num_encoder_embeds；
5. num_freeable_slots -= num_encoder_embeds。
```

位置：`encoder_cache_manager.py:184` 到 `encoder_cache_manager.py:210`

注意这里仍然没有 GPU tensor 写入。真实写入发生在 ModelRunner `_execute_mm_encoder()` 后。

### 8.4 request 完成 / preempt 后释放引用

`free_encoder_input()` 释放的是 request 对 encoder output 的引用：

```text
1. 从 request_cached_ids[req_id] 移除 input_id；
2. 从 cached[key] 的 request set 移除 req_id；
3. 如果 cached[key] 已无 request 引用，把 key 放入 freeable；
4. 增加 num_freeable_slots。
```

位置：`encoder_cache_manager.py:216` 到 `encoder_cache_manager.py:241`

`free(request)` 会释放该 request 的所有 cached encoder input：

位置：`encoder_cache_manager.py:243` 到 `encoder_cache_manager.py:253`

关键点：

```text
释放 request 引用 ≠ 立即删除 GPU encoder output。
```

无引用 entry 会先留在 `freeable`，以后空间不够时才被 eviction。这样相同多模态数据的后续请求仍可能复用 encoder output。

### 8.5 freed 如何通知 Worker

当 `can_allocate()` 真的淘汰 freeable entry 时，会把 key 放入 `self.freed`。

Scheduler 构造 `SchedulerOutput` 时：

```python
free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes()
```

位置：`scheduler.py:1057` 到 `scheduler.py:1072`

`get_freed_mm_hashes()` 返回并清空 `freed`：

位置：`encoder_cache_manager.py:255` 到 `encoder_cache_manager.py:266`

Worker 侧 `_update_states()` 中删除 GPU tensor：

```python
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    self.encoder_cache.pop(mm_hash, None)
```

位置：`gpu_model_runner.py:1158` 到 `gpu_model_runner.py:1160`

所以删除链路是：

```text
Scheduler can_allocate() 淘汰 freeable key
  → EncoderCacheManager.freed.append(key)
  → SchedulerOutput.free_encoder_mm_hashes
  → GPUModelRunner._update_states()
  → encoder_cache.pop(key)
```

---

## 9. SchedulerOutput 如何表达 encoder 调度

`SchedulerOutput` 中和本问题直接相关的字段是：

```python
scheduled_encoder_inputs: dict[str, list[int]]
free_encoder_mm_hashes: list[str]
```

位置：`output.py:201` 到 `output.py:215`

含义：

```text
scheduled_encoder_inputs：
  req_id → 这个 request 本轮需要执行 encoder 的 input_id 列表。

free_encoder_mm_hashes：
  本轮 Worker 需要从 GPU encoder_cache 删除的 key 列表。
```

示例：

```text
scheduled_encoder_inputs = {
  "req-1": [0, 2],
  "req-3": [1],
}

表示：
  req-1 的第 0、2 个 MultiModalFeatureSpec 本轮要跑 encoder；
  req-3 的第 1 个 MultiModalFeatureSpec 本轮要跑 encoder。
```

如果某个 input 的 encoder output 已经命中 `EncoderCacheManager`，它不会出现在 `scheduled_encoder_inputs` 中，但后续 `_gather_mm_embeddings()` 仍会从 `GPUModelRunner.encoder_cache` 取它。

---

## 10. ModelRunner 如何使用 encoder cache

### 10.1 从 SchedulerOutput 取本轮要编码的 features

`_batch_mm_inputs_from_scheduler()` 读取：

```python
scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
```

位置：`gpu_model_runner.py:2866`

然后遍历：

```python
for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
    req_state = self.requests[req_id]
    for mm_input_id in encoder_input_ids:
        mm_feature = req_state.mm_features[mm_input_id]
```

位置：`gpu_model_runner.py:2875` 到 `gpu_model_runner.py:2879`

如果 `mm_feature.data is None`，会跳过：

```python
if mm_feature.data is None:
    continue
```

位置：`gpu_model_runner.py:2879` 到 `gpu_model_runner.py:2881`

否则收集：

```python
mm_hashes.append(mm_feature.identifier)
mm_kwargs.append((mm_feature.modality, mm_feature.data))
mm_lora_refs.append((req_id, mm_feature.mm_position))
```

位置：`gpu_model_runner.py:2883` 到 `gpu_model_runner.py:2885`

这再次说明 ModelRunner 侧 encoder cache key 使用 `identifier`。

### 10.2 执行 encoder 并写入 GPU encoder_cache

`_execute_mm_encoder()` 会先从 scheduler output batch 本轮 encoder inputs：

位置：`gpu_model_runner.py:2889` 到 `gpu_model_runner.py:2897`

普通多模态输入会经过：

```python
model.embed_multimodal(**mm_kwargs_batch)
```

位置：`gpu_model_runner.py:3082` 到 `gpu_model_runner.py:3085`

执行完成后写入：

```python
for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self.encoder_cache[mm_hash] = output
```

位置：`gpu_model_runner.py:3092` 到 `gpu_model_runner.py:3096`

这里变量名叫 `mm_hash`，但内容来自 `mm_feature.identifier`。

### 10.3 prompt_embeds 是 passthrough 特例

`prompt_embeds` 模态已经是模型 embedding 空间里的 tensor，不需要再跑 encoder。

代码会直接把它放入 encoder cache：

```python
self.encoder_cache[mm_hashes[i]] = pe_tensor.to(self.device)
```

位置：`gpu_model_runner.py:2899` 到 `gpu_model_runner.py:2915`

然后从待 encoder 的列表里过滤掉 `prompt_embeds`。

含义：

```text
prompt_embeds 也走 MultiModalFeatureSpec / PlaceholderRange / encoder_cache 对齐机制；
但它不走 model.embed_multimodal()。
```

### 10.4 从 encoder_cache 收集 embedding

`_gather_mm_embeddings()` 并不只看本轮 `scheduled_encoder_inputs`。它会根据本轮 token window 找到 overlap 的 `mm_features`：

```python
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_scheduled_tokens,
)
```

位置：`gpu_model_runner.py:3123` 到 `gpu_model_runner.py:3128`

然后对每个 feature：

```python
mm_hash = mm_feature.identifier
encoder_output = self.encoder_cache.get(mm_hash, None)
assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."
```

位置：`gpu_model_runner.py:3149` 到 `gpu_model_runner.py:3151`

这说明：

```text
无论这个 feature 的 encoder output 是本轮刚算出来的，还是之前缓存命中的，
只要当前 token window 要消费它，GPUModelRunner.encoder_cache 里必须有对应 output。
```

随后按 placeholder window 切片：

```text
start_idx / end_idx：当前调度 token window 在 placeholder 内的相对区间；
get_embeds_indices_in_range()：映射到 encoder output 的 embedding 区间；
mm_embeds_item = encoder_output[...]：取出本轮需要拼入 input embedding 的部分。
```

位置：`gpu_model_runner.py:3135` 到 `gpu_model_runner.py:3158`

最后返回：

```python
return mm_embeds, is_mm_embed
```

位置：`gpu_model_runner.py:3196`

---

## 11. 多模态 embeddings 如何进入 forward

在 `_preprocess()` 中，decoder-only 多模态模型的路径是：

```text
1. maybe_get_ec_connector_output(...)
2. _execute_mm_encoder(scheduler_output)
3. _gather_mm_embeddings(scheduler_output)
4. model.embed_input_ids(input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed)
5. _prepare_mm_inputs()
6. forward 使用 inputs_embeds
```

对应源码：`gpu_model_runner.py:3447` 到 `gpu_model_runner.py:3499`

关键代码：

```python
inputs_embeds_scheduled = self.model.embed_input_ids(
    self.input_ids.gpu[:num_scheduled_tokens],
    multimodal_embeddings=mm_embeds,
    is_multimodal=is_mm_embed,
)
```

位置：`gpu_model_runner.py:3483` 到 `gpu_model_runner.py:3488`

这说明多模态模型最终不是把 image/audio/video tensor 直接塞进 language model forward，而是先变成 embedding，再与文本 token embedding 合并为 `inputs_embeds`。

对比普通文本路径：

```text
文本请求：
  input_ids → model forward

多模态请求：
  input_ids + encoder_cache[identifier] → inputs_embeds → model forward
```

---

## 12. 三层 cache 的边界

### 12.1 Processor cache / MultiModalCache

缓存内容：

```text
processor 输出后的 MultiModalKwargsItem / prompt_updates / shared-memory address。
```

典型位置：

```text
cache.py:326  MultiModalProcessorOnlyCache
cache.py:379  MultiModalProcessorSenderCache
cache.py:437  ShmObjectStoreSenderCache
cache.py:630  MultiModalReceiverCache
cache.py:678  ShmObjectStoreReceiverCache
```

key：

```text
feature.mm_hash 优先；fallback feature.identifier。
```

主要收益：

```text
减少重复 processor 计算；
减少 P0 → P1 的 tensor IPC；
减少 frontend process 保留大 tensor 的内存压力。
```

不负责：

```text
不缓存 model.embed_multimodal() 的 GPU output；
不决定本轮 encoder 是否调度；
不管理 decoder KV block。
```

### 12.2 Encoder cache

缓存内容：

```text
model.embed_multimodal() 或 prompt_embeds passthrough 产生的 encoder output tensor。
```

两层结构：

```text
EncoderCacheManager：Scheduler 侧账本；
GPUModelRunner.encoder_cache：Worker 侧真实 tensor。
```

key：

```text
feature.identifier。
```

主要收益：

```text
避免同一多模态 item 重复执行 vision/audio/video encoder；
支持 chunked prefill 时跨 step 复用同一个 encoder output；
支持不同请求复用相同 media 的 encoder output，前提是 identifier 相同。
```

不负责：

```text
不缓存 raw media；
不缓存 processor prompt_updates；
不缓存 decoder self-attention KV。
```

### 12.3 Decoder KV cache

缓存内容：

```text
decoder self-attention 的 key/value blocks。
```

它和多模态 feature cache 的关系是：

```text
encoder cache 解决“多模态输入如何变成 embedding”；
KV cache 解决“decoder token 的 attention 历史如何复用”。
```

即使一个 image 的 encoder output 命中 encoder cache，decoder 仍然需要正常分配 / 写入 KV cache；即使 prefix KV cache 命中，也不代表 processor cache 或 encoder cache 一定命中。

---

## 13. 一条完整生命周期

### 13.1 第一次请求某张图片

```text
1. API 收到 raw image；
2. MultiModalProcessor 处理 image，生成 processed kwargs、placeholder、mm_hash；
3. InputProcessor 构造 MultiModalFeatureSpec：
     data = processed kwargs
     mm_hash = base media hash
     identifier = base hash 或 LoRA-prefixed hash
     mm_position = placeholder range
4. P0 processor cache 未命中，写入 sender cache / shared memory；
5. WorkerWrapperBase 的 receiver cache 未命中，写入 P1 processor cache；
6. Scheduler 的 EncoderCacheManager 未命中；
7. Scheduler 把 input_id 放入 scheduled_encoder_inputs；
8. GPUModelRunner._execute_mm_encoder() 执行 model.embed_multimodal()；
9. GPUModelRunner.encoder_cache[identifier] = encoder_output；
10. _gather_mm_embeddings() 从 encoder_cache 取 output 并拼入 inputs_embeds；
11. decoder forward 写入 KV cache。
```

### 13.2 同一图片再次请求，processor cache 命中但 encoder cache 未必命中

```text
1. P0 processor cache 命中：可能返回 None 或 shared-memory address；
2. P1 receiver cache / shm receiver 得到 processed kwargs；
3. Scheduler 检查 EncoderCacheManager：
     如果 encoder output 仍在账本中且 key 相同，命中；
     否则仍要重新调度 encoder。
```

processor cache 命中只说明：

```text
processor output 可复用。
```

不自动说明：

```text
GPU encoder output 可复用。
```

### 13.3 encoder cache 命中

```text
1. EncoderCacheManager.check_and_update_cache() 返回 True；
2. Scheduler 不把该 input_id 放进 scheduled_encoder_inputs；
3. Worker 不重新执行 _execute_mm_encoder()；
4. _gather_mm_embeddings() 仍按 identifier 从 GPUModelRunner.encoder_cache 取 output；
5. 当前 token window 需要哪段，就切哪段。
```

### 13.4 请求结束

```text
1. Scheduler 调用 EncoderCacheManager.free(request)；
2. 该 request 对 encoder output 的引用被释放；
3. 如果没有其他 request 引用，对应 key 进入 freeable；
4. GPU tensor 暂时不删，等待后续空间不足时淘汰；
5. 真正淘汰时 key 进入 freed；
6. 下一个 SchedulerOutput.free_encoder_mm_hashes 通知 Worker 删除 GPU tensor。
```

---

## 14. LoRA 对 hash 的影响

`InputProcessor` 构造 `MultiModalFeatureSpec` 时：

```text
mm_hash = base_mm_hash
identifier = _get_mm_identifier(base_mm_hash, lora_request)
```

位置：`input_processor.py:356` 到 `input_processor.py:367`

这说明：

```text
processor output 通常只依赖原始 media 和 processor 参数，不依赖 LoRA；
encoder output 可能依赖 LoRA，尤其 tower / connector LoRA 场景；
所以 encoder cache key 需要区分 LoRA，而 processor cache key 可以尽量复用。
```

因此：

```text
同一张图 + 不同 LoRA：
  processor cache 可以用同一个 mm_hash；
  encoder cache 可能使用不同 identifier。
```

这也是 receiver cache 里优先使用 `feature.mm_hash or feature.identifier` 的原因。

---

## 15. 为什么 feature.data 可以是 None

`MultiModalFeatureSpec.data` 的注释明确说明：

```text
Can be None if the item is cached, to skip IPC between API server and engine core processes.
```

位置：`inputs.py:311` 到 `inputs.py:317`

这只发生在 processor / IPC cache 层。

含义是：

```text
P0 判断 P1 已经有 processed item，
所以不再把大 tensor 放进请求 payload。
```

WorkerWrapperBase 会在真实 Worker 执行前用 receiver cache 修复：

```text
feature.data = receiver_cache.get_and_update_item(feature.data, cache_key)
```

位置：`cache.py:605` 到 `cache.py:608`

如果到了 `_batch_mm_inputs_from_scheduler()` 仍然是 `data is None`，ModelRunner 会跳过该 item：

位置：`gpu_model_runner.py:2879` 到 `gpu_model_runner.py:2881`

正常情况下，被调度执行 encoder 的 feature 应该能在 receiver cache 后拿到真实 processed kwargs；否则后续如果需要 embedding，会在 `_gather_mm_embeddings()` 触发 encoder cache miss。

---

## 16. 与 chunked prefill 的关系

多模态 feature 可能跨多个 scheduling step 被消费。

例如：

```text
image placeholder length = 576
max_num_batched_tokens 较小
本轮只 prefill 到 image placeholder 的前 128 个 token
```

这时：

```text
encoder output 应该只算一次；
_gather_mm_embeddings() 每轮按 token window 从同一个 encoder output 中切片。
```

关键方法：

```text
get_mm_features_in_window(mm_features, start, end)：
  找出当前 token window 覆盖哪些多模态 feature。

PlaceholderRange.get_embeds_indices_in_range(start_idx, end_idx)：
  把 placeholder token window 映射到 encoder output embedding slice。
```

对应源码：

```text
inputs.py:158 到 inputs.py:177
gpu_model_runner.py:3123 到 gpu_model_runner.py:3158
```

这也是为什么 encoder cache 容量单位是“encoder embeddings 数量”，而不是“placeholder token 总数”。`EncoderCacheManager` 注释也说明：它按 multimodal embeddings 管理，不把多模态 embedding 之间的 break/text tokens 算进 cache size。

位置：`encoder_cache_manager.py:41` 到 `encoder_cache_manager.py:45`

---

## 17. 和 EC connector 的关系

文档主线可以先把 EC connector 看成 encoder cache 的外部传输扩展。

ModelRunner 在多模态 preprocess 中包了一层：

```python
with self.maybe_get_ec_connector_output(
    scheduler_output,
    encoder_cache=self.encoder_cache,
) as ec_connector_output:
    self._execute_mm_encoder(scheduler_output)
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
```

位置：`gpu_model_runner.py:3447` 到 `gpu_model_runner.py:3454`

含义是：

```text
本地 GPUModelRunner.encoder_cache 仍是 _gather_mm_embeddings() 的直接读取对象；
EC connector 可以在上下文中加载 / 保存 encoder cache，辅助跨 worker / disaggregated 场景复用；
但它不改变 MultiModalFeatureSpec 的基本语义。
```

---

## 18. 和 encoder-decoder 模型的差异

Scheduler 初始化时，如果是 encoder-decoder 模型，会用 `EncoderDecoderCacheManager`：

```python
self.encoder_cache_manager = (
    EncoderDecoderCacheManager(cache_size=encoder_cache_size)
    if self.is_encoder_decoder
    else EncoderCacheManager(cache_size=encoder_cache_size)
)
```

位置：`scheduler.py:217` 到 `scheduler.py:225`

`EncoderDecoderCacheManager` 是临时实现：

```text
不真正复用 encoder output；
更多用于 scheduling / capacity 管理；
check_and_update_cache() 永远返回 False。
```

位置：`encoder_cache_manager.py:319` 到 `encoder_cache_manager.py:382`

ModelRunner 侧 encoder-decoder 路径也不同：

```python
if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
    encoder_outputs = self._execute_mm_encoder(scheduler_output)
    model_kwargs.update({"encoder_outputs": encoder_outputs})
```

位置：`gpu_model_runner.py:3552` 到 `gpu_model_runner.py:3559`

它不是把多模态 embedding 拼回 decoder token embedding，而是把 `encoder_outputs` 作为模型 kwargs 传给 decoder。

---

## 19. 常见误区

### 19.1 MultiModalFeatureSpec 是 cache 吗？

不是。

它只是一个 feature 描述对象，包含：

```text
processed data / modality / cache keys / placeholder position。
```

缓存命中、淘汰、容量统计由其他对象负责。

### 19.2 MultiModalCache 缓存的是 encoder output 吗？

不是。

`MultiModalCache` 及其 processor sender / receiver cache 缓存的是 processor 输出，也就是模型 encoder 执行前的输入 kwargs。

encoder output 在：

```text
GPUModelRunner.encoder_cache
```

由：

```text
EncoderCacheManager
```

在 Scheduler 侧做账本管理。

### 19.3 processor cache 命中后还会跑 encoder 吗？

可能会。

processor cache 命中只避免重复 processor / IPC。如果 `EncoderCacheManager` 未命中，Scheduler 仍会把 input_id 放进 `scheduled_encoder_inputs`，Worker 仍会执行 `_execute_mm_encoder()`。

### 19.4 encoder cache 命中后还需要 processor data 吗？

通常不需要重新跑 encoder，因此本轮不会把该 input_id 放进 `scheduled_encoder_inputs`。

但 request 的 `mm_features` 和 `mm_position` 仍然需要存在，因为 `_gather_mm_embeddings()` 要靠它们判断当前 token window 应该从 encoder output 里取哪一段。

### 19.5 free_encoder_mm_hashes 是不是请求结束就产生？

不一定。

请求结束通常只是释放引用，让 entry 进入 `freeable`。只有后续空间不足、`can_allocate()` 淘汰 freeable entry 时，才会产生 `freed`，进而出现在 `SchedulerOutput.free_encoder_mm_hashes`。

### 19.6 identifier 和 mm_hash 可以混用吗？

不能随便混用。

当前代码的边界是：

```text
processor cache：feature.mm_hash or feature.identifier
encoder cache：feature.identifier
```

文档或调试日志里变量名可能都叫 `mm_hash`，但要看实际值来自哪里。

---

## 20. 最小心智模型

把一次多模态请求拆成三层：

```text
第一层：processor 层
  raw image/audio/video → processed kwargs
  key = mm_hash
  cache = MultiModalProcessorOnlyCache / SenderCache / ReceiverCache / Shm cache

第二层：encoder 层
  processed kwargs → encoder output embeddings
  key = identifier
  scheduler ledger = EncoderCacheManager
  gpu tensor cache = GPUModelRunner.encoder_cache

第三层：decoder 层
  token embeddings / multimodal embeddings → language model forward
  cache = decoder KV cache
```

如果只记住一句话：

```text
MultiModalFeatureSpec 把一个多模态 item 的 data、位置和两个 cache key 串起来；processor cache 用它避免重复处理和传输，encoder cache 用它决定是否重复执行多模态 encoder，decoder KV cache 则是另一套 attention 历史缓存。
```
