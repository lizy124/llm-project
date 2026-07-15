# 09. Worker / ModelRunner 如何使用 KV Cache？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/core/kv_cache_manager.py`
- `code/vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `code/vllm/vllm/v1/core/block_pool.py`

本问题关注：Scheduler 已经把请求调度好并分配了 KV block 后，Worker / ModelRunner 如何真正使用这些 KV cache；block table 如何映射到请求；slot mapping 如何写入 attention kernel；prefix cache / external KV / lookahead / encoder cache 对 KV 使用有什么影响；KV connector 如何把远端 KV 的 load/save、finished_recving、finished_sending 穿过执行链路。

---

## 1. 一句话回答

Scheduler 决定 **要不要给请求分配 KV block**，Worker / ModelRunner 决定 **这些 block 在本轮 forward 里怎么被使用**。

核心关系可以概括成：

```text
Scheduler：
  分配 block，决定哪些请求本轮能跑。

Worker / ModelRunner：
  维护 block table，构造 slot mapping，
  把 KV block 交给 attention backend，
  并在 KV connector / prefix cache / external KV / lookahead 场景下处理特殊状态。
```

Worker 侧实际关心的是：

```text
1. 请求的 block_ids 是什么；
2. 当前 batch 里的每个请求对应哪个 block table row；
3. 每个 token 应该写入 KV cache 的哪个 slot；
4. 新 block 是否需要 zero；
5. CoW block copy 是否需要在 forward 前应用；
6. prefix cache / external KV / lookahead 该如何影响本轮执行；
7. KV connector 的 load / save / finished_recving / finished_sending / invalid_block_ids 如何贯穿执行。
```

---

## 2. KV cache 在 Worker 侧的总体结构

Worker / ModelRunner 侧的 KV cache 使用大致分成三层：

```text
1. 底层 KV cache tensor
   → initialize_kv_cache() 分配的 GPU cache 物理内存。

2. InputBatch.block_table
   → 把 request index 映射到 block ids。

3. slot mapping / attention metadata
   → 把每个 token 映射到 KV cache 的具体写入 slot，供 attention backend 使用。
```

对应主链路：

```text
Scheduler 选出请求
  → 分配 block ids
  → Worker 把 block ids 写入 InputBatch.block_table
  → _prepare_inputs() 计算 slot mapping
  → _build_attention_metadata() 组装 attention metadata
  → forward 时 attention backend 使用 block table + slot mapping
```

---

## 3. KV cache 初始化发生在哪里

### 3.1 Worker 先加载模型，再初始化 KV cache

在 `GPUWorker.initialize_from_config()` 中，KV cache 的初始化由：

```python
self.model_runner.initialize_kv_cache(kv_cache_config)
```

完成。

位置：`gpu_worker.py:717` 到 `gpu_worker.py:743`

这里的顺序是：

```text
1. 更新 cache_config.num_gpu_blocks；
2. 初始化 kv transfer；
3. 进入 memory pool；
4. 调用 model_runner.initialize_kv_cache()；
5. 初始化 routed experts / KV zero metadata 等辅助结构。

`GPUModelRunner.initialize_kv_cache()` 还会初始化 attention backend、metadata builders、按真实 KV cache group 重新初始化 InputBatch、绑定/注册 KV cache tensors；如果存在 KV transfer group，会注册统一 cross-layer KV cache 或普通 kv_caches，并设置 host transfer buffer ops。

位置：`gpu_model_runner.py:7467` 到 `gpu_model_runner.py:7524`
```

### 3.2 KV zero metadata

为了避免新分配 block 里有脏数据，Worker 还会在外层预先构造：

```python
self._init_kv_zero_meta()
```

位置：`gpu_model_runner.py:1124` 到 `gpu_model_runner.py:1138`

之后真正需要 zero 的 block id 会通过：

```python
self._zero_block_ids(block_ids)
```

位置：`gpu_model_runner.py:1140` 到 `gpu_model_runner.py:1143`

结合 `GPUModelRunner._update_states()` 中：

```python
if scheduler_output.new_block_ids_to_zero:
    self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
if scheduler_output.kv_cache_block_copies:
    copy_kv_cache_blocks_inplace(...)
```

位置：`gpu_model_runner.py:1188` 到 `gpu_model_runner.py:1197`

可以理解为：

```text
Scheduler 分配了新 block；
Worker 在真正使用前清零这些 block；
如果存在 partial-hit / CoW block copy，再在 forward 前执行 KV block copy。
```

---

## 4. KV block 如何映射到请求

### 4.1 请求侧保存 block_ids

在 Worker 侧，每个请求缓存状态 `CachedRequestState` 都保存：

```python
block_ids: tuple[list[int], ...]
```

位置：`gpu_input_batch.py:41` 到 `gpu_input_batch.py:44`

这表示：

```text
一个请求可能对应多个 KV cache group 的 block ids。
```

### 4.2 InputBatch.block_table

`InputBatch` 中的 block table 是核心映射：

```python
self.block_table = MultiGroupBlockTable(...)
```

位置：`gpu_input_batch.py:170` 到 `gpu_input_batch.py:181`

它的作用是：

```text
把 request index → block ids 的关系组织成 attention backend 可用的 block table。
```

### 4.3 add_request 时写入 block table

当请求被加入 InputBatch 时：

```python
self.block_table.add_row(request.block_ids, req_index)
```

位置：`gpu_input_batch.py:378`

这一步非常关键，意味着：

```text
请求不是直接拿着“block ids 列表”去 forward，
而是先进入 batch 的 block table，再由 attention metadata 读取。
```

### 4.4 remove_request / condense 时同步移动

删除请求时：

```python
self.block_table.clear_row(req_index)
```

位置：`gpu_input_batch.py:528`

紧缩 batch 时：

```python
self.block_table.move_row(last_req_index, empty_index)
```

位置：`gpu_input_batch.py:757`

这说明 block table 是 batch 紧凑化过程的一部分，不是独立静态表。

---

## 5. slot mapping 是什么

`slot mapping` 是 attention kernel 真正使用的“token → KV cache slot”映射。

### 5.1 compute_slot_mapping 的入口

在 `_prepare_inputs()` 中：

```python
self.input_batch.block_table.compute_slot_mapping(
    num_reqs,
    self.query_start_loc.gpu[: num_reqs + 1],
    self.positions[:total_num_scheduled_tokens],
)
```

位置：`gpu_model_runner.py:2167` 到 `gpu_model_runner.py:2171`

这一步把：

```text
block table + 每个 token 的 position
```

转成：

```text
slot mapping
```

供 attention backend 使用。

### 5.2 slot mapping 在 attention metadata 里被使用

随后 `_build_attention_metadata()` 会把 slot mapping 填入 CommonAttentionMetadata：

```python
cm_base = CommonAttentionMetadata(
    ...
    block_table_tensor=block_table_gid_0,
    slot_mapping=slot_mapping_gid_0,
    ...
)
```

位置：`gpu_model_runner.py:2394` 到 `gpu_model_runner.py:2412`

也就是说：

```text
block table 是请求级结构；
slot mapping 是 token 级结构；
attention backend 最终消费两者。
```

---

## 6. 为什么 attention backend 需要 block table + slot mapping

attention backend 需要知道：

```text
1. 当前 token 的 KV 应写到哪里；
2. 当前 query 应读取哪些历史 KV blocks；
3. prefix / cascade / sliding window 应该如何解释这些 blocks。
```

因此 `CommonAttentionMetadata` 包含：

```text
query_start_loc
seq_lens
_num_computed_tokens_cpu
block_table_tensor
slot_mapping
positions
is_prefilling
rswa_prefix_lens
```

位置：`gpu_model_runner.py:2394` 到 `gpu_model_runner.py:2412`

attention backend 不是自己推断 block 关系，而是直接消费这份 metadata。

---

## 7. Prefix cache 在 Worker 侧怎么体现

Prefix cache 的结果并不直接等于“本轮少算多少 token”，而是先体现在 Scheduler 里：

```text
prefix cache 命中 → num_computed_tokens 增加 → allocate_slots 时复用已有 blocks。
```

Worker 侧主要消费的是已经确定的 block ids 和 computed token 数。

### 7.1 请求状态里保存 num_computed_tokens

在 `CachedRequestState` 中：

```python
num_computed_tokens: int
```

位置：`gpu_input_batch.py:41` 到 `gpu_input_batch.py:44`

### 7.2 _prepare_inputs 里用它计算 positions

```python
positions_np = (
    self.input_batch.num_computed_tokens_cpu[req_indices]
    + self.query_pos.np[: cu_num_tokens[-1]]
)
```

位置：`gpu_model_runner.py:1961` 到 `gpu_model_runner.py:1965`

因此 prefix cache 的影响最终表现为：

```text
当前 token 位置从 num_computed_tokens 继续往后排；
已经复用的历史 KV 不需要重新写；
block table 里已有的 block 继续使用。
```

### 7.3 partial prompt / chunked prefill 也依赖它

如果只是部分 prompt 被本轮计算，`num_computed_tokens` 和 `num_prompt_tokens` 的关系会决定：

```text
当前请求处于 prefill 还是 decode；
query positions 和 slot mapping 怎么算；
哪些 token 需要 sample。
```

---

## 8. external KV / KV connector 如何影响 Worker

### 8.1 KV connector mixin 的职责

`KVConnectorModelRunnerMixin` 把 KV transfer 的生命周期封装到执行链路里。

核心接口：

```text
kv_connector_no_forward()
maybe_get_kv_connector_output()
finalize_kv_connector()
_get_kv_connector_output()
```

位置：`kv_connector_model_runner_mixin.py:34` 起

### 8.2 forward 前绑定 metadata

```python
kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
kv_connector.start_load_kv(get_forward_context())
```

位置：`kv_connector_model_runner_mixin.py:85` 到 `kv_connector_model_runner_mixin.py:96`

这说明：

```text
Scheduler 已经决定 KV transfer 相关元信息；
Worker 在 forward 前把这份 metadata 绑定到连接器。
```

### 8.3 forward 后收尾

```python
output.finished_sending, output.finished_recving = (
    kv_connector.get_finished(scheduler_output.finished_req_ids)
)
output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
output.kv_connector_stats = kv_connector.get_kv_connector_stats()
output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()
```

位置：`kv_connector_model_runner_mixin.py:99` 到 `kv_connector_model_runner_mixin.py:112`

所以 KV connector 的输出最终会被塞进 `ModelRunnerOutput.kv_connector_output`，然后传回 Scheduler。`KVConnectorOutput` 还带有 `expected_finished_count`，用于多 worker 聚合 finished sending / recving 通知。

位置：`outputs.py:195` 到 `outputs.py:221`

---

## 9. 什么是 kv_connector_no_forward

当本轮没有可执行 token，但仍然需要 KV transfer 维护时，会走：

```python
kv_connector_no_forward(scheduler_output, vllm_config)
```

位置：`kv_connector_model_runner_mixin.py:36` 到 `kv_connector_model_runner_mixin.py:48`

这表示：

```text
即使没有 forward，也要给 KV connector 一个执行机会：
绑定 metadata、启动异步 load，并回收 finished_sending / finished_recving / invalid_block_ids 等输出。
```

注意这里传入的是 `wait_for_save=False`，所以它不是完整的 forward 后 save 等待路径。

这在以下场景很常见：

```text
- 只有 KV transfer 在推进；
- 0-token step；
- 某些请求本轮只需要收尾，不需要模型计算。
```

---

## 10. KV connector 和 execute_model 的关系

在 `GPUModelRunner.execute_model()` 里：

```python
with self.maybe_get_kv_connector_output(
    scheduler_output,
    defer_finalize=defer_kv_connector_finalize,
) as kv_connector_output:
    model_output = self._model_forward(...)
```

位置：`gpu_model_runner.py:4375` 到 `gpu_model_runner.py:4378`

这说明 KV connector 是包在 forward 周围的上下文，不是 forward 后单独补一次。

### 10.1 defer finalize

如果启用了 speculative decoding，会设置：

```python
defer_kv_connector_finalize = self.speculative_config is not None
```

位置：`gpu_model_runner.py:4351` 到 `gpu_model_runner.py:4354`

原因是：

```text
draft model 可能也要用 KV connector；
所以 connector finalize 要延后到 draft 之后。
```

### 10.2 finalize_kv_connector()

只有 `defer_finalize=True` 的 spec decode 场景，sample 阶段才会在 draft model 之后调用：

```python
self.finalize_kv_connector()
```

位置：`gpu_model_runner.py:4683` 到 `gpu_model_runner.py:4687`

它会：

```text
wait_for_save()
clear_connector_metadata()
```

位置：`kv_connector_model_runner_mixin.py:64` 到 `kv_connector_model_runner_mixin.py:73`

非 spec decode 场景下，`_get_kv_connector_output()` 的 `finally` 分支会在退出 forward context 时完成 `wait_for_save()` 和 `clear_connector_metadata()`。

---

## 11. async load KV 的特殊路径

Scheduler 在 waiting 阶段如果发现外部 KV 命中并且可以异步 load，会令请求进入：

```text
WAITING_FOR_REMOTE_KVS
```

并且本轮可能 `num_new_tokens = 0`。

Worker 侧对应行为是：

```text
不一定 forward，但要保留 block / connector 状态；
等待 finished_recving 后再恢复请求。
```

对应 `SchedulerOutput` 会携带：

```text
kv_connector_metadata
finished_req_ids
```

Worker / ModelRunner 通过 KV connector mixin 把这条链路串起来；`execute_model()` 在 0-token step 且存在 KV transfer group 时会调用 `kv_connector_no_forward()`，否则才返回空输出。

位置：`gpu_model_runner.py:4149` 到 `gpu_model_runner.py:4165`

Scheduler 侧在收到 worker 的 `finished_recving` 后，先把请求 ID 记入 `finished_recving_kv_req_ids`，后续 `_try_promote_blocked_waiting_request()` 才会调用 `_update_waiting_for_remote_kv()`，把请求从 `WAITING_FOR_REMOTE_KVS` 恢复到可调度状态。

位置：`scheduler.py:2488` 到 `scheduler.py:2541`、`scheduler.py:2559` 到 `scheduler.py:2586`

---

## 12. 新 block zero 为什么要放在 Worker 侧

`Scheduler` 只知道应该分配哪些 block，不负责真正把 GPU memory 清零。

Worker 侧：

```python
if scheduler_output.new_block_ids_to_zero:
    self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
```

位置：`gpu_model_runner.py:1153` 到 `gpu_model_runner.py:1156`

原因是：

```text
zero 是设备侧内存操作；
Scheduler 只做资源账本；
Worker 负责物理内存状态。
```

---

## 13. 为什么 slot mapping 要在 prepare_inputs 阶段算

`slot mapping` 与本轮 token 的位置、已计算 token 数、spec decode、chunked prefill 都有关，因此必须在 batch 已经稳定后再算。

它依赖：

```text
InputBatch.req_id_to_index
InputBatch.num_computed_tokens_cpu
InputBatch.block_table
query_start_loc
positions
```

位置：`gpu_model_runner.py:1930` 到 `gpu_model_runner.py:2171`

所以顺序是：

```text
_update_states()
  → InputBatch 就位
  → _prepare_inputs()
  → block_table.commit_block_table()
  → compute_slot_mapping()
  → _build_attention_metadata()
```

这也是为什么 KV cache 交互和 batch 维护要放在一起看。

---

## 14. attention metadata 中的 KV 相关字段

`CommonAttentionMetadata` 会携带很多和 KV 相关的信息：

```text
block_table_tensor
slot_mapping
seq_lens
_num_computed_tokens_cpu
is_prefilling
max_seq_len
positions
mm_req_doc_ranges
rswa_prefix_lens
```

位置：`gpu_model_runner.py:2394` 到 `gpu_model_runner.py:2412`

这些字段说明：

```text
attention backend 不只是看 query / key / value 张量，
还需要知道这个 batch 的 block 布局和位置布局。
```

---

## 15. KV cache 和 batch 执行的关系

Worker 侧 batch 执行不是简单“把 token 拼成一个大张量”，而是：

```text
1. 通过 block table 记录每个请求的 KV 布局；
2. 通过 slot mapping 告诉 attention kernel 本轮 token 写入哪些 slot；
3. 通过 seq_lens / num_computed_tokens 跟踪每个请求当前上下文长度；
4. 通过 _prepare_inputs() 和 _build_attention_metadata() 形成完整 forward 输入。
```

因此 KV cache 和 batch 执行是绑定在一起的：

```text
没有 block table，就无法稳定复用历史 KV；
没有 slot mapping，就无法把当前 token 写到正确位置；
没有 num_computed_tokens，就无法知道每个请求当前应该从哪里继续。
```

---

## 16. 与 Scheduler 的边界

### 16.1 Scheduler 负责

```text
- allocate_slots() 判断 block 是否够；
- prefix cache / external KV / lookahead / encoder budget 的调度决策；
- 抢占和恢复；
- 分配 new_block_ids_to_zero；
- 决定哪些请求进入 running / waiting。
```

### 16.2 Worker / ModelRunner 负责

```text
- 接收 scheduler 分配结果；
- 把 block ids 写进 InputBatch.block_table；
- 计算 slot mapping；
- 在 attention metadata 里使用这些映射；
- 处理 KV connector 的实际 load/save/finalize；
- 真正 zero 设备上的新 block。
```

### 16.3 一句话总结边界

```text
Scheduler 管“资源账本”，Worker 管“物理落地”。
```

---

## 17. 容易疑惑的点

### 17.1 block table 和 slot mapping 是一回事吗？

不是。

```text
block table：请求 → block ids
slot mapping：token → 具体 KV slot
```

### 17.2 prefix cache 是不是 Worker 自己发现的？

不是。

prefix cache 命中与可复用 block 的决定主要在 Scheduler / KVCacheManager；Worker 只消费结果。

### 17.3 外部 KV 命中是不是 forward 之后才处理？

不是。

KV connector 是包在 forward 上下文中的，load/save 和 finalize 都穿过执行链路。

### 17.4 0-token step 是否完全不碰 KV cache？

不是。

如果有 KV connector，0-token step 仍然可能需要 load/save/finalize 或 no_forward 处理。

### 17.5 新 block 为什么还要 zero？

为了防止残留数据污染 attention / SSM 计算，尤其是新分配显存并不会天然清零。

---

## 18. 总结

Worker / ModelRunner 使用 KV cache 的主链路可以压缩成：

```text
Scheduler 分配 block
  → Worker 把 block ids 写进 InputBatch.block_table
  → _prepare_inputs() 计算 slot mapping
  → _build_attention_metadata() 组装 block table / slot mapping
  → forward / attention backend 使用这些映射
  → KV connector 处理远端 KV load/save/finalize
```

如果只记住一句话：

```text
Scheduler 负责“分配 KV 资源”，Worker 负责“把 KV 资源变成模型 forward 能用的 block table 和 slot mapping”。
```
