# 04. Worker / Model Runner / Attention Backend 边界

## 1. 边界总览

Scheduler 的输出不是 tensor，而是“执行计划”。Worker/model runner 负责把执行计划转成真实 GPU 上可运行的输入。

```text
SchedulerOutput
  -> GPUModelRunner.execute_model()
      -> finish_requests()
      -> add_requests()
      -> update_requests()
      -> block_tables.apply_staged_writes()
      -> prepare_inputs()
      -> prepare_attn()
      -> model forward
      -> sample / pool
      -> ModelRunnerOutput
```

关键文件：

- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/block_table.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py`
- `D:/lzy/project/kv_pool/code/vllm/vllm/v1/core/sched/output.py`

## 2. Scheduler 与 Worker 的职责划分

### 2.1 Scheduler 负责

- 请求是否进入 running。
- 每个请求本轮执行多少 token。
- 使用哪些 KV block ids。
- 哪些请求完成或被抢占。
- 哪些 block 是新分配的，需要清零。
- KV connector 本轮 load/save metadata。
- encoder input 是否需要计算。

### 2.2 Worker / GPUModelRunner 负责

- 缓存 request state。
- 维护 worker 侧 req index。
- 维护 block table。
- 将 block ids 转成 attention backend 需要的 block table tensor 和 slot mapping。
- 准备 input ids、positions、seq lens、query start loc。
- 执行模型 forward。
- 执行 sampler 或 pooling。
- 返回 sampled tokens、logprobs、pooler output、KV connector output。

### 2.3 Attention backend 负责

- 使用 KV cache tensor、block tables、slot mappings 做高性能 attention。
- 不关心请求队列、priority、preemption。
- 不决定 block 分配。

## 3. GPUModelRunner 核心方法

`GPUModelRunner` 定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:120`。

关键方法：

- `get_kv_cache_spec()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:403`
- `initialize_kv_cache()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:406`
- `profile_run()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:653`
- `add_requests()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:771`
- `update_requests()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:818`
- `prepare_inputs()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:845`
- `prepare_attn()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:1011`
- `sample()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:1038`
- `execute_model()`：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:1101`

## 4. Worker 接收 SchedulerOutput

`GPUModelRunner.execute_model()` 非 dummy 路径起始逻辑位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:1110`。

顺序：

1. `update_pp_decode_requests()`
2. `finish_requests(scheduler_output)`
3. `free_states(scheduler_output)`
4. `add_requests(scheduler_output)`
5. `update_requests(scheduler_output)`
6. `block_tables.apply_staged_writes()`

这个顺序很重要：

- 先删除 finished/preempted 请求，释放 req index。
- 再添加新请求，避免 req index 冲突。
- 再更新已有请求新增 block。
- 最后一次性 apply staged writes，减少 GPU 写入开销。

## 5. 新请求：add_requests()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:771`

对每个 `scheduled_new_reqs`：

1. 如果是 streaming input update，先 `_remove_request(req_id)` 清理旧 state。
2. 计算 `prompt_len`。
3. `req_states.add_request(...)` 创建 worker 侧 request state。
4. 拿到 `req_index`。
5. encoder cache 记录多模态 feature。
6. `model_state.add_request(req_index, new_req_data)`。
7. `block_tables.append_block_ids(req_index, new_req_data.block_ids, overwrite=True)`。
8. `lora_state.add_request(...)`。
9. 如果是最后一个 pipeline rank，初始化 sampler 和 prompt logprobs worker。

注意：Scheduler 给的是 `block_ids`，worker 侧要把它写入 block table。

## 6. 已有请求：update_requests()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:818`

对 `scheduled_cached_reqs`：

1. 根据 `req_id` 找到 `req_index`。
2. 更新 `req_states.num_computed_tokens_np[req_index]`。
3. 如果有 `req_new_block_ids`，调用：

```text
block_tables.append_block_ids(req_index, req_new_block_ids, overwrite=False)
```

4. 更新 prefill token 统计。
5. 如果 `scheduler_output.new_block_ids_to_zero` 非空，调用 `kv_block_zeroer.zero_block_ids(...)` 清零。

清零的目的：避免新分配的 cache block 中残留 NaN 或旧数据污染 attention / SSM 计算。

## 7. BlockTables

`BlockTables` 定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/block_table.py:17`。

它是 worker 侧把 scheduler 分配 block id 转成 attention backend 输入的核心结构。

### 7.1 初始化字段

- `block_sizes`
- `kernel_block_sizes`
- `max_num_reqs`
- `max_num_batched_tokens`
- `max_num_blocks_per_group`
- `num_kv_cache_groups`
- `blocks_per_kv_block`
- `block_tables`
- `num_blocks`
- `input_block_tables`
- `slot_mappings`

### 7.2 block_tables

`block_tables` 是 per KV cache group 的 staged write tensor。

形状：

```text
num_kv_cache_groups x [max_num_reqs, max_num_blocks]
```

每一行对应一个 worker 侧 req index，内容是该请求使用的 block ids。

### 7.3 input_block_tables

`input_block_tables` 是 model forward 实际使用的 block table tensor。

`gather_block_tables()` 会根据当前 batch 的 `idx_mapping` 从全局 block table 中收集本轮请求所需的行。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/block_table.py:134`。

### 7.4 slot_mappings

`slot_mappings` 形状：

```text
[num_kv_cache_groups, max_num_batched_tokens]
```

它把本轮 token 的 position 映射到具体 KV cache slot。

`compute_slot_mappings()` 位于 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/block_table.py:160`。

它调用 Triton kernel `_compute_slot_mappings_kernel`，根据：

- request index mapping
- query start loc
- positions
- block table
- block size
- context parallel rank/size

计算每个 token 对应的 slot id。

## 8. append_block_ids()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/block_table.py:107`

参数：

- `req_index`
- `new_block_ids`
- `overwrite`

语义：

- `overwrite=True`：新请求或恢复请求，重新写入 block ids。
- `overwrite=False`：已有请求，追加新 block ids。

如果 `blocks_per_kv_block > 1`，说明 scheduler KV block 和 kernel block 粒度不同，需要展开：

```text
block_ids = [b * bpk + k for b in block_ids for k in range(bpk)]
```

## 9. prepare_inputs()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:845`

它把 `SchedulerOutput` 转成 `InputBatch`。

关键步骤：

1. 读取 `total_num_scheduled_tokens`。
2. 从 `num_scheduled_tokens` 获取本轮 request 集合。
3. 按每个请求 scheduled token 数排序：decode 通常在 prefill 前。
4. 构建 `idx_mapping`：batch index -> worker req index。
5. 处理 spec decode draft token 数量。
6. 构建 `query_start_loc`。
7. 判断哪些请求在 prefill。
8. 对 prefill 请求准备 prompt input ids。
9. 准备 positions 和 seq_lens。
10. 处理 context parallel local seq lens。
11. 合并 last sampled tokens 和 draft tokens。
12. 构建 `InputBatch` 返回。

### 9.1 Decode 优先于 Prefill

源码中注释：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:854`

```text
Decode first, then prefill.
```

这是通过对 `num_scheduled_tokens` 排序完成的。通常 decode 每个请求 token 数小，prefill chunk token 数大。

### 9.2 prefill 判断

`is_prefilling_np` 由：

```text
num_computed_prefill_tokens_np < prefill_len_np
```

得到，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:914`。

如果存在 prefill 请求，调用 `prepare_prefill_inputs()`。

### 9.3 positions 与 seq_lens

调用 `prepare_pos_seq_lens()`，位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:929`。

这些数据会传给 attention metadata builder。

## 10. prepare_attn()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:1011`

它做两件事：

1. `block_tables.gather_block_tables(...)`
2. `block_tables.compute_slot_mappings(...)`

返回：

```text
(block_tables, slot_mappings)
```

其中：

- block tables：告诉 attention 每个请求有哪些历史 KV blocks。
- slot mappings：告诉 attention 本轮输入 token 的 KV 要写到哪个 slot。

## 11. KV cache tensor 初始化

相关文件：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/attn_utils.py`

### 11.1 get_kv_cache_spec()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/attn_utils.py:39`

它扫描模型 attention layers，调用每层的 `get_kv_cache_spec(vllm_config)`，收集 layer -> KVCacheSpec。

如果某层使用 KV sharing，则跳过该层，后续绑定到目标层。

### 11.2 init_attn_backend()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/attn_utils.py:62`

分三阶段：

1. 按 KV cache group 发现 attention groups。
2. 为每个 KV cache group 选择所有 backend 都支持的 kernel block size。
3. 创建 metadata builders，并判断 cudagraph support。

### 11.3 _allocate_kv_cache()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/attn_utils.py:154`

为 `kv_cache_config.kv_cache_tensors` 创建 raw tensor：

```text
torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
```

然后按 layer name 建立 raw tensor 映射。

### 11.4 _reshape_kv_cache()

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/attn_utils.py:173`

它把 raw int8 tensor reshape 成 backend 需要的 KV cache shape。

关键点：

- 对 attention spec，使用 backend 的 `get_kv_cache_shape()`。
- 考虑 `storage_block_size`、`kernel_block_size`、`page_size_bytes`、padding、stride order。
- 对 Mamba spec，构建 state tensor。
- 处理 shared KV cache layers。

## 12. Attention Backend 的输入契约

Attention backend 不接触 scheduler 的 request queue。

它通常需要：

- KV cache tensors。
- block tables。
- slot mappings。
- query start loc。
- sequence lengths。
- positions。
- common prefix blocks。
- backend-specific metadata。

worker/model runner 在 forward 前构造这些数据。scheduler 只负责给 block ids 和 token 数。

## 13. KV Connector 在 worker 侧的位置

文件：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py`

`KVConnector` 接口定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py:29`。

方法：

- `pre_forward(scheduler_output)`
- `post_forward(finished_req_ids, wait_for_save=True)`
- `no_forward(scheduler_output)`
- `set_disabled(disabled)`

`ActiveKVConnector` 定义在 `D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/kv_connector.py:47`。

它会：

1. 注册 worker 侧 KV cache tensors。
2. 设置 host transfer buffer copy op。
3. 在 pre-forward 处理 preemption metadata、绑定 connector metadata、开始加载 KV。
4. 在 post-forward 等待保存完成、获取 finished sending/receiving、invalid blocks、stats、events。

## 14. execute_model() 中的无 forward 路径

如果 `scheduler_output.total_num_scheduled_tokens == 0`，worker 不执行模型 forward。

位置：`D:/lzy/project/kv_pool/code/vllm/vllm/v1/worker/gpu/model_runner.py:1118`。

这时会调用：

```text
self.kv_connector.no_forward(scheduler_output)
```

用于 remote KV load 等不需要本地 forward 的场景。

## 15. ModelRunnerOutput 返回给 Scheduler

`ModelRunnerOutput` 包含：

- sampled token ids
- logprobs
- prompt logprobs
- pooler output
- req id 到 batch index 映射
- KV connector output
- cudagraph stats
- routed experts 信息

Scheduler 用它在 `update_from_output()` 中推进请求状态。

## 16. 一句话总结

Scheduler 产出“请求 + token 数 + block ids”的计划；GPUModelRunner 把计划转成 block table、slot mapping、input batch 和 attention metadata；attention backend 只按这些 tensor 执行计算，不参与调度决策。
