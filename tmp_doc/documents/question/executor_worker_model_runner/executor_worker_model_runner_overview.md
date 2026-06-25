# vLLM V1 Executor / Worker / ModelRunner 逻辑梳理

源码位置：

- `code/vllm/vllm/v1/executor/abstract.py`
- `code/vllm/vllm/v1/executor/uniproc_executor.py`
- `code/vllm/vllm/v1/executor/multiproc_executor.py`
- `code/vllm/vllm/v1/executor/ray_executor.py`
- `code/vllm/vllm/v1/worker/worker_base.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/engine/core.py`

本文按“先定边界，再走主链路，再拆关键阶段，最后总结接口和数据结构”的方式，梳理 vLLM V1 执行层里 `Executor`、`Worker`、`ModelRunner` 三层的关系。

它和 `scheduler` 目录的总览文档一样，不是逐个 helper 函数的字典式说明，而是先回答几个最关键的问题：

```text
1. Executor / Worker / ModelRunner 各自负责什么？
2. 它们和 EngineCore、Scheduler 的关系是什么？
3. SchedulerOutput 如何进入执行层？
4. Worker / ModelRunner 如何把调度计划变成真正的模型 forward？
5. ModelRunnerOutput 如何回到 Scheduler 并闭环？
6. KV cache、attention metadata、sampling、LoRA、spec decode、multimodal 在哪一层接入？
```

---

## 0. 梳理规划

参考 `scheduler` 目录的写法，本文按“先定角色，再走主链路，再拆关键阶段”的顺序组织。

要回答的问题分成 8 组：

```text
1. Executor 是哪一层？负责什么，不负责什么？
2. Worker 是哪一层？为什么需要 WorkerBase / WorkerWrapperBase / GPU Worker？
3. ModelRunner 是哪一层？为什么它是执行层的核心？
4. SchedulerOutput 是怎样从 EngineCore 进入 Worker / ModelRunner 的？
5. execute_model() 到 sample_tokens() 的完整主线是什么？
6. Worker / ModelRunner 如何准备输入、attention metadata、KV cache、sampling 所需状态？
7. ModelRunnerOutput 如何返回 Scheduler，并触发状态更新与资源释放？
8. KV cache、LoRA、multimodal、spec decode、pipeline parallel、sleep / wake_up / profile 如何挂接？
```

阅读顺序建议：

```text
executor_worker_model_runner_overview.md
  → 01_executor_role.md
  → 02_worker_role.md
  → 03_model_runner_role.md
  → 04_execute_model_flow.md
  → 05_input_batch_and_state_update.md
  → 06_prepare_inputs_and_attention_metadata.md
  → 07_model_forward_and_logits.md
  → 08_sampling_and_model_runner_output.md
  → 09_worker_kv_cache_interaction.md
  → 10_executor_worker_lifecycle.md
```

如果只想先抓住一条主线，可以先读总览，再读 `01`、`02`、`03`，最后看 `04`。

---

## 1. 三层各自是什么

### 1.1 Executor

`Executor` 是 EngineCore 和 Worker 之间的执行分发层。

源码位置：`code/vllm/vllm/v1/executor/abstract.py:37`

它负责：

```text
- 根据配置选择执行后端；
- 创建并管理 Worker；
- 通过 collective RPC 分发控制命令；
- 把 SchedulerOutput 送到 Worker；
- 把 ModelRunnerOutput 收回来；
- 转发 profile / sleep / wake_up / LoRA / shutdown / health 等控制接口。
```

它不负责：

```text
- 调度决策；
- token budget / request state 管理；
- 模型 forward 的具体计算；
- 采样算法本身；
- KV block 的分配与抢占。
```

一句话记忆：

```text
Executor 负责“把这一轮交给谁跑”。
```

### 1.2 Worker

`Worker` 是执行层里的设备侧承载对象。

源码位置：

- `code/vllm/vllm/v1/worker/worker_base.py:39`
- `code/vllm/vllm/v1/worker/gpu_worker.py:117`

它负责：

```text
- 初始化 device、distributed environment、workspace；
- 创建并持有 ModelRunner；
- 加载模型；
- profile 可用显存；
- 初始化 KV cache；
- warmup / compile / CUDA graph capture；
- 接收并执行 SchedulerOutput；
- 转发 sample_tokens / profile / sleep / wake_up / shutdown / weight transfer / LoRA 等控制能力。
```

它不负责：

```text
- 决定本轮调度哪些请求；
- 决定 token budget；
- 直接处理 scheduler 队列；
- 最终构造对上层客户端可见的 RequestOutput。
```

一句话记忆：

```text
Worker 负责“把执行资源准备好，并把执行请求交给 ModelRunner”。
```

### 1.3 ModelRunner

`ModelRunner` 是 Worker 内部真正把 `SchedulerOutput` 变成模型执行的组件。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:418`

它负责：

```text
- 维护 worker 侧请求状态和 persistent batch；
- 准备 input_ids / inputs_embeds / positions / slot mapping / block table；
- 构造 attention metadata；
- 执行模型 forward；
- 计算 logits 或 pooling output；
- 运行 sampler；
- 处理 structured output / grammar；
- 处理 speculative decoding；
- 处理 KV cache、encoder cache、multimodal 输入、LoRA、Mamba、pipeline parallel 等执行细节；
- 产出 ModelRunnerOutput。
```

它不负责：

```text
- 接收用户请求；
- 决定 schedule；
- 执行跨 worker RPC；
- 做最终的 engine 层状态汇总。
```

一句话记忆：

```text
ModelRunner 负责“把一轮调度计划真正跑进模型”。
```

---

## 2. 一句话总览链路

最小主链路是：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
  → _model_forward()
  → logits / pooling / IntermediateTensors
  → sample_tokens()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
```

对应源码主入口：

- `code/vllm/vllm/v1/engine/core.py:479`
- `code/vllm/vllm/v1/executor/abstract.py:221`
- `code/vllm/vllm/v1/worker/worker_base.py:142`
- `code/vllm/vllm/v1/worker/gpu_worker.py:808`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`

如果把执行层再压缩一层，可以记成：

```text
SchedulerOutput
  → Executor 分发
  → Worker 承载
  → ModelRunner 计算
  → ModelRunnerOutput 回收
```

---

## 3. 这三层为什么要分开

这个分层不是纯粹的代码组织，而是为了把不同类型的职责拆开：

### 3.1 Executor 适合放“分发与编排”

Executor 需要统一处理：

```text
- 单进程 / 多进程 / Ray / external launcher；
- collect RPC；
- 控制命令广播；
- 非阻塞执行；
- health check；
- lifecycle / shutdown。
```

这些东西和模型结构关系不大，更接近运行时编排。

### 3.2 Worker 适合放“设备生命周期”

Worker 需要统一处理：

```text
- 初始化分布式环境；
- 绑定 device；
- 加载模型权重；
- profile 显存；
- 初始化 KV cache；
- warmup / compile；
- sleep / wake_up；
- 权重更新；
- 清理资源。
```

这些内容和具体 batch 的每一步 token 调度无关，但和设备状态强相关。

### 3.3 ModelRunner 适合放“一轮 batch 执行逻辑”

ModelRunner 要处理的是最重的一层：

```text
- 当前 batch 中有哪些 request；
- 每个 request 这一步该喂什么输入；
- 需要哪些 attention metadata；
- 要不要跑 encoder / multimodal；
- logits / pooling / sampler / grammar / spec decode 怎么接；
- KV cache / slot mapping / block table 怎么同步。
```

这就是为什么 `ModelRunner` 是执行层最复杂、也最值得单独拆文档的地方。

---

## 4. 从 SchedulerOutput 到执行层

Scheduler 生成的不是最终结果，而是执行计划 `SchedulerOutput`。

源码位置：`code/vllm/vllm/v1/core/sched/output.py:181`

它包含的核心信息包括：

```text
- scheduled_new_reqs
- scheduled_cached_reqs
- num_scheduled_tokens
- total_num_scheduled_tokens
- scheduled_spec_decode_tokens
- scheduled_encoder_inputs
- num_common_prefix_blocks
- finished_req_ids
- free_encoder_mm_hashes
- kv_connector_metadata
- ec_connector_metadata
- new_block_ids_to_zero
```

EngineCore 在每轮 `step()` 里会：

```text
1. 调度得到 SchedulerOutput；
2. 交给 Executor.execute_model(scheduler_output)；
3. 等待结果；
4. 必要时再调用 sample_tokens()；
5. 最后把 ModelRunnerOutput 交回 Scheduler.update_from_output()。
```

源码位置：`code/vllm/vllm/v1/engine/core.py:479` 到 `code/vllm/vllm/v1/engine/core.py:508`

这里的关键点是：

```text
SchedulerOutput 先进入执行层，执行层完成模型前向和采样，再把结果回传给 Scheduler 做状态闭环。
```

---

## 5. execute_model() 的主链路

### 5.1 EngineCore 的位置

`EngineCore.step()` 是整条链路的入口。

源码位置：`code/vllm/vllm/v1/engine/core.py:479`

核心步骤是：

```text
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```

### 5.2 Executor 的作用

Executor 把一次 `execute_model()` 变成对所有 Worker 的 RPC。

对于单进程执行，`UniProcExecutor.collective_rpc()` 会直接在 driver worker 上执行方法。

源码位置：`code/vllm/vllm/v1/executor/uniproc_executor.py:79`

对于多进程 / Ray 后端，则会把调用分发到多个 worker 进程。

### 5.3 Worker 的作用

Worker 收到 `scheduler_output` 后，不是直接 forward，而是先做设备侧和 batch 侧准备：

```text
- 更新输入 batch / request state；
- 清理 finished request；
- zero 新分配 block；
- 构造输入 tensor；
- 构造 attention metadata；
- 执行模型；
- 产生 logits / pooling / intermediate tensors；
- 在需要时延后到 sample_tokens() 才输出最终结果。
```

### 5.4 ModelRunner 的作用

`GPUModelRunner.execute_model()` 是这条链路里最关键的执行入口。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`

它负责把 `SchedulerOutput` 展开成：

```text
- persistent batch 更新
- input_ids / inputs_embeds
- positions
- slot mapping
- attention metadata
- hidden states
- logits / pooling output
- execute_model_state
```

如果 batch 需要异步采样或分阶段处理，它可能先返回 `None`，再由 `sample_tokens()` 完成后半段。

---

## 6. Worker 侧生命周期

WorkerBase 定义抽象接口，GPU Worker 实现真正逻辑。

### 6.1 WorkerBase

源码位置：`code/vllm/vllm/v1/worker/worker_base.py:39`

它定义了 Worker 必须具备的能力：

```text
get_kv_cache_spec
compile_or_warm_up_model
init_device
load_model
execute_model
sample_tokens
get_cache_block_size_bytes
add_lora / remove_lora / pin_lora / list_loras
shutdown
```

### 6.2 WorkerWrapperBase

WorkerWrapperBase 负责“先包装、后初始化”。

源码位置：`code/vllm/vllm/v1/worker/worker_base.py:187`

它的意义是：

```text
- 延迟创建真正 Worker；
- 在 worker 实例化前注入环境变量；
- 处理 RPC rank / global rank；
- 统一生命周期入口。
```

### 6.3 GPU Worker 的初始化顺序

`Worker.__init__()` 先缓存配置，再准备设备相关状态。

源码位置：`code/vllm/vllm/v1/worker/gpu_worker.py:117`

典型流程是：

```text
init_device()
  → init distributed env
  → set seed
  → snapshot memory
  → construct model_runner
load_model()
  → load actual model weights
  → setup wrappers / offloader / transfer engines
initialize_from_config()
  → allocate KV cache
  → warmup / compile
```

### 6.4 sleep / wake_up / profile / shutdown

Worker 还负责设备生命周期控制：

```text
- sleep(level)
- wake_up(tags)
- determine_available_memory()
- profile()
- shutdown()
```

这些接口由 Executor 统一转发。

---

## 7. ModelRunner 侧主职责

`GPUModelRunner` 是执行层的核心。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:418`

它初始化时会缓存很多状态：

```text
- model_config / cache_config / scheduler_config / parallel_config
- device / dtype / kv cache dtype
- sampler
- multimodal registry
- speculative config
- encoder cache
- input batch
- routed experts state
- async scheduling state
- cudagraph / ubatching / PP / TP 相关状态
```

这意味着它不是一个“轻包装类”，而是一个拥有完整 batch 生命周期状态机的执行器。

### 7.1 一次执行的核心对象

`ExecuteModelState` 是 `execute_model()` 和 `sample_tokens()` 之间的临时状态桥梁。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:402`

它保存：

```text
- scheduler_output
- logits
- spec_decode_metadata
- common attention metadata
- hidden_states
- sample_hidden_states
- aux_hidden_states
- ec_connector_output
- cudagraph_stats
- slot_mappings
```

也就是说：

```text
execute_model() 负责把中间态准备好；
sample_tokens() 负责消费这些中间态并生成最终采样结果。
```

---

## 8. ModelRunner 的主线执行阶段

### 8.1 `_update_states()`

`_update_states()` 把 SchedulerOutput 合入 worker 侧缓存状态和 persistent batch。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1127`

它会处理：

```text
- 删除 finished requests；
- 从 input_batch 中移除已结束或不在本轮调度中的请求；
- 处理新请求和恢复请求；
- 更新 block IDs；
- 更新 num_computed_tokens；
- 处理 speculative decoding 的乐观/修正逻辑；
- 处理 mamba / hybrid / async scheduling 的状态修正；
- 注册 pooling / late interaction / multimodal 相关状态。
```

这一步是后面所有输入准备的基础。

### 8.2 `_prepare_inputs()`

`_prepare_inputs()` 把 batch 状态变成真正送给模型的输入张量。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1889`

它会做的事情包括：

```text
- commit block table；
- 计算 req_indices / query_start_loc / positions；
- 处理 prompt embeds 和 token ids；
- 处理 M-RoPE / XD-RoPE；
- 更新 num_computed_tokens；
- 生成 discard_request_mask；
- 准备 slot mapping；
- 计算 logits_indices；
- 准备 spec decode metadata；
- 准备 async scheduling 所需的 CPU/GPU 缓冲区。
```

这一步是 ModelRunner 把“请求级状态”降维成“张量级输入”的关键。

### 8.3 `_build_attention_metadata()`

`_build_attention_metadata()` 把 request batch 和 KV 信息变成 attention backend 需要的元数据。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2208`

它会构造：

```text
- CommonAttentionMetadata
- 每个 KV cache group 的具体 metadata
- block table / slot mapping
- seq_lens / max_seq_len / is_prefilling
- cascade attention prefix lengths
- speculative decoding 所需的额外字段
```

这一层是“模型执行”真正开始前最重的准备工作之一。

### 8.4 `_preprocess()`

`_preprocess()` 负责把 multimodal / encoder / prompt embeds / PP 中间张量等统合成 forward 需要的输入。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3426`

它会决定：

```text
- input_ids 还是 inputs_embeds；
- 是否需要运行 multimodal encoder；
- 是否需要 encoder-decoder encoder_outputs；
- positions 用哪套；
- PP 下是否要同步 intermediate_tensors；
- model_kwargs 里要附带哪些额外元数据。
```

### 8.5 `_model_forward()`

`_model_forward()` 是真正调用模型 forward 的地方。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3757`

默认实现只是：

```python
return self.model(...)
```

它之所以单独抽出来，是为了让子类或特化 runner 可以只替换 forward 这一步，而不必重写整条执行链。

### 8.6 `_pool()` / logits / `sample_tokens()`

如果是 pooling 模型，会走 `_pool()`。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3342`

如果是生成模型，会在 `execute_model()` 阶段拿到 logits，然后在 `sample_tokens()` 阶段完成采样。

源码位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4423`

`sample_tokens()` 还会处理：

```text
- grammar bitmask；
- speculative decoding rejection / acceptance；
- async scheduling 下的 draft token 回写；
- logprobs / prompt logprobs；
- routed experts 数据；
- 输出 token 的最终整理。
```

---

## 9. ModelRunnerOutput 是什么

`ModelRunnerOutput` 是执行层交回 Scheduler 的结果容器。

源码位置：`code/vllm/vllm/v1/outputs.py:231`

它主要包含：

```text
- req_ids
- req_id_to_index
- sampled_token_ids
- logprobs
- prompt_logprobs_dict
- pooler_output
- kv_connector_output
- ec_connector_output
- num_nans_in_logits
- cudagraph_stats
- routed_experts
```

它有两个重要特点：

### 9.1 它是“跨进程可序列化”的

`ModelRunnerOutput` 需要从 worker 回传到 scheduler，所以很多字段尽量使用 list / dict / CPU 友好结构，而不是只放 GPU tensor。

### 9.2 它不是最终用户输出

它仍然只是执行层输出。

真正对上层客户端可见的结果，还要经过 Scheduler 的 `update_from_output()` 消化，变成 `EngineCoreOutputs`。

---

## 10. 执行层和 Scheduler 的闭环

执行层完成后，Scheduler 会回收结果并推进请求状态。

对应总入口：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`

闭环的核心是：

```text
- sampled token appended 到 request；
- stop condition 检查；
- spec decode 的接受 / 拒绝回退；
- grammar 状态推进；
- KV / EC connector 完成状态更新；
- 结束请求释放 block / encoder cache / connector metadata；
- 返回 EngineCoreOutputs。
```

所以整个链路实际上是：

```text
Scheduler 决定怎么跑
  → Executor / Worker / ModelRunner 执行
  → Scheduler 再把执行结果消化回状态机
```

---

## 11. KV cache / attention / sampling 在哪一层接入

### 11.1 KV cache

KV cache 的“分配决策”在 Scheduler，
但“输入准备和使用方式”在 ModelRunner。

- Scheduler 负责 block 分配、抢占、释放；
- ModelRunner 负责 block table / slot mapping / attention metadata；
- Worker 在初始化阶段负责真正创建 KV cache 张量。

### 11.2 attention metadata

attention metadata 主要由 ModelRunner 构造，因为它依赖：

```text
- 当前 batch 的请求顺序；
- seq_lens / positions；
- block table；
- slot mapping；
- padding / cudagraph / ubatching；
- cascade attention / speculative decoding。
```

### 11.3 sampling

采样逻辑也在 ModelRunner 侧完成，因为它需要直接消费 logits、sampling metadata、grammar 和 spec decode 状态。

### 11.4 multimodal / encoder

multimodal encoder、encoder-decoder encoder_outputs、prompt embeds 等都挂在 ModelRunner 的 preprocess 阶段。

### 11.5 LoRA / sleep / profile / shutdown

这些更像执行层运行控制能力：

```text
- LoRA：Executor 转发，Worker / ModelRunner 实际加载和维护；
- sleep / wake_up：Executor 暴露接口，Worker 实施；
- profile：Worker 执行；
- shutdown：Executor / Worker 共同收尾。
```

---

## 12. 关键数据结构关系

### 12.1 `SchedulerOutput`

这是“本轮该怎么跑”的执行计划。

### 12.2 `CachedRequestData`

缓存请求在 worker 侧的增量状态，减少每轮重复传输。

### 12.3 `InputBatch`

worker 侧的批处理总状态，负责把 request state 映射成模型前向输入。

### 12.4 `ExecuteModelState`

`execute_model()` 到 `sample_tokens()` 的临时桥梁。

### 12.5 `ModelRunnerOutput`

worker 返回 scheduler 的执行结果。

### 12.6 `EngineCoreOutputs`

Scheduler 消化 `ModelRunnerOutput` 后，返回给上层客户端的最终输出。

---

## 13. 推荐阅读路线

### 13.1 快速建立全局印象

```text
executor_worker_model_runner_overview.md
  → 01_executor_role.md
  → 02_worker_role.md
  → 03_model_runner_role.md
```

### 13.2 按执行链路完整阅读

```text
executor_worker_model_runner_overview.md
  → 04_execute_model_flow.md
  → 05_input_batch_and_state_update.md
  → 06_prepare_inputs_and_attention_metadata.md
  → 07_model_forward_and_logits.md
  → 08_sampling_and_model_runner_output.md
```

### 13.3 和 Scheduler 联动阅读

```text
../scheduler/03_running_decode_prefill.md
  → ../scheduler/08_update_after_worker_output.md
  → 04_execute_model_flow.md
  → 09_worker_kv_cache_interaction.md
```

### 13.4 深入生命周期与控制面

```text
01_executor_role.md
  → 02_worker_role.md
  → 10_executor_worker_lifecycle.md
```

---

## 14. 文档定位

```text
executor_worker_model_runner_overview.md：
  总览主文档，适合快速建立执行层全局图。

01-10：
  按问题拆开的专题文档，适合逐段精读执行层源码。
```

---

## 15. 一句话总结

`Executor` 负责分发，`Worker` 负责设备生命周期，`ModelRunner` 负责把 `SchedulerOutput` 变成一次真正的模型执行；它们共同完成从调度计划到 `ModelRunnerOutput` 的闭环。

最核心的主线是：

```text
SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → ModelRunner.execute_model()
  → forward / logits / pooling
  → sample_tokens()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
```
