# 06 executor_worker_model_runner 背诵文档

## 1. 专题定位

`executor_worker_model_runner` 讲的是 vLLM V1 的执行层。

它解释 Scheduler 生成的 `SchedulerOutput` 如何真正进入设备、变成模型 forward、logits、sampling，并最终返回 `ModelRunnerOutput`。

一句话：

```text
Executor 负责分发，Worker 负责设备生命周期，ModelRunner 负责把 SchedulerOutput 变成真正的模型执行。
```

## 2. 最小心智模型

主链路是：

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
  → logits / pooling
  → sample_tokens()
  → ModelRunnerOutput
  → Scheduler.update_from_output()
```

要背住：

```text
SchedulerOutput 是计划；Executor 分发计划；Worker 承载计划；ModelRunner 执行计划；ModelRunnerOutput 回给 Scheduler 对账。
```

## 3. 三层各自是什么

### Executor

Executor 是 EngineCore 和 Worker 之间的执行分发层。

它负责：

```text
选择执行后端
创建和管理 Worker
通过 collective RPC 分发命令
把 SchedulerOutput 送到 Worker
把 ModelRunnerOutput 收回来
转发 profile / sleep / wake_up / LoRA / shutdown / health 等控制接口
```

它不负责：

```text
调度决策
token budget
模型 forward 细节
采样算法
KV block 分配和抢占
```

一句话：

```text
Executor 负责“把这一轮交给哪些 Worker 跑”。
```

### Worker

Worker 是设备侧承载对象。

它负责：

```text
初始化 device
初始化 distributed environment
创建 ModelRunner
加载模型
profile 可用显存
初始化 KV cache
warmup / compile / CUDA graph capture
接收并执行 SchedulerOutput
转发 sample_tokens / profile / sleep / wake_up / shutdown / LoRA 等能力
```

它不负责：

```text
决定本轮调度哪些请求
决定 token budget
维护 Scheduler 队列
构造最终 RequestOutput
```

一句话：

```text
Worker 负责“把设备和模型运行环境准备好”。
```

### ModelRunner

ModelRunner 是 Worker 内部真正执行一轮 batch 的核心。

它负责：

```text
维护 worker 侧请求状态和 persistent batch
准备 input_ids / inputs_embeds / positions
准备 slot mapping / block table
构造 attention metadata
执行模型 forward
计算 logits 或 pooling output
运行 sampler
处理 grammar / structured output
处理 spec decode
处理 KV cache / encoder cache / multimodal / LoRA / Mamba / PP
产出 ModelRunnerOutput
```

一句话：

```text
ModelRunner 负责“把一轮调度计划真正跑进模型”。
```

## 4. 为什么要分三层

### Executor 适合分发

因为执行后端可能是：

```text
单进程
多进程
Ray
external launcher
自定义 executor
```

Executor 统一处理：

```text
collective RPC
控制命令广播
非阻塞执行
health check
shutdown
```

### Worker 适合设备生命周期

Worker 统一处理：

```text
device 绑定
分布式初始化
权重加载
显存 profile
KV cache 初始化
warmup / compile
sleep / wake_up
资源清理
```

### ModelRunner 适合 batch 执行逻辑

ModelRunner 处理：

```text
当前 batch 有哪些请求
每个请求本轮喂哪些 token
需要哪些 attention metadata
是否要跑 encoder / multimodal
如何做 logits / pooling / sampling
KV cache / block table / slot mapping 如何同步
```

一句话：

```text
Executor 解决分发问题，Worker 解决设备生命周期问题，ModelRunner 解决一轮 batch 怎么执行的问题。
```

## 5. SchedulerOutput 如何进入执行层

EngineCore 每轮：

```text
scheduler_output = scheduler.schedule()
future = model_executor.execute_model(scheduler_output, non_block=True)
```

Executor 收到 `SchedulerOutput` 后：

```text
collective_rpc("execute_model", args=(scheduler_output,))
```

Worker 收到后：

```text
Worker.execute_model(scheduler_output)
  → model_runner.execute_model(scheduler_output, intermediate_tensors)
```

ModelRunner 再展开具体执行。

## 6. SchedulerOutput 包含什么

它是执行计划，包含：

```text
scheduled_new_reqs
scheduled_cached_reqs
num_scheduled_tokens
total_num_scheduled_tokens
scheduled_spec_decode_tokens
scheduled_encoder_inputs
num_common_prefix_blocks
finished_req_ids
free_encoder_mm_hashes
preempted_req_ids
kv_connector_metadata
ec_connector_metadata
new_block_ids_to_zero
has_structured_output_requests
pending_structured_output_tokens
```

ModelRunner 会用这些信息更新 worker 侧 batch 状态。

## 7. Worker 生命周期

GPU Worker 典型生命周期：

```text
init_device()
  → 设置 CUDA device
  → 初始化 distributed env
  → 设置随机种子
  → 记录显存快照
  → 构造 ModelRunner

load_model()
  → 选择 loader
  → 加载模型和权重
  → 包装 LoRA / drafter / offloader / compile wrapper

initialize_from_config()
  → 分配 KV cache

compile_or_warm_up_model()
  → warmup
  → torch.compile / vLLM compile
  → CUDA graph capture

execute_model()
  → 执行真实请求
```

要背住：

```text
Worker 先准备设备，再加载模型，再初始化 KV cache，再 warmup / compile，最后才处理请求。
```

## 8. WorkerBase 定义的能力

Worker 抽象接口包括：

```text
get_kv_cache_spec
init_device
load_model
initialize_from_config
compile_or_warm_up_model
execute_model
sample_tokens
get_cache_block_size_bytes
add_lora / remove_lora / pin_lora / list_loras
profile
sleep / wake_up
shutdown
```

这些接口通常由 Executor 调用。

## 9. ModelRunner 的核心状态

GPUModelRunner 初始化时会缓存：

```text
model_config
cache_config
scheduler_config
parallel_config
device / dtype / kv_cache_dtype
sampler
multimodal registry
speculative config
encoder cache
input batch
routed experts state
async scheduling state
cudagraph / ubatching / PP / TP 状态
```

这说明：

```text
ModelRunner 不是轻量 wrapper，而是 worker 侧 batch 生命周期状态机。
```

## 10. ExecuteModelState

`ExecuteModelState` 是 `execute_model()` 和 `sample_tokens()` 之间的桥。

它保存：

```text
scheduler_output
logits
spec_decode_metadata
common attention metadata
hidden_states
sample_hidden_states
aux_hidden_states
ec_connector_output
cudagraph_stats
slot_mappings
```

一句话：

```text
execute_model 准备中间态；sample_tokens 消费中间态并生成最终 ModelRunnerOutput。
```

## 11. execute_model 主阶段

### _update_states()

把 `SchedulerOutput` 合入 worker 侧 persistent batch。

处理：

```text
删除 finished requests
移除不在本轮的请求
添加新请求和恢复请求
更新 block ids
更新 num_computed_tokens
处理 spec decode 状态修正
处理 Mamba / hybrid / async scheduling
注册 pooling / multimodal 状态
```

一句话：

```text
_update_states 把 Scheduler 的计划同步到 Worker 的请求缓存。
```

### _prepare_inputs()

把 batch 状态变成模型输入张量。

处理：

```text
commit block table
计算 req_indices
计算 query_start_loc
计算 positions
处理 input_ids / prompt_embeds
处理 M-RoPE / XD-RoPE
更新 seq_lens
准备 discard_request_mask
准备 slot mapping
计算 logits_indices
准备 spec decode metadata
准备 async scheduling buffers
```

一句话：

```text
_prepare_inputs 把请求级状态压平成 token 级张量。
```

### _build_attention_metadata()

构造 attention backend 需要的 metadata。

处理：

```text
CommonAttentionMetadata
block table
slot mapping
seq_lens
max_seq_len
query_start_loc
is_prefilling
cascade attention
spec decode metadata
microbatch metadata
CUDA graph padding metadata
```

一句话：

```text
_build_attention_metadata 把 batch 和 KV 信息翻译成 attention backend 可消费的格式。
```

### _preprocess()

准备 forward 前的输入。

处理：

```text
input_ids 或 inputs_embeds
multimodal encoder
encoder-decoder encoder_outputs
prompt embeds
PP intermediate_tensors
model_kwargs
```

一句话：

```text
_preprocess 把普通 token、多模态 embedding、encoder output、PP 中间张量合并成模型 forward 输入。
```

### _model_forward()

真正调用模型：

```text
self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)
```

它输出：

```text
hidden_states
IntermediateTensors
pooling-related outputs
```

### logits / pooling / sample_tokens()

如果是 pooling 模型：

```text
hidden_states → pooler → pooler_output
```

如果是 generation 模型：

```text
hidden_states[logits_indices]
  → model.compute_logits()
  → sample_tokens()
  → sampler / rejection sampler
  → ModelRunnerOutput
```

## 12. sample_tokens 做什么

`sample_tokens()` 消费 `ExecuteModelState`。

它处理：

```text
grammar bitmask
structured output
ordinary sampler
spec decode rejection sampler
logprobs
prompt logprobs
routed experts
async GPU → CPU output copy
next-round draft proposal
ModelRunnerOutput 构造
```

一句话：

```text
sample_tokens 把 logits 和采样状态变成 Scheduler 能对账的 ModelRunnerOutput。
```

## 13. ModelRunnerOutput 是什么

`ModelRunnerOutput` 是执行层返回 Scheduler 的结果对象。

包含：

```text
req_ids
req_id_to_index
sampled_token_ids
logprobs
prompt_logprobs_dict
pooler_output
kv_connector_output
ec_connector_output
num_nans_in_logits
cudagraph_stats
routed_experts
```

它不是用户输出。

它还要经过：

```text
Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor
  → RequestOutput
```

## 14. 执行层和 Scheduler 的闭环

完整闭环：

```text
Scheduler 决定怎么跑
  → SchedulerOutput
  → Executor / Worker / ModelRunner 执行
  → ModelRunnerOutput
  → Scheduler.update_from_output 消化结果
  → Request 状态更新 / 资源释放 / EngineCoreOutputs
```

要背住：

```text
执行层不直接修改 Scheduler 状态；它返回结果，由 Scheduler 对账。
```

## 15. KV cache 在哪一层

KV cache 分三层看：

```text
Scheduler / KVCacheManager：
  决定 block 分配、抢占、释放。

Worker：
  初始化真实 KV cache tensor。

ModelRunner：
  构造 block table、slot mapping、attention metadata，让 attention backend 读写 KV cache。
```

一句话：

```text
Scheduler 管 KV block 账本；Worker 管物理 KV tensor；ModelRunner 管 KV 如何参与本轮 forward。
```

## 16. attention metadata 在哪一层

attention metadata 主要由 ModelRunner 构造。

因为它依赖：

```text
当前 batch 请求顺序
seq_lens
positions
block table
slot mapping
padding / CUDA graph / ubatching
cascade attention
spec decode
```

Scheduler 不构造 attention metadata。

它只提供计划和 block ids。

## 17. multimodal 在哪一层

多模态链路中：

```text
InputProcessor：构造 mm_features。
Scheduler：调度 encoder input。
ModelRunner：执行 multimodal encoder，合并 inputs_embeds。
```

ModelRunner 在 `_preprocess()` 中处理：

```text
_execute_mm_encoder()
_gather_mm_embeddings()
model.embed_input_ids(...)
```

## 18. LoRA 在哪一层

LoRA 控制面：

```text
Executor.add_lora()
  → Worker
  → ModelRunner
  → LoRA manager
```

LoRA 执行面：

```text
SchedulerOutput.NewRequestData.lora_request
  → _update_states()
  → InputBatch.request_lora_mapping
  → set_active_loras()
  → LoRA layer forward
```

ModelRunner 负责把本轮 token 到 LoRA adapter 的 mapping 准备好。

## 19. spec decode 在哪一层

spec decode 横跨：

```text
Scheduler：调度 draft tokens，回滚 rejected tokens。
ModelRunner：把 scheduled draft tokens 放入 InputBatch，构造 SpecDecodeMetadata。
RejectionSampler：接受 / 拒绝 draft tokens。
EngineCore：post_step 回写下一轮 DraftTokenIds。
```

ModelRunner 的职责是：

```text
target model forward 验证 draft tokens
用 RejectionSampler 产出真实 sampled_token_ids
生成下一轮 draft proposal
```

## 20. pipeline parallel 影响

PP 下：

```text
非 last PP rank：forward 后返回 IntermediateTensors。
last PP rank：产生 final hidden states、logits、sampling 输出。
```

Executor 通常只从最后一个 PP stage 的某个输出 rank 收 `ModelRunnerOutput`。

## 21. sleep / wake_up / profile / shutdown

这些属于执行层控制能力。

大致路径：

```text
Engine / EngineCore
  → Executor collective_rpc
  → Worker
  → ModelRunner / device runtime
```

Worker 负责实际执行：

```text
profile
sleep(level)
wake_up(tags)
shutdown
```

## 22. 容易混淆的点

### Executor 不是 Worker

```text
Executor 分发；Worker 承载设备和模型。
```

### Worker 不是 ModelRunner

```text
Worker 管设备生命周期；ModelRunner 管一轮 batch 执行。
```

### SchedulerOutput 不是输入 tensor

```text
SchedulerOutput 是执行计划；ModelRunner 把它转换成 input_ids、positions、slot mapping、metadata。
```

### ModelRunnerOutput 不是 RequestOutput

```text
ModelRunnerOutput 是 Worker → Scheduler 的内部结果；RequestOutput 是 OutputProcessor 生成的用户输出。
```

### execute_model 不一定直接返回 token

```text
generation 路径可能 execute_model 返回 None，再由 sample_tokens 返回 ModelRunnerOutput。
```

## 23. 与其他专题的关系

```text
scheduler：产生 SchedulerOutput，消费 ModelRunnerOutput。
attention：ModelRunner 构造 attention metadata，attention backend 执行 kernel。
sampling_and_output：sample_tokens 和 ModelRunnerOutput 后续输出链路。
parallelism：Executor / Worker / ModelRunner 如何分布在多 rank 上。
multimodal：ModelRunner 执行 encoder 和 inputs_embeds 拼接。
lora_and_adapters：ModelRunner 维护 active LoRA mapping。
compilation_and_cuda_graph：ModelRunner forward 前判断 CUDA graph / compile 路径。
kv_cache_transfer：KV connector metadata 在 Worker forward 前后执行。
```

## 24. 背诵总结

背这一段：

```text
Executor / Worker / ModelRunner 是 vLLM V1 的执行层三段式。Executor 负责把 EngineCore 的 execute_model 分发到一个或多个 Worker；Worker 负责设备、分布式、模型加载、KV cache、warmup、sleep 和 profile 等生命周期；ModelRunner 负责把 SchedulerOutput 转成真实 batch 输入，更新 persistent batch，准备 input_ids、positions、slot mapping、attention metadata，执行模型 forward，计算 logits 或 pooling，并通过 sample_tokens 产出 ModelRunnerOutput。Scheduler 决定怎么跑，执行层负责把计划跑完，Scheduler 再用结果更新请求状态。
```
