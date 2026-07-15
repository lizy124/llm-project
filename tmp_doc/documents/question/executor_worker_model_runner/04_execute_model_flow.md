# 04. execute_model() 如何从 EngineCore 走到模型执行？

源码位置：

- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/executor/abstract.py`
- `code/vllm/vllm/v1/executor/uniproc_executor.py`
- `code/vllm/vllm/v1/executor/multiproc_executor.py`
- `code/vllm/vllm/v1/executor/ray_executor.py`
- `code/vllm/vllm/v1/worker/worker_base.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py`

本问题关注：`Scheduler.schedule()` 生成 `SchedulerOutput` 后，`EngineCore` 如何把它交给执行层；`Executor` 如何按单进程 / 多进程 / Ray 后端分发；`Worker` 如何接住请求；`GPUModelRunner.execute_model()` 如何真正完成输入准备、attention metadata、forward、logits / pooling，并在必要时把采样拆到 `sample_tokens()` 阶段。

---

## 1. 一句话回答

`execute_model()` 是 vLLM V1 中把“调度计划”变成“模型执行”的主桥梁。

主链路是：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output)
  → Executor.collective_rpc("execute_model")
  → WorkerWrapperBase.execute_model()
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
  → _update_states()
  → _prepare_inputs()
  → _build_attention_metadata()
  → _preprocess()
  → _model_forward()
  → logits / pooling / IntermediateTensors
  → sample_tokens()  # 生成类模型通常在这里采样
  → ModelRunnerOutput
```

所以：

```text
SchedulerOutput 是执行计划；
Executor 是分发器；
Worker 是设备侧运行时；
ModelRunner 是真正把计划变成 forward / sampling 的地方。
```

---

## 2. 最小主链路

从 `EngineCore.step()` 看，执行阶段夹在 schedule 和 update 之间。

```text
scheduler.schedule()
  → model_executor.execute_model(scheduler_output)
  → scheduler.get_grammar_bitmask(scheduler_output)
  → future.result()
  → 如果 model_output is None，则 model_executor.sample_tokens(grammar_output)
  → _process_aborts_queue()
  → scheduler.update_from_output(scheduler_output, model_output)
```

对应源码：`core.py:488` 到 `core.py:517`

关键点：

```text
1. Scheduler 先生成 SchedulerOutput；
2. Executor 先启动 execute_model；
3. Scheduler 同时准备 grammar bitmask；
4. 如果 execute_model 没直接返回 ModelRunnerOutput，就用 sample_tokens 补齐；
5. 在回收 Worker 输出前处理执行期间到达的 abort 请求；
6. 最后 Scheduler.update_from_output() 对账并生成 EngineCoreOutputs。
```

这个设计把一轮执行拆成三段：

```text
计划：SchedulerOutput
执行：execute_model / sample_tokens
回收：update_from_output
```

---

## 3. EngineCore 中 execute_model 的位置

入口：

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
```

位置：`core.py:488`

核心代码：

```python
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
...
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

位置：`core.py:499` 到 `core.py:508`

这里有几个重要细节。

### 3.1 `model_executor` 实际上就是 Executor

在 EngineCore 初始化时：

```python
self.model_executor = executor_class(vllm_config)
```

位置：`core.py:123`

所以 `model_executor.execute_model()` 不是直接调用模型，而是调用某个具体 Executor 后端。

### 3.2 execute_model 使用 non_block=True

`EngineCore.step()` 调用：

```python
self.model_executor.execute_model(scheduler_output, non_block=True)
```

位置：`core.py:500`

这说明 EngineCore 希望执行层可以异步启动模型执行，然后自己继续准备 grammar bitmask 等内容。在 `step_with_batch_queue()` 路径中，EngineCore 还会根据 `pending_structured_output_tokens` 决定是否立即调用 `sample_tokens(non_block=True)`，或者把 sampling 延后到上一批输出处理后再做。

时间线可以理解为：

```text
T0: Scheduler 生成 SchedulerOutput
T1: Executor 启动 Worker 执行，返回 Future
T2: Scheduler 准备 grammar bitmask
T3: EngineCore 等 Future.result()
T4: 如果需要，再调用 sample_tokens(grammar_output)
```

### 3.3 update_from_output 仍然需要原始 SchedulerOutput

执行完成后：

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

位置：`core.py:513` 到 `core.py:515`

说明 `SchedulerOutput` 不只是发给 Worker 的执行计划，也是回收阶段的对账凭证。

---

## 4. Executor 抽象层如何转发 execute_model

抽象定义在：`abstract.py:221`

```python
def execute_model(
    self, scheduler_output: SchedulerOutput, non_block: bool = False
) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
    output = self.collective_rpc(
        "execute_model", args=(scheduler_output,), non_block=non_block
    )
    return output[0]
```

位置：`abstract.py:221` 到 `abstract.py:227`

这说明抽象层做的事非常明确：

```text
把 SchedulerOutput 作为参数，广播调用 worker.execute_model()。
```

### 4.1 输入是什么

输入是：

```text
SchedulerOutput
```

它来自 Scheduler，包含本轮 Worker 需要知道的执行计划，例如：

```text
scheduled_new_reqs
scheduled_cached_reqs
num_scheduled_tokens
total_num_scheduled_tokens
scheduled_spec_decode_tokens
scheduled_encoder_inputs
num_common_prefix_blocks
finished_req_ids
kv_connector_metadata
ec_connector_metadata
```

### 4.2 输出是什么

输出可能是：

```text
ModelRunnerOutput
AsyncModelRunnerOutput
Future[ModelRunnerOutput]
None
```

其中 `None` 表示：

```text
forward 已经执行或中间状态已保存，但最终 ModelRunnerOutput 还要通过 sample_tokens() 取得。
```

---

## 5. UniProcExecutor 路径

源码：`uniproc_executor.py`

单进程路径最直接。

### 5.1 初始化

`UniProcExecutor._init_executor()` 会：

```text
1. 创建 WorkerWrapperBase；
2. 准备 distributed_init_method / rank / local_rank；
3. init_worker；
4. init_device；
5. load_model。
```

位置：`uniproc_executor.py:45` 到 `uniproc_executor.py:70`

### 5.2 execute_model

```python
def execute_model(self, scheduler_output, non_block=False):
    output = self.collective_rpc(
        "execute_model",
        args=(scheduler_output,),
        non_block=non_block,
        single_value=True,
    )
    return output
```

位置：`uniproc_executor.py:108` 到 `uniproc_executor.py:121`

### 5.3 collective_rpc

单进程里 `collective_rpc()` 本质是本地方法调用：

```python
result = run_method(self.driver_worker, method, args, kwargs)
```

位置：`uniproc_executor.py:91` 到 `uniproc_executor.py:95`

因此单进程链路是：

```text
EngineCore
  → UniProcExecutor.execute_model()
  → run_method(driver_worker, "execute_model")
  → WorkerWrapperBase.execute_model()
  → Worker.execute_model()
```

如果返回 `AsyncModelRunnerOutput`，会调用 `get_output()` 或包装成 `AsyncOutputFuture`。

位置：`uniproc_executor.py:91` 到 `uniproc_executor.py:106`

---

## 6. MultiprocExecutor 路径

源码：`multiproc_executor.py`

多进程路径的核心是：

```text
主进程 Executor 通过 MessageQueue 把命令广播给 WorkerProc，
WorkerProc 在子进程里调用 Worker，
再通过 response queue 把结果返回。
```

### 6.1 execute_model

```python
def execute_model(self, scheduler_output, non_block=False):
    return self.collective_rpc(
        "execute_model",
        args=(scheduler_output,),
        unique_reply_rank=self.output_rank,
        non_block=non_block,
        timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        kv_output_aggregator=self.kv_output_aggregator,
    )
```

位置：`multiproc_executor.py:310` 到 `multiproc_executor.py:320`

和抽象层相比，多进程路径多了几个关键参数。

### 6.2 `unique_reply_rank`

```text
只从某个 rank 收 ModelRunnerOutput。
```

因为在 TP / PP 场景下，通常不是所有 worker 都产生用户侧需要的输出。

`output_rank` 的计算逻辑是：

```text
只从最后一个 PP stage 的第一个 TP rank 返回 ModelRunnerOutput。
```

位置：`multiproc_executor.py:498` 到 `multiproc_executor.py:512`

### 6.3 `kv_output_aggregator`

如果启用了 KV connector，可能需要收集多个 worker 的 KV transfer 输出并聚合。

所以这时不能简单只看一个 rank，需要由 `KVOutputAggregator` 聚合。

位置：`multiproc_executor.py:364` 到 `multiproc_executor.py:368`

### 6.4 collective_rpc 的核心流程

```text
1. 把 method / args / kwargs / output_rank 放入 rpc_broadcast_mq；
2. 选择 response_mqs；
3. 从 response_mq 读取结果；
4. 如果失败，抛出 worker error；
5. 用 FutureWrapper 支持 non_block。
```

对应源码：`multiproc_executor.py:343` 到 `multiproc_executor.py:405`

核心发送：

```python
self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
```

位置：`multiproc_executor.py:377`

核心接收：

```python
status, result = mq.dequeue(timeout=dequeue_timeout)
```

位置：`multiproc_executor.py:390`

---

## 7. RayDistributedExecutor 路径

源码：`ray_executor.py`

Ray 路径比较特殊，因为它可能把 `execute_model()` 和 `sample_tokens()` 拆得更明显。

### 7.1 execute_model

```python
def execute_model(self, scheduler_output, non_block=False):
    if self.scheduler_output is not None:
        raise RuntimeError(...)

    if not self.uses_sampler or not scheduler_output.total_num_scheduled_tokens:
        return self._execute_dag(scheduler_output, None, non_block)

    self.scheduler_output = scheduler_output
    return COMPLETED_NONE_FUTURE if non_block else None
```

位置：`ray_executor.py:390` 到 `ray_executor.py:407`

含义：

```text
如果不需要 sampler，或者本轮没有 token，则直接执行 DAG；
如果需要采样，则先保存 scheduler_output，返回 None，等待 sample_tokens()。
```

### 7.2 sample_tokens

```python
def sample_tokens(self, grammar_output, non_block=False):
    scheduler_output = self.scheduler_output
    if scheduler_output is None:
        return COMPLETED_NONE_FUTURE if non_block else None
    self.scheduler_output = None
    return self._execute_dag(scheduler_output, grammar_output, non_block)
```

位置：`ray_executor.py:409` 到 `ray_executor.py:432`

所以 Ray 路径的语义是：

```text
execute_model() 记录这轮要执行的 SchedulerOutput；
sample_tokens() 带着 grammar_output 触发真正 DAG 执行并返回结果。
```

### 7.3 _execute_dag

```text
1. 第一次执行时构建 compiled Ray DAG；
2. 执行 forward_dag；
3. 没有 KV connector 时，只取一个输出；
4. 有 KV connector 时，聚合多个输出。
```

位置：`ray_executor.py:434` 到 `ray_executor.py:468`

---

## 8. WorkerWrapperBase 如何接住 execute_model

源码：`worker_base.py`

`WorkerWrapperBase` 是 Executor 和真实 Worker 之间的一层包装。

### 8.1 初始化真实 Worker

`init_worker()` 会根据 `parallel_config.worker_cls` 动态解析 Worker 类：

```python
worker_class = resolve_obj_by_qualname(parallel_config.worker_cls)
```

位置：`worker_base.py:249` 到 `worker_base.py:253`

然后创建真实 Worker：

```python
self.worker = worker_class(**kwargs)
```

位置：`worker_base.py:317` 到 `worker_base.py:319`

### 8.2 execute_model 包装

```python
def execute_model(self, scheduler_output):
    self._apply_mm_cache(scheduler_output)
    return self.worker.execute_model(scheduler_output)
```

位置：`worker_base.py:346` 到 `worker_base.py:351`

这里有一个细节：

```text
多模态 receiver cache 会在进入真实 Worker 前应用到 scheduler_output.scheduled_new_reqs。
```

也就是说，WorkerWrapperBase 不只是简单转发，还会补一层多模态 cache 适配。

---

## 9. GPU Worker 如何执行 execute_model

源码：`gpu_worker.py`

入口：

```python
def execute_model(self, scheduler_output):
```

位置：`gpu_worker.py:1002`

### 9.1 前置：等待上一次 PP send 完成

```python
if self._pp_send_work:
    for handle in self._pp_send_work:
        handle.wait()
    self._pp_send_work = []
```

位置：`gpu_worker.py:1005` 到 `gpu_worker.py:1009`

说明 pipeline parallel 下，上一轮异步发送的 intermediate tensors 需要先完成。

### 9.2 判断本轮是否真的 forward

```python
forward_pass = scheduler_output.total_num_scheduled_tokens > 0
```

位置：`gpu_worker.py:1011` 到 `gpu_worker.py:1014`

如果没有 scheduled tokens，后面可能走空输出或 connector-only 路径。

### 9.3 非 first PP rank 接收中间张量

如果是 pipeline parallel 且当前 rank 不是 first stage：

```python
tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(...)
intermediate_tensors = AsyncIntermediateTensors(...)
```

位置：`gpu_worker.py:1047` 到 `gpu_worker.py:1059`

这说明：

```text
非首个 PP stage 的输入不是 token ids，而是前一个 PP stage 发来的 IntermediateTensors。
```

### 9.4 调用 ModelRunner

```python
output = self.model_runner.execute_model(
    scheduler_output, intermediate_tensors
)
```

位置：`gpu_worker.py:1061` 到 `gpu_worker.py:1064`

如果返回的是：

```text
ModelRunnerOutput / AsyncModelRunnerOutput / None
```

则直接返回给 Executor。

位置：`gpu_worker.py:1065` 到 `gpu_worker.py:1074`

### 9.5 如果返回 IntermediateTensors

如果 ModelRunner 返回 `IntermediateTensors`，说明当前 rank 不是最后一个 PP stage，需要发给下一个 stage：

```python
self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors, ...)
return None
```

位置：`gpu_worker.py:1076` 到 `gpu_worker.py:1090`

状态可以理解为：

```text
当前 PP stage 已完成自己的 forward，
但还不是最终输出 rank，
所以只发送中间结果并返回 None。
```

---

## 10. GPUModelRunner.execute_model 做了什么

源码：`gpu_model_runner.py`

入口：

```python
def execute_model(self, scheduler_output, intermediate_tensors=None)
```

位置：`gpu_model_runner.py:4097`

它是执行链路最核心的函数。

---

## 11. GPUModelRunner 第一阶段：状态和特殊路径处理

### 11.1 防止 execute_model / sample_tokens 顺序错误

```python
if self.execute_model_state is not None:
    raise RuntimeError(
        "State error: sample_tokens() must be called after execute_model() returns None."
    )
```

位置：`gpu_model_runner.py:4102` 到 `gpu_model_runner.py:4106`

这说明：

```text
如果上一次 execute_model 返回 None，必须先调用 sample_tokens() 清理状态，不能再次 execute_model。
```

### 11.2 spec decode ngram GPU 需要复制 SchedulerOutput

如果启用 ngram GPU speculative decoding，会复制 `num_scheduled_tokens` 和 `scheduled_spec_decode_tokens`。

位置：`gpu_model_runner.py:4111` 到 `gpu_model_runner.py:4126`

原因是避免 Worker 侧修改影响 EngineCore 进程里的原始 `SchedulerOutput`。

### 11.3 KV transfer preemption 处理

如果存在 KV transfer group：

```python
get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
```

位置：`gpu_model_runner.py:4128` 到 `gpu_model_runner.py:4131`

这说明执行层也要消费 SchedulerOutput 中携带的 KV connector metadata。

---

## 12. GPUModelRunner 第二阶段：更新 batch 状态

```python
deferred_state_corrections_fn = self._update_states(scheduler_output)
```

位置：`gpu_model_runner.py:4138` 到 `gpu_model_runner.py:4139`

这一步很关键：

```text
SchedulerOutput 进入 Worker 后，首先要更新 Worker 侧持久 batch 状态。
```

它会让 Worker / ModelRunner 知道：

```text
哪些请求是新来的；
哪些请求已经在 batch 中；
每个请求本轮调度多少 token；
哪些请求 finished；
哪些 KV / encoder / spec 状态要更新。
```

这一部分会在后续 `05_input_batch_and_state_update.md` 里展开。

---

## 13. GPUModelRunner 第三阶段：0-token / connector-only 路径

如果本轮没有调度 token：

```python
if not num_scheduled_tokens:
    if not has_kv_transfer_group():
        return EMPTY_MODEL_RUNNER_OUTPUT
    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
```

位置：`gpu_model_runner.py:4149` 到 `gpu_model_runner.py:4165`

这表示：

```text
SchedulerOutput 可能存在，但本轮没有真正 forward。
```

常见原因：

```text
- 只有 finished_req_ids / connector metadata 需要处理；
- 远端 KV load / save 相关状态推进；
- DP / external launcher 的空步协调；
- 没有可执行 token。
```

所以 `execute_model()` 不等于“一定跑模型 forward”。

---

## 14. GPUModelRunner 第四阶段：准备输入

### 14.1 准备 logits indices 和 spec metadata

```python
logits_indices, spec_decode_metadata = self._prepare_inputs(
    scheduler_output,
    num_scheduled_tokens_np,
)
```

位置：`gpu_model_runner.py:4181` 到 `gpu_model_runner.py:4184`

这一步会把 SchedulerOutput 中的 token 调度信息转成模型输入所需的索引和 spec decode 元数据。

### 14.2 决定 batch 执行形态和 padding

```python
(
    cudagraph_mode,
    batch_desc,
    should_ubatch,
    num_tokens_across_dp,
    cudagraph_stats,
) = self._determine_batch_execution_and_padding(...)
```

位置：`gpu_model_runner.py:4196` 到 `gpu_model_runner.py:4209`

这一步决定：

```text
- 是否使用 CUDA graph；
- batch 是否需要 padding；
- 是否需要 microbatch / ubatch；
- DP 维度上的 token 数；
- cudagraph 统计信息；
- 后续是否需要生成 DBO / ubatch slices。
```

### 14.3 计算 slot mapping

```python
slot_mappings_by_group, slot_mappings = self._get_slot_mappings(...)
```

位置：`gpu_model_runner.py:4297` 到 `gpu_model_runner.py:4306`

`slot_mapping` 是 KV Cache 写入位置的重要桥梁。

### 14.4 构建 attention metadata

```python
attn_metadata, spec_decode_common_attn_metadata = self._build_attention_metadata(...)
```

位置：`gpu_model_runner.py:4308` 到 `gpu_model_runner.py:4322`

这一步会把：

```text
num_tokens
num_reqs
max_query_len
slot_mappings
cascade attention prefix length
spec decode 信息
num_common_prefix_blocks
```

转成 attention backend 可理解的 metadata。

### 14.5 预处理模型输入

```python
(
    input_ids,
    inputs_embeds,
    positions,
    intermediate_tensors,
    model_kwargs,
    ec_connector_output,
) = self._preprocess(...)
```

位置：`gpu_model_runner.py:4324` 到 `gpu_model_runner.py:4333`

这一步准备真正传给模型的输入。

---

## 15. GPUModelRunner 第五阶段：模型 forward

核心 forward：

```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`gpu_model_runner.py:4380` 到 `gpu_model_runner.py:4386`

forward 外面包了 `set_forward_context()`：

```python
set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    ubatch_slices=ubatch_slices_padded,
    slot_mapping=slot_mappings,
    skip_compiled=has_encoder_input,
)
```

位置：`gpu_model_runner.py:4362` 到 `gpu_model_runner.py:4378`

这说明模型 forward 不是裸调用，而是在一个包含 attention metadata、CUDA graph、batch descriptor、slot mapping 的上下文中执行。

---

## 16. GPUModelRunner 第六阶段：forward 后处理

forward 后有三种主要路径。

### 16.1 非最后 PP rank：返回 IntermediateTensors

```python
if not get_pp_group().is_last_rank:
    assert isinstance(hidden_states, IntermediateTensors)
    self.kv_connector_output = kv_connector_output
    return hidden_states
```

位置：`gpu_model_runner.py:4397` 到 `gpu_model_runner.py:4403`

这表示当前 rank 只完成中间层计算，输出要交给下一个 PP stage。

### 16.2 pooling 模型：直接返回 pooling output

```python
if self.is_pooling_model:
    return self._pool(...)
```

位置：`gpu_model_runner.py:4405` 到 `gpu_model_runner.py:4412`

Pooling / embedding 类任务不走 token sampling。

### 16.3 generation 模型：计算 logits，但暂不返回最终输出

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`gpu_model_runner.py:4414` 到 `gpu_model_runner.py:4415`

然后保存临时状态：

```python
self.execute_model_state = ExecuteModelState(...)
self.kv_connector_output = kv_connector_output
return None
```

位置：`gpu_model_runner.py:4446` 到 `gpu_model_runner.py:4465`

这就是生成类模型中 `execute_model()` 经常返回 `None` 的根本原因：

```text
forward 和 logits 已经完成；
但 grammar bitmask / sampling / bookkeeping / ModelRunnerOutput 构造在 sample_tokens() 中完成。
```

---

## 17. sample_tokens() 如何接住 execute_model 的 None

入口：

```python
def sample_tokens(self, grammar_output)
```

位置：`gpu_model_runner.py:4483`

### 17.1 没有 execute_model_state 的情况

如果 `execute_model_state is None`：

```python
return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

位置：`gpu_model_runner.py:4486` 到 `gpu_model_runner.py:4494`

这通常表示没有可采样的 forward 状态，只需要把 KV connector output 传回。

### 17.2 有 execute_model_state 的情况

正常 generation 路径会解包 execute_model 保存的状态：

```python
(
    scheduler_output,
    logits,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states,
    ec_connector_output,
    cudagraph_stats,
    slot_mappings,
) = self.execute_model_state
```

位置：`gpu_model_runner.py:4496` 到 `gpu_model_runner.py:4508`

然后清掉状态：

```python
self.execute_model_state = None
```

位置：`gpu_model_runner.py:4509` 到 `gpu_model_runner.py:4510`

### 17.3 应用 grammar bitmask

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`gpu_model_runner.py:4512` 到 `gpu_model_runner.py:4516`

这解释了为什么 EngineCore 要先执行 `get_grammar_bitmask()`，然后在 `execute_model()` 返回 None 时传给 `sample_tokens()`。

### 17.4 采样

```python
sampler_output = self._sample(logits, spec_decode_metadata)
```

位置：`gpu_model_runner.py:4518` 到 `gpu_model_runner.py:4519`

### 17.5 更新 Worker 侧状态

```python
self._update_states_after_model_execute(
    sampler_output.sampled_token_ids, scheduler_output
)
```

位置：`gpu_model_runner.py:4521` 到 `gpu_model_runner.py:4523`

这一步更新 Worker / InputBatch 侧的 sampled token 状态。

### 17.6 构造 ModelRunnerOutput

在构造输出前，spec decode 路径会先 finalize KV connector，并执行 EPLB step；async scheduling 路径还会为 routed experts 创建设备侧快照，避免异步 D2H copy 读到下一步覆盖后的共享 buffer。

```python
output = ModelRunnerOutput(
    req_ids=req_ids_output_copy,
    req_id_to_index=req_id_to_index_output_copy,
    sampled_token_ids=valid_sampled_token_ids,
    logprobs=logprobs_lists,
    prompt_logprobs_dict=prompt_logprobs_dict,
    kv_connector_output=kv_connector_output,
    ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
    num_nans_in_logits=num_nans_in_logits,
    cudagraph_stats=cudagraph_stats,
    routed_experts=None,
)
```

位置：`gpu_model_runner.py:4696` 到 `gpu_model_runner.py:4710`

如果不是 async scheduling，直接返回：

```python
return output
```

位置：`gpu_model_runner.py:4712` 到 `gpu_model_runner.py:4722`

如果是 async scheduling，则包装成 `AsyncGPUModelRunnerOutput`：

```python
async_output = AsyncGPUModelRunnerOutput(...)
return async_output
```

位置：`gpu_model_runner.py:4724` 到 `gpu_model_runner.py:4770`

---

## 18. execute_model 何时返回 None

这点非常重要。

### 18.1 生成模型的常见路径

在 GPUModelRunner 中，generation 模型 forward 后会：

```text
保存 ExecuteModelState
返回 None
等待 sample_tokens()
```

原因是采样阶段还需要 grammar bitmask、logprobs bookkeeping、spec decode draft proposal、异步 D2H copy 等处理。

### 18.2 非最后 PP stage

GPU Worker 如果收到 `IntermediateTensors`，会发送给下一个 PP stage，然后返回 None。

```text
当前 PP stage 不是最终输出 stage：
  ModelRunner 返回 IntermediateTensors
  Worker 异步发送到下一个 PP stage
  Worker 返回 None
```

位置：`gpu_worker.py:1076` 到 `gpu_worker.py:1090`

### 18.3 Ray 后端需要 sample_tokens 触发 DAG

RayDistributedExecutor 中，如果需要 sampler：

```text
execute_model() 保存 scheduler_output
返回 None
sample_tokens() 再执行 DAG
```

位置：`ray_executor.py:390` 到 `ray_executor.py:432`

### 18.4 没有可执行 token

如果 `total_num_scheduled_tokens == 0`，ModelRunner 可能返回：

```text
EMPTY_MODEL_RUNNER_OUTPUT
或 connector-only output
```

不一定是 None。

位置：`gpu_model_runner.py:4149` 到 `gpu_model_runner.py:4165`

---

## 19. execute_model 的输入输出对象

### 19.1 输入：SchedulerOutput

`SchedulerOutput` 是 Scheduler 给执行层的一轮计划。

它告诉 Worker：

```text
哪些请求是新请求；
哪些请求已经缓存；
每个请求本轮跑多少 token；
是否有 spec decode tokens；
是否有 encoder inputs；
哪些请求 finished；
KV connector / EC connector metadata 是什么；
公共 prefix block 信息是什么。
```

### 19.2 中间对象：IntermediateTensors

Pipeline parallel 场景中，中间 PP stage 之间传递的是：

```text
IntermediateTensors
```

它不是用户输出，而是模型层间的 hidden states / residual 等中间张量。

### 19.3 输出：ModelRunnerOutput

最终给 Scheduler.update_from_output() 的是：

```text
ModelRunnerOutput
```

它包含：

```text
req_ids
req_id_to_index
sampled_token_ids
logprobs
prompt_logprobs_dict
kv_connector_output
ec_connector_output
num_nans_in_logits
cudagraph_stats
routed_experts
```

Scheduler 会用它和原始 `SchedulerOutput` 对账。

---

## 20. execute_model 和 Scheduler 的边界

### 20.1 Scheduler 负责

```text
- 决定哪些请求执行；
- 决定每个请求执行多少 token；
- 分配 KV block；
- 构造 SchedulerOutput；
- 根据 ModelRunnerOutput 更新请求状态；
- 释放资源并生成 EngineCoreOutputs。
```

### 20.2 execute_model 负责

```text
- 消费 SchedulerOutput；
- 更新 Worker 侧 batch 状态；
- 准备模型输入；
- 构造 attention metadata；
- 执行 forward；
- 计算 logits / pooling；
- 执行采样或保存采样状态；
- 生成 ModelRunnerOutput。
```

### 20.3 两者靠什么对接

```text
SchedulerOutput：Scheduler → Worker 的执行计划
ModelRunnerOutput：Worker → Scheduler 的真实结果
```

---

## 21. 容易疑惑的点

### 21.1 execute_model 是不是一定会跑 forward？

不是。

如果本轮 `total_num_scheduled_tokens == 0`，可能返回空输出或 connector-only 输出。

### 21.2 execute_model 是不是一定返回 ModelRunnerOutput？

不是。

它可能返回：

```text
ModelRunnerOutput
AsyncModelRunnerOutput
IntermediateTensors
Future
None
```

不同层看到的返回值不完全一样。

### 21.3 为什么 EngineCore 要处理 model_output is None？

因为生成模型常把 forward/logits 和 sampling 拆开：

```text
execute_model() 完成 forward 并保存 ExecuteModelState；
sample_tokens() 应用 grammar、采样并构造 ModelRunnerOutput。
```

### 21.4 为什么 SchedulerOutput 还要传回 update_from_output？

因为 Worker 只返回真实执行结果，而 Scheduler 还需要知道：

```text
这批结果对应原计划里的哪些请求、哪些 token、哪些 spec tokens、哪些 connector metadata。
```

所以 update 阶段必须同时拿：

```text
SchedulerOutput + ModelRunnerOutput
```

### 21.5 Executor 是不是执行模型的地方？

不是。

Executor 主要做分发和通信；真正模型执行在 Worker / ModelRunner。

---

## 22. 总结

`execute_model()` 主链路可以压缩成：

```text
EngineCore.step()
  → Scheduler.schedule()
  → Executor.execute_model(SchedulerOutput)
  → Worker.execute_model(SchedulerOutput)
  → ModelRunner.execute_model(SchedulerOutput)
  → prepare inputs / attention metadata
  → model forward
  → logits / pooling / intermediate tensors
  → sample_tokens() 生成 ModelRunnerOutput
  → Scheduler.update_from_output(SchedulerOutput, ModelRunnerOutput)
```

如果只记住一句话：

```text
execute_model() 是 SchedulerOutput 进入真实模型执行层的入口；它本身不只是一次 forward，而是一段跨 EngineCore、Executor、Worker、ModelRunner 的执行协议。
```

再压缩成最小心智模型：

```text
SchedulerOutput 是计划；
Executor 负责派发；
Worker 负责设备侧运行；
ModelRunner 负责 forward / sampling；
ModelRunnerOutput 是回收凭证。
```
