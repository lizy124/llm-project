# 06 Executor、Worker、ModelRunner 执行层

本篇梳理 SchedulerOutput 如何通过 Executor/Worker/GPUModelRunner 变成真正的模型 forward 和采样结果。

## 1. 执行层的位置

执行层位于 Scheduler 和模型层之间：

```text
EngineCore
  ↓ SchedulerOutput
Executor
  ↓ collective_rpc("execute_model")
Worker
  ↓
GPUModelRunner
  ↓
model_executor 模型 forward
  ↓
Attention/backend/csrc
```

它负责把调度结果落到设备上。

## 2. Executor 抽象

`Executor` 定义在 `code/vllm/vllm/v1/executor/abstract.py:37`。

它的定义说明：

> An executor is responsible for executing the model on one device, or it can be a distributed executor that can execute the model on multiple devices.

也就是说 executor 可以是单设备，也可以是多设备分布式执行器。

## 3. Executor 类型选择

`Executor.get_class()` 位于 `code/vllm/vllm/v1/executor/abstract.py:48`。

它根据 `parallel_config.distributed_executor_backend` 选择具体 executor：

| backend | executor |
|---|---|
| `uni` | `UniProcExecutor` |
| `mp` | `MultiprocExecutor` |
| `ray` | `RayDistributedExecutor` 或 `RayExecutorV2` |
| `external_launcher` | `ExecutorWithExternalLauncher` |
| 自定义字符串 | 通过 qualname resolve |
| 自定义 class | 直接使用 |

## 4. Executor 的主要接口

| 方法 | 作用 |
|---|---|
| `_init_executor()` | 子类初始化具体 worker/backend |
| `initialize_from_config()` | 初始化 worker KV cache 并 warmup/compile |
| `determine_available_memory()` | 向 worker 查询可用于 KV cache 的显存 |
| `get_kv_cache_specs()` | 向 worker 查询模型需要的 KV cache spec |
| `collective_rpc()` | 对所有 worker 执行 RPC |
| `execute_model()` | 让 worker 执行一个 SchedulerOutput |
| `sample_tokens()` | 单独执行采样阶段 |
| `take_draft_token_ids()` | 取 spec decode draft tokens |
| `shutdown()` | 关闭 worker |
| `sleep()/wake_up()` | 显存睡眠/唤醒 |
| `add_lora()/remove_lora()` | 动态 LoRA 管理 |

核心执行接口：

```text
Executor.execute_model(scheduler_output)
  -> collective_rpc("execute_model", args=(scheduler_output,))
  -> return output[0]
```

位置：`code/vllm/vllm/v1/executor/abstract.py:221`。

## 5. collective_rpc 的意义

`collective_rpc()` 是 executor 到 worker 的控制面 RPC。

在单进程 executor 中，它可能就是函数调用；在多进程/Ray 中，它会跨进程/跨节点调用所有 worker。

常见 RPC：

```text
initialize_from_config
compile_or_warm_up_model
determine_available_memory
get_kv_cache_spec
execute_model
sample_tokens
shutdown
sleep
wake_up
```

注意：真正大规模张量数据不一定通过这个 RPC 传输。注释也说明它更适合控制消息，数据面通信通常由分布式通信机制完成。

## 6. Worker：设备侧执行入口

GPU Worker 定义在 `code/vllm/vllm/v1/worker/gpu_worker.py:117`。

它负责：

- 初始化设备；
- 初始化 torch distributed / model parallel；
- 初始化 KV transfer / EC transfer；
- 创建 GPUModelRunner；
- 加载模型；
- profile 可用显存；
- 分配 KV cache；
- warmup / CUDA graph capture；
- 执行模型；
- sleep/wake；
- weight transfer；
- LoRA 管理。

## 7. Worker 初始化与 ModelRunner 创建

Worker 会根据配置创建不同 model runner。

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:326`。

```text
if self.use_v2_model_runner:
  from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
else:
  from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
```

当前文档重点梳理 `vllm/v1/worker/gpu_model_runner.py`，同时注意新路径 `vllm/v1/worker/gpu/model_runner.py` 也存在并被 `use_v2_model_runner` 使用。

## 8. Worker 的关键阶段

### 8.1 load_model

`Worker.load_model()` 在 `code/vllm/vllm/v1/worker/gpu_worker.py:349`。

它调用：

```text
self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
```

并在需要时创建 weight transfer engine。

### 8.2 determine_available_memory

`Worker.determine_available_memory()` 在 `code/vllm/vllm/v1/worker/gpu_worker.py:371`。

它通过 dummy run/profile 估算可用于 KV cache 的显存。

### 8.3 initialize_from_config

`Worker.initialize_from_config()` 在 `code/vllm/vllm/v1/worker/gpu_worker.py:562`。

流程：

```text
1. 更新 cache_config.num_gpu_blocks
2. ensure_kv_transfer_initialized(...)
3. model_runner.initialize_kv_cache(kv_cache_config)
4. init routed experts capturer if needed
5. 初始化 KV-zero metadata
```

### 8.4 compile_or_warm_up_model

`Worker.compile_or_warm_up_model()` 在 `code/vllm/vllm/v1/worker/gpu_worker.py:591`。

它负责：

- 编译指定 batch size；
- kernel warmup；
- CUDA graph capture；
- 清理/恢复 LoRA；
- 返回 compilation times。

## 9. GPUModelRunner 的定位

`GPUModelRunner` 定义在 `code/vllm/vllm/v1/worker/gpu_model_runner.py:418`。

它是 worker 侧最核心的执行类。

它不是简单调用模型，而是负责整个 GPU batch 执行生命周期：

```text
SchedulerOutput
  ↓
更新 persistent batch state
  ↓
准备 input ids / positions / mrope / xdrope
  ↓
准备 block table / slot mapping
  ↓
构造 attention metadata
  ↓
准备 multimodal encoder inputs
  ↓
set_forward_context
  ↓
model forward
  ↓
compute logits
  ↓
sample tokens
  ↓
更新 request/batch 状态
  ↓
ModelRunnerOutput
```

## 10. GPUModelRunner.execute_model 主流程

`execute_model()` 定义在 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`。

高层流程如下：

```text
1. 检查上一次 execute_model 是否已经 sample_tokens
2. 处理 ngram_gpu spec decode 的 scheduler_output copy
3. KV connector 处理 preemptions
4. num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
5. preprocess 阶段：
   - _update_states(scheduler_output)
   - 如有 EC transfer producer，执行 encoder 并返回
   - 如果没有 token，返回 empty output 或 connector no-forward
   - _prepare_inputs(...)
   - _determine_batch_execution_and_padding(...)
   - maybe_create_ubatch_slices(...)
   - Mamba preprocess
   - _get_slot_mappings(...)
   - _build_attention_metadata(...)
   - _preprocess(...)
6. set_forward_context(...)
7. _model_forward(...)
8. postprocess：
   - 取 hidden states
   - pipeline parallel 中可能返回 intermediate tensors
   - pooling 模型走 _pool
   - generation 模型 compute_logits
9. 保存 execute_model_state
10. 返回 None，等待 sample_tokens()
```

## 11. 为什么 execute_model 返回 None

常见生成路径里，`execute_model()` 只做到 logits，并把状态存在：

```text
self.execute_model_state = ExecuteModelState(...)
```

然后返回 `None`。

EngineCore 看到 `model_output is None` 后，会调用：

```text
Executor.sample_tokens(grammar_output)
```

这样结构化输出 grammar bitmask 能插在 logits 和 sample 之间。

## 12. sample_tokens 主流程

`sample_tokens()` 定义在 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`。

流程：

```text
1. 如果 execute_model_state 为空：
   - 说明只有 KV connector output 或 PP 非末级 rank
   - 返回对应 ModelRunnerOutput
2. 取出 execute_model_state
3. 如果有 grammar_output，apply_grammar_bitmask
4. _sample(logits, spec_decode_metadata)
5. _update_states_after_model_execute(sampled_token_ids, scheduler_output)
6. async scheduling + PP 时广播 sampled tokens
7. 清理 draft token 临时状态
8. 生成 ModelRunnerOutput
```

## 13. Batch state 与 input preparation

GPUModelRunner 内部维护 persistent batch state。

常见相关方法：

| 方法 | 作用 |
|---|---|
| `_update_states()` | 根据 SchedulerOutput 更新 batch/request state |
| `_prepare_input_ids()` | 准备 input_ids |
| `_prepare_inputs()` | 准备 input_ids、positions、embeds、intermediate tensors 等 |
| `_preprocess()` | 更综合的模型输入预处理 |
| `_get_positions()` | 生成 position ids |
| `_calc_mrope_positions()` | 多模态 RoPE position |
| `_calc_spec_decode_metadata()` | spec decode metadata |

这些步骤把 scheduler 的逻辑调度结果转换为模型 forward 能接受的张量。

## 14. Attention metadata 与 slot mapping

执行 attention 前必须准备：

- 每个 token 对应的 KV slot；
- 每个 request 的 sequence length；
- block table；
- common prefix blocks；
- query length；
- max query len；
- cascade attention metadata；
- spec decode metadata。

关键方法：

| 方法 | 位置 | 作用 |
|---|---|---|
| `_get_slot_mappings()` | `code/vllm/vllm/v1/worker/gpu_model_runner.py:3960` | 生成 token 到 KV slot 的映射 |
| `_build_attention_metadata()` | `code/vllm/vllm/v1/worker/gpu_model_runner.py:2208` | 构造 attention backend 需要的 metadata |
| `initialize_attn_backend()` | `code/vllm/vllm/v1/worker/gpu_model_runner.py:6736` | 初始化 attention backend |
| `initialize_metadata_builders()` | `code/vllm/vllm/v1/worker/gpu_model_runner.py:6843` | 初始化 metadata builder |

## 15. CUDA Graph / padding / ubatching

`GPUModelRunner` 会根据 batch 形状决定执行方式：

```text
_determine_batch_execution_and_padding(...)
```

它会考虑：

- 是否使用 CUDA graph；
- batch 是否需要 padding 到 capture size；
- 是否使用 ubatching；
- DP 下 token 数协调；
- 是否是 uniform decode；
- 是否要 skip compiled path。

这些逻辑是 vLLM 性能优化的重要部分。

## 16. Pipeline Parallel 情况

如果开启 PP：

- 非最后一个 PP rank 可能返回 `IntermediateTensors`；
- 最后一个 PP rank 才 compute logits / sample；
- 某些配置下 logits 会 broadcast 给所有 rank；
- async scheduling 下 sampled token ids 也需要跨 PP rank 通信。

相关代码在 `execute_model()` 和 `sample_tokens()` 中。

## 17. 执行层的输入输出

### 输入

- `SchedulerOutput`；
- KV cache config；
- model weights；
- attention backend config；
- distributed parallel config；
- grammar output；
- LoRA/quantization/spec decode 配置。

### 输出

- `ModelRunnerOutput`；
- sampled token ids；
- logprobs；
- prompt logprobs；
- pooling output；
- KV connector output；
- draft token ids；
- intermediate tensors。

## 18. 一句话总结

Executor 是 EngineCore 到 worker 的执行抽象，Worker 是设备侧生命周期管理器，GPUModelRunner 是真正把 SchedulerOutput 变成模型 forward、logits 和 sampled tokens 的核心执行器。
