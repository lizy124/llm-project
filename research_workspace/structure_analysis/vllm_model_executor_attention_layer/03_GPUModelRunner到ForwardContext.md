# 03 GPUModelRunner 到 ForwardContext

本篇梳理 worker 侧 `GPUModelRunner` 如何把 Scheduler 的调度结果转换成模型 forward 所需的张量、attention metadata、slot mapping，并通过 `ForwardContext` 传递给模型层。

## 1. GPUModelRunner 的定位

`GPUModelRunner` 是模型执行层的直接调用者，定义在：

```text
vllm/v1/worker/gpu_model_runner.py
```

类定义位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:418`。

它在链路中的位置：

```text
SchedulerOutput
  ↓
Executor.execute_model
  ↓
Worker.execute_model
  ↓
GPUModelRunner.execute_model
  ↓
set_forward_context
  ↓
model forward
  ↓
Attention.forward
```

## 2. execute_model 的总体流程

`GPUModelRunner.execute_model()` 位于 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`。

高层流程：

```text
1. 检查 execute_model_state，确保上一次 forward 后已 sample
2. 处理 routed experts buffer
3. 处理 spec decode ngram_gpu 的 SchedulerOutput copy
4. 处理 KV connector preemptions
5. 读取本步 total_num_scheduled_tokens
6. preprocess 阶段：
   - _update_states
   - encoder / EC transfer 特殊路径
   - 无 token 时返回 empty output
   - _prepare_inputs
   - _determine_batch_execution_and_padding
   - maybe_create_ubatch_slices
   - mamba preprocess
   - _get_slot_mappings
   - _build_attention_metadata
   - _preprocess
7. set_forward_context
8. _model_forward
9. postprocess hidden states / logits / pooling / PP
10. 保存 execute_model_state
11. 返回 None，等待 sample_tokens
```

## 3. 为什么 GPUModelRunner 不只是调用模型

模型 forward 需要的不只是 `input_ids`。

对于 vLLM 来说，每一步还需要：

- 每个 request 本步执行几个 token；
- 每个 token 的 position；
- 每个 token 对应 KV cache 的 slot；
- 每个 request 的 block table；
- attention backend 所需 metadata；
- prefix cache/common prefix 信息；
- spec decode metadata；
- multimodal encoder output；
- LoRA 状态；
- PP/TP/DP 通信上下文；
- CUDA graph padding 信息；
- ubatching 信息。

这些都是 GPUModelRunner 在 forward 前准备的。

## 4. SchedulerOutput 到 batch state

第一步是 `_update_states(scheduler_output)`，位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1127`。

它负责把调度器输出同步到 worker 的 persistent batch state：

- 新 request 加入 input batch；
- 已完成 request 从 batch state 移除；
- 更新 request 的 computed token 数；
- 更新 block ids；
- 更新 encoder/multimodal 状态；
- 更新 spec decode 状态；
- 准备本步 token 数数组。

这一步建立了 worker 侧对所有 active request 的本地视图。

## 5. 输入张量准备

关键方法：

| 方法 | 作用 |
|---|---|
| `_prepare_input_ids()` | 准备本步 input token ids |
| `_get_positions()` | 准备 positions |
| `_prepare_inputs()` | 汇总 input_ids、positions、embeds、intermediate_tensors、model_kwargs |
| `_preprocess()` | 处理模型 forward 前的综合准备 |
| `_calc_mrope_positions()` | 多模态 RoPE positions |
| `_calc_xdrope_positions()` | X-D RoPE positions |
| `_calc_spec_decode_metadata()` | spec decode metadata |

`_prepare_inputs()` 位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:1889`。

`_preprocess()` 位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3426`。

## 6. slot mapping

slot mapping 是 Attention 写入/读取 KV cache 的关键。

生成位置：

```text
GPUModelRunner._get_slot_mappings()
```

方法位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3960`。

在 `execute_model()` 中调用位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4244`。

### slot mapping 的含义

```text
本 step 的第 i 个 token
  ↓
属于哪个 request
  ↓
是该 request 的第几个 token
  ↓
对应哪个 logical block
  ↓
对应哪个 physical block id
  ↓
block 内 offset
  ↓
KV cache tensor 中的写入 slot
```

没有 slot mapping，Attention 就不知道当前 key/value 应该写到 KV cache 的哪里。

## 7. attention metadata

构造方法：

```text
GPUModelRunner._build_attention_metadata()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:2208`。

在 `execute_model()` 中调用位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4255`。

metadata 通常包含：

- query lengths；
- sequence lengths；
- max query len；
- max seq len；
- block table；
- slot mapping；
- common prefix blocks；
- cascade attention prefix lens；
- prefill/decode 标记；
- spec decode metadata；
- multimodal prefix-lm 区间；
- DCP/PCP 相关 seq lens。

GPUModelRunner 先构造公共 metadata，然后通过 backend 的 `AttentionMetadataBuilder` 构造 backend-specific metadata。

## 8. batch execution 与 CUDA graph

`_determine_batch_execution_and_padding()` 位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3810`。

它决定：

- 是否使用 CUDA graph；
- 是否 padding 到 capture size；
- 是否使用 ubatching；
- 当前 batch descriptor；
- DP 下 token 数对齐；
- 是否 uniform decode。

这一步影响后续 tensor shape、attention metadata shape、slot mapping padding。

## 9. ForwardContext 的作用

`ForwardContext` 位于：

```text
vllm/forward_context.py
```

GPUModelRunner 在模型 forward 前调用 `set_forward_context(...)`。

在 `execute_model()` 中，调用形态大致是：

```text
with set_forward_context(
    attn_metadata,
    vllm_config,
    num_tokens=...,
    num_tokens_across_dp=...,
    cudagraph_runtime_mode=...,
    batch_descriptor=...,
    ubatch_slices=...,
    slot_mapping=slot_mappings,
):
    model_output = self._model_forward(...)
```

对应位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`。

## 10. ForwardContext 中放了什么

常见内容：

- `attn_metadata`：attention backend metadata；
- `slot_mapping`：KV cache slot mapping；
- `no_compile_layers`：静态 layer registry；
- `vllm_config`；
- `num_tokens`；
- `batch_descriptor`；
- CUDA graph runtime mode；
- DP/TP/PP/ubatch 信息；
- MoE/routed experts 相关状态。

模型层 Attention、MoE、LoRA 会通过这个 context 找回运行时信息。

## 11. 为什么需要 static_forward_context

Attention layer 初始化时会注册：

```text
compilation_config.static_forward_context[prefix] = self
```

位置：`code/vllm/vllm/model_executor/layers/attention/attention.py:397`。

ForwardContext 会拿到这个静态表。forward 中的 custom op 或 helper 可以通过 layer name 找回具体 layer。

这解决了一个问题：

- 编译/图捕获/自定义 op 里不方便把复杂 Python layer 对象作为普通参数传来传去；
- 通过静态 layer registry + layer name，可以在 forward 时查回对应对象。

## 12. model forward

GPUModelRunner 调模型的位置：

```text
model_output = self._model_forward(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4320`。

传入参数包括：

- `input_ids`；
- `positions`；
- `intermediate_tensors`；
- `inputs_embeds`；
- `model_kwargs`。

Attention metadata 和 slot mapping 不作为模型 forward 的普通参数显式传入，而是通过 ForwardContext 被 Attention 层读取。

## 13. forward 后处理

模型返回后：

- 如果是 PP 非最后 rank，可能返回 `IntermediateTensors`；
- 如果是 pooling 模型，走 `_pool()`；
- generation 模型在最后 rank 上取 logits；
- logits 会被保存到 `execute_model_state`；
- 后续 `sample_tokens()` 做采样。

保存状态位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4386`。

## 14. sample_tokens 与 grammar bitmask

`execute_model()` 常返回 `None`，EngineCore 会再调用 `sample_tokens()`。

`sample_tokens()` 位于 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`。

流程：

```text
1. 取 execute_model_state
2. 如有 grammar_output，apply_grammar_bitmask
3. _sample(logits, spec_decode_metadata)
4. _update_states_after_model_execute
5. 返回 ModelRunnerOutput
```

这样结构化输出约束可以插在 logits 之后、采样之前。

## 15. 一句话总结

GPUModelRunner 是运行时和模型层之间的核心转换器：它把 SchedulerOutput 转换成 input tensors、slot mapping 和 attention metadata，用 ForwardContext 传给模型层，然后调用模型 forward 并完成 logits/sampling 后处理。
