# vLLM 模型执行与 Attention 层技术点问答

本文基于本目录已有梳理文档，整理面试、代码走查、技术考察中可能被问到的核心技术点，并给出可直接回答的参考答案。

覆盖范围：

- 模型执行层整体架构
- 模型注册与加载链路
- GPUModelRunner 到 ForwardContext
- Attention 层实现
- AttentionBackend / MetadataBuilder
- KVCacheSpec / KV cache tensor / slot mapping
- 量化、KV cache quant、MoE、LoRA、多模态
- Python 到 CUDA/csrc kernel 调用链
- 常见调试定位问题

---

## 一、整体架构与职责边界

### Q1：vLLM 模型执行与 Attention 层在整体推理链路中的位置是什么？

答：它位于 Worker/GPUModelRunner 之后、底层 CUDA/C++ kernel 之前。

整体链路可以概括为：

```text
API / AsyncLLM
  -> EngineCore
  -> Scheduler
  -> Executor / Worker
  -> GPUModelRunner
  -> model_executor 模型执行层
  -> Attention / MLP / MoE / Norm / Sampler
  -> AttentionBackend / custom ops
  -> csrc CUDA/C++/CPU kernels
```

其中 Scheduler 决定“本步算哪些 request/token/block”，GPUModelRunner 把 SchedulerOutput 转成模型 forward 所需的张量、slot mapping、attention metadata，模型执行层负责具体模型 forward，AttentionBackend/csrc 负责高性能 attention 和 cache kernel。

---

### Q2：`model_executor` 的核心职责是什么？

答：`vllm/model_executor` 不是简单包一层 HuggingFace 模型，它负责 vLLM 模型运行时的核心模型层能力，包括：

1. 模型注册与 architecture 映射；
2. 模型实例化；
3. 权重文件发现、下载与加载；
4. 具体模型结构定义；
5. Attention、MLP、MoE、Norm、Rotary Embedding 等层实现；
6. 量化、LoRA、多模态适配；
7. 向运行时声明 KV cache spec；
8. 在 forward 中消费 ForwardContext 里的 attention metadata、slot mapping、KV cache 等运行时信息。

---

### Q3：为什么说 vLLM 的 Attention 不是普通 PyTorch Attention？

答：因为 vLLM 的 Attention 层不仅执行 attention 计算，还承担运行时桥接职责。

它会：

1. 初始化时选择 AttentionBackend；
2. 创建 backend-specific AttentionImpl；
3. 注册自身到 `static_forward_context`；
4. 声明当前层 KVCacheSpec；
5. forward 时从 ForwardContext 获取当前层 KV cache、attention metadata、slot mapping；
6. 更新 KV cache；
7. 调 backend/csrc 执行 paged attention；
8. 处理 KV cache quant scales、sliding window、MLA、sink、prefix-lm 等特殊逻辑。

因此它是模型层与 vLLM runtime/backend/kernel 之间的 glue layer。

---

### Q4：模型执行层、GPUModelRunner、AttentionBackend 三者分别解决什么问题？

答：

- 模型执行层决定“模型怎么算”：embedding、attention、MLP/MoE、norm、logits 等。
- GPUModelRunner 决定“怎么把调度结果喂给模型”：input_ids、positions、slot mapping、attention metadata、CUDA graph padding、ubatching 等。
- AttentionBackend 决定“attention 如何高性能执行”：paged attention、FlashAttention、FlashInfer、Triton、cache update、量化 KV cache 等。

---

### Q5：本层最重要的几个桥梁对象是什么？

答：主要有四个：

1. `ForwardContext`
   - 连接 GPUModelRunner 和模型层。
   - Attention forward 不直接接收所有 metadata，而是从 ForwardContext 中取。

2. `KVCacheSpec`
   - 模型层向运行时声明当前层需要什么类型、形状、大小的 KV cache。

3. `CommonAttentionMetadata`
   - GPUModelRunner 构造的 batch 级公共 attention metadata。
   - 后续交给 backend-specific MetadataBuilder 转换。

4. `slot_mapping`
   - 告诉 Attention 当前 token 的 key/value 应该写入 KV cache 的哪个 slot。

---

## 二、模型注册与加载链路

### Q6：vLLM 模型加载的总体链路是什么？

答：总体链路是：

```text
VllmConfig / ModelConfig / LoadConfig
  -> get_model_loader(load_config)
  -> loader.load_model(vllm_config, model_config)
  -> architecture registry 查找模型类
  -> 实例化 vLLM 模型类
  -> 加载权重文件
  -> model.load_weights(...)
  -> process_weights_after_loading
  -> GPUModelRunner.model 可用
```

Worker 侧入口是 `Worker.load_model()`，它调用 `GPUModelRunner.load_model()`，再进入 `get_model(vllm_config)` 和具体 loader。

---

### Q7：`get_model_loader()` 和 `get_model()` 分别做什么？

答：

- `get_model_loader(load_config)` 根据 `load_config.load_format` 选择具体 loader。
- `get_model(vllm_config)` 获取 loader，并调用 `loader.load_model(vllm_config=vllm_config, model_config=model_config)` 返回模型实例。

`load_format` 决定权重如何发现、是否下载、是否 streaming load、是否支持量化或分片等。

---

### Q8：vLLM 支持哪些类型的模型加载方式？

答：取决于当前仓库版本，但通常包括：

- 默认 safetensors/bin 加载；
- dummy weights；
- tensorizer；
- bitsandbytes；
- sharded state；
- GGUF；
- runai streamer；
- fastsafetensors；
- remote/model-specific loader。

不同 loader 负责不同权重格式、分片方式、量化格式和 streaming load 方式。

---

### Q9：默认模型 loader 的典型职责是什么？

答：默认 loader 通常负责：

1. 根据模型路径或 HF repo 找权重文件；
2. 识别 safetensors、bin、pt 等格式；
3. 过滤不需要的权重文件；
4. 迭代权重 tensor；
5. 处理 MoE expert parallel 的权重过滤；
6. 调用模型实例的 `load_weights()`；
7. 检查缺失权重或未加载权重；
8. 处理量化权重加载后的状态。

---

### Q10：模型 registry 的作用是什么？

答：`vllm/model_executor/models/registry.py` 负责把 HuggingFace config 中的 architecture 名称映射到 vLLM 内部模型类。

例如：

```text
config.architectures = ["LlamaForCausalLM"]
  -> registry 查找
  -> vLLM 的 LlamaForCausalLM 实现
```

它还会处理模型是否支持 generation、pooling、multimodal、LoRA、pipeline parallel、特定任务、lazy import 等能力。

---

### Q11：vLLM 模型类通常需要实现哪些接口？

答：常见接口包括：

- `forward()`；
- `load_weights()`；
- `compute_logits()`；
- sampler 相关能力；
- pooling 相关能力；
- embedding 相关能力；
- multimodal 输入处理能力；
- pipeline parallel intermediate tensor 支持；
- `get_input_embeddings()`；
- `make_empty_intermediate_tensors()` 等辅助能力。

不同任务类型和模型族会实现不同组合。

---

### Q12：`load_weights()` 需要处理哪些复杂情况？

答：`load_weights()` 通常接收 `(name, tensor)` 迭代器，并将权重落到模型参数上。它需要处理：

1. 权重名与参数名匹配；
2. fused qkv/proj/mlp 权重；
3. Tensor Parallel 切分；
4. Pipeline Parallel 下本 rank 只持有部分层；
5. Expert Parallel 下 MoE expert 权重过滤；
6. 量化权重格式、scale、packing；
7. tied embedding / lm_head；
8. 返回已加载参数集合，用于缺失权重检查。

---

### Q13：`process_weights_after_loading()` 有什么作用？

答：很多层在权重加载后还需要后处理，例如：

- 量化权重后处理；
- KV cache scale 默认值设置；
- FP8 scale reshape；
- MoE 权重重排；
- LoRA 相关状态处理；
- backend-specific weight preprocess。

Attention 层也有自己的 `process_weights_after_loading()`，用于处理 KV cache quant scale 等。

---

### Q14：模型加载和并行配置有什么关系？

答：模型加载不是单卡视角，会受多种并行配置影响：

| 并行方式 | 对加载的影响 |
|---|---|
| Tensor Parallel | 权重按 head/hidden dim 切分 |
| Pipeline Parallel | 不同 rank 只加载部分层 |
| Expert Parallel | MoE experts 只加载本 rank 负责部分 |
| Data Parallel | 每个 DP replica 加载一份或一组 shard |
| Quantization | 权重、scale、packing 格式不同 |

所以同一个模型在不同 parallel_config 下，每个 worker 实际加载的权重可能不同。

---

### Q15：模型加载和任务类型有什么关系？

答：vLLM 支持 generation、pooling、embedding、scoring、classification、multimodal generation、encoder-decoder 等任务。

模型 registry 和接口层需要判断模型支持哪些任务。GPUModelRunner 后续也会根据 runner type 决定：

- 是否 compute logits；
- 是否 sample；
- 是否走 pooling；
- 是否需要 encoder cache；
- 是否需要 multimodal encoder。

---

## 三、GPUModelRunner 到 ForwardContext

### Q16：GPUModelRunner 的核心定位是什么？

答：GPUModelRunner 是运行时和模型层之间的核心转换器。

它把 SchedulerOutput 转换为模型 forward 需要的：

- input_ids；
- positions；
- inputs_embeds；
- intermediate_tensors；
- slot mapping；
- attention metadata；
- CUDA graph padding 信息；
- ubatching 信息；
- LoRA 状态；
- spec decode metadata；
- multimodal encoder output。

然后通过 `set_forward_context()` 把 runtime metadata 暴露给模型层，再调用模型 forward。

---

### Q17：为什么 GPUModelRunner 不只是简单调用模型 forward？

答：因为 vLLM 的模型 forward 需要大量调度和运行时信息，而不仅是 `input_ids`。

例如：

- 每个 request 本步执行几个 token；
- 每个 token 的 position；
- 每个 token 对应 KV cache 的 slot；
- block table；
- attention backend metadata；
- prefix cache/common prefix 信息；
- spec decode metadata；
- multimodal encoder output；
- LoRA adapter 状态；
- CUDA graph padding；
- ubatching；
- PP/TP/DP 通信上下文。

这些信息都要在模型 forward 前准备好。

---

### Q18：`GPUModelRunner.execute_model()` 的总体流程是什么？

答：高层流程：

```text
1. 检查 execute_model_state，确保上一次 forward 后已 sample
2. 处理 routed experts buffer
3. 处理 spec decode ngram_gpu 的 SchedulerOutput copy
4. 处理 KV connector preemptions
5. 读取本步 total_num_scheduled_tokens
6. preprocess：
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

---

### Q19：`_update_states(scheduler_output)` 做什么？

答：它把 SchedulerOutput 同步到 worker 的 persistent batch state。

主要包括：

- 新 request 加入 input batch；
- 已完成 request 从 batch state 移除；
- 更新 request 的 computed token 数；
- 更新 block ids；
- 更新 encoder/multimodal 状态；
- 更新 spec decode 状态；
- 准备本步 token 数数组。

这一步建立 worker 侧对所有 active request 的本地视图。

---

### Q20：GPUModelRunner 如何准备输入张量？

答：主要方法包括：

| 方法 | 作用 |
|---|---|
| `_prepare_input_ids()` | 准备本步 input token ids |
| `_get_positions()` | 准备 positions |
| `_prepare_inputs()` | 汇总 input_ids、positions、embeds、intermediate_tensors、model_kwargs |
| `_preprocess()` | 模型 forward 前综合准备 |
| `_calc_mrope_positions()` | 多模态 RoPE positions |
| `_calc_xdrope_positions()` | X-D RoPE positions |
| `_calc_spec_decode_metadata()` | spec decode metadata |

这些方法将 SchedulerOutput 和 persistent batch state 转换为模型可消费的输入。

---

### Q21：slot mapping 是什么？为什么重要？

答：slot mapping 是当前 batch 中 token 到 KV cache 写入位置的映射。

可以理解为：

```text
本 step 的第 i 个 token
  -> 属于哪个 request
  -> 是该 request 的第几个 token
  -> 对应哪个 logical block
  -> 对应哪个 physical block id
  -> block 内 offset
  -> KV cache tensor 中的写入 slot
```

如果没有 slot mapping，Attention 就不知道当前 key/value 应该写入 KV cache 的哪里。

简化公式：

```text
slot = physical_block_id * block_size + offset_in_block
```

实际实现还要考虑 cache group、padding、CUDA graph、ubatching、Mamba/hybrid layout、sliding window、spec decode lookahead 等。

---

### Q22：attention metadata 包含哪些信息？

答：GPUModelRunner 构造的 attention metadata 通常包含：

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
- DCP/PCP local seq lens。

GPUModelRunner 先构造公共 metadata，再由 backend 的 `AttentionMetadataBuilder` 转换成 backend-specific metadata。

---

### Q23：`_determine_batch_execution_and_padding()` 决定什么？

答：它决定当前 batch 的执行方式，包括：

- 是否使用 CUDA graph；
- 是否 padding 到 capture size；
- 是否使用 ubatching；
- 当前 batch descriptor；
- DP 下 token 数对齐；
- 是否 uniform decode。

这一步会影响后续 tensor shape、attention metadata shape、slot mapping padding。

---

### Q24：ForwardContext 的作用是什么？

答：ForwardContext 是 GPUModelRunner 和模型层之间的运行时上下文桥梁。

GPUModelRunner 在模型 forward 前调用 `set_forward_context(...)`，把 attention metadata、slot mapping、vllm_config、num_tokens、batch descriptor、CUDA graph runtime mode、ubatch_slices 等放入上下文。

模型层 Attention、MoE、LoRA 等不需要通过普通 forward 参数层层传递这些信息，而是在 forward 中从 ForwardContext 获取。

---

### Q25：ForwardContext 中通常放了什么？

答：常见内容包括：

- `attn_metadata`；
- `slot_mapping`；
- `no_compile_layers` 静态 layer registry；
- `vllm_config`；
- `num_tokens`；
- `batch_descriptor`；
- CUDA graph runtime mode；
- DP/TP/PP/ubatch 信息；
- MoE/routed experts 相关状态。

---

### Q26：为什么需要 `static_forward_context`？

答：因为在 torch compile、CUDA graph、自定义 op 场景中，不适合把复杂 Python layer 对象作为普通参数在模型 forward 中层层传递。

vLLM 让 Attention layer 初始化时注册：

```text
compilation_config.static_forward_context[prefix] = self
```

forward 中通过 layer name 找回 layer 对象，从而获得该层的 backend、impl、KV cache、quant scale 等信息。

这兼顾了运行时灵活性和编译/图捕获友好性。

---

### Q27：为什么 `execute_model()` 后还要 `sample_tokens()`？

答：generation 路径中，`execute_model()` 主要做 forward 和 logits 准备，通常会把 logits、metadata 等保存到 `execute_model_state`，然后返回 `None`。

EngineCore 会随后调用 `sample_tokens(grammar_output)`，在采样前应用 grammar bitmask，再执行 sampling、logprobs、状态更新，最终返回 ModelRunnerOutput。

这样结构化输出约束可以插在 logits 之后、采样之前，同时也支持 async scheduling 和 PP 场景。

---

## 四、Attention 层核心实现

### Q28：`Attention` 类的核心职责是什么？

答：`Attention` 是模型里的 `nn.Module`，负责一层 attention 的运行时逻辑。

它主要负责：

1. 保存 attention 结构参数；
2. 选择 attention backend；
3. 创建 backend implementation；
4. 初始化 KV cache quant scale；
5. 保存/绑定 KV cache tensor；
6. 注册自身到 `static_forward_context`；
7. 声明当前层 KVCacheSpec；
8. forward 时获取 ForwardContext；
9. 更新 KV cache；
10. 调用 backend 计算 attention；
11. 处理量化、sink、prefix-lm、kv-sharing 等特殊逻辑。

---

### Q29：Attention 初始化时保存哪些关键参数？

答：常见参数包括：

| 参数 | 作用 |
|---|---|
| `num_heads` | query heads 数 |
| `head_size` | 每个 head 维度 |
| `scale` | attention scale |
| `num_kv_heads` | KV heads 数，GQA/MQA 时小于 query heads |
| `alibi_slopes` | ALiBi 参数 |
| `sliding_window` | sliding window attention 大小 |
| `kv_cache_dtype` | KV cache dtype |
| `blocksparse_params` | block sparse attention 参数 |
| `logits_soft_cap` | logits soft cap |
| `attn_type` | decoder/self/cross/prefix 等 attention 类型 |
| `prefix` | layer name，用于 context 查找 |
| `use_mla` | 是否 MLA attention |
| `sinks` | attention sink 参数 |
| `per_layer_sliding_window` | 每层 sliding window |

这些参数会影响 backend 选择、KVCacheSpec、metadata 和 kernel 路径。

---

### Q30：Attention backend 选择会考虑哪些条件？

答：backend 选择会考虑：

- 当前平台：CUDA/ROCm/CPU/XPU；
- head size；
- dtype；
- KV cache dtype；
- block size；
- 是否 MLA；
- 是否 sink attention；
- 是否支持 mm prefix；
- 是否需要 non-causal；
- 是否支持 per-head quant scales；
- 是否支持 batch invariance；
- 是否支持 KV connector；
- 用户是否指定 attention backend。

最终通过 V1 attention selector 和平台能力返回合适的 AttentionBackend。

---

### Q31：`AttentionImpl` 是什么？

答：AttentionBackend 选定后，会创建 backend 对应的 `AttentionImpl`。

`AttentionImpl` 是单层 attention 的具体实现对象，负责执行 backend-specific forward。它可能调用：

- Triton kernel；
- FlashAttention；
- FlashInfer；
- ROCm kernel；
- CPU kernel；
- vLLM paged attention op；
- fused cache update op；
- torch custom op。

Attention 自身不直接写 CUDA kernel，而是通过 impl/backend 调用底层实现。

---

### Q32：Attention 为什么要注册到 `static_forward_context`？

答：为了让 forward/custom op 可以通过稳定的 layer name 找回 Attention layer 对象。

注册形式：

```text
compilation_config.static_forward_context[prefix] = self
```

这样在 torch compile、CUDA graph 或 custom op 中，可以通过 layer name 找到该层的 backend、impl、KV cache、quant scale、metadata 等，不需要把复杂 Python 对象作为普通参数传递。

---

### Q33：`Attention.forward()` 的主流程是什么？

答：高层流程：

```text
1. 如需要，计算 KV scales
2. reshape query/key/value 到 backend 需要的形状
3. 判断当前平台是否使用 opaque custom op
4. 如果 backend 不在 forward 中更新 KV cache：
   -> 调 unified_kv_cache_update
5. 调 unified_attention_with_output 或直接 self.impl.forward
6. 返回 attention output
```

其中 KV cache update 和 attention compute 是否融合，取决于 backend 能力。

---

### Q34：KV cache update 是如何完成的？

答：当 backend 不在 forward 内部更新 KV cache 时，Attention.forward 会调用 `unified_kv_cache_update()`。

它会：

1. 通过 `get_attention_context(layer_name)` 获取当前层 context；
2. 找到当前层 KV cache tensor；
3. 找到当前 token 的 slot mapping；
4. 调 backend 的 `do_rope_and_kv_cache_update()` 或相关实现；
5. 把本步 key/value 写入 KV cache。

---

### Q35：`unified_attention_with_output()` 做什么？

答：它负责执行实际 attention 计算，并将结果写入 output。

流程：

```text
Attention.forward
  -> unified_attention_with_output
  -> get_attention_context
  -> 找到当前 layer 的 metadata
  -> 找到 KV cache
  -> self.impl.forward(...)
  -> backend kernel/custom op
```

它是 Attention.forward 到 backend impl 的统一入口之一。

---

### Q36：`get_attention_context()` 返回什么？

答：它从 ForwardContext 中取出当前层运行时上下文，通常包括：

- 当前 layer 的 Attention 对象；
- 当前 layer 的 KV cache；
- 当前 batch 的 attention metadata；
- 当前 batch 的 slot mapping；
- backend-specific metadata。

这是 Attention 与 GPUModelRunner/ForwardContext 衔接的核心方法。

---

### Q37：`Attention.get_kv_cache_spec()` 的作用是什么？

答：它根据当前 Attention 层属性生成 KV cache 规格，告诉运行时该层需要什么 cache。

它会考虑：

- full attention；
- sliding window attention；
- MLA attention；
- encoder-only attention；
- cross attention；
- sink attention；
- quantized KV cache；
- attention type；
- head size / num kv heads；
- backend layout；
- block size；
- non-causal 等。

这些 spec 会被 EngineCore/Worker 收集，用于计算 KV cache 内存、分组、block size 和 tensor shape。

---

### Q38：Attention 中 KV quant scales 如何处理？

答：相关函数包括：

- `_init_kv_cache_quant()`：初始化 q/k/v/prob scale；
- `maybe_calc_kv_scales()`：必要时运行时计算 scale；
- `Attention.calc_kv_scales()`：计算当前层 KV scales；
- `Attention.process_weights_after_loading()`：权重加载后处理 scale。

它们处理 checkpoint 中加载 scale、默认 scale、FP8 KV cache、per-token-head scale、prob scale 等。

---

### Q39：Attention 中 custom op 注册有什么意义？

答：`attention.py` 中通过 `direct_register_custom_op` 注册了：

- `maybe_calc_kv_scales`
- `unified_kv_cache_update`
- `unified_attention_with_output`

这些 op 通过 `torch.ops.vllm.*` 暴露给 PyTorch graph/compile/runtime，使 attention forward 可以更好地配合 torch compile、CUDA graph 和 custom dispatch，同时连接 Python runtime 与底层 kernel。

---

## 五、AttentionBackend 与 MetadataBuilder

### Q40：`Attention`、`AttentionBackend`、`AttentionImpl`、`AttentionMetadataBuilder` 的关系是什么？

答：四者关系：

```text
Attention 层
  -> 选择/持有 AttentionBackend
  -> AttentionBackend 创建 AttentionImpl
  -> GPUModelRunner 构造 CommonAttentionMetadata
  -> AttentionMetadataBuilder 转成 backend-specific metadata
  -> AttentionImpl.forward 执行实际 attention
```

职责拆分：

| 对象 | 职责 |
|---|---|
| `Attention` | 模型层 nn.Module，持有 backend/impl 并调用它们 |
| `AttentionBackend` | 描述 backend 能力、KV cache shape/layout、支持配置 |
| `AttentionImpl` | 单层 attention 的实际实现对象 |
| `AttentionMetadataBuilder` | 把公共 batch metadata 转成 backend-specific metadata |
| `CommonAttentionMetadata` | GPUModelRunner 构造的公共 attention metadata |

---

### Q41：`AttentionBackend` 需要声明哪些能力？

答：主要包括三类：

1. backend identity：
   - `get_name()`
   - `get_impl_cls()`
   - `get_builder_cls()`

2. KV cache shape/layout：
   - `get_kv_cache_shape()`
   - `get_kv_cache_block_dim()`
   - `get_kv_cache_stride_order()`
   - `get_required_kv_cache_layout()`

3. 能力检查：
   - `supports_head_size()`
   - `supports_dtype()`
   - `supports_kv_cache_dtype()`
   - `supports_block_size()`
   - `supports_attn_type()`
   - `supports_compute_capability()`
   - `supports_combination()`
   - `validate_configuration()`

此外还有 MLA、sink、non-causal、mm prefix、KV connector、batch invariance 等特性声明。

---

### Q42：`get_attn_backend()` 的选择流程是什么？

答：简化流程：

```text
Attention.__init__
  -> get_attn_backend(...)
  -> 构造 AttentionSelectorConfig
  -> _cached_get_attn_backend(...)
  -> current_platform.get_attn_backend_cls(...)
  -> backend.validate_configuration(...)
  -> 返回 AttentionBackend class
```

`current_platform.get_attn_backend_cls` 根据 CUDA/ROCm/CPU/XPU 等平台能力选择对应 backend。

---

### Q43：常见 AttentionBackend 类型有哪些？

答：常见 backend 类型包括：

- FlashAttention；
- FlashInfer；
- Triton；
- ROCm attention；
- CPU attention；
- MLA attention；
- sparse attention；
- no attention；
- platform-specific optimized backend。

实际可用 backend 取决于当前环境、安装依赖、平台能力和模型配置。

---

### Q44：`CommonAttentionMetadata` 包含哪些信息？

答：它是 GPUModelRunner 生成的公共 attention metadata，通常包含：

- `query_start_loc`；
- `seq_lens`；
- `block_table_tensor`；
- `slot_mapping`；
- `positions`；
- `num_computed_tokens`；
- `is_prefilling`；
- `max_query_len`；
- `max_seq_len`；
- multimodal prefix ranges；
- DCP/PCP local seq lens；
- common prefix/cascade attention 信息。

它是 backend-specific metadata 的输入。

---

### Q45：`AttentionMetadataBuilder` 的职责是什么？

答：它接收 `CommonAttentionMetadata`，根据具体 backend 需求构造 backend-specific metadata。

主要职责：

1. 处理 prefill/decode 差异；
2. 构造当前 batch backend metadata；
3. 处理 CUDA graph capture metadata；
4. 更新 block table；
5. 判断是否使用 cascade attention；
6. 为 speculative drafting 构造 metadata。

常见方法：

- `build()`；
- `update_block_table()`；
- `build_for_cudagraph_capture()`；
- `build_for_drafting()`；
- `use_cascade_attention()`；
- `get_cudagraph_support()`。

---

### Q46：backend 选择会影响哪些方面？

答：backend 一旦选定，会影响：

1. KV cache tensor shape；
2. KV cache layout/stride；
3. metadata builder 类型；
4. attention forward kernel；
5. 是否 forward 内更新 KV cache；
6. 是否支持 CUDA graph；
7. 是否支持 prefix/multimodal/sink/non-causal；
8. 是否支持 KV quant；
9. 是否支持特定 head size/block size。

因此 backend 选择错误可能导致初始化失败、KV cache shape 不匹配、kernel 报错、输出错误或性能下降。

---

### Q47：AttentionBackend 和 GPUModelRunner 的关系是什么？

答：AttentionBackend 不只在 Attention.forward 中生效，还会影响 GPUModelRunner 的 KV cache 分配和 metadata 构造。

GPUModelRunner 会使用 backend 提供的信息：

- `backend.get_kv_cache_shape()`；
- `backend.get_kv_cache_stride_order()`；
- `backend.get_builder_cls()`；
- `metadata_builder.build()`。

所以 backend 从 KV cache 初始化阶段就影响运行时布局。

---

### Q48：MLA AttentionImpl 有什么特殊性？

答：MLA AttentionImpl 面向 DeepSeek 等 MLA 架构，提供不同于普通 MHA 的 forward 接口，例如：

- `forward_mha()`；
- `forward_mqa()`；
- `do_kv_cache_update()`。

MLA/sparse MLA 需要特殊 KV cache spec、metadata 和 kernel 路径，因此会有专门的 impl。

---

## 六、KVCacheSpec、KV Tensor、SlotMapping

### Q49：KV cache 的总体链路是什么？

答：总体链路是：

```text
Attention layer
  -> get_kv_cache_spec()
  -> KVCacheSpec / AttentionSpec
  -> EngineCore 收集与合并
  -> KVCacheConfig / KVCacheGroupSpec
  -> Worker/GPUModelRunner 分配 KV cache tensor
  -> KV cache tensor
  <- slot mapping / block table
  <- SchedulerOutput 中的 block ids
  <- Scheduler + KVCacheManager 分配逻辑 block
```

模型层声明需求，运行时合并配置，worker 分配物理 tensor，scheduler/block table/slot mapping 将 token 映射到物理 KV cache。

---

### Q50：`KVCacheSpec` 是什么？

答：`KVCacheSpec` 表示某类 cache 的规格，提供：

- `page_size_bytes()`：每页/block 占多少字节；
- `storage_block_size()`：实际存储 block size；
- `max_memory_usage_bytes()`：最大内存占用估算；
- `copy_with_new_block_size()`：复制并替换 block size；
- `merge()`：同类 spec 合并；
- `is_uniform_with_collection()`：判断是否可归为 uniform group。

它是模型层对运行时 cache 需求的声明。

---

### Q51：`AttentionSpec` 包含哪些信息？

答：`AttentionSpec` 是 attention 层 KV cache spec 的基类，通常包含：

- block size；
- num kv heads；
- head size；
- dtype；
- kv quant mode；
- attention type；
- non-causal 标记；
- per-layer 特殊配置。

它的 `page_size_bytes()` 会根据 head 数、head size、dtype、block size、量化模式计算一页 KV cache 占用。

---

### Q52：常见 KVCacheSpec 类型有哪些？

答：常见类型包括：

| 类型 | 含义 |
|---|---|
| `FullAttentionSpec` | 普通 full causal attention |
| `TQFullAttentionSpec` | TurboQuant-aware page size |
| `MLAAttentionSpec` | MLA attention |
| `HiddenStateCacheSpec` | hidden state cache 变体 |
| `ChunkedLocalAttentionSpec` | chunked local attention |
| `SlidingWindowSpec` | sliding window attention |
| `SlidingWindowMLASpec` | sliding window MLA |
| `MambaSpec` | Mamba/SSM state cache |
| `EncoderOnlyAttentionSpec` | encoder-only attention |
| `CrossAttentionSpec` | encoder-decoder cross attention |
| `SinkFullAttentionSpec` | attention sink |

---

### Q53：`KVQuantMode` 是什么？

答：`KVQuantMode` 描述 KV cache 量化模式，常见模式有：

- `NONE`：不量化；
- `FP8_PER_TENSOR`：FP8 per tensor；
- `INT8_PER_TOKEN_HEAD`：INT8 per token/head；
- `FP8_PER_TOKEN_HEAD`：FP8 per token/head；
- `NVFP4`：NVFP4 cache。

它会影响 KV cache dtype、page size、scale tensor、kernel 选择和 cache update 方式。

---

### Q54：KVCacheSpec 如何影响内存估算？

答：`page_size_bytes()` 直接影响 KV cache 可容纳 block 数。

普通 full attention 的 page size 主要由以下因素决定：

```text
block_size
num_kv_heads
head_size
key + value 两份 cache
dtype bytes
quant scale overhead
```

MLA、NVFP4、sliding window、Mamba 等会改变计算方式和 page size。

---

### Q55：`KVCacheConfig`、`KVCacheGroupSpec`、`KVCacheTensor` 分别是什么？

答：

- `KVCacheTensor`：描述实际 KV cache tensor 规格；
- `KVCacheGroupSpec`：描述一个 cache group；
- `KVCacheConfig`：最终运行时 cache 配置，包括 cache groups、num blocks、tensor specs、是否有 Mamba layers、是否需要 zeroing 等。

它们是多个 layer 的 KVCacheSpec 被收集、合并、分组后的运行时配置。

---

### Q56：KV cache tensor 在哪里分配？

答：worker 侧分配入口：

```text
Worker.initialize_from_config()
  -> GPUModelRunner.initialize_kv_cache(kv_cache_config)
```

GPUModelRunner 会：

- 分配底层 tensor；
- 按 backend layout reshape；
- 遍历 attention group；
- 初始化 attention backend；
- 初始化 metadata builders；
- 将 KV cache 绑定到 forward context / model runner。

---

### Q57：block id 与物理 KV cache tensor 的关系是什么？

答：Scheduler/KVCacheManager 管理逻辑 block 分配：

```text
request -> block ids
```

KV cache tensor 是一大块预分配显存：

```text
[num_blocks, ... cache layout ...]
```

block id 就是访问这个大 tensor 的 page/block 索引。但具体 token 写到哪个位置，还需要 slot mapping。

---

### Q58：block table 是什么？

答：block table 告诉 attention kernel：某个 request 的逻辑第几个 block 对应哪个物理 block id。

attention kernel 在计算历史 attention 时，会根据 block table 找到历史 K/V 所在的 pages。

slot mapping 主要负责当前新 token 写入位置，block table 主要负责历史 K/V 读取位置。

---

### Q59：KV cache update 和 attention read 分别如何发生？

答：

写入当前 step 新 token 的 key/value：

```text
unified_kv_cache_update
  -> backend.do_rope_and_kv_cache_update
  -> cache kernel
  -> 根据 slot_mapping 写入 KV cache
```

读取历史 K/V 并计算 attention：

```text
unified_attention_with_output
  -> self.impl.forward
  -> paged attention kernel
  -> 根据 metadata/block table 读取历史 K/V
```

---

### Q60：逻辑 KV cache 和物理 KV cache 为什么分离？

答：因为 vLLM 需要高效支持动态请求、prefix cache、preemption、分页复用和连续 batch 推理。

- Scheduler/KVCacheManager 管逻辑 block 分配。
- Worker/GPUModelRunner 管物理 tensor 分配。
- block table 连接 request 的逻辑 block 与物理 block。
- slot mapping 连接当前 token 与物理写入 slot。

这种设计使 KV cache 可以像分页内存一样被复用和管理。

---

## 七、量化、MoE、LoRA、多模态

### Q61：权重量化会影响哪些层级？

答：权重量化主要影响：

1. 权重文件加载方式；
2. 参数 tensor 的 shape/packing；
3. scale/zero-point 的保存和加载；
4. linear kernel 的选择；
5. MoE kernel 的选择；
6. 是否支持 fused operation；
7. CUDA graph / torch compile 兼容性。

相关路径包括 `model_executor/layers/quantization/`、`linear.py`、`fused_moe/`、model_loader 和 csrc quant kernel。

---

### Q62：KV cache 量化和权重量化有什么区别？

答：权重量化影响模型参数存储和 linear/MoE 计算；KV cache 量化影响历史 key/value 的存储和 attention 读取。

KV cache 量化会影响：

- KV cache dtype；
- KVQuantMode；
- page size；
- scale tensor；
- cache update kernel；
- attention backend 是否支持；
- csrc kernel 调用路径。

因此二者发生在不同对象和不同执行阶段。

---

### Q63：KV cache 量化的大体流程是什么？

答：流程：

```text
Attention 初始化
  -> _init_kv_cache_quant
  -> 加载 checkpoint 中的 scale 或设置默认 scale
  -> get_kv_cache_spec 中声明 kv_quant_mode
  -> KV cache tensor 按量化模式分配
  -> forward 中 maybe_calc_kv_scales
  -> unified_kv_cache_update 使用 scale 写入 quantized KV
```

---

### Q64：MoE 在加载阶段有什么特殊处理？

答：MoE expert 权重通常很大，在 expert parallel 下每个 rank 只需要加载部分 expert。

默认 loader 会根据 expert parallel rank 过滤权重，避免每个 rank 加载全部 expert。

典型流程：

```text
model_loader
  -> expert parallel filter
  -> 只加载本 rank 负责的 expert weights
```

---

### Q65：MoE forward 的基本流程是什么？

答：MoE forward 通常包括：

```text
hidden states
  -> router / gating
  -> top-k expert selection
  -> token dispatch
  -> expert FFN kernel
  -> combine outputs
```

vLLM 中 fused MoE 层会尽量调用高性能 fused kernel 或 custom op，并且可能涉及 expert parallel 和 EPLB。

---

### Q66：MoE 和 ForwardContext 有什么关系？

答：MoE runner 也会使用 ForwardContext 和静态 layer 注册机制。

和 Attention 类似：

```text
layer 初始化时注册到 static_forward_context
  -> forward/custom op 中通过 layer_name 找回 layer
  -> 执行 backend/kernel
```

这种模式让 Attention、MoE、LoRA 都能在 torch compile/custom op 场景中保留 Python layer 对象与运行时 metadata 的连接。

---

### Q67：LoRA 在 vLLM 中如何运行时管理？

答：LoRA 可以动态添加/移除，链路通常是：

```text
AsyncLLM.add_lora/remove_lora
  -> EngineCore
  -> Executor.add_lora/remove_lora
  -> Worker/GPUModelRunner
  -> 模型层 LoRA manager
```

Executor 抽象中通常有：

- `add_lora()`；
- `remove_lora()`；
- `list_loras()`；
- `pin_lora()`。

---

### Q68：LoRA 主要影响 Attention 的哪部分？

答：LoRA 主要影响 linear 层，而不是 attention kernel 本身。

但是 attention 中的 q/k/v/o projection 都是 linear，因此 LoRA 可以作用到 attention 投影层。

运行时通过 LoRA manager、base_linear、request lora id、active adapter mapping 等机制，让同一个 batch 内不同 request 可以使用不同 adapter。

---

### Q69：LoRA 为什么也要用 static_forward_context？

答：LoRA 层注册到静态上下文后，可以通过 `ForwardContext.no_compile_layers[layer_name]` 找到当前 layer 对象并执行 adapter 逻辑。

这样可以同时支持：

- 动态 LoRA；
- torch compile；
- CUDA graph；
- 多请求混合 LoRA；
- batch 内不同 request 使用不同 adapter。

---

### Q70：多模态能力影响哪些层级？

答：多模态影响：

1. 输入处理：图片/视频/音频被预处理成 multimodal embeddings 或 encoder inputs；
2. Scheduler：需要 encoder compute budget 和 encoder cache；
3. GPUModelRunner：执行 multimodal encoder 或使用 cached encoder outputs；
4. 模型 forward：需要把 text tokens 和 multimodal embeddings 对齐；
5. Attention metadata：prefix-lm / mm prefix 场景需要局部非因果 attention 区间。

---

### Q71：多模态和 Attention metadata 有什么关系？

答：`CommonAttentionMetadata` 中包含 multimodal prefix 相关字段，例如 mm request/doc ranges。

用途：

- PrefixLM 场景中，某些 prefix token 可以双向 attention；
- 文本 token 仍保持因果 attention；
- backend 需要知道哪些区间是非因果/双向区间。

如果 backend 不支持 `mm_prefix` 或 non-causal，则需要回退或禁用某些优化。

---

### Q72：多模态和 KV cache 有什么关系？

答：多模态模型可能带来额外 cache：

- encoder cache；
- multimodal feature cache；
- cross attention cache；
- decoder self-attention KV cache。

因此运行时不只处理 decoder-only full attention，还要处理 encoder-only、cross attention、prefix/non-causal 等 spec。

---

### Q73：量化、MoE、LoRA、多模态如何共同影响 backend 选择？

答：Attention backend 选择需要同时满足：

- dtype；
- KV cache dtype；
- quant mode；
- head size；
- block size；
- sliding window；
- MLA；
- sink；
- sparse；
- non-causal；
- mm prefix；
- per-head scales；
- batch invariance；
- KV connector。

所以同一个模型在不同配置下可能选择不同 attention backend。

---

## 八、CUDA/csrc Kernel 调用链

### Q74：Python Attention 到 CUDA kernel 的总体调用链是什么？

答：总体链路：

```text
GPUModelRunner.execute_model
  -> set_forward_context
  -> 模型 forward
  -> Attention.forward
  -> unified_kv_cache_update / unified_attention_with_output / backend direct call
  -> torch.ops.vllm.* 或 backend op
  -> C++ torch binding
  -> CUDA / ROCm / CPU kernel
  -> KV cache update / paged attention / quantized cache / MoE 等
```

---

### Q75：`torch.ops.vllm.*` 是怎么来的？

答：Python 层通过 `direct_register_custom_op` 注册 custom op，例如：

- `maybe_calc_kv_scales`
- `unified_kv_cache_update`
- `unified_attention_with_output`

这些 op 通过 `torch.ops.vllm.*` 暴露给 PyTorch runtime/compile。

底层 csrc 中通过 `TORCH_LIBRARY` 或类似 binding 将 C++/CUDA 函数注册成 PyTorch 可调用 op。

---

### Q76：`get_attention_context()` 在 Python 到 kernel 链路中起什么作用？

答：`unified_kv_cache_update()` 和 `unified_attention_with_output()` 都会通过 `get_attention_context(layer_name)` 取出：

- Attention layer 对象；
- 当前层 KV cache；
- backend-specific attention metadata；
- slot mapping。

这一步把 layer name 映射回真实 runtime 对象，是 Python layer/custom op 与运行时状态之间的桥。

---

### Q77：csrc 中 attention kernel 主要看哪些文件？

答：重点关注：

```text
csrc/libtorch_stable/attention/paged_attention_v1.cu
csrc/libtorch_stable/attention/paged_attention_v2.cu
csrc/libtorch_stable/attention/attention_kernels.cuh
csrc/attention/attention_generic.cuh
csrc/attention/attention_dtypes.h
```

这些文件负责 paged attention 计算、block table 间接寻址、Q/K/V dtype 处理、head size/block size 模板化、softmax/reduction、value 聚合等。

---

### Q78：csrc 中 cache kernel 主要看哪些文件？

答：重点关注：

```text
csrc/libtorch_stable/cache_kernels.cu
csrc/libtorch_stable/cache_kernels_fused.cu
csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu
csrc/cache.h
```

这些文件负责：

- reshape and cache；
- key/value 写入 KV cache；
- fused RoPE + KV cache update；
- FP8/NVFP4 cache 写入；
- scale 处理；
- cache copy/swap/zero 等操作。

---

### Q79：Paged Attention kernel 通常需要哪些输入？

答：通常需要：

- query tensor；
- KV cache tensor；
- block table；
- sequence lengths；
- scale；
- head mapping；
- max context len；
- alibi/sink/sliding window 等可选参数；
- KV cache dtype/scale。

其中 block table 和 seq lens 来自 attention metadata，KV cache tensor 来自 worker 初始化，query 来自模型当前层。

---

### Q80：KV cache update kernel 通常需要哪些输入？

答：通常需要：

- key tensor；
- value tensor；
- KV cache tensor；
- slot mapping；
- scale；
- optional RoPE position；
- quant mode；
- dtype/layout 信息。

slot mapping 是写入位置的关键。

---

### Q81：哪些 backend 会把 KV cache update 和 attention forward 融合？

答：不同 backend 策略不同：

1. 有些 backend forward 中同时做：

```text
write K/V to cache + attention compute
```

2. 有些 backend 分两步：

```text
unified_kv_cache_update
  -> unified_attention_with_output
```

Attention.forward 会根据 backend 的 `forward_includes_kv_cache_update` 等能力决定路径。

---

### Q82：CUDA graph 对 attention kernel 调用有什么影响？

答：CUDA graph 要求 shape 较稳定，因此 GPUModelRunner 会：

- 判断当前 batch 是否适合 graph；
- padding token/request 到 capture size；
- 构造 padded slot mapping；
- 构造 capture-compatible metadata；
- 使用静态 buffer。

如果出现 shape 或越界问题，要同时检查 batch padding、slot mapping、attention metadata 和 builder 的 cudagraph capture metadata。

---

## 九、调试与故障定位

### Q83：Attention backend 选错时如何排查？

答：优先检查：

```text
Attention.get_attn_backend
v1/attention/selector.py
v1/attention/backends/registry.py
AttentionBackend.supports_*
AttentionBackend.validate_configuration
```

重点确认：

- head size 是否支持；
- dtype 是否支持；
- KV cache dtype 是否支持；
- block size 是否支持；
- attention type 是否支持；
- 当前平台是否有对应依赖；
- 是否被环境变量强制指定 backend；
- 是否需要 mm prefix/non-causal/sink/MLA 等特殊能力。

---

### Q84：KV cache shape/layout 错误如何排查？

答：优先检查：

```text
Attention.get_kv_cache_spec
KVCacheSpec.page_size_bytes
AttentionBackend.get_kv_cache_shape
GPUModelRunner.initialize_kv_cache
GPUModelRunner._reshape_kv_cache_tensors
```

重点确认：

- spec 中 head size、num kv heads 是否正确；
- MLA/sliding window/quant mode 是否正确；
- backend required layout 是否匹配；
- worker 分配 tensor shape 是否和 kernel 期望一致；
- cache group 是否正确合并。

---

### Q85：slot mapping 越界如何排查？

答：优先检查：

```text
SchedulerOutput block ids
GPUModelRunner._update_states
GPUModelRunner._get_slot_mappings
worker block_table
gpu block_table
unified_kv_cache_update
```

重点确认：

- block id 是否超过 num_blocks；
- block 内 offset 是否超过 block_size；
- padding 后 slot mapping 是否同步 padding；
- separate KV update backend 是否要求 padded slot mapping；
- spec decode lookahead 是否正确预留；
- sliding window 是否改变了逻辑位置映射。

---

### Q86：paged attention 输出错误如何排查？

答：优先检查：

```text
GPUModelRunner._build_attention_metadata
CommonAttentionMetadata
backend MetadataBuilder
AttentionImpl.forward
csrc paged_attention_v1/v2
```

重点确认：

- seq_lens 是否正确；
- query_start_loc 是否正确；
- block table 是否正确；
- max_query_len/max_seq_len 是否正确；
- causal/sliding window/non-causal 标记是否正确；
- prefix/multimodal ranges 是否正确；
- prefill/decode metadata 是否混淆。

---

### Q87：KV cache quant 输出异常如何排查？

答：优先检查：

```text
KVQuantMode
Attention._init_kv_cache_quant
Attention.process_weights_after_loading
maybe_calc_kv_scales
cache_kernels_fused
nvfp4_kv_cache_kernels
```

重点确认：

- scale 是否从 checkpoint 正确加载；
- 默认 scale 是否为 1.0；
- per-token-head scale shape 是否正确；
- cache dtype 和 kernel 是否匹配；
- FP8/NVFP4 page size 是否正确；
- backend 是否支持该 KV quant mode。

---

### Q88：CUDA illegal memory access 如何排查？

答：这种问题通常不是单个 kernel 文件能看出来，要从 SchedulerOutput 到 GPUModelRunner metadata 一路核对。

优先检查：

```text
slot mapping
block table
KV cache tensor shape
attention metadata
paged attention kernel launch args
CUDA graph padding
```

常见原因：

- block id 越界；
- slot mapping 越界；
- KV cache tensor layout 和 kernel 预期不一致；
- CUDA graph padding 后 metadata 未同步；
- seq_lens/query_start_loc 错误；
- spec decode lookahead block 未正确预留。

---

### Q89：量化模型加载失败怎么定位？

答：优先看：

- `model_loader` 是否识别该量化格式；
- `layers/quantization` 中对应 quant method 是否存在；
- 具体模型 `load_weights()` 是否支持该格式权重名；
- scale/zero-point 是否加载；
- linear/MoE 层是否支持该 quant method；
- 并行切分下 quantized weight shape 是否匹配。

---

### Q90：MoE expert 权重缺失怎么定位？

答：优先看：

- default loader expert filter；
- expert parallel rank/world size；
- 模型 `load_weights()` 中 expert 权重名匹配；
- 当前 rank 是否应该加载该 expert；
- EP 分组是否正确；
- 权重 checkpoint 的 expert 命名是否和模型实现一致。

---

### Q91：MoE forward 性能或错误怎么定位？

答：优先检查：

- router/gating 输出是否正常；
- top-k expert selection 是否正确；
- token dispatch/combine 是否正确；
- fused_moe runner 是否选到预期 kernel；
- expert parallel 通信是否正确；
- EPLB 是否引入额外重排；
- quantized MoE weight/scale 是否正确。

---

### Q92：LoRA 不生效怎么定位？

答：优先检查：

- LoRA 是否成功通过 Executor/Worker 添加；
- request 是否带正确 lora id；
- LoRA manager 是否加载 adapter；
- base_linear 是否被 LoRA 包装；
- `set_active_loras()` 是否根据 InputBatch 设置 active mapping；
- batch 内不同 request 的 adapter mapping 是否正确；
- LoRA 层是否正确注册到 static_forward_context。

---

### Q93：多模态 attention 异常怎么定位？

答：优先检查：

- multimodal registry 和输入预处理；
- encoder cache 是否命中或正确计算；
- mm embeddings 是否和 text tokens 对齐；
- CommonAttentionMetadata 中 mm prefix ranges 是否正确；
- backend 是否支持 `supports_mm_prefix()` 或 non-causal；
- prefix-lm attention 区间是否正确。

---

### Q94：如果怀疑 CUDA graph padding 导致错误，应该看哪些点？

答：优先检查：

```text
_determine_batch_execution_and_padding
_get_slot_mappings
_build_attention_metadata
AttentionMetadataBuilder.build_for_cudagraph_capture
```

重点确认：

- capture size 是否正确；
- padding token 的 slot mapping 是否安全；
- padded metadata 是否和 padded tensor shape 一致；
- block table 是否包含 padding 需要的安全值；
- CUDA graph 静态 buffer 是否复用正确。

---

## 十、高频综合题

### Q95：请完整描述从 SchedulerOutput 到 Attention kernel 的链路。

答：

```text
SchedulerOutput
  -> Worker.execute_model
  -> GPUModelRunner.execute_model
  -> _update_states 更新 worker batch state
  -> _prepare_inputs 准备 input_ids/positions/embeds
  -> _get_slot_mappings 生成 KV 写入位置
  -> _build_attention_metadata 构造 CommonAttentionMetadata 与 backend metadata
  -> set_forward_context 写入 attn_metadata/slot_mapping/no_compile_layers
  -> model forward
  -> Attention.forward
  -> get_attention_context(layer_name)
  -> 取当前 layer 的 KV cache、metadata、slot mapping
  -> unified_kv_cache_update 写入本步 K/V
  -> unified_attention_with_output 或 impl.forward
  -> torch.ops.vllm/backend op
  -> C++ binding
  -> CUDA/csrc paged attention/cache kernel
```

核心是：Scheduler 决定逻辑 block/token，GPUModelRunner 构造运行时张量和 metadata，ForwardContext 把 metadata 暴露给模型层，Attention/Backend/csrc 完成 KV cache update 和 attention compute。

---

### Q96：请解释 KV cache 中 block table 和 slot mapping 的区别。

答：

- block table：request 级别的逻辑 block 到物理 block id 的映射，主要用于 attention kernel 读取历史 K/V。
- slot mapping：当前 step token 到 KV cache 物理 slot 的映射，主要用于写入本步新产生的 key/value。

简化说：

```text
block table: 历史 K/V 去哪里读
slot mapping: 当前 K/V 写到哪里
```

两者都来自 Scheduler/GPURunner 的 block 状态，但服务于不同 kernel 输入。

---

### Q97：为什么 backend 选择会影响 KV cache 分配？

答：因为 backend 会声明 KV cache 的 shape、layout、stride order 和 required layout。

Worker/GPUModelRunner 在分配 KV cache tensor 时必须按照 backend 的要求分配/reshape，否则后续 kernel 读取时 layout 不匹配。

因此 backend 不只是 forward 时选择 kernel，它从 KV cache 初始化阶段就影响物理内存布局。

---

### Q98：为什么 vLLM 要用 paged attention？

答：paged attention 让 KV cache 像分页内存一样管理。

优势：

1. 动态请求长度下可以按 block 分配 KV cache；
2. 减少连续大块显存需求；
3. 支持 prefix cache 和 block 复用；
4. 支持请求抢占、释放、复用；
5. 更适合 continuous batching。

代价是 attention kernel 需要通过 block table 做间接寻址，因此 metadata 和 block table 正确性非常关键。

---

### Q99：为什么 vLLM Attention forward 依赖全局 ForwardContext，而不是普通参数传递？

答：因为运行时信息复杂且跨层共享，包括 attention metadata、slot mapping、KV cache、layer registry、CUDA graph 信息、MoE/LoRA 状态等。

如果都作为普通参数传递，会侵入每个模型 forward 签名，也不利于 torch compile/custom op/CUDA graph。

ForwardContext + static_forward_context 允许通过 layer name 找回对应 layer 和 runtime metadata，降低模型接口复杂度并兼容编译图捕获。

---

### Q100：如果让你概括模型执行与 Attention 层的核心设计，你会怎么说？

答：vLLM 的模型执行与 Attention 层是一套“模型层声明能力 + runner 构造运行时上下文 + backend 选择高性能内核 + KV cache 分页寻址”的系统。

它的核心不是普通 PyTorch forward，而是：

- 模型层 Attention 声明 KVCacheSpec 和 backend；
- Scheduler 决定 token/block 调度；
- GPUModelRunner 构造 input tensors、slot mapping、attention metadata；
- ForwardContext 把运行时信息传给模型层；
- Attention.forward 更新 KV cache 并调用 backend；
- backend/custom op/csrc kernel 完成高性能 paged attention 和 cache update。

---

## 十一、代码阅读与考察补充题

### Q101：读这层代码时应该按什么顺序？

答：推荐顺序：

```text
1. GPUModelRunner.execute_model
2. set_forward_context / ForwardContext
3. Attention.__init__ / Attention.forward
4. Attention.get_kv_cache_spec
5. v1/attention/selector.py:get_attn_backend
6. v1/attention/backend.py:AttentionBackend / MetadataBuilder / AttentionImpl
7. GPUModelRunner.initialize_kv_cache
8. GPUModelRunner._get_slot_mappings
9. GPUModelRunner._build_attention_metadata
10. csrc torch_bindings / cache_kernels / paged_attention kernels
```

这样先理解主链路，再展开 backend、KV cache 和 kernel。

---

### Q102：如果要定位模型权重加载问题，优先看哪些文件？

答：优先看：

```text
vllm/model_executor/model_loader/__init__.py
vllm/model_executor/model_loader/default_loader.py
vllm/model_executor/models/registry.py
vllm/model_executor/models/interfaces.py
具体模型文件，例如 llama/qwen/deepseek
vllm/model_executor/layers/linear.py
vllm/model_executor/layers/quantization/
```

重点看 load_format、architecture 映射、load_weights 权重名匹配、并行切分、量化 scale/packing。

---

### Q103：如果要定位 attention 输出错误，优先看哪些文件？

答：优先看：

```text
vllm/v1/worker/gpu_model_runner.py
vllm/model_executor/layers/attention/attention.py
vllm/v1/attention/backend.py
vllm/v1/attention/selector.py
具体 backend 实现
csrc/libtorch_stable/attention/paged_attention_v1.cu
csrc/libtorch_stable/attention/paged_attention_v2.cu
```

重点核对 input_ids/positions、slot mapping、block table、seq_lens、query_start_loc、KV cache layout、backend kernel 参数。

---

### Q104：如果要定位 KV cache 内存不足或 block 数异常，优先看什么？

答：优先看：

- `Attention.get_kv_cache_spec()`；
- `KVCacheSpec.page_size_bytes()`；
- `AttentionSpec` 中 num kv heads/head size/dtype/block size/quant mode；
- Worker 的可用显存 profiling；
- EngineCore 计算 KVCacheConfig 的逻辑；
- GPUModelRunner.initialize_kv_cache 分配 tensor 的 shape。

重点确认 page size 是否异常、量化 scale overhead 是否计算正确、MLA/Mamba/sliding window 是否使用了正确 spec。

---

### Q105：如果要解释 vLLM Attention 性能优化点，可以从哪些方面回答？

答：可以从以下方面回答：

1. Paged attention：通过 block table 管理非连续 KV cache；
2. KV cache 预分配和分页复用；
3. backend 自动选择 FlashAttention/FlashInfer/Triton/平台优化 kernel；
4. CUDA graph capture 减少 launch overhead；
5. fused RoPE + KV cache update；
6. KV cache quant 降低显存带宽和容量压力；
7. cascade attention/common prefix 优化 prefix 共享；
8. continuous batching 提高吞吐；
9. spec decode、ubatching、DP padding 等运行时优化；
10. MoE fused kernel 和 expert parallel。

---

### Q106：`CommonAttentionMetadata` 为什么不能直接让所有 backend 共用？

答：因为不同 backend 对 metadata 的 layout、字段、padding、CUDA graph 支持、prefill/decode 表示方式不同。

`CommonAttentionMetadata` 是公共输入格式，但最终需要 `AttentionMetadataBuilder` 转成 backend-specific metadata。

这样 GPUModelRunner 不必为每个 backend 写一套完整逻辑，同时 backend 可以保留自己的高性能 kernel 输入格式。

---

### Q107：什么情况下 backend 需要回退或禁用优化？

答：常见情况：

- head size 不支持；
- dtype 或 KV cache dtype 不支持；
- block size 不支持；
- 当前平台缺依赖；
- 需要 non-causal/mm prefix，但 backend 不支持；
- 需要 per-head quant scales，但 backend 不支持；
- 需要 KV connector，但 backend 不支持；
- CUDA graph shape 不稳定或不支持；
- MLA/sparse/sink/sliding window 组合不被 backend 支持。

---

### Q108：为什么 KV connector 会影响 KV cache layout？

答：KV connector 可能希望 KV cache 跨层连续布局，以便高效传输。例如 cross-layer blocks 可以让多层 KV cache 按 block 连续存放，减少传输时的碎片化和多次拷贝。

因此 KV transfer 初始化要发生在 KV cache 分配前，让 model runner 在分配 KV cache group/tensor 时考虑 connector 的 layout 偏好。

---

### Q109：prefix cache/common prefix 和 attention metadata 有什么关系？

答：Scheduler/GPUModelRunner 会识别多个 request 共享的公共 prefix blocks，并在 attention metadata 中记录 common prefix/cascade attention 相关信息。

backend MetadataBuilder 可以据此判断是否启用 cascade attention，减少重复 prefix attention 计算或优化访问模式。

---

### Q110：prefill 和 decode 在 attention metadata 上有什么差异？

答：

- prefill：query length 通常大于 1，需要处理一段 prompt token，可能涉及大 query_start_loc、prefill attention、prefix/multimodal 区间。
- decode：通常每个 request 只生成少量 token，query length 小，重点是从 KV cache 中读取长历史。

metadata builder 会根据 prefill/decode 构造不同的 backend metadata，kernel 路径也可能不同。

---

## 十二、简短背诵版总结

### Q111：一句话解释 GPUModelRunner。

答：GPUModelRunner 把 SchedulerOutput 转换成模型 forward 需要的 input tensors、slot mapping 和 attention metadata，并通过 ForwardContext 驱动模型执行和采样。

### Q112：一句话解释 Attention。

答：Attention 是模型层 glue layer，初始化时声明 backend 和 KVCacheSpec，forward 时从 ForwardContext 取 KV cache/metadata/slot mapping，并调用 backend kernel。

### Q113：一句话解释 AttentionBackend。

答：AttentionBackend 描述某套 attention 实现的能力、KV cache layout 和 metadata builder，并创建实际执行的 AttentionImpl。

### Q114：一句话解释 KVCacheSpec。

答：KVCacheSpec 是模型层对 KV cache 需求的声明，决定 KV cache page size、分组、tensor shape 和内存估算。

### Q115：一句话解释 slot mapping。

答：slot mapping 把当前 batch 中每个新 token 映射到物理 KV cache slot，是写入 K/V cache 的关键索引。

### Q116：一句话解释 block table。

答：block table 把每个 request 的逻辑 block 映射到物理 KV cache block，是 paged attention 读取历史 K/V 的关键索引。

### Q117：一句话解释 ForwardContext。

答：ForwardContext 是运行时上下文，用来让模型层在 forward 中访问 attention metadata、slot mapping、KV cache 和静态 layer registry。

### Q118：一句话解释 paged attention。

答：paged attention 通过 block table 间接寻址分页 KV cache，让动态请求的 KV cache 可以按 block 分配、复用和释放。

### Q119：一句话解释 KV cache quant。

答：KV cache quant 将历史 K/V 以 FP8/INT8/NVFP4 等格式存储，降低显存占用和带宽压力，但要求 scale、page size、backend kernel 全链路匹配。

### Q120：一句话解释 Python 到 csrc 的调用链。

答：Attention.forward 通过 custom op 或 AttentionImpl 调用 `torch.ops.vllm.*`，再经 C++ binding 启动 CUDA/ROCm/CPU kernel 完成 cache update 和 paged attention。
