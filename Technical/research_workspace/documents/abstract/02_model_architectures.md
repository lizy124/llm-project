# 02 model_architectures 背诵文档

## 1. 专题定位

`model_architectures` 讲的是 vLLM 如何把外部模型配置和 checkpoint 适配成统一可执行的模型类。

它连接两个世界：

```text
外部世界：Hugging Face config / architectures / checkpoint 命名 / 模型结构差异
内部世界：vLLM ModelRunner 能统一调用的 nn.Module / forward / compute_logits / load_weights
```

一句话：

```text
model_architectures 负责把各种模型架构适配成 vLLM 执行层能统一调用的 model class。
```

## 2. 最小心智模型

主链路是：

```text
ModelConfig / HF config
  → hf_config.architectures
  → ModelRegistry.inspect_model_cls()
  → ModelRegistry.resolve_model_cls()
  → vLLM model class
  → initialize_model(vllm_config, prefix)
  → model class 构造 embedding / layers / attention / MLP / norm / lm_head
  → model.load_weights(weights_iterator)
  → GPUModelRunner 调 model.forward(...)
  → hidden states
  → compute_logits() 或 pooler()
```

要背住：

```text
ModelRegistry 负责找类，model class 负责建结构，load_weights 负责接 checkpoint，forward 契约负责交给 ModelRunner 执行。
```

## 3. 这个专题解决的问题

它回答：

1. model architecture 在 vLLM 中是哪一层。
2. HF `architectures` 如何解析成 vLLM model class。
3. `ModelConfig` 如何判断 task、runner_type、模型能力。
4. 一个模型类如何构造 embedding、decoder layers、attention、MLP、norm、lm_head。
5. ModelRunner 对 `forward` 接口有什么约定。
6. 权重加载如何处理 fused layer、TP shard、PP missing layer、checkpoint 命名差异。
7. LoRA、quantization、parallelism、multimodal 如何 hook 到模型架构。
8. 新增模型架构需要实现哪些接口。

## 4. 在系统中的位置

可以把 vLLM 模型执行拆成五层：

```text
配置层：
  ModelConfig / ParallelConfig / CacheConfig / LoRAConfig / QuantConfig
  决定模型类型、任务、dtype、长度、并行和功能开关。

注册层：
  ModelRegistry
  根据 HF architectures 找到 model class，并 inspect 能力。

模型架构层：
  vllm/model_executor/models/*.py
  定义每个模型如何构造 layers、forward、load_weights。

基础 layer 层：
  Attention / Linear / Embedding / MoE / Norm / RoPE / LogitsProcessor
  提供复用组件。

执行层：
  Worker / GPUModelRunner
  按统一接口调用 model。
```

一句话边界：

```text
model architecture 负责模型结构适配，不负责调度、KV block 分配和采样策略。
```

## 5. ModelConfig 如何驱动架构选择

`ModelConfig` 先读取 HF config：

```text
self.hf_config
self.hf_text_config
self.model_arch_config
architectures = self.architectures
```

然后通过 registry 判断：

```text
is_text_generation_model
is_pooling_model
supports_multimodal
supports_pp
is_attention_free
is_moe
```

再推导：

```text
runner_type：generate / pooling / draft
convert_type：none / embed / classify
```

最后缓存：

```text
self._model_info
self._architecture
```

要背住：

```text
ModelConfig 不是只读一个 architecture 字符串，它会提前 inspect 模型能力，并反过来影响 runner、pooler、多模态和并行配置。
```

## 6. ModelRegistry 的作用

`ModelRegistry` 有两个核心入口：

```text
inspect_model_cls()
  用于配置阶段，返回模型能力信息。

resolve_model_cls()
  用于模型初始化阶段，返回真实 nn.Module class。
```

解析路径：

```text
HF config.architectures
  → normalize architecture
  → 查 vLLM in-tree registry
  → 必要时 transformers fallback
  → trust_remote_code 动态模块
  → 返回 model class
```

registry 不只是字典查表。

它还会生成 `_ModelInfo`：

```text
is_text_generation_model
is_pooling_model
attn_type
score_type
supports_multimodal
supports_pp
has_inner_state
is_attention_free
is_hybrid
supports_transcription
supports_mrope / xdrope
```

一句话：

```text
ModelRegistry 把 architecture name 解析为 model class，同时把模型类能力摘要暴露给配置层。
```

## 7. initialize_model 如何构造模型

`initialize_model()` 是 loader 侧创建模型对象的入口。

主线：

```text
initialize_model(vllm_config)
  → get_model_architecture(model_config)
  → registry.resolve_model_cls(architectures)
  → 如果 convert_type=embed/classify，包 adapter
  → configure_quant_config(quant_config, model_class)
  → model_class(vllm_config=vllm_config, prefix=prefix)
  → record_metadata_for_reloading(model)
```

新式模型类推荐构造函数：

```python
class XxxForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        ...
```

为什么要传 `prefix`：

```text
prefix 保证参数名、LoRA、quant、loader、权重映射都能定位到正确模块。
```

## 8. 一个典型 model class 的结构

以 decoder-only generation 模型为模板。

内部模型通常有：

```text
embed_tokens
layers
norm
```

外层 `ForCausalLM` 通常有：

```text
self.model
self.lm_head
self.logits_processor
```

典型职责：

```text
inner model：embedding → decoder layers → norm → hidden states
outer model：forward 委托 inner model，compute_logits 计算 logits，load_weights 加载权重
```

## 9. ModelRunner 对 forward 的统一契约

ModelRunner 最终按统一方式调用模型：

```python
self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

模型类至少要理解：

```text
input_ids：文本 token ids。
positions：位置 ids，可能是一维，也可能是 M-RoPE / XD-RoPE 多维。
intermediate_tensors：pipeline parallel 中间 stage 输入或输出。
inputs_embeds：多模态或 prompt embeds 路径。
model_kwargs：encoder_outputs、cross attention、多模态、Mamba 等特殊输入。
```

Generation 模型还需要：

```text
compute_logits(hidden_states) -> logits
```

Pooling 模型还需要：

```text
pooler(hidden_states) -> embedding / score / pooler output
```

权重加载还需要：

```text
load_weights(weights_iterator)
```

## 10. 模型架构层统一哪些差异

### 结构差异

```text
Decoder-only：Llama / Qwen / Mistral / Gemma / DeepSeek
Encoder-decoder：Whisper / speech-to-text
Encoder-only：BERT-like embedding / rerank / classify
Multimodal：LLaVA / Qwen2-VL / audio-video models
MoE：Mixtral / DeepSeek-MoE
Attention-free / hybrid：Mamba / hybrid attention
```

### Attention 差异

模型架构决定：

```text
MHA / MQA / GQA
MLA
sliding window
cross attention
head_dim / num_heads / num_kv_heads
RoPE / M-RoPE / XD-RoPE
```

### MLP / MoE 差异

```text
普通 MLP：gate_up_proj + down_proj / fused SwiGLU
MoE：router + routed experts + shared experts + fused MoE
```

### 输出差异

```text
generation：hidden_states → compute_logits → sampling
pooling：hidden_states → pooler → embeddings
classify / score / rerank：hidden_states → scores
transcription：audio path → transcription output
```

### 权重差异

```text
HF checkpoint 可能是 q_proj / k_proj / v_proj。
vLLM runtime 可能是 qkv_proj。
HF checkpoint 可能是 gate_proj / up_proj。
vLLM runtime 可能是 gate_up_proj。
TP 下参数需要按 shard_id 加载。
PP 下当前 rank 没有的 layer 要跳过。
quantized checkpoint 有 qweight / scale / zero_point。
tie_word_embeddings 时 lm_head 可能不单独加载。
```

## 11. 权重加载的关键

默认 loader 最终调用：

```text
model.load_weights(weights_iterator)
```

所以模型类需要处理：

```text
checkpoint tensor name → runtime parameter name
unfused → fused
TP shard slicing
PP missing layer skip
tie weight skip
quant scale / zero point mapping
prefix mapping
```

常见工具和约定：

```text
AutoWeightsLoader
WeightsMapper
packed_modules_mapping
embedding_modules
param.weight_loader
```

一句话：

```text
loader 负责枚举 checkpoint tensor，model.load_weights 负责把 tensor 放到正确参数里。
```

## 12. model interfaces 是能力开关

模型类会通过接口声明能力。

常见接口：

```text
SupportsLoRA：允许 LoRA 注入。
SupportsPP：允许 pipeline parallel。
SupportsMultiModal：允许多模态 processor / encoder / embedding 合并。
SupportsQuant：允许量化配置 hook。
SupportsTranscription：语音转文本。
SupportsEagle / SupportsEagle3：spec decode。
SupportsMRoPE / SupportsXDRoPE：特殊多维位置编码。
SupportsEncoderCudaGraph：encoder CUDA graph。
```

这些接口不是装饰性文档。

registry 会 inspect 它们，并写入 `_ModelInfo`。

## 13. 新增模型架构的最小检查项

### 注册和构造

```text
1. HF architectures 能解析到模型类。
2. __init__ 支持 vllm_config 和 prefix。
3. 使用 vLLM shared layers。
4. 正确传递 prefix。
```

### forward 契约

```text
1. forward 接受 input_ids / positions / intermediate_tensors / inputs_embeds。
2. first PP rank 能从 input_ids 或 inputs_embeds 生成 hidden_states。
3. 非 first PP rank 能从 IntermediateTensors 恢复 hidden_states。
4. 非 last PP rank 返回 IntermediateTensors。
5. last PP rank 返回 hidden_states 或任务输出。
```

### 输出契约

```text
Generation：实现 compute_logits。
Pooling：实现 pooler 或使用 adapter。
Multimodal：实现 embed_multimodal / embed_input_ids。
```

### 权重加载

```text
1. 实现 load_weights。
2. 处理 fused mapping。
3. 处理 TP shard。
4. 处理 PP missing layer。
5. 处理 tie embeddings。
6. 处理量化参数。
```

## 14. 与其他专题的关系

```text
config_and_model_loading：解释 ModelConfig / ModelLoader 如何触发架构选择。
executor_worker_model_runner：解释 ModelRunner 如何调用 model.forward。
attention：模型架构决定 attention layer 参数和 backend 需求。
parallelism：模型层构造时要适配 TP / PP / EP。
quantization：layer 构造时接入 quant_config / quant_method。
lora_and_adapters：SupportsLoRA 和 packed mapping 决定 LoRA target modules。
multimodal：多模态模型通过 SupportsMultiModal 和 processor 接入。
sampling_and_output：generation 输出 logits，pooling 输出 pooler_output。
```

## 15. 容易混淆的点

### ModelRegistry 不等于模型加载

```text
registry 找到模型类；ModelLoader 才实例化和加载权重。
```

### forward 不等于 compute_logits

```text
forward 产生 hidden states；compute_logits 才把 hidden states 变成 vocab logits。
```

### model class 不负责调度

```text
模型类只表达结构和接口；调度、KV block、batch 状态属于 Scheduler / ModelRunner。
```

### checkpoint 命名不等于 runtime 参数命名

```text
load_weights 的核心工作就是处理这种差异。
```

## 16. 背诵总结

背这一段：

```text
vLLM 的 model architectures 是模型适配层。ModelConfig 从 HF config 得到 architectures，并通过 ModelRegistry inspect 模型能力；加载阶段 registry.resolve_model_cls 返回真实 model class，initialize_model 用 vllm_config 和 prefix 实例化模型。模型类用 vLLM shared layers 构造 embedding、attention、MLP、norm、lm_head 或 pooler，并通过统一 forward / compute_logits / pooler / load_weights 契约交给 ModelRunner 执行。它的核心价值是让 Llama、Qwen、DeepSeek、Mixtral、BERT、Whisper、多模态和 MoE 等不同模型，都能落到同一套 vLLM 调度、执行、KV cache、LoRA、量化和并行框架里。
```
