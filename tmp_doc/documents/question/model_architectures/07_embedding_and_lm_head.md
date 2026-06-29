# 07. Embedding、LM head 和 logits processor 如何接入？

源码位置：

- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\vocab_parallel_embedding.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\layers\logits_processor.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\llama.py`
- `E:\lizy\code\vllm-project\vllm\vllm\model_executor\models\interfaces.py`
- `E:\lizy\code\vllm-project\vllm\vllm\lora\layers\vocal_parallel_embedding.py`
- `E:\lizy\code\vllm-project\vllm\vllm\lora\layers\logits_processor.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\worker\gpu_model_runner.py`
- `E:\lizy\code\vllm-project\vllm\vllm\v1\worker\gpu\model_runner.py`

本问题关注：输入 embedding、输出 LM head、tie weights、LoRA added vocab、多模态 / prompt embeds 覆盖、logits gather / soft cap / scale，以及这些模块如何接入 ModelRunner 的 forward / sampling 链路。

---

## 1. 一句话回答

Embedding 和 LM head 是模型架构与 vLLM 执行链路的两个边界：

```text
input_ids / inputs_embeds
  → embed_tokens / multimodal merge
  → transformer hidden_states
  → hidden_states[logits_indices]
  → lm_head weight + logits_processor
  → logits
  → sampler
```

在 vLLM 中，`embed_tokens` 通常是 `VocabParallelEmbedding`；`lm_head` 通常是 `ParallelLMHead`；真正把 hidden states 投影成 logits 的不是 `lm_head.forward()`，而是 `LogitsProcessor` 通过 `lm_head.quant_method.apply(...)` 使用 LM head 权重完成。

---

## 2. 在模型类里如何接入

以 LLaMA 为例，模型分成两层：

```text
LlamaModel：
  负责 embedding + decoder layers + final norm，返回 hidden_states。

LlamaForCausalLM：
  包住 LlamaModel，额外提供 lm_head、logits_processor、compute_logits()。
```

### 2.1 input embedding 接入点

`LlamaModel.__init__()` 中创建：

```python
self.embed_tokens = VocabParallelEmbedding(
    self.vocab_size,
    config.hidden_size,
    quant_config=quant_config,
)
```

位置：`vllm/vllm/model_executor/models/llama.py:365` 到 `llama.py:372`

然后封装成：

```python
def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    return self.embed_tokens(input_ids)
```

位置：`llama.py:389` 到 `llama.py:390`

### 2.2 forward 中使用 embedding

首个 PP rank 的 forward：

```python
if get_pp_group().is_first_rank:
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.embed_input_ids(input_ids)
    residual = None
```

位置：`llama.py:400` 到 `llama.py:405`

这说明：

```text
如果传了 inputs_embeds：
  直接作为 hidden_states，绕过 token embedding 查表。

如果没传 inputs_embeds：
  用 input_ids 调 embed_input_ids()，也就是 VocabParallelEmbedding.forward()。
```

### 2.3 LM head 和 logits processor 接入点

`LlamaForCausalLM.__init__()` 中创建：

```python
self.lm_head = ParallelLMHead(
    config.vocab_size,
    config.hidden_size,
    quant_config=quant_config,
    prefix=maybe_prefix(prefix, "lm_head"),
)
```

位置：`llama.py:518` 到 `llama.py:524`

然后创建 logits processor：

```python
logit_scale = getattr(config, "logit_scale", 1.0)
self.logits_processor = LogitsProcessor(
    config.vocab_size, scale=logit_scale
)
```

位置：`llama.py:528` 到 `llama.py:531`

最终通过 `compute_logits()` 暴露给 ModelRunner：

```python
def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
    logits = self.logits_processor(self.lm_head, hidden_states)
    return logits
```

位置：`llama.py:562` 到 `llama.py:567`

---

## 3. input embedding 的主链路

普通文本输入可以记成：

```text
SchedulerOutput
  → ModelRunner 准备 input_ids / positions
  → model.forward(input_ids=..., inputs_embeds=None)
  → LlamaModel.forward()
  → embed_input_ids(input_ids)
  → VocabParallelEmbedding.forward()
  → hidden_states
```

关键点是：ModelRunner 只负责准备 `input_ids` / `inputs_embeds`，具体查 embedding 的逻辑在模型架构类里。

V1 `GPUModelRunner._model_forward()` 只是把输入传给模型：

```python
return self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3784` 到 `gpu_model_runner.py:3790`

所以 embedding 接入点不是 ModelRunner 的特殊分支，而是模型 `forward()` 内部对 `inputs_embeds` 和 `input_ids` 的选择。

---

## 4. VocabParallelEmbedding 做什么

`VocabParallelEmbedding` 定义在：`vocab_parallel_embedding.py:198`

它的核心职责是：

```text
1. 按 tensor parallel rank 切分 vocab 维度；
2. 对原始 vocab 和 added vocab 分别 padding；
3. 当前 rank 只查自己负责的 token 范围；
4. 不属于当前 rank 的 token 输出置 0；
5. 对所有 TP rank 做 all-reduce，合成完整 embedding 输出。
```

### 4.1 为什么按 vocab 维度切分

embedding 权重形状可以理解为：

```text
[vocab_size, hidden_size]
```

vLLM 在 TP 下按 vocab 维度切：

```text
rank 0: token range A 的 embedding rows
rank 1: token range B 的 embedding rows
...
```

每个 rank 对整批 `input_ids` 都执行一次 embedding，但只有落在自己 vocab 范围内的 token 有真实输出，其他 token 被 mask 成 0。最后 all-reduce 后，每个 token 的真实 embedding 会保留下来。

### 4.2 padding 和 added vocab 的布局

`VocabParallelEmbedding` 不只是简单把 `num_embeddings` 平分，还要处理：

```text
org_vocab：模型原始词表；
org_vocab_padding：为了 TP / kernel 对齐补的 padding；
added_vocab：LoRA 或 tokenizer 扩展出来的新 token；
added_vocab_padding：added vocab 后面的 padding。
```

类注释里的逻辑可以压缩为：

```text
原始 vocab 先 pad 到 padding_size；
再追加 added vocab；
最后再 pad 到 padding_size；
TP shard 时，base 和 added vocab 的区间分别计算。
```

相关字段：

```text
org_vocab_size
org_vocab_size_padded
num_embeddings
num_embeddings_padded
num_org_embeddings_per_partition
num_added_embeddings_per_partition
shard_indices
```

位置：`vocab_parallel_embedding.py:254` 到 `vocab_parallel_embedding.py:315`

### 4.3 mask + all-reduce 的 forward

核心 forward：

```python
if self.tp_size > 1:
    masked_input, input_mask = get_masked_input_and_mask(...)
else:
    masked_input = input_

output_parallel = self.quant_method.embedding(self, masked_input.long())

if self.tp_size > 1:
    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)

output = tensor_model_parallel_all_reduce(output_parallel)
return output
```

位置：`vocab_parallel_embedding.py:472` 到 `vocab_parallel_embedding.py:492`

含义是：

```text
每个 rank：
  input_ids → 本地 vocab 范围映射 → 本地 embedding 查表 → 非本 rank token 置 0

所有 rank：
  all-reduce 求和 → 得到完整 hidden states
```

### 4.4 quantized embedding

`VocabParallelEmbedding` 支持 quantization config：

```python
quant_method = quant_config.get_quant_method(self, prefix=prefix)
```

位置：`vocab_parallel_embedding.py:276` 到 `vocab_parallel_embedding.py:280`

但如果这个层本身就是 `VocabParallelEmbedding`，量化方法必须实现 `embedding()`：

```python
if is_embedding_layer and not quant_method_implements_embedding:
    raise NotImplementedError(...)
```

位置：`vocab_parallel_embedding.py:282` 到 `vocab_parallel_embedding.py:293`

也就是说：

```text
embedding 层不能只支持 linear/GEMM；
它必须能执行 embedding lookup。
```

---

## 5. inputs_embeds / prompt_embeds / 多模态 embeddings 如何覆盖 input embedding

vLLM 有两类常见“绕过或覆盖 token embedding”的入口。

### 5.1 inputs_embeds 直接传入模型

模型 forward 通常支持：

```python
inputs_embeds: torch.Tensor | None = None
```

如果 `inputs_embeds` 非空，首个 PP rank 直接使用它：

```text
inputs_embeds → hidden_states
```

而不是：

```text
input_ids → embed_tokens → hidden_states
```

LLaMA 的逻辑见：`llama.py:400` 到 `llama.py:405`

### 5.2 prompt_embeds 从请求进入

请求侧可以带 `prompt_embeds`，但需要开启：

```text
--enable-prompt-embeds
```

校验逻辑包括：

```text
必须是 2D tensor: (num_tokens, hidden_size)；
hidden_size 必须等于模型 hidden_size；
dtype 必须能安全转换到模型 dtype；
进程间传输前放到 CPU。
```

位置：

- `vllm/vllm/renderers/embed_utils.py:16` 到 `embed_utils.py:80`
- `vllm/vllm/renderers/base.py:753` 到 `base.py:778`

这类输入最终会作为 embedding 形态进入后续执行链路。

### 5.3 多模态 embedding merge

多模态模型实现 `SupportsMultiModal.embed_input_ids()`：

```text
1. 先用 language_model.embed_input_ids(input_ids) 得到文本 embeddings；
2. 如果有 multimodal_embeddings，就按 is_multimodal mask scatter 覆盖；
3. 返回合并后的 inputs_embeds。
```

核心代码：

```python
inputs_embeds = self._embed_text_input_ids(
    input_ids,
    self.get_language_model().embed_input_ids,
    is_multimodal=is_multimodal,
)

return _merge_multimodal_embeddings(
    inputs_embeds=inputs_embeds,
    multimodal_embeddings=multimodal_embeddings,
    is_multimodal=_require_is_multimodal(is_multimodal),
)
```

位置：`vllm/vllm/model_executor/models/interfaces.py:397` 到 `interfaces.py:410`

如果多模态 token 超出 vocab，vLLM 会先把这些 token id mask 成 0 再做文本 embedding，避免 embedding lookup 越界；随后多模态 embedding 会覆盖对应位置。

位置：`interfaces.py:363` 到 `interfaces.py:373`

---

## 6. Pipeline Parallel 下 embedding 和 lm_head 在哪些 rank 上

PP 下不是每个 rank 都拥有完整的 embedding / lm_head。

### 6.1 embedding 所在 rank

LLaMA 中：

```python
if get_pp_group().is_first_rank or (
    config.tie_word_embeddings and get_pp_group().is_last_rank
):
    self.embed_tokens = VocabParallelEmbedding(...)
else:
    self.embed_tokens = PPMissingLayer()
```

位置：`llama.py:365` 到 `llama.py:374`

含义是：

```text
普通情况：
  first PP rank 持有 input embedding。

如果 tie_word_embeddings=True：
  last PP rank 也可能需要 embed_tokens，用于给 lm_head tie weights。

其他 rank：
  embed_tokens 是 PPMissingLayer 占位。
```

### 6.2 lm_head 所在 rank

LLaMA 中：

```python
if get_pp_group().is_last_rank:
    self.lm_head = ParallelLMHead(...)
else:
    self.lm_head = PPMissingLayer()
```

位置：`llama.py:518` 到 `llama.py:533`

因此 PP 链路是：

```text
PP first rank：
  input_ids / inputs_embeds → embedding → partial layers → IntermediateTensors

PP middle rank：
  IntermediateTensors → partial layers → IntermediateTensors

PP last rank：
  IntermediateTensors → final layers / norm → hidden_states → lm_head/logits_processor
```

### 6.3 非最后 PP rank 不算 logits

V1 ModelRunner 中：

```python
if not get_pp_group().is_last_rank:
    assert isinstance(hidden_states, IntermediateTensors)
    self.kv_connector_output = kv_connector_output
    return hidden_states
```

位置：`gpu_model_runner.py:4340` 到 `gpu_model_runner.py:4346`

只有最后 PP rank 会继续：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`gpu_model_runner.py:4357` 到 `gpu_model_runner.py:4358`

---

## 7. tied embeddings 如何实现

如果 HuggingFace config 中有：

```text
tie_word_embeddings=True
```

模型会把 `lm_head` 权重和 `embed_tokens` 权重绑定。

LLaMA：

```python
if config.tie_word_embeddings:
    self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
```

位置：`llama.py:525` 到 `llama.py:526`

`ParallelLMHead.tie_weights()` 内部调用 quant method：

```python
def tie_weights(self, embed_tokens: VocabParallelEmbedding):
    return self.quant_method.tie_weights(self, embed_tokens)
```

位置：`vocab_parallel_embedding.py:558` 到 `vocab_parallel_embedding.py:560`

默认未量化实现是：

```python
def tie_weights(self, layer, embed_tokens):
    layer.weight = embed_tokens.weight
    return layer
```

位置：`vocab_parallel_embedding.py:80` 到 `vocab_parallel_embedding.py:84`

所以 tied embeddings 的本质是：

```text
lm_head.weight 与 embed_tokens.weight 指向同一个 Parameter。
```

加载权重时也会配合跳过 `lm_head.` 前缀，避免重复加载 tied weight：

```python
skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None)
```

位置：`llama.py:569` 到 `llama.py:574`

---

## 8. ParallelLMHead 为什么 forward 会报错

`ParallelLMHead` 继承自 `VocabParallelEmbedding`，但它不是拿来直接 forward 的。

源码：

```python
def forward(self, input_):
    del input_
    raise RuntimeError("LMHead's weights should be used in the sampler.")
```

位置：`vocab_parallel_embedding.py:562` 到 `vocab_parallel_embedding.py:564`

原因是：

```text
LM head 在 vLLM 中只是“输出投影权重容器”；
真正的 logits 计算由 LogitsProcessor._get_logits() 调用 quant_method.apply() 完成。
```

即：

```text
错误理解：
  hidden_states → lm_head(hidden_states) → logits

实际 vLLM：
  hidden_states → logits_processor(lm_head, hidden_states) → quant_method.apply(lm_head, hidden_states) → logits
```

---

## 9. LogitsProcessor 做什么

`LogitsProcessor` 定义在：`logits_processor.py:19`

它的职责有三步：

```text
1. 用 lm_head weight 从 hidden_states 计算本 rank logits；
2. TP gather / all-gather 成完整 vocab logits；
3. 应用 soft cap 和 scale。
```

### 9.1 logits 计算

```python
logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
```

位置：`logits_processor.py:96`

这里会走普通 GEMM 或量化 GEMM，取决于 `lm_head.quant_method`。

### 9.2 TP gather

```python
logits = self._gather_logits(logits)
```

位置：`logits_processor.py:98` 到 `logits_processor.py:99`

`_gather_logits()` 有两种路径：

```python
if self.use_all_gather:
    logits = tensor_model_parallel_all_gather(logits)
else:
    logits = tensor_model_parallel_gather(logits)
```

位置：`logits_processor.py:75` 到 `logits_processor.py:87`

区别是：

```text
gather：
  通常只有 rank 0 得到完整 logits，其他 rank 可能得到 None。

all-gather：
  每个 rank 都得到完整 logits，某些平台如 XLA / TPU 需要严格 SPMD。
```

### 9.3 删除 padding vocab

```python
if logits is not None:
    logits = logits[..., : self.org_vocab_size]
```

位置：`logits_processor.py:101` 到 `logits_processor.py:104`

这一步把为了 TP / kernel 对齐而加的 padding vocab 去掉，保证输出 logits 的最后一维对应真实原始词表。

### 9.4 soft cap 和 scale

`LogitsProcessor.forward()` 中：

```python
if self.soft_cap is not None:
    logits = logits / self.soft_cap
    logits = torch.tanh(logits)
    logits = logits * self.soft_cap

if self.scale != 1.0:
    logits *= self.scale
```

位置：`logits_processor.py:65` 到 `logits_processor.py:72`

含义：

```text
soft_cap：
  用 tanh 把 logits 限制在一个软上限内，常见于 Gemma 2 等模型。

scale：
  模型配置里的 logit_scale，用于整体缩放 logits。
```

---

## 10. logits_indices 为什么在 LM head 之前

ModelRunner 不会对所有 hidden states 都算 logits。

V1 中：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`gpu_model_runner.py:4357` 到 `gpu_model_runner.py:4358`

V2 中：

```python
sample_hidden_states = hidden_states[input_batch.logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`vllm/vllm/v1/worker/gpu/model_runner.py:1043` 到 `model_runner.py:1044`

这表示：

```text
transformer forward 可以产生很多 token 的 hidden_states；
但 LM head 只投影本轮真正需要 logits 的位置。
```

常见原因：

```text
prefill：
  通常只需要最后一个 prompt token 的 logits 用于采样。

prompt logprobs：
  需要额外 prompt token 位置的 logits。

chunked prefill：
  中间 chunk 可能不需要采样 logits。

decode：
  每个请求通常只需要当前 decode token 的 logits。

spec decode：
  target model 可能要验证多个 draft token 位置。
```

所以完整链路是：

```text
hidden_states
  → hidden_states[logits_indices]
  → compute_logits()
  → logits_processor(lm_head, sample_hidden_states)
  → logits
```

---

## 11. LoRA 对 embedding 和 logits 的影响

LLaMA 声明了 LoRA 相关模块名：

```python
embedding_modules = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}
```

位置：`llama.py:494` 到 `llama.py:498`

含义是：

```text
embed_tokens 可以被 LoRA 作为 input_embeddings 处理；
lm_head 可以被 LoRA 作为 output_embeddings 处理。
```

### 11.1 embedding LoRA

`VocabParallelEmbeddingWithLoRA.forward()` 先调用原始 embedding：

```python
full_output = self.base_layer.forward(x)
```

然后通过 Punica 加上 LoRA embedding 增量：

```python
lora_output = self.punica_wrapper.add_lora_embedding(
    full_output, full_lora_a_embeddings, self.lora_b_stacked, add_input=True
)
```

位置：`vllm/vllm/lora/layers/vocal_parallel_embedding.py:102` 到 `vocal_parallel_embedding.py:126`

如果存在 added vocab，LoRA wrapper 会把 base layer 中 added vocab 对应的权重暂存到 `embeddings_weights`，并清零 base layer 对应区域，避免重复叠加。

位置：`vocal_parallel_embedding.py:30` 到 `vocal_parallel_embedding.py:47`

### 11.2 logits LoRA

`LogitsProcessorWithLoRA._get_logits()` 先按基础 LM head 算 logits：

```python
logits = actual_lm_head.quant_method.apply(actual_lm_head, hidden_states)
```

再 gather：

```python
logits = self.base_layer._gather_logits(logits)
```

然后追加 LoRA logits 增量：

```python
lora_output = self.punica_wrapper.add_lora_logits(
    logits, hidden_states, self.lora_a_stacked, self.lora_b_stacked, 1.0
)
```

最后裁掉 padding：

```python
logits = logits[:, : self.base_layer.vocab_size]
```

位置：`vllm/vllm/lora/layers/logits_processor.py:141` 到 `logits_processor.py:190`

### 11.3 sharded_to_full_mapping

LoRA logits 里还有一个重要重排：

```python
logits = logits[:, self.sharded_to_full_mapping_gpu]
```

位置：`vllm/vllm/lora/layers/logits_processor.py:162` 到 `logits_processor.py:180`

原因是：TP gather 后的 logits 顺序可能仍然保持“每个 shard 内 base / added / padding”的布局，不一定满足：

```text
logits[..., token_id] 对应 token_id
```

`VocabParallelEmbedding.get_sharded_to_full_mapping()` 会构造重排索引，让 gather 后的 logits 恢复为 token id 顺序。

位置：`vocab_parallel_embedding.py:365` 到 `vocab_parallel_embedding.py:428`

---

## 12. quantized LM head

`ParallelLMHead` 继承 `VocabParallelEmbedding`，初始化时同样接收：

```python
quant_config: QuantizationConfig | None = None
```

位置：`vocab_parallel_embedding.py:523` 到 `vocab_parallel_embedding.py:542`

区别是：

```text
作为 embedding 层时：
  quant_method 必须实现 embedding()。

作为 LM head 时：
  不要求实现 embedding()，但必须能通过 apply() 做 hidden_states × weight^T。
```

源码里专门判断了：

```python
is_embedding_layer = type(self) is VocabParallelEmbedding
```

位置：`vocab_parallel_embedding.py:282` 到 `vocab_parallel_embedding.py:285`

所以量化 LM head 的关键路径是：

```text
ParallelLMHead.weight（可能是量化权重）
  → lm_head.quant_method.apply(lm_head, hidden_states)
  → local logits
  → gather
```

---

## 13. final logits dtype 在哪里决定

`LogitsProcessor` 本身不会无条件把 logits 转成 float32。它调用的是：

```text
lm_head.quant_method.apply(...)
```

因此 logits 初始 dtype 主要取决于：

```text
hidden_states dtype；
lm_head weight / quant method；
底层 GEMM 或量化 kernel 输出 dtype；
embedding_bias dtype。
```

随后 `LogitsProcessor` 会执行：

```text
soft_cap / scale
```

这些操作可能保持原 dtype，也可能因 PyTorch 类型提升而变化。

真正采样时，Sampler 通常会把 logits 按采样需要转成 float32 或计算 float32 logprobs。比如 V1 采样文档中提到的采样阶段，会先处理 raw logits / logprobs，再做 penalties、temperature、top-k/top-p 等逻辑。

因此可以记成：

```text
compute_logits 阶段：
  输出“模型/量化 kernel 产生并经 gather/scale/softcap 后的 logits”。

sampler 阶段：
  根据采样与 logprobs 需求再做 float32 化或 log_softmax。
```

---

## 14. pooling / embedding 模型和 LM head 的关系

vLLM 中“embedding model”通常不是指 `VocabParallelEmbedding` 层，而是指任务类型为 embedding / pooling 的模型。

这类模型主链路是：

```text
input_ids / inputs_embeds
  → transformer hidden_states
  → pooler(hidden_states, pooling_metadata)
  → PoolerOutput / ModelRunnerOutput.pooler_output
```

不是：

```text
hidden_states → lm_head → logits → sampler
```

### 14.1 Pooler 接口

`Pooler` 定义：

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    pooling_metadata: PoolingMetadata,
) -> PoolerOutput:
    ...
```

位置：`vllm/vllm/model_executor/layers/pooler/abstract.py:30` 到 `abstract.py:36`

### 14.2 embedding model mixin

Transformers pooling backend 中：

```python
self.pooler = DispatchPooler.for_embedding(pooler_config)
```

位置：`vllm/vllm/model_executor/models/transformers/pooling.py:33` 到 `pooling.py:46`

### 14.3 ModelRunner 中 pooling 分支

V1 ModelRunner：

```python
if self.is_pooling_model:
    return self._pool(...)
```

位置：`gpu_model_runner.py:4348` 到 `gpu_model_runner.py:4355`

所以：

```text
生成模型：
  hidden_states[logits_indices] → compute_logits() → sampler

pooling / embedding 任务模型：
  hidden_states → pooler → pooler_output
```

---

## 15. 完整生成链路

把输入、模型、LM head、采样串起来：

```text
请求输入
  → tokenization / prompt_embeds / multimodal processing
  → SchedulerOutput
  → ModelRunner 准备 input_ids / positions / inputs_embeds
  → model.forward(...)
      → 如果 inputs_embeds 非空：直接作为 hidden_states
      → 否则：embed_tokens(input_ids)
      → decoder layers
      → final norm
      → hidden_states
  → hidden_states[logits_indices]
  → model.compute_logits(sample_hidden_states)
      → logits_processor(lm_head, sample_hidden_states)
      → lm_head.quant_method.apply(...)
      → TP gather / all-gather
      → remove vocab padding
      → soft_cap / scale
  → ExecuteModelState
  → sample_tokens()
  → Sampler
  → ModelRunnerOutput
```

其中 `lm_head.forward()` 不在链路里。

---

## 16. 容易疑惑的点

### 16.1 Embedding 是在 ModelRunner 里做的吗？

不是主要在 ModelRunner 里做。ModelRunner 准备 `input_ids` / `inputs_embeds`，真正 token embedding lookup 在模型 `forward()` 里调用 `embed_input_ids()` 完成。

### 16.2 inputs_embeds 和 input_ids 谁优先？

在 LLaMA 这类模型中，首个 PP rank 上 `inputs_embeds` 优先：如果它非空，就直接作为 hidden states；否则才用 `input_ids` 查 embedding。

### 16.3 LM head 是不是一个普通 Linear？

概念上类似 `hidden_states × W_vocab^T`，但 vLLM 中实现为 `ParallelLMHead` + `LogitsProcessor`。`ParallelLMHead.forward()` 会直接报错，权重由 `LogitsProcessor` 使用。

### 16.4 logits processor 是 OpenAI API 里的自定义 logits_processor 吗？

不是同一个层级。这里的 `model_executor.layers.logits_processor.LogitsProcessor` 是模型执行层，负责 LM head 投影、TP gather、soft cap、scale。采样阶段的 bad words、allowed tokens、temperature、top-k/top-p 等在 Sampler / sampling metadata 侧处理。

### 16.5 为什么 logits 可能是 None？

TP gather 路径下，`tensor_model_parallel_gather()` 可能只在目标 rank 返回完整 logits，其他 rank 返回 None；另外非最后 PP rank 不会进入普通 logits 计算路径。

### 16.6 tied embeddings 下 PP 为什么 last rank 也要 embed_tokens？

因为 last rank 的 `lm_head` 需要和 `embed_tokens.weight` 绑定。如果 embedding 本来只在 first rank，而 LM head 在 last rank，那么 tie weights 场景下 last rank 也要构造 `embed_tokens` 以持有可绑定的权重。

### 16.7 added vocab 为什么要单独处理？

LoRA / tokenizer 扩展可能带来 added tokens。vLLM 为了兼容权重加载、TP shard、padding 和 logits 重排，把原始 vocab 和 added vocab 分开计算 shard 区间，再放到同一个权重张量中。

### 16.8 embedding model 会用 LM head 吗？

通常不会。任务类型为 embedding / pooling 时，forward 后走 `pooler`，不会走 `compute_logits()` 和 sampler。

---

## 17. 总结

Embedding、LM head 和 logits processor 的关系可以压缩为：

```text
VocabParallelEmbedding：
  input_ids → vocab-parallel lookup → all-reduce → hidden_states

ParallelLMHead：
  保存 vocab-parallel 输出投影权重，不直接 forward

LogitsProcessor：
  hidden_states + lm_head weight → local logits → TP gather → remove padding → soft_cap / scale

ModelRunner：
  只对 logits_indices 对应 hidden states 调 compute_logits()，再交给 sampler
```

一句话总结：

```text
vLLM 把输入侧 embedding 做成 vocab-parallel lookup，把输出侧 LM head 做成“权重容器 + logits processor 投影”，并通过 inputs_embeds、多模态 merge、tie weights、LoRA added vocab、TP gather 和 logits_indices，把模型架构与高性能 serving 链路接起来。
```
