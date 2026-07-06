# 08. logits 与 sampling 相关算子如何工作？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\sampler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\metadata.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\ops\topk_topp_sampler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\rejection_sampler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\ops\logprobs.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\ops\penalties.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\ops\bad_words.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\structured_output\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\_custom_ops.py`

这个问题关注：logits processor、logprobs、top-k / top-p、temperature、repetition penalty、rejection sampling、structured output bitmask 等输出侧计算如何通过算子或张量操作完成。

---

## 1. 一句话回答

sampling / logits 算子负责把模型 forward 产出的 hidden states 或 logits，变成当前 step 的 sampled token ids，并在同一条链路上完成结构化输出约束、bad words、penalty、temperature、top-k / top-p、logprobs 和 speculative decoding 的接受 / 拒绝计算。

最小链路是：

```text
hidden states
  → logits processor / lm_head
  → grammar bitmask / allowed ids / bad words
  → penalties / logits processors
  → greedy or random sampler
  → sampled token ids / logprobs tensors
  → ModelRunnerOutput
```

在 V1 GPU worker 中，这条链路主要分成两段：

```text
execute_model()
  → forward
  → compute logits
  → 暂存 ExecuteModelState

sample_tokens(grammar_output)
  → apply_grammar_bitmask()
  → _sample()
  → Sampler 或 RejectionSampler
  → bookkeeping
  → ModelRunnerOutput
```

---

## 2. logits 从哪里来

采样阶段的输入不是直接从模型 forward 返回给用户，而是由 `GPUModelRunner.execute_model()` 在 forward 后保存到 `ExecuteModelState`。

核心状态包括：

```text
scheduler_output
logits
spec_decode_metadata
spec_decode_common_attn_metadata
hidden_states
sample_hidden_states
aux_hidden_states
ec_connector_output
cudagraph_stats
slot_mappings
```

位置：`vllm/v1/worker/gpu_model_runner.py:4386`

这样做的原因是：

```text
forward 只负责得到 hidden states / logits；
sampling 还需要 grammar、spec decode、logprobs、bookkeeping 等额外输入。
```

所以 generation 模型中经常看到：

```text
execute_model() 返回 None 或暂不返回最终 token；
sample_tokens() 才构造 ModelRunnerOutput。
```

---

## 3. sample_tokens 的输出侧主链路

`sample_tokens()` 的关键路径在 `GPUModelRunner` 中：

```python
def sample_tokens(self, grammar_output: "GrammarOutput | None")
```

位置：`vllm/v1/worker/gpu_model_runner.py:4422`

主要步骤是：

```text
1. 取出 execute_model_state 中的 logits 和 metadata；
2. 如果 grammar_output 存在，先对 logits 应用 grammar bitmask；
3. 调用 _sample(logits, spec_decode_metadata)；
4. 更新 input batch / spec decode / hybrid model 状态；
5. 做 sampled token、logprobs、prompt logprobs 的 bookkeeping；
6. 处理 draft tokens、KV connector、routed experts；
7. 构造 ModelRunnerOutput；
8. async scheduling 下包装成 AsyncGPUModelRunnerOutput。
```

普通采样和 speculative decoding 在 `_sample()` 分流：

```python
if spec_decode_metadata is None:
    return self.sampler(logits=logits, sampling_metadata=sampling_metadata)

sampler_output = self.rejection_sampler(
    spec_decode_metadata,
    draft_probs,
    logits,
    sampling_metadata,
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3570`

---

## 4. Sampler 的职责边界

`Sampler` 定义在：`vllm/v1/sample/sampler.py:20`

它的 `forward()` 是普通 generation step 的核心采样入口：

```python
def forward(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    predict_bonus_token: bool = False,
    logprobs_mode_override: LogprobsMode | None = None,
) -> SamplerOutput
```

位置：`vllm/v1/sample/sampler.py:72`

从代码注释和实现可以把它理解成 9 步：

```text
1. 如果请求 logprobs，先保留 raw logprobs 或 raw logits；
2. logits 转 float32；
3. 应用 allowed token ids whitelist；
4. 应用 bad words exclusion；
5. 应用会影响 argmax 的 logits processors；
6. 应用 repetition / frequency / presence penalties；
7. 执行 greedy 或 random sampling；
8. 收集 top logprobs、sampled token rank 或指定 token logprobs；
9. 返回 SamplerOutput。
```

这里的 `SamplerOutput` 仍是 GPU 张量结构：

```text
sampled_token_ids: torch.Tensor  # [num_requests, 1] 或 spec decode 下 [batch, max_spec_len + 1]
logprobs_tensors: LogprobsTensors | None
```

位置：`vllm/v1/outputs.py:189`

---

## 5. SamplingMetadata 携带哪些控制信息

Sampler 本身不直接读取用户请求对象，而是读取 `SamplingMetadata`。

位置：`vllm/v1/sample/metadata.py`

它通常携带：

```text
temperature
top_k
top_p
presence_penalties
frequency_penalties
repetition_penalties
prompt_token_ids
output_token_ids
bad_words_token_ids
allowed_token_ids_mask
logits processors
max_num_logprobs
logprob_token_ids
per-request generators
spec_token_ids
thinking budget state
```

所以 sampling 算子不是只看 logits，还要看每个 request 的 sampling params 和历史输出 token。

---

## 6. grammar / structured output 如何影响 logits

结构化输出约束在采样前执行，而不是采样后修正。

`sample_tokens()` 中：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`vllm/v1/worker/gpu_model_runner.py:4452`

这一步的作用是：

```text
grammar bitmask
  → 标记当前 request 允许 / 禁止的 token
  → 对 logits 中非法 token 写入 -inf
  → 后续 Sampler 只能从合法 token 空间采样
```

因此 structured output 是采样空间约束，不是输出文本后处理。

---

## 7. allowed ids、bad words 和 logits processors

`Sampler.apply_logits_processors()` 负责采样前的 logits 修改。

位置：`vllm/v1/sample/sampler.py:371`

顺序是：

```text
1. allowed_token_ids_mask：不允许的 token 直接 masked_fill(-inf)；
2. bad_words：根据已输出 token 序列屏蔽 bad words continuation；
3. non_argmax_invariant logits processors：例如 min tokens、logit bias；
4. penalties：repetition / frequency / presence；
5. thinking budget state：必要时继续修改 logits。
```

关键代码：

```python
if sampling_metadata.allowed_token_ids_mask is not None:
    logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

if bad_words_token_ids:
    apply_bad_words(logits, bad_words_token_ids, output_token_ids)

for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
    logits = processor.apply(logits)

logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
```

位置：`vllm/v1/sample/sampler.py:395`

这里大多数操作是 in-place 的，目的是减少额外 tensor 分配。

---

## 8. penalties 如何应用

penalty 入口是：

```python
return apply_all_penalties(
    logits,
    sampling_metadata.prompt_token_ids,
    sampling_metadata.presence_penalties,
    sampling_metadata.frequency_penalties,
    sampling_metadata.repetition_penalties,
    output_token_ids,
)
```

位置：`vllm/v1/sample/sampler.py:423`

它需要同时看：

```text
prompt token ids
已经生成的 output token ids
presence penalty
frequency penalty
repetition penalty
```

所以 penalty 本质是“基于历史 token 的 logits 重写”。这也是为什么 async scheduling 下需要先把上一轮 sampled token 更新进 `input_batch`，否则 penalty 看到的历史会落后一轮。

---

## 9. greedy sampling

如果不是全 random，Sampler 会先算 greedy token：

```python
greedy_sampled = logits.argmax(dim=-1).view(-1)
```

位置：`vllm/v1/sample/sampler.py:240`

如果 `sampling_metadata.all_greedy` 为真，流程会提前返回：

```text
logits processors / penalties
  → argmax
  → sampled token
  → 可选 processed logprobs
```

位置：`vllm/v1/sample/sampler.py:257`

这种路径不需要 top-k / top-p，也不需要随机数。

---

## 10. random sampling：temperature、top-k、top-p

随机采样入口是：

```python
random_sampled, processed_logprobs = self.topk_topp_sampler(
    logits,
    sampling_metadata.generators,
    sampling_metadata.top_k,
    sampling_metadata.top_p,
)
```

位置：`vllm/v1/sample/sampler.py:286`

在此之前会先应用 temperature：

```python
logits = logits.div_(temp.unsqueeze(dim=1))
```

位置：`vllm/v1/sample/sampler.py:227`

并应用只影响 random sampling、但不改变 greedy argmax 的 processors，例如默认的 min-p processor：

```python
for processor in sampling_metadata.logitsprocs.argmax_invariant:
    logits = processor.apply(logits)
```

位置：`vllm/v1/sample/sampler.py:280`

如果 batch 中有 greedy 和 random 混合请求，最终用 temperature 是否接近 0 来选择 greedy 或 random 结果：

```python
sampled = torch.where(
    sampling_metadata.temperature < _SAMPLING_EPS,
    greedy_sampled,
    random_sampled,
)
```

位置：`vllm/v1/sample/sampler.py:296`

---

## 11. TopKTopPSampler 的 backend 选择

`TopKTopPSampler` 定义在：`vllm/v1/sample/ops/topk_topp_sampler.py:70`

初始化时直接根据平台选择 `forward` 实现：

```text
CUDA：优先 FlashInfer sampler，否则 native；
CPU：部分架构走 native，否则 forward_cpu；
XPU：按 VLLM_XPU_USE_SAMPLER_KERNEL 选择 XPU kernel 或 native；
ROCm：如果 aiter sampling 可用，走 aiter，否则 native；
其他：native。
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:86`

### 11.1 CUDA / FlashInfer 路径

CUDA 上如果满足条件，使用：

```python
self.forward = self.forward_cuda
```

但 FlashInfer 不能用于所有情况：

```text
- logprobs_mode 不能是 processed_logits / processed_logprobs；
- 必须启用 VLLM_USE_FLASHINFER_SAMPLER；
- 必须是 CUDA 平台；
- GPU compute capability 必须被 FlashInfer 支持；
- 当前 step 不能有 per-request generators；
- use_fp64_gumbel 时回退 native；
- k 和 p 都为空时回退 native。
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:21` 和 `vllm/v1/sample/ops/topk_topp_sampler.py:147`

FlashInfer 的优势是避免对 logits 做完整排序：

```text
logits / probs
  → flashinfer.sampling.top_p / top_k / top_k_top_p
  → next_token_ids
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:471`

### 11.2 native 路径

native 路径是：

```text
apply_top_k_top_p()
  → softmax
  → exponential noise sampling
  → argmax
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:123`

随机采样没有直接用 `torch.multinomial`，而是用 exponential noise：

```python
q.exponential_()
return probs.div(q).argmax(dim=-1).view(-1)
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:446`

原因是注释中明确写了：`torch.multinomial` 会造成 CPU-GPU synchronization。

---

## 12. top-k / top-p mask 怎么算

入口是：

```python
def apply_top_k_top_p(logits, k, p)
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:345`

选择逻辑：

```text
p 和 k 都为空：直接返回 logits；
CPU：有 Triton 用 Triton，否则 PyTorch；
非 CPU 且 HAS_TRITON 且 batch >= 8：用 Triton；
小 batch：用 PyTorch sort/topk。
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:345`

PyTorch fallback 中：

```text
1. 对 vocab 维度排序；
2. top-k：低于第 k 大阈值的 logits 设为 -inf；
3. top-p：按概率累计，低概率尾部设为 -inf；
4. scatter 回原 vocab 顺序。
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:363`

top-k only 在 CPU 上还有一个避免完整排序的路径：

```python
return apply_top_k_only(logits, k)
```

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:407`

但注释也提醒：这个路径涉及 GPU→CPU sync，可能影响 async scheduling 性能。

---

## 13. logprobs 如何计算和收集

Sampler 支持几类 logprobs 模式：

```text
raw_logprobs：对原始 logits 做 log_softmax；
raw_logits：直接保留原始 logits；
processed_logprobs：对处理后的 logits 做 log_softmax；
processed_logits：保留处理后的 logits。
```

普通路径中，如果请求 `num_logprobs` 或 `logprob_token_ids`，会先准备 `raw_logprobs`：

```python
raw_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
```

位置：`vllm/v1/sample/sampler.py:85` 和 `vllm/v1/sample/sampler.py:304`

### 13.1 top logprobs

`gather_logprobs()` 做三件事：

```text
1. torch.topk(logprobs, num_logprobs) 取 top token；
2. gather 当前 sampled token 的 logprob；
3. batched_count_greater_than 计算 sampled token rank；
4. 拼成 LogprobsTensors。
```

位置：`vllm/v1/sample/sampler.py:309`

这里还用到：

```python
torch._dynamo.decorators.mark_unbacked(logprobs, 0)
```

位置：`vllm/v1/sample/sampler.py:345`

目的是避免 `torch.compile` 在 batch size 从 1 变到大于 1 时产生不必要的 specialization recompile。

### 13.2 指定 token logprobs

`gather_specific_token_logprobs()` 用于 generative scoring 这类 API，只取用户指定 token 的 logprobs，而不是整行 top-k。

位置：`vllm/v1/sample/sampler.py:151`

它会构造 padded token id 矩阵，把 sampled token 放在第 0 列，然后一次 gather：

```text
[token sampled, token A, token B, ...]
  → gather logprobs
  → mask padding 为 -inf
  → 计算 sampled token rank
```

---

## 14. prompt logprobs 与 sampled logprobs 的区别

sampled logprobs 对应本 step 生成出来的 token。

prompt logprobs 对应 prompt token 本身的概率，通常在 bookkeeping / output processor 阶段按 request 组织成：

```text
prompt_logprobs_dict: dict[str, LogprobsTensors | None]
```

位置：`vllm/v1/outputs.py:251`

也就是说：

```text
Sampler 负责生成底层 LogprobsTensors；
GPUModelRunner bookkeeping 负责把它们按 request id、prompt / output 语义重新组织。
```

---

## 15. speculative decoding 的 rejection sampler

如果 `spec_decode_metadata` 不为空，`GPUModelRunner._sample()` 不走普通 `Sampler`，而是走：

```python
self.rejection_sampler(
    spec_decode_metadata,
    draft_probs,
    logits,
    sampling_metadata,
)
```

位置：`vllm/v1/worker/gpu_model_runner.py:3592`

`RejectionSampler` 定义在：`vllm/v1/sample/rejection_sampler.py:37`

它实现的是论文 `Accelerating Large Language Model Decoding with Speculative Sampling` 中的接受 / 拒绝算法。代码注释把 token 分成三类：

```text
accepted tokens：draft token 被 target distribution 接受；
recovered tokens：拒绝后从调整后的分布重新采样；
bonus tokens：所有 draft 都接受时，额外追加一个 target token。
```

最终输出：

```text
output tokens = accepted tokens + recovered tokens + bonus token
```

位置：`vllm/v1/sample/rejection_sampler.py:37`

---

## 16. RejectionSampler 的执行顺序

`RejectionSampler.forward()` 的关键步骤：

```text
1. 根据 bonus_logits_indices 取 bonus logits；
2. 调普通 Sampler 采样 bonus token；
3. 根据 target_logits_indices 取 target logits；
4. 对 target logits 应用 penalties / bad words / allowed ids / min tokens；
5. 对 target logits 应用 temperature / top-k / top-p；
6. 调 rejection_sample() 生成最终 token 矩阵；
7. 如需 logprobs，重新收集 accepted / bonus token 的 logprobs；
8. 返回 SamplerOutput。
```

位置：`vllm/v1/sample/rejection_sampler.py:88`

注意：bonus token 是通过普通 Sampler 采样出来的，因为 bonus token 仍然需要支持 top-k / top-p 等普通采样策略。

---

## 17. rejection_sample 的 kernel 路径

`rejection_sample()` 里会先创建输出 buffer：

```python
output_token_ids = torch.full(
    (batch_size, max_spec_len + 1),
    PLACEHOLDER_TOKEN_ID,
    dtype=torch.int32,
    device=device,
)
```

位置：`vllm/v1/sample/rejection_sampler.py:427`

然后分两类：

### 17.1 greedy spec decode

如果不是全 random，会先跑 greedy rejection kernel：

```python
rejection_greedy_sample_kernel[(batch_size,)](...)
```

位置：`vllm/v1/sample/rejection_sampler.py:453`

它比较 draft token 和 target argmax：

```text
相同：接受 draft token；
不同：写入 target argmax，并停止继续接受后续 draft；
全部接受：追加 bonus token。
```

Triton kernel 定义位置：`vllm/v1/sample/rejection_sampler.py:713`

### 17.2 random spec decode

random sampling 下会：

```text
1. target_logits → target_probs；
2. sample_recovered_tokens() 为每个位置准备 recovered token；
3. rejection_random_sample_kernel 根据 draft_probs、target_probs、uniform_probs 接受或拒绝；
4. 输出 [batch, max_spec_len + 1] token 矩阵。
```

位置：`vllm/v1/sample/rejection_sampler.py:471`

`sample_recovered_tokens()` 也使用 Triton kernel：

```python
sample_recovered_tokens_kernel[(batch_size, max_spec_len)](...)
```

位置：`vllm/v1/sample/rejection_sampler.py:663`

---

## 18. grammar、penalty 和 spec decode 的组合

spec decode 下不能简单把 per-request metadata 直接套到 target logits 上，因为 target logits 是按 draft token 展平的：

```text
[num_tokens, vocab_size]
```

因此 `RejectionSampler.apply_logits_processors()` 会先构造 repeat indices：

```python
original_indices.repeat_interleave(num_draft_tokens)
```

位置：`vllm/v1/sample/rejection_sampler.py:312`

然后把 request 级别的 penalty、allowed ids、thinking budget state 展开到 token 级别。

这解释了为什么 spec decode 的 sampling 代码比普通 sampling 多一层 `cu_num_draft_tokens`、`num_draft_tokens` 和 index 映射。

---

## 19. CPU / GPU 边界

sampling 链路中需要特别注意几类同步：

```text
- sampled_token_ids 最终要转成 Python list；
- logprobs tensors 最终要转成 LogprobsLists；
- per-request generators 可能让某些 optimized backend 回退；
- top-k only 的某些 fallback 可能触发 CPU sync；
- async scheduling 下会把 D2H copy 放到独立 stream。
```

普通同步输出会在 `_bookkeeping_sync()` 里完成 CPU 化；async scheduling 则用 `AsyncGPUModelRunnerOutput` 延迟等待 copy event。

位置：`vllm/v1/worker/gpu_model_runner.py:3601` 和 `vllm/v1/worker/gpu_model_runner.py:239`

---

## 20. 与 CUDA Graph / torch compile 的关系

sampling 里有几处明显为 compile / cudagraph 做的处理：

```text
- 多数 logits 修改尽量 in-place，减少动态分配；
- TopKTopPSampler 初始化时固定 forward 分支，避免运行中频繁选择；
- gather_logprobs 使用 mark_unbacked 避免 batch size specialization；
- rejection sampler 的 Triton kernel 对 max_spec_len 使用 do_not_specialize；
- FlashInfer / Triton / PyTorch fallback 的分支会影响可捕获性和性能。
```

但严格来说，采样不一定都进入 full CUDA graph。模型 forward、attention、MLP 等算子更典型地处在 cudagraph replay 路径；sampling 更常见的问题是 D2H 同步和动态 shape / 动态 request metadata 的开销。

---

## 21. 常见性能判断

### 21.1 为什么 top-p 慢

top-p 需要排序或专用采样 backend：

```text
sort vocab
  → softmax
  → cumsum
  → mask
  → scatter
```

大 vocab、大 batch 下 PyTorch fallback 会比较重，所以 CUDA 上优先 FlashInfer 或 Triton。

### 21.2 为什么 per-request seed 会变慢

`generators` 非空时，FlashInfer / XPU / ROCm aiter 等路径可能不支持 per-request generator，只能 fallback 到 native，并逐 request 生成随机数。

位置：`vllm/v1/sample/ops/topk_topp_sampler.py:156`

### 21.3 为什么 logprobs 会增加成本

请求 logprobs 会额外做：

```text
log_softmax over vocab
torch.topk 或 gather
rank 计算
D2H copy / Python list 转换
```

因此 `max_num_logprobs`、`prompt_logprobs`、`logprob_token_ids` 都会增加输出侧开销。

---

## 22. 容易疑惑的点

### 22.1 logits processor 和 sampler 是一回事吗？

不是。

```text
logits processor：修改 logits 分布；
sampler：从分布中选择 token。
```

但在 V1 里二者都封装在 `Sampler.forward()` 主流程中。

### 22.2 structured output 为什么要改 logits？

因为只有在采样前把非法 token 设为 `-inf`，才能保证采样结果天然满足 grammar 约束。

### 22.3 top-k / top-p 是 CUDA custom op 吗？

不固定。可能是 FlashInfer、ROCm aiter、XPU custom op、Triton，也可能是 PyTorch fallback。

### 22.4 sampled_token_ids 为什么最后是 int32？

Sampler 采样后会转成 int64 以便索引，然后在返回前转成 int32 减少 tensor 大小。

位置：`vllm/v1/sample/sampler.py:105` 和 `vllm/v1/sample/sampler.py:138`

### 22.5 spec decode 为什么输出二维 token 矩阵？

因为一个 request 在一个 step 内可能接受多个 draft token，并额外追加 bonus token，所以输出 shape 是：

```text
[batch_size, max_spec_len + 1]
```

无效位置用 `PLACEHOLDER_TOKEN_ID = -1` 标记，后续 parse / bookkeeping 会过滤。

---

## 23. 总结

logits / sampling 链路可以压缩成：

```text
logits
  → grammar / allowed ids / bad words
  → logits processors / penalties
  → temperature
  → top-k / top-p backend
  → greedy or random token
  → logprobs gather
  → spec decode rejection（可选）
  → bookkeeping / async output copy
```

如果只记住一句话：

```text
vLLM 的 sampling 不是单个 multinomial 调用，而是一条围绕 request metadata、输出约束、logprobs、spec decode 和 backend fallback 组织起来的输出侧算子流水线。
```
