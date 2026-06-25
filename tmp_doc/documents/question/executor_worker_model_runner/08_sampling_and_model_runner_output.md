# 08. sample_tokens() 如何生成 ModelRunnerOutput？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu\model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\sample\sampler.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\outputs.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\structured_output\utils.py`

本问题关注：`execute_model()` 已经完成 forward 和 logits 计算后，`sample_tokens()` 如何把 logits 变成真正的 sampled token、logprobs、prompt logprobs、routed experts、KV connector output、EC connector output，并最终构造 `ModelRunnerOutput`。同时也要解释异步输出包装 `AsyncGPUModelRunnerOutput` 如何把 GPU 张量复制到 CPU 并返回最终结果。

---

## 1. 一句话回答

`sample_tokens()` 是采样链路的收尾阶段：

```text
logits
  → apply_grammar_bitmask()
  → Sampler.sample()
  → logprobs / prompt_logprobs / bookkeeping
  → speculative decoding / draft proposal
  → ModelRunnerOutput
  → （异步场景）AsyncModelRunnerOutput.get_output()
```

如果是 generation 模型，通常流程是：

```text
execute_model()
  → forward + logits
  → 保存 ExecuteModelState
  → return None

sample_tokens(grammar_output)
  → 解包 ExecuteModelState
  → 应用 grammar bitmask
  → 调用 Sampler
  → 更新状态 / 计算 logprobs / draft tokens
  → 构造 ModelRunnerOutput
  → 同步返回或异步包装返回
```

所以可以记成：

```text
execute_model 负责“算出 logits”；
sample_tokens 负责“把 logits 变成输出”。
```

---

## 2. 采样主链路

GPUModelRunner 中的主链路在：

```python
def sample_tokens(self, grammar_output: "GrammarOutput | None")
```

位置：`gpu_model_runner.py:4422`

核心步骤：

```text
1. 从 self.execute_model_state 解包 forward 的临时状态；
2. 如果有 grammar_output，就应用 grammar bitmask；
3. 调用 self._sample(logits, spec_decode_metadata)；
4. 调用 _update_states_after_model_execute()；
5. 处理 async scheduling / PP 广播；
6. 处理 speculative decoding draft tokens；
7. finalize KV connector；
8. 可选附加 routed experts；
9. 构造 ModelRunnerOutput；
10. 如果是 async scheduling，则包装成 AsyncGPUModelRunnerOutput。
```

对应源码：`gpu_model_runner.py:4436` 到 `gpu_model_runner.py:4682`

---

## 3. sample_tokens 的输入来自哪里

`sample_tokens()` 的输入不是 raw logits，而是 `execute_model()` 暂存的 `ExecuteModelState`。

`ExecuteModelState` 包含：

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

位置：`gpu_model_runner.py:4386` 到 `gpu_model_runner.py:4397`

### 3.1 为什么要先暂存

因为生成型输出不是单纯“跑完 forward 就结束”，而是还要进行：

```text
grammar bitmask
Sampler.sample()
logprobs 收集
prompt_logprobs 计算
speculative decoding
bookkeeping
```

所以 forward 和 sampling 被拆成两个阶段。

### 3.2 如果 execute_model 失败

如果 `self.execute_model_state is None`，说明上一次 `execute_model()` 没有成功完成：

```python
if self.execute_model_state is None:
    kv_connector_output = self.kv_connector_output
    self.kv_connector_output = None
    if self.use_async_scheduling and not get_pp_group().is_last_rank:
        self._pp_receive_prev_sampled_token_ids_to_input_batch()
    return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

位置：`gpu_model_runner.py:4426` 到 `gpu_model_runner.py:4434`

这时只可能返回一个带 KV connector 信息的空输出，或者更一般地说，说明前一步 forward 没有留下可采样状态。

---

## 4. grammar / structured output 如何参与采样

`sample_tokens()` 先处理结构化输出约束：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`gpu_model_runner.py:4452` 到 `gpu_model_runner.py:4456`

### 4.1 grammar_output 是什么

`grammar_output` 是 Scheduler 生成的 grammar bitmask，来自：

```python
self.scheduler.get_grammar_bitmask(scheduler_output)
```

位置：`engine/core.py:492`

它告诉采样器：

```text
当前哪些 token 允许采样，哪些 token 必须屏蔽。
```

### 4.2 为什么不是后处理

结构化输出不是采样后再修正，而是在采样前通过 bitmask 限制 logits：

```text
apply_grammar_bitmask()
  → 修改 logits 可选空间
  → Sampler.sample()
```

这样采样结果从一开始就满足结构约束。

### 4.3 与 Scheduler 的关系

Scheduler 负责：

```text
- 判断请求是否处于 grammar 等待状态；
- 生成本轮 grammar bitmask；
- 把它交给 EngineCore；
- EngineCore 传给 sample_tokens。
```

Worker / ModelRunner 只负责按 bitmask 执行。

---

## 5. _sample() 是什么

采样核心封装在：

```python
def _sample(
    self,
    logits: torch.Tensor | None,
    spec_decode_metadata: SpecDecodeMetadata | None,
) -> SamplerOutput:
```

位置：`gpu_model_runner.py:3570`

它负责把 logits 交给 `Sampler`，得到：

```text
sampled_token_ids
logprobs_tensors
```

### 5.1 Sampler 的工作方式

`Sampler` 定义在：`vllm/v1/sample/sampler.py:20`

其采样流程注释写得很清楚：

```text
1. 如需 logprobs，先计算 raw_logprobs 或保留 raw_logits；
2. logits 转 float32；
3. 处理 allowed token ids；
4. 处理 bad words；
5. 处理非 argmax-invariant logits processors；
6. 处理 penalties；
7. 执行 greedy 或 random sample；
8. 收集 top-k / sampled token 的 logprobs；
9. 返回 SamplerOutput。
```

位置：`sample.py:20` 到 `sample.py:58`

### 5.2 Sampler.sample 的输入输出

```python
sampled, processed_logprobs = self.sample(logits, sampling_metadata)
```

位置：`sample.py:243` 起

它会返回：

```text
sampled: [batch] token ids
processed_logprobs: 可选的 logprobs tensor
```

然后 `Sampler.forward()` 会把 sampled token reshape 成 `SamplerOutput.sampled_token_ids`：

```python
sampled_token_ids=sampled.unsqueeze(-1)
```

位置：`sample.py:141` 到 `sample.py:148`

---

## 6. Sampler 具体做了什么

Sampler 的核心步骤如下。

### 6.1 计算 raw_logprobs

如果请求了 logprobs 或 `logprob_token_ids`，Sampler 会先计算：

```python
raw_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
```

位置：`sample.py:85` 到 `sample.py:94`，以及 `sample.py:305` 到 `sample.py:306`

如果 `logprobs_mode == raw_logits`，则直接保留 logits 作为“原始 logprobs 载体”。

### 6.2 处理 logits processors 和 penalties

Sampler 依次应用：

```text
allowed token ids
bad words
non argmax-invariant logits processors
temperature
random/greedy sampling
```

位置：`sample.py:98` 到 `sample.py:102`，`sample.py:273` 到 `sample.py:302`

### 6.3 采样 token

```python
random_sampled, processed_logprobs = self.topk_topp_sampler(...)
```

位置：`sample.py:286` 到 `sample.py:291`

如果 `all_greedy`，则直接返回 argmax；
如果 `all_random`，则直接随机采样；
否则根据 temperature 决定 greedy 和 random 的混合策略。

### 6.4 收集 logprobs

Sampler 支持两种 logprobs 需求：

```text
- num_logprobs：取 top-k logprobs；
- logprob_token_ids：只取指定 token 的 logprobs。
```

位置：`sample.py:84` 到 `sample.py:137`

最终返回：

```python
SamplerOutput(
    sampled_token_ids=sampled.unsqueeze(-1),
    logprobs_tensors=logprobs_tensors,
)
```

位置：`sample.py:141` 到 `sample.py:148`

---

## 7. sample_tokens 里的采样输出结构

`SamplerOutput` 是 GPU 侧张量输出，包含：

```text
sampled_token_ids: torch.Tensor
logprobs_tensors: LogprobsTensors | None
```

位置：`outputs.py:189` 到 `outputs.py:193`

而 `sample_tokens()` 里最终要产出的是：

```text
ModelRunnerOutput
```

它是用户可见输出之前的内部输出结构。

---

## 8. 采样后为什么还要更新状态

采样后立即调用：

```python
self._update_states_after_model_execute(
    sampler_output.sampled_token_ids, scheduler_output
)
```

位置：`gpu_model_runner.py:4461` 到 `gpu_model_runner.py:4463`

这是为了处理：

```text
speculative decoding
hybrid models / Mamba
accepted token 计数
GPU / CPU 状态同步
```

该步骤不直接生成输出，但会影响下一轮 batch 和 token 状态。

---

## 9. Speculative decoding 在 sample 阶段怎么走

`sample_tokens()` 里有一整段 speculative decoding 逻辑。

### 9.1 先清理旧 draft 状态

```python
self._draft_token_ids = None
self._draft_probs = None
self._draft_prob_req_ids = None
self._draft_token_req_ids = None
self.valid_sampled_token_count_gpu = None
self.input_batch.prev_sampled_token_ids = None
```

位置：`gpu_model_runner.py:4474` 到 `gpu_model_runner.py:4479`

### 9.2 生成 draft tokens 的辅助函数

内部定义：

```python
def propose_draft_token_ids(sampled_token_ids):
    ...
    self._draft_token_ids = self.propose_draft_token_ids(...)
    self._copy_draft_token_ids_to_cpu(scheduler_output)
```

位置：`gpu_model_runner.py:4481` 到 `gpu_model_runner.py:4496`

### 9.3 draft model / ngram GPU / Eagle 等分支

当 `speculative_config` 存在时，会根据方法选择：

```text
EAGLE / draft model / extract hidden states
ngram GPU
其他方法
```

位置：`gpu_model_runner.py:4497` 到 `gpu_model_runner.py:4573`

核心目标是：

```text
生成下一轮 draft tokens；
如果输入不适合 drafter，则清零 draft tokens，避免调度到旧 draft。
```

### 9.4 为什么 spec decode 需要在 bookkeeping 前后分支

有些 spec decode 方法使用 GPU sampled tokens，可以先提 draft；
有些基于 CPU sampled tokens，则要等 bookkeeping 完成后再提 draft。

所以代码里有：

```python
propose_drafts_after_bookkeeping = True/False
```

位置：`gpu_model_runner.py:4498` 到 `gpu_model_runner.py:4594`

---

## 10. bookkeeping 是什么

采样后会进入：

```python
with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
    (
        num_nans_in_logits,
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
        invalid_req_indices,
    ) = self._bookkeeping_sync(...)
```

位置：`gpu_model_runner.py:4574` 到 `gpu_model_runner.py:4589`

它负责：

```text
- 处理 sampled token 的有效性；
- 生成 logprobs_lists；
- 生成 prompt_logprobs_dict；
- 记录 num_nans_in_logits；
- 生成最终输出用的 req_ids / req_id_to_index 副本；
- 标记 invalid request indices。
```

bookkeeping 是从 GPU 采样结果到结构化输出对象之间的重要桥梁。

---

## 11. prompt logprobs 是怎么来的

`ModelRunnerOutput` 里有：

```python
prompt_logprobs_dict: dict[str, LogprobsTensors | None]
```

位置：`outputs.py:251` 到 `outputs.py:257`

在 V1 GPUModelRunner 中，这部分通常在 bookkeeping 阶段计算并填充。

在 sampler / logprobs 侧，`SamplingMetadata` 会携带：

```text
max_num_logprobs
logprob_token_ids
```

位置：`sample/metadata.py:25` 到 `sample/metadata.py:49`

Sampler 也支持：

```python
gather_specific_token_logprobs()
```

位置：`sample.py:151` 到 `sample.py:225`

这意味着：

```text
如果用户要求 prompt_logprobs 或特定 token logprobs，
采样阶段会保留足够的 raw logprobs，以便在 bookkeeping / output 阶段组织成最终格式。
```

---

## 12. ModelRunnerOutput 的字段含义

定义在：`outputs.py:234`

### 12.1 req_ids / req_id_to_index

```python
req_ids: list[str]
req_id_to_index: dict[str, int]
```

表示：

```text
这一批输出对应哪些请求；
每个 request id 在 batch 里的索引是什么。
```

### 12.2 sampled_token_ids

```python
sampled_token_ids: list[list[int]]
```

表示：

```text
每个请求在当前 step 生成的 token ids。
```

注释说明这里可能不是固定每个请求 1 个 token，因为 speculative / jump decoding 下每个请求生成 token 数可以不同。

### 12.3 logprobs

```python
logprobs: LogprobsLists | None
```

这是最终的生成 token logprobs 结果。

`LogprobsLists` 包含：

```text
logprob_token_ids
logprobs
sampled_token_ranks
cu_num_generated_tokens
```

位置：`outputs.py:27` 到 `outputs.py:46`

### 12.4 prompt_logprobs_dict

```python
prompt_logprobs_dict: dict[str, LogprobsTensors | None]
```

表示每个请求的 prompt token logprobs，或 None。

### 12.5 pooler_output

```python
pooler_output: list[torch.Tensor | None] | None
```

这是 pooling 模型用的字段；生成模型一般不用。

### 12.6 kv_connector_output

```python
kv_connector_output: KVConnectorOutput | None
```

表示 KV transfer / connector 的回传信息。

### 12.7 ec_connector_output

```python
ec_connector_output: ECConnectorOutput | None
```

表示 encoder cache / multimodal connector 的输出。

### 12.8 num_nans_in_logits

```python
num_nans_in_logits: dict[str, int] | None
```

用于记录 logits 中 NaN 数量，方便诊断。

### 12.9 cudagraph_stats

```python
cudagraph_stats: CUDAGraphStat | None
```

记录本 step 的 CUDA graph 执行统计。

### 12.10 routed_experts

```python
routed_experts: RoutedExpertsLists | None
```

用于 MoE / routed experts 的逐步路由输出。

---

## 13. async scheduling 下的输出包装

如果 `self.use_async_scheduling` 为真，`sample_tokens()` 最后不会立刻返回普通 `ModelRunnerOutput`，而会包装成：

```python
AsyncGPUModelRunnerOutput(
    model_runner_output=output,
    sampled_token_ids=sampler_output.sampled_token_ids,
    logprobs_tensors=sampler_output.logprobs_tensors,
    invalid_req_indices=invalid_req_indices,
    async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
    vocab_size=self.input_batch.vocab_size,
    routed_experts=routed_experts_snapshot,
)
```

位置：`gpu_model_runner.py:4663` 到 `gpu_model_runner.py:4671`

### 13.1 为什么需要包装

因为异步场景下：

```text
GPU tensor 的 D2H copy 可以和下一步 compute 重叠；
不能等所有拷贝完再继续调度。
```

所以先返回一个异步包装对象，真正的 CPU 结果由 `get_output()` 取出。

### 13.2 AsyncGPUModelRunnerOutput 的工作方式

定义在：`gpu_model_runner.py:239`

它会：

```text
1. 在单独 copy stream 上启动 sampled_token_ids / logprobs / routed_experts 的 D2H 拷贝；
2. 用 event 标记 copy 完成；
3. get_output() 时等待 event；
4. 转成 list / numpy / Python 结构；
5. 回填到 ModelRunnerOutput。
```

位置：`gpu_model_runner.py:239` 到 `gpu_model_runner.py:316`

### 13.3 sampled_token_ids 和 logprobs 的 CPU 化

`get_output()` 里会：

```text
- 将 sampled_token_ids_cpu 转成 list[list[int]]；
- 将 logprobs_tensors_cpu 转成 LogprobsLists；
- 对 invalid requests 清空 token 列表；
- 把 routed_experts 回填到 output。
```

位置：`gpu_model_runner.py:282` 到 `gpu_model_runner.py:316`

---

## 14. 返回 ModelRunnerOutput 之后发生什么

`ModelRunnerOutput` 不是最终用户输出，但已经是 Scheduler 可消费的内部结果。

它会被 `Scheduler.update_from_output(scheduler_output, model_output)` 使用，继续做：

```text
- request 状态推进；
- sampled token 写回；
- stop / finish 判断；
- KV block 释放；
- EngineCoreOutputs 生成。
```

所以 `ModelRunnerOutput` 是：

```text
Worker → Scheduler 的对账凭证。
```

---

## 15. pooled / generation / PP 三类输出的差异

### 15.1 generation 模型

```text
forward → hidden_states → logits → sampler → ModelRunnerOutput
```

### 15.2 pooling 模型

```text
forward → pooler → pooling output → ModelRunnerOutput
```

### 15.3 PP 非最后 rank

```text
forward → IntermediateTensors → 下一 stage
sample_tokens() 可能只负责接收 / 转发 / kv connector output
```

这三条路径共享 `sample_tokens()` 这个名字，但实际输出结构不同。

---

## 16. 一个完整的采样时间线

可以把一次采样理解成下面的顺序：

```text
1. execute_model() 保存 logits 和上下文；
2. EngineCore 取 grammar_output；
3. sample_tokens(grammar_output)；
4. apply_grammar_bitmask()；
5. Sampler.sample()；
6. _update_states_after_model_execute()；
7. speculative decoding 处理；
8. bookkeeping；
9. 构造 ModelRunnerOutput；
10. 必要时包装为 AsyncGPUModelRunnerOutput；
11. Scheduler.update_from_output()。
```

---

## 17. 容易疑惑的点

### 17.1 sample_tokens 是不是只做采样？

不是。

它还负责：

```text
grammar bitmask
spec decode draft proposal
bookkeeping
KV connector finalize
routed experts
async output packaging
```

### 17.2 为什么 sampled_token_ids 是 list[list[int]]？

因为 speculative / jump decoding 下每个请求一轮生成的 token 数可能不同。

### 17.3 为什么 logprobs 不是直接 tensor 返回？

因为最终用户侧和 Scheduler 侧更容易消费 Python list / 结构化对象，而不是 raw tensor。

### 17.4 为什么 async 需要 AsyncGPUModelRunnerOutput？

为了把 D2H copy 和后续 compute 重叠，减少同步等待。

### 17.5 为什么 grammar bitmask 在采样前应用？

因为结构化输出要求采样结果本身就满足约束，而不是采样后再修正。

### 17.6 为什么 model_output 有时是 None？

因为 `execute_model()` 已经完成 forward，但真正输出要在 `sample_tokens()` 中完成。

---

## 18. 总结

采样链路可以压缩为：

```text
logits
  → grammar bitmask
  → Sampler
  → logprobs / prompt logprobs
  → speculative decoding
  → bookkeeping
  → ModelRunnerOutput
  → Async wrapper（可选）
```

如果只记住一句话：

```text
sample_tokens() 把 forward 得到的 logits 变成可回传 Scheduler 的结构化输出，同时处理 grammar、logprobs、spec decode 和异步拷贝。
```
