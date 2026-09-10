# 05 LogitsProcessor、Logprobs 与 Structured Output

本篇集中梳理三个容易混淆的部分：

1. hidden states 到 logits 的模型层 `LogitsProcessor`；
2. sampling 阶段对 logits 的各种修改和 logprobs 计算；
3. structured output / reasoning 如何影响采样。

## 1. 两类“logits processor”不要混淆

vLLM 里有两类名字相近但层级不同的处理：

### 1.1 模型层 LogitsProcessor

文件：`code/vllm/vllm/model_executor/layers/logits_processor.py`

定义：`code/vllm/vllm/model_executor/layers/logits_processor.py:18`

职责：

```text
hidden states -> lm_head projection -> vocab logits
```

它主要处理：

- lm head 投影；
- tensor parallel logits gather；
- vocab padding 裁剪；
- soft cap；
- logits scale。

关键锚点：

- `forward()`：`code/vllm/vllm/model_executor/layers/logits_processor.py:54`
- `_get_logits()`：`code/vllm/vllm/model_executor/layers/logits_processor.py:89`
- lm head 投影：`code/vllm/vllm/model_executor/layers/logits_processor.py:96`

### 1.2 sampling 阶段 logits processors / states

新 GPU worker 中，这些逻辑分散在：

- `code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py`
- `code/vllm/vllm/v1/worker/gpu/sample/penalties.py`
- `code/vllm/vllm/v1/worker/gpu/sample/bad_words.py`
- `code/vllm/vllm/v1/worker/gpu/sample/sampler.py`

旧路径中有更显式的 logits processor 分组：

- interface：`code/vllm/vllm/v1/sample/logits_processor/interface.py:60`
- state：`code/vllm/vllm/v1/sample/logits_processor/state.py:148`

旧路径会区分：

- `argmax_invariant`；
- `non_argmax_invariant`。

含义：有些处理不改变 argmax 结果，可以在 temperature 之后、top-k/top-p 之前应用；有些处理会影响 greedy argmax，需要更早应用。

## 2. sampling 阶段 logits 修改顺序

新 GPU sampler 主入口：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:72`

sampling 参数应用核心：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:146`

顺序：

```text
logits
  ↓ fp32
logit bias / allowed token ids / min tokens / stop ids
  ↓
repetition / frequency / presence penalties
  ↓
bad words mask
  ↓
temperature
  ↓
min-p
  ↓
top-k / top-p
  ↓
采样
```

锚点：

- fp32：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:156`
- logit bias/allowed/min tokens：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:159`
- penalties：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:164`
- bad words：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:173`
- temperature：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:182`
- min-p：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:187`
- top-k/top-p：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:193`

## 3. logit_bias、allowed_token_ids、min_tokens

文件：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py`

类：`LogitBiasState`，`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:15`

### 3.1 allowed_token_ids

字段锚点：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:19`

实现思路：

```text
如果请求配置 allowed_token_ids：
  先把该请求整行 logits 设置为 -inf
  再把 allowed token 对应 logits 恢复
```

相关锚点：

- 整行 mask：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:177`
- 恢复 allowed tokens：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:192`
- 应用位置：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:201`

### 3.2 logit_bias

字段锚点：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:28`

它把用户指定 token id 的 logits 加上一个 bias。OpenAI API 中常用来禁止或鼓励某些 token。

### 3.3 min_tokens / stop ids

字段锚点：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:40`

当生成 token 数还没达到 `min_tokens` 时，EOS/stop token ids 会被 mask 掉，防止过早停止。

注意：

- 这是采样前的约束；
- 真正判断“已经停止”仍在 scheduler 的 `check_stop()` 中完成。

## 4. penalties

文件：`code/vllm/vllm/v1/worker/gpu/sample/penalties.py`

类：`PenaltiesState`，`code/vllm/vllm/v1/worker/gpu/sample/penalties.py:14`

应用入口：`code/vllm/vllm/v1/worker/gpu/sample/penalties.py:81`

处理：

- repetition penalty；
- frequency penalty；
- presence penalty。

这些 penalty 需要知道 prompt tokens 和 already generated tokens，因此 batch state 需要维护请求历史。

## 5. bad words

文件：`code/vllm/vllm/v1/worker/gpu/sample/bad_words.py`

类：`BadWordsState`，`code/vllm/vllm/v1/worker/gpu/sample/bad_words.py:15`

应用入口：`code/vllm/vllm/v1/worker/gpu/sample/bad_words.py:72`

bad words 通常是 token 序列，不只是单 token。它需要结合当前已生成后缀判断下一 token 是否会形成 bad word。

## 6. logprobs 数据结构

### 6.1 LogprobsLists

定义：`code/vllm/vllm/v1/outputs.py:27`

字段：

- `logprob_token_ids`；
- `logprobs`；
- `sampled_token_ranks`；
- `cu_num_generated_tokens`。

`slice_request()` 用于每个请求生成 token 数不同的场景，尤其是 spec decode。

### 6.2 LogprobsTensors

定义：`code/vllm/vllm/v1/outputs.py:52`

功能：

- tensor 形式保存 token ids/logprobs/ranks；
- 支持转 CPU；
- 支持 filter；
- 支持转 Python list。

常见流转：

```text
GPU sampler -> LogprobsTensors
  -> AsyncOutput D2H
  -> LogprobsLists
  -> Scheduler.slice_request()
  -> EngineCoreOutput.new_logprobs
  -> LogprobsProcessor
  -> CompletionOutput.logprobs
```

## 7. GPU logprobs 计算

文件：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py`

关键函数：

- `compute_token_logprobs()`：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:78`
- `compute_topk_logprobs()`：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:101`

设计重点：不 materialize 完整 `[batch_size, vocab_size]` logprobs。

原因：完整 vocab logprobs 显存和带宽成本很高。vLLM 尽量只计算：

- sampled token 的 logprob；
- top-k token 的 logprobs；
- 用户指定 `logprob_token_ids` 的 logprobs。

关键锚点：

- kernel 内计算 max/logsumexp：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:81`
- sampled token logprob：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:113`
- top-k token ids：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:117`
- 指定 token ids：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:120`
- sampled token rank：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:156`
- 返回 `LogprobsTensors`：`code/vllm/vllm/v1/worker/gpu/sample/logprob.py:165`

## 8. Scheduler 中的 logprobs 提取

`Scheduler.update_from_output()`：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`

提取 sample logprobs 的位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1669`

只有满足以下条件才提取：

- `sampling_params is not None`；
- `sampling_params.num_logprobs is not None`；
- `model_runner_output.logprobs` 存在。

然后调用：

```python
new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

锚点：`code/vllm/vllm/v1/core/sched/scheduler.py:1675`

## 9. Frontend LogprobsProcessor

文件：`code/vllm/vllm/v1/engine/logprobs.py`

类：`LogprobsProcessor`，`code/vllm/vllm/v1/engine/logprobs.py:29`

保存：

- tokenizer；
- sample logprobs；
- prompt logprobs；
- cumulative logprob；
- num_logprobs；
- num_prompt_logprobs。

更新入口：`code/vllm/vllm/v1/engine/logprobs.py:348`

### 9.1 sample logprobs

`_update_sample_logprobs()`：`code/vllm/vllm/v1/engine/logprobs.py:69`

逻辑：

- sampler 把 sampled token logprob 放在第一个；
- 累加 `cumulative_logprob`；
- append 每个位置的 logprobs；
- 对 token 做非增量 detokenize；
- 处理 UTF-8 replacement char 修正。

### 9.2 prompt logprobs

`_update_prompt_logprobs()`：`code/vllm/vllm/v1/engine/logprobs.py:121`

把 prompt logprobs tensor 转成 Python list，每个 prompt position 对应一组 logprobs。

### 9.3 DELTA 模式

`pop_prompt_logprobs()`：`code/vllm/vllm/v1/engine/logprobs.py:189`

流式 DELTA 模式下，prompt logprobs 只应该返回一次，因此会被 pop 掉。

## 10. StructuredOutputsParams 到 StructuredOutputRequest

参数定义：`code/vllm/vllm/sampling_params.py:72`

request 结构：`code/vllm/vllm/v1/structured_output/request.py:21`

`StructuredOutputRequest` 保存：

- `params`；
- `_grammar`，可为 Future，支持异步 grammar 编译；
- `reasoning_ended`；
- `reasoning_parser_kwargs`；
- request-local `reasoner`。

构造入口：`code/vllm/vllm/v1/structured_output/request.py:31`

如果没有 structured outputs 或没有 constraint，则返回 None。

`get_structured_output_key()`：`code/vllm/vllm/v1/structured_output/request.py:77`

把参数转为 grammar cache key，支持：

- JSON；
- JSON_OBJECT；
- REGEX；
- CHOICE；
- GRAMMAR；
- STRUCTURAL_TAG。

## 11. grammar 初始化与编译

文件：`code/vllm/vllm/v1/structured_output/__init__.py`

`grammar_init()`：`code/vllm/vllm/v1/structured_output/__init__.py:115`

支持 backend：

- xgrammar；
- guidance；
- outlines；
- lm-format-enforcer。

可异步编译 grammar。

锚点：

- 初始化 backend：`code/vllm/vllm/v1/structured_output/__init__.py:125`
- 异步编译：`code/vllm/vllm/v1/structured_output/__init__.py:167`

## 12. grammar bitmask 生成

### 12.1 Scheduler 入口

`Scheduler.get_grammar_bitmask()`：`code/vllm/vllm/v1/core/sched/scheduler.py:1439`

逻辑：

1. 如果本 step 没有 structured output request，返回 None；
2. 收集使用 structured output 且不是 prefill chunk 的请求；
3. 调用 `structured_output_manager.grammar_bitmask()`；
4. 返回 `GrammarOutput`。

### 12.2 GrammarOutput

定义：`code/vllm/vllm/v1/core/sched/output.py:262`

字段：

- `structured_output_request_ids`；
- `grammar_bitmask`。

### 12.3 StructuredOutputManager.grammar_bitmask()

入口：`code/vllm/vllm/v1/structured_output/__init__.py:204`

职责：为 structured output batch 生成 token bitmask。

spec decode 下更复杂：

- 每个请求需要多个 mask；
- 包括 spec token positions 和 bonus/non-spec token position；
- 对每个 speculative token，会先 fill bitmask，再临时 `accept_tokens()` 推进 grammar，最后 rollback。

锚点：

- bitmask 主入口：`code/vllm/vllm/v1/structured_output/__init__.py:204`
- speculative token 处理：`code/vllm/vllm/v1/structured_output/__init__.py:275`
- rollback：`code/vllm/vllm/v1/structured_output/__init__.py:296`

## 13. GPU 应用 grammar bitmask

文件：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py`

入口：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:23`

调用点：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1046`

执行：

1. bitmask copy 到 GPU；
2. request id 映射到 logits row；
3. kernel 把非法 token logits 置为 `-inf`。

这一步发生在 sampler 前，因此 structured output 会直接改变采样分布。

## 14. grammar FSM 前进

Scheduler 在 `update_from_output()` 中推进结构化输出 FSM。

相关位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1598`

逻辑：

- 如果请求有新 token；
- 且 `structured_output_manager.should_advance(request)` 为真；
- 调用 grammar `accept_tokens()`；
- 如果 grammar 拒绝 token，请求置为 `FINISHED_ERROR`。

## 15. Reasoning 与 structured output gate

Reasoning parser 抽象：`code/vllm/vllm/reasoning/abs_reasoning_parsers.py:26`

关键接口：

- `is_reasoning_end()`：`code/vllm/vllm/reasoning/abs_reasoning_parsers.py:73`
- `is_reasoning_end_streaming()`：`code/vllm/vllm/reasoning/abs_reasoning_parsers.py:90`
- `extract_reasoning()`：`code/vllm/vllm/reasoning/abs_reasoning_parsers.py:146`
- `extract_reasoning_streaming()`：`code/vllm/vllm/reasoning/abs_reasoning_parsers.py:167`

Chat serving 会把 reasoning 状态传给 engine：

- `reasoning_ended` 计算：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:346`
- 传入 generate：`code/vllm/vllm/entrypoints/openai/chat_completion/serving.py:358`

StructuredOutputManager 中：

- request-local reasoner：`code/vllm/vllm/v1/structured_output/__init__.py:100`
- `should_fill_bitmask()`：`code/vllm/vllm/v1/structured_output/__init__.py:305`
- `should_advance()`：`code/vllm/vllm/v1/structured_output/__init__.py:325`

规则：

- 如果 `enable_in_reasoning=True`，reasoning 阶段也应用 grammar；
- 否则要等 reasoning 结束后再填 bitmask；
- streaming 中可用 `is_reasoning_end_streaming()` 判断本 step 是否刚结束 reasoning；
- 刚结束 reasoning 时通常延迟到下一 step 再推进 grammar，避免把 reasoning boundary token 误计入结构化内容。

## 16. Async scheduling 与 structured output

异步调度文件：`code/vllm/vllm/v1/core/sched/async_scheduler.py`

`pending_structured_output_tokens` 相关：`code/vllm/vllm/v1/core/sched/async_scheduler.py:31`

原因：grammar bitmask 可能依赖上一轮真实输出 token。如果 async scheduling 中还有 output placeholders 没处理完，就不能提前算下一步 grammar bitmask。

EngineCore 对应处理：

- `code/vllm/vllm/v1/engine/core.py:559`
- `code/vllm/vllm/v1/engine/core.py:612`

## 17. 本篇小结

- 模型层 `LogitsProcessor` 负责 hidden states 到 logits。
- sampling 阶段的 logit bias、allowed ids、penalties、bad words、temperature、top-p/top-k 才是真正采样策略。
- logprobs 计算尽量避免完整 vocab logprobs，只计算需要返回的部分。
- structured output 通过 grammar bitmask 在 sampler 前屏蔽非法 token。
- reasoning parser 可以决定 structured output 约束是否等 reasoning 结束后再启用。
