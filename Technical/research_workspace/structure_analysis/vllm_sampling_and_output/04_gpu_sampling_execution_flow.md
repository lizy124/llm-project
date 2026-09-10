# 04 GPU 采样执行链路

本篇梳理 V1 worker/GPU 侧从模型 forward 后到 sampled token ids/logprobs 的完整链路，重点覆盖新 GPU worker 路径。

## 1. 总体链路

新 GPU worker 主文件：`code/vllm/vllm/v1/worker/gpu/model_runner.py`

核心链路：

```text
GPUModelRunner.execute_model()
  ↓
prepare model inputs / attention metadata / batch state
  ↓
model forward
  ↓
hidden_states
  ↓
GPUModelRunner.sample_tokens()
  ↓
GPUModelRunner.sample()
  ↓
sample_hidden_states = hidden_states[input_batch.logits_indices]
  ↓
model.compute_logits(sample_hidden_states)
  ↓
structured output grammar mask
  ↓
Sampler 或 RejectionSampler
  ↓
SamplerOutput(sampled_token_ids, logprobs_tensors)
  ↓
AsyncOutput D2H
  ↓
ModelRunnerOutput
```

关键锚点：

- `GPUModelRunner.execute_model()`：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1101`
- model forward 分支：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1256`
- hidden states 保存：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1295`
- `sample_tokens()`：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1327`
- `sample()`：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1038`

## 2. execute_model() 的角色

`execute_model()` 不只执行模型，它还负责把 scheduler 输出转换成 GPU 可执行的 batch 状态。

主要职责：

1. 接收 `SchedulerOutput`；
2. 更新 input batch；
3. 准备 input ids、positions、attention metadata；
4. 执行模型 forward；
5. 在最后一个 pipeline rank 保存 hidden states；
6. 由后续 `sample_tokens()` 完成采样。

重要点：采样不是直接在 forward 中完成，而是 forward 后独立调用。

## 3. hidden states 到 logits

调用位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1044`

核心逻辑：

```python
sample_hidden_states = hidden_states[input_batch.logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

含义：

- forward 输出的 hidden states 可能包含 prompt/prefill 多个位置；
- 不是所有位置都需要 logits；
- `input_batch.logits_indices` 指明哪些位置需要参与采样或 logprobs 计算；
- `compute_logits()` 将 hidden states 投影到 vocab logits。

## 4. 模型层 LogitsProcessor

文件：`code/vllm/vllm/model_executor/layers/logits_processor.py`

关键类：`LogitsProcessor`，`code/vllm/vllm/model_executor/layers/logits_processor.py:18`

入口：`forward()`，`code/vllm/vllm/model_executor/layers/logits_processor.py:54`

底层 `_get_logits()`：`code/vllm/vllm/model_executor/layers/logits_processor.py:89`

职责：

1. 如果 `logits_as_input=True`，输入已经是 logits，直接使用；
2. 否则调用 `lm_head.quant_method.apply()` 将 hidden states 投影到 vocab；
3. tensor parallel 下 gather/all-gather logits；
4. 裁掉 vocab padding，只保留原始 vocab size；
5. 应用 soft cap；
6. 应用 scale。

锚点：

- 直接 logits 输入：`code/vllm/vllm/model_executor/layers/logits_processor.py:60`
- TP gather：`code/vllm/vllm/model_executor/layers/logits_processor.py:75`
- lm head 投影：`code/vllm/vllm/model_executor/layers/logits_processor.py:96`
- 去 padding：`code/vllm/vllm/model_executor/layers/logits_processor.py:101`
- soft cap/scale：`code/vllm/vllm/model_executor/layers/logits_processor.py:65`

注意：这个 `LogitsProcessor` 主要负责 hidden states 到 logits，不等同于 sampling 阶段的 logit bias、top-p、penalty 等逻辑。

## 5. InputBatch 中与采样相关的字段

文件：`code/vllm/vllm/v1/worker/gpu/input_batch.py`

类定义：`code/vllm/vllm/v1/worker/gpu/input_batch.py:35`

关键字段：

| 字段 | 作用 |
|---|---|
| `idx_mapping` | batch index 到 request state index 的映射 |
| `idx_mapping_np` | CPU/numpy 形式映射 |
| `expanded_idx_mapping` | spec decode 下展开后的 request 映射 |
| `expanded_local_pos` | 每个 logits row 在 request 内的位置 |
| `input_ids` | 当前 batch 输入 token |
| `positions` | position ids |
| `logits_indices` | 从 hidden states 中选哪些位置算 logits |
| `cu_num_logits` | 每个 request 对应 logits row 的前缀和 |

锚点：

- `idx_mapping`：`code/vllm/vllm/v1/worker/gpu/input_batch.py:42`
- `expanded_idx_mapping`：`code/vllm/vllm/v1/worker/gpu/input_batch.py:46`
- `expanded_local_pos`：`code/vllm/vllm/v1/worker/gpu/input_batch.py:48`
- `input_ids`：`code/vllm/vllm/v1/worker/gpu/input_batch.py:82`
- `logits_indices`：`code/vllm/vllm/v1/worker/gpu/input_batch.py:87`
- `cu_num_logits`：`code/vllm/vllm/v1/worker/gpu/input_batch.py:89`

## 6. 普通解码与 spec decode 的 logits row 差异

### 6.1 无 draft tokens

每个 request 通常只需要一个 logits row，用于采样下一个 token。

准备位置：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:864`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:870`

### 6.2 有 draft tokens

spec decode 下，一个 request 可能一次需要多个 logits row：

- draft token 的每个位置；
- bonus token 位置。

相关处理：

- 构造 `cu_num_logits`：`code/vllm/vllm/v1/worker/gpu/model_runner.py:879`
- 构造 `expanded_idx_mapping`：`code/vllm/vllm/v1/worker/gpu/model_runner.py:887`
- 构造 `expanded_local_pos`：`code/vllm/vllm/v1/worker/gpu/model_runner.py:895`
- 组合 sampled token 与 draft tokens：`code/vllm/vllm/v1/worker/gpu/model_runner.py:951`

## 7. structured output grammar mask 的位置

调用位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1046`

顺序是：

```text
compute_logits()
  ↓
apply_grammar_bitmask()
  ↓
Sampler/RejectionSampler
```

这意味着 grammar mask 比 temperature/top-p/top-k 等 sampling 参数更早生效。

GPU 实现文件：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py`

关键类：`StructuredOutputsWorker`，`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:12`

入口：`apply_grammar_bitmask()`，`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:23`

处理流程：

1. 异步把 grammar bitmask 拷到 GPU；
2. 按 `input_batch.req_ids` 和 `input_batch.cu_num_logits_np` 映射到 logits row；
3. Triton kernel 把不允许 token 的 logits 写成 `-inf`。

锚点：

- 拷贝 bitmask：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:33`
- request id 映射：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:39`
- logits row 映射：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:46`
- Triton kernel：`code/vllm/vllm/v1/worker/gpu/structured_outputs.py:85`

## 8. 普通 GPU Sampler

文件：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py`

类：`Sampler`，`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:30`

入口：`__call__()`，`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:72`

### 8.1 __call__() 做什么

1. 读取 batch 映射、positions、input ids；
2. 判断是否需要 logprobs；
3. 调用 `sample()` 产生 token ids；
4. 如需要，计算 top-k / sampled / specific token logprobs；
5. 返回 GPU 侧 `SamplerOutput`。

锚点：

- 读取映射：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:77`
- 判断 logprobs：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:88`
- 调用 sample：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:94`
- 计算 logprobs：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:104`
- 返回输出：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:134`

普通采样下 `sampled_token_ids` shape 是 `[num_requests, 1]`。

### 8.2 sampling 参数应用顺序

核心函数：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:146`

顺序：

1. logits 转 fp32；
2. logit bias / allowed token ids / min tokens / stop ids；
3. repetition/frequency/presence penalties；
4. bad words masking；
5. temperature；
6. min-p；
7. top-k/top-p；
8. 采样。

锚点：

- fp32：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:156`
- logit bias/allowed/min tokens：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:159`
- penalties：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:164`
- bad words：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:173`
- temperature：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:182`
- min-p：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:187`
- top-k/top-p：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:193`

### 8.3 最终采样

入口：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:198`

流程：

1. 应用 top-k/top-p 之前的 sampling params；
2. 读取每个 request 的 top-k/top-p；
3. 如果条件允许，走 FlashInfer sampling；
4. 否则应用 top-k/top-p，再走 Gumbel sample。

锚点：

- sampling params：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:208`
- top-k/top-p 参数：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:217`
- FlashInfer：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:220`
- fallback：`code/vllm/vllm/v1/worker/gpu/sample/sampler.py:234`

## 9. Sampling state 组件

### 9.1 SamplingStates

文件：`code/vllm/vllm/v1/worker/gpu/sample/states.py`

类定义：`code/vllm/vllm/v1/worker/gpu/sample/states.py:17`

保存：

- temperature；
- top_k；
- top_p；
- min_p；
- seeds；
- num_logprobs。

### 9.2 LogitBiasState

文件：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py`

类定义：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:15`

负责：

- allowed token ids；
- logit bias；
- min tokens / stop token ids。

allowed token ids 的实现方式是先把整行 logits 置为 `-inf`，再恢复 allowed tokens。

相关锚点：

- allowed ids：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:19`
- logit bias：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:28`
- min tokens：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:40`
- mask 实现：`code/vllm/vllm/v1/worker/gpu/sample/logit_bias.py:177`

### 9.3 PenaltiesState

文件：`code/vllm/vllm/v1/worker/gpu/sample/penalties.py`

类定义：`code/vllm/vllm/v1/worker/gpu/sample/penalties.py:14`

负责：

- repetition penalty；
- frequency penalty；
- presence penalty。

应用入口：`code/vllm/vllm/v1/worker/gpu/sample/penalties.py:81`

### 9.4 BadWordsState

文件：`code/vllm/vllm/v1/worker/gpu/sample/bad_words.py`

类定义：`code/vllm/vllm/v1/worker/gpu/sample/bad_words.py:15`

应用入口：`code/vllm/vllm/v1/worker/gpu/sample/bad_words.py:72`

## 10. Spec decode / RejectionSampler

初始化位置：

- speculator 初始化：`code/vllm/vllm/v1/worker/gpu/model_runner.py:188`
- sampler/rejection sampler 初始化：`code/vllm/vllm/v1/worker/gpu/model_runner.py:324`

分支位置：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1056`

- 无 draft tokens 或无 rejection sampler：普通 sampler；
- 有 draft tokens：rejection sampler。

文件：`code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`

类定义：`code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:43`

流程：

1. draft sampled token 来自 `input_batch.input_ids[input_batch.logits_indices]`；
2. 对 target logits 应用普通 sampling params；
3. 调 Triton rejection sampling；
4. 计算 spec decode 场景 logprobs；
5. 返回多 token `SamplerOutput`。

锚点：

- draft tokens：`code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:108`
- sampling params：`code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:110`
- rejection sampling：`code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:118`
- logprobs：`code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:133`
- 输出：`code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:150`

## 11. SamplerOutput 与异步 D2H

GPU 侧输出结构：`code/vllm/vllm/v1/worker/gpu/sample/output.py:10`

字段：

- `sampled_token_ids`；
- `logprobs_tensors`；
- `num_nans`；
- `num_sampled`；
- `num_rejected`。

异步 GPU -> CPU copy 文件：`code/vllm/vllm/v1/worker/gpu/async_utils.py`

关键类：`code/vllm/vllm/v1/worker/gpu/async_utils.py:12`

流程：

1. copy stream 上异步拷贝 sampled token ids；
2. 异步拷贝 logprobs tensors；
3. 异步拷贝每个 request 实际 sampled token 数；
4. `get_output()` 等待 copy event；
5. 按 `num_sampled_tokens` 截断 padded token ids；
6. 写入 `ModelRunnerOutput.sampled_token_ids` 和 `ModelRunnerOutput.logprobs`。

锚点：

- token ids copy：`code/vllm/vllm/v1/worker/gpu/async_utils.py:32`
- logprobs copy：`code/vllm/vllm/v1/worker/gpu/async_utils.py:34`
- sampled 数量 copy：`code/vllm/vllm/v1/worker/gpu/async_utils.py:41`
- get output：`code/vllm/vllm/v1/worker/gpu/async_utils.py:48`
- 截断：`code/vllm/vllm/v1/worker/gpu/async_utils.py:55`
- 写入 ModelRunnerOutput：`code/vllm/vllm/v1/worker/gpu/async_utils.py:59`

## 12. 旧路径对照

旧/顶层 runner：`code/vllm/vllm/v1/worker/gpu_model_runner.py`

关键锚点：

- `_sample()`：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3570`
- 普通 sampler：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3580`
- rejection sampler：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3592`
- forward 后 compute logits：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354`
- structured output mask：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4452`

旧 sampler：`code/vllm/vllm/v1/sample/sampler.py`

- `Sampler.forward()`：`code/vllm/vllm/v1/sample/sampler.py:72`
- `sample()`：`code/vllm/vllm/v1/sample/sampler.py:243`
- `SamplingMetadata`：`code/vllm/vllm/v1/sample/metadata.py:14`

## 13. 本篇小结

GPU 采样可以概括为：

```text
hidden states
  -> 选择 logits positions
  -> model.compute_logits()
  -> grammar mask
  -> logit bias / allowed ids / min tokens / penalties / bad words
  -> temperature / min-p / top-k / top-p
  -> random or greedy sampling / rejection sampling
  -> sampled_token_ids + logprobs_tensors
  -> ModelRunnerOutput
```

重点：真正改变 token 分布的逻辑主要在 GPU sampler 与 structured output mask 中，而不是在最终 OutputProcessor 中。
