# 10. Spec decode 有哪些限制和边界场景？

源码位置：

- `code/vllm/vllm/config/speculative.py`
- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/config/compilation.py`
- `code/vllm/vllm/sampling_params.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu_input_batch.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/__init__.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/utils.py`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/sample/logits_processor/__init__.py`
- `code/vllm/vllm/v1/structured_output/utils.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/worker/cp_utils.py`

本问题关注：spec decode 与配置校验、采样参数、logits processor、structured output、KV cache、chunked prefill、prefix cache、async scheduling、Pipeline Parallel、CUDA graph、V2 ModelRunner 等能力之间的限制和边界场景。

---

## 0. 梳理规划

参考 `executor_worker_model_runner` 目录文档风格，本篇按“先讲硬限制，再讲运行时边界，最后讲容易混淆点”的方式梳理。

要回答的问题分成 15 组：

```text
1. spec decode 最核心的限制来自哪里？
2. SpeculativeConfig 有哪些硬校验？
3. 哪些 speculative method 在 V1 / V2 / async scheduling 下可用？
4. sampling 参数中哪些能力支持，哪些能力受限？
5. top-k / top-p / min_p / logit_bias 在 rejection sampler 中如何处理？
6. allowed_token_ids / bad_words / penalties 如何按 draft tokens 展开？
7. logprobs / prompt_logprobs 的边界是什么？
8. structured output 会如何裁剪 draft tokens？
9. KV cache / prefix cache / external KV connector 有哪些一致性边界？
10. chunked prefill、max_model_len、draft model max_model_len 如何影响 speculation？
11. accepted / rejected / recovered / bonus token 输出有哪些边界？
12. async scheduling、PP、routed experts、CUDA graph 有哪些特殊处理？
13. 哪些模型 / backend / 并行配置会禁用或限制 spec decode？
14. 什么时候 spec decode 不会提速，甚至会变慢？
15. 遇到异常时应该如何定位？
```

本篇不重复展开 RejectionSampler 的概率算法，也不重复展开 Scheduler 回收细节；重点是“哪些场景不能用、不能按普通 decode 理解，或者需要额外对齐状态”。

阅读顺序建议：

```text
01_spec_decode_role.md
  → 03_scheduler_spec_decode_flow.md
  → 06_rejection_sampler_flow.md
  → 07_kv_cache_and_num_computed_tokens.md
  → 08_structured_output_interaction.md
  → 09_output_recovery_and_scheduler_update.md
  → 10_limitations_and_edge_cases.md
```

---

## 1. 一句话回答

Spec decode 的限制本质上来自一句话：

```text
它一次让 target model 验证多个 draft token，但只有 accepted / recovered / bonus tokens 才能成为真实输出。
```

所以任何会影响下面三类状态的能力，都必须和 spec decode 对齐：

```text
1. token 合法性：structured output、allowed_token_ids、bad_words、min_tokens、stop / EOS；
2. 采样分布：temperature、top-k、top-p、min_p、logit_bias、penalties、custom logits processors；
3. 进度状态：KV cache、num_computed_tokens、num_output_placeholders、prefix cache、async scheduling。
```

如果某个功能只适用于“每步只生成 1 个 token”的普通 decode，spec decode 就不能直接套用；它必须知道：

```text
本轮验证了哪些 draft；
哪些 draft 被接受；
哪些 draft 被拒绝；
最终输出了几个 token；
KV / grammar / logprobs / request 进度如何回滚或推进。
```

---

## 2. 总体限制地图

可以把限制分成五层：

```text
配置层：
  method / num_speculative_tokens / draft model / vocab / TP / max_model_len。

调度层：
  token budget / max_model_len / chunked prefill / prefix cache / KV connector。

模型执行层：
  attention metadata / CUDA graph / PP / CP / draft model fits / drafter backend。

采样层：
  rejection sampler / sampling constraints / logits processors / logprobs。

输出回收层：
  accepted/rejected 统计 / num_computed_tokens 回滚 / grammar accept / stop 裁剪。
```

主线可以记为：

```text
SpeculativeConfig 决定能不能启用；
Scheduler 决定本轮能验证几个 draft；
ModelRunner 决定能不能生成下一轮 draft；
RejectionSampler 决定哪些 token 输出；
Scheduler.update_from_output() 决定最终状态是否一致。
```

---

## 3. 配置层硬限制：num_speculative_tokens 必须有效

`SpeculativeConfig` 中最基础的限制是：

```python
num_speculative_tokens: int = Field(default=None, gt=0)
```

位置：`code/vllm/vllm/config/speculative.py:86`

后续 validator 还会检查：

```python
if self.num_speculative_tokens is None:
    raise ValueError(...)

if self.num_speculative_tokens <= 0:
    raise ValueError(...)
```

位置：`code/vllm/vllm/config/speculative.py:1165` 到 `code/vllm/vllm/config/speculative.py:1173`

含义：

```text
除非 draft model config 自己能推导 n_predict，否则必须显式提供 num_speculative_tokens。
它必须大于 0。
```

这不是性能参数这么简单，它会影响：

```text
- 每个请求最多带多少 spec_token_ids；
- RejectionSampler 输出 shape = max_spec_len + 1；
- CUDA graph capture size 是否需要按 num_speculative_tokens + 1 对齐；
- async placeholders / num_computed_tokens 的回滚上限；
- draft model 是否超出自己的 max_model_len。
```

---

## 4. draft model 和 target model 的限制

### 4.1 draft model 词表必须和 target 一致

如果使用 `method == "draft_model"`，vLLM 会校验 target / draft 词表大小一致：

```python
if target_vocab_size != draft_vocab_size:
    raise ValueError(...)
```

位置：`code/vllm/vllm/config/speculative.py:1215` 到 `code/vllm/vllm/config/speculative.py:1229`

原因：

```text
draft token id 会直接进入 target model 验证；
如果 tokenizer / vocab 不一致，同一个 token id 表示的语义可能不同；
更严重时会出现 out-of-bounds 或 logits / probs shape 不匹配。
```

所以 draft-model spec decode 不能随便拿一个“小模型”当 drafter。

最低要求是：

```text
1. tokenizer / vocab 对齐；
2. draft tokens 能被 target model 同语义解释；
3. draft probabilities 能和 target probabilities 做 ratio test。
```

### 4.2 draft tensor parallel size 只能是 1 或等于 target TP

校验逻辑：

```python
elif speculative_draft_tensor_parallel_size not in (
    1,
    target_parallel_config.tensor_parallel_size,
):
    raise ValueError(...)
```

位置：`code/vllm/vllm/config/speculative.py:1099` 到 `code/vllm/vllm/config/speculative.py:1107`

含义：

```text
Draft TP 不能任意设置。
它要么单卡 / 单 TP 运行，
要么和 target TP 对齐。
```

原因是 draft model 和 target model 虽然是两个执行链路，但它们共享同一个 serving batch 的请求状态、KV / attention metadata、token ids 和分布式拓扑约束。

### 4.3 draft max_model_len 会被 target max_model_len 截断

逻辑：

```python
result = min(draft_max_model_len, target_max_model_len)
```

位置：`code/vllm/vllm/config/speculative.py:1031` 到 `code/vllm/vllm/config/speculative.py:1060`

如果用户显式设置 `speculative_max_model_len`，还会检查不能超过 draft / target：

```python
if speculative_max_model_len > draft_max_model_len:
    raise ValueError(...)

if speculative_max_model_len > target_max_model_len:
    raise ValueError(...)
```

位置：`code/vllm/vllm/config/speculative.py:1047` 到 `code/vllm/vllm/config/speculative.py:1060`

这会影响运行时边界：

```text
当请求接近 draft model 或 target model 的上下文上限时，drafter 可能不能继续生成 draft。
ModelRunner 会 zero out draft tokens，避免 scheduler 继续消费 stale drafts。
```

相关运行时代码：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4565` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4644`

---

## 5. speculative method 的支持范围

### 5.1 配置中声明的 method 很多，但运行时不一定都支持

`SpeculativeMethod` 包括：

```text
ngram
medusa
mlp_speculator
draft_model
suffix
custom_class
eagle / eagle3 / mtp / dflash / 多种 MTP model type
ngram_gpu
```

位置：`code/vllm/vllm/config/speculative.py:64` 到 `code/vllm/vllm/config/speculative.py:74`

但 GPU V1 worker 的 `init_speculator()` 只处理：

```text
dflash
gemma4_mtp
mtp
eagle / eagle3 / mtp / dflash 这类 use_eagle() 路径
```

否则：

```python
raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
```

位置：`code/vllm/vllm/v1/worker/gpu/spec_decode/__init__.py:8` 到 `code/vllm/vllm/v1/worker/gpu/spec_decode/__init__.py:40`

这说明要区分两件事：

```text
SpeculativeConfig 能解析某个 method；
当前 worker / runner / backend 是否真的实现了这个 method。
```

### 5.2 V2 ModelRunner 的 spec decode 支持更窄

V2 model runner unsupported features 中对 spec decode 有额外限制：

```text
- ngram / ngram_gpu speculative decoding 暂不支持；
- method 不是 eagle / eagle3 / mtp / dflash 时不支持；
- dynamic speculative decoding 不支持；
- EAGLE parallel_drafting 不支持，DFlash 例外；
- EAGLE3 + pipeline parallelism 不支持。
```

位置：`code/vllm/vllm/config/vllm.py:2121` 到 `code/vllm/vllm/config/vllm.py:2143`

所以同样的 speculative config，在 V1 runner 和 V2 runner 下可用性可能不同。

---

## 6. async scheduling 的限制

### 6.1 显式开启 async scheduling 时，不兼容会直接报错

当 `scheduler_config.async_scheduling` 显式为 true：

```python
if self.speculative_config is not None:
    if method not in EagleModelTypes and method not in NgramGPUTypes and method != "draft_model":
        raise ValueError(...)
    if self.speculative_config.disable_padded_drafter_batch:
        raise ValueError(...)
```

位置：`code/vllm/vllm/config/vllm.py:1019` 到 `code/vllm/vllm/config/vllm.py:1034`

也就是说 async scheduling 当前只支持：

```text
EAGLE / MTP / DFlash 类；
NGram GPU；
Draft Model；
且不能和 disable_padded_drafter_batch=True 同时使用。
```

### 6.2 async scheduling 自动模式下会禁用不兼容组合

如果 async scheduling 是默认自动模式，遇到不支持的 spec method 会 warning 并关闭：

```python
logger.warning_once(
    "Async scheduling not supported with %s-based speculative decoding and will be disabled.",
    self.speculative_config.method,
)
self.scheduler_config.async_scheduling = False
```

位置：`code/vllm/vllm/config/vllm.py:1053` 到 `code/vllm/vllm/config/vllm.py:1065`

如果 `disable_padded_drafter_batch=True`，也会关闭 async scheduling：

位置：`code/vllm/vllm/config/vllm.py:1065` 到 `code/vllm/vllm/config/vllm.py:1072`

### 6.3 async spec decode 需要 placeholders 回滚

async scheduling 下，Scheduler 可能提前放置输出占位。

如果 draft 被拒绝，回收时必须同步回滚：

```python
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1661`

否则会出现：

```text
- 下一轮 request.num_tokens_with_spec 过大；
- token positions 偏移；
- KV / encoder cache 释放过早；
- structured output grammar 看到错误 token 边界。
```

---

## 7. custom logits processors 不支持

构建 logits processors 时，如果启用了 spec decode：

```python
if vllm_config.speculative_config:
    if custom_logitsprocs:
        raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
    logger.warning(
        "min_p and logit_bias parameters won't work with speculative decoding."
    )
    return LogitsProcessors(
        [MinTokensLogitsProcessor(vllm_config, device, is_pin_memory)]
    )
```

位置：`code/vllm/vllm/v1/sample/logits_processor/__init__.py:202` 到 `code/vllm/vllm/v1/sample/logits_processor/__init__.py:209`

错误信息定义：

```python
"Custom logits processors are not supported when speculative decoding is enabled."
```

位置：`code/vllm/vllm/v1/sample/logits_processor/__init__.py:44`

含义：

```text
Spec decode 下不能加载用户自定义 logits processor；
内置 processor 也只保留 MinTokensLogitsProcessor；
min_p 和 logit_bias 会提示不生效。
```

原因是 custom logits processor 可能改变每个 token 位置的分布。

普通 decode 只处理一个 next-token logits；spec decode 需要同时处理：

```text
- 每个 draft token 位置的 target logits；
- bonus token logits；
- accepted / rejected 后的 recovered distribution；
- logprobs 对齐。
```

如果 logits processor 不是 argmax-invariant，或者不能按 draft-token 级别展开，就会破坏 rejection sampling 的正确性。

---

## 8. top-k / top-p / min_p / logit_bias 的边界

### 8.1 bonus token 可以走普通 sampler 约束

`RejectionSampler` 注释明确说明：

```text
bonus token 是从 target probabilities 采样出来的；
它作为参数传入 rejection sampler；
这样 bonus token 可以使用 top_p / top_k 等采样策略。
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:47` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:54`

对应代码：

```python
bonus_sampler_output = self.sampler(
    logits=bonus_logits,
    sampling_metadata=replace(...),
    predict_bonus_token=True,
    logprobs_mode_override=...,
)
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:130` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:143`

所以：

```text
all accepted 后额外输出的 bonus token，可以更接近普通 sampling 行为。
```

### 8.2 draft 验证 / recovered token 不等同于普通 top-k / top-p sampling

RejectionSampler 的注释也说明：

```text
bonus token 可以用 top_p / top_k，
而 spec decode does not support these sampling strategies。
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:49` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:54`

更具体地说：

```text
top-k / top-p 对 bonus token 适用；
但 accepted / recovered 的主体逻辑是 rejection sampling，
不能简单理解成每个 draft 位置都完整执行了一次普通 sampler。
```

### 8.3 min_p / logit_bias 在 spec decode 下不会按普通方式生效

构建 logits processors 时会 warning：

```text
min_p and logit_bias parameters won't work with speculative decoding.
```

位置：`code/vllm/vllm/v1/sample/logits_processor/__init__.py:204` 到 `code/vllm/vllm/v1/sample/logits_processor/__init__.py:205`

因此如果请求依赖：

```text
min_p
logit_bias
custom logits processors
```

不应假设 spec decode 和普通 decode 的采样分布完全一致。

---

## 9. allowed_token_ids / bad_words / penalties 的边界

### 9.1 allowed_token_ids 会按 draft token 位置展开

`RejectionSampler.apply_logits_processors()` 中，如果存在 allowed token mask、penalties 或 thinking budget，会构造 `repeat_indices`：

```python
repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:307` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:319`

然后应用 allowed token ids：

```python
if sampling_metadata.allowed_token_ids_mask is not None:
    token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
    logits.masked_fill_(token_mask, float("-inf"))
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:324` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:328`

含义：

```text
allowed_token_ids 不是只作用于 batch 中每个请求的一行 logits；
它必须扩展到该请求的每个 draft token 位置。
```

### 9.2 bad_words 需要合并 output_token_ids 和 spec_token_ids

在 spec decode 下，如果 bad words 或 penalties 需要历史输出，不能只看已经正式输出的 tokens。

代码会先合并 spec tokens：

```python
output_token_ids = self._combine_outputs_with_spec_tokens(
    output_token_ids,
    sampling_metadata.spec_token_ids,
)
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:298` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:303`

然后应用 bad words：

```python
apply_bad_words_with_drafts(
    logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
)
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:330` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:333`

原因：

```text
第 i 个 draft token 的合法性，依赖它之前的 accepted draft prefix；
不能只用 request.output_token_ids，否则 bad_words / penalties 会少看本轮已假设接受的 draft 前缀。
```

### 9.3 penalties 也按 draft token 展开

penalties 处理同样使用 `repeat_indices`：

```python
prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:359` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:364`

再调用：

```python
apply_all_penalties(...)
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:366` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:373`

这说明 penalties 支持 spec decode，但代价是：

```text
每个 draft 位置都要拥有正确的“假设历史”；
否则 repetition / presence / frequency 的语义会错位。
```

---

## 10. logprobs 的边界

### 10.1 spec decode 下每个请求本轮输出 token 数可能不同

`LogprobsLists` 专门有：

```python
cu_num_generated_tokens: list[int] | None = None
```

注释：

```text
Used for slicing the logprobs in cases like speculative decoding where the number of generated tokens may be different for each request.
```

位置：`code/vllm/vllm/v1/outputs.py:27` 到 `code/vllm/vllm/v1/outputs.py:46`

Scheduler 回收时按实际 `len(new_token_ids)` 切片：

```python
new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1779`

这解决的是：

```text
请求 A 本轮可能输出 4 个 token；
请求 B 第一个 draft 就 rejected，只输出 1 个 token；
请求 C 没有 draft，只输出普通 1 个 token。
```

logprobs 不能再按固定 `[batch_size, 1]` 理解。

### 10.2 rejected token 的 logprobs 会先算，后过滤

`_get_logprobs_tensors()` 注释说明：

```text
为了避免 CPU-GPU sync，会先为所有 draft tokens 计算 indices，包含 rejected ones；
rejected tokens 会在 parse_output 中过滤。
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:218` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:220`

过滤发生在：

```python
valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
    output_token_ids_np < vocab_size
)
filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:267` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:276`

含义：

```text
内部可以临时为 rejected token 准备 logprobs；
但输出给 Scheduler / OutputProcessor 的 logprobs 只对齐最终有效 tokens。
```

### 10.3 prompt_logprobs 和 prefix cache 有额外关系

`SamplingParams` 中：

```python
if self.skip_reading_prefix_cache is None:
    self.skip_reading_prefix_cache = self.prompt_logprobs is not None
```

位置：`code/vllm/vllm/sampling_params.py:500` 到 `code/vllm/vllm/sampling_params.py:504`

这不是 spec decode 专属限制，但和 spec decode 同时出现时容易混淆：

```text
prompt_logprobs 属于 prompt / prefill 位置；
spec decode 主要影响 generated token 的多 token logprobs；
如果 prompt_logprobs 要求读取 prompt 每个位置的概率，prefix cache 读取可能被跳过以保证 logprobs 完整性。
```

---

## 11. structured output 的边界

### 11.1 grammar bitmask 必须按 spec logits rows 对齐

`apply_grammar_bitmask()` 会根据每个请求的 scheduled spec tokens 调整 logits row：

```python
logit_index = batch_index + cumulative_offset
cumulative_offset += len(spec_tokens.get(req_id, ()))
```

位置：`code/vllm/vllm/v1/structured_output/utils.py:112` 到 `code/vllm/vllm/v1/structured_output/utils.py:120`

对 structured output 请求，会为 `1 + num_spec_tokens` 行填 bitmask：

```python
for i in range(1 + num_spec_tokens):
    bitmask_index = logit_idx + i
    sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
```

位置：`code/vllm/vllm/v1/structured_output/utils.py:132` 到 `code/vllm/vllm/v1/structured_output/utils.py:140`

所以 structured output 下，bitmask 不是简单按 batch index 对齐。

它要对齐：

```text
请求的 bonus / normal sample row；
请求的每个 draft token target logits row；
不同请求由于 draft token 数不同产生的 cumulative offset。
```

### 11.2 draft tokens 会先被 grammar validate_tokens 裁剪

同步路径更新 draft tokens 时：

```python
if self.structured_output_manager.should_advance(request):
    metadata = request.structured_output_request
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
request.spec_token_ids = spec_token_ids
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2023` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2025`

batch queue / deferred sampling 路径还会把无效 token 计入 `num_invalid_spec_tokens`：

```python
spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
if num_invalid_tokens:
    spec_token_ids.extend([-1] * num_invalid_tokens)
    num_invalid_spec_tokens[req_id] = num_invalid_tokens
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2030` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2064`

含义：

```text
structured output 可以提前让某些 draft tokens 失效；
这些 token 不应该被当成 target rejection；
统计时要通过 num_invalid_spec_tokens 修正 acceptance rate。
```

### 11.3 最终 accept_tokens 只接受 confirmed grammar content

输出回收阶段：

```python
if new_token_ids and self.structured_output_manager.should_advance(request):
    advance_token_ids = self.structured_output_manager.trim_reasoning_for_advance(
        request, new_token_ids
    )
    if advance_token_ids and not grammar.accept_tokens(req_id, advance_token_ids):
        request.status = RequestStatus.FINISHED_ERROR
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1698` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1717`

注意这里传的是：

```text
new_token_ids = accepted draft tokens + recovered token 或 bonus token；
不包含 rejected draft tokens；
如果包含 reasoning 结束 marker 前后的混合内容，会先 trim 掉 reasoning 前缀，只把 grammar content 交给 grammar。
```

如果 grammar 在这里拒绝，说明前面的 draft validation / grammar bitmask / logits row 对齐出现不一致，vLLM 会把请求置为 `FINISHED_ERROR`。

---

## 12. chunked prefill 的边界

### 12.1 Scheduler 没有硬编码 prefill / decode 两套逻辑

Scheduler 注释说明：

```text
There's no "decoding phase" nor "prefill phase" in the scheduler.
Each request just has num_computed_tokens and num_tokens_with_spec.
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:433` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:445`

这使得：

```text
chunked prefill、prefix caching、speculative decoding 都能统一成
num_computed_tokens 追赶 num_tokens_with_spec。
```

### 12.2 prefill chunk 会忽略 draft tokens

当 drafter 返回 draft tokens 时，如果 request 还在 prefill chunk：

```python
if request.is_prefill_chunk:
    # Ignore draft tokens for prefill chunks.
    if request.spec_token_ids:
        request.spec_token_ids = []
    continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2015` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2019`

含义：

```text
spec decode 是 decode 加速；
prefill chunk 还没完成时，不应该把 draft tokens 当成可验证输出。
```

### 12.3 deferred output 中 draft tokens 可能被裁到 scheduled 长度

batch queue / async sampling 场景中：

```python
orig_num_spec_tokens = len(placeholder_spec_tokens)
del spec_token_ids[orig_num_spec_tokens:]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:2046` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2049`

注释说明这对 chunked prefill 等场景是必要的。

原因：

```text
Drafter 可能生成了更多 draft；
但 SchedulerOutput 本轮只预留 / 调度了其中一部分；
输出回收必须以本轮 scheduled_spec_decode_tokens 为准。
```

---

## 13. max_model_len 边界

Scheduler 在调度 running request 时会限制：

```python
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len
    - request.num_computed_tokens
    - self.num_sampled_tokens_per_step,
)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:519` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:527`

注释写明：

```text
Make sure the input position does not exceed the max model len.
This is necessary when using spec decoding.
```

含义：

```text
spec decode 不只要容纳 draft tokens；
还要保留普通 sampled / bonus token 的空间。
```

如果接近 max_model_len，就可能出现：

```text
- 本轮可调度 draft 数减少；
- drafter 不再生成 draft；
- draft tokens 被 zero out；
- request 退化成普通单 token decode 或直接触发长度停止。
```

---

## 14. KV cache / prefix cache 的边界

### 14.1 rejected draft 不能推进真实 num_computed_tokens

回收逻辑：

```python
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
request.num_computed_tokens -= num_rejected
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1649` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1659`

核心原则：

```text
target forward 可以计算 rejected draft 位置；
但 rejected draft 不是正式上下文；
下一轮调度不能从 rejected token 后继续。
```

所以 spec decode 的 KV cache 边界是：

```text
计算过 ≠ 被接受；
写过 KV ≠ 能推进 request 进度；
Scheduler 必须按 rejected 数回滚 num_computed_tokens。
```

### 14.2 external KV connector 可能让请求跳过本轮回收

如果 KV connector 报告 invalid blocks：

```python
if kv_connector_output and kv_connector_output.invalid_block_ids:
    failed_kv_load_req_ids = self._handle_invalid_blocks(...)
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1578` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1584`

随后：

```python
if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
    continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1619` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1621`

含义：

```text
如果外部 KV load 失败，请求可能需要 recompute 或 error；
这类请求不会照常提交本轮 generated_token_ids。
```

### 14.3 spec decode 会延后 KV connector finalize

ModelRunner forward 外层会设置：

```python
defer_kv_connector_finalize = self.speculative_config is not None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4357`

sample_tokens 之后再 finalize：

```python
if spec_config is not None:
    self.finalize_kv_connector()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4683` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4687`

原因：

```text
target model forward 之后，drafter 也可能运行并保存 KV；
如果过早 finalize KV connector，会漏掉 drafter 相关 KV 状态。
```

---

## 15. prefix cache 的边界

Spec decode 和 prefix cache 在 Scheduler 里是同一个进度模型：

```text
num_computed_tokens 表示已经可复用 / 已计算的 token 数；
num_tokens_with_spec 包含 prompt + output + spec tokens；
Scheduler 让 num_computed_tokens 追赶 num_tokens_with_spec。
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:433` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:445`

但边界在于：

```text
prefix cache 命中的是已确认上下文；
spec tokens 是临时候选；
rejected spec tokens 不能进入 prefix cache 的真实进度语义。
```

因此以下场景要特别注意：

```text
1. prompt_logprobs 可能设置 skip_reading_prefix_cache；
2. rejected draft 会回滚 num_computed_tokens；
3. external KV invalid blocks 会触发 recompute / skip output；
4. encoder cache 释放要避开 pending placeholders。
```

encoder cache 释放中也能看到 placeholder 边界：

```python
start_pos + num_tokens + spec_lookahead <= request.num_computed_tokens - request.num_output_placeholders
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1997` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:2003`

这说明：

```text
只要 async placeholders 还可能因为 draft rejection 回滚，
就不能过早释放相关 encoder input。
```

---

## 16. 输出边界：accepted 可以为 0

如果第一个 draft 就被拒绝：

```text
scheduled_spec_token_ids = [A, B, C]
RejectionSampler output = [X, -1, -1, -1]
generated_token_ids = [X]
```

Scheduler 计算：

```text
num_accepted = len([X]) - 1 = 0
num_rejected = 3
```

对应代码：

```python
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1649` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1650`

这说明：

```text
spec decode 不保证每轮都接受 draft；
最差情况下只输出 recovered token；
甚至如果请求被 discard / abort，本轮可能不提交 token。
```

---

## 17. 输出边界：全部接受时会有 bonus token

RejectionSampler 的术语定义：

```text
If all proposed tokens are accepted, the bonus token is added to the end of the sequence.
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:47` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:50`

因此：

```text
scheduled draft = [A, B, C]
all accepted
bonus = D
最终 output = [A, B, C, D]
```

Scheduler 没有显式 bonus 标签。

它通过长度推断：

```text
num_accepted = len(output) - num_sampled_tokens_per_step
```

所以当 `num_sampled_tokens_per_step = 1`：

```text
len([A,B,C,D]) - 1 = 3 accepted draft tokens。
```

---

## 18. 输出边界：stop / EOS 可能裁掉多 token 输出后缀

spec decode 一轮可能输出多个 token。

Scheduler 提交时逐个 append 并检查 stop：

```python
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)
    stopped = check_stop(request, self.max_model_len)
    if stopped:
        del new_token_ids[num_new:]
        break
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1961` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1970`

边界：

```text
RejectionSampler 可能给出 [A, B, C, D]；
如果 B 触发 stop，Scheduler 最终只提交 [A, B]；
C/D 不会出现在 EngineCoreOutput.new_token_ids。
```

这也意味着：

```text
accepted/rejected 统计可能在裁剪前完成；
用户可见输出以 stop 裁剪后的 new_token_ids 为准。
```

---

## 19. stale draft tokens 边界

当 drafter 输入不再适合继续生成 draft：

```python
if not input_fits_in_drafter:
    self._draft_token_ids = torch.zeros(...).expand(...)
    self._draft_probs = None
    self._draft_prob_req_ids = None
    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4634` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4644`

注释说明：

```text
Zero out draft tokens so the scheduler doesn't schedule stale drafts from the previous step.
```

原因：

```text
如果上一轮 draft 没有清掉，Scheduler 可能在下一轮误消费 stale draft；
对 Nemotron-H 这类含 Mamba recurrent state 的模型，stale tokens 还可能污染状态和 logprobs。
```

所以 spec decode 的一个重要防线是：

```text
不能生成可靠 draft 时，宁可返回 zero / placeholder，不能保留旧 draft。
```

---

## 20. Pipeline Parallel 的边界

在普通非 PP 场景，ModelRunner 可以把 sampled tokens 缓存在 worker 本地，下一轮输入准备直接使用。

但 `_bookkeeping_sync()` 注释说明：

```text
when using PP, the scheduler sends the sampled tokens back,
because there's no direct communication between the first-stage worker and the last-stage worker.
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:3748` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:3750`

原因：

```text
PP first rank 负责后续 input token / embedding；
PP last rank 才产生 logits / sampled tokens；
两者之间不能假设有直接本地共享 sampled token 状态。
```

V2 还有更明确的不支持项：

```text
EAGLE3 with pipeline parallelism unsupported。
```

位置：`code/vllm/vllm/config/vllm.py:2143`

---

## 21. Context Parallel / interleave 的边界

`check_attention_cp_compatibility()` 中：

```python
if vllm_config.speculative_config is not None and interleave_size > 1:
    assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (...)
```

位置：`code/vllm/vllm/v1/worker/cp_utils.py:22` 到 `code/vllm/vllm/v1/worker/cp_utils.py:33`

含义：

```text
Spec decode + CP KV cache interleave size > 1 需要 attention backend 显式支持；
否则断言失败。
```

原因是：

```text
spec decode 会让 decode query len 变成 num_speculative_tokens + 1；
CP / DCP / interleave 会改变 KV / attention 分片布局；
attention backend 必须能同时理解这两套布局。
```

---

## 22. CUDA graph / compilation 的边界

当 FULL decode CUDA graph 且 uniform decode query length 大于 1 时：

```python
self.adjust_cudagraph_sizes_for_spec_decode(
    uniform_decode_query_len,
    tensor_parallel_size,
)
```

位置：`code/vllm/vllm/config/compilation.py:1437` 到 `code/vllm/vllm/config/compilation.py:1443`

其中 `uniform_decode_query_len` 对 spec decode 通常可以理解为：

```text
num_speculative_tokens + 1
```

如果同时启用 sequence parallelism，还要求 CUDA graph sizes 同时是两个数的倍数：

```python
if multiple_of % uniform_decode_query_len != 0 or multiple_of % tensor_parallel_size != 0:
    raise ValueError(...)
```

位置：`code/vllm/vllm/config/compilation.py:1469` 到 `code/vllm/vllm/config/compilation.py:1489`

报错提示也明确说：

```text
please adjust num_speculative_tokens or disable sequence parallelism
```

所以：

```text
num_speculative_tokens 不只影响算法；
还会影响 CUDA graph capture size 和编译形态。
```

---

## 23. routed experts 的边界

如果启用 routed experts 返回，普通 decode 和 spec decode 的切片方向不同。

Scheduler 中：

```python
if scheduled_spec_token_ids:
    # Spec decode: accepted tokens at the START of the scheduled range,
    # rejected at the end.
    routed_experts = routing_data[
        req_offset : req_offset + len(new_token_ids)
    ]
else:
    # Normal decode / re-prefill: token(s) at the END.
    routed_experts = routing_data[end - len(new_token_ids) : end]
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1750` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1757`

边界：

```text
普通 decode 输出通常对应 scheduled range 的尾部；
spec decode 的 accepted prefix 对应 scheduled range 的开头；
rejected draft 位于后面，不能作为输出 routed experts 返回。
```

V2 runner 里 routed experts capture 还被列为 unsupported：

位置：`code/vllm/vllm/config/vllm.py:2151` 到 `code/vllm/vllm/config/vllm.py:2152`

---

## 24. structured output + async draft token 回传边界

`DraftTokensHandler.set_draft_tokens()` 中：

```python
if not input_batch.has_structured_output_reqs:
    self.draft_tokens_np = None
    return
```

位置：`code/vllm/vllm/v1/worker/gpu/spec_decode/utils.py:22` 到 `code/vllm/vllm/v1/worker/gpu/spec_decode/utils.py:30`

如果 batch 有 structured output request，才异步把 draft tokens 拷回 CPU：

```python
self.draft_tokens_np = async_copy_to_np(draft_tokens)
```

位置：`code/vllm/vllm/v1/worker/gpu/spec_decode/utils.py:32` 到 `code/vllm/vllm/v1/worker/gpu/spec_decode/utils.py:42`

`get_draft_tokens()` 中，如果没有 structured output 且 async scheduling disabled，会返回：

```python
draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
```

位置：`code/vllm/vllm/v1/worker/gpu/spec_decode/utils.py:45` 到 `code/vllm/vllm/v1/worker/gpu/spec_decode/utils.py:52`

含义：

```text
没有 structured output 时，Scheduler 不一定需要知道真实 draft token ids；
有 structured output 时，Scheduler 必须拿到真实 draft token ids 做 grammar.validate_tokens()。
```

这也是 spec decode + structured output 性能边界：

```text
为了 grammar 校验，draft tokens 可能需要 GPU→CPU 回传和同步点。
```

---

## 25. ngram / CPU proposer / GPU proposer 的边界

ModelRunner 中选择何时 propose drafts：

```python
use_gpu_toks = (
    spec_config.use_eagle()
    or spec_config.uses_draft_model()
    or spec_config.uses_extract_hidden_states()
) and not spec_config.disable_padded_drafter_batch
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4565` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4584`

如果可以使用 GPU sampled tokens，drafter 可以在 bookkeeping 前运行：

```python
if input_fits_in_drafter:
    propose_draft_token_ids(sampled_token_ids)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4588`

否则：

```python
propose_drafts_after_bookkeeping = input_fits_in_drafter
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4631`

注释说明：

```text
ngram and other speculative decoding methods use the sampled tokens on the CPU,
so they are run after bookkeeping.
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4664`

所以不同 proposer 的性能边界不同：

```text
GPU drafter：可以减少 CPU sync，尽早生成下一轮 draft；
CPU / ngram 类 proposer：可能要等 sampled tokens 回 CPU 后再生成 draft；
disable_padded_drafter_batch=True：会影响这一优化路径，也会禁用 async scheduling。
```

---

## 26. synthetic rejection sampling 的边界

`SpeculativeConfig` 支持：

```text
rejection_sample_method = standard | synthetic
```

位置：`code/vllm/vllm/config/speculative.py:211` 到 `code/vllm/vllm/config/speculative.py:219`

synthetic 模式要求二选一：

```text
synthetic_acceptance_rates
或
synthetic_acceptance_length
```

校验逻辑：

```python
if (rates is None) == (length is None):
    raise ValueError(...)
```

位置：`code/vllm/vllm/config/speculative.py:244` 到 `code/vllm/vllm/config/speculative.py:274`

并要求：

```text
rates 长度等于 num_speculative_tokens；
rates 每项在 [0, 1]；
rates 单调不增；
acceptance_length 在 [1, num_speculative_tokens + 1]。
```

位置：`code/vllm/vllm/config/speculative.py:244` 到 `code/vllm/vllm/config/speculative.py:274`

运行时：

```python
self.synthetic_conditional_rates = torch.tensor(
    unconditional_to_conditional_rates(...)
)
self.synthetic_mode = self.synthetic_conditional_rates is not None
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:73` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:87`

含义：

```text
synthetic 不是普通 target/draft 概率 ratio test；
它主要用于模拟 acceptance 行为或 benchmark；
不能把 synthetic acceptance rate 当成真实 draft model 质量。
```

---

## 27. 性能边界：spec decode 不保证一定更快

Spec decode 提速依赖：

```text
1. draft tokens 接受率足够高；
2. drafter 成本明显低于 target decode；
3. 一次 target forward 验证多个 token 的收益超过额外 bookkeeping；
4. batch / CUDA graph / attention backend 能处理更长 decode query；
5. structured output / logprobs / bad_words 等额外同步开销不大。
```

可能变慢的场景：

```text
- acceptance rate 很低，第一个 draft 经常 rejected；
- draft model 太大或 TP / communication 成本高；
- structured output 频繁裁剪 draft，且需要回传 draft tokens；
- 采样参数复杂，penalties / bad_words / logprobs 需要展开处理；
- batch 很小，drafter overhead 占比高；
- batch 很大，target logits / logprobs / output parse 成本变高；
- max_model_len 附近频繁 zero out drafts；
- async scheduling / CUDA graph 因配置不兼容被关闭。
```

所以 spec decode 的正确心智不是：

```text
启用后一定更快。
```

而是：

```text
当 draft 便宜且接受率高时，用额外状态复杂度换 decode 吞吐。
```

---

## 28. 常见边界场景示例

### 28.1 mixed batch：有些请求没有 draft

一个 batch 中可以同时有：

```text
请求 A：scheduled_spec_decode_tokens = [a1, a2, a3]
请求 B：scheduled_spec_decode_tokens = []
请求 C：没有 spec tokens
```

RejectionSampler / ModelRunnerOutput 最终仍按每个 request 给出 `sampled_token_ids`。

Scheduler 只有在：

```python
scheduled_spec_token_ids and (...)
```

成立时才计算 accepted / rejected。

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1640` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1644`

所以没有 draft 的请求会退化成普通输出回收。

### 28.2 structured output 把所有 draft 都裁掉

如果 grammar.validate_tokens() 返回空：

```text
request.spec_token_ids = []
```

后续 Scheduler 可能不再把该请求视为本轮有 scheduled spec tokens。

结果：

```text
本轮只走普通 sampled token；
不会把 grammar 裁掉的 draft 当成 target rejection；
统计中 num_invalid_spec_tokens 用于记录这类情况。
```

### 28.3 request 执行期间被 abort

Scheduler 回收时会检查：

```python
request = self.requests.get(req_id)
if request is None or request.is_finished():
    continue
```

位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1624` 到 `code/vllm/vllm/v1/core/sched/scheduler.py:1630`

即使 ModelRunner 已经产生 tokens，请求如果执行期间被 abort / finish，也不会继续落账。

### 28.4 discard sampled tokens

ModelRunner bookkeeping 中会根据 `discard_request_mask` 清理 sampled tokens：

```python
discard_sampled_tokens_req_indices = np.nonzero(...)
...
valid_sampled_token_ids[int(i)].clear()
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4803` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4812`

spec decode 的 `parse_output()` 也支持 `discard_req_indices`：

```python
if len(discard_req_indices) > 0:
    valid_mask[discard_req_indices] = False
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:278` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:283`

---

## 29. 容易混淆的点

### 29.1 `scheduled_spec_decode_tokens` 是最终输出吗？

不是。

它表示：

```text
本轮交给 target model 验证的 draft tokens。
```

最终输出是：

```text
ModelRunnerOutput.sampled_token_ids
```

其中只包含 accepted draft tokens，以及 recovered / bonus token。

### 29.2 rejected draft 是否进入 Request.output_token_ids？

不会。

`RejectionSampler.parse_output()` 会过滤 `-1` placeholder：

```python
valid_mask = output_token_ids_np != PLACEHOLDER_TOKEN_ID
```

位置：`code/vllm/vllm/v1/sample/rejection_sampler.py:267` 到 `code/vllm/vllm/v1/sample/rejection_sampler.py:283`

Scheduler 只 append `generated_token_ids`。

### 29.3 accepted draft 是否还需要 target model 计算？

需要。

Spec decode 不是“不跑 target model”，而是：

```text
一次 target forward 验证多个 draft 位置。
```

accepted 只是说明 draft token 通过 target 分布检验，不能省略 target verification。

### 29.4 bonus token 是 draft model 给的吗？

不是。

bonus token 来自 target model 的 bonus logits，并通过普通 sampler 路径采样。

### 29.5 min_p / logit_bias 能和 spec decode 完全一致吗？

不能按普通 decode 预期理解。

代码会 warning：

```text
min_p and logit_bias parameters won't work with speculative decoding.
```

### 29.6 custom logits processor 能否自己保证兼容？

当前不行。

只要启用 spec decode，传入 custom logits processors 会直接报错。

### 29.7 structured output 下 draft token 被 grammar 裁掉，算 target rejected 吗？

不应该算。

这类 token 是 grammar invalid draft，不是 target model rejection。

Scheduler 用 `num_invalid_spec_tokens` 修正统计。

### 29.8 spec decode 下 logprobs 为什么外层 list 长度会大于 1？

因为一轮可能输出多个 confirmed tokens。

OutputProcessor 的 logprobs processor 也承认这个场景：

```text
Outer lists are only of len > 1 if EngineCore made >1 tokens in prior step (e.g. in spec decoding).
```

位置：`code/vllm/vllm/v1/engine/logprobs.py:72` 到 `code/vllm/vllm/v1/engine/logprobs.py:73`

---

## 30. 排查问题时看哪些对象

如果 spec decode 输出不符合预期，优先看这几个对象：

```text
SpeculativeConfig：
  method / num_speculative_tokens / draft model / draft TP / max_model_len。

Request：
  output_token_ids / spec_token_ids / num_computed_tokens / num_output_placeholders。

SchedulerOutput：
  num_scheduled_tokens / scheduled_spec_decode_tokens / num_invalid_spec_tokens。

SpecDecodeMetadata：
  draft_token_ids / num_draft_tokens / target_logits_indices / bonus_logits_indices / logits_indices。

SamplerOutput：
  sampled_token_ids tensor，包含 accepted / recovered / bonus / -1 placeholder。

ModelRunnerOutput：
  sampled_token_ids list[list[int]]，已经过滤 rejected / padding。

EngineCoreOutput：
  new_token_ids / new_logprobs / finish_reason / stop_reason。
```

最小定位链路：

```text
Request.spec_token_ids
  → SchedulerOutput.scheduled_spec_decode_tokens
  → SpecDecodeMetadata.draft_token_ids
  → RejectionSampler.sampled_token_ids
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler.update_from_output()
  → Request.output_token_ids / EngineCoreOutput.new_token_ids
```

---

## 31. 从“回答问题”的角度总结

如果要问：

```text
Spec decode 有哪些限制和边界场景？
```

可以回答：

```text
Spec decode 的主要限制来自状态一致性。它把一次 decode 扩展成“多个 draft token + 一个 bonus/recovered token”的验证过程，因此任何会影响 token 合法性、采样分布或 KV 进度的功能，都必须按 draft-token 级别对齐。

配置上，num_speculative_tokens 必须有效；draft_model 需要和 target vocab 一致；draft TP 只能是 1 或 target TP；draft max_model_len 不能超过 draft / target 能力。不同 method 在 V1、V2、async scheduling 下支持范围不同，V2 对 ngram、dynamic spec decode、EAGLE parallel drafting、EAGLE3+PP 等还有额外限制。

采样上，自定义 logits processor 不支持；min_p 和 logit_bias 在 spec decode 下会 warning 不按普通方式生效；bonus token 可以走普通 sampler 的 top-k/top-p，但 accepted/recovered 主体是 rejection sampling，不等同于每个 draft 位置都执行普通 sampling。allowed_token_ids、bad_words、penalties 需要按 draft token 展开并合并本轮 spec tokens。

structured output 下，draft tokens 需要 grammar.validate_tokens() 提前裁剪，grammar bitmask 也必须按 1 + num_spec_tokens 的 logits rows 对齐。最终 grammar.accept_tokens() 只接受 confirmed new_token_ids；如果 rejected draft 或 grammar-invalid draft 进入最终输出，就说明状态错位。

KV / 调度上，Scheduler 会乐观推进 num_computed_tokens，但 rejected draft 必须回滚；async scheduling 还要回滚 num_output_placeholders。prefill chunk 会忽略 draft tokens；接近 max_model_len 或 draft model 容量不足时会 zero out draft，避免 stale draft 被下一轮误用。external KV connector、prefix cache、encoder cache 释放都要避开 pending draft rejection 带来的进度回滚。

输出上，一轮 spec decode 可能 accepted=0，也可能全部 accepted 并额外产生 bonus token；stop/EOS 可能出现在多 token 输出中间，Scheduler 会裁掉后缀。OutputProcessor 最终只消费 EngineCoreOutput.new_token_ids，不再区分 accepted、recovered、bonus。
```

职责边界可以概括为：

```text
配置层决定能不能启用；
Scheduler 决定能验证哪些 draft；
ModelRunner / drafter 决定能不能生成下一轮 draft；
RejectionSampler 决定 accepted / recovered / bonus；
Scheduler.update_from_output() 负责回滚 rejected 并提交 confirmed tokens。
```

---

## 32. 最关键流程图

```text
配置阶段

SpeculativeConfig
  → method / num_speculative_tokens 校验
  → draft model / vocab / TP / max_model_len 校验
  → async / V2 / backend 兼容性判断
```

```text
调度阶段

Request.spec_token_ids
  → Scheduler.schedule()
      → scheduled_spec_decode_tokens
      → num_scheduled_tokens
      → max_model_len / token_budget / chunked prefill 限制
```

```text
执行阶段

GPUModelRunner.execute_model()
  → _prepare_inputs()
  → SpecDecodeMetadata(logits_indices / target_logits_indices / bonus_logits_indices)
  → target model forward
  → logits
  → grammar bitmask 按 spec rows 对齐
  → RejectionSampler
```

```text
采样阶段

RejectionSampler
  → bonus token：普通 sampler 路径
  → target logits：apply penalties / bad_words / allowed ids / min_tokens
  → rejection_sample()
  → [accepted..., recovered/bonus, -1 padding]
  → parse_output()
  → valid_sampled_token_ids
```

```text
回收阶段

Scheduler.update_from_output()
  → generated_token_ids = ModelRunnerOutput.sampled_token_ids[req_index]
  → scheduled_spec_token_ids = SchedulerOutput.scheduled_spec_decode_tokens[req_id]
  → num_accepted = len(generated_token_ids) - num_sampled
  → num_rejected = len(scheduled_spec_token_ids) - num_accepted
  → rollback num_computed_tokens / num_output_placeholders
  → append confirmed new_token_ids
  → grammar.accept_tokens(new_token_ids)
  → slice logprobs / routed experts
  → EngineCoreOutput.new_token_ids
```

---

## 33. 最小心智模型

如果只记一条：

```text
Spec decode 的边界不是“多采几个 token”，而是“多验证几个候选 token 后，只提交 confirmed tokens，并回滚所有没有成为真实上下文的状态”。
```

再压缩成一句话：

```text
只要某个功能会改变 token 合法性、采样分布或 KV 进度，就必须和 accepted / rejected / recovered / bonus 的语义对齐；对不具备这种对齐能力的功能，vLLM 要么禁用，要么降级，要么在回收阶段显式修正。
```
