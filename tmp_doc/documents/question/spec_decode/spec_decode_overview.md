# vLLM V1 Spec Decode 逻辑梳理

源码位置：

- `code/vllm/vllm/v1/spec_decode/`
- `code/vllm/vllm/v1/spec_decode/metadata.py`
- `code/vllm/vllm/v1/core/sched/scheduler.py`
- `code/vllm/vllm/v1/core/sched/output.py`
- `code/vllm/vllm/v1/request.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/`
- `code/vllm/vllm/v1/sample/rejection_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- `code/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`
- `code/vllm/vllm/v1/outputs.py`
- `code/vllm/vllm/v1/structured_output/utils.py`
- `code/vllm/vllm/sampling_params.py`

本文按“先定边界，再走主链路，再拆关键阶段，最后总结接口和状态修正”的方式，梳理 vLLM V1 speculative decoding 的完整机制。

它和 `scheduler`、`executor_worker_model_runner`、`sampling_and_output` 都有关，但这里专门把分散逻辑串成一个闭环。

---

## 0. 梳理规划

本目录要回答的问题分成 10 组：

```text
1. Spec decode 在 vLLM 中解决什么问题？
2. draft tokens 从哪里来，如何挂在 request 状态上？
3. Scheduler 如何决定本轮带哪些 draft tokens？
4. SpecDecodeMetadata 如何描述 draft / target / bonus logits 位置？
5. ModelRunner 如何把 spec decode 请求展开成 target model forward 输入？
6. RejectionSampler 如何接受、拒绝、采样 bonus token？
7. KV cache、num_computed_tokens、output_token_ids 如何保持一致？
8. structured output / grammar 如何和 spec decode 交互？
9. Scheduler.update_from_output() 如何回收 accepted / rejected tokens？
10. Spec decode 有哪些限制、边界和不支持场景？
```

阅读顺序建议：

```text
spec_decode_overview.md
  → 01_spec_decode_role.md
  → 02_draft_tokens_and_request_state.md
  → 03_scheduler_spec_decode_flow.md
  → 04_spec_decode_metadata.md
  → 05_model_runner_spec_forward.md
  → 06_rejection_sampler_flow.md
  → 07_kv_cache_and_num_computed_tokens.md
  → 08_structured_output_interaction.md
  → 09_output_recovery_and_scheduler_update.md
  → 10_limitations_and_edge_cases.md
```

---

## 1. 一句话回答

Spec decode 的本质是：

```text
用便宜路径先猜多个 draft tokens，
再用 target model 一次 forward 验证这些 token，
通过 rejection sampling 接受一段前缀，
必要时重新采样或补一个 bonus token，
最后把 request、KV cache、grammar、输出状态修正到一致。
```

最小主线是：

```text
Request 当前已有 tokens
  → draft model / proposer 产生 draft tokens
  → Scheduler 把 draft tokens 放入 SchedulerOutput
  → ModelRunner._prepare_inputs()
      → 构造 spec decode positions / logits_indices
      → 构造 SpecDecodeMetadata
  → target model forward
      → target logits
  → RejectionSampler
      → accepted tokens
      → rejected 后替代 token
      → bonus token
  → ModelRunnerOutput
  → Scheduler.update_from_output()
      → append accepted tokens
      → 修正 rejected / invalid draft
      → 更新 num_computed_tokens / KV 状态
  → OutputProcessor
```

---

## 2. Spec decode 横跨哪些模块

### 2.1 Scheduler 层

Scheduler 负责决定：

```text
- 哪些请求本轮可以带 draft tokens；
- 每个请求本轮带多少 draft tokens；
- draft tokens 是否受 grammar / structured output 修剪；
- spec decode token 数如何影响 token budget；
- 请求状态如何在 accepted / rejected 后更新。
```

### 2.2 ModelRunner 层

ModelRunner 负责：

```text
- 把 draft tokens 展开到本轮 input ids；
- 计算 target logits 对应位置；
- 构造 SpecDecodeMetadata；
- 保存 ExecuteModelState；
- 调用 RejectionSampler；
- 将采样结果转成 ModelRunnerOutput。
```

### 2.3 Sampler 层

Sampler / RejectionSampler 负责：

```text
- 比较 draft probability 和 target probability；
- 决定接受几个 draft tokens；
- 拒绝时从修正分布重新采样；
- 全部接受时采样 bonus token；
- 复用 SamplingMetadata 中的 temperature / top-k / top-p / penalty / allowed tokens 等约束。
```

### 2.4 KV cache / output 层

状态回收需要保证：

```text
- output_token_ids 只包含 accepted tokens；
- num_computed_tokens 和真实 KV cache 进度一致；
- rejected draft token 不会污染最终输出；
- grammar 状态只接受合法输出；
- logprobs / prompt logprobs 和最终 token 对齐。
```

---

## 3. 总体流程图

```text
上一轮输出结束
  → draft proposer / draft model 准备 draft tokens
  → Scheduler.schedule()
      → selected requests
      → scheduled_spec_decode_tokens
      → token budget / KV allocation
  → SchedulerOutput
  → GPUModelRunner._update_states()
      → 同步 request / draft state
  → GPUModelRunner._prepare_inputs()
      → input_ids 包含 target model 要验证的位置
      → logits_indices 不再只是每个请求最后一个 token
      → SpecDecodeMetadata
  → _build_attention_metadata()
      → attention 仍按本轮 tokens 构造 metadata
  → _model_forward()
      → target hidden states
      → target logits
  → sample_tokens()
      → grammar bitmask 对齐 spec logits rows
      → _sample()
      → RejectionSampler
  → ModelRunnerOutput
      → accepted / sampled tokens
      → logprobs
      → spec decode 相关结果
  → Scheduler.update_from_output()
      → 更新 request.output_token_ids
      → 修正 num_computed_tokens
      → grammar accept_tokens
      → KV / connector / finished 状态处理
  → EngineCoreOutputs
  → OutputProcessor
```

---

## 4. 和普通 decode 的关键差异

| 阶段 | 普通 decode | spec decode |
|---|---|---|
| 本轮 token | 每个请求通常 1 个新 token | 每个请求可能带多个 draft tokens + bonus 位置 |
| logits_indices | 每个请求最后一个 query 位置 | target logits positions + bonus logits positions |
| sampler | `Sampler` | `RejectionSampler`，内部可能调用普通 `Sampler` |
| 输出 token | sampler 直接给出新 token | draft tokens 需要接受 / 拒绝后才成为输出 |
| KV 状态 | 单步推进 | 需要与 accepted tokens 对齐，拒绝部分要修正 |
| structured output | mask 当前一步 logits | 还要验证 / trim draft tokens，并对齐 bitmask offset |
| Scheduler 回收 | append sampled token + stop check | append accepted tokens + 修正 rejected / bonus / draft state |

---

## 5. 后续专题占位

```text
01_spec_decode_role.md：
  定义 spec decode 在 vLLM 中的职责、收益、边界，以及和普通 decode 的区别。

02_draft_tokens_and_request_state.md：
  梳理 draft tokens 从哪里来，如何保存在 request / scheduler / worker 状态中。

03_scheduler_spec_decode_flow.md：
  梳理 Scheduler 如何选择 spec decode 请求、调度 draft tokens、处理 token budget 和 grammar 修剪。

04_spec_decode_metadata.md：
  梳理 SpecDecodeMetadata 中 draft_token_ids、target_logits_indices、bonus_logits_indices 等字段。

05_model_runner_spec_forward.md：
  梳理 ModelRunner 如何准备 spec decode 输入、positions、logits_indices、attention metadata 和 target logits。

06_rejection_sampler_flow.md：
  梳理 RejectionSampler 如何接受 / 拒绝 draft tokens，以及如何采样 replacement / bonus token。

07_kv_cache_and_num_computed_tokens.md：
  梳理 accepted / rejected token 如何影响 KV cache、slot mapping、num_computed_tokens 和 recompute。

08_structured_output_interaction.md：
  梳理 grammar bitmask、draft token trim、invalid spec tokens 和 structured output 状态推进。

09_output_recovery_and_scheduler_update.md：
  梳理 ModelRunnerOutput 到 Scheduler.update_from_output() 的回收逻辑。

10_limitations_and_edge_cases.md：
  梳理 spec decode 与 sampling 参数、backend、chunked prefill、logprobs、grammar、KV connector 等限制。
```

---

## 6. 一句话总结

Spec decode 是 vLLM 推理链路中的“多 token 猜测 + target 验证 + 状态修正”机制：

```text
它不是单独的 sampler 技巧，
而是贯穿 Scheduler、ModelRunner、RejectionSampler、KV cache 和 Output 回收的完整执行模式。
```
