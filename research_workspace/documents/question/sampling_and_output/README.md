# Sampling and Output 文档目录

本目录用于梳理 vLLM V1 中从 **model forward 产出 hidden states / logits** 到 **sampler 采样**，再到 **ModelRunnerOutput / Scheduler.update_from_output / OutputProcessor / RequestOutput** 的完整输出链路。

它接在 `executor_worker_model_runner` 之后，重点回答：

```text
模型已经 forward 完成之后：

1. logits 是在哪里算出来的？
2. sampler 消费哪些 metadata？
3. top-k / top-p / temperature / seed / penalties / logprobs 怎么进入采样？
4. spec decode 的 draft / target / bonus token 如何影响采样？
5. structured output / grammar bitmask 在哪里接入？
6. ModelRunnerOutput 是什么？它和 RequestOutput 有什么区别？
7. Scheduler 如何根据采样结果更新 request 状态？
8. OutputProcessor 如何把内部输出转换成用户可见输出？
9. streaming 输出、finished reason、logprobs、prompt logprobs 如何组织？
10. pooling / embedding / rerank 这类非生成输出如何走不同路径？
```

---

## 阅读顺序建议

```text
sampling_and_output_overview.md
  → 01_sampling_output_role.md
  → 02_sampling_params_and_metadata.md
  → 03_logits_and_logprobs.md
  → 04_sampler_flow.md
  → 05_spec_decode_sampling.md
  → 06_structured_output_and_grammar.md
  → 07_model_runner_output.md
  → 08_scheduler_update_output.md
  → 09_output_processor_request_output.md
  → 10_streaming_and_client_outputs.md
  → 11_pooling_and_embedding_outputs.md
```

如果只想先抓住主线，可以先读：

```text
sampling_and_output_overview.md
  → 03_logits_and_logprobs.md
  → 04_sampler_flow.md
  → 07_model_runner_output.md
  → 08_scheduler_update_output.md
  → 09_output_processor_request_output.md
```

---

## 文档定位

```text
sampling_and_output_overview.md：
  总览主文档，建立从 logits 到用户输出的全局图。

01-11：
  按问题拆开的专题文档，用于逐篇核对源码细节和输出链路边界。
```
