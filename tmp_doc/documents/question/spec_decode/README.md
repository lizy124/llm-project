# Spec Decode 文档目录

本目录用于梳理 vLLM V1 中 speculative decoding 的完整链路。

Spec decode 不是 sampling 阶段的一个小分支，而是横跨：

```text
Scheduler
  → draft token 生成 / 管理
  → SchedulerOutput
  → ModelRunner input preparation
  → SpecDecodeMetadata
  → target model forward
  → RejectionSampler
  → KV cache / num_computed_tokens 修正
  → Scheduler.update_from_output()
  → OutputProcessor
```

它要回答的核心问题是：

```text
如何用 draft model 或 draft tokens 提前猜测多个 token，
再用 target model 一次验证，
最终只接受合法 token，
并保持 request 状态、KV cache、输出和 grammar 状态一致。
```

---

## 阅读顺序建议

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

如果只想先抓主线，可以先读：

```text
spec_decode_overview.md
  → 03_scheduler_spec_decode_flow.md
  → 04_spec_decode_metadata.md
  → 06_rejection_sampler_flow.md
  → 09_output_recovery_and_scheduler_update.md
```

---

## 文档定位

```text
spec_decode_overview.md：
  总览主文档，建立 speculative decoding 的完整闭环。

01-10：
  按问题拆开的专题文档，后续逐篇补源码细节。
```
