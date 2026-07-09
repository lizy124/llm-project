# vLLM V1 Scheduler 问题目录

源码位置：`vllm/vllm/v1/core/sched/scheduler.py`

这个目录按问题拆解 vLLM V1 `Scheduler` 的主流程。建议先读总览，再按 `01` 到 `08` 顺序读专题；遇到 speculative decoding 相关概念时，再读补充专题。

---

## 1. 总览文档

- [vLLM V1 Scheduler 逻辑梳理](vllm_scheduler.md)

适合第一次建立全局印象。它按源码主链路梳理：

```text
add_request
  → waiting / running / skipped_waiting
  → schedule()
  → SchedulerOutput
  → ModelRunner forward
  → update_from_output()
  → stop / free / connector cleanup
```

如果只想快速了解 Scheduler 做什么，先读这篇。

---

## 2. 主线专题阅读顺序

### 01. 请求状态与队列

- [当前有哪些请求在等待、运行、阻塞？](01_request_states.md)

回答：

```text
self.requests 是什么？
self.running / self.waiting / self.skipped_waiting 分别保存什么？
blocked waiting 和 temporarily skipped waiting 有什么区别？
请求状态如何在 WAITING / RUNNING / PREEMPTED / blocked / finished 之间迁移？
```

建议作为所有专题的起点。

### 02. token budget

- [本轮最多能调度多少 token？](02_token_budget.md)

回答：

```text
max_num_scheduled_tokens 从哪里来？
token_budget 如何初始化和消耗？
running 与 waiting 如何共同使用 token budget？
为什么实际调度 token 数可能小于上限？
```

### 03. running 请求继续推进

- [哪些 running 请求继续 decode / prefill？](03_running_decode_prefill.md)

回答：

```text
self.running 里的请求本轮是否一定执行？
running 请求如何计算 num_new_tokens？
async / PP cadence、DP prefill balancing、encoder、Mamba、KV block 如何让 running 请求被跳过？
本轮真正执行的 scheduled_running_reqs 如何进入 SchedulerOutput？
```

### 04. waiting 请求进入 running

- [哪些 waiting 请求可以进入运行态？](04_waiting_to_running.md)

回答：

```text
waiting 阶段什么时候执行？
waiting 和 skipped_waiting 如何选队列？
blocked waiting 如何尝试恢复？
LoRA、KV Connector、encoder、chunked prefill、Mamba、KV block 如何影响 waiting → running？
WAITING 和 PREEMPTED 请求如何分别进入 scheduled_new_reqs / scheduled_resumed_reqs？
```

### 05. prefix cache / 外部 KV cache 命中

- [prefix cache / 外部 KV cache 命中了多少 token？](05_prefix_and_external_kv_hits.md)

回答：

```text
本地 prefix cache 如何查询？
为什么 full hit 仍要重算最后一个 token？
KV Connector 如何返回外部额外命中？
num_computed_tokens = local hit + external hit 如何影响 num_new_tokens？
async external KV load 为什么进入 WAITING_FOR_REMOTE_KVS？
```

### 06. KV block 分配与抢占

- [KV Cache block 是否够用，不够时是否需要抢占？](06_kv_block_allocation_and_preemption.md)

回答：

```text
allocate_slots() 如何判断 KV block 是否够？
computed / external / new / lookahead blocks 如何布局？
running 请求 block 不够时如何抢占？
waiting 请求 block 不够时为什么停止 waiting 阶段而不是抢占？
请求结束后 block 如何释放或 deferred free？
```

### 07. 附加调度能力

- [多模态 encoder 输入、结构化输出、投机解码等附加能力如何同步调度？](07_auxiliary_scheduling_features.md)

回答：

```text
encoder input 如何嵌入 running / waiting 调度？
structured output grammar 如何阻塞、生成 bitmask、推进状态？
spec decode 如何影响 num_tokens_with_spec、lookahead blocks、SchedulerOutput 和回退？
LoRA、Mamba、DP prefill balancing、pause state、KV / EC Connector metadata 如何影响调度？
```

### 08. Worker 输出回收

- [Worker 执行完后，如何更新请求状态、释放 block、返回输出？](08_update_after_worker_output.md)

回答：

```text
schedule() 发出 SchedulerOutput 后，update_from_output() 如何消化 ModelRunnerOutput？
sampled token、pooling output、logprobs、spec 接受/拒绝、grammar accept_tokens 如何处理？
stop、streaming / resumable、_free_request、connector finished_recving / finished_sending 如何闭环？
EngineCoreOutputs 如何按 client_index 返回？
```

这是主流程收尾篇。

---

## 3. 补充专题

### token 状态专题

- [Scheduler 中各种 token 数和请求状态到底是什么](token_states.md)

建议在读 `02`、`03`、`04`、`08` 时配合阅读。

它专门解释 Scheduler 中容易混淆的基础 token / state 概念：

```text
num_tokens
num_computed_tokens
num_tokens_with_spec
num_output_placeholders
num_scheduled_tokens
WAITING / RUNNING / PREEMPTED / blocked / finished
```

### Spec Decode token 状态专题

- [Spec Decode 中各种 token 和状态到底是什么](spec_decode_token_states.md)

建议在读 `03`、`07`、`08` 时配合阅读。

它专门解释 speculative decoding / async scheduling 中容易混淆的概念：

```text
num_tokens
num_tokens_with_spec
spec_token_ids
scheduled_spec_decode_tokens
num_output_placeholders
num_computed_tokens
num_scheduled_tokens
accepted draft tokens
rejected draft tokens
target-sampled token
bonus token
```

重点回答：

```text
num_tokens_with_spec 里包含的 spec token，
到底是还没调度、正在 forward、还是已经验证完？
```

---

## 4. 推荐阅读路线

### 4.1 快速建立主流程

```text
vllm_scheduler.md
  → 01_request_states.md
  → token_states.md
  → 02_token_budget.md
  → 03_running_decode_prefill.md
  → 04_waiting_to_running.md
  → 08_update_after_worker_output.md
```

适合先理解 Scheduler 从接收请求到返回输出的闭环。

### 4.2 深入 KV cache / KVPool / 外部 KV

```text
01_request_states.md
  → 04_waiting_to_running.md
  → 05_prefix_and_external_kv_hits.md
  → 06_kv_block_allocation_and_preemption.md
  → 08_update_after_worker_output.md
```

重点关注：

```text
prefix cache 命中
external KV 命中
async remote KV load
WAITING_FOR_REMOTE_KVS
allocate_slots
deferred free
finished_recving / finished_sending
```

### 4.3 深入 speculative decoding

```text
spec_decode_token_states.md
  → 03_running_decode_prefill.md
  → 07_auxiliary_scheduling_features.md
  → 08_update_after_worker_output.md
```

重点关注：

```text
spec_token_ids 如何进入 num_tokens_with_spec
scheduled_spec_decode_tokens 如何进入 SchedulerOutput
lookahead tokens 如何影响 KV block
accepted / rejected draft tokens 如何回退 num_computed_tokens
```

### 4.4 深入多模态 / encoder / structured output

```text
07_auxiliary_scheduling_features.md
  → 04_waiting_to_running.md
  → 08_update_after_worker_output.md
```

重点关注：

```text
encoder_compute_budget
scheduled_encoder_inputs
ECConnector metadata
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
grammar bitmask
grammar.accept_tokens
```

---

## 5. 文档定位

这个文件夹里的文档可以这样理解：

```text
vllm_scheduler.md：
  总览主文档，适合快速建立全局图。

01-08：
  按问题拆开的专题文档，适合逐段精读 scheduler.py。

token_states.md：
  Scheduler 基础 token / request state 概念补充，适合解决 token 计数和状态迁移混淆。

spec_decode_token_states.md：
  spec decode 概念补充，适合解决投机解码 token 状态混淆。

README.md：
  当前目录索引和阅读路线。
```

如果后续继续补充新专题，建议也在这里追加到“补充专题”或“推荐阅读路线”中。