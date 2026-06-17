# 04 Scheduler 调度机制

本篇详细梳理 V1 `Scheduler`。这是 vLLM 推理引擎层最核心的状态机之一。

`Scheduler` 定义在 `code/vllm/vllm/v1/core/sched/scheduler.py:68`，调度主函数是 `schedule()`，位于 `code/vllm/vllm/v1/core/sched/scheduler.py:387`。

## 1. Scheduler 的职责

Scheduler 不执行模型，它负责回答每一步最关键的问题：

1. 这一步要执行哪些 request？
2. 每个 request 执行多少 token？
3. 这些 token 需要哪些 KV cache block？
4. 哪些 request 要从 waiting 进入 running？
5. 哪些 running request 要继续 decode/prefill？
6. KV cache 不够时要不要 preempt？
7. prefix cache 命中多少 token？
8. spec decode 的 draft token 如何参与调度？
9. encoder/multimodal 输入是否有预算？
10. 请求完成后如何释放资源？

一句话：Scheduler 是 request/token/KV resource 的统一调度器。

## 2. Scheduler 内部状态

初始化时，Scheduler 会保存大量配置和状态。

关键状态包括：

| 状态 | 作用 |
|---|---|
| `self.requests` | `req_id -> Request` 全量请求表 |
| `self.waiting` | 等待调度的新请求队列 |
| `self.skipped_waiting` | 因资源/依赖暂时跳过的 waiting 请求 |
| `self.running` | 正在运行或已经进入 batch 管理的请求 |
| `self.finished_req_ids` | 已完成、需要通知 worker 清理的请求 |
| `self.max_num_running_reqs` | 最大并发序列数，来自 scheduler config |
| `self.max_num_scheduled_tokens` | 单步 token budget |
| `self.kv_cache_manager` | KV block 分配、复用、释放 |
| `self.encoder_cache_manager` | encoder/multimodal cache 管理 |
| `self.connector` | KV transfer/offload connector |
| `self.ec_connector` | encoder cache transfer connector |
| `self.structured_output_manager` | 结构化输出约束 |

初始化相关代码：

- scheduling constraints：`code/vllm/vllm/v1/core/sched/scheduler.py:106`
- request queues：`code/vllm/vllm/v1/core/sched/scheduler.py:170`
- finished req ids：`code/vllm/vllm/v1/core/sched/scheduler.py:185`
- encoder cache manager：`code/vllm/vllm/v1/core/sched/scheduler.py:221`
- spec decode 配置：`code/vllm/vllm/v1/core/sched/scheduler.py:227`
- KVCacheManager 创建：`code/vllm/vllm/v1/core/sched/scheduler.py:253`

## 3. Scheduler 的核心思想：没有固定 prefill/decode 阶段

`schedule()` 的注释非常关键，在 `code/vllm/vllm/v1/core/sched/scheduler.py:389` 附近：

> There's no "decoding phase" nor "prefill phase" in the scheduler.

它的意思是：Scheduler 不先判断“现在是 prefill 还是 decode”，而是对每个 request 维护：

```text
num_computed_tokens
num_tokens_with_spec
```

每一步都尝试让：

```text
num_computed_tokens 追上 num_tokens_with_spec
```

这样可以统一支持：

- 普通 prefill；
- chunked prefill；
- decode；
- prefix caching；
- speculative decoding；
- 将来的 jump decoding；
- encoder-decoder/multimodal 输入。

## 4. schedule() 主流程

`schedule()` 的高层流程：

```text
1. current_step += 1
2. 初始化本步输出容器
3. 设置 token_budget
4. kv_cache_manager.new_step_starts()
5. 先调度 RUNNING 请求
6. 再调度 WAITING 请求
7. 构造 SchedulerOutput
8. 更新内部状态
9. 返回 SchedulerOutput 给 EngineCore
```

### 先调度 running，再调度 waiting

这点非常重要。

vLLM 优先保证已经进入 running 的请求继续前进，否则 decode 请求容易被大量 prefill 饿死。

主流程：

```text
for request in running:
  计算这个 request 本步需要多少 token
  检查 token budget / max_model_len / encoder budget
  kv_cache_manager.allocate_slots(...)
  如果 KV 不够，preempt 低优先级 request
  记录 scheduled_running_reqs

while waiting and token_budget > 0:
  取 waiting request
  检查 LoRA / KV connector / prefix cache / encoder budget
  kv_cache_manager.allocate_slots(...)
  放入 running
```

## 5. token budget

Scheduler 每步有一个 token budget：

```text
token_budget = self.max_num_scheduled_tokens
```

来源：

```text
scheduler_config.max_num_scheduled_tokens
或 scheduler_config.max_num_batched_tokens
```

作用：限制单个 engine step 内总共执行的 token 数。

每当一个 request 被调度：

```text
token_budget -= num_new_tokens
```

当 token_budget 用完，本步不再调度更多 token。

## 6. running request 调度细节

running request 调度从 `code/vllm/vllm/v1/core/sched/scheduler.py:429` 开始。

对每个 running request，Scheduler 会：

1. 跳过已经达到 max tokens 的请求；
2. 检查 PP/async scheduling 的 decode cadence；
3. 在 DP prefill balancing 时可能延迟 prefill chunk；
4. 计算 `num_new_tokens`；
5. 应用 `long_prefill_token_threshold`；
6. 限制不超过 token budget；
7. 限制不超过 `max_model_len`；
8. 尝试调度 encoder inputs；
9. 如果 Mamba hybrid 模型要求 block aligned，则对 token 数做对齐切分；
10. 调 `kv_cache_manager.allocate_slots()` 分配 KV block；
11. KV 不足时 preempt request；
12. 成功后记录到本步调度输出。

## 7. waiting request 调度细节

waiting request 调度从 `code/vllm/vllm/v1/core/sched/scheduler.py:624` 开始。

它会检查更多 admission 条件：

- running 请求数是否已经到 `max_num_running_reqs`；
- request 是否处于 blocked 状态；
- LoRA 数量是否超过 `max_loras`；
- prefix cache 命中情况；
- KV connector 是否能提供外部 KV；
- 是否需要异步 load remote KV；
- encoder/multimodal budget；
- full sequence 是否必须一次 fit；
- KV block 是否足够。

新 request 成功调度后，会从 waiting 移到 running。

## 8. prefix caching 在调度中的位置

对于 waiting request，尤其是 `request.num_computed_tokens == 0` 时，Scheduler 会尝试找本地 prefix cache：

```text
new_computed_blocks, num_new_local_computed_tokens =
    self.kv_cache_manager.get_computed_blocks(request)
```

对应代码在 `code/vllm/vllm/v1/core/sched/scheduler.py:709`。

含义：

- 如果 prompt 的前缀已经在 KV cache 中，直接复用这些 block；
- 复用部分计入 computed tokens；
- 后续只调度未命中的 token；
- 但如果全部命中，也通常需要重算最后一个 token 以获得 logits。

## 9. KV 不够时的 preemption

running 请求调度时，如果 `allocate_slots()` 返回 `None`，说明 KV block 不够。

Scheduler 会尝试 preempt 请求：

- 如果是 priority policy，选择优先级最低的请求；
- 否则通常 pop running 队尾请求；
- 被 preempt 的请求会释放/回收部分状态，后续重新进入等待或恢复。

对应逻辑从 `code/vllm/vllm/v1/core/sched/scheduler.py:534` 开始。

preemption 是 vLLM 在高负载下维持吞吐和资源利用的重要机制，但也会导致请求重新计算部分 KV。

## 10. speculative decoding 调度

Scheduler 中 spec decode 相关状态：

- `self.num_spec_tokens`
- `self.num_lookahead_tokens`
- `self.use_eagle`
- `self.dynamic_sd_lookup`

调度时：

1. request 可能携带 `spec_token_ids`；
2. Scheduler 会把 spec tokens 计入 `num_tokens_with_spec`；
3. `allocate_slots()` 会额外分配 lookahead tokens；
4. 本步执行后，worker/model runner 产生 draft tokens；
5. `EngineCore.post_step()` 取回 draft token；
6. `Scheduler.update_draft_token_ids()` 更新到 request，供下一步使用。

## 11. encoder/multimodal 调度

Scheduler 会计算 encoder budget：

```text
self.max_num_encoder_input_tokens
```

并维护：

```text
self.encoder_cache_manager
```

当 request 有 encoder inputs 时，Scheduler 调用 `_try_schedule_encoder_inputs()`：

- 判断本步是否有 encoder compute budget；
- 判断 encoder cache 是否够；
- 为 encoder inputs 分配 cache；
- 生成 `scheduled_encoder_inputs`，传给 worker。

这使文本生成、encoder-decoder、多模态输入可以统一进入同一个 step 调度。

## 12. SchedulerOutput 包含什么

`SchedulerOutput` 定义在 `vllm/v1/core/sched/output.py`。

它大致包含：

- 本步新请求数据；
- 本步 cached request 数据；
- 每个 request 的 scheduled token 数；
- 新分配的 block ids；
- common prefix blocks；
- scheduled encoder inputs；
- scheduled spec decode tokens；
- finished request ids；
- structured output / grammar 相关数据；
- KV connector metadata；
- scheduler stats。

它是 Scheduler 和 Worker 之间最重要的数据契约。

## 13. update_from_output：模型输出回写状态

模型执行返回后，EngineCore 会调用：

```text
scheduler.update_from_output(scheduler_output, model_output)
```

方法位置：`code/vllm/vllm/v1/core/sched/scheduler.py:1463`。

它负责：

- 根据 sampled token 更新 request output token；
- 检查 stop condition；
- 更新 request status；
- 处理 streaming update；
- 处理 spec decode accept/reject；
- 处理 KV connector output；
- 释放完成请求的 encoder/KV 资源；
- 生成 `EngineCoreOutputs`。

## 14. finish/free/reset

Scheduler 还负责资源清理：

| 方法 | 作用 |
|---|---|
| `finish_requests()` | 标记请求完成/取消 |
| `_free_request()` | 清理 request 对象 |
| `_free_blocks()` | 释放 KV blocks |
| `_free_request_blocks()` | 释放请求占用的 blocks |
| `_drain_deferred_frees()` | 处理延迟释放 |
| `reset_prefix_cache()` | 清理 prefix cache |
| `reset_encoder_cache()` | 清理 encoder cache |

资源释放是正确性关键点。尤其是 overlapping batches、KV connector、async scheduling 场景，不能过早释放正在被 kernel 写入的 block。

## 15. Scheduler 的核心输入输出

### 输入

- 新请求：`Request`；
- 当前 request 状态；
- KV cache 状态；
- token budget；
- ModelRunnerOutput；
- connector output；
- draft token ids；
- pause/resume 状态。

### 输出

- `SchedulerOutput`：给 worker 执行；
- `EngineCoreOutputs`：给前台输出；
- scheduler stats；
- prefix cache stats；
- spec decode stats。

## 16. 理解 Scheduler 的一句话

Scheduler 的本质不是简单的“prefill/decode 排队器”，而是一个逐步推进 `num_computed_tokens` 的资源调度器：它在每个 step 内用有限 token budget 和 KV block，把 running 和 waiting 请求推进一小段，并把这个决定编码成 `SchedulerOutput` 交给 worker。
