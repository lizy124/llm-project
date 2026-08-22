# vLLM V1 Scheduler 总览

源码仓库：vLLM

源码路径均相对于 vLLM 仓库根目录；如果本地源码放在 `code/vllm/` 下，实际路径前面加上 `code/vllm/` 即可。本文只使用文件路径和符号名定位源码，不依赖行号。

核心源码文件：

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/sched/interface.py`
- `vllm/v1/core/sched/output.py`
- `vllm/v1/request.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/encoder_cache_manager.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/executor/abstract.py`
- `vllm/v1/worker/gpu_model_runner.py`

---

## 1. 一句话理解 Scheduler

`Scheduler` 是 vLLM V1 中负责“把请求转换为本轮模型执行计划”的组件。

它每一轮回答四个问题：

1. 哪些请求本轮应该执行？
2. 每个请求本轮需要计算多少 token？
3. 这些 token 需要哪些本地或外部 KV cache block？
4. Worker 执行结束后，如何更新请求状态并释放资源？

可以把它抽象成一个带 KV cache 账本的 token 调度器：

```text
请求状态
  -> token 预算
  -> 本地 prefix / 外部 KV 命中
  -> KV block 分配或抢占
  -> SchedulerOutput
  -> Worker / ModelRunner 执行
  -> ModelRunnerOutput
  -> 请求状态、KV block、输出结果更新
```

Scheduler 负责的是“计划、账本和状态推进”，不是模型计算本身。

---

## 2. Scheduler 在系统中的位置

### 2.1 上游：EngineCore

源码文件：`vllm/v1/engine/core.py`  
核心符号：`EngineCore.add_request()`、`EngineCore.step()`

`EngineCore` 把外部请求转换成内部 `Request`，交给 Scheduler；每轮执行时，先调用 Scheduler 生成计划，再把计划交给 executor 和 worker。

```text
LLMEngine / AsyncLLM
  -> InputProcessor
  -> EngineCoreRequest
  -> EngineCore
  -> Request
  -> Scheduler.add_request()
```

### 2.2 下游：Executor、Worker 和 ModelRunner

源码文件：`vllm/v1/executor/abstract.py`、`vllm/v1/worker/gpu_model_runner.py`  
核心符号：`Executor.execute_model()`、`GPUModelRunner.execute_model()`

Scheduler 不直接执行 `forward`。它生成的 `SchedulerOutput` 会被 executor 传给 worker，worker 再根据其中的 token 数、block table、请求列表和辅助元数据准备模型输入。

```text
SchedulerOutput
  -> Executor.execute_model()
  -> Worker / ModelRunner
  -> model forward / sampling / pooling
  -> ModelRunnerOutput
```

### 2.3 相关组件的职责边界

| 组件 | 主要职责 |
| --- | --- |
| `EngineCore` | 串起请求接收、调度、执行和输出回收的一轮闭环 |
| `Scheduler` | 选择请求、计算 token 预算、管理请求状态和调度结果 |
| `Request` | 保存单个请求的 token、状态、输出、KV 和多模态信息 |
| `KVCacheManager` | 将请求进度映射为 KV block，负责命中、分配、释放和缓存事件 |
| `BlockPool` | 管理 KV block 的可用列表、引用计数和缓存索引 |
| KV Connector | 查询或传输外部 KV，连接本地 block 与远端 KVPool |
| Encoder cache manager | 管理多模态或 encoder-decoder 的 encoder 输出缓存 |
| `SchedulerOutput` | 描述本轮“应该如何执行”的计划 |
| Worker / ModelRunner | 根据计划构造输入并执行模型 |
| `ModelRunnerOutput` | 返回 sampled token、pooling、KV transfer 等执行结果 |

---

## 3. 源码地图：先看哪些文件和符号

### 3.1 调度器本体

源码文件：`vllm/v1/core/sched/scheduler.py`  
核心符号：`Scheduler`

建议重点阅读这些符号：

- `Scheduler.__init__()`：创建队列、读取配置、初始化 KV 和 encoder 相关组件；
- `Scheduler.add_request()`：接收新请求或 streaming 后续 chunk；
- `Scheduler.schedule()`：生成一轮 `SchedulerOutput`；
- `Scheduler.update_from_output()`：消费 worker 返回结果并更新状态；
- `Scheduler.free_request()`：结束请求并处理资源释放；
- `_update_after_schedule()`：提交本轮调度后的请求进度和异步账本；
- `_connector_finished()`：处理请求结束时的 KV connector 收尾；
- `_drain_deferred_frees()`：回收因异步传输而延迟释放的 block。

### 3.2 请求对象

源码文件：`vllm/v1/request.py`  
核心符号：`Request`、`RequestStatus`

这里定义 Scheduler 操作的最小状态单元。阅读 Scheduler 时，应同时跟踪：

- `request_id`：请求身份；
- `status`：请求当前阶段；
- `prompt_token_ids`、`_all_token_ids`：输入和已生成 token；
- `num_tokens`、`num_prompt_tokens`：当前 token 长度和 prompt 长度；
- `num_computed_tokens`：Scheduler 认为已经推进的计算位置；
- `num_in_flight_tokens`：已经发出但尚未由 Scheduler 回收的计算量；
- `num_output_placeholders`：异步执行中尚未写回的输出位置；
- `spec_token_ids`：待验证的 draft token；
- `block_hashes`、KV transfer 参数和多模态特征。

### 3.3 调度接口和输出

源码文件：`vllm/v1/core/sched/interface.py`、`vllm/v1/core/sched/output.py`  
核心符号：`SchedulerInterface`、`SchedulerOutput`

`SchedulerInterface` 定义 EngineCore 依赖的调度边界；`SchedulerOutput` 是 Scheduler 和 Worker 之间的契约，也是 `update_from_output()` 回收结果时的对账依据。

### 3.4 KV cache 相关实现

源码文件：`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`  
核心符号：`KVCacheManager`、`BlockPool`

Scheduler 决定“本轮需要多少 KV 空间”，但 block 的具体账本由这些组件维护。不要把 Scheduler 的 `num_computed_tokens` 和 GPU 上已经写完的 KV tensor 混为一谈。

---

## 4. Scheduler 内部的四类请求容器

源码文件：`vllm/v1/core/sched/scheduler.py`  
核心字段：`requests`、`waiting`、`skipped_waiting`、`running`

### 4.1 `requests`

`requests` 是 request id 到 `Request` 的总表。请求从 waiting 或 running 队列移除后，只要仍有异步 KV、encoder 或输出收尾工作，通常仍可能保留在这里。

### 4.2 `waiting`

等待获得本轮执行资格的请求队列。新请求通常通过 `add_request()` 进入这里；队列顺序由 `SchedulingPolicy` 决定，常见策略是 FCFS 或 priority。

### 4.3 `skipped_waiting`

本轮暂时不能继续调度、但并未结束的请求队列。常见原因包括：

- 等待远端 KV load；
- 等待 structured output grammar；
- 等待下一段 streaming input；
- encoder 输入或 encoder cache 暂时不可用；
- LoRA、KV connector 或其他资源限制。

它不是失败队列，而是“暂时跳过，后续重新尝试”的状态。

### 4.4 `running`

已经进入模型执行流、后续调度通常优先推进的请求列表。running 不代表每轮一定执行：token budget、KV block、encoder cadence、PP/DP 约束和异步状态都可能使请求本轮被跳过或抢占。

---

## 5. 请求状态机

不同 vLLM 版本可能增加状态，但可以用下面的主线理解当前 V1 Scheduler：

```text
新请求
  -> WAITING
  -> RUNNING
  -> RUNNING（多轮 decode / chunked prefill）
  -> FINISHED_* 
  -> 资源释放完成
```

需要特别区分的旁路状态：

```text
WAITING
  -> WAITING_FOR_REMOTE_KVS
  -> skipped_waiting
  -> WAITING 或 RUNNING

RUNNING
  -> PREEMPTED
  -> waiting
  -> RUNNING
```

结束状态由停止 token、长度上限、abort、异常、structured output 结束或外部取消触发。状态结束后不一定立即删除请求，因为 KV save、KV send、encoder cache 和延迟 free 可能仍在进行。

### 5.1 四条容易混淆的进度线

| 字段 | 表示什么 | 不表示什么 |
| --- | --- | --- |
| `num_tokens` | 请求当前拥有的真实 token 数 | 不表示本轮要执行多少 token |
| `num_computed_tokens` | Scheduler 记账上的计算进度 | 不保证 worker 已经返回 |
| `num_in_flight_tokens` | 已经发出、尚未回收的计算量 | 不等于输出 token 数 |
| `num_output_placeholders` | 异步执行中等待返回的输出位置 | 不等于 draft token 数 |

Scheduler 计算本轮新增工作时，核心思想是：

```text
本轮需要调度的 token
  = 目标计算位置
  - Scheduler 已记账的计算位置
```

目标位置可能包含普通 token、speculative token 和异步 output placeholder，因此不能只看 `num_tokens`。

---

## 6. 一次请求的完整生命周期

### 6.1 请求进入：`add_request()`

源码文件：`vllm/v1/core/sched/scheduler.py`  
核心符号：`Scheduler.add_request()`、`Scheduler._enqueue_waiting_request()`

主要职责：

1. 判断 request id 是否已经存在；
2. 识别新请求和 streaming 后续 chunk；
3. 初始化或更新 request 状态；
4. 将请求放入 waiting 相关队列；
5. 通知 KV connector 有新请求到达。

```text
EngineCoreRequest
  -> Request.from_engine_core_request()
  -> Scheduler.add_request()
  -> requests[request_id]
  -> waiting / skipped_waiting
```

### 6.2 生成计划：`schedule()`

源码文件：`vllm/v1/core/sched/scheduler.py`  
核心符号：`Scheduler.schedule()`

一次 `schedule()` 可以按以下阶段理解：

```text
1. 计算本轮 token budget
2. 优先推进 running 请求
3. 在资源允许时接纳 waiting 请求
4. 查询本地 prefix cache 和外部 KV
5. 为请求分配 KV block
6. 必要时抢占 running 请求
7. 记录 running / new / resumed 请求
8. 构造 encoder、grammar、spec decode、KV connector 元数据
9. 生成 SchedulerOutput
```

### 6.3 Worker 执行

源码文件：`vllm/v1/executor/abstract.py`、`vllm/v1/worker/gpu_model_runner.py`  
核心符号：`Executor.execute_model()`、`GPUModelRunner.execute_model()`

Worker 根据 `SchedulerOutput`：

- 构造输入 token 和位置编码信息；
- 使用 block table 读取或写入 KV cache；
- 执行模型 forward；
- 执行 sampling、pooling 或 structured output 相关处理；
- 执行 KV connector 的 load/save；
- 返回 `ModelRunnerOutput`。

### 6.4 输出回收：`update_from_output()`

源码文件：`vllm/v1/core/sched/scheduler.py`  
核心符号：`Scheduler.update_from_output()`

回收阶段主要做以下事情：

1. 按 `SchedulerOutput` 对账本轮调度的请求；
2. 回收 in-flight token 和 output placeholder；
3. 写入 sampled token、logprobs、pooling 等结果；
4. 处理 speculative decoding 的接受、拒绝和回退；
5. 处理 grammar accept tokens 和 grammar bitmask；
6. 记录 KV/encoder transfer 的完成、失败或无效 block；
7. 判断 stop、abort、长度上限和异常；
8. 对未结束请求保留 running 状态，对结束请求调用资源释放逻辑。

### 6.5 请求结束：`free_request()`

源码文件：`vllm/v1/core/sched/scheduler.py`  
核心符号：`Scheduler.free_request()`、`Scheduler._connector_finished()`

结束请求时需要协调：

```text
Request finished
  -> connector.request_finished()
  -> KV save / send 收尾
  -> encoder cache 释放
  -> KV block 立即释放或延迟释放
  -> requests 总表删除
```

如果外部 KV 传输尚未结束，释放动作会被延迟；这就是“请求逻辑已结束”和“所有内存资源已可复用”之间可能存在时间差的原因。

---

## 7. `schedule()` 的核心算法

### 7.1 第一步：建立本轮预算

预算来自调度配置、模型上下文上限、当前 running 数量和并行/异步约束。常见限制包括：

- `max_num_scheduled_tokens`：本轮最多安排的 token 数；
- `max_num_running_reqs`：同时运行的请求数；
- `max_model_len`：单请求最大上下文长度；
- encoder compute budget；
- speculative decoding 的 lookahead token；
- pipeline parallel 或 async scheduling 的额外占位。

预算不是简单的请求数限制。一个请求可能消耗多个 token 预算，也可能因为 prefix cache 命中而减少本轮真正需要计算的 token。

### 7.2 第二步：优先推进 running

先处理 running 的原因是避免已经占用 GPU/KV 资源的请求长期停顿。对每个 running 请求，Scheduler 通常会：

```text
读取请求进度
  -> 计算目标位置
  -> 得到 num_new_tokens
  -> 检查 encoder / Mamba / PP / DP 约束
  -> 分配新 KV block
  -> 记录 scheduled_running_reqs
```

如果 KV block 不足，Scheduler 可能从 running 中选择请求进行 preemption，把它退回 waiting，并释放可以回收的 block。

### 7.3 第三步：接纳 waiting

waiting 请求只有在 token budget、running request 数、LoRA、encoder 和 KV block 都允许时才会进入执行计划。

对 waiting 请求的检查顺序可以理解为：

```text
选择队首请求
  -> 检查请求是否可运行
  -> 查询本地 prefix cache
  -> 查询外部 KV connector
  -> 计算真正需要的新 token
  -> 分配 KV block
  -> 构造 new / resumed request 记录
  -> 加入 running
```

遇到暂时不能运行的请求，不一定是错误：它可能进入 `skipped_waiting`，等待下一轮重试。

### 7.4 本地 prefix cache 和外部 KV

源码文件：`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/sched/scheduler.py`  
核心符号：`KVCacheManager.get_computed_blocks()`、`KVCacheManager.allocate_slots()`、KV connector 的 matched-token 查询接口

命中处理的语义是：

```text
prompt / prefix token
  -> 计算 block hash
  -> 查询本地 prefix cache
  -> 如有 connector，再查询外部 KV
  -> 把命中 token 转成可使用的本地 block
  -> 只为未命中部分安排 forward
```

外部 KV 查询可能是同步的，也可能是异步的：

```text
同步命中
  -> 分配本地 block
  -> 直接进入本轮执行

异步命中
  -> 分配暂存 block
  -> WAITING_FOR_REMOTE_KVS
  -> Worker 完成 load
  -> update_from_output() 记录 finished_recving
  -> 后续 schedule() 重新提升请求
```

### 7.5 KV block 分配和抢占

源码文件：`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`  
核心符号：`KVCacheManager.allocate_slots()`、`KVCacheManager.free()`、`BlockPool`

每个请求的 KV 需求可能包括：

- 本地 prefix 命中的 block；
- 外部 KV 命中后需要承载 load 的 block；
- 本轮新 token 对应的 block；
- speculative decoding 的 lookahead block。

分配失败时，处理顺序通常是：

```text
尝试为 waiting 请求分配
  -> 不足则停止接纳更多 waiting

running 请求继续推进但 block 不足
  -> 选择请求 preempt
  -> 释放可回收 block
  -> 请求回到 waiting / PREEMPTED
```

抢占的目标不是丢弃请求，而是用释放 KV 空间换取其他请求继续执行；被抢占请求之后需要重新建立可执行的 block 计划。

### 7.6 生成 `SchedulerOutput`

源码文件：`vllm/v1/core/sched/output.py`  
核心符号：`SchedulerOutput` 及其请求级调度记录

`SchedulerOutput` 是本轮执行计划，通常需要表达：

- scheduled running requests；
- scheduled new requests；
- scheduled resumed requests；
- 每个请求的 `num_scheduled_tokens`；
- block table 和新分配的 block；
- encoder 输入和 encoder cache 信息；
- speculative decoding 相关 token；
- structured output grammar bitmask；
- KV connector metadata；
- block copy、finished request 和其他收尾信息。

它不是最终用户输出，而是 Worker 执行和 Scheduler 回收共同使用的“本轮对账单”。

---

## 8. `update_from_output()` 的回收语义

### 8.1 普通 sampled token

Worker 返回 sampled token 后，Scheduler 将 token 写入请求状态，更新 token 计数，并检查停止条件。未停止的请求留在 running，下一轮继续 decode。

### 8.2 speculative decoding

源码文件：`vllm/v1/core/sched/scheduler.py`、`vllm/v1/spec_decode/`  
核心字段：`spec_token_ids`、`num_tokens_with_spec`、`num_output_placeholders`

一次 spec decode 需要区分：

```text
draft token
  -> target model 验证
  -> accepted draft tokens
  -> rejected draft tokens
  -> replacement / target-sampled token
  -> 更新真实 token、计算进度和下一轮计划
```

`num_tokens_with_spec` 表示目标模型需要覆盖的计算范围；它不等于已经接受的真实 token 数。拒绝 draft 后，Scheduler 可能回退计算进度和 KV block，再把替代 token 纳入后续状态。

### 8.3 structured output

源码文件：`vllm/v1/structured_output/`、`vllm/v1/core/sched/scheduler.py`  
核心符号：grammar manager、`get_grammar_bitmask()`、grammar accept 处理逻辑

Scheduler 需要在模型执行前提供约束 mask，在输出回收后把 sampled token 反馈给 grammar 状态机。grammar 尚未准备好时，请求可以暂时停留在 waiting 体系中。

### 8.4 KV / encoder transfer

Worker 返回的 `ModelRunnerOutput` 可能包含：

- `finished_recving`：外部 KV load 已完成；
- `finished_sending`：外部 KV save/send 已完成；
- `invalid_block_ids`：部分 block 无效，需要重算或失败；
- KV/encoder transfer 参数。

Scheduler 必须先消费这些状态，再决定请求是否可以继续运行、哪些 block 可以复用、哪些 block 必须延迟释放。

### 8.5 stop、abort 和资源释放

结束判断和资源释放是两个相连但不完全同步的动作：

```text
检测 stop / abort
  -> 生成最终 EngineCoreOutput
  -> 通知 connector
  -> 释放 encoder cache
  -> 立即或延迟释放 KV block
  -> 从请求总表删除
```

---

## 9. 多种能力如何嵌入调度

### 9.1 Chunked prefill

长 prompt 不必一次性全部执行。Scheduler 将 prompt 拆成多个可调度片段，让 prefill 与其他请求的 decode 共享 token budget。

### 9.2 Multi-modal / encoder 输入

源码文件：`vllm/v1/core/encoder_cache_manager.py`、`vllm/multimodal/`  
核心概念：encoder compute budget、scheduled encoder inputs、encoder cache

Scheduler 需要同时管理文本 token 和 encoder 计算预算，避免同一个图像或音频特征被重复计算。

### 9.3 LoRA

LoRA 会影响同一批次可容纳的 adapter 数和请求组合。Scheduler 在接纳 waiting 请求时，需要考虑当前 batch 的 LoRA 资源约束。

### 9.4 Mamba / 状态空间模型

Mamba 类模型除了 KV block 外，还可能要求请求按特定状态边界推进。相关逻辑会改变 running 请求的可调度 token 数和 block 对齐方式。

### 9.5 Pipeline / data parallel 和异步调度

PP cadence、DP prefill balancing 和 async scheduling 会引入额外的 in-flight 计算或输出占位。此时 Scheduler 的逻辑进度可能先于 Worker 的实际返回，必须依赖 `num_in_flight_tokens`、`num_output_placeholders` 和回收对账维持一致性。

---

## 10. 最重要的不变量

阅读或修改 Scheduler 时，可以用下面的不变量检查行为是否合理：

1. 一个 request id 在 `requests` 中最多对应一个 `Request` 对象；
2. waiting、skipped_waiting、running 之间的归属必须明确，不能同时出现在互斥队列中；
3. `num_scheduled_tokens` 不能超过本轮 token budget；
4. `num_computed_tokens` 的推进必须与本轮调度记录一致；
5. Worker 回收必须能用 `SchedulerOutput` 对上本轮请求和 token 数；
6. 被抢占请求不能丢失真实 token 和必要的 KV 状态；
7. 远端 KV 未完成接收时，相关 block 不能被当作普通可复用 block；
8. 延迟释放的 block 必须等到 KV/encoder consumer 完成后才能回到 free list；
9. 请求结束后，encoder cache、KV block、connector 状态和请求总表最终都要收敛；
10. spec decode 的 draft token、已接受 token、替代 token 和真实 token 计数不能混用。

---

## 11. 推荐阅读顺序

### 11.1 先建立主流程

```text
scheduler_overview.md
  -> vllm_scheduler.md
  -> 01_request_states.md
  -> 02_token_budget.md
  -> 03_running_decode_prefill.md
  -> 04_waiting_to_running.md
  -> 08_update_after_worker_output.md
```

### 11.2 再看 KV cache 和抢占

```text
01_request_states.md
  -> 05_prefix_and_external_kv_hits.md
  -> 06_kv_block_allocation_and_preemption.md
  -> ../kv_cache_transfer/04_scheduler_kv_connector_flow.md
```

### 11.3 最后看复杂特性

```text
07_auxiliary_scheduling_features.md
  -> token_states.md
  -> spec_decode_token_states.md
  -> ../spec_decode/03_scheduler_spec_decode_flow.md
```

### 11.4 与 EngineCore 对照阅读

```text
../engine_core/engine_core_overview.md
  -> EngineCore.step()
  -> Scheduler.schedule()
  -> Executor.execute_model()
  -> Scheduler.update_from_output()
```

---

## 12. 如何定位源码而不依赖行号

当代码发生插入、删除或重排时，按下面的顺序定位：

1. 先打开源码文件路径；
2. 搜索类名或方法名，例如 `class Scheduler`、`def schedule`；
3. 再搜索字段名，例如 `num_computed_tokens`、`skipped_waiting`；
4. 最后沿调用关系确认上下游，例如 `Scheduler.schedule()` 的返回值如何进入 `Executor.execute_model()`。

文档中推荐使用这种格式：

```text
源码文件：`vllm/v1/core/sched/scheduler.py`
核心符号：`Scheduler.schedule()`

职责：为本轮运行请求和等待请求计算 token 预算，分配 KV block，
并生成交给 Worker 的 `SchedulerOutput`。
```

如果必须复现某个历史版本，再额外记录源码提交号；行号只作为临时阅读提示，不作为文档的稳定引用。

---

## 13. 总结

Scheduler 的核心不是一个简单的 FIFO 队列，而是一个同时协调以下状态的闭环：

```text
请求队列
  + token 计算进度
  + 本地 prefix cache
  + 外部 KV transfer
  + KV block 生命周期
  + encoder / multimodal 预算
  + speculative decoding
  + Worker in-flight 输出
  + stop / abort / resource free
```

最值得记住的主链路是：

```text
add_request()
  -> waiting
  -> schedule()
  -> prefix / external KV lookup
  -> allocate_slots() 或 preempt
  -> SchedulerOutput
  -> Worker / ModelRunner
  -> ModelRunnerOutput
  -> update_from_output()
  -> stop / continue / retry / free_request()
```

只要沿着这条链路，再用 `Request`、`SchedulerOutput` 和 KV block 这三个对象对账，就能理解大多数 Scheduler 行为，而不需要依赖会漂移的行号。
