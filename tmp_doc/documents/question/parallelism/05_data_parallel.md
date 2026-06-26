# 05. Data Parallel 如何做请求级并行？

源码位置：

- `vllm/vllm/config/parallel.py`
- `vllm/vllm/distributed/parallel_state.py`
- `vllm/vllm/v1/engine/async_llm.py`
- `vllm/vllm/v1/engine/input_processor.py`
- `vllm/vllm/v1/engine/core_client.py`
- `vllm/vllm/v1/engine/core.py`
- `vllm/vllm/v1/engine/coordinator.py`
- `vllm/vllm/v1/engine/utils.py`
- `vllm/vllm/v1/executor/`
- `vllm/vllm/v1/worker/dp_utils.py`
- `vllm/vllm/v1/worker/gpu_worker.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`

本问题关注：Data Parallel 在 vLLM V1 中如何把请求分给不同 DP replica；每个 DP replica 内部如何继续使用 TP / PP / PCP / DCP / EP；EngineCore、Executor、Worker、DP coordinator 如何协同；以及 DP 下 SchedulerOutput、KV cache、batch padding、cudagraph、load balancing、MoE wave coordination 分别在哪里处理。

---

## 1. 一句话回答

Data Parallel 是请求级并行：

```text
多个模型 replica 同时存在；
不同请求 / batch 被分发到不同 replica；
每个 replica 内部可以再使用 TP / PP / EP / CP 完成单个请求的模型计算。
```

在 vLLM V1 里可以把 DP 分成两层理解：

```text
请求路由层：
  前端 / EngineCoreClient 选择某个 DP engine，把请求发给它。

模型执行层：
  每个 DP engine 内部有自己的 Scheduler、Executor、Worker、KV cache、InputBatch。
```

一句话记忆：

```text
DP 是“不同请求给不同模型副本跑”；TP / PP / CP 是“一个副本内部怎么一起跑一次 forward”。
```

---

## 2. 本文要回答的问题

```text
DP replica 如何定义？
DP 和 TP / PP / EP / CP 的边界是什么？
Scheduler 是全局调度还是每个 DP rank 独立调度？
请求如何被路由到某个 DP rank？
internal LB / external LB / hybrid LB 有什么区别？
MoE DP 为什么还需要 wave coordination 和 dummy batch？
DP 下 KV cache 是否 replica-local？
SchedulerOutput / ModelRunnerOutput 在 DP 下如何归属？
GPUModelRunner 为什么还要 coordinate_batch_across_dp？
```

---

## 3. 最小主链路

DP 请求执行链路可以压缩成：

```text
AsyncLLM.add_request(..., data_parallel_rank=None/指定值)
  → InputProcessor.process_inputs()
      → EngineCoreRequest(data_parallel_rank=...)
  → EngineCoreClient
      → DPAsyncMPClient / DPLBAsyncMPClient
      → 选择目标 DP engine
  → 目标 EngineCoreProc / DPEngineCoreProc
      → 本 replica 的 Scheduler.schedule()
      → 本 replica 的 Executor.execute_model(SchedulerOutput)
      → 本 replica 的 Worker / GPUModelRunner
      → 本 replica 的 KV cache / InputBatch / attention metadata
      → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutputs
  → OutputProcessor
  → 用户输出
```

这里最重要的是：

```text
SchedulerOutput 是 replica-local 的；
KV cache 是 replica-local 的；
InputBatch 是 replica-local 的；
ModelRunnerOutput 也回到生成它的那个 EngineCore。
```

DP 不是把一个 SchedulerOutput 切成多份给多个 replica，而是每个 DP replica 自己有独立调度器。

---

## 4. DP 和其他并行的边界

### 4.1 TP / PP / CP 是副本内部并行

`ParallelConfig.world_size` 的定义是：

位置：`vllm/vllm/config/parallel.py:791`

```text
world_size = pipeline_parallel_size
           * tensor_parallel_size
           * prefill_context_parallel_size
```

也就是说，`world_size` 先描述单个 DP replica 内部需要多少 worker rank。

### 4.2 DP 把这个副本复制多份

`world_size_across_dp` 才把 DP 算进去：

位置：`vllm/vllm/config/parallel.py:516`

```text
world_size_across_dp = world_size * data_parallel_size
```

因此：

```text
一个 DP replica 内部：TP × PP × PCP
所有 replica 加起来：DP × TP × PP × PCP
```

如果一个请求进入某个 DP replica，那么这个请求的一次 forward 仍然可能由该 replica 内部的 TP / PP / CP ranks 合作完成。

### 4.3 parallel_state 中的 rank 布局

`initialize_model_parallel()` 的注释给了全局 rank 维度顺序：

位置：`vllm/vllm/distributed/parallel_state.py:1760`

```text
ExternalDP x DP x PP x PCP x TP
```

它构造的主要 group 是：

| group | 维度含义 |
|---|---|
| TP | tensor parallel group |
| PP | pipeline parallel group |
| PCP | prefill context parallel group |
| DCP | decode context parallel group，复用 TP 维度 |
| DP | data parallel group |
| EP | expert parallel group，MoE 时通常会组合 DP / TP / PCP |
| EPLB | expert parallel load balancing 专用 group |

DP group 的构造在：

位置：`vllm/vllm/distributed/parallel_state.py:1853`

```text
all_ranks.transpose(1, 4)
  → reshape(-1, data_parallel_size)
  → 每个 group 横跨 DP 维度
```

这表示：

```text
同一个 TP/PP/PCP 坐标上的不同 DP rank 会组成一个 DP group。
```

### 4.4 EP 和 DP 的关系

`ParallelConfig.data_parallel_size` 的字段说明里明确提到：

位置：`vllm/vllm/config/parallel.py:126`

```text
MoE layers will be sharded according to the product of
the tensor parallel size and data parallel size.
```

所以对 MoE 模型，DP 不只是“多份完全独立副本”这么简单：

```text
MoE expert parallel 可能把 expert 维度和 DP 维度组合起来，
这会带来跨 DP rank 的 collective 约束。
```

这也是为什么 MoE DP 要启用 `DPEngineCoreProc` 和 wave coordination。

---

## 5. DP 相关配置

核心配置集中在 `ParallelConfig`。

位置：`vllm/vllm/config/parallel.py:117`

### 5.1 data_parallel_size

```text
data_parallel_size：DP replica 总数。
```

如果是 `data_parallel_size = 4`，可以理解为系统中有 4 个 DP engine，每个 engine 有自己的 Scheduler / KV cache / Worker 组。

### 5.2 data_parallel_size_local

```text
data_parallel_size_local：当前进程 / 当前节点本地管理的 DP engine 数。
```

位置：`vllm/vllm/config/parallel.py:129`

它还有一个特殊值：

```text
0 表示 engine-args 层用它作为 sentinel，说明 DP 是外部指定的。
```

### 5.3 data_parallel_rank / data_parallel_rank_local

```text
data_parallel_rank：当前 DP rank 的全局编号。
data_parallel_rank_local：当前 DP rank 在本地进程 / 节点内的编号。
```

位置：`vllm/vllm/config/parallel.py:133`

`data_parallel_rank` 会影响：

```text
- EngineCore identity；
- DP process group 初始化；
- GPU local rank 偏移；
- ZMQ handshake；
- 请求路由；
- MoE DP wave 协调。
```

### 5.4 data_parallel_backend

```text
data_parallel_backend = "mp" | "ray"
```

位置：`vllm/vllm/config/parallel.py:144`

它控制 DP EngineCore 的启动方式：

```text
mp：本地 / 多进程启动 CoreEngineProcManager；
ray：通过 CoreEngineActorManager 启动 Ray actor。
```

注意它和 `distributed_executor_backend` 不是一个概念：

```text
data_parallel_backend：DP engine 怎么启动；
distributed_executor_backend：单个 engine 内部 worker/executor 怎么分布式执行。
```

### 5.5 data_parallel_external_lb

```text
data_parallel_external_lb：外部负载均衡模式。
```

位置：`vllm/vllm/config/parallel.py:146`

典型语义是：

```text
每个 vLLM 实例 / pod 只管理本地 DP rank；
跨 rank 的请求分流由外部 LB 做。
```

在这种模式下，vLLM 内部不会在所有 DP rank 之间做完整 internal LB。

### 5.6 data_parallel_hybrid_lb

```text
data_parallel_hybrid_lb：混合负载均衡模式。
```

位置：`vllm/vllm/config/parallel.py:153`

语义是：

```text
节点内：vLLM 自己在本地 DP ranks 之间做 LB；
节点间：外部 LB 负责分流。
```

### 5.7 local_engines_only

`local_engines_only` 是一个派生属性：

位置：`vllm/vllm/config/parallel.py:531`

```text
local_engines_only = data_parallel_external_lb or data_parallel_hybrid_lb
```

它决定 client 管理范围：

```text
纯 internal LB：client 管理 local + remote EngineCores；
external / hybrid LB：client 只管理本地 EngineCores。
```

### 5.8 offline SPMD 环境变量路径

如果不是通过 engine args 显式指定 DP，`__post_init__()` 会从环境变量读取：

位置：`vllm/vllm/config/parallel.py:849`

```text
VLLM_DP_SIZE
VLLM_DP_RANK
VLLM_DP_RANK_LOCAL
VLLM_DP_MASTER_IP
VLLM_DP_MASTER_PORT
```

并且源码里限制：

```text
Offline data parallel mode is not supported/useful for dense models.
```

位置：`vllm/vllm/config/parallel.py:857`

---

## 6. EngineCore 是如何按 DP 拆分的

### 6.1 每个 DP rank 对应一个 EngineCore 进程 / actor

`launch_core_engines()` 根据 DP 配置启动 engine。

位置：`vllm/vllm/v1/engine/utils.py:1072`

它会读取：

```text
dp_size = data_parallel_size
local_engine_count = data_parallel_size_local
local_start_index = data_parallel_rank_local
dp_rank = data_parallel_rank
local_engines_only = external/hybrid LB
```

然后决定：

```text
- 是否启动 DPCoordinator；
- 是否用 Ray actor；
- 当前前端需要和哪些 EngineCore handshake；
- 本地启动哪些 EngineCoreProc。
```

### 6.2 CoreEngineProcManager 启动本地 DP engines

`CoreEngineProcManager` 会为每个本地 DP rank 启动一个子进程。

位置：`vllm/vllm/v1/engine/utils.py:120`

核心启动参数是：

```text
EngineCoreProc.run_engine_core(
    dp_rank=global_index,
    local_dp_rank=local_index,
)
```

位置：`vllm/vllm/v1/engine/utils.py:164`

这意味着：

```text
DP rank 是 EngineCore 级别的身份；
EngineCore 内部再创建自己的 Executor / Worker。
```

### 6.3 EngineCoreProc.run_engine_core 如何选择普通 EngineCore 还是 DPEngineCoreProc

位置：`vllm/vllm/v1/engine/core.py:1152`

核心逻辑是：

```text
data_parallel = data_parallel_size > 1 or dp_rank > 0

if data_parallel and model is MoE:
    parallel_config.data_parallel_rank = dp_rank
    engine_core = DPEngineCoreProc(...)
else:
    # Non-MoE DP ranks are completely independent
    parallel_config.data_parallel_size = 1
    parallel_config.data_parallel_rank = 0
    engine_core = EngineCoreProc(..., engine_index=dp_rank)
```

位置：`vllm/vllm/v1/engine/core.py:1186`

这点非常关键。

对 dense 模型来说，DP 更像：

```text
多个独立 EngineCoreProc + 前端请求分流。
```

对 MoE 模型来说，DP 更像：

```text
多个 DPEngineCoreProc + DP process group + wave coordination + dummy step。
```

原因是 MoE / EP 可能存在跨 DP rank 的 collective，不能让某些 DP rank 独自停下来。

---

## 7. 请求如何路由到 DP rank

### 7.1 AsyncLLM.add_request 支持显式 data_parallel_rank

`AsyncLLM.add_request()` 有参数：

位置：`vllm/vllm/v1/engine/async_llm.py:280`

```text
data_parallel_rank: int | None = None
```

它会继续传给 `InputProcessor.process_inputs()`。

位置：`vllm/vllm/v1/engine/async_llm.py:349`

### 7.2 EngineCoreRequest 会保存 data_parallel_rank

`InputProcessor.process_inputs()` 会校验这个 rank 是否在可管理范围内。

位置：`vllm/vllm/v1/engine/input_processor.py:259`

```text
num_ranks = data_parallel_size_local if local_engines_only else data_parallel_size

if data_parallel_rank is not None and not 0 <= data_parallel_rank < num_ranks:
    raise ValueError
```

然后写入请求：

位置：`vllm/vllm/v1/engine/input_processor.py:370`

```text
EngineCoreRequest(..., data_parallel_rank=data_parallel_rank)
```

所以：

```text
请求可以显式指定要去哪个 DP rank；
如果不指定，则由 vLLM 的 client LB 选择。
```

### 7.3 make_async_mp_client 如何选择 DP client

`EngineCoreClient.make_async_mp_client()` 会根据 DP 配置选择 client。

位置：`vllm/vllm/v1/engine/core_client.py:126`

```text
if data_parallel_size > 1:
    if data_parallel_external_lb:
        return DPAsyncMPClient(...)
    return DPLBAsyncMPClient(...)
```

含义是：

| 模式 | client 类型 | 路由方式 |
|---|---|---|
| external LB | `DPAsyncMPClient` | 当前 client 默认只面向外部 LB 分到的本地 rank |
| internal / hybrid LB | `DPLBAsyncMPClient` | vLLM 根据各 engine 负载选择 rank |
| DP=1 | `AsyncMPClient` | 单 engine |

### 7.4 DPLBAsyncMPClient 的内部负载均衡策略

`DPLBAsyncMPClient.get_core_engine_for_request()` 会先看请求是否指定 `data_parallel_rank`。

位置：`vllm/vllm/v1/engine/core_client.py:1413`

```text
if request.data_parallel_rank is not None:
    使用指定 engine
else:
    根据 lb_engines 选择 score 最小的 engine
```

评分公式是：

```text
score = waiting * 4 + running
```

位置：`vllm/vllm/v1/engine/core_client.py:1429`

选择后还会本地把 waiting count 增加，减少 stats 更新间隔内的倾斜：

```text
current_counts[eng_index][0] += client_count
```

位置：`vllm/vllm/v1/engine/core_client.py:1436`

### 7.5 请求发送前会带 current_wave

DP client 在发送请求前会写入：

位置：`vllm/vllm/v1/engine/core_client.py:1359`

```text
request.current_wave = self.current_wave
request.client_index = self.client_index
chosen_engine = self.get_core_engine_for_request(request)
```

如果 engines 当前处于 paused 状态，还会通知 coordinator：

```text
("FIRST_REQ", chosen_engine)
```

位置：`vllm/vllm/v1/engine/core_client.py:1367`

这部分只在 MoE / coordinated DP 场景里特别重要。

---

## 8. Scheduler 是全局调度还是 replica-local 调度

结论：

```text
Scheduler 是每个 EngineCore replica-local 的。
```

`EngineCore.__init__()` 中每个 EngineCore 都会创建自己的：

```text
self.model_executor = executor_class(vllm_config)
kv_cache_config = self._initialize_kv_caches(vllm_config)
self.scheduler = Scheduler(...)
```

位置：`vllm/vllm/v1/engine/core.py:123`

这意味着每个 DP rank 有自己的：

```text
- Scheduler
- KV cache manager
- block pool
- prefix cache 状态
- request table
- running / waiting queues
```

`DPCoordinator` 不做真正调度。它只维护：

```text
- 每个 engine 的 waiting / running 数；
- 当前 request wave；
- engines 是否 running；
- START_DP_WAVE 广播。
```

位置：`vllm/vllm/v1/engine/coordinator.py:23`

所以不能把 DP 理解成“一个全局 Scheduler 把一个 batch 切给多个 replica”。更准确是：

```text
前端把请求分给某个 EngineCore；
该 EngineCore 自己调度自己的请求。
```

---

## 9. SchedulerOutput / ModelRunnerOutput 在 DP 下的归属

### 9.1 SchedulerOutput 是本 DP rank 的本轮计划

`EngineCore.step()` 的主链路没有因为 DP 改变基本结构：

位置：`vllm/vllm/v1/engine/core.py:479`

```text
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

这个 `scheduler_output` 只包含当前 EngineCore 所拥有请求的执行计划。

### 9.2 Executor 不负责跨 DP 分发请求

`Executor.execute_model()` 仍然是把一个 `SchedulerOutput` 发给该 EngineCore 内部的 worker ranks。

也就是说：

```text
DP 请求分发发生在 EngineCoreClient → EngineCore 之前；
Executor 处理的是一个 replica 内部的 TP / PP worker 分发。
```

### 9.3 ModelRunnerOutput 回到对应 Scheduler

`GPUModelRunner` 返回的 `ModelRunnerOutput` 会回到同一个 EngineCore 的 Scheduler：

```text
本 replica SchedulerOutput
  → 本 replica Executor / Worker / ModelRunner
  → 本 replica ModelRunnerOutput
  → 本 replica Scheduler.update_from_output()
```

不会出现一个 DP rank 的 `ModelRunnerOutput` 被另一个 DP rank 的 Scheduler 消费。

---

## 10. DPCoordinator 做什么

### 10.1 coordinator 的职责

`DPCoordinator` 的 docstring 已经说明了三个职责。

位置：`vllm/vllm/v1/engine/coordinator.py:23`

```text
1. 收集每个 DP engine 的 waiting / running queue lengths；
2. 发布这些 stats 给 frontends，用于 load balancing；
3. 维护 DP request wave 和 engines running 状态；
4. 广播 START_DP_WAVE，让 paused engines 重新进入 running 状态。
```

### 10.2 coordinator 何时启动

`launch_core_engines()` 中：

位置：`vllm/vllm/v1/engine/utils.py:1106`

```text
run_coordinator = (
    vllm_config.needs_dp_coordinator
    and not offline_mode
    and dp_rank == 0
)
```

启动后会把 coordinator 的 socket 地址写回：

```text
addresses.coordinator_input
addresses.coordinator_output
addresses.frontend_stats_publish_address
```

位置：`vllm/vllm/v1/engine/utils.py:1114`

### 10.3 internal / hybrid LB 才发布 request stats

`EngineCoreProc.__init__()` 里：

位置：`vllm/vllm/v1/engine/core.py:947`

```text
internal_dp_balancing = has_coordinator and not data_parallel_external_lb
publish_dp_lb_stats = internal_dp_balancing
```

也就是说：

```text
internal LB：发布 stats；
hybrid LB：发布本地相关 stats；
external LB：不依赖 vLLM 内部 stats 做跨 rank 分流。
```

### 10.4 coordinator 如何发布 stats

`DPEngineCoreProc._maybe_publish_request_counts()` 会从 Scheduler 读取：

位置：`vllm/vllm/v1/engine/core.py:1901`

```text
counts = self.scheduler.get_request_counts()
stats = SchedulerStats(
    num_waiting_reqs,
    num_running_reqs,
    step_counter,
    current_wave,
)
```

然后发给 coordinator。

Coordinator 收到后更新对应 engine 的：

```text
request_counts = [waiting, running]
```

位置：`vllm/vllm/v1/engine/coordinator.py:371`

再定期发布给前端：

```text
(engine_req_counts_list, current_wave, engines_running)
```

位置：`vllm/vllm/v1/engine/coordinator.py:276`

---

## 11. MoE DP 的 wave coordination

### 11.1 为什么 MoE DP 需要 wave

对 dense 模型，每个 DP engine 可以基本独立地跑。

但对 MoE / expert parallel，DP 维度可能参与 expert sharding 和 all-to-all collective。源码里也写明：

位置：`vllm/vllm/v1/engine/core.py:1191`

```text
Non-MoE DP ranks are completely independent, so treat like DP=1.
```

反过来说明：

```text
MoE DP ranks 不能完全独立停止 / 前进。
```

如果某些 DP rank 没请求、另一些 DP rank 正在跑 MoE collective，就可能出现 collective 不匹配或 hang。因此 vLLM 用 wave 协调所有 DP ranks 的 running / paused 状态。

### 11.2 DPEngineCoreProc 初始化 DP group

`DPEngineCoreProc._init_data_parallel()` 会创建 stateless DP process group：

位置：`vllm/vllm/v1/engine/core.py:1795`

```text
dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
self.dp_group = dp_group
self.dp_store = dp_store
```

`ParallelConfig.stateless_init_dp_group()` 使用 gloo，因为 engine process 可能没有 CUDA device。

位置：`vllm/vllm/config/parallel.py:590`

### 11.3 run_busy_loop 的 DP 特殊逻辑

普通 `EngineCoreProc.run_busy_loop()` 做的是：

```text
process input queue
process engine step
```

`DPEngineCoreProc.run_busy_loop()` 多了几件事。

位置：`vllm/vllm/v1/engine/core.py:1923`

核心流程：

```text
1. 处理输入队列；
2. 发布 request counts；
3. 执行本地 engine step；
4. 再发布 request counts；
5. 如果本 rank 没执行模型但全局仍在 running，执行 dummy batch；
6. 周期性 all-reduce 判断所有 DP rank 是否都结束；
7. 如果全局结束，发送 wave_complete 并进入 paused。
```

### 11.4 dummy batch 的作用

如果某个 rank 本轮没有 ready request，但其他 DP rank 还在跑，源码会执行：

位置：`vllm/vllm/v1/engine/core.py:1950`

```text
self.execute_dummy_batch()
```

语义是：

```text
让空闲 DP rank 也进入必要的模型执行 / collective 节奏，
避免 MoE / EP 的跨 rank collective 不匹配。
```

### 11.5 全局 unfinished 状态同步

`DPEngineCoreProc._has_global_unfinished_reqs()` 每 32 step 做一次 DP all-reduce。

位置：`vllm/vllm/v1/engine/core.py:1982`

底层调用：

```text
ParallelConfig.sync_dp_state(dp_group, has_unfinished, pending_pause)
```

位置：`vllm/vllm/config/parallel.py:699`

`sync_dp_state()` 用一个 2 元 tensor 做 SUM all-reduce：

```text
[0] = 本 rank 是否还有 unfinished work
[1] = 本 rank 是否 pending_pause
```

得到：

```text
has_unfinished_global = 任意 rank 还有工作，或 pause 还没达成共识
pause_consensus = 所有 rank 都 pending_pause
```

### 11.6 START_DP_WAVE 如何唤醒 engines

如果 engines 已经 paused，而某个 frontend 发送第一个新请求，`DPAsyncMPClient` 会发：

```text
("FIRST_REQ", chosen_engine)
```

位置：`vllm/vllm/v1/engine/core_client.py:1367`

Coordinator 收到后广播：

```text
EngineCoreRequestType.START_DP_WAVE
```

位置：`vllm/vllm/v1/engine/coordinator.py:444`

`DPEngineCoreProc._handle_client_request()` 接到后设置：

```text
self.current_wave = new_wave
self.engines_running = True
```

位置：`vllm/vllm/v1/engine/core.py:1881`

这就是 DP wave 的完整闭环。

---

## 12. Worker / GPUModelRunner 中的 DP 同步

### 12.1 EngineCore wave 同步不等于 batch padding 同步

MoE DP 的 wave coordination 解决的是：

```text
不同 DP rank 是否一起处于 running 状态。
```

但模型执行时还需要解决另一个问题：

```text
不同 DP rank 本轮 token 数不同，cudagraph / ubatching / DBO 如何保持一致？
```

这由 `coordinate_batch_across_dp()` 处理。

### 12.2 coordinate_batch_across_dp 的入口

`GPUModelRunner._determine_batch_execution_and_padding()` 中：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3882`

```text
if data_parallel_size > 1:
    should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
        coordinate_batch_across_dp(...)
    )
```

它发生在：

```text
本地 batch token 数已知；
本地 cudagraph dispatch 已经初步决定；
真正构建 attention metadata 和 forward 前。
```

### 12.3 all-reduce 交换哪些信息

`dp_utils._run_ar()` 构造一个形状为 `[4, dp_size]` 的 tensor。

位置：`vllm/vllm/v1/worker/dp_utils.py:36`

每个 rank 写入自己的列：

```text
row 0：orig_num_tokens_per_ubatch
row 1：padded_num_tokens_per_ubatch
row 2：是否想 ubatch
row 3：cudagraph_mode
```

然后对 DP group 做 all-reduce。

### 12.4 同步后的决策

`coordinate_batch_across_dp()` 返回：

位置：`vllm/vllm/v1/worker/dp_utils.py:164`

```text
should_ubatch
num_tokens_after_padding
synced_cudagraph_mode
```

它的规则是：

```text
1. ubatching 必须所有 DP ranks 都同意；
2. cudagraph_mode 取所有 rank 的最小值；
3. 如果 synced_cudagraph_mode != NONE 或 should_ubatch，
   所有 DP ranks pad 到相同 token 数；
4. 否则保留各 rank 自己的 token 数。
```

位置：`vllm/vllm/v1/worker/dp_utils.py:139`

### 12.5 num_tokens_across_dp 是什么

`num_tokens_across_dp` 是一个长度为 `dp_size` 的 CPU tensor：

```text
[num_tokens_rank0, num_tokens_rank1, ..., num_tokens_rankN]
```

如果需要 DP padding，它里面每个值都是全局最大 token 数。

`GPUModelRunner` 会取当前 rank 对应的值：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:3897`

```text
dp_rank = self.parallel_config.data_parallel_rank
num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
```

然后重新 dispatch cudagraph，确保 batch descriptor 和 DP 同步后的 token 数一致。

### 12.6 为什么 forward context 还要带 num_tokens_across_dp

`set_forward_context()` 会收到：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4306`

```text
num_tokens=num_tokens_padded
num_tokens_across_dp=num_tokens_across_dp
```

这个字段给后续 compiled / ubatch / DP-aware 路径使用，确保不同 DP rank 对 batch 形状和 token 对齐有一致视图。

### 12.7 0-token 也可能需要 dummy run

`GPUModelRunner.execute_model()` 中，如果本轮没有 scheduled tokens，通常会直接返回 empty output。

但有一个特殊情况：

位置：`vllm/vllm/v1/worker/gpu_model_runner.py:4099`

```text
external_launcher + data_parallel_size > 1 + num_scheduled_tokens == 0
  → self._dummy_run(1)
```

注释说明原因是：

```text
确保 coordinate_batch_across_dp 被调用，避免 DP ranks 之间 out of sync。
```

---

## 13. DP 下 GPU / rank 如何落到设备

### 13.1 GPUWorker 会按 DP local rank 偏移 local_rank

`GPUWorker.init_device()` 中：

位置：`vllm/vllm/v1/worker/gpu_worker.py:249`

如果不是 Ray / external launcher，并且 DP backend 也不是 Ray，会做：

```text
dp_local_rank = data_parallel_rank_local
             or data_parallel_index

tp_pp_world_size = pipeline_parallel_size * tensor_parallel_size

self.local_rank += dp_local_rank * tp_pp_world_size
```

位置：`vllm/vllm/v1/worker/gpu_worker.py:261`

这表示单机多 DP 时，设备布局大致是：

```text
DP0 使用 local ranks: 0..TP*PP-1
DP1 使用 local ranks: TP*PP..2*TP*PP-1
DP2 使用 local ranks: 2*TP*PP..3*TP*PP-1
```

### 13.2 CoreEngineProcManager 也会按 DP rank 设置物理 GPU 映射

`CoreEngineProcManager` 启动进程前，会在需要时调用：

```text
set_assigned_physical_gpu_ids_for_dp_rank(...)
```

位置：`vllm/vllm/v1/engine/utils.py:183`

它的作用是：

```text
为当前 DP shard 设置 logical GPU → physical GPU 的映射，
让 topology / NIC affinity / P2P 检查使用正确设备集合。
```

---

## 14. DP 下 KV cache 如何处理

### 14.1 KV cache 是 replica-local 的

每个 EngineCore 初始化时都会调用 `_initialize_kv_caches()`。

位置：`vllm/vllm/v1/engine/core.py:239`

它会从本 EngineCore 的 model executor 收集 KV cache specs，并构造本地 scheduler 用的 `KVCacheConfig`。

所以默认情况：

```text
DP rank 0 的 KV cache 和 block pool 只服务 rank 0 的请求；
DP rank 1 的 KV cache 和 block pool 只服务 rank 1 的请求。
```

### 14.2 prefix cache 也是 replica-local 的

因为 prefix cache 依赖 scheduler 的 block pool / block hash / cached blocks，而这些都在本 EngineCore 的 scheduler 内部，所以 prefix cache 命中通常也只在当前 replica 内发生。

请求如果被分到不同 DP rank：

```text
即使 prompt 相同，也不会天然共享已缓存 KV。
```

除非启用 KV connector / KV transfer 这类额外机制。

### 14.3 KV connector 下 engine_id 会追加 DP 后缀

`EngineCoreProc.run_engine_core()` 中，如果是 DP 且有 `kv_transfer_config`，会改写 engine_id：

位置：`vllm/vllm/v1/engine/core.py:1175`

```text
engine_id = f"{engine_id}_dp{local_dp_rank}"
```

这是为了避免不同 DP rank 的 KV transfer 身份冲突。

---

## 15. DP 和 Executor / Worker 的关系

### 15.1 每个 EngineCore 都有自己的 Executor

`EngineCore.__init__()` 中：

位置：`vllm/vllm/v1/engine/core.py:123`

```text
self.model_executor = executor_class(vllm_config)
```

所以 DP 并不是一个 Executor 管理所有 replica 的请求；而是：

```text
每个 DP EngineCore 内部都有自己的 Executor。
```

### 15.2 Executor 仍然负责 replica 内部分发

对某个 DP rank 来说，`Executor.execute_model()` 的职责仍然是：

```text
把当前 SchedulerOutput 分发给本 replica 内部的 workers。
```

如果该 replica 内部有 TP / PP：

```text
Executor → WorkerWrapper → Worker → GPUModelRunner
```

这条链路和非 DP 基本一致。

### 15.3 DP 请求分发发生在 Executor 之前

可以把边界记成：

```text
EngineCoreClient：决定请求去哪个 DP replica；
Scheduler：在该 replica 内决定本轮跑哪些 token；
Executor：在该 replica 内把 SchedulerOutput 发给 worker；
Worker / ModelRunner：执行本轮 forward / sampling。
```

---

## 16. internal LB / external LB / hybrid LB 对比

| 模式 | 配置 | client 管理范围 | 请求分流者 | stats / coordinator |
|---|---|---|---|---|
| internal LB | `data_parallel_size > 1`，未启用 external/hybrid | 通常管理全部 DP engines | vLLM `DPLBAsyncMPClient` | 使用 coordinator stats |
| external LB | `data_parallel_external_lb=True` | 只管理本地 engine | 外部 LB | coordinator 主要用于 wave，不发布跨 rank LB stats |
| hybrid LB | `data_parallel_hybrid_lb=True` | 管理本地 engines | 节点内 vLLM，节点间外部 LB | 本地/协调 stats |
| offline SPMD | 环境变量指定 DP rank | 每个进程一个 local engine | 用户 / 启动脚本 | 通常无在线 coordinator |

### 16.1 internal LB

前端看到所有 DP engines，根据 coordinator 发布的 waiting / running 数选择 rank。

最小评分：

```text
score = waiting * 4 + running
```

位置：`vllm/vllm/v1/engine/core_client.py:1429`

### 16.2 external LB

外部系统先把请求打到某个 vLLM 实例 / pod，这个实例只把请求发给自己负责的 DP rank。

源码上 `make_async_mp_client()` 会选择 `DPAsyncMPClient`，而不是 `DPLBAsyncMPClient`。

位置：`vllm/vllm/v1/engine/core_client.py:126`

### 16.3 hybrid LB

hybrid 模式下，外部 LB 只负责节点间流量，节点内多个本地 DP engines 仍可由 vLLM 选择。

`local_engines_only` 会让 client 只管理本地 DP engines。

位置：`vllm/vllm/config/parallel.py:531`

---

## 17. 启动握手和 DP 配置校验

### 17.1 EngineHandshakeMetadata

EngineCore 启动时会通过 ZMQ handshake 收到：

位置：`vllm/vllm/v1/engine/utils.py:78`

```text
EngineHandshakeMetadata(
    addresses=EngineZmqAddresses(...),
    parallel_config={...},
)
```

DP coordinated 场景会下发：

位置：`vllm/vllm/v1/engine/utils.py:1313`

```text
data_parallel_master_ip
data_parallel_master_port
_data_parallel_master_port_list
data_parallel_size
```

### 17.2 READY 阶段会校验 parallel_config hash

`EngineCoreProc._perform_handshake()` 在 READY 消息里带上：

位置：`vllm/vllm/v1/engine/core.py:1105`

```text
parallel_config_hash = parallel_config.compute_hash()
```

`wait_for_engine_startup()` 对 coordinated DP 会校验所有 worker hash 一致。

位置：`vllm/vllm/v1/engine/utils.py:1333`

目的：

```text
避免不同 DP workers 使用会影响 collective 通信结构的不同配置，
否则可能在 MoE / EP collective 中 hang。
```

### 17.3 compute_hash 会忽略 rank / port 等运行时差异

`ParallelConfig.compute_hash()` 会排除：

位置：`vllm/vllm/config/parallel.py:735`

```text
data_parallel_rank
data_parallel_rank_local
data_parallel_size_local
data_parallel_backend
data_parallel_external_lb
data_parallel_hybrid_lb
data_parallel_master_ip / port
rank / node_rank / worker_cls / placement_group 等
```

保留的是会影响计算图或 collective 语义的配置。

---

## 18. 几个容易混淆的点

### 18.1 DP 下是不是只有一个全局 Scheduler？

不是。

```text
每个 DP EngineCore 都有自己的 Scheduler。
```

DPCoordinator 只做 stats / wave，不做 token 调度。

### 18.2 data_parallel_rank 是请求路由 rank 还是 torch rank？

两者相关但不是一回事。

```text
data_parallel_rank：DP replica 编号；
torch distributed rank：会按 DP rank、TP/PP rank 组合成全局 rank。
```

`init_distributed_environment()` 会在 DP 场景把 rank 偏移：

位置：`vllm/vllm/distributed/parallel_state.py:1566`

```text
rank = data_parallel_rank * world_size + rank
world_size = world_size_across_dp
```

### 18.3 data_parallel_index 和 data_parallel_rank 有什么区别？

`data_parallel_index` 默认等于 `data_parallel_rank`。

位置：`vllm/vllm/config/parallel.py:863`

但 dense DP 下，`run_engine_core()` 可能把 `data_parallel_size` 和 `data_parallel_rank` 重置成单 replica 视角，同时保留 `data_parallel_index` 表示原始 DP engine index。

位置：`vllm/vllm/v1/engine/core.py:1186`

所以：

```text
data_parallel_rank：参与 DP process group / MoE coordinated DP 的 rank；
data_parallel_index：保留这个 EngineCore 原本属于哪个 DP engine 的语义索引。
```

### 18.4 DP 下 KV cache 会自动跨 replica 共享吗？

不会。

默认是：

```text
每个 replica 自己维护 KV cache 和 prefix cache。
```

跨 replica 共享需要 KV connector / KV transfer 等额外机制，而且还要处理每个 DP rank 的 engine identity。

### 18.5 DP 和 Ray / mp 是什么关系？

`data_parallel_backend` 决定 DP engine 的启动方式：

```text
mp：CoreEngineProcManager 启动进程；
ray：CoreEngineActorManager 启动 Ray actors。
```

`distributed_executor_backend` 决定单个 replica 内部 executor / worker 怎么启动。

这两个配置处在不同层级。

### 18.6 num_tokens_across_dp 是全局 batch 吗？

不是。

它不是把所有 DP rank 的 token 拼成一个 batch，而是：

```text
每个 DP rank 本轮要跑多少 token 的向量。
```

主要用于：

```text
- cudagraph mode 对齐；
- DP padding；
- ubatching / DBO 一致性；
- forward context 中的 DP-aware compiled 路径。
```

### 18.7 为什么有时候没有请求也要 dummy batch？

MoE / EP collective 需要 DP ranks 以一致节奏进入某些通信路径。

所以某个 rank 没有本地请求时，只要全局 wave 还没结束，就可能需要 dummy batch 维持同步。

---

## 19. 最终可以记成一张表

| 层级 | DP 下做什么 | 关键对象 |
|---|---|---|
| 配置层 | 定义 DP size/rank/backend/LB 模式 | `ParallelConfig` |
| 前端层 | 接收请求，可指定 `data_parallel_rank` | `AsyncLLM.add_request()` |
| 输入处理 | 校验 DP rank 并写入请求 | `InputProcessor.process_inputs()` |
| Client 路由 | 选择目标 DP engine | `DPAsyncMPClient` / `DPLBAsyncMPClient` |
| Coordinator | 发布 queue stats、维护 wave、广播 START_DP_WAVE | `DPCoordinator` |
| EngineCore | 每个 DP rank 独立 Scheduler / Executor / KV cache | `EngineCoreProc` / `DPEngineCoreProc` |
| Scheduler | replica-local 调度 | `Scheduler.schedule()` |
| Executor | replica 内 worker 分发 | `Executor.execute_model()` |
| Worker | 设备初始化和 local rank 偏移 | `GPUWorker.init_device()` |
| ModelRunner | DP batch padding / cudagraph / ubatch 同步 | `coordinate_batch_across_dp()` |
| distributed groups | 构造 TP/PP/DP/EP/EPLB groups | `initialize_model_parallel()` |

---

## 20. 主链路复盘

可以把 vLLM V1 的 DP 记成这条链：

```text
ParallelConfig
  → data_parallel_size / rank / backend / LB mode
  → world_size_across_dp = DP × TP × PP × PCP

launch_core_engines()
  → 可选启动 DPCoordinator
  → 为每个本地 DP rank 启动 EngineCoreProc / DPEngineCoreProc
  → handshake 下发 DP 地址 / 端口 / size

AsyncLLM.add_request()
  → InputProcessor 写入 request.data_parallel_rank
  → DPLBAsyncMPClient 根据指定 rank 或 stats 选择 EngineCore
  → 请求进入对应 DP replica

每个 DP replica 内部
  → Scheduler.schedule()
  → Executor.execute_model(SchedulerOutput)
  → Worker / GPUModelRunner
  → replica-local KV cache / InputBatch / attention metadata
  → ModelRunnerOutput
  → Scheduler.update_from_output()

MoE coordinated DP 额外有
  → DPEngineCoreProc
  → stateless DP group
  → request wave
  → dummy batch
  → sync_dp_state all-reduce

GPUModelRunner 额外有
  → coordinate_batch_across_dp()
  → 同步 ubatch / cudagraph / token padding
  → num_tokens_across_dp 写入 forward context
```

---

## 21. 一句话总结

Data Parallel 在 vLLM 中本质是“多个 EngineCore replica 的请求级扩展”：

```text
前端负责把请求路由到某个 DP replica；
每个 replica 独立调度、独立维护 KV cache、独立执行 SchedulerOutput；
TP / PP / CP / EP 是 replica 内部的模型并行；
MoE DP 由于跨 DP collective，需要额外的 coordinator、wave 和 dummy batch；
GPUModelRunner 还要用 coordinate_batch_across_dp 对齐 cudagraph / ubatching / padding。
```

如果只记住一句话：

```text
DP 不是切分一次 forward，而是复制执行系统；请求先选 replica，进入 replica 后再走普通 EngineCore → Scheduler → Executor → Worker → ModelRunner 链路。
```
