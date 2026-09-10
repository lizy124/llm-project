# 08 关键文件阅读顺序与调试地图

本篇给出 vLLM 推理引擎层的推荐阅读顺序、关键文件职责和按问题类型的调试地图。

## 1. 推荐阅读总顺序

建议按“从外到内、从控制流到数据结构、从 Python 到 kernel”的顺序读。

```text
1. entrypoints/openai/api_server.py
2. v1/engine/async_llm.py
3. v1/engine/input_processor.py
4. v1/engine/output_processor.py
5. v1/engine/core_client.py
6. v1/engine/core.py
7. v1/core/sched/scheduler.py
8. v1/core/sched/output.py
9. v1/core/kv_cache_manager.py
10. v1/core/block_pool.py
11. v1/executor/abstract.py
12. v1/executor/uniproc_executor.py 或 multiproc_executor.py
13. v1/worker/gpu_worker.py
14. v1/worker/gpu_model_runner.py
15. v1/worker/block_table.py / v1/worker/gpu/block_table.py
16. v1/attention/backend.py
17. model_executor/layers/attention/attention.py
18. model_executor/models/具体模型
19. csrc/attention 与 csrc/libtorch_stable/attention
20. distributed/parallel_state.py 与 device_communicators
```

## 2. 新人最快理解路径

如果只是想理解“请求怎么跑起来”，优先读这几个：

1. `code/vllm/vllm/v1/engine/async_llm.py:70`
   - 看 `AsyncLLM` 如何接请求、创建 EngineCoreClient。

2. `code/vllm/vllm/v1/engine/core.py:96`
   - 看 `EngineCore` 如何初始化 executor、KV cache、scheduler。

3. `code/vllm/vllm/v1/engine/core.py:479`
   - 看 `step()` 如何 schedule、execute、sample、update。

4. `code/vllm/vllm/v1/core/sched/scheduler.py:387`
   - 看调度主逻辑。

5. `code/vllm/vllm/v1/executor/abstract.py:221`
   - 看 executor 如何把 SchedulerOutput 发给 worker。

6. `code/vllm/vllm/v1/worker/gpu_model_runner.py:4044`
   - 看 worker 侧如何准备输入并执行模型。

7. `code/vllm/vllm/model_executor/layers/attention/attention.py:438`
   - 看模型 attention 如何走 backend。

## 3. 关键文件职责表

| 文件 | 职责 | 阅读重点 |
|---|---|---|
| `vllm/entrypoints/openai/api_server.py` | OpenAI API server | 请求如何进入 AsyncLLM |
| `vllm/engine/arg_utils.py` | EngineArgs/配置转换 | CLI 参数到 VllmConfig |
| `vllm/v1/engine/async_llm.py` | 前台异步引擎 | add_request/generate/output handler |
| `vllm/v1/engine/input_processor.py` | 输入处理 | EngineInput 到 EngineCoreRequest |
| `vllm/v1/engine/output_processor.py` | 输出处理 | EngineCoreOutputs 到 RequestOutput |
| `vllm/v1/engine/core_client.py` | 前后台通信 | Inproc/MP/AsyncMP client |
| `vllm/v1/engine/core.py` | V1 engine core | 初始化、step、batch queue、EngineCoreProc |
| `vllm/v1/core/sched/scheduler.py` | 调度器 | schedule/update_from_output/free/preempt |
| `vllm/v1/core/sched/output.py` | 调度输出结构 | SchedulerOutput 字段含义 |
| `vllm/v1/request.py` | 请求内部状态 | Request/RequestStatus |
| `vllm/v1/core/kv_cache_manager.py` | KV 管理入口 | allocate_slots/get_computed_blocks/free |
| `vllm/v1/core/block_pool.py` | block 池 | get_new_blocks/cache_full_blocks/evict |
| `vllm/v1/core/kv_cache_coordinator.py` | hybrid cache 协调 | 多 cache group 管理 |
| `vllm/config/cache.py` | cache 配置 | block_size/cache_dtype/prefix/offload |
| `vllm/v1/executor/abstract.py` | Executor 抽象 | collective_rpc/execute_model/sample_tokens |
| `vllm/v1/executor/uniproc_executor.py` | 单进程执行 | 本地 worker 调用 |
| `vllm/v1/executor/multiproc_executor.py` | 多进程执行 | worker 进程管理 |
| `vllm/v1/executor/ray_executor*.py` | Ray 执行 | 分布式 worker 管理 |
| `vllm/v1/worker/worker_base.py` | Worker 接口 | worker 必须实现哪些能力 |
| `vllm/v1/worker/gpu_worker.py` | GPU worker | load/profile/kv init/warmup/execute |
| `vllm/v1/worker/gpu_model_runner.py` | GPU 执行核心 | batch/input/attention/forward/sample |
| `vllm/v1/worker/gpu/model_runner.py` | V2 GPU model runner | 新执行路径 |
| `vllm/v1/worker/gpu/input_batch.py` | batch 状态 | request state/token buffer |
| `vllm/v1/worker/block_table.py` | block table | logical block 到 physical block |
| `vllm/v1/worker/gpu/attn_utils.py` | attention 辅助 | backend/cache/slot 初始化 |
| `vllm/v1/attention/backend.py` | attention backend 抽象 | metadata/backend capability |
| `vllm/model_executor/layers/attention/attention.py` | 模型 Attention 层 | forward/get_kv_cache_spec/context |
| `vllm/model_executor/model_loader/` | 模型加载 | 权重加载 |
| `vllm/model_executor/models/` | 模型定义 | 具体模型 forward |
| `vllm/distributed/` | 分布式通信 | TP/PP/DP/communicator |
| `csrc/` | 底层 kernel | paged attention/cache/all-reduce/MoE |

## 4. 按问题类型的调试地图

### 4.1 请求没有返回 / 卡住

优先看：

```text
AsyncLLM.generate/add_request
OutputProcessor.process_outputs
EngineCoreClient.get_output_async
EngineCoreProc.run_busy_loop
EngineCore.step
Scheduler.has_requests
```

重点判断：

- request 是否成功进入 scheduler；
- output handler 是否运行；
- EngineCore 是否还活着；
- scheduler 是否认为有请求；
- worker execute 是否卡住。

### 4.2 请求排队太久 / 吞吐低

优先看：

```text
Scheduler.schedule
max_num_scheduled_tokens
max_num_running_reqs
long_prefill_token_threshold
chunked prefill
preemption
KV block free 数量
```

重点判断：

- token budget 是否太小；
- running 请求是否占满；
- waiting 是否被 LoRA/encoder/KV connector 条件跳过；
- KV block 是否不足导致频繁 preempt；
- chunked prefill 是否影响 decode latency。

### 4.3 OOM / KV cache 太小

优先看：

```text
config/cache.py
EngineCore._initialize_kv_caches
gpu_worker.determine_available_memory
GPUModelRunner.profile_run
get_kv_cache_configs
cache_config.num_gpu_blocks
```

重点判断：

- `gpu_memory_utilization` 是否太高/太低；
- 是否手动指定 `kv_cache_memory_bytes`；
- CUDA graph 是否额外占用显存；
- max model len 是否过大；
- quantization/cache dtype 是否符合预期。

### 4.4 Prefix cache 不生效

优先看：

```text
cache_config.enable_prefix_caching
EngineCore.request_block_hasher
Request.block_hashes
KVCacheManager.get_computed_blocks
BlockPool.get_cached_block
KVCacheManager.cache_blocks
```

重点判断：

- 是否启用 prefix caching；
- non-causal attention 是否禁用了 prefix caching；
- request 是否 `skip_reading_prefix_cache`；
- block hash 是否一致；
- block 是否完整并已 cache；
- hash_block_size 与 block_size 是否符合预期。

### 4.5 请求被频繁 preempt

优先看：

```text
Scheduler.schedule
KVCacheManager.allocate_slots
BlockPool.get_num_free_blocks
Scheduler._preempt_request
Request.num_preemptions
```

重点判断：

- KV blocks 是否不足；
- running 请求数是否过多；
- max_num_batched_tokens 是否过大；
- prefill chunk 太大；
- priority policy 是否导致低优先级请求被抢占。

### 4.6 输出 token 错误 / 采样异常

优先看：

```text
GPUModelRunner.execute_model
GPUModelRunner.sample_tokens
apply_grammar_bitmask
_sample
Scheduler.update_from_output
OutputProcessor.process_outputs
```

重点判断：

- logits 是否正常；
- grammar bitmask 是否错误屏蔽 token；
- sampling params 是否正确；
- spec decode accept/reject 是否异常；
- output processor 是否正确拼接 streaming 输出。

### 4.7 Attention/kernel 报错

优先看：

```text
GPUModelRunner._get_slot_mappings
GPUModelRunner._build_attention_metadata
set_forward_context
Attention.forward
AttentionBackend.validate_configuration
csrc attention/cache kernels
```

重点判断：

- block table 是否正确；
- slot mapping 是否越界；
- KV cache shape/layout 是否和 backend 要求一致；
- head size/block size/dtype 是否被 backend 支持；
- CUDA graph padding 是否导致维度不一致。

### 4.8 分布式/多卡问题

优先看：

```text
parallel_config
Executor.get_class
MultiprocExecutor/RayExecutor
Worker init_distributed_environment
vllm/distributed/parallel_state.py
device_communicators
Pipeline parallel intermediate tensors
custom all-reduce
```

重点判断：

- TP/PP/DP rank 是否正确；
- executor backend 是否符合预期；
- NCCL/custom all-reduce 是否初始化；
- PP 非末级 rank 是否只返回 intermediate tensors；
- DP batch coordination 是否同步。

### 4.9 KV transfer / disaggregated prefill 问题

优先看：

```text
vllm/distributed/kv_transfer
Scheduler.connector
Worker.ensure_kv_transfer_initialized
get_kv_connector_handshake_metadata
Scheduler._build_kv_connector_meta
GPUModelRunner maybe_get_kv_connector_output
```

重点判断：

- scheduler 和 worker connector role 是否匹配；
- handshake metadata 是否收集完整；
- remote KV 命中 token 数是否正确；
- async load 状态是否推进；
- KV load failure policy 是 recompute 还是失败。

## 5. 阅读不同目标的最短路径

### 只想理解普通在线生成

```text
async_llm.py
core.py step
scheduler.py schedule
executor/abstract.py execute_model
gpu_worker.py
gpu_model_runner.py execute_model/sample_tokens
attention.py forward
```

### 想理解 KV cache

```text
config/cache.py
core.py _initialize_kv_caches
kv_cache_utils.py
kv_cache_manager.py
block_pool.py
scheduler.py schedule
worker/block_table.py
gpu_model_runner.py _get_slot_mappings
```

### 想理解性能优化

```text
scheduler.py token budget/chunked prefill
GPUModelRunner._determine_batch_execution_and_padding
CUDA graph capture_model
ubatching
attention backend
csrc kernels
```

### 想理解分布式

```text
parallel_config
Executor.get_class
multiproc_executor.py / ray_executor.py
gpu_worker.py distributed init
distributed/parallel_state.py
GPUModelRunner PP/DP/TP 相关逻辑
```

### 想理解结构化输出

```text
async_llm.py sampling params
structured_output manager
scheduler.get_grammar_bitmask
GPUModelRunner.sample_tokens
apply_grammar_bitmask
OutputProcessor
```

## 6. 建议画在脑子里的总图

```text
用户请求
  ↓
API / AsyncLLM
  ↓ EngineCoreRequest
EngineCoreClient
  ↓
EngineCore
  ↓
Scheduler -- KVCacheManager -- BlockPool
  ↓ SchedulerOutput
Executor
  ↓ collective_rpc
Worker
  ↓
GPUModelRunner -- BlockTable -- AttentionMetadata
  ↓
ModelExecutor Model
  ↓
Attention Layer -- ForwardContext -- AttentionBackend
  ↓
CUDA/C++ Kernel
  ↓
Sampler
  ↓ ModelRunnerOutput
Scheduler.update_from_output
  ↓ EngineCoreOutputs
OutputProcessor
  ↓
用户输出
```

## 7. 最后建议

阅读 vLLM 推理引擎层不要从 `model_executor/models` 的具体模型开始。具体模型很多，容易陷进去。应该先抓住这条主链：

```text
AsyncLLM -> EngineCore -> Scheduler -> Executor -> Worker -> GPUModelRunner -> Attention
```

然后再根据问题深入 KV cache、attention backend、distributed 或具体模型实现。
