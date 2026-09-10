# vLLM 仓库概念总览：从用户入口到核心执行引擎

源码根目录：以本仓库根目录下的 `vllm/` Python 包为主，底层实现还涉及 `csrc/`、`rust/`、`tests/`、`benchmarks/`、`docs/` 等目录。

本文基于 `vllm` 仓库源码进行梳理，目标是从浅入深解释 vLLM 的主要模块、它们之间的关系、一次请求如何流动，以及各模块在代码中的证据位置。

> 说明：本文不是 API 使用教程，而是面向源码阅读的概念地图。每个核心概念尽量给出对应源码路径和类 / 函数位置，避免凭空猜测。

---

## 1. vLLM 是什么

从项目描述看，vLLM 是一个用于 LLM 推理和服务的高吞吐、内存高效推理引擎。

项目 README 对它的定位是：

```text
vLLM is a fast and easy-to-use library for LLM inference and serving.
```

位置：`README.md:22` 到 `README.md:25`

`pyproject.toml` 中的项目描述是：

```toml
description = "A high-throughput and memory-efficient inference and serving engine for LLMs"
```

位置：`pyproject.toml:22`

从功能上看，vLLM 主要解决这些问题：

```text
1. 如何把用户请求接入推理服务；
2. 如何把请求转成内部调度对象；
3. 如何高效调度 prefill / decode；
4. 如何管理 KV cache / prefix cache；
5. 如何把 batch 交给 GPU / CPU / XPU Worker 执行；
6. 如何支持 tensor parallel / pipeline parallel / data parallel / expert parallel；
7. 如何支持 OpenAI API、offline LLM、batch、pooling、embedding、多模态、LoRA、结构化输出、投机解码；
8. 如何通过 attention backend、CUDA graph、torch.compile、自定义 kernel 提升性能。
```

README 列出的核心能力包括：

```text
PagedAttention；
continuous batching；
chunked prefill；
prefix caching；
CUDA/HIP graphs；
quantization；
FlashAttention / FlashInfer / TRTLLM-GEN / FlashMLA / Triton；
speculative decoding；
torch.compile；
disaggregated prefill / decode / encode；
OpenAI-compatible API server；
multi-LoRA；
多硬件平台支持。
```

位置：`README.md:28` 到 `README.md:51`

---

## 2. 仓库整体分层

从源码包 `vllm/` 看，可以把仓库分成这些层：

注意：这里的 `v1/core/` 主要放 Scheduler、KV cache、block pool 等核心调度 / 缓存组件；`EngineCore` 本体在 `v1/engine/core.py`。

```text
用户入口层
  entrypoints/
  engine/
  outputs.py
  sampling_params.py
  pooling_params.py

配置层
  config/
  engine/arg_utils.py

V1 核心运行时
  v1/engine/
  v1/core/
  v1/executor/
  v1/worker/
  v1/outputs.py

模型执行层
  model_executor/model_loader/
  model_executor/models/
  model_executor/layers/

调度与缓存层
  v1/core/sched/
  v1/core/kv_cache_manager.py
  v1/core/block_pool.py
  v1/core/encoder_cache_manager.py

采样、结构化输出、投机解码
  v1/sample/
  sampling_params.py
  v1/structured_output/
  v1/spec_decode/

分布式与平台层
  distributed/
  platforms/
  v1/executor/ray_executor.py
  v1/executor/ray_executor_v2.py
  v1/executor/ray_utils.py

高性能底层
  v1/attention/
  v1/attention/ops/
  compilation/
  model_executor/kernels/
  model_executor/layers/
  vllm_flash_attn/
  _custom_ops.py
  _aiter_ops.py
  _xpu_ops.py

高级能力
  multimodal/
  lora/
  tokenizers/
  transformers_utils/
  parser/
  reasoning/
  tool_parsers/
  renderers/

工程支撑
  benchmarks/
  tests/
  docs/
  profiler/
  tracing/
  logging_utils/
  usage/
```

最小心智模型：

```text
用户入口
  → 配置系统
  → Engine / AsyncLLM / LLMEngine
  → EngineCore
  → Scheduler
  → Executor
  → Worker / ModelRunner
  → Model / Attention / Sampler
  → EngineCoreOutputs
  → OutputProcessor
  → 用户可见输出
```

---

## 3. 从用户视角看 vLLM 的入口

vLLM 有两类典型使用方式：

```text
1. Offline / Python API：
   直接在 Python 里创建 LLM 对象，调用 generate / encode / classify 等。

2. Online serving：
   启动 OpenAI-compatible API server，通过 HTTP 接收请求。
```

### 3.1 命令行入口

`pyproject.toml` 声明了命令行脚本：

```toml
[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"
```

位置：`pyproject.toml:43` 到 `pyproject.toml:44`

这说明用户运行：

```bash
vllm ...
```

最终会进入：

```text
vllm.entrypoints.cli.main:main
```

CLI 子命令中，serve 相关入口在：

```text
vllm/entrypoints/cli/serve.py
```

代表类：

```text
ServeSubcommand
```

位置：`vllm/entrypoints/cli/serve.py:44`

### 3.2 Offline LLM 入口

Python API 的主要入口是：

```text
vllm/entrypoints/llm.py
```

核心类：

```text
LLM
```

位置：`vllm/entrypoints/llm.py`

这个层级通常面向：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="...")
outputs = llm.generate(prompts, sampling_params)
```

它的职责不是自己做底层调度，而是：

```text
接收用户参数；
创建底层 engine；
把 generate / encode / pooling 等调用转成 engine 请求；
收集并返回 RequestOutput / PoolingRequestOutput。
```

### 3.3 OpenAI API Server 入口

OpenAI-compatible API 相关代码在：

```text
vllm/entrypoints/openai/
```

典型服务类包括：

```text
OpenAIServingChat
OpenAIServingCompletion
OpenAIServingResponses
```

代码证据：

```text
vllm/entrypoints/openai/chat_completion/serving.py:106
  OpenAIServingChat

vllm/entrypoints/openai/completion/serving.py:55
  OpenAIServingCompletion

vllm/entrypoints/openai/responses/serving.py:150
  OpenAIServingResponses
```

这些模块负责：

```text
HTTP 请求协议解析；
OpenAI request schema 转换；
chat template / tool calling / reasoning parser；
把请求转给 AsyncLLM；
把 RequestOutput 转成 OpenAI-compatible response；
支持 streaming。
```

### 3.4 entrypoints 层的总体职责

`entrypoints/` 可以理解为：

```text
vLLM 的用户接口适配层。
```

它把不同外部协议：

```text
Python offline API；
OpenAI Chat Completions；
OpenAI Completions；
OpenAI Responses；
Embeddings；
Pooling；
Tokenization；
Batch；
Speech-to-text；
```

统一转成内部 engine 能处理的请求。

---

## 4. Engine 层：外层引擎和内部 EngineCore

vLLM V1 的运行时核心集中在：

```text
vllm/v1/engine/
```

这个目录是理解 vLLM 主链路的关键。

### 4.1 外层 Engine：LLMEngine 和 AsyncLLM

V1 里典型外层 Engine 有两个：

```text
LLMEngine：
  同步 / legacy 兼容路径。

AsyncLLM：
  异步 / API server 常用路径。
```

代码证据：

```text
vllm/v1/engine/llm_engine.py:48
  class LLMEngine

vllm/v1/engine/async_llm.py:70
  class AsyncLLM
```

外层 Engine 主要负责：

```text
接收外部请求；
调用 InputProcessor 转成 EngineCoreRequest；
通过 EngineCoreClient 送入 EngineCore；
从 EngineCoreClient 拉取 EngineCoreOutputs；
调用 OutputProcessor 转成 RequestOutput / PoolingRequestOutput。
```

### 4.2 EngineCore：内部执行主循环

`EngineCore` 定义在：

```text
vllm/v1/engine/core.py
```

源码注释：

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""
```

位置：`vllm/v1/engine/core.py:96`

它是内部执行闭环总控。

核心 `step()`：

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output.
```

位置：`vllm/v1/engine/core.py:488`

普通一轮 step 做：

```text
scheduler.has_requests()
  → scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output, non_block=True)
  → scheduler.get_grammar_bitmask(...)
  → future.result()
  → model_executor.sample_tokens(grammar_output)  # 当 execute_model 未直接返回 model_output 时
  → scheduler.update_from_output(scheduler_output, model_output)
  → EngineCoreOutputs
```

### 4.3 EngineCoreClient：外层访问内部 Core 的桥

外层 Engine 通常不直接持有 `EngineCore`，而是持有 `EngineCoreClient`。

代码证据：

```text
vllm/v1/engine/core_client.py:71
  class EngineCoreClient
```

其注释说明它负责不同运行模式下的 push / pull：

```text
EngineCoreClient: subclasses handle different methods for pushing
and pulling from the EngineCore for asyncio / multiprocessing.
```

位置：`vllm/v1/engine/core_client.py:71` 到 `vllm/v1/engine/core_client.py:80`

主要子类：

```text
InprocClient：
  同进程 EngineCore。

SyncMPClient：
  同步多进程 client，通过 ZMQ 连接后台 EngineCore。

AsyncMPClient：
  异步多进程 client，通过 asyncio + ZMQ 连接后台 EngineCore。
```

### 4.4 EngineCoreProc：后台进程包装器

多进程模式下，后台运行的是：

```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
```

位置：`vllm/v1/engine/core.py:905`

它可以理解为：

```text
EngineCore + ZMQ 输入输出 + input_queue / output_queue + busy loop + shutdown 状态机。
```

它不是叫 `EngineCoreServer`，源码里没有这个类名。

如果用 client/server 类比：

```text
EngineCoreClient：client / proxy
EngineCoreProc：server-side process wrapper
EngineCore：真正内部执行核心
```

---

## 5. 一次请求的完整主链路

这里先给一条从用户输入到用户输出的全链路。

```text
用户 / API server / offline LLM
  → entrypoints 层
  → LLMEngine / AsyncLLM
  → InputProcessor
  → EngineCoreRequest
  → EngineCoreClient
  → EngineCore.preprocess_add_request()
  → Request
  → Scheduler.add_request()
  → EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → Executor.execute_model()
  → Worker.execute_model()
  → ModelRunner.execute_model()
  → model forward / sample / pooling
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → EngineCoreOutput / EngineCoreOutputs
  → OutputProcessor.process_outputs()
  → RequestOutput / PoolingRequestOutput
  → 用户 / API response
```

核心对象变化：

```text
外部请求协议
  → EngineCoreRequest
  → Request
  → SchedulerOutput
  → ModelRunnerOutput
  → EngineCoreOutput
  → EngineCoreOutputs
  → RequestOutput / PoolingRequestOutput
```

这个对象链很重要，因为它对应 vLLM 的职责边界：

```text
entrypoints / Engine：协议和用户接口
InputProcessor：输入适配
EngineCore：内部闭环控制
Scheduler：请求状态和调度
Executor / Worker / ModelRunner：模型执行
OutputProcessor：输出适配和 detokenize
```

---

## 6. 配置系统：VllmConfig 如何连接所有模块

vLLM 的配置系统集中在：

```text
vllm/config/
vllm/engine/arg_utils.py
```

### 6.1 EngineArgs：用户参数入口

CLI / Python API 的很多配置最终会进入 `EngineArgs`。

代码证据：

```text
vllm/engine/arg_utils.py:417
  EngineArgs
```

`EngineArgs` 负责把用户提供的模型、并行、缓存、量化、LoRA、多模态、调度等参数收拢起来，并转换为内部配置。

### 6.2 VllmConfig：全局配置聚合对象

核心配置对象是：

```text
VllmConfig
```

代码证据：

```text
vllm/config/vllm.py:288
  VllmConfig
```

`VllmConfig` 是一个聚合配置，它把多个子配置组合起来：

```text
ModelConfig
CacheConfig
ParallelConfig
SchedulerConfig
DeviceConfig
LoadConfig
OffloadConfig
AttentionConfig
MambaConfig
KernelConfig
LoRAConfig
SpeculativeConfig
DiffusionConfig
StructuredOutputsConfig
ObservabilityConfig
QuantizationConfig
CompilationConfig
ProfilerConfig
KVTransferConfig
KVEventsConfig
ECTransferConfig
ReasoningConfig
WeightTransferConfig
optimization_level / performance_mode
```

这些配置分散在：

```text
vllm/config/model.py
vllm/config/cache.py
vllm/config/parallel.py
vllm/config/scheduler.py
vllm/config/device.py
vllm/config/load.py
vllm/config/offload.py
vllm/config/attention.py
vllm/config/mamba.py
vllm/config/kernel.py
vllm/config/lora.py
vllm/config/speculative.py
vllm/config/diffusion.py
vllm/config/structured_outputs.py
vllm/config/observability.py
vllm/config/quantization.py
vllm/config/compilation.py
vllm/config/profiler.py
vllm/config/kv_transfer.py
vllm/config/kv_events.py
vllm/config/ec_transfer.py
vllm/config/reasoning.py
vllm/config/weight_transfer.py
```

代码证据示例：

```text
vllm/config/cache.py:44
  CacheConfig

vllm/config/parallel.py:117
  ParallelConfig

vllm/config/compilation.py:381
  CompilationConfig

vllm/config/kv_transfer.py:23
  KVTransferConfig
```

### 6.3 配置系统为什么重要

vLLM 的很多模块不是手动互相配置，而是通过 `VllmConfig` 连接。

例如：

```text
EngineCore 创建 model_executor 时传入 vllm_config；
Scheduler 初始化时接收 vllm_config 和 kv_cache_config；
Executor 根据 vllm_config.parallel_config 选择 uniproc / multiproc / ray；
Worker 根据 vllm_config.device_config / cache_config / model_config 初始化设备和模型；
Attention backend 根据模型、平台、配置选择；
LoRA / multimodal / structured output / speculative decoding 都由对应 config 控制是否启用。
```

所以阅读 vLLM 时要记住：

```text
VllmConfig 是贯穿全系统的配置总线。
```

---

## 7. Scheduler：请求队列、token budget 和 KV block 调度

调度器在：

```text
vllm/v1/core/sched/
```

核心类：

```text
Scheduler
SchedulerInterface
SchedulerOutput
```

代码证据：

```text
vllm/v1/core/sched/scheduler.py:69
  class Scheduler

vllm/v1/core/sched/scheduler.py:433
  Scheduler.schedule

vllm/v1/core/sched/interface.py:37
  SchedulerInterface

vllm/v1/core/sched/output.py:183
  SchedulerOutput
```

### 7.1 Scheduler 负责什么

Scheduler 的职责是：

```text
维护 waiting / running 请求；
决定本轮哪些请求执行；
决定每个请求本轮执行多少 token；
处理 prefill / decode / chunked prefill；
分配 KV cache blocks；
处理 prefix cache 命中；
处理抢占和重新调度；
处理 structured output grammar 状态；
处理 speculative decoding 状态；
处理 encoder inputs；
在 Worker 返回后更新 Request 状态；
释放资源；
生成 EngineCoreOutputs。
```

### 7.2 schedule()：发任务

`Scheduler.schedule()` 生成 `SchedulerOutput`。

`SchedulerOutput` 里包含：

```text
scheduled_new_reqs：首次调度的新请求；
scheduled_cached_reqs：Worker 已缓存请求的增量信息；
num_scheduled_tokens：每个请求本轮调度多少 token；
total_num_scheduled_tokens：本轮总 token 数；
scheduled_spec_decode_tokens：投机解码 draft tokens；
scheduled_encoder_inputs：本轮需要处理的 encoder inputs；
num_common_prefix_blocks：公共 prefix blocks；
finished_req_ids：通知 Worker 清理的 finished 请求；
kv_connector_metadata / ec_connector_metadata：connector 元信息。
```

代码证据：`vllm/v1/core/sched/output.py:183`

### 7.3 update_from_output()：收结果

Worker 返回 `ModelRunnerOutput` 后，Scheduler 调用：

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
```

位置：`vllm/v1/core/sched/scheduler.py:1551`

它的作用：

```text
把本轮计划 SchedulerOutput 和真实结果 ModelRunnerOutput 对账；
append 新 token；
检查 stop；
处理 logprobs；
处理 pooling output；
处理 spec decode 接受 / 拒绝；
处理 structured output grammar；
处理 KV connector output；
释放 finished request；
构造 EngineCoreOutput；
按 client_index 返回 EngineCoreOutputs。
```

---

## 8. KV Cache、PagedAttention 和 Block 管理

vLLM 高吞吐的一个核心是 KV cache 管理。

相关代码主要在：

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/block_pool.py
vllm/v1/core/encoder_cache_manager.py
vllm/v1/attention/
```

代码证据：

```text
vllm/v1/core/kv_cache_manager.py:114
  KVCacheManager

vllm/v1/core/block_pool.py:143
  BlockPool

vllm/v1/core/encoder_cache_manager.py:17
  EncoderCacheManager
```

### 8.1 KVCacheManager

`KVCacheManager` 管理请求和 KV blocks 的关系。

它关心：

```text
每个请求已经计算了多少 token；
哪些 token 对应哪些 block；
哪些 blocks 可以复用；
哪些 blocks 需要新分配；
哪些 blocks 需要释放；
prefix cache 是否命中；
preemption 时如何回收；
外部 KV transfer 时如何处理。
```

### 8.2 BlockPool

`BlockPool` 管理底层 block 的分配与释放。

它对应 PagedAttention 的思想：

```text
不要为每个请求连续分配一大段 KV cache；
而是把 KV cache 切成 blocks，
请求按需持有 block 列表。
```

这样可以支持：

```text
continuous batching；
请求动态加入 / 结束；
prefix cache 复用；
preemption；
高效显存利用。
```

### 8.3 EncoderCacheManager

多模态和 encoder-decoder 场景中，还需要管理 encoder output。

`EncoderCacheManager` 负责：

```text
缓存多模态 encoder 输出；
判断哪些 encoder input 已经安全处理；
在请求推进后释放不再需要的 encoder cache。
```

---

## 9. Executor：把 EngineCore 的执行请求分发给 Worker

Executor 在：

```text
vllm/v1/executor/
```

核心抽象：

```text
Executor
UniProcExecutor
MultiprocExecutor
RayDistributedExecutor / RayExecutorV2
```

代码证据：

```text
vllm/v1/executor/abstract.py:37
  class Executor

vllm/v1/executor/abstract.py:48
  Executor.get_class

vllm/v1/executor/uniproc_executor.py:45
  UniProcExecutor

vllm/v1/executor/multiproc_executor.py:70
  FutureWrapper

vllm/v1/executor/ray_executor_v2.py:219
  RayExecutorV2
```

### 9.1 Executor 的职责

Executor 负责：

```text
选择执行后端；
创建和管理 Worker；
在所有 Worker 上执行 RPC；
初始化 KV cache；
执行模型；
执行 sample_tokens；
转发 profile / sleep / wake_up / LoRA / collective_rpc；
处理 Worker failure。
```

`Executor` 注释说明：

```text
An executor is responsible for executing the model on one device,
or it can be a distributed executor that can execute the model on multiple devices.
```

位置：`vllm/v1/executor/abstract.py:37` 到 `vllm/v1/executor/abstract.py:42`

### 9.2 execute_model()

Executor 对 EngineCore 暴露：

```text
execute_model(scheduler_output)
sample_tokens(grammar_output)
```

抽象实现会通过 `collective_rpc` 调 Worker：

```text
Executor.execute_model()
  → collective_rpc("execute_model")
  → Worker.execute_model()
```

位置：`vllm/v1/executor/abstract.py:221` 到 `vllm/v1/executor/abstract.py:227`

---

## 10. Worker 和 ModelRunner：真正执行模型

Worker 代码在：

```text
vllm/v1/worker/
```

核心文件：

```text
worker_base.py
gpu_worker.py
gpu_model_runner.py
gpu_input_batch.py
cpu_model_runner.py
```

代码证据：

```text
vllm/v1/worker/worker_base.py:39
  WorkerBase

vllm/v1/worker/gpu_worker.py:130
  Worker

vllm/v1/worker/gpu_worker.py:424
  Worker.load_model

vllm/v1/worker/gpu_worker.py:1002
  Worker.execute_model

vllm/v1/worker/gpu_input_batch.py:92
  InputBatch

vllm/v1/worker/gpu_model_runner.py:445
  GPUModelRunner

vllm/v1/worker/gpu_model_runner.py:4097
  GPUModelRunner.execute_model
```

### 10.1 Worker 负责什么

Worker 是设备侧执行实体。

它负责：

```text
初始化 device；
加载模型；
profile 可用显存；
初始化 KV cache；
持有 ModelRunner；
处理 Executor 发来的 RPC；
执行模型；
采样；
管理 LoRA；
处理 sleep / wake_up / profile。
```

### 10.2 ModelRunner 负责什么

ModelRunner 是真正把 `SchedulerOutput` 变成模型 forward 输入的地方。

典型流程：

```text
_update_states(scheduler_output)
  → 更新 Worker 侧请求 / batch 状态

_prepare_inputs(...)
  → 准备 token ids、positions、logits indices、spec decode metadata

_build_attention_metadata(...)
  → 准备 attention backend 需要的 metadata

_preprocess(...)
  → 准备 input_ids / inputs_embeds / model_kwargs / connector output

_model_forward(...)
  → 真正调用模型 forward

compute_logits / pooling
  → 生成 logits 或 pooling output

sample_tokens(grammar_output)
  → 应用 grammar bitmask、采样、构造 ModelRunnerOutput
```

真正 forward 位置：

```text
vllm/v1/worker/gpu_model_runner.py:4380
  self._model_forward(...)
```

ModelRunnerOutput 构造位置：

```text
vllm/v1/worker/gpu_model_runner.py:4697
  ModelRunnerOutput(...)
```

---

## 11. 模型加载与模型实现

模型相关主要在：

```text
vllm/model_executor/model_loader/
vllm/model_executor/models/
vllm/model_executor/layers/
```

### 11.1 model_loader：模型实例化与权重加载

核心代码证据：

```text
vllm/model_executor/model_loader/base_loader.py:25
  BaseModelLoader

vllm/model_executor/model_loader/__init__.py:122
  get_model_loader

vllm/model_executor/model_loader/__init__.py:130
  get_model

vllm/model_executor/model_loader/utils.py:42
  initialize_model

vllm/model_executor/model_loader/weight_utils.py:828
  safetensors_weights_iterator
```

model_loader 负责：

```text
根据 LoadConfig 选择加载器；
下载 / 查找权重；
初始化模型类；
加载 safetensors / PyTorch / sharded state / tensorizer / bitsandbytes / dummy 等权重格式；
处理量化权重；
把模型放到目标 device。
```

### 11.2 models registry：根据 HF config 找模型类

模型 registry 在：

```text
vllm/model_executor/models/registry.py
```

代码证据：

```text
vllm/model_executor/models/registry.py:1007
  _ModelRegistry

vllm/model_executor/models/registry.py:1253
  resolve_model_cls

vllm/model_executor/models/registry.py:1323
  is_multimodal_model
```

它负责：

```text
读取 Hugging Face config 的 architectures；
解析对应 vLLM 模型类；
判断模型能力：文本生成、多模态、pooling、embedding 等；
支持大量模型架构。
```

### 11.3 layers：高性能模型层

模型层在：

```text
vllm/model_executor/layers/
```

代表代码：

```text
vllm/model_executor/layers/vocab_parallel_embedding.py:198
  VocabParallelEmbedding

vllm/model_executor/layers/vocab_parallel_embedding.py:505
  ParallelLMHead

vllm/model_executor/layers/layernorm.py:37
  RMSNorm

vllm/model_executor/layers/mla.py:34
  MultiHeadLatentAttentionWrapper
```

这一层提供：

```text
并行 embedding；
并行 linear；
LM head；
RMSNorm / LayerNorm；
MoE；
MLA；
logits processor；
activation；
rotary embedding；
量化 layer；
LoRA 适配 layer。
```

---

## 12. Attention 子系统

vLLM 的 attention 抽象在：

```text
vllm/v1/attention/
```

核心代码证据：

```text
vllm/v1/attention/backend.py:56
  AttentionBackend

vllm/v1/attention/backend.py:600
  AttentionMetadataBuilder

vllm/v1/attention/backend.py:769
  AttentionImplBase

vllm/v1/attention/selector.py:54
  get_attn_backend

vllm/v1/attention/backends/flashinfer.py:341
  FlashInferBackend
```

### 12.1 AttentionBackend

`AttentionBackend` 是 backend 抽象。

它需要回答：

```text
这个 backend 支持什么 KV cache layout；
如何构造 metadata；
如何执行 forward；
是否支持 cascade attention；
是否支持 sliding window；
是否支持 prefix / decode 特定优化；
在当前平台是否可用。
```

### 12.2 AttentionMetadataBuilder

ModelRunner 在每轮执行前，需要把 SchedulerOutput 和 batch 状态转成 attention backend 可理解的 metadata。

这类 metadata 包括：

```text
query length；
sequence length；
slot mapping；
block tables；
prefix length；
KV cache group；
spec decode metadata；
page table / paged KV cache 信息。
```

### 12.3 backend selector

`get_attn_backend` 负责选择具体 backend。

选择依据通常包括：

```text
当前平台；
模型 dtype；
head size；
KV cache dtype；
是否支持 sliding window；
是否安装 FlashInfer / FlashAttention；
配置指定的 backend；
硬件能力。
```

---

## 13. Sampling：从 logits 到 token

采样相关模块：

```text
vllm/sampling_params.py
vllm/v1/sample/
```

代码证据：

```text
vllm/sampling_params.py
  SamplingParams

vllm/v1/sample/sampler.py:20
  Sampler

vllm/v1/sample/sampler.py:243
  Sampler.sample

vllm/v1/sample/ops/topk_topp_sampler.py:70
  TopKTopPSampler

vllm/v1/sample/logits_processor/interface.py:60
  LogitsProcessor
```

### 13.1 SamplingParams

`SamplingParams` 是用户侧采样参数对象。

它通常包含：

```text
temperature；
top_p；
top_k；
min_p；
max_tokens；
stop；
stop_token_ids；
presence_penalty；
frequency_penalty；
repetition_penalty；
logprobs；
prompt_logprobs；
n；
best_of；
seed；
structured output 相关参数。
```

### 13.2 Sampler

Sampler 输入 logits，输出 sampled token ids 和 logprobs。

它要处理：

```text
temperature；
top-k / top-p；
penalties；
min tokens；
logprobs；
greedy / random sampling；
structured output grammar mask。
```

Speculative decoding 的 accepted / rejected 统计和 `num_computed_tokens` 修正在 `Scheduler.update_from_output()` 中根据 `scheduled_spec_decode_tokens` 和本轮生成 token 数处理，不属于 `Sampler` 的职责。

---

## 14. 输出对象：内部输出和用户输出

输出相关有两个层次：

```text
vllm/v1/outputs.py：
  内部 Worker / ModelRunner / Scheduler 用的输出结构。

vllm/outputs.py：
  用户可见输出结构。
```

代码证据：

```text
vllm/v1/outputs.py:186
  SamplerOutput

vllm/v1/outputs.py:234
  ModelRunnerOutput

vllm/outputs.py:22
  CompletionOutput

vllm/outputs.py:85
  RequestOutput

vllm/outputs.py:208
  PoolingRequestOutput
```

### 14.1 ModelRunnerOutput

`ModelRunnerOutput` 是 Worker / ModelRunner 返回给 Scheduler 的 batch 级结果。

它包含：

```text
req_ids；
req_id_to_index；
sampled_token_ids；
logprobs；
prompt_logprobs_dict；
pooler_output；
kv_connector_output；
ec_connector_output；
num_nans_in_logits；
cudagraph_stats；
routed_experts。
```

### 14.2 RequestOutput

`RequestOutput` 是用户最终更常见的输出对象。

它通常包含：

```text
request_id；
prompt；
prompt_token_ids；
outputs: list[CompletionOutput]；
finished；
metrics；
kv_transfer_params；
```

### 14.3 OutputProcessor

`OutputProcessor` 位于：

```text
vllm/v1/engine/output_processor.py
```

它负责：

```text
把 EngineCoreOutput / EngineCoreOutputs 转成 RequestOutput；
detokenize；
处理 stop string；
更新 stats；
在 AsyncLLM 中推送到每个请求的 queue。
```

---

## 15. 分布式系统：TP / PP / DP / EP 和通信

分布式相关模块在：

```text
vllm/distributed/
vllm/v1/executor/
vllm/config/parallel.py
```

代码证据：

```text
vllm/config/parallel.py:117
  ParallelConfig

vllm/distributed/communication_op.py:12
  tensor_model_parallel_all_reduce

vllm/distributed/communication_op.py:38
  broadcast_tensor_dict

vllm/distributed/device_communicators/all2all.py:42
  AgRsAll2AllManager
```

### 15.1 ParallelConfig

`ParallelConfig` 配置：

```text
tensor_parallel_size；
pipeline_parallel_size；
data_parallel_size；
expert_parallel；
distributed_executor_backend；
worker_cls；
ray / mp / external launcher 相关参数。
```

### 15.2 Tensor Parallel

Tensor Parallel 把模型参数和计算切分到多个设备上。

典型通信包括：

```text
all_reduce；
gather；
scatter；
broadcast。
```

代码入口：

```text
vllm/distributed/communication_op.py
```

### 15.3 Pipeline Parallel

Pipeline Parallel 把模型层切成多个 stage。

在 Worker 执行中会看到：

```text
不是 first rank 的 stage 接收 intermediate tensors；
不是 last rank 的 stage 发送 intermediate tensors；
最后 stage 通常产生 logits / output。
```

相关代码在：

```text
vllm/v1/worker/gpu_worker.py
vllm/v1/worker/gpu_model_runner.py
```

### 15.4 Data Parallel

Data Parallel 让多个 engine / rank 分担请求。

EngineCoreProc 中的 `engine_index`、`client_index`、DP coordinator、wave_complete / start_wave 等字段都服务于多 engine 调度。

### 15.5 Expert Parallel

Expert Parallel 主要用于 MoE 模型，将 expert 分布到不同 rank。

相关通信常见于：

```text
all-to-all；
expert load balancing；
EPLB；
MoE kernels。
```

代码分布在：

```text
vllm/distributed/
vllm/model_executor/layers/fused_moe/
vllm/v1/worker/gpu_model_runner.py
```

---

## 16. KV Transfer 和解耦式执行

KV transfer / disaggregated prefill 相关配置：

```text
vllm/config/kv_transfer.py
vllm/config/ec_transfer.py
vllm/distributed/kv_transfer/
vllm/distributed/ec_transfer/
```

代码证据：

```text
vllm/config/kv_transfer.py:23
  KVTransferConfig
```

它们支持：

```text
不同进程 / 节点之间传输 KV cache；
prefill 和 decode 解耦；
外部 KV load；
KV save / recv 完成通知；
invalid blocks 处理；
encoder cache transfer。
```

在 Scheduler 回收阶段，会通过 `kv_connector_output` 处理：

```text
finished_sending；
finished_recving；
invalid_block_ids；
kv_connector_stats；
kv_cache_events。
```

这类能力是 vLLM 支持大规模服务和 disaggregated serving 的基础。

---

## 17. MultiModal：多模态输入

多模态模块在：

```text
vllm/multimodal/
vllm/assets/
vllm/config/multimodal.py
```

代码证据：

```text
vllm/multimodal/parse.py:490
  MultiModalDataParser

vllm/multimodal/inputs.py:302
  MultiModalFeatureSpec

vllm/multimodal/cache.py:98
  MultiModalCache

vllm/multimodal/encoder_budget.py:44
  MultiModalBudget
```

### 17.1 多模态输入做什么

多模态输入包括：

```text
image；
audio；
video；
多模态 placeholder；
多模态 feature；
encoder 输出。
```

多模态模块负责：

```text
解析用户输入；
调用 processor；
生成 MultiModalFeatureSpec；
管理 processor cache；
估算 encoder token budget；
配合 Scheduler 调度 encoder input；
配合 Worker 执行 mm encoder；
配合 EncoderCacheManager 缓存和释放 encoder output。
```

### 17.2 和主链路的关系

多模态特征进入 `EngineCoreRequest.mm_features`。

后续流转：

```text
EngineCoreRequest.mm_features
  → Request.mm_features
  → SchedulerOutput.scheduled_encoder_inputs
  → ModelRunner._execute_mm_encoder()
  → encoder cache / decoder KV cache
```

---

## 18. LoRA：动态 Adapter 支持

LoRA 模块在：

```text
vllm/lora/
vllm/config/lora.py
```

代码证据：

```text
vllm/lora/request.py:8
  LoRARequest

vllm/lora/lora_model.py:60
  LoRAModel

vllm/lora/worker_manager.py:26
  WorkerLoRAManager

vllm/lora/layers/base_linear.py:70
  BaseLinearLayerWithLoRA
```

LoRA 负责：

```text
表示用户请求需要哪个 adapter；
加载 LoRA 权重；
管理 active adapters；
把 linear / embedding / logits / MoE layer 替换或包装为 LoRA-aware layer；
在 Worker 侧动态 add / remove / pin LoRA。
```

EngineCore 也提供 LoRA utility 转发：

```text
add_lora；
remove_lora；
list_loras；
pin_lora。
```

这些最终会转发给 `model_executor` / Worker。

---

## 19. Structured Output：结构化输出约束

结构化输出模块在：

```text
vllm/v1/structured_output/
vllm/config/structured_outputs.py
```

代码证据：

```text
vllm/v1/structured_output/__init__.py:36
  StructuredOutputManager

vllm/v1/structured_output/backend_types.py:31
  StructuredOutputGrammar

vllm/v1/structured_output/backend_types.py:99
  StructuredOutputBackend

vllm/v1/structured_output/utils.py:85
  apply_grammar_bitmask
```

它支持：

```text
JSON schema；
regex；
choice；
grammar；
Guidance / XGrammar / Outlines / LM Format Enforcer 后端。
```

主链路：

```text
请求进入 EngineCore
  → StructuredOutputManager.grammar_init()
  → Scheduler 检查 grammar 状态
  → Scheduler.get_grammar_bitmask()
  → ModelRunner.sample_tokens(grammar_output)
  → apply_grammar_bitmask()
  → Sampler 在 mask 后 logits 上采样
```

所以结构化输出不是简单后处理，而是在采样前通过 bitmask 限制可选 token。

---

## 20. Speculative Decoding：投机解码

投机解码模块在：

```text
vllm/v1/spec_decode/
vllm/config/speculative.py
```

代码证据：

```text
vllm/v1/spec_decode/llm_base_proposer.py:68
  SpecDecodeBaseProposer

vllm/v1/spec_decode/llm_base_proposer.py:502
  propose

vllm/v1/spec_decode/draft_model.py:19
  DraftModelProposer

vllm/v1/spec_decode/ngram_proposer.py:12
  NgramProposer

vllm/v1/spec_decode/metrics.py:18
  SpecDecodingStats
```

投机解码的基本思想：

```text
先用 draft 模型 / n-gram / EAGLE 等方法提出候选 tokens；
再用 target model 验证这些 tokens；
接受的 tokens 可以一次推进多个；
拒绝的 tokens 需要回退或重新采样。
```

在 vLLM 主链路中的体现：

```text
SchedulerOutput.scheduled_spec_decode_tokens
  → Worker / ModelRunner 执行验证
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler.update_from_output()
  → 计算 accepted / rejected
  → 修正 request.num_computed_tokens
  → 更新下一轮 draft_token_ids
```

EngineCore 的 `post_step()` 会在非 async scheduling 场景下从 Worker 取 draft token ids，并写回 Scheduler：

```text
model_executor.take_draft_token_ids()
  → scheduler.update_draft_token_ids()
```

---

## 21. Pooling、Embedding、Classification、Scoring

vLLM 不只支持文本生成，还支持 pooling / embedding / classification / reward / scoring 等任务。

相关对象：

```text
vllm/pooling_params.py
vllm/outputs.py
vllm/model_executor/models/
vllm/entrypoints/openai/
```

用户可见输出：

```text
PoolingRequestOutput
```

位置：`vllm/outputs.py:208`

内部执行区别：

```text
generation：
  forward → logits → sample_tokens → sampled_token_ids

pooling / embedding：
  forward → pooling_output → PoolingRequestOutput
```

在 ModelRunner 中，如果是 pooling model，forward 后可能直接返回 pooling output，而不走 token sampling。

---

## 22. Tokenizer、输入解析、Chat / Tool / Reasoning

相关模块：

```text
tokenizers/
transformers_utils/
parser/
reasoning/
tool_parsers/
renderers/
entrypoints/openai/
```

这些模块位于外层协议和输入处理附近。

职责包括：

```text
加载 Hugging Face tokenizer；
处理 chat template；
解析 OpenAI messages；
处理 tool calling；
解析 reasoning content；
处理 prompt tokenization；
处理 streaming 输出中的增量 detokenization。
```

它们不属于 `EngineCore.step()` 的核心调度执行，但决定了用户请求如何被转换成 EngineCoreRequest，以及输出如何被格式化。

---

## 23. Platform：硬件平台抽象

平台相关模块在：

```text
vllm/platforms/
```

代码证据：

```text
vllm/platforms/interface.py:134
  Platform

vllm/platforms/__init__.py:211
  resolve_current_platform_cls_qualname

vllm/platforms/cpu.py:42
  CpuPlatform

vllm/platforms/rocm.py:444
  RocmPlatform

vllm/platforms/xpu.py:103
  XPUPlatform
```

平台层负责：

```text
识别当前硬件；
判断 CUDA / ROCm / CPU / XPU / TPU 等能力；
选择默认 attention backend；
判断支持的 dtype；
提供设备 memory 查询；
决定是否支持 custom ops / CUDA graph / compile；
提供平台特定的初始化逻辑。
```

这是 vLLM 支持多硬件的关键抽象层。

---

## 24. Compilation 和 CUDA Graph

编译相关模块：

```text
vllm/compilation/
vllm/config/compilation.py
```

代码证据：

```text
vllm/compilation/backends.py:800
  VllmBackend

vllm/compilation/wrapper.py:47
  TorchCompileWithNoGuardsWrapper

vllm/compilation/cuda_graph.py:145
  CUDAGraphWrapper

vllm/compilation/piecewise_backend.py:86
  PiecewiseBackend

vllm/config/compilation.py:381
  CompilationConfig
```

这一层负责：

```text
torch.compile 接入；
FX graph 捕获与拆分；
piecewise compilation；
CUDA graph capture / replay；
编译缓存；
图级 pass / fusion；
sequence parallelism 相关变换；
降低 Python overhead 和 kernel launch overhead。
```

概念上：

```text
Scheduler / Worker 决定“跑什么”；
Compilation / CUDA Graph 优化“怎么跑得更快”。
```

---

## 25. Custom Ops、Kernels 和底层性能层

底层性能模块包括：

```text
vllm/model_executor/kernels/
vllm/model_executor/layers/
vllm/v1/attention/ops/
vllm/v1/attention/backends/
vllm/vllm_flash_attn/
vllm/_custom_ops.py
vllm/_aiter_ops.py
vllm/_xpu_ops.py
csrc/
rust/
```

这些模块负责：

```text
自定义 CUDA / HIP / XPU op 绑定；
Triton kernel 和 attention ops；
FlashAttention 相关实现；
MoE kernel；
量化 kernel；
采样 kernel；
平台特定加速算子；
Rust tokenizer / parser / rendering 等辅助能力。
```

在概念文档中不需要逐个 kernel 展开，但要理解：

```text
vLLM 的高性能不仅来自调度，
还来自 attention backend、CUDA graph、torch.compile、自定义 kernel 和硬件平台适配共同作用。
```

---

## 26. Benchmarks、Tests 和 Docs

### 26.1 Benchmarks

Benchmark 代码在：

```text
benchmarks/
```

代码证据：

```text
vllm/benchmarks/throughput.py:50
  run_vllm

vllm/benchmarks/serve.py:768
  benchmark

vllm/benchmarks/latency.py:80
  main

vllm/benchmarks/sweep/param_sweep.py:7
  ParameterSweep
```

覆盖：

```text
throughput；
latency；
serve benchmark；
startup benchmark；
multimodal processor benchmark；
参数 sweep；
plot。
```

### 26.2 Tests

测试目录在：

```text
tests/
```

代表性测试：

```text
tests/basic_correctness/test_basic_correctness.py
tests/compile/test_config.py
tests/distributed/test_comm_ops.py
tests/multimodal/test_video.py
```

测试覆盖：

```text
基础正确性；
编译配置；
分布式通信；
多模态；
entrypoints；
模型；
采样；
LoRA；
OpenAI server；
性能和 benchmark。
```

### 26.3 Docs

设计文档在：

```text
docs/design/
```

代表文档：

```text
docs/design/arch_overview.md
docs/design/attention_backends.md
docs/design/cuda_graphs.md
docs/configuration/engine_args.md
```

这些可以作为后续深入阅读索引。

---

## 27. 由浅入深的源码阅读路线

### 27.1 第一阶段：用户入口

```text
README.md
pyproject.toml
vllm/entrypoints/llm.py
vllm/entrypoints/cli/main.py
vllm/entrypoints/cli/serve.py
vllm/entrypoints/openai/
```

目标：知道用户如何调用 vLLM。

### 27.2 第二阶段：配置系统

```text
vllm/engine/arg_utils.py
vllm/config/vllm.py
vllm/config/model.py
vllm/config/cache.py
vllm/config/parallel.py
vllm/config/scheduler.py
vllm/config/compilation.py
```

目标：知道用户参数如何变成内部配置。

### 27.3 第三阶段：V1 Engine 主链路

```text
vllm/v1/engine/llm_engine.py
vllm/v1/engine/async_llm.py
vllm/v1/engine/core_client.py
vllm/v1/engine/core.py
vllm/v1/engine/output_processor.py
```

目标：理解请求进入、step 执行、输出返回。

### 27.4 第四阶段：Scheduler 和 KV cache

```text
vllm/v1/core/sched/scheduler.py
vllm/v1/core/sched/output.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/block_pool.py
vllm/v1/core/encoder_cache_manager.py
```

目标：理解 vLLM 的 continuous batching 和 KV block 管理。

### 27.5 第五阶段：Executor / Worker / ModelRunner

```text
vllm/v1/executor/abstract.py
vllm/v1/executor/uniproc_executor.py
vllm/v1/executor/multiproc_executor.py
vllm/v1/worker/gpu_worker.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/gpu_input_batch.py
```

目标：理解模型真正如何执行。

### 27.6 第六阶段：模型、attention、sampling

```text
vllm/model_executor/model_loader/
vllm/model_executor/models/registry.py
vllm/model_executor/layers/
vllm/v1/attention/
vllm/v1/sample/
vllm/sampling_params.py
```

目标：理解模型加载、attention backend、采样输出。

### 27.7 第七阶段：高级特性

```text
vllm/multimodal/
vllm/lora/
vllm/v1/structured_output/
vllm/v1/spec_decode/
vllm/distributed/
vllm/compilation/
vllm/platforms/
```

目标：理解 vLLM 的扩展能力和性能优化。

---

## 28. 关键对象速查表

| 对象 | 所在位置 | 作用 |
|---|---|---|
| `LLM` | `vllm/entrypoints/llm.py` | Python offline API 入口 |
| `EngineArgs` | `vllm/engine/arg_utils.py:417` | 用户参数聚合 |
| `VllmConfig` | `vllm/config/vllm.py:288` | 全局配置对象 |
| `LLMEngine` | `vllm/v1/engine/llm_engine.py:48` | 同步外层 Engine |
| `AsyncLLM` | `vllm/v1/engine/async_llm.py:70` | 异步外层 Engine |
| `EngineCoreClient` | `vllm/v1/engine/core_client.py:71` | 外层访问 EngineCore 的客户端抽象 |
| `EngineCore` | `vllm/v1/engine/core.py:96` | 内部执行闭环总控 |
| `EngineCoreProc` | `vllm/v1/engine/core.py:905` | 多进程后台 EngineCore 包装器 |
| `Scheduler` | `vllm/v1/core/sched/scheduler.py:69` | 请求调度和状态管理 |
| `SchedulerOutput` | `vllm/v1/core/sched/output.py:183` | 一轮执行计划 |
| `KVCacheManager` | `vllm/v1/core/kv_cache_manager.py:114` | KV cache block 管理 |
| `Executor` | `vllm/v1/executor/abstract.py:37` | Worker 执行后端抽象 |
| `Worker` | `vllm/v1/worker/gpu_worker.py:130` | 设备侧执行实体 |
| `GPUModelRunner` | `vllm/v1/worker/gpu_model_runner.py` | 准备输入并执行模型 |
| `ModelRunnerOutput` | `vllm/v1/outputs.py:234` | Worker 返回的内部执行结果 |
| `EngineCoreOutputs` | `vllm/v1/engine/__init__.py:221` | EngineCore 返回给外层的内部输出 |
| `RequestOutput` | `vllm/outputs.py:85` | 用户可见请求输出 |
| `SamplingParams` | `vllm/sampling_params.py` | 用户采样参数 |
| `Sampler` | `vllm/v1/sample/sampler.py:20` | logits 到 token 的采样器 |
| `AttentionBackend` | `vllm/v1/attention/backend.py:56` | attention backend 抽象 |
| `StructuredOutputManager` | `vllm/v1/structured_output/__init__.py:36` | 结构化输出管理 |
| `LoRARequest` | `vllm/lora/request.py:8` | LoRA 请求对象 |
| `MultiModalDataParser` | `vllm/multimodal/parse.py:490` | 多模态输入解析 |
| `Platform` | `vllm/platforms/interface.py:134` | 硬件平台抽象 |
| `VllmBackend` | `vllm/compilation/backends.py:800` | torch.compile 后端 |

---

## 29. 最小闭环总结

如果只记住一条主线：

```text
用户请求
  → entrypoints
  → LLMEngine / AsyncLLM
  → EngineCoreClient
  → EngineCore
  → Scheduler
  → Executor
  → Worker / ModelRunner
  → Sampler / Pooling
  → Scheduler.update_from_output
  → EngineCoreOutputs
  → OutputProcessor
  → RequestOutput
```

如果只记住一个核心：

```text
EngineCore.step() = schedule → execute → update → output
```

如果只记住 vLLM 性能来源：

```text
continuous batching
  + paged KV cache / prefix cache
  + 高效 attention backend
  + CUDA graph / torch.compile
  + 自定义 kernel
  + 分布式并行
  + 高效输出处理
```

如果只记住模块职责：

```text
entrypoints：外部协议入口
config：配置总线
engine：外层 Engine 和内部 EngineCore
scheduler：调度和状态账本
kv cache：显存 block 管理
executor：Worker 调用抽象
worker/model runner：真正执行模型
model_executor：模型加载和模型层
attention/sample：核心计算和采样
distributed/platforms/compilation：性能和扩展
multimodal/lora/structured_output/spec_decode：高级能力
outputs/output_processor：内部输出到用户输出
```
