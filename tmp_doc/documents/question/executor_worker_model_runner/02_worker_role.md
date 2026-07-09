# 02. Worker 在 vLLM V1 里负责什么？

源码位置：

- `code/vllm/vllm/v1/worker/worker_base.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py`
- `code/vllm/vllm/v1/executor/abstract.py`
- `code/vllm/vllm/v1/executor/uniproc_executor.py`
- `code/vllm/vllm/v1/executor/multiproc_executor.py`

本问题关注：`Worker` 在 Executor / ModelRunner 中间到底负责什么，Worker 如何初始化设备、加载模型、profile 显存、初始化 KV cache、执行 `SchedulerOutput`、处理 pipeline parallel 中间张量、管理 sleep / wake_up / profile / LoRA / shutdown，以及它和 `Executor`、`ModelRunner` 的职责边界。

---

## 0. 梳理规划

参考 Scheduler 文档的写法，本篇按“先定边界，再走主链路，再拆关键能力”的方式梳理 Worker。

要回答的问题分成 8 组：

```text
1. Worker 是哪一层？
2. WorkerBase / WorkerWrapperBase / GPU Worker 分别是什么？
3. Worker 如何被 Executor 创建？
4. Worker 初始化 device、distributed、ModelRunner 的流程是什么？
5. Worker 如何加载模型、profile 显存、初始化 KV cache、warmup / CUDA graph？
6. Worker.execute_model() 如何把 SchedulerOutput 交给 ModelRunner？
7. Worker.sample_tokens()、LoRA、profile、sleep / wake_up、shutdown 如何转发？
8. Worker 和 Executor / ModelRunner / Scheduler 的边界是什么？
```

阅读顺序建议：

```text
worker_base.py
  → gpu_worker.py: Worker.__init__ / init_device / load_model
  → gpu_worker.py: determine_available_memory / initialize_from_config / compile_or_warm_up_model
  → gpu_worker.py: execute_model / sample_tokens
  → gpu_model_runner.py: GPUModelRunner.execute_model / sample_tokens
```

本篇不会展开 `ModelRunner` 内部的输入准备、attention metadata、forward、sampling 细节，这些留给后续：

```text
03_model_runner_role.md
04_execute_model_flow.md
05_input_batch_and_state_update.md
06_prepare_inputs_and_attention_metadata.md
07_model_forward_and_logits.md
08_sampling_and_model_runner_output.md
09_worker_kv_cache_interaction.md
```

---

## 1. 一句话回答

Worker 是 vLLM V1 执行层的“设备侧承载对象”。

它不负责调度策略，也不直接决定本轮哪些请求执行；这些由 Scheduler 完成。它也不把 `SchedulerOutput` 细拆成 token ids、positions、attention metadata；这些由 ModelRunner 完成。

Worker 的核心职责是：

```text
接收 Executor 发来的 RPC；
初始化设备和分布式环境；
创建并持有 ModelRunner；
加载模型权重；
profile 可用显存；
初始化 KV cache；
做 warmup / CUDA graph capture；
执行 execute_model / sample_tokens；
管理 profile、sleep、wake_up、LoRA、weight transfer、shutdown 等设备侧生命周期能力。
```

最小链路是：

```text
EngineCore
  → model_executor.execute_model(scheduler_output)
  → Executor.collective_rpc("execute_model")
  → Worker.execute_model(scheduler_output)
  → ModelRunner.execute_model(scheduler_output, intermediate_tensors)
  → ModelRunnerOutput / IntermediateTensors / None
```

所以可以这样记：

```text
Executor 负责分发；
Worker 负责设备侧承载和生命周期；
ModelRunner 负责真正把 SchedulerOutput 变成模型执行。
```

---

## 2. Worker 在执行链路中的位置

从 EngineCore 视角看，执行层入口是 `model_executor`：

```python
self.model_executor = executor_class(vllm_config)
```

位置：`code/vllm/vllm/v1/engine/core.py:122` 到 `code/vllm/vllm/v1/engine/core.py:123`

KV cache 初始化后，EngineCore 会创建 Scheduler：

```python
kv_cache_config = self._initialize_kv_caches(vllm_config)
...
self.scheduler: SchedulerInterface = Scheduler(...)
```

位置：`code/vllm/vllm/v1/engine/core.py:132` 到 `code/vllm/vllm/v1/engine/core.py:158`

一轮执行时，EngineCore 不直接找 Worker，而是调用 Executor：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput
  → model_executor.execute_model(scheduler_output)
  → Worker.execute_model()
  → ModelRunner.execute_model()
```

Executor 的 `execute_model()` 本质上是 RPC 到 Worker：

```python
output = self.collective_rpc(
    "execute_model", args=(scheduler_output,), non_block=non_block
)
return output[0]
```

位置：`code/vllm/vllm/v1/executor/abstract.py:221` 到 `code/vllm/vllm/v1/executor/abstract.py:227`

这说明 Worker 的上游是 Executor，下游是 ModelRunner：

```text
Executor
  → Worker
  → ModelRunner
```

Worker 处在“执行分发”和“模型运行”之间。

---

## 3. WorkerBase：Worker 的统一接口

`WorkerBase` 是不同硬件 Worker 的抽象接口。

源码注释说明：

```python
class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """
```

位置：`code/vllm/vllm/v1/worker/worker_base.py:39` 到 `code/vllm/vllm/v1/worker/worker_base.py:43`

它保存了通用配置：

```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
self.load_config = vllm_config.load_config
self.parallel_config = vllm_config.parallel_config
self.scheduler_config = vllm_config.scheduler_config
self.device_config = vllm_config.device_config
self.speculative_config = vllm_config.speculative_config
self.observability_config = vllm_config.observability_config
self.kv_transfer_config = vllm_config.kv_transfer_config
self.compilation_config = vllm_config.compilation_config
```

位置：`code/vllm/vllm/v1/worker/worker_base.py:64` 到 `code/vllm/vllm/v1/worker/worker_base.py:75`

并定义了 Worker 必须实现或暴露的核心能力：

```text
init_device()
load_model()
get_kv_cache_spec()
compile_or_warm_up_model()
execute_model()
sample_tokens()
add_lora() / remove_lora() / pin_lora() / list_loras()
shutdown()
```

其中执行相关接口是：

```python
def execute_model(
    self, scheduler_output: SchedulerOutput
) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
    """If this method returns None, sample_tokens should be called immediately after
    to obtain the ModelRunnerOutput.
    ...
    """
```

位置：`code/vllm/vllm/v1/worker/worker_base.py:142` 到 `code/vllm/vllm/v1/worker/worker_base.py:151`

以及：

```python
def sample_tokens(
    self, grammar_output: GrammarOutput
) -> ModelRunnerOutput | AsyncModelRunnerOutput:
    """Should be called immediately after execute_model iff it returned None."""
```

位置：`code/vllm/vllm/v1/worker/worker_base.py:153` 到 `code/vllm/vllm/v1/worker/worker_base.py:157`

这两个注释很关键：

```text
Worker.execute_model() 可能不直接返回 ModelRunnerOutput；
如果返回 None，EngineCore / Executor 随后要调用 sample_tokens() 拿输出。
```

这个设计主要服务于：

```text
forward / logits 和 grammar bitmask / sampling 分阶段执行。
```

---

## 4. WorkerWrapperBase：Worker 进程里的懒初始化包装

`WorkerWrapperBase` 不是 Worker 本体，而是 Executor / worker process 侧的包装器。

源码注释：

```python
class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    ...
    """
```

位置：`code/vllm/vllm/v1/worker/worker_base.py:187` 到 `code/vllm/vllm/v1/worker/worker_base.py:194`

它的作用是：

```text
1. 保存 rpc_rank / global_rank；
2. 根据 parallel_config.worker_cls 动态解析 Worker 类；
3. 加载插件；
4. 注入 worker extension；
5. 创建多模态 receiver cache；
6. 真正实例化 Worker；
7. 把 initialize_from_config / init_device / execute_model 等调用转发给 Worker。
```

真正创建 Worker 的地方：

```python
with set_current_vllm_config(self.vllm_config):
    # To make vLLM config available during worker initialization
    self.worker = worker_class(**kwargs)
```

位置：`code/vllm/vllm/v1/worker/worker_base.py:311` 到 `code/vllm/vllm/v1/worker/worker_base.py:313`

执行入口转发也在 wrapper 中：

```python
def execute_model(
    self, scheduler_output: SchedulerOutput
) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
    self._apply_mm_cache(scheduler_output)

    return self.worker.execute_model(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/worker_base.py:340` 到 `code/vllm/vllm/v1/worker/worker_base.py:345`

这里还有一个多模态细节：

```python
for req_data in scheduler_output.scheduled_new_reqs:
    req_data.mm_features = mm_cache.get_and_update_features(req_data.mm_features)
```

位置：`code/vllm/vllm/v1/worker/worker_base.py:335` 到 `code/vllm/vllm/v1/worker/worker_base.py:337`

也就是说，Wrapper 在把 `SchedulerOutput` 交给 Worker 前，可能会把 EngineCore 侧传来的多模态特征替换为 worker receiver cache 中的版本。

---

## 5. GPU Worker 是什么

GPU Worker 的具体类是：

```python
class Worker(WorkerBase):
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:117`

它继承 `WorkerBase`，代表一个 GPU 设备侧执行实体。

它在 `__init__()` 中做一些轻量初始化：

```python
super().__init__(...)
...
torch.set_float32_matmul_precision(precision)
...
self.elastic_ep_executor = ElasticEPScalingExecutor(self)
...
self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
...
self.weight_transfer_engine: WeightTransferEngine | None = None
...
self.profiler: Any | None = None
...
self.use_v2_model_runner = vllm_config.use_v2_model_runner
self._pp_send_work: list[Handle] = []
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:126` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:163`

注意：这里还没有真正创建 device、加载模型、分配 KV cache。

GPU Worker 的重活分布在后续几个阶段：

```text
init_device()：初始化 CUDA device / distributed / ModelRunner；
load_model()：加载权重；
determine_available_memory()：profile 可用于 KV cache 的显存；
initialize_from_config()：分配 KV cache；
compile_or_warm_up_model()：warmup / CUDA graph capture；
execute_model()：执行 SchedulerOutput；
sample_tokens()：采样并返回 ModelRunnerOutput；
shutdown()：释放资源。
```

---

## 6. Worker 如何被 Executor 创建

### 6.1 UniProcExecutor 路径

单进程执行器创建一个 driver worker：

```python
self.driver_worker = WorkerWrapperBase(rpc_rank=0)
```

位置：`code/vllm/vllm/v1/executor/uniproc_executor.py:48`

然后准备参数：

```python
kwargs = dict(
    vllm_config=self.vllm_config,
    local_rank=local_rank,
    rank=rank,
    distributed_init_method=distributed_init_method,
    is_driver_worker=True,
    shared_worker_lock=Lock(),
)
```

位置：`code/vllm/vllm/v1/executor/uniproc_executor.py:50` 到 `code/vllm/vllm/v1/executor/uniproc_executor.py:57`

初始化顺序是：

```python
self.driver_worker.init_worker(all_kwargs=[kwargs])
self.driver_worker.init_device()
...
self.driver_worker.load_model()
```

位置：`code/vllm/vllm/v1/executor/uniproc_executor.py:62` 到 `code/vllm/vllm/v1/executor/uniproc_executor.py:68`

所以 UniProc 路径是：

```text
UniProcExecutor._init_executor()
  → WorkerWrapperBase(rpc_rank=0)
  → init_worker()
  → Worker(...)
  → init_device()
  → load_model()
```

### 6.2 MultiprocExecutor 路径

多进程执行器会创建多个 worker 进程。

类定义：

```python
class MultiprocExecutor(Executor):
    supports_pp: bool = True
```

位置：`code/vllm/vllm/v1/executor/multiproc_executor.py:103` 到 `code/vllm/vllm/v1/executor/multiproc_executor.py:104`

初始化时会根据 TP / PP / PCP 等并行度计算 world size：

```python
tp_size, pp_size, pcp_size = self._get_parallel_sizes()
assert self.world_size == tp_size * pp_size * pcp_size
```

位置：`code/vllm/vllm/v1/executor/multiproc_executor.py:117` 到 `code/vllm/vllm/v1/executor/multiproc_executor.py:123`

并创建用于广播 `SchedulerOutput`、收集 `ModelRunnerOutput` 的消息队列：

```python
# Initialize worker and set up message queues for SchedulerOutputs
# and ModelRunnerOutputs
```

位置：`code/vllm/vllm/v1/executor/multiproc_executor.py:133` 到 `code/vllm/vllm/v1/executor/multiproc_executor.py:134`

因此，多进程路径可以理解为：

```text
MultiprocExecutor
  → 多个 WorkerWrapperBase / Worker process
  → 每个 Worker 绑定一个 rank / local_rank
  → collective_rpc 广播控制调用
  → worker 侧执行 execute_model / sample_tokens
```

---

## 7. init_device：Worker 初始化设备和分布式环境

GPU Worker 的设备初始化入口是：

```python
@instrument(span_name="Init device")
def init_device(self):
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:249` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:250`

### 7.1 选择 CUDA device

如果设备类型是 CUDA，Worker 会根据 local rank / DP rank 修正 device index：

```python
self.device = torch.device(f"cuda:{self.local_rank}")
torch.accelerator.set_device_index(self.device)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:285` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:286`

并检查 dtype 是否被当前平台支持：

```python
current_platform.check_if_supports_dtype(self.model_config.dtype)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:288`

### 7.2 初始化分布式环境

Worker 会在 memory snapshot 前初始化分布式环境：

```python
init_worker_distributed_environment(
    self.vllm_config,
    self.rank,
    self.distributed_init_method,
    self.local_rank,
    current_platform.dist_backend,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:294` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:300`

源码注释说明为什么要在 snapshot 前做：

```text
Initialize the distributed environment BEFORE taking memory snapshot
This ensures NCCL buffers are allocated before we measure available memory
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:290` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:293`

`init_worker_distributed_environment()` 内部会：

```python
init_distributed_environment(...)
ensure_model_parallel_initialized(
    tensor_parallel_size,
    pipeline_parallel_size,
    prefill_context_parallel_size,
    decode_context_parallel_size,
)
ensure_ec_transfer_initialized(vllm_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:1188` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:1206`

### 7.3 设置随机种子和显存快照

Worker 设置随机种子：

```python
set_random_seed(self.model_config.seed)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:305` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:306`

然后清理缓存并记录初始显存：

```python
gc.collect()
torch.accelerator.empty_cache()
self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
self.requested_memory = request_memory(init_snapshot, self.cache_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:309` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:314`

这个 snapshot 会在 `determine_available_memory()` 中用于计算 KV cache 可用显存。

### 7.4 创建 ModelRunner

最后，Worker 创建 ModelRunner：

```python
if self.use_v2_model_runner:
    ...
    self.model_runner = GPUModelRunnerV2(self.vllm_config, self.device)
else:
    ...
    self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:327` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:341`

这说明：

```text
Worker 持有 ModelRunner；
真正的输入准备、forward、sampling 后面都交给 ModelRunner。
```

---

## 8. load_model：Worker 让 ModelRunner 加载模型权重

Worker 的 `load_model()` 很薄，核心是调用 ModelRunner：

```python
def load_model(self, *, load_dummy_weights: bool = False) -> None:
    with (...):
        self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:349` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:356`

它还会根据配置创建 weight transfer engine：

```python
if self.vllm_config.weight_transfer_config is not None:
    self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:358` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:363`

真正加载模型发生在 `GPUModelRunner.load_model()`：

```python
model_loader = get_model_loader(self.load_config)
self.model = model_loader.load_model(
    vllm_config=self.vllm_config, model_config=self.model_config
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5163` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5166`

如果启用 LoRA，会包装模型：

```python
if self.lora_config:
    self.model = self.load_lora_model(self.model, self.vllm_config, self.device)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5167` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5170`

如果启用 speculative decoding 的 drafter，也会加载 drafter model：

```python
if hasattr(self, "drafter"):
    logger.info_once("Loading drafter model...")
    if hasattr(self.drafter, "load_model"):
        self.drafter.load_model(self.model)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5171` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5174`

最后还可能做：

```text
MoE / EPLB 初始化；
prepare_communication_buffer_for_model；
多模态能力检查；
torch.compile / CUDA graph wrapper 包装；
offloader post init。
```

相关位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5200` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5318`

因此：

```text
Worker.load_model() 负责生命周期入口；
GPUModelRunner.load_model() 负责真正模型实例化和权重加载。
```

---

## 9. determine_available_memory：Worker 负责 profile 可用 KV cache 显存

Worker 通过 `determine_available_memory()` 计算可用于 KV cache 的显存：

```python
@torch.inference_mode()
def determine_available_memory(self) -> int:
    """Profiles the peak memory usage of the model to determine how much
    memory can be used for KV cache without OOMs.
    ...
    """
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:371` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:383`

如果用户显式设置了 `kv_cache_memory_bytes`，仍然会做一次 profile run 来编译模型：

```python
if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
    # still need a profile run which compiles the model for
    # max_num_batched_tokens
    self.model_runner.profile_run()
    ...
    return kv_cache_memory_bytes
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:384` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:402`

普通情况下，Worker 会执行 dummy forward profile：

```python
with memory_profiling(...) as profile_result:
    self.model_runner.profile_run()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:406` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:410`

然后计算：

```python
self.available_kv_cache_memory_bytes = (
    self.requested_memory
    - profile_result.non_kv_cache_memory
    - cudagraph_memory_estimate_applied
)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:461` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:465`

所以 profile 阶段的含义是：

```text
先估算权重、激活、非 torch 内存、CUDA graph 等非 KV 开销；
再从目标可用显存中扣掉这些开销；
剩下的才是 KV cache 可以用的空间。
```

这个值会被 EngineCore 用来生成 `KVCacheConfig`，再传回 Worker 初始化 KV cache。

---

## 10. initialize_from_config：Worker 分配 KV cache

KV cache 初始化入口：

```python
@instrument(span_name="Allocate KV cache")
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    """Allocate GPU KV cache with the specified kv_cache_config."""
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:562` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:564`

它先把 profile 后的 block 数写回本地 cache config：

```python
self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:566` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:568`

然后初始化 KV transfer：

```python
ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:570` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:575`

再让 ModelRunner 初始化 KV cache：

```python
with self._maybe_get_memory_pool_context(tag="kv_cache"):
    self.model_runner.initialize_kv_cache(kv_cache_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:577` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:578`

`GPUModelRunner.initialize_kv_cache()` 内部会：

```text
1. 保存 kv_cache_config；
2. 初始化 attention backend；
3. 初始化 Mamba backend；
4. 计算 kernel block size；
5. 创建 attention metadata builders；
6. 初始化 / 重建 InputBatch；
7. 分配 KV cache tensors；
8. 注册 KV transfer group 可访问的 KV cache。
```

对应代码：

```python
self.kv_cache_config = kv_cache_config
...
self.initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)
initialize_mamba_ssu_backend(...)
kernel_block_sizes = prepare_kernel_block_sizes(...)
self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)
self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
kv_caches = self.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7314` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7340`

如果有 KV transfer group，还会注册 KV cache：

```python
kv_transfer_group.register_kv_caches(kv_caches)
kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7351` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:7360`

所以：

```text
Scheduler 负责 KV block 的逻辑分配；
Worker / ModelRunner 负责物理 KV cache tensor 的创建和使用。
```

---

## 11. compile_or_warm_up_model：Worker 做 warmup 和 CUDA graph capture

Worker 的 warmup 入口：

```python
@instrument(span_name="Warmup (GPU)")
def compile_or_warm_up_model(self) -> CompilationTimes:
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:591` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:592`

主要做这些事：

```text
1. 根据 compilation config 准备 warmup sizes；
2. 对这些 size 做 dummy run，触发 compile / kernel 初始化；
3. 调 kernel_warmup；
4. capture CUDA graph；
5. 对 last PP rank 预热 sampler / pooler；
6. 重置随机种子；
7. 启动 JIT monitor；
8. freeze worker heap，减少 GC 影响。
```

部分关键代码：

```python
for size in sorted(warmup_sizes, reverse=True):
    logger.info("Compile and warming up model for size %d", size)
    self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
...
kernel_warmup(self)
...
if not self.model_config.enforce_eager:
    cuda_graph_memory_bytes = self.model_runner.capture_model()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:619` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:630`

last PP rank 还会预热 sampler / pooler：

```python
if self.model_runner.is_pooling_model:
    self.model_runner._dummy_pooler_run(hidden_states)
else:
    self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:725` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:728`

因此，Worker 的初始化不是“加载完模型就结束”，而是：

```text
load weights
  → profile memory
  → allocate KV cache
  → warmup kernels
  → CUDA graph capture
  → sampler / pooler buffer 预热
```

---

## 12. execute_model：Worker 如何执行 SchedulerOutput

Worker 的执行入口是：

```python
@torch.inference_mode()
def execute_model(
    self, scheduler_output: "SchedulerOutput"
) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:807` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:810`

### 12.1 等待上一次 pipeline parallel 发送完成

开头先处理上一次非阻塞 PP send：

```python
if self._pp_send_work:
    for handle in self._pp_send_work:
        handle.wait()
    self._pp_send_work = []
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:811` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:815`

含义：

```text
如果上一次这个 worker 作为非 last PP rank，异步发送了 intermediate tensors，
下一次执行前要确保发送完成，避免中间张量生命周期或通信顺序出问题。
```

### 12.2 判断是否有 forward pass

```python
forward_pass = scheduler_output.total_num_scheduled_tokens > 0
num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:817` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:819`

如果本轮没有 token，ModelRunner 可能返回空输出或仅处理 KV connector 输出。

### 12.3 pipeline parallel 接收中间张量

如果启用 PP，且当前 rank 不是 first rank，会接收上游 intermediate tensors：

```python
if forward_pass and not get_pp_group().is_first_rank:
    tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(...)
    intermediate_tensors = AsyncIntermediateTensors(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:853` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:865`

这里的 `AsyncIntermediateTensors` 支持 lazy communication sync：只有访问 `.tensors` 时才等待通信完成。

### 12.4 调用 ModelRunner.execute_model()

核心调用：

```python
output = self.model_runner.execute_model(
    scheduler_output, intermediate_tensors
)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:867` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:870`

如果 V2 pooling model 返回 None，则调用 pool：

```python
if self.use_v2_model_runner and self.model_runner.is_pooling_model and output is None:
    output = self.model_runner.pool()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:871` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:876`

如果 ModelRunner 已经返回 `ModelRunnerOutput` / `AsyncModelRunnerOutput` / `None`，Worker 直接返回：

```python
if isinstance(output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType):
    return output
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:877` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:880`

### 12.5 非 last PP rank 发送中间张量

如果返回的是 `IntermediateTensors`，说明当前不是 last PP rank，需要发给下一个 PP stage：

```python
self._pp_send_work = get_pp_group().isend_tensor_dict(
    output.tensors,
    all_gather_group=get_tp_group(),
    all_gather_tensors=all_gather_tensors,
)

return None
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:889` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:896`

因此，Worker.execute_model() 的返回值有几种情况：

| 返回值 | 含义 |
|---|---|
| `ModelRunnerOutput` | 本轮已经完成 forward / sampling / pooling，可返回 Scheduler |
| `AsyncModelRunnerOutput` | 异步输出包装，Executor 后续取结果 |
| `None` | 要么需要随后 `sample_tokens()`，要么当前 PP rank 只是发送 intermediate tensors |
| `IntermediateTensors` | ModelRunner 返回给 Worker，Worker 会发送给下一个 PP rank，不直接回 EngineCore |

---

## 13. Worker.execute_model() 和 GPUModelRunner.execute_model() 的关系

Worker 只是执行入口，真正把 `SchedulerOutput` 变成模型输入的是 `GPUModelRunner.execute_model()`。

ModelRunner 的入口：

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4043` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4048`

它首先更新 persistent batch 状态：

```python
deferred_state_corrections_fn = self._update_states(scheduler_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4085` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4086`

如果本轮无 token：

```python
if not num_scheduled_tokens:
    ...
    if not has_kv_transfer_group():
        return EMPTY_MODEL_RUNNER_OUTPUT
    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4096` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4112`

普通有 token 的流程包括：

```text
_update_states(scheduler_output)
  → _prepare_inputs(...)
  → _determine_batch_execution_and_padding(...)
  → _get_slot_mappings(...)
  → _build_attention_metadata(...)
  → _preprocess(...)
  → _model_forward(...)
  → compute_logits / pooling
  → 保存 execute_model_state
  → return None
```

关键代码：

```python
logits_indices, spec_decode_metadata = self._prepare_inputs(...)
...
attn_metadata, spec_decode_common_attn_metadata = self._build_attention_metadata(...)
...
input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_connector_output = self._preprocess(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4128` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4280`

真正 forward：

```python
model_output = self._model_forward(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4320` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4326`

如果是 last PP rank 且 generation model，会计算 logits：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4355`

然后保存临时状态：

```python
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,
    spec_decode_metadata,
    spec_decode_common_attn_metadata,
    hidden_states,
    sample_hidden_states,
    aux_hidden_states,
    ec_connector_output,
    cudagraph_stats,
    slot_mappings,
)
self.kv_connector_output = kv_connector_output
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4386` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4398`

最后返回：

```python
return None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4405`

这解释了为什么 WorkerBase 说：

```text
如果 execute_model() 返回 None，后面要马上调用 sample_tokens()。
```

因为 generation 路径中，ModelRunner 可能先完成 forward / logits，并把采样所需状态存在 `execute_model_state` 中，等 EngineCore 拿到 grammar bitmask 后再采样。

---

## 14. sample_tokens：Worker 如何触发采样

Worker 的 `sample_tokens()` 非常薄：

```python
@torch.inference_mode()
def sample_tokens(
    self, grammar_output: "GrammarOutput | None"
) -> ModelRunnerOutput | AsyncModelRunnerOutput:
    return self.model_runner.sample_tokens(grammar_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:801` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:805`

真正采样在 `GPUModelRunner.sample_tokens()`。

入口：

```python
@torch.inference_mode
def sample_tokens(
    self, grammar_output: "GrammarOutput | None"
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4422` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4425`

如果没有 `execute_model_state`，说明当前 rank 可能没有采样状态，只返回 KV connector output：

```python
if self.execute_model_state is None:
    kv_connector_output = self.kv_connector_output
    self.kv_connector_output = None
    ...
    return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4426` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4434`

否则取出 `execute_model_state`：

```python
scheduler_output, logits, spec_decode_metadata, ... = self.execute_model_state
self.execute_model_state = None
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4436` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4450`

如果有结构化输出 grammar bitmask，会先应用到 logits：

```python
if grammar_output is not None:
    apply_grammar_bitmask(
        scheduler_output, grammar_output, self.input_batch, logits
    )
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4452` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4456`

然后采样：

```python
sampler_output = self._sample(logits, spec_decode_metadata)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4458` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4459`

采样后更新 worker 侧状态：

```python
self._update_states_after_model_execute(
    sampler_output.sampled_token_ids, scheduler_output
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4461` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4463`

后续还会处理：

```text
async scheduling 下 PP sampled token 广播；
spec decode draft token proposal；
bookkeeping；
logprobs / prompt logprobs；
构造 ModelRunnerOutput。
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4464` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:4588`

因此采样阶段的职责边界是：

```text
Scheduler：生成 grammar bitmask；
Worker：调用 sample_tokens()；
ModelRunner：应用 grammar bitmask，调用 Sampler，更新 worker batch 状态，构造 ModelRunnerOutput。
```

---

## 15. profile / sleep / wake_up：Worker 的设备侧控制能力

### 15.1 profile

Worker 的 profile 入口：

```python
def profile(self, is_start: bool = True, profile_prefix: str | None = None):
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:901`

如果没配置 profiler，会直接报错：

```python
if self.profiler_config is None or self.profiler_config.profiler is None:
    raise RuntimeError(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:902` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:909`

支持 torch profiler 和 cuda profiler：

```python
if profiler_type == "torch":
    self.profiler = TorchProfilerWrapper(...)
elif profiler_type == "cuda":
    self.profiler = CudaProfilerWrapper(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:925` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:938`

### 15.2 sleep

Worker 的 sleep：

```python
def sleep(self, level: int = 1) -> None:
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:165`

它通过 allocator 释放 / offload 内存：

```python
allocator = get_mem_allocator_instance()
allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:175` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:177`

如果是 level 2，还会先保存 model buffers 到 CPU：

```python
if level == 2:
    model = self.model_runner.model
    self._sleep_saved_buffers = {
        name: buffer.cpu().clone() for name, buffer in model.named_buffers()
    }
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:168` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:173`

### 15.3 wake_up

Worker 的 wake_up：

```python
def wake_up(self, tags: list[str] | None = None) -> None:
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:187`

它会唤醒 allocator：

```python
allocator = get_mem_allocator_instance()
allocator.wake_up(tags)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:188` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:189`

如果保存过 buffers，会拷回 GPU：

```python
if len(self._sleep_saved_buffers):
    model = self.model_runner.model
    for name, buffer in model.named_buffers():
        if name in self._sleep_saved_buffers:
            buffer.data.copy_(self._sleep_saved_buffers[name].data)
    self._sleep_saved_buffers = {}
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:191` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:197`

如果唤醒 KV cache，还会通知 ModelRunner：

```python
if tags is None or "kv_cache" in tags:
    self.model_runner.post_kv_cache_wake_up()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:199` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:200`

所以 sleep / wake_up 是典型的 Worker 职责：

```text
EngineCore / Executor 发起控制；
Worker 操作设备内存和 ModelRunner 状态。
```

---

## 16. LoRA / reset cache / weight transfer：Worker 作为控制 API 转发点

Worker 提供 LoRA 控制接口：

```python
def add_lora(self, lora_request: LoRARequest) -> bool:
    return self.model_runner.add_lora(lora_request)

def remove_lora(self, lora_id: int) -> bool:
    return self.model_runner.remove_lora(lora_id)

def list_loras(self) -> set[int]:
    return self.model_runner.list_loras()

def pin_lora(self, lora_id: int) -> bool:
    return self.model_runner.pin_lora(lora_id)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:958` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:968`

reset cache 也转给 ModelRunner：

```python
def reset_mm_cache(self) -> None:
    self.model_runner.reset_mm_cache()

def reset_encoder_cache(self) -> None:
    self.model_runner.reset_encoder_cache()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:754` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:758`

weight transfer 相关能力则由 Worker 持有 `weight_transfer_engine`，并在 `update_weights()` 等方法里操作模型参数：

```python
self.weight_transfer_engine.receive_weights(...)
...
torch.accelerator.synchronize()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:1082` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:1113`

这类接口共同体现了 Worker 的控制面职责：

```text
Executor 统一调用；
Worker 找到本 rank / 本 device 的模型和缓存；
ModelRunner 或 weight_transfer_engine 做具体操作。
```

---

## 17. shutdown：Worker 释放设备侧资源

Worker shutdown：

```python
def shutdown(self) -> None:
    gc.unfreeze()

    if ensure_kv_transfer_shutdown is not None:
        ensure_kv_transfer_shutdown()
    if ensure_ec_transfer_shutdown is not None:
        ensure_ec_transfer_shutdown()
    if self.profiler is not None:
        self.profiler.shutdown()

    if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):
        weight_transfer_engine.shutdown()

    if model_runner := getattr(self, "model_runner", None):
        model_runner.shutdown()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:1141` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:1158`

释放对象包括：

```text
GC freeze 状态；
KV transfer group；
EC transfer group；
profiler；
weight transfer engine；
ModelRunner 持有的 GPU tensors、模型权重、KV cache、workspace。
```

这说明 Worker 是设备侧资源生命周期的归口。

---

## 18. Worker 和 Executor 的职责边界

### 18.1 Executor 负责什么

Executor 是分发层，负责：

```text
选择后端：uni / mp / ray / external_launcher；
创建 WorkerWrapper / Worker 进程；
实现 collective_rpc；
把 execute_model / sample_tokens / profile / sleep / LoRA 等调用广播到 workers；
聚合返回值；
管理 executor 级 failure callback。
```

Executor 抽象注释：

```python
class Executor(ABC):
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """
```

位置：`code/vllm/vllm/v1/executor/abstract.py:37` 到 `code/vllm/vllm/v1/executor/abstract.py:42`

### 18.2 Worker 负责什么

Worker 负责：

```text
本 rank / 本 device 的初始化；
本 rank / 本 device 的模型加载；
本 rank / 本 device 的 KV cache tensor 分配；
本 rank / 本 device 的 ModelRunner 调用；
本 rank / 本 device 的 profiler / sleep / wake / LoRA / shutdown。
```

### 18.3 边界总结

```text
Executor 不知道每个 batch 如何变成 attention metadata；
Worker 不决定请求调度策略；
Executor 负责“调用哪些 Worker”；
Worker 负责“这个 Worker 在自己的设备上怎么准备并执行”。
```

---

## 19. Worker 和 ModelRunner 的职责边界

### 19.1 Worker 不做什么

Worker 不直接做：

```text
根据 SchedulerOutput 更新 InputBatch；
构造 token ids / positions；
构造 slot mapping / block table；
构造 attention metadata；
调用模型 forward 的具体参数组织；
计算 logits；
调用 sampler；
构造 ModelRunnerOutput 的细节。
```

这些都在 ModelRunner。

### 19.2 Worker 做什么

Worker 做的是：

```text
创建 ModelRunner；
调用 ModelRunner.load_model()；
调用 ModelRunner.initialize_kv_cache()；
调用 ModelRunner.profile_run() / capture_model()；
调用 ModelRunner.execute_model()；
调用 ModelRunner.sample_tokens()；
转发 reset / LoRA / sleep 后修复等控制动作。
```

### 19.3 边界总结

```text
Worker 是设备侧生命周期壳；
ModelRunner 是模型执行器。
```

更形象地说：

```text
Worker 管“这张卡/这个 rank 怎么启动、怎么活、怎么停”；
ModelRunner 管“这一批 token 到底怎么喂进模型”。
```

---

## 20. Worker 和 Scheduler 的关系

Worker 不直接访问 Scheduler 的 waiting / running 队列，也不做 token budget 决策。

它只消费 Scheduler 产出的 `SchedulerOutput`。

`SchedulerOutput` 对 Worker 来说是执行计划，里面包含：

```text
scheduled_new_reqs；
scheduled_cached_reqs；
num_scheduled_tokens；
total_num_scheduled_tokens；
scheduled_spec_decode_tokens；
scheduled_encoder_inputs；
kv_connector_metadata；
ec_connector_metadata；
finished_req_ids；
```

Worker / ModelRunner 根据这些字段更新本地 batch 状态、准备输入、执行 forward。

执行后，Worker / ModelRunner 返回 `ModelRunnerOutput`，再由 Scheduler 在 `update_from_output()` 中回收。

所以关系是：

```text
SchedulerOutput：Scheduler 给 Worker 的计划
ModelRunnerOutput：Worker 给 Scheduler 的结果
```

Worker 不修改 Scheduler 的队列状态；Scheduler 根据返回结果自己更新请求状态。

---

## 21. Worker 的完整生命周期

可以把 Worker 生命周期串成一条线：

```text
Executor._init_executor()
  → WorkerWrapperBase.init_worker()
  → Worker.__init__()
  → Worker.init_device()
      → 设置 CUDA device
      → 初始化 distributed / TP / PP / DCP / EC transfer
      → 记录显存 snapshot
      → 创建 GPUModelRunner
  → Worker.load_model()
      → GPUModelRunner.load_model()
      → 加载模型权重 / LoRA / drafter / MoE / compile wrapper
  → EngineCore._initialize_kv_caches()
      → model_executor.get_kv_cache_specs()
      → model_executor.determine_available_memory()
      → 生成 KVCacheConfig
  → Executor.initialize_from_config()
      → Worker.initialize_from_config()
      → GPUModelRunner.initialize_kv_cache()
  → Worker.compile_or_warm_up_model()
      → dummy run / kernel warmup / CUDA graph capture
  → 推理循环
      → Worker.execute_model(scheduler_output)
      → GPUModelRunner.execute_model()
      → Worker.sample_tokens(grammar_output)
      → GPUModelRunner.sample_tokens()
  → 控制 API
      → profile / sleep / wake_up / reset / LoRA / weight transfer
  → shutdown
      → KV / EC transfer shutdown
      → profiler shutdown
      → weight transfer shutdown
      → model_runner shutdown
```

---

## 22. 一个完整例子：普通单 GPU generation 请求

假设使用 `UniProcExecutor`、单 GPU、非 PP、generation model。

初始化：

```text
UniProcExecutor
  → WorkerWrapperBase
  → Worker.init_device()
  → GPUModelRunner(...)
  → Worker.load_model()
  → GPUModelRunner.load_model()
  → determine_available_memory()
  → initialize_from_config()
  → compile_or_warm_up_model()
```

执行一轮：

```text
EngineCore.step()
  → Scheduler.schedule()
  → SchedulerOutput(total_num_scheduled_tokens > 0)
  → Executor.execute_model(non_block=True)
  → Worker.execute_model()
  → GPUModelRunner.execute_model()
      → _update_states()
      → _prepare_inputs()
      → _build_attention_metadata()
      → _preprocess()
      → _model_forward()
      → compute_logits()
      → 保存 execute_model_state
      → return None
  → EngineCore 得到 None
  → Scheduler.get_grammar_bitmask()
  → Executor.sample_tokens(grammar_output)
  → Worker.sample_tokens()
  → GPUModelRunner.sample_tokens()
      → apply_grammar_bitmask()
      → _sample()
      → _update_states_after_model_execute()
      → bookkeeping
      → ModelRunnerOutput
  → Scheduler.update_from_output()
```

这里 Worker 主要负责把调用送到 ModelRunner，并保证设备侧环境、缓存、profile、PP 通信等都处于正确状态。

---

## 23. 一个完整例子：Pipeline Parallel 中间 rank

如果启用 pipeline parallel，某个 Worker 不是 first rank，也不是 last rank。

它的执行路径可能是：

```text
Worker.execute_model()
  → 等待上一轮 _pp_send_work 完成
  → 从前一个 PP rank irecv_tensor_dict()
  → 包装成 AsyncIntermediateTensors
  → ModelRunner.execute_model(scheduler_output, intermediate_tensors)
  → 得到 IntermediateTensors
  → isend_tensor_dict() 发给下一个 PP rank
  → return None
```

关键代码：

```python
if forward_pass and not get_pp_group().is_first_rank:
    tensor_dict, comm_handles, comm_postprocess = get_pp_group().irecv_tensor_dict(...)
    intermediate_tensors = AsyncIntermediateTensors(...)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:853` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:865`

发送中间张量：

```python
self._pp_send_work = get_pp_group().isend_tensor_dict(output.tensors, ...)
return None
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:889` 到 `code/vllm/vllm/v1/worker/gpu_worker.py:896`

这说明：

```text
PP 中间 rank 的 Worker 不产生最终 sampled token；
它只是执行自己负责的模型 stage，并把 intermediate tensors 传下去。
```

最终 logits / sampling 通常发生在 last PP rank。

---

## 24. 一个完整例子：sleep / wake_up

如果外层调用 sleep：

```text
LLMEngine / AsyncLLM 控制 API
  → EngineCoreClient
  → EngineCore.sleep()
  → model_executor.sleep(level)
  → Executor.collective_rpc("sleep")
  → Worker.sleep(level)
```

Worker 内部：

```text
level 1：主要 offload weights；
level 2：先把 model buffers 保存到 CPU，再释放更多内存；
allocator.sleep(...)
记录释放了多少显存。
```

wake_up 时：

```text
model_executor.wake_up(tags)
  → Worker.wake_up(tags)
  → allocator.wake_up(tags)
  → 如有保存的 buffers，拷回 GPU
  → 如果唤醒 kv_cache，调用 model_runner.post_kv_cache_wake_up()
```

这说明 sleep / wake_up 的真实设备操作发生在 Worker。

---

## 25. 容易疑惑的点

### 25.1 Worker 是不是直接执行模型 forward？

不是直接执行。

Worker 调用：

```python
self.model_runner.execute_model(...)
```

真正 `_model_forward()` 在 ModelRunner 中。

### 25.2 Worker 是不是 Scheduler 的一部分？

不是。

Scheduler 生成 `SchedulerOutput`，Worker 消费它。

Worker 不维护 `waiting` / `running` 队列，也不决定 token budget。

### 25.3 Worker 和 Executor 谁更靠近设备？

Worker 更靠近设备。

```text
Executor：分发和 RPC 抽象；
Worker：具体 rank / device 上的执行实体。
```

### 25.4 Worker 和 ModelRunner 谁负责 KV cache？

两者都有关系，但层次不同：

```text
Worker：负责初始化入口和生命周期控制；
ModelRunner：负责物理 KV cache tensor、attention backend、slot mapping、forward 使用。
Scheduler：负责逻辑 block 分配。
```

### 25.5 为什么 Worker.execute_model() 可能返回 None？

常见原因有两个：

```text
1. generation 路径中，ModelRunner 先完成 forward / logits，保存 execute_model_state，等待 sample_tokens()；
2. pipeline parallel 非 last rank 只发送 intermediate tensors，不产生最终 ModelRunnerOutput。
```

### 25.6 WorkerWrapperBase 是不是 Worker？

不是。

`WorkerWrapperBase` 是 worker process 里的包装器，负责懒初始化和转发调用；真正设备逻辑在 `Worker`。

---

## 26. 从“回答问题”的角度总结

如果要问：

```text
Worker 在 vLLM V1 里负责什么？
```

可以回答：

```text
Worker 是 vLLM V1 执行层里的设备侧承载对象。

它由 Executor 创建和调用，持有具体的 ModelRunner；
负责初始化 CUDA device 和分布式环境，加载模型，profile 可用显存，
根据 EngineCore 生成的 KVCacheConfig 分配 KV cache，做 warmup / CUDA graph capture，
并在推理循环中接收 SchedulerOutput，调用 ModelRunner.execute_model() 和 sample_tokens()。

Worker 不负责调度策略，不维护 Scheduler 的 waiting / running 队列，
也不负责把用户输出 detokenize 成 RequestOutput；
它的核心职责是把 Executor 发来的执行和控制 RPC，落到本 rank / 本 device 的模型、KV cache 和 ModelRunner 上。
```

---

## 27. 最关键流程图

```text
Executor 初始化
  │
  ├─ WorkerWrapperBase.init_worker()
  │    └─ Worker(...)
  │
  ├─ Worker.init_device()
  │    ├─ 设置 CUDA device
  │    ├─ 初始化 distributed / model parallel / EC transfer
  │    ├─ 记录显存 snapshot
  │    └─ 创建 GPUModelRunner
  │
  ├─ Worker.load_model()
  │    └─ GPUModelRunner.load_model()
  │         ├─ get_model_loader()
  │         ├─ load_model()
  │         ├─ LoRA / drafter / MoE
  │         └─ compile / cudagraph wrapper
  │
  ├─ Worker.determine_available_memory()
  │    └─ GPUModelRunner.profile_run()
  │
  ├─ Worker.initialize_from_config()
  │    └─ GPUModelRunner.initialize_kv_cache()
  │         ├─ initialize_attn_backend()
  │         ├─ initialize_metadata_builders()
  │         ├─ initialize_kv_cache_tensors()
  │         └─ register_kv_caches()
  │
  ├─ Worker.compile_or_warm_up_model()
  │    ├─ dummy run
  │    ├─ kernel_warmup
  │    ├─ capture_model
  │    └─ sampler / pooler warmup
  │
  └─ 推理循环
       ├─ Worker.execute_model(SchedulerOutput)
       │    ├─ PP receive intermediate tensors
       │    ├─ GPUModelRunner.execute_model()
       │    ├─ PP send intermediate tensors
       │    └─ return ModelRunnerOutput / None
       │
       └─ Worker.sample_tokens(grammar_output)
            └─ GPUModelRunner.sample_tokens()
                 ├─ apply_grammar_bitmask
                 ├─ sampler
                 ├─ update worker-side batch state
                 └─ ModelRunnerOutput
```

---

## 28. 最关键对象关系

```text
Executor
  分发执行请求和控制请求。

WorkerWrapperBase
  worker process 包装器，负责懒初始化 Worker 和调用转发。

Worker / GPU Worker
  本 rank / 本 device 的执行实体，负责设备、模型、KV cache、profile、sleep、shutdown 等生命周期。

GPUModelRunner
  真正消费 SchedulerOutput，准备输入、attention metadata、forward、logits、sampling、ModelRunnerOutput。

SchedulerOutput
  Scheduler 发给执行层的一轮计划。

ModelRunnerOutput
  Worker / ModelRunner 返回给 Scheduler 的一轮结果。
```

---

## 29. 和其它专题的关系

本篇只回答 Worker 的定位和主职责。

后续专题可以这样衔接：

```text
01_executor_role.md
  重点看 Executor 如何选择 uni / mp / ray 后端、如何 collective_rpc。

03_model_runner_role.md
  重点看 ModelRunner 如何真正消费 SchedulerOutput。

04_execute_model_flow.md
  重点串起 Executor.execute_model → Worker.execute_model → ModelRunner.execute_model。

05_input_batch_and_state_update.md
  重点看 ModelRunner._update_states() 和 InputBatch。

06_prepare_inputs_and_attention_metadata.md
  重点看 token ids、positions、slot mapping、attention metadata。

07_model_forward_and_logits.md
  重点看 _model_forward() 和 compute_logits()。

08_sampling_and_model_runner_output.md
  重点看 sample_tokens()、Sampler、ModelRunnerOutput。

09_worker_kv_cache_interaction.md
  重点看 Worker / ModelRunner 如何使用 Scheduler 分配好的 KV blocks。

10_executor_worker_lifecycle.md
  重点看初始化、warmup、sleep、shutdown、异常传播。
```

最终最小心智模型：

```text
Worker 是设备侧执行壳；
Executor 通过 RPC 调它；
它持有 ModelRunner；
它负责把本 rank / 本 device 的模型、KV cache、profile、sleep、LoRA、shutdown 管起来；
真正的 batch 输入准备、forward、sampling 交给 ModelRunner。
```
