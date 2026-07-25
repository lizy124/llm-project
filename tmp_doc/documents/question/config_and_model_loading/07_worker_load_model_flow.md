# 07. Worker / ModelRunner 如何触发模型加载？

源码位置：

- `code/vllm/vllm/v1/executor/uniproc_executor.py`
- `code/vllm/vllm/v1/executor/multiproc_executor.py`
- `code/vllm/vllm/v1/executor/abstract.py`
- `code/vllm/vllm/v1/worker/worker_base.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/model_executor/model_loader/__init__.py`
- `code/vllm/vllm/model_executor/model_loader/base_loader.py`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`

这个问题关注：Engine / Executor 创建 Worker 后，Worker 如何初始化 device、创建 ModelRunner、调用 `load_model()`；`GPUModelRunner.load_model()` 如何选择 model loader、初始化模型结构、读取权重、挂 LoRA / drafter / CUDAGraph wrapper；以及模型加载后为什么还要进入 memory profiling、KV cache 初始化、warmup / compile / CUDA graph capture。

---

## 1. 一句话回答

vLLM V1 里，模型加载真正发生在 **Worker 进程内的 ModelRunner**。

完整链路可以压缩成：

```text
EngineCore 创建 Executor
  → Executor 创建 WorkerWrapperBase
  → WorkerWrapperBase.init_worker()
      → 根据 parallel_config.worker_cls 实例化真实 Worker
  → Worker.init_device()
      → 设置 device / distributed / seed / memory snapshot
      → 创建 GPUModelRunner
  → Worker.load_model()
      → GPUModelRunner.load_model()
      → get_model_loader(load_config)
      → BaseModelLoader.load_model()
      → initialize_model()
      → load_weights()
      → process_weights_after_loading()
  → EngineCore._initialize_kv_caches()
      → Worker.determine_available_memory()
      → get_kv_cache_configs()
      → Worker.initialize_from_config()
      → Worker.compile_or_warm_up_model()
```

一句话说：

```text
Executor 负责把 Worker 拉起来；
Worker 负责把设备环境准备好；
ModelRunner 负责真正构造模型并加载权重；
EngineCore 在模型加载完成后再 profile 显存、分配 KV cache、做 warmup / compile。
```

---

## 2. 总览链路

按时间顺序看，启动阶段和模型加载相关的链路是：

```text
EngineCore.__init__()
  → executor_class(vllm_config)
      → Executor.__init__()
      → Executor._init_executor()
          → WorkerWrapperBase.init_worker()
          → WorkerWrapperBase.init_device()
          → WorkerWrapperBase.load_model()
  → EngineCore._initialize_kv_caches(vllm_config)
      → model_executor.get_kv_cache_specs()
      → model_executor.determine_available_memory()
      → get_kv_cache_configs()
      → model_executor.initialize_from_config(kv_cache_configs)
          → Worker.initialize_from_config()
          → Worker.compile_or_warm_up_model()
  → Scheduler(...)
```

注意两个阶段的边界：

```text
load_model 阶段：只加载模型权重，不分配最终 KV cache；
_initialize_kv_caches 阶段：先 profile 已加载模型的内存，再分配 KV cache，再 warmup / capture。
```

这两个阶段必须分开，因为 KV cache 能分多大，取决于模型权重和一次 dummy forward 的实际显存占用。

---

## 3. Executor 如何触发 Worker 加载模型

### 3.1 单进程路径：UniProcExecutor

单进程路径在 `vllm/v1/executor/uniproc_executor.py:45`。

核心逻辑：

```python
self.driver_worker = WorkerWrapperBase(rpc_rank=0)
distributed_init_method, rank, local_rank = self._distributed_args()
...
self.driver_worker.init_worker(all_kwargs=[kwargs])
self.driver_worker.init_device()
self.driver_worker.load_model()
```

位置：`uniproc_executor.py:45` 到 `uniproc_executor.py:69`。

这条路径的特点是：

```text
1. driver worker 和 engine 在同一进程；
2. init_worker / init_device / load_model 是直接函数调用；
3. load_model 完成后才返回 Executor 初始化；
4. current_platform.update_block_size_for_backend() 在模型加载后更新 block size。
```

如果开启 `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH`，会走：

```python
self.driver_worker.elastic_ep_execute("load_model")
```

否则直接：

```python
self.driver_worker.load_model()
```

### 3.2 多进程路径：MultiprocExecutor

多进程路径在 `vllm/v1/executor/multiproc_executor.py:110`。

父进程会：

```text
1. 计算 TP / PP / PCP / local_world_size；
2. 创建 MessageQueue；
3. 为每个 local_rank 调 WorkerProc.make_worker_process()；
4. wait_for_ready() 等待所有本地 Worker 初始化完成；
5. 初始化 response queues 和 monitor。
```

位置：`multiproc_executor.py:117` 到 `multiproc_executor.py:236`。

真正的 Worker 加载发生在子进程 `WorkerProc.__init__()` 中：

```python
wrapper.init_worker(all_kwargs)
self.worker = wrapper

self.worker.init_device()
if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
    self.worker.elastic_ep_execute("load_model")
else:
    self.worker.load_model()
```

位置：`multiproc_executor.py:597` 到 `multiproc_executor.py:637`。

关键点：

```text
WorkerProc 在发送 READY 之前已经完成 init_device() 和 load_model()；
所以 MultiprocExecutor.wait_for_ready() 返回时，本地 worker 进程里的模型已经加载完成。
```

READY 发送位置：`multiproc_executor.py:875` 到 `multiproc_executor.py:882`。

这避免了一个常见误解：

```text
不是父进程收到 READY 后再 RPC 调 load_model；
而是子进程完成模型加载后，才告诉父进程 READY。
```

---

## 4. WorkerWrapperBase：真实 Worker 是怎么创建的

`WorkerWrapperBase` 定义在 `vllm/v1/worker/worker_base.py:187`。

它不是实际执行模型的 Worker，而是 executor/engine 看到的包装层。

### 4.1 init_worker 做什么

入口：`worker_base.py:230`

主要步骤：

```text
1. 从 all_kwargs[self.rpc_rank] 取当前 worker 参数；
2. 保存 vllm_config；
3. 开启函数调用 tracing；
4. load_general_plugins();
5. 从 parallel_config.worker_cls 动态解析真实 worker_class；
6. 可选注入 worker_extension_cls；
7. 创建 multimodal receiver cache；
8. 在 set_current_vllm_config(vllm_config) 上下文中实例化真实 Worker。
```

关键代码：

```python
parallel_config = vllm_config.parallel_config
worker_class = resolve_obj_by_qualname(parallel_config.worker_cls)
...
with set_current_vllm_config(self.vllm_config):
    self.worker = worker_class(**kwargs)
```

位置：`worker_base.py:249` 到 `worker_base.py:313`。

这说明 Worker 类型本身也是配置驱动的，默认通常是 GPU Worker，但也可以通过 `worker_cls` 指向其它实现。

### 4.2 init_device 和 initialize_from_config 的包装

`WorkerWrapperBase.init_device()` 会把调用包在 `set_current_vllm_config` 中：

```python
with set_current_vllm_config(self.vllm_config):
    self.worker.init_device()
```

位置：`worker_base.py:321` 到 `worker_base.py:325`。

`initialize_from_config()` 也类似：

```python
kv_cache_config = kv_cache_configs[self.global_rank]
with set_current_vllm_config(self.vllm_config):
    self.worker.initialize_from_config(kv_cache_config)
```

位置：`worker_base.py:315` 到 `worker_base.py:319`。

这保证 Worker 初始化、KV cache 初始化、模型构造期间都能通过全局上下文拿到当前 `VllmConfig`。

---

## 5. GPUWorker.init_device：加载模型前先准备什么

GPU Worker 的设备初始化入口在 `vllm/v1/worker/gpu_worker.py:249`。

核心流程：

```text
1. 处理 CUDA 环境变量；
2. 根据 DP rank 修正 local_rank；
3. 设置 torch device；
4. 检查 dtype 是否被当前平台支持；
5. 初始化 distributed environment；
6. 设置随机种子；
7. 清理缓存并记录初始显存快照；
8. 根据 CacheConfig 计算 requested_memory；
9. 初始化 workspace manager；
10. 创建 GPUModelRunner V1 或 V2；
11. rank 0 上报 usage stats。
```

关键代码：

```python
self.device = torch.device(f"cuda:{self.local_rank}")
torch.accelerator.set_device_index(self.device)
visible_device_index = current_platform.logical_device_id_to_visible_device_id(self.local_rank)
self.device = torch.device(f"cuda:{visible_device_index}")
torch.accelerator.set_device_index(self.device)
current_platform.check_if_supports_dtype(self.model_config.dtype)

init_worker_distributed_environment(
    self.vllm_config,
    self.rank,
    self.distributed_init_method,
    self.local_rank,
    current_platform.dist_backend,
)

set_random_seed(self.model_config.seed)
self.init_snapshot = MemorySnapshot(device=self.device)
self.requested_memory = request_memory(init_snapshot, self.cache_config)
```

位置：`gpu_worker.py:357` 到 `gpu_worker.py:390`。

然后创建 ModelRunner：

```python
if self.use_v2_model_runner:
    self.model_runner = GPUModelRunnerV2(self.vllm_config, self.device)
else:
    self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
```

位置：`gpu_worker.py:401` 到 `gpu_worker.py:416`。

为什么必须先 `init_device()` 再 `load_model()`？

```text
因为 load_model 需要：
- 已经设置当前 CUDA device；
- 已经初始化分布式通信和 model parallel；
- 已经知道本 worker 的 rank / local_rank；
- 已经建立 workspace manager；
- 已经创建 ModelRunner；
- 已经记录初始显存状态，后续 profiling 要用。
```

---

## 6. GPUWorker.load_model：Worker 层做了什么

入口在 `vllm/v1/worker/gpu_worker.py:424`。

核心代码：

```python
with (
    self._maybe_get_memory_pool_context(tag="weights"),
    set_current_vllm_config(self.vllm_config),
    self._scoped_allocator_max_split(max_split_size_mb=20),
):
    self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
```

位置：`gpu_worker.py:424` 到 `gpu_worker.py:431`。

Worker 层做三件事：

```text
1. 把权重加载放入 tag="weights" 的内存池上下文；
2. 设置当前线程的 VllmConfig 上下文；
3. 临时把 allocator max_split_size_mb 调成 20MiB；
4. 调用 ModelRunner.load_model() 真正加载模型。
```

加载后，如果启用了 weight transfer：

```python
self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(
    self.vllm_config.weight_transfer_config,
    self.vllm_config,
    self.device,
    self.model_runner.get_model(),
)
```

位置：`gpu_worker.py:433` 到 `gpu_worker.py:439`。

这说明 `Worker.load_model()` 本身不直接读权重文件，它是模型加载的设备侧保护壳：负责内存池、配置上下文和加载后的权重传输引擎。

---

## 7. GPUModelRunner.load_model：真正加载模型

V1 `GPUModelRunner.load_model()` 在 `vllm/v1/worker/gpu_model_runner.py:5230`。

### 7.1 选择 model loader

核心代码：

```python
if load_dummy_weights:
    self.load_config.load_format = "dummy"
model_loader = get_model_loader(self.load_config)
self.model = model_loader.load_model(
    vllm_config=self.vllm_config,
    model_config=self.model_config,
)
```

位置：`gpu_model_runner.py:5249` 到 `gpu_model_runner.py:5254`。

这里的选择由 `LoadConfig.load_format` 决定。

`get_model_loader()` 在 `vllm/model_executor/model_loader/__init__.py:122`：

```python
load_format = load_config.load_format
if load_format not in _LOAD_FORMAT_TO_MODEL_LOADER:
    raise ValueError(...)
return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)
```

支持的 load format 映射包括：

```text
auto / hf / safetensors / fastsafetensors / instanttensor / npcache / pt / mistral
  → DefaultModelLoader
bitsandbytes
  → BitsAndBytesModelLoader
dummy
  → DummyModelLoader
modelexpress
  → ModelExpressModelLoader
runai_streamer
  → RunaiModelStreamerLoader
runai_streamer_sharded / sharded_state
  → ShardedStateLoader
tensorizer
  → TensorizerLoader
```

位置：`__init__.py:33` 到 `__init__.py:66`。

### 7.2 DeviceMemoryProfiler 统计权重显存

`GPUModelRunner.load_model()` 会用 `DeviceMemoryProfiler` 包住 loader：

```python
with DeviceMemoryProfiler() as m:
    time_before_load = time.perf_counter()
    ...
    self.model = model_loader.load_model(...)
    ...
    time_after_load = time.perf_counter()
self.model_memory_usage = m.consumed_memory
```

位置：`gpu_model_runner.py:5247` 到 `gpu_model_runner.py:5321`。

这个值后面会用于 `Worker.determine_available_memory()`：

```python
with memory_profiling(
    self.init_snapshot,
    weights_memory=int(self.model_runner.model_memory_usage),
) as profile_result:
```

位置：`gpu_worker.py:482` 到 `gpu_worker.py:488`。

所以模型加载阶段不仅产生 `self.model`，还会记录权重显存占用。

### 7.3 加载 LoRA、drafter、EAGLE3 辅助输出

模型本体加载后，ModelRunner 还会处理扩展能力：

```text
LoRA：如果 lora_config 存在，调用 load_lora_model();
Speculative drafter：如果有 drafter，调用 drafter.load_model(self.model);
EAGLE3：设置 auxiliary hidden state outputs；
MoE / EPLB：识别 MoE 模型并加入 EplbState。
```

对应位置：

```text
gpu_model_runner.py:5255 到 5258    LoRA
gpu_model_runner.py:5259 到 5288    drafter / drafter MoE EPLB
gpu_model_runner.py:5290             EAGLE3 auxiliary hidden states
gpu_model_runner.py:5292 到 5318    MoE / EPLB
```

### 7.4 OOM 错误处理

如果加载模型时 CUDA OOM，会补充更可操作的错误提示：

```text
Try lowering --gpu-memory-utilization,
increasing --tensor-parallel-size,
or using --quantization.
```

位置：`gpu_model_runner.py:5322` 到 `gpu_model_runner.py:5332`。

### 7.5 加载后准备通信 buffer 和多模态状态

权重加载成功后：

```python
prepare_communication_buffer_for_model(self.model)
if (drafter := getattr(self, "drafter", None)) and (
    drafter_model := getattr(drafter, "model", None)
):
    prepare_communication_buffer_for_model(drafter_model)
```

位置：`gpu_model_runner.py:5338` 到 `gpu_model_runner.py:5343`。

随后会设置：

```text
is_multimodal_pruning_enabled
requires_sequential_video_encoding
EPLB async loop
```

位置：`gpu_model_runner.py:5344` 到 `gpu_model_runner.py:5361`。

### 7.6 根据编译配置包装模型

如果使用 stock torch compile：

```python
backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
self.model.compile(fullgraph=True, backend=backend)
return
```

位置：`gpu_model_runner.py:5363` 到 `gpu_model_runner.py:5373`。

否则，vLLM 自己控制 cudagraph 行为。

根据配置可能包上：

```text
BreakableCUDAGraphWrapper
CUDAGraphWrapper
UBatchWrapper
```

位置：`gpu_model_runner.py:5377` 到 `gpu_model_runner.py:5407`。

最后调用：

```python
get_offloader().post_init()
```

位置：`gpu_model_runner.py:5408`。

---

## 8. BaseModelLoader.load_model：模型如何实例化和写入权重

所有 loader 的公共加载入口在 `vllm/model_executor/model_loader/base_loader.py:42`。

核心流程：

```python
load_device = device_config.device if load_config.device is None else load_config.device
target_device = torch.device(load_device)
with set_default_torch_dtype(model_config.dtype):
    with target_device:
        model = initialize_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )

    log_model_inspection(model)
    self.load_weights(model, model_config)
    ...
    if _has_online_quant(model):
        finalize_layerwise_processing(model, model_config)
    process_weights_after_loading(model, model_config, target_device)

return model.eval()
```

位置：`base_loader.py:42` 到 `base_loader.py:82`。

这一步可以拆成五件事：

```text
1. 决定权重加载目标 device；
2. 用 model_config.dtype 设置默认 torch dtype；
3. initialize_model() 构造空模型结构；
4. load_weights() 把 checkpoint 权重写进去；
5. 处理 online quant / attention 后处理 / kernel 格式转换；
6. 返回 eval() 模式的模型。
```

### 8.1 initialize_model 如何选模型类

`initialize_model()` 在 `vllm/model_executor/model_loader/utils.py:40`。

核心流程：

```text
1. 如果没显式传 model_class，则 get_model_architecture(model_config)；
2. 如果 vllm_config.quant_config 不为空，先 configure_quant_config；
3. 检查模型类 __init__ 签名；
4. 新式模型类接收 vllm_config 和 prefix；
5. 在 set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix) 中实例化模型；
6. record_metadata_for_reloading(model)。
```

关键代码：

```python
if model_class is None:
    model_class, _ = get_model_architecture(model_config)

if vllm_config.quant_config is not None:
    configure_quant_config(vllm_config.quant_config, model_class)

with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
    model = model_class(vllm_config=vllm_config, prefix=prefix)
    record_metadata_for_reloading(model)
    return model
```

位置：`utils.py:50` 到 `utils.py:65`。

这说明模型结构初始化已经能访问完整 `VllmConfig`，包括 cache、parallel、quant、lora、scheduler 等信息。

### 8.2 process_weights_after_loading 做什么

`process_weights_after_loading()` 在 `utils.py:100`。

它会遍历模块：

```text
1. 如果模块有 QuantizeMethodBase，调用 quant_method.process_weights_after_loading(module)；
2. 对 Attention / MLAAttention / MMEncoderAttention 调 process_weights_after_loading(model_config.dtype)；
3. 对 HpcModule 调 process_weights_after_loading(model)；
4. torchao 场景设置 reload 相关属性。
```

位置：`utils.py:101` 到 `utils.py:141`。

这一步通常用于：

```text
权重 repack；
在线量化后处理；
attention 权重格式转换；
为 kernel 运行准备内部格式。
```

---

## 9. DefaultModelLoader：默认权重加载怎么做

`DefaultModelLoader` 在 `vllm/model_executor/model_loader/default_loader.py:43`。

### 9.1 准备权重文件

`_prepare_weights()` 在 `default_loader.py:97`。

它会：

```text
1. 如果使用 ModelScope，先尝试 maybe_download_from_modelscope；
2. 判断 model_name_or_path 是否本地目录；
3. 根据 load_format 决定允许的文件模式；
4. 非本地模型通过 download_weights_from_hf 下载；
5. safetensors 场景下载 / 使用 index 文件并过滤重复文件；
6. bin / pt 场景过滤推理不需要的文件；
7. 找不到权重文件则报错。
```

load_format 到文件模式大致是：

```text
auto：优先检测 mistral consolidated*.safetensors，否则当 hf；
hf：*.safetensors / *.bin；
safetensors / fastsafetensors / instanttensor：*.safetensors；
mistral：consolidated*.safetensors；
pt：*.pt；
npcache：*.bin。
```

位置：`default_loader.py:139` 到 `default_loader.py:242`。

### 9.2 创建权重 iterator

`_get_weights_iterator()` 在 `default_loader.py:211`。

它根据文件类型和 load_format 选择 iterator：

```text
npcache → np_cache_weights_iterator
fastsafetensors → fastsafetensors_weights_iterator
instanttensor → instanttensor_weights_iterator
safetensors → safetensors_weights_iterator 或 multi_thread_safetensors_weights_iterator
pt/bin → pt_weights_iterator 或 multi_thread_pt_weights_iterator
```

位置：`default_loader.py:256` 到 `default_loader.py:319`。

### 9.3 load_weights 调模型自己的 load_weights

`DefaultModelLoader.load_weights()` 在 `default_loader.py:381`。

核心代码：

```python
self._init_ep_weight_filter(model_config)
loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

位置：`default_loader.py:425` 到 `default_loader.py:427`。

这说明具体每个权重名如何映射到模型参数，最终由模型类自己的 `load_weights()` 实现决定。

加载后还会：

```text
1. 记录加载耗时；
2. 如果开启 weights track，检查 checkpoint 中哪些权重没加载；
3. 对 online quant / postprocess quant 的参数做例外处理。
```

位置：`default_loader.py:429` 到 `default_loader.py:464`。

### 9.4 EP weight filter

如果是 MoE + expert parallel + enable_ep_weight_filter，DefaultModelLoader 会提前计算当前 rank 需要加载哪些 expert：

```text
ep_size = dp_size * pcp_size * tp_size
ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
local_expert_ids = compute_local_expert_ids(...)
```

位置：`default_loader.py:351` 到 `default_loader.py:413`。

这样可以在读取 safetensors 时跳过非本 rank 的 expert 权重，降低大 MoE 模型的存储 I/O 和内存压力。

---

## 10. 模型加载后为什么还要 determine_available_memory

模型加载完成后，EngineCore 会进入 `_initialize_kv_caches()`。

入口：`vllm/v1/engine/core.py:239`。

关键流程：

```text
1. register_all_kvcache_specs(vllm_config);
2. model_executor.get_kv_cache_specs();
3. 如果发现 non_causal KV cache spec，禁用 chunked prefill / prefix caching；
4. 如果模型有 KV cache，调用 model_executor.determine_available_memory();
5. get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory);
6. 如果 auto-fit 改了 max_model_len，同步给 workers；
7. generate_scheduler_kv_cache_config();
8. 写回 cache_config.num_gpu_blocks / block_size / kv_cache_size_tokens / max_concurrency；
9. validate_block_size();
10. model_executor.initialize_from_config(kv_cache_configs)。
```

位置：`core.py:243` 到 `core.py:321`。

### 10.1 determine_available_memory 做什么

入口：`gpu_worker.py:448`。

如果用户显式设置 `kv_cache_memory_bytes`：

```text
仍然执行一次 model_runner.profile_run()，用于编译 max_num_batched_tokens 相关路径；
但跳过自动显存估算，直接返回用户指定的 KV cache bytes。
```

位置：`gpu_worker.py:462` 到 `gpu_worker.py:480`。

否则会：

```text
1. 用 memory_profiling 包住一次 model_runner.profile_run();
2. 统计 torch peak、non-torch increase、weights_memory；
3. 如启用 CUDA graph，估算 cudagraph memory；
4. available_kv_cache_memory_bytes = requested_memory - non_kv_cache_memory - cudagraph_memory_estimate；
5. 返回可用于 KV cache 的显存。
```

位置：`gpu_worker.py:482` 到 `gpu_worker.py:606`。

这解释了为什么必须先 load model：

```text
没有 self.model，就不能跑 profile_run；
没有 model_memory_usage，就不能把权重显存从 KV cache budget 中扣掉；
没有 profile_run，就不知道 activation peak 和非 torch 内存。
```

---

## 11. initialize_from_config：KV cache 什么时候真正分配

Worker 分配 KV cache 的入口是 `GPUWorker.initialize_from_config()`。

位置：`gpu_worker.py:717`。

核心代码：

```python
self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

with self._maybe_get_memory_pool_context(tag="kv_cache"):
    self.model_runner.initialize_kv_cache(kv_cache_config)
```

位置：`gpu_worker.py:720` 到 `gpu_worker.py:743`。

这里有几个关键点：

```text
1. num_gpu_blocks 是 profile 和 get_kv_cache_configs 后才知道的；
2. KV transfer connector 要在 initialize_kv_cache 前初始化；
3. KV cache 分配放入 tag="kv_cache" 的内存池；
4. enable_return_routed_experts 时，还会初始化 routed experts capturer；
5. 需要 KV zeroing 时，会初始化 KV-zero metadata。
```

这一步才是真正的 KV cache 分配，不在 `load_model()` 阶段完成。

---

## 12. compile_or_warm_up_model：加载后如何进入可推理状态

`Executor.initialize_from_config()` 会在广播 KV cache 初始化后，再广播 warmup：

```python
self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
compilation_times = self.collective_rpc("compile_or_warm_up_model")
```

位置：`vllm/v1/executor/abstract.py:118` 到 `abstract.py:137`。

GPU Worker 的实现入口在 `gpu_worker.py:745`。

它会：

```text
1. 根据 compilation_config.compile_sizes / cudagraph_capture_sizes / compile_ranges 计算 warmup sizes；
2. 对每个 warmup size 跑 model_runner._dummy_run()；
3. kernel_warmup(self) 预热执行期 kernel；
4. 如果不是 enforce_eager，调用 model_runner.capture_model() 捕获 CUDA graph；
5. 对比 CUDA graph 实际内存和 profile 阶段估计值；
6. V2 ModelRunner 跑 warmup_kernels；
7. V1 最后一个 PP rank 预热 sampler / pooler；
8. 重置随机种子；
9. 激活 Triton JIT monitor；
10. freeze_gc_heap()，减少推理期 GC 扫描静态对象。
```

位置：`gpu_worker.py:745` 到 `gpu_worker.py:921`。

这里的顺序也很重要：

```text
先加载模型权重；
再 profile 得到 KV cache 空间；
再分配 KV cache；
最后 warmup / compile / capture。
```

因为 CUDA graph capture 和实际执行路径需要模型权重、KV cache、buffer 都已经到位。

---

## 13. 为什么 load_model 和 initialize_kv_cache 分开

这不是代码组织偶然，而是启动流程的资源依赖决定的。

```text
1. 模型权重必须先加载，才能知道 weights_memory；
2. 模型必须先能 forward，才能 profile activation peak；
3. profile 后才知道剩余多少显存能给 KV cache；
4. KV cache block 数量确定后，才能初始化 Scheduler 的 block 管理；
5. KV cache 分配完成后，才能做最终 warmup / CUDA graph capture；
6. CUDA graph capture 后，推理循环才进入稳定执行状态。
```

如果把 KV cache 分配放在 load_model 之前，会遇到两个问题：

```text
- 不知道权重和 activation 会占多少显存，容易 OOM 或浪费显存；
- 不知道模型实际 KV cache spec，尤其 hybrid attention / mamba / encoder-decoder / sliding window 模型。
```

因此 vLLM 把启动拆成：

```text
load weights
  → profile memory
  → create KV cache config
  → allocate KV cache
  → warmup / compile / capture
```

---

## 14. 单进程和多进程路径的关键差异

```text
UniProcExecutor：
  Engine 进程内直接调用 WorkerWrapperBase.init_worker/init_device/load_model。

MultiprocExecutor：
  父进程启动 WorkerProc；
  子进程内执行 init_worker/init_device/load_model；
  load_model 完成后子进程发送 READY；
  父进程 wait_for_ready 返回后才继续。
```

共同点：

```text
1. 都通过 WorkerWrapperBase 创建真实 Worker；
2. 都先 init_device，再 load_model；
3. 真正加载权重都在 ModelRunner.load_model；
4. KV cache 初始化都由 EngineCore._initialize_kv_caches 后续触发；
5. warmup/compile 都在 initialize_from_config 之后执行。
```

---

## 15. 关键对象职责边界

```text
EngineCore：
  负责总体启动顺序，尤其是模型加载后 profile、生成 KV cache config、触发 KV cache 初始化和 warmup。

Executor：
  负责创建 Worker，并把 initialize_from_config / compile_or_warm_up_model 等生命周期方法广播到 Worker。

WorkerWrapperBase：
  负责动态解析 worker_cls，实例化真实 Worker，并在 set_current_vllm_config 上下文中转发生命周期调用。

GPUWorker：
  负责设备初始化、分布式初始化、显存快照、内存池上下文、调用 ModelRunner.load_model、profile 显存、分配 KV cache。

GPUModelRunner：
  负责构造模型、选择 loader、加载权重、挂 LoRA/drafter、准备通信 buffer、包装 CUDAGraph/UBatch、执行 profile_run 和 warmup dummy run。

ModelLoader：
  负责根据 load_format 找权重文件、创建模型实例、读取 checkpoint、调用模型类 load_weights、做权重后处理。
```

---

## 16. 排查模型加载问题时看哪里

如果卡在模型加载阶段，可以按这个顺序定位：

```text
1. Worker 是否创建成功？
   看 UniProcExecutor._init_executor 或 MultiprocExecutor / WorkerProc 初始化。

2. device / distributed 是否初始化成功？
   看 GPUWorker.init_device()。

3. 是否进入 ModelRunner.load_model？
   看 GPUWorker.load_model() 和 GPUModelRunner.load_model() 日志。

4. loader 是否选对？
   看 LoadConfig.load_format 和 get_model_loader() 映射。

5. 权重文件是否找到？
   看 DefaultModelLoader._prepare_weights()。

6. 权重名是否匹配模型？
   看 DefaultModelLoader.load_weights() 和具体模型类 load_weights()。

7. OOM 发生在哪？
   load_model OOM 多半是权重加载阶段；
   determine_available_memory / profile_run OOM 多半是 dummy forward 或 CUDA graph memory；
   initialize_kv_cache OOM 多半是 KV cache block 估算过大。

8. 加载后性能路径是否按预期？
   看 CompilationConfig、cudagraph_mode、enforce_eager、capture_model、warmup sizes。
```

---

## 17. 总结

`Worker / ModelRunner` 的模型加载链路可以概括为：

```text
Executor 拉起 Worker，
Worker 准备设备和分布式环境，
ModelRunner 选择 loader 并加载模型权重，
EngineCore 再根据已加载模型 profile 显存，
最后分配 KV cache 并 warmup / compile。
```

其中最重要的边界是：

```text
load_model 只解决“模型权重在设备上”；
_initialize_kv_caches 解决“KV cache 分多少、怎么分”；
compile_or_warm_up_model 解决“推理路径预热、编译、CUDA graph capture”。
```

所以 vLLM V1 的启动不是一次性“加载模型完成就能推理”，而是一个分阶段的设备准备流程：

```text
模型权重加载完成
  ≠ KV cache 已经可用
  ≠ CUDA graph 已经捕获
  ≠ sampler / kernel 已经预热
```

只有 `load_model → determine_available_memory → initialize_from_config → compile_or_warm_up_model` 全部完成后，Worker 才真正进入稳定的 `execute_model()` 推理循环。
