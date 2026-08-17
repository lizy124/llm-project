# 07. GPU / CPU / XPU Worker 生命周期

vLLM V1 中真实执行 worker 主要包括：

- GPU/CUDA/ROCm：`vllm.v1.worker.gpu_worker.Worker`
- CPU：`vllm.v1.worker.cpu_worker.CPUWorker`
- XPU：`vllm.v1.worker.xpu_worker.XPUWorker`

其中 CPUWorker 和 XPUWorker 都继承 GPU worker 的大部分逻辑，只覆盖平台相关部分。

相关文件：

- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/cpu_worker.py`
- `code/vllm/vllm/v1/worker/xpu_worker.py`

## 1. GPU worker 总览

GPU worker 类实际命名为 `Worker`。

类位置：

- `code/vllm/vllm/v1/worker/gpu_worker.py:117`

主要职责：

- 初始化 CUDA/ROCm device。
- 初始化 distributed environment。
- 创建 GPUModelRunner。
- 加载模型。
- profile 可用显存。
- 初始化 KV cache。
- 编译、warmup、CUDA graph capture。
- 执行模型。
- 执行采样。
- 处理 pipeline parallel intermediate tensors。
- 管理 LoRA API。
- 管理 KV transfer / EC transfer。
- profiler。
- weight transfer。
- shutdown。

## 2. GPU Worker __init__

初始化阶段主要保存状态，不立即创建 device/model。

关键内容：

- 调用 `WorkerBase.__init__()`。
- 设置 float32 matmul precision。
- 创建 `ElasticEPScalingExecutor`。
- 初始化 sleep mode buffer。
- 初始化 weight transfer 状态。
- 保存 profiler config。
- 判断是否使用 V2 model runner。
- 初始化 PP 非阻塞 send 状态。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:117`
- `code/vllm/vllm/v1/worker/gpu_worker.py:134`
- `code/vllm/vllm/v1/worker/gpu_worker.py:138`
- `code/vllm/vllm/v1/worker/gpu_worker.py:145`
- `code/vllm/vllm/v1/worker/gpu_worker.py:151`
- `code/vllm/vllm/v1/worker/gpu_worker.py:161`

## 3. GPU init_device

`init_device()` 是 GPU worker 真正进入设备生命周期的入口。

流程：

```text
Worker.init_device
  -> 校验 device_type == cuda
  -> 清理 Ray 可能设置的 NCCL_ASYNC_ERROR_HANDLING
  -> DP 单节点场景修正 local_rank
  -> 设置 self.device = cuda:{local_rank}
  -> torch.accelerator.set_device_index
  -> 检查 dtype 支持
  -> 初始化 distributed environment
  -> 设置随机种子
  -> GC + empty_cache
  -> 记录 MemorySnapshot
  -> 计算 requested_memory
  -> 初始化 workspace manager
  -> 创建 GPUModelRunner 或 GPUModelRunnerV2
  -> rank 0 上报 usage stats
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:249`
- `code/vllm/vllm/v1/worker/gpu_worker.py:254`
- `code/vllm/vllm/v1/worker/gpu_worker.py:285`
- `code/vllm/vllm/v1/worker/gpu_worker.py:294`
- `code/vllm/vllm/v1/worker/gpu_worker.py:313`
- `code/vllm/vllm/v1/worker/gpu_worker.py:322`
- `code/vllm/vllm/v1/worker/gpu_worker.py:326`

## 4. DP 场景 local_rank 修正

CUDA 后端中，如果：

- 不是 Ray executor。
- 不是 external launcher。
- DP backend 不是 Ray。
- `nnodes_within_dp == 1`。

则会把 DP local rank 纳入 `local_rank`：

```text
local_rank += dp_local_rank * (PP * TP)
```

目的是同节点多个 DP rank 映射到不同 GPU 段。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:251`
- `code/vllm/vllm/v1/worker/gpu_worker.py:261`
- `code/vllm/vllm/v1/worker/gpu_worker.py:266`
- `code/vllm/vllm/v1/worker/gpu_worker.py:272`
- `code/vllm/vllm/v1/worker/gpu_worker.py:285`

## 5. 分布式环境初始化

GPU worker 在拿内存快照前初始化 distributed environment。

原因：NCCL buffer 等分布式通信开销要计入非 KV cache 显存，避免 KV cache 分配过量。

调用链：

```text
Worker.init_device
  -> init_worker_distributed_environment
      -> init_distributed_environment
      -> ensure_model_parallel_initialized
      -> 初始化 EC transfer
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:290`
- `code/vllm/vllm/v1/worker/gpu_worker.py:294`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1164`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1188`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1197`
- `code/vllm/vllm/distributed/parallel_state.py:1516`

## 6. load_model

Worker 层 `load_model()` 主要做上下文和 memory pool 管理，实际模型加载交给 model runner。

流程：

```text
Worker.load_model
  -> _maybe_get_memory_pool_context(tag="weights")
  -> set_current_vllm_config
  -> 临时设置 allocator max_split_size_mb=20
  -> model_runner.load_model(load_dummy_weights=...)
  -> 如配置 weight_transfer_config，创建 weight transfer engine
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:349`
- `code/vllm/vllm/v1/worker/gpu_worker.py:350`
- `code/vllm/vllm/v1/worker/gpu_worker.py:356`
- `code/vllm/vllm/v1/worker/gpu_worker.py:358`

ModelRunner V1 加载：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5142`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5163`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5167`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5171`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5231`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5248`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:5273`

ModelRunner V2 加载：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:274`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:372`

## 7. determine_available_memory

该方法估算可用于 KV cache 的显存。

两条路径：

### 7.1 显式 kv_cache_memory_bytes

如果用户显式设置 `kv_cache_memory_bytes`：

- 仍执行 `model_runner.profile_run()`。
- 目的不是估算显存，而是编译最大 batch token 形状。
- 直接返回配置值。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:371`
- `code/vllm/vllm/v1/worker/gpu_worker.py:384`

### 7.2 自动估算

自动模式：

```text
memory_profiling
  -> model_runner.profile_run()
  -> 记录 torch peak
  -> 记录 non-torch increase
  -> 记录 weights memory
  -> 可额外估算 CUDA graph memory
  -> available_kv_cache_memory_bytes = requested_memory - non_kv_cache_memory - cudagraph_estimate
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:404`
- `code/vllm/vllm/v1/worker/gpu_worker.py:524`

ModelRunner profile：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6227`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6286`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6290`

## 8. get_kv_cache_spec

Worker 层直接转发给 model runner：

```text
Worker.get_kv_cache_spec
  -> model_runner.get_kv_cache_spec()
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:547`

V1 ModelRunner：

- 遍历 forward context 中的 attention layers。
- 对 KV sharing target layer 做映射。
- 跳过不需要 KV cache 的模块。
- 返回 layer name 到 `KVCacheSpec` 的 dict。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7459`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7473`

V2 ModelRunner：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:403`

## 9. initialize_from_config：KV cache 初始化

Worker 层流程：

```text
Worker.initialize_from_config(kv_cache_config)
  -> 写回 cache_config.num_gpu_blocks
  -> ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
  -> 在 kv_cache memory pool 中调用 model_runner.initialize_kv_cache(kv_cache_config)
  -> 可初始化 routed experts capturer
  -> 如需要 KV cache zeroing，初始化 zero metadata
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:562`
- `code/vllm/vllm/v1/worker/gpu_worker.py:570`
- `code/vllm/vllm/v1/worker/gpu_worker.py:575`

注意：KV transfer 初始化放在 KV cache 初始化前，因为 connector 依赖 `kv_cache_config`，且可能影响 KV cache group/layout。

ModelRunner V1 KV cache 初始化：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7303`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7220`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7351`

V2：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:406`

## 10. compile_or_warm_up_model

Worker 层负责模型编译、warmup、CUDA graph capture 等。

流程包括：

- 对需要编译但非 cudagraph capture size 的 batch size 做 dummy run。
- LoRA warmup 后移除 dummy LoRA。
- kernel warmup。
- 如果不是 enforce eager，调用 `model_runner.capture_model()`。
- V2 runner warmup execute/sample 相关 Triton kernel。
- V1 last PP rank 对 sampler/pooler 做 dummy run。
- 重置随机种子。
- 激活 Triton JIT monitor。
- freeze GC heap。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:591`

ModelRunner V1 capture：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:6583`

ModelRunner V2 capture：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:690`

## 11. execute_model 与 sample_tokens

Worker 层 execute：

```text
Worker.execute_model(scheduler_output)
  -> 等待上一轮 PP send 完成
  -> 如启用 PP 且非 first rank，接收 intermediate tensors
  -> model_runner.execute_model(scheduler_output, intermediate_tensors)
  -> 如果输出是 ModelRunnerOutput / AsyncModelRunnerOutput / None，直接返回
  -> 如果输出是 IntermediateTensors，发送给下一 PP rank并返回 None
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:807`
- `code/vllm/vllm/v1/worker/gpu_worker.py:811`
- `code/vllm/vllm/v1/worker/gpu_worker.py:853`
- `code/vllm/vllm/v1/worker/gpu_worker.py:867`
- `code/vllm/vllm/v1/worker/gpu_worker.py:871`
- `code/vllm/vllm/v1/worker/gpu_worker.py:882`
- `code/vllm/vllm/v1/worker/gpu_worker.py:896`

Worker 层 sample：

```text
Worker.sample_tokens(grammar_output)
  -> model_runner.sample_tokens(grammar_output)
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:801`

## 12. profiler

Worker `profile()` 支持：

- `torch` profiler：CPU + CUDA。
- `cuda` profiler。

start 时为 trace name 加 rank suffix。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:901`

profile annotation：

- `code/vllm/vllm/v1/worker/gpu_worker.py:775`

## 13. shutdown

GPU worker shutdown 会清理：

- KV transfer。
- EC transfer。
- profiler。
- weight transfer engine。
- model runner。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:1141`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1145`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1147`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1149`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1152`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1157`

## 14. CPUWorker

类位置：

- `code/vllm/vllm/v1/worker/cpu_worker.py:33`

CPUWorker 继承 GPU worker，但覆盖：

- CPU NUMA/内存节点初始化。
- CPU 设备初始化。
- CPU distributed backend。
- CPUModelRunner。
- CPU 内存 profiling。
- CPU profiler。
- 不支持 sleep/wake。

### 14.1 __init__

流程：

- 获取可见 memory node 和 allowed CPU list。
- 选择第一个 CPU core 的 NUMA node。
- 初始化 CPU memory env。
- 根据 `gpu_memory_utilization` 计算 CPU backend 计划保留内存。
- 显式/自动 KV cache memory 校验。
- 调用父类初始化。
- 禁用 custom all reduce。
- 如果 profiler 为 torch，创建 CPU-only profiler。

源码：

- `code/vllm/vllm/v1/worker/cpu_worker.py:45`
- `code/vllm/vllm/v1/worker/cpu_worker.py:60`
- `code/vllm/vllm/v1/worker/cpu_worker.py:67`
- `code/vllm/vllm/v1/worker/cpu_worker.py:93`
- `code/vllm/vllm/v1/worker/cpu_worker.py:95`

### 14.2 init_device

流程：

```text
CPUWorker.init_device
  -> self.device = cpu
  -> Linux 下检查 tcmalloc / libiomp
  -> 替换 torch.set_num_threads 为 warning no-op
  -> 设置 VLLM_DIST_IDENT
  -> init_worker_distributed_environment(backend=current_platform.dist_backend)
  -> 设置随机种子
  -> 创建 CPUModelRunner 或 CPUModelRunnerV2
```

源码：

- `code/vllm/vllm/v1/worker/cpu_worker.py:107`
- `code/vllm/vllm/v1/worker/cpu_worker.py:110`
- `code/vllm/vllm/v1/worker/cpu_worker.py:139`
- `code/vllm/vllm/v1/worker/cpu_worker.py:150`
- `code/vllm/vllm/v1/worker/cpu_worker.py:160`

### 14.3 determine_available_memory

CPU 版本使用 CPU node memory/RSS 估算。

源码：

- `code/vllm/vllm/v1/worker/cpu_worker.py:180`

### 14.4 compile_or_warm_up_model / profile

源码：

- `code/vllm/vllm/v1/worker/cpu_worker.py:239`
- `code/vllm/vllm/v1/worker/cpu_worker.py:252`

## 15. CPUModelRunner

CPUModelRunner 继承旧版 GPUModelRunner，通过 wrapper/no-op 替换 CUDA 相关 API。

关键点：

- 初始化时把 torch accelerator 相关 API 置为 no-op。
- 禁用 CUDA graph、cascade attention。
- 将 CpuGpuBuffer 的 gpu 指向 cpu tensor。
- 替换部分 Triton kernel 为 CPU fallback。
- `load_model()` 使用 `get_model(vllm_config=...)`。

源码：

- `code/vllm/vllm/v1/worker/cpu_model_runner.py:22`
- `code/vllm/vllm/v1/worker/cpu_model_runner.py:23`
- `code/vllm/vllm/v1/worker/cpu_model_runner.py:33`
- `code/vllm/vllm/v1/worker/cpu_model_runner.py:39`
- `code/vllm/vllm/v1/worker/cpu_model_runner.py:62`
- `code/vllm/vllm/v1/worker/cpu_model_runner.py:97`
- `code/vllm/vllm/v1/worker/cpu_model_runner.py:119`
- `code/vllm/vllm/v1/worker/cpu_model_runner.py:127`
- `code/vllm/vllm/v1/worker/cpu_model_runner.py:146`

## 16. XPUWorker

类位置：

- `code/vllm/vllm/v1/worker/xpu_worker.py:24`

XPUWorker 继承 GPU worker，主要覆盖：

- XPU 设备初始化。
- CCL 环境变量。
- XPU memory snapshot。
- XPUModelRunner。
- XPU profiler activity。

初始化：

- `code/vllm/vllm/v1/worker/xpu_worker.py:27`

### 16.1 init_device

流程：

```text
XPUWorker.init_device
  -> DP 模式调整 local_rank
  -> self.device = xpu:{local_rank}
  -> torch.accelerator.set_device_index
  -> 检查 dtype
  -> empty_cache
  -> 设置 CCL / LOCAL_RANK 环境变量
  -> 初始化 distributed
  -> XCCL all_reduce warmup
  -> 设置随机种子
  -> memory snapshot
  -> 初始化 workspace manager
  -> 创建 XPUModelRunner 或 XPUModelRunnerV2
```

源码：

- `code/vllm/vllm/v1/worker/xpu_worker.py:42`
- `code/vllm/vllm/v1/worker/xpu_worker.py:45`
- `code/vllm/vllm/v1/worker/xpu_worker.py:71`
- `code/vllm/vllm/v1/worker/xpu_worker.py:87`
- `code/vllm/vllm/v1/worker/xpu_worker.py:95`
- `code/vllm/vllm/v1/worker/xpu_worker.py:104`
- `code/vllm/vllm/v1/worker/xpu_worker.py:118`
- `code/vllm/vllm/v1/worker/xpu_worker.py:125`
- `code/vllm/vllm/v1/worker/xpu_worker.py:129`

### 16.2 profile

XPU profiler 使用 CPU + XPU activities。

源码：

- `code/vllm/vllm/v1/worker/xpu_worker.py:139`

## 17. XPUModelRunner

XPUModelRunner 通过替换 torch.cuda API 到 torch.xpu API 复用 GPUModelRunner。

包括替换：

- `Stream`
- `current_stream`
- `stream`
- `mem_get_info`
- `Event`
- `set_stream`
- XPU graph 支持时替换 graph/CUDAGraph/graph_pool_handle。

源码：

- `code/vllm/vllm/v1/worker/xpu_model_runner.py:15`
- `code/vllm/vllm/v1/worker/xpu_model_runner.py:29`
- `code/vllm/vllm/v1/worker/xpu_model_runner.py:41`

## 18. 生命周期总览

```text
WorkerWrapperBase.init_worker
  -> 实例化 Worker / CPUWorker / XPUWorker

Worker.init_device
  -> 设备初始化
  -> 分布式初始化
  -> 创建 model runner

Worker.load_model
  -> model_runner.load_model

Worker.get_kv_cache_spec
  -> model_runner.get_kv_cache_spec

Worker.determine_available_memory
  -> model_runner.profile_run
  -> 计算 KV cache 可用容量

Worker.initialize_from_config
  -> ensure_kv_transfer_initialized
  -> model_runner.initialize_kv_cache

Worker.compile_or_warm_up_model
  -> dummy run / kernel warmup / graph capture

Worker.execute_model
  -> PP recv
  -> model_runner.execute_model
  -> PP send or return output

Worker.sample_tokens
  -> model_runner.sample_tokens

Worker.shutdown
  -> transfer/profiler/model_runner cleanup
```

## 19. 关键理解

1. GPU worker 类名叫 `Worker`。
2. CPU/XPU 继承 GPU worker 是为了复用绝大多数执行逻辑。
3. `init_device()` 在内存快照前初始化 distributed，是为了把 NCCL 等非 KV 显存算进去。
4. `determine_available_memory()` 是 KV cache 容量决策的关键输入。
5. `initialize_from_config()` 是 KV cache 真正分配点。
6. Worker 处理设备和生命周期，ModelRunner 处理真实执行。
