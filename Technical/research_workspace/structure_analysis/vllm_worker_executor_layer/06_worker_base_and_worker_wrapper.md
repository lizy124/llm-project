# 06. WorkerBase 与 WorkerWrapper

Worker 层有两个容易混淆的概念：

- `WorkerBase`：真实 worker 的抽象基类，定义硬件无关接口。
- `WorkerWrapperBase`：executor 直接持有和调用的 wrapper，负责延迟创建真实 worker，并转发生命周期方法。

源码文件：

- `code/vllm/vllm/v1/worker/worker_base.py`

## 1. WorkerBase 定位

`WorkerBase` 是所有具体 worker 的抽象基础。

职责：

- 保存 `vllm_config` 及其子配置。
- 保存 rank/local_rank/distributed init method。
- 定义 worker 必须实现的接口。
- 提供 LoRA、execute、KV cache 等通用抽象。

类位置：

- `code/vllm/vllm/v1/worker/worker_base.py:39`

## 2. WorkerBase 初始化字段

初始化时保存的关键配置：

- `model_config`
- `cache_config`
- `lora_config`
- `load_config`
- `parallel_config`
- `scheduler_config`
- `device_config`
- `speculative_config`
- `observability_config`
- `kv_transfer_config`
- `compilation_config`

同时记录：

- `local_rank`
- `rank`
- `distributed_init_method`
- `is_driver_worker`
- `device`
- `model_runner`

源码位置：

- `code/vllm/vllm/v1/worker/worker_base.py:45`
- `code/vllm/vllm/v1/worker/worker_base.py:58`
- `code/vllm/vllm/v1/worker/worker_base.py:64`
- `code/vllm/vllm/v1/worker/worker_base.py:74`
- `code/vllm/vllm/v1/worker/worker_base.py:81`

## 3. WorkerBase 接口

核心接口包括：

### 3.1 get_kv_cache_spec

返回模型 KV cache 规格。

- `code/vllm/vllm/v1/worker/worker_base.py:98`

GPU worker 中实际转发给 model runner：

- `code/vllm/vllm/v1/worker/gpu_worker.py:547`

### 3.2 compile_or_warm_up_model

执行模型编译、kernel warmup、CUDA graph capture 等。

- `code/vllm/vllm/v1/worker/worker_base.py:102`
- `code/vllm/vllm/v1/worker/gpu_worker.py:591`

### 3.3 init_device

初始化 device、distributed environment、model runner 等。

- `code/vllm/vllm/v1/worker/worker_base.py:114`
- `code/vllm/vllm/v1/worker/gpu_worker.py:249`
- `code/vllm/vllm/v1/worker/cpu_worker.py:107`
- `code/vllm/vllm/v1/worker/xpu_worker.py:42`

### 3.4 load_model

加载模型权重。

- `code/vllm/vllm/v1/worker/worker_base.py:138`
- `code/vllm/vllm/v1/worker/gpu_worker.py:349`

### 3.5 execute_model

执行一次 scheduler output。

- `code/vllm/vllm/v1/worker/worker_base.py:142`
- `code/vllm/vllm/v1/worker/gpu_worker.py:807`

### 3.6 sample_tokens

当 `execute_model()` 返回 `None` 时，后续调用采样。

- `code/vllm/vllm/v1/worker/worker_base.py:153`
- `code/vllm/vllm/v1/worker/gpu_worker.py:801`

### 3.7 LoRA 管理

包括：

- `add_lora`
- `remove_lora`
- `pin_lora`
- `list_loras`

源码：

- `code/vllm/vllm/v1/worker/worker_base.py:165`
- `code/vllm/vllm/v1/worker/gpu_worker.py:958`

## 4. WorkerWrapperBase 定位

`WorkerWrapperBase` 是 executor 直接操作的对象。

它不是模型执行主体，而是一个进程级包装器。

职责：

- 延迟构造真实 worker。
- 加载插件。
- 注入每个 rank 的环境变量。
- 支持 worker extension。
- 管理多模态 receiver cache。
- 转发 `init_device`、`load_model`、`execute_model` 等方法。

类位置：

- `code/vllm/vllm/v1/worker/worker_base.py:187`

## 5. update_environment_variables

`update_environment_variables()` 根据 `rpc_rank` 从环境变量列表中选出当前 worker 对应值，并更新环境变量。

用途：

- 多 worker/rank 下，每个 worker 可能需要不同环境变量。
- executor 可以统一传入 env vars，wrapper 按 rank 应用。

源码：

- `code/vllm/vllm/v1/worker/worker_base.py:222`

## 6. init_worker：真实 worker 的创建点

`WorkerWrapperBase.init_worker()` 是真实 worker 的创建入口。

主要流程：

```text
init_worker(all_kwargs)
  -> 根据 rpc_rank 取当前 rank kwargs
  -> 保存 vllm_config
  -> 加载 general plugins
  -> 从 parallel_config.worker_cls 解析 worker 类
  -> 如有 worker_extension_cls，动态扩展类
  -> 初始化 multimodal receiver cache
  -> set_current_vllm_config
  -> 实例化真实 worker
```

源码位置：

- `code/vllm/vllm/v1/worker/worker_base.py:229`
- `code/vllm/vllm/v1/worker/worker_base.py:314`

这里解释了为什么 executor 初始化时通常只知道 wrapper，而不是直接知道 GPU/CPU/XPU worker。

## 7. worker_cls 平台映射

真实 worker class 由平台配置决定。

常见映射：

- CUDA：`vllm.v1.worker.gpu_worker.Worker`
- ROCm：`vllm.v1.worker.gpu_worker.Worker`
- CPU：`vllm.v1.worker.cpu_worker.CPUWorker`
- XPU：`vllm.v1.worker.xpu_worker.XPUWorker`

源码：

- `code/vllm/vllm/platforms/cuda.py:269`
- `code/vllm/vllm/platforms/rocm.py:786`
- `code/vllm/vllm/platforms/cpu.py:153`
- `code/vllm/vllm/platforms/xpu.py:230`

注意：GPU worker 类实际名为 `Worker`，不是 `GPUWorker`。

## 8. initialize_from_config

Executor 在 KV cache config 计算完成后调用 wrapper 的 `initialize_from_config()`。

wrapper 行为：

```text
initialize_from_config(kv_cache_configs)
  -> 根据 self.global_rank 取当前 worker 的 KVCacheConfig
  -> self.worker.initialize_from_config(kv_cache_config)
```

源码：

- `code/vllm/vllm/v1/worker/worker_base.py:315`
- `code/vllm/vllm/v1/worker/worker_base.py:320`

GPU worker 实现：

- `code/vllm/vllm/v1/worker/gpu_worker.py:562`

## 9. init_device 转发

wrapper 的 `init_device()` 会包一层 `set_current_vllm_config()` 再转发给真实 worker。

源码：

- `code/vllm/vllm/v1/worker/worker_base.py:321`

真实 worker 初始化：

- GPU：`code/vllm/vllm/v1/worker/gpu_worker.py:249`
- CPU：`code/vllm/vllm/v1/worker/cpu_worker.py:107`
- XPU：`code/vllm/vllm/v1/worker/xpu_worker.py:42`

## 10. execute_model 转发与多模态 cache

`WorkerWrapperBase.execute_model()` 不只是简单转发，它会先应用多模态 receiver cache。

流程：

```text
execute_model(scheduler_output)
  -> _apply_mm_cache(scheduler_output)
  -> self.worker.execute_model(scheduler_output)
```

源码：

- `code/vllm/vllm/v1/worker/worker_base.py:340`
- `code/vllm/vllm/v1/worker/worker_base.py:345`

## 11. wrapper 与 executor 的关系

不同 executor 都以某种方式创建或持有 wrapper：

- UniProc：本进程直接创建一个 `WorkerWrapperBase`。
- Multiproc：每个 `WorkerProc` 子进程创建一个 wrapper。
- RayDistributedExecutor：Ray actor 中封装 worker wrapper。
- RayExecutorV2：Ray actor 中运行 WorkerProc-like wrapper。

## 12. wrapper 与真实 worker 的边界

可以用下面的层次理解：

```text
Executor
  -> WorkerWrapperBase
       - env vars
       - plugins
       - worker class resolution
       - MM receiver cache
       - method forwarding
  -> Real Worker
       - device
       - distributed
       - model runner
       - KV cache
       - execute/sample
```

## 13. 关键理解

1. `WorkerBase` 是真实 worker 的接口抽象。
2. `WorkerWrapperBase` 是 executor 的进程级代理对象。
3. 真实 worker class 是通过配置字符串动态解析的。
4. GPU worker 类名叫 `Worker`，不是 `GPUWorker`。
5. wrapper 负责环境和转发，真实 worker 负责设备和执行。
