# 07. Worker / ModelRunner 如何触发模型加载？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/v1/worker/worker_base.py`
- `code/vllm/vllm/v1/executor/abstract.py`
- `code/vllm/vllm/v1/executor/uniproc_executor.py`
- `code/vllm/vllm/v1/executor/multiproc_executor.py`

本问题关注：Engine / Executor 创建 Worker 后，Worker 如何初始化 device、创建 ModelRunner、调用 `load_model()`，并在模型加载后进入 memory profiling、KV cache 初始化、warmup / compile。

---

## 1. 一句话回答占位

占位：后续补充 Worker 是模型加载落到具体设备的执行实体。

```text
Executor 初始化 Worker
  → Worker.init_device()
  → 创建 GPUModelRunner
  → Worker.load_model()
  → GPUModelRunner.load_model()
  → model_loader.get_model()
  → Worker.determine_available_memory()
  → Worker.initialize_from_config()
  → initialize_kv_cache()
  → compile_or_warm_up_model()
```

---

## 2. Worker 初始化阶段占位

```text
init_device：
  设置 device、分布式环境、seed、内存快照。

load_model：
  调用 ModelRunner 加载实际模型权重。

determine_available_memory：
  profile 模型占用，估算 KV cache 可用空间。

initialize_from_config：
  根据 KV cache config 分配 cache，并进行 warmup / compile。
```

---

## 3. 为什么 load_model 和 initialize_kv_cache 分开占位

后续补充：

```text
模型权重必须先加载，才能 profile 显存；
profile 后才能知道 KV cache 能分配多少 block；
KV cache 初始化后，才能进行完整 warmup / CUDA graph capture。
```

---

## 4. 后续待补源码证据

占位：补充 Executor 初始化 Worker、`Worker.load_model()`、`GPUModelRunner.load_model()`、`initialize_kv_cache()`、warmup 顺序。
