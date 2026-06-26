# 10. 从配置到运行时对象的生命周期顺序是什么？

源码位置：

- `code/vllm/vllm/engine/arg_utils.py`
- `code/vllm/vllm/config/`
- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/executor/`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/model_executor/model_loader/`

本问题关注：把前面所有内容串成一条生命周期链，说明从用户参数到可执行模型，中间对象按什么顺序创建、校验和使用。

---

## 1. 一句话回答占位

占位：后续补充配置先构造，再初始化 engine / executor / worker，模型先加载，再 profile KV cache，最后 warmup / compile。

```text
用户参数
  → EngineArgs
  → VllmConfig
  → Engine / EngineCore
  → Executor
  → Worker.init_device()
  → Worker.load_model()
  → profile memory
  → initialize KV cache
  → warmup / compile / CUDA graph capture
  → ready for EngineCore.step()
```

---

## 2. 启动阶段顺序占位

```text
1. 解析用户参数；
2. 构造 EngineArgs；
3. 创建 VllmConfig；
4. 校验子配置；
5. 初始化 Engine / EngineCore；
6. 创建 Scheduler；
7. 创建 Executor；
8. 创建 Worker；
9. 初始化 device 和分布式环境；
10. 加载模型权重；
11. profile 可用 KV cache 内存；
12. 初始化 KV cache；
13. warmup / compile / CUDA graph capture；
14. 开始接收请求并执行 step。
```

---

## 3. 为什么顺序重要占位

```text
模型配置必须先于模型加载；
模型加载必须先于显存 profile；
显存 profile 必须先于 KV cache block 数确定；
KV cache 初始化必须先于真实 forward；
attention backend 和 compilation config 会影响 warmup / capture；
SchedulerConfig 和 CacheConfig 会共同限制每轮 batch 的规模。
```

---

## 4. 和主链路闭环的关系占位

启动完成后，配置影响每轮执行：

```text
SchedulerConfig
  → Scheduler.schedule()

CacheConfig
  → KVCacheManager / block_pool / Worker KV cache

ModelConfig
  → ModelRunner._model_forward()

ParallelConfig
  → Executor / Worker / distributed groups

CompilationConfig
  → cudagraph dispatch / compiled model

LoadConfig
  → 主要在启动加载阶段使用，运行时通常不再参与每轮 step。
```

---

## 5. 后续待补源码证据

占位：补充 Engine 初始化、Executor 初始化、Worker 初始化、load_model、determine_available_memory、initialize_from_config、compile_or_warm_up_model 的具体源码位置。
