# 10. Executor / Worker / ModelRunner 的生命周期如何管理？

源码位置：

- `code/vllm/vllm/v1/executor/abstract.py`
- `code/vllm/vllm/v1/executor/uniproc_executor.py`
- `code/vllm/vllm/v1/executor/multiproc_executor.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/worker_base.py`
- `code/vllm/vllm/v1/engine/core.py`

本问题关注：Executor / Worker / ModelRunner 从初始化、KV cache 构建、模型 warmup、profile、sleep / wake_up，到 shutdown、异常监控和恢复，是如何组织成完整生命周期闭环的。

---

## 1. 一句话回答

vLLM V1 的生命周期可以压缩成：

```text
Executor 创建 Worker
  → Worker init_device()
  → Worker.load_model()
  → EngineCore 初始化 KV cache
  → Worker compile / warmup
  → 正常 execute_model / sample_tokens 循环
  → profile / sleep / wake_up / control APIs
  → shutdown / failure callback / process cleanup
```

更细一点说：

```text
EngineCore 负责协调生命周期阶段；
Executor 负责把生命周期控制广播给所有 Worker；
Worker / ModelRunner 负责真正的设备初始化、模型加载、KV cache 分配、profiling 和资源释放。
```

---

## 2. 生命周期总览

可以按时间顺序拆成六段：

```text
1. 初始化前准备
2. Worker 启动与设备初始化
3. 模型加载与 KV cache 初始化
4. warmup / compile / profiling
5. 运行期控制：sleep / wake_up / profile / utility APIs
6. shutdown / 异常处理 / 资源回收
```

下面按这六段展开。

---

## 3. 初始化前准备：Executor 是怎么选出来的

Executor 的具体实现由配置决定：

```python
Executor.get_class(vllm_config)
```

位置：`abstract.py:47` 到 `abstract.py:92`

它会根据 `parallel_config.distributed_executor_backend` 选择：

```text
UniProcExecutor
MultiprocExecutor
RayDistributedExecutor
RayExecutorV2
ExecutorWithExternalLauncher
```

这一步决定了生命周期的通信方式：

```text
- 单进程：本地直接调用；
- 多进程：MessageQueue + worker process；
- Ray：Ray actor + compiled DAG；
- external_launcher：外部进程统一调度。
```

---

## 4. Worker 启动与设备初始化

### 4.1 Worker 是如何被创建的

在多进程路径里，`MultiprocExecutor` 会启动 `WorkerProc`，再在子进程里创建真实 Worker。

在单进程路径里，`UniProcExecutor` 直接创建 `WorkerWrapperBase`，再初始化真实 Worker。

#### UniProcExecutor

初始化逻辑：

```text
1. 创建 driver_worker；
2. 计算 distributed_init_method / rank / local_rank；
3. init_worker();
4. init_device();
5. load_model();
```

位置：`uniproc_executor.py:45` 到 `uniproc_executor.py:70`

#### MultiprocExecutor

多进程初始化中会：

```text
1. 父进程创建 distributed_init_method 和 RPC broadcast queue；
2. 父进程启动 WorkerProc；
3. 子进程 WorkerProc.__init__ 中创建 WorkerWrapperBase，并执行 init_worker / init_device / load_model；
4. 子进程初始化 message queues 后发送 READY；
5. 父进程等待所有 worker ready，启动 worker monitor，并等待 queues ready。
```

位置：`multiproc_executor.py:110` 到 `multiproc_executor.py:236`，以及 `multiproc_executor.py:593` 到 `multiproc_executor.py:653`

---

### 4.2 init_device() 做什么

`WorkerBase.init_device()` 是设备初始化接口，GPU 实现由 `GPUWorker.init_device()` 完成。

关键步骤：

```text
1. 设置当前 device；
2. 初始化分布式环境；
3. 设置随机种子；
4. 清理缓存；
5. 建立 workspace manager；
6. 创建 ModelRunner；
7. 报告 usage stats。
```

位置：`gpu_worker.py:249` 到 `gpu_worker.py:346`

其中最关键的是：

```python
init_worker_distributed_environment(...)
```

位置：`gpu_worker.py:294` 到 `gpu_worker.py:300`

它会：

```text
- 初始化 distributed environment；
- 初始化 model parallel；
- 初始化 EC transfer；
- 设置 custom all-reduce；
- 配置超时。
```

### 4.3 为什么先 init_device 再 load_model

因为模型加载需要：

```text
- 已知 device；
- 已初始化分布式环境；
- 已配置 memory allocator / workspace；
- 已创建 model runner。
```

所以 `init_device()` 是“设备侧环境搭建”，`load_model()` 才是“模型权重真正装载”。

---

## 5. 模型加载：load_model()

### 5.1 GPUWorker.load_model()

```python
with (
    self._maybe_get_memory_pool_context(tag="weights"),
    set_current_vllm_config(self.vllm_config),
    self._scoped_allocator_max_split(max_split_size_mb=20),
):
    self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
```

位置：`gpu_worker.py:349` 到 `gpu_worker.py:357`

这说明模型加载期间还会做：

```text
- 使用特定内存池；
- 设置当前 VllmConfig；
- 调整 allocator 分裂策略；
- 让 ModelRunner 真正加载权重。
```

### 5.2 载入后可能还要初始化 weight transfer engine

如果配置了 `weight_transfer_config`：

```python
self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(...)
```

位置：`gpu_worker.py:358` 到 `gpu_worker.py:364`

这说明模型加载之后，Worker 还可能参与后续权重热更新。

---

## 6. KV cache 初始化：initialize_from_config()

### 6.1 EngineCore 先做 KV cache 配置推导

在 `EngineCore._initialize_kv_caches()` 中：

```python
kv_cache_specs = self.model_executor.get_kv_cache_specs()
available_gpu_memory = self.model_executor.determine_available_memory()
kv_cache_configs = get_kv_cache_configs(...)
self.model_executor.initialize_from_config(kv_cache_configs)
```

位置：`engine/core.py:239` 到 `engine/core.py:348`

这说明：

```text
EngineCore 先根据 Worker 提供的 KV spec 和可用显存，算出 KV cache 配置；
然后再把配置下发给 Worker 初始化真正的 KV cache。
```

### 6.2 Worker 侧 initialize_from_config()

`GPUWorker.initialize_from_config()` 负责：

```text
1. 更新 num_gpu_blocks；
2. 初始化 KV transfer；
3. 进入 KV cache memory pool；
4. 让 ModelRunner 初始化 KV cache；
5. 初始化 routed experts capturer；
6. 初始化 KV zero metadata。
```

位置：`gpu_worker.py:562` 到 `gpu_worker.py:590`

### 6.3 为什么要先初始化 KV transfer

因为 KV cache 初始化前，连接器可能就需要知道：

```text
- KV cache 配置；
- block size；
- group layout；
- 何时可开始 transfer。
```

所以 `ensure_kv_transfer_initialized(vllm_config, kv_cache_config)` 必须在 `initialize_kv_cache()` 之前。

---

## 7. compile / warmup

### 7.1 初始化之后立即 warmup

在 `Executor.initialize_from_config()` 里会紧接着调用：

```python
compilation_times = self.collective_rpc("compile_or_warm_up_model")
```

位置：`abstract.py:118` 到 `abstract.py:127`

### 7.2 GPUWorker.compile_or_warm_up_model()

这个过程主要包含：

```text
1. 编译 / warmup 不同 batch size；
2. kernel warmup；
3. capture CUDA graph；
4. 预热 sampler / logits buffer；
5. 可选启动 spec decode drafter warmup。
```

位置：`gpu_worker.py:591` 起

这说明 “模型能跑” 和 “模型已 warmup 到适合 serving” 是两件事。

### 7.3 EngineCore 为什么要传播 compilation time

`Executor.initialize_from_config()` 会把 worker 里记录的 compilation time 回写到 `vllm_config.compilation_config`。

原因是：

```text
- compile 在 worker 进程里发生；
- 主进程的 config 需要同步这个统计；
- 以便日志和后续判断使用。
```

---

## 8. profiling

### 8.1 Executor 层的 profile()

`Executor.profile()` 只是广播控制：

```python
self.collective_rpc("profile", args=(is_start, profile_prefix))
```

位置：`abstract.py:256` 到 `abstract.py:257`

### 8.2 GPUWorker.profile()

GPU 侧 profiling 逻辑：

```text
1. 检查 profiler_config；
2. 生成 rank suffix / trace name；
3. 首次 start 时创建 profiler wrapper；
4. start / stop profiling；
5. 记录日志。
```

位置：`gpu_worker.py:901` 到 `gpu_worker.py:953`

### 8.3 profile 的生命周期特点

```text
- 不是每次调用都重建 profiler；
- 第一次 start 时创建 wrapper；
- 后续 start/stop 复用同一个实例；
- stop 时若没有 profiler，会 warning。
```

### 8.4 profile 和 dummy run / warmup 的关系

`GPUWorker.determine_available_memory()` 和 `compile_or_warm_up_model()` 都可能涉及 dummy run、profile run，但 profiling API 本身是控制面能力，不等同于 warmup。

---

## 9. sleep / wake_up

sleep / wake_up 是生命周期里最容易被误解的一组接口。

### 9.1 Executor 统一广播

```python
self.collective_rpc("sleep", kwargs=dict(level=level))
self.collective_rpc("wake_up", kwargs=dict(tags=tags))
```

位置：`abstract.py:318` 到 `abstract.py:356`

Executor 会维护：

```text
is_sleeping
sleeping_tags = {"weights", "kv_cache"}
```

### 9.2 sleep 的层级

Executor 的 `sleep(level)` 会把 `level` 转发给 worker。

GPUWorker 中：

```python
if level == 2:
    self._sleep_saved_buffers = {name: buffer.cpu().clone() ...}
allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
```

位置：`gpu_worker.py:165` 到 `gpu_worker.py:185`

这意味着：

```text
level 1：主要 offload weights；
level 2：除了 weights，还会保存 model buffers 的 CPU 副本，便于恢复。
```

### 9.3 wake_up 的恢复逻辑

```python
allocator.wake_up(tags)
if len(self._sleep_saved_buffers):
    restore model buffers
if tags is None or "kv_cache" in tags:
    self.model_runner.post_kv_cache_wake_up()
```

位置：`gpu_worker.py:187` 到 `gpu_worker.py:200`

这说明 wake_up 不只是“把内存搬回来”，还要让 ModelRunner 恢复 KV cache 相关状态。

### 9.4 Executor 的睡眠状态管理

Executor 会在 sleep / wake 之间维护 tags：

```text
sleeping_tags = {"weights", "kv_cache"}
```

如果只唤醒其中一部分 tag，Executor 会检查 tag 是否存在，避免错误恢复。

---

## 10. 运行期控制接口

除了 profile 和 sleep / wake_up，Executor 还统一暴露：

```text
add_lora()
remove_lora()
pin_lora()
list_loras()
reset_mm_cache()
reset_encoder_cache()
execute_dummy_batch()
take_draft_token_ids()
save_sharded_state()
save_tensorized_model()
reinitialize_distributed()
```

这些接口本质上都是把生命周期之外的控制操作统一通过 Executor 转发到 Worker / ModelRunner。

---

## 11. shutdown

### 11.1 Executor shutdown

抽象层：

```python
self.collective_rpc("shutdown")
```

位置：`abstract.py:276` 到 `abstract.py:278`

### 11.2 多进程 executor 的 shutdown

`MultiprocExecutor.shutdown()` 会：

```text
1. 标记 shutting_down；
2. 关闭 death_writer；
3. 等待 worker 正常退出；
4. 必要时 SIGTERM；
5. 再不退出则 SIGKILL；
6. 关闭 worker_response_mq / rpc_broadcast_mq / response_mqs。
```

位置：`multiproc_executor.py:456` 到 `multiproc_executor.py:489`

### 11.3 Worker shutdown

`GPUWorker.shutdown()` 会：

```text
1. gc.unfreeze();
2. shutdown KV transfer / EC transfer；
3. 停止 profiler；
4. 停止 weight transfer engine；
5. 调用 model_runner.shutdown()；
6. 回收 GPU tensor / KV cache / workspace。
```

位置：`gpu_worker.py:1141` 到 `gpu_worker.py:1159`

### 11.4 ModelRunner shutdown

`GPUModelRunner.shutdown()` 会：

```text
1. 调用 _cleanup_profiling_kv_cache()，同步设备并清理 kv_caches / attn_groups / kv_cache_config / layer.kv_cache；
2. ROCm 场景额外清理 captured graphs；
3. 清空 static_forward_context；
4. 将 self.model 置空；
5. 清空 RoPE cache；
6. reset_workspace_manager()；
7. 在清理过程中执行 gc.collect() 和 empty_cache()。
```

位置：`gpu_model_runner.py:6340` 到 `gpu_model_runner.py:6396`

### 11.5 关闭顺序为什么重要

生命周期里通常是：

```text
Executor shutdown
  → Worker shutdown
  → ModelRunner shutdown
  → device resources / queues / transfer groups cleanup
```

原因是：

```text
- 先让 worker 收到退出信号并执行自身 shutdown；
- 再确保进程终止并关闭 message queues；
- 避免队列或进程还在使用时提前删掉关键资源。
```

---

## 12. 异常传播与恢复

### 12.1 worker 意外退出怎么发现

`MultiprocExecutor` 里有 worker monitor：

```python
died = multiprocessing.connection.wait(sentinels)
_self.is_failed = True
_self.shutdown()
callback()
```

位置：`multiproc_executor.py:268` 到 `multiproc_executor.py:291`

这说明多进程路径有主动监控 worker 存活的机制。

### 12.2 失败后的行为

一旦 `is_failed` 被设置：

```python
raise RuntimeError("Executor failed.")
```

位置：`multiproc_executor.py:355` 到 `multiproc_executor.py:357`

后续 `collective_rpc()` 不再正常执行。

### 12.3 failure callback

Executor 支持注册 failure callback：

```python
register_failure_callback(callback)
```

位置：`abstract.py:139` 到 `abstract.py:144`

在 `MultiprocExecutor` 中，如果 worker 已经失败，会直接触发 callback；否则保存 callback，等 monitor 检测到异常时触发。

### 12.4 EngineCore 如何感知异常

EngineCore 在构造 `model_executor` 时可以传入 `executor_fail_callback`。

位置：`engine/core.py:96` 到 `engine/core.py:126`

这表示：

```text
Worker 崩溃 → Executor monitor 检测到 → 回调 EngineCore/Engine 层 → 上层感知失败并停止。
```

### 12.5 timeout 与 RPC 失败

`collective_rpc()` 会为消息队列接收设置 timeout；超时后会抛出：

```text
TimeoutError("RPC call to ... timed out.")
```

位置：`multiproc_executor.py:383` 到 `multiproc_executor.py:390`

### 12.6 为什么恢复不是自动的

vLLM V1 的生命周期恢复更偏向：

```text
- 失败后上层重建；
- 而不是自动在同一 executor 中无限重试。
```

这能避免不一致的 worker 状态继续污染运行。

---

## 13. determine_available_memory()

这也是生命周期的一部分，因为它直接决定 KV cache 能分多少。

### 13.1 Worker 的职责

`GPUWorker.determine_available_memory()` 会：

```text
1. profile / dummy run；
2. 统计 weights、activation、non-torch、cudagraph memory；
3. 算出可用于 KV cache 的 bytes；
4. 返回给 EngineCore。
```

位置：`gpu_worker.py:372` 到 `gpu_worker.py:524`

### 13.2 EngineCore 为什么需要它

`EngineCore._initialize_kv_caches()` 依赖这个结果来构造：

```text
kv_cache_configs
scheduler_kv_cache_config
cache_config.num_gpu_blocks
cache_config.block_size
```

位置：`engine/core.py:271` 到 `engine/core.py:348`

所以 memory profiling 是生命周期初始化的重要一环。

---

## 14. init_kv_cache 和 warmup 的关系

### 14.1 Worker.initialize_from_config 只负责 KV cache 初始化相关工作

`GPUWorker.initialize_from_config()` 包含：

```text
- 更新 cache_config.num_gpu_blocks；
- kv transfer 初始化；
- 在 kv_cache memory pool 中调用 model_runner.initialize_kv_cache()；
- routed experts capturer 初始化；
- KV zero metadata 初始化。
```

真正的 warmup / compile / CUDA graph capture 不在这个 Worker 方法里，而是在 `Executor.initialize_from_config()` 随后通过 `collective_rpc("compile_or_warm_up_model")` 调用 `GPUWorker.compile_or_warm_up_model()` 完成。

### 14.2 为什么 warmup 在 KV cache 初始化之后

因为 warmup 依赖：

```text
- 已知 batch size / cache 配置；
- 已分配好 KV cache；
- 已初始化模型和 attention backend。
```

所以顺序必须是：

```text
load_model
  → initialize_kv_cache
  → compile_or_warm_up_model
```

---

## 15. 一个完整的生命周期时间线

可以把整个生命周期记成如下顺序：

```text
1. EngineCore 选择 Executor
2. Executor 创建 Worker / WorkerProc / Ray actor
3. Worker init_device()
4. Worker load_model()
5. EngineCore 计算 KV cache 配置
6. Worker initialize_from_config(kv_cache_config)
7. Worker compile_or_warm_up_model()
8. 正常 execute_model / sample_tokens 循环
9. profile / sleep / wake_up / control APIs
10. shutdown 或 worker failure monitor 触发退出
11. 释放 queues / profiler / kv cache / model / workspace
```

---

## 16. 容易疑惑的点

### 16.1 initialize_from_config 是初始化模型还是 KV cache？

在 Worker 侧它主要是 KV cache 初始化；模型权重已经在 `load_model()` 阶段加载。

在 Executor 层的 `initialize_from_config()` 会先广播 Worker 的 `initialize_from_config()`，再广播 `compile_or_warm_up_model()`，所以从 EngineCore 的视角看这一大阶段包含“初始化 KV cache + warmup / compile”。

### 16.2 sleep 会不会影响模型权重？

会，至少在 level 1 下会 offload weights；level 2 还会保存 buffer 状态，以便恢复。

### 16.3 wake_up 后模型一定完全恢复吗？

依赖 tags。

如果只唤醒部分 tag，Executor 会只恢复对应资源；如果 tags 为空或包含 kv_cache，则会进一步调用 `post_kv_cache_wake_up()`。

### 16.4 shutdown 是不是只关进程？

不是。

它还会关闭 profiler、KV transfer、EC transfer、weight transfer、model runner、队列和 GPU 资源。

### 16.5 worker 崩溃后能自动恢复吗？

默认不是自动热恢复，而是通过 failure callback 通知上层失败，通常由上层重建。

---

## 17. 总结

生命周期可以压缩成：

```text
创建 Executor
  → 启动 Worker
  → init_device
  → load_model
  → initialize_kv_cache
  → compile / warmup
  → 正常运行
  → profile / sleep / wake_up
  → shutdown / failure callback
```

如果只记住一句话：

```text
Executor 负责生命周期控制面，Worker 负责设备侧资源，ModelRunner 负责真正的模型执行和资源释放。
```
