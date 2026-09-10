# 09. KV Cache、LoRA、KV Transfer、Profiler

Worker / ModelRunner 执行层不仅负责 forward 和 sampling，还承载多个横切能力：

- KV cache spec 与初始化。
- KV cache zeroing。
- LoRA 加载与激活。
- KV transfer connector。
- EC transfer。
- Profiler。
- Weight transfer。

这些能力散布在 Worker、ModelRunner 和 mixin 中。

## 1. KV cache 生命周期总览

KV cache 在 V1 中是分阶段完成的：

```text
Worker.get_kv_cache_spec
  -> ModelRunner 扫描模型 attention layers
  -> 返回每层 KVCacheSpec

Worker.determine_available_memory
  -> profile_run 估算非 KV 显存
  -> 返回可用于 KV cache 的内存

EngineCore
  -> 根据 spec + available memory 计算 KVCacheConfig

Executor.initialize_from_config
  -> 下发每个 worker 的 KVCacheConfig

Worker.initialize_from_config
  -> ensure_kv_transfer_initialized
  -> ModelRunner.initialize_kv_cache
  -> 分配、reshape、bind KV cache tensors
```

关键入口：

- `code/vllm/vllm/v1/worker/gpu_worker.py:547`
- `code/vllm/vllm/v1/worker/gpu_worker.py:371`
- `code/vllm/vllm/v1/worker/gpu_worker.py:562`

## 2. KV cache spec

Worker 层：

```text
Worker.get_kv_cache_spec
  -> model_runner.get_kv_cache_spec()
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:547`

ModelRunner V1：

- 遍历 static forward context 中的 `AttentionLayerBase`。
- 对 KV sharing target layer，不创建自己的 KVCacheSpec，而是记录 shared mapping。
- 跳过不需要 KV cache 的模块。
- 返回 layer name -> KVCacheSpec。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7459`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7473`

ModelRunner V2：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:403`

## 3. KV cache 初始化

Worker 层：

```text
initialize_from_config(kv_cache_config)
  -> cache_config.num_gpu_blocks = ...
  -> ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
  -> with memory_pool("kv_cache"):
         model_runner.initialize_kv_cache(kv_cache_config)
  -> routed experts capturer init
  -> kv zero metadata init
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:562`
- `code/vllm/vllm/v1/worker/gpu_worker.py:570`
- `code/vllm/vllm/v1/worker/gpu_worker.py:575`

为什么 KV transfer 要先初始化：

- connector 依赖 KVCacheConfig。
- connector 可能要求 uniform / cross-layer KV layout。
- model runner 初始化 KV cache group 前需要知道 connector 偏好。

## 4. ModelRunner V1 KV cache 初始化

`initialize_kv_cache()` 做的事情：

- deep copy KVCacheConfig。
- 添加 encoder-only layers。
- 处理 cross-layer KV sharing。
- 初始化 attention backend。
- 初始化 Mamba SSU backend。
- 计算 kernel block sizes。
- 初始化 metadata builders。
- 根据真实 block size 可能重建 InputBatch。
- 分配并绑定 KV cache tensor。
- spec decode extract hidden states 校验。
- 注册 KV cache 到 KV connector。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7303`

KV cache tensor 分配：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7220`

connector 注册：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:7351`

## 5. uniform / cross-layer KV layout

KV transfer connector 可能偏好跨层连续化 KV cache。

判断条件：

- 存在 KV transfer group。
- connector 偏好 cross-layer blocks。
- KV cache 只有单 group/单 attention group。
- backend 支持带 layer 维度 layout。

相关源码：

- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:115`
- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:186`

目的：

- 让跨层 KV cache 连续分配。
- 更便于按 block 做高效传输。

## 6. KV cache zeroing

部分场景需要对新分配 KV block 清零。

机制：

- `_init_kv_zero_meta()` 创建 `KVBlockZeroer`。
- `_update_states()` 遇到 `scheduler_output.new_block_ids_to_zero` 时调用 `_zero_block_ids()`。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1090`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1105`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:1153`

CPUModelRunner 中 `_zero_block_ids()` 是 no-op，因为 CPU attention 对非法位置赋 `-INF`，旧 KV 不影响计算。

- `code/vllm/vllm/v1/worker/cpu_model_runner.py:146`

## 7. LoRA 管理

Worker 层 LoRA API 只是转发：

- `add_lora`
- `remove_lora`
- `list_loras`
- `pin_lora`

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:958`

LoRA 实际逻辑在 mixin：

- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py`

## 8. LoRA 加载与激活

### 8.1 load_lora_model

流程：

- 检查模型是否支持 LoRA。
- 创建 `LRUCacheWorkerLoRAManager`。
- 用 manager 包装 model。

源码：

- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:31`

### 8.2 set_active_loras

执行时：

- 从 `InputBatch.make_lora_inputs()` 构造 prompt/token LoRA mapping。
- 设置 active adapters。

源码：

- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:73`

ModelRunner 准备输入时启用：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:2193`

### 8.3 dummy LoRA warmup

用于 warmup / CUDA graph capture：

- `maybe_setup_dummy_loras()`
- `maybe_select_dummy_loras()`
- `maybe_dummy_run_with_lora()`

源码：

- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:93`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:132`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:236`

### 8.4 LoRA 管理 API

源码：

- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:274`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:278`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:282`
- `code/vllm/vllm/v1/worker/lora_model_runner_mixin.py:286`

## 9. KV Transfer / KV Connector

KV transfer 负责跨 worker / 跨实例 / 分层存储等场景下的 KV cache 传输。

Worker 初始化入口：

- `code/vllm/vllm/v1/worker/gpu_worker.py:575`

ModelRunner mixin：

- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:33`

## 10. KV connector handshake metadata

Worker 可以返回 KV connector 握手 metadata。

逻辑：

- 无 KV transfer group：返回 None。
- connector 不需要握手 metadata：返回 None。
- 否则以 `(pp_rank, tp_rank)` 为 key 返回 metadata。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:526`

## 11. KV connector execute 期逻辑

关键方法：

### 11.1 kv_connector_no_forward

没有 forward 时仍执行 KV send/recv。

源码：

- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:36`

### 11.2 maybe_get_kv_connector_output

如果有 transfer group，进入 connector context；否则 no-op。

源码：

- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:50`

### 11.3 _get_kv_connector_output

流程：

```text
_get_kv_connector_output
  -> bind scheduler metadata
  -> start_load_kv(get_forward_context())
  -> forward 执行期间 connector 工作
  -> finally:
       wait_for_save
       collect finished sending/receiving
       collect invalid block ids
       collect stats/events/worker meta
       clear connector metadata
```

源码：

- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:78`

### 11.4 finalize_kv_connector

spec decode 时 forward 阶段可能延迟 finalize 到 drafter 之后。

源码：

- `code/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py:64`

## 12. EC transfer

EC transfer 在 worker 分布式初始化和 shutdown 中被处理。

初始化相关：

- `code/vllm/vllm/v1/worker/gpu_worker.py:1164`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1197`

shutdown：

- `code/vllm/vllm/v1/worker/gpu_worker.py:1141`

## 13. Profiler

GPU worker 支持：

- torch profiler：CPU + CUDA。
- cuda profiler。

start 时 trace name 加 rank suffix。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:901`

profile annotation 根据 scheduler output 生成：

- context/generation 请求数量。
- token 数。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:775`

CPU profiler：

- CPU-only torch profiler。
- `code/vllm/vllm/v1/worker/cpu_worker.py:95`
- `code/vllm/vllm/v1/worker/cpu_worker.py:252`

XPU profiler：

- CPU + XPU activities。
- `code/vllm/vllm/v1/worker/xpu_worker.py:139`

## 14. Weight transfer

Worker 加载模型后，如果配置了 `weight_transfer_config`，会基于 model 创建 weight transfer engine。

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:358`

shutdown 清理：

- `code/vllm/vllm/v1/worker/gpu_worker.py:1147`
- `code/vllm/vllm/v1/worker/gpu_worker.py:1149`

## 15. 与 SchedulerOutput 的关系

这些横切能力很多都由 `SchedulerOutput` 驱动：

- `finished_req_ids`：worker/model runner 清理请求状态。
- `preempted_req_ids`：KV connector 处理抢占。
- `new_block_ids`：更新 block table。
- `new_block_ids_to_zero`：KV zeroing。
- `kv_connector_metadata`：KV transfer。
- `ec_connector_metadata`：EC transfer。
- `scheduled_spec_decode_tokens`：spec decode。
- `scheduled_encoder_inputs`：多模态/encoder。

`SchedulerOutput` 定义：

- `code/vllm/vllm/v1/core/sched/output.py:180`

## 16. 关键理解

1. KV cache 生命周期跨越 EngineCore、Executor、Worker、ModelRunner。
2. Worker 的 `initialize_from_config()` 是 KV cache 真正初始化入口。
3. KV transfer 会影响 KV cache layout，因此必须在 KV cache 分配前初始化。
4. LoRA 管理 API 在 worker，但真实逻辑在 ModelRunner mixin。
5. profiler、weight transfer、EC transfer 都是 worker 生命周期的一部分。
6. 很多横切能力都由 SchedulerOutput 中的元数据驱动。
