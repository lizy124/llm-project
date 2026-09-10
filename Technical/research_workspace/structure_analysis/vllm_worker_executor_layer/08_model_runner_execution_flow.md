# 08. ModelRunner 执行链路

ModelRunner 是 Worker 下方真正执行模型 forward、sampling、KV/LoRA/spec decode 等逻辑的主体。

vLLM V1 当前有两条 GPU ModelRunner 路径：

- 旧版 V1：`code/vllm/vllm/v1/worker/gpu_model_runner.py`
- 新版 V2：`code/vllm/vllm/v1/worker/gpu/model_runner.py`

Worker 会根据 `vllm_config.use_v2_model_runner` 选择具体实现。

相关创建位置：

- `code/vllm/vllm/v1/worker/gpu_worker.py:326`

## 1. 两阶段执行模型

vLLM V1 的 generation 路径通常拆成两步：

```text
execute_model(scheduler_output)
  -> 更新状态
  -> 准备输入
  -> 执行 forward
  -> 保存 logits/metadata 到 ExecuteModelState
  -> 返回 None

sample_tokens(grammar_output)
  -> 读取 ExecuteModelState
  -> 应用 grammar bitmask
  -> sampling
  -> bookkeeping
  -> 返回 ModelRunnerOutput
```

拆分原因：

- EngineCore 可以在 forward 期间准备 structured output grammar bitmask。
- 支持 async scheduling。
- 支持 PP 场景下非最后 rank 只传 intermediate tensors。
- 支持 spec decode、KV connector finalize 等复杂后处理。

关键位置：

- V1 execute：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4043`
- V1 sample：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`
- V2 execute：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1102`
- V2 sample：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1327`

## 2. Worker 到 ModelRunner

Worker 的 execute 负责 PP 通信包装，然后调用 model runner。

```text
Worker.execute_model
  -> 等上一轮 PP send
  -> 非 first PP rank 接收 intermediate tensors
  -> model_runner.execute_model(scheduler_output, intermediate_tensors)
  -> 如果返回 IntermediateTensors，则发送到下一 PP rank
  -> 否则返回 ModelRunnerOutput / AsyncModelRunnerOutput / None
```

源码：

- `code/vllm/vllm/v1/worker/gpu_worker.py:807`
- `code/vllm/vllm/v1/worker/gpu_worker.py:811`
- `code/vllm/vllm/v1/worker/gpu_worker.py:853`
- `code/vllm/vllm/v1/worker/gpu_worker.py:867`
- `code/vllm/vllm/v1/worker/gpu_worker.py:871`
- `code/vllm/vllm/v1/worker/gpu_worker.py:882`

Worker 的 sample 只是转发：

- `code/vllm/vllm/v1/worker/gpu_worker.py:801`

## 3. GPUModelRunner V1 execute_model 主流程

入口：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4043`

流程：

### 3.1 状态检查

如果上一次 `execute_model()` 返回 `None` 后还没调用 `sample_tokens()`，说明状态机错乱，直接报错。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4049`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4053`

### 3.2 spec ngram 拷贝 scheduler_output

部分 speculative decoding ngram 路径需要复制 scheduler output，避免后续状态更新影响。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4058`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4073`

### 3.3 KV transfer preemption

如果启用 KV transfer，需要先处理 preemption。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4075`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4078`

### 3.4 更新持久 batch/request 状态

调用 `_update_states(scheduler_output)`，将 scheduler output 中的新请求、已缓存请求、finished/preempted 请求等更新到 ModelRunner 内部状态。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4081`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4087`

### 3.5 EC transfer producer 特殊路径

EC transfer producer 情况下可能只跑 encoder，并返回空 encoder output。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4088`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4094`

### 3.6 无 scheduled token 路径

如果本 step 没有 token：

- 没有 KV transfer：返回 empty output。
- 有 KV transfer：执行 no-forward connector 路径。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4096`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4112`

### 3.7 准备输入

准备内容包括：

- 每个 request 的 scheduled token 数。
- input ids。
- positions。
- token indices。
- slot mapping。
- LoRA active mapping。
- spec decode metadata。
- attention metadata。
- CUDA graph / padding / ubatch 决策。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4121`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4128`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4143`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4177`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4244`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4255`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4271`

### 3.8 forward

forward 包在：

- `set_forward_context(...)`
- `maybe_get_kv_connector_output(...)`
- `_model_forward(...)`

中执行。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4303`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4315`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4320`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4326`

### 3.9 postprocess

三种主要返回：

1. 非最后 PP rank：返回 `IntermediateTensors`。
2. pooling model：直接返回 pooling output。
3. generation model：保存 `ExecuteModelState`，返回 `None`，等待 `sample_tokens()`。

源码：

- 非最后 PP：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4337`
- pooling：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4345`
- logits：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4354`
- 保存状态：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4386`
- 返回 None：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4405`

## 4. GPUModelRunner V1 sample_tokens 主流程

入口：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4422`

### 4.1 无 ExecuteModelState

如果没有 `execute_model_state`，通常是：

- PP 非最后 rank。
- 只有 KV connector output。

此时返回 KV connector only output。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4426`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4434`

### 4.2 解包状态

读取并清空 `execute_model_state`。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4436`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4450`

### 4.3 grammar bitmask

结构化输出 grammar 约束在采样前应用。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4452`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4456`

### 4.4 sampling

调用 `_sample()` 产生 sampled token。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4458`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4459`

### 4.5 状态更新与 PP 广播

采样后更新 ModelRunner 状态。

async PP 场景下可能广播 sampled token ids 给其他 PP rank。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4461`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4464`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4472`

### 4.6 speculative decoding

处理 draft tokens、接受/拒绝等 spec decode 逻辑。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4481`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4573`

### 4.7 bookkeeping

`_bookkeeping_sync(...)` 产出：

- `num_nans_in_logits`
- `logprobs_lists`
- `valid_sampled_token_ids`
- `prompt_logprobs_dict`
- `req_ids_output_copy`
- `req_id_to_index_output_copy`
- `invalid_req_indices`

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4574`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4589`

### 4.8 构造 ModelRunnerOutput

字段包括：

- `req_ids`
- `req_id_to_index`
- `sampled_token_ids`
- `logprobs`
- `prompt_logprobs_dict`
- `kv_connector_output`
- `ec_connector_output`
- `num_nans_in_logits`
- `cudagraph_stats`

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4609`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4623`

### 4.9 同步 / 异步输出

同步路径直接返回 `ModelRunnerOutput`。

async scheduling 路径返回 `AsyncGPUModelRunnerOutput`，异步拷贝 sampled token/logprobs/routed experts 到 CPU。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4625`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4637`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4672`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:4682`

## 5. AsyncGPUModelRunnerOutput

作用：把 GPU 输出异步拷贝到 CPU，避免主执行流阻塞。

创建时：

- 在 copy stream 上发起 sampled token ids / logprobs / routed experts 的异步 copy。

`get_output()` 时：

- 等待 copy event。
- 把 tensor 转成 list。
- 写回 `ModelRunnerOutput`。
- 返回 CPU 可见结果。

源码：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py:263`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:280`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:287`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:293`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:308`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py:316`

## 6. GPUModelRunner V2 execute_model

入口：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1102`

V2 更模块化，依赖：

- `gpu.attn_utils`
- `gpu.kv_connector`
- `gpu.lora_utils`
- `gpu.model_states`
- `gpu.sample`
- `gpu.spec_decode`
- `gpu.pp_utils`
- `gpu.cudagraph_utils`

主要流程：

```text
execute_model
  -> update_pp_decode_requests
  -> finish_requests
  -> free_states
  -> add_requests
  -> update_requests
  -> block_tables.apply_staged_writes
  -> 如无 token: kv_connector.no_forward
  -> dispatch_cg_and_sync_dp
  -> prepare_inputs
  -> prepare_attn
  -> multimodal embedding
  -> model forward: full cudagraph or eager/piecewise
  -> 保存 ExecuteModelState
  -> 非 last PP rank 返回 IntermediateTensors
  -> last PP rank 返回 None
```

源码：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1110`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1117`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1118`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1143`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1159`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1162`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1198`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1209`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1229`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1257`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1272`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1310`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1320`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1323`

## 7. GPUModelRunner V2 sample_tokens

入口：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1327`

非 last PP rank：

```text
sample_tokens
  -> 接收 last rank 广播 sampled tokens
  -> 更新本地状态
  -> 返回 KV connector only output
```

源码：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1342`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1349`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1356`

last PP rank：

```text
sample_tokens
  -> sample
  -> PP 广播 sampled tokens
  -> prompt logprobs
  -> 构造 ModelRunnerOutput
  -> 创建 AsyncOutput
  -> postprocess_sampled 更新状态
  -> speculator propose draft tokens
  -> kv_connector.post_forward
  -> 返回 async output
```

源码：

- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1360`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1365`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1374`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1384`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1393`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1417`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1430`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1464`
- `code/vllm/vllm/v1/worker/gpu/model_runner.py:1468`

## 8. ModelRunnerOutput

定义文件：

- `code/vllm/vllm/v1/outputs.py:233`

关键字段：

- `req_ids`：输出请求顺序。
- `req_id_to_index`：请求 ID 到输出 index。
- `sampled_token_ids`：本 step 生成 token。
- `logprobs`：采样 logprobs。
- `prompt_logprobs_dict`：prompt logprobs。
- `pooler_output`：pooling 输出。
- `kv_connector_output`：KV connector 输出。
- `ec_connector_output`：EC connector 输出。
- `num_nans_in_logits`：logits NaN 统计。
- `cudagraph_stats`：CUDA graph 统计。
- `routed_experts`：MoE routed experts 数据。

源码：

- `code/vllm/vllm/v1/outputs.py:235`
- `code/vllm/vllm/v1/outputs.py:240`
- `code/vllm/vllm/v1/outputs.py:246`
- `code/vllm/vllm/v1/outputs.py:251`
- `code/vllm/vllm/v1/outputs.py:259`
- `code/vllm/vllm/v1/outputs.py:262`
- `code/vllm/vllm/v1/outputs.py:264`
- `code/vllm/vllm/v1/outputs.py:266`
- `code/vllm/vllm/v1/outputs.py:269`
- `code/vllm/vllm/v1/outputs.py:272`

## 9. PP intermediate tensors

Pipeline parallel 非最后 stage 不返回最终 token，而是返回 `IntermediateTensors`。

Worker 层发现后：

- 通过 PP group 发送给下一 stage。
- 当前 rank 返回 `None`。

源码：

- V1 model runner 返回：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4337`
- Worker 发送：`code/vllm/vllm/v1/worker/gpu_worker.py:882`
- V2 model runner 返回：`code/vllm/vllm/v1/worker/gpu/model_runner.py:1320`

## 10. 关键理解

1. Worker 是生命周期层，ModelRunner 是执行层。
2. `execute_model()` 做 forward，`sample_tokens()` 做采样，是 V1 generation 路径主设计。
3. `execute_model()` 返回 `None` 不是错误，而是表示需要继续调用 `sample_tokens()`。
4. 非最后 PP rank 返回 intermediate tensors，不产生最终输出。
5. async output 通过后台 copy 或 future 延迟把 GPU 结果转为 CPU 可见结果。
