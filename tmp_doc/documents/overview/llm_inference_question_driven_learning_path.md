# 大模型推理问题驱动学习路线图

本文基于 `tmp_doc/documents/question` 目录整理，目标是把已有材料转成一套可以长期反复使用的技术问题清单。使用方式不是直接背答案，而是围绕每个问题主动查资料、读代码、画链路、写复盘，通过回答问题来建立大模型推理系统能力。

适用周期：

- 4 个月：完成核心主线，能系统讲清 vLLM 推理链路、调度、KV Cache、执行层、Attention、分布式和性能问题。
- 8 个月：在 4 个月基础上深入 CUDA Graph、operators、量化、Spec Decode、LoRA、多模态、MoE、KV Transfer、源码改造和性能排查。

本文只整理需要回答的问题，不给标准答案。每个问题都建议你按「定义边界 -> 主链路 -> 关键对象 -> 源码证据 -> 取舍与排查」的方式自己完成。

---

## 0. 参考材料地图

主要参考目录：

```text
tmp_doc/documents/question/
  vllm_overview.md
  config_and_model_loading/
  engine/
  engine_core/
  executor_worker_model_runner/
  kv_cache_transfer/
  attention/
  attention/attention_methods/
  sampling_and_output/
  parallelism/
  quantization/
  spec_decode/
  compilation_and_cuda_graph/
  operators/
  model_architectures/
  lora_and_adapters/
  multimodal/
```

这些材料的关系可以压缩为一条主线：

```text
用户请求
  -> entrypoints / EngineArgs / VllmConfig
  -> LLMEngine / AsyncLLM
  -> InputProcessor
  -> EngineCoreRequest
  -> EngineCoreClient
  -> EngineCore
  -> Scheduler
  -> KVCacheManager / BlockPool
  -> Executor
  -> Worker
  -> GPUModelRunner
  -> InputBatch / attention metadata / slot mapping / block table
  -> Model / Attention / Operators / Kernels
  -> Sampler / RejectionSampler / Pooling
  -> ModelRunnerOutput
  -> Scheduler.update_from_output()
  -> EngineCoreOutputs
  -> OutputProcessor
  -> RequestOutput / streaming chunk
```

---

## 1. 问题回答深度分级

每个问题不要只写一句定义。建议按下面 4 层递进。

### L1：概念边界

你需要回答：

- 它是什么？
- 它解决什么问题？
- 它不负责什么？
- 它和相邻模块的边界是什么？

合格标准：能用 3 到 5 句话说明职责边界，不混淆相邻模块。

### L2：端到端链路

你需要回答：

- 输入是什么？
- 输出是什么？
- 中间经过哪些对象？
- 哪个对象负责状态变化？
- 哪个对象负责真实计算？

合格标准：能画出链路图，并能解释每个箭头为什么存在。

### L3：源码对象

你需要回答：

- 相关类 / 函数在哪里？
- 哪些字段承载关键状态？
- 哪些方法改变状态？
- 哪些配置会改变执行路径？
- 哪些对象是计划，哪些对象是结果？

合格标准：能用源码路径定位关键对象，能说明调用关系。

### L4：取舍、性能与排查

你需要回答：

- 这个设计为什么这样做？
- 它优化了什么指标？
- 它牺牲了什么？
- 什么情况下收益不明显？
- 出问题时如何排查？
- 如何设计 benchmark 验证？

合格标准：能结合 TTFT、TPOT、吞吐、显存、通信、kernel launch、正确性风险分析问题。

---

## 2. 答题记录模板

建议每个问题都按这个模板写到自己的笔记里。

```text
问题：

我的回答：

一句话结论：

模块边界：
- 负责：
- 不负责：

主链路：

关键对象：
- 输入对象：
- 状态对象：
- 输出对象：
- 配置对象：

源码证据：
- path:line
- path:line

性能或正确性取舍：

容易混淆点：

如果出问题，如何排查：

我还不确定的点：
```

---

## 3. 4 个月核心路线总览

### 第 1 个月：建立 vLLM 主链路和请求生命周期

目标：回答「一个请求如何从用户入口走到 EngineCore，再走回用户输出」。

重点模块：

- `vllm_overview.md`
- `config_and_model_loading/`
- `engine/`
- `engine_core/`
- `sampling_and_output/`

最终验收：

- 能画出 vLLM 全链路图。
- 能解释 Engine、EngineCore、Scheduler、Executor、Worker、ModelRunner、OutputProcessor 的职责边界。
- 能解释配置系统如何影响运行时。
- 能解释从 logits 到 RequestOutput 的后半段链路。

### 第 2 个月：深入 Scheduler、KV Cache、执行层和 Attention

目标：回答「每轮到底执行哪些 token，KV block 如何管理，attention 如何读写 KV」。

重点模块：

- `executor_worker_model_runner/`
- `kv_cache_transfer/`
- `attention/`
- `attention/attention_methods/`
- `operators/` 的 attention / KV cache 部分

最终验收：

- 能讲清 `EngineCore.step() = schedule -> execute -> update -> output`。
- 能讲清 `SchedulerOutput -> ModelRunnerOutput -> EngineCoreOutputs` 的区别。
- 能讲清 KVCacheManager、BlockPool、prefix cache、slot mapping、block table 的关系。
- 能讲清 prefill、decode、chunked prefill、mixed batch 下 attention metadata 的差异。

### 第 3 个月：掌握分布式、并行、KV Transfer 和高级推理机制

目标：回答「单机多卡、多节点、PD 分离、Spec Decode、量化如何接入主链路」。

重点模块：

- `parallelism/`
- `kv_cache_transfer/`
- `spec_decode/`
- `quantization/`
- `model_architectures/`

最终验收：

- 能解释 TP / PP / DP / EP / CP / DCP 分别切什么。
- 能解释 world size、rank group、collective、输出 rank 的关系。
- 能解释 KV Transfer / KVPool 从 Scheduler 到 Worker 的闭环。
- 能解释 Spec Decode 为什么是跨 Scheduler、ModelRunner、Sampler、KV cache 的协议。
- 能解释量化从配置到 kernel dispatch 的链路。

### 第 4 个月：补齐生产能力、性能优化和专题表达

目标：回答「如何优化、如何排查、如何扩展模型和能力」。

重点模块：

- `compilation_and_cuda_graph/`
- `operators/`
- `lora_and_adapters/`
- `multimodal/`
- `model_architectures/`
- benchmark / tests / profiler 相关材料

最终验收：

- 能解释动态 serving batch 如何适配 CUDA Graph。
- 能解释 operators / layer / backend / kernel 的区别。
- 能解释 LoRA 和多模态如何挂入 Engine / Scheduler / ModelRunner。
- 能设计一套 TTFT、TPOT、吞吐、显存、通信瓶颈排查流程。

---

## 4. 8 个月深入路线总览

第 1 到第 4 个月完成核心系统链路，第 5 到第 8 个月做深入专题。

### 第 5 个月：CUDA / Operators / Kernel 深入

目标：从 Python 调用链深入到底层 kernel 和 profiler。

重点问题：

- Python operator wrapper 如何调用 `torch.ops`、Triton、FlashAttention、FlashInfer、CUTLASS 或 torch fallback？
- attention、KV cache、quantized linear、MoE、sampling 各自有哪些算子族？
- profiler 里看到的 kernel 名称如何映射回 vLLM layer？
- 什么情况下会 fallback 到 torch native？
- CUDA Graph、torch.compile、backend selection 如何改变实际执行路径？

### 第 6 个月：分布式和大规模 serving 深入

目标：能分析多卡、多节点、MoE、PD 分离、KV Transfer 的系统瓶颈。

重点问题：

- TP / PP / DP / EP 组合时 world size 如何计算？
- 哪些通信发生在模型层，哪些发生在 Executor / Worker 层？
- PP 下 intermediate tensors 如何传递，哪里产生 logits？
- MoE expert dispatch 如何触发 all-to-all？
- PD 分离中 KV transfer 的网络、显存和一致性问题如何分析？

### 第 7 个月：高级能力组合与复杂场景

目标：理解 LoRA、多模态、Spec Decode、量化、结构化输出之间的组合限制。

重点问题：

- LoRA + quantization 如何共存？
- LoRA + TP / PP / DP 时 adapter 权重如何切分和同步？
- 多模态 encoder cache 和 decoder KV cache 有什么区别？
- Spec Decode 和 structured output grammar 如何保持一致？
- CUDA Graph 在 LoRA、多模态、Spec Decode、PP、MoE 场景下为什么容易 fallback？

### 第 8 个月：源码改造、性能实验和体系化复盘

目标：从会读代码变成能改代码、能验证、能复盘。

建议实验：

- 改一个 Scheduler 参数或策略，比较 TTFT / TPOT / throughput。
- 增加一个 KV cache / prefix cache 统计指标。
- 跟踪一次 KV Transfer 的 load/save 生命周期。
- 对比 eager / compile / CUDA Graph 的 decode 性能。
- 对比不同量化方法的显存、延迟和输出质量。
- 对比 TP size、batch size、max_num_batched_tokens 对性能的影响。

---

# 第一部分：基础主链路问题

## 5. vLLM 总览问题

来源：`vllm_overview.md`

### P0：必须回答

1. vLLM 的核心定位是什么？它主要解决 LLM serving 中哪些问题？
2. vLLM 的高吞吐来自哪些机制的组合，而不是单一机制？
3. 从源码分层看，vLLM 可以分成哪些层？每层解决什么问题？
4. 用户入口、配置层、Engine 层、Scheduler 层、Worker 层、Model 层、Attention 层、底层 kernel 层之间如何连接？
5. 一次请求从用户输入到用户输出，最小对象流转链路是什么？
6. `EngineCoreRequest`、`Request`、`SchedulerOutput`、`ModelRunnerOutput`、`EngineCoreOutputs`、`RequestOutput` 分别处在哪一层？
7. `continuous batching`、`paged KV cache`、`prefix caching`、`CUDA graph`、`torch.compile`、`custom kernel` 分别优化了什么？
8. vLLM 支持的任务不只有 generation，还包括哪些任务？这些任务和 generation 的执行差异是什么？
9. `VllmConfig` 为什么可以看作贯穿全系统的配置总线？
10. 如果只记住一条 vLLM 主链路，你应该如何画？

### P1：重点回答

1. vLLM 的 online serving 和 offline LLM API 在入口层有什么区别？
2. OpenAI-compatible API server 和 Python `LLM.generate()` 最终如何进入统一 engine？
3. `entrypoints` 层为什么不应该承担调度和模型执行职责？
4. `EngineCore.step()` 为什么可以概括为 `schedule -> execute -> update -> output`？
5. Scheduler 和 KVCacheManager 的关系是什么？谁管请求状态，谁管 KV block 账本？
6. Executor 和 Worker 的关系是什么？为什么不能让 EngineCore 直接调用 GPUModelRunner？
7. ModelRunner 为什么是执行层最核心的桥接对象？
8. Attention backend 为什么不是单纯一个 kernel，而是一套 metadata、layout、impl、selector 的协议？
9. OutputProcessor 和 Scheduler.update_from_output() 的区别是什么？
10. benchmark、tests、docs 在学习源码时分别应该怎么用？

### P2：进阶追问

1. 如果一个请求卡住没有输出，你会沿着哪些对象逐层排查？
2. 如果吞吐低但 GPU 利用率高，可能是哪些模块造成的？
3. 如果 GPU 利用率低但请求延迟高，可能是哪些模块造成的？
4. 如果请求取消后 KV block 没有释放，应该检查哪些状态？
5. 如果 streaming 输出乱序或重复，应该检查哪些输出对象？
6. 如果某个模型架构无法加载，应该从配置、registry、model_loader、weights 哪些阶段排查？
7. 如果某个 attention backend 不生效，应该检查哪些配置和平台条件？
8. 如果 CUDA Graph 没有 replay，应该从 batch shape、feature flags、metadata、fallback 哪些角度排查？

---

## 6. 配置与模型加载问题

来源：`config_and_model_loading/`

### P0：必须回答

1. 用户传入的 CLI / Python API 参数如何进入 `EngineArgs`？
2. `EngineArgs` 负责什么？它不负责什么？
3. `EngineArgs.create_engine_config()` 如何构造 `VllmConfig`？
4. `VllmConfig` 包含哪些子配置？每个子配置影响哪些运行时模块？
5. `ModelConfig` 如何读取 Hugging Face config？
6. `ModelConfig` 如何判断模型的 task、runner_type、dtype、max_model_len、quantization？
7. `LoadConfig` 如何决定权重加载方式？
8. `model registry` 如何根据 `hf_config.architectures` 找到 vLLM model class？
9. `model_loader` 如何 instantiate model，再加载 safetensors / sharded / quantized weights？
10. `Worker.load_model()` 和 `GPUModelRunner.load_model()` 在启动生命周期中处于什么位置？
11. 模型加载、KV cache 初始化、warmup / compile / CUDA graph capture 的顺序是什么？
12. 为什么配置构造阶段不能直接执行模型 forward？

### P1：重点回答

1. `ModelConfig` 和 `LoadConfig` 的边界是什么？
2. `ModelConfig`、`CacheConfig`、`ParallelConfig`、`SchedulerConfig` 如何共同影响 serving 行为？
3. `max_model_len` 会受到哪些配置和模型属性影响？
4. dtype 推断会受到哪些因素影响？
5. quantization 信息可能来自用户参数，也可能来自 checkpoint metadata，vLLM 如何处理优先级？
6. 如果 `trust_remote_code` 影响模型加载，它具体影响哪个阶段？
7. `get_model_loader(load_config)` 需要考虑哪些权重格式？
8. `initialize_model(vllm_config)` 如何把全局配置传给模型类？
9. `process_weights_after_loading()` 为什么重要？哪些量化或 packed weight 需要后处理？
10. `Worker.determine_available_memory()` 为什么通常发生在模型加载之后、KV cache 初始化之前？

### P2：进阶追问

1. 如果模型 config 的 `architectures` 不在 registry 中，应该如何扩展？
2. 如果 checkpoint 权重名和 vLLM 模型类参数名不一致，应该如何映射？
3. 如果 TP 下某个权重 shard shape 不对，应该检查哪些 layer 和 weight loader？
4. 如果加载量化模型时缺少 scale / zero point，应该定位到哪个阶段？
5. 如果模型支持 generation 和 embedding 多任务，runner_type 应如何判断？
6. 如果某个模型需要自定义 RoPE / M-RoPE / MLA，应该挂到模型架构层还是 attention backend 层？
7. 如果新增一个模型，需要实现哪些接口、权重加载方法和能力声明？
8. 如果模型加载成功但 forward 报 shape mismatch，如何区分是模型层问题、input preprocess 问题还是并行切分问题？

---

## 7. Engine 层问题

来源：`engine/`

### P0：必须回答

1. vLLM V1 中的 Engine 是一个具体类，还是一个架构层概念？
2. `LLMEngine` 和 `AsyncLLM` 分别适合什么路径？
3. 外层 Engine 和 `EngineCore` 的职责边界是什么？
4. `InputProcessor` 负责什么？输入和输出分别是什么？
5. `OutputProcessor` 负责什么？输入和输出分别是什么？
6. `EngineCoreClient` 为什么是外层 Engine 和内部 EngineCore 之间的桥？
7. `LLMEngine.engine_core` 字段为什么实际可能是 `EngineCoreClient`？
8. in-process 和 multi-process 模式下 `EngineCoreClient` 有什么区别？
9. 一个同步请求进入 `LLMEngine.add_request()` 后经历哪些对象？
10. `OutputProcessor.add_request()` 为什么在请求进入 EngineCore 前就需要记录状态？
11. `abort_request()` 应该如何沿着 Engine / EngineCore / Scheduler 传递？
12. 外层 Engine 为什么不直接处理 Scheduler 队列？

### P1：重点回答

1. `LLMEngine.step()` 和 `EngineCore.step()` 的职责有什么不同？
2. `AsyncLLM` 为什么需要异步队列和后台输出处理？
3. OpenAI API server 的 streaming 输出如何依赖 `OutputProcessor`？
4. `InputProcessor.assign_request_id()` 解决什么问题？
5. `n > 1` 或多输出请求会如何影响 `OutputProcessor` 的请求管理？
6. 外层 Engine 的 profile / reset / sleep / wake_up / LoRA 控制接口如何转发到底层？
7. Engine 层如何处理 pooling / embedding 和 generation 的不同输出？
8. `EngineCoreRequest` 中应该包含哪些已经处理好的信息？
9. Engine 层和 tokenizer / detokenizer 的关系是什么？
10. 为什么外层 Engine 是协议适配层，而不是性能优化核心？

### P2：进阶追问

1. 如果 API server 收到请求但 EngineCore 没有看到 request，应该检查哪些对象？
2. 如果 EngineCore 已经输出 token，但客户端没有收到 streaming chunk，应该检查哪里？
3. 如果 request id 冲突或输出串流混乱，应该检查哪些状态？
4. 如果异步多进程模式下输出延迟异常，可能是 EngineCoreClient、ZMQ、OutputProcessor 哪个环节？
5. 如果 abort 后请求仍然占用 KV cache，应该从 Engine 到 Scheduler 如何追踪？
6. 如果 sleep / wake_up 后请求失败，应该检查 Engine 控制接口和 Worker 状态的哪些环节？

---

## 8. EngineCore 问题

来源：`engine_core/`

### P0：必须回答

1. `EngineCore` 在 vLLM V1 中的定位是什么？
2. `EngineCore` 负责什么？不负责什么？
3. 为什么说 `EngineCore` 是内部执行闭环总控？
4. `EngineCore.step()` 的主流程是什么？
5. `SchedulerOutput`、`ModelRunnerOutput`、`EngineCoreOutputs` 三者分别代表什么？
6. `EngineCore.preprocess_add_request()` 如何把 `EngineCoreRequest` 转成内部 `Request`？
7. 结构化输出的 grammar 初始化为什么发生在 EngineCore 接收请求阶段？
8. `Scheduler.add_request()` 是在哪个阶段调用的？
9. `model_executor.execute_model(..., non_block=True)` 为什么可能异步返回 future？
10. `model_output is None` 时为什么还需要 `sample_tokens(grammar_output)`？
11. `Scheduler.update_from_output()` 在 EngineCore 主循环中扮演什么角色？
12. `EngineCore.step()` 返回的 bool 标志表示什么？

### P1：重点回答

1. `EngineCore` 为什么不直接 detokenize？
2. `EngineCore` 为什么不直接分配 KV block，而是交给 Scheduler / KVCacheManager？
3. `EngineCore` 如何处理 abort queue？
4. `EngineCore` 如何处理 structured output grammar bitmask？
5. `EngineCore` 和 `EngineCoreProc` 的关系是什么？
6. 多进程模式下 `EngineCoreProc` 的输入输出队列如何工作？
7. `EngineCoreClient` 和 `EngineCoreProc` 可以如何类比 client / server？
8. 如果没有请求，`EngineCore.step()` 为什么直接返回空输出？
9. `post_step()` 和 spec decode draft token 回写有什么关系？
10. async scheduling 会改变 EngineCore 和 Worker 的哪些边界？

### P2：进阶追问

1. 如果 `Scheduler.schedule()` 生成了空执行计划，EngineCore 应该如何表现？
2. 如果 Worker 执行成功但 `Scheduler.update_from_output()` 报错，可能是计划和结果哪里不一致？
3. 如果 `ModelRunnerOutput` 中某个 req_id 不存在，应该如何排查？
4. 如果 grammar bitmask 生成慢，会影响 EngineCore 主循环的哪个阶段？
5. 如果多进程 EngineCoreProc 无法退出，应该检查哪些生命周期状态？
6. 如果 EngineCore busy loop CPU 占用高，可能是什么原因？
7. 如果 output queue 堵塞，EngineCore 如何受影响？

---

## 9. Sampling 与 Output 问题

来源：`sampling_and_output/`

### P0：必须回答

1. Sampling / Output 在完整推理链路中处于哪一段？
2. `SamplingParams` 如何从用户参数变成 worker / sampler 可消费的状态？
3. logits 是在哪里计算的？sample logprobs 和 prompt logprobs 分别在哪里计算？
4. sampler 如何处理 temperature、top-k、top-p、min-p、penalties、seed？
5. structured output grammar bitmask 如何限制采样？
6. `ModelRunnerOutput` 是什么？为什么它还不是用户输出？
7. `Scheduler.update_from_output()` 如何消费 sampled tokens 并更新 request 状态？
8. `OutputProcessor.process_outputs()` 如何把内部输出变成 `RequestOutput`？
9. streaming 输出如何决定本轮返回哪些 token、logprobs、finish reason？
10. pooling / embedding / rerank 输出为什么不走普通 token sampling？

### P1：重点回答

1. ModelRunner / Sampler 层负责什么？它不负责什么？
2. Scheduler update 层负责什么？它和 OutputProcessor 的区别是什么？
3. OutputProcessor 为什么要处理 detokenize 和 stop string？
4. prompt logprobs 和 generated token logprobs 的计算位置有什么区别？
5. logits processor、grammar bitmask、bad words、penalties 的应用顺序如何理解？
6. `finish_reason` 可能有哪些来源？
7. `stop_token_ids` 和 stop string 的处理边界在哪里？
8. streaming 场景下如何避免重复输出已经返回过的文本？
9. 如果 request 本轮没有新 token，OutputProcessor 应该如何处理？
10. pooling 输出如何被包装成 `PoolingRequestOutput`？

### P2：进阶追问

1. 如果输出 token 正确但文本 detokenize 错误，应该检查哪一层？
2. 如果 logprobs 数量和输出 token 数对不上，应该检查哪些对象？
3. 如果 structured output 生成非法 JSON，是 grammar、sampler、OutputProcessor 哪层可能出问题？
4. 如果 streaming chunk 丢失，如何沿着 EngineCoreOutputs 到 RequestOutput 排查？
5. 如果 top-p / top-k 设置后输出异常，如何确认参数是否进入 sampler？
6. 如果 prompt logprobs 导致性能下降，应该如何分析额外开销？
7. 如果 spec decode 下输出 token 数不固定，OutputProcessor 如何保持状态一致？

---

# 第二部分：执行层、KV Cache 与 Attention

## 10. Executor / Worker / ModelRunner 问题

来源：`executor_worker_model_runner/`

### P0：必须回答

1. Executor 是哪一层？负责什么？不负责什么？
2. Worker 是哪一层？负责什么？不负责什么？
3. ModelRunner 是哪一层？为什么它是执行层核心？
4. `SchedulerOutput` 如何从 EngineCore 进入 Worker / ModelRunner？
5. `execute_model()` 到 `sample_tokens()` 的完整主线是什么？
6. Worker 初始化 device、distributed environment、workspace、model、KV cache 的顺序是什么？
7. Worker 为什么需要 profile 可用显存？
8. ModelRunner 如何维护 worker 侧请求状态和 persistent batch？
9. ModelRunner 如何准备 `input_ids`、`positions`、`slot_mapping`、`block_table`？
10. ModelRunner 如何构造 attention metadata？
11. ModelRunner 如何执行 model forward、compute logits、sampling？
12. `ModelRunnerOutput` 如何回到 Scheduler 并闭环？

### P1：重点回答

1. `UniProcExecutor`、`MultiprocExecutor`、`RayExecutor` 的差异是什么？
2. `collective_rpc` 解决什么问题？
3. WorkerWrapperBase 为什么存在？
4. `InputBatch` 在 ModelRunner 中承载什么状态？
5. `_update_states()` 负责什么？
6. `_prepare_inputs()` 负责什么？
7. `_build_attention_metadata()` 负责什么？
8. `_preprocess()` 负责什么？
9. `_model_forward()` 的输入可能有哪些形态？
10. `sample_tokens()` 为什么可能和 `execute_model()` 分开执行？
11. KV cache、LoRA、multimodal、spec decode、pipeline parallel 分别在哪个阶段挂接？
12. sleep / wake_up / profile 为什么属于 Worker 控制能力？

### P2：进阶追问

1. 如果 Worker 初始化失败，如何区分 device、distributed、model loading、KV cache 哪个阶段失败？
2. 如果 ModelRunnerOutput 缺少某个请求，如何排查 batch 状态？
3. 如果 input_ids 和 positions 长度不一致，可能是哪个阶段的问题？
4. 如果 slot_mapping 出错，attention 会表现出什么问题？
5. 如果 pipeline parallel 下非最后 rank 返回 None，为什么可能是正常行为？
6. 如果 worker OOM 发生在 warmup，而不是真实请求，应该如何分析？
7. 如果多进程 executor 中某个 worker hang 住，如何定位 collective_rpc？
8. 如果 sleep 后 wake_up 显存状态异常，应该检查哪些资源？

---

## 11. KV Cache / KV Transfer / KVPool 问题

来源：`kv_cache_transfer/`

### P0：必须回答

1. KV Cache 在大模型推理中解决什么问题？
2. KV cache 显存占用和哪些模型参数、序列参数、dtype 有关？
3. `KVCacheManager` 是哪一层？负责什么？不负责什么？
4. `BlockPool` 管理的是什么？它是否直接管理 GPU KV tensor？
5. `KVCacheBlock` 需要哪些元数据？
6. 本地 prefix cache 如何命中？命中后如何减少本轮 prefill token？
7. `KVCacheManager.get_computed_blocks()` 和 `allocate_slots()` 分别做什么？
8. `slot_mapping` 和 `block_table` 分别解决什么问题？
9. external KV hit 如何转成本地 block 分配和 Worker load metadata？
10. `SchedulerOutput.kv_connector_metadata` 如何传到 Worker / ModelRunner？
11. Worker / ModelRunner 如何执行 KV load / save？
12. `ModelRunnerOutput.kv_connector_output` 如何回到 Scheduler？
13. `finished_recving`、`finished_sending`、`invalid_block_ids` 分别意味着什么？
14. 请求结束后为什么 external KV save 可能需要延迟释放 blocks？
15. deferred free 要解决什么竞态？

### P1：重点回答

1. Paged KV cache 为什么比连续 KV cache 更适合 continuous batching？
2. prefix cache 的 key 应该包含哪些信息才能保证正确性？
3. prefix cache 命中 full block 和 partial block 有什么区别？
4. block ref count 在共享 prefix cache 中如何工作？
5. preemption 时 KV blocks 如何回收或复用？
6. `WAITING_FOR_REMOTE_KVS` 状态为什么存在？
7. `load_kv_async=True` 会如何改变调度状态机？
8. invalid block 出现后为什么可能需要 recompute？
9. 外部 KV load 成功前，attention 能否读取对应 block？为什么？
10. KVPool hit 对 TTFT、显存和网络带宽分别有什么影响？
11. KV save 的收益和成本是什么？
12. 本地 prefix cache、外部 KV cache、PD 分离之间是什么关系？

### P2：进阶追问

1. 如果 prefix cache 命中率很低，应该检查哪些 key 和 workload 特征？
2. 如果 KVPool hit 了但输出错误，可能是什么正确性问题？
3. 如果 external KV load 慢导致请求堆积，应该从网络、connector、Scheduler 哪些角度排查？
4. 如果 `invalid_block_ids` 频繁出现，可能是什么原因？
5. 如果 deferred free 配置不当，可能造成什么复用竞态？
6. 如果 KV save 后显存不能释放，应该检查 finished_sending 还是 Scheduler release？
7. 如果 block pool free list 耗尽，系统可能如何表现？
8. 如果长上下文请求导致小请求延迟变高，KV cache 和 Scheduler 侧分别如何优化？
9. 如果多 GPU 下 KV block id 和 physical tensor layout 对不上，可能是哪层映射错误？
10. 如果要实现自己的 KV Connector，需要定义哪些 metadata 和状态回传？

---

## 12. Attention 子系统问题

来源：`attention/`

### P0：必须回答

1. Attention 子系统在 vLLM V1 中处于哪一层？
2. Attention 子系统负责什么？不负责什么？
3. `AttentionBackend`、`AttentionMetadataBuilder`、`AttentionImplBase` 各自负责什么？
4. vLLM 如何选择 FlashAttention、FlashInfer、FlashMLA、Triton 等 backend？
5. backend selection 会受平台、dtype、head size、KV cache dtype、sliding window、配置等哪些因素影响？
6. `CommonAttentionMetadata` 包含哪些公共字段？
7. backend-specific metadata 为什么需要单独 builder？
8. GPUModelRunner 如何从 `SchedulerOutput` 构造 attention metadata？
9. prefill、decode、chunked prefill、mixed batch 的 metadata 有什么差异？
10. spec decode 会如何改变 attention metadata？
11. slot mapping 如何把 token 映射到 paged KV cache slot？
12. block table 如何把 request 的 logical block 映射到 physical KV block？
13. attention forward 如何写入当前 token 的 K/V？
14. attention forward 如何读取历史 KV？
15. `ForwardContext` 在 attention 执行中扮演什么角色？

### P1：重点回答

1. FlashAttention 和 PagedAttention 是同一类优化吗？它们分别优化什么？
2. prefill attention 和 decode attention 的计算形态有什么差异？
3. decode attention 为什么通常更偏 memory bandwidth bound？
4. MHA、MQA、GQA 对 KV cache shape 和 attention kernel 有什么影响？
5. MLA 会如何改变 KV cache 和 attention backend 的设计？
6. sliding window / local attention 如何影响 block table 和 metadata？
7. cascade attention 解决什么场景？它如何利用 prefix cache 或共享 prefix？
8. attention backend 为什么需要声明 KV cache layout？
9. KV connector hook 如何挂在 attention layer 边界？
10. CUDA Graph / torch.compile 对 attention metadata 有什么约束？
11. padding token 的 slot mapping 为什么可能是 -1？
12. FULL CUDA graph 下 metadata 为什么可能包含 padded token？

### P2：进阶追问

1. 如果 attention backend 选择不符合预期，如何排查 selector 和 platform？
2. 如果某个 head size 不支持当前 backend，系统会如何 fallback？
3. 如果 block table 错误，会出现哪些输出或崩溃现象？
4. 如果 slot mapping 错误，KV 写入会污染哪些请求？
5. 如果 decode 性能差，如何区分是 attention kernel 慢、KV layout 差还是 batch 太小？
6. 如果 FlashAttention 在 prefill 快但 decode 不明显，如何解释？
7. 如果 cascade attention 带来额外 overhead，什么场景收益会变差？
8. 如果 CUDA Graph replay 时 attention metadata shape 不稳定，应该检查哪些字段？
9. 如果 KV cache dtype 是 FP8，attention backend 需要额外处理什么？
10. 如果要增加一个新的 attention backend，需要实现哪些接口？

---

## 13. Attention 方法族问题

来源：`attention/attention_methods/`

### P0：必须回答

1. FlashAttention 的核心思想是什么？它减少了哪类内存访问？
2. FlashAttention v1 / v2 / v3 的改进方向大致是什么？
3. PagedAttention 解决的是计算问题还是 KV memory management 问题？
4. MHA、MQA、GQA 的区别是什么？它们如何影响 KV cache 显存？
5. MLA 的核心思想是什么？它为什么能降低 KV cache 压力？
6. Sliding Window Attention 适合什么长上下文场景？
7. FlashInfer、FlashMLA、Triton backend 的定位有什么区别？
8. HMA / KV cache layout 与 attention backend 有什么关系？

### P1：重点回答

1. 为什么 FlashAttention 要做 tiling 和 online softmax？
2. 为什么不能简单把完整 attention matrix 存下来？
3. PagedAttention 如何类比操作系统分页？这个类比有哪些边界？
4. GQA 为什么在推理中常见？它相对 MHA 的收益是什么？
5. MLA 对 RoPE、KV cache、attention backend 有哪些额外要求？
6. Sliding Window 会不会影响模型输出质量？为什么？
7. 不同 backend 对 prefill 和 decode 的优化侧重点有什么不同？
8. backend 选择是否一定越新越好？哪些场景可能 fallback 或性能更差？

### P2：进阶追问

1. 如果让你比较 FlashAttention、PagedAttention、FlashInfer，你会从哪些维度比较？
2. 如果长上下文下显存不足，你会优先考虑 GQA、MLA、Sliding Window、KV quant 还是 offload？为什么？
3. 如果 decode batch 很小，attention kernel 的瓶颈可能在哪里？
4. 如果 prefill batch 很大，attention kernel 的瓶颈可能在哪里？
5. 如果模型使用 MLA，原有 block table / slot mapping 是否需要变化？
6. 如果 backend 支持矩阵和模型配置不匹配，怎样设计 fallback 策略？

---

# 第三部分：分布式、高级推理和模型能力

## 14. Parallelism 并行体系问题

来源：`parallelism/`

### P0：必须回答

1. vLLM 中有哪些并行维度？TP、PP、DP、EP、PCP、DCP、SP 分别是什么？
2. 每个并行维度切分的对象是什么？权重、layer、请求、expert、context、KV cache 分别归谁？
3. `ParallelConfig` 如何决定 `world_size`、`world_size_across_dp`、rank mesh 和 group？
4. `world_size = PP * TP * PCP` 这个公式如何理解？
5. 为什么 DCP 不乘进 world size？
6. 为什么 EP 不单独乘进 world size？
7. Worker 初始化时如何建立 TP / PP / DP / EP / PCP / DCP group？
8. 一次 `SchedulerOutput` 进入执行层后，会经过哪些 rank？
9. forward 中哪些地方触发 all-reduce、all-gather、all-to-all、send-recv？
10. logits、sampling、`ModelRunnerOutput` 最终在哪个 rank 产生？
11. TP 下 attention heads 和 MLP 权重如何切分？
12. PP 下 layers 和 intermediate tensors 如何切分？
13. DP 下请求如何分配到 replica？
14. EP 下 experts 和 token dispatch 如何组织？

### P1：重点回答

1. TP 中 column parallel linear 和 row parallel linear 分别需要什么通信？
2. 为什么 TP 常常需要 all-reduce？
3. PP 为什么会有 bubble？推理 serving 中 PP 的收益和代价是什么？
4. DP 和普通多副本 serving 有什么关系？
5. MoE 下 EP 为什么常触发 all-to-all？
6. DCP 复用 TP rank 会带来什么约束？
7. SP 为什么不是独立 group？
8. attention backend 如何读取 CP / DCP metadata？
9. KV cache 在 TP / PP / DP 下的归属有什么差异？
10. 并行组合时哪些维度乘进 worker 数，哪些复用已有 rank？
11. distributed executor backend 如何影响 Worker 创建方式？
12. `parallel_state` 和 `communication_op` 分别负责什么？

### P2：进阶追问

1. 如果 TP size 增大但吞吐没有提升，可能是什么通信瓶颈？
2. 如果 PP 下最后一个 rank 负载高，可能是什么原因？
3. 如果 all-reduce 很慢，如何区分 NCCL、网络、tensor size、同步点问题？
4. 如果 MoE all-to-all 成为瓶颈，可以从哪些角度优化？
5. 如果 DP replica 间负载不均衡，Scheduler 或 router 层可以怎么做？
6. 如果 TP 和 quantization 同时启用，scale / group size 如何切分？
7. 如果 TP 和 LoRA 同时启用，adapter weight 如何切分？
8. 如果 PP 下 sampling 只在最后 rank 发生，其他 rank 的输出应该是什么？
9. 如果多节点 rank mapping 错误，会表现为什么问题？
10. 如何设计一个实验比较 TP=1/2/4/8 的吞吐和延迟？

---

## 15. Speculative Decoding 问题

来源：`spec_decode/`

### P0：必须回答

1. Spec Decode 解决什么问题？
2. 它为什么不是单纯 sampler 分支，而是一条跨层协议？
3. draft tokens 从哪里来？如何挂在 `Request` 状态上？
4. `Request.spec_token_ids` 表示什么？
5. Scheduler 如何把 draft tokens 调度给 target model 验证？
6. `SchedulerOutput.scheduled_spec_decode_tokens` 表示什么？
7. `SpecDecodeMetadata` 如何描述 logits 行号布局？
8. ModelRunner 如何构造 spec decode forward 的输入？
9. RejectionSampler 如何接受或拒绝 draft tokens？
10. accepted tokens、recovered token、bonus token 分别是什么？
11. `Scheduler.update_from_output()` 如何把 accepted / rejected 结果回写到账本？
12. KV cache 和 `num_computed_tokens` 如何保持一致？
13. structured output / grammar 如何与 spec decode 交互？
14. output recovery 为什么必要？

### P1：重点回答

1. 普通 decode 和 spec decode 在请求状态、KV allocation、logits 布局、sampler、输出 token 数上有什么区别？
2. acceptance rate 如何影响 spec decode 收益？
3. draft model / n-gram / EAGLE 等 drafter 的差异是什么？
4. target model 一次 forward 为什么可以验证多个 draft tokens？
5. 拒绝某个 draft token 后，为什么需要 recovered token？
6. 所有 draft 都接受后，为什么可以有 bonus token？
7. `num_tokens_with_spec` 和普通 `num_tokens` 有什么区别？
8. `num_lookahead_tokens` 如何影响 KV slot 分配？
9. spec decode 下 logprobs 如何对应最终输出 token？
10. async scheduling、PP、chunked prefill、KV connector 会给 spec decode 带来哪些限制？

### P2：进阶追问

1. 如果 acceptance rate 很低，spec decode 可能比普通 decode 更慢吗？为什么？
2. 如果 draft token 被 grammar 判定非法，系统应该如何处理？
3. 如果 rejected token 的 KV cache 已经写入，如何保证后续状态正确？
4. 如果 `num_computed_tokens` 更新错误，会导致什么问题？
5. 如果 spec decode 和 chunked prefill 同时发生，调度复杂在哪里？
6. 如果 PP 下 logits 只在最后 stage 产生，spec decode metadata 如何跨 stage 保持一致？
7. 如果 output recovery 出错，用户可能看到什么现象？
8. 如何设计 benchmark 判断 spec decode 是否真的带来收益？

---

## 16. Quantization 量化问题

来源：`quantization/`

### P0：必须回答

1. vLLM 的量化为什么不是一个单独开关，而是一套从配置到 kernel dispatch 的协议？
2. 用户参数 / checkpoint metadata 如何决定 `ModelConfig.quantization`？
3. `VllmConfig.quant_config` 是什么？
4. `QuantizationConfig` 如何为具体 layer 选择 `quant_method`？
5. `layer.quant_method` 负责哪些事情？
6. `create_weights()`、`process_weights_after_loading()`、`apply()` 分别发生在哪些阶段？
7. weight-only quantization 和 activation quantization 的区别是什么？
8. GPTQ、AWQ、FP8、INT8、INT4、Marlin、Machete、CUTLASS 大致分别处在哪条链路？
9. KV cache quantization 和权重量化有什么区别？
10. `kv_cache_dtype`、`cache_dtype`、`KVQuantMode` 分别是什么？
11. quantization 如何影响 Linear、MoE、Attention、KV cache backend？
12. 量化对显存、吞吐、延迟、精度和数值稳定性有什么取舍？

### P1：重点回答

1. checkpoint 中 qweight、scales、qzeros、g_idx 如何映射到 vLLM 参数？
2. packed weight 为什么需要后处理或 repack？
3. TP 下量化参数如何切分？scale 和 group size 如何保持一致？
4. MoE quantization 和普通 Linear quantization 有什么不同？
5. LoRA 与量化 base model 共存时，哪些路径需要特殊处理？
6. attention backend 为什么需要知道 KV cache dtype 和 scale？
7. FP8 KV cache 会对 attention kernel 带来什么额外要求？
8. dynamic quantization 的 scale 何时计算？
9. per-token scale、per-channel scale、per-block scale 分别适合什么场景？
10. 量化方法不支持某个 layer 时，系统如何 fallback 或报错？

### P2：进阶追问

1. 如果量化模型输出质量明显下降，你会从哪些配置和数值路径排查？
2. 如果量化后吞吐没有提升，可能是哪里没有走到 quantized kernel？
3. 如果 FP8 在某张 GPU 上不可用，应该检查平台能力还是 backend 支持？
4. 如果 KV cache quantization 省显存但延迟变高，可能是什么原因？
5. 如果 TP 下 qweight shape 不一致，应该检查 checkpoint 还是 vLLM shard 逻辑？
6. 如果 LoRA + quantization 结果异常，可能是 base output 和 LoRA delta 的 dtype / scale 问题吗？
7. 如果 MoE quantized kernel 慢，应该检查 expert batch、token dispatch 还是 kernel 本身？
8. 如何设计实验比较 FP16、FP8、INT8、INT4 的显存、吞吐和输出质量？

---

## 17. Model Architectures 问题

来源：`model_architectures/`

### P0：必须回答

1. model architecture 层在 vLLM 中处于哪一层？
2. 它负责把什么适配成什么？
3. `ModelConfig` 如何根据 HF config 判断 architecture、task、runner_type、dtype、max_model_len？
4. `ModelRegistry` 如何根据 architecture name 解析 vLLM model class？
5. 一个 vLLM model class 如何构造 embedding、decoder layers、norm、lm_head、pooler？
6. ModelRunner 对 model forward 接口有什么约定？
7. generation model 和 pooling / embedding / rerank model 的执行差异是什么？
8. Attention、MLP、Norm、RoPE 等基础 blocks 如何复用？
9. MoE 模型如何组织 router、experts、fused MoE？
10. 多模态模型如何组织 vision encoder、projector、M-RoPE、inputs_embeds？
11. 权重加载如何处理 fused layer、TP shard、PP missing layer、tie weights、name mapping？
12. Quantization、LoRA、parallelism 如何 hook 到模型架构中？
13. 新增一个模型架构需要实现哪些接口和检查项？

### P1：重点回答

1. `model.forward()` 为什么需要支持 `input_ids`、`positions`、`intermediate_tensors`、`inputs_embeds` 等输入？
2. PP 下某些 layer missing 或 intermediate tensors 传递如何体现在模型类中？
3. embedding 和 lm_head 在 TP 下如何并行？
4. logits processor 和 model architecture 的边界是什么？
5. RoPE、M-RoPE、XD-RoPE 分别可能影响哪些输入字段？
6. MoE 的 expert weight 和 dense MLP weight 加载有什么不同？
7. pooling 模型为什么不需要 sampler？
8. 多模态模型如何把 encoder output 拼回 text token embedding？
9. model class 如何声明支持的任务和特殊能力？
10. 模型层如何适配量化 layer 和 LoRA layer？

### P2：进阶追问

1. 如果新增模型的 forward 能跑但输出不对，如何排查 position ids、RoPE、attention mask、KV cache？
2. 如果 checkpoint 权重名复杂，如何设计 name mapping？
3. 如果模型有 fused QKV weight，TP 切分和加载要注意什么？
4. 如果模型使用 MLA，应该在模型层、attention layer、backend 哪些位置改？
5. 如果模型支持多任务，如何避免 generation 和 embedding 路径互相污染？
6. 如果模型是 MoE，多卡 expert 分布如何影响 model class？
7. 如果模型需要特殊 logits processor，应该放在模型层还是采样层？
8. 如何写一份新增模型的最小 checklist？

---

# 第四部分：生产能力、性能优化和扩展能力

## 18. Compilation / CUDA Graph 问题

来源：`compilation_and_cuda_graph/`

### P0：必须回答

1. Compilation 和 CUDA Graph 在 vLLM 中分别解决什么问题？
2. torch.compile / vLLM compile 优化的对象是什么？
3. CUDA Graph capture / replay 优化的对象是什么？
4. LLM serving 的动态 batch 为什么和 CUDA Graph 的固定 shape 要求矛盾？
5. vLLM 如何把动态请求整理成 `BatchDescriptor + padded buffers + capture-compatible metadata + ForwardContext`？
6. Worker warmup / compile / CUDA graph capture 生命周期是什么？
7. `CompilationConfig`、cudagraph 配置和 runtime mode 如何决定执行路径？
8. ModelRunner 每轮如何选择 eager、compiled、cudagraph replay？
9. 动态 batch 如何通过 padding / shape bucket 变成可 replay 形态？
10. Attention metadata 在 capture 和 replay 时有什么特殊路径？
11. sampler / output 是否被 capture？forward graph 的边界在哪里？
12. 哪些功能会导致 fallback 或禁用 CUDA Graph？

### P1：重点回答

1. CUDA Graph 需要固定 shape、固定地址、固定 kernel launch 序列，这三个条件分别是什么意思？
2. FULL 和 PIECEWISE CUDA Graph 的差异是什么？
3. uniform decode 为什么更适合 CUDA Graph replay？
4. batch padding 会增加哪些额外计算？为什么仍可能值得？
5. `CudagraphDispatcher.dispatch()` 的输入输出是什么？
6. `ForwardContext` 中 cudagraph runtime mode、batch descriptor、attn metadata、slot mapping 如何被下游读取？
7. sequence parallel padding 如何影响 capture？
8. LoRA、encoder output、cascade attention、MoE、PP、DBO 分别可能如何影响 graph replay？
9. compile overhead 和 runtime replay 收益如何平衡？
10. cudagraph miss 应该如何统计和排查？

### P2：进阶追问

1. 如果 CUDA Graph capture 成功但 replay 没走，应该检查哪些 runtime 条件？
2. 如果 shape bucket 太多，会带来什么内存和 warmup 成本？
3. 如果 padding 过多导致吞吐下降，如何调 bucket？
4. 如果 compile 时间过长，如何判断是否值得？
5. 如果 attention metadata 中含动态字段，如何做 capture-compatible 处理？
6. 如果某个 custom op 不支持 capture，系统如何 fallback？
7. 如果 PP / TP 下 CUDA Graph 行为不一致，如何定位 rank 间差异？
8. 如何设计实验对比 eager、compile、CUDA Graph 的 decode TPOT？

---

## 19. Operators / Kernels 问题

来源：`operators/`

### P0：必须回答

1. vLLM 中哪些东西算 operator / kernel？
2. Layer、operator wrapper、backend、kernel 四个词的区别是什么？
3. 算子层在系统中的位置是什么？它负责什么，不负责什么？
4. Python 层如何调用 CUDA extension、Triton、FlashAttention、FlashInfer、torch fallback？
5. `_custom_ops.py`、`torch.ops._C / torch.ops.vllm`、Triton kernel、third-party backend 分别处在哪一层？
6. attention kernels 包含哪些类别？
7. KV cache kernels 包含哪些类别？
8. quantization kernels 包含哪些类别？
9. MoE kernels 包含哪些类别？
10. norm、activation、RoPE、sampling、communication primitive 各自可能有哪些 kernel？
11. backend selection、CUDA Graph、torch compile、parallelism 如何影响实际执行路径？
12. profiler 里看到的 kernel 如何映射回 vLLM layer？

### P1：重点回答

1. operator wrapper 通常负责哪些参数整理、shape 检查和 fallback？
2. custom CUDA extension 和 Triton kernel 的取舍是什么？
3. FlashAttention / FlashInfer / CUTLASS / Marlin 等 backend 适合哪些算子族？
4. KV cache write / reshape / copy / swap 为什么需要专门 kernel？
5. sampling 为什么也可能需要 GPU kernel？
6. quantized linear 为什么通常不只是普通 GEMM？
7. fused MoE kernel 为什么需要处理 token routing 和 expert batching？
8. communication primitive 在 profiler 中如何体现？
9. fake impl / compile support 对 torch.compile 有什么意义？
10. kernel fallback 到 torch native 可能带来哪些性能变化？

### P2：进阶追问

1. 如果 profiler 显示某个 attention kernel 很慢，如何定位到 backend 和 metadata？
2. 如果某个 custom op 报 undefined symbol，应该检查构建、版本还是平台？
3. 如果 Triton kernel 首次运行很慢，如何区分 JIT compile 和 runtime 性能？
4. 如果 quantized GEMM 没有走 Marlin / CUTLASS，如何排查？
5. 如果 MoE kernel 性能差，如何区分 routing、all-to-all、GEMM、load balance 问题？
6. 如果 sampling kernel 成为瓶颈，可能和 vocab size、logprobs、top-k/top-p 有什么关系？
7. 如果 CUDA Graph capture 失败，可能是哪个 operator 不支持 capture？
8. 如果要新增一个 custom op，需要哪些 Python wrapper、C++ binding、CUDA kernel、fallback、test？

---

## 20. LoRA / Adapters 问题

来源：`lora_and_adapters/`

### P0：必须回答

1. vLLM 中 LoRA / adapter 是什么层级的能力？
2. base model 和 LoRA adapter 的加载、生命周期和请求粒度有什么区别？
3. `LoRARequest` 如何从 API / Engine 进入 Scheduler 和 Worker？
4. LoRA manager 如何加载、缓存、pin、卸载 adapter？
5. Worker / ModelRunner 如何维护当前 batch 的 active LoRA 状态？
6. `InputBatch.request_lora_mapping` 表示什么？
7. `InputBatch.make_lora_inputs()` 做什么？
8. `LoRAModelRunnerMixin.set_active_loras()` 如何影响 Worker 侧 adapter 状态？
9. LoRA layer 如何注入 Linear、Embedding、LM head、MoE？
10. 同一个 batch 中混合多个 LoRA 请求如何执行？
11. LoRA 权重如何加载、命名映射和切分？
12. add_lora / remove_lora / pin_lora / list_loras 生命周期如何走？

### P1：重点回答

1. LoRAConfig 决定哪些能力边界？
2. Engine / Scheduler 层为什么只携带 LoRARequest，而不执行 LoRA forward？
3. manager 层和 layer 层的边界是什么？
4. punica wrapper / metadata 在 mixed LoRA batch 中有什么作用？
5. LoRA adapter cache 的淘汰策略会影响哪些场景？
6. LoRA 和量化 base model 共存时，dtype 和 scale 如何处理？
7. LoRA 和 TP / PP / DP 如何交互？
8. LoRA 和多模态模型是否有特殊限制？
9. LoRA 和 CUDA Graph 是否容易冲突？为什么？
10. LoRA 请求失败时应该如何区分 adapter 未加载、mapping 错误、layer 注入错误？

### P2：进阶追问

1. 如果 batch 中每个请求使用不同 LoRA，吞吐会发生什么变化？
2. 如果 adapter 频繁加载卸载，会对延迟和显存有什么影响？
3. 如果 LoRA rank 很大，会影响哪些 kernel 或 memory layout？
4. 如果 LoRA + quantization 输出异常，如何定位 base output 和 delta？
5. 如果 TP 下 LoRA 权重切分错误，会出现什么现象？
6. 如果 pin_lora 后显存无法释放，应该检查 manager 哪些状态？
7. 如何设计 benchmark 比较 single LoRA、multi LoRA mixed batch、no LoRA 的性能？

---

## 21. Multimodal 多模态问题

来源：`multimodal/`

### P0：必须回答

1. Multimodal 子系统在 vLLM V1 中是哪一层？负责什么？不负责什么？
2. image / audio / video / embeds 如何从用户请求进入 `EngineCoreRequest`？
3. `MultiModalRegistry`、`DataParser`、`Processor`、`Mapper` 各自负责什么？
4. `MultiModalFeatureSpec` 承载什么信息？
5. placeholder token 如何和 prompt token、encoder output 对齐？
6. `mm_position`、`mm_hash`、`identifier` 分别是什么？
7. Scheduler 为什么要调度 encoder input？
8. encoder budget 和 encoder cache 如何工作？
9. `EncoderCacheManager` 和 `GPUModelRunner.encoder_cache` 有什么区别？
10. GPUModelRunner 如何执行 `_execute_mm_encoder()`？
11. GPUModelRunner 如何执行 `_gather_mm_embeddings()` 和 `_preprocess()`？
12. 多模态模型类需要提供哪些接口？
13. 多模态输出如何回到普通 decoder forward、logits、sampling、output 链路？

### P1：重点回答

1. 多模态输入为什么不是简单把 image 当作普通 token？
2. processor cache 和 encoder cache 的区别是什么？
3. placeholder 对齐错误会导致什么问题？
4. encoder token budget 如何影响 chunked prefill？
5. 多模态 encoder output 什么时候可以复用？
6. 多模态和 decoder KV cache 有什么关系？
7. 多模态和 EC connector / KV connector 有什么关系？
8. 多模态和 LoRA / quantization / parallelism 如何组合？
9. 多模态模型如何处理 `inputs_embeds` 而不是 `input_ids`？
10. 多图、多视频输入会如何影响 Scheduler 和 ModelRunner？

### P2：进阶追问

1. 如果 image placeholder 数量和 encoder output 数量不一致，如何排查？
2. 如果多模态请求 TTFT 很高，是 processor、encoder、scheduler 还是 decoder 的瓶颈？
3. 如果 encoder cache 命中率低，应该检查 mm_hash 还是 workload？
4. 如果视频输入导致显存暴涨，如何分析 encoder cache 和 KV cache？
5. 如果多模态 + CUDA Graph 不 replay，可能是哪些动态 shape 导致？
6. 如果多模态 + PP 下 intermediate tensors 出错，应该检查哪个 stage？
7. 如何设计 benchmark 比较文本请求、多图请求、视频请求的 TTFT 和 throughput？

---

# 第五部分：阶段化问题路线

## 22. 第 1 个月逐周问题路线

### 第 1 周：总览与主链路

本周目标：能画出 vLLM 系统图。

必须完成的问题：

1. vLLM 的系统分层是什么？
2. 一次 generation 请求的完整对象流转是什么？
3. `EngineCore.step()` 为什么是核心闭环？
4. `SchedulerOutput`、`ModelRunnerOutput`、`EngineCoreOutputs`、`RequestOutput` 分别是什么？
5. vLLM 的性能来源有哪些？
6. generation、pooling、embedding 的链路差异是什么？
7. 哪些模块属于用户协议层，哪些模块属于核心执行层？
8. 哪些模块属于性能优化层？

输出物：

- 一张全链路图。
- 一张关键对象表。
- 10 个自己还不确定的问题。

### 第 2 周：配置与模型加载

本周目标：能讲清 vLLM 从用户参数到模型对象的启动链路。

必须完成的问题：

1. EngineArgs 如何变成 VllmConfig？
2. VllmConfig 中哪些子配置最影响推理性能？
3. ModelConfig 如何判断模型能力？
4. ModelRegistry 如何解析模型类？
5. model_loader 如何加载权重？
6. quantization 信息如何进入模型加载链路？
7. Worker.load_model() 在启动生命周期中的位置是什么？
8. KV cache 初始化为什么依赖显存 profiling？

输出物：

- 一张启动生命周期图。
- 一张配置对象影响范围表。

### 第 3 周：Engine 与 EngineCore

本周目标：能讲清外层 Engine 和内部 EngineCore 的边界。

必须完成的问题：

1. LLMEngine 和 AsyncLLM 的区别是什么？
2. InputProcessor 和 OutputProcessor 分别负责什么？
3. EngineCoreClient 为什么存在？
4. EngineCore 如何接收请求并交给 Scheduler？
5. EngineCore.step() 每一步的输入输出是什么？
6. 多进程 EngineCoreProc 如何包装 EngineCore？
7. abort、sleep、wake_up、profile 控制接口如何传递？
8. streaming 输出为什么依赖 OutputProcessor 状态？

输出物：

- 一张 Engine / EngineCore 边界图。
- 一份同步和异步路径对比表。

### 第 4 周：Sampling 与 Output

本周目标：能讲清从 logits 到用户输出的后半段链路。

必须完成的问题：

1. SamplingParams 如何影响 sampler？
2. logits、logprobs、prompt_logprobs 在哪里计算？
3. structured output grammar bitmask 如何应用？
4. ModelRunnerOutput 为什么不是用户输出？
5. Scheduler.update_from_output() 如何改变 Request 状态？
6. OutputProcessor 如何处理 detokenize、stop、streaming？
7. pooling output 和 generation output 的差异是什么？
8. 输出异常时如何沿链路排查？

输出物：

- 一张 logits 到 RequestOutput 的链路图。
- 一份采样参数作用表。

---

## 23. 第 2 个月逐周问题路线

### 第 5 周：Executor / Worker / ModelRunner

本周目标：能讲清调度计划如何变成真实模型 forward。

必须完成的问题：

1. Executor / Worker / ModelRunner 三层各自负责什么？
2. SchedulerOutput 如何进入执行层？
3. Worker 初始化和模型加载生命周期是什么？
4. ModelRunner 如何维护 persistent batch？
5. `_update_states()`、`_prepare_inputs()`、`_build_attention_metadata()` 分别做什么？
6. ModelRunner 如何处理 logits、pooling、sampling？
7. ModelRunnerOutput 如何返回 Scheduler？
8. LoRA、多模态、Spec Decode、KV connector 分别在哪个阶段挂接？

输出物：

- 一张 SchedulerOutput 到 ModelRunnerOutput 的链路图。
- 一份执行层对象职责表。

### 第 6 周：KV Cache 与 KV Transfer

本周目标：能讲清本地 KV block 和外部 KV transfer 的生命周期。

必须完成的问题：

1. KV cache 显存公式是什么？
2. KVCacheManager 和 BlockPool 的边界是什么？
3. prefix cache 如何命中？
4. allocate_slots 如何分配 block？
5. slot mapping 和 block table 如何连接 attention？
6. external KV hit 如何进入 Scheduler？
7. Worker 如何 load/save external KV？
8. invalid blocks、deferred free、finished_sending 如何保证正确性？

输出物：

- 一张 KV block 生命周期图。
- 一张 external KV load/save 图。

### 第 7 周：Attention 子系统

本周目标：能讲清 attention metadata、backend、paged KV cache 如何协作。

必须完成的问题：

1. AttentionBackend、MetadataBuilder、AttentionImplBase 分别做什么？
2. backend 如何选择？
3. CommonAttentionMetadata 包含什么？
4. prefill / decode / mixed batch metadata 差异是什么？
5. slot mapping 和 block table 如何被 attention 使用？
6. attention forward 如何读写 KV cache？
7. cascade attention 和 prefix cache 有什么关系？
8. CUDA Graph / compile 对 attention 有什么约束？

输出物：

- 一张 attention metadata 构造图。
- 一张 prefill vs decode attention 对比表。

### 第 8 周：Attention 方法族与算子初步

本周目标：能把 Attention 算法和底层 kernel 概念对应起来。

必须完成的问题：

1. FlashAttention 优化什么？
2. PagedAttention 优化什么？
3. FlashAttention 和 PagedAttention 的区别是什么？
4. MHA / MQA / GQA 如何影响 KV cache？
5. MLA 如何改变 KV cache 和 backend？
6. Sliding Window Attention 解决什么问题？
7. operator wrapper、backend、kernel 的区别是什么？
8. profiler 中 kernel 如何映射回 vLLM layer？

输出物：

- 一张 attention 技术对比表。
- 一张 layer -> operator -> backend -> kernel 映射图。

---

## 24. 第 3 个月逐周问题路线

### 第 9 周：Parallelism 并行体系

本周目标：能讲清 vLLM 多 GPU 并行的拓扑和通信。

必须完成的问题：

1. TP、PP、DP、EP、PCP、DCP 分别切什么？
2. world_size 和 world_size_across_dp 如何计算？
3. DCP 和 EP 为什么不单独乘进 world size？
4. Worker 初始化时如何建立 group？
5. TP 中哪些层需要 all-reduce？
6. PP 中 intermediate tensors 如何传递？
7. EP 中 token dispatch 和 all-to-all 如何发生？
8. logits 和 sampling 在哪个 rank 产生？

输出物：

- 一张 rank mesh 图。
- 一张并行维度与通信原语对照表。

### 第 10 周：KV Transfer / PD 分离 / KVPool 深入

本周目标：能分析 KVPool 和解耦式执行场景。

必须完成的问题：

1. external KV cache 与本地 prefix cache 的关系是什么？
2. KVPool hit 如何降低 prefill 成本？
3. decode worker 如何知道哪些 KV 已经可用？
4. load_kv_async 为什么需要 WAITING_FOR_REMOTE_KVS？
5. save KV 为什么可能要等 finished_sending 才能释放 block？
6. invalid block 如何触发 recompute 或 fail？
7. PD 分离中 prefill 和 decode 的资源需求有什么不同？
8. KV transfer 的网络成本如何影响收益？

输出物：

- 一张 PD / KVPool 端到端图。
- 一份 KV transfer 风险清单。

### 第 11 周：Spec Decode

本周目标：能讲清 Spec Decode 跨层协议。

必须完成的问题：

1. draft tokens 如何产生、调度、验证、回收？
2. Scheduler 如何调度 spec tokens？
3. SpecDecodeMetadata 如何描述 logits 布局？
4. RejectionSampler 如何保证输出正确？
5. accepted / recovered / bonus tokens 如何回到账本？
6. KV cache 和 num_computed_tokens 如何修正？
7. grammar 和 structured output 如何影响 draft tokens？
8. acceptance rate 如何影响收益？

输出物：

- 一张 spec decode 状态机图。
- 一份普通 decode vs spec decode 对比表。

### 第 12 周：Quantization

本周目标：能讲清量化从配置到 kernel 的完整链路。

必须完成的问题：

1. ModelConfig.quantization 和 VllmConfig.quant_config 的关系是什么？
2. QuantizationConfig 如何为 layer 选择 quant_method？
3. create_weights / load / process / apply 分别在何时发生？
4. weight-only quantization 与 activation quantization 区别是什么？
5. KV cache quantization 与权重量化区别是什么？
6. TP / PP / EP 下量化参数如何切分？
7. LoRA + quantization 如何共存？
8. 量化如何影响显存、吞吐、延迟、精度？

输出物：

- 一张量化全链路图。
- 一份量化方法取舍表。

---

## 25. 第 4 个月逐周问题路线

### 第 13 周：Model Architectures

本周目标：能理解 vLLM 如何适配不同模型。

必须完成的问题：

1. ModelRegistry 如何解析模型类？
2. ModelConfig 如何判断 task 和 runner_type？
3. model class 如何构造 layers？
4. ModelRunner 对 forward 接口有什么要求？
5. MoE、多模态、pooling 模型和 generation 模型差异是什么？
6. 权重加载如何处理 name mapping 和 TP shard？
7. Quantization、LoRA、parallelism 如何 hook 到模型层？
8. 新增模型 checklist 是什么？

输出物：

- 一张模型加载到 forward 的链路图。
- 一份新增模型 checklist。

### 第 14 周：CUDA Graph / Compilation

本周目标：能理解动态 batch 如何适配固定 graph replay。

必须完成的问题：

1. compile 和 CUDA Graph 分别优化什么？
2. CUDA Graph 为什么要求固定 shape / address / launch sequence？
3. vLLM 如何做 padding 和 shape bucket？
4. ModelRunner 如何决定 eager / compile / cudagraph replay？
5. attention metadata 如何 capture-compatible？
6. sampler 和 output 是否被 capture？
7. 哪些功能会导致 fallback？
8. 如何排查 cudagraph miss？

输出物：

- 一张 capture / replay 生命周期图。
- 一份 fallback 原因表。

### 第 15 周：LoRA 与 Multimodal

本周目标：能理解请求级扩展能力如何进入主链路。

必须完成的问题：

1. LoRARequest 如何进入 Scheduler 和 Worker？
2. LoRA manager 如何管理 adapter cache？
3. mixed LoRA batch 如何执行？
4. LoRA layer 如何注入 Linear / Embedding / LM head / MoE？
5. 多模态输入如何进入 EngineCoreRequest？
6. placeholder 如何与 encoder output 对齐？
7. Scheduler 如何调度 encoder input？
8. ModelRunner 如何把 multimodal embeddings 拼回 inputs_embeds？

输出物：

- 一张 LoRA request flow 图。
- 一张 multimodal request flow 图。

### 第 16 周：Operators、性能排查与体系复盘

本周目标：把所有模块串成可排查、可优化的系统能力。

必须完成的问题：

1. Layer / operator wrapper / backend / kernel 区别是什么？
2. attention、KV cache、quantization、MoE、sampling 各自有哪些算子族？
3. profiler 里 kernel 如何映射回 vLLM 模块？
4. TTFT 高如何排查？
5. TPOT 高如何排查？
6. throughput 低如何排查？
7. OOM 如何排查？
8. GPU 利用率低但延迟高如何排查？
9. 通信瓶颈如何排查？
10. 如何设计一个可靠 benchmark？

输出物：

- 一张性能排查决策树。
- 一份 4 个月总复盘。

---

# 第六部分：第 5 到第 8 个月深入问题路线

## 26. 第 5 个月：Operators / CUDA / Profiling 深入

### Week 17：Python 到 Kernel 调用链

1. `_custom_ops.py` 如何包装 torch custom ops？
2. Python wrapper 如何决定调用 custom CUDA、Triton、third-party backend 或 torch fallback？
3. custom op 的 fake impl 对 compile 有什么作用？
4. C++ binding 和 CUDA kernel 如何注册到 `torch.ops`？
5. 如何从 Python stack trace 找到最终 kernel？
6. 如何从 profiler kernel name 找回 Python layer？
7. 如果 custom op 找不到，如何排查 build、import、symbol、ABI？
8. 如何为一个 op 设计 torch fallback？

### Week 18：Attention / KV Cache Kernel 深入

1. decode attention kernel 的内存访问模式是什么？
2. prefill attention kernel 的计算模式是什么？
3. paged KV cache layout 如何影响 coalesced access？
4. KV cache write kernel 如何根据 slot mapping 写入？
5. FP8 KV cache 读写和 scale 应该在哪里处理？
6. sliding window / cascade attention 对 kernel 参数有什么影响？
7. 如何用 Nsight 判断 attention 是 memory-bound 还是 compute-bound？
8. 如何比较 FlashAttention、FlashInfer、Triton backend？

### Week 19：Quantized GEMM / MoE Kernel 深入

1. quantized linear 和普通 linear 的 kernel 差异是什么？
2. Marlin / Machete / CUTLASS 分别适合哪些权重格式？
3. packed qweight 如何影响 memory layout？
4. activation scale 如何进入 kernel？
5. fused MoE kernel 如何处理 routing、expert batching、GEMM？
6. MoE token dispatch 和 all-to-all 如何影响 kernel 利用率？
7. 如何定位 MoE 是通信瓶颈还是 compute 瓶颈？
8. 如何 benchmark dense model 和 MoE model 的推理差异？

### Week 20：Profiling 方法论

1. TTFT、TPOT、ITL、throughput、GPU utilization 分别如何采集？
2. Nsight Systems 和 Nsight Compute 分别适合看什么？
3. torch profiler 在 vLLM serving 中有什么局限？
4. 如何区分 CPU 调度瓶颈、GPU kernel 瓶颈、NCCL 通信瓶颈？
5. 如何分析 kernel launch overhead？
6. 如何分析 HBM bandwidth？
7. 如何分析 graph replay 是否生效？
8. 如何把 profiler 结论转成代码优化或配置调整？

---

## 27. 第 6 个月：分布式与大规模 Serving 深入

### Week 21：TP / PP / DP 组合实验

1. TP size 增大为什么不一定线性提速？
2. PP stage 如何划分更合理？
3. DP replica 如何做负载均衡？
4. max_num_batched_tokens 和 TP size 的关系是什么？
5. PP 下 chunked prefill 和 decode 如何影响 bubble？
6. 多节点 TP 和单节点 TP 的通信差异是什么？
7. 如何 benchmark TP/PP/DP 组合？
8. 如何选择一个模型的并行策略？

### Week 22：EP / MoE / EPLB

1. Expert Parallel 切分 experts 的基本策略是什么？
2. Router 输出如何变成 expert dispatch？
3. All-to-All 在 MoE 中传什么？
4. expert load imbalance 如何影响延迟？
5. EPLB 如何缓解负载不均？
6. MoE KV cache 和 dense model 有什么不同？
7. MoE + quantization 有哪些特殊问题？
8. MoE + CUDA Graph 是否容易受动态 routing 影响？

### Week 23：PD 分离与 KV Transfer 深入

1. Prefill 和 Decode 的资源画像有什么不同？
2. PD 分离如何提升资源利用率？
3. KV transfer 的单位是 token、block、layer 还是 tensor？
4. Pull、Push、Layerwise Push 的差异是什么？
5. KV transfer 如何保证 decode 端读取到正确 block？
6. Prefill 端什么时候可以释放 KV？
7. 网络带宽不足时 PD 分离可能变差吗？
8. 如何设计 PD 分离的端到端 benchmark？

### Week 24：稳定性与故障恢复

1. Worker crash 后 EngineCore 如何感知？
2. KV transfer 失败后请求应该 fail 还是 recompute？
3. NCCL hang 如何定位？
4. 多进程 worker 某个 rank 卡住如何排查？
5. request cancellation 如何释放跨模块状态？
6. OOM 后能否恢复？哪些状态需要清理？
7. 外部 connector 超时时 Scheduler 如何处理？
8. 如何设计健康检查和指标？

---

## 28. 第 7 个月：高级能力组合深入

### Week 25：LoRA 组合场景

1. multi-LoRA mixed batch 为什么复杂？
2. LoRA adapter cache 如何影响 tail latency？
3. LoRA + quantization 的数值路径如何保证正确？
4. LoRA + TP 的权重切分如何验证？
5. LoRA + CUDA Graph 需要哪些 shape 和 metadata 稳定性？
6. LoRA + MoE 是否会影响 expert layer？
7. LoRA + 多模态模型 adapter 是否可能只作用于部分 tower？
8. 如何设计 LoRA serving 的容量规划？

### Week 26：Multimodal 组合场景

1. 多模态 processor 是 CPU 瓶颈还是 GPU 瓶颈？
2. encoder budget 如何影响文本 token 调度？
3. encoder cache 和 KV cache 如何同时管理？
4. 多图 / 视频输入如何影响 TTFT？
5. 多模态 + PP 时 encoder 在哪个 stage 执行？
6. 多模态 + LoRA 时 adapter 是否作用于 vision tower？
7. 多模态 + Spec Decode 是否有特殊限制？
8. 如何 benchmark 图文、多图、视频请求？

### Week 27：Structured Output / Sampling / Spec Decode 组合

1. grammar bitmask 如何影响 top-k / top-p？
2. structured output 和 spec decode 如何同时保证 token 合法？
3. 如果 draft token 不满足 grammar，如何处理？
4. logprobs 和 grammar mask 的顺序如何影响结果？
5. streaming JSON 输出如何避免中间态误判？
6. best_of / n / beam search 类需求会如何影响 Scheduler？
7. request-level sampling params 如何映射到 batch-level sampler state？
8. 如何验证结构化输出正确性？

### Week 28：Long Context 综合问题

1. 长上下文下 TTFT 为什么高？
2. 长上下文下 KV cache 为什么是主要显存瓶颈？
3. chunked prefill 如何缓解 decode 饥饿？
4. prefix cache 对长上下文 RAG 有什么收益？
5. sliding window / attention sink / KV compression / KV quant / offload 如何取舍？
6. 长上下文 + TP / CP / DCP 如何组合？
7. 长上下文 + PD 分离是否更有价值？
8. 如何设计长上下文 workload benchmark？

---

## 29. 第 8 个月：源码改造与验证

### Week 29：Scheduler 小改造

1. 你能否增加一个新的 scheduler metric？
2. 你能否记录每轮 prefill / decode token 数？
3. 你能否统计 chunked prefill 被触发的次数？
4. 你能否统计每个请求等待时间和运行时间？
5. 你能否改变 waiting queue 的优先级策略？
6. 你能否验证策略是否改善 P95 latency？
7. 你能否确保改动不破坏 request lifecycle？
8. 你能否写出最小测试或 benchmark？

### Week 30：KV Cache 小改造

1. 你能否增加 prefix cache hit / miss 更细粒度统计？
2. 你能否输出每个请求分配的 block 数？
3. 你能否跟踪 block ref_cnt 生命周期？
4. 你能否检测 deferred free 堆积？
5. 你能否模拟 invalid block 并验证 recompute？
6. 你能否比较开启和关闭 prefix cache 的 TTFT？
7. 你能否分析不同 block size 对碎片和吞吐的影响？
8. 你能否写出 KV cache 状态可视化？

### Week 31：Execution / Attention 小改造

1. 你能否记录每轮 attention metadata 的关键 shape？
2. 你能否输出 prefill / decode / mixed batch 分类？
3. 你能否统计每种 attention backend 被选择的次数？
4. 你能否强制切换 backend 并比较性能？
5. 你能否定位 slot mapping 的构造位置并加 debug？
6. 你能否验证 CUDA Graph replay 与 attention metadata 的关系？
7. 你能否找出某个 kernel 在 profiler 中的上层来源？
8. 你能否写一份 attention 执行链路调试指南？

### Week 32：最终体系复盘

1. 你能否从 0 画出 vLLM 全系统图？
2. 你能否从请求、调度、KV、执行、attention、kernel、输出 7 层讲一次完整链路？
3. 你能否解释 10 个关键对象的职责和生命周期？
4. 你能否解释 5 个性能指标和对应瓶颈？
5. 你能否解释 5 个高级能力如何接入主链路？
6. 你能否解释 3 个真实改造或实验结果？
7. 你能否说清当前体系里自己仍然薄弱的模块？
8. 你能否给下一阶段定 3 个明确深入方向？

---

# 第七部分：横向专题问题库

## 30. 性能指标与排查问题

### TTFT 高

1. prompt 是否过长？
2. prefill batch 是否过大？
3. chunked prefill 是否开启？参数是否合适？
4. tokenizer / processor 是否成为 CPU 瓶颈？
5. 多模态 encoder 是否占用大量时间？
6. prefix cache 是否命中？为什么没有命中？
7. KVPool 是否命中？external KV load 是否慢？
8. Scheduler waiting queue 是否堆积？
9. GPU prefill kernel 是否 compute-bound？
10. TP / PP 通信是否拖慢 prefill？

### TPOT / ITL 高

1. decode batch 是否过小？
2. CUDA Graph replay 是否生效？
3. decode attention 是否 memory-bound？
4. KV cache layout 是否适合当前 backend？
5. kernel launch overhead 是否明显？
6. sampling / logprobs 是否成为瓶颈？
7. structured output grammar 是否过慢？
8. TP all-reduce 是否过慢？
9. Spec Decode acceptance rate 是否足够高？
10. LoRA mixed batch 是否降低 kernel 效率？

### Throughput 低

1. max_num_batched_tokens 是否过小？
2. max_num_seqs 是否过小？
3. GPU memory utilization 是否限制 KV blocks？
4. batch 中 prefill / decode 混合是否合理？
5. Scheduler 是否过于偏向低延迟导致吞吐不足？
6. quantization 是否真正走到高效 kernel？
7. CUDA Graph / compile 是否启用并生效？
8. TP / DP 配置是否匹配模型和硬件？
9. CPU entrypoints / detokenize / streaming 是否成为瓶颈？
10. 网络传输或 external connector 是否拖慢整体？

### OOM

1. 模型权重显存占用是多少？
2. KV cache 预算是多少？
3. block size、max_model_len、max_num_seqs 如何影响 KV cache？
4. 是否启用 LoRA / 多模态 encoder cache / graph capture 额外显存？
5. CUDA Graph capture 是否占用额外 buffer？
6. quantization 是否降低了权重但没有降低 KV cache？
7. prefix cache ref_cnt 是否导致 blocks 不能释放？
8. deferred free 是否堆积？
9. request cancellation 是否正确释放资源？
10. 多进程 / 多 rank 显存是否均衡？

### GPU 利用率低

1. 请求并发是否不足？
2. CPU tokenization / processor 是否拖慢？
3. Scheduler 是否频繁产生小 batch？
4. decode batch 是否太小？
5. kernel launch overhead 是否大？
6. 通信同步是否导致 GPU 等待？
7. external KV load 是否阻塞执行？
8. streaming / detokenize 是否反压？
9. CUDA Graph 是否没有 replay？
10. benchmark 是否包含 warmup 和稳定阶段？

---

## 31. 正确性问题库

1. KV cache 复用为什么可能导致输出污染？
2. prefix cache key 少包含一个字段会有什么风险？
3. slot mapping 错误会污染哪些 token？
4. block table 错误会让 attention 读到什么？
5. deferred free 如果过早释放会发生什么？
6. spec decode 如果错误接受 draft token 会发生什么？
7. grammar bitmask 如果和 sampler 对不上会发生什么？
8. LoRA adapter mapping 错误会发生什么？
9. 多模态 placeholder 对齐错误会发生什么？
10. TP rank 权重 shard 错误会发生什么？
11. PP intermediate tensors 顺序错误会发生什么？
12. quant scale 错误会发生什么？
13. KV cache dtype 和 attention backend 不匹配会发生什么？
14. tokenizer 和 model config 不匹配会发生什么？
15. request abort 后状态未清理会发生什么？

---

## 32. 系统设计问题库

1. 如果让你设计一个 LLM serving engine，你会如何分层？
2. 为什么需要 continuous batching？
3. 为什么需要 paged KV cache？
4. 为什么需要 prefix cache？
5. 为什么需要 chunked prefill？
6. 为什么需要把 Scheduler 和 Worker 分离？
7. 为什么需要 EngineCoreClient？
8. 为什么需要 attention backend 抽象？
9. 为什么需要 KV Connector？
10. 为什么需要 CUDA Graph？
11. 为什么需要 torch.compile？
12. 为什么需要 quantization config 和 quant_method 两层抽象？
13. 为什么 LoRA 是请求级 adapter，而不是重新加载模型？
14. 为什么多模态需要 encoder budget？
15. 为什么 Spec Decode 要改 Scheduler 而不只是改 sampler？
16. 为什么分布式需要 parallel_state 和 group coordinator？
17. 为什么 output processor 不能合并进 Scheduler？
18. 为什么 model architecture 要统一 forward 接口？
19. 为什么 operators 需要 fallback？
20. 为什么 benchmark 必须区分 prefill 和 decode？

---

## 33. 每月复盘问题

### 第 1 个月复盘

1. 你能否不看资料画出主链路？
2. 你能否解释 10 个核心对象？
3. 你能否讲清 Engine 和 EngineCore 的边界？
4. 你能否讲清配置到模型加载的生命周期？
5. 你能否讲清 logits 到 RequestOutput 的链路？
6. 你还有哪些对象只知道名字但不会解释？

### 第 2 个月复盘

1. 你能否讲清 SchedulerOutput 如何变成 forward 输入？
2. 你能否讲清 KV block 生命周期？
3. 你能否讲清 prefix cache 命中和释放？
4. 你能否讲清 slot mapping 和 block table？
5. 你能否讲清 attention backend 和 metadata builder？
6. 你能否比较 FlashAttention 和 PagedAttention？

### 第 3 个月复盘

1. 你能否讲清 TP / PP / DP / EP？
2. 你能否讲清 KV Transfer / KVPool？
3. 你能否讲清 Spec Decode 跨层协议？
4. 你能否讲清量化全链路？
5. 你能否讲清 model architecture 适配机制？
6. 你能否设计 3 个性能实验？

### 第 4 个月复盘

1. 你能否讲清 CUDA Graph 为什么需要固定 shape？
2. 你能否讲清 operators / backend / kernel？
3. 你能否讲清 LoRA request flow？
4. 你能否讲清 multimodal request flow？
5. 你能否写出性能排查决策树？
6. 你能否把 4 个月内容压缩成一张系统图？

### 第 8 个月复盘

1. 你是否完成至少 3 个源码级小改造？
2. 你是否完成至少 5 组 benchmark？
3. 你是否能解释每个 benchmark 的瓶颈？
4. 你是否能从 profiler 映射到源码模块？
5. 你是否能独立定位一个输出正确性问题？
6. 你是否能独立定位一个性能问题？
7. 你是否能新增一个小特性并验证？
8. 你是否形成了自己的大模型推理系统知识图谱？

---

## 34. 最终使用建议

1. 不要一次性回答所有问题。每周选 20 到 40 个问题，先写短答案，再补源码证据。
2. 每个阶段至少画一张图。图比文字更能暴露链路是否真的理解。
3. 每个模块都要区分「负责什么」和「不负责什么」。大多数混乱来自边界不清。
4. 每个机制都要追问「优化什么指标」和「牺牲什么」。
5. 每个高级特性都要放回主链路里理解，避免孤立学习。
6. 每个月至少做一次总复盘，把问题重新按模块归类。
7. 第 4 个月结束后，重点从“会解释”转向“会验证、会排查、会改代码”。
8. 第 8 个月结束时，目标是形成自己的推理系统方法论：链路、状态、资源、性能、正确性、验证。
