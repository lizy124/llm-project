# vLLM V1 16 个专题总体背诵总览

本文把 `abstract` 目录下 16 个专题的结尾总结重新整理成一篇整体背诵文档。每一节都按学习顺序展开，目标是先建立 vLLM V1 的全局理解，再进入各专题细节。

整体主线可以先背成一句话：

```text
vLLM 从用户参数和请求出发，先构造配置并加载模型，再通过 Engine / EngineCore / Scheduler / Executor / Worker / ModelRunner 完成请求调度、模型执行、attention、采样和输出；在主链路之外，parallelism、quantization、LoRA、multimodal、spec decode、compile / CUDA graph 和 KV transfer 作为横向能力接入不同阶段。
```

## 01. config_and_model_loading

`config_and_model_loading` 是 vLLM 启动链路的基础专题。它解释用户通过 CLI 或 Python API 传入的松散参数，如何被 `EngineArgs` 收拢，再被拆分和推导成 `ModelConfig`、`LoadConfig`、`CacheConfig`、`ParallelConfig`、`SchedulerConfig`、`CompilationConfig` 等子配置，最后聚合为贯穿全系统的 `VllmConfig`。其中 `ModelConfig` 负责读取 Hugging Face config，理解模型架构、任务类型、dtype、最大上下文长度、量化方式、多模态能力、MoE 能力等；`LoadConfig` 负责决定权重格式和 loader；`VllmConfig` 则作为 Engine、Executor、Worker、ModelRunner 和 model loader 的共同上下文。

这个专题最重要的边界是：配置阶段决定“要加载什么、怎么加载、运行时有哪些约束”，但不会真正加载完整模型权重。真正的模型加载发生在 Worker 侧，由 `GPUModelRunner.load_model()` 调用 `get_model_loader()`、`initialize_model()` 和 `loader.load_weights()`，创建真实 `nn.Module` 并把 checkpoint tensor 写入模型参数。之后才会进行显存 profile、KV cache 初始化、warmup、compile 和 CUDA graph capture。因此，这一专题建立的是从“用户参数”到“设备侧可执行模型”的启动心智模型。

## 02. model_architectures

`model_architectures` 解释 vLLM 如何把各种外部模型适配成统一可执行的模型类。外部模型来自 Hugging Face config、`architectures` 字段和 checkpoint 命名，而 vLLM 内部希望 ModelRunner 能用统一方式调用 `forward()`、`compute_logits()`、`pooler()` 和 `load_weights()`。这中间的关键层就是 `ModelRegistry` 和具体 model class。`ModelRegistry.inspect_model_cls()` 在配置阶段检查模型能力，判断它是否支持 generation、pooling、多模态、pipeline parallel、LoRA、特殊 attention 或 transcription；`resolve_model_cls()` 在加载阶段返回真正的模型类。

模型类本身负责构造 embedding、decoder layers、attention、MLP、norm、lm_head、pooler、多模态 tower、MoE experts 等结构，并通过 `load_weights()` 处理 checkpoint 到 runtime 参数的映射。这里要特别理解 checkpoint 命名和 runtime 参数命名可能不同，例如 HF 里是 `q_proj/k_proj/v_proj`，vLLM 里可能是 fused `qkv_proj`；HF 里是 `gate_proj/up_proj`，vLLM 里可能是 `gate_up_proj`。TP shard、PP missing layer、tie embeddings、量化 scale、LoRA target modules 都需要在架构层正确适配。这个专题的核心价值，是让 Llama、Qwen、DeepSeek、Mixtral、BERT、Whisper、VLM、MoE 等不同模型都能落到同一套 vLLM 执行框架里。

## 03. engine

`engine` 是外层推理引擎专题，解释用户请求如何进入 vLLM，以及内部输出如何变成用户可见输出。vLLM V1 中的 Engine 不是单个固定类，而是外层体系的统称，主要形态包括同步路径的 `LLMEngine` 和异步路径的 `AsyncLLM`。它们都不直接做 token 调度和模型 forward，而是通过 `InputProcessor` 把用户输入转成 `EngineCoreRequest`，通过 `OutputProcessor` 把 `EngineCoreOutputs` 转成 `RequestOutput` 或 `PoolingRequestOutput`，并通过 `EngineCoreClient` 驱动内部 `EngineCore`。

同步路径中，调用方通过 `LLMEngine.add_request()` 添加请求，再通过 `step()` 主动拉取输出。异步路径中，`AsyncLLM.generate()` 返回 async generator，后台 `output_handler` 持续从 `EngineCoreClient` 拉取输出，并由 `OutputProcessor` 推入每个请求自己的 queue。Engine 这一层的边界非常重要：它负责用户接口、request id、输入预处理、输出 detokenize、stop string 检查、streaming queue、abort 和控制接口转发；但不负责 waiting / running 队列、token budget、KV block 分配、prefix cache、模型 forward 或 sampling。可以背成：Engine 管输入输出，EngineCore 管内部执行闭环，Scheduler 管调度账本，Worker / ModelRunner 管模型计算。

## 04. engine_core

`engine_core` 是 vLLM V1 的内部执行闭环总控。外层 Engine 把已经标准化的 `EngineCoreRequest` 交给 EngineCore，EngineCore 先把它转换成 Scheduler 使用的 `Request`，再交给 `Scheduler.add_request()`。每一轮 `EngineCore.step()` 都围绕三步展开：`Scheduler.schedule()` 生成本轮执行计划 `SchedulerOutput`，`model_executor.execute_model()` 把计划交给 Executor、Worker 和 ModelRunner 执行，最后 `Scheduler.update_from_output()` 用 `SchedulerOutput` 和 `ModelRunnerOutput` 对账，生成 `EngineCoreOutputs` 返回外层 Engine。

EngineCore 本身不是 Scheduler，也不是 Worker。它不决定每个请求调度多少 token，不管理 KV block 抢占细节，也不构造 input_ids、attention metadata 或最终用户输出。它的职责是把 Scheduler、model_executor、structured output、KV cache 初始化、batch queue、abort queue、sleep / wake_up、profile、LoRA 控制等能力编排成一个内部 loop。要特别记住三个对象：`SchedulerOutput` 是计划，`ModelRunnerOutput` 是 Worker 执行结果，`EngineCoreOutputs` 是 Scheduler 消化结果后返回外层的内部输出协议。EngineCore 的核心价值，是把请求从外层 Engine 带入内部 schedule、execute、update 的闭环。

## 05. scheduler

`scheduler` 是 vLLM V1 的调度和请求状态账本中心。它维护所有未释放请求的 `requests` 字典，以及 `waiting`、`skipped_waiting`、`running` 三类队列。每轮 `schedule()` 先从 running 请求开始，计算它们还需要跑多少 token，为新增 token 分配 KV block；如果 KV block 不够，按策略抢占其他 running 请求。只有在没有抢占且 Scheduler 未暂停新请求时，才会调度 waiting 请求。waiting 请求会先查本地 prefix cache，再查外部 KV connector 命中，然后根据 token budget、chunked prefill、encoder budget、LoRA active 数、KV capacity 等条件决定是否进入 running。

Scheduler 生成的 `SchedulerOutput` 是本轮执行说明书，包含新请求、cached 请求、每个请求调度 token 数、spec decode tokens、多模态 encoder inputs、KV connector metadata、需要清零的新 block、已完成请求等。Worker 返回 `ModelRunnerOutput` 后，`update_from_output()` 会把 sampled tokens、logprobs、pooling output、spec decode 接受/拒绝、grammar 状态、KV connector 回执和资源释放全部对回 Request 状态。Scheduler 的核心不是“跑模型”，而是维护 token 进度、KV block 账本、请求生命周期和输出对账。可以背成：schedule 发任务，update_from_output 收结果并落账。

## 06. executor_worker_model_runner

`executor_worker_model_runner` 是 vLLM V1 的执行层专题，解释 `SchedulerOutput` 如何真正进入设备并变成模型计算。三层分工很清楚：Executor 是 EngineCore 和 Worker 之间的分发层，负责选择单进程、多进程、Ray 或 external launcher 等执行后端，并通过 collective RPC 把命令发给 Worker；Worker 是设备侧承载对象，负责初始化 CUDA device、distributed environment、加载模型、profile 显存、初始化 KV cache、warmup、compile、sleep、wake_up、shutdown 等生命周期；ModelRunner 是 Worker 内部真正执行一轮 batch 的核心。

`GPUModelRunner.execute_model()` 会先 `_update_states()`，把 SchedulerOutput 合入 worker 侧 persistent batch；再 `_prepare_inputs()`，把请求状态压平成 input_ids、positions、query_start_loc、slot mapping、logits_indices 等 token 级张量；再 `_build_attention_metadata()`，构造 attention backend 需要的 metadata；然后 `_preprocess()` 处理 multimodal encoder、prompt embeds、encoder_outputs、PP intermediate tensors 等；最后 `_model_forward()` 调用真实模型。如果是 generation 模型，还会通过 `sample_tokens()` 消费 logits、grammar bitmask 和 sampler state，构造 `ModelRunnerOutput`。这一专题的核心是：Scheduler 决定怎么跑，执行层负责把计划跑完，Scheduler 再用结果更新请求状态。

## 07. attention

`attention` 专题解释 vLLM V1 如何把调度结果、KV cache、模型 attention layer 和底层 backend 串起来。它不是单独一个 FlashAttention 调用，而是一条从 `SchedulerOutput` 到 backend kernel 的翻译链。Scheduler 负责决定本轮跑哪些 token，并通过 KVCacheManager 分配 block；ModelRunner 在 `_prepare_inputs()` 中根据请求状态计算 positions、query_start_loc、seq_lens 和 slot mapping，在 `_build_attention_metadata()` 中构造 `CommonAttentionMetadata`，再由每个 backend 的 `AttentionMetadataBuilder` 转成 backend-specific metadata。

要理解两个关键结构：block table 和 slot mapping。block table 是请求级映射，表示 request 的逻辑 token block 对应哪些物理 KV block，主要用于读取历史 KV；slot mapping 是 token 级映射，表示本轮每个 token 的 key/value 要写入哪个 KV cache slot，主要用于写当前 KV。模型 forward 前，ModelRunner 通过 `ForwardContext` 注入 attention metadata、slot mapping、cudagraph mode 等信息；模型执行到某层 Attention layer 时，该层从 context 取出自己的 metadata、KV cache 和 slot mapping，先写当前 K/V，再由 `AttentionImpl` 调用 FlashAttention、FlashInfer、Triton、MLA 或其他 backend 读取历史 KV 并计算 hidden states。attention 输出不是 token，而是后续 logits 和 sampler 的输入。

## 08. sampling_and_output

`sampling_and_output` 是从模型数值结果到用户可见输出的后半段。ModelRunner 先用 `logits_indices` 从 hidden states 中选出需要计算 logits 的位置，再调用 `model.compute_logits()` 得到 vocab logits。之后会按顺序应用 structured output 的 grammar bitmask、logit bias、allowed tokens、bad words、presence/frequency/repetition penalties、temperature、min-p、top-k、top-p 等采样约束，最后通过普通 Sampler 或 spec decode 的 RejectionSampler 产生 sampled token ids。需要 logprobs 或 prompt logprobs 时，还会额外计算对应概率信息。

Worker 返回给 Scheduler 的不是用户输出，而是 `ModelRunnerOutput`，其中包括 `sampled_token_ids`、`logprobs`、`prompt_logprobs_dict`、`pooler_output`、KV connector output、CUDA graph stats 等内部结果。Scheduler 使用本轮 `SchedulerOutput` 作为计划账本，把 batch 级结果重新对回每个 Request，append 新 token，检查 EOS、stop token、max token、spec decode accepted/rejected、grammar 状态和资源释放，生成 `EngineCoreOutputs`。最后 `OutputProcessor` 才负责 detokenize、stop string 检查、streaming delta、logprobs 格式化、finish_reason 和 `RequestOutput` 构造。因此这一专题可以背成三层：ModelRunner 选 token，Scheduler 落状态，OutputProcessor 变用户输出。

## 09. operators

`operators` 专题讲 vLLM 中模型计算最终如何落到硬件 backend。它位于 Model Layer 之下，是 attention、KV cache、linear、MoE、norm、activation、RoPE、sampling、logprobs、communication 等计算真正执行的地方。要区分四个词：Layer 表达模型要算什么，例如 Attention layer、Linear、RMSNorm、FusedMoE；operator wrapper 是 Python 调用入口，负责整理参数、分配输出、判断 dtype / shape / platform，并调用 torch.ops、Triton 或第三方库；backend 是一组具体实现，例如 FlashAttention、FlashInfer、CUTLASS、Marlin、Triton、torch fallback；kernel 才是 profiler 中看到的 CUDA、Triton、NCCL 或 aten 执行单元。

算子路径由很多因素共同决定：平台、GPU 架构、dtype、head size、hidden size、block size、模型结构、量化方法、attention backend、CUDA graph mode、TP / PP / DP / EP / CP、LoRA、spec decode、structured output、logprobs、KV connector 以及依赖库是否可用。一个算子问题通常不能只看某个 kernel 函数，而要同时检查 rank-local shape、metadata 是否对齐、backend 是否 fallback、通信顺序是否一致、CUDA graph 是否改变 padding。这个专题的价值，是让你从底层理解 vLLM 性能和报错来源：ModelRunner 准备“本轮怎么跑”，model layer 表达“模型要算什么”，operator backend 决定“这段计算由哪个 kernel 跑”。

## 10. parallelism

`parallelism` 专题建立 vLLM 多卡执行的全局模型。并行不是一个通信 API，而是“切分对象 + rank group + 通信原语 + 状态归属 + 输出合并”的组合。TP 切单层 tensor、attention heads、MLP、embedding 和 vocab；PP 切 Transformer layers，不同 stage 之间传 `IntermediateTensors`；DP 切请求和模型副本，不同 replica 服务不同请求；EP 服务 MoE，把 token 按 router 结果 all-to-all 发到持有目标 expert 的 rank；PCP 切 prefill context；DCP 切 decode context，但复用 TP ranks，不增加 world size，并需要 softmax LSE merge。

核心公式是：单个 DP replica 内 `world_size = PP * TP * PCP`，所有 replica 总数再乘 DP，得到 `world_size_across_dp`。DCP、EP、EPLB 不单独乘进 world size，而是在已有 rank 上建立 group。vLLM 会把 ranks 组织成 `all_ranks[external_dp, dp, pp, pcp, tp]` 的 rank mesh，再从中派生 TP group、PP group、DP group、PCP group、DCP group、EP group。执行时，Executor 把 `SchedulerOutput` 分发到对应 Workers，模型层按 group 做本地计算和通信，通常只有 last PP stage 的输出 rank 产生 `ModelRunnerOutput`。这一专题的核心是：先看切什么，再看哪些 rank 合作，再看用什么通信对齐。

## 11. quantization

`quantization` 专题说明 vLLM 的量化不是一个简单开关，而是跨配置、模型层、权重加载、参数布局和 kernel dispatch 的协议。用户参数或 checkpoint metadata 先进入 `ModelConfig.quantization`，再生成 `VllmConfig.quant_config`，这是全局 `QuantizationConfig` 对象。模型 layer 构造时根据这个对象获取自己的 `quant_method`，由 `quant_method.create_weights()` 创建 qweight、scale、zero point、g_idx 等量化参数。loader 负责把 checkpoint tensor 写入这些参数，`process_weights_after_loading()` 再把它们 repack、transpose、finalize 或转换成 kernel-ready layout。

forward 时，量化 layer 不走普通 dense weight，而是由 `quant_method.apply()` 或 fused MoE、attention backend 调用特定量化 kernel。要区分几条主线：权重量化主要作用于 Linear、QKV、MLP、LM head、MoE experts；activation / dynamic quantization 会在 forward 中处理 per-token 或 per-block scale；KV cache quantization 由 `kv_cache_dtype`、`KVQuantMode` 和 attention backend 决定，控制 runtime key/value 的存储格式。`load_format` 也不是量化方法，它只决定怎么读文件；`--quantization` 决定权重和 layer 如何解释与执行。量化专题的背诵重点是：QuantizationConfig 管全局策略，quant_method 管单层参数和执行，weight_loader 管 checkpoint 映射，backend 管最终 kernel。

## 12. lora_and_adapters

`lora_and_adapters` 专题解释 vLLM 如何支持请求级动态 adapter。base model 在 Worker 初始化时固定加载，所有请求共享；LoRA adapter 可以动态 add、remove、pin、list，并由请求通过 `LoRARequest` 指定使用哪个 adapter。这个请求级信息会从 `EngineCoreRequest.lora_request` 进入 `Request.lora_request`，再通过 `SchedulerOutput.NewRequestData.lora_request` 下发给 Worker。Scheduler 本身不执行 LoRA forward，但会考虑 batch 内 active LoRA 数量不能超过 `max_loras`。

Worker / ModelRunner 侧会把请求级 LoRA 转成 batch 和 token 级 mapping。`InputBatch.request_lora_mapping` 表示 req_index 到 lora_id；`make_lora_inputs()` 根据本轮 scheduled token 数展开成 `token_lora_mapping` 和 `prompt_lora_mapping`；`LoRAModelRunnerMixin.set_active_loras()` 把这些 mapping 交给 `WorkerLoRAManager`。manager 保证 adapter 权重已加载到 GPU slot，并让 `LoRAModelManager` 更新 punica metadata。模型加载阶段，LoRA target modules 已经被替换成 LoRA wrapper，例如 ColumnParallelLinear、RowParallelLinear、QKVParallelLinear、Embedding、LM head、MoE。forward 时每个 LoRA layer 计算 base output，再按 token mapping 叠加 `x @ A @ B * scaling`。核心边界是：LoRARequest 是选择信息，manager 保证权重可用，InputBatch 提供 token mapping，LoRA layer 负责实际 delta。

## 13. multimodal

`multimodal` 专题讲 vLLM 如何把 image、audio、video、prompt_embeds 等非文本输入接入普通模型执行链路。用户输入先经过 entrypoints、renderer、`MultiModalDataParser` 和模型注册的 `MultiModalProcessor`，变成 `MultiModalInput`，里面包含带 placeholder 的 `prompt_token_ids`、模型 encoder 需要的 `mm_kwargs`、每个多模态 item 在 prompt 中的位置 `mm_placeholders`，以及用于缓存的 `mm_hashes`。之后 `InputProcessor` 按 placeholder 位置排序，构造 `MultiModalFeatureSpec`，写入 `EngineCoreRequest.mm_features` 和 Scheduler 侧 `Request.mm_features`。

Scheduler 不运行多模态 encoder，但必须判断本轮 decoder token window 是否触达某个 placeholder。如果触达，就要确保对应 encoder output 已经可用，或者本轮有 encoder compute budget 和 encoder cache 空间去执行它。调度结果通过 `SchedulerOutput.scheduled_encoder_inputs` 下发给 ModelRunner。Worker 侧 `GPUModelRunner._execute_mm_encoder()` 根据这些索引调用 `model.embed_multimodal()`，把输出写入 `encoder_cache[identifier]`；`_gather_mm_embeddings()` 再按当前 batch 顺序、token window 和 `mm_position` 取出对应 embedding slice；最后 `_preprocess()` 调 `model.embed_input_ids()`，把文本 token embedding 和多模态 embedding 合成 `inputs_embeds`。因此多模态最终不是绕过普通 forward，而是把非文本输入转换成 embeddings 后汇入普通 decoder / logits / sampling 链路。

## 14. spec_decode

`spec_decode` 专题解释 speculative decoding 如何跨层工作。普通 decode 一次 target model forward 通常只生成一个 token；spec decode 则先由 drafter 猜多个 draft tokens，再让 target model 一次 forward 验证这些 tokens，通过 rejection sampling 接受一段前缀。如果某个 draft 被拒绝，就从修正分布采一个 recovered token；如果全部接受，就再补一个 bonus token。最终写入请求的不是原始 draft，而是 accepted、recovered、bonus 后的真实输出 token。

Scheduler 用 `Request.spec_token_ids` 保存下一轮待验证 draft tokens，并用 `num_tokens_with_spec = num_tokens + len(spec_token_ids)` 把 draft 纳入调度模型。调度后，实际验证的 draft tokens 写入 `SchedulerOutput.scheduled_spec_decode_tokens`，并清空 Request 上的旧 spec tokens，避免重复验证。ModelRunner 收到后，把 draft tokens 追加到 InputBatch 同一 request row 的真实 tokens 后面，并构造 `SpecDecodeMetadata`，说明哪些 hidden state rows 需要计算 target logits，哪些 rows 用于 bonus logits。`sample_tokens()` 中如果存在 spec metadata，就调用 `RejectionSampler` 而不是普通 Sampler。Scheduler 回收时根据原 scheduled draft 和真实 sampled tokens 计算 accepted / rejected，回退 rejected 对应的 `num_computed_tokens`，append 真正输出，并通过 EngineCore post-step 回写下一轮 DraftTokenIds。核心是：drafter 猜，target 验，sampler 判，Scheduler 修账。

## 15. compilation_and_cuda_graph

`compilation_and_cuda_graph` 专题讲 vLLM 执行层的性能优化。`CompilationMode` 决定底层 forward callable 是 eager、stock torch.compile、Dynamo trace once 还是 vLLM compile；`CUDAGraphMode` 决定每轮是否以 FULL、PIECEWISE 或 NONE 方式 replay CUDA graph。二者是两层概念：CUDA graph 为 NONE 不代表一定是 eager，因为底层 runnable 仍可能是 compiled callable；FULL CUDA graph 通常覆盖的是 model forward，不是完整 Engine step，不包含 Scheduler、OutputProcessor、detokenize、通常也不包含 sampler。

核心矛盾是 serving batch 动态变化，而 CUDA graph replay 需要固定 shape、固定地址和稳定 kernel launch 序列。vLLM 通过 `GPUModelRunner._determine_batch_execution_and_padding()` 和 `CudagraphDispatcher.dispatch()` 判断本轮是否可 graph，把真实 batch 归一成 `BatchDescriptor`，必要时 padding 到 captured shape。padding 不能污染语义，所以 padding token 的 slot_mapping 要设为 `-1`，padding request 的 block table 要用 NULL block，且 padding 位置不能参与 logits 和输出。随后 `_build_attention_metadata()` 按 padded shape 构造可 capture 的 metadata，`ForwardContext` 把 runtime mode、BatchDescriptor、slot mapping 和 attention metadata 传给模型内部，`CUDAGraphWrapper` 决定 pass-through、capture 或 replay。不满足条件时安全 fallback 到 PIECEWISE 或 NONE，正确性优先，性能其次。

## 16. kv_cache_transfer

`kv_cache_transfer` 是最高阶的 KV cache 专题，解释本地 prefix cache、外部 KV cache / KVPool、KV Connector、异步 load/save、invalid blocks 和 deferred free 如何闭环。Scheduler 先通过 `KVCacheManager.get_computed_blocks()` 查询本地 prefix cache，再通过 Scheduler 侧 KV Connector 查询外部 KV 命中。外部命中不代表可以直接跳过一切，因为这些 KV 仍必须 load 到本地 GPU KV cache tensor 中，所以 Scheduler 还要调用 `KVCacheManager.allocate_slots()` 为 external computed tokens 分配本地 KV blocks，再通过 `connector.update_state_after_alloc()` 告诉 connector 外部 KV 要写到哪些 block ids。

调度结果通过 `SchedulerOutput.kv_connector_metadata` 下发给 Worker。Worker 侧 connector 在 forward 前绑定 metadata 并启动 load；在 attention layer entry 等待某层 KV load 完成，在 attention layer exit 保存该层 KV；forward 后返回 `KVConnectorOutput`，包括 `finished_recving`、`finished_sending`、`invalid_block_ids`、stats 和 worker_meta。Scheduler 回收时，如果收到 `finished_recving`，说明远端 KV 已 load 到本地 blocks，可以恢复 `WAITING_FOR_REMOTE_KVS` 请求；如果收到 `finished_sending`，说明请求结束后的 KV save 已完成，可以释放延迟保护 blocks；如果收到 `invalid_block_ids`，说明某些本地 block load 失败或不可信，需要把对应请求的 `num_computed_tokens` 回退到失败 block 前，并按策略选择 recompute 或 fail。核心边界是：Scheduler 管 KV block 账本和状态机，Worker 管真实 KV tensor 和传输执行。

## 总体串联背诵

如果要把 16 个专题串成一条完整主线，可以背下面这段：

```text
vLLM 先通过 config_and_model_loading 把用户启动参数变成 VllmConfig，并在 Worker 侧加载出真实模型；model_architectures 负责把 HF architecture 和 checkpoint 适配成统一 model class；engine 负责外层输入输出，engine_core 负责内部 schedule、execute、update 闭环；scheduler 维护请求队列、token budget、KV block 和状态账本；executor_worker_model_runner 把 SchedulerOutput 分发到 Worker 并由 ModelRunner 执行；attention 把 block table、slot mapping 和 metadata 转成 backend kernel；sampling_and_output 把 logits 变成 token、状态和用户输出；operators 是底层 kernel 执行面；parallelism 决定多 rank 如何切 tensor、layer、request、expert 和 context；quantization、LoRA、multimodal、spec_decode、compilation_and_cuda_graph、kv_cache_transfer 则分别作为横向能力接入配置、模型加载、调度、执行、attention、sampling 和资源管理的不同阶段。
```
