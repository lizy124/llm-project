# 大模型推理方向四个月技术路线图

本文用于 4 个月系统梳理大模型推理方向，目标不是泛泛学习 LLM，而是围绕「推理系统工程师 / vLLM / CUDA Kernel / 分布式推理 / KV Cache / 性能优化」建立可落地、可验证、可表达的知识体系。

结合当前 `llm-project` 已有材料，重点参考：

- `research_workspace/structure_analysis/vllm_entrypoints_api_layer`：API 层、OpenAI Server、请求入口。
- `research_workspace/structure_analysis/vllm_inference_engine_layer`：EngineCore、Scheduler、请求生命周期。
- `research_workspace/structure_analysis/vllm_scheduler_kv_cache`：调度、KV Cache、Prefix Cache、KV Connector。
- `research_workspace/structure_analysis/vllm_worker_executor_layer`：Executor、Worker、ModelRunner。
- `research_workspace/structure_analysis/vllm_model_executor_attention_layer`：模型执行、Attention、KV Tensor、Slot Mapping。
- `research_workspace/structure_analysis/vllm_native_acceleration_layer`：CUDA / C++ / CUTLASS / Marlin / Machete / MoE kernel。
- `research_workspace/structure_analysis/vllm_distributed_communication`：TP/PP/DP/EP、ProcessGroup、Collective、Runtime flow。
- `research_workspace/documents/pool`：KV Cache 池化、复用、存储、ChunkedTokenDatabase。
- `research_workspace/documents/distributed_DP`：PD 分离、Prefill/Decode 解耦、KV Transfer、Layerwise Push。

---

## 0. 总目标与方向定位

### 目标能力画像

本路线图面向偏系统工程的大模型推理方向，重点通常不是训练算法，而是：

1. 能不能讲清一个请求从 HTTP 入口到 GPU kernel 执行再到 token 输出的完整链路。
2. 能不能解释 vLLM / TensorRT-LLM / SGLang / llama.cpp 等推理框架的核心设计取舍。
3. 能不能围绕吞吐、延迟、显存、并发、长上下文、KV Cache、调度策略做性能分析。
4. 能不能读懂并改动真实推理框架代码。
5. 能不能理解 CUDA kernel、Attention 优化、通信并行和分布式部署。
6. 能不能把自己的项目讲成一个有技术深度、有性能指标、有工程决策的故事。

### 四个月最终产出

到第 4 个月结束，建议至少形成这些成果：

- 一张「大模型推理系统总架构图」。
- 一条「OpenAI API 请求到 CUDA Attention kernel」端到端链路讲解。
- 一份「vLLM Scheduler + KV Cache」深度讲稿。
- 一份「PagedAttention / FlashAttention / Continuous Batching」对比讲稿。
- 一份「Prefill/Decode 分离 + KV Transfer」专题讲稿。
- 一份「分布式推理并行策略」专题讲稿。
- 一份「CUDA / Attention Kernel 性能优化」专题讲稿。
- 一个可展示的项目：例如基于 vLLM 的 KV Cache 池化、PD 分离、调度策略改造、性能 profiling 或 kernel 分析。
- 80 到 120 道高频技术题的答案库。
- 5 到 8 个能展开 10 分钟以上的深度专题。

---

## 1. 技术点优先级总览

### S 级：必须掌握，核心主线

这些内容决定你是否像一个真正做过推理系统的人。

1. LLM 推理基本流程
   - Tokenizer、prompt、prefill、decode、sampling、streaming output。
   - Prefill 与 decode 的计算特征差异。
   - 自回归生成为什么无法像普通 batch inference 一样简单处理。
   - TTFT、TPOT、ITL、吞吐、QPS、并发、显存占用之间的关系。

2. Transformer 推理计算与显存结构
   - Attention 的 Q/K/V、RoPE、MHA、MQA、GQA。
   - FFN / MLP、SwiGLU、RMSNorm、LayerNorm。
   - KV Cache 的形状、生命周期、显存占用公式。
   - batch size、sequence length、num layers、num heads、head dim、dtype 对显存的影响。

3. vLLM 核心架构
   - API 层、AsyncLLM、EngineCore、Scheduler、Executor、Worker、GPUModelRunner、Attention backend、CUDA kernel。
   - V0 / V1 架构差异和 V1 主链路。
   - 请求生命周期：add request、schedule、execute model、process output、finish。
   - `llm-project/research_workspace/structure_analysis/vllm_inference_engine_layer` 是主参考。

4. Scheduler 与 Continuous Batching
   - waiting / running 队列。
   - token budget、max_num_batched_tokens、max_num_seqs。
   - prefill/decode 混合调度。
   - chunked prefill。
   - preemption、recompute、swap/offload。
   - speculative decoding 调度影响。
   - 公平性、吞吐、延迟之间的取舍。

5. KV Cache 与 PagedAttention
   - 为什么需要 KV Cache。
   - 为什么传统 contiguous KV cache 浪费显存。
   - PagedAttention 的 block 管理思想。
   - block table、slot mapping、physical block、logical block。
   - Prefix Caching 的命中逻辑和收益边界。
   - KV Cache eviction、reuse、offload、transfer。
   - `llm-project/research_workspace/structure_analysis/vllm_scheduler_kv_cache` 和 `research_workspace/documents/pool` 是主参考。

6. GPU 执行链路
   - Worker 初始化、显存 profiling、KV cache 初始化。
   - GPUModelRunner 如何准备 input ids、position ids、block table、attention metadata。
   - model forward 如何进入 Attention layer。
   - Attention backend 如何选择 FlashAttention / FlashInfer / Triton / XFormers / native kernel。
   - CUDA/C++ extension 的调用路径。

7. 性能指标与瓶颈分析
   - TTFT 主要受 prefill 影响。
   - TPOT / ITL 主要受 decode step、batching、memory bandwidth、kernel launch、communication 影响。
   - decode 往往 memory-bound，prefill 更偏 compute-bound。
   - 显存瓶颈、算力瓶颈、带宽瓶颈、通信瓶颈、CPU 调度瓶颈如何区分。
   - 如何做 profiling：Nsight Systems、Nsight Compute、torch profiler、vLLM metrics。

### A 级：强相关重点，决定技术深度

1. Attention 优化
   - FlashAttention 原理：tiling、online softmax、减少 HBM 读写。
   - FlashAttention v1/v2/v3 的大致改进方向。
   - PagedAttention 与 FlashAttention 的关系和差异。
   - Decode attention 与 prefill attention 的不同 kernel 形态。
   - Cascade attention、MLA、Sliding Window Attention、Prefix attention。

2. 分布式推理
   - Tensor Parallelism：列并行、行并行、AllReduce。
   - Pipeline Parallelism：microbatch、bubble、适合场景。
   - Data Parallelism：副本扩展、负载均衡。
   - Expert Parallelism：MoE 专用并行、All-to-All。
   - Context Parallelism / Sequence Parallelism 的基本思想。
   - NCCL collective：AllReduce、AllGather、ReduceScatter、Broadcast、All-to-All。
   - vLLM 中 parallel config、process group、worker topology。
   - `llm-project/research_workspace/structure_analysis/vllm_distributed_communication` 是主参考。

3. Prefill/Decode 分离
   - 为什么要做 PD 分离。
   - Prefill 和 Decode 对资源的不同需求。
   - KV Cache 从 Prefill 节点传到 Decode 节点的方式。
   - KV Transfer Connector、handshake、memory registration。
   - 分离后的调度、延迟释放、完成确认。
   - Layerwise Push / Pull 模式。
   - `llm-project/research_workspace/documents/distributed_DP` 是主参考。

4. 量化推理
   - FP16、BF16、FP8、INT8、INT4 的差异。
   - Weight-only quantization 与 weight-activation quantization。
   - GPTQ、AWQ、SmoothQuant、Marlin、Machete、bitsandbytes、CUTLASS GEMM。
   - KV Cache quantization 的收益与风险。
   - 量化对精度、吞吐、延迟、显存的影响。

5. Speculative Decoding
   - draft model / target model 流程。
   - verify 阶段如何保证分布正确。
   - acceptance rate 对收益的影响。
   - Medusa、EAGLE、ngram spec decode 的基本区别。
   - 对 scheduler、KV cache、batching 的影响。

6. Serving 系统工程
   - OpenAI-compatible API。
   - streaming response、request cancellation、timeout。
   - admission control、rate limit、priority scheduling。
   - autoscaling、模型热加载、LoRA serving、多租户隔离。
   - metrics、tracing、logging、health check。

### B 级：加分项，能显著增强竞争力

1. CUDA Kernel 与底层优化
   - CUDA thread/block/grid、warp、SM、occupancy。
   - memory hierarchy：register、shared memory、L1/L2、HBM。
   - coalesced access、bank conflict、warp divergence。
   - GEMM 基础、tiling、tensor core、wmma/mma。
   - CUTLASS 基本抽象。
   - Triton 编程模型。
   - vLLM csrc 中 attention、kv cache、quantization、MoE 算子的调用地图。

2. MoE 推理
   - Router、Top-k expert selection。
   - Expert Parallelism、token dispatch、All-to-All。
   - load balance、capacity factor、expert batching。
   - MoE inference 的主要瓶颈。
   - DeepSeek 类 MoE 架构对推理系统的挑战。

3. Long Context 推理
   - KV Cache 显存膨胀问题。
   - Sliding Window、Attention sink、Paged KV、KV compression。
   - Prefix cache 在 RAG / agent / multi-turn chat 中的价值。
   - 长上下文下 TTFT、decode latency、显存、调度的权衡。

4. 多模态推理
   - Vision encoder + LLM 的基本链路。
   - image token / video token 对 prefill 的压力。
   - encoder cache、cross attention cache。
   - 多模态 batch 的调度特殊性。

5. 其他框架对比
   - TensorRT-LLM：engine build、plugin、inflight batching、强性能优化。
   - SGLang：RadixAttention、前缀复用、结构化生成。
   - llama.cpp：CPU / edge inference、GGUF、KV cache、quantization。
   - TGI / LMDeploy / FasterTransformer 的定位。

### C 级：选学项，按岗位需要补充

1. 训练系统基础
   - ZeRO、FSDP、activation checkpointing、optimizer state。
   - 训练与推理在显存、通信、调度上的区别。

2. 模型结构演进
   - LLaMA、Qwen、Mistral、Mixtral、DeepSeek、GLM、Gemma。
   - RoPE scaling、YaRN、ALiBi。
   - MLA、MTP、RMSNorm、SwiGLU。

3. 工程部署生态
   - Kubernetes、GPU operator、MIG、NVIDIA DCGM。
   - Triton Inference Server、Ray Serve、KServe。
   - 容器镜像、驱动、CUDA runtime、NCCL 版本兼容。

---

## 2. 分层技术树

### 第一层：请求入口与服务层

核心问题：用户请求如何进入推理系统？

必须掌握：

- OpenAI API：chat completions、completions、embeddings。
- HTTP server 生命周期：启动、路由、请求校验、异步处理。
- streaming：SSE、chunked response、增量 token 输出。
- cancellation：用户断开连接后如何取消正在运行的 request。
- request id、prompt tokenization、sampling params。

需要能讲清：

- 一个 ChatCompletion 请求进入 vLLM 后经历哪些对象。
- API 层和 Engine 层边界是什么。
- 为什么推理服务通常要异步化。
- streaming 时 token 是如何逐步返回的。

参考资料：

- `research_workspace/structure_analysis/vllm_entrypoints_api_layer/README.md`
- `research_workspace/structure_analysis/vllm_entrypoints_api_layer/05_Chat与Completion请求链路.md`
- `research_workspace/structure_analysis/vllm_entrypoints_api_layer/07_API层与Engine边界.md`

### 第二层：EngineCore 与请求生命周期

核心问题：一个请求在推理引擎内部如何流转？

必须掌握：

- EngineCore 的职责。
- EngineCoreClient / AsyncLLM / OutputProcessor 的分工。
- request state：waiting、running、finished、aborted。
- step loop：每轮调度、执行、返回输出。
- V0 / V1 架构差异。

需要能讲清：

- 为什么推理引擎一般是一个持续运行的主循环。
- 每个 step 为什么可能同时处理多个请求的不同 token 数。
- 输出 token 如何从 GPU 执行结果转成用户可见文本。

参考资料：

- `research_workspace/structure_analysis/vllm_inference_engine_layer/01_推理引擎层总览.md`
- `research_workspace/structure_analysis/vllm_inference_engine_layer/02_请求生命周期_API到EngineCore.md`
- `research_workspace/structure_analysis/vllm_inference_engine_layer/03_EngineCore主循环与V0兼容.md`

### 第三层：Scheduler 与 Continuous Batching

核心问题：这一轮到底该算哪些请求、多少 token、如何分配资源？

必须掌握：

- continuous batching 与 static batching 的区别。
- waiting queue / running queue。
- prefill scheduling、decode scheduling。
- chunked prefill：长 prompt 拆分，降低 decode 饥饿。
- token budget / seq budget。
- max_num_batched_tokens、max_num_seqs。
- preemption：资源不足时暂停或重算。
- priority scheduling、fairness、latency-throughput tradeoff。

需要能讲清：

- vLLM 为什么吞吐高。
- chunked prefill 解决什么问题，又带来什么副作用。
- prefill-heavy 和 decode-heavy 场景如何调参。
- 如果 TTFT 很高或 TPOT 很高，scheduler 侧可能怎么排查。

参考资料：

- `research_workspace/structure_analysis/vllm_inference_engine_layer/04_Scheduler调度机制.md`
- `research_workspace/structure_analysis/vllm_scheduler_kv_cache/01_scheduler_overview.md`
- `research_workspace/structure_analysis/vllm_scheduler_kv_cache/03_request_lifecycle.md`

### 第四层：KV Cache、PagedAttention 与 Prefix Caching

核心问题：推理中最大的显存资产如何管理？

必须掌握：

- KV Cache 的作用：避免重复计算历史 token 的 K/V。
- KV Cache 显存公式：layers * tokens * kv heads * head dim * 2(K/V) * dtype bytes。
- MHA / MQA / GQA 对 KV cache 大小的影响。
- PagedAttention：逻辑 token block 到物理 cache block 的映射。
- block table、slot mapping、free block、ref count。
- prefix caching：hash、命中、共享 block、释放。
- KV Cache fragmentation 与复用。
- KV offload / transfer / pooling。

需要能讲清：

- PagedAttention 类比操作系统分页为什么合理。
- Prefix Cache 为什么对 system prompt / RAG / agent 有价值。
- KV cache 复用有哪些正确性风险。
- 长上下文场景下 KV cache 为什么是核心瓶颈。

参考资料：

- `research_workspace/structure_analysis/vllm_inference_engine_layer/05_KVCache_Block_PrefixCaching.md`
- `research_workspace/structure_analysis/vllm_scheduler_kv_cache/02_kv_cache_architecture.md`
- `research_workspace/structure_analysis/vllm_model_executor_attention_layer/06_KVCacheSpec与KV_Tensor_SlotMapping.md`
- `research_workspace/documents/pool/05_推理时存储KV_Cache.md`
- `research_workspace/documents/pool/06_从池子复用KV_Cache.md`
- `research_workspace/documents/pool/07_ChunkedTokenDatabase_键管理与地址计算.md`

### 第五层：Executor、Worker、ModelRunner

核心问题：调度结果如何变成一次真实 GPU forward？

必须掌握：

- Executor 抽象：uniproc、multiproc、ray。
- Worker 生命周期：init device、load model、init cache、execute model。
- GPUModelRunner：构造 batch、metadata、execute_model。
- model input：input ids、positions、intermediate tensors。
- attention metadata、block table、slot mapping。
- logits processor、sampler、output processor。

需要能讲清：

- Scheduler output 进入 Worker 后如何被转成模型输入。
- 为什么 ModelRunner 是调度层和模型层的关键边界。
- 多进程 worker 和 Ray worker 的差异。
- 初始化 KV cache 为什么需要 profiling。

参考资料：

- `research_workspace/structure_analysis/vllm_worker_executor_layer/README.md`
- `research_workspace/structure_analysis/vllm_worker_executor_layer/02_Executor抽象与后端选择.md`
- `research_workspace/structure_analysis/vllm_worker_executor_layer/07_GPU_CPU_XPU_Worker生命周期.md`
- `research_workspace/structure_analysis/vllm_worker_executor_layer/08_ModelRunner执行链路.md`
- `research_workspace/structure_analysis/vllm_worker_executor_layer/11_端到端调用链路.md`

### 第六层：Attention、模型执行与底层 Kernel

核心问题：模型 forward 中最关键的算子如何执行和优化？

必须掌握：

- Transformer block 推理链路。
- Attention layer 的 QKV projection、RoPE、attention、output projection。
- prefill attention 与 decode attention 差异。
- FlashAttention：减少 HBM 访问。
- PagedAttention：面向 KV cache block 管理。
- backend 选择：FlashAttention、FlashInfer、Triton、XFormers、native。
- CUDA extension 从 Python 到 C++/CUDA 的调用路径。

需要能讲清：

- FlashAttention 主要优化了什么。
- 为什么 decode attention 往往是 memory bandwidth bound。
- block table / slot mapping 如何影响 attention kernel 读取 KV。
- 一个 attention 调用从 Python module 到 CUDA kernel 的路径。

参考资料：

- `research_workspace/structure_analysis/vllm_model_executor_attention_layer/README.md`
- `research_workspace/structure_analysis/vllm_model_executor_attention_layer/04_Attention层核心实现.md`
- `research_workspace/structure_analysis/vllm_model_executor_attention_layer/05_AttentionBackend选择与MetadataBuilder.md`
- `research_workspace/structure_analysis/vllm_model_executor_attention_layer/08_CUDA_csrc_kernel调用链与调试地图.md`
- `research_workspace/structure_analysis/vllm_native_acceleration_layer/04_CUDA核心算子_Attention与KVCache.md`

### 第七层：分布式推理与通信

核心问题：单卡放不下或吞吐不够时，如何扩展到多 GPU / 多节点？

必须掌握：

- TP：切分权重矩阵，forward 中需要 collective。
- PP：切分 layer，需要 pipeline schedule。
- DP：复制模型，提升服务吞吐。
- EP：MoE expert 切分和 token dispatch。
- NCCL process group 与 collective。
- 通信和计算 overlap。
- 多节点部署时的 rank、world size、local rank、node rank。

需要能讲清：

- TP 中为什么需要 AllReduce。
- PP 为什么有 bubble，推理中是否总是适合 PP。
- MoE 为什么经常需要 All-to-All。
- 分布式推理中遇到通信瓶颈怎么定位。

参考资料：

- `research_workspace/structure_analysis/vllm_distributed_communication/README.md`
- `research_workspace/structure_analysis/vllm_distributed_communication/02_parallel_config_and_topology.md`
- `research_workspace/structure_analysis/vllm_distributed_communication/04_process_groups_and_collectives.md`
- `research_workspace/structure_analysis/vllm_distributed_communication/05_runtime_parallel_flows.md`
- `research_workspace/structure_analysis/vllm_worker_executor_layer/10_分布式并行与通信.md`

### 第八层：PD 分离与 KV Transfer

核心问题：如何把 prefill 和 decode 分给更适合的资源池？

必须掌握：

- Prefill compute-heavy，decode memory/latency-sensitive。
- PD 分离的动机：提升资源利用、隔离长 prompt 影响、独立扩缩容。
- KV Cache transfer：prefill 生成 KV，decode 拉取或接收 KV。
- handshake、metadata、block mapping、内存注册。
- push / pull / layerwise push。
- decode 端完成后如何通知 prefill 端释放资源。

需要能讲清：

- PD 分离和普通 DP serving 的区别。
- KV transfer 为什么比重新 prefill 更划算，但也有网络与一致性代价。
- layerwise push 能降低什么延迟。
- PD 分离下 scheduler 需要新增哪些约束。

参考资料：

- `research_workspace/documents/distributed_DP/README.md`
- `research_workspace/documents/distributed_DP/01_总览_PD分离架构.md`
- `research_workspace/documents/distributed_DP/05_Decode端拉取KV_Cache.md`
- `research_workspace/documents/distributed_DP/06_Prefill端延迟释放与完成确认.md`
- `research_workspace/documents/distributed_DP/08_Layerwise_Push模式.md`

### 第九层：量化、LoRA、MoE、多模态

核心问题：真实生产场景中模型能力和成本如何平衡？

重点掌握：

- 量化：INT8、INT4、FP8、AWQ、GPTQ、SmoothQuant。
- 量化 GEMM：Marlin、Machete、CUTLASS。
- LoRA serving：adapter loading、multi-LoRA batching、显存管理。
- MoE serving：router、expert dispatch、load balance、EP。
- 多模态：encoder、image token、encoder cache、prefill 压力。

需要能讲清：

- weight-only quantization 为什么常用于 LLM serving。
- FP8 相比 INT4/INT8 的工程取舍。
- 多 LoRA 合批为什么比单模型服务更复杂。
- MoE 推理为什么通信和负载均衡压力大。

参考资料：

- `research_workspace/structure_analysis/vllm_model_executor_attention_layer/07_量化_MoE_LoRA_多模态.md`
- `research_workspace/structure_analysis/vllm_native_acceleration_layer/05_量化_GEMM_CUTLASS_Marlin_Machete.md`
- `research_workspace/structure_analysis/vllm_native_acceleration_layer/06_MoE底层算子.md`
- `research_workspace/structure_analysis/vllm_worker_executor_layer/09_KVCache_LoRA_KVTransfer_Profiler.md`

### 第十层：性能分析、稳定性与工程交付

核心问题：线上推理系统出问题时，你能否定位并优化？

必须掌握：

- 指标：QPS、TPS、TTFT、TPOT、ITL、P50/P95/P99、GPU utilization、SM occupancy、HBM bandwidth、KV cache usage。
- profiling 工具：Nsight Systems、Nsight Compute、torch profiler、Prometheus、Grafana。
- 常见瓶颈：CPU tokenization、调度主循环、GPU kernel、显存不足、KV fragmentation、NCCL 通信、网络传输、采样。
- 稳定性：OOM、timeout、request cancellation、worker crash、model loading failure。
- benchmark 方法：固定模型、固定输入输出长度、控制并发、区分 warmup。

需要能讲清：

- 如何设计一个公平的 vLLM benchmark。
- TTFT 高、TPOT 高、吞吐低分别如何排查。
- 显存碎片或 KV cache 不足时会发生什么。
- 如何判断 bottleneck 是 compute、memory 还是 communication。

---

## 3. 四个月学习节奏

### 第 1 个月：打通主链路和核心概念

目标：能完整讲清「请求进入 vLLM 到输出 token」的主链路。

第 1 周：LLM 推理基础

- 学 prefill / decode / sampling / streaming。
- 推导 KV Cache 显存公式。
- 理解 TTFT、TPOT、吞吐、并发。
- 输出物：一页「LLM 推理指标与瓶颈」笔记。

第 2 周：vLLM API 到 EngineCore

- 阅读 `vllm_entrypoints_api_layer`。
- 阅读 `vllm_inference_engine_layer` 前 3 篇。
- 画请求生命周期图。
- 输出物：一份「OpenAI API 到 EngineCore」讲稿。

第 3 周：Scheduler

- 深读 Scheduler 调度机制。
- 理解 waiting/running、token budget、chunked prefill。
- 整理调度参数对性能的影响。
- 输出物：一份「Continuous Batching 和 Scheduler」讲稿。

第 4 周：KV Cache 和 PagedAttention

- 深读 KV cache 相关材料。
- 理解 block table、slot mapping、prefix caching。
- 结合 `pool` 资料理解 KV cache 复用。
- 输出物：一份「KV Cache / PagedAttention / Prefix Cache」讲稿。

月末验收：

- 能白板画出 vLLM V1 主链路。
- 能独立解释 vLLM 为什么吞吐高。
- 能回答 KV cache 显存怎么估算。
- 能回答 chunked prefill 解决什么问题。

### 第 2 个月：深入执行层、Attention 和 CUDA

目标：能讲清调度结果如何变成 GPU kernel，理解 attention 性能优化。

第 5 周：Executor / Worker / ModelRunner

- 阅读 worker_executor_layer。
- 梳理 Worker 初始化、KV cache 初始化、execute_model。
- 输出物：一份「Scheduler output 到 GPUModelRunner」链路图。

第 6 周：模型执行和 Attention backend

- 阅读 model_executor_attention_layer。
- 理解 Attention metadata、backend 选择、ForwardContext。
- 输出物：一份「Attention 从 Python 到 backend」讲稿。

第 7 周：FlashAttention / PagedAttention / Decode Attention

- 系统整理 FlashAttention 原理。
- 对比 PagedAttention 和 FlashAttention。
- 理解 prefill attention 与 decode attention。
- 输出物：一张「Attention 优化技术对比表」。

第 8 周：CUDA 与底层算子

- 阅读 native_acceleration_layer。
- 补 CUDA 基础：thread/block/warp、shared memory、tensor core、memory coalescing。
- 理解 vLLM csrc 调用地图。
- 输出物：一份「CUDA kernel 基础 + vLLM 调用链」笔记。

月末验收：

- 能讲清一次 forward 前 ModelRunner 准备了哪些 metadata。
- 能讲清 attention kernel 如何通过 block table 读 KV。
- 能解释 FlashAttention 为什么减少 HBM IO。
- 能说出 decode 为什么经常 memory-bound。

### 第 3 个月：分布式、PD 分离、量化与高级特性

目标：能处理大模型、多卡、多节点、长上下文和生产优化问题。

第 9 周：分布式并行

- 阅读 distributed_communication。
- 整理 TP/PP/DP/EP 的适用场景和通信模式。
- 输出物：一份「分布式推理并行策略」讲稿。

第 10 周：PD 分离和 KV Transfer

- 阅读 distributed_DP。
- 理解 Prefill/Decode 解耦、KV 拉取/推送、延迟释放。
- 输出物：一份「PD 分离架构与 KV Transfer」讲稿。

第 11 周：量化、LoRA、MoE

- 整理 AWQ/GPTQ/SmoothQuant/FP8/INT4。
- 理解 multi-LoRA serving。
- 理解 MoE 推理瓶颈。
- 输出物：一份「量化和 MoE 推理」技术笔记。

第 12 周：性能分析和 benchmark

- 设计 benchmark 表格。
- 学会区分 TTFT、TPOT、吞吐瓶颈。
- 整理 profiling 方法。
- 输出物：一份「推理性能排查手册」。

月末验收：

- 能讲清 TP 中 AllReduce 发生在哪里。
- 能讲清 PD 分离为什么适合长 prompt 或混合负载。
- 能讲清量化对显存、吞吐、精度的影响。
- 能给出一套线上性能问题排查流程。

### 第 4 个月：项目沉淀、技术表达和查漏补缺

目标：把知识变成可证明、可复盘、可迁移的能力。

第 13 周：项目主线整理

- 选择一个主项目重点包装：建议围绕 KV Cache 池化 / PD 分离 / vLLM 调度分析。
- 梳理背景、问题、方案、核心实现、性能收益、取舍、后续优化。
- 输出物：一份 10 分钟项目讲稿。

第 14 周：高频题答案库

- 整理 80 到 120 道题。
- 每题答案控制在 1 到 3 分钟。
- 输出物：技术问答库。

第 15 周：模拟讲解与白板表达

- 每天抽 3 个深度题白板讲解。
- 重点练链路题和 tradeoff 题。
- 输出物：录音或文字复盘。

第 16 周：查漏补缺和岗位定制

- 针对目标公司 JD 补短板。
- 如果偏框架：加强 vLLM / SGLang / TensorRT-LLM 对比。
- 如果偏 kernel：加强 CUDA / Triton / CUTLASS。
- 如果偏平台：加强 serving、K8s、监控、稳定性。
- 输出物：最终技术速查手册。

月末验收：

- 能讲 3 个 10 分钟深度专题。
- 能讲 1 个 15 分钟项目。
- 能回答核心题不依赖资料。
- 能针对 JD 解释自己为什么匹配。

---

## 4. 技术题分层清单

### S 级高频题

1. 大模型推理中 prefill 和 decode 有什么区别？
2. KV Cache 是什么？为什么能加速推理？
3. KV Cache 显存怎么估算？GQA/MQA 对它有什么影响？
4. vLLM 为什么比普通 HuggingFace Transformers serving 吞吐高？
5. PagedAttention 解决了什么问题？和操作系统分页有什么相似点？
6. Continuous batching 是什么？相比 static batching 有什么优势？
7. chunked prefill 解决什么问题？有什么代价？
8. TTFT 和 TPOT 分别受什么影响？如何优化？
9. 一个请求从 OpenAI API 到 GPU kernel 经过哪些模块？
10. Scheduler 如何决定每一轮执行哪些 token？
11. Prefix caching 的原理是什么？什么场景收益最大？
12. FlashAttention 为什么快？核心优化点是什么？
13. Decode 阶段为什么常常 memory-bound？
14. Tensor Parallelism 中为什么需要 AllReduce？
15. 如果线上 vLLM 服务 OOM，你如何排查？
16. 如果 P99 latency 升高，你如何定位？
17. 如果 GPU utilization 不高但延迟很高，可能是什么原因？
18. vLLM 中 Executor、Worker、ModelRunner 分别负责什么？
19. block table 和 slot mapping 是什么？
20. 如何设计一个推理 benchmark？

### A 级深度题

1. PagedAttention 和 FlashAttention 是同一类优化吗？区别是什么？
2. vLLM 的 Scheduler 如何在吞吐和延迟之间做权衡？
3. Prefix cache 如何保证复用正确性？hash key 应该包含什么？
4. KV cache offload / transfer 的收益和代价是什么？
5. Prefill/Decode 分离为什么有意义？适合什么负载？
6. layerwise KV push 相比整段 KV transfer 有什么优势？
7. speculative decoding 为什么不会改变最终采样分布？
8. acceptance rate 对 speculative decoding 的收益有什么影响？
9. FP8 / INT8 / INT4 量化分别适合什么场景？
10. AWQ 和 GPTQ 的核心差异是什么？
11. Multi-LoRA serving 为什么难以高效合批？
12. MoE 推理瓶颈在哪里？为什么 All-to-All 很关键？
13. TP、PP、DP、EP 如何组合？
14. NCCL AllReduce 慢可能是什么原因？
15. 长上下文推理中 KV cache 有哪些优化方向？
16. 为什么有时增大 batch 会降低单请求 latency 但提高吞吐？
17. GPU kernel launch overhead 对 decode 有什么影响？
18. 如何判断一个 attention kernel 是 compute-bound 还是 memory-bound？
19. CUDA shared memory 在 attention/GEMM 中怎么用？
20. CUTLASS / Triton / handwritten CUDA 的取舍是什么？

### B 级扩展题

1. TensorRT-LLM 和 vLLM 的设计重点有什么不同？
2. SGLang 的 RadixAttention 和 vLLM Prefix Caching 有什么异同？
3. llama.cpp 为什么适合本地和边缘场景？
4. RoPE scaling 会对长上下文推理带来什么影响？
5. MLA 会如何改变 KV cache 大小和 attention 实现？
6. Sliding Window Attention 如何减少长上下文成本？
7. 多模态 LLM serving 有哪些额外缓存和调度问题？
8. 如何实现 request cancellation？
9. 如何做推理服务的 admission control？
10. 如何做多租户模型服务隔离？

---

## 5. 必须能画的图

建议至少反复画熟这些图：

1. vLLM 端到端请求链路图
   - OpenAI API -> AsyncLLM -> EngineCoreClient -> EngineCore -> Scheduler -> Executor -> Worker -> GPUModelRunner -> Model -> Attention Backend -> CUDA Kernel。

2. Scheduler 状态机图
   - waiting queue、running queue、scheduled req、finished req、preempted req。

3. PagedAttention / KV Cache block 映射图
   - request logical block -> block table -> physical KV block -> slot mapping -> attention kernel。

4. Prefill / Decode 对比图
   - prefill 大矩阵并行计算；decode 单步增量生成，依赖历史 KV。

5. PD 分离架构图
   - client/router -> prefill worker -> KV transfer -> decode worker -> output。

6. Tensor Parallelism 通信图
   - column parallel linear、row parallel linear、AllReduce。

7. FlashAttention IO 图
   - Q/K/V 分块、shared memory、online softmax、减少 HBM 中间矩阵写回。

8. 性能排查决策树
   - TTFT 高 / TPOT 高 / 吞吐低 / OOM / GPU 利用率低。

---

## 6. 项目包装建议

### 推荐主项目方向 1：KV Cache 池化与复用

适合结合当前 `pool` 材料。

讲法：

- 背景：大模型推理中长 prompt 和重复 prompt 导致 prefill 计算和 KV cache 显存成本高。
- 问题：如何跨请求复用 KV cache，并保证正确性、生命周期和地址映射。
- 方案：设计 connector、token chunk key、KV block 存储、prefix 匹配、引用计数、释放策略。
- 难点：hash key 正确性、block 对齐、chunk 边界、并发读写、显存生命周期、scheduler 协同。
- 指标：TTFT 降低、prefill tokens 减少、cache hit rate、显存占用、吞吐变化。
- 风险：错误复用导致输出污染；缓存元数据过大；命中率不足导致收益有限。

### 推荐主项目方向 2：Prefill/Decode 分离与 KV Transfer

适合结合当前 `distributed_DP` 材料。

讲法：

- 背景：prefill 和 decode 对计算资源需求不同，混部会导致长 prompt 阻塞 decode。
- 问题：如何将 prefill 产生的 KV cache 高效交给 decode 节点。
- 方案：prefill worker 计算 KV，decode worker 通过 connector 拉取或接收 KV，完成后通知释放。
- 难点：KV 地址映射、跨节点通信、延迟释放、失败恢复、调度一致性。
- 指标：TTFT、TPOT、GPU 利用率、网络带宽、decode batch 稳定性。
- 风险：KV transfer 网络开销过高；调度复杂度上升；跨节点错误恢复难。

### 推荐主项目方向 3：vLLM Scheduler 性能分析和改造

讲法：

- 背景：不同 workload 下 scheduler 参数对吞吐和延迟影响明显。
- 问题：长 prompt、多短请求、混合请求下如何调度更稳定。
- 方案：分析 token budget、chunked prefill、priority、preemption，提出策略或调参方案。
- 难点：吞吐和延迟 tradeoff；不同请求长度分布下最优策略不同。
- 指标：P50/P95/P99 TTFT、TPOT、throughput、GPU utilization。

---

## 7. 每周固定训练方法

### 阅读

每周至少精读 3 到 5 篇已有文档，不追求数量，追求能复述链路。

推荐方式：

1. 先读 README 或总览。
2. 再读端到端链路。
3. 最后读关键模块细节。
4. 每读完一篇，用 5 句话写总结：职责、输入、输出、关键数据结构、技术价值。

### 白板

每周至少画 2 次图：

- 一次画端到端链路。
- 一次画核心机制，例如 KV block 映射、TP 通信、PD 分离。

### 题库

每周沉淀 15 到 20 道题：

- 5 道基础题。
- 5 道机制题。
- 5 道 tradeoff 题。
- 2 到 5 道项目追问题。

### 代码

每周至少跟一条代码链路：

- API 到 Engine。
- Scheduler 到 KV manager。
- Worker 到 ModelRunner。
- Attention 到 CUDA extension。
- Distributed config 到 process group。

### 表达

每周录一次 10 分钟讲解，重点检查：

- 有没有只背概念，没有链路。
- 有没有只讲优点，没有 tradeoff。
- 有没有指标和验证方法。
- 有没有结合项目经验。

---

## 8. 技术点掌握标准

### 只知道名词：不合格

表现：

- 能说 PagedAttention、FlashAttention、PD 分离，但说不出输入输出和解决的问题。
- 遇到追问只能重复定义。

### 能讲机制：基本合格

表现：

- 能画图说明结构。
- 能说清为什么需要这个机制。
- 能说出常见参数和限制。

### 能讲 tradeoff：实践可用

表现：

- 能说收益、代价、适用场景。
- 能结合 TTFT、TPOT、吞吐、显存讲影响。
- 能解释为什么某个优化在某些场景无效。

### 能讲实现：强竞争力

表现：

- 能说出 vLLM 中大致类和模块。
- 能说出数据结构如何传递。
- 能说出关键代码路径。

### 能讲排查和改造：高竞争力

表现：

- 给出 profiling 方法。
- 给出 benchmark 设计。
- 给出优化方案和验证指标。
- 能基于项目说出真实问题和解决过程。

---

## 9. 查漏补缺清单

### 数学与模型基础

- Attention softmax 公式。
- RoPE 基本原理。
- LayerNorm / RMSNorm。
- SwiGLU。
- Sampling：temperature、top-k、top-p、repetition penalty。
- Beam search 和 sampling 的区别。

### 系统基础

- GPU memory hierarchy。
- CUDA execution model。
- NCCL collective。
- Linux process / multiprocessing / shared memory 基础。
- 网络带宽和延迟概念。

### 工程基础

- Python async / multiprocessing。
- PyTorch module / custom op / extension。
- CMake / CUDA extension 基本构建。
- Prometheus metrics。
- Docker / CUDA runtime / driver 兼容。

### 技术表达短板

- 概念题要压缩到 1 分钟。
- 机制题要画图讲 3 到 5 分钟。
- 项目题要能被追问 15 分钟。
- 每个优化都要能讲收益、代价、指标、失败场景。

---

## 10. 推荐阅读顺序

### vLLM 主线

1. `research_workspace/structure_analysis/vllm_entrypoints_api_layer/README.md`
2. `research_workspace/structure_analysis/vllm_inference_engine_layer/README.md`
3. `research_workspace/structure_analysis/vllm_inference_engine_layer/02_请求生命周期_API到EngineCore.md`
4. `research_workspace/structure_analysis/vllm_inference_engine_layer/04_Scheduler调度机制.md`
5. `research_workspace/structure_analysis/vllm_inference_engine_layer/05_KVCache_Block_PrefixCaching.md`
6. `research_workspace/structure_analysis/vllm_worker_executor_layer/08_ModelRunner执行链路.md`
7. `research_workspace/structure_analysis/vllm_model_executor_attention_layer/04_Attention层核心实现.md`
8. `research_workspace/structure_analysis/vllm_model_executor_attention_layer/08_CUDA_csrc_kernel调用链与调试地图.md`
9. `research_workspace/structure_analysis/vllm_native_acceleration_layer/04_CUDA核心算子_Attention与KVCache.md`
10. `research_workspace/structure_analysis/vllm_distributed_communication/README.md`

### KV Pool / PD 分离主线

1. `research_workspace/documents/pool/01_总览_池化系统架构.md`
2. `research_workspace/documents/pool/05_推理时存储KV_Cache.md`
3. `research_workspace/documents/pool/06_从池子复用KV_Cache.md`
4. `research_workspace/documents/pool/07_ChunkedTokenDatabase_键管理与地址计算.md`
5. `research_workspace/documents/pool/08_全链路串联总结.md`
6. `research_workspace/documents/distributed_DP/01_总览_PD分离架构.md`
7. `research_workspace/documents/distributed_DP/05_Decode端拉取KV_Cache.md`
8. `research_workspace/documents/distributed_DP/06_Prefill端延迟释放与完成确认.md`
9. `research_workspace/documents/distributed_DP/08_Layerwise_Push模式.md`
10. `research_workspace/documents/distributed_DP/09_全链路串联总结.md`

---

## 11. 最终技术速查版

如果只剩最后两周，优先掌握这些主线：

1. vLLM 请求主链路。
2. Scheduler / Continuous Batching。
3. KV Cache / PagedAttention / Prefix Cache。
4. Prefill vs Decode。
5. FlashAttention vs PagedAttention。
6. Worker / ModelRunner / Attention backend。
7. TP/PP/DP/EP 和 NCCL collective。
8. PD 分离和 KV Transfer。
9. TTFT / TPOT / throughput 性能排查。
10. 一个自己的深度项目。

最终判断标准：

- 任何题都尽量从「问题背景 -> 机制 -> tradeoff -> 实现位置 -> 指标验证」五步回答。
- 任何系统设计题都尽量落到「调度、KV cache、GPU kernel、通信、监控」五个维度。
- 任何项目题都尽量落到「为什么做、怎么做、难点、指标、后续」五个维度。
