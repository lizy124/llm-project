# 01 模型执行与 Attention 总览

## 1. 本层在 vLLM 推理链路中的位置

vLLM 的推理链路可以粗略分为：

```text
API / AsyncLLM
  ↓
EngineCore
  ↓
Scheduler
  ↓
Executor / Worker
  ↓
GPUModelRunner
  ↓
model_executor 模型执行层
  ↓
Attention / MLP / MoE / Norm / Sampler
  ↓
AttentionBackend / custom ops
  ↓
csrc CUDA/C++/CPU kernels
```

本目录关注的是从 `GPUModelRunner` 往下：模型如何被加载、如何 forward、Attention 如何拿到 KV cache 与 metadata、backend 如何选择、底层 kernel 如何被调用。

## 2. 模型执行层的核心职责

`vllm/model_executor` 负责：

1. 模型注册与 architecture 映射；
2. 模型实例化；
3. 权重文件发现、下载与加载；
4. 具体模型结构定义；
5. Attention、MLP、MoE、Norm、Rotary Embedding 等层实现；
6. 量化、LoRA、多模态模型适配；
7. 给运行时提供 KV cache spec；
8. 在 forward 中消费 `ForwardContext` 里的 attention metadata 和 slot mapping。

主要路径：

```text
vllm/model_executor/
  model_loader/
  models/
  layers/
    attention/
    quantization/
    fused_moe/
    linear.py
    layernorm.py
    rotary_embedding.py
```

## 3. Attention 层的核心职责

Attention 层不是普通 PyTorch attention。它承担运行时桥接职责：

1. 初始化时选择 Attention backend；
2. 初始化 backend-specific implementation；
3. 注册自身到 `static_forward_context`；
4. 声明当前层需要的 KV cache spec；
5. forward 时通过 `ForwardContext` 获取：
   - 当前层 KV cache tensor；
   - attention metadata；
   - slot mapping；
   - runtime config；
6. 更新 KV cache；
7. 调 backend/csrc 执行 paged attention；
8. 处理 KV cache quant scales。

关键文件：

```text
vllm/model_executor/layers/attention/attention.py
vllm/model_executor/layers/attention/mla_attention.py
vllm/model_executor/layers/attention/cross_attention.py
vllm/model_executor/layers/attention/encoder_only_attention.py
vllm/model_executor/layers/attention/prefill_prefix_lm_attention.py
```

## 4. v1/attention 的核心职责

`vllm/v1/attention` 是 V1 attention 抽象与 backend 选择层：

```text
vllm/v1/attention/
  backend.py
  selector.py
  backends/
  ops/
```

它负责：

- 定义 `AttentionBackend` 抽象；
- 定义 `AttentionImpl` 抽象；
- 定义 `AttentionMetadataBuilder`；
- 定义公共 metadata：`CommonAttentionMetadata`；
- 根据平台、dtype、head size、KV cache dtype、block size、attention type 选择 backend；
- 为具体 backend 构造 metadata；
- 封装 paged attention、FlashAttention、FlashInfer、Triton、ROCm、CPU、MLA 等实现。

## 5. 模型执行与 Attention 的端到端链路

```text
1. Worker.load_model()
2. GPUModelRunner.load_model()
3. model_loader 选择 loader 并构建模型
4. 模型类初始化具体 layers
5. Attention layer 初始化：
   - 选择 backend
   - 创建 impl
   - 注册到 static_forward_context
   - 准备 quant scales
6. EngineCore 初始化 KV cache：
   - 收集每层 KVCacheSpec
   - profile 可用显存
   - 分配 KV cache tensor
7. Scheduler 每步产生 SchedulerOutput
8. GPUModelRunner.execute_model()：
   - 更新 batch state
   - 准备 input_ids / positions
   - 生成 slot mapping
   - 构造 attention metadata
   - set_forward_context
9. 具体模型 forward
10. Attention.forward()
11. unified_kv_cache_update / unified_attention_with_output
12. backend impl / custom op / csrc kernel
13. GPUModelRunner compute logits / sample
```

## 6. 四个最重要的桥梁对象

### 6.1 `ForwardContext`

路径：`vllm/forward_context.py`

它是运行时上下文，连接 GPUModelRunner 和模型层。模型层 Attention 不直接接收所有 metadata，而是从 ForwardContext 中取。

### 6.2 `KVCacheSpec`

路径：`code/vllm/vllm/v1/kv_cache_interface.py:96`

它是模型层向运行时声明“我需要什么 KV cache”的规格对象。

### 6.3 `CommonAttentionMetadata`

路径：`code/vllm/vllm/v1/attention/backend.py:362`

它是 batch 级 attention metadata 的公共格式，由 GPUModelRunner 构造，再交给 backend-specific builder 转换。

### 6.4 `slot_mapping`

由 GPUModelRunner 构造，用来告诉 Attention：当前 token 的 key/value 应该写入 KV cache 的哪个 slot。

## 7. 模型执行层与推理引擎层的关系

```text
Scheduler 决定“算什么”
  - 哪些 request
  - 每个 request 几个 token
  - 哪些 block id

GPUModelRunner 决定“怎么喂给模型”
  - input_ids
  - positions
  - slot mapping
  - attention metadata
  - cudagraph padding

model_executor 决定“模型怎么算”
  - embedding
  - attention
  - MLP/MoE
  - norm
  - logits

AttentionBackend/csrc 决定“attention 怎么高性能执行”
  - paged attention
  - flash attention
  - cache update
  - quantized cache
```

## 8. 阅读本层时最容易混淆的点

### 8.1 `Attention` 与 `AttentionBackend` 不是一回事

- `Attention` 是模型里的 `nn.Module`；
- `AttentionBackend` 是 backend 能力与 layout 约定；
- `AttentionImpl` 是具体实现对象；
- `AttentionMetadataBuilder` 是 metadata 构造器。

### 8.2 KV cache 的“逻辑分配”和“物理 tensor”不在一个地方

- Scheduler/KVCacheManager 管逻辑 block；
- Worker/GPUModelRunner 分配物理 tensor；
- slot mapping 连接逻辑 token 和物理 tensor slot；
- Attention kernel 根据 metadata/block table 访问 tensor。

### 8.3 模型 forward 期间很多信息来自全局 forward context

Attention、MoE、LoRA 都可能通过 `ForwardContext.no_compile_layers[layer_name]` 找到 layer 实例和运行时状态，而不是普通参数传递。

## 9. 一句话总结

vLLM 的模型执行与 Attention 层，本质是一套“模型层声明能力 + runner 构造运行时上下文 + backend 选择高性能内核 + KV cache 分页寻址”的系统，而不只是普通神经网络层的 forward。
