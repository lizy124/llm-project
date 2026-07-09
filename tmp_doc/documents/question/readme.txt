# vLLM 整体框架概览

vLLM 的整体框架可以按“大的功能层”和“核心运行链路”两种方式理解。

## 一、按功能层划分

### 1. `vllm/`：核心 Python 框架

vLLM 的主体代码都在这里，包括推理引擎、调度器、配置、入口 API、模型执行逻辑等。

关键子模块：

- `vllm/entrypoints/`：OpenAI API、CLI、服务入口
- `vllm/engine/`：推理引擎
- `vllm/v1/`：新版 V1 架构，包括 engine、scheduler、executor、worker、KV cache、spec decode、structured output 等
- `vllm/model_executor/`：模型加载与执行
- `vllm/config/`：配置系统
- `vllm/distributed/`：分布式执行、KV transfer / EC transfer、通信状态管理
- `vllm/attention/`：attention 后端抽象
- `vllm/compilation/`：编译优化、CUDA graph、torch.compile 相关

### 2. `csrc/`：C++ / CUDA / CPU 原生算子层

- vLLM 的底层高性能实现。
- 包括 attention kernel、cache 操作、custom all-reduce、CPU kernel、CUTLASS 扩展等。
- 主要服务于推理性能。

### 3. `rust/`：Rust 辅助模块

- 主要负责 tokenizer、chat template、tool calling、structured output、gRPC proto 等周边能力。
- 属于服务协议、模板解析、结构化输出相关的高性能辅助层。

### 4. `docs/`：文档与设计说明

- 用户指南
- API 文档
- 开发文档
- 功能设计说明
- 架构图

### 5. `tests/`：测试体系

覆盖正确性、CUDA、编译、配置、benchmark、entrypoints 等多个维度。

### 6. `benchmarks/`：性能基准

包含各类吞吐、延迟、serving、offline inference benchmark。

### 7. `examples/`：使用示例

包含离线推理、OpenAI API serving、多模态、LoRA、分布式等示例。

### 8. 工程化与构建模块

- `setup.py`
- `pyproject.toml`
- `CMakeLists.txt`
- `scripts/`
- `tools/`
- `.github/`

## 二、按核心运行链路划分

```text
入口层
  vllm/entrypoints/
    ↓
引擎层
  vllm/engine/ 或 vllm/v1/engine/
    ↓
调度层
  scheduler / request / KVCacheManager / encoder cache
    ↓
执行层
  executor / worker / model_runner
    ↓
模型层
  model_executor / attention / distributed / spec_decode
    ↓
底层算子层
  csrc/ + CUDA/CPU kernels
```

## 三、核心模块总结

当前已经整理出的主干分析可以分为：

1. 服务入口与 API 层
2. 推理引擎层
3. 调度与 KV Cache 管理层
4. Worker / Executor 执行层
5. 模型执行与 Attention 层
6. 分布式通信层
7. C++ / CUDA / Rust 底层加速层

当前 `question/` 目录还进一步拆出了 Engine、EngineCore、Scheduler、Executor / Worker / ModelRunner、KV cache transfer、Attention、Parallelism、Sampling、Spec Decode、Quantization、配置与模型加载、模型结构、多模态、LoRA / adapters、编译与 CUDA graph、算子层等更细专题。

在当前专题体系基础上，可以把全仓库视角继续归纳为以下补充专题：

8. 配置与模型加载层：`config_and_model_loading`
   - 覆盖 `EngineArgs`、`VllmConfig`、`ModelConfig`、`CacheConfig`、`ParallelConfig`、`SchedulerConfig`、`LoadConfig`。
   - 覆盖模型注册、模型加载、权重格式、量化权重、dummy weights、weight transfer。
   - 重点回答“用户参数如何影响引擎、worker、KV cache、模型加载和分布式行为”。

9. 采样与输出层：`sampling_and_output`
   - 覆盖 `SamplingParams`、logits processor、sampler、logprobs、stop condition。
   - 覆盖 structured output / grammar bitmask。
   - 覆盖 `ModelRunnerOutput -> EngineCoreOutputs -> RequestOutput -> OpenAI response`。
   - 重点回答“模型 logits 如何变成用户最终看到的 streaming / non-streaming 输出”。

10. 多模态层：`multimodal`
    - 覆盖 multimodal registry、mm processor、image/video/audio 输入处理。
    - 覆盖 mm features、mm embeddings、encoder cache、receiver cache。
    - 覆盖多模态请求如何进入 scheduler、worker、model runner。
    - 重点回答“多模态输入如何被预处理、缓存、调度并送入模型”。

11. 模型结构层：`model_architectures`
    - 覆盖模型注册、架构解析、模型类构造、forward 接口、embedding / LM head、MoE、多模态模型和权重映射。
    - 重点回答“一个模型架构如何被 vLLM 识别、实例化、执行并接入加载、量化、LoRA、并行等机制”。

12. LoRA 与 adapters 层：`lora_and_adapters`
    - 覆盖 LoRARequest、LoRA manager/cache、层注入、batch mixed LoRA、权重映射、量化和并行交互。
    - 重点回答“一个请求如何携带 LoRA，并在 worker/model runner/layer 侧动态生效”。

13. 编译与 CUDA graph 层：`compilation_and_cuda_graph`
    - 覆盖 torch.compile、CUDA graph capture/replay、attention metadata、sampler/output 与并行交互。
    - 重点回答“vLLM 如何在保持动态调度的同时利用编译和 CUDA graph 降低开销”。

14. 算子层：`operators`
    - 覆盖 Python custom op 封装、C++/CUDA/CPU kernel、attention 和 cache 相关底层算子。
    - 重点回答“高层 model runner 和 attention backend 最终如何落到底层算子实现”。
