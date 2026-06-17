# vLLM 整体框架概览

vLLM 的整体框架可以按“大的功能层”和“核心运行链路”两种方式理解。

## 一、按功能层划分

### 1. `vllm/`：核心 Python 框架

vLLM 的主体代码都在这里，包括推理引擎、调度器、配置、入口 API、模型执行逻辑等。

关键子模块：

- `vllm/entrypoints/`：OpenAI API、CLI、服务入口
- `vllm/engine/`：推理引擎
- `vllm/v1/`：新版 V1 架构，包括 scheduler、worker、KV cache、executor 等
- `vllm/model_executor/`：模型加载与执行
- `vllm/config/`：配置系统
- `vllm/distributed/`：分布式执行
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
  scheduler / request / sequence / block manager
    ↓
执行层
  executor / worker / model_runner
    ↓
模型层
  model_executor / attention / distributed
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

在此基础上，为了补齐全仓库视角，还可以继续补充 4 个专题：

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

11. 可观测性、服务运维与测试调试层：`observability_tests_debugging`
    - 覆盖 metrics、logging、tracing、profiler、health check、server lifecycle、failure handling。
    - 覆盖 sleep/wake、reset prefix cache、utility RPC 等运维接口。
    - 覆盖 tests 目录结构、关键测试、benchmark/profiling 脚本和修改不同模块后的验证路径。
    - 重点回答“线上服务如何观测、排障，以及修改代码后如何验证”。
