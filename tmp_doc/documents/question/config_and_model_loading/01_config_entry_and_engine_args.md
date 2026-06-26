# 01. 用户参数如何进入配置系统？

源码位置：

- `code/vllm/vllm/engine/arg_utils.py`
- `code/vllm/vllm/entrypoints/llm.py`
- `code/vllm/vllm/entrypoints/cli/`
- `code/vllm/vllm/entrypoints/openai/`
- `code/vllm/vllm/v1/engine/llm_engine.py`
- `code/vllm/vllm/v1/engine/async_llm.py`

本问题关注：CLI、Python `LLM`、OpenAI API server 等入口中的用户参数，如何进入 `EngineArgs` / `AsyncEngineArgs`，再转换成内部 `VllmConfig`。

---

## 1. 一句话回答占位

占位：后续补充 `EngineArgs` 是用户参数到内部配置的第一层聚合对象。

```text
CLI / Python API / API server 参数
  → EngineArgs / AsyncEngineArgs
  → create_engine_config()
  → VllmConfig
```

---

## 2. 入口类型占位

```text
Offline LLM：
  vllm.entrypoints.llm.LLM

CLI serve：
  vllm.entrypoints.cli.main
  vllm.entrypoints.cli.serve

OpenAI API server：
  vllm.entrypoints.openai

Engine 层：
  LLMEngine
  AsyncLLM
```

---

## 3. EngineArgs 职责占位

后续补充：

```text
- 收拢 model / tokenizer / dtype / trust_remote_code 等模型参数；
- 收拢 tensor_parallel_size / pipeline_parallel_size / distributed_executor_backend；
- 收拢 gpu_memory_utilization / block_size / kv_cache_dtype；
- 收拢 scheduler 相关参数；
- 收拢 LoRA / multimodal / speculative / compilation / observability / kv_transfer；
- 提供 create_engine_config() 生成 VllmConfig。
```

---

## 4. 后续待补源码证据

占位：补充 `EngineArgs` 定义、CLI parser 构造、`LLM.__init__()` 和 server 入口如何创建 engine args。
