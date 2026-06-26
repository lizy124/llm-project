# 02. VllmConfig 如何聚合全系统配置？

源码位置：

- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/config/cache.py`
- `code/vllm/vllm/config/parallel.py`
- `code/vllm/vllm/config/scheduler.py`
- `code/vllm/vllm/config/device.py`
- `code/vllm/vllm/config/load.py`
- `code/vllm/vllm/config/compilation.py`

本问题关注：`VllmConfig` 如何作为全局配置总线，把用户参数拆分成各个子配置，并被 EngineCore、Scheduler、Executor、Worker、ModelRunner、Attention、Sampler 等模块共同消费。

---

## 1. 一句话回答占位

占位：后续补充 `VllmConfig` 是贯穿 vLLM 运行时的配置聚合对象。

```text
EngineArgs
  → VllmConfig
      → ModelConfig
      → CacheConfig
      → ParallelConfig
      → SchedulerConfig
      → DeviceConfig
      → LoadConfig
      → LoRAConfig
      → MultiModalConfig
      → SpeculativeConfig
      → CompilationConfig
      → StructuredOutputsConfig
      → KVTransferConfig / ECTransferConfig
      → QuantizationConfig
```

---

## 2. 子配置影响范围占位

```text
ModelConfig：
  影响模型类、dtype、max_model_len、任务类型、tokenizer、HF config。

CacheConfig：
  影响 KV cache block size、KV dtype、prefix caching、显存使用。

ParallelConfig：
  影响 executor backend、TP / PP / DP / EP、worker class。

SchedulerConfig：
  影响 token budget、batch size、chunked prefill、async scheduling。

LoadConfig：
  影响 model_loader 选择和权重格式。

CompilationConfig：
  影响 torch.compile、CUDA graph、capture sizes。
```

---

## 3. 校验与派生字段占位

后续补充：

```text
- 子配置之间如何互相校验；
- max_model_len 如何被推导；
- dtype / quantization 如何联动；
- parallel config 如何影响 worker / executor；
- scheduler config 如何受 model / cache 限制；
- compilation config 如何受 platform / attention backend 限制。
```

---

## 4. 后续待补源码证据

占位：补充 `VllmConfig` dataclass 字段、`__post_init__`、verify 方法、子配置构造入口。
