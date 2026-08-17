# 06 离线 LLM API

## 1. 离线 API 的入口

用户通常这样使用 vLLM 离线推理：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="...")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=32))
```

这里的 `LLM` 来自：

```text
vllm.entrypoints.llm.LLM
```

包级别导出在 `vllm/__init__.py` 中：

```python
"LLM": ".entrypoints.llm:LLM"
```

源码位置：`code/vllm/vllm/__init__.py:16-22`。

## 2. LLM 类定位

`LLM` 类定义：

```python
class LLM(BeamSearchOfflineMixin, PoolingOfflineMixin, OfflineInferenceMixin)
```

源码位置：`code/vllm/vllm/entrypoints/llm.py:66`。

源码注释说明它包含：

- tokenizer
- language model
- GPU memory for intermediate states / KV cache
- batching mechanism
- efficient memory management

源码位置：`code/vllm/vllm/entrypoints/llm.py:66-74`。

同时它明确说明：

> 这个类用于 offline inference；online serving 应使用 AsyncLLMEngine。

源码位置：`code/vllm/vllm/entrypoints/llm.py:171-174`。

在当前代码中，它实际接入 V1 `LLMEngine`：

```python
from vllm.v1.engine.llm_engine import LLMEngine
```

源码位置：`code/vllm/vllm/entrypoints/llm.py:54-55`。

## 3. 构造函数的职责

`LLM.__init__()` 参数非常多，主要覆盖：

- model/tokenizer
- tokenizer mode
- trust remote code
- tensor parallel size
- dtype/quantization
- revision/tokenizer_revision
- chat template
- seed
- GPU memory utilization
- KV cache memory bytes
- CPU offload
- eager/CUDA graph
- HF token/overrides
- multimodal processor kwargs
- pooling config
- structured outputs config
- profiler config
- attention config
- compilation config
- speculative decoding alias 参数
- 其他 EngineArgs 参数

构造函数源码起点：`code/vllm/vllm/entrypoints/llm.py:176-221`。

## 4. 构造函数内部流程

### 4.1 默认关闭 stats log

如果调用者没有传 `disable_log_stats`，离线 LLM 默认设置为 True：

源码位置：`code/vllm/vllm/entrypoints/llm.py:235-237`。

### 4.2 处理 worker_cls 序列化

如果传入 Python class 形式的 `worker_cls`，会用 `cloudpickle.dumps()` 序列化，避免 pickling 问题：

源码位置：`code/vllm/vllm/entrypoints/llm.py:238-244`。

### 4.3 处理 KV transfer config

如果 `kv_transfer_config` 是 dict，会转成 `KVTransferConfig` 对象：

源码位置：`code/vllm/vllm/entrypoints/llm.py:245-263`。

### 4.4 配置对象归一化

`_make_config()` 把 dict/None/实例统一转成 config instance。

涉及：

- `CompilationConfig`
- `StructuredOutputsConfig`
- `ProfilerConfig`
- `AttentionConfig`

源码位置：`code/vllm/vllm/entrypoints/llm.py:267-288`。

### 4.5 数据并行限制

离线单进程 `LLM(data_parallel_size > 1)` 会被限制，避免 hang。源码提示应使用 explicit multi-process data parallel example。

源码位置：`code/vllm/vllm/entrypoints/llm.py:290-303`。

### 4.6 构造 EngineArgs

`LLM.__init__()` 的大量参数最终被汇总为：

```python
engine_args = EngineArgs(...)
```

源码位置：`code/vllm/vllm/entrypoints/llm.py:305-345`。

### 4.7 创建 LLMEngine

核心调用：

```python
self.llm_engine = LLMEngine.from_engine_args(
    engine_args=engine_args,
    usage_context=UsageContext.LLM_CLASS,
)
```

源码位置：`code/vllm/vllm/entrypoints/llm.py:349-351`。

也就是说，离线 `LLM` 是 `LLMEngine` 的高级封装。

## 5. LLM 初始化后的重要字段

构造完成后，`LLM` 会保存：

- `self.model_config`
- `self.engine_class`
- `self.request_counter`
- `self.default_sampling_params`
- `self.supported_tasks`
- `self.runner_type`
- `self.renderer`
- `self.chat_template`
- `self.input_processor`

源码位置：`code/vllm/vllm/entrypoints/llm.py:352-365`。

其中：

- `renderer` 负责 chat/multimodal 输入渲染。
- `input_processor` 负责把外部输入处理成 engine 请求。
- `supported_tasks` 决定是否能 generate/pool/embed/classify。

## 6. generate API

### 6.1 方法签名

```python
def generate(
    self,
    prompts,
    sampling_params=None,
    *,
    use_tqdm=True,
    lora_request=None,
    priority=None,
    tokenization_kwargs=None,
    mm_processor_kwargs=None,
) -> list[RequestOutput]
```

源码位置：`code/vllm/vllm/entrypoints/llm.py:422-432`。

### 6.2 逻辑

`generate()`：

1. 检查 `runner_type == "generate"`。
2. 如果未传 sampling params，使用模型默认 sampling params。
3. 调用 `_run_completion()`。

源码位置：`code/vllm/vllm/entrypoints/llm.py:465-485`。

也就是说，`LLM.generate()` 自身不直接调度 engine，而是进入 `OfflineInferenceMixin` 封装的离线执行流程。

## 7. enqueue + wait_for_completion

除了一次性 `generate()`，LLM 还支持先入队再执行：

```python
request_ids = llm.enqueue(prompts, sampling_params)
outputs = llm.wait_for_completion()
```

### 7.1 enqueue

`enqueue()`：

1. 检查 runner type。
2. 默认 sampling params。
3. 调用 `_add_completion_requests()`，只把请求加入 engine 队列，不开始处理。

源码位置：`code/vllm/vllm/entrypoints/llm.py:487-530`。

### 7.2 wait_for_completion

`wait_for_completion()`：

1. 默认接受 `RequestOutput` 和 `PoolingRequestOutput`。
2. 调用 `_run_engine()` 等待所有已入队请求完成。

源码位置：`code/vllm/vllm/entrypoints/llm.py:532-569`。

这个接口适合想要控制入队和执行边界的离线场景。

## 8. chat API

### 8.1 方法定位

`LLM.chat()` 接收 OpenAI 风格 messages，并通过 chat template 转成 prompt，再调用 generate。

源码注释：`code/vllm/vllm/entrypoints/llm.py:632-640`。

方法签名起点：`code/vllm/vllm/entrypoints/llm.py:616-631`。

### 8.2 逻辑

`chat()`：

1. 检查 runner type。
2. 默认 sampling params。
3. 调用 `_run_chat()`。

源码位置：`code/vllm/vllm/entrypoints/llm.py:682-708`。

支持参数包括：

- `messages`
- `sampling_params`
- `lora_request`
- `chat_template`
- `chat_template_content_format`
- `add_generation_prompt`
- `continue_final_message`
- `tools`
- `chat_template_kwargs`
- tokenization/multimodal kwargs

## 9. 控制类 API

`LLM` 还暴露一些 engine 控制能力：

| 方法 | 作用 | 源码位置 |
|---|---|---|
| `collective_rpc()` | 对所有 worker 执行 RPC | `code/vllm/vllm/entrypoints/llm.py:571-601` |
| `apply_model()` | 在 worker 内模型对象上执行函数 | `code/vllm/vllm/entrypoints/llm.py:603-614` |
| `start_profile()` | 开始 profiling | `code/vllm/vllm/entrypoints/llm.py:787-795` |
| `stop_profile()` | 停止 profiling | `code/vllm/vllm/entrypoints/llm.py:797-798` |
| `reset_prefix_cache()` | 重置 prefix cache | `code/vllm/vllm/entrypoints/llm.py:800-805` |
| `sleep()` | 暂停/睡眠 engine | `code/vllm/vllm/entrypoints/llm.py:807-830` |
| `wake_up()` | 唤醒 engine | `code/vllm/vllm/entrypoints/llm.py:832-845` |
| `get_metrics()` | 获取 Prometheus metrics snapshot | `code/vllm/vllm/entrypoints/llm.py:847-857` |

## 10. 权重更新 / RLHF 相关 API

`LLM` 还提供一组权重传输和更新接口：

- `init_weight_transfer_engine()`
- `start_weight_update()`
- `update_weights()`
- `finish_weight_update()`

源码位置：`code/vllm/vllm/entrypoints/llm.py:859-900`。

这些方法底层都是通过：

```python
self.llm_engine.collective_rpc(...)
```

向 worker 广播控制命令。

## 11. 离线 LLM 与在线 API 的关系

| 维度 | 离线 `LLM` | 在线 OpenAI server |
|---|---|---|
| 入口 | Python class | HTTP/CLI |
| 主要文件 | `entrypoints/llm.py` | `entrypoints/openai/api_server.py` |
| Engine | `LLMEngine` | `AsyncLLM` as `EngineClient` |
| 请求方式 | 同步批量 | 异步请求/流式 |
| 输出 | `list[RequestOutput]` | JSON/SSE |
| 适用 | offline inference/batch | online serving |

## 12. 离线 API 调用链

```text
from vllm import LLM
  -> vllm.__getattr__("LLM")
  -> vllm.entrypoints.llm.LLM.__init__
  -> EngineArgs(...)
  -> LLMEngine.from_engine_args(...)
  -> LLM.generate / LLM.chat
  -> OfflineInferenceMixin._run_completion / _run_chat
  -> LLMEngine.add_request
  -> LLMEngine.step
  -> RequestOutput
```

## 13. 关键结论

`LLM` 是“离线易用封装”，不是 HTTP 服务入口。它把复杂的 engine 配置、请求入队、执行循环和输出收集包装成同步 Python API；而在线服务则使用异步 `EngineClient`，通过 FastAPI router 和 serving 对象把 HTTP 请求转换成 engine 请求。