# 01. 用户参数如何进入配置系统？

源码位置：

- `code/vllm/vllm/engine/arg_utils.py`
- `code/vllm/vllm/entrypoints/llm.py`
- `code/vllm/vllm/entrypoints/cli/serve.py`
- `code/vllm/vllm/entrypoints/openai/cli_args.py`
- `code/vllm/vllm/entrypoints/openai/api_server.py`
- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/v1/engine/llm_engine.py`
- `code/vllm/vllm/v1/engine/async_llm.py`

本问题关注：CLI、Python `LLM`、OpenAI API server 等入口中的用户参数，如何进入 `EngineArgs` / `AsyncEngineArgs`，再转换成内部 `VllmConfig`，最后被 V1 Engine、Executor、Worker、ModelRunner 使用。

---

## 0. 梳理规划

本篇按“入口参数从哪里来 → 如何进入 EngineArgs → 如何展开成 VllmConfig → 后续谁消费配置”的顺序梳理。

要回答的问题分成 8 组：

```text
1. EngineArgs 在配置系统里处于哪一层？
2. Python LLM(...) 参数如何进入 EngineArgs？
3. vllm serve / OpenAI server 参数如何进入 AsyncEngineArgs？
4. EngineArgs.add_cli_args() 如何把 Config 字段注册成 CLI 参数？
5. EngineArgs.from_cli_args() 如何从 argparse.Namespace 取值？
6. create_engine_config() 如何创建 ModelConfig / CacheConfig / ParallelConfig / SchedulerConfig 等子配置？
7. usage_context 如何影响默认 max_num_batched_tokens / max_num_seqs？
8. VllmConfig 生成后，被 LLMEngine / AsyncLLM / Executor 如何消费？
```

最核心的文件是：

```text
arg_utils.py      # EngineArgs / AsyncEngineArgs / create_engine_config()
llm.py            # Python offline LLM 入口
cli_args.py       # OpenAI server CLI 参数注册
api_server.py     # OpenAI server 创建 AsyncEngineArgs 和 AsyncLLM
llm_engine.py     # offline engine 消费 VllmConfig
async_llm.py      # online serving engine 消费 VllmConfig
config/vllm.py    # VllmConfig 最终聚合结构
```

---

## 1. 一句话回答

`EngineArgs` 是 vLLM 用户参数到内部配置系统的第一层聚合对象。

它做三件事：

```text
1. 收拢来自 Python API / CLI / OpenAI server 的用户参数；
2. 根据平台、模型、usage_context 和互斥规则推导默认值；
3. 把扁平参数拆成 ModelConfig / CacheConfig / ParallelConfig / SchedulerConfig 等子配置，最终组装成 VllmConfig。
```

主链路是：

```text
Python LLM(...) / vllm serve ... / OpenAI API server args
  → EngineArgs 或 AsyncEngineArgs
  → EngineArgs.create_engine_config()
  → VllmConfig
  → LLMEngine / AsyncLLM
  → Executor.get_class(vllm_config)
  → EngineCore / Worker / ModelRunner
```

一句话压缩：

```text
EngineArgs 面向用户入口，VllmConfig 面向 vLLM 内部运行时。
```

---

## 2. 配置入口总览

用户参数主要有三类入口。

### 2.1 Python offline LLM 入口

入口类：`vllm.entrypoints.llm.LLM`

位置：`code/vllm/vllm/entrypoints/llm.py:66`

`LLM.__init__()` 暴露常用参数，例如：

```text
model
tokenizer
tokenizer_mode
trust_remote_code
tensor_parallel_size
dtype
quantization
revision
gpu_memory_utilization
kv_cache_memory_bytes
cpu_offload_gb
enforce_eager
hf_token
mm_processor_kwargs
pooler_config
structured_outputs_config
profiler_config
attention_config
compilation_config
spec_method / spec_model / spec_tokens
**kwargs
```

位置：`code/vllm/vllm/entrypoints/llm.py:176` 到 `code/vllm/vllm/entrypoints/llm.py:220`

随后它创建 `EngineArgs`：

```python
engine_args = EngineArgs(
    model=model,
    runner=runner,
    convert=convert,
    tokenizer=tokenizer,
    tokenizer_mode=tokenizer_mode,
    skip_tokenizer_init=skip_tokenizer_init,
    trust_remote_code=trust_remote_code,
    tensor_parallel_size=tensor_parallel_size,
    dtype=dtype,
    quantization=quantization,
    gpu_memory_utilization=gpu_memory_utilization,
    kv_cache_memory_bytes=kv_cache_memory_bytes,
    ...
    **kwargs,
)
```

位置：`code/vllm/vllm/entrypoints/llm.py:305` 到 `code/vllm/vllm/entrypoints/llm.py:345`

最后进入 engine：

```python
self.llm_engine = LLMEngine.from_engine_args(
    engine_args=engine_args,
    usage_context=UsageContext.LLM_CLASS,
)
```

位置：`code/vllm/vllm/entrypoints/llm.py:349` 到 `code/vllm/vllm/entrypoints/llm.py:351`

因此 offline 链路是：

```text
LLM(...)
  → EngineArgs(...)
  → LLMEngine.from_engine_args(..., UsageContext.LLM_CLASS)
  → engine_args.create_engine_config()
  → VllmConfig
```

### 2.2 vllm serve / OpenAI-compatible server 入口

CLI 子命令在：`code/vllm/vllm/entrypoints/cli/serve.py:44`

`ServeSubcommand.subparser_init()` 会调用 `make_arg_parser()` 注册 serve 参数：

```python
serve_parser = make_arg_parser(serve_parser)
```

位置：`code/vllm/vllm/entrypoints/cli/serve.py:153` 到 `code/vllm/vllm/entrypoints/cli/serve.py:165`

`make_arg_parser()` 做两类参数注册：

```python
parser = FrontendArgs.add_cli_args(parser)
parser = AsyncEngineArgs.add_cli_args(parser)
```

位置：`code/vllm/vllm/entrypoints/openai/cli_args.py:340` 到 `code/vllm/vllm/entrypoints/openai/cli_args.py:383`

含义是：

```text
FrontendArgs：HTTP server / OpenAI 协议 / CORS / tool parser / chat template 等前端参数；
AsyncEngineArgs：模型、缓存、并行、调度、LoRA、编译、可观测性等 engine 参数。
```

`vllm serve [model_tag]` 的位置参数会覆盖 `args.model`：

```python
if hasattr(args, "model_tag") and args.model_tag is not None:
    args.model = args.model_tag
```

位置：`code/vllm/vllm/entrypoints/cli/serve.py:50` 到 `code/vllm/vllm/entrypoints/cli/serve.py:53`

### 2.3 OpenAI API server 创建 AsyncEngineArgs

OpenAI server 构造 engine client 的入口是：

```python
async def build_async_engine_client(args, usage_context=UsageContext.OPENAI_API_SERVER, ...)
```

位置：`code/vllm/vllm/entrypoints/openai/api_server.py:117` 到 `code/vllm/vllm/entrypoints/openai/api_server.py:122`

这里从 argparse namespace 创建 `AsyncEngineArgs`：

```python
engine_args = AsyncEngineArgs.from_cli_args(args)
```

位置：`code/vllm/vllm/entrypoints/openai/api_server.py:134`

然后进入：

```text
build_async_engine_client_from_engine_args()
  → engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
  → AsyncLLM.from_vllm_config(...)
```

对应源码：

```python
vllm_config = engine_args.create_engine_config(usage_context=usage_context)

async_llm = AsyncLLM.from_vllm_config(
    vllm_config=vllm_config,
    usage_context=usage_context,
    enable_log_requests=engine_args.enable_log_requests,
    aggregate_engine_logging=engine_args.aggregate_engine_logging,
    disable_log_stats=engine_args.disable_log_stats,
    ...
)
```

位置：`code/vllm/vllm/entrypoints/openai/api_server.py:163` 到 `code/vllm/vllm/entrypoints/openai/api_server.py:184`

因此 server 链路是：

```text
vllm serve ...
  → argparse.Namespace
  → AsyncEngineArgs.from_cli_args(args)
  → create_engine_config(UsageContext.OPENAI_API_SERVER)
  → AsyncLLM.from_vllm_config(vllm_config)
```

---

## 3. EngineArgs 是什么

`EngineArgs` 定义在：`code/vllm/vllm/engine/arg_utils.py:417`

```python
@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
```

它是一个很大的 dataclass，字段多数直接取自各个内部 Config 类的默认值。

### 3.1 字段来源

典型字段映射如下：

```text
EngineArgs.model                         ← ModelConfig.model
EngineArgs.tokenizer                     ← ModelConfig.tokenizer
EngineArgs.runner                        ← ModelConfig.runner
EngineArgs.dtype                         ← ModelConfig.dtype
EngineArgs.kv_cache_dtype                ← CacheConfig.cache_dtype
EngineArgs.distributed_executor_backend  ← ParallelConfig.distributed_executor_backend
EngineArgs.pipeline_parallel_size        ← ParallelConfig.pipeline_parallel_size
EngineArgs.tensor_parallel_size          ← ParallelConfig.tensor_parallel_size
EngineArgs.gpu_memory_utilization        ← CacheConfig.gpu_memory_utilization
EngineArgs.max_num_partial_prefills      ← SchedulerConfig.max_num_partial_prefills
EngineArgs.scheduling_policy             ← SchedulerConfig.policy
EngineArgs.compilation_config            ← VllmConfig.compilation_config
EngineArgs.attention_config              ← VllmConfig.attention_config
EngineArgs.kernel_config                 ← VllmConfig.kernel_config
EngineArgs.reasoning_config              ← VllmConfig.reasoning_config
```

位置：`code/vllm/vllm/engine/arg_utils.py:420` 到 `code/vllm/vllm/engine/arg_utils.py:719`

这说明 `EngineArgs` 不是最终配置对象，而是面向用户入口的“扁平参数层”。

### 3.2 它收拢哪些配置域

可以按配置域理解：

```text
Model / tokenizer：model, tokenizer, dtype, revision, hf_token, trust_remote_code
Load：load_format, download_dir, safetensors_load_strategy, model_loader_extra_config
Parallel：TP, PP, DP, executor backend, Ray/mp/external launcher, expert parallel
Cache：block_size, gpu_memory_utilization, kv_cache_memory_bytes, kv_cache_dtype
Scheduler：max_num_batched_tokens, max_num_seqs, chunked prefill, policy
Multimodal：limit_mm_per_prompt, mm_processor_kwargs, media domains, renderer workers
LoRA：enable_lora, max_loras, max_lora_rank, lora_dtype, default_mm_loras
Speculative：speculative_config, spec_method, spec_model, spec_tokens
Compilation：compilation_config, cudagraph_capture_sizes, max_cudagraph_capture_size
Attention / kernel：attention_backend, attention_config, moe_backend, linear_backend
Observability：traces, metrics, request logging, cudagraph metrics
Transfer：kv_transfer_config, ec_transfer_config, weight_transfer_config
Offload：cpu_offload_gb, prefetch offload, kv_offloading
```

### 3.3 __post_init__ 做什么

`EngineArgs.__post_init__()` 处理两类事情：

```text
1. 把 dict 形式的配置转换成对应 Config 对象；
2. 做初始化期的插件加载、量化配置解析、HF offline model/tokenizer 路径替换。
```

关键逻辑包括：

```python
if isinstance(self.compilation_config, dict):
    self.compilation_config = CompilationConfig(**self.compilation_config)
if isinstance(self.attention_config, dict):
    self.attention_config = AttentionConfig(**self.attention_config)
...
self.quantization_config = resolve_quantization_config(
    self.quantization, self.quantization_config
)
load_general_plugins()
```

位置：`code/vllm/vllm/engine/arg_utils.py:721` 到 `code/vllm/vllm/engine/arg_utils.py:770`

HF offline 场景下，如果 `HF_HUB_OFFLINE=True`，它会把 model/tokenizer id 替换成本地路径：

```python
self.model = get_model_path(self.model, self.revision)
self.tokenizer = get_model_path(self.tokenizer, self.tokenizer_revision)
```

位置：`code/vllm/vllm/engine/arg_utils.py:772` 到 `code/vllm/vllm/engine/arg_utils.py:789`

---

## 4. CLI 参数如何注册到 EngineArgs

CLI 参数注册入口：`EngineArgs.add_cli_args()`

位置：`code/vllm/vllm/engine/arg_utils.py:795`

它不是完全手写每个参数的类型和 help，而是先从 Config 类生成 argparse kwargs：

```python
model_kwargs = get_kwargs(ModelConfig)
load_kwargs = get_kwargs(LoadConfig)
```

位置：`code/vllm/vllm/engine/arg_utils.py:799` 和 `code/vllm/vllm/engine/arg_utils.py:894`

然后按配置域添加参数组：

```text
ModelConfig
LoadConfig
ParallelConfig
CacheConfig
MultiModalConfig
LoRAConfig
SchedulerConfig
SpeculativeConfig
ObservabilityConfig
CompilationConfig
AttentionConfig
MambaConfig
KernelConfig
ProfilerConfig
KVTransferConfig / KVEventsConfig / ECTransferConfig
```

这种设计的意义是：

```text
Config 类字段是参数定义的源头；
EngineArgs.add_cli_args() 负责把这些字段暴露成命令行参数；
EngineArgs.from_cli_args() 再把解析结果收回 dataclass。
```

### 4.1 from_cli_args 如何取值

`from_cli_args()` 很直接：

```python
attrs = [attr.name for attr in dataclasses.fields(cls)]
engine_args = cls(
    **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1610` 到 `code/vllm/vllm/engine/arg_utils.py:1617`

也就是说，argparse namespace 中只有和 `EngineArgs` dataclass 字段同名的值，才会进入 `EngineArgs`。

这也解释了为什么 OpenAI server 的 `FrontendArgs` 不会混进 engine 配置：

```text
args.host / args.port / args.chat_template / args.allowed_origins 等属于 frontend；
只有 EngineArgs / AsyncEngineArgs dataclass 中存在的字段才会被 from_cli_args() 取走。
```

### 4.2 AsyncEngineArgs 多了什么

`AsyncEngineArgs` 继承 `EngineArgs`：

```python
@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""

    enable_log_requests: bool = False
```

位置：`code/vllm/vllm/engine/arg_utils.py:2704` 到 `code/vllm/vllm/engine/arg_utils.py:2707`

它的 `add_cli_args()` 默认先注册 `EngineArgs` 参数，再额外注册异步 server 相关参数：

```python
if not async_args_only:
    parser = EngineArgs.add_cli_args(parser)
parser.add_argument("--enable-log-requests", ...)
```

位置：`code/vllm/vllm/engine/arg_utils.py:2710` 到 `code/vllm/vllm/engine/arg_utils.py:2729`

所以：

```text
EngineArgs：offline 和 engine 通用参数；
AsyncEngineArgs：EngineArgs + async serving 额外参数。
```

---

## 5. create_engine_config() 是核心转换点

`EngineArgs.create_engine_config()` 定义在：`code/vllm/vllm/engine/arg_utils.py:1833`

它的职责是把 `EngineArgs` 的扁平字段转换为内部结构化配置：

```python
def create_engine_config(
    self,
    usage_context: UsageContext | None = None,
    headless: bool = False,
) -> VllmConfig:
```

位置：`code/vllm/vllm/engine/arg_utils.py:1833` 到 `code/vllm/vllm/engine/arg_utils.py:1837`

主流程可以压缩成：

```text
1. 注册并更新平台信息；
2. 创建 DeviceConfig；
3. 校验环境变量；
4. 处理 speculator 模型覆盖；
5. 创建 ModelConfig；
6. 根据模型能力推导 chunked prefill / prefix caching；
7. 解析 kv_cache_dtype；
8. 创建 CacheConfig；
9. 推导 DP / PP / TP / backend / address / rank；
10. 创建 ParallelConfig；
11. 创建 SpeculativeConfig / DiffusionConfig；
12. 根据 usage_context 推导 scheduler 默认 batch 上限；
13. 创建 SchedulerConfig；
14. 创建 LoRAConfig；
15. 应用 AttentionConfig / MambaConfig / KernelConfig 顶层覆盖；
16. 创建 LoadConfig / ObservabilityConfig / CompilationConfig / OffloadConfig；
17. 组装并返回 VllmConfig。
```

### 5.1 平台、设备和环境校验

开头会先做平台注册和设备配置：

```python
current_platform.pre_register_and_update()
device_config = DeviceConfig(device=cast(Device, current_platform.device_type))
envs.validate_environ(self.fail_on_environ_validation)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1843` 到 `code/vllm/vllm/engine/arg_utils.py:1847`

这意味着后面的默认值推导不是纯静态的，会依赖当前平台：

```text
CUDA / ROCm / TPU / CPU
设备显存
设备名称
Ray 是否初始化
是否处于 Ray actor
```

### 5.2 创建 ModelConfig

`create_model_config()` 负责把模型相关字段变成 `ModelConfig`：

```python
return ModelConfig(
    model=self.model,
    model_weights=self.model_weights,
    hf_config_path=self.hf_config_path,
    runner=self.runner,
    convert=self.convert,
    tokenizer=self.tokenizer,
    tokenizer_mode=self.tokenizer_mode,
    trust_remote_code=self.trust_remote_code,
    allowed_local_media_path=self.allowed_local_media_path,
    allowed_media_domains=self.allowed_media_domains,
    dtype=self.dtype,
    seed=self.seed,
    revision=self.revision,
    code_revision=self.code_revision,
    hf_token=self.hf_token,
    hf_overrides=self.hf_overrides,
    tokenizer_revision=self.tokenizer_revision,
    max_model_len=self.max_model_len,
    quantization=self.quantization,
    quantization_config=self.quantization_config,
    ...
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1619` 到 `code/vllm/vllm/engine/arg_utils.py:1693`

在调用它之前，`create_engine_config()` 还会检查当前 `model` 是否是 speculator，并可能改写 model/tokenizer/speculative_config：

```python
(self.model, self.tokenizer, self.speculative_config) = maybe_override_with_speculators(...)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1849` 到 `code/vllm/vllm/engine/arg_utils.py:1865`

创建后会把规范化后的值写回 `EngineArgs`：

```python
self.model = model_config.model
self.model_weights = model_config.model_weights
self.tokenizer = model_config.tokenizer
```

位置：`code/vllm/vllm/engine/arg_utils.py:1866` 到 `code/vllm/vllm/engine/arg_utils.py:1869`

### 5.3 推导 chunked prefill 和 prefix caching

模型配置创建后，会调用：

```python
self._set_default_chunked_prefill_and_prefix_caching_args(model_config)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1871` 到 `code/vllm/vllm/engine/arg_utils.py:1873`

默认值来自模型能力：

```python
default_chunked_prefill = model_config.is_chunked_prefill_supported
default_prefix_caching = model_config.is_prefix_caching_supported
```

位置：`code/vllm/vllm/engine/arg_utils.py:2497` 到 `code/vllm/vllm/engine/arg_utils.py:2505`

含义是：

```text
如果用户没显式设置 enable_chunked_prefill，就按模型是否支持来启用；
如果用户没显式设置 enable_prefix_caching，就按模型是否支持来启用；hybrid model 当前保持默认关闭；
如果用户强行设置了模型不推荐的组合，会给 warning 或禁用。
```

RISC-V CPU 场景会强制关闭这两项：

位置：`code/vllm/vllm/engine/arg_utils.py:2553` 到 `code/vllm/vllm/engine/arg_utils.py:2569`

### 5.4 创建 CacheConfig

KV cache dtype 先从字符串解析成实际 dtype：

```python
resolved_cache_dtype = resolve_kv_cache_dtype_string(
    self.kv_cache_dtype,
    model_config,
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1881` 到 `code/vllm/vllm/engine/arg_utils.py:1884`

然后创建 `CacheConfig`：

```python
cache_config = CacheConfig(
    block_size=self.block_size,
    gpu_memory_utilization=self.gpu_memory_utilization,
    kv_cache_memory_bytes=self.kv_cache_memory_bytes,
    cache_dtype=resolved_cache_dtype,
    is_attention_free=model_config.is_attention_free,
    num_gpu_blocks_override=self.num_gpu_blocks_override,
    sliding_window=sliding_window,
    enable_prefix_caching=self.enable_prefix_caching,
    prefix_caching_hash_algo=self.prefix_caching_hash_algo,
    calculate_kv_scales=self.calculate_kv_scales,
    kv_cache_dtype_skip_layers=self.kv_cache_dtype_skip_layers,
    kv_sharing_fast_prefill=self.kv_sharing_fast_prefill,
    prefix_match_unit=self.prefix_match_unit,
    mamba_cache_dtype=self.mamba_cache_dtype,
    mamba_ssm_cache_dtype=self.mamba_ssm_cache_dtype,
    mamba_block_size=self.mamba_block_size,
    mamba_cache_mode=self.mamba_cache_mode,
    kv_offloading_size=self.kv_offloading_size,
    kv_offloading_backend=self.kv_offloading_backend,
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1890` 到 `code/vllm/vllm/engine/arg_utils.py:1910`

这里有两个关键点：

```text
1. CacheConfig 不只来自用户参数，也依赖 ModelConfig，例如 attention-free、sliding window；
2. kv_cache_dtype="auto" 这类值会在这里解析成实际 dtype。
```

### 5.5 创建 ParallelConfig

`create_engine_config()` 会先处理 Ray runtime env、placement group、DP rank、DP local size、DP address、RPC port 等推导，再创建 `ParallelConfig`。

位置：`code/vllm/vllm/engine/arg_utils.py:1923` 到 `code/vllm/vllm/engine/arg_utils.py:2093`

创建 `ParallelConfig` 时会收拢：

```python
parallel_config = ParallelConfig(
    pipeline_parallel_size=self.pipeline_parallel_size,
    tensor_parallel_size=self.tensor_parallel_size,
    prefill_context_parallel_size=self.prefill_context_parallel_size,
    data_parallel_size=self.data_parallel_size,
    data_parallel_rank=self.data_parallel_rank or 0,
    data_parallel_external_lb=data_parallel_external_lb,
    data_parallel_size_local=data_parallel_size_local,
    master_addr=self.master_addr,
    master_port=self.master_port,
    nnodes=self.nnodes,
    node_rank=self.node_rank,
    distributed_executor_backend=self.distributed_executor_backend,
    worker_cls=self.worker_cls,
    worker_extension_cls=self.worker_extension_cls,
    ...
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:2094` 到 `code/vllm/vllm/engine/arg_utils.py:2143`

这一步决定了后续 Executor 选择和分布式拓扑。

例如：

```text
distributed_executor_backend = "uni" / "mp" / "ray" / "external_launcher" / 自定义 Executor 类
pipeline_parallel_size       = PP 大小
tensor_parallel_size         = TP 大小
data_parallel_size           = DP 大小
data_parallel_rank           = 当前 DP rank
```

### 5.6 创建 SchedulerConfig

在创建 scheduler 配置前，会先调用：

```python
self._set_default_max_num_seqs_and_batched_tokens_args(
    usage_context,
    model_config,
    parallel_config,
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:2151` 到 `code/vllm/vllm/engine/arg_utils.py:2155`

然后创建 `SchedulerConfig`：

```python
scheduler_config = SchedulerConfig(
    runner_type=model_config.runner_type,
    max_num_batched_tokens=self.max_num_batched_tokens,
    max_num_scheduled_tokens=self.max_num_scheduled_tokens,
    max_num_seqs=self.max_num_seqs,
    max_model_len=model_config.max_model_len,
    enable_chunked_prefill=self.enable_chunked_prefill,
    disable_chunked_mm_input=self.disable_chunked_mm_input,
    is_multimodal_model=model_config.is_multimodal_model,
    is_encoder_decoder=model_config.is_encoder_decoder,
    policy=self.scheduling_policy,
    scheduler_cls=self.scheduler_cls,
    max_num_partial_prefills=self.max_num_partial_prefills,
    max_long_partial_prefills=self.max_long_partial_prefills,
    long_prefill_token_threshold=self.long_prefill_token_threshold,
    scheduler_reserve_full_isl=self.scheduler_reserve_full_isl,
    watermark=self.watermark,
    prefill_schedule_interval=self.prefill_schedule_interval,
    disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
    async_scheduling=self.async_scheduling,
    stream_interval=self.stream_interval,
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:2167` 到 `code/vllm/vllm/engine/arg_utils.py:2188`

注意 `SchedulerConfig` 不是只包含用户传的 scheduler 参数，它还依赖：

```text
ModelConfig.runner_type
ModelConfig.max_model_len
ModelConfig.is_multimodal_model
ModelConfig.is_encoder_decoder
chunked prefill / prefix caching 默认推导结果
usage_context 对 batch 默认值的影响
```

---

## 6. usage_context 如何影响默认 batch 上限

`usage_context` 是这条链路里很容易忽略的参数。

offline `LLM` 传的是：

```text
UsageContext.LLM_CLASS
```

位置：`code/vllm/vllm/entrypoints/llm.py:349` 到 `code/vllm/vllm/entrypoints/llm.py:351`

OpenAI server 传的是：

```text
UsageContext.OPENAI_API_SERVER
```

位置：`code/vllm/vllm/entrypoints/openai/api_server.py:117` 到 `code/vllm/vllm/entrypoints/openai/api_server.py:122`

`EngineArgs.get_batch_defaults()` 会根据 usage context、设备类型、显存、world size 给出不同默认值。

位置：`code/vllm/vllm/engine/arg_utils.py:2414` 到 `code/vllm/vllm/engine/arg_utils.py:2495`

典型规则：

```text
大显存非 A100 GPU：
  LLM_CLASS         max_num_batched_tokens = 16384, max_num_seqs = 1024
  OPENAI_API_SERVER max_num_batched_tokens = 8192,  max_num_seqs = 1024

普通 GPU：
  LLM_CLASS         max_num_batched_tokens = 8192, max_num_seqs = 256
  OPENAI_API_SERVER max_num_batched_tokens = 2048, max_num_seqs = 256

CPU：
  LLM_CLASS         max_num_batched_tokens = 4096 * world_size, max_num_seqs = 256 * world_size
  OPENAI_API_SERVER max_num_batched_tokens = 2048 * world_size, max_num_seqs = 128 * world_size
```

如果用户没有显式传 `max_num_batched_tokens` 或 `max_num_seqs`，才会使用这些默认值；batched DP MoE 会先使用 `SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP`：

```python
if self.max_num_batched_tokens is None:
    self.max_num_batched_tokens = default_max_num_batched_tokens.get(...)

if self.max_num_seqs is None:
    self.max_num_seqs = default_max_num_seqs.get(...)
```

位置：`code/vllm/vllm/engine/arg_utils.py:2626` 到 `code/vllm/vllm/engine/arg_utils.py:2641`

后续还会继续修正：

```text
performance_mode == "throughput" 时，如果用户没显式设置，就把默认 batch 上限翻倍；
未启用 chunked prefill 时，max_num_batched_tokens 至少要覆盖 max_model_len；
多模态 prefix-LM 模型可能抬高 max_num_batched_tokens，保证一个多模态 item 能放进 batch；
默认 max_num_batched_tokens 不能超过 max_num_seqs * max_model_len；
默认 max_num_seqs 不能超过 max_num_batched_tokens。
```

位置：`code/vllm/vllm/engine/arg_utils.py:2643` 到 `code/vllm/vllm/engine/arg_utils.py:2700`

因此，同样不传 `--max-num-batched-tokens`：

```text
LLM(...) 和 vllm serve ... 的默认值可能不同；
GPU 和 CPU 的默认值可能不同；
大显存 GPU 和普通 GPU 的默认值可能不同；
throughput mode 会进一步改变默认值。
```

---

## 7. 其他子配置如何生成

除了 Model / Cache / Parallel / Scheduler，`create_engine_config()` 还会生成或整理其他子配置。

### 7.1 LoadConfig

`create_load_config()` 处理模型加载相关参数。

位置：`code/vllm/vllm/engine/arg_utils.py:1704` 到 `code/vllm/vllm/engine/arg_utils.py:1729`

关键逻辑：

```text
quantization == "bitsandbytes" 时，load_format 会改为 "bitsandbytes"；
load_format == "tensorizer" 时，会把 model_loader_extra_config 整理成 tensorizer_config；
最终创建 LoadConfig。
```

### 7.2 SpeculativeConfig

投机解码配置由 `create_speculative_config()` 创建。

位置：`code/vllm/vllm/engine/arg_utils.py:1731` 到 `code/vllm/vllm/engine/arg_utils.py:1766`

它支持两种来源：

```text
--speculative-config 字典；
--spec-method / --spec-model / --spec-tokens 顶层别名。
```

如果两种方式同时设置同一个 key，会报互斥错误：

```python
raise ValueError(
    f"{flag} and --speculative-config['{key}'] are mutually exclusive"
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1748` 到 `code/vllm/vllm/engine/arg_utils.py:1751`

并且它会把目标模型和目标并行配置塞进 speculative config：

```python
self.speculative_config.update(
    {
        "target_model_config": target_model_config,
        "target_parallel_config": target_parallel_config,
    }
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:1760` 到 `code/vllm/vllm/engine/arg_utils.py:1766`

### 7.3 LoRAConfig

只有 `enable_lora=True` 时才创建 `LoRAConfig`，否则是 `None`：

```python
lora_config = LoRAConfig(...) if self.enable_lora else None
```

位置：`code/vllm/vllm/engine/arg_utils.py:2196` 到 `code/vllm/vllm/engine/arg_utils.py:2213`

如果给非多模态模型传了 `default_mm_loras`，会报错：

位置：`code/vllm/vllm/engine/arg_utils.py:2190` 到 `code/vllm/vllm/engine/arg_utils.py:2194`

### 7.4 AttentionConfig / MambaConfig / KernelConfig

这些配置先复制 `self.*_config`，再应用顶层参数覆盖。

Attention：

```text
如果传了 attention_backend，且 attention_config.backend 也已经设置，则报互斥错误；
否则把 attention_backend 写入 attention_config.backend。
```

位置：`code/vllm/vllm/engine/arg_utils.py:2233` 到 `code/vllm/vllm/engine/arg_utils.py:2245`

Mamba：

```text
mamba_backend 字符串会转换为 MambaBackendEnum；
enable_mamba_cache_stochastic_rounding / mamba_cache_philox_rounds 会写入 MambaConfig。
```

位置：`code/vllm/vllm/engine/arg_utils.py:2259` 到 `code/vllm/vllm/engine/arg_utils.py:2273`

Kernel：

```text
enable_flashinfer_autotune 与 kernel_config.enable_flashinfer_autotune 互斥；
moe_backend != "auto" 时写入 kernel_config.moe_backend；
linear_backend != "auto" 时写入 kernel_config.linear_backend；
ir_op_priority 顶层配置会合并到 kernel_config.ir_op_priority。
```

位置：`code/vllm/vllm/engine/arg_utils.py:2275` 到 `code/vllm/vllm/engine/arg_utils.py:2305`

### 7.5 ObservabilityConfig / CompilationConfig / OffloadConfig

ObservabilityConfig：

```python
observability_config = self.create_observability_config()
```

`create_observability_config()` 负责把 metrics、trace、NVTX、MFU、MM processor stats、JIT monitor 等观测参数组装成 `ObservabilityConfig`。

位置：`code/vllm/vllm/engine/arg_utils.py:1817` 到 `code/vllm/vllm/engine/arg_utils.py:1831`，调用位置在 `code/vllm/vllm/engine/arg_utils.py:2317`

CompilationConfig：

```text
cudagraph_capture_sizes 与 compilation_config.cudagraph_capture_sizes 互斥；
max_cudagraph_capture_size 与 compilation_config.max_cudagraph_capture_size 互斥；
顶层参数会覆盖进 compilation_config。
```

位置：`code/vllm/vllm/engine/arg_utils.py:2319` 到 `code/vllm/vllm/engine/arg_utils.py:2337`

OffloadConfig：

```python
offload_config = OffloadConfig(
    offload_backend=self.offload_backend,
    uva=UVAOffloadConfig(...),
    prefetch=PrefetchOffloadConfig(...),
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:2338` 到 `code/vllm/vllm/engine/arg_utils.py:2350`

---

## 8. VllmConfig 是最终内部配置对象

`VllmConfig` 定义在：`code/vllm/vllm/config/vllm.py:288`

```python
@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """
```

位置：`code/vllm/vllm/config/vllm.py:288` 到 `code/vllm/vllm/config/vllm.py:292`

它包含的主要子配置是：

```text
model_config
cache_config
parallel_config
scheduler_config
device_config
load_config
offload_config
attention_config
mamba_config
kernel_config
lora_config
speculative_config
diffusion_config
structured_outputs_config
observability_config
quant_config
compilation_config
profiler_config
kv_transfer_config
kv_events_config
ec_transfer_config
reasoning_config
additional_config
instance_id
optimization_level
performance_mode
weight_transfer_config
shutdown_timeout
```

位置：`code/vllm/vllm/config/vllm.py:295` 到 `code/vllm/vllm/config/vllm.py:382`

`create_engine_config()` 最后组装：

```python
config = VllmConfig(
    model_config=model_config,
    cache_config=cache_config,
    parallel_config=parallel_config,
    scheduler_config=scheduler_config,
    device_config=device_config,
    load_config=load_config,
    offload_config=offload_config,
    attention_config=attention_config,
    mamba_config=mamba_config,
    kernel_config=kernel_config,
    lora_config=lora_config,
    speculative_config=speculative_config,
    diffusion_config=diffusion_config,
    structured_outputs_config=self.structured_outputs_config,
    observability_config=observability_config,
    compilation_config=compilation_config,
    kv_transfer_config=self.kv_transfer_config,
    kv_events_config=self.kv_events_config,
    ec_transfer_config=self.ec_transfer_config,
    reasoning_config=self.reasoning_config,
    profiler_config=self.profiler_config,
    additional_config=self.additional_config,
    optimization_level=self.optimization_level,
    performance_mode=self.performance_mode,
    weight_transfer_config=self.weight_transfer_config,
    shutdown_timeout=self.shutdown_timeout,
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:2355` 到 `code/vllm/vllm/engine/arg_utils.py:2382`

然后返回：

```python
return config
```

位置：`code/vllm/vllm/engine/arg_utils.py:2384`

---

## 9. VllmConfig 后续被谁消费

### 9.1 LLMEngine.from_engine_args

offline 路径会调用：

```python
vllm_config = engine_args.create_engine_config(usage_context)
executor_class = Executor.get_class(vllm_config)
```

位置：`code/vllm/vllm/v1/engine/llm_engine.py:161` 到 `code/vllm/vllm/v1/engine/llm_engine.py:172`

随后创建 `LLMEngine`：

```python
return cls(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=not engine_args.disable_log_stats,
    usage_context=usage_context,
    stat_loggers=stat_loggers,
    multiprocess_mode=enable_multiprocessing,
)
```

位置：`code/vllm/vllm/v1/engine/llm_engine.py:179` 到 `code/vllm/vllm/v1/engine/llm_engine.py:186`

### 9.2 LLMEngine.__init__ 消费哪些配置

`LLMEngine.__init__()` 保存：

```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.observability_config = vllm_config.observability_config
```

位置：`code/vllm/vllm/v1/engine/llm_engine.py:51` 到 `code/vllm/vllm/v1/engine/llm_engine.py:65`

然后使用配置创建：

```text
renderer_from_config(vllm_config)
InputProcessor(vllm_config, renderer)
OutputProcessor(..., stream_interval=vllm_config.scheduler_config.stream_interval)
EngineCoreClient.make_client(vllm_config, executor_class, ...)
StatLoggerManager(vllm_config, ...)
```

位置：`code/vllm/vllm/v1/engine/llm_engine.py:91` 到 `code/vllm/vllm/v1/engine/llm_engine.py:121`

### 9.3 AsyncLLM.from_vllm_config

server 路径通常已经在 `api_server.py` 里创建好了 `VllmConfig`，然后调用：

```python
AsyncLLM.from_vllm_config(vllm_config=..., ...)
```

`AsyncLLM.from_vllm_config()` 会通过配置选择 executor：

```python
return cls(
    vllm_config=vllm_config,
    executor_class=Executor.get_class(vllm_config),
    ...
)
```

位置：`code/vllm/vllm/v1/engine/async_llm.py:203` 到 `code/vllm/vllm/v1/engine/async_llm.py:229`

### 9.4 AsyncLLM.__init__ 消费哪些配置

`AsyncLLM.__init__()` 保存：

```python
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.observability_config = vllm_config.observability_config
```

位置：`code/vllm/vllm/v1/engine/async_llm.py:73` 到 `code/vllm/vllm/v1/engine/async_llm.py:112`

然后创建：

```text
renderer_from_config(vllm_config)
InputProcessor(vllm_config, renderer)
OutputProcessor(..., stream_interval=vllm_config.scheduler_config.stream_interval)
EngineCoreClient.make_async_mp_client(vllm_config, executor_class, ...)
StatLoggerManager(vllm_config, ...)
```

位置：`code/vllm/vllm/v1/engine/async_llm.py:132` 到 `code/vllm/vllm/v1/engine/async_llm.py:166`

---

## 10. 完整链路图

### 10.1 Python LLM 链路

```text
用户代码：
  LLM(model="...", tensor_parallel_size=2, gpu_memory_utilization=0.9, ...)

进入：
  vllm.entrypoints.llm.LLM.__init__()

构造：
  EngineArgs(...)

创建 engine：
  LLMEngine.from_engine_args(engine_args, UsageContext.LLM_CLASS)

转换配置：
  engine_args.create_engine_config(UsageContext.LLM_CLASS)

得到：
  VllmConfig(
    model_config=...,
    cache_config=...,
    parallel_config=...,
    scheduler_config=...,
    ...
  )

选择后端：
  Executor.get_class(vllm_config)

创建运行时：
  LLMEngine
    → renderer
    → InputProcessor
    → OutputProcessor
    → EngineCoreClient
    → EngineCore / Executor / Worker
```

### 10.2 OpenAI server 链路

```text
命令行：
  vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 2 --max-num-seqs 128 ...

注册参数：
  make_arg_parser()
    → FrontendArgs.add_cli_args()
    → AsyncEngineArgs.add_cli_args()
      → EngineArgs.add_cli_args()

解析得到：
  argparse.Namespace

serve 子命令处理：
  model_tag 覆盖 args.model
  api_server_count / headless / DP LB 模式默认值处理

server 创建 engine client：
  AsyncEngineArgs.from_cli_args(args)

转换配置：
  engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

创建异步运行时：
  AsyncLLM.from_vllm_config(vllm_config)
    → Executor.get_class(vllm_config)
    → EngineCoreClient.make_async_mp_client(...)
```

### 10.3 headless serve 链路

headless 模式不启动 API server，但仍然会创建 engine config：

```python
engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
vllm_config = engine_args.create_engine_config(
    usage_context=UsageContext.OPENAI_API_SERVER,
    headless=True,
)
```

位置：`code/vllm/vllm/entrypoints/cli/serve.py:173` 到 `code/vllm/vllm/entrypoints/cli/serve.py:182`

随后 headless worker 会使用这个 `VllmConfig` 创建 executor。

---

## 11. 关键边界：EngineArgs 不负责什么

`EngineArgs` 负责“参数到配置”，但不负责真正运行模型。

它不负责：

```text
不负责 tokenizer 具体 encode；
不负责请求队列和调度循环；
不负责 KV cache block 的实际分配；
不负责 Executor/Worker 进程启动细节；
不负责模型权重真正加载；
不负责 attention metadata 构造；
不负责 forward / logits / sampling。
```

这些事情分别在后续层完成：

```text
InputProcessor / renderer：输入处理、多模态渲染、tokenizer 前处理；
Scheduler：请求调度、token budget、running/waiting 管理；
EngineCore：主循环和调度执行编排；
Executor：执行后端选择后的 RPC / worker 管理；
Worker：设备初始化、KV cache 初始化、模型加载；
ModelRunner：准备 batch、attention metadata、forward、logits、sampling。
```

`EngineArgs` 真正的边界是：

```text
把用户可见的扁平参数，转换成内部各层可以稳定消费的 VllmConfig。
```

---

## 12. 小结

配置入口可以分成两层：

```text
EngineArgs / AsyncEngineArgs：入口层，面向用户参数和 CLI 参数；
VllmConfig：运行时层，面向 vLLM 内部组件。
```

最重要的转换点是：

```text
EngineArgs.create_engine_config()
```

它不是简单字段拷贝，而会做：

```text
平台识别；
环境校验；
模型配置规范化；
HF offline 路径替换；
speculator 自动覆盖；
chunked prefill / prefix caching 默认推导；
KV cache dtype 解析；
DP / TP / PP / executor backend 推导；
usage_context 相关 batch 默认值推导；
LoRA / speculative / attention / kernel / compilation 互斥校验和合并；
最终组装 VllmConfig。
```

所以如果要追踪一个用户参数最终在哪里生效，通用阅读路径是：

```text
入口参数名
  → EngineArgs 字段
  → add_cli_args() 是否注册 CLI
  → from_cli_args() 或 LLM.__init__() 如何赋值
  → create_engine_config() 写入哪个子 Config
  → VllmConfig.<子配置>
  → Engine / Executor / Worker / ModelRunner 中哪个组件读取它
```
