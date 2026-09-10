# 10. 从配置到运行时对象的生命周期顺序是什么？

源码位置：

- `code/vllm/vllm/engine/arg_utils.py`
- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/config/model.py`
- `code/vllm/vllm/v1/engine/llm_engine.py`
- `code/vllm/vllm/v1/engine/core_client.py`
- `code/vllm/vllm/v1/engine/core.py`
- `code/vllm/vllm/v1/executor/abstract.py`
- `code/vllm/vllm/v1/executor/uniproc_executor.py`
- `code/vllm/vllm/v1/worker/worker_base.py`
- `code/vllm/vllm/v1/worker/gpu_worker.py`
- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/model_executor/model_loader/`

本问题关注：把配置和模型加载专题串成一条完整生命周期链，说明 vLLM V1 从用户参数到可执行模型，中间对象按什么顺序创建、校验、加载、profile、分配 KV cache、warmup / compile / CUDA graph capture，最后进入 `EngineCore.step()` 的运行时闭环。

---

## 1. 一句话回答

vLLM V1 的启动顺序可以概括为：

```text
先把用户参数变成 VllmConfig，
再用 VllmConfig 创建 EngineCore / Executor / Worker / ModelRunner，
然后加载模型权重，
再基于真实模型 profile 可用显存并分配 KV cache，
最后 warmup / compile / CUDA graph capture，
进入可执行的 step 循环。
```

最小主链是：

```text
用户参数
  → EngineArgs
  → EngineArgs.create_engine_config()
  → VllmConfig
  → LLMEngine / EngineCoreClient
  → EngineCore
  → Executor
  → WorkerWrapper / GPUWorker
  → GPUModelRunner
  → load_model()
  → get_kv_cache_spec()
  → determine_available_memory() / profile_run()
  → get_kv_cache_configs()
  → initialize_kv_cache()
  → compile_or_warm_up_model()
  → capture_model()
  → EngineCore.step()
```

一句话压缩：

```text
配置先决定“能怎么跑”，启动阶段再把这些配置逐层固化成 engine、executor、worker、model、KV cache 和 compiled runtime。
```

---

## 2. 总体生命周期顺序

从冷启动到第一轮执行，可以分成 8 个阶段：

```text
1. 参数阶段：用户参数 → EngineArgs；
2. 配置阶段：EngineArgs → VllmConfig 和各子配置；
3. Engine 阶段：LLMEngine / EngineCoreClient / EngineCore 创建；
4. 执行层阶段：Executor 创建 Worker；
5. 设备阶段：Worker 初始化 device / distributed / ModelRunner；
6. 模型阶段：ModelRunner 加载模型类和权重；
7. KV cache 阶段：查询 spec、profile 显存、生成 KVCacheConfig、分配物理 KV cache；
8. 预热阶段：compile / warmup / CUDA graph capture，之后进入 step。
```

完整链路图：

```text
EngineArgs
  → create_engine_config()
  → VllmConfig.__post_init__()
  → LLMEngine.from_engine_args()
  → EngineCoreClient.make_client()
  → EngineCore.__init__()
  → Executor.__init__() / _init_executor()
  → WorkerWrapperBase.init_worker()
  → GPUWorker.init_device()
  → GPUModelRunner.__init__()
  → GPUWorker.load_model()
  → GPUModelRunner.load_model()
  → BaseModelLoader.load_model()
  → initialize_model() + load_weights()
  → EngineCore._initialize_kv_caches()
  → get_kv_cache_spec()
  → determine_available_memory() / profile_run()
  → get_kv_cache_configs()
  → initialize_from_config() / initialize_kv_cache()
  → compile_or_warm_up_model()
  → capture_model()
  → LLMEngine.step()
```

---

## 3. 为什么这个顺序不能乱

这个生命周期顺序不是随便组织的。

关键依赖是：

```text
ModelConfig 必须先于模型类解析；
ParallelConfig 必须先于 Executor / Worker 拓扑；
LoadConfig 必须先于 model loader 选择；
模型必须先加载，才能知道真实权重占用和 attention layers；
KV cache spec 必须来自真实模型结构；
显存 profile 必须在模型加载后进行；
KV cache block 数必须在 profile 后才能确定；
物理 KV cache 必须在真实 forward 前初始化；
warmup / compile / CUDA graph capture 必须在模型和 KV cache ready 后进行；
Scheduler 必须拿到 KV cache config 后才能调度逻辑 blocks。
```

如果顺序打乱，会出现典型问题：

```text
- 还没加载模型就 profile 显存：不知道模型权重实际占用；
- 还没拿 KV cache spec 就分配 cache：不知道每层需要什么 cache；
- 还没初始化 distributed 就构造模型：TP / PP rank 信息不完整；
- 还没分配 KV cache 就 capture：capture 的 forward 缺少真实 cache 张量；
- 还没确定 runner / convert 就解析模型类：可能拿错 generation / pooling class。
```

---

## 4. 第一阶段：用户参数进入 EngineArgs

`EngineArgs` 是用户参数进入 vLLM 配置系统的入口。

位置：`code/vllm/vllm/engine/arg_utils.py:417`

```python
class EngineArgs:
```

`EngineArgs.__post_init__()` 在：`code/vllm/vllm/engine/arg_utils.py:737`

它负责对用户输入做第一层整理，例如：

```text
- 处理 model / tokenizer / revision；
- 处理 quantization 相关参数；
- 加载插件；
- 规范化部分参数默认值；
- 为后续 create_engine_config() 准备字段。
```

这一阶段还不是运行时对象创建阶段。

可以理解为：

```text
EngineArgs = 用户参数的结构化容器。
```

---

## 5. 第二阶段：EngineArgs 创建 VllmConfig

核心入口是：`code/vllm/vllm/engine/arg_utils.py:1833`

```python
def create_engine_config(self, usage_context: UsageContext | None = None) -> VllmConfig:
```

它会创建一系列子配置。

典型顺序：

```text
ModelConfig
CacheConfig
TokenizerPoolConfig
LoadConfig
ParallelConfig
SchedulerConfig
DeviceConfig
SpeculativeConfig
ObservabilityConfig
CompilationConfig
StructuredOutputsConfig
KVTransferConfig
```

几个关键位置：

```text
ModelConfig      code/vllm/vllm/engine/arg_utils.py:1629
CacheConfig      code/vllm/vllm/engine/arg_utils.py:1890
ParallelConfig   code/vllm/vllm/engine/arg_utils.py:2094
SchedulerConfig  code/vllm/vllm/engine/arg_utils.py:2167
LoadConfig       code/vllm/vllm/engine/arg_utils.py:1719 / 2306
VllmConfig       code/vllm/vllm/engine/arg_utils.py:2355
```

这一阶段最重要的是：

```text
用户参数不再是一堆 CLI flag，
而是变成了 VllmConfig 聚合对象。
```

---

## 6. VllmConfig 是什么

`VllmConfig` 定义在：`code/vllm/vllm/config/vllm.py:288`

它是 vLLM 的总配置对象，聚合了启动和运行时需要的大部分子配置。

初始化后会进入：`code/vllm/vllm/config/vllm.py:917`

```python
def __post_init__(self):
```

然后调用：`code/vllm/vllm/config/vllm.py:926`

```python
self.try_verify_and_update_config()
```

这一步会做跨配置校验和更新。

它检查的不是单个 config 内部字段，而是多配置之间的兼容性，例如：

```text
- model / scheduler / cache 的约束是否一致；
- executor capability 是否支持当前设置；
- async scheduling 是否可用；
- 并行配置、设备配置、编译配置是否能同时成立；
- V1 / V0 或 backend 相关兼容性。
```

因此：

```text
VllmConfig = 启动前最终配置账本。
```

后续 EngineCore、Executor、Worker、ModelRunner 都会从同一个 `VllmConfig` 中取配置。

---

## 7. ModelConfig 在生命周期中的特殊地位

虽然 `VllmConfig` 是总配置，但 `ModelConfig` 是最早触发大量模型相关解析的子配置。

`ModelConfig.__post_init__()` 会：

```text
1. 读取 HF config；
2. 提取 hf_text_config；
3. 生成 ModelArchitectureConfig；
4. 读取 architectures；
5. 通过 ModelRegistry inspect 模型能力；
6. 决定 runner_type / convert_type；
7. 初始化 pooling / multimodal 相关配置；
8. 推导 dtype / max_model_len / attention 类型等。
```

关键位置：

```text
读取 HF config              code/vllm/vllm/config/model.py:555
保存 hf_config              code/vllm/vllm/config/model.py:565
保存 hf_text_config         code/vllm/vllm/config/model.py:568
生成 model_arch_config      code/vllm/vllm/config/model.py:569
读取 architectures          code/vllm/vllm/config/model.py:578
判断 generation / pooling   code/vllm/vllm/config/model.py:580
决定 runner_type            code/vllm/vllm/config/model.py:583
决定 convert_type           code/vllm/vllm/config/model.py:586
inspect model class         code/vllm/vllm/config/model.py:616
保存 _model_info/_architecture code/vllm/vllm/config/model.py:617
```

这说明：

```text
模型类的能力信息在模型真正实例化前就已经被检查过。
```

后续很多运行时配置都依赖 `_model_info`：

```text
pooling 默认策略；
multimodal_config 是否创建；
PP 是否支持；
attention 类型；
prefix caching / chunked prefill 是否支持。
```

---

## 8. 第三阶段：LLMEngine 创建 EngineCoreClient

V1 的外层 Engine 入口是：`code/vllm/vllm/v1/engine/llm_engine.py:161`

```python
@classmethod
def from_engine_args(cls, engine_args: EngineArgs, ...):
```

它会先创建配置：

```python
vllm_config = engine_args.create_engine_config(usage_context)
```

位置：`code/vllm/vllm/v1/engine/llm_engine.py:171`

然后选择 executor 类：

```python
executor_class = Executor.get_class(vllm_config)
```

位置：`code/vllm/vllm/v1/engine/llm_engine.py:172`

再创建 `EngineCoreClient`：

```python
self.engine_core = EngineCoreClient.make_client(...)
```

位置：`code/vllm/vllm/v1/engine/llm_engine.py:104`

这说明外层 `LLMEngine` 本身不是直接加载模型的地方。

它的职责更像：

```text
1. 持有 VllmConfig；
2. 创建 EngineCoreClient；
3. 接请求、加请求、拉输出；
4. 调用 EngineCore 的 step 结果。
```

---

## 9. EngineCoreClient 到 EngineCore

`EngineCoreClient.make_client()` 在：`code/vllm/vllm/v1/engine/core_client.py:83`

不同运行模式可能创建不同 client。

单进程 inproc 路径里：

```text
EngineCoreClient.make_client()
  → InprocClient
  → EngineCore(...)
```

`InprocClient` 定义在：`code/vllm/vllm/v1/engine/core_client.py:276`

它的初始化位置：`code/vllm/vllm/v1/engine/core_client.py:287`

```python
self.engine_core = EngineCore(...)
```

这一步开始进入真正的核心运行时对象创建。

---

## 10. 第四阶段：EngineCore 初始化

`EngineCore` 定义在：`code/vllm/vllm/v1/engine/core.py:96`

初始化入口：`code/vllm/vllm/v1/engine/core.py:99`

```python
class EngineCore:
    def __init__(...):
```

关键动作：

```python
self.model_executor = executor_class(vllm_config)
```

位置：`code/vllm/vllm/v1/engine/core.py:123`

然后初始化 KV caches：

```python
self._initialize_kv_caches(vllm_config)
```

位置：`code/vllm/vllm/v1/engine/core.py:133`

然后创建 scheduler：

```python
self.scheduler = Scheduler(...)
```

位置：`code/vllm/vllm/v1/engine/core.py:150`

这个顺序很关键：

```text
Executor / Worker / ModelRunner / model 先创建；
KV cache profiling 和初始化随后发生；
Scheduler 最后拿着 KV cache config 初始化。
```

因为 Scheduler 需要知道：

```text
KV cache blocks 有多少、每组 cache 怎么组织、能调度多少请求/token。
```

---

## 11. 第五阶段：Executor 选择和初始化

Executor 类选择入口：`code/vllm/vllm/v1/executor/abstract.py:48`

```python
Executor.get_class(vllm_config)
```

它根据 `parallel_config.distributed_executor_backend` 选择：

```text
uni
mp
ray
ray_v2
external_launcher
自定义 executor class
```

Executor 初始化入口：`code/vllm/vllm/v1/executor/abstract.py:95`

```python
def __init__(self, vllm_config: VllmConfig) -> None:
```

它会保存关键配置：

```text
model_config
cache_config
lora_config
load_config
parallel_config
scheduler_config
device_config
speculative_config
observability_config
```

然后调用具体后端的：

```python
self._init_executor()
```

位置：`code/vllm/vllm/v1/executor/abstract.py:109`

也就是说：

```text
Executor 抽象层保存配置，具体子类负责创建 Worker 和通信后端。
```

---

## 12. UniProcExecutor 示例：创建 Worker 并加载模型

以单进程路径为例。

入口：`code/vllm/vllm/v1/executor/uniproc_executor.py:46`

```python
def _init_executor(self) -> None:
```

关键步骤：

```text
1. 创建 driver_worker；
2. init_worker；
3. init_device；
4. load_model。
```

对应位置：

```text
init_worker   code/vllm/vllm/v1/executor/uniproc_executor.py:62
init_device   code/vllm/vllm/v1/executor/uniproc_executor.py:63
load_model    code/vllm/vllm/v1/executor/uniproc_executor.py:68
```

注意这里有一个非常重要的顺序：

```text
Worker 在 Executor 初始化期间就会 load_model。
```

也就是说，到 `EngineCore._initialize_kv_caches()` 执行显存 profile 时，模型权重已经在设备上了。

---

## 13. WorkerWrapperBase 如何创建真实 Worker

`WorkerWrapperBase.init_worker()` 位置：`code/vllm/vllm/v1/worker/worker_base.py:230`

它会解析：

```text
parallel_config.worker_cls
```

位置：`code/vllm/vllm/v1/worker/worker_base.py:251`

然后实例化真实 worker：

```text
GPUWorker / CPUWorker / 其它 platform worker
```

实例化位置：`code/vllm/vllm/v1/worker/worker_base.py:311`

所以 worker 类型不是写死在 executor 里，而是由配置和平台共同决定。

可以理解为：

```text
Executor 持有 WorkerWrapper；
WorkerWrapper 根据 worker_cls 创建真实 Worker；
真实 Worker 再持有 ModelRunner。
```

---

## 14. 第六阶段：GPUWorker 初始化 device

GPU worker 的 device 初始化入口：`code/vllm/vllm/v1/worker/gpu_worker.py:297`

```python
def init_device(self):
```

它做的关键事情：

```text
1. 设置 CUDA device；
2. 初始化 distributed environment；
3. 设置随机种子；
4. 清理和记录初始显存状态；
5. 构造 GPUModelRunner。
```

关键位置：

```text
设置 CUDA device              code/vllm/vllm/v1/worker/gpu_worker.py:361
初始化 distributed environment code/vllm/vllm/v1/worker/gpu_worker.py:369
记录初始显存快照              code/vllm/vllm/v1/worker/gpu_worker.py:388
构造 model runner             code/vllm/vllm/v1/worker/gpu_worker.py:408
默认 GPUModelRunnerV1         code/vllm/vllm/v1/worker/gpu_worker.py:416
```

这个阶段还没有加载模型权重。

它的目标是让当前 rank / device 准备好：

```text
device ready + distributed ready + model_runner object ready
```

---

## 15. GPUModelRunner 初始化

`GPUModelRunner` 定义在：`code/vllm/vllm/v1/worker/gpu_model_runner.py:445`

初始化入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:448`

```python
class GPUModelRunner(...):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
```

它会保存几乎所有执行相关配置：

```text
vllm_config
model_config
cache_config
offload_config
compilation_config
lora_config
load_config
parallel_config
scheduler_config
speculative_config
observability_config
```

同时初始化：

```text
- max_model_len / max_num_tokens / max_num_reqs；
- 多模态 registry 和能力标志；
- sampler；
- KV cache 相关空字段；
- attention group / metadata builder 相关字段；
- requests 状态表；
- InputBatch；
- persistent GPU buffers；
- CUDA graph / compile 相关状态。
```

但注意：

```text
GPUModelRunner.__init__() 创建的是执行容器，
真正模型权重还没加载，物理 KV cache 也还没分配。
```

---

## 16. 第七阶段：Worker.load_model 加载模型

GPUWorker 加载模型入口：`code/vllm/vllm/v1/worker/gpu_worker.py:424`

```python
def load_model(self) -> None:
```

它会调用：

```python
self.model_runner.load_model()
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:431`

ModelRunner 侧入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5231`

```python
def load_model(self, load_dummy_weights: bool = False) -> None:
```

核心逻辑：

```python
model_loader = get_model_loader(self.load_config)
self.model = model_loader.load_model(
    vllm_config=self.vllm_config, model_config=self.model_config
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5251` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5253`

之后还会处理：

```text
- LoRA model 包装；
- speculative drafter 加载；
- MoE / EPLB；
- communication buffer；
- torch.compile；
- CUDA graph wrapper；
- offloader post_init；
- 模型内存使用统计。
```

因此，模型加载不是单纯 `from_pretrained()`，而是 vLLM 的模型类解析、权重加载、包装和执行优化准备的集合。

---

## 17. model loader 内部发生什么

model loader 选择入口：`code/vllm/vllm/model_executor/model_loader/__init__.py:122`

```python
get_model_loader(load_config)
```

基础加载流程在：`code/vllm/vllm/model_executor/model_loader/base_loader.py:43`

```python
def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig) -> nn.Module:
```

典型流程：

```text
1. initialize_model() 创建空模型对象；
2. load_weights() 加载权重；
3. process_weights_after_loading() 做量化/attention 后处理；
4. 返回模型实例。
```

关键位置：

```text
initialize_model code/vllm/vllm/model_executor/model_loader/base_loader.py:55
load_weights      code/vllm/vllm/model_executor/model_loader/base_loader.py:64
process_weights_after_loading code/vllm/vllm/model_executor/model_loader/base_loader.py:80
```

`initialize_model()` 定义在：`code/vllm/vllm/model_executor/model_loader/utils.py:42`

它会先解析模型类：

```python
if model_class is None:
    model_class, _ = get_model_architecture(model_config)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:53`

然后实例化：

```python
model = model_class(vllm_config=vllm_config, prefix=prefix)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:63`

权重加载的默认实现入口在：`code/vllm/vllm/model_executor/model_loader/default_loader.py:415`

```python
DefaultModelLoader.load_weights
```

这条链路把前面 registry 解析到的模型类真正变成可执行的模型实例。

---

## 18. 第八阶段：EngineCore 初始化 KV caches

模型加载完成后，`EngineCore.__init__()` 会调用：

```python
self._initialize_kv_caches(vllm_config)
```

位置：`code/vllm/vllm/v1/engine/core.py:133`

方法定义：`code/vllm/vllm/v1/engine/core.py:240`

```python
def _initialize_kv_caches(self, vllm_config: VllmConfig) -> None:
```

这一步是启动过程中最关键的资源规划阶段。

主流程：

```text
1. 向 ModelRunner 查询 KV cache spec；
2. profile 可用显存；
3. 根据 spec 和可用显存生成 KVCacheConfig；
4. 创建 scheduler 侧 KV cache config；
5. 让 worker / model runner 分配物理 KV cache；
6. warmup / compile / capture。
```

---

## 19. 查询 KV cache spec

EngineCore 先调用：

```python
kv_cache_specs = self.model_executor.get_kv_cache_specs()
```

位置：`code/vllm/vllm/v1/engine/core.py:247`

Executor 抽象接口：`code/vllm/vllm/v1/executor/abstract.py:149`

```python
def get_kv_cache_specs(self):
```

Worker 侧入口：`code/vllm/vllm/v1/worker/gpu_worker.py:701`

```python
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
```

ModelRunner 侧真正实现：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7623`

```python
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
```

它会基于真实模型里的 attention layers 生成每层 KV cache 需求。

这一步必须在模型加载后发生，因为：

```text
只有模型实例存在后，才能遍历 attention layers，知道每个 layer 的 KV cache spec。
```

---

## 20. profile 可用显存

EngineCore 随后调用：

```python
available_gpu_memory = self.model_executor.determine_available_memory()
```

位置：`code/vllm/vllm/v1/engine/core.py:283`

Worker 侧入口：`code/vllm/vllm/v1/worker/gpu_worker.py:448`

```python
def determine_available_memory(self) -> int:
```

关键动作：

```text
1. 清理缓存；
2. 在 memory_profiling 上下文里跑 profile；
3. 调用 model_runner.profile_run()；
4. 根据峰值占用估算可用于 KV cache 的显存。
```

关键位置：

```text
memory_profiling       code/vllm/vllm/v1/worker/gpu_worker.py:484
model_runner.profile_run code/vllm/vllm/v1/worker/gpu_worker.py:488
GPUModelRunner.profile_run code/vllm/vllm/v1/worker/gpu_model_runner.py:6345
```

为什么要 profile？

```text
模型权重占用、激活峰值、临时 buffer、编译相关开销，
都不是只靠配置字段就能精确知道的。
```

所以 vLLM 会先实际跑一次 profile，再决定 KV cache 能分多少 blocks。

---

## 21. 生成 KVCacheConfig 和 scheduler cache config

profile 完成后，EngineCore 会生成 KV cache 配置。

关键位置：

```text
get_kv_cache_configs             code/vllm/vllm/v1/engine/core.py:294
generate_scheduler_kv_cache_config code/vllm/vllm/v1/engine/core.py:305
```

这一阶段会把：

```text
KV cache spec
可用 GPU memory
cache_config
parallel_config
model_config
```

合并成最终的 KV cache 规划。

这里有两个层面的配置：

```text
Worker / ModelRunner 侧：物理 KV cache 怎么分配；
Scheduler 侧：逻辑 block 怎么管理和调度。
```

两者来自同一个规划结果，但用途不同。

---

## 22. 分配物理 KV cache

EngineCore 调用：

```python
self.model_executor.initialize_from_config(kv_cache_configs)
```

位置：`code/vllm/vllm/v1/engine/core.py:321`

Executor 抽象实现：`code/vllm/vllm/v1/executor/abstract.py:118`

```python
def initialize_from_config(self, kv_cache_configs):
```

Worker 侧入口：`code/vllm/vllm/v1/worker/gpu_worker.py:717`

```python
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
```

它会写回：

```python
self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:722`

然后调用：

```python
self.model_runner.initialize_kv_cache(kv_cache_config)
```

位置：`code/vllm/vllm/v1/worker/gpu_worker.py:732`

ModelRunner 侧入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:7467`

```python
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig, ...):
```

它会：

```text
1. 保存 KVCacheConfig；
2. 初始化 attention backend；
3. 初始化 metadata builders；
4. 必要时重建 InputBatch；
5. 分配 KV cache tensors；
6. bind KV cache 到 attention layers；
7. 注册 KV transfer 可访问的 cache。
```

物理分配相关位置：

```text
initialize_kv_cache_tensors code/vllm/vllm/v1/worker/gpu_model_runner.py:7384
bind_kv_cache               code/vllm/vllm/v1/worker/gpu_model_runner.py:7431
initialize_kv_cache          code/vllm/vllm/v1/worker/gpu_model_runner.py:7467
```

这一步完成后，模型 forward 所需的 KV cache 才真正存在。

---

## 23. warmup / compile / CUDA graph capture

`Executor.initialize_from_config()` 不只是初始化 KV cache，还会继续 warmup / compile。

位置：`code/vllm/vllm/v1/executor/abstract.py:124`

它会通过 RPC 调用 worker：

```text
compile_or_warm_up_model
```

Worker 入口：`code/vllm/vllm/v1/worker/gpu_worker.py:746`

```python
def compile_or_warm_up_model(self) -> None:
```

关键动作：

```text
1. 根据 compilation_config 选择 compile sizes；
2. 对 compile sizes 做 dummy run；
3. kernel warmup；
4. capture CUDA graph；
5. sampler / pooler dummy warmup；
6. reset seed。
```

关键位置：

```text
_dummy_run              code/vllm/vllm/v1/worker/gpu_worker.py:775
kernel_warmup           code/vllm/vllm/v1/worker/gpu_worker.py:780
model_runner.capture_model code/vllm/vllm/v1/worker/gpu_worker.py:784
sampler/pooler warmup   code/vllm/vllm/v1/worker/gpu_worker.py:865
reset seed              code/vllm/vllm/v1/worker/gpu_worker.py:888
```

ModelRunner capture 入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6702`

```python
def capture_model(self) -> None:
```

实际 warmup / capture：`code/vllm/vllm/v1/worker/gpu_model_runner.py:6769`

```python
_warmup_and_capture
```

这一步结束后，运行时才进入：

```text
模型已加载；
KV cache 已分配；
attention backend ready；
编译 / graph capture ready；
sampler / pooler warmup ready。
```

也就是 ready 状态。

---

## 24. Scheduler 为什么在 KV cache 后创建

在 `EngineCore.__init__()` 中，Scheduler 创建发生在 KV cache 初始化之后。

位置：`code/vllm/vllm/v1/engine/core.py:150`

原因是 Scheduler 需要依赖：

```text
scheduler_config
model_config
cache_config
kv_cache_config
structured output manager
KV cache manager / block pool
```

更具体地说，Scheduler 要知道：

```text
- 总共有多少 KV blocks；
- 每个 request 最多能占多少 token；
- 每轮 max_num_batched_tokens；
- max_num_seqs；
- 是否启用 prefix caching；
- 是否有 encoder cache / multimodal cache；
- 是否有 external KV transfer。
```

所以 Scheduler 不是纯配置对象，它依赖初始化阶段产生的 runtime KV cache 规划。

---

## 25. 启动完成后的第一轮 step

外层 step 入口：`code/vllm/vllm/v1/engine/llm_engine.py:296`

```python
def step(self) -> list[RequestOutput]:
```

Inproc client 会调用 EngineCore：`code/vllm/vllm/v1/engine/core_client.py:289`

```text
InprocClient.get_output
```

EngineCore 主循环入口：`code/vllm/vllm/v1/engine/core.py:488`

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
```

主流程：

```python
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(...)
```

关键位置：

```text
scheduler.schedule        code/vllm/vllm/v1/engine/core.py:499
model_executor.execute_model code/vllm/vllm/v1/engine/core.py:500
sample_tokens fallback    code/vllm/vllm/v1/engine/core.py:507
scheduler.update_from_output code/vllm/vllm/v1/engine/core.py:513
```

这时启动阶段创建的对象开始进入运行时闭环：

```text
Scheduler 使用 KV cache manager 做调度；
Executor 把 SchedulerOutput 发给 Worker；
Worker 调用 ModelRunner；
ModelRunner 使用模型实例、InputBatch、attention backend、KV cache 执行 forward；
输出返回 Scheduler 更新状态。
```

---

## 26. execute_model 运行时链路

Executor 抽象入口：`code/vllm/vllm/v1/executor/abstract.py:221`

```python
self.collective_rpc("execute_model", args=(scheduler_output,), ...)
```

UniProc 路径：`code/vllm/vllm/v1/executor/uniproc_executor.py:108`

```python
def execute_model(...):
```

Worker 入口：`code/vllm/vllm/v1/worker/gpu_worker.py:1002`

```python
def execute_model(self, scheduler_output: SchedulerOutput, ...):
```

ModelRunner 入口：`code/vllm/vllm/v1/worker/gpu_model_runner.py:4097`

```python
def execute_model(self, scheduler_output, intermediate_tensors=None):
```

ModelRunner 内部关键阶段：

```text
_update_states()
_prepare_inputs()
_get_slot_mappings()
_build_attention_metadata()
_preprocess()
set_forward_context()
_model_forward()
compute_logits / pooling
暂存 ExecuteModelState 并返回 None
EngineCore 侧调用 sample_tokens(grammar_output)
ModelRunnerOutput
```

关键位置：

```text
_prepare_inputs          code/vllm/vllm/v1/worker/gpu_model_runner.py:4181
_build_attention_metadata code/vllm/vllm/v1/worker/gpu_model_runner.py:4308
set_forward_context      code/vllm/vllm/v1/worker/gpu_model_runner.py:4363
_model_forward 调用点    code/vllm/vllm/v1/worker/gpu_model_runner.py:4380
_model_forward 定义      code/vllm/vllm/v1/worker/gpu_model_runner.py:3810
sample_tokens            code/vllm/vllm/v1/worker/gpu_model_runner.py:4483
```

这说明启动阶段的所有准备最终都会汇聚到 `GPUModelRunner.execute_model()`。

---

## 27. 各子配置在生命周期中的使用位置

### 27.1 ModelConfig

主要影响：

```text
- HF config 加载；
- architecture / registry 解析；
- runner_type / convert_type；
- dtype / max_model_len；
- multimodal / pooling / attention 类型；
- 模型类实例化；
- ModelRunner forward 行为。
```

使用阶段：

```text
配置阶段、模型加载阶段、KV spec 阶段、运行时 forward 阶段。
```

### 27.2 LoadConfig

主要影响：

```text
- 选择 model loader；
- 权重格式；
- download / safetensors / pt / dummy weights 等加载路径。
```

主要使用阶段：

```text
GPUModelRunner.load_model()
  → get_model_loader(load_config)
  → model_loader.load_model()
```

运行时每轮 step 通常不再直接依赖 LoadConfig。

### 27.3 ParallelConfig

主要影响：

```text
- executor backend；
- worker_cls；
- TP / PP / DP / DCP；
- distributed groups；
- 每个 rank 的模型层范围；
- PP 是否需要 intermediate tensors。
```

使用阶段：

```text
Executor 创建、Worker 初始化 distributed、ModelRunner 初始化、模型 forward、KV cache 分布。
```

### 27.4 CacheConfig

主要影响：

```text
- block size；
- GPU memory utilization；
- num_gpu_blocks；
- prefix caching；
- mamba cache；
- KV cache dtype。
```

使用阶段：

```text
profile 前提供约束；
profile 后写回 num_gpu_blocks；
initialize_kv_cache 分配物理 cache；
runtime step 中由 Scheduler / ModelRunner 共同使用。
```

### 27.5 SchedulerConfig

主要影响：

```text
- max_num_batched_tokens；
- max_num_seqs；
- max_model_len 调度边界；
- chunked prefill；
- async scheduling；
- structured output 相关调度配合。
```

使用阶段：

```text
Scheduler 创建、ModelRunner InputBatch 大小、每轮 schedule。
```

### 27.6 CompilationConfig

主要影响：

```text
- compile sizes；
- CUDA graph capture sizes；
- cudagraph dispatch；
- torch.compile 行为；
- warmup / capture 策略。
```

使用阶段：

```text
GPUModelRunner 初始化、load_model wrapper、compile_or_warm_up_model、capture_model、runtime cudagraph replay。
```

---

## 28. 启动对象和运行时对象的边界

### 配置对象

```text
EngineArgs
VllmConfig
ModelConfig
CacheConfig
ParallelConfig
SchedulerConfig
LoadConfig
CompilationConfig
```

这些对象回答：

```text
应该怎么构建运行时。
```

### 运行时控制对象

```text
LLMEngine
EngineCoreClient
EngineCore
Executor
WorkerWrapper
GPUWorker
```

这些对象回答：

```text
谁负责调度、分发、设备生命周期和控制面。
```

### 执行对象

```text
GPUModelRunner
model instance
InputBatch
KV cache tensors
attention backend
Sampler / Pooler
CUDA graph wrappers
```

这些对象回答：

```text
一次 batch 如何真正跑进模型。
```

边界一句话：

```text
配置对象决定结构，控制对象组织生命周期，执行对象完成 forward / sampling。
```

---

## 29. 一个完整启动例子：单进程 GPU V1

以最常见的单进程 GPU V1 为例，顺序是：

```text
1. 用户传入 model、dtype、max_model_len、gpu_memory_utilization 等参数；
2. EngineArgs 保存用户参数；
3. create_engine_config() 创建 ModelConfig / CacheConfig / ParallelConfig / SchedulerConfig / LoadConfig；
4. ModelConfig 读取 HF config，解析 architectures，inspect 模型能力；
5. VllmConfig 做跨配置校验；
6. LLMEngine.from_engine_args() 选择 Executor class；
7. EngineCoreClient 创建 InprocClient；
8. InprocClient 创建 EngineCore；
9. EngineCore 创建 UniProcExecutor；
10. UniProcExecutor 创建 WorkerWrapper；
11. WorkerWrapper 创建 GPUWorker；
12. GPUWorker.init_device() 设置 device / distributed，并创建 GPUModelRunner；
13. GPUWorker.load_model() 调 GPUModelRunner.load_model()；
14. ModelRunner 选择 model loader，initialize_model() 解析 model class 并实例化；
15. model loader 加载权重；
16. EngineCore._initialize_kv_caches() 查询 KV spec；
17. Worker profile_run() 估算可用 KV cache 显存；
18. EngineCore 生成 KVCacheConfig；
19. Worker / ModelRunner initialize_kv_cache() 分配物理 KV cache；
20. Worker compile_or_warm_up_model() 做 dummy run、kernel warmup、CUDA graph capture；
21. Scheduler 创建完成，Engine 进入 ready；
22. 第一次 step 时 SchedulerOutput 被发给 ModelRunner 执行。
```

---

## 30. 容易混淆的点

### 30.1 VllmConfig 创建后模型是否已经加载？

没有。

`VllmConfig` 只是配置聚合和校验结果。

模型加载发生在：

```text
Executor._init_executor()
  → Worker.load_model()
  → GPUModelRunner.load_model()
```

### 30.2 ModelConfig inspect 了模型类，是否等于实例化模型？

不是。

`ModelConfig` 阶段只是通过 registry 获取 `_ModelInfo`。

真正实例化发生在：

```text
initialize_model()
```

### 30.3 Scheduler 是最早创建的吗？

不是。

在 V1 `EngineCore` 中，Scheduler 创建在 KV cache 初始化之后。

原因是 Scheduler 需要 KV cache block 规划。

### 30.4 为什么 load_model 在 determine_available_memory 前？

因为 profile 显存必须包含模型权重占用。

如果不先加载模型，profile 出来的可用显存会过高，KV cache block 数会不真实。

### 30.5 initialize_from_config 为什么既初始化 KV cache 又 warmup？

因为 `Executor.initialize_from_config()` 的语义是：

```text
让 Worker 从 KVCacheConfig 进入真正可执行状态。
```

这不仅包括分配 cache，也包括 compile / warmup / capture。

### 30.6 LoadConfig 是否影响每轮 step？

通常不直接影响。

`LoadConfig` 主要在启动加载阶段使用。

每轮 step 更多依赖：

```text
SchedulerConfig
CacheConfig
ModelConfig
ParallelConfig
CompilationConfig
```

---

## 31. 从“回答问题”的角度总结

如果要问：

```text
从配置到运行时对象的生命周期顺序是什么？
```

可以回答：

```text
vLLM 先用 EngineArgs 承接用户参数，再通过 create_engine_config() 构造 VllmConfig 和各子配置。
ModelConfig 会在这个阶段读取 HF config、解析 architecture、inspect 模型能力，并决定 runner / convert / multimodal / pooling 等关键属性。

随后 LLMEngine 创建 EngineCoreClient，EngineCore 创建 Executor。
Executor 创建 Worker，Worker 初始化 device 和 distributed，再创建 GPUModelRunner 并加载模型。
模型加载完成后，EngineCore 查询模型的 KV cache spec，调用 Worker profile 显存，生成 KVCacheConfig 和 scheduler KV cache config。
接着 Worker / ModelRunner 分配物理 KV cache，执行 warmup、compile、CUDA graph capture。
最后 Scheduler 和执行层进入 ready 状态，每轮 EngineCore.step() 由 Scheduler 生成 SchedulerOutput，经 Executor / Worker 送到 ModelRunner 执行；当前实现中 forward 可先返回 `None` 并由 EngineCore 再调用 `sample_tokens()` 取出 ModelRunnerOutput，随后更新 Scheduler 状态。
```

最短心智模型：

```text
配置构造 → 运行时对象创建 → 模型加载 → KV cache 规划和分配 → 编译预热 → step 闭环。
```

---

## 32. 最关键流程图

```text
用户参数 / CLI / API
  ↓
EngineArgs
  ↓ create_engine_config()
VllmConfig
  ├─ ModelConfig
  │    ├─ get_config()
  │    ├─ architectures
  │    ├─ ModelRegistry.inspect_model_cls()
  │    ├─ runner_type / convert_type
  │    └─ _model_info / _architecture
  ├─ CacheConfig
  ├─ ParallelConfig
  ├─ SchedulerConfig
  ├─ LoadConfig
  └─ CompilationConfig
  ↓
LLMEngine.from_engine_args()
  ↓
EngineCoreClient.make_client()
  ↓
EngineCore.__init__()
  ├─ Executor.get_class()
  ├─ executor_class(vllm_config)
  │    ↓
  │  Executor.__init__()
  │    ↓ _init_executor()
  │  WorkerWrapperBase.init_worker()
  │    ↓
  │  GPUWorker
  │    ├─ init_device()
  │    │    ├─ set device
  │    │    ├─ init distributed
  │    │    └─ GPUModelRunner.__init__()
  │    └─ load_model()
  │         ↓
  │       GPUModelRunner.load_model()
  │         ↓
  │       get_model_loader(load_config)
  │         ↓
  │       BaseModelLoader.load_model()
  │         ├─ initialize_model()
  │         │    ├─ get_model_architecture()
  │         │    └─ model_class(vllm_config, prefix)
  │         └─ load_weights()
  │
  ├─ _initialize_kv_caches()
  │    ├─ get_kv_cache_specs()
  │    │    └─ GPUModelRunner.get_kv_cache_spec()
  │    ├─ determine_available_memory()
  │    │    └─ GPUModelRunner.profile_run()
  │    ├─ get_kv_cache_configs()
  │    ├─ generate_scheduler_kv_cache_config()
  │    └─ model_executor.initialize_from_config()
  │         ├─ GPUWorker.initialize_from_config()
  │         ├─ GPUModelRunner.initialize_kv_cache()
  │         │    ├─ initialize attention backend
  │         │    ├─ allocate KV cache tensors
  │         │    └─ bind KV cache
  │         └─ GPUWorker.compile_or_warm_up_model()
  │              ├─ dummy run
  │              ├─ kernel warmup
  │              └─ GPUModelRunner.capture_model()
  │
  └─ Scheduler(...)
       ↓
ready
       ↓
LLMEngine.step()
  ↓
EngineCore.step()
  ├─ scheduler.schedule()
  ├─ model_executor.execute_model()
  │    ↓
  │  Worker.execute_model()
  │    ↓
  │  GPUModelRunner.execute_model()
  │    ├─ _update_states()
  │    ├─ _prepare_inputs()
  │    ├─ _build_attention_metadata()
  │    ├─ _model_forward()
  │    └─ 暂存 ExecuteModelState / 返回 None
  ├─ model_executor.sample_tokens()
  └─ scheduler.update_from_output()
```

---

## 33. 最小心智模型

只记一条线：

```text
EngineArgs 把用户输入变成 VllmConfig；
VllmConfig 决定运行时对象怎么创建；
Executor / Worker / ModelRunner 把配置变成真实 device 上的模型实例；
EngineCore 用模型 profile 出 KV cache 规模并完成分配；
warmup / compile / capture 之后，EngineCore.step() 才进入稳定执行闭环。
```

再压缩成一句：

```text
vLLM 的启动不是“加载模型然后跑”，而是“配置解析 → 对象拓扑创建 → 模型加载 → 资源规划 → 执行优化 → 调度闭环”。
```
