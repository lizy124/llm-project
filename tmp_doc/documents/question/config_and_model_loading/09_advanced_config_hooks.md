# 09. 高级能力如何挂到配置和模型加载？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\config\vllm.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\lora.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\multimodal.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\speculative.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\compilation.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\config\kv_transfer.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\engine\arg_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\lora\worker_manager.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\lora_model_runner_mixin.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\kv_connector_model_runner_mixin.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu\model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\spec_decode\`
- `D:\lzy\project\kv_pool\code\vllm\vllm\compilation\`

本问题关注：LoRA、多模态、Speculative Decoding、Compilation / CUDA graph、KV Transfer 等高级能力，如何从用户参数进入对应 Config，再被 `VllmConfig` 汇总、校验、参与 hash，最后在模型实例化、模型加载、batch 输入准备、forward 上下文、采样和 KV cache 生命周期里挂接。

---

## 1. 一句话回答

vLLM 的高级能力通常不是直接散落在模型代码里，而是先被收敛到对应子配置，再挂到 `VllmConfig`，最后由 Worker / ModelRunner 在固定阶段读取这些配置并安装 hook。

主链路是：

```text
用户参数 / Python API
  → LoRAConfig / MultiModalConfig / SpeculativeConfig / CompilationConfig / KVTransferConfig
  → VllmConfig
  → VllmConfig.__post_init__ 校验、补默认值、构造派生配置
  → VllmConfig.compute_hash() 影响编译 / 图缓存 key
  → Worker / GPUModelRunner 初始化
  → 模型加载阶段 hook
  → 输入准备阶段 hook
  → forward / sampling / KV cache 阶段 hook
```

所以可以把高级能力分成三类：

```text
结构型 hook：改变模型结构或包装模型，例如 LoRA、spec draft model、compilation wrapper；
输入型 hook：改变输入构造，例如 multimodal embeddings、LoRA mapping、spec decode tokens；
运行时 hook：改变 forward 周边协议，例如 CUDA graph、KV transfer、rejection sampling、encoder cache transfer。
```

---

## 2. 高级配置在 VllmConfig 中的位置

`VllmConfig` 统一持有这些子配置。

相关字段：

```python
lora_config: LoRAConfig | None = None
speculative_config: SpeculativeConfig | None = None
compilation_config: CompilationConfig = Field(default_factory=CompilationConfig)
kv_transfer_config: KVTransferConfig | None = None
```

位置：`vllm.py:326` 到 `vllm.py:355`

多模态配置不直接挂在 `VllmConfig` 顶层，而是在 `ModelConfig` 内部：

```text
vllm_config.model_config.multimodal_config
```

典型读取位置：`gpu_model_runner.py:5254`

这意味着：

```text
LoRA / speculative / compilation / KV transfer 是运行时全局能力；
Multimodal 更偏模型能力，绑定在 ModelConfig 上。
```

---

## 3. 高级配置如何参与 hash

`VllmConfig.compute_hash()` 会收集各子配置的 `compute_hash()`。

关键片段：

```python
if self.lora_config:
    vllm_factors.append(self.lora_config.compute_hash())
...
if self.speculative_config:
    vllm_factors.append(self.speculative_config.compute_hash())
...
if self.compilation_config:
    vllm_factors.append(self.compilation_config.compute_hash())
...
if self.kv_transfer_config:
    vllm_factors.append(self.kv_transfer_config.compute_hash())
```

位置：`vllm.py:450` 到 `vllm.py:478`

多模态有一个特殊点：只有在编译 multimodal encoder 时，`MultiModalConfig.compute_hash()` 才会进入 `VllmConfig` hash。

```python
if (
    self.compilation_config
    and getattr(self.compilation_config, "compile_mm_encoder", False)
    and self.model_config.multimodal_config
):
    vllm_factors.append(self.model_config.multimodal_config.compute_hash())
```

位置：`vllm.py:413` 到 `vllm.py:419`

含义：

```text
影响计算图结构或编译产物的高级配置，必须进入 hash；
只影响 IO、调度、connector metadata 的配置，可以不影响计算图 hash。
```

例如 `KVTransferConfig.compute_hash()` 明确没有 factors：

```python
# this config will not affect the computation graph.
factors: list[Any] = []
```

位置：`kv_transfer.py:74` 到 `kv_transfer.py:90`

---

## 4. 高级能力挂接的阶段图

按执行时间线看，高级能力大致挂在这些阶段：

```text
1. 参数解析阶段
   → arg_utils 创建各子 Config

2. VllmConfig post_init 阶段
   → 校验 LoRA dtype
   → 构造 speculative draft ModelConfig / ParallelConfig
   → 初始化 compilation backend / pass / cudagraph sizes
   → 由 cache offloading 派生 KVTransferConfig

3. ModelRunner 初始化阶段
   → speculative_config 创建 drafter / proposer / rejection_sampler
   → LoRA / KV connector mixin 准备能力入口
   → multimodal cache / encoder cache / flags 初始化

4. 模型加载阶段
   → base model load_model()
   → LoRA 包装模型 layer
   → drafter.load_model()
   → compilation / CUDA graph wrapper 包装模型
   → multimodal pruning / encoder cudagraph 能力探测

5. execute_model 输入准备阶段
   → LoRA set_active_loras()
   → multimodal encoder 执行并合并 embeddings
   → spec decode 准备 draft tokens / logits indices
   → KV connector 处理 preemption / no-forward path

6. forward / sampling 阶段
   → set_forward_context 带入 attention metadata / compilation runtime mode
   → KV connector start_load_kv / wait_for_save / output metadata
   → spec rejection sampling
   → LoRA active adapters 影响 LoRA layer forward
```

---

## 5. LoRAConfig：如何挂到模型加载和 batch 执行

源码：`lora.py`、`lora_model_runner_mixin.py`、`worker_manager.py`

### 5.1 LoRAConfig 描述什么

`LoRAConfig` 定义在：`lora.py:30` 到 `lora.py:132`

核心字段包括：

```text
max_lora_rank：最大 LoRA rank；
max_loras：单个 batch 最多 active LoRA 数；
fully_sharded_loras：是否全分片 LoRA；
max_cpu_loras：CPU 侧缓存 adapter 数；
lora_dtype：LoRA 权重 dtype；
target_modules：限制哪些模块挂 LoRA；
default_mm_loras：多模态场景按 modality 自动启用 LoRA；
enable_tower_connector_lora：给多模态 tower / connector 启用 LoRA；
specialize_active_lora：按 active LoRA 数 specialization CUDA graph；
enable_mixed_moe_lora_format：MoE LoRA 格式兼容开关。
```

位置：`lora.py:34` 到 `lora.py:79`

### 5.2 LoRAConfig 的校验

`max_cpu_loras` 默认等于 `max_loras`，并且不能小于 `max_loras`。

```python
if self.max_cpu_loras is None:
    self.max_cpu_loras = self.max_loras
elif self.max_cpu_loras < self.max_loras:
    raise ValueError(...)
```

位置：`lora.py:108` 到 `lora.py:116`

`lora_dtype` 会在 `VllmConfig.__post_init__` 中结合 `ModelConfig` 确定：

```python
if self.lora_config is not None:
    self.lora_config.verify_with_model_config(self.model_config)
```

位置：`vllm.py:905` 到 `vllm.py:906`

具体逻辑：

```python
if self.lora_dtype in (None, "auto"):
    self.lora_dtype = model_config.dtype
elif isinstance(self.lora_dtype, str):
    self.lora_dtype = getattr(torch, self.lora_dtype)
```

位置：`lora.py:127` 到 `lora.py:131`

### 5.3 LoRA 如何影响 hash

`LoRAConfig.compute_hash()` 包含：

```text
max_lora_rank
max_loras
fully_sharded_loras
lora_dtype
enable_tower_connector_lora
enable_mixed_moe_lora_format
target_modules
```

位置：`lora.py:81` 到 `lora.py:106`

这说明这些字段会影响计算图或模型 wrapper 结构。

例如：

```text
target_modules 改变哪些 layer 被 LoRA 替换；
max_loras / max_lora_rank 改变 LoRA buffer 和 kernel 形态；
enable_tower_connector_lora 改变多模态 encoder / connector 的 LoRA hook。
```

### 5.4 模型加载阶段：LoRA 包装模型

GPUModelRunner 加载 base model 后，如果 `self.lora_config` 存在，就调用：

```python
self.model = self.load_lora_model(
    self.model, self.vllm_config, self.device
)
```

位置：`gpu_model_runner.py:5167` 到 `gpu_model_runner.py:5170`

`load_lora_model()` 来自 `LoRAModelRunnerMixin`：

```python
if not supports_lora(model):
    raise ValueError(...)

self.lora_manager = LRUCacheWorkerLoRAManager(
    vllm_config,
    device,
    model.embedding_modules,
)
return self.lora_manager.create_lora_manager(model, vllm_config)
```

位置：`lora_model_runner_mixin.py:31` 到 `lora_model_runner_mixin.py:46`

这里发生了两件事：

```text
1. 校验模型是否 supports_lora；
2. 创建 WorkerLoRAManager，并把模型中可 LoRA 化的层替换 / 包装成 LoRA layer。
```

### 5.5 WorkerLoRAManager 负责 adapter 生命周期

`WorkerLoRAManager` 定义在：`worker_manager.py:25`

它保存：

```python
self.lora_config = vllm_config.lora_config
self.vocab_size = vllm_config.model_config.get_vocab_size()
```

位置：`worker_manager.py:46` 到 `worker_manager.py:48`

创建 LoRA manager：

```python
lora_manager = create_lora_manager(
    model,
    max_num_seqs=self.max_num_seqs,
    max_num_batched_tokens=self.max_num_batched_tokens,
    vocab_size=self.vocab_size,
    lora_config=self.lora_config,
    device=self.device,
    lora_manager_cls=self._manager_cls,
    vllm_config=vllm_config,
)
self._adapter_manager = lora_manager
return lora_manager.model
```

位置：`worker_manager.py:81` 到 `worker_manager.py:97`

加载 adapter 时会读 PEFT 配置并校验 LoRA config：

```python
peft_helper.validate_legal(self.lora_config)
```

位置：`worker_manager.py:120` 到 `worker_manager.py:122`

### 5.6 执行阶段：每个 batch 设置 active LoRA

在 `_prepare_inputs()` 末尾，如果 LoRA 开启：

```python
if self.lora_config:
    self.set_active_loras(
        self.input_batch, num_scheduled_tokens, num_sampled_tokens
    )
```

位置：`gpu_model_runner.py:2193` 到 `gpu_model_runner.py:2201`

`set_active_loras()` 会从 `InputBatch` 中生成：

```text
prompt_lora_mapping
token_lora_mapping
lora_requests
```

再调用：

```python
self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
```

位置：`lora_model_runner_mixin.py:73` 到 `lora_model_runner_mixin.py:91`

因此 LoRA 的执行 hook 是：

```text
请求携带 LoRARequest
  → InputBatch 维护 request_lora_mapping
  → _prepare_inputs() 生成 token/prompt LoRA mapping
  → lora_manager.set_active_adapters()
  → LoRA layer forward 使用当前 active adapter slots
```

### 5.7 多模态 tower / connector LoRA

如果开启 `enable_tower_connector_lora` 且 manager 支持，`_execute_mm_encoder()` 会为 encoder 输入单独构造 tower mapping：

```python
if self.lora_config and self.lora_manager.supports_tower_connector_lora():
    ...
    tower_mapping = LoRAMapping(..., type=LoRAMappingType.TOWER)
    self.lora_manager.set_active_adapters(lora_requests, tower_mapping)
```

位置：`gpu_model_runner.py:2941` 到 `gpu_model_runner.py:2974`

如果模型还有 connector，会继续设置 connector mapping：

```python
connector_mapping = LoRAMapping(..., type=LoRAMappingType.CONNECTOR)
self.lora_manager.set_active_adapters(lora_requests, connector_mapping)
```

位置：`gpu_model_runner.py:2975` 到 `gpu_model_runner.py:3008`

所以 LoRA 在多模态里可能有三套 mapping：

```text
LANGUAGE：语言模型 token；
TOWER：视觉 / 音频 encoder token；
CONNECTOR：encoder 到 language model 的 projector / connector token。
```

---

## 6. MultiModalConfig：如何挂到输入预处理和模型 forward

源码：`multimodal.py`、`gpu_model_runner.py`

### 6.1 MultiModalConfig 描述什么

`MultiModalConfig` 定义在：`multimodal.py:73` 到 `multimodal.py:320`

核心字段包括：

```text
language_model_only：禁用多模态输入；
limit_per_prompt：每个 prompt 允许多少 image/video/audio；
enable_mm_embeds：允许直接传预计算多模态 embeddings；
media_io_kwargs：媒体读取参数；
mm_processor_kwargs：processor 参数覆盖；
mm_processor_cache_gb / type：processor cache；
mm_encoder_only：只跑 encoder，不跑 language model；
mm_encoder_tp_mode：encoder 用 TP 切权重还是切数据；
mm_encoder_attn_backend：encoder attention backend；
mm_encoder_attn_dtype / fp8 scale：ViT encoder FP8 attention；
interleave_mm_strings：string chat template 下交错多模态；
skip_mm_profiling：跳过 MM profiling；
video_pruning_rate：视频 token pruning；
mm_tensor_ipc：多进程间 MM tensor 传输方式。
```

位置：`multimodal.py:77` 到 `multimodal.py:199`

### 6.2 limit_per_prompt 的格式归一化

`limit_per_prompt` 支持老格式：

```text
{"image": 16, "video": 2}
```

也支持带参数格式：

```text
{"video": {"count": 1, "num_frames": 32, "width": 512, "height": 512}}
```

validator 会把 int 转成 dummy options：

```python
if isinstance(v, int):
    v = {"count": v}
```

位置：`multimodal.py:201` 到 `multimodal.py:224`

### 6.3 多模态配置如何影响 hash

`MultiModalConfig.compute_hash()` 只包含会影响 encoder 计算图的字段：

```text
mm_encoder_attn_backend
mm_encoder_tp_mode
mm_encoder_attn_dtype
mm_encoder_fp8_scale_path
```

位置：`multimodal.py:287` 到 `multimodal.py:308`

并且只有 `compile_mm_encoder=True` 时，才进入 `VllmConfig.compute_hash()`。

位置：`vllm.py:413` 到 `vllm.py:419`

这说明：

```text
普通多模态输入限制、processor cache、media IO 参数影响输入处理；
但不一定改变模型计算图。
```

### 6.4 模型加载后：探测多模态能力

GPUModelRunner 加载模型后，会读取：

```python
mm_config = self.model_config.multimodal_config
self.is_multimodal_pruning_enabled = (
    supports_multimodal_pruning(self.get_model())
    and mm_config is not None
    and mm_config.is_multimodal_pruning_enabled()
)
self.requires_sequential_video_encoding = hasattr(
    self.get_model(), "requires_sequential_video_encoding"
)
```

位置：`gpu_model_runner.py:5254` 到 `gpu_model_runner.py:5262`

这里不是加载权重，而是在模型加载完成后根据模型接口和配置决定执行路径。

### 6.5 执行阶段：先跑 encoder，再合并 embeddings

多模态输入在 `_preprocess()` 中进入模型 forward 前处理。

关键代码：

```python
if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
    with self.maybe_get_ec_connector_output(...) as ec_connector_output:
        self._execute_mm_encoder(scheduler_output)
        mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

    inputs_embeds_scheduled = self.model.embed_input_ids(
        self.input_ids.gpu[:num_scheduled_tokens],
        multimodal_embeddings=mm_embeds,
        is_multimodal=is_mm_embed,
    )
```

位置：`gpu_model_runner.py:3447` 到 `gpu_model_runner.py:3493`

主链路是：

```text
SchedulerOutput.scheduled_encoder_inputs
  → _batch_mm_inputs_from_scheduler()
  → _execute_mm_encoder()
  → model.embed_multimodal(**mm_kwargs_batch)
  → encoder_cache[mm_hash] = output
  → _gather_mm_embeddings()
  → model.embed_input_ids(..., multimodal_embeddings, is_multimodal)
  → inputs_embeds 进入模型 forward
```

### 6.6 prompt_embeds 是特殊 passthrough modality

如果 modality 是 `prompt_embeds`，不会跑 encoder，而是直接放入 encoder cache：

```python
pe_tensor = mm_kwargs[i][1]["embedding"].data
self.encoder_cache[mm_hashes[i]] = pe_tensor.to(self.device)
```

位置：`gpu_model_runner.py:2899` 到 `gpu_model_runner.py:2924`

这对应 `enable_mm_embeds` / prompt embeddings 类输入：

```text
用户已经提供 embedding；
不需要 image/audio/video encoder；
但仍复用 encoder_cache 和 embedding splice 逻辑。
```

### 6.7 encoder-decoder 模型的多模态路径

如果是 encoder-decoder 且有 encoder inputs：

```python
encoder_outputs = self._execute_mm_encoder(scheduler_output)
model_kwargs.update({"encoder_outputs": encoder_outputs})
```

位置：`gpu_model_runner.py:3552` 到 `gpu_model_runner.py:3559`

这和 VLM prompt replacement 不同：

```text
普通多模态 decoder-only：encoder output 被 splice 成 inputs_embeds；
encoder-decoder：encoder output 作为 encoder_outputs 传给 decoder。
```

---

## 7. SpeculativeConfig：如何挂到 draft model 和采样

源码：`speculative.py`、`gpu_model_runner.py`、`v1/spec_decode/`

### 7.1 SpeculativeConfig 描述什么

`SpeculativeConfig` 定义在：`speculative.py:74`

主要字段：

```text
num_speculative_tokens：每轮草稿 token 数；
model：draft model / eagle head / proposer 资源；
method：ngram、draft_model、eagle、eagle3、mtp、suffix、custom_class 等；
draft_tensor_parallel_size：draft model TP；
quantization / moe_backend / attention_backend：draft model 专用配置；
max_model_len / revision / code_revision：draft model 模型配置；
prompt_lookup_min/max：ngram proposer 参数；
parallel_drafting：并行草稿；
draft_model_config / draft_parallel_config：post_init 生成的派生配置；
draft_load_config：draft model 的加载配置；
rejection_sample_method / draft_sample_method：验证采样策略。
```

位置：`speculative.py:78` 到 `speculative.py:271`

### 7.2 post_init 推断 method

`__post_init__()` 会根据 `model` 和 `method` 推断 speculative 方法。

```python
if self.model is not None and "." in self.model ...:
    self.method = "custom_class"
elif self.method is None:
    if self.model in ("ngram", "[ngram]"):
        self.method = "ngram"
    else:
        self.method = "draft_model"
```

位置：`speculative.py:564` 到 `speculative.py:588`

旧的 MTP method 名会归一成 `mtp`：

```python
if self.method in get_args(MTPModelTypes) and self.method != "mtp":
    self.method = "mtp"
```

位置：`speculative.py:589` 到 `speculative.py:593`

### 7.3 ngram / suffix / custom_class 不一定需要 draft model

ngram 路径会设置默认 lookup window，并把 draft config 指向 target config：

```python
self.draft_model_config = self.target_model_config
self.draft_parallel_config = self.target_parallel_config
```

位置：`speculative.py:633` 到 `speculative.py:665`

custom class 也不创建独立 draft model：

```python
self.prompt_lookup_max = 0
self.prompt_lookup_min = 0
self.draft_model_config = self.target_model_config
self.draft_parallel_config = self.target_parallel_config
```

位置：`speculative.py:668` 到 `speculative.py:679`

suffix 会走 `_validate_suffix_decoding()`，并依赖 Arctic Inference。

位置：`speculative.py:666` 到 `speculative.py:667`，`speculative.py:850` 到 `speculative.py:884`

### 7.4 draft model 路径会构造新的 ModelConfig

如果 `self.model` 存在，post_init 会创建 draft `ModelConfig`：

```python
self.draft_model_config = ModelConfig(
    model=self.model,
    runner="draft",
    tokenizer=self.target_model_config.tokenizer,
    ...
    hf_overrides=SpeculativeConfig.hf_config_override,
)
```

位置：`speculative.py:712` 到 `speculative.py:733`

然后自动识别 method：

```text
model name 包含 eagle- → eagle；
model name 包含 eagle3 → eagle3；
model name 包含 dflash → dflash；
hf_config.model_type == medusa → medusa；
hf_config.model_type == mlp_speculator → mlp_speculator；
hf_config.model_type 属于 MTPModelTypes → mtp；
否则如果 method=draft_model → 普通 draft model。
```

位置：`speculative.py:735` 到 `speculative.py:771`

### 7.5 hf_config_override 是模型结构 hook

`SpeculativeConfig.hf_config_override()` 会把某些 target / draft HF config 改写成 MTP / EAGLE 等 vLLM 可识别架构。

例如：

```python
if hf_config.model_type == "deepseek_mtp":
    hf_config.update(
        {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
    )
```

位置：`speculative.py:310` 到 `speculative.py:562`

这属于配置层直接影响模型类解析的 hook：

```text
原始 HF config
  → hf_config_override
  → architectures/model_type 被改写
  → ModelRegistry 解析到 draft / MTP / EAGLE 模型类
```

### 7.6 ModelRunner 初始化阶段创建 drafter

`GPUModelRunner.__init__()` 中，如果存在 `speculative_config` 且当前是最后一个 PP rank，会创建 drafter。

```python
if self.speculative_config and get_pp_group().is_last_rank:
    if self.speculative_config.method == "custom_class":
        self.drafter = create_custom_proposer(...)
    elif self.speculative_config.method == "ngram":
        self.drafter = NgramProposer(...)
    elif self.speculative_config.uses_draft_model():
        self.drafter = DraftModelProposer(...)
    elif self.speculative_config.use_ngram_gpu():
        self.drafter = NgramProposerGPU(...)
    ...
    self.rejection_sampler = RejectionSampler(...)
```

位置：`gpu_model_runner.py:545` 到 `gpu_model_runner.py:620`

含义：

```text
SpeculativeConfig 不只是采样参数；
它决定要创建哪种 proposer / drafter，并决定是否需要 rejection sampler。
```

### 7.7 模型加载阶段加载 drafter

base model 加载完、LoRA 包装后，如果有 drafter：

```python
if hasattr(self, "drafter"):
    logger.info_once("Loading drafter model...")
    if hasattr(self.drafter, "load_model"):
        self.drafter.load_model(self.model)
```

位置：`gpu_model_runner.py:5171` 到 `gpu_model_runner.py:5174`

所以 draft model 的加载不是通过主 `LoadConfig` 的普通主模型路径完成，而是在 target model 加载后由 proposer 自己挂接。

如果 `SpeculativeConfig.draft_load_config` 存在，draft proposer 可以使用 draft 专用加载配置。

位置：`speculative.py:196` 到 `speculative.py:198`

### 7.8 EAGLE3 / DFlash 会改变 target model 输出

`SpeculativeConfig.compute_hash()` 关注：

```python
uses_aux_hidden_states = self.method in (
    "eagle3",
    "extract_hidden_states",
    "dflash",
)
factors.append(uses_aux_hidden_states)
```

位置：`speculative.py:273` 到 `speculative.py:307`

如果需要 aux hidden states，模型加载后会设置输出层：

```python
self._setup_eagle3_aux_hidden_state_outputs()
```

位置：`gpu_model_runner.py:5200`

具体设置：

```python
if not supports_eagle3(self.get_model()):
    raise RuntimeError(...)
aux_layers = self._get_eagle3_aux_layers_from_config()
...
self.model.set_aux_hidden_state_layers(aux_layers)
```

位置：`gpu_model_runner.py:5320` 到 `gpu_model_runner.py:5339`

这说明某些 speculative 方法会改变 target model forward 输出内容，所以必须进入 hash。

### 7.9 执行阶段：准备 spec decode metadata

`_prepare_inputs()` 中会根据 draft tokens 计算 spec decode metadata：

```python
spec_decode_metadata = self._calc_spec_decode_metadata(
    num_draft_tokens, cu_num_tokens
)
logits_indices = spec_decode_metadata.logits_indices
num_sampled_tokens = num_draft_tokens + 1
```

位置：`gpu_model_runner.py:2183` 到 `gpu_model_runner.py:2188`

这会影响：

```text
本轮 forward 要验证多少 target tokens；
logits 应该从哪些 hidden states 上取；
采样阶段如何做 accepted / rejected token 处理。
```

---

## 8. CompilationConfig：如何挂到 torch.compile 和 CUDA graph

源码：`compilation.py`、`gpu_model_runner.py`、`compilation/`

### 8.1 CompilationConfig 描述什么

`CompilationConfig` 定义在：`compilation.py:378`

它分三类配置：

```text
Top-level compilation：mode、backend、custom_ops、splitting_ops、compile_mm_encoder；
CUDA graph：cudagraph_mode、capture_sizes、warmups、copy_inputs、full/piecewise 模式；
Inductor：compile_sizes、compile_ranges_endpoints、inductor_compile_config、inductor_passes、PassConfig。
```

位置：`compilation.py:386` 到 `compilation.py:425`

### 8.2 compilation mode

`CompilationMode` 包括：

```text
NONE = 0：纯 eager；
STOCK_TORCH_COMPILE = 1：标准 torch.compile；
DYNAMO_TRACE_ONCE = 2：单次 Dynamo trace；
VLLM_COMPILE = 3：vLLM 自定义 Inductor backend。
```

位置：`compilation.py:36` 到 `compilation.py:50`

### 8.3 cudagraph mode

`CUDAGraphMode` 包括：

```text
NONE
PIECEWISE
FULL
FULL_DECODE_ONLY
FULL_AND_PIECEWISE
```

位置：`compilation.py:53` 到 `compilation.py:104`

关键方法：

```text
has_full_cudagraphs()
has_piecewise_cudagraphs()
requires_piecewise_compilation()
decode_mode()
mixed_mode()
```

位置：`compilation.py:65` 到 `compilation.py:97`

这些方法会被 ModelRunner 用来决定 wrapper 和 runtime mode。

### 8.4 __post_init__ 初始化 backend / passes / custom ops

`CompilationConfig.__post_init__()` 做很多归一化。

例如：

```python
if KEY not in self.inductor_compile_config:
    self.inductor_compile_config[KEY] = False
```

位置：`compilation.py:887` 到 `compilation.py:903`

自定义 Inductor passes 会从 qualified name 解析成 callable：

```python
for k, v in self.inductor_passes.items():
    ...
    func = __import__(module).__dict__[func_name]
    self.inductor_compile_config[k] = CallableInductorPass(func)
```

位置：`compilation.py:923` 到 `compilation.py:938`

某些 fusion 会自动要求 custom op：

```python
if self.pass_config.enable_qk_norm_rope_fusion and "+rotary_embedding" not in self.custom_ops:
    self.custom_ops.append("+rotary_embedding")
```

位置：`compilation.py:940` 到 `compilation.py:953`

默认 backend：

```python
if self.backend == "":
    self.backend = current_platform.get_compile_backend()
```

位置：`compilation.py:1023` 到 `compilation.py:1024`

### 8.5 init_backend 返回 torch.compile backend

`init_backend()` 会根据 mode 返回不同 backend。

```python
if self.mode in [STOCK_TORCH_COMPILE, DYNAMO_TRACE_ONCE]:
    if self.backend in torch_backends:
        return self.backend
    return resolve_obj_by_qualname(self.backend)

assert self.mode == VLLM_COMPILE
return VllmBackend(vllm_config, prefix=prefix, is_encoder=is_encoder)
```

位置：`compilation.py:1026` 到 `compilation.py:1068`

这就是配置连接到 `torch.compile` / vLLM backend 的入口。

### 8.6 cudagraph sizes 后处理

`post_init_cudagraph_sizes()` 会展开 `compile_sizes`。

```python
if isinstance(x, str):
    assert x == "cudagraph_capture_sizes"
    computed_compile_sizes.extend(self.cudagraph_capture_sizes)
else:
    computed_compile_sizes.append(x)
self.compile_sizes = computed_compile_sizes
self.cudagraph_capture_sizes.sort()
```

位置：`compilation.py:1070` 到 `compilation.py:1095`

因此：

```text
compile_sizes 可以引用 cudagraph_capture_sizes；
最终都会归一成整数列表；
cudagraph_capture_sizes 会排序，并与 max_cudagraph_capture_size 对齐。
```

### 8.7 动态 speculative decoding 会改写 cudagraph mode

`VllmConfig` 中有一个高级交互：

```python
if speculative_config.uses_dynamic_speculative_decoding()
   and self.compilation_config.cudagraph_mode.has_full_cudagraphs():
    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

位置：`vllm.py:768` 到 `vllm.py:783`

原因：

```text
动态 speculative decoding 的 target verification length 会在运行时变化；
full CUDA graph 对 shape 更敏感；
所以为了可靠性降级为 PIECEWISE。
```

### 8.8 模型加载阶段：应用 compile / cudagraph wrapper

GPUModelRunner 加载模型后，如果是 stock torch.compile：

```python
backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
self.model.compile(fullgraph=True, backend=backend)
return
```

位置：`gpu_model_runner.py:5273` 到 `gpu_model_runner.py:5283`

其他模式下，cudagraph 由 wrapper 控制：

```python
if is_breakable_cudagraph_enabled() and cudagraph_mode != CUDAGraphMode.NONE:
    self.model = BreakableCUDAGraphWrapper(self.model, self.vllm_config)
elif cudagraph_mode.has_full_cudagraphs():
    self.model = CUDAGraphWrapper(
        self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
    )
elif self.parallel_config.use_ubatching:
    self.model = UBatchWrapper(...)
```

位置：`gpu_model_runner.py:5287` 到 `gpu_model_runner.py:5316`

所以 compilation hook 主要在两个点生效：

```text
加载后包装模型；
forward 时通过 set_forward_context 传入 cudagraph runtime mode / batch descriptor。
```

### 8.9 forward 阶段：runtime mode 进入上下文

`execute_model()` 在 forward 前会调用：

```python
set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    cudagraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    ...
)
```

位置：`gpu_model_runner.py:4303` 到 `gpu_model_runner.py:4313`

这让 attention backend、compiled graph、CUDA graph wrapper 能在 forward 时知道当前 batch 的执行形态。

---

## 9. KVTransferConfig：如何挂到 KV cache 生命周期

源码：`kv_transfer.py`、`kv_connector_model_runner_mixin.py`、`gpu_model_runner.py`

### 9.1 KVTransferConfig 描述什么

`KVTransferConfig` 定义在：`kv_transfer.py:22`

核心字段：

```text
kv_connector：connector 名称；
engine_id：engine id；
kv_buffer_device / kv_buffer_size：connector buffer；
kv_role：kv_producer / kv_consumer / kv_both；
kv_rank / kv_parallel_size：KV transfer 拓扑；
kv_ip / kv_port：连接信息；
kv_connector_extra_config：connector 额外参数；
kv_connector_module_path：V1 动态加载 connector；
enable_permute_local_kv：HND/NHD KV transfer 实验开关；
kv_load_failure_policy：load 失败时 recompute 还是 fail。
```

位置：`kv_transfer.py:26` 到 `kv_transfer.py:72`

### 9.2 post_init 校验 connector / role

```python
if self.engine_id is None:
    self.engine_id = str(uuid.uuid4())

if self.kv_connector is not None and self.kv_role is None:
    raise ValueError("Please specify kv_role when kv_connector is set...")
```

位置：`kv_transfer.py:92` 到 `kv_transfer.py:106`

并提供属性：

```python
is_kv_transfer_instance
is_kv_producer
is_kv_consumer
```

位置：`kv_transfer.py:108` 到 `kv_transfer.py:118`

### 9.3 Cache offloading 会派生 KVTransferConfig

`VllmConfig._post_init_kv_transfer_config()` 会读取 `cache_config.kv_offloading_size`。

如果配置了 KV offloading，但没有显式 `KVTransferConfig`，会创建一个：

```python
if self.kv_transfer_config is None:
    self.kv_transfer_config = KVTransferConfig()
```

位置：`vllm.py:791` 到 `vllm.py:799`

native offloading：

```python
self.kv_transfer_config.kv_connector = "OffloadingConnector"
self.kv_transfer_config.kv_connector_extra_config.update(
    {"cpu_bytes_to_use": kv_offloading_size * (1 << 30)}
)
```

位置：`vllm.py:801` 到 `vllm.py:809`

LMCache offloading：

```python
self.kv_transfer_config.kv_connector = "LMCacheMPConnector"
```

位置：`vllm.py:810` 到 `vllm.py:817`

最后统一：

```python
self.kv_transfer_config.kv_role = "kv_both"
```

位置：`vllm.py:818` 到 `vllm.py:819`

这说明 KVTransferConfig 不一定只来自用户显式参数，也可能由 cache offload 自动派生。

### 9.4 execute_model 前处理 preemptions

`GPUModelRunner.execute_model()` 开始阶段，如果启用 KV transfer：

```python
if has_kv_transfer_group():
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
```

位置：`gpu_model_runner.py:4075` 到 `gpu_model_runner.py:4078`

说明 scheduler 会把 connector metadata 放进 `SchedulerOutput`，Worker 在执行前消费它。

### 9.5 0-token 时也可能要跑 KV connector

如果本轮没有 scheduled tokens：

```python
if not has_kv_transfer_group():
    return EMPTY_MODEL_RUNNER_OUTPUT
return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
```

位置：`gpu_model_runner.py:4096` 到 `gpu_model_runner.py:4112`

`kv_connector_no_forward()` 会在没有 forward 的情况下仍然执行 connector 生命周期：

```python
with (
    set_forward_context(None, vllm_config),
    _get_kv_connector_output(scheduler_output, wait_for_save=False) as kv_connector_output,
):
    pass
return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
```

位置：`kv_connector_model_runner_mixin.py:35` 到 `kv_connector_model_runner_mixin.py:48`

这说明：

```text
KV transfer 不是模型 forward 的附属品；
它也可能在没有 token 计算时推进 load/save 状态。
```

### 9.6 forward 阶段的 KV connector 生命周期

`execute_model()` 在 forward 上下文中包了：

```python
self.maybe_get_kv_connector_output(
    scheduler_output,
    defer_finalize=defer_kv_connector_finalize,
) as kv_connector_output
```

位置：`gpu_model_runner.py:4315` 到 `gpu_model_runner.py:4318`

`_get_kv_connector_output()` 会：

```python
kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
kv_connector.start_load_kv(get_forward_context())
...
kv_connector.wait_for_save()
output.finished_sending, output.finished_recving = kv_connector.get_finished(...)
output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
output.kv_connector_stats = kv_connector.get_kv_connector_stats()
output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()
kv_connector.clear_connector_metadata()
```

位置：`kv_connector_model_runner_mixin.py:77` 到 `kv_connector_model_runner_mixin.py:112`

也就是说 KV transfer 的 hook 是围绕 forward context 的：

```text
SchedulerOutput.kv_connector_metadata
  → bind_connector_metadata
  → start_load_kv(forward_context)
  → model forward 期间异步 load/save KV
  → wait_for_save / get_finished / stats / events
  → KVConnectorOutput 放入 ModelRunnerOutput
```

### 9.7 Spec decode 会影响 KV connector finalize 时机

如果启用了 spec config，target forward 之后可能需要 draft model 也保存 KV，所以 finalize 会延后：

```python
# draft model runs. Deferred from target model forward to allow
# draft model to also save its KV cache.
if spec_config is not None:
    self.finalize_kv_connector()
```

位置：`gpu_model_runner.py:4597` 到 `gpu_model_runner.py:4600`

`finalize_kv_connector()`：

```python
kv_connector.wait_for_save()
kv_connector.clear_connector_metadata()
```

位置：`kv_connector_model_runner_mixin.py:63` 到 `kv_connector_model_runner_mixin.py:72`

这是高级能力之间交互的典型例子：

```text
Speculative decoding 改变 target/draft forward 顺序；
KV transfer 必须推迟 finalize；
否则 draft model 的 KV save 可能错过 connector metadata。
```

---

## 10. 高级能力之间的组合关系

这些能力不是完全独立的，代码里有不少组合 hook。

### 10.1 LoRA + Multimodal

LoRA 不只挂语言模型，也可以挂多模态 tower / connector。

配置：

```text
LoRAConfig.enable_tower_connector_lora
```

位置：`lora.py:62` 到 `lora.py:66`

执行 hook：

```text
_execute_mm_encoder()
  → TOWER LoRA mapping
  → CONNECTOR LoRA mapping
  → model.embed_multimodal()
```

位置：`gpu_model_runner.py:2941` 到 `gpu_model_runner.py:3008`

### 10.2 Spec Decode + Compilation

动态 speculative decoding 会把 full cudagraph 降级到 piecewise：

```python
self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

位置：`vllm.py:768` 到 `vllm.py:783`

原因是 verification length 动态变化，full CUDA graph 的 shape 约束更强。

### 10.3 Spec Decode + KV Transfer

Spec decode 会推迟 KV connector finalize。

位置：`gpu_model_runner.py:4597` 到 `gpu_model_runner.py:4600`

### 10.4 Multimodal + Compilation

`CompilationConfig` 有 multimodal encoder 相关开关：

```text
compile_mm_encoder
cudagraph_mm_encoder
encoder_cudagraph_token_budgets
encoder_cudagraph_max_vision_items_per_batch
encoder_cudagraph_max_frames_per_batch
```

位置：`compilation.py:516` 到 `compilation.py:553`

执行时，如果 encoder cudagraph manager 可用：

```python
if self.encoder_cudagraph_manager is not None and supports_modality(modality):
    cudagraph_output = self.encoder_cudagraph_manager.execute(mm_kwargs_batch)
```

位置：`gpu_model_runner.py:3073` 到 `gpu_model_runner.py:3085`

### 10.5 LoRA + Compilation

LoRA 有 cudagraph warmup / active LoRA specialization 相关逻辑。

`LoRAModelRunnerMixin` 提供 dummy LoRA context：

```text
maybe_setup_dummy_loras()
maybe_select_dummy_loras()
maybe_dummy_run_with_lora()
```

位置：`lora_model_runner_mixin.py:93` 到 `lora_model_runner_mixin.py:260`

用途是：

```text
CUDA graph capture / warmup 时模拟 active LoRA；
避免运行时第一次遇到 LoRA batch 才初始化相关 buffer / graph。
```

---

## 11. 高级 hook 和模型加载的边界

容易混淆的一点是：不是所有高级能力都在“加载权重”阶段生效。

可以这样分：

```text
LoRA：
  模型加载后包装 layer；adapter 权重按请求动态加载。

Multimodal：
  模型类需要支持 SupportsMultiModal；主要在输入预处理阶段运行 encoder 和 splice embeddings。

Speculative decoding：
  VllmConfig post_init 生成 draft config；ModelRunner init 创建 proposer；模型加载后加载 drafter；执行阶段做 draft/verify/rejection。

Compilation：
  配置 post_init 决定 backend/pass/graph sizes；模型加载后包装 compile/cudagraph；forward 时通过 context 选择 runtime mode。

KV Transfer：
  配置决定 connector；调度输出携带 metadata；forward 周期里 load/save KV；输出 KVConnectorOutput 回给 Scheduler。
```

如果只看 `model_loader.load_model()`，只能看到 base model 的加载。高级能力通常在它前后插入：

```text
before load_model：VllmConfig post_init 归一化高级配置；
inside load_model：模型类可能接收 vllm_config，并据此构造支持高级能力的模块；
after load_model：LoRA、drafter、compile wrapper、EPLB 等包装或注册；
execute_model：多模态、LoRA mapping、spec decode、KV transfer 真正参与 batch。
```

---

## 12. 容易疑惑的点

### 12.1 高级 Config 是否一定影响模型结构？

不一定。

```text
LoRAConfig 的 target_modules / max_loras 会影响 LoRA wrapper；
SpeculativeConfig 的 EAGLE3 / DFlash 会影响 hidden state 输出；
CompilationConfig 会影响 compiled graph；
KVTransferConfig 通常不影响计算图；
MultiModalConfig 只有部分 encoder 相关字段影响 encoder 计算图。
```

### 12.2 LoRA adapter 权重是在主模型加载时一起加载吗？

不是。

主模型加载后只是安装 LoRA manager 和 LoRA wrapper。具体 adapter 通常由请求侧 `LoRARequest` 触发，WorkerLoRAManager 动态加载和缓存。

### 12.3 多模态是不是一定走 input_ids？

不是。

多模态 decoder-only 模型通常会把文本 token 和多模态 embedding 统一成 `inputs_embeds`，再进入 forward。

位置：`gpu_model_runner.py:3447` 到 `gpu_model_runner.py:3495`

### 12.4 Speculative decoding 的 draft model 是在哪里加载的？

不是在主模型 `get_model_loader()` 那一步一起加载，而是在 target model 加载后：

```python
self.drafter.load_model(self.model)
```

位置：`gpu_model_runner.py:5171` 到 `gpu_model_runner.py:5174`

### 12.5 CompilationConfig 是不是只控制 torch.compile？

不是。

它同时控制：

```text
torch.compile backend；
vLLM custom Inductor backend；
custom ops；
Inductor passes；
CUDA graph full / piecewise capture；
multimodal encoder compile / cudagraph。
```

### 12.6 KVTransferConfig 为什么 compute_hash 为空？

因为它不改变从 input ids / embeddings 到 hidden states 的计算图。它影响 KV cache 的传输协议、存储位置和 runtime metadata，而不是模型数学结构。

位置：`kv_transfer.py:74` 到 `kv_transfer.py:90`

### 12.7 为什么有些 hook 在最后一个 PP rank？

Speculative drafter 当前只在最后一个 PP rank 创建：

```python
if self.speculative_config and get_pp_group().is_last_rank:
    ... create drafter ...
```

位置：`gpu_model_runner.py:542` 到 `gpu_model_runner.py:620`

因为 draft / rejection / sampling 通常需要最终 hidden states / logits，而这些只在最后一个 PP stage 完整可见。

---

## 13. 总结

高级能力挂到 vLLM 的主链路可以压缩成：

```text
用户参数
  → 子 Config
  → VllmConfig post_init
  → compute_hash / 编译缓存 key
  → ModelRunner 初始化能力对象
  → base model load_model()
  → after-load wrapper / drafter / manager
  → execute_model 输入准备 hook
  → forward context hook
  → sample / connector / output hook
```

如果只记住一句话：

```text
vLLM 的高级能力不是直接塞进模型 forward，而是先通过 Config 变成可校验、可 hash、可分发的运行时协议，再由 ModelRunner 在加载、输入准备、forward 和采样阶段逐点挂接。
```

再压缩成最小心智模型：

```text
LoRA：模型加载后装 wrapper，batch 执行时切 active adapter；
Multimodal：输入阶段跑 encoder，把多模态输出合进 inputs_embeds；
Spec Decode：配置阶段生成 draft config，运行时创建 drafter 并做验证采样；
Compilation：加载后包 compile / CUDA graph，forward 时按 batch 选择 runtime mode；
KV Transfer：调度输出携带 metadata，forward 周期里异步传 KV 并回传 connector output。
```
