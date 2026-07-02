# 06. 权重加载和量化如何接入？

源码位置：

- `code/vllm/vllm/v1/worker/gpu_model_runner.py`
- `code/vllm/vllm/model_executor/model_loader/__init__.py`
- `code/vllm/vllm/model_executor/model_loader/base_loader.py`
- `code/vllm/vllm/model_executor/model_loader/default_loader.py`
- `code/vllm/vllm/model_executor/model_loader/weight_utils.py`
- `code/vllm/vllm/model_executor/model_loader/bitsandbytes_loader.py`
- `code/vllm/vllm/model_executor/model_loader/sharded_state_loader.py`
- `code/vllm/vllm/model_executor/model_loader/tensorizer_loader.py`
- `code/vllm/vllm/model_executor/model_loader/dummy_loader.py`
- `code/vllm/vllm/model_executor/model_loader/utils.py`
- `code/vllm/vllm/model_executor/models/utils.py`
- `code/vllm/vllm/model_executor/models/llama.py`
- `code/vllm/vllm/model_executor/layers/linear.py`
- `code/vllm/vllm/model_executor/layers/quantization/__init__.py`
- `code/vllm/vllm/model_executor/layers/quantization/base_config.py`
- `code/vllm/vllm/config/vllm.py`
- `code/vllm/vllm/config/quantization.py`

本问题关注：模型实例创建后，vLLM 如何根据 `LoadConfig` 选择 loader，如何查找和下载权重文件，如何把 checkpoint tensor 迭代成 `(name, tensor)`，如何映射到 vLLM 模型参数，以及量化配置如何影响模型层创建、参数形状、权重切片、后处理和最终 kernel 调用。

---

## 0. 梳理规划

本篇按“谁触发加载 → 选择哪个 loader → 权重文件如何变成 tensor 流 → tensor 流如何进入模型参数 → 量化如何改变这一切”的顺序梳理。

要回答的问题分成 10 组：

```text
1. 权重加载在 V1 执行链路的哪里触发？
2. LoadConfig.load_format 如何决定 ModelLoader？
3. BaseModelLoader.load_model() 的固定流程是什么？
4. DefaultModelLoader 如何查找、下载、过滤权重文件？
5. safetensors / pt / fastsafetensors / instanttensor iterator 如何产出权重？
6. model.load_weights() 如何把 checkpoint 名称映射到 vLLM 参数？
7. QKV / GateUp 这类 fused 参数如何加载？
8. quant_config 在哪里生成并校验？
9. Linear / MoE 等层如何根据 quant_config 创建量化参数和 quant_method？
10. 权重加载后，量化方法如何做 repack / online quant / kernel 格式转换？
```

阅读顺序建议：

```text
01_config_entry_and_engine_args.md
  → 02_vllm_config_aggregation.md
  → 04_load_config_and_model_loader.md
  → 06_weight_loading_and_quantization.md
  → executor_worker_model_runner/03_model_runner_role.md
```

---

## 1. 一句话回答

权重加载是 `ModelLoader` 把外部 checkpoint 的 `(weight_name, tensor)` 流映射进 vLLM 模型参数的过程；量化会改变“参数怎么创建、tensor 怎么切片、加载后怎么 repack、forward 时用哪个 quant_method/kernel”。

最小主线是：

```text
GPUModelRunner.load_model()
  → get_model_loader(load_config)
  → BaseModelLoader.load_model()
  → initialize_model(vllm_config)
  → loader.load_weights(model, model_config)
  → weights_iterator: Iterable[(name, tensor)]
  → model.load_weights(weights)
  → param.weight_loader(...)
  → process_weights_after_loading()
  → model.eval()
```

量化主线是：

```text
EngineArgs.quantization / quantization_config
  → ModelConfig.quantization / quantization_config
  → VllmConfig.__post_init__()
  → VllmConfig._get_quantization_config()
  → vllm_config.quant_config
  → initialize_model() / configure_quant_config()
  → LinearBase.quant_config.get_quant_method(layer, prefix)
  → quant_method.create_weights()
  → quant_method.apply() / process_weights_after_loading()
```

一句话压缩：

```text
LoadConfig 决定怎么读 checkpoint，QuantizationConfig 决定 checkpoint tensor 被解释成什么参数和用什么 kernel 跑。
```

---

## 2. 权重加载在哪里触发

V1 GPU 路径的触发点在 `GPUModelRunner.load_model()`。

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5142`

核心代码是：

```python
if load_dummy_weights:
    self.load_config.load_format = "dummy"
model_loader = get_model_loader(self.load_config)
self.model = model_loader.load_model(
    vllm_config=self.vllm_config,
    model_config=self.model_config,
)
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5161` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5166`

因此执行层的加载链路是：

```text
Worker.load_model()
  → ModelRunner.load_model()
  → get_model_loader(load_config)
  → loader.load_model(vllm_config, model_config)
```

加载完成后，`GPUModelRunner.load_model()` 还会继续处理：

```text
LoRA 包装；
drafter 模型加载；
MoE / EPLB 状态注册；
通信 buffer 准备；
stock torch compile；
CUDAGraphWrapper / UBatchWrapper 包装；
offloader post_init。
```

对应位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5167` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5318`

也就是说，权重加载不是孤立阶段，而是模型进入可执行状态前的核心初始化阶段。

---

## 3. LoadConfig 如何选择 ModelLoader

入口函数是：

```python
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
```

位置：`code/vllm/vllm/model_executor/model_loader/__init__.py:122`

它根据 `load_config.load_format` 查表：

```python
_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {
    "auto": DefaultModelLoader,
    "hf": DefaultModelLoader,
    "bitsandbytes": BitsAndBytesModelLoader,
    "dummy": DummyModelLoader,
    "fastsafetensors": DefaultModelLoader,
    "instanttensor": DefaultModelLoader,
    "mistral": DefaultModelLoader,
    "modelexpress": ModelExpressModelLoader,
    "npcache": DefaultModelLoader,
    "pt": DefaultModelLoader,
    "runai_streamer": RunaiModelStreamerLoader,
    "runai_streamer_sharded": ShardedStateLoader,
    "safetensors": DefaultModelLoader,
    "sharded_state": ShardedStateLoader,
    "tensorizer": TensorizerLoader,
}
```

位置：`code/vllm/vllm/model_executor/model_loader/__init__.py:33` 到 `code/vllm/vllm/model_executor/model_loader/__init__.py:66`

因此常见格式含义是：

```text
auto / hf / safetensors / pt / npcache / mistral：DefaultModelLoader；
fastsafetensors / instanttensor：仍走 DefaultModelLoader，但 iterator 不同；
bitsandbytes：BitsAndBytesModelLoader；
sharded_state / runai_streamer_sharded：ShardedStateLoader；
tensorizer：TensorizerLoader；
dummy：DummyModelLoader；
modelexpress / runai_streamer：专用 loader。
```

`get_model()` 只是一个轻包装：

```python
loader = get_model_loader(load_config or vllm_config.load_config)
return loader.load_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
```

位置：`code/vllm/vllm/model_executor/model_loader/__init__.py:130` 到 `code/vllm/vllm/model_executor/model_loader/__init__.py:142`

---

## 4. BaseModelLoader 的固定流程

所有 loader 都继承 `BaseModelLoader`。

位置：`code/vllm/vllm/model_executor/model_loader/base_loader.py:25`

它定义两个抽象方法：

```python
def download_model(self, model_config: ModelConfig) -> None

def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None
```

位置：`code/vllm/vllm/model_executor/model_loader/base_loader.py:31` 到 `code/vllm/vllm/model_executor/model_loader/base_loader.py:40`

真正的模板方法是 `BaseModelLoader.load_model()`：

```python
with set_default_torch_dtype(model_config.dtype):
    with target_device:
        model = initialize_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )

    self.load_weights(model, model_config)

    if _has_online_quant(model):
        finalize_layerwise_processing(model, model_config)

    process_weights_after_loading(model, model_config, target_device)

return model.eval()
```

位置：`code/vllm/vllm/model_executor/model_loader/base_loader.py:42` 到 `code/vllm/vllm/model_executor/model_loader/base_loader.py:82`

这个模板说明了几个关键边界：

```text
initialize_model()：只创建模型结构和空参数；
loader.load_weights()：把 checkpoint tensor 加进参数；
finalize_layerwise_processing()：online quant 场景的 layer-wise 处理；
process_weights_after_loading()：量化 repack、attention 后处理、torchao reload 元信息；
model.eval()：返回可推理模型。
```

---

## 5. 模型实例如何创建，quant_config 如何传进去

模型初始化函数是：

```python
def initialize_model(vllm_config: VllmConfig, ..., model_config: ModelConfig | None = None) -> nn.Module:
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:40` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:47`

它先解析模型架构：

```python
model_class, _ = get_model_architecture(model_config)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:51` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:52`

如果 `vllm_config.quant_config` 存在，会先配置量化映射：

```python
if vllm_config.quant_config is not None:
    configure_quant_config(vllm_config.quant_config, model_class)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:54` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:55`

新式模型类会直接接收完整 `VllmConfig`：

```python
model = model_class(vllm_config=vllm_config, prefix=prefix)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:57` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:64`

这意味着模型内部可以读取：

```text
vllm_config.model_config
vllm_config.cache_config
vllm_config.parallel_config
vllm_config.load_config
vllm_config.quant_config
vllm_config.lora_config
vllm_config.compilation_config
```

以 LLaMA 为例，`LlamaForCausalLM.__init__()` 直接取：

```python
config = vllm_config.model_config.hf_config
quant_config = vllm_config.quant_config
```

位置：`code/vllm/vllm/model_executor/models/llama.py:500` 到 `code/vllm/vllm/model_executor/models/llama.py:510`

然后创建 `ParallelLMHead(..., quant_config=quant_config)`：

位置：`code/vllm/vllm/model_executor/models/llama.py:518` 到 `code/vllm/vllm/model_executor/models/llama.py:524`

内部 LLaMA 层也会把 `quant_config` 继续传给 attention / MLP 里的 Linear 层。

---

## 6. DefaultModelLoader 如何准备权重文件

`DefaultModelLoader` 是最常用的 loader。

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:43`

### 6.1 _prepare_weights() 的职责

```python
def _prepare_weights(...):
    """Prepare weights for the model.

    If the model is not local, it will be downloaded.
    """
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:97` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:107`

它主要做：

```text
1. ModelScope 下载替换；
2. 判断 model_or_path 是否本地目录；
3. 根据 load_format 决定 allow_patterns；
4. 非本地模型从 HuggingFace 下载；
5. 本地模型直接 glob 查找；
6. safetensors 场景下载/使用 index 文件过滤重复 shard；
7. pt/bin 场景过滤训练用文件；
8. 找不到任何权重就报错。
```

### 6.2 load_format 到文件模式的映射

`_prepare_weights()` 中的规则是：

```text
load_format == "auto"：
  如果 repo 里有 consolidated*.safetensors，按 mistral；否则按 hf。

load_format == "hf"：
  allow_patterns = ["*.safetensors", "*.bin"]

load_format in {"safetensors", "fastsafetensors", "instanttensor"}：
  allow_patterns = ["*.safetensors"]

load_format == "mistral"：
  allow_patterns = ["consolidated*.safetensors"]
  index_file = "consolidated.safetensors.index.json"

load_format == "pt"：
  allow_patterns = ["*.pt"]

load_format == "npcache"：
  allow_patterns = ["*.bin"]
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:118` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:153`

如果 `fall_back_to_pt=True`，还会追加：

```text
*.pt
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:155` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:156`

### 6.3 远程下载和本地查找

非本地模型会调用：

```python
hf_folder = download_weights_from_hf(...)
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:161` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:169`

本地模型则直接使用目录：

```python
hf_folder = model_name_or_path
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:170` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:171`

之后按 pattern 查找：

```python
hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:176` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:182`

### 6.4 safetensors index 过滤

safetensors 场景下，为避免同时加载 consolidated 和 sharded 文件，会用 index 过滤：

```python
hf_weights_files = filter_duplicate_safetensors_files(
    hf_weights_files, hf_folder, index_file
)
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:184` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:200`

`filter_duplicate_safetensors_files()` 会读取 `model.safetensors.index.json` 里的 `weight_map`，只保留 index 中出现的文件。

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:582` 到 `code/vllm/vllm/model_executor/model_loader/weight_utils.py:600`

### 6.5 非 safetensors 文件过滤

非 safetensors 场景会过滤训练态文件：

```text
training_args.bin
optimizer.bin
optimizer.pt
scheduler.pt
scaler.pt
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:603` 到 `code/vllm/vllm/model_executor/model_loader/weight_utils.py:619`

---

## 7. HuggingFace 下载逻辑

下载函数是：

```python
def download_weights_from_hf(model_name_or_path, cache_dir, allow_patterns, revision, ...)
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:430`

它做几件事：

```text
1. 如果不是 HF offline，先尝试列 repo 文件；
2. 如果要下载 safetensors 且存在 index 文件，就只下载 weight_map 指向的 shard；
3. 否则选择第一个能匹配到文件的 allow_pattern；
4. 用文件锁避免多进程重复下载；
5. 调用 hf_api().snapshot_download()；
6. 如果当前 pattern 已经下载到权重，就不再尝试后续 pattern。
```

关键代码位置：

```text
列 repo 文件和 safetensors index 优化：
  code/vllm/vllm/model_executor/model_loader/weight_utils.py:459 到 496

文件锁和 snapshot_download：
  code/vllm/vllm/model_executor/model_loader/weight_utils.py:506 到 535
```

这说明 `DefaultModelLoader` 的“下载”不是简单拉整个 repo，而是尽量只拉推理需要的权重文件。

---

## 8. 权重 iterator 如何产出 `(name, tensor)`

`DefaultModelLoader._get_weights_iterator()` 是 iterator 选择点。

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:211`

它先调用 `_prepare_weights()` 得到：

```text
hf_folder
hf_weights_files
use_safetensors
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:215` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:222`

然后根据格式选择 iterator：

```text
npcache                  → np_cache_weights_iterator()
fastsafetensors          → fastsafetensors_weights_iterator()
instanttensor            → instanttensor_weights_iterator()
safetensors 普通路径      → safetensors_weights_iterator()
safetensors 多线程        → multi_thread_safetensors_weights_iterator()
pt/bin 普通路径           → pt_weights_iterator()
pt/bin 多线程             → multi_thread_pt_weights_iterator()
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:223` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:281`

最后它会给每个权重名加 source prefix：

```python
return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:283` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:286`

### 8.1 safetensors_weights_iterator

`safetensors_weights_iterator()` 定义在：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:820`

它的输入是 safetensors shard 文件列表，输出是：

```text
yield name, param
```

普通路径：

```python
with safe_open(st_file, framework="pt") as f:
    for name in f.keys():
        if should_skip_weight(name, local_expert_ids):
            continue
        param = f.get_tensor(name)
        yield name, param
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:948` 到 `code/vllm/vllm/model_executor/model_loader/weight_utils.py:954`

它还支持：

```text
eager：一次性读入整个 safetensors 文件；
torchao：从 flattened torchao tensor subclass 反扁平；
prefetch：网络文件系统且权重能放进 RAM 时预取；
local_expert_ids：EP 场景下读盘前跳过非本 rank expert 权重。
```

对应位置：

```text
eager：code/vllm/vllm/model_executor/model_loader/weight_utils.py:912 到 917
torchao：code/vllm/vllm/model_executor/model_loader/weight_utils.py:918 到 947
prefetch 判定：code/vllm/vllm/model_executor/model_loader/weight_utils.py:841 到 903
EP skip：code/vllm/vllm/model_executor/model_loader/weight_utils.py:829 到 834
```

### 8.2 pt_weights_iterator

`pt_weights_iterator()` 定义在：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:1133`

它逐个 `torch.load()` checkpoint shard：

```python
state = torch.load(bin_file, map_location=pt_load_map_location, weights_only=True)
yield from state.items()
del state
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:1133` 到 `code/vllm/vllm/model_executor/model_loader/weight_utils.py:1149`

### 8.3 fastsafetensors / instanttensor

`fastsafetensors_weights_iterator()` 使用 `fastsafetensors.parallel_loader.ParallelLoader`，支持流水加载和 GDS fallback。

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:1024` 到 `code/vllm/vllm/model_executor/model_loader/weight_utils.py:1090`

`instanttensor_weights_iterator()` 使用 `instanttensor.safe_open()`，要求 NVIDIA GPU。

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:1093` 到 `code/vllm/vllm/model_executor/model_loader/weight_utils.py:1130`

---

## 9. DefaultModelLoader 如何调用模型 load_weights

`DefaultModelLoader.get_all_weights()` 先生成主权重 source：

```python
primary_weights = DefaultModelLoader.Source(
    model_config.model,
    model_config.revision,
    prefix="",
    fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
    allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
)
yield from self._get_weights_iterator(primary_weights)
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:288` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:300`

如果模型声明了 `secondary_weights`，还会继续加载第二来源：

```python
secondary_weights = getattr(model, "secondary_weights", ())
for source in secondary_weights:
    yield from self._get_weights_iterator(source)
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:302` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:307`

真正加载权重的位置是：

```python
loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:381` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:395`

加载后会记录耗时，并在特定场景检查哪些参数没加载：

```text
默认只对非量化模型启用 loaded weights tracking；
量化模型因为有 online scale、postprocess 参数、checkpoint 辅助 tensor 等特殊项，默认不严格检查所有 parameter 都来自 checkpoint。
```

位置：`code/vllm/vllm/model_executor/model_loader/default_loader.py:396` 到 `code/vllm/vllm/model_executor/model_loader/default_loader.py:437`

---

## 10. 模型 load_weights 如何映射参数

vLLM 模型一般有两种加载方式：

```text
1. 模型自己手写 load_weights()；
2. 使用 AutoWeightsLoader 通用递归加载器。
```

### 10.1 LLaMA 手写 inner model load_weights

`LlamaModel.load_weights()` 展示了典型 fused 参数映射。

位置：`code/vllm/vllm/model_executor/models/llama.py:433`

它定义：

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]
```

位置：`code/vllm/vllm/model_executor/models/llama.py:433` 到 `code/vllm/vllm/model_executor/models/llama.py:441`

含义是：

```text
checkpoint 里的 q_proj / k_proj / v_proj
  → vLLM 模型里的 qkv_proj 一个 fused 参数

checkpoint 里的 gate_proj / up_proj
  → vLLM 模型里的 gate_up_proj 一个 fused 参数
```

加载流程是：

```python
params_dict = dict(self.named_parameters())
for name, loaded_weight in weights:
    ...
    for param_name, weight_name, shard_id in stacked_params_mapping:
        if weight_name not in name:
            continue
        name = name.replace(weight_name, param_name)
        param = params_dict[name]
        weight_loader = param.weight_loader
        weight_loader(param, loaded_weight, shard_id)
        break
    else:
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:442` 到 `code/vllm/vllm/model_executor/models/llama.py:482`

几个细节：

```text
rotary_emb.inv_freq / cos_cached / sin_cached 会跳过；
scale / zero_point 可能通过 maybe_remap_kv_scale_name() 改名；
GPTQ 多出来的 bias 如果模型参数里没有，会跳过；
PP 当前 rank 缺失的参数会跳过；
fused 参数会把 shard_id 传给 param.weight_loader。
```

### 10.2 LlamaForCausalLM 使用 AutoWeightsLoader

外层 `LlamaForCausalLM.load_weights()` 使用通用加载器：

```python
loader = AutoWeightsLoader(
    self,
    skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
)
return loader.load_weights(weights)
```

位置：`code/vllm/vllm/model_executor/models/llama.py:569` 到 `code/vllm/vllm/model_executor/models/llama.py:574`

如果词嵌入和 lm_head tie weight，就跳过 checkpoint 中的 `lm_head.*`。

### 10.3 AutoWeightsLoader 做什么

`AutoWeightsLoader` 定义在：`code/vllm/vllm/model_executor/models/utils.py:124`

它的目标是：

```text
递归遍历 module / child module / parameter；
遇到子 module 自己实现 load_weights() 就委托给它；
遇到 parameter 就调用 param.weight_loader 或 default_weight_loader；
支持 skip_prefixes / skip_substrs；
支持 WeightsMapper 做名称改写；
支持 quant_config.get_cache_scale_mapper() 自动处理 KV cache scale 名称。
```

参数加载的关键代码是：

```python
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, weight_data)
```

位置：`code/vllm/vllm/model_executor/models/utils.py:206` 到 `code/vllm/vllm/model_executor/models/utils.py:236`

递归加载模块的关键代码是：

```python
if callable(module_load_weights):
    loaded_params = module_load_weights(weights)
...
elif child_prefix in child_params:
    yield from self._load_param(prefix, child_params[child_prefix], child_weights)
```

位置：`code/vllm/vllm/model_executor/models/utils.py:268` 到 `code/vllm/vllm/model_executor/models/utils.py:346`

最终入口是：

```python
if mapper is not None:
    weights = mapper.apply(weights)
weights = ((name, weight) for name, weight in weights if not self._can_skip(name))
autoloaded_weights = set(self._load_module("", self.module, weights))
```

位置：`code/vllm/vllm/model_executor/models/utils.py:348` 到 `code/vllm/vllm/model_executor/models/utils.py:377`

### 10.4 WeightsMapper

`WeightsMapper` 用来把 checkpoint 名称改成 vLLM 模型名称，或忽略某些权重。

位置：`code/vllm/vllm/model_executor/models/utils.py:41`

支持的映射方式包括：

```text
orig_to_new_renamings
orig_to_new_regex
orig_to_new_substr
orig_to_new_prefix
orig_to_new_suffix
```

如果某个映射目标是 `None`，对应权重会被忽略。

位置：`code/vllm/vllm/model_executor/models/utils.py:66` 到 `code/vllm/vllm/model_executor/models/utils.py:121`

---

## 11. param.weight_loader 从哪里来

权重最终不是简单 `param.data.copy_(tensor)`，而是经常走参数自己的 `weight_loader`。

### 11.1 LinearBase 选择 quant_method

`LinearBase.__init__()` 会根据 `quant_config` 选择线性层实现：

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
    self.quant_method = quant_method
else:
    raise ValueError("All linear layers should support quant method.")
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:242` 到 `code/vllm/vllm/model_executor/layers/linear.py:274`

这一步决定了：

```text
参数如何创建；
参数是否 packed；
是否存在 scale / zero_point；
forward 调用哪个 apply；
加载后是否需要 process_weights_after_loading。
```

### 11.2 quant_method.create_weights() 创建参数并挂 weight_loader

未量化线性层的 `UnquantizedLinearMethod.create_weights()` 会创建 `ModelWeightParameter`：

```python
weight = ModelWeightParameter(
    data=torch.empty(...),
    input_dim=1,
    output_dim=0,
    weight_loader=weight_loader,
)
layer.register_parameter("weight", weight)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:179` 到 `code/vllm/vllm/model_executor/layers/linear.py:210`

量化方法也实现同样接口，但创建的参数可能是：

```text
PackedColumnParameter
PackedvLLMParameter
BlockQuantScaleParameter
PerTensorScaleParameter
RowvLLMParameter
各种 scale / zero_point / group size 相关参数
```

这些参数的形状、packed_factor、packed_dim、input_dim、output_dim 都会影响加载时如何切片。

### 11.3 ColumnParallelLinear 的 weight_loader

`ColumnParallelLinear` 创建权重时会把 loader 传给 quant method：

```python
self.quant_method.create_weights(
    layer=self,
    input_size_per_partition=self.input_size_per_partition,
    output_partition_sizes=self.output_partition_sizes,
    input_size=self.input_size,
    output_size=self.output_size,
    params_dtype=self.params_dtype,
    weight_loader=(
        self.weight_loader_v2
        if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
        else self.weight_loader
    ),
)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:461` 到 `code/vllm/vllm/model_executor/layers/linear.py:473`

普通 `weight_loader()` 会按 TP rank 切 output 维：

```python
if output_dim is not None and not is_sharded_weight:
    shard_size = param_data.shape[output_dim]
    start_idx = self.tp_rank * shard_size
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
param_data.copy_(loaded_weight)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:517` 到 `code/vllm/vllm/model_executor/layers/linear.py:538`

`weight_loader_v2()` 则交给 `BasevLLMParameter`：

```python
param.load_column_parallel_weight(loaded_weight=loaded_weight)
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:540` 到 `code/vllm/vllm/model_executor/layers/linear.py:546`

### 11.4 MergedColumnParallelLinear 如何加载 fused 参数

`MergedColumnParallelLinear` 用于多个矩阵沿 output 维拼接，例如 GateUp。

位置：`code/vllm/vllm/model_executor/layers/linear.py:577`

它的 `weight_loader()` 接收 `loaded_shard_id`：

```python
def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:662` 到 `code/vllm/vllm/model_executor/layers/linear.py:667`

如果 checkpoint 已经 fused，`loaded_shard_id is None`；如果 checkpoint 是拆开的 `gate_proj` / `up_proj`，模型的 `load_weights()` 会传 `0` / `1`。

它会处理：

```text
output_dim 切片；
TP rank 切片；
packed_dim / packed_factor 修正；
Marlin tile size 修正；
BlockQuantScaleParameter 的 block scale shard 修正；
bitsandbytes 4bit shard 修正；
per-tensor scale 标量加载到 fused array。
```

关键位置：`code/vllm/vllm/model_executor/layers/linear.py:670` 到 `code/vllm/vllm/model_executor/layers/linear.py:805`

### 11.5 QKVParallelLinear 如何加载 Q/K/V

`QKVParallelLinear` 继承 `ColumnParallelLinear`，专门处理 attention 里的 Q/K/V 拼接。

位置：`code/vllm/vllm/model_executor/layers/linear.py:914`

模型 checkpoint 通常是：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

vLLM 参数通常是：

```text
qkv_proj.weight
```

模型 `load_weights()` 会把 `shard_id` 传成：

```text
"q" / "k" / "v"
```

随后 `QKVParallelLinear.weight_loader()` 根据 shard id、TP rank、KV head 分布、packed_dim 等规则写入 `qkv_proj` 的对应位置。

相关位置：

```text
QKVParallelLinear 定义：code/vllm/vllm/model_executor/layers/linear.py:914
QKV weight_loader_v2：code/vllm/vllm/model_executor/layers/linear.py:1078
QKV weight_loader：code/vllm/vllm/model_executor/layers/linear.py:1125
```

---

## 12. quant_config 在哪里生成

量化入口参数来自 `EngineArgs`：

```text
quantization
quantization_config
allow_deprecated_quantization
```

在 `EngineArgs.__post_init__()` 里，会先解析用户传入的 online quant shorthand 和 `quantization_config`：

```python
self.quantization_config = resolve_quantization_config(
    self.quantization,
    self.quantization_config,
)
```

位置：`code/vllm/vllm/engine/arg_utils.py:743` 到 `code/vllm/vllm/engine/arg_utils.py:747`

`resolve_quantization_config()` 定义在：`code/vllm/vllm/config/quantization.py:147`

它支持：

```text
--quantization fp8_per_tensor
--quantization fp8_per_block
--quantization fp8_per_channel
--quantization mxfp8
--quantization int8_per_channel_weight_only
--quantization online
--quantization-config {...}
```

位置：`code/vllm/vllm/config/quantization.py:112` 到 `code/vllm/vllm/config/quantization.py:183`

这些 online shorthand 会被转换成 `QuantizationConfigArgs`，用于在线量化。

### 12.1 VllmConfig.__post_init__ 创建 quant_config

`VllmConfig.__post_init__()` 中，如果 `quant_config` 还没有设置，会根据 `model_config.quantization` 创建：

```python
if self.quant_config is None and self.model_config is not None:
    self.quant_config = VllmConfig._get_quantization_config(
        self.model_config, self.load_config
    )
```

位置：`code/vllm/vllm/config/vllm.py:920` 到 `code/vllm/vllm/config/vllm.py:923`

实际创建函数是：

```python
quant_config = get_quant_config(model_config, load_config)
```

位置：`code/vllm/vllm/config/vllm.py:618` 到 `code/vllm/vllm/config/vllm.py:651`

这里还会校验：

```text
当前 GPU capability >= quant_config.get_min_capability()；
model_config.dtype 在 quant_config.get_supported_act_dtypes() 里；
quant_config.maybe_update_config(model, hf_config) 可按模型补充配置。
```

位置：`code/vllm/vllm/config/vllm.py:629` 到 `code/vllm/vllm/config/vllm.py:650`

### 12.2 get_quant_config 如何读取量化配置

`get_quant_config()` 定义在：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:240`

它先根据 quantization 字符串找到配置类：

```python
quant_cls = get_quantization_config(model_config.quantization)
```

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:243` 到 `code/vllm/vllm/model_executor/model_loader/weight_utils.py:246`

然后按优先级读取配置：

```text
1. HF config 里的 quantization_config；
2. HF text_config.quantization_config；
3. HF config 里的 compression_config；
4. hf_overrides["quantization_config_file"]；
5. hf_overrides["quantization_config_dict_json"]；
6. model_config.quantization_config，即 online quant；
7. bitsandbytes 的空 config；
8. 从模型目录下载/读取量化 json 文件；
9. 如果 quant method 没有 config 文件要求，则使用默认 quant_cls()。
```

对应源码：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:247` 到 `code/vllm/vllm/model_executor/model_loader/weight_utils.py:394`

### 12.3 quantization 字符串到 Config 类的注册表

`get_quantization_config()` 定义在：`code/vllm/vllm/model_executor/layers/quantization/__init__.py:107`

它支持的典型方法包括：

```text
awq
fp8
modelopt / modelopt_fp4 / modelopt_mxfp8 / modelopt_mixed
auto_gptq / gptq / gptq_marlin
awq_marlin
compressed-tensors
bitsandbytes
experts_int8
quark
moe_wna16
torchao
mxfp4 / gpt_oss_mxfp4
deepseek_v4_fp8
online
fp8_per_tensor / fp8_per_block / fp8_per_channel / int8_per_channel_weight_only / mxfp8
```

位置：`code/vllm/vllm/model_executor/layers/quantization/__init__.py:12` 到 `code/vllm/vllm/model_executor/layers/quantization/__init__.py:45`

映射表位置：`code/vllm/vllm/model_executor/layers/quantization/__init__.py:140` 到 `code/vllm/vllm/model_executor/layers/quantization/__init__.py:182`

也支持外部注册：

```python
@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    ...
```

位置：`code/vllm/vllm/model_executor/layers/quantization/__init__.py:57` 到 `code/vllm/vllm/model_executor/layers/quantization/__init__.py:104`

---

## 13. QuantizationConfig / QuantizeMethodBase 的接口边界

基础接口在：`code/vllm/vllm/model_executor/layers/quantization/base_config.py`

### 13.1 QuantizationConfig

`QuantizationConfig` 要实现：

```python
def get_name(self) -> QuantizationMethods

def get_supported_act_dtypes(self) -> list[torch.dtype]

def get_min_capability(cls) -> int

def get_config_filenames() -> list[str]

def from_config(cls, config: dict[str, Any]) -> QuantizationConfig

def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None
```

位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:77` 到 `code/vllm/vllm/model_executor/layers/quantization/base_config.py:170`

其中最关键的是：

```text
get_quant_method(layer, prefix)
```

它决定某个 layer 是否使用该量化方法，以及返回哪种 `QuantizeMethodBase`。

### 13.2 QuantizeMethodBase

`QuantizeMethodBase` 要实现：

```python
def create_weights(self, layer, *weight_args, **extra_weight_attrs)

def apply(self, layer, *args, **kwargs) -> torch.Tensor
```

位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:19` 到 `code/vllm/vllm/model_executor/layers/quantization/base_config.py:41`

可选后处理：

```python
def process_weights_after_loading(self, layer: nn.Module) -> None:
    return
```

位置：`code/vllm/vllm/model_executor/layers/quantization/base_config.py:57` 到 `code/vllm/vllm/model_executor/layers/quantization/base_config.py:62`

所以量化方法的职责边界是：

```text
create_weights()：定义参数形状、packed 参数、scale、zero point、weight_loader 属性；
weight_loader()：在加载阶段把 checkpoint tensor 放入参数；
process_weights_after_loading()：加载完后 repack / 转置 / online quant；
apply()：forward 阶段调用对应 kernel。
```

---

## 14. 权重加载后处理

`BaseModelLoader.load_model()` 在 `load_weights()` 后调用：

```python
process_weights_after_loading(model, model_config, target_device)
```

位置：`code/vllm/vllm/model_executor/model_loader/base_loader.py:75` 到 `code/vllm/vllm/model_executor/model_loader/base_loader.py:80`

`process_weights_after_loading()` 遍历所有 module：

```python
quant_method = getattr(module, "quant_method", None)
if isinstance(quant_method, QuantizeMethodBase):
    with device_loading_context(module, target_device):
        quant_method.process_weights_after_loading(module)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:100` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:116`

然后再处理 Attention / MLA / MM encoder 的 post-load 权重：

```python
if isinstance(module, (Attention, MLAAttention, MMEncoderAttention)):
    module.process_weights_after_loading(model_config.dtype)
```

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:118` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:126`

如果是 torchao，还会设置 reload 属性：

位置：`code/vllm/vllm/model_executor/model_loader/utils.py:128` 到 `code/vllm/vllm/model_executor/model_loader/utils.py:131`

### 14.1 online quant 的特殊处理

`BaseModelLoader._has_online_quant()` 会检查 module 的 `quant_method.uses_meta_device`：

```python
if getattr(quant_method, "uses_meta_device", False):
    return True
```

位置：`code/vllm/vllm/model_executor/model_loader/base_loader.py:95` 到 `code/vllm/vllm/model_executor/model_loader/base_loader.py:101`

如果存在 online quant，会调用：

```python
finalize_layerwise_processing(model, model_config)
```

位置：`code/vllm/vllm/model_executor/model_loader/base_loader.py:75` 到 `code/vllm/vllm/model_executor/model_loader/base_loader.py:78`

含义是：

```text
online quant 通常不是直接从 checkpoint 读取 packed 权重，而是在加载 fp16/bf16 权重过程中或之后按层量化；
因此它需要额外的 layer-wise finalize 流程，避免峰值显存过高并生成量化后的参数格式。
```

---

## 15. 量化如何影响 kernel 和编译配置

量化不仅影响权重加载，也会影响 kernel 选择和编译配置。

### 15.1 Linear.forward 调用 quant_method.apply

所有 Linear 层 forward 最终会调用：

```python
output = self.quant_method.apply(self, x, bias)
```

例如 `ReplicatedLinear.forward()`：

位置：`code/vllm/vllm/model_executor/layers/linear.py:372` 到 `code/vllm/vllm/model_executor/layers/linear.py:383`

`ColumnParallelLinear.forward()`：

位置：`code/vllm/vllm/model_executor/layers/linear.py:548` 到 `code/vllm/vllm/model_executor/layers/linear.py:566`

所以量化方法真正接管 forward 的位置是：

```text
quant_method.apply(layer, input, bias)
```

不同 quant method 可以在这里调用不同 kernel。

### 15.2 VllmConfig 根据量化打开 custom op

`VllmConfig.__post_init__()` 后面会根据 quant_config 判断是否有 blocked weights：

```python
if self.quant_config is not None:
    if hasattr(self.quant_config, "weight_block_size"):
        return self.quant_config.weight_block_size is not None
    elif hasattr(self.quant_config, "has_blocked_weights"):
        return self.quant_config.has_blocked_weights()
```

位置：`code/vllm/vllm/config/vllm.py:1126` 到 `code/vllm/vllm/config/vllm.py:1132`

如果有 blocked weights，会打开 `quant_fp8` custom op：

```python
if has_blocked_weights():
    custom_ops = self.compilation_config.custom_ops
    if "-quant_fp8" not in custom_ops:
        custom_ops.append("+quant_fp8")
```

位置：`code/vllm/vllm/config/vllm.py:1563` 到 `code/vllm/vllm/config/vllm.py:1567`

这说明量化信息会继续影响编译/算子选择，不只是加载时用一次。

---

## 16. 特殊 loader 和特殊格式的定位

### 16.1 BitsAndBytesModelLoader

`load_format="bitsandbytes"` 会选择 `BitsAndBytesModelLoader`。

位置：`code/vllm/vllm/model_executor/model_loader/__init__.py:50` 到 `code/vllm/vllm/model_executor/model_loader/__init__.py:65`

另外，`EngineArgs.create_load_config()` 中如果 `quantization == "bitsandbytes"`，会强制：

```python
self.load_format = "bitsandbytes"
```

位置：`code/vllm/vllm/engine/arg_utils.py:1652` 到 `code/vllm/vllm/engine/arg_utils.py:1655`

在 `create_engine_config()` 中也有同类保护：

```python
if model_config.quantization == "bitsandbytes":
    self.quantization = self.load_format = "bitsandbytes"
```

位置：`code/vllm/vllm/engine/arg_utils.py:2117` 到 `code/vllm/vllm/engine/arg_utils.py:2120`

因此 bitsandbytes 既是量化方法，也是专门的加载格式。

### 16.2 ShardedStateLoader

`load_format="sharded_state"` 或 `runai_streamer_sharded` 会使用 `ShardedStateLoader`。

它用于已经按 vLLM 运行时分片保存的 checkpoint，通常不需要像 HF checkpoint 那样再做完整的 QKV/GateUp 名称映射和 TP 切片。

选择位置：`code/vllm/vllm/model_executor/model_loader/__init__.py:50` 到 `code/vllm/vllm/model_executor/model_loader/__init__.py:65`

### 16.3 TensorizerLoader

`load_format="tensorizer"` 会使用 `TensorizerLoader`。

`EngineArgs.create_load_config()` 中会把 tensorizer 参数整理到：

```text
model_loader_extra_config["tensorizer_config"]
```

位置：`code/vllm/vllm/engine/arg_utils.py:1656` 到 `code/vllm/vllm/engine/arg_utils.py:1665`

### 16.4 DummyModelLoader

`load_format="dummy"` 会使用 `DummyModelLoader`。

`GPUModelRunner.load_model(load_dummy_weights=True)` 会临时设置：

```python
self.load_config.load_format = "dummy"
```

位置：`code/vllm/vllm/v1/worker/gpu_model_runner.py:5161` 到 `code/vllm/vllm/v1/worker/gpu_model_runner.py:5163`

这种路径通常用于 profiling、dummy run 或不需要真实权重的初始化流程。

---

## 17. 完整链路图

### 17.1 普通 safetensors / HF 权重加载

```text
用户参数：
  --model Qwen/... --load-format auto/safetensors/hf

配置阶段：
  EngineArgs.create_engine_config()
    → LoadConfig(load_format=...)
    → VllmConfig(load_config=...)

执行层：
  GPUModelRunner.load_model()
    → get_model_loader(vllm_config.load_config)
    → DefaultModelLoader.load_model()

模型实例化：
  initialize_model(vllm_config)
    → get_model_architecture(model_config)
    → model_class(vllm_config=vllm_config)
    → LinearBase(..., quant_config=vllm_config.quant_config)

权重准备：
  DefaultModelLoader._prepare_weights()
    → download_weights_from_hf() 或本地 glob
    → filter_duplicate_safetensors_files()
    → hf_weights_files

权重读取：
  safetensors_weights_iterator(hf_weights_files)
    → yield (name, tensor)

模型加载：
  model.load_weights(weights)
    → AutoWeightsLoader 或模型自定义 load_weights
    → param.weight_loader(param, tensor, shard_id?)
    → param.data.copy_ 或 BasevLLMParameter.load_*

后处理：
  process_weights_after_loading()
    → quant_method.process_weights_after_loading()
    → attention.process_weights_after_loading()

返回：
  model.eval()
```

### 17.2 量化 checkpoint 加载

```text
用户参数或 HF config：
  --quantization fp8 / awq / gptq / compressed-tensors / bitsandbytes / ...
  或 config.json: quantization_config / compression_config

配置阶段：
  VllmConfig.__post_init__()
    → get_quant_config(model_config, load_config)
    → get_quantization_config(method)
    → QuantizationConfig.from_config(...)
    → capability / dtype 校验
    → vllm_config.quant_config

模型实例化：
  initialize_model(vllm_config)
    → configure_quant_config(quant_config, model_class)
    → model_class(vllm_config=...)
    → LinearBase(quant_config=quant_config)
    → quant_config.get_quant_method(layer, prefix)
    → quant_method.create_weights(...)

权重加载：
  checkpoint tensor
    → model.load_weights()
    → param.weight_loader / weight_loader_v2
    → packed weight / scale / zero_point / block scale 参数

后处理：
  quant_method.process_weights_after_loading()
    → repack / transpose / online quant / kernel layout conversion

forward：
  Linear.forward()
    → quant_method.apply(layer, x, bias)
```

### 17.3 bitsandbytes 特殊链路

```text
--quantization bitsandbytes
  → EngineArgs.create_load_config(): load_format = "bitsandbytes"
  → get_model_loader(): BitsAndBytesModelLoader
  → BitsAndBytesConfig
  → bitsandbytes 4bit shard 特殊切片逻辑
```

在 `MergedColumnParallelLinear.weight_loader()` 中能看到 bitsandbytes 4bit 的特殊 shard 修正：

```text
adjust_bitsandbytes_4bit_shard()
```

位置：`code/vllm/vllm/model_executor/layers/linear.py:94` 到 `code/vllm/vllm/model_executor/layers/linear.py:108`

---

## 18. 关键边界：谁负责什么

### 18.1 LoadConfig / ModelLoader 负责

```text
决定用哪个 loader；
查找本地权重文件；
从 HF / ModelScope 下载权重；
过滤重复或无关文件；
构造权重 iterator；
调用 model.load_weights()；
触发加载后处理。
```

### 18.2 模型 load_weights 负责

```text
识别 checkpoint 名称；
跳过无用权重；
处理 HF 名称和 vLLM 名称差异；
处理 q/k/v、gate/up、专家权重等映射；
处理 PP 当前 rank 不存在的参数；
找到目标 parameter 并调用 weight_loader。
```

### 18.3 parameter / layer weight_loader 负责

```text
按 TP rank 切片；
按 fused shard_id 写入对应片段；
处理 packed_dim / packed_factor；
处理 block quant scale shape；
处理 bitsandbytes / Marlin 等特殊布局；
最终把 tensor 写入参数。
```

### 18.4 QuantizationConfig / QuantizeMethod 负责

```text
从 checkpoint config 或用户配置创建量化配置；
校验硬件 capability 和 dtype；
决定哪些 layer 使用哪种 quant method；
创建 packed weights / scale / zero_point 参数；
加载后做 repack / online quant / layout 转换；
forward 时调用对应 quant kernel。
```

### 18.5 不属于权重加载层的职责

```text
不负责请求调度；
不负责 KV cache block 分配；
不负责一次 batch 的 input_ids / positions / attention metadata 构造；
不负责采样；
不负责 detokenize 和 RequestOutput 构造。
```

这些分别属于 Scheduler、Worker KV cache 初始化、ModelRunner、Sampler、OutputProcessor。

---

## 19. 小结

权重加载可以理解为三层协作：

```text
文件层：
  LoadConfig / ModelLoader / weight_utils
  负责找到权重文件，并产出 (name, tensor) 流。

模型映射层：
  model.load_weights() / AutoWeightsLoader / WeightsMapper
  负责把 checkpoint 名称映射到 vLLM 参数名称。

参数解释层：
  Linear / parameter.weight_loader / QuantizeMethod
  负责 TP 切片、fused 参数切片、量化 packed 格式和后处理。
```

追踪一个权重为什么加载失败或形状不匹配时，建议按这个顺序看：

```text
1. load_format 选中了哪个 ModelLoader？
2. _prepare_weights() 找到了哪些文件？
3. iterator yield 出来的 name 和 tensor shape 是什么？
4. model.load_weights() 是否改名或跳过了这个 name？
5. 目标 param 是否存在？
6. param.weight_loader 是哪个 layer 挂上的？
7. 是否涉及 TP、QKV/GateUp fused、PP missing layer？
8. 是否涉及 quant_config、packed_factor、scale、zero_point、block scale？
9. process_weights_after_loading() 是否又改变了参数格式？
```

这也是量化接入的核心判断：

```text
同一个 checkpoint tensor，在非量化路径下可能只是 copy_；
在量化路径下，它可能被切片、解包、重排、repack，或作为 scale / zero_point 辅助参数参与 kernel 选择。
```
