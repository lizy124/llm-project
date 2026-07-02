# 04. LoadConfig 如何决定模型加载方式？

源码位置：

- `D:\lzy\project\kv_pool\code\vllm\vllm\config\load.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\__init__.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\base_loader.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\default_loader.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\dummy_loader.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\sharded_state_loader.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\tensorizer_loader.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\bitsandbytes_loader.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\runai_streamer_loader.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\modelexpress_loader.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\model_executor\model_loader\weight_utils.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu_model_runner.py`
- `D:\lzy\project\kv_pool\code\vllm\vllm\v1\worker\gpu\model_runner.py`

本问题关注：`LoadConfig` 如何描述“权重从哪里来、用什么格式读、读到什么设备、是否使用特殊加载器”；`get_model_loader()` 如何把 `load_format` 映射成具体 loader；默认 loader 如何查找 safetensors / PyTorch / npcache / Mistral 权重；特殊 loader 如何处理 dummy、sharded state、tensorizer、bitsandbytes、Run:ai Streamer、ModelExpress 等加载路径。

---

## 1. 一句话回答

`LoadConfig` 不直接加载权重，它负责把“加载策略”写进配置；真正执行时，Worker / ModelRunner 通过 `get_model_loader(load_config)` 选择一个 `BaseModelLoader` 子类，再由这个 loader 完成模型实例化、权重文件准备、权重迭代、`model.load_weights()` 和加载后处理。

主链路是：

```text
Worker.load_model()
  → ModelRunner.load_model()
  → get_model_loader(load_config)
  → BaseModelLoader 子类
  → loader.load_model(vllm_config, model_config)
  → initialize_model()
  → loader.load_weights(model, model_config)
  → model.load_weights(weights_iterator)
  → process_weights_after_loading()
  → model.eval()
```

所以：

```text
LoadConfig 是加载策略；
get_model_loader 是策略分发器；
BaseModelLoader.load_model 是通用加载骨架；
DefaultModelLoader / 特殊 loader 是具体权重读取实现；
模型自己的 load_weights() 是参数名映射和张量落位的最终入口。
```

---

## 2. LoadConfig 关心什么

`LoadConfig` 定义在 `load.py:25` 到 `load.py:148`。

核心字段可以分成几类。

### 2.1 加载格式：load_format

```python
load_format: str | LoadFormats = "auto"
```

位置：`load.py:29`

它决定最终选择哪个 loader，以及默认 loader 内部应该用哪类权重文件。

内置格式包括：

```text
auto
hf
safetensors
fastsafetensors
instanttensor
pt
npcache
mistral
dummy
tensorizer
bitsandbytes
sharded_state
runai_streamer
runai_streamer_sharded
modelexpress
```

`load_format` 会被 validator 转成小写：

```python
@field_validator("load_format", mode="after")
def _lowercase_load_format(cls, load_format: str) -> str:
    return load_format.lower()
```

位置：`load.py:134` 到 `load.py:136`

所以命令行里写 `SAFETENSORS`、`Safetensors`，最后都会变成 `safetensors`。

### 2.2 权重来源：download_dir / ignore_patterns

```python
download_dir: str | None = None
ignore_patterns: list[str] | str = Field(default_factory=lambda: ["original/**/*"])
```

位置：`load.py:59` 和 `load.py:99`

含义是：

```text
download_dir：远端 Hugging Face / ModelScope 权重下载到哪里；
ignore_patterns：下载或查找时忽略哪些文件，默认跳过 original/**/*。
```

`ignore_patterns` 不为空且不是默认值时，会打印日志：

```python
logger.info(
    "Ignoring the following patterns when downloading weights: %s",
    ignore_patterns,
)
```

位置：`load.py:142` 到 `load.py:146`

### 2.3 safetensors 读取策略

相关字段：

```python
safetensors_load_strategy: str | None = None
safetensors_prefetch_num_threads: int = 8
safetensors_prefetch_block_size: int = 16 * 1024 * 1024
```

位置：`load.py:62` 到 `load.py:92`

`safetensors_load_strategy` 支持：

```text
None：默认 lazy mmap；如果检测到 NFS 且 checkpoint 能放进内存，可能自动 prefetch；
lazy：明确使用 mmap lazy loading，不做 NFS auto-prefetch；
eager：先整体读入 CPU 内存，适合网络文件系统；
prefetch：先把 checkpoint 读进 OS page cache；
torchao：用于 torchao 量化 safetensors 的重建。
```

这个字段不会在 `LoadConfig` 里执行读取，它会传给 `safetensors_weights_iterator()`。

位置：`default_loader.py:254` 到 `default_loader.py:265`

### 2.4 loader 额外配置

```python
model_loader_extra_config: dict | TensorizerConfig = Field(default_factory=dict)
```

位置：`load.py:93`

它是给具体 loader 的扩展参数，不同 loader 接受的 key 不一样。

例如默认 loader 只接受：

```text
enable_multithread_load
num_threads
enable_weights_track
```

位置：`default_loader.py:78` 到 `default_loader.py:91`

`ShardedStateLoader` 接受：

```text
pattern
```

位置：`sharded_state_loader.py:43` 到 `sharded_state_loader.py:54`

`RunaiModelStreamerLoader` 接受：

```text
distributed
concurrency
memory_limit
```

位置：`runai_streamer_loader.py:30` 到 `runai_streamer_loader.py:65`

`TensorizerLoader` 使用 `tensorizer_config`，并拒绝让用户覆盖 `device`、`dtype`、`mode`。

位置：`tensorizer_loader.py:30` 到 `tensorizer_loader.py:54`

### 2.5 加载设备和 PyTorch map_location

```python
device: str | None = None
pt_load_map_location: str | dict[str, str] = "cpu"
```

位置：`load.py:96` 和 `load.py:105`

区别是：

```text
device：模型权重最终加载到哪个 device；默认跟 device_config.device 一致。
pt_load_map_location：torch.load 读取 .pt / .bin 时用的 map_location。
```

在通用加载骨架里，最终加载设备这样确定：

```python
load_device = (
    device_config.device if load_config.device is None else load_config.device
)
target_device = torch.device(load_device)
```

位置：`base_loader.py:47` 到 `base_loader.py:52`

在 PyTorch 权重 iterator 里，`pt_load_map_location` 会传给 `pt_weights_iterator()` 或 `multi_thread_pt_weights_iterator()`。

位置：`default_loader.py:267` 到 `default_loader.py:281`

---

## 3. Worker 侧什么时候使用 LoadConfig

V1 GPU 主路径中，模型加载发生在 `GPUModelRunner.load_model()`。

入口：`gpu_model_runner.py:5143`

核心代码：

```python
if load_dummy_weights:
    self.load_config.load_format = "dummy"
model_loader = get_model_loader(self.load_config)
self.model = model_loader.load_model(
    vllm_config=self.vllm_config, model_config=self.model_config
)
```

位置：`gpu_model_runner.py:5161` 到 `gpu_model_runner.py:5166`

较新的 `v1/worker/gpu/model_runner.py` 也有同样模式：

```python
if load_dummy_weights:
    self.load_config.load_format = "dummy"
model_loader = get_model_loader(self.vllm_config.load_config)
self.model = model_loader.load_model(...)
```

位置：`model_runner.py:274` 到 `model_runner.py:286`

这说明：

```text
LoadConfig 在 Engine / Worker 初始化阶段已经构造好；
真正模型加载时，ModelRunner 读取 LoadConfig；
如果外部要求 load_dummy_weights，会临时把 load_format 改成 dummy；
然后统一走 get_model_loader()。
```

---

## 4. get_model_loader 如何选择 loader

`get_model_loader()` 定义在 `model_loader/__init__.py:122` 到 `model_loader/__init__.py:127`。

核心代码：

```python
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    load_format = load_config.load_format
    if load_format not in _LOAD_FORMAT_TO_MODEL_LOADER:
        raise ValueError(f"Load format `{load_format}` is not supported")
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)
```

位置：`__init__.py:122` 到 `__init__.py:127`

映射表是：

```text
auto                   → DefaultModelLoader
hf                     → DefaultModelLoader
safetensors            → DefaultModelLoader
fastsafetensors        → DefaultModelLoader
instanttensor          → DefaultModelLoader
pt                     → DefaultModelLoader
npcache                → DefaultModelLoader
mistral                → DefaultModelLoader
bitsandbytes           → BitsAndBytesModelLoader
dummy                  → DummyModelLoader
sharded_state          → ShardedStateLoader
runai_streamer         → RunaiModelStreamerLoader
runai_streamer_sharded → ShardedStateLoader
tensorizer             → TensorizerLoader
modelexpress           → ModelExpressModelLoader
```

位置：`__init__.py:33` 到 `__init__.py:66`

所以 `LoadConfig.load_format` 分两层起作用：

```text
第一层：选择 loader 类；
第二层：如果选中 DefaultModelLoader，再决定它内部查什么文件、用什么 iterator。
```

---

## 5. 对外统一入口 get_model()

`get_model()` 是一个便捷入口：

```python
def get_model(*, vllm_config, model_config=None, prefix="", load_config=None):
    loader = get_model_loader(load_config or vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )
```

位置：`__init__.py:130` 到 `__init__.py:142`

它做了三件事：

```text
1. 从 load_config 或 vllm_config.load_config 里取加载策略；
2. 从 vllm_config.model_config 里取模型配置；
3. 调用 loader.load_model()。
```

CPU runner、spec decode draft model、隐藏状态提取等路径会直接用 `get_model()`；GPU 主路径通常显式调用 `get_model_loader()`。

---

## 6. BaseModelLoader.load_model 的通用骨架

`BaseModelLoader` 是所有 loader 的抽象基类。

位置：`base_loader.py:25` 到 `base_loader.py:82`

它要求子类实现：

```python
def download_model(self, model_config: ModelConfig) -> None

def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None
```

位置：`base_loader.py:31` 到 `base_loader.py:40`

通用 `load_model()` 流程是：

```python
with set_default_torch_dtype(model_config.dtype):
    with target_device:
        model = initialize_model(...)

    log_model_inspection(model)
    self.load_weights(model, model_config)

    if _has_online_quant(model):
        finalize_layerwise_processing(model, model_config)

    process_weights_after_loading(model, model_config, target_device)

return model.eval()
```

位置：`base_loader.py:53` 到 `base_loader.py:82`

可以拆成五步：

```text
1. 设置默认 dtype；
2. 在目标 device 上 instantiate 模型结构；
3. 调用具体 loader 的 load_weights() 读权重；
4. 对在线量化 / attention / torchao 等做加载后处理；
5. 返回 eval() 模型。
```

注意：

```text
BaseModelLoader 负责“骨架”；
子类 loader 负责“权重来源和读取方式”；
模型类自己的 load_weights() 负责“参数名匹配和张量切片落位”。
```

---

## 7. initialize_model 如何实例化模型

`initialize_model()` 定义在 `utils.py:40` 到 `utils.py:97`。

主路径：

```python
if model_class is None:
    model_class, _ = get_model_architecture(model_config)

if vllm_config.quant_config is not None:
    configure_quant_config(vllm_config.quant_config, model_class)

with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
    model = model_class(vllm_config=vllm_config, prefix=prefix)
    record_metadata_for_reloading(model)
    return model
```

位置：`utils.py:49` 到 `utils.py:64`

这说明模型类不是由 `LoadConfig` 决定的，而是由 `ModelConfig` / Hugging Face config / registry 决定的。

模型类解析链路是：

```text
model_config.hf_config.architectures
  → model_config.registry.resolve_model_cls(...)
  → get_model_architecture()
  → model_class
```

位置：`utils.py:182` 到 `utils.py:215`

如果请求 embedding / classify，会在解析后包装模型类：

```text
convert_type == embed    → as_embedding_model(model_cls)
convert_type == classify → as_seq_cls_model(model_cls)
```

位置：`utils.py:203` 到 `utils.py:211`

所以要区分：

```text
ModelConfig 决定加载哪个模型结构；
LoadConfig 决定权重如何读取。
```

---

## 8. DefaultModelLoader 支持哪些 load_format

`DefaultModelLoader` 是最常见 loader。

位置：`default_loader.py:43`

它处理这些格式：

```text
auto
hf
safetensors
fastsafetensors
instanttensor
pt
npcache
mistral
```

这些格式都走同一个 loader，但在 `_prepare_weights()` 和 `_get_weights_iterator()` 中分支。

---

## 9. DefaultModelLoader 第一阶段：准备权重文件

入口：`default_loader.py:97`

```python
def _prepare_weights(...):
```

返回：

```text
hf_folder：本地权重目录；
hf_weights_files：实际要读的权重文件列表；
use_safetensors：后续是否按 safetensors 读取。
```

位置：`default_loader.py:97` 到 `default_loader.py:209`

### 9.1 ModelScope / 本地目录判断

```python
model_name_or_path = (
    maybe_download_from_modelscope(model_name_or_path, revision)
    or model_name_or_path
)
is_local = os.path.isdir(model_name_or_path)
```

位置：`default_loader.py:108` 到 `default_loader.py:113`

含义：

```text
如果 ModelScope 可处理，先尝试 ModelScope；
否则保留原始 model path / HF repo id；
如果是本地目录，直接查本地文件；
如果不是本地目录，后续从 HF 下载。
```

### 9.2 auto 会先判断 Mistral 官方格式

```python
if load_format == "auto":
    load_format = (
        "mistral"
        if len(list_filtered_repo_files(... allow_patterns=["consolidated*.safetensors"])) > 0
        else "hf"
    )
```

位置：`default_loader.py:120` 到 `default_loader.py:133`

也就是说：

```text
auto 不只是 safetensors 优先；
auto 会先看是否存在 consolidated*.safetensors；
如果有，按 Mistral 官方 consolidated 格式处理；
否则按 hf 格式处理。
```

### 9.3 不同格式对应不同 allow_patterns

核心分支：

```text
hf                         → *.safetensors, *.bin
safetensors/fast/instant   → *.safetensors
mistral                    → consolidated*.safetensors
pt                         → *.pt
npcache                    → *.bin
```

位置：`default_loader.py:135` 到 `default_loader.py:153`

如果 `fall_back_to_pt` 为真，还会追加：

```python
allow_patterns += ["*.pt"]
```

位置：`default_loader.py:155` 到 `default_loader.py:156`

`fall_back_to_pt` 默认来自模型对象：

```python
fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True)
```

位置：`default_loader.py:293` 到 `default_loader.py:299`

所以某些模型可以通过自身属性禁止或允许 fallback 到 `.pt`。

### 9.4 远端下载或本地查找

非本地路径会调用：

```python
hf_folder = download_weights_from_hf(
    model_name_or_path,
    self.load_config.download_dir,
    allow_patterns,
    revision,
    subfolder=subfolder,
    ignore_patterns=self.load_config.ignore_patterns,
)
```

位置：`default_loader.py:161` 到 `default_loader.py:169`

本地路径则直接使用：

```python
hf_folder = model_name_or_path
```

位置：`default_loader.py:170` 到 `default_loader.py:171`

### 9.5 按 pattern 找到第一组可用权重

```python
for pattern in allow_patterns:
    hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
    if len(hf_weights_files) > 0:
        if pattern.endswith(".safetensors"):
            use_safetensors = True
        break
```

位置：`default_loader.py:176` 到 `default_loader.py:183`

关键点：

```text
不是把所有 pattern 的文件都混在一起读；
而是按 allow_patterns 顺序找，找到第一批非空文件就停止。
```

例如 `hf`：

```text
先找 *.safetensors；
找到了就读 safetensors；
没找到才找 *.bin；
如果 fall_back_to_pt，还可能找 *.pt。
```

### 9.6 safetensors 去重和 index 过滤

如果使用 safetensors：

```python
hf_weights_files = filter_duplicate_safetensors_files(
    hf_weights_files, hf_folder, index_file
)
```

位置：`default_loader.py:184` 到 `default_loader.py:200`

原因是有些模型目录里同时存在：

```text
model-00001-of-000xx.safetensors
model.safetensors.index.json
consolidated.safetensors
```

如果混读，可能重复加载或加载错误，所以需要按 index 文件过滤。

如果不是 safetensors：

```python
hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)
```

位置：`default_loader.py:201` 到 `default_loader.py:202`

它会过滤掉 optimizer、training state 等推理不需要的文件。

---

## 10. DefaultModelLoader 第二阶段：选择权重 iterator

入口：`default_loader.py:211`

```python
def _get_weights_iterator(self, source):
```

它会先调用 `_prepare_weights()`，再根据格式返回不同 iterator。

### 10.1 npcache

```python
if self.load_config.load_format == "npcache":
    weights_iterator = np_cache_weights_iterator(...)
```

位置：`default_loader.py:223` 到 `default_loader.py:232`

限制：

```text
npcache 当前只支持 *.bin checkpoint。
```

位置：`default_loader.py:223` 到 `default_loader.py:225`

### 10.2 safetensors / fastsafetensors / instanttensor

如果 `use_safetensors` 为真：

```text
fastsafetensors → fastsafetensors_weights_iterator()
instanttensor   → instanttensor_weights_iterator()
其他            → safetensors_weights_iterator() 或 multi_thread_safetensors_weights_iterator()
```

位置：`default_loader.py:233` 到 `default_loader.py:265`

其中普通 safetensors 会接收：

```text
use_tqdm_on_load
safetensors_load_strategy
local_expert_ids
safetensors_prefetch_num_threads
safetensors_prefetch_block_size
```

位置：`default_loader.py:254` 到 `default_loader.py:264`

### 10.3 PyTorch bin / pt

如果不是 safetensors：

```text
enable_multithread_load=True  → multi_thread_pt_weights_iterator()
否则                          → pt_weights_iterator()
```

位置：`default_loader.py:266` 到 `default_loader.py:281`

这里会使用：

```text
pt_load_map_location
```

用于 `torch.load()` 的 `map_location`。

### 10.4 权重名前缀

最后统一加 prefix：

```python
return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
```

位置：`default_loader.py:283` 到 `default_loader.py:286`

这让 secondary weights 或子模块加载时可以把权重名挂到某个前缀下。

---

## 11. DefaultModelLoader 第三阶段：主权重和 secondary weights

`get_all_weights()` 会先读主权重，再读模型声明的 secondary weights。

```python
primary_weights = DefaultModelLoader.Source(
    model_config.model,
    model_config.revision,
    prefix="",
    fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
    allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
)
yield from self._get_weights_iterator(primary_weights)

secondary_weights = getattr(model, "secondary_weights", ())
for source in secondary_weights:
    yield from self._get_weights_iterator(source)
```

位置：`default_loader.py:288` 到 `default_loader.py:307`

这说明模型实现可以影响 loader 行为：

```text
fall_back_to_pt_during_load：是否允许 fallback 到 .pt；
allow_patterns_overrides：强制只按模型指定 pattern 查权重；
secondary_weights：额外加载第二组权重，并可带 prefix。
```

---

## 12. DefaultModelLoader 第四阶段：load_weights

入口：`default_loader.py:381`

核心代码：

```python
if model_config.quantization == "torchao":
    quant_config = get_quant_config(model_config, self.load_config)
    if quant_config.is_checkpoint_torchao_serialized ...:
        self.load_config.safetensors_load_strategy = "torchao"

self._init_ep_weight_filter(model_config)
loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

位置：`default_loader.py:381` 到 `default_loader.py:395`

这里有三个关键点。

### 12.1 torchao 可能改写 safetensors 策略

如果是 torchao checkpoint，并且版本满足要求，loader 会把：

```text
safetensors_load_strategy = torchao
```

位置：`default_loader.py:383` 到 `default_loader.py:390`

这样底层 safetensors iterator 会按 torchao tensor subclass 方式重建权重。

### 12.2 EP weight filter 会提前跳过非本地 expert 权重

`_init_ep_weight_filter()` 会在这些条件都满足时计算本 rank 需要的 expert ids：

```text
model_config.is_moe
enable_expert_parallel
enable_ep_weight_filter
未启用 EPLB
```

位置：`default_loader.py:318` 到 `default_loader.py:380`

结果保存在：

```python
self.local_expert_ids
```

然后传给 `safetensors_weights_iterator()`。

位置：`default_loader.py:254` 到 `default_loader.py:264`

含义是：

```text
MoE + Expert Parallel 下，每个 rank 可以只读取自己负责的 expert 权重，减少 IO 和内存压力。
```

### 12.3 strict 权重检查

加载后会根据 `loaded_weights` 检查哪些参数没有从 checkpoint 初始化：

```python
weights_to_load = {name for name, _ in model.named_parameters()}
weights_not_loaded = weights_to_load - loaded_weights
if weights_not_loaded:
    raise ValueError(...)
```

位置：`default_loader.py:401` 到 `default_loader.py:437`

默认只对非量化模型启用：

```python
default_enable_weights_track = (
    model_config.quantization is None and loaded_weights is not None
)
```

位置：`default_loader.py:401` 到 `default_loader.py:410`

也可以用 `model_loader_extra_config.enable_weights_track` 覆盖。

---

## 13. DummyModelLoader：不读真实权重

`dummy` 格式映射到 `DummyModelLoader`。

位置：`dummy_loader.py:22`

它的 `download_model()` 什么都不做：

```python
def download_model(self, model_config):
    pass
```

位置：`dummy_loader.py:33` 到 `dummy_loader.py:34`

`load_weights()` 会遍历模型模块：

```python
for layer in model.modules():
    info = get_layerwise_info(layer)
    if info.can_load():
        self._process_online_quant_layer(layer, info)
    else:
        initialize_dummy_weights(layer, model_config)
```

位置：`dummy_loader.py:36` 到 `dummy_loader.py:44`

用途：

```text
profile / benchmark / 内存评估；
只需要模型结构和随机参数；
不需要访问真实 checkpoint。
```

注意：如果 `load_dummy_weights=True`，GPU ModelRunner 会直接把 `load_format` 改成 `dummy`。

位置：`gpu_model_runner.py:5161` 到 `gpu_model_runner.py:5163`

---

## 14. ShardedStateLoader：每个 TP rank 只读自己的 shard

`sharded_state` 和 `runai_streamer_sharded` 都映射到 `ShardedStateLoader`。

位置：`__init__.py:61` 到 `__init__.py:64`

默认文件名模式：

```python
DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"
```

位置：`sharded_state_loader.py:38`

可以通过：

```text
model_loader_extra_config.pattern
```

覆盖。

位置：`sharded_state_loader.py:43` 到 `sharded_state_loader.py:54`

### 14.1 查找本 rank 文件

加载时先取 tensor parallel rank：

```python
rank = get_tensor_model_parallel_rank()
pattern = os.path.join(
    local_model_path,
    self.pattern.format(rank=rank, part="*"),
)
```

位置：`sharded_state_loader.py:118` 到 `sharded_state_loader.py:122`

如果是 S3，会用 S3 glob；否则用本地 glob。

位置：`sharded_state_loader.py:125` 到 `sharded_state_loader.py:129`

找不到就报错：

```text
only pre-sharded checkpoints are currently supported
```

位置：`sharded_state_loader.py:130` 到 `sharded_state_loader.py:135`

### 14.2 直接 copy 到 state_dict

```python
state_dict = self._filter_subtensors(model.state_dict())
for key, tensor in self.iterate_over_files(filepaths):
    param_data = state_dict[key].data
    ...
    param_data.copy_(tensor)
    state_dict.pop(key)
```

位置：`sharded_state_loader.py:137` 到 `sharded_state_loader.py:155`

最后如果还有没加载的参数：

```python
if state_dict:
    raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")
```

位置：`sharded_state_loader.py:161` 到 `sharded_state_loader.py:162`

它和 DefaultModelLoader 的主要差异是：

```text
DefaultModelLoader：读取完整或 HF shard checkpoint，再由模型 load_weights 做 TP 切片/映射；
ShardedStateLoader：checkpoint 已经按 rank 预切好，每个 worker 只读自己的 state dict shard。
```

---

## 15. TensorizerLoader：tensorizer 序列化格式

`tensorizer` 格式映射到 `TensorizerLoader`。

位置：`tensorizer_loader.py:43`

初始化时会从 `model_loader_extra_config` 构造 `TensorizerConfig`：

```python
if isinstance(load_config.model_loader_extra_config, TensorizerConfig):
    self.tensorizer_config = load_config.model_loader_extra_config
else:
    validate_config(load_config.model_loader_extra_config)
    self.tensorizer_config = TensorizerConfig(
        **load_config.model_loader_extra_config["tensorizer_config"]
    )
```

位置：`tensorizer_loader.py:47` 到 `tensorizer_loader.py:54`

它拒绝用户配置这些字段：

```text
device
dtype
mode
```

位置：`tensorizer_loader.py:30` 到 `tensorizer_loader.py:40`

原因是这些由 vLLM 自己决定。

### 15.1 TP 下按 rank 改写 URI

```python
if parallel_config.tensor_parallel_size > 1:
    self.tensorizer_config.tensorizer_uri = (
        self.tensorizer_config.tensorizer_uri % get_tensor_model_parallel_rank()
    )
```

位置：`tensorizer_loader.py:121` 到 `tensorizer_loader.py:127`

也就是说 tensorizer URI 可以包含 rank 占位符。

### 15.2 vLLM tensorized 和普通 tensorizer 路径

如果是 vLLM tensorized：

```python
model = init_tensorizer_model(...)
self.load_weights(model, model_config)
return model
```

位置：`tensorizer_loader.py:129` 到 `tensorizer_loader.py:139`

`load_weights()` 内部会调用：

```python
deserialize_tensorizer_model(model, tensorizer_config)
```

位置：`tensorizer_loader.py:103` 到 `tensorizer_loader.py:113`

如果不是 vLLM tensorized，则走 `_load_model_serialized_cpu()`：

```text
initialize_model()
model.load_weights(tensorizer_weights_iterator(...))
```

位置：`tensorizer_loader.py:68` 到 `tensorizer_loader.py:87`

---

## 16. BitsAndBytesModelLoader：边加载边量化或读取预量化权重

`bitsandbytes` 格式映射到 `BitsAndBytesModelLoader`。

位置：`bitsandbytes_loader.py:56`

它的职责不只是读文件，还要处理：

```text
目标 module 识别；
HF 权重名到 vLLM 权重名映射；
TP shard 切分；
4bit / 8bit quant state；
预量化 checkpoint 的限制；
MoE expert quant state 融合。
```

### 16.1 权重文件查找

BNB loader 支持：

```text
*.safetensors
*.bin
*.pt
```

位置：`bitsandbytes_loader.py:119` 到 `bitsandbytes_loader.py:157`

如果是 safetensors，也会下载 / 使用 index 文件去重：

```python
hf_weights_files = filter_duplicate_safetensors_files(
    hf_weights_files, hf_folder, index_file
)
```

位置：`bitsandbytes_loader.py:133` 到 `bitsandbytes_loader.py:148`

### 16.2 bitsandbytes 依赖检查

```python
import bitsandbytes
if version.parse(bitsandbytes.__version__) < version.parse("0.46.1"):
    raise ImportError(...)
```

位置：`bitsandbytes_loader.py:199` 到 `bitsandbytes_loader.py:213`

### 16.3 预量化模型限制

如果 HF config 里声明 `quant_method == "bitsandbytes"`：

```python
self.pre_quant = True
```

位置：`bitsandbytes_loader.py:539` 到 `bitsandbytes_loader.py:546`

如果是预量化 BNB 且 tensor parallel size 大于 1，会报错：

```python
if self.pre_quant and get_tensor_model_parallel_world_size() > 1:
    raise ValueError(
        "Prequant BitsAndBytes models with tensor parallelism is not supported..."
    )
```

位置：`bitsandbytes_loader.py:548` 到 `bitsandbytes_loader.py:554`

原因是预量化模型里的 quant_states 不能和切分后的 weight tensor 正确配合。

### 16.4 load_weights 主流程

```python
self._verify_model_compatibility(model, model_config)
self._initialize_loader_state(model, model_config)
qweight_iterator, quant_state_dict = self._get_quantized_weights_iterator(...)
loaded_weights = model.load_weights(qweight_iterator)
...
stacked_quant_state_dict = self._stack_quantization_states(model, quant_state_dict)
self._bind_quant_states_to_params(model, stacked_quant_state_dict)
torch.accelerator.empty_cache()
```

位置：`bitsandbytes_loader.py:804` 到 `bitsandbytes_loader.py:836`

最终效果是：

```text
权重张量进入模型参数；
quant state 作为属性绑定到对应 parameter；
推理时量化 kernel 可以从 parameter 属性取 quant state。
```

---

## 17. RunaiModelStreamerLoader：用 Run:ai Streamer 读 safetensors

`runai_streamer` 格式映射到 `RunaiModelStreamerLoader`。

位置：`runai_streamer_loader.py:21`

它可以从这些位置读 safetensors：

```text
local FS
S3
GCS
Azure Blob Storage
Hugging Face 下载后的本地缓存
```

位置：`runai_streamer_loader.py:21` 到 `runai_streamer_loader.py:25`

### 17.1 extra config

支持：

```text
distributed：是否启用分布式 streamer；
concurrency：写入 RUNAI_STREAMER_CONCURRENCY；
memory_limit：写入 RUNAI_STREAMER_MEMORY_LIMIT。
```

位置：`runai_streamer_loader.py:30` 到 `runai_streamer_loader.py:65`

如果设置了 `AWS_ENDPOINT_URL` 但没设置 `RUNAI_STREAMER_S3_ENDPOINT`，它会补：

```python
os.environ["RUNAI_STREAMER_S3_ENDPOINT"] = aws_endpoint_url
```

位置：`runai_streamer_loader.py:67` 到 `runai_streamer_loader.py:70`

### 17.2 权重读取

```python
hf_weights_files = list_safetensors(path=hf_folder)
return runai_safetensors_weights_iterator(
    hf_weights_files, self.load_config.use_tqdm_on_load, self._is_distributed
)
```

位置：`runai_streamer_loader.py:95` 到 `runai_streamer_loader.py:119`

最终加载仍然是：

```python
model.load_weights(self._get_weights_iterator(model_weights, model_config.revision))
```

位置：`runai_streamer_loader.py:125` 到 `runai_streamer_loader.py:132`

---

## 18. ModelExpressModelLoader：外部 loader 包装层

`modelexpress` 格式映射到 `ModelExpressModelLoader`。

位置：`modelexpress_loader.py:33`

它本身是薄包装，会动态导入：

```python
_MODELEXPRESS_LOADER_MODULE = "modelexpress.engines.vllm.loader"
module = importlib.import_module(_MODELEXPRESS_LOADER_MODULE)
ModelExpressVllmLoader = module.MxModelLoader
return ModelExpressVllmLoader(load_config)
```

位置：`modelexpress_loader.py:15` 到 `modelexpress_loader.py:50`

如果没安装 `modelexpress`，会抛出明确错误：

```text
Install it with `pip install modelexpress`.
```

位置：`modelexpress_loader.py:26` 到 `modelexpress_loader.py:30`

它的 `download_model()`、`load_weights()`、`load_model()` 都委托给外部 loader。

位置：`modelexpress_loader.py:52` 到 `modelexpress_loader.py:70`

---

## 19. 插件如何注册自定义 loader

vLLM 支持通过 `register_model_loader()` 注册自定义加载器。

位置：`__init__.py:69` 到 `__init__.py:119`

用法形态：

```python
@register_model_loader("my_loader")
class MyModelLoader(BaseModelLoader):
    ...
```

注册时会检查：

```python
if not issubclass(model_loader_cls, BaseModelLoader):
    raise ValueError(...)
```

位置：`__init__.py:107` 到 `__init__.py:110`

如果注册名已经存在，会打印 warning，并覆盖原 loader。

位置：`__init__.py:99` 到 `__init__.py:111`

所以 `LoadConfig` 的 docstring 里说 “Other custom values can be supported via plugins”，具体就是通过这里扩展。

位置：`load.py:56` 到 `load.py:57`

---

## 20. 权重真正如何进入模型参数

无论是哪种 loader，最终基本都会走到模型对象的：

```python
model.load_weights(weights_iterator)
```

不同 loader 提供的 iterator 形态可能不同：

```text
DefaultModelLoader：yield (weight_name, tensor)
RunaiModelStreamerLoader：yield (weight_name, tensor)
TensorizerLoader：yield (weight_name, tensor) 或直接 deserialize_tensorizer_model
BitsAndBytesModelLoader：yield (original_name, processed_quant_weight)
ShardedStateLoader：绕过 model.load_weights，直接 copy 到 state_dict
DummyModelLoader：绕过 checkpoint，初始化随机权重
```

这很重要：

```text
LoadConfig / loader 决定“读什么文件、怎么产出 tensor”；
模型实现的 load_weights() 决定“这个 tensor 应该放进哪个 parameter，以及是否需要切片、合并、跳过”。
```

例如 QKV fused 权重、MLP gate/up 投影、TP 分片、量化 scale 等，一般不是在 `LoadConfig` 里处理，而是在模型 `load_weights()` 或量化 loader 的映射逻辑里处理。

---

## 21. 加载后处理做了什么

`BaseModelLoader.load_model()` 在 `self.load_weights()` 后还会做两类后处理。

### 21.1 online quant finalize

```python
if _has_online_quant(model):
    finalize_layerwise_processing(model, model_config)
```

位置：`base_loader.py:77` 到 `base_loader.py:78`

`_has_online_quant()` 会检查模块的 `quant_method.uses_meta_device`。

位置：`base_loader.py:95` 到 `base_loader.py:101`

### 21.2 process_weights_after_loading

```python
process_weights_after_loading(model, model_config, target_device)
```

位置：`base_loader.py:80`

它会：

```text
1. 遍历所有模块，如果 quant_method 是 QuantizeMethodBase，则调用 quant_method.process_weights_after_loading(module)；
2. 对 Attention / MLAAttention / MMEncoderAttention 调用 process_weights_after_loading；
3. torchao 模型设置 reload 相关属性。
```

位置：`utils.py:100` 到 `utils.py:131`

如果某些参数因为 CPU offload 在 CPU 上，`device_loading_context()` 会临时把它们搬到 target device 做处理，结束后再搬回去。

位置：`utils.py:134` 到 `utils.py:176`

---

## 22. download_model 和 load_weights 的边界

所有 loader 都有 `download_model()` 和 `load_weights()`，但职责不同。

```text
download_model(model_config)：只确保权重可用，通常用于预下载或缓存准备；
load_weights(model, model_config)：在已有模型结构上真正填充参数。
```

DefaultModelLoader 的 `download_model()` 只是调用 `_prepare_weights()`：

```python
def download_model(self, model_config):
    self._prepare_weights(...)
```

位置：`default_loader.py:309` 到 `default_loader.py:316`

Run:ai、tensorizer、sharded loader 也都有对应的预准备逻辑。

这说明：

```text
下载和加载是可分离的；
但常规 Worker load_model 路径通常会直接 load_weights，必要时在其中触发下载。
```

---

## 23. 不同 load_format 的最小心智模型

```text
auto：默认推荐路径；先识别 Mistral consolidated，再按 HF 权重优先 safetensors、fallback bin/pt。

hf：DefaultModelLoader；按 *.safetensors → *.bin → 可选 *.pt 查找。

safetensors：DefaultModelLoader；只查 *.safetensors，并使用 safetensors iterator。

fastsafetensors：DefaultModelLoader；只查 *.safetensors，但使用 fastsafetensors iterator。

instanttensor：DefaultModelLoader；只查 *.safetensors，但使用 InstantTensor iterator。

pt：DefaultModelLoader；查 *.pt，使用 torch.load map_location。

npcache：DefaultModelLoader；查 *.bin，并维护 numpy cache 加速后续加载。

mistral：DefaultModelLoader；查 consolidated*.safetensors，使用 consolidated.safetensors.index.json。

dummy：DummyModelLoader；不读 checkpoint，随机初始化权重。

sharded_state：ShardedStateLoader；每个 TP rank 读取 model-rank-{rank}-part-*.safetensors。

runai_streamer：RunaiModelStreamerLoader；通过 Run:ai Streamer 读 safetensors。

runai_streamer_sharded：ShardedStateLoader；用 Run:ai safetensors iterator 读预分片 checkpoint。

tensorizer：TensorizerLoader；使用 CoreWeave tensorizer 序列化格式。

bitsandbytes：BitsAndBytesModelLoader；处理 BNB 4bit/8bit 权重、量化状态和模块映射。

modelexpress：ModelExpressModelLoader；委托给 modelexpress 包提供的 vLLM loader。
```

---

## 24. 容易疑惑的点

### 24.1 LoadConfig 是否决定模型架构？

不决定。

模型架构由 `ModelConfig`、HF config 的 `architectures`、vLLM model registry、`model_impl`、`convert_type` 等决定。

`LoadConfig` 决定的是：

```text
权重文件怎么找；
用哪个 loader；
用哪种 iterator；
加载到哪个 device；
是否启用特殊加载策略。
```

### 24.2 auto 是否总是 safetensors 优先？

大体上是，但更准确地说：

```text
auto 先检查 Mistral consolidated*.safetensors；
如果存在，走 mistral；
否则转成 hf；
hf 再按 *.safetensors → *.bin → 可选 *.pt 查找。
```

位置：`default_loader.py:120` 到 `default_loader.py:183`

### 24.3 safetensors_load_strategy 是否影响所有格式？

不是。

它只在普通 safetensors iterator 路径里有意义：

```python
safetensors_weights_iterator(..., self.load_config.safetensors_load_strategy, ...)
```

位置：`default_loader.py:254` 到 `default_loader.py:265`

`pt`、`npcache`、`dummy`、`sharded_state` 等路径不使用这个字段。

### 24.4 load_format=safetensors 和 load_format=fastsafetensors 有什么区别？

它们都只查 `*.safetensors`。

区别在 iterator：

```text
safetensors      → safetensors_weights_iterator()
fastsafetensors  → fastsafetensors_weights_iterator()
instanttensor    → instanttensor_weights_iterator()
```

位置：`default_loader.py:233` 到 `default_loader.py:265`

### 24.5 sharded_state 是否能读普通 HF checkpoint？

不能。

`ShardedStateLoader` 明确要求预分片文件：

```text
model-rank-{rank}-part-{part}.safetensors
```

找不到本 rank 文件就报错。

位置：`sharded_state_loader.py:118` 到 `sharded_state_loader.py:135`

### 24.6 为什么有些 loader 不调用 model.load_weights()？

因为权重格式已经和模型 state_dict 一一对应，或根本没有真实 checkpoint。

```text
ShardedStateLoader：直接按 key copy 到 state_dict；
DummyModelLoader：直接初始化 module 参数；
TensorizerLoader：vLLM tensorized 路径可能直接 deserialize 到模型。
```

### 24.7 model_loader_extra_config 可以随便传吗？

不能。

每个 loader 都会校验自己支持的 key。比如 DefaultModelLoader 遇到未知 key 会报错：

```python
unexpected_keys = set(extra_config.keys()) - allowed_keys
if unexpected_keys:
    raise ValueError(...)
```

位置：`default_loader.py:78` 到 `default_loader.py:91`

---

## 25. 总结

`LoadConfig` 到模型权重加载的主链路可以压缩成：

```text
LoadConfig(load_format, download_dir, strategy, extra_config, device, ...)
  → get_model_loader(load_config)
  → 选择 BaseModelLoader 子类
  → BaseModelLoader.load_model()
  → initialize_model() 由 ModelConfig 决定模型类
  → loader.load_weights()
  → 准备权重文件 / 权重 iterator
  → model.load_weights() 或 loader 直接写入 state_dict
  → finalize_layerwise_processing()
  → process_weights_after_loading()
  → model.eval()
```

如果只记住一句话：

```text
LoadConfig 是权重加载策略的配置入口；它不决定模型长什么样，而是决定用哪个 loader、从哪里找权重、按什么格式读权重，以及读完后如何进入模型参数。
```

再压缩成最小心智模型：

```text
ModelConfig 决定模型结构；
LoadConfig 决定权重加载方式；
get_model_loader 决定 loader 类；
BaseModelLoader 提供通用加载骨架；
具体 loader 负责文件和 iterator；
model.load_weights() 负责参数落位。
```
