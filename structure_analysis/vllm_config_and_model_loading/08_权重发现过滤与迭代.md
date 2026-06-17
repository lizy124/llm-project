# 08 权重发现过滤与迭代

本篇梳理 vLLM 模型加载中“权重文件如何被找到、过滤、打开并迭代”的链路。这部分主要由 `DefaultModelLoader` 和 `weight_utils.py` 完成。理解它可以解释为什么同一个模型仓库里有很多文件，但 vLLM 最终只读取其中一部分，以及为什么不同 `load_format` 会走不同 iterator。

## 1. 总体链路

```text
DefaultModelLoader.load_weights(model, model_config)
  ↓
_prepare_weights(model_name_or_path, revision, fall_back_to_pt, ...)
  ↓
确定权重目录 / 下载模型 / 获取候选文件
  ↓
filter_duplicate_safetensors_files(...)
filter_files_not_needed_for_inference(...)
ignore_patterns
  ↓
_get_weights_iterator(...)
  ↓
safetensors / pt / npcache / fastsafetensors / instanttensor / runai iterator
  ↓
model.load_weights(weights_iterator)
```

关键入口：

- `_prepare_weights()`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:97`
- `_get_weights_iterator()`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:211`
- `get_all_weights()`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:288`
- `load_weights()`：`code/vllm/vllm/model_executor/model_loader/default_loader.py:382`
- 公共工具：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:154`

## 2. `_prepare_weights()` 的职责

`DefaultModelLoader._prepare_weights()` 做的是“把模型路径和 load_format 转换成可迭代的权重文件列表”。

主要输入包括：

- `model_name_or_path`；
- `revision`；
- `fall_back_to_pt`；
- `load_format`；
- `download_dir`；
- `ignore_patterns`；
- safetensors/bin/pt 相关策略。

主要输出是：

```text
weights_dir
weight_files
use_safetensors
```

它需要处理：

1. 本地目录；
2. HF repo；
3. ModelScope repo；
4. safetensors index；
5. PyTorch bin fallback；
6. npcache；
7. fastsafetensors；
8. instanttensor；
9. Mistral 权重模式；
10. ignore patterns。

## 3. 本地 / HF / ModelScope 下载

权重下载相关公共函数在 `weight_utils.py`。

ModelScope 下载入口：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:154`。

配置读取侧的 repo 工具在：

- `code/vllm/vllm/transformers_utils/repo_utils.py:77`
- `code/vllm/vllm/transformers_utils/repo_utils.py:203`
- `code/vllm/vllm/transformers_utils/repo_utils.py:225`

可以把文件来源分成三类：

| 来源 | 行为 |
|---|---|
| 本地路径 | 直接扫描本地目录下符合模式的权重文件。 |
| HuggingFace Hub | 使用 revision/download_dir 下载或读取缓存。 |
| ModelScope | 启用 ModelScope 环境时切换到 ModelScope snapshot/list 逻辑。 |

## 4. safetensors index 与重复文件过滤

入口：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:582`。

`filter_duplicate_safetensors_files(...)` 的作用是利用 index 文件过滤重复 safetensors 分片。

常见 index：

```text
model.safetensors.index.json
consolidated.safetensors.index.json
```

为什么需要过滤？

模型目录里可能同时存在：

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
其他备份或重复 safetensors
```

如果简单 glob `*.safetensors`，可能读到 index 外的重复文件，导致同一参数被重复加载或冲突。vLLM 会优先根据 index 精确确定需要的 shard 集合。

## 5. 非推理文件过滤

入口：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:603`。

`filter_files_not_needed_for_inference(...)` 会剔除训练态或推理无关文件，例如：

- optimizer states；
- scheduler states；
- trainer states；
- RNG states；
- checkpoint metadata；
- 非模型参数文件。

这一步可以避免错误读取训练 checkpoint 附属文件，也能减少扫描和打开文件的开销。

## 6. ignore_patterns

`LoadConfig.ignore_patterns` 允许用户显式排除部分文件。

它会在 `_prepare_weights()` 阶段影响候选文件集合。

典型用途：

- 模型目录中同时有多套权重格式，只想读其中一种；
- 排除 adapter/optimizer/backup；
- 避免错误读取不兼容 shard；
- 临时绕开坏文件。

但要注意：过度 ignore 可能导致必要权重缺失，最终在 `model.load_weights()` 或 missing weight 检查时报错。

## 7. iterator 分发

默认 loader 的 `_get_weights_iterator()` 入口：`code/vllm/vllm/model_executor/model_loader/default_loader.py:211`。

它根据 `load_format` 和文件类型选择不同 iterator。

主要 iterator：

| iterator | 位置 | 用途 |
|---|---|---|
| `safetensors_weights_iterator` | `code/vllm/vllm/model_executor/model_loader/weight_utils.py:820` | 标准 safetensors。 |
| `runai_safetensors_weights_iterator` | `code/vllm/vllm/model_executor/model_loader/weight_utils.py:987` | RunAI safetensors streaming。 |
| `fastsafetensors_weights_iterator` | `code/vllm/vllm/model_executor/model_loader/weight_utils.py:1024` | fastsafetensors 加载路径。 |
| `instanttensor_weights_iterator` | `code/vllm/vllm/model_executor/model_loader/weight_utils.py:1093` | instanttensor 路径。 |
| `pt_weights_iterator` | `code/vllm/vllm/model_executor/model_loader/weight_utils.py:1133` | PyTorch bin/pt 权重。 |
| `multi_thread_pt_weights_iterator` | `code/vllm/vllm/model_executor/model_loader/weight_utils.py:1152` | 多线程 PyTorch 权重读取。 |

这些 iterator 最终都要产出类似：

```python
Iterable[tuple[str, torch.Tensor]]
```

然后交给模型类的 `load_weights()`。

## 8. safetensors iterator

`safetensors_weights_iterator(...)` 是最常见路径。

位置：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:820`。

它通常会：

1. 逐个打开 safetensors 文件；
2. 遍历 tensor key；
3. 读取 tensor；
4. yield `(name, tensor)`；
5. 尽量避免一次性把所有文件全部读入内存。

优点：

- 安全性比 pickle/bin 更好；
- 支持 metadata；
- 支持分片；
- 适合大模型流式读取。

## 9. PyTorch pt/bin iterator

入口：

- `pt_weights_iterator(...)`：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:1133`
- `multi_thread_pt_weights_iterator(...)`：`code/vllm/vllm/model_executor/model_loader/weight_utils.py:1152`

这类路径通常用于 `.bin` / `.pt` 权重。

注意：PyTorch pickle 格式在安全性和加载行为上与 safetensors 不同，因此 vLLM 在有 safetensors 时一般优先 safetensors。

## 10. fastsafetensors / instanttensor / RunAI

这些 iterator 的目标是优化特定 I/O 路径：

- `fastsafetensors`：加速 safetensors 读取；
- `instanttensor`：走 instant tensor 相关加载路径；
- `RunAI`：面向远程对象存储和 streaming 场景。

它们保留统一的 `(name, tensor)` 输出接口，使上层模型 `load_weights()` 不必知道底层 I/O 细节。

## 11. `get_all_weights()` 的意义

入口：`code/vllm/vllm/model_executor/model_loader/default_loader.py:288`。

`get_all_weights()` 主要用于需要重新获取权重 iterator 的场景，例如 reload。

它把准备文件和 iterator 选择封装起来，让 reload 路径不用重复实现权重发现逻辑。

## 12. Expert Parallel 权重过滤

入口：`code/vllm/vllm/model_executor/model_loader/default_loader.py:318`。

`_init_ep_weight_filter(...)` 说明并行策略已经进入权重读取阶段。

Expert Parallel 下，并不是每个 rank 都需要所有 expert 权重。EP filter 可以在加载阶段过滤不属于当前 rank 的 expert 参数，减少无效读取或无效加载。

这和普通 tensor parallel 的“参数切片”不同：EP 更像是“某些 expert 整体归某些 rank”。

## 13. 和模型类 `load_weights()` 的边界

iterator 只负责产出 `(name, tensor)`，但如何处理这些 tensor 是模型类的职责。

模型类 `load_weights()` 通常会做：

- 权重名替换；
- fused QKV 映射；
- gate/up projection fused 映射；
- tensor parallel shard 选择；
- pipeline parallel 层过滤；
- expert parallel expert 过滤；
- 量化权重 scale/zero 处理；
- tied embedding 跳过或共享；
- 返回 loaded parameter names。

因此同一个 iterator 输出，交给不同模型类会有完全不同的映射逻辑。

## 14. 缺失权重和多余权重

加载结束后，vLLM 通常会检查：

- 必要参数是否都加载；
- 某些参数是否允许缺失；
- 某些 checkpoint tensor 是否被跳过；
- tied weights 是否特殊处理；
- quantization scale 是否齐全。

缺失不一定都是错误，例如：

- pipeline parallel rank 不持有某些层；
- tied embedding/lm_head 可能共享；
- 多模态子模块按任务可能不加载；
- 某些 scale 在后处理阶段生成。

但如果模型类没有正确声明/处理这些例外，就会出现加载失败。

## 15. 调试建议

### 15.1 找不到权重文件

检查：

```text
LoadConfig.load_format
LoadConfig.download_dir
LoadConfig.ignore_patterns
ModelConfig.model
ModelConfig.revision
本地目录是否存在 config.json / safetensors / bin
是否启用 ModelScope
```

### 15.2 读到了错误文件

检查：

```text
目录中是否同时存在多种格式
safetensors index 是否正确
ignore_patterns 是否遗漏
filter_duplicate_safetensors_files 是否生效
filter_files_not_needed_for_inference 是否覆盖目标文件
```

### 15.3 iterator 正常但模型加载失败

重点转向模型类：

```text
model.load_weights 实现
权重 key 是否匹配
TP/PP/EP rank 是否符合预期
quantization_config 是否和 checkpoint 匹配
是否用了错误 architecture/model class
```

## 16. 一句话总结

vLLM 的权重文件读取不是简单 glob 后 `torch.load`：它会根据 `LoadConfig`、模型路径、revision、index 文件、ignore patterns、并行策略和 load_format 精确确定文件集合，再用统一 iterator 把不同格式权重转换成 `(name, tensor)` 流，最终交给具体模型类完成参数映射。
