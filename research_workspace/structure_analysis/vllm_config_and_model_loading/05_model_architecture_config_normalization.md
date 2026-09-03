# 05 ModelArchitectureConfig 归一化

本篇梳理 vLLM 如何把 HuggingFace config 中各模型族不统一的字段转换成统一的 `ModelArchitectureConfig`。这一步是配置链与模型加载链之间的关键桥梁：HF config 可以千差万别，但 vLLM 后续的 registry、cache、parallel、scheduler、loader 都需要稳定字段。

## 1. 入口

`ModelConfig.get_model_arch_config()`：`code/vllm/vllm/config/model.py:728`。

简化逻辑：

```text
model_type = hf_config.model_type
  ↓
在 MODEL_ARCH_CONFIG_CONVERTOR_REGISTRY 中查 convertor
  ↓
如果有专用 convertor，使用专用 convertor
  ↓
否则使用 ModelArchConfigConvertorBase
  ↓
convert(hf_config, model_config)
  ↓
ModelArchitectureConfig
```

专用 convertor 映射表：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:580`。

基础 convertor 主体：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:25`。

## 2. 为什么不能直接用 HF config

不同模型族字段命名差异很大，例如：

| 语义 | 常见字段名 |
|---|---|
| attention heads | `num_attention_heads`、`n_head`、`n_heads` |
| KV heads | `num_key_value_heads`、`num_kv_heads`、`n_head_kv`、`multi_query_group_num` |
| hidden size | `hidden_size`、`n_embd`、`d_model` |
| layers | `num_hidden_layers`、`n_layer`、`num_layers` |
| experts | `num_experts`、`moe_num_experts`、`n_routed_experts`、`num_local_experts` |
| max length | `max_position_embeddings`、`n_positions`、`seq_length`、`model_max_length` |
| quant config | `quantization_config`、`compression_config` |

如果 runtime 到处直接读取 HF config，就会出现大量模型族特判。vLLM 通过 `ModelArchitectureConfig` 把这些字段归一化。

## 3. `ModelArchitectureConfig` 包含什么

基础 convertor 构造统一字段的位置：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:345`。

核心字段包括：

```text
ModelArchitectureConfig
  ├─ architectures
  ├─ model_type
  ├─ text_model_type
  ├─ hidden_size
  ├─ total_num_hidden_layers
  ├─ total_num_attention_heads
  ├─ head_size
  ├─ vocab_size
  ├─ total_num_kv_heads
  ├─ num_experts
  ├─ quantization_config
  ├─ is_deepseek_mla
  ├─ is_mm_prefix_lm
  └─ derived_max_model_len_and_key
```

这些字段后续会被用于：

- attention 层构建；
- KV cache 大小估算；
- tensor parallel 切分；
- expert parallel 过滤；
- model registry 能力判断；
- max model len 校验；
- quantization 加载逻辑。

## 4. head_size 推导

入口：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:48`。

推导顺序不是简单的 `hidden_size // num_attention_heads`：

```text
1. DeepSeek MLA 特判
2. 优先读取 head_dim
3. 再读取 hidden_size_per_head
4. 最后 fallback 到 hidden_size // num_attention_heads
```

原因：部分模型的 head dim 并不等于 hidden size 除以 head 数，尤其是 MLA、GQA/MQA、特殊 rope/head 设计的模型。

## 5. qk_rope_head_dim 修正

入口：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:74`。

这里处理 transformers v5.4+ 的一个属性映射问题：如果发现 `qk_rope_head_dim` 和 `qk_nope_head_dim` 被错误映射成一样，vLLM 会重新读取原始 `config.json`，修正内存中的 `cfg.qk_rope_head_dim`。

这属于典型的“HF config 实例化后还要根据原始文件反补字段”的兼容逻辑。

## 6. KV heads 归一化

入口：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:106`。

基础 convertor 会依次尝试：

```text
n_head_kv
num_kv_heads
num_key_value_heads
multi_query_group_num
num_attention_groups
```

如果都不存在，就回退到 `num_attention_heads`。

这个字段对 vLLM 很重要，因为 KV cache 大小和 attention backend 的 KV layout 都依赖 KV heads，而不是只依赖 query heads。

## 7. experts 数归一化

入口：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:125`。

优先尝试字段：

```text
num_experts
moe_num_experts
n_routed_experts
num_local_experts
```

如果没有统一字段，会从 `block_configs` 中扫描 heterogeneous blocks，并取最大专家数。

这对 MoE 模型很关键，因为 expert parallel、EP weight filter、MoE quant state 都需要知道 expert 数和 expert 分布。

## 8. dtype 推导

入口：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:164`。

推导顺序：

```text
1. root config.dtype
2. text_config.dtype
3. vision_config.dtype
4. encoder_config.dtype
5. safetensors metadata
6. fallback: torch.float32
```

这说明 dtype 不完全由用户参数决定。用户传 `dtype=auto` 时，vLLM 会尽量从模型配置或权重元数据中推断。

## 9. quantization_config 归一化

入口：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:206`。

vLLM 会处理：

- `quantization_config`；
- `compression_config`；
- ModelOpt 的 `producer.name`；
- `quant_algo` 到社区标准 `quant_method` 的转换。

这一步让后续 `ModelConfig` 和 `weight_utils.get_quant_config()` 能够以较统一的方式识别量化方法。

## 10. max model len 候选值

入口：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:310`。

convertor 负责从多个候选字段中找出原始上下文长度候选，例如：

```text
max_position_embeddings
n_positions
max_seq_len
seq_length
model_max_length
```

注意：convertor 只负责提取候选值，最终 `max_model_len` 的完整校验和修正在 `ModelConfig._get_and_verify_max_len(...)`。

最终校验入口：`code/vllm/vllm/config/model.py:2090`。

该函数会继续考虑：

- 用户显式传入 `max_model_len`；
- tokenizer config 的 `model_max_length`；
- rope scaling；
- yarn / longrope；
- sliding window；
- encoder config 的 `max_seq_length`；
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN` 环境变量。

## 11. architectures 缺失与补全

`get_config()` 中如果 `config.architectures` 为空，会尝试根据 transformers 的 `MODEL_MAPPING_NAMES` 自动补全。

位置：`code/vllm/vllm/transformers_utils/config.py:709`。

如果补全失败，会提示用户通过 `hf_overrides` 指定。

这一步会影响模型注册：后续 `registry.resolve_model_cls(...)` 通常需要 architecture 名称来映射具体 vLLM 模型类。

## 12. 常见专用 convertor

专用 convertor 映射表：`code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:580`。

几个典型特化：

| 模型/类型 | 特化逻辑 | 位置 |
|---|---|---|
| Falcon | `multi_query=True` 且非新 decoder 架构时，KV heads 强制为 1 | `code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:420` |
| MPT | 从 `attn_config["kv_n_heads"]` 取 KV heads | `code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:439` |
| Dbrx | 从 `attn_config.kv_n_heads` 取 KV heads | `code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:446` |
| Nemotron NAS | 从 block config 推断结构 | `code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:455` |
| MiMo V2 | 删除错误的 `attention_chunk_size`，避免影响 hybrid KV cache | `code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:484` |
| Gemma4 | `head_size` 取 `max(head_dim, global_head_dim)` | `code/vllm/vllm/transformers_utils/model_arch_config_convertor.py:564` |

## 13. 和并行加载的关系

`ModelArchitectureConfig` 的字段会直接影响模型加载：

- `total_num_hidden_layers`：pipeline parallel 中每个 rank 加载哪些层；
- `total_num_attention_heads`：tensor parallel 中 attention 权重如何切；
- `total_num_kv_heads`：GQA/MQA 的 KV cache 和 KV projection 权重切分；
- `num_experts`：expert parallel 与 EP weight filter；
- `quantization_config`：选择量化实现和权重后处理；
- `hidden_size` / `head_size`：线性层、attention、KV cache shape；
- `derived_max_model_len_and_key`：scheduler/cache 对上下文长度的限制。

所以它不仅服务于 config 层，也直接影响 loader 和 model class 的行为。

## 14. 测试线索

模型架构配置的集成测试：`code/vllm/tests/config/test_model_arch_config.py:117`。

该测试会对一批模型构造 `ModelConfig(...)`，再断言 `model_arch_config` 字段和相关 getter 返回值。

辅助断言位置：`code/vllm/tests/config/test_model_arch_config.py:66`。

这类测试说明 vLLM 把 `ModelArchitectureConfig` 当成稳定内部接口，而不是临时变量。

## 15. 一句话总结

`ModelArchitectureConfig` 是 vLLM 对外部模型配置的统一抽象：它屏蔽不同模型族 HF config 字段命名差异，为 registry、loader、parallel、cache、scheduler、quantization 提供稳定的模型结构视图。
