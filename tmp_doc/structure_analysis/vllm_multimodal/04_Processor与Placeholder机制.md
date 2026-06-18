# 04 Processor 与 Placeholder 机制

本篇梳理 vLLM 多模态 processor 如何把原始媒体输入变成模型输入 kwargs，同时修改 prompt、定位 placeholder，并最终生成 engine 可消费的多模态输入。这里是 vLLM 多模态链路中最复杂也最关键的一层。

## 1. processor 总体职责

核心类：`BaseMultiModalProcessor`。

位置：`code/vllm/vllm/multimodal/processing/processor.py:972`。

processor 做四类事情：

1. 解析多模态数据；
2. 调 HF processor，得到 tensor kwargs；
3. 根据模型规则对 prompt 做 insertion/replacement；
4. 找出每个多模态 item 对应的 placeholder token 区间。

最终输出：

```text
prompt_token_ids
mm_kwargs
mm_hashes
mm_placeholders
```

入口方法：`code/vllm/vllm/multimodal/processing/processor.py:1663`。

## 2. `ProcessorInputs`

processor 输入包装：`code/vllm/vllm/multimodal/processing/inputs.py:13`。

它承载：

- prompt 文本或 token；
- 多模态数据；
- processor kwargs；
- 多模态 UUID；
- 其他请求级输入。

这一层把入口 prompt schema 和 processor 内部逻辑隔开。

## 3. 调用 HF processor

核心函数：

- `_apply_hf_processor_main()`：`code/vllm/vllm/multimodal/processing/processor.py:1258`
- `_cached_apply_hf_processor()`：`code/vllm/vllm/multimodal/processing/processor.py:1441`

HF processor 负责模型家族特定的预处理，例如：

- 图像 resize/crop/normalize；
- video frame 采样/patchify 前准备；
- audio feature extraction；
- 生成 pixel_values、input_features、grid_thw、attention_mask 等字段。

vLLM processor 对 HF processor 的输出做进一步包装，拆成按 modality/item 组织的 `MultiModalKwargsItems`。

## 4. `MultiModalKwargsItem` 与 `MultiModalKwargsItems`

定义位置：

- `MultiModalKwargsItem`：`code/vllm/vllm/multimodal/inputs.py:854`
- `MultiModalKwargsItems`：`code/vllm/vllm/multimodal/inputs.py:882`
- `from_hf_inputs()`：`code/vllm/vllm/multimodal/inputs.py:919`

语义：

```text
MultiModalKwargsItem
  = 单个 image/audio/video item 对应的一组模型输入 kwargs

MultiModalKwargsItems
  = 多个 modality、多 item 的集合
```

例如 Qwen2-VL 可能生成：

```text
image item kwargs:
  pixel_values
  image_grid_thw

video item kwargs:
  pixel_values_videos
  video_grid_thw
```

Qwen2-Audio 可能生成：

```text
input_features
audio_attention_mask
feature_attention_mask
```

## 5. PlaceholderRange

定义：`code/vllm/vllm/multimodal/inputs.py:118`。

它记录某个多模态 item 在最终 prompt token 序列中的占位区间：

```text
offset: 起始 token 位置
length: 占用 token 数
is_embed: 是否是 embedding placeholder
```

这不是可有可无的 metadata，而是后续 scheduler 和 GPU worker 对齐多模态 encoder 输出的坐标系统。

## 6. prompt update 类型

processor 中有一组 prompt 更新结构：

- `PromptUpdate`：`code/vllm/vllm/multimodal/processing/processor.py:298`
- `PromptInsertion`：`code/vllm/vllm/multimodal/processing/processor.py:354`
- `PromptReplacement`：`code/vllm/vllm/multimodal/processing/processor.py:423`
- `ResolvedPromptUpdate`：`code/vllm/vllm/multimodal/processing/processor.py:526`

语义：

| 类型 | 作用 |
|---|---|
| insertion | 在 prompt 中插入模型所需的多模态占位 token 或文本。 |
| replacement | 把已有占位串替换成模型需要的占位 token 序列。 |
| resolved update | 在 text/token 级应用前解析完成的更新。 |

这解释了为什么多模态 processor 既要处理 tensor，又要处理 prompt。

## 7. token-space 优先，text-space 兜底

关键函数：

- `iter_token_matches()`：`code/vllm/vllm/multimodal/processing/processor.py:619`
- `replace_token_matches()`：`code/vllm/vllm/multimodal/processing/processor.py:648`
- `apply_token_matches()`：`code/vllm/vllm/multimodal/processing/processor.py:831`
- `apply_text_matches()`：`code/vllm/vllm/multimodal/processing/processor.py:848`

设计思想：

```text
如果已有 token 序列
  → 优先在 token 序列中匹配/替换 placeholder
否则或 token 匹配不可行
  → 回退到 text 级别匹配/替换
```

原因是：多模态 placeholder 最终必须以 token offset/length 表示；在 token-space 处理能减少二次 tokenization 导致的偏移和不一致。

## 8. placeholder discovery

发现 placeholder 的函数：`code/vllm/vllm/multimodal/processing/processor.py:934`。

`find_mm_placeholders()` 会在 prompt update 应用之后重新扫描最终 prompt/token 序列，找到每个多模态 item 对应的 `PlaceholderRange`。

也就是说：

```text
用户输入里的 <image>
  不一定直接等于最终 placeholder 区间

最终区间由 processor 根据模型规则、tokenization、replacement/insertion 后的结果确定
```

## 9. `_get_mm_prompt_updates()` 与 `_apply_prompt_updates()`

关键位置：

- `_get_mm_prompt_updates()`：`code/vllm/vllm/multimodal/processing/processor.py:1055`
- `_apply_prompt_updates()`：`code/vllm/vllm/multimodal/processing/processor.py:1528`

前者负责根据模型 processor info 产生“需要对 prompt 做什么更新”，后者负责把这些更新实际应用到 prompt token/text 上。

不同模型的差异主要体现在 prompt update 规则上：

- LLaVA 类模型通常需要图像占位 token；
- Qwen2-VL 需要 image/video placeholder 和 grid 对齐；
- Gemma3-MM 可能涉及 image token、newline token、crop/pan-and-scan；
- Pixtral 可能更依赖 chat template 预插入图像 token。

## 10. token merge 的准确含义

这里的“token merge”不是把 image tensor 直接插进 token list，而是让三类信息对齐：

```text
prompt_token_ids
mm_kwargs
mm_placeholders
```

合并分两阶段。

### 10.1 processor 阶段

`apply()` 最终产出 `MultiModalInput`，其中包含：

- `prompt_token_ids`；
- `mm_kwargs`；
- `mm_hashes`；
- `mm_placeholders`。

入口：`code/vllm/vllm/multimodal/processing/processor.py:1663`。

这是第一次并轨：文本 token 序列和多模态数据在同一个对象中对齐。

### 10.2 engine 阶段

V1 `InputProcessor` 会读取 processor 输出：

- `decoder_inputs["mm_kwargs"]`；
- `decoder_inputs["mm_placeholders"]`；
- `decoder_inputs["mm_hashes"]`。

然后构造 `MultiModalFeatureSpec`。

关键位置：`code/vllm/vllm/v1/engine/input_processor.py:333`。

排序函数：`code/vllm/vllm/multimodal/utils.py:137`。

这是第二次并轨：按 prompt 中真实位置排序，形成 runtime 的 `Request.mm_features`。

## 11. dummy inputs 与 processor 的关系

dummy input builder 会构造假的文本和多模态输入，再走标准 processor 流程。

入口：`code/vllm/vllm/multimodal/processing/dummy_inputs.py:67`。

用途：

- 估算最大 placeholder 长度；
- 估算 encoder token 数；
- memory profiling；
- capacity planning。

因此 dummy inputs 不是另一个特殊处理器，而是复用标准 parser/processor 的 profile 输入生成器。

## 12. 常见错误定位

### 12.1 placeholder 数量和多模态 item 数量不匹配

检查：

```text
MultiModalDataItems 中 item 数量
PromptReplacement/PromptInsertion 是否生成了对应占位
find_mm_placeholders 是否找到同等数量范围
limit_mm_per_prompt 是否提前截断/报错
```

### 12.2 placeholder 长度和 encoder 输出长度不匹配

检查：

```text
模型 get_num_mm_encoder_tokens / get_num_mm_connector_tokens
processor 生成的 placeholder length
image/video grid_thw 或 audio effective length
InputProcessor 的合法性校验
```

V1 校验位置：`code/vllm/vllm/v1/engine/input_processor.py:434`。

### 12.3 token prompt 下匹配失败

检查：

```text
token ids 是否包含模型要求的 placeholder token
processor 是否只支持 text-space replacement
chat template 是否已经应用
apply_token_matches 是否能找到匹配
```

## 13. 一句话总结

`BaseMultiModalProcessor` 是 vLLM 多模态预处理的核心：它一边把媒体交给 HF processor 生成 tensor kwargs，一边按模型规则改写 prompt 并定位 placeholder token 区间；最终输出的 `prompt_token_ids + mm_kwargs + mm_hashes + mm_placeholders` 是 engine/runtime 处理多模态的基础。
