# 大模型技术问题知识库设计方案

## 1. 背景与目标

在和 Claude 对话、阅读 vLLM / vllm-ascend / KV Pool 相关代码的过程中，会持续产生大量技术问题，例如：

- hybrid KV cache 是什么？
- DeepSeek V4 的 c4 / c128 是什么？
- KVPoolScheduler 和 vLLM Scheduler 是什么关系？
- CP 和 SP 有什么区别？
- `cp_scale = pcp_size * dcp_size` 为什么要这样算？

如果这些问题只记录在一个线性的 `question.md` 里，后续会越来越难检索、分类和复用。

因此建议在 `llm-project` 中建立一套结构化的技术问题知识库，用来沉淀：

1. 用户在学习和代码阅读过程中提出的问题；
2. 每个问题对应的简要回答和详细解释；
3. 问题与代码、文档、模型原理之间的关联；
4. 问题之间的层级关系和知识网络；
5. 最终形成一套完整的大模型 / vLLM / KV Cache / KV Pool 知识体系。

目标不是简单记流水账，而是形成可持续扩展的“树状问题体系”。

---

## 2. 推荐目录位置

建议知识库放在：

```text
D:\lzy\project\kv_pool\llm-project\llm_knowledge_base
```

后续可以根据需要调整为：

```text
D:\lzy\project\kv_pool\llm-project\documents\knowledge
```

当前先作为归档和设计草案存放在 `llm_knowledge_base` 目录。

---

## 3. 推荐目录结构

建议采用树状目录：

```text
llm_knowledge_base/
├── README.md
├── _index.md
├── _taxonomy.md
├── 01_llm_foundation/
├── 02_vllm_architecture/
├── 03_kv_cache/
├── 04_kv_pool/
├── 05_parallelism/
├── 06_scheduler/
├── 07_ascend/
├── 08_code_reading/
└── 99_inbox/
```

各文件/目录含义：

- `README.md`：知识库说明，介绍使用方式和维护规则。
- `_index.md`：总索引，列出所有问题文件。
- `_taxonomy.md`：知识体系树，维护大模型相关知识分类。
- `01_llm_foundation/`：大模型基础知识。
- `02_vllm_architecture/`：vLLM 架构相关问题。
- `03_kv_cache/`：KV Cache 原理、类型、block、hybrid KV 等。
- `04_kv_pool/`：KV Pool、外部 KV Cache、load/save/lookup 等。
- `05_parallelism/`：TP、PP、DP、EP、CP、SP 等并行策略。
- `06_scheduler/`：vLLM Scheduler、KVPoolScheduler、调度流程。
- `07_ascend/`：Ascend / NPU / vllm-ascend 相关问题。
- `08_code_reading/`：按代码文件组织的阅读笔记。
- `99_inbox/`：暂时无法准确归类的问题，后续再整理。

---

## 4. 推荐知识分类树

初始知识树可以这样设计：

```text
大模型知识体系
├── 01 LLM 基础
│   ├── Transformer
│   ├── Attention
│   ├── Prefill / Decode
│   └── KV Cache 基础
│
├── 02 vLLM 架构
│   ├── Engine
│   ├── Scheduler
│   ├── Worker
│   ├── BlockManager
│   └── KVConnector
│
├── 03 KV Cache
│   ├── FullAttention KV
│   ├── Sliding Window KV
│   ├── Hybrid KV Cache
│   ├── MLA / Latent KV
│   └── KV Cache Block
│
├── 04 KV Pool
│   ├── 外部 KV 池
│   ├── Load / Save
│   ├── Lookup
│   ├── ReqMeta
│   ├── cache_transfer_granularity
│   └── c1 / c4 / c128
│
├── 05 并行策略
│   ├── TP Tensor Parallel
│   ├── PP Pipeline Parallel
│   ├── DP Data Parallel
│   ├── EP Expert Parallel
│   ├── CP Context Parallel
│   └── SP Sequence Parallel
│
├── 06 Scheduler 调度
│   ├── vLLM Scheduler
│   ├── KVPoolScheduler
│   ├── Prefill 调度
│   ├── Decode 调度
│   └── Block 分配
│
├── 07 Ascend / NPU
│   ├── vllm-ascend
│   ├── HCCL
│   ├── NPU KV Cache
│   └── AscendStoreConnector
│
└── 08 代码阅读笔记
    ├── pool_scheduler.py
    ├── pool_worker.py
    ├── config_data.py
    └── connector.py
```

这棵树不是一次性定死的，后续可以随着问题增加不断调整。

---

## 5. 单问题单文件

建议不要把所有问题堆在一个 Markdown 文件里，而是采用“单问题单文件”。

例如问题：

```text
CP 和 SP 有什么区别？
```

可以整理成：

```text
llm_knowledge_base/05_parallelism/context-parallel-vs-sequence-parallel.md
```

问题：

```text
DeepSeek V4 的 c4 / c128 是什么？
```

可以整理成：

```text
llm_knowledge_base/03_kv_cache/deepseek-v4-c4-c128-kv-cache.md
```

或者：

```text
llm_knowledge_base/04_kv_pool/deepseek-v4-c4-c128-transfer-granularity.md
```

如果一个问题横跨多个主题，优先放在主语义最强的目录中，再通过 `related` 字段和索引链接到其他主题。

---

## 6. 单个问题文件模板

每个问题文件建议使用统一结构：

```markdown
---
title: CP 和 SP 有什么区别？
category: parallelism
tags:
  - context-parallel
  - sequence-parallel
  - kv-cache
  - vllm
related:
  - ../03_kv_cache/hybrid-kv-cache.md
  - ../04_kv_pool/cp-scale-in-kvpool.md
source:
  - ../../code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py
---

# CP 和 SP 有什么区别？

## 问题

CP 和 SP 有什么区别？

## 简要回答

SP 是把 sequence 上的中间 activation 切开计算；CP 是把上下文 / KV Cache 沿 token 维度切开存储和 attention。

## 详细解释

这里写完整解释。

## 和当前项目的关系

说明这个问题和当前 vllm-ascend / KV Pool 代码的关系。

例如：

```python
cp_scale = self.pcp_size * self.dcp_size
```

这里使用的是 CP 语义，不是 SP 语义。

## 相关问题

- hybrid KV cache 是什么？
- DeepSeek V4 的 c4 / c128 是什么？
- 为什么 KV Pool 要按 cache_transfer_granularity 对齐？
```

---

## 7. 文件内容字段说明

推荐每个问题至少包含以下部分：

### 7.1 Frontmatter

```yaml
---
title: 问题标题
category: 一级分类
tags:
  - 标签1
  - 标签2
related:
  - 相对路径1
  - 相对路径2
source:
  - 相关代码或文档路径
---
```

用途：

- 后续可以自动生成索引；
- 可以按 tag 检索问题；
- 可以维护问题之间的关系；
- 可以链接到代码和原始文档。

### 7.2 问题

保留用户原始问题或稍微规范化后的问题。

### 7.3 简要回答

用一两句话给出核心结论。

这部分方便快速复习。

### 7.4 详细解释

展开讲清楚概念、背景、原因、例子。

这部分负责真正形成知识沉淀。

### 7.5 和当前项目的关系

说明这个知识点和当前代码仓的关系，例如：

- 出现在哪个文件；
- 对哪个类/函数有影响；
- 为什么代码里这样写；
- 对 KV Pool / Scheduler / Worker 有什么意义。

代码引用建议使用格式：

```text
code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:90
```

### 7.6 相关问题

列出相关问题，形成知识网络。

---

## 8. Skill 行为设计

希望后续形成一个“技术问题记录 Skill”。

这个 Skill 的职责：

1. 识别当前对话中的技术问题；
2. 提炼问题标题；
3. 判断应该归入哪个知识目录；
4. 检查是否已有类似问题；
5. 如果已有，则更新原问题文件；
6. 如果没有，则创建新问题文件；
7. 写入问题、简要回答、详细解释、项目关联、相关问题；
8. 更新 `_index.md`；
9. 必要时更新 `_taxonomy.md`；
10. 如果暂时无法分类，先放入 `99_inbox/`。

---

## 9. 推荐触发方式

不建议每一句技术对话都自动落盘，因为容易产生重复、碎片和低质量笔记。

推荐三种触发方式。

### 9.1 显式触发

用户说：

```text
把这个问题整理到知识库
```

或者：

```text
记录这个问题
```

然后 Claude 将当前问题和回答整理到知识库中。

这是最稳妥的方式。

### 9.2 会话末批量整理

用户说：

```text
整理本轮所有技术问题
```

Claude 从当前对话中提取重要技术问题，批量归档。

适合长对话后统一整理。

### 9.3 Slash Command / Skill 触发

如果后续配置项目级 skill，可以设计为：

```text
/record-question
```

或：

```text
/knowledge
```

功能：

```text
把当前对话中的技术问题沉淀到 llm_knowledge_base 中。
```

---

## 10. 与现有 question.md 的关系

当前已有文件：

```text
D:\lzy\project\kv_pool\llm-project\documents\question\question.md
```

这个文件已经记录了一些问题，例如：

- KV 连接器注册机制；
- Scheduler / Worker 为什么共用 AscendStoreConnector；
- KVPoolScheduler 和 vLLM Scheduler 的关系；
- hybrid KV cache；
- DeepSeek V4 的 c4 / c128。

后续建议：

1. 保留 `question.md` 作为早期问题汇总或临时问题池；
2. 将其中 Q1-Q8 拆分成独立知识文件；
3. 新问题优先进入 `llm_knowledge_base`；
4. 如果只是临时记录，可以先写入 `99_inbox/`；
5. 等问题稳定后再归入正式目录。

---

## 11. 推荐落地步骤

### 第一步：建立知识库骨架

创建目录：

```text
llm_knowledge_base/
├── README.md
├── _index.md
├── _taxonomy.md
├── 01_llm_foundation/
├── 02_vllm_architecture/
├── 03_kv_cache/
├── 04_kv_pool/
├── 05_parallelism/
├── 06_scheduler/
├── 07_ascend/
├── 08_code_reading/
└── 99_inbox/
```

### 第二步：迁移已有 question.md

将 `documents/question/question.md` 中的 Q1-Q8 拆成独立文件。

建议初始迁移方向：

```text
Q1 注册机制 → 02_vllm_architecture/kv-connector-registration.md
Q2 register_connector 参数 → 02_vllm_architecture/register-connector-parameters.md
Q3 AscendStoreConnector 同时用于 Scheduler 和 Worker → 02_vllm_architecture/ascend-store-connector-role-split.md
Q4 为什么需要 AscendStoreConnector → 02_vllm_architecture/ascend-store-connector-adapter.md
Q5 为什么接口同时实现 Scheduler 和 Worker 方法 → 02_vllm_architecture/kv-connector-interface-design.md
Q6 池化调度和 vLLM 调度关系 → 06_scheduler/kvpool-scheduler-vs-vllm-scheduler.md
Q7 hybrid KV cache → 03_kv_cache/hybrid-kv-cache.md
Q8 DeepSeek V4 c4/c128 → 03_kv_cache/deepseek-v4-c4-c128-kv-cache.md
```

### 第三步：维护索引

更新：

```text
llm_knowledge_base/_index.md
```

按目录列出所有问题。

### 第四步：后续持续记录

以后用户说：

```text
把这个问题整理到知识库
```

Claude 就按照模板新增或更新问题文件。

---

## 12. 推荐最终工作模式

最适合当前项目的模式是：

```text
显式触发 + 树状知识库 + 单问题单文件 + 自动维护索引
```

即：

1. 用户平时自然提问；
2. Claude 正常回答；
3. 用户觉得值得沉淀时，说“整理到知识库”；
4. Claude 将当前问题结构化写入 `llm_knowledge_base`；
5. Claude 同步更新索引和相关链接；
6. 随着问题增多，逐渐形成完整的大模型知识体系。

---

## 13. 后续可扩展能力

未来可以继续增强：

1. 自动生成 tag 索引；
2. 自动生成代码文件到问题的反向索引；
3. 自动维护“相关问题”链接；
4. 按学习路径生成阅读顺序；
5. 按模块生成专题文档，例如：
   - KV Cache 专题；
   - vLLM Scheduler 专题；
   - KV Pool 专题；
   - AscendStoreConnector 专题；
6. 将问题体系迁移为 MkDocs / Docusaurus 站点；
7. 后续支持搜索、问答和复习清单。

---

## 14. 一句话总结

建议将技术问题记录从单一 `question.md` 升级为结构化知识库：

```text
一个问题一个文件，按知识树分类，维护总索引和相关链接，在持续对话中逐步沉淀成完整的大模型知识体系。
```
