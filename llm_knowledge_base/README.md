# LLM Knowledge Base

这是一个用于沉淀大模型、vLLM、KV Cache、KV Pool、并行策略、调度机制和 Ascend 相关知识的问题知识库。

## 目标

- 形成可持续扩展的树状问题体系；
- 每个技术问题单独成文，避免堆在一个 Markdown 文件里；
- 每个问题都包含规范化问题、简要回答和详细解答；
- 通过 `_index.md` 和 `_taxonomy.md` 维护索引与知识分类；
- 随仓库 `D:\lzy\project\kv_pool\llm-project` 一起同步到远程。

## 目录结构

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

## 记录规则

每个问题使用一个独立 Markdown 文件，推荐结构：

```markdown
---
title: <问题标题>
category: <分类>
tags:
  - <标签>
related:
  - <相关问题链接>
source:
  - <相关代码或文档路径>
---

# <问题标题>

## 问题

## 简要回答

## 详细解答

## 和当前项目的关系

## 相关问题
```

## 索引说明

- `_index.md`：按分类列出所有问题。
- `_taxonomy.md`：维护当前知识树。
- `99_inbox/`：临时放置暂时无法归类的问题。
