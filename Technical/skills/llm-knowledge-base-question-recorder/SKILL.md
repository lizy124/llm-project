---
name: llm-knowledge-base-question-recorder
description: Record technical questions and answers into the llm_knowledge_base tree as one Markdown file per question, maintaining index and taxonomy files.
---

# LLM Knowledge Base Question Recorder

## Purpose

Use this skill when the user asks to record, archive, organize, or add a technical question into the LLM knowledge base.

The knowledge base root is:

```text
D:\lzy\project\kv_pool\llm-project\llm_knowledge_base
```

The goal is to build a sustainable, expandable, tree-shaped question system for LLM, vLLM, KV Cache, KV Pool, parallelism, scheduler, Ascend, and code-reading knowledge.

Do not put all questions into one Markdown file. Use **one question per Markdown file**.

## Trigger Phrases

Invoke this skill when the user says things like:

- 把这个问题整理到知识库
- 记录这个问题
- 加入问题体系
- 归档这个技术问题
- 整理本轮所有技术问题
- 写入 llm_knowledge_base
- 放到知识库里
- 形成问题文档

If the user asks for a new skill or asks to modify this skill, update this `SKILL.md` or related files instead of recording a question.

## Knowledge Base Directory Tree

The initial tree should be:

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

This tree is not fixed forever. It may evolve as the user's question system grows.

## Category Rules

Choose the most appropriate directory for each question:

- `01_llm_foundation/`: Transformer, attention, prefill/decode, general LLM concepts.
- `02_vllm_architecture/`: vLLM Engine, Scheduler, Worker, BlockManager, KVConnector architecture.
- `03_kv_cache/`: KV Cache basics, FullAttention KV, Sliding Window KV, Hybrid KV Cache, MLA / latent KV, KV cache block.
- `04_kv_pool/`: external KV pool, lookup, load/save, ReqMeta, metadata, cache transfer granularity, c1/c4/c128 as KV Pool behavior.
- `05_parallelism/`: TP, PP, DP, EP, CP, SP, and distributed parallel strategy concepts.
- `06_scheduler/`: vLLM Scheduler, KVPoolScheduler, scheduling loop, prefill/decode scheduling, block allocation.
- `07_ascend/`: Ascend NPU, vllm-ascend, HCCL, AscendStoreConnector, platform-specific details.
- `08_code_reading/`: notes organized by source code file or implementation reading path.
- `99_inbox/`: unclear, temporary, or cross-cutting questions that cannot be classified confidently yet.

If a question belongs to multiple categories, put it in the most specific primary category and add related links/tags to connect it to the other categories.

## Required File Format

Each question file must be Markdown and should use this structure:

```markdown
---
title: <normalized question title>
category: <category-name>
tags:
  - <tag-1>
  - <tag-2>
related:
  - <relative-link-if-known>
source:
  - <code-or-document-path-if-relevant>
---

# <normalized question title>

## 问题

<slightly normalized version of the user's question>

## 简要回答

<short answer, usually 1-3 paragraphs or a few bullets>

## 详细解答

<clear detailed explanation with examples, diagrams, code snippets, or reasoning when useful>

## 和当前项目的关系

<explain how this question relates to the current repository, vLLM, vllm-ascend, KV Pool, code file, or document. If not relevant, write “暂无直接项目关联。”>

## 相关问题

- <related question or link>
```

The user explicitly wants at least:

1. a slightly normalized question;
2. a brief answer;
3. a detailed answer.

Do not omit these three parts.

## File Naming Rules

Use lower-case kebab-case English filenames when possible.

Examples:

```text
context-parallel-vs-sequence-parallel.md
hybrid-kv-cache.md
deepseek-v4-c4-c128-kv-cache.md
kvpool-scheduler-vs-vllm-scheduler.md
cp-scale-in-kvpool-scheduler.md
```

Avoid spaces and Chinese punctuation in filenames.

## Index Rules

Maintain:

```text
llm_knowledge_base/_index.md
```

The index should group question files by category. Add one line per question:

```markdown
- [问题标题](relative/path.md) — one-line hook
```

Do not duplicate entries. If updating an existing question, update its index line if needed.

## Taxonomy Rules

Maintain:

```text
llm_knowledge_base/_taxonomy.md
```

This file records the current tree-shaped knowledge system. Update it when:

- a new category is created;
- a new important subtopic appears;
- an existing subtopic becomes clearer.

Do not over-edit taxonomy for every small question. Keep it stable and readable.

## README Rules

Maintain:

```text
llm_knowledge_base/README.md
```

If it does not exist, create it with:

- purpose of the knowledge base;
- directory structure;
- one-question-one-file rule;
- how to record a question;
- relationship between `_index.md`, `_taxonomy.md`, and question files.

Do not rewrite README unnecessarily if it already exists.

## Workflow

When recording a question:

1. Identify the technical question from the current conversation.
2. Normalize the question title without changing its meaning.
3. Decide the primary category directory.
4. Check whether a similar question already exists in the target category or `_index.md`.
5. If similar file exists, update that file instead of creating a duplicate.
6. If no similar file exists, create a new Markdown file.
7. Include the required sections: 问题, 简要回答, 详细解答.
8. Add 和当前项目的关系 and 相关问题 sections.
9. Update `_index.md`.
10. Update `_taxonomy.md` only when useful.
11. Keep the final user response short: state the file path(s) created or updated.

## Existing Question Migration Guidance

If migrating from:

```text
D:\lzy\project\kv_pool\llm-project\documents\question\question.md
```

Recommended mapping:

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

## Style

- Use Chinese for content unless the user asks otherwise.
- Be concise but clear.
- Prefer explanation that connects concept → code → project behavior.
- Use code references in clickable form when possible, such as:

```text
code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:90
```

- Do not create unnecessary documentation outside `llm_knowledge_base` unless asked.
- Do not add unrelated questions.
- Do not invent code facts. If code details matter, read the relevant file first.

## Git / Daily Push Note

The user pushes:

```text
D:\lzy\project\kv_pool\llm-project
```

to a remote repository daily.

Therefore all knowledge base files should be stored under:

```text
D:\lzy\project\kv_pool\llm-project\llm_knowledge_base
```

so they are included in the repository push.
