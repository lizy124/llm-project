# Draft

这里存放尚未稳定、需要评审或等待交接的阶段性材料。

## 子目录

- `analysis_ascend_mp/`：Ascend MP 分析——审核计划、证据矩阵、代码走读与验证计划。
- `block_hash_to_str/`：Issue #14148 block hash→str 的设计、分析和单测验证记录。
- `issues_10/`：KV Pool 相关问题清单、审计和优先级评估。
- `issues_14143/`：Issue #14143 传输线程态→进程态分析。
- `issues_14145/`：Issue #14145 ZMQ lookup payload 兼容性分析、复核和交接材料。
- `mla_load/`：MLA read dedup 综合分析（与 `transfer_data/kv_pool_issue/mla_read_dedup_analysis.md` 同主题，见下）。
- `validation_pr_cz/`：PR 验证过程及结论。

## 边界与归档

- 完成后的内容应迁移到 `llm_knowledge_base/` 或 `research_workspace/`，不要把同一结论长期维护在多个目录。
- MLA read dedup 横跨 `draft/mla_load/` 与 `transfer_data/kv_pool_issue/`：前者放综合分析初稿，后者放收敛后的设计分析；两者是演进关系（初稿 → 收敛稿），不再各自追加新内容。
