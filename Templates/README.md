# Templates — 项目执行手册（可复用模板区）

> 这是围绕 LLM / vLLM / KV Pool / vLLM-Ascend 的**项目执行手册**，给出"新 PR / 新专项"从头走到尾的可复用方法论。
> 定位：**方法论模板**（跨专项复用），与 `Technical/`（长期知识/技术记录）、`Proposals/`（具体专项的方案/规划实例）分层。

---

## 1. 项目工作流（七章纵向骨架）

新 PR 到来时的完整流程：

```
新 PR / 新专项
   │
   ▼
[01] 需求分析        ← Templates/01_requirements/
   │
   ▼
[02] 方案设计        ← Templates/02_design/
   │
   ▼
[03] 代码开发        ← （不在本手册，由用户/agent 自行完成）
   │
   ▼
[04] 测试验证        ← Templates/03_env_setup → 04_backend_startup
                       → 05_vllm_launch → 06_test_method → 07_pass_criteria
   │
   ▼
产出：Proposals/<专项>/（实例） + Technical/<专项>/（技术记录与证据） + assets/<专项>/（脚本/数据）
```

## 2. 目录结构

| 目录 | 内容 | 来源 |
|------|------|------|
| `_common/common_constraints.md` | IRON RULE、版本配对表、端口规划、共享服务器纪律 | 各处汇总 |
| `01_requirements/how_to_analyze.md` | 怎么拆需求、提问、识别风险、界定改动面 | 方法论 + Layerwise 样板 |
| `02_design/how_to_design.md` | 怎么写设计提案、实施计划、测试计划 | 方法论 + Layerwise 样板 |
| `03_env_setup/env_setup_guide.md` | 镜像选择 → 建容器 → 配代理 → 查版本/卸载 → 装 vllm/vllm-ascend | transfer_data/playbook/create_env.md |
| `04_backend_startup/mooncake_startup.md` | mooncake master 拉起规范（端口/配置/探活/停止） | verify_guide §2 |
| `04_backend_startup/memcache_startup.md` | memcache MetaService 拉起规范（两份 conf/环境变量/探针/版本匹配铁律/独立端口变体） | verify_guide §3 + test_pool/01_memcache_kv_pool_setup.md |
| `05_vllm_launch/vllm_pool_launch.md` | 池化场景启动（关键参数/READY 等待/启动失败速查） | verify_guide §4 |
| `06_test_method/test_design.md` | prompt 门槛/验证矩阵/标准测试流程/对照实验/脚本纪律 | verify_guide §5 |
| `07_pass_criteria/pass_criteria.md` | 三维证据链/虚假通过防范/双轮夹逼/验收表模板/硬限 | verify_guide §6-8 |

## 3. 使用方式

新 PR 到来时：
1. 按 `01_requirements/how_to_analyze.md` 拆需求，产出 `Proposals/<专项>/01_requirements_analysis.md`
2. 按 `02_design/how_to_design.md` 写设计/计划，产出 `Proposals/<专项>/02_design_proposal.md` 等
3. 测试时按 `03_env_setup` → `04_backend_startup` → `05_vllm_launch` → `06_test_method` → `07_pass_criteria` 走，产出 `Technical/<专项>/`（技术记录与证据）+ `assets/<专项>/`（脚本/数据）

## 4. 维护规则

1. **新 lesson 必须回写**：任何服务器上实操得到的新经验/新坑/新判定标准/新硬限，完成后回写 `Templates/` 对应章节。
2. **服务器特定信息不放这里**：某台服务器的端口占用/权重路径/容器名等，放 `Technical/` 对应专项。
3. **只放通用方法论**：换个服务器、换个 PR 依然成立的规则才入库。
4. **修改同步检查**：`_common/common_constraints.md` 的公共约束（IRON RULE、端口）与其他区文档保持一致。

## 5. 边界与归档

- `Templates/` 是"怎么做"，不重复存放 `Technical/` 已有的"是什么/做过什么"；模板中可链接到 `Technical/` 的沉淀，但不复制整篇。
- `Proposals/<专项>/` 是实例（该专项的需求/设计/计划），`Templates/` 是模板（怎么写需求/设计/计划）——两者是"用"与"被用"关系，不重复。
- 稳定后的草稿应归档到 `Technical/llm_knowledge_base` 或 `Technical/research_workspace`，避免同一结论长期存放在多个目录。