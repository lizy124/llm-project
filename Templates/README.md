# Templates — 项目执行手册（可复用模板区）

> 这是围绕 LLM / vLLM / KV Pool / vLLM-Ascend 的**项目执行手册**，给出"新 PR / 新专项"从头走到尾的可复用方法论。
> 定位：**任务规范与方法论模板**（跨专项复用），与 `Technical/`（长期知识）、`Proposals/`（具体专项与任务）分层。
> 新任务默认遵循 [任务开发、验证与归档规范（试行）](08_task_docs/task_lifecycle_spec_draft.md)。01–07 提供参考方法，08 规定任务组织与交付方式。

---

## 1. 项目工作流

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
[04] 测试验证        ← Templates/03_env_setup → 04_pool_setup → 05_vllm_launch
                       → 06_test_method → 07_pass_criteria
   │
   ▼
产出：Proposals/<专项>/<任务>/（五个工作目录，含任务脚本、配置与证据）
      → 任务内 archive/（五份归档文档及关键证据）
```

## 2. 目录结构

| 目录 | 内容 | 来源 |
|------|------|------|
| `_common/common_constraints.md` | IRON RULE、版本配对表、端口规划、共享服务器纪律 | 各处汇总 |
| `01_requirements/how_to_analyze.md` | 怎么拆需求、提问、识别风险、界定改动面 | 方法论 + Layerwise 样板 |
| `02_design/how_to_design.md` | 怎么写设计提案、实施计划、测试计划 | 方法论 + Layerwise 样板 |
| `03_env_setup/env_setup_guide.md` | 镜像选择 → 建容器 → 配代理 → 查版本/卸载 → 装 vllm/vllm-ascend | transfer_data/playbook/create_env.md |
| `04_pool_setup/memcache-colocated.md` | memcache 共置模式完整可跑方案（hugepages→MetaService→vllm 接入→验证→坑） | test_pool/01a_memcache_kv_pool_setup.md |
| `04_pool_setup/memcache-standalone.md` | memcache standalone（LocalService 独立进程）模式完整方案 | test_pool/01b_memcache_standalone.md |
| `04_pool_setup/mooncake-single-node.md` | mooncake 单机基线完整方案（端口/配置/启动/验证/证据） | test_pool/01c_mooncake_single_node_baseline.md |
| `05_vllm_launch/vllm_pool_launch.md` | 池化场景启动（关键参数/READY 等待/启动失败速查） | verify_guide §4 |
| `06_test_method/test_design.md` | prompt 门槛/验证矩阵/标准测试流程/对照实验/脚本纪律 | verify_guide §5 |
| `07_pass_criteria/pass_criteria.md` | 三维证据链/虚假通过防范/双轮夹逼/验收表模板/硬限 | verify_guide §6-8 |
| `08_task_docs/` | **任务试行规范**：五个工作目录、脚本分类、分次验证、五文档归档；`skeleton/` 为归档文档骨架 | 任务组织与交付规范 |

## 3. 使用方式

新 PR 到来时：
1. 先读试行规范，明确专项、任务与测试场景，建立任务入口及五个主题工作目录。
2. 参考 `01_requirements/`、`02_design/`，在任务的 01、02 目录维护需求方案和开发记录，在 05 制定测试计划。
3. 参考 `03_env_setup` 至 `07_pass_criteria` 准备和执行验证；脚本、配置放任务 03，操作说明放 04，每次运行与证据放 05。
4. 收尾时提炼为 `archive/` 下五份归档文档，使用 `08_task_docs/skeleton/` 骨架；关键脚本与配置全文整合进归档 03，必要证据归入 `archive/evidence/`。

## 4. 维护规则

1. **新 lesson 必须回写**：任何服务器上实操得到的新经验/新坑/新判定标准/新硬限，完成后回写 `Templates/` 对应章节。
2. **服务器特定信息不放这里**：某次任务的端口、权重路径、容器名等，放任务的 04 和实际运行记录中；既有历史记录可继续引用。
3. **只放通用方法论**：换个服务器、换个 PR 依然成立的规则才入库。
4. **修改同步检查**：`_common/common_constraints.md` 的公共约束（IRON RULE、端口）与其他区文档保持一致。

## 5. 边界与归档

- `Templates/` 是"怎么做"，不重复存放 `Technical/` 已有的"是什么/做过什么"；模板中可链接到 `Technical/` 的沉淀，但不复制整篇。
- `Proposals/` 承载任务从需求到验证、归档的完整材料；任务专属脚本和证据随任务存放。
- `Technical/` 承载可复用知识，`assets/` 承载共享与历史资产。原强制分散任务材料的目录限制已废除。
- 任务归档保留在任务中；通用经验按需沉淀并链接引用，不重复维护当前结论。