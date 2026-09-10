# 02 — 方案设计（怎么写设计提案、实施计划、测试计划）

> 需求分析完成后，进入方案设计阶段。本章给出"怎么写设计提案、实施计划、测试计划"的方法论。
> **样板**：[Layerwise-Pooling-Optimization/02_design_proposal.md](../../Proposals/ascend_project/Layerwise-Pooling-Optimization/02_design_proposal.md)、[03_implementation_plan.md](../../Proposals/ascend_project/Layerwise-Pooling-Optimization/03_implementation_plan.md)、[04_dev_plan.md](../../Proposals/ascend_project/Layerwise-Pooling-Optimization/04_dev_plan.md)、[05_test_plan.md](../../Proposals/ascend_project/Layerwise-Pooling-Optimization/05_test_plan.md)

---

## 1. 设计提案（design_proposal）

**目标**：说明"怎么做"，给出技术方案与架构设计。

应包含：
- **技术方案**：核心思路、架构设计、关键算法
- **改动点**：涉及的模块、文件、代码路径（与需求分析的改动面对应）
- **接口设计**：新增/修改的接口、配置项、环境变量
- **兼容性**：与现有功能的兼容性、降级方案
- **性能预估**：性能影响、优化点

## 2. 实施计划（implementation_plan）

**目标**：说明"怎么落地"，给出分阶段的实施步骤。

应包含：
- **阶段划分**：P0/P1/P2 等阶段，每阶段的目标与产出
- **任务清单**：每阶段的具体任务、负责人、预估工作量
- **依赖关系**：任务间的依赖、阻塞点
- **风险与应对**：每阶段的风险点与应对措施
- **验收标准**：每阶段的验收条件

## 3. 开发计划（dev_plan）

**目标**：说明"怎么开发"，给出代码开发的具体安排。

应包含：
- **代码结构**：新增/修改的文件、目录结构
- **开发顺序**：先做什么、后做什么、为什么
- **单元测试**：每个模块的 UT 设计
- **代码审查**：PR 拆分、审查要点
- **集成测试**：模块间的集成测试设计

## 4. 测试计划（test_plan）

**目标**：说明"怎么测试"，给出完整的测试方案。

应包含：
- **测试范围**：功能测试、性能测试、兼容性测试、压力测试
- **测试矩阵**：backend × layerwise × 模型 等维度的组合
- **测试环境**：环境搭建（指向 03_env_setup）、存储后端拉起（指向 04_backend_startup）
- **测试用例**：具体测试场景、输入、预期输出
- **判定标准**：PASS / FAIL / EXPECTED_FAIL 的判定条件（指向 07_pass_criteria）
- **测试脚本**：自动化测试脚本（放 assets/<专项>/）

## 5. 样板参考

- [02_design_proposal.md](../../Proposals/ascend_project/Layerwise-Pooling-Optimization/02_design_proposal.md) — 设计提案实例
- [03_implementation_plan.md](../../Proposals/ascend_project/Layerwise-Pooling-Optimization/03_implementation_plan.md) — 实施计划实例
- [04_dev_plan.md](../../Proposals/ascend_project/Layerwise-Pooling-Optimization/04_dev_plan.md) — 开发计划实例
- [05_test_plan.md](../../Proposals/ascend_project/Layerwise-Pooling-Optimization/05_test_plan.md) — 测试计划实例