# 01 — 需求分析（怎么分析一个新 PR/专项）

> 新项目（如一个池化新 PR）到来时，第一步是需求分析。本章给出"怎么拆需求、怎么提问、怎么识别风险、怎么界定改动面"的方法论。
> **样板**：[Layerwise-Pooling-Optimization/01_requirements_analysis.md](../../Proposals/archive/Layerwise-Pooling-Optimization/01_requirements_analysis.md)

---

## 1. 需求分析的目标

把"一个 PR"变成"一个可执行的专项"：
- 这个 PR 要解决什么问题？（用户痛点 / 上游 issue / 性能瓶颈）
- 改动面有多大？（涉及哪些模块、哪些代码路径、哪些配置）
- 风险点在哪？（与现有功能的兼容性、性能回退、环境依赖）
- 验证标准是什么？（怎么算 PASS、怎么算 FAIL、怎么算 EXPECTED_FAIL）

## 2. 分析步骤

| 步骤 | 动作 | 产出 |
|------|------|------|
| 1 | 读 PR 描述 / issue / 上游文档 | 一句话需求概述 |
| 2 | 读 PR diff / 代码改动 | 改动面清单（模块、文件、代码路径） |
| 3 | 识别依赖与约束 | 环境依赖（版本配对、硬件、网络）、上游硬限 |
| 4 | 识别风险点 | 兼容性、性能、稳定性风险 |
| 5 | 定义验证标准 | PASS / FAIL / EXPECTED_FAIL 的判定条件 |

## 3. 提问清单（分析时必答）

- 这个 PR 的核心功能是什么？（一句话）
- 它改动了哪些代码路径？（哪些模块、哪些函数）
- 它依赖什么环境？（vllm/vllm-ascend 版本、硬件型号、网络拓扑）
- 它与现有功能有什么交互？（兼容性、冲突点）
- 它的上游硬限是什么？（已知的不可绕过的限制）
- 怎么验证它是否工作？（测试方法、判定标准）

## 4. 产出格式

需求分析文档应包含：
- **需求概述**：一句话说明这个 PR 要解决什么问题
- **改动面**：涉及的模块、文件、代码路径
- **依赖与约束**：环境依赖、版本配对、硬件要求
- **风险点**：兼容性、性能、稳定性风险
- **验证标准**：PASS / FAIL / EXPECTED_FAIL 的判定条件
- **下一步**：指向 02_design（方案设计）

## 5. 样板参考

[Layerwise-Pooling-Optimization/01_requirements_analysis.md](../../Proposals/archive/Layerwise-Pooling-Optimization/01_requirements_analysis.md) — KV 池化 Layerwise 传输加速专项的需求分析实例，包含：
- 需求背景与目标
- 改动面（涉及的模块与代码路径）
- 环境依赖与约束
- 风险点识别
- 验证标准定义