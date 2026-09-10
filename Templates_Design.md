# Templates 设计方案 — 项目执行手册（可复用模板区）

> 状态：设计方案，待用户审阅后再动手搭建。
> 本方案的决策前提（2026-09-11）：
> - 骨架放在**顶层 `Templates/`**，与 Technical/Proposals/assets/utils 并列；
> - **`Proposals/` 不动**（保持"具体专项的方案/规划区"）；
> - 先出设计方案，再搭建。

---

## 1. 定位与职责

| 区 | 职责 | 内容性质 | 生命周期 |
|---|---|---|---|
| `Technical/` | 长期知识、研究、技术记录、证据 | 已沉淀的技术结论 | 长期稳定 |
| `Proposals/` | **具体专项**的需求分析、设计提案、实施/测试计划、PR 描述与走读 | 每个 PR/专项的实例 | 项目完结后可归档 |
| `Templates/` | **跨专项复用**的项目执行手册（方法论模板） | 换 PR/换服务器依然成立的方法 | 长期演进、回写沉淀 |
| `assets/` | 非 md 资产（脚本/数据/图片/证据） | 被文档引用的原料 | 长期 |
| `utils/` | 通用环境工具备忘 | 与项目无关 | 独立 |

**Templates 的定位**：新项目（如一个池化新 PR）到来时，给出"从头走到尾"的可复用方法论；每个专项按这套方法论执行，产出自己的 Proposals 实例（需求分析/设计/计划）。

## 2. 项目工作流（骨架的纵向骨架）

新 PR 到来时的完整流程：

```
新 PR / 新专项
   │
   ▼
[01] 需求分析        ← 模板：Templates/01_requirements/
   │
   ▼
[02] 方案设计        ← 模板：Templates/02_design/
   │
   ▼
[03] 代码开发        ← （不在本手册，由用户/agent 自行完成）
   │
   ▼
[04] 测试验证        ← 模板：Templates/03_env_setup → 04_backend_startup
                       → 05_vllm_launch → 06_test_method → 07_pass_criteria
   │
   ▼
产出：Proposals/<专项>/（实例） + Technical/<专项>/（技术记录与证据） + assets/<专项>/（脚本/数据）
```

## 3. 骨架结构

```
Templates/
├── README.md                     # 手册总入口：工作流图、七章索引、维护规则（新 lesson 回写）
├── _common/
│   └── common_constraints.md     # IRON RULE、版本配对表、端口规划、共享服务器纪律
├── 01_requirements/
│   └── how_to_analyze.md         # 新 PR 怎么拆需求、提问、识别风险、界定改动面
├── 02_design/
│   └── how_to_design.md          # 设计提案、实施计划、测试计划怎么写（可附样板）
├── 03_env_setup/
│   └── env_setup_guide.md        # 镜像选择 → 建容器 → 配代理 → 查版本/卸载 → 装 vllm/vllm-ascend
├── 04_backend_startup/
│   ├── mooncake_startup.md       # mooncake master 拉起规范（端口/配置/探活/停止）
│   └── memcache_startup.md       # memcache MetaService 拉起规范（两份 conf/环境变量/探针）
├── 05_vllm_launch/
│   └── vllm_pool_launch.md       # 池化场景启动（关键参数/READY 等待/启动失败速查）
├── 06_test_method/
│   └── test_design.md            # prompt 门槛/验证矩阵/标准测试流程/对照实验/脚本纪律
└── 07_pass_criteria/
    └── pass_criteria.md          # 三维证据链/虚假通过防范/双轮夹逼/验收表模板/硬限
```

命名说明：
- 序号 `01–07` 表达工作流顺序；`_common` 前缀下划线保证排在最前。
- 每章单一入口 md（单一入口、可直接照做，延续现有 create_env/verify_guide 的文档形态）。

## 4. 内容来源映射（关键：骨架先搭，内容从现有文档抽取）

| 现有文档 | 进入 | 说明 |
|---|---|---|
| `Technical/transfer_data/playbook/create_env.md` | `Templates/03_env_setup/env_setup_guide.md` | 全文迁移，已在"通用规则层"形态，无需大改 |
| `Technical/transfer_data/playbook/verify_guide.md` §1 | `Templates/_common/`（概念速查） | KV Pool 底层概念，是各章的共享背景层 |
| 同上 §2 / §3 | `Templates/04_backend_startup/mooncake_startup.md` / `memcache_startup.md` | 拆分 |
| 同上 §4 | `Templates/05_vllm_launch/vllm_pool_launch.md` | 拆分 |
| 同上 §5 | `Templates/06_test_method/test_design.md` | 拆分 |
| 同上 §6–§8 | `Templates/07_pass_criteria/pass_criteria.md` | 拆分 |
| `Proposals/ascend_project/test_pool/01_memcache_kv_pool_setup.md` | `Templates/04_backend_startup/memcache_startup.md` | 与 verify_guide §3 合并（同一主题），服务器特定部分（135/165）留 `Technical/` 相应专项目录 |
| `Proposals/ascend_project/Layerwise-Pooling-Optimization/01_requirements_analysis.md` | `Templates/01_requirements/how_to_analyze.md` 样板 | 作为"怎么写需求分析"的实例 |
| `Proposals/ascend_project/Layerwise-Pooling-Optimization/02_design_proposal.md` / `03_implementation_plan.md` / `04_dev_plan.md` / `05_test_plan.md` | `Templates/02_design/how_to_design.md` 样板 | 作为"怎么写设计/计划"的实例 |
| 各处 IRON RULE、端口规划、版本配对表 | `Templates/_common/common_constraints.md` | 汇总公共约束 |

**拆分原则**：verify_guide 的 §2-8 按主题拆到 04-07 各章，每章保持"通用方法论 + 服务器特定信息指向"的形态（服务器特定端口/路径等放 `Technical/` 相应专项，不入模板）。

## 5. 维护规则（与 create_env/verify_guide 现有规则一致）

1. **新 lesson 必须回写**：任何服务器上实操得到的新经验/新坑/新判定标准/新硬限，完成后回写 `Templates/` 对应章节。
2. **服务器特定信息不放这里**：某台服务器的端口占用/权重路径/容器名等，放 `Technical/` 对应专项。
3. **只放通用方法论**：换个服务器、换个 PR 依然成立的规则才入库。
4. **修改同步检查**：`Templates/_common/common_constraints.md` 的公共约束（IRON RULE、端口）与其他区文档保持一致。

## 6. 边界与归档

- `Templates/` 是"怎么做"，不重复存放 `Technical/` 已有的"是什么/做过什么"（长期知识/技术记录）；模板中可链接到 `Technical/` 的沉淀，但不复制整篇。
- `Proposals/<专项>/` 是实例（该专项的需求/设计/计划），`Templates/` 是模板（怎么写需求/设计/计划）——两者是"用"与"被用"关系，不重复。
- 原 `Technical/transfer_data/playbook/` 下的 `create_env.md` 和 `verify_guide.md` 迁移到 `Templates/` 后，playbook/ 只留 `run_dir/`（或与 transfer_data 其他子目录一起整理）。

## 7. 搭建顺序（动手阶段执行）

1. 新建 `Templates/` 骨架目录 + 各章占位 md + 总入口 README.md。
2. 从 `Technical/transfer_data/playbook/` 抽取 `create_env.md` 和 `verify_guide.md`（git mv 保留历史），按 §4 映射拆分到各章。
3. 从 `Proposals/ascend_project/test_pool/` 抽取 memcache 拉起内容合并到 `Templates/04_backend_startup/memcache_startup.md`。
4. 从 `Proposals/ascend_project/Layerwise-Pooling-Optimization/` 抽取 01-05 作为样板到 `Templates/01_requirements/` 和 `02_design/`。
5. 汇总 `_common/common_constraints.md`。
6. 清理原 playbook/ 空目录（`Technical/transfer_data/playbook/` 下只剩 run_dir 或并入 transfer_data 整理）。
7. 更新顶层 `README.md` 的目录分组图，加入 `Templates/`。

## 8. 验证标准

- 所有迁移用 `git mv`，git 历史保留（`R` 标记）。
- 迁移后文件总数 = 迁移前（零丢失）。
- 每章 md 可读性：新 PR 到来时，agent 照 `Templates/` 七章即可独立完成环境搭建 + 池化拉起 + 测试设计 + 结果判定，无需回头翻 `Technical/` 的专项记录。