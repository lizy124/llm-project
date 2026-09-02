# run_dir — 池化服务验证运行手册（按场景组织，通用层）

> 定位：**可照抄执行的运行手册**。verify_guide.md 讲"为什么这样判"（方法论），
> 本目录讲"怎么拉起来、怎么发请求"（操作步骤 + 命令）。**与具体 PR 解耦**：
> 环境变量、启动命令、请求方式、判定命令是通用资产；某次验证的 commit/结果/归档
> 属于实例层，放 `map_XX/<task>/record_final/`，不进本目录。
>
> 组织规则：**一个场景一篇**（不按日期）；新增场景类型时新建文档并更新索引；
> 场景手册内的实测数字仅作"通过形态示例"参考，判据以命令本身为准。

## 索引

| 文档 | 场景 | 验证目标 |
|------|------|---------|
| [common-prerequisites.md](common-prerequisites.md) | 公共前置 | hugepages / mmc conf / MetaService / mooncake master / 场景间清理 |
| [memcache-layerwise.md](memcache-layerwise.md) | 单实例 + memcache + layerwise | layerwise 路径非零激活（hit/存入正计数） |
| [mooncake-non-layerwise.md](mooncake-non-layerwise.md) | 单实例 + mooncake 非 layerwise | 通用路径零回归（layerwise 标记缺席），兼作 layerwise 场景的阴性对照 |
| [pd-multiconn.md](pd-multiconn.md) | PD 分离（MultiConnector 双 connector + proxy） | PD 分诊初始化链路 + KV 传输链路完整 |

## 通用约定

- 路径约定：`$BASE` = 服务器工作区（如 `/home/lizhongyang/map_XX`）、容器名 `CTR`（如
  `refactor_XX`）；启动/测试脚本在**容器内**执行，多场景编排（前置 → 逐场景 start/test/stop
  → clean）在**宿主机** nohup 执行，本地 ssh tail 日志轮询
- 模型/权重路径按服务器实际（`$MODEL_PATH` 可覆盖）；端口规划见 verify_guide.md §2.1/§3.1
- 场景间必须 stop + clean（vllm 崩溃残留 worker 占 HBM，下一场景起不来）
- 判定设计遵循 verify_guide.md §6.6 三问（正计数 / 跨进程证人 / 阴性对照）
- 每次实际验证的版本快照（env.txt：git rev-parse + pip）随场景归档，不写死在手册里
