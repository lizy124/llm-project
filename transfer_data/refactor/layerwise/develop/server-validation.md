# PR #15307 服务器验证清单

> 对应设计文档 §6 验收标准 2/3/4(服务器侧实测项);UT/ruff/mypy 已本地验证
> (314 passed / 2 pre-existing failures,详见 `PR-B record.md`)。

# PR #15367(refactor_layerwise_part1)服务器验证清单

> 从 upstream/main 新开的分批实施第一批:仅收敛核心逻辑(GATE 入 memcache 协议模块/IFACE/KEY/gate 下沉),
> 5 个行为保持提交(见 PR Commits 列表),UT/CI 由 GitHub checks 覆盖,服务器侧
> 实测项与下方 #15307 清单的 1/2/3 完全相同,逐项执行即可。
> 代码:`refactor_layerwise_part1` @ 1ff8dc141(5 commits,基于 e8f47fc11,
> 即 2026-09-01 的 upstream/main;本 PR 删除 #15291 热修的 connector 侧
> 派生,属取代关系)。

差异说明(相对 #15307,影响验证关注点):

- 验证项 1(MultiConnector PD)关注点不同:#15367 中 connector 不再持有
  `use_gva_layerwise`,gate 在 `KVPoolWorker.set_external_slot_release_waiter`;
  验收标准不变(初始化无 AttributeError、成功率 100%)
- 验证项 2/3 关注点不变:key 构造与派生逻辑为纯平移,行为应与 main 一致
- #15291 已合入 main(2026-08-31):若在含 #15291 的 main 上验证,PD 冒烟的
  对照基线应取 #15291 后的 main,而非更早版本


## 环境要求

- 服务器:165(执行计划指定)
- 容器:cxy 镜像(refactor_812 与新基线不兼容,禁止使用)
- 代码(按所验 PR 取用,勿混用):
  - #15367:`refactor_layerwise_part1` @ 1ff8dc141(5 commits,基于 e8f47fc11)
  - #15307(已搁置,仅留档):`refactor_layerwise_B` @ b3a141331(10 commits,含 PR-A 5 个)
- 日志:观察 debug 级日志需将 vllm logger 级别调至 DEBUG

## 验证项(按优先级)

### 1. MultiConnector PD 分离冒烟 —— 优先级最高

- 目的:#14465 回归修复生效点(connector 侧 `use_gva_layerwise` 派生恢复);
  设计验收标准 4 明确要求实测而非仅 UT
- 配置:P 4×TP + D consumer,proxy 模式,GSM8K prefix-cache
- 验收标准:请求成功率 100%;初始化链路无 AttributeError
- 修复前失败特征:`AscendMultiConnector.__init__` →
  `_configure_layerwise_reuse_completion` → `set_external_slot_release_waiter`
  即 AttributeError(main 活回归)

### 2. memcache layerwise 冒烟

- 目的:UT5(load 路径非零 gva 探针)的真环境对应面,排除静默失效
- 配置:TP=4,`backend=memcache`,`use_layerwise=true`,长前缀 load 场景
- 验收标准:
  - debug 日志 `load_gvas: ... valid_gvas=N` 中 N > 0(命中块 gva 非零)
  - debug 日志 `hit_check: ... hit_tokens=N` 中 N > 0
  - 无静默失效:hit_check 正常但 valid_gvas 恒 0 = load 路径失效
- 附带观察(平移敏感点,任一异常即停):save 失败传导、h2d stagger、
  layer 事件时序(layer_save_finished_events / layer_load_finished_events)

### 3. mooncake 非 layerwise 冒烟

- 目的:通用路径零回归(设计验收标准 2)
- 配置:默认 `backend=mooncake`,非 layerwise
- 验收标准:与 main 基线行为一致

## 辅助检查

- #15307 CI 28 checks 全绿跟踪(checks 页)
- 上述 1/2/3 通过后,将结果(日期/配置/成功率/关键日志摘录)贴入 PR 评论

## 结果记录

| 项 | 日期 | 服务器 | 结果 | 证据(日志/截图) |
|---|---|---|---|---|
| 1. MultiConnector PD | 2026-09-01 | 165 / refactor_165 | PASS | 5/5 请求成功(3 合成共享前缀 4177 tokens + 2 GSM8K-lite);P/D 日志无 AttributeError,AscendMultiConnector 初始化路径确认走过;P 侧 load_gvas 28 行 + MetaService query_successes=1520;D 侧 layerwise recv 52 行。证据 `../test/evidence/s1_pd_multiconn/`,详报 `../test/e2e-report-20260901.md` |
| 2. memcache layerwise | 2026-09-01 | 165 / refactor_165 | PASS | `hit_check hit_tokens=3456>0`;`load_gvas valid_gvas=27>0 lease_fail=0`;三维证据链 alloc_successes=28 stored_keys=28 query_successes=400;external hits=10240。证据 `../test/evidence/s2_memcache_layerwise/` |
| 3. mooncake 非 layerwise | 2026-09-01 | 165 / refactor_165 | PASS | 请求 100%;无 load_gvas/hit_check(layerwise 标记缺席);master 三维 allocated_bytes=939.5MB key_count=112 active_clients=4;external hits=10240。证据 `../test/evidence/s3_mooncake_non_layerwise/` |

验证代码:`refactor_layerwise_part1` @ `2a239d18a`(7 commits,基于 e8f47fc11
之后 main)+ vllm @ `ba07e4a48`。拓扑:场景 1 为 DSV2-Lite P TP=4 + D TP=4 +
layerwise proxy(单机 8 卡);场景 2 DSV2-Lite TP=4;场景 3 Qwen3-32B TP=4。
过程问题均为测试基建修正(子 connector 禁写 engine_id、HBM 残留清理、
pipefail 下 grep 静默退出、指标名按 memcache 1.2.0 实态修正),被测代码零改动。

## 复测轮(rebase 后,2026-09-01 18:31–18:57)

检视返工(4 commits)+ mypy 修复 + rebase 到 upstream/main `72a988f9d`(带入
#15364 KV Pool 改动)后,同清单三项重测。验证代码:`refactor_layerwise_part1`
@ `63be9e03b`(git bundle 部署,git rev-parse 证实;pip 版本串滞后系 editable
构建缓存)+ vllm @ `ba07e4a48`(未动)。拓扑/模型/参数与首轮一致。

| 项 | 日期 | 服务器 | 结果 | 证据 |
|---|---|---|---|---|
| 1. MultiConnector PD | 2026-09-01 | 165 / refactor_165 | PASS | 5/5(3 合成 4177 tokens + 2 GSM8K-lite);P/D 无 AttributeError;AscendMultiConnector ×5;P 侧 load_gvas 16 行 + MetaService query_successes=770;D 侧 recv 34 行;27 层 LayerMetadata 完整。证据 `record_final/s1_pd_multiconn_20260901_rebase/`,详报 `../test/e2e-report-20260901-rebase.md` |
| 2. memcache layerwise | 2026-09-01 | 165 / refactor_165 | PASS | `hit_tokens=3328>0`;`valid_gvas=26>0 lease_fail=0`;三维 alloc=28 stored=28 query=130;external hits=3328。证据 `record_final/s2_memcache_layerwise_20260901_rebase/` |
| 3. mooncake 非 layerwise | 2026-09-01 | 165 / refactor_165 | PASS | 请求 100%;无 load_gvas/hit_check;master 三维 939.5MB / 112 keys / 4 clients;external hits=3328。证据 `record_final/s3_mooncake_non_layerwise_20260901_rebase/` |

两轮(返工前 `2a239d18a` / 返工+rebase 后 `63be9e03b`)全部 PASS,行为
一致互证。数值差异(hit_tokens 3456→3328、load_gvas 行数 28→16)为滑动
窗口命中块数/请求分布浮动,判据(>0)不变。过程问题 1 项:服务器到 GitHub
间歇断连,改 git bundle 部署(测试基建,被测代码零改动)。
