# PR #15307 服务器验证清单

> 对应设计文档 §6 验收标准 2/3/4(服务器侧实测项);UT/ruff/mypy 已本地验证
> (314 passed / 2 pre-existing failures,详见 `PR-B record.md`)。

# PR #15367(refactor_layerwise_part1)服务器验证清单

> 从 upstream/main 新开的分批实施第一批:仅收敛核心逻辑(CAP/IFACE/KEY/gate 下沉),
> 5 个行为保持提交(见 PR Commits 列表),UT/CI 由 GitHub checks 覆盖,服务器侧
> 实测项与下方 #15307 清单的 1/2/3 完全相同,逐项执行即可。
> 代码:`refactor_layerwise_part1` @ 9f5c199ea(5 commits,基于 40f9834ee,
> 即 2026-08-31 的 upstream/main;本 PR 删除 #15291 热修的 connector 侧
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
  - #15367:`refactor_layerwise_part1` @ 9f5c199ea(5 commits,基于 40f9834ee)
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
| 1. MultiConnector PD | | | | |
| 2. memcache layerwise | | | | |
| 3. mooncake 非 layerwise | | | | |
