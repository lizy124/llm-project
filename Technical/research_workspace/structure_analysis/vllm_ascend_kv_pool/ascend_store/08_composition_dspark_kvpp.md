# 08. MultiConnector、DSpark 与 KVPP 的组合边界

基线见 [00](../00_source_baseline.md)。本篇补上单个 Store 文件看不到的跨组件资源协议。

## 1. MultiConnector 不只是循环调用子 connector

`ascend_multi_connector.py::AscendMultiConnector` 继承上游组合器，额外处理：
- HMA：启用时所有子 connector 都必须支持。
- allocation：选中的加载 connector 接收真实 blocks；声明需要完整 block 账本的其他子 connector 也接收。
- PreemptOffload：若有对应被抢占 request 状态，优先尝试恢复其可能未 hash 的 partial block。
- slot reuse：producer 释放 gate 与 Store 覆盖顺序。
- Mamba copy：所有相关子 load 完成后再执行。
- DSpark：只选择能够同时恢复 target/draft 的 provider。

配置顺序不能替代这些语义规则。尤其不能让普通 prefix connector 抢先接管一个包含未 hash 尾块的抢占恢复。

## 2. 外部 slot release 协议

某个 P/D producer 可能仍在读取共享 KV slot，即使 Store 自己的上一层保存已完成，也不能覆盖该 slot。

组合器识别具备 `supports_layerwise_buffer_reuse`、`wait_for_layer_reuse()` 的 producer，作为 release providers。随后把 composite waiter 安装给 Store 的 `set_external_slot_release_waiter()`。

```text
producer cache-write/save hook 先关闭相应 gate
  -> sibling Store 提交可能触发复用的任务
  -> recv thread 在覆盖 slot 前调用 composite waiter
  -> 所有 providers 确认旧读者退出
  -> 才可写入新 layer KV
```

如果 Store 接受 waiter，由它负责 layer-entry reuse 等待，组合器省去重复 provider 等待；没有 sink 时保留 provider 优先等待的保护。

shutdown 也先处理 providers，再 join Store 接收线程，否则后者可能一直等一个已经无法打开的 gate。这是多组件依赖，不能只在 Store 内部看 layer event。

## 3. Mamba copy 为什么必须晚于所有 load

align copy 会读取已经恢复的 conv/SSM 状态。若在某个 child 尚未加载完就复制，会读到旧状态或半加载数据。

- layerwise：`prepare_mamba_state_copy()` 接管 copy；组合器逐层等完 children 后执行，最后 `finish_mamba_state_copy()`。
- V1 runner 使用 copy buffers。
- V2 runner 可传 Mamba model state；`worker/v2/kv_connector.py::MambaCopyKVConnector` 还负责同步 bulk load 后、forward 前的 copy 顺序。
- no-forward 路径不运行模型状态预处理，不能强行套相同 copy 流程。

## 4. DSpark：target 命中不够

在 SFA DSpark P/D producer 中，跳过 target prefix 计算还意味着 draft/auxiliary context 也必须被初始化。仅 target store 命中会跳过 draft 所需材料。

`AscendMultiConnector`：
1. 识别需要完整 DSpark prompt 的场景。
2. 尝试配置能够联合恢复的 Store provider。
3. 未找到 provider 时返回零外部贡献，保留完整 prompt 计算。
4. 配置成功后只使用该 provider 的 prefix 结果。
5. 多个 provider 同时声称支持则拒绝，避免两份 target/draft 状态来源。

Store 的 opt-in 条件是 **layerwise + GVA + kv_producer**，再由 `DSparkPrefixKeys.from_config()` 检查 speculative method。其他 serving/speculative 路径不会自动换成 DSpark companion keys。

## 5. Draft companion 的身份与布局

`DSparkPrefixKeys` 的身份 hash 包含 target/draft 模型配置与 revision、dtype/量化、draft width、auxiliary layer ids、TP/PP、block size、prefix match unit 等。

```text
dspark-prefix-v1@IDENTITY@pp<final-stage>@tp<RANK>@BLOCK_HASH
```

draft keys 覆盖所有需要的 TP replicas，位于协议指定的最终 PP stage。Scheduler 的 target group key 检查追加这些 companion keys；draft-only group 单独使用 draft 身份。

`DSparkPrefixCache.register()` 校验：
- 每个 draft cache 都有 group mapping。
- 逻辑 block size 一致，kernel pages 能完整覆盖逻辑页。
- 内层 contiguous，scaled pages 相邻，stride 不重叠。
- backend 支持 atomic write publication。
- 每对象的 `page_bytes` 精确匹配注册的所有 companion pages。

没有这些条件，不能仅凭同一个 token hash 复用 draft KV。

## 6. Draft save/restore 的事务边界

### 6.1 Save

```text
完整 hashed prompt blocks
  -> readable probe，已有对象不覆盖
  -> alloc 缺失对象
  -> 只 copy 本 writer 实际拥有的有效 GVA
  -> batch_write_finish 发布
  -> remove lease
```

partial tail 不作为完整 prefix 保存。allocation 冲突可能是另一 DP writer 正在写，不能抢写或把别人的对象标失败。SSD sentinel 不能当普通地址 copy。

### 6.2 Restore

```text
验证 scheduler LoadSpec 与完整 block offset
  -> for_load 查询/可能 rewarm
  -> add lease
  -> 在 lease 下重新取 GVA 与 page size
  -> 按当前各 group block table 写入
  -> remove lease
```

rewarm 要在 lease 之前完成；lease 后重新取地址，防止使用迁移前的旧地址。失败只清理本次实际获得的 lease，不能误释放其他 owner。

Store 同时保留 prompt 最后块重算区，保护 target/draft generation point 所需的新状态。相关行为测试见 09。

## 7. KVPP：层 owner 与 target scratch

相关文件：
- `worker/kvpp_cache.py::allocate_kvpp_cache`。
- `core/kv_cache_placement.py` 的 allocation/layout/owner plan。
- `worker/v2/kvpp.py`。
- `KVPoolWorker` 的 `use_kvpp`、owned-layer filter、key head/rank 推导。

当前 allocation 建立 contiguous layer bundles，为非本 rank owner 的 target 层使用两个共享 scratch buffers；owned layers 使用自己的持久 buffer。Store 注册/传输过滤到本 rank 实际拥有的层，不能将临时 scratch 当作全模型持久副本。

Worker 将 KVPP owner rank 纳入 key/rank 推导；普通发送线程在 KVPP 下不再把 PCP owner 当成 KV block replicas。**层的分布式所有权**与 **context parallel 的 token 切分**不是同一概念。

这也与 MemCache layerwise 本地 slot reuse 不同：两者都出现共享 buffer，但 owner、通信和恢复协议不同。

## 8. 阅读组合路径的检查点

| 问题 | 必须确认 |
|---|---|
| 谁能接管 prefix | target/draft 是否共同恢复；是否有抢占 partial 状态 |
| 谁允许写共享 slot | Store save 完成 + 外部 producer release |
| 谁执行 Mamba copy | 单 Store 或组合器接管，必须晚于所有相关 load |
| 谁拥有真实 KV | KVPP persistent owner、target scratch、普通 tensor 不可混同 |
| 谁负责停机解锁 | provider→transfer threads→backend 的清理顺序 |

对应测试和排障见 [09](09_observability_testing.md)。
