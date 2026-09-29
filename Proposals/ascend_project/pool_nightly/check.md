## P0：直接影响看护有效性或首跑成功率
### 1. S3 的 benchmark 顺序会同时污染“读路径精度”和“开销水位”两个信号
单机 runner 已核实： 同一 case 内所有 benchmark 条目按 yaml 顺序对同一服务串行执行 （`test_single_node.py:565-578` ，`benchmark_keys` 保持 dict 序）。§3.4 的顺序是`acc-gpqa → acc-prefix → perf` ，导致：

- acc-prefix 首跑时池是冷的 ：第一波（batch 32）只能写、不能读，读路径覆盖打折；
- perf 跑在暖池上 ：prefix90 负载的读收益混进“开销水位”，开销信号不纯。
改法 ：顺序改为`acc-gpqa → perf → acc-prefix` 。perf 在冷池上跑=纯写池开销水位（正好对应矩阵里“开销水位”的语义）；perf 顺带把 prefix90 的共享前缀全部写入池，acc-prefix 全量请求池命中=读路径判卷语义干净。 零成本，只调 yaml 里的条目顺序。
> **[已采纳+语义校准 2026-09-28]** 顺序已按 proposal §3.4 落稿；但"perf=纯写池开销水位"经后续代码核实校准——prefix90 负载下只有首波并发请求是冷池，第二波起 gva 路径即查询命中、池读收益混入，聚合数字为"池化净效应"而非纯开销（见 proposal §3.4 perf 语义校准、§4.2① 粒度门路径分化）。acc-prefix 判卷语义不变（顺序设计的核心收益）。

### 2. R9 的依赖链已从代码侧证实，建议把 Phase-2 预验证写成有明确判据的 gate
两条 lookup 路径都直接消费`request.block_hashes` ：

- 非 layerwise 路径：`pool_scheduler.py:611` （`self.client.lookup(token_len, request.block_hashes, ...)` ）
- layerwise 路径：`pool_scheduler.py:312` （`block_hashes_to_query = request.block_hashes[...]` ）
而`--no-enable-prefix-caching` 下 block_hashes 是否仍生成是上游版本行为，本机 vllm 源码不可核（workspace 里 vllm 目录为空）。 建议在 §5 Phase 2 的"block_hashes 预验证"里写死判据 ，避免到时候再设计实验：
 预验证=任一 a3 测试环境起 1 卡池化服务（`--no-enable-prefix-caching` + memcache），发 2 条共享前缀请求；判据=第 2 条请求的`hit_check` 日志（`pool_scheduler.py:482` 有现成的`hit_check: req=... no participating groups` 签名可 grep）显示池命中 ≥ 1 block。命中不达即触发 R9 回退，S3 不提 PR。
> **[本判据作废 2026-09-28]** `hit_check` 签名属 block_key_layerwise 路径（S3 不走）且为 debug 级。最终判据以 proposal §5 Phase 2 第 2 步为准：第 2 条请求 INFO 级日志 `kvpool hit tokens: >0`（恒 0 且无报错 = 哑失效 = gate 不过）。
顺带一提：我核查过`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` ——它是池感知的“逐出后保留宽限期”（`pool_scheduler.py:109` 、`patch_kv_cache_coordinator.py:138` ）， 不能 用来强制 HBM miss，所以 R9 现有回退菜单（保 prefix + 超量负载制造淘汰）维持原样是对的。

### 3. S2#1 的 R2 回退阶梯缺一档：fp8 KV × mooncake layerwise 是先例之外的新颖组合
方案说 R2 的依据是“框架与 template.md 文档支持已确认”，但 Qwen 先例（`Qwen3.5-397B-Memcache-perf.yaml` ）是 BF16 KV ；a5 case #1 的服务形态是`--kv-cache-dtype fp8` +`--attention_config.indexer_kv_dtype fp8` （共享`_server_cmd` anchor，`DeepSeek_V4_Flash_A5.yaml:33-36` ）。 mooncake 池 + layerwise + fp8 KV 页 叠加在“无实跑先例”上再叠一层。

改法 ：R2 阶梯补第③档——case #1 内联完整`server_cmd` （与 anchor 解耦，`--kv-cache-dtype` 改 auto）。这与 R6 已有的“池化差异收敛在 case #1 自有字段”原则一致，只是把解耦提前到设计里而非回退时。

## P1：低成本对齐，提高首跑一次通过率
### 4. [已撤回 2026-09-28] S1 的 benchmark 顺序也应与 weekly 对齐（perf 在前）
多机 runner 同样按 yaml 顺序跑 benchmark（`test_external_dp.py:295-298` ）。weekly 先例是`perf_TPOT50 → acc_aime2025` （先例文件 line 182-203），而 §3.1 是“追加”在 acc 之后——15569.48 这个基线是在 冷服务+冷池 上量的，顺序不一致会引入暖态偏差。把 perf 放前面，同形对标口径才严格成立。
> **撤回原因**：这是唯一真正会动线上条目位置的改动（把线上 acc_aime2025 从 #1 挪到 #2，使其在 4096 条高压洪峰之后跑）——洪峰若造成内存碎片/资源劣化，红的会是 AIME 精度考题，精度考题替性能问题背锅，属"把线上搞坏"类型。维持"追加"序：AIME 测量语境与线上完全一致（#1、冷服务）；perf 暖跑偏差方向偏快、不误报红（`_task_passed`：吞吐 ≥ 0.95×基线），真有大偏差由 R4 首跑校准兜底。**降级为：仅当首跑 perf 偏差超 0.95 阈值、且排除硬件差异后的备选调整**。

### 5. Q1 的理由可以补强：prefix-caching 是该用例的注册身份
`test_name: "DeepSeek-V4-Flash-w8a8-PD-prefix"` ——用例名里就有 "prefix"，prefill 端 prefix-caching 保持 ON 不只是“保守不改”，而是 该 nightly 入口的语义组成部分 （decode 端本来就有`--no-enable-prefix-caching` ，差异只存在于 prefill 端）。Q1 选 A 的理由应写上这一条，防止后人“顺手对齐先例”改变用例含义。

### 6. R1 增加一条具体的“第一怀疑点”
`MooncakeConnectorScheduler` 从`vllm_config.kv_transfer_config.engine_id` 读 engine_id（`mooncake_hybrid_connector.py:1212` ）。MultiConnector 里子条目的 engine_id 是否被上游逐条目重建转发是实现细节（本机无上游源码可核），而 weekly 先例的子条目 没有 engine_id 也照常工作。建议 R1 补一句： 首跑若 PD 握手/路由异常，第一怀疑点=子条目 engine_id，回退动作=删掉它、完全对齐 weekly 。

### 7. 文档勘误与提交前自检
- 附录 B：`ascend_multi_connector.py` 实际路径是`vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py` （不在`kv_pool/ascend_store/` 下）。
- §5 增加零号步骤： 提交前本地静态校验 ——`yaml.safe_load` + 对每条 kv-transfer-config`json.loads` 过一遍（runner 无 schema 预检，`config.py` 是运行时才解析；S1 的两段单行大 JSON、S4 的新 anchor、S2#1 的解耦展开都值得这一步，不占 CI 资源）。
## P2：记录级改进，评审时拍板
### 8. 显式记录 kv_load_failure_policy 的地雷与正确语义
`platform.py:1065-1067` ：默认策略是`fail` ，且 hybrid 模型配`recompute` 会被断言直接拒绝 （"Hybrid models do not support recompute mode..."）。这意味着：

- template.md 的 MultiConnector 示例带`kv_load_failure_policy: "recompute"` ， 谁照抄谁起不来服务 （DSV4-Flash 是 hybrid）；
- 方案各载体省略该字段=默认`fail` ，对 nightly 看护恰好是正确语义： 池读失败响亮失败，不静默 recompute 掩盖读路径 bug ——这对 S3 读路径判卷尤其重要（若 recompute 可用，加载失败会被静默吞掉，acc-prefix 形同虚设）。
建议在 §4.1 或注意事项里写一行：“勿抄 template.md 的`recompute` ；默认`fail` 即看护所需语义”。

### 9. 端口纪律对齐（引用项目自己的 common_constraints）
- memcache 5000/6000 符合端口规划表 ✓；S2#1 mooncake master 50088 ✓（50051 被宿主 DTMA 占用的教训已有纪律）；
- mooncake metrics 用 50089 是 template.md/nightly 语境，而`common_constraints.md` 端口规划表的纪律值是 9008（共享验证服务器语境，且注明“换机勿照抄”）——建议方案加一句说明二者关系，防止后人拿纪律表“纠正” nightly；
- §5 检查清单加“首跑前确认 5000/6000/50088/50089 空闲”（共享服务器纪律本就要求端口预检）；
- 我已核实 CI 注册：S1 走 multi_node runner（a3-800i-16×2）、S4 走 single_node runner（a3-800i-8），两者同用 memcache 5000/6000 但 不同 runner 池 ，并发冲突风险低——记一句即可。
### 10. R7 可补一个更便宜的救济备选（仅记录，评审拍板）
a5 的 case #6（Maximum_Context，1M）和 case #5 目前完全没有 benchmarks （yaml 佐证：两者均无`benchmarks:` 键）——是纯“起服务→停服务”冒烟 case。因此 R7 的救济菜单可加 备选 C：给 case #6 挂 GPQA 精度条目 ，零新服务、零新 case、恢复 a5 非池化 BF16 精度看护。代价：a5 夜间时长 +GPQA 全程（约 1-3h）；且 #6 形态是 MTP4+block_verify≠#1 的 MTP1，是“新建非池化参考”而非 87.4 的趋势延续。是否值得，交给评审。

### 11. R10 的归因流程写进 runbook
已核实 S4 的接受率测量时点：`capture_baseline` （warmup）→ benchmarks → 接受率断言（`test_single_node.py:295-324` ），即接受率是 在含池化的 perf 流量上测的 。两 case 同挂后 nightly 内无非池化对照，建议 R10 明确写上：“接受率掉线时的归因动作=本地跑一次非池化对照（移除 kv_pool + kv-transfer-config 即恢复纯哨兵，可逆门）”，把诊断路径固化。

## 另外核实过、方案无需改的点
项 结论 占位判定语义 `_task_passed` ：accuracy`abs(x-0)≤100` 恒过、perf`吞吐≥0.97×1` 有正吞吐即过，不误报 ✓ S4 双 case memcache 端口 per-case`with` 生命周期串行起停（`test_single_node.py:598,613-621` ），无冲突 ✓ 多机 env 注入 template.md 证实`MOONCAKE_CONFIG_PATH/MMC_LOCAL_CONFIG_PATH` 按节点生成、注入所有 rank，node-0 起池服务 ✓ MultiConnector HMA 断言 `ascend_multi_connector.py:28` 要求全部子连接器 supports HMA；MooncakeConnector 与 AscendStoreConnector 均为 SupportsHMA，weekly 先例实跑佐证 ✓ 池 lookup 粒度门 `cache_transfer_granularity` =组块尺寸 LCM（`metadata.py:324-332` ），量级=块尺寸级，S4 的 3500-token 负载不会被拦截 ✓ S2#1 anchor 解耦必要性 `*benchmarks_per` 确被 04/05/06 引用、`*benchmarks_acc` 仅 #1 引用，方案判断正确 ✓

一句话总结 ：方案骨架与可行性声明全部经得起代码核对；最值得改的是 S3 benchmark 顺序 （P0-1，改顺序即可让读路径判卷和开销水位两个语义都变纯）、 R2 补 fp8 回退档 （P0-3）和 R9 预验证判据固化 （P0-2）；其余是对齐与记录级完善。